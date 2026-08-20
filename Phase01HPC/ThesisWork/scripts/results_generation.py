import os
import sys
import json
import random
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import schema
import pandas as pd
from model import LegalPromptModel
from strategy_functions import build_prompt

# Repo layout is now Phase01HPC/ThesisWork/scripts/ — same extra
# dirname() call as build_demo_n_shot.py and build_eval_pools.py
# to reach the true repo root.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
THESISWORK_DIR = os.path.dirname(SCRIPTS_DIR)
PHASE01HPC_DIR = os.path.dirname(THESISWORK_DIR)
REPO_ROOT = os.path.dirname(PHASE01HPC_DIR)
THESIS_SELECTION_PATH = os.path.join(REPO_ROOT, "preparations", "thesisSelection.csv")


def load_task_field_map():
    with open(os.path.join(config.HPC_ROOT, "data", "task_field_map.json")) as f:
        return json.load(f)


def load_question_templates():
    with open(os.path.join(config.HPC_ROOT, "data", "question_templates.json")) as f:
        return json.load(f)


def load_manifest() -> pd.DataFrame:
    if not os.path.exists(THESIS_SELECTION_PATH):
        raise FileNotFoundError(
            f"{THESIS_SELECTION_PATH} not found. Run thesis_test.py first."
        )
    df = pd.read_csv(THESIS_SELECTION_PATH)
    return df.rename(columns={"task_name": "task_id"})  # align with rest of pipeline


def run_task_id_key(strategy, rephrasing_id, run_id, instance_task_id):
    return f"{strategy}|{rephrasing_id}|{run_id}|{instance_task_id}"


def output_file_key(task_id, model_name):
    safe_model_name = model_name.replace("/", "_")
    return f"{task_id}__{safe_model_name}"


def load_existing_records(file_key: str) -> list:
    path = os.path.join(config.RAW_GEN_DIR, f"{file_key}_generations.jsonl")
    if not os.path.exists(path):
        return []

    raw_records = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                raw_records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  WARNING: skipping corrupted line {line_num} in {file_key} (likely a partial write from an earlier crash)")
                continue

    deduped = {}
    for r in raw_records:
        key = run_task_id_key(r["strategy"], r["rephrasing_id"], r["run_id"], r["task_id"])
        deduped[key] = r

    if len(deduped) < len(raw_records):
        print(f"  {file_key}: {len(raw_records) - len(deduped)} duplicate record(s) found — deduplicated.")

    with open(path, "w") as f:
        for r in deduped.values():
            f.write(json.dumps(r) + "\n")

    return list(deduped.values())


def append_record(file_key: str, record: dict):
    path = os.path.join(config.RAW_GEN_DIR, f"{file_key}_generations.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_demonstration_sets(manifest_df: pd.DataFrame) -> dict:
    """
    Loads the per-k demonstration files built by build_demo_n_shot.py
    into a nested structure: {task_id: {k: [Demonstration, ...]}}.
    A missing k for a given task (e.g. its train pool was too small)
    is simply absent from that task's inner dict — build_prompt()
    handles this by falling back to an empty list.
    """
    demonstration_sets = {}
    required_ks = sorted(set(config.DEMO_REQUIRED_STRATEGIES.values()))

    for task_id in manifest_df["task_id"]:
        demonstration_sets[task_id] = {}
        for k in required_ks:
            demo_path = os.path.join(config.DEMO_DIR, f"{task_id}_demos_k{k}.json")
            if not os.path.exists(demo_path):
                print(f"  WARNING: {demo_path} not found — {task_id} will have no k={k} demonstrations available.")
                continue
            with open(demo_path) as f:
                demo_raw = json.load(f)
            demonstration_sets[task_id][k] = [
                schema.Demonstration(context=d["context"], question=d["question"], label=d["label"])
                for d in demo_raw
            ]

    return demonstration_sets


def normalize_answer(s) -> str:
    """
    Normalizes an answer for comparison: lowercase, strips
    surrounding whitespace, strips a trailing period. This means
    "Yes." and "yes" are correctly treated as the same answer,
    rather than requiring exact character-for-character matches.
    """
    if s is None:
        return ""
    return str(s).strip().lower().rstrip(".")


def build_legal_task(row: dict, task_id: str, category: str, idx: int,
                      task_field_map: dict, question_templates: dict) -> schema.LegalTask:
    field_map = task_field_map[task_id]
    context = str(row.get(field_map["context"], ""))

    if field_map.get("question"):
        question = str(row.get(field_map["question"], ""))
    else:
        question = question_templates.get(task_id)

    return schema.LegalTask(
        task_id=f"{task_id}_{idx}",
        task_type=category,
        context=context,
        question=question,
        label_options=[],
        expected_output=str(row.get("answer", "")),
        jurisdiction="US General",
        source_dataset="LegalBench",
    )


def main():
    args = config.get_model_name_from_args()
    model_name = args.model
    sample_size = args.sample_size

    os.makedirs(config.RAW_GEN_DIR, exist_ok=True)
    print(f"=== Stage A run starting: model={model_name}, sample_size={sample_size or 'FULL'} ===")

    task_field_map = load_task_field_map()
    question_templates = load_question_templates()

    manifest_df = load_manifest()
    print(f"Loaded manifest from {THESIS_SELECTION_PATH}: {len(manifest_df)} tasks")

    eval_pools = {}
    for task_id in manifest_df["task_id"]:
        with open(os.path.join(config.EVAL_POOLS_DIR, f"{task_id}_eval.json")) as f:
            eval_pools[task_id] = json.load(f)

    demonstration_sets = load_demonstration_sets(manifest_df)

    legal_tasks = {}
    for _, row in manifest_df.iterrows():
        task_id, category = row["task_id"], row["category"]
        pool = eval_pools[task_id]
        if sample_size:
            pool = pool.copy()
            random.Random(config.CLUSTERING_RANDOM_STATE).shuffle(pool)
            pool = pool[:sample_size]

        instances = [
            build_legal_task(r, task_id, category, i, task_field_map, question_templates)
            for i, r in enumerate(pool)
        ]
        label_options = sorted(set(t.expected_output for t in instances))
        for t in instances:
            t.label_options = label_options
        legal_tasks[task_id] = instances
        print(f"{task_id}: {len(instances)} instances loaded (sample_size={sample_size or 'full'})")

    total_planned = sum(len(v) for v in legal_tasks.values()) * len(config.ALL_STRATEGIES) * config.N_REPHRASINGS * config.N_RUNS

    lpm = LegalPromptModel(model_name)
    lpm.load()

    task_existing_keys = {}
    completed_count = 0
    for task_id in manifest_df["task_id"]:
        file_key = output_file_key(task_id, model_name)
        existing = load_existing_records(file_key)
        task_existing_keys[task_id] = {
            run_task_id_key(r["strategy"], r["rephrasing_id"], r["run_id"], r["task_id"])
            for r in existing
        }
        completed_count += len(existing)
        print(f"{task_id}: {len(existing)} generations already saved for {model_name}")

    print(f"\nStarting from {completed_count}/{total_planned} already complete.\n")

    start_time = time.time()
    max_instances = max(len(v) for v in legal_tasks.values())

    for instance_idx in range(max_instances):
        print(f"\n########## Instance index {instance_idx + 1}/{max_instances} (across all tasks) ##########")

        for _, row in manifest_df.iterrows():
            task_id, category = row["task_id"], row["category"]
            if instance_idx >= len(legal_tasks[task_id]):
                continue

            task_instance = legal_tasks[task_id][instance_idx]
            existing_keys = task_existing_keys[task_id]
            file_key = output_file_key(task_id, model_name)

            for strategy in config.ALL_STRATEGIES:
                for rephrasing_id in range(config.N_REPHRASINGS):
                    for run_id in range(config.N_RUNS):
                        key = run_task_id_key(strategy, rephrasing_id, run_id, task_instance.task_id)
                        if key in existing_keys:
                            continue

                        prompt_text = build_prompt(task_instance, strategy, rephrasing_id, task_id, demonstration_sets)
                        raw_output, parsed_answer = lpm.generate_and_parse(prompt_text)
                        is_correct = (
                            normalize_answer(parsed_answer) == normalize_answer(task_instance.expected_output)
                        ) if parsed_answer else False

                        record = schema.GenerationRecord(
                            task_id=task_instance.task_id,
                            category=category,
                            strategy=strategy,
                            rephrasing_id=rephrasing_id,
                            run_id=run_id,
                            model_name=model_name,
                            prompt_text=prompt_text,
                            raw_output=raw_output,
                            parsed_answer=parsed_answer,
                            is_correct=is_correct,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        )
                        append_record(file_key, record.__dict__)
                        existing_keys.add(key)
                        completed_count += 1

                        if completed_count % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = completed_count / elapsed if elapsed > 0 else 0
                            remaining = total_planned - completed_count
                            eta_hours = (remaining / rate / 3600) if rate > 0 else float("inf")
                            print(f"  Progress: {completed_count}/{total_planned} "
                                  f"({rate:.2f} gen/sec, ETA: {eta_hours:.1f} hrs) "
                                  f"[currently: {task_id}, instance {instance_idx}]")

            print(f"  Finished instance {instance_idx} for {task_id} ({category})")

    lpm.unload()
    print(f"\n=== Stage A run COMPLETE for {model_name}. Total generations: {completed_count} ===")


if __name__ == "__main__":
    main()