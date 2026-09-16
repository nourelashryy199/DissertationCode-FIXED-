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

#finding the repository root from the location of this script, so thesisSelection.csv can be loaded from the preparations folder.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
THESISWORK_DIR = os.path.dirname(SCRIPTS_DIR)
PHASE01HPC_DIR = os.path.dirname(THESISWORK_DIR)
REPO_ROOT = os.path.dirname(PHASE01HPC_DIR)
THESIS_SELECTION_PATH = os.path.join(REPO_ROOT, "preparations", "thesisSelection.csv")


def load_task_field_map():
    #loads the JSON file that tells the pipeline which dataset fields contain the context and question for each LegalBench task.
    with open(os.path.join(config.HPC_ROOT, "data", "task_field_map.json")) as f:
        return json.load(f)


def load_question_templates():
    #loads the fixed question templates used for tasks that do not already have their own question field.
    with open(os.path.join(config.HPC_ROOT, "data", "question_templates.json")) as f:
        return json.load(f)


def load_manifest() -> pd.DataFrame:
    #loads the final selected-task manifest. If it does not exist yet, the earlier task-selection stage needs to be run first.
    if not os.path.exists(THESIS_SELECTION_PATH):
        raise FileNotFoundError(
            f"{THESIS_SELECTION_PATH} not found. Run thesis_test.py first."
        )
    df = pd.read_csv(THESIS_SELECTION_PATH)
    return df.rename(columns={"task_name": "task_id"})  #renames task_name to task_id so it uses the same name as the rest of the pipeline


def run_task_id_key(strategy, rephrasing_id, run_id, instance_task_id):
    #creates one unique key for each strategy/rephrasing/run/instance combination. This is used to identify completed generations when resuming a run.
    return f"{strategy}|{rephrasing_id}|{run_id}|{instance_task_id}"


def output_file_key(task_id, model_name):
    #creates the filename identifier for a task/model combination. "/" is replaced because Hugging Face model names normally contain it.
    safe_model_name = model_name.replace("/", "_")
    return f"{task_id}__{safe_model_name}"


def load_existing_records(file_key: str) -> list:
    #loads generations that have already been saved, which allows an interrupted experiment to continue instead of starting again.
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
                #a partially-written line can be left behind if a job stops while writing to the file, so corrupted lines are skipped when recovering the saved generations.
                print(f"  WARNING: skipping corrupted line {line_num} in {file_key} (likely a partial write from an earlier crash)")
                continue

    #deduplicates the saved generations using the same unique experimental-condition key used during generation.
    deduped = {}
    for r in raw_records:
        key = run_task_id_key(r["strategy"], r["rephrasing_id"], r["run_id"], r["task_id"])
        deduped[key] = r

    if len(deduped) < len(raw_records):
        print(f"  {file_key}: {len(raw_records) - len(deduped)} duplicate record(s) found — deduplicated.")

    #writes the cleaned set back to the JSONL file before generation resumes.
    with open(path, "w") as f:
        for r in deduped.values():
            f.write(json.dumps(r) + "\n")

    return list(deduped.values())


def append_record(file_key: str, record: dict):
    #immediately appends one completed generation to its JSONL file instead of waiting for the whole experiment to finish.
    path = os.path.join(config.RAW_GEN_DIR, f"{file_key}_generations.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_demonstration_sets(manifest_df: pd.DataFrame) -> dict:
    """
    Loads the fixed demonstration sets that were prepared separately for k=1, k=2 and k=3.
    They are stored here as:
        {task_id: {k: [Demonstration, ...]}}
    If a task does not have enough training examples for one value of k, that demonstration set will simply be missing.
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
    Normalizes the predicted and expected answers before comparing them.
    Surrounding whitespace is removed, the answer is converted to lowercase, and a trailing period is removed.
    This prevents differences such as "Yes." and "yes" from being counted as different classifications.
    """
    if s is None:
        return ""
    return str(s).strip().lower().rstrip(".")


def build_legal_task(row: dict, task_id: str, category: str, idx: int,
                      task_field_map: dict, question_templates: dict) -> schema.LegalTask:
    #converts one row from the selected LegalBench task into the common LegalTask structure used by the generation pipeline.
    field_map = task_field_map[task_id]
    context = str(row.get(field_map["context"], ""))

    #some LegalBench tasks contain their own question field, while the others use the fixed task-level question template loaded earlier.
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
    #reads the model and optional evaluation sample size passed through the command line.
    args = config.get_model_name_from_args()
    model_name = args.model
    sample_size = args.sample_size

    os.makedirs(config.RAW_GEN_DIR, exist_ok=True)
    print(f"=== Stage A run starting: model={model_name}, sample_size={sample_size or 'FULL'} ===")

    task_field_map = load_task_field_map()
    question_templates = load_question_templates()

    #loads the five tasks selected for the final dissertation experiment.
    manifest_df = load_manifest()
    print(f"Loaded manifest from {THESIS_SELECTION_PATH}: {len(manifest_df)} tasks")

    #loads the evaluation pool prepared for each selected task.
    eval_pools = {}
    for task_id in manifest_df["task_id"]:
        with open(os.path.join(config.EVAL_POOLS_DIR, f"{task_id}_eval.json")) as f:
            eval_pools[task_id] = json.load(f)

    demonstration_sets = load_demonstration_sets(manifest_df)

    #builds the actual LegalTask instances that will be sent through the generation experiment.
    legal_tasks = {}
    for _, row in manifest_df.iterrows():
        task_id, category = row["task_id"], row["category"]
        pool = eval_pools[task_id]
        if sample_size:
            #for the dissertation experiment, the pool is shuffled using the fixed seed and the first 45 instances are selected.
            pool = pool.copy()
            random.Random(config.CLUSTERING_RANDOM_STATE).shuffle(pool)
            pool = pool[:sample_size]

        instances = [
            build_legal_task(r, task_id, category, i, task_field_map, question_templates)
            for i, r in enumerate(pool)
        ]
        #constructs the answer options from the gold labels represented in the selected evaluation instances, then gives the same options to every instance from that task.
        label_options = sorted(set(t.expected_output for t in instances))
        for t in instances:
            t.label_options = label_options
        legal_tasks[task_id] = instances
        print(f"{task_id}: {len(instances)} instances loaded (sample_size={sample_size or 'full'})")

    #total number of generations expected for this model = instances x 13 strategies x 3 rephrasings x 3 runs.
    total_planned = sum(len(v) for v in legal_tasks.values()) * len(config.ALL_STRATEGIES) * config.N_REPHRASINGS * config.N_RUNS

    #loads whichever model was specified for this job.
    lpm = LegalPromptModel(model_name)
    lpm.load()

    #checks the existing output files before starting so already-completed generations are not repeated.
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

    #generation is interleaved by instance index across the five tasks rather than completing an entire task before moving to the next one.
    for instance_idx in range(max_instances):
        print(f"\n########## Instance index {instance_idx + 1}/{max_instances} (across all tasks) ##########")

        for _, row in manifest_df.iterrows():
            task_id, category = row["task_id"], row["category"]
            if instance_idx >= len(legal_tasks[task_id]):
                continue

            task_instance = legal_tasks[task_id][instance_idx]
            existing_keys = task_existing_keys[task_id]
            file_key = output_file_key(task_id, model_name)

            #for each instance, every strategy is evaluated under each of the 3 rephrasings and each of the 3 stochastic runs.
            for strategy in config.ALL_STRATEGIES:
                for rephrasing_id in range(config.N_REPHRASINGS):
                    for run_id in range(config.N_RUNS):
                        key = run_task_id_key(strategy, rephrasing_id, run_id, task_instance.task_id)
                        if key in existing_keys:
                            #skip this combination if it was already completed in an earlier run of the job.
                            continue

                        prompt_text = build_prompt(task_instance, strategy, rephrasing_id, task_id, demonstration_sets)
                        raw_output, parsed_answer = lpm.generate_and_parse(prompt_text)
                        is_correct = (
                            normalize_answer(parsed_answer) == normalize_answer(task_instance.expected_output)
                        ) if parsed_answer else False

                        #stores both the raw response and the parsed/scored result together with the experimental condition that produced them.
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
                        #the record is saved immediately so progress is preserved if the job is interrupted.
                        append_record(file_key, record.__dict__)
                        existing_keys.add(key)
                        completed_count += 1

                        #prints a progress update every 100 completed generations, including the current generation rate and estimated remaining time.
                        if completed_count % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = completed_count / elapsed if elapsed > 0 else 0
                            remaining = total_planned - completed_count
                            eta_hours = (remaining / rate / 3600) if rate > 0 else float("inf")
                            print(f"  Progress: {completed_count}/{total_planned} "
                                  f"({rate:.2f} gen/sec, ETA: {eta_hours:.1f} hrs) "
                                  f"[currently: {task_id}, instance {instance_idx}]")

            print(f"  Finished instance {instance_idx} for {task_id} ({category})")

    #unloads the model and clears its GPU memory after every planned generation has been completed.
    lpm.unload()
    print(f"\n=== Stage A run COMPLETE for {model_name}. Total generations: {completed_count} ===")


if __name__ == "__main__":
    main()