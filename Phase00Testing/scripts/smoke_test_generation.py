

import os
import sys
import json
from datetime import datetime, timezone

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../Phase00Testing/scripts
PHASE00_DIR = os.path.dirname(SCRIPTS_DIR)                        # .../Phase00Testing
REPO_ROOT = os.path.dirname(PHASE00_DIR)                          # .../DissertationCode-FIXED-
PHASE1_DIR = os.path.join(REPO_ROOT, "Phase01HPC", "ThesisWork")
THESIS_SELECTION_PATH = os.path.join(REPO_ROOT, "preparations", "thesisSelection.csv")

if not os.path.isdir(PHASE1_DIR):
    print(f"ERROR: expected to find ThesisWork at {PHASE1_DIR} — adjust the path if your repo layout differs.")
    sys.exit(1)

sys.path.insert(0, PHASE1_DIR)

import config
import schema
import pandas as pd
from model import LegalPromptModel
from strategy_functions import build_prompt

SMOKE_STRATEGIES = ["zero_shot", "cot", "irac", "few_shot_2"]  # generic, CoT, legal framework, and one n-shot
PHASE00_OUTPUT_DIR = os.path.join(PHASE00_DIR, "outputs")


def normalize_answer(s) -> str:
    if s is None:
        return ""
    return str(s).strip().lower().rstrip(".")


def load_manifest() -> pd.DataFrame:
    if not os.path.exists(THESIS_SELECTION_PATH):
        raise FileNotFoundError(f"{THESIS_SELECTION_PATH} not found. Run thesis_test.py first.")
    df = pd.read_csv(THESIS_SELECTION_PATH)
    return df.rename(columns={"task_name": "task_id"})


def main():
    model_name = config.get_model_name_from_args().model
    os.makedirs(PHASE00_OUTPUT_DIR, exist_ok=True)

    print(f"=== Smoke test starting: model={model_name} ===\n")

    manifest_df = load_manifest()
    smoke_task_id = manifest_df["task_id"].iloc[0]
    smoke_category = manifest_df[manifest_df["task_id"] == smoke_task_id]["category"].iloc[0]

    with open(os.path.join(config.EVAL_POOLS_DIR, f"{smoke_task_id}_eval.json")) as f:
        eval_pool = json.load(f)

    # Load ALL per-k demonstration files for this task (matches the
    # nested {task_id: {k: [...]}} structure build_prompt() expects).
    required_ks = sorted(set(config.DEMO_REQUIRED_STRATEGIES.values()))
    demonstration_sets = {smoke_task_id: {}}
    for k in required_ks:
        demo_path = os.path.join(config.DEMO_DIR, f"{smoke_task_id}_demos_k{k}.json")
        if not os.path.exists(demo_path):
            print(f"WARNING: {demo_path} not found — k={k} demonstrations unavailable for smoke test.")
            continue
        with open(demo_path) as f:
            demo_raw = json.load(f)
        demonstration_sets[smoke_task_id][k] = [
            schema.Demonstration(context=d["context"], question=d["question"], label=d["label"])
            for d in demo_raw
        ]
        print(f"Loaded {len(demo_raw)} real k={k} demonstrations for {smoke_task_id} "
              f"(confirms build_demo_n_shot.py output is being read correctly)")

    with open(os.path.join(config.HPC_ROOT, "data", "task_field_map.json")) as f:
        task_field_map = json.load(f)
    with open(os.path.join(config.HPC_ROOT, "data", "question_templates.json")) as f:
        question_templates = json.load(f)

    field_map = task_field_map[smoke_task_id]
    row = eval_pool[0]
    context = str(row.get(field_map["context"], ""))
    question = str(row.get(field_map["question"], "")) if field_map.get("question") else question_templates.get(smoke_task_id)

    smoke_instance = schema.LegalTask(
        task_id=f"{smoke_task_id}_0",
        task_type=smoke_category,
        context=context,
        question=question,
        label_options=sorted(set(str(r.get("answer", "")) for r in eval_pool[:20])),  # small sample for label set
        expected_output=str(row.get("answer", "")),
    )
    print(f"\nSmoke test instance: {smoke_instance.task_id} (category: {smoke_category})\n")

    lpm = LegalPromptModel(model_name)
    lpm.load()

    output_path = os.path.join(PHASE00_OUTPUT_DIR, f"smoke_test__{model_name.replace('/', '_')}.jsonl")
    results = []

    for strategy in SMOKE_STRATEGIES:
        prompt_text = build_prompt(smoke_instance, strategy, rephrasing_id=0,
                                    task_id=smoke_task_id, demonstration_sets=demonstration_sets)
        raw_output, parsed_answer = lpm.generate_and_parse(prompt_text)
        is_correct = (
            normalize_answer(parsed_answer) == normalize_answer(smoke_instance.expected_output)
        ) if parsed_answer else False

        record = {
            "task_id": smoke_instance.task_id,
            "category": smoke_category,
            "strategy": strategy,
            "model_name": model_name,
            "prompt_text": prompt_text,
            "raw_output": raw_output,
            "parsed_answer": parsed_answer,
            "expected_output": smoke_instance.expected_output,
            "is_correct": is_correct,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        results.append(record)

        print(f"--- {strategy} ---")
        print(f"Prompt (first 200 chars): {prompt_text[:200]}...")
        print(f"Expected: {smoke_instance.expected_output}")
        print(f"Parsed:   {parsed_answer}")
        print(f"Correct:  {is_correct}\n")

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    lpm.unload()

    n_correct = sum(r["is_correct"] for r in results)
    n_parsed = sum(r["parsed_answer"] is not None for r in results)
    print(f"=== Smoke test complete: {n_parsed}/{len(results)} parsed successfully, "
          f"{n_correct}/{len(results)} correct ===")
    print(f"Full output saved to: {output_path}")
    print("\nThis does NOT validate the model's accuracy (n=4, not a real result) — "
          "it validates that model loading, real demo sets (including per-k loading), "
          "prompt construction, generation, parsing, and normalized scoring all work "
          "together end-to-end with the current code.")


if __name__ == "__main__":
    main()