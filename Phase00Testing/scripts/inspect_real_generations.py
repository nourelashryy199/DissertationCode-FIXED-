

import os
import sys
import json
import argparse
import random

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE00_DIR = os.path.dirname(SCRIPTS_DIR)
REPO_ROOT = os.path.dirname(PHASE00_DIR)
PHASE1_DIR = os.path.join(REPO_ROOT, "Phase01HPC", "ThesisWork")
THESIS_SELECTION_PATH = os.path.join(REPO_ROOT, "preparations", "thesisSelection.csv")

if not os.path.isdir(PHASE1_DIR):
    print(f"ERROR: expected to find ThesisWork at {PHASE1_DIR} — adjust the path if your repo layout differs.")
    sys.exit(1)

sys.path.insert(0, PHASE1_DIR)

import config
import pandas as pd


def load_manifest() -> pd.DataFrame:
    if not os.path.exists(THESIS_SELECTION_PATH):
        raise FileNotFoundError(f"{THESIS_SELECTION_PATH} not found. Run thesis_test.py first.")
    df = pd.read_csv(THESIS_SELECTION_PATH)
    return df.rename(columns={"task_name": "task_id"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL_NAME)
    parser.add_argument("--n", type=int, default=10, help="Number of records to sample and print")
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    safe_model_name = args.model.replace("/", "_")
    manifest_df = load_manifest()

    all_records = []
    for task_id in manifest_df["task_id"]:
        filepath = os.path.join(config.RAW_GEN_DIR, f"{task_id}__{safe_model_name}_generations.jsonl")
        if not os.path.exists(filepath):
            print(f"WARNING: {filepath} not found — skipping")
            continue
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))

    if not all_records:
        print("No records found. Has results_generation.py been run for this model yet?")
        return

    print(f"Loaded {len(all_records)} total real generations for {args.model}\n")

    # Sample diversely: try to cover a spread of strategies, not just random rows,
    # since a pure random sample can easily miss legal-framework strategies
    # (7 generic strategies vs. 6 framework ones, similar counts, but worth
    # deliberately ensuring coverage rather than leaving it to chance).
    by_strategy = {}
    for r in all_records:
        by_strategy.setdefault(r["strategy"], []).append(r)

    rng = random.Random(args.seed)
    sample = []
    strategies = list(by_strategy.keys())
    per_strategy = max(1, args.n // len(strategies))
    for strategy in strategies:
        sample.extend(rng.sample(by_strategy[strategy], min(per_strategy, len(by_strategy[strategy]))))
    sample = sample[:args.n]

    for i, r in enumerate(sample, 1):
        print("=" * 70)
        print(f"[{i}/{len(sample)}] task_id={r['task_id']}  category={r['category']}  strategy={r['strategy']}  "
              f"rephrasing={r.get('rephrasing_id')}  run={r.get('run_id')}")
        print("-" * 70)
        prompt = r.get("prompt_text", "")
        print(f"PROMPT (first 300 chars):\n{prompt[:300]}{'...' if len(prompt) > 300 else ''}\n")
        raw = r.get("raw_output", "")
        print(f"RAW OUTPUT (first 400 chars):\n{raw[:400]}{'...' if len(raw) > 400 else ''}\n")
        print(f"PARSED ANSWER: {r.get('parsed_answer')}")
        print(f"IS_CORRECT (normalized match, as scored at generation time): {r.get('is_correct')}")
        print(f"RAW OUTPUT LENGTH (word count, truncation proxy): {len(str(raw).split())}")
        print()

    print("=" * 70)
    print(f"Printed {len(sample)} records spanning {len(set(r['strategy'] for r in sample))} strategies. "
          "Manually check: does each prompt look well-formed for its strategy? Does the raw output look "
          "coherent (not cut off mid-sentence)? Does the parsed answer plausibly come from that raw output?")


if __name__ == "__main__":
    main()