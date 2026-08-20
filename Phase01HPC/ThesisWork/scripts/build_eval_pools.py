import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import pandas as pd

# Repo layout is now Phase01HPC/ThesisWork/scripts/ — same extra
# dirname() call as build_demo_n_shot.py to reach the true repo root.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
THESISWORK_DIR = os.path.dirname(SCRIPTS_DIR)
PHASE01HPC_DIR = os.path.dirname(THESISWORK_DIR)
REPO_ROOT = os.path.dirname(PHASE01HPC_DIR)
THESIS_SELECTION_PATH = os.path.join(REPO_ROOT, "preparations", "thesisSelection.csv")


def load_test_data(task_id: str) -> list:
    """Reads ONLY test.csv for a task — the sole source of evaluation instances."""
    test_path = os.path.join(config.DATA_DIR, task_id, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"test.csv not found for task '{task_id}' at {test_path} — "
            "an evaluation pool cannot be built without a test split."
        )
    return pd.read_csv(test_path).to_dict(orient="records")


def main():
    if not os.path.exists(THESIS_SELECTION_PATH):
        print(f"ERROR: {THESIS_SELECTION_PATH} not found. Run thesis_test.py first.")
        return

    manifest_df = pd.read_csv(THESIS_SELECTION_PATH)
    manifest_df = manifest_df.rename(columns={"task_name": "task_id"})  # align with rest of pipeline

    os.makedirs(config.EVAL_POOLS_DIR, exist_ok=True)

    for _, row in manifest_df.iterrows():
        task_id = row["task_id"]
        test_pool = load_test_data(task_id)

        with open(os.path.join(config.EVAL_POOLS_DIR, f"{task_id}_eval.json"), "w") as f:
            json.dump(test_pool, f, indent=2, default=str)

        print(f"{task_id}: eval_pool_size={len(test_pool)} (from test.csv, used in full)")

    print("\nEval pool building complete.")


if __name__ == "__main__":
    main()