import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import pandas as pd

#finding the repository root from the location of this script, so thesisSelection.csv can be loaded from the preparations folder.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
THESISWORK_DIR = os.path.dirname(SCRIPTS_DIR)
PHASE01HPC_DIR = os.path.dirname(THESISWORK_DIR)
REPO_ROOT = os.path.dirname(PHASE01HPC_DIR)
THESIS_SELECTION_PATH = os.path.join(REPO_ROOT, "preparations", "thesisSelection.csv")


def load_test_data(task_id: str) -> list:
    """reads only test.csv for each task because evaluation instances should come from the test split, not from the training data used for demonstrations."""
    test_path = os.path.join(config.DATA_DIR, task_id, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"test.csv not found for task '{task_id}' at {test_path} — "
            "an evaluation pool cannot be built without a test split."
        )
    return pd.read_csv(test_path).to_dict(orient="records")


def main():
    #checks that the final task-selection file exists before trying to build the evaluation pools.
    if not os.path.exists(THESIS_SELECTION_PATH):
        print(f"ERROR: {THESIS_SELECTION_PATH} not found. Run thesis_test.py first.")
        return

    #loads the selected LegalBench tasks and renames task_name to task_id so it matches the name used throughout the rest of the pipeline.
    manifest_df = pd.read_csv(THESIS_SELECTION_PATH)
    manifest_df = manifest_df.rename(columns={"task_name": "task_id"})  #using the same task_id column name as the rest of the pipeline

    os.makedirs(config.EVAL_POOLS_DIR, exist_ok=True)

    #builds one evaluation pool for each selected task using its complete test split.
    for _, row in manifest_df.iterrows():
        task_id = row["task_id"]
        test_pool = load_test_data(task_id)

        #saves the test instances as JSON so the generation stage can load the prepared evaluation pool directly.
        with open(os.path.join(config.EVAL_POOLS_DIR, f"{task_id}_eval.json"), "w") as f:
            json.dump(test_pool, f, indent=2, default=str)

        print(f"{task_id}: eval_pool_size={len(test_pool)} (from test.csv, used in full)")

    print("\nEval pool building complete.")


if __name__ == "__main__":
    main()