

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "candidate_tasks.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "finalSelection.csv")

N_PER_CATEGORY = 3

# Categories that get merged into one, and the name for the merged result.
MERGE_CATEGORIES = ["rule-application", "rule-conclusion"]
MERGED_NAME = "rule-application_conclusion"


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found. Run ExtractingCandidateTasks.py first.")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows ({df['task_name'].nunique()} unique tasks) from candidate_tasks.csv")

    # Merge rule-application and rule-conclusion into one category name.
    df["category"] = df["category"].replace(MERGE_CATEGORIES, MERGED_NAME)

    # A task that appeared under BOTH rule-application and rule-conclusion
    # now has two identical rows under the merged category name — drop
    # the duplicate so it's only counted once when ranking.
    df = df.drop_duplicates(subset=["category", "task_name"])

    n_categories = df["category"].nunique()
    print(f"Categories after merge: {n_categories} ({sorted(df['category'].unique())})")

    # Rank within each category: highest n_unique_labels first,
    # highest unique_label_ratio as tiebreaker.
    df_sorted = df.sort_values(
        ["category", "n_unique_labels", "unique_label_ratio"],
        ascending=[True, False, False],
    )

    selected = df_sorted.groupby("category", group_keys=False).head(N_PER_CATEGORY)
    selected = selected.sort_values(["category", "n_unique_labels", "unique_label_ratio"],
                                     ascending=[True, False, False])

    selected.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSelected {len(selected)} tasks ({N_PER_CATEGORY} per category x {n_categories} categories):")
    print(selected[["category", "task_name", "n_unique_labels", "unique_label_ratio"]].to_string(index=False))
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()