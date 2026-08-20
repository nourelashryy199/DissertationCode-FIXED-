
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "finalSelection.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "thesisSelection.csv")


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found. Run SelectingClassificationTasks.py first.")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} tasks ({df['category'].nunique()} categories) from finalSelection.csv")

    # Take the first row per category. finalSelection.csv is already
    # sorted by (category, n_unique_labels desc, unique_label_ratio desc),
    # so this selects each category's top-ranked task under the same
    # priority used for the original 3-per-category selection —
    # not an arbitrary or re-ranked pick.
    reduced = df.groupby("category", as_index=False, sort=False).first()

    reduced.to_csv(OUTPUT_PATH, index=False)

    print(f"\nReduced to {len(reduced)} tasks (1 per category):")
    print(reduced[["category", "task_name", "n_unique_labels", "unique_label_ratio"]].to_string(index=False))
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()