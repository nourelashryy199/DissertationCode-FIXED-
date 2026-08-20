

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGALBENCH_ALL_DIR = os.path.join(SCRIPT_DIR, "data", "legalbench_all")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "candidate_tasks.csv")

# --- Classification-style screening thresholds ---
MAX_UNIQUE_LABEL_RATIO = 0.1     # unique answers should be <=10% of examples
MAX_UNIQUE_LABELS_ABSOLUTE = 20  # hard cap regardless of dataset size

# --- Prefix-based category assignment ---
# Several task groups in Table 10 are listed under one umbrella name
# (e.g. "CUAD Tasks", "MAUD Tasks") but actually expand to dozens of
# real, differently-named sub-tasks (cuad_anti-assignment,
# cuad_audit_rights, etc.). Rather than hardcode every sub-variant,
# these families are matched by prefix instead of exact name.
CATEGORY_PREFIX_MAP = {
    "cuad_": "interpretation",
    "contract_nli_": "interpretation",
    "maud_": "interpretation",
    "opp115_": "interpretation",
    "supply_chain_disclosure_": "interpretation",
    "textualism_tool_": "rhetorical-understanding",
    "learned_hands_": "issue-spotting",
    "diversity_": "rule-application",  # also applies to rule-conclusion — added below
}

# --- Exact-name category assignment ---
# Standalone tasks (one task = one paper entry, no sub-variants),
# per Guha et al., LegalBench paper, Table 10.
#
# NOTE: rule-application and rule-conclusion share the SAME task
# list in the paper's own table — this is intentional, not a
# duplication error.
CATEGORY_TASK_MAP = {
    "issue-spotting": [
        "corporate_lobbying",
    ],
    "rule-recall": [
        "citation_prediction_classification",
        "international_citizenship_questions",
        "nys_judicial_ethics",  # corrected slug (was ny_state_judicial_ethics)
        "rule_qa",
    ],
    "rule-application": [
        "abercrombie",
        "hearsay", "personal_jurisdiction", "successor_liability",
        "telemarketing_sales_rule", "ucc_v_common_law",
    ],
    "rule-conclusion": [
        # Identical to rule-application per Table 10 — see note above.
        "abercrombie",
        "hearsay", "personal_jurisdiction", "successor_liability",
        "telemarketing_sales_rule", "ucc_v_common_law",
    ],
    "interpretation": [
        "consumer_contracts_qa", "contract_qa",
        "insurance_policy_interpretation", "jcrew_blocker",
        "privacy_policy_entailment", "privacy_policy_qa", "proa",
        "sara_entailment", "securities_complaint_extraction",
        "unfair_tos",
    ],
    "rhetorical-understanding": [
        "canada_tax_court_outcomes", "definition_classification",
        "function_of_decision_section", "legal_reasoning_causality",
        "oral_argument_question_purpose", "overruling", "scalr",
    ],
}

# Build a reverse lookup: task_name -> list of categories it belongs to
# (a task can appear in more than one category — see rule-application /
# rule-conclusion note above — so this maps to a list, not a single value).
TASK_TO_CATEGORIES = {}
for category, task_names in CATEGORY_TASK_MAP.items():
    for task_name in task_names:
        TASK_TO_CATEGORIES.setdefault(task_name, []).append(category)

# diversity_* belongs to BOTH rule-application and rule-conclusion,
# same as the exact-name entries above — added here since it's
# prefix-matched rather than individually enumerated.
DIVERSITY_EXTRA_CATEGORY = "rule-conclusion"


def categories_for_task(task_name: str) -> list:
    """Returns the list of categories a task belongs to, via exact match first, then prefix match."""
    if task_name in TASK_TO_CATEGORIES:
        return TASK_TO_CATEGORIES[task_name]

    for prefix, category in CATEGORY_PREFIX_MAP.items():
        if task_name.startswith(prefix):
            if prefix == "diversity_":
                return [category, DIVERSITY_EXTRA_CATEGORY]
            return [category]

    return ["unmapped"]


def screen_task(task_name):
    """Screens one locally-downloaded task's test.csv for classification structure."""
    test_path = os.path.join(LEGALBENCH_ALL_DIR, task_name, "test.csv")
    if not os.path.exists(test_path):
        print(f"SKIP {task_name}: no local test.csv found — likely a train-only or single-split task.")
        return None

    df = pd.read_csv(test_path)

    answer_col = None
    for candidate_col in ["answer", "label"]:
        if candidate_col in df.columns:
            answer_col = candidate_col
            break

    if answer_col is None:
        print(f"SKIP {task_name}: no recognizable answer/label column — likely open-ended.")
        return None

    n_total = len(df)
    if n_total == 0:
        print(f"SKIP {task_name}: test.csv is empty.")
        return None

    n_unique = df[answer_col].nunique()
    ratio = n_unique / n_total
    is_classification = (n_unique <= MAX_UNIQUE_LABELS_ABSOLUTE) and (ratio <= MAX_UNIQUE_LABEL_RATIO)

    categories = categories_for_task(task_name)

    return {
        "task_name": task_name,
        "category": categories,
        "test_size": n_total,
        "n_unique_labels": n_unique,
        "unique_label_ratio": round(ratio, 4),
        "is_classification": is_classification,
    }


def main():
    if not os.path.isdir(LEGALBENCH_ALL_DIR):
        print(f"ERROR: {LEGALBENCH_ALL_DIR} not found. Run DowloadingALLLegalBench.py first.")
        return

    task_names = sorted(
        d for d in os.listdir(LEGALBENCH_ALL_DIR)
        if os.path.isdir(os.path.join(LEGALBENCH_ALL_DIR, d))
    )
    print(f"Found {len(task_names)} locally-downloaded tasks to screen.\n")

    records = []
    for task_name in task_names:
        result = screen_task(task_name)
        if result is not None:
            # A task belonging to multiple categories (rule-application /
            # rule-conclusion) gets one row per category, so each category
            # can be filtered independently in the final CSV.
            for category in result["category"]:
                row = dict(result)
                row["category"] = category
                records.append(row)

    all_df = pd.DataFrame(records)
    n_total_rows = len(all_df)

    # Keep only rows that passed the classification screen — this file
    # is meant to contain candidate (classification-style) tasks only.
    candidates_df = all_df[all_df["is_classification"]].drop(columns=["is_classification"])
    candidates_df = candidates_df[
        ["category", "task_name", "test_size", "n_unique_labels", "unique_label_ratio"]
    ]
    candidates_df = candidates_df.sort_values(["category", "task_name"])
    candidates_df.to_csv(OUTPUT_PATH, index=False)

    n_unmapped = (candidates_df["category"] == "unmapped").sum()
    print(f"\n{len(candidates_df)} / {n_total_rows} rows passed the classification screen and were kept.")
    if n_unmapped:
        print(f"{n_unmapped} of those are 'unmapped' — real local tasks not matched to any Table 10 category. "
              f"Check these manually against the paper if you want to categorize them.")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()