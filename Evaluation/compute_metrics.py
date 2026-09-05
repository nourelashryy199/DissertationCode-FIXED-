import os
import sys
import json
import random

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVAL_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "Phase01HPC", "ThesisWork"))

import config
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support


def load_eval_pool_labels(sample_size=None):
    """
    Builds a lookup {(task_name, instance_idx): true_label} from every
    eval_pools/{task}_eval.json file.

    CRITICAL: results_generation.py shuffles each pool with a fixed seed
    before slicing to sample_size and enumerating, so a generation's
    task_id index (e.g. "abercrombie_0") refers to a position in the
    SHUFFLED pool, not the raw eval_pools/*.json file order. This function
    must replicate that exact same shuffle before numbering, or every
    true_label lookup will point at the wrong instance.
    """
    labels = {}
    for fname in os.listdir(config.EVAL_POOLS_DIR):
        if not fname.endswith("_eval.json"):
            continue
        task_name = fname[: -len("_eval.json")]
        with open(os.path.join(config.EVAL_POOLS_DIR, fname)) as f:
            pool = json.load(f)

        if sample_size:
            pool = pool.copy()
            random.Random(config.CLUSTERING_RANDOM_STATE).shuffle(pool)
            pool = pool[:sample_size]

        for idx, row in enumerate(pool):
            labels[(task_name, idx)] = str(row.get("answer", ""))
    return labels


def attach_true_labels(df: pd.DataFrame, sample_size=None) -> pd.DataFrame:
    """
    task_id in generation records is instance-specific, e.g. "abercrombie_23"
    (task name + instance index). Split on the LAST underscore, not the
    first, since some task names (e.g. citation_prediction_classification)
    contain underscores themselves.
    """
    label_lookup = load_eval_pool_labels(sample_size)

    def lookup(task_id: str):
        task_name, idx_str = task_id.rsplit("_", 1)
        return label_lookup.get((task_name, int(idx_str)))

    df["true_label"] = df["task_id"].apply(lookup)

    n_missing = df["true_label"].isna().sum()
    if n_missing > 0:
        print(f"WARNING: {n_missing} generation(s) could not be matched to a true label "
              f"— these will be excluded from macro-F1 / precision / recall.")

    return df


def normalize_answer(s) -> str:
    """
    Identical to results_generation.py's normalize_answer(), reapplied here
    since ground truth in eval_pools/*.json is stored raw, unnormalized
    (straight from test.csv) — only is_correct's inline comparison at
    generation time actually normalized before comparing. F1/precision/recall
    must apply the same rule, or "Yes." and "yes" would be scored as two
    different classes despite being judged identical by is_correct.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip().lower().rstrip(".")


def macro_f1(group: pd.DataFrame) -> float:
    """
    Macro-F1 for one group of generations. A missing parsed_answer (parsing
    failure) is treated as a distinct, always-wrong predicted class, so a
    strategy with a high parsing failure rate is penalised in its F1 exactly
    as it already is in accuracy — consistent, not a separate convention.
    """
    valid = group.dropna(subset=["true_label"])
    if len(valid) == 0:
        return float("nan")
    y_true = valid["true_label"].apply(normalize_answer)
    y_pred = valid["parsed_answer"].fillna("__PARSE_FAILURE__").apply(normalize_answer)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def main():
    args = config.get_model_name_from_args()
    model_name = args.model
    sample_size = args.sample_size
    safe_model_name = model_name.replace("/", "_")

    parsed_path = os.path.join(config.PARSED_DIR, f"all_generations_parsed__{safe_model_name}.csv")
    if not os.path.exists(parsed_path):
        print(f"ERROR: {parsed_path} not found. Run parse_predictions.py first.")
        return

    df = pd.read_csv(parsed_path)
    print(f"Loaded {len(df)} generations for {model_name}")

    df["is_correct"] = df["is_correct"].fillna(False)
    df = attach_true_labels(df, sample_size)

    # Persist true_label back into the parsed CSV so downstream scripts
    # (analysis.py, visualize_metrics.py) can read it directly without
    # repeating this join.
    df.to_csv(parsed_path, index=False)
    print(f"Added true_label column and re-saved: {parsed_path}")

    # ============================================================
    # Per (category, strategy, rephrasing): accuracy + macro-F1
    # ============================================================
    per_strategy_rephrasing = (
        df.groupby(["category", "strategy", "rephrasing_id"])
        .apply(lambda g: pd.Series({
            "n_generations": len(g),
            "accuracy": g["is_correct"].mean(),
            "macro_f1": macro_f1(g),
        }))
        .reset_index()
    )
    print("\n=== Per (category, strategy, rephrasing) accuracy + macro-F1 ===")
    print(per_strategy_rephrasing.to_string(index=False))

    # ============================================================
    # Per (category, strategy): accuracy (mean, std) + macro-F1
    # ============================================================
    per_strategy = (
        df.groupby(["category", "strategy"])
        .apply(lambda g: pd.Series({
            "n_generations": len(g),
            "accuracy_mean": g["is_correct"].mean(),
            "accuracy_std": g["is_correct"].std(),
            "macro_f1": macro_f1(g),
        }))
        .reset_index()
    )
    print("\n=== Per (category, strategy) accuracy (mean, std) + macro-F1 ===")
    print(per_strategy.to_string(index=False))

    # ============================================================
    # Per (category, strategy, run): accuracy + macro-F1  (NEW)
    # Previously computed only transiently inside analysis.py and discarded.
    # ============================================================
    per_strategy_run = (
        df.groupby(["category", "strategy", "run_id"])
        .apply(lambda g: pd.Series({
            "n_generations": len(g),
            "accuracy": g["is_correct"].mean(),
            "macro_f1": macro_f1(g),
        }))
        .reset_index()
    )
    print("\n=== Per (category, strategy, run) accuracy + macro-F1 ===")
    print(per_strategy_run.to_string(index=False))

    # ============================================================
    # Per (category, strategy, label): precision, recall, F1  (NEW, own CSV)
    # ============================================================
    per_class_rows = []
    for (cat, strat), group in df.groupby(["category", "strategy"]):
        valid = group.dropna(subset=["true_label"])
        if len(valid) == 0:
            continue
        y_true = valid["true_label"].apply(normalize_answer)
        y_pred = valid["parsed_answer"].fillna("__PARSE_FAILURE__").apply(normalize_answer)
        labels = sorted(y_true.unique())
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        for lbl, p, r, f, s in zip(labels, precision, recall, f1, support):
            per_class_rows.append({
                "category": cat, "strategy": strat, "label": lbl,
                "precision": p, "recall": r, "f1": f, "support": s,
            })
    per_class = pd.DataFrame(per_class_rows)
    print("\n=== Per (category, strategy, label) precision/recall/F1 ===")
    print(per_class.to_string(index=False))

    # ============================================================
    # Zero-shot baseline per category
    # ============================================================
    zero_shot_baseline = (
        per_strategy[per_strategy["strategy"] == "zero_shot"]
        .set_index("category")["accuracy_mean"]
    )
    print("\n=== Zero-shot baseline per category ===")
    print(zero_shot_baseline)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    out1 = os.path.join(config.RESULTS_DIR, f"metrics_per_strategy_rephrasing__{safe_model_name}.csv")
    per_strategy_rephrasing.to_csv(out1, index=False)

    out2 = os.path.join(config.RESULTS_DIR, f"metrics_per_strategy__{safe_model_name}.csv")
    per_strategy.to_csv(out2, index=False)

    out3 = os.path.join(config.RESULTS_DIR, f"metrics_per_strategy_run__{safe_model_name}.csv")
    per_strategy_run.to_csv(out3, index=False)

    out4 = os.path.join(config.RESULTS_DIR, f"metrics_per_class__{safe_model_name}.csv")
    per_class.to_csv(out4, index=False)

    print(f"\nSaved: {out1}")
    print(f"Saved: {out2}")
    print(f"Saved: {out3}")
    print(f"Saved: {out4}")


if __name__ == "__main__":
    main()