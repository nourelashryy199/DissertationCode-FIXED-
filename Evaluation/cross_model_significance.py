import os
import sys
from itertools import combinations

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVAL_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "Phase01HPC", "ThesisWork"))

import config
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

# Hardcoded, since this script's whole purpose is comparing across all
# three models at once — unlike every other Evaluation/ script, there is
# no single --model to parse from the command line here.
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]


def load_model_df(model_name: str) -> pd.DataFrame:
    safe_name = model_name.replace("/", "_")
    path = os.path.join(config.PARSED_DIR, f"all_generations_parsed__{safe_name}.csv")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — skipping {model_name}.")
        return None
    df = pd.read_csv(path)
    df["is_correct"] = df["is_correct"].fillna(False)
    df["match_key"] = (
        df["task_id"].astype(str) + "|" +
        df["rephrasing_id"].astype(str) + "|" +
        df["run_id"].astype(str)
    )
    return df


# ============================================================
# RQ3 — McNemar's test: does a category's champion strategy behave
# the same way when the model changes? Paired on exact matching
# (task_id, rephrasing_id, run_id) between two models.
# ============================================================

def mcnemar_test(n_a_correct_b_wrong: int, n_a_wrong_b_correct: int):
    """
    Standard adaptive McNemar's test:
    - exact two-sided binomial test when discordant pairs < 25
      (chi-square approximation is unreliable at small n)
    - continuity-corrected chi-square otherwise
    Returns (statistic_or_None, p_value, method_used).
    """
    b, c = n_a_correct_b_wrong, n_a_wrong_b_correct
    n_discordant = b + c
    if n_discordant == 0:
        return None, 1.0, "no_discordant_pairs"
    if n_discordant < 25:
        result = scipy_stats.binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided")
        return None, result.pvalue, "exact_binomial"
    chi2 = ((abs(b - c) - 1) ** 2) / n_discordant
    p_val = scipy_stats.chi2.sf(chi2, df=1)
    return chi2, p_val, "chi_square_corrected"


def run_mcnemar_all_pairs(dfs: dict) -> pd.DataFrame:
    rows = []
    for model_a, model_b in combinations(dfs.keys(), 2):
        df_a, df_b = dfs[model_a], dfs[model_b]
        for category in config.CATEGORIES:
            for strategy in config.ALL_STRATEGIES:
                sub_a = df_a[(df_a["category"] == category) & (df_a["strategy"] == strategy)]
                sub_b = df_b[(df_b["category"] == category) & (df_b["strategy"] == strategy)]
                merged = sub_a[["match_key", "is_correct"]].merge(
                    sub_b[["match_key", "is_correct"]], on="match_key", suffixes=("_a", "_b")
                )
                if len(merged) == 0:
                    continue
                b_count = ((merged["is_correct_a"]) & (~merged["is_correct_b"])).sum()
                c_count = ((~merged["is_correct_a"]) & (merged["is_correct_b"])).sum()
                both_correct = (merged["is_correct_a"] & merged["is_correct_b"]).sum()
                both_wrong = ((~merged["is_correct_a"]) & (~merged["is_correct_b"])).sum()
                stat, p_val, method = mcnemar_test(b_count, c_count)
                rows.append({
                    "model_a": model_a, "model_b": model_b, "category": category, "strategy": strategy,
                    "n_matched_pairs": len(merged),
                    "both_correct": both_correct, "both_wrong": both_wrong,
                    f"a_correct_b_wrong": b_count, f"a_wrong_b_correct": c_count,
                    "accuracy_a": merged["is_correct_a"].mean(), "accuracy_b": merged["is_correct_b"].mean(),
                    "test_statistic": stat, "p_value": p_val, "method": method,
                    "significant_at_0.05": (p_val is not None) and (p_val < 0.05),
                })
    return pd.DataFrame(rows)


# ============================================================
# RQ4 — Two-proportion z-test: generic vs. legal-framework
# strategies, pooled correctness, per category, per model, and
# overall (pooled across all categories and all models).
# ============================================================

def two_proportion_z_test(x1, n1, x2, n2):
    """Standard two-proportion z-test (normal approximation; valid given
    the large sample sizes here -- thousands of generations per group)."""
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return p1 - p2, np.nan, np.nan
    z = (p1 - p2) / se
    p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
    return p1 - p2, z, p_val


def run_generic_vs_framework_tests(dfs: dict) -> pd.DataFrame:
    rows = []
    for model_name, df in dfs.items():
        df = df.copy()
        df["strategy_type"] = df["strategy"].apply(
            lambda s: "legal_framework" if s in config.LEGAL_FRAMEWORK_STRATEGIES else "generic"
        )
        # per category, within this model
        for category in config.CATEGORIES:
            sub = df[df["category"] == category]
            generic = sub[sub["strategy_type"] == "generic"]
            framework = sub[sub["strategy_type"] == "legal_framework"]
            diff, z, p_val = two_proportion_z_test(
                generic["is_correct"].sum(), len(generic),
                framework["is_correct"].sum(), len(framework),
            )
            rows.append({
                "model": model_name, "category": category,
                "generic_accuracy": generic["is_correct"].mean(),
                "framework_accuracy": framework["is_correct"].mean(),
                "accuracy_diff_generic_minus_framework": diff,
                "z_statistic": z, "p_value": p_val,
                "significant_at_0.05": (not np.isnan(p_val)) and (p_val < 0.05),
            })
        # overall, pooled across all categories, within this model
        generic = df[df["strategy_type"] == "generic"]
        framework = df[df["strategy_type"] == "legal_framework"]
        diff, z, p_val = two_proportion_z_test(
            generic["is_correct"].sum(), len(generic),
            framework["is_correct"].sum(), len(framework),
        )
        rows.append({
            "model": model_name, "category": "__OVERALL__",
            "generic_accuracy": generic["is_correct"].mean(),
            "framework_accuracy": framework["is_correct"].mean(),
            "accuracy_diff_generic_minus_framework": diff,
            "z_statistic": z, "p_value": p_val,
            "significant_at_0.05": (not np.isnan(p_val)) and (p_val < 0.05),
        })

    # fully pooled across ALL models and ALL categories
    all_df = pd.concat(dfs.values(), ignore_index=True)
    all_df["strategy_type"] = all_df["strategy"].apply(
        lambda s: "legal_framework" if s in config.LEGAL_FRAMEWORK_STRATEGIES else "generic"
    )
    generic = all_df[all_df["strategy_type"] == "generic"]
    framework = all_df[all_df["strategy_type"] == "legal_framework"]
    diff, z, p_val = two_proportion_z_test(
        generic["is_correct"].sum(), len(generic),
        framework["is_correct"].sum(), len(framework),
    )
    rows.append({
        "model": "__ALL_MODELS_POOLED__", "category": "__OVERALL__",
        "generic_accuracy": generic["is_correct"].mean(),
        "framework_accuracy": framework["is_correct"].mean(),
        "accuracy_diff_generic_minus_framework": diff,
        "z_statistic": z, "p_value": p_val,
        "significant_at_0.05": (not np.isnan(p_val)) and (p_val < 0.05),
    })

    return pd.DataFrame(rows)

# ============================================================
# CHAMPION COMPARISON TABLE (RQ1/RQ3: category x model, showing
# each model's Joint Fit Score champion side by side, so cross-model
# champion stability can be read at a glance without manually
# cross-referencing three separate joint_champions__*.csv files)
# ============================================================

def build_champion_comparison_table() -> pd.DataFrame:
    rows = {}
    for model_name in MODELS:
        safe_name = model_name.replace("/", "_")
        path = os.path.join(config.RESULTS_DIR, f"joint_champions__{safe_name}.csv")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found — run analysis.py for {model_name} first. "
                  f"Skipping this model in the champion comparison table.")
            continue
        champions = pd.read_csv(path, index_col=0).iloc[:, 0]
        rows[model_name] = champions

    table = pd.DataFrame(rows)
    table.index.name = "category"

    if table.shape[1] > 1:
        table["all_models_agree"] = table.apply(lambda row: row.nunique() == 1, axis=1)

    return table

# ============================================================
# POOLED DISPERSION VS PENALTY CORRELATION (RQ2: same check as
# analysis.py's per-model version, but pooled across all three
# models -- n=15 category-model pairs instead of n=5, giving more
# statistical power to the "does Dispersion predict Transfer
# Penalty cost" question)
# ============================================================

def compute_pooled_dispersion_penalty_correlation() -> dict:
    rows = []
    for model_name in MODELS:
        safe_name = model_name.replace("/", "_")
        dispersion_path = os.path.join(config.RESULTS_DIR, f"strategy_dispersion__{safe_name}.csv")
        risk_path = os.path.join(config.RESULTS_DIR, f"joint_risk_summary__{safe_name}.csv")
        if not os.path.exists(dispersion_path) or not os.path.exists(risk_path):
            print(f"WARNING: dispersion or risk summary missing for {model_name} — "
                  f"skipping in pooled correlation. Run analysis.py for this model first.")
            continue
        dispersion = pd.read_csv(dispersion_path)
        risk = pd.read_csv(risk_path, index_col=0)
        merged = dispersion.set_index("category").join(risk, how="inner")
        merged["model"] = model_name
        rows.append(merged)

    if len(rows) == 0:
        return {"n": 0, "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_rho": np.nan, "spearman_p": np.nan}

    pooled = pd.concat(rows)
    pearson_r, pearson_p = scipy_stats.pearsonr(pooled["strategy_dispersion"], pooled["mean_penalty"])
    spearman_rho, spearman_p = scipy_stats.spearmanr(pooled["strategy_dispersion"], pooled["mean_penalty"])

    return {
        "n": len(pooled),
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_rho": spearman_rho, "spearman_p": spearman_p,
    }
def main():
    dfs = {}
    for model_name in MODELS:
        df = load_model_df(model_name)
        if df is not None:
            dfs[model_name] = df

    if len(dfs) < 2:
        print(f"ERROR: need at least 2 models' parsed data to run cross-model tests; found {len(dfs)}.")
        return

    print(f"Loaded {len(dfs)} models: {list(dfs.keys())}")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("\n=== RQ3: McNemar's Test (model-to-model, per category/strategy) ===")
    mcnemar_results = run_mcnemar_all_pairs(dfs)
    print(mcnemar_results.to_string(index=False))
    mcnemar_path = os.path.join(config.RESULTS_DIR, "mcnemar_cross_model.csv")
    mcnemar_results.to_csv(mcnemar_path, index=False)
    print(f"Saved: {mcnemar_path}")

    n_sig = mcnemar_results["significant_at_0.05"].sum()
    print(f"\n{n_sig} / {len(mcnemar_results)} (category, strategy, model-pair) comparisons "
          f"show a statistically significant difference (p < 0.05).")

    print("\n=== RQ4: Two-Proportion Z-Test (generic vs. legal-framework strategies) ===")
    generic_vs_framework = run_generic_vs_framework_tests(dfs)
    print(generic_vs_framework.to_string(index=False))
    gvf_path = os.path.join(config.RESULTS_DIR, "generic_vs_framework_significance.csv")
    generic_vs_framework.to_csv(gvf_path, index=False)
    print(f"Saved: {gvf_path}")
    print("\n=== Champion Comparison: Joint Fit Score champion per category, per model (RQ1/RQ3) ===")
    champion_comparison = build_champion_comparison_table()
    print(champion_comparison.to_string())
    champion_comparison_path = os.path.join(config.RESULTS_DIR, "joint_champion_comparison_across_models.csv")
    champion_comparison.to_csv(champion_comparison_path)
    print(f"Saved: {champion_comparison_path}")
    print("\n=== Pooled Dispersion vs Transfer Penalty Correlation (across all 3 models, n=15) (RQ2) ===")
    pooled_corr = compute_pooled_dispersion_penalty_correlation()
    print(pooled_corr)
    pd.DataFrame([pooled_corr]).to_csv(
        os.path.join(config.RESULTS_DIR, "pooled_dispersion_penalty_correlation.csv"), index=False
    )
    print(f"Saved: {os.path.join(config.RESULTS_DIR, 'pooled_dispersion_penalty_correlation.csv')}")


if __name__ == "__main__":
    main()