import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVAL_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "Phase01HPC", "ThesisWork"))

import config
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


# ============================================================
# LEVEL-ACCURACY COMPUTATION (generic — used for Run, Rephrasing,
# and Joint Fit Score alike, by varying which columns are grouped)
# ============================================================

def compute_level_accuracy(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Generic replacement for the old compute_run_level_accuracy().
    group_cols determines which "level" this is:
      ["category","strategy","run_id"]          -> run-level (3 obs/strategy)
      ["category","strategy","rephrasing_id"]    -> rephrasing-level (3 obs/strategy)
      ["category","strategy","rephrasing_id","run_id"] -> joint-level (9 obs/strategy)
    """
    return (
        df.groupby(group_cols)["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "level_accuracy"})
    )


# ============================================================
# FIT SCORE (generic construction, reused for Run/Rephrasing/Joint)
# ============================================================

def compute_fit_scores(level_df: pd.DataFrame) -> pd.DataFrame:
    """
    FitScore(category, strategy) = mean(level_accuracy gain over zero-shot)
    / std(level_accuracy), computed across whatever "level" observations
    level_df was built at (run, rephrasing, or joint run x rephrasing cells).
    """
    zero_shot = (
        level_df[level_df["strategy"] == "zero_shot"]
        .groupby("category")["level_accuracy"]
        .mean()
        .rename("zero_shot_baseline")
    )

    summary = (
        level_df.groupby(["category", "strategy"])["level_accuracy"]
        .agg(mean_accuracy="mean", std_accuracy="std", n_observations="count")
        .reset_index()
        .merge(zero_shot, on="category", how="left")
    )

    summary["accuracy_gain"] = summary["mean_accuracy"] - summary["zero_shot_baseline"]

    # Divide-by-zero guard: a strategy with zero variance across its
    # observations gets capped at +-100, sign determined by gain direction.
    summary["fit_score"] = np.where(
        summary["std_accuracy"] > 1e-9,
        summary["accuracy_gain"] / summary["std_accuracy"],
        np.where(summary["accuracy_gain"] > 0, 100.0, np.where(summary["accuracy_gain"] < 0, -100.0, 0.0)),
    )

    return summary


def add_normalized_fit_scores(fit_scores: pd.DataFrame) -> pd.DataFrame:
    """Min-max scales fit_score to 0-1 WITHIN each category."""
    def minmax_normalize(s):
        lo, hi = s.min(), s.max()
        span = hi - lo
        if span > 1e-9:
            return (s - lo) / span
        return pd.Series(0.5, index=s.index)

    fit_scores = fit_scores.copy()
    fit_scores["fit_score_normalized"] = (
        fit_scores.groupby("category")["fit_score"].transform(minmax_normalize)
    )
    return fit_scores


def identify_champions(fit_scores: pd.DataFrame) -> pd.Series:
    """For each category, the strategy with the highest Fit Score."""
    idx = fit_scores.groupby("category")["fit_score"].idxmax()
    return fit_scores.loc[idx].set_index("category")["strategy"]


# ============================================================
# TRANSFER PENALTY (generic — reused for Run/Rephrasing/Joint)
# ============================================================

def compute_transfer_penalty(fit_scores: pd.DataFrame, champions: pd.Series, score_col: str = "fit_score") -> pd.DataFrame:
    """Penalty(i -> j) = FitScore(j, champion_j) - FitScore(j, champion_i)"""
    categories = list(champions.index)
    matrix = pd.DataFrame(index=categories, columns=categories, dtype=float)
    fit_lookup = fit_scores.set_index(["category", "strategy"])[score_col]

    for j in categories:
        champion_j_score = fit_lookup.get((j, champions[j]), np.nan)
        for i in categories:
            if i == j:
                matrix.loc[i, j] = 0.0
                continue
            champion_i_on_j_score = fit_lookup.get((j, champions[i]), np.nan)
            if pd.isna(champion_i_on_j_score) or pd.isna(champion_j_score):
                matrix.loc[i, j] = np.nan
            else:
                matrix.loc[i, j] = champion_j_score - champion_i_on_j_score

    return matrix


def compute_risk_summary(penalty_matrix: pd.DataFrame) -> pd.DataFrame:
    off_diagonal = penalty_matrix.copy()
    np.fill_diagonal(off_diagonal.values, np.nan)
    return pd.DataFrame({
        "mean_penalty": off_diagonal.mean(axis=0),
        "std_penalty": off_diagonal.std(axis=0),
    })


# ============================================================
# STRATEGY DISPERSION (fundamentally different: one number per
# CATEGORY only, computed across the 13 strategies, not per
# category-strategy pair, and no baseline subtraction)
# ============================================================

def compute_strategy_dispersion(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each category, the standard deviation of pooled raw accuracy
    ACROSS all 13 strategies. High dispersion = category is sensitive
    to which strategy is used; low dispersion = category is forgiving.
    """
    per_strategy_acc = (
        df.groupby(["category", "strategy"])["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "accuracy"})
    )
    dispersion = (
        per_strategy_acc.groupby("category")["accuracy"]
        .agg(strategy_dispersion="std", mean_accuracy_across_strategies="mean")
        .reset_index()
    )
    return dispersion, per_strategy_acc


# ============================================================
# TWO-WAY ANOVA DECOMPOSITION (for Joint Fit Score's denominator:
# how much of a strategy's instability is attributable to rephrasing,
# how much to run, how much to their interaction)
# ============================================================

def two_way_anova(df: pd.DataFrame, category: str, strategy: str) -> dict:
    """
    Decomposes total variance in this (category, strategy)'s 405
    individual correctness observations into: rephrasing main effect,
    run main effect, rephrasing x run interaction, and residual
    (within-cell instance-level noise). Implemented manually (no
    statsmodels dependency) using standard balanced two-way ANOVA
    with replication formulas -- valid here since every cell has
    exactly 45 replicate instances.
    """
    sub = df[(df["category"] == category) & (df["strategy"] == strategy)]
    if len(sub) == 0:
        return None

    grand_mean = sub["is_correct"].mean()
    a_levels = sorted(sub["rephrasing_id"].unique())
    b_levels = sorted(sub["run_id"].unique())
    a, b = len(a_levels), len(b_levels)
    n_per_cell = len(sub) / (a * b)

    row_means = sub.groupby("rephrasing_id")["is_correct"].mean()
    col_means = sub.groupby("run_id")["is_correct"].mean()
    cell_means = sub.groupby(["rephrasing_id", "run_id"])["is_correct"].mean()

    ss_total = ((sub["is_correct"] - grand_mean) ** 2).sum()
    ss_a = (b * n_per_cell) * ((row_means - grand_mean) ** 2).sum()
    ss_b = (a * n_per_cell) * ((col_means - grand_mean) ** 2).sum()

    ss_ab = 0.0
    for (ai, bi), cell_mean in cell_means.items():
        ss_ab += n_per_cell * (cell_mean - row_means[ai] - col_means[bi] + grand_mean) ** 2

    ss_residual = ss_total - ss_a - ss_b - ss_ab

    df_a, df_b, df_ab = a - 1, b - 1, (a - 1) * (b - 1)
    df_residual = len(sub) - a * b
    df_total = len(sub) - 1

    def f_test(ss, df_num):
        if df_num <= 0 or df_residual <= 0 or ss_residual <= 0:
            return np.nan, np.nan
        ms = ss / df_num
        ms_residual = ss_residual / df_residual
        f_stat = ms / ms_residual if ms_residual > 0 else np.nan
        p_val = scipy_stats.f.sf(f_stat, df_num, df_residual) if not np.isnan(f_stat) else np.nan
        return f_stat, p_val

    f_a, p_a = f_test(ss_a, df_a)
    f_b, p_b = f_test(ss_b, df_b)
    f_ab, p_ab = f_test(ss_ab, df_ab)

    return {
        "category": category, "strategy": strategy,
        "ss_rephrasing": ss_a, "df_rephrasing": df_a, "F_rephrasing": f_a, "p_rephrasing": p_a,
        "ss_run": ss_b, "df_run": df_b, "F_run": f_b, "p_run": p_b,
        "ss_interaction": ss_ab, "df_interaction": df_ab, "F_interaction": f_ab, "p_interaction": p_ab,
        "ss_residual": ss_residual, "df_residual": df_residual,
        "ss_total": ss_total, "df_total": df_total,
        "pct_variance_rephrasing": 100 * ss_a / ss_total if ss_total > 0 else np.nan,
        "pct_variance_run": 100 * ss_b / ss_total if ss_total > 0 else np.nan,
        "pct_variance_interaction": 100 * ss_ab / ss_total if ss_total > 0 else np.nan,
    }


def compute_anova_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cat, strat), _ in df.groupby(["category", "strategy"]):
        result = two_way_anova(df, cat, strat)
        if result is not None:
            rows.append(result)
    return pd.DataFrame(rows)


# ============================================================
# SPEARMAN CHECK (RQ1: does dividing gain by variance -- i.e.
# ranking by Run Fit Score instead of by raw gain alone -- change
# which strategy ranks highest, per category)
# ============================================================

def compute_spearman_gain_vs_fitscore(fit_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat, group in fit_scores.groupby("category"):
        rho, p_val = scipy_stats.spearmanr(group["accuracy_gain"], group["fit_score"])
        gain_champion = group.loc[group["accuracy_gain"].idxmax(), "strategy"]
        fit_champion = group.loc[group["fit_score"].idxmax(), "strategy"]
        rows.append({
            "category": cat, "spearman_rho": rho, "p_value": p_val,
            "gain_champion": gain_champion, "fit_score_champion": fit_champion,
            "champion_changed": gain_champion != fit_champion,
        })
    return pd.DataFrame(rows)

# ============================================================
# CHAMPION DETAIL TABLE (RQ1: for each category's Joint Fit Score
# champion, show its Run/Rephrasing/Joint scores side by side with
# its ANOVA variance-source breakdown, so this doesn't need to be
# manually assembled from four separate CSVs every time)
# ============================================================

def build_champion_detail_table(joint_champions, run_fit_scores, rephrasing_fit_scores,
                                  joint_fit_scores, anova_table) -> pd.DataFrame:
    rows = []
    for category, strategy in joint_champions.items():
        run_row = run_fit_scores[(run_fit_scores["category"] == category) &
                                   (run_fit_scores["strategy"] == strategy)]
        rephrasing_row = rephrasing_fit_scores[(rephrasing_fit_scores["category"] == category) &
                                                 (rephrasing_fit_scores["strategy"] == strategy)]
        joint_row = joint_fit_scores[(joint_fit_scores["category"] == category) &
                                       (joint_fit_scores["strategy"] == strategy)]
        anova_row = anova_table[(anova_table["category"] == category) &
                                  (anova_table["strategy"] == strategy)]

        rows.append({
            "category": category,
            "champion_strategy": strategy,
            "run_fit_score": run_row["fit_score"].iloc[0] if len(run_row) else np.nan,
            "rephrasing_fit_score": rephrasing_row["fit_score"].iloc[0] if len(rephrasing_row) else np.nan,
            "joint_fit_score": joint_row["fit_score"].iloc[0] if len(joint_row) else np.nan,
            "pct_variance_rephrasing": anova_row["pct_variance_rephrasing"].iloc[0] if len(anova_row) else np.nan,
            "pct_variance_run": anova_row["pct_variance_run"].iloc[0] if len(anova_row) else np.nan,
            "pct_variance_interaction": anova_row["pct_variance_interaction"].iloc[0] if len(anova_row) else np.nan,
        })
    return pd.DataFrame(rows)
# ============================================================
# MAIN
# ============================================================
# ============================================================
# CHAMPION MARGIN (quantifies "basically tied" — the gap between
# the winning strategy and the runner-up, per category, at the
# Joint Fit Score level)
# ============================================================

def mcnemar_test(n_a_correct_b_wrong: int, n_a_wrong_b_correct: int):
    """Same adaptive McNemar's test as cross_model_significance.py, reused
    here to compare a champion against its runner-up WITHIN one model."""
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


def compute_champion_margin(joint_fit_scores: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, group in joint_fit_scores.groupby("category"):
        ranked = group.sort_values("fit_score_normalized", ascending=False).reset_index(drop=True)
        champion = ranked.loc[0, "strategy"]
        runner_up = ranked.loc[1, "strategy"] if len(ranked) > 1 else None

        if runner_up is None:
            rows.append({"category": category, "champion_strategy": champion,
                         "runner_up_strategy": None, "p_value": np.nan,
                         "effectively_tied": None})
            continue

        champ_data = df[(df["category"] == category) & (df["strategy"] == champion)]
        runner_data = df[(df["category"] == category) & (df["strategy"] == runner_up)]
        merged = champ_data[["task_id", "rephrasing_id", "run_id", "is_correct"]].merge(
            runner_data[["task_id", "rephrasing_id", "run_id", "is_correct"]],
            on=["task_id", "rephrasing_id", "run_id"], suffixes=("_champ", "_runner")
        )
        b_count = ((merged["is_correct_champ"]) & (~merged["is_correct_runner"])).sum()
        c_count = ((~merged["is_correct_champ"]) & (merged["is_correct_runner"])).sum()
        _, p_val, method = mcnemar_test(b_count, c_count)

        rows.append({
            "category": category,
            "champion_strategy": champion,
            "champion_score_normalized": ranked.loc[0, "fit_score_normalized"],
            "runner_up_strategy": runner_up,
            "runner_up_score_normalized": ranked.loc[1, "fit_score_normalized"],
            "champion_correct_runner_wrong": b_count,
            "champion_wrong_runner_correct": c_count,
            "p_value": p_val,
            "method": method,
            "effectively_tied": p_val >= 0.05,
        })
    return pd.DataFrame(rows)


# ============================================================
# COST OF NOT SWITCHING (quantifies whether accounting for
# consistency, i.e. using Fit Score instead of raw gain, actually
# matters in practice — the size of the reliability gap between
# the two candidate champions when they differ)
# ============================================================

def compute_consistency_value(fit_scores: pd.DataFrame, spearman_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in spearman_results.iterrows():
        category = row["category"]
        gain_champ = row["gain_champion"]
        fit_champ = row["fit_score_champion"]
        cat_scores = fit_scores[fit_scores["category"] == category].set_index("strategy")
        if row["champion_changed"]:
            fit_score_of_gain_champ = cat_scores.loc[gain_champ, "fit_score"] if gain_champ in cat_scores.index else np.nan
            fit_score_of_fit_champ = cat_scores.loc[fit_champ, "fit_score"] if fit_champ in cat_scores.index else np.nan
            gap = fit_score_of_fit_champ - fit_score_of_gain_champ
        else:
            gap = 0.0
        rows.append({
            "category": category,
            "champion_changed": row["champion_changed"],
            "gain_champion": gain_champ,
            "fit_score_champion": fit_champ,
            "fit_score_gap_if_stuck_with_gain_champion": gap,
        })
    return pd.DataFrame(rows)

def main():
    model_name = config.get_model_name_from_args().model
    safe_model_name = model_name.replace("/", "_")

    parsed_path = os.path.join(config.PARSED_DIR, f"all_generations_parsed__{safe_model_name}.csv")
    if not os.path.exists(parsed_path):
        print(f"ERROR: {parsed_path} not found. Run parse_predictions.py first.")
        return

    df = pd.read_csv(parsed_path)
    df["is_correct"] = df["is_correct"].fillna(False)
    print(f"Loaded {len(df)} generations for {model_name}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ---------- RUN FIT SCORE ----------
    run_level = compute_level_accuracy(df, ["category", "strategy", "run_id"])
    run_fit_scores = add_normalized_fit_scores(compute_fit_scores(run_level))
    run_champions = identify_champions(run_fit_scores)
    run_penalty = compute_transfer_penalty(run_fit_scores, run_champions, "fit_score")
    run_penalty_norm = compute_transfer_penalty(run_fit_scores, run_champions, "fit_score_normalized")
    run_risk = compute_risk_summary(run_penalty)

    print("\n=== Run Fit Score: Category Champions ===")
    print(run_champions)
    print("\n=== Run Fit Scores ===")
    print(run_fit_scores.sort_values(["category", "fit_score"], ascending=[True, False]).to_string(index=False))
    print("\n=== Run Transfer Penalty Matrix ===")
    print(run_penalty.round(3))

    run_fit_scores.to_csv(os.path.join(config.RESULTS_DIR, f"run_fit_scores__{safe_model_name}.csv"), index=False)
    run_champions.to_csv(os.path.join(config.RESULTS_DIR, f"run_champions__{safe_model_name}.csv"))
    run_penalty.to_csv(os.path.join(config.RESULTS_DIR, f"run_transfer_penalty__{safe_model_name}.csv"))
    run_penalty_norm.to_csv(os.path.join(config.RESULTS_DIR, f"run_transfer_penalty_normalized__{safe_model_name}.csv"))
    run_risk.to_csv(os.path.join(config.RESULTS_DIR, f"run_risk_summary__{safe_model_name}.csv"))

    # ---------- REPHRASING FIT SCORE ----------
    rephrasing_level = compute_level_accuracy(df, ["category", "strategy", "rephrasing_id"])
    rephrasing_fit_scores = add_normalized_fit_scores(compute_fit_scores(rephrasing_level))
    rephrasing_champions = identify_champions(rephrasing_fit_scores)
    rephrasing_penalty = compute_transfer_penalty(rephrasing_fit_scores, rephrasing_champions, "fit_score")
    rephrasing_penalty_norm = compute_transfer_penalty(rephrasing_fit_scores, rephrasing_champions, "fit_score_normalized")
    rephrasing_risk = compute_risk_summary(rephrasing_penalty)

    print("\n=== Rephrasing Fit Score: Category Champions ===")
    print(rephrasing_champions)
    print("\n=== Rephrasing Transfer Penalty Matrix ===")
    print(rephrasing_penalty.round(3))

    rephrasing_fit_scores.to_csv(os.path.join(config.RESULTS_DIR, f"rephrasing_fit_scores__{safe_model_name}.csv"), index=False)
    rephrasing_champions.to_csv(os.path.join(config.RESULTS_DIR, f"rephrasing_champions__{safe_model_name}.csv"))
    rephrasing_penalty.to_csv(os.path.join(config.RESULTS_DIR, f"rephrasing_transfer_penalty__{safe_model_name}.csv"))
    rephrasing_penalty_norm.to_csv(os.path.join(config.RESULTS_DIR, f"rephrasing_transfer_penalty_normalized__{safe_model_name}.csv"))
    rephrasing_risk.to_csv(os.path.join(config.RESULTS_DIR, f"rephrasing_risk_summary__{safe_model_name}.csv"))

    # ---------- JOINT FIT SCORE (variance over full 9-cell rephrasing x run grid) ----------
    joint_level = compute_level_accuracy(df, ["category", "strategy", "rephrasing_id", "run_id"])
    joint_fit_scores = add_normalized_fit_scores(compute_fit_scores(joint_level))
    joint_champions = identify_champions(joint_fit_scores)
    joint_penalty = compute_transfer_penalty(joint_fit_scores, joint_champions, "fit_score")
    joint_penalty_norm = compute_transfer_penalty(joint_fit_scores, joint_champions, "fit_score_normalized")
    joint_risk = compute_risk_summary(joint_penalty)

    print("\n=== Joint Fit Score: Category Champions ===")
    print(joint_champions)
    print("\n=== Joint Transfer Penalty Matrix ===")
    print(joint_penalty.round(3))

    joint_fit_scores.to_csv(os.path.join(config.RESULTS_DIR, f"joint_fit_scores__{safe_model_name}.csv"), index=False)
    joint_champions.to_csv(os.path.join(config.RESULTS_DIR, f"joint_champions__{safe_model_name}.csv"))
    joint_penalty.to_csv(os.path.join(config.RESULTS_DIR, f"joint_transfer_penalty__{safe_model_name}.csv"))
    joint_penalty_norm.to_csv(os.path.join(config.RESULTS_DIR, f"joint_transfer_penalty_normalized__{safe_model_name}.csv"))
    joint_risk.to_csv(os.path.join(config.RESULTS_DIR, f"joint_risk_summary__{safe_model_name}.csv"))

    # ---------- TWO-WAY ANOVA (decomposes Joint Fit Score's variance source) ----------
    anova_table = compute_anova_table(df)
    print("\n=== Two-Way ANOVA: Rephrasing vs Run vs Interaction (per category, strategy) ===")
    print(anova_table[["category", "strategy", "pct_variance_rephrasing", "pct_variance_run",
                        "pct_variance_interaction", "p_rephrasing", "p_run", "p_interaction"]]
          .to_string(index=False))
    anova_table.to_csv(os.path.join(config.RESULTS_DIR, f"anova_decomposition__{safe_model_name}.csv"), index=False)

        # ---------- CHAMPION DETAIL TABLE (RQ1) ----------
    champion_detail = build_champion_detail_table(
        joint_champions, run_fit_scores, rephrasing_fit_scores, joint_fit_scores, anova_table
    )
    print("\n=== Champion Detail: Run/Rephrasing/Joint Fit Score + ANOVA breakdown, per category's champion ===")
    print(champion_detail.to_string(index=False))
    champion_detail.to_csv(os.path.join(config.RESULTS_DIR, f"champion_detail__{safe_model_name}.csv"), index=False)
    # ---------- STRATEGY DISPERSION ----------
    dispersion, per_strategy_acc_for_dispersion = compute_strategy_dispersion(df)
    print("\n=== Strategy Dispersion (per category, across all 13 strategies) ===")
    print(dispersion.to_string(index=False))
    dispersion.to_csv(os.path.join(config.RESULTS_DIR, f"strategy_dispersion__{safe_model_name}.csv"), index=False)

    # ---------- SPEARMAN: does dividing by variance change the champion? (RQ1) ----------
    spearman_results = compute_spearman_gain_vs_fitscore(run_fit_scores)
    print("\n=== Spearman: Raw Gain Ranking vs Run Fit Score Ranking (per category) ===")
    print(spearman_results.to_string(index=False))
    spearman_results.to_csv(os.path.join(config.RESULTS_DIR, f"spearman_gain_vs_fitscore__{safe_model_name}.csv"), index=False)

        # ---------- CHAMPION MARGIN: is the win decisive or "basically tied"? (RQ1) ----------
    champion_margin = compute_champion_margin(joint_fit_scores, df)
    print("\n=== Champion Margin: gap over runner-up, per category (Joint Fit Score, normalized) ===")
    print(champion_margin.to_string(index=False))
    champion_margin.to_csv(os.path.join(config.RESULTS_DIR, f"champion_margin__{safe_model_name}.csv"), index=False)

    # ---------- CONSISTENCY VALUE: does accounting for consistency change anything meaningfully? (RQ1) ----------
    consistency_value = compute_consistency_value(run_fit_scores, spearman_results)
    print("\n=== Consistency Value: Fit Score gap when champion changes (Run Fit Score) ===")
    print(consistency_value.to_string(index=False))
    consistency_value.to_csv(os.path.join(config.RESULTS_DIR, f"consistency_value__{safe_model_name}.csv"), index=False)

    print(f"\nAll results saved to {config.RESULTS_DIR}")
    print("\nNOTE: McNemar's test (cross-model statistical significance, RQ3/RQ4) is NOT "
          "computed here -- it requires loading two models' generations simultaneously, "
          "which this single-model script cannot do. This belongs in a separate script.")


if __name__ == "__main__":
    main()