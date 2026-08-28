import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVAL_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "Phase01HPC", "ThesisWork"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import config

sns.set_theme(style="whitegrid")


def savefig(fig, name, safe_model_name):
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    path = os.path.join(config.FIGURES_DIR, f"{name}__{safe_model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def try_read_csv(path, index_col=None):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — skipping figures that depend on it.")
        return None
    return pd.read_csv(path, index_col=index_col)


def plot_penalty_matrix(path, title, fig_name, safe_model_name, vmin=None, vmax=None):
    matrix = try_read_csv(path, index_col=0)
    if matrix is None:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="RdYlGn_r", center=0 if vmin is None else None,
                vmin=vmin, vmax=vmax, cbar_kws={"label": "Penalty(i -> j)"}, ax=ax)
    ax.set_xlabel("Category j (strategy applied TO)")
    ax.set_ylabel("Category i (champion strategy FROM)")
    ax.set_title(title)
    savefig(fig, fig_name, safe_model_name)


def plot_fit_score_heatmap(path, title, fig_name, safe_model_name, category_order, strategy_order):
    fit_scores = try_read_csv(path)
    if fit_scores is None:
        return
    pivot = fit_scores.pivot(index="category", columns="strategy", values="fit_score_normalized")
    pivot = pivot.reindex(index=category_order, columns=strategy_order)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                cbar_kws={"label": "Fit Score (normalized 0-1)"}, ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    savefig(fig, fig_name, safe_model_name)


def main():
    model_name = config.get_model_name_from_args().model
    safe_model_name = model_name.replace("/", "_")

    parsed_path = os.path.join(config.PARSED_DIR, f"all_generations_parsed__{safe_model_name}.csv")
    per_strategy_path = os.path.join(config.RESULTS_DIR, f"metrics_per_strategy__{safe_model_name}.csv")
    per_strategy_rephrasing_path = os.path.join(config.RESULTS_DIR, f"metrics_per_strategy_rephrasing__{safe_model_name}.csv")
    per_strategy_run_path = os.path.join(config.RESULTS_DIR, f"metrics_per_strategy_run__{safe_model_name}.csv")
    per_class_path = os.path.join(config.RESULTS_DIR, f"metrics_per_class__{safe_model_name}.csv")

    for p in [parsed_path, per_strategy_path, per_strategy_rephrasing_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run parse_predictions.py and compute_metrics.py first.")
            return

    df_raw = pd.read_csv(parsed_path)
    df_raw["is_correct"] = df_raw["is_correct"].fillna(False)
    df_strategy = pd.read_csv(per_strategy_path)
    df_rephrasing = pd.read_csv(per_strategy_rephrasing_path)
    df_run = try_read_csv(per_strategy_run_path)
    df_class = try_read_csv(per_class_path)

    strategy_order = config.ALL_STRATEGIES
    category_order = config.CATEGORIES

    # ============================================================
    # EXISTING FIGURES (unchanged logic, from compute_metrics.py output)
    # ============================================================

    pivot = df_strategy.pivot(index="category", columns="strategy", values="accuracy_mean")
    pivot = pivot.reindex(index=category_order, columns=strategy_order)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                cbar_kws={"label": "Accuracy"}, ax=ax)
    ax.set_title(f"Accuracy Heatmap: Category x Strategy ({model_name})")
    plt.xticks(rotation=45, ha="right")
    savefig(fig, "heatmap_category_strategy", safe_model_name)

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.barplot(data=df_strategy, x="category", y="accuracy_mean", hue="strategy",
                order=category_order, hue_order=strategy_order, ax=ax)
    ax.set_title(f"Accuracy per Strategy, Grouped by Category ({model_name})")
    ax.set_ylabel("Mean Accuracy")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    savefig(fig, "grouped_bar_category_strategy", safe_model_name)

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, strategy in enumerate(strategy_order):
        sub = df_strategy[df_strategy["strategy"] == strategy].set_index("category").reindex(category_order)
        x = np.arange(len(category_order)) + i * 0.06
        ax.errorbar(x, sub["accuracy_mean"], yerr=sub["accuracy_std"], fmt="o", capsize=3, label=strategy, markersize=4)
    ax.set_xticks(np.arange(len(category_order)) + 0.06 * len(strategy_order) / 2)
    ax.set_xticklabels(category_order, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (mean +/- std)")
    ax.set_title(f"Accuracy with Variability, per Category and Strategy ({model_name})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    savefig(fig, "errorbar_category_strategy", safe_model_name)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df_strategy, x="strategy", y="accuracy_mean", order=strategy_order, ax=ax)
    ax.set_title(f"Accuracy Distribution per Strategy, Across Categories ({model_name})")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    savefig(fig, "boxplot_strategy_spread", safe_model_name)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    for ax, category in zip(axes.flat, category_order):
        sub = df_rephrasing[df_rephrasing["category"] == category]
        for strategy in strategy_order:
            s = sub[sub["strategy"] == strategy].sort_values("rephrasing_id")
            ax.plot(s["rephrasing_id"], s["accuracy"], marker="o", label=strategy, linewidth=1)
        ax.set_title(category)
        ax.set_xlabel("Rephrasing ID")
        ax.set_xticks([0, 1, 2])
    axes.flat[0].set_ylabel("Accuracy")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"Accuracy Across Rephrasings, per Category and Strategy ({model_name})")
    savefig(fig, "lineplot_rephrasing_sensitivity", safe_model_name)

    overall = df_strategy.groupby("strategy")["accuracy_mean"].mean().reindex(strategy_order).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=overall.values, y=overall.index, ax=ax, palette="viridis")
    ax.set_title(f"Overall Strategy Ranking (Mean Accuracy Across Categories) ({model_name})")
    ax.set_xlabel("Mean Accuracy")
    savefig(fig, "ranked_bar_overall_strategy", safe_model_name)

    top5 = overall.head(5).index.tolist()
    angles = np.linspace(0, 2 * np.pi, len(category_order), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for strategy in top5:
        sub = df_strategy[df_strategy["strategy"] == strategy].set_index("category").reindex(category_order)
        values = sub["accuracy_mean"].tolist()
        values += values[:1]
        ax.plot(angles, values, marker="o", label=strategy)
        ax.fill(angles, values, alpha=0.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(category_order, fontsize=8)
    ax.set_title(f"Top 5 Strategies: Accuracy Profile Across Categories ({model_name})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    savefig(fig, "radar_top5_strategy_profile", safe_model_name)

    df_raw["parsed_fail"] = df_raw["parsed_answer"].isna()
    fail_rate = (
        df_raw.groupby(["category", "strategy"])["parsed_fail"]
        .mean().reset_index()
        .pivot(index="category", columns="strategy", values="parsed_fail")
        .reindex(index=category_order, columns=strategy_order)
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(fail_rate, annot=True, fmt=".1%", cmap="Reds", ax=ax, cbar_kws={"label": "Parsing Failure Rate"})
    ax.set_title(f"Parsing Failure Rate: Category x Strategy ({model_name})")
    plt.xticks(rotation=45, ha="right")
    savefig(fig, "heatmap_parsing_failures", safe_model_name)

    counts = df_raw.groupby("category").size().reindex(category_order)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Blues_d")
    ax.set_title(f"Total Generations per Category ({model_name})")
    ax.set_ylabel("Generation Count")
    plt.xticks(rotation=30, ha="right")
    savefig(fig, "bar_generation_counts", safe_model_name)

    # ============================================================
    # NEW: macro-F1 heatmap (direct F1 equivalent of accuracy heatmap)
    # ============================================================
    if "macro_f1" in df_strategy.columns:
        pivot_f1 = df_strategy.pivot(index="category", columns="strategy", values="macro_f1")
        pivot_f1 = pivot_f1.reindex(index=category_order, columns=strategy_order)
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(pivot_f1, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                    cbar_kws={"label": "Macro-F1"}, ax=ax)
        ax.set_title(f"Macro-F1 Heatmap: Category x Strategy ({model_name})")
        plt.xticks(rotation=45, ha="right")
        savefig(fig, "heatmap_macro_f1", safe_model_name)

    # ============================================================
    # NEW: per-run accuracy boxplot (from the previously-discarded,
    # now-saved per-run breakdown)
    # ============================================================
    if df_run is not None:
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=df_run, x="strategy", y="accuracy", order=strategy_order, ax=ax)
        ax.set_title(f"Per-Run Accuracy Distribution per Strategy ({model_name})")
        ax.set_ylabel("Accuracy (3 runs per category-strategy pair)")
        plt.xticks(rotation=45, ha="right")
        savefig(fig, "boxplot_per_run_accuracy", safe_model_name)

    # ============================================================
    # NEW: per-class F1 heatmap — worst-performing class per (category, strategy)
    # Directly checks for the majority-class-guessing pattern
    # (Trautmann2023-style finding)
    # ============================================================
    if df_class is not None and len(df_class) > 0:
        worst_class_f1 = df_class.groupby(["category", "strategy"])["f1"].min().reset_index()
        pivot_worst = worst_class_f1.pivot(index="category", columns="strategy", values="f1")
        pivot_worst = pivot_worst.reindex(index=category_order, columns=strategy_order)
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(pivot_worst, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                    cbar_kws={"label": "Worst-Class F1"}, ax=ax)
        ax.set_title(f"Worst-Performing Class F1: Category x Strategy ({model_name})\n"
                     f"(Low values indicate a strategy may be defaulting to a majority label)")
        plt.xticks(rotation=45, ha="right")
        savefig(fig, "heatmap_worst_class_f1", safe_model_name)

    # ============================================================
    # NEW: Run / Rephrasing / Joint Fit Score heatmaps (normalized, 0-1)
    # ============================================================
    plot_fit_score_heatmap(
        os.path.join(config.RESULTS_DIR, f"run_fit_scores__{safe_model_name}.csv"),
        f"Run Fit Score (normalized): Category x Strategy ({model_name})",
        "heatmap_run_fit_score", safe_model_name, category_order, strategy_order)

    plot_fit_score_heatmap(
        os.path.join(config.RESULTS_DIR, f"rephrasing_fit_scores__{safe_model_name}.csv"),
        f"Rephrasing Fit Score (normalized): Category x Strategy ({model_name})",
        "heatmap_rephrasing_fit_score", safe_model_name, category_order, strategy_order)

    plot_fit_score_heatmap(
        os.path.join(config.RESULTS_DIR, f"joint_fit_scores__{safe_model_name}.csv"),
        f"Joint Fit Score (normalized): Category x Strategy ({model_name})",
        "heatmap_joint_fit_score", safe_model_name, category_order, strategy_order)

    # ============================================================
    # NEW: Run / Rephrasing / Joint Transfer Penalty matrices
    # ============================================================
    plot_penalty_matrix(
        os.path.join(config.RESULTS_DIR, f"run_transfer_penalty__{safe_model_name}.csv"),
        f"Run Transfer Penalty Matrix ({model_name})", "matrix_run_transfer_penalty", safe_model_name)
    plot_penalty_matrix(
        os.path.join(config.RESULTS_DIR, f"run_transfer_penalty_normalized__{safe_model_name}.csv"),
        f"Run Transfer Penalty Matrix — Normalized ({model_name})",
        "matrix_run_transfer_penalty_normalized", safe_model_name, vmin=0, vmax=1)

    plot_penalty_matrix(
        os.path.join(config.RESULTS_DIR, f"rephrasing_transfer_penalty__{safe_model_name}.csv"),
        f"Rephrasing Transfer Penalty Matrix ({model_name})", "matrix_rephrasing_transfer_penalty", safe_model_name)
    plot_penalty_matrix(
        os.path.join(config.RESULTS_DIR, f"rephrasing_transfer_penalty_normalized__{safe_model_name}.csv"),
        f"Rephrasing Transfer Penalty Matrix — Normalized ({model_name})",
        "matrix_rephrasing_transfer_penalty_normalized", safe_model_name, vmin=0, vmax=1)

    plot_penalty_matrix(
        os.path.join(config.RESULTS_DIR, f"joint_transfer_penalty__{safe_model_name}.csv"),
        f"Joint Transfer Penalty Matrix ({model_name})", "matrix_joint_transfer_penalty", safe_model_name)
    plot_penalty_matrix(
        os.path.join(config.RESULTS_DIR, f"joint_transfer_penalty_normalized__{safe_model_name}.csv"),
        f"Joint Transfer Penalty Matrix — Normalized ({model_name})",
        "matrix_joint_transfer_penalty_normalized", safe_model_name, vmin=0, vmax=1)

    # ============================================================
    # NEW: Strategy Dispersion bar chart (one bar per category)
    # ============================================================
    dispersion_path = os.path.join(config.RESULTS_DIR, f"strategy_dispersion__{safe_model_name}.csv")
    dispersion = try_read_csv(dispersion_path)
    if dispersion is not None:
        dispersion = dispersion.set_index("category").reindex(category_order).reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=dispersion, x="category", y="strategy_dispersion", ax=ax, palette="mako")
        ax.set_title(f"Strategy Dispersion per Category ({model_name})\n"
                     f"(How much accuracy varies across the 13 strategies)")
        ax.set_ylabel("Strategy Dispersion (std of accuracy across strategies)")
        plt.xticks(rotation=30, ha="right")
        savefig(fig, "bar_strategy_dispersion", safe_model_name)

    # ============================================================
    # NEW: ANOVA decomposition — % variance from rephrasing vs run vs
    # interaction, averaged across strategies, one stacked bar per category
    # ============================================================
    anova_path = os.path.join(config.RESULTS_DIR, f"anova_decomposition__{safe_model_name}.csv")
    anova = try_read_csv(anova_path)
    if anova is not None and len(anova) > 0:
        anova_summary = (
            anova.groupby("category")[["pct_variance_rephrasing", "pct_variance_run", "pct_variance_interaction"]]
            .mean()
            .reindex(category_order)
        )
        anova_summary["pct_variance_residual"] = 100 - anova_summary.sum(axis=1)
        fig, ax = plt.subplots(figsize=(10, 6))
        anova_summary.plot(kind="bar", stacked=True, ax=ax,
                            color=["#4C72B0", "#DD8452", "#C44E52", "#CCCCCC"])
        ax.set_title(f"ANOVA Variance Decomposition per Category, Averaged Across Strategies ({model_name})")
        ax.set_ylabel("% of Total Variance")
        ax.legend(["Rephrasing", "Run", "Interaction", "Residual (instance-level)"],
                  bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=30, ha="right")
        savefig(fig, "stacked_bar_anova_decomposition", safe_model_name)

    # ============================================================
    # NEW: does the champion change once you divide gain by variance? (RQ1)
    # ============================================================
    spearman_path = os.path.join(config.RESULTS_DIR, f"spearman_gain_vs_fitscore__{safe_model_name}.csv")
    spearman = try_read_csv(spearman_path)
    if spearman is not None:
        spearman = spearman.set_index("category").reindex(category_order).reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#C44E52" if c else "#55A868" for c in spearman["champion_changed"]]
        sns.barplot(data=spearman, x="category", y="spearman_rho", ax=ax, palette=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Spearman Correlation: Raw Gain Ranking vs Run Fit Score Ranking ({model_name})\n"
                     f"(Red bar = the top-ranked strategy differs between the two rankings)")
        ax.set_ylabel("Spearman's rho")
        plt.xticks(rotation=30, ha="right")
        savefig(fig, "bar_spearman_gain_vs_fitscore", safe_model_name)

    print("\nAll visualizations generated.")


if __name__ == "__main__":
    main()