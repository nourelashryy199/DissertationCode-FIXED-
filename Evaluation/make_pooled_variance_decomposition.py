import pandas as pd

RESULTS_DIR = "../Phase01HPC/ThesisWork/results"

models = {
    "7B": "Qwen_Qwen2.5-7B-Instruct",
    "14B": "Qwen_Qwen2.5-14B-Instruct",
    "32B": "Qwen_Qwen2.5-32B-Instruct",
}

frames = []
for label, tag in models.items():
    t = pd.read_csv(f"{RESULTS_DIR}/anova_decomposition__{tag}.csv")
    t["model"] = label
    frames.append(t)

all_cells = pd.concat(frames, ignore_index=True)
ss_total = all_cells["ss_total"].sum()

pooled = {
    "pct_variance_rephrasing": 100 * all_cells["ss_rephrasing"].sum() / ss_total,
    "pct_variance_run": 100 * all_cells["ss_run"].sum() / ss_total,
    "pct_variance_interaction": 100 * all_cells["ss_interaction"].sum() / ss_total,
    "pct_variance_residual": 100 * all_cells["ss_residual"].sum() / ss_total,
    "n_cells": len(all_cells),
}
pd.DataFrame([pooled]).to_csv(f"{RESULTS_DIR}/pooled_variance_decomposition__all_models.csv", index=False)
print(pooled)