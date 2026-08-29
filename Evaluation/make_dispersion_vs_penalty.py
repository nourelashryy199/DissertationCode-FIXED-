import pandas as pd
from scipy.stats import pearsonr, spearmanr

RESULTS_DIR = "../Phase01HPC/ThesisWork/results"

models = {
    "7B": "Qwen_Qwen2.5-7B-Instruct",
    "14B": "Qwen_Qwen2.5-14B-Instruct",
    "32B": "Qwen_Qwen2.5-32B-Instruct",
}

rows = []
for label, tag in models.items():
    risk = pd.read_csv(f"{RESULTS_DIR}/joint_risk_summary__{tag}.csv", index_col=0)
    risk.index.name = "category"
    disp = pd.read_csv(f"{RESULTS_DIR}/strategy_dispersion__{tag}.csv")
    merged = risk.reset_index().merge(disp, on="category")
    merged["model"] = label
    rows.append(merged)

table = pd.concat(rows, ignore_index=True)
table.to_csv(f"{RESULTS_DIR}/dispersion_vs_penalty__combined.csv", index=False)
print(table.to_string(index=False))
print()

for label, g in table.groupby("model"):
    r_p, p_p = pearsonr(g["mean_penalty"], g["strategy_dispersion"])
    r_s, p_s = spearmanr(g["mean_penalty"], g["strategy_dispersion"])
    print(f"{label}: Pearson r={r_p:.3f} (p={p_p:.3f}) | Spearman rho={r_s:.3f} (p={p_s:.3f})  n=5")
