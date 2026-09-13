import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "../Phase01HPC/ThesisWork/results"
FIGURES_DIR = "../Phase01HPC/ThesisWork/figures"

models = {
    "7B": "Qwen_Qwen2.5-7B-Instruct",
    "14B": "Qwen_Qwen2.5-14B-Instruct",
    "32B": "Qwen_Qwen2.5-32B-Instruct",
    "Pooled": "all_models",
}

rows = []
for label, tag in models.items():
    row = pd.read_csv(f"{RESULTS_DIR}/pooled_variance_decomposition__{tag}.csv").iloc[0]
    rows.append({"model": label, "Rephrasing": row["pct_variance_rephrasing"],
                 "Run": row["pct_variance_run"], "Interaction": row["pct_variance_interaction"],
                 "Residual": row["pct_variance_residual"]})

df = pd.DataFrame(rows).set_index("model")
print(df)

fig, ax = plt.subplots(figsize=(8, 5))
df.plot(kind="bar", stacked=True, ax=ax,
        color=["#4C72B0", "#DD8452", "#C44E52", "#CCCCCC"])
ax.set_ylabel("% of Total Variance")
ax.set_title("Pooled Variance Decomposition, by Model Scale and Overall")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=0)
fig.savefig(f"{FIGURES_DIR}/bar_pooled_variance_decomposition.png", dpi=150, bbox_inches="tight")
print(f"Saved: {FIGURES_DIR}/bar_pooled_variance_decomposition.png")
