import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../Phase01HPC/ThesisWork/results/generic_vs_framework_significance.csv")
df = df[df["category"] != "__OVERALL__"]
df = df[df["model"] != "__ALL_MODELS_POOLED__"]

models = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct"]
model_labels = ["7B", "14B", "32B"]
categories = df["category"].unique()

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax, model, label in zip(axes, models, model_labels):
    sub = df[df["model"] == model].set_index("category").reindex(categories)
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, sub["generic_accuracy"], width, label="Generic", color="#4C72B0")
    ax.bar(x + width/2, sub["framework_accuracy"], width, label="Framework", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_title(label)
    ax.set_ylim(0, 1.0)

axes[0].set_ylabel("Accuracy")
axes[0].legend()
fig.suptitle("Generic vs. Framework Strategy Accuracy by Category and Model Scale")
fig.tight_layout()
fig.savefig("../Phase01HPC/ThesisWork/figures/bar_generic_vs_framework__all_models.png", dpi=150)
print("Saved: ../Phase01HPC/ThesisWork/figures/bar_generic_vs_framework__all_models.png")
