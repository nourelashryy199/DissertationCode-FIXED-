import pandas as pd

RESULTS_DIR = "../Phase01HPC/ThesisWork/results"

champs = pd.read_csv(f"{RESULTS_DIR}/joint_champion_comparison_across_models.csv")
mc = pd.read_csv(f"{RESULTS_DIR}/mcnemar_cross_model.csv")

model_cols = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]

champ_lookup = {}
for _, row in champs.iterrows():
    for model in model_cols:
        champ_lookup[(row["category"], model)] = row[model]

def is_relevant(row):
    ca = champ_lookup.get((row["category"], row["model_a"]))
    cb = champ_lookup.get((row["category"], row["model_b"]))
    return row["strategy"] in (ca, cb)

filtered = mc[mc.apply(is_relevant, axis=1)].copy()
filtered = filtered.sort_values(["category", "model_a", "model_b"])
filtered.to_csv(f"{RESULTS_DIR}/mcnemar_champions_filtered.csv", index=False)

print(f"Filtered to {len(filtered)} rows (from {len(mc)})")
print(filtered[["model_a","model_b","category","strategy","accuracy_a","accuracy_b","p_value","significant_at_0.05"]].to_string(index=False))
