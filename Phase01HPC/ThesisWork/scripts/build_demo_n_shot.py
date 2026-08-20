import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

# Repo layout is now Phase01HPC/ThesisWork/scripts/ (one level
# deeper than before) — an extra dirname() call is needed to reach
# the true repo root, where preparations/ lives as a sibling of
# Phase01HPC/, not of ThesisWork/.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../Phase01HPC/ThesisWork/scripts
THESISWORK_DIR = os.path.dirname(SCRIPTS_DIR)                     # .../Phase01HPC/ThesisWork
PHASE01HPC_DIR = os.path.dirname(THESISWORK_DIR)                  # .../Phase01HPC
REPO_ROOT = os.path.dirname(PHASE01HPC_DIR)                       # .../DissertationCode-FIXED- (true root)
THESIS_SELECTION_PATH = os.path.join(REPO_ROOT, "preparations", "thesisSelection.csv")


def load_task_field_map():
    with open(os.path.join(config.HPC_ROOT, "data", "task_field_map.json")) as f:
        return json.load(f)


def load_train_data(task_id: str) -> list:
    """Reads ONLY train.csv for a task — the sole source of demonstrations."""
    train_path = os.path.join(config.DATA_DIR, task_id, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"train.csv not found for task '{task_id}' at {train_path} — "
            "demonstrations cannot be built without a train split."
        )
    return pd.read_csv(train_path).to_dict(orient="records")


def pick_demos_at_k(train_pool, embeddings, k):
    """
    Runs an independent k-means clustering at exactly k clusters, and
    returns the k demonstrations closest to their respective centroids.
    For k=1, this is just the single instance closest to the overall
    centroid of all embeddings (k-means with one cluster is trivial).
    """
    if k == 1:
        centroid = embeddings.mean(axis=0)
        dists = np.linalg.norm(embeddings - centroid, axis=1)
        indices = [int(np.argmin(dists))]
    else:
        kmeans = KMeans(n_clusters=k, random_state=config.CLUSTERING_RANDOM_STATE, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        indices = []
        for cluster_id in range(k):
            member_idxs = np.where(cluster_labels == cluster_id)[0]
            cluster_embeddings = embeddings[member_idxs]
            centroid = kmeans.cluster_centers_[cluster_id]
            dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            indices.append(int(member_idxs[np.argmin(dists)]))

    return indices


def main():
    if not os.path.exists(THESIS_SELECTION_PATH):
        print(f"ERROR: {THESIS_SELECTION_PATH} not found. Run thesis_test.py first.")
        return

    manifest_df = pd.read_csv(THESIS_SELECTION_PATH)
    manifest_df = manifest_df.rename(columns={"task_name": "task_id"})  # align with rest of pipeline

    task_field_map = load_task_field_map()
    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    # e.g. {1, 2, 3} — every distinct demonstration count any strategy needs.
    required_ks = sorted(set(config.DEMO_REQUIRED_STRATEGIES.values()))

    os.makedirs(config.DEMO_DIR, exist_ok=True)

    for _, row in manifest_df.iterrows():
        task_id = row["task_id"]
        train_pool = load_train_data(task_id)
        field_map = task_field_map[task_id]

        texts = [str(r.get(field_map["context"], "")) for r in train_pool]
        embeddings = embedder.encode(texts, show_progress_bar=False)

        for k in required_ks:
            if len(train_pool) < k:
                print(f"WARNING: {task_id} — train_pool={len(train_pool)} < k={k}, "
                      f"cannot build a {k}-shot demonstration set for this task.")
                continue

            demo_indices = pick_demos_at_k(train_pool, embeddings, k)

            demos = []
            for idx in demo_indices:
                r = train_pool[idx]
                context = str(r.get(field_map["context"], ""))
                question = str(r.get(field_map["question"], "")) if field_map.get("question") else ""
                label = str(r.get("answer", ""))
                demos.append({"context": context, "question": question, "label": label})

            out_path = os.path.join(config.DEMO_DIR, f"{task_id}_demos_k{k}.json")
            with open(out_path, "w") as f:
                json.dump(demos, f, indent=2)

            print(f"{task_id}: k={k} -> {len(demos)} demos (independently clustered, train_pool={len(train_pool)})")

    print("\nDemonstration building complete. Run build_eval_pools.py next.")


if __name__ == "__main__":
    main()