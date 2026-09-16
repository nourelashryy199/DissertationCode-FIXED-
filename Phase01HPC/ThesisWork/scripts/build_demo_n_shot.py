import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

#finding the repository root from the location of this script, so thesisSelection.csv can be loaded from the preparations folder.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))          #the scripts folder
THESISWORK_DIR = os.path.dirname(SCRIPTS_DIR)                     #the ThesisWork folder
PHASE01HPC_DIR = os.path.dirname(THESISWORK_DIR)                  #the Phase01HPC folder
REPO_ROOT = os.path.dirname(PHASE01HPC_DIR)                       #the main DissertationCode-FIXED- repository folder
THESIS_SELECTION_PATH = os.path.join(REPO_ROOT, "preparations", "thesisSelection.csv")


def load_task_field_map():
    #loads the JSON file that tells the script which field contains the context and question for each LegalBench task.
    with open(os.path.join(config.HPC_ROOT, "data", "task_field_map.json")) as f:
        return json.load(f)


def load_train_data(task_id: str) -> list:
    """reads only train.csv for each task because demonstrations are selected from the training split, while the test split is kept for evaluation."""
    train_path = os.path.join(config.DATA_DIR, task_id, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"train.csv not found for task '{task_id}' at {train_path} — "
            "demonstrations cannot be built without a train split."
        )
    return pd.read_csv(train_path).to_dict(orient="records")


def pick_demos_at_k(train_pool, embeddings, k):
    """
    selects the demonstrations for one value of k using the embeddings of the training instances.
    for k=1, the demonstration closest to the mean embedding of the whole training pool is selected.
    for k=2 or k=3, KMeans is run independently with exactly k clusters, and the training instance closest to each cluster centroid is selected.

    if KMeans produces an empty cluster, the script instead chooses an unused instance that is as far as possible from the demonstrations already selected.
    this is a fallback for cases where the training instances do not separate cleanly into k distinct clusters.
    """
    if k == 1:
        centroid = embeddings.mean(axis=0)
        dists = np.linalg.norm(embeddings - centroid, axis=1)
        indices = [int(np.argmin(dists))]
        return indices

    kmeans = KMeans(n_clusters=k, random_state=config.CLUSTERING_RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    indices = []
    used = set()
    degenerate_fallback_used = False

    for cluster_id in range(k):
        member_idxs = np.where(cluster_labels == cluster_id)[0]

        if len(member_idxs) == 0:
            #if this cluster is empty, the fallback selects the unused instance that is farthest from the demonstrations already chosen, so the selected examples are still as diverse as possible.
            degenerate_fallback_used = True
            remaining = [i for i in range(len(embeddings)) if i not in used]
            if not remaining:
                break  #stops if there are no unused training instances left
            if indices:
                chosen_embeds = embeddings[indices]
                dists_to_chosen = np.array([
                    np.min(np.linalg.norm(embeddings[i] - chosen_embeds, axis=1))
                    for i in remaining
                ])
                pick = remaining[int(np.argmax(dists_to_chosen))]
            else:
                pick = remaining[0]
            indices.append(pick)
            used.add(pick)
            continue

        #for a normal non-empty cluster, selects the training instance whose embedding is closest to that cluster's centroid.
        cluster_embeddings = embeddings[member_idxs]
        centroid = kmeans.cluster_centers_[cluster_id]
        dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        chosen = int(member_idxs[np.argmin(dists)])
        indices.append(chosen)
        used.add(chosen)

    if degenerate_fallback_used:
        print(f"    NOTE: k={k} clustering was degenerate for this task (fewer than {k} "
              f"distinct clusters found, likely near-duplicate training instances) — "
              f"diversity-based fallback was used to fill the gap.")

    return indices


def main():
    #checks that the final task-selection file exists before trying to build demonstrations for the selected tasks.
    if not os.path.exists(THESIS_SELECTION_PATH):
        print(f"ERROR: {THESIS_SELECTION_PATH} not found. Run thesis_test.py first.")
        return

    #loads the final selected LegalBench tasks and uses task_id as the common task-name column throughout this pipeline.
    manifest_df = pd.read_csv(THESIS_SELECTION_PATH)
    manifest_df = manifest_df.rename(columns={"task_name": "task_id"})  #using the same task_id column name as the rest of the pipeline

    task_field_map = load_task_field_map()
    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    #gets all the different numbers of demonstrations needed by the one-shot and few-shot strategies; in this experiment these are k=1, k=2 and k=3.
    required_ks = sorted(set(config.DEMO_REQUIRED_STRATEGIES.values()))

    os.makedirs(config.DEMO_DIR, exist_ok=True)

    #builds the fixed demonstration sets separately for every selected LegalBench task.
    for _, row in manifest_df.iterrows():
        task_id = row["task_id"]
        train_pool = load_train_data(task_id)
        field_map = task_field_map[task_id]

        #embeds the context of every training instance using the sentence-transformer model specified in config.py.
        texts = [str(r.get(field_map["context"], "")) for r in train_pool]
        embeddings = embedder.encode(texts, show_progress_bar=False)

        #k=1, k=2 and k=3 are clustered independently rather than building one large demonstration set and taking subsets from it.
        for k in required_ks:
            if len(train_pool) < k:
                print(f"WARNING: {task_id} — train_pool={len(train_pool)} < k={k}, "
                      f"cannot build a {k}-shot demonstration set for this task.")
                continue

            demo_indices = pick_demos_at_k(train_pool, embeddings, k)

            #uses the selected indices to save the context, question (when the task has one), and correct label for each demonstration.
            demos = []
            for idx in demo_indices:
                r = train_pool[idx]
                context = str(r.get(field_map["context"], ""))
                question = str(r.get(field_map["question"], "")) if field_map.get("question") else ""
                label = str(r.get("answer", ""))
                demos.append({"context": context, "question": question, "label": label})

            #each task and value of k gets its own JSON file so the same fixed demonstrations can be reused during generation.
            out_path = os.path.join(config.DEMO_DIR, f"{task_id}_demos_k{k}.json")
            with open(out_path, "w") as f:
                json.dump(demos, f, indent=2)

            print(f"{task_id}: k={k} -> {len(demos)} demos (independently clustered, train_pool={len(train_pool)})")

    print("\nDemonstration building complete. Run build_eval_pools.py next.")


if __name__ == "__main__":
    main()