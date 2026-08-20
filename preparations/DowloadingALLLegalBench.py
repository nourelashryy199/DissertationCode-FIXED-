import os
import pandas as pd
from datasets import get_dataset_config_names, load_dataset

LEGALBENCH_DATASET = "nguha/legalbench"
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "legalbench_all")


def download_task(task_name: str) -> dict:
    task_dir = os.path.join(OUTPUT_ROOT, task_name)
    os.makedirs(task_dir, exist_ok=True)

    ds = load_dataset(LEGALBENCH_DATASET, task_name)

    split_sizes = {}
    merged_frames = []

    for split_name, split_data in ds.items():
        df = split_data.to_pandas()
        split_sizes[split_name] = len(df)

        split_path = os.path.join(task_dir, f"{split_name}.csv")
        df.to_csv(split_path, index=False)

        df_labeled = df.copy()
        df_labeled["split"] = split_name
        merged_frames.append(df_labeled)

    merged_df = pd.concat(merged_frames, ignore_index=True)
    merged_path = os.path.join(task_dir, "merged.csv")
    merged_df.to_csv(merged_path, index=False)

    return {
        "task_name": task_name,
        "splits_found": list(ds.keys()),
        "split_sizes": split_sizes,
        "merged_size": len(merged_df),
    }


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("Fetching full list of LegalBench task configs from Hugging Face...")
    all_task_configs = get_dataset_config_names(LEGALBENCH_DATASET)
    print(f"Total LegalBench task configs available: {len(all_task_configs)}\n")

    summaries = []
    failed = []

    for i, task_name in enumerate(all_task_configs, 1):
        print(f"[{i}/{len(all_task_configs)}] Downloading '{task_name}'...")
        try:
            summary = download_task(task_name)
            summaries.append(summary)
            splits_str = ", ".join(f"{k}={v}" for k, v in summary["split_sizes"].items())
            print(f"    OK — splits: {splits_str} | merged={summary['merged_size']}")
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append(task_name)

    print(f"\n{len(summaries)} / {len(all_task_configs)} tasks downloaded successfully.")
    if failed:
        print(f"Failed tasks ({len(failed)}): {failed}")

    # Save an overall index so the later filtering step doesn't need
    # to re-query Hugging Face or re-scan every folder from scratch.
    index_rows = [
        {
            "task_name": s["task_name"],
            "splits_found": ";".join(s["splits_found"]),
            "merged_size": s["merged_size"],
            **{f"{split}_size": size for split, size in s["split_sizes"].items()},
        }
        for s in summaries
    ]
    index_df = pd.DataFrame(index_rows)
    index_path = os.path.join(OUTPUT_ROOT, "_download_index.csv")
    index_df.to_csv(index_path, index=False)
    print(f"\nSaved download index to {index_path}")
    print(f"All task folders saved under {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()