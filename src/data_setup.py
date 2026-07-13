import os

import numpy as np
import pandas as pd
from datasets import load_dataset


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "gold_set_raw.csv")


def setup_gold_set(sample_size=750, num_shards=8):
    print(
        f"--- Loading Big-Vul (bstee615/bigvul) "
        f"for {sample_size} samples ---"
    )

    try:
        # 1. Load dataset
        dataset = load_dataset(
            "bstee615/bigvul",
            split="train",
        )
        df = pd.DataFrame(dataset)

        # 2. Keep vulnerable samples containing source code
        vulnerable_df = (
            df[df["vul"] == 1]
            .dropna(subset=["func_before"])
            .copy()
        )

        # 3. Rename columns for the verification phase
        column_mapping = {
            "func_before": "code",
            "CWE ID": "cwe_id",
            "CVE ID": "cve_id",
            "commit_message": "dev_note",
        }
        vulnerable_df = vulnerable_df.rename(columns=column_mapping)

        # 4. Construct the ground-truth description
        vulnerable_df["truth_description"] = (
            "Vulnerability Type: "
            + vulnerable_df["cwe_id"].fillna("N/A").astype(str)
            + "\nContext: "
            + vulnerable_df["dev_note"]
            .fillna("No developer context available.")
            .astype(str)
        )

        # 5. Reproducible random sampling
        actual_sample_size = min(
            len(vulnerable_df),
            sample_size,
        )

        gold_set = vulnerable_df.sample(
            n=actual_sample_size,
            random_state=42,
        ).copy()

        # 6. Assign samples to shards
        gold_set = gold_set.reset_index()
        gold_set["shard_id"] = (
            np.arange(len(gold_set)) % num_shards
        )

        # 7. Save to the root-level data directory
        os.makedirs(DATA_DIR, exist_ok=True)

        final_cols = [
            "index",
            "code",
            "cwe_id",
            "cve_id",
            "truth_description",
            "project",
            "shard_id",
        ]

        gold_set[final_cols].to_csv(
            OUTPUT_PATH,
            index=False,
        )

        print("-" * 60)
        print(f"SUCCESS: Saved {actual_sample_size} samples.")
        print(
            f"Parallelization: {num_shards} shards created "
            f"(approximately {actual_sample_size // num_shards} "
            "samples per shard)."
        )
        print(f"Output: {OUTPUT_PATH}")
        print("-" * 60)

    except Exception as error:
        print(f"CRITICAL ERROR: {error}")
        raise


if __name__ == "__main__":
    setup_gold_set(
        sample_size=750,
        num_shards=8,
    )