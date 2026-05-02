import pandas as pd
from datasets import load_dataset
import os
import numpy as np

def setup_gold_set(sample_size=750, num_shards=8):
    print(f"--- Loading Big-Vul (bstee615/bigvul) for {sample_size} samples ---")
    try:
        # 1. Load Dataset
        dataset = load_dataset("bstee615/bigvul", split='train')
        df = pd.DataFrame(dataset)
        
        # 2. Hard Filter: Must be vulnerable and must have code
        vulnerable_df = df[df['vul'] == 1].dropna(subset=['func_before']).copy()
        
        # 3. Rename columns for clarity in the "Judge" phase
        column_mapping = {
            'func_before': 'code',
            'CWE ID': 'cwe_id',
            'CVE ID': 'cve_id',
            'commit_message': 'dev_note'
        }
        vulnerable_df = vulnerable_df.rename(columns=column_mapping)

        # 4. Construct Professional Truth Profile
        vulnerable_df['truth_description'] = (
            "Vulnerability Type: " + vulnerable_df['cwe_id'].fillna("N/A") + 
            "\nContext: " + vulnerable_df['dev_note'].fillna("No developer context available.")
        )
            
        # 5. Strategic Sampling (Diversified CWEs)
        actual_sample_size = min(len(vulnerable_df), sample_size)
        gold_set = vulnerable_df.sample(n=actual_sample_size, random_state=42).copy()

        # 6. Parallelization Prep: Assign Shards
        gold_set = gold_set.reset_index()
        gold_set['shard_id'] = np.arange(len(gold_set)) % num_shards
        
        # 7. Save Dataset
        os.makedirs('data', exist_ok=True)
        output_path = 'data/gold_set_raw.csv'
        
        # Keep relevant columns for the full pipeline
        final_cols = ['index', 'code', 'cwe_id', 'cve_id', 'truth_description', 'project', 'shard_id']
        gold_set[final_cols].to_csv(output_path, index=False)
        
        print("-" * 30)
        print(f"SUCCESS: Saved {actual_sample_size} samples.")
        print(f"Parallelization: {num_shards} shards created (approx {actual_sample_size//num_shards} per GPU).")
        print(f"Output: {output_path}")
        print("-" * 30)

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    # Split into 8 shards parallel GPU processing, with a total of 750 samples (approx 94 per shard)
    setup_gold_set(sample_size=750, num_shards=8)