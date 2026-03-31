import pandas as pd
from datasets import load_dataset
import os

def setup_gold_set(sample_size):
    print("--- Loading Big-Vul (bstee615/bigvul) ---")
    try:
        dataset = load_dataset("bstee615/bigvul", split='train')
        df = pd.DataFrame(dataset)
        
        # 1. Filter for vulnerable samples
        vulnerable_df = df[df['vul'] == 1].dropna(subset=['func_before'])
        
        # 2. Select columns relevant for the "Oracle" Baseline
        # include commit_message because it acts as the developer's explanation
        keep_cols = ['func_before', 'CWE ID', 'CVE ID', 'commit_message', 'project', 'vul']
        existing_cols = [c for c in keep_cols if c in df.columns]
        vulnerable_df = vulnerable_df[existing_cols]
        
        # 3. CONSTRUCT TRUTH PROFILE (forthe Filter Stage)
        # combine ID and Message into a single 'Oracle' field
        vulnerable_df['truth_summary'] = (
            "Type: " + vulnerable_df['CWE ID'].fillna("Unknown") + 
            " | Context: " + vulnerable_df['commit_message'].fillna("No commit message provided.")
        )
            
        # 4. Sample with a fixed seed
        actual_sample_size = min(len(vulnerable_df), sample_size)
        gold_set = vulnerable_df.sample(n=actual_sample_size, random_state=42)
        
        os.makedirs('data', exist_ok=True)
        output_path = 'data/gold_set.csv'
        gold_set.to_csv(output_path, index=False)
        
        print(f"--- Success! Saved {actual_sample_size} samples with Truth Profiles ---")
    except Exception as e:
        print(f"Failed to load dataset: {e}")

if __name__ == "__main__":
    setup_gold_set(150)