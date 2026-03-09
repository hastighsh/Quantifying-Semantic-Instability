import pandas as pd
from datasets import load_dataset
import os

def setup_gold_set(sample_size=50):
    print("--- Loading Big-Vul from HuggingFace ---")
    
    try:
        # Load the bstee615 version which is currently the active mirror
        dataset = load_dataset("bstee615/bigvul", split='train')
        
        df = pd.DataFrame(dataset)
        
        # Column Check: 'func_before' for the buggy code 
        # and 'vul' (int) where 1 = Vulnerable.
        if 'func_before' in df.columns and 'vul' in df.columns:
            vulnerable_df = df[df['vul'] == 1].dropna(subset=['func_before'])
        else:
            print(f"Warning: Unexpected columns found. Columns are: {df.columns}")
            # Fallback for other variants that might use 'func' or 'target'
            code_col = 'func_before' if 'func_before' in df.columns else 'func'
            label_col = 'vul' if 'vul' in df.columns else 'target'
            vulnerable_df = df[df[label_col] == 1].dropna(subset=[code_col])
            
        # Ensure we have enough samples
        actual_sample_size = min(len(vulnerable_df), sample_size)
        
        # Sample with a fixed seed
        gold_set = vulnerable_df.sample(n=actual_sample_size, random_state=42)
        
        # Ensure the data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Save to CSV
        output_path = 'data/gold_set.csv'
        gold_set.to_csv(output_path, index=False)
        
        print(f"--- Success! Saved {actual_sample_size} samples to {output_path} ---")
        print("\nQuick Audit of First Sample:")
        print(f"Vulnerability Type (CWE): {gold_set.iloc[0].get('CWE ID', 'N/A')}")
        print(f"Code Preview:\n{gold_set.iloc[0]['func_before'][:150]}...")

    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Tip: Ensure your internet connection is active and 'datasets' library is updated.")

if __name__ == "__main__":
    setup_gold_set(50)