import pandas as pd

def baseline_cleaning(path):
    df = pd.read_csv(path)
    
    # 1. Fix the Byte-Level artifacts
    # Ġ is the 'Metaspace' for a space, Ċ is a 'Newline'
    df['baseline_explanation'] = df['baseline_explanation'].replace({'Ġ': ' ', 'Ċ': '\n'}, regex=True)
    
    # 2. Basic cleanup
    df['baseline_explanation'] = df['baseline_explanation'].str.strip()
    
    # Save the 'Clean' version for the Judge
    clean_path = path.replace(".csv", "_ready_for_judge.csv")
    df.to_csv(clean_path, index=False)
    print(f"Dataset sanitized. Ready for DeepGPU3 at: {clean_path}")

baseline_cleaning("data/baseline_results.csv")