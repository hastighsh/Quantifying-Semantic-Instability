import os
import sys
import pandas as pd

# Paths setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "verified_baseline_complete_4gpu.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "filtered_experimental_set.csv")


def parse_boolean_column(series):
    """
    Robust parsing wrapper to accurately convert string-based judgments
    or mixed types cleanly into standard Python Booleans.
    """
    return series.astype(str).str.strip().str.upper().isin(["TRUE", "1", "YES"])


def analyze_results(df, source_name="verified_baseline_complete_4gpu.csv"):
    # Ensure standard boolean alignment
    df['is_correct'] = parse_boolean_column(df['is_correct'])

    total_samples = len(df)
    total_passed = df['is_correct'].sum()

    print("\n" + "=" * 60)
    print(f"VERIFICATION SURVIVAL ANALYSIS: {source_name}")
    print("=" * 60)
    print(f"Total Samples Processed: {total_samples}")
    print(f"Total Passed (Judge YES): {total_passed}")
    print(f"Overall Baseline Accuracy: {(total_passed / total_samples) * 100:.2f}%")
    print("-" * 60)

    # Fallback checking if CWE column is missing or named differently
    cwe_col = 'cwe' if 'cwe' in df.columns else ('CWE' if 'CWE' in df.columns else None)
    
    if cwe_col:
        # CWE analysis
        cwe_counts = df.groupby(cwe_col).size().rename('total_samples')
        cwe_passed = (
            df[df['is_correct'] == True]
            .groupby(cwe_col)
            .size()
            .rename('passed_samples')
        )

        analysis = pd.concat([cwe_counts, cwe_passed], axis=1)
        analysis = analysis.fillna(0).astype(int)
        analysis['survival_rate_%'] = (
            analysis['passed_samples'] / analysis['total_samples'] * 100
        ).round(2)
        analysis = analysis.sort_values(by='total_samples', ascending=False)

        print("SURVIVAL RATE BY CWE TYPE:")
        print(analysis.to_string())
        print("-" * 60)

        reliable = analysis[analysis['passed_samples'] >= 2]
        print(f"\nReliable CWEs (>= 2 correct): {len(reliable)}")
        print(reliable.index.tolist())
    else:
        print("[WARNING] 'cwe' column not found in dataset. Skipping per-vulnerability stats.")

    print("=" * 60 + "\n")


def main():
    # Sanity checks
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Could not find input file at: {INPUT_PATH}\n"
            f"Please double-check your pathing configuration setup."
        )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load data
    df = pd.read_csv(INPUT_PATH)

    # Normalize boolean column using robust parsing
    df['is_correct'] = parse_boolean_column(df['is_correct'])

    # Keep only high-quality baseline entries (Judge Verdict == YES)
    final_set = df[df['is_correct'] == True].copy()

    # Reset index but preserve original dataframe tracking metrics securely
    final_set = final_set.reset_index(drop=True)

    # Save filtered set to disk
    final_set.to_csv(OUTPUT_PATH, index=False)

    # Print extraction metrics summary
    print(f"\n--- Extraction Complete ---")
    print(f"Original Samples Evaluated: {len(df)}")
    print(f"High-Quality Baseline Samples Kept (YES): {len(final_set)}")
    print(f"Discarded (Inaccurate / Flawed Explanations): {len(df) - len(final_set)}")
    print(f"Clean experimental workspace saved directly to: {OUTPUT_PATH}")

    # Run the comprehensive metrics and survival engine
    analyze_results(df, os.path.basename(INPUT_PATH))


if __name__ == "__main__":
    main()
