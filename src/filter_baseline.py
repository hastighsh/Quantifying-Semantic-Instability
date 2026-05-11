import os
import sys
import pandas as pd

# Paths setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

input_path = os.path.join(PROJECT_ROOT, "data", "verified_baseline.csv")
output_path = os.path.join(PROJECT_ROOT, "data", "filtered_experimental_set.csv")


def analyze_results(df, source_name="verified_baseline.csv"):
    # Ensure boolean
    if df['is_correct'].dtype == 'object':
        df['is_correct'] = (
            df['is_correct']
            .astype(str)
            .str.upper()
            .str.strip() == 'TRUE'
        )

    total_samples = len(df)
    total_passed = df['is_correct'].sum()

    print("\n" + "=" * 60)
    print(f"VERIFICATION SURVIVAL ANALYSIS: {source_name}")
    print("=" * 60)
    print(f"Total Samples Processed: {total_samples}")
    print(f"Total Passed (Judge YES): {total_passed}")
    print(f"Overall Accuracy: {(total_passed / total_samples) * 100:.2f}%")
    print("-" * 60)

    # CWE analysis
    cwe_counts = df.groupby('cwe').size().rename('total_samples')

    cwe_passed = (
        df[df['is_correct'] == True]
        .groupby('cwe')
        .size()
        .rename('passed_samples')
    )

    analysis = pd.concat([cwe_counts, cwe_passed], axis=1)
    analysis = analysis.fillna(0).astype(int)

    analysis['survival_rate_%'] = (
        analysis['passed_samples'] / analysis['total_samples'] * 100
    ).round(2)

    analysis = analysis.sort_values(
        by='total_samples',
        ascending=False
    )

    print("SURVIVAL RATE BY CWE TYPE:")
    print(analysis.to_string())

    print("-" * 60)

    reliable = analysis[analysis['passed_samples'] >= 2]

    print(f"\nReliable CWEs (>= 2 correct): {len(reliable)}")
    print(reliable.index.tolist())

    print("=" * 60 + "\n")


def main():
    # Sanity checks
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Could not find input file at: {input_path}"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data
    df = pd.read_csv(input_path)

    # Remove accidental duplicate index column
    if 'index' in df.columns:
        df = df.drop(columns=['index'])

    # Ensure boolean
    if df['is_correct'].dtype == 'object':
        df['is_correct'] = (
            df['is_correct']
            .astype(str)
            .str.upper()
            .str.strip() == 'TRUE'
        )

    # Keep only YES
    final_set = df[df['is_correct'] == True].copy()

    # Reset index
    final_set = final_set.reset_index(drop=True)
    final_set.index.name = 'index'

    # Save filtered set
    final_set.to_csv(output_path, index=True)

    # Basic stats
    print(f"\n--- Extraction Complete ---")
    print(f"Original Samples: {len(df)}")
    print(f"High-Quality Samples Kept: {len(final_set)}")
    print(f"Discarded (Inaccurate): {len(df) - len(final_set)}")
    print(f"Clean data saved to: {output_path}")

    # Run analysis
    analyze_results(df, "verified_baseline.csv")


if __name__ == "__main__":
    main()