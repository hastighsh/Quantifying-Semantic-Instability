import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
from sentence_transformers import SentenceTransformer, util
from scipy.stats import kruskal, mannwhitneyu

try:
    import scikit_posthocs as sp
    HAS_POSTHOC = True
except ImportError:
    HAS_POSTHOC = False

# PATH SETUP
PROJECT_ROOT = os.getcwd()
INPUT_FILE = os.path.join(PROJECT_ROOT, "results", "rq1_results_qwen14b_transformers.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "analysis_v3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_jaccard(text1, text2):
    """Calculates keyword-based Jaccard similarity to detect categorical shifts."""
    pattern = re.compile(r'cwe-\d+|vulnerability|overflow|injection|null|pointer', re.IGNORECASE)
    words1 = set(pattern.findall(str(text1).lower()))
    words2 = set(pattern.findall(str(text2).lower()))
    
    if not words1 and not words2: return 1.0 
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 1.0

def analyze_rq1():
    print("--- 1. Data Preparation & Cleaning ---")
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: File not found at {INPUT_FILE}")
        return

    raw_df = pd.read_csv(INPUT_FILE)
    
    # Filter out inference failures
    raw_df = raw_df[~raw_df['explanation'].str.contains("INFERENCE_ERROR", na=False)]
    
    # Pivot to compare Baseline vs Perturbed
    df_base = raw_df[raw_df['experiment_type'] == 'baseline'].rename(
        columns={'explanation': 'base_exp', 'code_used': 'base_code'}
    )
    df_pert = raw_df[raw_df['experiment_type'] == 'perturbed'].rename(
        columns={'explanation': 'pert_exp', 'code_used': 'pert_code'}
    )
    
    # Merge on 'index' - keep 'cwe' from the baseline side
    df = pd.merge(df_base[['index', 'cwe', 'base_exp', 'base_code']], 
                  df_pert[['index', 'pert_exp', 'pert_code']], on='index')

    print("--- 2. Computing Multi-Modal Metrics ---")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    base_embs = st_model.encode(df['base_exp'].tolist(), convert_to_tensor=True)
    pert_embs = st_model.encode(df['pert_exp'].tolist(), convert_to_tensor=True)
    
    cos_sims = util.pytorch_cos_sim(base_embs, pert_embs).diagonal().tolist()
    df['semantic_variance'] = [1 - s for s in cos_sims]

    # Jaccard for Keyword/CWE Shifts
    df['jaccard_instability'] = 1 - df.apply(lambda x: calculate_jaccard(x['base_exp'], x['pert_exp']), axis=1)

    # Proxy for Code Complexity Increase
    df['code_len_increase'] = df['pert_code'].str.len() - df['base_code'].str.len()

    # Categorize Perturbations
    def get_pert_type(row):
        base = str(row['base_code']).lower()
        pert = str(row['pert_code']).lower()
        if 'while' in pert and 'while' not in base: return 'Structural (While)'
        if '?' in pert and '?' not in base: return 'Structural (Ternary)'
        return 'Lexical (Renaming)'
    
    df['perturbation_type'] = df.apply(get_pert_type, axis=1)

    print("--- 3. Statistical Analysis ---")
    groups_data = [group['semantic_variance'].values for _, group in df.groupby('perturbation_type')]
    stat, p_value = kruskal(*groups_data)
    
    posthoc_results = "N/A"
    if p_value < 0.05 and HAS_POSTHOC:
        posthoc_results = sp.posthoc_dunn(df, val_col='semantic_variance', group_col='perturbation_type', p_adjust='bonferroni')

    # CWE Sensitivity Summary (n >= 3)
    cwe_summary = df.groupby('cwe')['semantic_variance'].agg(['mean', 'std', 'count'])
    cwe_summary = cwe_summary[cwe_summary['count'] >= 3].sort_values(by='mean', ascending=False)

    print("--- 4. Visualization Suite (Individual Plots) ---")
    sns.set_theme(style="whitegrid")

    # 1. Boxenplot: Instability Distribution
    plt.figure(figsize=(10, 6))
    sns.boxenplot(x='perturbation_type', y='semantic_variance', data=df, hue='perturbation_type', palette="Set2", legend=False)
    plt.title(f"Semantic Instability by Transformation (p={p_value:.4f})", fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "1_instability_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. CWE Barplot: Fragility Ranking
    plt.figure(figsize=(8, 6))
    
    # Improved Font Handling for Linux Servers
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif", "serif"]
    
    top_cwe = cwe_summary.head(10)
    global_mean = df['semantic_variance'].mean()

    ax = sns.barplot(
        x=top_cwe['mean'], 
        y=top_cwe.index, 
        hue=top_cwe.index, 
        palette="flare", 
        legend=False
    )

    # Styling Axis and Spines for IEEE
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.tick_params(axis='both', colors='#333333', labelsize=10)
    
    # Extend X-limit slightly to give room for the box
    plt.xlim(0, 0.6)

    # Global Mean Variance: Square Box with Stacked Text
    # \n creates the newline for the number
    plt.xlim(0, 0.6)

    plt.text(
        0.88, 0.08,  # Changed from 0.95 to 0.88 to move it left
        f'Global Mean\nVariance:\n{global_mean:.4f}', 
        transform=ax.transAxes, 
        fontsize=10,
        fontweight='bold',
        verticalalignment='bottom', 
        horizontalalignment='center', 
        bbox=dict(
            facecolor='white', 
            alpha=1.0, 
            edgecolor='black', 
            boxstyle='square,pad=0.8', 
            linewidth=1
        )
    )

    # Bold labels
    plt.title("CWE Sensitivity Ranking (Mean Variance)", fontsize=12, fontweight='bold')
    plt.xlabel("Semantic Instability (1 - Cosine Similarity)", fontsize=11, fontweight='bold')
    plt.ylabel("CWE Identifier", fontsize=11, fontweight='bold')

    plt.savefig(os.path.join(OUTPUT_DIR, "2_cwe_fragility.png"), dpi=600, bbox_inches='tight')
    plt.close()

    # 3. Regression Plot: Complexity vs Instability
    plt.figure(figsize=(10, 6))
    
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif", "serif"]

    ax = sns.regplot(
        x='code_len_increase', 
        y='semantic_variance', 
        data=df, 
        scatter_kws={'alpha':0.4, 'color':'teal'}, 
        line_kws={'color':'red'}
    )

    plt.title("Impact of Structural Complexity on Instability", fontsize=14, fontweight='bold')
    plt.xlabel("Code Volume Delta (Characters)", fontsize=12, fontweight='bold')
    plt.ylabel("Semantic Variance", fontsize=12, fontweight='bold')

    plt.savefig(os.path.join(OUTPUT_DIR, "3_complexity_impact_ieee.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Heatmap: Categorical Drift
    plt.figure(figsize=(12, 8))
    pivot_table = df.pivot_table(values='jaccard_instability', index='cwe', columns='perturbation_type', aggfunc='mean')
    pivot_table = pivot_table.loc[top_cwe.index] # Align with top CWEs
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Keyword Shift Score'})
    plt.title("Categorical Drift Matrix (Jaccard Instability)", fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "4_drift_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Export and Logging
    df.to_csv(os.path.join(OUTPUT_DIR, "rq1_refined_results.csv"), index=False)
    
    print("\n" + "="*50)
    print("RQ1 FINAL STATISTICAL REPORT")
    print("="*50)
    print(f"Total Samples Analyzed: {len(df)}")
    print(f"Global Mean Semantic Variance: {df['semantic_variance'].mean():.4f}")
    print(f"Kruskal-Wallis p-value: {p_value:.8f}")
    
    if p_value < 0.05:
        print(">> Result: Transformation type has a SIGNIFICANT impact.")
        if HAS_POSTHOC:
            print("\nPost-hoc Dunn's Test (Significance Matrix):")
            print(posthoc_results)
    else:
        print(">> Result: No statistically significant difference between transformation types.")
    
    print("-" * 50)
    print("MOST SEMANTICALLY FRAGILE CWEs:")
    print(cwe_summary[['mean', 'count']].head(5))
    print("="*50)

if __name__ == "__main__":
    analyze_rq1()