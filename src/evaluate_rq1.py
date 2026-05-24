import os
import re
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from scipy import stats

try:
    import scikit_posthocs as sp
    HAS_POSTHOC = True
except ImportError:
    HAS_POSTHOC = False

# 1. Structural Configuration Setup
PROJECT_ROOT = os.getcwd()
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "analysis_v3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path dictionaries matching your 3-tier experimental framework
MODEL_TIERS = {
    "Low-Tier (7B)": {
        "path": os.path.join(PROJECT_ROOT, "results", "rq1_results_qwen7b_transformers.csv"),
        "color": "#4A90E2"
    },
    "Mid-Tier (14B)": {
        "path": os.path.join(PROJECT_ROOT, "results", "rq1_results_qwen14b_transformers.csv"),
        "color": "#50E3C2"
    },
    "High-Tier (32B)": {
        "path": os.path.join(PROJECT_ROOT, "results", "rq1_results_qwen32b_transformers.csv"),
        "color": "#B8E986"
    }
}


def calculate_jaccard(text1, text2):
    """Calculates keyword-based Jaccard similarity to detect categorical shifts."""
    pattern = re.compile(r'cwe-\d+|vulnerability|overflow|injection|null|pointer', re.IGNORECASE)
    words1 = set(pattern.findall(str(text1).lower()))
    words2 = set(pattern.findall(str(text2).lower()))
    
    if not words1 and not words2: 
        return 1.0 
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 1.0


def process_single_tier(file_path, st_model, device):
    """ Cleans, extracts, and computes embedding-space distances for a target file. """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing core empirical file path at: {file_path}")

    raw_df = pd.read_csv(file_path)
    
    # Filter out inference drops
    raw_df = raw_df[~raw_df['explanation'].str.contains("INFERENCE_ERROR", na=False)]
    
    df_base = raw_df[raw_df['experiment_type'] == 'baseline'].rename(
        columns={'explanation': 'base_exp', 'code_used': 'base_code'}
    )
    df_pert = raw_df[raw_df['experiment_type'] == 'perturbed'].rename(
        columns={'explanation': 'pert_exp', 'code_used': 'pert_code'}
    )
    
    df = pd.merge(
        df_base[['index', 'cwe', 'base_exp', 'base_code']], 
        df_pert[['index', 'pert_exp', 'pert_code']], 
        on='index'
    )

    # Tensor embedding logic processing assigned cleanly to active device space
    base_embs = st_model.encode(df['base_exp'].tolist(), convert_to_tensor=True, device=device)
    pert_embs = st_model.encode(df['pert_exp'].tolist(), convert_to_tensor=True, device=device)
    
    cos_sims = util.pytorch_cos_sim(base_embs, pert_embs).diagonal().tolist()
    df['semantic_variance'] = [1 - s for s in cos_sims]
    df['jaccard_instability'] = 1 - df.apply(lambda x: calculate_jaccard(x['base_exp'], x['pert_exp']), axis=1)
    df['code_len_increase'] = df['pert_code'].str.len() - df['base_code'].str.len()

    def get_pert_type(row):
        base = str(row['base_code']).lower()
        pert = str(row['pert_code']).lower()
        if 'while' in pert and 'while' not in base: return 'Structural (While)'
        if '?' in pert and '?' not in base: return 'Structural (Ternary)'
        return 'Lexical (Renaming)'
    
    df['perturbation_type'] = df.apply(get_pert_type, axis=1)
    return df


def generate_unified_reports(compiled_data, stats_records):
    """ Compiles full multi-model performance plots and LaTeX-ready data frames. """
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "serif"
    
    # FIGURE 1: Consolidated Violin Plot (Cross-Tier Distribution)
    plt.figure(figsize=(11, 6))
    palette_colors = {tier: cfg["color"] for tier, cfg in MODEL_TIERS.items()}
    
    sns.violinplot(
        x='perturbation_type', 
        y='semantic_variance', 
        hue='model_tier',
        data=compiled_data, 
        palette=palette_colors,
        split=False,
        inner="quartile"
    )
    plt.title("Semantic Instability Profiles Across Parameter Scales", fontsize=13, fontweight='bold')
    plt.xlabel("Applied Code Transformation Model", fontsize=11, fontweight='bold')
    plt.ylabel("Semantic Instability (1 - Cosine Similarity)", fontsize=11, fontweight='bold')
    plt.legend(title="Evaluated Tier Model")
    plt.savefig(os.path.join(OUTPUT_DIR, "rq1_cross_tier_violin.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # FIGURE 2: Multi-Facet Regression Lines
    g = sns.FacetGrid(compiled_data, col="model_tier", hue="model_tier", palette=palette_colors, height=5, aspect=1)
    g.map(sns.regplot, "code_len_increase", "semantic_variance", scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    g.set_axis_labels("Code Volume Delta (Chars)", "Semantic Variance")
    g.set_titles(col_template="{col_name}")
    plt.savefig(os.path.join(OUTPUT_DIR, "rq1_complexity_vs_scale_regression.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # FIGURE 3: Comparative Heatmap Matrix
    plt.figure(figsize=(14, 8))
    pivot_matrix = compiled_data.pivot_table(
        values='semantic_variance', 
        index='cwe', 
        columns='model_tier', 
        aggfunc='mean'
    ).fillna(0)
    
    # Filter matrix to only look at high-volume categories
    top_cwes = compiled_data['cwe'].value_counts().head(10).index
    pivot_matrix = pivot_matrix.loc[top_cwes]
    
    sns.heatmap(pivot_matrix, annot=True, cmap="mako", fmt=".4f", cbar_kws={'label': 'Mean Instability Metric'})
    plt.title("CWE Vulnerability Fragility Mapping Matrix Across Model Tiers", fontsize=13, fontweight='bold')
    plt.xlabel("Evaluated Parameter Scale", fontsize=11, fontweight='bold')
    plt.ylabel("CWE Categorization", fontsize=11, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "rq1_cwe_tier_instability_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("LAUNCHING MULTI-TIER COGNITIVE INSTABILITY SUITE (RQ1)")
    print("=" * 70)
    
    # Accelerate embedding operations via host hardware if active
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Assigning SentenceTransformer vector evaluation logic onto: [{device}]")
    
    st_model = SentenceTransformer(
        '/scratch/hghanesh/models/all-MiniLM-L6-v2',
        local_files_only=True
    )    
    master_frames = []
    summary_stats = []

    for tier_name, config in MODEL_TIERS.items():
        print(f">> Executing data pull on target file for: {tier_name}")
        try:
            tier_df = process_single_tier(config["path"], st_model, device)
            tier_df["model_tier"] = tier_name
            master_frames.append(tier_df)
            
            # Extract high-fidelity numbers
            n = len(tier_df)
            mean_v = tier_df['semantic_variance'].mean()
            std_v = tier_df['semantic_variance'].std()
            
            # Wilcoxon Significance Checks
            _, p_wilc = stats.wilcoxon(tier_df['semantic_variance'])
            
            summary_stats.append({
                "Model Tier": tier_name,
                "Sample Count": n,
                "Mean Instability": round(mean_v, 4),
                "StdDev": round(std_v, 4),
                "Wilcoxon p-val": f"{p_wilc:.4e}"
            })
            
        except FileNotFoundError as err:
            print(f"[SKIPPED] {tier_name} source data table not discovered. Path target: {config['path']}")

    if not master_frames:
        print("[CRITICAL ERROR] No viable data targets parsed. Execution aborted.")
        return

    # Unify dataframe blocks
    compiled_df = pd.concat(master_frames, ignore_index=True)
    
    # Generate multi-tiered graphics suite
    print(">> Rendering consolidated comparative plot sheets...")
    generate_unified_reports(compiled_df, summary_stats)
    
    # Save processed dataframe array back to disk
    compiled_df.to_csv(os.path.join(OUTPUT_DIR, "rq1_multitier_master_results.csv"), index=False)

    # Render clean structural markdown table output inside the terminal screen console
    stats_df = pd.DataFrame(summary_stats)
    
    print("\n" + "=" * 70)
    print("EMPIRICAL COMPARATIVE TABLE: MODEL ACCURACY VS SEMANTIC BRITTLENESS")
    print("=" * 70)
    print(stats_df.to_string(index=False))
    print("=" * 70)
    
    # Cross-Tier Omni-Kruskal Check (Does scale statistically matter?)
    groups = [g['semantic_variance'].values for _, g in compiled_df.groupby('model_tier')]
    if len(groups) > 1:
        h_val, p_val = stats.kruskal(*groups)
        print(f"\n[OMNIBUS METRIC] Cross-Tier Kruskal-Wallis Evaluation:")
        print(f"H-Statistic: {h_val:.4f}, Asymptotic Probability value (p-value): {p_val:.8e}")
        if p_val < 0.05:
            print(">> SUCCESSFUL EMPIRICAL PROOF: Scaling parameter count introduces a STATISTICALLY SIGNIFICANT shift in reasoning stability.")
        else:
            print(">> EXPERIMENTAL DISCOVERY: Parameter scaling did not show significant resistance against semantic layout modifications.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()