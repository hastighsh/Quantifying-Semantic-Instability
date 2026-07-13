#!/usr/bin/env python3
import os
import re
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sentence_transformers import SentenceTransformer, util


try:
    import scikit_posthocs as sp
    HAS_POSTHOC = True
except ImportError:
    HAS_POSTHOC = False


warnings.filterwarnings("ignore", category=UserWarning)


MODEL_TIERS = {
    "Low-Tier (7B)": {
        "filename": "rq1_results_qwen7b_transformers.csv",
        "color": "#4A90E2",
    },
    "Mid-Tier (14B)": {
        "filename": "rq1_results_qwen14b_transformers.csv",
        "color": "#50E3C2",
    },
    "High-Tier (32B)": {
        "filename": "rq1_results_qwen32b_transformers.csv",
        "color": "#B8E986",
    },
}

MODEL_TIER_ORDER = ["Low-Tier (7B)", "Mid-Tier (14B)", "High-Tier (32B)"]

PERTURBATION_ORDER = [
    "Lexical (Renaming)",
    "Structural (While)",
    "Structural (Ternary)",
]


# IEEE/report figure style
FIG_DPI = 600
TITLE_SIZE = 22
AXIS_LABEL_SIZE = 19
TICK_SIZE = 16
LEGEND_SIZE = 15
LEGEND_TITLE_SIZE = 16
ANNOT_SIZE = 15
FACET_TITLE_SIZE = 18


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RQ1 semantic instability across Qwen model tiers."
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root. Default: parent of this script directory.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing rq1_results_qwen*_transformers.csv files.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where analysis outputs will be saved.",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model ID or local model path.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Embedding device. Default: auto.",
    )

    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for SentenceTransformer encoding.",
    )

    parser.add_argument(
        "--only-complete-tiers",
        action="store_true",
        help="Only analyze tiers whose CSV files exist.",
    )

    return parser.parse_args()


def resolve_project_root(args):
    if args.project_root:
        return os.path.abspath(args.project_root)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(
        os.path.join(script_dir, "..", "..")
    )


def choose_device(device_arg):
    if device_arg == "cpu":
        return "cpu"

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            print("[CRITICAL ERROR] --device cuda requested, but CUDA is unavailable.")
            sys.exit(1)
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_plot_style():
    sns.set_theme(style="whitegrid", context="paper")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "legend.title_fontsize": LEGEND_TITLE_SIZE,
        "figure.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.2,
        "savefig.dpi": FIG_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def calculate_jaccard(text1, text2):
    """
    Keyword-based Jaccard instability proxy.
    This is intentionally narrow and security-oriented.
    """
    pattern = re.compile(
        r"cwe-\d+|vulnerability|overflow|underflow|injection|null|pointer|"
        r"use-after-free|double-free|buffer|bounds|integer|format|string|"
        r"race|memory|leak|dangling|validation|sanitize|authentication|authorization",
        re.IGNORECASE,
    )

    words1 = set(pattern.findall(str(text1).lower()))
    words2 = set(pattern.findall(str(text2).lower()))

    if not words1 and not words2:
        return 1.0

    union = len(words1.union(words2))
    if union == 0:
        return 1.0

    intersection = len(words1.intersection(words2))
    return intersection / union


def safe_wilcoxon(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if np.allclose(values, 0):
        return 0.0, 1.0

    try:
        stat, p_val = stats.wilcoxon(values)
        return stat, p_val
    except Exception:
        return np.nan, np.nan


def infer_perturbation_type(base_code, pert_code):
    base = str(base_code).lower()
    pert = str(pert_code).lower()

    if "while" in pert and "while" not in base:
        return "Structural (While)"

    if "?" in pert and "?" not in base:
        return "Structural (Ternary)"

    return "Lexical (Renaming)"


def read_generation_csv(file_path):
    """
    Robust CSV reader for multiline model explanations and code.
    pandas handles quoted multiline CSV fields correctly.
    """
    try:
        return pd.read_csv(file_path, dtype=str)
    except Exception as first_error:
        print(f"[WARN] Standard CSV reader failed for {file_path}: {repr(first_error)}")
        print("[WARN] Retrying with python engine...")
        return pd.read_csv(file_path, dtype=str, engine="python")


def validate_generation_columns(raw_df, file_path):
    required_cols = {
        "index",
        "cwe",
        "model_tier",
        "model_id",
        "experiment_type",
        "explanation",
        "code_used",
    }

    missing = required_cols - set(raw_df.columns)
    if missing:
        raise ValueError(
            f"File {file_path} is missing required columns: {sorted(missing)}"
        )


def process_single_tier(file_path, tier_name, st_model, device, embedding_batch_size):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    raw_df = read_generation_csv(file_path)
    validate_generation_columns(raw_df, file_path)

    raw_df = raw_df.copy()
    raw_df["experiment_type"] = raw_df["experiment_type"].astype(str).str.strip()
    raw_df["explanation"] = raw_df["explanation"].astype(str)
    raw_df["code_used"] = raw_df["code_used"].astype(str)

    raw_df = raw_df[
        ~raw_df["explanation"].str.contains("INFERENCE_ERROR", na=False)
    ].dropna(subset=["explanation", "code_used", "index"])

    raw_df["index"] = raw_df["index"].astype(int)

    df_base = raw_df[raw_df["experiment_type"] == "baseline"].rename(
        columns={
            "explanation": "base_exp",
            "code_used": "base_code",
            "model_id": "base_model_id",
        }
    )

    df_pert = raw_df[raw_df["experiment_type"] == "perturbed"].rename(
        columns={
            "explanation": "pert_exp",
            "code_used": "pert_code",
            "model_id": "pert_model_id",
        }
    )

    df = pd.merge(
        df_base[["index", "cwe", "base_exp", "base_code", "base_model_id"]],
        df_pert[["index", "pert_exp", "pert_code", "pert_model_id"]],
        on="index",
        how="inner",
    )

    if df.empty:
        raise ValueError(f"Merged baseline/perturbed table is empty for {file_path}")

    print(f"   Loaded pairs: {len(df)}")

    base_texts = df["base_exp"].fillna("").astype(str).tolist()
    pert_texts = df["pert_exp"].fillna("").astype(str).tolist()

    with torch.no_grad():
        base_embs = st_model.encode(
            base_texts,
            convert_to_tensor=True,
            device=device,
            batch_size=embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        pert_embs = st_model.encode(
            pert_texts,
            convert_to_tensor=True,
            device=device,
            batch_size=embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        cos_sims = util.pytorch_cos_sim(base_embs, pert_embs).diagonal()
        cos_sims = cos_sims.detach().cpu().numpy()

    df["cosine_similarity"] = cos_sims
    df["semantic_variance"] = 1.0 - df["cosine_similarity"]

    df["jaccard_similarity"] = df.apply(
        lambda x: calculate_jaccard(x["base_exp"], x["pert_exp"]),
        axis=1,
    )
    df["jaccard_instability"] = 1.0 - df["jaccard_similarity"]

    df["code_len_increase"] = (
        df["pert_code"].astype(str).str.len() - df["base_code"].astype(str).str.len()
    )

    df["perturbation_type"] = df.apply(
        lambda x: infer_perturbation_type(x["base_code"], x["pert_code"]),
        axis=1,
    )

    df["model_tier"] = tier_name

    return df


def save_figure(base_path):
    """
    Save both PNG and PDF.
    base_path should not include extension.
    """
    png_path = base_path + ".png"
    pdf_path = base_path + ".pdf"

    plt.tight_layout()
    plt.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"   Saved figure: {png_path}")
    print(f"   Saved figure: {pdf_path}")


def polish_axes(ax):
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.2)
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.title.set_size(TITLE_SIZE)

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")


def generate_unified_reports(compiled_data, output_dir):
    configure_plot_style()

    palette_colors = {
        tier: cfg["color"]
        for tier, cfg in MODEL_TIERS.items()
        if tier in set(compiled_data["model_tier"])
    }

    compiled_data = compiled_data.copy()
    compiled_data["perturbation_type"] = pd.Categorical(
        compiled_data["perturbation_type"],
        categories=[p for p in PERTURBATION_ORDER if p in set(compiled_data["perturbation_type"])],
        ordered=True,
    )

    # ------------------------------------------------------------------
    # Figure 1: Cross-tier violin plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(15, 8))
    ax = sns.violinplot(
        x="perturbation_type",
        y="semantic_variance",
        hue="model_tier",
        data=compiled_data,
        order=[p for p in PERTURBATION_ORDER if p in set(compiled_data["perturbation_type"])],
        hue_order=MODEL_TIER_ORDER,
        palette=palette_colors,
        split=False,
        inner="quartile",
        cut=0,
        linewidth=1.4,
    )

    ax.set_title("Semantic Instability Profiles Across Model Scales", pad=18)
    ax.set_xlabel("Applied Code Transformation", labelpad=14)
    ax.set_ylabel("Semantic Instability (1 - Cosine Similarity)", labelpad=14)
    ax.tick_params(axis="x", rotation=12)
    polish_axes(ax)

    legend = ax.legend(
        title="Model Tier",
        frameon=True,
        loc="upper right",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
    )
    legend.get_title().set_fontweight("bold")

    save_figure(os.path.join(output_dir, "rq1_cross_tier_violin_ieee"))

    # ------------------------------------------------------------------
    # Figure 2: Complexity vs instability regression
    # ------------------------------------------------------------------
    g = sns.FacetGrid(
        compiled_data,
        col="model_tier",
        hue="model_tier",
        col_order=MODEL_TIER_ORDER,
        hue_order=MODEL_TIER_ORDER,
        palette=palette_colors,
        height=5.6,
        aspect=1.15,
        sharex=False,
        sharey=True,
        despine=False,
    )

    g.map_dataframe(
        sns.regplot,
        x="code_len_increase",
        y="semantic_variance",
        scatter_kws={"alpha": 0.45, "s": 45, "edgecolor": "black"},
        line_kws={"color": "black", "linewidth": 2.4},
    )

    g.set_axis_labels("Code Volume Delta (Characters)", "Semantic Instability")
    g.set_titles(col_template="{col_name}", size=FACET_TITLE_SIZE, weight="bold")

    for ax in g.axes.flat:
        polish_axes(ax)
        ax.grid(True, linestyle="--", alpha=0.45)

    g.fig.set_size_inches(17, 6.4)
    g.fig.suptitle(
        "Code Complexity Delta vs. Semantic Instability",
        y=1.04,
        fontsize=TITLE_SIZE + 1,
        fontweight="bold",
    )

    regression_png = os.path.join(output_dir, "rq1_complexity_vs_scale_regression_ieee.png")
    regression_pdf = os.path.join(output_dir, "rq1_complexity_vs_scale_regression_ieee.pdf")
    g.savefig(regression_png, dpi=FIG_DPI, bbox_inches="tight")
    g.savefig(regression_pdf, bbox_inches="tight")
    plt.close()

    print(f"   Saved figure: {regression_png}")
    print(f"   Saved figure: {regression_pdf}")

    # ------------------------------------------------------------------
    # Figure 3: CWE heatmap
    # ------------------------------------------------------------------
    top_cwes = compiled_data["cwe"].value_counts().head(10).index.tolist()

    pivot_matrix = compiled_data.pivot_table(
        values="semantic_variance",
        index="cwe",
        columns="model_tier",
        aggfunc="mean",
        observed=False,
    )

    pivot_matrix = pivot_matrix.loc[
        [cwe for cwe in top_cwes if cwe in pivot_matrix.index]
    ]

    ordered_columns = [tier for tier in MODEL_TIER_ORDER if tier in pivot_matrix.columns]
    pivot_matrix = pivot_matrix[ordered_columns]

    plt.figure(figsize=(12.8, 8.6))
    ax = sns.heatmap(
        pivot_matrix,
        annot=True,
        cmap="mako",
        fmt=".4f",
        linewidths=0.8,
        linecolor="white",
        annot_kws={"fontsize": ANNOT_SIZE, "fontweight": "bold"},
        cbar_kws={"label": "Mean Instability", "shrink": 0.86},
    )

    ax.set_title("CWE Fragility Matrix Across Model Tiers", pad=18)
    ax.set_xlabel("Evaluated Parameter Scale", labelpad=14)
    ax.set_ylabel("CWE Category", labelpad=14)
    ax.tick_params(axis="x", rotation=18)
    ax.tick_params(axis="y", rotation=0)
    polish_axes(ax)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.set_label("Mean Instability", fontsize=AXIS_LABEL_SIZE, fontweight="bold")

    save_figure(os.path.join(output_dir, "rq1_cwe_tier_instability_heatmap_ieee"))

    # ------------------------------------------------------------------
    # Figure 4: Mean instability bar plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(10.5, 6.8))
    ax = sns.barplot(
        data=compiled_data,
        x="model_tier",
        y="semantic_variance",
        hue="model_tier",
        order=MODEL_TIER_ORDER,
        hue_order=MODEL_TIER_ORDER,
        palette=palette_colors,
        errorbar=("ci", 95),
        capsize=0.12,
        err_kws={"linewidth": 2.0},
        legend=False,
        edgecolor="black",
        linewidth=1.2,
    )

    ax.set_title("Mean Semantic Instability by Model Tier", pad=18)
    ax.set_xlabel("Model Tier", labelpad=14)
    ax.set_ylabel("Mean Semantic Instability", labelpad=14)
    ax.tick_params(axis="x", rotation=10)
    polish_axes(ax)

    save_figure(os.path.join(output_dir, "rq1_mean_instability_by_tier_ieee"))


def generate_summary_stats(compiled_df):
    summary_records = []

    for tier_name, tier_df in compiled_df.groupby("model_tier", sort=False, observed=False):
        n = len(tier_df)
        mean_v = tier_df["semantic_variance"].mean()
        median_v = tier_df["semantic_variance"].median()
        std_v = tier_df["semantic_variance"].std()
        min_v = tier_df["semantic_variance"].min()
        max_v = tier_df["semantic_variance"].max()

        wilc_stat, wilc_p = safe_wilcoxon(tier_df["semantic_variance"].values)

        summary_records.append(
            {
                "Model Tier": tier_name,
                "Sample Count": n,
                "Mean Instability": round(mean_v, 6),
                "Median Instability": round(median_v, 6),
                "StdDev": round(std_v, 6),
                "Min": round(min_v, 6),
                "Max": round(max_v, 6),
                "Wilcoxon Statistic": wilc_stat,
                "Wilcoxon p-val": wilc_p,
            }
        )

    return pd.DataFrame(summary_records)


def run_cross_tier_tests(compiled_df, output_dir):
    test_lines = []

    groups = [
        group["semantic_variance"].dropna().values
        for _, group in compiled_df.groupby("model_tier", sort=False, observed=False)
    ]

    names = [
        name
        for name, _ in compiled_df.groupby("model_tier", sort=False, observed=False)
    ]

    if len(groups) <= 1:
        test_lines.append("Only one model tier available. Cross-tier tests skipped.")
    else:
        h_val, p_val = stats.kruskal(*groups)
        test_lines.append("Cross-Tier Kruskal-Wallis Test")
        test_lines.append(f"Groups: {names}")
        test_lines.append(f"H-statistic: {h_val:.6f}")
        test_lines.append(f"p-value: {p_val:.10e}")

        if p_val < 0.05:
            test_lines.append(
                "Conclusion: Parameter scale is associated with a statistically significant shift in semantic instability."
            )
        else:
            test_lines.append(
                "Conclusion: No statistically significant cross-tier shift detected at alpha=0.05."
            )

        if HAS_POSTHOC:
            posthoc = sp.posthoc_dunn(
                compiled_df,
                val_col="semantic_variance",
                group_col="model_tier",
                p_adjust="holm",
            )

            ordered_labels = [tier for tier in MODEL_TIER_ORDER if tier in posthoc.index]
            posthoc = posthoc.loc[ordered_labels, ordered_labels]

            posthoc_path = os.path.join(output_dir, "rq1_posthoc_dunn_holm.csv")
            posthoc.to_csv(posthoc_path)
            test_lines.append(f"Posthoc Dunn-Holm matrix saved to: {posthoc_path}")
        else:
            test_lines.append(
                "scikit-posthocs is not installed. Posthoc Dunn-Holm test skipped."
            )

    output_path = os.path.join(output_dir, "rq1_cross_tier_stats.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + "\n")

    print("\n".join(test_lines))
    print(f"Saved statistical test report: {output_path}")


def main():
    args = parse_args()

    project_root = resolve_project_root(args)
    results_dir = args.results_dir or os.path.join(project_root, "results")
    output_dir = args.output_dir or os.path.join(results_dir, "analysis_v3_ieee")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 78)
    print("LAUNCHING MULTI-TIER SEMANTIC INSTABILITY EVALUATION SUITE (RQ1)")
    print("=" * 78)
    print(f"Project root     : {project_root}")
    print(f"Results dir      : {results_dir}")
    print(f"Output dir       : {output_dir}")
    print(f"Embedding model  : {args.embedding_model}")

    device = choose_device(args.device)
    print(f"Embedding device : {device}")
    print("=" * 78)

    try:
        st_model = SentenceTransformer(args.embedding_model, device=device)
    except TypeError:
        st_model = SentenceTransformer(args.embedding_model)
        st_model = st_model.to(device)

    master_frames = []

    for tier_name, config in MODEL_TIERS.items():
        file_path = os.path.join(results_dir, config["filename"])

        print(f"\n>> Processing tier: {tier_name}")
        print(f"   Source file: {file_path}")

        try:
            tier_df = process_single_tier(
                file_path=file_path,
                tier_name=tier_name,
                st_model=st_model,
                device=device,
                embedding_batch_size=args.embedding_batch_size,
            )
            master_frames.append(tier_df)

        except FileNotFoundError:
            print(f"   [SKIPPED] File not found: {file_path}")
            if not args.only_complete_tiers:
                continue

        except Exception as e:
            print(f"   [ERROR] Failed processing {tier_name}: {repr(e)}")
            continue

    if not master_frames:
        print("\n[NOTICE] No valid inference result tables were available.")
        print("Run inference first, then run this evaluation script.\n")
        return

    compiled_df = pd.concat(master_frames, ignore_index=True)

    compiled_df["model_tier"] = pd.Categorical(
        compiled_df["model_tier"],
        categories=MODEL_TIER_ORDER,
        ordered=True,
    )

    compiled_df = compiled_df.sort_values(["model_tier", "index"]).reset_index(drop=True)

    master_path = os.path.join(output_dir, "rq1_multitier_master_results.csv")
    compiled_df.to_csv(master_path, index=False)
    print(f"\nSaved master dataframe: {master_path}")

    stats_df = generate_summary_stats(compiled_df)
    stats_path = os.path.join(output_dir, "rq1_summary_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    print("\n" + "=" * 78)
    print("EMPIRICAL COMPARATIVE TABLE: SEMANTIC BRITTLENESS")
    print("=" * 78)
    print(stats_df.to_string(index=False))
    print("=" * 78)
    print(f"Saved summary stats: {stats_path}")

    print("\n>> Rendering IEEE-ready figures...")
    generate_unified_reports(compiled_df, output_dir)

    print("\n>> Running cross-tier statistical tests...")
    run_cross_tier_tests(compiled_df, output_dir)

    print("\n[SUCCESS] RQ1 analysis complete.")
    print(f"[SUCCESS] Outputs saved under: {output_dir}\n")


if __name__ == "__main__":
    main()
