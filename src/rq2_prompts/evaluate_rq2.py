#!/usr/bin/env python3
import argparse
import os
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from sentence_transformers import SentenceTransformer, util

try:
    import scikit_posthocs as sp
    HAS_POSTHOC = True
except ImportError:
    HAS_POSTHOC = False

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_TIERS = {
    "7B": {"label": "Low-Tier (7B)", "color": "#4A90E2"},
    "14B": {"label": "Mid-Tier (14B)", "color": "#50E3C2"},
    "32B": {"label": "High-Tier (32B)", "color": "#B8E986"},
}

MODEL_SIZE_ORDER = ["7B", "14B", "32B"]
MODEL_LABEL_ORDER = [
    "Low-Tier (7B)",
    "Mid-Tier (14B)",
    "High-Tier (32B)",
]

PERSONA_ORDER = ["naive", "formal", "expert"]
PERSONA_LABELS = {
    "naive": "Naive",
    "formal": "Formal",
    "expert": "Expert",
}
PERSONA_LABEL_ORDER = ["Naive", "Formal", "Expert"]
PERSONA_COLORS = {
    "naive": "#4A90E2",
    "formal": "#50E3C2",
    "expert": "#B8E986",
}

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
        description="RQ2 Multi-Tier Multi-Persona Consolidated Analyzer"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root. Default: parent directory of this script.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing rq2_*_results_qwen*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for RQ2 analysis outputs.",
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
        default=16,
        help="Batch size for SentenceTransformer encoding.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=MODEL_SIZE_ORDER,
        choices=MODEL_SIZE_ORDER,
        help="Model sizes to include. Default: 7B 14B 32B.",
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        default=PERSONA_ORDER,
        choices=PERSONA_ORDER,
        help="Personas to include. Default: naive formal expert.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected RQ2 result file is missing.",
    )
    return parser.parse_args()


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


def save_figure(base_path):
    png_path = base_path + ".png"
    pdf_path = base_path + ".pdf"

    plt.tight_layout()
    plt.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {png_path}", flush=True)
    print(f"Saved figure: {pdf_path}", flush=True)


def polish_axes(ax):
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.2)

    if ax.get_xlabel():
        ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
        ax.xaxis.label.set_fontweight("bold")

    if ax.get_ylabel():
        ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
        ax.yaxis.label.set_fontweight("bold")

    if ax.get_title():
        ax.title.set_size(TITLE_SIZE)
        ax.title.set_fontweight("bold")

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")


def resolve_project_root(args):
    if args.project_root:
        return os.path.abspath(args.project_root)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, ".."))


def choose_device(device_arg):
    if device_arg == "cpu":
        return "cpu"

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            print("[CRITICAL ERROR] --device cuda requested, but CUDA is unavailable.")
            sys.exit(1)
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


def clean_explanation(text):
    """Extract the main analysis section for embedding comparison."""
    if not isinstance(text, str) or len(text.strip()) < 5:
        return "N/A"

    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

    analysis_match = re.search(
        r"ANALYSIS:\s*(.*?)(?=IMPACT:|VULNERABILITY:|RECOMMENDATION:|MITIGATION:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if analysis_match:
        content = analysis_match.group(1).strip()
    else:
        content = re.sub(
            r"^\s*VULNERABILITY:\s*.*?(?:\n|$)",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    return " ".join(content.split()[:120])


def detect_security_attributes(text):
    """Detect source, sink, and trigger terminology."""
    if not isinstance(text, str):
        text = ""

    text_lower = text.lower()

    patterns = {
        "source_identified": (
            r"\b("
            r"input|user input|attacker-controlled|external input|parameter|argument|argv|argc|"
            r"recv|read|scanf|gets|file|socket|network|request|payload|data from user|"
            r"untrusted|tainted"
            r")\b"
        ),
        "sink_identified": (
            r"\b("
            r"memcpy|strcpy|strncpy|sprintf|snprintf|strcat|malloc|calloc|realloc|free|"
            r"buffer|array|pointer|index|heap|stack|memory write|memory read|dereference|"
            r"copy|allocation|sink|write|read"
            r")\b"
        ),
        "trigger_identified": (
            r"\b("
            r"overflow|out-of-bounds|bounds|boundary|length check|missing check|"
            r"validation|unchecked|insufficient validation|null dereference|null pointer|"
            r"use-after-free|double free|integer overflow|wraparound|race condition|"
            r"format string|leak|corruption|trigger|exceeds|underflow|dangling"
            r")\b"
        ),
    }

    detected = {
        key: bool(re.search(pattern, text_lower))
        for key, pattern in patterns.items()
    }

    detected["completeness_score"] = (
        int(detected["source_identified"])
        + int(detected["sink_identified"])
        + int(detected["trigger_identified"])
    )

    return pd.Series(detected)


def extract_key_security_insights(row):
    insights = []

    if row.get("source_identified", False):
        insights.append("Source")
    if row.get("sink_identified", False):
        insights.append("Sink")
    if row.get("trigger_identified", False):
        insights.append("Trigger")

    return ", ".join(insights) if insights else "None"


def calculate_schema_adherence(text):
    if not isinstance(text, str):
        return 0.0

    required_sections = ["VULNERABILITY", "ANALYSIS", "IMPACT"]
    text_upper = text.upper()
    hits = sum(section in text_upper for section in required_sections)

    return hits / len(required_sections)


def calculate_instruction_stability(text):
    if not isinstance(text, str):
        return 0.0

    word_count = len(text.split())
    schema_score = calculate_schema_adherence(text)
    verbosity_penalty = min(word_count / 700.0, 1.0)

    stability = (0.80 * schema_score) + (0.20 * (1.0 - verbosity_penalty))
    return float(max(0.0, min(1.0, stability)))


def calculate_information_density(similarity, word_count):
    if word_count <= 0:
        return 0.0

    return float(similarity / np.log1p(word_count))


def normalize_bool_series(series):
    if series.dtype == bool:
        return series.astype(bool)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
        })
        .fillna(False)
        .astype(bool)
    )


def find_truth_column(truth_df):
    candidates = [
        "truth_description",
        "ground_truth",
        "ground_truth_explanation",
        "reference_explanation",
        "explanation",
        "description",
    ]

    for column in candidates:
        if column in truth_df.columns:
            return column

    return None


def load_ground_truth(path):
    if not os.path.exists(path):
        print(f"[CRITICAL ERROR] Ground-truth dataset not found: {path}")
        sys.exit(1)

    truth_df = pd.read_csv(path, dtype=str)

    if "index" not in truth_df.columns:
        if "original_index" in truth_df.columns:
            truth_df["index"] = truth_df["original_index"]
        else:
            truth_df = truth_df.reset_index().rename(columns={"index": "index"})

    if "cwe" not in truth_df.columns:
        truth_df["cwe"] = "Unknown"

    truth_col = find_truth_column(truth_df)

    if truth_col is None:
        print("[WARN] No explicit ground-truth explanation column found.")
        print("[WARN] Falling back to using the code column as reference text.")
        print(
            "[WARN] SBERT similarity will measure explanation-to-code "
            "relatedness, not explanation accuracy."
        )

        if "code" not in truth_df.columns:
            print(
                "[CRITICAL ERROR] No ground-truth text column "
                "and no code column available."
            )
            sys.exit(1)

        truth_col = "code"

    truth_df["index"] = truth_df["index"].astype(int)
    return truth_df, truth_col


def expected_result_path(results_dir, persona, size):
    return os.path.join(
        results_dir,
        f"rq2_{persona}_results_qwen{size.lower()}.csv",
    )


def load_rq2_file(path, size, persona):
    df = pd.read_csv(path, dtype=str)

    required = {"generated_explanation"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    if "index" not in df.columns:
        if "original_index" in df.columns:
            df["index"] = df["original_index"]
        else:
            raise ValueError(f"{path} has neither index nor original_index column.")

    if "original_index" not in df.columns:
        df["original_index"] = df["index"]

    if "cwe" not in df.columns:
        df["cwe"] = "Unknown"

    df["index"] = df["index"].astype(int)
    df["original_index"] = df["original_index"].astype(int)
    df["model_size"] = size
    df["model_tier"] = MODEL_TIERS[size]["label"]
    df["persona"] = persona
    df["persona_label"] = PERSONA_LABELS[persona]

    if "adheres_to_schema" in df.columns:
        df["adheres_to_schema_raw"] = normalize_bool_series(
            df["adheres_to_schema"]
        )
    else:
        df["adheres_to_schema_raw"] = False

    if "instructional_drift" in df.columns:
        df["instructional_drift_raw"] = normalize_bool_series(
            df["instructional_drift"]
        )
    else:
        df["instructional_drift_raw"] = False

    return df


def compute_embeddings(st_model, device, texts_a, texts_b, batch_size):
    with torch.no_grad():
        emb_a = st_model.encode(
            texts_a,
            convert_to_tensor=True,
            device=device,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        emb_b = st_model.encode(
            texts_b,
            convert_to_tensor=True,
            device=device,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        similarities = util.pytorch_cos_sim(emb_a, emb_b).diagonal()
        return similarities.detach().cpu().numpy()


def build_analysis_dataframe(
    args,
    results_dir,
    truth_df,
    truth_col,
    st_model,
    device,
):
    frames = []

    for size in args.sizes:
        for persona in args.personas:
            path = expected_result_path(results_dir, persona, size)

            print(f"\n>> Loading RQ2 result: size={size}, persona={persona}")
            print(f"   Path: {path}")

            if not os.path.exists(path):
                message = f"[MISSING] {path}"

                if args.strict:
                    print(f"[CRITICAL ERROR] {message}")
                    sys.exit(1)

                print(message)
                continue

            try:
                df = load_rq2_file(path, size, persona)
            except Exception as error:
                print(f"[ERROR] Failed reading {path}: {repr(error)}")

                if args.strict:
                    sys.exit(1)

                continue

            merged = pd.merge(
                df,
                truth_df[["index", "cwe", truth_col]].rename(
                    columns={
                        "cwe": "truth_cwe",
                        truth_col: "truth_text",
                    }
                ),
                on="index",
                how="inner",
            )

            if merged.empty:
                print(f"[WARN] Merge produced zero rows for {path}")
                continue

            merged["generated_explanation"] = (
                merged["generated_explanation"].fillna("").astype(str)
            )
            merged["truth_text"] = merged["truth_text"].fillna("").astype(str)
            merged["cleaned_generation"] = (
                merged["generated_explanation"].apply(clean_explanation)
            )
            merged["word_count"] = merged["generated_explanation"].apply(
                lambda value: len(str(value).split())
            )

            attribute_df = merged["generated_explanation"].apply(
                detect_security_attributes
            )
            merged = pd.concat([merged, attribute_df], axis=1)

            merged["key_security_insights"] = merged.apply(
                extract_key_security_insights,
                axis=1,
            )
            merged["schema_adherence_score"] = (
                merged["generated_explanation"].apply(
                    calculate_schema_adherence
                )
            )
            merged["instruction_stability_score"] = (
                merged["generated_explanation"].apply(
                    calculate_instruction_stability
                )
            )

            print(f"   Rows merged: {len(merged)}")
            print("   Encoding SBERT similarities...")

            merged["sbert_similarity"] = compute_embeddings(
                st_model=st_model,
                device=device,
                texts_a=merged["truth_text"].tolist(),
                texts_b=merged["cleaned_generation"].tolist(),
                batch_size=args.embedding_batch_size,
            )

            merged["information_density"] = merged.apply(
                lambda row: calculate_information_density(
                    row["sbert_similarity"],
                    row["word_count"],
                ),
                axis=1,
            )

            frames.append(merged)

    if not frames:
        print("[CRITICAL ERROR] No RQ2 result files were loaded.")
        sys.exit(1)

    full_df = pd.concat(frames, ignore_index=True)

    full_df["model_tier"] = pd.Categorical(
        full_df["model_tier"],
        categories=MODEL_LABEL_ORDER,
        ordered=True,
    )
    full_df["model_size"] = pd.Categorical(
        full_df["model_size"],
        categories=MODEL_SIZE_ORDER,
        ordered=True,
    )
    full_df["persona"] = pd.Categorical(
        full_df["persona"],
        categories=PERSONA_ORDER,
        ordered=True,
    )
    full_df["persona_label"] = pd.Categorical(
        full_df["persona_label"],
        categories=PERSONA_LABEL_ORDER,
        ordered=True,
    )

    return full_df.sort_values(
        ["model_size", "persona", "index"]
    ).reset_index(drop=True)


def make_summary_tables(full_df, output_dir):
    summary = (
        full_df.groupby(
            ["model_tier", "persona"],
            observed=False,
        )
        .agg(
            sample_count=("index", "count"),
            mean_sbert_similarity=("sbert_similarity", "mean"),
            median_sbert_similarity=("sbert_similarity", "median"),
            std_sbert_similarity=("sbert_similarity", "std"),
            mean_word_count=("word_count", "mean"),
            mean_completeness=("completeness_score", "mean"),
            source_rate=("source_identified", "mean"),
            sink_rate=("sink_identified", "mean"),
            trigger_rate=("trigger_identified", "mean"),
            mean_information_density=("information_density", "mean"),
            mean_schema_adherence=("schema_adherence_score", "mean"),
            raw_schema_adherence_rate=("adheres_to_schema_raw", "mean"),
            raw_instructional_drift_rate=("instructional_drift_raw", "mean"),
            instruction_stability=("instruction_stability_score", "mean"),
        )
        .reset_index()
    )

    summary_path = os.path.join(
        output_dir,
        "rq2_multitier_persona_summary.csv",
    )
    summary.to_csv(summary_path, index=False)

    pivot_similarity = summary.pivot(
        index="persona",
        columns="model_tier",
        values="mean_sbert_similarity",
    )

    pivot_similarity = pivot_similarity.loc[
        [
            persona
            for persona in PERSONA_ORDER
            if persona in pivot_similarity.index
        ],
        [
            tier
            for tier in MODEL_LABEL_ORDER
            if tier in pivot_similarity.columns
        ],
    ]

    pivot_path = os.path.join(
        output_dir,
        "rq2_similarity_persona_by_tier.csv",
    )
    pivot_similarity.to_csv(pivot_path)

    print("\n" + "=" * 90)
    print("RQ2 MULTI-TIER PERSONA SUMMARY")
    print("=" * 90)
    print(summary.to_string(index=False))
    print("=" * 90)
    print(f"Saved summary: {summary_path}")
    print(f"Saved similarity pivot: {pivot_path}")

    return summary, pivot_similarity


def run_statistical_tests(full_df, output_dir):
    lines = [
        "RQ2 Statistical Evaluation",
        "=" * 80,
        "\nPersona Effect Within Each Model Tier",
        "-" * 80,
    ]

    for tier in MODEL_LABEL_ORDER:
        subset = full_df[full_df["model_tier"] == tier]

        groups = [
            subset[subset["persona"] == persona]["sbert_similarity"]
            .dropna()
            .values
            for persona in PERSONA_ORDER
            if not subset[subset["persona"] == persona].empty
        ]

        names = [
            persona
            for persona in PERSONA_ORDER
            if not subset[subset["persona"] == persona].empty
        ]

        if len(groups) < 2:
            lines.append(f"{tier}: skipped; fewer than two persona groups.")
            continue

        h_value, p_value = stats.kruskal(*groups)

        lines.append(
            f"{tier}: Kruskal-Wallis H={h_value:.6f}, "
            f"p={p_value:.10e}, groups={names}"
        )

        if HAS_POSTHOC and p_value < 0.05:
            posthoc = sp.posthoc_dunn(
                subset,
                val_col="sbert_similarity",
                group_col="persona",
                p_adjust="holm",
            )

            ordered = [
                persona
                for persona in PERSONA_ORDER
                if persona in posthoc.index
            ]
            posthoc = posthoc.loc[ordered, ordered]

            tier_name = (
                str(tier)
                .split()[-1]
                .lower()
                .replace("(", "")
                .replace(")", "")
            )

            path = os.path.join(
                output_dir,
                f"rq2_posthoc_persona_{tier_name}.csv",
            )

            posthoc.to_csv(path)
            lines.append(f"  Posthoc Dunn-Holm saved: {path}")

    lines.extend([
        "\nModel-Tier Effect Within Each Persona",
        "-" * 80,
    ])

    for persona in PERSONA_ORDER:
        subset = full_df[full_df["persona"] == persona]

        groups = [
            subset[subset["model_tier"] == tier]["sbert_similarity"]
            .dropna()
            .values
            for tier in MODEL_LABEL_ORDER
            if not subset[subset["model_tier"] == tier].empty
        ]

        names = [
            tier
            for tier in MODEL_LABEL_ORDER
            if not subset[subset["model_tier"] == tier].empty
        ]

        if len(groups) < 2:
            lines.append(
                f"{persona}: skipped; fewer than two model-tier groups."
            )
            continue

        h_value, p_value = stats.kruskal(*groups)

        lines.append(
            f"{persona}: Kruskal-Wallis H={h_value:.6f}, "
            f"p={p_value:.10e}, groups={names}"
        )

        if HAS_POSTHOC and p_value < 0.05:
            posthoc = sp.posthoc_dunn(
                subset,
                val_col="sbert_similarity",
                group_col="model_tier",
                p_adjust="holm",
            )

            ordered = [
                tier
                for tier in MODEL_LABEL_ORDER
                if tier in posthoc.index
            ]
            posthoc = posthoc.loc[ordered, ordered]

            path = os.path.join(
                output_dir,
                f"rq2_posthoc_model_tier_{persona}.csv",
            )

            posthoc.to_csv(path)
            lines.append(f"  Posthoc Dunn-Holm saved: {path}")

    lines.extend([
        "\nInterpretation Note",
        "-" * 80,
        (
            "Kruskal-Wallis tests evaluate non-parametric distributional "
            "shifts. Persona tests compare naive/formal/expert within a "
            "model tier; model-tier tests compare 7B/14B/32B within a persona."
        ),
    ])

    output_path = os.path.join(
        output_dir,
        "rq2_statistical_tests.txt",
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved statistical report: {output_path}")


def export_qualitative_examples(full_df, output_dir):
    examples = []

    for tier in MODEL_LABEL_ORDER:
        for persona in PERSONA_ORDER:
            subset = full_df[
                (full_df["model_tier"] == tier)
                & (full_df["persona"] == persona)
            ].copy()

            if subset.empty:
                continue

            subset["example_rank"] = (
                subset["sbert_similarity"]
                + (subset["completeness_score"] / 3.0)
                + subset["schema_adherence_score"]
                - (subset["word_count"] / 2000.0)
            )

            best = subset.sort_values(
                "example_rank",
                ascending=False,
            ).iloc[0]

            worst = subset.sort_values(
                "example_rank",
                ascending=True,
            ).iloc[0]

            for label, row in [("best", best), ("worst", worst)]:
                examples.append({
                    "case_type": label,
                    "model_tier": tier,
                    "persona": persona,
                    "index": row["index"],
                    "cwe": row.get(
                        "cwe",
                        row.get("truth_cwe", "Unknown"),
                    ),
                    "word_count": row["word_count"],
                    "sbert_similarity": row["sbert_similarity"],
                    "completeness_score": row["completeness_score"],
                    "schema_adherence_score": row["schema_adherence_score"],
                    "key_security_insights": row["key_security_insights"],
                    "generated_explanation": row["generated_explanation"],
                })

    examples_df = pd.DataFrame(examples)

    path = os.path.join(
        output_dir,
        "rq2_qualitative_best_worst_examples.csv",
    )

    examples_df.to_csv(path, index=False)
    print(f"Saved qualitative examples: {path}")


def create_visualizations(full_df, summary, output_dir):
    configure_plot_style()

    model_palette = {
        MODEL_TIERS[size]["label"]: MODEL_TIERS[size]["color"]
        for size in MODEL_SIZE_ORDER
    }

    persona_palette = {
        persona: PERSONA_COLORS[persona]
        for persona in PERSONA_ORDER
    }

    # Similarity by persona and model tier
    plt.figure(figsize=(13.5, 7.2))

    ax = sns.barplot(
        data=full_df,
        x="persona",
        y="sbert_similarity",
        hue="model_tier",
        order=PERSONA_ORDER,
        hue_order=MODEL_LABEL_ORDER,
        palette=model_palette,
        errorbar=("ci", 95),
        capsize=0.12,
        err_kws={"linewidth": 2.0},
        edgecolor="black",
        linewidth=1.2,
    )

    ax.set_title(
        "RQ2: Reference Similarity by Persona and Model Tier",
        pad=18,
    )
    ax.set_xlabel("Prompt Persona", labelpad=14)
    ax.set_ylabel("SBERT Similarity to Reference", labelpad=14)
    ax.set_xticklabels(
        [PERSONA_LABELS[persona] for persona in PERSONA_ORDER]
    )
    ax.set_ylim(
        0,
        max(1.0, full_df["sbert_similarity"].max() * 1.10),
    )
    polish_axes(ax)

    legend = ax.legend(
        title="Model Tier",
        frameon=True,
        loc="best",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
    )
    legend.get_title().set_fontweight("bold")

    save_figure(
        os.path.join(
            output_dir,
            "rq2_similarity_by_persona_and_tier_ieee",
        )
    )

    # Information density
    plt.figure(figsize=(13.5, 7.2))

    ax = sns.barplot(
        data=full_df,
        x="persona",
        y="information_density",
        hue="model_tier",
        order=PERSONA_ORDER,
        hue_order=MODEL_LABEL_ORDER,
        palette=model_palette,
        errorbar=("ci", 95),
        capsize=0.12,
        err_kws={"linewidth": 2.0},
        edgecolor="black",
        linewidth=1.2,
    )

    ax.set_title(
        "RQ2: Information Density by Persona and Model Tier",
        pad=18,
    )
    ax.set_xlabel("Prompt Persona", labelpad=14)
    ax.set_ylabel("Information Density", labelpad=14)
    ax.set_xticklabels(
        [PERSONA_LABELS[persona] for persona in PERSONA_ORDER]
    )
    polish_axes(ax)

    legend = ax.legend(
        title="Model Tier",
        frameon=True,
        loc="best",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
    )
    legend.get_title().set_fontweight("bold")

    save_figure(
        os.path.join(
            output_dir,
            "rq2_information_density_by_persona_and_tier_ieee",
        )
    )

    # Word count versus similarity
    plt.figure(figsize=(12.5, 8))

    ax = sns.scatterplot(
        data=full_df,
        x="word_count",
        y="sbert_similarity",
        hue="persona",
        style="model_tier",
        hue_order=PERSONA_ORDER,
        style_order=MODEL_LABEL_ORDER,
        palette=persona_palette,
        alpha=0.78,
        s=90,
        edgecolor="black",
    )

    ax.set_title("RQ2: Verbosity vs. Reference Similarity", pad=18)
    ax.set_xlabel("Word Count", labelpad=14)
    ax.set_ylabel("SBERT Similarity to Reference", labelpad=14)
    polish_axes(ax)

    legend = ax.legend(
        title="Persona / Model Tier",
        frameon=True,
        loc="best",
        fontsize=LEGEND_SIZE - 1,
        title_fontsize=LEGEND_TITLE_SIZE,
    )
    legend.get_title().set_fontweight("bold")

    save_figure(
        os.path.join(
            output_dir,
            "rq2_verbosity_vs_similarity_ieee",
        )
    )

    # Completeness heatmap
    completeness_pivot = summary.pivot(
        index="persona",
        columns="model_tier",
        values="mean_completeness",
    )

    completeness_pivot = completeness_pivot.loc[
        [
            persona
            for persona in PERSONA_ORDER
            if persona in completeness_pivot.index
        ],
        [
            tier
            for tier in MODEL_LABEL_ORDER
            if tier in completeness_pivot.columns
        ],
    ]

    completeness_pivot.index = [
        PERSONA_LABELS[persona]
        for persona in completeness_pivot.index
    ]

    plt.figure(figsize=(11.8, 6.8))

    ax = sns.heatmap(
        completeness_pivot,
        annot=True,
        fmt=".3f",
        cmap="crest",
        vmin=0,
        vmax=3,
        linewidths=0.8,
        linecolor="white",
        annot_kws={
            "fontsize": ANNOT_SIZE,
            "fontweight": "bold",
        },
        cbar_kws={
            "label": "Mean Completeness Score",
            "shrink": 0.84,
        },
    )

    ax.set_title("RQ2: Security Reasoning Completeness", pad=18)
    ax.set_xlabel("Model Tier", labelpad=14)
    ax.set_ylabel("Prompt Persona", labelpad=14)
    ax.tick_params(axis="x", rotation=18)
    ax.tick_params(axis="y", rotation=0)
    polish_axes(ax)

    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=TICK_SIZE)
    colorbar.set_label(
        "Mean Completeness Score",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold",
    )

    save_figure(
        os.path.join(
            output_dir,
            "rq2_completeness_heatmap_ieee",
        )
    )

    # Schema adherence heatmap
    schema_pivot = summary.pivot(
        index="persona",
        columns="model_tier",
        values="mean_schema_adherence",
    )

    schema_pivot = schema_pivot.loc[
        [
            persona
            for persona in PERSONA_ORDER
            if persona in schema_pivot.index
        ],
        [
            tier
            for tier in MODEL_LABEL_ORDER
            if tier in schema_pivot.columns
        ],
    ]

    schema_pivot.index = [
        PERSONA_LABELS[persona]
        for persona in schema_pivot.index
    ]

    plt.figure(figsize=(11.8, 6.8))

    ax = sns.heatmap(
        schema_pivot,
        annot=True,
        fmt=".3f",
        cmap="mako",
        vmin=0,
        vmax=1,
        linewidths=0.8,
        linecolor="white",
        annot_kws={
            "fontsize": ANNOT_SIZE,
            "fontweight": "bold",
        },
        cbar_kws={
            "label": "Schema Adherence Score",
            "shrink": 0.84,
        },
    )

    ax.set_title("RQ2: Instruction Schema Adherence", pad=18)
    ax.set_xlabel("Model Tier", labelpad=14)
    ax.set_ylabel("Prompt Persona", labelpad=14)
    ax.tick_params(axis="x", rotation=18)
    ax.tick_params(axis="y", rotation=0)
    polish_axes(ax)

    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=TICK_SIZE)
    colorbar.set_label(
        "Schema Adherence Score",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold",
    )

    save_figure(
        os.path.join(
            output_dir,
            "rq2_schema_adherence_heatmap_ieee",
        )
    )

    # Security pillar identification
    pillar_summary = (
        full_df.groupby(
            ["model_tier", "persona"],
            observed=False,
        )[
            [
                "source_identified",
                "sink_identified",
                "trigger_identified",
            ]
        ]
        .mean()
        .reset_index()
    )

    pillar_melted = pillar_summary.melt(
        id_vars=["model_tier", "persona"],
        value_vars=[
            "source_identified",
            "sink_identified",
            "trigger_identified",
        ],
        var_name="security_pillar",
        value_name="identification_rate",
    )

    pillar_melted["security_pillar"] = pillar_melted[
        "security_pillar"
    ].map({
        "source_identified": "Source",
        "sink_identified": "Sink",
        "trigger_identified": "Trigger",
    })

    grid = sns.catplot(
        data=pillar_melted,
        x="persona",
        y="identification_rate",
        hue="security_pillar",
        col="model_tier",
        kind="bar",
        order=PERSONA_ORDER,
        col_order=MODEL_LABEL_ORDER,
        height=5.8,
        aspect=1.12,
        palette="Set2",
        edgecolor="black",
        linewidth=1.1,
        errorbar=None,
        legend=False,
    )

    grid.set_axis_labels(
        "Prompt Persona",
        "Identification Rate",
    )
    grid.set_titles(
        "{col_name}",
        size=FACET_TITLE_SIZE,
        weight="bold",
    )

    for axis in grid.axes.flat:
        axis.set_xticklabels(
            [PERSONA_LABELS[persona] for persona in PERSONA_ORDER]
        )
        axis.set_ylim(0, 1.05)
        polish_axes(axis)

    from matplotlib.patches import Patch

    palette = sns.color_palette("Set2")

    legend_handles = [
        Patch(
            facecolor=palette[0],
            edgecolor="black",
            label="Source",
        ),
        Patch(
            facecolor=palette[1],
            edgecolor="black",
            label="Sink",
        ),
        Patch(
            facecolor=palette[2],
            edgecolor="black",
            label="Trigger",
        ),
    ]

    legend = grid.fig.legend(
        handles=legend_handles,
        title="Security Pillar",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        frameon=True,
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
    )
    legend.get_title().set_fontweight("bold")

    grid.fig.set_size_inches(19, 7.8)
    grid.fig.subplots_adjust(
        top=0.82,
        bottom=0.20,
        wspace=0.08,
    )
    grid.fig.suptitle(
        "RQ2: Security Pillar Identification Rates",
        y=0.98,
        fontsize=TITLE_SIZE + 1,
        fontweight="bold",
    )

    png_path = os.path.join(
        output_dir,
        "rq2_security_pillar_rates_ieee.png",
    )
    pdf_path = os.path.join(
        output_dir,
        "rq2_security_pillar_rates_ieee.pdf",
    )

    grid.savefig(
        png_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
    )
    grid.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved figure: {png_path}", flush=True)
    print(f"Saved figure: {pdf_path}", flush=True)

    # Instruction stability
    plt.figure(figsize=(13.5, 7.2))

    ax = sns.barplot(
        data=full_df,
        x="persona",
        y="instruction_stability_score",
        hue="model_tier",
        order=PERSONA_ORDER,
        hue_order=MODEL_LABEL_ORDER,
        palette=model_palette,
        errorbar=("ci", 95),
        capsize=0.12,
        err_kws={"linewidth": 2.0},
        edgecolor="black",
        linewidth=1.2,
    )

    ax.set_title(
        "RQ2: Instruction Stability by Persona and Model Tier",
        pad=18,
    )
    ax.set_xlabel("Prompt Persona", labelpad=14)
    ax.set_ylabel("Instruction Stability Score", labelpad=14)
    ax.set_xticklabels(
        [PERSONA_LABELS[persona] for persona in PERSONA_ORDER]
    )
    ax.set_ylim(0, 1.05)
    polish_axes(ax)

    legend = ax.legend(
        title="Model Tier",
        frameon=True,
        loc="best",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
    )
    legend.get_title().set_fontweight("bold")

    save_figure(
        os.path.join(
            output_dir,
            "rq2_instruction_stability_by_persona_and_tier_ieee",
        )
    )


def export_latex_tables(summary, output_dir):
    summary_latex = (
        summary.style
        .format(precision=4)
        .hide(axis="index")
        .to_latex()
    )

    path = os.path.join(
        output_dir,
        "rq2_multitier_persona_summary.tex",
    )

    with open(path, "w", encoding="utf-8") as file:
        file.write(summary_latex)

    print(f"Saved LaTeX table: {path}")


def main():
    args = parse_args()
    project_root = resolve_project_root(args)

    results_dir = args.results_dir or os.path.join(
        project_root,
        "results",
    )
    output_dir = args.output_dir or os.path.join(
        results_dir,
        "analysis_rq2_ieee",
    )

    os.makedirs(output_dir, exist_ok=True)

    truth_path = os.path.join(
        project_root,
        "data",
        "filtered_experimental_set.csv",
    )

    print("=" * 90)
    print("LAUNCHING RQ2 MULTI-TIER MULTI-PERSONA EVALUATION")
    print("=" * 90)
    print(f"Project root    : {project_root}")
    print(f"Results dir     : {results_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Truth file      : {truth_path}")
    print(f"Embedding model : {args.embedding_model}")
    print(f"Sizes           : {args.sizes}")
    print(f"Personas        : {args.personas}")

    device = choose_device(args.device)

    print(f"Embedding device: {device}")
    print("=" * 90)

    truth_df, truth_col = load_ground_truth(truth_path)
    print(f"Reference text column: {truth_col}")

    try:
        sentence_model = SentenceTransformer(
            args.embedding_model,
            device=device,
        )
    except TypeError:
        sentence_model = SentenceTransformer(
            args.embedding_model
        )
        sentence_model = sentence_model.to(device)

    full_df = build_analysis_dataframe(
        args=args,
        results_dir=results_dir,
        truth_df=truth_df,
        truth_col=truth_col,
        st_model=sentence_model,
        device=device,
    )

    master_path = os.path.join(
        output_dir,
        "rq2_multitier_master_analysis.csv",
    )

    full_df.to_csv(master_path, index=False)
    print(f"\nSaved master analysis dataframe: {master_path}")

    summary, pivot_similarity = make_summary_tables(
        full_df,
        output_dir,
    )

    run_statistical_tests(full_df, output_dir)
    export_qualitative_examples(full_df, output_dir)

    print("\nRendering IEEE-ready visualizations...")
    create_visualizations(full_df, summary, output_dir)
    export_latex_tables(summary, output_dir)

    print("\n[SUCCESS] RQ2 evaluation complete.")
    print(f"[SUCCESS] Outputs saved under: {output_dir}\n")


if __name__ == "__main__":
    main()