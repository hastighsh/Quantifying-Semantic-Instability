import pandas as pd
import numpy as np
import re
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import scikit_posthocs as sp
from sentence_transformers import SentenceTransformer, util

# 1. CONFIGURATION
PROJECT_ROOT = os.getcwd()

GROUND_TRUTH_PATH = os.path.join(
    PROJECT_ROOT, "data", "filtered_experimental_set.csv"
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PERSONAS = ["naive", "formal", "expert"]
MODEL_NAME = "all-MiniLM-L6-v2"


# 2. TEXT CLEANING
def clean_explanation(text):
    """
    Extracts the technical reasoning core from the generated explanation.
    This reduces persona-specific filler before SBERT comparison.
    """
    if not isinstance(text, str) or len(text.strip()) < 10:
        return "N/A"

    # Remove fenced code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Prefer ANALYSIS section if present
    analysis_match = re.search(
        r"ANALYSIS:\s*(.*?)(?=IMPACT:|STOP:|VULNERABILITY:|$)",
        text,
        re.IGNORECASE | re.DOTALL
    )

    if analysis_match:
        content = analysis_match.group(1).strip()
    else:
        # Remove vulnerability heading if present
        content = re.sub(
            r"^.*?VULNERABILITY:.*?\n",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL
        ).strip()

    # Keep first 100 words to reduce verbosity bias
    return " ".join(content.split()[:100])


# 3. COMPLETENESS SCORE: SOURCE / SINK / TRIGGER
def detect_security_attributes(text):
    """
    Detects whether the explanation identifies the three key security reasoning pillars:
    Source, Sink, and Trigger.

    Score range:
        0 = none identified
        1 = one attribute identified
        2 = two attributes identified
        3 = source, sink, and trigger identified
    """
    if not isinstance(text, str):
        text = ""

    text_lower = text.lower()

    patterns = {
        "source_identified": (
            r"\b("
            r"input|user input|attacker-controlled|external input|parameter|argument|argv|argc|"
            r"recv|read|scanf|gets|file|socket|network|request|payload|data from user"
            r")\b"
        ),

        "sink_identified": (
            r"\b("
            r"memcpy|strcpy|strncpy|sprintf|snprintf|strcat|malloc|calloc|realloc|free|"
            r"buffer|array|pointer|index|heap|stack|memory write|memory read|dereference|"
            r"copy|allocation|sink"
            r")\b"
        ),

        "trigger_identified": (
            r"\b("
            r"overflow|out-of-bounds|bounds|boundary|length check|missing check|"
            r"validation|unchecked|insufficient validation|null dereference|null pointer|"
            r"use-after-free|double free|integer overflow|wraparound|race condition|"
            r"format string|leak|corruption|trigger|when .* exceeds|exceeds .* size"
            r")\b"
        )
    }

    detected = {}

    for key, pattern in patterns.items():
        detected[key] = bool(re.search(pattern, text_lower))

    detected["completeness_score"] = (
        int(detected["source_identified"])
        + int(detected["sink_identified"])
        + int(detected["trigger_identified"])
    )

    return pd.Series(detected)


def extract_key_security_insights(row):
    """
    Produces a compact qualitative label showing which security attributes
    were identified by the explanation.
    """
    insights = []

    if row["source_identified"]:
        insights.append("Source")
    if row["sink_identified"]:
        insights.append("Sink")
    if row["trigger_identified"]:
        insights.append("Trigger")

    if not insights:
        return "None"

    return ", ".join(insights)


# 4. SCHEMA AND INSTRUCTION ADHERENCE FALLBACKS
def calculate_schema_adherence(text):
    """
    Measures whether the output follows the expected response schema.
    Used as a fallback if the result CSV does not already include adheres_to_schema.
    """
    if not isinstance(text, str):
        return 0

    text_upper = text.upper()

    required_sections = ["VULNERABILITY", "ANALYSIS", "IMPACT"]
    hits = sum(section in text_upper for section in required_sections)

    return hits / len(required_sections)


def calculate_instructional_drift(text):
    """
    Approximates instruction stability.
    Higher value means lower drift / stronger compliance.

    This fallback rewards concise schema-following answers and penalizes
    excessive verbosity without structure.
    """
    if not isinstance(text, str):
        return 0

    word_count = len(text.split())
    schema_score = calculate_schema_adherence(text)

    # Penalize very long responses if schema adherence is weak
    verbosity_penalty = min(word_count / 600, 1.0)

    drift_score = (0.75 * schema_score) + (0.25 * (1 - verbosity_penalty))

    return max(0, min(1, drift_score))


# 5. INFORMATION DENSITY
def calculate_information_density(similarity, word_count):
    """
    Measures accuracy per log-word.

    Higher score means the explanation gives more semantically useful
    information with fewer words.
    """
    if word_count <= 0:
        return 0

    return similarity / np.log1p(word_count)


# 6. MAIN EVALUATION
def run_evaluation():
    print("--- Starting Consolidated RQ2 Evaluation ---")

    if not os.path.exists(GROUND_TRUTH_PATH):
        alt_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "filtered_experimental_set.csv"
        )

        if os.path.exists(alt_path):
            current_truth_path = alt_path
        else:
            print(f"ERROR: Ground truth not found at {GROUND_TRUTH_PATH}")
            return None
    else:
        current_truth_path = GROUND_TRUTH_PATH

    truth_df = pd.read_csv(current_truth_path)

    if "truth_description" in truth_df.columns:
        truth_col = "truth_description"
    elif "explanation" in truth_df.columns:
        truth_col = "explanation"
    elif "description" in truth_df.columns:
        truth_col = "description"
    else:
        raise ValueError(
            "Could not find a ground-truth explanation column. "
            "Expected one of: truth_description, explanation, description."
        )

    if "index" not in truth_df.columns:
        raise ValueError("Ground truth file must contain an 'index' column.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sbert_model = SentenceTransformer(MODEL_NAME).to(device)

    all_results = []

    for persona in PERSONAS:
        res_path = os.path.join(RESULTS_DIR, f"rq2_{persona}_results.csv")

        if not os.path.exists(res_path):
            print(f"Skipping {persona}: File not found at {res_path}")
            continue

        print(f"Processing Persona: {persona.upper()}")

        df = pd.read_csv(res_path)

        if "index" not in df.columns:
            raise ValueError(f"{res_path} must contain an 'index' column.")

        if "generated_explanation" not in df.columns:
            raise ValueError(
                f"{res_path} must contain a 'generated_explanation' column."
            )

        merged = pd.merge(
            df,
            truth_df[["index", truth_col]],
            on="index",
            how="inner"
        )

        merged["persona"] = persona

        # Basic metrics
        merged["cleaned_res"] = merged["generated_explanation"].apply(clean_explanation)
        merged["word_count"] = merged["generated_explanation"].apply(
            lambda x: len(str(x).split())
        )

        # Completeness metrics
        attribute_scores = merged["generated_explanation"].apply(
            detect_security_attributes
        )

        merged = pd.concat([merged, attribute_scores], axis=1)

        merged["key_security_insights"] = merged.apply(
            extract_key_security_insights,
            axis=1
        )

        # Fallback adherence metrics if missing
        if "adheres_to_schema" not in merged.columns:
            merged["adheres_to_schema"] = merged["generated_explanation"].apply(
                calculate_schema_adherence
            )

        if "instructional_drift" not in merged.columns:
            merged["instructional_drift"] = merged["generated_explanation"].apply(
                calculate_instructional_drift
            )

        # SBERT semantic similarity
        gen_emb = sbert_model.encode(
            merged["cleaned_res"].tolist(),
            convert_to_tensor=True
        )

        truth_emb = sbert_model.encode(
            merged[truth_col].astype(str).tolist(),
            convert_to_tensor=True
        )

        cos_sim = util.cos_sim(truth_emb, gen_emb)
        merged["sbert_sim"] = torch.diag(cos_sim).cpu().numpy()

        # Information density
        merged["info_density"] = merged.apply(
            lambda row: calculate_information_density(
                row["sbert_sim"],
                row["word_count"]
            ),
            axis=1
        )

        all_results.append(merged)

    if not all_results:
        print("No results to process.")
        return None

    final_df = pd.concat(all_results, ignore_index=True)

    consolidated_path = os.path.join(
        RESULTS_DIR,
        "rq2_consolidated_analysis.csv"
    )

    final_df.to_csv(consolidated_path, index=False)
    print(f"Saved consolidated results to: {consolidated_path}")

    print("\n--- Statistical Significance (Persona Impact) ---")
    persona_groups = [group['sbert_sim'].values for _, group in final_df.groupby('persona')]
    stat, p_val = kruskal(*persona_groups)
    print(f"Kruskal-Wallis p-value for Personas: {p_val:.8f}")

    if p_val < 0.05:
        # Granular pairwise analysis
        posthoc_p = sp.posthoc_dunn(final_df, val_col='sbert_sim', group_col='persona', p_adjust='bonferroni')
        posthoc_p.to_csv(os.path.join(RESULTS_DIR, "rq2_persona_posthoc.csv"))
        print("Granular Persona comparisons saved to rq2_persona_posthoc.csv")

    return final_df


# 7. TABLE GENERATION
def generate_summary_tables(full_df):
    print("\n--- RQ2 Aggregated Stats ---")

    stats = full_df.groupby("persona").agg({
        "sbert_sim": "mean",
        "word_count": "mean",
        "completeness_score": "mean",
        "source_identified": "mean",
        "sink_identified": "mean",
        "trigger_identified": "mean",
        "info_density": "mean",
        "adheres_to_schema": "mean",
        "instructional_drift": "mean"
    }).reset_index()

    stats = stats.rename(columns={
        "sbert_sim": "SBERT Similarity",
        "word_count": "Mean Word Count",
        "completeness_score": "Completeness Score",
        "source_identified": "Source Identified Rate",
        "sink_identified": "Sink Identified Rate",
        "trigger_identified": "Trigger Identified Rate",
        "info_density": "Information Density",
        "adheres_to_schema": "Schema Adherence",
        "instructional_drift": "Instruction Stability"
    })

    stats_path = os.path.join(RESULTS_DIR, "rq2_persona_summary_stats.csv")
    stats.to_csv(stats_path, index=False)

    print(stats.to_string(index=False))
    print(f"\nSaved summary table to: {stats_path}")

    # Security insights table
    insight_table = full_df.groupby("persona").agg({
        "source_identified": "sum",
        "sink_identified": "sum",
        "trigger_identified": "sum",
        "completeness_score": "mean"
    }).reset_index()

    insight_table = insight_table.rename(columns={
        "source_identified": "Source Count",
        "sink_identified": "Sink Count",
        "trigger_identified": "Trigger Count",
        "completeness_score": "Mean Completeness"
    })

    insight_path = os.path.join(
        RESULTS_DIR,
        "rq2_key_security_insights_by_persona.csv"
    )

    insight_table.to_csv(insight_path, index=False)

    print("\n--- Key Security Insights Identified by Persona ---")
    print(insight_table.to_string(index=False))
    print(f"\nSaved insight table to: {insight_path}")

    return stats, insight_table


# 8. QUALITATIVE EXAMPLE EXTRACTION
def export_qualitative_examples(full_df):
    """
    Exports one representative explanation per persona.

    Preference:
    - high completeness
    - moderate/short word count
    - high SBERT similarity
    """
    examples = []

    for persona in PERSONAS:
        sub = full_df[full_df["persona"] == persona].copy()

        if sub.empty:
            continue

        sub["example_rank"] = (
            sub["completeness_score"]
            + sub["sbert_sim"]
            - (sub["word_count"] / 1000)
        )

        best = sub.sort_values("example_rank", ascending=False).iloc[0]

        examples.append({
            "persona": persona,
            "index": best["index"],
            "word_count": best["word_count"],
            "sbert_sim": best["sbert_sim"],
            "completeness_score": best["completeness_score"],
            "key_security_insights": best["key_security_insights"],
            "generated_explanation": best["generated_explanation"]
        })

    examples_df = pd.DataFrame(examples)

    examples_path = os.path.join(
        RESULTS_DIR,
        "rq2_qualitative_persona_examples.csv"
    )

    examples_df.to_csv(examples_path, index=False)

    print(f"\nSaved qualitative examples to: {examples_path}")

    return examples_df


# 9. VISUALIZATIONS
def create_visualizations(full_df):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

    # 1. Sycophancy trap scatter
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=full_df,
        x="word_count",
        y="sbert_sim",
        hue="persona",
        style="persona",
        s=100,
        alpha=0.7,
        palette="viridis"
    )
    plt.title(
        "The Sycophancy Trap: Verbosity vs. Accuracy",
        fontweight="bold"
    )
    plt.xlabel("Word Count (Verbosity)", fontweight="bold")
    plt.ylabel("SBERT Similarity to Ground Truth", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "rq2_sycophancy_trap.png"),
        dpi=300
    )
    plt.close()

    # 2. Information density
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=full_df,
        x="persona",
        y="info_density",
        hue="persona",
        palette="magma",
        legend=False
    )
    plt.title(
        "Information Density (Accuracy per Log-Word)",
        fontweight="bold"
    )
    plt.xlabel("persona")
    plt.ylabel("Density Score", fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "rq2_info_density.png"),
        dpi=300
    )
    plt.close()

    # 3. Instruction adherence
    adherence_df = full_df.groupby("persona")[
        ["adheres_to_schema", "instructional_drift"]
    ].mean().reset_index()

    adherence_melted = adherence_df.melt(
        id_vars="persona",
        var_name="Metric",
        value_name="Rate"
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=adherence_melted,
        x="persona",
        y="Rate",
        hue="Metric",
        palette="coolwarm"
    )
    plt.title("Instruction Adherence by Persona", fontweight="bold")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "rq2_instruction_adherence.png"),
        dpi=300
    )
    plt.close()

    # 4. Completeness score by persona
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=full_df,
        x="persona",
        y="completeness_score",
        hue="persona",
        palette="crest",
        legend=False
    )
    plt.title(
        "Completeness Score by Persona",
        fontweight="bold"
    )
    plt.xlabel("persona")
    plt.ylabel("Mean Completeness Score")
    plt.ylim(0, 3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "rq2_completeness_score.png"),
        dpi=300
    )
    plt.close()

    # 5. Source / Sink / Trigger rates
    pillar_df = full_df.groupby("persona")[
        ["source_identified", "sink_identified", "trigger_identified"]
    ].mean().reset_index()

    pillar_melted = pillar_df.melt(
        id_vars="persona",
        var_name="Security Attribute",
        value_name="Identification Rate"
    )

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=pillar_melted,
        x="persona",
        y="Identification Rate",
        hue="Security Attribute",
        palette="Set2"
    )
    plt.title(
        "Key Security Insights Identified by Persona",
        fontweight="bold"
    )
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "rq2_security_insights_by_persona.png"),
        dpi=300
    )
    plt.close()

    print("\nSaved all RQ2 visualizations.")


# 10. TABLE EXPORT
def export_tables(stats, insight_table):
    stats_path = os.path.join(
        RESULTS_DIR,
        "rq2_persona_summary_stats.tex"
    )

    insight_path = os.path.join(
        RESULTS_DIR,
        "rq2_key_security_insights_by_persona.tex"
    )

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(stats.to_latex(index=False, float_format="%.4f"))

    with open(insight_path, "w", encoding="utf-8") as f:
        f.write(insight_table.to_latex(index=False, float_format="%.4f"))

    print(f"Saved stats table to: {stats_path}")
    print(f"Saved insight table to: {insight_path}")


# 11. MAIN
if __name__ == "__main__":
    full_df = run_evaluation()

    if full_df is not None:
        stats, insight_table = generate_summary_tables(full_df)
        export_qualitative_examples(full_df)
        create_visualizations(full_df)
        export_tables(stats, insight_table)

        print("\n--- RQ2 Evaluation Complete ---")