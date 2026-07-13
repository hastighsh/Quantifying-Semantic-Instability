# Auditing the Auditors: Robustness of LLM Security Explanations

This repository contains the code, prompts, and evaluation scripts for the paper **“Auditing the Auditors: Robustness of LLM Security Explanations.”**

The project evaluates whether code-specialized Large Language Models preserve vulnerability reasoning under:

- **RQ1:** Behaviour-preserving syntactic changes
- **RQ2:** Different prompt formality levels
- **RQ3:** Misleading natural-language comments added to unchanged code

## Repository Structure

```text
repository-root/
├── data/
├── prompts/
│   ├── naive_persona.txt
│   ├── formal_persona.txt
│   └── expert_persona.txt
├── results/
├── src/
│   ├── data_setup.py
│   ├── add_baseline.py
│   ├── verify_parallel.py
│   ├── filter_baseline.py
│   └── rq1_syntax/
│       ├── perturb_code.py
│       ├── syntactic_inference.py
│       └── evaluate_rq1.py
├── rq2_prompts/
│   ├── prompt_inference.py
│   └── evaluate_rq2.py
├── rq3_context/
│   ├── deceptive_data_injection.py
│   ├── context_inference.py
│   └── evaluate_rq3.py
├── requirements.txt
└── README.md
```

## Models

| Model | Usage |
|---|---|
| `Qwen/Qwen2.5-Coder-7B-Instruct` | RQ1–RQ3 inference |
| `Qwen/Qwen2.5-Coder-14B-Instruct` | RQ1–RQ3 inference |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | Baseline and RQ1–RQ3 inference |
| `meta-llama/Llama-3.1-70B-Instruct` | Baseline verification |
| `sentence-transformers/all-MiniLM-L6-v2` | Semantic-similarity evaluation |

## Setup

Run all commands from the repository root.

```bash
git clone <repository-url>
cd <repository-name>

python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install a CUDA-compatible PyTorch version, then install the remaining dependencies:

```bash
pip install pandas numpy datasets tqdm transformers accelerate bitsandbytes
pip install tree-sitter tree-sitter-c sentence-transformers scipy
pip install matplotlib seaborn scikit-posthocs sacrebleu jinja2
```

Alternatively:

```bash
pip install -r requirements.txt
```

Llama-3.1-70B may require Hugging Face authentication and license approval.

An optional Hugging Face cache directory can be configured with:

```bash
export HF_HOME=/path/to/huggingface/cache
```

## Execution Order

```text
data_setup.py
    ↓
add_baseline.py
    ↓
merge baseline shards
    ↓
verify_parallel.py
    ↓
filter_baseline.py
    ↓
filtered_experimental_set.csv
    ├── RQ1
    │   └── perturb_code.py
    │       → syntactic_inference.py
    │       → evaluate_rq1.py
    │
    ├── RQ2
    │   └── prompt_inference.py
    │       → evaluate_rq2.py
    │
    └── RQ3
        └── deceptive_data_injection.py
            → context_inference.py
            → evaluate_rq3.py
                ↑
                rq2_formal_results_qwen*.csv
```

RQ1 and RQ2 only require the verified baseline dataset. RQ3 evaluation also uses the formal RQ2 outputs as its non-deceptive baseline.

# 1. Baseline Construction

## Create the Initial Dataset

```bash
python src/data_setup.py
```

Output:

```text
data/gold_set_raw.csv
```

## Generate Baseline Explanations

Run all eight shards:

```bash
for shard_id in {0..7}; do
    python src/add_baseline.py \
        --shard-id "$shard_id" \
        --num-shards 8
done
```

Outputs are written to:

```text
data/baseline_shards/
```

The script supports resume mode and skips completed samples.

## Merge the Shards

Run from the repository root:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

shard_dir = Path("data/baseline_shards")
output_path = Path("data/baseline_results_complete.csv")

files = sorted(shard_dir.glob("baseline_results_shard_*_of_8.csv"))

if len(files) != 8:
    raise RuntimeError(f"Expected 8 shard files, found {len(files)}.")

df = pd.concat(
    [pd.read_csv(path, dtype=str) for path in files],
    ignore_index=True,
)

df["original_index"] = pd.to_numeric(
    df["original_index"],
    errors="raise",
).astype(int)

if df["original_index"].duplicated().any():
    raise RuntimeError("Duplicate original_index values found.")

df = df.sort_values("original_index").reset_index(drop=True)
df.to_csv(output_path, index=False)

print(f"Saved {len(df)} rows to {output_path}")
PY
```

Output:

```text
data/baseline_results_complete.csv
```

## Verify and Filter the Baseline

```bash
python src/verify_parallel.py
python src/filter_baseline.py
```

Outputs:

```text
data/verified_baseline_complete_4gpu.csv
data/filtered_experimental_set.csv
```

`filtered_experimental_set.csv` is the input to RQ1, RQ2, and RQ3.

# 2. RQ1: Syntactic Sensitivity

RQ1 applies behaviour-preserving identifier renaming and loop rewriting.

## Generate Perturbed Code

```bash
python src/rq1_syntax/perturb_code.py
```

Output:

```text
data/perturbed_set.csv
```

## Run Inference

```bash
python src/rq1_syntax/syntactic_inference.py --size 7B
python src/rq1_syntax/syntactic_inference.py --size 14B
python src/rq1_syntax/syntactic_inference.py --size 32B
```

Outputs:

```text
results/rq1_results_qwen7b_transformers.csv
results/rq1_results_qwen14b_transformers.csv
results/rq1_results_qwen32b_transformers.csv
```

## Evaluate RQ1

```bash
python src/rq1_syntax/evaluate_rq1.py
```

Results are written to:

```text
results/analysis_v3_ieee/
```

Optional configuration:

```bash
python src/rq1_syntax/evaluate_rq1.py \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --device auto
```

# 3. RQ2: Prompt Formality

RQ2 compares three prompt personas:

```text
naive
formal
expert
```

The prompt files must be stored in:

```text
prompts/
├── naive_persona.txt
├── formal_persona.txt
└── expert_persona.txt
```

## Run Inference

Run all model-persona combinations:

```bash
for size in 7B 14B 32B; do
    for persona in naive formal expert; do
        python rq2_prompts/prompt_inference.py \
            --size "$size" \
            --persona "$persona"
    done
done
```

Outputs follow this naming format:

```text
results/rq2_<persona>_results_qwen<size>.csv
```

Examples:

```text
results/rq2_naive_results_qwen7b.csv
results/rq2_formal_results_qwen14b.csv
results/rq2_expert_results_qwen32b.csv
```

## Evaluate RQ2

```bash
python rq2_prompts/evaluate_rq2.py --strict
```

Results are written to:

```text
results/analysis_rq2_ieee/
```

Optional configuration:

```bash
python rq2_prompts/evaluate_rq2.py \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --device auto \
    --sizes 7B 14B 32B \
    --personas naive formal expert \
    --strict
```

# 4. RQ3: Deceptive Context

RQ3 injects misleading comments into unchanged source code.

Deception strategies include:

```text
FALSE_FIX
SWAP
PHANTOM_BUG
RED_HERRING
AUTHORITY_APPEAL
```

Comments are inserted at:

```text
PREFIX
INSIDE
SUFFIX
```

## Generate the Deceptive Dataset

```bash
python rq3_context/deceptive_data_injection.py
```

Output:

```text
data/deceptive_experimental_set.csv
```

Overwrite an existing file with:

```bash
python rq3_context/deceptive_data_injection.py --overwrite
```

## Run RQ3 Inference

```bash
python rq3_context/context_inference.py --size 7B
python rq3_context/context_inference.py --size 14B
python rq3_context/context_inference.py --size 32B
```

Outputs:

```text
results/rq3_deceptive_results_qwen7b.csv
results/rq3_deceptive_results_qwen14b.csv
results/rq3_deceptive_results_qwen32b.csv
```

RQ3 uses the formal prompt by default:

```text
prompts/formal_persona.txt
```

## Required Formal RQ2 Baselines

RQ3 evaluation compares deceptive outputs against these formal non-deceptive outputs:

```text
results/rq2_formal_results_qwen7b.csv
results/rq2_formal_results_qwen14b.csv
results/rq2_formal_results_qwen32b.csv
```

Create them with:

```bash
python rq2_prompts/prompt_inference.py --size 7B --persona formal
python rq2_prompts/prompt_inference.py --size 14B --persona formal
python rq2_prompts/prompt_inference.py --size 32B --persona formal
```

## Evaluate RQ3

```bash
python rq3_context/evaluate_rq3.py --strict
```

Results are written to:

```text
results/analysis_rq3_ieee/
```

Optional configuration:

```bash
python rq3_context/evaluate_rq3.py \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --device auto \
    --sizes 7B 14B 32B \
    --strict
```

# Complete Command Summary

```bash
# Baseline
python src/data_setup.py

for shard_id in {0..7}; do
    python src/add_baseline.py \
        --shard-id "$shard_id" \
        --num-shards 8
done

# Merge the baseline shards before continuing.
python src/verify_parallel.py
python src/filter_baseline.py

# RQ1
python src/rq1_syntax/perturb_code.py
python src/rq1_syntax/syntactic_inference.py --size 7B
python src/rq1_syntax/syntactic_inference.py --size 14B
python src/rq1_syntax/syntactic_inference.py --size 32B
python src/rq1_syntax/evaluate_rq1.py

# RQ2
for size in 7B 14B 32B; do
    for persona in naive formal expert; do
        python rq2_prompts/prompt_inference.py \
            --size "$size" \
            --persona "$persona"
    done
done

python rq2_prompts/evaluate_rq2.py --strict

# RQ3
python rq3_context/deceptive_data_injection.py
python rq3_context/context_inference.py --size 7B
python rq3_context/context_inference.py --size 14B
python rq3_context/context_inference.py --size 32B
python rq3_context/evaluate_rq3.py --strict
```

## Reproducibility Notes

- Run commands from the repository root.
- Preserve `original_index` throughout the pipeline.
- Complete and merge all baseline shards before verification.
- Run baseline verification before RQ1–RQ3.
- Keep prompts and inference parameters unchanged across compared models.
- Use the same embedding model for all evaluations.
- Complete the formal RQ2 runs before evaluating RQ3.
- Do not combine outputs generated with different model revisions.

## Hardware

Inference requires CUDA-capable GPUs.

- 7B can generally run on one suitable GPU.
- 14B and 32B may require high-memory or multiple GPUs.
- Llama-3.1-70B verification is configured for multi-GPU 4-bit inference.
- Evaluation scripts can run on CPU or CUDA.

Default inference batch sizes are:

| Model | Batch size |
|---|---:|
| 7B | 4 |
| 14B | 2 |
| 32B | 1 |

Reduce the batch size through the command line when GPU memory is limited.

## Citation

This repository accompanies an anonymous paper submission. Citation information will be added after the review process.
