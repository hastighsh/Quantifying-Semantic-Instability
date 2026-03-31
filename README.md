# Quantifying-Semantic-Instability in LLM-Based Vulnerability Explanations

This repository contains the experimental pipeline and datasets for investigating the structural and semantic dependencies of **DeepSeek-Coder-7B** reasoning in automated software vulnerability identification.

## Project Overview
While Large Language Models (LLMs) are increasingly used to generate human-readable security insights, their **deterministic consistency** remains a concern. This project quantifies the "Robustness" of these explanations across three perturbation axes:

1.  **Syntactic Sensitivity (RQ1):** Impact of identifier renaming (e.g., `buffer_limit` → `var_1`).
2.  **Prompt Formalism (RQ2):** Impact of persona-based phrasing (Naive, Formal, Expert).
3.  **Context Interference (RQ3):** Impact of deceptive code comments (False Fixes and Vulnerability Swaps).

## Key Findings
* **Lexical Dependency:** Renaming variables induced a **0.54 Mean Semantic Variance**. In 96.8% of cases, the model's reasoning collapsed when descriptive names were removed.
* **The Expertise Paradox:** "Expert" personas maximize information density but are prone to stylistic "Logorrhea" (technical repetition).
* **Silent Hijacking:** Deceptive comments achieved a **21.88% Hijack Rate** for vulnerability swaps. 
* **Metric Limitation:** A Wilcoxon signed-rank test on RQ3 yielded $p=1.00$, proving that **SBERT embedding similarity fails to capture logical correctness** in security reasoning.

## Methodology
The pipeline utilizes a **Baseline Verification (Judge) Phase**:
1.  **Initial Dataset:** 150 samples from the Big-Vul dataset.
2.  **Self-Critique Audit:** DeepSeek-Coder-7B acted as a "Senior Auditor" to verify its own baseline accuracy against developer ground truth (commit messages).
3.  **Filtered Set:** Only the 64 samples (42.6%) with verified baseline accuracy were used for perturbation testing.

## Setup & Usage
This project is optimized for **Google Colab T4 GPU** instances.

### 1. Install Dependencies
Run the following command in a Colab cell to set up the environment:
```bash
pip install -q transformers accelerate bitsandbytes sentence-transformers scipy seaborn
```

### 2. Configuration
To ensure the notebooks can access your datasets and save results, follow these configuration steps:

* **Google Drive Integration:** Mount your drive at the start of each notebook to access the project directory:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
* **Project Path:** Update the `DRIVE_PATH` variable in the configuration cell of each notebook. It must point to the specific directory containing your `/data` and `/results` folders:
    ```python
    DRIVE_PATH = '/content/drive/MyDrive/Project' # Adjust this to your folder structure
    ```
* **Directory Integrity:** Ensure that the `/data` folder contains `filtered_experimental_set.csv` before running the RQ2 and RQ3 notebooks.

### 3. Run Evaluation
To ensure data consistency and baseline verification, execute the notebooks in the following specific order. This sequence allows the outputs from the baseline verification and prompt inference to feed into the respective evaluation scripts.

**Phase 1: Verification & Pre-processing**
1. `verify_baseline.ipynb`: Validates the 150 original samples using the self-critique (Judge) method.
2. `filter_baseline.ipynb`: Generates the final $N=64$ verified dataset (`gold_set.csv`).
3. `perturb_code.ipynb`: Performs programmatic identifier renaming for RQ1.
4. `deceptive_data_injection.ipynb`: Injects misleading comments into the codebase for RQ3.

**Phase 2: Inference & Analysis**
5. `syntactic_inference.ipynb` → `evaluate_rq1.ipynb`: Runs and analyzes Syntactic Sensitivity (RQ1).
6. `prompt_inference.ipynb` → `rq2_post_processing.ipynb` → `evaluate_rq2.ipynb`: Executes and assesses Persona Alignment (RQ2).
7. `context_inference.ipynb` → `evaluate_rq3.ipynb`: Conducts and evaluates Deceptive Context Interference (RQ3).
