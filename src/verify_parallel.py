import os
import gc
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

torch.backends.cuda.matmul.allow_tf32 = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "baseline_results_complete.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "verified_baseline_complete_4gpu.csv"
)

REPO_ID = "meta-llama/Llama-3.1-70B-Instruct"
MODEL_REVISION = "1605565b47bb9346c5515c34102e054115b4f98b"

# Uses the user's Hugging Face cache configuration.
# If no environment variable is set, Hugging Face uses its default cache.
CACHE_DIR = (
    os.getenv("HF_HUB_CACHE")
    or os.getenv("HUGGINGFACE_HUB_CACHE")
    or os.getenv("HF_HOME")
)

BATCH_SIZE = 4
MAX_INPUT_TOKENS = 3072
MAX_NEW_TOKENS = 256
SAVE_EVERY_BATCHES = 10


def build_prompt(row):
    return f"""You are a Senior Security Auditor. Your task is to evaluate an AI-generated vulnerability explanation against a verified ground-truth description.

GROUND TRUTH VULNERABILITY DESCRIPTION:
{row.get("truth_description", "N/A")}

AI-GENERATED SECURITY EXPLANATION:
{row.get("baseline_explanation", "N/A")}

CRITERIA:
Does the AI-generated explanation correctly identify the true security root cause outlined in the ground truth?

Provide your evaluation in this exact format:
Reasoning: <Your brief analysis of whether the core vulnerability matches>
Final Verdict: <YES or NO>
"""


def run_judge_multi_gpu():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print("--- Initializing 4x H100 Multi-GPU Distributed Judge ---")
    print(f"--- Target Output File: {OUTPUT_FILE} ---")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        REPO_ID,
        revision=MODEL_REVISION,
        cache_dir=CACHE_DIR,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    # Load the same pinned model snapshot through Hugging Face
    # instead of using a machine-specific local snapshot path.
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        revision=MODEL_REVISION,
        cache_dir=CACHE_DIR,
        quantization_config=bnb_config,
        device_map="balanced",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    model.eval()

    print("\n--- Model Memory Footprint Distributed Map ---")
    for name, device in list(model.hf_device_map.items())[:80:20]:
        print(
            f"Submodule Block [{name}] mapped securely to "
            f"-> CUDA DEVICE:{device}"
        )
    print("------------------------------------------------\n")

    df = pd.read_csv(INPUT_FILE, dtype=str)

    if "original_index" not in df.columns:
        df = df.reset_index().rename(
            columns={"index": "original_index"}
        )

    df["original_index"] = df["original_index"].astype(int)
    df = df.reset_index(drop=True)

    results = []
    done_indices = set()

    if os.path.exists(OUTPUT_FILE):
        try:
            done_df = pd.read_csv(OUTPUT_FILE, dtype=str)

            if "original_index" in done_df.columns:
                done_df["original_index"] = done_df[
                    "original_index"
                ].astype(int)

                done_indices = set(
                    done_df["original_index"].tolist()
                )

                results = done_df.to_dict("records")

                df = df[
                    ~df["original_index"].isin(done_indices)
                ]

                print(
                    f"--- Resuming: {len(done_indices)} samples "
                    f"processed, {len(df)} remaining ---"
                )

        except Exception as e:
            print(f"Skipping corrupt output file: {e}")

    print(f"--- Processing {len(df)} remaining samples ---")
    batch_counter = 0

    for start in tqdm(
        range(0, len(df), BATCH_SIZE),
        desc="4-GPU Evaluator Loop",
    ):
        batch_df = df.iloc[start:start + BATCH_SIZE]

        prompts = []
        rows = []

        for _, row in batch_df.iterrows():
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise, objective senior "
                        "code auditing judge."
                    ),
                },
                {
                    "role": "user",
                    "content": build_prompt(row),
                },
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompts.append(prompt)
            rows.append(row)

        if not prompts:
            continue

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to("cuda:0")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

        generated_tokens = outputs[
            :,
            inputs["input_ids"].shape[1]:,
        ]

        responses = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for row, response in zip(rows, responses):
            response_text = response.strip()

            verdict_segment = (
                response_text
                .split("Final Verdict:")[-1]
                .strip()
                .upper()
            )

            is_correct = "YES" in verdict_segment

            output_row = row.to_dict()

            output_row.update(
                {
                    "judge_response": response_text,
                    "is_correct": is_correct,
                }
            )

            results.append(output_row)

        batch_counter += 1

        if batch_counter % SAVE_EVERY_BATCHES == 0:
            pd.DataFrame(results).to_csv(
                OUTPUT_FILE,
                index=False,
            )

        del inputs, outputs, generated_tokens

        torch.cuda.empty_cache()
        gc.collect()

    pd.DataFrame(results).to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"--- Job Complete! Saved output to: "
        f"{OUTPUT_FILE} ---"
    )


if __name__ == "__main__":
    run_judge_multi_gpu()