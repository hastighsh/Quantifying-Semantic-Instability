import os
import gc
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

torch.backends.cuda.matmul.allow_tf32 = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "baseline_results_ready_for_judge.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "judge_shards")

MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"

BATCH_SIZE = 1
MAX_INPUT_TOKENS = 3072
# Workspace tokens to let the judge explain its reasoning before giving the verdict
MAX_NEW_TOKENS = 256
SAVE_EVERY_BATCHES = 5


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    return parser.parse_args()


def run_judge(shard_id, num_shards):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(
        OUTPUT_DIR,
        f"verified_baseline_shard_{shard_id}_of_{num_shards}.csv",
    )

    print(f"--- Initializing Llama-3.1-70B Judge | shard {shard_id}/{num_shards} ---")
    print(f"--- Output file: {output_file} ---")

    # 4-bit Quantization Config to compress the 70B model down to ~42GB VRAM so it fits on 1x H100
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # Kept in bfloat16 to match H100 compute architecture natively
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    df = pd.read_csv(INPUT_FILE, dtype=str)
    if "original_index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "original_index"})
    
    df["original_index"] = df["original_index"].astype(int)
    df = df.reset_index(drop=True)
    
    # Partition dataset into shards
    df = df[df["original_index"] % num_shards == shard_id].copy()

    results = []
    done_indices = set()

    # Resume capability logic block
    if os.path.exists(output_file):
        try:
            done_df = pd.read_csv(output_file, dtype=str)
            if "original_index" in done_df.columns:
                done_df["original_index"] = done_df["original_index"].astype(int)
                done_indices = set(done_df["original_index"].tolist())
                results = done_df.to_dict("records")
                df = df[~df["original_index"].isin(done_indices)]
                print(f"--- Resuming shard {shard_id}: {len(done_indices)} already done, {len(df)} remaining ---")
        except Exception as e:
            print(f"Skipping corrupt checkpoint file: {e}")

    print(f"--- Running shard {shard_id}: {len(df)} samples ---")

    batch_counter = 0

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc=f"Shard {shard_id}"):
        batch_df = df.iloc[start:start + BATCH_SIZE]

        prompts = []
        rows = []
        for _, row in batch_df.iterrows():
            messages = [
                {"role": "system", "content": "You are a precise, objective senior code auditing judge."},
                {"role": "user", "content": build_prompt(row)}
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
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        responses = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for row, response in zip(rows, responses):
            response_text = response.strip()
            
            # Extract final verdict block safely
            verdict_segment = response_text.split("Final Verdict:")[-1].strip().upper()
            is_correct = "YES" in verdict_segment

            output_row = row.to_dict()
            output_row.update({
                "judge_response": response_text,
                "is_correct": is_correct,
                "shard_id": int(shard_id),
                "num_shards": int(num_shards),
            })
            results.append(output_row)

        batch_counter += 1

        if batch_counter % SAVE_EVERY_BATCHES == 0:
            pd.DataFrame(results).to_csv(output_file, index=False)

        del inputs, outputs, generated_tokens
        torch.cuda.empty_cache()
        gc.collect()

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"--- Shard {shard_id} complete. Saved to {output_file} ---")


if __name__ == "__main__":
    args = parse_args()
    run_judge(args.shard_id, args.num_shards)