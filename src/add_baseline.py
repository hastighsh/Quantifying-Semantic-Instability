import os
import gc
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Allow TensorFloat-32 for faster matrix multiplications on Ampere/Hopper (H100) architecture
torch.backends.cuda.matmul.allow_tf32 = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "data", "gold_set_raw.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "baseline_shards")

# Upgraded to the massive 32B SOTA Code Model
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

BATCH_SIZE = 1
MAX_INPUT_TOKENS = 2560
MAX_NEW_TOKENS = 512
SAVE_EVERY_BATCHES = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    return parser.parse_args()


def run_baseline_inference(shard_id, num_shards):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"baseline_results_shard_{shard_id}_of_{num_shards}.csv"
    )

    print(f"--- Loading Model: {MODEL_ID} | shard {shard_id}/{num_shards} ---")
    print(f"--- Output: {output_path} ---")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    # Loaded in native unquantized bfloat16 precision. Fits flawlessly on 1x H100!
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    df = df.reset_index().rename(columns={"index": "original_index"})

    # shard by original index
    df = df[df["original_index"] % num_shards == shard_id].copy()

    results = []

    # Resume support
    if os.path.exists(output_path):
        done_df = pd.read_csv(output_path)
        if "original_index" in done_df.columns:
            done_indices = set(done_df["original_index"].tolist())
            results = done_df.to_dict("records")
            df = df[~df["original_index"].isin(done_indices)]
            print(
                f"--- Resuming shard {shard_id}: "
                f"{len(done_indices)} done, {len(df)} remaining ---"
            )

    print(f"--- Starting Baseline Inference on shard {shard_id}: {len(df)} samples ---")

    batch_counter = 0

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc=f"Shard {shard_id}"):
        batch_df = df.iloc[start:start + BATCH_SIZE]

        prompts = []
        rows = []

        for _, row in batch_df.iterrows():
            code = row["code"]
            
            # Formulating structure for chat template compatibility
            user_message = f"""You are an expert C security researcher. Analyze the following code for vulnerabilities.

Structure your response as follows:
1. Logic Flow: Briefly describe what the code does.
2. Root Cause: Identify the exact line or logic that is vulnerable.
3. Danger: Explain why this is dangerous.

CODE:
{code}

Final Explanation:"""

            messages = [
                {"role": "system", "content": "You are a helpful, precise, and objective academic security code auditing assistant."},
                {"role": "user", "content": user_message}
            ]
            
            # Dynamically compile prompt with proper chat templates
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_text)
            rows.append(row)

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
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        # Slice strictly the new token generations
        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]

        explanations = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for row, explanation in zip(rows, explanations):
            cwe = row.get("CWE ID", row.get("cwe_id", row.get("cwe", "Unknown")))
            cve = row.get("CVE ID", row.get("cve_id", row.get("cve", "Unknown")))
            truth = row.get("truth_description", "N/A")

            results.append({
                "original_index": row["original_index"],
                "cwe": cwe,
                "cve": cve,
                "truth_description": truth,
                "code": row["code"],
                "baseline_explanation": explanation.strip(),
                "shard_id": shard_id,
                "num_shards": num_shards,
            })

        batch_counter += 1

        if batch_counter % SAVE_EVERY_BATCHES == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)

        # Secure memory cleaning block
        del inputs, outputs, generated_tokens
        torch.cuda.empty_cache()
        gc.collect()

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"--- Shard {shard_id} complete. Saved to {output_path} ---")


if __name__ == "__main__":
    args = parse_args()
    run_baseline_inference(args.shard_id, args.num_shards)