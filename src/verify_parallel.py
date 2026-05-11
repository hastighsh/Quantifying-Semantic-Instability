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

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

BATCH_SIZE = 2
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 5
SAVE_EVERY_BATCHES = 5


def build_prompt(row):
    return f"""You are a Senior Security Auditor.

GROUND TRUTH:
{row.get("truth_description", "N/A")}

AI EXPLANATION:
{row.get("baseline_explanation", "N/A")}

Does the AI correctly identify the security root cause?

Answer only YES or NO.
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

    print(f"--- Initializing 8B Judge | shard {shard_id}/{num_shards} ---")
    print(f"--- Output file: {output_file} ---")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0}, 
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.eval()

    df = pd.read_csv(INPUT_FILE)
    if "original_index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "original_index"})

    df = df.reset_index(drop=True)
    
    # Take only this shard
    df = df[df["original_index"] % num_shards == shard_id].copy()

    results = []

    # Resume support for this shard
    if os.path.exists(output_file):
        done_df = pd.read_csv(output_file)
        if "original_index" in done_df.columns:
            done_indices = set(done_df["original_index"].tolist())
            df = df[~df["original_index"].isin(done_indices)]
            results = done_df.to_dict("records")
            print(f"--- Resuming shard {shard_id}: {len(done_indices)} already done, {len(df)} remaining ---")

    print(f"--- Running shard {shard_id}: {len(df)} samples ---")

    batch_counter = 0

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc=f"Shard {shard_id}"):
        batch_df = df.iloc[start:start + BATCH_SIZE]

        prompts = []
        for _, row in batch_df.iterrows():
            messages = [{"role": "user", "content": build_prompt(row)}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

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

        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        responses = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for (_, row), response in zip(batch_df.iterrows(), responses):
            response = response.strip()
            is_correct = response.upper().startswith("YES")

            results.append({
                **row.to_dict(),
                "judge_response": response,
                "is_correct": is_correct,
                "shard_id": shard_id,
                "num_shards": num_shards,
            })

        batch_counter += 1

        if batch_counter % SAVE_EVERY_BATCHES == 0:
            pd.DataFrame(results).to_csv(output_file, index=False)
            print(f"Shard {shard_id}: saved checkpoint with {len(results)} samples")

        del inputs, outputs, generated_tokens
        torch.cuda.empty_cache()
        gc.collect()

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"--- Shard {shard_id} complete. Saved to {output_file} ---")


if __name__ == "__main__":
    args = parse_args()
    run_judge(args.shard_id, args.num_shards)