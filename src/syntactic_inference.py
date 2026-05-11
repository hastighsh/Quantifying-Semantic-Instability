import os
import gc
import sys
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# PATH SETUP
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "perturbed_set.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# MODEL
MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"


def build_prompt(code):
    return (
        "Identify the vulnerability in this C code and explain why it is dangerous.\n\n"
        f"{code}\n\n"
        "Explanation:"
    )


def run_inference(model_id=MODEL_ID):
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Could not find input file: {INPUT_PATH}")
        return

    output_file = os.path.join(OUTPUT_DIR, "rq1_results_qwen14b_transformers.csv")

    if os.path.exists(output_file):
        os.remove(output_file)

    print(f"--- Loading model: {model_id} ---")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.eval()

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded dataset with {len(df)} rows.")
    print(f"Total generations: {len(df) * 2}")

    header_written = False

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        results = []

        for code_col, label in [
            ("code", "baseline"),
            ("perturbed_code", "perturbed"),
        ]:
            if code_col not in row or pd.isna(row[code_col]):
                continue

            prompt = build_prompt(row[code_col])

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            explanation = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            results.append({
                "index": row["index"] if "index" in row else idx,
                "cwe": row["cwe"] if "cwe" in row else "Unknown",
                "model_tier": "medium",
                "model_id": model_id,
                "experiment_type": label,
                "explanation": explanation,
                "code_used": row[code_col],
            })

            del inputs, outputs, generated_tokens
            torch.cuda.empty_cache()

        pd.DataFrame(results).to_csv(
            output_file,
            mode="a",
            index=False,
            header=not header_written,
        )
        header_written = True

    print("Inference complete.")
    print(f"Saved to: {output_file}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_inference()