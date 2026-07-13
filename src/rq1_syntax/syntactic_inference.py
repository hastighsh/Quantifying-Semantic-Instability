#!/usr/bin/env python3
import os
import gc
import sys
import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


torch.backends.cuda.matmul.allow_tf32 = True


MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 512
SAVE_EVERY_BATCHES = 10


DEFAULT_BATCH_SIZE_BY_MODEL = {
    "7B": 4,
    "14B": 2,
    "32B": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ1 Multi-Tier Quantitative Automated Inference Script"
    )

    parser.add_argument(
        "--size",
        type=str,
        required=True,
        choices=["7B", "14B", "32B"],
        help="Model size parameter scale",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional manual batch size override. Defaults: 7B=4, 14B=2, 32B=1.",
    )

    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=MAX_INPUT_TOKENS,
        help=f"Maximum input tokens. Default: {MAX_INPUT_TOKENS}",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Maximum generated tokens. Default: {MAX_NEW_TOKENS}",
    )

    return parser.parse_args()


def build_prompt(code):
    return (
        "Identify the vulnerability in this C code and explain why it is dangerous.\n\n"
        f"{code}\n\n"
        "Explanation:"
    )


def get_input_device():
    if not torch.cuda.is_available():
        print("[CRITICAL ERROR] CUDA is not available.")
        print(f"torch version        : {torch.__version__}")
        print(f"torch cuda version   : {torch.version.cuda}")
        print(f"torch device count   : {torch.cuda.device_count()}")
        sys.exit(1)

    return torch.device("cuda:0")


def get_device_map():
    gpu_count = torch.cuda.device_count()

    if gpu_count <= 0:
        print("[CRITICAL ERROR] No CUDA GPUs detected.")
        sys.exit(1)

    if gpu_count == 1:
        return {"": 0}

    return "balanced"


def print_cuda_report():
    print("\n--- CUDA Runtime Report ---")
    print(f"torch version      : {torch.__version__}")
    print(f"torch cuda version : {torch.version.cuda}")
    print(f"cuda available     : {torch.cuda.is_available()}")
    print(f"cuda device count  : {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"cuda:{i}            : {torch.cuda.get_device_name(i)}")
    print("---------------------------\n")


def main():
    args = parse_args()

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else DEFAULT_BATCH_SIZE_BY_MODEL[args.size]
    )

    input_device = get_input_device()
    print_cuda_report()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Script location: repository_root/src/rq1_syntax/
    project_root = os.path.abspath(
        os.path.join(script_dir, "..", "..")
    )

    input_path = os.path.join(
        project_root,
        "data",
        "perturbed_set.csv",
    )

    output_dir = os.path.join(
        project_root,
        "results",
    )

    os.makedirs(output_dir, exist_ok=True)

    model_id = f"Qwen/Qwen2.5-Coder-{args.size}-Instruct"

    output_file = os.path.join(
        output_dir,
        f"rq1_results_qwen{args.size.lower()}_transformers.csv",
    )

    if not os.path.exists(input_path):
        print(f"[CRITICAL ERROR] Input file not found: {input_path}")
        sys.exit(1)

    print("\n=========================================================")
    print(f"LOADING BF16 MODEL: {model_id}")
    print(f"Model Source        : {model_id}")
    print(f"Input File Source   : {input_path}")
    print(f"Output File Dest.   : {output_file}")
    print(f"Batch Size          : {batch_size}")
    print(f"Max Input Tokens    : {args.max_input_tokens}")
    print(f"Max New Tokens      : {args.max_new_tokens}")
    print("Quantization        : DISABLED")
    print("Reason              : bitsandbytes CUDA 13.2 binary is unavailable")
    print("=========================================================\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=get_device_map(),
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    except Exception as e:
        print(f"[CRITICAL ERROR] Weight loader system crashed: {repr(e)}")
        sys.exit(1)

    model.eval()

    print("\n--- Model Memory Footprint Distributed Map ---")
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for name, device in list(model.hf_device_map.items())[:80:20]:
            print(f"Submodule Block [{name}] mapped securely to -> {device}")
    else:
        print(f"Model placed completely onto device: {model.device}")
    print("------------------------------------------------\n")

    df = pd.read_csv(input_path, dtype=str)
    print(f"Loaded dataset containing {len(df)} rows.")

    if "original_index" not in df.columns:
        df = df.reset_index().rename(
            columns={"index": "original_index"}
        )

    df["original_index"] = df["original_index"].astype(int)
    df = df.reset_index(drop=True)

    results = []
    done_keys = set()

    if os.path.exists(output_file):
        try:
            done_df = pd.read_csv(output_file, dtype=str)

            required_done_cols = {"index", "experiment_type"}

            if required_done_cols.issubset(set(done_df.columns)):
                done_df["index"] = done_df["index"].astype(int)

                done_keys = set(
                    zip(
                        done_df["index"].astype(int).tolist(),
                        done_df["experiment_type"].astype(str).tolist(),
                    )
                )

                results = done_df.to_dict("records")

                print(
                    f"--- Resuming: {len(done_keys)} "
                    "completed generations found ---"
                )

            else:
                print(
                    "--- Existing output file found but missing "
                    "resume columns. Starting fresh. ---"
                )
                os.remove(output_file)

        except Exception as e:
            print(
                "Skipping corrupt output file and starting fresh: "
                f"{repr(e)}"
            )
            os.remove(output_file)

    inference_items = []

    for idx, row in df.iterrows():
        original_index = int(row["original_index"])

        for code_col, label in [
            ("code", "baseline"),
            ("perturbed_code", "perturbed"),
        ]:
            if code_col not in row:
                continue

            if (
                pd.isna(row[code_col])
                or str(row[code_col]).strip() == ""
            ):
                continue

            resume_key = (original_index, label)

            if resume_key in done_keys:
                continue

            inference_items.append(
                {
                    "row": row,
                    "original_index": original_index,
                    "code_col": code_col,
                    "experiment_type": label,
                    "prompt": build_prompt(row[code_col]),
                }
            )

    print(
        f"--- Processing {len(inference_items)} "
        "remaining prompt generations ---"
    )

    batch_counter = 0

    for start in tqdm(
        range(0, len(inference_items), batch_size),
        desc=f"Qwen {args.size} Inference Loop",
    ):
        batch_items = inference_items[
            start:start + batch_size
        ]

        prompts = [
            item["prompt"]
            for item in batch_items
        ]

        if not prompts:
            continue

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(input_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_tokens = outputs[
            :,
            inputs["input_ids"].shape[1]:,
        ]

        explanations = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for item, explanation in zip(
            batch_items,
            explanations,
        ):
            row = item["row"]

            result_row = {
                "index": item["original_index"],
                "cwe": (
                    row["cwe"]
                    if "cwe" in row and pd.notna(row["cwe"])
                    else "Unknown"
                ),
                "model_tier": args.size,
                "model_id": model_id,
                "experiment_type": item["experiment_type"],
                "explanation": explanation.strip(),
                "code_used": row[item["code_col"]],
            }

            results.append(result_row)

        batch_counter += 1

        if batch_counter % SAVE_EVERY_BATCHES == 0:
            pd.DataFrame(results).to_csv(
                output_file,
                index=False,
            )

            print(
                f"--- Checkpoint saved to "
                f"{output_file} ---"
            )

        del inputs, outputs, generated_tokens
        torch.cuda.empty_cache()
        gc.collect()

    pd.DataFrame(results).to_csv(
        output_file,
        index=False,
    )

    print(
        f"\n[SUCCESS] Inference routine concluded "
        f"for size {args.size}."
    )
    print(f"[SUCCESS] Saved output to: {output_file}\n")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()