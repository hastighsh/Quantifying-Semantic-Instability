#!/usr/bin/env python3
import argparse
import gc
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True

DEFAULT_BATCH_SIZE_BY_MODEL = {
    "7B": 4,
    "14B": 2,
    "32B": 1,
}

DEFAULT_MAX_INPUT_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 400
SAVE_EVERY_BATCHES = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ3 Multi-Tier Deceptive Context Inference"
    )

    parser.add_argument(
        "--size",
        type=str,
        required=True,
        choices=["7B", "14B", "32B"],
        help="Target model size tier.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override. Defaults: 7B=4, 14B=2, 32B=1.",
    )

    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help=f"Maximum input tokens. Default: {DEFAULT_MAX_INPUT_TOKENS}",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum generated tokens. Default: {DEFAULT_MAX_NEW_TOKENS}",
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root. Default: parent directory of this script.",
    )

    parser.add_argument(
        "--persona-prompt",
        type=str,
        default=None,
        help="Prompt path. Default: prompts/formal_persona.txt",
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input deceptive CSV. Default: data/deceptive_experimental_set.csv",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output CSV path.",
    )

    return parser.parse_args()


def resolve_project_root(args):
    if args.project_root:
        return os.path.abspath(args.project_root)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))


def cuda_report():
    print("\n--- CUDA Runtime Report ---")
    print(f"torch version      : {torch.__version__}")
    print(f"torch cuda version : {torch.version.cuda}")
    print(f"cuda available     : {torch.cuda.is_available()}")
    print(f"cuda device count  : {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"cuda:{i}            : {torch.cuda.get_device_name(i)}")

    print("---------------------------\n")


def get_input_device():
    if not torch.cuda.is_available():
        print("[CRITICAL ERROR] CUDA is not available.")
        print(f"torch version      : {torch.__version__}")
        print(f"torch cuda version : {torch.version.cuda}")
        print(f"device count       : {torch.cuda.device_count()}")
        sys.exit(42)

    return torch.device("cuda:0")


def get_device_map():
    gpu_count = torch.cuda.device_count()

    if gpu_count <= 0:
        print("[CRITICAL ERROR] No CUDA GPUs detected.")
        sys.exit(42)

    if gpu_count == 1:
        return {"": 0}

    return "balanced"


def normalize_input_dataframe(df):
    df = df.copy()

    if "code" not in df.columns:
        print("[CRITICAL ERROR] Input dataset must contain a 'code' column.")
        sys.exit(1)

    if "base_code" not in df.columns:
        df["base_code"] = df["code"]

    if "cwe" not in df.columns:
        df["cwe"] = "Unknown"

    if "true_cwe" not in df.columns:
        df["true_cwe"] = df["cwe"]

    if "original_index" not in df.columns:
        if "index" in df.columns:
            df["original_index"] = df["index"]
        else:
            df = df.reset_index().rename(
                columns={"index": "original_index"}
            )

    if "index" not in df.columns:
        df["index"] = df["original_index"]

    if "model_tier" not in df.columns:
        print(
            "[WARN] Input has no model_tier column. "
            "This script will assume rows apply to all tiers."
        )
        df["model_tier"] = "ALL"

    metadata_columns = [
        "deception_type",
        "injection_placement",
        "injected_comment",
        "deception_id",
    ]

    for column in metadata_columns:
        if column not in df.columns:
            df[column] = "Unknown"

    df["original_index"] = df["original_index"].astype(int)
    return df


def load_system_prompt(prompt_path):
    if not os.path.exists(prompt_path):
        print(f"[CRITICAL ERROR] Prompt not found: {prompt_path}")
        sys.exit(1)

    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def build_chat_prompt(tokenizer, system_instruction, code):
    user_content = (
        "CODE TO EVALUATE:\n"
        f"{code}\n\n"
        "Provide your analysis below:"
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_generation(tokenizer, output_ids, input_len):
    generated_tokens = output_ids[input_len:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def load_model_and_tokenizer(model_source):
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            device_map=get_device_map(),
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            device_map=get_device_map(),
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    model.eval()
    return tokenizer, model


def main():
    args = parse_args()

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else DEFAULT_BATCH_SIZE_BY_MODEL[args.size]
    )

    project_root = resolve_project_root(args)

    input_path = (
        os.path.abspath(args.input)
        if args.input
        else os.path.join(
            project_root,
            "data",
            "deceptive_experimental_set.csv",
        )
    )

    prompt_path = (
        os.path.abspath(args.persona_prompt)
        if args.persona_prompt
        else os.path.join(
            project_root,
            "prompts",
            "formal_persona.txt",
        )
    )

    output_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)

    model_name = f"Qwen2.5-Coder-{args.size}-Instruct"
    model_source = f"Qwen/{model_name}"

    output_path = (
        os.path.abspath(args.output)
        if args.output
        else os.path.join(
            output_dir,
            f"rq3_deceptive_results_qwen{args.size.lower()}.csv",
        )
    )

    if not os.path.exists(input_path):
        print(
            f"[CRITICAL ERROR] Deceptive dataset not found: "
            f"{input_path}"
        )
        sys.exit(1)

    input_device = get_input_device()
    cuda_report()

    print("\n=========================================================")
    print("RUNNING RQ3 DECEPTIVE CONTEXT INFERENCE")
    print(f"Model Configuration : {model_name}")
    print(f"Model Source        : {model_source}")
    print(f"Input Dataset       : {input_path}")
    print(f"Prompt Path         : {prompt_path}")
    print(f"Output Path         : {output_path}")
    print(f"Batch Size          : {batch_size}")
    print(f"Max Input Tokens    : {args.max_input_tokens}")
    print(f"Max New Tokens      : {args.max_new_tokens}")
    print("Quantization        : DISABLED")
    print("Precision           : BF16")
    print("=========================================================\n")

    system_instruction = load_system_prompt(prompt_path)

    df = pd.read_csv(input_path, dtype=str)
    df = normalize_input_dataframe(df)

    if (
        "model_tier" in df.columns
        and "ALL" not in set(df["model_tier"].astype(str))
    ):
        df = df[
            df["model_tier"].astype(str) == args.size
        ].copy()

    df = df.reset_index(drop=True)

    print(f"Rows selected for {args.size}: {len(df)}")

    if df.empty:
        print(
            f"[CRITICAL ERROR] No rows selected for "
            f"model_tier={args.size}"
        )
        sys.exit(1)

    done_indices = set()
    results = []

    if os.path.exists(output_path):
        try:
            done_df = pd.read_csv(output_path, dtype=str)

            if "original_index" in done_df.columns:
                done_df["original_index"] = done_df[
                    "original_index"
                ].astype(int)

                done_indices = set(
                    done_df["original_index"].tolist()
                )

                results = done_df.to_dict("records")

                print(
                    f"--- Resuming: {len(done_indices)} "
                    "completed rows found ---"
                )
            else:
                print(
                    "[WARN] Existing output lacks original_index. "
                    "Starting fresh."
                )
                os.remove(output_path)

        except Exception as error:
            print(
                f"[WARN] Existing output unreadable: "
                f"{repr(error)}"
            )
            print("[WARN] Starting fresh.")
            os.remove(output_path)

    pending_df = df[
        ~df["original_index"].astype(int).isin(done_indices)
    ].copy()

    pending_df = pending_df.reset_index(drop=True)

    print(f"Rows remaining for this run: {len(pending_df)}")

    if pending_df.empty:
        print("[SUCCESS] Nothing to do. Output already complete.")
        return

    try:
        tokenizer, model = load_model_and_tokenizer(
            model_source
        )
    except Exception as error:
        print(
            f"[CRITICAL ERROR] Model/tokenizer loading failed: "
            f"{repr(error)}"
        )
        sys.exit(1)

    print("\n--- Model Device Map ---")

    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for name, device in list(
            model.hf_device_map.items()
        )[:80:20]:
            print(
                f"Submodule Block [{name}] mapped securely "
                f"to -> {device}"
            )
    else:
        print(f"Model placed onto device: {model.device}")

    print("------------------------\n")

    batch_counter = 0

    for start in tqdm(
        range(0, len(pending_df), batch_size),
        desc=f"RQ3 Qwen-{args.size} deceptive",
    ):
        batch_df = pending_df.iloc[
            start:start + batch_size
        ]

        prompts = [
            build_chat_prompt(
                tokenizer,
                system_instruction,
                row["code"],
            )
            for _, row in batch_df.iterrows()
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(input_device)

        input_lengths = inputs[
            "attention_mask"
        ].sum(dim=1).tolist()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        for local_i, (_, row) in enumerate(
            batch_df.iterrows()
        ):
            explanation = extract_generation(
                tokenizer=tokenizer,
                output_ids=outputs[local_i],
                input_len=int(input_lengths[local_i]),
            )

            result_row = {
                "index": row["index"],
                "original_index": int(
                    row["original_index"]
                ),
                "cwe": row["cwe"],
                "true_cwe": row["true_cwe"],
                "model_tier": args.size,
                "model_id": model_name,
                "deception_type": row["deception_type"],
                "injection_placement": row[
                    "injection_placement"
                ],
                "deception_id": row["deception_id"],
                "injected_comment": row["injected_comment"],
                "generated_explanation": explanation,
                "code_used": row["code"],
                "base_code": row["base_code"],
            }

            if "truth_description" in row:
                result_row["truth_description"] = row[
                    "truth_description"
                ]

            results.append(result_row)

        batch_counter += 1

        if batch_counter % SAVE_EVERY_BATCHES == 0:
            pd.DataFrame(results).to_csv(
                output_path,
                index=False,
            )

            print(
                f"--- Checkpoint saved to "
                f"{output_path} ---"
            )

        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

    final_df = pd.DataFrame(results)
    final_df.to_csv(output_path, index=False)

    print("\n[SUCCESS] RQ3 deceptive inference complete.")
    print(f"[SUCCESS] Output saved to: {output_path}\n")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()