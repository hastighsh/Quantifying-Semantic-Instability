#!/usr/bin/env python3
import os
import sys
import gc
import argparse

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


torch.backends.cuda.matmul.allow_tf32 = True


DEFAULT_BATCH_SIZE_BY_MODEL = {
    "7B": 4,
    "14B": 2,
    "32B": 1,
}

DEFAULT_MAX_INPUT_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 512
SAVE_EVERY_BATCHES = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ2 Multi-Tier Persona Robustness Inference Pipeline"
    )

    parser.add_argument(
        "--size",
        type=str,
        required=True,
        choices=["7B", "14B", "32B"],
        help="Model size parameter scale",
    )

    parser.add_argument(
        "--persona",
        type=str,
        required=True,
        choices=["naive", "formal", "expert"],
        help="Target system instruction persona",
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


def load_persona_prompt(prompt_path):
    if not os.path.exists(prompt_path):
        print(f"[CRITICAL ERROR] Persona prompt file missing: {prompt_path}")
        sys.exit(1)

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


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


def evaluate_schema(clean_exp):
    required_headers = ["VULNERABILITY:", "ANALYSIS:", "IMPACT:"]
    adheres = all(header in clean_exp for header in required_headers)

    if "VULNERABILITY:" in clean_exp:
        pre_header = clean_exp.split("VULNERABILITY:")[0].strip()
        has_filler = len(pre_header) > 0
    else:
        has_filler = True

    return adheres, has_filler


def normalize_input_dataframe(df):
    df = df.copy()

    if "original_index" not in df.columns:
        if "index" in df.columns:
            df["original_index"] = df["index"]
        else:
            df = df.reset_index().rename(columns={"index": "original_index"})

    if "index" not in df.columns:
        df["index"] = df["original_index"]

    if "cwe" not in df.columns:
        df["cwe"] = "Unknown"

    if "code" not in df.columns:
        print("[CRITICAL ERROR] Input dataset must contain a 'code' column.")
        sys.exit(1)

    df["original_index"] = df["original_index"].astype(int)

    return df


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

    input_path = os.path.join(
        project_root,
        "data",
        "filtered_experimental_set.csv",
    )

    prompt_dir = os.path.join(
        project_root,
        "prompts",
    )

    output_dir = os.path.join(
        project_root,
        "results",
    )

    os.makedirs(output_dir, exist_ok=True)

    model_name = f"Qwen2.5-Coder-{args.size}-Instruct"
    model_source = f"Qwen/{model_name}"

    prompt_file_path = os.path.join(
        prompt_dir,
        f"{args.persona}_persona.txt",
    )

    output_path = os.path.join(
        output_dir,
        f"rq2_{args.persona}_results_qwen{args.size.lower()}.csv",
    )

    if not os.path.exists(input_path):
        print(f"[CRITICAL ERROR] Input dataset missing: {input_path}")
        sys.exit(1)

    input_device = get_input_device()
    cuda_report()

    print("\n=========================================================")
    print("RUNNING RQ2 PERSONA ROBUSTNESS SWEEP")
    print(f"Model Configuration : {model_name}")
    print(f"Target Persona      : {args.persona.upper()}")
    print(f"Model Source        : {model_source}")
    print(f"Input Dataset       : {input_path}")
    print(f"Persona Prompt      : {prompt_file_path}")
    print(f"Destination Path    : {output_path}")
    print(f"Batch Size          : {batch_size}")
    print(f"Max Input Tokens    : {args.max_input_tokens}")
    print(f"Max New Tokens      : {args.max_new_tokens}")
    print("Quantization        : DISABLED")
    print("Precision           : BF16")
    print("=========================================================\n")

    system_instruction = load_persona_prompt(prompt_file_path)

    df = pd.read_csv(input_path, dtype=str)
    df = normalize_input_dataframe(df)
    print(f"Loaded input dataset with {len(df)} rows.")

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

        except Exception as e:
            print(
                f"[WARN] Existing output file is unreadable: "
                f"{repr(e)}"
            )
            print("[WARN] Starting fresh.")
            os.remove(output_path)

    pending_df = df[
        ~df["original_index"].astype(int).isin(done_indices)
    ].copy()

    pending_df = pending_df.reset_index(drop=True)

    print(f"Rows remaining for this run: {len(pending_df)}")

    if pending_df.empty:
        print("[SUCCESS] Nothing to do. Output is already complete.")
        return

    try:
        tokenizer, model = load_model_and_tokenizer(
            model_source
        )
    except Exception as e:
        print(
            f"[CRITICAL ERROR] Model/tokenizer loading failed: "
            f"{repr(e)}"
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
        desc=f"RQ2 Qwen-{args.size} {args.persona}",
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
            clean_exp = extract_generation(
                tokenizer=tokenizer,
                output_ids=outputs[local_i],
                input_len=int(input_lengths[local_i]),
            )

            adheres, has_filler = evaluate_schema(
                clean_exp
            )

            result_row = {
                "index": row["index"],
                "original_index": int(
                    row["original_index"]
                ),
                "cwe": row["cwe"],
                "persona": args.persona,
                "model_tier": args.size,
                "model_id": model_name,
                "generated_explanation": clean_exp,
                "adheres_to_schema": adheres,
                "instructional_drift": has_filler,
                "code_used": row["code"],
            }

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

    final_df.to_csv(
        output_path,
        index=False,
    )

    print("\n[SUCCESS] RQ2 suite complete.")
    print(f"[SUCCESS] Output saved to: {output_path}\n")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()