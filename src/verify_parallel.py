import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import os
import gc

# 1. Paths (Shared across cluster)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "baseline_results.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "verified_baseline.csv")

def run_strong_judge(model_id="meta-llama/Meta-Llama-3.1-70B-Instruct"):
    print(f"--- Initializing 70B Judge on DEEPGPU3 (8x11GB) ---")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # device_map="auto" will shard the model across all 8 GPUs automatically
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        # Using a higher max_memory for GPU 0 just in case of overhead
        max_memory={i: "10GiB" for i in range(8)}
    )

    df = pd.read_csv(INPUT_FILE)
    results = []

    print(f"--- Running Strong Judge on {len(df)} samples ---")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        judge_prompt = f"""[INST] You are a Senior Security Auditor. 
Compare the AI EXPLANATION against the developer-provided GROUND TRUTH.

GROUND TRUTH: {row.get('truth_description', 'N/A')}
AI EXPLANATION: {row['baseline_explanation']}

Does the AI correctly identify the security root cause? 
Answer 'YES' or 'NO' followed by a one-sentence reason. [/INST]"""

        inputs = tokenizer(judge_prompt, return_tensors="pt", truncation=True, max_length=4096).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        is_correct = response.upper().startswith("YES")

        results.append({
            **row.to_dict(),
            'judge_response': response,
            'is_correct': is_correct
        })

        # Memory Cleanup
        del inputs, outputs
        if index % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"--- Verification complete. Saved to {OUTPUT_FILE} ---")

if __name__ == "__main__":
    run_strong_judge()