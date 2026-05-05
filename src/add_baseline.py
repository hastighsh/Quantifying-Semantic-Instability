import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import os
import gc

# 1. Paths (Cluster Local)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "data", "gold_set_raw.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "baseline_results.csv")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def run_baseline_inference(model_id="deepseek-ai/deepseek-coder-7b-instruct-v1.5"):
    print(f"--- Loading Model: {model_id} ---")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto", 
        trust_remote_code=True
    )

    # Load gold set from cluster local path
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found!")
        return

    df = pd.read_csv(INPUT_PATH)
    results = []

    print(f"--- Starting Baseline Inference on {len(df)} samples ---")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        code = row['code']
        cwe = row.get('CWE ID', row.get('cwe', 'Unknown'))
        cve = row.get('CVE ID', row.get('cve', 'Unknown'))
        truth = row.get('truth_description', 'N/A')

        # PROMPT: Chain-of-Thought & Security Auditor Persona
        prompt = f"""[INST] You are an expert C security researcher. Analyze the following code for vulnerabilities.
                    Structure your response as follows:
                    1. Logic Flow: Briefly describe what the code does.
                    2. Root Cause: Identify the exact line or logic that is vulnerable.
                    3. Danger: Explain why this is dangerous.

                    CODE:
                    {code}

                    Final Explanation: [/INST]"""
        
        # Tokenize with a safety margin for long C functions
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2560).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                # Explicitly set the pad token to avoid warnings
                pad_token_id=tokenizer.eos_token_id 
            )

        explanation = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        results.append({
            'index': index,
            'cwe': cwe,
            'cve': cve,
            'truth_description': truth,
            'code': code,
            'baseline_explanation': explanation.strip()
        })
        
        # Periodic memory cleanup for long runs
        del inputs
        del outputs
        gc.collect()
        torch.cuda.empty_cache()

    # Save to cluster
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"--- Baseline results saved to {OUTPUT_PATH} ---")

if __name__ == "__main__":
    run_baseline_inference()