#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import os
import gc


# In[ ]:


# 1. Path Configuration - Local Cluster Setup
BASE_DIR = "./data/data"
INPUT_FILE = os.path.join(BASE_DIR, "baseline_results.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "verified_baseline.csv")

os.makedirs(BASE_DIR, exist_ok=True)


# In[ ]:


def cleanup_memory():
    """Clear GPU cache and garbage collect."""
    if 'model' in locals():
        del model
    if 'tokenizer' in locals():
        del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# In[ ]:


def run_verification(model_id="deepseek-ai/deepseek-coder-7b-instruct-v1.5"):
    cleanup_memory()
    
    print(f"--- Loading Judge Model: {model_id} ---")
    
    # 4-bit Config for Cluster GPUs
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
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # Load results from the local folder
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Could not find {INPUT_FILE}. Please ensure the baseline results are in the data/data folder.")
        return

    df = pd.read_csv(INPUT_FILE)
    verified_results = []

    print(f"--- Starting Verification Process for {len(df)} samples ---")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        ground_truth_block = f"""
        - Vulnerability Type: {row.get('cwe', 'Unknown')}
        - Reference ID: {row.get('cve', 'Unknown')}
        - Developer Commit Message (Root Cause): {row.get('truth_description', 'No description available')}
        """

        judge_prompt = f"""[INST] You are a Senior Security Auditor. 
Verify if the AI's explanation correctly identifies the specific logical root cause described in the GROUND TRUTH.

CODE: 
{row['code']}

GROUND TRUTH: 
{ground_truth_block}

AI EXPLANATION: 
{row['baseline_explanation']}

CRITERIA:
1. Does the AI identify the correct variable/function failure mentioned in the Commit Message?
2. Does the AI match the vulnerability type (CWE)?
3. Ignore general "best practice" advice if it is NOT the core issue in the Commit Message.

Does the AI EXPLANATION accurately identify the specific vulnerability? 
Answer ONLY 'YES' or 'NO' followed by a one-sentence reason. [/INST]"""
        
        inputs = tokenizer(judge_prompt, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150, 
                temperature=0.1, 
                do_sample=False # Explicitly set for deterministic judging
            )
        
        judge_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        # Determine correctness
        is_correct = "YES" in judge_response.upper()[:10]

        verified_results.append({
            **row.to_dict(),
            'judge_response': judge_response,
            'is_correct': is_correct
        })

    # Save to cluster storage
    output_df = pd.DataFrame(verified_results)
    output_df.to_csv(OUTPUT_FILE, index=False)
    
    pass_count = output_df['is_correct'].sum()
    print(f"\n--- Verification Complete! {pass_count}/{len(df)} samples passed. ---")
    print(f"Verified data saved to {OUTPUT_FILE}")

    # Display summary of types
    passed_df = output_df[output_df['is_correct'] == True]
    if not passed_df.empty:
        print("\nSummary of Validated Bugs by CWE:")
        print(passed_df['cwe'].value_counts())


# In[ ]:


if __name__ == "__main__":
    run_verification()

