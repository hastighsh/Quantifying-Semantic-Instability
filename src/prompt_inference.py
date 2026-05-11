import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from torch.utils.data import Dataset
from tqdm import tqdm
import gc

# 1. PATH SETUP
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "filtered_experimental_set.csv")
PROMPT_DIR = os.path.join(PROJECT_ROOT, "prompts")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"

# 2. DATASET CLASS FOR BATCHING
class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list
    def __len__(self):
        return len(self.original_list)
    def __getitem__(self, i):
        return self.original_list[i]

def run_rq2_batch_pipeline(persona_choice):
    print(f"\n--- Starting RQ2 {persona_choice.upper()} (Transformers Batch Mode) ---")
    
    # Load Data & Prompt
    df = pd.read_csv(INPUT_PATH)
    with open(os.path.join(PROMPT_DIR, f"{persona_choice}_persona.txt"), 'r') as f:
        system_instruction = f.read().strip()

    # Model Configuration 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16 
    )

    print(f"--- Loading Model: {MODEL_ID} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # Initialize Pipeline
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    # Prepare Prompts
    prompt_list = [f"{system_instruction}\n\nCODE:\n{row['code']}\n\nExplanation:" for _, row in df.iterrows()]
    dataset = ListDataset(prompt_list)

    print(f"--- Running Inference on {len(df)} samples ---")
    results = []
    
    for out in tqdm(gen_pipeline(dataset, batch_size=4, max_new_tokens=512, 
                                 temperature=0.1, do_sample=True, 
                                 repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)):
        
        explanation = out[0]['generated_text']
        # Extract only the generated part (after the prompt)
        clean_exp = explanation.split("Explanation:")[-1].strip()
        
        # Schema Checks
        required_headers = ['VULNERABILITY:', 'ANALYSIS:', 'IMPACT:']
        adheres = all(h in clean_exp for h in required_headers)
        
        # Instructional Drift Check
        has_filler = False
        if "VULNERABILITY:" in clean_exp:
            pre_header = clean_exp.split("VULNERABILITY:")[0].strip()
            has_filler = len(pre_header) > 0

        results.append({
            'generated_explanation': clean_exp,
            'adheres_to_schema': adheres,
            'instructional_drift': has_filler
        })

    res_df = pd.DataFrame(results)
    final_df = pd.concat([df[['index', 'cwe']].reset_index(drop=True), res_df], axis=1)
    final_df['persona'] = persona_choice

    output_path = os.path.join(OUTPUT_DIR, f"rq2_{persona_choice}_results.csv")
    final_df.to_csv(output_path, index=False)
    print(f"--- Process Complete. Saved to {output_path} ---")

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
        print(f"Using persona from argument: {choice}")
    else:
        choice = input("Enter persona (naive/formal/expert): ").strip().lower()
    
    run_rq2_batch_pipeline(choice)