import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from torch.utils.data import Dataset
from tqdm import tqdm
import gc
import sys

# 1. LOCAL CONFIGURATION
PROJECT_ROOT = os.getcwd()
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "deceptive_experimental_set.csv")
PROMPT_PATH = os.path.join(PROJECT_ROOT, "prompts", "formal_persona.txt")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "rq3_deceptive_results.csv")
MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"

# 2. DATASET CLASS FOR BATCHING
class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list
    def __len__(self):
        return len(self.original_list)
    def __getitem__(self, i):
        return self.original_list[i]

def run_rq3_deceptive_inference():
    print("--- Starting RQ3 Deceptive Context Inference (Batch Mode) ---")
    
    # Check paths
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Deceptive dataset not found at {INPUT_PATH}")
        return
    if not os.path.exists(PROMPT_PATH):
        print(f"ERROR: Prompt not found at {PROMPT_PATH}")
        return

    # Load Data & Persona
    df = pd.read_csv(INPUT_PATH)
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        system_instruction = f.read().strip()

    # BitsAndBytes Config
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

    print(f"--- Running Batch Inference on {len(df)} samples ---")
    generated_results = []
    
    for out in tqdm(gen_pipeline(dataset, batch_size=4, max_new_tokens=400, 
                                 temperature=0.1, do_sample=True, 
                                 repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)):
        
        full_text = out[0]['generated_text']
        # Extract everything after the prompt header
        explanation = full_text.split("Explanation:")[-1].strip()
        generated_results.append(explanation)

    # Final Save
    df['generated_explanation'] = generated_results
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"--- Success! RQ3 Results saved to: {OUTPUT_PATH} ---")
    
    # Cleanup VRAM
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_rq3_deceptive_inference()