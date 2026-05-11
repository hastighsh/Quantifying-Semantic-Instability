import pandas as pd
import re
import os

# 1. Path Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Inputs/Outputs
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "filtered_experimental_set.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "perturbed_set.csv")

def rename_variables(code):
    """
    Lexical perturbation: Renames identifiers to var_1, var_2, etc., 
    while ignoring C-style keywords and strings.
    """
    keywords = {
        'int', 'char', 'float', 'double', 'struct', 'if', 'else', 'while', 'for', 
        'return', 'break', 'continue', 'switch', 'case', 'default', 'sizeof', 
        'static', 'const', 'void', 'unsigned', 'signed', 'long', 'short', 'NULL'
    }

    # Pattern: Group 1 captures strings; Group 2 captures potential identifiers
    pattern = r'("[^"]*")|(\b[a-zA-Z_][a-zA-Z0-9_]*\b)'

    # 1. First pass: Identify variables to rename
    identifiers = set()
    for match in re.finditer(pattern, code):
        if match.group(2):  # If it's a word and not a string
            word = match.group(2)
            if word not in keywords and len(word) > 1:
                identifiers.add(word)

    # 2. Create mapping (sorted by length descending to prevent partial replacement)
    targets = sorted(list(identifiers), key=len, reverse=True)
    mapping = {old: f"var_{i+1}" for i, old in enumerate(targets)}

    # 3. Second pass: Replace only Group 2 matches found in mapping
    def replace_func(match):
        if match.group(1): 
            return match.group(1)  # Return strings untouched
        word = match.group(2)
        return mapping.get(word, word)

    return re.sub(pattern, replace_func, code)

def transform_logic(code, index):
    """
    Applies structural changes. 
    Even rows get 'while' conversion, odd rows stay 'for' for 50/50 split.
    """
    # Regex to capture for(init; cond; inc) {
    for_pattern = r'for\s*\(([^;]*);([^;]*);([^)]*)\)\s*\{'
    
    def for_replacer(match):
        init, cond, inc = match.groups()
        # Guaranteed 50/50 split based on the row number
        if index % 2 == 0: 
            return f"{init.strip()};\n    while({cond.strip()}) {{\n        {inc.strip()};"
        else:
            return match.group(0)

    code = re.sub(for_pattern, for_replacer, code)

    # Ternary transformation (applied to all applicable blocks)
    ternary_pattern = r'if\s*\(([^)]+)\)\s*{\s*(\w+)\s*=\s*([^;]+);\s*}\s*else\s*{\s*\2\s*=\s*([^;]+);\s*}'
    code = re.sub(ternary_pattern, r'\2 = (\1) ? \3 : \4;', code)

    return code

def run_perturbation():
    print("--- Starting Perturbation Stage ---")
    
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Could not find {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    perturbed_list = []

    print(f"Processing {len(df)} samples...")

    for idx, row in df.iterrows():
        # 1. Structural change (passes index for consistency)
        code_transformed = transform_logic(row['code'], idx)
        
        # 2. Lexical change (Variable renaming)
        code_final = rename_variables(code_transformed)
        
        perturbed_list.append(code_final)

    df['perturbed_code'] = perturbed_list

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    
    print("-" * 30)
    print("--- Success! Perturbations applied ---")
    print(f"Total samples: {len(df)}")
    print(f"Saved to: {OUTPUT_PATH}")

    # Preview first sample
    print("\n--- PREVIEW (Sample 0) ---")
    print("ORIGINAL (Start):\n", df.iloc[0]['code'][:100], "...")
    print("\nPERTURBED (Start):\n", df.iloc[0]['perturbed_code'][:100], "...")

if __name__ == "__main__":
    run_perturbation()