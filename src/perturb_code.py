import os
import re
import pandas as pd
from tree_sitter import Language, Parser
import tree_sitter_c

# 1. Structural Path Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "filtered_experimental_set.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "perturbed_set.csv")

# Initialize Tree-Sitter Parser for C
C_LANGUAGE = Language(tree_sitter_c.language())
parser = Parser(C_LANGUAGE)


def get_node_text(node, source_bytes):
    """Safely extracts text string from a specific AST node range."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def ast_rename_variables(code_str):
    """
    AST-based Lexical Perturbation: Maps and renames local variable identifiers
    while ignoring standard C keywords, functions, macros, and string literals.
    """
    source_bytes = code_str.encode("utf-8")
    tree = parser.parse(source_bytes)
    root_node = tree.root_node

    keywords = {
        'int', 'char', 'float', 'double', 'struct', 'if', 'else', 'while', 'for', 
        'return', 'break', 'continue', 'switch', 'case', 'default', 'sizeof', 
        'static', 'const', 'void', 'unsigned', 'signed', 'long', 'short', 'NULL'
    }

    identifiers_to_rename = set()

    def traverse_for_identifiers(node):
        # We target specific identifier types while ensuring we don't rename system function names
        if node.type == "identifier":
            parent = node.parent
            # Skip if it's a function declaration identifier or an external macro call name
            if parent and parent.type in ["function_declarator", "call_expression"] and parent.child_by_field_name("name") == node:
                pass
            else:
                word = get_node_text(node, source_bytes)
                if word not in keywords and len(word) > 1 and not word.isupper():
                    identifiers_to_rename.add((node.start_byte, node.end_byte, word))
        
        for child in node.children:
            traverse_for_identifiers(child)

    traverse_for_identifiers(root_node)

    # Sort identifiers by reverse byte order to update text from back to front without misaligning indices
    sorted_occurrences = sorted(list(identifiers_to_rename), key=lambda x: x[0], reverse=True)
    
    # Create distinct mappings for old variable names
    unique_names = sorted(list(set([x[2] for x in sorted_occurrences])))
    var_mapping = {old: f"var_{i+1}" for i, old in enumerate(unique_names)}

    # Modify the string via byte slices
    modified_bytes = bytearray(source_bytes)
    for start_byte, end_byte, old_name in sorted_occurrences:
        new_name = var_mapping[old_name]
        modified_bytes[start_byte:end_byte] = new_name.encode("utf-8")

    return modified_bytes.decode("utf-8")


def ast_transform_loops(code_str, index):
    """
    AST-based Logic Transformation: Identifies 'for' loops structures natively
    and refactors them into logically identical 'while' structures for even rows.
    """
    # Maintain a 50/50 balance across dataset slices based on row evaluation indices
    if index % 2 != 0:
        return code_str

    source_bytes = code_str.encode("utf-8")
    tree = parser.parse(source_bytes)
    root_node = tree.root_node

    for_loops = []

    def find_for_loops(node):
        if node.type == "for_statement":
            for_loops.append(node)
        for child in node.children:
            find_for_loops(child)

    find_for_loops(root_node)

    if not for_loops:
        return code_str

    # Process back to front to ensure character positions remain aligned
    modified_code = code_str
    for loop_node in sorted(for_loops, key=lambda n: n.start_byte, reverse=True):
        # Extract init, condition, update, and body blocks
        init_node = loop_node.child_by_field_name("initializer")
        cond_node = loop_node.child_by_field_name("condition")
        update_node = loop_node.child_by_field_name("update")
        body_node = loop_node.child_by_field_name("body")

        if not (init_node and cond_node and update_node and body_node):
            continue

        loop_bytes = modified_code.encode("utf-8")
        init_txt = get_node_text(init_node, loop_bytes)
        cond_txt = get_node_text(cond_node, loop_bytes)
        update_txt = get_node_text(update_node, loop_bytes)
        body_txt = get_node_text(body_node, loop_bytes)

        # Handle inner content braces cleaning
        if body_txt.startswith("{") and body_txt.endswith("}"):
            body_core = body_txt[1:-1].strip()
        else:
            body_core = body_txt.strip()

        # Re-synthesize structurally accurate C code block
        while_structure = (
            f"{init_txt};\n"
            f"while ({cond_txt}) {{\n"
            f"    {body_core}\n"
            f"    {update_txt};\n"
            f"}}"
        )
        
        start, end = loop_node.start_byte, loop_node.end_byte
        modified_code = modified_code[:start] + while_structure + modified_code[end:]

    return modified_code


def regex_ternary_fallback(code):
    """
    Robust Regex-based fallback pattern to compress standard conditional variable
    assignments into concise inline ternary variants.
    """
    ternary_pattern = r'if\s*\(([^)]+)\)\s*{\s*(\w+)\s*=\s*([^;]+);\s*}\s*else\s*{\s*\2\s*=\s*([^;]+);\s*}'
    return re.sub(ternary_pattern, r'\2 = (\1) ? \3 : \4;', code)


def run_perturbation():
    print("=" * 60)
    print("--- RUNNING ADVANCED AST PERTURBATION FRAMEWORK ---")
    print("=" * 60)
    
    if not os.path.exists(INPUT_PATH):
        print(f"[CRITICAL ERROR] Target input workspace missing at: {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    perturbed_list = []

    print(f"Imported {len(df)} pristine high-quality baseline rows.")

    for idx, row in df.iterrows():
        original_code = row['code']
        
        try:
            # Step 1: Structural Loop Transformation (AST Engine)
            code_step1 = ast_transform_loops(original_code, idx)
            
            # Step 2: Idiomatic Assignment Minimization (Regex Filter)
            code_step2 = regex_ternary_fallback(code_step1)
            
            # Step 3: Local Scope Variable Scrambling (AST Engine)
            code_final = ast_rename_variables(code_step2)
            
            perturbed_list.append(code_final)
        except Exception as err:
            print(f"[WARNING] AST failure on row idx {idx}, employing strict fallback routing. Error: {err}")
            perturbed_list.append(original_code)

    df['perturbed_code'] = perturbed_list

    # Output Management
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print("-" * 60)
    print("--- SUCCESS: PERTURBED DATA EXPERIMENTAL WORKSPACE COMPILED ---")
    print(f"Total Records Generated: {len(df)}")
    print(f"Saved directly to target path: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    run_perturbation()