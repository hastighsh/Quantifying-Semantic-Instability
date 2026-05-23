import os
import pandas as pd
from tree_sitter import Language, Parser
import tree_sitter_c

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "filtered_experimental_set.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "perturbed_set.csv")

C_LANGUAGE = Language(tree_sitter_c.language())
parser = Parser(C_LANGUAGE)


def get_node_text(node, source_bytes):
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def collect_identifiers_from_declarator(node, source_bytes, out_set):
    """
    Recursively unwrap declarators until we reach the actual local identifier name.
    """
    if node is None:
        return

    if node.type == "identifier":
        out_set.add(get_node_text(node, source_bytes))
        return

    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        collect_identifiers_from_declarator(declarator, source_bytes, out_set)


def ast_scope_pure_perturbation(code_str, index):
    source_bytes = code_str.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    declared_names = set()

    # PASS 1: COLLECT TRUE LOCAL VARIABLES + PARAMETERS
    def find_declarations(node):
        if node.type == "parameter_declaration":
            declarator = node.child_by_field_name("declarator")
            collect_identifiers_from_declarator(declarator, source_bytes, declared_names)

        elif node.type == "declaration":
            declarator = node.child_by_field_name("declarator")
            if declarator is not None:
                collect_identifiers_from_declarator(declarator, source_bytes, declared_names)

            for child in node.children:
                if child.type == "init_declarator":
                    declarator = child.child_by_field_name("declarator")
                    collect_identifiers_from_declarator(declarator, source_bytes, declared_names)

        for child in node.children:
            find_declarations(child)

    find_declarations(root)

    # BUILD RENAME MAP
    rename_map = {
        old_name: f"var_{i + 1}"
        for i, old_name in enumerate(sorted(declared_names))
    }

    # PASS 2: FIND SAFE IDENTIFIER USAGES
    mutation_targets = []

    def should_skip_identifier(node):
        parent = node.parent
        if parent is None:
            return False

        # 1. Never rename function names inside call expressions
        if parent.type == "call_expression":
            if parent.child_by_field_name("function") == node:
                return True

        # 2. FIXED NESTED STRUCT ACCESS PROTECTION:
        # Trace up through all field_expression parents. If our identifier node 
        # is EVER found acting as a field name rather than the base argument 
        # object at any level in the chain, skip it completely.
        current = node
        while current.parent is not None:
            p = current.parent
            if p.type == "field_expression":
                # If it's not the base object of this specific field expression, it's a property.
                if p.child_by_field_name("argument") != current:
                    return True
            current = p

        return False

    def collect_usage_targets(node):
        if node.type == "identifier":
            name = get_node_text(node, source_bytes)
            if name in rename_map:
                if not should_skip_identifier(node):
                    mutation_targets.append((node.start_byte, node.end_byte, name))

        for child in node.children:
            collect_usage_targets(child)

    collect_usage_targets(root)

    # APPLY MUTATIONS BACKWARDS
    mutable = bytearray(source_bytes)
    unique_targets = sorted(
        set(mutation_targets),
        key=lambda x: x[0],
        reverse=True
    )

    for start, end, old_name in unique_targets:
        new_name = rename_map[old_name]
        mutable[start:end] = new_name.encode("utf-8")

    output_code = mutable.decode("utf-8", errors="ignore")

    # OPTIONAL LOOP PERTURBATION
    if index % 2 == 0:
        output_code = loop_inversion_ast(output_code)

    return output_code


def loop_inversion_ast(code_str):
    source_bytes = code_str.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    loops = []
    def collect_loops(node):
        if node.type == "for_statement":
            loops.append(node)
        for child in node.children:
            collect_loops(child)

    collect_loops(root)
    updated = code_str

    for loop in sorted(loops, key=lambda n: n.start_byte, reverse=True):
        current_bytes = updated.encode("utf-8")
        current_tree = parser.parse(current_bytes)
        current_root = current_tree.root_node

        match = None
        def locate(node):
            nonlocal match
            if match is not None:
                return
            if node.type == "for_statement" and node.start_byte == loop.start_byte:
                match = node
                return
            for child in node.children:
                locate(child)

        locate(current_root)

        if match is None:
            continue

        init = match.child_by_field_name("initializer")
        cond = match.child_by_field_name("condition")
        update = match.child_by_field_name("update")
        body = match.child_by_field_name("body")

        if not (init and cond and update and body):
            continue

        init_t = get_node_text(init, current_bytes)
        cond_t = get_node_text(cond, current_bytes)
        update_t = get_node_text(update, current_bytes)
        body_t = get_node_text(body, current_bytes)

        body_inner = body_t[1:-1].strip() if body_t.startswith("{") and body_t.endswith("}") else body_t.strip()

        replacement = (
            f"{init_t};\n"
            f"while ({cond_t}) {{\n"
            f"    {body_inner}\n"
            f"    {update_t};\n"
            f"}}"
        )

        updated = updated[:match.start_byte] + replacement + updated[match.end_byte:]

    return updated


def main():
    print("\n--- Running Scope-Corrected AST Mutation Engine ---\n")
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Missing file: {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    perturbed = []

    for idx, row in df.iterrows():
        try:
            mutated = ast_scope_pure_perturbation(str(row["code"]), idx)
        except Exception as e:
            print(f"FAILED ROW {idx}: {e}")
            mutated = row["code"]
        perturbed.append(mutated)

    df["perturbed_code"] = perturbed
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSUCCESS: Compiled pristine experimental workspace at:\n{OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
