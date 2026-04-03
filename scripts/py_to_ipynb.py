import nbformat as nbf
import os
import re

def convert_py_to_ipynb(input_py, output_ipynb):
    with open(input_py, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into cells based on # %%
    # Using a simple split, each block starts with # %% (and optional [markdown])
    cells = re.split(r'^# %%', content, flags=re.MULTILINE)

    nb = nbf.v4.new_notebook()

    for i, cell_raw in enumerate(cells):
        if not cell_raw.strip() and i == 0:
            continue

        lines = cell_raw.splitlines()
        if not lines:
            continue

        first_line = lines[0].strip()
        is_markdown = '[markdown]' in first_line

        # Remove the first line if it's just the # %% part (which would be empty or [markdown])
        # Since we split by # %%, the first line here is the suffix of the marker line
        actual_content = "\n".join(lines[1:]).strip()

        if is_markdown:
            # For markdown cells, remove leading '# ' or '#' from each line if it's there
            md_lines = []
            for line in lines[1:]:
                # If it's a markdown cell, it might be commented out in Python
                md_lines.append(re.sub(r'^#\s?', '', line))
            nb.cells.append(nbf.v4.new_markdown_cell("\n".join(md_lines).strip()))
        else:
            nb.cells.append(nbf.v4.new_code_cell(actual_content))

    with open(output_ipynb, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    input_path = "notebooks/velm_colab.py"
    output_path = "notebooks/velm_colab.ipynb"
    if os.path.exists(input_path):
        convert_py_to_ipynb(input_path, output_path)
        print(f"Successfully converted {input_path} to {output_path}")
    else:
        print(f"File {input_path} not found.")
