import json

path = '/home/ty/Repositories/ai_workspace/VELM/notebooks/velm_colab.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

# Cell 2 is at index 3 (0-indexed: Cell 0=0, markdown=1, Cell 1=2, Cell 2=3)
# Let's verify it's the right cell
cell = nb['cells'][4] # Wait, I need to check the exact index.
# I'll iterate and find the one with "Hardware-adaptive config"

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('Hardware-adaptive config' in s for s in cell['source']):
        print(f"Found at index {i}")
        source = cell['source']
        
        new_source = []
        seen_rdt = False
        skip = False
        
        for line in source:
            if '# Mythos-Enhanced RDT Configuration' in line:
                if seen_rdt:
                    skip = True
                else:
                    seen_rdt = True
                    new_source.append(line)
            elif skip and ('cfg["n_loops"]' in line or 'cfg["use_act"]' in line or 'print(f"  Mythos RDT:' in line):
                continue
            elif skip and line.strip() == '':
                continue
            else:
                skip = False
                new_source.append(line)
        
        cell['source'] = new_source
        break

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)
