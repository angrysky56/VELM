import json

path = '/home/ty/Repositories/ai_workspace/VELM/notebooks/velm_colab.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

cell = nb['cells'][4]
source = cell['source']

new_source = []
found_rdt = False
skip_rdt = False

for line in source:
    if '# Mythos-Enhanced RDT Configuration' in line:
        if not found_rdt:
            new_source.append(line)
            found_rdt = True
        else:
            skip_rdt = True
    elif skip_rdt:
        if 'cfg["n_loops"]' in line or 'cfg["use_act"]' in line or 'print(f"  Mythos RDT:' in line:
            continue
        elif line.strip() == '':
            continue
        else:
            skip_rdt = False
            new_source.append(line)
    else:
        new_source.append(line)

cell['source'] = new_source

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)
