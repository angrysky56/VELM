import json

path = '/home/ty/Repositories/ai_workspace/VELM/notebooks/velm_colab.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

found = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('Hardware-adaptive config' in s for s in cell['source']):
        print(f"Found Hardware-adaptive config at index {i}")
        found = True
        source = cell['source']
        
        new_source = []
        found_block = False
        j = 0
        while j < len(source):
            if '# Mythos-Enhanced RDT Configuration' in source[j]:
                if not found_block:
                    print(f"Keeping first occurrence at line {j}")
                    new_source.append(source[j])
                    found_block = True
                    j += 1
                else:
                    print(f"Skipping redundant block starting at line {j}")
                    j += 1
                    while j < len(source) and not ('# Mythos-Enhanced RDT Configuration' in source[j]) and (
                        'cfg["n_loops"]' in source[j] or 
                        'cfg["use_act"]' in source[j] or 
                        'print(f"  Mythos RDT:' in source[j] or
                        source[j].strip() == ''
                    ):
                        j += 1
            else:
                new_source.append(source[j])
                j += 1
        
        cell['source'] = new_source
        break

if not found:
    print("Could not find cell with 'Hardware-adaptive config'")

with open(path, 'w') as f:
    json.dump(nb, f, indent=2) # Use indent=2 for standard look
