import json

path = '/home/ty/Repositories/ai_workspace/VELM/notebooks/velm_colab.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('Hardware-adaptive config' in s for s in cell['source']):
        source = cell['source']
        
        # Define the block we want to keep once
        block_to_keep = [
            "# Mythos-Enhanced RDT Configuration\n",
            "cfg[\"n_loops\"] = cfg.get(\"n_loops\", 1)\n",
            "cfg[\"use_act\"] = cfg.get(\"use_act\", False)\n",
            "print(f\"  Mythos RDT: n_loops={cfg['n_loops']} | use_act={cfg['use_act']}\")\n"
        ]
        
        new_source = []
        found_block = False
        i = 0
        while i < len(source):
            # Check if this line starts a block
            if '# Mythos-Enhanced RDT Configuration' in source[i]:
                if not found_block:
                    # Keep this first occurrence
                    new_source.append(source[i])
                    found_block = True
                    i += 1
                else:
                    # Skip this and the next 3 lines if they match the pattern
                    i += 1
                    while i < len(source) and (
                        'cfg["n_loops"]' in source[i] or 
                        'cfg["use_act"]' in source[i] or 
                        'print(f"  Mythos RDT:' in source[i] or
                        source[i].strip() == ''
                    ):
                        if i < len(source) and '# Mythos-Enhanced RDT Configuration' in source[i]:
                            break # Don't skip a new header, let the outer loop handle it
                        i += 1
            else:
                new_source.append(source[i])
                i += 1
        
        cell['source'] = new_source
        break

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)
