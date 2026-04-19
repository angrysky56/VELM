import json
import re

path = '/home/ty/Repositories/ai_workspace/VELM/notebooks/velm_colab.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            if 'from google.colab import' in line and '# noqa: E402' not in line:
                # Add noqa: E402
                line = re.sub(r'(\n?)$', r'  # noqa: E402\1', line)
            new_source.append(line)
        cell['source'] = new_source

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)
