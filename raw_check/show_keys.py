import json
with open('raw_check/MAD_full_dataset.json') as f:
    full = json.load(f)

print("Все уникальные ChatDev эксперименты (key) и сколько трейсов в каждом:")
from collections import Counter
keys = Counter()
for e in full:
    if e.get('mas_name')=='ChatDev':
        tr = e.get('trace')
        k = tr.get('key') if isinstance(tr, dict) else None
        keys[k] += 1
for k, n in keys.items():
    print(f'  {k}: {n} traces')

print()
print("ChatDev trace_id=3 — все кандидаты с их уникальным ключом:")
for e in full:
    if e.get('mas_name')=='ChatDev' and e.get('trace_id')==3:
        tr = e.get('trace')
        k = tr.get('key') if isinstance(tr, dict) else None
        idx = tr.get('index') if isinstance(tr, dict) else None
        print(f'  key={k}, index={idx}')
