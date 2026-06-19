import json
with open('raw_check/MAD_full_dataset.json') as f:
    full = json.load(f)

# взять первый трейс и посмотреть его mast_annotation / метки
e = full[0]
print('Ключи трейса:', list(e.keys()))
print()
# найти поле с аннотациями
for k, v in e.items():
    if 'annot' in k.lower() or 'mast' in k.lower() or 'label' in k.lower() or 'failure' in k.lower():
        print(f'=== поле {k} ===')
        print(json.dumps(v, indent=2)[:2000])
