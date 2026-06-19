import json
with open('raw_check/MAD_human_labelled_dataset.json') as f:
    human = json.load(f)

print(f'Всего человеческих записей: {len(human)}')
print('Поля:', list(human[0].keys()))
print()
e = human[0]
for k, v in e.items():
    s = str(v)
    print(f'{k}: {s[:200]}')
    print('---')
