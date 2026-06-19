import json
with open('raw_check/MAD_human_labelled_dataset.json') as f:
    human = json.load(f)

# Round 1 AppWorld 0 — показать ВСЕ режимы с названиями и голосами
e = [x for x in human if x.get('mas_name')=='AppWorld' and x.get('trace_id')==0][0]
print(f'=== {e["mas_name"]} {e["trace_id"]} | {e["round"]} ===')
print(f'{"HUMAN #":8} {"TITLE":50} VOTES')
for a in e['annotations']:
    lines = a['failure mode'].split('\n')
    num = lines[0].split()[0]
    title = ' '.join(lines[0].split()[1:])
    votes = [a.get('annotator_1'), a.get('annotator_2'), a.get('annotator_3')]
    yes = sum(1 for v in votes if v)
    mark = 'YES' if yes >= 2 else ''
    print(f'{num:8} {title[:50]:50} {mark}')
