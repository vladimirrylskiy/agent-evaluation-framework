import json, os

with open('raw_check/MAD_human_labelled_dataset.json') as f:
    human = json.load(f)
with open('raw_check/MAD_full_dataset.json') as f:
    full = json.load(f)

# human ChatDev id=3
h = [e for e in human if e.get('mas_name')=='ChatDev' and e.get('trace_id')==3][0]
with open('raw_check/HUMAN_chatdev_3.txt', 'w') as out:
    out.write(str(h.get('trace')))

# both full ChatDev id=3
f3 = [e for e in full if e.get('mas_name')=='ChatDev' and e.get('trace_id')==3]
for i, e in enumerate(f3):
    traj = e['trace'].get('trajectory') if isinstance(e.get('trace'), dict) else e.get('trace')
    with open(f'raw_check/FULL_chatdev_3_{i}.txt', 'w') as out:
        out.write(str(traj))

print('Saved txt files in raw_check/:')
for fn in sorted(os.listdir('raw_check')):
    if fn.endswith('.txt'):
        print(' ', fn, os.path.getsize(f'raw_check/{fn}'), 'bytes')
