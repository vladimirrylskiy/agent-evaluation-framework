import json, sys
sys.path.insert(0, '.')
from experiment_core import build_ground_truth

with open('raw_check/MAD_human_labelled_dataset.json') as f:
    human = json.load(f)

for idx in [0, 5, 10, 15]:
    e = human[idx]
    gt = build_ground_truth(e)
    present = [m for m, v in gt.items() if v]
    print(f'idx={idx} {e["mas_name"]} {e["round"]}: present modes = {present}')
    raw_yes = []
    for a in e['annotations']:
        votes = [a.get('annotator_1'), a.get('annotator_2'), a.get('annotator_3')]
        if sum(1 for v in votes if v) >= 2:
            raw_yes.append(a['failure mode'].split(chr(10))[0][:40])
    print(f'   raw human YES: {raw_yes}')
    print()
