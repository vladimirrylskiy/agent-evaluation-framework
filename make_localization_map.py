import csv
from collections import defaultdict

with open('results/localization_map.csv') as f:
    rows = list(csv.DictReader(f))

# для каждого режима: сколько раз steps / global / none (среди baseline=1)
mode_stats = defaultdict(lambda: {'steps':0, 'global':0, 'none':0, 'baseline_yes':0})
for r in rows:
    md = r['mode']
    rt = r['result_type']
    if r.get('baseline_verdict') == '1':
        mode_stats[md]['baseline_yes'] += 1
        if rt in ('steps','global','none'):
            mode_stats[md][rt] += 1

def mk(x): a,b = x.split('.'); return (int(a),int(b))
print('LOCALIZABILITY MAP — per mode (among baseline-confirmed cases)')
print(f'{"Mode":6} {"BaselineYES":>11} {"Steps":>7} {"Global":>7} {"None":>6} {"Verdict":>12}')
for md in sorted(mode_stats, key=mk):
    s = mode_stats[md]
    by = s['baseline_yes']
    if by == 0:
        verdict = 'never detected'
    elif s['global'] > s['steps']:
        verdict = 'GLOBAL'
    elif s['steps'] > 0:
        verdict = 'STEP-LOCAL'
    else:
        verdict = 'unclear'
    print(f'{md:6} {by:>11} {s["steps"]:>7} {s["global"]:>7} {s["none"]:>6} {verdict:>12}')

# экспорт
with open('results/localizability_map.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['mode','baseline_yes','steps','global','none','verdict'])
    for md in sorted(mode_stats, key=mk):
        s = mode_stats[md]
        by = s['baseline_yes']
        verdict = ('never' if by==0 else 'GLOBAL' if s['global']>s['steps']
                   else 'STEP-LOCAL' if s['steps']>0 else 'unclear')
        w.writerow([md, by, s['steps'], s['global'], s['none'], verdict])
print('\n✅ results/localizability_map.csv')
