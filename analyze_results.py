import csv
from collections import defaultdict

# ── загрузка ────────────────────────────────────────────────
with open('results/human_validation.csv') as f:
    rows = list(csv.DictReader(f))

# уникальные (trace, model) → agreement_pct (берём из любой строки группы)
trace_model_agree = {}
for r in rows:
    key = (r['trace'], r['mas_name'], r['model'])
    trace_model_agree[key] = float(r['agreement_pct'])

models = sorted({m for (_, _, m) in trace_model_agree})

# ── 1. ГЛАВНАЯ ТАБЛИЦА: trace × model + средние ─────────────
print('=' * 70)
print('1. MAIN TABLE — agreement per trace per model')
print('=' * 70)
traces = sorted({(t, mas) for (t, mas, _) in trace_model_agree})
print(f'{"Trace":24} ' + ' '.join(f'{m:>16}' for m in models))
for (t, mas) in traces:
    cells = []
    for m in models:
        v = trace_model_agree.get((t, mas, m))
        cells.append(f'{v:>15.1f}%' if v is not None else f'{"-":>16}')
    print(f'{t:24} ' + ' '.join(cells))
# средние по модели
print('-' * 70)
for m in models:
    vals = [v for (t, mas, mm), v in trace_model_agree.items() if mm == m]
    print(f'MEAN {m}: {sum(vals)/len(vals):.1f}%  (n={len(vals)})')

# ── 2. PER-FRAMEWORK средние ────────────────────────────────
print('\n' + '=' * 70)
print('2. PER-FRAMEWORK mean agreement')
print('=' * 70)
fw = defaultdict(lambda: defaultdict(list))
for (t, mas, m), v in trace_model_agree.items():
    fw[mas][m].append(v)
print(f'{"Framework":14} ' + ' '.join(f'{m:>16}' for m in models) + '   n_traces')
for mas in sorted(fw):
    cells = []
    for m in models:
        vals = fw[mas][m]
        cells.append(f'{sum(vals)/len(vals):>15.1f}%' if vals else f'{"-":>16}')
    n = len(fw[mas][models[0]])
    print(f'{mas:14} ' + ' '.join(cells) + f'   {n}')

# ── 3. PER-MODE разбивка: где судья ошибается ───────────────
print('\n' + '=' * 70)
print('3. PER-MODE judge accuracy (match rate across all traces/models)')
print('=' * 70)
# для каждого mode: сколько match=1 из всех (trace×model) где mode сравнивался
mode_stats = defaultdict(lambda: {'match': 0, 'total': 0,
                                   'human_yes': 0, 'judge_yes': 0,
                                   'false_pos': 0, 'false_neg': 0})
for r in rows:
    mode = r['mode']
    s = mode_stats[mode]
    h = int(r['human']); j = int(r['judge']); m = int(r['match'])
    s['total'] += 1
    s['match'] += m
    s['human_yes'] += h
    s['judge_yes'] += j
    if h == 0 and j == 1: s['false_pos'] += 1   # судья нашёл, человек нет
    if h == 1 and j == 0: s['false_neg'] += 1   # судья пропустил

print(f'{"Mode":6} {"Accuracy":>9} {"HumanYES":>9} {"JudgeYES":>9} {"FalsePos":>9} {"FalseNeg":>9}')
def mode_key(x):
    a, b = x.split('.'); return (int(a), int(b))
for mode in sorted(mode_stats, key=mode_key):
    s = mode_stats[mode]
    acc = 100 * s['match'] / s['total']
    print(f'{mode:6} {acc:>8.1f}% {s["human_yes"]:>9} {s["judge_yes"]:>9} '
          f'{s["false_pos"]:>9} {s["false_neg"]:>9}')

# ── экспорт CSV для таблиц/графиков в тезис ─────────────────
with open('results/summary_per_framework.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['framework'] + models + ['n_traces'])
    for mas in sorted(fw):
        row = [mas] + [f'{sum(fw[mas][m])/len(fw[mas][m]):.1f}' for m in models]
        row.append(len(fw[mas][models[0]]))
        w.writerow(row)

with open('results/summary_per_mode.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['mode', 'accuracy_pct', 'human_yes', 'judge_yes', 'false_pos', 'false_neg'])
    for mode in sorted(mode_stats, key=mode_key):
        s = mode_stats[mode]
        w.writerow([mode, f'{100*s["match"]/s["total"]:.1f}',
                    s['human_yes'], s['judge_yes'], s['false_pos'], s['false_neg']])

print('\n✅ Exported: results/summary_per_framework.csv, results/summary_per_mode.csv')
