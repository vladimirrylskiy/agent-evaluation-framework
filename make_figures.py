import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ════════════════════════════════════════════════════════════
# ЗАГРУЗКА
# ════════════════════════════════════════════════════════════
with open('results/human_validation.csv') as f:
    rows = list(csv.DictReader(f))

# (trace, framework, model) -> agreement
tm = {}
for r in rows:
    tm[(r['trace'], r['mas_name'], r['model'])] = float(r['agreement_pct'])
models = sorted({m for (_, _, m) in tm})
traces = sorted({(t, mas) for (t, mas, _) in tm})

# ════════════════════════════════════════════════════════════
# ФИГУРА 1: agreement по трейсам (flash vs pro)
# ════════════════════════════════════════════════════════════
labels = [t for (t, mas) in traces]
x = np.arange(len(traces))
w = 0.38
fig, ax = plt.subplots(figsize=(13, 6))
for i, m in enumerate(models):
    vals = [tm.get((t, mas, m), 0) for (t, mas) in traces]
    ax.bar(x + (i - 0.5) * w, vals, w, label=m)
    mean = np.mean([v for v in vals if v > 0])
    ax.axhline(mean, ls='--', lw=1, color=f'C{i}', alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
ax.set_ylabel('Judge\u2013human agreement (%)')
ax.set_title('Judge\u2013human agreement per trace (dashed = model mean)')
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('results/fig1_agreement_per_trace.png', dpi=150)
plt.close()

# ════════════════════════════════════════════════════════════
# ФИГУРА 2: per-framework
# ════════════════════════════════════════════════════════════
fw = defaultdict(lambda: defaultdict(list))
for (t, mas, m), v in tm.items():
    fw[mas][m].append(v)
frameworks = sorted(fw)
x = np.arange(len(frameworks))
fig, ax = plt.subplots(figsize=(9, 5.5))
for i, m in enumerate(models):
    vals = [np.mean(fw[mas][m]) for mas in frameworks]
    ax.bar(x + (i - 0.5) * w, vals, w, label=m)
ax.set_xticks(x)
ax.set_xticklabels(frameworks)
ax.set_ylabel('Mean agreement (%)')
ax.set_title('Mean judge\u2013human agreement per framework')
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('results/fig2_per_framework.png', dpi=150)
plt.close()

# ════════════════════════════════════════════════════════════
# ФИГУРА 3: per-mode FalsePos vs FalseNeg
# ════════════════════════════════════════════════════════════
mode_stats = defaultdict(lambda: {'fp': 0, 'fn': 0, 'match': 0, 'total': 0})
for r in rows:
    s = mode_stats[r['mode']]
    h, j = int(r['human']), int(r['judge'])
    s['total'] += 1
    s['match'] += int(r['match'])
    if h == 0 and j == 1: s['fp'] += 1
    if h == 1 and j == 0: s['fn'] += 1
def mk(x): a, b = x.split('.'); return (int(a), int(b))
modes = sorted(mode_stats, key=mk)
x = np.arange(len(modes))
fig, ax = plt.subplots(figsize=(11, 5.5))
fp = [mode_stats[md]['fp'] for md in modes]
fn = [mode_stats[md]['fn'] for md in modes]
ax.bar(x - w/2, fp, w, label='False positives (judge over-detects)', color='#d9534f')
ax.bar(x + w/2, fn, w, label='False negatives (judge misses)', color='#5bc0de')
ax.set_xticks(x)
ax.set_xticklabels(modes, rotation=45, ha='right')
ax.set_ylabel('Count (across all traces \u00d7 models)')
ax.set_title('Per-mode error breakdown: over-detection vs missed detection')
ax.legend()
plt.tight_layout()
plt.savefig('results/fig3_per_mode_errors.png', dpi=150)
plt.close()

# ════════════════════════════════════════════════════════════
# ТАБЛИЦЫ как markdown-файл
# ════════════════════════════════════════════════════════════
lines = []
lines.append('# RQ C \u2014 Initial Results (Gemini flash vs pro, 19 human-labelled traces)\n')

lines.append('## Table 1 \u2014 Agreement per trace\n')
lines.append('| Trace | Framework | ' + ' | '.join(models) + ' |')
lines.append('|---|---|' + '---|' * len(models))
for (t, mas) in traces:
    cells = [f'{tm.get((t,mas,m),0):.1f}' for m in models]
    lines.append(f'| {t} | {mas} | ' + ' | '.join(cells) + ' |')
means = [np.mean([tm[(t,mas,m)] for (t,mas) in traces if (t,mas,m) in tm]) for m in models]
lines.append('| **MEAN** | | ' + ' | '.join(f'**{x:.1f}**' for x in means) + ' |')

lines.append('\n## Table 2 \u2014 Per-framework mean agreement\n')
lines.append('| Framework | ' + ' | '.join(models) + ' | n |')
lines.append('|---|' + '---|' * (len(models)+1))
for mas in frameworks:
    cells = [f'{np.mean(fw[mas][m]):.1f}' for m in models]
    n = len(fw[mas][models[0]])
    lines.append(f'| {mas} | ' + ' | '.join(cells) + f' | {n} |')

lines.append('\n## Table 3 \u2014 Per-mode error breakdown\n')
lines.append('| Mode | Accuracy % | HumanYES | JudgeYES | FalsePos | FalseNeg |')
lines.append('|---|---|---|---|---|---|')
for md in modes:
    s = mode_stats[md]
    acc = 100*s['match']/s['total']
    # пересчёт human/judge yes
    hy = sum(1 for r in rows if r['mode']==md and int(r['human'])==1)
    jy = sum(1 for r in rows if r['mode']==md and int(r['judge'])==1)
    lines.append(f'| {md} | {acc:.1f} | {hy} | {jy} | {s["fp"]} | {s["fn"]} |')

lines.append('\n*Localization map: in progress (step-decomposition recently fixed).*')

with open('results/initial_results.md', 'w') as f:
    f.write('\n'.join(lines))

print('✅ Создано:')
print('  results/fig1_agreement_per_trace.png')
print('  results/fig2_per_framework.png')
print('  results/fig3_per_mode_errors.png')
print('  results/initial_results.md  (три таблицы)')
