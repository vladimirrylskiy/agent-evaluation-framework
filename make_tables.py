import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

with open('results/human_validation.csv') as f:
    rows = list(csv.DictReader(f))
tm = {}
for r in rows:
    tm[(r['trace'], r['mas_name'], r['model'])] = float(r['agreement_pct'])
models = sorted({m for (_, _, m) in tm})
traces = sorted({(t, mas) for (t, mas, _) in tm})

def render_table(col_labels, cell_data, title, fname, col_widths=None, highlight_last=False):
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels)*1.6), 0.5 + 0.32*len(cell_data)))
    ax.axis('off')
    ax.set_title(title, fontsize=13, weight='bold', pad=12)
    tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    # стиль заголовка
    for j in range(len(col_labels)):
        c = tbl[0, j]
        c.set_facecolor('#2c3e50'); c.set_text_props(color='white', weight='bold')
    # зебра
    for i in range(1, len(cell_data)+1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                tbl[i, j].set_facecolor('#f2f4f6')
    if highlight_last:
        for j in range(len(col_labels)):
            tbl[len(cell_data), j].set_facecolor('#dfe9f5')
            tbl[len(cell_data), j].set_text_props(weight='bold')
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()

# ── Table 1 ──
data1 = []
for (t, mas) in traces:
    data1.append([t, mas] + [f'{tm.get((t,mas,m),0):.1f}' for m in models])
means = [np.mean([tm[(t,mas,m)] for (t,mas) in traces if (t,mas,m) in tm]) for m in models]
data1.append(['MEAN', ''] + [f'{x:.1f}' for x in means])
render_table(['Trace', 'Framework'] + models, data1,
             'Table 1 — Judge–human agreement per trace (%)',
             'results/table1_per_trace.png', highlight_last=True)

# ── Table 2 ──
fw = defaultdict(lambda: defaultdict(list))
for (t, mas, m), v in tm.items():
    fw[mas][m].append(v)
data2 = []
for mas in sorted(fw):
    data2.append([mas] + [f'{np.mean(fw[mas][m]):.1f}' for m in models] + [str(len(fw[mas][models[0]]))])
render_table(['Framework'] + models + ['n'], data2,
             'Table 2 — Mean agreement per framework (%)',
             'results/table2_per_framework.png')

# ── Table 3 ──
mode_stats = defaultdict(lambda: {'fp':0,'fn':0,'match':0,'total':0})
for r in rows:
    s = mode_stats[r['mode']]
    h,j = int(r['human']), int(r['judge'])
    s['total']+=1; s['match']+=int(r['match'])
    if h==0 and j==1: s['fp']+=1
    if h==1 and j==0: s['fn']+=1
def mk(x): a,b=x.split('.'); return (int(a),int(b))
data3 = []
for md in sorted(mode_stats, key=mk):
    s = mode_stats[md]
    hy = sum(1 for r in rows if r['mode']==md and int(r['human'])==1)
    jy = sum(1 for r in rows if r['mode']==md and int(r['judge'])==1)
    data3.append([md, f'{100*s["match"]/s["total"]:.1f}', str(hy), str(jy), str(s['fp']), str(s['fn'])])
render_table(['Mode','Accuracy %','Human YES','Judge YES','False Pos','False Neg'], data3,
             'Table 3 — Per-mode error breakdown',
             'results/table3_per_mode.png')

print('✅ Создано:')
print('  results/table1_per_trace.png')
print('  results/table2_per_framework.png')
print('  results/table3_per_mode.png')
