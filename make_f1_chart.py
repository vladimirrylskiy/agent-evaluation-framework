import csv
import numpy as np
import matplotlib.pyplot as plt

with open('results/human_validation.csv') as f:
    rows = list(csv.DictReader(f))

def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    f=2*p*r/(p+r) if p+r else 0; return p,r,f
def mk(x): a,b=x.split('.'); return (int(a),int(b))

models = sorted({r['model'] for r in rows})
modes = sorted({r['mode'] for r in rows}, key=mk)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, model in zip(axes, models):
    mr = [r for r in rows if r['model']==model
          and r['human']!='excluded' and r['judge']!='excluded']
    precs, recs, f1s = [], [], []
    for md in modes:
        rr=[r for r in mr if r['mode']==md]
        tp=fp=fn=0
        for r in rr:
            h,j=int(r['human']),int(r['judge'])
            if h==1 and j==1:tp+=1
            elif h==0 and j==1:fp+=1
            elif h==1 and j==0:fn+=1
        p,rc,f=prf(tp,fp,fn)
        precs.append(p); recs.append(rc); f1s.append(f)
    macro = np.mean(f1s)
    x=np.arange(len(modes)); w=0.27
    ax.bar(x-w, precs, w, label='Precision', color='#5B8FF9')
    ax.bar(x,    recs,  w, label='Recall',    color='#5AD8A6')
    ax.bar(x+w,  f1s,   w, label='F1',        color='#F6BD16')
    ax.axhline(macro, ls='--', color='#E8684A', lw=1.5,
               label=f'Macro-F1={macro:.2f}')
    ax.set_xticks(x); ax.set_xticklabels(modes, rotation=45, ha='right')
    ax.set_title(f'{model}  (per-mode P/R/F1)')
    ax.set_ylim(0,1.05); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

axes[0].set_ylabel('Score')
fig.suptitle('Per-mode Precision / Recall / F1 vs human labels (full 19 traces)',
             fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('results/fig_f1_per_mode.png', dpi=150, bbox_inches='tight')
print('✅ results/fig_f1_per_mode.png')
