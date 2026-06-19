import csv
import numpy as np
import matplotlib.pyplot as plt

with open('results/human_validation.csv') as f:
    rows = list(csv.DictReader(f))

def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    f=2*p*r/(p+r) if p+r else 0; return p,r,f
def kappa(pairs):
    n=len(pairs)
    if not n: return 0
    po=sum(1 for h,j in pairs if h==j)/n
    ph=sum(h for h,j in pairs)/n; pj=sum(j for h,j in pairs)/n
    pe=ph*pj+(1-ph)*(1-pj); return (po-pe)/(1-pe) if (1-pe) else 0
def mk(x): a,b=x.split('.'); return (int(a),int(b))

NAMES={'1.1':'Disobey Task','1.2':'Disobey Role','1.3':'Step Repetition',
'1.4':'Loss of Conv. History','1.5':'Unaware Termination','2.1':'Conversation Reset',
'2.2':'Fail Ask Clarification','2.3':'Task Derailment','2.4':'Info Withholding',
'2.5':'Ignored Input','2.6':'Action-Reasoning Mism.','3.1':'Premature Termination',
'3.2':'No/Incomplete Verif.','3.3':'Incorrect Verif.'}

models = sorted({r['model'] for r in rows})
modes = sorted({r['mode'] for r in rows}, key=mk)

def build(model):
    mr=[r for r in rows if r['model']==model and r['human']!='excluded' and r['judge']!='excluded']
    data=[]; f1s=[]; allp=[]
    for md in modes:
        rr=[r for r in mr if r['mode']==md]
        tp=fp=fn=0
        for r in rr:
            h,j=int(r['human']),int(r['judge']); allp.append((h,j))
            if h==1 and j==1:tp+=1
            elif h==0 and j==1:fp+=1
            elif h==1 and j==0:fn+=1
        p,rc,f=prf(tp,fp,fn); f1s.append(f)
        data.append([f'{md} {NAMES[md]}', f'{p:.2f}', f'{rc:.2f}', f'{f:.2f}', str(tp), str(fp), str(fn)])
    # bootstrap CI
    traces=sorted({r['trace'] for r in mr})
    rng=np.random.default_rng(42); bf=[]; bk=[]
    bt={t:[r for r in mr if r['trace']==t] for t in traces}
    for _ in range(1000):
        samp=rng.choice(len(traces),len(traces),replace=True)
        sel=[]
        for i in samp: sel.extend(bt[traces[i]])
        ff=[]; pp=[]
        for md in modes:
            rr=[r for r in sel if r['mode']==md]; tp=fp=fn=0
            for r in rr:
                h,j=int(r['human']),int(r['judge']); pp.append((h,j))
                if h==1 and j==1:tp+=1
                elif h==0 and j==1:fp+=1
                elif h==1 and j==0:fn+=1
            _,_,f=prf(tp,fp,fn); ff.append(f)
        bf.append(np.mean(ff)); bk.append(kappa(pp))
    macro=np.mean(f1s); kap=kappa(allp)
    ci_f=(np.percentile(bf,2.5),np.percentile(bf,97.5))
    ci_k=(np.percentile(bk,2.5),np.percentile(bk,97.5))
    return data, macro, kap, ci_f, ci_k

def render(ax, model):
    data, macro, kap, ci_f, ci_k = build(model)
    cols=['Mode','Prec','Rec','F1','TP','FP','FN']
    ax.axis('off')
    ax.set_title(f'{model}', fontsize=12, weight='bold', pad=8)
    tbl=ax.table(cellText=data, colLabels=cols, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1,1.35)
    for j in range(len(cols)):
        c=tbl[0,j]; c.set_facecolor('#2c3e50'); c.set_text_props(color='white',weight='bold')
    tbl[0,0].set_text_props(color='white',weight='bold')
    for i in range(1,len(data)+1):
        for j in range(len(cols)):
            tbl[i,j].set_facecolor('#f2f4f6' if i%2==0 else 'white')
            if j==0: tbl[i,j].set_text_props(ha='left')
    # footer line
    ax.text(0.5,-0.04,
        f'Macro-F1 = {macro:.3f}  [{ci_f[0]:.2f}, {ci_f[1]:.2f}]      '
        f"Cohen's kappa = {kap:.3f}  [{ci_k[0]:.2f}, {ci_k[1]:.2f}]",
        ha='center', va='top', transform=ax.transAxes, fontsize=9.5, weight='bold')

fig, axes = plt.subplots(1,2, figsize=(15,6))
for ax,m in zip(axes, models): render(ax,m)
fig.suptitle('Per-mode Precision / Recall / F1 vs human labels — full 19 traces\n(macro-F1 and Cohen\u2019s kappa with bootstrap 95% CI)',
             fontsize=13, weight='bold')
plt.tight_layout(rect=[0,0.02,1,0.95])
plt.savefig('results/table_metrics_full.png', dpi=170, bbox_inches='tight')
print('✅ results/table_metrics_full.png')
