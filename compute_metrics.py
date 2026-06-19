import csv
import numpy as np
from collections import defaultdict

with open('results/human_validation.csv') as f:
    rows = list(csv.DictReader(f))

models = sorted({r['model'] for r in rows})

def prf(tp, fp, fn):
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    f = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f

def cohen_kappa(pairs):
    # pairs: list of (human, judge) 0/1
    n = len(pairs)
    if n == 0: return 0.0
    po = sum(1 for h,j in pairs if h==j)/n
    ph1 = sum(h for h,j in pairs)/n; pj1 = sum(j for h,j in pairs)/n
    pe = ph1*pj1 + (1-ph1)*(1-pj1)
    return (po-pe)/(1-pe) if (1-pe) else 0.0

def mk(x): a,b=x.split('.'); return (int(a),int(b))

for model in models:
    mr = [r for r in rows if r['model']==model]
    print(f'\n{"="*70}\nMODEL: {model}\n{"="*70}')
    # per-mode
    modes = sorted({r['mode'] for r in mr}, key=mk)
    print(f'{"Mode":6} {"Prec":>6} {"Recall":>7} {"F1":>6} {"TP":>3} {"FP":>3} {"FN":>3} {"TN":>3}')
    f1s = []
    all_pairs = []
    for md in modes:
        rr = [r for r in mr if r['mode']==md]
        tp=fp=fn=tn=0
        for r in rr:
            if r['human']=='excluded' or r['judge']=='excluded': continue
            h,j = int(r['human']), int(r['judge'])
            all_pairs.append((h,j))
            if h==1 and j==1: tp+=1
            elif h==0 and j==1: fp+=1
            elif h==1 and j==0: fn+=1
            else: tn+=1
        p,rec,f = prf(tp,fp,fn)
        f1s.append(f)
        print(f'{md:6} {p:>6.2f} {rec:>7.2f} {f:>6.2f} {tp:>3} {fp:>3} {fn:>3} {tn:>3}')
    macro_f1 = np.mean(f1s)
    kappa = cohen_kappa(all_pairs)
    print(f'\n  Macro-F1: {macro_f1:.3f}')
    print(f'  Cohen kappa (overall): {kappa:.3f}')

    # bootstrap CI over traces for macro-F1 and kappa
    traces = sorted({(r['trace'],r['mas_name']) for r in mr})
    by_trace = defaultdict(list)
    for r in mr:
        by_trace[(r['trace'],r['mas_name'])].append(r)
    rng = np.random.default_rng(42)
    boot_f1, boot_k = [], []
    for _ in range(1000):
        samp = rng.choice(len(traces), len(traces), replace=True)
        sel = []
        for i in samp:
            sel.extend(by_trace[traces[i]])
        # macro-F1 on sample
        f1s_b = []
        pairs_b = []
        for md in modes:
            rr=[r for r in sel if r['mode']==md]
            tp=fp=fn=0
            for r in rr:
                if r['human']=='excluded' or r['judge']=='excluded': continue
                h,j=int(r['human']),int(r['judge'])
                pairs_b.append((h,j))
                if h==1 and j==1: tp+=1
                elif h==0 and j==1: fp+=1
                elif h==1 and j==0: fn+=1
            _,_,f=prf(tp,fp,fn); f1s_b.append(f)
        boot_f1.append(np.mean(f1s_b))
        boot_k.append(cohen_kappa(pairs_b))
    print(f'  Macro-F1 95% CI: [{np.percentile(boot_f1,2.5):.3f}, {np.percentile(boot_f1,97.5):.3f}]')
    print(f'  Kappa 95% CI:    [{np.percentile(boot_k,2.5):.3f}, {np.percentile(boot_k,97.5):.3f}]')
