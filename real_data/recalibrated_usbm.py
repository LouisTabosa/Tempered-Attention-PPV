"""USBM law recalibrated with a constant (multiplicative) bias,
fitted leaving out the tested blast (LOBO), against the attention law with a constant tau.
Reads field_eval.csv and real_results.json (written by real_data_test.py); writes recalibrated_results.json.
Usage: python real_data/recalibrated_usbm.py"""
import os, json, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
T = pd.read_csv('field_eval.csv'); R = json.load(open('real_results.json'))
m30 = T.D >= 30

def lobo(base, kind):
    p = np.empty(len(T))
    for b in T.blast.unique():
        tr = (T.blast != b) & m30; te = (T.blast == b).values
        r = T.y[tr] - T[base][tr]
        if kind == 'bias':
            p[te] = T[base][te] + r.mean()
        else:  # attention, equal-delay form: ln PPV = s + tau ln N
            p[te] = T[base][te] + np.sum(r * T.lnN[tr]) / np.sum(T.lnN[tr] ** 2) * T.lnN[te]
    return p

def metrics(e):
    return dict(EF=round(float(np.exp(np.median(np.abs(e)))), 2), RMSE=round(float(np.sqrt(np.mean(e ** 2))), 2),
                bias=round(float(e.mean()), 2))

out = {}
for base, name in [('law', 'site law'), ('ref', 'signature hole')]:
    P = {'W per delay': T[base].values, 'constant bias (LOBO)': lobo(base, 'bias'), 'attention tau (LOBO)': lobo(base, 'tau')}
    out[name] = {'n=16': {}, 'n=20': {}, 'by blast (n=16)': {}}
    for sub, m in [('n=16', m30), ('n=20', T.D > 0)]:
        for k, p in P.items():
            out[name][sub][k] = metrics(p[m.values] - T.y[m].values)
    for b in sorted(T.blast.unique()):
        mm = (m30 & (T.blast == b)).values
        out[name]['by blast (n=16)'][f"{b} (N={int(T.N[T.blast == b].iloc[0])}, n={mm.sum()})"] = \
            {k: metrics(p[mm] - T.y.values[mm]) for k, p in P.items()}
json.dump(out, open('recalibrated_results.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
