"""5-95% energy duration of the field segments: 15-hole blasts against the signature hole,
at the three stations closer than 80 m. Reports the vector (sum of the three squared components)
and each component separately, to show the sensitivity of the measure.
Usage: python real_data/duration_check.py <path to Waveform-Simulation-PPV-main>"""
import sys, os, json, numpy as np
seg = os.path.join(sys.argv[1], '1.Data', 'RESULTS', 'segments_signature')
idx = json.load(open(os.path.join(seg, 'index.json')))
D = {(e['rec'], e['blast']): e['xlsx_geo']['dist_m'] for e in idx}
def load(rec, b):
    d = np.genfromtxt(os.path.join(seg, f'{rec}_{b}.csv'), delimiter=',', names=True); n = d.dtype.names
    return d[n[0]], [d[c] - d[c].mean() for c in n[1:4]], n[1:4]
def dur(t, e):
    E = np.cumsum(e); E /= E[-1]; return t[np.searchsorted(E, .95)] - t[np.searchsorted(E, .05)]
recs = sorted({k[0] for k in D if D[(k[0], 'Signature')] < 80})
_, _, comps = load(recs[0], 'Signature')
for label, f in [('vector', lambda X: sum(x**2 for x in X))] + [(c, (lambda j: lambda X: X[j]**2)(j)) for j, c in enumerate(comps)]:
    diffs = []
    for rec in recs:
        ts, Xs, _ = load(rec, 'Signature'); ds = dur(ts, f(Xs))
        for b in ('Test_1', 'Test_3', 'Test_4'):
            t, X, _ = load(rec, b); diffs.append(dur(t, f(X)) - ds)
    print(f'{label:10s} extra duration (s): min {min(diffs):.2f}  median {np.median(diffs):.2f}  max {max(diffs):.2f}  '
          f'positive in {sum(d > 0 for d in diffs)}/{len(diffs)} pairs')
