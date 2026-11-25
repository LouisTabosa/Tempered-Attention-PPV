import json, time, sys, numpy as np
sys.path.insert(0, '.')
from experiments import *
R = {}; t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
R['E1'] = e1_limits(); log('E1')
pool, test = make_pools(); log('pools')
R['E2'], fitted = e2_compare(pool, test); log('E2')
R['E5'] = e5_tau(fitted); log('E5')
R['E6'] = e6_attribution(fitted[0]); log('E6')
R['E3'] = e3_learning(pool, test); log('E3')
R['E4'] = e4_extrapolation(pool, test); log('E4')
R['E7'] = e7_scatter(); log('E7')
# raw per-seed metrics are kept so figures can be redrawn without re-running
json.dump(R, open('results.json', 'w'), indent=1, ensure_ascii=False, default=float)
log('done')
