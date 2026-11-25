"""
Tempered attention as an aggregation law for PPV: synthetic generator, models and metrics.

Core idea
---------
The USBM law, PPV = K (D / W^alpha)^(-beta), is an unnormalised attention
score: PPV/K = exp(w . phi), with phi = [ln D, ln W] (key) and
w = [-beta, alpha*beta] (query).

For a blast with N delay groups we propose
    ln PPV = tau * logsumexp(s_i / tau),   s_i = ln K - beta ln D_i + alpha beta ln W_i
which is the softmax denominator with temperature tau (the role of sqrt(d_k)).
    tau -> 0 : max_i s_i  (classical law with maximum charge per delay)
    tau = 1  : ln sum_i PPV_i (in-phase superposition, upper bound)
The softmax weights a_i = d ln PPV / d s_i are the sensitivities of the prediction to each delay.

Synthetic generator: linear superposition of "signature hole" waveforms
(one per delay group), with nominal delay, initiation-time scatter,
propagation delay and log-normal site noise.

Notation: W = charge per delay, K = site constant.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import minimize, brentq
from scipy.special import logsumexp, expit
from sklearn.ensemble import GradientBoostingRegressor

# ----------------------------------------------------------------------------
# 1. Synthetic generator
# ----------------------------------------------------------------------------


@dataclass
class SiteParams:
    K: float = 1140.0        # site constant (mm/s), typical USBM
    alpha: float = 0.5       # charge exponent (square-root scaling, USBM)
    beta: float = 1.6        # attenuation exponent
    f0: float = 45.0         # dominant frequency at 100 m (Hz)
    gamma: float = 0.35      # frequency decay with distance
    zeta: float = 0.12       # damping of the signature-hole waveform
    c: float = 3000.0        # apparent velocity (m/s)
    sigma_group: float = 0.15  # log-amplitude variability per group
    sigma_freq: float = 0.10   # log-frequency variability per group
    sigma_phase: float = 0.5   # phase scatter between groups (rad)
    sigma_site: float = 0.20   # log-normal noise of the record (measurement/site error)
    timing_scatter: float = 0.5e-3  # std of the initiation time (s)
    timing_scatter_rel: float = 0.0  # additional std as a fraction of the nominal firing time (E7: 0.03)
    spacing: float = 6.0     # distance between group centres along the bench (m)
    fs: float = 2000.0       # sampling rate (Hz)


@dataclass
class Blast:
    D: np.ndarray            # distance from each group to the sensor (m)
    W: np.ndarray            # charge of each group (kg)
    dt: float                # nominal delay interval (s)
    Dc: float                # distance from the centroid to the sensor (m)
    s_true: np.ndarray       # log of the stand-alone amplitude of each group (with variability)
    ppv_true: float          # PPV without measurement noise (mm/s)
    ppv_obs: float           # observed PPV (mm/s)
    f_dom: float             # dominant frequency of the record (Hz)
    contrib: np.ndarray = field(default=None)  # exact decomposition of the peak by group (sums to 1)


def _signature(t, f, zeta, phase):
    """Signature-hole waveform: causal damped sinusoid, unit peak."""
    w = 2 * np.pi * f
    y = np.where(t >= 0, np.exp(-zeta * w * np.clip(t, 0, None)) * np.sin(w * np.clip(t, 0, None) + phase), 0.0)
    return y


def simulate_blast(rng, sp: SiteParams, N=None, dt=None, Dc=None, W=None,
                   keep_trace=False, equal_charge=False):
    if N is None:
        N = int(rng.integers(5, 41))
    if dt is None:
        dt = rng.uniform(0.0, 0.120)
    if Dc is None:
        Dc = float(np.exp(rng.uniform(np.log(80), np.log(1500))))
    if W is None:
        if equal_charge:
            W = np.full(N, 150.0)
        else:
            W = np.clip(rng.lognormal(np.log(150), 0.5, size=N), 30, 800)
    W = np.asarray(W, float)
    N = len(W)

    # geometry: groups aligned on the bench, sensor at a random azimuth
    x = (np.arange(N) - (N - 1) / 2) * sp.spacing
    th = rng.uniform(0, np.pi)
    sx, sy = Dc * np.cos(th), Dc * np.sin(th)
    D = np.hypot(sx - x, sy)

    # stand-alone amplitude (USBM law per delay) and frequency of each group
    s = (np.log(sp.K) - sp.beta * np.log(D) + sp.alpha * sp.beta * np.log(W)
         + rng.normal(0, sp.sigma_group, N))
    A = np.exp(s)
    f = sp.f0 * (D / 100.0) ** (-sp.gamma) * np.exp(rng.normal(0, sp.sigma_freq, N))
    ph = rng.normal(0, sp.sigma_phase, N)

    # arrival times: nominal delay + scatter + propagation
    t_nom = np.arange(N) * dt
    t_arr = t_nom + rng.normal(0, 1, N) * (sp.timing_scatter + sp.timing_scatter_rel * t_nom) + D / sp.c
    t_arr -= t_arr.min()
    T = t_arr.max() + 0.8
    t = np.arange(0, T, 1 / sp.fs)

    traces = np.empty((N, t.size))
    for i in range(N):
        y = _signature(t - t_arr[i], f[i], sp.zeta, ph[i])
        traces[i] = A[i] * y / np.max(np.abs(y))  # stand-alone peak = A_i
    total = traces.sum(0)
    k = int(np.argmax(np.abs(total)))
    ppv_true = float(np.abs(total[k]))
    contrib = traces[:, k] * np.sign(total[k]) / ppv_true  # sums exactly to 1

    spec = np.abs(np.fft.rfft(total))
    freqs = np.fft.rfftfreq(t.size, 1 / sp.fs)
    f_dom = float(freqs[1:][np.argmax(spec[1:])])

    ppv_obs = ppv_true * np.exp(rng.normal(0, sp.sigma_site))
    b = Blast(D=D, W=W, dt=dt, Dc=Dc, s_true=s, ppv_true=ppv_true,
              ppv_obs=ppv_obs, f_dom=f_dom, contrib=contrib)
    if keep_trace:
        return b, t, total, traces
    return b


def simulate_catalog(n, seed, sp: SiteParams, **kw):
    rng = np.random.default_rng(seed)
    return [simulate_blast(rng, sp, **kw) for _ in range(n)]


# ----------------------------------------------------------------------------
# 2. Packing into arrays (ragged -> padded + mask)
# ----------------------------------------------------------------------------


@dataclass
class Data:
    lnD: np.ndarray    # (n, Nmax)
    lnW: np.ndarray    # (n, Nmax)
    mask: np.ndarray   # (n, Nmax) bool
    dt: np.ndarray     # (n,) s
    lnDc: np.ndarray   # (n,)
    y: np.ndarray      # (n,) observed ln PPV
    y_true: np.ndarray  # (n,) noise-free ln PPV
    lnf: np.ndarray    # (n,) ln f_dom of the record

    def subset(self, idx):
        return Data(**{k: getattr(self, k)[idx] for k in self.__dataclass_fields__})

    def __len__(self):
        return len(self.y)


def pack(blasts):
    n = len(blasts)
    Nmax = max(len(b.D) for b in blasts)
    lnD = np.zeros((n, Nmax)); lnW = np.zeros((n, Nmax)); mask = np.zeros((n, Nmax), bool)
    for j, b in enumerate(blasts):
        m = len(b.D)
        lnD[j, :m] = np.log(b.D); lnW[j, :m] = np.log(b.W); mask[j, :m] = True
    return Data(lnD=lnD, lnW=lnW, mask=mask,
                dt=np.array([b.dt for b in blasts]),
                lnDc=np.log([b.Dc for b in blasts]),
                y=np.log([b.ppv_obs for b in blasts]),
                y_true=np.log([b.ppv_true for b in blasts]),
                lnf=np.log([b.f_dom for b in blasts]))


def summary_features(d: Data, fhat=None):
    """Aggregate variables per blast, as in the usual databases."""
    W = np.where(d.mask, np.exp(d.lnW), 0.0)
    lnWmax = np.log(W.max(1)); lnWtot = np.log(W.sum(1)); N = d.mask.sum(1)
    cols = [d.lnDc, lnWmax, lnWtot, N, d.dt * 1e3]
    if fhat is not None:
        cols.append(d.dt * fhat.predict(d.lnDc))
    return np.column_stack(cols), lnWmax, lnWtot


# ----------------------------------------------------------------------------
# 3. Models
# ----------------------------------------------------------------------------


class FreqLaw:
    """ln f_dom = c0 + c1 ln Dc, fitted on the training records."""
    def fit(self, d: Data):
        X = np.column_stack([np.ones_like(d.lnDc), d.lnDc])
        self.c = np.linalg.lstsq(X, d.lnf, rcond=None)[0]
        return self

    def predict(self, lnDc):
        return np.exp(self.c[0] + self.c[1] * lnDc)


class USBM:
    """ln PPV = ln K - beta ln Dc + alpha beta ln W_agg (W_agg = max or total)."""
    def __init__(self, agg="max"):
        self.agg = agg

    def _X(self, d):
        _, lnWmax, lnWtot = summary_features(d)
        lnW = lnWmax if self.agg == "max" else lnWtot
        return np.column_stack([np.ones(len(d)), d.lnDc, lnW])

    def fit(self, d):
        self.coef = np.linalg.lstsq(self._X(d), d.y, rcond=None)[0]
        self.lnK, self.beta = self.coef[0], -self.coef[1]
        self.alpha = self.coef[2] / max(self.beta, 1e-6)
        return self

    def predict(self, d):
        return self._X(d) @ self.coef


TAU_MIN = 1e-3


def lse_tau(s, mask, tau):
    """Masked tau * logsumexp(s / tau); tau may be a vector (n,)."""
    tau = np.broadcast_to(np.asarray(tau, float), (s.shape[0],))[:, None]
    z = np.where(mask, s / tau, -np.inf)
    return tau[:, 0] * logsumexp(z, axis=1)


def softmax_weights(s, mask, tau):
    tau = np.broadcast_to(np.asarray(tau, float), (s.shape[0],))[:, None]
    z = np.where(mask, s / tau, -np.inf)
    z = z - z.max(1, keepdims=True)
    e = np.where(mask, np.exp(z), 0.0)
    return e / e.sum(1, keepdims=True)


class LSEAttention:
    """
    Physics-structured attention over the delay groups.
      s_i = ln K - beta ln D_i + alpha beta ln W_i
      tau = TAU_MIN + (1-TAU_MIN) sigmoid(a - b x),  x = dt * fhat(Dc)  (delay in cycles)
      ln PPV = tau logsumexp(s/tau)
    mode="const": constant tau (4 parameters); mode="dt": tau(x) (5 parameters);
    mode="max": tau -> 0 with individual D_i (3 parameters).
    """
    def __init__(self, mode="dt"):
        self.mode = mode

    def _unpack(self, th):
        lnK, alpha, beta = th[:3]
        return lnK, alpha, beta, th[3:]

    def scores(self, th, d):
        lnK, alpha, beta, _ = self._unpack(th)
        return lnK - beta * d.lnD + alpha * beta * d.lnW

    def tau(self, th, d):
        _, _, _, r = self._unpack(th)
        if self.mode == "max":
            return np.full(len(d), TAU_MIN)
        if self.mode == "const":
            return TAU_MIN + (1 - TAU_MIN) * expit(r[0]) * np.ones(len(d))
        x = d.dt * self.fhat.predict(d.lnDc)
        return TAU_MIN + (1 - TAU_MIN) * expit(r[0] - r[1] * x)

    def _pred(self, th, d):
        return lse_tau(self.scores(th, d), d.mask, self.tau(th, d))

    def fit(self, d, fhat=None, init=None, n_starts=4, seed=0):
        self.fhat = fhat if fhat is not None else FreqLaw().fit(d)
        u = USBM("max").fit(d)
        base = [u.lnK, np.clip(u.alpha, 0.1, 1.5), np.clip(u.beta, 0.3, 3.0)]
        extra = {"max": [], "const": [0.0], "dt": [1.0, 1.0]}[self.mode]
        rng = np.random.default_rng(seed)
        best = None
        for k in range(n_starts):
            th0 = np.array(base + extra, float)
            if k > 0:
                th0 = th0 + rng.normal(0, [0.5, 0.1, 0.2] + [1.0] * len(extra))
            loss = lambda th: np.mean((self._pred(th, d) - d.y) ** 2)
            res = minimize(loss, th0, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
        self.theta = best.x
        return self

    def predict(self, d):
        return self._pred(self.theta, d)

    def weights(self, d):
        return softmax_weights(self.scores(self.theta, d), d.mask, self.tau(self.theta, d))

    def tau_of_x(self, x):
        r = self.theta[3:]
        return TAU_MIN + (1 - TAU_MIN) * expit(r[0] - r[1] * x)


class GBR:
    """Gradient boosting on aggregate variables per blast (optionally on the USBM residual)."""
    def __init__(self, residual=False, seed=0):
        self.residual = residual; self.seed = seed

    def fit(self, d, fhat=None):
        self.fhat = fhat if fhat is not None else FreqLaw().fit(d)
        X, _, _ = summary_features(d, self.fhat)
        y = d.y
        if self.residual:
            self.u = USBM("max").fit(d); y = y - self.u.predict(d)
        self.m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                           subsample=0.8, min_samples_leaf=5,
                                           random_state=self.seed).fit(X, y)
        return self

    def predict(self, d):
        X, _, _ = summary_features(d, self.fhat)
        p = self.m.predict(X)
        return p + self.u.predict(d) if self.residual else p


class GenericAttention:
    """
    Generic attention (no physics) over the same delay groups:
    tokens [ln D_i, ln W_i, firing time, dt, ln Dc] -> tanh embedding (H)
    -> attention pooling with a learned query -> MLP -> ln PPV.
    About 520 parameters, Adam with L2 and early stopping. It separates the gain
    of the physical structure from the gain of simply using per-delay data.
    """
    def __init__(self, H=16, dk=8, steps=2500, lr=0.01, l2=1e-3, seed=0):
        self.H, self.dk, self.steps, self.lr, self.l2, self.seed = H, dk, steps, lr, l2, seed

    def _tokens(self, d):
        n, M = d.lnD.shape
        tfire = np.arange(M)[None, :] * d.dt[:, None]
        X = np.stack([d.lnD, d.lnW, tfire * 10, np.repeat(d.dt[:, None] * 10, M, 1),
                      np.repeat(d.lnDc[:, None], M, 1)], -1)
        return (X - self.mu) / self.sd

    def _forward(self, P, X, mask):
        import autograd.numpy as anp
        h = anp.tanh(anp.dot(X, P["W1"]) + P["b1"])               # (n,M,H)
        k = anp.dot(h, P["Wk"])                                    # (n,M,dk)
        sc = anp.dot(k, P["q"]) / np.sqrt(self.dk)                 # (n,M)
        sc = anp.where(mask, sc, -1e9)
        sc = sc - anp.max(sc, axis=1, keepdims=True)
        e = anp.exp(sc) * mask
        a = e / anp.sum(e, axis=1, keepdims=True)
        pooled = anp.sum(a[:, :, None] * h, axis=1)                # (n,H)
        z = anp.tanh(anp.dot(pooled, P["W2"]) + P["b2"])
        return anp.dot(z, P["w3"]) + P["b3"]

    def fit(self, d, fhat=None):
        from autograd import grad
        import autograd.numpy as anp
        rng = np.random.default_rng(self.seed)
        n, M = d.lnD.shape
        tfire = np.arange(M)[None, :] * d.dt[:, None]
        raw = np.stack([d.lnD, d.lnW, tfire * 10, np.repeat(d.dt[:, None] * 10, M, 1),
                        np.repeat(d.lnDc[:, None], M, 1)], -1)
        self.mu = raw[d.mask].mean(0); self.sd = raw[d.mask].std(0) + 1e-9
        X = self._tokens(d); mask = d.mask.astype(float)
        self.ym, self.ys = d.y.mean(), d.y.std()
        y = (d.y - self.ym) / self.ys
        F, H, dk = X.shape[-1], self.H, self.dk
        P = {"W1": rng.normal(0, 1 / np.sqrt(F), (F, H)), "b1": np.zeros(H),
             "Wk": rng.normal(0, 1 / np.sqrt(H), (H, dk)), "q": rng.normal(0, 1 / np.sqrt(dk), dk),
             "W2": rng.normal(0, 1 / np.sqrt(H), (H, H)), "b2": np.zeros(H),
             "w3": rng.normal(0, 1 / np.sqrt(H), H), "b3": 0.0}
        idx = rng.permutation(n); nv = max(int(0.2 * n), 5)
        va, tr = idx[:nv], idx[nv:]

        def loss(P, ii):
            r = self._forward(P, X[ii], mask[ii]) - y[ii]
            reg = sum(anp.sum(P[k] ** 2) for k in ("W1", "Wk", "W2", "w3"))
            return anp.mean(r ** 2) + self.l2 * reg
        g = grad(loss)
        m = {k: np.zeros_like(v) for k, v in P.items()}; v2 = {k: np.zeros_like(v) for k, v in P.items()}
        best, bestP, patience = np.inf, None, 0
        for t in range(1, self.steps + 1):
            G = g(P, tr)
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * G[k]; v2[k] = 0.999 * v2[k] + 0.001 * G[k] ** 2
                P[k] = P[k] - self.lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8)
            if t % 25 == 0:
                lv = float(np.mean((self._forward(P, X[va], mask[va]) - y[va]) ** 2))
                if lv < best - 1e-5:
                    best, bestP, patience = lv, {k: np.copy(vv) for k, vv in P.items()}, 0
                else:
                    patience += 1
                    if patience >= 12:
                        break
        self.P = bestP if bestP is not None else P
        return self

    def predict(self, d):
        X = self._tokens(d)
        return self._forward(self.P, X, d.mask.astype(float)) * self.ys + self.ym


# ----------------------------------------------------------------------------
# 4. Metrics and utilities
# ----------------------------------------------------------------------------


def error_factor(pred, y):
    """Median error factor: exp(median |ln pred - ln obs|)."""
    return float(np.exp(np.median(np.abs(pred - y))))


def rmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def empirical_tau(s, ln_ppv):
    """Solve for tau in (0,1] with tau*LSE(s/tau) = ln PPV. d/dtau = softmax entropy >= 0."""
    lo, hi = s.max(), logsumexp(s)
    if ln_ppv <= lo:
        return 0.0
    if ln_ppv >= hi:
        return 1.0
    g = lambda tau: tau * logsumexp(s / tau) - ln_ppv
    return brentq(g, 1e-4, 1.0)


MODEL_NAMES = {
    "USBM_max": "USBM (max W per delay)",
    "USBM_tot": "USBM (W total)",
    "GBR": "Gradient boosting",
    "GBR_res": "USBM + GB on residual",
    "GenAttn": "Generic attention (no physics)",
    "LSE_max": "Attention, tau -> 0",
    "LSE_const": "Attention, constant tau",
    "LSE_dt": "Attention, tau(dt f)",
}


def fit_all(dtr: Data, seed=0, which=None):
    fhat = FreqLaw().fit(dtr)
    which = which or list(MODEL_NAMES)
    makers = {
        "USBM_max": lambda: USBM("max").fit(dtr),
        "USBM_tot": lambda: USBM("tot").fit(dtr),
        "GBR": lambda: GBR(False, seed).fit(dtr, fhat),
        "GBR_res": lambda: GBR(True, seed).fit(dtr, fhat),
        "GenAttn": lambda: GenericAttention(seed=seed).fit(dtr, fhat),
        "LSE_max": lambda: LSEAttention("max").fit(dtr, fhat, seed=seed),
        "LSE_const": lambda: LSEAttention("const").fit(dtr, fhat, seed=seed),
        "LSE_dt": lambda: LSEAttention("dt").fit(dtr, fhat, seed=seed),
    }
    return {k: makers[k]() for k in which}
