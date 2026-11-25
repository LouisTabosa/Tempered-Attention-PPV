"""Experiments E1-E7. Each function saves figure(s) in fig/ and returns results."""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import spearmanr
from ppv_attention import *

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(FIG, exist_ok=True)

# reference palette (fixed order) + text tones
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#9a9893", "grid": "#e6e5e1"}
MODEL_COLOR = {"USBM_max": C["orange"], "GBR_res": C["aqua"], "GenAttn": C["violet"], "LSE_dt": C["blue"]}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink2"],
    "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.titlesize": 11,
    "axes.titlecolor": C["ink"], "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight", "lines.linewidth": 2,
})

SP = SiteParams()


def make_pools(n_pool=20000, n_test=3000, sp=SP, seed_pool=1, seed_test=999):
    pool = pack(simulate_catalog(n_pool, seed_pool, sp))
    test = pack(simulate_catalog(n_test, seed_test, sp))
    return pool, test


def iqr_str(a):
    a = np.asarray(a)
    return f"{np.median(a):.3f} [{np.quantile(a, .25):.3f}; {np.quantile(a, .75):.3f}]"


# ---------------------------------------------------------------------------
# E1  limits of the log-sum-exp and physical motivation
# ---------------------------------------------------------------------------
def e1_limits(seed=0):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))

    # (a) LSE_tau between the maximum and the sum
    s = np.log(np.array([10, 8, 7, 6, 5, 4, 4, 3, 3, 2.0]))  # stand-alone PPVs (mm/s)
    taus = np.linspace(0.005, 1, 200)
    v = [np.exp(t * logsumexp(s / t)) for t in taus]
    ax[0].plot(taus, v, color=C["blue"])
    ax[0].axhline(np.exp(s.max()), color=C["orange"], lw=1.2, ls="--")
    ax[0].axhline(np.exp(s).sum(), color=C["ink2"], lw=1.2, ls=":")
    ax[0].text(0.98, np.exp(s.max()) * 1.04, "maximum (USBM, max W per delay)", ha="right", color=C["ink2"], fontsize=8.5)
    ax[0].text(0.98, np.exp(s).sum() * 0.93, "in-phase sum", ha="right", va="top", color=C["ink2"], fontsize=8.5)
    ax[0].set(xlabel=r"temperature $\tau$", ylabel="predicted PPV (mm/s)",
              title=r"(a) $\tau\,\ln\sum e^{s_i/\tau}$ spans maximum and sum")

    # (b) synthetic PPV vs delay, fixed geometry
    dts = np.linspace(0, 0.12, 49)
    N, Dc, W = 20, 300.0, np.full(20, 150.0)
    sp0 = SiteParams(sigma_site=0.0, sigma_group=0.0)
    med, q1, q3 = [], [], []
    for dt in dts:
        p = [simulate_blast(rng, sp0, dt=dt, Dc=Dc, W=W).ppv_true for _ in range(40)]
        med.append(np.median(p)); q1.append(np.quantile(p, .25)); q3.append(np.quantile(p, .75))
    single = SP.K * (Dc / 150 ** SP.alpha) ** (-SP.beta)
    ax[1].fill_between(dts * 1e3, q1, q3, color=C["blue"], alpha=0.18, lw=0)
    ax[1].plot(dts * 1e3, med, color=C["blue"])
    ax[1].axhline(single, color=C["orange"], lw=1.2, ls="--")
    ax[1].text(118, single * 1.06, "USBM with max W per delay", ha="right", color=C["ink2"], fontsize=8.5)
    ax[1].set_yscale("log")
    ax[1].set(xlabel="delay interval (ms)", ylabel="PPV (mm/s)",
              title="(b) 20 delays of 150 kg at 300 m")

    # (c) example records
    for k, (dt, col) in enumerate([(0.0, C["orange"]), (0.009, C["violet"]), (0.042, C["blue"])]):
        b, t, tot, _ = simulate_blast(np.random.default_rng(3), sp0, dt=dt, Dc=Dc, W=np.full(12, 150.0), keep_trace=True)
        ax[2].plot(t * 1e3, tot + k * 0, color=col, lw=1.0, label=f"{dt*1e3:.0f} ms (PPV {b.ppv_true:.1f} mm/s)")
    ax[2].set_xlim(0, 700)
    ax[2].set(xlabel="time (ms)", ylabel="particle velocity (mm/s)", title="(c) synthetic records, 12 delays")
    ax[2].legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/E1_limits.png"); plt.close(fig)
    return {"ppv_single_300m": single, "ppv_dt0_median": med[0], "ppv_dt120_median": med[-1]}


# ---------------------------------------------------------------------------
# E2  main comparison (n = 150, several seeds)
# ---------------------------------------------------------------------------
def e2_compare(pool, test, n_train=150, n_seeds=20):
    res = {k: {"EF": [], "RMSE_true": []} for k in MODEL_NAMES}
    params = {k: [] for k in ("USBM_max", "LSE_dt")}
    fitted_lse = []
    for sd in range(n_seeds):
        rng = np.random.default_rng(100 + sd)
        tr = pool.subset(rng.choice(len(pool), n_train, replace=False))
        M = fit_all(tr, seed=sd)
        for k, m in M.items():
            p = m.predict(test)
            res[k]["EF"].append(error_factor(p, test.y)); res[k]["RMSE_true"].append(rmse(p, test.y_true))
        params["USBM_max"].append([M["USBM_max"].lnK, M["USBM_max"].alpha, M["USBM_max"].beta])
        params["LSE_dt"].append(list(M["LSE_dt"].theta[:3]))
        fitted_lse.append(M["LSE_dt"])
    oracle_EF = error_factor(test.y_true, test.y)

    order = sorted(MODEL_NAMES, key=lambda k: -np.median(res[k]["RMSE_true"]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for j, (met, lab) in enumerate([("EF", "median error factor (observed)"),
                                    ("RMSE_true", "RMSE in ln PPV against the noise-free value")]):
        for i, k in enumerate(order):
            a = np.array(res[k][met]); col = C["blue"] if k.startswith("LSE") else C["muted"]
            if k == "LSE_dt": col = C["blue"]
            elif k.startswith("LSE"): col = "#8fb6e8"
            ax[j].plot([np.quantile(a, .25), np.quantile(a, .75)], [i, i], color=col, lw=3, solid_capstyle="round")
            ax[j].plot(np.median(a), i, "o", color=col, ms=8, mec="white", mew=1.5)
            ax[j].text(np.quantile(a, .75), i + 0.28, f"{np.median(a):.3f}", fontsize=8, color=C["ink2"], ha="right")
        if met == "EF":
            ax[j].axvline(oracle_EF, color=C["ink2"], lw=1, ls=":")
            ax[j].text(oracle_EF, len(order) - 0.4, " noise floor", fontsize=8, color=C["ink2"])
        ax[j].set_xlabel(lab); ax[j].grid(axis="y", visible=False)
    ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels([MODEL_NAMES[k] for k in order])
    fig.suptitle(f"E2: out-of-sample test (n train = {n_train}, {n_seeds} seeds; median and interquartile range)",
                 color=C["ink"], fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/E2_comparison.png"); plt.close(fig)

    table = {k: {"EF": iqr_str(res[k]["EF"]), "RMSE_true": iqr_str(res[k]["RMSE_true"])} for k in order[::-1]}
    pr = {k: np.array(v) for k, v in params.items()}
    ptab = {k: {"lnK": iqr_str(v[:, 0]), "alpha": iqr_str(v[:, 1]), "beta": iqr_str(v[:, 2])} for k, v in pr.items()}
    ptab["true"] = {"lnK": f"{np.log(SP.K):.3f}", "alpha": f"{SP.alpha:.3f}", "beta": f"{SP.beta:.3f}"}
    return {"table": table, "oracle_EF": oracle_EF, "params": ptab, "raw": res}, fitted_lse


# ---------------------------------------------------------------------------
# E3  learning curve
# ---------------------------------------------------------------------------
def e3_learning(pool, test, ns=(30, 60, 120, 240, 480), n_seeds=12,
                which=("USBM_max", "GBR_res", "GenAttn", "LSE_dt")):
    out = {k: {n: [] for n in ns} for k in which}
    for n in ns:
        for sd in range(n_seeds):
            rng = np.random.default_rng(1000 + 37 * n + sd)
            tr = pool.subset(rng.choice(len(pool), n, replace=False))
            M = fit_all(tr, seed=sd, which=list(which))
            for k, m in M.items():
                out[k][n].append(rmse(m.predict(test), test.y_true))
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for k in which:
        med = [np.median(out[k][n]) for n in ns]
        q1 = [np.quantile(out[k][n], .25) for n in ns]; q3 = [np.quantile(out[k][n], .75) for n in ns]
        ax.fill_between(ns, q1, q3, color=MODEL_COLOR[k], alpha=0.15, lw=0)
        ax.plot(ns, med, "-o", color=MODEL_COLOR[k], ms=6, mec="white", mew=1.5, label=MODEL_NAMES[k])
        ax.text(ns[-1] * 1.06, med[-1], MODEL_NAMES[k], color=C["ink2"], fontsize=8.5, va="center")
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(ns[0] * 0.85, ns[-1] * 3.2)
    ax.set(xlabel="number of training blasts", ylabel="RMSE in ln PPV (against the noise-free value)",
           title="E3: learning curve")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/E3_learning_curve.png"); plt.close(fig)
    return {k: {n: iqr_str(v) for n, v in d.items()} for k, d in out.items()}


# ---------------------------------------------------------------------------
# E4  extrapolation (delay and distance)
# ---------------------------------------------------------------------------
def e4_extrapolation(pool, test, n_train=200, n_seeds=12,
                     which=("USBM_max", "GBR_res", "GenAttn", "LSE_dt")):
    scen = {
        "delay: train dt >= 20 ms, test dt <= 8 ms": (pool.dt >= 0.020, test.dt <= 0.008),
        "distance: train Dc <= 400 m, test Dc >= 700 m": (pool.lnDc <= np.log(400), test.lnDc >= np.log(700)),
        "reference: same distribution": (np.ones(len(pool), bool), np.ones(len(test), bool)),
    }
    out = {}
    for name, (mtr, mte) in scen.items():
        te = test.subset(np.where(mte)[0]); itr = np.where(mtr)[0]
        out[name] = {k: [] for k in which}
        for sd in range(n_seeds):
            rng = np.random.default_rng(2000 + sd)
            tr = pool.subset(rng.choice(itr, n_train, replace=False))
            M = fit_all(tr, seed=sd, which=list(which))
            for k, m in M.items():
                out[name][k].append(rmse(m.predict(te), te.y_true))
    fig, ax = plt.subplots(figsize=(9, 4.0))
    names = list(scen); w = 0.19
    for j, k in enumerate(which):
        meds = [np.median(out[s][k]) for s in names]
        x = np.arange(len(names)) + (j - 1.5) * w
        ax.bar(x, meds, width=w - 0.02, color=MODEL_COLOR[k], label=MODEL_NAMES[k], zorder=3)
        for xi, mi in zip(x, meds):
            ax.text(xi, mi + 0.01, f"{mi:.2f}", ha="center", fontsize=7.5, color=C["ink2"])
    ax.set_xticks(np.arange(len(names))); ax.set_xticklabels([s.replace(": ", ":\n") for s in names], fontsize=8.5)
    ax.set(ylabel="RMSE in ln PPV (median)", title=f"E4: extrapolation (n train = {n_train}, {n_seeds} seeds)")
    ax.grid(axis="x", visible=False); ax.set_ylim(0, 1.25)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(); fig.savefig(f"{FIG}/E4_extrapolation.png"); plt.close(fig)
    return {s: {k: iqr_str(v) for k, v in d.items()} for s, d in out.items()}


# ---------------------------------------------------------------------------
# E5  does the recovered temperature follow the physical hypothesis?
# ---------------------------------------------------------------------------
def e5_tau(fitted_lse, n=3000, seed=7):
    blasts = simulate_catalog(n, seed, SP)
    d = pack(blasts)
    fh = fitted_lse[0].fhat
    x = d.dt * fh.predict(d.lnDc)
    tau_emp = np.array([empirical_tau(b.s_true, np.log(b.ppv_true)) for b in blasts])
    bins = np.arange(0, 3.01, 0.2); mids = 0.5 * (bins[1:] + bins[:-1])
    med = [np.median(tau_emp[(x >= a) & (x < b)]) if ((x >= a) & (x < b)).sum() > 10 else np.nan
           for a, b in zip(bins[:-1], bins[1:])]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(x, tau_emp, s=6, color=C["muted"], alpha=0.35, lw=0, label=r"empirical $\tau$ per blast (noise-free)")
    ax.plot(mids, med, "o", color=C["ink"], ms=6, mec="white", label="median per bin")
    xx = np.linspace(0, 3, 200)
    for i, m in enumerate(fitted_lse):
        ax.plot(xx, m.tau_of_x(xx), color=C["blue"], lw=1, alpha=0.5,
                label=r"fitted $\tau(x)$ (one line per seed)" if i == 0 else None)
    ax.set_xlim(0, 3); ax.set_ylim(-0.02, 1.0)
    ax.set(xlabel=r"$x = \Delta t\,\hat f_d$ (delay in cycles)", ylabel=r"$\tau$",
           title="E5: temperature against dimensionless delay")
    ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/E5_tau.png"); plt.close(fig)
    rho = spearmanr(x, tau_emp).correlation
    ab = np.array([m.theta[3:] for m in fitted_lse])
    return {"spearman_x_tau": float(rho), "median_tau_bins": dict(zip([f"{m:.1f}" for m in mids], [None if np.isnan(v) else round(float(v), 3) for v in med])),
            "a": iqr_str(ab[:, 0]), "b": iqr_str(ab[:, 1]),
            "frac_tau0_x_gt1": float(np.mean(tau_emp[x > 1] == 0))}


# ---------------------------------------------------------------------------
# E6  do the attention weights attribute the peak correctly?
# ---------------------------------------------------------------------------
def e6_attribution(model, n=2000, seed=11):
    blasts = simulate_catalog(n, seed, SiteParams(sigma_site=0.0))
    d = pack(blasts)
    A = model.weights(d)
    tv = {"attention (fitted tau)": [], "single group (implicit in USBM)": [], "uniform": []}
    for j, b in enumerate(blasts):
        m = len(b.D); c = b.contrib; a = A[j, :m]
        onehot = np.zeros(m); onehot[np.argmax(a)] = 1
        uni = np.full(m, 1 / m)
        for k, w in zip(tv, (a, onehot, uni)):
            tv[k].append(0.5 * np.abs(w - c).sum())
    tv = {k: np.array(v) for k, v in tv.items()}
    bins = [(0, 0.005), (0.005, 0.017), (0.017, 0.042), (0.042, 0.121)]
    labels = ["0 to 5", "5 to 17", "17 to 42", "42 to 120"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    cols = [C["blue"], C["orange"], C["muted"]]; w = 0.26
    for i, (k, col) in enumerate(zip(tv, cols)):
        meds = [np.median(tv[k][(d.dt >= a) & (d.dt < b)]) for a, b in bins]
        x = np.arange(len(bins)) + (i - 1) * w
        ax.bar(x, meds, width=w - 0.03, color=col, label=k, zorder=3)
        for xi, mi in zip(x, meds):
            ax.text(xi, mi + 0.01, f"{mi:.2f}", ha="center", fontsize=7.5, color=C["ink2"])
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(labels)
    ax.set(xlabel="delay interval (ms)", ylabel="total variation distance (median)",
           title="E6: weights against the exact peak decomposition (lower is better)")
    ax.grid(axis="x", visible=False); ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/E6_attribution.png"); plt.close(fig)
    out = {k: iqr_str(v) for k, v in tv.items()}
    out["by_bin"] = {lab: {k: round(float(np.median(v[(d.dt >= a) & (d.dt < b)])), 3) for k, v in tv.items()}
                        for lab, (a, b) in zip(labels, bins)}
    return out


# ---------------------------------------------------------------------------
# E7  sensitivity to the initiation scatter (proportional to the nominal firing time)
# ---------------------------------------------------------------------------
def e7_scatter(n_train=150, n_seeds=12, rel=0.03):
    sp = SiteParams(timing_scatter_rel=rel)
    pool = pack(simulate_catalog(6000, 21, sp)); test = pack(simulate_catalog(2000, 22, sp))
    which = ["USBM_max", "GBR_res", "GenAttn", "LSE_const", "LSE_dt"]
    out = {k: [] for k in which}
    for sd in range(n_seeds):
        rng = np.random.default_rng(3000 + sd)
        tr = pool.subset(rng.choice(len(pool), n_train, replace=False))
        M = fit_all(tr, seed=sd, which=which)
        for k, m in M.items():
            out[k].append(rmse(m.predict(test), test.y_true))
    return {k: iqr_str(v) for k, v in out.items()}
