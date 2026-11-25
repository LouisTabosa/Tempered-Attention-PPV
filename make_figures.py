"""Figures of the article, drawn at the final print size of IEEE Access.

IEEE Access: column width 3.5 in, page width 7.16 in, labels of 8 to 10 pt,
sans-serif fonts (Arial/Helvetica), subfigure labels (a), (b), ... centred below
each panel in 8 pt Times. Figures are saved as vector PDF with embedded fonts.

E2, E3 and E4 are redrawn from results.json; E1 is re-simulated (fast); E5 refits
only the proposed model on the 20 E2 training sets (same seeds, deterministic) and
re-simulates the 3000 noise-free blasts; the field figure is redrawn from
real_data/field_eval.csv and real_data/real_results.json.

Usage:  python make_figures.py [output folder, default figures/]   [--skip-e5 | --only-e5]
The E5 refit is cached in e5_cache.pkl (about 1 min the first time).
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ppv_attention import *          # noqa: E402,F403

args = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = args[0] if args else os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
SKIP_E5 = "--skip-e5" in sys.argv

# ---------------------------------------------------------------- style
COL_W, PAGE_W = 3.45, 7.16                      # inches
avail = {f.name for f in font_manager.fontManager.ttflist}
SANS = [f for f in ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans") if f in avail] or ["DejaVu Sans"]
SERIF = [f for f in ("Times New Roman", "Liberation Serif", "Nimbus Roman") if f in avail] or ["DejaVu Serif"]

INK, INK2, GRID, GREY = "#222222", "#4d4d4d", "#e6e6e6", "#a2a3a5"
# soft palette, with the proposed model emphasised
PAL = {"USBM_max": "#e9a178", "GBR_res": "#7dc3b0", "GenAttn": "#b5a3d6", "LSE_dt": "#3e77b6",
       "LSE_light": "#9dbbdf", "grey": GREY}
LABEL = {"LSE_dt": r"Physics-structured, $\tau(x)$",
         "LSE_const": r"Physics-structured, constant $\tau$",
         "LSE_max": r"Physics-structured, $\tau \to 0$",
         "GenAttn": "Generic attention",
         "GBR_res": "USBM + GB on residual",
         "GBR": "Gradient boosting",
         "USBM_max": "USBM, max W per delay",
         "USBM_tot": "USBM, total W"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": SANS, "mathtext.fontset": "custom",
    "mathtext.rm": SANS[0], "mathtext.it": SANS[0] + ":italic", "mathtext.bf": SANS[0] + ":bold",
    "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.edgecolor": "#8c8c8c", "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
    "lines.linewidth": 1.4, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def sublabel(ax, s, y=-0.30):
    ax.text(0.5, y, s, transform=ax.transAxes, ha="center", va="top",
            fontsize=8, family=SERIF, color="black")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600)
    plt.close(fig)
    print("saved", name)


def parse_iqr(s):
    med, q = s.split(" [")
    q1, q3 = q.rstrip("]").split("; ")
    return float(med), float(q1), float(q3)


R = json.load(open(os.path.join(HERE, "results.json")))
SP = SiteParams()

# ---------------------------------------------------------------- E1
def fig_e1(seed=0):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(3, 1, figsize=(COL_W, 6.9))
    # (a)
    s = np.log(np.array([10, 8, 7, 6, 5, 4, 4, 3, 3, 2.0]))
    taus = np.linspace(0.005, 1, 200)
    ax[0].plot(taus, [np.exp(t * logsumexp(s / t)) for t in taus], color=PAL["LSE_dt"])
    ax[0].axhline(np.exp(s.max()), color=PAL["USBM_max"], lw=1.1, ls="--")
    ax[0].axhline(np.exp(s).sum(), color=INK2, lw=1.0, ls=":")
    ax[0].text(0.55, np.exp(s.max()) - 1.5, "maximum (USBM, max W per delay)", ha="center", va="top", color=INK2, fontsize=7.5)
    ax[0].text(0.02, np.exp(s).sum() - 1.5, "in-phase sum", ha="left", va="top", color=INK2, fontsize=7.5)
    ax[0].set_ylim(0, 58)
    ax[0].set(xlabel=r"Temperature $\tau$", ylabel="Predicted PPV (mm/s)")
    # (b)
    dts = np.linspace(0, 0.12, 49)
    Dc, W = 300.0, np.full(20, 150.0)
    sp0 = SiteParams(sigma_site=0.0, sigma_group=0.0)
    med, q1, q3 = [], [], []
    for dt in dts:
        p = [simulate_blast(rng, sp0, dt=dt, Dc=Dc, W=W).ppv_true for _ in range(40)]
        med.append(np.median(p)); q1.append(np.quantile(p, .25)); q3.append(np.quantile(p, .75))
    single = SP.K * (Dc / 150 ** SP.alpha) ** (-SP.beta)
    ax[1].fill_between(dts * 1e3, q1, q3, color=PAL["LSE_dt"], alpha=0.18, lw=0)
    ax[1].plot(dts * 1e3, med, color=PAL["LSE_dt"])
    ax[1].axhline(single, color=PAL["USBM_max"], lw=1.1, ls="--")
    ax[1].text(118, single * 0.9, "USBM, max W per delay", ha="right", va="top", color=INK2, fontsize=7.5)
    ax[1].set_yscale("log"); ax[1].set_ylim(4, 80)
    ax[1].set_yticks([5, 10, 20, 50]); ax[1].set_yticklabels(["5", "10", "20", "50"]); ax[1].minorticks_off()
    ax[1].set(xlabel="Delay interval (ms)", ylabel="PPV (mm/s)")
    # (c)
    for dt, col in [(0.0, PAL["USBM_max"]), (0.009, "#8a78c0"), (0.042, PAL["LSE_dt"])]:
        b, t, tot, _ = simulate_blast(np.random.default_rng(3), sp0, dt=dt, Dc=Dc, W=np.full(12, 150.0), keep_trace=True)
        ax[2].plot(t * 1e3, tot, color=col, lw=0.8, label=f"{dt*1e3:.0f} ms (PPV {b.ppv_true:.1f} mm/s)")
    ax[2].set_xlim(0, 700)
    ax[2].set(xlabel="Time (ms)", ylabel="Particle velocity (mm/s)")
    ax[2].legend(loc="upper right", title="Delay interval", title_fontsize=7.5, handlelength=1.5)
    for a, lab in zip(ax, "abc"):
        sublabel(a, f"({lab})")
    fig.subplots_adjust(hspace=0.62)
    save(fig, "E1_limits")
    return single, med[0], med[-1]


# ---------------------------------------------------------------- E2
def fig_e2():
    raw = R["E2"]["raw"]; floor = R["E2"]["oracle_EF"]
    order = sorted(raw, key=lambda k: -np.median(raw[k]["RMSE_true"]))
    fig, ax = plt.subplots(2, 1, figsize=(COL_W, 5.4))
    span = [max(np.quantile(raw[k][m], .75) for k in raw) - min(np.quantile(raw[k][m], .25) for k in raw) for m in ("EF", "RMSE_true")]
    for j, (met, lab) in enumerate([("EF", "Median error factor (observed)"),
                                    ("RMSE_true", r"RMSE in ln PPV (against the noise-free value)")]):
        for i, k in enumerate(order):
            a = np.array(raw[k][met])
            col = PAL["LSE_dt"] if k == "LSE_dt" else (PAL["LSE_light"] if k.startswith("LSE") else GREY)
            ax[j].plot([np.quantile(a, .25), np.quantile(a, .75)], [i, i], color=col, lw=3, solid_capstyle="round")
            ax[j].plot(np.median(a), i, "o", color=col, ms=5.5, mec="white", mew=1)
            ax[j].text(np.quantile(a, .75) + span[j] * 0.015, i, f"{np.median(a):.3f}", fontsize=7, color=INK2, ha="left", va="center")
        if met == "EF":
            ax[j].axvline(floor, color=INK2, lw=0.8, ls=":")
            ax[j].text(floor, -0.9, " noise floor", fontsize=7, color=INK2, va="center")
            ax[j].set_xlim(floor - 0.01, 1.315)
        else:
            ax[j].set_xlim(0.17, 0.50)
        ax[j].set_yticks(range(len(order)))
        ax[j].set_yticklabels([LABEL[k] for k in order])
        ax[j].set_ylim(-1.3, len(order) - 0.3)
        ax[j].set_xlabel(lab); ax[j].grid(axis="y", visible=False)
    for a, lab in zip(ax, "ab"):
        sublabel(a, f"({lab})", y=-0.22)
    fig.subplots_adjust(hspace=0.48)
    save(fig, "E2_comparison")


# ---------------------------------------------------------------- E3
def fig_e3():
    res = R["E3"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    for k in ("USBM_max", "GBR_res", "GenAttn", "LSE_dt"):
        ns = sorted(int(n) for n in res[k])
        v = np.array([parse_iqr(res[k][str(n)]) for n in ns])
        ax.fill_between(ns, v[:, 1], v[:, 2], color=PAL[k], alpha=0.25, lw=0)
        ax.plot(ns, v[:, 0], "-o", color=PAL[k], ms=4, mec="white", mew=0.8, label=LABEL[k],
                lw=1.8 if k == "LSE_dt" else 1.4, zorder=3)
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off()
    ax.set(xlabel="Number of training blasts", ylabel="RMSE in ln PPV")
    ax.legend(loc="upper right")
    save(fig, "E3_learning_curve")


# ---------------------------------------------------------------- E4
def fig_e4():
    res = R["E4"]
    keys = ["delay: train dt >= 20 ms, test dt <= 8 ms", "distance: train Dc <= 400 m, test Dc >= 700 m",
            "reference: same distribution"]
    ticks = ["Delay\ntrain $\\geq$ 20 ms\ntest $\\leq$ 8 ms", "Distance\ntrain $\\leq$ 400 m\ntest $\\geq$ 700 m",
             "Reference\nsame\ndistribution"]
    which = ["USBM_max", "GBR_res", "GenAttn", "LSE_dt"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.9)); w = 0.2
    for j, k in enumerate(which):
        meds = [parse_iqr(res[s][k])[0] for s in keys]
        x = np.arange(len(keys)) + (j - 1.5) * w
        ax.bar(x, meds, width=w - 0.025, color=PAL[k], label=LABEL[k], zorder=3, edgecolor="white", lw=0.4)
        for xi, mi in zip(x, meds):
            ax.text(xi, mi + 0.015, f"{mi:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2, rotation=90)
    ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels(ticks)
    ax.set_ylabel("RMSE in ln PPV (median)"); ax.set_ylim(0, 1.32)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.33), columnspacing=1.0, handlelength=1.2)
    save(fig, "E4_extrapolation")


# ---------------------------------------------------------------- E5
def fig_e5():
    cache = os.path.join(HERE, "e5_cache.pkl")
    if os.path.exists(cache):
        fitted, x, tau_emp = pickle.load(open(cache, "rb"))
    else:
        pool = pack(simulate_catalog(20000, 1, SP))
        fitted = []
        for sd in range(20):                     # same training sets as E2
            rng = np.random.default_rng(100 + sd)
            tr = pool.subset(rng.choice(len(pool), 150, replace=False))
            fitted.append(fit_all(tr, seed=sd, which=["LSE_dt"])["LSE_dt"])
            print("  E5 refit seed", sd, flush=True)
        blasts = simulate_catalog(3000, 7, SP)
        d = pack(blasts)
        x = d.dt * fitted[0].fhat.predict(d.lnDc)
        tau_emp = np.array([empirical_tau(b.s_true, np.log(b.ppv_true)) for b in blasts])
        pickle.dump((fitted, x, tau_emp), open(cache, "wb"))
    ab = np.array([m.theta[3:] for m in fitted])
    print("  E5 check  a median %.3f (results.json: %s)   b median %.3f (results.json: %s)"
          % (np.median(ab[:, 0]), R["E5"]["a"], np.median(ab[:, 1]), R["E5"]["b"]))
    bins = np.arange(0, 3.01, 0.2); mids = 0.5 * (bins[1:] + bins[:-1])
    med = [np.median(tau_emp[(x >= a) & (x < b)]) if ((x >= a) & (x < b)).sum() > 10 else np.nan
           for a, b in zip(bins[:-1], bins[1:])]
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.scatter(x, tau_emp, s=3, color=GREY, alpha=0.35, lw=0, label=r"Empirical $\tau$ per blast", rasterized=True)
    xx = np.linspace(0, 3, 200)
    for i, m in enumerate(fitted):
        ax.plot(xx, m.tau_of_x(xx), color=PAL["LSE_dt"], lw=0.7, alpha=0.5,
                label=r"Fitted $\tau(x)$, one line per seed" if i == 0 else None)
    ax.plot(mids, med, "o", color=INK, ms=4, mec="white", mew=0.8, label="Median per bin", zorder=4)
    ax.set_xlim(0, 3); ax.set_ylim(-0.02, 1.0)
    ax.set(xlabel=r"Delay in cycles, $x = \Delta t\,\hat{f}_d$", ylabel=r"Temperature $\tau$")
    ax.legend(loc="upper right", markerscale=1.5)
    save(fig, "E5_tau")


# ---------------------------------------------------------------- field data
def fig_field():
    rd = os.path.join(HERE, "real_data")
    T = pd.read_csv(os.path.join(rd, "field_eval.csv")); Rr = json.load(open(os.path.join(rd, "real_results.json")))
    beta = Rr["law"]["beta"]; ab = 0.5 * beta
    m = T.D >= 30
    p_att = np.empty(len(T))
    for b in T.blast.unique():                    # same LOBO fit as real_data_test.py
        tr = (T.blast != b) & m; te = (T.blast == b).values
        tau = np.sum((T.y[tr] - T.ref[tr]) * T.lnN[tr]) / np.sum(T.lnN[tr] ** 2); p_att[te] = T.ref[te] + tau * T.lnN[te]
    fig, ax = plt.subplots(1, 2, figsize=(PAGE_W, 3.1), gridspec_kw={"width_ratios": [1, 1.15]})
    obs = np.exp(T.y[m])
    for lab, p, col, mk in [(r"W per delay ($\tau$ = 0)", T.ref, PAL["USBM_max"], "o"),
                            (rf"Total W ($\tau$ = {ab:.2f})", T.ref + ab * T.lnN, GREY, "s"),
                            (r"Attention, LOBO $\tau$", p_att, PAL["LSE_dt"], "D")]:
        ax[0].scatter(obs, np.exp(np.asarray(p)[m.values]), s=24, color=col, marker=mk, edgecolor="white", lw=0.6, label=lab, zorder=3)
    lim = [3, 400]
    ax[0].plot(lim, lim, color=INK2, lw=0.8)
    for f in (2, 0.5):
        ax[0].plot(lim, np.array(lim) * f, color=INK2, lw=0.6, ls=":")
    ax[0].text(4.2, 12, "factor 2", fontsize=7, color=INK2, ha="left")
    ax[0].set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Measured PPV (mm/s)", ylabel="Predicted PPV (mm/s)")
    tk = [3, 10, 30, 100, 300]
    for f in (ax[0].set_xticks, ax[0].set_yticks): f(tk)
    ax[0].set_xticklabels([str(t) for t in tk]); ax[0].set_yticklabels([str(t) for t in tk]); ax[0].minorticks_off()
    ax[0].legend(loc="lower right", handletextpad=0.2, borderaxespad=0.1, labelspacing=0.3, fontsize=7)
    mk = {"Test_1": "o", "Test_2": "s", "Test_3": "^", "Test_4": "D"}
    for b, g in T[m].groupby("blast"):
        ax[1].scatter(g.f_sig, g.tau, marker=mk[b], s=30, color=PAL["LSE_dt"], edgecolor="white", lw=0.6,
                      label=b.replace("Test_", "Test ") + f" ({int(g.N.iloc[0])} holes)", zorder=3)
    tg = Rr["tau_global"]
    for yv, col, ls, txt in [(tg, PAL["LSE_dt"], "--", rf"global $\tau$ = {tg:.2f}"),
                             (ab, GREY, ":", rf"total W, $\tau$ = {ab:.2f}"),
                             (0.0, PAL["USBM_max"], ":", r"W per delay, $\tau$ = 0")]:
        ax[1].axhline(yv, color=col, lw=0.9, ls=ls)
        ax[1].text(20.0, yv + 0.015, txt, fontsize=7, color=INK2, ha="center", va="bottom")
    ax[1].set(xlabel="Centroid frequency of the signature hole (Hz)", ylabel=r"Empirical temperature $\tau$", ylim=(-0.06, 0.9))
    ax[1].legend(loc="center", bbox_to_anchor=(0.5, 0.62))
    for a, lab in zip(ax, "ab"):
        sublabel(a, f"({lab})", y=-0.2)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "field_data")


if __name__ == "__main__":
    print("fonts:", SANS[0], "/", SERIF[0])
    if "--only-e5" in sys.argv:
        fig_e5(); sys.exit()
    e1 = fig_e1(); print("  E1 check:", [round(v, 2) for v in e1], "results.json:", R["E1"])
    fig_e2(); fig_e3(); fig_e4(); fig_field()
    if not SKIP_E5:
        fig_e5()
