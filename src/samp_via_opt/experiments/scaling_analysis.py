"""
Post-processing of the d / kappa scaling experiment (ULA vs ULMC).

Question
    How many gradient evaluations T are needed to reach W2 <= eps, and how does T grow with
    kappa and d?  Proposal: ULA ~ kappa, ULMC ~ sqrt(kappa) (and d vs sqrt(d)).

Definitions
    - convergence:  floor-corrected, median-smoothed W2 is <= eps from step T onwards until the
                    end of the budget (see check_W2_converged in evaluation.py).
                    Primary criterion: whitened W2 (eps_white). Robustness: normalised Euclidean W2.
    - T(sampler, d, kappa) = min over step size h (and gamma for ULMC) of the cost until convergence.
                    Cost = gradient evaluations (primary) or wall-clock seconds (robustness).
    - censored:     no configuration converged within the budget n_max(kappa) = max(2000, 10*kappa).
                    Such cells are drawn as hollow triangles at the budget and excluded from all fits.

Outputs (in --out)
    fig1_T_vs_kappa.png     top: T vs kappa, bottom: local slope d*log(T) / d*log(kappa), one column per d
    fig2_speedup.png        T_ULA / T_ULMC vs kappa, reference sqrt(kappa)
    fig3_T_vs_dim.png       T(d)/T(d_min) vs d for several kappa (only if >= 2 dimensions)
    table_T.csv             T per (kappa) x (sampler, d)
    exponents.csv           kappa-exponents, speed-up exponents, d-exponents (one row per fit)
    best_configs.csv        chosen (h, gamma) and quality flags for every cell
    checks.csv              sanity checks per sampler
    robustness.csv          kappa-exponents for other criteria / cost definitions
    diagnostics/            step size & gamma of the optimum, U-shape of final error over h

Usage
    python scaling_analysis.py dimension_conditioning_scaling_results.csv --out analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAMPLERS = ("ULA", "ULMC")
COLORS = {"ULA": "tab:blue", "ULMC": "tab:red"}
# exponents predicted by the proposal: T ~ kappa^a * d^b
PROPOSAL = {"ULA": {"kappa": 1.0, "dim": 1.0}, "ULMC": {"kappa": 0.5, "dim": 0.5}}
KEY = ["sampler", "dimension", "kappa"]
FIRST_CHECKPOINT = 10  # make_eval_steps(first=...): converged here => left-censored
NEAR_OPT = 1.2  # "near optimal" = within 20 % of the best T

# gamma values in the experiment: c/sqrt(kappa) for c in (1, 2, 4), plus the constants sqrt(2) and 2
GAMMA_ORDER = ["1/√κ", "2/√κ", "4/√κ", "√2", "2"]

CRITERIA = {
    "white": dict(
        sfx="_white",
        final="final_w2_white",
        eps="eps_white",
        floor_ok="floor_ok_white",
        borderline="borderline_white",
        label="whitened W2",
    ),
    "euclid": dict(
        sfx="",
        final="final_w2_norm",
        eps="eps",
        floor_ok="floor_ok",
        borderline="borderline",
        label="normalised Euclidean W2",
    ),
}


# ---------------------------------------------------------------------------------
# 0. load: one unified table, independent of the criterion / cost that is analysed
# ---------------------------------------------------------------------------------
def gamma_label(gamma: float, kappa: float) -> str:
    """Name of the gamma rule (the raw value changes with kappa for the c/sqrt(kappa) rules)."""
    if pd.isna(gamma):
        return "-"
    if np.isclose(gamma, np.sqrt(2), rtol=1e-4):
        return "√2"
    if np.isclose(gamma, 2.0, rtol=1e-4):
        return "2"
    for c in (1, 2, 4):
        if np.isclose(gamma * np.sqrt(kappa), c, rtol=1e-4):
            return f"{c}/√κ"
    return f"{gamma:.3g}"


def add_max_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Budget n_max per cell. The budget now depends on kappa (make_eval_steps), so it can no longer be a
    global constant. If the CSV has no `max_steps` column, recover it from the stored timing:
    sec_per_step = runtime_sec_total / (n_max * n_step_sizes).
    """
    if "max_steps" in df:
        return df
    n_h = df.groupby(KEY + ["gamma_label"])["step_size"].transform("nunique")
    return df.assign(
        max_steps=np.round(df["runtime_sec_total"] / (df["sec_per_step"] * n_h))
    )


def prepare(raw: pd.DataFrame, criterion: str, cost: str) -> pd.DataFrame:
    """Select samplers, pick the criterion, and define the cost T (inf = did not converge)."""
    c = CRITERIA[criterion]
    df = raw[raw["sampler"].isin(SAMPLERS)].copy()
    if df.empty:
        raise SystemExit(f"no rows for samplers {SAMPLERS} in the csv")
    df["gamma_label"] = [gamma_label(g, k) for g, k in zip(df["gamma"], df["kappa"])]
    df = add_max_steps(df)

    sfx = c["sfx"]
    ok = df[f"converged{sfx}"].astype(bool) & ~df["diverged"].astype(bool)
    df["steps_conv"] = np.where(ok, df[f"steps_to_conv{sfx}"], np.inf)
    unit = df["grads_per_step"] if cost == "grads" else df["sec_per_step"]
    df["T"] = df["steps_conv"] * unit
    df["budget"] = df["max_steps"] * unit  # right-censoring level, in the unit of T
    df["final_err"] = df[c["final"]].replace([np.inf, -np.inf], np.nan)
    df["eps_used"] = df[c["eps"]]
    df["floor_ok_used"] = df[c["floor_ok"]].astype(bool)
    df["borderline_used"] = df[c["borderline"]].astype(bool)
    return df


# ---------------------------------------------------------------------------------
# 1. best configuration per cell + quality flags
# ---------------------------------------------------------------------------------
def best_configs(df: pd.DataFrame) -> pd.DataFrame:
    """
    T* = min over (h, gamma) per (sampler, d, kappa).
    Cells where nothing converged keep the run with the smallest final error (for diagnostics only).

    Note: ULMC has 5 gamma values and ULA has none, so ULMC gets more 'tries' at the minimum of a noisy
    quantity. n_near_opt (how many (h, gamma) are within 20 % of T*) tells how flat the optimum is.
    """
    best = (
        df.sort_values(["T", "final_err"])
        .drop_duplicates(KEY)
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    tmin = df.groupby(KEY)["T"].transform("min")
    near = (
        df[np.isfinite(df["T"]) & (df["T"] <= NEAR_OPT * tmin)]
        .groupby(KEY)
        .size()
        .rename("n_near_opt")
        .reset_index()
    )
    best = best.merge(near, on=KEY, how="left").fillna({"n_near_opt": 0})

    h_lo, h_hi = df["step_size"].min(), df["step_size"].max()
    best["censored"] = ~np.isfinite(best["T"])
    best["h_at_edge"] = np.isclose(best["step_size"], h_lo) | np.isclose(
        best["step_size"], h_hi
    )
    best["left_censored"] = ~best["censored"] & (best["steps_conv"] <= FIRST_CHECKPOINT)
    return best


def checks(df: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    per_cell = best.groupby("sampler").agg(
        cells=("T", "size"),
        censored=("censored", "sum"),
        h_at_grid_edge=("h_at_edge", "sum"),
        left_censored=("left_censored", "sum"),
        borderline_optimum=("borderline_used", "sum"),
        median_n_near_opt=("n_near_opt", "median"),
    )
    per_run = df.groupby("sampler").agg(
        diverged_share=("diverged", "mean"),
        floor_ok_everywhere=("floor_ok_used", "all"),
    )
    return per_cell.join(per_run)


# ---------------------------------------------------------------------------------
# 2. exponents (all fits are log-log OLS on uncensored cells; se = scatter around the line)
# ---------------------------------------------------------------------------------
def ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    X = np.c_[np.ones(n), x]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    if n == 2:
        return float(beta[1]), np.nan
    resid = y - X @ beta
    se = np.sqrt((resid @ resid) / (n - 2) / ((x - x.mean()) ** 2).sum())
    return float(beta[1]), float(se)


def kappa_exponents(best: pd.DataFrame, kappa_min: float) -> pd.DataFrame:
    """alpha in T ~ kappa^alpha per (sampler, d), fitted on kappa >= kappa_min (asymptotic regime)."""
    rows = []
    for (s, d), g in best[~best["censored"]].groupby(["sampler", "dimension"]):
        g = g[g["kappa"] >= kappa_min]
        a, se = ols_slope(np.log(g["kappa"].to_numpy()), np.log(g["T"].to_numpy()))
        rows.append(
            dict(
                sampler=s,
                dimension=d,
                alpha=a,
                se=se,
                n_points=len(g),
                proposal=PROPOSAL[s]["kappa"],
            )
        )
    return pd.DataFrame(
        rows, columns=["sampler", "dimension", "alpha", "se", "n_points", "proposal"]
    )


def local_slopes(best: pd.DataFrame) -> pd.DataFrame:
    """Slope of log T between neighbouring kappa values. Shows where a power law starts to hold."""
    rows = []
    for (s, d), g in best[~best["censored"]].groupby(["sampler", "dimension"]):
        g = g.sort_values("kappa")
        lk, lt = np.log(g["kappa"].to_numpy()), np.log(g["T"].to_numpy())
        for i in range(len(g) - 1):
            rows.append(
                dict(
                    sampler=s,
                    dimension=d,
                    kappa_mid=float(np.exp((lk[i] + lk[i + 1]) / 2)),
                    slope=(lt[i + 1] - lt[i]) / (lk[i + 1] - lk[i]),
                )
            )
    return pd.DataFrame(rows, columns=["sampler", "dimension", "kappa_mid", "slope"])


def speedup_table(best: pd.DataFrame) -> pd.DataFrame:
    """
    T_ULA / T_ULMC per (d, kappa). If ULA is censored but ULMC converged, the ratio is only a lower
    bound (budget_ULA / T_ULMC) and flagged. If ULMC is censored the cell is dropped.
    """
    cols = [
        "dimension",
        "kappa",
        "T_ULA",
        "T_ULMC",
        "budget_ULA",
        "speedup",
        "is_lower_bound",
    ]
    if not set(SAMPLERS) <= set(best["sampler"]):
        return pd.DataFrame(columns=cols)
    w = best.pivot_table(
        index=["dimension", "kappa"],
        columns="sampler",
        values=["T", "budget"],
        aggfunc="first",
    )
    t_ula, t_ulmc = w[("T", "ULA")], w[("T", "ULMC")]
    sp = pd.DataFrame(
        {"T_ULA": t_ula, "T_ULMC": t_ulmc, "budget_ULA": w[("budget", "ULA")]}
    )
    both = np.isfinite(t_ula) & np.isfinite(t_ulmc)
    sp["is_lower_bound"] = ~np.isfinite(t_ula) & np.isfinite(t_ulmc)
    sp["speedup"] = np.where(both, t_ula / t_ulmc, np.nan)
    sp.loc[sp["is_lower_bound"], "speedup"] = (sp["budget_ULA"] / sp["T_ULMC"])[
        sp["is_lower_bound"]
    ]
    return sp.reset_index()[cols]


def speedup_exponents(sp: pd.DataFrame, kappa_min: float) -> pd.DataFrame:
    rows = []
    for d, g in sp.groupby("dimension"):
        g = g[(g["kappa"] >= kappa_min) & ~g["is_lower_bound"] & g["speedup"].notna()]
        a, se = ols_slope(
            np.log(g["kappa"].to_numpy()), np.log(g["speedup"].to_numpy())
        )
        rows.append(dict(dimension=d, alpha=a, se=se, n_points=len(g), proposal=0.5))
    return pd.DataFrame(
        rows, columns=["dimension", "alpha", "se", "n_points", "proposal"]
    )


def dim_exponents(best: pd.DataFrame, kappa_min: float) -> pd.DataFrame:
    """
    beta in T ~ d^beta at fixed kappa (needs >= 2 dimensions). Only meaningful for the gradient-count
    cost: wall-clock per step grows with d because the number of chains is scaled with d.
    """
    rows = []
    g = best[~best["censored"] & (best["kappa"] >= kappa_min)]
    for (s, k), gk in g.groupby(["sampler", "kappa"]):
        if gk["dimension"].nunique() < 2:
            continue
        b, se = ols_slope(
            np.log(gk["dimension"].to_numpy()), np.log(gk["T"].to_numpy())
        )
        rows.append(
            dict(
                sampler=s,
                kappa=k,
                beta=b,
                se=se,
                n_points=len(gk),
                proposal=PROPOSAL[s]["dim"],
            )
        )
    return pd.DataFrame(
        rows, columns=["sampler", "kappa", "beta", "se", "n_points", "proposal"]
    )


# ---------------------------------------------------------------------------------
# 3. main figures
# ---------------------------------------------------------------------------------
def _legend_label(sampler: str, a: float, se: float) -> str:
    if not np.isfinite(a):
        return sampler
    return f"{sampler}  α={a:.2f}" + (f"±{se:.2f}" if np.isfinite(se) else "")


def fig_T_vs_kappa(best, kexp, ls, kappa_min, ylabel, eps_label, path):
    dims = sorted(best["dimension"].unique())
    fit = {(r.sampler, r.dimension): (r.alpha, r.se) for r in kexp.itertuples()}
    fig, axes = plt.subplots(
        2,
        len(dims),
        figsize=(4.3 * len(dims), 6.2),
        sharex=True,
        squeeze=False,
        gridspec_kw=dict(height_ratios=[2, 1]),
    )
    k_lo = best["kappa"].min()
    for j, d in enumerate(dims):
        ax, ax2 = axes[0, j], axes[1, j]
        for s in SAMPLERS:
            g = best[(best["sampler"] == s) & (best["dimension"] == d)].sort_values(
                "kappa"
            )
            if g.empty:
                continue
            ok = ~g["censored"]
            a, se = fit.get((s, d), (np.nan, np.nan))
            ax.plot(
                g.loc[ok, "kappa"],
                g.loc[ok, "T"],
                "o-",
                color=COLORS[s],
                label=_legend_label(s, a, se),
            )
            ax.plot(
                g.loc[~ok, "kappa"],
                g.loc[~ok, "budget"],
                "v",
                mfc="none",
                color=COLORS[s],
            )
            gm = g[ok & (g["kappa"] >= kappa_min)]
            if len(
                gm
            ):  # reference slope of the proposal, anchored at the first asymptotic point
                k0, t0 = gm.iloc[0]["kappa"], gm.iloc[0]["T"]
                kk = g.loc[g["kappa"] >= kappa_min, "kappa"].to_numpy()
                ax.plot(
                    kk,
                    t0 * (kk / k0) ** PROPOSAL[s]["kappa"],
                    "--",
                    color=COLORS[s],
                    alpha=0.45,
                    lw=1,
                )
            ll = ls[(ls["sampler"] == s) & (ls["dimension"] == d)]
            ax2.plot(ll["kappa_mid"], ll["slope"], "o-", color=COLORS[s])
        if kappa_min > k_lo:
            for a_ in (ax, ax2):
                a_.axvspan(k_lo / 1.3, kappa_min / 1.15, color="0.93", zorder=0)
        for p in (1.0, 0.5):
            ax2.axhline(p, c="gray", ls=":", lw=1)
        ax.set(xscale="log", yscale="log", title=f"d = {d}")
        ax2.set(xscale="log", xlabel="κ (condition number)")
        ax.legend(
            fontsize=8,
            title="fit κ ≥ %g; dashed: proposal slope" % kappa_min,
            title_fontsize=7,
        )
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel("local slope  d log T / d log κ")
    fig.suptitle(
        f"Cost until {eps_label} ≤ ε  (best h, γ per point;  ▽ = not converged within budget)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_speedup(sp, kappa_min, path):
    sp = sp[sp["speedup"].notna()]
    if sp.empty:
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for i, (d, g) in enumerate(sp.groupby("dimension")):
        g = g.sort_values("kappa")
        c = f"C{i}"
        ex = g[~g["is_lower_bound"]]
        lb = g[g["is_lower_bound"]]
        ax.plot(ex["kappa"], ex["speedup"], "o-", color=c, label=f"d = {d}")
        ax.plot(
            lb["kappa"], lb["speedup"], "^", mfc="none", color=c
        )  # ULA censored: lower bound
        a = ex[ex["kappa"] >= kappa_min]
        if len(a):
            k0, s0 = a.iloc[0]["kappa"], a.iloc[0]["speedup"]
            kk = g.loc[g["kappa"] >= kappa_min, "kappa"].to_numpy()
            ax.plot(kk, s0 * (kk / k0) ** 0.5, "--", color=c, alpha=0.45, lw=1)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="κ",
        ylabel="T_ULA / T_ULMC",
        title="Speed-up of ULMC over ULA (dashed: √κ)",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_T_vs_dim(best, kappa_min, path) -> bool:
    dims = sorted(best["dimension"].unique())
    if len(dims) < 2:
        return False
    d = np.array(dims, dtype=float)
    fig, axes = plt.subplots(
        1, len(SAMPLERS), figsize=(4.5 * len(SAMPLERS), 3.6), sharey=True, squeeze=False
    )
    for ax, s in zip(axes[0], SAMPLERS):
        g_s = best[
            (best["sampler"] == s) & ~best["censored"] & (best["kappa"] >= kappa_min)
        ]
        for k, g in g_s.groupby("kappa"):
            g = g.sort_values("dimension")
            if g["dimension"].nunique() < 2:
                continue
            ax.plot(g["dimension"], g["T"] / g["T"].iloc[0], "o-", label=f"κ = {k:g}")
        for p, ls_ in ((1.0, ":"), (0.5, "--")):
            ax.plot(d, (d / d[0]) ** p, c="k", ls=ls_, lw=1, label=f"d^{p:g}")
        ax.set(xscale="log", yscale="log", xlabel="d", title=s)
    axes[0, 0].set_ylabel("T(d) / T(d_min)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------------
# 4. diagnostics (explain *why* the numbers look the way they do)
# ---------------------------------------------------------------------------------
def fig_diag_hyperparameters(best, path):
    dims = sorted(best["dimension"].unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for s in SAMPLERS:
        for i, d in enumerate(dims):
            g = best[
                (best["sampler"] == s) & (best["dimension"] == d) & ~best["censored"]
            ].sort_values("kappa")
            if g.empty:
                continue
            a1.plot(
                g["kappa"],
                g["step_size"],
                "o-",
                color=COLORS[s],
                alpha=0.4 + 0.6 * (i + 1) / len(dims),
                ms=4,
                label=f"{s}, d={d}",
            )
            e = g[g["h_at_edge"]]
            a1.plot(
                e["kappa"], e["step_size"], "kx", ms=8
            )  # optimum at edge of the h grid -> extend grid
    a1.axhline(2.0, c="r", ls=":", lw=1)
    a1.set(
        xscale="log",
        yscale="log",
        xlabel="κ",
        ylabel="best step size h*",
        title="optimal step size (red: 2/L)",
    )
    a1.legend(fontsize=6)
    for i, d in enumerate(dims):
        g = best[
            (best["sampler"] == "ULMC") & (best["dimension"] == d) & ~best["censored"]
        ].sort_values("kappa")
        y = [
            GAMMA_ORDER.index(x) if x in GAMMA_ORDER else np.nan
            for x in g["gamma_label"]
        ]
        a2.plot(
            g["kappa"],
            np.array(y) + 0.12 * (i - (len(dims) - 1) / 2),
            "o",
            label=f"d={d}",
        )
    a2.set_yticks(range(len(GAMMA_ORDER)))
    a2.set_yticklabels(GAMMA_ORDER)
    a2.set(
        xscale="log", xlabel="κ", title="ULMC: which γ wins? (c/√κ = momentum regime)"
    )
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_diag_ushape(df, best, path):
    ks = sorted({df["kappa"].min(), df["kappa"].max()})
    ds = sorted({df["dimension"].min(), df["dimension"].max()})
    fig, axes = plt.subplots(
        len(ks),
        len(ds),
        figsize=(4.2 * len(ds), 3.3 * len(ks)),
        squeeze=False,
        sharex=True,
    )
    for i, k in enumerate(ks):
        for j, d in enumerate(ds):
            ax = axes[i, j]
            for s in SAMPLERS:
                b = best[
                    (best["sampler"] == s)
                    & (best["dimension"] == d)
                    & (best["kappa"] == k)
                ]
                r = df[
                    (df["sampler"] == s) & (df["dimension"] == d) & (df["kappa"] == k)
                ]
                if b.empty or r.empty:
                    continue
                r = r[r["gamma_label"] == b["gamma_label"].iloc[0]].sort_values(
                    "step_size"
                )
                ax.plot(
                    r["step_size"], r["final_err"], "o-", ms=3, color=COLORS[s], label=s
                )
                if not b["censored"].iloc[0]:
                    ax.plot(b["step_size"], b["final_err"], "k*", ms=9)
                ax.axhline(r["eps_used"].iloc[0], c="gray", ls=":")
            ax.set(xscale="log", yscale="log", title=f"d={d}, κ={k:g}")
    axes[-1, 0].set_xlabel("step size h")
    axes[0, 0].set_ylabel("error at end of budget")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "small h: not converged within budget   |   large h: discretisation bias   (★ = chosen h*, dotted = ε)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------
def fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv")
    ap.add_argument("--out", default="analysis")
    ap.add_argument(
        "--kappa-min",
        type=float,
        default=100.0,
        help="exponents are fitted for kappa >= this value (below, T is dominated by constants)",
    )
    ap.add_argument("--criterion", choices=list(CRITERIA), default="euclid")
    ap.add_argument("--cost", choices=["grads", "wall"], default="grads")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(a.csv)

    df = prepare(raw, a.criterion, a.cost)
    best = best_configs(df)
    best.to_csv(out / "best_configs.csv", index=False)

    chk = checks(df, best)
    chk.to_csv(out / "checks.csv")
    print(
        f"\n=== sanity ({CRITERIA[a.criterion]['label']}, cost={a.cost}) ===\n{chk.to_string()}"
    )

    kexp = kappa_exponents(best, a.kappa_min)
    ls = local_slopes(best)
    sp = speedup_table(best)
    sexp = speedup_exponents(sp, a.kappa_min)
    dexp = dim_exponents(best, a.kappa_min) if a.cost == "grads" else pd.DataFrame()

    tidy = pd.concat(
        [
            kexp.assign(kind="kappa_exponent"),
            sexp.assign(kind="speedup_exponent", sampler="ULA/ULMC"),
            dexp.rename(columns={"beta": "alpha"}).assign(kind="dim_exponent")
            if len(dexp)
            else None,
        ],
        ignore_index=True,
    )
    tidy.to_csv(out / "exponents.csv", index=False)
    print(
        f"\n=== T ~ kappa^alpha, fitted for kappa >= {a.kappa_min:g} (se = scatter around the line) ===\n{fmt(kexp)}"
    )
    if len(sexp):
        print(f"\n=== speed-up ~ kappa^alpha (proposal 0.5) ===\n{fmt(sexp)}")
    if len(dexp):
        print(
            f"\n=== T ~ d^beta at fixed kappa, median over kappa ===\n"
            f"{fmt(dexp.groupby('sampler')[['beta', 'proposal']].median().reset_index())}"
        )
    for s in SAMPLERS:
        r = df[df["sampler"] == s]["sec_per_step"].median()
        print(f"median sec_per_step {s}: {r:.3g}")

    ylabel = (
        "gradient evaluations" if a.cost == "grads" else "wall-clock seconds (ensemble)"
    )
    eps_label = "W2 (" + CRITERIA[a.criterion]["label"] + ")"
    fig_T_vs_kappa(
        best, kexp, ls, a.kappa_min, ylabel, eps_label, out / "fig1_T_vs_kappa.png"
    )
    fig_speedup(sp, a.kappa_min, out / "fig2_speedup.png")
    if a.cost == "grads":
        fig_T_vs_dim(best, a.kappa_min, out / "fig3_T_vs_dim.png")
    best.assign(T=best["T"].replace(np.inf, np.nan)).pivot_table(
        index="kappa", columns=["sampler", "dimension"], values="T", aggfunc="first"
    ).to_csv(out / "table_T.csv")
    fig_diag_hyperparameters(best, out / "diagnostics" / "hyperparameters.png")
    fig_diag_ushape(df, best, out / "diagnostics" / "ushape.png")

    # robustness: same kappa-exponents under the other criterion / cost definition (fits only, no figures)
    rob = []
    for crit, cost in [("white", "grads"), ("euclid", "grads"), ("white", "wall")]:
        b = best_configs(prepare(raw, crit, cost))
        rob.append(kappa_exponents(b, a.kappa_min).assign(criterion=crit, cost=cost))
    rob = pd.concat(rob, ignore_index=True)
    rob.to_csv(out / "robustness.csv", index=False)
    piv = rob.pivot_table(
        index=["sampler", "dimension"], columns=["criterion", "cost"], values="alpha"
    ).round(2)
    print(f"\n=== robustness of the kappa-exponent ===\n{piv.to_string()}")
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
