"""
Post-processing of the d / kappa scaling experiment (ULA vs ULMC vs RMULMC).

Question
    How many gradient evaluations T are needed to reach W2 <= eps, and how does T grow with
    kappa and d?  Reference (proposal): ULA ~ kappa, momentum samplers ~ sqrt(kappa).
    The reference exponents of every sampler live in one place: the SAMPLERS table below.

Definitions
    - convergence:  floor-corrected, median-smoothed W2 is <= eps from step T onwards until the
                    end of the budget (see check_W2_converged in evaluation.py).
                    Primary criterion: whitened W2 (eps_white). Robustness: normalised Euclidean W2.
    - T(sampler, d, kappa) = min over step size h (and gamma, if the sampler has one) of the cost
                    until convergence. Cost = gradient evaluations (primary; RMULMC counts 2 per step,
                    see `grads_per_step` in the csv) or wall-clock seconds (robustness).
    - censored:     no configuration converged within the budget n_max(kappa)
                    (= min(max(2000, 10*kappa), 50000) with the default make_eval_steps; read from the csv).
                    Such cells are drawn as hollow triangles at the budget and excluded from all fits.

Outputs (in --out)
    fig1_T_vs_kappa.png     top: T vs kappa, bottom: local slope d log T / d log kappa, one column per d
    fig2_speedup.png        T_ULA / T_sampler vs kappa for every other sampler, dashed = reference slope
    fig3_T_vs_dim.png       T(d)/T(d_min) vs d for several kappa (only if >= 2 dimensions)
    table_T.csv             T per (kappa) x (sampler, d)
    exponents.csv           kappa-exponents, speed-up exponents, d-exponents (one row per fit)
    best_configs.csv        chosen (h, gamma) and quality flags for every cell
    checks.csv              sanity checks per sampler
    robustness.csv          kappa-exponents for other criteria / cost definitions
    diagnostics/            step size & gamma of the optimum, U-shape of final error over h

How to read fig1
    Points on a straight line (log-log) = power law, its slope alpha is in the legend.
    The lower row shows the same data as local slopes: a plateau = clean power law, and
    that is where the fit region (kappa >= --kappa-min) should start.

Usage
    python scaling_analysis.py dimension_conditioning_scaling_results.csv --out analysis
    python scaling_analysis.py results.csv --usetex      # real LaTeX (needs latex + dvipng installed)

Adding a sampler (e.g. RMULA): add one entry to SAMPLERS, nothing else has to change.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------
# configuration: everything sampler-specific is in this one table
# ---------------------------------------------------------------------------------
BASELINE = "ULA"  # speed-ups are measured relative to this sampler

# kappa_ref / dim_ref: exponents a, b of the reference law T ~ kappa^a * d^b (dashed lines in the plots).
# NOTE: the RMULMC values are a hypothesis (same acceleration as ULMC in kappa, d^(1/3) as in the
# Shen & Lee analysis). Change them here if you want to compare against a different claim.
SAMPLERS: dict[str, dict] = {
    "ULA": dict(
        color="tab:blue", marker="o", kappa_ref=1.0, dim_ref=1.0, uses_gamma=False
    ),
    "ULMC": dict(
        color="tab:red", marker="s", kappa_ref=0.5, dim_ref=0.5, uses_gamma=True
    ),
    "RMULMC": dict(
        color="tab:green", marker="^", kappa_ref=0.5, dim_ref=1 / 3, uses_gamma=True
    ),
}

KEY = ["sampler", "dimension", "kappa"]
NEAR_OPT = 1.2  # "near optimal" = within 20 % of the best T

# gamma values in the experiment: c/sqrt(kappa) for c in (1, 2, 4), plus the constants sqrt(2) and 2.
# key (used in csv files) -> LaTeX label (used in plots)
GAMMA_RULES = {
    "1/sqrt(k)": r"$1/\sqrt{\kappa}$",
    "2/sqrt(k)": r"$2/\sqrt{\kappa}$",
    "4/sqrt(k)": r"$4/\sqrt{\kappa}$",
    "sqrt(2)": r"$\sqrt{2}$",
    "2": r"$2$",
}

CRITERIA = {
    "white": dict(
        sfx="_white",
        final="final_w2_white",
        eps="eps_white",
        floor_ok="floor_ok_white",
        borderline="borderline_white",
        label=r"whitened $W_2$",
    ),
    "euclid": dict(
        sfx="",
        final="final_w2_norm",
        eps="eps",
        floor_ok="floor_ok",
        borderline="borderline",
        label=r"normalised Euclidean $W_2$",
    ),
}


# ---------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------
def set_style(usetex: bool) -> None:
    """
    LaTeX look for all figures. Default: matplotlib's built-in math renderer with Computer Modern
    (no LaTeX installation needed, looks like LaTeX). usetex=True: the real thing.
    """
    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"]
            if usetex
            else ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,  # cmr10 has no unicode minus
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        }
    )


def present(df: pd.DataFrame) -> list[str]:
    """Samplers that occur in df, in the order of the SAMPLERS table."""
    have = set(df["sampler"])
    return [s for s in SAMPLERS if s in have]


def exp_str(p: float) -> str:
    """0.5 -> '1/2', 1/3 -> '1/3', 1.0 -> '1' (for labels like $d^{1/3}$)."""
    f = Fraction(p).limit_denominator(6)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def save(fig, path, title: str | None = None) -> None:
    if title:
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------
# 0. load: one unified table, independent of the criterion / cost that is analysed
# ---------------------------------------------------------------------------------
def gamma_label(gamma: float, kappa: float) -> str:
    """Key of the gamma rule (the raw value changes with kappa for the c/sqrt(kappa) rules)."""
    if pd.isna(gamma):
        return "-"  # samplers without gamma (ULA, RMULMC)
    if np.isclose(gamma, np.sqrt(2), rtol=1e-4):
        return "sqrt(2)"
    if np.isclose(gamma, 2.0, rtol=1e-4):
        return "2"
    for c in (1, 2, 4):
        if np.isclose(gamma * np.sqrt(kappa), c, rtol=1e-4):
            return f"{c}/sqrt(k)"
    return f"{gamma:.3g}"


def add_max_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Budget n_max per cell. The budget depends on kappa (make_eval_steps), so it is not a global
    constant. Newer csv files have a `max_steps` column. For older ones recover it from the stored
    timing: sec_per_step = runtime_sec_total / (n_max * n_step_sizes).
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
    df = raw[raw["sampler"].isin(list(SAMPLERS))].copy()
    if df.empty:
        raise SystemExit(f"none of the samplers {list(SAMPLERS)} found in the csv")
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
def best_configs(df: pd.DataFrame, first_checkpoint: int) -> pd.DataFrame:
    """
    T* = min over (h, gamma) per (sampler, d, kappa).
    Cells where nothing converged keep the run with the smallest final error (for diagnostics only).

    Note: ULMC has 5 gamma values, ULA and RMULMC have none, so ULMC gets more 'tries' at the minimum
    of a noisy quantity. n_near_opt (how many (h, gamma) are within 20 % of T*) tells how flat the
    optimum is.
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
    # converged already at the first checkpoint -> true T may be smaller (left-censored)
    best["left_censored"] = ~best["censored"] & (best["steps_conv"] <= first_checkpoint)
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
        median_sec_per_step=("sec_per_step", "median"),
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
            dict(sampler=s, dimension=d, alpha=a, se=se, n_points=len(g),
                 proposal=SAMPLERS[s]["kappa_ref"])
        )  # fmt: skip
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
    T_BASELINE / T_sampler per (sampler, d, kappa) for every sampler other than the baseline.
    If the baseline is censored but the sampler converged, the ratio is only a lower bound
    (budget_baseline / T_sampler) and flagged. If the sampler is censored the cell is dropped.
    """
    cols = ["sampler", "dimension", "kappa", "T_base", "T_sampler", "budget_base",
            "speedup", "is_lower_bound"]  # fmt: skip
    if BASELINE not in set(best["sampler"]):
        return pd.DataFrame(columns=cols)

    base = best[best["sampler"] == BASELINE][
        ["dimension", "kappa", "T", "budget"]
    ].rename(columns={"T": "T_base", "budget": "budget_base"})
    other = best[best["sampler"] != BASELINE][
        ["sampler", "dimension", "kappa", "T"]
    ].rename(columns={"T": "T_sampler"})
    sp = other.merge(base, on=["dimension", "kappa"])

    base_ok, s_ok = np.isfinite(sp["T_base"]), np.isfinite(sp["T_sampler"])
    sp["is_lower_bound"] = ~base_ok & s_ok
    sp["speedup"] = np.where(base_ok & s_ok, sp["T_base"] / sp["T_sampler"], np.nan)
    lb = sp["is_lower_bound"]
    sp.loc[lb, "speedup"] = (sp["budget_base"] / sp["T_sampler"])[lb]
    return (
        sp[cols].sort_values(["sampler", "dimension", "kappa"]).reset_index(drop=True)
    )


def speedup_exponents(sp: pd.DataFrame, kappa_min: float) -> pd.DataFrame:
    """alpha in speedup ~ kappa^alpha; the reference is kappa_ref(baseline) - kappa_ref(sampler)."""
    rows = []
    for (s, d), g in sp.groupby(["sampler", "dimension"]):
        g = g[(g["kappa"] >= kappa_min) & ~g["is_lower_bound"] & g["speedup"].notna()]
        a, se = ols_slope(
            np.log(g["kappa"].to_numpy()), np.log(g["speedup"].to_numpy())
        )
        ref = SAMPLERS[BASELINE]["kappa_ref"] - SAMPLERS[s]["kappa_ref"]
        rows.append(
            dict(sampler=f"{BASELINE}/{s}", dimension=d, alpha=a, se=se, n_points=len(g),
                 proposal=ref)
        )  # fmt: skip
    return pd.DataFrame(
        rows, columns=["sampler", "dimension", "alpha", "se", "n_points", "proposal"]
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
            dict(sampler=s, kappa=k, beta=b, se=se, n_points=len(gk),
                 proposal=SAMPLERS[s]["dim_ref"])
        )  # fmt: skip
    return pd.DataFrame(
        rows, columns=["sampler", "kappa", "beta", "se", "n_points", "proposal"]
    )


# ---------------------------------------------------------------------------------
# 3. main figures
# ---------------------------------------------------------------------------------
def _fit_label(sampler: str, a: float, se: float) -> str:
    if not np.isfinite(a):
        return sampler
    err = rf"\pm{se:.2f}" if np.isfinite(se) else ""
    return rf"{sampler}: $\alpha={a:.2f}{err}$"


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
        for s in present(best):
            cfg = SAMPLERS[s]
            g = best[(best["sampler"] == s) & (best["dimension"] == d)].sort_values(
                "kappa"
            )
            ok = ~g["censored"]
            a, se = fit.get((s, d), (np.nan, np.nan))
            # converged cells: line with markers; censored cells: hollow triangle at the budget
            ax.plot(g.loc[ok, "kappa"], g.loc[ok, "T"], marker=cfg["marker"], ls="-",
                    color=cfg["color"], label=_fit_label(s, a, se))  # fmt: skip
            ax.plot(
                g.loc[~ok, "kappa"],
                g.loc[~ok, "budget"],
                "v",
                mfc="none",
                color=cfg["color"],
            )
            # reference slope, anchored at the first point of the fit region
            gm = g[ok & (g["kappa"] >= kappa_min)]
            if len(gm):
                k0, t0 = gm.iloc[0]["kappa"], gm.iloc[0]["T"]
                kk = g.loc[g["kappa"] >= kappa_min, "kappa"].to_numpy()
                ax.plot(kk, t0 * (kk / k0) ** cfg["kappa_ref"], "--", color=cfg["color"],
                        alpha=0.45, lw=1)  # fmt: skip
            ll = ls[(ls["sampler"] == s) & (ls["dimension"] == d)]
            ax2.plot(
                ll["kappa_mid"],
                ll["slope"],
                marker=cfg["marker"],
                ls="-",
                color=cfg["color"],
            )
        if kappa_min > k_lo:  # grey = not used for the fit
            for a_ in (ax, ax2):
                a_.axvspan(k_lo / 1.3, kappa_min / 1.15, color="0.93", zorder=0)
        for p in sorted({SAMPLERS[s]["kappa_ref"] for s in present(best)}):
            ax2.axhline(p, c="gray", ls=":", lw=1)
        ax.set(xscale="log", yscale="log", title=rf"$d={d}$")
        ax2.set(xscale="log", xlabel=r"condition number $\kappa$")
        ax.legend(fontsize=8, title=rf"fit for $\kappa\geq{kappa_min:g}$; dashed: reference slope",
                  title_fontsize=7)  # fmt: skip
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(r"local slope $\mathrm{d}\log T\,/\,\mathrm{d}\log\kappa$")
    save(
        fig,
        path,
        rf"Cost until {eps_label} $\leq\varepsilon$  (best $h,\gamma$ per point;  "
        r"$\triangledown$ = not converged within budget)",
    )


def fig_speedup(sp, kappa_min, path):
    sp = sp[sp["speedup"].notna()]
    samplers = [s for s in SAMPLERS if s in set(sp["sampler"])]
    if not samplers:
        return
    fig, axes = plt.subplots(
        1, len(samplers), figsize=(4.8 * len(samplers), 3.8), sharey=True, squeeze=False
    )
    for ax, s in zip(axes[0], samplers):
        ref = SAMPLERS[BASELINE]["kappa_ref"] - SAMPLERS[s]["kappa_ref"]
        for i, (d, g) in enumerate(sp[sp["sampler"] == s].groupby("dimension")):
            g = g.sort_values("kappa")
            c = f"C{i}"
            ex, lb = g[~g["is_lower_bound"]], g[g["is_lower_bound"]]
            ax.plot(ex["kappa"], ex["speedup"], "o-", color=c, label=rf"$d={d}$")
            ax.plot(
                lb["kappa"], lb["speedup"], "^", mfc="none", color=c
            )  # baseline censored
            a = ex[ex["kappa"] >= kappa_min]
            if len(a):
                k0, s0 = a.iloc[0]["kappa"], a.iloc[0]["speedup"]
                kk = g.loc[g["kappa"] >= kappa_min, "kappa"].to_numpy()
                ax.plot(kk, s0 * (kk / k0) ** ref, "--", color=c, alpha=0.45, lw=1)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel=r"condition number $\kappa$",
            title=rf"{s} vs {BASELINE} (dashed: $\kappa^{{{exp_str(ref)}}}$)",
        )
        ax.set_ylabel(rf"speed-up $T_{{\mathrm{{{BASELINE}}}}}/T_{{\mathrm{{{s}}}}}$")
        ax.legend(fontsize=8)
    save(fig, path)


def fig_T_vs_dim(best, kappa_min, path) -> bool:
    dims = sorted(best["dimension"].unique())
    if len(dims) < 2:
        return False
    d = np.array(dims, dtype=float)
    samplers = present(best)
    fig, axes = plt.subplots(
        1, len(samplers), figsize=(4.5 * len(samplers), 3.6), sharey=True, squeeze=False
    )
    for ax, s in zip(axes[0], samplers):
        g_s = best[
            (best["sampler"] == s) & ~best["censored"] & (best["kappa"] >= kappa_min)
        ]
        for k, g in g_s.groupby("kappa"):
            g = g.sort_values("dimension")
            if g["dimension"].nunique() < 2:
                continue
            ax.plot(
                g["dimension"], g["T"] / g["T"].iloc[0], "o-", label=rf"$\kappa={k:g}$"
            )
        # reference: the sampler's own claim (dashed) and linear growth for comparison (dotted)
        refs = {1.0: ":"}
        refs[SAMPLERS[s]["dim_ref"]] = (
            "--"  # own claim overrides the dotted line if both coincide
        )
        for p, style in sorted(refs.items()):
            ax.plot(
                d,
                (d / d[0]) ** p,
                c="k",
                ls=style,
                lw=1,
                label=rf"$d^{{{exp_str(p)}}}$",
            )
        ax.set(xscale="log", yscale="log", xlabel=r"dimension $d$", title=s)
    axes[0, 0].set_ylabel(r"$T(d)\,/\,T(d_{\min})$")
    axes[0, 0].legend(fontsize=7)
    save(fig, path)
    return True


# ---------------------------------------------------------------------------------
# 4. diagnostics (explain *why* the numbers look the way they do)
# ---------------------------------------------------------------------------------
def fig_diag_hyperparameters(best, path):
    dims = sorted(best["dimension"].unique())
    gamma_samplers = [s for s in present(best) if SAMPLERS[s]["uses_gamma"]]
    fig, axes = plt.subplots(1, 1 + bool(gamma_samplers), figsize=(5 * (1 + bool(gamma_samplers)), 3.8),
                             squeeze=False)  # fmt: skip
    a1 = axes[0, 0]
    for s in present(best):
        for i, d in enumerate(dims):
            g = best[
                (best["sampler"] == s) & (best["dimension"] == d) & ~best["censored"]
            ]
            g = g.sort_values("kappa")
            if g.empty:
                continue
            a1.plot(g["kappa"], g["step_size"], marker=SAMPLERS[s]["marker"], ls="-",
                    color=SAMPLERS[s]["color"], alpha=0.4 + 0.6 * (i + 1) / len(dims), ms=4,
                    label=rf"{s}, $d={d}$")  # fmt: skip
            e = g[g["h_at_edge"]]
            a1.plot(e["kappa"], e["step_size"], "kx", ms=8)  # optimum at edge of h grid
    a1.plot([], [], "kx", label="optimum at grid edge (extend grid)")
    a1.axhline(2.0, c="r", ls=":", lw=1)
    a1.set(xscale="log", yscale="log", xlabel=r"condition number $\kappa$",
           ylabel=r"best step size $h^\ast$",
           title=r"optimal step size (red: ULA stability limit $2/L$)")  # fmt: skip
    a1.legend(fontsize=6)

    if gamma_samplers:  # which friction rule wins? (only samplers that have a gamma)
        a2 = axes[0, 1]
        keys = list(GAMMA_RULES)
        for s in gamma_samplers:
            for i, d in enumerate(dims):
                g = best[
                    (best["sampler"] == s)
                    & (best["dimension"] == d)
                    & ~best["censored"]
                ]
                g = g.sort_values("kappa")
                y = [keys.index(x) if x in keys else np.nan for x in g["gamma_label"]]
                a2.plot(g["kappa"], np.array(y) + 0.12 * (i - (len(dims) - 1) / 2), "o",
                        color=SAMPLERS[s]["color"], alpha=0.4 + 0.6 * (i + 1) / len(dims),
                        label=rf"{s}, $d={d}$")  # fmt: skip
        a2.set_yticks(range(len(keys)))
        a2.set_yticklabels(list(GAMMA_RULES.values()))
        a2.set(xscale="log", xlabel=r"condition number $\kappa$",
               title=r"which $\gamma$ wins? ($c/\sqrt{\kappa}$ = momentum regime)")  # fmt: skip
        a2.legend(fontsize=7)
    save(fig, path)


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
            for s in present(df):
                cell = (
                    (df["sampler"] == s) & (df["dimension"] == d) & (df["kappa"] == k)
                )
                b = best[
                    (best["sampler"] == s)
                    & (best["dimension"] == d)
                    & (best["kappa"] == k)
                ]
                r = df[cell]
                if b.empty or r.empty:
                    continue
                # only the runs with the winning gamma (for ULA / RMULMC: all runs)
                r = r[r["gamma_label"] == b["gamma_label"].iloc[0]].sort_values(
                    "step_size"
                )
                ax.plot(r["step_size"], r["final_err"], marker=SAMPLERS[s]["marker"], ls="-", ms=3,
                        color=SAMPLERS[s]["color"], label=s)  # fmt: skip
                if not b["censored"].iloc[0]:
                    ax.plot(b["step_size"], b["final_err"], "k*", ms=9)
                ax.axhline(r["eps_used"].iloc[0], c="gray", ls=":")
            ax.set(xscale="log", yscale="log", title=rf"$d={d}$, $\kappa={k:g}$")
    axes[-1, 0].set_xlabel(r"step size $h$")
    axes[0, 0].set_ylabel("error at end of budget")
    axes[0, 0].legend(fontsize=7)
    save(
        fig,
        path,
        r"small $h$: not converged within budget   |   large $h$: discretisation bias   "
        r"($\star$ = chosen $h^\ast$, dotted = $\varepsilon$)",
    )


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
    ap.add_argument(
        "--first-checkpoint",
        type=int,
        default=5,
        help="make_eval_steps(first=...); convergence at this step is flagged as left-censored",
    )
    ap.add_argument("--usetex", action="store_true",
                    help="render text with a real LaTeX installation (default: matplotlib mathtext)")  # fmt: skip
    a = ap.parse_args()

    set_style(a.usetex)
    out = Path(a.out)
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(a.csv)

    ignored = sorted(set(raw["sampler"]) - set(SAMPLERS))
    if ignored:
        print(f"note: samplers in the csv but not in SAMPLERS are ignored: {ignored}")

    df = prepare(raw, a.criterion, a.cost)
    print(f"samplers analysed: {present(df)}")
    best = best_configs(df, a.first_checkpoint)
    best.to_csv(out / "best_configs.csv", index=False)

    chk = checks(df, best)
    chk.to_csv(out / "checks.csv")
    print(f"\n=== sanity ({a.criterion}, cost={a.cost}) ===\n{chk.to_string()}")

    kexp = kappa_exponents(best, a.kappa_min)
    ls = local_slopes(best)
    sp = speedup_table(best)
    sexp = speedup_exponents(sp, a.kappa_min)
    dexp = dim_exponents(best, a.kappa_min) if a.cost == "grads" else pd.DataFrame()

    parts = [kexp.assign(kind="kappa_exponent"), sexp.assign(kind="speedup_exponent")]
    if len(dexp):
        parts.append(dexp.rename(columns={"beta": "alpha"}).assign(kind="dim_exponent"))
    pd.concat([p for p in parts if len(p)], ignore_index=True).to_csv(
        out / "exponents.csv", index=False
    )

    print(
        f"\n=== T ~ kappa^alpha, fitted for kappa >= {a.kappa_min:g} (se = scatter around the line) ===\n{fmt(kexp)}"
    )
    if len(sexp):
        print(f"\n=== speed-up over {BASELINE} ~ kappa^alpha ===\n{fmt(sexp)}")
    if len(dexp):
        print(
            "\n=== T ~ d^beta at fixed kappa, median over kappa ===\n"
            f"{fmt(dexp.groupby('sampler')[['beta', 'proposal']].median().reset_index())}"
        )

    ylabel = (
        "gradient evaluations $T$"
        if a.cost == "grads"
        else "wall-clock seconds $T$ (ensemble)"
    )
    eps_label = CRITERIA[a.criterion]["label"]
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
        b = best_configs(prepare(raw, crit, cost), a.first_checkpoint)
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
