"""
Post-processing of dimension_conditioning_scaling_results.csv

Pipeline
  0. load + sanity checks (floor_ok, divergence, censoring, grid boundaries)
  1. select best configuration per (sampler, d, kappa)  [min cost over step size, gamma]
  2. exponent fits   log T = a + alpha*log(kappa) + beta*log(d)   (+ separability check)
  3. derived quantities: speedup ULA/ULMC, local slopes, h*(kappa, d), best gamma, wall-clock overhead
  4. figures + tables (one folder per analysis configuration, plus a robustness summary)

Usage:  python scaling_analysis.py --csv dimension_conditioning_scaling_results.csv --out analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (kappa exponent, d exponent) from theory. RM-methods: exploratory only.
THEORY = {"ULA": (1.0, 1.0), "ULMC": (0.5, 0.5)}
COLORS = {
    "ULA": "tab:blue",
    "RMULA": "tab:cyan",
    "ULMC": "tab:red",
    "RMULMC": "tab:orange",
}
COST = {
    "grad_evals": "grad_evals_to_conv",
    "steps": "steps_to_conv",
    "wall": "wall_to_conv",
}
CONFIG = dict(
    num_steps=20_000, first=20, n_eval=100
)  # must match the experiment script


# ---------------------------------------------------------------------------------
# 0. load + sanity
# ---------------------------------------------------------------------------------
def gamma_label(gamma: float, kappa: float) -> str:
    if pd.isna(gamma):
        return "-"
    for label, value in (
        ("sqrt2", np.sqrt(2)),
        ("2", 2.0),
        ("2/sqrt(k)", 2 / np.sqrt(kappa)),
    ):
        if np.isclose(gamma, value):
            return label
    return f"{gamma:.3g}"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["gamma_label"] = [gamma_label(g, k) for g, k in zip(df["gamma"], df["kappa"])]
    for sfx in ("", "_white"):  # wall-clock of ONE step-size run until convergence
        df[f"wall_to_conv{sfx}"] = np.where(
            df[f"converged{sfx}"],
            df[f"steps_to_conv{sfx}"] * df["sec_per_step"],
            np.inf,
        )
    return df


def sanity(df: pd.DataFrame, best: pd.DataFrame, tag: str) -> None:
    print(f"\n=== sanity [{tag}] ===")
    bad = df.loc[
        ~df["floor_ok"] | ~df["floor_ok_white"], ["dimension", "kappa"]
    ].drop_duplicates()
    print(f"cells where MC floor does not resolve eps: {len(bad)}")
    print(
        "censored (never converged) per sampler:\n",
        best.groupby("sampler")["censored"].mean().round(2).to_string(),
    )
    print(
        "optimum at step-size grid boundary:\n",
        best.groupby("sampler")["at_boundary"].mean().round(2).to_string(),
    )
    print(
        "left-censored (converged at first checkpoint):",
        int(best["left_censored"].sum()),
    )
    print("diverged fraction of runs:", round(df["diverged"].mean(), 3))


# ---------------------------------------------------------------------------------
# 1. best configuration
# ---------------------------------------------------------------------------------
def best_configs(df, white: bool, cost: str, extra: tuple = ()) -> pd.DataFrame:
    sfx = "_white" if white else ""
    conv, c = f"converged{sfx}", COST[cost] + sfx
    fb = "final_w2_white" if white else "final_w2_norm"
    key = ["sampler", "dimension", "kappa", *extra]
    g = df.assign(
        T=np.where(df[conv] & ~df["diverged"], df[c], np.inf),
        fb=np.where(
            df["diverged"], np.inf, df[fb]
        ),  # fallback: best final W2 if never converged
    )
    best = (
        g.sort_values(["T", "fb"])
        .drop_duplicates(key)
        .sort_values(key)
        .reset_index(drop=True)
    )
    best["censored"] = ~np.isfinite(best["T"])
    lo, hi = df["step_size"].min(), df["step_size"].max()
    best["at_boundary"] = np.isclose(best["step_size"], lo) | np.isclose(
        best["step_size"], hi
    )
    best["left_censored"] = best[f"steps_to_conv{sfx}"] <= CONFIG["first"]
    unit = {
        "grad_evals": best["grads_per_step"],
        "steps": 1.0,
        "wall": best["sec_per_step"],
    }[cost]
    best["budget"] = CONFIG["num_steps"] * unit  # right-censoring level
    return best


# ---------------------------------------------------------------------------------
# 2. exponent fits (log-log OLS + discretisation-sensitivity interval + leave-one-out)
# ---------------------------------------------------------------------------------
def ols(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    sst = ((y - y.mean()) ** 2).sum()
    return b, (1 - ((y - X @ b) ** 2).sum() / sst) if sst > 0 else np.nan


def fit(X, y, jitter, rng, n_boot=2000):
    """
    Add multiple runs per cell for statistically accuarate estimators.

    Currently:
    Only ONE run per cell exists, so there is no sampling variance. The interval below propagates the
    quantisation of T (checkpoint grid x step-size grid): y_i + U(-.5,.5)*jitter. It is a sensitivity
    band, NOT a statistical CI. Real CIs need replicate seeds.
    """
    b, r2 = ols(X, y)
    bs = np.array(
        [ols(X, y + rng.uniform(-0.5, 0.5, len(y)) * jitter)[0] for _ in range(n_boot)]
    )
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    if len(y) > X.shape[1]:
        loo = np.array(
            [ols(np.delete(X, i, 0), np.delete(y, i))[0] for i in range(len(y))]
        )
    else:
        loo = np.full((1, X.shape[1]), np.nan)
    return b, r2, lo, hi, loo.min(0), loo.max(0)


def exponent_table(best, jitter, rng, value="T") -> pd.DataFrame:
    rows = []

    def add(sampler, model, fixed, names, X, y):
        if len(y) < X.shape[1] + 1:
            return
        b, r2, lo, hi, l0, l1 = fit(X, y, jitter, rng)
        for j, n in enumerate(names):
            if n == "const" or (model == "joint+interaction" and n != "kappa:dim"):
                continue
            th = THEORY.get(sampler)
            t = (
                th[0 if n == "kappa" else 1]
                if (th and n in ("kappa", "dim"))
                else np.nan
            )
            if value == "speedup" and n in ("kappa", "dim"):
                t = 0.5
            rows.append(
                dict(
                    sampler=sampler,
                    model=model,
                    fixed=fixed,
                    term=n,
                    est=b[j],
                    lo=lo[j],
                    hi=hi[j],
                    loo_min=l0[j],
                    loo_max=l1[j],
                    r2=r2,
                    n=len(y),
                    theory=t,
                )
            )

    for sampler, g in best[~best["censored"]].groupby("sampler"):
        g = g.assign(
            lk=np.log(g["kappa"]), ld=np.log(g["dimension"]), ly=np.log(g[value])
        )
        for dim, gd in g.groupby("dimension"):
            add(
                sampler,
                "kappa|d",
                f"d={dim}",
                ["const", "kappa"],
                np.c_[np.ones(len(gd)), gd["lk"]],
                gd["ly"].to_numpy(),
            )
        for kappa, gk in g.groupby("kappa"):
            add(
                sampler,
                "dim|kappa",
                f"kappa={kappa:g}",
                ["const", "dim"],
                np.c_[np.ones(len(gk)), gk["ld"]],
                gk["ly"].to_numpy(),
            )
        one = np.ones(len(g))
        add(
            sampler,
            "joint",
            "all",
            ["const", "kappa", "dim"],
            np.c_[one, g["lk"], g["ld"]],
            g["ly"].to_numpy(),
        )
        add(
            sampler,
            "joint+interaction",
            "all",
            ["const", "kappa", "dim", "kappa:dim"],
            np.c_[one, g["lk"], g["ld"], g["lk"] * g["ld"]],
            g["ly"].to_numpy(),
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------
# 3. derived quantities
# ---------------------------------------------------------------------------------
def speedup_best(best, num="ULA", den="ULMC") -> pd.DataFrame:
    p = best.pivot_table(
        index=["dimension", "kappa"], columns="sampler", values="T", aggfunc="first"
    )
    if num not in p or den not in p:
        return pd.DataFrame(
            columns=["dimension", "kappa", "speedup", "sampler", "censored"]
        )
    out = (
        (p[num] / p[den])
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .rename("speedup")
        .reset_index()
    )
    return out.assign(
        sampler=f"{num}/{den}", censored=False
    )  # inf cells (ULA never converged) dropped


def local_slopes(best) -> pd.DataFrame:
    out = []
    for (sampler, dim), g in best[~best["censored"]].groupby(["sampler", "dimension"]):
        g = g.sort_values("kappa")
        if len(g) < 2:
            continue
        slope = np.diff(np.log(g["T"])) / np.diff(np.log(g["kappa"]))
        mid = np.sqrt(g["kappa"].to_numpy()[1:] * g["kappa"].to_numpy()[:-1])
        out += [
            dict(sampler=sampler, dimension=dim, kappa_mid=m, slope=v)
            for m, v in zip(mid, slope)
        ]
    return pd.DataFrame(out)


def overhead_table(df) -> pd.DataFrame:
    """Measured time per step relative to ULA (same d, kappa => same number of chains)."""
    t = (
        df.groupby(["sampler", "dimension", "kappa"])["sec_per_step"]
        .median()
        .unstack("sampler")
    )
    rel = t.div(t["ULA"], axis=0) if "ULA" in t else t
    return rel.reset_index()


# ---------------------------------------------------------------------------------
# 4. plots
# ---------------------------------------------------------------------------------
def plot_scaling(best, x, by, path, ylabel):
    vals = sorted(best[by].unique())
    fig, axes = plt.subplots(
        1, len(vals), figsize=(4.0 * len(vals), 3.6), sharey=True, squeeze=False
    )
    for ax, v in zip(axes[0], vals):
        for s, g in best[best[by] == v].groupby("sampler"):
            g = g.sort_values(x)
            ok = ~g["censored"]
            ax.plot(g.loc[ok, x], g.loc[ok, "T"], "o-", color=COLORS.get(s), label=s)
            ax.plot(
                g.loc[~ok, x],
                g.loc[~ok, "budget"],
                "v",
                mfc="none",
                color=COLORS.get(s),
            )  # right-censored
            if s in THEORY and ok.any():
                p = THEORY[s][0 if x == "kappa" else 1]
                x0, y0 = g.loc[ok, x].iloc[0], g.loc[ok, "T"].iloc[0]
                ax.plot(
                    g[x],
                    y0 * (g[x] / x0) ** p,
                    "--",
                    color=COLORS.get(s),
                    alpha=0.4,
                    lw=1,
                )
        ax.set(xscale="log", yscale="log", xlabel=x, title=f"{by}={v:g}")
    axes[0, 0].set_ylabel(ylabel)
    axes[0, 0].legend(fontsize=7, title="dashed: proposal slope", title_fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forest(exp, path):
    j = exp[exp["model"] == "joint"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, term in zip(axes, ["kappa", "dim"]):
        t = j[j["term"] == term].reset_index(drop=True)
        xerr = np.clip([t["est"] - t["lo"], t["hi"] - t["est"]], 0, None)
        ax.errorbar(t["est"], range(len(t)), xerr=xerr, fmt="o", capsize=3, label="fit")
        ax.plot(t["theory"], range(len(t)), "kx", label="proposal")
        ax.set_yticks(range(len(t)))
        ax.set_yticklabels(t["sampler"])
        ax.set_title(f"exponent of {term}")
        ax.axvline(0, c="gray", lw=0.5)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_collapse(best, exp, path):
    j = exp[exp["model"] == "joint"]
    ss = list(j["sampler"].unique())
    fig, axes = plt.subplots(1, len(ss), figsize=(3.8 * len(ss), 3.4), squeeze=False)
    for ax, s in zip(axes[0], ss):
        a = j[(j.sampler == s) & (j.term == "kappa")]["est"].iloc[0]
        b = j[(j.sampler == s) & (j.term == "dim")]["est"].iloc[0]
        g = best[(best.sampler == s) & ~best["censored"]]
        for d, gd in g.groupby("dimension"):
            ax.plot(
                gd["kappa"], gd["T"] / (gd["kappa"] ** a * d**b), "o-", label=f"d={d}"
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="kappa",
            title=f"{s}: T/(k^{a:.2f} d^{b:.2f})",
        )
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)  # flat lines = power law adequate


def plot_speedup(sp, path):
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    for d, g in sp.groupby("dimension"):
        ax.plot(g["kappa"], g["speedup"], "o-", label=f"d={d}")
    k = np.sort(sp["kappa"].unique())
    ax.plot(
        k,
        sp["speedup"].min() * (k / k[0]) ** 0.5,
        "k--",
        lw=1,
        label=r"$\sqrt{\kappa}$",
    )
    ax.set(xscale="log", yscale="log", xlabel="kappa", ylabel="T_ULA / T_ULMC")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_local_slopes(ls, path):
    ss = list(ls["sampler"].unique())
    fig, axes = plt.subplots(
        1, len(ss), figsize=(3.4 * len(ss), 3.2), sharey=True, squeeze=False
    )
    for ax, s in zip(axes[0], ss):
        for d, g in ls[ls.sampler == s].groupby("dimension"):
            ax.plot(g["kappa_mid"], g["slope"], "o-", label=f"d={d}")
        for th, ls_ in ((1.0, ":"), (0.5, "--")):
            ax.axhline(th, c="gray", ls=ls_, lw=1)
        ax.set(xscale="log", xlabel="kappa (midpoint)", title=s)
    axes[0, 0].set_ylabel("local exponent dlogT/dlogkappa")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stepsize(best, path):
    ss = list(best["sampler"].unique())
    fig, axes = plt.subplots(
        1, len(ss), figsize=(3.4 * len(ss), 3.2), sharey=True, squeeze=False
    )
    for ax, s in zip(axes[0], ss):
        for d, g in best[(best.sampler == s) & ~best["censored"]].groupby("dimension"):
            ax.plot(g["kappa"], g["step_size"], "-", alpha=0.5)
            ax.plot(g["kappa"], g["step_size"], "o", mfc="none", label=f"d={d}")
            b = g[g["at_boundary"]]
            ax.plot(
                b["kappa"], b["step_size"], "kx"
            )  # optimum at grid edge -> extend grid
        ax.axhline(2.0, c="r", ls=":", lw=1)  # 2/L stability limit
        ax.set(xscale="log", yscale="log", xlabel="kappa", title=s)
    axes[0, 0].set_ylabel("best step size h*")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gamma(best_g, path):
    g_ = best_g[best_g.sampler == "ULMC"]
    if g_.empty:
        return
    ds = sorted(g_["dimension"].unique())
    fig, axes = plt.subplots(
        1, len(ds), figsize=(3.6 * len(ds), 3.4), sharey=True, squeeze=False
    )
    for ax, d in zip(axes[0], ds):
        for lab, g in g_[g_.dimension == d].groupby("gamma_label"):
            g = g.sort_values("kappa")
            ax.plot(
                g["kappa"], g["T"].replace(np.inf, np.nan), "o-", label=f"gamma={lab}"
            )
        ax.set(xscale="log", yscale="log", xlabel="kappa", title=f"ULMC, d={d}")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ushape(df, best, white, path):
    fb = "final_w2_white" if white else "final_w2_norm"
    eps = "eps_white" if white else "eps"
    ds = [df["dimension"].min(), df["dimension"].max()]
    ks = [df["kappa"].min(), df["kappa"].max()]
    fig, axes = plt.subplots(
        len(ks),
        len(ds),
        figsize=(4 * len(ds), 3.2 * len(ks)),
        squeeze=False,
        sharex=True,
    )
    for i, k in enumerate(ks):
        for j, d in enumerate(ds):
            ax = axes[i, j]
            for s in df["sampler"].unique():
                b = best[
                    (best.sampler == s) & (best.dimension == d) & (best.kappa == k)
                ]
                r = df[(df.sampler == s) & (df.dimension == d) & (df.kappa == k)]
                if b.empty or r.empty:
                    continue
                if pd.notna(b["gamma"].iloc[0]):
                    r = r[np.isclose(r["gamma"], b["gamma"].iloc[0])]
                r = r.sort_values("step_size")
                ax.plot(
                    r["step_size"],
                    r[fb].replace(np.inf, np.nan),
                    "o-",
                    ms=3,
                    color=COLORS.get(s),
                    label=s,
                )
                ax.axhline(r[eps].iloc[0], c="gray", ls=":")
            ax.set(xscale="log", yscale="log", title=f"d={d}, kappa={k:g}")
    axes[-1, 0].set_xlabel("step size h")
    axes[0, 0].set_ylabel("final W2")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)  # small h: not converged in N steps; large h: bias


def plot_overhead(ov, path):
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for s in [c for c in ov.columns if c not in ("dimension", "kappa", "ULA")]:
        g = ov.groupby("dimension")[s].median()
        ax.plot(g.index, g.values, "o-", color=COLORS.get(s), label=s)
    ax.axhline(1, c="gray", lw=0.5)
    ax.set(xscale="log", xlabel="d", ylabel="time/step relative to ULA")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------
def run(df, out: Path, white: bool, cost: str, rng) -> pd.DataFrame:
    tag = f"{'white' if white else 'euclid'}_{cost}"
    out = out / tag
    out.mkdir(parents=True, exist_ok=True)
    best = best_configs(df, white, cost)
    sanity(df, best, tag)
    best.to_csv(out / "best_configs.csv", index=False)

    hs = np.sort(df["step_size"].unique())
    jitter = np.log(np.exp(np.diff(np.log(hs)).mean())) + np.log(
        (CONFIG["num_steps"] / CONFIG["first"]) ** (1 / (CONFIG["n_eval"] - 1))
    )

    exp = exponent_table(best, jitter, rng)
    sp = speedup_best(best)
    if not sp.empty:
        sp_exp = exponent_table(
            sp.rename(columns={"speedup": "T"}).assign(censored=False),
            jitter,
            rng,
            value="T",
        )
        sp_exp["theory"] = np.where(sp_exp["term"].isin(["kappa", "dim"]), 0.5, np.nan)
        exp = pd.concat([exp, sp_exp.assign(sampler="ULA/ULMC")], ignore_index=True)
        plot_speedup(sp, out / "fig_speedup.png")
    exp.to_csv(out / "exponents.csv", index=False)
    with open(out / "exponents_joint.tex", "w") as f:
        f.write(exp[exp.model == "joint"].round(2).to_latex(index=False))

    ylab = {
        "grad_evals": "gradient evaluations to eps",
        "steps": "iterations to eps",
        "wall": "seconds to eps",
    }[cost]
    plot_scaling(best, "kappa", "dimension", out / "fig_T_vs_kappa.png", ylab)
    plot_scaling(best, "dimension", "kappa", out / "fig_T_vs_dim.png", ylab)
    if not exp.empty:
        plot_forest(exp[exp.sampler != "ULA/ULMC"], out / "fig_exponents.png")
        plot_collapse(best, exp, out / "fig_collapse.png")
    ls = local_slopes(best)
    ls.to_csv(out / "local_slopes.csv", index=False)
    if not ls.empty:
        plot_local_slopes(ls, out / "fig_local_slopes.png")
    plot_stepsize(best, out / "fig_stepsize.png")
    plot_gamma(
        best_configs(df, white, cost, extra=("gamma_label",)),
        out / "fig_ulmc_gamma.png",
    )
    plot_ushape(df, best, white, out / "fig_ushape.png")
    ov = overhead_table(df)
    ov.to_csv(out / "overhead.csv", index=False)
    plot_overhead(ov, out / "fig_overhead.png")

    # compact table: best T per (sampler, kappa) x d
    best.assign(
        Tstr=best["T"].map(lambda v: "-" if not np.isfinite(v) else f"{v:.0f}")
    ).pivot_table(
        index="kappa", columns=["sampler", "dimension"], values="Tstr", aggfunc="first"
    ).to_csv(out / "table_T.csv")
    return exp.assign(config=tag)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dimension_conditioning_scaling_results.csv")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--num-steps", type=int, default=20_000)
    a = ap.parse_args()
    CONFIG["num_steps"] = a.num_steps

    df, out, rng = load(a.csv), Path(a.out), np.random.default_rng(0)
    # primary: whitened criterion + gradient evals; robustness: Euclidean criterion, wall-clock
    exps = [
        run(df, out, w, c, rng)
        for w, c in [(True, "grad_evals"), (False, "grad_evals"), (True, "wall")]
    ]
    allexp = pd.concat(exps, ignore_index=True)
    allexp.to_csv(out / "robustness_exponents.csv", index=False)
    piv = (
        allexp[(allexp.model == "joint") & allexp.term.isin(["kappa", "dim"])]
        .pivot_table(index=["sampler", "term"], columns="config", values="est")
        .round(2)
    )
    print("\n=== joint exponents across analysis configurations ===\n", piv)
