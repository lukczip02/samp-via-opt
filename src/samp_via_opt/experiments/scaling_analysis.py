"""Post-processing of the dimension / condition-number scaling experiments.

Compares ULA, ULMC and RMULMC (plus RMULMC with the fixed friction gamma = 2 of Shen & Lee)
on Gaussian targets. For every (sampler, dimension d, condition number kappa) the cost T is the
number of gradient evaluations until the Wasserstein-2 error stays below a tolerance, minimised
over step size h (and friction gamma, if the sampler has one).

Convergence criterion:
    The W2 error is normalised, in one of two ways (``--tolerance`` must match the run that
    produced the csv, i.e. ``use_rel_tolerance`` of ``dimension_conditioning.py``):

    * ``relative``: W2 <= eps * sqrt(d / m). The absolute tolerance grows like sqrt(d), which
      cancels the d-dependence of the theoretical bounds (reference d-exponent 0). Use for
      the kappa sweep.
    * ``absolute``: W2 <= eps * sqrt(1 / m). Reference d-exponents are p = 1, 1/2, 1/3 for
      ULA, ULMC, RMULMC. Use for the dimension sweep (needs ~d^2 chains, so small d only).

    ``--criterion white`` uses the whitened W2 instead (W2_white <= eps_white * sqrt(d) for the
    relative and W2_white <= eps_white for the absolute tolerance).

Censoring:
    If no configuration converged within the budget of a cell, the cell is "censored": it is
    drawn as a hollow triangle at the budget and excluded from all fits.

Outputs (written to ``--out``):
    fig1_cost_vs_kappa.png        Cost vs kappa and local slopes, one column per dimension.
    fig2_speedup.png              Speed-up over ULA vs kappa, dashed = reference slope.
    fig3_cost_vs_dimension.png    Cost(d) / cost(d_min) vs d (needs >= 2 dimensions).
    fig4_step_size_and_gamma.png  Step size and friction of the optimal configuration.
    best_configs.csv              Optimal (h, gamma) and quality flags for every cell.
    exponents.csv                 All fitted exponents (kappa, speed-up, dimension).
    table_cost.csv                Cost per kappa x (sampler, dimension).
    checks.csv                    Sanity checks per sampler.

Example:
    python scaling_analysis.py conditioning_scaling_results.csv --out analysis_kappa \\
        --tolerance relative --fig1-dims 2 16 64
    python scaling_analysis.py dimension_scaling_results.csv --out analysis_dim \\
        --tolerance absolute

Adding a sampler (e.g. RMULA) only requires one new entry in ``SAMPLERS``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplerConfig:
    """Plot style, reference exponents and grid conventions of one analysed sampler.

    Attributes:
        label: Name used in legends and titles (may contain mathtext).
        color: Matplotlib color.
        marker: Matplotlib marker.
        kappa_exponent: Reference exponent a in cost ~ kappa^a (dashed lines).
        dim_exponent_abs: Exponent p in the theory bound cost ~ d^p / eps_abs^q for a fixed
            absolute tolerance eps_abs.
        eps_exponent: Exponent q in the same bound. With the relative tolerance
            eps_abs = eps * sqrt(d / m) the reference d-exponent becomes p - q / 2 (= 0 for
            all samplers below).
        has_free_gamma: Whether the friction gamma is a tuned hyperparameter.
        step_in_gamma_units: Whether the step-size grid of the experiment is
            ``gamma * rel_step_sizes`` (ULMC) instead of ``rel_step_sizes``. Only used to
            detect optima at the edge of the grid.
        subscript: LaTeX subscript for axis labels. Defaults to the sampler name.
        base_sampler: If set, this entry is a variant that reuses the runs of another sampler.
        only_gamma: Key of ``GAMMA_RULES`` the variant is restricted to.
    """

    label: str
    color: str
    marker: str
    kappa_exponent: float
    dim_exponent_abs: float
    eps_exponent: float
    has_free_gamma: bool
    step_in_gamma_units: bool = False
    subscript: str | None = None
    base_sampler: str | None = None
    only_gamma: str | None = None


# kappa_exponent of the RMULMC variants are hypotheses: "RMULMC" accelerates like ULMC;
# "RMULMC_g2" has gamma = 2 >> 2 / sqrt(kappa) (overdamped regime), so ULA-like kappa^1 is expected.
SAMPLERS: dict[str, SamplerConfig] = {
    "ULA": SamplerConfig(
        label="ULA", color="tab:blue", marker="o",
        kappa_exponent=1.0, dim_exponent_abs=1.0, eps_exponent=2.0,
        has_free_gamma=False,
    ),
    "ULMC": SamplerConfig(
        label="ULMC", color="tab:red", marker="s",
        kappa_exponent=0.5, dim_exponent_abs=0.5, eps_exponent=1.0,
        has_free_gamma=True, step_in_gamma_units=True,
    ),
    "RMULMC": SamplerConfig(
        label="RMULMC", color="tab:green", marker="^",
        kappa_exponent=0.5, dim_exponent_abs=1 / 3, eps_exponent=2 / 3,
        has_free_gamma=True,
    ),
    "RMULMC_g2": SamplerConfig(
        label=r"RMULMC ($\gamma=2$)", color="tab:orange", marker="D",
        kappa_exponent=1.0, dim_exponent_abs=1 / 3, eps_exponent=2 / 3,
        has_free_gamma=False, subscript=r"\mathrm{RMULMC},\gamma{=}2",
        base_sampler="RMULMC", only_gamma="2",
    ),
}  # fmt: skip

BASELINE = "ULA"  # speed-ups are measured relative to this sampler
NEAR_OPTIMAL_FACTOR = 1.2  # "near optimal" = cost within 20 % of the best cost
CELL_KEY = ["sampler", "dimension", "kappa"]
ULA_STABILITY_LIMIT = 2.0  # 2 / L with L = 1 by construction of the targets

# Friction rules of the experiment: key (as derived by ``gamma_rule``) -> axis label.
GAMMA_RULES = {
    "1/sqrt(k)": r"$1/\sqrt{\kappa}$",
    "2/sqrt(k)": r"$2/\sqrt{\kappa}$",
    "4/sqrt(k)": r"$4/\sqrt{\kappa}$",
    "sqrt(2)": r"$\sqrt{2}$",
    "2": r"$2$",
}


@dataclass(frozen=True)
class CriterionColumns:
    """Names of the csv columns that belong to one convergence criterion."""

    suffix: str
    final_error: str
    epsilon: str
    floor_ok: str
    borderline: str
    w2_symbol: str
    epsilon_symbol: str
    relative_scale: str
    absolute_scale: str


CRITERIA = {
    "euclid": CriterionColumns(
        suffix="", final_error="final_w2_norm", epsilon="eps", floor_ok="floor_ok",
        borderline="borderline", w2_symbol=r"W_2", epsilon_symbol=r"\varepsilon",
        relative_scale=r"\sqrt{d/m}", absolute_scale=r"\sqrt{1/m}",
    ),
    "white": CriterionColumns(
        suffix="_white", final_error="final_w2_white", epsilon="eps_white",
        floor_ok="floor_ok_white", borderline="borderline_white",
        w2_symbol=r"W_2^{\mathrm{white}}", epsilon_symbol=r"\varepsilon_{\mathrm{white}}",
        relative_scale=r"\sqrt{d}", absolute_scale="1",
    ),
}  # fmt: skip

FIT_COLUMNS = ["kind", "sampler", "dimension", "kappa", "exponent", "std_error",
               "num_points", "reference"]  # fmt: skip


@dataclass(frozen=True)
class Setup:
    """Analysis settings shared by all figures.

    Attributes:
        criterion: ``"euclid"`` or ``"white"``.
        tolerance: ``"relative"`` or ``"absolute"``.
        epsilon: Tolerance value eps of the criterion (read from the csv).
        kappa_min: Smallest kappa used for fits and reference lines.
    """

    criterion: str
    tolerance: str
    epsilon: float
    kappa_min: float

    @property
    def criterion_text(self) -> str:
        """Mathtext of the convergence criterion, e.g. ``W_2 <= eps * sqrt(d/m)``."""
        cols = CRITERIA[self.criterion]
        scale = (
            cols.relative_scale if self.tolerance == "relative" else cols.absolute_scale
        )
        scale = "" if scale == "1" else rf"\,{scale}"
        return (
            rf"${cols.w2_symbol} \leq {cols.epsilon_symbol}{scale}$ "
            rf"({self.tolerance} tolerance, ${cols.epsilon_symbol}={self.epsilon:g}$)"
        )

    def title(self, headline: str) -> str:
        """Figure title: what is shown plus the (explicit) convergence criterion."""
        return f"{headline}\nconvergence criterion: {self.criterion_text}"


# ---------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------


def set_plot_style(use_latex: bool) -> None:
    """Sets a LaTeX-like look for all figures.

    Args:
        use_latex: If True, render text with a real LaTeX installation. Otherwise use
            matplotlib's built-in math renderer with Computer Modern fonts.
    """
    plt.rcParams.update(
        {
            "text.usetex": use_latex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"]
            if use_latex
            else ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,  # cmr10 has no unicode minus
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        }
    )


def present_samplers(table: pd.DataFrame) -> list[str]:
    """Returns the samplers occurring in ``table``, in the order of ``SAMPLERS``."""
    available = set(table["sampler"])
    return [name for name in SAMPLERS if name in available]


def sampler_subscript(sampler: str) -> str:
    """Returns the LaTeX subscript of a sampler for axis labels like T_{...}."""
    return SAMPLERS[sampler].subscript or rf"\mathrm{{{sampler}}}"


def reference_dim_exponent(sampler: str, tolerance: str) -> float:
    """Returns the theoretical exponent b in cost ~ d^b for the given tolerance.

    Args:
        sampler: Key of ``SAMPLERS``.
        tolerance: ``"relative"`` or ``"absolute"``.

    Returns:
        ``p`` for an absolute tolerance, ``p - q / 2`` for a relative one (see
        ``SamplerConfig``).
    """
    config = SAMPLERS[sampler]
    if tolerance == "absolute":
        return config.dim_exponent_abs
    return config.dim_exponent_abs - config.eps_exponent / 2


def format_exponent(exponent: float) -> str:
    """Formats an exponent as a short fraction, e.g. 0.5 -> '1/2', 1.0 -> '1'."""
    fraction = Fraction(exponent).limit_denominator(6)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def format_table(table: pd.DataFrame) -> str:
    """Formats a data frame for console output (without the ``kind`` and all-empty columns)."""
    table = table.drop(columns="kind", errors="ignore").dropna(axis=1, how="all")
    return table.to_string(index=False, float_format=lambda value: f"{value:.2f}")


def save_figure(figure: plt.Figure, path: Path, title: str | None = None) -> None:
    """Applies the layout, optionally adds a title, saves and closes the figure."""
    if title:
        figure.suptitle(title, fontsize=14)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        figure.tight_layout()
    figure.savefig(
        path, dpi=150, bbox_inches="tight"
    )  # also keeps legends placed outside
    plt.close(figure)


def set_readable_log_ticks(axis: plt.Axes, values: np.ndarray) -> None:
    """Labels the y-axis of a narrow log-scaled plot with round numbers.

    Args:
        axis: Axis with a logarithmic y-scale.
        values: Plotted y-values, used to choose the tick range.
    """
    low, high = np.nanmin(values) / 1.2, np.nanmax(values) * 1.2
    if high / low < 2.5:  # narrow range: finer ticks
        candidates = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.5, 1.7, 2])
    else:
        candidates = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64])
    ticks = candidates[(candidates >= low) & (candidates <= high)]
    axis.set_yticks(ticks)
    axis.set_yticklabels([f"{tick:g}" for tick in ticks])
    axis.minorticks_off()


# ---------------------------------------------------------------------------------------
# Loading: one table with a cost per run
# ---------------------------------------------------------------------------------------


def gamma_rule(gamma: float, kappa: float) -> str:
    """Identifies which friction rule produced a gamma value.

    The raw value changes with kappa for the c / sqrt(kappa) rules, so runs are grouped by rule.

    Args:
        gamma: Friction value of the run (NaN for samplers without friction).
        kappa: Condition number of the run.

    Returns:
        Key of ``GAMMA_RULES``, ``"-"`` if the sampler has no gamma, or the formatted value
        if no rule matches.
    """
    if pd.isna(gamma):
        return "-"
    if np.isclose(gamma, np.sqrt(2), rtol=1e-4):
        return "sqrt(2)"
    if np.isclose(gamma, 2.0, rtol=1e-4):
        return "2"
    for factor in (1, 2, 4):
        if np.isclose(gamma * np.sqrt(kappa), factor, rtol=1e-4):
            return f"{factor}/sqrt(k)"
    return f"{gamma:.3g}"


def add_sampler_variants(runs: pd.DataFrame) -> pd.DataFrame:
    """Creates one block of rows per entry of ``SAMPLERS``.

    A variant (e.g. RMULMC with gamma = 2) keeps only the runs of its base sampler with the
    fixed friction rule, so it is analysed independently of whether that rule is optimal.

    Args:
        runs: Runs of the base samplers with a ``gamma_rule`` column.

    Returns:
        Runs relabelled by the keys of ``SAMPLERS``.
    """
    blocks = []
    for name, config in SAMPLERS.items():
        rows = runs[runs["sampler"] == (config.base_sampler or name)]
        if config.only_gamma is not None:
            rows = rows[rows["gamma_rule"] == config.only_gamma]
        blocks.append(rows.assign(sampler=name))
    return pd.concat(blocks, ignore_index=True)


def prepare_runs(raw: pd.DataFrame, criterion: str) -> pd.DataFrame:
    """Selects the analysed samplers and defines the cost of every run.

    Args:
        raw: Content of the experiment csv.
        criterion: ``"euclid"`` or ``"white"``.

    Returns:
        One row per run with the columns ``cost`` (gradient evaluations until convergence,
        inf if not converged), ``budget`` (right-censoring level in the same unit),
        ``steps_converged``, ``final_error``, ``eps_used``, ``gamma_rule`` and
        ``relative_step`` (step size in the units of the experiment grid).
    """
    columns = CRITERIA[criterion]
    base_samplers = {config.base_sampler or name for name, config in SAMPLERS.items()}
    runs = raw[raw["sampler"].isin(base_samplers)].copy()
    if runs.empty:
        raise SystemExit(
            f"none of the samplers {sorted(base_samplers)} found in the csv"
        )
    runs["gamma_rule"] = [gamma_rule(gamma, kappa)
                          for gamma, kappa in zip(runs["gamma"], runs["kappa"])]  # fmt: skip
    runs = add_sampler_variants(runs)

    in_gamma_units = runs["sampler"].map(
        lambda name: SAMPLERS[name].step_in_gamma_units
    )
    runs["relative_step"] = np.where(in_gamma_units, runs["step_size"] / runs["gamma"],
                                     runs["step_size"])  # fmt: skip

    converged = runs[f"converged{columns.suffix}"].astype(bool) & ~runs[
        "diverged"
    ].astype(bool)
    runs["steps_converged"] = np.where(
        converged, runs[f"steps_to_conv{columns.suffix}"], np.inf
    )
    runs["cost"] = runs["steps_converged"] * runs["grads_per_step"]
    runs["budget"] = runs["max_steps"] * runs["grads_per_step"]
    runs["final_error"] = runs[columns.final_error].replace([np.inf, -np.inf], np.nan)
    runs["eps_used"] = runs[columns.epsilon]
    runs["floor_ok_used"] = runs[columns.floor_ok].astype(bool)
    runs["borderline_used"] = runs[columns.borderline].astype(bool)
    return runs


# ---------------------------------------------------------------------------------------
# Best configuration per cell and sanity checks
# ---------------------------------------------------------------------------------------


def select_best_configs(runs: pd.DataFrame, first_checkpoint: int) -> pd.DataFrame:
    """Selects the cheapest configuration of every (sampler, dimension, kappa) cell.

    The tuned samplers get more tries at the minimum of a noisy quantity than ULA (144 / 192
    vs 48 runs per cell); ``n_near_optimal`` shows how flat the optimum is.

    Args:
        runs: Output of ``prepare_runs``.
        first_checkpoint: First evaluation step of the experiment (``make_eval_steps``).
            Convergence at this step is flagged as left-censored.

    Returns:
        One row per cell. Cells where nothing converged keep the run with the smallest final
        error (``censored`` = True, for diagnostics only). Extra columns: ``n_near_optimal``,
        ``step_at_edge`` and ``left_censored``.
    """
    best = (
        runs.sort_values(["cost", "final_error"])
        .drop_duplicates(CELL_KEY)
        .sort_values(CELL_KEY)
        .reset_index(drop=True)
    )
    best_cost = runs.groupby(CELL_KEY)["cost"].transform("min")
    near_optimal = runs[
        np.isfinite(runs["cost"]) & (runs["cost"] <= NEAR_OPTIMAL_FACTOR * best_cost)
    ]
    counts = (
        near_optimal.groupby(CELL_KEY).size().rename("n_near_optimal").reset_index()
    )
    best = best.merge(counts, on=CELL_KEY, how="left").fillna({"n_near_optimal": 0})

    # The step-size grid is the same relative window for all samplers: detect its edges there.
    lowest, highest = runs["relative_step"].min(), runs["relative_step"].max()
    best["censored"] = ~np.isfinite(best["cost"])
    best["step_at_edge"] = (np.isclose(best["relative_step"], lowest)
                            | np.isclose(best["relative_step"], highest))  # fmt: skip
    best["left_censored"] = ~best["censored"] & (
        best["steps_converged"] <= first_checkpoint
    )
    return best


def run_sanity_checks(runs: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Summarises per sampler how trustworthy the optima are.

    Args:
        runs: Output of ``prepare_runs``.
        best: Output of ``select_best_configs``.

    Returns:
        Counts of censored / edge / left-censored / borderline cells, the median number of
        near-optimal configurations, the share of diverged runs and whether the Monte Carlo
        noise floor was below the tolerance everywhere.
    """
    per_cell = best.groupby("sampler").agg(
        cells=("cost", "size"),
        censored=("censored", "sum"),
        step_at_edge=("step_at_edge", "sum"),
        left_censored=("left_censored", "sum"),
        borderline=("borderline_used", "sum"),
        median_n_near_optimal=("n_near_optimal", "median"),
    )
    per_run = runs.groupby("sampler").agg(
        diverged_share=("diverged", "mean"),
        floor_ok_everywhere=("floor_ok_used", "all"),
    )
    return per_cell.join(per_run)


# ---------------------------------------------------------------------------------------
# Exponents (log-log least squares on uncensored cells)
# ---------------------------------------------------------------------------------------


def fit_power_law(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    """Fits y ~ x^exponent by least squares in log-log space.

    Args:
        x_values: Positive x-values.
        y_values: Positive y-values.

    Returns:
        ``(exponent, std_error)``. The standard error is the scatter around the line and NaN
        for fewer than three points; the exponent is NaN for fewer than two.
    """
    log_x, log_y = np.log(x_values), np.log(y_values)
    num_points = len(log_x)
    if num_points < 2:
        return np.nan, np.nan
    design = np.c_[np.ones(num_points), log_x]
    intercept_and_slope = np.linalg.lstsq(design, log_y, rcond=None)[0]
    if num_points == 2:
        return float(intercept_and_slope[1]), np.nan
    residuals = log_y - design @ intercept_and_slope
    variance = (residuals @ residuals) / (num_points - 2)
    std_error = np.sqrt(variance / ((log_x - log_x.mean()) ** 2).sum())
    return float(intercept_and_slope[1]), float(std_error)


def kappa_exponents(best: pd.DataFrame, kappa_min: float) -> pd.DataFrame:
    """Fits cost ~ kappa^alpha per (sampler, dimension) for kappa >= ``kappa_min``."""
    rows = []
    for (sampler, dimension), group in best[~best["censored"]].groupby(
        ["sampler", "dimension"]
    ):
        group = group[group["kappa"] >= kappa_min]
        exponent, std_error = fit_power_law(
            group["kappa"].to_numpy(), group["cost"].to_numpy()
        )
        rows.append(dict(kind="kappa", sampler=sampler, dimension=dimension, exponent=exponent,
                         std_error=std_error, num_points=len(group),
                         reference=SAMPLERS[sampler].kappa_exponent))  # fmt: skip
    return pd.DataFrame(rows, columns=FIT_COLUMNS)


def dimension_exponents(
    best: pd.DataFrame, kappa_min: float, tolerance: str
) -> pd.DataFrame:
    """Fits cost ~ d^beta per (sampler, kappa); needs >= 2 dimensions per fit."""
    rows = []
    candidates = best[~best["censored"] & (best["kappa"] >= kappa_min)]
    for (sampler, kappa), group in candidates.groupby(["sampler", "kappa"]):
        if group["dimension"].nunique() < 2:
            continue
        exponent, std_error = fit_power_law(
            group["dimension"].to_numpy(), group["cost"].to_numpy()
        )
        rows.append(dict(kind="dimension", sampler=sampler, kappa=kappa, exponent=exponent,
                         std_error=std_error, num_points=len(group),
                         reference=reference_dim_exponent(sampler, tolerance)))  # fmt: skip
    return pd.DataFrame(rows, columns=FIT_COLUMNS)


def local_kappa_slopes(best: pd.DataFrame) -> pd.DataFrame:
    """Slope of log(cost) between neighbouring kappa values (shows where a power law starts)."""
    rows = []
    for (sampler, dimension), group in best[~best["censored"]].groupby(
        ["sampler", "dimension"]
    ):
        group = group.sort_values("kappa")
        log_kappa, log_cost = (
            np.log(group["kappa"].to_numpy()),
            np.log(group["cost"].to_numpy()),
        )
        for index in range(len(group) - 1):
            rows.append(dict(
                sampler=sampler,
                dimension=dimension,
                kappa_mid=float(np.exp((log_kappa[index] + log_kappa[index + 1]) / 2)),
                slope=(log_cost[index + 1] - log_cost[index])
                / (log_kappa[index + 1] - log_kappa[index]),
            ))  # fmt: skip
    return pd.DataFrame(rows, columns=["sampler", "dimension", "kappa_mid", "slope"])


def speedup_table(best: pd.DataFrame) -> pd.DataFrame:
    """Computes cost(BASELINE) / cost(sampler) for every other sampler and cell.

    If the baseline is censored but the sampler converged, the ratio is only a lower bound
    (baseline budget / sampler cost) and flagged. If the sampler is censored, the cell is
    dropped.

    Args:
        best: Output of ``select_best_configs``.

    Returns:
        Columns ``sampler``, ``dimension``, ``kappa``, ``speedup``, ``is_lower_bound``.
    """
    columns = ["sampler", "dimension", "kappa", "speedup", "is_lower_bound"]
    if BASELINE not in set(best["sampler"]):
        return pd.DataFrame(columns=columns)

    baseline = best[best["sampler"] == BASELINE][
        ["dimension", "kappa", "cost", "budget"]
    ]
    baseline = baseline.rename(
        columns={"cost": "baseline_cost", "budget": "baseline_budget"}
    )
    others = best[best["sampler"] != BASELINE][
        ["sampler", "dimension", "kappa", "cost"]
    ]
    table = others.merge(baseline, on=["dimension", "kappa"])

    baseline_ok = np.isfinite(table["baseline_cost"])
    sampler_ok = np.isfinite(table["cost"])
    table["is_lower_bound"] = ~baseline_ok & sampler_ok
    table["speedup"] = np.where(
        baseline_ok & sampler_ok, table["baseline_cost"] / table["cost"], np.nan
    )
    bound = table["is_lower_bound"]
    table.loc[bound, "speedup"] = (table["baseline_budget"] / table["cost"])[bound]
    return (
        table[columns]
        .sort_values(["sampler", "dimension", "kappa"])
        .reset_index(drop=True)
    )


def speedup_exponents(speedups: pd.DataFrame, kappa_min: float) -> pd.DataFrame:
    """Fits speedup ~ kappa^alpha; the reference is the difference of the cost exponents."""
    rows = []
    for (sampler, dimension), group in speedups.groupby(["sampler", "dimension"]):
        group = group[(group["kappa"] >= kappa_min) & ~group["is_lower_bound"]
                      & group["speedup"].notna()]  # fmt: skip
        exponent, std_error = fit_power_law(
            group["kappa"].to_numpy(), group["speedup"].to_numpy()
        )
        reference = SAMPLERS[BASELINE].kappa_exponent - SAMPLERS[sampler].kappa_exponent
        rows.append(dict(kind="speedup", sampler=f"{BASELINE}/{sampler}", dimension=dimension,
                         exponent=exponent, std_error=std_error, num_points=len(group),
                         reference=reference))  # fmt: skip
    return pd.DataFrame(rows, columns=FIT_COLUMNS)


# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------


def plot_cost_vs_kappa(best: pd.DataFrame, fits: pd.DataFrame, slopes: pd.DataFrame,
                       dimensions: list[int], setup: Setup, path: Path) -> None:  # fmt: skip
    """Figure 1: cost vs kappa (top) and local slopes (bottom), one column per dimension.

    Args:
        best: Output of ``select_best_configs``.
        fits: Kappa exponents from ``kappa_exponents``.
        slopes: Local slopes from ``local_kappa_slopes``.
        dimensions: Dimensions to show (one column each).
        setup: Analysis settings.
        path: Output file.
    """
    fitted = {(row.sampler, row.dimension): (row.exponent, row.std_error)
              for row in fits.itertuples()}  # fmt: skip
    figure, axes = plt.subplots(2, len(dimensions), figsize=(4.3 * len(dimensions), 6.2),
                                sharex=True, squeeze=False,
                                gridspec_kw=dict(height_ratios=[2, 1]))  # fmt: skip
    smallest_kappa = best["kappa"].min()
    for column, dimension in enumerate(dimensions):
        cost_axis, slope_axis = axes[0, column], axes[1, column]
        for sampler in present_samplers(best):
            config = SAMPLERS[sampler]
            cell = best[(best["sampler"] == sampler) & (best["dimension"] == dimension)]
            cell = cell.sort_values("kappa")
            converged = ~cell["censored"]
            exponent, std_error = fitted.get((sampler, dimension), (np.nan, np.nan))
            label = config.label
            if np.isfinite(exponent):
                error_text = rf"\pm{std_error:.2f}" if np.isfinite(std_error) else ""
                label = rf"{config.label}: $\alpha={exponent:.2f}{error_text}$"
            cost_axis.plot(cell.loc[converged, "kappa"], cell.loc[converged, "cost"],
                           marker=config.marker, ls="-", color=config.color, label=label)  # fmt: skip
            cost_axis.plot(
                cell.loc[~converged, "kappa"],
                cell.loc[~converged, "budget"],
                "v",
                mfc="none",
                color=config.color,
            )  # censored: hollow triangle at budget
            fit_region = cell[converged & (cell["kappa"] >= setup.kappa_min)]
            if len(
                fit_region
            ):  # reference slope, anchored at the first point of the fit region
                anchor_kappa, anchor_cost = (
                    fit_region.iloc[0]["kappa"],
                    fit_region.iloc[0]["cost"],
                )
                kappas = cell.loc[cell["kappa"] >= setup.kappa_min, "kappa"].to_numpy()
                cost_axis.plot(kappas, anchor_cost * (kappas / anchor_kappa) ** config.kappa_exponent,
                               "--", color=config.color, alpha=0.45, lw=1)  # fmt: skip
            sampler_slopes = slopes[
                (slopes["sampler"] == sampler) & (slopes["dimension"] == dimension)
            ]
            slope_axis.plot(sampler_slopes["kappa_mid"], sampler_slopes["slope"],
                            marker=config.marker, ls="-", color=config.color)  # fmt: skip
        if setup.kappa_min > smallest_kappa:  # grey = not used for the fit
            for axis in (cost_axis, slope_axis):
                axis.axvspan(
                    smallest_kappa / 1.3, setup.kappa_min / 1.15, color="0.93", zorder=0
                )
        for reference in sorted(
            {SAMPLERS[name].kappa_exponent for name in present_samplers(best)}
        ):
            slope_axis.axhline(reference, c="gray", ls=":", lw=1)
        cost_axis.set(xscale="log", yscale="log", title=rf"$d={dimension}$")
        slope_axis.set(xscale="log", xlabel=r"condition number $\kappa$")
        legend_title = (
            rf"fit for $\kappa\geq{setup.kappa_min:g}$; dashed: reference slope"
        )
        cost_axis.legend(fontsize=8, title=legend_title, title_fontsize=7)
    axes[0, 0].set_ylabel("gradient evaluations $T$")
    axes[1, 0].set_ylabel(r"local slope $\mathrm{d}\log T\,/\,\mathrm{d}\log\kappa$")
    headline = r"Cost $T$ vs condition number (best $h,\gamma$ per point; $\triangledown$ = not converged)"
    save_figure(figure, path, setup.title(headline))


def plot_speedup(speedups: pd.DataFrame, setup: Setup, path: Path) -> None:
    """Figure 2: speed-up over the baseline vs kappa, one panel per sampler.

    Args:
        speedups: Output of ``speedup_table``.
        setup: Analysis settings.
        path: Output file.
    """
    speedups = speedups[speedups["speedup"].notna()]
    samplers = [name for name in SAMPLERS if name in set(speedups["sampler"])]
    if not samplers:
        return
    figure, axes = plt.subplots(1, len(samplers), figsize=(4.8 * len(samplers), 4.0),
                                sharey=True, squeeze=False)  # fmt: skip
    for axis, sampler in zip(axes[0], samplers):
        reference = SAMPLERS[BASELINE].kappa_exponent - SAMPLERS[sampler].kappa_exponent
        for color_index, (dimension, group) in enumerate(
            speedups[speedups["sampler"] == sampler].groupby("dimension")
        ):
            group = group.sort_values("kappa")
            color = f"C{color_index}"
            exact, bounds = (
                group[~group["is_lower_bound"]],
                group[group["is_lower_bound"]],
            )
            axis.plot(
                exact["kappa"],
                exact["speedup"],
                "o-",
                color=color,
                label=rf"$d={dimension}$",
            )
            axis.plot(bounds["kappa"], bounds["speedup"], "^", mfc="none", color=color)
            fit_region = exact[exact["kappa"] >= setup.kappa_min]
            if len(fit_region):
                anchor_kappa, anchor_speedup = (
                    fit_region.iloc[0]["kappa"],
                    fit_region.iloc[0]["speedup"],
                )
                kappas = group.loc[
                    group["kappa"] >= setup.kappa_min, "kappa"
                ].to_numpy()
                axis.plot(kappas, anchor_speedup * (kappas / anchor_kappa) ** reference, "--",
                          color=color, alpha=0.45, lw=1)  # fmt: skip
        panel_title = (f"{SAMPLERS[sampler].label} vs {BASELINE} "
                       rf"(dashed: $\kappa^{{{format_exponent(reference)}}}$)")  # fmt: skip
        axis.set(
            xscale="log",
            yscale="log",
            xlabel=r"condition number $\kappa$",
            title=panel_title,
        )
        axis.set_ylabel(
            rf"speed-up $T_{{\mathrm{{{BASELINE}}}}}/T_{{{sampler_subscript(sampler)}}}$"
        )
        axis.legend(fontsize=8)
    headline = (
        r"Speed-up over "
        + BASELINE
        + r" ($\triangle$ = lower bound, baseline not converged)"
    )
    save_figure(figure, path, setup.title(headline))


def plot_cost_vs_dimension(best: pd.DataFrame, setup: Setup, path: Path) -> bool:
    """Figure 3: cost(d) / cost(d_min) at fixed kappa, one panel per sampler.

    The dashed line is the theoretical reference (flat for the relative tolerance, d^p for the
    absolute tolerance). The legend gives the fitted exponent beta (cost ~ d^beta) per kappa.

    Args:
        best: Output of ``select_best_configs``.
        setup: Analysis settings.
        path: Output file.

    Returns:
        False (and no figure) if the data contain fewer than two dimensions.
    """
    dimensions = sorted(best["dimension"].unique())
    if len(dimensions) < 2:
        return False
    dimension_values = np.array(dimensions, dtype=float)
    samplers = present_samplers(best)
    figure, axes = plt.subplots(1, len(samplers), figsize=(4.2 * len(samplers), 4.0),
                                sharey=True, squeeze=False)  # fmt: skip
    plotted = []
    for axis, sampler in zip(axes[0], samplers):
        candidates = best[(best["sampler"] == sampler) & ~best["censored"]
                          & (best["kappa"] >= setup.kappa_min)]  # fmt: skip
        for kappa, group in candidates.groupby("kappa"):
            group = group.sort_values("dimension")
            if (
                group["dimension"].nunique() < 2
                or group["dimension"].iloc[0] != dimensions[0]
            ):
                continue
            exponent, _ = fit_power_law(
                group["dimension"].to_numpy(), group["cost"].to_numpy()
            )
            relative_cost = group["cost"] / group["cost"].iloc[0]
            plotted.append(relative_cost.to_numpy())
            axis.plot(group["dimension"], relative_cost, "o-",
                      label=rf"$\kappa={kappa:g}$, $\beta={exponent:.2f}$")  # fmt: skip
        reference = reference_dim_exponent(sampler, setup.tolerance)
        reference_curve = (dimension_values / dimension_values[0]) ** reference
        plotted.append(reference_curve)
        axis.plot(dimension_values, reference_curve, c="k", ls="--", lw=1,
                  label=rf"theory: $d^{{{format_exponent(reference)}}}$")  # fmt: skip
        axis.set(
            xscale="log",
            yscale="log",
            xlabel=r"dimension $d$",
            title=SAMPLERS[sampler].label,
        )
        axis.set_xticks(dimension_values)
        axis.set_xticklabels([f"{int(value)}" for value in dimension_values])
        axis.minorticks_off()
    set_readable_log_ticks(axes[0, 0], np.concatenate(plotted))
    axes[0, 0].set_ylabel(r"$T(d)\,/\,T(d_{\min})$")
    axes[0, 0].legend(fontsize=6.5)
    if setup.tolerance == "relative":
        headline = r"Dimension dependence of the cost (theory for the relative tolerance: independent of $d$)"
    else:
        headline = r"Dimension dependence of the cost (theory for the absolute tolerance: $T\sim d^{p}$)"
    save_figure(figure, path, setup.title(headline))
    return True


def plot_step_size_and_gamma(best: pd.DataFrame, setup: Setup, path: Path) -> None:
    """Figure 4: absolute step size and winning friction rule of the optimal configurations.

    Left: optimal step size h* (absolute, so samplers can be compared directly; ULMC's h*
    scales with its gamma). Right: which friction rule wins, for samplers with a free gamma.
    Crosses mark optima at the edge of the step-size grid, i.e. where the grid should be extended.

    Args:
        best: Output of ``select_best_configs``.
        setup: Analysis settings.
        path: Output file.
    """
    dimensions = sorted(best["dimension"].unique())
    gamma_samplers = [
        name for name in present_samplers(best) if SAMPLERS[name].has_free_gamma
    ]
    num_panels = 1 + bool(gamma_samplers)
    figure, axes = plt.subplots(
        1, num_panels, figsize=(5.4 * num_panels, 5.2), squeeze=False
    )
    step_axis = axes[0, 0]

    def shade(dimension_index: int) -> float:
        """Opacity that increases with the dimension."""
        return 0.4 + 0.6 * (dimension_index + 1) / len(dimensions)

    for sampler in present_samplers(best):
        config = SAMPLERS[sampler]
        for dimension_index, dimension in enumerate(dimensions):
            cell = best[(best["sampler"] == sampler) & (best["dimension"] == dimension)
                        & ~best["censored"]].sort_values("kappa")  # fmt: skip
            if cell.empty:
                continue
            step_axis.plot(cell["kappa"], cell["step_size"], marker=config.marker, ls="-", ms=4,
                           color=config.color, alpha=shade(dimension_index),
                           label=rf"{config.label}, $d={dimension}$")  # fmt: skip
            at_edge = cell[cell["step_at_edge"]]
            step_axis.plot(at_edge["kappa"], at_edge["step_size"], "kx", ms=8)
    step_axis.plot([], [], "kx", label="optimum at grid edge (extend grid)")
    step_axis.axhline(ULA_STABILITY_LIMIT, c="r", ls=":", lw=1)
    step_axis.set(xscale="log", yscale="log", xlabel=r"condition number $\kappa$",
                  ylabel=r"optimal step size $h^\ast$",
                  title=r"step size (red: ULA stability limit $2/L$)")  # fmt: skip
    step_axis.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))

    if gamma_samplers:
        gamma_axis = axes[0, 1]
        rule_keys = list(GAMMA_RULES)
        for sampler_index, sampler in enumerate(gamma_samplers):
            for dimension_index, dimension in enumerate(dimensions):
                cell = best[(best["sampler"] == sampler) & (best["dimension"] == dimension)
                            & ~best["censored"]].sort_values("kappa")  # fmt: skip
                rule_positions = np.array([rule_keys.index(rule) if rule in rule_keys else np.nan
                                           for rule in cell["gamma_rule"]])  # fmt: skip
                # offsets per sampler and per dimension keep overlapping points visible
                offset = (0.28 * (sampler_index - (len(gamma_samplers) - 1) / 2)
                          + 0.08 * (dimension_index - (len(dimensions) - 1) / 2))  # fmt: skip
                gamma_axis.plot(cell["kappa"], rule_positions + offset, "o",
                                color=SAMPLERS[sampler].color, alpha=shade(dimension_index),
                                label=rf"{SAMPLERS[sampler].label}, $d={dimension}$")  # fmt: skip
        gamma_axis.set_yticks(range(len(rule_keys)))
        gamma_axis.set_yticklabels(list(GAMMA_RULES.values()))
        gamma_axis.set(xscale="log", xlabel=r"condition number $\kappa$",
                       title=r"winning $\gamma$ ($c/\sqrt{\kappa}$ = momentum regime)")  # fmt: skip
        gamma_axis.legend(
            fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2)
        )
    save_figure(figure, path, setup.title("Optimal hyperparameters"))


# ---------------------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parses the command line (see the module docstring for the meaning of the options)."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("csv", help="result file of dimension_conditioning.py")
    parser.add_argument("--out", default="analysis", help="output directory")
    parser.add_argument("--tolerance", choices=["relative", "absolute"], default="relative",
                        help="tolerance of the run: relative = W2 <= eps*sqrt(d/m) (kappa sweep), "
                        "absolute = W2 <= eps*sqrt(1/m) (dimension sweep)")  # fmt: skip
    parser.add_argument("--criterion", choices=list(CRITERIA), default="euclid",
                        help="euclid = normalised Euclidean W2, white = whitened W2")  # fmt: skip
    parser.add_argument("--kappa-min", type=float, default=100.0,
                        help="fit exponents for kappa >= this value (below, constants dominate)")  # fmt: skip
    parser.add_argument("--fig1-dims", type=int, nargs="+", default=None, metavar="D",
                        help="dimensions shown in figure 1 (default: all)")  # fmt: skip
    parser.add_argument("--first-checkpoint", type=int, default=5,
                        help="first evaluation step (make_eval_steps); convergence there is "
                        "flagged as left-censored")  # fmt: skip
    parser.add_argument("--usetex", action="store_true",
                        help="render text with a real LaTeX installation")  # fmt: skip
    return parser.parse_args()


def main() -> None:
    """Runs the analysis and writes tables and figures to the output directory."""
    args = parse_arguments()
    set_plot_style(args.usetex)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.csv)
    known = {config.base_sampler or name for name, config in SAMPLERS.items()}
    if ignored := sorted(set(raw["sampler"]) - known):
        print(f"note: samplers in the csv but not in SAMPLERS are ignored: {ignored}")

    runs = prepare_runs(raw, args.criterion)
    setup = Setup(
        args.criterion, args.tolerance, float(runs["eps_used"].iloc[0]), args.kappa_min
    )
    best = select_best_configs(runs, args.first_checkpoint)
    checks = run_sanity_checks(runs, best)
    print(f"samplers analysed: {present_samplers(runs)}")
    print(
        f"\n=== sanity checks ({args.criterion}, {args.tolerance} tolerance) ===\n{checks.to_string()}"
    )

    kappa_fits = kappa_exponents(best, args.kappa_min)
    speedups = speedup_table(best)
    speedup_fits = speedup_exponents(speedups, args.kappa_min)
    dimension_fits = dimension_exponents(best, args.kappa_min, args.tolerance)
    print(
        f"\n=== cost ~ kappa^alpha for kappa >= {args.kappa_min:g} ===\n{format_table(kappa_fits)}"
    )
    if len(speedup_fits):
        print(
            f"\n=== speed-up over {BASELINE} ~ kappa^alpha ===\n{format_table(speedup_fits)}"
        )
    if len(dimension_fits):
        medians = (
            dimension_fits.groupby("sampler")[["exponent", "reference"]]
            .median()
            .reset_index()
        )
        print(f"\n=== cost ~ d^beta, median over kappa ===\n{format_table(medians)}")

    available_dimensions = sorted(best["dimension"].unique())
    fig1_dimensions = args.fig1_dims or available_dimensions
    if unknown := sorted(set(fig1_dimensions) - set(available_dimensions)):
        raise SystemExit(
            f"--fig1-dims {unknown} not in the data (available: {available_dimensions})"
        )

    plot_cost_vs_kappa(best, kappa_fits, local_kappa_slopes(best), sorted(fig1_dimensions), setup,
                       output_dir / "fig1_cost_vs_kappa.png")  # fmt: skip
    plot_speedup(speedups, setup, output_dir / "fig2_speedup.png")
    plot_cost_vs_dimension(best, setup, output_dir / "fig3_cost_vs_dimension.png")
    plot_step_size_and_gamma(best, setup, output_dir / "fig4_step_size_and_gamma.png")

    best.to_csv(output_dir / "best_configs.csv", index=False)
    checks.to_csv(output_dir / "checks.csv")
    pd.concat(
        [fits for fits in (kappa_fits, speedup_fits, dimension_fits) if len(fits)]
    ).to_csv(output_dir / "exponents.csv", index=False)
    best.assign(cost=best["cost"].replace(np.inf, np.nan)).pivot_table(
        index="kappa", columns=["sampler", "dimension"], values="cost", aggfunc="first"
    ).to_csv(output_dir / "table_cost.csv")
    print(f"\nwritten to {output_dir}/")


if __name__ == "__main__":
    main()
