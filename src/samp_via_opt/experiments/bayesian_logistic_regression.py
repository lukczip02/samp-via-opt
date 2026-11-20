"""
Real-world illustration: Bayesian logistic regression on the UCI heart disease data (Cleveland, d = 13).

Setting follows the Bayesian experiment of the preconditioning paper: anisotropic Gaussian prior with
linearly interpolated variances in [0.1, 10] on the regression coefficients.

Differences to the main experiments (dimension_conditioning.py):
- Single problem, no sweep over (d, kappa). Same samplers, same step-size / friction grids, same metrics.
- The posterior is not Gaussian and has no closed-form moments. Reference mean / covariance are estimated
  once with a MALA ensemble. The ensemble evaluator then compares the first two
  moments of the sampler ensemble to these reference moments (Gaussian W2 distance, lower bound on actual W2 which
  is accurate for near-Gaussian posteriors; skewness / kurtosis of the reference are printed as a check).
"""

import math
import urllib.request
from pathlib import Path
from typing import NamedTuple

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from samp_via_opt.experiments.dimension_conditioning import SamplerSpec
from samp_via_opt.experiments.evaluation import (
    build_ensemble_evaluator,
    convergence_table,
    make_eval_steps,
    noise_floor,
    slow_direction_init,
    timed_call,
)
from samp_via_opt.samplers import ULA, ULMC, RMULMC, RMULMCAdv
from samp_via_opt.targets.gaussian import GaussianTarget
from samp_via_opt.targets.logistic_regression import LogisticRegressionTarget

HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------


def load_heart_disease(
    path: str | Path = "data/processed.cleveland.data", scaling: str = "minmax"
) -> tuple[jax.Array, jax.Array]:
    """
    Load the Cleveland heart disease data, drop rows with missing values (N = 297) and binarise the label
    (any disease -> 1). The file is downloaded from UCI if it does not exist yet.

    The feature scaling decides how ill-conditioned the posterior is (no intercept column is added, d = 13):
        "minmax":   features in [0, 1]. Default, matches the range of the main experiments.
        "standard": z-scores.

    Args:
        path: Location of processed.cleveland.data.
        scaling: Either "minmax" or "standard".

    Returns:
        Design matrix X (N, 13) and binary labels y (N,).
    """
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HEART_URL, path)

    df = pd.read_csv(path, header=None, na_values="?").dropna()
    X = df.iloc[:, :13].to_numpy(dtype=float)
    y = (df.iloc[:, 13].to_numpy() > 0).astype(float)

    if scaling == "minmax":
        X = (X - X.min(0)) / (X.max(0) - X.min(0))
    elif scaling == "standard":
        X = (X - X.mean(0)) / X.std(0)
    else:
        raise ValueError(f"unknown scaling: {scaling}")
    return jnp.asarray(X), jnp.asarray(y)


# ----------------------------------------------------------------------------------
# reference posterior (ground truth for the evaluator)
# ----------------------------------------------------------------------------------


class ReferencePosterior(NamedTuple):
    mean: jax.Array  # (d,)
    cov: jax.Array  # (d, d)
    accept_rate: float  # mean MALA acceptance probability
    drift: float  # relative change of lambda_max(cov) between T/2 and T (should be ~ 0)
    skewness: jax.Array  # (d,) of the standardised marginals
    excess_kurtosis: jax.Array  # (d,)


def reference_posterior(
    target: LogisticRegressionTarget,
    key: jax.Array,
    num_chains: int = 50_000,
    num_steps: int = 50_000,
    step_size: float | None = None,
) -> ReferencePosterior:
    """
    Estimate posterior mean and covariance with a MALA ensemble (blackJAX).

    Chains start from prior draws and all run for `num_steps` steps. Moments are taken across the chains
    at the final step only.

    Args:
        target: Logistic regression posterior.
        key: PRNGKey.
        num_chains: Number of parallel MALA chains. Larger than the evaluated samplers
            (num_chains in run_logreg_experiment), so that the reference error is minimal.
        num_steps: MALA steps per chain. Large value to ensure convergence of the slowest direction (lambda_max(cov)).
        step_size: Fixed MALA step size. If None, set to 2.0 / L (L = target.L after rescaling).

    Returns:
        Reference moments and diagnostics.
    """
    d = target.dim

    if step_size is None:
        step_size = 2.0 / target.L

    mala = blackjax.mala(target.log_prob, step_size)
    batched_step = jax.vmap(mala.step)

    def advance(states, key, n):
        def body(states, k):
            states, info = batched_step(jax.random.split(k, num_chains), states)
            return states, jnp.mean(info.acceptance_rate)

        return jax.lax.scan(body, states, jax.random.split(key, n))

    # initialize chains from the prior (~ N(0, diag(1/prior_prec)))
    key_init, key_a, key_b = jax.random.split(key, 3)
    x0 = jax.random.normal(key_init, (num_chains, d)) / jnp.sqrt(target.prior_prec)
    states = jax.vmap(mala.init)(x0)

    # two halves, so that we can check that the second half did not change the (slowest) covariance direction
    states, acc_a = advance(states, key_a, num_steps // 2)
    lam_half = jnp.linalg.eigvalsh(jnp.cov(states.position.T))[-1]
    states, acc_b = advance(states, key_b, num_steps - num_steps // 2)

    x = states.position
    mean, cov = jnp.mean(x, axis=0), jnp.cov(x.T)
    u = (x - mean) / jnp.sqrt(jnp.diag(cov))  # standardised marginals
    return ReferencePosterior(
        mean=mean,
        cov=cov,
        accept_rate=float(jnp.mean(jnp.concatenate([acc_a, acc_b]))),
        drift=float(lam_half / jnp.linalg.eigvalsh(cov)[-1] - 1.0),  # should be ~ 0
        skewness=jnp.mean(u**3, axis=0),
        excess_kurtosis=jnp.mean(u**4, axis=0) - 3.0,
    )


# ----------------------------------------------------------------------------------
# experiment
# ----------------------------------------------------------------------------------


def run_logreg_experiment(
    target: LogisticRegressionTarget,
    sampler_specs: dict[str, SamplerSpec],
    kappa_eff: float,
    rel_step_sizes: jax.Array,
    eval_steps: np.ndarray,
    num_chains: int,
    target_eps: float = 0.1,
    target_eps_white: float = 0.2,
    init_sigma: float = 3.0,
    use_float32: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Step-size grid search for all samplers on one logistic regression posterior.
    Mirrors the inner loop of run_grid_search (absolute tolerance: W2 <= sqrt(1/m) * eps).
    `target` must carry the reference moments (see LogisticRegressionTarget.with_moments).

    Args:
        target: Posterior with reference mean / cov.
        sampler_specs: Sampler name -> SamplerSpec (same objects as in the main experiments).
        kappa_eff: Effective condition number L * lambda_max(posterior covariance) (= L / lambda_min(H) for a Gaussian).
            Used for the friction parameter choice. Gaussian approximation.
        rel_step_sizes: Shared relative step-size window, see SamplerSpec.step_unit.
        eval_steps: Step numbers at which to evaluate the metrics.
        num_chains: Number of parallel chains of the evaluated samplers.
        target_eps: Convergence threshold (tolerance-normalised W2).
        target_eps_white: Convergence threshold in whitened coordinates (primary criterion).
        init_sigma: Start x0 is `init_sigma` posterior std devs out along the slowest direction.
        use_float32: Run chains in float32, metrics in float64.
        seed: Random seed.

    Returns:
        DataFrame in the format of convergence_table(), plus gamma, kappa_eff, num_chains.
    """
    dim = target.dim
    key_run, key_floor, key_floor_w = jax.random.split(jax.random.PRNGKey(seed), 3)
    target_white = GaussianTarget(mean=jnp.zeros(dim), cov=jnp.eye(dim), diag=True)
    x0 = slow_direction_init(target, n_sigma=init_sigma)

    norm = float(jnp.sqrt(1.0 / target.m))
    norm_white = 1.0

    # MC floor of the ensemble estimator, using exact draws from N(ref mean, ref cov)
    floor = float(noise_floor(target, key_floor, num_chains))
    floor_white = float(noise_floor(target_white, key_floor_w, num_chains))
    print(f"Number of chains: {num_chains}")
    print(f"MC noise floor (normalised): {floor / norm:.4f}  vs eps = {target_eps}")
    print(f"whitened floor: {floor_white:.4f}  vs eps_white = {target_eps_white}")

    tables = []
    for name, spec in sampler_specs.items():
        for gamma in spec.gammas(kappa_eff):
            sampler = spec.factory(target, gamma)
            step_sizes = spec.step_unit(gamma) * rel_step_sizes
            eval_fn = build_ensemble_evaluator(
                sampler, num_chains, eval_steps, use_float32
            )
            metrics, sec = timed_call(eval_fn, step_sizes, x0, key_run, target)

            label = name if gamma is None else f"{name} (gamma={gamma:.3f})"
            print(f"{label} done in {sec:.2f}s")

            df = convergence_table(
                metrics,
                step_sizes,
                name,
                dim,
                eval_steps,
                target_eps,
                target_eps_white,
                norm,
                norm_white,
                floor,
                floor_white,
                runtime_sec=sec,
            )
            tables.append(
                df.assign(gamma=gamma, kappa_eff=kappa_eff, num_chains=num_chains)
            )
    return pd.concat(tables, ignore_index=True)


def summarize(df: pd.DataFrame, criterion: str) -> pd.DataFrame:
    """Best step size per (sampler, gamma) w.r.t. gradient evaluations to the specified criterion."""
    rows = []
    for (name, gamma), g in df.groupby(["sampler", "gamma"], dropna=False, sort=False):
        r = g.loc[g[criterion].idxmin()]
        rows.append(
            {
                "sampler": name,
                "gamma": gamma,
                "best_step_size": r["step_size"],
                "grad_evals_white": r["grad_evals_to_conv_white"],
                "grad_evals_norm": r["grad_evals_to_conv"],
                "final_mean_err_white": r["final_mean_err_white"],
                "final_cov_err_white": r["final_cov_err_white"],
                "sec_per_step": r["sec_per_step"],
                "at_grid_edge": bool(
                    r["step_size"] in (g["step_size"].min(), g["step_size"].max())
                ),
            }
        )
    return pd.DataFrame(rows)


def make_sampler_specs(sampler_dtype) -> dict[str, SamplerSpec]:
    """Same sampler configurations as in dimension_conditioning.py (valid as is, since L = 1 here)."""
    return {
        "ULA": SamplerSpec(
            factory=lambda t, g: ULA(step_size=1.0, dtype=sampler_dtype),
            gammas=lambda kappa: [None],
            step_unit=lambda g: 1.0,
        ),
        "ULMC": SamplerSpec(
            factory=lambda t, g: ULMC(
                step_size=1.0, alpha=g, beta=1.0 / t.L, dtype=sampler_dtype
            ),
            gammas=lambda kappa: [2.0 / math.sqrt(kappa), math.sqrt(2.0), 2.0],
            step_unit=lambda g: g if g is not None else 1.0,
        ),
        "RMULMC": SamplerSpec(
            factory=lambda t, g: RMULMC(
                step_size=1.0, gamma=g, u=1.0 / t.L, dtype=sampler_dtype
            ),
            gammas=lambda kappa: [c / math.sqrt(kappa) for c in (1.0, 2.0, 4.0)]
            + [2.0],
            step_unit=lambda g: 1.0,
        ),
        "RMULMCAdv": SamplerSpec(
            factory=lambda t, g: RMULMCAdv(
                step_size=1.0, gamma=g, beta=1.0 / t.L, dtype=sampler_dtype
            ),
            gammas=lambda kappa: [c / math.sqrt(kappa) for c in (1.0, 2.0, 4.0)]
            + [2.0],
            step_unit=lambda g: 1.0,
        ),
    }


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    use_float32 = True
    scaling = "minmax"
    num_chains = 10_000
    target_eps = 0.1  # convergence threshold in original coordinates
    target_eps_white = 0.2  # convergence threshold in whitened coordinates
    rel_step_sizes = jnp.logspace(jnp.log10(0.1), jnp.log10(8.0), 48)

    # problem and reference posterior
    X, y = load_heart_disease(scaling=scaling)
    d = X.shape[1]
    prior_var = jnp.linspace(0.1, 10.0, d)  # linearly interpolated prior variances
    target = LogisticRegressionTarget(X, y, prior_var)

    # step size yields high acceptance ~ 0.8, further increasing step size is unstable (high skewness and kurtosis)
    ref = reference_posterior(target, jax.random.PRNGKey(0), step_size=2.0 / target.L)
    target = target.with_moments(ref.mean, ref.cov)

    # kappa_global: worst case over the whole space (theory); kappa_eff: curvature where the mass is
    kappa_global = target.L / target.m
    kappa_eff = target.L * float(
        jnp.linalg.eigvalsh(ref.cov)[-1]
    )  # Gaussian approximation, i.e. L / lambda_min(H)
    print(f"N = {X.shape[0]}, d = {d}, scaling = {scaling}")
    print(f"kappa_global = {kappa_global:.0f}, kappa_eff = {kappa_eff:.0f}")
    print(
        f"reference MALA (blackJAX): acceptance = {ref.accept_rate:.2f}, "
        f"drift of lambda_max(cov) between T/2 and T = {ref.drift:+.3f}"
    )
    print(
        f"reference |skew|_max = {float(jnp.abs(ref.skewness).max()):.2f}, "
        f"|excess kurtosis|_max = {float(jnp.abs(ref.excess_kurtosis).max()):.2f}"
    )

    # samplers
    sampler_dtype = jnp.float32 if use_float32 else jnp.float64
    df = run_logreg_experiment(
        target,
        make_sampler_specs(sampler_dtype),
        kappa_eff=kappa_eff,
        rel_step_sizes=rel_step_sizes,
        eval_steps=make_eval_steps(cond_numbers=[kappa_eff])[0],
        num_chains=num_chains,
        target_eps=target_eps,
        target_eps_white=target_eps_white,
        use_float32=use_float32,
        seed=42,
    )
    df.to_csv(
        f"logreg_heart_results_{scaling}_e{str(target_eps).replace('.', '')}_ew{str(target_eps_white).replace('.', '')}_f32.csv",
        index=False,
    )

    print(
        f"\nBest step size per configuration (original criterion, eps = {target_eps}):"
    )
    print(
        summarize(df, criterion="grad_evals_to_conv").to_string(
            index=False, float_format=lambda v: f"{v:.3g}"
        )
    )

    print(
        f"\nBest step size per configuration (whitened criterion, eps_white = {target_eps_white}):"
    )
    print(
        summarize(df, criterion="grad_evals_to_conv_white").to_string(
            index=False, float_format=lambda v: f"{v:.3g}"
        )
    )
