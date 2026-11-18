"""
Real-world illustration: Bayesian logistic regression on the UCI heart disease data (Cleveland, d = 13).

Setting follows the Bayesian experiment of the preconditioning paper: anisotropic Gaussian prior with
linearly interpolated variances in [0.1, 10] on the regression coefficients.

Differences to the main experiments (dimension_conditioning.py):
- Single problem, no sweep over (d, kappa). Same samplers, same step-size / friction grids, same metrics.
- The posterior is not Gaussian and has no closed-form moments. Reference mean / covariance are estimated
  once with a Laplace-preconditioned MALA ensemble. The ensemble evaluator then compares the first two
  moments of the sampler ensemble to these reference moments (Gaussian W2 distance, lower bound on actual W2 which
  is accurate for near-Gaussian posteriors; skewness / kurtosis of the reference are printed as a check).
"""

import math
import urllib.request
from pathlib import Path
from typing import NamedTuple

import blackjax
import equinox as eqx
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
        "minmax":   features in [0, 1]. kappa_eff ~ 3e2, prior still matters. Default, matches the range of
                    the main experiments.
        "standard": z-scores. Posterior is nearly well-conditioned (kappa ~ 10), the prior hardly matters.
        "none":     raw features. kappa ~ 1e6, far out of reach for all samplers without preconditioning.

    Args:
        path: Location of processed.cleveland.data.
        scaling: One of "minmax", "standard", "none".

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
    elif scaling != "none":
        raise ValueError(f"unknown scaling: {scaling}")
    return jnp.asarray(X), jnp.asarray(y)


# ----------------------------------------------------------------------------------
# reference posterior (ground truth for the evaluator)
# ----------------------------------------------------------------------------------


class ReferencePosterior(NamedTuple):
    mean: jax.Array  # (d,)
    cov: jax.Array  # (d, d)
    x_map: jax.Array  # (d,) posterior mode
    hess_map: jax.Array  # (d, d) Hessian of the potential at the mode
    step_size: float  # MALA step size
    accept_rate: float  # MALA acceptance rate
    skewness: jax.Array  # (d,) of the standardised marginals
    excess_kurtosis: jax.Array  # (d,)


def find_map(target: LogisticRegressionTarget, num_iter: int = 50):
    """
    Find x_map = argmin Ψ(x) and the Hessian there. Uses Newton's method with exact Hessian.
    Run 'num_iter' Newton iterations from x=0.
    """

    def newton(x, _):
        g = -target.grad_log_prob(x)
        return x - jnp.linalg.solve(target.hessian_potential(x), g), None

    x_map, _ = jax.lax.scan(newton, jnp.zeros(target.dim), None, length=num_iter)
    return x_map, target.hessian_potential(x_map)


@eqx.filter_jit
def _run_mala(target, key, x_map, chol, num_chains, burn_in, num_samples, step_size):
    """
    Ensemble MALA (using blackJAX) in Laplace-whitened coordinates z, x = x_map + chol @ z (chol chol^T = H_map^{-1}).
        Laplace approximation:
            at the mode ∇Ψ = 0, so by 2nd-order Taylor expansion Ψ(x) ≈ Ψ(x_map) + 1/2 (x-x_map)^T H (x-x_map).
            -> π ≈ N(x_map, H^{-1}) (for large n (Bernstein-von Mises), exact for Gaussian posterior)
        if x ~ N(x_map, H^{-1}) then z ~ N(0, I) (isotropic in whitened coordinates)
        -> fast convergence of MALA in z

    The step size is fixed (no adaptation). The Metropolis correction keeps π invariant for any step size.
    Chains start from the Laplace approximation, run `burn_in` steps, and then `num_samples` steps in which
    first and second moments are accumulated over all chains and all steps.
    """
    dim = target.dim

    def logdensity(z):
        return target.log_prob(x_map + chol @ z)

    mala = blackjax.mala(logdensity, step_size)
    batched_init = jax.vmap(mala.init)
    batched_step = jax.vmap(mala.step)

    def advance(states, key):
        """One MALA step for every chain; also returns the mean acceptance probability."""
        states, info = batched_step(jax.random.split(key, num_chains), states)
        return states, jnp.mean(info.acceptance_rate)

    key_init, key_burn_in, key_sampling = jax.random.split(key, 3)
    z0 = jax.random.normal(
        key_init, (num_chains, dim)
    )  # exact draws from the Laplace approximation
    states = batched_init(z0)

    # burn-in
    states, _ = jax.lax.scan(advance, states, jax.random.split(key_burn_in, burn_in))

    # sampling: accumulate sum_z and sum_zzT over all chains and steps
    def sampling_step(carry, key):
        states, sum_z, sum_zzT = carry
        states, accept_prob = advance(states, key)
        z = states.position
        return (states, sum_z + z.sum(axis=0), sum_zzT + z.T @ z), accept_prob

    init = (states, jnp.zeros(dim, z0.dtype), jnp.zeros((dim, dim), z0.dtype))
    (states, sum_z, sum_zzT), accept_probs = jax.lax.scan(
        sampling_step, init, jax.random.split(key_sampling, num_samples)
    )

    n = num_samples * num_chains
    mean_z = sum_z / n
    cov_z = sum_zzT / n - jnp.outer(mean_z, mean_z)  # E[zzᵀ] - μμᵀ
    return mean_z, cov_z, states.position, jnp.mean(accept_probs)


def reference_posterior(
    target: LogisticRegressionTarget,
    key: jax.Array,
    num_chains: int = 2_000,
    burn_in: int = 1_000,
    num_samples: int = 2_000,
    step_size: float | None = None,
) -> ReferencePosterior:
    """
    Estimate posterior mean and covariance. Plays the role of a long MALA run,
    but is very cheap because of Laplace preconditioning (see _run_mala()).

    Args:
        target: Logistic regression posterior.
        key: PRNGKey.
        num_chains: Number of parallel MALA chains.
        burn_in: Burn-in steps.
        num_samples: Sampling steps per chain after burn-in (moments are accumulated after every step).
        step_size: Fixed MALA step size, proposal z' = z + step_size * ∇log π(z) + sqrt(2 * step_size) * noise.
            Default: Roberts & Rosenthal (1998) optimal scaling for a standard normal target (acceptance ~ 0.57),
            sig^2 = 1.65^2 * d^(-1/3) with proposal variance sig^2 = 2*step_size, i.e. step_size = sig^2 / 2.

    Returns:
        Reference moments and diagnostics.
    """
    if step_size is None:
        step_size = 0.5 * 1.65**2 * target.dim ** (-1.0 / 3.0)

    x_map, H = find_map(target)
    chol = jnp.linalg.cholesky(jnp.linalg.inv(H))
    mean_z, cov_z, z_final, acc = _run_mala(
        target, key, x_map, chol, num_chains, burn_in, num_samples, step_size
    )

    # transform back to original coordinates
    mean = x_map + chol @ mean_z
    cov = chol @ cov_z @ chol.T
    cov = 0.5 * (cov + cov.T)

    # Gaussianity check on the final states of all chains (standardised marginals)
    u = (x_map + z_final @ chol.T - mean) / jnp.sqrt(jnp.diag(cov))
    return ReferencePosterior(
        mean=mean,
        cov=cov,
        x_map=x_map,
        hess_map=H,
        step_size=float(step_size),
        accept_rate=float(acc),
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
    target_eps_white: float = 0.1,
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
        kappa_eff: Effective condition number L / lambda_min(Hessian at the mode). Used for the friction parameter choice.
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
    scaling = "minmax"  # "minmax" | "standard" | "none", see load_heart_disease
    num_chains = 30_000
    target_eps = 0.1  # convergence threshold in original coordinates
    target_eps_white = 0.1  # convergence threshold in whitened coordinates
    rel_step_sizes = jnp.logspace(jnp.log10(0.3), jnp.log10(8.0), 32)

    # problem and reference posterior
    X, y = load_heart_disease(scaling=scaling)
    d = X.shape[1]
    prior_var = jnp.linspace(0.1, 10.0, d)  # linearly interpolated prior variances
    target = LogisticRegressionTarget(X, y, prior_var)

    ref = reference_posterior(target, jax.random.PRNGKey(0))
    target = target.with_moments(ref.mean, ref.cov)

    # kappa_global: worst case over the whole space (theory); kappa_eff: curvature where the mass is
    kappa_global = target.L / target.m
    kappa_eff = target.L / float(jnp.linalg.eigvalsh(ref.hess_map)[0])
    print(f"N = {X.shape[0]}, d = {d}, scaling = {scaling}")
    print(f"kappa_global = {kappa_global:.0f}, kappa_eff = {kappa_eff:.0f}")
    print(
        f"reference MALA (blackJAX): step_size = {ref.step_size:.2f}, acceptance = {ref.accept_rate:.2f}"
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
