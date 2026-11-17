"""
Real-world illustration: Bayesian logistic regression on the UCI heart disease data (Cleveland, d = 13).

Setting follows the Bayesian experiment of the preconditioning paper: anisotropic Gaussian prior with
linearly interpolated variances in [0.1, 10] on the regression coefficients.

Differences to the main experiments (dimension_conditioning.py):
- Single problem, no sweep over (d, kappa). Same samplers, same step-size / friction grids, same metrics.
- The posterior is not Gaussian and has no closed-form moments. Reference mean / covariance are estimated
  once with a Laplace-preconditioned MALA ensemble. The ensemble evaluator then compares the first two
  moments of the sampler ensemble to these reference moments (Frechet distance, a lower bound on W2 which
  is accurate for near-Gaussian posteriors; skewness / kurtosis of the reference are printed as a check).
- The whitened metric (error in units of posterior standard deviations) is the primary criterion.
"""

import math
import urllib.request
from pathlib import Path
from typing import NamedTuple

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
    step_size: float  # tuned MALA step size (in Laplace-whitened coordinates)
    accept_rate: float
    skewness: jax.Array  # (d,) of the standardised marginals
    excess_kurtosis: jax.Array  # (d,)


def find_map(target: LogisticRegressionTarget, num_iter: int = 50):
    """Newton iterations for the posterior mode (strongly convex, so this is safe from x = 0)."""

    def newton(x, _):
        g = -target.grad_log_prob(x)
        return x - jnp.linalg.solve(target.hessian_potential(x), g), None

    x_map, _ = jax.lax.scan(newton, jnp.zeros(target.dim), None, length=num_iter)
    return x_map, target.hessian_potential(x_map)


@eqx.filter_jit
def _run_mala(
    target, key, x_map, chol, num_chains, burn_in, num_samples, thin, target_acc
):
    """
    Ensemble MALA in Laplace-whitened coordinates z, x = x_map + chol @ z (chol chol^T = H_map^{-1}).
    Step size is adapted (Robbins-Monro on the mean acceptance probability) during burn-in and then frozen.
    """
    d = target.dim

    def logp_grad(z):
        x = x_map + chol @ z
        return target.log_prob(x), chol.T @ target.grad_log_prob(x)

    v_logp_grad = jax.vmap(logp_grad)

    def step(z, lp, g, h, key):
        k_n, k_u = jax.random.split(key)
        xi = jax.random.normal(k_n, z.shape, z.dtype)
        z_new = z + 0.5 * h * g + jnp.sqrt(h) * xi
        lp_new, g_new = v_logp_grad(z_new)
        log_q_fwd = -0.5 * jnp.sum(xi**2, -1)
        log_q_bwd = -0.5 / h * jnp.sum((z - z_new - 0.5 * h * g_new) ** 2, -1)
        log_alpha = lp_new - lp + log_q_bwd - log_q_fwd
        log_alpha = jnp.where(jnp.isfinite(log_alpha), log_alpha, -jnp.inf)
        accept = jnp.log(jax.random.uniform(k_u, lp.shape)) < log_alpha
        z = jnp.where(accept[:, None], z_new, z)
        lp = jnp.where(accept, lp_new, lp)
        g = jnp.where(accept[:, None], g_new, g)
        return z, lp, g, jnp.mean(jnp.exp(jnp.minimum(log_alpha, 0.0)))

    key_init, key_burn, key_samp = jax.random.split(key, 3)
    z = jax.random.normal(
        key_init, (num_chains, d)
    )  # start from the Laplace approximation
    lp, g = v_logp_grad(z)

    # burn-in with step-size adaptation
    def burn(carry, t_key):
        z, lp, g, log_h = carry
        t, k = t_key
        z, lp, g, acc = step(z, lp, g, jnp.exp(log_h), k)
        log_h = log_h + 2.0 / (1.0 + t) ** 0.6 * (acc - target_acc)
        return (z, lp, g, log_h), None

    (z, lp, g, log_h), _ = jax.lax.scan(
        burn,
        (z, lp, g, jnp.zeros(())),
        (jnp.arange(burn_in, dtype=z.dtype), jax.random.split(key_burn, burn_in)),
    )
    h = jnp.exp(log_h)

    # sampling with frozen step size: accumulate first and second moments over chains and (thinned) time
    def record(carry, k):
        def inner(i, c):
            z, lp, g, acc = c
            z, lp, g, a = step(z, lp, g, h, jax.random.fold_in(k, i))
            return z, lp, g, acc + a

        z, lp, g, s1, s2, acc_sum = carry
        z, lp, g, acc = jax.lax.fori_loop(0, thin, inner, (z, lp, g, jnp.zeros(())))
        return (z, lp, g, s1 + z.sum(0), s2 + z.T @ z, acc_sum + acc / thin), None

    n_rec = num_samples // thin
    init = (z, lp, g, jnp.zeros(d), jnp.zeros((d, d)), jnp.zeros(()))
    (z, _, _, s1, s2, acc_sum), _ = jax.lax.scan(
        record, init, jax.random.split(key_samp, n_rec)
    )
    n = n_rec * num_chains
    mean_z = s1 / n
    cov_z = s2 / n - jnp.outer(mean_z, mean_z)
    return mean_z, cov_z, z, h, acc_sum / n_rec


def reference_posterior(
    target: LogisticRegressionTarget,
    key: jax.Array,
    num_chains: int = 2_000,
    burn_in: int = 1_000,
    num_samples: int = 2_000,
    thin: int = 4,
    target_acc: float = 0.55,
) -> ReferencePosterior:
    """
    Estimate posterior mean and covariance. Plays the role of the paper's long MALA run (5e5 steps x 1e4 chains),
    but is much cheaper thanks to Laplace preconditioning: in the whitened coordinates the posterior is almost
    isotropic and MALA mixes within a few steps.

    Args:
        target: Logistic regression posterior.
        key: PRNGKey.
        num_chains: Number of parallel MALA chains.
        burn_in: Burn-in steps (also used to tune the step size to `target_acc`).
        num_samples: Sampling steps per chain after burn-in.
        thin: Moments are accumulated every `thin` steps.
        target_acc: Target acceptance rate (0.574 is the asymptotic optimum, paper uses 0.5).

    Returns:
        Reference moments and diagnostics.
    """
    x_map, H = find_map(target)
    chol = jnp.linalg.cholesky(jnp.linalg.inv(H))
    mean_z, cov_z, z_final, h, acc = _run_mala(
        target, key, x_map, chol, num_chains, burn_in, num_samples, thin, target_acc
    )

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
        step_size=float(h),
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
        kappa_eff: Effective condition number L / lambda_min(Hessian at the mode). Used for the friction grids.
        rel_step_sizes: Shared relative step-size window, see SamplerSpec.step_unit.
        eval_steps: Step numbers at which to evaluate the metrics.
        num_chains: Number of parallel chains of the evaluated samplers.
        target_eps: Convergence threshold (tolerance-normalised W2).
        target_eps_white: Convergence threshold in whitened coordinates (primary criterion).
        init_sigma: Start x0 is `init_sigma` posterior std devs out along the slowest direction.
        use_float32: Run chains in float32, metrics in float64.
        seed: Random seed.

    Returns:
        DataFrame in the format of convergence_table, plus gamma, kappa_eff, num_chains.
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


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Best step size per (sampler, gamma) w.r.t. gradient evaluations to the whitened criterion."""
    rows = []
    for (name, gamma), g in df.groupby(["sampler", "gamma"], dropna=False, sort=False):
        r = g.loc[g["grad_evals_to_conv_white"].idxmin()]
        rows.append(
            {
                "sampler": name,
                "gamma": gamma,
                "best_step": r["step_size"],
                "grad_evals_white": r["grad_evals_to_conv_white"],
                "grad_evals_norm": r["grad_evals_to_conv"],
                "final_mean_err_white": r["final_mean_err_white"],
                "final_cov_err_white": r["final_cov_err_white"],
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
    num_chains = 10_000  # as in the paper; whitened floor ~ 0.08 < 0.5 * eps_white
    rel_step_sizes = jnp.logspace(
        jnp.log10(0.1), jnp.log10(4.0), 24
    )  # even number needed by the evaluator

    # --- problem and reference posterior ---
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
    print(f"reference MALA: step {ref.step_size:.2f}, acceptance {ref.accept_rate:.2f}")
    print(
        f"reference |skew| max = {float(jnp.abs(ref.skewness).max()):.2f}, "
        f"|excess kurtosis| max = {float(jnp.abs(ref.excess_kurtosis).max()):.2f} (0 for a Gaussian)"
    )

    # --- samplers ---
    sampler_dtype = jnp.float32 if use_float32 else jnp.float64
    df = run_logreg_experiment(
        target,
        make_sampler_specs(sampler_dtype),
        kappa_eff=kappa_eff,
        rel_step_sizes=rel_step_sizes,
        eval_steps=make_eval_steps(cond_numbers=[kappa_eff])[0],
        num_chains=num_chains,
        use_float32=use_float32,
        seed=42,
    )
    df.to_csv(f"logreg_heart_results_{scaling}_f32.csv", index=False)

    print("\nBest step size per configuration (whitened criterion, eps_white = 0.2):")
    print(summarize(df).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
