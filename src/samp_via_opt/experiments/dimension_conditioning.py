"""
Experiments to evaluate the effect of dimension d and condition number kappa on sampler performance.

For each (d, kappa) pair, we run a grid search over step sizes (adapted from theoretical results)
and samplers, and record the convergence metrics.
The targets are multivariate Gaussians constructed to have the desired condition number / dimension.
As a baseline a noise floor is computed to determine the Monte Carlo noise level for the given number of chains.
"""

import jax
import jax.numpy as jnp
import pandas as pd
from typing import Any, Callable

from samp_via_opt.experiments.evaluation import (
    build_ensemble_evaluator,
    convergence_table,
    noise_floor,
    slow_direction_init,
    timed_call,
)
from samp_via_opt.targets.gaussian import GaussianTarget
from samp_via_opt.samplers import ULA, RMULA, ULMC, RMULMC, TIPreL


# sampler factory builds the sampler from the target (and gamma for momentum methods)
# step_size is a placeholder: the evaluator overwrites it via eqx.tree_at
SamplerFactory = Callable[[GaussianTarget, float | None], Any]


def anisotropic_gaussian_cov(dim: int, cond_number: float, key: jax.Array) -> jax.Array:
    """
    Covariance with eigenvalues log-spaced in [1, kappa] (so L = 1, m = 1/kappa)
    and Haar-random orientation.

    Args:
        dim: Dimension of the Gaussian.
        cond_number: Condition number kappa = L / m.
        key: PRNG key for random number generation.

    Returns:
        Covariance matrix of shape (dim, dim).
    """
    eigenvals = jnp.logspace(0, -jnp.log10(cond_number), num=dim)
    A = jax.random.normal(key, shape=(dim, dim))
    Q, R = jnp.linalg.qr(A)
    Q = Q * jnp.sign(jnp.diag(R))
    return Q @ jnp.diag(1.0 / eigenvals) @ Q.T


def run_grid_search(
    dims: list[int],
    cond_numbers: list[float],
    step_sizes: jax.Array,
    gamma_fn: Callable[[float], list[float]],
    sampler_registry: dict[str, tuple[SamplerFactory, bool]],
    target_epsilon: float = 0.1,
    num_chains: int = 1000,
    num_steps: int = 100_000,
    init_sigma: float = 3.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Runs a grid search over dimensions, condition numbers, step sizes, and sampler configurations.

    Args:
        dims: List of dimensions to test.
        cond_numbers: List of condition numbers to test.
        step_sizes: Array of step sizes to evaluate.
        gamma_fn: Function that returns a list of gamma values for a given condition number.
        sampler_registry: Dictionary mapping sampler names to (factory, uses_gamma) tuples.
        target_epsilon: Convergence threshold for the target.
        num_chains: Number of chains to run in parallel.
        num_steps: Number of steps to run for each chain.
        init_sigma: Initial standard deviation for the starting point.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame containing the results of the grid search.
    """

    master_key = jax.random.PRNGKey(seed)
    tables = []

    for dim in dims:
        for kappa in cond_numbers:
            print(f"\n--- d = {dim}, kappa = {kappa} ---")
            master_key, cov_key, run_key, floor_key = jax.random.split(master_key, 4)

            target = GaussianTarget(
                mean=jnp.zeros(dim), cov=anisotropic_gaussian_cov(dim, kappa, cov_key)
            )
            x0 = slow_direction_init(target, n_sigma=init_sigma)

            # scale number of chains with dimension to keep MC noise roughly constant
            chains_for_dim = num_chains * dim // 2

            # normalisation: 1 / sqrt(m)
            norm = float(1.0 / jnp.sqrt(target.m))
            floor = float(noise_floor(target, floor_key, chains_for_dim))
            print(
                f"MC noise floor (normalised): {floor / norm:.4f}  vs eps = {target_epsilon}"
            )

            gammas = gamma_fn(kappa)

            for name, (factory, uses_gamma) in sampler_registry.items():
                for gamma in gammas if uses_gamma else [None]:
                    sampler = factory(target, gamma)
                    eval_fn = build_ensemble_evaluator(
                        sampler, chains_for_dim, num_steps
                    )
                    metrics, sec = timed_call(eval_fn, step_sizes, x0, run_key, target)

                    if gamma is None:
                        print(f"{name} done in {sec:.2f}s")
                    else:
                        print(f"{name} (gamma={gamma:.3f}) done in {sec:.2f}s")

                    df = convergence_table(
                        metrics,
                        step_sizes,
                        name,
                        dim,
                        num_steps,
                        target_epsilon,
                        norm,
                        floor,
                        runtime_sec=sec,
                    )
                    tables.append(
                        df.assign(gamma=gamma, kappa=kappa, num_chains=chains_for_dim)
                    )

    return pd.concat(tables, ignore_index=True)


def best_per_sampler(
    df: pd.DataFrame,
    cost: str = "grad_evals_to_conv",
    fallback_metric: str = "final_w2_norm",
) -> pd.DataFrame:
    """
    Best configuration per (dimension, condition_number, sampler).
    For samplers that converged: picks the run minimizing parameter `cost`.
    For samplers that never converged: falls back to the best non-diverged run minimizing `fallback_metric`.

    Args:
        df: DataFrame containing the results of the grid search.
        cost: Metric to minimize for converged runs.
        fallback_metric: Metric to minimize for non-diverged runs.

    Returns:
        DataFrame containing the best configuration per sampler.
    """

    group_cols = ["dimension", "kappa", "sampler"]
    all_keys = df[group_cols].drop_duplicates().set_index(group_cols)

    # best converged configuration
    conv = (
        df[df["converged"] & ~df["diverged"]]
        .sort_values(cost)
        .groupby(group_cols)
        .first()
        .assign(never_converged=False)
    )

    # best non-diverged fallback
    non_div = (
        df[~df["diverged"]]
        .sort_values(fallback_metric)
        .groupby(group_cols)
        .first()
        .assign(never_converged=True)
    )

    # combine: keep `conv` rows first, fill missing index entries from `non_div`, then `all_keys`
    best = conv.combine_first(non_div).combine_first(all_keys).reset_index()

    # handle edge case where a sampler diverged on every single run
    best["never_converged"] = best["never_converged"].fillna(True)

    return best.sort_values(["dimension", "kappa", cost]).reset_index(drop=True)


if __name__ == "__main__":
    # jax.config.update("jax_enable_x64", True)   # enable for the final runs

    cond_number = 100.0
    num_steps = 10000

    sampler_registry: dict[str, tuple[SamplerFactory, bool]] = {
        # name: (factory, uses_gamma)
        "ULA": (lambda t, g: ULA(step_size=1.0), False),
        "RMULA": (lambda t, g: RMULA(step_size=1.0), False),
        "ULMC": (lambda t, g: ULMC(step_size=1.0, alpha=g, beta=1.0 / t.L), True),
        "RMULMC": (lambda t, g: RMULMC(step_size=1.0, u=1.0 / t.L), False),
        "TIPreL": (
            lambda t, g: TIPreL(step_size=1.0, num_steps=num_steps, emp_cov=t.cov),
            False,
        ),
    }

    df = run_grid_search(
        dims=[2, 10],
        cond_numbers=[10.0, 100.0],
        step_sizes=jnp.logspace(-4, -0.5, 10),
        gamma_fn=lambda kappa: [float(2.0 / jnp.sqrt(kappa)), float(jnp.sqrt(2)), 2.0],
        sampler_registry=sampler_registry,
        target_epsilon=0.1,
        num_chains=1,
        num_steps=num_steps,
        seed=42,
    )
    df.to_csv("dimension_conditioning_scaling_results.csv", index=False)

    print("\n--- Best configuration per sampler (cost = gradient evaluations) ---")
    print(best_per_sampler(df))
    print("\n--- Best configuration per sampler (cost = iterations) ---")
    print(best_per_sampler(df, cost="steps_to_conv"))
