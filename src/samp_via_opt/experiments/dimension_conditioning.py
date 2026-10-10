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
import numpy as np
from typing import Any, Callable

from samp_via_opt.experiments.evaluation import (
    build_ensemble_evaluator,
    convergence_table,
    noise_floor,
    slow_direction_init,
    make_eval_steps,
    timed_call,
)
from samp_via_opt.targets.gaussian import GaussianTarget
from samp_via_opt.samplers import ULA, RMULA, ULMC, RMULMC


# sampler factory builds the sampler from the target (and gamma for momentum methods)
# step_size is a placeholder: the evaluator overwrites it via eqx.tree_at
SamplerFactory = Callable[[GaussianTarget, float | None], Any]


def anisotropic_gaussian_cov(dim: int, cond_number: float, key: jax.Array) -> jax.Array:
    """
    Covariance with eigenvalues log-spaced in [1, kappa] (so L = 1, m = 1/kappa).
    Rotate diagonal matrix of specified eigenvalues via QR method.
    Applied rotation doesn't favor any spatial direction (Haar uniform).

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
    Q = Q * jnp.sign(jnp.diag(R))  # Q is Haar uniform
    return Q @ jnp.diag(1.0 / eigenvals) @ Q.T


def run_grid_search(
    dims: list[int],
    cond_numbers: list[float],
    step_sizes: jax.Array,
    gamma_fn: Callable[[float], list[float]],
    sampler_registry: dict[str, tuple[SamplerFactory, bool]],
    eval_steps: np.ndarray,
    target_eps: float = 0.1,
    target_eps_white: float = 0.2,
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
        eval_steps: Step numbers at which to evaluate convergence metrics.
        target_eps: Convergence threshold for the target.
        target_eps_white: Convergence threshold for the target in whitened coordinates.
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
            master_key, cov_key, run_key, floor_key, floor_key_w = jax.random.split(
                master_key, 5
            )

            target = GaussianTarget(
                mean=jnp.zeros(dim), cov=anisotropic_gaussian_cov(dim, kappa, cov_key)
            )
            target_white = GaussianTarget(mean=jnp.zeros(dim), cov=jnp.eye(dim))

            x0 = slow_direction_init(target, n_sigma=init_sigma)

            # scale number of chains with dimension to keep MC noise roughly constant
            # noise^2 = E[W2^2] ~ E[|mu|_2^2] + 1/4 E[|Σ_hat - I|_F^2] ~ (d^2 + 5*d)/(4*C)
            floor_target = 0.3 * target_eps_white  # to be safe, actual target: 0.5*
            num_chains = int(np.ceil((dim**2 + 5 * dim) / (4 * floor_target**2)))

            # normalisation: 1/sqrt(m)
            norm = float(1.0 / jnp.sqrt(target.m))
            floor = float(noise_floor(target, floor_key, num_chains))
            print(
                f"MC noise floor (normalised): {floor / norm:.4f}  vs eps = {target_eps}"
            )

            floor_white = float(noise_floor(target_white, floor_key_w, num_chains))
            print(
                f"whitened floor: {floor_white:.4f}  vs eps_white = {target_eps_white}"
            )

            gammas = gamma_fn(kappa)

            for name, (factory, uses_gamma) in sampler_registry.items():
                for gamma in gammas if uses_gamma else [None]:
                    sampler = factory(target, gamma)
                    eval_fn = build_ensemble_evaluator(sampler, num_chains, eval_steps)
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
                        eval_steps,
                        target_eps,
                        target_eps_white,
                        norm,
                        floor,
                        floor_white,
                        runtime_sec=sec,
                    )
                    tables.append(
                        df.assign(gamma=gamma, kappa=kappa, num_chains=num_chains)
                    )

    return pd.concat(tables, ignore_index=True)


# def best_per_sampler(df: pd.DataFrame, white: bool=False, unit: str="grad_evals"):
#     """
#     Best configuration per (dimension, condition_number, sampler).
#     For samplers that converged: picks the run minimizing parameter `unit`.
#     For samplers that never converged: falls back to the best non-diverged run minimizing `final_w2_norm`.
#     Filtering can be based on whitened metrics if desired.

#     Args:
#         df: DataFrame containing the results of the grid search.
#         white: Use whitened metrics if set to True.
#         unit: Metric to minimize for converged runs.
#               Defaults to steps_to_conv if not set as "grad_evals".

#     Returns:
#         DataFrame containing the best configuration per sampler.
#     """

#     sfx = "_white" if white else ""
#     conv_col = f"converged{sfx}"
#     cost = f"{'grad_evals' if unit == 'grad_evals' else 'steps'}_to_conv{sfx}"
#     fallback = "final_w2_white" if white else "final_w2_norm"
#     keys = ["dimension", "kappa", "sampler"]

#     ok = df[~df["diverged"]]

#     conv = (ok[ok[conv_col]].sort_values(cost)
#             .drop_duplicates(keys).assign(never_converged=False))

#     fb = ok.sort_values(fallback).drop_duplicates(keys)
#     fb = fb.merge(conv[keys], on=keys, how="left", indicator=True)
#     fb = (fb[fb["_merge"] == "left_only"].drop(columns="_merge")
#           .assign(never_converged=True))

#     best = pd.concat([conv, fb], ignore_index=True)

#     # groups that diverged on every step size
#     miss = df[keys].drop_duplicates().merge(best[keys], on=keys, how="left", indicator=True)
#     miss = (miss[miss["_merge"] == "left_only"].drop(columns="_merge")
#             .assign(never_converged=True))
#     best = pd.concat([best, miss], ignore_index=True)

#     h_lo, h_hi = df["step_size"].min(), df["step_size"].max()
#     best["at_boundary"] = (np.isclose(best["step_size"], h_lo)
#                            | np.isclose(best["step_size"], h_hi))
#     return best.sort_values(["dimension", "kappa", cost]).reset_index(drop=True)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    num_steps = 20_000  # fixed budget

    sampler_registry_dim_kappa: dict[str, tuple[SamplerFactory, bool]] = {
        # name: (factory, uses_gamma)
        "ULA": (lambda t, g: ULA(step_size=1.0), False),
        "RMULA": (lambda t, g: RMULA(step_size=1.0), False),
        "ULMC": (lambda t, g: ULMC(step_size=1.0, alpha=g, beta=1.0 / t.L), True),
        "RMULMC": (lambda t, g: RMULMC(step_size=1.0, u=1.0 / t.L), False),
    }

    df_dim_kappa = run_grid_search(
        dims=[2, 10, 50],
        cond_numbers=[10.0, 100.0, 1000.0, 10_000.0],
        # upper stability limit: h=2/L=2, L=1 by construction
        # to reach accuracy eps: T_ULA ~ kappa*log (1 / eps), T_ULMC ~ sqrt(kappa)*log(1 / eps)
        # for kappa=10_000: T_ULA~23_000, T_ULMC~230 -> h_min_ULA = 23_000 / N !<= 2 -> fix N=20_000
        # allow reasonably small step sizes for small kappas
        step_sizes=jnp.logspace(-3, jnp.log10(1.95), 25),
        # optimal gamma (Nesterov): 2*sqrt(m)=2/sqrt(kappa); sqrt(2*L)=sqrt(2) and 2 from sampling theorems
        gamma_fn=lambda kappa: [float(2.0 / jnp.sqrt(kappa)), float(jnp.sqrt(2)), 2.0],
        sampler_registry=sampler_registry_dim_kappa,
        eval_steps=make_eval_steps(num_steps),
        target_eps=0.1,
        target_eps_white=0.2,
        seed=42,
    )
    df_dim_kappa.to_csv("dimension_conditioning_scaling_results.csv", index=False)
