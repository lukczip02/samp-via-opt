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


def diagonal_gaussian_cov(dim: int, cond_number: float) -> jax.Array:
    """
    Create diagonal covariance with eigenvalues log-spaced in [1, kappa] (so L = 1, m = 1/kappa).

    Args:
        dim: Dimension of the Gaussian.
        cond_number: Condition number kappa = L / m.
        key: PRNG key for random number generation.

    Returns:
        Covariance matrix of shape (dim, dim).
    """
    eigenvals = jnp.logspace(0, -jnp.log10(cond_number), num=dim)
    return jnp.diag(1.0 / eigenvals)


def anisotropic_gaussian_cov(dim: int, cond_number: float, key: jax.Array) -> jax.Array:
    """
    Create covariance with eigenvalues log-spaced in [1, kappa] (so L = 1, m = 1/kappa).
    Rotate diagonal matrix of specified eigenvalues via QR method.
    Applied rotation doesn't favour any spatial direction (Haar uniform).

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


def clustered_bimodal_gaussian_cov(dim: int, cond_number: float) -> jax.Array:
    """
    Create covariance with eigenvalues in {1, kappa} (so L = 1, m = 1/kappa).
    Eigenvalues are equally distributed at two modes: l_min = 1 and l_max = kappa.

    Args:
        dim: Dimension of the Gaussian.
        cond_number: Condition number kappa = L / m.

    Returns:
        Covariance matrix of shape (dim, dim).
    """
    n = dim // 2

    # ensure at least one dimension is at each extreme if dim >= 2
    n = max(1, min(n, dim - 1)) if dim > 1 else 1

    eigenvals = jnp.concatenate([cond_number * jnp.ones(n), jnp.ones(dim - n)])

    return jnp.diag(eigenvals)


def run_grid_search(
    dims: list[int],
    cond_numbers: list[float],
    step_sizes: jax.Array,
    gamma_fn: Callable[[float], list[float]],
    sampler_registry: dict[str, tuple[SamplerFactory, bool]],
    eval_steps: list[np.ndarray],
    target_eps: float = 0.1,
    target_eps_white: float = 0.2,
    init_sigma: float = 3.0,
    use_float32: bool = False,
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
        eval_steps: Step numbers at which to evaluate convergence metrics. Unique for each kappa.
        target_eps: Convergence threshold for the target.
        target_eps_white: Convergence threshold for the target in whitened coordinates.
        num_steps: Number of steps to run for each chain.
        init_sigma: Initial standard deviation for the starting point.
        use_float32: To improve performance, chains can be run in float32 precision.
                     Numerically critical computations still use float64.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame containing the results of the grid search.
    """

    master_key = jax.random.PRNGKey(seed)
    tables = []

    assert len(eval_steps) == len(cond_numbers)

    for dim in dims:
        for kappa, eval_h in zip(cond_numbers, eval_steps):
            print(f"\n--- d = {dim}, kappa = {kappa} ---")
            master_key, cov_key, run_key, floor_key, floor_key_w = jax.random.split(
                master_key, 5
            )

            target = GaussianTarget(
                mean=jnp.zeros(dim),
                cov=clustered_bimodal_gaussian_cov(dim, kappa),
                diag=True,
            )
            target_white = GaussianTarget(
                mean=jnp.zeros(dim), cov=jnp.eye(dim), diag=True
            )

            x0 = slow_direction_init(target, n_sigma=init_sigma)

            # scale number of chains with dimension to keep MC noise roughly constant
            # noise^2 = E[W2^2] ~ E[|mu|_2^2] + 1/4 E[|Σ_hat - I|_F^2] ~ (d^2 + 5*d)/(4*C)
            floor_target = 0.35 * target_eps_white  # to be safe, actual target: 0.5*
            # num_chains = int(np.ceil((dim**2 + 5 * dim) / (4 * floor_target**2)))
            num_chains = int(np.ceil((dim + 5) / (4 * floor_target**2)))
            print(f"Number of chains: {num_chains}")

            # normalisation: sqrt(d/m)
            norm = float(jnp.sqrt(dim / target.m))
            norm_white = float(jnp.sqrt(dim))
            floor = float(noise_floor(target, floor_key, num_chains))
            print(
                f"MC noise floor (normalised): {floor / norm:.4f}  vs eps = {target_eps}"
            )

            floor_white = float(noise_floor(target_white, floor_key_w, num_chains))
            print(
                f"whitened floor: {floor_white / norm_white:.4f}  vs eps_white = {target_eps_white}"
            )

            gammas = gamma_fn(kappa)

            for name, (factory, uses_gamma) in sampler_registry.items():
                for gamma in gammas if uses_gamma else [None] * len(gammas):
                    sampler = factory(target, gamma)
                    eval_fn = build_ensemble_evaluator(
                        sampler, num_chains, eval_h, use_float32
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
                        eval_h,
                        target_eps,
                        target_eps_white,
                        norm,
                        norm_white,
                        floor,
                        floor_white,
                        runtime_sec=sec,
                    )
                    tables.append(
                        df.assign(gamma=gamma, kappa=kappa, num_chains=num_chains)
                    )

    return pd.concat(tables, ignore_index=True)


if __name__ == "__main__":
    # by default use float64 precision
    jax.config.update("jax_enable_x64", True)

    # option to use float32 precision for most of the computations inside of the chains
    # numerically critical computations are still kept in float64
    use_float32 = False
    sampler_dtype = jnp.float32 if use_float32 else jnp.float64

    sampler_registry_dim_kappa: dict[str, tuple[SamplerFactory, bool]] = {
        # name: (factory, uses_gamma)
        "ULA": (lambda t, g: ULA(step_size=1.0, dtype=sampler_dtype), False),
        "RMULA": (lambda t, g: RMULA(step_size=1.0, dtype=sampler_dtype), False),
        "ULMC": (
            lambda t, g: ULMC(
                step_size=1.0, alpha=g, beta=1.0 / t.L, dtype=sampler_dtype
            ),
            True,
        ),
        "RMULMC": (lambda t, g: RMULMC(step_size=1.0, u=1.0 / t.L), False),
    }

    cond_numbers = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10_000.0]

    df_dim_kappa = run_grid_search(
        dims=[2, 10, 50],
        cond_numbers=cond_numbers,
        # upper stability limit: h=2/L=2, L=1 by construction
        # allow reasonably small step sizes for small kappas
        # to reach accuracy eps: T_ULA ~ kappa*log (1 / eps), T_ULMC (in theory) ~ sqrt(kappa)*log(1 / eps)
        # T_ULA ~ 2.3*kappa -> h_min_ULA = 2.3*kappa / N !<= 2 -> choose at least N ~ 2*kappa
        step_sizes=jnp.logspace(-3, jnp.log10(1.95), 40),
        # optimal gamma (Nesterov): 2*sqrt(m)=2/sqrt(kappa), also allow 0.5* and 2.0* that theoretical value
        # sqrt(2*L)=sqrt(2) and 2 from ULMC sampling theorems
        gamma_fn=lambda kappa: [
            float(1.0 / jnp.sqrt(kappa)),
            float(2.0 / jnp.sqrt(kappa)),
            float(4.0 / jnp.sqrt(kappa)),
            float(jnp.sqrt(2)),
            2.0,
        ],
        sampler_registry=sampler_registry_dim_kappa,
        eval_steps=make_eval_steps(cond_numbers=cond_numbers),
        target_eps=0.1,
        target_eps_white=0.2,
        use_float32=use_float32,
        seed=42,
    )
    df_dim_kappa.to_csv(
        "dimension_conditioning_scaling_results_d2__d10_e01_f64.csv", index=False
    )
