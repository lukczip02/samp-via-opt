"""
Experiments to evaluate the effect of dimension d and condition number kappa on sampler performance.

For each (d, kappa) pair, we run a grid search over step sizes (adapted from theoretical results)
and samplers, and record the convergence metrics.
The targets are multivariate Gaussians constructed to have the desired condition number / dimension.
As a baseline a noise floor is computed to determine the Monte Carlo noise level for the given number of chains.
"""

import math
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from typing import Any, Callable
from dataclasses import dataclass
from collections.abc import Sequence

from samp_via_opt.experiments.evaluation import (
    build_ensemble_evaluator,
    convergence_table,
    noise_floor,
    slow_direction_init,
    make_eval_steps,
    timed_call,
)
from samp_via_opt.targets.gaussian import GaussianTarget
from samp_via_opt.samplers import ULA, ULMC, RMULMC


# sampler factory builds the sampler from the target (and gamma for momentum methods)
SamplerFactory = Callable[[GaussianTarget, float | None], Any]


@dataclass(frozen=True)
class SamplerSpec:
    """
    Allows individual fine-tuning of samplers in optimum step_size and gamma regions.

    Attributes:
        factory: (target, gamma) -> sampler. gamma is None for samplers without friction.
        gammas: kappa -> list of friction values to try ([None] for samplers without friction).
        step_unit: gamma -> natural unit of the step size. The grid is step_unit(gamma) * rel_step_sizes.
                   ULMC: stability limit scales with gamma (optimum at h/gamma ~ 0.3-0.8), use gamma.
                   ULA, RMULMC: limit hardly depends on gamma, use 1.0.
    """

    factory: SamplerFactory
    gammas: Callable[[float], Sequence[float | None]]
    step_unit: Callable[[float | None], float]


def diagonal_gaussian_cov(dim: int, cond_number: float) -> jax.Array:
    """
    Create diagonal covariance with eigenvalues log-spaced in [1, kappa] (so L = 1, m = 1/kappa).

    Args:
        dim: Dimension of the Gaussian.
        cond_number: Condition number kappa = L / m.

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
    rel_step_sizes: jax.Array,
    sampler_specs: dict[str, SamplerSpec],
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
        rel_step_sizes: Shared relative step-size window. The step sizes actually run for a sampler are
                        spec.step_unit(gamma) * rel_step_sizes.
        sampler_specs: Dictionary mapping sampler names to their SamplerSpec.
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
            # floor_target = 0.35 * target_eps_white  # to be safe, actual target: 0.5*
            floor_target = 0.7 * target_eps
            # num_chains = int(np.ceil((dim**2 + 5 * dim) / (4 * floor_target**2)))
            num_chains = int(np.ceil((dim + 5) / (4 * floor_target**2)))
            print(f"Number of chains: {num_chains}")

            # normalisation: W2 <= sqrt(d/m) * eps (unwhitenend)
            #                W2 <= sqrt(d) * eps (whitened, independent of m)
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

            for name, spec in sampler_specs.items():
                for gamma in spec.gammas(kappa):
                    sampler = spec.factory(target, gamma)
                    step_sizes = spec.step_unit(gamma) * rel_step_sizes
                    eval_fn = build_ensemble_evaluator(
                        sampler, num_chains, eval_h, use_float32
                    )
                    metrics, sec = timed_call(eval_fn, step_sizes, x0, run_key, target)

                    label = name if gamma is None else f"{name} (gamma={gamma:.3f})"
                    print(f"{label} done in {sec:.2f}s")

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
    use_float32 = True
    sampler_dtype = jnp.float32 if use_float32 else jnp.float64

    # common relative step-size window: 48 log-spaced points on [0.1, 4] (about 8 % spacing)
    # relative to each sampler's natural unit (see SamplerSpec.step_unit)
    # previously (empirically) observed optima / stability limits in these units:
    #   ULA:    h* ~ 0.6, stability limit 2/L = 2 (L = 1 by construction)
    #   ULMC:   h/gamma* ~ 0.5-2.2, convergence cliff at about h/gamma ~ 2.3
    #   RMULMC: h* ~ 0.4-1.2 for gamma = c/sqrt(kappa), cliff at about 0.6-1.9
    # generous margins on both sides of these empirical findings
    rel_step_sizes = jnp.logspace(jnp.log10(0.1), jnp.log10(4.0), 48)

    # step_size is a placeholder: the evaluator overwrites it via eqx.tree_at
    sampler_specs: dict[str, SamplerSpec] = {
        "ULA": SamplerSpec(
            factory=lambda t, g: ULA(step_size=1.0, dtype=sampler_dtype),
            gammas=lambda kappa: 3 * [None],  # use a few runs for fair comparison
            step_unit=lambda g: 1.0,
        ),
        # gamma: 2/sqrt(kappa) (Nesterov-optimal) plus two overdamped-like values sqrt(2.0) and 2.0
        "ULMC": SamplerSpec(
            factory=lambda t, g: ULMC(
                step_size=1.0, alpha=g, beta=1.0 / t.L, dtype=sampler_dtype
            ),
            gammas=lambda kappa: [2.0 / math.sqrt(kappa), math.sqrt(2.0), 2.0],
            step_unit=lambda g: g if g is not None else 1.0,
        ),
        # gamma: c/sqrt(kappa) with c in {1, 2, 4} plus 2.0
        # uses gamma extendend version of RMULMC, gamma = 2.0 is the original Shen & Lee.
        "RMULMC": SamplerSpec(
            factory=lambda t, g: RMULMC(
                step_size=1.0, gamma=g, u=1.0 / t.L, dtype=sampler_dtype
            ),
            gammas=lambda kappa: [c / math.sqrt(kappa) for c in (1.0, 2.0, 4.0)]
            + [2.0],
            step_unit=lambda g: 1.0,
        ),
    }

    cond_numbers = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10_000.0]
    dims = [2, 4, 8, 16]

    df_dim_kappa = run_grid_search(
        dims=dims,
        cond_numbers=cond_numbers,
        rel_step_sizes=rel_step_sizes,
        sampler_specs=sampler_specs,
        eval_steps=make_eval_steps(cond_numbers=cond_numbers),
        target_eps=0.1,
        target_eps_white=0.2,
        use_float32=use_float32,
        seed=42,
    )
    df_dim_kappa.to_csv(
        f"dimension_conditioning_scaling_results_d{dims[0]}-{dims[-1]}_e01_ULA_ULMC_RMULMC_f32.csv",
        index=False,
    )
