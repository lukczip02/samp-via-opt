"""
Ensemble (parallel-chain) evaluation of samplers on Gaussian targets.

Instead of estimating moments from a time window of ONE chain, we run C independent
chains from the same start x0 and estimate the law of X_n across chains at every
checkpoint n. This matches what the theorems bound (W2 of the law of X_n) and removes
burn-in / autocorrelation from the estimator. Only per-checkpoint moments are stored,
never the full trajectory.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd

from samp_via_opt.diagnostics import gaussian_wasserstein_distance

# gradient evaluations per step (cost model for iteration-vs-gradient comparisons)
grads_per_step = {"ULA": 1, "ULMC": 1, "RMULA": 2, "RMULMC": 2, "TIPreL": 1}


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def slow_direction_init(target: Any, n_sigma: float = 3.0) -> jax.Array:
    """
    Deterministic start n_sigma standard deviations out along the slowest
    (largest-variance) eigen-direction of the target covariance.

    Args:
        target: Target distribution with .mean and .cov attributes.
        n_sigma: Number of standard deviations to move along the slowest direction.

    Returns:
        Initial point x0 for the sampler.
    """
    evals, evecs = jnp.linalg.eigh(target.cov)
    return target.mean + n_sigma * jnp.sqrt(evals[-1]) * evecs[:, -1]


def _moments(xs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Calculate mean and unbiased covariance of an ensemble of samples (i.e. from multiple chains).

    Args:
        xs (C, d): Ensemble of samples.

    Returns:
        Mean (d,) and unbiased covariance (d, d) of the ensemble.
    """
    mean = jnp.mean(xs, axis=0)
    diff = xs - mean
    cov = (diff.T @ diff) / (xs.shape[0] - 1)
    return mean, cov


def _gelbrich(mean: jax.Array, cov: jax.Array, target: Any) -> jax.Array:
    """
    Gelbrich (moment-based) lower bound on W2, equals W2 iff both laws are Gaussian.
    Returns +inf for non-finite moments so threshold comparisons are simply False.

    Args:
        mean (d,): Mean of the sample ensemble.
        cov (d, d): Covariance of the sample ensemble.
        target: Target distribution with .mean and .cov attributes.

    Returns:
        Scalar Gelbrich distance between the sample ensemble and the target.
    """
    finite = jnp.all(jnp.isfinite(mean)) & jnp.all(jnp.isfinite(cov))
    safe_mean = jnp.where(finite, mean, 0.0)
    safe_cov = jnp.where(finite, cov, jnp.eye(cov.shape[0], dtype=cov.dtype))
    w2 = gaussian_wasserstein_distance(safe_mean, safe_cov, target.mean, target.cov)
    return jnp.where(finite, w2, jnp.inf)


def noise_floor(
    target: Any, key: jax.Array, num_chains: int, num_reps: int = 16
) -> jax.Array:
    """
    Monte-Carlo floor of the estimator: mean Gelbrich distance between C EXACT
    target samples and the target. A sampler cannot look better than this.

    Args:
        target: Target distribution with .mean and .cov attributes.
        key: PRNGKey for sampling.
        num_chains: Number of independent chains that are run.
        num_reps: Number of repetitions to estimate the mean floor.

    Returns:
        Scalar Monte-Carlo noise floor of the Gelbrich distance estimator.
    """

    chol = jnp.linalg.cholesky(target.cov)
    d = target.mean.shape[0]

    def single(k):
        z = jax.random.normal(k, (num_chains, d))
        xs = target.mean + z @ chol.T
        mean, cov = _moments(xs)
        return _gelbrich(mean, cov, target)

    return jnp.mean(jax.vmap(single)(jax.random.split(key, num_reps)))


# ----------------------------------------------------------------------------------
# core: one step size, C chains
# ----------------------------------------------------------------------------------
def run_ensemble(
    sampler: Any,
    target: Any,
    x0: jax.Array,
    step_size: jax.Array,
    key: jax.Array,
    num_chains: int,
    num_steps: int,
    num_eval_points: int,
) -> dict[str, jax.Array]:
    """
    Run C independent chains of a sampler for a given step size, and evaluate convergence metrics at checkpoints.

    Args:
        sampler: Sampler instance with .init_state and .step methods.
        target: Target distribution with .mean and .cov attributes.
        x0 (d,): Initial point for all chains.
        step_size: Step size for the sampler.
        key: PRNGKey for random number generation.
        num_chains: Number of independent chains to run in parallel.
        num_steps: Total number of steps to run for each chain.
        num_eval_points: Number of checkpoints at which to evaluate convergence metrics.

    Returns:
        Dictionary containing arrays of convergence metrics at each checkpoint.
    """

    steps_per_eval_block = num_steps // num_eval_points
    d = x0.shape[0]

    # only h changes, all other hyperparameters are kept as constructed
    sampler = eqx.tree_at(lambda s: s.step_size, sampler, step_size)

    key_init, key_run = jax.random.split(key)
    x0s = jnp.broadcast_to(x0, (num_chains, d))  # same initial point for all chains
    init_keys = jax.random.split(key_init, num_chains)

    # initialize all chains in parallel
    state = jax.vmap(lambda x, k: sampler.init_state(x, key=k))(x0s, init_keys)

    # scan over chains for one time step, return only the new state (discard aux info)
    vstep = jax.vmap(lambda s, k: sampler.step(s, target, k)[0])

    def map_chains(st, k):
        return vstep(st, jax.random.split(k, num_chains)), None

    # scan over all time steps in one evaluation block
    def map_eval_blocks(st, k):
        st, _ = jax.lax.scan(map_chains, st, jax.random.split(k, steps_per_eval_block))
        return st, _moments(st.x)

    # scan over evaluation blocks
    _, (means, covs) = jax.lax.scan(
        map_eval_blocks, state, jax.random.split(key_run, num_eval_points)
    )

    w2 = jax.vmap(lambda mean, cov: _gelbrich(mean, cov, target))(means, covs)
    mean_err = jnp.linalg.norm(means - target.mean, axis=-1)
    cov_err = jnp.linalg.norm(covs - target.cov, ord="fro", axis=(-2, -1))
    return {
        "w2_gelbrich": w2,  # (n_eval,)
        "mean_error": mean_err,  # (n_eval,)
        "cov_error": cov_err,  # (n_eval,)
    }


def build_ensemble_evaluator(
    sampler: Any, num_chains: int, num_steps: int, num_eval_points: int = 50
):
    """
    Returns jitted eval_fn(step_sizes, x0, key, target) -> dict of arrays (n_h, n_eval).
    Step sizes are vmapped, chains are vmapped inside. Independent keys per step size.

    Args:
        sampler: Sampler instance with .init_state and .step methods.
        num_chains: Number of independent chains to run in parallel.
        num_steps: Total number of steps to run for each chain.
        num_eval_points: Number of checkpoints at which to evaluate convergence metrics.

    Returns:
        Jitted function that evaluates the sampler for a given set of step sizes, initial point, random key, and target.
    """

    def eval_fn(step_sizes, x0, key, target):
        keys = jax.random.split(key, step_sizes.shape[0])
        return jax.vmap(
            lambda h, k: run_ensemble(
                sampler, target, x0, h, k, num_chains, num_steps, num_eval_points
            )
        )(step_sizes, keys)

    return eqx.filter_jit(eval_fn)


def timed_call(eval_fn, *args) -> tuple[dict, float]:
    """
    Warm-up (compile) once, then time a second call.

    Args:
        eval_fn: Function to evaluate.
        *args: Arguments to pass to eval_fn.

    Returns:
        Tuple of (output of eval_fn, time taken in seconds).
    """

    import time

    jax.block_until_ready(eval_fn(*args))
    t0 = time.perf_counter()
    out = jax.block_until_ready(eval_fn(*args))
    return out, time.perf_counter() - t0


# ----------------------------------------------------------------------------------
# post-processing
# ----------------------------------------------------------------------------------
def convergence_table(
    metrics: dict[str, jax.Array],
    step_sizes: jax.Array,
    sampler_name: str,
    dim: int,
    num_steps: int,
    eps: float,
    norm: float,
    floor: float,
    runtime_sec: float | None = None,
) -> pd.DataFrame:
    """
    Construct a DataFrame summarizing convergence metrics for a given sampler and step sizes.

    Args:
        metrics: Convergence metrics at each checkpoint (as returned by build_ensemble_evaluator).
        step_sizes: Array of step sizes evaluated.
        sampler_name: Name of the sampler.
        dim: Dimension of the target distribution.
        num_steps: Total number of steps run for each chain.
        eps: Convergence threshold for the target.
        norm: Normalization factor for W2 distance.
        floor: Monte Carlo noise floor for the estimator.
        runtime_sec: Optional runtime in seconds for the evaluation.

    Returns:
        DataFrame summarizing convergence metrics for each step size.
    """

    w2n = metrics["w2_gelbrich"] / norm  # (n_h, n_eval)
    n_h, n_eval = w2n.shape
    eval_steps = (num_steps // n_eval) * jnp.arange(1, n_eval + 1)

    # 'converged' means normalised W2 <= eps at some checkpoint and
    # stays below at all later checkpoints
    below = (w2n <= eps).astype(jnp.int32)
    stays = jnp.flip(jnp.cumprod(jnp.flip(below, -1), -1), -1).astype(bool)
    converged = stays.any(-1)
    steps_to_conv = jnp.where(converged, eval_steps[jnp.argmax(stays, -1)], -1)

    grads = grads_per_step.get(sampler_name, 1)
    return pd.DataFrame(
        {
            "sampler": sampler_name,
            "dimension": dim,
            "step_size": jnp.asarray(step_sizes).tolist(),
            "converged": converged.tolist(),
            "steps_to_conv": steps_to_conv.tolist(),
            "grad_evals_to_conv": jnp.where(
                converged, steps_to_conv * grads, -1
            ).tolist(),
            "final_w2_norm": w2n[:, -1].tolist(),
            "final_mean_error": metrics["mean_error"][
                :, -1
            ].tolist(),  # only use final checkpoint
            "final_cov_error": metrics["cov_error"][:, -1].tolist(),
            "diverged": (~jnp.isfinite(metrics["w2_gelbrich"][:, -1])).tolist(),
            "floor_norm": float(floor) / norm,
            "eps": eps,
            "floor_ok": float(floor) / norm
            < 0.5 * eps,  # is eps resolvable with C chains?
            "runtime_sec_total": runtime_sec,
        }
    )
