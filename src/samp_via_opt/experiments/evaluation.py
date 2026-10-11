"""
Ensemble (parallel-chain) evaluation of samplers on Gaussian targets.

Instead of estimating moments from a time window of ONE chain, we run C independent
chains from the same start x0 and estimate the law of X_n across chains at every
checkpoint n. This matches what the theorems bound (W2 of the law of X_n) and removes
burn-in / autocorrelation from the estimator. Only per-checkpoint moments are stored,
never the full trajectory.
"""

from typing import Any

import numpy as np
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


def make_eval_steps(
    num_steps: int, num_eval_points: int = 100, first: int = 20
) -> np.ndarray:
    """
    Log-spaced, strictly increasing checkpoint indices in [first, num_steps].

    Args:
        num_steps: Step budget of the sampler chain.
        num_eval_points: Number of desired checkpoints / evaluation points.
        first: Step number correspoding to first checkpoint.

    Returns:
        Log-spaced checkpoint indices.
    """
    steps = np.geomspace(first, num_steps, num_eval_points)
    return np.unique(np.round(steps).astype(np.int64))


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


def _gaussian_W2(
    mean: jax.Array, cov: jax.Array, target_mean: jax.Array, target_cov: jax.Array
) -> jax.Array:
    """
    Moment-based lower bound on W2, equals W2 iff both laws are Gaussian.
    Sometimes referred to as Frechet distance.
    Returns +inf for non-finite moments so threshold comparisons are simply False.

    Args:
        mean (d,): Mean of the sample ensemble.
        cov (d, d): Covariance of the sample ensemble.
        target_mean (d,): Mean of the target distribution.
        target_cov (d, d): Covariance of the target distribution.

    Returns:
        W2 distance between the sample ensemble and the target.
    """
    finite = jnp.all(jnp.isfinite(mean)) & jnp.all(jnp.isfinite(cov))
    safe_mean = jnp.where(finite, mean, 0.0)
    safe_cov = jnp.where(finite, cov, jnp.eye(cov.shape[0], dtype=cov.dtype))
    w2 = gaussian_wasserstein_distance(safe_mean, safe_cov, target_mean, target_cov)
    return jnp.where(finite, w2, jnp.inf)


def noise_floor(
    target: Any, key: jax.Array, num_chains: int, num_reps: int = 32
) -> jax.Array:
    """
    Monte-Carlo floor of the estimator: mean Gelbrich distance between C EXACT
    target samples and the target. Samplers cannot be more accurate than noise floor.

    Args:
        target: Target distribution with .mean and .cov attributes.
        key: PRNGKey for sampling.
        num_chains: Number of independent chains that are run.
        num_reps: Number of repetitions to estimate the mean floor.

    Returns:
        Scalar Monte-Carlo noise floor based on W2 distance.
    """

    chol = jnp.linalg.cholesky(target.cov)
    d = target.mean.shape[0]

    def single(k):
        z = jax.random.normal(k, (num_chains, d))
        xs = target.mean + z @ chol.T
        mean, cov = _moments(xs)
        return _gaussian_W2(mean, cov, target.mean, target.cov)

    return jnp.mean(jax.lax.map(single, jax.random.split(key, num_reps)))


def check_W2_converged(
    w2n: jax.Array, eps: float, steps: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Checks whether the sampler converged to the target,
    i.e. whether the normalised W2 <= eps at some checkpoint and stays below at all later checkpoints.

    Args:
        w2n (n_h, n_eval): Normalised W2 distance at n_eval checkpoints across n_h step sizes.
        eps: Convergence threshold.
        steps: Step numbers at which convergence metrics were evaluated.

    Returns:
        Indicator whether the sampler converged and at which checkpoint.
    """

    below = (w2n <= eps).astype(jnp.int32)
    stays = jnp.flip(jnp.cumprod(jnp.flip(below, -1), -1), -1).astype(bool)
    converged = stays.any(-1)
    steps_to_conv = jnp.where(converged, steps[jnp.argmax(stays, -1)], jnp.inf)
    return converged, steps_to_conv


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
    eval_steps: np.ndarray,
) -> dict[str, jax.Array]:
    """
    Run C independent chains of a sampler for a given step size, and evaluate convergence metrics at checkpoints.

    Args:
        sampler: Sampler instance with .init_state and .step methods.
        target: Target distribution with .mean and .cov attributes.
        x0 (d,): Initial point for all chains.
        step_size: Step sizes for the sampler.
        key: PRNGKey for random number generation.
        num_chains: Number of independent chains to run in parallel.
        eval_steps: Step numbers at which to evaluate convergence metrics.

    Returns:
        Dictionary containing convergence metrics at each checkpoint.
    """

    d = x0.shape[0]
    block_lengths = jnp.asarray(
        np.diff(np.concatenate([[0], eval_steps])), dtype=jnp.int32
    )

    # only h changes, all other hyperparameters are kept as constructed
    sampler = eqx.tree_at(lambda s: s.step_size, sampler, step_size)

    key_init, key_run = jax.random.split(key)
    x0s = jnp.broadcast_to(x0, (num_chains, d))  # same initial point for all chains

    # initialize all chains in parallel
    init_keys = jax.random.split(key_init, num_chains)
    state = jax.vmap(lambda x, k: sampler.init_state(x, key=k))(x0s, init_keys)

    # scan over chains for one time step, return only the new state (discard aux info)
    vstep = jax.vmap(lambda s, k: sampler.step(s, target, k)[0])

    def block(st, block_params):
        block_length, block_key = block_params

        def map_chains(i, s):
            return vstep(
                s, jax.random.split(jax.random.fold_in(block_key, i), num_chains)
            )

        # scan over all time steps in one evaluation block
        st = jax.lax.fori_loop(0, block_length, map_chains, st)  # dynamic trip count
        return st, _moments(st.x)

    # scan over evaluation blocks
    block_keys = jax.random.split(key_run, len(eval_steps))
    _, (means, covs) = jax.lax.scan(block, state, (block_lengths, block_keys))

    # use additional whitening, otherwise later normalization by sqrt(kappa)
    # might mask big error in direction with small std in standard W2
    # (e.g. sig_1 = 1, sig_2 = 100, err_1 = err_2 = 1, kappa = 1e4
    #  -> normalized W2: err_1/sqrt(kappa) = 0.01 < eps, whitened W2: err_1/sig_1 = 1 > eps)
    evals, V = jnp.linalg.eigh(target.cov)
    S = (
        V / jnp.sqrt(evals)
    ) @ V.T  # S = Σ^{-1/2}, rescales eigen-directions by 1/sig_i
    w2_white = jax.vmap(
        lambda m, c: _gaussian_W2(
            S @ (m - target.mean), S @ c @ S, jnp.zeros(d), jnp.eye(d)
        )
    )(means, covs)

    w2 = jax.vmap(lambda mean, cov: _gaussian_W2(mean, cov, target.mean, target.cov))(
        means, covs
    )

    # for mean and cov only report whitened metric
    means_white = (means - target.mean) @ S  # (n_eval, d)
    covs_white = S @ covs @ S  # (n_eval, d, d)

    mean_err = jnp.linalg.norm(means_white, axis=-1)
    cov_err = jnp.linalg.norm(covs_white - jnp.eye(d), ord="fro", axis=(-2, -1))

    return {
        "w2_gaussian": w2,  # (n_eval,)
        "w2_white": w2_white,
        "mean_error": mean_err,
        "cov_error": cov_err,
    }


def build_ensemble_evaluator(sampler: Any, num_chains: int, eval_steps: np.ndarray):
    """
    Returns jitted eval_fn(step_sizes, x0, key, target) -> dict of arrays (n_h, n_eval).
    Step sizes are vmapped, chains are vmapped inside. Independent keys per step size.

    Args:
        sampler: Sampler instance with .init_state and .step methods.
        num_chains: Number of independent chains to run in parallel.
        eval_steps: Step numbers at which to evaluate convergence metrics.

    Returns:
        Jitted function that evaluates the sampler for a given set of step sizes, initial point, random key, and target.
    """

    def eval_fn(step_sizes, x0, key, target):
        keys = jax.random.split(key, step_sizes.shape[0])
        return jax.lax.map(
            lambda h_k: run_ensemble(
                sampler, target, x0, h_k[0], h_k[1], num_chains, eval_steps
            ),
            (step_sizes, keys),
            batch_size=5,
        )

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
    eval_steps: np.ndarray,
    eps: float,
    eps_white: float,
    norm: float,
    floor: float,
    floor_white: float,
    runtime_sec: float | None = None,
) -> pd.DataFrame:
    """
    Construct a DataFrame summarizing convergence metrics for a given sampler and step sizes.

    Args:
        metrics: Convergence metrics at each checkpoint (as returned by build_ensemble_evaluator).
        step_sizes: Array of step sizes evaluated.
        sampler_name: Name of the sampler.
        dim: Dimension of the target distribution.
        eval_steps: Step numbers at which to evaluate convergence metrics.
        eps: Convergence threshold for the target.
        eps_white: Convergence threshold for the target in whitened coordinates.
        norm: Normalization factor for W2 distance.
        floor: Monte Carlo noise floor for the estimator.
        floor_white: Whitened noise floor (slightly larger than unwhitened floor).
        runtime_sec: Optional runtime in seconds for the evaluation.

    Returns:
        DataFrame summarizing convergence metrics for each step size.
    """

    w2n = metrics["w2_gaussian"] / norm  # (n_h, n_eval)
    w2n_white = metrics["w2_white"]
    steps = jnp.asarray(eval_steps)

    converged, steps_to_conv = check_W2_converged(w2n, eps, steps)
    converged_white, steps_to_conv_white = check_W2_converged(
        w2n_white, eps_white, steps
    )

    grads = grads_per_step.get(sampler_name, 1)
    return pd.DataFrame(
        {
            "sampler": sampler_name,
            "dimension": dim,
            "step_size": jnp.asarray(step_sizes).tolist(),
            "converged": converged.tolist(),
            "steps_to_conv": steps_to_conv.tolist(),
            "converged_white": converged_white.tolist(),
            "steps_to_conv_white": steps_to_conv_white.tolist(),
            "grad_evals_to_conv": jnp.where(
                converged, steps_to_conv * grads, jnp.inf
            ).tolist(),
            "grad_evals_to_conv_white": jnp.where(
                converged_white, steps_to_conv_white * grads, jnp.inf
            ).tolist(),
            "grads_per_step": grads,
            "final_w2_norm": w2n[:, -1].tolist(),
            "final_w2_white": w2n_white[:, -1].tolist(),
            "final_mean_err_white": metrics["mean_error"][
                :, -1
            ].tolist(),  # only use final checkpoint
            "final_cov_err_white": metrics["cov_error"][:, -1].tolist(),
            "diverged": (~jnp.isfinite(metrics["w2_gaussian"][:, -1])).tolist(),
            "floor_norm": floor / norm,
            "eps": eps,
            "floor_ok": floor / norm < 0.5 * eps,  # is eps resolvable with C chains?
            "floor_white": floor_white,
            "eps_white": eps_white,
            "floor_ok_white": floor_white < 0.5 * eps_white,
            "runtime_sec_total": runtime_sec,
            "sec_per_step": runtime_sec / (eval_steps[-1] * len(step_sizes)),
        }
    )
