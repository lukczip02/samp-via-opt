"""
Ensemble (parallel-chain) evaluation of samplers on Gaussian targets.

We run C independent chains from the same start x0 and estimate the law of X_n across
chains at every checkpoint n. Matches theoretical bounds (W2 of the law of X_n) and
removes burn-in from the estimator. Only per-checkpoint moments are stored,
not the full trajectory.
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
    cond_numbers: list[float],
    steps_per_kappa: float = 3.0,
    min_steps: int = 1_000,
    max_steps: int = 30_000,
    first: int = 5,
    rel_spacing: float = 0.05,
) -> list[np.ndarray]:
    """
    Create one strictly increasing, log-spaced checkpoint array per kappa.

    Args:
        cond_numbers: List of kappas that each get their own set of evaluation points.
        steps_per_kappa: Factor that increases the maximimum number of steps for increasing kappa.
        min_steps: Minimum number of sampler steps.
        max_steps: Maximum number of sampler steps.
        first: First index at which an evaluation point is set.
        rel_spacing: Relative spacing between evaluation points.

    Returns:
        List of log-spaced checkpoints, one per kappa.
    """

    out = []
    for kappa in cond_numbers:
        n_max = int(min(max(min_steps, steps_per_kappa * kappa), max_steps))
        # ratio = 1 + rel_spacing = (n_max / first)^(1/(n_pts − 1))
        # -> n_pts = ln(n_max / first) / ln(1 + rel_spacing) + 1
        n_pts = int(np.ceil(np.log(n_max / first) / np.log1p(rel_spacing))) + 1
        steps = np.geomspace(first, n_max, n_pts)
        out.append(np.unique(np.round(steps).astype(np.int64)))
    return out


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
    Monte-Carlo floor of the estimator: mean W2 distance between C exact target
    samples and the target. Samplers cannot be more accurate than noise floor.

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
    w2n: jax.Array,
    eps: float,
    steps: jax.Array,
    floor: float = 0.0,
    window: int = 5,
) -> tuple[jax.Array, jax.Array]:
    """
    Checks whether the sampler converged to the target, i.e. whether the (smoothed, floor-corrected)
    W2 <= eps at some checkpoint and stays below at all later checkpoints.

    Two measures make the criterion robust against Monte Carlo noise:
    - The noise floor is subtracted in quadrature: sqrt(max(w2^2 - floor^2, 0)).
    - A rolling median over `window` neighbouring checkpoints suppresses isolated peaks
       (up to window // 2 of them) above eps.

    Args:
        w2n (n_h, n_eval): W2 distance at n_eval checkpoints across n_h step sizes.
        eps: Convergence threshold.
        steps: Step numbers at which convergence metrics were evaluated.
        floor: Monte Carlo noise floor in the same units as w2n (0 disables the correction).
        window: Odd window length of the rolling median (1 disables smoothing).

    Returns:
        Indicator whether the sampler converged and at which step number.
    """
    n_eval = w2n.shape[-1]

    # noise-floor correction
    w2c = jnp.sqrt(jnp.maximum(w2n**2 - floor**2, 0.0))

    # centred rolling median, edges padded by repetition, only apply padding to checkpoint dimension
    if window > 1:
        half = window // 2
        padded = jnp.pad(w2c, [(0, 0)] * (w2c.ndim - 1) + [(half, half)], mode="edge")
        idx = (
            jnp.arange(n_eval)[:, None] + jnp.arange(window)[None, :]
        )  # (n_eval, window)
        w2c = jnp.median(padded[..., idx], axis=-1)

    below = (w2c <= eps).astype(jnp.int32)
    stays = jnp.flip(jnp.cumprod(jnp.flip(below, -1), -1), -1).astype(bool)
    converged = stays.any(-1)
    steps_to_conv = jnp.where(converged, steps[jnp.argmax(stays, -1)], jnp.inf)
    return converged, steps_to_conv


def asymptotic_error(w2n: jax.Array, floor: float, last: int = 10) -> jax.Array:
    """
    Late-time error level of each run: the median over the last `last` checkpoints
    of the floor-corrected W2, sqrt(max(w2^2 - floor^2, 0)).
    For small step sizes this is still transient, for large step sizes it approximates
    the discretisation bias.

    Args:
        w2n (n_h, n_eval): W2 distance at all checkpoints for each step size.
        floor: Monte Carlo noise floor in the same units as w2n.
        last: Number of final checkpoints to use.

    Returns:
        (n_h,) asymptotic error per step size.
    """
    w2c = jnp.sqrt(jnp.maximum(w2n**2 - floor**2, 0.0))
    return jnp.median(w2c[..., -last:], axis=-1)


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
    use_float32: bool,
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
        use_float32: To improve performance, chains can be run in float32 precision.
                     Numerically critical computations still use float64.

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

    sampler_target = (
        jax.tree.map(
            lambda a: a.astype(jnp.float32) if eqx.is_inexact_array(a) else a, target
        )
        if use_float32
        else target
    )

    # scan over chains for one time step, return only the new state (discard aux info)
    vstep = eqx.filter_vmap(lambda s, k: sampler.step(s, sampler_target, k)[0])

    def block(st, block_params):
        block_length, block_key = block_params

        def map_chains(i, s):
            return vstep(
                s, jax.random.split(jax.random.fold_in(block_key, i), num_chains)
            )

        # scan over all time steps in one evaluation block
        st = jax.lax.fori_loop(0, block_length, map_chains, st)  # dynamic trip count
        return st, _moments(
            st.x.astype(jnp.float64)
        )  # cast back to float_64 for metric computations

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


def build_ensemble_evaluator(
    sampler: Any, num_chains: int, eval_steps: np.ndarray, use_float32: bool
):
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
                sampler, target, x0, h_k[0], h_k[1], num_chains, eval_steps, use_float32
            ),
            (step_sizes, keys),
            batch_size=len(step_sizes) // 2,  # use even number of step sizes
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
    norm_white: float,
    floor: float,
    floor_white: float,
    runtime_sec: float | None = None,
    borderline_tol: float = 0.03,
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
        norm_white: Normalization factor for whitened W2 distance.
        floor: Monte Carlo noise floor for the estimator.
        floor_white: Whitened noise floor (slightly larger than unwhitened floor).
        runtime_sec: Optional runtime in seconds for the evaluation.

    Returns:
        DataFrame summarizing convergence metrics for each step size.
    """

    w2n = metrics["w2_gaussian"] / norm  # (n_h, n_eval)
    w2n_white = metrics["w2_white"] / norm_white
    steps = jnp.asarray(eval_steps)

    converged, steps_to_conv = check_W2_converged(w2n, eps, steps, floor=floor / norm)
    converged_white, steps_to_conv_white = check_W2_converged(
        w2n_white, eps_white, steps, floor=floor_white / norm_white
    )

    asympt = asymptotic_error(w2n, floor / norm)
    asympt_white = asymptotic_error(w2n_white, floor_white / norm_white)
    borderline = jnp.abs(asympt - eps) < borderline_tol
    borderline_white = jnp.abs(asympt_white - eps_white) < borderline_tol

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
            "diverged": (
                ~jnp.isfinite(w2n[:, -1]) | (w2n[:, -1] > 10.0)
            ).tolist(),  # high W2 else not captured in float64
            "floor_norm": floor / norm,
            "eps": eps,
            "floor_ok": floor / norm < 0.5 * eps,  # is eps resolvable with C chains?
            "floor_white": floor_white / norm_white,
            "eps_white": eps_white,
            "floor_ok_white": floor_white / norm_white < 0.5 * eps_white,
            "runtime_sec_total": runtime_sec,
            "sec_per_step": runtime_sec / (eval_steps[-1] * len(step_sizes)),
            "asympt_err_norm": asympt.tolist(),
            "asympt_err_white": asympt_white.tolist(),
            "borderline": borderline.tolist(),
            "borderline_white": borderline_white.tolist(),
            "borderline_tol": borderline_tol,
            "max_steps": eval_steps[-1].tolist(),
        }
    )
