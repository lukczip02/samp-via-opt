"""
Numerical experiment script evaluating samplers on the 2D Rosenbrock target,
following the experimental methodology in Falk et al. (2026) (arXiv:2605.06091).
"""

from typing import Any
import time
import jax
import jax.numpy as jnp
import pandas as pd
import equinox as eqx

from samp_via_opt.samplers.base import BaseSampler
from samp_via_opt.targets.base import BaseTarget
from samp_via_opt.targets.rosenbrock import Rosenbrock2DTarget
from samp_via_opt.samplers import ULA, RMULA, ULMC, RMULMC, TIPreL
from samp_via_opt.diagnostics import sliced_wasserstein_distance


grads_per_step = {
    "ULA": 1,
    "ULMC": 1,
    "RMULA": 2,
    "RMULMC": 2,
    "TIPreL": 1,  # 1 gradient evaluation + 1 Hessian matrix evaluation
}


@eqx.filter_jit
def run_pilot_chain(
    sampler: BaseSampler,
    target: BaseTarget,
    x0: jax.Array,
    key: jax.Array,
    num_steps: int = 2000,
    burn_in: int = 500,
) -> jax.Array:
    """
    Runs a single pilot chain and returns non-burn-in trajectory positions.

    Args:
        sampler: Sampler instance (e.g. ULA).
        target: Target distribution.
        x0: Initial position in target space.
        key: PRNG key.
        num_steps: Total number of steps in the pilot run.
        burn_in: Number of initial steps to discard.

    Returns:
        Array of valid trajectory positions, shape (num_steps - burn_in, d).
    """
    key_init, key_sample = jax.random.split(key)
    init_state = sampler.init_state(x0, key=key_init)
    _, xs, _ = sampler.sample(target, init_state, key_sample, num_steps, save_aux=False)
    return xs[burn_in:]


def estimate_pilot_cov(
    sampler: BaseSampler,
    target: BaseTarget,
    x0: jax.Array,
    key: jax.Array,
    num_steps: int = 2000,
    burn_in: int = 500,
) -> jax.Array:
    """
    Computes empirical covariance from a pilot chain trajectory.
    """
    valid_xs = run_pilot_chain(sampler, target, x0, key, num_steps, burn_in)
    return jnp.cov(valid_xs, rowvar=False)


def rosenbrock_slow_start(
    target: Rosenbrock2DTarget, n_sigma: float = 3.0
) -> jax.Array:
    """
    Initializes x0 along the principal direction of maximal variance from the target mean.
    """
    evals, evecs = jnp.linalg.eigh(target.cov)
    return target.mean + n_sigma * jnp.sqrt(evals[-1]) * evecs[:, -1]


def run_rosenbrock_ensemble(
    sampler: Any,
    target: Rosenbrock2DTarget,
    gt_samples: jax.Array,
    x0: jax.Array,
    step_size: jax.Array,
    key: jax.Array,
    num_chains: int,
    num_steps: int,
    num_eval_points: int,
    num_sw_projections: int = 64,
) -> dict[str, jax.Array]:
    """
    Runs C parallel chains and records SW_2 and moment errors across evaluation checkpoints.
    """
    steps_per_block = num_steps // num_eval_points
    d = x0.shape[0]

    # Update step size hyperparameter
    sampler = eqx.tree_at(lambda s: s.step_size, sampler, step_size)

    key_init, key_run, key_sw = jax.random.split(key, 3)
    x0s = jnp.broadcast_to(x0, (num_chains, d))
    init_keys = jax.random.split(key_init, num_chains)

    state = jax.vmap(lambda x, k: sampler.init_state(x, key=k))(x0s, init_keys)
    vstep = jax.vmap(lambda s, k: sampler.step(s, target, k)[0])

    def map_chains(st, k):
        return vstep(st, jax.random.split(k, num_chains)), None

    def map_eval_blocks(st, k):
        st, _ = jax.lax.scan(map_chains, st, jax.random.split(k, steps_per_block))
        return st, st.x

    _, ensemble_checkpoints = jax.lax.scan(
        map_eval_blocks, state, jax.random.split(key_run, num_eval_points)
    )

    sw_keys = jax.random.split(key_sw, num_eval_points)

    def eval_checkpoint(xs, k_sw):
        sw2 = sliced_wasserstein_distance(
            xs, gt_samples, k_sw, num_projections=num_sw_projections
        )
        emp_mean = jnp.mean(xs, axis=0)
        emp_cov = jnp.cov(xs, rowvar=False)
        mean_err = jnp.linalg.norm(emp_mean - target.mean)
        cov_err = jnp.linalg.norm(emp_cov - target.cov, ord="fro")
        return sw2, mean_err, cov_err

    sw2_arr, mean_err_arr, cov_err_arr = jax.vmap(eval_checkpoint)(
        ensemble_checkpoints, sw_keys
    )

    return {
        "sw2_distance": sw2_arr,
        "mean_error": mean_err_arr,
        "cov_error": cov_err_arr,
    }


def build_rosenbrock_evaluator(
    sampler: Any,
    num_chains: int,
    num_steps: int,
    num_eval_points: int = 50,
    num_sw_projections: int = 64,
):
    """
    Returns a JIT-compiled evaluator over array of step sizes.
    """

    def eval_fn(step_sizes, target, gt_samples, x0, key):
        keys = jax.random.split(key, step_sizes.shape[0])
        return jax.vmap(
            lambda h, k: run_rosenbrock_ensemble(
                sampler,
                target,
                gt_samples,
                x0,
                h,
                k,
                num_chains,
                num_steps,
                num_eval_points,
                num_sw_projections,
            )
        )(step_sizes, keys)

    return eqx.filter_jit(eval_fn)


def rosenbrock_convergence_table(
    metrics: dict[str, jax.Array],
    step_sizes: jax.Array,
    sampler_name: str,
    num_steps: int,
    target_sw2_eps: float,
    runtime_sec: float | None = None,
) -> pd.DataFrame:
    """
    Summarizes SW_2 convergence metrics and computational cost per step size.
    """
    sw2 = metrics["sw2_distance"]  # shape (n_h, n_eval)
    n_h, n_eval = sw2.shape
    eval_steps = (num_steps // n_eval) * jnp.arange(1, n_eval + 1)

    below = (sw2 <= target_sw2_eps).astype(jnp.int32)
    stays = jnp.flip(jnp.cumprod(jnp.flip(below, -1), -1), -1).astype(bool)
    converged = stays.any(-1)
    steps_to_conv = jnp.where(converged, eval_steps[jnp.argmax(stays, -1)], -1)

    grads = grads_per_step.get(sampler_name, 1)

    return pd.DataFrame(
        {
            "sampler": sampler_name,
            "step_size": jnp.asarray(step_sizes).tolist(),
            "converged": converged.tolist(),
            "steps_to_conv": steps_to_conv.tolist(),
            "grad_evals_to_conv": jnp.where(
                converged, steps_to_conv * grads, -1
            ).tolist(),
            "final_sw2": sw2[:, -1].tolist(),
            "min_sw2": jnp.min(sw2, axis=-1).tolist(),
            "final_mean_error": metrics["mean_error"][:, -1].tolist(),
            "final_cov_error": metrics["cov_error"][:, -1].tolist(),
            "diverged": (~jnp.isfinite(sw2[:, -1])).tolist(),
            "runtime_sec_total": runtime_sec,
        }
    )


def run_rosenbrock_experiment(
    a: float = 1.0,
    b: float = 100.0,
    num_chains: int = 1000,
    num_steps: int = 20_000,
    gt_sample_size: int = 10_000,
    step_sizes: jax.Array = jnp.logspace(-5, -1, 10),
    target_sw2_eps: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Runs multi-sampler grid search comparison on 2D Rosenbrock target.
    """
    master_key = jax.random.PRNGKey(seed)
    master_key, gt_key, run_key, pilot_key = jax.random.split(master_key, 4)

    target = Rosenbrock2DTarget(a=a, b=b)
    gt_samples = target.sample(gt_key, shape=(gt_sample_size,))
    x0 = rosenbrock_slow_start(target, n_sigma=3.0)

    # 1. Run a short pilot chain with standard ULA
    pilot_sampler = ULA(step_size=1e-3)
    emp_cov_pilot = estimate_pilot_cov(
        sampler=pilot_sampler,
        target=target,
        x0=x0,
        key=pilot_key,
        num_steps=3000,
        burn_in=1000,
    )

    sampler_registry = {
        "ULA": lambda t, g: ULA(step_size=1.0),
        "RMULA": lambda t, g: RMULA(step_size=1.0),
        "ULMC": lambda t, g: ULMC(step_size=1.0, alpha=2.0, beta=1.0),
        "RMULMC": lambda t, g: RMULMC(step_size=1.0, u=1.0),
        "TIPreL": lambda t, g: TIPreL(
            step_size=1.0, num_steps=num_steps, emp_cov=emp_cov_pilot
        ),
    }

    tables = []

    for name, factory in sampler_registry.items():
        print(f"\nRunning experiment for sampler: {name}...")
        sampler = factory(target, None)
        eval_fn = build_rosenbrock_evaluator(
            sampler, num_chains=num_chains, num_steps=num_steps
        )

        run_key, subkey = jax.random.split(run_key)

        # Warmup and JIT compile
        jax.block_until_ready(eval_fn(step_sizes, target, gt_samples, x0, subkey))

        t0 = time.perf_counter()
        metrics = jax.block_until_ready(
            eval_fn(step_sizes, target, gt_samples, x0, subkey)
        )
        sec = time.perf_counter() - t0

        df = rosenbrock_convergence_table(
            metrics,
            step_sizes,
            sampler_name=name,
            num_steps=num_steps,
            target_sw2_eps=target_sw2_eps,
            runtime_sec=sec,
        )
        tables.append(df)
        print(f"{name} completed in {sec:.2f}s")

    return pd.concat(tables, ignore_index=True)


if __name__ == "__main__":
    results_df = run_rosenbrock_experiment(
        a=1.0,
        b=100.0,
        num_chains=1000,
        num_steps=10_000,
        target_sw2_eps=0.08,
    )
    results_df.to_csv("rosenbrock_sampler_comparison_results.csv", index=False)
