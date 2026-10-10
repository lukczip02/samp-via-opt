"""
Implementation of time-inhomgeneuous preconditioned langevin dynamics following Falk et al. (2026) https://arxiv.org/abs/2605.06091.
As an analog to Newtons method in optimization, the theoretical aspect we investigate is the independence of condition number.
"""

import jax
import jax.numpy as jnp
from typing import Optional
from samp_via_opt.samplers.base import BaseSampler, SamplerState
from samp_via_opt.targets.base import BaseTarget


class TIPreLState(SamplerState):
    """
    State for the time-inhomgeneuous preconditioned langevin dynamics sampler.

    Attributes:
        x: Current position in the target space.
        step: Current step number in the sampling process.
    """

    step: int = 0


class TIPreL(BaseSampler[TIPreLState]):
    """
    Time-Inhomgeneuous Preconditioned Langevin Dynamics (TIPreL) sampler.
    Uses a time-inhomgeneuous preconditioner that interpolates between the empirical covariance and the local curvature of the target.

    Attributes:
        num_steps: Total number of steps in the sampling process. Has to be divisible by num_eval_points.
        step_size: Step size for the Langevin dynamics.
        emp_cov: Empirical covariance matrix used for preconditioning.
    """

    num_steps: int
    step_size: float
    emp_cov: jax.Array
    use_taming: bool = True

    def init_state(self, x0: jax.Array, key: Optional[jax.Array] = None) -> TIPreLState:
        return TIPreLState(x=x0, step=0)

    def step(
        self, state: TIPreLState, target: BaseTarget, key: jax.Array
    ) -> tuple[TIPreLState, dict[str, jax.Array]]:
        def compute_B_and_divB(
            x: jax.Array, lam: jax.Array, eps: float = 1e-6
        ) -> tuple[jax.Array, jax.Array]:
            if getattr(target, "constant_hessian", False):
                B1 = getattr(target, "B1", getattr(target, "cov"))
                div_B = jnp.zeros_like(x)
                B = (1.0 - lam) * self.emp_cov + lam * B1
            else:

                def compute_B(x: jax.Array) -> jax.Array:
                    eigval, V = jnp.linalg.eigh(-jax.hessian(target.log_prob)(x))
                    B1 = (V / jnp.maximum(jnp.abs(eigval), eps)) @ V.T
                    return (1 - lam) * self.emp_cov + lam * B1

                B = compute_B(x)

                # uses one jvp per coordinate to calculate divB
                def column(j: jax.Array) -> jax.Array:
                    e_j = jnp.zeros(d, state.x.dtype).at[j].set(1.0)
                    _, dB = jax.jvp(compute_B, (x,), (e_j,))
                    return dB[:, j]

                div_B = jax.vmap(column)(jnp.arange(d)).sum(0)

            return B, div_B

        d = state.x.shape[0]
        h = self.step_size
        lam = jnp.minimum(2.0 * state.step / self.num_steps, 1.0)

        B, div_B = compute_B_and_divB(state.x, lam)

        # SDE drift
        grad = target.grad_log_prob(state.x)
        b = B @ grad + div_B

        # tamed drift
        b_h = (b / (1.0 + h * jnp.linalg.norm(b))) if self.use_taming else b

        B_safe = B + 1e-6 * jnp.eye(d, dtype=B.dtype)  # jitter for numerical stability
        b_cholesky = jnp.linalg.cholesky(B_safe)
        noise = b_cholesky @ jax.random.normal(key, shape=state.x.shape)

        x_new = state.x + h * b_h + jnp.sqrt(2.0 * h) * noise

        new_state = TIPreLState(x=x_new, step=state.step + 1)
        aux_info = {"grad_norm": jnp.linalg.norm(grad)}

        return new_state, aux_info
