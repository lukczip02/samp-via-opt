"""
Implementation of time-inhomgeneuous preconditioned langevin dynamics following Falk et al. (2026) (https://arxiv.org/abs/2605.06091).
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
        B: Preconditioner matrix.
        lam: Time-inhomgeneuous parameter controlling the interpolation between empirical covariance and local curvature.
        step: Current step number in the sampling process.
    """

    B: jax.Array
    lam: jax.Array
    step: int = 0


class TIPreL(BaseSampler[TIPreLState]):
    """
    Time-Inhomgeneuous Preconditioned Langevin Dynamics (TIPreL) sampler.
    Uses a time-inhomgeneuous preconditioner that interpolates between the empirical covariance and the local curvature of the target.

    Attributes:
        num_steps: Total number of steps in the sampling process.
        step_size: Step size for the Langevin dynamics.
        emp_cov: Empirical covariance matrix used for preconditioning.
    """

    num_steps: int
    step_size: float
    emp_cov: jax.Array

    def init_state(self, x0: jax.Array, key: Optional[jax.Array] = None) -> TIPreLState:
        return TIPreLState(x=x0, B=self.emp_cov, lam=jnp.array(0.0), step=0)

    def step(
        self, state: TIPreLState, target: BaseTarget, key: jax.Array
    ) -> tuple[TIPreLState, dict[str, jax.Array]]:
        def compute_preconditioner(x: jax.Array, eps: float = 1e-6) -> jax.Array:
            H_pos = -jax.hessian(target.log_prob)(x)
            eigval, eigvec = jnp.linalg.eigh(H_pos)
            B_1 = eigvec @ jnp.diag(1.0 / jnp.maximum(eigval, eps)) @ eigvec.T
            return (1.0 - state.lam) * self.emp_cov + state.lam * B_1

        h = self.step_size
        B = compute_preconditioner(state.x)

        # compute row-wise matrix divergence: (div B)_i = sum_j d(B_ij) / d(x_j)
        def rowwise_divergence(x: jax.Array) -> jax.Array:
            jac = jax.jacrev(compute_preconditioner)(
                x
            )  # shape (d, d, d) -> dB_{ij} / dx_k
            return jnp.einsum("ijj->i", jac)

        # SDE drift
        div_B = rowwise_divergence(state.x)
        grad = target.grad_log_prob(state.x)
        b = B @ grad + div_B

        # tamed drift
        b_h = b / (1.0 + h * jnp.linalg.norm(b))

        b_cholesky = jnp.linalg.cholesky(B)
        noise = b_cholesky @ jax.random.normal(key, shape=state.x.shape)

        x_new = state.x + h * b_h + jnp.sqrt(2.0 * h) * noise
        step_new = state.step + 1

        lam_new = jnp.minimum(2.0 * (step_new) / self.num_steps, 1.0)

        new_state = TIPreLState(x=x_new, B=B, lam=lam_new, step=step_new)
        aux_info = {"grad_norm": jnp.linalg.norm(grad)}

        return new_state, aux_info
