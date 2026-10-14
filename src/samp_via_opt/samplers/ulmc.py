"""
Implementation of Underdamped Langevin Monte Carlo (ULMC) sampler using a simple first-order discretization.
ULMC is the sampling equivalent of Momentum based optimization methods, such as Nesterov's accelerated gradient descent.
The algorithm follows Habring et al. (2025) https://arxiv.org/abs/2507.12432.
"""

from jax.typing import DTypeLike
import jax
import jax.numpy as jnp
from samp_via_opt.samplers.base import BaseSampler, SamplerState
from samp_via_opt.targets.base import BaseTarget


class ULMCState(SamplerState):
    """
    State for the Unadjusted Langevin Monte Carlo (ULMC) sampler.

    Attributes:
        x: Current position in the target space.
        v: Current velocity in the target space.
    """

    v: jax.Array


class ULMC(BaseSampler[ULMCState]):
    """
    Unadjusted Langevin Monte Carlo (ULMC) sampler.

    Attributes:
        step_size: Step size for the Langevin dynamics.
        alpha: Friction coefficient for the velocity update.
        beta: Temperature parameter controlling the noise level.
        dtype: Used floating point precision, either float32 or float64.
    """

    step_size: float
    alpha: float | None = 2.0
    beta: float = 1.0
    dtype: DTypeLike = jnp.float32

    def init_state(self, x0: jax.Array, key: jax.Array | None = None) -> ULMCState:
        x0 = jnp.asarray(x0, dtype=self.dtype)
        if key is None:
            v0 = jnp.zeros_like(x0)  # deterministic fallback
        else:
            # Sample from stationary velocity distribution N(0, beta * I)
            v0 = jnp.sqrt(jnp.asarray(self.beta, dtype=self.dtype)) * jax.random.normal(
                key, shape=x0.shape, dtype=self.dtype
            )
        return ULMCState(x=x0, v=v0)

    def step(
        self, state: ULMCState, target: BaseTarget, key: jax.Array
    ) -> tuple[ULMCState, dict[str, jax.Array]]:
        d = state.x.shape[0]

        # use high precision for cov calculation
        h = jnp.asarray(self.step_size, dtype=jnp.float64)
        alpha = jnp.asarray(
            2.0 if self.alpha is None else self.alpha, dtype=jnp.float64
        )
        beta = jnp.asarray(self.beta, dtype=jnp.float64)

        exp_ah = jnp.exp(-h * alpha)
        one_minus_exp_ah = -jnp.expm1(-h * alpha)
        one_minus_exp_2ah = -jnp.expm1(-2.0 * h * alpha)
        inv_a = 1.0 / alpha
        coeff_v = one_minus_exp_ah * inv_a

        coeff_grad_x = beta * inv_a * (h - coeff_v)
        coeff_grad_v = beta * inv_a * one_minus_exp_ah

        cov_xx = (
            2.0
            * beta
            * inv_a
            * (h - 2.0 * inv_a * one_minus_exp_ah + 0.5 * inv_a * one_minus_exp_2ah)
        )
        cov_xv = beta * inv_a * (one_minus_exp_ah**2)
        cov_vv = beta * one_minus_exp_2ah

        cov = jnp.array([[cov_xx, cov_xv], [cov_xv, cov_vv]])

        # Sample (x, v) via transformation with Cholesky factor
        L_2d = jnp.linalg.cholesky(cov)

        # only now go to state dtype
        exp_ah, coeff_v, coeff_grad_x, coeff_grad_v, L_2d = (
            a.astype(state.x.dtype)
            for a in (exp_ah, coeff_v, coeff_grad_x, coeff_grad_v, L_2d)
        )

        grad = target.grad_log_prob(state.x)

        mu_x = state.x + coeff_v * state.v + coeff_grad_x * grad
        mu_v = exp_ah * state.v + coeff_grad_v * grad

        z = jax.random.normal(key, shape=(2, d), dtype=state.x.dtype)
        noise = L_2d @ z  # Shape (2, d)

        x_new = mu_x + noise[0]
        v_new = mu_v + noise[1]

        new_state = ULMCState(x=x_new, v=v_new)
        aux_info = {"grad_norm": jnp.linalg.norm(grad)}

        return new_state, aux_info
