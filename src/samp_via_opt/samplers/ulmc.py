"""
Implementation of Unadjusted Langevin Monte Carlo (ULMC) sampler using a simple first-order discretization.
ULMC is the sampling equivalent of Momentum based optimization methods, such as Nesterov's accelerated gradient descent.
The algorithm follows Habring et al. (2025) https://arxiv.org/abs/2507.12432.
"""

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
    """

    step_size: float
    alpha: float | None = 2.0
    beta: float = 1.0

    def init_state(self, x0: jax.Array, key: jax.Array | None = None) -> ULMCState:
        x0 = jnp.asarray(x0)
        if key is None:
            v0 = jnp.zeros_like(x0)  # deterministic fallback
        else:
            # Sample from stationary velocity distribution N(0, beta * I)
            v0 = jnp.sqrt(self.beta) * jax.random.normal(key, shape=x0.shape)
        return ULMCState(x=x0, v=v0)

    def step(
        self, state: ULMCState, target: BaseTarget, key: jax.Array
    ) -> tuple[ULMCState, dict[str, jax.Array]]:
        d = state.x.shape[0]
        h = self.step_size
        alpha = self.alpha if self.alpha is not None else 2.0
        beta = self.beta

        exp_ah = jnp.exp(-h * alpha)
        one_minus_exp_ah = -jnp.expm1(-h * alpha)
        one_minus_exp_2ah = -jnp.expm1(-2.0 * h * alpha)
        inv_a = 1.0 / alpha
        coeff_v = (1.0 - exp_ah) * inv_a

        grad = target.grad_log_prob(state.x)

        mu_x = state.x + state.v * coeff_v + beta * inv_a * grad * (h - coeff_v)
        mu_v = state.v * exp_ah + beta * inv_a * grad * one_minus_exp_ah

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
        z = jax.random.normal(key, shape=(2, d))
        noise = L_2d @ z  # Shape (2, d)

        x_new = mu_x + noise[0]
        v_new = mu_v + noise[1]

        new_state = ULMCState(x=x_new, v=v_new)
        aux_info = {"grad_norm": jnp.linalg.norm(grad)}

        return new_state, aux_info
