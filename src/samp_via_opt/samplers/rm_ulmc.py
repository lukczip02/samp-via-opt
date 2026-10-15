"""
Implementation of Randomized Midpoint Unadjusted Langevin Monte Carlo (RM-ULMC) sampler.
The RM-ULMC sampler follows Shen and Lee (2019) https://proceedings.neurips.cc/paper_files/paper/2019/file/eb86d510361fc23b59f18c1bc9802cc6-Paper.pdf.
Additionally we add a friction parameter gamma to match ULMC.
"""

import jax
import jax.numpy as jnp
from samp_via_opt.samplers.base import BaseSampler
from samp_via_opt.samplers.ulmc import ULMCState
from samp_via_opt.targets.base import BaseTarget


class RMULMC(BaseSampler[ULMCState]):
    """
    Randomized Midpoint Unadjusted Langevin Algorithm (RM-ULMC) sampler.

    Attributes:
        step_size: Step size for the Langevin dynamics.
        gamma: Friction coefficient for the velocity update.
               Not in original paper.
        u: Temperature parameter controlling the noise level.
    """

    step_size: float
    gamma: float | None = 2.0  # 2.0 = Shen & Lee
    u: float = 1.0

    @property
    def _h(self):  # in converted time units
        return (
            0.5 * self.gamma * self.step_size
            if self.gamma is not None
            else self.step_size
        )

    @property
    def _u(self):  # in converted time units
        return 4.0 * self.u / self.gamma**2 if self.gamma is not None else self.u

    def init_state(self, x0: jax.Array, key: jax.Array | None = None) -> ULMCState:
        x0 = jnp.asarray(x0)
        if key is None:
            v0 = jnp.zeros_like(x0)  # deterministic fallback
        else:
            # Sample from stationary velocity distribution N(0, u * I)
            v0 = jnp.sqrt(self._u) * jax.random.normal(key, shape=x0.shape)
        return ULMCState(x=x0, v=v0)

    def _matrix_sqrt_2x2(self, a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
        """
        Computes stable cholesky factor for 2x2 symmetric matrix [[a, b], [b, c]].

        Args:
            a: Element (0, 0) of the matrix.
            b: Element (0, 1) and (1, 0) of the matrix.
            c: Element (1, 1) of the matrix.

        Returns:
            A 2x2 lower triangular matrix L such that L @ L.T = [[a, b], [b, c]].
        """

        a_safe = jnp.maximum(a, 1e-12)
        l11 = jnp.sqrt(a_safe)
        l21 = b / l11
        det = a * c - b * b
        l22 = jnp.sqrt(jnp.maximum(det / a_safe, 1e-12))
        return jnp.array([[l11, 0.0], [l21, l22]])

    def step(
        self, state: ULMCState, target: BaseTarget, key: jax.Array
    ) -> tuple[ULMCState, dict[str, jax.Array]]:
        d = state.x.shape[0]
        h = self._h
        u = self._u

        midpoint_key, noise1_key, noise2_key = jax.random.split(key, 3)

        # clip alpha away from boundaries (0, 1) to prevent zero-variance matrices
        alpha = jnp.clip(jax.random.uniform(midpoint_key, shape=()), 1e-5, 1.0 - 1e-5)

        ah = alpha * h
        dh = h - ah

        # covariance 1
        a1 = 0.25 * jnp.expm1(4.0 * ah)
        b1 = 0.5 * jnp.expm1(2.0 * ah)
        c1 = ah
        L1 = self._matrix_sqrt_2x2(a1, b1, c1)

        z1 = jax.random.normal(noise1_key, shape=(2, d))
        gh1 = L1 @ z1
        g1, h1_noise = gh1[0], gh1[1]

        # covariance 2 (e^(4h) - e^(4ah) = e^(4ah) * expm1(4dh))
        exp_4ah = jnp.exp(4.0 * ah)
        exp_2ah = jnp.exp(2.0 * ah)
        a2 = 0.25 * exp_4ah * jnp.expm1(4.0 * dh)
        b2 = 0.5 * exp_2ah * jnp.expm1(2.0 * dh)
        c2 = dh
        L2 = self._matrix_sqrt_2x2(a2, b2, c2)

        z2 = jax.random.normal(noise2_key, shape=(2, d))
        gh2 = L2 @ z2
        g2, h2_noise = gh2[0], gh2[1]

        exp_neg_2ah = jnp.exp(-2.0 * ah)
        exp_neg_2h = jnp.exp(-2.0 * h)
        exp_neg_2dh = jnp.exp(-2.0 * dh)

        W1 = h1_noise - exp_neg_2ah * g1
        W2 = (h1_noise + h2_noise) - exp_neg_2h * (g1 + g2)
        W3 = exp_neg_2h * (g1 + g2)

        grad = target.grad_log_prob(state.x)

        one_minus_exp_2ah = -jnp.expm1(-2.0 * ah)
        one_minus_exp_2h = -jnp.expm1(-2.0 * h)
        one_minus_exp_2dh = -jnp.expm1(-2.0 * dh)

        x_mid = (
            state.x
            + 0.5 * one_minus_exp_2ah * state.v
            + 0.5 * u * (ah - 0.5 * one_minus_exp_2ah) * grad
            + jnp.sqrt(u) * W1
        )

        grad_mid = target.grad_log_prob(x_mid)

        x_new = (
            state.x
            + 0.5 * one_minus_exp_2h * state.v
            + 0.5 * u * h * one_minus_exp_2dh * grad_mid
            + jnp.sqrt(u) * W2
        )

        v_new = (
            exp_neg_2h * state.v
            + u * h * exp_neg_2dh * grad_mid
            + 2.0 * jnp.sqrt(u) * W3
        )

        new_state = ULMCState(x=x_new, v=v_new)
        aux_info = {"grad_norm": jnp.linalg.norm(grad)}

        return new_state, aux_info
