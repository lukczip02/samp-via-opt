"""
Implementation of Randomized Midpoint Unadjusted Langevin Monte Carlo (RM-ULMC) sampler.
The RM-ULMC sampler follows Shen and Lee (2019) https://proceedings.neurips.cc/paper_files/paper/2019/file/eb86d510361fc23b59f18c1bc9802cc6-Paper.pdf.
Additionally we add a friction parameter gamma to match ULMC.

Numerical modifications (float32 safety):
--
The noise over one step is built from two independent Brownian segments
(durations T1 = alpha*h and T2 = (1-alpha)*h). Per segment of duration T we need the
jointly Gaussian pair
    G = int_0^T e^{-2(T-s)} dB_s       Var(G) = (1-e^{-4T})/4
    B = int_0^T dB_s                   Var(B) = T
                                       Cov(G, B) = (1-e^{-2T})/2

Its Cholesky factor has the closed form (ordering (G, B))
    l11 = sqrt((1-e^{-4T}) / 4),  l21 = sqrt(tanh(T)),  l22 = sqrt(T-tanh(T))
because Cov^2/Var(G) = tanh(T) exactly.
This avoids the determinant a*c-b^2 of a naive 2x2 factorisation, which cancels like
O(T^2) and is garbage in float32 for small T (alpha is uniform, so small T occurs regularly).
Only T-tanh(T) needs a series for small T.

The noise terms are then
    W1 = B1 - G1,    W3 = e^{-2*dh} G1 + G2,    W2 = B1 + B2 - W3,
which is algebraically identical to the original formulation.
"""

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from samp_via_opt.samplers.base import BaseSampler
from samp_via_opt.samplers.ulmc import ULMCState
from samp_via_opt.targets.base import BaseTarget


def _t_minus_tanh_t(T: jax.Array) -> jax.Array:
    """Computes T-tanh(T) without cancellation for small T (series below 0.2)."""
    T2 = T * T
    series = T * T2 * (1 / 3 - T2 * (2 / 15 - T2 * (17 / 315 - T2 * (62 / 2835))))
    direct = T - jnp.tanh(T)
    return jnp.maximum(jnp.where(T < 0.2, series, direct), 0.0)


def _segment_noise(T: jax.Array, z: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Jointly Gaussian pair (G, B) for a Brownian segment of duration T.

    Args:
        T: Segment duration (scalar, >= 0; T = 0 yields zeros, no NaNs).
        z: Standard normals of shape (2, d).

    Returns:
        G, B, each of shape (d,).
    """
    l11 = jnp.sqrt(-jnp.expm1(-4.0 * T) / 4.0)
    l21 = jnp.sqrt(jnp.tanh(T))
    l22 = jnp.sqrt(_t_minus_tanh_t(T))
    return l11 * z[0], l21 * z[0] + l22 * z[1]


class RMULMC(BaseSampler[ULMCState]):
    """
    Randomized Midpoint Unadjusted Langevin Algorithm (RM-ULMC) sampler.

    Attributes:
        step_size: Step size for the Langevin dynamics.
        gamma: Friction coefficient for the velocity update.
               Not in original paper.
        u: Temperature parameter controlling the noise level.
        dtype: Used floating point precision, either float32 or float64.
    """

    step_size: float
    gamma: float | None = 2.0  # 2.0 = Shen & Lee
    u: float = 1.0
    dtype: DTypeLike = jnp.float32

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
        x0 = jnp.asarray(x0, dtype=self.dtype)
        if key is None:
            v0 = jnp.zeros_like(x0)  # deterministic fallback
        else:
            # Sample from stationary velocity distribution N(0, u * I)
            u = jnp.asarray(self._u, dtype=self.dtype)
            v0 = jnp.sqrt(u) * jax.random.normal(key, shape=x0.shape, dtype=x0.dtype)
        return ULMCState(x=x0, v=v0)

    def step(
        self, state: ULMCState, target: BaseTarget, key: jax.Array
    ) -> tuple[ULMCState, dict[str, jax.Array]]:
        dtype = state.x.dtype
        d = state.x.shape[0]
        h = jnp.asarray(self._h, dtype=dtype)
        u = jnp.asarray(self._u, dtype=dtype)
        sqrt_u = jnp.sqrt(u)

        midpoint_key, noise1_key, noise2_key = jax.random.split(key, 3)

        # alpha ~ U[0, 1)
        # beta = 1 - alpha is exact in floating point, unlike h - alpha * h.
        alpha = jax.random.uniform(midpoint_key, shape=(), dtype=dtype)
        beta = 1.0 - alpha
        ah = alpha * h
        dh = beta * h

        z1 = jax.random.normal(noise1_key, shape=(2, d), dtype=dtype)
        z2 = jax.random.normal(noise2_key, shape=(2, d), dtype=dtype)
        G1, B1 = _segment_noise(ah, z1)
        G2, B2 = _segment_noise(dh, z2)

        exp_neg_2h = jnp.exp(-2.0 * h)
        exp_neg_2dh = jnp.exp(-2.0 * dh)

        W1 = B1 - G1
        W3 = exp_neg_2dh * G1 + G2
        W2 = B1 + B2 - W3

        grad = jnp.asarray(target.grad_log_prob(state.x), dtype=dtype)

        one_minus_exp_2ah = -jnp.expm1(-2.0 * ah)
        one_minus_exp_2h = -jnp.expm1(-2.0 * h)
        one_minus_exp_2dh = -jnp.expm1(-2.0 * dh)

        x_mid = (
            state.x
            + 0.5 * one_minus_exp_2ah * state.v
            + 0.5 * u * (ah - 0.5 * one_minus_exp_2ah) * grad
            + sqrt_u * W1
        )

        grad_mid = jnp.asarray(target.grad_log_prob(x_mid), dtype=dtype)

        x_new = (
            state.x
            + 0.5 * one_minus_exp_2h * state.v
            + 0.5 * u * h * one_minus_exp_2dh * grad_mid
            + sqrt_u * W2
        )

        v_new = (
            exp_neg_2h * state.v + u * h * exp_neg_2dh * grad_mid + 2.0 * sqrt_u * W3
        )

        new_state = ULMCState(
            x=jnp.asarray(x_new, dtype=dtype), v=jnp.asarray(v_new, dtype=dtype)
        )
        aux_info = {"grad_norm": jnp.linalg.norm(grad)}

        return new_state, aux_info
