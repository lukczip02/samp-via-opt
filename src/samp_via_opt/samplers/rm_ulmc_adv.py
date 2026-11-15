"""
Implementation of the Randomized Midpoint Underdamped Langevin Monte Carlo (RM-ULMC) sampler.
The algorithm is the modified variant of Altschuler et al. (2025), Sec. 5.3.3
(Shen and Lee (2019) use one random midpoint, here two independent ones
u_n ~ q_u and v_n ~ q_v are drawn per step).

Dynamics (temperature beta, friction gamma, potential V = -log p):
    dX = P dt,    dP = (-gamma P + beta * grad log p(X)) dt + sqrt(2 gamma beta) dB.

One step costs three gradient evaluations (at x, at the u-midpoint and at the v-midpoint).

Numerical design (float32 safe, no linear algebra):
- The joint noise (xi1(t_u), xi1(t_v), xi1(h), xi2(h)) is built from three independent Brownian segments
  [0, a], [a, b], [b, h] with a = min(t_u, t_v), b = max(t_u, t_v). Per segment of duration T we need
      G = int e^{-gamma (T-s)} dB_s,   B = int dB_s,
  whose Cholesky factor is known in closed form (x = gamma*T/2)
      l11 = sqrt((1 - e^{-2 gamma T}) / (2 gamma)),  l21 = sqrt(2/gamma * tanh(x)),  l22 = sqrt(2/gamma * (x - tanh(x))).
  This replaces the 4x4 covariance + eigh of a naive implementation, which cancels badly for small
  gamma*h in float32 and is slow for batched eigh on GPUs.
- v_n has a closed-form inverse CDF. u_n is obtained from the CDF F(u) = 1 - psi(g(1-u)) / psi(g), g = gamma*h,
  psi(z) = e^{-z} - 1 + z, by a few Newton steps on psi(z) = (1 - U) psi(g) (convex, started left of the root).
"""

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from samp_via_opt.samplers.base import BaseSampler
from samp_via_opt.samplers.ulmc import ULMCState
from samp_via_opt.samplers.rm_ulmc import _t_minus_tanh_t
from samp_via_opt.targets.base import BaseTarget


def _psi(z: jax.Array) -> jax.Array:
    """psi(z) = exp(-z) - 1 + z >= 0."""
    return z + jnp.expm1(-z)


def _sample_u(U: jax.Array, gh: jax.Array, newton_steps: int = 3) -> jax.Array:
    """
    Inverse CDF sampling of q_u(u) ~ 1 - exp(-gh (1-u)) on [0, 1] with a few Newton steps.

    Args:
        U: Uniform sample in [0, 1).
        gh: gamma * h.
        newton_steps: Number of Newton iterations (3 gives ~1e-11 / ~1e-5 accuracy in float64 / float32).

    Returns:
        u in [0, 1].
    """
    c = (1.0 - U) * _psi(gh)
    z = jnp.sqrt(2.0 * c)  # psi(z) ~ z^2 / 2, starts left of the root
    tiny = jnp.finfo(c.dtype).tiny
    for _ in range(newton_steps):
        z = z - (_psi(z) - c) / jnp.maximum(-jnp.expm1(-z), tiny)
        z = jnp.clip(z, 0.0, gh)
    return 1.0 - z / gh


def _sample_v(U: jax.Array, gh: jax.Array) -> jax.Array:
    """Exact inverse CDF sampling of q_v(v) ~ exp(-gh (1-v)) on [0, 1]."""
    v = 1.0 + jnp.log1p((1.0 - U) * jnp.expm1(-gh)) / gh
    return jnp.clip(v, 0.0, 1.0)  # guards against rounding at U = 0


def _segment_noise(
    T: jax.Array, gamma: jax.Array, z: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Jointly Gaussian pair (G, B) for a Brownian segment of duration T.

    Args:
        T: Segment duration (scalar, >= 0; T = 0 yields zeros, no NaNs).
        gamma: Friction coefficient.
        z: Standard normals of shape (2, d).

    Returns:
        G, B, each of shape (d,).
    """
    x = 0.5 * gamma * T
    l11 = jnp.sqrt(-jnp.expm1(-2.0 * gamma * T) / (2.0 * gamma))
    l21 = jnp.sqrt(2.0 / gamma * jnp.tanh(x))
    l22 = jnp.sqrt(2.0 / gamma * _t_minus_tanh_t(x))
    return l11 * z[0], l21 * z[0] + l22 * z[1]


class RMULMCAdv(BaseSampler[ULMCState]):
    """
    Randomized Midpoint Underdamped Langevin Monte Carlo (RM-ULMC) sampler based on Altschuler et al. (2025).

    Attributes:
        step_size: Step size for the Langevin dynamics.
        gamma: Friction coefficient.
        beta: Temperature parameter controlling the noise level (1 in the paper).
        dtype: Used floating point precision, either float32 or float64.
    """

    step_size: float
    gamma: float | None = 2.0
    beta: float = 1.0
    dtype: DTypeLike = jnp.float32

    def init_state(self, x0: jax.Array, key: jax.Array | None = None) -> ULMCState:
        x0 = jnp.asarray(x0, dtype=self.dtype)
        if key is None:
            v0 = jnp.zeros_like(x0)  # deterministic fallback
        else:
            # Sample from stationary velocity distribution N(0, beta * I)
            beta = jnp.asarray(self.beta, dtype=self.dtype)
            v0 = jnp.sqrt(beta) * jax.random.normal(
                key, shape=x0.shape, dtype=self.dtype
            )
        return ULMCState(x=x0, v=v0)

    def step(
        self, state: ULMCState, target: BaseTarget, key: jax.Array
    ) -> tuple[ULMCState, dict[str, jax.Array]]:
        dtype = state.x.dtype
        d = state.x.shape[0]
        h = jnp.asarray(self.step_size, dtype=dtype)
        gamma = jnp.asarray(self.gamma, dtype=dtype)
        beta = jnp.asarray(self.beta, dtype=dtype)
        sqrt_beta = jnp.sqrt(beta)

        mid_key, noise_key = jax.random.split(key)

        # random midpoints t_u = u h, t_v = v h
        U_u, U_v = jax.random.uniform(mid_key, shape=(2,), dtype=dtype)
        t_u = _sample_u(U_u, gamma * h) * h
        t_v = _sample_v(U_v, gamma * h) * h

        # Noise on the three independent segments [0, a], [a, b], [b, h]
        a, b = jnp.minimum(t_u, t_v), jnp.maximum(t_u, t_v)
        z = jax.random.normal(noise_key, shape=(6, d), dtype=dtype)
        G1, B1 = _segment_noise(a, gamma, z[0:2])
        G2, B2 = _segment_noise(b - a, gamma, z[2:4])
        G3, B3 = _segment_noise(h - b, gamma, z[4:6])

        # E_t = int_0^t e^{-gamma (t-s)} dB_s at t = b and t = h
        E_b = jnp.exp(-gamma * (b - a)) * G1 + G2
        E_h = jnp.exp(-gamma * (h - b)) * E_b + G3

        # xi1(t) = sqrt(2/gamma) (B_t - E_t), xi2(h) = sqrt(2 gamma) E_h (all for beta = 1)
        s = jnp.sqrt(2.0 / gamma)
        xi1_a = s * (B1 - G1)
        xi1_b = s * (B1 + B2 - E_b)
        xi1_h = s * (B1 + B2 + B3 - E_h)
        xi2_h = jnp.sqrt(2.0 * gamma) * E_h

        # assign the xi1 values to the matching midpoint (which midpoint comes first is random)
        u_first = t_u <= t_v
        xi1_u = jnp.where(u_first, xi1_a, xi1_b)
        xi1_v = jnp.where(u_first, xi1_b, xi1_a)

        def c1(t):  # (1 - e^{-gamma t}) / gamma
            return -jnp.expm1(-gamma * t) / gamma

        def c2(t):  # (t - c1(t)) / gamma
            return (t - c1(t)) / gamma

        grad = jnp.asarray(target.grad_log_prob(state.x), dtype=dtype)

        x_mid_u = (
            state.x + c1(t_u) * state.v + beta * c2(t_u) * grad + sqrt_beta * xi1_u
        )
        x_mid_v = (
            state.x + c1(t_v) * state.v + beta * c2(t_v) * grad + sqrt_beta * xi1_v
        )

        grad_mid_u = jnp.asarray(target.grad_log_prob(x_mid_u), dtype=dtype)
        grad_mid_v = jnp.asarray(target.grad_log_prob(x_mid_v), dtype=dtype)

        x_new = (
            state.x + c1(h) * state.v + beta * c2(h) * grad_mid_u + sqrt_beta * xi1_h
        )
        v_new = (
            jnp.exp(-gamma * h) * state.v
            + beta * c1(h) * grad_mid_v
            + sqrt_beta * xi2_h
        )

        new_state = ULMCState(x=x_new, v=v_new)
        aux_info = {"grad_norm": jnp.linalg.norm(grad)}

        return new_state, aux_info
