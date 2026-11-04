"""
Randomized Midpoint Underdamped Langevin Monte Carlo (RM-ULMC) of Shen and Lee (2019), derived directly
for a general friction gamma > 0 from the integral formulation (no time rescaling).

Dynamics (temperature u, friction gamma, g = grad log pi):
    dx = v dt,    dv = (-gamma v + u g(x)) dt + sqrt(2 gamma u) dB,
with the exact solution over a time t
    x_t = x + c1(t) v + u int_0^t c1(t-s) g(x_s) ds + sqrt(2u/gamma) int_0^t (1 - e^{-gamma (t-s)}) dB_s
    v_t = e^{-gamma t} v + u int_0^t e^{-gamma (t-s)} g(x_s) ds + sqrt(2 gamma u) int_0^t e^{-gamma (t-s)} dB_s
with c1(t) = (1 - e^{-gamma t}) / gamma. The gradient integrals are estimated with one uniform random midpoint
alpha ~ U(0, 1) (as in Shen and Lee, in contrast to the weighted midpoints of rm_ulmc_adv.py):
    x_mid  = x + c1(alpha h) v + u c2(alpha h) g(x) + sqrt(u) xi1(alpha h),    c2(t) = (t - c1(t)) / gamma
    x_new  = x + c1(h) v + u h c1((1-alpha) h) g(x_mid) + sqrt(u) xi1(h)
    v_new  = e^{-gamma h} v + u h e^{-gamma (1-alpha) h} g(x_mid) + sqrt(u) xi2(h)
For gamma = 2 this is Algorithm 3 of Shen and Lee. For general gamma it is mathematically identical to
rm_ulmc.py (which rescales time by gamma/2), up to the scaling of v: rm_ulmc.py stores v' = 2 v / gamma, here v is
the physical velocity (stationary law N(0, u), as in ULMC and RMULMCAdv). Positions x agree for equal random numbers.

Noise: built from two independent Brownian segments of duration alpha*h and (1-alpha)*h with the closed-form
Cholesky factor of rm_ulmc_adv._segment_noise (float32 safe, no determinants):
    xi1(alpha h) = sqrt(2/gamma) (B1 - G1),   E = e^{-gamma (1-alpha) h} G1 + G2
    xi1(h)       = sqrt(2/gamma) (B1 + B2 - E),   xi2(h) = sqrt(2 gamma) E
"""

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from samp_via_opt.samplers.base import BaseSampler
from samp_via_opt.samplers.ulmc import ULMCState
from samp_via_opt.samplers.rm_ulmc_adv import _segment_noise
from samp_via_opt.targets.base import BaseTarget


class RMULMCGamma(BaseSampler[ULMCState]):
    """
    Randomized Midpoint ULMC (Shen and Lee, 2019) with general friction gamma.

    Attributes:
        step_size: Step size h of the Langevin dynamics (physical time).
        gamma: Friction coefficient, gamma = 2.0 is Shen and Lee.
        u: Temperature parameter controlling the noise level (stationary law of v is N(0, u)).
        dtype: Used floating point precision, either float32 or float64.
    """

    step_size: float
    gamma: float | None = 2.0
    u: float = 1.0
    dtype: DTypeLike = jnp.float32

    def init_state(self, x0: jax.Array, key: jax.Array | None = None) -> ULMCState:
        x0 = jnp.asarray(x0, dtype=self.dtype)
        if key is None:
            v0 = jnp.zeros_like(x0)  # deterministic fallback
        else:
            # Sample from stationary velocity distribution N(0, u * I)
            u = jnp.asarray(self.u, dtype=self.dtype)
            v0 = jnp.sqrt(u) * jax.random.normal(key, shape=x0.shape, dtype=self.dtype)
        return ULMCState(x=x0, v=v0)

    def step(
        self, state: ULMCState, target: BaseTarget, key: jax.Array
    ) -> tuple[ULMCState, dict[str, jax.Array]]:
        dtype = state.x.dtype
        d = state.x.shape[0]
        h = jnp.asarray(self.step_size, dtype=dtype)
        gamma = jnp.asarray(self.gamma, dtype=dtype)
        u = jnp.asarray(self.u, dtype=dtype)
        sqrt_u = jnp.sqrt(u)

        midpoint_key, noise1_key, noise2_key = jax.random.split(key, 3)

        # alpha ~ U[0, 1); 1 - alpha is exact in floating point, unlike h - alpha * h
        alpha = jax.random.uniform(midpoint_key, shape=(), dtype=dtype)
        ah = alpha * h
        dh = (1.0 - alpha) * h

        # noise on the two independent segments [0, alpha h] and [alpha h, h]
        z1 = jax.random.normal(noise1_key, shape=(2, d), dtype=dtype)
        z2 = jax.random.normal(noise2_key, shape=(2, d), dtype=dtype)
        G1, B1 = _segment_noise(ah, gamma, z1)
        G2, B2 = _segment_noise(dh, gamma, z2)
        E = jnp.exp(-gamma * dh) * G1 + G2

        s = jnp.sqrt(2.0 / gamma)
        xi1_mid = s * (B1 - G1)
        xi1_end = s * (B1 + B2 - E)
        xi2_end = jnp.sqrt(2.0 * gamma) * E

        def c1(t):  # (1 - e^{-gamma t}) / gamma
            return -jnp.expm1(-gamma * t) / gamma

        def c2(t):  # (t - c1(t)) / gamma
            return (t - c1(t)) / gamma

        grad = jnp.asarray(target.grad_log_prob(state.x), dtype=dtype)

        x_mid = state.x + c1(ah) * state.v + u * c2(ah) * grad + sqrt_u * xi1_mid

        grad_mid = jnp.asarray(target.grad_log_prob(x_mid), dtype=dtype)

        x_new = state.x + c1(h) * state.v + u * h * c1(dh) * grad_mid + sqrt_u * xi1_end
        v_new = (
            jnp.exp(-gamma * h) * state.v
            + u * h * jnp.exp(-gamma * dh) * grad_mid
            + sqrt_u * xi2_end
        )

        new_state = ULMCState(x=x_new, v=v_new)
        aux_info = {"grad_norm": jnp.linalg.norm(grad)}

        return new_state, aux_info
