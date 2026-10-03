"""
Implementation of Randomized Midpoint Unadjusted Langevin Algorithm (RMULA) sampler.
Introduces a randomized midpoint discretization of the Langevin dynamics to improve convergence compared to standard ULA.
"""

import jax
import jax.numpy as jnp
from typing import Optional
from samp_via_opt.samplers.base import BaseSampler
from samp_via_opt.samplers.ula import ULAState
from samp_via_opt.targets.base import BaseTarget


class RMULA(BaseSampler[ULAState]):
    """
    Randomized Midpoint Unadjusted Langevin Algorithm (RMULA) sampler.

    Attributes:
        step_size: Step size for the Langevin dynamics.
    """

    step_size: float

    def init_state(self, x0: jax.Array, key: Optional[jax.Array] = None) -> ULAState:
        return ULAState(x=jnp.asarray(x0))

    def step(
        self, state: ULAState, target: BaseTarget, key: jax.Array
    ) -> tuple[ULAState, dict[str, jax.Array]]:
        h = self.step_size

        noise_key_mid, noise_key_end, midpoint_key = jax.random.split(key, 3)

        midpoint = jax.random.uniform(midpoint_key, shape=())
        noise_mid = jax.random.normal(noise_key_mid, shape=state.x.shape)
        noise_end = jax.random.normal(noise_key_end, shape=state.x.shape)

        grad = target.grad_log_prob(state.x)

        x_mid = state.x + h * midpoint * grad + jnp.sqrt(2.0 * h * midpoint) * noise_mid
        grad_mid = target.grad_log_prob(x_mid)

        x_new = (
            state.x
            + h * grad_mid
            + jnp.sqrt(2.0 * h * midpoint) * noise_mid
            + jnp.sqrt(2.0 * (1 - midpoint) * h) * noise_end
        )

        new_state = ULAState(x=x_new)
        aux_info = {"grad_norm": jnp.linalg.norm(grad)}

        return new_state, aux_info
