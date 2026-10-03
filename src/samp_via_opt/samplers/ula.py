"""
Implementation of Unadjusted Langevin Algorithm (ULA) sampler using a simple first-order discretization.
Represents the sampling equivalent of gradient descent in optimization. The sampler is based on the Langevin dynamics.
"""

from typing import Optional
import jax
import jax.numpy as jnp
from samp_via_opt.samplers.base import BaseSampler, SamplerState
from samp_via_opt.targets.base import BaseTarget


class ULAState(SamplerState):
    """
    State for the Unadjusted Langevin Algorithm (ULA) sampler.

    Attributes:
        x: Current position in the target space.
    """

    pass


class ULA(BaseSampler[ULAState]):
    """
    Unadjusted Langevin Algorithm (ULA) sampler.

    Attributes:
        step_size: Step size for the Langevin dynamics.
    """

    step_size: float

    def init_state(self, x0: jax.Array, key: Optional[jax.Array] = None) -> ULAState:
        return ULAState(x=jnp.asarray(x0))

    def step(
        self, state: ULAState, target: BaseTarget, key: jax.Array
    ) -> tuple[ULAState, dict[str, jax.Array]]:
        grad = target.grad_log_prob(state.x)
        noise = jax.random.normal(key, shape=state.x.shape)
        x_new = state.x + self.step_size * grad + jnp.sqrt(2.0 * self.step_size) * noise

        new_state = ULAState(x=x_new)
        aux_info = {"grad_norm": jnp.linalg.norm(grad)}

        return new_state, aux_info
