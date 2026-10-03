import jax
import equinox as eqx
from typing import Generic, Optional, TypeVar
from abc import ABC, abstractmethod
from samp_via_opt.targets.base import BaseTarget


class SamplerState(eqx.Module):
    """
    Base class for sampler state, updated at each step of the chain.
    Subclasses add additional state variables.
    """

    x: jax.Array


StateType = TypeVar("StateType", bound=SamplerState)


class BaseSampler(eqx.Module, ABC, Generic[StateType]):
    """
    Base class for samplers. Subclasses implement specific sampling algorithms.
    """

    @abstractmethod
    def init_state(self, x0: jax.Array, key: Optional[jax.Array] = None) -> StateType:
        """
        Initialize the sampler state.

        Args:
            x0: Initial position in the target space.
            key: Optional JAX random key for stochastic initialization.

        Returns:
            An instance of the sampler state.
        """
        pass

    @abstractmethod
    def step(
        self, state: StateType, target: BaseTarget, key: jax.Array
    ) -> tuple[StateType, dict[str, jax.Array]]:
        """
        Forwards the chain of the sampler by one step.

        Args:
            state: Current sampler state.
            target: Target distribution to sample from.
            key: JAX random key for stochastic updates.

        Returns:
            A tuple containing the new sampler state and a dictionary of auxiliary information.
        """
        pass

    # def sample(self, target: BaseTarget, init_state: StateType, key: jax.Array, num_steps: int, save_aux: bool = False) -> tuple[StateType, jax.Array, dict[str, jax.Array]]:
    #     step_keys = jax.random.split(key, num_steps)

    #     def scan_fn(carrier_state: StateType, step_key: jax.Array):
    #         new_state, aux = self.step(carrier_state, target, step_key)
    #         out_aux = aux if save_aux else {}
    #         return new_state, (new_state.x, out_aux)

    #     final_state, (xs, auxs) = jax.lax.scan(scan_fn, init_state, step_keys)
    #     return final_state, xs, auxs
