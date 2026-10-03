from abc import ABC, abstractmethod

import jax
import equinox as eqx


class BaseTarget(eqx.Module, ABC):
    """
    Base class for target distributions. Subclasses implement specific target distributions.
    """

    @abstractmethod
    def log_prob(self, x: jax.Array) -> jax.Array:
        """
        Compute the log probability of the target distribution at position x.

        Args:
            x: Position in the target space.

        Returns:
            Log probability of the target distribution at x.
        """
        pass

    def grad_log_prob(self, x: jax.Array) -> jax.Array:
        """
        Compute the gradient of the log probability of the target distribution at position x.

        Args:
            x: Position in the target space.

        Returns:
            Gradient of the log probability of the target distribution at x.
        """
        return jax.grad(self.log_prob)(x)
