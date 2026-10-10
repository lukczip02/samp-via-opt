import jax
import jax.numpy as jnp
from samp_via_opt.targets.base import BaseTarget


class Rosenbrock2DTarget(BaseTarget):
    a: float = 1.0
    b: float = 100.0
    constant_hessian: bool = False

    @property
    def log_partition_const(self) -> float:
        # log(sqrt(b) / pi) = 0.5 * log(b) - log(pi)
        return float(0.5 * jnp.log(self.b) - jnp.log(jnp.pi))

    def log_prob(self, x: jax.Array) -> jax.Array:
        psi = (self.a - x[0]) ** 2 + self.b * (x[1] - x[0] ** 2) ** 2
        return -psi + self.log_partition_const

    def sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> jax.Array:
        """
        Generates exact ground truth samples via ancestral sampling:
            X1 ~ N(a, 1/2)
            X2 ~ N(X1^2, 1/(2b))
        """
        key1, key2 = jax.random.split(key)
        z1 = jax.random.normal(key1, shape)
        z2 = jax.random.normal(key2, shape)

        x1 = self.a + z1 / jnp.sqrt(2.0)
        x2 = x1**2 + z2 / jnp.sqrt(2.0 * self.b)
        return jnp.stack([x1, x2], axis=-1)

    @property
    def mean(self) -> jax.Array:
        """
        Exact analytical mean:
            E[X1] = a
            E[X2] = E[X1^2] = Var(X1) + E[X1]^2 = 0.5 + a^2
        """
        return jnp.array([self.a, self.a**2 + 0.5])

    @property
    def cov(self) -> jax.Array:
        """
        Exact analytical covariance matrix:
            Var(X1) = 0.5
            Cov(X1, X2) = a
            Var(X2) = 1/(2b) + 2*a^2 + 0.5
        """
        var_x1 = 0.5
        cov_x1_x2 = self.a
        var_x2 = (1.0 / (2.0 * self.b)) + 2.0 * (self.a**2) + 0.5
        return jnp.array([[var_x1, cov_x1_x2], [cov_x1_x2, var_x2]])
