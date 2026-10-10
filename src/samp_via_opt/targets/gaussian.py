"""
Multivariate Gaussian target constructed to investigate scaling with condition number kappa and dimension d.
"""

import jax
import jax.numpy as jnp
from samp_via_opt.targets.base import BaseTarget


class GaussianTarget(BaseTarget):
    """
    Multivariate Gaussian target distribution.

    Attributes:
        mean: Mean vector of the Gaussian distribution.
        cov: Covariance matrix of the Gaussian distribution.
        inv_cov: Inverse of the covariance matrix (also precision matrix).
        logdet: Log-determinant of the covariance matrix.
        L: Smoothness parameter (largest eigenvalue of the precision matrix).
        m: Strong convexity parameter (smallest eigenvalue of the precision matrix).
        B1: Precomputation of B1 for TIPreL to avoid redundancy.
        constant_hessian: Hessian of the Gaussian does not depend on x, therefore B1 can be precomputed once.
    """

    mean: jax.Array
    cov: jax.Array
    inv_cov: jax.Array
    logdet: jax.Array
    L: float
    m: float
    B1: jax.Array
    constant_hessian: bool = True

    def __init__(self, mean: jax.Array, cov: jax.Array, eps: float = 1e-6):
        self.mean = jnp.asarray(mean)
        self.cov = jnp.asarray(cov)
        self.inv_cov = jnp.linalg.inv(self.cov)

        precision_eigenvals = jnp.linalg.eigvalsh(self.inv_cov)
        self.L = float(jnp.max(precision_eigenvals))
        self.m = float(jnp.min(precision_eigenvals))

        _, self.logdet = jnp.linalg.slogdet(self.cov)

        self.B1 = self.cov

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    def log_prob(self, x: jax.Array) -> jax.Array:
        diff = x - self.mean

        return -0.5 * (
            jnp.dot(diff, jnp.dot(self.inv_cov, diff))
            + self.dim * jnp.log(2.0 * jnp.pi)
            + self.logdet
        )

    def grad_log_prob(self, x: jax.Array):
        return -self.inv_cov @ (x - self.mean)
