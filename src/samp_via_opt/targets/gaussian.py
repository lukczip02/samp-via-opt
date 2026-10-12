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
        cov (d,d): Covariance matrix of the Gaussian distribution.
        inv_cov: Inverse of the covariance matrix. (d,d) in general case, (d,) if cov is diagonal.
        logdet: Log-determinant of the covariance matrix.
        L: Smoothness parameter (largest eigenvalue of the precision matrix).
        m: Strong convexity parameter (smallest eigenvalue of the precision matrix).
        B1: Precomputation of B1 for TIPreL to avoid redundancy.
        diagonal_cov: Use simplified computations in case cov is diagonal.
        constant_hessian: Hessian of the Gaussian does not depend on x, therefore B1 can be precomputed once.
    """

    mean: jax.Array
    cov: jax.Array
    inv_cov: jax.Array
    logdet: jax.Array
    L: float
    m: float
    B1: jax.Array
    diagonal_cov: bool
    constant_hessian: bool = True

    def __init__(self, mean: jax.Array, cov: jax.Array, diag: bool = False):
        self.diagonal_cov = diag
        self.mean = jnp.asarray(mean)
        self.cov = jnp.asarray(cov)  # (d, d)
        if self.diagonal_cov:
            self.inv_cov = 1.0 / jnp.diag(self.cov)  # (d, )
            self.logdet = jnp.sum(jnp.log(jnp.diag(self.cov)))
            inv_cov_evals = self.inv_cov
        else:
            self.inv_cov = jnp.linalg.inv(self.cov)
            inv_cov_evals = jnp.linalg.eigvalsh(self.inv_cov)
            _, self.logdet = jnp.linalg.slogdet(self.cov)

        self.L = float(jnp.max(inv_cov_evals))
        self.m = float(jnp.min(inv_cov_evals))

        self.B1 = self.cov

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    def log_prob(self, x: jax.Array) -> jax.Array:
        diff = x - self.mean
        quad = (
            jnp.sum(self.inv_cov * diff**2)
            if self.diagonal_cov
            else jnp.dot(diff, self.inv_cov @ diff)
        )
        return -0.5 * (quad + self.dim * jnp.log(2.0 * jnp.pi) + self.logdet)

    def grad_log_prob(self, x: jax.Array):
        if self.diagonal_cov:
            return -self.inv_cov * (x - self.mean)
        else:
            return -self.inv_cov @ (x - self.mean)
