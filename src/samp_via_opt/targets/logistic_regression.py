"""
Bayesian logistic regression posterior with an anisotropic Gaussian prior.

Real-data counterpart to GaussianTarget: log-concave and smooth, but not Gaussian, and the
posterior mean / covariance are not known in closed form (they are estimated once by a
reference run, see experiments/bayesian_logistic_regression.py).
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from samp_via_opt.targets.base import BaseTarget


class LogisticRegressionTarget(BaseTarget):
    """
    Posterior p(w | X, y) ∝ prod_i Bernoulli(y_i | sigmoid(x_i^T w)) * N(w | 0, diag(prior_var)),
    i.e. the potential is  Psi(w) = sum_i [log(1 + exp(x_i^T w)) - y_i x_i^T w] + 1/2 w^T Sigma^{-1} w.

    Coordinates are rescaled (w = sqrt(L) * beta) so that the global smoothness bound is L = 1,
    exactly as for the Gaussian targets of the main experiments. Step-size and friction grids are
    therefore directly comparable. Error metrics are invariant to this rescaling (W2 scales by 1/s and √(1/m) scales by 1/s).

    Attributes:
        X (N, d): Design matrix, already multiplied by `scale`.
        y (N,): Binary labels in {0, 1}.
        prior_prec (d,): Prior precisions 1 / sigma_i^2 in the rescaled coordinates.
        mean (d,): Posterior mean. Placeholder until set via `with_moments` (needed by the evaluator).
        cov (d, d): Posterior covariance. Placeholder until set via `with_moments`.
        L: Global smoothness bound  lambda_max(X^T X) / 4 + max_i prior_prec_i  (= 1 after rescaling).
        m: Global strong-convexity constant  min_i prior_prec_i  (log-likelihood is convex, prior gives m).
        scale: Rescaling factor, original coefficients are beta = scale * w.
    """

    X: jax.Array
    y: jax.Array
    prior_prec: jax.Array
    mean: jax.Array
    cov: jax.Array
    L: float
    m: float
    scale: float

    def __init__(
        self,
        X: jax.Array,
        y: jax.Array,
        prior_var: jax.Array,
        normalize: bool = True,
    ):
        X = jnp.asarray(X)
        prior_prec = 1.0 / jnp.asarray(prior_var)

        # sigmoid'(u) <= 1/4  =>  Hessian <= X^T X / 4 + Sigma^{-1}
        def smoothness(Xm, prec):
            return float(jnp.linalg.eigvalsh(Xm.T @ Xm)[-1] / 4.0 + jnp.max(prec))

        # rescale such that L=1, since L of raw data in general not 1
        # s = 1/\sqrt(L_0) where L_0 is the smoothness of the original target
        s = 1.0 / math.sqrt(smoothness(X, prior_prec)) if normalize else 1.0
        self.scale = s
        self.X = s * X
        self.y = jnp.asarray(y, dtype=self.X.dtype)
        self.prior_prec = s**2 * prior_prec  # 1/2 β^T Σ^{-1} β = 1/2 s^2 w^T Σ^{-1} w
        self.L = smoothness(
            self.X, self.prior_prec
        )  # s^2 λ_max(XᵀX)/4 + s^2 prec_max = s^2 L_0 = 1
        self.m = float(jnp.min(self.prior_prec))

        d = X.shape[1]
        self.mean = jnp.zeros(d, dtype=self.X.dtype)
        self.cov = jnp.eye(d, dtype=self.X.dtype)

    @property
    def dim(self) -> int:
        return self.X.shape[1]

    def with_moments(
        self, mean: jax.Array, cov: jax.Array
    ) -> "LogisticRegressionTarget":
        """Return a copy whose .mean / .cov are the (estimated) posterior moments."""
        return eqx.tree_at(lambda t: (t.mean, t.cov), self, (mean, cov))

    def log_prob(self, x: jax.Array) -> jax.Array:
        """Unnormalised log posterior, i.e. -Psi(x) up to a constant."""
        z = self.X @ x
        loglik = jnp.sum(self.y * z - jax.nn.softplus(z))
        logprior = -0.5 * jnp.sum(self.prior_prec * x**2)
        return loglik + logprior

    def grad_log_prob(self, x: jax.Array) -> jax.Array:
        return -(self.X.T @ (jax.nn.sigmoid(self.X @ x) - self.y) + self.prior_prec * x)

    def hessian_potential(self, x: jax.Array) -> jax.Array:
        """Hessian of the potential Psi (= negative Hessian of log_prob), shape (d, d)."""
        p = jax.nn.sigmoid(self.X @ x)
        return self.X.T @ ((p * (1.0 - p))[:, None] * self.X) + jnp.diag(
            self.prior_prec
        )
