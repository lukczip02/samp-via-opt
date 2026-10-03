import jax
import jax.numpy as jnp
from samp_via_opt.samplers.base import BaseSampler
from samp_via_opt.samplers.ulmc import ULMCState
from samp_via_opt.targets.base import BaseTarget


class RMULMCAdv(BaseSampler[ULMCState]):
    """Randomized Midpoint Unadjusted Langevin Monte Carlo (RMULMC) sampler based on Altschuler et al. (2025)."""

    step_size: float
    alpha: float = 2.0  # gamma represents alpha, beta is defaulted to 1.0
    beta: float = 1.0

    def init_state(self, x0: jax.Array, key: jax.Array | None = None) -> ULMCState:
        x0 = jnp.asarray(x0)
        if key is None:
            v0 = jnp.zeros_like(x0)  # deterministic fallback
        else:
            # Sample from stationary velocity distribution N(0, beta * I)
            v0 = jnp.sqrt(self.beta) * jax.random.normal(key, shape=x0.shape)
        return ULMCState(x=x0, v=v0)

    def step(
        self, state: ULMCState, target: BaseTarget, key: jax.Array
    ) -> tuple[ULMCState, dict[str, jax.Array]]:
        def sample_v(key, gamma, h, shape=()):
            U = jax.random.uniform(key, shape=shape)
            gh = gamma * h
            term = (1.0 - U) * jnp.expm1(-gh)
            return 1.0 + jnp.log1p(term) / jnp.maximum(gh, 1e-12)

        def sample_u(key, gamma, h, shape=(), steps=4):
            U = jax.random.uniform(key, shape=shape)
            gh = gamma * h

            def u_taylor(_):
                R = jnp.sqrt(jnp.maximum(1.0 - U, 1e-12))
                # 2nd-order Taylor fallback for small x = gamma * h
                x = gh
                u = (1.0 - R) * (1.0 + (x / 6.0) * R - (x**2 / 36.0) * (1.0 - R) * R)
                return u

            def u_newton(_):
                exp_neg_gh = jnp.exp(-gh)
                denom_CDF = 1.0 + jnp.expm1(-gh) / gh
                denom_pdf = h + jnp.expm1(-gh) / gamma

                u_val = U
                for _ in range(steps):
                    F_val = (
                        u_val - (jnp.exp(-gamma * (1.0 - u_val) * h) - exp_neg_gh) / gh
                    ) / jnp.maximum(denom_CDF, 1e-12)
                    f_val = (
                        h
                        * (-jnp.expm1(-gamma * (1.0 - u_val) * h))
                        / jnp.maximum(denom_pdf, 1e-12)
                    )
                    u_val = u_val - (F_val - U) / jnp.maximum(f_val, 1e-7)
                    u_val = jnp.clip(u_val, 0.0, 1.0)
                return u_val

            return jax.lax.cond(gh < 1e-2, u_taylor, u_newton, operand=None)

        def compute_noise_covariance(t_u, t_v, h, gamma):
            """Computes the exact 4x4 noise covariance matrix for (xi1(tu), xi1(tv), xi1(h), xi2(h))."""

            def cov_xi1_xi1(ta, tb):
                tm = jnp.minimum(ta, tb)
                term1 = tm
                term2 = (jnp.exp(-gamma * (ta - tm)) - jnp.exp(-gamma * ta)) / gamma
                term3 = (jnp.exp(-gamma * (tb - tm)) - jnp.exp(-gamma * tb)) / gamma
                term4 = (
                    jnp.exp(-gamma * (ta + tb - 2.0 * tm)) - jnp.exp(-gamma * (ta + tb))
                ) / (2.0 * gamma)
                return (2.0 / gamma) * (term1 - term2 - term3 + term4)

            def cov_xi1_xi2_h(ta):
                return (1.0 / gamma) * (
                    jnp.exp(-gamma * (h - ta))
                    - 2.0 * jnp.exp(-gamma * h)
                    + jnp.exp(-gamma * (ta + h))
                )

            c11 = cov_xi1_xi1(t_u, t_u)
            c12 = cov_xi1_xi1(t_u, t_v)
            c13 = cov_xi1_xi1(t_u, h)
            c14 = cov_xi1_xi2_h(t_u)

            c22 = cov_xi1_xi1(t_v, t_v)
            c23 = cov_xi1_xi1(t_v, h)
            c24 = cov_xi1_xi2_h(t_v)

            c33 = cov_xi1_xi1(h, h)
            c34 = cov_xi1_xi2_h(h)

            c44 = -jnp.expm1(-2.0 * gamma * h)

            Cov = jnp.array(
                [
                    [c11, c12, c13, c14],
                    [c12, c22, c23, c24],
                    [c13, c23, c33, c34],
                    [c14, c24, c34, c44],
                ]
            )
            return 0.5 * (Cov + Cov.T)
            # return Cov + 1e-8 * jnp.eye(4)  # Small diagonal jitter for Cholesky stability

        d = state.x.shape[0]
        h = self.step_size
        gamma = self.alpha

        u_mid_key, v_mid_key, noise_key = jax.random.split(key, 3)

        u_mid = jnp.clip(sample_u(u_mid_key, gamma, h, shape=()), 1e-5, 1.0 - 1e-5)
        v_mid = jnp.clip(sample_v(v_mid_key, gamma, h, shape=()), 1e-5, 1.0 - 1e-5)

        t_u = u_mid * h
        t_v = v_mid * h

        Cov = compute_noise_covariance(t_u, t_v, h, gamma)

        # Spectral decomposition for robust matrix square root
        evals, evecs = jnp.linalg.eigh(Cov)
        evals = jnp.maximum(evals, 1e-12)
        L = evecs @ jnp.diag(jnp.sqrt(evals))

        z = jax.random.normal(noise_key, shape=(4, d))
        noise = L @ z

        noise1_mid_u = noise[0]  # xi^(1)_{nh, u_n h}
        noise1_mid_v = noise[1]  # xi^(1)_{nh, v_n h}
        noise1_end = noise[2]  # xi^(1)_{nh, h}
        noise2_end = noise[3]  # xi^(2)_{nh, h}

        grad = target.grad_log_prob(state.x)

        one_minus_exp_tu = -jnp.expm1(-gamma * t_u)
        one_minus_exp_tv = -jnp.expm1(-gamma * t_v)
        one_minus_exp_h = -jnp.expm1(-gamma * h)

        x_mid_u = (
            state.x
            + one_minus_exp_tu / gamma * state.v
            + (1.0 / gamma) * (t_u - one_minus_exp_tu / gamma) * grad
            + noise1_mid_u
        )

        x_mid_v = (
            state.x
            + one_minus_exp_tv / gamma * state.v
            + (1.0 / gamma) * (t_v - one_minus_exp_tv / gamma) * grad
            + noise1_mid_v
        )

        grad_mid_u = target.grad_log_prob(x_mid_u)
        grad_mid_v = target.grad_log_prob(x_mid_v)

        x_new = (
            state.x
            + one_minus_exp_h / gamma * state.v
            + (1.0 / gamma) * (h - one_minus_exp_h / gamma) * grad_mid_u
            + noise1_end
        )

        v_new = (
            jnp.exp(-gamma * h) * state.v
            + (1.0 / gamma) * one_minus_exp_h * grad_mid_v
            + noise2_end
        )

        new_state = ULMCState(x=x_new, v=v_new)
        aux_info = {"grad_norm": jnp.linalg.norm(grad)}

        return new_state, aux_info
