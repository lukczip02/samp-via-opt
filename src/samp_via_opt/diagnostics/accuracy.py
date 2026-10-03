from functools import partial
import jax
import jax.numpy as jnp


def compute_moment_errors(
    trajectory: jax.Array, true_mean: jax.Array, true_cov: jax.Array
) -> dict[str, float]:
    """
    Computes mean error and Frobenius norm error of covariance.

    Args:
        trajectory (num_samples, d): Full d-dimensional state history of the Markov chain across all iterations.
        true_mean (d,): True mean of the target distribution.
        true_cov (d, d): True covariance of the target distribution.

    Returns:
        Dictionary containing mean error and covariance error.
    """
    emp_mean = jnp.mean(trajectory, axis=0)
    emp_cov = jnp.cov(trajectory, rowvar=False)

    mean_err = jnp.linalg.norm(emp_mean - true_mean)
    cov_err = jnp.linalg.norm(emp_cov - true_cov, ord="fro")
    return {"mean_error": float(mean_err), "cov_error": float(cov_err)}


def wasserstein_1d(u: jax.Array, v: jax.Array) -> jax.Array:
    """
    Computes 1D Wasserstein distance between two sets of 1D samples.

    Args:
        u (N,): 1D array of samples from distribution U.
        v (M,): 1D array of samples from distribution V.

    Returns:
        Scalar Wasserstein distance W_1(U, V).
    """
    n = u.shape[0]
    m = v.shape[0]

    u_sorted = jnp.sort(u)
    v_sorted = jnp.sort(v)

    if n == m:
        return jnp.mean(jnp.abs(u_sorted - v_sorted))
    else:  # Use quantile interpolation for unequal sample sizes
        num_quantiles = max(int(n), int(m))
        grid = jnp.linspace(0.0, 1.0, num=num_quantiles)
        u_quantiles = jnp.quantile(u_sorted, grid)
        v_quantiles = jnp.quantile(v_sorted, grid)
        return jnp.mean(jnp.abs(u_quantiles - v_quantiles))


@partial(jax.jit, static_argnames=("num_projections",))
def sliced_wasserstein_distance(
    x: jax.Array,
    y: jax.Array,
    key: jax.Array,
    num_projections: int = 64,
) -> jax.Array:
    """
    Computes Sliced 2-Wasserstein distance (SW_2) between d-dimensional sample sets.

    Args:
        x (N, d): Sample matrix.
        y (M, d): Sample or target reference matrix.
        key (L, d): PRNGKey for sampling random projection directions on S^{d-1}.
        num_projections: Number of random 1D projections (L).

    Returns:
        Scalar SW_2 distance.
    """
    d = x.shape[1]

    # sample random directions uniformly on the unit sphere S^{d-1}
    raw_dirs = jax.random.normal(key, shape=(num_projections, d))
    projections = raw_dirs / jnp.linalg.norm(raw_dirs, axis=1, keepdims=True)  # (L, d)

    # pProject d-dimensional samples onto 1D lines: (N, d) x (d, L) -> (N, L)
    x_proj = x @ projections.T  # (N, L)
    y_proj = y @ projections.T  # (M, L)

    # vectorize 1D W_2 calculation across all L projection slices
    def compute_slice_w2_sq(proj_x, proj_y):
        return wasserstein_1d(proj_x, proj_y) ** 2

    squared_w2_slices = jax.vmap(compute_slice_w2_sq, in_axes=(1, 1))(x_proj, y_proj)

    # integrate across sphere via mean over projections
    return jnp.sqrt(jnp.mean(squared_w2_slices))


@jax.jit
def gaussian_wasserstein_distance(
    mu_x: jax.Array,
    cov_x: jax.Array,
    mu_y: jax.Array,
    cov_y: jax.Array,
) -> jax.Array:
    """
    Computes closed-form W_2 distance between two Gaussian distributions N(mu_x, cov_x)
    and N(mu_y, cov_y). See e.g. Knott and Smith (1984) https://doi.org/10.1007/BF00934745.

    Args:
        mu_x (d,): Mean of the first Gaussian.
        cov_x (d, d): Covariance of the first Gaussian.
        mu_y (d,): Mean of the second Gaussian.
        cov_y (d, d): Covariance of the second Gaussian.

    Returns:
        Scalar W_2 distance between the two Gaussian distributions.
    """
    # force 1D vectors and 2D matrices to ensure compatibility with d=1 scalars
    mu_x = jnp.atleast_1d(mu_x)
    mu_y = jnp.atleast_1d(mu_y)
    cov_x = jnp.atleast_2d(cov_x)
    cov_y = jnp.atleast_2d(cov_y)

    mean_diff_sq = jnp.sum((mu_x - mu_y) ** 2)

    # compute matrix square root of cov_x: cov_x^{1/2}
    evals_x, evecs_x = jnp.linalg.eigh(cov_x)
    evals_x_sqrt = jnp.sqrt(jnp.maximum(evals_x, 0.0))
    cov_x_sqrt = evecs_x @ (evals_x_sqrt[:, None] * evecs_x.T)

    # inner matrix: cov_x^{1/2} @ cov_y @ cov_x^{1/2}
    inner_mat = cov_x_sqrt @ cov_y @ cov_x_sqrt
    evals_inner = jnp.linalg.eigvalsh(inner_mat)
    trace_inner_sqrt = jnp.sum(jnp.sqrt(jnp.maximum(evals_inner, 0.0)))

    trace_x = jnp.trace(cov_x)
    trace_y = jnp.trace(cov_y)

    w2_sq = mean_diff_sq + trace_x + trace_y - 2.0 * trace_inner_sqrt
    return jnp.sqrt(jnp.maximum(w2_sq, 0.0))
