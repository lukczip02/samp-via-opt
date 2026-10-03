import jax
import jax.numpy as jnp


def ess_1d(x_chain: jax.Array) -> jax.Array:
    """
    Computes Effective Sample Size across time steps using FFT autocorrelation.

    Args:
        x_chain (N,): 1D array of samples from a Markov chain.

    Returns:
        Scalar effective sample size (ESS).
    """
    n = x_chain.shape[0]
    x_centered = x_chain - jnp.mean(x_chain)

    # Fast autocorrelation via FFT
    fft_val = jnp.abs(jnp.fft.fft(x_centered, n=2 * n)) ** 2
    autocorr = jnp.fft.ifft(fft_val)[:n].real
    autocorr = autocorr / autocorr[0]

    neg = autocorr < 0
    max_lag = jnp.where(neg.any(), jnp.argmax(neg), n // 2)
    lags = jnp.arange(n)
    tau = 1.0 + 2.0 * jnp.sum(jnp.where((lags >= 1) & (lags < max_lag), autocorr, 0.0))

    return n / tau


@jax.jit
def compute_ess(trajectory: jax.Array) -> jax.Array:
    """
    Vectorized ESS computation across all coordinate dimensions.

    Args:
        trajectory (N, d): Full d-dimensional state history of the Markov chain across all iterations.

    Returns:
        (d,): Array of ESS values for each coordinate dimension.
    """
    return jax.vmap(ess_1d, in_axes=1)(trajectory)
