"""heston_fft.py — Snelle Heston-prijzen via Carr-Madan FFT"""
import numpy as np
from scipy.stats import norm


def heston_cf(u, S, T, r, q, v0, kappa, theta, sigma_v, rho):
    i = 1j
    xi = kappa - sigma_v * rho * i * u
    d = np.sqrt(xi ** 2 + sigma_v ** 2 * (i * u + u ** 2))
    g = (xi - d) / (xi + d)
    C = (kappa * theta / sigma_v ** 2) * (
        (xi - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    )
    D = ((xi - d) / sigma_v ** 2) * (
        (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    )
    return np.exp(i * u * (np.log(S) + (r - q) * T) + C + D * v0)


def heston_call_fft(K_array, S, T, r, q, v0, kappa, theta, sigma_v, rho,
                     alpha=1.5, N=4096, eta=0.25):
    """
    Prijs Europese calls voor een array van strikes via Carr-Madan FFT.

    K_array: array van strikes
    N: aantal FFT-punten (moet macht van 2 zijn voor snelheid)
    eta: stapgrootte in u-ruimte
    alpha: dempingsfactor (1.5 werkt meestal goed)

    Return: array met call-prijzen op K_array
    """
    # Lambda: stapgrootte in log-strike ruimte
    lam = 2 * np.pi / (N * eta)

    # Log-strikes waarop FFT werkt
    b = np.log(S)  # centreren rond spot
    k_fft = -b + lam * np.arange(N)

    # u-waarden
    u = eta * np.arange(N)

    # Karakteristieke functie geëvalueerd op u - (alpha+1)i
    cf = heston_cf(u - (alpha + 1) * 1j, S, T, r, q,
                    v0, kappa, theta, sigma_v, rho)

    # Carr-Madan integrand
    denom = alpha ** 2 + alpha - u ** 2 + 1j * u * (2 * alpha + 1)
    integrand = np.exp(-r * T) * cf / denom

    # Simpson's weights
    simpson = np.ones(N)
    simpson[0] = simpson[-1] = 1/3
    simpson[1:-1:2] = 4/3
    simpson[2:-1:2] = 2/3

    # FFT
    x = np.exp(1j * b * u) * integrand * eta * simpson
    fft_result = np.fft.fft(x)

    # Prijzen op k_fft-grid
    call_prices_fft = np.real(np.exp(-alpha * k_fft) / np.pi * fft_result)

    # Interpoleer naar gevraagde strikes
    k_target = np.log(K_array)
    prices = np.interp(k_target, k_fft, call_prices_fft)

    return np.maximum(prices, 0.0)


# ============================================================
# Vergelijking: FFT vs quad — laat de snelheid zien
# ============================================================
def benchmark(S=100, K=100, T=1, r=0.05, q=0.02,
              v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.7):
    """Vergelijk FFT vs quad op één strike."""
    import time
    from heston import heston_call

    # quad
    t0 = time.time()
    for _ in range(10):
        prijs_quad = heston_call(S, K, T, r, q, v0, kappa, theta,
                                  sigma_v, rho)
    t_quad = (time.time() - t0) / 10

    # FFT (één strike)
    t0 = time.time()
    for _ in range(10):
        prijs_fft = heston_call_fft(np.array([K]), S, T, r, q,
                                     v0, kappa, theta, sigma_v, rho,
                                     N=2048, eta=0.25)[0]
    t_fft = (time.time() - t0) / 10

    # FFT (100 strikes tegelijk) — waar het echt wint
    K_array = np.linspace(80, 120, 100)
    t0 = time.time()
    for _ in range(10):
        prijzen_array = heston_call_fft(K_array, S, T, r, q,
                                         v0, kappa, theta, sigma_v, rho,
                                         N=2048, eta=0.25)
    t_fft_batch = (time.time() - t0) / 10
    t_quad_batch = t_quad * 100

    return {
        "quad_1_strike": t_quad,
        "fft_1_strike": t_fft,
        "quad_100_strikes": t_quad_batch,
        "fft_100_strikes": t_fft_batch,
        "fft_vs_quad_100x": t_quad_batch / t_fft_batch,
        "prijs_quad": prijs_quad,
        "prijs_fft": prijs_fft,
    }