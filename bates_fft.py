"""bates_fft.py — Bates model via Carr-Madan FFT (30x sneller dan quad)"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def bates_cf(u, S, T, r, q, v0, kappa, theta, sigma_v, rho,
             lam, mu_J, sigma_J):
    """Karakteristieke functie van Bates (Heston + Merton jumps)."""
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
    heston_part = np.exp(i * u * (np.log(S) + (r - q) * T) + C + D * v0)

    jump_cf = np.exp(
        lam * T * (np.exp(i * u * mu_J - 0.5 * u ** 2 * sigma_J ** 2) - 1)
        - i * u * lam * T * (np.exp(mu_J + 0.5 * sigma_J ** 2) - 1)
    )
    return heston_part * jump_cf


def bates_call_fft(K_array, S, T, r, q, v0, kappa, theta, sigma_v, rho,
                    lam, mu_J, sigma_J, alpha=1.5, N=1024, eta=0.25):
    """Prijs Europese calls voor een array van strikes via Carr-Madan FFT."""
    lam_grid = 2 * np.pi / (N * eta)
    b = np.log(S)
    k_fft = -b + lam_grid * np.arange(N)

    u = eta * np.arange(N)

    cf = bates_cf(u - (alpha + 1) * 1j, S, T, r, q,
                   v0, kappa, theta, sigma_v, rho, lam, mu_J, sigma_J)

    denom = alpha ** 2 + alpha - u ** 2 + 1j * u * (2 * alpha + 1)
    integrand = np.exp(-r * T) * cf / denom

    simpson = np.ones(N)
    simpson[0] = simpson[-1] = 1/3
    simpson[1:-1:2] = 4/3
    simpson[2:-1:2] = 2/3

    x = np.exp(1j * b * u) * integrand * eta * simpson
    fft_result = np.fft.fft(x)

    call_prices_fft = np.real(np.exp(-alpha * k_fft) / np.pi * fft_result)
    k_target = np.log(K_array)
    prices = np.interp(k_target, k_fft, call_prices_fft)
    return np.maximum(prices, 0.0)


def kalibreer_bates_fft(markt_data, S, r, q, n_start=2, verbose=True):
    """
    Snelle Bates-kalibratie in prijsruimte met vega-gewichten.
    markt_data: lijst van (K, T, iv_markt).
    """
    # Converteer IV-data naar prijzen (eenmalig, buiten de optimizer)
    data_prijs = []
    vega_weights = {}
    for K, T, iv in markt_data:
        # BS call prijs
        d1 = (np.log(S / K) + (r - q + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        prijs = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        data_prijs.append((K, T, iv, prijs))
        vega_weights[(K, T)] = 1.0 / max(vega ** 2, 1e-8)

    # Groepeer per maturity voor batch FFT
    per_T_data = {}
    for K, T, iv, prijs in data_prijs:
        per_T_data.setdefault(T, []).append((K, prijs, vega_weights[(K, T)]))

    def doel(params_log):
        v0 = np.exp(params_log[0])
        kappa = params_log[1]
        theta = np.exp(params_log[2])
        sigma_v = np.exp(params_log[3])
        rho = params_log[4]
        lam = np.exp(params_log[5])
        mu_J = params_log[6]
        sigma_J = np.exp(params_log[7])

        # Bounds check
        if (v0 <= 1e-6 or kappa <= 0.1 or theta <= 1e-6 or sigma_v <= 1e-6 or
            abs(rho) >= 0.99 or lam < 1e-6 or sigma_J <= 1e-6 or
            lam > 3.0 or sigma_J > 1.5):
            return 1e10

        ss = 0.0
        for T, punten in per_T_data.items():
            strikes = np.array([k for k, _, _ in punten])
            prijzen_markt = np.array([p for _, p, _ in punten])
            weights = np.array([w for _, _, w in punten])

            try:
                prijzen_model = bates_call_fft(
                    strikes, S, T, r, q,
                    v0, kappa, theta, sigma_v, rho,
                    lam, mu_J, sigma_J,
                    N=512, eta=0.3
                )
                ss += np.sum(weights * (prijzen_markt - prijzen_model) ** 2)
            except Exception:
                ss += 1e6
        return ss

    starts = [
        [np.log(0.04), 2.0, np.log(0.04), np.log(0.3), -0.7, np.log(0.1), -0.1, np.log(0.15)],
        [np.log(0.03), 1.5, np.log(0.05), np.log(0.5), -0.6, np.log(0.2), -0.15, np.log(0.20)],
    ][:n_start]

    grenzen_log = [
        (np.log(1e-4), np.log(1.0)),   # v0
        (0.1, 10.0),                   # kappa
        (np.log(1e-4), np.log(1.0)),   # theta
        (np.log(0.01), np.log(2.0)),   # sigma_v
        (-0.99, 0.0),                  # rho
        (np.log(1e-4), np.log(2.0)),   # lam
        (-1.0, 0.5),                   # mu_J
        (np.log(0.01), np.log(1.0)),   # sigma_J
    ]

    beste = None
    beste_ss = np.inf

    for i, start in enumerate(starts):
        if verbose:
            print(f"    Start {i+1}/{len(starts)}...")
        try:
            res = minimize(
                doel, start, method="L-BFGS-B", bounds=grenzen_log,
                options={"maxiter": 80, "ftol": 1e-8}
            )
            if verbose:
                print(f"      SS = {res.fun:.6e}")
            if res.fun < beste_ss:
                beste_ss = res.fun
                beste = res
                if verbose:
                    print(f"      Nieuw beste!")
        except Exception as e:
            if verbose:
                print(f"      Fout: {e}")

    if beste is None:
        return None, np.inf

    x = beste.x
    params = np.array([
        np.exp(x[0]),
        x[1],
        np.exp(x[2]),
        np.exp(x[3]),
        x[4],
        np.exp(x[5]),
        x[6],
        np.exp(x[7]),
    ])
    return params, beste_ss