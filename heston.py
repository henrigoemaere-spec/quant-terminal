"""heston.py — Heston model met FFT-prijzen en IV-kalibratie"""
import os
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, brentq
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


def heston_call(S, K, T, r, q, v0, kappa, theta, sigma_v, rho, alpha=1.5):
    def integrand(u):
        cf = heston_cf(u - (alpha + 1) * 1j, S, T, r, q,
                       v0, kappa, theta, sigma_v, rho)
        denom = alpha ** 2 + alpha - u ** 2 + 1j * u * (2 * alpha + 1)
        return np.real(np.exp(-1j * u * np.log(K)) * cf / denom)
    integraal, _ = quad(integrand, 0, 200, limit=200)
    prijs = np.exp(-alpha * np.log(K)) / np.pi * integraal * np.exp(-r * T)
    return max(prijs, 0.0)


def heston_put(S, K, T, r, q, v0, kappa, theta, sigma_v, rho):
    call = heston_call(S, K, T, r, q, v0, kappa, theta, sigma_v, rho)
    return call - S * np.exp(-q * T) + K * np.exp(-r * T)


def heston_price(S, K, T, r, q, v0, kappa, theta, sigma_v, rho, optie="call"):
    if optie == "call":
        return heston_call(S, K, T, r, q, v0, kappa, theta, sigma_v, rho)
    return heston_put(S, K, T, r, q, v0, kappa, theta, sigma_v, rho)


def bs_call(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def implied_vol(prijs, S, K, T, r, optie="call", q=0.0):
    f = bs_call if optie == "call" else bs_put
    def g(sigma):
        return f(S, K, T, r, sigma, q) - prijs
    try:
        return brentq(g, 1e-6, 5.0)
    except ValueError:
        return np.nan


def kalibreer_heston_iv(markt_data, S, r, q, start=(0.04, 2.0, 0.04, 0.5, -0.7)):
    """Kalibratie in IV-ruimte. markt_data: (K, T, iv_markt)."""
    def doelfunctie(params):
        v0, kappa, theta, sigma_v, rho = params
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma_v <= 0 or abs(rho) >= 1:
            return 1e10
        ss = 0.0
        for K, T, iv_markt in markt_data:
            try:
                prijs = heston_call(S, K, T, r, q,
                                    v0, kappa, theta, sigma_v, rho)
                iv_model = implied_vol(prijs, S, K, T, r, "call", q)
                if np.isnan(iv_model):
                    ss += 1e4
                else:
                    ss += (iv_model - iv_markt) ** 2
            except Exception:
                ss += 1e4
        return ss

    grenzen = [
        (1e-4, 1.0), (0.1, 10.0), (1e-4, 1.0), (0.01, 2.0), (-0.99, 0.0),
    ]
    res = minimize(doelfunctie, start, method="L-BFGS-B", bounds=grenzen,
                   options={"maxiter": 150})
    return res.x, res.fun


def kalibreer_heston(markt_data, S, r, q, start=(0.04, 2.0, 0.04, 0.5, -0.7)):
    """Kalibratie in prijsruimte. markt_data: (K, T, prijs, optie)."""
    def doelfunctie(params):
        v0, kappa, theta, sigma_v, rho = params
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma_v <= 0 or abs(rho) >= 1:
            return 1e10
        ss = 0.0
        for K, T, markt_prijs, optie in markt_data:
            try:
                model_prijs = heston_price(S, K, T, r, q,
                                           v0, kappa, theta, sigma_v, rho, optie)
                ss += (model_prijs - markt_prijs) ** 2
            except Exception:
                ss += 1e6
        return ss
    grenzen = [(1e-4, 1.0), (0.1, 10.0), (1e-4, 1.0), (0.01, 2.0), (-0.99, 0.0)]
    res = minimize(doelfunctie, start, method="L-BFGS-B", bounds=grenzen,
                   options={"maxiter": 100})
    return res.x, res.fun


def plot_heston_vs_market(markt_data, S, r, q, params, output_dir=None):
    """Plot Heston-fit vs markt. Accepteert (K, T, iv) en (K, T, prijs, optie)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(output_dir, exist_ok=True)

    v0, kappa, theta, sigma_v, rho = params

    per_T = {}
    for item in markt_data:
        K, T = item[0], item[1]
        if len(item) == 3:
            iv_markt = item[2]
        else:
            prijs = item[2]
            optie = item[3] if len(item) > 3 else "call"
            iv_markt = implied_vol(prijs, S, K, T, r, optie, q)
        if iv_markt is not None and not np.isnan(iv_markt):
            per_T.setdefault(T, []).append((K, iv_markt))

    if not per_T:
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    kleuren = plt.cm.plasma(np.linspace(0, 1, max(len(per_T), 2)))

    for (T, punten), kleur in zip(sorted(per_T.items()), kleuren):
        punten_sorted = sorted(punten, key=lambda x: x[0])
        strikes = [k for k, _ in punten_sorted]
        markt_ivs = [iv for _, iv in punten_sorted]
        model_ivs = []
        for K in strikes:
            prijs = heston_call(S, K, T, r, q, v0, kappa, theta,
                                sigma_v, rho)
            iv_model = implied_vol(prijs, S, K, T, r, "call", q)
            model_ivs.append(iv_model if not np.isnan(iv_model) else 0)
        ax.plot(strikes, np.array(markt_ivs) * 100, "o", color=kleur,
                alpha=0.5, markersize=6)
        ax.plot(strikes, np.array(model_ivs) * 100, "-", color=kleur,
                linewidth=2, label=f"T={T:.2f}")

    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"Heston kalibratie\nv0={v0:.4f}, kappa={kappa:.2f}, "
                 f"theta={theta:.4f}, sigma_v={sigma_v:.3f}, rho={rho:.3f}")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pad = os.path.join(output_dir, "heston_fit.png")
    fig.savefig(pad, dpi=120)
    plt.close(fig)
    return pad


def genereer_heston_marktdata(S=100.0, r=0.05, q=0.02,
                              echte_params=(0.04, 2.0, 0.04, 0.5, -0.7)):
    v0, kappa, theta, sigma_v, rho = echte_params
    strikes = [90, 95, 100, 105, 110]
    maturities = [3/12, 6/12, 1.0, 2.0]
    data = []
    for T in maturities:
        for K in strikes:
            prijs = heston_price(S, K, T, r, q, v0, kappa, theta,
                                 sigma_v, rho, "call")
            data.append((K, T, prijs, "call"))
    return data