"""risk.py — Professionele risk-analyse: vega bucketing + P&L attribution"""
import os
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


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


def bs_price(S, K, T, r, sigma, optie="call", q=0.0):
    return bs_call(S, K, T, r, sigma, q) if optie == "call" \
        else bs_put(S, K, T, r, sigma, q)


def implied_vol(prijs, S, K, T, r, optie="call", q=0.0):
    f = bs_call if optie == "call" else bs_put
    def g(sigma):
        return f(S, K, T, r, sigma, q) - prijs
    try:
        return brentq(g, 1e-6, 5.0)
    except ValueError:
        return np.nan


def vega_bs(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def vega_buckets(positie, S, r, q, buckets_strikes, buckets_maturities):
    vega_matrix = np.zeros((len(buckets_maturities), len(buckets_strikes)))
    for K_pos, T_pos, aantal, optie, iv in positie:
        i_T = int(np.argmin(np.abs(buckets_maturities - T_pos)))
        i_K = int(np.argmin(np.abs(buckets_strikes - K_pos)))
        v = vega_bs(S, K_pos, T_pos, r, iv, q)
        vega_matrix[i_T, i_K] += aantal * v
    return vega_matrix


def pnl_attribution(positie, S, r, q, shock_S_pct=0.01, shock_vol_pts=0.01,
                    shock_r_bp=10.0):
    dS = S * shock_S_pct
    dSigma = shock_vol_pts
    dr = shock_r_bp / 10000.0

    total_delta_pnl = 0.0
    total_gamma_pnl = 0.0
    total_vega_pnl = 0.0
    total_rho_pnl = 0.0

    for K, T, aantal, optie, iv in positie:
        d1 = (np.log(S / K) + (r - q + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)

        if optie == "call":
            delta = np.exp(-q * T) * norm.cdf(d1)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            delta = -np.exp(-q * T) * norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * iv * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        total_delta_pnl += aantal * delta * dS
        total_gamma_pnl += aantal * 0.5 * gamma * dS ** 2
        total_vega_pnl += aantal * vega * dSigma
        total_rho_pnl += aantal * rho * dr

    totaal = total_delta_pnl + total_gamma_pnl + total_vega_pnl + total_rho_pnl

    return {
        "Delta P&L": total_delta_pnl,
        "Gamma P&L": total_gamma_pnl,
        "Vega P&L": total_vega_pnl,
        "Rho P&L": total_rho_pnl,
        "Totaal P&L": totaal,
    }


def portfolio_greeks(positie, S, r, q):
    totalen = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    marktwaarde = 0.0

    for K, T, aantal, optie, iv in positie:
        d1 = (np.log(S / K) + (r - q + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)

        if optie == "call":
            delta = np.exp(-q * T) * norm.cdf(d1)
            theta = (-S * pdf_d1 * iv * np.exp(-q * T) / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * norm.cdf(d2)
                     + q * S * np.exp(-q * T) * norm.cdf(d1))
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            delta = -np.exp(-q * T) * norm.cdf(-d1)
            theta = (-S * pdf_d1 * iv * np.exp(-q * T) / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)
                     - q * S * np.exp(-q * T) * norm.cdf(-d1))
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        gamma = np.exp(-q * T) * pdf_d1 / (S * iv * np.sqrt(T))
        vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)

        totalen["delta"] += aantal * delta
        totalen["gamma"] += aantal * gamma
        totalen["vega"] += aantal * vega
        totalen["theta"] += aantal * theta
        totalen["rho"] += aantal * rho

        marktwaarde += aantal * bs_price(S, K, T, r, iv, optie, q)

    return totalen, marktwaarde


def plot_vega_heatmap(vega_matrix, buckets_strikes, buckets_maturities,
                      output_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(vega_matrix, cmap="RdYlGn", aspect="auto",
                   origin="lower", interpolation="nearest")
    ax.set_xticks(range(len(buckets_strikes)))
    ax.set_xticklabels([f"{k:.0f}" for k in buckets_strikes])
    ax.set_yticks(range(len(buckets_maturities)))
    ax.set_yticklabels([f"{T:.2f}" for T in buckets_maturities])
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (jaar)")
    ax.set_title("Vega bucketing (rood = negatief, groen = positief)")

    for i in range(len(buckets_maturities)):
        for j in range(len(buckets_strikes)):
            val = vega_matrix[i, j]
            if abs(val) > 0.01:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Vega")
    fig.tight_layout()
    pad = os.path.join(output_dir, "vega_heatmap.png")
    fig.savefig(pad, dpi=120)
    plt.close(fig)
    return pad


def voorbeeld_portfolio():
    return [
        (100.0, 1.0, +100, "call", 0.20),
        (110.0, 1.0, -200, "call", 0.18),
        (90.0, 0.5, +50, "put", 0.24),
        (105.0, 2.0, -100, "call", 0.21),
    ]