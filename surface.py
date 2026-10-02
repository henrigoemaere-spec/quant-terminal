"""surface.py — Volatility surface module"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq
from scipy.stats import norm


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


def implied_vol(marktprijs, S, K, T, r, optie="call", q=0.0):
    f = bs_call if optie == "call" else bs_put
    def g(sigma):
        return f(S, K, T, r, sigma, q) - marktprijs
    try:
        return brentq(g, 1e-6, 5.0)
    except ValueError:
        return np.nan


def genereer_synthetische_data(S=100.0, r=0.05, q=0.02):
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)
    maturities = np.array([1/12, 3/12, 6/12, 1.0, 2.0, 3.0])

    rijen = []
    for T in maturities:
        atm_vol = 0.18 + 0.02 * T
        skew = -0.10 * np.exp(-T)
        smile_coef = 0.30

        for K in strikes:
            m = np.log(K / S)
            iv = atm_vol + skew * m + smile_coef * m ** 2
            call_prijs = bs_call(S, K, T, r, iv, q)
            put_prijs = bs_put(S, K, T, r, iv, q)
            rijen.append({
                "S": S, "K": K, "T": T, "r": r, "q": q,
                "call_prijs": round(call_prijs, 4),
                "put_prijs": round(put_prijs, 4),
                "echte_iv": round(iv, 4),
            })
    return pd.DataFrame(rijen)


def sla_demo_csv_op(pad="marketdata_demo.csv", S=100.0, r=0.05, q=0.02):
    df = genereer_synthetische_data(S, r, q)
    df.to_csv(pad, index=False)
    return pad, df


def bouw_surface(df, optie="call"):
    rijen = []
    for _, row in df.iterrows():
        S, K, T, r, q = row["S"], row["K"], row["T"], row["r"], row["q"]
        prijs = row["call_prijs"] if optie == "call" else row["put_prijs"]
        iv = implied_vol(prijs, S, K, T, r, optie, q)
        rijen.append({"K": K, "T": T, "prijs": prijs, "iv": iv})

    surface = pd.DataFrame(rijen)
    pivot = surface.pivot(index="T", columns="K", values="iv")
    return surface, pivot


def interpoleer(pivot, K, T):
    strikes = pivot.columns.values
    maturities = pivot.index.values
    ivs = pivot.values
    spline = RectBivariateSpline(maturities, strikes, ivs, kx=1, ky=1)
    return float(spline(T, K)[0, 0])


def check_butterfly(pivot, S, r, q):
    problemen = []
    for T in pivot.index:
        strikes = pivot.columns.values
        call_prijzen = np.array([
            bs_call(S, K, T, r, pivot.loc[T, K], q) for K in strikes
        ])
        d2 = np.diff(call_prijzen, n=2)
        if np.any(d2 < -1e-6):
            problemen.append((T, "butterfly: niet-convexe call-prijzen in K"))
    return problemen


def check_calendar(pivot, S, r, q):
    problemen = []
    strikes = pivot.columns.values
    maturities = pivot.index.values
    for K in strikes:
        w = np.array([pivot.loc[T, K] ** 2 * T for T in maturities])
        if np.any(np.diff(w) < -1e-6):
            problemen.append((K, "calendar: totale variantie daalt in T"))
    return problemen


def maak_plots(pivot, output_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Gebruik altijd een absolute pad, gebaseerd op de locatie van dit bestand
    if output_dir is None:
        basis = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(basis, "plots")

    os.makedirs(output_dir, exist_ok=True)

    strikes = pivot.columns.values
    maturities = pivot.index.values

    # Plot 1: Smile per maturity
    fig, ax = plt.subplots(figsize=(8, 5))
    for T in maturities:
        ax.plot(strikes, pivot.loc[T].values * 100,
                marker="o", label=f"T={T:.2f}j")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title("Volatility smile per maturity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pad1 = os.path.join(output_dir, "smile.png")
    fig.savefig(pad1, dpi=120)
    plt.close(fig)

    # Plot 2: Term structure per strike
    fig, ax = plt.subplots(figsize=(8, 5))
    for K in strikes:
        ax.plot(maturities, pivot[K].values * 100,
                marker="s", label=f"K={K:.0f}")
    ax.set_xlabel("Maturity T (jaar)")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title("Term structure per strike")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pad2 = os.path.join(output_dir, "term_structure.png")
    fig.savefig(pad2, dpi=120)
    plt.close(fig)

    # Plot 3: 3D surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(strikes, maturities)
    Z = pivot.values * 100
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", alpha=0.85)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity T")
    ax.set_zlabel("IV (%)")
    ax.set_title("Volatility surface")
    fig.tight_layout()
    pad3 = os.path.join(output_dir, "surface_3d.png")
    fig.savefig(pad3, dpi=120)
    plt.close(fig)

    return pad1, pad2, pad3


def export_surface(pivot, pad=None):
    if pad is None:
        basis = os.path.dirname(os.path.abspath(__file__))
        pad = os.path.join(basis, "surface_export.csv")
    pivot.to_csv(pad)
    return pad