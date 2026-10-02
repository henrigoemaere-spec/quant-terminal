"""sabr.py — SABR model met Hagan-formule (ATM-vol input)"""
import os
import numpy as np
from scipy.optimize import minimize


def sabr_iv(F, K, T, alpha, beta, rho, nu):
    """
    SABR implied volatility via Hagan (2002).

    INPUTS:
    - F: forward price (float)
    - K: strike (float)
    - T: time to maturity (float)
    - alpha: ATM volatility (bv 0.20 voor 20%)
    - beta: CEV exponent (0.5 typisch)
    - rho: correlatie (-1 < rho < 1)
    - nu: vol-of-vol
    """
    # Zorg dat alles float is
    try:
        F = float(F)
        K = float(K)
        T = float(T)
        alpha = float(alpha)
        beta = float(beta)
        rho = float(rho)
        nu = float(nu)
    except (TypeError, ValueError):
        return np.nan

    if T <= 0 or alpha <= 0 or nu <= 0 or abs(rho) >= 1 or F <= 0 or K <= 0:
        return np.nan

    alpha_H = alpha * F ** (1 - beta)
    eps = 1e-7
    log_FK = np.log(F / K)

    # ATM fallback
    if abs(log_FK) < eps:
        FK_beta = F ** (1 - beta)
        term1 = alpha_H / FK_beta
        term2 = 1 + (
            ((1 - beta) ** 2 / 24) * (alpha_H ** 2 / F ** (2 - 2 * beta))
            + 0.25 * rho * beta * nu * alpha_H / FK_beta
            + (2 - 3 * rho ** 2) / 24 * nu ** 2
        ) * T
        result = term1 * term2
        return float(result) if np.isfinite(result) else np.nan

    # Niet-ATM
    FK_beta = (F * K) ** ((1 - beta) / 2)

    z = (nu / alpha_H) * FK_beta * log_FK

    # Bescherm tegen numerieke problemen
    inner = 1 - 2 * rho * z + z ** 2
    if inner <= 0:
        return np.nan

    disc = np.sqrt(inner)
    num_x = disc + z - rho
    if num_x <= 0:
        return np.nan

    x_z = np.log(num_x / (1 - rho))

    denom_A = FK_beta * (
        1
        + ((1 - beta) ** 2 / 24) * log_FK ** 2
        + ((1 - beta) ** 4 / 1920) * log_FK ** 4
    )
    A = alpha_H / denom_A

    B = 1 + (
        ((1 - beta) ** 2 / 24) * (alpha_H ** 2) / (FK_beta ** 2)
        + 0.25 * rho * beta * nu * alpha_H / FK_beta
        + ((2 - 3 * rho ** 2) / 24) * nu ** 2
    ) * T

    if abs(x_z) < eps:
        z_over_x = 1.0
    else:
        z_over_x = z / x_z

    result = A * z_over_x * B
    return float(result) if np.isfinite(result) else np.nan


def forward(S, r, q, T):
    return S * np.exp((r - q) * T)


def kalibreer_sabr_smile(F, T, strikes, markt_ivs, beta=0.5,
                         start=(0.20, -0.3, 0.4)):
    """Kalibreer alpha, rho, nu op één maturity."""
    strikes = np.asarray(strikes, dtype=float)
    markt_ivs = np.asarray(markt_ivs, dtype=float)

    def doel(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 0.99:
            return 1e10
        ss = 0.0
        for K, iv_markt in zip(strikes, markt_ivs):
            iv_model = sabr_iv(F, K, T, alpha, beta, rho, nu)
            if not np.isfinite(iv_model):
                ss += 1e4
            else:
                diff = iv_model - iv_markt
                ss += diff * diff
        return ss

    grenzen = [(0.01, 1.0), (-0.99, 0.99), (0.01, 2.0)]
    res = minimize(doel, start, method="L-BFGS-B", bounds=grenzen,
                   options={"maxiter": 200})
    return res.x, res.fun


def kalibreer_sabr_volledig(markt_data, S, r, q, beta=0.5):
    """Kalibreer SABR per maturity."""
    per_T = {}
    for K, T, iv in markt_data:
        per_T.setdefault(T, []).append((K, iv))

    resultaten = {}
    for T, punten in sorted(per_T.items()):
        strikes = np.array([k for k, _ in punten])
        ivs = np.array([iv for _, iv in punten])
        F = forward(S, r, q, T)

        atm_idx = np.argmin(np.abs(strikes - F))
        atm_vol = float(ivs[atm_idx])
        start = (atm_vol, -0.5, 0.5)

        params, ss = kalibreer_sabr_smile(F, T, strikes, ivs, beta=beta,
                                           start=start)
        alpha, rho, nu = params
        resultaten[T] = {
            "alpha": float(alpha), "beta": float(beta),
            "rho": float(rho), "nu": float(nu),
            "residueel": float(ss),
        }
    return resultaten


def plot_sabr_fit(markt_data, S, r, q, params_per_T, output_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(output_dir, exist_ok=True)

    per_T = {}
    for K, T, iv in markt_data:
        per_T.setdefault(T, []).append((K, iv))

    fig, ax = plt.subplots(figsize=(9, 6))
    kleuren = plt.cm.viridis(np.linspace(0, 1, len(per_T)))

    for (T, punten), kleur in zip(sorted(per_T.items()), kleuren):
        punten_sorted = sorted(punten, key=lambda x: x[0])
        strikes = np.array([k for k, _ in punten_sorted])
        markt_ivs = np.array([iv for _, iv in punten_sorted])

        F = forward(S, r, q, T)
        p = params_per_T[T]
        model_ivs = np.array([
            sabr_iv(F, K, T, p["alpha"], p["beta"], p["rho"], p["nu"])
            for K in strikes
        ])

        ax.plot(strikes, markt_ivs * 100, "o", color=kleur, alpha=0.6,
                markersize=8, label=f"Markt T={T:.2f}")
        ax.plot(strikes, model_ivs * 100, "-", color=kleur, linewidth=2,
                label=f"SABR T={T:.2f}")

    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title("SABR kalibratie: model vs markt")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pad = os.path.join(output_dir, "sabr_fit.png")
    fig.savefig(pad, dpi=120)
    plt.close(fig)
    return pad