"""dupire.py — Dupire local volatility met smoothing"""
import os
import numpy as np
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline


def bs_call(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bouw_dupire_surface(df, S, r, q, n_T=10, n_K=10, smooth_sigma=1.5):
    """Bouw Dupire local-vol surface met regularisatie."""
    strikes = np.sort(df["K"].unique())
    maturities = np.sort(df["T"].unique())

    iv_matrix = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            mask = (df["T"] == T) & (df["K"] == K)
            if mask.any():
                iv_matrix[i, j] = df.loc[mask, "iv_markt"].iloc[0]
            else:
                iv_matrix[i, j] = 0.20

    iv_smooth = gaussian_filter(iv_matrix, sigma=smooth_sigma, mode="nearest")

    spline = RectBivariateSpline(maturities, strikes, iv_smooth, kx=2, ky=2)

    def iv_at(T, K):
        # BELANGRIJK: [0, 0] omdat RectBivariateSpline altijd 2D teruggeeft
        return float(spline(T, K)[0, 0])

    K_grid = np.linspace(strikes.min(), strikes.max(), n_K)
    T_grid = np.linspace(maturities.min() + 0.02, maturities.max() - 0.02, n_T)

    LV_matrix = np.zeros((n_T, n_K))

    eps_T = 0.02
    eps_K_rel = 0.03

    for i, T in enumerate(T_grid):
        for j, K in enumerate(K_grid):
            try:
                iv = iv_at(T, K)
                if iv <= 0.01 or iv > 2.0:
                    LV_matrix[i, j] = 0.20
                    continue

                T_up = min(T + eps_T, maturities.max())
                T_dn = max(T - eps_T, maturities.min())
                dIV_dT = (iv_at(T_up, K) - iv_at(T_dn, K)) / (T_up - T_dn)

                eps_K = eps_K_rel * K
                K_up = min(K + eps_K, strikes.max())
                K_dn = max(K - eps_K, strikes.min())
                iv_up = iv_at(T, K_up)
                iv_dn = iv_at(T, K_dn)
                dIV_dK = (iv_up - iv_dn) / (K_up - K_dn)
                d2IV_dK2 = (iv_up - 2 * iv + iv_dn) / (((K_up - K_dn) / 2) ** 2)

                d1 = (np.log(S / K) + (r - q + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
                d2 = d1 - iv * np.sqrt(T)

                numer = iv ** 2 + 2 * iv * T * (dIV_dT + (r - q) * K * dIV_dK)
                denom = ((1 + K * d1 * np.sqrt(T) * dIV_dK) ** 2
                         + K ** 2 * T * iv * (d2IV_dK2 - d1 * np.sqrt(T) * dIV_dK ** 2))

                if abs(denom) < 1e-8 or numer <= 1e-6:
                    LV_matrix[i, j] = 0.20
                    continue

                lv2 = numer / denom
                if lv2 <= 0 or lv2 > 4.0:
                    LV_matrix[i, j] = 0.20
                else:
                    LV_matrix[i, j] = np.sqrt(lv2)

            except Exception:
                LV_matrix[i, j] = 0.20

    LV_matrix = gaussian_filter(LV_matrix, sigma=0.5, mode="nearest")

    return K_grid, T_grid, LV_matrix


def plot_dupire_surface(K_grid, T_grid, LV_matrix, output_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(LV_matrix * 100, cmap="viridis", aspect="auto",
                   origin="lower",
                   extent=[K_grid.min(), K_grid.max(),
                           T_grid.min(), T_grid.max()])
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity T")
    ax.set_title("Dupire local volatility surface (%)")
    fig.colorbar(im, ax=ax, label="Local vol (%)")
    fig.tight_layout()
    pad1 = os.path.join(output_dir, "dupire_heatmap.png")
    fig.savefig(pad1, dpi=120)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(K_grid, T_grid)
    Z = LV_matrix * 100
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity T")
    ax.set_zlabel("Local vol (%)")
    ax.set_title("Dupire local volatility surface")
    fig.tight_layout()
    pad2 = os.path.join(output_dir, "dupire_3d.png")
    fig.savefig(pad2, dpi=120)
    plt.close(fig)

    return pad1, pad2