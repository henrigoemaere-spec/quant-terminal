"""test_dupire.py — Directe debug van Dupire local volatility"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
import marktdata as md

print("Marktdata genereren...")
pad, df = md.sla_realistische_marktdata_op(None, 4500.0, 0.03, 0.015, seed=42)
print(f"Data: {len(df)} prijzen\n")

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

print("IV-matrix ruw (in %):")
print(f"{'T':>8} | " + " ".join(f"{k:>7.0f}" for k in strikes))
for i, T in enumerate(maturities):
    print(f"{T:>8.3f} | " + " ".join(f"{100*iv_matrix[i,j]:>7.2f}" for j in range(len(strikes))))

iv_smooth = gaussian_filter(iv_matrix, sigma=1.5, mode="nearest")

print("\nIV-matrix gesmoothd (in %):")
print(f"{'T':>8} | " + " ".join(f"{k:>7.0f}" for k in strikes))
for i, T in enumerate(maturities):
    print(f"{T:>8.3f} | " + " ".join(f"{100*iv_smooth[i,j]:>7.2f}" for j in range(len(strikes))))

spline = RectBivariateSpline(maturities, strikes, iv_smooth, kx=2, ky=2)


def iv_at(T, K):
    """Helper: haal scalar IV op uit spline."""
    return float(spline(T, K)[0, 0])


S = 4500.0
r = 0.03
q = 0.015

print("\n--- Debug per gridpunt ---")
print(f"{'T':>6} {'K':>6} | {'iv':>6} {'dT':>10} {'dK':>10} {'d2K':>10} | "
      f"{'numer':>10} {'denom':>10} | {'lv':>8}")

K_test = np.linspace(strikes.min() + 50, strikes.max() - 50, 8)
T_test = np.linspace(maturities.min() + 0.05, maturities.max() - 0.05, 6)

for T in T_test:
    for K in K_test:
        try:
            iv = iv_at(T, K)
            eps_T = 0.02
            eps_K = 0.03 * K

            dIV_dT = (iv_at(T + eps_T, K) - iv_at(T - eps_T, K)) / (2 * eps_T)
            iv_up = iv_at(T, K + eps_K)
            iv_dn = iv_at(T, K - eps_K)
            dIV_dK = (iv_up - iv_dn) / (2 * eps_K)
            d2IV_dK2 = (iv_up - 2 * iv + iv_dn) / (eps_K ** 2)

            d1 = (np.log(S / K) + (r - q + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
            d2 = d1 - iv * np.sqrt(T)

            numer = iv ** 2 + 2 * iv * T * (dIV_dT + (r - q) * K * dIV_dK)
            denom = ((1 + K * d1 * np.sqrt(T) * dIV_dK) ** 2
                     + K ** 2 * T * iv * (d2IV_dK2 - d1 * np.sqrt(T) * dIV_dK ** 2))

            if abs(denom) < 1e-10:
                lv = 0.0
            else:
                lv2 = numer / denom
                lv = np.sqrt(lv2) if lv2 > 0 else -np.sqrt(-lv2)

            print(f"{T:>6.3f} {K:>6.0f} | {iv:>6.4f} {dIV_dT:>10.4f} {dIV_dK:>10.6f} "
                  f"{d2IV_dK2:>10.4f} | {numer:>10.4f} {denom:>10.4f} | {lv:>8.4f}")
        except Exception as e:
            print(f"{T:>6.3f} {K:>6.0f} | FOUT: {e}")