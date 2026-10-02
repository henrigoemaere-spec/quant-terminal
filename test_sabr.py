"""test_sabr.py — SABR op CBOE SPX met diagnose"""
import pandas as pd
import numpy as np
import sabr as sb
from scipy.stats import norm
from scipy.optimize import brentq


def bs_call(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def iv_uit_mid(mid, S, K, T, r, q=0.0):
    def g(sigma):
        return bs_call(S, K, T, r, sigma, q) - mid
    try:
        return brentq(g, 1e-6, 5.0)
    except ValueError:
        return np.nan


df = pd.read_csv("cboe_SPX.csv")
spot = df["S"].iloc[0]
r, q = 0.045, 0.0

print(f"\nSPX: S={spot:.2f}, r={r:.4f}, q={q:.4f}")
print(f"Ruwe data: {len(df)} opties\n")

# Alleen calls
calls = df[df["type"] == "call"].copy()

# Diagnose: vergelijk CBOE IV met IV uit mid
print("DIAGNOSE: CBOE IV vs IV uit mid-prijs")
print(f"{'K':>8} {'T':>8} {'mid':>10} {'iv_cboe':>10} {'iv_mid':>10} {'verschil':>10}")
print("-" * 60)

eerste_exp = calls["expiry"].unique()[0]
sample = calls[calls["expiry"] == eerste_exp].head(10)
for _, row in sample.iterrows():
    iv_mid = iv_uit_mid(row["mid"], spot, row["K"], row["T"], r, q)
    verschil = row["iv_cboe"] - iv_mid
    print(f"{row['K']:>8.0f} {row['T']:>8.4f} {row['mid']:>10.2f} "
          f"{row['iv_cboe']:>10.4f} {iv_mid:>10.4f} {verschil:>+10.4f}")
print()

# Filter strikes binnen 10% van forward
F_approx = spot * np.exp((r - q) * calls["T"])
calls = calls[(calls["K"] > F_approx * 0.90) & (calls["K"] < F_approx * 1.10)]
print(f"Na filter strikes 90-110%: {len(calls)}")

# IV range
calls = calls[(calls["iv_cboe"] > 0.05) & (calls["iv_cboe"] < 0.50)]
print(f"Na filter IV 5-50%: {len(calls)}\n")

# Per expiratie
geldige = []
print("Per expiratie:")
for exp, groep in calls.groupby("expiry"):
    if len(groep) < 8:
        continue
    T = groep["T"].iloc[0]
    geldige.append(exp)
    iv_min = groep["iv_cboe"].min()
    iv_max = groep["iv_cboe"].max()
    atm_idx = (groep["K"] - spot).abs().idxmin()
    iv_atm = groep.loc[atm_idx, "iv_cboe"]
    print(f"  {exp} (T={T:.3f}): {len(groep):2d} strikes, "
          f"ATM={iv_atm:.2%}, range [{iv_min:.1%}, {iv_max:.1%}]")

calls = calls[calls["expiry"].isin(geldige)]

if len(calls) < 20:
    print("\nTe weinig data voor SABR.")
    raise SystemExit()

print()

# Kalibreer SABR
markt = [(row["K"], row["T"], row["iv_cboe"]) for _, row in calls.iterrows()]
params = sb.kalibreer_sabr_volledig(markt, spot, r, q, beta=0.5)

print(f"{'T':>8} {'alpha':>8} {'rho':>8} {'nu':>8} {'resid':>12}")
print("-" * 55)
for T, p in params.items():
    print(f"{T:>8.3f} {p['alpha']:>8.4f} {p['rho']:>+8.4f} "
          f"{p['nu']:>8.4f} {p['residueel']:>12.2e}")

pad = sb.plot_sabr_fit(markt, spot, r, q, params)
print(f"\nPlot opgeslagen: {pad}")