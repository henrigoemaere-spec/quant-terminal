"""marktdata.py — Realistische synthetische marktdata (equity index)"""
import numpy as np
import pandas as pd
import os


def genereer_realistische_marktdata(S=4500.0, r=0.03, q=0.015,
                                     seed=42, ruis_niveau=0.002):
    """Realistische S&P-achtige optie-surface met skew, term structure en ruis."""
    rng = np.random.default_rng(seed)
    strikes = np.array([4000, 4100, 4200, 4300, 4400, 4450, 4500,
                        4550, 4600, 4700, 4800, 4900, 5000], dtype=float)
    maturities = np.array([1/12, 2/12, 3/12, 6/12, 1.0, 1.5, 2.0])
    from scipy.stats import norm

    rijen = []
    for T in maturities:
        atm_vol = 0.15 + 0.03 * (1 - np.exp(-5 * T))
        skew = -0.15 * np.exp(-2 * T) - 0.05
        smile = 0.40
        for K in strikes:
            m = np.log(K / S)
            iv_clean = atm_vol + skew * m + smile * m ** 2
            ruis = rng.normal(0, ruis_niveau)
            iv_markt = iv_clean + ruis
            d1 = (np.log(S / K) + (r - q + 0.5 * iv_markt ** 2) * T) / (iv_markt * np.sqrt(T))
            d2 = d1 - iv_markt * np.sqrt(T)
            call_prijs = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            put_prijs = call_prijs - S * np.exp(-q * T) + K * np.exp(-r * T)
            rijen.append({
                "S": S, "K": K, "T": T, "r": r, "q": q,
                "call_prijs": round(call_prijs, 4),
                "put_prijs": round(put_prijs, 4),
                "iv_markt": round(iv_markt, 4),
                "iv_clean": round(iv_clean, 4),
            })
    return pd.DataFrame(rijen)


def sla_realistische_marktdata_op(pad=None, S=4500.0, r=0.03, q=0.015, seed=42):
    if pad is None:
        basis = os.path.dirname(os.path.abspath(__file__))
        pad = os.path.join(basis, "marktdata_realistisch.csv")
    df = genereer_realistische_marktdata(S, r, q, seed)
    df.to_csv(pad, index=False)
    return pad, df