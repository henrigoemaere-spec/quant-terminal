"""cboe_data.py — Haal optieketens op via CBOE delayed quotes API"""
import os
import json
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime
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


def implied_vol(prijs, S, K, T, r, optie="call", q=0.0):
    f = bs_call if optie == "call" else bs_put
    def g(sigma):
        return f(S, K, T, r, sigma, q) - prijs
    try:
        return brentq(g, 1e-6, 5.0)
    except ValueError:
        return np.nan


def haal_cboe_data(symbool="SPX", timeout=15):
    """Haalt de volledige optieketen op van CBOE."""
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{symbool}.json"

    print(f"  Ophalen van CBOE: {symbool}...")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"CBOE API fout: {e}")

    if "data" not in data:
        raise ValueError(f"Onverwacht formaat van CBOE voor {symbool}")

    spot = float(data["data"].get("current_price", 0))
    if spot <= 0:
        raise ValueError(f"Geen spot voor {symbool}")

    opties = data["data"].get("options", [])
    print(f"  Spot {symbool}: {spot:.2f}")
    print(f"  {len(opties)} opties ontvangen")

    vandaag = datetime.now().date()
    rijen = []

    for opt in opties:
        try:
            symbool_str = opt.get("option", "")
            if len(symbool_str) < 15:
                continue

            code = symbool_str[-15:]
            exp_str = code[0:6]
            cp = code[6]
            strike_str = code[7:15]

            exp_date = datetime.strptime(exp_str, "%y%m%d").date()
            K = float(strike_str) / 1000.0

            dagen = (exp_date - vandaag).days
            if dagen <= 0:
                continue
            T = dagen / 365.25

            bid = float(opt.get("bid", 0) or 0)
            ask = float(opt.get("ask", 0) or 0)
            iv_cboe = float(opt.get("iv", 0) or 0)
            volume = float(opt.get("volume", 0) or 0)
            oi = float(opt.get("open_interest", 0) or 0)
            delta = float(opt.get("delta", 0) or 0)

            if bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2

            rijen.append({
                "S": spot,
                "K": K,
                "T": round(T, 4),
                "expiry": exp_date.strftime("%Y-%m-%d"),
                "type": "call" if cp == "C" else "put",
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "iv_cboe": iv_cboe,
                "volume": volume,
                "openInterest": oi,
                "delta": delta,
            })
        except Exception:
            continue

    if not rijen:
        raise ValueError(f"Geen geldige opties voor {symbool}")

    df = pd.DataFrame(rijen)
    return spot, df


def sla_cboe_data_op(symbool="SPX", min_volume=50, max_T=1.5):
    """Haalt CBOE data op en slaat op als CSV."""
    spot, df = haal_cboe_data(symbool)

    print(f"\n  Ruw: {len(df)} opties")

    # Eerste 15 expiraties (meer T-variatie)
    expiraties = sorted(df["expiry"].unique())
    eerste = expiraties[:15]
    df = df[df["expiry"].isin(eerste)]
    print(f"  Na filter eerste 15 expiraties: {len(df)}")

    df = df[df["T"] <= max_T]
    print(f"  Na filter T <= {max_T}: {len(df)}")

    df = df[(df["volume"] >= min_volume) | (df["openInterest"] >= min_volume * 10)]
    print(f"  Na filter liquiditeit: {len(df)}")

    df = df[(df["iv_cboe"] > 0.03) & (df["iv_cboe"] < 1.5)]
    print(f"  Na filter IV-range: {len(df)}")

    basis = os.path.dirname(os.path.abspath(__file__))
    pad = os.path.join(basis, f"cboe_{symbool}.csv")
    df.to_csv(pad, index=False)
    print(f"\n  Opgeslagen in: {pad}")

    return spot, df, pad


def toon_cboe_overzicht(df):
    """Print overzicht van de CBOE data."""
    print(f"\n  CBOE Data-overzicht:")
    print(f"  {'-' * 70}")
    print(f"  Totaal: {len(df)} opties")
    print(f"  Expiraties: {df['expiry'].nunique()}")
    print(f"  Strikes: {df['K'].nunique()}")
    print(f"  Calls: {len(df[df['type'] == 'call'])}, Puts: {len(df[df['type'] == 'put'])}")
    print()
    print(f"  Per expiratie (calls):")
    calls = df[df["type"] == "call"]
    for exp, groep in calls.groupby("expiry"):
        T = groep["T"].iloc[0]
        spot = groep["S"].iloc[0]
        atm_idx = (groep["K"] - spot).abs().idxmin()
        iv_atm = groep.loc[atm_idx, "iv_cboe"]
        iv_min = groep["iv_cboe"].min()
        iv_max = groep["iv_cboe"].max()
        print(f"    {exp} (T={T:.3f}): {len(groep):3d} calls, "
              f"ATM IV={iv_atm:.2%}, range [{iv_min:.1%}, {iv_max:.1%}]")
    print()