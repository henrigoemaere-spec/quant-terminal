"""market_data.py — Haal echte optieketens op via Yahoo Finance en CBOE"""
import os
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


# ============================================================
# Yahoo Finance — haal optieketens op
# ============================================================
def haal_yahoo_data(ticker="SPY", max_expiries=4, min_volume=10):
    """
    Haalt optieketens op via Yahoo Finance.

    ticker     : symbool (SPY, AAPL, QQQ, etc.)
    max_expiries : maximaal aantal expiraties om op te halen
    min_volume : filter opties met te weinig volume

    Retourneert: (spot, r, q, DataFrame)
    DataFrame-kolommen: K, T, call_prijs, put_prijs, iv_markt, volume, openInterest
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("Installeer yfinance: pip install yfinance")

    t = yf.Ticker(ticker)
    info = t.info
    spot = info.get("currentPrice") or info.get("regularMarketPrice")
    if spot is None:
        hist = t.history(period="1d")
        if len(hist) > 0:
            spot = float(hist["Close"].iloc[-1])
        else:
            raise ValueError(f"Kan spotprijs voor {ticker} niet ophalen")

    # Dividend yield
    div_yield = info.get("dividendYield", 0.0) or 0.0
    q = float(div_yield)
    r = 0.045  # aanname — kan vervangen worden door actuele risicovrije rente

    expiraties = t.options
    if not expiraties:
        raise ValueError(f"Geen opties beschikbaar voor {ticker}")

    print(f"  Spot {ticker}: {spot:.2f}")
    print(f"  Dividend yield: {q:.2%}")
    print(f"  Rente (aanname): {r:.2%}")
    print(f"  Beschikbare expiraties: {len(expiraties)}")
    print(f"  Ophalen eerste {min(max_expiries, len(expiraties))}...")

    vandaag = datetime.now().date()
    rijen = []

    for exp_str in expiraties[:max_expiries]:
        try:
            keten = t.option_chain(exp_str)
            calls = keten.calls
            puts = keten.puts

            # Bereken T (in jaren)
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dagen = (exp_date - vandaag).days
            if dagen <= 0:
                continue
            T = dagen / 365.25

            # Merge calls en puts op strike
            for _, call in calls.iterrows():
                K = float(call["strike"])
                call_prijs = float(call.get("lastPrice", 0) or 0)
                if call_prijs <= 0:
                    call_prijs = float((call.get("bid", 0) + call.get("ask", 0)) / 2 or 0)
                if call_prijs <= 0:
                    continue
                volume = float(call.get("volume", 0) or 0)
                oi = float(call.get("openInterest", 0) or 0)
                if volume < min_volume and oi < min_volume * 10:
                    continue

                # Zoek bijbehorende put
                put_row = puts[puts["strike"] == K]
                put_prijs = np.nan
                if len(put_row) > 0:
                    p = put_row.iloc[0]
                    put_prijs = float(p.get("lastPrice", 0) or 0)
                    if put_prijs <= 0:
                        put_prijs = float((p.get("bid", 0) + p.get("ask", 0)) / 2 or 0)

                iv_markt = implied_vol(call_prijs, spot, K, T, r, "call", q)

                rijen.append({
                    "ticker": ticker,
                    "S": spot,
                    "K": K,
                    "T": round(T, 4),
                    "expiry": exp_str,
                    "call_prijs": round(call_prijs, 4),
                    "put_prijs": round(put_prijs, 4) if not np.isnan(put_prijs) else None,
                    "iv_markt": round(iv_markt, 4) if not np.isnan(iv_markt) else None,
                    "volume": volume,
                    "openInterest": oi,
                })
        except Exception as e:
            print(f"    Fout bij expiratie {exp_str}: {e}")
            continue

    if not rijen:
        raise ValueError("Geen data opgehaald")

    df = pd.DataFrame(rijen)
    df = df.dropna(subset=["iv_markt"])
    # Filter onrealistische IV's
    df = df[(df["iv_markt"] > 0.01) & (df["iv_markt"] < 2.0)]

    return spot, r, q, df


def sla_yahoo_data_op(ticker="SPY", max_expiries=4, min_volume=10):
    """Haalt Yahoo-data op en slaat op als CSV."""
    spot, r, q, df = haal_yahoo_data(ticker, max_expiries, min_volume)
    basis = os.path.dirname(os.path.abspath(__file__))
    pad = os.path.join(basis, f"market_data_{ticker}.csv")
    df.to_csv(pad, index=False)
    print(f"\n  {len(df)} opties opgeslagen in: {pad}")
    return spot, r, q, df, pad


# ============================================================
# Overzicht van de data
# ============================================================
def toon_overzicht(df):
    """Print een overzicht van de opgehaalde data."""
    print(f"\n  Data-overzicht:")
    print(f"  {'-' * 60}")
    print(f"  Totaal aantal opties: {len(df)}")
    print(f"  Aantal expiraties: {df['expiry'].nunique()}")
    print(f"  Aantal strikes: {df['K'].nunique()}")
    print()
    print(f"  Per expiratie:")
    for exp, groep in df.groupby("expiry"):
        T = groep["T"].iloc[0]
        strikes = f"{groep['K'].min():.0f} - {groep['K'].max():.0f}"
        iv_min = groep["iv_markt"].min()
        iv_max = groep["iv_markt"].max()
        iv_atm = groep.iloc[(groep["K"] - groep.iloc[0]["S"]).abs().argsort()[:1]]["iv_markt"].iloc[0]
        print(f"    {exp} (T={T:.3f}): {len(groep):3d} strikes {strikes:>13s}  "
              f"ATM IV={iv_atm:.2%}  bereik [{iv_min:.1%}, {iv_max:.1%}]")
    print()