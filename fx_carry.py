"""fx_carry.py — Carry trades, valuta-exposure en valutarisico."""
import numpy as np
import pandas as pd
import yfinance as yf


RENTES = {
    "USD": 4.75, "EUR": 3.25, "GBP": 4.75, "JPY": 0.25,
    "CHF": 0.50, "AUD": 4.35, "NZD": 4.75, "CAD": 3.75,
    "MXN": 10.25, "TRY": 50.00, "ZAR": 8.25,
}

CARRY_PAREN = {
    "AUDJPY=X": ("JPY", "AUD"),
    "NZDJPY=X": ("JPY", "NZD"),
    "USDJPY=X": ("JPY", "USD"),
    "USDMXN=X": ("USD", "MXN"),
    "USDTRY=X": ("USD", "TRY"),
    "USDZAR=X": ("USD", "ZAR"),
}


def carry_score(pair, rente_funding, rente_target, vol_jaar):
    carry = rente_target - rente_funding
    if vol_jaar <= 0:
        return carry, 0.0
    return carry, carry / vol_jaar


def maak_carry_rapport(start="2023-01-01"):
    rijen = []
    for pair, (funding, target) in CARRY_PAREN.items():
        try:
            data = yf.download(pair, start=start, progress=False, auto_adjust=True)
            if data.empty:
                continue
            close = data["Close"].squeeze()
            ret = np.log(close / close.shift(1)).dropna()
            vol_jaar = ret.std() * np.sqrt(252) * 100
            rf = RENTES.get(funding, 0.0)
            rt = RENTES.get(target, 0.0)
            carry, score = carry_score(pair, rf, rt, vol_jaar)
            rijen.append({
                "pair": pair, "funding": funding, "target": target,
                "rente_funding": rf, "rente_target": rt,
                "carry": carry, "vol_jaar": vol_jaar,
                "score": score, "spot": float(close.iloc[-1]),
            })
        except Exception as e:
            print(f"  {pair}: FOUT — {e}")
    return rijen


def print_carry_rapport(rijen):
    print(f"\n  Carry-trade analyse")
    print(f"  {'=' * 78}")
    print(f"  {'Pair':<10} {'Funding':>8} {'Target':>7} {'Carry%':>8} "
          f"{'Vol%':>7} {'Score':>8} {'Spot':>10}")
    print(f"  {'-' * 78}")
    for r in sorted(rijen, key=lambda x: -x["score"]):
        print(f"  {r['pair']:<10} {r['funding']:>8} {r['target']:>7} "
              f"{r['carry']:>8.2f} {r['vol_jaar']:>7.2f} "
              f"{r['score']:>8.3f} {r['spot']:>10.4f}")
    print()
    print("  Score > 0.5  : aantrekkelijke risk-adjusted carry")
    print("  Score 0.2-0.5: matig, let op risk-off momenten")
    print("  Score < 0.2  : carry weegt niet op tegen volatiliteit")
    print()


def bereken_exposure(posities):
    totaal = sum(p["waarde"] for p in posities)
    if totaal == 0:
        return {}, 0.0
    per_valuta = {}
    for p in posities:
        per_valuta[p["valuta"]] = per_valuta.get(p["valuta"], 0) + p["waarde"]
    exposure = {v: w / totaal * 100 for v, w in per_valuta.items()}
    return exposure, totaal


def print_exposure_rapport(posities):
    exposure, totaal = bereken_exposure(posities)
    print(f"\n  Valuta-exposure in portefeuille")
    print(f"  {'=' * 55}")
    print(f"  Totale waarde: {totaal:,.2f}")
    print(f"  {'-' * 55}")
    print(f"  {'Valuta':<8} {'Bedrag':>15} {'Percentage':>12}")
    print(f"  {'-' * 55}")
    for v, pct in sorted(exposure.items(), key=lambda x: -x[1]):
        bedrag = totaal * pct / 100
        print(f"  {v:<8} {bedrag:>15,.2f} {pct:>11.2f}%")
    print()
    if len(exposure) > 1:
        grootste = max(exposure.values())
        if grootste > 50:
            print(f"  ! Concentratierisico: {grootste:.1f}% in één valuta.")
        else:
            print(f"  Verdeling redelijk gespreid (max {grootste:.1f}%).")
    print()


def risico_score(exposure_pct, fx_signaal):
    signaal_score = {"KOOP": 1.0, "Geen": 0.5, "VERKOOP": -1.0}.get(fx_signaal, 0.0)
    return exposure_pct / 100 * signaal_score


def print_risico_rapport(posities, fx_signalen):
    exposure, _ = bereken_exposure(posities)
    print(f"\n  Valutarisico in portefeuille")
    print(f"  {'=' * 70}")
    print(f"  {'Valuta':<8} {'Exposure':>10} {'FX-signaal':>12} {'Risicoscore':>13}")
    print(f"  {'-' * 70}")
    totaal_score = 0.0
    for v, pct in sorted(exposure.items(), key=lambda x: -x[1]):
        sig = fx_signalen.get(v, "Geen")
        score = risico_score(pct, sig)
        totaal_score += score
        print(f"  {v:<8} {pct:>9.2f}% {sig:>12} {score:>13.3f}")
    print(f"  {'-' * 70}")
    print(f"  {'Totaal':<8} {'':>10} {'':>12} {totaal_score:>13.3f}")
    print()
    if totaal_score > 0.3:
        print("  -> Positief: valutablootstelling werkt mee met FX-trend.")
    elif totaal_score < -0.3:
        print("  -> Negatief: overweeg hedge of herweging van valuta's.")
    else:
        print("  -> Neutraal: geen duidelijke richting in valutarisico.")
    print()