"""fx.py — Valuta-analyse module"""
import os
import numpy as np
import pandas as pd
import datetime as dt


# ============================================================
# Data ophalen
# ============================================================
def haal_fx_data(pair, start="2020-01-01", end=None):
    """
    Haalt FX data op via Yahoo Finance.

    pair: valutapaar zoals 'EURUSD=X', 'EURGBP=X', 'USDJPY=X'
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("Installeer yfinance: pip install yfinance")

    if end is None:
        end = dt.datetime.now().strftime("%Y-%m-%d")

    if not pair.endswith("=X"):
        pair = pair + "=X"

    df = yf.download(pair, start=start, end=end, progress=False,
                     auto_adjust=True)

    if len(df) == 0:
        raise ValueError(f"Geen FX data voor {pair}")

    # Splits kolommen
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" in df.columns:
        close = df["Close"]
    else:
        close = df.iloc[:, 0]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close.name = pair.replace("=X", "")
    close = close.dropna()
    return close


def haal_meerdere_fx(pairs, start="2020-01-01", end=None):
    """Haalt meerdere valutaparen op."""
    resultaat = {}
    for pair in pairs:
        try:
            serie = haal_fx_data(pair, start, end)
            resultaat[serie.name] = serie
        except Exception as e:
            print(f"  Fout bij {pair}: {e}")
    if not resultaat:
        raise ValueError("Geen FX data opgehaald")
    df = pd.concat(resultaat.values(), axis=1)
    df.columns = list(resultaat.keys())
    df = df.dropna()
    return df


# ============================================================
# Analyse
# ============================================================
def bereken_statistieken(serie):
    """Basis statistieken voor een FX pair."""
    returns = np.log(serie / serie.shift(1)).dropna()
    n = 252

    return {
        "spot": float(serie.iloc[-1]),
        "spot_prev": float(serie.iloc[-2]) if len(serie) > 1 else float(serie.iloc[-1]),
        "spot_change_pct": float((serie.iloc[-1] - serie.iloc[-2]) / serie.iloc[-2] * 100) if len(serie) > 1 else 0.0,
        "mu_dag": float(returns.mean()),
        "sigma_dag": float(returns.std()),
        "vol_jaar": float(returns.std() * np.sqrt(n) * 100),
        "min_52w": float(serie.iloc[-252:].min()) if len(serie) >= 252 else float(serie.min()),
        "max_52w": float(serie.iloc[-252:].max()) if len(serie) >= 252 else float(serie.max()),
        "min_hist": float(serie.min()),
        "max_hist": float(serie.max()),
        "laatste_datum": serie.index[-1].strftime("%Y-%m-%d"),
        "n_observaties": len(serie),
        "returns": returns,
    }


def technische_analyse(serie):
    """Technische indicatoren voor een FX pair."""
    close = serie

    # SMA
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    # RSI (14 dagen)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    return {
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": signal,
        "macd_hist": histogram,
        "bb_mid": bb_mid,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
    }


def genereer_signalen(serie, stats, tech):
    """Genereert buy/sell signalen op basis van technische indicatoren."""
    signalen = []
    close = float(serie.iloc[-1])

    # RSI
    rsi_huidig = float(tech["rsi"].iloc[-1])
    if not np.isnan(rsi_huidig):
        if rsi_huidig > 70:
            signalen.append(("RSI", f"{rsi_huidig:.1f}", "Overbought", "VERKOOP"))
        elif rsi_huidig < 30:
            signalen.append(("RSI", f"{rsi_huidig:.1f}", "Oversold", "KOOP"))
        else:
            signalen.append(("RSI", f"{rsi_huidig:.1f}", "Neutraal", "Geen"))

    # MACD
    macd_h = float(tech["macd"].iloc[-1])
    sig_h = float(tech["macd_signal"].iloc[-1])
    macd_p = float(tech["macd"].iloc[-2]) if len(tech["macd"]) > 1 else macd_h
    sig_p = float(tech["macd_signal"].iloc[-2]) if len(tech["macd_signal"]) > 1 else sig_h

    if macd_p < sig_p and macd_h > sig_h:
        signalen.append(("MACD", f"{macd_h:.5f}", "Bullish crossover", "KOOP"))
    elif macd_p > sig_p and macd_h < sig_h:
        signalen.append(("MACD", f"{macd_h:.5f}", "Bearish crossover", "VERKOOP"))
    else:
        positie = "boven" if macd_h > sig_h else "onder"
        signalen.append(("MACD", f"{macd_h:.5f}", f"{positie} signal", "Geen"))

    # SMA 50 vs 200 (Golden/Death Cross)
    sma50_v = float(tech["sma50"].iloc[-1])
    sma200_v = float(tech["sma200"].iloc[-1])
    if not (np.isnan(sma50_v) or np.isnan(sma200_v)):
        if sma50_v > sma200_v:
            signalen.append(("SMA 50/200", f"{sma50_v:.4f}/{sma200_v:.4f}",
                            "Golden Cross (bullish)", "KOOP"))
        else:
            signalen.append(("SMA 50/200", f"{sma50_v:.4f}/{sma200_v:.4f}",
                            "Death Cross (bearish)", "VERKOOP"))

    # Bollinger
    bb_up = float(tech["bb_upper"].iloc[-1])
    bb_dn = float(tech["bb_lower"].iloc[-1])
    if close > bb_up:
        signalen.append(("Bollinger", f"{close:.4f}", "Boven bovenste band", "VERKOOP"))
    elif close < bb_dn:
        signalen.append(("Bollinger", f"{close:.4f}", "Onder onderste band", "KOOP"))
    else:
        signalen.append(("Bollinger", f"{close:.4f}", "Binnen banden", "Geen"))

    return signalen


def carry_trade_analyse(pair, rente_base, rente_quote):
    """
    Berekent carry trade rendement.

    pair: bv 'AUDJPY=X' (base = AUD, quote = JPY)
    rente_base: rente van base valuta (in %)
    rente_quote: rente van quote valuta (in %)
    """
    verschil = rente_base - rente_quote
    return {
        "carry": verschil,
        "richting": "Long " + pair[:3] if verschil > 0 else "Short " + pair[:3],
        "advies": (
            f"Leen in {pair[3:6]} ({rente_quote:.2f}%), "
            f"beleg in {pair[:3]} ({rente_base:.2f}%)"
            if verschil > 0 else
            f"Leen in {pair[:3]} ({rente_base:.2f}%), "
            f"beleg in {pair[3:6]} ({rente_quote:.2f}%)"
        ),
    }


# ============================================================
# Rapport
# ============================================================
def maak_fx_rapport(pair, start="2020-01-01", end=None):
    """Genereert een volledig FX rapport."""
    serie = haal_fx_data(pair, start, end)
    stats = bereken_statistieken(serie)
    tech = technische_analyse(serie)
    signalen = genereer_signalen(serie, stats, tech)
    return {
        "pair": pair.replace("=X", ""),
        "serie": serie,
        "stats": stats,
        "tech": tech,
        "signalen": signalen,
    }


def print_fx_rapport(rapport):
    """Print het FX rapport."""
    s = rapport["stats"]
    print()
    print("=" * 70)
    print(f"  FX ANALYSE — {rapport['pair']}")
    print("=" * 70)
    print()
    print(f"  Laatste datum: {s['laatste_datum']}")
    print(f"  Spot rate    : {s['spot']:.4f}")
    print(f"  Verandering  : {s['spot_change_pct']:+.4f}%")
    print()
    print(f"  Statistieken:")
    print(f"    52-week range : {s['min_52w']:.4f} — {s['max_52w']:.4f}")
    print(f"    Hist. range   : {s['min_hist']:.4f} — {s['max_hist']:.4f}")
    print(f"    Volatiliteit  : {s['vol_jaar']:.2f}% (jaarbasis)")
    print(f"    Observaties   : {s['n_observaties']} dagen")
    print()
    print(f"  Signalen:")
    print(f"    {'Indicator':<15} {'Waarde':<20} {'Status':<25} {'Signaal':<12}")
    print("  " + "-" * 80)
    for naam, waarde, status, signaal in rapport["signalen"]:
        print(f"    {naam:<15} {waarde:<20} {status:<25} {signaal:<12}")
    print()


# ============================================================
# Plot
# ============================================================
def plot_fx(rapport, output_dir=None, dagen=252):
    """Maakt 3-paneel FX plot: prijs + SMA, MACD, RSI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(output_dir, exist_ok=True)

    serie = rapport["serie"].iloc[-dagen:]
    tech = rapport["tech"]
    idx = serie.index

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.15)

    # Plot 1: prijs + SMA + Bollinger
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(idx, serie.values, color="#e6edf3", linewidth=1.8, label="Spot")
    ax1.plot(idx, tech["sma50"].iloc[-dagen:], color="#d97706",
             linewidth=1.5, label="SMA 50")
    ax1.plot(idx, tech["sma200"].iloc[-dagen:], color="#58a6ff",
             linewidth=1.5, label="SMA 200")
    ax1.plot(idx, tech["bb_upper"].iloc[-dagen:], color="#8b949e",
             linewidth=1, linestyle="--", alpha=0.6, label="BB boven")
    ax1.plot(idx, tech["bb_lower"].iloc[-dagen:], color="#8b949e",
             linewidth=1, linestyle="--", alpha=0.6, label="BB onder")
    ax1.set_title(f"{rapport['pair']} — FX Analyse ({dagen} dagen)",
                  fontsize=14, fontweight="bold", color="#e6edf3")
    ax1.set_ylabel("Rate", color="#e6edf3")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#161b22")
    ax1.tick_params(colors="#e6edf3")

    # Plot 2: MACD
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(idx, tech["macd"].iloc[-dagen:], color="#d97706",
             linewidth=1.5, label="MACD")
    ax2.plot(idx, tech["macd_signal"].iloc[-dagen:], color="#58a6ff",
             linewidth=1.5, label="Signal")
    hist = tech["macd_hist"].iloc[-dagen:]
    kleuren = ["#3fb950" if v >= 0 else "#f85149" for v in hist]
    ax2.bar(idx, hist, color=kleuren, alpha=0.5, width=1)
    ax2.axhline(y=0, color="#8b949e", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("MACD", color="#e6edf3")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#161b22")
    ax2.tick_params(colors="#e6edf3")
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Plot 3: RSI
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(idx, tech["rsi"].iloc[-dagen:], color="#d97706", linewidth=1.5)
    ax3.axhline(y=70, color="#f85149", linewidth=0.8, linestyle="--")
    ax3.axhline(y=30, color="#3fb950", linewidth=0.8, linestyle="--")
    ax3.fill_between(idx, 70, 100, color="#f85149", alpha=0.08)
    ax3.fill_between(idx, 0, 30, color="#3fb950", alpha=0.08)
    ax3.set_ylabel("RSI", color="#e6edf3")
    ax3.set_ylim(0, 100)
    ax3.set_xlabel("Datum", color="#e6edf3")
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor("#161b22")
    ax3.tick_params(colors="#e6edf3")

    fig.patch.set_facecolor("#0d1117")
    pad = os.path.join(output_dir, f"fx_{rapport['pair']}.png")
    fig.savefig(pad, dpi=100, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return pad