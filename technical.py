"""technical.py — Technische analyse indicatoren"""
import os
import numpy as np
import pandas as pd


# ============================================================
# Data ophalen
# ============================================================
def haal_ohlc_data(ticker, start, end):
    """Haalt OHLC + volume op via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("Installeer yfinance: pip install yfinance")

    df = yf.download(ticker, start=start, end=end, progress=False,
                     auto_adjust=False)

    if len(df) == 0:
        raise ValueError(f"Geen data voor {ticker}")

    # Herstructureer kolommen als nodig
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df.index.name = "Date"
    return df


# ============================================================
# Moving Averages
# ============================================================
def sma(prices, window=20):
    """Simple Moving Average."""
    return prices.rolling(window=window).mean()


def ema(prices, span=20):
    """Exponential Moving Average."""
    return prices.ewm(span=span, adjust=False).mean()


# ============================================================
# MACD
# ============================================================
def macd(prices, fast=12, slow=26, signal=9):
    """
    MACD = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD, signal)
    Histogram = MACD - Signal
    """
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_lijn = ema_fast - ema_slow
    signal_lijn = macd_lijn.ewm(span=signal, adjust=False).mean()
    histogram = macd_lijn - signal_lijn
    return {
        "macd": macd_lijn,
        "signal": signal_lijn,
        "histogram": histogram,
    }


# ============================================================
# RSI
# ============================================================
def rsi(prices, period=14):
    """Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_waarde = 100 - (100 / (1 + rs))
    return rsi_waarde


# ============================================================
# Bollinger Bands
# ============================================================
def bollinger_bands(prices, window=20, num_std=2):
    """
    Bollinger Bands: middel + 2 standaardafwijkingen.
    """
    midden = sma(prices, window)
    std = prices.rolling(window=window).std()
    boven = midden + num_std * std
    onder = midden - num_std * std
    return {
        "midden": midden,
        "boven": boven,
        "onder": onder,
        "bandbreedte": (boven - onder) / midden * 100,
    }


# ============================================================
# Parabolic SAR
# ============================================================
def parabolic_sar(high, low, af=0.02, max_af=0.2):
    """
    Parabolic SAR (Stop And Reverse).
    """
    n = len(high)
    sar = np.zeros(n)
    sar[0] = low.iloc[0]

    trend = 1  # 1 = uptrend, -1 = downtrend
    ep = high.iloc[0]  # extreme point
    acc = af

    for i in range(1, n):
        sar[i] = sar[i - 1] + acc * (ep - sar[i - 1])

        if trend == 1:  # uptrend
            if low.iloc[i] < sar[i]:
                # Trend reversal
                trend = -1
                sar[i] = ep
                ep = low.iloc[i]
                acc = af
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    acc = min(acc + af, max_af)
        else:  # downtrend
            if high.iloc[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high.iloc[i]
                acc = af
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    acc = min(acc + af, max_af)

    return pd.Series(sar, index=high.index)


# ============================================================
# Stochastic Oscillator
# ============================================================
def stochastic(high, low, close, k_period=14, d_period=3, smooth_k=3):
    """
    Fast en slow stochastic oscillator.
    %K = 100 * (Close - Low_n) / (High_n - Low_n)
    %D = SMA(%K, d_period)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    fast_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    fast_d = fast_k.rolling(window=d_period).mean()

    slow_k = fast_k.rolling(window=smooth_k).mean()
    slow_d = slow_k.rolling(window=d_period).mean()

    return {
        "fast_k": fast_k,
        "fast_d": fast_d,
        "slow_k": slow_k,
        "slow_d": slow_d,
    }


# ============================================================
# Alle indicatoren in één keer
# ============================================================
def bereken_alle_indicatoren(df):
    """Berekent alle indicatoren op een OHLC DataFrame."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    resultaat = {
        "SMA20": sma(close, 20),
        "SMA50": sma(close, 50),
        "SMA100": sma(close, 100),
        "SMA200": sma(close, 200),
        "EMA20": ema(close, 20),
        "EMA50": ema(close, 50),
    }

    macd_dict = macd(close)
    resultaat["MACD"] = macd_dict["macd"]
    resultaat["MACD_signal"] = macd_dict["signal"]
    resultaat["MACD_hist"] = macd_dict["histogram"]

    resultaat["RSI"] = rsi(close)

    bb = bollinger_bands(close)
    resultaat["BB_midden"] = bb["midden"]
    resultaat["BB_boven"] = bb["boven"]
    resultaat["BB_onder"] = bb["onder"]

    resultaat["PSAR"] = parabolic_sar(high, low)

    stoch = stochastic(high, low, close)
    resultaat["stoch_fast_k"] = stoch["fast_k"]
    resultaat["stoch_fast_d"] = stoch["fast_d"]
    resultaat["stoch_slow_k"] = stoch["slow_k"]
    resultaat["stoch_slow_d"] = stoch["slow_d"]

    return resultaat


# ============================================================
# Signalen (interpretatie)
# ============================================================
def genereer_signalen(df, indicatoren):
    """Genereer een overzicht van buy/sell signalen op basis van indicatoren."""
    close = df["Close"].iloc[-1]
    signalen = []

    # RSI
    rsi_huidig = indicatoren["RSI"].iloc[-1]
    if not np.isnan(rsi_huidig):
        if rsi_huidig > 70:
            signalen.append(("RSI", f"{rsi_huidig:.1f}", "Overbought",
                            "🔴 Verkoop signaal"))
        elif rsi_huidig < 30:
            signalen.append(("RSI", f"{rsi_huidig:.1f}", "Oversold",
                            "🟢 Koop signaal"))
        else:
            signalen.append(("RSI", f"{rsi_huidig:.1f}", "Neutraal",
                            "⚪ Geen signaal"))

    # MACD crossover
    macd_huidig = indicatoren["MACD"].iloc[-1]
    macd_sig = indicatoren["MACD_signal"].iloc[-1]
    macd_prev = indicatoren["MACD"].iloc[-2]
    sig_prev = indicatoren["MACD_signal"].iloc[-2]

    if macd_prev < sig_prev and macd_huidig > macd_sig:
        signalen.append(("MACD", f"{macd_huidig:.4f}", "Bullish crossover",
                        "🟢 Koop signaal"))
    elif macd_prev > sig_prev and macd_huidig < macd_sig:
        signalen.append(("MACD", f"{macd_huidig:.4f}", "Bearish crossover",
                        "🔴 Verkoop signaal"))
    else:
        positie = "boven" if macd_huidig > macd_sig else "onder"
        signalen.append(("MACD", f"{macd_huidig:.4f}",
                        f"{positie} signal line", "⚪ Geen crossover"))

    # Bollinger
    bb_boven = indicatoren["BB_boven"].iloc[-1]
    bb_onder = indicatoren["BB_onder"].iloc[-1]
    if close > bb_boven:
        signalen.append(("Bollinger", f"{close:.2f}", "Boven bovenste band",
                        "🔴 Overbought"))
    elif close < bb_onder:
        signalen.append(("Bollinger", f"{close:.2f}", "Onder onderste band",
                        "🟢 Oversold"))
    else:
        signalen.append(("Bollinger", f"{close:.2f}", "Binnen banden",
                        "⚪ Neutraal"))

    # Parabolic SAR
    psar_huidig = indicatoren["PSAR"].iloc[-1]
    if close > psar_huidig:
        signalen.append(("PSAR", f"{psar_huidig:.2f}", "SAR onder prijs",
                        "🟢 Uptrend"))
    else:
        signalen.append(("PSAR", f"{psar_huidig:.2f}", "SAR boven prijs",
                        "🔴 Downtrend"))

    # SMA crossover (50 vs 200 — golden cross / death cross)
    sma50 = indicatoren["SMA50"].iloc[-1]
    sma200 = indicatoren["SMA200"].iloc[-1]
    sma50_prev = indicatoren["SMA50"].iloc[-2]
    sma200_prev = indicatoren["SMA200"].iloc[-2]

    if not (np.isnan(sma50) or np.isnan(sma200)):
        if sma50_prev < sma200_prev and sma50 > sma200:
            signalen.append(("SMA 50/200", f"{sma50:.2f} / {sma200:.2f}",
                            "Golden Cross", "🟢 Sterk koop signaal"))
        elif sma50_prev > sma200_prev and sma50 < sma200:
            signalen.append(("SMA 50/200", f"{sma50:.2f} / {sma200:.2f}",
                            "Death Cross", "🔴 Sterk verkoop signaal"))
        else:
            positie = "boven" if sma50 > sma200 else "onder"
            signalen.append(("SMA 50/200", f"{sma50:.2f} / {sma200:.2f}",
                            f"SMA50 {positie} SMA200", "⚪ Geen crossover"))

    # Stochastic
    stoch_k = indicatoren["stoch_slow_k"].iloc[-1]
    stoch_d = indicatoren["stoch_slow_d"].iloc[-1]
    if not (np.isnan(stoch_k) or np.isnan(stoch_d)):
        if stoch_k > 80 and stoch_d > 80:
            signalen.append(("Stochastic", f"{stoch_k:.1f}",
                            "Overbought zone", "🔴 Verkoop signaal"))
        elif stoch_k < 20 and stoch_d < 20:
            signalen.append(("Stochastic", f"{stoch_k:.1f}",
                            "Oversold zone", "🟢 Koop signaal"))
        else:
            signalen.append(("Stochastic", f"{stoch_k:.1f}",
                            "Neutrale zone", "⚪ Geen signaal"))

    return signalen


def print_samenvatting(ticker, df, indicatoren):
    """Print een samenvatting van de indicatoren."""
    close = df["Close"].iloc[-1]
    datum = df.index[-1].strftime("%Y-%m-%d")

    print()
    print("=" * 70)
    print(f"  TECHNISCHE ANALYSE — {ticker}")
    print("=" * 70)
    print(f"\n  Laatste datum: {datum}")
    print(f"  Slotprijs    : {close:.2f}")
    print()

    signalen = genereer_signalen(df, indicatoren)

    print(f"  {'Indicator':<15} {'Waarde':<22} {'Status':<22} {'Signaal':<20}")
    print("  " + "-" * 80)
    for naam, waarde, status, signaal in signalen:
        print(f"  {naam:<15} {waarde:<22} {status:<22} {signaal:<20}")
    print()


# ============================================================
# Plots
# ============================================================
def plot_technisch(ticker, df, indicatoren, output_dir=None, periode_dagen=180):
    """
    Maakt een multi-paneel technische analyse plot:
    1. Prijs + SMA's + Bollinger Bands + PSAR
    2. Volume
    3. MACD + signal + histogram
    4. RSI
    5. Stochastic
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Laatste N dagen
    df_plot = df.iloc[-periode_dagen:].copy()
    idx = df_plot.index

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1.5, 1.5, 1.5], hspace=0.15)

    # --- Plot 1: prijs + indicatoren ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(idx, df_plot["Close"], color="#e6edf3", linewidth=1.8,
             label=f"{ticker} Close")
    ax1.plot(idx, indicatoren["SMA50"].iloc[-periode_dagen:],
             color="#d97706", linewidth=1.5, alpha=0.8, label="SMA 50")
    ax1.plot(idx, indicatoren["SMA200"].iloc[-periode_dagen:],
             color="#58a6ff", linewidth=1.5, alpha=0.8, label="SMA 200")
    ax1.plot(idx, indicatoren["BB_boven"].iloc[-periode_dagen:],
             color="#8b949e", linewidth=1, linestyle="--", alpha=0.6,
             label="Bollinger boven")
    ax1.plot(idx, indicatoren["BB_onder"].iloc[-periode_dagen:],
             color="#8b949e", linewidth=1, linestyle="--", alpha=0.6,
             label="Bollinger onder")
    ax1.scatter(idx, indicatoren["PSAR"].iloc[-periode_dagen:],
                color="#f85149", s=8, alpha=0.6, label="Parabolic SAR")
    ax1.set_ylabel("Prijs")
    ax1.set_title(f"{ticker} — Technische analyse ({periode_dagen} dagen)",
                  fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9, ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#161b22")

    # --- Plot 2: volume ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    kleuren = ["#3fb950" if df_plot["Close"].iloc[i] >= df_plot["Open"].iloc[i]
               else "#f85149" for i in range(len(df_plot))]
    ax2.bar(idx, df_plot["Volume"], color=kleuren, alpha=0.6, width=1)
    ax2.set_ylabel("Volume")
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#161b22")
    plt.setp(ax2.get_xticklabels(), visible=False)

    # --- Plot 3: MACD ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(idx, indicatoren["MACD"].iloc[-periode_dagen:],
             color="#d97706", linewidth=1.5, label="MACD")
    ax3.plot(idx, indicatoren["MACD_signal"].iloc[-periode_dagen:],
             color="#58a6ff", linewidth=1.5, label="Signal")
    hist = indicatoren["MACD_hist"].iloc[-periode_dagen:]
    kleuren_hist = ["#3fb950" if v >= 0 else "#f85149" for v in hist]
    ax3.bar(idx, hist, color=kleuren_hist, alpha=0.5, width=1)
    ax3.axhline(y=0, color="#8b949e", linewidth=0.5, linestyle="--")
    ax3.set_ylabel("MACD")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor("#161b22")
    plt.setp(ax3.get_xticklabels(), visible=False)

    # --- Plot 4: RSI ---
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(idx, indicatoren["RSI"].iloc[-periode_dagen:],
             color="#d97706", linewidth=1.5)
    ax4.axhline(y=70, color="#f85149", linewidth=0.8, linestyle="--")
    ax4.axhline(y=30, color="#3fb950", linewidth=0.8, linestyle="--")
    ax4.fill_between(idx, 70, 100, color="#f85149", alpha=0.08)
    ax4.fill_between(idx, 0, 30, color="#3fb950", alpha=0.08)
    ax4.set_ylabel("RSI")
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor("#161b22")
    plt.setp(ax4.get_xticklabels(), visible=False)

    # --- Plot 5: Stochastic ---
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.plot(idx, indicatoren["stoch_slow_k"].iloc[-periode_dagen:],
             color="#d97706", linewidth=1.5, label="%K (slow)")
    ax5.plot(idx, indicatoren["stoch_slow_d"].iloc[-periode_dagen:],
             color="#58a6ff", linewidth=1.5, label="%D (slow)")
    ax5.axhline(y=80, color="#f85149", linewidth=0.8, linestyle="--")
    ax5.axhline(y=20, color="#3fb950", linewidth=0.8, linestyle="--")
    ax5.set_ylabel("Stochastic")
    ax5.set_ylim(0, 100)
    ax5.set_xlabel("Datum")
    ax5.legend(loc="upper left", fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_facecolor("#161b22")

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.tick_params(colors="#e6edf3")

    fig.patch.set_facecolor("#0d1117")
    fig.tight_layout()
    pad = os.path.join(output_dir, f"technical_{ticker}.png")
    fig.savefig(pad, dpi=100, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return pad