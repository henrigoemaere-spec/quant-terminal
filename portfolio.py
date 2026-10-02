"""portfolio.py — Portfolio-constructie, CAPM, performance-ratios, VaR, drawdown"""
import os
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import norm


def haal_aandelen_data(tickers, start, end):
    """Haalt slotkoersen op voor een lijst tickers."""
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("Installeer yfinance: pip install yfinance")

    series_lijst = []
    geldige_tickers = []

    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False,
                             auto_adjust=True)
            if len(df) > 0:
                # Zorg dat we een Series hebben
                serie = df["Close"]
                if isinstance(serie, pd.DataFrame):
                    serie = serie.iloc[:, 0]
                serie.name = t
                series_lijst.append(serie)
                geldige_tickers.append(t)
        except Exception:
            continue

    if not series_lijst:
        raise ValueError("Geen data opgehaald voor de opgegeven tickers")

    # Concat met axis=1 (kolommen)
    df = pd.concat(series_lijst, axis=1)
    df.columns = geldige_tickers
    df = df.dropna()
    return df


def bouw_portefeuille(prijzen, weights=None, methode="gelijk"):
    """Bouwt een portefeuille uit prijzen DataFrame."""
    tickers = prijzen.columns.tolist()
    n = len(tickers)

    if methode == "gelijk":
        weights = np.ones(n) / n
    elif methode == "marktkap":
        laatste = prijzen.iloc[-1].values
        weights = laatste / laatste.sum()
    elif methode == "handmatig":
        if weights is None:
            raise ValueError("weights moet meegegeven worden voor handmatig")
        weights = np.array(weights, dtype=float)
        if len(weights) != n:
            raise ValueError(f"Aantal weights ({len(weights)}) != aantal tickers ({n})")
        weights = weights / weights.sum()
    else:
        raise ValueError(f"Onbekende methode: {methode}")

    log_returns = np.log(prijzen / prijzen.shift(1)).dropna()
    port_returns = log_returns.dot(weights)

    return {
        "tickers": tickers,
        "weights": weights,
        "log_returns": log_returns,
        "port_returns": port_returns,
    }


def statistieken(port_returns):
    n_dagen = 252
    mu_dag = port_returns.mean()
    sigma_dag = port_returns.std()
    return {
        "mu_dag": mu_dag,
        "sigma_dag": sigma_dag,
        "mu_jaar": mu_dag * n_dagen,
        "sigma_jaar": sigma_dag * np.sqrt(n_dagen),
        "n_observaties": len(port_returns),
    }


def bereken_beta(port_returns, markt_returns):
    df = pd.concat([port_returns, markt_returns], axis=1).dropna()
    df.columns = ["port", "markt"]
    cov = df["port"].cov(df["markt"])
    var = df["markt"].var()
    return cov / var if var > 0 else np.nan


def bereken_alpha(port_returns, markt_returns, rf_dag):
    df = pd.concat([port_returns, markt_returns], axis=1).dropna()
    df.columns = ["port", "markt"]
    beta = bereken_beta(df["port"], df["markt"])
    excess_port = df["port"].mean() - rf_dag
    excess_markt = df["markt"].mean() - rf_dag
    alpha = excess_port - beta * excess_markt
    return alpha, beta


def sharpe_ratio(port_returns, rf_jaar=0.0):
    n = 252
    rf_dag = rf_jaar / n
    excess = port_returns - rf_dag
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(n)


def traynor_ratio(port_returns, markt_returns, rf_jaar=0.0):
    n = 252
    rf_dag = rf_jaar / n
    beta = bereken_beta(port_returns, markt_returns)
    if beta == 0 or np.isnan(beta):
        return np.nan
    excess = (port_returns.mean() - rf_dag) * n
    return excess / beta


def information_ratio(port_returns, benchmark_returns):
    df = pd.concat([port_returns, benchmark_returns], axis=1).dropna()
    df.columns = ["port", "bench"]
    diff = df["port"] - df["bench"]
    te = diff.std()
    if te == 0:
        return np.nan
    return diff.mean() / te * np.sqrt(252)


def var_historisch(port_returns, confidence=0.95):
    percentiel = (1 - confidence) * 100
    return np.percentile(port_returns, percentiel)


def var_parametrisch(port_returns, confidence=0.95, positie=1_000_000):
    mu = port_returns.mean()
    sigma = port_returns.std()
    alpha = norm.ppf(1 - confidence)
    var_pct = mu - sigma * alpha
    var_eur = positie * var_pct
    return var_pct, var_eur


def var_meerdaags(port_returns, confidence=0.95, positie=1_000_000, dagen=10):
    mu = port_returns.mean()
    sigma = port_returns.std()
    alpha = norm.ppf(1 - confidence)
    var_pct = mu * dagen - sigma * alpha * np.sqrt(dagen)
    var_eur = positie * var_pct
    return var_pct, var_eur


def bereken_drawdown(port_returns):
    cum = (1 + port_returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum / running_max) - 1
    max_dd = drawdown.min()
    idx_min = drawdown.idxmin()
    peak_idx = cum.loc[:idx_min].idxmax()
    duur = len(cum.loc[peak_idx:idx_min])

    na_min = cum.loc[idx_min:]
    herstel = na_min[na_min >= cum.loc[peak_idx]]
    hersteltijd = len(herstel) if len(herstel) > 0 else None

    return cum, drawdown, {
        "max_drawdown": max_dd,
        "duur_dagen": duur,
        "hersteltijd_dagen": hersteltijd,
        "peak_datum": str(peak_idx.date()) if hasattr(peak_idx, "date") else str(peak_idx),
        "dal_datum": str(idx_min.date()) if hasattr(idx_min, "date") else str(idx_min),
    }


def maak_rapport(portefeuille, markt_returns, rf_jaar=0.03):
    port_returns = portefeuille["port_returns"]
    n = 252
    rf_dag = rf_jaar / n

    stats = statistieken(port_returns)
    beta = bereken_beta(port_returns, markt_returns)
    alpha, _ = bereken_alpha(port_returns, markt_returns, rf_dag)

    cum, dd_series, dd_stats = bereken_drawdown(port_returns)

    var95_pct = var_historisch(port_returns, 0.95)
    var99_pct = var_historisch(port_returns, 0.99)
    var95_param_pct, var95_param_eur = var_parametrisch(port_returns, 0.95, 1_000_000)
    var99_param_pct, var99_param_eur = var_parametrisch(port_returns, 0.99, 1_000_000)
    var10d_pct, var10d_eur = var_meerdaags(port_returns, 0.99, 1_000_000, 10)

    return {
        "tickers": portefeuille["tickers"],
        "weights": portefeuille["weights"],
        "statistieken": stats,
        "sharpe": sharpe_ratio(port_returns, rf_jaar),
        "traynor": traynor_ratio(port_returns, markt_returns, rf_jaar),
        "information": information_ratio(port_returns, markt_returns),
        "beta": beta,
        "alpha": alpha,
        "var95_pct": var95_pct,
        "var99_pct": var99_pct,
        "var95_param_pct": var95_param_pct,
        "var95_param_eur": var95_param_eur,
        "var99_param_pct": var99_param_pct,
        "var99_param_eur": var99_param_eur,
        "var10d_pct": var10d_pct,
        "var10d_eur": var10d_eur,
        "drawdown": dd_stats,
        "cum_returns": cum,
        "drawdown_series": dd_series,
    }


def plot_portefeuille(portefeuille, markt_returns, rapport, output_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(output_dir, exist_ok=True)

    cum = rapport["cum_returns"]
    dd = rapport["drawdown_series"]
    port_returns = portefeuille["port_returns"]

    # Plot 1: cumulatief
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cum.index, cum.values, color="#d97706", linewidth=2, label="Portefeuille")
    markt_cum = (1 + markt_returns).cumprod()
    ax.plot(markt_cum.index, markt_cum.values, color="#58a6ff",
            linewidth=1.5, alpha=0.7, label="Benchmark (SPY)")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Cumulatief rendement")
    ax.set_title("Cumulatieve performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pad1 = os.path.join(output_dir, "portfolio_cumulatief.png")
    fig.savefig(pad1, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: drawdown
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(dd.index, dd.values * 100, 0, color="#f85149", alpha=0.4)
    ax.plot(dd.index, dd.values * 100, color="#f85149", linewidth=1.5)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(f"Drawdown (max: {rapport['drawdown']['max_drawdown']*100:.2f}%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pad2 = os.path.join(output_dir, "portfolio_drawdown.png")
    fig.savefig(pad2, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: rolling Sharpe
    rolling = port_returns.rolling(30).mean() / port_returns.rolling(30).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(rolling.index, rolling.values, color="#3fb950", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="#3fb950", linestyle=":", alpha=0.5, label="Sharpe = 1")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Rolling Sharpe (30d)")
    ax.set_title("Rolling Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pad3 = os.path.join(output_dir, "portfolio_rolling_sharpe.png")
    fig.savefig(pad3, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return pad1, pad2, pad3


def print_rapport(rapport, rf_jaar):
    print()
    print("=" * 70)
    print("  PORTEFEUILLE-RAPPORT")
    print("=" * 70)
    print()

    print(f"  Tickers: {', '.join(rapport['tickers'])}")
    print(f"  Gewichten:")
    for t, w in zip(rapport["tickers"], rapport["weights"]):
        print(f"    {t:<8}: {w:.2%}")
    print()

    s = rapport["statistieken"]
    print(f"  Rendement (jaarbasis):")
    print(f"    Gemiddeld   : {s['mu_jaar']:.2%}")
    print(f"    Volatiliteit: {s['sigma_jaar']:.2%}")
    print(f"    Observaties : {s['n_observaties']} dagen")
    print()

    print(f"  Performance-ratios (rf = {rf_jaar:.2%}):")
    print(f"    Sharpe Ratio      : {rapport['sharpe']:.4f}")
    print(f"    Traynor Ratio     : {rapport['traynor']:.4f}")
    print(f"    Information Ratio : {rapport['information']:.4f}")
    print()

    print(f"  CAPM:")
    print(f"    Beta  : {rapport['beta']:.4f}")
    print(f"    Alpha : {rapport['alpha']:.4%}")
    if rapport["beta"] > 1:
        print(f"    Interpretatie: Portefeuille is volatieler dan markt")
    elif rapport["beta"] > 0:
        print(f"    Interpretatie: Portefeuille is minder volatiel dan markt")
    else:
        print(f"    Interpretatie: Portefeuille beweegt tegengesteld aan markt")
    print()

    print(f"  Value at Risk (positie = EUR 1.000.000):")
    print(f"    Historisch 95%       : {rapport['var95_pct']:.4%}")
    print(f"    Historisch 99%       : {rapport['var99_pct']:.4%}")
    print(f"    Parametrisch 95%     : {rapport['var95_param_pct']:.4%}  "
          f"(EUR {rapport['var95_param_eur']:,.0f})")
    print(f"    Parametrisch 99%     : {rapport['var99_param_pct']:.4%}  "
          f"(EUR {rapport['var99_param_eur']:,.0f})")
    print(f"    10-daags 99%         : {rapport['var10d_pct']:.4%}  "
          f"(EUR {rapport['var10d_eur']:,.0f})")
    print()

    dd = rapport["drawdown"]
    print(f"  Drawdown:")
    print(f"    Max drawdown    : {dd['max_drawdown']:.2%}")
    print(f"    Duur            : {dd['duur_dagen']} dagen")
    if dd['hersteltijd_dagen']:
        print(f"    Hersteltijd     : {dd['hersteltijd_dagen']} dagen")
    else:
        print(f"    Hersteltijd     : nog niet hersteld")
    print(f"    Peak datum      : {dd['peak_datum']}")
    print(f"    Dal datum       : {dd['dal_datum']}")
    print()