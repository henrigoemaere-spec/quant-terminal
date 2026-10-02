"""fred.py — Macro-economische data via FRED API"""
import os
import numpy as np
import pandas as pd
import datetime as dt


FRED_CODES = {
    "fed_funds":         "FEDFUNDS",
    "treasury_3m":       "TB3MS",
    "treasury_10y":      "DGS10",
    "treasury_2y":       "DGS2",
    "cpi":               "CPIAUCSL",
    "inflation":         "FPCPITOTLZGUSA",
    "gdp":               "GDPC1",
    "gdp_nominal":       "GDP",
    "unemployment":      "UNRATE",
    "industrial_prod":   "INDPRO",
    "sp500":             "SP500",
    "vix":               "VIXCLS",
    "dollar_index":      "DTWEXBGS",
    "yield_spread":      "T10Y2Y",
    "credit_spread":     "BAA10Y",
}


def haal_fred_data(code, start="2000-01-01", end=None):
    """Haalt een enkele FRED reeks op."""
    try:
        import pandas_datareader as pdr
    except ImportError:
        raise RuntimeError("Installeer pandas-datareader: pip install pandas-datareader")

    if end is None:
        end = dt.datetime.now().strftime("%Y-%m-%d")

    if code in FRED_CODES:
        code = FRED_CODES[code]

    try:
        serie = pdr.DataReader(code, "fred", start, end)
    except Exception as e:
        raise RuntimeError(f"FRED error voor {code}: {e}")

    if isinstance(serie, pd.DataFrame):
        serie = serie.iloc[:, 0]
    serie.name = code
    serie = serie.dropna()
    return serie


def haal_meerdere_fred(aliases, start="2000-01-01", end=None):
    resultaat = {}
    for alias in aliases:
        try:
            serie = haal_fred_data(alias, start, end)
            resultaat[alias] = serie
        except Exception as e:
            print(f"  Fout bij {alias}: {e}")
            continue
    if not resultaat:
        raise ValueError("Geen FRED data opgehaald")
    df = pd.concat(resultaat.values(), axis=1)
    df.columns = list(resultaat.keys())
    return df


# ============================================================
# Risicovrije rente
# ============================================================
def bereken_risicovrije_rente(start="2000-01-01", end=None):
    """
    Berekent de real risk-free rate op basis van 10Y Treasury en CPI inflatie.
    """
    try:
        treasury = haal_fred_data("treasury_10y", start, end)
    except Exception as e:
        raise RuntimeError(f"Treasury data mislukt: {e}")

    try:
        cpi = haal_fred_data("cpi", start, end)
    except Exception as e:
        raise RuntimeError(f"CPI data mislukt: {e}")

    # Jaar-op-jaar inflatie uit CPI
    inflation = cpi.pct_change(12) * 100
    inflation.name = "inflation"

    # Combineer
    df = pd.concat([treasury, inflation], axis=1).dropna()
    df.columns = ["nominal", "inflation"]

    # Interpoleer en drop
    df = df.interpolate(method="linear").dropna()
    df["real"] = df["nominal"] - df["inflation"]

    if len(df) < 1:
        raise RuntimeError("Geen overlap tussen treasury en CPI data")

    return {
        "nominal_huidig": float(df["nominal"].iloc[-1]),
        "inflation_huidig": float(df["inflation"].iloc[-1]),
        "real_huidig": float(df["real"].iloc[-1]),
        "laatste_datum": df.index[-1].strftime("%Y-%m-%d"),
        "df": df,
    }


# ============================================================
# Yield curve
# ============================================================
def bereken_yield_curve(start="2000-01-01", end=None):
    y2 = haal_fred_data("treasury_2y", start, end)
    y10 = haal_fred_data("treasury_10y", start, end)

    df = pd.concat([y2, y10], axis=1).dropna()
    df.columns = ["2y", "10y"]
    df = df.interpolate(method="linear").dropna()
    df["spread"] = df["10y"] - df["2y"]

    if len(df) < 1:
        raise RuntimeError("Geen yield curve data")

    return {
        "2y_huidig": float(df["2y"].iloc[-1]),
        "10y_huidig": float(df["10y"].iloc[-1]),
        "spread_huidig": float(df["spread"].iloc[-1]),
        "inversie": bool(df["spread"].iloc[-1] < 0),
        "laatste_datum": df.index[-1].strftime("%Y-%m-%d"),
        "df": df,
    }


# ============================================================
# Groei
# ============================================================
def bereken_groei(start="2000-01-01", end=None):
    gdp = haal_fred_data("gdp", start, end)
    gdp_change = gdp.pct_change() * 100

    return {
        "gdp_huidig": float(gdp.iloc[-1] / 1000),
        "gdp_groei_huidig": float(gdp_change.iloc[-1]),
        "gdp_groei_jaar": float(gdp_change.iloc[-5:].mean()) if len(gdp_change) >= 5 else float("nan"),
        "laatste_datum": gdp.index[-1].strftime("%Y-%m-%d"),
        "df": gdp_change,
    }


# ============================================================
# Volledig rapport
# ============================================================
def maak_macro_rapport(start="2015-01-01", end=None):
    print("  FRED data ophalen...")
    rapport = {}

    try:
        rapport["risicovrije_rente"] = bereken_risicovrije_rente(start, end)
    except Exception as e:
        print(f"  Fout bij risicovrije rente: {e}")
        rapport["risicovrije_rente"] = None

    try:
        rapport["yield_curve"] = bereken_yield_curve(start, end)
    except Exception as e:
        print(f"  Fout bij yield curve: {e}")
        rapport["yield_curve"] = None

    try:
        rapport["groei"] = bereken_groei(start, end)
    except Exception as e:
        print(f"  Fout bij groei: {e}")
        rapport["groei"] = None

    try:
        un = haal_fred_data("unemployment", start, end)
        rapport["werkloosheid"] = {
            "huidig": float(un.iloc[-1]),
            "jaar_geleden": float(un.iloc[-12]) if len(un) >= 12 else float("nan"),
            "laatste_datum": un.index[-1].strftime("%Y-%m-%d"),
        }
    except Exception as e:
        print(f"  Fout bij werkloosheid: {e}")
        rapport["werkloosheid"] = None

    try:
        vix = haal_fred_data("vix", start, end)
        rapport["vix"] = {
            "huidig": float(vix.iloc[-1]),
            "gemiddelde": float(vix.mean()),
            "max": float(vix.max()),
            "laatste_datum": vix.index[-1].strftime("%Y-%m-%d"),
            "df": vix,
        }
    except Exception as e:
        print(f"  Fout bij VIX: {e}")
        rapport["vix"] = None

    try:
        sp = haal_fred_data("sp500", start, end)
        rapport["sp500"] = {
            "huidig": float(sp.iloc[-1]),
            "jaar_geleden": float(sp.iloc[-252]) if len(sp) >= 252 else float("nan"),
            "laatste_datum": sp.index[-1].strftime("%Y-%m-%d"),
            "df": sp,
        }
    except Exception as e:
        print(f"  Fout bij S&P 500: {e}")
        rapport["sp500"] = None

    return rapport


def print_macro_rapport(rapport):
    print()
    print("=" * 70)
    print("  MACRO-ECONOMISCH RAPPORT")
    print("=" * 70)
    print()

    rf = rapport.get("risicovrije_rente")
    if rf:
        print(f"  Risicovrije rente (laatste: {rf['laatste_datum']}):")
        print(f"    Nominaal (10Y Treasury): {rf['nominal_huidig']:.3f}%")
        print(f"    Inflatie               : {rf['inflation_huidig']:.3f}%")
        print(f"    Real rate              : {rf['real_huidig']:.3f}%")
        print()

    yc = rapport.get("yield_curve")
    if yc:
        print(f"  Yield curve (laatste: {yc['laatste_datum']}):")
        print(f"    2Y Treasury  : {yc['2y_huidig']:.3f}%")
        print(f"    10Y Treasury : {yc['10y_huidig']:.3f}%")
        print(f"    10Y - 2Y     : {yc['spread_huidig']:+.3f}%")
        if yc["inversie"]:
            print(f"    ! INVERSIE - recessie signaal")
        print()

    gr = rapport.get("groei")
    if gr:
        print(f"  Groei (laatste: {gr['laatste_datum']}):")
        print(f"    Real GDP       : ${gr['gdp_huidig']:.2f} biljoen")
        print(f"    Groei (QoQ)    : {gr['gdp_groei_huidig']:+.2f}%")
        if not np.isnan(gr["gdp_groei_jaar"]):
            print(f"    Groei (5Q gem) : {gr['gdp_groei_jaar']:+.2f}%")
        print()

    un = rapport.get("werkloosheid")
    if un:
        print(f"  Werkloosheid (laatste: {un['laatste_datum']}):")
        print(f"    Huidig         : {un['huidig']:.2f}%")
        if not np.isnan(un["jaar_geleden"]):
            print(f"    1 jaar geleden : {un['jaar_geleden']:.2f}%")
        print()

    vix = rapport.get("vix")
    if vix:
        print(f"  VIX (laatste: {vix['laatste_datum']}):")
        print(f"    Huidig      : {vix['huidig']:.2f}")
        print(f"    Gemiddelde  : {vix['gemiddelde']:.2f}")
        print(f"    Max         : {vix['max']:.2f}")
        print()

    sp = rapport.get("sp500")
    if sp:
        print(f"  S&P 500 (laatste: {sp['laatste_datum']}):")
        print(f"    Huidig      : {sp['huidig']:.2f}")
        if not np.isnan(sp["jaar_geleden"]):
            change = (sp["huidig"] - sp["jaar_geleden"]) / sp["jaar_geleden"] * 100
            print(f"    Jaar change : {change:+.2f}%")
        print()


# ============================================================
# Plots
# ============================================================
def plot_macro(rapport, output_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0d1117")

    kleuren = ["#d97706", "#3fb950", "#58a6ff", "#f85149"]

    # Plot 1: rente
    ax1 = axes[0, 0]
    rf = rapport.get("risicovrije_rente")
    if rf:
        df = rf["df"]
        ax1.plot(df.index, df["nominal"], color=kleuren[0], linewidth=1.8,
                 label="Nominaal (10Y)")
        ax1.plot(df.index, df["inflation"], color=kleuren[1], linewidth=1.8,
                 label="Inflatie")
        ax1.plot(df.index, df["real"], color=kleuren[2], linewidth=1.8,
                 label="Real rate")
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax1.legend(loc="best", fontsize=9)
    ax1.set_title("Rente vs Inflatie", color="#e6edf3", fontsize=12)
    ax1.set_ylabel("%", color="#e6edf3")
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#161b22")
    ax1.tick_params(colors="#e6edf3")

    # Plot 2: yield curve
    ax2 = axes[0, 1]
    yc = rapport.get("yield_curve")
    if yc:
        df = yc["df"]
        ax2.plot(df.index, df["2y"], color=kleuren[0], linewidth=1.8, label="2Y")
        ax2.plot(df.index, df["10y"], color=kleuren[1], linewidth=1.8, label="10Y")
        ax2.fill_between(df.index, df["spread"], 0,
                          where=(df["spread"] < 0),
                          color="#f85149", alpha=0.3, label="Inversie")
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax2.legend(loc="best", fontsize=9)
    ax2.set_title("Yield curve (2Y vs 10Y)", color="#e6edf3", fontsize=12)
    ax2.set_ylabel("%", color="#e6edf3")
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#161b22")
    ax2.tick_params(colors="#e6edf3")

    # Plot 3: VIX
    ax3 = axes[1, 0]
    vix = rapport.get("vix")
    if vix:
        df = vix["df"]
        ax3.plot(df.index, df.values, color=kleuren[3], linewidth=1.5,
                 label="VIX")
        ax3.axhline(y=vix["gemiddelde"], color="gray", linestyle="--",
                    alpha=0.5, label=f"Gem: {vix['gemiddelde']:.1f}")
        ax3.fill_between(df.index, df.values, 0, color=kleuren[3], alpha=0.1)
        ax3.legend(loc="best", fontsize=9)
    ax3.set_title("VIX - Volatiliteitsindex", color="#e6edf3", fontsize=12)
    ax3.set_ylabel("VIX", color="#e6edf3")
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor("#161b22")
    ax3.tick_params(colors="#e6edf3")

    # Plot 4: S&P 500
    ax4 = axes[1, 1]
    sp = rapport.get("sp500")
    if sp:
        df = sp["df"]
        ax4.plot(df.index, df.values, color=kleuren[2], linewidth=1.8,
                 label="S&P 500")
        ax4.fill_between(df.index, df.values, 0, color=kleuren[2], alpha=0.1)
        ax4.legend(loc="best", fontsize=9)
    ax4.set_title("S&P 500", color="#e6edf3", fontsize=12)
    ax4.set_ylabel("Index", color="#e6edf3")
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor("#161b22")
    ax4.tick_params(colors="#e6edf3")

    fig.tight_layout()
    pad = os.path.join(output_dir, "macro_rapport.png")
    fig.savefig(pad, dpi=110, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return pad