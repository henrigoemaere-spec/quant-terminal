"""backtest.py — Dagelijkse kalibratie-archief en analyse"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime


def sla_snapshot_op(symbool="SPX"):
    """
    Haalt vandaag CBOE data op en slaat deze op in een archief.
    Bestandsnaam: backtest_archief/SPX_2026-10-02.csv
    """
    import cboe_data as cd

    basis = os.path.dirname(os.path.abspath(__file__))
    archief_dir = os.path.join(basis, "backtest_archief")
    os.makedirs(archief_dir, exist_ok=True)

    vandaag = datetime.now().strftime("%Y-%m-%d")
    pad = os.path.join(archief_dir, f"{symbool}_{vandaag}.csv")

    if os.path.exists(pad):
        print(f"  Snapshot voor {vandaag} bestaat al: {pad}")
        return pad

    print(f"  Snapshot ophalen voor {symbool} op {vandaag}...")
    spot, df, _ = cd.sla_cboe_data_op(symbool)
    df.to_csv(pad, index=False)
    print(f"  Opgeslagen: {pad}")
    return pad


def laad_archief(symbool="SPX"):
    """Laadt alle snapshots voor een symbool."""
    basis = os.path.dirname(os.path.abspath(__file__))
    archief_dir = os.path.join(basis, "backtest_archief")

    if not os.path.exists(archief_dir):
        return {}

    bestanden = sorted([f for f in os.listdir(archief_dir)
                        if f.startswith(f"{symbool}_") and f.endswith(".csv")])

    archief = {}
    for bestand in bestanden:
        datum = bestand.replace(f"{symbool}_", "").replace(".csv", "")
        pad = os.path.join(archief_dir, bestand)
        archief[datum] = pd.read_csv(pad)

    return archief


def kalibreer_dag(df, spot, r, q, beta=0.5, min_T=0.05, max_T=1.0):
    """Kalibreert SABR per maturity op één dag data."""
    import sabr as sb

    # Alleen calls
    calls = df[df["type"] == "call"].copy()

    # Filters
    calls = calls[(calls["T"] >= min_T) & (calls["T"] <= max_T)]
    if len(calls) < 30:
        return None

    F = spot * np.exp((r - q) * calls["T"])
    calls = calls[(calls["K"] > F * 0.85) & (calls["K"] < F * 1.15)]
    calls = calls[(calls["iv_cboe"] > 0.05) & (calls["iv_cboe"] < 0.60)]

    if len(calls) < 20:
        return None

    markt = [(row["K"], row["T"], row["iv_cboe"]) for _, row in calls.iterrows()]

    try:
        params = sb.kalibreer_sabr_volledig(markt, spot, r, q, beta=beta)
    except Exception:
        return None

    resultaat = {"spot": spot, "parameters": {}}
    for T, p in params.items():
        resultaat["parameters"][T] = {
            "alpha": p["alpha"],
            "beta": p["beta"],
            "rho": p["rho"],
            "nu": p["nu"],
            "residueel": p["residueel"],
        }

    return resultaat


def analyseer_archief(symbool="SPX", r=0.045, q=0.013, beta=0.5):
    """Kalibreert SABR op alle snapshots en analyseert tijdreeksen."""
    archief = laad_archief(symbool)

    if len(archief) < 2:
        print(f"\n  Onvoldoende snapshots voor analyse ({len(archief)} dagen).")
        print(f"  Voer dagelijks uit: python -c \"import backtest; backtest.sla_snapshot_op('{symbool}')\"")
        return None

    print(f"\n  Analyse van {len(archief)} dagen {symbool}:")
    print(f"  {'-' * 70}")

    per_datum = {}
    for datum, df in sorted(archief.items()):
        spot = df["S"].iloc[0]
        resultaat = kalibreer_dag(df, spot, r, q, beta=beta)
        if resultaat is None:
            print(f"  {datum}: onvoldoende data")
            continue
        per_datum[datum] = resultaat

        n_mats = len(resultaat["parameters"])
        alpha_avg = np.mean([p["alpha"] for p in resultaat["parameters"].values()])
        rho_avg = np.mean([p["rho"] for p in resultaat["parameters"].values()])
        nu_avg = np.mean([p["nu"] for p in resultaat["parameters"].values()])
        print(f"  {datum}: spot={spot:>7.2f}  maturities={n_mats}  "
              f"alpha={alpha_avg:.4f}  rho={rho_avg:+.4f}  nu={nu_avg:.4f}")

    if len(per_datum) < 2:
        print("\n  Te weinig succesvolle kalibraties voor tijdreeksanalyse.")
        return None

    return per_datum


def plot_tijdreeks(per_datum, symbool="SPX"):
    """Plot alpha, rho, nu over tijd (gemiddeld over maturities)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    basis = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(basis, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    datums = sorted(per_datum.keys())
    alpha_lijst = []
    rho_lijst = []
    nu_lijst = []
    spot_lijst = []

    for datum in datums:
        params = per_datum[datum]["parameters"]
        alpha_lijst.append(np.mean([p["alpha"] for p in params.values()]))
        rho_lijst.append(np.mean([p["rho"] for p in params.values()]))
        nu_lijst.append(np.mean([p["nu"] for p in params.values()]))
        spot_lijst.append(per_datum[datum]["spot"])

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(datums, alpha_lijst, "o-", color="blue")
    axes[0].set_ylabel("alpha (ATM vol)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"SABR parameters over tijd — {symbool}")

    axes[1].plot(datums, rho_lijst, "o-", color="red")
    axes[1].set_ylabel("rho (skew)")
    axes[1].axhline(y=0, color="gray", linestyle="--")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(datums, nu_lijst, "o-", color="green")
    axes[2].set_ylabel("nu (vol-of-vol)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(datums, spot_lijst, "o-", color="black")
    axes[3].set_ylabel("Spot")
    axes[3].set_xlabel("Datum")
    axes[3].grid(True, alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    pad = os.path.join(plots_dir, "backtest_tijdreeks.png")
    fig.savefig(pad, dpi=120)
    plt.close(fig)

    return pad