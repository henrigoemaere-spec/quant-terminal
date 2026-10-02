"""test_backtest.py — Demo van backtest op gesimuleerde tijdreeks"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import backtest as bt
import cboe_data as cd


def genereer_gesimuleerd_archief(aantal_dagen=5):
    """Genereer een gesimuleerd archief van dagelijkse snapshots."""
    print("  Echte CBOE data ophalen als basis...")
    spot, df = cd.haal_cboe_data("SPX")

    calls = df[df["type"] == "call"].copy()
    calls = calls[calls["T"] > 0.05]

    if len(calls) < 50:
        print(f"  Te weinig calls ({len(calls)}), kan niet simuleren")
        return

    basis = os.path.dirname(os.path.abspath(__file__))
    archief_dir = os.path.join(basis, "backtest_archief")
    os.makedirs(archief_dir, exist_ok=True)

    vandaag = datetime.now()
    rng = np.random.default_rng(42)

    for i in range(aantal_dagen):
        datum = (vandaag - timedelta(days=i)).strftime("%Y-%m-%d")
        pad = os.path.join(archief_dir, f"SPX_{datum}.csv")

        if os.path.exists(pad):
            print(f"  Bestaat al: {datum}")
            continue

        spot_variatie = spot * (1 + rng.normal(0, 0.005))
        iv_variatie = rng.normal(0, 0.01)

        df_dag = calls.copy()
        df_dag["S"] = spot_variatie
        df_dag["iv_cboe"] = df_dag["iv_cboe"] + iv_variatie
        df_dag["iv_cboe"] = np.clip(df_dag["iv_cboe"], 0.05, 0.60)

        df_dag.to_csv(pad, index=False)
        print(f"  Gesimuleerd: {datum} (spot={spot_variatie:.2f})")

    print(f"\n  Archief opgebouwd in: {archief_dir}")


def main():
    print("=" * 60)
    print("  Backtest — Demo met gesimuleerd archief")
    print("=" * 60)
    print()

    genereer_gesimuleerd_archief(aantal_dagen=5)

    print()
    resultaten = bt.analyseer_archief("SPX", r=0.045, q=0.013, beta=0.5)

    if resultaten is None:
        print("\n  Analyse kon niet worden uitgevoerd.")
        return

    pad = bt.plot_tijdreeks(resultaten, "SPX")
    print(f"\n  Plot opgeslagen: {pad}")

    print(f"\n  Samenvatting over {len(resultaten)} dagen:")
    print(f"  {'-' * 70}")

    alphas = []
    rhos = []
    nus = []
    for datum, r in resultaten.items():
        params = r["parameters"]
        alphas.append(np.mean([p["alpha"] for p in params.values()]))
        rhos.append(np.mean([p["rho"] for p in params.values()]))
        nus.append(np.mean([p["nu"] for p in params.values()]))

    print(f"  alpha: gem={np.mean(alphas):.4f}  std={np.std(alphas):.4f}  "
          f"min={np.min(alphas):.4f}  max={np.max(alphas):.4f}")
    print(f"  rho  : gem={np.mean(rhos):+.4f}  std={np.std(rhos):.4f}  "
          f"min={np.min(rhos):+.4f}  max={np.max(rhos):+.4f}")
    print(f"  nu   : gem={np.mean(nus):.4f}  std={np.std(nus):.4f}  "
          f"min={np.min(nus):.4f}  max={np.max(nus):.4f}")
    print()

    print(f"  Stabiliteit:")
    print(f"  {'-' * 70}")
    alpha_cv = np.std(alphas) / np.mean(alphas)
    rho_cv = np.std(rhos) / abs(np.mean(rhos))
    nu_cv = np.std(nus) / np.mean(nus)

    for naam, cv in [("alpha", alpha_cv), ("rho", rho_cv), ("nu", nu_cv)]:
        status = "stabiel" if cv < 0.10 else ("redelijk" if cv < 0.20 else "instabiel")
        print(f"    {naam}: CV={cv:.4f} -> {status}")


if __name__ == "__main__":
    main()