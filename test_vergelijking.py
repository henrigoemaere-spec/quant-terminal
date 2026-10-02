"""test_vergelijking.py — Vergelijk SABR, Heston en Bates op CBOE SPX"""
import os
import traceback
import numpy as np
import model_vergelijking as mv


def main():
    print("=" * 70)
    print("  MODEL-VERGELIJKING OP CBOE SPX")
    print("=" * 70)
    print()

    print("  CBOE data laden...")
    try:
        S_spot, markt = mv.laad_cboe_data("cboe_SPX.csv")
    except FileNotFoundError:
        print("  FOUT: cboe_SPX.csv niet gevonden")
        return

    # BELANGRIJK: gebruik unieke namen, niet r en q, om conflicten te voorkomen
    rente = 0.045
    dividend = 0.013
    print(f"  Spot {S_spot:.2f}, {len(markt)} opties, r={rente}, q={dividend}")
    print()

    print("  [1/3] SABR kalibreren...")
    sabr_res = mv.kalibreer_sabr(markt, S_spot, rente, dividend)
    print(f"        Residueel: {sabr_res['residual']:.6e}")
    print(f"        Tijd: {sabr_res['tijd']:.2f}s ({sabr_res['n_params']} params)")
    print()

    print("  [2/3] Heston kalibreren...")
    heston_res = mv.kalibreer_heston(markt, S_spot, rente, dividend)
    print(f"        Residueel: {heston_res['residual']:.6e}")
    print(f"        Tijd: {heston_res['tijd']:.2f}s ({heston_res['n_params']} params)")
    print()

    print("  [3/3] Bates kalibreren...")
    bates_res = mv.kalibreer_bates(markt, S_spot, rente, dividend)
    print(f"        Residueel: {bates_res['residual']:.6e}")
    print(f"        Tijd: {bates_res['tijd']:.2f}s ({bates_res['n_params']} params)")
    print()

    resultaten = {
        "SABR": sabr_res,
        "Heston": heston_res,
        "Bates": bates_res,
    }

    # Samenvatting — gebruik 'res' als loop-variabele, NIET 'r'
    kandidaten = [(naam, res["residual"]) for naam, res in resultaten.items()
                  if isinstance(res["residual"], (int, float, np.floating))
                  and res["residual"] < 1e9]
    beste_res = min(kandidaten, key=lambda x: x[1])[0] if kandidaten else "n.v.t."

    print("=" * 70)
    print("  SAMENVATTING")
    print("=" * 70)
    print()
    print(f"  {'Model':<10} {'Params':>8} {'Residueel':>16} {'Tijd (s)':>12} {'Winnaar':>10}")
    print("  " + "-" * 65)
    for naam in ["SABR", "Heston", "Bates"]:
        res = resultaten[naam]
        if res["residual"] >= 1e9:
            print(f"  {naam:<10} {'-':>8} {'MISLUKT':>16} {'-':>12} {'-':>10}")
        else:
            ster = "*" if naam == beste_res else ""
            print(f"  {naam:<10} {res['n_params']:>8} {res['residual']:>16.6e} "
                  f"{res['tijd']:>12.2f} {ster:>10}")
    print()
    print(f"  Beste fit: {beste_res}")
    print()

    # Parameters
    print("=" * 70)
    print("  PARAMETERS")
    print("=" * 70)
    print()

    if sabr_res.get("params"):
        print("  SABR (per maturity):")
        print(f"  {'T':>8} {'alpha':>10} {'rho':>10} {'nu':>10} {'resid':>12}")
        print("  " + "-" * 55)
        for T, p in sorted(sabr_res["params"].items()):
            print(f"  {T:>8.3f} {p['alpha']:>10.4f} {p['rho']:>+10.4f} "
                  f"{p['nu']:>10.4f} {p['residueel']:>12.2e}")
        print()

    if heston_res["params"] is not None:
        params_h = heston_res["params"]
        print("  Heston (globaal, alle maturities):")
        print(f"    v0      = {float(params_h[0]):.6f}  "
              f"(sqrt = {np.sqrt(float(params_h[0])):.2%})")
        print(f"    kappa   = {float(params_h[1]):.4f}")
        print(f"    theta   = {float(params_h[2]):.6f}  "
              f"(sqrt = {np.sqrt(float(params_h[2])):.2%})")
        print(f"    sigma_v = {float(params_h[3]):.4f}")
        print(f"    rho     = {float(params_h[4]):+.4f}")
        print()

    if bates_res["params"] is not None:
        params_b = bates_res["params"]
        print("  Bates (globaal, Heston + jumps):")
        print(f"    v0      = {float(params_b[0]):.6f}")
        print(f"    kappa   = {float(params_b[1]):.4f}")
        print(f"    theta   = {float(params_b[2]):.6f}")
        print(f"    sigma_v = {float(params_b[3]):.4f}")
        print(f"    rho     = {float(params_b[4]):+.4f}")
        print(f"    lambda  = {float(params_b[5]):.6f}  (jumps/jaar)")
        print(f"    mu_J    = {float(params_b[6]):+.4f}")
        print(f"    sigma_J = {float(params_b[7]):.4f}")
        print()

    # Interpretatie
    print("=" * 70)
    print("  INTERPRETATIE")
    print("=" * 70)
    print()
    print(f"  Beste fit: {beste_res}")
    print()

    if beste_res == "SABR":
        print("  SABR wint omdat het per-maturity kalibreert.")
        print("  Voordeel: perfecte fit van elke smile apart.")
        print("  Nadeel: parameters zijn niet consistent over T.")
        print("  Gebruik voor: vanilla opties, smile-kalibratie.")
    elif beste_res == "Heston":
        print("  Heston wint met 5 parameters voor alle T.")
        print("  Voordeel: dynamisch consistent.")
        print("  Gebruik voor: exotische opties, dynamisch hedgen.")
    elif beste_res == "Bates":
        print("  Bates wint met Heston + jumps.")
        print("  Voordeel: vangt crash-risico.")
        print("  Gebruik voor: equity index opties (VS standaard).")
    print()

    # Plots — gebruik rente en dividend, niet r en q
    print("  Plots genereren...")
    try:
        pad1 = mv.plot_vergelijking(markt, S_spot, rente, dividend, resultaten)
        print(f"    OK: {pad1}")
    except Exception as e:
        print(f"    FOUT vergelijkingsplot: {e}")
        print("    VOLLEDIGE TRACEBACK:")
        traceback.print_exc()

    try:
        pad2 = mv.plot_residualen(resultaten)
        print(f"    OK: {pad2}")
    except Exception as e:
        print(f"    FOUT residualenplot: {e}")
        traceback.print_exc()

    print()
    print("=" * 70)
    print("  KLAAR")
    print("=" * 70)


if __name__ == "__main__":
    main()