"""optieprijzer.py — CLI tool voor opties, portfolio, technisch, macro, FX"""
import argparse
import csv
import time
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes(S, K, T, r, sigma, optie="call", q=0.0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if optie == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if optie == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def binomial(S, K, T, r, sigma, N=500, optie="put", q=0.0, amerikaans=True,
             barrier=None, barrier_type=None):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(N + 1)
    S_T = S * u ** j * d ** (N - j)
    if optie == "call":
        V = np.maximum(S_T - K, 0)
    else:
        V = np.maximum(K - S_T, 0)
    if barrier is not None and barrier_type and "out" in barrier_type:
        V = _knock_out(V, S_T, barrier, barrier_type)
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[1:i + 2] + (1 - p) * V[0:i + 1])
        if amerikaans:
            j = np.arange(i + 1)
            S_i = S * u ** j * d ** (i - j)
            if optie == "call":
                V = np.maximum(V, S_i - K)
            else:
                V = np.maximum(V, K - S_i)
        if barrier is not None and barrier_type and "out" in barrier_type:
            j = np.arange(i + 1)
            S_i = S * u ** j * d ** (i - j)
            V = _knock_out(V, S_i, barrier, barrier_type)
    prijs_out = V[0]
    if barrier is not None and barrier_type and "in" in barrier_type:
        vanilla = binomial(S, K, T, r, sigma, N=N, optie=optie, q=q,
                           amerikaans=amerikaans, barrier=None, barrier_type=None)
        return vanilla - prijs_out
    return prijs_out


def _knock_out(V, S_vals, barrier, barrier_type):
    if "down" in barrier_type:
        mask = S_vals <= barrier
    else:
        mask = S_vals >= barrier
    return np.where(mask, 0.0, V)


def greeks(S, K, T, r, sigma, optie="call", q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    if optie == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (-S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (-S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(marktprijs, S, K, T, r, optie="call", q=0.0):
    def f(sigma):
        return black_scholes(S, K, T, r, sigma, optie, q) - marktprijs
    try:
        return brentq(f, 1e-6, 5.0)
    except ValueError:
        return None


def monte_carlo(S, K, T, r, sigma, optie="call", q=0.0, N=100_000,
                barrier=None, barrier_type=None, seed=None):
    rng = np.random.default_rng(seed)
    if barrier is None:
        Z = rng.standard_normal(N)
        S_T = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        payoff = np.maximum(S_T - K, 0) if optie == "call" else np.maximum(K - S_T, 0)
    else:
        stappen = 252
        dt = T / stappen
        paden = np.zeros((N, stappen + 1))
        paden[:, 0] = S
        for t in range(1, stappen + 1):
            Z = rng.standard_normal(N)
            paden[:, t] = paden[:, t - 1] * np.exp(
                (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
            )
        if "down" in barrier_type:
            geraakt = np.any(paden <= barrier, axis=1)
        else:
            geraakt = np.any(paden >= barrier, axis=1)
        S_T = paden[:, -1]
        payoff = np.maximum(S_T - K, 0) if optie == "call" else np.maximum(K - S_T, 0)
        if "out" in barrier_type:
            payoff = np.where(geraakt, 0.0, payoff)
        else:
            payoff = np.where(geraakt, payoff, 0.0)
    prijs = np.exp(-r * T) * payoff.mean()
    std_err = np.exp(-r * T) * payoff.std(ddof=1) / np.sqrt(N)
    return prijs, std_err


def volatility_smile(csv_path, S, T, r, q=0.0):
    rijen = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            K = float(row["K"])
            prijs = float(row["prijs"])
            optie = row.get("type", "call").strip().lower()
            iv = implied_vol(prijs, S, K, T, r, optie, q)
            rijen.append((K, prijs, optie, iv))
    print(f"\n  Volatility smile — S={S}, T={T}, r={r}")
    print(f"  {'-' * 55}")
    print(f"  {'Strike':>8}  {'Type':>5}  {'Marktprijs':>11}  {'IV':>10}")
    print(f"  {'-' * 55}")
    for K, prijs, optie, iv in rijen:
        iv_str = f"{iv:.2%}" if iv is not None else "n.v.t."
        print(f"  {K:>8.2f}  {optie:>5}  {prijs:>11.4f}  {iv_str:>10}")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Quant CLI: opties + portfolio + technisch + macro + FX")
    sub = p.add_subparsers(dest="commando", required=True)

    pp = sub.add_parser("price", help="Prijs een optie")
    pp.add_argument("--S", type=float, required=True)
    pp.add_argument("--K", type=float, required=True)
    pp.add_argument("--T", type=float, required=True)
    pp.add_argument("--r", type=float, required=True)
    pp.add_argument("--sigma", type=float, required=True)
    pp.add_argument("--q", type=float, default=0.0)
    pp.add_argument("--type", choices=["call", "put"], default="call")
    pp.add_argument("--style", choices=["europees", "amerikaans"], default="europees")
    pp.add_argument("--N", type=int, default=500)
    pp.add_argument("--barrier", type=float, default=None)
    pp.add_argument("--barrier-type", default=None,
                    choices=["down-and-out", "up-and-out", "down-and-in", "up-and-in"])

    pg = sub.add_parser("greeks", help="Bereken Greeks")
    pg.add_argument("--S", type=float, required=True)
    pg.add_argument("--K", type=float, required=True)
    pg.add_argument("--T", type=float, required=True)
    pg.add_argument("--r", type=float, required=True)
    pg.add_argument("--sigma", type=float, required=True)
    pg.add_argument("--q", type=float, default=0.0)
    pg.add_argument("--type", choices=["call", "put"], default="call")

    pi = sub.add_parser("iv", help="Implied volatility")
    pi.add_argument("--prijs", type=float, required=True)
    pi.add_argument("--S", type=float, required=True)
    pi.add_argument("--K", type=float, required=True)
    pi.add_argument("--T", type=float, required=True)
    pi.add_argument("--r", type=float, required=True)
    pi.add_argument("--q", type=float, default=0.0)
    pi.add_argument("--type", choices=["call", "put"], default="call")

    pc = sub.add_parser("compare", help="Vergelijk alle methoden")
    pc.add_argument("--S", type=float, required=True)
    pc.add_argument("--K", type=float, required=True)
    pc.add_argument("--T", type=float, required=True)
    pc.add_argument("--r", type=float, required=True)
    pc.add_argument("--sigma", type=float, required=True)
    pc.add_argument("--q", type=float, default=0.0)
    pc.add_argument("--type", choices=["call", "put"], default="call")
    pc.add_argument("--N", type=int, default=500)

    pm = sub.add_parser("mc", help="Monte Carlo simulatie")
    pm.add_argument("--S", type=float, required=True)
    pm.add_argument("--K", type=float, required=True)
    pm.add_argument("--T", type=float, required=True)
    pm.add_argument("--r", type=float, required=True)
    pm.add_argument("--sigma", type=float, required=True)
    pm.add_argument("--q", type=float, default=0.0)
    pm.add_argument("--type", choices=["call", "put"], default="call")
    pm.add_argument("--N", type=int, default=100_000)
    pm.add_argument("--seed", type=int, default=None)
    pm.add_argument("--barrier", type=float, default=None)
    pm.add_argument("--barrier-type", default=None,
                    choices=["down-and-out", "up-and-out", "down-and-in", "up-and-in"])

    ps = sub.add_parser("smile", help="Volatility smile uit CSV")
    ps.add_argument("--csv", required=True)
    ps.add_argument("--S", type=float, required=True)
    ps.add_argument("--T", type=float, required=True)
    ps.add_argument("--r", type=float, required=True)
    ps.add_argument("--q", type=float, default=0.0)

    psu = sub.add_parser("surface", help="Volatility surface")
    psu.add_argument("--csv", default=None)
    psu.add_argument("--S", type=float, default=100.0)
    psu.add_argument("--r", type=float, default=0.05)
    psu.add_argument("--q", type=float, default=0.02)
    psu.add_argument("--interpoleer", nargs=2, type=float, default=None,
                     metavar=("K", "T"))

    ph = sub.add_parser("heston", help="Heston model")
    ph.add_argument("--S", type=float, default=100.0)
    ph.add_argument("--K", type=float, default=100.0)
    ph.add_argument("--T", type=float, default=1.0)
    ph.add_argument("--r", type=float, default=0.05)
    ph.add_argument("--q", type=float, default=0.02)
    ph.add_argument("--type", choices=["call", "put"], default="call")
    ph.add_argument("--v0", type=float, default=0.04)
    ph.add_argument("--kappa", type=float, default=2.0)
    ph.add_argument("--theta", type=float, default=0.04)
    ph.add_argument("--sigma-v", type=float, default=0.5, dest="sigma_v")
    ph.add_argument("--rho", type=float, default=-0.7)

    phk = sub.add_parser("heston-kalibreer",
                         help="Kalibreer Heston op synthetische marktdata")
    phk.add_argument("--S", type=float, default=100.0)
    phk.add_argument("--r", type=float, default=0.05)
    phk.add_argument("--q", type=float, default=0.02)

    pr = sub.add_parser("risk", help="Risk-analyse op een optie-portfolio")
    pr.add_argument("--S", type=float, default=100.0)
    pr.add_argument("--r", type=float, default=0.05)
    pr.add_argument("--q", type=float, default=0.02)

    psab = sub.add_parser("sabr", help="SABR model (Hagan formule)")
    psab.add_argument("--S", type=float, default=100.0)
    psab.add_argument("--K", type=float, default=100.0)
    psab.add_argument("--T", type=float, default=1.0)
    psab.add_argument("--r", type=float, default=0.05)
    psab.add_argument("--q", type=float, default=0.02)
    psab.add_argument("--alpha", type=float, default=0.2)
    psab.add_argument("--beta", type=float, default=0.5)
    psab.add_argument("--rho", type=float, default=-0.3)
    psab.add_argument("--nu", type=float, default=0.4)

    psabk = sub.add_parser("sabr-kalibreer",
                           help="Kalibreer SABR op realistische marktdata")
    psabk.add_argument("--S", type=float, default=4500.0)
    psabk.add_argument("--r", type=float, default=0.03)
    psabk.add_argument("--q", type=float, default=0.015)
    psabk.add_argument("--beta", type=float, default=0.5)

    pverg = sub.add_parser("vergelijk-modellen",
                           help="Vergelijk SABR vs Heston op dezelfde marktdata")
    pverg.add_argument("--S", type=float, default=4500.0)
    pverg.add_argument("--r", type=float, default=0.03)
    pverg.add_argument("--q", type=float, default=0.015)

    pdup = sub.add_parser("dupire", help="Dupire local volatility uit marktdata")
    pdup.add_argument("--S", type=float, default=4500.0)
    pdup.add_argument("--r", type=float, default=0.03)
    pdup.add_argument("--q", type=float, default=0.015)

    pb = sub.add_parser("bates", help="Bates model (Heston + jumps)")
    pb.add_argument("--S", type=float, default=100.0)
    pb.add_argument("--K", type=float, default=100.0)
    pb.add_argument("--T", type=float, default=1.0)
    pb.add_argument("--r", type=float, default=0.05)
    pb.add_argument("--q", type=float, default=0.02)
    pb.add_argument("--type", choices=["call", "put"], default="call")
    pb.add_argument("--v0", type=float, default=0.04)
    pb.add_argument("--kappa", type=float, default=2.0)
    pb.add_argument("--theta", type=float, default=0.04)
    pb.add_argument("--sigma-v", type=float, default=0.5, dest="sigma_v")
    pb.add_argument("--rho", type=float, default=-0.7)
    pb.add_argument("--lam", type=float, default=0.1)
    pb.add_argument("--mu-j", type=float, default=-0.1, dest="mu_J")
    pb.add_argument("--sigma-j", type=float, default=0.15, dest="sigma_J")

    pbk = sub.add_parser("bates-kalibreer",
                          help="Kalibreer Bates op realistische marktdata")
    pbk.add_argument("--S", type=float, default=4500.0)
    pbk.add_argument("--r", type=float, default=0.03)
    pbk.add_argument("--q", type=float, default=0.015)

    pbench = sub.add_parser("benchmark-fft",
                             help="Vergelijk FFT vs quad voor Heston")
    pbench.add_argument("--S", type=float, default=100.0)
    pbench.add_argument("--K", type=float, default=100.0)
    pbench.add_argument("--T", type=float, default=1.0)
    pbench.add_argument("--r", type=float, default=0.05)
    pbench.add_argument("--q", type=float, default=0.02)

    pmd = sub.add_parser("market-data",
                          help="Haal echte marktdata op via Yahoo Finance")
    pmd.add_argument("--ticker", default="SPY")
    pmd.add_argument("--max-expiries", type=int, default=4, dest="max_expiries")
    pmd.add_argument("--min-volume", type=int, default=10, dest="min_volume")
    pmd.add_argument("--kalibreer", action="store_true",
                      help="Kalibreer direct op de opgehaalde data")

    ppo = sub.add_parser("portfolio", help="Portfolio-analyse: CAPM, ratios, VaR")
    ppo.add_argument("--tickers", required=True,
                     help="Komma-gescheiden tickers, bv AAPL,MSFT,GOOGL")
    ppo.add_argument("--weights", default=None,
                     help="Komma-gescheiden gewichten (optioneel)")
    ppo.add_argument("--start", default="2020-01-01")
    ppo.add_argument("--end", default=None)
    ppo.add_argument("--rf", type=float, default=0.03,
                     help="Risicovrije rente (jaarbasis)")
    ppo.add_argument("--benchmark", default="SPY")
    ppo.add_argument("--positie", type=float, default=1_000_000,
                     help="Portefeuille-waarde voor VaR in euro's")

    pt = sub.add_parser("technical", help="Technische analyse van een aandeel")
    pt.add_argument("--ticker", default="AAPL")
    pt.add_argument("--start", default="2023-01-01")
    pt.add_argument("--end", default=None)
    pt.add_argument("--dagen", type=int, default=180,
                     help="Aantal dagen in de plot")

    pmac = sub.add_parser("macro", help="FRED macro-economische data")
    pmac.add_argument("--start", default="2015-01-01")
    pmac.add_argument("--end", default=None)

    pfx = sub.add_parser("fx", help="Valuta-analyse")
    pfx.add_argument("--pair", default="EURUSD=X")
    pfx.add_argument("--start", default="2023-01-01")
    pfx.add_argument("--end", default=None)
    pfx.add_argument("--dagen", type=int, default=252)

    pfxc = sub.add_parser("fx-vergelijk", help="Vergelijk meerdere valutaparen")
    pfxc.add_argument("--pairs", required=True,
                     help="Komma-gescheiden paren, bv EURUSD=X,EURGBP=X")
    pfxc.add_argument("--start", default="2023-01-01")

    pcarry = sub.add_parser("fx-carry", help="Carry-trade analyse")
    pcarry.add_argument("--start", default="2023-01-01")

    pexp = sub.add_parser("fx-exposure", help="Valuta-exposure in portefeuille")
    pexp.add_argument("--posities", required=True,
                      help="bv: ASML:10000:EUR,AAPL:15000:USD")

    prisico = sub.add_parser("fx-risico", help="Valutarisico in portefeuille")
    prisico.add_argument("--posities", required=True,
                         help="bv: ASML:10000:EUR,AAPL:15000:USD")
    prisico.add_argument("--pairs", default="EURUSD=X,GBPUSD=X,USDJPY=X",
                         help="FX-paren voor signaaldetectie")
    prisico.add_argument("--start", default="2023-01-01")

    return p.parse_args()


def main():
    args = parse_args()

    if args.commando == "price":
        if args.style == "europees" and args.barrier is None:
            prijs = black_scholes(args.S, args.K, args.T, args.r, args.sigma,
                                  args.type, args.q)
            methode = "Black-Scholes (Europees)"
        else:
            prijs = binomial(args.S, args.K, args.T, args.r, args.sigma,
                             N=args.N, optie=args.type, q=args.q,
                             amerikaans=(args.style == "amerikaans"),
                             barrier=args.barrier, barrier_type=args.barrier_type)
            if args.barrier:
                methode = f"Boom + barrière ({args.barrier_type}, N={args.N})"
            else:
                methode = f"Binomiale boom ({args.style}, N={args.N})"
        print(f"\n  {args.type.upper()} -- {methode}")
        print(f"  {'-' * 40}")
        print(f"  Prijs : {prijs:.6f}\n")

    elif args.commando == "greeks":
        g = greeks(args.S, args.K, args.T, args.r, args.sigma, args.type, args.q)
        print(f"\n  Greeks -- {args.type.upper()}")
        print(f"  {'-' * 40}")
        for naam, waarde in g.items():
            print(f"  {naam:<8}: {waarde: .6f}")
        print()

    elif args.commando == "iv":
        iv = implied_vol(args.prijs, args.S, args.K, args.T, args.r, args.type, args.q)
        print(f"\n  Implied volatility")
        print(f"  {'-' * 40}")
        if iv is None:
            print(f"  Geen oplossing gevonden voor prijs {args.prijs}")
        else:
            print(f"  Marktprijs : {args.prijs:.4f}")
            print(f"  IV         : {iv:.4%}")
        print()

    elif args.commando == "compare":
        bs = black_scholes(args.S, args.K, args.T, args.r, args.sigma, args.type, args.q)
        eu = binomial(args.S, args.K, args.T, args.r, args.sigma,
                      N=args.N, optie=args.type, q=args.q, amerikaans=False)
        am = binomial(args.S, args.K, args.T, args.r, args.sigma,
                      N=args.N, optie=args.type, q=args.q, amerikaans=True)
        print(f"\n  Vergelijking -- {args.type.upper()}")
        print(f"  {'-' * 45}")
        print(f"  Black-Scholes (Europees)     : {bs:.6f}")
        print(f"  Boom, Europees (N={args.N:<4})      : {eu:.6f}")
        print(f"  Boom, Amerikaans (N={args.N:<4})    : {am:.6f}")
        print(f"  Vroegtijdig-uitoefen premie  : {am - eu:+.6f}\n")

    elif args.commando == "mc":
        prijs, fout = monte_carlo(
            args.S, args.K, args.T, args.r, args.sigma,
            optie=args.type, q=args.q, N=args.N,
            barrier=args.barrier, barrier_type=args.barrier_type,
            seed=args.seed
        )
        print(f"\n  Monte Carlo -- {args.type.upper()}")
        print(f"  {'-' * 45}")
        print(f"  Paden          : {args.N:,}")
        if args.barrier:
            print(f"  Barrière       : {args.barrier_type} op {args.barrier}")
        print(f"  Prijs          : {prijs:.6f}")
        print(f"  Std. fout      : {fout:.6f}")
        print(f"  95% interval   : [{prijs - 1.96*fout:.6f}, {prijs + 1.96*fout:.6f}]\n")

    elif args.commando == "smile":
        volatility_smile(args.csv, args.S, args.T, args.r, args.q)

    elif args.commando == "surface":
        import pandas as pd
        import surface as surf
        if args.csv is None:
            print("\n  Geen CSV opgegeven — synthetische demo wordt gegenereerd.")
            pad, df = surf.sla_demo_csv_op("marketdata_demo.csv", args.S, args.r, args.q)
            print(f"  Demo opgeslagen in: {pad}")
        else:
            df = pd.read_csv(args.csv)
        surface_df, pivot = surf.bouw_surface(df, optie="call")
        print("\n  Volatility surface (IV in %)")
        print("  " + "-" * 80)
        header = "  T \\ K  " + "".join(f"{k:>8.0f}" for k in pivot.columns)
        print(header)
        for T in pivot.index:
            rij = f"  {T:>6.3f} " + "".join(
                f"{100*pivot.loc[T, K]:>8.2f}" for K in pivot.columns
            )
            print(rij)
        print()
        print("  Arbitrage-checks:")
        bf = surf.check_butterfly(pivot, args.S, args.r, args.q)
        cal = surf.check_calendar(pivot, args.S, args.r, args.q)
        if not bf and not cal:
            print("  OK — geen butterfly- of calendar-arbitrage gevonden.")
        else:
            for T, msg in bf:
                print(f"  FOUT T={T:.3f}: {msg}")
            for K, msg in cal:
                print(f"  FOUT K={K:.0f}: {msg}")
        print()
        if args.interpoleer is not None:
            K_vraag, T_vraag = args.interpoleer
            iv_interp = surf.interpoleer(pivot, K_vraag, T_vraag)
            print(f"  Geïnterpoleerde IV voor K={K_vraag}, T={T_vraag}: {iv_interp:.4%}\n")
        p1, p2, p3 = surf.maak_plots(pivot)
        print(f"  Plots opgeslagen:")
        print(f"    - {p1}")
        print(f"    - {p2}")
        print(f"    - {p3}")
        pad_csv = surf.export_surface(pivot, "surface_export.csv")
        print(f"    - {pad_csv}\n")

    elif args.commando == "heston":
        import heston as h
        prijs = h.heston_price(args.S, args.K, args.T, args.r, args.q,
                               args.v0, args.kappa, args.theta,
                               args.sigma_v, args.rho, args.type)
        iv = h.implied_vol(prijs, args.S, args.K, args.T, args.r, args.type, args.q)
        print(f"\n  Heston -- {args.type.upper()}")
        print(f"  {'-' * 45}")
        print(f"  Parameters:")
        print(f"    v0      = {args.v0:.4f}  (sqrt(v0) = {np.sqrt(args.v0):.2%})")
        print(f"    kappa   = {args.kappa:.4f}")
        print(f"    theta   = {args.theta:.4f}  (sqrt(theta) = {np.sqrt(args.theta):.2%})")
        print(f"    sigma_v = {args.sigma_v:.4f}")
        print(f"    rho     = {args.rho:.4f}")
        print(f"  Prijs  : {prijs:.6f}")
        if iv is not None:
            print(f"  IV     : {iv:.4%}")
        print()

    elif args.commando == "heston-kalibreer":
        import heston as h
        print("\n  Synthetische marktdata genereren...")
        markt = h.genereer_heston_marktdata(args.S, args.r, args.q)
        print(f"  {len(markt)} optieprijzen gegenereerd.\n")
        print("  Kalibreren (dit kan even duren)...")
        params, ss = h.kalibreer_heston(markt, args.S, args.r, args.q)
        v0, kappa, theta, sigma_v, rho = params
        print(f"\n  Gekalibreerde parameters:")
        print(f"  {'-' * 55}")
        print(f"    v0      = {v0:.4f}   (echt: 0.0400)")
        print(f"    kappa   = {kappa:.4f}   (echt: 2.0000)")
        print(f"    theta   = {theta:.4f}   (echt: 0.0400)")
        print(f"    sigma_v = {sigma_v:.4f}   (echt: 0.5000)")
        print(f"    rho     = {rho:.4f}   (echt: -0.7000)")
        print(f"  Residuele som van kwadraten: {ss:.6f}")
        print()
        pad = h.plot_heston_vs_market(markt, args.S, args.r, args.q, params)
        print(f"  Plot opgeslagen: {pad}\n")

    elif args.commando == "risk":
        import risk as rk
        positie = rk.voorbeeld_portfolio()
        S, r, q = args.S, args.r, args.q
        print(f"\n  Risk-analyse portfolio")
        print(f"  {'=' * 60}")
        print(f"  Markt: S={S}, r={r}, q={q}")
        print(f"  Posities:")
        for K, T, aantal, optie, iv in positie:
            print(f"    {aantal:+5d} x {optie:>4} K={K:>6.1f}  T={T:.2f}j  IV={iv:.2%}")
        print()
        totalen, marktwaarde = rk.portfolio_greeks(positie, S, r, q)
        print(f"  Portfolio Greeks:")
        print(f"  {'-' * 60}")
        print(f"    Marktwaarde : {marktwaarde:>12.4f}")
        for naam, waarde in totalen.items():
            print(f"    {naam:<12}: {waarde:>12.4f}")
        print()
        buckets_strikes = np.array([90.0, 100.0, 110.0, 120.0])
        buckets_maturities = np.array([0.25, 0.5, 1.0, 2.0])
        vega_matrix = rk.vega_buckets(
            positie, S, r, q, buckets_strikes, buckets_maturities
        )
        print(f"  Vega bucketing (per strike en maturity):")
        print(f"  {'-' * 60}")
        header = "  T \\ K  " + "".join(f"{k:>10.0f}" for k in buckets_strikes)
        print(header)
        for i, T in enumerate(buckets_maturities):
            rij = f"  {T:>6.2f} " + "".join(
                f"{vega_matrix[i, j]:>10.2f}" for j in range(len(buckets_strikes))
            )
            print(rij)
        print()
        scenarios = [
            ("Underlying +1%", 0.01, 0.0, 0.0),
            ("Underlying -1%", -0.01, 0.0, 0.0),
            ("Vol +1 punt", 0.0, 0.01, 0.0),
            ("Vol -1 punt", 0.0, -0.01, 0.0),
            ("Rente +10bp", 0.0, 0.0, 10.0),
            ("Alles tegelijk", 0.01, 0.01, 10.0),
        ]
        print(f"  P&L attribution per scenario:")
        print(f"  {'-' * 60}")
        print(f"  {'Scenario':<20} {'Delta':>10} {'Gamma':>10} {'Vega':>10} {'Rho':>10} {'Totaal':>10}")
        print(f"  {'-' * 60}")
        for naam, dS_pct, dSigma, dR_bp in scenarios:
            pnl = rk.pnl_attribution(positie, S, r, q, dS_pct, dSigma, dR_bp)
            print(f"  {naam:<20} "
                  f"{pnl['Delta P&L']:>10.2f} "
                  f"{pnl['Gamma P&L']:>10.2f} "
                  f"{pnl['Vega P&L']:>10.2f} "
                  f"{pnl['Rho P&L']:>10.2f} "
                  f"{pnl['Totaal P&L']:>10.2f}")
        print()
        pad = rk.plot_vega_heatmap(vega_matrix, buckets_strikes, buckets_maturities)
        print(f"  Vega heatmap opgeslagen: {pad}\n")

    elif args.commando == "sabr":
        import sabr as sb
        F = sb.forward(args.S, args.r, args.q, args.T)
        iv = sb.sabr_iv(F, args.K, args.T, args.alpha, args.beta,
                        args.rho, args.nu)
        print(f"\n  SABR implied volatility")
        print(f"  {'-' * 45}")
        print(f"  Forward F  : {F:.4f}")
        print(f"  Strike K   : {args.K:.4f}")
        print(f"  Maturity T : {args.T:.4f}")
        print(f"  Parameters: alpha={args.alpha}, beta={args.beta}, "
              f"rho={args.rho}, nu={args.nu}")
        print(f"  IV         : {iv:.4%}\n")

    elif args.commando == "sabr-kalibreer":
        import sabr as sb
        import marktdata as md
        print("\n  Realistische marktdata genereren...")
        pad, df = md.sla_realistische_marktdata_op(None, args.S, args.r, args.q, seed=42)
        print(f"  Opgeslagen in: {pad}")
        print(f"  {len(df)} optieprijzen\n")
        markt_data = [(row["K"], row["T"], row["iv_markt"]) for _, row in df.iterrows()]
        print("  SABR kalibreren per maturity...")
        params = sb.kalibreer_sabr_volledig(markt_data, args.S, args.r, args.q,
                                             beta=args.beta)
        print(f"\n  Gekalibreerde SABR parameters per maturity:")
        print(f"  {'-' * 70}")
        print(f"  {'T':>6}  {'alpha':>8}  {'beta':>6}  {'rho':>8}  {'nu':>8}  {'residueel':>12}")
        print(f"  {'-' * 70}")
        for T, p in params.items():
            print(f"  {T:>6.3f}  {p['alpha']:>8.4f}  {p['beta']:>6.2f}  "
                  f"{p['rho']:>8.4f}  {p['nu']:>8.4f}  {p['residueel']:>12.2e}")
        print()
        pad_fit = sb.plot_sabr_fit(markt_data, args.S, args.r, args.q, params)
        print(f"  SABR-fit plot opgeslagen: {pad_fit}\n")

    elif args.commando == "vergelijk-modellen":
        import sabr as sb
        import heston as h
        import marktdata as md
        print("\n  Realistische marktdata genereren...")
        pad, df = md.sla_realistische_marktdata_op(None, args.S, args.r, args.q, seed=42)
        print(f"  Opgeslagen in: {pad}")
        print(f"  {len(df)} optieprijzen\n")
        markt_iv_data = [(row["K"], row["T"], row["iv_markt"]) for _, row in df.iterrows()]

        print("  SABR kalibreren...")
        params_sabr = sb.kalibreer_sabr_volledig(markt_iv_data, args.S, args.r, args.q, beta=0.5)
        sabr_resid = sum(p["residueel"] for p in params_sabr.values())

        print("  Heston kalibreren (IV-ruimte)...")
        params_heston, heston_resid = h.kalibreer_heston_iv(
            markt_iv_data, args.S, args.r, args.q
        )

        print(f"\n  Model-vergelijking (som van kwadratische IV-fouten):")
        print(f"  {'-' * 60}")
        print(f"  SABR   residueel : {sabr_resid:.6e}")
        print(f"  Heston residueel : {heston_resid:.6e}")
        if heston_resid < sabr_resid:
            print(f"  --> Heston fit beter (factor {heston_resid/sabr_resid:.2f})")
        else:
            print(f"  --> SABR fit beter (factor {sabr_resid/heston_resid:.2f})")
        print()
        print(f"  SABR parameters per maturity:")
        print(f"  {'-' * 60}")
        for T, p in params_sabr.items():
            print(f"  T={T:.3f}:  alpha={p['alpha']:.4f}  "
                  f"rho={p['rho']:+.4f}  nu={p['nu']:.4f}  "
                  f"resid={p['residueel']:.2e}")
        print()
        v0, kappa, theta, sigma_v, rho = params_heston
        print(f"  Heston parameters (globaal, alle T):")
        print(f"  {'-' * 60}")
        print(f"    v0      = {v0:.4f}")
        print(f"    kappa   = {kappa:.4f}")
        print(f"    theta   = {theta:.4f}")
        print(f"    sigma_v = {sigma_v:.4f}")
        print(f"    rho     = {rho:.4f}")
        print()
        pad_heston = h.plot_heston_vs_market(markt_iv_data, args.S, args.r, args.q,
                                              params_heston)
        print(f"  Heston-fit plot opgeslagen: {pad_heston}")
        pad_sabr = sb.plot_sabr_fit(markt_iv_data, args.S, args.r, args.q, params_sabr)
        print(f"  SABR-fit plot opgeslagen: {pad_sabr}\n")

    elif args.commando == "dupire":
        import pandas as pd
        import marktdata as md
        import dupire as dup
        print("\n  Realistische marktdata genereren...")
        pad, df = md.sla_realistische_marktdata_op(None, args.S, args.r, args.q, seed=42)
        print(f"  Opgeslagen in: {pad}")
        print(f"  {len(df)} optieprijzen\n")
        print("  Dupire local-vol surface bouwen...")
        K_grid, T_grid, LV = dup.bouw_dupire_surface(df, args.S, args.r, args.q,
                                                      n_T=8, n_K=8)
        print(f"\n  Local volatility (in %) op grid:")
        print(f"  {'-' * 90}")
        header = "  T \\ K  " + "".join(f"{k:>8.0f}" for k in K_grid[::3])
        print(header)
        for i in range(0, len(T_grid), 2):
            T = T_grid[i]
            rij = f"  {T:>6.3f} " + "".join(
                f"{100*LV[i, j]:>8.2f}" for j in range(0, len(K_grid), 3)
            )
            print(rij)
        print()
        p1, p2 = dup.plot_dupire_surface(K_grid, T_grid, LV)
        print(f"  Dupire plots opgeslagen:")
        print(f"    - {p1}")
        print(f"    - {p2}\n")

    elif args.commando == "bates":
        import bates as bt
        prijs = bt.bates_price(args.S, args.K, args.T, args.r, args.q,
                                args.v0, args.kappa, args.theta,
                                args.sigma_v, args.rho,
                                args.lam, args.mu_J, args.sigma_J,
                                args.type)
        iv = bt.implied_vol(prijs, args.S, args.K, args.T, args.r, args.type, args.q)
        print(f"\n  Bates -- {args.type.upper()}")
        print(f"  {'-' * 50}")
        print(f"  Heston-deel:")
        print(f"    v0      = {args.v0:.4f}  (sqrt(v0) = {np.sqrt(args.v0):.2%})")
        print(f"    kappa   = {args.kappa:.4f}")
        print(f"    theta   = {args.theta:.4f}  (sqrt(theta) = {np.sqrt(args.theta):.2%})")
        print(f"    sigma_v = {args.sigma_v:.4f}")
        print(f"    rho     = {args.rho:.4f}")
        print(f"  Jump-deel (Merton):")
        print(f"    lambda  = {args.lam:.4f}  (jumps per jaar)")
        print(f"    mu_J    = {args.mu_J:.4f}")
        print(f"    sigma_J = {args.sigma_J:.4f}")
        print(f"  Prijs : {prijs:.6f}")
        if iv is not None:
            print(f"  IV    : {iv:.4%}")
        print()

    elif args.commando == "bates-kalibreer":
        import bates_fft as btf
        import marktdata as md
        print("\n  Realistische marktdata genereren...")
        pad, df = md.sla_realistische_marktdata_op(None, args.S, args.r, args.q, seed=42)
        print(f"  Opgeslagen in: {pad}")
        print(f"  {len(df)} optieprijzen\n")
        markt_iv_data = [(row["K"], row["T"], row["iv_markt"]) for _, row in df.iterrows()]
        print("  Bates kalibreren met FFT (multi-start)...")
        params, ss = btf.kalibreer_bates_fft(markt_iv_data, args.S, args.r, args.q,
                                              verbose=True)
        v0, kappa, theta, sigma_v, rho, lam, mu_J, sigma_J = params
        print(f"\n  Gekalibreerde Bates parameters:")
        print(f"  {'-' * 60}")
        print(f"  Heston-deel:")
        print(f"    v0      = {v0:.4f}")
        print(f"    kappa   = {kappa:.4f}")
        print(f"    theta   = {theta:.4f}")
        print(f"    sigma_v = {sigma_v:.4f}")
        print(f"    rho     = {rho:.4f}")
        print(f"  Jump-deel:")
        print(f"    lambda  = {lam:.4f}")
        print(f"    mu_J    = {mu_J:.4f}")
        print(f"    sigma_J = {sigma_J:.4f}")
        print(f"  Residuele SS: {ss:.6e}")
        print()
        import bates as bt
        pad_fit = bt.plot_bates_vs_market(markt_iv_data, args.S, args.r, args.q, params)
        print(f"  Bates-fit plot opgeslagen: {pad_fit}\n")

    elif args.commando == "benchmark-fft":
        import heston_fft as hf
        print("\n  Benchmark: FFT vs quad voor Heston")
        print(f"  {'-' * 55}")
        print(f"  Parameters: S={args.S}, K={args.K}, T={args.T}, r={args.r}, q={args.q}")
        print()
        resultaat = hf.benchmark(args.S, args.K, args.T, args.r, args.q)
        print(f"  Enkele strike:")
        print(f"    quad : {resultaat['quad_1_strike']*1000:.2f} ms")
        print(f"    FFT  : {resultaat['fft_1_strike']*1000:.2f} ms")
        print()
        print(f"  100 strikes tegelijk:")
        print(f"    quad : {resultaat['quad_100_strikes']*1000:.2f} ms")
        print(f"    FFT  : {resultaat['fft_100_strikes']*1000:.2f} ms")
        print()
        print(f"  Speed-up FFT vs quad (100 strikes): {resultaat['fft_vs_quad_100x']:.1f}x")
        print()
        print(f"  Prijs (quad)  : {resultaat['prijs_quad']:.6f}")
        print(f"  Prijs (FFT)   : {resultaat['prijs_fft']:.6f}")
        print(f"  Verschil      : {abs(resultaat['prijs_quad'] - resultaat['prijs_fft']):.6e}")
        print()

    elif args.commando == "market-data":
        import market_data as mkd
        print(f"\n  Marktdata ophalen voor {args.ticker}...")
        try:
            spot, r, q, df, pad = mkd.sla_yahoo_data_op(
                args.ticker, args.max_expiries, args.min_volume
            )
            mkd.toon_overzicht(df)
        except Exception as e:
            print(f"\n  FOUT: {e}")
            print(f"  Controleer of yfinance is geïnstalleerd: pip install yfinance")
            return

        if args.kalibreer:
            print("  Modellen kalibreren op echte marktdata...")
            import sabr as sb
            markt_data = [(row["K"], row["T"], row["iv_markt"])
                          for _, row in df.iterrows()]

            print("\n  SABR kalibreren per maturity...")
            params_sabr = sb.kalibreer_sabr_volledig(markt_data, spot, r, q, beta=0.5)
            sabr_resid = sum(p["residueel"] for p in params_sabr.values())
            print(f"  SABR residueel: {sabr_resid:.6e}")
            print(f"\n  SABR parameters per maturity:")
            for T, p in params_sabr.items():
                print(f"    T={T:.3f}: alpha={p['alpha']:.4f}  "
                      f"rho={p['rho']:+.4f}  nu={p['nu']:.4f}  "
                      f"resid={p['residueel']:.2e}")
            print()

    elif args.commando == "portfolio":
        import portfolio as pf
        import datetime as dt

        tickers = [t.strip() for t in args.tickers.split(",")]
        end = args.end or dt.datetime.now().strftime("%Y-%m-%d")

        print(f"\n  Portfolio-analyse")
        print(f"  {'=' * 70}")
        print(f"  Tickers: {', '.join(tickers)}")
        print(f"  Periode: {args.start} tot {end}")
        print(f"  Benchmark: {args.benchmark}")
        print(f"  Risicovrije rente: {args.rf:.2%}")
        print()

        print("  Data ophalen...")
        try:
            prijzen = pf.haal_aandelen_data(tickers, args.start, end)
            markt = pf.haal_aandelen_data([args.benchmark], args.start, end)
        except Exception as e:
            print(f"  FOUT: {e}")
            return

        print(f"  {len(prijzen)} handelsdagen")

        markt_returns = np.log(markt[args.benchmark] / markt[args.benchmark].shift(1)).dropna()

        weights = None
        methode = "gelijk"
        if args.weights:
            weights = [float(w) for w in args.weights.split(",")]
            methode = "handmatig"

        port = pf.bouw_portefeuille(prijzen, weights=weights, methode=methode)

        rapport = pf.maak_rapport(port, markt_returns, rf_jaar=args.rf)

        from scipy.stats import norm as _norm
        mu = port["port_returns"].mean()
        sigma = port["port_returns"].std()
        alpha95 = _norm.ppf(1 - 0.95)
        alpha99 = _norm.ppf(1 - 0.99)
        rapport["var95_param_eur"] = args.positie * (mu - sigma * alpha95)
        rapport["var99_param_eur"] = args.positie * (mu - sigma * alpha99)
        rapport["var10d_eur"] = args.positie * (mu * 10 - sigma * alpha99 * np.sqrt(10))

        pf.print_rapport(rapport, args.rf)

        print("  Plots genereren...")
        try:
            p1, p2, p3 = pf.plot_portefeuille(port, markt_returns, rapport)
            print(f"    OK: {p1}")
            print(f"    OK: {p2}")
            print(f"    OK: {p3}")
        except Exception as e:
            print(f"    FOUT bij plots: {e}")

        print()
        print("=" * 70)

    elif args.commando == "technical":
        import technical as ta
        import datetime as dt
        end = args.end or dt.datetime.now().strftime("%Y-%m-%d")
        print(f"\n  Technische analyse voor {args.ticker}...")
        try:
            df = ta.haal_ohlc_data(args.ticker, args.start, end)
        except Exception as e:
            print(f"  FOUT: {e}")
            return
        print(f"  {len(df)} handelsdagen opgehaald")
        print("  Indicatoren berekenen...")
        indicatoren = ta.bereken_alle_indicatoren(df)
        ta.print_samenvatting(args.ticker, df, indicatoren)
        print("  Plot genereren...")
        pad = ta.plot_technisch(args.ticker, df, indicatoren,
                                 periode_dagen=args.dagen)
        print(f"  Plot opgeslagen: {pad}")
        print()
        print("=" * 70)

    elif args.commando == "macro":
        import fred
        print(f"\n  FRED macro-economische data ophalen ({args.start} tot nu)...")
        rapport = fred.maak_macro_rapport(start=args.start, end=args.end)
        fred.print_macro_rapport(rapport)
        print("  Plot genereren...")
        try:
            pad = fred.plot_macro(rapport)
            print(f"  Plot opgeslagen: {pad}")
        except Exception as e:
            print(f"  Fout bij plot: {e}")
        print()
        print("=" * 70)

    elif args.commando == "fx":
        import fx as fx_mod
        print(f"\n  Valuta-analyse voor {args.pair}...")
        try:
            rapport = fx_mod.maak_fx_rapport(args.pair, start=args.start,
                                               end=args.end)
        except Exception as e:
            print(f"  FOUT: {e}")
            return
        fx_mod.print_fx_rapport(rapport)
        print("  Plot genereren...")
        pad = fx_mod.plot_fx(rapport, dagen=args.dagen)
        print(f"  Plot opgeslagen: {pad}")
        print()
        print("=" * 70)

    elif args.commando == "fx-vergelijk":
        import fx as fx_mod
        pairs = [p.strip() for p in args.pairs.split(",")]
        print(f"\n  Vergelijking van {len(pairs)} valutaparen...")
        print()
        print(f"  {'Pair':<10} {'Spot':>12} {'Vol %':>10} {'52w min':>12} {'52w max':>12} {'Signaal':<15}")
        print("  " + "-" * 75)
        for p in pairs:
            try:
                rapport = fx_mod.maak_fx_rapport(p, start=args.start)
                s = rapport["stats"]
                sterk = "Geen"
                for _, _, _, sig in rapport["signalen"]:
                    if sig in ["KOOP", "VERKOOP"]:
                        sterk = sig
                        break
                print(f"  {rapport['pair']:<10} {s['spot']:>12.4f} "
                      f"{s['vol_jaar']:>10.2f} {s['min_52w']:>12.4f} "
                      f"{s['max_52w']:>12.4f} {sterk:<15}")
            except Exception as e:
                print(f"  {p}: FOUT — {e}")
        print()
        print("=" * 70)

    elif args.commando == "fx-carry":
        import fx_carry
        print(f"\n  Carry-trade analyse (start={args.start})...")
        rijen = fx_carry.maak_carry_rapport(start=args.start)
        fx_carry.print_carry_rapport(rijen)
        print("=" * 70)

    elif args.commando == "fx-exposure":
        import fx_carry
        posities = []
        for stuk in args.posities.split(","):
            naam, waarde, valuta = stuk.split(":")
            posities.append({
                "naam": naam.strip(),
                "waarde": float(waarde),
                "valuta": valuta.strip().upper(),
            })
        fx_carry.print_exposure_rapport(posities)
        print("=" * 70)

    elif args.commando == "fx-risico":
        import fx_carry
        import fx as fx_mod

        posities = []
        for stuk in args.posities.split(","):
            naam, waarde, valuta = stuk.split(":")
            posities.append({
                "naam": naam.strip(),
                "waarde": float(waarde),
                "valuta": valuta.strip().upper(),
            })

        paren = [p.strip() for p in args.pairs.split(",")]
        print(f"\n  FX-signalen ophalen voor {len(paren)} paren...")
        fx_signalen = {}
        for pair in paren:
            try:
                rapport = fx_mod.maak_fx_rapport(pair, start=args.start)
                sig = "Geen"
                for _, _, _, s in rapport["signalen"]:
                    if s in ["KOOP", "VERKOOP"]:
                        sig = s
                        break
                basis = pair[:3].upper()
                quote = pair[3:6].upper()
                fx_signalen[basis] = sig
                fx_signalen[quote] = "VERKOOP" if sig == "KOOP" else (
                    "KOOP" if sig == "VERKOOP" else "Geen"
                )
                print(f"    {pair}: {sig}")
            except Exception as e:
                print(f"    {pair}: FOUT — {e}")

        fx_carry.print_risico_rapport(posities, fx_signalen)
        print("=" * 70)


if __name__ == "__main__":
    main()