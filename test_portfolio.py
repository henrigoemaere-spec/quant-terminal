"""test_portfolio.py — Demo portfolio-analyse"""
import datetime as dt
import portfolio as pf


def main():
    print("=" * 70)
    print("  PORTEFEUILLE-ANALYSE — Demo")
    print("=" * 70)

    # Portefeuille: 4 grote tech-aandelen
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    start = "2020-01-01"
    end = dt.datetime.now().strftime("%Y-%m-%d")
    rf_jaar = 0.03

    print(f"\n  Ophalen data voor: {', '.join(tickers)}")
    print(f"  Periode: {start} tot {end}")

    # Data ophalen
    prijzen = pf.haal_aandelen_data(tickers, start, end)
    print(f"  {len(prijzen)} handelsdagen opgehaald")

    # Markt (benchmark)
    markt = pf.haal_aandelen_data(["SPY"], start, end)
    markt_returns = markt["SPY"].pct_change().dropna()
    markt_returns = markt_returns.apply(lambda x: __import__("numpy").log(1 + x))

    # Portefeuille bouwen (gelijke gewichten)
    port = pf.bouw_portefeuille(prijzen, methode="gelijk")
    print(f"\n  Portefeuille gewichten:")
    for t, w in zip(port["tickers"], port["weights"]):
        print(f"    {t}: {w:.2%}")

    # Rapport maken
    rapport = pf.maak_rapport(port, markt_returns, rf_jaar=rf_jaar)
    pf.print_rapport(rapport, rf_jaar)

    # Plots maken
    print("  Plots genereren...")
    p1, p2, p3 = pf.plot_portefeuille(port, markt_returns, rapport)
    print(f"    OK: {p1}")
    print(f"    OK: {p2}")
    print(f"    OK: {p3}")

    print()
    print("=" * 70)
    print("  KLAAR")
    print("=" * 70)


if __name__ == "__main__":
    main()