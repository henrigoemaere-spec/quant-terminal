"""test_technical.py — Demo technische analyse"""
import datetime as dt
import technical as ta


def main():
    print("=" * 70)
    print("  TECHNISCHE ANALYSE — Demo")
    print("=" * 70)

    ticker = "AAPL"
    start = "2023-01-01"
    end = dt.datetime.now().strftime("%Y-%m-%d")

    print(f"\n  Data ophalen voor {ticker} ({start} tot {end})...")
    df = ta.haal_ohlc_data(ticker, start, end)
    print(f"  {len(df)} handelsdagen opgehaald")

    print("\n  Indicatoren berekenen...")
    indicatoren = ta.bereken_alle_indicatoren(df)

    # Samenvatting
    ta.print_samenvatting(ticker, df, indicatoren)

    # Plot
    print("  Plot genereren...")
    pad = ta.plot_technisch(ticker, df, indicatoren, periode_dagen=180)
    print(f"  Opgeslagen: {pad}")

    print()
    print("=" * 70)
    print("  KLAAR")
    print("=" * 70)


if __name__ == "__main__":
    main()