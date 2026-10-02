"""update.py — Ververs de DuckDB database met de laatste marktdata."""
import sys
import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

import database as db


def update_alles(start="2020-01-01", alleen=None):
    """
    Haal alle tickers uit de watchlist op en sla op in DuckDB.
    alleen : optionele lijst van categorieën ['aandelen', 'fx', ...]
    """
    print(f"\n{'='*60}")
    print(f"  Database update gestart: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}\n")

    db.maak_tabellen()

    totaal_opgeslagen = 0
    fouten = []

    for categorie, tickers in db.WATCHLIST.items():
        if alleen and categorie not in alleen:
            continue

        print(f"\n📂 Categorie: {categorie.upper()} ({len(tickers)} tickers)")
        print("-" * 60)

        for i, ticker in enumerate(tickers, 1):
            try:
                print(f"  [{i:>2}/{len(tickers)}] {ticker:<12} ... ",
                      end="", flush=True)
                df = yf.download(ticker, start=start, progress=False,
                                  auto_adjust=False, threads=False)
                if df.empty:
                    print("geen data")
                    continue

                # yfinance geeft multi-index kolommen bij sommige versies
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                aantal = db.sla_prijzen_op(df, ticker, categorie)
                totaal_opgeslagen += aantal
                laatste = df.index[-1].strftime("%Y-%m-%d")
                print(f"{aantal:>5} rijen (t/m {laatste})")

                time.sleep(0.15)  # beetje rust voor yfinance

            except Exception as e:
                print(f"FOUT: {e}")
                fouten.append((ticker, str(e)))

    # Metadata opslaan
    db.sla_meta_op("laatste_update", datetime.now().isoformat())
    db.sla_meta_op("start_datum", start)

    print(f"\n{'='*60}")
    print(f"  Klaar. Totaal opgeslagen: {totaal_opgeslagen} rijen")
    if fouten:
        print(f"  Fouten ({len(fouten)}):")
        for t, m in fouten:
            print(f"    - {t}: {m}")
    print(f"{'='*60}\n")


def update_snel():
    """Alleen data van de laatste 90 dagen verversen (sneller)."""
    start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    print(f"Snelle update sinds {start}")
    update_alles(start=start)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "snel":
        update_snel()
    elif args and args[0] in ["aandelen", "fx", "indices", "etf"]:
        update_alles(alleen=[args[0]])
    else:
        update_alles()