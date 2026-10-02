"""database.py — SQLite opslag voor marktdata (Fase 1)"""
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


DB_PAD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "market.db")


# ============================================================
# Watchlist — vaste lijst met tickers
# ============================================================
WATCHLIST = {
    "aandelen": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "PG", "MA",
        "HD", "CVX", "ABBV", "KO", "PEP",
    ],
    "fx": [
        "EURUSD=X", "EURGBP=X", "GBPUSD=X", "USDJPY=X",
        "AUDUSD=X", "USDCHF=X", "USDCAD=X", "NZDUSD=X",
        "AUDJPY=X", "NZDJPY=X", "USDMXN=X", "USDTRY=X",
        "USDZAR=X", "EURJPY=X", "GBPJPY=X",
    ],
    "indices": [
        "^GSPC", "^NDX", "^DJI", "^RUT", "^VIX", "^FTSE", "^GDAXI",
    ],
    "etf": [
        "SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "EEM", "FXI",
    ],
}

BENCHMARK = "SPY"


# ============================================================
# Connectie
# ============================================================
def verbind():
    """Open een connectie naar de SQLite database."""
    os.makedirs(os.path.dirname(DB_PAD), exist_ok=True)
    con = sqlite3.connect(DB_PAD)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


# ============================================================
# Tabellen aanmaken
# ============================================================
def maak_tabellen():
    con = verbind()
    con.execute("""
        CREATE TABLE IF NOT EXISTS prijzen (
            ticker      TEXT,
            datum       TEXT,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      INTEGER,
            categorie   TEXT,
            updated_at  TEXT,
            PRIMARY KEY (ticker, datum)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key         TEXT PRIMARY KEY,
            value       TEXT,
            updated_at  TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_prijzen_ticker ON prijzen(ticker)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_prijzen_datum ON prijzen(datum)")
    con.commit()
    con.close()
    print("✓ Tabellen aangemaakt in", DB_PAD)


# ============================================================
# Data opslaan
# ============================================================
def sla_prijzen_op(df, ticker, categorie):
    """Sla OHLCV op voor één ticker."""
    if df is None or df.empty:
        return 0

    df = df.copy()
    df.index.name = "datum"
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]

    if "adj close" in df.columns:
        df = df.drop(columns=["adj close"])

    for kol in ["open", "high", "low", "close", "volume"]:
        if kol not in df.columns:
            df[kol] = np.nan

    df["ticker"] = ticker
    df["categorie"] = categorie
    df["updated_at"] = datetime.now().isoformat()
    df["datum"] = pd.to_datetime(df["datum"]).dt.strftime("%Y-%m-%d")

    rijen = df[["ticker", "datum", "open", "high", "low", "close",
                  "volume", "categorie", "updated_at"]].values.tolist()

    con = verbind()
    con.execute("DELETE FROM prijzen WHERE ticker = ?", [ticker])
    con.executemany("""
        INSERT INTO prijzen
        (ticker, datum, open, high, low, close, volume, categorie, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rijen)
    con.commit()
    aantal = con.execute(
        "SELECT COUNT(*) FROM prijzen WHERE ticker = ?", [ticker]
    ).fetchone()[0]
    con.close()
    return aantal


def sla_meta_op(key, value):
    con = verbind()
    con.execute("""
        INSERT INTO meta (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT (key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
    """, [key, str(value), datetime.now().isoformat()])
    con.commit()
    con.close()


# ============================================================
# Data ophalen
# ============================================================
def haal_prijzen(ticker, start=None, end=None):
    """Haal OHLCV op voor één ticker."""
    con = verbind()
    query = "SELECT datum, open, high, low, close, volume FROM prijzen WHERE ticker = ?"
    params = [ticker]
    if start:
        query += " AND datum >= ?"
        params.append(start)
    if end:
        query += " AND datum <= ?"
        params.append(end)
    query += " ORDER BY datum"
    df = pd.read_sql_query(query, con, params=params)
    con.close()

    if df.empty:
        return df

    df["datum"] = pd.to_datetime(df["datum"])
    df = df.set_index("datum")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


def haal_meerdere_prijzen(tickers, start=None, end=None, kolom="close"):
    """Haal een DataFrame met Close-prijzen voor meerdere tickers."""
    con = verbind()
    plaatsen = ",".join(["?"] * len(tickers))
    query = f"""
        SELECT datum, ticker, {kolom}
        FROM prijzen
        WHERE ticker IN ({plaatsen})
    """
    params = list(tickers)
    if start:
        query += " AND datum >= ?"
        params.append(start)
    if end:
        query += " AND datum <= ?"
        params.append(end)
    query += " ORDER BY datum"
    df = pd.read_sql_query(query, con, params=params)
    con.close()

    if df.empty:
        return df

    df["datum"] = pd.to_datetime(df["datum"])
    pivot = df.pivot(index="datum", columns="ticker", values=kolom)
    pivot.columns.name = None
    return pivot


def haal_laatste_prijzen():
    """Laatste close per ticker + datum."""
    con = verbind()
    df = pd.read_sql_query("""
        SELECT p1.ticker, p1.categorie, p1.datum AS laatste_datum, p1.close AS laatste_close
        FROM prijzen p1
        INNER JOIN (
            SELECT ticker, MAX(datum) AS max_datum
            FROM prijzen
            GROUP BY ticker
        ) p2 ON p1.ticker = p2.ticker AND p1.datum = p2.max_datum
        ORDER BY p1.categorie, p1.ticker
    """, con)
    con.close()
    return df


def haal_meta(key):
    con = verbind()
    result = con.execute(
        "SELECT value FROM meta WHERE key = ?", [key]
    ).fetchone()
    con.close()
    return result[0] if result else None


def statistieken():
    """Overzicht van wat er in de database zit."""
    con = verbind()
    totaal = con.execute("SELECT COUNT(*) FROM prijzen").fetchone()[0]
    per_cat = pd.read_sql_query("""
        SELECT categorie, COUNT(DISTINCT ticker) AS n_tickers,
               COUNT(*) AS n_rijen,
               MIN(datum) AS min_datum, MAX(datum) AS max_datum
        FROM prijzen
        GROUP BY categorie
        ORDER BY categorie
    """, con)
    laatste_update = con.execute(
        "SELECT MAX(updated_at) FROM prijzen"
    ).fetchone()[0]
    con.close()
    return {
        "totaal_rijen": totaal,
        "per_categorie": per_cat,
        "laatste_update": laatste_update,
    }


# ============================================================
# Initialisatie
# ============================================================
if __name__ == "__main__":
    maak_tabellen()
    print("\nDatabase statistieken:")
    stats = statistieken()
    print(f"  Totaal rijen: {stats['totaal_rijen']}")
    print(f"  Laatste update: {stats['laatste_update']}")
    print("\n  Per categorie:")
    print(stats["per_categorie"].to_string(index=False))