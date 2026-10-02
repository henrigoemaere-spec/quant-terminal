"""app_backtest.py — Quant Terminal in Bloomberg style"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

import database as db


st.set_page_config(
    page_title="Quant Terminal",
    page_icon="🟠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Bloomberg style CSS
# ============================================================
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
    }
    html, body, [class*="css"] {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        color: #e6e6e6;
        font-size: 13px;
    }
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #333;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ff6600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
        border-bottom: 1px solid #333;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
    }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] select,
    section[data-testid="stSidebar"] textarea {
        background-color: #111 !important;
        color: #ffcc00 !important;
        border: 1px solid #333 !important;
        border-radius: 0 !important;
        font-family: 'Consolas', monospace !important;
        font-size: 12px !important;
    }
    .bb-topbar {
        background: linear-gradient(90deg, #ff6600 0%, #cc5200 100%);
        padding: 0.5rem 1rem;
        margin: -1rem -1rem 1rem -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #ff6600;
    }
    .bb-topbar .brand {
        color: #000;
        font-weight: 900;
        font-size: 1rem;
        letter-spacing: 3px;
        font-family: 'Consolas', monospace;
    }
    .bb-topbar .clock {
        color: #000;
        font-weight: 700;
        font-size: 0.85rem;
        font-family: 'Consolas', monospace;
    }
    .bb-kpi {
        background: #0a0a0a;
        border: 1px solid #333;
        padding: 0.5rem 0.7rem;
        border-left: 3px solid #ff6600;
        font-family: 'Consolas', monospace;
        height: 100%;
        min-height: 70px;
    }
    .bb-kpi-label {
        color: #888;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.25rem;
    }
    .bb-kpi-value {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .bb-kpi-value.small { font-size: 1.05rem; }
    .bb-kpi-sub {
        color: #666;
        font-size: 0.7rem;
        margin-top: 0.2rem;
    }
    .bb-up { color: #4CAF50; }
    .bb-down { color: #FF3B30; }
    .bb-flat { color: #888; }
    .bb-section {
        color: #ff6600;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 1.2rem 0 0.6rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #333;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #000;
        border-bottom: 1px solid #333;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #888;
        border-radius: 0;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.78rem;
        font-family: 'Consolas', monospace;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #ff6600 !important;
        border-bottom: 2px solid #ff6600 !important;
    }
    .stButton > button {
        background-color: #1a1a1a;
        color: #ff6600;
        border: 1px solid #333;
        border-radius: 0;
        font-weight: 600;
        padding: 0.4rem 0.8rem;
        font-size: 0.78rem;
        font-family: 'Consolas', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        background-color: #ff6600;
        color: #000;
        border-color: #ff6600;
    }
    .stDataFrame {
        border-radius: 0;
        border: 1px solid #333;
        font-family: 'Consolas', monospace;
    }
    .stDataFrame [data-testid="stTable"] {
        font-family: 'Consolas', monospace !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.8rem !important;
        padding: 4px 0 !important;
        color: #ccc !important;
        font-family: 'Consolas', monospace !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #ff6600 !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Consolas', monospace !important;
        font-size: 1.1rem !important;
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Consolas', monospace !important;
        font-size: 0.72rem !important;
        color: #888 !important;
        text-transform: uppercase;
    }
    .stAlert {
        background-color: #0a0a0a !important;
        color: #e6e6e6 !important;
        border: 1px solid #333 !important;
        border-radius: 0 !important;
        font-family: 'Consolas', monospace !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .bb-status {
        display: inline-block;
        padding: 2px 8px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-left: 6px;
    }
    .bb-status.live { background: #4CAF50; color: #000; }
    .bb-status.db { background: #333; color: #ff6600; }
    hr {
        border-color: #222;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Cache functions
# ============================================================
@st.cache_data(ttl=60)
def load_archive_cached(symbol):
    import backtest as bt
    return bt.laad_archief(symbol)


@st.cache_data(ttl=60)
def calibrate_archive_cached(symbol, r, q, beta):
    import backtest as bt
    archief = bt.laad_archief(symbol)
    if len(archief) < 1:
        return None
    per_datum = {}
    for datum, df in sorted(archief.items()):
        spot = df["S"].iloc[0]
        resultaat = bt.kalibreer_dag(df, spot, r, q, beta=beta)
        if resultaat is not None:
            per_datum[datum] = resultaat
    return per_datum


@st.cache_data(ttl=60)
def get_prices_cached(tickers_tuple, start, end, column="close"):
    return db.haal_meerdere_prijzen(list(tickers_tuple), start, end, column)


@st.cache_data(ttl=60)
def get_ohlc_cached(ticker, start, end):
    return db.haal_prijzen(ticker, start, end)


@st.cache_data(ttl=600)
def get_macro_cached(start):
    import fred
    return fred.maak_macro_rapport(start=start)


@st.cache_data(ttl=60)
def make_fx_report_cached(pair, start):
    import fx as fx_mod
    return fx_mod.maak_fx_rapport(pair, start=start)


@st.cache_data(ttl=60)
def make_carry_report_cached(start):
    import fx_carry
    return fx_carry.maak_carry_rapport(start=start)


@st.cache_data(ttl=600)
def build_surface_cached(S, r, q, seed):
    import marktdata as md
    import surface as surf
    _, df = md.sla_realistische_marktdata_op(None, S, r, q, seed=seed)
    _, pivot = surf.bouw_surface(df, optie="call")
    return df, pivot


@st.cache_data(ttl=60)
def get_statistics_cached():
    return db.statistieken()


# ============================================================
# Topbar
# ============================================================
now = datetime.now().strftime("%H:%M:%S  %d-%m-%Y")
stats = get_statistics_cached()
db_rows = stats["totaal_rijen"]

st.markdown(f"""
<div class="bb-topbar">
    <div class="brand">🟠 QUANT TERMINAL</div>
    <div class="clock">
        <span class="bb-status live">● LIVE</span>
        <span class="bb-status db">DB {db_rows:,}</span>
        &nbsp; {now}
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### NAVIGATION")

    page = st.radio(
        "",
        [
            "📊 Backtest (SABR)",
            "💼 Portfolio & Risk",
            "📈 Technical",
            "🌍 Macro",
            "💱 FX Analysis",
            "💸 Carry Trades",
            "🌍 Currency Exposure",
            "⚠️ FX Risk",
            "🌊 Volatility Surface",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
<div style="font-family: 'Consolas', monospace; font-size: 0.7rem; color: #666;">
DB ROWS: <span style="color:#ff6600;">{db_rows:,}</span><br>
LAST UPD: <span style="color:#ff6600;">{str(stats['laatste_update'])[:19] if stats['laatste_update'] else 'N/A'}</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE 1: Backtest (SABR)
# ============================================================
if page == "📊 Backtest (SABR)":

    with st.sidebar:
        st.markdown("### CONFIG")
        symbol = st.selectbox("SYMBOL", ["SPX", "NDX", "RUT", "VIX"])
        r = st.number_input("RATE (r)", value=0.045, step=0.005, format="%.4f")
        q = st.number_input("DIVIDEND (q)", value=0.013, step=0.005, format="%.4f")
        beta = st.slider("SABR β", 0.0, 1.0, 0.5, 0.1)
        st.markdown("### ACTIONS")
        if st.button("📥 NEW SNAPSHOT", use_container_width=True):
            with st.spinner("Fetching CBOE data..."):
                try:
                    import backtest as bt
                    path = bt.sla_snapshot_op(symbol)
                    st.success(f"OK: {os.path.basename(path)}")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"ERROR: {e}")
        if st.button("🔄 RELOAD", use_container_width=True):
            st.cache_data.clear()
            st.info("Cache cleared")

    archive = load_archive_cached(symbol)

    if len(archive) == 0:
        st.warning(f"No snapshots for {symbol}")
        st.info("Click **NEW SNAPSHOT** in the sidebar.")
        st.code("python test_backtest.py", language="bash")
        st.stop()

    with st.spinner("Calibrating SABR..."):
        results = calibrate_archive_cached(symbol, r, q, beta)

    if results is None or len(results) == 0:
        st.error("Calibration failed.")
        st.stop()

    dates = sorted(results.keys())
    alphas, rhos, nus, spots = [], [], [], []
    for date in dates:
        params = results[date]["parameters"]
        alphas.append(np.mean([p["alpha"] for p in params.values()]))
        rhos.append(np.mean([p["rho"] for p in params.values()]))
        nus.append(np.mean([p["nu"] for p in params.values()]))
        spots.append(results[date]["spot"])

    tab1, tab2, tab3, tab4 = st.tabs([
        "OVERVIEW", "TIME SERIES", "HEATMAP", "DATA"
    ])

    with tab1:
        last_spot = spots[-1]
        prev_spot = spots[-2] if len(spots) > 1 else spots[-1]
        spot_change = (last_spot - prev_spot) / prev_spot * 100
        alpha_change = (alphas[-1] - alphas[-2]) * 100 if len(alphas) > 1 else 0
        rho_change = (rhos[-1] - rhos[-2]) if len(rhos) > 1 else 0
        nu_change = (nus[-1] - nus[-2]) if len(nus) > 1 else 0

        def kc(v): return "bb-up" if v > 0 else ("bb-down" if v < 0 else "bb-flat")
        def ap(v): return "▲" if v > 0 else ("▼" if v < 0 else "●")

        st.markdown(f'<div class="bb-section">LAST UPDATE: {max(results.keys())}</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'''
<div class="bb-kpi">
<div class="bb-kpi-label">{symbol} SPOT</div>
<p class="bb-kpi-value">{last_spot:,.2f}</p>
<div class="bb-kpi-sub {kc(spot_change)}">{ap(spot_change)} {spot_change:+.2f}%</div>
</div>''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''
<div class="bb-kpi">
<div class="bb-kpi-label">ATM VOL (α)</div>
<p class="bb-kpi-value">{alphas[-1]:.2%}</p>
<div class="bb-kpi-sub {kc(alpha_change)}">{ap(alpha_change)} {alpha_change:+.3f} pp</div>
</div>''', unsafe_allow_html=True)
        with c3:
            st.markdown(f'''
<div class="bb-kpi">
<div class="bb-kpi-label">SKEW (ρ)</div>
<p class="bb-kpi-value">{rhos[-1]:+.4f}</p>
<div class="bb-kpi-sub {kc(rho_change)}">{ap(rho_change)} {rho_change:+.4f}</div>
</div>''', unsafe_allow_html=True)
        with c4:
            st.markdown(f'''
<div class="bb-kpi">
<div class="bb-kpi-label">VOL-OF-VOL (ν)</div>
<p class="bb-kpi-value">{nus[-1]:.4f}</p>
<div class="bb-kpi-sub {kc(nu_change)}">{ap(nu_change)} {nu_change:+.4f}</div>
</div>''', unsafe_allow_html=True)

    with tab2:
        if len(dates) < 2:
            st.info("Need more than 1 day.")
        else:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                subplot_titles=("ALPHA", "RHO", "NU", "SPOT"),
                vertical_spacing=0.07)
            fig.add_trace(go.Scatter(x=dates, y=alphas, mode="lines+markers",
                name="α", line=dict(color="#ff6600", width=2.5),
                fill="tozeroy", fillcolor="rgba(255, 102, 0, 0.15)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=dates, y=rhos, mode="lines+markers",
                name="ρ", line=dict(color="#4CAF50", width=2.5)), row=2, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="#666", row=2, col=1)
            fig.add_trace(go.Scatter(x=dates, y=nus, mode="lines+markers",
                name="ν", line=dict(color="#FFD700", width=2.5)), row=3, col=1)
            fig.add_trace(go.Scatter(x=dates, y=spots, mode="lines+markers",
                name="Spot", line=dict(color="#58a6ff", width=2.5)), row=4, col=1)
            fig.update_layout(height=900, showlegend=False, hovermode="x unified",
                paper_bgcolor="#000000", plot_bgcolor="#0a0a0a",
                font=dict(color="#e6e6e6", size=11, family="Consolas"))
            fig.update_xaxes(gridcolor="#222", tickangle=45)
            fig.update_yaxes(gridcolor="#222")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        all_T = set()
        for r_day in results.values():
            all_T.update(r_day["parameters"].keys())
        all_T = sorted(all_T)
        if len(all_T) == 0:
            st.info("No maturities.")
        else:
            parameter = st.selectbox("PARAMETER", ["alpha", "rho", "nu", "residueel"])
            matrix = np.full((len(all_T), len(dates)), np.nan)
            for j, date in enumerate(dates):
                params = results[date]["parameters"]
                for i, T in enumerate(all_T):
                    if T in params:
                        matrix[i, j] = params[T][parameter]
            colorscale = "RdYlGn" if parameter == "rho" else "Hot"
            fig = go.Figure(data=go.Heatmap(x=dates,
                y=[f"T={T:.3f}" for T in all_T], z=matrix,
                colorscale=colorscale, xgap=2, ygap=2))
            fig.update_layout(height=max(450, len(all_T) * 35),
                xaxis_title="DATE", yaxis_title="MATURITY",
                paper_bgcolor="#000000", plot_bgcolor="#0a0a0a",
                font=dict(color="#e6e6e6", size=11, family="Consolas"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab4:
        rows = []
        for date in dates:
            for T, p in results[date]["parameters"].items():
                rows.append({"Date": date, "T": f"{T:.4f}",
                    "Spot": f"{results[date]['spot']:.2f}",
                    "α": f"{p['alpha']:.4f}", "β": f"{p['beta']:.2f}",
                    "ρ": f"{p['rho']:+.4f}", "ν": f"{p['nu']:.4f}",
                    "Residual": f"{p['residueel']:.2e}"})
        df_table = pd.DataFrame(rows)
        st.dataframe(df_table, use_container_width=True, hide_index=True, height=500)


# ============================================================
# PAGE 2: Portfolio & Risk
# ============================================================
elif page == "💼 Portfolio & Risk":

    with st.sidebar:
        st.markdown("### PORTFOLIO")
        tickers_input = st.text_input("TICKERS", value="AAPL,MSFT,GOOGL,AMZN")
        weights_input = st.text_input("WEIGHTS", value="")
        benchmark = st.text_input("BENCHMARK", value="SPY")
        start = st.text_input("START", value="2020-01-01")
        rf_year = st.number_input("RF RATE", value=0.03, step=0.005, format="%.4f")
        position = st.number_input("POSITION (EUR)", value=1_000_000.0,
                                    step=100_000.0, format="%.0f")
        st.markdown("### ACTION")
        analyze = st.button("🚀 ANALYZE", use_container_width=True)

    if analyze:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        weights = None
        if weights_input.strip():
            try:
                weights = [float(w) for w in weights_input.split(",")]
                if len(weights) != len(tickers):
                    st.error(f"Number of weights ({len(weights)}) != number of tickers ({len(tickers)})")
                    st.stop()
            except ValueError:
                st.error("Invalid weights")
                st.stop()

        end = datetime.now().strftime("%Y-%m-%d")

        with st.spinner("Loading data from database..."):
            try:
                prices = get_prices_cached(tuple(tickers), start, end)
                market_prices = get_prices_cached((benchmark,), start, end)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        if prices.empty:
            st.error("No data in database for these tickers. Run `python update.py`.")
            st.stop()

        with st.spinner("Analyzing portfolio..."):
            import portfolio as pf
            port = pf.bouw_portefeuille(prices,
                                          weights=weights,
                                          methode="handmatig" if weights else "gelijk")
            market_returns = np.log(market_prices.iloc[:, 0] / market_prices.iloc[:, 0].shift(1)).dropna()
            report = pf.maak_rapport(port, market_returns, rf_jaar=rf_year)

        s = report["statistieken"]
        beta_val = report["beta"]
        alpha_val = report["alpha"]
        sharpe_val = report["sharpe"]

        from scipy.stats import norm
        mu = port["port_returns"].mean()
        sigma = port["port_returns"].std()
        var95_eur = position * (mu - sigma * norm.ppf(0.05))
        var99_eur = position * (mu - sigma * norm.ppf(0.01))
        var10d_eur = position * (mu * 10 - sigma * norm.ppf(0.01) * np.sqrt(10))

        st.markdown(f'<div class="bb-section">PORTFOLIO OVERVIEW — {", ".join(tickers)} · {start} → {end} · BM: {benchmark}</div>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">RETURN (YEAR)</div><p class="bb-kpi-value small">{s["mu_jaar"]:.2%}</p></div>''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">VOLATILITY</div><p class="bb-kpi-value small">{s["sigma_jaar"]:.2%}</p></div>''', unsafe_allow_html=True)
        with c3:
            kc = "bb-up" if sharpe_val > 0.5 else ("bb-flat" if sharpe_val > 0 else "bb-down")
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">SHARPE</div><p class="bb-kpi-value small">{sharpe_val:.3f}</p><div class="bb-kpi-sub {kc}">rf={rf_year:.2%}</div></div>''', unsafe_allow_html=True)
        with c4:
            kc = "bb-up" if beta_val > 1 else "bb-flat"
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">BETA</div><p class="bb-kpi-value small">{beta_val:.3f}</p><div class="bb-kpi-sub {kc}">{"volatile" if beta_val > 1 else "defensive"}</div></div>''', unsafe_allow_html=True)
        with c5:
            kc = "bb-up" if alpha_val > 0 else "bb-down"
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">ALPHA</div><p class="bb-kpi-value small">{alpha_val:.4%}</p><div class="bb-kpi-sub {kc}">{"outperform" if alpha_val > 0 else "underperform"}</div></div>''', unsafe_allow_html=True)

        st.markdown(f'<div class="bb-section">VALUE AT RISK — EUR {position:,.0f}</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">VAR 95% (1D)</div><p class="bb-kpi-value small bb-down">EUR {abs(var95_eur):,.0f}</p></div>''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">VAR 99% (1D)</div><p class="bb-kpi-value small bb-down">EUR {abs(var99_eur):,.0f}</p></div>''', unsafe_allow_html=True)
        with c3:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">VAR 99% (10D)</div><p class="bb-kpi-value small bb-down">EUR {abs(var10d_eur):,.0f}</p></div>''', unsafe_allow_html=True)

        st.markdown(f'<div class="bb-section">RATIOS</div>', unsafe_allow_html=True)
        ratios_df = pd.DataFrame({
            "RATIO": ["SHARPE", "TRAYNOR", "INFORMATION", "BETA", "ALPHA"],
            "VALUE": [f"{report['sharpe']:.4f}", f"{report['traynor']:.4f}",
                        f"{report['information']:.4f}", f"{report['beta']:.4f}",
                        f"{report['alpha']:.4%}"],
        })
        st.dataframe(ratios_df, use_container_width=True, hide_index=True)

        dd = report["drawdown"]
        st.markdown(f'<div class="bb-section">DRAWDOWN</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">MAX DD</div><p class="bb-kpi-value small bb-down">{dd["max_drawdown"]:.2%}</p></div>''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">DURATION</div><p class="bb-kpi-value small">{dd["duur_dagen"]} d</p></div>''', unsafe_allow_html=True)
        with c3:
            recovery = f"{dd['hersteltijd_dagen']} d" if dd['hersteltijd_dagen'] else "N/A"
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">RECOVERY</div><p class="bb-kpi-value small">{recovery}</p></div>''', unsafe_allow_html=True)
        with c4:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">OBSERVATIONS</div><p class="bb-kpi-value small">{s["n_observaties"]} d</p></div>''', unsafe_allow_html=True)

        st.markdown(f'<div class="bb-section">CHARTS</div>', unsafe_allow_html=True)

        cum = report["cum_returns"]
        market_cum = (1 + market_returns).cumprod()
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines",
            name="Portfolio", line=dict(color="#ff6600", width=2),
            fill="tozeroy", fillcolor="rgba(255, 102, 0, 0.1)"))
        fig1.add_trace(go.Scatter(x=market_cum.index, y=market_cum.values,
            mode="lines", name=f"BM ({benchmark})",
            line=dict(color="#58a6ff", width=1.5, dash="dot")))
        fig1.update_layout(height=400, hovermode="x unified",
            paper_bgcolor="#000000", plot_bgcolor="#0a0a0a",
            font=dict(color="#e6e6e6", size=11, family="Consolas"),
            title=dict(text="CUMULATIVE PERFORMANCE", font=dict(color="#ff6600", size=11)))
        fig1.update_xaxes(gridcolor="#222")
        fig1.update_yaxes(gridcolor="#222")
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

        dd_series = report["drawdown_series"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values * 100,
            mode="lines", name="Drawdown",
            line=dict(color="#FF3B30", width=1.5),
            fill="tozeroy", fillcolor="rgba(255, 59, 48, 0.25)"))
        fig2.update_layout(height=300, hovermode="x unified",
            paper_bgcolor="#000000", plot_bgcolor="#0a0a0a",
            font=dict(color="#e6e6e6", size=11, family="Consolas"),
            title=dict(text="DRAWDOWN (%)", font=dict(color="#ff6600", size=11)))
        fig2.update_xaxes(gridcolor="#222")
        fig2.update_yaxes(gridcolor="#222")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    else:
        st.info("👈 Enter tickers and click **ANALYZE**")


# ============================================================
# PAGE 3: Technical
# ============================================================
elif page == "📈 Technical":

    with st.sidebar:
        st.markdown("### TECHNICAL")
        t_ticker = st.text_input("TICKER", value="AAPL", key="t_ticker")
        t_start = st.text_input("START", value="2023-01-01", key="t_start")
        t_days = st.slider("DAYS IN CHART", 60, 500, 180, 20, key="t_days")
        t_analyze = st.button("🔍 ANALYZE", use_container_width=True, key="t_button")

    if t_analyze:
        import technical as ta
        import datetime as dt
        end = dt.datetime.now().strftime("%Y-%m-%d")

        with st.spinner("Loading data..."):
            try:
                df = get_ohlc_cached(t_ticker.upper(), t_start, end)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        if df.empty:
            st.error(f"No data for {t_ticker}.")
            st.stop()

        with st.spinner("Computing indicators..."):
            indicators = ta.bereken_alle_indicatoren(df)
            signals = ta.genereer_signalen(df, indicators)

        close = df["Close"].iloc[-1]
        date = df.index[-1].strftime("%Y-%m-%d")

        st.markdown(f'<div class="bb-section">{t_ticker} — {date}</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">CLOSE</div><p class="bb-kpi-value">{close:.2f}</p></div>''', unsafe_allow_html=True)
        with c2:
            rsi_val = indicators["RSI"].iloc[-1]
            kc = "bb-up" if rsi_val < 30 else ("bb-down" if rsi_val > 70 else "bb-flat")
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">RSI (14D)</div><p class="bb-kpi-value {kc}">{rsi_val:.1f}</p></div>''', unsafe_allow_html=True)
        with c3:
            macd_val = indicators["MACD"].iloc[-1]
            kc = "bb-up" if macd_val > 0 else "bb-down"
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">MACD</div><p class="bb-kpi-value small {kc}">{macd_val:.4f}</p></div>''', unsafe_allow_html=True)

        st.markdown(f'<div class="bb-section">SIGNALS</div>', unsafe_allow_html=True)
        signals_df = pd.DataFrame(signals,
                                     columns=["INDICATOR", "VALUE", "STATUS", "SIGNAL"])
        st.dataframe(signals_df, use_container_width=True, hide_index=True)

        df_plot = df.iloc[-t_days:]
        idx = df_plot.index

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=idx, y=df_plot["Close"], mode="lines",
            name="Close", line=dict(color="#ffffff", width=1.8)))
        fig1.add_trace(go.Scatter(x=idx, y=indicators["SMA50"].iloc[-t_days:],
            mode="lines", name="SMA50", line=dict(color="#ff6600", width=1.3)))
        fig1.add_trace(go.Scatter(x=idx, y=indicators["SMA200"].iloc[-t_days:],
            mode="lines", name="SMA200", line=dict(color="#58a6ff", width=1.3)))
        fig1.add_trace(go.Scatter(x=idx, y=indicators["BB_boven"].iloc[-t_days:],
            mode="lines", name="BB+",
            line=dict(color="#666", width=1, dash="dot")))
        fig1.add_trace(go.Scatter(x=idx, y=indicators["BB_onder"].iloc[-t_days:],
            mode="lines", name="BB-",
            line=dict(color="#666", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(102, 102, 102, 0.08)"))
        fig1.update_layout(height=450, hovermode="x unified",
            paper_bgcolor="#000000", plot_bgcolor="#0a0a0a",
            font=dict(color="#e6e6e6", size=11, family="Consolas"),
            title=dict(text=f"{t_ticker} — PRICE + INDICATORS",
                        font=dict(color="#ff6600", size=11)))
        fig1.update_xaxes(gridcolor="#222")
        fig1.update_yaxes(gridcolor="#222")
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=idx, y=indicators["MACD"].iloc[-t_days:],
            mode="lines", name="MACD", line=dict(color="#ff6600", width=1.8)))
        fig2.add_trace(go.Scatter(x=idx, y=indicators["MACD_signal"].iloc[-t_days:],
            mode="lines", name="Signal", line=dict(color="#58a6ff", width=1.8)))
        fig2.add_trace(go.Bar(x=idx, y=indicators["MACD_hist"].iloc[-t_days:],
            name="Hist",
            marker=dict(color=["#4CAF50" if v >= 0 else "#FF3B30"
                               for v in indicators["MACD_hist"].iloc[-t_days:]])))
        fig2.add_hline(y=0, line_dash="dot", line_color="#666")
        fig2.update_layout(height=300, hovermode="x unified",
            paper_bgcolor="#000000", plot_bgcolor="#0a0a0a",
            font=dict(color="#e6e6e6", size=11, family="Consolas"),
            title=dict(text="MACD", font=dict(color="#ff6600", size=11)))
        fig2.update_xaxes(gridcolor="#222")
        fig2.update_yaxes(gridcolor="#222")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=idx, y=indicators["RSI"].iloc[-t_days:],
            mode="lines", name="RSI", line=dict(color="#ff6600", width=1.8)))
        fig3.add_hline(y=70, line_dash="dot", line_color="#FF3B30")
        fig3.add_hline(y=30, line_dash="dot", line_color="#4CAF50")
        fig3.update_layout(height=250, hovermode="x unified",
            paper_bgcolor="#000000", plot_bgcolor="#0a0a0a",
            font=dict(color="#e6e6e6", size=11, family="Consolas"),
            title=dict(text="RSI", font=dict(color="#ff6600", size=11)),
            yaxis=dict(range=[0, 100]))
        fig3.update_xaxes(gridcolor="#222")
        fig3.update_yaxes(gridcolor="#222")
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    else:
        st.info("👈 Enter ticker and click **ANALYZE**")


# ============================================================
# PAGE 4: Macro
# ============================================================
elif page == "🌍 Macro":

    with st.sidebar:
        st.markdown("### MACRO")
        m_start = st.text_input("START", value="2015-01-01", key="m_start")
        m_analyze = st.button("🔍 LOAD MACRO", use_container_width=True, key="m_button")

    if m_analyze:
        with st.spinner("Fetching FRED data..."):
            try:
                report = get_macro_cached(m_start)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.markdown(f'<div class="bb-section">MACRO INDICATORS</div>', unsafe_allow_html=True)

        rf = report.get("risicovrije_rente")
        yc = report.get("yield_curve")
        un = report.get("werkloosheid")
        vix = report.get("vix")

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            if rf:
                st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">10Y TREASURY</div><p class="bb-kpi-value small">{rf["nominal_huidig"]:.2f}%</p></div>''', unsafe_allow_html=True)
        with c2:
            if rf:
                st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">INFLATION</div><p class="bb-kpi-value small">{rf["inflation_huidig"]:.2f}%</p></div>''', unsafe_allow_html=True)
        with c3:
            if rf:
                kc = "bb-up" if rf["real_huidig"] > 0 else "bb-down"
                st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">REAL RATE</div><p class="bb-kpi-value small {kc}">{rf["real_huidig"]:.2f}%</p></div>''', unsafe_allow_html=True)
        with c4:
            if un:
                st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">UNEMPLOYMENT</div><p class="bb-kpi-value small">{un["huidig"]:.2f}%</p></div>''', unsafe_allow_html=True)
        with c5:
            if vix:
                st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">VIX</div><p class="bb-kpi-value small">{vix["huidig"]:.2f}</p></div>''', unsafe_allow_html=True)

        if yc:
            st.markdown(f'<div class="bb-section">YIELD CURVE</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">2Y</div><p class="bb-kpi-value small">{yc["2y_huidig"]:.3f}%</p></div>''', unsafe_allow_html=True)
            with c2:
                st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">10Y</div><p class="bb-kpi-value small">{yc["10y_huidig"]:.3f}%</p></div>''', unsafe_allow_html=True)
            with c3:
                if yc["inversie"]:
                    st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">10Y - 2Y</div><p class="bb-kpi-value small">{yc["spread_huidig"]:+.3f}%</p><div class="bb-kpi-sub bb-down">INVERTED</div></div>''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">10Y - 2Y</div><p class="bb-kpi-value small">{yc["spread_huidig"]:+.3f}%</p><div class="bb-kpi-sub bb-up">NORMAL</div></div>''', unsafe_allow_html=True)

        if rf:
            df_rf = rf["df"]
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df_rf.index, y=df_rf["nominal"],
                mode="lines", name="10Y", line=dict(color="#ff6600", width=1.8)))
            fig1.add_trace(go.Scatter(x=df_rf.index, y=df_rf["inflation"],
                mode="lines", name="Inflation", line=dict(color="#4CAF50", width=1.8)))
            fig1.add_trace(go.Scatter(x=df_rf.index, y=df_rf["real"],
                mode="lines", name="Real", line=dict(color="#58a6ff", width=1.8, dash="dot")))
            fig1.add_hline(y=0, line_dash="dot", line_color="#666")
            fig1.update_layout(height=350, hovermode="x unified",
                paper_bgcolor="#000000", plot_bgcolor="#0a0a0a",
                font=dict(color="#e6e6e6", size=11, family="Consolas"),
                title=dict(text="RATES VS INFLATION", font=dict(color="#ff6600", size=11)))
            fig1.update_xaxes(gridcolor="#222")
            fig1.update_yaxes(gridcolor="#222")
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

        if vix:
            df_vix = vix["df"]
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df_vix.index, y=df_vix.values,
                mode="lines", name="VIX",
                line=dict(color="#FF3B30", width=1.5),
                fill="tozeroy", fillcolor="rgba(255, 59, 48, 0.15)"))
            fig3.update_layout(height=300, hovermode="x unified",
                paper_bgcolor="#000000", plot_bgcolor="#0a0a0a",
                font=dict(color="#e6e6e6", size=11, family="Consolas"),
                title=dict(text="VIX", font=dict(color="#ff6600", size=11)))
            fig3.update_xaxes(gridcolor="#222")
            fig3.update_yaxes(gridcolor="#222")
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    else:
        st.info("👈 Click **LOAD MACRO**")


# ============================================================
# PAGE 5: FX Analysis
# ============================================================
elif page == "💱 FX Analysis":

    with st.sidebar:
        st.markdown("### FX ANALYSIS")
        fx_pair = st.selectbox(
            "PAIR",
            ["EURUSD=X", "EURGBP=X", "GBPUSD=X", "USDJPY=X",
             "AUDUSD=X", "USDCHF=X", "USDCAD=X", "NZDUSD=X"],
            key="fx_pair"
        )
        fx_start = st.text_input("START", value="2023-01-01", key="fx_start")
        fx_days = st.slider("DAYS", 60, 500, 180, 20, key="fx_days")
        fx_analyze = st.button("🔍 ANALYZE FX", use_container_width=True, key="fx_button")

    if fx_analyze:
        with st.spinner("Analyzing FX..."):
            try:
                report = make_fx_report_cached(fx_pair, fx_start)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        s = report["stats"]

        st.markdown(f'<div class="bb-section">{fx_pair}</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">SPOT</div><p class="bb-kpi-value">{s["spot"]:.4f}</p></div>''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">VOL (YEAR)</div><p class="bb-kpi-value small">{s["vol_jaar"]:.2f}%</p></div>''', unsafe_allow_html=True)
        with c3:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">52W MIN</div><p class="bb-kpi-value small">{s["min_52w"]:.4f}</p></div>''', unsafe_allow_html=True)
        with c4:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">52W MAX</div><p class="bb-kpi-value small">{s["max_52w"]:.4f}</p></div>''', unsafe_allow_html=True)

        st.markdown(f'<div class="bb-section">SIGNALS</div>', unsafe_allow_html=True)
        signals_df = pd.DataFrame(
            report["signalen"],
            columns=["INDICATOR", "VALUE", "STATUS", "SIGNAL"]
        )
        st.dataframe(signals_df, use_container_width=True, hide_index=True)

    else:
        st.info("👈 Choose pair and click **ANALYZE FX**")


# ============================================================
# PAGE 6: Carry Trades
# ============================================================
elif page == "💸 Carry Trades":

    with st.sidebar:
        st.markdown("### CARRY")
        carry_start = st.text_input("START", value="2023-01-01", key="carry_start")
        carry_calc = st.button("🚀 CALCULATE", use_container_width=True, key="carry_button")

    st.markdown(f'<div class="bb-section">CARRY TRADES</div>', unsafe_allow_html=True)

    if carry_calc:
        with st.spinner("Analyzing carry..."):
            try:
                rows = make_carry_report_cached(carry_start)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        if not rows:
            st.warning("No data.")
            st.stop()

        df_carry = pd.DataFrame(rows)
        df_carry = df_carry.sort_values("score", ascending=False).reset_index(drop=True)

        df_show = df_carry[["pair", "funding", "target", "carry",
                              "vol_jaar", "score", "spot"]].copy()
        df_show.columns = ["PAIR", "FUNDING", "TARGET", "CARRY %",
                            "VOL %", "SCORE", "SPOT"]
        st.dataframe(df_show.round(4), use_container_width=True, hide_index=True)

        colors = ["#4CAF50" if s > 0.5 else ("#ff6600" if s > 0.2 else "#FF3B30")
                    for s in df_carry["score"]]
        fig = go.Figure(data=[go.Bar(
            x=df_carry["pair"],
            y=df_carry["score"],
            marker_color=colors,
            text=[f"{s:.3f}" for s in df_carry["score"]],
            textposition="outside",
            textfont=dict(color="#e6e6e6", size=11, family="Consolas"),
        )])
        fig.add_hline(y=0.5, line_dash="dot", line_color="#4CAF50")
        fig.add_hline(y=0.2, line_dash="dot", line_color="#ff6600")
        fig.update_layout(height=400, showlegend=False,
            paper_bgcolor="#000000", plot_bgcolor="#0a0a0a",
            font=dict(color="#e6e6e6", size=11, family="Consolas"),
            title=dict(text="SCORE PER PAIR", font=dict(color="#ff6600", size=11)))
        fig.update_xaxes(gridcolor="#222")
        fig.update_yaxes(gridcolor="#222")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("👈 Click **CALCULATE**")


# ============================================================
# PAGE 7: Currency Exposure
# ============================================================
elif page == "🌍 Currency Exposure":

    with st.sidebar:
        st.markdown("### EXPOSURE")
        exp_input = st.text_area(
            "POSITIONS",
            value="ASML:10000:EUR\nAAPL:15000:USD\nSHELL:8000:GBP\nTM:5000:JPY",
            height=180,
            key="exp_input"
        )
        exp_calc = st.button("🚀 CALCULATE", use_container_width=True, key="exp_button")

    st.markdown(f'<div class="bb-section">CURRENCY EXPOSURE</div>', unsafe_allow_html=True)

    if exp_calc:
        import fx_carry

        positions = []
        for line in exp_input.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                name, value, currency = line.split(":")
                positions.append({
                    "naam": name.strip(),
                    "waarde": float(value),
                    "valuta": currency.strip().upper(),
                })
            except ValueError:
                st.error(f"Invalid line: {line}")
                st.stop()

        exposure, total = fx_carry.bereken_exposure(positions)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">TOTAL VALUE</div><p class="bb-kpi-value small">{total:,.2f}</p></div>''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">CURRENCIES</div><p class="bb-kpi-value small">{len(exposure)}</p></div>''', unsafe_allow_html=True)

        df_exp = pd.DataFrame([
            {"CURRENCY": v, "AMOUNT": total * pct / 100, "PERCENTAGE": pct}
            for v, pct in sorted(exposure.items(), key=lambda x: -x[1])
        ])
        st.dataframe(df_exp.round(2), use_container_width=True, hide_index=True)

        fig = go.Figure(data=[go.Pie(
            labels=list(exposure.keys()),
            values=list(exposure.values()),
            hole=0.4,
            marker=dict(colors=["#ff6600", "#4CAF50", "#58a6ff", "#FFD700",
                                 "#FF3B30", "#a371f7", "#39d353"])
        )])
        fig.update_layout(height=400,
            paper_bgcolor="#000000", plot_bgcolor="#0a0a0a",
            font=dict(color="#e6e6e6", size=11, family="Consolas"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        largest = max(exposure.values())
        if largest > 50:
            st.warning(f"⚠ Concentration risk: {largest:.1f}%")
        else:
            st.success(f"✓ Well diversified (max {largest:.1f}%)")

    else:
        st.info("👈 Click **CALCULATE**")


# ============================================================
# PAGE 8: FX Risk
# ============================================================
elif page == "⚠️ FX Risk":

    with st.sidebar:
        st.markdown("### RISK")
        risk_input = st.text_area(
            "POSITIONS",
            value="ASML:10000:EUR\nAAPL:15000:USD\nSHELL:8000:GBP",
            height=150,
            key="risk_input"
        )
        risk_pairs = st.text_input("PAIRS", value="EURUSD=X,GBPUSD=X,USDJPY=X", key="risk_pairs")
        risk_start = st.text_input("START", value="2023-01-01", key="risk_start")
        risk_calc = st.button("🚀 CALCULATE", use_container_width=True, key="risk_button")

    st.markdown(f'<div class="bb-section">FX RISK</div>', unsafe_allow_html=True)

    if risk_calc:
        import fx_carry

        positions = []
        for line in risk_input.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                name, value, currency = line.split(":")
                positions.append({
                    "naam": name.strip(),
                    "waarde": float(value),
                    "valuta": currency.strip().upper(),
                })
            except ValueError:
                st.error(f"Invalid line: {line}")
                st.stop()

        pairs = [p.strip() for p in risk_pairs.split(",") if p.strip()]

        with st.spinner("Fetching FX signals..."):
            fx_signals = {}
            for pair in pairs:
                try:
                    report = make_fx_report_cached(pair, risk_start)
                    sig = "None"
                    for _, _, _, s in report["signalen"]:
                        if s in ["KOOP", "VERKOOP"]:
                            sig = "BUY" if s == "KOOP" else "SELL"
                            break
                    base = pair[:3].upper()
                    quote = pair[3:6].upper()
                    fx_signals[base] = sig
                    fx_signals[quote] = ("SELL" if sig == "BUY"
                                          else ("BUY" if sig == "SELL" else "None"))
                except Exception as e:
                    st.warning(f"{pair}: {e}")

        exposure, total = fx_carry.bereken_exposure(positions)

        sig_df = pd.DataFrame([
            {"PAIR": p, "SIGNAL": fx_signals.get(p[:3].upper(), "None")}
            for p in pairs
        ])
        st.dataframe(sig_df, use_container_width=True, hide_index=True)

        rows = []
        total_score = 0.0
        for v, pct in sorted(exposure.items(), key=lambda x: -x[1]):
            sig = fx_signals.get(v, "None")
            signaal_score = {"BUY": 1.0, "None": 0.5, "SELL": -1.0}.get(sig, 0.0)
            score = pct / 100 * signaal_score
            total_score += score
            rows.append({
                "CURRENCY": v,
                "EXPOSURE %": f"{pct:.2f}%",
                "FX SIGNAL": sig,
                "RISK SCORE": round(score, 3),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        kc = ("bb-up" if total_score > 0.3
              else ("bb-down" if total_score < -0.3 else "bb-flat"))
        st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">TOTAL SCORE</div><p class="bb-kpi-value small {kc}">{total_score:+.3f}</p><div class="bb-kpi-sub {kc}">{"POSITIVE" if total_score > 0.3 else ("NEGATIVE" if total_score < -0.3 else "NEUTRAL")}</div></div>''', unsafe_allow_html=True)

        advice = []
        for v, pct in sorted(exposure.items(), key=lambda x: -x[1]):
            sig = fx_signals.get(v, "None")
            if sig == "SELL" and pct > 20:
                hedge_pct = min(pct * 0.6, 70)
                advice.append(f"**{v}** — hedge ~{hedge_pct:.0f}% of {pct:.1f}% exposure")
            elif sig == "BUY" and pct > 30:
                advice.append(f"**{v}** — no hedge needed")
            elif pct > 40:
                advice.append(f"**{v}** — consider diversifying ({pct:.1f}%)")

        if advice:
            st.markdown(f'<div class="bb-section">HEDGE ADVICE</div>', unsafe_allow_html=True)
            for a in advice:
                st.markdown(f"- {a}")
    else:
        st.info("👈 Click **CALCULATE**")


# ============================================================
# PAGE 9: Volatility Surface
# ============================================================
elif page == "🌊 Volatility Surface":

    with st.sidebar:
        st.markdown("### SURFACE")
        sf_S = st.number_input("SPOT S", value=4500.0, step=50.0, key="sf_S")
        sf_r = st.number_input("RATE r", value=0.03, step=0.005, format="%.4f", key="sf_r")
        sf_q = st.number_input("DIVIDEND q", value=0.015, step=0.005, format="%.4f", key="sf_q")
        sf_seed = st.number_input("SEED", value=42, step=1, key="sf_seed")
        sf_build = st.button("🚀 BUILD", use_container_width=True, key="sf_button")

    st.markdown(f'<div class="bb-section">VOLATILITY SURFACE</div>', unsafe_allow_html=True)

    if sf_build:
        with st.spinner("Building surface..."):
            try:
                df_surf, pivot = build_surface_cached(sf_S, sf_r, sf_q, sf_seed)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        iv_min = float(pivot.values.min())
        iv_max = float(pivot.values.max())

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">IV MIN</div><p class="bb-kpi-value small">{iv_min:.2%}</p></div>''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''<div class="bb-kpi"><div class="bb-kpi-label">IV MAX</div><p class="bb-kpi-value small">{iv_max:.2%}</p></div>''', unsafe_allow_html=True)

        strikes = pivot.columns.values
        maturities = pivot.index.values
        X, Y = np.meshgrid(strikes, maturities)
        Z = pivot.values * 100

        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Hot")])
        fig.update_layout(
            height=600,
            scene=dict(
                xaxis_title="STRIKE",
                yaxis_title="MATURITY",
                zaxis_title="IV (%)",
                bgcolor="#000000",
                xaxis=dict(gridcolor="#333", color="#e6e6e6"),
                yaxis=dict(gridcolor="#333", color="#e6e6e6"),
                zaxis=dict(gridcolor="#333", color="#e6e6e6"),
            ),
            paper_bgcolor="#000000",
            font=dict(color="#e6e6e6", size=11, family="Consolas"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        fig2 = go.Figure()
        colors_list = ["#ff6600", "#4CAF50", "#58a6ff", "#FFD700",
                        "#FF3B30", "#a371f7"]
        for i, T in enumerate(maturities):
            color = colors_list[i % len(colors_list)]
            fig2.add_trace(go.Scatter(
                x=strikes, y=pivot.loc[T].values * 100,
                mode="lines+markers",
                name=f"T={T:.2f}y",
                line=dict(color=color, width=2),
            ))
        fig2.update_layout(
            height=400, hovermode="x unified",
            paper_bgcolor="#000000", plot_bgcolor="#0a0a0a",
            font=dict(color="#e6e6e6", size=11, family="Consolas"),
            xaxis_title="STRIKE", yaxis_title="IV (%)",
            title=dict(text="SMILE PER MATURITY", font=dict(color="#ff6600", size=11)),
        )
        fig2.update_xaxes(gridcolor="#222")
        fig2.update_yaxes(gridcolor="#222")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("👈 Click **BUILD**")