"""app.py — Streamlit webinterface voor de optieprijzer"""
import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq

# Onze eigen modules
import optieprijzer as op
import heston as h
import sabr as sb
import surface as surf
import marktdata as md
import risk as rk
import dupire as dup


# ============================================================
# Pagina-configuratie
# ============================================================
st.set_page_config(
    page_title="Optieprijzer",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# Zijbalk — navigatie
# ============================================================
st.sidebar.title("📈 Optieprijzer")
st.sidebar.markdown("Professionele toolkit voor optieprijzing en risk")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Kies een module:",
    [
        "🏠 Home",
        "💰 Prijs een optie",
        "📐 Greeks",
        "🔍 Implied volatility",
        "🌊 Volatility surface",
        "🎯 SABR",
        "⚙️ Heston",
        "📊 Dupire Local Vol",
        "🛡️ Risk-analyse",
        "🔬 Modellen vergelijken",
        "💱 FX Analyse",
        "💸 Carry Trades",
        "🌍 Valuta-Exposure",
        "⚠️ Valutarisico",
    ]
)


# ============================================================
# Home
# ============================================================
if pagina == "🏠 Home":
    st.title("📈 Optieprijzer")
    st.markdown("""
    **Een professionele toolkit voor het prijzen van opties en risk-analyse.**

    ### Beschikbare modules

    | Module | Wat het doet |
    |--------|--------------|
    | **Prijs een optie** | Black-Scholes, binomiale boom, Amerikaans, barrier |
    | **Greeks** | Delta, gamma, vega, theta, rho |
    | **Implied volatility** | Bereken IV uit marktprijs |
    | **Volatility surface** | IV-surface met arbitrage-checks |
    | **SABR** | SABR model + kalibratie per maturity |
    | **Heston** | Stochastische volatiliteit + kalibratie |
    | **Dupire** | Local volatility surface |
    | **Risk-analyse** | Vega bucketing + P&L attribution |
    | **Modellen vergelijken** | SABR vs Heston vs Dupire |
    | **FX Analyse** | EUR/USD, EUR/GBP — RSI, MACD, SMA, Bollinger |
    | **Carry Trades** | Risk-adjusted carry scores voor alle paren |
    | **Valuta-Exposure** | Blootstelling per valuta in je portefeuille |
    | **Valutarisico** | Combineert exposure + FX-signalen + hedge-advies |

    ### Gebruik
    Kies links een module en vul de parameters in. Alles werkt real-time.
    """)

    st.info("💡 Tip: begin met **Prijs een optie** om het systeem te verkennen.")


# ============================================================
# Prijs een optie
# ============================================================
elif pagina == "💰 Prijs een optie":
    st.title("💰 Prijs een optie")

    col1, col2, col3 = st.columns(3)

    with col1:
        S = st.number_input("Spotprijs S", value=100.0, step=1.0)
        K = st.number_input("Strike K", value=100.0, step=1.0)
        T = st.number_input("Maturity T (jaar)", value=1.0, step=0.05, min_value=0.01)

    with col2:
        r = st.number_input("Rente r", value=0.05, step=0.001, format="%.4f")
        q = st.number_input("Dividend q", value=0.02, step=0.001, format="%.4f")
        sigma = st.number_input("Volatiliteit σ", value=0.20, step=0.01, format="%.4f")

    with col3:
        optie_type = st.selectbox("Type", ["call", "put"])
        style = st.selectbox("Stijl", ["europees", "amerikaans"])
        N_boom = st.slider("Boom N", 50, 2000, 500, 50)

    st.markdown("---")

    bs_prijs = op.black_scholes(S, K, T, r, sigma, optie_type, q)
    boom_prijs = op.binomial(S, K, T, r, sigma, N=N_boom, optie=optie_type,
                              q=q, amerikaans=(style == "amerikaans"))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Black-Scholes (Europees)", f"{bs_prijs:.4f}")
    col_b.metric(f"Boom ({style}, N={N_boom})", f"{boom_prijs:.4f}")
    col_c.metric("Verschil", f"{boom_prijs - bs_prijs:+.4f}")

    st.subheader("Optieprijs vs spotprijs")
    S_range = np.linspace(S * 0.5, S * 1.5, 80)
    prijzen = [op.black_scholes(s, K, T, r, sigma, optie_type, q) for s in S_range]
    payoff = [max(s - K, 0) if optie_type == "call" else max(K - s, 0) for s in S_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=prijzen, mode="lines",
                              name="Optieprijs", line=dict(color="blue", width=2)))
    fig.add_trace(go.Scatter(x=S_range, y=payoff, mode="lines",
                              name="Payoff bij expiratie",
                              line=dict(color="gray", dash="dash")))
    fig.add_vline(x=S, line_dash="dot", line_color="red", annotation_text="Spot")
    fig.add_vline(x=K, line_dash="dot", line_color="orange", annotation_text="Strike")
    fig.update_layout(xaxis_title="Spot S", yaxis_title="Prijs",
                       height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Greeks
# ============================================================
elif pagina == "📐 Greeks":
    st.title("📐 Greeks")

    col1, col2, col3 = st.columns(3)
    with col1:
        S = st.number_input("S", value=100.0, step=1.0, key="g_S")
        K = st.number_input("K", value=100.0, step=1.0, key="g_K")
    with col2:
        T = st.number_input("T (jaar)", value=1.0, step=0.05, key="g_T")
        r = st.number_input("r", value=0.05, step=0.001, format="%.4f", key="g_r")
    with col3:
        q = st.number_input("q", value=0.02, step=0.001, format="%.4f", key="g_q")
        sigma = st.number_input("σ", value=0.20, step=0.01, format="%.4f", key="g_sig")

    optie_type = st.radio("Type", ["call", "put"], horizontal=True, key="g_type")

    g = op.greeks(S, K, T, r, sigma, optie_type, q)

    st.markdown("---")
    cols = st.columns(5)
    cols[0].metric("Delta", f"{g['delta']:.4f}")
    cols[1].metric("Gamma", f"{g['gamma']:.4f}")
    cols[2].metric("Vega", f"{g['vega']:.4f}")
    cols[3].metric("Theta", f"{g['theta']:.4f}")
    cols[4].metric("Rho", f"{g['rho']:.4f}")

    st.subheader("Greeks als functie van S")
    S_range = np.linspace(S * 0.5, S * 1.5, 80)
    greeks_over_S = {"delta": [], "gamma": [], "vega": [], "theta": []}
    for s in S_range:
        gg = op.greeks(s, K, T, r, sigma, optie_type, q)
        for key in greeks_over_S:
            greeks_over_S[key].append(gg[key])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=greeks_over_S["delta"],
                              name="Delta", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=S_range, y=greeks_over_S["gamma"],
                              name="Gamma", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=S_range, y=greeks_over_S["vega"] / 100,
                              name="Vega / 100", line=dict(color="green")))
    fig.update_layout(xaxis_title="Spot S", yaxis_title="Waarde",
                       height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Implied volatility
# ============================================================
elif pagina == "🔍 Implied volatility":
    st.title("🔍 Implied volatility uit marktprijs")

    col1, col2, col3 = st.columns(3)
    with col1:
        S = st.number_input("S", value=100.0, step=1.0, key="iv_S")
        K = st.number_input("K", value=105.0, step=1.0, key="iv_K")
    with col2:
        T = st.number_input("T", value=0.5, step=0.05, key="iv_T")
        r = st.number_input("r", value=0.05, step=0.001, format="%.4f", key="iv_r")
    with col3:
        q = st.number_input("q", value=0.02, step=0.001, format="%.4f", key="iv_q")
        marktprijs = st.number_input("Marktprijs", value=5.50, step=0.10, key="iv_prijs")

    optie_type = st.radio("Type", ["call", "put"], horizontal=True, key="iv_type")

    iv = op.implied_vol(marktprijs, S, K, T, r, optie_type, q)

    st.markdown("---")
    if iv is None:
        st.error("Geen oplossing gevonden. Marktprijs is te hoog of te laag.")
    else:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Marktprijs", f"{marktprijs:.4f}")
        col_b.metric("Implied volatility", f"{iv:.2%}")
        bs_prijs = op.black_scholes(S, K, T, r, iv, optie_type, q)
        col_c.metric("BS-prijs met IV", f"{bs_prijs:.4f}",
                      delta=f"{bs_prijs - marktprijs:+.6f}")


# ============================================================
# Volatility surface
# ============================================================
elif pagina == "🌊 Volatility surface":
    st.title("🌊 Volatility surface")
    st.markdown("Realistische synthetische S&P-achtige optie-surface.")

    S = st.number_input("Spot S", value=4500.0, step=50.0, key="sf_S")
    r = st.number_input("Rente r", value=0.03, step=0.005, format="%.4f", key="sf_r")
    q = st.number_input("Dividend q", value=0.015, step=0.005, format="%.4f", key="sf_q")

    if st.button("Bereken surface"):
        with st.spinner("Surface bouwen..."):
            _, df = md.sla_realistische_marktdata_op(None, S, r, q, seed=42)
            _, pivot = surf.bouw_surface(df, optie="call")

        st.subheader("IV-matrix (in %)")
        styled = (pivot * 100).round(2)
        st.dataframe(styled, use_container_width=True)

        bf = surf.check_butterfly(pivot, S, r, q)
        cal = surf.check_calendar(pivot, S, r, q)
        if not bf and not cal:
            st.success("✓ Geen butterfly- of calendar-arbitrage gevonden.")
        else:
            for T, msg in bf:
                st.warning(f"T={T:.3f}: {msg}")
            for K, msg in cal:
                st.warning(f"K={K:.0f}: {msg}")

        st.subheader("3D surface")
        strikes = pivot.columns.values
        maturities = pivot.index.values
        X, Y = np.meshgrid(strikes, maturities)
        Z = pivot.values * 100

        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
        fig.update_layout(scene=dict(xaxis_title="Strike",
                                       yaxis_title="Maturity",
                                       zaxis_title="IV (%)"),
                           height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Smile per maturity")
        fig2 = go.Figure()
        for T in maturities:
            fig2.add_trace(go.Scatter(x=strikes, y=pivot.loc[T].values * 100,
                                        mode="lines+markers", name=f"T={T:.2f}"))
        fig2.update_layout(xaxis_title="Strike", yaxis_title="IV (%)",
                            height=400, hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# SABR
# ============================================================
elif pagina == "🎯 SABR":
    st.title("🎯 SABR model")

    tab1, tab2 = st.tabs(["Prijs een IV", "Kalibreer op marktdata"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            S = st.number_input("S", value=100.0, step=1.0, key="sabr_S")
            K = st.number_input("K", value=100.0, step=1.0, key="sabr_K")
            T = st.number_input("T", value=1.0, step=0.05, key="sabr_T")
        with col2:
            r = st.number_input("r", value=0.05, step=0.001, format="%.4f", key="sabr_r")
            q = st.number_input("q", value=0.02, step=0.001, format="%.4f", key="sabr_q")
            alpha = st.slider("α (ATM vol)", 0.01, 1.0, 0.20, 0.01)
        with col3:
            beta = st.slider("β (CEV)", 0.0, 1.0, 0.5, 0.1)
            rho = st.slider("ρ (correlatie)", -0.99, 0.99, -0.30, 0.05)
            nu = st.slider("ν (vol-of-vol)", 0.01, 1.5, 0.40, 0.05)

        F = sb.forward(S, r, q, T)
        iv = sb.sabr_iv(F, K, T, alpha, beta, rho, nu)
        st.metric("SABR Implied Volatility", f"{iv:.4%}")

        K_range = np.linspace(S * 0.7, S * 1.3, 50)
        ivs = [sb.sabr_iv(F, k, T, alpha, beta, rho, nu) for k in K_range]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=K_range, y=np.array(ivs) * 100,
                                  mode="lines", line=dict(color="blue", width=2)))
        fig.add_vline(x=K, line_dash="dot", line_color="red")
        fig.update_layout(xaxis_title="Strike", yaxis_title="IV (%)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("Kalibratie op realistische S&P-achtige data.")
        S2 = st.number_input("Spot S", value=4500.0, step=50.0, key="sabr_cal_S")
        r2 = st.number_input("Rente r", value=0.03, step=0.005, format="%.4f", key="sabr_cal_r")
        q2 = st.number_input("Dividend q", value=0.015, step=0.005, format="%.4f", key="sabr_cal_q")

        if st.button("Kalibreer SABR"):
            with st.spinner("Kalibreren..."):
                _, df = md.sla_realistische_marktdata_op(None, S2, r2, q2, seed=42)
                markt_data = [(row["K"], row["T"], row["iv_markt"])
                              for _, row in df.iterrows()]
                params = sb.kalibreer_sabr_volledig(markt_data, S2, r2, q2, beta=0.5)

            rows = []
            for T, p in params.items():
                rows.append({
                    "T": f"{T:.3f}",
                    "α": f"{p['alpha']:.4f}",
                    "β": f"{p['beta']:.2f}",
                    "ρ": f"{p['rho']:+.4f}",
                    "ν": f"{p['nu']:.4f}",
                    "Residueel": f"{p['residueel']:.2e}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.subheader("SABR-fit vs markt")
            per_T = {}
            for K, T, iv in markt_data:
                per_T.setdefault(T, []).append((K, iv))

            fig = go.Figure()
            for T, punten in sorted(per_T.items()):
                punten_sorted = sorted(punten, key=lambda x: x[0])
                strikes = [k for k, _ in punten_sorted]
                markt_ivs = [iv for _, iv in punten_sorted]
                F = sb.forward(S2, r2, q2, T)
                p = params[T]
                model_ivs = [sb.sabr_iv(F, K, T, p["alpha"], p["beta"],
                                          p["rho"], p["nu"])
                              for K in strikes]
                fig.add_trace(go.Scatter(x=strikes, y=np.array(markt_ivs) * 100,
                                          mode="markers", name=f"Markt T={T:.2f}",
                                          marker=dict(size=8, opacity=0.6)))
                fig.add_trace(go.Scatter(x=strikes, y=np.array(model_ivs) * 100,
                                          mode="lines", name=f"SABR T={T:.2f}",
                                          line=dict(width=2)))
            fig.update_layout(xaxis_title="Strike", yaxis_title="IV (%)",
                                height=500, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Heston
# ============================================================
elif pagina == "⚙️ Heston":
    st.title("⚙️ Heston stochastische-volatiliteitsmodel")

    tab1, tab2 = st.tabs(["Prijs een optie", "Kalibreer"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            S = st.number_input("S", value=100.0, step=1.0, key="h_S")
            K = st.number_input("K", value=100.0, step=1.0, key="h_K")
            T = st.number_input("T", value=1.0, step=0.05, key="h_T")
        with col2:
            r = st.number_input("r", value=0.05, step=0.001, format="%.4f", key="h_r")
            q = st.number_input("q", value=0.02, step=0.001, format="%.4f", key="h_q")
            v0 = st.slider("v0 (initiële variantie)", 0.001, 0.5, 0.04, 0.005)
        with col3:
            kappa = st.slider("κ (mean reversion)", 0.1, 5.0, 2.0, 0.1)
            theta = st.slider("θ (lange-termijn var)", 0.001, 0.5, 0.04, 0.005)
            sigma_v = st.slider("σ_v (vol-of-vol)", 0.01, 1.5, 0.5, 0.05)
            rho = st.slider("ρ (correlatie)", -0.99, 0.0, -0.7, 0.05)

        prijs = h.heston_price(S, K, T, r, q, v0, kappa, theta, sigma_v, rho, "call")
        iv = h.implied_vol(prijs, S, K, T, r, "call", q)

        col_a, col_b = st.columns(2)
        col_a.metric("Heston call prijs", f"{prijs:.4f}")
        col_b.metric("Implied volatility", f"{iv:.4%}" if iv else "n.v.t.")

        st.caption(f"√v0 = {np.sqrt(v0):.2%}  |  √θ = {np.sqrt(theta):.2%}")

    with tab2:
        st.markdown("Kalibratie op synthetische marktdata (uit Heston zelf).")
        if st.button("Kalibreer Heston"):
            with st.spinner("Kalibreren (duurt 1-2 minuten)..."):
                markt = h.genereer_heston_marktdata(100.0, 0.05, 0.02)
                params, ss = h.kalibreer_heston(markt, 100.0, 0.05, 0.02)

            v0, kappa, theta, sigma_v, rho = params
            data = [
                {"Parameter": "v0", "Gekalibreerd": f"{v0:.4f}", "Echt": "0.0400"},
                {"Parameter": "κ", "Gekalibreerd": f"{kappa:.4f}", "Echt": "2.0000"},
                {"Parameter": "θ", "Gekalibreerd": f"{theta:.4f}", "Echt": "0.0400"},
                {"Parameter": "σ_v", "Gekalibreerd": f"{sigma_v:.4f}", "Echt": "0.5000"},
                {"Parameter": "ρ", "Gekalibreerd": f"{rho:.4f}", "Echt": "-0.7000"},
            ]
            st.dataframe(pd.DataFrame(data), use_container_width=True)
            st.metric("Residuele SS", f"{ss:.6f}")


# ============================================================
# Dupire
# ============================================================
elif pagina == "📊 Dupire Local Vol":
    st.title("📊 Dupire Local Volatility")
    st.markdown("Bouw een local-volatility surface uit marktdata.")

    S = st.number_input("Spot S", value=4500.0, step=50.0, key="dup_S")
    r = st.number_input("Rente r", value=0.03, step=0.005, format="%.4f", key="dup_r")
    q = st.number_input("Dividend q", value=0.015, step=0.005, format="%.4f", key="dup_q")

    if st.button("Bouw Dupire surface"):
        with st.spinner("Bouwen..."):
            _, df = md.sla_realistische_marktdata_op(None, S, r, q, seed=42)
            K_grid, T_grid, LV = dup.bouw_dupire_surface(df, S, r, q, n_T=10, n_K=10)

        st.subheader("Local volatility (in %)")
        tabel = pd.DataFrame((LV * 100).round(2),
                              index=[f"T={t:.3f}" for t in T_grid],
                              columns=[f"K={k:.0f}" for k in K_grid])
        st.dataframe(tabel, use_container_width=True)

        X, Y = np.meshgrid(K_grid, T_grid)
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=LV * 100, colorscale="Viridis")])
        fig.update_layout(scene=dict(xaxis_title="Strike",
                                       yaxis_title="Maturity",
                                       zaxis_title="Local vol (%)"),
                           height=600)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure(data=go.Heatmap(x=K_grid, y=T_grid, z=LV * 100,
                                           colorscale="Viridis"))
        fig2.update_layout(xaxis_title="Strike", yaxis_title="Maturity", height=400)
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# Risk-analyse
# ============================================================
elif pagina == "🛡️ Risk-analyse":
    st.title("🛡️ Risk-analyse portfolio")

    st.markdown("### Portfolio")
    positie_default = pd.DataFrame({
        "K": [100.0, 110.0, 90.0, 105.0],
        "T": [1.0, 1.0, 0.5, 2.0],
        "Aantal": [100, -200, 50, -100],
        "Type": ["call", "call", "put", "call"],
        "IV": [0.20, 0.18, 0.24, 0.21],
    })
    edited_df = st.data_editor(positie_default, num_rows="dynamic",
                                 use_container_width=True)

    S = st.number_input("Spot S", value=100.0, step=1.0, key="rk_S")
    r = st.number_input("Rente r", value=0.05, step=0.001, format="%.4f", key="rk_r")
    q = st.number_input("Dividend q", value=0.02, step=0.001, format="%.4f", key="rk_q")

    if st.button("Bereken risk"):
        positie = [
            (row["K"], row["T"], int(row["Aantal"]), row["Type"], row["IV"])
            for _, row in edited_df.iterrows()
        ]

        totalen, marktwaarde = rk.portfolio_greeks(positie, S, r, q)

        st.subheader("Portfolio Greeks")
        cols = st.columns(6)
        cols[0].metric("Marktwaarde", f"{marktwaarde:.2f}")
        cols[1].metric("Delta", f"{totalen['delta']:.2f}")
        cols[2].metric("Gamma", f"{totalen['gamma']:.2f}")
        cols[3].metric("Vega", f"{totalen['vega']:.2f}")
        cols[4].metric("Theta", f"{totalen['theta']:.2f}")
        cols[5].metric("Rho", f"{totalen['rho']:.2f}")

        st.subheader("Vega bucketing")
        buckets_strikes = np.array([90.0, 100.0, 110.0, 120.0])
        buckets_maturities = np.array([0.25, 0.5, 1.0, 2.0])
        vega_matrix = rk.vega_buckets(positie, S, r, q, buckets_strikes,
                                        buckets_maturities)

        fig = go.Figure(data=go.Heatmap(
            x=[f"K={k:.0f}" for k in buckets_strikes],
            y=[f"T={t:.2f}" for t in buckets_maturities],
            z=vega_matrix,
            colorscale="RdYlGn",
            zmid=0,
        ))
        fig.update_layout(height=400, title="Vega per bucket")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("P&L attribution")
        scenarios = [
            ("Underlying +1%", 0.01, 0.0, 0.0),
            ("Underlying -1%", -0.01, 0.0, 0.0),
            ("Vol +1 punt", 0.0, 0.01, 0.0),
            ("Vol -1 punt", 0.0, -0.01, 0.0),
            ("Rente +10bp", 0.0, 0.0, 10.0),
            ("Alles tegelijk", 0.01, 0.01, 10.0),
        ]
        pnl_rows = []
        for naam, dS, dSig, dR in scenarios:
            pnl = rk.pnl_attribution(positie, S, r, q, dS, dSig, dR)
            pnl_rows.append({
                "Scenario": naam,
                "Delta": f"{pnl['Delta P&L']:+.2f}",
                "Gamma": f"{pnl['Gamma P&L']:+.2f}",
                "Vega": f"{pnl['Vega P&L']:+.2f}",
                "Rho": f"{pnl['Rho P&L']:+.2f}",
                "Totaal": f"{pnl['Totaal P&L']:+.2f}",
            })
        st.dataframe(pd.DataFrame(pnl_rows), use_container_width=True)


# ============================================================
# Modellen vergelijken
# ============================================================
elif pagina == "🔬 Modellen vergelijken":
    st.title("🔬 SABR vs Heston vs Dupire")
    st.markdown("Vergelijk drie modellen op dezelfde realistische marktdata.")

    S = st.number_input("Spot S", value=4500.0, step=50.0, key="cmp_S")
    r = st.number_input("Rente r", value=0.03, step=0.005, format="%.4f", key="cmp_r")
    q = st.number_input("Dividend q", value=0.015, step=0.005, format="%.4f", key="cmp_q")

    if st.button("Vergelijk modellen"):
        with st.spinner("Modellen kalibreren (SABR snel, Heston traag)..."):
            _, df = md.sla_realistische_marktdata_op(None, S, r, q, seed=42)
            markt_iv_data = [(row["K"], row["T"], row["iv_markt"])
                              for _, row in df.iterrows()]

            params_sabr = sb.kalibreer_sabr_volledig(markt_iv_data, S, r, q, beta=0.5)
            sabr_resid = sum(p["residueel"] for p in params_sabr.values())

            params_heston, heston_resid = h.kalibreer_heston_iv(markt_iv_data, S, r, q)

        col_a, col_b = st.columns(2)
        col_a.metric("SABR residueel", f"{sabr_resid:.2e}")
        col_b.metric("Heston residueel", f"{heston_resid:.2e}")

        if heston_resid < sabr_resid:
            st.info(f"**Heston** fit beter (factor {heston_resid/sabr_resid:.2f})")
        else:
            st.info(f"**SABR** fit beter (factor {sabr_resid/heston_resid:.2f})")

        v0, kappa, theta, sigma_v, rho = params_heston
        st.subheader("Heston parameters")
        st.dataframe(pd.DataFrame([
            {"Parameter": "v0", "Waarde": f"{v0:.4f}"},
            {"Parameter": "κ", "Waarde": f"{kappa:.4f}"},
            {"Parameter": "θ", "Waarde": f"{theta:.4f}"},
            {"Parameter": "σ_v", "Waarde": f"{sigma_v:.4f}"},
            {"Parameter": "ρ", "Waarde": f"{rho:.4f}"},
        ]), use_container_width=True)

        st.subheader("SABR parameters per maturity")
        sabr_rows = [
            {"T": f"{T:.3f}", "α": f"{p['alpha']:.4f}", "ρ": f"{p['rho']:+.4f}",
             "ν": f"{p['nu']:.4f}", "Residueel": f"{p['residueel']:.2e}"}
            for T, p in params_sabr.items()
        ]
        st.dataframe(pd.DataFrame(sabr_rows), use_container_width=True)


# ============================================================
# FX Analyse
# ============================================================
elif pagina == "💱 FX Analyse":
    st.title("💱 FX Analyse")
    st.markdown("Analyseer een valutapaar met RSI, MACD, SMA en Bollinger Bands.")

    col1, col2, col3 = st.columns(3)
    with col1:
        pair = st.selectbox("Valutapaar",
                             ["EURUSD=X", "EURGBP=X", "GBPUSD=X", "USDJPY=X",
                              "AUDUSD=X", "USDCHF=X", "USDCAD=X", "NZDUSD=X"])
    with col2:
        start = st.text_input("Start datum", value="2023-01-01", key="fx_start")
    with col3:
        dagen = st.slider("Dagen in plot", 60, 500, 180, 20, key="fx_dagen")

    if st.button("Analyseer FX"):
        import fx as fx_mod
        with st.spinner(f"FX-analyse voor {pair}..."):
            try:
                rapport = fx_mod.maak_fx_rapport(pair, start=start)
            except Exception as e:
                st.error(f"Fout: {e}")
                st.stop()

        s = rapport["stats"]
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Spot", f"{s['spot']:.4f}")
        c2.metric("Volatiliteit (jaar)", f"{s['vol_jaar']:.2f}%")
        c3.metric("52w min", f"{s['min_52w']:.4f}")
        c4.metric("52w max", f"{s['max_52w']:.4f}")

        st.subheader("Signalen")
        signalen_df = pd.DataFrame(
            rapport["signalen"],
            columns=["Indicator", "Waarde", "Status", "Signaal"]
        )
        st.dataframe(signalen_df, use_container_width=True, hide_index=True)

        with st.spinner("Plot genereren..."):
            try:
                pad = fx_mod.plot_fx(rapport, dagen=dagen)
                st.success(f"Plot opgeslagen: {pad}")
                if os.path.exists(pad):
                    st.image(pad, use_container_width=True)
            except Exception as e:
                st.warning(f"Plot kon niet worden gemaakt: {e}")


# ============================================================
# Carry Trades
# ============================================================
elif pagina == "💸 Carry Trades":
    st.title("💸 Carry Trades")
    st.markdown("""
    **Carry trade** = lenen in een valuta met **lage rente**, beleggen in een
    valuta met **hogere rente**. Het verschil is de *carry*.

    De **score** is carry gedeeld door volatiliteit — hoe hoger, hoe beter de
    risk-adjusted carry.
    """)

    start = st.text_input("Start datum", value="2023-01-01", key="carry_start")

    if st.button("Bereken carry trades"):
        import fx_carry
        with st.spinner("Carry-analyse..."):
            rijen = fx_carry.maak_carry_rapport(start=start)

        if not rijen:
            st.warning("Geen data beschikbaar.")
            st.stop()

        df_carry = pd.DataFrame(rijen)
        df_carry = df_carry.sort_values("score", ascending=False).reset_index(drop=True)

        st.subheader("Carry-trade scores")
        df_show = df_carry[["pair", "funding", "target", "carry",
                              "vol_jaar", "score", "spot"]].copy()
        df_show.columns = ["Pair", "Funding", "Target", "Carry %",
                            "Vol %", "Score", "Spot"]
        st.dataframe(df_show.round(4), use_container_width=True, hide_index=True)

        st.subheader("Score per pair (staafdiagram)")
        kleuren = ["#3fb950" if s > 0.5 else ("#d97706" if s > 0.2 else "#f85149")
                    for s in df_carry["score"]]
        fig = go.Figure(data=[go.Bar(
            x=df_carry["pair"],
            y=df_carry["score"],
            marker_color=kleuren,
            text=[f"{s:.3f}" for s in df_carry["score"]],
            textposition="outside",
        )])
        fig.add_hline(y=0.5, line_dash="dot", line_color="#3fb950",
                       annotation_text="Aantrekkelijk (>0.5)")
        fig.add_hline(y=0.2, line_dash="dot", line_color="#d97706",
                       annotation_text="Matig (>0.2)")
        fig.update_layout(xaxis_title="Valutapaar", yaxis_title="Score",
                            height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Carry % vs Volatiliteit")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_carry["vol_jaar"], y=df_carry["carry"],
            mode="markers+text",
            text=df_carry["pair"],
            textposition="top center",
            marker=dict(size=14, color=df_carry["score"],
                          colorscale="RdYlGn", showscale=True,
                          colorbar=dict(title="Score")),
        ))
        fig2.update_layout(xaxis_title="Volatiliteit (%)", yaxis_title="Carry (%)",
                             height=450, hovermode="closest")
        st.plotly_chart(fig2, use_container_width=True)

        st.info("""
        **Interpretatie:**
        - **Score > 0.5** — aantrekkelijke risk-adjusted carry
        - **Score 0.2 – 0.5** — matig, let op risk-off momenten
        - **Score < 0.2** — carry weegt niet op tegen volatiliteit
        """)


# ============================================================
# Valuta-Exposure
# ============================================================
elif pagina == "🌍 Valuta-Exposure":
    st.title("🌍 Valuta-Exposure")
    st.markdown("Bereken de blootstelling per valuta in je portefeuille.")

    posities_input = st.text_area(
        "Posities (één per regel, formaat: naam:waarde:valuta)",
        value="ASML:10000:EUR\nAAPL:15000:USD\nSHELL:8000:GBP\nTM:5000:JPY",
        height=150,
        key="exp_input"
    )

    if st.button("Bereken exposure"):
        import fx_carry
        posities = []
        for regel in posities_input.strip().split("\n"):
            regel = regel.strip()
            if not regel:
                continue
            try:
                naam, waarde, valuta = regel.split(":")
                posities.append({
                    "naam": naam.strip(),
                    "waarde": float(waarde),
                    "valuta": valuta.strip().upper(),
                })
            except ValueError:
                st.error(f"Ongeldige regel: {regel}")
                st.stop()

        exposure, totaal = fx_carry.bereken_exposure(posities)

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Totale waarde", f"{totaal:,.2f}")
        c2.metric("Aantal valuta's", len(exposure))

        st.subheader("Exposure per valuta")
        df_exp = pd.DataFrame([
            {"Valuta": v, "Bedrag": totaal * pct / 100, "Percentage": pct}
            for v, pct in sorted(exposure.items(), key=lambda x: -x[1])
        ])
        st.dataframe(df_exp.round(2), use_container_width=True, hide_index=True)

        st.subheader("Verdeling (taartdiagram)")
        fig = go.Figure(data=[go.Pie(
            labels=list(exposure.keys()),
            values=list(exposure.values()),
            hole=0.4,
            textinfo="label+percent",
        )])
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        grootste = max(exposure.values())
        if grootste > 50:
            st.warning(f"⚠ Concentratierisico: {grootste:.1f}% in één valuta.")
        else:
            st.success(f"✓ Verdeling redelijk gespreid (max {grootste:.1f}%).")


# ============================================================
# Valutarisico
# ============================================================
elif pagina == "⚠️ Valutarisico":
    st.title("⚠️ Valutarisico")
    st.markdown("""
    Combineert je **valuta-exposure** met **FX-signalen** (KOOP/VERKOOP)
    tot een risicoscore. Plus een **hedge-advies**.
    """)

    col1, col2 = st.columns([1, 1])
    with col1:
        posities_input = st.text_area(
            "Posities (naam:waarde:valuta, één per regel)",
            value="ASML:10000:EUR\nAAPL:15000:USD\nSHELL:8000:GBP",
            height=150,
            key="risk_input"
        )
    with col2:
        pairs_input = st.text_input(
            "FX-paren voor signalen (komma-gescheiden)",
            value="EURUSD=X,GBPUSD=X,USDJPY=X",
            key="risk_pairs"
        )
        start = st.text_input("Start datum", value="2023-01-01", key="risk_start")

    if st.button("Bereken valutarisico"):
        import fx_carry
        import fx as fx_mod

        posities = []
        for regel in posities_input.strip().split("\n"):
            regel = regel.strip()
            if not regel:
                continue
            try:
                naam, waarde, valuta = regel.split(":")
                posities.append({
                    "naam": naam.strip(),
                    "waarde": float(waarde),
                    "valuta": valuta.strip().upper(),
                })
            except ValueError:
                st.error(f"Ongeldige regel: {regel}")
                st.stop()

        paren = [p.strip() for p in pairs_input.split(",") if p.strip()]

        with st.spinner(f"FX-signalen ophalen voor {len(paren)} paren..."):
            fx_signalen = {}
            for pair in paren:
                try:
                    rapport = fx_mod.maak_fx_rapport(pair, start=start)
                    sig = "Geen"
                    for _, _, _, s in rapport["signalen"]:
                        if s in ["KOOP", "VERKOOP"]:
                            sig = s
                            break
                    basis = pair[:3].upper()
                    quote = pair[3:6].upper()
                    fx_signalen[basis] = sig
                    fx_signalen[quote] = ("VERKOOP" if sig == "KOOP"
                                            else ("KOOP" if sig == "VERKOOP" else "Geen"))
                except Exception as e:
                    st.warning(f"{pair}: {e}")

        exposure, totaal = fx_carry.bereken_exposure(posities)

        st.markdown("---")
        st.subheader("FX-signalen")
        sig_df = pd.DataFrame([
            {"Pair": p, "Signaal": fx_signalen.get(p[:3].upper(), "Geen")}
            for p in paren
        ])
        st.dataframe(sig_df, use_container_width=True, hide_index=True)

        st.subheader("Valutarisico per valuta")
        rows = []
        totaal_score = 0.0
        for v, pct in sorted(exposure.items(), key=lambda x: -x[1]):
            sig = fx_signalen.get(v, "Geen")
            score = fx_carry.risico_score(pct, sig)
            totaal_score += score
            rows.append({
                "Valuta": v,
                "Exposure %": f"{pct:.2f}%",
                "FX-signaal": sig,
                "Risicoscore": round(score, 3),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown(f"### Totaalscore: **{totaal_score:+.3f}**")
        if totaal_score > 0.3:
            st.success("→ Positief: valutablootstelling werkt mee met FX-trend.")
        elif totaal_score < -0.3:
            st.error("→ Negatief: overweeg hedge of herweging van valuta's.")
        else:
            st.info("→ Neutraal: geen duidelijke richting in valutarisico.")

        st.subheader("Hedge-advies")
        adviezen = []
        for v, pct in sorted(exposure.items(), key=lambda x: -x[1]):
            sig = fx_signalen.get(v, "Geen")
            if sig == "VERKOOP" and pct > 20:
                advies_pct = min(pct * 0.6, 70)
                adviezen.append(f"**{v}** — hedge ongeveer {advies_pct:.0f}% "
                                  f"van je {pct:.1f}% exposure (signaal: VERKOOP)")
            elif sig == "KOOP" and pct > 30:
                adviezen.append(f"**{v}** — geen hedge nodig, "
                                  f"FX-trend is gunstig ({pct:.1f}% exposure)")
            elif pct > 40:
                adviezen.append(f"**{v}** — overweeg spreiding, "
                                  f"concentratie is {pct:.1f}%")

        if adviezen:
            for a in adviezen:
                st.markdown(f"- {a}")
        else:
            st.info("Geen specifiek hedge-advies — risico is beperkt.")


# ============================================================
# Footer
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Optieprijzer**  \n"
    "Professionele toolkit  \n"
    "Black-Scholes · Boom · MC  \n"
    "SABR · Heston · Dupire  \n"
    "Risk-analyse · FX · Carry"
)