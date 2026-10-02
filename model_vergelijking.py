"""model_vergelijking.py — Vergelijk SABR, Heston en Bates op dezelfde marktdata"""
import os
import time
import traceback
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, minimize


warnings.filterwarnings("ignore", category=Warning)


def bs_call(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def laad_cboe_data(pad="cboe_SPX.csv"):
    """Laadt CBOE data met ruime ATM-filter (10%)."""
    if not os.path.exists(pad):
        raise FileNotFoundError(f"Bestand niet gevonden: {pad}")

    df = pd.read_csv(pad)
    S_spot = float(df["S"].iloc[0])

    calls = df[df["type"] == "call"].copy()
    calls = calls[(calls["T"] > 0.02) & (calls["T"] < 1.0)]

    F = S_spot * np.exp(0.032 * calls["T"])
    calls = calls[(calls["K"] > F * 0.90) & (calls["K"] < F * 1.10)]
    calls = calls[(calls["iv_cboe"] > 0.05) & (calls["iv_cboe"] < 0.60)]

    geldige = [exp for exp, g in calls.groupby("expiry") if len(g) >= 4]
    calls = calls[calls["expiry"].isin(geldige)]

    markt = [(float(row["K"]), float(row["T"]), float(row["iv_cboe"]))
             for _, row in calls.iterrows()]
    return S_spot, markt


def kalibreer_sabr(markt, S_spot, r, q, beta=0.5):
    """SABR-kalibratie per maturity."""
    import sabr as sb
    t0 = time.time()

    per_T = {}
    for K, T, iv in markt:
        per_T.setdefault(T, []).append((K, iv))

    resultaten = {}
    for T, punten in sorted(per_T.items()):
        try:
            strikes = np.array([k for k, _ in punten])
            ivs = np.array([iv for _, iv in punten])
            F = float(S_spot) * np.exp((r - q) * T)

            atm_idx = np.argmin(np.abs(strikes - F))
            atm_vol = float(ivs[atm_idx])
            start = (atm_vol, -0.5, 0.5)

            params, ss = sb.kalibreer_sabr_smile(
                F, T, strikes, ivs, beta=beta, start=start
            )
            alpha, rho, nu = params
            resultaten[float(T)] = {
                "alpha": float(alpha), "beta": float(beta),
                "rho": float(rho), "nu": float(nu),
                "residueel": float(ss),
            }
        except Exception as e:
            print(f"    SABR fout bij T={T}: {e}")
            continue

    residual = sum(p["residueel"] for p in resultaten.values())
    return {
        "model": "SABR", "residual": float(residual),
        "tijd": time.time() - t0, "params": resultaten,
        "n_params": 3 * len(resultaten),
    }


def kalibreer_heston(markt, S_spot, r, q):
    """Heston via FFT met ruime grenzen."""
    import heston_fft as hf
    t0 = time.time()

    per_T = {}
    for K, T, iv_markt in markt:
        per_T.setdefault(T, []).append((K, iv_markt))

    data_prijs = {}
    for T, punten in per_T.items():
        strikes = np.array([k for k, _ in punten])
        prijzen = []
        for K, iv in punten:
            d1 = (np.log(S_spot / K) + (r - q + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
            d2 = d1 - iv * np.sqrt(T)
            prijs = (S_spot * np.exp(-q * T) * norm.cdf(d1)
                     - K * np.exp(-r * T) * norm.cdf(d2))
            prijzen.append(prijs)
        data_prijs[T] = (strikes, np.array(prijzen))

    def doel(params):
        v0, kappa, theta, sigma_v, rho = params

        if (v0 <= 1e-4 or kappa <= 0.05 or theta <= 1e-4 or
                sigma_v <= 1e-3 or abs(rho) >= 0.99):
            return 1e10

        ss = 0.0
        for T, (strikes, prijzen_markt) in data_prijs.items():
            try:
                prijzen_model = hf.heston_call_fft(
                    strikes, S_spot, T, r, q,
                    v0, kappa, theta, sigma_v, rho,
                    N=2048, eta=0.15
                )
                ss += np.sum((prijzen_markt - prijzen_model) ** 2)
            except Exception:
                ss += 1e6
        return ss

    starts = [
        [0.04, 2.0, 0.04, 0.5, -0.7],
        [0.03, 1.5, 0.05, 0.6, -0.6],
        [0.05, 3.0, 0.03, 0.4, -0.8],
        [0.02, 1.0, 0.03, 0.7, -0.5],
    ]

    grenzen = [
        (0.001, 0.5),
        (0.1, 10.0),
        (0.001, 0.5),
        (0.01, 2.0),
        (-0.95, 0.5),
    ]

    beste = None
    beste_ss = np.inf
    for start in starts:
        try:
            res = minimize(doel, start, method="L-BFGS-B",
                            bounds=grenzen, options={"maxiter": 100})
            if res.fun < beste_ss:
                beste_ss = res.fun
                beste = res
        except Exception:
            pass

    if beste is None:
        return {"model": "Heston", "residual": np.inf,
                "tijd": time.time() - t0, "params": None, "fout": "faalde"}

    params = np.array(beste.x)
    return {
        "model": "Heston", "residual": float(beste_ss),
        "tijd": time.time() - t0, "params": params, "n_params": 5,
    }


def kalibreer_bates(markt, S_spot, r, q):
    import bates_fft as btf
    t0 = time.time()
    try:
        params, residual = btf.kalibreer_bates_fft(
            markt, S_spot, r, q, n_start=2, verbose=False
        )
        return {
            "model": "Bates", "residual": float(residual),
            "tijd": time.time() - t0, "params": params, "n_params": 8,
        }
    except Exception as e:
        return {"model": "Bates", "residual": np.inf,
                "tijd": time.time() - t0, "params": None, "fout": str(e)}


def bereken_model_ivs(markt_in, S_spot_in, r_in, q_in, resultaten_in):
    """Bereken IV's per model — alles lokaal, geen conflicten."""
    import sabr as sb
    import heston as h
    import bates as bt

    _S = float(S_spot_in)
    _r = float(r_in)
    _q = float(q_in)
    _markt = list(markt_in)
    _resultaten = dict(resultaten_in)

    per_T = {}
    for K, T, iv_markt in _markt:
        per_T.setdefault(T, []).append((K, iv_markt))

    resultaat_ivs = {}

    # --- SABR ---
    try:
        params_s = _resultaten.get("SABR", {}).get("params")
        if isinstance(params_s, dict):
            resultaat_ivs["SABR"] = {}
            for T, punten in per_T.items():
                if T not in params_s:
                    continue
                p = params_s[T]
                if not isinstance(p, dict):
                    continue
                try:
                    alpha_v = float(p["alpha"])
                    beta_v = float(p["beta"])
                    rho_v = float(p["rho"])
                    nu_v = float(p["nu"])
                except (KeyError, TypeError, ValueError):
                    continue
                F_v = _S * np.exp((_r - _q) * T)
                ivs = []
                for K, _ in punten:
                    try:
                        iv = sb.sabr_iv(F_v, float(K), float(T),
                                          alpha_v, beta_v, rho_v, nu_v)
                        if iv is None or not np.isfinite(iv):
                            ivs.append(0.0)
                        else:
                            ivs.append(float(iv))
                    except Exception:
                        ivs.append(0.0)
                resultaat_ivs["SABR"][T] = ivs
    except Exception as e:
        print(f"    SABR IVs fout: {e}")

    # --- Heston ---
    try:
        params_h = _resultaten.get("Heston", {}).get("params")
        if isinstance(params_h, np.ndarray) and len(params_h) == 5:
            v0 = float(params_h[0])
            kappa = float(params_h[1])
            theta = float(params_h[2])
            sigma_v = float(params_h[3])
            rho = float(params_h[4])
            resultaat_ivs["Heston"] = {}
            for T, punten in per_T.items():
                ivs = []
                for K, _ in punten:
                    try:
                        prijs = h.heston_call(_S, float(K), float(T),
                                                _r, _q, v0, kappa, theta,
                                                sigma_v, rho)
                        iv = h.implied_vol(prijs, _S, float(K),
                                             float(T), _r, "call", _q)
                        ivs.append(float(iv) if iv is not None
                                    and np.isfinite(iv) else 0.0)
                    except Exception:
                        ivs.append(0.0)
                resultaat_ivs["Heston"][T] = ivs
    except Exception as e:
        print(f"    Heston IVs fout: {e}")

    # --- Bates ---
    try:
        params_b = _resultaten.get("Bates", {}).get("params")
        if isinstance(params_b, np.ndarray) and len(params_b) == 8:
            v0 = float(params_b[0])
            kappa = float(params_b[1])
            theta = float(params_b[2])
            sigma_v = float(params_b[3])
            rho = float(params_b[4])
            lam = float(params_b[5])
            mu_J = float(params_b[6])
            sigma_J = float(params_b[7])
            resultaat_ivs["Bates"] = {}
            for T, punten in per_T.items():
                ivs = []
                for K, _ in punten:
                    try:
                        prijs = bt.bates_call(_S, float(K), float(T),
                                                _r, _q, v0, kappa, theta,
                                                sigma_v, rho,
                                                lam, mu_J, sigma_J)
                        iv = bt.implied_vol(prijs, _S, float(K),
                                              float(T), _r, "call", _q)
                        ivs.append(float(iv) if iv is not None
                                    and np.isfinite(iv) else 0.0)
                    except Exception:
                        ivs.append(0.0)
                resultaat_ivs["Bates"][T] = ivs
    except Exception as e:
        print(f"    Bates IVs fout: {e}")

    return resultaat_ivs, per_T


def plot_vergelijking(markt, S_spot, r, q, resultaten, output_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(output_dir, exist_ok=True)

    ivs, per_T = bereken_model_ivs(markt, S_spot, r, q, resultaten)

    Ts = sorted(per_T.keys())
    n_plots = min(6, len(Ts))
    indices = np.linspace(0, len(Ts) - 1, n_plots, dtype=int)
    Ts_select = [Ts[i] for i in indices]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    kleuren = {"SABR": "#d97706", "Heston": "#3fb950", "Bates": "#58a6ff"}

    for idx, T in enumerate(Ts_select):
        ax = axes[idx]
        punten = sorted(per_T[T], key=lambda x: x[0])
        strikes = [k for k, _ in punten]
        markt_ivs = [iv for _, iv in punten]

        ax.plot(strikes, np.array(markt_ivs) * 100, "o",
                color="black", markersize=8, alpha=0.6, label="Markt", zorder=5)

        for model, kleur in kleuren.items():
            if model in ivs and T in ivs[model]:
                model_ivs = ivs[model][T]
                if len(model_ivs) == len(strikes):
                    ax.plot(strikes, np.array(model_ivs) * 100, "-",
                            color=kleur, linewidth=2.5, label=model)

        ax.set_xlabel("Strike K", fontsize=11)
        ax.set_ylabel("Implied vol (%)", fontsize=11)
        ax.set_title(f"T = {T:.3f} jaar", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    for i in range(n_plots, 6):
        axes[i].axis("off")

    fig.suptitle(f"Model-vergelijking — {len(markt)} opties, S={S_spot:.2f}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    pad = os.path.join(output_dir, "model_vergelijking.png")
    fig.savefig(pad, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return pad


def plot_residualen(resultaten, output_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(output_dir, exist_ok=True)

    modellen = []
    residualen = []
    tijden = []
    n_params = []

    for naam in ["SABR", "Heston", "Bates"]:
        if naam in resultaten:
            r = resultaten[naam]
            res_val = r["residual"]
            if isinstance(res_val, (int, float, np.floating)) and res_val < 1e9:
                modellen.append(naam)
                residualen.append(float(res_val))
                tijden.append(float(r["tijd"]))
                n_params.append(int(r["n_params"]))

    if not modellen:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    kleuren = {"SABR": "#d97706", "Heston": "#3fb950", "Bates": "#58a6ff"}
    kleur_lijst = [kleuren[m] for m in modellen]

    bars1 = ax1.bar(modellen, residualen, color=kleur_lijst,
                     edgecolor="white", linewidth=1.5)
    ax1.set_ylabel("Residuele SS (IV²)", fontsize=12)
    ax1.set_title("Model-fit kwaliteit (lager = beter)",
                   fontsize=13, fontweight="bold")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars1, residualen):
        h_bar = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h_bar * 1.2,
                 f"{val:.2e}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")

    ax2_twin = ax2.twinx()

    bars2 = ax2.bar(modellen, tijden, color=kleur_lijst,
                     alpha=0.7, edgecolor="white", linewidth=1.5)
    ax2.set_ylabel("Kalibratietijd (s)", fontsize=12)
    ax2.set_title("Snelheid vs complexiteit", fontsize=13, fontweight="bold")

    ax2_twin.plot(modellen, n_params, "ro-", linewidth=2.5, markersize=12)
    ax2_twin.set_ylabel("Aantal parameters", fontsize=12, color="red")
    ax2_twin.tick_params(axis="y", labelcolor="red")

    ax2.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars2, tijden):
        h_bar = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h_bar * 1.05,
                 f"{val:.2f}s", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")

    legend_elements = [
        Patch(facecolor="gray", alpha=0.7, label="Tijd"),
        Line2D([0], [0], color="red", marker="o", linewidth=2.5,
                markersize=10, label="Aantal params")
    ]
    ax2.legend(handles=legend_elements, loc="upper left", fontsize=10)

    fig.tight_layout()
    pad = os.path.join(output_dir, "model_residualen.png")
    fig.savefig(pad, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return pad