"""bates.py — Bates model (Heston + Merton jumps) met verbeterde kalibratie"""
import os
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, brentq
from scipy.stats import norm


def bates_cf(u, S, T, r, q, v0, kappa, theta, sigma_v, rho,
             lam, mu_J, sigma_J):
    i = 1j
    xi = kappa - sigma_v * rho * i * u
    d = np.sqrt(xi ** 2 + sigma_v ** 2 * (i * u + u ** 2))
    g = (xi - d) / (xi + d)
    C = (kappa * theta / sigma_v ** 2) * (
        (xi - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    )
    D = ((xi - d) / sigma_v ** 2) * (
        (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    )
    heston_part = np.exp(i * u * (np.log(S) + (r - q) * T) + C + D * v0)

    jump_cf = np.exp(
        lam * T * (np.exp(i * u * mu_J - 0.5 * u ** 2 * sigma_J ** 2) - 1)
        - i * u * lam * T * (np.exp(mu_J + 0.5 * sigma_J ** 2) - 1)
    )
    return heston_part * jump_cf


def bates_call(S, K, T, r, q, v0, kappa, theta, sigma_v, rho,
               lam, mu_J, sigma_J, alpha=1.5):
    def integrand(u):
        cf = bates_cf(u - (alpha + 1) * 1j, S, T, r, q,
                      v0, kappa, theta, sigma_v, rho,
                      lam, mu_J, sigma_J)
        denom = alpha ** 2 + alpha - u ** 2 + 1j * u * (2 * alpha + 1)
        return np.real(np.exp(-1j * u * np.log(K)) * cf / denom)
    integraal, _ = quad(integrand, 0, 200, limit=200)
    prijs = np.exp(-alpha * np.log(K)) / np.pi * integraal * np.exp(-r * T)
    return max(prijs, 0.0)


def bates_put(S, K, T, r, q, v0, kappa, theta, sigma_v, rho,
              lam, mu_J, sigma_J):
    call = bates_call(S, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                      lam, mu_J, sigma_J)
    return call - S * np.exp(-q * T) + K * np.exp(-r * T)


def bates_price(S, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                lam, mu_J, sigma_J, optie="call"):
    if optie == "call":
        return bates_call(S, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                          lam, mu_J, sigma_J)
    return bates_put(S, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                     lam, mu_J, sigma_J)


def bs_call(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def implied_vol(prijs, S, K, T, r, optie="call", q=0.0):
    f = bs_call if optie == "call" else bs_put
    def g(sigma):
        return f(S, K, T, r, sigma, q) - prijs
    try:
        return brentq(g, 1e-6, 5.0)
    except ValueError:
        return np.nan


# ============================================================
# Kalibratie met log-scaling en multi-start
# ============================================================
def _log_transform(params_log):
    """Zet log-parameters om in echte parameters."""
    v0, kappa, theta, sigma_v, rho, lam, mu_J, sigma_J = params_log
    return (
        np.exp(v0),      # v0 > 0
        kappa,           # kappa > 0 (bounds bewaren)
        np.exp(theta),   # theta > 0
        np.exp(sigma_v), # sigma_v > 0
        rho,             # -1 < rho < 1
        np.exp(lam),     # lam > 0
        mu_J,            # mu_J vrij
        np.exp(sigma_J), # sigma_J > 0
    )


def _doel_log(params_log, markt_data, S, r, q):
    """Doelfunctie in log-ruimte."""
    # Bounds check in originele ruimte
    v0, kappa, theta, sigma_v, rho, lam, mu_J, sigma_J = _log_transform(params_log)

    if (v0 <= 0 or kappa <= 0 or theta <= 0 or sigma_v <= 0 or
        abs(rho) >= 0.99 or lam < 0 or sigma_J <= 0 or lam > 5 or sigma_J > 2):
        return 1e10

    ss = 0.0
    for K, T, iv_markt in markt_data:
        try:
            prijs = bates_call(S, K, T, r, q, v0, kappa, theta,
                                sigma_v, rho, lam, mu_J, sigma_J)
            iv_model = implied_vol(prijs, S, K, T, r, "call", q)
            if np.isnan(iv_model):
                ss += 1e4
            else:
                ss += (iv_model - iv_markt) ** 2
        except Exception:
            ss += 1e4
    return ss


def kalibreer_bates(markt_data, S, r, q, n_start=3, verbose=False):
    """
    Kalibreer Bates met multi-start en log-scaling.
    markt_data: lijst van (K, T, iv_markt).
    """
    # Meerdere startpunten in LOG-ruimte
    starts = [
        # v0, kappa, theta, sigma_v, rho, lam, mu_J, sigma_J
        [np.log(0.04), 2.0, np.log(0.04), np.log(0.3), -0.7, np.log(0.1), -0.1, np.log(0.15)],
        [np.log(0.03), 1.5, np.log(0.05), np.log(0.5), -0.6, np.log(0.2), -0.15, np.log(0.20)],
        [np.log(0.05), 3.0, np.log(0.03), np.log(0.4), -0.8, np.log(0.05), -0.05, np.log(0.10)],
    ][:n_start]

    grenzen_log = [
        (np.log(1e-4), np.log(1.0)),   # v0
        (0.1, 10.0),                   # kappa
        (np.log(1e-4), np.log(1.0)),   # theta
        (np.log(0.01), np.log(2.0)),   # sigma_v
        (-0.99, 0.0),                  # rho
        (np.log(1e-4), np.log(2.0)),   # lam
        (-1.0, 0.5),                   # mu_J
        (np.log(0.01), np.log(1.0)),   # sigma_J
    ]

    beste_resultaat = None
    beste_ss = np.inf

    for i, start in enumerate(starts):
        if verbose:
            print(f"    Start {i+1}/{len(starts)}...")
        try:
            res = minimize(
                _doel_log, start, args=(markt_data, S, r, q),
                method="L-BFGS-B", bounds=grenzen_log,
                options={"maxiter": 300, "ftol": 1e-10}
            )
            if res.fun < beste_ss:
                beste_ss = res.fun
                beste_resultaat = res
                if verbose:
                    print(f"      SS = {res.fun:.6e} (nieuw beste)")
        except Exception as e:
            if verbose:
                print(f"      Fout: {e}")

    if beste_resultaat is None:
        return None, np.inf

    # Converteer terug naar echte parameters
    params = _log_transform(beste_resultaat.x)
    return np.array(params), beste_ss


def plot_bates_vs_market(markt_data, S, r, q, params, output_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(output_dir, exist_ok=True)

    v0, kappa, theta, sigma_v, rho, lam, mu_J, sigma_J = params

    per_T = {}
    for item in markt_data:
        K, T = item[0], item[1]
        if len(item) == 3:
            iv_markt = item[2]
        else:
            prijs = item[2]
            optie = item[3] if len(item) > 3 else "call"
            iv_markt = implied_vol(prijs, S, K, T, r, optie, q)
        if iv_markt is not None and not np.isnan(iv_markt):
            per_T.setdefault(T, []).append((K, iv_markt))

    fig, ax = plt.subplots(figsize=(9, 6))
    kleuren = plt.cm.plasma(np.linspace(0, 1, max(len(per_T), 2)))

    for (T, punten), kleur in zip(sorted(per_T.items()), kleuren):
        punten_sorted = sorted(punten, key=lambda x: x[0])
        strikes = [k for k, _ in punten_sorted]
        markt_ivs = [iv for _, iv in punten_sorted]
        model_ivs = []
        for K in strikes:
            prijs = bates_call(S, K, T, r, q, v0, kappa, theta,
                                sigma_v, rho, lam, mu_J, sigma_J)
            iv_model = implied_vol(prijs, S, K, T, r, "call", q)
            model_ivs.append(iv_model if not np.isnan(iv_model) else 0)
        ax.plot(strikes, np.array(markt_ivs) * 100, "o", color=kleur,
                alpha=0.5, markersize=6)
        ax.plot(strikes, np.array(model_ivs) * 100, "-", color=kleur,
                linewidth=2, label=f"T={T:.2f}")

    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(
        f"Bates kalibratie (multi-start)\n"
        f"v0={v0:.4f}, kappa={kappa:.2f}, theta={theta:.4f}, sigma_v={sigma_v:.3f}, rho={rho:.3f}\n"
        f"lambda={lam:.3f}, mu_J={mu_J:.4f}, sigma_J={sigma_J:.4f}"
    )
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pad = os.path.join(output_dir, "bates_fit.png")
    fig.savefig(pad, dpi=120)
    plt.close(fig)
    return pad