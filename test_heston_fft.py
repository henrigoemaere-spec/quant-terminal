"""test_heston_fft.py — Vergelijk FFT-prijzen met analytische Heston-prijzen"""
import numpy as np
import heston as h
import heston_fft as hf

S = 7666.0
r, q = 0.045, 0.013
v0, kappa, theta, sigma_v, rho = 0.04, 2.0, 0.04, 0.5, -0.7

print("Vergelijk FFT vs analytische Heston-prijzen:")
print(f"S={S}, r={r}, q={q}")
print(f"v0={v0}, kappa={kappa}, theta={theta}, sigma_v={sigma_v}, rho={rho}")
print()
print(f"{'T':>6} {'K':>8} {'FFT':>14} {'Analytisch':>14} {'Verschil':>12}")
print("-" * 60)

for T in [0.05, 0.1, 0.25, 0.5]:
    for K in [7300, 7500, 7666, 7800, 8000]:
        try:
            prijs_fft = hf.heston_call_fft(
                np.array([K]), S, T, r, q,
                v0, kappa, theta, sigma_v, rho,
                N=512, eta=0.25
            )[0]
        except Exception as e:
            prijs_fft = f"FOUT:{e}"

        try:
            prijs_h = h.heston_call(S, K, T, r, q,
                                      v0, kappa, theta, sigma_v, rho)
        except Exception as e:
            prijs_h = f"FOUT:{e}"

        if isinstance(prijs_fft, (int, float, np.floating)) and \
           isinstance(prijs_h, (int, float, np.floating)):
            verschil = float(prijs_fft) - float(prijs_h)
            print(f"{T:>6.2f} {K:>8.0f} {float(prijs_fft):>14.2f} "
                  f"{float(prijs_h):>14.2f} {verschil:>12.2f}")
        else:
            print(f"{T:>6.2f} {K:>8.0f} {prijs_fft} {prijs_h}")

# Test ook met ATM strikes uit de echte data
print()
print("Test met SPX-achtige strikes (ATM rond 7666):")
print(f"{'T':>6} {'K':>8} {'FFT':>14} {'Analytisch':>14} {'Verschil':>12}")
print("-" * 60)

for T in [0.05, 0.1]:
    for K in [7600, 7650, 7666, 7700, 7750]:
        try:
            prijs_fft = hf.heston_call_fft(
                np.array([K]), S, T, r, q,
                v0, kappa, theta, sigma_v, rho,
                N=2048, eta=0.15
            )[0]
            prijs_h = h.heston_call(S, K, T, r, q,
                                      v0, kappa, theta, sigma_v, rho)
            verschil = float(prijs_fft) - float(prijs_h)
            print(f"{T:>6.2f} {K:>8.0f} {float(prijs_fft):>14.2f} "
                  f"{float(prijs_h):>14.2f} {verschil:>12.2f}")
        except Exception as e:
            print(f"{T:>6.2f} {K:>8.0f} FOUT: {e}")