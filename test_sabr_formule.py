"""test_sabr_formule.py — Controleer SABR-formule met bekende waarden"""
import sabr as sb
import numpy as np

# Test: ATM call
F = 100.0
K = 100.0
T = 1.0
alpha = 0.20
beta = 0.5
rho = -0.3
nu = 0.4

iv = sb.sabr_iv(F, K, T, alpha, beta, rho, nu)
print(f"ATM test: F={F}, K={K}, T={T}")
print(f"  Input: alpha={alpha}, beta={beta}, rho={rho}, nu={nu}")
print(f"  SABR IV: {iv:.4%}")
print(f"  Verwacht ~20% (want ATM vol = alpha)")
print()

# Test: OTM strikes
print("Smile test (F=100, T=1):")
print(f"{'K':>8} {'IV':>10}")
for K in [80, 90, 95, 100, 105, 110, 120]:
    iv = sb.sabr_iv(F, K, T, alpha, beta, rho, nu)
    print(f"{K:>8.0f} {iv:>10.4%}")

print()
print("Verwachte vorm:")
print("  - ATM (K=100) rond 20%")
print("  - Lagere strikes (K=80,90) iets hoger (put skew)")
print("  - Hogere strikes (K=110,120) iets lager")
print("  - Verschil tussen K=80 en K=120 moet 2-5% zijn")