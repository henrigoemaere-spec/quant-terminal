"""run_tests.py — Voer alle tests uit"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  Optieprijzer — Test suite")
print("=" * 60)
print()

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    capture_output=False
)

sys.exit(result.returncode)