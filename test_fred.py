"""test_fred.py — Demo van FRED macro-economische data"""
import fred


def main():
    print("=" * 70)
    print("  FRED MACRO-ECONOMISCH RAPPORT — Demo")
    print("=" * 70)
    print()

    # Volledig rapport
    rapport = fred.maak_macro_rapport(start="2015-01-01")
    fred.print_macro_rapport(rapport)

    # Plot
    print("  Plot genereren...")
    pad = fred.plot_macro(rapport)
    print(f"  Opgeslagen: {pad}")

    print()
    print("=" * 70)
    print("  KLAAR")
    print("=" * 70)


if __name__ == "__main__":
    main()