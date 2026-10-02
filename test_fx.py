"""test_fx.py — Demo valutamarkten analyse"""
import fx


def main():
    print("=" * 70)
    print("  VALUTA-ANALYSE — Demo")
    print("=" * 70)

    # Belangrijkste valutaparen
    paren = ["EURUSD=X", "EURGBP=X", "USDJPY=X", "EURCHF=X"]

    for pair in paren:
        print(f"\n  Analyseren: {pair}...")
        try:
            rapport = fx.maak_fx_rapport(pair, start="2023-01-01")
            fx.print_fx_rapport(rapport)
            pad = fx.plot_fx(rapport)
            print(f"  Plot opgeslagen: {pad}")
        except Exception as e:
            print(f"  Fout bij {pair}: {e}")

    # Carry trade voorbeeld
    print()
    print("=" * 70)
    print("  CARRY TRADE ANALYSE")
    print("=" * 70)
    print()
    carry = fx.carry_trade_analyse("AUDJPY=X", rente_base=4.35, rente_quote=-0.10)
    print(f"  Pair: AUD/JPY")
    print(f"  Rente AUD: 4.35%")
    print(f"  Rente JPY: -0.10%")
    print(f"  Carry    : {carry['carry']:+.2f}%")
    print(f"  Advies   : {carry['advies']}")
    print()


if __name__ == "__main__":
    main()