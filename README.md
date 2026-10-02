\# Optieprijzer — Professionele toolkit voor optieprijzing



Een volledig systeem voor het prijzen van opties en risk-analyse, met 8 modellen

en zowel een CLI als een webinterface.



\## Features



\### Modellen

\- \*\*Black-Scholes\*\* — analytisch, Europees

\- \*\*Binomiale boom\*\* — Amerikaans, barrier

\- \*\*Monte Carlo\*\* — simulatie, validatie

\- \*\*Heston\*\* — stochastische volatiliteit

\- \*\*Heston FFT\*\* — snelle versie (Carr-Madan)

\- \*\*SABR\*\* — Hagan-formule, smile-kalibratie

\- \*\*Bates\*\* — Heston + jumps (Merton)

\- \*\*Dupire\*\* — local volatility



\### Analyse

\- Greeks (delta, gamma, vega, theta, rho)

\- Implied volatility uit marktprijs

\- Volatility surface met arbitrage-checks

\- Vega bucketing per strike × maturity

\- P\&L attribution voor scenario's



\### Data

\- Yahoo Finance (aandelen, ETF's)

\- CBOE delayed quotes (SPX, VIX, NDX)

\- Synthetische realistische marktdata

\- CSV import/export



\### Interfaces

\- \*\*CLI\*\* — 15+ subcommando's

\- \*\*Streamlit\*\* — webinterface met schuifjes en grafieken



\## Installatie



```bash

pip install -r requirements.txt

```



\## Snelstart



\### CLI



```bash

\# Prijs een Europese call

python optieprijzer.py price --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --type call



\# Amerikaanse put

python optieprijzer.py price --S 50 --K 50 --T 0.41666 --r 0.1 --sigma 0.4 --type put --style amerikaans



\# Greeks

python optieprijzer.py greeks --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --type call



\# Implied volatility

python optieprijzer.py iv --prijs 5.50 --S 100 --K 105 --T 0.5 --r 0.05 --type call



\# Vergelijk BS vs boom vs Amerikaans

python optieprijzer.py compare --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --type put



\# Risk-analyse op een portfolio

python optieprijzer.py risk



\# SABR kalibreren op synthetische data

python optieprijzer.py sabr-kalibreer



\# Bates kalibreren (snel via FFT)

python optieprijzer.py bates-kalibreer



\# Echte marktdata ophalen

python optieprijzer.py market-data --ticker SPY --max-expiries 8

```



\### Webinterface



```bash

streamlit run app.py

```



Open dan http://localhost:8501 in je browser.



\## Commando-overzicht



| Commando | Doel |

|----------|------|

| `price` | Prijs een optie (BS / boom / barrier) |

| `greeks` | Bereken alle Greeks |

| `iv` | Implied volatility uit marktprijs |

| `compare` | Vergelijk BS vs boom vs Amerikaans |

| `mc` | Monte Carlo simulatie |

| `smile` | Volatility smile uit CSV |

| `surface` | Volatility surface + arbitrage-checks |

| `heston` | Heston prijzen |

| `heston-kalibreer` | Heston kalibratie |

| `sabr` | SABR prijzen |

| `sabr-kalibreer` | SABR per maturity |

| `bates` | Bates prijzen |

| `bates-kalibreer` | Bates met FFT |

| `dupire` | Local volatility surface |

| `risk` | Vega bucketing + P\&L attribution |

| `benchmark-fft` | FFT vs quad snelheidstest |

| `market-data` | Yahoo Finance marktdata |



\## Tests



```bash

pytest tests/ -v

```



Of:



```bash

python run\_tests.py

```



\## Structuur



```

optieprijzer/

├── optieprijzer.py    # Hoofd CLI

├── app.py             # Streamlit webinterface

├── surface.py         # Volatility surface

├── heston.py          # Heston model

├── heston\_fft.py      # Snelle Heston via FFT

├── sabr.py            # SABR model

├── bates.py           # Bates model

├── bates\_fft.py       # Snelle Bates via FFT

├── dupire.py          # Dupire local volatility

├── marktdata.py       # Synthetische data

├── market\_data.py     # Yahoo Finance

├── cboe\_data.py       # CBOE delayed quotes

├── risk.py            # Risk-analyse

├── config.yaml        # Configuratie

├── requirements.txt   # Dependencies

├── tests/             # Pytest tests

└── plots/             # Output plots

```



\## Wetenschappelijke basis



Dit systeem implementeert de methoden uit:



\*\*Rüdiger U. Seydel\*\*, \*Tools for Computational Finance\*, 6e editie, Springer, 2017.



De implementaties zijn geverifieerd tegen:

\- Analytische Black-Scholes formules

\- Bekende benchmark-waarden

\- Put-call parity

\- Arbitrage-vrije eigenschappen



\## Belangrijke opmerkingen



\- \*\*Yahoo Finance\*\* data is vertraagd (15 min) en bevat ruis — gebruik voor onderzoek

\- \*\*CBOE data\*\* is schoner maar ook delayed

\- \*\*SABR\*\* werkt niet voor T < 0.05 (te korte maturities)

\- Voor \*\*echte handel\*\* zijn commerciële databronnen nodig (Bloomberg, Refinitiv)



\## Licentie



Dit is een educatief/professioneel hulpmiddel. Gebruik op eigen risico.

