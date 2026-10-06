"""Universo de ETFs apalancados y su subyacente de referencia.

`underlying` es el ETF 1x que se usa como proxy del índice que replica el LETF
(total return vía precio ajustado). `index_long` es un índice con historia más
larga, usado para sintetizar LETFs antes de su lanzamiento.

Notas de proxies imperfectos:
- SOXL/SOXS replican el ICE Semiconductor Index; SOXX lo sigue desde 2021
  (antes seguía el PHLX SOX), así que el tracking pre-2021 tiene ruido extra.
- LABU/LABD replican el S&P Biotech Select Industry (equiponderado) = XBI.
- TMF/TMV replican el ICE 20+ Year Treasury; TLT es un proxy muy cercano.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LETF:
    ticker: str
    leverage: float
    underlying: str
    expense_ratio: float  # anual, fracción
    family: str
    index_long: str | None = None  # índice para historia sintética larga
    # fecha desde la que rige `leverage` (antes era otro apalancamiento).
    # Direxion pasó NUGT/DUST/ERX/ERY de 3x a 2x el 2020-04-01.
    leverage_since: str | None = None


UNIVERSE: list[LETF] = [
    # S&P 500
    LETF("SSO", 2, "SPY", 0.0089, "SP500", "^SP500TR"),
    LETF("UPRO", 3, "SPY", 0.0091, "SP500", "^SP500TR"),
    LETF("SPXL", 3, "SPY", 0.0087, "SP500", "^SP500TR"),
    LETF("SH", -1, "SPY", 0.0089, "SP500", "^SP500TR"),
    LETF("SDS", -2, "SPY", 0.0089, "SP500", "^SP500TR"),
    LETF("SPXU", -3, "SPY", 0.0089, "SP500", "^SP500TR"),
    # Nasdaq-100
    LETF("QLD", 2, "QQQ", 0.0095, "NDX", "^NDX"),
    LETF("TQQQ", 3, "QQQ", 0.0084, "NDX", "^NDX"),
    LETF("PSQ", -1, "QQQ", 0.0095, "NDX", "^NDX"),
    LETF("QID", -2, "QQQ", 0.0095, "NDX", "^NDX"),
    LETF("SQQQ", -3, "QQQ", 0.0095, "NDX", "^NDX"),
    # Russell 2000
    LETF("TNA", 3, "IWM", 0.0108, "R2K", "^RUT"),
    LETF("TZA", -3, "IWM", 0.0104, "R2K", "^RUT"),
    # Dow
    LETF("UDOW", 3, "DIA", 0.0095, "DOW", "^DJI"),
    LETF("SDOW", -3, "DIA", 0.0095, "DOW", "^DJI"),
    # Semis
    LETF("SOXL", 3, "SOXX", 0.0075, "SEMI", "^SOX"),
    LETF("SOXS", -3, "SOXX", 0.0097, "SEMI", "^SOX"),
    # Tech sector
    LETF("TECL", 3, "XLK", 0.0085, "TECH", None),
    LETF("TECS", -3, "XLK", 0.0102, "TECH", None),
    # Financials
    LETF("FAS", 3, "XLF", 0.0089, "FIN", None),
    LETF("FAZ", -3, "XLF", 0.0104, "FIN", None),
    # Biotech
    LETF("LABU", 3, "XBI", 0.0093, "BIO", None),
    LETF("LABD", -3, "XBI", 0.0107, "BIO", None),
    # Long Treasuries
    LETF("TMF", 3, "TLT", 0.0091, "UST20", None),
    LETF("TMV", -3, "TLT", 0.0092, "UST20", None),
    # Gold miners
    LETF("NUGT", 2, "GDX", 0.0114, "GDX", None, "2020-04-01"),
    LETF("DUST", -2, "GDX", 0.0093, "GDX", None, "2020-04-01"),
    # Energy
    LETF("ERX", 2, "XLE", 0.0102, "ENERGY", None, "2020-04-01"),
    LETF("ERY", -2, "XLE", 0.0107, "ENERGY", None, "2020-04-01"),
]

BY_TICKER: dict[str, LETF] = {e.ticker: e for e in UNIVERSE}

RISK_FREE = "^IRX"  # T-bill 13 semanas, % anual (tasa de descuento)

# Pares bull/bear sobre el mismo índice y |L| igual, útiles para decay harvesting.
PAIRS: list[tuple[str, str]] = [
    ("TQQQ", "SQQQ"),
    ("UPRO", "SPXU"),
    ("SSO", "SDS"),
    ("QLD", "QID"),
    ("TNA", "TZA"),
    ("SOXL", "SOXS"),
    ("UDOW", "SDOW"),
    ("TECL", "TECS"),
    ("FAS", "FAZ"),
    ("LABU", "LABD"),
    ("TMF", "TMV"),
    ("NUGT", "DUST"),
    ("ERX", "ERY"),
]


def all_tickers() -> list[str]:
    s = {e.ticker for e in UNIVERSE} | {e.underlying for e in UNIVERSE}
    return sorted(s)
