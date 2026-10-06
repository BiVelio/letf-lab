"""Utilidades compartidas por las líneas de investigación."""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

from letf_lab import data
from letf_lab.universe import BY_TICKER

log = logging.getLogger(__name__)

# Rendimiento por dividendo aproximado para índices de precio (sin dividendos).
DIV_YIELD = {"^NDX": 0.008, "^RUT": 0.015, "^DJI": 0.022, "^SOX": 0.012, "^GSPC": 0.035}


def rets(tickers: list[str] | str) -> pd.DataFrame:
    """Retornos diarios total-return (adj_close) — sin rellenar huecos."""
    tickers = [tickers] if isinstance(tickers, str) else tickers
    px = data.panel(tickers, "adj_close")
    return px.pct_change(fill_method=None)


def rf(index: pd.DatetimeIndex) -> pd.Series:
    return data.risk_free_daily(index)


def pair(letf: str, clean: bool = True) -> pd.DataFrame:
    """DataFrame con retornos alineados del LETF, su subyacente y rf.

    - Recorta al periodo con el apalancamiento actual (`leverage_since`).
    - clean=True: los días en que el LETF se aparta >20 pp de L·r_u son errores
      de datos (típicamente reverse splits mal ajustados en Yahoo) y se sustituyen
      por L·r_u. Se registran en `r.attrs["bad_days"]`.
    """
    e = BY_TICKER[letf]
    r = rets([letf, e.underlying]).dropna()
    r.columns = ["letf", "u"]
    if e.leverage_since:
        r = r[r.index >= e.leverage_since]
    r["rf"] = rf(r.index)
    bad = (r.letf - e.leverage * r.u).abs() > 0.20
    if clean and bad.any():
        log.warning("%s: %d días con dato erróneo sustituidos: %s", letf, bad.sum(),
                    list(r.index[bad].date))
        r.loc[bad, "letf"] = e.leverage * r.loc[bad, "u"]
    r.attrs["bad_days"] = list(r.index[bad])
    return r


@lru_cache(maxsize=None)
def long_underlying(letf: str) -> pd.Series:
    """Retorno total diario del subyacente con la historia más larga posible.

    Usa el ETF 1x (total return) donde existe y, antes, el índice largo
    (+ dividendo aproximado si es un índice de precio).
    """
    e = BY_TICKER[letf]
    etf = rets(e.underlying)[e.underlying].dropna()
    if not e.index_long:
        return etf.rename("u")
    idx_px = data.panel([e.index_long], "adj_close")[e.index_long]
    idx = idx_px.pct_change(fill_method=None).dropna()
    if e.index_long in DIV_YIELD:
        idx = idx + DIV_YIELD[e.index_long] / 252
    if e.index_long == "^SP500TR":  # antes de 1988, ^GSPC + dividendo aprox.
        g = data.panel(["^GSPC"], "adj_close")["^GSPC"].pct_change(fill_method=None).dropna()
        g = g[g.index >= "1960-01-01"] + DIV_YIELD["^GSPC"] / 252
        idx = pd.concat([g[g.index < idx.index[0]], idx])
    spliced = pd.concat([idx[idx.index < etf.index[0]], etf])
    return spliced.rename("u")
