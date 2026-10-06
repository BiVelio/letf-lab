"""Pares de ETFs inversos y alcistas del artículo 6 (datos compartidos por R31 y las líneas siguientes).

El pre-registro (letf/preregistro/articulo6_inversos.md, commit 8012967) fija los pares reales
SQQQ/TQQQ, SPXU/UPRO, SDS/SSO, QID/QLD, SH, PSQ, SOXS/SOXL, TZA/TNA, FAZ/FAS, LABD/LABU y
SDOW/UDOW «y los que la caché y Yahoo permitan», con su vida real (2006–2026 como mucho), y una
historia larga sintética con el simulador del laboratorio: S&P 500 desde 1928 y Nasdaq-100 desde
1985-10. Este módulo es el cargador común:

- PAIRS: los pares (inverso, alcista con el mismo |L|, |L|, ETF 1x que hace de índice, grupo,
  si está en el pre-registro y el tramo con ese apalancamiento). Ampliación, fuera del
  pre-registro: SPXS/SPXL, SRTY/URTY, TECS/TECL, SMDD/UMDD, TWM/UWM, DXD/DDM, MZZ/MVV, DOG, RWM,
  TMV/TMF y TBT/UBT (bonos), y ERY/ERX y DUST/NUGT, 3x hasta el 31-3-2020 y 2x desde el 1-4-2020
  (Direxion cambió su apalancamiento; universe.LETF.leverage_since).
- pair_frame / load_pairs: rendimientos diarios alineados del inverso, el alcista, el 1x (u), la
  letra (^IRX, data.risk_free_daily) y el tipo de financiación (f, history.funding_long), en las
  sesiones comunes hasta vix.END. Los días en que un fondo se aparta más de 20 pp de L·u son
  errores de datos (contrasplits mal ajustados en Yahoo) y se sustituyen por L·u, como
  common.pair; quedan en attrs["días_corregidos"].
- borrow_grid: la rejilla de préstamo del pre-registro por ticker (mínimo, mediana y p90 de la
  tasa diaria de IBKR en iBorrowDesk, vix.borrow_summary).
- synthetic_history: los fondos sintéticos de R22 (r22.synthetic con el s calibrado con cada ETF
  real, el 3x del laboratorio) sobre el S&P 500 desde 1928 y el Nasdaq-100 desde el 1-10-1985,
  para L = −3, −2, −1, 1, 2 y 3 (1 es el índice).
- download: descarga lo que falte (precios de Yahoo con sources.daily(source="yf") vía
  vix.etp_prices y JSON de iBorrowDesk con vix.download_borrow) a data/cache, sin refrescar lo
  que ya está.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from letf_lab.data import history, sources, vix
from letf_lab.universe import BY_TICKER

log = logging.getLogger(__name__)

END = vix.END
BAD_DAY = 0.20                     # |fondo − L·u| > 20 pp: dato erróneo (common.pair)
MIN_SESSIONS = 252                 # un par con menos sesiones comunes no se carga
LEVERAGE_CHANGE = "2020-04-01"     # ERY/ERX y DUST/NUGT pasan de 3x a 2x (universe.py)
SP, NDX = "S&P 500", "Nasdaq-100"
HIST_FROM = {SP: "1928-01-01", NDX: "1985-10-01"}      # historia larga sintética del pre-registro
HIST_L = (-3, -2, -1, 1, 2, 3)


@dataclass(frozen=True)
class Pair:
    inverse: str
    long: str | None           # alcista con el mismo |L|; None si |L| = 1 (el alcista es el propio 1x)
    L: int                     # |L|
    proxy: str                 # ETF 1x (rentabilidad total) que hace de índice
    index: str                 # índice, en castellano
    group: str                 # «índice amplio», «sector», «bonos», «materias primas»
    prereg: bool               # nombrado en el pre-registro
    since: str | None = None   # primer día con este apalancamiento
    until: str | None = None   # último día con este apalancamiento

    @property
    def key(self) -> str:
        base = self.inverse if self.long is None else f"{self.inverse}/{self.long}"
        if self.until:
            return f"{base} {self.L}x (hasta {pd.Timestamp(self.until).year})"
        if self.since:
            return f"{base} {self.L}x (desde {pd.Timestamp(self.since).year})"
        return base

    @property
    def funds(self) -> tuple[str, ...]:
        return (self.inverse,) if self.long is None else (self.inverse, self.long)


PAIRS: tuple[Pair, ...] = (
    # pre-registro
    Pair("SQQQ", "TQQQ", 3, "QQQ", "Nasdaq-100", "índice amplio", True),
    Pair("SPXU", "UPRO", 3, "SPY", "S&P 500", "índice amplio", True),
    Pair("SDS", "SSO", 2, "SPY", "S&P 500", "índice amplio", True),
    Pair("QID", "QLD", 2, "QQQ", "Nasdaq-100", "índice amplio", True),
    Pair("SH", None, 1, "SPY", "S&P 500", "índice amplio", True),
    Pair("PSQ", None, 1, "QQQ", "Nasdaq-100", "índice amplio", True),
    Pair("SOXS", "SOXL", 3, "SOXX", "semiconductores", "sector", True),
    Pair("TZA", "TNA", 3, "IWM", "Russell 2000", "índice amplio", True),
    Pair("FAZ", "FAS", 3, "XLF", "financieras", "sector", True),
    Pair("LABD", "LABU", 3, "XBI", "biotecnología", "sector", True),
    Pair("SDOW", "UDOW", 3, "DIA", "Dow Jones", "índice amplio", True),
    # ampliación («los que la caché y Yahoo permitan»)
    Pair("SPXS", "SPXL", 3, "SPY", "S&P 500", "índice amplio", False),
    Pair("SRTY", "URTY", 3, "IWM", "Russell 2000", "índice amplio", False),
    Pair("SMDD", "UMDD", 3, "MDY", "S&P MidCap 400", "índice amplio", False),
    Pair("TECS", "TECL", 3, "XLK", "tecnología", "sector", False),
    Pair("TWM", "UWM", 2, "IWM", "Russell 2000", "índice amplio", False),
    Pair("DXD", "DDM", 2, "DIA", "Dow Jones", "índice amplio", False),
    Pair("MZZ", "MVV", 2, "MDY", "S&P MidCap 400", "índice amplio", False),
    Pair("DOG", None, 1, "DIA", "Dow Jones", "índice amplio", False),
    Pair("RWM", None, 1, "IWM", "Russell 2000", "índice amplio", False),
    Pair("TMV", "TMF", 3, "TLT", "Tesoro a 20+ años", "bonos", False),
    Pair("TBT", "UBT", 2, "TLT", "Tesoro a 20+ años", "bonos", False),
    Pair("ERY", "ERX", 3, "XLE", "energía", "sector", False, until="2020-03-31"),
    Pair("ERY", "ERX", 2, "XLE", "energía", "sector", False, since=LEVERAGE_CHANGE),
    Pair("DUST", "NUGT", 3, "GDX", "mineras de oro", "materias primas", False, until="2020-03-31"),
    Pair("DUST", "NUGT", 2, "GDX", "mineras de oro", "materias primas", False, since=LEVERAGE_CHANGE),
)
BY_KEY = {p.key: p for p in PAIRS}


def prereg_pairs() -> tuple[Pair, ...]:
    return tuple(p for p in PAIRS if p.prereg)


def tickers(pairs=PAIRS) -> list[str]:
    """Todos los fondos y ETFs 1x de los pares, sin repetir."""
    out = []
    for p in pairs:
        for t in (*p.funds, p.proxy):
            if t not in out:
                out.append(t)
    return out


def inverse_tickers(pairs=PAIRS) -> list[str]:
    return list(dict.fromkeys(p.inverse for p in pairs))


def expense_ratio(ticker: str) -> float:
    """Gasto anual del folleto si el laboratorio lo tiene (universe.py); si no, NaN."""
    e = BY_TICKER.get(ticker)
    return float(e.expense_ratio) if e is not None else float("nan")


# ───────────────────────────── descargas ─────────────────────────────

def missing_prices(pairs=PAIRS) -> list[str]:
    """Tickers sin fichero diario en la caché (ni de IB ni de Yahoo)."""
    return [t for t in tickers(pairs)
            if not any(sources._cache_path(s, t, "1d").exists() for s in ("ib", "yf"))]


def download(pairs=PAIRS, borrow: bool = True) -> dict:
    """Descarga lo que falte a data/cache (nunca refresca lo que ya está): precios diarios de
    Yahoo (vix.etp_prices → sources.daily(source="yf")) y el JSON de iBorrowDesk de cada inverso
    (vix.download_borrow). Devuelve lo que se pidió y lo que sigue faltando."""
    need = missing_prices(pairs)
    if need:
        vix.etp_prices(need, end=None)
    still = missing_prices(pairs)
    out = {"precios_pedidos": need, "precios_sin_datos": still}
    if borrow:
        inv = inverse_tickers(pairs)
        want = [t for t in inv if not (vix.BORROW_DIR / f"ibd_{t}.json").exists()]
        if want:
            vix.download_borrow(want)
        out["préstamo_pedido"] = want
        out["préstamo_sin_datos"] = [t for t in inv if not (vix.BORROW_DIR / f"ibd_{t}.json").exists()]
    return out


# ───────────────────────────── pares reales ─────────────────────────────

_RET: dict[tuple[str, pd.Timestamp], pd.Series] = {}


def _returns(tks: tuple[str, ...], end=END) -> pd.DataFrame:
    """Rendimientos diarios (adj_close de la caché) hasta `end`, sin rellenar huecos. Solo lee
    los tickers que ya están en la caché: aquí no se descarga nada (para eso, download)."""
    out = {}
    for t in tks:
        k = (t, pd.Timestamp(end))
        if k not in _RET:
            if not any(sources._cache_path(s, t, "1d").exists() for s in ("ib", "yf")):
                continue
            _RET[k] = vix.etp_prices([t], "adj_close", end)[t].pct_change(fill_method=None)
        out[t] = _RET[k]
    return pd.DataFrame(out)


def pair_frame(p: Pair, end=END, clean: bool = True) -> pd.DataFrame | None:
    """Rendimientos diarios alineados del par: «inv», «long» (si lo hay), «u» (el 1x), «rf» (letra
    diaria, ^IRX) y «f» (financiación diaria, fed funds con un mes de retraso) en las sesiones en
    que todos existen, dentro del tramo del apalancamiento y hasta `end`. clean=True sustituye por
    L·u los días en que un fondo se aparta más de 20 pp de L·u (attrs["días_corregidos"])."""
    cols = {"inv": p.inverse, "u": p.proxy}
    if p.long is not None:
        cols["long"] = p.long
    R = _returns(tuple(cols.values()), end)
    if any(t not in R.columns for t in cols.values()):
        return None
    df = pd.DataFrame({k: R[t] for k, t in cols.items()}).dropna()
    if p.since:
        df = df[df.index >= pd.Timestamp(p.since)]
    if p.until:
        df = df[df.index <= pd.Timestamp(p.until)]
    if len(df) < MIN_SESSIONS:
        return None
    fixed = {}
    for k, sgn in (("inv", -1), ("long", 1)):
        if k not in df:
            continue
        Lk = sgn * p.L
        bad = (df[k] - Lk * df.u).abs() > BAD_DAY
        fixed[cols[k]] = [d.date() for d in df.index[bad]]
        if clean and bad.any():
            log.warning("%s: %d días con dato erróneo sustituidos por L·u", cols[k], int(bad.sum()))
            df.loc[bad, k] = Lk * df.loc[bad, "u"]
    df["rf"] = sources.risk_free_daily(df.index)
    df["f"] = history.funding_long(df.index)
    df.attrs.update(par=p.key, L=p.L, inverso=p.inverse, alcista=p.long, índice=p.proxy,
                    días_corregidos=fixed)
    return df


def load_pairs(pairs=PAIRS, end=END, clean: bool = True) -> dict[str, pd.DataFrame]:
    """{clave del par: pair_frame} de los pares con datos (los que no tienen, fuera)."""
    out = {}
    for p in pairs:
        df = pair_frame(p, end, clean)
        if df is None:
            log.warning("%s: sin datos suficientes en la caché", p.key)
            continue
        out[p.key] = df
    return out


def borrow_grid(tks=None) -> pd.DataFrame:
    """Rejilla de préstamo del pre-registro por ticker (% anual): mínimo, mediana y p90 de la tasa
    diaria de IBKR (iBorrowDesk, ~12 meses), con las fechas y los días; los que no tengan JSON en
    la caché no aparecen (vix.ibkr_borrow descarga los que falten)."""
    tks = list(tks or inverse_tickers())
    have = [t for t in tks if (vix.BORROW_DIR / f"ibd_{t}.json").exists()]
    if not have:
        return pd.DataFrame(columns=["mín", "mediana", "p90"])
    s = vix.borrow_summary(have)
    return s[["desde", "hasta", "días", "mín", "mediana", "p90", "máx", "disponibles_mediana", "disponibles_mín"]]


# ───────────────────────────── historia larga sintética ─────────────────────────────

def synthetic_history(D: dict | None = None) -> dict[str, pd.DataFrame]:
    """Fondos sintéticos de R22 (r22.synthetic: rf + L·(u − f) − |L|·s − gasto, acotado en −100%
    y sin fondo muerto) con el s y el gasto de cada ETF real (r22.fund_params; el 3x, los del
    laboratorio) sobre el índice largo del laboratorio: S&P 500 desde 1928 y Nasdaq-100 desde el
    1-10-1985 (antes es la reconstrucción). Columnas: L (−3, −2, −1, 1, 2, 3; 1 es el índice), u,
    rf y f, hasta END."""
    from letf_lab.research import r13_lrs as r13
    from letf_lab.research import r22_inverse as r22

    D = D or r13.load()
    P = r22.fund_params(D)
    out = {}
    for mkt, start in HIST_FROM.items():
        d = D[mkt]
        cols = {}
        for L in HIST_L:
            if L == 1:
                cols[L] = d["u"]
            elif L == 3:
                cols[L] = r22.synthetic(d, 3.0, d["spread"], d["er"])
            else:
                tk = r22.FUNDS[mkt][L]
                cols[L] = r22.synthetic(d, float(L), P[tk]["s"], P[tk]["er"])
        df = pd.DataFrame(cols)
        df["u"], df["rf"], df["f"] = d["u"], d["rf"], d["f"]
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= END)].dropna()
        df.attrs.update(mercado=mkt, s={L: (d["spread"] if L == 3 else (P[r22.FUNDS[mkt][L]]["s"] if L != 1 else 0.0))
                                        for L in HIST_L},
                        gasto={L: (d["er"] if L == 3 else (P[r22.FUNDS[mkt][L]]["er"] if L != 1 else 0.0))
                               for L in HIST_L})
        out[mkt] = df
    return out


def dataset_table(frames: dict[str, pd.DataFrame], pairs=PAIRS) -> pd.DataFrame:
    """Ficha de cada par cargado: fondos, |L|, índice, grupo, fechas, sesiones y días corregidos."""
    rows = []
    for p in pairs:
        df = frames.get(p.key)
        if df is None:
            continue
        fx = df.attrs.get("días_corregidos", {})
        rows.append({"par": p.key, "pre-registro": "sí" if p.prereg else "no", "|L|": p.L,
                     "índice (1x)": f"{p.index} ({p.proxy})", "grupo": p.group,
                     "desde": df.index[0].date(), "hasta": df.index[-1].date(), "sesiones": len(df),
                     "días corregidos": sum(len(v) for v in fx.values()),
                     "gasto inverso": expense_ratio(p.inverse),
                     "gasto alcista": expense_ratio(p.long) if p.long else np.nan})
    return pd.DataFrame(rows).set_index("par")
