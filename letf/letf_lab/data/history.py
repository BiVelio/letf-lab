"""Historia larga: S&P 500 diario desde 1928 y Nasdaq-100 reconstruido desde 1971.

Fuentes públicas, descargadas una vez y guardadas en data/cache/:
- ^GSPC diario (Yahoo, desde el 30-12-1927) + dividendos mensuales de Shiller
  (ie_data.xls) → rentabilidad total diaria del S&P 500.
- Tipo sin riesgo: NBER M1329AUSM193NNBR (1920–1934), FRED TB3MS (1934–) y
  ^IRX diario (1960–), en ese orden de preferencia inversa.
- ^IXIC diario (Nasdaq Composite, desde 1971) para reconstruir el Nasdaq-100
  antes de 1985 por imputación múltiple: regresión de los rendimientos diarios
  del NDX sobre los del Composite en el primer tramo común, y M series
  sintéticas con incertidumbre en los parámetros y residuos remuestreados.
"""

from __future__ import annotations

import io
import logging
import urllib.request

import numpy as np
import pandas as pd

from letf_lab.data import sources

log = logging.getLogger(__name__)

SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
NDX_FIT = ("1985-10-01", "1995-12-31")  # tramo común antes de que dominen los megacaps
NDX_DIV = 0.008
# Cierres de ^GSPC erróneos en Yahoo, marcados por la regla R1 de R14 (par de
# reversión > 4σ con el mercado CRSP plano), fijada antes de mirar resultados.
# Se sustituye el cierre por la media geométrica de sus vecinos, lo que conserva
# el movimiento de los dos días. R14 lista además pares revisados a mano que NO
# se corrigen aquí porque la regla no los marca.
GSPC_BAD_CLOSES = ["1935-04-16", "1935-08-16", "1936-09-16", "1961-04-17"]


def gspc_close(clean: bool = True) -> pd.Series:
    """Cierre diario de ^GSPC (Yahoo) desde 1927; clean=True corrige GSPC_BAD_CLOSES."""
    px = sources.panel(["^GSPC"], "close")["^GSPC"].copy()
    if clean:
        for d in pd.to_datetime(GSPC_BAD_CLOSES):
            i = px.index.get_loc(d)
            px.iloc[i] = float(np.sqrt(px.iloc[i - 1] * px.iloc[i + 1]))
    return px


def _cached(name: str, fetch) -> pd.DataFrame:
    path = sources.CACHE / f"hist_{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def fred(series: str) -> pd.Series:
    def fetch():
        raw = urllib.request.urlopen(FRED_URL.format(series), timeout=30).read()
        df = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True)
        return df.apply(pd.to_numeric, errors="coerce")
    return _cached(f"fred_{series}", fetch).iloc[:, 0].dropna()


def shiller_dividend_yield() -> pd.Series:
    """Rendimiento por dividendo anual mensual (D/P) del S&P Composite, 1871–."""
    def fetch():
        raw = urllib.request.urlopen(SHILLER_URL, timeout=60).read()
        x = pd.read_excel(io.BytesIO(raw), sheet_name="Data", header=None, skiprows=8)
        x = x.iloc[:, :3].dropna()
        x.columns = ["fecha", "P", "D"]
        x = x[pd.to_numeric(x.fecha, errors="coerce").notna()]
        f = x.fecha.astype(float)
        year = f.astype(int)
        month = np.rint((f - year) * 100).astype(int)
        idx = pd.to_datetime({"year": year, "month": month, "day": 1})
        return pd.DataFrame({"dy": (x.D.astype(float) / x.P.astype(float)).values}, index=idx)
    return _cached("shiller_dy", fetch)["dy"]


def _monthly_to_daily(monthly: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Serie mensual (media del mes, fechada el día 1) → diaria sin mirar el futuro:
    cada día usa la media del mes *anterior*, que ya se conocía."""
    m = monthly.copy()
    m.index = m.index + pd.offsets.MonthBegin(1)
    return m.reindex(index.union(m.index)).ffill().reindex(index)


def rf_long(index: pd.DatetimeIndex) -> pd.Series:
    """Tipo sin riesgo diario (fracción) desde 1920, empalmando NBER → FRED → ^IRX."""
    nber = fred("M1329AUSM193NNBR") / 100
    tb3 = fred("TB3MS") / 100
    monthly = pd.concat([nber[nber.index < tb3.index[0]], tb3]).sort_index()
    daily_m = _monthly_to_daily(monthly, index)
    irx = sources.risk_free_daily(index) * 252  # anual
    irx = irx.where(index >= pd.Timestamp("1960-01-04"))
    return (irx.fillna(daily_m) / 252).rename("rf")


def funding_long(index: pd.DatetimeIndex) -> pd.Series:
    """Tipo al que se financia el apalancamiento (fracción diaria).

    Los swaps y futuros se financian a tipo interbancario, no al T-bill:
    call money de Nueva York (NBER M13001USM156NNBR) hasta junio de 1954 y
    fed funds (FRED FEDFUNDS) después; ambos con un mes de retraso."""
    call = fred("M13001USM156NNBR") / 100
    ff = fred("FEDFUNDS") / 100
    monthly = pd.concat([call[call.index < ff.index[0]], ff]).sort_index()
    return (_monthly_to_daily(monthly, index) / 252).rename("f")


def sp500_long(clean: bool = True) -> pd.Series:
    """Rentabilidad total diaria del S&P 500 desde 1928.

    Hasta el inicio de ^SP500TR (1988) usa ^GSPC + D/P mensual de Shiller
    repartido por día hábil (método validado en 1988–2023 frente al índice
    oficial); después, la serie del laboratorio (^SP500TR y SPY). Antes del
    4-3-1957 ^GSPC es el índice de 90 valores que precedió al S&P 500. Antes de
    1952 el NYSE abría los sábados y Yahoo no tiene esas sesiones: el lunes suma
    dos (efecto medido en R14: +0,4 pp/año para un 3x en 1928–52).
    clean=False devuelve la serie sin corregir los cierres erróneos (para R14)."""
    from letf_lab import data
    from letf_lab.research.common import long_underlying

    px = gspc_close(clean)
    r = px.pct_change(fill_method=None).dropna()
    dy = shiller_dividend_yield()
    dy_d = dy.reindex(r.index.union(dy.index)).ffill().reindex(r.index) / 252
    tr_start = data.panel(["^SP500TR"], "adj_close")["^SP500TR"].first_valid_index()
    early = (r + dy_d)[r.index <= tr_start]
    modern = long_underlying("UPRO")
    return pd.concat([early, modern[modern.index > tr_start]]).rename("u")


def relative_drift_sd(ndx: pd.Series, ix: pd.Series, start: str, end: str) -> float:
    """Desviación típica anual del diferencial de rentabilidad log NDX − Composite."""
    both = pd.concat([ndx, ix], axis=1, keys=["ndx", "ix"]).dropna().loc[start:end]
    yearly = np.log1p(both).groupby(both.index.year).sum()
    yearly = yearly[both.groupby(both.index.year).size() > 200]
    return float((yearly.ndx - yearly.ix).std())


def ndx_backcast(M: int = 50, seed: int = 0) -> dict:
    """M reconstrucciones del Nasdaq-100 (rentabilidad total) desde 1971.

    Dinámica: regresión en el tramo común NDX_FIT, r_NDX = a + b·r_IXIC + e;
    cada imputación toma b de su distribución muestral y residuos remuestreados
    por bloques de 21 días. Media: la regresión no la identifica (validación en
    R12), así que se fija en la del Composite más el dividendo del NDX y un
    diferencial aleatorio δ ~ N(0, s²/años), con s la desviación típica anual
    del diferencial NDX − Composite observado desde 1986. Así la imputación
    múltiple recoge la incertidumbre principal, que es la media."""
    from letf_lab.research.common import long_underlying

    ix = sources.panel(["^IXIC"], "close")["^IXIC"].pct_change(fill_method=None).dropna()
    ndx = long_underlying("TQQQ")
    both = pd.concat([ndx, ix], axis=1, keys=["ndx", "ix"]).dropna()
    fit = both.loc[NDX_FIT[0]:NDX_FIT[1]]
    X = np.column_stack([np.ones(len(fit)), fit.ix.values])
    beta, *_ = np.linalg.lstsq(X, fit.ndx.values, rcond=None)
    res = fit.ndx.values - X @ beta
    XtX = np.linalg.inv(X.T @ X)
    cov = XtX @ (X.T * res ** 2) @ X @ XtX  # HC0
    pre = ix[ix.index < ndx.index[0]]
    years = len(pre) / 252
    s_rel = relative_drift_sd(ndx, ix, "1986-01-01", "2025-12-31")
    rng = np.random.default_rng(seed)
    draws, deltas = [], []
    n, blk = len(pre), 21
    m_pre = pre.mean()
    for _ in range(M):
        _, b = rng.multivariate_normal(beta, cov)
        starts = rng.integers(0, len(res) - blk, n // blk + 1)
        e = np.concatenate([res[s:s + blk] for s in starts])[:n]
        e = e - e.mean()
        delta = rng.normal(0, s_rel / np.sqrt(years))
        a = (1 - b) * m_pre + (NDX_DIV + delta) / 252
        synth = pd.Series(a + b * pre.values + e, index=pre.index)
        draws.append(pd.concat([synth, ndx]).rename("u"))
        deltas.append(delta)
    return {"draws": draws, "a": beta[0], "b": beta[1], "resid_sd": res.std(),
            "r2": 1 - res.var() / fit.ndx.var(), "fit": fit, "beta_cov": cov,
            "s_rel": s_rel, "deltas": np.array(deltas)}
