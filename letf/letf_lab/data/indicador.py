"""Datos del artículo 7 (indicador del apalancamiento óptimo): rango diario, bonos del Tesoro
sintéticos a vencimiento constante, fondos apalancados de bonos y el índice MOVE.

Pre-registro: letf/preregistro/articulo7_indicador.md (commits aafe958 y 02faefc, con la
ampliación a bonos de «Desviaciones»). Este módulo solo prepara datos: no contrasta nada.

DISCIPLINA DE LA MUESTRA RESERVADA (1-1-2019 → 25-9-2026)
- Toda función pública recorta en `end`, que por defecto es DEV_END (31-12-2018). end=None
  llega hasta END (25-9-2026): solo para la evaluación única de la prueba reservada, con los
  parámetros ya congelados, y para describir la calidad de los datos.
- Todo lo que aquí se estima (diferencial de los huecos de FRED, vencimiento del bono que imita
  a TLT e IEF, sobrecoste s de los fondos apalancados) usa como mucho datos hasta FIT_END =
  DEV_END, sea cual sea `end`: los parámetros no dependen de la prueba reservada.

1. OHLC diario de Yahoo (ohlc, range_variances, variance_inputs): ^GSPC, ^NDX, TLT e IEF.
   Precios ajustados por dividendos (adj_close/close aplicado a apertura, máximo y mínimo), así
   que el rendimiento de cierre a cierre y el nocturno son de rentabilidad total en los ETFs.
   Marcas por sesión: falta algún precio; apertura, máximo o mínimo iguales al cierre; apertura
   igual al cierre anterior (no hay apertura real: en ^GSPC, entre el 19% y el 94% de las
   sesiones de cada año de 1982–1995, el 95–98% en 1996–2005, el 43% en 2006, el 16% en 2007 y
   entre el 1% y el 26% en 2008–2013; en ^NDX, entre el 77% y el 99% de cada año hasta 1998 salvo
   1988 (34%), el 54% en 1999 y el 15% en 2000); sin rango (máximo = mínimo); incoherente (máximo <
   max(apertura, cierre) o mínimo > min(apertura, cierre)); rango teórico (^GSPC antes del
   21-4-1982, ver RANGE_START). open_ok: apertura real (ni falta ni es el cierre anterior ni
   coincide con máximo, mínimo y cierre a la vez). El rendimiento r de ^GSPC es de precio (sin
   dividendos); la rentabilidad total del S&P 500 es history.sp500_long.
   Estimadores diarios (en log-rendimiento², sin anualizar), con o = ln(A_t/C_{t−1}),
   c = ln(C_t/A_t), u = ln(M_t/A_t), d = ln(m_t/A_t):
     Parkinson  PK = ln(M/m)² / (4 ln 2)
     Garman–Klass GK = ½ ln(M/m)² − (2 ln 2 − 1)·c²
     Rogers–Satchell RS = u(u − c) + d(d − c)
     Yang–Zhang diario YZ = o² + k·c² + (1 − k)·RS, k = 0,34 / (1,34 + (n+1)/(n−1)), n = YZ_N
   (PK, GK y RS miden solo el tramo apertura → cierre; YZ suma la noche.) yang_zhang() da el
   estimador de ventana de Yang y Zhang (2000) con varianzas muestrales. Semivarianzas de cierre
   a cierre (Barndorff-Nielsen, Kinnebrock y Shephard, 2010, con un rendimiento por día):
   rv_neg = r²·1{r<0} y rv_pos = r²·1{r>0}. Los estimadores de rango son NaN donde range_ok es
   falso; con apertura = cierre anterior, o = 0 y RS abarca de cierre a cierre.

2. Bonos del Tesoro sintéticos (cmt_curve, cmt_bond, bond_underlying): rendimientos diarios de
   FRED (DGS10, DGS20, DGS30; DGS5 y DGS7 solo para interpolar la curva por debajo de 10 años).
   Huecos cubiertos con el vencimiento más cercano más el diferencial medio de las GAP_WINDOW
   sesiones contiguas anteriores (posteriores si no las hay), como en R24, y declarados en
   gap_table(): DGS20 del 2-1-1987 al 30-9-1993 (con DGS30), DGS30 antes del 15-2-1977 (con
   DGS20) y DGS30 del 19-2-2002 al 8-2-2006. En ese último tramo FRED trae valores, pero el
   Tesoro no emitía el bono a 30 años y el diferencial 30 − 20 salta unos 14 pb en los dos
   extremos: se tratan como hueco (según FRED, en ese tramo el Tesoro publicaba un factor para
   estimar el 30 años a partir del 20 años; sin verificar).
   Bono a la par de M años (cupón semestral): se compra al cierre de t−1 a la par (cupón
   c = y_{t−1}(M)) y al cierre de t vence en M − Δ (Δ = días naturales / 365,25) y se valora
   con la curva de t interpolada linealmente en M − Δ. Rentabilidad = cupón devengado c·Δ
   (carry) + rodadura (el precio con la curva de t−1 en M − Δ, menos 1) + cambio de la curva.
   Es par_bond_returns de R24 con el envejecimiento del bono (la rodadura), que R24 no tiene.
   En las sesiones de bolsa sin mercado de bonos la curva se mantiene (solo carry y rodadura).
   Ex ante, al cierre de t: carry_ann = y_t(M) − f_t (f: financiación interbancaria del
   laboratorio, history.funding_long) y roll_ann = precio a un año vista sobre la curva de hoy
   (vencimiento M − 1, cupón y_t(M)) − 1; prior_ann = carry_ann + roll_ann es el prior de la
   deriva del pre-registro para el bono.
   Vencimiento del bono que imita a cada ETF, elegido con datos hasta FIT_END por el menor error
   de seguimiento semanal frente al ETF en una rejilla fijada de antemano (MATURITY_GRID: TLT →
   20, 25 o 30 años; IEF → 7, 8,5 o 10). bond_underlying empalma el sintético con el ETF real
   desde su primer rendimiento (31-7-2002).

3. Fondos apalancados de bonos (BOND_FUNDS, bond_fund, make_bond): UBT, TMF (2x y 3x del largo),
   UST, TYD (2x y 3x del medio) y, solo como referencia, TBT, TMV, TTT y PST. Sintético =
   rf + L·(u − f) − |L|·s − gasto (r24.lev_fund, la fórmula de r12.make), con s calibrado para
   que la media diaria del sintético iguale a la del fondo real en su vida HASTA FIT_END
   (r24.calibrate_spread); en TMF, TYD y TMV, desde el 1-1-2011 (CALIB_DIREXION: hasta finales
   de 2010 seguían a otro índice). Sintético antes del primer rendimiento real y real después. Los
   rendimientos reales pasan la regla de common.pair: un día que se aparta más de 20 pp de
   L·u se sustituye por L·u y se registra.

   Precios viejos: etf_flags marca las sesiones sin volumen o con el cierre ajustado igual al
   anterior (TYD cotizó sin volumen en cientos de sesiones de 2013–2018; su rendimiento diario
   real concentra en un día movimientos de varios) y fund_flags, además, los días en que el
   fondo se aparta más de 5 pp de L·u (TYD: −12,7% el 26-11-2014 y +15,8% el 1-12-2014 con IEF
   casi plano, dos cierres de muy poco volumen). Solo se marcan; no se corrigen.

4. Índice MOVE (move): Yahoo Finance, símbolo ^MOVE, desde el 12-11-2002 (MOVE_SOURCE). FRED
   no lo publica (licencia de ICE) y no hay otra historia diaria pública descargable sin
   registro; antes de 2002-11 no hay MOVE.

5. Fondos apalancados de acciones del artículo (equity_fund, make_equity): SSO y UPRO (2x y 3x
   del S&P 500) y QLD y TQQQ (2x y 3x del Nasdaq-100), con la misma fórmula y el mismo s
   calibrado HASTA FIT_END. r12.make calibra s con toda la vida del fondo real (hasta 2026), así
   que sus fondos no sirven para el artículo 7 sin filtrar la prueba reservada; estos sí.
   Subyacente: history.sp500_long (S&P 500, rentabilidad total desde 1928) y
   common.long_underlying("TQQQ") (^NDX con un dividendo de 0,8% hasta QQQ; desde 1985-10).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from letf_lab.data import history, sources

log = logging.getLogger(__name__)

# ───────────────────────────── fechas del pre-registro ─────────────────────────────

DEV_END = pd.Timestamp("2018-12-31")      # fin de la muestra de estimación y desarrollo
TEST_START = pd.Timestamp("2019-01-01")   # prueba reservada: TEST_START → END, se evalúa una vez
END = pd.Timestamp("2026-09-25")          # último día de todas las ventanas
FIT_END = DEV_END                         # tope de los datos con que se estima cualquier parámetro


def horizon(end=DEV_END) -> pd.Timestamp:
    """Fecha de corte efectiva: `end` (por defecto DEV_END); None → END; nunca después de END."""
    e = END if end is None else pd.Timestamp(end)
    return min(e, END)


def cut(obj, end=DEV_END):
    """Recorta una serie o tabla con índice de fechas en horizon(end)."""
    return obj[obj.index <= horizon(end)]


def _fit(obj):
    """Recorte de los datos con que se estima algo: nunca después de FIT_END."""
    return obj[obj.index <= FIT_END]


@lru_cache(maxsize=None)
def _sessions_all() -> pd.DatetimeIndex:
    idx = sources.daily(["^GSPC"], source="yf")["^GSPC"].index
    return idx[idx <= END]


def sessions(start="1962-01-02", end=DEV_END) -> pd.DatetimeIndex:
    """Sesiones de la bolsa de Nueva York (calendario de ^GSPC en Yahoo) entre start y end."""
    s = _sessions_all()
    return s[(s >= pd.Timestamp(start)) & (s <= horizon(end))]


# ───────────────────────────── 1. OHLC y varianza de rango ─────────────────────────────

OHLC_TICKERS = ("^GSPC", "^NDX", "TLT", "IEF")
# Primer día con máximo y mínimo REALES. En ^GSPC, Yahoo trae máximo y mínimo desde el
# 2-1-1962, pero hasta el 20-4-1982 son los «teóricos» (agregan el máximo y el mínimo de cada
# valor, no el del índice): Parkinson / r² vale 1,4–4,5 al año en 1962–81 y 0,5–0,9 desde mayo
# de 1982. El 21-4-1982 empezaron los futuros del S&P 500 en el CME; el 20-4-1982 ya trae una
# apertura distinta del cierre anterior, pero un rango del 2,0%, típico del tramo teórico.
RANGE_START = {"^GSPC": pd.Timestamp("1982-04-21"), "^NDX": pd.Timestamp("1985-10-01"),
               "TLT": pd.Timestamp("2002-07-30"), "IEF": pd.Timestamp("2002-07-30")}
HL_FROM = {"^GSPC": pd.Timestamp("1962-01-02")}   # antes, apertura = máximo = mínimo = cierre
YZ_N = 21                                          # ventana de k de Yang–Zhang (el horizonte principal)
_REL = 1e-7                                        # tolerancia relativa de las igualdades de precios


def yz_k(n: int = YZ_N) -> float:
    """Peso k de Yang y Zhang (2000) que minimiza la varianza del estimador con n sesiones."""
    return 0.34 / (1.34 + (n + 1) / (n - 1))


@lru_cache(maxsize=None)
def _ohlc_raw(ticker: str) -> pd.DataFrame:
    df = sources.daily([ticker], source="yf")[ticker]
    return df[df.index <= END].copy()


def _eq(a: pd.Series, b: pd.Series) -> pd.Series:
    return ((a - b).abs() <= _REL * b.abs()) & a.notna() & b.notna()


def ohlc_flags(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
               range_start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Marcas de calidad por sesión sobre precios SIN ajustar (las igualdades son de Yahoo)."""
    missing = pd.concat([o, h, l, c], axis=1).isna().any(axis=1) | (pd.concat([o, h, l, c], axis=1) <= 0).any(axis=1)
    hi, lo = np.maximum(o, c), np.minimum(o, c)
    incons = (h < hi * (1 - _REL)) | (l > lo * (1 + _REL)) | (h < l)
    theo = pd.Series(False, index=c.index) if range_start is None else pd.Series(c.index < range_start, index=c.index)
    f = pd.DataFrame({
        "flag_missing": missing,
        "flag_open_eq_close": _eq(o, c),
        "flag_high_eq_close": _eq(h, c),
        "flag_low_eq_close": _eq(l, c),
        "flag_open_prev_close": _eq(o, c.shift(1)),
        "flag_no_range": _eq(h, l),
        "flag_inconsistent": incons & ~missing,
        "flag_theoretical": theo,
    })
    f["flag_any_eq_close"] = f[["flag_open_eq_close", "flag_high_eq_close", "flag_low_eq_close"]].any(axis=1)
    f["range_ok"] = ~(f.flag_missing | f.flag_no_range | f.flag_inconsistent | f.flag_theoretical)
    flat = f.flag_open_eq_close & f.flag_high_eq_close & f.flag_low_eq_close
    f["open_ok"] = ~(f.flag_missing | f.flag_open_prev_close | flat) & c.shift(1).notna()
    return f


def ohlc(ticker: str, end=DEV_END) -> pd.DataFrame:
    """OHLC diario de Yahoo ajustado por dividendos (adj_close/close) y sus marcas de calidad.

    Columnas: open, high, low, close (ajustados), close_raw, volume, flag_*, range_ok (rango
    utilizable: los estimadores de rango solo existen ahí) y open_ok (apertura real)."""
    raw = _ohlc_raw(ticker)
    a = raw.adj_close / raw.close
    px = pd.DataFrame({"open": raw.open * a, "high": raw.high * a, "low": raw.low * a,
                       "close": raw.adj_close, "close_raw": raw.close, "volume": raw.volume})
    flags = ohlc_flags(raw.open, raw.high, raw.low, raw.close, RANGE_START.get(ticker))
    return cut(pd.concat([px, flags], axis=1), end)


def range_variances(x: pd.DataFrame, n: int = YZ_N) -> pd.DataFrame:
    """Varianzas diarias (log-rendimiento², sin anualizar) a partir de la salida de ohlc().

    r: rendimiento logarítmico de cierre a cierre; r2 = r²; rv_neg, rv_pos: semivarianzas;
    on2: nocturno² (apertura sobre el cierre anterior); oc2: (apertura → cierre)²; pk, gk, rs:
    Parkinson, Garman–Klass y Rogers–Satchell (tramo apertura → cierre); yz: Yang–Zhang diario
    (noche + día, k de yz_k(n)). Los de rango son NaN donde range_ok es falso."""
    lo, lh, ll, lc = (np.log(x[k].where(x[k] > 0)) for k in ("open", "high", "low", "close"))
    r = lc.diff()
    on = lo - lc.shift(1)
    oc = lc - lo
    u, d, hl = lh - lo, ll - lo, lh - ll
    ok = x["range_ok"].astype(bool)
    pk = hl ** 2 / (4 * np.log(2))
    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * oc ** 2
    rs = u * (u - oc) + d * (d - oc)
    k = yz_k(n)
    out = pd.DataFrame({
        "r": r, "r2": r ** 2,
        "rv_neg": r ** 2 * (r < 0), "rv_pos": r ** 2 * (r > 0),
        "on2": (on ** 2).where(~x["flag_missing"].astype(bool)),
        "oc2": (oc ** 2).where(ok),
        "pk": pk.where(ok), "gk": gk.where(ok), "rs": rs.where(ok),
        "yz": (on ** 2 + k * oc ** 2 + (1 - k) * rs).where(ok & on.notna()),
    })
    out.loc[r.isna(), ["rv_neg", "rv_pos"]] = np.nan
    return out


def yang_zhang(x: pd.DataFrame, n: int = YZ_N) -> pd.Series:
    """Estimador de ventana de Yang y Zhang (2000) con n sesiones (varianza diaria):
    V_noche + k·V_día + (1 − k)·media(RS), con varianzas muestrales; NaN si falta alguna sesión
    válida en la ventana."""
    lo, lh, ll, lc = (np.log(x[k].where(x[k] > 0)) for k in ("open", "high", "low", "close"))
    ok = x["range_ok"].astype(bool)
    on = (lo - lc.shift(1)).where(ok)
    oc = (lc - lo).where(ok)
    u, d = lh - lo, ll - lo
    rs = (u * (u - oc) + d * (d - oc)).where(ok)
    k = yz_k(n)
    v = on.rolling(n, min_periods=n).var() + k * oc.rolling(n, min_periods=n).var() \
        + (1 - k) * rs.rolling(n, min_periods=n).mean()
    return v.rename(f"yz{n}")


def variance_inputs(ticker: str, end=DEV_END) -> pd.DataFrame:
    """ohlc() + range_variances() de un índice o ETF de OHLC_TICKERS, recortado en `end`."""
    x = ohlc(ticker, end=None)
    return cut(pd.concat([x, range_variances(x)], axis=1), end)


# ───────────────────────────── 2. bonos del Tesoro sintéticos ─────────────────────────────

FRED_TENORS = {5.0: "DGS5", 7.0: "DGS7", 10.0: "DGS10", 20.0: "DGS20", 30.0: "DGS30"}
BOND_START = pd.Timestamp("1962-01-02")
GAP_WINDOW = 250
NEAREST = {20.0: 30.0, 30.0: 20.0}         # vencimiento con que se cubre cada hueco (20: empate 10/30 → 30, como R24)
DGS30_ESTIMATED = (pd.Timestamp("2002-02-19"), pd.Timestamp("2006-02-08"))
BOND_ETF = {"largo": "TLT", "medio": "IEF"}
MATURITY_GRID = {"largo": (20.0, 25.0, 30.0), "medio": (7.0, 8.5, 10.0)}
KIND_ES = {"largo": "bono largo (ICE 20+ años; TLT)", "medio": "bono medio (ICE 7–10 años; IEF)"}


@lru_cache(maxsize=None)
def _fred_raw() -> pd.DataFrame:
    """Rendimientos de FRED (fracción) en los días del mercado de bonos (los que tienen DGS10)."""
    Y = pd.concat({m: history.fred(s) for m, s in FRED_TENORS.items()}, axis=1) / 100.0
    Y = Y[Y[10.0].notna()]
    return Y[(Y.index >= BOND_START) & (Y.index <= END)]


def fill_gaps(Y: pd.DataFrame, window: int = GAP_WINDOW) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cubre los huecos de los plazos de NEAREST con el plazo más cercano más el diferencial
    medio de las `window` sesiones comunes anteriores al hueco (posteriores si no hay), con datos
    hasta FIT_END como mucho. Los valores de DGS30 en DGS30_ESTIMATED se tratan como hueco.

    Devuelve (curva cubierta, marcas de relleno por plazo, tabla de huecos)."""
    Y = Y.copy()
    raw = Y.copy()
    a, b = DGS30_ESTIMATED
    raw.loc[(raw.index >= a) & (raw.index <= b), 30.0] = np.nan
    filled = pd.DataFrame(False, index=Y.index, columns=Y.columns)
    rows = []
    for T, N in NEAREST.items():
        miss = raw[T].isna()
        run_id = (miss != miss.shift()).cumsum()[miss]
        common = _fit(raw[[T, N]].dropna())
        for _, days in run_id.groupby(run_id):
            d0, d1 = days.index[0], days.index[-1]
            before = common[common.index < d0].tail(window)
            side = "anteriores"
            if len(before) < window:
                before, side = common[common.index > d1].head(window), "posteriores"
            spread = float((before[T] - before[N]).mean())
            src = raw.loc[d0:d1, N]
            ok = src.notna()
            Y.loc[src.index[ok], T] = src[ok] + spread
            Y.loc[src.index[~ok], T] = np.nan
            filled.loc[src.index[ok], T] = True
            rows.append({"plazo": f"{T:g} años ({FRED_TENORS[T]})", "desde": d0.date(), "hasta": d1.date(),
                         "sesiones": int(ok.sum()), "cubierto con": f"{N:g} años ({FRED_TENORS[N]})",
                         "diferencial (pb)": 1e4 * spread,
                         "ventana del diferencial": f"{len(before)} sesiones {side} "
                                                    f"({before.index[0].date()} → {before.index[-1].date()})",
                         "nota": ("FRED trae valores, tratados como hueco (el Tesoro no emitía el 30 años)"
                                  if T == 30.0 and d0 >= a and d1 <= b else "")})
    return Y, filled, pd.DataFrame(rows)


@lru_cache(maxsize=None)
def _curve_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return fill_gaps(_fred_raw())


def cmt_curve(end=DEV_END) -> pd.DataFrame:
    """Curva de FRED (fracción, días del mercado de bonos) con los huecos de 20 y 30 años cubiertos
    y columnas filled_20/filled_30 que marcan los días cubiertos."""
    Y, F, _ = _curve_all()
    out = Y.copy()
    out.columns = [f"y{m:g}" for m in Y.columns]
    for m in NEAREST:
        out[f"filled_{m:g}"] = F[m]
    return cut(out, end)


def gap_table() -> pd.DataFrame:
    """Huecos de FRED cubiertos: plazo, tramo, sesiones, plazo de relleno y diferencial."""
    return _curve_all()[2].copy()


def curve_yield(C: pd.DataFrame | np.ndarray, M, tenors=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rendimiento al vencimiento M (escalar o uno por fila) por interpolación lineal entre los
    plazos disponibles cada día; sin extrapolar (NaN fuera de [5, 30]). Devuelve (y, plazo
    inferior, plazo superior) para poder marcar los días que usan un plazo cubierto."""
    if isinstance(C, pd.DataFrame):
        tenors = np.array(C.columns, float) if tenors is None else np.asarray(tenors, float)
        V = C.to_numpy(float)
    else:
        V = np.asarray(C, float)
        tenors = np.asarray(tenors, float)
    n = V.shape[0]
    M = np.broadcast_to(np.asarray(M, float), (n,)).astype(float)
    Tm = np.where(np.isnan(V), np.nan, tenors[None, :])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # filas sin ningún plazo válido: NaN
        lo = np.nanmax(np.where(Tm <= M[:, None] + 1e-12, Tm, np.nan), axis=1)
        hi = np.nanmin(np.where(Tm >= M[:, None] - 1e-12, Tm, np.nan), axis=1)
    y = np.full(n, np.nan)
    good = ~np.isnan(lo) & ~np.isnan(hi) & ~np.isnan(M)
    if good.any():
        rows = np.nonzero(good)[0]
        il = np.searchsorted(tenors, lo[good])
        ih = np.searchsorted(tenors, hi[good])
        yl, yh = V[rows, il], V[rows, ih]
        w = np.where(hi[good] > lo[good], (M[good] - lo[good]) / np.where(hi[good] > lo[good], hi[good] - lo[good], 1.0), 0.0)
        y[rows] = yl + w * (yh - yl)
    return y, lo, hi


def par_price(y, c, T):
    """Precio por 1 de nominal de un bono con cupón anual c pagado por semestres, T años de vida
    y rendimiento y (capitalización semestral); vida fraccionaria sin cupón corrido."""
    y, c, T = (np.asarray(v, float) for v in (y, c, T))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (1 + y / 2) ** (-2 * T)
        return np.where(np.abs(y) > 1e-10, c / y * (1 - v) + v, 1 + c * T)


def par_duration(y, M):
    """Duración modificada de un bono a la par de M años: (1 − (1 + y/2)^(−2M)) / y."""
    y, M = np.asarray(y, float), np.asarray(M, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(y) > 1e-10, (1 - (1 + y / 2) ** (-2 * M)) / y, M)


def bond_path(C: pd.DataFrame, M: float, index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Bono a la par de M años sobre la curva C (columnas = plazos en años) en las fechas `index`
    (por defecto las de C; la curva se mantiene en las fechas sin dato).

    ret = carry + roll + curve: carry = c·Δ (c = y_{t−1}(M)); roll = P(y_{t−1}(M − Δ), c, M − Δ) − 1;
    curve = P(y_t(M − Δ), c, M − Δ) − P(y_{t−1}(M − Δ), c, M − Δ)."""
    index = C.index if index is None else index
    Cs = C.reindex(C.index.union(index)).ffill().reindex(index)
    tenors = np.array(Cs.columns, float)
    V = Cs.to_numpy(float)
    Vp = np.vstack([np.full((1, V.shape[1]), np.nan), V[:-1]])
    dt = index.to_series().diff().dt.days.to_numpy(float) / 365.25
    y, lo, hi = curve_yield(V, M, tenors)
    c = np.concatenate([[np.nan], y[:-1]])
    T = M - dt
    p_now = par_price(curve_yield(V, T, tenors)[0], c, T)
    p_old = par_price(curve_yield(Vp, T, tenors)[0], c, T)
    y1 = curve_yield(V, M - 1.0, tenors)[0]
    out = pd.DataFrame({
        "y": y,
        "carry": c * dt, "roll": p_old - 1, "curve": p_now - p_old,
        "dur": par_duration(y, M),
        "roll_ann": par_price(y1, y, M - 1.0) - 1,
        "lo": lo, "hi": hi,
    }, index=index)
    out["ret"] = out.carry + out.roll + out.curve
    out["dy_bp"] = 1e4 * out.y.diff()
    out["stale"] = ~index.isin(C.index)
    return out


@lru_cache(maxsize=None)
def _cmt_bond_all(M: float) -> pd.DataFrame:
    Y, F, _ = _curve_all()
    idx = sessions(BOND_START, end=None)
    out = bond_path(Y, M, idx)
    Fs = F.reindex(F.index.union(idx)).ffill().reindex(idx).fillna(False).astype(bool)
    fl = np.zeros(len(idx), bool)
    for m in NEAREST:
        fl |= Fs[m].to_numpy() & ((out.lo.to_numpy() == m) | (out.hi.to_numpy() == m))
    out["filled"] = fl
    f = history.funding_long(idx) * 252
    out["f_ann"] = f
    out["carry_ann"] = out.y - f
    out["prior_ann"] = out.carry_ann + out.roll_ann
    return out.drop(columns=["lo", "hi"])


def cmt_bond(M: float, end=DEV_END) -> pd.DataFrame:
    """Bono sintético a la par de M años (5 ≤ M ≤ 30) en sesiones de NYSE desde 1962.

    Columnas: ret (rentabilidad total diaria) = carry + roll + curve; y (rendimiento a M,
    fracción anual); dy_bp (cambio diario del rendimiento, pb); dur (duración modificada);
    ex ante al cierre de t, en fracción anual: f_ann (financiación), carry_ann = y − f_ann,
    roll_ann (rodadura a un año sobre la curva de hoy) y prior_ann = carry_ann + roll_ann;
    stale (bolsa abierta y mercado de bonos cerrado: curva mantenida); filled (usa un plazo
    cubierto de gap_table)."""
    return cut(_cmt_bond_all(float(M)).copy(), end)


@lru_cache(maxsize=None)
def _etf_returns(ticker: str) -> pd.Series:
    """Rendimiento total diario de un ETF (adj_close de Yahoo) hasta END."""
    px = _ohlc_raw(ticker)["adj_close"]
    return px.pct_change(fill_method=None).dropna().rename(ticker)


def weekly(r: pd.Series) -> pd.Series:
    """Rendimientos semanales compuestos (miércoles a miércoles), sin semanas vacías."""
    r = r.dropna()
    g = np.log1p(r).resample("W-WED")
    return np.expm1(g.sum()[g.count() > 0])


def tracking_stats(real: pd.Series, synth: pd.Series) -> dict:
    """Seguimiento del sintético frente al real en sus días comunes: correlación diaria y
    semanal, beta semanal (real sobre sintético), error de seguimiento anual (desviación típica
    de la diferencia, diaria × √252 y semanal × √52), crecimiento logarítmico anual de cada uno y
    su diferencia Δg = g(real) − g(sintético)."""
    j = pd.concat([real, synth], axis=1, keys=["real", "sint"], sort=True).dropna()
    w = pd.concat([weekly(j.real), weekly(j.sint)], axis=1, keys=["real", "sint"], sort=True).dropna()
    g = np.log1p(j).mean() * 252
    return {"desde": j.index[0].date(), "hasta": j.index[-1].date(), "sesiones": len(j),
            "corr_diaria": j.real.corr(j.sint), "corr_semanal": w.real.corr(w.sint),
            "beta_semanal": float(np.cov(w.real, w.sint)[0, 1] / w.sint.var()),
            "vol_real": j.real.std() * np.sqrt(252), "vol_sint": j.sint.std() * np.sqrt(252),
            "TE_diario": (j.real - j.sint).std() * np.sqrt(252),
            "TE_semanal": (w.real - w.sint).std() * np.sqrt(52),
            "g_real": g.real, "g_sint": g.sint, "Δg": g.real - g.sint}


@lru_cache(maxsize=None)
def maturity_table(kind: str) -> pd.DataFrame:
    """Rejilla MATURITY_GRID[kind] frente al ETF con datos hasta FIT_END (tracking_stats)."""
    real = _fit(_etf_returns(BOND_ETF[kind]))
    rows = {M: tracking_stats(real, _fit(_cmt_bond_all(M)["ret"])) for M in MATURITY_GRID[kind]}
    t = pd.DataFrame(rows).T
    t.index.name = "vencimiento (años)"
    return t


def maturity(kind: str) -> float:
    """Vencimiento del bono sintético que imita al ETF de `kind` («largo» → TLT, «medio» → IEF):
    el de menor error de seguimiento semanal hasta FIT_END."""
    t = maturity_table(kind)
    return float(pd.to_numeric(t["TE_semanal"]).idxmin())


def bond_underlying(kind: str, end=DEV_END, synthetic: bool = False) -> pd.Series:
    """Rentabilidad total diaria del bono de `kind` en sesiones de NYSE desde 1962: sintético
    (vencimiento maturity(kind)) hasta el primer rendimiento del ETF y ETF real después
    (synthetic=True: solo el sintético)."""
    syn = _cmt_bond_all(maturity(kind))["ret"]
    if synthetic:
        return cut(syn.rename(f"{kind}_sint"), end)
    real = _etf_returns(BOND_ETF[kind])
    u = pd.concat([syn[syn.index < real.index[0]], real]).sort_index()
    u = u[~u.index.duplicated()].reindex(syn.index)
    return cut(u.rename("u"), end)


def bond_variance_inputs(kind: str, end=DEV_END) -> pd.DataFrame:
    """Entradas de varianza del bono de `kind`: r = log(1 + u) de bond_underlying (sintético antes
    del ETF, real después), r2, rv_neg, rv_pos, dy_bp y dur del sintético y, desde 2002, los
    estimadores de rango del ETF (pk, gk, rs, yz, on2, oc2, range_ok; NaN antes). source dice de
    dónde sale r en cada sesión."""
    u = bond_underlying(kind, end=None)
    etf = BOND_ETF[kind]
    r = np.log1p(u)
    out = pd.DataFrame({"r": r, "r2": r ** 2, "rv_neg": r ** 2 * (r < 0), "rv_pos": r ** 2 * (r > 0)})
    out.loc[r.isna(), ["rv_neg", "rv_pos"]] = np.nan
    syn = _cmt_bond_all(maturity(kind))
    out["dy_bp"] = syn["dy_bp"]
    out["dur"] = syn["dur"]
    rv = variance_inputs(etf, end=None)
    for col in ("on2", "oc2", "pk", "gk", "rs", "yz"):
        out[col] = rv[col].reindex(out.index)
    out["range_ok"] = rv["range_ok"].reindex(out.index).fillna(False).astype(bool)
    first = _etf_returns(etf).index[0]
    out["source"] = np.where(out.index < first, "sintético", etf)
    return cut(out, end)


# ───────────────────────────── 3. fondos apalancados de bonos ─────────────────────────────

@dataclass(frozen=True)
class BondFund:
    ticker: str
    leverage: float
    kind: str          # "largo" (ICE 20+ años, como TLT) o "medio" (ICE 7–10 años, como IEF)
    er: float          # gasto anual de folleto aproximado (sin verificar; s absorbe la diferencia)
    issuer: str
    role: str          # "decisión" (entra en el indicador) o "referencia"
    calib_from: str | None = None   # primer día con que se calibra s (None: desde el lanzamiento)


# Los tres de Direxion de 2009 no siguieron a un índice como el de TLT o IEF hasta finales de 2010:
# su beta diaria frente a 3 × TLT o 3 × IEF (en valor absoluto) fue de 3,0–3,6 en cada bimestre de
# mayo de 2009 a octubre de 2010 y de 2,85–3,03 desde noviembre de 2010 (en su lanzamiento se llamaban «30-Year» y «10-Year Treasury»: bono
# corriente a 30 y 10 años; fecha exacta del cambio de índice sin verificar). Su s se calibra desde
# CALIB_DIREXION; el fondo real se usa igual desde su lanzamiento (es lo que se podía comprar).
CALIB_DIREXION = "2011-01-01"

BOND_FUNDS: dict[str, BondFund] = {f.ticker: f for f in (
    BondFund("UBT", 2.0, "largo", 0.0095, "ProShares", "decisión"),
    BondFund("TMF", 3.0, "largo", 0.0091, "Direxion", "decisión", CALIB_DIREXION),   # gasto como universe.py
    BondFund("UST", 2.0, "medio", 0.0095, "ProShares", "decisión"),
    BondFund("TYD", 3.0, "medio", 0.0109, "Direxion", "decisión", CALIB_DIREXION),
    BondFund("TBT", -2.0, "largo", 0.0090, "ProShares", "referencia"),
    BondFund("TMV", -3.0, "largo", 0.0092, "Direxion", "referencia", CALIB_DIREXION),  # gasto como universe.py
    BondFund("TTT", -3.0, "largo", 0.0095, "ProShares", "referencia"),
    BondFund("PST", -2.0, "medio", 0.0095, "ProShares", "referencia"),
)}
DECISION = {"largo": {1: "TLT", 2: "UBT", 3: "TMF"}, "medio": {1: "IEF", 2: "UST", 3: "TYD"}}
BAD_DAY = 0.20                                     # regla de common.pair


@lru_cache(maxsize=None)
def _rates() -> tuple[pd.Series, pd.Series]:
    idx = sessions(BOND_START, end=None)
    return history.rf_long(idx), history.funding_long(idx)


def real_fund(ticker: str, end=DEV_END) -> pd.Series:
    """Rendimiento diario del fondo real (adj_close de Yahoo). Un día que se aparta más de 20 pp
    de L·u (u: ETF subyacente) se sustituye por L·u (regla de common.pair); attrs["bad_days"]."""
    e = BOND_FUNDS[ticker]
    r = _etf_returns(ticker).copy()
    u = _etf_returns(BOND_ETF[e.kind]).reindex(r.index)
    bad = (r - e.leverage * u).abs() > BAD_DAY
    if bad.any():
        log.warning("%s: %d días con dato erróneo sustituidos: %s", ticker, bad.sum(), list(r.index[bad].date))
        r[bad] = e.leverage * u[bad]
    out = cut(r, end)
    out.attrs["bad_days"] = list(r.index[bad])
    return out


def etf_flags(ticker: str, end=DEV_END) -> pd.DataFrame:
    """Marcas de precio viejo de un ETF o fondo de Yahoo: zero_volume (sesión sin volumen) y
    unchanged (cierre ajustado igual al anterior: rendimiento exactamente cero)."""
    raw = _ohlc_raw(ticker)
    out = pd.DataFrame({"zero_volume": raw["volume"].fillna(0) <= 0,
                        "unchanged": _eq(raw["adj_close"], raw["adj_close"].shift(1))})
    return cut(out, end)


BIG_DEV = 0.05                                     # desvío diario frente a L·u que se marca (no se corrige)


def fund_flags(ticker: str, end=DEV_END) -> pd.DataFrame:
    """etf_flags de un fondo apalancado (de bonos o de acciones) más dev = r − L·u (u: el ETF 1x
    del fondo, TLT/IEF o SPY/QQQ) y big_dev = |dev| > BIG_DEV. Solo marca: los días de más de
    20 pp ya los sustituye real_fund (o common.pair); los demás se dejan como están, porque
    también los hay de mercado (marzo de 2020, con TLT lejos de su valor liquidativo)."""
    if ticker in BOND_FUNDS:
        e = BOND_FUNDS[ticker]
        r, L = real_fund(ticker, end=None), e.leverage
        u = _etf_returns(BOND_ETF[e.kind]).reindex(r.index)
    else:
        name, L = EQUITY_FUNDS[ticker]
        r = _equity_real(ticker)
        u = _equity_base(name)[0].reindex(r.index)       # SPY o QQQ durante la vida del fondo
    out = etf_flags(ticker, end=None).reindex(r.index)
    out["dev"] = r - L * u
    out["big_dev"] = out["dev"].abs() > BIG_DEV
    return cut(out, end)


@lru_cache(maxsize=None)
def fund_spread(ticker: str) -> float:
    """Sobrecoste anual s del fondo (por unidad de |L|), calibrado como r12.make/r24: la media
    diaria de rf + L·(u − f) − |L|·s − gasto iguala a la del fondo real en su vida hasta FIT_END
    (desde calib_from en los de Direxion, ver CALIB_DIREXION)."""
    from letf_lab.research.r24_folk import calibrate_spread
    e = BOND_FUNDS[ticker]
    rf, f = _rates()
    u = _fit(bond_underlying(e.kind, end=None))
    real = _fit(real_fund(ticker, end=None))
    if e.calib_from:
        real = real[real.index >= pd.Timestamp(e.calib_from)]
    return calibrate_spread(u, _fit(rf), _fit(f), e.leverage, e.er, real)


def bond_fund(ticker: str, end=DEV_END, synthetic: bool = False) -> pd.Series:
    """Fondo apalancado de bonos en sesiones de NYSE desde 1962: sintético (r24.lev_fund con s de
    fund_spread) hasta su primer rendimiento real y real después (synthetic=True: solo el
    sintético, también durante la vida del real)."""
    from letf_lab.research.r24_folk import lev_fund
    e = BOND_FUNDS[ticker]
    rf, f = _rates()
    u = bond_underlying(e.kind, end=None)
    syn = lev_fund(u, rf, f, e.leverage, e.er, fund_spread(ticker)).rename(ticker)
    if synthetic:
        return cut(syn, end)
    real = real_fund(ticker, end=None)
    out = pd.concat([syn[syn.index < real.index[0]], real]).sort_index()
    out = out[~out.index.duplicated()].reindex(syn.index).rename(ticker)
    return cut(out, end)


def bond_funds(end=DEV_END, tickers=None) -> pd.DataFrame:
    """bond_fund de varios fondos en columnas (por defecto todos los de BOND_FUNDS)."""
    tickers = tickers or list(BOND_FUNDS)
    return pd.concat({t: bond_fund(t, end) for t in tickers}, axis=1)


def make_bond(kind: str, end=DEV_END) -> dict:
    """Todo lo que necesita el indicador en un bono (análogo a r12.make para las acciones):
    u (1x: sintético y después el ETF), rf (letra), f (financiación interbancaria), x2 y x3 (los
    fondos de DECISION[kind]: sintéticos antes de su lanzamiento, reales después), spread y er por
    apalancamiento, tickers, maturity, launch (primer rendimiento real de cada fondo) y carry
    (y, dur, f_ann, carry_ann, roll_ann, prior_ann del bono sintético del mismo vencimiento)."""
    rf, f = _rates()
    t = DECISION[kind]
    M = maturity(kind)
    syn = _cmt_bond_all(M)
    d = {"u": bond_underlying(kind, end), "rf": cut(rf, end), "f": cut(f, end),
         "x2": bond_fund(t[2], end), "x3": bond_fund(t[3], end),
         "spread": {2: fund_spread(t[2]), 3: fund_spread(t[3])},
         "er": {2: BOND_FUNDS[t[2]].er, 3: BOND_FUNDS[t[3]].er},
         "tickers": dict(t), "maturity": M, "kind": kind,
         "launch": {L: _etf_returns(tk).index[0] for L, tk in t.items()},
         "carry": cut(syn[["y", "dur", "f_ann", "carry_ann", "roll_ann", "prior_ann"]], end)}
    return d


# ───────────────────────────── 4. índice MOVE ─────────────────────────────

MOVE_TICKER = "^MOVE"
MOVE_SOURCE = ("Yahoo Finance, símbolo ^MOVE (ICE BofA MOVE Index: volatilidad implícita a un mes de "
               "opciones sobre el Tesoro a 2, 5, 10 y 30 años, en pb anuales), descargado con yfinance")


def move(end=DEV_END) -> pd.DataFrame:
    """Cierre diario del MOVE (Yahoo) con marcas: repeated (igual al cierre anterior) y gap_days
    (días naturales desde el dato anterior). Yahoo da solo el cierre (apertura = máximo = mínimo)."""
    raw = _ohlc_raw(MOVE_TICKER)
    c = raw["close"].where(raw["close"] > 0)
    out = pd.DataFrame({"move": c, "repeated": _eq(c, c.shift(1)),
                        "gap_days": c.index.to_series().diff().dt.days})
    return cut(out, end)


# ───────────────────────────── 5. fondos apalancados de acciones ─────────────────────────────

EQUITY_FUNDS = {"SSO": ("S&P 500", 2.0), "UPRO": ("S&P 500", 3.0),
                "QLD": ("Nasdaq-100", 2.0), "TQQQ": ("Nasdaq-100", 3.0)}
EQUITY_DECISION = {"S&P 500": {2: "SSO", 3: "UPRO"}, "Nasdaq-100": {2: "QLD", 3: "TQQQ"}}
NDX_FROM = pd.Timestamp("1985-10-01")          # primer día de ^NDX en Yahoo (el pre-registro empieza aquí)


@lru_cache(maxsize=None)
def _equity_base(name: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(u, rf, f) de un índice de acciones hasta END: u = rentabilidad total diaria."""
    from letf_lab.research.common import long_underlying
    u = history.sp500_long() if name == "S&P 500" else long_underlying("TQQQ")
    u = u[u.index <= END].rename("u")
    return u, history.rf_long(u.index), history.funding_long(u.index)


@lru_cache(maxsize=None)
def _equity_real(ticker: str) -> pd.Series:
    """Fondo real de acciones hasta END (r12.real_letf: common.pair con la regla de 20 pp)."""
    from letf_lab.research.r12_long_ml import real_letf
    r = real_letf(ticker)
    return r[r.index <= END]


@lru_cache(maxsize=None)
def equity_spread(ticker: str) -> float:
    """Sobrecoste anual s del fondo de acciones calibrado como fund_spread, HASTA FIT_END."""
    from letf_lab.research.r24_folk import calibrate_spread
    from letf_lab.universe import BY_TICKER
    name, L = EQUITY_FUNDS[ticker]
    u, rf, f = _equity_base(name)
    return calibrate_spread(_fit(u), _fit(rf), _fit(f), L, BY_TICKER[ticker].expense_ratio,
                            _fit(_equity_real(ticker)))


def equity_fund(ticker: str, end=DEV_END, synthetic: bool = False) -> pd.Series:
    """SSO, UPRO, QLD o TQQQ: sintético rf + L·(u − f) − L·s − gasto (s de equity_spread) hasta
    el primer rendimiento real y real después (synthetic=True: solo el sintético)."""
    from letf_lab.research.r24_folk import lev_fund
    from letf_lab.universe import BY_TICKER
    name, L = EQUITY_FUNDS[ticker]
    u, rf, f = _equity_base(name)
    syn = lev_fund(u, rf, f, L, BY_TICKER[ticker].expense_ratio, equity_spread(ticker)).rename(ticker)
    if synthetic:
        return cut(syn, end)
    real = _equity_real(ticker)
    out = pd.concat([syn[syn.index < real.index[0]], real]).sort_index()
    out = out[~out.index.duplicated()].reindex(syn.index).rename(ticker)
    return cut(out, end)


def make_equity(name: str, end=DEV_END) -> dict:
    """Análogo de make_bond para «S&P 500» o «Nasdaq-100»: u, rf, f, x2, x3 (fondos de
    EQUITY_DECISION, sintéticos antes de su lanzamiento y reales después), spread, er, tickers
    y launch. El S&P 500 empieza en 1928 y el Nasdaq-100 en NDX_FROM."""
    from letf_lab.universe import BY_TICKER
    u, rf, f = _equity_base(name)
    t = EQUITY_DECISION[name]
    return {"u": cut(u, end), "rf": cut(rf, end), "f": cut(f, end),
            "x2": equity_fund(t[2], end), "x3": equity_fund(t[3], end),
            "spread": {L: equity_spread(tk) for L, tk in t.items()},
            "er": {L: BY_TICKER[tk].expense_ratio for L, tk in t.items()},
            "tickers": {1: "SPY" if name == "S&P 500" else "QQQ", **t}, "name": name,
            "launch": {L: _equity_real(tk).index[0] for L, tk in t.items()}}


def clear_cache() -> None:
    """Vacía las cachés en memoria del módulo (p. ej., después de un monkeypatch en las pruebas)."""
    for fn in (_sessions_all, _ohlc_raw, _fred_raw, _curve_all, _cmt_bond_all, _etf_returns,
               maturity_table, _rates, fund_spread, _equity_base, _equity_real, equity_spread):
        fn.cache_clear()


# ───────────────────────────── API para las demás líneas ─────────────────────────────

API = [
    ("indicador.DEV_END, TEST_START, END, FIT_END", "31-12-2018; 1-1-2019; 25-9-2026; tope de datos de toda estimación"),
    ("indicador.horizon(end), cut(obj, end)", "corte efectivo (DEV_END por defecto; None → END) y recorte"),
    ("indicador.ohlc(ticker, end)", "OHLC ajustado de ^GSPC, ^NDX, TLT o IEF con marcas flag_*, range_ok y open_ok"),
    ("indicador.range_variances(x) / variance_inputs(ticker, end)",
     "r, r2, rv_neg, rv_pos, on2, oc2, pk, gk, rs, yz diarios (log², sin anualizar)"),
    ("indicador.yang_zhang(x, n)", "estimador de ventana de Yang–Zhang de n sesiones"),
    ("indicador.cmt_curve(end) / gap_table()", "curva de FRED con los huecos de 20 y 30 años cubiertos y su tabla"),
    ("indicador.cmt_bond(M, end)", "bono a la par de M años: ret = carry + roll + curve, y, dur, prior_ann…"),
    ("indicador.maturity(kind) / maturity_table(kind)", "vencimiento que imita a TLT («largo») o IEF («medio»), hasta 2018"),
    ("indicador.bond_underlying(kind, end, synthetic)", "1x del bono: sintético hasta 30-7-2002 y ETF real después"),
    ("indicador.bond_variance_inputs(kind, end)", "r, semivarianzas, dy_bp, dur y rango del ETF desde 2002"),
    ("indicador.real_fund(t) / fund_spread(t) / bond_fund(t, end, synthetic)",
     "fondos UBT, TMF, UST, TYD (y TBT, TMV, TTT, PST): real, s calibrado hasta 2018, empalmado"),
    ("indicador.make_bond(kind, end)", "dict como r12.make: u, rf, f, x2, x3, spread, er, carry, launch…"),
    ("indicador.etf_flags(ticker, end)", "sesiones sin volumen o con el cierre igual al anterior (precio viejo)"),
    ("indicador.fund_flags(ticker, end)", "etf_flags + desvío diario r − L·u y marca de desvío > 5 pp"),
    ("indicador.move(end)", "MOVE de Yahoo desde 12-11-2002 con marcas de calidad"),
    ("indicador.equity_spread(t) / equity_fund(t, end, synthetic)",
     "SSO, UPRO, QLD, TQQQ con s calibrado hasta 2018 (no r12.make, que usa toda la vida real)"),
    ("indicador.make_equity(name, end)", "dict como make_bond para «S&P 500» o «Nasdaq-100»"),
]
