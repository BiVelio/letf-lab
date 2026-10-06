"""R13 — Leverage Rotation Strategy (Gayed y Bilello 2016) y sus variantes.

La LRS de Gayed y Bilello (2016): apalancado cuando el índice de RENTABILIDAD
TOTAL (con dividendos) cierra por encima de su media de 200 sesiones; en letras
del Tesoro cuando cierra por debajo. La media se calcula sobre la serie de
rentabilidad total (su nota 15: «We use daily closing prices of the total
return series to calculate the Moving Average for the S&P 500»; nota 16: todo
el análisis es de rentabilidad total). Es la «LRS 3x (original)» de este
informe, ejecutada con el protocolo del laboratorio (abajo): los autores no
fijan el retraso de ejecución. La misma regla sobre el índice de PRECIO es una
sensibilidad del laboratorio, «LRS 3x (señal de precio)», no la regla de los
autores. Se recalcula con los costes del laboratorio y se compara con
variantes fijadas de antemano (sin optimizar parámetros):

1. LRS con banda de histéresis (±2%): entra y sale solo con cruces claros.
2. LRS rotación: 3x sobre la media, 1x debajo (nunca en caja).
3. Medio Kelly con volatilidad GJR-GARCH (R11–R12), sin señal de tendencia.
4. LRS-Kelly: fuera bajo la media; sobre ella, medio Kelly con la volatilidad
   prevista (exposición continua entre 0 y 3x).
5. LRS-Kelly suavizado: la exposición de Kelly se multiplica por la fracción
   de medias (50, 100, 150, 200, 250 sesiones) por encima de las que cotiza.
Todas deciden con la misma señal (rentabilidad total); cada una tiene su
gemela con la señal de precio como sensibilidad.

La fracción ½ de Kelly y el GJR-GARCH salen de R10–R12, estimados con la misma
muestra del S&P 500: no son elecciones fuera de muestra, y el test SPA se hace
sobre la familia completa de reglas probadas para tenerlo en cuenta. En el
S&P 500 esa familia incluye la LRS 3x con la señal de precio, que el
laboratorio probó y publicó. En el Nasdaq-100 y en semis no hay índice de
precio diario en la base: el «precio aproximado» es la rentabilidad total menos
un dividendo constante, se muestra como sensibilidad y queda fuera del SPA.

Protocolo:
- Señal: índice de rentabilidad total del S&P 500. Desde su primer día en la
  base (4-1-1988) es el índice oficial ^SP500TR, la serie de Gayed y Bilello,
  también después de 1993, cuando u pasa a ser el ETF SPY. Antes es ^GSPC más el
  D/P de Shiller del mes ANTERIOR: el del mes en curso usa la media de los
  cierres del mes, que aún no se conocen al cerrar cada sesión
  (sp500_signal_returns). Las rentabilidades de las carteras y del índice con
  que se comparan no cambian (usan la serie u: desde 1993, el SPY ajustado).
- Ejecución: señal al cierre de t, orden al cierre de t+1 (EXEC_LAG = 2 en R12).
  Variante «mismo cierre» (MOC con la señal calculada a las 15:50), que antes
  de 1990 aprovecha una autocorrelación de precios desfasados que no era
  operable. En octubre de 1987 las dos señales cruzan su media con un día de
  diferencia, y con la orden al cierre siguiente eso decide si la regla soporta
  el lunes 19 (tabla de octubre de 1987, abajo).
- Financiación: el 3x sintético paga sobre toda su exposición el tipo de call
  money (préstamos a intermediarios con garantía de valores) hasta 1954 y el de
  los fondos federales después, con un mes de retraso, más un sobrecoste del
  swap calibrado con los ETFs reales, y los gastos anuales del fondo (tabla de
  parámetros, abajo). La caja rinde la letra del Tesoro; 5 pb por unidad de
  rotación.
- S&P 500 desde 1928 (reglas sin estimación) y desde 1938 (todas); Nasdaq-100
  evaluado solo con datos reales (desde 1985-10) y con 10 reconstrucciones de
  1971–85 para el calentamiento; semiconductores desde 1994; ETFs reales desde
  2010. El Nasdaq-100 no es un mercado virgen: R6 ya exploró TQQQ.

Mundos sintéticos (Monte Carlo): N_PATHS historias de 30 años del S&P 500 por
mundo, con tres hipótesis sobre el mercado:
- bootstrap estacionario de la historia real (bloques medios de un año):
  conserva tendencias y rachas de volatilidad;
- GJR-GARCH en log-rentabilidades con residuos estandarizados reales (con sus
  partes negativa y positiva reescaladas para no inflar la persistencia) y
  varianza de largo plazo igual a la histórica: colas y agrupamiento, sin
  tendencias;
- cambio de régimen de Markov (calma/crisis) estimado con los datos reales.
En los mundos, las estrategias Kelly prevén la varianza con EWMA (λ = 0,94) y
no con GJR, para no darles la ventaja de conocer el modelo que genera los datos.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import brentq

from letf_lab import data, kelly, metrics, stats
from letf_lab.data import history
from letf_lab.report import FIG_W, Report, comma_fig, legend_band, num_es, pct_es
from letf_lab.research import r12_long_ml as r12

log = logging.getLogger(__name__)

COST_BPS = 5.0
KELLY = 0.5
BAND = 0.02
MIN_STEP = 0.10      # sólo se reajusta la exposición continua si cambia > 10 pp
SEED = 13
B = 2000
EXEC_LAG = r12.EXEC_LAG
REAL_FROM = "2010-02-12"  # TQQQ, el más tardío de los tres ETFs reales
N_PATHS = 1000
YEARS, BURN = 30, 10       # se simulan 40 años y se evalúan los 30 últimos
N_NDX = 10                 # reconstrucciones del Nasdaq-100 que se evalúan
DIV = {"S&P 500": None, "Nasdaq-100": history.NDX_DIV, "Semis": 0.012}

# Señal de la LRS. La original (Gayed y Bilello 2016, nota 15) mira la media de 200
# sesiones del índice de rentabilidad total; la de precio es una sensibilidad.
SIGNAL = "rentabilidad total"
PRICE = "precio"
ORIGINAL = "LRS 3x (original)"
PX_TAG = "señal de precio"
PX_APPROX_TAG = "señal de precio aproximado"   # Nasdaq-100 y semis: TR menos un dividendo constante
PRICE_BASE = {"LRS 1x (original)": "LRS 1x", "LRS 2x (original)": "LRS 2x",
              "LRS 3x (original)": "LRS 3x", "LRS-Kelly": "LRS-Kelly"}


def price_label(key: str, approx: bool = False) -> str:
    """Nombre de la gemela de `key` con la señal de precio: «LRS 3x (señal de precio)»,
    «LRS 3x + banda ±2% · señal de precio»; con approx, «… señal de precio aproximado»."""
    tag = PX_APPROX_TAG if approx else PX_TAG
    if key in PRICE_BASE:
        return f"{PRICE_BASE[key]} ({tag})"
    return f"{key} · {tag}"


PRICE_LABELS = {k: price_label(k) for k in PRICE_BASE}
PX_LRS = PRICE_LABELS[ORIGINAL]           # «LRS 3x (señal de precio)»
PX_KELLY = PRICE_LABELS["LRS-Kelly"]      # «LRS-Kelly (señal de precio)»

# Ventanas de inicio fijadas (S&P 500): las medias de 200 sesiones de las dos señales
# existen desde el cierre del 17-10-1928 y, con la orden al cierre siguiente, la primera
# rentabilidad de la LRS es la del 19-10-1928. La Gran Depresión, la conciliación con
# Gayed y Bilello y la ventana «1928–» empiezan ese día, como la ventana 1928–2026 de R18.
FIRST_RULE_DAY = "1928-10-19"
GB_END = "2015-10-30"      # fin de la muestra de Gayed y Bilello (octubre de 2015)
GB_FEE = 0.01              # «leverage fee of 1% per year» de Gayed y Bilello
# Tabla 8 de Gayed y Bilello (2016), octubre de 1928 a octubre de 2015: cifras publicadas
GB_TABLE8 = {"índice (S&P 500)": 0.091, "3x comprar y mantener": 0.153, "LRS 3x": 0.268,
             "LRS 3x, caída máxima": -0.922}
COST_GRID = (0, 5, 10, 25, 50, 75, 100)   # pb por unidad de rotación
COST_MAX = 2000.0                         # pb: límite de la búsqueda del coste que anula Δg


# ───────────────────────────── datos ─────────────────────────────

def price_index(d: dict, name: str) -> pd.Series:
    """Índice de PRECIO, solo para la sensibilidad «señal de precio» (la LRS original
    de Gayed y Bilello usa el de rentabilidad total, tr_index). En el S&P 500 es ^GSPC
    (con los cierres corregidos); en el Nasdaq-100 y en semis, aproximado: la
    rentabilidad total menos un dividendo constante (DIV)."""
    u = d["u"]
    if name == "S&P 500":
        return history.gspc_close().reindex(u.index).ffill()
    return (1 + (u - DIV[name] / 252).fillna(0)).cumprod()


def sp500_tr_level() -> pd.Series:
    """Nivel diario del índice oficial de rentabilidad total del S&P 500 (^SP500TR,
    cierre ajustado de la caché), sin días vacíos."""
    return data.panel(["^SP500TR"], "adj_close")["^SP500TR"].dropna()


def sp500_signal_returns(u: pd.Series, dy: pd.Series | None = None,
                         tr_start: pd.Timestamp | None = None,
                         tr_level: pd.Series | None = None) -> pd.Series:
    """Rentabilidad diaria con dividendos del S&P 500 para la SEÑAL, sin mirar el futuro.

    Es la única implementación de la señal de rentabilidad total del S&P 500 (r10 la
    usa para los estados de R9–R11; R13–R20, para la LRS original y sus variantes):
    - Hasta el primer día de ^SP500TR (tr_start, el 4-1-1988): history.sp500_long suma
      a ^GSPC el D/P de Shiller del mes EN CURSO, cuyo precio es la media de los
      cierres de ese mes y al cierre de cada sesión aún no se conoce. Para la señal se
      usa el del mes ANTERIOR (history._monthly_to_daily, la convención de rf_long y
      funding_long): u + (D/P del mes anterior − D/P del mes en curso)/252.
    - Después: la rentabilidad del índice oficial ^SP500TR (cierre ajustado,
      pct_change), la serie de Gayed y Bilello, y no la de u, que desde 1993 es el ETF
      SPY (neto de su comisión y con los dividendos en efectivo hasta el reparto).
      Un día de u sin nivel de ^SP500TR usa la rentabilidad de u ese día (y se avisa
      en el registro); el día siguiente con nivel mide el índice desde el último nivel
      conocido compuesto con esos días, así que el índice de la señal vuelve a
      coincidir con ^SP500TR sin contar dos veces el hueco. Los días de u posteriores
      al último nivel de ^SP500TR también usan u. Si u empieza después de tr_start (una
      submuestra), su primer día se mide desde el último nivel anterior a ese día, así que
      la señal coincide día a día con la de la serie completa.
    Las rentabilidades de las carteras y del índice con que se comparan siguen con u.
    dy, tr_start y tr_level (nivel de ^SP500TR) solo se pasan en las pruebas."""
    dy = history.shiller_dividend_yield() if dy is None else dy
    tr_level = sp500_tr_level() if tr_level is None else tr_level.dropna()
    if tr_start is None:
        tr_start = tr_level.index[0]
    idx = u.index
    early = idx <= tr_start
    cur = dy.reindex(idx.union(dy.index)).ffill().reindex(idx)
    prev = history._monthly_to_daily(dy, idx)
    adj = ((prev - cur) / 252).where(early, 0.0).fillna(0.0)
    out = (u + adj).to_numpy(dtype=float, copy=True)
    # desde tr_start: rentabilidad de ^SP500TR sobre el último nivel conocido. Si u empieza
    # después de tr_start (una submuestra), la base es el último nivel ANTERIOR a su primer
    # día: con el de tr_start, la primera rentabilidad sumaría años de índice en un día.
    ref = idx[0] if len(idx) and idx[0] > tr_start else None
    lv = tr_level[tr_level.index < ref] if ref is not None else tr_level[tr_level.index <= tr_start]
    last = float(lv.iloc[-1]) if len(lv) else np.nan
    levels = tr_level.reindex(idx).to_numpy(float)
    uv = u.to_numpy(float)
    filled = []
    for i in np.flatnonzero(~early):
        lvl = levels[i]
        if np.isfinite(lvl) and np.isfinite(last):
            out[i] = lvl / last - 1
            last = lvl
        else:                      # hueco de ^SP500TR (o sin nivel previo): u ese día
            out[i] = uv[i]
            if np.isfinite(lvl):
                last = lvl
            elif np.isfinite(last) and np.isfinite(uv[i]):
                last *= 1 + uv[i]
            filled.append(idx[i])
    if filled:
        log.warning("señal del S&P 500: %d días sin ^SP500TR usan u (%s … %s)", len(filled),
                    filled[0].date(), filled[-1].date())
    s = pd.Series(out, index=idx, name=u.name)
    s.attrs["días_sin_SP500TR"] = [d.date() for d in filled]
    return s


def legacy_signal_returns(d: dict, tr_start: pd.Timestamp | None = None) -> pd.Series:
    """Señal de la versión anterior del laboratorio: la de sp500_signal_returns hasta el
    primer día de ^SP500TR y u (desde 1993, el SPY) después. Solo para la sensibilidad."""
    tr_start = sp500_tr_level().index[0] if tr_start is None else tr_start
    u = d["u"]
    return d["u_sig"].where(u.index <= tr_start, u)


def legacy_signal_table(d: dict, R: dict, start, tr_start: pd.Timestamp | None = None
                        ) -> tuple[pd.DataFrame, int]:
    """LRS 3x (original) con la señal de ^SP500TR (la base) y con la de la versión anterior
    (SPY desde 1993), desde `start`, desde 1993 y desde 2000; y nº de sesiones con señal distinta."""
    old = dict(d, u_sig=legacy_signal_returns(d, tr_start))
    s_new, s_old = above_sma(tr_index(d)), above_sma(tr_index(old))
    r_old = run_weights(old, s_old)
    one = R["1x comprar y mantener"]
    rows = []
    for lab, r in ((f"{ORIGINAL}: señal ^SP500TR (base)", R[ORIGINAL]),
                   ("LRS 3x: señal SPY desde 1993 (versión anterior)", r_old)):
        for w, a in (("primer día con previsión GJR", start), ("1993 (1 de febrero)", "1993-02-01"),
                     ("2000", "2000-01-03")):
            x = r[r.index >= pd.Timestamp(a)].dropna()
            b = one.reindex(x.index)
            rows.append({"señal": lab, "desde": w, "inicio": x.index[0].date(),
                         "CAGR": metrics.cagr(x), "maxDD": metrics.max_dd(x),
                         "Δg_frente_1x_pp": float((np.log1p(x) - np.log1p(b)).mean() * 252 * 100)})
    n_diff = int((s_new != s_old)[s_new.notna() & s_old.notna()].sum())
    return pd.DataFrame(rows), n_diff


def var_forecast(d: dict, kind: str = "gjr") -> pd.Series:
    """Varianza anual prevista a 21 días (GJR/GARCH/ARCH con reestimación)."""
    return r12.garch_walk(d["u"].dropna(), kind).reindex(d["u"].index)


def load() -> dict:
    """Mercados de R12 con el índice de precio (px, sensibilidad) y, en el S&P 500,
    la rentabilidad para la señal sin mirar el futuro (u_sig) y px_real = True: solo
    ahí la señal de precio usa un índice de precio real (^GSPC)."""
    D = r12.load_all()
    nd = D.pop("_ndx")
    for name, d in D.items():
        d["px"] = price_index(d, name)
        d["px_real"] = name == "S&P 500"
        if name == "S&P 500":
            d["u_sig"] = sp500_signal_returns(d["u"])
    D["_ndx"] = nd
    return D


# ───────────────────────────── señales ─────────────────────────────

def tr_index(d: dict) -> pd.Series:
    """Índice de rentabilidad total para la señal de la LRS original: cumprod(1 + u),
    con u_sig (sin mirar el futuro) si el mercado la tiene."""
    u = d.get("u_sig", d["u"])
    return (1 + u.reindex(d["u"].index).fillna(0)).cumprod()


def signal_index(d: dict, kind: str | None = None) -> pd.Series:
    """Índice sobre el que se calcula la media de la LRS: kind = SIGNAL (rentabilidad
    total, la original) o PRICE (d['px'], sensibilidad). Sin kind, d['signal'] si el
    diccionario la fija, y si no la original."""
    kind = kind or d.get("signal") or SIGNAL
    if kind == SIGNAL:
        return tr_index(d)
    if kind == PRICE:
        return d["px"]
    raise ValueError(f"señal desconocida: {kind}")


def above_sma(px: pd.Series, n: int = 200, band: float = 0.0) -> pd.Series:
    """1 sobre la media, 0 debajo; con banda, sólo cambia al cruzar ±band."""
    ma = px.rolling(n).mean()
    if band == 0:
        return (px > ma).astype(float).where(ma.notna())
    state = np.where(px > ma * (1 + band), 1.0, np.where(px < ma * (1 - band), 0.0, np.nan))
    return pd.Series(state, index=px.index).ffill().where(ma.notna())


def kelly_exposure(d: dict, var_fc: pd.Series) -> pd.Series:
    """Fracción del 3x (0–1) que da medio Kelly con la varianza prevista.

    L* = (μ̂ − f̂ − s)/σ²: toda la exposición (L veces el capital) se financia a f y
    μ̂ − f̂ es la prima compartida (kelly.premium: media creciente de u − f con datos
    hasta t − 1)."""
    prem = kelly.premium(d["u"], d["f"], r12.MIN_OBS)
    L = KELLY * (prem - d["spread"]) / var_fc
    return (L / 3).clip(0, 1)


def sticky(target: pd.Series) -> pd.Series:
    """Reajusta sólo si el objetivo se aleja más de MIN_STEP (o va a 0 / 1)."""
    vals, cur = [], np.nan
    for v in target.values:
        if np.isnan(v):
            vals.append(np.nan)
            continue
        if np.isnan(cur) or abs(v - cur) > MIN_STEP or v in (0.0, 1.0):
            cur = v
        vals.append(cur)
    return pd.Series(vals, index=target.index)


# ───────────────────────────── carteras ─────────────────────────────

def run_weights(d: dict, w3: pd.Series, w1: pd.Series | None = None, lag: int = EXEC_LAG,
                x: pd.Series | None = None, cost_bps: float = COST_BPS) -> pd.Series:
    """w3 en el fondo apalancado x (3x por defecto), w1 en el 1x, resto en letras.

    Decidido al cierre de t; con lag = 2 cobra la rentabilidad desde el cierre
    de t+1 (orden al cierre siguiente), con lag = 1 desde el cierre de t.
    cost_bps: coste por unidad de rotación (5 pb en todo el laboratorio)."""
    x = d["x3"] if x is None else x
    w3 = w3.shift(lag)
    w1 = w1.shift(lag) if w1 is not None else pd.Series(0.0, index=w3.index)
    valid = w3.notna() & w1.notna()
    w3, w1 = w3.fillna(0), w1.fillna(0)
    turn = w3.diff().abs().fillna(0) + w1.diff().abs().fillna(0)
    r = w3 * x + w1 * d["u"] + (1 - w3 - w1) * d["rf"] - turn * cost_bps / 1e4
    return r.where(valid)


def strategies(d: dict, var_fc: pd.Series, lag: int = EXEC_LAG,
               signal: str | None = None) -> tuple[dict, dict]:
    """Reglas de R13 con la señal `signal` (por defecto la original, rentabilidad
    total; véase signal_index). Mismas claves con cualquier señal."""
    u = d["u"]
    sx = signal_index(d, signal)
    x2 = r12.letf(d, 2.0)
    sig = above_sma(sx)
    sig_b = above_sma(sx, band=BAND)
    kel = kelly_exposure(d, var_fc)
    trend = pd.concat([above_sma(sx, n) for n in (50, 100, 150, 200, 250)], axis=1).mean(axis=1)
    R = {"1x comprar y mantener": u, "3x comprar y mantener": d["x3"]}
    E = {"1x comprar y mantener": pd.Series(1.0, index=u.index),
         "3x comprar y mantener": pd.Series(3.0, index=u.index)}
    zero = pd.Series(0.0, index=u.index)
    R["LRS 1x (original)"], E["LRS 1x (original)"] = run_weights(d, zero, w1=sig, lag=lag), sig
    R["LRS 2x (original)"], E["LRS 2x (original)"] = run_weights(d, sig, lag=lag, x=x2), 2 * sig
    R["LRS 3x (original)"], E["LRS 3x (original)"] = run_weights(d, sig, lag=lag), 3 * sig
    R["LRS 3x + banda ±2%"], E["LRS 3x + banda ±2%"] = run_weights(d, sig_b, lag=lag), 3 * sig_b
    R["LRS rotación 3x/1x"] = run_weights(d, sig, w1=1 - sig, lag=lag)
    E["LRS rotación 3x/1x"] = 3 * sig + (1 - sig)
    for name, w in (("½ Kelly GJR (sin tendencia)", sticky(kel)),
                    ("LRS-Kelly", sticky(kel * sig)),
                    ("LRS-Kelly suavizado", sticky(kel * trend))):
        R[name], E[name] = run_weights(d, w, lag=lag), 3 * w
    return R, E


PRICE_DROP = ("1x comprar y mantener", "3x comprar y mantener", "½ Kelly GJR (sin tendencia)")


def price_variants(d: dict, var_fc: pd.Series, lag: int = EXEC_LAG) -> tuple[dict, dict]:
    """Sensibilidad «señal de precio»: las reglas de strategies con la media sobre
    d['px'], con los nombres de price_label («LRS 3x (señal de precio)», …; «señal de
    precio aproximado» si el mercado no tiene un índice de precio real). Sin comprar
    y mantener ni ½ Kelly GJR (sin tendencia), que no dependen de la señal."""
    R, E = strategies(d, var_fc, lag=lag, signal=PRICE)
    approx = not d.get("px_real", False)
    keep = [k for k in R if k not in PRICE_DROP]
    return ({price_label(k, approx): R[k] for k in keep}, {price_label(k, approx): E[k] for k in keep})


def family(d: dict, R: dict, lag: int = EXEC_LAG, signal: str | None = None) -> tuple[dict, dict]:
    """Familia completa de reglas probadas para el SPA.

    Principal: todas las reglas de R13 + medio Kelly con los modelos de
    varianza baratos de R12 (EWMA, ARCH, GARCH, GJR). Con la señal original y un
    índice de precio real (S&P 500), además la LRS 3x con la señal de precio, que
    el laboratorio probó y publicó. Sensibilidad: además, la rejilla de 30
    combinaciones de media y banda sobre signal_index(d, signal)."""
    main = {k: v for k, v in R.items() if "comprar y mantener" not in k}
    if (signal or d.get("signal") or SIGNAL) == SIGNAL and d.get("px_real", False) and PX_LRS not in main:
        main[PX_LRS] = run_weights(d, above_sma(signal_index(d, PRICE)), lag=lag)
    u = d["u"].dropna()
    ew = (u ** 2 * 252).ewm(alpha=0.06, adjust=False).mean().reindex(d["u"].index)
    for label, v in (("EWMA", ew), ("ARCH(1)", var_forecast(d, "arch")),
                     ("GARCH(1,1)", var_forecast(d, "garch"))):
        main[f"½ Kelly {label} (R12)"] = r12.strategy(d, v, lag=lag)[0]
    grid = dict(main)
    sx = signal_index(d, signal)
    for n in (50, 100, 150, 200, 250, 300):
        for band in (0.0, 0.01, 0.02, 0.03, 0.05):
            grid[f"LRS 3x media {n} banda {band:.0%}"] = run_weights(d, above_sma(sx, n, band), lag=lag)
    return main, grid


# ───────────────────────────── tablas ─────────────────────────────

def stats_table(R: dict, rf_: pd.Series, start, end=None) -> pd.DataFrame:
    rows = {}
    for k, r in R.items():
        r = r.loc[start:end].dropna()
        if len(r) < 50:
            continue
        m = metrics.summary(r, rf_)
        yr = (1 + r).groupby(r.index.year).prod() - 1
        rows[k] = {"CAGR": m["CAGR"], "vol": m["vol"], "sharpe": m["sharpe"], "sortino": m["sortino"],
                   "maxDD": m["maxDD"], "calmar": m["calmar"], "ulcer": m["ulcer"],
                   "años_bajo_agua": m["max_underwater_d"] / 252, "peor_año": yr.min(),
                   "10k_final": 10_000 * (1 + r).prod()}
    return pd.DataFrame(rows).T


def trading_stats(E: dict, start) -> pd.DataFrame:
    rows = {}
    for k, e in E.items():
        e = e.loc[start:].dropna()
        ch = e.diff().abs() > 1e-9
        rows[k] = {"exposición_media": e.mean(), "tiempo_invertido": (e > 0).mean(),
                   "operaciones_año": ch.sum() / (len(e) / 252)}
    return pd.DataFrame(rows).T


def log_growth(r: pd.Series) -> float:
    """Crecimiento logarítmico anual: media de log(1 + r) ×252."""
    return float(np.log1p(r.dropna().clip(lower=-0.9999)).mean() * 252)


def price_diff_table(R: dict, Rp: dict, rf_: pd.Series, start, end=None,
                     approx: bool = False) -> pd.DataFrame:
    """Diferencias de la señal de precio frente a la original, regla a regla, en la
    misma ventana: ΔCAGR, Δ caída máxima (positivo = cae menos) y Δg anual."""
    rows = {}
    for k in R:
        pk = price_label(k, approx)
        if pk not in Rp:
            continue
        both = pd.concat([R[k], Rp[pk]], axis=1, keys=["o", "p"]).loc[start:end].dropna()
        if len(both) < 50:
            continue
        rows[k] = {"ΔCAGR": metrics.cagr(both.p) - metrics.cagr(both.o),
                   "Δcaída_máx": metrics.max_dd(both.p) - metrics.max_dd(both.o),
                   "Δg_anual": log_growth(both.p) - log_growth(both.o)}
    return pd.DataFrame(rows).T


def drawdown_dates(r: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Caída máxima (como metrics.max_dd), fecha del máximo previo y del mínimo."""
    w = (1 + r.fillna(0)).cumprod()
    dd = w / w.cummax() - 1
    trough = dd.idxmin()
    return float(dd.min()), w.loc[:trough].idxmax(), trough


def windows_table(R: dict, keys, starts: dict, one: str = "1x comprar y mantener",
                  tol_days: int = 10) -> pd.DataFrame:
    """Ventanas de inicio fijadas: para cada inicio y regla, CAGR, caída máxima con las
    fechas de su máximo previo y de su mínimo, y si rinde más y cae menos que el 1x en
    la misma ventana. Una regla que empieza más de tol_days sesiones después del inicio
    (p. ej. las de Kelly antes de 1938) no se muestra en esa ventana."""
    rows = []
    for lab, s in starts.items():
        s = pd.Timestamp(s)
        base = R[one].loc[s:].dropna()
        b_cagr, b_dd = metrics.cagr(base), metrics.max_dd(base)
        for k in keys:
            r = R[k].loc[s:].dropna()
            if len(r) < 50 or len(base.loc[:r.index[0]]) > tol_days:
                continue
            dd, peak, trough = drawdown_dates(r)
            c = metrics.cagr(r)
            rows.append({"inicio": lab, "desde": r.index[0].date(), "regla": k, "CAGR": c, "caída_máx": dd,
                         "máximo_previo": peak.date(), "mínimo": trough.date(),
                         "rinde_más_que_1x": "sí" if c > b_cagr else "no",
                         "cae_menos_que_1x": "sí" if dd > b_dd else "no"})
    return pd.DataFrame(rows)


def drawdown_episodes(rule: pd.DataFrame) -> list[tuple]:
    """Filas de windows_table de UNA regla (índice = ventana) agrupadas por episodio de
    caída: [(caída, máximo previo, mínimo, [ventanas])], en el orden de aparición. Se
    agrupa por las fechas del episodio y no por la caída, que en un mismo episodio
    difiere en las últimas cifras de una ventana a otra (distinto producto acumulado)."""
    out = []
    for (pk, tr), grp in rule.groupby(["máximo_previo", "mínimo"], sort=False):
        out.append((float(grp["caída_máx"].iloc[0]), pk, tr, list(grp.index)))
    return out


def gb_reconciliation(d: dict, start=FIRST_RULE_DAY, end=GB_END) -> pd.DataFrame:
    """Del 26,8% de Gayed y Bilello a la LRS 3x (original) del laboratorio, paso a paso,
    en su muestra (S&P 500, de octubre de 1928 a octubre de 2015; aquí desde la primera
    sesión de la LRS con la orden al cierre siguiente, FIRST_RULE_DAY).

    P1 son sus supuestos: señal de rentabilidad total, orden en el mismo cierre, un 3x
    de 3u − 1%/252 (solo la «leverage fee» del 1% anual, sin intereses por lo prestado)
    y sin costes de rotación. Cada paso añade una fricción: P2 paga el préstamo al tipo
    de la letra; P3 usa el 3x del laboratorio (swap a f + s y gasto del fondo); P4
    ejecuta al cierre siguiente; P5 cobra 5 pb por unidad de rotación, y es la LRS 3x
    (original). P4′ y P5′ son P4 y P5 con la señal de precio."""
    u, rf = d["u"], d["rf"]
    x_gb = 3 * u - GB_FEE / 252
    x_tb = 3 * u - 2 * rf - GB_FEE / 252
    s_tr, s_px = above_sma(signal_index(d, SIGNAL)), above_sma(signal_index(d, PRICE))
    rows = {
        "1x (índice con dividendos)": u,
        "3x de Gayed y Bilello comprar y mantener (3u − 1%)": x_gb,
        "P1 · supuestos de Gayed y Bilello (señal de rentabilidad total, mismo cierre, 3u − 1%, "
        "sin intereses ni costes de rotación)": run_weights(d, s_tr, lag=1, x=x_gb, cost_bps=0),
        "P2 · + préstamo al tipo de la letra (3u − 2rf − 1%)": run_weights(d, s_tr, lag=1, x=x_tb, cost_bps=0),
        "P3 · + 3x del laboratorio (swap a f + s y gasto del fondo)": run_weights(d, s_tr, lag=1, cost_bps=0),
        "P4 · + orden al cierre siguiente": run_weights(d, s_tr, lag=EXEC_LAG, cost_bps=0),
        f"P5 · + {COST_BPS:g} pb por unidad de rotación (= {ORIGINAL})": run_weights(d, s_tr, lag=EXEC_LAG),
        "P4′ · P4 con la señal de precio (sin costes de rotación)": run_weights(d, s_px, lag=EXEC_LAG, cost_bps=0),
        f"P5′ · P5 con la señal de precio (= {PX_LRS})": run_weights(d, s_px, lag=EXEC_LAG),
    }
    out = {}
    for k, r in rows.items():
        r = r.loc[start:end].dropna()
        out[k] = {"desde": r.index[0].date(), "hasta": r.index[-1].date(), "CAGR": metrics.cagr(r),
                  "caída_máx": metrics.max_dd(r)}
    return pd.DataFrame(out).T


def cost_table(d: dict, sig: pd.Series, start, lag: int = EXEC_LAG,
               grid=COST_GRID, cost_max: float = COST_MAX) -> tuple[pd.DataFrame, dict]:
    """La LRS 3x con la señal `sig` y distintos costes por unidad de rotación (pb).

    Δg frente al 1x y frente al L constante de su exposición media (L̄/3 en el 3x y
    el resto en letras, reequilibrado a diario sin coste: la referencia de R16 y R17),
    en crecimiento logarítmico anual, y el coste por unidad que anula cada Δg (brentq
    hasta cost_max pb; inf = no se anula en ese intervalo, nan = sin ventaja con coste 0)."""
    L = float((3 * sig).shift(lag).loc[start:].mean())
    r_c = (L / 3) * d["x3"] + (1 - L / 3) * d["rf"]
    idx = run_weights(d, sig, lag=lag).loc[start:].dropna().index
    g = lambda r: log_growth(r.reindex(idx))  # noqa: E731
    g1, gc = g(d["u"]), g(r_c)
    rule = lambda c: g(run_weights(d, sig, lag=lag, cost_bps=c))  # noqa: E731
    e = (3 * sig).loc[idx[0]:]
    ch = (e.diff().abs() > 1e-9).sum() / (len(e) / 252)
    rows = {}
    for c in grid:
        r = run_weights(d, sig, lag=lag, cost_bps=c).reindex(idx)
        rows[f"{c:g} pb"] = {"CAGR": metrics.cagr(r), "Δg_frente_1x": rule(c) - g1,
                             "Δg_frente_L_constante": rule(c) - gc}

    def zero(ref: float) -> float:
        h = lambda c: rule(c) - ref  # noqa: E731
        if h(0.0) <= 0:
            return np.nan
        if h(cost_max) > 0:
            return np.inf
        return float(brentq(h, 0.0, cost_max, xtol=0.1))

    info = {"L_constante": L, "cambios_año": float(ch), "desde": idx[0].date(),
            "coste_anula_Δg_1x": zero(g1), "coste_anula_Δg_L": zero(gc)}
    return pd.DataFrame(rows).T, info


def fmt_cost(x: float) -> str:
    """Coste que anula una ventaja, para el texto: «no se anula», «sin ventaja» o «≈ 75 pb»."""
    if np.isinf(x):
        return f"no se anula por debajo de {num_es(COST_MAX, 0)} pb"
    if np.isnan(x):
        return "sin ventaja ni con coste cero"
    return f"≈ {num_es(x, 0)} pb"


def params_table(D: dict) -> pd.DataFrame:
    """Parámetros del 3x sintético de cada mercado (r12.make): gasto del fondo, sobrecoste
    del swap sobre la letra (R3) y recalibrado sobre f, y la media de f − rf en la vida del
    ETF real y en toda la muestra."""
    from letf_lab.research.r3_synthetic import synthetic
    rows = {}
    for name, d in D.items():
        t = d["e"].ticker
        launch = data.panel([t], "adj_close")[t].first_valid_index()
        fr = (d["f"] - d["rf"]).reindex(d["u"].dropna().index)
        rows[name] = {"ETF": t, "lanzamiento": launch.date(), "gasto_del_fondo": d["er"],
                      "s_sobre_la_letra_R3": synthetic(t)[1], "s_sobre_f": d["spread"],
                      "f_menos_rf_vida_ETF": float(fr.loc[launch:].mean() * 252),
                      "f_menos_rf_toda_la_muestra": float(fr.mean() * 252),
                      "muestra_desde": fr.index[0].date()}
    return pd.DataFrame(rows).T


GRID_N = (50, 100, 150, 200, 250, 300)
GRID_BANDS = (0.0, 0.01, 0.02, 0.03, 0.05)


def robust_grid(d: dict, sx: pd.Series, start) -> pd.DataFrame:
    """Rejilla de robustez de la LRS 3x: media de n sesiones y banda sobre el índice sx."""
    g = []
    for n in GRID_N:
        for band in GRID_BANDS:
            r = run_weights(d, above_sma(sx, n, band)).loc[start:].dropna()
            g.append({"media": n, "banda": band, "CAGR": metrics.cagr(r),
                      "sharpe": metrics.sharpe(r, d["rf"]), "maxDD": metrics.max_dd(r)})
    return pd.DataFrame(g)


def grid_rank(g: pd.DataFrame, n: int = 200, band: float = 0.0) -> dict:
    """Puesto (1 = mejor) de la combinación (n, band) en la rejilla por CAGR, Sharpe y
    caída máxima (la menos profunda es la mejor)."""
    i = g.index[(g.media == n) & (g.banda == band)][0]
    return {c: int(g[c].rank(ascending=False, method="min").loc[i]) for c in ("CAGR", "sharpe", "maxDD")}


def distance_1987(d: dict, a: str = "1987-10-13", b: str = "1987-10-20") -> pd.DataFrame:
    """Distancia de cada índice de señal a su media de 200 sesiones (índice/media − 1)
    en octubre de 1987: el primer día negativo es el día en que la señal pasa a letras."""
    out = {}
    for kind, lab in ((SIGNAL, "rentabilidad total (original)"), (PRICE, "precio")):
        s = signal_index(d, kind)
        out[f"señal de {lab}"] = (s / s.rolling(200).mean() - 1).loc[a:b]
    df = pd.DataFrame(out)
    df.index = df.index.date
    df.index.name = "sesión"
    return df


def spa_tests(fam: dict, R: dict, start, rng) -> list[dict]:
    logs = pd.DataFrame({k: np.log1p(v.loc[start:].clip(lower=-0.999)) for k, v in {**fam, **R}.items()})
    logs = logs.dropna()
    cands = list(fam)
    out = []
    for bench in ("LRS 3x (original)", "1x comprar y mantener", "3x comprar y mantener"):
        cc = [c for c in cands if c != bench]
        D = logs[cc].sub(logs[bench], axis=0).values
        spa = stats.spa_test(D, 63, B, rng)
        best = int(D.mean(axis=0).argmax())
        out.append({"frente a": bench, "reglas": len(cc), "mejor": cc[best],
                    "Δ_crec_anual": D.mean(axis=0)[best] * 252, "p_SPA_c": spa["p_SPA_c"]})
    return out


def ndx_imputations(nd: dict, rng) -> pd.DataFrame:
    """Métricas del Nasdaq-100 (evaluado desde 1985-10) en N_NDX reconstrucciones:
    el pasado reconstruido sólo entra por el calentamiento de μ y del GJR."""
    real = r12.real_letf("TQQQ")
    rows = []
    for k, u in enumerate(nd["draws"][:N_NDX]):
        d = r12.make(u, history.rf_long(u.index), "TQQQ", real, eval_from=r12.NDX_EVAL_FROM)
        d["px"] = price_index(d, "Nasdaq-100")
        R, _ = strategies(d, var_forecast(d))
        t = stats_table(R, d["rf"], r12.NDX_EVAL_FROM)
        for strat in ("1x comprar y mantener", "3x comprar y mantener", "LRS 3x (original)",
                      "½ Kelly GJR (sin tendencia)", "LRS-Kelly"):
            rows.append({"reconstrucción": k, "estrategia": strat, "CAGR": t.loc[strat, "CAGR"],
                         "maxDD": t.loc[strat, "maxDD"]})
    df = pd.DataFrame(rows)
    return df.groupby("estrategia").agg(CAGR_mediana=("CAGR", "median"), CAGR_mín=("CAGR", "min"),
                                        CAGR_máx=("CAGR", "max"), maxDD_mediana=("maxDD", "median"))


# ───────────────────────────── mundos sintéticos ─────────────────────────────

def gen_paths(method: str, u: np.ndarray, n: int, T: int, rng) -> np.ndarray:
    """n historias de T días de rentabilidades diarias simples del índice (> −100%)."""
    u = u[~np.isnan(u)]
    if method == "bootstrap":
        idx = np.stack([stats.stationary_indices(len(u), 252, rng, T) for _ in range(n)])
        return u[idx]
    ell = np.log1p(u)
    if method == "gjr":
        p = stats.garch_fit(ell, "gjr")
        m = ell.mean()
        h = np.r_[np.var(ell), stats.garch_filter(ell - m, p)[:-1]]
        z = (ell - m) / np.sqrt(h)
        z = z / z.std()
        # Los residuos reales son asimétricos (E[z²·1{z<0}] ≈ 0,55, no 0,5), lo que
        # eleva la persistencia efectiva del GJR (0,9952 frente a 0,9905) y hace
        # explotar la varianza en algunas historias. Se reescalan por separado las
        # partes negativa y positiva para que cada una aporte la mitad de la varianza,
        # conservando la forma de las colas.
        z[z < 0] *= np.sqrt(0.5 / np.mean(z ** 2 * (z < 0)))
        z[z >= 0] *= np.sqrt(0.5 / np.mean(z ** 2 * (z >= 0)))
        z = (z - z.mean()) / z.std()
        p_eff = p["alpha"] * np.mean(z ** 2) + p["gamma"] * np.mean(z ** 2 * (z < 0)) + p["beta"]
        omega = np.var(ell) * (1 - p_eff)
        hv = np.full(n, np.var(ell))
        out = np.empty((n, T))
        for t in range(T):
            e = rng.choice(z, n) * np.sqrt(hv)
            out[:, t] = m + e
            hv = omega + (p["alpha"] + p["gamma"] * (e < 0)) * e ** 2 + p["beta"] * hv
        return np.expm1(out)
    if method == "regímenes":
        wk = pd.Series(ell).groupby(np.arange(len(ell)) // 5).sum().values
        f = stats.hmm_fit(wk)
        mu_d, sd_d = f["mu"] / 5, f["sd"] / np.sqrt(5)
        stay = np.diag(f["P"]) ** (1 / 5)
        state = rng.choice(2, n, p=f["smoothed"].mean(axis=0))
        out = np.empty((n, T))
        for t in range(T):
            out[:, t] = mu_d[state] + sd_d[state] * rng.standard_normal(n)
            switch = rng.random(n) > stay[state]
            state = np.where(switch, 1 - state, state)
        return np.expm1(out)
    raise ValueError(method)


def world_diagnostics(U: np.ndarray, u_real: np.ndarray) -> dict:
    """Cómo se parece cada mundo a la historia real (por 30 años)."""
    per30 = lambda cnt, days: cnt / days * 252 * YEARS
    real = u_real[~np.isnan(u_real)]
    Ue = U[:, BURN * 252:]
    return {
        "vol_anual_mediana": float(np.median(Ue.std(axis=1)) * np.sqrt(252)),
        "vol_anual_cuadrática_media": float(np.sqrt(np.mean(Ue.var(axis=1))) * np.sqrt(252)),
        "vol_anual_historia": float(real.std() * np.sqrt(252)),
        "días<−10% por 30 años": float(np.mean((Ue < -0.10).sum(axis=1))),
        "historia: días<−10%": per30((real < -0.10).sum(), len(real)),
        "días<−20% por 30 años": float(np.mean((Ue < -0.20).sum(axis=1))),
        "historia: días<−20%": per30((real < -0.20).sum(), len(real)),
        "historias con un día ≤ −33,3%": float(np.mean((Ue <= -1 / 3).any(axis=1))),
    }


def _sma_signal(px: np.ndarray, n: int, band: float = 0.0) -> np.ndarray:
    cs = np.cumsum(px, axis=1)
    ma = np.full_like(px, np.nan)
    ma[:, n - 1:] = (cs[:, n - 1:] - np.c_[np.zeros(len(px)), cs[:, :-n]]) / n
    if band == 0:
        return np.where(np.isnan(ma), np.nan, (px > ma).astype(float))
    sig = np.full_like(px, np.nan)
    cur = np.full(len(px), np.nan)
    for t in range(px.shape[1]):
        up, dn = px[:, t] > ma[:, t] * (1 + band), px[:, t] < ma[:, t] * (1 - band)
        cur = np.where(up, 1.0, np.where(dn, 0.0, cur))
        sig[:, t] = np.where(np.isnan(ma[:, t]), np.nan, cur)
    return sig


def mc_strategies(U: np.ndarray, rf_d: float, f_d: float, spread: float, er: float,
                  lag: int = EXEC_LAG) -> dict:
    """Aplica las estrategias a todas las historias a la vez (filas = historias).

    El 3x que usan las estrategias puede perder hasta el 100% en un día; si ocurre,
    la estrategia compra al día siguiente un fondo nuevo. «3x comprar y mantener»
    se trata aparte en mc_summary: tras una liquidación queda a cero."""
    x3 = np.maximum(rf_d + 3 * (U - f_d) - (3 * spread + er) / 252, -1.0)
    px = np.cumprod(1 + U, axis=1)
    sig = _sma_signal(px, 200)
    sig_b = _sma_signal(px, 200, BAND)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # medias aún sin definir al inicio
        trend = np.nanmean(np.stack([_sma_signal(px, k) for k in (50, 100, 150, 200, 250)]), axis=0)
    from scipy.signal import lfilter
    lam = 0.94
    ewma = lfilter([1 - lam], [1.0, -lam], U ** 2, axis=1,
                   zi=np.full((len(U), 1), lam * np.var(U[:, :252])))[0]
    var_fc = ewma * 252
    t_idx = np.arange(1, U.shape[1] + 1)
    mu = np.c_[np.full(len(U), np.nan), (np.cumsum(U, axis=1) / t_idx)[:, :-1]] * 252
    mu[:, : BURN * 252] = np.nan
    kel = np.clip(KELLY * (mu - f_d * 252 - spread) / var_fc / 3, 0, 1)

    def sticky_mat(target):
        out = np.full_like(target, np.nan)
        cur = np.full(len(target), np.nan)
        for t in range(target.shape[1]):
            v = target[:, t]
            move = np.isnan(cur) | (np.abs(v - cur) > MIN_STEP) | (v == 0) | (v == 1)
            cur = np.where(np.isnan(v), np.nan, np.where(move, v, cur))
            out[:, t] = cur
        return out

    def shift(w):
        return np.c_[np.zeros((len(w), lag)), np.nan_to_num(w)[:, :-lag]]

    def run(w3, w1=None):
        w3 = shift(w3)
        w1 = np.zeros_like(w3) if w1 is None else shift(w1)
        turn = np.abs(np.diff(np.c_[np.zeros(len(w3)), w3], axis=1)) + \
            np.abs(np.diff(np.c_[np.zeros(len(w1)), w1], axis=1))
        return w3 * x3 + w1 * U + (1 - w3 - w1) * rf_d - turn * COST_BPS / 1e4

    return {
        "1x comprar y mantener": U,
        "3x comprar y mantener": x3,
        "LRS 3x (original)": run(sig),
        "LRS 3x + banda ±2%": run(sig_b),
        "½ Kelly (sin tendencia)": run(sticky_mat(kel)),
        "LRS-Kelly": run(sticky_mat(kel * np.nan_to_num(sig))),
        "LRS-Kelly suavizado": run(sticky_mat(kel * np.nan_to_num(trend))),
    }


def mc_summary(S: dict, eval_from: int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rows, wealth, lws = {}, {}, {}
    years = (next(iter(S.values())).shape[1] - eval_from) / 252
    for k, r in S.items():
        r = r[:, eval_from:].copy()
        if k == "3x comprar y mantener":  # tras liquidarse, a cero para siempre
            dead = np.cumsum(r <= -1, axis=1) > 0
            first = dead & ~np.c_[np.zeros((len(r), 1), bool), dead[:, :-1]]
            r = np.where(dead & ~first, 0.0, r)
        w = np.cumprod(1 + r, axis=1)
        lw = np.log(np.maximum(w[:, -1], 1e-300))
        cagr = np.maximum(w[:, -1], 0) ** (1 / years) - 1
        dd = (w / np.maximum.accumulate(w, axis=1) - 1).min(axis=1)
        rows[k] = {"CAGR_mediana": np.median(cagr), "CAGR_p5": np.percentile(cagr, 5),
                   "CAGR_p95": np.percentile(cagr, 95), "maxDD_mediana": np.median(dd),
                   "P(caída > 90%)": (dd < -0.9).mean(), "P(pierde dinero)": (lw < 0).mean()}
        wealth[k] = np.log(np.maximum(w, 1e-300))
        lws[k] = lw
    base = lws["1x comprar y mantener"]
    for k in rows:
        rows[k]["P(gana al 1x)"] = float((lws[k] > base).mean())
    # comparaciones emparejadas (mismas historias)
    keys = list(lws)
    pair = pd.DataFrame({a: {b: float((lws[a] > lws[b]).mean()) for b in keys} for a in keys}).T
    return pd.DataFrame(rows).T, wealth, pair


def monte_carlo(d: dict, rng) -> dict:
    u = d["u"].dropna().values
    rf_d = float(d["rf"].mean())
    f_d = float(d["f"].mean())
    T = (YEARS + BURN) * 252
    results = {}
    for method, label in (("bootstrap", "bootstrap de la historia real"),
                          ("gjr", "GJR-GARCH (sin tendencias)"),
                          ("regímenes", "cambio de régimen")):
        U = gen_paths(method, u, N_PATHS, T, rng)
        S = mc_strategies(U, rf_d, f_d, d["spread"], d["er"])
        t, W, pair = mc_summary(S, BURN * 252)
        results[label] = {"tabla": t, "riqueza": W, "pares": pair,
                          "diagnóstico": world_diagnostics(U, u)}
    return results


# ───────────────────────────── figuras ─────────────────────────────

SHOW = ["1x comprar y mantener", "3x comprar y mantener", "LRS 3x (original)", "LRS-Kelly",
        "LRS-Kelly suavizado"]
COLORS = {"1x comprar y mantener": "k", "3x comprar y mantener": "0.65", "LRS 3x (original)": "C0",
          "LRS-Kelly": "C3", "LRS-Kelly suavizado": "C2", "LRS 3x + banda ±2%": "C1",
          "½ Kelly (sin tendencia)": "C4", "½ Kelly GJR (sin tendencia)": "C4"}


def fig_equity(R: dict, start, title: str, name: str, caption: str, rep: Report, end=None,
               show: list[str] | None = None, dashed: dict | None = None):
    """Riqueza y caída de cada regla. dashed = {nombre: (serie, clave cuyo color toma)}:
    líneas discontinuas adicionales (p. ej. la señal de precio con el color de la original)."""
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    lines = [(k, R[k], COLORS.get(k), "-") for k in show or SHOW]
    lines += [(k, s, COLORS.get(c), "--") for k, (s, c) in (dashed or {}).items()]
    for k, s, col, ls in lines:
        r = s.loc[start:end].dropna()
        if len(r) < 50:
            continue
        w = (1 + r).cumprod() * 10_000
        axes[0].semilogy(w, color=col, ls=ls, lw=1.6 if "Kelly" in k else 1.1,
                         label=f"{k}: {pct_es(metrics.cagr(r))}/año")
        axes[1].plot((w / w.cummax() - 1) * 100, color=col, ls=ls, lw=1)
    axes[0].set(title=title, ylabel="Valor de 10.000 $ (log)")
    axes[0].legend(loc="upper left", fontsize=11)
    axes[1].set(ylabel="Caída desde máximos (%)")
    comma_fig(fig)
    rep.fig(fig, name, caption)


def fig_1987(R_lag: dict, R_moc: dict, Rp_lag: dict, Rp_moc: dict, rep: Report):
    """La LRS 3x en octubre de 1987 con las dos señales y las dos ejecuciones."""
    fig, ax = plt.subplots(figsize=(12, 6))
    a, b = "1987-08-01", "1988-06-30"
    k3 = "3x comprar y mantener"
    r = R_lag[k3].loc[a:b].dropna()
    ax.plot((1 + r).cumprod() * 10_000, color=COLORS[k3], lw=1.6, label=k3)
    for Rl, Rm, key, lab, col in ((R_lag, R_moc, ORIGINAL, "señal de rentabilidad total (original)", "C0"),
                                  (Rp_lag, Rp_moc, PX_LRS, "señal de precio", "C1")):
        for R, ex, ls in ((Rl, "orden al cierre siguiente", "-"), (Rm, "mismo cierre", "--")):
            r = R[key].loc[a:b].dropna()
            ax.plot((1 + r).cumprod() * 10_000, ls=ls, color=col, lw=1.6, label=f"LRS 3x, {lab}, {ex}")
    ax.axvline(pd.Timestamp("1987-10-19"), color="r", lw=0.8)
    ax.text(pd.Timestamp("1987-10-19"), ax.get_ylim()[1], " 19-10-1987", color="r", va="top")
    ax.set(title="Octubre de 1987: el resultado depende de la señal y de un día de ejecución",
           ylabel="Valor de 10.000 $")
    ax.legend(loc="lower left", fontsize=10)
    rep.fig(fig, "octubre_1987", "La LRS 3x en octubre de 1987 con la señal de rentabilidad total (la original) "
            "y con la de precio, según se ejecute en el mismo cierre o al siguiente")


def fig_exposure(px: pd.Series, E: dict, start, rep: Report):
    """px: el índice de la señal original (rentabilidad total)."""
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 9.5), sharex=True)
    p = px.loc[start:]
    axes[0].semilogy(p, color="k", lw=1, label="índice con dividendos")
    axes[0].semilogy(px.rolling(200).mean().loc[start:], color="C1", lw=1, label="su media de 200 sesiones")
    axes[0].set(title="S&P 500 con dividendos y su media de 200 sesiones", ylabel="Índice (log)")
    axes[0].legend(loc="upper left")
    for ax, k in zip(axes[1:], ("LRS 3x (original)", "LRS-Kelly")):
        e = E[k].loc[start:]
        ax.fill_between(e.index, 0, e, color=COLORS[k], alpha=0.5, step="post")
        ax.set(title=f"{k}: apalancamiento efectivo", ylabel="Exposición (x)", ylim=(0, 3.2))
    comma_fig(fig)
    rep.fig(fig, "exposicion", "Apalancamiento efectivo de la LRS original y de la LRS-Kelly a lo largo del tiempo")


def fig_rolling(R: dict, start, rep: Report):
    fig, ax = plt.subplots(figsize=(12, 6))
    for k in SHOW:
        r = np.log1p(R[k].loc[start:].dropna().clip(lower=-0.999))
        roll = np.expm1(r.rolling(2520).sum() / 10) * 100
        ax.plot(roll, color=COLORS[k], lw=1.4 if "Kelly" in k else 1, label=k)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set(title="Rentabilidad anual de cada década móvil (S&P 500)", ylabel="% anual en los 10 años previos")
    ax.legend(loc="lower left")
    rep.fig(fig, "decadas_moviles", "Rentabilidad anualizada en ventanas móviles de diez años")


def fig_scatter(tables: dict, rep: Report):
    fig, axes = plt.subplots(len(tables), 1, figsize=(11, 6 * len(tables)))
    for ax, (name, t) in zip(np.atleast_1d(axes), tables.items()):
        for k, row in t.iterrows():
            ax.scatter(row.vol * 100, row.CAGR * 100, s=40 + 400 * abs(row.maxDD),
                       color=COLORS.get(k, "C5"), alpha=0.7, edgecolor="k")
            ax.annotate(k, (row.vol * 100, row.CAGR * 100), fontsize=10,
                        xytext=(6, 4), textcoords="offset points")
        ax.set(title=f"{name}: rentabilidad frente a volatilidad (tamaño = caída máxima)",
               xlabel="Volatilidad anual (%)", ylabel="Rentabilidad anual compuesta (%)")
    rep.fig(fig, "riesgo_rentabilidad", "Rentabilidad frente a volatilidad de cada estrategia; el tamaño del punto es la caída máxima")


def fig_robust(g: pd.DataFrame, rep: Report):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    for ax, col, fmt_ in ((axes[0], "sharpe", lambda v: num_es(v, 2)), (axes[1], "maxDD", lambda v: pct_es(v, 0))):
        m = g.pivot(index="media", columns="banda", values=col)
        im = ax.imshow(m.values, cmap="RdYlGn", aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, fmt_(m.values[i, j]), ha="center", va="center", fontsize=11)
        ax.set_xticks(range(m.shape[1]), [f"±{pct_es(b, 0)}" for b in m.columns])
        ax.set_yticks(range(m.shape[0]), m.index)
        ax.set(xlabel="Banda de histéresis", ylabel="Sesiones de la media",
               title={"sharpe": "Ratio de Sharpe", "maxDD": "Caída máxima"}[col])
        fig.colorbar(im, ax=ax)
    fig.suptitle("LRS 3x en el S&P 500 (señal de rentabilidad total): robustez a la media y a la banda")
    rep.fig(fig, "robustez", "Ratio de Sharpe y caída máxima de la LRS 3x (señal de rentabilidad total, la "
            "original) para cada combinación de media y banda")


def fig_mc(mc: dict, rep: Report):
    worlds = list(mc)
    fig, axes = plt.subplots(len(worlds), 1, figsize=(FIG_W, 3.6 * len(worlds) + 0.6))
    axes = np.atleast_1d(axes)
    for ax, wname in zip(axes, worlds):
        W = mc[wname]["riqueza"]
        for k in ("1x comprar y mantener", "3x comprar y mantener", "LRS 3x (original)", "LRS-Kelly"):
            lw = W[k]
            years = np.arange(lw.shape[1]) / 252
            med = np.exp(np.median(lw, axis=0)) * 10_000
            lo, hi = (np.exp(np.percentile(lw, q, axis=0)) * 10_000 for q in (10, 90))
            ax.fill_between(years, np.maximum(lo, 1), hi, color=COLORS[k], alpha=0.12)
            ax.semilogy(years, med, color=COLORS[k], lw=1.6, label=k)
        ax.set(title=f"Mundo sintético: {wname}", xlabel="Años", ylabel="Valor de 10.000 $ (log)",
               ylim=(100, None))
    h, lab = axes[0].get_legend_handles_labels()
    legend_band(fig, h, lab, ncol=2)
    fig.suptitle(f"{num_es(N_PATHS, 0)} historias sintéticas de {YEARS} años por mundo\n"
                 "(mediana y banda del 10 al 90%)")
    rep.fig(fig, "mundos_sinteticos", "Evolución mediana y banda del 10 al 90% de cada estrategia en miles de historias sintéticas")


def fig_mc_density(mc: dict, rep: Report):
    from scipy.stats import gaussian_kde
    fig, axes = plt.subplots(len(mc), 1, figsize=(11, 4.8 * len(mc)))
    xs = np.linspace(-0.4, 0.5, 400)
    for ax, (wname, res) in zip(np.atleast_1d(axes), mc.items()):
        W = res["riqueza"]
        for k in ("1x comprar y mantener", "3x comprar y mantener", "LRS 3x (original)", "LRS-Kelly"):
            cagr = np.clip(np.expm1(W[k][:, -1] / YEARS), -0.4, 0.5)
            ax.plot(xs * 100, gaussian_kde(cagr)(xs), color=COLORS[k], lw=2, label=k)
        ax.axvline(0, color="0.5", lw=0.6)
        ax.set(title=f"Mundo sintético: {wname}", xlabel=f"Rentabilidad anual compuesta a {YEARS} años (%)",
               ylabel="Densidad", yticks=[])
        ax.legend(fontsize=10)
    fig.suptitle("Distribución de la rentabilidad a 30 años en las historias sintéticas")
    rep.fig(fig, "mundos_densidad", "Distribución de la rentabilidad anual a 30 años de cada estrategia en las historias sintéticas")


# ───────────────────────────── informe ─────────────────────────────

PCTS = {"CAGR", "vol", "maxDD", "ulcer", "peor_año"}


DPCT = ["ΔCAGR", "Δcaída_máx", "Δg_anual"]


def as_pp(df: pd.DataFrame, cols, unit: str = "pp") -> pd.DataFrame:
    """Copia con `cols` ×100 y la unidad en el nombre (para mostrarlas con dos decimales)."""
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype(float) * 100
    return out.rename(columns={c: f"{c} ({unit})" for c in cols})


def _date(x) -> str:
    return pd.Timestamp(x).strftime("%-d-%-m-%Y")


def run():
    rep = Report("r13_lrs", "R13 — Leverage Rotation Strategy y sus variantes")
    rep.p(__doc__.split("\n", 2)[2])
    rng = np.random.default_rng(SEED)
    D = load()
    nd = D.pop("_ndx")

    rep.h("Parámetros del 3x sintético de cada mercado")
    pt = params_table(D)
    rep.table(as_pp(pt, ["gasto_del_fondo", "s_sobre_la_letra_R3", "s_sobre_f", "f_menos_rf_vida_ETF",
                         "f_menos_rf_toda_la_muestra"], "% anual"), digits=2)
    rep.p("El 3x sintético es rf + 3·(u − f − s) − gasto: toda su exposición se financia al tipo f "
          "(call money hasta 1954 y fondos federales después, con un mes de retraso) más el sobrecoste "
          "s del swap, y paga el gasto anual del fondo. s se calibra en R3 sobre la letra con la vida del "
          "ETF real y aquí se le resta la media de f − rf en esa vida (s_sobre_f), para que el sintético "
          "siga reproduciendo al real y antes pague la diferencia histórica f − rf. Desde su lanzamiento, "
          "el 3x es el ETF real.")

    tables, all_R, spa_rows = {}, {}, []
    for name, d in D.items():
        var_fc = var_forecast(d)
        R, E = strategies(d, var_fc)
        R_moc, _ = strategies(d, var_fc, lag=1)
        Rp, Ep = price_variants(d, var_fc)
        Rp_moc, _ = price_variants(d, var_fc, lag=1)
        approx = not d["px_real"]
        all_R[name] = (R, E, R_moc, Rp, Ep, Rp_moc)
        start = var_fc.first_valid_index()
        if d["eval_from"] is not None:
            start = max(start, d["eval_from"])
        rep.h(f"{name}: desde {start.date()} (orden al cierre siguiente)")
        t = stats_table(R, d["rf"], start)
        tables[name] = t
        rep.table(t, pct=PCTS, digits=2)
        rep.table(trading_stats(E, start), pct={"tiempo_invertido"}, digits=2)
        tag = PX_APPROX_TAG if approx else PX_TAG
        rep.h(f"Sensibilidad: {tag}", 3)
        if approx:
            rep.p(f"Sin índice de precio diario en la base: el precio aproximado es la rentabilidad total "
                  f"menos un dividendo constante del {pct_es(DIV[name], 1)} anual. Solo es una sensibilidad: "
                  "no entra en el contraste SPA.")
        else:
            rep.p("Las mismas reglas con la media de 200 sesiones sobre el índice de precio (^GSPC, con los "
                  "cierres corregidos) en lugar del de rentabilidad total. Es una variante del laboratorio, "
                  "no la regla de Gayed y Bilello; la LRS 3x con esta señal entra en la familia del SPA.")
        rep.table(stats_table(Rp, d["rf"], start), pct=PCTS, digits=2)
        rep.table(trading_stats(Ep, start), pct={"tiempo_invertido"}, digits=2)
        rep.p("Diferencia de cada regla con la señal de precio frente a la misma regla con la original "
              "(Δcaída_máx positiva: cae menos; Δg_anual: crecimiento logarítmico anual).")
        rep.table(as_pp(price_diff_table(R, Rp, d["rf"], start, approx=approx), DPCT), digits=2)
        rep.h("Sensibilidad a la ejecución: mismo cierre (MOC) frente a cierre siguiente", 3)
        t_moc = stats_table(R_moc, d["rf"], start)
        rep.table(pd.DataFrame({"CAGR_cierre_siguiente": t.CAGR, "CAGR_mismo_cierre": t_moc.CAGR,
                                "maxDD_cierre_siguiente": t.maxDD, "maxDD_mismo_cierre": t_moc.maxDD}),
                  pct={"CAGR_cierre_siguiente", "CAGR_mismo_cierre", "maxDD_cierre_siguiente",
                       "maxDD_mismo_cierre"}, digits=2)
        main, grid = family(d, R)
        for fam_name, fam in (("principal", main), ("con rejilla", grid)):
            for row in spa_tests(fam, R, start, rng):
                spa_rows.append({"mercado": name, "familia": fam_name, **row})
        rep.h(f"Periodo con ETFs reales ({REAL_FROM} → hoy)", 3)
        rep.table(stats_table(R, d["rf"], REAL_FROM), pct=PCTS, digits=2)
        rep.p(f"Con la {tag}:")
        rep.table(stats_table(Rp, d["rf"], REAL_FROM), pct=PCTS, digits=2)
        rep.table(as_pp(price_diff_table(R, Rp, d["rf"], REAL_FROM, approx=approx), DPCT), digits=2)

    rep.h("Contraste SPA (Hansen 2005) sobre la familia completa de reglas probadas")
    spa = pd.DataFrame(spa_rows)
    principal = spa.familia == "principal"
    spa.loc[principal, "p_Holm"] = stats.holm(spa.loc[principal, "p_SPA_c"].values)
    rep.table(spa.set_index(["mercado", "familia", "frente a"]), pct={"Δ_crec_anual"}, digits=3)
    n1 = spa[principal & (spa["frente a"] == "1x comprar y mantener")].set_index("mercado").reglas
    rep.p(f"Bootstrap estacionario con bloques medios de 63 sesiones y {B} remuestreos. "
          "p_Holm corrige las 9 comparaciones principales (3 mercados × 3 referencias). La familia "
          f"principal tiene {n1['S&P 500']} reglas frente al 1x en el S&P 500, entre ellas {PX_LRS}, "
          f"y {n1['Nasdaq-100']} en el Nasdaq-100 y en semis, donde la señal de precio aproximado queda "
          "fuera. «Con rejilla» añade las 30 combinaciones de media y banda sobre el índice de "
          "rentabilidad total.")

    rep.h(f"Nasdaq-100: {N_NDX} reconstrucciones del periodo 1971–85 (evaluado desde 1985-10)")
    rep.table(ndx_imputations(nd, rng), pct={"CAGR_mediana", "CAGR_mín", "CAGR_máx", "maxDD_mediana"},
              digits=2)

    R, E, R_moc, Rp, Ep, Rp_moc = all_R["S&P 500"]
    d = D["S&P 500"]
    start = R["LRS-Kelly"].first_valid_index()
    RA = {**R, **Rp}
    rep.h("S&P 500: subperiodos (orden al cierre siguiente)")
    sub_keys = SHOW[:2] + ["LRS 2x (original)"] + SHOW[2:] + ["LRS 3x + banda ±2%", "LRS 2x (señal de precio)",
                                                               PX_LRS]
    for a, b in (("1938", "1969"), ("1970", "1999"), ("2000", "2026")):
        rep.h(f"{a}–{b}", 3)
        rep.table(stats_table({k: RA[k] for k in sub_keys}, d["rf"], a, b), pct=PCTS, digits=2)
    rules = ("1x comprar y mantener", "3x comprar y mantener", "LRS 1x (original)", "LRS 2x (original)",
             "LRS 3x (original)", "LRS 3x + banda ±2%")
    rep.h("S&P 500: la Gran Depresión (1928–1938, solo reglas sin estimación)")
    rep.table(stats_table({k: RA[k] for k in rules + ("LRS 2x (señal de precio)", PX_LRS)}, d["rf"],
                          FIRST_RULE_DAY, "1938-12-31"), pct=PCTS, digits=2)

    rep.h("S&P 500: ventanas de inicio fijadas (orden al cierre siguiente)")
    starts = {"primera sesión de la LRS": FIRST_RULE_DAY,
              "primer día con previsión GJR": start,
              "1974 (1 de enero)": "1974-01-02", "mínimo de 1973–74": "1974-10-03",
              "2000": "2000-01-03", "ETFs reales": REAL_FROM,
              "tras la muestra de Gayed y Bilello": "2016-01-04",
              "primer año completo tras la publicación": "2017-01-03"}
    wkeys = ["1x comprar y mantener", "3x comprar y mantener", "LRS 1x (original)", "LRS 2x (original)",
             ORIGINAL, "LRS 2x (señal de precio)", PX_LRS, "LRS-Kelly", "½ Kelly GJR (sin tendencia)"]
    wt = windows_table(RA, wkeys, starts)
    rep.table(wt, pct={"CAGR", "caída_máx"}, index=False, digits=2)
    lrs2 = wt[wt.regla == "LRS 2x (original)"].set_index("inicio")
    one = wt[wt.regla == "1x comprar y mantener"].set_index("inicio")
    both = [i for i in lrs2.index if lrs2.loc[i, "rinde_más_que_1x"] == "sí" and lrs2.loc[i, "cae_menos_que_1x"] == "sí"]
    worse = lrs2[lrs2.cae_menos_que_1x == "no"]
    episodes = [f"{pct_es(dd)} entre el {_date(pk)} y el {_date(tr)} (ventanas «{'», «'.join(wins)}»; "
                f"el 1x, {' / '.join(dict.fromkeys(pct_es(one.loc[i, 'caída_máx']) for i in wins))})"
                for dd, pk, tr, wins in drawdown_episodes(worse)]
    rep.p("Cada fila empieza en la primera sesión de su ventana y acaba en la última de la muestra; la caída "
          "máxima lleva la fecha de su máximo previo y la de su mínimo, y las dos últimas columnas comparan con "
          "el 1x en la misma ventana. Las reglas con estimación (LRS-Kelly, ½ Kelly GJR) solo aparecen desde "
          f"1938. La LRS 2x (original) rinde más que el 1x y cae menos en {len(both)} de las {len(lrs2)} "
          f"ventanas ({', '.join(f'«{i}»' for i in both) or 'ninguna'}). "
          + (f"Cae más que el 1x en las otras {len(worse)}: " + "; ".join(episodes) + "." if len(worse) else ""))

    rep.h("Conciliación con Gayed y Bilello (2016)")
    gb = gb_reconciliation(d)
    sig0 = max(above_sma(signal_index(d, k)).first_valid_index() for k in (SIGNAL, PRICE))
    rep.table(gb, pct={"CAGR", "caída_máx"}, digits=2)
    rep.p(f"Cifras publicadas por Gayed y Bilello (2016, tabla 8), de octubre de 1928 a octubre de 2015: índice "
          f"{pct_es(GB_TABLE8['índice (S&P 500)'])}, 3x comprar y mantener {pct_es(GB_TABLE8['3x comprar y mantener'])}, "
          f"LRS 3x {pct_es(GB_TABLE8['LRS 3x'])} con una caída máxima del {pct_es(GB_TABLE8['LRS 3x, caída máxima'])}. "
          "Sus supuestos: una comisión del 1% anual como único coste del apalancamiento («a leverage fee of "
          "1% per year»), sin intereses por lo prestado; la media, sobre la serie de rentabilidad total, y "
          "ningún retraso de ejecución explícito. Los costes de operar no los mencionan en la tabla 8 (su "
          "«no cost to execute» se refiere a otro análisis, la media de 200 sesiones sin apalancar de su "
          "tabla 4); que la tabla 8 no los descuenta se deduce de que P1, sin ellos, queda cerca de su cifra, y "
          "reconocen que desde 1928 ejecutar la estrategia «would have been hampered by higher transaction "
          "costs». Aquí la ventana empieza el "
          f"{_date(gb.desde.iloc[0])}, primera sesión de la LRS con la orden al cierre siguiente (las medias de "
          f"200 sesiones de las dos señales existen desde el cierre del {_date(sig0)}), y acaba el "
          f"{_date(gb.hasta.iloc[0])}. P1 reproduce sus supuestos con nuestros datos; cada paso siguiente añade "
          "una fricción del laboratorio, y P5 es la LRS 3x (original) de este informe en esa ventana. P4′ y P5′ "
          "son P4 y P5 con la señal de precio.")

    rep.h("Sensibilidad: la señal sobre el SPY desde 1993 (versión anterior del laboratorio)")
    sens, n_diff = legacy_signal_table(d, R, start)
    rep.table(sens, pct={"CAGR", "maxDD"}, index=False, digits=2)
    rep.p("La LRS 3x (original) calcula su media sobre el índice oficial con dividendos ^SP500TR desde "
          "1988 (la serie de Gayed y Bilello). Hasta esta revisión, desde 1993 la calculaba sobre u, el ETF "
          "SPY ajustado por dividendos, que descuenta su comisión y mantiene en efectivo los dividendos hasta "
          f"repartirlos: la señal cambia en {n_diff} sesiones. Las carteras y el índice (1x) se miden con u "
          "en las dos versiones; Δg_frente_1x_pp: crecimiento log anual de la regla menos el del 1x en la "
          "misma ventana, en puntos. El contraste de la hipótesis principal (Δg frente al 1x con bootstrap) "
          "está en R19.")

    rep.h("Coste por operación (S&P 500, desde el primer día con previsión GJR)")
    cost_info = {}
    for lab, kind in (("original (rentabilidad total)", SIGNAL), ("señal de precio", PRICE)):
        ct, info = cost_table(d, above_sma(signal_index(d, kind)), start)
        cost_info[kind] = info
        rep.h(f"LRS 3x, {lab}", 3)
        rep.table(as_pp(ct, ["Δg_frente_1x", "Δg_frente_L_constante"]), pct={"CAGR"}, digits=2)
        rep.p(f"{num_es(info['cambios_año'], 1)} cambios al año; L constante de su exposición media "
              f"{num_es(info['L_constante'], 2)} (L/3 en el 3x y el resto en letras, reequilibrado a diario sin "
              f"coste). El coste por unidad de rotación que anula la ventaja sobre el 1x: "
              f"{fmt_cost(info['coste_anula_Δg_1x'])}; sobre el L constante de su exposición: "
              f"{fmt_cost(info['coste_anula_Δg_L'])}.")
    rep.p(f"El laboratorio cobra {COST_BPS:g} pb por unidad de rotación (un cambio entre el 3x y las letras "
          "es una unidad). Es un coste de hoy: antes de 1975 las comisiones eran fijas, y Jones (2002) estima "
          "costes de ida del orden de medio punto a un punto porcentual durante buena parte del siglo XX. "
          "Hasta los futuros sobre el S&P 500 (1982) y los ETFs apalancados (UPRO, 2009) la regla no era "
          "ejecutable tal como se simula.")

    rep.h("Octubre de 1987: distancia de cada señal a su media de 200 sesiones")
    dist = distance_1987(d)
    rep.table(dist, pct=set(dist.columns), digits=2)
    first = {c: dist.index[(dist[c] < 0).values][0] for c in dist}
    c_tr, c_px = dist.columns
    crash = pd.Timestamp("1987-10-19")
    loss = {(s, lag): Rx[k].loc[crash] for s, (Rl, Rm, k) in
            (("tr", (R, R_moc, ORIGINAL)), ("px", (Rp, Rp_moc, PX_LRS)))
            for lag, Rx in ((2, Rl), (1, Rm))}
    rep.p(f"La señal de precio pasa a letras al cierre del {_date(first[c_px])} "
          f"({pct_es(-dist.loc[first[c_px], c_px], 2)} bajo su media) y la de rentabilidad total al del "
          f"{_date(first[c_tr])} (la víspera estaba {pct_es(dist[c_tr].shift(1).loc[first[c_tr]], 2)} sobre la "
          f"suya). El lunes 19 de octubre, con la orden al cierre siguiente, la LRS 3x (original) rinde "
          f"{pct_es(loss[('tr', 2)], 2)} y la de precio {pct_es(loss[('px', 2)], 2)}; en el mismo cierre, "
          f"{pct_es(loss[('tr', 1)], 2)} y {pct_es(loss[('px', 1)], 2)} (fuera del mercado cobra la letra "
          "menos el coste de la rotación).")

    rep.h("S&P 500: rejilla de robustez de la LRS 3x")
    g_tr = robust_grid(d, signal_index(d, SIGNAL), start)
    g_px = robust_grid(d, signal_index(d, PRICE), start)
    rep.h("Señal de rentabilidad total (la original)", 3)
    rep.table(g_tr.set_index(["media", "banda"]), pct={"CAGR", "maxDD"}, digits=2)
    rep.h("Señal de precio (sensibilidad)", 3)
    rep.table(g_px.set_index(["media", "banda"]), pct={"CAGR", "maxDD"}, digits=2)
    ranks = pd.DataFrame({"señal de rentabilidad total": grid_rank(g_tr), "señal de precio": grid_rank(g_px)}).T
    rep.p(f"Puesto de la media de 200 sesiones sin banda entre las {len(g_tr)} combinaciones (1 = la mejor; en "
          "caída máxima, la menos profunda):")
    rep.table(ranks, digits=0)

    fig_equity(R, start, "S&P 500 desde 1938: 10.000 $ en cada estrategia", "curvas_sp500",
               "Evolución de 10.000 dólares y caída desde máximos de cada estrategia en el S&P 500; en "
               "discontinua, la LRS 3x con la señal de precio", rep, dashed={PX_LRS: (Rp[PX_LRS], ORIGINAL)})
    fig_equity(RA, FIRST_RULE_DAY, "La Gran Depresión: 1928–1938", "gran_depresion",
               "Las estrategias sin parámetros estimados durante el crash de 1929 y la Gran Depresión", rep,
               end="1938-12-31", show=list(rules))
    fig_1987(R, R_moc, Rp, Rp_moc, rep)
    for name, fname in (("Nasdaq-100", "curvas_nasdaq"), ("Semis", "curvas_semis")):
        Rn = all_R[name][0]
        st = max(Rn["LRS-Kelly"].first_valid_index(), D[name]["eval_from"] or Rn["LRS-Kelly"].first_valid_index())
        fig_equity(Rn, st, f"{name}: 10.000 $ en cada estrategia", fname,
                   f"Evolución de 10.000 dólares en el {name} con cada estrategia", rep)
    fig_exposure(signal_index(d), E, start, rep)
    fig_rolling(R, start, rep)
    fig_scatter(tables, rep)
    fig_robust(g_tr, rep)

    rep.h(f"Mundos sintéticos: {num_es(N_PATHS, 0)} historias de {YEARS} años del S&P 500 por mundo")
    mc = monte_carlo(d, rng)
    for wname, res in mc.items():
        rep.h(wname, 3)
        rep.table(pd.Series(res["diagnóstico"]).to_frame("valor"), digits=3)
        rep.table(res["tabla"], pct=set(res["tabla"].columns), digits=3)
        rep.p("Comparación emparejada (mismas historias): probabilidad de que la estrategia "
              "de la fila acabe con más riqueza que la de la columna.")
        rep.table(res["pares"], pct=set(res["pares"].columns), digits=2)
    rep.p("En los mundos sintéticos la señal es la media de 200 sesiones del índice simulado con dividendos "
          "(la misma que la LRS 3x original en la historia) y la varianza de las estrategias Kelly se prevé "
          "con EWMA (λ = 0,94); la media de rentabilidad se aprende en cada historia a partir de su año 10.")
    fig_mc(mc, rep)
    fig_mc_density(mc, rep)
    rep.save()


if __name__ == "__main__":
    run()
