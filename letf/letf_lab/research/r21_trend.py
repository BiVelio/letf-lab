"""R21 — Señales de tendencia más allá de la media de 200 sesiones.

Pregunta (artículo 3). La LRS de Gayed y Bilello (2016), la «LRS 3x (original)» de
R13, mira si el S&P 500 con dividendos cierra por encima de su media simple de 200
sesiones. ¿Lo hacen mejor otras señales de tendencia populares: la media
exponencial de 200 (EMA), el MACD, los cruces de medias, las rupturas de Donchian o
combinaciones de la media con el MACD? ¿Protegen mejor en los cracs y a qué precio?

Señales (doce, fijadas antes de mirar los resultados de este módulo; todas sobre
el mismo índice que la LRS original: la rentabilidad total del S&P 500, ^SP500TR
desde 1988 y ^GSPC más el D/P de Shiller del mes anterior antes; en el Nasdaq-100,
su índice con dividendos):
- EMA 200: dentro si el índice cierra sobre su media exponencial de 200 sesiones
  (α = 2/201); y con banda ±2% (entra al superar la media un 2%, sale al caer un 2%
  por debajo), como la «LRS 3x + banda ±2%» de R13.
- MACD(12, 26, 9) de Appel: EMA de 12 menos EMA de 26; dentro si es positivo
  («MACD>0») o si supera su propia EMA de 9 («MACD>señal»). Diario y semanal (con
  el último cierre de cada semana; el valor rige desde ese cierre, o desde la primera
  sesión de la semana siguiente si la semana acabó antes por un cierre imprevisto).
- Cruce dorado y de la muerte: dentro si la media de 50 sesiones está sobre la de
  200 (simples o exponenciales).
- Rupturas de Donchian con cierres: entra al cerrar por encima del máximo de los n
  cierres anteriores y sale al cerrar por debajo del mínimo de los m anteriores:
  252/126 (máximos de 52 semanas, mínimos de 26) y 55/20 (el sistema 2 de las
  «tortugas»).
- Combinaciones con la media de 200: «SMA 200 y MACD>0» (dentro solo si las dos lo
  dicen) y «SMA 200 o MACD>0» (dentro si lo dice cualquiera).
Cada señal, en dos patas: 3x ↔ letras del Tesoro (como la LRS) y 3x ↔ 1x (como la
«LRS rotación 3x/1x»). Con la gemela de rotación de la banda ±2% sobre la SMA, que
R13 no tenía, son 25 variantes nuevas; las referencias (LRS 3x original, su rotación
y su banda, 1x y 3x comprar y mantener) son las de R13 y ya están en el censo de R19.

Mercados: S&P 500 desde 1938 (la ventana del artículo 2), Nasdaq-100 desde 1985-10 en las
10 reconstrucciones de R13, semiconductores desde 2004-05-06 (como R13) y los ETFs reales
(UPRO, TQQQ, SOXL) desde 2010-02-12.

Protocolo (el de R13 y el artículo 2): señal al cierre de t, orden al cierre de t+1
(r13.EXEC_LAG), 3x sintético financiado a f + s hasta el ETF real (UPRO, TQQQ), la
caja en letras, 5 pb por unidad de rotación, todo con r13.run_weights. Como
sensibilidad, las 25 al mismo cierre (MOC), que antes de 1990 no era operable.

Medidas:
- Δg: diferencia de crecimiento logarítmico anual (media diaria de ln(1 + r) ×252)
  frente a la LRS 3x original y frente al 1x, con el bootstrap estacionario
  emparejado (bloque medio de 63 sesiones, 2.000 remuestreos, los mismos índices para
  todas las reglas de una ventana): IC percentil del 90% y p unilateral (H0: Δg ≤ 0)
  con el error típico bootstrap, como r19.boot_test.
- Búsqueda: SPA de Hansen (stats.spa_test, con su recentrado) sobre la familia,
  Holm y el step-down de Romano y Wolf con los mismos remuestreos.
- Salidas: cada tramo fuera del 3x vale Σ ln(1 + alternativa) − Σ ln(1 + 3x) en el
  tramo. Positivo, salida acertada; negativo o cero, latigazo. Esos valores más el
  coste de operar suman exactamente la diferencia de log-riqueza con el 3x comprar y
  mantener.
- Cracs: 1987 y 2020 día a día, y los mercados bajistas del S&P 500 (caídas de al
  menos un 19% desde un máximo histórico del índice de precio, del máximo al mínimo y
  hasta recuperar el máximo).
- Operaciones y tenencia: ventas del 3x al año, días naturales entre la compra y la
  venta, y parte de las ventas con más de un año (largo plazo en EE. UU.).
- Mecanismo: L* aproximado de Kelly (kelly.lstar_approx) en los días dentro y fuera
  de cada señal, y rentabilidad y varianza de las 21 sesiones siguientes a cada estado
  (MCO con errores de Hansen-Hodrick, stats.hac_overlap(21)).
"""

from __future__ import annotations

import logging
import warnings
import zlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy.stats import norm

from letf_lab import kelly, metrics, stats
from letf_lab.data import history
from letf_lab.report import FIG_W, Report, comma_fig, legend_band, num_es, pct_es
from letf_lab.research import r12_long_ml as r12
from letf_lab.research import r13_lrs as r13

log = logging.getLogger(__name__)

SEED = 21
B = 2000                   # remuestreos del bootstrap y del SPA
BLOCK = 63                 # bloque medio del bootstrap estacionario (sesiones)
LAG = r13.EXEC_LAG         # señal al cierre de t, orden al cierre de t+1
H_FWD = 21                 # horizonte de la regresión de rentabilidad y varianza futuras
DEPTH = 0.19               # umbral de mercado bajista (caída desde máximo histórico)
FAST = 75                  # crac rápido: ≤ 75 sesiones del máximo al mínimo
LONG_TERM_DAYS = 365       # venta a largo plazo: más de un año natural
SHORT_OUT = 21             # tramo fuera «corto»: ≤ 21 sesiones

LEG_TB, LEG_1X = "3x/letras", "3x/1x"
LEGS = (LEG_TB, LEG_1X)
ONE, THREE = "1x comprar y mantener", "3x comprar y mantener"
LRS = r13.ORIGINAL                         # «LRS 3x (original)»
ROT = "LRS rotación 3x/1x"
BANDED = "LRS 3x + banda ±2%"
SMA_BAND_ROT = f"SMA 200 ±2% · {LEG_1X}"   # la gemela de rotación de la banda: nueva
REF_SIGNALS = ("SMA 200", "SMA 200 ±2%")
SIGNALS = ("EMA 200", "EMA 200 ±2%", "MACD>0", "MACD>señal", "MACD semanal>0", "MACD semanal>señal",
           "Cruce SMA 50/200", "Cruce EMA 50/200", "Donchian 252/126", "Donchian 55/20",
           "SMA 200 y MACD>0", "SMA 200 o MACD>0")
ALL_SIGNALS = REF_SIGNALS + SIGNALS
SAME_CLOSE = "mismo cierre"

# Ventanas (S&P 500). El inicio principal es el del artículo 2 (primer día con previsión
# GJR en R13: 1938-02-04); las reglas de este módulo no estiman nada.
SUBPERIODS = (("1938–1973", None, "1973-12-31"), ("1974–1999", "1974-01-01", "1999-12-31"),
              ("2000–2026", "2000-01-01", None))
POST = ("2016–2026", "2016-01-04", None)   # tras la muestra de Gayed y Bilello (acaba en 2015)
NDX_WINDOWS = (("1985–1999", r12.NDX_EVAL_FROM, "1999-12-31"), ("2000–2026", "2000-01-01", None),
               ("2016–2026", "2016-01-04", None))
# Cierres imprevistos de la bolsa de Nueva York en días laborables (anunciados el mismo día o
# después del cierre anterior): el huracán Gloria (viernes 27-9-1985), el 11-S (del martes 11 al
# viernes 14 de septiembre de 2001) y el huracán Sandy (29 y 30-10-2012). Los demás cierres
# excepcionales de la historia (funerales de Estado, 1968) se anunciaron con días de antelación.
# Solo los dos primeros dejan una semana del índice sin su sesión del viernes.
UNSCHEDULED_CLOSURES = ("1985-09-27", "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",
                        "2012-10-29", "2012-10-30")
CRASH_DAYS = {"1987": "1987-10-19", "2020": "2020-03-16"}   # los peores días de cada crac
CRASH_VIEW = {"1987": ("1987-08-03", "1988-06-30"), "2020": ("2020-01-15", "2020-09-30")}

# Fuentes de cada señal. Las referencias bibliográficas no se han comprobado en línea en
# esta revisión (sin acceso a la red): se marcan como tales en el informe.
SOURCES = {
    "SMA 200": ("Gayed y Bilello (2016), «Leverage for the Long Run»: la LRS 3x original de R13",
                "publicada"),
    "SMA 200 ±2%": ("R13: banda de histéresis fijada a priori en el artículo 2", "laboratorio"),
    "EMA 200": ("variante habitual de la SMA 200 en foros y plataformas; sin fuente canónica",
                "práctica"),
    "EMA 200 ±2%": ("la banda de R13 aplicada a la EMA", "laboratorio"),
    "MACD>0": ("Appel, G. (2005), Technical Analysis: Power Tools for Active Investors "
               "(MACD de finales de los años 70); parámetros 12/26/9 estándar", "publicada, sin comprobar"),
    "MACD>señal": ("ídem: cruce del MACD con su media de 9", "publicada, sin comprobar"),
    "MACD semanal>0": ("el mismo MACD sobre cierres semanales, uso habitual en plataformas",
                       "práctica"),
    "MACD semanal>señal": ("ídem", "práctica"),
    "Cruce SMA 50/200": ("«golden cross / death cross», regla popular; Brock, Lakonishok y LeBaron "
                         "(1992, Journal of Finance) contrastan cruces de medias cortas y largas "
                         "(1–200, 2–200, 5–150) en el Dow Jones", "práctica; literatura sin comprobar"),
    "Cruce EMA 50/200": ("el cruce anterior con medias exponenciales", "práctica"),
    "Donchian 252/126": ("ruptura de máximos de 52 semanas y salida en mínimos de 26; "
                         "parámetros fijados por el laboratorio", "laboratorio"),
    "Donchian 55/20": ("sistema 2 de las «tortugas» (Faith, C. 2007, Way of the Turtle), aquí con "
                       "cierres en vez de máximos y mínimos del día", "publicada, sin comprobar"),
    "SMA 200 y MACD>0": ("combinación fijada por el laboratorio en la exploración previa",
                         "laboratorio, a posteriori posible"),
    "SMA 200 o MACD>0": ("ídem", "laboratorio, a posteriori posible"),
}


# ───────────────────────────── señales ─────────────────────────────

def ema(x: pd.Series, n: int, min_periods: int | None = None) -> pd.Series:
    """Media exponencial con α = 2/(n + 1), recursiva desde el primer dato (adjust=False)."""
    return x.ewm(span=n, adjust=False, min_periods=n if min_periods is None else min_periods).mean()


def above_ema(x: pd.Series, n: int = 200, band: float = 0.0) -> pd.Series:
    """Como r13.above_sma, con la media exponencial de n sesiones: 1 encima, 0 debajo; con
    banda, solo cambia al cruzar ±band y mantiene el estado entre medias."""
    ma = ema(x, n)
    if band == 0:
        return (x > ma).astype(float).where(ma.notna())
    state = np.where(x > ma * (1 + band), 1.0, np.where(x < ma * (1 - band), 0.0, np.nan))
    return pd.Series(state, index=x.index).ffill().where(ma.notna())


def macd(x: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> tuple[pd.Series, pd.Series]:
    """MACD de Appel: EMA(fast) − EMA(slow) y su EMA de `sig` periodos (la «señal»).
    Válido desde la observación slow + sig (antes, NaN)."""
    m = ema(x, fast, 0) - ema(x, slow, 0)
    s = ema(m, sig, 0)
    ok = pd.Series(np.arange(len(x)) >= slow + sig, index=x.index)
    return m.where(ok), s.where(ok)


def macd_states(x: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(MACD > 0, MACD > señal) como 1/0, NaN antes de ser válidos."""
    m, s = macd(x)
    return (m > 0).astype(float).where(m.notna()), (m > s).astype(float).where(s.notna())


def last_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    """True en la última sesión de cada semana (lunes a viernes) del índice, vista a posteriori.
    La última sesión del índice cuenta como fin de semana (la serie acaba ahí)."""
    per = index.to_period("W-FRI")
    return np.asarray(np.r_[per[1:] != per[:-1], True], bool)


def week_ends(index: pd.DatetimeIndex) -> np.ndarray:
    """True en las sesiones que, AL CIERRE, se sabe que son las últimas de su semana: la última
    de cada semana del índice, salvo si detrás quedan, en esa misma semana, días laborables sin
    sesión por un cierre imprevisto (UNSCHEDULED_CLOSURES). Los festivos se conocen de antemano
    (el jueves anterior a Viernes Santo es fin de semana); el lunes 10-9-2001 no lo era: nadie
    sabía al cierre que la bolsa no abriría hasta el lunes 17."""
    last = last_of_week(index)
    bad = pd.DatetimeIndex(UNSCHEDULED_CLOSURES)
    out = last.copy()
    for i in np.flatnonzero(last & (index.weekday < 4)):
        t = index[i]
        rest = pd.bdate_range(t + pd.Timedelta(days=1), t + pd.Timedelta(days=4 - t.weekday()))
        if rest.isin(bad).any():
            out[i] = False
    return out


def weekly_macd_states(x: pd.Series) -> tuple[pd.Series, pd.Series]:
    """MACD(12, 26, 9) sobre los cierres semanales (última sesión de cada semana). El valor
    calculado con el cierre de esa sesión rige desde que se sabe que es el cierre de la semana:
    desde esa misma sesión si se sabía de antemano (week_ends) y, si la semana acabó antes por un
    cierre imprevisto, desde la primera sesión de la semana siguiente. El día a día no mira el
    futuro."""
    last = last_of_week(x.index)
    known = week_ends(x.index)
    a, b = macd_states(x[last])
    pos = np.flatnonzero(last)
    eff = np.where(known[pos], pos, pos + 1)          # sesión desde la que rige cada valor semanal
    ok = eff < len(x)
    when = x.index[eff[ok]]
    out = []
    for s in (a, b):
        s = pd.Series(s.to_numpy()[ok], index=when)
        out.append(s[~s.index.duplicated(keep="last")].reindex(x.index).ffill())
    return out[0], out[1]


def cross(x: pd.Series, fast: int = 50, slow: int = 200, kind: str = "sma") -> pd.Series:
    """Cruce dorado / de la muerte: 1 si la media rápida está sobre la lenta."""
    if kind == "sma":
        f, s = x.rolling(fast).mean(), x.rolling(slow).mean()
    elif kind == "ema":
        f, s = ema(x, fast), ema(x, slow)
    else:
        raise ValueError(kind)
    return (f > s).astype(float).where(f.notna() & s.notna())


def donchian(x: pd.Series, n_in: int, n_out: int) -> pd.Series:
    """Ruptura de Donchian con cierres: entra al cerrar por encima del máximo de los n_in
    cierres anteriores y sale al cerrar por debajo del mínimo de los n_out anteriores;
    entre medias mantiene el estado. Fuera hasta la primera entrada; NaN mientras no haya
    n_in cierres anteriores."""
    hi = x.shift(1).rolling(n_in).max()
    lo = x.shift(1).rolling(n_out).min()
    ev = np.where(x > hi, 1.0, np.where(x < lo, 0.0, np.nan))
    s = pd.Series(ev, index=x.index).ffill().fillna(0.0)
    return s.where(hi.notna() & lo.notna())


def signals(x: pd.Series) -> dict[str, pd.Series]:
    """Las 14 señales (dos de referencia y las 12 nuevas) sobre el índice x, como 1/0."""
    sma = r13.above_sma(x)
    m0, ms = macd_states(x)
    w0, ws = weekly_macd_states(x)
    both = pd.concat([sma, m0], axis=1)
    return {
        "SMA 200": sma,
        "SMA 200 ±2%": r13.above_sma(x, band=r13.BAND),
        "EMA 200": above_ema(x),
        "EMA 200 ±2%": above_ema(x, band=r13.BAND),
        "MACD>0": m0,
        "MACD>señal": ms,
        "MACD semanal>0": w0,
        "MACD semanal>señal": ws,
        "Cruce SMA 50/200": cross(x, kind="sma"),
        "Cruce EMA 50/200": cross(x, kind="ema"),
        "Donchian 252/126": donchian(x, 252, 126),
        "Donchian 55/20": donchian(x, 55, 20),
        "SMA 200 y MACD>0": both.min(axis=1, skipna=False),
        "SMA 200 o MACD>0": both.max(axis=1, skipna=False),
    }


def rule_name(sig: str, leg: str) -> str:
    """Nombre de la regla: las de la SMA 200 conservan los de R13."""
    if sig == "SMA 200":
        return LRS if leg == LEG_TB else ROT
    if sig == "SMA 200 ±2%":
        return BANDED if leg == LEG_TB else SMA_BAND_ROT
    return f"{sig} · {leg}"


RULES = tuple(rule_name(s, leg) for s in ALL_SIGNALS for leg in LEGS)       # 28 reglas
NAMES = (ONE, THREE) + RULES                                                # + comprar y mantener
NEW = tuple(k for k in RULES if k not in (LRS, ROT, BANDED))               # 25 variantes nuevas
TB_RULES = tuple(rule_name(s, LEG_TB) for s in ALL_SIGNALS)
ROT_RULES = tuple(rule_name(s, LEG_1X) for s in ALL_SIGNALS)
SIGNAL_OF = {rule_name(s, leg): s for s in ALL_SIGNALS for leg in LEGS}


def short(name: str) -> str:
    """Nombre corto para figuras y textos: la señal (con «(LRS)» en la original)."""
    if name == LRS:
        return "SMA 200 (LRS original)"
    if name == BANDED:
        return "SMA 200 ±2%"
    return SIGNAL_OF.get(name, name)


# ───────────────────────────── carteras ─────────────────────────────

def build(d: dict, lag: int = LAG) -> tuple[dict, dict, dict]:
    """Rentabilidades diarias (R), pesos objetivo sin desfasar (W: (w3, w1)) y señales (S)
    de las 28 reglas y de comprar y mantener, con r13.run_weights sobre el índice de la
    señal original (r13.signal_index: rentabilidad total)."""
    S = signals(r13.signal_index(d))
    R = {ONE: d["u"], THREE: d["x3"]}
    W: dict[str, tuple] = {}
    for sig in ALL_SIGNALS:
        s = S[sig]
        for leg in LEGS:
            w1 = 1 - s if leg == LEG_1X else None
            k = rule_name(sig, leg)
            R[k] = r13.run_weights(d, s, w1=w1, lag=lag)
            W[k] = (s, w1)
    return R, W, S


def held(W: dict, name: str, lag: int = LAG) -> pd.Series:
    """Peso en el 3x que cobra cada día (el objetivo desfasado `lag` sesiones)."""
    return W[name][0].shift(lag)


# ───────────────────────────── estadística ─────────────────────────────

def rng_for(tag: str) -> np.random.Generator:
    """Generador determinista propio de cada sección: los resultados no dependen del orden."""
    return np.random.default_rng([SEED, zlib.crc32(tag.encode())])


def log_frame(R: dict, names, start, end=None) -> pd.DataFrame:
    """ln(1 + r) de cada regla en la ventana, solo con los días en que todas existen."""
    return pd.DataFrame({k: np.log1p(R[k].loc[start:end].clip(lower=-0.9999)) for k in names}).dropna()


def boot_draws(X: np.ndarray, rng, B_: int = B, block: float = BLOCK) -> np.ndarray:
    """B × K medias de un bootstrap estacionario con los MISMOS índices para todas las columnas
    (stats.stationary_indices): cualquier diferencia entre dos reglas queda emparejada."""
    X = np.asarray(X, float)
    T = len(X)
    out = np.empty((B_, X.shape[1]))
    for b in range(B_):
        out[b] = X[stats.stationary_indices(T, block, rng)].mean(axis=0)
    return out


def paired(L: pd.DataFrame, draws: np.ndarray, bench: str, cands) -> tuple[pd.DataFrame, np.ndarray]:
    """Δg anual de cada candidata frente a `bench` con los remuestreos `draws` (de L):
    IC percentil del 90%, t = Δg/EE* y p unilateral normal (H0: Δg ≤ 0), como r19.boot_test.
    Devuelve también los t* centrados (B × K) para Romano–Wolf."""
    cols = list(L.columns)
    j = cols.index(bench)
    c = [cols.index(k) for k in cands]
    m = L.to_numpy(float).mean(axis=0)
    dm = m[c] - m[j]
    dd = draws[:, c] - draws[:, [j]]
    se = dd.std(axis=0, ddof=1)
    safe = np.where(se > 0, se, np.inf)
    t = dm / safe
    lo, hi = np.percentile(dd, [5, 95], axis=0)
    out = pd.DataFrame({"Δg": dm * 252, "IC90_bajo": lo * 252, "IC90_alto": hi * 252,
                        "t": t, "p": norm.sf(t)}, index=list(cands))
    return out, (dd - dm) / safe


def romano_wolf(t_obs, t_boot) -> np.ndarray:
    """Step-down de Romano y Wolf (2005) para H0_k: Δg_k ≤ 0 (una cola), con los t* centrados
    de los mismos remuestreos (misma fórmula que r19.romano_wolf, sin importar R19)."""
    t_obs = np.asarray(t_obs, float)
    tb = np.asarray(t_boot, float)
    B_, K = tb.shape
    order = np.argsort(-t_obs, kind="stable")
    tail = np.maximum.accumulate(tb[:, order][:, ::-1], axis=1)[:, ::-1]
    p = (1 + (tail >= t_obs[order]).sum(axis=0)) / (1 + B_)
    p = np.minimum(np.maximum.accumulate(p), 1.0)
    out = np.empty(K)
    out[order] = p
    return out


def spa(L: pd.DataFrame, bench: str, cands, rng, label: str = "") -> dict:
    """SPA de Hansen (stats.spa_test, versión consistente) de `cands` frente a `bench`."""
    cands = [c for c in cands if c != bench]
    D = L[cands].sub(L[bench], axis=0).to_numpy(float)
    res = stats.spa_test(D, BLOCK, B, rng)
    mean = D.mean(axis=0)
    best = int(mean.argmax())
    return {"familia": label, "frente a": bench, "reglas": len(cands), "mejor": cands[best],
            "Δg_mejor": mean[best] * 252, "p_SPA_c": res["p_SPA_c"], "p_SPA_l": res["p_SPA_l"]}


def spa_sets(extra_three: bool = False) -> list[tuple[str, str, tuple]]:
    """Familias del SPA: (rótulo, referencia, candidatas)."""
    sets = [("25 nuevas frente a la LRS original", LRS, NEW),
            ("25 nuevas + rotación y banda de R13 frente a la LRS original", LRS, NEW + (ROT, BANDED)),
            ("25 nuevas + las 3 de R13 frente al 1x", ONE, NEW + (LRS, ROT, BANDED)),
            ("13 rotaciones nuevas frente a la LRS rotación 3x/1x", ROT,
             tuple(k for k in ROT_RULES if k != ROT))]
    if extra_three:
        sets.append(("25 nuevas + las 3 de R13 frente al 3x comprar y mantener", THREE,
                     NEW + (LRS, ROT, BANDED)))
    return sets


def trading(W: dict, names, idx: pd.DatetimeIndex, lag: int = LAG) -> pd.DataFrame:
    """Cambios de exposición al año y tiempo en el 3x, con los pesos que cobra cada día."""
    rows = {}
    years = len(idx) / 252
    for k in names:
        if k in (ONE, THREE):
            rows[k] = {"cambios_año": 0.0, "tiempo_en_3x": float(k == THREE)}
            continue
        w3, w1 = W[k]
        e = (3 * w3 + (w1 if w1 is not None else 0)).shift(lag).reindex(idx)
        rows[k] = {"cambios_año": float((e.diff().abs() > 1e-9).sum() / years),
                   "tiempo_en_3x": float(held(W, k, lag).reindex(idx).mean())}
    return pd.DataFrame(rows).T


def evaluate(R: dict, W: dict, d: dict, start, end, tag: str, names=NAMES, spa_families=None,
             multiple: bool = True) -> dict:
    """Una ventana: métricas, Δg frente a la LRS original y al 1x (bootstrap emparejado),
    Holm y Romano–Wolf de las nuevas y el SPA de cada familia de `spa_families`."""
    rng = rng_for(tag)
    L = log_frame(R, names, start, end)
    draws = boot_draws(L.to_numpy(float), rng)
    t = r13.stats_table({k: R[k].reindex(L.index) for k in names}, d["rf"], L.index[0], L.index[-1])
    t = t[["CAGR", "vol", "sharpe", "maxDD", "peor_año"]].copy()
    t["g"] = L.mean() * 252
    t = t.join(trading(W, names, L.index))
    vl, tbl = paired(L, draws, LRS, [k for k in names if k != LRS])
    v1, tb1 = paired(L, draws, ONE, [k for k in names if k != ONE])
    t = t.join(vl.add_suffix("_LRS")).join(v1.add_suffix("_1x"))
    out = {"tabla": t, "desde": L.index[0], "hasta": L.index[-1], "sesiones": len(L), "L": L}
    if multiple:
        new = list(NEW)
        iL = [list(vl.index).index(k) for k in new]
        tL = vl.loc[new, "t"].to_numpy()
        # dirección contraria (H0: Δg ≥ 0, «no es peor que la LRS»): p = Φ(t) y Romano–Wolf con −t
        # y los −t* centrados de los mismos remuestreos
        mt = pd.DataFrame({"Δg_LRS": vl.loc[new, "Δg"], "p_LRS": vl.loc[new, "p"],
                           "p_Holm_LRS": stats.holm(vl.loc[new, "p"].to_numpy()),
                           "p_RW_LRS": romano_wolf(tL, tbl[:, iL]),
                           "p_peor_LRS": norm.cdf(tL),
                           "p_RW_peor_LRS": romano_wolf(-tL, -tbl[:, iL])}, index=new)
        c1 = new + [LRS, ROT, BANDED]
        i1 = [list(v1.index).index(k) for k in c1]
        m1 = pd.DataFrame({"Δg_1x": v1.loc[c1, "Δg"], "p_1x": v1.loc[c1, "p"],
                           "p_Holm_1x": stats.holm(v1.loc[c1, "p"].to_numpy()),
                           "p_RW_1x": romano_wolf(v1.loc[c1, "t"].to_numpy(), tb1[:, i1])}, index=c1)
        out["múltiples"] = mt.join(m1, how="outer").reindex(c1)
    if spa_families:
        out["spa"] = [spa(L, bench, cands, rng, lab) for lab, bench, cands in spa_families]
    return out


# ───────────────────────────── tramos: salidas e impuestos ─────────────────────────────

def runs(w: pd.Series) -> pd.DataFrame:
    """Tramos consecutivos de igual valor de una serie de pesos sin NaN: valor, primer y
    último día y sesiones."""
    w = w.dropna()
    if w.empty:
        return pd.DataFrame(columns=["w", "first", "last", "n"])
    v = w.to_numpy(float)
    brk = np.r_[True, v[1:] != v[:-1]]
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:] - 1, len(v) - 1]
    return pd.DataFrame({"w": v[starts], "first": w.index[starts], "last": w.index[ends],
                         "n": ends - starts + 1})


def exit_decomposition(d: dict, R: dict, W: dict, name: str, start, end=None, lag: int = LAG) -> dict:
    """Reparte la diferencia de log-riqueza con el 3x comprar y mantener entre los tramos fuera
    del 3x (valor = Σ ln(1 + alternativa) − Σ ln(1 + 3x); > 0 salida acertada, ≤ 0 latigazo)
    y el coste de operar (Σ [ln(1 + r) − ln(1 + r sin costes)]). La suma es exacta:
    valor de las salidas + coste = Σ ln(1 + r) − Σ ln(1 + 3x). Todo por año en la ventana."""
    w3, w1 = W[name]
    h = w3.shift(lag).loc[start:end].dropna()
    idx = h.index
    alt = d["u"] if w1 is not None else d["rf"]
    lx = np.log1p(d["x3"].reindex(idx).clip(lower=-0.9999))
    la = np.log1p(alt.reindex(idx))
    lr = np.log1p(R[name].reindex(idx).clip(lower=-0.9999))
    lg = np.log1p(r13.run_weights(d, w3, w1=w1, lag=lag, cost_bps=0).reindex(idx).clip(lower=-0.9999))
    years = len(idx) / 252
    rr = runs(h)
    out_ = rr[rr.w == 0]
    diff = (la - lx).to_numpy()
    pos = {dt: i for i, dt in enumerate(idx)}
    vals = np.array([diff[pos[a]:pos[b] + 1].sum() for a, b in zip(out_["first"], out_["last"])])
    n = np.asarray(out_["n"], int)
    good, bad = vals > 0, vals <= 0
    cost = float((lr - lg).sum())
    total = float((lr - lx).sum())
    res = {"salidas_año": len(vals) / years,
           "latigazos_%": float(bad.mean()) if len(vals) else np.nan,
           "latigazos_cortos_%": float((bad & (n <= SHORT_OUT)).sum() / max(bad.sum(), 1)),
           "sesiones_fuera_mediana": float(np.median(n)) if len(n) else np.nan,
           "ganancia_salidas_acertadas": float(vals[good].sum()) / years,
           "coste_latigazos": float(vals[bad].sum()) / years,
           "valor_neto_salidas": float(vals.sum()) / years,
           "coste_operar": cost / years,
           "Δg_frente_3x": total / years}
    res["error_identidad"] = abs(res["valor_neto_salidas"] + res["coste_operar"] - res["Δg_frente_3x"])
    return res


def holding_periods(d: dict, W: dict, name: str, start, end=None, lag: int = LAG) -> dict:
    """Ventas del 3x cuyo último día cobrado cae en la ventana: compra al cierre anterior al
    primer día cobrado, venta al cierre del último; días naturales entre las dos. Largo
    plazo si superan LONG_TERM_DAYS. Ganancia de cada tenencia en logaritmos (Σ ln(1 + 3x))."""
    h = held(W, name, lag).dropna()
    idx = h.index
    rr = runs(h)
    ins = rr[(rr.w == 1) & (rr["last"] < idx[-1])]      # completas: hay un día fuera después
    a, b = pd.Timestamp(start), pd.Timestamp(end) if end is not None else idx[-1]
    ins = ins[(ins["last"] >= a) & (ins["last"] <= b)]
    win = h.loc[a:b]
    years = len(win) / 252
    if ins.empty:
        return {"ventas_3x_año": 0.0, "tenencia_mediana_días": np.nan, "ventas_largo_plazo_%": np.nan,
                "ganancia_corto_plazo_%": np.nan}
    pos = {dt: i for i, dt in enumerate(idx)}
    lx = np.log1p(d["x3"].reindex(idx).clip(lower=-0.9999)).to_numpy()
    buy = [idx[max(pos[f] - 1, 0)] for f in ins["first"]]
    days = np.array([(l - bb).days for l, bb in zip(ins["last"], buy)])
    gain = np.array([lx[pos[f]:pos[l] + 1].sum() for f, l in zip(ins["first"], ins["last"])])
    pos_gain = gain > 0
    short_ = days <= LONG_TERM_DAYS
    return {"ventas_3x_año": len(ins) / years,
            "tenencia_mediana_días": float(np.median(days)),
            "ventas_largo_plazo_%": float((~short_).mean()),
            "ganancia_corto_plazo_%": float(gain[pos_gain & short_].sum() / gain[pos_gain].sum())
            if pos_gain.any() else np.nan}


# ───────────────────────────── cracs y mercados bajistas ─────────────────────────────

def ath_episodes(px: pd.Series, depth: float = DEPTH) -> pd.DataFrame:
    """Caídas desde un máximo histórico de al menos `depth`: máximo, mínimo, recuperación
    del máximo (NaT si no se ha recuperado), caída y sesiones del máximo al mínimo."""
    p = px.dropna()
    v = p.to_numpy(float)
    peak_i, trough_i = 0, 0
    eps = []
    for i in range(1, len(v)):
        if v[i] >= v[peak_i]:
            if v[trough_i] / v[peak_i] - 1 <= -depth:
                eps.append((peak_i, trough_i, i))
            peak_i = trough_i = i
        elif v[i] < v[trough_i]:
            trough_i = i
    if v[trough_i] / v[peak_i] - 1 <= -depth:
        eps.append((peak_i, trough_i, None))
    rows = [{"máximo": p.index[a], "mínimo": p.index[b],
             "recuperación": p.index[c] if c is not None else pd.NaT,
             "caída": v[b] / v[a] - 1, "sesiones_hasta_mínimo": b - a} for a, b, c in eps]
    E = pd.DataFrame(rows, columns=["máximo", "mínimo", "recuperación", "caída", "sesiones_hasta_mínimo"])
    E["episodio"] = [f"{a.year}–{str(b.year)[2:]}" if a.year != b.year else str(a.year)
                     for a, b in zip(E["máximo"], E["mínimo"])]
    return E


def cycle_episodes(px: pd.Series, th: float = DEPTH) -> pd.DataFrame:
    """Datación por ciclos (sensibilidad): un mercado bajista empieza en el máximo del ciclo
    alcista tras caer `th` desde él y acaba en el mínimo tras subir `th` desde ese mínimo.
    A diferencia de ath_episodes, no exige un máximo histórico (cuenta 1938–49, 1976–78 y
    2011) y parte en varios los mercados bajistas con rebotes grandes (2000–02, 2007–09)."""
    p = px.dropna()
    v = p.to_numpy(float)
    bull, pk, tr = True, 0, 0
    eps = []
    for i in range(1, len(v)):
        if bull:
            if v[i] > v[pk]:
                pk = i
            elif v[i] <= v[pk] * (1 - th):
                bull, tr = False, i
        else:
            if v[i] < v[tr]:
                tr = i
            elif v[i] >= v[tr] * (1 + th):
                eps.append((pk, tr))
                bull, pk = True, i
    if not bull:
        eps.append((pk, tr))
    return pd.DataFrame([{"máximo": p.index[a], "mínimo": p.index[b], "caída": v[b] / v[a] - 1}
                         for a, b in eps])


def wealth_between(r: pd.Series, a, b) -> float:
    """Rentabilidad del cierre de a al cierre de b."""
    x = r.loc[a:b].iloc[1:]
    return float((1 + x.fillna(0)).prod() - 1)


def episode_table(R: dict, names, eps: pd.DataFrame, end_default) -> pd.DataFrame:
    """Rentabilidad de cada regla del máximo al mínimo y del máximo a la recuperación."""
    rows = []
    for _, e in eps.iterrows():
        rec = e["recuperación"] if pd.notna(e["recuperación"]) else end_default
        for k in names:
            rows.append({"episodio": e["episodio"], "regla": k,
                         "máximo→mínimo": wealth_between(R[k], e["máximo"], e["mínimo"]),
                         "máximo→recuperación": wealth_between(R[k], e["máximo"], rec)})
    return pd.DataFrame(rows)


def bear_summary(C: pd.DataFrame, eps: pd.DataFrame, names) -> pd.DataFrame:
    """Por regla: mediana del máximo al mínimo (todos y rápidos), episodios en que pierde menos
    que la LRS original y mediana del máximo a la recuperación."""
    piv = C.pivot(index="regla", columns="episodio", values="máximo→mínimo").reindex(list(names))
    rec = C.pivot(index="regla", columns="episodio", values="máximo→recuperación").reindex(list(names))
    all_ = list(eps["episodio"])
    fast = list(eps.loc[eps["sesiones_hasta_mínimo"] <= FAST, "episodio"])
    return pd.DataFrame({
        "mediana_máx→mín": piv[all_].median(axis=1),
        "pierde_menos_que_LRS": piv[all_].gt(piv.loc[LRS, all_], axis=1).sum(axis=1),
        "mediana_máx→mín_rápidos": piv[fast].median(axis=1),
        "pierde_menos_que_LRS_rápidos": piv[fast].gt(piv.loc[LRS, fast], axis=1).sum(axis=1),
        "mediana_máx→recuperación": rec[all_].median(axis=1),
    })


def orders(W: dict, name: str, a, b, idx: pd.DatetimeIndex, lag: int = LAG) -> str:
    """Órdenes ejecutadas entre a y b: «vende 16-10-1987; compra …» (fecha del cierre en que
    se ejecuta la orden: la sesión anterior al primer día con el peso nuevo)."""
    h = held(W, name, lag)
    x = h.loc[a:b]
    ch = x.diff()
    out = []
    for dt, v in ch[ch.fillna(0) != 0].items():
        ex = idx[idx.get_loc(dt) - 1]
        out.append(f"{'vende' if v < 0 else 'compra'} {fdate(ex)}")
    return "; ".join(out) if out else "—"


def last_sale(W: dict, name: str, day, idx: pd.DatetimeIndex, lag: int = LAG) -> pd.Timestamp | None:
    """Cierre en que se ejecutó la última venta del 3x en `day` o antes (None si no hay). Una venta
    ejecutada al cierre de `day` deja de cobrar el 3x la sesión siguiente, así que se mira hasta ella."""
    pos = idx.get_loc(pd.Timestamp(day))
    h = held(W, name, lag).loc[:idx[min(pos + 1, len(idx) - 1)]]
    drops = h.index[(h.diff() < 0).to_numpy()]
    if not len(drops):
        return None
    return idx[idx.get_loc(drops[-1]) - 1]


def crash_table(d: dict, R: dict, W: dict, names, eps: pd.DataFrame) -> pd.DataFrame:
    """1987 y 2020: órdenes, si estaba en el 3x el peor día y rentabilidad del máximo al mínimo
    y a la recuperación del índice."""
    rows = []
    idx = d["u"].index
    for lab, day in CRASH_DAYS.items():
        e = eps[(eps["máximo"] <= day) & (eps["mínimo"] >= day)].iloc[0]
        a0, b0 = CRASH_VIEW[lab]
        rec = e["recuperación"]
        for k in names:
            rows.append({"crac": lab, "regla": k,
                         "órdenes": orders(W, k, a0, b0, idx) if k not in (ONE, THREE) else "—",
                         "en_3x_el_peor_día": float(held(W, k).loc[day]) if k not in (ONE, THREE) else float(k == THREE),
                         "rent_peor_día": float(R[k].loc[day]),
                         "máximo→mínimo": wealth_between(R[k], e["máximo"], e["mínimo"]),
                         "máximo→recuperación": wealth_between(R[k], e["máximo"], rec)})
    return pd.DataFrame(rows)


# ───────────────────────────── mecanismo ─────────────────────────────

def lstar_by_state(d: dict, W: dict, name: str, start, end=None) -> dict:
    """L* aproximado de Kelly (kelly.lstar_approx) en los días en que la regla cobra el 3x
    (dentro) y en los que no (fuera), en la ventana."""
    h = held(W, name).loc[start:end]
    df = pd.concat([d["u"], d["f"], h], axis=1, keys=["u", "f", "w"]).loc[start:end].dropna()
    s = d["spread"]
    inn, out = df[df.w == 1], df[df.w == 0]
    return {"días_dentro_%": len(inn) / len(df),
            f"{kelly.LSTAR_APROX} dentro": kelly.lstar_approx(inn.u, inn.f, s),
            f"{kelly.LSTAR_APROX} fuera": kelly.lstar_approx(out.u, out.f, s) if len(out) > 50 else np.nan,
            "rent_anual_dentro": float(inn.u.mean() * 252), "rent_anual_fuera": float(out.u.mean() * 252),
            "vol_dentro": float(inn.u.std() * np.sqrt(252)), "vol_fuera": float(out.u.std() * np.sqrt(252))}


def forward_state(d: dict, s: pd.Series, start, end=None, h: int = H_FWD, lag: int = LAG) -> dict:
    """Rentabilidad en exceso de las letras (log) y varianza realizada de las h sesiones que
    cobraría la orden de t (de t + lag a t + lag + h − 1), regresadas sobre el estado de la
    señal en t. Errores de Hansen-Hodrick con h − 1 retardos (stats.hac_overlap(h))."""
    u, rf = d["u"], d["rf"]
    ex = np.log1p(u) - np.log1p(rf)
    k = lag + h - 1
    y_r = ex.rolling(h).sum().shift(-k) * 252 / h
    y_v = (u ** 2).rolling(h).sum().shift(-k) * 252 / h
    X = s.rename("dentro").to_frame()
    sl = slice(start, end)
    fr = stats.ols_hac(y_r.loc[sl], X.loc[sl], **stats.hac_overlap(h))
    fv = stats.ols_hac(y_v.loc[sl], X.loc[sl], **stats.hac_overlap(h))
    v_out = fv.beta[0]
    v_in = fv.beta[0] + fv.beta[1]
    return {"Δrent_anual_dentro−fuera": fr.beta[1], "t_rent": fr.t[1],
            "vol_dentro": np.sqrt(max(v_in, 0)), "vol_fuera": np.sqrt(max(v_out, 0)),
            "t_var": fv.t[1], "estimador": fr.estimator, "n": fr.n}


# ───────────────────────────── EMA frente a SMA ─────────────────────────────

def ema_vs_sma(L: pd.DataFrame, a: str, b: str = LRS, trim: int = 5) -> dict:
    """Diferencia de log-riqueza entre a y b por años naturales: total, años que gana a,
    y sin los `trim` peores, sin los `trim` mejores y sin ambos (recorte simétrico)."""
    x = (L[a] - L[b])
    yr = x.groupby(x.index.year).sum()
    s = yr.sort_values()
    return {"log_total": float(yr.sum()), "años": len(yr), "años_que_gana": int((yr > 0).sum()),
            f"sin_{trim}_peores": float(s.iloc[trim:].sum()), f"sin_{trim}_mejores": float(s.iloc[:-trim].sum()),
            "recorte_simétrico": float(s.iloc[trim:-trim].sum()),
            "peores_años": ", ".join(str(y) for y in s.index[:trim]),
            "mejores_años": ", ".join(str(y) for y in s.index[::-1][:trim])}


# ───────────────────────────── censo ─────────────────────────────

CENSUS_FAMILY = "R21: señales de tendencia"
CENSUS_START = "1939-04-03"   # primera sesión de la ventana común del censo de R19 (y de R25)
CENSUS_GROUP = "Otras reglas y referencias"   # uno de los tres grupos de r19.GROUP_COLOR


def tested_rules(D: dict | None = None, same_close: bool = True) -> tuple[dict, dict]:
    """Rentabilidades diarias netas en el S&P 500 de TODAS las reglas probadas aquí que no
    están ya en el censo de R19 (las 25 nuevas con la orden al cierre siguiente y, con
    same_close, sus 25 gemelas al mismo cierre, «… · mismo cierre») y sus fichas."""
    D = r13.load() if D is None else D
    d = D["S&P 500"]
    R, _, _ = build(d)
    out = {k: R[k] for k in NEW}
    meta = {k: {"retraso": LAG, "operable": True} for k in NEW}
    if same_close:
        R1, _, _ = build(d, lag=1)
        for k in NEW:
            out[f"{k} · {SAME_CLOSE}"] = R1[k]
            meta[f"{k} · {SAME_CLOSE}"] = {"retraso": 1, "operable": False}
    return out, meta


def census_p_lrs_1x(R: dict, start=CENSUS_START) -> dict:
    """p de la hipótesis principal del artículo 2 (LRS 3x original frente al 1x) como lo calcula el
    censo: r19.boot_test sobre el exceso logarítmico en la ventana del censo, con el generador de R19
    recién creado. Los remuestreos de r19.boot_test solo dependen del número de sesiones y del
    generador, así que el p coincide con el de R19 (y el de R25) sin reconstruir el censo."""
    from letf_lab.research import r19_census as r19
    both = pd.concat([R[LRS], R[ONE]], axis=1, keys=["lrs", "one"]).loc[start:].dropna()
    a = r19.active(both[["lrs"]], both["one"])
    bt = r19.boot_test(a.to_numpy(float), r19.B_BOOT, r19.BLOCK, np.random.default_rng(r19.SEED))
    return {"p": float(bt["p"][0]), "Δg": float(a["lrs"].mean() * 252), "desde": both.index[0],
            "hasta": both.index[-1]}


def census_rows(D: dict | None = None, prefix: str = "R21 · ", same_close: bool = True,
                index: pd.DatetimeIndex | None = None) -> dict[str, pd.Series]:
    """Serie diaria de rentabilidad NETA en el S&P 500 (costes y financiación incluidos) de
    cada regla probada en R21 que no está en el censo de R19, para añadirla a él:
    {prefix + nombre: pd.Series}. Por defecto en el índice diario completo del S&P 500
    (d["u"].index, 1928 → hoy), el mismo de las series de r19.Census.R; con `index` (p. ej.
    el de censo_rentabilidades_diarias.csv.gz, la ventana común 1939-04-03 → …) se
    reindexan a él. Las de mismo cierre llevan «mismo cierre» en el nombre, así que
    r19.is_same_close las marca como no operables. Fichas: census_meta."""
    rows, _ = tested_rules(D, same_close)
    out = {f"{prefix}{k}": v for k, v in rows.items()}
    if index is not None:
        out = {k: v.reindex(index) for k, v in out.items()}
    return out


def census_meta(prefix: str = "R21 · ", same_close: bool = True) -> pd.DataFrame:
    """Ficha de cada serie de census_rows para r19.Census.add(name, r, módulo, familia, grupo,
    lrs_param=False): módulo «R21», familia CENSUS_FAMILY, grupo CENSUS_GROUP."""
    names = list(NEW) + ([f"{k} · {SAME_CLOSE}" for k in NEW] if same_close else [])
    return pd.DataFrame({"módulo": "R21", "familia": CENSUS_FAMILY, "grupo": CENSUS_GROUP,
                         "parametrización_LRS3x": False,
                         "operable": [SAME_CLOSE not in k for k in names]},
                        index=[f"{prefix}{k}" for k in names])


# ───────────────────────────── cálculo ─────────────────────────────

def fdate(x) -> str:
    return pd.Timestamp(x).strftime("%-d-%-m-%Y")


def ndx_markets(nd: dict) -> list[dict]:
    """Las N_NDX reconstrucciones del Nasdaq-100 de R13 (evaluadas desde 1985-10-01)."""
    real = r12.real_letf("TQQQ")
    out = []
    for u in nd["draws"][:r13.N_NDX]:
        out.append(r12.make(u, history.rf_long(u.index), "TQQQ", real, eval_from=r12.NDX_EVAL_FROM))
    return out


def compute(D: dict | None = None) -> dict:
    """Todos los cálculos del informe (sin figuras ni texto)."""
    D = r13.load() if D is None else D
    nd = D.get("_ndx")
    d = D["S&P 500"]
    res: dict = {}
    start = r13.var_forecast(d).first_valid_index()        # 1938-02-04, la ventana del artículo 2
    res["start"] = start
    R, W, S = build(d)
    R1, _, _ = build(d, lag=1)
    res.update(R=R, W=W, S=S, R1=R1, d=d)

    # comprobación de partida: las referencias de R13, idénticas y con sus cifras
    R13, _ = r13.strategies(d, r13.var_forecast(d))
    same = {k: float((R[k] - R13[k]).abs().max()) for k in (ONE, THREE, LRS, ROT, BANDED)}
    res["sanity_diff"] = same
    res["sanity"] = r13.stats_table({k: R13[k] for k in (ONE, THREE, LRS, ROT, BANDED)}, d["rf"], start)
    res["census_p"] = census_p_lrs_1x(R)

    # S&P 500: ventana principal, subperiodos y tras 2016
    res["main"] = evaluate(R, W, d, start, None, "sp500 principal", spa_families=spa_sets())
    res["sub"] = {lab: evaluate(R, W, d, a or start, b, f"sp500 {lab}", multiple=False)
                  for lab, a, b in SUBPERIODS}
    _, a, b = POST
    res["post"] = evaluate(R, W, d, a, b, "sp500 2016", spa_families=spa_sets())
    res["rules_start"] = log_frame(R, NAMES, "1928-01-01").index[0]
    res["early"] = evaluate(R, W, d, res["rules_start"], start - pd.Timedelta(days=1), "sp500 1929-1938",
                            multiple=False)

    # salidas, tenencia y mecanismo (S&P 500, ventana principal)
    res["exits"] = pd.DataFrame({k: exit_decomposition(d, R, W, k, start) for k in RULES}).T
    res["hold"] = pd.DataFrame({s: holding_periods(d, W, rule_name(s, LEG_TB), start)
                                for s in ALL_SIGNALS}).T
    res["lstar"] = pd.DataFrame({s: lstar_by_state(d, W, rule_name(s, LEG_TB), start)
                                 for s in ALL_SIGNALS}).T
    uf = pd.concat([d["u"], d["f"]], axis=1, keys=["u", "f"]).loc[start:].dropna()
    res["lstar_all"] = kelly.lstar_approx(uf.u, uf.f, d["spread"])
    res["fwd"] = pd.DataFrame({s: forward_state(d, S[s], start) for s in ALL_SIGNALS}).T

    # EMA frente a SMA: por años y según el inicio
    Lm = res["main"]["L"]
    res["ema_years"] = pd.DataFrame({k: ema_vs_sma(Lm, rule_name(s, LEG_TB)) for k, s in
                                     (("EMA 200 − SMA 200", "EMA 200"),
                                      ("EMA 200 ±2% − SMA 200", "EMA 200 ±2%"))}).T
    starts = {"1938-02-04": start, "1939-04-03 (censo)": "1939-04-03", "1950": "1950-01-03",
              "1974": "1974-01-02", "2000": "2000-01-03", "2010-02-12": r13.REAL_FROM, "2016": "2016-01-04"}
    Lall = log_frame(R, NAMES, "1929-01-01")
    res["ema_starts"] = pd.DataFrame({lab: {s: float((Lall[rule_name(s, LEG_TB)] - Lall[LRS]).loc[a:].mean() * 252)
                                            for s in ("EMA 200", "EMA 200 ±2%", "Cruce SMA 50/200",
                                                      "MACD semanal>0", "SMA 200 o MACD>0")}
                                      for lab, a in starts.items()})

    # cracs y mercados bajistas
    eps_all = ath_episodes(d["px"])
    eps = eps_all[eps_all["máximo"] >= start].reset_index(drop=True)
    res["eps_all"], res["eps"] = eps_all, eps
    res["bear"] = episode_table(R, NAMES, eps_all, d["u"].index[-1])
    res["bear_sum"] = bear_summary(res["bear"][res["bear"].episodio.isin(eps.episodio)], eps, NAMES)
    cyc = cycle_episodes(d["px"])
    cyc = cyc[cyc["máximo"] >= start].reset_index(drop=True)
    cyc["episodio"] = [f"{a.date()}" for a in cyc["máximo"]]
    cyc["sesiones_hasta_mínimo"] = [len(d["px"].loc[a:b]) - 1 for a, b in zip(cyc["máximo"], cyc["mínimo"])]
    cyc["recuperación"] = pd.NaT
    res["cyc"] = cyc
    Cc = episode_table(R, NAMES, cyc, d["u"].index[-1])
    res["cyc_sum"] = bear_summary(Cc, cyc, NAMES).drop(columns="mediana_máx→recuperación")
    res["crash"] = crash_table(d, R, W, NAMES, eps_all)

    # mismo cierre (sensibilidad)
    rows = {}
    for k in RULES:
        row = {}
        for lab_, a_, b_ in (("1938–1989", start, "1989-12-31"), ("1990–2026", "1990-01-01", None)):
            c2 = metrics.cagr(R[k].loc[a_:b_].dropna())
            c1 = metrics.cagr(R1[k].loc[a_:b_].dropna())
            row[f"CAGR {lab_} cierre siguiente"] = c2
            row[f"CAGR {lab_} mismo cierre"] = c1
        rows[k] = row
    res["moc"] = pd.DataFrame(rows).T
    sig_ret = r13.signal_index(d).pct_change()
    px_ret = d["px"].pct_change()
    res["autocorr"] = {lab_: {"índice de la señal (rentabilidad total)": float(sig_ret.loc[a_:b_].autocorr(1)),
                              "índice de precio (^GSPC)": float(px_ret.loc[a_:b_].autocorr(1))}
                       for lab_, a_, b_ in (("1938–1989", start, "1989-12-31"), ("1990–2026", "1990-01-01", None))}

    # Nasdaq-100: 10 reconstrucciones
    if nd is not None:
        ndx = []
        for k_, dk in enumerate(ndx_markets(nd)):
            Rk, Wk, _ = build(dk)
            ev = evaluate(Rk, Wk, dk, r12.NDX_EVAL_FROM, None, f"ndx {k_}",
                          spa_families=spa_sets()[:3])
            subs = {lab_: evaluate(Rk, Wk, dk, a_, b_, f"ndx {k_} {lab_}", multiple=False)["tabla"]
                    for lab_, a_, b_ in NDX_WINDOWS}
            ndx.append({"ev": ev, "sub": subs, "d": dk, "R": Rk, "W": Wk})
        res["ndx"] = ndx
        d0, R0, W0 = ndx[0]["d"], ndx[0]["R"], ndx[0]["W"]
        res["ndx_exits"] = pd.DataFrame({k: exit_decomposition(d0, R0, W0, k, r12.NDX_EVAL_FROM)
                                         for k in TB_RULES}).T
        res["ndx_hold"] = pd.DataFrame({s: holding_periods(d0, W0, rule_name(s, LEG_TB), r12.NDX_EVAL_FROM)
                                        for s in ALL_SIGNALS}).T
        e_n = ath_episodes(r13.price_index(d0, "Nasdaq-100").loc[r12.NDX_EVAL_FROM:])
        res["ndx_eps"] = e_n
        res["ndx_bear"] = episode_table(R0, NAMES, e_n, d0["u"].index[-1])
        res["ndx_bear_sum"] = bear_summary(res["ndx_bear"], e_n, NAMES)
        L0 = ndx[0]["ev"]["L"]
        res["ndx_ema_years"] = ema_vs_sma(L0, rule_name("EMA 200", LEG_TB))

    # ETFs reales (UPRO y TQQQ) desde REAL_FROM
    res["real"] = {"S&P 500 (UPRO)": evaluate(R, W, d, r13.REAL_FROM, None, "real sp500",
                                              spa_families=spa_sets(extra_three=True))}
    if nd is not None:
        res["real"]["Nasdaq-100 (TQQQ)"] = evaluate(res["ndx"][0]["R"], res["ndx"][0]["W"], res["ndx"][0]["d"],
                                                    r13.REAL_FROM, None, "real ndx",
                                                    spa_families=spa_sets(extra_three=True))
    # semiconductores (R13): desde su primer día con previsión GJR, como la tabla de R13
    ds = D.get("Semis")
    if ds is not None:
        s_start = r13.var_forecast(ds).first_valid_index()
        Rs, Ws, _ = build(ds)
        R13s, _ = r13.strategies(ds, r13.var_forecast(ds))
        res["semis_sanity"] = {k: float((Rs[k] - R13s[k]).abs().max()) for k in (ONE, THREE, LRS, ROT, BANDED)}
        res["semis"] = evaluate(Rs, Ws, ds, s_start, None, "semis", spa_families=spa_sets())
        res["semis"]["R"], res["semis"]["W"], res["semis"]["d"] = Rs, Ws, ds
        res["semis_bear_eps"] = ath_episodes(r13.price_index(ds, "Semis").loc[s_start:])
        res["semis_bear"] = bear_summary(episode_table(Rs, NAMES, res["semis_bear_eps"], ds["u"].index[-1]),
                                         res["semis_bear_eps"], NAMES)
        res["real"]["Semis (SOXL)"] = evaluate(Rs, Ws, ds, r13.REAL_FROM, None, "real semis",
                                               spa_families=spa_sets(extra_three=True))
    return res


# ───────────────────────────── figuras ─────────────────────────────

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
GREY = "#9a9994"


def _style(ax, grid_axis: str = "x"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _repel(ys: list[float], gap: float) -> list[float]:
    """Separa posiciones verticales (en unidades del eje) para que dos rótulos no se pisen."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return list(out)


def _pp_fmt(ax, axis: str = "x", d: int = 0):
    f = FuncFormatter(lambda v, _: num_es(v, d))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


FIG_RULES = ("EMA 200 · 3x/letras", "MACD>0 · 3x/letras", "Cruce SMA 50/200 · 3x/letras",
             "SMA 200 o MACD>0 · 3x/letras")
FIG_COLORS = dict(zip(FIG_RULES, (BLUE, ORANGE, AQUA, VIOLET)))


def fig_relative(res: dict, rep: Report):
    """Riqueza de cuatro señales relativa a la LRS original (S&P 500, ventana principal)."""
    L = res["main"]["L"]
    fig, ax = plt.subplots(figsize=(FIG_W, 5.4))
    ends = []
    for k in FIG_RULES:
        rel = np.exp((L[k] - L[LRS]).cumsum())
        ax.semilogy(rel.index, rel.values, color=FIG_COLORS[k], lw=1.6, label=short(k))
        ends.append((k, float(rel.iloc[-1])))
    ax.axhline(1, color=INK, lw=1, ls="--")
    ax.text(L.index[0], 1.08, "igual que la LRS original", color=MUTED, fontsize=11, va="bottom")
    ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
    lo = min(v for _, v in ends) * 0.7
    ax.set_ylim(max(lo, 0.003), 2.5)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num_es(v, 3 if v < 0.01 else 2 if v < 0.1 else 1)
                                               .rstrip("0").rstrip(",")))
    logs_ = _repel([np.log10(v) for _, v in ends], 0.13)
    x_end = L.index[-1] + pd.Timedelta(days=500)
    for (k, v), ly in zip(ends, logs_):
        # cada cociente final, del color de su línea: dos líneas que se cruzan no se confunden
        ax.text(x_end, 10 ** ly, f"{num_es(v, 3 if v < 0.1 else 2)}×", color=FIG_COLORS[k], fontsize=12,
                fontweight="bold", va="center")
    ax.set_xlim(L.index[0], L.index[-1] + pd.Timedelta(days=2600))
    _style(ax, "y")
    ax.set(ylabel="Riqueza relativa (escala log)",
           title="S&P 500 desde 1938: riqueza de cada señal (pata de letras)\nentre la de la LRS original")
    h, lab = ax.get_legend_handles_labels()
    legend_band(fig, h, lab, ncol=2)
    rep.fig(fig, "riqueza_relativa_sp500",
            "Riqueza de cuatro señales de tendencia (3x/letras) dividida entre la de la LRS 3x original, S&P 500 "
            "desde 1938; por debajo de 1, la señal acumula menos riqueza. A la derecha, el cociente final")


def fig_tradeoff(res: dict, rep: Report):
    """Por señal (pata de letras): salidas al año, mediana de la caída en los mercados bajistas y Δg
    frente a la LRS original con su IC del 90%."""
    t = res["main"]["tabla"]
    ex = res["exits"]
    bs = res["bear_sum"]
    rows = sorted(TB_RULES, key=lambda k: ex.loc[k, "salidas_año"])
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 7.6), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.1, 1.3]})
    a, b, c = axes
    a.barh(y, [ex.loc[k, "salidas_año"] for k in rows], color=GREY, height=0.62)
    for yi, k in zip(y, rows):
        a.text(ex.loc[k, "salidas_año"] + 0.2, yi, num_es(ex.loc[k, "salidas_año"], 1), va="center",
               fontsize=10, color=INK)
    a.set_xlim(0, max(ex.loc[list(rows), "salidas_año"]) * 1.35)
    a.set(title="Salidas\nal año")
    b.barh(y, [100 * bs.loc[k, "mediana_máx→mín"] for k in rows],
           color=[ORANGE if k != LRS else INK for k in rows], height=0.62)
    b.set(title=f"Caída mediana en los\n{len(res['eps'])} mercados bajistas (%)")
    b.set_xlim(min(100 * bs.loc[list(rows), "mediana_máx→mín"]) * 1.15, 0)
    dg = [100 * t.loc[k, "Δg_LRS"] if k != LRS else 0.0 for k in rows]
    lo = [100 * t.loc[k, "IC90_bajo_LRS"] if k != LRS else 0.0 for k in rows]
    hi = [100 * t.loc[k, "IC90_alto_LRS"] if k != LRS else 0.0 for k in rows]
    c.axvline(0, color=INK, lw=1)
    c.hlines(y, lo, hi, color=BLUE, lw=2)
    c.plot(dg, y, "o", color=BLUE, ms=7, mec="white", mew=1.2)
    c.set(title="Δg frente a la LRS\n(pp/año, IC 90%)")
    for ax in axes:
        _style(ax, "x")
        ax.tick_params(axis="y", length=0)
    a.set_yticks(y, [short(k) for k in rows], fontsize=11)
    for lab in a.get_yticklabels():
        if lab.get_text() == short(LRS):
            lab.set_fontweight("bold")
    a.invert_yaxis()
    comma_fig(fig)
    fig.suptitle("S&P 500 desde 1938: rapidez, protección y crecimiento")
    rep.fig(fig, "rapidez_proteccion_crecimiento",
            "Cada señal de tendencia en la pata de letras, ordenadas de la más lenta a la más rápida: salidas del 3x "
            "al año, mediana de la caída del máximo al mínimo en los mercados bajistas y diferencia de crecimiento "
            "logarítmico con la LRS original, con su intervalo bootstrap del 90%")


def fig_whipsaws(res: dict, rep: Report):
    """Valor de las salidas acertadas, coste de los latigazos y neto (pata de letras)."""
    ex = res["exits"]
    rows = sorted(TB_RULES, key=lambda k: ex.loc[k, "salidas_año"])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(FIG_W, 6.8))
    g = 100 * ex.loc[rows, "ganancia_salidas_acertadas"].to_numpy(float)
    c = 100 * ex.loc[rows, "coste_latigazos"].to_numpy(float)
    n = 100 * (ex.loc[rows, "valor_neto_salidas"] + ex.loc[rows, "coste_operar"]).to_numpy(float)
    ax.barh(y, g, color=AQUA, height=0.62, label="salidas acertadas")
    ax.barh(y, c, color=ORANGE, height=0.62, label="latigazos")
    ax.plot(n, y, "D", color=INK, ms=7, mec="white", mew=1.0, label="neto, con el coste de operar")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [short(k) for k in rows], fontsize=11)
    ax.tick_params(axis="y", length=0)
    for lab in ax.get_yticklabels():
        if lab.get_text() == short(LRS):
            lab.set_fontweight("bold")
    ax.invert_yaxis()
    _style(ax, "x")
    ax.set(xlabel="Crecimiento logarítmico frente al 3x comprar y mantener (pp/año)",
           title="S&P 500 desde 1938: lo que ganan las salidas acertadas\ny lo que cuestan los latigazos (pata de letras)")
    comma_fig(fig)
    h, lab = ax.get_legend_handles_labels()
    order = [lab.index("salidas acertadas"), lab.index("latigazos"), lab.index("neto, con el coste de operar")]
    legend_band(fig, [h[i] for i in order], [lab[i] for i in order], ncol=3)
    rep.fig(fig, "latigazos",
            "Por señal, ordenadas de la más lenta a la más rápida: suma anual de lo que ganaron los tramos fuera del 3x "
            "que esquivaron caídas (verde) y de lo que costaron los que no (naranja), en crecimiento logarítmico "
            "frente a mantener el 3x; el rombo es el neto después del coste de operar")


def fig_crashes(res: dict, rep: Report):
    """1987 y 2020 día a día: 100 invertidos al inicio de cada ventana."""
    R = res["R"]
    show = ((THREE, GREY, "3x comprar y mantener"), (LRS, INK, short(LRS))) + \
        tuple((k, FIG_COLORS[k], short(k)) for k in FIG_RULES[:3])
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 8.6))
    for ax, (lab, (a, b)) in zip(axes, CRASH_VIEW.items()):
        for k, col, name in show:
            r = R[k].loc[a:b].iloc[1:]
            w = 100 * (1 + r).cumprod()
            w = pd.concat([pd.Series([100.0], index=[pd.Timestamp(a)]), w])
            # la LRS original encima de todas (antes del crac coincide con las que están dentro)
            ax.plot(w.index, w.values, color=col, lw=2.4 if k == LRS else 1.4, label=name,
                    zorder=5 if k == LRS else 2, ls="-" if k in (LRS, THREE) else (0, (5, 1.5)))
        day = pd.Timestamp(CRASH_DAYS[lab])
        ax.axvline(day, color=MUTED, lw=0.8, ls="--")
        ax.text(day, ax.get_ylim()[1], f" {fdate(day)}", color=MUTED, fontsize=11, va="top")
        _style(ax, "y")
        ax.set(ylabel="Valor de 100 invertidos", title=f"{lab}: del {fdate(a)} al {fdate(b)}")
    comma_fig(fig)
    import matplotlib.dates as mdates
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-m-%Y"))
    h, lab = axes[0].get_legend_handles_labels()
    fig.suptitle("1987 y 2020: cada regla ante el peor día del crac")
    legend_band(fig, h, lab, ncol=3)
    rep.fig(fig, "cracs_1987_2020",
            "100 invertidos al inicio de cada ventana en el 3x comprar y mantener, la LRS original y tres señales "
            "(pata de letras); la línea vertical discontinua marca el peor día de cada crac. Mientras varias reglas "
            "están en el 3x sus líneas coinciden, y la de la LRS original queda encima")


def fig_markets(res: dict, rep: Report):
    """Δg frente a la LRS original (pata de letras) con IC del 90% en cuatro mercados y ventanas."""
    panels = [("S&P 500, 1938–2026", res["main"]["tabla"])]
    if "ndx" in res:
        panels.append(("Nasdaq-100, 1985–2026", res["ndx"][0]["ev"]["tabla"]))
    if "semis" in res:
        panels.append((f"Semis, {res['semis']['desde'].year}–2026", res["semis"]["tabla"]))
    for k, ev in res["real"].items():
        panels.append((f"{k.replace(' (', ' con ').rstrip(')')}, 2010–2026", ev["tabla"]))
    rows = [k for k in TB_RULES if k != LRS]
    y = np.arange(len(rows))
    ncol = 2
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIG_W, 4.6 * nrow + 0.4), sharey=True)
    axes = np.atleast_2d(axes)
    for ax, (title, t) in zip(axes.flat, panels):
        dg = 100 * t.loc[rows, "Δg_LRS"].to_numpy(float)
        lo = 100 * t.loc[rows, "IC90_bajo_LRS"].to_numpy(float)
        hi = 100 * t.loc[rows, "IC90_alto_LRS"].to_numpy(float)
        ax.axvline(0, color=INK, lw=1)
        ax.hlines(y, lo, hi, color=BLUE, lw=2)
        ax.plot(dg, y, "o", color=BLUE, ms=6, mec="white", mew=1.0)
        _style(ax, "x")
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=12)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_yticks(y, [short(k) for k in rows], fontsize=12)
    axes[0, 0].invert_yaxis()
    for ax in axes[-1, :]:
        ax.set_xlabel("Δg frente a la LRS (pp/año)")
    comma_fig(fig)
    fig.suptitle("Diferencia de crecimiento con la LRS original, pata de letras (IC del 90%)")
    rep.fig(fig, "mercados",
            "Diferencia de crecimiento logarítmico anual de cada señal (pata de letras) con la LRS 3x original, con su "
            "intervalo bootstrap del 90%, en el S&P 500 desde 1938, el Nasdaq-100 desde 1985, los semiconductores "
            "desde 2004 y con los ETFs reales desde 2010 (Nasdaq-100: reconstrucción 0 del calentamiento anterior a "
            "1985). Los intervalos son nominales: no descuentan la búsqueda")


# ───────────────────────────── informe ─────────────────────────────

DEFINITIONS = {
    "SMA 200": "índice > su media simple de 200 sesiones",
    "SMA 200 ±2%": "entra al superar la media simple un 2%, sale al caer un 2% bajo ella",
    "EMA 200": "índice > su media exponencial de 200 sesiones (α = 2/201)",
    "EMA 200 ±2%": "la EMA 200 con la banda ±2%",
    "MACD>0": "EMA 12 − EMA 26 > 0 (diario)",
    "MACD>señal": "EMA 12 − EMA 26 > su EMA de 9 (diario)",
    "MACD semanal>0": "MACD(12, 26, 9) > 0 con cierres semanales",
    "MACD semanal>señal": "MACD > señal con cierres semanales",
    "Cruce SMA 50/200": "media simple de 50 > media simple de 200",
    "Cruce EMA 50/200": "EMA 50 > EMA 200",
    "Donchian 252/126": "entra al cerrar sobre el máximo de 252 cierres; sale bajo el mínimo de 126",
    "Donchian 55/20": "entra al cerrar sobre el máximo de 55 cierres; sale bajo el mínimo de 20",
    "SMA 200 y MACD>0": "dentro solo si SMA 200 y MACD>0 lo dicen las dos",
    "SMA 200 o MACD>0": "dentro si lo dice cualquiera de las dos",
}


def y_join(items) -> str:
    """«a», «a y b», «a, b y c»."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1] if items else ""


def pp(x: float, d: int = 1, sign: bool = True) -> str:
    """Fracción anual en puntos porcentuales: 0,0123 → «+1,2 pp»."""
    v = 100 * float(x)
    s = num_es(v, d)
    return ("+" + s if sign and round(v, d) > 0 else s) + " pp"


def fp(p: float) -> str:
    """p con tres decimales, redondeado como en las tablas (np.round: 0,7555 → 0,756)."""
    return "< 0,001" if p < 0.001 else num_es(float(np.round(p, 3)), 3)


def as_pp(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype(float) * 100
    return out.rename(columns={c: f"{c} (pp)" for c in cols})


def truncate_orders(s: str, n: int = 4) -> str:
    parts = s.split("; ")
    return s if len(parts) <= n else "; ".join(parts[:n]) + f"; … (+{len(parts) - n})"


def ndx_aggregate(res: dict) -> pd.DataFrame:
    T = pd.concat([x["ev"]["tabla"].assign(rec=i) for i, x in enumerate(res["ndx"])])
    g = T.groupby(level=0)
    out = pd.DataFrame({"CAGR mediana": g.CAGR.median(), "CAGR mín": g.CAGR.min(), "CAGR máx": g.CAGR.max(),
                        "maxDD mediana": g.maxDD.median(), "cambios_año": g.cambios_año.median(),
                        "Δg_LRS mediana": g.Δg_LRS.median(), "p_LRS mediana": g.p_LRS.median(),
                        "p_LRS mín": g.p_LRS.min(), "Δg_1x mediana": g.Δg_1x.median(),
                        "p_1x mediana": g.p_1x.median()})
    return out.reindex(list(NAMES))


def ndx_spa(res: dict) -> pd.DataFrame:
    S = pd.concat([pd.DataFrame(x["ev"]["spa"]).assign(rec=i) for i, x in enumerate(res["ndx"])])
    g = S.groupby("familia", sort=False)
    return pd.DataFrame({"reglas": g.reglas.first(), "mejor (más frecuente)": g.mejor.agg(lambda s: s.mode().iloc[0]),
                         "Δg_mejor mediana": g.Δg_mejor.median(), "p_SPA mín": g.p_SPA_c.min(),
                         "p_SPA mediana": g.p_SPA_c.median(), "p_SPA máx": g.p_SPA_c.max()})


def ndx_sub(res: dict) -> pd.DataFrame:
    cols = {}
    for lab, _, _ in NDX_WINDOWS:
        T = pd.concat([x["sub"][lab].assign(rec=i) for i, x in enumerate(res["ndx"])])
        g = T.groupby(level=0)
        cols[f"CAGR {lab}"] = g.CAGR.median()
        cols[f"Δg_LRS {lab}"] = g.Δg_LRS.median()
        cols[f"p_LRS {lab}"] = g.p_LRS.median()
    return pd.DataFrame(cols).reindex(list(NAMES))


def write_report(res: dict, rep: Report):
    d = res["d"]
    start = res["start"]
    m = res["main"]
    t = m["tabla"]
    ex, hold, bs = res["exits"], res["hold"], res["bear_sum"]
    spa_m = pd.DataFrame(m["spa"])
    mt = m["múltiples"]
    post = res["post"]
    tp = post["tabla"]
    spa_p = pd.DataFrame(post["spa"])
    new_tab = t.loc[list(NEW)]
    n_neg = int((new_tab["Δg_LRS"] < 0).sum())
    pos = new_tab.index[new_tab["Δg_LRS"] > 0].tolist()
    c1 = list(NEW) + [LRS, ROT, BANDED]
    n_p1 = int((mt.loc[c1, "p_1x"] < 0.05).sum())
    rw1 = mt.loc[c1, "p_RW_1x"]
    ema, sma = "EMA 200 · 3x/letras", LRS
    ey = res["ema_years"].loc["EMA 200 − SMA 200"]
    es = res["ema_starts"].loc["EMA 200"]
    # ¿SMA o EMA? es la pregunta del propietario, fijada antes de mirar: se contrasta en las dos
    # direcciones (bilateral). La dirección «peor» sola, con la búsqueda de las 25, en la tabla 5
    p_ema = float(t.loc[ema, "p_LRS"])
    p_ema2 = 2 * min(p_ema, 1 - p_ema)
    p_ema_rw = float(mt.loc[ema, "p_RW_peor_LRS"])
    worse = t.loc[ema, "Δg_LRS"] < 0
    ema_verdict = (
        f"Como pregunta fijada de antemano (la del propietario: ¿SMA o EMA?), el contraste bilateral da p = "
        f"{fp(p_ema2)}" + (
            ("; si solo se mirara la dirección «peor», p = " + fp(1 - p_ema) + " (" +
             fp(p_ema_rw) + " con Romano–Wolf sobre las 25 nuevas). "
             + ("Es una desventaja consistente y significativa al 5% como pregunta bilateral."
                if p_ema2 < 0.05 else
                "Es una desventaja consistente en el tiempo, pero no significativa al 5% como pregunta bilateral."))
            if worse else ": la diferencia no se distingue del azar."))
    sub = res["sub"]
    sub_labs = [lab for lab, _, _ in SUBPERIODS]
    ema_sub = [sub[lab]["tabla"].loc[ema, "Δg_LRS"] for lab in sub_labs]
    mac0, macs = "MACD>0 · 3x/letras", "MACD>señal · 3x/letras"
    w0 = "MACD semanal>0 · 3x/letras"
    andk, ork = "SMA 200 y MACD>0 · 3x/letras", "SMA 200 o MACD>0 · 3x/letras"
    ork1 = "SMA 200 o MACD>0 · 3x/1x"
    cr_sma, don = "Cruce SMA 50/200 · 3x/letras", "Donchian 252/126 · 3x/letras"
    don55 = "Donchian 55/20 · 3x/letras"
    crash = res["crash"].set_index(["crac", "regla"])
    eps = res["eps"]
    n_eps = len(eps)
    fwd, lst = res["fwd"], res["lstar"]
    moc = res["moc"]
    ac = res["autocorr"]
    one_cagr = t.loc[ONE, "CAGR"]

    # ── resumen ──
    rep.h("Resumen")
    bullets = []
    san = res["sanity"]
    bullets.append(
        f"**Comprobación de partida.** Las referencias construidas aquí coinciden con las de R13 (diferencia "
        f"máxima {num_es(max(res['sanity_diff'].values()), 0)}): desde el {fdate(start)}, la LRS 3x original rinde "
        f"{pct_es(san.loc[LRS, 'CAGR'])} al año con una caída máxima del {pct_es(san.loc[LRS, 'maxDD'])}; el 1x, "
        f"{pct_es(san.loc[ONE, 'CAGR'])}, y el 3x comprar y mantener, {pct_es(san.loc[THREE, 'CAGR'])}, como en el "
        "informe de R13 y en el artículo 2.")
    only = (f"La única que crece más es «{pos[0]}» ({pct_es(t.loc[pos[0], 'CAGR'])} frente a "
            f"{pct_es(t.loc[LRS, 'CAGR'])}, Δg {pp(t.loc[pos[0], 'Δg_LRS'])}/año, p = {fp(t.loc[pos[0], 'p_LRS'])}), "
            "que no es una señal nueva: es la banda ±2% del artículo 2 con la pata 1x de su rotación. "
            if len(pos) == 1 else
            f"Crecen más {len(pos)}: {', '.join(f'«{k}»' for k in pos)}. " if pos else "")
    bullets.append(
        f"**En el S&P 500 desde 1938 ninguna señal nueva supera a la media de 200.** De las 25 variantes nuevas, "
        f"{n_neg} crecen menos que la LRS 3x original. {only}Descontada la búsqueda, el SPA de las 25 frente a la "
        f"LRS original da p = {fp(spa_m.p_SPA_c.iloc[0])}; " + (
            "ni Holm ni Romano–Wolf rechazan ninguna."
            if not ((mt.loc[list(NEW), 'p_Holm_LRS'] < 0.05).any() or (mt.loc[list(NEW), 'p_RW_LRS'] < 0.05).any())
            else f"Holm rechaza {int((mt.loc[list(NEW), 'p_Holm_LRS'] < 0.05).sum())} y Romano–Wolf "
                 f"{int((mt.loc[list(NEW), 'p_RW_LRS'] < 0.05).sum())}."))
    bullets.append(
        f"**Frente al 1x.** {n_p1} de las 28 reglas (las 25 nuevas y las tres de R13) tienen un p nominal inferior al 5%. "
        f"Con la búsqueda descontada, el SPA de las 28 da p = {fp(spa_m.p_SPA_c.iloc[2])} y Romano–Wolf "
        + ("no deja ninguna" if not (rw1 < 0.05).any() else f"deja {int((rw1 < 0.05).sum())}")
        + f" por debajo del 5% (la de menor p ajustado, «{rw1.idxmin()}», {fp(rw1.min())}). "
        "La LRS original, "
        f"contrastada sola como hipótesis fijada de antemano, da Δg {pp(t.loc[LRS, 'Δg_1x'])}/año y p = "
        f"{fp(t.loc[LRS, 'p_1x'])} en esta ventana. La cifra que cita el artículo es la del censo: p = "
        f"{fp(res['census_p']['p'])} (R19 y R25, del {fdate(res['census_p']['desde'])} al "
        f"{fdate(res['census_p']['hasta'])}, recalculado aquí con sus mismos remuestreos); la de aquí difiere por la "
        "ventana y por los remuestreos, no por el método.")
    bullets.append(
        f"**EMA 200 frente a SMA 200.** La media exponencial crece menos: {pct_es(t.loc[ema, 'CAGR'])} frente a "
        f"{pct_es(t.loc[sma, 'CAGR'])}, Δg {pp(t.loc[ema, 'Δg_LRS'])}/año (IC 90%: {pp(t.loc[ema, 'IC90_bajo_LRS'])} a "
        f"{pp(t.loc[ema, 'IC90_alto_LRS'])}), con una caída máxima del {pct_es(t.loc[ema, 'maxDD'])} frente al "
        f"{pct_es(t.loc[sma, 'maxDD'])} y {num_es(t.loc[ema, 'cambios_año'], 1)} cambios al año frente a "
        f"{num_es(t.loc[sma, 'cambios_año'], 1)}. La desventaja es persistente: negativa en los tres subperiodos "
        f"({', '.join(pp(x) for x in ema_sub)}) y con cualquiera de los {len(es)} inicios probados ({pp(es.max())} a "
        f"{pp(es.min())}); la EMA gana {ey['años_que_gana']} de {ey['años']} años. Quitar sus 5 peores años borra la "
        f"diferencia acumulada ({num_es(ey['sin_5_peores'], 2)} en logaritmo, frente a {num_es(ey['log_total'], 2)}), "
        f"pero quitar los 5 mejores la agranda ({num_es(ey['sin_5_mejores'], 2)}) y el recorte simétrico la deja en "
        f"{num_es(ey['recorte_simétrico'], 2)}. " + ema_verdict)
    if "ndx" in res:
        na = ndx_aggregate(res)
        ns = ndx_spa(res)
        bullets.append(
            f"**En el Nasdaq-100 (desde 1985-10, mediana de 10 reconstrucciones) el signo se invierte:** la EMA 200 rinde "
            f"{pct_es(na.loc[ema, 'CAGR mediana'])} frente al {pct_es(na.loc[LRS, 'CAGR mediana'])} de la LRS original "
            f"(Δg {pp(na.loc[ema, 'Δg_LRS mediana'])}/año, p nominal mediano {fp(na.loc[ema, 'p_LRS mediana'])}). "
            f"El SPA de las 25 frente a la LRS da p entre {fp(ns.iloc[0]['p_SPA mín'])} y {fp(ns.iloc[0]['p_SPA máx'])} "
            "según la reconstrucción: nada es significativo tras la búsqueda, y el mismo par de reglas da signos "
            "opuestos en los dos índices.")
    if "semis" in res:
        st = res["semis"]["tabla"]
        ss = pd.DataFrame(res["semis"]["spa"])
        up_s = [k for k in NEW if st.loc[k, "Δg_LRS"] > 0]
        bullets.append(
            f"**Semiconductores (desde el {fdate(res['semis']['desde'])}, el inicio de R13).** La LRS original rinde "
            f"{pct_es(st.loc[LRS, 'CAGR'])} frente al {pct_es(st.loc[ONE, 'CAGR'])} del 1x y al "
            f"{pct_es(st.loc[THREE, 'CAGR'])} del 3x; la EMA 200, {pct_es(st.loc[ema, 'CAGR'])} (Δg "
            f"{pp(st.loc[ema, 'Δg_LRS'])}/año). De las 25 nuevas, {len(up_s)} crecen más que la LRS; el SPA de las 25 "
            f"frente a ella da p = {fp(ss.p_SPA_c.iloc[0])} y el de las 28 frente al 1x, p = {fp(ss.p_SPA_c.iloc[2])}. "
            f"Son {num_es(res['semis']['sesiones'] / 252, 1)} años de un índice muy volátil: poca potencia.")
    bullets.append(
        f"**El MACD diario es la peor familia.** MACD>0 (3x/letras) rinde {pct_es(t.loc[mac0, 'CAGR'])}, menos que el "
        f"1x ({pct_es(one_cagr)}), y MACD>señal {pct_es(t.loc[macs, 'CAGR'])} con {num_es(t.loc[macs, 'cambios_año'], 1)} "
        f"cambios al año. El MACD semanal>0, mucho más lento ({num_es(t.loc[w0, 'cambios_año'], 1)} cambios al año), "
        f"llega al {pct_es(t.loc[w0, 'CAGR'])} (Δg {pp(t.loc[w0, 'Δg_LRS'])}/año frente a la LRS).")
    bullets.append(
        f"**Salidas acertadas y latigazos.** La SMA 200 sale del 3x {num_es(ex.loc[LRS, 'salidas_año'], 1)} veces al "
        f"año y el {pct_es(ex.loc[LRS, 'latigazos_%'], 0)} de esas salidas son latigazos. Aun así, lo que ganan las "
        f"acertadas ({pp(ex.loc[LRS, 'ganancia_salidas_acertadas'])}/año de crecimiento logarítmico frente al 3x) "
        f"supera lo que cuestan los latigazos ({pp(ex.loc[LRS, 'coste_latigazos'])}): neto "
        f"{pp(ex.loc[LRS, 'valor_neto_salidas'])}, y {pp(ex.loc[LRS, 'Δg_frente_3x'])} tras el coste de operar. MACD>0: "
        f"{pp(ex.loc[mac0, 'ganancia_salidas_acertadas'])} y {pp(ex.loc[mac0, 'coste_latigazos'])}, neto "
        f"{pp(ex.loc[mac0, 'valor_neto_salidas'])}. MACD>señal: {num_es(ex.loc[macs, 'salidas_año'], 1)} salidas al año, "
        f"neto {pp(ex.loc[macs, 'valor_neto_salidas'])}. Con el 3x, cada latigazo cuesta unas tres veces el movimiento "
        "del índice.")
    c87, c20 = crash.xs("1987"), crash.xs("2020")
    idx = d["u"].index
    W = res["W"]
    d87, d20 = pd.Timestamp(CRASH_DAYS["1987"]), pd.Timestamp(CRASH_DAYS["2020"])
    sale = lambda k, day: last_sale(W, k, day, idx)  # noqa: E731
    slow20 = [cr_sma, "Cruce EMA 50/200 · 3x/letras", w0]
    in20 = [k for k in slow20 if c20.loc[k, "en_3x_el_peor_día"] == 1]
    lrs87 = ("vendió al cierre del lunes 19 y soportó ese día" if sale(LRS, d87) == d87 else
             "estaba en el 3x y soportó ese día") if c87.loc[LRS, "en_3x_el_peor_día"] == 1 else "estaba fuera"
    bullets.append(
        f"**Cracs.** En 1987, con la señal de rentabilidad total, la LRS original {lrs87} un "
        f"{pct_es(c87.loc[LRS, 'rent_peor_día'])} ({pct_es(c87.loc[LRS, 'máximo→mínimo'])} del máximo al "
        f"mínimo del índice). MACD>0 y «SMA 200 y MACD>0» ya habían vendido el "
        f"{y_join(dict.fromkeys(fdate(sale(k, d87)) for k in (mac0, andk)))}: "
        f"{pct_es(c87.loc[mac0, 'máximo→mínimo'])} del máximo al mínimo. En 2020 la LRS "
        f"{'estaba fuera' if c20.loc[LRS, 'en_3x_el_peor_día'] == 0 else 'estaba dentro'} el {fdate(d20)} (había "
        f"vendido el {fdate(sale(LRS, d20))}; {pct_es(c20.loc[LRS, 'máximo→mínimo'])}); "
        + (f"{y_join(short(k) for k in in20)} seguían dentro y perdieron entre el "
           f"{pct_es(-max(c20.loc[in20, 'máximo→mínimo']), 0)} y el {pct_es(-min(c20.loc[in20, 'máximo→mínimo']), 0)}; "
           if in20 else "")
        + f"el Donchian 252/126 vendió el {fdate(sale(don, d20))} ({pct_es(c20.loc[don, 'máximo→mínimo'])}).")
    bullets.append(
        f"**Mercados bajistas.** En los {n_eps} desde 1938 (caídas de al menos un 19% desde un máximo histórico del "
        f"índice), la caída mediana de la LRS original del máximo al mínimo es del {pct_es(bs.loc[LRS, 'mediana_máx→mín'])}. "
        f"«SMA 200 y MACD>0» se queda en el {pct_es(bs.loc[andk, 'mediana_máx→mín'])} y pierde menos en "
        f"{int(bs.loc[andk, 'pierde_menos_que_LRS'])} de {n_eps}; el Donchian 55/20, en el "
        f"{pct_es(bs.loc[don55, 'mediana_máx→mín'])} ({int(bs.loc[don55, 'pierde_menos_que_LRS'])} de {n_eps}). Pero en "
        f"toda la muestra rinden {pct_es(t.loc[andk, 'CAGR'])} y {pct_es(t.loc[don55, 'CAGR'])}: las señales que más "
        "protegen en los cracs son las que más latigazos acumulan fuera de ellos. Con la datación por ciclos "
        f"({len(res['cyc'])} episodios), «SMA 200 y MACD>0» pierde menos que la LRS en "
        f"{int(res['cyc_sum'].loc[andk, 'pierde_menos_que_LRS'])} (mediana {pct_es(res['cyc_sum'].loc[andk, 'mediana_máx→mín'])} "
        f"frente a {pct_es(res['cyc_sum'].loc[LRS, 'mediana_máx→mín'])}).")
    bullets.append(
        f"**Operaciones e impuestos.** La LRS original vende el 3x {num_es(hold.loc['SMA 200', 'ventas_3x_año'], 1)} veces "
        f"al año con una tenencia mediana de {num_es(hold.loc['SMA 200', 'tenencia_mediana_días'], 0)} días naturales; "
        f"solo el {pct_es(hold.loc['SMA 200', 'ventas_largo_plazo_%'], 0)} de las ventas supera el año, y el "
        f"{pct_es(hold.loc['SMA 200', 'ganancia_corto_plazo_%'], 0)} de la ganancia logarítmica realizada es a corto plazo. "
        f"El cruce SMA 50/200 vende {num_es(hold.loc['Cruce SMA 50/200', 'ventas_3x_año'], 2)} veces al año, con "
        f"{num_es(hold.loc['Cruce SMA 50/200', 'tenencia_mediana_días'], 0)} días de mediana y el "
        f"{pct_es(hold.loc['Cruce SMA 50/200', 'ventas_largo_plazo_%'], 0)} de las ventas a largo plazo; el MACD semanal>0 "
        "y el Donchian 252/126 se le parecen. Es el único argumento sólido a favor de las señales lentas, y aquí no se "
        "mide el resultado después de impuestos.")
    rr = fwd["t_rent"].astype(float)
    tv = fwd["t_var"].astype(float)
    bullets.append(
        f"**Mecanismo.** Ninguna señal distingue la rentabilidad de las 21 sesiones siguientes: la diferencia entre estar "
        f"dentro y fuera tiene un t (Hansen-Hodrick) entre {num_es(rr.min(), 1)} y {num_es(rr.max(), 1)}. Todas "
        f"distinguen la varianza (t entre {num_es(tv.min(), 1)} y {num_es(tv.max(), 1)}): con la SMA 200, "
        f"{pct_es(fwd.loc['SMA 200', 'vol_dentro'], 0)} de volatilidad dentro frente al "
        f"{pct_es(fwd.loc['SMA 200', 'vol_fuera'], 0)} fuera, y un {kelly.LSTAR_APROX} de "
        f"{num_es(lst.loc['SMA 200', f'{kelly.LSTAR_APROX} dentro'], 1)} dentro y {num_es(lst.loc['SMA 200', f'{kelly.LSTAR_APROX} fuera'], 1)} "
        f"fuera ({num_es(res['lstar_all'], 2)} en toda la ventana). Tras una salida del MACD>0 el índice rinde, si acaso, "
        f"más ({pp(fwd.loc['MACD>0', 'Δrent_anual_dentro−fuera'])}/año dentro frente a fuera, t = "
        f"{num_es(fwd.loc['MACD>0', 't_rent'], 1)}, no significativo): sale en caídas cortas que se recuperan.")
    bullets.append(
        f"**Después de 2016.** «SMA 200 o MACD>0» fue la mejor: {pct_es(tp.loc[ork, 'CAGR'])} en la pata de letras y "
        f"{pct_es(tp.loc[ork1, 'CAGR'])} en la de 1x, frente al {pct_es(tp.loc[LRS, 'CAGR'])} de la LRS (p nominal "
        f"{fp(tp.loc[ork, 'p_LRS'])} y {fp(tp.loc[ork1, 'p_LRS'])}). Descontada la búsqueda, SPA p = "
        f"{fp(spa_p.p_SPA_c.iloc[0])}. En 1938–1973 y 1974–1999 pierde frente a la LRS "
        f"({pp(sub['1938–1973']['tabla'].loc[ork, 'Δg_LRS'])} y {pp(sub['1974–1999']['tabla'].loc[ork, 'Δg_LRS'])}/año) y en "
        f"toda la muestra {pp(t.loc[ork, 'Δg_LRS'])}. Es una regla que gana en el tramo reciente y no antes: el patrón "
        "de una elegida después de ver los datos.")
    re_ = res["real"]
    best_real = {k: ev["tabla"].loc[list(RULES), "CAGR"].idxmax() for k, ev in re_.items()}
    beat = [k for k, ev in re_.items() if ev["tabla"].loc[best_real[k], "CAGR"] > ev["tabla"].loc[THREE, "CAGR"]]
    head = ("**Con ETFs reales desde 2010, ninguna regla supera al 3x comprar y mantener:** " if not beat else
            f"**Con ETFs reales desde 2010, alguna regla supera al 3x comprar y mantener solo en {y_join(beat)}:** ")
    bullets.append(
        head + "; ".join(
            f"{k}: 3x {pct_es(ev['tabla'].loc[THREE, 'CAGR'])} frente al {pct_es(ev['tabla'].loc[best_real[k], 'CAGR'])} "
            f"de la mejor regla («{best_real[k]}»); SPA de las 28 frente al 3x, p = "
            f"{fp(pd.DataFrame(ev['spa']).p_SPA_c.iloc[-1])}"
            for k, ev in re_.items())
        + (". En ese tramo, salir del 3x costó más de lo que protegió." if not beat else "."))
    bullets.append(
        f"**Ejecución.** Las señales rápidas dependen de operar en el mismo cierre, que antes de 1990 no era operable. "
        f"La autocorrelación diaria del índice de la señal es de {num_es(ac['1938–1989']['índice de la señal (rentabilidad total)'], 3)} "
        f"en 1938–1989 y de {num_es(ac['1990–2026']['índice de la señal (rentabilidad total)'], 3)} en 1990–2026. MACD>señal "
        f"pasa del {pct_es(moc.loc[macs, 'CAGR 1938–1989 cierre siguiente'])} al "
        f"{pct_es(moc.loc[macs, 'CAGR 1938–1989 mismo cierre'])} anual en 1938–1989 si opera al mismo cierre. Una prueba "
        "de señales rápidas al mismo cierre exagera su valor.")
    bullets.append(
        "**Censo.** Las 50 series de este módulo en el S&P 500 (las 25 nuevas y sus gemelas al mismo cierre) entran "
        "en el censo ampliado del artículo 3 (R25, con census_rows y census_meta), que da el SPA, Romano–Wolf y el "
        "número efectivo de pruebas con todas las reglas de R19 y de R21–R24.")
    for b_ in bullets:
        rep.p(f"- {b_}")

    # ── 1. comprobación ──
    rep.h("1. Comprobación de partida")
    rep.p("**Tabla 1. Referencias del artículo 2 recalculadas con r13.strategies, S&P 500 desde el "
          f"{fdate(start)} (orden al cierre siguiente).**")
    rep.table(res["sanity"][["CAGR", "vol", "sharpe", "maxDD", "peor_año", "10k_final"]],
              pct={"CAGR", "vol", "maxDD", "peor_año"}, digits=2)
    rep.p("Las cinco series se construyen otra vez aquí con r13.run_weights y la señal de r13.signal_index; la "
          "diferencia máxima diaria con las de r13.strategies es "
          f"{num_es(max(res['sanity_diff'].values()), 0)}. Las cifras son las del informe de R13 (17,4%, −80,0%, 11,5% "
          "y 14,9%) y las del artículo 2.")

    # ── 2. reglas ──
    rep.h("2. Las reglas probadas")
    rt = pd.DataFrame({"definición (dentro del 3x si…)": DEFINITIONS, "fuente": {k: v[0] for k, v in SOURCES.items()},
                       "tipo": {k: v[1] for k, v in SOURCES.items()}}).reindex(list(ALL_SIGNALS))
    rep.p("**Tabla 2. Las 14 señales: dos de referencia (R13) y doce nuevas.**")
    rep.table(rt, digits=2)
    rep.p("Sin acceso a la red en esta revisión, las referencias bibliográficas marcadas «sin comprobar» (Appel 2005, "
          "Faith 2007, Brock, Lakonishok y LeBaron 1992) no se han cotejado con el original; las reglas se implementan "
          "con sus parámetros estándar. Las dos combinaciones con el MACD las fijó el laboratorio en la exploración "
          "previa a este informe, y no se puede documentar que se fijaran antes de ver resultados. Las señales "
          "semanales usan el último cierre de cada semana; los festivos se conocen de antemano, pero si la semana "
          "acabó antes por un cierre imprevisto (el huracán Gloria en 1985 y el 11-S en 2001) el valor semanal rige "
          "desde la primera sesión de la semana siguiente, cuando ya se sabe que la semana se cerró, y "
          "el Donchian usa cierres porque la base no tiene máximos y mínimos diarios en toda la historia.")
    rep.p("Recuento. 12 señales nuevas × 2 patas (3x/letras y 3x/1x) + «SMA 200 ±2% · 3x/1x» = **25 variantes nuevas** con "
          "la orden al cierre siguiente, y sus 25 gemelas al mismo cierre (sensibilidad, sección 10): **50 variantes** en "
          f"el S&P 500. Las 25 principales se evalúan además en las {r13.N_NDX} reconstrucciones del Nasdaq-100 y con "
          "UPRO y TQQQ desde 2010. Las referencias (1x, 3x, LRS original, rotación y banda) son las de R13.")

    # ── 3. S&P 500 ──
    rep.h(f"3. S&P 500 desde {start.year}")
    cols = ["CAGR", "vol", "sharpe", "maxDD", "peor_año", "cambios_año", "tiempo_en_3x", "Δg_LRS", "IC90_bajo_LRS",
            "IC90_alto_LRS", "p_LRS", "Δg_1x", "p_1x"]
    rep.p(f"**Tabla 3. S&P 500, del {fdate(m['desde'])} al {fdate(m['hasta'])} ({num_es(m['sesiones'], 0)} sesiones): "
          "métricas y diferencia de crecimiento logarítmico (Δg, pp/año) frente a la LRS original y frente al 1x.**")
    rep.table(as_pp(t[cols], ["Δg_LRS", "IC90_bajo_LRS", "IC90_alto_LRS", "Δg_1x"]),
              pct={"CAGR", "vol", "maxDD", "peor_año", "tiempo_en_3x"}, digits=3)
    rep.p("Δg es la diferencia de la media anual de ln(1 + r) con la referencia en los mismos días; IC90 es el intervalo "
          f"percentil del bootstrap estacionario emparejado (bloque medio de {BLOCK} sesiones, {num_es(B, 0)} remuestreos, "
          "los mismos para todas las reglas) y p el unilateral de H0: Δg ≤ 0 con su error típico. Son p nominales: no "
          "descuentan la búsqueda. «cambios_año» cuenta los cambios de exposición (una salida y su vuelta son dos).")
    fig_relative(res, rep)
    rep.h("Contrastes que descuentan la búsqueda", 3)
    rep.p("**Tabla 4. SPA de Hansen (versión consistente, con su recentrado) de cada familia, S&P 500 desde 1938.**")
    rep.table(as_pp(spa_m.set_index("familia"), ["Δg_mejor"]), digits=3)
    rep.p("**Tabla 5. Holm y Romano–Wolf: las 25 nuevas frente a la LRS original y las 28 frente al 1x (mismos "
          "remuestreos que la tabla 3); y, para las 25 nuevas, la dirección contraria («peor que la LRS»).**")
    rep.table(as_pp(mt, ["Δg_LRS", "Δg_1x"]), digits=3)
    rep.p("p_LRS, p_Holm_LRS y p_RW_LRS contrastan H0: Δg ≤ 0 (la variante no crece más que la LRS); p_peor_LRS y "
          "p_RW_peor_LRS, la dirección contraria, H0: Δg ≥ 0 (no crece menos), con p = Φ(t) y el step-down de "
          "Romano–Wolf sobre −t y los mismos remuestreos. Un p_peor bajo dice que la variante es peor que la LRS; "
          "con p_RW_peor, descontada la búsqueda de las 25. "
          + (("Descontada la búsqueda, " + ("es peor que la LRS original "
                                              if int((mt["p_RW_peor_LRS"] < 0.05).sum()) == 1 else
                                              "son peores que la LRS original: ")
              + y_join([f"«{k}» (p_RW_peor {fp(mt.loc[k, 'p_RW_peor_LRS'])})"
                        for k in mt.index[mt["p_RW_peor_LRS"] < 0.05]]) + ".")
             if (mt["p_RW_peor_LRS"] < 0.05).any() else
             "Ninguna es peor que la LRS original con la búsqueda descontada (p_RW_peor ≥ 0,05 en todas)."))
    rep.p(f"Frente a la LRS original, {n_neg} de las 25 variantes nuevas crecen menos que ella y ninguna tiene un p "
          f"ajustado por debajo del 5%. Frente al 1x, {n_p1} de 28 tienen un p nominal inferior al 5%, pero el SPA "
          f"de la familia da {fp(spa_m.p_SPA_c.iloc[2])} y "
          + ("ningún p de Romano–Wolf baja" if not (rw1 < 0.05).any() else f"{int((rw1 < 0.05).sum())} p de Romano–Wolf bajan")
          + " del 5%. "
          f"Tratada como una más de las 28, la LRS original tendría un p de Romano–Wolf de "
          f"{fp(mt.loc[LRS, 'p_RW_1x'])}; no es su caso, porque es la hipótesis que Gayed y Bilello fijaron antes.")

    # ── 4. subperiodos ──
    rep.h("4. Subperiodos y tras la publicación")
    wins = [(lab, sub[lab]) for lab in sub_labs] + [(POST[0], post), ("1929–1938", res["early"])]
    ct = pd.DataFrame({f"{lab} ({ev['desde'].year}–{ev['hasta'].year})": ev["tabla"]["CAGR"] for lab, ev in wins})
    dt = pd.DataFrame({lab: ev["tabla"]["Δg_LRS"] * 100 for lab, ev in wins})
    pt = pd.DataFrame({lab: ev["tabla"]["p_LRS"] for lab, ev in wins})
    rep.p("**Tabla 6. Rentabilidad anual compuesta por subperiodo (S&P 500).** La última columna es la Gran Depresión, "
          f"del {fdate(res['early']['desde'])} (primer día con todas las señales) al {fdate(res['early']['hasta'])}, "
          "fuera de la ventana del artículo 2.")
    rep.table(ct.reindex(list(NAMES)), pct=set(ct.columns), digits=3)
    rep.p("**Tabla 7. Δg frente a la LRS original por subperiodo (pp/año) y su p nominal.**")
    both = pd.concat([dt.add_suffix(" Δg (pp)"), pt.add_suffix(" p")], axis=1)
    both = both[[c for lab, _ in wins for c in (f"{lab} Δg (pp)", f"{lab} p")]]
    rep.table(both.reindex(list(RULES)), digits=3)
    rep.p(f"**Tabla 8. SPA tras la muestra de Gayed y Bilello ({fdate(post['desde'])} → {fdate(post['hasta'])}).**")
    rep.table(as_pp(spa_p.set_index("familia"), ["Δg_mejor"]), digits=3)
    et = res["early"]["tabla"]
    win_e = [k for k in TB_RULES if k != LRS and et.loc[k, "Δg_LRS"] > 0]
    slow_e = [cr_sma, "Cruce EMA 50/200 · 3x/letras", w0]
    win_txt = ", ".join(f"{short(k)} ({pct_es(et.loc[k, 'CAGR'])})" for k in win_e) or "ninguna"
    rep.p(f"Tras 2016 la mejor es «{spa_p.mejor.iloc[0]}» (Δg {pp(spa_p.Δg_mejor.iloc[0])}/año frente a la LRS), con "
          f"un SPA de {fp(spa_p.p_SPA_c.iloc[0])}. En la Gran Depresión (1929–1938) la LRS original rinde "
          f"{pct_es(et.loc[LRS, 'CAGR'])} al año; en la pata de letras la superan {win_txt}, "
          f"sin significación (p nominal mínimo {fp(et.loc[win_e, 'p_LRS'].min()) if win_e else '—'}), y las lentas "
          f"({', '.join(short(k) for k in slow_e)}) pierden entre el {pct_es(-et.loc[slow_e, 'CAGR'].max())} y el "
          f"{pct_es(-et.loc[slow_e, 'CAGR'].min())} al año: con el 3x, salir tarde de un crac largo es lo más caro.")

    # ── 5. EMA frente a SMA ──
    rep.h("5. EMA 200 frente a SMA 200")
    rep.p("**Tabla 9. Diferencia de log-riqueza acumulada (EMA − SMA, pata de letras) por años naturales, S&P 500 "
          "desde 1938.**")
    rep.table(res["ema_years"], digits=3)
    rep.p("**Tabla 10. Δg frente a la LRS original (pp/año) según la fecha de inicio (hasta 2026-09-25), pata de letras.**")
    rep.table(res["ema_starts"] * 100, digits=2)
    if "ndx_ema_years" in res:
        ny = res["ndx_ema_years"]
        rep.p(f"En el Nasdaq-100 (reconstrucción 0, desde 1985-10) la EMA gana {ny['años_que_gana']} de {ny['años']} años y "
              f"acumula {num_es(ny['log_total'], 2)} en logaritmo; sin sus 5 mejores años, {num_es(ny['sin_5_mejores'], 2)}, "
              f"y con recorte simétrico, {num_es(ny['recorte_simétrico'], 2)}.")
    ndx_p2, sig_n = "", False
    if "ndx" in res:
        p2s = [2 * min(x["ev"]["tabla"].loc[ema, "p_LRS"], 1 - x["ev"]["tabla"].loc[ema, "p_LRS"]) for x in res["ndx"]]
        sig_n = float(np.median(p2s)) < 0.05
        ndx_p2 = f"; en el Nasdaq-100, a favor de la EMA, p = {fp(float(np.median(p2s)))} (mediana de las {len(p2s)} reconstrucciones)"
    n_sig = int(p_ema2 < 0.05) + int(sig_n)
    sig_txt = {0: "Ninguna de las dos diferencias es significativa al 5%",
               1: "Solo una de las dos es significativa al 5%", 2: "Las dos son significativas al 5%"}[n_sig]
    rep.p("La EMA reacciona antes que la SMA porque pesa más los últimos cierres. En el S&P 500 eso se traduce en más "
          "cambios y en peores resultados en todos los subperiodos; en el Nasdaq-100, en mejores (tabla 24). La "
          "pregunta «¿SMA o EMA?» estaba fijada antes de mirar, así que se contrasta sola y en las dos direcciones: "
          f"en el S&P 500 el p bilateral de Δg = 0 es {fp(p_ema2)}{ndx_p2}. {sig_txt}, y en la dirección «peor», "
          "con la búsqueda de las 25 descontada, el p de Romano–Wolf de "
          f"la EMA en el S&P 500 es {fp(p_ema_rw)} (tabla 5). Con dos historias muy correlacionadas no hay forma de "
          "distinguir una razón de fondo del azar.")

    # ── 6. salidas ──
    rep.h("6. Salidas acertadas y latigazos")
    rep.p("**Tabla 11. Descomposición de las salidas frente al 3x comprar y mantener, S&P 500 desde 1938 (pp/año de "
          "crecimiento logarítmico).**")
    xcols = ["ganancia_salidas_acertadas", "coste_latigazos", "valor_neto_salidas", "coste_operar", "Δg_frente_3x"]
    rep.table(as_pp(ex.drop(columns="error_identidad"), xcols), pct={"latigazos_%", "latigazos_cortos_%"}, digits=2)
    rep.p("Cada tramo fuera del 3x vale lo que rindió la alternativa (letras o 1x) menos lo que rindió el 3x en esos "
          "mismos días, en logaritmos. Si es positivo, la salida acertó: esquivó una caída. Si es negativo o cero, fue "
          "un latigazo. «latigazos_cortos_%» es la parte de los latigazos que duró 21 sesiones o menos. La suma de los "
          "tramos más el coste de operar es exactamente la diferencia con el 3x (error máximo "
          f"{num_es(ex['error_identidad'].max(), 0)}).")
    fig_whipsaws(res, rep)

    # ── 7. cracs ──
    rep.h("7. Cracs y mercados bajistas")
    keys = [ONE, THREE] + list(TB_RULES)
    for lab in CRASH_DAYS:
        c_ = crash.xs(lab).reindex(keys).copy()
        c_["órdenes"] = c_["órdenes"].map(truncate_orders)
        e = res["eps_all"][(res["eps_all"]["máximo"] <= CRASH_DAYS[lab]) & (res["eps_all"]["mínimo"] >= CRASH_DAYS[lab])].iloc[0]
        rep.p(f"**Tabla {12 if lab == '1987' else 13}. Crac de {lab}: órdenes ejecutadas entre el {fdate(CRASH_VIEW[lab][0])} "
              f"y el {fdate(CRASH_VIEW[lab][1])} (fecha del cierre de ejecución), si cada regla estaba en el 3x el "
              f"{fdate(CRASH_DAYS[lab])} y su rentabilidad del máximo del índice ({fdate(e['máximo'])}) al mínimo "
              f"({fdate(e['mínimo'])}) y a la recuperación ({fdate(e['recuperación'])}).**")
        rep.table(c_, pct={"rent_peor_día", "máximo→mínimo", "máximo→recuperación"}, digits=0)
    fig_crashes(res, rep)
    piv = res["bear"].pivot(index="regla", columns="episodio", values="máximo→mínimo")
    piv = piv[list(res["eps_all"]["episodio"])].reindex(keys)
    rep.p("**Tabla 14. Rentabilidad del máximo al mínimo del índice en cada mercado bajista (caída ≥ 19% desde un máximo "
          "histórico del S&P 500 de precio).** La primera columna (1929–32) queda fuera de la ventana principal y de los "
          "resúmenes.")
    rep.table(piv, pct=set(piv.columns), digits=0)
    rep.p(f"**Tabla 15. Resumen de los {n_eps} mercados bajistas desde 1938.** «rápidos»: {int((eps.sesiones_hasta_mínimo <= FAST).sum())} "
          f"con {FAST} sesiones o menos del máximo al mínimo "
          f"({', '.join(eps.loc[eps.sesiones_hasta_mínimo <= FAST, 'episodio'])}).")
    rep.table(bs, pct={"mediana_máx→mín", "mediana_máx→mín_rápidos", "mediana_máx→recuperación"}, digits=0)
    rep.p(f"**Tabla 16. Sensibilidad: datación por ciclos ({len(res['cyc'])} episodios desde 1938).** Un mercado bajista "
          "empieza en el máximo del ciclo tras caer un 19% y acaba en el mínimo tras subir un 19%; incluye 1938–49, "
          "1976–78 y 2011 y parte 2000–02 y 2007–09 en varios tramos.")
    rep.table(res["cyc_sum"], pct={"mediana_máx→mín", "mediana_máx→mín_rápidos"}, digits=0)
    fig_tradeoff(res, rep)
    prot = bs.loc[list(TB_RULES), "mediana_máx→mín"].sort_values(ascending=False).index[:5].tolist()
    rep.p(f"Las cinco señales que menos pierden en los mercados bajistas ({', '.join(short(k) for k in prot)}) crecen "
          f"entre el {pct_es(t.loc[prot, 'CAGR'].min())} y el {pct_es(t.loc[prot, 'CAGR'].max())} al año en el conjunto, "
          f"frente al {pct_es(t.loc[LRS, 'CAGR'])} de la LRS: sus salidas acertadas no compensan los latigazos de los "
          "mercados alcistas. Las lentas (cruces 50/200, MACD semanal>0, Donchian 252/126) operan poco pero llegan tarde "
          f"a los cracs rápidos (mediana en los rápidos entre el {pct_es(bs.loc[[cr_sma, w0, don], 'mediana_máx→mín_rápidos'].min())} "
          f"y el {pct_es(bs.loc[[cr_sma, w0, don], 'mediana_máx→mín_rápidos'].max())}, frente al "
          f"{pct_es(bs.loc[LRS, 'mediana_máx→mín_rápidos'])} de la LRS). La SMA 200 queda en medio en rapidez y en "
          "protección, y arriba en crecimiento.")

    # ── 8. operaciones e impuestos ──
    rep.h("8. Operaciones, tenencia e impuestos")
    rep.p("**Tabla 17. Ventas del 3x, tenencia y plazo fiscal, S&P 500 desde 1938.** La pata 1x tiene las mismas ventas del "
          "3x y además vende el 1x en cada vuelta al 3x.")
    rep.table(hold, pct={"ventas_largo_plazo_%", "ganancia_corto_plazo_%"}, digits=2)
    rep.p("La tenencia se cuenta en días naturales del cierre de compra al de venta; una venta es a largo plazo si supera "
          f"{LONG_TERM_DAYS} días (el umbral de EE. UU.). «ganancia_corto_plazo_%» es la parte de la ganancia logarítmica "
          "de las tenencias ganadoras que se realiza a corto plazo. No es un cálculo del resultado después de impuestos: "
          "no hay dólares, ni compensación de pérdidas, ni tipos.")

    # ── 9. mecanismo ──
    rep.h("9. Mecanismo: qué distingue cada señal")
    rep.p(f"**Tabla 18. {kelly.LSTAR_APROX} ({kelly.LSTAR_DESC[kelly.LSTAR_APROX]}) en los días dentro y fuera del 3x de "
          "cada señal, con su rentabilidad y volatilidad anuales, S&P 500 desde 1938.**")
    rep.table(lst, pct={"días_dentro_%", "rent_anual_dentro", "rent_anual_fuera", "vol_dentro", "vol_fuera"}, digits=2)
    rep.p(f"En toda la ventana, el {kelly.LSTAR_APROX} es {num_es(res['lstar_all'], 2)} (el 2,32 del artículo 1 se estimó "
          "desde 1960). Es una descripción dentro de la muestra, no un apalancamiento recomendable: el L* estimado tiene "
          "un error grande (artículo 1).")
    rep.p(f"**Tabla 19. Rentabilidad en exceso de las letras (log, anualizada) y volatilidad de las {H_FWD} sesiones que "
          f"cobraría la orden de cada día, según el estado de la señal. MCO sobre el estado, con errores de "
          f"{fwd['estimador'].iloc[0]} (stats.hac_overlap({H_FWD})).**")
    ft = fwd.drop(columns=["estimador"]).astype(float)
    rep.table(ft, pct={"Δrent_anual_dentro−fuera", "vol_dentro", "vol_fuera"}, digits=2)
    lin, lout = lst[f"{kelly.LSTAR_APROX} dentro"], lst[f"{kelly.LSTAR_APROX} fuera"]
    gap = (lin - lout)
    rep.p("Es el hallazgo del artículo 1 visto desde las señales: la varianza se puede anticipar y la rentabilidad no. "
          f"Todas las señales salen cuando la volatilidad es alta, y el L* de los días dentro es mayor que el de los días "
          f"fuera en {int((gap > 0).sum())} de {len(gap)} (de {num_es(lin[gap.idxmin()], 1)} frente a "
          f"{num_es(lout[gap.idxmin()], 1)} con {gap.idxmin()} a {num_es(lin[gap.idxmax()], 1)} frente a "
          f"{num_es(lout[gap.idxmax()], 1)} con {gap.idxmax()}). Ninguna separa con significación la rentabilidad "
          "siguiente; en las rápidas, el punto estimado va al revés (fuera rinde más), y cada salida de ésas es un "
          "latigazo con el 3x.")

    # ── 10. mismo cierre ──
    rep.h("10. Sensibilidad: ejecución en el mismo cierre")
    rep.p("**Tabla 20. Rentabilidad anual con la orden al cierre siguiente (el protocolo) y al mismo cierre, antes y "
          "después de 1990.**")
    rep.table(moc, pct=set(moc.columns), digits=3)
    rep.p("**Tabla 21. Autocorrelación de primer orden de las rentabilidades diarias.**")
    rep.table(pd.DataFrame(ac), digits=3)
    rep.p("Antes de 1990 los precios de cierre del índice arrastraban retrasos (acciones poco negociadas), y la "
          "rentabilidad de un día anticipaba la del siguiente. Una señal rápida calculada y ejecutada en el mismo cierre "
          "se aprovecha de eso, pero ese cierre no era operable. Las 25 variantes al mismo cierre se cuentan en el "
          "recuento y se entregan al censo marcadas como no operables.")

    # ── 11. Nasdaq-100 ──
    if "ndx" in res:
        rep.h(f"11. Nasdaq-100 ({r13.N_NDX} reconstrucciones, evaluado desde 1985-10-01)")
        rep.p("La historia del Nasdaq-100 anterior a 1985 se reconstruye (R12) y solo sirve para calentar las medias; las "
              "señales coinciden en las 10 reconstrucciones desde 1987–88. El 3x es sintético hasta TQQQ (2010-02-11).")
        rep.p("**Tabla 22. Nasdaq-100: métricas y Δg (pp/año), mediana, mínimo y máximo de las 10 reconstrucciones.**")
        rep.table(as_pp(na, ["Δg_LRS mediana", "Δg_1x mediana"]),
                  pct={"CAGR mediana", "CAGR mín", "CAGR máx", "maxDD mediana"}, digits=3)
        rep.p("**Tabla 23. SPA por familia en las 10 reconstrucciones.**")
        rep.table(as_pp(ns, ["Δg_mejor mediana"]), digits=3)
        rep.p("**Tabla 24. Nasdaq-100 por subperiodo: CAGR, Δg frente a la LRS (pp/año) y p nominal (medianas de las 10 "
              "reconstrucciones).**")
        s_ = ndx_sub(res)
        rep.table(as_pp(s_, [c for c in s_.columns if c.startswith("Δg")]),
                  pct={c for c in s_.columns if c.startswith("CAGR")}, digits=3)
        rep.p("**Tabla 25. Nasdaq-100 (reconstrucción 0): salidas y latigazos (pp/año) y ventas del 3x.**")
        rep.table(as_pp(res["ndx_exits"].drop(columns="error_identidad"), xcols),
                  pct={"latigazos_%", "latigazos_cortos_%"}, digits=2)
        rep.table(res["ndx_hold"], pct={"ventas_largo_plazo_%", "ganancia_corto_plazo_%"}, digits=2)
        ne = res["ndx_eps"]
        rep.p(f"**Tabla 26. Nasdaq-100: resumen de los {len(ne)} mercados bajistas desde 1985-10 (caída ≥ 19% desde un "
              f"máximo histórico del índice de precio aproximado: {', '.join(ne.episodio)}).**")
        rep.table(res["ndx_bear_sum"], pct={"mediana_máx→mín", "mediana_máx→mín_rápidos", "mediana_máx→recuperación"},
                  digits=0)
        up = [k for k in TB_RULES if k != LRS and na.loc[k, "Δg_LRS mediana"] > 0.02]
        up_txt = ", ".join(f"{short(k)} ({pp(na.loc[k, 'Δg_LRS mediana'])})" for k in up)
        rep.p(f"En el Nasdaq-100 crecen más que la LRS original, con más de 2 pp/año de mediana, {up_txt}; el SPA de las "
              f"25 no baja de {fp(ns.iloc[0]['p_SPA mín'])} en ninguna reconstrucción. Las rápidas vuelven a proteger más "
              "en los mercados bajistas (tabla 26) y a crecer menos.")

    # ── 12. semiconductores ──
    n_tab = 27
    if "semis" in res:
        se = res["semis"]
        rep.h(f"12. Semiconductores (desde el {fdate(se['desde'])})")
        rep.p("El índice de semiconductores de R13 (con dividendos; 3x sintético hasta SOXL, real después), con las "
              "mismas 28 reglas sobre su propio índice de rentabilidad total. La ventana empieza el primer día con "
              "previsión GJR de R13, como su tabla de semis, para que las referencias coincidan con las de R13 "
              f"(diferencia máxima {num_es(max(res['semis_sanity'].values()), 0)}).")
        rep.p(f"**Tabla {n_tab}. Semiconductores, del {fdate(se['desde'])} al {fdate(se['hasta'])} "
              f"({num_es(se['sesiones'], 0)} sesiones): métricas y Δg (pp/año) frente a la LRS original y al 1x.**")
        rep.table(as_pp(se["tabla"][cols], ["Δg_LRS", "IC90_bajo_LRS", "IC90_alto_LRS", "Δg_1x"]),
                  pct={"CAGR", "vol", "maxDD", "peor_año", "tiempo_en_3x"}, digits=3)
        rep.p(f"**Tabla {n_tab + 1}. SPA por familia, semiconductores.**")
        rep.table(as_pp(pd.DataFrame(se["spa"]).set_index("familia"), ["Δg_mejor"]), digits=3)
        sb = res["semis_bear"]
        se_eps = res["semis_bear_eps"]
        rep.p(f"**Tabla {n_tab + 2}. Semiconductores: resumen de los {len(se_eps)} mercados bajistas desde "
              f"{se['desde'].year} (caída ≥ 19% desde un máximo histórico del índice de precio aproximado: "
              f"{', '.join(se_eps.episodio)}).**")
        rep.table(sb, pct={"mediana_máx→mín", "mediana_máx→mín_rápidos", "mediana_máx→recuperación"}, digits=0)
        st = se["tabla"]
        mts = se["múltiples"]
        up_s = [k for k in NEW if st.loc[k, "Δg_LRS"] > 0]
        n_rw_s = int((mts.loc[list(NEW), "p_RW_LRS"] < 0.05).sum())
        rep.p(f"En semiconductores crecen más que la LRS {len(up_s)} de las 25 nuevas"
              + (f" (la mejor, «{st.loc[list(NEW), 'Δg_LRS'].idxmax()}», {pp(st.loc[list(NEW), 'Δg_LRS'].max())}/año)"
                 if up_s else "")
              + ("; Romano–Wolf frente a la LRS no deja ninguna por debajo del 5%" if n_rw_s == 0 else
                 f"; Romano–Wolf frente a la LRS deja {n_rw_s} por debajo del 5%")
              + f". La LRS original frente al 1x: Δg {pp(st.loc[LRS, 'Δg_1x'])}/año, p = {fp(st.loc[LRS, 'p_1x'])}. "
              "Es la muestra más corta y más volátil del laboratorio: las diferencias entre reglas son grandes y "
              "los intervalos, más.")
        n_tab += 3

    # ── 13. ETFs reales ──
    rep.h(f"13. ETFs reales desde {r13.REAL_FROM}")
    for i, (k, ev) in enumerate(re_.items()):
        rep.p(f"**Tabla {n_tab + 2 * i}. {k}, del {fdate(ev['desde'])} al {fdate(ev['hasta'])}: 3x real, 1x y reglas.**")
        rep.table(as_pp(ev["tabla"][cols], ["Δg_LRS", "IC90_bajo_LRS", "IC90_alto_LRS", "Δg_1x"]),
                  pct={"CAGR", "vol", "maxDD", "peor_año", "tiempo_en_3x"}, digits=3)
        rep.p(f"**Tabla {n_tab + 1 + 2 * i}. SPA, {k}.**")
        rep.table(as_pp(pd.DataFrame(ev["spa"]).set_index("familia"), ["Δg_mejor"]), digits=3)
    rep.p("La pata 1x es el índice (SPY, QQQ o el índice de semiconductores ajustados), sin comisión ni horquilla de "
          "compraventa: favorece un poco a las rotaciones 3x/1x frente a las de letras. En semis el 3x es sintético "
          f"hasta el primer día de SOXL en la base ({fdate(r12.real_letf('SOXL').first_valid_index())}).")
    fig_markets(res, rep)

    # ── 14. censo ──
    rep.h("14. Estas reglas en el censo")
    rep.p("census_rows() devuelve las 50 series netas del S&P 500 (las 25 nuevas con la orden al cierre siguiente y sus "
          "25 gemelas al mismo cierre, «… · mismo cierre», que r19.is_same_close marca como no operables) en el índice "
          "diario completo, y census_meta() sus fichas. El censo ampliado del artículo 3 (R25) las añade a las 222 "
          "reglas de R19 junto con las de R22–R24 y da el SPA frente a la LRS original y frente al 1x, Romano–Wolf y "
          "el número efectivo de pruebas.")

    # ── 15. limitaciones ──
    rep.h("15. Limitaciones")
    for s in (
        "Todo es dentro de muestra. Las señales se fijaron antes de este informe con sus parámetros estándar, pero salen "
        "de una literatura popular que conoce esta historia, y las dos combinaciones con el MACD son del laboratorio.",
        f"Los p son unilaterales (H0: Δg ≤ 0) con el error típico del bootstrap estacionario; con {num_es(B, 0)} "
        "remuestreos, el error de Monte Carlo cerca de p = 0,05 es de unos ±0,005.",
        "El Nasdaq-100 anterior a 1985 está reconstruido; solo afecta al calentamiento de las señales.",
        "Costes: 5 pb por unidad de rotación, como en el artículo 2. Las reglas de 8 a 21 cambios al año (MACD diario, "
        "«SMA 200 y MACD>0») son las más expuestas a horquillas y deslizamientos reales: con más fricción quedarían peor.",
        "Impuestos: solo se miden la rotación y los plazos de tenencia, no el resultado tras impuestos.",
        "Los mercados bajistas se fechan con el índice de precio y máximos históricos; la datación por ciclos es una "
        "sensibilidad con más episodios.",
    ):
        rep.p(f"- {s}")


def run():
    rep = Report("r21_trend", "R21 — Señales de tendencia más allá de la media de 200 sesiones")
    for f in rep.dir.glob("table_*.csv"):       # Report numera las tablas por línea: sin huérfanas
        f.unlink()
    rep.p(__doc__.split("\n", 2)[2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = compute()
    write_report(res, rep)
    rep.save()
    return res


if __name__ == "__main__":
    run()
