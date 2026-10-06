"""R28 — ¿Se puede cronometrar la prima de la volatilidad? H1, H2 y H4 del artículo 4.

Pregunta. El corto de futuros del VIX gana de media porque el futuro cotiza por encima del
VIX que acaba liquidando (la prima). ¿Se puede saber cuándo? Tres hipótesis del pre-registro
(letf/preregistro/articulo4_volatilidad.md, commit 7cded63), contrastadas tal como están
escritas allí; H3 y H5 van en otra línea:
- H1: la prima ex ante de Cheng (2019) (futuro menos una previsión del VIX estimada solo con
  datos pasados) predice el rendimiento a un mes del futuro, con β > 0 (HAC, 5%, una cola),
  fuera de su muestra: 2016-01 → 2026-09.
- H2: ningún filtro de la familia {F1 > VIX, VIX < VIX3M, F2 > F1, roll diario > 0,10
  (Simon y Campasano 2014), prima de Cheng > 0}, aplicado a un −0,5x, supera a estar siempre
  −0,5x con p_SPA < 0,10; si alguno lo hace, se repite quitando sus 3 mejores días y, si
  entonces Δg ≤ 0, se informa como frágil.
- H4: escalar el −0,5x por la varianza prevista, a igual exposición media, no mejora Δg
  (p_SPA > 0,10).

Datos: letf_lab.data.vix (R26): futuros VX de CFE, VIX y VIX3M de los CSV de Cboe (nunca el
^VIX de Yahoo), el −0,5x del pre-registro (vix.minus_half_spliced: sintético hasta el
27-2-2018 y SVXY real desde el 28-2-2018) y la letra del laboratorio.

Protocolo común (el del laboratorio y el del pre-registro): señal al cierre de t y orden al
cierre de t+1 (LAG = 2); el «mismo cierre» (LAG = 1) se da solo como referencia no operable
(la liquidación del VX y el cierre del VIX son de las 16:15 ET hasta 2020, después del cierre
de los ETPs); 5 pb por unidad negociada; la caja cobra la letra; Δg (crecimiento logarítmico
anual) con bootstrap estacionario emparejado de bloque medio 63 (2.000 remuestreos, los mismos
para toda la familia), p unilateral normal con el error del bootstrap, step-down de Romano y
Wolf (r19_census.romano_wolf) y SPA de Hansen (stats.spa_test, versión consistente) por
familia. Cada regla y cada variante calculada cuenta en su familia.

Decisiones de implementación que el pre-registro no detalla, fijadas en este módulo antes de
ejecutar los contrastes:

H1.
- Previsión del VIX: ARMA(2,2) en niveles, como Cheng, sobre el VIX de Cboe desde el
  2-1-1990, por mínimos cuadrados condicionales (CSS), reestimado en la última sesión de cada
  mes con los datos hasta esa sesión (ventana creciente) y usado hasta la reestimación
  siguiente: solo datos pasados. Sensibilidad: los coeficientes que publica Cheng (estimados
  con datos anteriores a 2004).
- Estrategia de Cheng: en cada sesión lleva el contrato que vence el mes siguiente; en la
  última sesión del mes pasa al que vence dentro de dos meses (si ese mes no cotiza, el primero
  que venza después). Prima del día escalada a un mes: VP_t = 21/h_t·(F_t − Ê_t[VIX_T]), con
  h_t las sesiones hasta el vencimiento; rendimiento esperado del mes, escalado a un mes como
  la prima: ER_t1 = 21/h·(Ê_t1[VIX_T]/F_t1 − 1) en la última sesión del mes anterior.
  Corrección declarada: la primera versión usaba Ê/F − 1 sin escalar; la comprobación con la
  muestra de Cheng dio entonces β = 0,565 (0,176), 21/34 de su 0,92 (0,29) con el mismo R², y
  con la escala sale 0,908 (0,289). El informe da las dos versiones fuera de muestra.
- Contraste principal: regresión mensual del rendimiento del mes del contrato sobre ER_t1,
  con la orden al cierre siguiente (entra al cierre de t1+1 y sale al cierre de t2+1, con t2
  la última sesión del mes), errores de Newey-West con 3 retardos (los de Cheng), meses
  2016-01 → 2026-08 (el último mes completo antes del 25-9-2026). β > 0 con p unilateral <
  0,05. β > 0 aquí equivale a que una prima mayor anuncie un rendimiento mayor del corto.
- Sensibilidades: mismo cierre (la cronología de Cheng, no operable); Newey-West con 12
  retardos (stats.NONOV_LAGS); regresión diaria con el rendimiento de las 21 sesiones
  siguientes y Hansen-Hodrick (stats.hac_overlap(21)); coeficientes publicados; meses hasta
  2018-02 y desde 2018-03; sin el 5 y el 6-2-2018. Comprobación: la muestra de Cheng
  (2004-04 → 2015-12, 141 meses) con sus coeficientes y su cronología. Diagnóstico añadido
  después de ver los resultados (no pre-registrado): β quitando cada mes (h1_influence).

H2.
- F1, F2: los dos primeros contratos listados con vencimiento posterior a t (settle de CFE);
  VIX y VIX3M de Cboe. Roll diario de Simon y Campasano: (F − VIX)/n con F el primer contrato
  con más de 10 sesiones hasta el vencimiento y n esas sesiones. Prima de Cheng: VP_t de H1.
- Regla: −0,5x cuando la condición se cumple al cierre de t, letras si no. Sin dato (VIX3M
  empieza en Cboe el 18-9-2009) la regla lleva el −0,5x, igual que la referencia.
- Ventana principal 21-12-2005 → 25-9-2026, convención «liquidación» (la de los fondos de
  ProShares y del índice de S&P); veredicto con p_SPA < 0,10 de la familia de 5 (versión
  consistente, p_SPA_c, como en R21–R25; la inferior se da al lado). Si se
  rechaza, pasan la regla de fragilidad los filtros con p de Romano–Wolf < 0,10 (o el mejor,
  si ninguno): se quitan sus 3 días con mayor ventaja diaria sobre la referencia.
- Sensibilidades del pre-registro: antes y desde el 28-2-2018; 16:00; sin el 5 y el
  6-2-2018 en las dos convenciones; SVXY real desde 2011-10 (½ SVXY −1x + ½ letras hasta el
  27-2-2018); ventana con VIX3M (desde el 22-9-2009); mismo cierre (no operable).

H4.
- Previsiones de la varianza diaria del propio −0,5x, conocidas al cierre de t: RV 21 (media
  de r² de las 21 sesiones previas, como Moreira y Muir 2017), EWMA con λ = 0,94 y
  GJR-GARCH(1,1) (stats.garch_fit, reestimado cada 1 de enero con los datos anteriores).
  Antes del 21-12-2005 se calientan con −0,5 × el índice corto (solo para prever; 325
  sesiones con dato desde el 21-4-2004): el GJR se estima por primera vez con 150 de ellas.
- Peso en el −0,5x: w_t = mín(c/σ̂²_t, 2), el resto en letras (con w > 1, préstamo a la letra
  + 1,5%, el diferencial minorista de R22). «Igual exposición media» (el pre-registro): c tal
  que la media de w en la ventana evaluada sea 1 (ex post, como Moreira y Muir). Familia de H4:
  las 3 previsiones con esa normalización, frente a siempre −0,5x. Variante en tiempo real,
  operable: c_t = 1/media creciente de 1/σ̂² (mínimo 150 sesiones); su exposición media queda
  por debajo de 1 (unos dos tercios), así que no cumple la condición del pre-registro y va
  aparte, como familia de sensibilidad de 3 reglas con su propio SPA. (Hasta la revisión del
  laboratorio las 6 reglas formaban una sola familia; el cambio se hizo con los resultados ya
  vistos.) Con w = 2 la exposición llega a −1x,
  así que se dan las dos convenciones y con y sin el 5 y el 6-2-2018 (quitados antes de prever).
  Un día de −100% arruina la cuenta (vix.absorb).
- Mismas ventanas y sensibilidades que H2.
"""

from __future__ import annotations

import warnings
import zlib
from functools import lru_cache

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.signal import lfilter
from scipy.stats import norm

from letf_lab import metrics, stats
from letf_lab.data import vix
from letf_lab.report import FIG_W, Report, comma_fig, legend_band, num_es, pct_es
from letf_lab.research.r19_census import romano_wolf
from letf_lab.research.r26_vix_data import Tables, dmy

SEED = 28
B = 2000                   # remuestreos del bootstrap y del SPA (como R13, R19 y R26)
BLOCK = 63                 # bloque medio del bootstrap estacionario (sesiones)
LAG = 2                    # señal al cierre de t, orden al cierre de t+1 (r13.EXEC_LAG)
SAME = 1                   # mismo cierre: referencia no operable
COST_BPS = 5.0             # por unidad negociada
ANN = 252
CLIP = -0.9999             # pérdidas acotadas al tomar logaritmos (r13.log_growth)
END = vix.END
FIRST = pd.Timestamp("2005-12-21")        # primer rendimiento del −0,5x (índice corto desde el 20-12-2005)
PRE_END = vix.DESIGN_2018 - pd.Timedelta(days=1)
VIX3M_FIRST = pd.Timestamp("2009-09-22")  # primera sesión que cobra la regla con VIX3M de Cboe (dato del 18-9-2009)
BENCH = "siempre −0,5x"
LRS = "LRS 3x (original)"
CENSUS = vix.sources.ROOT / "reports" / "r19_census" / "censo_rentabilidades_diarias.csv.gz"

# ───────────── H1 ─────────────
VIX_FROM = pd.Timestamp("1990-01-02")
FIRST_FIT = pd.Timestamp("2003-12-31")    # primera reestimación del ARMA (antes del primer futuro)
CHENG_PUB = {"mu": 20.083, "phi1": 1.651, "phi2": -0.654, "th1": -0.714, "th2": -0.064}
CHENG_BETA = (0.92, 0.29)                 # su tabla 4, fila 1: β y error de Newey-West (3 retardos)
CHENG_MONTHS = ("2004-04", "2015-12")     # sus 141 meses
OOS_MONTHS = ("2016-01", "2026-08")
SPLIT_MONTH = "2018-02"                   # antes: meses ≤ 2018-02; después: ≥ 2018-03
NW_CHENG = 3
NW_LAB = stats.NONOV_LAGS
H_D = 21
ONE_MONTH = 21

# ───────────── H2 ─────────────
F_F1 = "F1 > VIX"
F_TS = "VIX < VIX3M"
F_F2 = "F2 > F1"
F_SC = "roll diario > 0,10"
F_CH = "prima de Cheng > 0"
FILTERS = (F_F1, F_TS, F_F2, F_SC, F_CH)
SC_MIN = 10
SC_THR = 0.10
P_SPA = 0.10
TOP_DAYS = 3

# ───────────── H4 ─────────────
FORECASTS = ("RV 21", "EWMA 0,94", "GJR")
NORMS = ("ex post", "tiempo real")
MANAGED = tuple(f"{f} · {n}" for n in NORMS for f in FORECASTS)
EXPOST = tuple(f"{f} · ex post" for f in FORECASTS)          # familia de H4 (igual exposición media)
REALTIME = tuple(f"{f} · tiempo real" for f in FORECASTS)    # sensibilidad operable (exposición < 1)
W_MAX = 2.0
MARGIN_EXTRA = 0.015
RV_N = 21
EWMA_L = 0.94
RT_MIN = 150               # sesiones mínimas de la media creciente de 1/σ̂² (normalización en tiempo real)
GJR_MIN = 150              # sesiones de la primera estimación del GJR (dentro del calentamiento)

# ───────────── escenarios (H2 y H4) ─────────────
# nombre → (convención, producto real, sin 5-6-2-2018, lag, desde, hasta)
MAIN = "principal"
SCENARIOS = {
    MAIN: ("settle", False, False, LAG, FIRST, END),
    "antes del 28-2-2018": ("settle", False, False, LAG, FIRST, PRE_END),
    "desde el 28-2-2018": ("settle", False, False, LAG, vix.DESIGN_2018, END),
    "desde el 22-9-2009 (con VIX3M)": ("settle", False, False, LAG, VIX3M_FIRST, END),
    "16:00": ("1600", False, False, LAG, FIRST, END),
    "liquidación sin 5-6-2-2018": ("settle", False, True, LAG, FIRST, END),
    "16:00 sin 5-6-2-2018": ("1600", False, True, LAG, FIRST, END),
    "SVXY real desde 2011-10": ("settle", True, False, LAG, pd.Timestamp("2011-10-05"), END),
    "mismo cierre (no operable)": ("settle", False, False, SAME, FIRST, END),
}
H4_SKIP = ("desde el 22-9-2009 (con VIX3M)",)

INK, MUTED = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, VIOLET, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948", "#9a9994"
F_COLORS = dict(zip(FILTERS, (BLUE, ORANGE, AQUA, VIOLET, RED)))
FC_COLORS = dict(zip(FORECASTS, (BLUE, ORANGE, AQUA)))


# ───────────────────────────── utilidades ─────────────────────────────

def rng_for(tag: str) -> np.random.Generator:
    """Generador determinista propio de cada contraste (como r21.rng_for)."""
    return np.random.default_rng([SEED, zlib.crc32(tag.encode())])


def log_growth(r: pd.Series) -> float:
    return float(np.log1p(r.dropna().clip(lower=CLIP)).mean() * ANN)


def pp(x: float, k: int = 1) -> str:
    """Fracción anual en puntos porcentuales con signo y coma: 0,0123 → «+1,2»."""
    v = 100 * float(x)
    return ("+" if round(v, k) > 0 else "") + num_es(v, k)


def ci_txt(lo: float, hi: float, k: int = 1) -> str:
    """IC en pp con coma decimal, como R22: «[−1,2; 3,4]»."""
    return f"[{num_es(100 * lo, k)}; {num_es(100 * hi, k)}]"


MESES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre")


def month_es(x) -> str:
    """«2020-03» → «marzo de 2020»."""
    per = pd.Period(str(x), "M")
    return f"{MESES[per.month - 1]} de {per.year}"


def fp(p: float) -> str:
    return "< 0,001" if p < 0.001 else num_es(p, 3)


def cfe_calendar(end=END) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Sesiones de CFE con precio y, detrás, días hábiles futuros sin los festivos de
    vix.FUTURE_HOLIDAYS (para contar sesiones hasta un vencimiento posterior al último dato)."""
    cal = pd.DatetimeIndex(sorted(vix.vx_panel(end)["date"].unique()))
    fut = pd.bdate_range(cal[-1] + pd.Timedelta(days=1), "2028-12-31")
    fut = fut[~fut.isin(pd.to_datetime(vix.FUTURE_HOLIDAYS))]
    return cal, cal.append(fut)


def sessions_to(ext: pd.DatetimeIndex, a, b) -> np.ndarray:
    """Sesiones en (a, b] del calendario ext (NaN si falta b)."""
    a = pd.DatetimeIndex(a)
    b = pd.DatetimeIndex(b)
    n = (np.searchsorted(ext.values, b.values, side="right")
         - np.searchsorted(ext.values, a.values, side="right")).astype(float)
    return np.where(b.isna(), np.nan, n)


# ───────────────────────────── H1: ARMA(2,2) del VIX ─────────────────────────────

def arma_resid(y: np.ndarray, p: dict) -> np.ndarray:
    """Innovaciones de y_t − φ1·y_{t−1} − φ2·y_{t−2} = e_t + θ1·e_{t−1} + θ2·e_{t−2} (y ya sin
    media), con condiciones iniciales nulas (CSS)."""
    return lfilter([1.0, -p["phi1"], -p["phi2"]], [1.0, p["th1"], p["th2"]], np.asarray(y, float))


def arma_css(x: np.ndarray, x0: dict | None = None) -> dict:
    """ARMA(2,2) con media por mínimos cuadrados condicionales (Nelder-Mead), estacionario e
    invertible. Devuelve mu, phi1, phi2, th1, th2, la suma de cuadrados y n."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]

    def sse(v):
        mu, a1, a2, b1, b2 = v
        if (np.abs(np.roots([1.0, -a1, -a2])) >= 0.9999).any() or \
                (np.abs(np.roots([1.0, b1, b2])) >= 0.9999).any():
            return 1e18
        e = arma_resid(x - mu, {"phi1": a1, "phi2": a2, "th1": b1, "th2": b2})
        return float(np.sum(e[2:] ** 2))

    start = ([x0[k] for k in ("mu", "phi1", "phi2", "th1", "th2")] if x0 is not None
             else [x.mean(), 1.6, -0.61, -0.7, -0.05])
    res = minimize(sse, start, method="Nelder-Mead",
                   options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-7, "fatol": 1e-9})
    mu, a1, a2, b1, b2 = res.x
    return {"mu": mu, "phi1": a1, "phi2": a2, "th1": b1, "th2": b2, "sse": res.fun, "n": len(x)}


def arma_forecast(x: np.ndarray, p: dict, pos: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Previsión de x_{t+h} con información hasta x_t (t = pos, h ≥ 1 sesiones), recursiva."""
    x = np.asarray(x, float)
    y = x - p["mu"]
    e = arma_resid(y, p)
    pos = np.asarray(pos, int)
    h = np.asarray(h, float)
    out = np.full(len(pos), np.nan)
    ok = (pos >= 1) & np.isfinite(h) & (h >= 1)
    if not ok.any():
        return out
    ps, hs = pos[ok], h[ok].astype(int)
    a1, a2, b1, b2 = p["phi1"], p["phi2"], p["th1"], p["th2"]
    f1 = a1 * y[ps] + a2 * y[ps - 1] + b1 * e[ps] + b2 * e[ps - 1]
    f2 = a1 * f1 + a2 * y[ps] + b2 * e[ps]
    res = np.where(hs == 1, f1, np.where(hs == 2, f2, np.nan))
    prev2, prev1 = f1, f2
    for k in range(3, int(hs.max()) + 1):
        fk = a1 * prev1 + a2 * prev2
        res = np.where(hs == k, fk, res)
        prev2, prev1 = prev1, fk
    out[ok] = p["mu"] + res
    return out


@lru_cache(maxsize=None)
def vix_series(end=END) -> pd.Series:
    v = vix.cboe_index("VIX", end)
    return v[v.index >= VIX_FROM]


@lru_cache(maxsize=None)
def arma_refits(end=END) -> pd.DataFrame:
    """Parámetros del ARMA(2,2) reestimado en la última sesión de cada mes (ventana creciente)."""
    v = vix_series(end)
    me = pd.Series(v.index, index=v.index).groupby(v.index.to_period("M")).last()
    dates = pd.DatetimeIndex(me[me >= FIRST_FIT].values)
    rows, x0 = {}, None
    for d in dates:
        fit = arma_css(v[v.index <= d].to_numpy(), x0)
        rows[d] = fit
        x0 = fit
    return pd.DataFrame(rows).T.astype(float)


def vix_forecast(dates: pd.DatetimeIndex, h: np.ndarray, model: str = "arma", end=END) -> np.ndarray:
    """Ê_t[VIX_{t+h}] en cada fecha t de `dates` (último VIX ≤ t) con el ARMA reestimado
    («arma») o con los coeficientes publicados por Cheng («cheng»)."""
    v = vix_series(end)
    x = v.to_numpy(float)
    dates = pd.DatetimeIndex(dates)
    pos = np.searchsorted(v.index.values, dates.values, side="right") - 1
    h = np.asarray(h, float)
    if model == "cheng":
        return arma_forecast(x, CHENG_PUB, pos, h)
    R = arma_refits(end)
    k = np.searchsorted(R.index.values, dates.values, side="right") - 1
    out = np.full(len(dates), np.nan)
    for j in np.unique(k[k >= 0]):
        m = k == j
        out[m] = arma_forecast(x, R.iloc[j].to_dict(), pos[m], h[m])
    return out


# ───────────────────────────── H1: la estrategia de Cheng ─────────────────────────────

@lru_cache(maxsize=None)
def _price_lookup(end=END) -> pd.Series:
    p = vix.vx_panel(end)
    s = p.set_index(["date", "expiry"])["px"]
    return s[~s.index.duplicated()]


def price_of(dates, expiries, end=END) -> np.ndarray:
    """Precio (settle) del contrato que vence en `expiries` en cada fecha de `dates`."""
    s = _price_lookup(end)
    return s.reindex(pd.MultiIndex.from_arrays([pd.DatetimeIndex(dates), pd.DatetimeIndex(expiries)])).to_numpy()


@lru_cache(maxsize=None)
def rolling_contract(end=END) -> pd.DataFrame:
    """Por sesión de CFE t: vencimiento T del contrato que la estrategia de Cheng lleva desde el
    cierre de t (el que vence el mes siguiente; en la última sesión del mes, el de dentro de dos
    meses; si ese mes no cotiza, el primero posterior), su precio F, las sesiones h hasta T y si
    t es la última sesión del mes."""
    p = vix.vx_panel(end)
    p = p[(p["expiry"] > p["date"]) & (p["px"] > 0)]
    cal, ext = cfe_calendar(end)
    me = pd.Series(ext[1:len(cal) + 1].month != cal.month, index=cal)
    need = (p["date"].dt.year * 12 + p["date"].dt.month + 1
            + me.reindex(p["date"]).to_numpy().astype(int))
    have = p["expiry"].dt.year * 12 + p["expiry"].dt.month
    c = (p[have.to_numpy() >= need.to_numpy()].sort_values(["date", "expiry"])
         .drop_duplicates("date").set_index("date"))
    out = pd.DataFrame({"T": c["expiry"], "F": c["px"]}).reindex(cal)
    out["h"] = sessions_to(ext, out.index, out["T"])
    out["fin_de_mes"] = me
    return out


@lru_cache(maxsize=None)
def rolling_returns(k: int = LAG, end=END) -> pd.Series:
    """Rendimiento diario de la estrategia de Cheng con el contrato elegido k sesiones antes:
    k = 1 es su cronología (cambia de contrato al cierre del último día del mes); k = 2, la
    orden al cierre siguiente (cambia al cierre del primer día del mes siguiente)."""
    rc = rolling_contract(end)
    cal = rc.index
    T = rc["T"].shift(k)
    prev = pd.Series(cal).shift(1)
    p1 = price_of(cal, T, end)
    p0 = price_of(prev, T, end)
    r = pd.Series(p1 / p0 - 1.0, index=cal, name=f"cheng_k{k}")
    return r.iloc[k:]


@lru_cache(maxsize=None)
def premium(model: str = "arma", end=END) -> pd.DataFrame:
    """Prima de Cheng en cada sesión de CFE: Ê (previsión del VIX al vencimiento T), er_raw =
    Ê/F − 1 (rendimiento esperado del largo hasta T), er = 21/h·er_raw y vp = 21/h·(F − Ê),
    la prima escalada a un mes (positiva: el corto espera ganar)."""
    rc = rolling_contract(end).copy()
    ok = rc["T"].notna() & rc["h"].notna()
    E = np.full(len(rc), np.nan)
    E[ok.to_numpy()] = vix_forecast(rc.index[ok], rc.loc[ok, "h"].to_numpy(), model, end)
    rc["E"] = E
    rc["er_raw"] = rc["E"] / rc["F"] - 1
    rc["er"] = ONE_MONTH / rc["h"] * rc["er_raw"]
    rc["vp"] = ONE_MONTH / rc["h"] * (rc["F"] - rc["E"])
    return rc


def month_returns(r: pd.Series, month_end: pd.Series, lag: int, drop=()) -> pd.DataFrame:
    """Rendimiento de cada mes de la estrategia: entre dos últimas sesiones de mes consecutivas
    t1 y t2, compone r en las sesiones (t1 + lag − 1, t2 + lag − 1]; lag = 1 es el mes de Cheng
    (del cierre de t1 al de t2) y lag = 2 la orden al cierre siguiente. `drop`: sesiones que se
    quitan de la composición. Índice: el mes de t2; columna t1: la fecha de la señal."""
    cal = month_end.index
    r = r.reindex(cal)
    pos = np.flatnonzero(month_end.fillna(False).to_numpy(bool))
    drop = pd.DatetimeIndex(drop)
    rows = []
    for i1, i2 in zip(pos[:-1], pos[1:]):
        a, b = i1 + lag - 1, i2 + lag - 1
        if b >= len(cal):
            break
        seg = r.iloc[a + 1:b + 1]
        seg = seg[~seg.index.isin(drop)]
        rows.append({"mes": cal[i2].to_period("M"), "t1": cal[i1], "R": float(np.prod(1 + seg.to_numpy()) - 1),
                     "sesiones": len(seg), "faltan": int(seg.isna().sum())})
    return pd.DataFrame(rows).set_index("mes")


def h1_data(model: str = "arma", lag: int = LAG, drop=(), end=END, scaled: bool = True) -> pd.DataFrame:
    """Meses con el rendimiento R de la estrategia y el rendimiento esperado ER de su t1, escalado
    a un mes (21/h·(Ê/F − 1), como la prima de Cheng; scaled = False: Ê/F − 1 sin escalar)."""
    pr = premium(model, end)
    M = month_returns(rolling_returns(lag, end), pr["fin_de_mes"], lag, drop)
    M["ER"] = pr["er" if scaled else "er_raw"].reindex(M["t1"]).to_numpy()
    M["VP"] = pr["vp"].reindex(M["t1"]).to_numpy()
    return M


def h1_daily(model: str = "arma", end=END, h: int = H_D) -> pd.DataFrame:
    """Regresión diaria: y_t = rendimiento de la estrategia con orden al cierre siguiente en las
    sesiones (t+1, t+1+h]; x_t = er (rendimiento esperado escalado a un mes)."""
    pr = premium(model, end)
    r = rolling_returns(LAG, end).reindex(pr.index)
    lr = np.log1p(r.to_numpy())
    cs = np.r_[0.0, np.nancumsum(lr)]
    bad = np.r_[0, np.cumsum(np.isnan(lr))]
    n = len(r)
    y = np.full(n, np.nan)
    i = np.arange(n)
    ok = i + 1 + h < n
    a, b = i[ok] + 2, i[ok] + 1 + h + 1          # filas (t+1, t+1+h] → [a, b) en la suma
    clean = bad[b] - bad[a] == 0
    y[np.flatnonzero(ok)[clean]] = np.expm1(cs[b[clean]] - cs[a[clean]])
    return pd.DataFrame({"y": y, "er": pr["er"].to_numpy()}, index=pr.index)


def h1_fit(y: pd.Series, x: pd.Series, lags: int | None = NW_CHENG, overlap: int | None = None) -> dict:
    """β de y sobre x con HAC: Newey-West (lags) o, con overlap = h, stats.hac_overlap(h).
    p unilateral de H0: β ≤ 0."""
    X = x.rename("ER").to_frame()
    fit = (stats.ols_hac(y, X, **stats.hac_overlap(overlap)) if overlap
           else stats.ols_hac(y, X, lags))
    j = fit.names.index("ER")
    return {"β": fit.beta[j], "EE": fit.se[j], "t": fit.t[j], "p (una cola)": float(norm.sf(fit.t[j])),
            "R²": fit.r2, "n": fit.n, "estimador": fit.estimator}


def months(M: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return M[(M.index >= pd.Period(a, "M")) & (M.index <= pd.Period(b, "M"))].dropna(subset=["R", "ER"])


H1_MAIN = "Principal: cierre siguiente, ARMA reestimado, NW 3"
H1_CHECK = "Muestra de Cheng, sus coeficientes y su cronología"
H1_RAW = "Rendimiento esperado sin escalar a un mes"
H1_RAW_CHECK = "Muestra de Cheng, sus coeficientes, sin escalar a un mes"


def h1_table(end=END) -> tuple[pd.DataFrame, dict]:
    """Todas las especificaciones de H1 (principal primero) y los datos de la principal."""
    Mn = h1_data("arma", LAG, end=end)
    Ms = h1_data("arma", SAME, end=end)
    Mc = h1_data("cheng", LAG, end=end)
    Mcs = h1_data("cheng", SAME, end=end)
    Md = h1_data("arma", LAG, drop=vix.FEB_2018, end=end)
    D = h1_daily("arma", end)
    a, b = OOS_MONTHS
    nxt = (pd.Period(SPLIT_MONTH, "M") + 1).strftime("%Y-%m")
    rows, kinds = {}, {}

    def add(name, kind, M, lags=NW_CHENG, window=(a, b)):
        m = months(M, *window)
        res = h1_fit(m["R"], m["ER"], lags)
        res.update({"desde": str(m.index[0]), "hasta": str(m.index[-1])})
        rows[name], kinds[name] = res, kind

    add(H1_MAIN, "principal", Mn)
    add("Mismo cierre (cronología de Cheng; no operable)", "sensibilidad", Ms)
    add("Newey-West con 12 retardos", "sensibilidad", Mn, NW_LAB)
    dd = D[(D.index >= pd.Timestamp(a)) & (D.index <= END)].dropna()
    res = h1_fit(dd["y"], dd["er"], overlap=H_D)
    res.update({"desde": str(dd.index[0].date()), "hasta": str(dd.index[-1].date())})
    rows["Diaria, 21 sesiones solapadas, Hansen-Hodrick"], kinds["Diaria, 21 sesiones solapadas, Hansen-Hodrick"] = res, "sensibilidad"
    add("Coeficientes publicados por Cheng", "sensibilidad", Mc)
    add(f"Meses {a} → {SPLIT_MONTH}", "sensibilidad", Mn, window=(a, SPLIT_MONTH))
    add(f"Meses {nxt} → {b}", "sensibilidad", Mn, window=(nxt, b))
    add("Sin el 5 y el 6-2-2018", "sensibilidad", Md)
    add(H1_RAW, "sensibilidad", h1_data("arma", LAG, end=end, scaled=False))
    add(H1_CHECK, "comprobación", Mcs, window=CHENG_MONTHS)
    add(H1_RAW_CHECK, "comprobación", h1_data("cheng", SAME, end=end, scaled=False), window=CHENG_MONTHS)
    add("Muestra de Cheng, ARMA reestimado, cierre siguiente", "comprobación", Mn, window=CHENG_MONTHS)
    add("2004-04 → 2026-08, ARMA reestimado, cierre siguiente", "comprobación", Mn, window=(CHENG_MONTHS[0], b))
    T = pd.DataFrame(rows).T
    T.insert(0, "tipo", pd.Series(kinds))
    for c in ("β", "EE", "t", "p (una cola)", "R²"):
        T[c] = T[c].astype(float)
    T["n"] = T["n"].astype(int)
    return T, {"M": Mn, "Ms": Ms, "D": D}


def h1_influence(M: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Diagnóstico no pre-registrado: β y p del contraste principal quitando cada mes; los n meses
    cuya ausencia más baja β, con su R y su ER."""
    a, b = OOS_MONTHS
    m = months(M, a, b)
    rows = {}
    for mo in m.index:
        f = h1_fit(m.drop(mo)["R"], m.drop(mo)["ER"])
        rows[str(mo)] = {"R del mes": m.loc[mo, "R"], "ER del mes": m.loc[mo, "ER"], "β sin el mes": f["β"],
                         "EE": f["EE"], "p (una cola)": f["p (una cola)"]}
    return pd.DataFrame(rows).T.sort_values("β sin el mes").head(n)


def arma_check(end=END) -> pd.DataFrame:
    """El ARMA(2,2) por CSS con el VIX de Cboe de 1990–2003 frente a los coeficientes de Cheng
    (estimados con datos anteriores a 2004; su fuente, Bloomberg), y en tres fechas de la
    reestimación creciente."""
    v = vix_series(end)
    fit = arma_css(v[v.index <= "2003-12-31"].to_numpy())
    R = arma_refits(end)
    cols = ["mu", "phi1", "phi2", "th1", "th2"]
    rows = {"Cheng (2019), publicado": {**CHENG_PUB, "n": np.nan},
            "CSS, VIX de Cboe 1990–2003": {**{k: fit[k] for k in cols}, "n": fit["n"]}}
    for d in ("2015-12-31", "2018-02-28", R.index[-1]):
        d = R.index[R.index <= pd.Timestamp(d)][-1]
        rows[f"reestimado el {dmy(d)}"] = {**{k: R.loc[d, k] for k in cols}, "n": R.loc[d, "n"]}
    out = pd.DataFrame(rows).T
    out["φ1 + φ2"] = out["phi1"] + out["phi2"]
    return out.rename(columns={"mu": "μ", "phi1": "φ1", "phi2": "φ2", "th1": "θ1", "th2": "θ2", "n": "sesiones"})


def premium_table(end=END) -> pd.DataFrame:
    """La prima de Cheng por periodo: media (puntos, escalada a un mes), días con prima > 0 y
    rendimiento esperado; y la correlación entre la previsión reestimada y la publicada."""
    A = premium("arma", end)
    C = premium("cheng", end)
    per = {"2004-03 → 2015-12 (muestra de Cheng)": ("2004-03-26", "2015-12-31"),
           "2016-01 → 2018-02": ("2016-01-01", "2018-02-28"),
           "2018-03 → 2026-09": ("2018-03-01", END)}
    rows = {}
    for k, (a, b) in per.items():
        x, c = A.loc[a:b].dropna(subset=["vp"]), C.loc[a:b].dropna(subset=["vp"])
        rows[k] = {"sesiones": len(x), "prima media (puntos)": x["vp"].mean(),
                   "prima mediana (puntos)": x["vp"].median(), "días con prima > 0": (x["vp"] > 0).mean(),
                   "ER medio del mes": x["er"].mean(),
                   "días con prima > 0 (publicado)": (c["vp"] > 0).mean(),
                   "corr. Ê reestimado–publicado": x["E"].corr(c["E"].reindex(x.index))}
    return pd.DataFrame(rows).T


# ───────────────────────────── H2: filtros ─────────────────────────────

def sc_roll(end=END) -> pd.Series:
    """Roll diario de Simon y Campasano: (F − VIX)/n con F el primer contrato con más de
    SC_MIN sesiones hasta el vencimiento y n esas sesiones (VIX de Cboe)."""
    p = vix.vx_panel(end)
    p = p[(p["expiry"] > p["date"]) & (p["px"] > 0)].copy()
    _, ext = cfe_calendar(end)
    p["n"] = sessions_to(ext, p["date"], p["expiry"])
    q = (p[p["n"] > SC_MIN].sort_values(["date", "expiry"]).drop_duplicates("date").set_index("date"))
    ts = vix.term_structure(end)
    return ((q["px"] - ts["vix"].reindex(q.index)) / q["n"]).rename("roll")


@lru_cache(maxsize=None)
def signals(end=END) -> pd.DataFrame:
    """Estado de cada filtro al cierre de cada sesión de CFE: 1 (−0,5x), 0 (letras), NaN sin dato."""
    ts = vix.term_structure(end)

    def state(cond, *need):
        ok = np.logical_and.reduce([s.notna() for s in need])
        return cond.astype(float).where(ok)

    dr = sc_roll(end).reindex(ts.index)
    vp = premium("arma", end)["vp"].reindex(ts.index)
    return pd.DataFrame({F_F1: state(ts["f1"] > ts["vix"], ts["f1"], ts["vix"]),
                         F_TS: state(ts["vix"] < ts["vix3m"], ts["vix"], ts["vix3m"]),
                         F_F2: state(ts["f2"] > ts["f1"], ts["f2"], ts["f1"]),
                         F_SC: state(dr > SC_THR, dr),
                         F_CH: state(vp > 0, vp)}, index=ts.index)


def to_index(s: pd.Series, index: pd.DatetimeIndex, limit: int = 5) -> pd.Series:
    """Último valor conocido en cada fecha de `index` (fechas de s ≤ fecha; sin mirar adelante)."""
    return s.reindex(s.index.union(index)).ffill(limit=limit).reindex(index)


def run_weights(x: pd.Series, rf: pd.Series, target: pd.Series, lag: int = LAG,
                cost_bps: float = COST_BPS, margin_extra: float = 0.0) -> tuple[pd.Series, pd.Series]:
    """Cartera con `target` (peso decidido al cierre de t) en x y el resto en letras: cobra desde
    el cierre de t + lag − 1 (como r13.run_weights), paga cost_bps por unidad negociada ese día
    y, si el peso pasa de 1, la letra + margin_extra por lo prestado. El retardo se cuenta en
    filas de `target`, cuyo índice debe contener el de x (y puede empezar antes, para que el
    primer día de x ya tenga peso). Devuelve (r, peso cobrado)."""
    w = target.shift(lag).reindex(x.index)
    turn = w.diff().abs().fillna(0.0)
    r = (w * x + (1 - w) * rf - (w - 1).clip(lower=0) * margin_extra / ANN - turn * cost_bps / 1e4)
    return r.where(w.notna()), w


def filter_target(state: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Peso objetivo en el −0,5x sobre las sesiones de CFE anteriores a index[0] seguidas de
    `index`: el estado del filtro (1/0) y, sin dato, 1 (la referencia)."""
    full = state.index[state.index < index[0]].append(pd.DatetimeIndex(index))
    return to_index(state, full).fillna(1.0)


@lru_cache(maxsize=None)
def product(conv: str = "settle", real: bool = False, end=END) -> pd.Series:
    """El −0,5x: vix.minus_half_spliced(conv) o, con real = True, el SVXY real (½ SVXY −1x +
    ½ letras hasta el 27-2-2018, reajustado a diario)."""
    if not real:
        return vix.minus_half_spliced(conv, end).dropna()
    s = vix.etp_returns(["SVXY"], end)["SVXY"].dropna()
    rf = vix.cash(s.index)
    early = s.index < vix.DESIGN_2018
    out = s.copy()
    out[early] = 0.5 * s[early] + 0.5 * rf[early]
    return out.rename("SVXY real")


def h2_returns(x: pd.Series, lag: int = LAG, end=END) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rendimientos y pesos de siempre −0,5x y de los 5 filtros sobre el calendario de x."""
    rf = vix.cash(x.index)
    S = signals(end)
    R, W = {BENCH: x}, {BENCH: pd.Series(1.0, index=x.index)}
    for f in FILTERS:
        R[f], W[f] = run_weights(x, rf, filter_target(S[f], x.index), lag)
    return pd.DataFrame(R), pd.DataFrame(W)


# ───────────────────────────── H4: gestión por volatilidad ─────────────────────────────

def presample(end=END) -> pd.Series:
    """−0,5 × el índice corto antes del 21-12-2005: solo para calentar las previsiones."""
    F = vix.futures_indices(end)
    s = F["st_ret"].dropna()
    return -0.5 * s[s.index < FIRST]


def gjr_forecast(x: pd.Series, first: pd.Timestamp) -> pd.Series:
    """h_{t+1} del GJR-GARCH(1,1) conocida al cierre de t, reestimado en `first` y cada 1 de
    enero posterior con los rendimientos anteriores (stats.garch_fit / garch_filter)."""
    xv = x.to_numpy(float)
    idx = x.index
    starts = [pd.Timestamp(first)] + [pd.Timestamp(f"{y}-01-01") for y in range(first.year + 1, idx[-1].year + 1)]
    out = pd.Series(np.nan, index=idx)
    for k, a in enumerate(starts):
        b = starts[k + 1] if k + 1 < len(starts) else idx[-1] + pd.Timedelta(days=1)
        p = stats.garch_fit(xv[idx < a], "gjr")
        h = stats.garch_filter(xv, p)
        m = (idx >= a) & (idx < b)
        out[m] = h[m]
    return out


def variance_forecasts(x_full: pd.Series) -> pd.DataFrame:
    """σ̂²_t de cada previsión de FORECASTS, conocida al cierre de t. El GJR se estima por primera
    vez con las GJR_MIN primeras sesiones de x_full y después cada 1 de enero."""
    x2 = x_full ** 2
    rv = x2.rolling(RV_N).mean()
    ew = x2.ewm(alpha=1 - EWMA_L, adjust=False).mean().where(np.arange(len(x2)) >= RV_N - 1)
    return pd.DataFrame({"RV 21": rv, "EWMA 0,94": ew, "GJR": gjr_forecast(x_full, x_full.index[GJR_MIN])})


def solve_c(inv_var: np.ndarray, w_max: float = W_MAX, target: float = 1.0) -> float:
    """c tal que la media de mín(c/σ̂², w_max) sea `target` (inv_var = 1/σ̂² de las sesiones)."""
    z = np.asarray(inv_var, float)
    z = z[np.isfinite(z) & (z > 0)]
    f = lambda c: np.minimum(c * z, w_max).mean() - target  # noqa: E731
    return float(brentq(f, 0.0, w_max / z.min(), xtol=1e-14, rtol=1e-12))


def h4_returns(x: pd.Series, x_full: pd.Series, a, b, lag: int = LAG) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Siempre −0,5x y las 6 reglas de MANAGED en la ventana [a, b] (c ex post resuelto en ella).
    Las previsiones y los pesos objetivo se calculan sobre x_full (con el calentamiento), así que
    el primer día de x ya tiene peso."""
    Vf = variance_forecasts(x_full)
    rf = vix.cash(x.index)
    win = (x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))
    R, W, info = {BENCH: x[win]}, {BENCH: pd.Series(1.0, index=x.index[win])}, {}
    for f in FORECASTS:
        inv = 1.0 / Vf[f]
        c = solve_c(inv.shift(lag).reindex(x.index)[win].to_numpy())
        info[f] = c
        rt_c = 1.0 / inv.expanding(RT_MIN).mean()
        for n, tgt in (("ex post", np.minimum(c * inv, W_MAX)), ("tiempo real", np.minimum(rt_c * inv, W_MAX))):
            r, w = run_weights(x, rf, tgt, lag, margin_extra=MARGIN_EXTRA)
            R[f"{f} · {n}"], W[f"{f} · {n}"] = r[win], w[win]
    return pd.DataFrame(R), pd.DataFrame(W), info


# ───────────────────────────── contrastes por familia ─────────────────────────────

def boot_draws(X: np.ndarray, rng, n_boot: int = B, block: float = BLOCK) -> np.ndarray:
    """B × K medias de un bootstrap estacionario con los mismos índices para todas las columnas."""
    X = np.asarray(X, float)
    T = len(X)
    out = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        out[b] = X[stats.stationary_indices(T, block, rng)].mean(axis=0)
    return out


def top_removed(d: pd.Series, k: int = TOP_DAYS) -> tuple[float, list]:
    """Δg anual sin los k días de mayor ventaja diaria (d = log regla − log referencia)."""
    top = d.nlargest(k)
    return float(d.drop(top.index).mean() * ANN), list(top.index)


def family_test(R: pd.DataFrame, W: pd.DataFrame, tag: str, rf: pd.Series | None = None) -> dict:
    """Δg de cada regla frente a BENCH (bootstrap emparejado: IC90, t, p unilateral), Romano–Wolf,
    SPA de la familia, Δg sin los 3 mejores días y métricas descriptivas. R ya en su ventana;
    un día de −100% arruina (vix.absorb desde el inicio de la ventana)."""
    R = R.dropna()
    R = R.apply(vix.absorb)
    W = W.reindex(R.index)
    L = np.log1p(R.clip(lower=CLIP))
    cands = [c for c in R.columns if c != BENCH]
    rng = rng_for(tag)
    draws = boot_draws(L.to_numpy(), rng)
    cols = list(L.columns)
    j = cols.index(BENCH)
    ci = [cols.index(c) for c in cands]
    m = L.to_numpy().mean(axis=0)
    dm = m[ci] - m[j]
    dd = draws[:, ci] - draws[:, [j]]
    se = dd.std(axis=0, ddof=1)
    safe = np.where(se > 0, se, np.inf)
    t = dm / safe
    lo, hi = np.percentile(dd, [5, 95], axis=0)
    p_rw = romano_wolf(t, (dd - dm) / safe)
    D = L[cands].sub(L[BENCH], axis=0)
    spa = stats.spa_test(D.to_numpy(), BLOCK, B, rng)
    rf = vix.cash(R.index) if rf is None else rf.reindex(R.index)
    yrs = len(R) / ANN
    rows = {}
    for k in cols:
        r = R[k]
        w = W[k]
        rows[k] = {"CAGR": metrics.cagr(r), "vol": metrics.vol(r), "sharpe": metrics.sharpe(r, rf),
                   "maxDD": metrics.max_dd(r), "peor día": r.min(), "g": L[k].mean() * ANN,
                   "exposición media": w.mean(), "tiempo dentro": (w > 0).mean(),
                   "rotación/año": w.diff().abs().sum() / yrs}
    tab = pd.DataFrame(rows).T
    for i, c in enumerate(cands):
        tab.loc[c, "Δg"] = dm[i] * ANN
        tab.loc[c, "IC90_bajo"] = lo[i] * ANN
        tab.loc[c, "IC90_alto"] = hi[i] * ANN
        tab.loc[c, "t"] = t[i]
        tab.loc[c, "p"] = norm.sf(t[i])
        tab.loc[c, "p_RW"] = p_rw[i]
        g3, days = top_removed(D[c])
        tab.loc[c, f"Δg sin {TOP_DAYS} mejores días"] = g3
        tab.loc[c, "mejores días"] = ", ".join(dmy(x) for x in sorted(days))
    return {"tabla": tab, "spa": spa, "desde": R.index[0], "hasta": R.index[-1], "sesiones": len(R),
            "R": R, "W": W, "D": D}


def scenario_x(conv: str, real: bool, drop: bool, end=END) -> tuple[pd.Series, pd.Series]:
    """(x del escenario, x con el calentamiento de las previsiones delante)."""
    x = product(conv, real, end)
    if real:
        syn = vix.minus_half_spliced("settle", end).dropna()
        pre = pd.concat([presample(end), syn[syn.index < x.index[0]]])
    else:
        pre = presample(end)
    full = pd.concat([pre[pre.index < x.index[0]], x])
    if drop:
        x, full = vix.drop_days(x), vix.drop_days(full)
    return x, full


def run_scenarios(end=END) -> dict:
    """H2 y H4 en cada escenario de SCENARIOS."""
    out = {"h2": {}, "h4": {}, "h4rt": {}, "c": {}}
    for name, (conv, real, drop, lag, a, b) in SCENARIOS.items():
        x, full = scenario_x(conv, real, drop, end)
        R, W = h2_returns(x, lag, end)
        win = (R.index >= a) & (R.index <= b)
        out["h2"][name] = family_test(R[win], W[win], f"h2|{name}")
        if name in H4_SKIP:
            continue
        R4, W4, info = h4_returns(x, full, a, b, lag)
        out["h4"][name] = family_test(R4[[BENCH, *EXPOST]], W4[[BENCH, *EXPOST]], f"h4|{name}")
        out["h4rt"][name] = family_test(R4[[BENCH, *REALTIME]], W4[[BENCH, *REALTIME]], f"h4rt|{name}")
        out["c"][name] = info
    return out


def verdict_h2(res: dict) -> dict:
    """Veredicto de H2 en el escenario principal con la regla de fragilidad del pre-registro."""
    t, spa = res["tabla"], res["spa"]
    rules = t.drop(index=BENCH)
    rejected = spa["p_SPA_c"] < P_SPA
    winners = list(rules.index[rules["p_RW"] < P_SPA])
    if rejected and not winners:
        winners = [rules["Δg"].idxmax()]
    frag = {w: bool(rules.loc[w, f"Δg sin {TOP_DAYS} mejores días"] <= 0) for w in winners} if rejected else {}
    return {"p_SPA_c": spa["p_SPA_c"], "p_SPA_l": spa["p_SPA_l"], "rechaza": rejected,
            "ganadores": winners, "frágil": frag, "mejor": rules["Δg"].idxmax(),
            "Δg_mejor": rules["Δg"].max()}


def scenario_grid(S: dict, rules) -> pd.DataFrame:
    """Δg (pp) de cada regla por escenario, con * si p_RW < 0,10, y el SPA de la familia."""
    rows = {}
    for name, res in S.items():
        t = res["tabla"]
        row = {"desde": res["desde"].date(), "hasta": res["hasta"].date(), "sesiones": res["sesiones"]}
        for k in rules:
            mark = "*" if t.loc[k, "p_RW"] < P_SPA else ""
            row[k] = f"{num_es(100 * t.loc[k, 'Δg'], 1)}{mark}"
        row["p_SPA_c"] = res["spa"]["p_SPA_c"]
        row["p_SPA_l"] = res["spa"]["p_SPA_l"]
        row["g referencia (%)"] = 100 * t.loc[BENCH, "g"]
        rows[name] = row
    return pd.DataFrame(rows).T


def worst_days(res: dict, rules, n: int = 10) -> pd.DataFrame:
    """Los n peores días de siempre −0,5x en la ventana y el peso que cobraba cada regla."""
    R, W = res["R"], res["W"]
    idx = R[BENCH].nsmallest(n).index
    out = pd.DataFrame({"fecha": [dmy(d) for d in idx], "siempre −0,5x": R.loc[idx, BENCH].to_numpy()})
    for k in rules:
        out[k] = W.loc[idx, k].to_numpy()
    return out.set_index("fecha")


def reference_rows(a, b, end=END) -> pd.DataFrame:
    """SPY y la LRS 3x (original) del artículo 2 (censo de R19) en la misma ventana."""
    out = {"SPY (1x, Yahoo)": vix.etp_returns(["SPY"], end)["SPY"]}
    if CENSUS.exists():
        head = pd.read_csv(CENSUS, nrows=0).columns
        if LRS in head:
            c = pd.read_csv(CENSUS, index_col=0, parse_dates=True, usecols=[head[0], LRS])
            out[f"{LRS}, censo de R19"] = c[LRS]
    rows = {}
    for k, r in out.items():
        r = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))].dropna()
        rf = vix.cash(r.index)
        rows[k] = {"CAGR": metrics.cagr(r), "vol": metrics.vol(r), "sharpe": metrics.sharpe(r, rf),
                   "maxDD": metrics.max_dd(r), "peor día": r.min(), "g": log_growth(r)}
    return pd.DataFrame(rows).T


def filter_stats(end=END) -> pd.DataFrame:
    """Tiempo dentro de cada filtro por periodo (estado al cierre, calendario de CFE)."""
    S = signals(end)
    per = {"2005-12 → 2018-02": (FIRST, PRE_END), "2018-02 → 2026-09": (vix.DESIGN_2018, END)}
    rows = {}
    for f in FILTERS:
        s = S[f]
        row = {"primer dato": s.first_valid_index().date()}
        for k, (a, b) in per.items():
            x = s.loc[a:b].dropna()
            row[f"dentro {k}"] = x.mean()
            row[f"cambios/año {k}"] = (x.diff().abs() > 0).sum() / (len(x) / ANN)
        rows[f] = row
    return pd.DataFrame(rows).T


# ───────────────────────────── cálculo ─────────────────────────────

def compute(end=END) -> dict:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    T1, H1 = h1_table(end)
    S = run_scenarios(end)
    return {"h1": T1, "h1_data": H1, "infl": h1_influence(H1["M"]), "arma": arma_check(end),
            "prem": premium_table(end),
            "fstats": filter_stats(end), "S": S, "v2": verdict_h2(S["h2"][MAIN]),
            "ref": reference_rows(FIRST, END, end),
            "ref_post": reference_rows(vix.DESIGN_2018, END, end)}


# ───────────────────────────── figuras ─────────────────────────────

def _style(ax, grid: str = "x"):
    ax.grid(axis=grid, alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_h1(C: dict, rep: Report):
    T = C["h1"]
    order = [k for k in T.index if T.loc[k, "tipo"] != "comprobación"] + \
            [k for k in T.index if T.loc[k, "tipo"] == "comprobación"]
    fig, ax = plt.subplots(figsize=(FIG_W, 6.4))
    ys = np.arange(len(order))[::-1].astype(float)
    for y, k in zip(ys, order):
        b, se, kind = T.loc[k, "β"], T.loc[k, "EE"], T.loc[k, "tipo"]
        col = BLUE if kind == "principal" else (GREY if kind == "comprobación" else INK)
        ax.plot([b - 1.645 * se, b + 1.645 * se], [y, y], color=col, lw=2.2, solid_capstyle="round")
        ax.plot(b, y, "o", color=col, ms=8, mec="white", mew=1.2, zorder=3)
    b0, s0 = CHENG_BETA
    y0 = ys[-1] - 1.2
    ax.plot([b0 - 1.645 * s0, b0 + 1.645 * s0], [y0, y0], color=ORANGE, lw=2.2)
    ax.plot(b0, y0, "D", color=ORANGE, ms=8, mec="white", mew=1.2, zorder=3)
    labels = list(order) + ["Cheng (2019), publicado, 2004–2015"]
    ax.set_yticks(list(ys) + [y0], labels, fontsize=10.5)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(1, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("β (IC del 90%)")
    fig.suptitle("H1: β del rendimiento del mes sobre el esperado")
    _style(ax, "x")
    comma_fig(fig)
    rep.fig(fig, "h1_coeficientes",
            "Figura 1: β de la regresión del rendimiento del futuro sobre el rendimiento esperado de Cheng con su "
            "intervalo del 90% (el extremo izquierdo es la cota unilateral del 95%). Azul: el contraste "
            "pre-registrado, fuera de la muestra de Cheng; negro: sensibilidades; gris: comprobaciones con su muestra; "
            "naranja: lo que publica Cheng. La línea de puntos marca β = 1, lo que daría una prima medida sin error")


def fig_h1_scatter(C: dict, rep: Report):
    M = C["h1_data"]["M"]
    a, b = OOS_MONTHS
    m = months(M, a, b)
    ins = months(M, *CHENG_MONTHS)
    fig, ax = plt.subplots(figsize=(FIG_W, 5.6))
    ax.scatter(100 * ins["ER"], 100 * ins["R"], s=22, color=GREY, alpha=0.6, lw=0,
               label=f"meses {CHENG_MONTHS[0]} → {CHENG_MONTHS[1]} (muestra de Cheng)")
    pre = m[m.index <= pd.Period(SPLIT_MONTH, "M")]
    post = m[m.index > pd.Period(SPLIT_MONTH, "M")]
    ax.scatter(100 * pre["ER"], 100 * pre["R"], s=34, color=ORANGE, lw=0.6, ec="white",
               label=f"{a} → {SPLIT_MONTH}")
    ax.scatter(100 * post["ER"], 100 * post["R"], s=34, color=BLUE, lw=0.6, ec="white",
               label=f"{(pd.Period(SPLIT_MONTH, 'M') + 1).strftime('%Y-%m')} → {b}")
    f = h1_fit(m["R"], m["ER"])
    alpha = float(m["R"].mean() - f["β"] * m["ER"].mean())
    xs = np.linspace(m["ER"].min(), m["ER"].max(), 50)
    ax.plot(100 * xs, 100 * (alpha + f["β"] * xs), color=INK, lw=1.8,
            label=f"recta {a} → {b} (β = {num_es(f['β'], 2)})")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Rendimiento esperado del mes al cierre de t1 (%)")
    ax.set_ylabel("Rendimiento del largo en el mes (%)")
    ax.set_title("Rendimiento esperado y realizado del futuro a un mes")
    _style(ax, "both")
    h, lab = ax.get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=2, fontsize=10.5)
    comma_fig(fig)
    rep.fig(fig, "h1_dispersion",
            "Figura 2: cada punto es un mes: rendimiento esperado por la prima de Cheng al cierre del mes anterior "
            "y rendimiento del contrato largo con la orden al cierre siguiente. Un corto gana lo contrario")


def _dg_dots(ax, S: dict, names: list, rules, colors: dict, markers: dict):
    ys = np.arange(len(rules))[::-1].astype(float)
    off = np.linspace(0.22, -0.22, len(names))
    for o, name in zip(off, names):
        t = S[name]["tabla"]
        for y, k in zip(ys, rules):
            dg, lo, hi = (100 * t.loc[k, c] for c in ("Δg", "IC90_bajo", "IC90_alto"))
            col = colors[k]
            ax.plot([lo, hi], [y + o, y + o], color=col, lw=1.8, alpha=0.9)
            ax.plot(dg, y + o, markers[name], color=col, ms=7, mec="white", mew=1.0, zorder=3,
                    mfc=col if markers[name] != "o" or name == names[0] else "white")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ys, list(rules))
    return ys


def fig_dg(S: dict, rep: Report, rules, colors: dict, name: str, title: str, caption: str):
    names = [MAIN, "antes del 28-2-2018", "desde el 28-2-2018"]
    markers = dict(zip(names, ("o", "s", "^")))
    fig, ax = plt.subplots(figsize=(FIG_W, 1.4 + 0.75 * len(rules)))
    _dg_dots(ax, S, names, rules, colors, markers)
    ax.set_xlabel("Δg frente a siempre −0,5x (pp/año, IC del 90%)")
    fig.suptitle(title)
    _style(ax, "x")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker=markers[n], color=INK, ls="", ms=7) for n in names]
    labs = ["21-12-2005 → 25-9-2026", "hasta el 27-2-2018 (sintético)", "desde el 28-2-2018 (SVXY real)"]
    legend_band(fig, hs, labs, where="bottom", ncol=3, fontsize=10.5)
    comma_fig(fig)
    rep.fig(fig, name, caption)


def fig_wealth(S: dict, rep: Report):
    res = S["h2"][MAIN]
    R = res["R"]
    fig, ax = plt.subplots(figsize=(FIG_W, 5.8))
    w = (1 + R[BENCH]).cumprod()
    ax.semilogy(w.index, w.values, color=GREY, lw=2.4, label=f"{BENCH}: {pct_es(metrics.cagr(R[BENCH]))}/año")
    for f in FILTERS:
        w = (1 + R[f]).cumprod()
        ax.semilogy(w.index, w.values, color=F_COLORS[f], lw=1.4, label=f"{f}: {pct_es(metrics.cagr(R[f]))}/año")
    ax.axvline(vix.DESIGN_2018, color=MUTED, lw=0.8, ls=":")
    spa = res["spa"]["p_SPA_c"]
    verdict = ("ningún filtro supera al −0,5x siempre tras corregir por la búsqueda" if spa >= P_SPA else
               "el SPA rechaza «ningún filtro supera al −0,5x siempre»")
    ax.text(0.99, 0.03, f"CAGR sin corregir por la búsqueda de 5 filtros;\n{verdict}\n(SPA p = {fp(spa)})",
            transform=ax.transAxes, fontsize=11, color=MUTED, va="bottom", ha="right", multialignment="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num_es(v, 2 if v < 1 else 0)))
    ax.set(ylabel="Valor de 1 $ (log)", title="H2: −0,5x siempre o solo cuando el filtro lo dice")
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style(ax, "y")
    h, lab = ax.get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=2, fontsize=10.5)
    comma_fig(fig)
    cap = ("Figura 4: valor de 1 dólar en el −0,5x siempre y con cada filtro (fuera: letras), orden al cierre "
           "siguiente, 5 pb por unidad negociada, convención de liquidación; sintético hasta el 27-2-2018 (línea de "
           "puntos) y SVXY real después. Sin impuestos. Los CAGR de la leyenda son los de la búsqueda, sin corregir")
    if spa >= P_SPA:
        cap += (f": tras la corrección (SPA de la familia de 5, p = {fp(spa)}) ningún filtro supera a estar siempre "
                "−0,5x")
    rep.fig(fig, "h2_riqueza", cap)


EPISODES = (("5-2-2018", "2017-12-01", "2018-03-30", ("2018-02-05",)),
            ("marzo de 2020", "2020-01-15", "2020-05-29", ("2020-02-24", "2020-03-16")),
            ("5-8-2024", "2024-06-17", "2024-09-30", ("2024-08-05",)))


def fig_h4_weights(S: dict, rep: Report):
    res = S["h4"][MAIN]
    W, R = res["W"], res["R"]
    fig, axes = plt.subplots(len(EPISODES), 1, figsize=(FIG_W, 8.2))
    for ax, (name, a, b, marks) in zip(axes, EPISODES):
        for f in FORECASTS:
            k = f"{f} · ex post"
            w = W[k].loc[a:b]
            ax.step(w.index, w.values, where="post", color=FC_COLORS[f], lw=1.6, label=k)
        for i, d in enumerate(marks):
            d = pd.Timestamp(d)
            ax.axvline(d, color=GREY, lw=1)
            ax.annotate(f"{dmy(d)}: −0,5x {pct_es(R.loc[d, BENCH])}", (d, W_MAX), xytext=(4, -2 - 17 * i),
                        textcoords="offset points", fontsize=10, color=MUTED, va="top")
        ax.axhline(1, color=MUTED, lw=1, ls=":")
        ax.set_ylim(0, W_MAX + 0.15)
        ax.set_ylabel("peso")
        ax.set_title(name, fontsize=12.5)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: (lambda d: f"{d.day}-{d.month}")(mdates.num2date(v))))
        _style(ax, "y")
    fig.suptitle("H4: peso cobrado en el −0,5x alrededor de tres saltos del VIX")
    h, lab = axes[0].get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=3, fontsize=10.5)
    comma_fig(fig)
    rep.fig(fig, "h4_pesos",
            "Figura 6: peso cobrado cada sesión en el −0,5x por las tres reglas de volatilidad con la normalización "
            "ex post (media 1 en la ventana; tope 2, es decir −1x), alrededor del 5-2-2018, de marzo de 2020 y del "
            "5-8-2024. La línea de puntos marca el peso de estar siempre −0,5x; las verticales, los días señalados, "
            "con el rendimiento de ese día de estar siempre −0,5x")


# ───────────────────────────── informe ─────────────────────────────

def fmt_family(res: dict, rules, extra: pd.DataFrame | None = None, inside: bool = True) -> pd.DataFrame:
    t = res["tabla"].copy()
    out = pd.DataFrame(index=t.index)
    for c in ("CAGR", "vol", "sharpe", "maxDD", "peor día", "exposición media"):
        out[c] = t[c]
    if inside:
        out["tiempo dentro"] = t["tiempo dentro"]
    out["rotación/año"] = t["rotación/año"]
    out["Δg (pp)"] = 100 * t["Δg"]
    out["IC90 (pp)"] = [ci_txt(lo, hi) if pd.notna(lo) else "" for lo, hi in zip(t["IC90_bajo"], t["IC90_alto"])]
    out["p"] = t["p"]
    out["p_RW"] = t["p_RW"]
    out[f"Δg sin {TOP_DAYS} mejores días (pp)"] = 100 * t[f"Δg sin {TOP_DAYS} mejores días"]
    if extra is not None:
        out = pd.concat([out, extra[[c for c in ("CAGR", "vol", "sharpe", "maxDD", "peor día") if c in extra]]])
        out["IC90 (pp)"] = out["IC90 (pp)"].fillna("")
    return out.loc[[BENCH] + list(rules) + ([] if extra is None else list(extra.index))]


def y_join(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


def _range_num(vals, k: int = 2) -> str:
    vals = list(vals)
    return f"entre {num_es(min(vals), k)} y {num_es(max(vals), k)}"


def _range_pp(vals) -> str:
    vals = [100 * v for v in vals]
    return f"entre {num_es(min(vals), 1)} y {num_es(max(vals), 1)}"


def write_report(C: dict, rep: Report):
    tabs = Tables(rep)
    num = {}

    def tab(key: str, title: str, df: pd.DataFrame, **kw):
        tabs(title, df, **kw)
        num[key] = tabs.n

    T1, S, v2 = C["h1"], C["S"], C["v2"]
    main1 = T1.loc[H1_MAIN]
    chk = T1.loc[H1_CHECK]
    raw, raw_chk = T1.loc[H1_RAW], T1.loc[H1_RAW_CHECK]
    full = T1.loc["2004-04 → 2026-08, ARMA reestimado, cierre siguiente"]
    sens = T1[T1["tipo"] == "sensibilidad"]
    h1_ok = bool(main1["β"] > 0 and main1["p (una cola)"] < 0.05)
    h2, h4 = S["h2"][MAIN], S["h4"][MAIN]
    t2, t4 = h2["tabla"], h4["tabla"]
    spa4 = h4["spa"]["p_SPA_c"]
    h4_holds = spa4 > P_SPA
    r4 = t4.drop(index=BENCH)
    best4 = r4["Δg"].idxmax()
    best2 = v2["mejor"]
    ref = C["ref"]
    lrs_row = [k for k in ref.index if k.startswith(LRS)]
    spy = ref.loc["SPY (1x, Yahoo)"]
    lrs = ref.loc[lrs_row[0]] if lrs_row else None
    a, b = OOS_MONTHS
    nxt = (pd.Period(SPLIT_MONTH, "M") + 1).strftime("%Y-%m")
    pre1, post1 = T1.loc[f"Meses {a} → {SPLIT_MONTH}"], T1.loc[f"Meses {nxt} → {b}"]
    d1 = T1.loc["Diaria, 21 sesiones solapadas, Hansen-Hodrick"]
    best2_all = [S["h2"][k]["tabla"].loc[best2, "Δg"] for k in S["h2"]]
    best2_rw = [S["h2"][k]["tabla"].loc[best2, "p_RW"] for k in S["h2"]]
    spa2_all = {k: S["h2"][k]["spa"]["p_SPA_c"] for k in S["h2"]}
    operable = [k for k in S["h2"] if not k.startswith("mismo cierre")]
    spa2_op = {k: spa2_all[k] for k in operable}
    worst = worst_days(h2, FILTERS)
    infl = C["infl"]
    top_m = month_es(infl.index[0])
    top_i = infl.iloc[0]
    feb = dmy(vix.FEB_2018[0])
    h4rt = S["h4rt"][MAIN]
    t4rt = h4rt["tabla"]
    spa4rt = h4rt["spa"]["p_SPA_c"]
    W4 = pd.concat([h4["W"], h4rt["W"].drop(columns=[BENCH])], axis=1)
    R4all = pd.concat([h4["R"], h4rt["R"].drop(columns=[BENCH])], axis=1)
    rt_exp = [W4[k].mean() for k in REALTIME]
    h4_neg_all = all((S["h4"][k]["tabla"].drop(index=BENCH)["Δg"] < 0).all() for k in S["h4"])
    h4_pos = [(k, r) for k in S["h4"] for r, v in S["h4"][k]["tabla"].drop(index=BENCH)["Δg"].items() if v > 0]
    rt_pos = [(k, r) for k in S["h4rt"] for r, v in S["h4rt"][k]["tabla"].drop(index=BENCH)["Δg"].items() if v > 0]
    spa4rt_min = min(S["h4rt"][k]["spa"]["p_SPA_c"] for k in S["h4rt"] if not k.startswith("mismo cierre"))

    # ───── cabecera ─────
    rep.p("Pregunta. El corto de futuros del VIX gana de media porque el futuro cotiza por encima del VIX con el "
          "que acaba liquidando: esa es la prima. ¿Se puede saber cuándo es mayor y cronometrar el corto? Se "
          "contrastan tres hipótesis del pre-registro: H1, que la prima ex ante de Cheng (2019) predice el "
          "rendimiento del futuro fuera de su muestra; H2, que ningún filtro de estructura de plazos mejora a estar "
          "siempre −0,5x tras corregir por la búsqueda; y H4, que gestionar el −0,5x por volatilidad a igual "
          "exposición media tampoco lo mejora.")
    rep.p("Pre-registro: letf/preregistro/articulo4_volatilidad.md (commit 7cded63). Esta línea contrasta H1, H2 y "
          "H4; H3 y H5 van en otra. Datos: letf_lab.data.vix (R26): futuros VX de CFE, el VIX y el VIX3M de los CSV "
          "de Cboe (nunca el ^VIX de Yahoo) y el −0,5x del pre-registro, sintético hasta el 27-2-2018 y SVXY real "
          "desde el 28-2-2018.")
    rep.p("Protocolo: señal al cierre de t y orden al cierre de t+1. El mismo cierre se muestra solo como referencia "
          "no operable: la liquidación del VX y el cierre del VIX son de las 16:15 ET hasta octubre de 2020, cuando "
          "los ETPs ya han cerrado. 5 pb por unidad negociada; la caja cobra la letra. Δg: crecimiento logarítmico "
          "anual frente a estar siempre −0,5x, con bootstrap estacionario emparejado (bloque medio 63, 2.000 "
          "remuestreos, los mismos para toda la familia) y p unilateral normal con el error del bootstrap. Por "
          "familia, step-down de Romano y Wolf (p_RW) y SPA de Hansen. El p del SPA es el de la versión consistente "
          "(p_SPA_c), como en R21–R25; la inferior (p_SPA_l) se da al lado. Cada regla y cada variante calculada "
          "cuenta en su familia: 5 filtros en H2 y 3 reglas en H4 (las que igualan la exposición media; las 3 en tiempo "
          "real, que no la igualan, forman una familia de sensibilidad aparte). Las decisiones que el pre-registro no "
          "detalla están fijadas en la cabecera de letf_lab/research/r28_vix_timing.py.")

    # ───── resumen ─────
    rep.h("Resumen")
    rep.p(f"- **H1 no se sostiene como se pre-registró: β es positivo pero no significativo.** Fuera de la muestra de "
          f"Cheng ({a} → {b}, {int(main1['n'])} meses, orden al cierre siguiente), el β del rendimiento del mes "
          f"sobre el rendimiento esperado es {num_es(main1['β'], 2)}, con un error de {num_es(main1['EE'], 2)} "
          f"(Newey-West de 3 retardos) y un p unilateral de {fp(main1['p (una cola)'])}, por encima del 0,05 "
          f"pre-registrado. Las {len(sens)} sensibilidades dan β entre {num_es(sens['β'].min(), 2)} y "
          f"{num_es(sens['β'].max(), 2)}, todas positivas, con p entre {fp(sens['p (una cola)'].min())} y "
          f"{fp(sens['p (una cola)'].max())} (tabla ⟦h1⟧). Además, el β depende de un mes: sin {top_m} "
          f"(rendimiento del largo {pct_es(top_i['R del mes'])}) baja a {num_es(top_i['β sin el mes'], 2)}, con "
          f"p = {fp(top_i['p (una cola)'])} (tabla ⟦infl⟧, diagnóstico no pre-registrado). Con su muestra, sus "
          f"coeficientes y su cronología, el laboratorio reproduce lo que publica Cheng: {num_es(chk['β'], 2)} "
          f"({num_es(chk['EE'], 2)}) frente a 0,92 (0,29).")
    others = ", ".join(f"{k} {pp(t2.loc[k, 'Δg'])}" for k in FILTERS if k != best2)
    if v2["rechaza"]:
        fr = "; ".join(f"«{w}» {'frágil' if v2['frágil'][w] else 'no frágil'} (Δg sin sus 3 mejores días: "
                       f"{pp(t2.loc[w, f'Δg sin {TOP_DAYS} mejores días'])} pp)" for w in v2["ganadores"])
        rep.p(f"- **H2 falsada en la ventana principal.** El SPA de la familia de 5 filtros da p = "
              f"{fp(v2['p_SPA_c'])}, por debajo de 0,10. Regla de fragilidad: {fr}.")
    else:
        rep.p(f"- **H2 se sostiene: ningún filtro supera a estar siempre −0,5x tras corregir por la búsqueda.** El "
              f"SPA de la familia de 5 filtros da p = {fp(v2['p_SPA_c'])} (versión inferior: "
              f"{fp(v2['p_SPA_l'])}"
              + (", que sí bajaría de 0,10: el veredicto depende de la versión del SPA, que el pre-registro no fija; "
                 "esta línea usa la consistente, la del resto del laboratorio" if v2["p_SPA_l"] < P_SPA else "")
              + f"). El mejor es «{best2}», que suma {pp(t2.loc[best2, 'Δg'])} pp/año (IC90 "
              f"{ci_txt(t2.loc[best2, 'IC90_bajo'], t2.loc[best2, 'IC90_alto'])}; p nominal "
              f"{fp(t2.loc[best2, 'p'])}; p_RW {fp(t2.loc[best2, 'p_RW'])}). Sin sus 3 mejores días le quedan "
              f"{pp(t2.loc[best2, f'Δg sin {TOP_DAYS} mejores días'])} pp. Los otros cuatro filtros restan: "
              f"{others} pp/año.")
    rep.p(f"- **La prima de Cheng apunta en la misma dirección en H1 y en H2, sin pasar ningún umbral.** Como filtro "
          f"tiene un Δg positivo en los {len(S['h2'])} escenarios ({_range_pp(best2_all)} pp/año), pero su p de "
          f"Romano–Wolf nunca baja de {fp(min(best2_rw))}, y el SPA de la familia no baja de "
          f"{fp(min(spa2_op.values()))} en ningún escenario operable (tabla ⟦grid2⟧). Es una pista, no un "
          f"resultado, y en H1 descansa sobre un solo mes.")
    rep.p(f"- **H4 {'se sostiene' if h4_holds else 'falsada'}: gestionar el −0,5x por volatilidad no lo mejora.** "
          f"Las 3 reglas de igual exposición media (normalización ex post) crecen menos que estar siempre −0,5x (Δg "
          f"{_range_pp(r4['Δg'])} pp/año; ningún IC90 excluye el cero) y el SPA de esa familia da p = {fp(spa4)}. Las 3 "
          f"operables en tiempo real, que no igualan la exposición media (peso medio {_range_num(rt_exp)}) y quedan "
          f"fuera de H4, tampoco (Δg {_range_pp(t4rt.drop(index=BENCH)['Δg'])} pp/año; SPA p = {fp(spa4rt)}). "
          f"El peso sube en las "
          f"épocas de calma, justo antes de los saltos del VIX: el {feb} la regla RV 21 ex post llevaba "
          f"{num_es(W4.loc[vix.FEB_2018[0], 'RV 21 · ex post'], 2)} veces el −0,5x y perdió "
          f"{pct_es(-h4['R'].loc[vix.FEB_2018[0], 'RV 21 · ex post'])} ese día. "
          + ("Ninguna regla de H4 crece más que la referencia en ningún escenario." if h4_neg_all else
             (f"De las {sum(len(EXPOST) for _ in S['h4'])} combinaciones de regla de H4 y escenario, solo "
              + ("una tiene" if len(h4_pos) == 1 else f"{len(h4_pos)} tienen") + " Δg positivo: "
              + "; ".join(f"«{r}» en «{k}» ({pp(S['h4'][k]['tabla'].loc[r, 'Δg'])} pp)" for k, r in h4_pos) + ".")))
    ref_txt = (f"la {LRS} del artículo 2, {pct_es(lrs['CAGR'])} (caída máxima {pct_es(lrs['maxDD'])}), y el SPY "
               f"{pct_es(spy['CAGR'])}" if lrs is not None else f"el SPY {pct_es(spy['CAGR'])}")
    rep.p(f"- **Referencias.** Del 21-12-2005 al 25-9-2026, estar siempre −0,5x creció {pct_es(t2.loc[BENCH, 'CAGR'])} "
          f"al año con una caída máxima del {pct_es(t2.loc[BENCH, 'maxDD'])}; en la misma ventana, {ref_txt}. El "
          f"filtro de la prima de Cheng creció {pct_es(t2.loc[best2, 'CAGR'])} (caída máxima "
          f"{pct_es(t2.loc[best2, 'maxDD'])}), una cifra de la búsqueda sin corregir. Nada de esto descuenta "
          f"impuestos.")
    rep.p(f"- **Corrección declarada en H1.** La primera versión de este módulo usaba el rendimiento esperado sin "
          f"escalar a un mes. Con él, la comprobación en la muestra de Cheng daba {num_es(raw_chk['β'], 3)} "
          f"({num_es(raw_chk['EE'], 3)}) con el mismo R², es decir, 21/34 de lo publicado. Se pasó a la escala de su "
          f"prima (ecuación 2) después de haber visto también el resultado fuera de muestra con la versión sin "
          f"escalar: β = {num_es(raw['β'], 3)}, p = {fp(raw['p (una cola)'])}. Las dos versiones llevan al mismo "
          f"veredicto y las dos están en la tabla ⟦h1⟧.")

    # ───── H1 ─────
    rep.h("1. H1: la prima de Cheng y el rendimiento del futuro")
    rep.p("Cheng (2019, *Review of Financial Studies* 32(1), 180–227, doi:10.1093/rfs/hhy062) define la prima del "
          "VIX como el precio del futuro menos la previsión del VIX en su vencimiento. La previsión sale de un "
          "ARMA(2,2) estimado con datos anteriores a 2004. Su estrategia lleva el contrato que vence el mes "
          "siguiente y, en la última sesión de cada mes, pasa al que vence dentro de dos meses: entra con unas 34 "
          "sesiones hasta el vencimiento y sale con unas 13. El rendimiento esperado del mes es 21/h·(Ê[VIX_T]/F − 1) "
          "al cierre del mes anterior, con h las sesiones hasta el vencimiento. Cheng regresa sobre él el "
          "rendimiento del mes y obtiene β = 0,92 (Newey-West con 3 retardos: 0,29) en 2004–2015; en teoría, una "
          "prima medida sin error daría β = 1. Aquí la previsión se reestima al final de cada mes con los datos "
          "hasta ese día, y el contraste pre-registrado es fuera de su muestra, con la orden al cierre siguiente.")
    tab("arma", "El ARMA(2,2) del VIX: los coeficientes que publica Cheng y los de este módulo (mínimos cuadrados "
        "condicionales). Modelo: VIX_t − μ = φ1(VIX_{t−1} − μ) + φ2(VIX_{t−2} − μ) + e_t + θ1·e_{t−1} + θ2·e_{t−2}",
        C["arma"], digits=3)
    rep.p("Con el VIX de Cboe de 1990–2003, los coeficientes AR y MA coinciden con los de Cheng hasta la segunda "
          "decimal. La media μ está mal identificada porque φ1 + φ2 queda muy cerca de 1, y a un mes apenas influye "
          "en la previsión.")
    tab("prem", "La prima de Cheng por periodo, en sesiones de CFE. Prima: puntos del VIX escalados a un mes; ER: "
        "rendimiento esperado del largo escalado a un mes; «publicado»: con los coeficientes de Cheng",
        C["prem"], pct={"días con prima > 0", "ER medio del mes", "días con prima > 0 (publicado)"}, digits=3)
    show = T1[["tipo", "desde", "hasta", "n", "β", "EE", "t", "p (una cola)", "R²", "estimador"]]
    tab("h1", "H1: regresión del rendimiento del futuro sobre el rendimiento esperado de Cheng. β > 0: cuanto mayor "
        "la prima, mayor el rendimiento del corto. p unilateral de H0: β ≤ 0. Mensual salvo la fila diaria; n: "
        "meses (o sesiones)", show, digits=3)
    rep.p(f"El contraste pre-registrado es la primera fila: β = {num_es(main1['β'], 2)} con un p unilateral de "
          f"{fp(main1['p (una cola)'])} en {int(main1['n'])} meses. "
          + ("H1 se sostiene al 5% unilateral." if h1_ok else
             "El pre-registro falsa H1 si β ≤ 0 o si no es significativo, así que H1 no se sostiene: el signo y "
             "el tamaño son los de Cheng, pero el error es demasiado grande para distinguirlo de cero al 5%.")
          + f" Antes de marzo de 2018 hay solo {int(pre1['n'])} meses (β = {num_es(pre1['β'], 2)}, p = "
          f"{fp(pre1['p (una cola)'])}); después, {int(post1['n'])} (β = {num_es(post1['β'], 2)}, p = "
          f"{fp(post1['p (una cola)'])}). La versión diaria, con el rendimiento de las 21 sesiones siguientes, da "
          f"β = {num_es(d1['β'], 2)} (p = {fp(d1['p (una cola)'])}, {d1['estimador']}). En toda la muestra, "
          f"2004–2026, β = {num_es(full['β'], 2)} con p = {fp(full['p (una cola)'])}, pero esa muestra incluye la "
          f"de Cheng y no es un contraste fuera de muestra.")
    tab("infl", "Diagnóstico no pre-registrado: β y p del contraste principal quitando un mes. Los 5 meses cuya "
        "ausencia más baja β; R: rendimiento del largo en ese mes, ER: rendimiento esperado al cierre del mes "
        "anterior", infl, pct={"R del mes", "ER del mes"}, digits=3)
    prev_m = month_es(pd.Period(infl.index[0], "M") - 1)
    rep.p(f"El coeficiente fuera de muestra se apoya en {top_m}: al cierre de {prev_m} la prima era muy "
          f"negativa (rendimiento esperado del largo de {pct_es(top_i['ER del mes'])} en el mes) y el largo ganó "
          f"{pct_es(top_i['R del mes'])}. Sin ese mes, β = {num_es(top_i['β sin el mes'], 2)} (p = "
          f"{fp(top_i['p (una cola)'])}); sin cualquier otro, β queda por encima de "
          f"{num_es(infl['β sin el mes'].iloc[1], 2)}. La figura 2 lo muestra: es el punto de arriba a la derecha.")
    fig_h1(C, rep)
    fig_h1_scatter(C, rep)

    # ───── H2 ─────
    rep.h("2. H2: filtros de estructura de plazos sobre el −0,5x")
    rep.p(f"Referencia: estar siempre −0,5x. Cada filtro lleva el −0,5x cuando su condición se cumple al cierre de t "
          f"y letras si no. Las condiciones son: F1 > VIX (el primer futuro por encima del VIX); VIX < VIX3M; F2 > F1; "
          f"el roll diario de Simon y Campasano (2014), (F − VIX)/n > {num_es(SC_THR, 2)}, con el primer contrato de "
          f"más de {SC_MIN} sesiones hasta el vencimiento; y prima de Cheng > 0, con la prima de H1. El VIX3M de Cboe "
          f"empieza el 18-9-2009; antes, esa regla lleva el −0,5x, igual que la referencia.")
    tab("fst", "Tiempo dentro de cada filtro (estado al cierre) y cambios de estado por año", C["fstats"],
        pct={c for c in C["fstats"].columns if c.startswith("dentro")})
    tab("fam2", f"H2 en la ventana principal ({dmy(h2['desde'])} → {dmy(h2['hasta'])}, convención de liquidación, "
        f"orden al cierre siguiente). Δg frente a siempre −0,5x, con IC90 y p unilateral por bootstrap emparejado; "
        f"p_RW: Romano–Wolf de la familia de 5. SPA de la familia: p_SPA_c = {fp(h2['spa']['p_SPA_c'])} y p_SPA_l = "
        f"{fp(h2['spa']['p_SPA_l'])}. Las dos últimas filas son referencias en la misma ventana, fuera de la familia",
        fmt_family(h2, FILTERS, C["ref"]), pct={"peor día", "tiempo dentro"})
    if v2["rechaza"]:
        rep.p("El SPA rechaza al 10%: " + "; ".join(
            f"«{w}» queda en {pp(t2.loc[w, f'Δg sin {TOP_DAYS} mejores días'])} pp/año sin sus {TOP_DAYS} mejores "
            f"días ({'frágil' if v2['frágil'][w] else 'no frágil'})" for w in v2["ganadores"]) + ".")
    else:
        rep.p(f"El SPA no rechaza (p = {fp(v2['p_SPA_c'])}), así que H2 se sostiene y la regla de fragilidad del "
              f"pre-registro no hace falta. La versión inferior del SPA, que es una cota por abajo del p, da "
              f"{fp(v2['p_SPA_l'])}; el veredicto usa la consistente, como el resto del laboratorio. Con 2.000 "
              f"remuestreos, el error de Monte Carlo de un p de 0,15 es de unas 0,008.")
    tb = t2.drop(index=BENCH)[["Δg", f"Δg sin {TOP_DAYS} mejores días", "mejores días"]].copy()
    tb["Δg"] = 100 * tb["Δg"]
    tb[f"Δg sin {TOP_DAYS} mejores días"] = 100 * tb[f"Δg sin {TOP_DAYS} mejores días"]
    tb.columns = ["Δg (pp)", f"Δg sin sus {TOP_DAYS} mejores días (pp)", "esos días"]
    tab("frag", f"Regla de fragilidad del pre-registro, aplicada a todos los filtros: Δg sin los {TOP_DAYS} días de "
        f"mayor ventaja diaria sobre la referencia (ventana principal)", tb)
    top_best = t2.loc[best2, "mejores días"]
    rep.p(f"Los tres mejores días de «{best2}» son {top_best}: los grandes saltos del VIX, con el filtro fuera. "
          f"Sin ellos conserva {pp(t2.loc[best2, f'Δg sin {TOP_DAYS} mejores días'])} pp/año; los demás filtros "
          f"quedan por debajo de cero.")
    tab("worst2", "Los 10 peores días de estar siempre −0,5x (ventana principal) y el peso que cobraba cada filtro "
        "(1: −0,5x; 0: letras)", worst, pct={"siempre −0,5x"})
    feb_in = [f for f in FILTERS if worst.loc[feb, f] > 0] if feb in worst.index else []
    ridx = h2["R"].index
    sig_day = ridx[ridx.get_loc(vix.FEB_2018[0]) - LAG]
    in_best = [d for d in worst.index if worst.loc[d, best2] > 0]
    rep.p(f"El {feb} (el peor día, {pct_es(worst.loc[feb, 'siempre −0,5x'])} con la liquidación) "
          + (f"{'estaba' if len(feb_in) == 1 else 'estaban'} dentro {y_join(['«' + f + '»' for f in feb_in])}: "
             f"con la señal del cierre del {dmy(sig_day)} (orden al cierre de la sesión siguiente) la curva seguía "
             f"en contango." if feb_in else "todos los filtros estaban fuera.")
          + " Ningún filtro esquiva todos los peores días"
          + (f"; «{best2}» estaba dentro {y_join(['el ' + d for d in in_best])}." if in_best else "."))
    tab("grid2", "H2 por escenario: Δg de cada filtro frente a siempre −0,5x (pp/año; * = p_RW < 0,10) y SPA de la "
        "familia. g referencia: crecimiento logarítmico de siempre −0,5x en el escenario (%)",
        scenario_grid(S["h2"], FILTERS), digits=3)
    same = spa2_all.get("mismo cierre (no operable)")
    rep.p(f"En ningún escenario operable el SPA baja de 0,10 (mínimo: {fp(min(spa2_op.values()))}, en "
          f"«{min(spa2_op, key=spa2_op.get)}»). Con el mismo cierre, no operable, "
          f"p = {fp(same)}. La prima de Cheng es el mejor filtro en "
          f"{sum(S['h2'][k]['tabla'].drop(index=BENCH)['Δg'].idxmax() == best2 for k in S['h2'])} de "
          f"{len(S['h2'])} escenarios.")
    fig_dg(S["h2"], rep, FILTERS, F_COLORS, "h2_dg", "H2: Δg de cada filtro frente al −0,5x siempre dentro",
           "Figura 3: Δg de cada filtro frente a estar siempre −0,5x, con su IC del 90% (bootstrap emparejado, sin "
           "corregir por la búsqueda), en la ventana principal y antes y después del 28-2-2018")
    fig_wealth(S, rep)

    # ───── H4 ─────
    rep.h("3. H4: el −0,5x gestionado por volatilidad")
    rep.p(f"Peso en el −0,5x: w = mín(c/σ̂², {num_es(W_MAX, 0)}), con σ̂² la varianza diaria prevista del propio "
          f"−0,5x al cierre de t. Tres previsiones: RV 21 (media de r² de 21 sesiones, como Moreira y Muir 2017), "
          f"EWMA con λ = 0,94 y GJR-GARCH(1,1) reestimado cada 1 de enero. «Ex post»: c tal que la media de w en la "
          f"ventana sea 1, que es la «igual exposición media» del pre-registro; usa toda la ventana, como Moreira y "
          f"Muir, y no es operable; esas 3 reglas son la familia de H4. «Tiempo real»: c es la inversa de la media "
          f"creciente de 1/σ̂², y su exposición media queda entre {num_es(min(rt_exp), 2)} y {num_es(max(rt_exp), 2)}: no "
          f"es la «igual exposición media» del pre-registro, así que esas 3 reglas van en una familia de sensibilidad "
          f"aparte, con su propio SPA, y no deciden H4. El "
          f"resto va en letras; con w > 1 se pide prestado a la letra + {num_es(100 * MARGIN_EXTRA, 1)}%. Con w = 2 "
          f"la exposición es −1x: por eso se dan las dos convenciones y los resultados sin el 5 y el 6-2-2018.")
    tab("fam4", f"H4 en la ventana principal ({dmy(h4['desde'])} → {dmy(h4['hasta'])}, convención de liquidación, "
        f"orden al cierre siguiente): las 3 reglas de igual exposición media. SPA de la familia: p_SPA_c = {fp(spa4)} y "
        f"p_SPA_l = {fp(h4['spa']['p_SPA_l'])}; p_RW en la familia de 3. Exposición media: peso medio en el −0,5x",
        fmt_family(h4, EXPOST, C["ref"], inside=False), pct={"peor día"})
    tab("fam4rt", f"Sensibilidad fuera de H4: las 3 reglas en tiempo real (operables, exposición media por debajo de 1), "
        f"misma ventana. SPA de esta familia: p_SPA_c = {fp(spa4rt)} y p_SPA_l = {fp(h4rt['spa']['p_SPA_l'])}; p_RW en "
        f"la familia de 3", fmt_family(h4rt, REALTIME, inside=False), pct={"peor día"})
    wt = pd.DataFrame({k: {"peso medio": W4[k].mean(), "sesiones en el tope": (W4[k] >= W_MAX - 1e-9).mean(),
                           "peso p10": W4[k].quantile(0.1), "peso p90": W4[k].quantile(0.9),
                           "peso el 5-2-2018": W4[k].get(vix.FEB_2018[0], np.nan),
                           "peso el 16-3-2020": W4[k].get(pd.Timestamp("2020-03-16"), np.nan),
                           "peso el 5-8-2024": W4[k].get(pd.Timestamp("2024-08-05"), np.nan)}
                       for k in MANAGED}).T
    tab("w4", "Pesos que cobran las reglas de volatilidad en la ventana principal", wt, pct={"sesiones en el tope"})
    tab("worst4", "Los 10 peores días de estar siempre −0,5x y el peso que cobraba cada regla de volatilidad (las 3 de "
        "H4 y las 3 en tiempo real)", worst_days({"R": R4all, "W": W4}, MANAGED), pct={"siempre −0,5x"})
    rep.p(f"El mecanismo se ve en las tablas ⟦w4⟧ y ⟦worst4⟧. La varianza prevista es baja justo antes de los saltos, "
          f"así que el peso está cerca del tope cuando llega el golpe; después del salto baja a casi nada y se pierde "
          f"la recuperación. El −0,5x ya es un corto de volatilidad: escalarlo por la inversa de su varianza le suma "
          f"exposición en calma y se la quita tras el susto, la parte que el corto cobra. La mejor regla, "
          f"«{best4}», queda en {pp(t4.loc[best4, 'Δg'])} pp/año (IC90 "
          f"{ci_txt(t4.loc[best4, 'IC90_bajo'], t4.loc[best4, 'IC90_alto'])}). Ni siquiera el Sharpe, la medida de "
          f"Moreira y Muir, mejora: {num_es(t4.loc[BENCH, 'sharpe'], 2)} siempre −0,5x frente a "
          f"entre {num_es(r4['sharpe'].min(), 2)} y {num_es(r4['sharpe'].max(), 2)} en las reglas.")
    tab("grid4", "H4 por escenario: Δg de cada regla de igual exposición media frente a siempre −0,5x (pp/año; * = "
        "p_RW < 0,10 en la familia de 3) y SPA de la familia. g referencia: crecimiento logarítmico de siempre −0,5x en "
        "el escenario (%)", scenario_grid(S["h4"], EXPOST), digits=3)
    tab("grid4rt", "Sensibilidad fuera de H4, por escenario: las 3 reglas en tiempo real (pp/año; * = p_RW < 0,10 en su "
        "familia de 3) y SPA de esa familia", scenario_grid(S["h4rt"], REALTIME), digits=3)
    wo = S["h4"]["liquidación sin 5-6-2-2018"]["tabla"].drop(index=BENCH)["Δg"]
    rep.p(f"Sin el 5 y el 6-2-2018 las 3 reglas de H4 siguen por debajo de la referencia ({_range_pp(wo)} pp/año), y "
          f"también desde el 28-2-2018 con SVXY real: el resultado no depende solo de febrero de 2018. En las 3 en tiempo "
          f"real, el SPA de su familia da "
          + ("p = 1 en todos los escenarios operables" if spa4rt_min > 0.999 else
             f"p ≥ {fp(spa4rt_min)} en todos los escenarios operables")
          + (f"; tienen Δg positivo en {len(rt_pos)} combinaciones de regla y escenario." if rt_pos else
             " y no crecen más que la referencia en ningún escenario."))
    merged = {k: {"tabla": pd.concat([S["h4"][k]["tabla"], S["h4rt"][k]["tabla"].drop(index=BENCH)])} for k in S["h4"]}
    fig_dg(merged, rep, MANAGED, {k: FC_COLORS[k.split(" · ")[0]] for k in MANAGED}, "h4_dg",
           "H4: Δg de la gestión por volatilidad frente al −0,5x siempre dentro",
           "Figura 5: Δg de cada regla de volatilidad frente a estar siempre −0,5x, con su IC del 90% (sin corregir por "
           "la búsqueda), en la ventana principal y antes y después del 28-2-2018. Las 3 reglas «ex post» son la familia "
           "de H4 (igual exposición media); las 3 «tiempo real» no igualan la exposición media y son una sensibilidad")
    fig_h4_weights(S, rep)
    cm = C["S"]["c"][MAIN]
    rep.p("Nivel de la normalización ex post en la ventana principal: el peso es 1 cuando la volatilidad prevista del "
          "−0,5x es de " + ", ".join(f"{pct_es(np.sqrt(v * ANN))} anual ({f})" for f, v in cm.items())
          + f"; la volatilidad realizada del −0,5x en la ventana fue de {pct_es(t4.loc[BENCH, 'vol'])}.")

    # ───── lectura y límites ─────
    rep.h("4. Qué dicen estos resultados y qué no")
    k_sig = (1.645 * main1["EE"]) ** 2
    k_pow = ((1.645 + 0.842) * main1["EE"]) ** 2
    rep.p(f"- H1 y H2 miran la misma variable: la prima de Cheng. El coeficiente fuera de muestra es del tamaño que "
          f"publicó Cheng, y el filtro que la usa es el mejor de su familia en "
          f"{sum(S['h2'][k]['tabla'].drop(index=BENCH)['Δg'].idxmax() == best2 for k in S['h2'])} de "
          f"{len(S['h2'])} escenarios. Pero ninguno de los dos pasa el umbral pre-registrado (5% unilateral en H1; "
          f"p_SPA < 0,10 en H2), y el β de H1 lo sostiene {top_m}. Con el error de {num_es(main1['EE'], 2)} de estos "
          f"{int(main1['n'])} meses, un β de 1 solo sería significativo al 5% con unas {num_es(k_sig, 1)} veces más "
          f"meses, y detectarlo con una potencia del 80% pediría unas {num_es(k_pow, 1)} veces más.")
    pre_f1 = S["h2"]["antes del 28-2-2018"]["tabla"].loc[F_F1, "Δg"]
    post_f1 = S["h2"]["desde el 28-2-2018"]["tabla"].loc[F_F1, "Δg"]
    rep.p(f"- Los filtros de pendiente (F1 > VIX, VIX < VIX3M, F2 > F1) y el roll de Simon y Campasano no mejoran al "
          f"−0,5x con la orden al cierre siguiente. La exploración previa, que no se cita, había visto en «F1 > VIX» "
          f"una ventaja que dependía de un día; con el protocolo del laboratorio su Δg es de "
          f"{pp(t2.loc[F_F1, 'Δg'])} pp/año en la ventana principal ({pp(pre_f1)} antes del 28-2-2018 y "
          f"{pp(post_f1)} después).")
    rep.p("- Gestionar por volatilidad el corto de volatilidad crece menos que no gestionarlo en todas las variantes "
          "y en casi todos los escenarios, aunque ningún intervalo individual excluye el cero. Moreira y Muir (2017) "
          "encuentran lo contrario en el mercado de acciones y en factores; Cederburg, O'Doherty, Wang y Yan (2020) "
          "ya advierten que fuera de muestra esa mejora suele desaparecer.")
    rep.h("Límites", 3)
    rep.p("- La muestra tiene pocos episodios extremos: 2008, el 5–6-2-2018, marzo de 2020 y agosto de 2024 deciden "
          "buena parte de cualquier Δg. La regla de los 3 mejores días lo mide para cada filtro.")
    rep.p("- El −0,5x es sintético hasta el 27-2-2018 (índice de S&P reconstruido en R26) y SVXY real después. El "
          "sintético no reproduce una liquidación anticipada como la del XIV.")
    rep.p("- H1 usa las liquidaciones diarias de CFE, sin horquilla. La prima depende de la previsión del VIX, y un "
          "error de medida sesga β hacia 0.")
    rep.p("- La normalización ex post de H4 usa la ventana entera y no es operable. La de tiempo real sí lo es, pero "
          "no iguala la exposición media, y por eso no entra en la familia de H4.")
    rep.p("- No se descuentan impuestos. Que un minorista de la UE no pueda comprar ETPs de EE. UU. sin KID (PRIIPs) "
          "viene de un extracto de IBKR y está sin verificar aquí.")
    rep.p(f"- La versión del SPA: el pre-registro dice «p_SPA < 0,10» sin fijar la versión. Esta línea usa la "
          f"consistente (p_SPA_c), como R21–R25; con la inferior (p_SPA_l = {fp(v2['p_SPA_l'])}) el veredicto de H2 "
          + ("cambiaría." if v2["p_SPA_l"] < P_SPA else "sería el mismo."))
    rep.p("- Cambio tras la revisión del laboratorio, con los resultados ya vistos: la familia de H4 pasó de 6 reglas "
          "a las 3 que igualan la exposición media; las 3 en tiempo real quedan como sensibilidad con su propio SPA.")
    rep.p("- Sin verificar: la fecha exacta del cambio de hora de la liquidación en 2020 y las demás marcas de R26.")

    rep.h("5. Reproducción")
    rep.p("`cd letf && .venv/bin/python -c \"from letf_lab.research import r28_vix_timing as m; m.run()\"`. "
          "Semilla 28, con un generador por contraste. Funciones: h1_table, premium, rolling_contract, arma_refits, "
          "signals, h2_returns, h4_returns, run_scenarios y family_test. Tests: tests/test_r28_vix_timing.py.")

    for key, n in num.items():
        rep.lines = [ln.replace(f"⟦{key}⟧", str(n)) for ln in rep.lines]


def run(C: dict | None = None):
    warnings.filterwarnings("ignore")
    rep = Report("r28_vix_timing", "R28 — ¿Se puede cronometrar la prima de la volatilidad? (H1, H2 y H4)")
    for f in rep.dir.glob("tabla_*.csv"):
        f.unlink()
    C = C or compute()
    write_report(C, rep)
    rep.save()
    return C
