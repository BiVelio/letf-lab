"""R17 — Referencias a igual riesgo para la LRS y la LRS-Kelly (I4).

Pregunta: ¿la LRS aporta algo por sincronizarse con el mercado, o su ventaja
sobre el 1x es sólo apalancamiento? Se comparan la LRS 3x original (Gayed y
Bilello 2016: media de 200 sesiones sobre el índice de rentabilidad total) y la
LRS-Kelly de R13 con referencias del mismo riesgo en el S&P 500 (1938–), el
Nasdaq-100 (1985-10–, datos reales) y los semiconductores (2004–). La LRS 3x
con la señal de precio (R13) se muestra como sensibilidad; en el S&P 500 entra
en la familia del SPA, y en los otros dos mercados, donde el precio es
aproximado, no.

Referencias estáticas (no se sincronizan con el mercado):
- 1x y 3x comprar y mantener;
- apalancamiento constante con la misma exposición media que la estrategia:
  una fracción fija L/3 en el 3x y el resto en letras, con los mismos
  instrumentos y costes que la LRS. OJO: la diferencia con él NO es sólo
  sincronización. Como g(L) es cóncava, una exposición que varía sin relación
  alguna con el mercado ya crece menos que su media fija (≈ −½·σ²·Var(L_t));
  por eso se añade la nula sin habilidad de abajo;
- apalancamiento constante con la misma volatilidad realizada (cota: el L se
  fija con la volatilidad de toda la muestra, que nadie conocía de antemano);
- ½ Kelly ex ante: L = ½·(μ̂ − f̂ − s)/σ̂² con la prima compartida μ̂ − f̂
  (kelly.premium: media creciente de la rentabilidad sobre la financiación f,
  con datos hasta la sesión anterior) y la varianza de toda la historia previa,
  revisado cada fin de año (sin reaccionar a la volatilidad del momento).
Referencias dinámicas de la literatura, con sus parámetros originales y
adaptadas a una cartera larga o en letras con el 3x:
- regla de 10 meses de Faber (2007): invertido si el índice al cierre de mes
  supera su media de 10 meses; si no, letras; con 1x y con 3x. Como en Faber
  («All data series are total return series including dividends», regla 2 de
  su sistema), la media se calcula sobre el índice de rentabilidad total, el
  mismo de la señal de la LRS original; la orden, como en todo el laboratorio,
  al cierre siguiente (Faber opera al cierre del día de la señal);
- momento de serie temporal a 12 meses (Moskowitz, Ooi y Pedersen 2012),
  adaptado: 3x si la rentabilidad de 12 meses sobre las letras es positiva; si
  no, letras. Sin cortos ni escalado por volatilidad, que el original sí tiene;
- cartera gestionada por volatilidad (Moreira y Muir 2017): exposición
  c/σ̂²_{t−1} con la varianza realizada del mes anterior, revisión mensual y
  tope de 3x (se hace con el 3x y letras). La c iguala la volatilidad a la del 1x; se estima en tiempo
  real con los meses ya observados (la crítica de Cederburg et al. 2020 a la
  c de muestra completa, que se muestra aparte como cota que mira el futuro).

Medidas:
- equivalente cierto CRRA con γ = 1, 2, 3 y 5 (MPPM de Goetzmann, Ingersoll,
  Spiegel y Welch 2007) sobre rentabilidades MENSUALES en exceso de las
  letras: Θ = 12/(1−γ)·ln E[((1+r)/(1+r_f))^(1−γ)], en % anual continuo;
- SPA de Hansen (2005) de la familia de reglas dinámicas frente a la mejor
  referencia estática del conjunto discreto de cada γ (elegida a posteriori, lo
  que endurece la prueba) y frente al mejor L constante de la rejilla 0–4 con
  cada una de las dos formas, con la diferencia de utilidad CRRA mensual como
  medida;
- umbral de sobrecoste de financiación: pb/año adicionales sobre el nocional
  del fondo apalancado (el diferencial del swap) que anulan la ventaja, pagados
  por todos (la estrategia y la referencia, que usa el mismo fondo 3x) o solo
  por la estrategia;
- equivalente cierto según el horizonte de evaluación: el mismo EC con sumas
  de log-excesos en bloques no solapados de 1, 12, 36 y 60 meses;
- frontera g(L): crecimiento logarítmico del apalancamiento constante L de 0
  a 4, con cada estrategia situada en su exposición media;
- nula sin habilidad para la sincronización: la trayectoria de pesos
  ejecutados de cada regla se desplaza circularmente dentro de la ventana
  (de dos años en dos años como mínimo, todos los desplazamientos de 21 en 21
  sesiones), con los mismos fondos, letras y costes de rotación. Conserva la
  distribución y la persistencia de la exposición pero rompe su relación con
  el mercado. Sincronización = Δg observado − media de Δg nulo; p = fracción
  de desplazamientos que igualan o superan lo observado;
- incertidumbre muestral de Δg y ΔEC frente a los L emparejados: bootstrap
  estacionario de meses (bloque medio de 3 meses, 2000 remuestreos), p
  unilateral de «no supera».

Protocolo común con R13: señal al cierre de t, orden al cierre de t+1; el 3x
paga sobre toda su exposición el tipo de call money (hasta 1954) o de los
fondos federales (después) más el diferencial calibrado y el gasto del fondo;
5 pb por unidad de rotación; 3x real desde su lanzamiento. Los p de titular se
exportan, con su procedencia, a p_titular.csv (los lee R19).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from letf_lab import kelly, metrics, provenance, stats
from letf_lab.report import FIG_W, Report, comma_axis, legend_band, num_es, pct_es
from letf_lab.research import r12_long_ml as r12
from letf_lab.research import r13_lrs as r13

SEED = 17
B = 2000                      # remuestreos del SPA
BLOCK_M = 3                   # bloque medio del bootstrap estacionario (meses ≈ 63 sesiones)
GAMMAS = (1, 2, 3, 5)
GAMMA_FINE = np.round(np.arange(1.0, 5.001, 0.25), 2)   # sólo para dibujar las curvas
L_GRID = np.round(np.arange(0.0, 4.0001, 0.05), 2)
LAG = r13.EXEC_LAG
KELLY = r13.KELLY
MM_MIN_MONTHS = 120           # diez años de meses antes de la primera c en tiempo real
MM_CAP = 3.0                  # la exposición de Moreira–Muir se hace con el 3x: tope 3x
FABER_MONTHS = 10
TSMOM_MONTHS = 12
COST_RANGE = (-3000.0, 5000.0)  # pb/año en los que se busca el umbral
NULL_MIN_SHIFT = 504          # nula sin habilidad: desplazamiento mínimo de la exposición (2 años)
NULL_STEP = 21                # … y todos los desplazamientos de 21 en 21 sesiones

FOCAL = ("LRS 3x (original)", "LRS-Kelly")
PX = r13.PX_LRS                                   # sensibilidad: señal de precio (S&P 500)
PX_APPROX = r13.price_label(r13.ORIGINAL, approx=True)   # Nasdaq-100 y semis: precio aproximado
EC_HORIZONS = (1, 12, 36, 60)                     # meses por bloque del EC según el horizonte
EC_H_GAMMAS = (2, 3, 5)
N_SHUFFLE = 200                                    # permutaciones de meses de la referencia barajada
SEED_SHUFFLE = 1700                                # su semilla propia (no mueve el resto del informe)
NEW_DYNAMIC = ("Faber 10 meses 1x", "Faber 10 meses 3x", "TSMOM 12 meses 3x",
               "Moreira–Muir (c en tiempo real)")
MM_FULL = "Moreira–Muir (c de muestra completa)"
KELLY_EX_ANTE = "½ Kelly ex ante (anual)"
COMPARE = FOCAL + NEW_DYNAMIC   # dinámicas que se muestran en tablas y figuras


def px_key(d: dict) -> str:
    """Nombre de la LRS 3x con la señal de precio en el mercado d (aproximado fuera del S&P 500)."""
    return PX if d.get("px_real", False) else PX_APPROX


def compare(d: dict) -> tuple:
    """Reglas dinámicas de las tablas y figuras del mercado d: las focales, la de precio y las
    de la literatura."""
    return FOCAL + (px_key(d),) + NEW_DYNAMIC


# ───────────────────────────── matemáticas (probadas en tests) ─────────────────────────────

def crra_ce(log_excess: np.ndarray, gamma: float, periods: int = 12) -> float:
    """Equivalente cierto anual (MPPM de Goetzmann et al. 2007).

    Entra x = ln((1+r)/(1+r_f)) por periodo. Θ = 1/((1−γ)Δt)·ln media(e^{(1−γ)x});
    con γ = 1, Θ = media(x)/Δt. Resultado en tasa continua anual."""
    x = np.asarray(log_excess, float)
    if gamma == 1:
        return float(periods * x.mean())
    a = (1 - gamma) * x
    m = a.max()
    return float(periods / (1 - gamma) * (m + np.log(np.mean(np.exp(a - m)))))


def crra_utility(log_excess: np.ndarray, gamma: float) -> np.ndarray:
    """Utilidad CRRA del rendimiento bruto en exceso R = e^x (misma ordenación que Θ)."""
    x = np.asarray(log_excess, float)
    if gamma == 1:
        return x
    return np.expm1((1 - gamma) * x) / (1 - gamma)


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Último día hábil de cada mes; el mes en curso (el último del índice) se descarta."""
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(idx.to_period("M")).max().values[:-1])


def to_daily(decision: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Decisión tomada al cierre de una fecha → vigente hasta la siguiente decisión."""
    return decision.reindex(idx).ffill()


def faber_signal(px: pd.Series, months: int = FABER_MONTHS) -> pd.Series:
    """1 si el precio al cierre de mes supera su media de `months` cierres de mes."""
    me = month_ends(px.index)
    pm = px.reindex(me)
    sma = pm.rolling(months).mean()
    return to_daily((pm > sma).astype(float).where(sma.notna()), px.index)


def tsmom_signal(u: pd.Series, rf: pd.Series, months: int = TSMOM_MONTHS) -> pd.Series:
    """1 si la rentabilidad de los últimos `months` meses supera a la de las letras."""
    cs = (np.log1p(u) - np.log1p(rf.reindex(u.index))).fillna(0).cumsum()
    c = cs.reindex(month_ends(u.index))
    past = c - c.shift(months)
    return to_daily((past > 0).astype(float).where(past.notna()), u.index)


def monthly_excess_and_rv(u: pd.Series, rf: pd.Series, min_days: int = 10) -> tuple[pd.Series, pd.Series]:
    """Rentabilidad mensual en exceso de las letras y varianza realizada anualizada
    del mes (Moreira–Muir), fechadas en el último día hábil de cada mes."""
    per = u.index.to_period("M")
    rf = rf.reindex(u.index).fillna(0)
    f = np.expm1(np.log1p(u).groupby(per).sum()) - np.expm1(np.log1p(rf).groupby(per).sum())
    rv = u.groupby(per).var(ddof=0) * 252
    rv = rv.where(u.groupby(per).count() >= min_days)
    me = month_ends(u.index)
    f, rv = f.iloc[:len(me)], rv.iloc[:len(me)]
    f.index = rv.index = me
    return f, rv


def mm_constant(f: pd.Series, rv: pd.Series, real_time: bool = True,
                min_months: int = MM_MIN_MONTHS, sample: slice | None = None) -> pd.Series:
    """c de Moreira–Muir: iguala la volatilidad de c·f_t/σ̂²_{t−1} a la de f_t.

    En tiempo real, c_t sólo usa los pares (f_s, σ̂²_{s−1}) con s ≤ t; con
    real_time=False es una constante calculada con los meses de `sample`."""
    x = f / rv.shift(1)
    fx = f.where(x.notna())
    if real_time:
        return fx.expanding(min_months).std() / x.expanding(min_months).std()
    sl = sample or slice(None)
    return pd.Series(float(fx.loc[sl].std() / x.loc[sl].std()), index=f.index)


def mm_exposure(u: pd.Series, rf: pd.Series, real_time: bool = True,
                sample: slice | None = None, cap: float = MM_CAP) -> tuple[pd.Series, pd.Series]:
    """Exposición diaria al índice (0–cap) decidida al cierre de cada mes, y la c usada."""
    f, rv = monthly_excess_and_rv(u, rf)
    c = mm_constant(f, rv, real_time, sample=sample)
    w = (c / rv).clip(upper=cap)
    return to_daily(w, u.index), c


def const_leverage(d: dict, L: float, efficient: bool = False) -> pd.Series:
    """Apalancamiento constante L, reequilibrado a diario.

    Por defecto con los instrumentos de la LRS: L/3 en el 3x y el resto en letras.
    efficient=True: la forma más barata — 1x + letras si L ≤ 1, 1x + 3x si
    1 < L ≤ 3. Por encima de 3x, el 3x más un préstamo al tipo interbancario más
    el diferencial del swap."""
    u, rf, x3 = d["u"], d["rf"], d["x3"]
    if L > 3:
        b = L / 3 - 1
        return (L / 3) * x3 - b * (d["f"] + d["spread"] / 252)
    if efficient and L <= 1:
        return L * u + (1 - L) * rf
    if efficient:
        w3 = (L - 1) / 2
        return w3 * x3 + (1 - w3) * u
    return (L / 3) * x3 + (1 - L / 3) * rf


def match_vol(d: dict, target: float, sl: slice) -> float:
    """L constante (3x + letras) con la misma volatilidad diaria que `target` en `sl`."""
    vol = lambda L: const_leverage(d, L).loc[sl].std() * np.sqrt(252)
    if target >= vol(4.0):
        return 4.0
    return float(brentq(lambda L: vol(L) - target, 1e-6, 4.0, xtol=1e-6))


def month_starts_pos(idx: pd.DatetimeIndex) -> np.ndarray:
    """Posiciones del primer día de cada mes (para agregar con reduceat)."""
    per = idx.to_period("M")
    return np.flatnonzero(np.r_[True, per[1:] != per[:-1]])


def monthly_log(r: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(np.log1p(np.maximum(r, -0.9999)), starts)


def break_even(r_s: np.ndarray, n_s: np.ndarray, r_b: np.ndarray, n_b: np.ndarray,
               lrf_m: np.ndarray, starts: np.ndarray, gamma: float, both: bool) -> float:
    """Sobrecoste adicional Δ (pb/año sobre el nocional apalancado n) que iguala el
    equivalente cierto de la estrategia al de la referencia.

    both=False: sólo paga la estrategia (colchón de su ventaja); both=True: el
    sobrecoste lo pagan las dos (financiación más cara para todos).

    Con ventaja (EC_s > EC_b) devuelve el Δ > 0 que la anula, o +inf si no se
    anula en COST_RANGE (p. ej. la referencia usa tanto nocional o más). Sin
    ventaja devuelve el Δ < 0 (abaratamiento) que la igualaría, o −inf si ni así."""
    def ce(r, n, delta):
        return crra_ce(monthly_log(r - delta / 1e4 / 252 * n, starts) - lrf_m, gamma)

    ce_b0 = ce(r_b, n_b, 0.0)
    h = lambda dl: ce(r_s, n_s, dl) - (ce(r_b, n_b, dl) if both else ce_b0)
    lo, hi = COST_RANGE
    h0 = h(0.0)
    if h0 > 0:
        return float(brentq(h, 0.0, hi, xtol=0.05)) if h(hi) < 0 else np.inf
    if h0 == 0:
        return 0.0
    return float(brentq(h, lo, 0.0, xtol=0.05)) if h(lo) > 0 else -np.inf


def crra_ce_rows(X: np.ndarray, gamma: float, periods: int = 12) -> np.ndarray:
    """crra_ce aplicado a cada fila de X (remuestreos o desplazamientos)."""
    X = np.atleast_2d(np.asarray(X, float))
    if gamma == 1:
        return periods * X.mean(axis=1)
    a = (1 - gamma) * X
    m = a.max(axis=1, keepdims=True)
    return periods / (1 - gamma) * (m[:, 0] + np.log(np.mean(np.exp(a - m), axis=1)))


def shift_grid(n: int, min_shift: int = NULL_MIN_SHIFT, step: int = NULL_STEP) -> np.ndarray:
    """Desplazamientos circulares de la nula: al menos `min_shift` sesiones en los dos sentidos."""
    return np.arange(min_shift, n - min_shift + 1, step)


def shifted_paths(w3: np.ndarray, w1: np.ndarray, x: np.ndarray, u: np.ndarray, rf: np.ndarray,
                  starts: np.ndarray, shifts: np.ndarray, cost_bps: float = r13.COST_BPS,
                  chunk: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nula sin habilidad: la regla con su trayectoria de pesos EJECUTADOS (w3 en el fondo
    x, w1 en el 1x, resto en letras) desplazada circularmente s sesiones, para cada s.

    El desplazamiento conserva la distribución y la persistencia de la exposición (y
    su rotación, que se cobra igual que en r13.run_weights) pero rompe su relación con
    el mercado. Devuelve, por desplazamiento, el crecimiento log anual (×252 por
    sesión), la volatilidad anual y las log-rentabilidades mensuales (sin restar letras)."""
    n = len(x)
    base = np.arange(n)
    G, V, M = [], [], []
    for i in range(0, len(shifts), chunk):
        s = np.asarray(shifts[i:i + chunk])
        ix = (base[None, :] - s[:, None]) % n
        a3, a1 = w3[ix], w1[ix]
        turn = np.abs(a3 - np.roll(a3, 1, axis=1)) + np.abs(a1 - np.roll(a1, 1, axis=1))
        r = a3 * x + a1 * u + (1 - a3 - a1) * rf - turn * cost_bps / 1e4
        lr = np.log1p(np.maximum(r, -0.9999))
        G.append(lr.mean(axis=1) * 252)
        V.append(r.std(axis=1, ddof=1) * np.sqrt(252))
        M.append(np.add.reduceat(lr, starts, axis=1))
    return np.concatenate(G), np.concatenate(V), np.vstack(M)


def jensen_penalty(u: np.ndarray, exposure: np.ndarray) -> float:
    """Aproximación de segundo orden a lo que pierde, en g anual, una exposición L_t que
    varía sin relación con el mercado frente a su media fija: −½·σ²·Var(L_t)."""
    return float(-0.5 * np.var(u) * 252 * np.var(exposure))


def perm_pvalue(null: np.ndarray, obs: float) -> float:
    """p de aleatorización: (1 + #{nulo ≥ observado}) / (1 + nº de desplazamientos)."""
    return float((1 + np.sum(null >= obs - 1e-12)) / (1 + len(null)))


def boot_delta(a: np.ndarray, b: np.ndarray, stat, idx: np.ndarray) -> tuple[float, float, float, float]:
    """Δ = stat(a) − stat(b) con remuestreos comunes de filas (meses) `idx` (B × n).

    `stat` recibe una matriz (una serie por fila) y devuelve un valor por fila.
    Devuelve (Δ, percentil 5, percentil 95, p unilateral de H0: Δ ≤ 0), con la
    distribución bootstrap recentrada en cero: p = P*(Δ* − Δ ≥ Δ)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = float(stat(a[None, :])[0] - stat(b[None, :])[0])
    draws = stat(a[idx]) - stat(b[idx])
    lo, hi = np.percentile(draws, [5, 95])
    return obs, float(lo), float(hi), float(np.mean(draws - obs >= obs))


# ───────────────────────────── estrategias ─────────────────────────────

def half_kelly_ex_ante(d: dict) -> pd.Series:
    """Fracción en el 3x (0–1) de ½ Kelly con momentos de toda la historia previa,
    decidida en la primera fecha con prima y cada fin de año. La prima μ̂ − f̂ es
    la compartida (kelly.premium, datos hasta la sesión anterior)."""
    u = d["u"]
    mu = kelly.premium(u, d["f"], r12.MIN_OBS)
    var = u.expanding(r12.MIN_OBS).var() * 252
    L = KELLY * (mu - d["spread"]) / var
    me = month_ends(u.index)
    dates = me[me.month == 12].union([L.first_valid_index()])
    return (to_daily(L.reindex(dates), u.index) / 3).clip(0, 1)


def r13_weights(d: dict, E: dict) -> dict:
    """Pesos objetivo (w3 en el fondo x, w1 en el 1x, fondo x) de las reglas de R13,
    deducidos de su exposición según r13.strategies (se comprueban en build)."""
    u = d["u"]
    one, zero = pd.Series(1.0, index=u.index), pd.Series(0.0, index=u.index)
    x2 = r12.letf(d, 2.0)
    WT = {"1x comprar y mantener": (zero, one, d["x3"]), "3x comprar y mantener": (one, zero, d["x3"])}
    for k, e in E.items():
        if k in WT:
            continue
        if k == "LRS 1x (original)":
            WT[k] = (zero, e, d["x3"])
        elif k == "LRS 2x (original)":
            WT[k] = (e / 2, zero, x2)
        elif k == "LRS rotación 3x/1x":
            sig = (e - 1) / 2
            WT[k] = (sig, 1 - sig, d["x3"])
        else:
            WT[k] = (e / 3, zero, d["x3"])
    return WT


def build(d: dict, var_fc: pd.Series) -> tuple[dict, dict, dict, dict, dict]:
    """Rentabilidades diarias R, exposición E al índice y nocional N en fondos
    apalancados (ambas objetivo, antes del retraso de ejecución), el tipo y los
    pesos objetivo WT = (w3, w1, fondo) con los que se construye cada regla."""
    R, E = r13.strategies(d, var_fc, lag=LAG)
    d["_r13_keys"] = list(R)
    # sensibilidad: la LRS 3x con la señal de precio (en el SPA solo en el S&P 500, vía r13.family)
    pk = px_key(d)
    Rp, Ep = r13.price_variants(d, var_fc, lag=LAG)
    R[pk], E[pk] = Rp[pk], Ep[pk]
    WT = r13_weights(d, E)
    u, rf = d["u"], d["rf"]
    N = {k: E[k].copy() for k in R}
    N["1x comprar y mantener"] = E["1x comprar y mantener"] * 0
    N["LRS 1x (original)"] = E["LRS 1x (original)"] * 0
    N["LRS rotación 3x/1x"] = 3 * (E["LRS rotación 3x/1x"] - 1) / 2   # 3·señal
    zero = pd.Series(0.0, index=u.index)

    # Faber (2007) calcula la media de 10 meses sobre la serie de rentabilidad total
    fab = faber_signal(r13.signal_index(d, r13.SIGNAL))
    R["Faber 10 meses 1x"] = r13.run_weights(d, zero, w1=fab, lag=LAG)
    E["Faber 10 meses 1x"], N["Faber 10 meses 1x"] = fab, zero.where(fab.notna())
    WT["Faber 10 meses 1x"] = (zero, fab, d["x3"])
    R["Faber 10 meses 3x"] = r13.run_weights(d, fab, lag=LAG)
    E["Faber 10 meses 3x"] = N["Faber 10 meses 3x"] = 3 * fab
    WT["Faber 10 meses 3x"] = (fab, zero, d["x3"])
    ts = tsmom_signal(u, rf)
    R["TSMOM 12 meses 3x"] = r13.run_weights(d, ts, lag=LAG)
    E["TSMOM 12 meses 3x"] = N["TSMOM 12 meses 3x"] = 3 * ts
    WT["TSMOM 12 meses 3x"] = (ts, zero, d["x3"])
    w_mm, c_mm = mm_exposure(u, rf, real_time=True)
    R["Moreira–Muir (c en tiempo real)"] = r13.run_weights(d, w_mm / 3, lag=LAG)
    E["Moreira–Muir (c en tiempo real)"] = N["Moreira–Muir (c en tiempo real)"] = w_mm
    WT["Moreira–Muir (c en tiempo real)"] = (w_mm / 3, zero, d["x3"])
    kx = half_kelly_ex_ante(d)
    R[KELLY_EX_ANTE] = r13.run_weights(d, kx, lag=LAG)
    E[KELLY_EX_ANTE] = N[KELLY_EX_ANTE] = 3 * kx
    WT[KELLY_EX_ANTE] = (kx, zero, d["x3"])
    kind = {k: "dinámica" for k in R}
    for k in ("1x comprar y mantener", "3x comprar y mantener", KELLY_EX_ANTE):
        kind[k] = "estática"
    d["_c_mm"] = c_mm
    # los pesos reconstruidos reproducen exactamente cada regla (la nula se apoya en ellos)
    for k in R:
        if "comprar y mantener" in k:
            continue
        w3, w1, x = WT[k]
        rr = r13.run_weights(d, w3, w1=w1, lag=LAG, x=x)
        ok = R[k].notna()
        assert np.allclose(rr[ok], R[k][ok], atol=1e-12, equal_nan=True), f"pesos de {k}"
    return R, E, N, kind, WT


def eval_window(d: dict, R: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Del primer mes completo en que todas las reglas tienen datos al último mes cerrado."""
    first = max(r.first_valid_index() for r in R.values())
    if d["eval_from"] is not None:
        first = max(first, d["eval_from"])
    idx = d["u"].index
    per = idx.to_period("M")
    p0 = first.to_period("M")
    if idx[idx >= first][0] != idx[per == p0][0]:
        p0 = p0 + 1
    start = idx[per == p0][0]
    return start, month_ends(idx)[-1]


def add_static(d: dict, R: dict, E: dict, N: dict, kind: dict, WT: dict, sl: slice,
               lag: int = LAG) -> dict:
    """Referencias estáticas emparejadas con las dos estrategias focales."""
    info = {}
    zero = pd.Series(0.0, index=d["u"].index)
    for k, short in zip(FOCAL, ("LRS 3x", "LRS-Kelly")):
        L_e = float(E[k].shift(lag).loc[sl].mean())
        L_v = match_vol(d, R[k].loc[sl].std() * np.sqrt(252), sl)
        for name, L, kd in ((f"L constante = exposición media de {short}", L_e, "estática"),
                            (f"L constante = volatilidad de {short} (cota)", L_v, "estática (cota)")):
            R[name] = const_leverage(d, L)
            E[name] = N[name] = pd.Series(L, index=d["u"].index)
            WT[name] = (pd.Series(min(L, 3.0) / 3, index=d["u"].index), zero, d["x3"])
            kind[name] = kd
            info[name] = L
    # Moreira–Muir con la c de muestra completa: mira el futuro, sólo como cota
    w, c = mm_exposure(d["u"], d["rf"], real_time=False, sample=sl)
    R[MM_FULL] = r13.run_weights(d, w / 3, lag=lag)
    E[MM_FULL] = N[MM_FULL] = w
    WT[MM_FULL] = (w / 3, zero, d["x3"])
    kind[MM_FULL] = "dinámica (cota, mira el futuro)"
    info["c_muestra_completa"] = float(c.iloc[0])
    return info


# ───────────────────────────── evaluación ─────────────────────────────

class Window:
    """Datos de un mercado recortados a la ventana de evaluación, en numpy."""

    def __init__(self, d: dict, R: dict, E: dict, N: dict, sl: slice, lag: int = LAG):
        self.sl = sl
        self.idx = d["u"].loc[sl].index
        self.starts = month_starts_pos(self.idx)
        self.rf = d["rf"].loc[sl].fillna(0).values
        self.u = d["u"].loc[sl].values
        self.n_days = len(self.idx)
        self.lrf_m = monthly_log(self.rf, self.starts)
        self.r = {k: v.loc[sl].values for k, v in R.items()}
        self.n = {k: v.shift(lag).loc[sl].fillna(0).values for k, v in N.items()}
        self.e = {k: float(v.shift(lag).loc[sl].mean()) for k, v in E.items()}
        for k, v in self.r.items():
            assert not np.isnan(v).any(), f"{k}: huecos en la ventana"

    def lx(self, r: np.ndarray) -> np.ndarray:
        return monthly_log(r, self.starts) - self.lrf_m

    def ce(self, k: str, gamma: float) -> float:
        return crra_ce(self.lx(self.r[k]), gamma)

    def g(self, r: np.ndarray) -> float:
        return float(np.mean(np.log1p(np.maximum(r, -0.9999))) * 252)


def frontier(d: dict, W: Window, efficient: bool = False) -> pd.DataFrame:
    rows = []
    for L in L_GRID:
        r = const_leverage(d, L, efficient).loc[W.sl].values
        lx = W.lx(r)
        rows.append({"L": L, "g": W.g(r), "vol": np.std(r, ddof=1) * np.sqrt(252),
                     **{f"EC_γ{g:g}": crra_ce(lx, g) for g in GAMMA_FINE}})
    return pd.DataFrame(rows).set_index("L")


def best_constant(fr: pd.DataFrame, fe: pd.DataFrame, gamma: float) -> tuple[float, bool, float]:
    """Mejor L constante a posteriori para γ entre las dos fronteras: (L, eficiente, EC)."""
    col = f"EC_γ{gamma:g}"
    a, b = fr[col], fe[col]
    if b.max() > a.max():
        return float(b.idxmax()), True, float(b.max())
    return float(a.idxmax()), False, float(a.max())


FORM_LRS, FORM_CHEAP = "3x + letras", "forma barata"


def best_constants(fr: pd.DataFrame, fe: pd.DataFrame, gamma: float) -> dict:
    """Mejor L constante a posteriori para γ en cada frontera por separado: con los
    instrumentos de la LRS (3x + letras) y con la forma más barata (1x + letras o 1x + 3x).
    {forma: (L, EC)}."""
    col = f"EC_γ{gamma:g}"
    return {FORM_LRS: (float(fr[col].idxmax()), float(fr[col].max())),
            FORM_CHEAP: (float(fe[col].idxmax()), float(fe[col].max()))}


def block_ce(lx: np.ndarray, months: int, gamma: float) -> tuple[float, float, float, int]:
    """Equivalente cierto anual con sumas de log-excesos mensuales en bloques no solapados de
    `months` meses, para cada desfase de inicio posible (0 … months − 1): media, mínimo y
    máximo del EC entre desfases y el menor número de bloques. Con months = 1 es crra_ce."""
    lx = np.asarray(lx, float)
    vals, nb = [], []
    for o in range(months):
        n = (len(lx) - o) // months
        if n < 2:
            continue
        b = lx[o:o + n * months].reshape(n, months).sum(axis=1)
        vals.append(crra_ce(b, gamma, periods=12 / months))
        nb.append(n)
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals)), int(min(nb))


def ce_table(W: Window, names: list[str], kind: dict) -> pd.DataFrame:
    """Tabla principal (fracciones): exposición, CAGR, vol, caída máxima y EC por γ."""
    rows = {}
    for k in names:
        s = pd.Series(W.r[k], index=W.idx)
        rows[k] = {"tipo": kind[k], "exposición media": W.e[k], "CAGR": metrics.cagr(s),
                   "vol": metrics.vol(s), "caída máx.": metrics.max_dd(s),
                   **{f"EC γ={g}": W.ce(k, g) for g in GAMMAS}}
    df = pd.DataFrame(rows).T
    num = [c for c in df.columns if c != "tipo"]
    df[num] = df[num].astype(float)
    return df


def as_pct(df: pd.DataFrame, cols, unit: str = "%") -> pd.DataFrame:
    """Copia con `cols` × 100 y la unidad en el nombre (para mostrar con 2 decimales)."""
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype(float) * 100
    return out.rename(columns={c: f"{c} ({unit})" for c in cols})


def fmt_p(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas de p-valores con tres decimales (report.fmt redondea todo con los mismos
    dígitos y un p de 0,003 saldría como 0); vacías donde no aplican."""
    out = df.copy()
    for c in out.columns:
        if str(c).startswith("p ") or str(c) == "p":
            out[c] = df[c].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    return out


def static_names(kind: dict) -> list[str]:
    return [k for k, v in kind.items() if v.startswith("estática")]


def best_static(W: Window, kind: dict, gamma: float) -> str:
    st = static_names(kind)
    return st[int(np.argmax([W.ce(k, gamma) for k in st]))]


def boot_indices(n_months: int, rng) -> np.ndarray:
    """B remuestreos estacionarios de meses (bloque medio BLOCK_M), comunes a todas las
    comparaciones de un mercado."""
    return np.array([stats.stationary_indices(n_months, BLOCK_M, rng) for _ in range(B)])


def null_paths(W: Window, wt: tuple, lag: int = LAG) -> dict:
    """Nula sin habilidad de una regla en la ventana W (véase shifted_paths)."""
    w3, w1, x = wt
    a3, a1 = w3.shift(lag).loc[W.sl].values, w1.shift(lag).loc[W.sl].values
    assert not (np.isnan(a3).any() or np.isnan(a1).any()), "pesos sin definir en la ventana"
    g, v, m = shifted_paths(a3, a1, x.loc[W.sl].values, W.u, W.rf, W.starts, shift_grid(W.n_days))
    return {"g": g, "vol": v, "lx": m - W.lrf_m}


def timing_tables(d: dict, W: Window, fr: pd.DataFrame, E: dict, WT: dict, names: list[str],
                  bidx: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Crecimiento de cada regla frente al apalancamiento constante emparejado (tabla A,
    con bootstrap) y frente a la nula sin habilidad de su misma exposición (tabla B).

    A: Δg frente al L constante (3x + letras) de igual exposición media, frente a la
    forma más barata de esa exposición y frente al L de igual volatilidad realizada;
    p unilateral bootstrap de «Δg ≤ 0» y banda del 90 %.
    B: la misma regla con su exposición desplazada en el tiempo. «Sincronización» es
    Δg − media(Δg nulo); como el desplazamiento conserva la exposición media, frente a
    igual exposición equivale a g − media(g nulo). Frente a igual volatilidad, cada
    desplazamiento se compara con el L constante de SU volatilidad (interpolado en la
    frontera)."""
    A, Bt, nulls = {}, {}, {}
    scale = 252 * len(W.starts) / W.n_days     # así 12·media mensual reproduce el Δg por sesión
    gstat = lambda X: scale / 12 * crra_ce_rows(X, 1)
    for k in names:
        r, e = W.r[k], W.e[k]
        g = W.g(r)
        vol = np.std(r, ddof=1) * np.sqrt(252)
        L_v = match_vol(d, vol, W.sl)
        r_e = const_leverage(d, e).loc[W.sl].values
        r_b = const_leverage(d, e, efficient=True).loc[W.sl].values
        r_v = const_leverage(d, L_v).loc[W.sl].values
        de, lo_e, hi_e, p_e = boot_delta(W.lx(r), W.lx(r_e), gstat, bidx)
        dv, lo_v, hi_v, p_v = boot_delta(W.lx(r), W.lx(r_v), gstat, bidx)
        A[k] = {"exposición media": e, "g": g, "Δg misma exposición": de, "IC 90% inf.": lo_e,
                "IC 90% sup.": hi_e, "p (Δg ≤ 0)": p_e,
                "Δg misma exposición, forma barata": g - W.g(r_b),
                "L misma vol": L_v, "Δg misma vol": dv, "p (Δg misma vol ≤ 0)": p_v}
        nl = nulls[k] = null_paths(W, WT[k])
        dn_e = nl["g"] - W.g(r_e)
        L_vs = np.interp(nl["vol"], fr["vol"].values, fr.index.values)
        dn_v = nl["g"] - np.interp(L_vs, fr.index.values, fr.g.values)
        ex = E[k].shift(LAG).loc[W.sl].values
        Bt[k] = {"Δg misma exposición": de, "Δg nulo: media": dn_e.mean(),
                 "Δg nulo: 5%": np.percentile(dn_e, 5), "Δg nulo: 95%": np.percentile(dn_e, 95),
                 "−½σ²Var(L)": jensen_penalty(W.u, ex),
                 "sincronización": de - dn_e.mean(), "p nula": perm_pvalue(dn_e, de),
                 "Δg misma vol": dv, "Δg nulo misma vol: media": dn_v.mean(),
                 "sincronización (vol)": dv - dn_v.mean(), "p nula (vol)": perm_pvalue(dn_v, dv),
                 "desplazamientos": len(dn_e)}
    return pd.DataFrame(A).T.astype(float), pd.DataFrame(Bt).T.astype(float), nulls


def ec_matched(W: Window, nulls: dict, spa: dict, bidx: np.ndarray, fr: pd.DataFrame | None = None,
               fe: pd.DataFrame | None = None) -> pd.DataFrame:
    """EC de las dos estrategias focales frente a referencias de SU mismo riesgo (L constante
    de igual exposición media y de igual volatilidad), con bootstrap, frente a la nula sin
    habilidad y, para separar la pregunta del nivel de apalancamiento, frente a la mejor
    referencia estática del conjunto discreto y (con las fronteras fr y fe) frente al mejor L
    constante a posteriori con cada forma."""
    rows = []
    for k in FOCAL:
        s = SHORT[k]
        ke, kv = f"L constante = exposición media de {s}", f"L constante = volatilidad de {s} (cota)"
        for g in GAMMAS:
            st = lambda X, g=g: crra_ce_rows(X, g)
            de, lo, hi, pe = boot_delta(W.lx(W.r[k]), W.lx(W.r[ke]), st, bidx)
            dv, _, _, pv = boot_delta(W.lx(W.r[k]), W.lx(W.r[kv]), st, bidx)
            ce_null = crra_ce_rows(nulls[k]["lx"], g)
            rows.append({"estrategia": s, "γ": g, "ΔEC misma exposición": de, "IC 90% inf.": lo,
                         "IC 90% sup.": hi, "p (≤ 0)": pe, "ΔEC misma vol": dv, "p (misma vol ≤ 0)": pv,
                         "ΔEC nulo: media": ce_null.mean() - W.ce(ke, g),
                         "p nula": perm_pvalue(ce_null, W.ce(k, g)),
                         "ΔEC mejor estático": spa[g][f"ΔEC {s}"]})
            if fr is not None:
                bc = best_constants(fr, fe, g)
                rows[-1].update({f"ΔEC L óptimo ({f})": W.ce(k, g) - bc[f][1] for f in (FORM_LRS, FORM_CHEAP)})
    return pd.DataFrame(rows)


def spa_block(W: Window, family: dict, bench_lx: np.ndarray, gamma: float, rng) -> dict:
    names = list(family)
    ub = crra_utility(bench_lx, gamma)
    D = np.column_stack([crra_utility(W.lx(family[k]), gamma) - ub for k in names])
    assert not np.isnan(D).any()
    spa = stats.spa_test(D, BLOCK_M, B, rng)
    ce_b = crra_ce(bench_lx, gamma)
    ces = np.array([crra_ce(W.lx(family[k]), gamma) for k in names])
    best = int(np.argmax(ces))
    out = {"reglas": len(names), "mejor regla": names[best], "ΔEC mejor": ces[best] - ce_b,
           "p_SPA_c": spa["p_SPA_c"], "p_SPA_l": spa["p_SPA_l"]}
    for k in FOCAL:
        j = names.index(k)
        out[f"ΔEC {SHORT[k]}"] = ces[j] - ce_b
        out[f"p individual {SHORT[k]}"] = stats.spa_test(D[:, [j]], BLOCK_M, B, rng)["p_SPA_c"]
    return out


def thresholds(d: dict, W: Window, kind: dict, names: list[str], fr: pd.DataFrame) -> pd.DataFrame:
    """Sobrecoste de financiación (pb/año sobre el nocional apalancado) que iguala el EC de
    cada regla al de la referencia. Frente a la mejor referencia estática de cada γ, en dos
    versiones: «pagan todos» (la referencia, que usa el mismo fondo 3x, paga el mismo
    sobrecoste sobre su nocional: la relevante) y «solo la estrategia» (colchón frente a una
    desventaja que solo sufriera ella). Con γ = 1, además frente al mejor L constante con 3x +
    letras (las dos versiones) y frente al 1x, que no paga swap."""
    rows = {}
    b1 = "1x comprar y mantener"
    L_opt = best_constants(fr, fr, 1)[FORM_LRS][0]
    r_o = const_leverage(d, L_opt).loc[W.sl].values
    n_o = np.full(len(r_o), L_opt)   # nocional en fondos apalancados de un L constante (como N en add_static)
    for k in names:
        row = {}
        for g in GAMMAS:
            b = best_static(W, kind, g)
            for lab, both in (("pagan todos", True), ("solo la estrategia", False)):
                row[f"γ={g}, {lab}"] = break_even(W.r[k], W.n[k], W.r[b], W.n[b], W.lrf_m, W.starts, g,
                                                  both=both)
        for lab, both in (("pagan todos", True), ("solo la estrategia", False)):
            row[f"γ=1 frente al L óptimo 3x + letras, {lab}"] = break_even(
                W.r[k], W.n[k], r_o, n_o, W.lrf_m, W.starts, 1, both=both)
        row["γ=1, frente al 1x"] = break_even(W.r[k], W.n[k], W.r[b1], W.n[b1], W.lrf_m, W.starts, 1,
                                              both=False)
        row["nocional medio"] = float(W.n[k].mean())
        row["nocional medio de la referencia (γ=1)"] = float(W.n[best_static(W, kind, 1)].mean())
        rows[k] = row
    return pd.DataFrame(rows).T


def fmt_bps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    for c in out.columns:
        if c.startswith("nocional medio"):
            out[c] = df[c].map(lambda v: num_es(v, 2))
        else:
            out[c] = df[c].map(lambda v: "no se anula" if v == np.inf else "sin ventaja"
                               if v == -np.inf else num_es(v, 0))
    return out


def rebalance_drag(W: Window, L: float) -> float:
    """Coste anual (5 pb por unidad) del reequilibrio diario que se supone gratis
    en una mezcla fija L/3 en el 3x y el resto en letras."""
    w = L / 3
    x3 = W.r["3x comprar y mantener"]
    drift = w * (1 + x3) / (w * (1 + x3) + (1 - w) * (1 + W.rf))
    return float(np.mean(np.abs(drift - w)) * r13.COST_BPS / 1e4 * 252)


# ───────────────────────────── figuras ─────────────────────────────

STYLE = {  # color fijo por estrategia (paleta validada) + marcador como segunda codificación
    "LRS 3x (original)": ("#2a78d6", "o", "-"),
    # la señal de precio, rombo blanco de borde azul: con un círculo hueco se confundía con la
    # media de la nula y, en el Nasdaq-100, quedaba tapada por la LRS 3x
    PX: ("#2a78d6", "d", "--"),
    PX_APPROX: ("#2a78d6", "d", "--"),
    "LRS-Kelly": ("#e34948", "s", "-"),
    "Faber 10 meses 1x": ("#1baf7a", "v", "--"),
    "Faber 10 meses 3x": ("#eb6834", "^", "--"),
    "TSMOM 12 meses 3x": ("#4a3aa7", "D", "-."),
    "Moreira–Muir (c en tiempo real)": ("#008300", "P", ":"),
    MM_FULL: ("#008300", "X", ":"),
    KELLY_EX_ANTE: ("#e87ba4", "h", "--"),
    "1x comprar y mantener": ("#222222", "*", "-"),
    "3x comprar y mantener": ("#8a8a8a", "*", "-"),
}
SHORT = {"LRS 3x (original)": "LRS 3x", "LRS-Kelly": "LRS-Kelly", PX: "LRS 3x, señal de precio",
         PX_APPROX: "LRS 3x, señal de precio aprox.", "Faber 10 meses 1x": "Faber 1x",
         "Faber 10 meses 3x": "Faber 3x", "TSMOM 12 meses 3x": "Momento de 12 meses, 3x",
         "Moreira–Muir (c en tiempo real)": "Moreira–Muir, c en tiempo real",
         MM_FULL: "Moreira–Muir, c de muestra completa",
         KELLY_EX_ANTE: "Medio Kelly ex ante", "1x comprar y mantener": "1x",
         "3x comprar y mantener": "3x"}
MKT = {"S&P 500": "S&P 500", "Nasdaq-100": "Nasdaq-100", "Semis": "Semiconductores"}   # nombres en figuras


PRICE_FACE = "white"   # cara de los marcadores de la señal de precio: tapan lo que tienen debajo
NULL_MEAN = dict(marker="_", ms=20, mew=3, color="0.15", ls="none")   # media de la nula: trazo oscuro


def fig_frontier(res: dict, rep: Report):
    n = len(res)
    fig, axes = plt.subplots(n, 1, figsize=(FIG_W, 4.4 * n + 2.2))
    axes = np.atleast_1d(axes)
    for i, (ax, (name, x)) in enumerate(zip(axes, res.items())):
        fr, fe, W = x["frontera"], x["frontera_eficiente"], x["W"]
        ax.plot(fr.index, fr.g * 100, color="k", lw=2, label="L constante: 3x + letras")
        ax.plot(fe.index, fe.g * 100, color="0.55", lw=2, ls="--",
                label="L constante: 1x + letras / 1x + 3x")
        Lk = fr.g.idxmax()
        ax.plot([Lk], [fr.g.max() * 100], marker="|", ms=18, color="k", mew=2)
        for k in FOCAL:   # nula sin habilidad: misma exposición desplazada en el tiempo
            c = STYLE[k][0]
            gn = x["nulls"][k]["g"] * 100
            lo, hi = np.percentile(gn, [5, 95])
            ax.plot([W.e[k]] * 2, [lo, hi], color=c, lw=7, alpha=0.3, solid_capstyle="butt", zorder=3,
                    label="nula sin habilidad (90%)" if i == 0 and k == FOCAL[0] else None)
            # la media, con el trazo horizontal oscuro de sincronizacion.png: un círculo hueco se
            # confundía con el de la señal de precio
            ax.plot([W.e[k]], [gn.mean()], zorder=4, **NULL_MEAN,
                    label="media de la nula" if i == 0 and k == FOCAL[0] else None)
        for k, (c, m, _) in STYLE.items():
            if k not in W.r:
                continue
            px = k in (PX, PX_APPROX)
            face = PRICE_FACE if px else "none" if k == MM_FULL else c
            lab = SHORT[k] if i == 0 else None     # el primer panel (S&P 500) lleva la leyenda
            # la señal de precio, encima de todos: en el Nasdaq-100 coincide con la LRS 3x y la Faber 3x
            ax.scatter(W.e[k], W.g(W.r[k]) * 100, s=190 if px else 140, marker=m, facecolor=face,
                       edgecolor=c, lw=2.2 if px else 2, zorder=6 if px else 5, label=lab)
        ax.axhline(0, color="0.7", lw=0.8)
        lo = min(fr.g.min(), min(W.g(W.r[k]) for k in STYLE if k in W.r),
                 min(np.percentile(x["nulls"][k]["g"], 5) for k in FOCAL)) * 100
        # en dos líneas: en una, el título de semiconductores no cabía a lo ancho
        ax.set(title=f"{MKT[name]} ({W.idx[0]:%Y-%m} a {W.idx[-1]:%Y-%m})\n"
                     f"máximo de la curva negra en L = {num_es(Lk, 2)}",
               xlim=(-0.05, 4.05),
               xlabel="Exposición media al índice (L)", ylabel="Crecimiento log (% anual)",
               ylim=(max(lo - 2, -40), None))
        ax.grid(color="0.9", lw=0.6)
        comma_axis(ax)
    legend_band(fig, *axes[0].get_legend_handles_labels(), where="bottom", ncol=2, fontsize=11)
    fig.suptitle("Frontera del apalancamiento constante\ny cada estrategia en su exposición media")
    rep.fig(fig, "frontera", "Crecimiento logarítmico anual de un apalancamiento constante L entre 0 y 4 "
            "(línea negra: con el 3x y letras, los instrumentos de la LRS; gris discontinua: la forma más "
            "barata, 1x con letras o con el 3x) y de cada estrategia situada en su exposición media. Un punto "
            "por encima de la curva negra crece más que mantener fija esa misma exposición. Un punto por "
            "debajo NO indica que la sincronización reste: por la concavidad de g(L), una exposición que varía "
            "sin relación con el mercado ya crece menos que su media fija. La barra vertical translúcida de la "
            "LRS 3x (azul) y de la LRS-Kelly (roja) es el 90% central de esa misma regla con su trayectoria de "
            "exposición desplazada en el tiempo (sin habilidad), y el trazo horizontal oscuro su media: la "
            "sincronización se mide desde ese trazo, no desde la curva. El trazo vertical negro marca el máximo "
            "de la curva negra. El rombo blanco de borde azul, dibujado encima de los demás marcadores, es la "
            "LRS 3x con la señal de precio (aproximado fuera del S&P 500).")


def fig_timing(res: dict, rep: Report):
    """Δg observado frente a la nula sin habilidad para las reglas dinámicas con estilo propio."""
    base = [k for k in STYLE if k not in ("1x comprar y mantener", "3x comprar y mantener", KELLY_EX_ANTE)]
    fig, axes = plt.subplots(len(res), 1, figsize=(FIG_W, 4.2 * len(res) + 0.4))
    axes = np.atleast_1d(axes)
    for i, (ax, (name, x)) in enumerate(zip(axes, res.items())):
        W = x["W"]
        tb = x["tb"]
        names = [k for k in base if k in x["nulls"]]
        for j, k in enumerate(names[::-1]):
            c, m, _ = STYLE[k]
            dn = x["nulls"][k]["g"] - W.g(const_leverage(x["d"], W.e[k]).loc[W.sl].values)
            lo, hi = np.percentile(dn * 100, [5, 95])
            ax.plot([lo, hi], [j, j], color="0.6", lw=9, alpha=0.5, solid_capstyle="butt",
                    label="nula sin habilidad (90%)" if i == 0 and j == 0 else None)
            ax.plot([dn.mean() * 100], [j], marker="|", ms=22, mew=3, color="0.3",
                    label="media de la nula" if i == 0 and j == 0 else None)
            px = k in (PX, PX_APPROX)
            face = PRICE_FACE if px else "none" if k == MM_FULL else c
            ax.scatter(tb.loc[k, "Δg misma exposición"] * 100, j, s=190 if px else 150, marker=m,
                       facecolor=face, edgecolor=c, lw=2.2 if px else 2, zorder=5)
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([SHORT[k] for k in names[::-1]])
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set(title=f"{MKT[name]} ({W.idx[0]:%Y-%m} a {W.idx[-1]:%Y-%m})",
               xlabel="Δg frente al L constante (puntos al año)")
        ax.grid(axis="x", color="0.9", lw=0.6)
        comma_axis(ax)
    # la leyenda en su propia franja, sin tapar ninguna barra
    legend_band(fig, *axes[0].get_legend_handles_labels(), ncol=2)
    fig.suptitle("Sincronización: cada regla frente a sí misma sin habilidad")
    rep.fig(fig, "sincronizacion", "Diferencia de crecimiento logarítmico anual entre cada regla dinámica "
            "(marcador de color) y el apalancamiento constante de su misma exposición media, con el 3x y "
            "letras. La barra gris es el 90% central de esa diferencia cuando la trayectoria de exposición "
            "de la regla se desplaza en el tiempo (al menos dos años, con los mismos fondos, letras y costes): "
            "lo que daría la misma regla sin ninguna relación con el mercado. El trazo oscuro es su media. La "
            "sincronización es la distancia entre el marcador y ese trazo, no entre el marcador y el cero. "
            "Moreira–Muir con la c de muestra completa (marcador verde hueco) mira el futuro; el rombo blanco "
            "de borde azul es la LRS 3x con la señal de precio.")


def fig_ce_gamma(res: dict, rep: Report):
    fig, axes = plt.subplots(len(res), 1, figsize=(FIG_W, 4.2 * len(res) + 1.3))
    axes = np.atleast_1d(axes)
    for i, (ax, (name, x)) in enumerate(zip(axes, res.items())):
        W, kind, fr, fe = x["W"], x["tipo"], x["frontera"], x["frontera_eficiente"]
        env = np.array([max(crra_ce(W.lx(W.r[k]), g) for k in static_names(kind)) for g in GAMMA_FINE])
        opt = np.array([best_constant(fr, fe, g)[2] for g in GAMMA_FINE])
        ax.plot(GAMMA_FINE, opt * 100, color="0.55", lw=2, ls=(0, (1, 1.5)),
                label="mejor L constante a posteriori (cota)" if i == 0 else None)
        ax.plot(GAMMA_FINE, env * 100, color="k", lw=4, alpha=0.8,
                label="mejor referencia estática del conjunto" if i == 0 else None)
        vals = [env]
        mk = np.isin(GAMMA_FINE, GAMMAS)
        for k in COMPARE:
            c, m, ls = STYLE[k]
            y = np.array([crra_ce(W.lx(W.r[k]), g) for g in GAMMA_FINE])
            vals.append(y)
            lw = 3 if k in FOCAL else 1.8
            ax.plot(GAMMA_FINE, y * 100, color=c, lw=lw, ls=ls, label=SHORT[k] if i == 0 else None)
            ax.plot(GAMMA_FINE[mk], y[mk] * 100, color=c, marker=m, ls="none", ms=9)
        hi = max(v.max() for v in vals) * 100
        ax.set(title=MKT[name], xlabel="Aversión relativa al riesgo γ", ylabel="Equivalente cierto (% anual)",
               xticks=GAMMAS, ylim=(-25, hi + 1.5))
        ax.axhline(0, color="0.7", lw=0.8)
        ax.grid(color="0.9", lw=0.6)
        comma_axis(ax)
    # la leyenda en su propia franja: dentro del panel tapaba curvas
    legend_band(fig, *axes[0].get_legend_handles_labels(), ncol=2, fontsize=11)
    fig.suptitle("Equivalente cierto CRRA en exceso de las letras\nsegún la aversión al riesgo")
    rep.fig(fig, "equivalente_cierto", "Equivalente cierto anual (MPPM con rentabilidades mensuales, en exceso "
            "de las letras) de las dos estrategias estudiadas (líneas gruesas) y de las reglas dinámicas de "
            "referencia frente a γ. La línea negra gruesa es la mejor referencia estática del conjunto de cada "
            "γ (1x, 3x, L constante emparejado o ½ Kelly ex ante); la gris punteada, el mejor L constante "
            "elegido a posteriori (de 0 a 4, con cualquiera de las dos formas). Los marcadores señalan γ = 1, 2, 3 y 5; las curvas que salen por abajo quedan por "
            "debajo del −25% anual.")


def fig_mm(res: dict, rep: Report):
    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=False)
    for name, x in res.items():
        W, d = x["W"], x["d"]
        c_rt = d["_c_mm"].loc[W.sl].dropna()
        ratio = c_rt / x["info"]["c_muestra_completa"]
        col = {"S&P 500": "#2a78d6", "Nasdaq-100": "#eb6834", "Semis": "#4a3aa7"}[name]
        axes[0].plot(ratio.index, ratio.values, color=col, lw=2, label=MKT[name])
    axes[0].axhline(1, color="k", ls="--", lw=1.2)
    axes[0].set(title="c en tiempo real dividida por la c de muestra completa", ylabel="Cociente",
                ylim=(0, None))
    axes[0].legend(loc="upper right")
    x = res["S&P 500"]
    W = x["W"]
    for k, lab, c, ls in (("Moreira–Muir (c en tiempo real)", "c en tiempo real", "#008300", "-"),
                          (MM_FULL, "c de muestra completa (mira el futuro)", "k", "--")):
        ex = x["E"][k].shift(LAG).loc[W.sl].rolling(252).mean()
        axes[1].plot(ex.index, ex.values, color=c, lw=2, ls=ls, label=lab)
    axes[1].set(title="S&P 500: exposición media de los 12 meses previos", ylabel="Exposición (x)",
                ylim=(0, 3.05))
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.grid(color="0.9", lw=0.6)
        comma_axis(ax)
    rep.fig(fig, "moreira_muir", "Arriba, cuánto se aparta la constante c de Moreira–Muir estimada cada mes "
            "sólo con el pasado de la c calculada con todo el periodo evaluado (1 = iguales), en cada mercado. Abajo, la exposición resultante al S&P 500 (media móvil de 12 meses, "
            "tope 3x): al salir de la Gran Depresión la c en tiempo real apunta a una volatilidad objetivo "
            "muy alta y la cartera va a 3x.")


# ───────────────────────────── informe ─────────────────────────────

POST_DG = "Δg frente a L constante de su exposición en el periodo"
POST_FROM = "2017-01-01"   # primer año completo tras la publicación de la LRS (SSRN, marzo de 2016)
POST_ROWS = ("1x comprar y mantener", "L constante = exposición media de LRS 3x", "LRS 3x (original)",
             "L constante = exposición media de LRS-Kelly", "LRS-Kelly", "Faber 10 meses 3x",
             "TSMOM 12 meses 3x", "Moreira–Muir (c en tiempo real)")
EC_COLS = [f"EC γ={g}" for g in GAMMAS]
THE = {"S&P 500": "el S&P 500", "Nasdaq-100": "el Nasdaq-100", "Semis": "los semiconductores"}


def _pp(v: float, d: int = 2) -> str:
    """Fracción → puntos porcentuales con signo y coma decimal: 0,0123 → «+1,23»."""
    return f"{v * 100:+.{d}f}".replace(".", ",").replace("-", "−")


def _p(v: float) -> str:
    return num_es(v, 3)


def _bps(v: float) -> str:
    """Umbral de sobrecoste para el texto (véase break_even)."""
    if v == np.inf:
        return "no se anula"
    if v == -np.inf:
        return "sin ventaja"
    if v < 0:
        return f"sin ventaja (empataría abaratando {num_es(-v, 0)} pb/año)"
    return f"{num_es(v, 0)} pb/año"


def ec_horizon_table(W: Window, kind: dict, names: list[str]) -> pd.DataFrame:
    """EC anual según el horizonte de evaluación (bloques no solapados de 1, 12, 36 y 60 meses;
    media, mínimo y máximo sobre los desfases de inicio de los bloques) para las reglas de
    `names` y la mejor referencia estática del conjunto de cada γ (elegida con meses)."""
    rows = []
    for k in names + ["mejor referencia estática"]:
        for h in EC_HORIZONS:
            row = {"regla": SHORT.get(k, k), "meses por bloque": h}
            for g in EC_H_GAMMAS:
                kk = best_static(W, kind, g) if k == "mejor referencia estática" else k
                m, lo, hi, n = block_ce(W.lx(W.r[kk]), h, g)
                row.update({f"EC γ={g}": m, f"mín γ={g}": lo, f"máx γ={g}": hi})
            row["bloques"] = n
            rows.append(row)
    return pd.DataFrame(rows)


def ec_horizon_shuffled(W: Window, names: list[str], n_perm: int = N_SHUFFLE,
                        seed: int = SEED_SHUFFLE) -> pd.DataFrame:
    """Referencia del EC por bloques con los meses BARAJADOS al azar (n_perm permutaciones; sin
    ninguna dependencia entre meses, el EC de la población no depende del horizonte): lo que sube
    el EC por bloques con muchos meses y pocos bloques es del estimador, no de la estrategia."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in names:
        lx = np.asarray(W.lx(W.r[k]), float)
        perms = [rng.permutation(lx) for _ in range(n_perm)]
        for h in EC_HORIZONS:
            row = {"regla": SHORT.get(k, k), "meses por bloque": h}
            for g in EC_H_GAMMAS:
                obs = block_ce(lx, h, g)[0]
                sh = np.array([block_ce(x, h, g)[0] for x in perms])
                row.update({f"EC γ={g}": obs, f"barajado γ={g}": float(np.mean(sh)),
                            f"obs − barajado γ={g}": obs - float(np.mean(sh))})
            rows.append(row)
    return pd.DataFrame(rows)


def titular_rows(res: dict, spa: pd.DataFrame) -> list[dict]:
    """Los p de titular de R17 (lista cerrada; los lee R19 desde p_titular.csv): el menor p del
    SPA frente a la mejor referencia estática YA CORREGIDO con Holm sobre sus len(spa)
    comparaciones (3 mercados × 4 γ; el mínimo de 12 p sin corregir no es un p válido), Δg
    frente al L constante de igual exposición y la sincronización frente a la nula sin
    habilidad de la LRS 3x original y la LRS-Kelly en cada mercado, y la sincronización de
    Faber 3x, la rotación 3x/1x, la LRS 2x y Moreira–Muir en el S&P 500. Los p de bootstrap
    que salen 0 se acotan en 1/(B + 1)."""
    lo = 1 / (B + 1)
    rows = [("3 mercados", f"SPA de la familia frente a la mejor referencia estática (mín., Holm sobre {len(spa)})",
             max(float(spa.p_Holm.min()), lo))]
    for k in FOCAL:
        for name, x in res.items():
            rows.append((name, f"Δg {SHORT[k]} frente a L de igual exposición",
                         max(float(x["tA"].loc[k, "p (Δg ≤ 0)"]), lo)))
    for k in FOCAL:
        for name, x in res.items():
            rows.append((name, f"sincronización {SHORT[k]} (nula sin habilidad)", float(x["tb"].loc[k, "p nula"])))
    tb = res["S&P 500"]["tb"]
    for k, lab in (("Faber 10 meses 3x", "Faber 3x"), ("LRS rotación 3x/1x", "rotación 3x/1x"),
                   ("LRS 2x (original)", "LRS 2x"), (MM_FULL, "Moreira–Muir (c de muestra completa)"),
                   ("Moreira–Muir (c en tiempo real)", "Moreira–Muir (c en tiempo real)")):
        rows.append(("S&P 500", f"sincronización {lab}", float(tb.loc[k, "p nula"])))
    return [{"módulo": "R17", "mercado": m, "contraste": c, "p": p} for m, c, p in rows]


def summary_lines(res: dict, spa: pd.DataFrame, pt: pd.DataFrame, ss: pd.DataFrame) -> list[str]:
    L = ["## Resumen", ""]
    for name, x in res.items():
        tA, tB, ecm, W, t, d = x["tA"], x["tb"], x["ecm"], x["W"], x["tabla"], x["d"]
        th = x["th"]
        a, b = FOCAL
        pk = px_key(d)
        ec = ecm.set_index(["estrategia", "γ"])
        ecs = lambda k, col: ", ".join(_pp(ec.loc[(SHORT[k], g), col]) for g in GAMMAS)  # noqa: E731
        L.append(f"- **{name}** ({W.idx[0]:%Y-%m} a {W.idx[-1]:%Y-%m}).")
        L.append(
            f"  - Crecimiento frente al L constante de su misma exposición media (mismos instrumentos): "
            f"LRS 3x {_pp(tA.loc[a, 'Δg misma exposición'])} pp/año (IC 90% "
            f"[{_pp(tA.loc[a, 'IC 90% inf.'])}, {_pp(tA.loc[a, 'IC 90% sup.'])}], p = "
            f"{_p(tA.loc[a, 'p (Δg ≤ 0)'])}); LRS-Kelly {_pp(tA.loc[b, 'Δg misma exposición'])} pp/año "
            f"(p = {_p(tA.loc[b, 'p (Δg ≤ 0)'])}); con la señal de precio"
            f"{' aproximado' if pk == PX_APPROX else ''}, {_pp(tA.loc[pk, 'Δg misma exposición'])} pp/año "
            f"(p = {_p(tA.loc[pk, 'p (Δg ≤ 0)'])}).")
        L.append(
            f"  - Una regla sin habilidad con la misma exposición (desplazada en el tiempo) crecería de media "
            f"{_pp(tB.loc[a, 'Δg nulo: media'])} y {_pp(tB.loc[b, 'Δg nulo: media'])} pp/año frente a ese "
            f"mismo L. Descontado eso, la sincronización suma {_pp(tB.loc[a, 'sincronización'])} pp/año a la "
            f"LRS 3x (p nula = {_p(tB.loc[a, 'p nula'])}) y {_pp(tB.loc[b, 'sincronización'])} a la LRS-Kelly "
            f"(p nula = {_p(tB.loc[b, 'p nula'])}).")
        L.append(
            f"  - Equivalente cierto, γ = 1, 2, 3 y 5 (pp). Frente al L constante de igual exposición: LRS 3x "
            f"{ecs(a, 'ΔEC misma exposición')}; LRS-Kelly {ecs(b, 'ΔEC misma exposición')}. Otra pregunta, el "
            f"nivel de apalancamiento: frente a la mejor referencia estática del conjunto (1x, 3x, L "
            f"emparejados, ½ Kelly ex ante), LRS 3x {ecs(a, 'ΔEC mejor estático')} y LRS-Kelly "
            f"{ecs(b, 'ΔEC mejor estático')}; frente al mejor L constante a posteriori con 3x + letras, "
            f"LRS 3x {ecs(a, f'ΔEC L óptimo ({FORM_LRS})')}; con la forma barata (1x + letras o 1x + 3x), "
            f"LRS 3x {ecs(a, f'ΔEC L óptimo ({FORM_CHEAP})')}.")
        L.append(
            f"  - Volatilidad de la LRS 3x: {pct_es(t.loc[a, 'vol'])} (la del 3x comprar y mantener, "
            f"{pct_es(t.loc['3x comprar y mantener', 'vol'])}; la de un L constante de "
            f"{num_es(tA.loc[a, 'L misma vol'], 2)}). Sobrecoste de financiación que anula su ventaja con "
            f"γ = 1 frente a la mejor referencia estática ({SHORT.get(x['b1'], x['b1'])}): "
            f"{_bps(th.loc[a, 'γ=1, pagan todos'])} si lo pagan todos, {_bps(th.loc[a, 'γ=1, solo la estrategia'])} "
            f"si solo lo paga la estrategia; frente al L óptimo con 3x + letras, "
            f"{_bps(th.loc[a, 'γ=1 frente al L óptimo 3x + letras, pagan todos'])} y "
            f"{_bps(th.loc[a, 'γ=1 frente al L óptimo 3x + letras, solo la estrategia'])}; frente al 1x, "
            f"{_bps(th.loc[a, 'γ=1, frente al 1x'])}.")
    j = int(spa.p_SPA_c.idxmin())
    verdict = ("ninguna regla dinámica supera con significación al 10% a la mejor referencia estática"
               if spa.p_SPA_c.min() >= 0.10 else "hay alguna comparación significativa al 10%")
    c3, cc = f"p_SPA_c frente a L óptimo ({FORM_LRS})", f"p_SPA_c frente a L óptimo ({FORM_CHEAP})"
    L.append(f"- **SPA frente a la mejor referencia estática del conjunto** (mezcla nivel de riesgo y "
             f"sincronización): el menor p_SPA_c de las 12 comparaciones (3 mercados × 4 γ) es "
             f"{_p(spa.p_SPA_c.min())} ({spa.loc[j, 'mercado']}, γ = {spa.loc[j, 'γ']}); con Holm, el menor "
             f"es {num_es(spa.p_Holm.min(), 2)}: {verdict}. Frente al mejor L constante a posteriori, el menor "
             f"es {_p(ss[c3].min())} con 3x + letras y {_p(ss[cc].min())} con la forma barata. El menor p "
             f"individual (sin corregir) de la LRS 3x es {_p(spa['p individual LRS 3x'].min())} y el de la "
             f"LRS-Kelly {_p(spa['p individual LRS-Kelly'].min())}.")
    x = res["S&P 500"]
    t, tB = x["tabla"], x["tb"]
    rt, fs = "Moreira–Muir (c en tiempo real)", MM_FULL
    L.append(f"- **Moreira–Muir** en el S&P 500: con la c en tiempo real, exposición media "
             f"{num_es(t.loc[rt, 'exposición media'], 2)}, caída máxima {pct_es(t.loc[rt, 'caída máx.'])} y EC "
             f"con γ = 3 de {pct_es(t.loc[rt, 'EC γ=3'], 2)}; con la c de muestra completa (que mira el futuro), "
             f"{num_es(t.loc[fs, 'exposición media'], 2)}, {pct_es(t.loc[fs, 'caída máx.'])} y "
             f"{pct_es(t.loc[fs, 'EC γ=3'], 2)}. Con esa c, Δg frente a su L constante "
             f"{_pp(tB.loc[fs, 'Δg misma exposición'])} pp/año, frente a una nula sin habilidad de "
             f"{_pp(tB.loc[fs, 'Δg nulo: media'])} (p nula = {num_es(tB.loc[fs, 'p nula'], 2)}).")
    post = pt.set_index(["mercado", "estrategia"])
    L.append(f"- **Tras la publicación** ({POST_FROM[:4]}–{res['S&P 500']['W'].idx[-1]:%Y}), sincronización "
             "(Δg frente al L constante de su propia exposición media en el periodo, menos la media de la nula "
             "rehecha en el periodo): LRS 3x " +
             ", ".join(f"{_pp(post.loc[(n, FOCAL[0]), 'sincronización'])} pp en {n}" for n in res) +
             "; LRS-Kelly " + ", ".join(f"{_pp(post.loc[(n, FOCAL[1]), 'sincronización'])}" for n in res) +
             " pp. Diez años y un camino por mercado: no permiten distinguir.")
    L.append("")
    return L


def reading(res: dict, spa: pd.DataFrame) -> str:
    """Lectura del informe con las cifras tal como salen de las tablas."""
    a, b = FOCAL
    mk = list(res)
    tb = {n: res[n]["tb"] for n in mk}
    tA = {n: res[n]["tA"] for n in mk}
    ec = {n: res[n]["ecm"].set_index(["estrategia", "γ"]) for n in mk}
    sgn = lambda v, pos, neg: pos if v > 0 else neg  # noqa: E731
    by = lambda k, col, f=_pp: ", ".join(f"{f(tb[n].loc[k, col])} en {THE[n]}" for n in mk)  # noqa: E731
    sync = lambda k: "; ".join(f"{_pp(tb[n].loc[k, 'sincronización'])} en {THE[n]} (p = {_p(tb[n].loc[k, 'p nula'])})"  # noqa: E731
                               for n in mk)
    sp = res["S&P 500"]
    fs = sp["tb"].loc[MM_FULL]
    p1 = ("**Lectura.** Hay que separar tres preguntas que la comparación con el 1x mezcla.\n\n"
          "*¿Tiene información la sincronización?* La nula adecuada no es el L constante de la misma exposición "
          "media, sino la misma regla sin habilidad: su trayectoria de exposición desplazada en el tiempo. Por la "
          "concavidad de g(L), una exposición que salta entre 0 y 3x sin relación con el mercado crece menos que su "
          f"media fija; para la de la LRS 3x, {by(a, 'Δg nulo: media')} pp/año (la aproximación −½·σ²·Var(L), "
          f"{by(a, '−½σ²Var(L)')}, ignora las colas y los costes de rotación). Frente a esa nula, la sincronización "
          f"de la LRS 3x es {sync(a)}; la de la LRS-Kelly, {sync(b)}. Moreira–Muir con la c de muestra completa en "
          f"el S&P 500 da {_pp(fs['Δg misma exposición'])} pp frente a su L constante, frente a "
          f"{_pp(fs['Δg nulo: media'])} sin habilidad (p = {num_es(fs['p nula'], 2)}).")
    ce = lambda n, k, col: ", ".join(_pp(ec[n].loc[(SHORT[k], g), col]) for g in GAMMAS)  # noqa: E731
    pr = lambda n, k, col: [ec[n].loc[(SHORT[k], g), col] for g in GAMMAS]  # noqa: E731
    pvals = [v for n in mk for v in pr(n, a, "p (≤ 0)")]
    p2 = ("*¿Mejora a mantener fijo el mismo riesgo?* Es la pregunta práctica. En crecimiento, la LRS 3x "
          + "; ".join(f"{sgn(tA[n].loc[a, 'Δg misma exposición'], 'supera', 'queda por debajo de')} al L constante "
                      f"de su exposición en {THE[n]} por {_pp(abs(tA[n].loc[a, 'Δg misma exposición']))} pp/año "
                      f"(IC 90% [{_pp(tA[n].loc[a, 'IC 90% inf.'])}, {_pp(tA[n].loc[a, 'IC 90% sup.'])}]; p = "
                      f"{_p(tA[n].loc[a, 'p (Δg ≤ 0)'])})" for n in mk)
          + ". En equivalente cierto frente al L constante de igual exposición, con γ = 1, 2, 3 y 5: "
          + "; ".join(f"{ce(n, a, 'ΔEC misma exposición')} pp en {THE[n]}" for n in mk)
          + f" (p bootstrap entre {_p(min(pvals))} y {_p(max(pvals))}); frente al de igual volatilidad: "
          + "; ".join(f"{ce(n, a, 'ΔEC misma vol')} pp en {THE[n]}" for n in mk) + ".")
    t_sp = sp["tabla"]
    rows3 = []
    for n in mk:
        best = res[n]["bestref"]
        rows3.append(f"en {THE[n]}, frente a la mejor referencia estática ({', '.join(best['estática'])} con γ = "
                     f"1, 2, 3 y 5) {ce(n, a, 'ΔEC mejor estático')}; frente al mejor L constante con 3x + letras "
                     f"(L = {', '.join(num_es(v, 2) for v in best[FORM_LRS])}) {ce(n, a, f'ΔEC L óptimo ({FORM_LRS})')}; "
                     f"con la forma barata (L = {', '.join(num_es(v, 2) for v in best[FORM_CHEAP])}) "
                     f"{ce(n, a, f'ΔEC L óptimo ({FORM_CHEAP})')}")
    p3 = ("*¿Es 3x el nivel adecuado para un γ dado?* Es otra pregunta. La LRS 3x no tiene el riesgo de un 3x: "
          f"en el S&P 500 su volatilidad es {pct_es(t_sp.loc[a, 'vol'])} (la de un L constante de "
          f"{num_es(sp['tA'].loc[a, 'L misma vol'], 2)}), frente a {pct_es(t_sp.loc['3x comprar y mantener', 'vol'])} "
          "del 3x. ΔEC de la LRS 3x (pp, γ = 1, 2, 3 y 5): " + "; ".join(rows3) + ". Con γ alto, las "
          "referencias de menos apalancamiento ganan con mucho. El SPA frente a la mejor referencia estática "
          f"da un p mínimo de {_p(spa.p_SPA_c.min())}.")
    th = {n: res[n]["th"] for n in mk}
    p4 = ("*Financiación.* Sobrecoste del swap que anula la ventaja de la LRS 3x con γ = 1 frente a la mejor "
          "referencia estática, que usa el mismo fondo 3x, si lo pagan todos (el caso relevante: un swap más "
          "caro encarece también la referencia): "
          + "; ".join(f"en {THE[n]}, {_bps(th[n].loc[a, 'γ=1, pagan todos'])}" for n in mk)
          + ". Si solo lo pagara la estrategia: "
          + "; ".join(f"en {THE[n]}, {_bps(th[n].loc[a, 'γ=1, solo la estrategia'])}" for n in mk)
          + ". Frente al 1x, que no paga swap: "
          + "; ".join(f"en {THE[n]}, {_bps(th[n].loc[a, 'γ=1, frente al 1x'])}" for n in mk) + ".")
    lit = []
    for k in NEW_DYNAMIC:
        wins = [f"{THE[n]} con γ = {g}" for n in mk for g in GAMMAS
                if res[n]["spa"][g]["ces"][k] > res[n]["spa"][g]["ce_b"]]
        lit.append(f"{SHORT[k]}, " + (", ".join(wins) if wins else "en ningún caso"))
    p5 = ("*Reglas de la literatura.* Superan a la mejor referencia estática del conjunto: " + "; ".join(lit)
          + ". Moreira–Muir en tiempo real arranca con una c estimada en la Gran Depresión (S&P 500) o en la "
          "burbuja de 2000 (semiconductores): la crítica de Cederburg et al. (2020) a la c de muestra completa "
          "tiene su contrapartida en la inestabilidad de la c en tiempo real (figura de Moreira–Muir).")
    return "\n\n".join([p1, p2, p3, p4, p5]) + "\n"


def caveats(res: dict, drag: pd.DataFrame) -> str:
    sp = res["S&P 500"]
    b1 = sp["b1"]
    n_static = len(static_names(sp["tipo"]))
    shifts = ", ".join(str(int(res[n]["tb"]["desplazamientos"].iloc[0])) for n in res)
    ec5 = res["Nasdaq-100"]["tabla"].loc["3x comprar y mantener", "EC γ=5"]
    sm0 = res["Semis"]["W"].idx[0]
    return f"""\
- Las referencias de L constante emparejadas (misma exposición media, misma volatilidad), la elección
  de la mejor referencia estática del conjunto y el mejor L constante de la rejilla 0–4 usan la muestra
  completa: son cotas a posteriori, no carteras que un inversor hubiera podido elegir. Elegir la mejor de
  {n_static} referencias a posteriori endurece el SPA; el L constante óptimo de cada γ, aún más exigente, se
  muestra como sensibilidad con las dos formas (3x + letras, los instrumentos de la LRS, y la forma barata).
- La LRS-Kelly y el ½ Kelly GJR heredan de R10–R13 la fracción ½ y el GJR, elegidos con el mismo S&P 500:
  su resultado en el S&P no es fuera de muestra. Faber (10 meses, 2007) y MOP (12 meses, 2012) usan sus
  parámetros publicados, fuera de muestra solo tras su publicación; la tabla «tras la publicación»
  mira el periodo {POST_FROM[:4]}–{sp['W'].idx[-1]:%Y}.
- La c de Moreira–Muir iguala la volatilidad de la cartera gestionada a la del 1x (su definición). La
  exposición se hace con el 3x y letras y se recorta en 3x, así que la c es la del problema sin tope.
  Con la c de muestra completa la regla mira el futuro: es una cota, fuera del SPA y de los umbrales.
- El equivalente cierto usa rentabilidades mensuales (como Goetzmann et al.). Con γ alto lo dominan unos
  pocos meses (octubre de 1987, 2000–2002, 2008) y es muy sensible a ellos; por eso con γ = 5 aparecen
  cifras como {pct_es(ec5, 0)} para el 3x del Nasdaq-100 (EC en tasa continua, que no tiene suelo en −100%).
  La tabla del EC según el horizonte lo repite con bloques de 12, 36 y 60 meses; con 60 meses hay muy
  pocos bloques y el resultado depende del mes de inicio (se da el rango).
- Las mezclas fijas L/3 en el 3x y letras, y las exposiciones fraccionales (Kelly, Moreira–Muir), suponen
  reequilibrio diario gratuito, como en R13; la tabla de coste de reequilibrio lo cifra entre
  {num_es(drag.values.min(), 0)} y {num_es(drag.values.max(), 0)} pb/año a 5 pb por unidad.
- El umbral de financiación grava el nocional de los fondos apalancados (el diferencial del swap): el 1x
  no paga. «Pagan todos» aplica el mismo sobrecoste a la referencia; cuando ésta usa tanto nocional medio
  como la estrategia, el sobrecoste no cambia su diferencia y la ventaja «no se anula». En el S&P 500 la
  mejor referencia estática con γ = 1 es {b1}, con un nocional medio de {num_es(sp['W'].n[b1].mean(), 2)}.
- Nasdaq-100: la evaluación empieza en 1985-10 con datos reales, pero las señales del arranque (media de
  200 sesiones, 10–12 meses, varianza de Kelly y c de Moreira–Muir) se calientan con una reconstrucción
  de 1971–85 (una sola, la primera de R12). Semiconductores: la ventana empieza en {sm0:%Y-%m} porque la c en
  tiempo real necesita diez años de meses.
- Tres mercados, un único camino histórico por mercado y muy correlacionados entre sí. Las bandas y los p
  bootstrap de Δg y ΔEC frente a los L emparejados son individuales: no corrigen por las dos estrategias ×
  tres mercados × cuatro γ que se miran (solo el SPA lo hace, y contesta otra pregunta). R19 corrige los p
  de titular de este informe junto a los de R16 y R18 (p_titular.csv).
- La nula sin habilidad desplaza la trayectoria de exposición dentro de la ventana, dando la vuelta al
  final. Conserva su distribución y su persistencia, pero los desplazamientos se solapan mucho ({shifts},
  de 21 en 21 sesiones), así que el p no es más fino de lo que permite un solo camino. Rompe a la vez la
  relación de la exposición con la rentabilidad futura y con la volatilidad futura: la «sincronización»
  incluye salir de los periodos turbulentos (volatilidad predecible), no solo acertar la dirección. Con γ
  alto, la nula del EC la dominan las pocas caídas en que la exposición desplazada cae a 3x. La nula del
  periodo tras la publicación se rehace dentro de ese periodo corto y es solo descriptiva.
- Los L constantes emparejados que sirven de referencia usan la exposición media y la volatilidad de toda
  la ventana; el bootstrap no refleja la incertidumbre de ese emparejamiento.
- La señal de precio fuera del S&P 500 es aproximada (rentabilidad total menos un dividendo constante): se
  muestra, pero no entra en el SPA.
"""


def post_table(res: dict) -> pd.DataFrame:
    rows = []
    for name, x in res.items():
        W0, d = x["W"], x["d"]
        sl = slice(pd.Timestamp(POST_FROM), W0.idx[-1])
        W = Window(d, x["R"], x["E"], x["N"], sl)
        for k in POST_ROWS:
            g_c = W.g(const_leverage(d, W.e[k]).loc[sl].values)
            dg = round(W.g(W.r[k]) - g_c, 12) + 0.0   # sin «−0» en las filas de L constante
            if x["tipo"][k].startswith("dinámica"):
                dn = null_paths(W, x["WT"][k])["g"] - g_c
                dn_mean, p = float(dn.mean()), perm_pvalue(dn, dg)
            else:   # exposición fija: desplazarla no cambia nada (la nula es la propia regla)
                dn_mean, p = dg, np.nan
            rows.append({"mercado": name, "estrategia": k, "exposición media": W.e[k], "g": W.g(W.r[k]),
                         POST_DG: dg, "Δg nulo: media": dn_mean, "sincronización": dg - dn_mean,
                         "p nula": p, **{f"EC γ={g}": W.ce(k, g) for g in GAMMAS}})
    return pd.DataFrame(rows)


def run():
    rep = Report("r17_benchmarks", "R17 — Referencias a igual riesgo para la LRS (I4)")
    rep.p(__doc__.split("\n", 2)[2])
    pos = len(rep.lines)
    rng = np.random.default_rng(SEED)            # SPA (mismo orden de sorteos que la versión anterior)
    rng_b = np.random.default_rng(SEED + 1000)   # bootstrap de Δg y ΔEC frente a los L emparejados
    rng_o = np.random.default_rng(SEED + 2000)   # SPA frente a cada uno de los dos L óptimos (añadido)
    D = r13.load()
    D.pop("_ndx")
    res, spa_rows, spa_sens, drag_rows = {}, [], [], []
    for name, d in D.items():
        var_fc = r13.var_forecast(d)
        R, E, N, kind, WT = build(d, var_fc)
        start, end = eval_window(d, R)
        sl = slice(start, end)
        info = add_static(d, R, E, N, kind, WT, sl)
        W = Window(d, R, E, N, sl)
        fr, fe = frontier(d, W), frontier(d, W, efficient=True)
        x = res[name] = {"W": W, "tipo": kind, "frontera": fr, "frontera_eficiente": fe, "d": d,
                         "R": R, "E": E, "N": N, "WT": WT, "info": info, "spa": {}}
        cmp_ = list(compare(d))

        # SPA: familia completa de reglas dinámicas implementables (R13 + R12 + I4); la señal de
        # precio entra por r13.family solo en el S&P 500
        main, grid = r13.family(d, {k: R[k] for k in d["_r13_keys"]}, lag=LAG)
        for k in NEW_DYNAMIC:
            main[k] = grid[k] = R[k]
        main = {k: v.loc[sl].values for k, v in main.items()}
        grid = {k: v.loc[sl].values for k, v in grid.items()}
        for g in GAMMAS:
            b = best_static(W, kind, g)
            row = x["spa"][g] = spa_block(W, main, W.lx(W.r[b]), g, rng)
            spa_rows.append({"mercado": name, "γ": g, "mejor estático": b, **row})
            L_opt, eff, _ = best_constant(fr, fe, g)
            s_opt = spa_block(W, main, W.lx(const_leverage(d, L_opt, eff).loc[sl].values), g, rng)
            s_grid = spa_block(W, grid, W.lx(W.r[b]), g, rng)
            bc = best_constants(fr, fe, g)
            s_f = {f: spa_block(W, main, W.lx(const_leverage(d, bc[f][0], f == FORM_CHEAP).loc[sl].values), g,
                                rng_o) for f in (FORM_LRS, FORM_CHEAP)}
            # EC de cada regla dinámica y de la mejor estática, para la lectura
            row["ces"] = {k: W.ce(k, g) for k in cmp_}
            row["ce_b"] = W.ce(b, g)
            spa_sens.append({"mercado": name, "γ": g, "p_SPA_c principal": row["p_SPA_c"],
                             f"p_SPA_c con rejilla ({len(grid)} reglas)": s_grid["p_SPA_c"],
                             "mejor L a posteriori (cualquier forma)": L_opt,
                             "p_SPA_c frente al mejor L (cualquier forma)": s_opt["p_SPA_c"],
                             f"L óptimo ({FORM_LRS})": bc[FORM_LRS][0],
                             f"p_SPA_c frente a L óptimo ({FORM_LRS})": s_f[FORM_LRS]["p_SPA_c"],
                             f"L óptimo ({FORM_CHEAP})": bc[FORM_CHEAP][0],
                             f"p_SPA_c frente a L óptimo ({FORM_CHEAP})": s_f[FORM_CHEAP]["p_SPA_c"],
                             "ΔEC mejor regla frente al mejor L": s_opt["ΔEC mejor"]})

        rep.h(f"{name}: {start.date()} → {end.date()} ({len(W.starts)} meses)")
        show = ["1x comprar y mantener", "3x comprar y mantener"] + \
            [k for k in R if k.startswith("L constante")] + [KELLY_EX_ANTE] + cmp_ + [MM_FULL]
        t = x["tabla"] = ce_table(W, show, kind)
        rep.p("Equivalente cierto (EC) CRRA en exceso de las letras, en % anual continuo, con "
              "rentabilidades mensuales. Exposición media al índice tras el retraso de ejecución.")
        rep.table(as_pct(t, ["CAGR", "vol", "caída máx."] + EC_COLS), digits=2)
        rep.p("L de las referencias emparejadas: " + "; ".join(
            f"{k.replace('L constante = ', '')}: {num_es(v, 3)}" for k, v in info.items()
            if k.startswith("L constante")) +
            f". c de Moreira–Muir con la muestra completa: {num_es(info['c_muestra_completa'], 4)} "
            f"(en tiempo real: {num_es(d['_c_mm'].loc[start:].dropna().iloc[0], 4)} al empezar y "
            f"{num_es(d['_c_mm'].loc[:end].iloc[-1], 4)} al final).")

        rep.h("Mejor referencia estática del conjunto y mejor L constante a posteriori", 3)
        rows, best_ref = [], {"estática": [], FORM_LRS: [], FORM_CHEAP: []}
        pk = px_key(d)
        for g in GAMMAS:
            b = best_static(W, kind, g)
            bc = best_constants(fr, fe, g)
            refs = [("mejor referencia estática del conjunto", SHORT.get(b, b), W.ce(b, g))] + \
                [(f"L óptimo a posteriori ({f})", num_es(bc[f][0], 2), bc[f][1]) for f in (FORM_LRS, FORM_CHEAP)]
            best_ref["estática"].append(SHORT.get(b, b))
            best_ref[FORM_LRS].append(bc[FORM_LRS][0])
            best_ref[FORM_CHEAP].append(bc[FORM_CHEAP][0])
            for lab, which, ce_r in refs:
                rows.append({"γ": g, "referencia": lab, "cuál / L": which, "EC referencia": ce_r,
                             "ΔEC LRS 3x": W.ce(FOCAL[0], g) - ce_r, f"ΔEC {SHORT[pk]}": W.ce(pk, g) - ce_r,
                             "ΔEC LRS-Kelly": W.ce(FOCAL[1], g) - ce_r})
        x["bestref"] = best_ref
        x["b1"] = best_static(W, kind, 1)
        bt = pd.DataFrame(rows)
        rep.table(as_pct(bt, ["EC referencia"]).pipe(
            lambda df: as_pct(df, [c for c in df.columns if c.startswith("ΔEC")], "pp")), index=False, digits=2)
        rep.p("La mejor referencia estática es la mejor de un conjunto discreto (1x, 3x, los L constantes "
              "emparejados y ½ Kelly ex ante); el mejor L constante a posteriori es el máximo de la rejilla "
              "0–4 (de 0,05 en 0,05) con los instrumentos de la LRS (3x + letras) o con la forma más barata "
              "(1x + letras hasta L = 1, 1x + 3x después). ΔEC en pp anuales.")
        gi = lambda L: float(np.interp(L, fr.index, fr.g)) * 100  # noqa: E731
        rep.p(f"La frontera es plana cerca de su máximo: con 3x + letras, g(L) vale {num_es(fr.g.max() * 100, 2)}% "
              f"en L = {num_es(fr.g.idxmax(), 2)}, {num_es(gi(2.1), 2)}% en L = 2,1, {num_es(gi(1.5), 2)}% en "
              f"L = 1,5 y {num_es(gi(1.0), 2)}% en L = 1 (el 1x directo: "
              f"{num_es(W.g(W.r['1x comprar y mantener']) * 100, 2)}%; la diferencia es el coste de obtener "
              "exposición 1 con un tercio en el 3x).")

        dyn = cmp_ + [k for k in R if kind[k] == "dinámica" and k not in cmp_] + [MM_FULL]
        bidx = boot_indices(len(W.starts), rng_b)
        tA, tB, nulls = timing_tables(d, W, fr, E, WT, dyn, bidx)
        x.update(tA=tA, tb=tB, nulls=nulls)
        rep.h("Crecimiento frente al apalancamiento constante emparejado", 3)
        rep.table(fmt_p(as_pct(as_pct(tA, ["g"]), ["Δg misma exposición", "IC 90% inf.", "IC 90% sup.",
                                                    "Δg misma exposición, forma barata", "Δg misma vol"], "pp")),
                  digits=2)
        rep.p("g: crecimiento logarítmico anual (ln(1 + CAGR)). Δg: diferencia con el apalancamiento "
              "constante de igual exposición media (con 3x + letras, o con la forma más barata) o de igual "
              "volatilidad realizada (3x + letras). IC 90% y p: bootstrap estacionario de las diferencias "
              f"mensuales de log-rentabilidad (bloque medio de {BLOCK_M} meses, {B} remuestreos); p es la "
              "probabilidad unilateral de «Δg ≤ 0». Esta diferencia NO es la sincronización: incluye lo que "
              "cuesta que la exposición varíe (véase la tabla siguiente).")

        rep.h("Sincronización: comparación con la misma regla sin habilidad", 3)
        rep.table(fmt_p(as_pct(tB.drop(columns="desplazamientos"),
                               ["Δg misma exposición", "Δg nulo: media", "Δg nulo: 5%", "Δg nulo: 95%",
                                "−½σ²Var(L)", "sincronización", "Δg misma vol", "Δg nulo misma vol: media",
                                "sincronización (vol)"], "pp")), digits=2)
        rep.p(f"Nula: la trayectoria de pesos ejecutados de cada regla desplazada circularmente dentro de la "
              f"ventana, todos los desplazamientos de {NULL_STEP} en {NULL_STEP} sesiones entre "
              f"{NULL_MIN_SHIFT} sesiones (dos años) y el tamaño de la ventana menos {NULL_MIN_SHIFT} "
              f"({int(tB['desplazamientos'].iloc[0])} desplazamientos), con los mismos fondos, letras y costes "
              "de rotación. Conserva la exposición media, su dispersión y su persistencia, pero no su relación "
              "con el mercado. «Δg nulo»: lo que esa regla sin habilidad crece de más o de menos que el L "
              "constante de su exposición media; «−½σ²Var(L)»: la aproximación de segundo orden a esa "
              "penalización (sin costes). «Sincronización» = Δg − media nula; «p nula»: fracción de "
              "desplazamientos que igualan o superan el Δg observado. Las columnas «vol» repiten el ejercicio "
              "frente al L constante de igual volatilidad (cada desplazamiento con el de su propia volatilidad). "
              "La sincronización incluye tanto la de la rentabilidad como la de la volatilidad (salir en los "
              "periodos turbulentos), que la nula no separa.")

        rep.h("Equivalente cierto frente a referencias de igual riesgo", 3)
        ecm = x["ecm"] = ec_matched(W, nulls, x["spa"], bidx, fr, fe)
        rep.table(fmt_p(as_pct(ecm, ["ΔEC misma exposición", "IC 90% inf.", "IC 90% sup.", "ΔEC misma vol",
                                     "ΔEC nulo: media", "ΔEC mejor estático", f"ΔEC L óptimo ({FORM_LRS})",
                                     f"ΔEC L óptimo ({FORM_CHEAP})"], "pp")), index=False, digits=2)
        rep.p("Dos preguntas distintas. (1) ¿Añade la sincronización equivalente cierto a igual riesgo? Columnas "
              "«misma exposición» y «misma vol»: ΔEC frente al L constante emparejado, con banda del 90% y p "
              "unilateral por bootstrap de meses; «ΔEC nulo» y «p nula»: lo mismo para la regla sin habilidad "
              "(exposición desplazada). (2) ¿Es su nivel de apalancamiento el adecuado para ese γ? Columnas "
              "«mejor estático» (la mejor referencia del conjunto discreto, a posteriori) y «L óptimo» (el mejor "
              "L constante a posteriori con cada forma): esas diferencias mezclan el nivel de riesgo con la "
              "sincronización.")

        rep.h("EC según el horizonte de evaluación", 3)
        eh = ec_horizon_table(W, kind, [FOCAL[0], pk, FOCAL[1], "1x comprar y mantener"])
        rep.table(as_pct(eh, [c for c in eh.columns if c.startswith(("EC γ", "mín γ", "máx γ"))]),
                  index=False, digits=2)
        rep.p("El EC de las tablas anteriores es miope: evalúa mes a mes. Aquí, con la suma de los log-excesos "
              "mensuales en bloques no solapados de 12, 36 y 60 meses (anualizada), para cada mes de inicio "
              "posible de los bloques: «EC» es la media sobre esos inicios y «mín»/«máx», su rango; «bloques», "
              f"el menor número de bloques. Con 60 meses son {int(eh[eh['meses por bloque'] == 60].bloques.min())} "
              "bloques: una estimación muy poco precisa. La mejor referencia estática se elige, para cada γ, con "
              "el EC mensual.")
        rep.h("EC por bloques con los meses barajados (referencia del estimador)", 3)
        es = ec_horizon_shuffled(W, [FOCAL[0], "1x comprar y mantener"])
        rep.table(as_pct(es, [c for c in es.columns if c.startswith(("EC γ", "barajado γ", "obs − barajado"))]),
                  index=False, digits=2)
        rep.p(f"Añadido tras una revisión externa. «barajado»: el mismo EC por bloques de la tabla anterior con los "
              f"meses de la regla en orden aleatorio ({N_SHUFFLE} permutaciones, semilla propia): sin dependencia "
              "entre meses el EC verdadero no cambia con el horizonte, así que lo que sube con los bloques largos "
              "es del estimador (el logaritmo de la media de pocos bloques, con 1/(1 − γ) < 0, sube en las "
              "estrategias con meses muy malos). «obs − barajado» es lo que no explica el estimador: lo que "
              "aporta el orden real de los meses (positivo si tras los meses malos vinieron buenos; las rachas "
              "lo bajarían), sin contraste formal.")

        rep.h("Umbral de sobrecoste de financiación (pb/año sobre el nocional apalancado)", 3)
        th = x["th"] = thresholds(d, W, kind, cmp_, fr)
        rep.lines += [fmt_bps(th).to_markdown(disable_numparse=True), ""]
        th.to_csv(rep.dir / f"umbrales_{name.replace(' ', '_').replace('&', 'y')}.csv")
        rep.p("Sobrecoste de financiación que iguala el EC de la regla al de la referencia. Frente a la mejor "
              "referencia estática de cada γ: «pagan todos» (la referencia, que usa el mismo fondo 3x, paga el "
              "mismo sobrecoste sobre su nocional: el caso relevante) y «solo la estrategia» (colchón frente a "
              "una desventaja de costes que solo sufriera ella). Positivo: colchón de la ventaja; negativo: "
              "abaratamiento que haría falta para empatar; «sin ventaja»: no la supera y ningún abaratamiento de "
              "hasta 3.000 pb lo cambia (o, si todos pagan, abaratar la financiación la aleja más). «No se "
              "anula»: la ventaja persiste con cualquier sobrecoste hasta 5.000 pb (con «pagan todos», cuando la "
              "referencia usa tanto nocional medio o más). «Frente al 1x»: referencia 1x, que no paga.")
        drag_rows.append({"mercado": name, **{f"L = {num_es(L, 1)}": rebalance_drag(W, L) * 1e4 for L in (1.0, 1.5, 2.0)}})

    rep.h("Contraste SPA frente a la mejor referencia estática de cada γ")
    spa = pd.DataFrame(spa_rows).drop(columns=["ces", "ce_b"], errors="ignore")
    spa["p_Holm"] = stats.holm(spa.p_SPA_c.values)
    dcols = ["ΔEC mejor"] + [f"ΔEC {SHORT[k]}" for k in FOCAL]
    rep.table(as_pct(spa, dcols, "pp"), index=False, digits=3)
    rep.p(f"Familia principal: las {spa.reglas.iloc[0]} reglas dinámicas implementables probadas en el S&P 500 "
          f"(R12–R13 e I4: LRS 1x/2x/3x, la LRS 3x con la señal de precio, banda, rotación, ½ Kelly "
          "GJR/EWMA/ARCH/GARCH, LRS-Kelly y su versión suavizada, Faber 1x y 3x, TSMOM 3x y Moreira–Muir en "
          f"tiempo real); {spa.reglas.iloc[-1]} en semiconductores y el Nasdaq-100, sin la señal de precio "
          "aproximado. La medida es la "
          "diferencia de utilidad CRRA mensual u(R_regla) − u(R_referente), cuyo signo medio coincide con "
          "el de la diferencia de equivalentes ciertos. Bootstrap estacionario con bloques medios de "
          f"{BLOCK_M} meses y {B} remuestreos. «p individual» es el mismo contraste para una sola regla "
          "(sin corregir por la búsqueda). p_Holm corrige las 12 comparaciones (3 mercados × 4 γ), muy "
          "dependientes entre sí: es conservador. Ojo: este contraste compara con la MEJOR referencia "
          "estática de cada γ (con γ ≥ 2, casi siempre el 1x), así que mezcla el nivel de apalancamiento con "
          "la sincronización; la pregunta de si la sincronización suma a igual riesgo se contrasta en las "
          "tablas de cada mercado.")
    rep.h("Sensibilidad del SPA", 3)
    ss = pd.DataFrame(spa_sens)
    rep.table(as_pct(ss, ["ΔEC mejor regla frente al mejor L"], "pp"), index=False, digits=3)
    rep.p("«Con rejilla» añade las 30 combinaciones de media (50–300 sesiones) y banda (0–5%) de la LRS 3x "
          "de R13, sobre el índice de rentabilidad total. «Frente al mejor L» usa como referente el mejor "
          "apalancamiento constante de la rejilla 0–4 elegido a posteriori para cada γ, con cualquiera de las "
          "dos formas o con cada una por separado (3x + letras, los instrumentos de la LRS, y la forma más "
          "barata): la referencia estática más exigente posible. Los contrastes frente a cada forma se "
          "añadieron tras la revisión, con su propio generador de sorteos.")

    rep.h(f"Tras la publicación de la LRS ({POST_FROM[:4]} → fin de la muestra)")
    pt = post_table(res)
    rep.table(fmt_p(as_pct(as_pct(pt, ["g"] + EC_COLS), [POST_DG, "Δg nulo: media", "sincronización"], "pp")),
              index=False, digits=2)
    rep.p(f"Mismas reglas; las filas «L constante» mantienen los L emparejados con la muestra completa. "
          f"{POST_FROM[:4]} es el primer año completo tras la publicación de Gayed y Bilello (SSRN, marzo de "
          "2016). Como la exposición media de cada regla cambia en el subperiodo, la columna Δg la compara con "
          "el L constante (3x + letras) de su propia exposición media en el periodo. «Δg nulo» y "
          "«sincronización»: la nula sin habilidad rehecha dentro del periodo (la trayectoria da la vuelta "
          "dentro de él); en las filas de exposición fija la nula es la propia regla. Casi diez años y un solo "
          "camino por mercado: descriptivo.")

    rep.h("Coste del reequilibrio diario que se supone gratis", 3)
    drag = pd.DataFrame(drag_rows).set_index("mercado")
    rep.table(drag, digits=1)
    rep.p("pb/año que costaría devolver cada día una mezcla fija L/3 en el 3x y el resto en letras a sus "
          "pesos, a 5 pb por unidad negociada. Lo omiten por igual las referencias de L constante y las "
          "reglas con exposición fraccional (½ Kelly, LRS-Kelly, Moreira–Muir), como en R13.")

    tit = provenance.write_titular(rep.dir, titular_rows(res, spa), "r17_benchmarks")
    rep.h("p de titular exportados (p_titular.csv)")
    rep.p("La lista cerrada de p de titular de este informe, que R19 lee para corregirlos junto a los de "
          "R16 y R18. Procedencia: commit " + tit.commit.iloc[0][:12]
          + (" con cambios sin commit en el código del que depende" if tit.cambios_sin_commit.iloc[0] else "")
          + f"; huella del código {tit.huella_código.iloc[0]}.")
    rep.table(tit[["mercado", "contraste", "p"]], index=False, digits=4)

    fig_frontier(res, rep)
    fig_timing(res, rep)
    fig_ce_gamma(res, rep)
    fig_mm(res, rep)
    rep.h("Salvedades")
    rep.p(caveats(res, drag))
    rep.lines[pos:pos] = summary_lines(res, spa, pt, ss) + [reading(res, spa), ""]
    rep.save()
    return res, spa, ss


if __name__ == "__main__":
    run()
