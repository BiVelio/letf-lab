"""R22 — Inversos, pares y el supuesto arbitraje.

Pregunta (artículo 3). ¿Mejoran los ETFs inversos (SH, SDS, SPXU; PSQ, QID, SQQQ) la
LRS 3x de Gayed y Bilello (2016)? ¿Se puede «arbitrar» el decaimiento de los fondos
apalancados vendiendo en corto los dos lados de un par (TQQQ y SQQQ, UPRO y SPXU, SSO y
SDS, QLD y QID) o vendiendo el 3x y comprando tres veces el índice? ¿Es mejor el 2x (SSO)
que el 3x con la misma regla, y hasta dónde convendría apalancar?

Cinco bloques, fijados antes de mirar resultados (los parámetros salen de la exploración
previa del artículo 3, cuyas cifras no se citan: se rehízo todo aquí con las convenciones
actuales):
(a) Inverso bajo la media. La LRS original pasa a letras cuando el índice de rentabilidad
    total cierra bajo su media de 200 sesiones; aquí pasa a un ETF inverso: 3x/−1x,
    3x/−2x, 3x/−3x, 2x/−2x y 1x/−1x; 3x/−3x con banda de ±2% y 3x/−3x solo si la media
    baja (pendiente de 20 sesiones). Además, la «LRS con corto del −3x»: la misma
    exposición alcista (+3) vendiendo en corto el −3x en lugar de comprar el 3x.
(b) Par corto («cosecha del decaimiento»): −0,5 en el fondo alcista y −0,5 en el bajista
    del mismo índice y |L|, reajustado cada día, semana, mes o trimestre, con una rejilla
    de préstamo de títulos (0, 2, 5 y 10% anual) y tres reglas de margen (ninguna;
    restaurar el objetivo; liquidar y esperar en caja) con el mantenimiento de FINRA
    para ETFs apalancados (30% × |L| del valor del corto). Su P&L de camino en cada
    periodo es ≈ (L²/2)·(Σx² − (Σx)²), con x = u − f: una apuesta a que el cociente de
    varianzas es menor que 1.
(c) Lever-arb: corto del ETF L× y L veces el índice en futuros (cobertura diaria, semanal,
    mensual o trimestral; préstamo 0, 1, 3 y 5%), el tamaño viable con margin call y la
    versión al contado en cuenta de margen.
(d) Par largo: comprar los dos lados, como control.
(e) 2x frente a 3x con la regla de Gayed y Bilello y la curva de apalancamiento de la
    regla (1,5x…6x sintéticos, con un fondo nuevo tras un día de −100% que la regla no
    llevaba), frente al L* de Kelly dentro y fuera de la tendencia (dentro de muestra; por
    encima de 3x no hay fondo cotizado en la base).

Protocolo (el de R13): señal al cierre de t con la media de 200 sesiones del índice de
rentabilidad total (en el S&P 500, ^SP500TR desde 1988 y antes ^GSPC con el D/P de Shiller
del mes anterior); orden al cierre de t+1; 5 pb por unidad negociada; la caja rinde la
letra. Las carteras de pesos fijos se construyen con r13.run_weights (una llamada por
fondo: por linealidad, la suma de las carteras de un fondo menos las letras sobrantes es la
cartera de varios fondos). Las carteras con cortos o con pesos que derivan entre reajustes
usan un simulador propio con las mismas convenciones (run_drift): el capital cobra la
letra, el producto de los cortos la letra − 0,5% y el préstamo de títulos se paga sobre el
valor vivo del corto; los futuros rinden u − f con 1 pb por unidad. Sensibilidad (añadida tras
la revisión): el producto de los cortos sin remunerar, como en muchas cuentas minoristas.

Fondos. Un fondo L× diario (también L < 0) es rf + L·(u − f) − |L|·s − ER, como r12.letf:
con L < 0 cobra el tipo de financiación f sobre el nocional corto (lo que recibe la pata
corta de un swap) y paga los dividendos (u es de rentabilidad total). El 3x es el del
laboratorio (d["x3"]); el sobrecoste s de los demás se calibra con la vida de cada ETF
real y se valida fuera de muestra (s de 2010–2017, evaluado en 2018–2026). Sintético antes
del lanzamiento y real después.

Mercados: S&P 500 (1928–2026; la ventana principal empieza el 4-2-1938, como R13) y
Nasdaq-100 desde 1985-10 (10 reconstrucciones del periodo previo para calentar la media;
los contrastes, sobre la reconstrucción 0). ETFs reales desde 2010-02-12.

Inferencia: Δg (crecimiento logarítmico anual) frente a la LRS 3x original, al 1x y a la
letra con bootstrap estacionario emparejado (bloque medio de 63 sesiones, 2.000
remuestreos, los mismos para todas las reglas de una familia), p unilateral normal con el
error del bootstrap (como R19), Holm y step-down de Romano y Wolf, y SPA de Hansen por
familia. Ninguna variante se llama mejora sin la corrección por la búsqueda.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from letf_lab import kelly, metrics, stats
from letf_lab.data import history
from letf_lab.report import FIG_W, Report, comma_fig, legend_band, num_es, pct_es
from letf_lab.research import r12_long_ml as r12
from letf_lab.research import r13_lrs as r13
from letf_lab.universe import BY_TICKER

log = logging.getLogger(__name__)

SEED = 22
B = 2000                   # remuestreos del bootstrap (p, IC, Romano–Wolf, SPA)
B_KELLY = 1000             # remuestreos del IC de L*
BLOCK = 63                 # bloque medio del bootstrap estacionario (sesiones)
BLOCK_KELLY = 252
LAG = r13.EXEC_LAG         # señal al cierre de t, orden al cierre de t+1
COST_BPS = r13.COST_BPS    # 5 pb por unidad negociada de un ETF
FUT_BPS = 1.0              # 1 pb por unidad negociada de un futuro
HAIRCUT = 0.005            # el producto de un corto cobra la letra − 0,5%
MAINT = 0.30               # FINRA 4210: mantenimiento de un corto = 30% × |L| de su valor (tope 100%)
CLIP = -0.9999             # pérdidas acotadas al tomar logaritmos (como r13.log_growth y R19)

ORIGINAL = r13.ORIGINAL
ONE = "1x comprar y mantener"
THREE = "3x comprar y mantener"
TBILL = "letras"
SP, NDX = "S&P 500", "Nasdaq-100"
SP_FROM = "1938-02-04"                 # primera sesión con previsión GJR: la ventana de R13
RULE_FROM = r13.FIRST_RULE_DAY         # primera sesión de la LRS (1928-10-19)
NDX_FROM = r12.NDX_EVAL_FROM
REAL_FROM = r13.REAL_FROM

FUNDS = {SP: {3: "UPRO", 2: "SSO", -1: "SH", -2: "SDS", -3: "SPXU"},
         NDX: {3: "TQQQ", 2: "QLD", -1: "PSQ", -2: "QID", -3: "SQQQ"}}
PAIRS = {SP: (("SSO", "SDS", 2), ("UPRO", "SPXU", 3)), NDX: (("QLD", "QID", 2), ("TQQQ", "SQQQ", 3))}
ARB = {SP: (("SSO", 2), ("UPRO", 3)), NDX: (("QLD", 2), ("TQQQ", 3))}
FREQS = ("D", "W", "M", "Q")
FREQ_ES = {"D": "diario", "W": "semanal", "M": "mensual", "Q": "trimestral"}
FREQ_FEM = {"D": "diaria", "W": "semanal", "M": "mensual", "Q": "trimestral"}
PER_YEAR = {"W": 52, "M": 12, "Q": 4}
PAIR_BORROW = (0.0, 0.02, 0.05, 0.10)
ARB_BORROW = (0.0, 0.01, 0.03, 0.05)
SHORT_BORROW = (0.0, 0.02, 0.05)
MARGIN_RULES = ("ninguna", "restaurar", "liquidar")
L_CURVE = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
L_MAX_KELLY = 8.0
CALIB_SPLIT = "2017-12-31"             # s de 2010–2017 para la prueba fuera de muestra (2018–2026)

SP_WINDOWS = {"1938–2026": (SP_FROM, None), "1928–2026": (RULE_FROM, None),
              "1928–1937": (RULE_FROM, "1937-12-31"), "1938–1973": (SP_FROM, "1973-12-31"),
              "1974–1999": ("1974-01-01", "1999-12-31"), "2000–2026": ("2000-01-01", None),
              "2016–2026": ("2016-01-04", None), "2017–2026": ("2017-01-03", None),
              "ETFs reales": (REAL_FROM, None)}
NDX_WINDOWS = {"1985–2026": (NDX_FROM, None), "1985–1999": (NDX_FROM, "1999-12-31"),
               "2000–2026": ("2000-01-01", None), "2016–2026": ("2016-01-04", None),
               "ETFs reales": (REAL_FROM, None)}
WINDOW_NOTE = {"2016–2026": "tras la muestra de Gayed y Bilello (acaba en 2015)",
               "2017–2026": "primer año completo tras la publicación (SSRN, marzo de 2016)",
               "ETFs reales": f"desde {REAL_FROM}, todos los fondos reales"}
ERAS = {SP: {"1928–1937": ("1928-01-01", "1937-12-31"), "1938–1973": ("1938-01-01", "1973-12-31"),
             "1974–1999": ("1974-01-01", "1999-12-31"), "2000–2026": ("2000-01-01", None)},
        NDX: {"1985–1999": (NDX_FROM, "1999-12-31"), "2000–2026": ("2000-01-01", None)}}
EPISODES = {"crac 1929–32": ("1929-09-03", "1932-07-08"), "rebote 1932–33": ("1932-07-08", "1933-07-18"),
            "octubre 1987": ("1987-10-01", "1987-10-30"), "2000–02": ("2000-03-24", "2002-10-09"),
            "2008–09": ("2008-09-01", "2009-03-09"), "covid 2020": ("2020-02-19", "2020-03-23"),
            "2022": ("2022-01-03", "2022-10-12")}
VR_H = (5, 21, 63)                     # horizontes del cociente de varianzas (sesiones)
N_IID, T_IID = 200_000, 21             # meses sorteados y sesiones por mes en el mundo iid

# familias (para el recuento y el SPA)
FAM_INV = "(a) inverso bajo la media"
FAM_SHORT3 = "(a) LRS con corto del −3x"
FAM_LEV = "(e) apalancamiento de la regla"
FAM_PAIR = "(b) par corto"
FAM_LONG = "(d) par largo"
FAM_ARB = "(c) lever-arb"
FAM_LEV_PX = "(e) apalancamiento con la señal de precio (sensibilidad)"
TREND_FAMS = (FAM_INV, FAM_SHORT3, FAM_LEV, FAM_LEV_PX)
FAMS = (FAM_INV, FAM_SHORT3, FAM_LEV, FAM_LEV_PX, FAM_PAIR, FAM_LONG, FAM_ARB)
PX_TAG = "señal de precio"
NEUTRAL_FAMS = (FAM_PAIR, FAM_ARB)
NEEDS_SHORT = (FAM_SHORT3, FAM_PAIR, FAM_ARB)
REALISTIC = "préstamo realista (pares ≥ 2%, lever-arb ≥ 1% con futuros)"

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, GREY = "#0b0b0b", "#8a8985"


# ───────────────────────────── utilidades ─────────────────────────────

def lx(L: float) -> str:
    """Rótulo de un apalancamiento con coma decimal: 2,5x; −3x."""
    s = num_es(L, 1).replace(",0", "") if float(L) != int(L) else num_es(L, 0)
    return f"{s}x"


def logr(r: pd.Series | np.ndarray):
    """log(1 + r) con las pérdidas acotadas en CLIP."""
    return np.log1p(np.maximum(r, CLIP)) if isinstance(r, np.ndarray) else np.log1p(r.clip(lower=CLIP))


def absorb(r: pd.Series) -> pd.Series:
    """Una cartera que pierde el 100% queda a cero: tras el primer día con r ≤ −1, rentabilidad 0."""
    dead = (r <= -1.0).cummax().shift(1, fill_value=False)
    return r.clip(lower=-1.0).mask(dead & r.notna(), 0.0)


def ruin_date(r: pd.Series):
    hit = r[r <= -1.0]
    return hit.index[0] if len(hit) else None


def rebalance_days(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """True en la última sesión de cada semana (ISO), mes o trimestre del calendario; 'D' = todas."""
    if freq == "D":
        return np.ones(len(index), bool)
    if freq == "W":
        iso = index.isocalendar()
        key = iso.year.to_numpy("int64") * 100 + iso.week.to_numpy("int64")
    elif freq == "M":
        key = (index.year * 100 + index.month).to_numpy("int64")
    elif freq == "Q":
        key = (index.year * 10 + index.quarter).to_numpy("int64")
    else:
        raise ValueError(freq)
    return key != np.r_[key[1:], -1]


# ───────────────────────────── fondos ─────────────────────────────

def synthetic(d: dict, L: float, spread: float | None = None, er: float | None = None) -> pd.Series:
    """Fondo diario L× (también L < 0) con el modelo del laboratorio (r12.letf con otro L, s y ER):
    r = rf + L·(u − f) − |L|·s/252 − ER/252, acotado en −100%. A diferencia de sim.letf_returns,
    un día de −100% no deja el fondo muerto para siempre: al día siguiente sigue la fórmula
    (un fondo nuevo). Quien lo llevaba ese día lo pierde todo (absorb en la cartera)."""
    s = d["spread"] if spread is None else spread
    e = d["er"] if er is None else er
    r = d["rf"] + L * (d["u"] - d["f"]) - abs(L) * s / 252 - e / 252
    return r.clip(lower=-1.0)


def calibrate_spread(real: pd.Series, d: dict, L: float, er: float, start=None, end=None) -> float:
    """Sobrecoste anual s con el que el sintético reproduce la media del ETF real en [start, end]."""
    df = pd.concat([real, d["u"], d["rf"], d["f"]], axis=1, keys=["x", "u", "rf", "f"]).loc[start:end].dropna()
    base = df.rf + L * (df.u - df.f) - er / 252
    return float(-(df.x - base).mean() * 252 / abs(L))


def fund_params(D: dict) -> dict:
    """Parámetros de cada ETF: L, gasto, s (vida real completa), s de 2010–2017 y la serie real.
    El 3x usa el s y el gasto del laboratorio (d['spread'], d['er']), con los que se construye
    d['x3']; su s calibrado con este método se guarda aparte para la tabla de validación."""
    out = {}
    for mkt, funds in FUNDS.items():
        d = D[mkt]
        for L, tk in funds.items():
            e = BY_TICKER[tk]
            real = r12.real_letf(tk).reindex(d["u"].index)
            s_full = calibrate_spread(real, d, L, e.expense_ratio)
            s_is = calibrate_spread(real, d, L, e.expense_ratio, end=CALIB_SPLIT)
            lab = L == 3
            out[tk] = {"mercado": mkt, "L": L, "er": d["er"] if lab else e.expense_ratio,
                       "s": d["spread"] if lab else s_full, "s_calibrado": s_full, "s_2010_2017": s_is,
                       "real": real, "lanzamiento": real.first_valid_index()}
    return out


def hybrid(d: dict, P: dict, tk: str) -> pd.Series:
    """Sintético (s calibrado) antes del lanzamiento y ETF real después; el 3x es d['x3']."""
    p = P[tk]
    if p["L"] == 3:
        return d["x3"]
    real = p["real"].reindex(d["u"].index)
    return real.where(real.notna(), synthetic(d, p["L"], p["s"], p["er"]))


def market_funds(d: dict, P: dict, mkt: str) -> dict:
    return {L: hybrid(d, P, tk) for L, tk in FUNDS[mkt].items()}


def validation_table(D: dict, P: dict) -> pd.DataFrame:
    """Sintético frente a ETF real desde REAL_FROM: con el s de toda la vida (dentro de muestra) y,
    en 2018–2026, con el s de 2010–2017 (fuera de muestra)."""
    rows = {}
    for tk, p in P.items():
        d = D[p["mercado"]]
        real = p["real"]
        for lab, s, a in (("2010–2026 (s de toda su vida)", p["s"], REAL_FROM),
                          ("2018–2026 (s de 2010–2017)", p["s_2010_2017"], "2018-01-01")):
            syn = synthetic(d, p["L"], s, p["er"])
            df = pd.concat([real, syn], axis=1, keys=["real", "sint"]).loc[a:].dropna()
            diff = df.real - df.sint
            key = (tk, lab)
            rows[key] = {"L": p["L"], "gasto": p["er"], "s": s, "CAGR_real": metrics.cagr(df.real),
                         "CAGR_sintético": metrics.cagr(df.sint), "diferencia_media_anual": diff.mean() * 252,
                         "error_de_seguimiento_anual": diff.std() * np.sqrt(252),
                         "correlación_diaria": df.real.corr(df.sint),
                         "caída_máx_real": metrics.max_dd(df.real), "caída_máx_sintético": metrics.max_dd(df.sint)}
    t = pd.DataFrame(rows).T
    t.index.names = ["ETF", "prueba"]
    return t


# ───────────────────────────── carteras ─────────────────────────────

def run_legs(d: dict, legs: list, lag: int = LAG) -> pd.Series:
    """Pesos decididos al cierre de t en varios fondos y el resto en letras, con r13.run_weights.

    run_weights(d, w_k, x=F_k) = w_k·F_k + (1 − w_k)·rf − |Δw_k|·5 pb. La suma sobre los K fondos
    menos (K − 1)·rf es Σ w_k·F_k + (1 − Σ w_k)·rf − Σ|Δw_k|·5 pb: la cartera de varios fondos
    con el mismo retraso, el mismo coste por unidad negociada y la misma máscara de validez."""
    out = None
    for w, x in legs:
        r = r13.run_weights(d, w, lag=lag, x=x)
        out = r if out is None else out + r - d["rf"]
    return out


def run_drift(target: pd.DataFrame, R: pd.DataFrame, rf: pd.Series, freq: str = "M",
              borrow: float | dict = 0.0, haircut: float = HAIRCUT, cost_bps: float = COST_BPS,
              lag: int = LAG, maint: dict | None = None, margin_rule: str = "ninguna",
              futures: tuple = (), fin: pd.Series | None = None, fin_extra: float = 0.0,
              fut_bps: float = FUT_BPS, netting: bool = False, rebate: bool = True) -> tuple[pd.Series, dict]:
    """Cartera por 1 $ de patrimonio con pesos que derivan entre reajustes (cortos incluidos).

    target: pesos objetivo por activo decididos al cierre de t (negativo = corto); con lag = 2 se
    ejecutan al cierre de t+1 (como r13.run_weights), en los días de reajuste de `freq` o cuando
    cambia el objetivo. R: rentabilidades diarias de cada activo (mismas columnas).
    - Fondos: los largos consumen caja; los cortos pagan `borrow` anual sobre su valor y su
      producto cobra la letra − haircut. El capital no invertido cobra la letra; si es negativo,
      es un préstamo al tipo `fin` (f por defecto la letra) + fin_extra.
    - Futuros (columnas en `futures`): exposición sin caja que rinde r − fin; fut_bps por unidad.
    - netting (cuenta de margen minorista): caja total = capital libre + producto de los cortos;
      si es positiva cobra la letra − haircut; si es negativa paga fin + fin_extra.
    - rebate=False (sensibilidad): el producto de los cortos no rinde nada y no compensa saldos (muchas
      cuentas minoristas); el capital libre cobra la letra y un saldo negativo paga fin + fin_extra.
    - maint: mantenimiento de cada corto (fracción de su valor). Si al cierre, antes de reajustar,
      Σ maint·|corto| supera el patrimonio, es un incumplimiento (se cuenta siempre, también con
      reajuste diario). Si ese cierre no es de reajuste, la regla decide: «restaurar» reajusta al
      objetivo en ese cierre; «liquidar» recompra todo y espera en caja al siguiente reajuste;
      «ninguna» no hace nada.
    - Un fondo que pierde el 100% en un día desaparece de la cartera (peso 0) y se repone al
      siguiente reajuste. Si el patrimonio llega a 0 (ruina), ese día rinde −100% y al cierre
      empieza una cuenta nueva con el objetivo: la serie sirve para empezar en cualquier fecha, y
      quien la evalúa en una ventana aplica absorb() desde el inicio de la ventana (una ruina dentro
      de la ventana deja la riqueza a cero hasta el final).
    El coste de negociar se descuenta en el cierre en que se negocia (run_weights lo descuenta
    de la sesión siguiente)."""
    cols = list(target.columns)
    K = len(cols)
    idx = target.index
    Rv = R.reindex(idx)[cols].fillna(0.0).clip(lower=-1.0).to_numpy().tolist()
    T = target.shift(lag - 1).to_numpy(dtype=float)
    valid = ~np.isnan(T).any(axis=1)
    Tl = np.nan_to_num(T).tolist()
    rfv = rf.reindex(idx).fillna(0.0).to_numpy().tolist()
    finv = ((fin if fin is not None else rf).reindex(idx).fillna(0.0) + fin_extra / 252).to_numpy().tolist()
    fut = [c in futures for c in cols]
    bor = [0.0 if fut[k] else (borrow.get(c, 0.0) if isinstance(borrow, dict) else borrow) / 252
           for k, c in enumerate(cols)]
    mnt = [(maint or {}).get(c, 0.0) for c in cols]
    check = any(m > 0 for m in mnt)
    cst = [(fut_bps if fut[k] else cost_bps) / 1e4 for k in range(K)]
    reb = rebalance_days(idx, freq).tolist()
    hc = haircut / 252
    held = [0.0] * K
    out = np.full(len(idx), np.nan)
    flag = np.zeros(len(idx), bool)
    started = parked = False
    prev = None
    n_calls = n_reb = n_ruin = 0
    days = 0
    for i in range(len(idx)):
        if not started:
            if not valid[i]:
                continue
            started = True
        days += 1
        r = Rv[i]
        longs = shorts = ret = 0.0
        for k in range(K):
            h = held[k]
            if fut[k]:
                ret += h * (r[k] - finv[i])
            elif h > 0:
                ret += h * r[k]
                longs += h
            elif h < 0:
                ret += h * r[k] + h * bor[k]
                shorts -= h
        cash = 1.0 - longs
        if not rebate:
            ret += cash * rfv[i] if cash >= 0 else cash * finv[i]
        elif netting:
            c = cash + shorts
            ret += c * (rfv[i] - hc) if c >= 0 else c * finv[i]
        else:
            ret += (cash * rfv[i] if cash >= 0 else cash * finv[i]) + shorts * (rfv[i] - hc)
        tgt = Tl[i]
        if ret <= -1.0:                       # ruina: cuenta nueva al cierre con el objetivo
            out[i] = -1.0
            n_ruin += 1
            held, prev, parked = list(tgt), tgt, False
            continue
        g = 1.0 + ret
        held = [held[k] * (1.0 + r[k]) / g for k in range(K)]
        changed = prev is None or any(abs(a - b) > 1e-12 for a, b in zip(tgt, prev))
        prev = tgt
        do = reb[i] or changed
        if parked and not do:
            out[i] = ret
            continue
        breach = check and sum(mnt[k] * -held[k] for k in range(K) if held[k] < 0) > 1.0
        if breach:
            n_calls += 1
            flag[i] = True
            if not do:
                if margin_rule == "restaurar":
                    do = True
                elif margin_rule == "liquidar":
                    tc = sum(abs(held[k]) * cst[k] for k in range(K))
                    ret = g * (1.0 - tc) - 1.0
                    held = [0.0] * K
                    parked = True
                    out[i] = ret
                    continue
        if do:
            parked = False
            tc = sum(abs(tgt[k] - held[k]) * cst[k] for k in range(K))
            ret = g * (1.0 - tc) - 1.0
            held = list(tgt)
            n_reb += 1
        out[i] = ret
    years = max(days, 1) / 252
    return pd.Series(out, index=idx), {"incumplimientos_año": n_calls / years, "reajustes_año": n_reb / years,
                                       "incumplimientos": n_calls, "ruinas": n_ruin,
                                       "días_incumplimiento": pd.Series(flag, index=idx)}


# ───────────────────────────── (a) y (e): reglas de tendencia ─────────────────────────────

def signals(d: dict) -> dict:
    """Señales de la LRS con el índice de rentabilidad total (la original) y sus derivadas."""
    sx = r13.signal_index(d)
    sig = r13.above_sma(sx)
    sig_b = r13.above_sma(sx, band=r13.BAND)
    sma = sx.rolling(200).mean()
    falling = (sma < sma.shift(20)).astype(float).where(sma.shift(20).notna())
    below = (1 - sig)
    inv = (below * falling).where(below.notna() & falling.notna())
    return {"sig": sig, "below": below, "sig_b": sig_b, "below_b": 1 - sig_b, "falling_inv": inv}


def short3_name(freq: str, b: float, rule: str = "ninguna") -> str:
    s = f"LRS con corto del −3x, {FREQ_ES[freq]}, préstamo {pct_es(b, 0)}"
    return s if rule == "ninguna" else f"{s}, margin call: {rule}"


def trend_rules(d: dict, F: dict, curve: bool = True) -> tuple[dict, dict, dict, dict]:
    """(a) y (e) en un mercado. Devuelve (R, E, familia, info de margen). F: fondos por L."""
    S = signals(d)
    sig, below = S["sig"], S["below"]
    idx = d["u"].index
    zero = pd.Series(0.0, index=idx)
    one = pd.Series(1.0, index=idx)
    R, E, fam, info = {}, {}, {}, {}
    R[ONE], E[ONE] = d["u"], one
    R[THREE], E[THREE] = d["x3"], 3 * one
    R["LRS 1x (original)"], E["LRS 1x (original)"] = r13.run_weights(d, zero, w1=sig), sig
    R["LRS 2x (original)"], E["LRS 2x (original)"] = r13.run_weights(d, sig, x=r12.letf(d, 2.0)), 2 * sig
    R[ORIGINAL], E[ORIGINAL] = r13.run_weights(d, sig), 3 * sig
    for up, dn in ((3, -1), (3, -2), (3, -3), (2, -2)):
        k = f"LRS {up}x / {lx(dn)}"
        R[k] = run_legs(d, [(sig, F[up]), (below, F[dn])])
        E[k], fam[k] = up * sig + dn * below, FAM_INV
    k = "LRS 1x / −1x"
    R[k] = run_legs(d, [(sig, d["u"]), (below, F[-1])])
    E[k], fam[k] = sig - below, FAM_INV
    k = "LRS 3x / −3x + banda ±2%"
    R[k] = run_legs(d, [(S["sig_b"], F[3]), (S["below_b"], F[-3])])
    E[k], fam[k] = 3 * S["sig_b"] - 3 * S["below_b"], FAM_INV
    k = "LRS 3x / −3x si la media baja"
    inv = S["falling_inv"]
    R[k] = run_legs(d, [(sig.where(inv.notna()), F[3]), (inv, F[-3])])
    E[k], fam[k] = 3 * sig - 3 * inv, FAM_INV
    # la misma exposición alcista (+3) vendiendo en corto el −3x
    Rs, infos = short3_rules(d, F, sig)
    for k in Rs:
        R[k], info[k] = Rs[k], infos[k]
        E[k], fam[k] = 3 * sig, FAM_SHORT3
    # (e) 2x con el ETF 2x (SSO/QLD, real desde 2006) y la curva de L sintética
    k = "LRS 2x con el ETF 2x"
    R[k], E[k], fam[k] = run_legs(d, [(sig, F[2])]), 2 * sig, FAM_LEV
    if curve:
        for L in L_CURVE:
            k = f"LRS {lx(L)} sintética"
            R[k] = run_legs(d, [(sig, synthetic(d, L))])
            E[k], fam[k] = L * sig, FAM_LEV
        if d.get("px_real", False):     # solo el S&P 500 tiene un índice de precio real (^GSPC)
            sig_px = r13.above_sma(r13.signal_index(d, r13.PRICE))
            for L in L_CURVE:
                k = f"LRS {lx(L)} sintética · {PX_TAG}"
                R[k] = run_legs(d, [(sig_px, synthetic(d, L))])
                E[k], fam[k] = L * sig_px, FAM_LEV_PX
    return R, E, fam, info


SHORT3_GRID = ([("M", b, "ninguna") for b in SHORT_BORROW] + [("M", 0.02, "restaurar"), ("M", 0.02, "liquidar")]
               + [("D", b, "ninguna") for b in SHORT_BORROW])


def short3_rules(d: dict, F: dict, sig: pd.Series, rebate: bool = True) -> tuple[dict, dict]:
    """(a) LRS con corto del −3x: la exposición alcista de la LRS (+3) vendiendo en corto el −3x por
    el 100% del patrimonio, con la rejilla SHORT3_GRID. rebate: si el producto del corto cobra la
    letra − HAIRCUT (True) o nada (False, sensibilidad)."""
    tgt = pd.DataFrame({"inv": -sig})
    Rinv = pd.DataFrame({"inv": F[-3]})
    R, info = {}, {}
    for freq, b, rule in SHORT3_GRID:
        k = short3_name(freq, b, rule)
        R[k], info[k] = run_drift(tgt, Rinv, d["rf"], freq, borrow=b, maint={"inv": MAINT * 3},
                                  margin_rule=rule, rebate=rebate)
    return R, info


def curve_refs(d: dict) -> dict:
    """Comprar y mantener cada L de la curva (fondo sintético con el s y el gasto del 3x)."""
    return {f"{lx(L)} comprar y mantener (sintético)": synthetic(d, L) for L in L_CURVE}


def dead_fund_check(d: dict, L: float = 5.0, start=SP_FROM) -> pd.DataFrame:
    """La LRS L× con el fondo de sim.letf_returns (muerto para siempre tras su primer −100%) frente
    al fondo nuevo de synthetic, con las dos señales: con la original la regla llevaba el fondo el
    día del −100% (se arruina con los dos); con la de precio no lo llevaba, y el fondo muerto la
    deja invirtiendo en un fondo que rinde 0 (el error de la exploración previa)."""
    dead = r12.letf(dict(d), L)            # sim.letf_returns: queda a 0 tras el primer −100%
    fresh = synthetic(d, L)
    ruin = ruin_date(fresh)
    rows = {}
    for kind, lab in ((r13.SIGNAL, "rentabilidad total (original)"), (r13.PRICE, "precio (sensibilidad)")):
        if kind == r13.PRICE and not d.get("px_real", False):
            continue
        sig = r13.above_sma(r13.signal_index(d, kind))
        row = {"día del −100% del fondo": None if ruin is None else ruin.date(),
               "la regla llevaba el fondo ese día": None if ruin is None else
               ("sí" if sig.shift(LAG).loc[ruin] == 1 else "no")}
        for flab, x in (("CAGR con el fondo muerto (sim.letf_returns)", dead),
                        ("CAGR con un fondo nuevo (este módulo)", fresh)):
            row[flab] = metrics.cagr(absorb(run_legs(d, [(sig, x)])).loc[start:].dropna())
        rows[f"LRS {lx(L)}, señal de {lab}"] = row
    return pd.DataFrame(rows).T


# ───────────────────────────── (b), (c), (d): pares y lever-arb ─────────────────────────────

def pair_name(bull: str, bear: str, freq: str, b: float | None = None, rule: str = "ninguna",
              side: str = "corto") -> str:
    s = f"Par {side} {bull}+{bear}, {FREQ_ES[freq]}"
    if b is not None:
        s += f", préstamo {pct_es(b, 0)}"
    return s if rule == "ninguna" else f"{s}, margin call: {rule}"


def pair_rules(d: dict, P: dict, mkt: str, start=None, rebate: bool = True) -> tuple[dict, dict, dict]:
    """(b) par corto con la rejilla completa y (d) par largo, desde `start` (vivo desde entonces).
    rebate: remuneración del producto de los cortos (run_drift); el par largo no tiene cortos."""
    idx = d["u"].index
    live = pd.Series(np.where(idx >= pd.Timestamp(start or idx[0]), 1.0, np.nan), index=idx)
    R, fam, info = {}, {}, {}
    for bull, bear, L in PAIRS[mkt]:
        X = pd.DataFrame({bull: hybrid(d, P, bull), bear: hybrid(d, P, bear)})
        maint = {bull: MAINT * L, bear: MAINT * L}
        short = pd.DataFrame({bull: -0.5 * live, bear: -0.5 * live})
        long_ = pd.DataFrame({bull: 0.5 * live, bear: 0.5 * live})
        for freq in FREQS:
            for b in PAIR_BORROW:
                for rule in MARGIN_RULES:
                    if freq == "D" and rule != "ninguna":
                        continue            # a diario se vuelve al objetivo en cada cierre
                    k = pair_name(bull, bear, freq, b, rule)
                    R[k], info[k] = run_drift(short, X, d["rf"], freq, borrow=b, maint=maint, margin_rule=rule,
                                              rebate=rebate)
                    fam[k] = FAM_PAIR
            k = pair_name(bull, bear, freq, side="largo")
            R[k], info[k] = run_drift(long_, X, d["rf"], freq)
            fam[k] = FAM_LONG
    return R, fam, info


def arb_name(tk: str, L: int, freq: str, b: float, impl: str = "futuros", size: str = "") -> str:
    leg = f"−{tk}/{L} +1x" if size == "viable" else f"−{tk} +{L}x"
    s = f"Lever-arb {leg} ({impl}), {FREQ_ES[freq]}, préstamo {pct_es(b, 0)}"
    return s + ", margin call: liquidar" if size == "viable" else s


def arb_rules(d: dict, P: dict, mkt: str, start=None, rebate: bool = True) -> tuple[dict, dict, dict]:
    """(c) lever-arb: corto del ETF L× y +L veces el índice (futuros, o al contado con margen).
    rebate: remuneración del producto del corto (run_drift)."""
    idx = d["u"].index
    live = pd.Series(np.where(idx >= pd.Timestamp(start or idx[0]), 1.0, np.nan), index=idx)
    R, fam, info = {}, {}, {}
    for tk, L in ARB[mkt]:
        X = pd.DataFrame({tk: hybrid(d, P, tk), "índice": d["u"]})
        full = pd.DataFrame({tk: -1.0 * live, "índice": L * live})
        for freq in FREQS:
            for b in ARB_BORROW:
                k = arb_name(tk, L, freq, b)
                R[k], info[k] = run_drift(full, X, d["rf"], freq, borrow={tk: b}, maint={tk: MAINT * L},
                                          futures=("índice",), fin=d["f"], rebate=rebate)
                fam[k] = FAM_ARB
        viable = pd.DataFrame({tk: -1.0 / L * live, "índice": 1.0 * live})
        for freq in ("W", "M", "Q"):
            k = arb_name(tk, L, freq, 0.01, size="viable")
            R[k], info[k] = run_drift(viable, X, d["rf"], freq, borrow={tk: 0.01}, maint={tk: MAINT * L},
                                      margin_rule="liquidar", futures=("índice",), fin=d["f"], rebate=rebate)
            fam[k] = FAM_ARB
        for freq in ("D", "M"):
            k = arb_name(tk, L, freq, 0.01, impl="al contado con margen")
            R[k], info[k] = run_drift(full, X, d["rf"], freq, borrow={tk: 0.01}, maint={tk: MAINT * L},
                                      fin=d["f"], fin_extra=0.015, netting=True, rebate=rebate)
            fam[k] = FAM_ARB
    return R, fam, info


def fund_fee(P: dict, tk: str) -> float:
    """Lo que el fondo cobra por encima de su réplica: gasto + |L|·s (anual)."""
    p = P[tk]
    return p["er"] + abs(p["L"]) * p["s"]


# ───────────────────────────── mecanismo ─────────────────────────────

def period_key(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Etiqueta del periodo de reajuste de cada sesión (semana ISO, mes o trimestre)."""
    if freq == "W":
        iso = index.isocalendar()
        return iso.year.to_numpy("int64") * 100 + iso.week.to_numpy("int64")
    if freq == "M":
        return (index.year * 100 + index.month).to_numpy("int64")
    if freq == "Q":
        return (index.year * 10 + index.quarter).to_numpy("int64")
    raise ValueError(freq)


def period_paths(ex: pd.DataFrame, w: dict, x: pd.Series, freq: str, coef: float) -> pd.DataFrame:
    """P&L de camino de cada periodo de reajuste, sin costes ni intereses, con las rentabilidades en
    exceso de cada pata (fondo − letra; futuro: u − f) y pesos w fijados al inicio del periodo:
    camino = Σ_k w_k·(Π(1 + e_k) − 1) − (Π(1 + Σ_k w_k·e_k) − 1), la cartera que deriva durante el
    periodo menos la reajustada cada día. Su aproximación de segundo orden es
    coef·(Σx² − (Σx)²) con x = u − f (coef = L²/2 en el par corto; (L² − L)/2 en el lever-arb).
    Cada periodo empieza con el mismo peso: una pérdida mayor que el 100% en un periodo es una ruina."""
    cols = list(w)
    df = pd.concat([ex[cols], x.rename("x")], axis=1).dropna()
    wv = np.array([w[c] for c in cols])
    key = period_key(df.index, freq)
    lg = np.log1p(df[cols].clip(lower=-0.999999))
    daily = np.log1p(np.maximum(df[cols].to_numpy() @ wv, -0.999999))
    agg = pd.DataFrame(lg.to_numpy(), columns=cols, index=df.index)
    agg["__d"], agg["x"], agg["x2"], agg["__k"] = daily, df.x, df.x ** 2, key
    g = agg.groupby("__k", sort=False)
    per = g.sum()
    per.index = g.apply(lambda f: f.index[-1])
    drift = sum(w[c] * np.expm1(per[c]) for c in cols)
    out = pd.DataFrame({"camino": drift - np.expm1(per["__d"]), "aprox": coef * (per.x2 - per.x ** 2),
                        "x": per.x, "x2": per.x2, "deriva": drift})
    return out.iloc[1:-1]          # periodos incompletos al principio y al final


def path_mechanism(d: dict, P: dict, mkt: str, start) -> tuple[pd.DataFrame, dict]:
    """Camino por periodo del par corto (−½, −½) y del lever-arb (−1 fondo, +L futuros) frente a su
    aproximación, por época y en toda la muestra. Devuelve la tabla y los periodos de cada caso."""
    rf = d["rf"]
    x = (d["u"] - d["f"]).loc[start:]
    specs = []
    for bull, bear, L in PAIRS[mkt]:
        ex = pd.DataFrame({bull: hybrid(d, P, bull) - rf, bear: hybrid(d, P, bear) - rf}).loc[start:]
        specs.append((f"par corto {bull}+{bear}", L, L ** 2 / 2, ex, {bull: -0.5, bear: -0.5}))
    for tk, L in ARB[mkt]:
        ex = pd.DataFrame({tk: hybrid(d, P, tk) - rf, "futuro": d["u"] - d["f"]}).loc[start:]
        specs.append((f"lever-arb −{tk} +{L}x", L, (L ** 2 - L) / 2, ex, {tk: -1.0, "futuro": float(L)}))
    rows, periods = [], {}
    for name, L, coef, ex, w in specs:
        for fq in ("W", "M", "Q"):
            per = period_paths(ex, w, x, fq, coef)
            periods[(name, fq)] = per
            eras = dict(ERAS[mkt])
            eras["toda la muestra"] = (start, None)
            for era, (a, b) in eras.items():
                pe = per.loc[a:b]
                p, ap = pe.camino, pe.aprox
                fit = stats.ols_hac(p, ap.to_frame("aprox"), stats.NONOV_LAGS)
                rows.append({"mercado": mkt, "estrategia": name, "L": L, "reajuste": FREQ_ES[fq], "época": era,
                             "periodos": len(pe), "camino_anual": p.mean() * PER_YEAR[fq],
                             "aproximación_anual": ap.mean() * PER_YEAR[fq],
                             "decaimiento_cosechado_anual": coef * pe.x2.mean() * PER_YEAR[fq],
                             "pérdida_por_tendencia_anual": coef * (pe.x ** 2).mean() * PER_YEAR[fq],
                             "cociente_de_varianzas": float((pe.x ** 2).mean() / pe.x2.mean()),
                             "correlación": float(np.corrcoef(p, ap)[0, 1]),
                             "pendiente": float(fit.beta[1]), "t_pendiente_NW": float(fit.t[1]),
                             "peor_periodo": float(pe.deriva.min()), "periodos_positivos": float((p > 0).mean()),
                             "periodos_con_ruina": int((pe.deriva <= -1).sum())})
    return pd.DataFrame(rows), periods


def variance_ratio_test(d: dict, mkt: str, start) -> pd.DataFrame:
    """Cociente de varianzas de x = u − f a h sesiones con ventanas solapadas (una por sesión):
    VR(h) = media de (Σ_h x)² / media de Σ_h x². z = Σ_h x² − (Σ_h x)² es, por (L²/2), el P&L de
    camino de un par corto reajustado cada h sesiones; su media se contrasta frente a 0 con la
    regresión sobre la constante y errores de Hansen-Hodrick (h − 1 retardos; stats.hac_overlap)."""
    x = (d["u"] - d["f"]).loc[start:].dropna()
    rows = []
    eras = dict(ERAS[mkt])
    eras["toda la muestra"] = (start, None)
    for era, (a, b) in eras.items():
        xe = x.loc[a:b]
        for h in VR_H:
            s1 = xe.rolling(h).sum().shift(-(h - 1))
            s2 = (xe ** 2).rolling(h).sum().shift(-(h - 1))
            z = (s2 - s1 ** 2).dropna()
            fit = stats.ols_hac(z, pd.DataFrame(index=z.index), **stats.hac_overlap(h))
            rows.append({"mercado": mkt, "época": era, "h (sesiones)": h, "ventanas": len(z),
                         "VR(h)": float((s1 ** 2).mean() / s2.mean()),
                         "z medio anual (Σx² − (Σx)²)·252/h": float(z.mean() * 252 / h),
                         "t (HAC)": float(fit.t[0]), "estimador": fit.estimator})
    return pd.DataFrame(rows)


def iid_world(d: dict, mkt: str, start, rng) -> pd.DataFrame:
    """P&L de camino mensual (sin costes) del par corto ±L y del lever-arb en la historia (meses
    naturales) y en meses de T_IID sesiones sorteadas: días independientes y bootstrap
    estacionario de bloque 21. Teoría iid del par: −(L²/2)·T(T − 1)·μ² por mes."""
    x = (d["u"] - d["f"]).loc[start:].dropna()
    xv = x.to_numpy()
    months = [g.to_numpy() for _, g in x.groupby([x.index.year, x.index.month]) if len(g) >= 15]

    def pnl(X, L):
        gL, gm, g1 = (np.prod(1 + a * X, axis=1) for a in (L, -L, 1.0))
        return -0.5 * (gL - 1) - 0.5 * (gm - 1), -(gL - 1) + L * (g1 - 1)

    rows = []
    Xi = xv[rng.integers(0, len(xv), (N_IID, T_IID))]
    Xb = np.stack([xv[stats.stationary_indices(len(xv), 21, rng, T_IID)] for _ in range(N_IID // 10)])
    for L in (2, 3):
        hist = [np.array([pnl(m[None, :], L)[j][0] for m in months]) for j in (0, 1)]
        for j, kind in ((0, "par corto"), (1, "lever-arb")):
            for world, p in (("historia (meses naturales)", hist[j]), ("días independientes", pnl(Xi, L)[j]),
                             ("bootstrap estacionario (bloque 21)", pnl(Xb, L)[j])):
                q1, q5, q50 = np.percentile(p, [1, 5, 50])
                rows.append({"mercado": mkt, "L": L, "estrategia": kind, "mundo": world, "meses": len(p),
                             "media_anual": p.mean() * 12, "mediana_anual": q50 * 12,
                             "meses_positivos": (p > 0).mean(), "percentil_5_mensual": q5,
                             "percentil_1_mensual": q1, "peor_mes": p.min(),
                             "asimetría": float(pd.Series(p).skew())})
        mu = xv.mean()
        rows.append({"mercado": mkt, "L": L, "estrategia": "par corto", "mundo": "teoría iid: −(L²/2)·T(T−1)·μ²",
                     "media_anual": -(L ** 2 / 2) * T_IID * (T_IID - 1) * mu ** 2 * 12})
    return pd.DataFrame(rows)


def episode_table(R: dict, d: dict, mkt: str, keys: list) -> pd.DataFrame:
    rows = []
    for ep, (a, b) in EPISODES.items():
        u = d["u"].loc[a:b].dropna()
        if len(u) < 5 or (mkt == NDX and pd.Timestamp(a) < pd.Timestamp(NDX_FROM)):
            continue
        for k in keys:
            r = absorb(R[k].loc[a:b].dropna())
            if len(r) < 5:
                continue
            w = (1 + r).cumprod()
            rows.append({"mercado": mkt, "episodio": ep, "estrategia": k, "índice": float((1 + u).prod() - 1),
                         "resultado": float(w.iloc[-1] - 1), "caída_máx": float((w / w.cummax() - 1).min()),
                         "peor_día": float(r.min()), "ruina": "sí" if (r <= -1).any() else "no"})
    return pd.DataFrame(rows)


# ───────────────────────────── Kelly dentro y fuera de la tendencia ─────────────────────────────

def kelly_split(d: dict, start, end=None, rng=None, B_: int = B_KELLY) -> pd.DataFrame:
    """L* dentro de la tendencia (sesiones en que la LRS lleva el 3x: señal de t − 2 igual a 1),
    fuera y en todas, con la muestra de [start, end] (dentro de muestra, no operable):
    kelly.LSTAR_APROX = (μ − f − s)/σ² con IC90 por bootstrap estacionario de bloque 252, y
    kelly.LSTAR_EXACTO: el L constante que maximiza el crecimiento de esas sesiones con L/3 en el
    3x (con su comisión) y, por encima de 3, préstamo a f + s (tope L_MAX_KELLY; np.inf = censurado)."""
    rng = rng or np.random.default_rng(SEED)
    s = signals(d)["sig"].shift(LAG)
    df = pd.concat([d["u"], d["f"], d["rf"], d["x3"], s], axis=1,
                   keys=["u", "f", "rf", "x3", "s"]).loc[start:end].dropna()
    sp = d["spread"]
    X = df[["u", "f", "s"]].to_numpy()
    rows = {}
    for lab, cond in (("dentro de la tendencia", 1.0), ("fuera de la tendencia", 0.0), ("todas las sesiones", None)):
        def ls(a, cond=cond):
            m = a if cond is None else a[a[:, 2] == cond]
            return kelly.lstar_approx(m[:, 0], m[:, 1], sp)
        est, lo, hi = stats.bootstrap_ci(X, ls, BLOCK_KELLY, B_, rng)
        sub = df if cond is None else df[df.s == cond]
        rows[lab] = {"sesiones": len(sub) / len(df), "μ_anual": sub.u.mean() * 252,
                     "compuesta_anual": float(np.expm1(np.log1p(sub.u).mean() * 252)),
                     "σ_anual": sub.u.std() * np.sqrt(252), "μ − f anual": (sub.u - sub.f).mean() * 252,
                     kelly.LSTAR_APROX: est, "IC90_bajo": lo, "IC90_alto": hi,
                     kelly.LSTAR_EXACTO: kelly.lstar_exact(sub.x3, sub.rf, sub.f, sp, L_max=L_MAX_KELLY)}
    return pd.DataFrame(rows).T


def kelly_realtime(d: dict) -> pd.DataFrame:
    """L* aproximado en tiempo real (datos hasta t − 1, desde el año 10): sin condicionar, con la
    prima de kelly.premium, y condicionado a las sesiones dentro y fuera de la tendencia hasta t − 1.
    Es una descripción de lo que se habría estimado cada día, no una regla probada."""
    s = signals(d)["sig"].shift(LAG)
    x = (d["u"] - d["f"]).where(s.notna())
    u = d["u"].where(s.notna())
    sp = d["spread"]
    prem = kelly.premium(u, d["f"])
    var_all = (u ** 2).shift(1).expanding(kelly.MIN_OBS).mean() - u.shift(1).expanding(kelly.MIN_OBS).mean() ** 2
    out = {"todas las sesiones": (prem - sp) / (var_all * 252)}
    for lab, cond in (("dentro de la tendencia", 1.0), ("fuera de la tendencia", 0.0)):
        m = (s == cond).astype(float).where(s.notna())
        n = m.shift(1).expanding().sum()
        mx = (x * m).shift(1).expanding().sum() / n
        mu = (u * m).shift(1).expanding().sum() / n
        m2 = (u ** 2 * m).shift(1).expanding().sum() / n
        L = (mx * 252 - sp) / ((m2 - mu ** 2) * 252)
        out[lab] = L.where(n >= kelly.MIN_OBS * (0.25 if cond == 0.0 else 0.75))
    return pd.DataFrame(out)


# ───────────────────────────── estadística ─────────────────────────────

def perf(r: pd.Series, rf: pd.Series, a=None, b=None) -> dict:
    """Métricas en [a, b]; una ruina dentro de la ventana deja la riqueza a cero (absorb)."""
    r = absorb(r.loc[a:b].dropna())
    yr = (1 + r).groupby(r.index.year).prod() - 1
    mo = (1 + r).resample("ME").prod() - 1
    return {"desde": r.index[0].date(), "hasta": r.index[-1].date(), "CAGR": metrics.cagr(r),
            "vol": metrics.vol(r), "sharpe": metrics.sharpe(r, rf), "maxDD": metrics.max_dd(r),
            "peor_año": yr.min(), "peor_mes": mo.min()}


def trade_stats(e: pd.Series, a=None, b=None) -> dict:
    e = e.shift(LAG).loc[a:b].dropna()
    ch = e.diff().abs() > 1e-9
    return {"exposición_media": float(e.mean()), "cambios_año": float(ch.sum() / (len(e) / 252))}


def window_logs(R: dict, bench: pd.Series, a=None, b=None) -> pd.DataFrame:
    """log(1 + r) de cada regla y de la referencia («__b») en las sesiones comunes de [a, b], con la
    ruina absorbida desde el inicio de la ventana."""
    L = pd.DataFrame({k: v.loc[a:b] for k, v in R.items()})
    L["__b"] = bench.loc[a:b]
    L = L.dropna()
    return L.apply(lambda c: logr(absorb(c)))


def boot_family(R: dict, bench: pd.Series, a=None, b=None, rng=None, B_: int = B) -> pd.DataFrame:
    """Δg anual de cada regla frente a `bench` en la ventana común, con el mismo bootstrap
    estacionario (bloque BLOCK) para todas: IC90 percentil, p unilateral normal con el error del
    bootstrap (H0: Δg ≤ 0, como R19), Holm y Romano–Wolf (step-down, r19)."""
    from letf_lab.research import r19_census as r19
    rng = rng or np.random.default_rng(SEED)
    names = list(R)
    L = window_logs(R, bench, a, b)
    X = L[names].sub(L["__b"], axis=0).to_numpy()
    same = np.all(np.abs(X) < 1e-14, axis=0)
    X = X[:, ~same] if same.any() else X
    keep = [n for n, s in zip(names, same) if not s]
    m = X.mean(axis=0)
    draws = r19.boot_means(X, B_, BLOCK, rng)
    se = draws.std(axis=0, ddof=1)
    safe = np.where(se > 0, se, np.inf)
    t = m / safe
    from scipy.stats import norm
    p = norm.sf(t)
    lo, hi = np.percentile(draws, [5, 95], axis=0)
    rw = r19.romano_wolf(t, (draws - m) / safe)
    out = pd.DataFrame({"Δg": m * 252, "IC90_bajo": lo * 252, "IC90_alto": hi * 252, "t": t, "p": p,
                        "p_Holm": stats.holm(p), "p_RW": rw}, index=keep)
    out.attrs.update(desde=L.index[0].date(), hasta=L.index[-1].date(), sesiones=len(L))
    return out


def spa(R: dict, bench: pd.Series, a=None, b=None, rng=None) -> dict:
    """SPA de Hansen (2005, versión consistente) de la familia R frente a bench."""
    rng = rng or np.random.default_rng(SEED)
    L = window_logs(R, bench, a, b)
    names = list(R)
    X = L[names].sub(L["__b"], axis=0).to_numpy()
    keep = ~np.all(np.abs(X) < 1e-14, axis=0)
    X = X[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    res = stats.spa_test(X, BLOCK, B, rng)
    j = int(X.mean(axis=0).argmax())
    return {"reglas": X.shape[1], "mejor": names[j], "Δg_mejor": float(X.mean(axis=0)[j] * 252),
            "p_SPA_c": res["p_SPA_c"], "t_SPA": res["t_SPA"]}


def window_table(R: dict, E: dict, rf: pd.Series, refs: dict, a=None, b=None, rng=None, keys=None) -> pd.DataFrame:
    """Métricas, operativa y Δg frente a cada referencia (mismos remuestreos por referencia)."""
    keys = keys or list(R)
    rows = {k: {**perf(R[k], rf, a, b), **(trade_stats(E[k], a, b) if k in E else {})} for k in keys}
    t = pd.DataFrame(rows).T
    for rn, ref in refs.items():
        cand = {k: R[k] for k in keys if R[k] is not ref}
        bt = boot_family(cand, ref, a, b, rng).reindex(t.index)
        t[f"Δg_vs_{rn}"] = bt["Δg"]
        t[f"p_vs_{rn}"] = bt["p"]
        t[f"IC90_bajo_vs_{rn}"] = bt["IC90_bajo"]
        t[f"IC90_alto_vs_{rn}"] = bt["IC90_alto"]
        t[f"IC90_vs_{rn}"] = [f"[{num_es(lo * 100, 1)}; {num_es(hi * 100, 1)}]" if np.isfinite(lo) else ""
                              for lo, hi in zip(bt["IC90_bajo"], bt["IC90_alto"])]
    return t


# ───────────────────────────── cálculo completo ─────────────────────────────

def ndx_market(nd: dict, k: int) -> dict:
    """Reconstrucción k del Nasdaq-100 con el 3x del laboratorio (como r13.ndx_imputations)."""
    u = nd["draws"][k]
    return r12.make(u, history.rf_long(u.index), "TQQQ", r12.real_letf("TQQQ"), eval_from=NDX_FROM)


def build_market(d: dict, P: dict, mkt: str) -> dict:
    """Todas las reglas de un mercado: tendencia (a, e), pares (b, d) y lever-arb (c)."""
    F = market_funds(d, P, mkt)
    R, E, fam, info = trend_rules(d, F)
    start = d["u"].first_valid_index() if mkt == SP else pd.Timestamp(NDX_FROM)
    Rp, famp, infop = pair_rules(d, P, mkt, start)
    Ra, fama, infoa = arb_rules(d, P, mkt, start)
    R.update(Rp), R.update(Ra), fam.update(famp), fam.update(fama), info.update(infop), info.update(infoa)
    return {"R": R, "E": E, "fam": fam, "info": info, "F": F, "refs": curve_refs(d)}


CENSUS_START = "1939-04-03"   # primera sesión de la ventana común del censo de R19 (y de R25)


def census_p_lrs_1x(R: dict, start=CENSUS_START) -> dict:
    """p de la hipótesis principal del artículo 2 (LRS 3x original frente al 1x) como lo calcula el
    censo de R19: r19.boot_test sobre el exceso logarítmico en su ventana, con su generador recién
    creado (sus remuestreos solo dependen del número de sesiones y del generador)."""
    from letf_lab.research import r19_census as r19
    both = pd.concat([R[ORIGINAL], R[ONE]], axis=1, keys=["lrs", "one"]).loc[start:].dropna()
    a = r19.active(both[["lrs"]], both["one"])
    bt = r19.boot_test(a.to_numpy(float), r19.B_BOOT, r19.BLOCK, np.random.default_rng(r19.SEED))
    return {"p": float(bt["p"][0]), "desde": both.index[0], "hasta": both.index[-1]}


def no_rebate_market(d: dict, P: dict, mkt: str) -> dict:
    """Las reglas que venden en corto (LRS con corto del −3x, par corto y lever-arb) con el producto de
    los cortos SIN remunerar (run_drift con rebate=False), con los mismos nombres que build_market."""
    F = market_funds(d, P, mkt)
    Rs, infos = short3_rules(d, F, signals(d)["sig"], rebate=False)
    start = d["u"].first_valid_index() if mkt == SP else pd.Timestamp(NDX_FROM)
    Rp, famp, infop = pair_rules(d, P, mkt, start, rebate=False)
    Ra, fama, infoa = arb_rules(d, P, mkt, start, rebate=False)
    R = {**Rs, **{k: v for k, v in Rp.items() if famp[k] == FAM_PAIR}, **Ra}
    fam = {**{k: FAM_SHORT3 for k in Rs}, **{k: f for k, f in famp.items() if f == FAM_PAIR}, **fama}
    info = {**infos, **infop, **infoa}
    return {"R": R, "fam": fam, "info": {k: info[k] for k in R}}


REBATES = ("letra − 0,5%", "sin remuneración")


def rebate_keys(mkt: str) -> dict:
    """Configuraciones que el informe cita, con su referencia: {nombre: «LRS» o «letras»}."""
    out = {short3_name(fq, b): "LRS" for fq in ("D", "M") for b in SHORT_BORROW}
    for bull, bear, L in PAIRS[mkt]:
        for fq in ("W", "M"):
            out[pair_name(bull, bear, fq, 0.02)] = "letras"
        out[pair_name(bull, bear, "W", 0.02, "restaurar")] = "letras"
    for tk, L in ARB[mkt]:
        for fq in ("D", "W"):
            out[arb_name(tk, L, fq, 0.01)] = "letras"
        out[arb_name(tk, L, "W", 0.01, size="viable")] = "letras"
    return out


def rebate_sensitivity(D: dict, P: dict, M: dict, NR: dict, rng) -> dict:
    """El supuesto del producto de los cortos: letra − 0,5% (el del informe) frente a nada. Por
    mercado: (1) Δg y p de las configuraciones citadas con los dos supuestos, emparejadas (mismos
    remuestreos); (2) SPA y Romano–Wolf de la familia de tendencia frente a la LRS original y de la
    familia con préstamo realista frente a las letras, con el supuesto «sin remuneración»; (3) el
    préstamo que anula la ventaja del corto diario del −3x y de los pares y lever-arb citados."""
    out = {}
    for mkt, wins in ((SP, SP_WINDOWS), (NDX, NDX_WINDOWS)):
        d, R, fam = D[mkt], M[mkt]["R"], M[mkt]["fam"]
        Rn = NR[mkt]["R"]
        rf = d["rf"]
        main_w = "1938–2026" if mkt == SP else "1985–2026"
        full_w = "1928–2026" if mkt == SP else "1985–2026"
        rows = []
        for ref_lab, wlist in (("LRS", (main_w, "ETFs reales")), ("letras", (full_w, "2000–2026", "ETFs reales"))):
            keys = [k for k, r in rebate_keys(mkt).items() if r == ref_lab]
            bench = R[ORIGINAL] if ref_lab == "LRS" else rf
            for w in wlist:
                a, b = wins[w]
                both = {**{f"{k}|{REBATES[0]}": R[k] for k in keys}, **{f"{k}|{REBATES[1]}": Rn[k] for k in keys}}
                bt = boot_family(both, bench, a, b, rng)
                for k in keys:
                    for rb in REBATES:
                        j = f"{k}|{rb}"
                        rows.append({"mercado": mkt, "regla": k, "frente a": ref_lab, "ventana": w,
                                     "producto del corto": rb,
                                     "CAGR": perf(R[k] if rb == REBATES[0] else Rn[k], rf, a, b)["CAGR"],
                                     "Δg": bt.loc[j, "Δg"] if j in bt.index else 0.0,
                                     "p": bt.loc[j, "p"] if j in bt.index else np.nan})
        tab = pd.DataFrame(rows)
        # familias con el supuesto «sin remuneración»: las reglas con cortos cambian, el resto no
        Rmix = {**R, **Rn}
        tr = trend_subsets(fam)["tendencia: todas"]
        a, b = wins[main_w]
        fams = [{"familia": "tendencia: todas", "ventana": main_w, "frente a": "LRS 3x (original)",
                 **family_summary({k: Rmix[k] for k in tr}, R[ORIGINAL], a, b, rng)}]
        rw_tr = boot_family({k: Rmix[k] for k in tr}, R[ORIGINAL], a, b, rng)
        real = neutral_subsets(fam)[REALISTIC]
        rw_nt = {}
        for w in (full_w, "2000–2026", "ETFs reales"):
            a, b = wins[w]
            fams.append({"familia": REALISTIC, "ventana": w, "frente a": "letras",
                         **family_summary({k: Rmix[k] for k in real}, rf, a, b, rng)})
            if w == full_w:
                rw_nt = boot_family({k: Rmix[k] for k in real}, rf, a, b, rng)
        # préstamo que anula la ventaja, sin remuneración (toda la muestra)
        a, b = wins[main_w]
        L = window_logs({k: Rn[k] for k in Rn if fam.get(k) == FAM_SHORT3}, R[ORIGINAL], a, b)
        t3 = pd.DataFrame({"Δg_vs_letras": (L.drop(columns="__b").sub(L["__b"], axis=0)).mean() * 252})
        sn = {("LRS con corto del −3x", fq): {bb: short3_name(fq, bb) for bb in SHORT_BORROW} for fq in ("D", "M")}
        a, b = wins[full_w]
        Ln = window_logs({k: Rn[k] for k in Rn if fam.get(k) in (FAM_PAIR, FAM_ARB)}, rf, a, b)
        tn = pd.DataFrame({"Δg_vs_letras": (Ln.drop(columns="__b").sub(Ln["__b"], axis=0)).mean() * 252})
        pn = {(f"par corto {bull}+{bear}", fq): {bb: pair_name(bull, bear, fq, bb) for bb in PAIR_BORROW}
              for bull, bear, L_ in PAIRS[mkt] for fq in ("W", "M")}
        an = {(f"lever-arb −{tk} +{L_}x (futuros)", fq): {bb: arb_name(tk, L_, fq, bb) for bb in ARB_BORROW}
              for tk, L_ in ARB[mkt] for fq in ("D", "W")}
        out[mkt] = {"tabla": tab, "familias": pd.DataFrame(fams), "rw_tendencia": rw_tr, "rw_neutral": rw_nt,
                    "préstamo_nulo_corto3": breakeven_borrow(t3, sn),
                    "préstamo_nulo_neutral": pd.concat([breakeven_borrow(tn, pn), breakeven_borrow(tn, an)])}
    return out


def census_series(D: dict | None = None, start=SP_FROM, with_meta: bool = False):
    """Rentabilidad diaria NETA de cada regla de R22 probada en el S&P 500, para añadirla al censo de
    R19: {«R22 · nombre»: serie sobre D['S&P 500']['u'].index, el índice del censo}.

    Todas las configuraciones de las familias de R22 en el S&P 500 ((a) inverso bajo la media, (a) LRS
    con corto del −3x, (e) apalancamiento de la regla, (e) apalancamiento con la señal de precio,
    (b) par corto, (d) par largo, (c) lever-arb), con el protocolo común (señal de t, orden al cierre de
    t+1, 5 pb por unidad). Deja fuera las referencias que ya están en R13 (1x, 3x, LRS 1x/2x/3x
    originales) y las series idénticas a una de ellas (la LRS 2x sintética es la LRS 2x (original)).
    Las configuraciones que dan la misma serie entre sí (una regla de margen que nunca se activa) se
    devuelven todas: r19.dedupe las cuenta una vez.

    Ruina: las series internas empiezan una cuenta nueva tras un día de −100% (run_drift y los
    fondos de synthetic), para poder evaluarlas desde cualquier fecha. Aquí se devuelven con NaN
    antes de `start` y con la ruina absorbida desde `start` (tras un −100%, rentabilidad 0 hasta el
    final), como las vería quien empezara ese día. Por defecto start = 1938-02-04, antes del inicio
    de la ventana del censo (1939): así una ruina de 1932 no cuenta y la de octubre de 1987 sí. Si el
    censo usa otra ventana, conviene pasar su primera sesión. with_meta=True devuelve además un
    DataFrame con la familia, si vende en corto y si es operable con ETFs cotizados (≤ 3x)."""
    D = D or r13.load()
    d = D[SP]
    P = fund_params(D)
    res = build_market(d, P, SP)
    R, fam = res["R"], res["fam"]
    idx = d["u"].index
    ref13 = {k: R[k].reindex(idx).to_numpy(float) for k in (ONE, THREE, "LRS 1x (original)", "LRS 2x (original)",
                                                                ORIGINAL)}
    out, meta = {}, {}
    for k, f in fam.items():
        raw = R[k].reindex(idx)
        if any(np.allclose(raw.to_numpy(float), v, equal_nan=True, atol=1e-14) for v in ref13.values()):
            continue
        r = raw.copy()
        if start is not None:
            r[r.index < pd.Timestamp(start)] = np.nan
            live = r.notna()
            r[live] = absorb(r[live])
        name = f"R22 · {k}"
        out[name] = r
        L = curve_L(k)
        meta[name] = {"familia": f, "vende_en_corto": f in NEEDS_SHORT,
                      "operable_con_ETFs": not (L is not None and L > 3),
                      "ruina_desde_start": (lambda x: "" if x is None else str(x.date()))(ruin_date(raw.loc[start:]))}
    if not with_meta:
        return out
    meta = pd.DataFrame(meta).T
    for c in ("vende_en_corto", "operable_con_ETFs"):
        meta[c] = meta[c].astype(bool)
    return out, meta


def variant_count(fams: dict, distinct_: dict | None = None) -> pd.DataFrame:
    """Configuraciones simuladas por mercado y familia (las referencias no cuentan) y, con distinct_,
    cuántas dan series distintas entre sí y de las referencias de R13."""
    rows = {}
    for mkt, fam in fams.items():
        for f in FAMS:
            rows.setdefault(f, {})[f"{mkt}: configuraciones"] = sum(1 for v in fam.values() if v == f)
            if distinct_ is not None:
                rows[f][f"{mkt}: series distintas"] = distinct_[mkt][f]
    t = pd.DataFrame(rows).T
    t.loc["total"] = t.sum()
    return t.astype(int)


# R13 (reports/r13_lrs/README.md, S&P 500 desde 1938-02-04): CAGR y caída máxima publicados
R13_PUBLISHED = {ONE: (0.115, -0.552), THREE: (0.149, -0.984), ORIGINAL: (0.174, -0.800)}


def sanity(R: dict, rf: pd.Series) -> pd.DataFrame:
    """Comprobación de partida: la LRS 3x (original), el 1x y el 3x desde 1938 como en R13."""
    rows = {}
    for k, (c, dd) in R13_PUBLISHED.items():
        p = perf(R[k], rf, SP_FROM)
        rows[k] = {"CAGR (R22)": p["CAGR"], "CAGR (R13)": c, "caída_máx (R22)": p["maxDD"], "caída_máx (R13)": dd,
                   "coincide": "sí" if (round(p["CAGR"], 3) == c and round(p["maxDD"], 3) == dd) else "NO"}
    return pd.DataFrame(rows).T


def breakeven_borrow(t: pd.DataFrame, names: dict) -> pd.DataFrame:
    """Préstamo anual que deja a 0 el Δg frente a la letra, interpolando en la rejilla.
    names = {(etiqueta, frecuencia): {préstamo: nombre de la regla}}."""
    rows = {}
    for (lab, fq), grid in names.items():
        bs = sorted(grid)
        g = np.array([t.loc[grid[b], "Δg_vs_letras"] for b in bs])
        if g[0] <= 0:
            be = "no gana ni sin préstamo"
        elif g[-1] > 0:
            be = f"> {pct_es(bs[-1], 0)}"
        else:
            j = int(np.argmax(g <= 0))
            be = pct_es(bs[j - 1] + (bs[j] - bs[j - 1]) * g[j - 1] / (g[j - 1] - g[j]), 1)
        rows[(lab, FREQ_ES[fq])] = {**{f"Δg con préstamo {pct_es(b, 0)}": v for b, v in zip(bs, g)},
                                    "préstamo que lo anula": be}
    out = pd.DataFrame(rows).T
    out.index.names = ["estrategia", "reajuste"]
    return out


def ndx_reconstructions(nd: dict, P: dict, n: int = r13.N_NDX) -> pd.DataFrame:
    """Reglas de tendencia (a, e) del Nasdaq-100 en n reconstrucciones del periodo previo a 1985-10
    (solo calientan la media): CAGR, caída máxima y Δg frente a la LRS original y al 1x desde 1985-10."""
    rows = []
    for k in range(n):
        d = ndx_market(nd, k)
        R, _, fam, _ = trend_rules(d, market_funds(d, P, NDX))
        L = pd.DataFrame({key: r.loc[NDX_FROM:] for key, r in R.items()}).dropna().apply(lambda c: logr(absorb(c)))
        for key in [ONE, THREE, "LRS 1x (original)", ORIGINAL] + list(fam):
            r = absorb(R[key].loc[NDX_FROM:].dropna())
            rows.append({"reconstrucción": k, "regla": key, "CAGR": metrics.cagr(r), "maxDD": metrics.max_dd(r),
                         "Δg_vs_LRS": float((L[key] - L[ORIGINAL]).mean() * 252),
                         "Δg_vs_1x": float((L[key] - L[ONE]).mean() * 252)})
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df.regla))
    g = df.groupby("regla").agg(CAGR_mediana=("CAGR", "median"), CAGR_mín=("CAGR", "min"), CAGR_máx=("CAGR", "max"),
                                maxDD_mediana=("maxDD", "median"), Δg_vs_LRS_mediana=("Δg_vs_LRS", "median"),
                                Δg_vs_LRS_mín=("Δg_vs_LRS", "min"), Δg_vs_LRS_máx=("Δg_vs_LRS", "max"),
                                Δg_vs_1x_mediana=("Δg_vs_1x", "median"))
    return g.loc[order]


def family_summary(R: dict, bench: pd.Series, a, b, rng) -> dict:
    """SPA de la familia y resumen de Holm y Romano–Wolf al 5% (mismos datos, otros remuestreos)."""
    s = spa(R, bench, a, b, rng)
    bt = boot_family(R, bench, a, b, rng)
    best = bt["Δg"].idxmax()
    rw = bt.index[bt.p_RW < 0.05].tolist()
    return {**s, "p_nominal_mejor": float(bt.loc[best, "p"]), "p_RW_mejor": float(bt.loc[best, "p_RW"]),
            "rechazos_Holm_5%": int((bt.p_Holm < 0.05).sum()), "rechazos_RW_5%": len(rw),
            "sobreviven_RW": "; ".join(rw) if rw else "ninguna"}


def breach_rate(info: dict | None, r: pd.Series, a=None, b=None) -> float:
    """Incumplimientos del mantenimiento por año en [a, b] (NaN si la regla no tiene cortos)."""
    if not info or "días_incumplimiento" not in info:
        return np.nan
    live = r.loc[a:b].dropna().index
    return float(info["días_incumplimiento"].reindex(live).sum() / (len(live) / 252))


def curve_L(k: str) -> float | None:
    """L de una regla de la curva («LRS 2,5x sintética…»), o None."""
    if "sintética" not in k:
        return None
    return float(k.split("LRS ")[1].split("x")[0].replace(",", "."))


def realistic_borrow(k: str) -> bool:
    """Préstamo realista: al menos 2% en el par corto y 1% en el lever-arb, y sin la versión al contado
    con margen, cuya financiación es optimista (el producto del corto paga la compra del índice sin
    haircut y, antes de 2010, cobra de hecho la diferencia f − rf)."""
    if "al contado con margen" in k:
        return False
    b = float(k.split("préstamo ")[1].split("%")[0].replace(",", ".")) / 100
    return b >= (0.02 if k.startswith("Par corto") else 0.01)


def trend_subsets(fam: dict) -> dict:
    tr = [k for k, f in fam.items() if f in TREND_FAMS]
    no_short = [k for k in tr if fam[k] != FAM_SHORT3]
    original = [k for k in no_short if fam[k] != FAM_LEV_PX]
    operable = [k for k in original if curve_L(k) is None or curve_L(k) <= 3]
    return {"tendencia: todas": tr, "tendencia: sin vender en corto": no_short,
            "tendencia: señal original, sin cortos": original,
            "tendencia: operables (≤ 3x, sin cortos)": operable}


def neutral_subsets(fam: dict) -> dict:
    nk = [k for k, f in fam.items() if f in NEUTRAL_FAMS]
    real = [k for k in nk if realistic_borrow(k)]
    return {"par corto y lever-arb: todas": nk,
            REALISTIC: real,
            "préstamo realista, sin reajuste diario": [k for k in real if ", diario" not in k]}


def distinct(R: dict, keys: list, extra: dict | None = None) -> int:
    """Series distintas entre `keys` (y distintas de las de `extra`): las configuraciones que dan la
    misma serie (p. ej. una regla de margen que nunca se activa) cuentan una vez."""
    seen = [np.nan_to_num(v.to_numpy(float), nan=9.0) for v in (extra or {}).values()]
    n = 0
    for k in keys:
        v = np.nan_to_num(R[k].to_numpy(float), nan=9.0)
        if any(v.shape == u.shape and np.max(np.abs(v - u)) < 1e-12 for u in seen):
            continue
        seen.append(v)
        n += 1
    return n


def analyse(D: dict, P: dict, M: dict, rng) -> dict:
    """Tablas del informe a partir de las series de build_market."""
    A = {}
    nd = D.get("_ndx")
    for mkt, wins in ((SP, SP_WINDOWS), (NDX, NDX_WINDOWS)):
        d, R, E, fam = D[mkt], M[mkt]["R"], M[mkt]["E"], M[mkt]["fam"]
        rf = d["rf"]
        R_all = {**R, TBILL: rf}
        refs13 = [ONE, THREE, "LRS 1x (original)", "LRS 2x (original)", ORIGINAL]
        trend = refs13 + [k for k, f in fam.items() if f in TREND_FAMS]
        neutral = [k for k, f in fam.items() if f in (FAM_PAIR, FAM_LONG, FAM_ARB)]
        full_w = "1928–2026" if mkt == SP else "1985–2026"
        main_w = "1938–2026" if mkt == SP else "1985–2026"
        A[mkt] = {"tendencia": {}, "neutral": {}}
        for w, (a, b) in wins.items():
            t = window_table(R, E, rf, {"LRS": R[ORIGINAL], "1x": R[ONE]}, a, b, rng, trend)
            t["incumplimientos_año"] = [breach_rate(M[mkt]["info"].get(k), R[k], a, b) for k in t.index]
            t["ruina"] = [(lambda x: "" if x is None else str(x.date()))(ruin_date(R[k].loc[a:b])) for k in t.index]
            A[mkt]["tendencia"][w] = t
            t = window_table(R_all, {}, rf, {"letras": rf}, a, b, rng, neutral)
            t["incumplimientos_año"] = [breach_rate(M[mkt]["info"].get(k), R[k], a, b) for k in t.index]
            t["ruina"] = [(lambda x: "" if x is None else str(x.date()))(ruin_date(R[k].loc[a:b])) for k in t.index]
            A[mkt]["neutral"][w] = t
        # préstamo que anula la ventaja (toda la muestra)
        full = A[mkt]["neutral"][full_w]
        pn = {(f"par corto {bull}+{bear}", fq): {b: pair_name(bull, bear, fq, b) for b in PAIR_BORROW}
              for bull, bear, L in PAIRS[mkt] for fq in FREQS}
        an = {(f"lever-arb −{tk} +{L}x (futuros)", fq): {b: arb_name(tk, L, fq, b) for b in ARB_BORROW}
              for tk, L in ARB[mkt] for fq in FREQS}
        A[mkt]["préstamo_nulo_pares"] = breakeven_borrow(full, pn)
        A[mkt]["préstamo_nulo_arb"] = breakeven_borrow(full, an)
        tt = A[mkt]["tendencia"][main_w]
        sn = {("LRS con corto del −3x", fq): {b: short3_name(fq, b) for b in SHORT_BORROW} for fq in ("D", "M")}
        A[mkt]["préstamo_nulo_corto3"] = breakeven_borrow(tt.rename(columns={"Δg_vs_LRS": "Δg_vs_letras"}), sn)
        start = d["u"].first_valid_index() if mkt == SP else pd.Timestamp(NDX_FROM)
        A[mkt]["mecanismo"], A[mkt]["periodos"] = path_mechanism(d, P, mkt, start)
        A[mkt]["cociente_varianzas"] = variance_ratio_test(d, mkt, start)
        A[mkt]["iid"] = iid_world(d, mkt, start, np.random.default_rng(SEED + (0 if mkt == SP else 1)))
        ep_keys = []
        for bull, bear, L in PAIRS[mkt]:
            ep_keys += [pair_name(bull, bear, fq, 0.02, rule) for fq in ("W", "M") for rule in MARGIN_RULES]
            ep_keys += [pair_name(bull, bear, "Q", 0.02)]
        for tk, L in ARB[mkt]:
            ep_keys += [arb_name(tk, L, fq, 0.01) for fq in ("D", "W", "M")] + [arb_name(tk, L, "W", 0.01, size="viable")]
        A[mkt]["episodios"] = episode_table(R, d, mkt, ep_keys)
        # 2x frente a 3x
        rows = {}
        for w, (a, b) in wins.items():
            bt = boot_family({ORIGINAL: R[ORIGINAL]}, R["LRS 2x con el ETF 2x"], a, b, rng)
            p2, p3 = perf(R["LRS 2x con el ETF 2x"], rf, a, b), perf(R[ORIGINAL], rf, a, b)
            rows[w] = {"CAGR 2x (ETF 2x)": p2["CAGR"], "CAGR 3x (original)": p3["CAGR"],
                       "caída_máx 2x": p2["maxDD"], "caída_máx 3x": p3["maxDD"],
                       "Δg 3x − 2x": bt["Δg"].iloc[0], "IC90_bajo": bt["IC90_bajo"].iloc[0],
                       "IC90_alto": bt["IC90_alto"].iloc[0], "p (3x > 2x)": bt["p"].iloc[0]}
        A[mkt]["2x_3x"] = pd.DataFrame(rows).T
        # Kelly
        kw = ({"1938–2026": (SP_FROM, None), "1928–2026": (RULE_FROM, None), "2000–2026": ("2000-01-01", None)}
              if mkt == SP else {"1985–2026": (NDX_FROM, None), "2000–2026": ("2000-01-01", None)})
        A[mkt]["kelly"] = pd.concat({w: kelly_split(d, a, b, np.random.default_rng(SEED + 7)) for w, (a, b) in kw.items()})
        # curva de L: CAGR de la regla y de comprar y mantener
        refs = M[mkt]["refs"]
        crows = []
        for w, (a, b) in wins.items():
            for L in L_CURVE:
                kr, kb = f"LRS {lx(L)} sintética", f"{lx(L)} comprar y mantener (sintético)"
                pr = perf(R[kr], rf, a, b)
                row = {"ventana": w, "L": L, "CAGR LRS": pr["CAGR"], "caída_máx LRS": pr["maxDD"],
                       "ruina LRS": (lambda x: "" if x is None else str(x.date()))(ruin_date(R[kr].loc[a:b])),
                       "CAGR comprar y mantener": perf(refs[kb], rf, a, b)["CAGR"]}
                kp = f"{kr} · {PX_TAG}"
                if kp in R:
                    row["CAGR LRS (señal de precio)"] = perf(R[kp], rf, a, b)["CAGR"]
                crows.append(row)
        A[mkt]["curva"] = pd.DataFrame(crows)
        # contrastes por la búsqueda
        fam_rows = []
        ts, ns = trend_subsets(fam), neutral_subsets(fam)
        tw = ({"1938–2026": wins["1938–2026"], "1928–2026": wins["1928–2026"]} if mkt == SP
              else {"1985–2026": wins["1985–2026"]})
        post = ({"2016–2026": wins["2016–2026"], "2017–2026": wins["2017–2026"]} if mkt == SP
                else {"2016–2026": wins["2016–2026"]})
        for w, (a, b) in {**tw, **post}.items():
            for sub, keys in ts.items():
                if w in post and sub != "tendencia: todas":
                    continue
                for bn, bench in (("LRS 3x (original)", R[ORIGINAL]), ("1x", R[ONE])):
                    fam_rows.append({"mercado": mkt, "familia": sub, "ventana": w, "frente a": bn,
                                     **family_summary({k: R[k] for k in keys}, bench, a, b, rng)})
        nw = {full_w: wins[full_w], "2000–2026": wins["2000–2026"], "ETFs reales": wins["ETFs reales"]}
        for w, (a, b) in nw.items():
            for sub, keys in ns.items():
                fam_rows.append({"mercado": mkt, "familia": sub, "ventana": w, "frente a": "letras",
                                 **family_summary({k: R[k] for k in keys}, rf, a, b, rng)})
        if mkt == SP:
            for bn, bench in (("LRS 3x (original)", R[ORIGINAL]), ("1x", R[ONE])):
                fam_rows.append({"mercado": mkt, "familia": "todas las de R22", "ventana": "1938–2026", "frente a": bn,
                                 **family_summary({k: R[k] for k in fam}, bench, *wins["1938–2026"], rng)})
        A[mkt]["familias"] = pd.DataFrame(fam_rows)
        tr_keys = ts["tendencia: todas"]
        A[mkt]["rw_tendencia"] = boot_family({k: R[k] for k in tr_keys}, R[ORIGINAL], *wins[main_w], rng)
        A[mkt]["rw_tendencia_1x"] = boot_family({k: R[k] for k in tr_keys}, R[ONE], *wins[main_w], rng)
        real = ns[REALISTIC]
        A[mkt]["rw_neutral"] = boot_family({k: R[k] for k in real}, rf, *wins[full_w], rng)
        A[mkt]["rw_neutral_2000"] = boot_family({k: R[k] for k in real}, rf, *wins["2000–2026"], rng)
        # recuento: configuraciones y series distintas (y distintas de las de R13)
        ref = {k: R[k] for k in refs13}
        A[mkt]["distintas"] = {f: distinct(R, [k for k, v in fam.items() if v == f], ref) for f in FAMS}
    A["ndx_reconstrucciones"] = ndx_reconstructions(nd, P) if nd is not None else None
    A["fondo_muerto"] = dead_fund_check(D[SP], 5.0)
    A["kelly_tiempo_real"] = kelly_realtime(D[SP])
    return A


def compute(D: dict | None = None) -> dict:
    """Todo el cálculo de R22 (sin informe). Semillas fijas."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    D = D or r13.load()
    P = fund_params(D)
    out = {"P": P, "validación": validation_table(D, P)}
    M = {mkt: build_market(D[mkt], P, mkt) for mkt in (SP, NDX)}
    out["M"] = M
    out["sanidad"] = sanity(M[SP]["R"], D[SP]["rf"])
    if (out["sanidad"]["coincide"] != "sí").any():
        raise RuntimeError("R22 no reproduce la LRS 3x (original) de R13:\n" + out["sanidad"].to_string())
    out["A"] = analyse(D, P, M, np.random.default_rng(SEED))
    out["recuento"] = variant_count({m: M[m]["fam"] for m in M}, {m: out["A"][m]["distintas"] for m in M})
    # sensibilidad al producto de los cortos (con su propio generador: no cambia nada de lo anterior)
    NR = {mkt: no_rebate_market(D[mkt], P, mkt) for mkt in (SP, NDX)}
    out["NR"] = NR
    out["remuneración"] = rebate_sensitivity(D, P, M, NR, np.random.default_rng(SEED + 50))
    out["censo_p"] = census_p_lrs_1x(M[SP]["R"])
    return out


# ───────────────────────────── figuras ─────────────────────────────

def _style(ax):
    ax.grid(True, color="#e3e2de", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _pp_axis(ax, axis: str = "y"):
    f = FuncFormatter(lambda v, _: num_es(v, 1 if abs(v) < 10 and v != int(v) else 0))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def fig_validation(D: dict, P: dict, rep: Report):
    """ETFs inversos reales frente a sus sintéticos desde REAL_FROM."""
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 8.4), sharex=True)
    for ax, mkt in zip(axes, (SP, NDX)):
        d = D[mkt]
        for j, L in enumerate((-1, -2, -3)):
            tk = FUNDS[mkt][L]
            p = P[tk]
            df = pd.concat([p["real"], synthetic(d, L, p["s"], p["er"])], axis=1,
                           keys=["real", "sint"]).loc[REAL_FROM:].dropna()
            w = (1 + df).cumprod() * 10_000
            ax.semilogy(w.real, color=PAL[j], lw=2.0)
            ax.semilogy(w.sint, color=INK, lw=0.9, ls="--")
            ax.annotate(f"{tk} ({lx(L)})", (w.index[-1], w.real.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=11, color=INK)
        ax.set(title=f"{mkt}: ETFs inversos reales y sintéticos", ylabel="Valor de 10.000 $ (log)")
        ax.set_xlim(right=pd.Timestamp("2029-09-30"))
        _style(ax)
    h = [plt.Line2D([], [], color=GREY, lw=2.0), plt.Line2D([], [], color=INK, lw=0.9, ls="--")]
    legend_band(fig, h, ["ETF real", "sintético (s calibrado con la vida del ETF)"], ncol=2)
    rep.fig(fig, "validacion_inversos", "Figura: ETFs inversos reales (línea continua) y su versión sintética "
            f"(discontinua) desde {REAL_FROM}; las dos líneas casi se superponen")


def fig_inverse(R: dict, rep: Report, start=SP_FROM):
    keys = [(ONE, INK), (ORIGINAL, PAL[0]), ("LRS 3x / −1x", PAL[1]), ("LRS 3x / −3x", PAL[2])]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 8.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    hs, ls = [], []
    for k, col in keys:
        r = absorb(R[k].loc[start:].dropna())
        w = (1 + r).cumprod() * 10_000
        h, = axes[0].semilogy(w, color=col, lw=1.5)
        axes[1].plot((w / w.cummax() - 1) * 100, color=col, lw=0.9)
        hs.append(h)
        ls.append(f"{k}: {pct_es(metrics.cagr(r))}/año")
    axes[0].set(ylabel="Valor de 10.000 $ (log)")
    axes[1].set(ylabel="Caída desde máximos (%)")
    for ax in axes:
        _style(ax)
    fig.suptitle("S&P 500 desde 1938: letras o inverso bajo la media de 200 sesiones")
    legend_band(fig, hs, ls, ncol=2)
    comma_fig(fig)
    rep.fig(fig, "inverso_bajo_la_media", "Figura: 10.000 dólares en la LRS 3x original (letras bajo la media) y en "
            "las variantes que pasan a un inverso −1x o −3x bajo la media, con el 1x como referencia")


def fig_regimes(A: dict, rep: Report):
    """μ, σ y L* del índice dentro y fuera de la tendencia (sesiones con la señal ejecutada)."""
    cases = [(SP, "1938–2026"), (NDX, "1985–2026")]
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 10.0))
    x = np.arange(len(cases))
    wbar = 0.36
    for ax, col, lab in zip(axes, ("μ_anual", "σ_anual", kelly.LSTAR_APROX),
                            ("Media de las rentabilidades diarias del índice ×252 (%)", "Volatilidad anual del índice (%)",
                             f"{kelly.LSTAR_APROX} (μ − f − s)/σ²")):
        for j, (st, c) in enumerate((("dentro de la tendencia", PAL[0]), ("fuera de la tendencia", PAL[1]))):
            vals = np.array([A[m]["kelly"].loc[(w, st), col] for m, w in cases], float)
            scale = 1 if col == kelly.LSTAR_APROX else 100
            pos = x + (j - 0.5) * wbar
            kw = {}
            if col == kelly.LSTAR_APROX:
                lo = np.array([A[m]["kelly"].loc[(w, st), "IC90_bajo"] for m, w in cases], float)
                hi = np.array([A[m]["kelly"].loc[(w, st), "IC90_alto"] for m, w in cases], float)
                kw = {"yerr": [vals - lo, hi - vals], "capsize": 4, "error_kw": {"lw": 1, "ecolor": INK}}
            ax.bar(pos, vals * scale, wbar * 0.92, color=c, label=st, **kw)
            for xi, v in zip(pos, vals * scale):
                if col == kelly.LSTAR_APROX:     # junto a la barra de error, no encima
                    ax.annotate(num_es(v, 1), (xi + wbar * 0.1, v), xytext=(3, 2), textcoords="offset points",
                                ha="left", va="bottom", fontsize=11, color=INK)
                else:
                    ax.annotate(num_es(v, 1), (xi, v), xytext=(0, 3 if v >= 0 else -12), textcoords="offset points",
                                ha="center", fontsize=11, color=INK)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xlim(-0.6, len(cases) - 0.4)
        if col == kelly.LSTAR_APROX:
            ax.axhline(3, color=GREY, lw=0.8, ls=":")
            ax.annotate("3x", (-0.58, 3), xytext=(0, 2), textcoords="offset points", va="bottom", fontsize=11,
                        color=GREY)
        ax.set_xticks(x, [f"{m} ({w})" for m, w in cases])
        ax.set_title(lab)
        _style(ax)
        ax.grid(False, axis="x")
    h, lab_ = axes[0].get_legend_handles_labels()
    fig.suptitle("El índice dentro y fuera de la tendencia (media de 200 sesiones)")
    legend_band(fig, h, lab_, ncol=2)
    comma_fig(fig)
    rep.fig(fig, "dentro_fuera_de_la_tendencia", "Figura: media aritmética de las rentabilidades diarias (×252; la "
            "compuesta es menor, sobre todo fuera, donde la volatilidad es mayor), volatilidad y L* de Kelly del índice "
            "en las sesiones en que la LRS está dentro (sobre la media) y fuera; en L*, el intervalo del 90%")


def fig_pair_formula(A: dict, rep: Report):
    cases = [(SP, "par corto UPRO+SPXU"), (NDX, "par corto TQQQ+SQQQ")]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 10.5))
    handles = {}
    for ax, (mkt, name) in zip(axes, cases):
        per = A[mkt]["periodos"][(name, "M")]
        for j, (lab, (a, b)) in enumerate((("hasta 1999", (None, "1999-12-31")), ("2000–2026", ("2000-01-01", None)))):
            pe = per.loc[a:b]
            handles[lab] = ax.scatter(pe.aprox * 100, pe.camino * 100, s=14, color=PAL[j], alpha=0.8, lw=0)
        lo_, hi_ = (np.r_[per.aprox, per.camino] * 100).min(), (np.r_[per.aprox, per.camino] * 100).max()
        pad = 0.05 * (hi_ - lo_)
        ax.plot([lo_ - pad, hi_ + pad], [lo_ - pad, hi_ + pad], color=GREY, lw=0.9, ls="--")
        ax.set(xlim=(lo_ - pad, hi_ + pad), ylim=(lo_ - pad, hi_ + pad),
               title=f"{mkt}: {name.replace('par corto ', '')} reajustado cada mes",
               xlabel="(L²/2)·(Σx² − (Σx)²) en el mes (% del capital)",
               ylabel="P&L de camino simulado (% del capital)")
        c = A[mkt]["mecanismo"]
        r = c[(c.estrategia == name) & (c.reajuste == "mensual") & (c.época == "toda la muestra")].iloc[0]
        ax.annotate(f"correlación {num_es(r.correlación, 3)}\npendiente {num_es(r.pendiente, 2)}",
                    (0.03, 0.96), xycoords="axes fraction", va="top", fontsize=11, color=INK)
        _style(ax)
    fig.suptitle("Par corto 3x: P&L mensual de camino frente a su aproximación")
    legend_band(fig, list(handles.values()), [f"meses {k}" for k in handles], ncol=2, markerscale=2)
    comma_fig(fig)
    rep.fig(fig, "par_corto_formula", "Figura: cada punto es un mes; en horizontal, la aproximación de segundo orden "
            "(L²/2)·(Σx² − (Σx)²) con x = u − f, y en vertical el P&L de camino del par corto sin costes; "
            "la línea discontinua es la igualdad")


def fig_pair_eras(A: dict, rep: Report):
    cases = [(SP, ("UPRO", "SPXU"), ["1928–1937", "1938–1973", "1974–1999", "2000–2026"]),
             (NDX, ("TQQQ", "SQQQ"), ["1985–1999", "2000–2026"])]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 9.4), gridspec_kw={"height_ratios": [4, 2.6]})
    for ax, (mkt, (bull, bear), wins) in zip(axes, cases):
        x = np.arange(len(wins))
        for j, fq in enumerate(("W", "M")):
            k = pair_name(bull, bear, fq, 0.02)
            vals, lo, hi = [], [], []
            for w in wins:
                t = A[mkt]["neutral"][w]
                vals.append(t.loc[k, "Δg_vs_letras"] * 100)
                lo.append(t.loc[k, "IC90_bajo_vs_letras"] * 100)
                hi.append(t.loc[k, "IC90_alto_vs_letras"] * 100)
            vals, lo, hi = map(np.array, (vals, lo, hi))
            pos = x + (j - 0.5) * 0.36
            ax.bar(pos, vals, 0.33, color=PAL[j], label=f"reajuste {FREQ_ES[fq]}, préstamo del 2%",
                   yerr=[vals - lo, hi - vals], capsize=3, error_kw={"lw": 0.9, "ecolor": INK})
        # el cociente de varianzas de los meses naturales (el del mecanismo del par mensual y el del texto)
        mc = A[mkt]["mecanismo"]
        labels = []
        for w in wins:
            row = mc[(mc.estrategia == f"par corto {bull}+{bear}") & (mc.reajuste == "mensual") & (mc["época"] == w)]
            labels.append(f"{w}\nVR mensual = {num_es(row['cociente_de_varianzas'].iloc[0], 2)}" if len(row) else w)
        ax.set_xticks(x, labels)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set(title=f"{mkt}: {bull}+{bear}", ylabel="Δg frente a las letras (pp/año)")
        _style(ax)
        ax.grid(False, axis="x")
    h, lab = axes[0].get_legend_handles_labels()
    fig.suptitle("Par corto 3x por época: diferencia de crecimiento con las letras")
    legend_band(fig, h, lab, ncol=2)
    comma_fig(fig)
    rep.fig(fig, "par_corto_epocas", "Figura: crecimiento logarítmico anual del par corto 3x menos el de las letras, "
            "por época, con su intervalo del 90% (bootstrap estacionario); bajo cada época, el cociente de "
            "varianzas de los meses naturales, (Σx)² medio entre Σx² medio (VR < 1: el mercado revierte dentro del "
            "mes y el par gana). Es el de la tabla del mecanismo; el de ventanas solapadas de 21 sesiones, en la "
            "tabla siguiente, difiere un poco")


def fig_pair_wealth(M: dict, D: dict, rep: Report):
    cases = [(SP, ("UPRO", "SPXU"), [("W", None), ("M", None), ("M", "restaurar")], None),
             (NDX, ("TQQQ", "SQQQ"), [("W", None), ("M", None), ("W", "restaurar")], NDX_FROM)]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 9.6))
    for ax, (mkt, (bull, bear), cfg, start) in zip(axes, cases):
        R, d = M[mkt]["R"], D[mkt]
        start = start or d["u"].first_valid_index()
        for j, (fq, rule) in enumerate(cfg):
            k = pair_name(bull, bear, fq, 0.02, rule or "ninguna")
            r = absorb(R[k].loc[start:].dropna())
            lab = f"{FREQ_ES[fq]}" + (f", margin call: {rule}" if rule else "")
            ax.semilogy((1 + r).cumprod() * 10_000, color=PAL[j], lw=1.3, label=f"{bull}+{bear} {lab}")
        tb = d["rf"].loc[start:]
        ax.semilogy((1 + tb).cumprod() * 10_000, color=INK, lw=1.0, ls="--", label="letras")
        for ep, (a, b) in EPISODES.items():
            if pd.Timestamp(a) >= pd.Timestamp(start):
                ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#d9d8d4", alpha=0.5, lw=0)
        ax.set(title=f"{mkt}: par corto 3x con un 2% de préstamo", ylabel="Valor de 10.000 $ (log)")
        ax.legend(loc="upper left", fontsize=10.5, frameon=False)
        _style(ax)
    rep.fig(fig, "par_corto_riqueza", "Figura: 10.000 dólares en el par corto 3x (vender a la vez el 3x y el −3x "
            "del mismo índice) con distintos reajustes, frente a las letras; en gris, los episodios de crisis "
            "de la tabla de episodios")


def fig_lever_arb(A: dict, rep: Report):
    cases = [(SP, "UPRO", 3, "1928–2026"), (NDX, "TQQQ", 3, "1985–2026")]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 9.0), sharex=True)
    for ax, (mkt, tk, L, w) in zip(axes, cases):
        t = A[mkt]["neutral"][w]
        bs = np.array(ARB_BORROW) * 100
        for j, fq in enumerate(("D", "W", "M")):
            ys = [t.loc[arb_name(tk, L, fq, b), "Δg_vs_letras"] * 100 for b in ARB_BORROW]
            ax.plot(bs, ys, color=PAL[j], lw=2, marker="o", ms=5, label=f"cobertura {FREQ_FEM[fq]}")
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set(title=f"{mkt} ({w}): corto de {tk} y +{L}x el índice en futuros",
               ylabel="Δg frente a las letras (pp/año)")
        _style(ax)
    axes[-1].set_xlabel("Coste anual del préstamo de títulos del ETF (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.suptitle("Lever-arb: lo que queda según el préstamo y la frecuencia de cobertura")
    legend_band(fig, h, lab, ncol=3)
    comma_fig(fig)
    rep.fig(fig, "lever_arb", "Figura: crecimiento anual del lever-arb (corto del ETF 3x y tres veces el índice en "
            "futuros) menos el de las letras, según el coste del préstamo de títulos y la frecuencia de la cobertura; "
            "la cobertura trimestral queda fuera (se arruina en 1932 en el S&P 500)")


def fig_curve(A: dict, rep: Report):
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 10.4))
    handles = {}
    for ax, (mkt, w) in zip(axes, ((SP, "1938–2026"), (NDX, "1985–2026"))):
        c = A[mkt]["curva"]
        c = c[c.ventana == w].set_index("L")
        ok = c["ruina LRS"] == ""
        handles["LRS con la señal original"], = ax.plot(c.index[ok], c.loc[ok, "CAGR LRS"] * 100, color=PAL[0], lw=2,
                                                        marker="o", ms=5)
        if "CAGR LRS (señal de precio)" in c:
            handles["LRS con la señal de precio (sensibilidad; solo S&P 500)"], = ax.plot(
                c.index, c["CAGR LRS (señal de precio)"] * 100, color=PAL[1], lw=1.6, ls="--", marker="s", ms=4)
        bh = c["CAGR comprar y mantener"] > -1
        handles["comprar y mantener L×"], = ax.plot(c.index[bh], c.loc[bh, "CAGR comprar y mantener"] * 100,
                                                     color=GREY, lw=1.4, marker="^", ms=4)
        k = A[mkt]["kelly"].loc[(w, "dentro de la tendencia")]
        for val, lab, ls in ((k[kelly.LSTAR_EXACTO], "L* exacto dentro de la tendencia", ":"),
                             (k[kelly.LSTAR_APROX], "L* aproximado dentro de la tendencia", "-.")):
            if np.isfinite(val):
                handles[lab] = ax.axvline(val, color=INK, lw=1.0, ls=ls)
        lo = ax.get_ylim()[0]
        if (~ok).any():
            dates = sorted(set(c.loc[~ok, "ruina LRS"]))
            handles[f"ruina de la LRS con la señal original ({', '.join(dmy(x) for x in dates)})"] = ax.scatter(
                c.index[~ok], [lo] * int((~ok).sum()), marker="x", s=70, color=PAL[0], zorder=5, clip_on=False)
        ax.set(title=f"{mkt} ({w}); L* dentro: exacto {kelly.fmt_lstar(k[kelly.LSTAR_EXACTO], 1, L_MAX_KELLY).replace('.', ',')}, "
                     f"aproximado {num_es(k[kelly.LSTAR_APROX], 1)}",
               xlabel="Apalancamiento L del fondo sintético", ylabel="Rentabilidad anual compuesta (%)")
        _style(ax)
    fig.suptitle("La regla de Gayed y Bilello con cada apalancamiento (dentro de muestra)")
    legend_band(fig, list(handles.values()), list(handles), ncol=1)
    comma_fig(fig)
    rep.fig(fig, "curva_apalancamiento", "Figura: rentabilidad anual de la LRS con un fondo sintético de L× (fondo "
            "nuevo tras un día de −100% que la regla no llevaba) y del mismo L× comprado y mantenido; las líneas "
            "verticales marcan el L* de Kelly de las sesiones dentro de la tendencia, estimado con toda la muestra")


def fig_kelly_rt(A: dict, rep: Report):
    kt = A["kelly_tiempo_real"].loc["1943":]
    fig, ax = plt.subplots(figsize=(FIG_W, 5.4))
    for j, (col, lab) in enumerate((("dentro de la tendencia", "dentro de la tendencia"),
                                    ("fuera de la tendencia", "fuera de la tendencia"),
                                    ("todas las sesiones", "sin condicionar (prima media hasta el día anterior)"))):
        ax.plot(kt[col], color=PAL[j] if j < 2 else INK, lw=1.4, label=lab)
    ax.axhline(3, color=GREY, lw=0.9, ls=":")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.annotate("3x", (kt.index[0], 3), xytext=(2, 3), textcoords="offset points", va="bottom", color=GREY)
    ax.set(title="S&P 500: L* aproximado con los datos disponibles cada día", ylabel="L* = (μ − f − s)/σ²")
    _style(ax)
    legend_band(fig, *ax.get_legend_handles_labels(), where="bottom", ncol=2)
    comma_fig(fig)
    rep.fig(fig, "kelly_tiempo_real", "Figura: L* de Kelly aproximado estimado cada día con las sesiones anteriores "
            "(desde el año 10), dentro y fuera de la tendencia y sin condicionar; es lo que se habría estimado, "
            "no una regla probada")


# ───────────────────────────── informe ─────────────────────────────

def fp(p: float) -> str:
    """p-valor para el texto: «< 0,001» o tres decimales."""
    if p is None or not np.isfinite(p):
        return "—"
    return "< 0,001" if p < 0.001 else num_es(p, 3)


def spp(x: float, d: int = 1) -> str:
    """Puntos porcentuales con signo: «+1,3 pp», «−0,4 pp»."""
    return ("+" if x > 0 else "") + num_es(100 * x, d) + " pp"


def dmy(x) -> str:
    return pd.Timestamp(x).strftime("%-d-%-m-%Y")


class Tables:
    """Tablas numeradas con título. El CSV guarda los valores sin redondear; el markdown, con las
    columnas numéricas redondeadas (enteros sin decimales), celdas vacías en lugar de NaN y los
    índices múltiples como columnas."""

    def __init__(self, rep: Report):
        self.rep, self.n = rep, 0

    def __call__(self, title: str, df: pd.DataFrame, pct: set | None = None, digits: int = 2,
                 index: bool = True):
        from letf_lab.report import PCT_COLS, fmt
        self.n += 1
        self.rep.p(f"**Tabla {self.n}. {title}**")
        raw = df
        dv = df.copy()
        if isinstance(dv.index, pd.MultiIndex):
            dv, index = dv.reset_index(), False
        pct = set(pct or ()) | PCT_COLS
        for c in dv.columns:
            num = pd.to_numeric(dv[c], errors="coerce")
            if num.notna().sum() != dv[c].notna().sum() or not num.notna().any() or dv[c].dtype == bool:
                continue
            if c in pct:
                dv[c] = num
            elif np.allclose(num.dropna(), num.dropna().round()):
                dv[c] = num.map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
            else:
                dv[c] = num.map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
        self.rep.lines += [fmt(dv, pct, digits).to_markdown(index=index), ""]
        raw.to_csv(self.rep.dir / f"tabla_{self.n:02d}.csv")
        return self.n


def peq(p: float) -> str:
    """«= 0,034» o «< 0,001», para escribir «p {peq(x)}»."""
    if p is None or not np.isfinite(p):
        return "= —"
    return "< 0,001" if p < 0.001 else "= " + num_es(p, 3)


TREND_COLS = {"CAGR": "CAGR", "vol": "vol", "maxDD": "maxDD", "peor_año": "peor_año",
              "exposición_media": "exposición media", "cambios_año": "cambios/año",
              "Δg_vs_LRS": "Δg vs LRS (pp)", "IC90_vs_LRS": "IC90 vs LRS (pp)", "p_vs_LRS": "p vs LRS",
              "Δg_vs_1x": "Δg vs 1x (pp)", "p_vs_1x": "p vs 1x"}
NEUTRAL_COLS = {"CAGR": "CAGR", "vol": "vol", "maxDD": "maxDD", "peor_mes": "peor_mes",
                "Δg_vs_letras": "Δg vs letras (pp)", "IC90_vs_letras": "IC90 (pp)", "p_vs_letras": "p",
                "incumplimientos_año": "incumplimientos/año", "ruina": "ruina"}
PCT = {"CAGR", "vol", "maxDD", "peor_año", "peor_mes"}


def view(t: pd.DataFrame, keys: list, cols: dict, extra: dict | None = None) -> pd.DataFrame:
    """Columnas elegidas, Δg en puntos porcentuales y nombres cortos."""
    cols = {**cols, **(extra or {})}
    df = t.loc[[k for k in keys if k in t.index], [c for c in cols if c in t.columns]].copy()
    for c in df.columns:
        if c.startswith("Δg"):
            df[c] = df[c].astype(float) * 100
    return df.rename(columns=cols)


def be_txt(be: str) -> str:
    """El préstamo que anula una ventaja (breakeven_borrow) en una frase."""
    if be.startswith("no gana"):
        return "no gana ni con un préstamo nulo"
    if be.startswith(">"):
        return f"sigue ganando con cualquier préstamo de la rejilla ({be})"
    return f"se anula con un préstamo del {be}"


def pivot_dg(A: dict, mkt: str, keys: list, windows: list, part: str = "tendencia", ref: str = "LRS") -> pd.DataFrame:
    """Δg (pp) frente a la referencia por ventana, con * si p < 0,05 (nominal)."""
    out = {}
    for w in windows:
        t = A[mkt][part][w]
        out[w] = [f"{num_es(t.loc[k, f'Δg_vs_{ref}'] * 100, 1)}{'*' if t.loc[k, f'p_vs_{ref}'] < 0.05 else ''}"
                  for k in keys]
    return pd.DataFrame(out, index=keys)


def run(C: dict | None = None):
    """Informe de R22. C: resultado de compute() ya calculado (si no, se calcula)."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rep = Report("r22_inverse", "R22 — Inversos, pares y el supuesto arbitraje")
    for f in rep.dir.glob("tabla_*.csv"):       # las tablas se numeran por orden: sin huérfanas de otra versión
        f.unlink()
    rep.p(__doc__.split("\n", 2)[2])
    D = r13.load()
    C = C or compute(D)
    A, M, P = C["A"], C["M"], C["P"]
    tab = Tables(rep)
    S, N = A[SP], A[NDX]
    T = S["tendencia"]["1938–2026"]
    TN = N["tendencia"]["1985–2026"]
    NS, NN = S["neutral"]["1928–2026"], N["neutral"]["1985–2026"]
    ks_in, ks_out = ("1938–2026", "dentro de la tendencia"), ("1938–2026", "fuera de la tendencia")
    K = S["kelly"]
    rec = A["ndx_reconstrucciones"]
    inv_keys = [k for k, f in M[SP]["fam"].items() if f == FAM_INV]
    sh_keys = [k for k, f in M[SP]["fam"].items() if f == FAM_SHORT3]
    fams = C["recuento"]

    # ── números del resumen ──
    pos = [(w, k) for w in SP_WINDOWS for k in inv_keys if S["tendencia"][w].loc[k, "Δg_vs_LRS"] > 0]
    n_pos = len(pos)
    pos_w = sorted({w for w, _ in pos})
    pos_sig = sum(1 for w, k in pos if S["tendencia"][w].loc[k, "p_vs_LRS"] < 0.05)
    n_all = len(SP_WINDOWS) * len(inv_keys)
    n_pos_n = sum(int(N["tendencia"][w].loc[k, "Δg_vs_LRS"] > 0) for w in NDX_WINDOWS for k in inv_keys)
    d_short = short3_name("D", 0.02)
    m_short = short3_name("M", 0.02)
    be3 = S["préstamo_nulo_corto3"].loc[("LRS con corto del −3x", "diario"), "préstamo que lo anula"]
    rw3 = S["rw_tendencia"]
    up_m, up_w, up_q = (pair_name("UPRO", "SPXU", fq, 0.02) for fq in ("M", "W", "Q"))
    tq_w = pair_name("TQQQ", "SQQQ", "W", 0.02)
    tq_wr = pair_name("TQQQ", "SQQQ", "W", 0.02, "restaurar")
    mech = S["mecanismo"]

    def mrow(mc, name, fq, era):
        return mc[(mc.estrategia == name) & (mc.reajuste == fq) & (mc["época"] == era)].iloc[0]

    iid_sp = S["iid"].set_index(["L", "estrategia", "mundo"])
    theo = "teoría iid: −(L²/2)·T(T−1)·μ²"
    arb_d_u, arb_w_u, arb_v_u = (arb_name("UPRO", 3, "D", 0.01), arb_name("UPRO", 3, "W", 0.01),
                                 arb_name("UPRO", 3, "W", 0.01, size="viable"))
    arb_d_t, arb_w_t, arb_v_t = (arb_name("TQQQ", 3, "D", 0.01), arb_name("TQQQ", 3, "W", 0.01),
                                 arb_name("TQQQ", 3, "W", 0.01, size="viable"))
    be_arb_s, be_arb_n = S["préstamo_nulo_arb"], N["préstamo_nulo_arb"]
    be_pair_s, be_pair_n = S["préstamo_nulo_pares"], N["préstamo_nulo_pares"]
    x23s, x23n = S["2x_3x"], N["2x_3x"]
    cs = S["curva"][S["curva"].ventana == "1938–2026"].set_index("L")
    cn = N["curva"][N["curva"].ventana == "1985–2026"].set_index("L")
    ok = cs["ruina LRS"] == ""
    Lbest = float(cs.loc[ok, "CAGR LRS"].idxmax())
    Lruin = float(cs.index[~ok].min()) if (~ok).any() else np.nan
    Lpx = float(cs["CAGR LRS (señal de precio)"].idxmax())
    Lbn = float(cn["CAGR LRS"].idxmax())
    fm = A["fondo_muerto"]
    famS = S["familias"].set_index(["familia", "ventana", "frente a"])
    famN = N["familias"].set_index(["familia", "ventana", "frente a"])
    rwn_s, rwn_n = S["rw_neutral"], N["rw_neutral"]
    RB = C["remuneración"]
    NRB = REBATES[1]

    def rbv(mkt, k, w, col="Δg", rb=NRB):
        """Valor de la sensibilidad al producto del corto: regla k, ventana w, supuesto rb."""
        t = RB[mkt]["tabla"]
        return float(t[(t.regla == k) & (t.ventana == w) & (t["producto del corto"] == rb)][col].iloc[0])

    rw3_nr = RB[SP]["rw_tendencia"]
    rwn_s_nr, rwn_n_nr = RB[SP]["rw_neutral"], RB[NDX]["rw_neutral"]
    surv_s_nr = rwn_s_nr.index[rwn_s_nr.p_RW < 0.05].tolist()
    surv_n_nr = rwn_n_nr.index[rwn_n_nr.p_RW < 0.05].tolist()
    be3_nr = RB[SP]["préstamo_nulo_corto3"].loc[("LRS con corto del −3x", "diario"), "préstamo que lo anula"]
    d_nr = rbv(SP, d_short, "1938–2026")
    d_nr_ok = d_nr > 0 and rw3_nr.loc[d_short, "p_RW"] < 0.05

    rep.h("Qué sale")
    rep.p("\n".join([
        f"1. **(a) Pasar a un inverso bajo la media empeora la LRS original.** En el S&P 500 desde 1938, la "
        f"3x/−1x rinde un {pct_es(T.loc['LRS 3x / −1x', 'CAGR'])} anual frente al {pct_es(T.loc[ORIGINAL, 'CAGR'])} "
        f"de la original (Δg {spp(T.loc['LRS 3x / −1x', 'Δg_vs_LRS'])}/año, IC90 {T.loc['LRS 3x / −1x', 'IC90_vs_LRS']}), "
        f"y la 3x/−3x un {pct_es(T.loc['LRS 3x / −3x', 'CAGR'])} con una caída máxima del "
        f"{pct_es(T.loc['LRS 3x / −3x', 'maxDD'])}. Crece más que la original en {n_pos} de {n_all} combinaciones de "
        f"regla y ventana del S&P 500 ({'todas en ' + ', '.join(pos_w) if n_pos else 'ninguna'}"
        f"{', ninguna con p < 0,05' if n_pos and not pos_sig else ''}) y en {n_pos_n} de {len(NDX_WINDOWS) * len(inv_keys)} del Nasdaq-100, donde la "
        f"3x/−3x pierde un {pct_es(-rec.loc['LRS 3x / −3x', 'CAGR_mediana'])} anual (mediana de las "
        f"{r13.N_NDX} reconstrucciones). Bajo la media el índice sigue subiendo: la media de sus rentabilidades "
        f"diarias equivale a un {pct_es(K.loc[ks_out, 'μ_anual'])} anual desde 1938 (media aritmética ×252; "
        f"compuesta, un {pct_es(K.loc[ks_out, 'compuesta_anual'])}), con más volatilidad "
        f"({pct_es(K.loc[ks_out, 'σ_anual'])} frente a {pct_es(K.loc[ks_in, 'σ_anual'])} dentro): el L* de Kelly "
        f"fuera de la tendencia es {num_es(K.loc[ks_out, kelly.LSTAR_APROX], 2)} (IC90 "
        f"{num_es(K.loc[ks_out, 'IC90_bajo'], 1)} a {num_es(K.loc[ks_out, 'IC90_alto'], 1)}), no negativo.",
        f"2. **Vender en corto el −3x en lugar de comprar el 3x** da la misma exposición alcista. Con reajuste "
        f"mensual y sin regla de margen se arruina el {dmy(T.loc[m_short, 'ruina'])}: la regla estaba dentro ese "
        f"día y el corto había crecido desde el reajuste. Con reajuste diario y un 2% de préstamo suma "
        f"{spp(T.loc[d_short, 'Δg_vs_LRS'], 2)}/año, que es la diferencia entre la comisión que cobra el inverso y "
        f"la que paga el 3x, y se anula con un préstamo del {be3}. Es la única regla de tendencia que sobrevive a "
        f"Romano–Wolf frente a la original (p_RW {peq(rw3.loc[d_short, 'p_RW'])}), pero incumple el mantenimiento "
        f"de FINRA al cierre {num_es(T.loc[d_short, 'incumplimientos_año'], 1)} veces al año y depende de un "
        f"préstamo de títulos que no está en los datos. Además supone que el producto del corto cobra la letra − 0,5%: "
        f"sin remunerarlo (lo habitual en una cuenta minorista) queda en {spp(d_nr, 2)}/año frente a la original "
        f"(p {peq(rbv(SP, d_short, '1938–2026', 'p'))}, p_RW {peq(rw3_nr.loc[d_short, 'p_RW'])})"
        + (": la ventaja sobrevive también sin remuneración." if d_nr_ok else
           ": la ventaja de implementación depende de ese supuesto y desaparece sin él."),
        f"3. **(b) El par corto no es un arbitraje.** Su P&L de camino en cada periodo es ≈ (L²/2)·(Σx² − (Σx)²) "
        f"(correlación {num_es(mrow(mech, 'par corto UPRO+SPXU', 'mensual', 'toda la muestra').correlación, 3)} con "
        f"lo simulado, mensual, desde 1928): gana cuando el mercado revierte dentro del periodo y pierde cuando tiene "
        f"tendencia. UPRO+SPXU mensual con un 2% de préstamo queda {spp(S['neutral']['1938–1973'].loc[up_m, 'Δg_vs_letras'])}"
        f"/año frente a las letras en 1938–1973 (cociente de varianzas mensual "
        f"{num_es(mrow(mech, 'par corto UPRO+SPXU', 'mensual', '1938–1973').cociente_de_varianzas, 2)}) y "
        f"{spp(S['neutral']['2000–2026'].loc[up_m, 'Δg_vs_letras'])} en 2000–2026 "
        f"({num_es(mrow(mech, 'par corto UPRO+SPXU', 'mensual', '2000–2026').cociente_de_varianzas, 2)}); con reajuste "
        f"trimestral se arruina el {dmy(NS.loc[up_q, 'ruina'])}. Con días independientes su media es ligeramente "
        f"negativa ({pct_es(iid_sp.loc[(3, 'par corto', 'días independientes'), 'media_anual'], 2)} al año; la teoría "
        f"da {pct_es(iid_sp.loc[(3, 'par corto', theo), 'media_anual'], 2)}) aunque gane en el "
        f"{pct_es(iid_sp.loc[(3, 'par corto', 'días independientes'), 'meses_positivos'], 0)} de los meses: vende "
        f"una cola a su precio.",
        f"4. **Nasdaq-100.** TQQQ+SQQQ semanal con un 2% de préstamo rinde {spp(NN.loc[tq_w, 'Δg_vs_letras'])}/año "
        f"sobre las letras de 1985 a 2026 (p {peq(NN.loc[tq_w, 'p_vs_letras'])}); con margin call «restaurar», "
        f"{spp(NN.loc[tq_wr, 'Δg_vs_letras'])} (p_RW {peq(rwn_n.loc[tq_wr, 'p_RW'])} en la familia con préstamo "
        f"realista). Con los ETFs reales (2010–2026) queda en {spp(N['neutral']['ETFs reales'].loc[tq_w, 'Δg_vs_letras'])} "
        f"(p {peq(N['neutral']['ETFs reales'].loc[tq_w, 'p_vs_letras'])}), y en toda la muestra se anula con un "
        f"préstamo del {be_pair_n.loc[('par corto TQQQ+SQQQ', 'semanal'), 'préstamo que lo anula']}. En el S&P 500 el "
        f"mismo par semanal da {spp(NS.loc[up_w, 'Δg_vs_letras'])} desde 1928 (p {peq(NS.loc[up_w, 'p_vs_letras'])}). "
        f"Sin remunerar el producto de los cortos, el TQQQ+SQQQ semanal «restaurar» baja a "
        f"{spp(rbv(NDX, tq_wr, '1985–2026'))}/año (p_RW {peq(rwn_n_nr.loc[tq_wr, 'p_RW'])} en la misma familia) y "
        f"el semanal sin regla de margen a {spp(rbv(NDX, tq_w, '1985–2026'))}.",
        f"5. **(c) El lever-arb con cobertura diaria** cobra lo que el fondo cobra por encima de su réplica "
        f"(gasto + |L|·s: {pct_es(fund_fee(P, 'UPRO'), 2)} en UPRO, {pct_es(fund_fee(P, 'TQQQ'), 2)} en TQQQ) menos el "
        f"préstamo y el haircut: {spp(NS.loc[arb_d_u, 'Δg_vs_letras'], 2)}/año en UPRO y "
        f"{spp(NN.loc[arb_d_t, 'Δg_vs_letras'], 2)} en TQQQ con un 1% de préstamo, que se anulan con un "
        f"{be_arb_s.loc[('lever-arb −UPRO +3x (futuros)', 'diario'), 'préstamo que lo anula']} y un "
        f"{be_arb_n.loc[('lever-arb −TQQQ +3x (futuros)', 'diario'), 'préstamo que lo anula']}; sin remunerar el "
        f"producto del corto, {spp(rbv(SP, arb_d_u, '1928–2026'), 2)} y {spp(rbv(NDX, arb_d_t, '1985–2026'), 2)}. "
        f"Con cobertura semanal "
        f"vuelve a ser la apuesta de cociente de varianzas: {spp(NS.loc[arb_w_u, 'Δg_vs_letras'])} y "
        f"{spp(NN.loc[arb_w_t, 'Δg_vs_letras'])} a tamaño completo (incumple el mantenimiento "
        f"{num_es(NS.loc[arb_w_u, 'incumplimientos_año'], 1)} y {num_es(NN.loc[arb_w_t, 'incumplimientos_año'], 1)} "
        f"veces al año) y {spp(NS.loc[arb_v_u, 'Δg_vs_letras'])} y {spp(NN.loc[arb_v_t, 'Δg_vs_letras'])} con un tercio "
        + ("del tamaño, que no lo incumple nunca." if NS.loc[arb_v_u, "incumplimientos_año"] == 0
           and NN.loc[arb_v_t, "incumplimientos_año"] == 0 else
           f"del tamaño, que lo incumple {num_es(NS.loc[arb_v_u, 'incumplimientos_año'], 2)} y "
           f"{num_es(NN.loc[arb_v_t, 'incumplimientos_año'], 2)} veces al año."),
        f"6. **(d) Comprar los dos lados pierde**, como se esperaba: UPRO+SPXU mensual "
        f"{spp(NS.loc[pair_name('UPRO', 'SPXU', 'M', side='largo'), 'Δg_vs_letras'])}/año frente a las letras desde 1928; "
        f"TQQQ+SQQQ {spp(NN.loc[pair_name('TQQQ', 'SQQQ', 'M', side='largo'), 'Δg_vs_letras'])} desde 1985.",
        f"7. **(e) 2x o 3x.** Con la regla original, la LRS con el ETF 2x (SSO) rinde un "
        f"{pct_es(x23s.loc['1938–2026', 'CAGR 2x (ETF 2x)'])} frente al {pct_es(x23s.loc['1938–2026', 'CAGR 3x (original)'])} "
        f"con el 3x desde 1938 (Δg {spp(x23s.loc['1938–2026', 'Δg 3x − 2x'])} a favor del 3x, p "
        f"{peq(x23s.loc['1938–2026', 'p (3x > 2x)'])}), con caídas máximas del {pct_es(x23s.loc['1938–2026', 'caída_máx 2x'])} "
        f"y del {pct_es(x23s.loc['1938–2026', 'caída_máx 3x'])}. En el Nasdaq-100 la diferencia (QLD "
        f"{pct_es(x23n.loc['1985–2026', 'CAGR 2x (ETF 2x)'])}, TQQQ {pct_es(x23n.loc['1985–2026', 'CAGR 3x (original)'])}) "
        f"no es significativa (p {peq(x23n.loc['1985–2026', 'p (3x > 2x)'])}).",
        f"8. **La curva de la regla.** En el S&P 500 desde 1938 crece hasta {lx(Lbest)} "
        f"({pct_es(cs.loc[Lbest, 'CAGR LRS'])} frente al {pct_es(cs.loc[3.0, 'CAGR LRS'])} del 3x sintético) y baja "
        f"después; desde {lx(Lruin)} la regla se arruina el {dmy(cs.loc[Lruin, 'ruina LRS'])}, porque con la señal "
        f"original y la orden al cierre siguiente estaba dentro ese día. El L* de Kelly dentro de la tendencia es "
        f"{num_es(K.loc[ks_in, kelly.LSTAR_APROX], 1)} aproximado (IC90 {num_es(K.loc[ks_in, 'IC90_bajo'], 1)}–"
        f"{num_es(K.loc[ks_in, 'IC90_alto'], 1)}) y {num_es(K.loc[ks_in, kelly.LSTAR_EXACTO], 1)} exacto (crecimiento "
        f"realizado): un 3x es unos tres cuartos de ese Kelly condicionado, estimado dentro de muestra. Con la señal "
        f"de precio la regla estaba fuera el 19-10-1987 y la curva sigue subiendo hasta {lx(Lpx)} "
        f"({pct_es(cs.loc[Lpx, 'CAGR LRS (señal de precio)'])}): la forma de la curva por encima de 4x depende de "
        f"un solo día. En el Nasdaq-100 el máximo está en {lx(Lbn)}. Ningún L > 3 sobrevive a la corrección por la "
        f"búsqueda frente a la original, y en la base no hay ningún fondo cotizado de más de 3x.",
        f"9. **Búsqueda.** Se simularon {fams.loc['total', f'{SP}: configuraciones']} configuraciones en el S&P 500 "
        f"({fams.loc['total', f'{SP}: series distintas']} series distintas) y {fams.loc['total', f'{NDX}: configuraciones']} "
        f"en el Nasdaq-100 ({fams.loc['total', f'{NDX}: series distintas']}). Frente a la LRS original en el S&P 500 "
        f"desde 1938, el SPA de toda la familia de tendencia da p "
        f"{peq(famS.loc[('tendencia: todas', '1938–2026', 'LRS 3x (original)'), 'p_SPA_c'])} por el corto diario del "
        f"−3x (sin remunerar el producto del corto, p "
        f"{peq(RB[SP]['familias'].set_index(['familia', 'ventana', 'frente a']).loc[('tendencia: todas', '1938–2026', 'LRS 3x (original)'), 'p_SPA_c'])}); "
        f"sin las reglas que venden en corto, p "
        f"{peq(famS.loc[('tendencia: sin vender en corto', '1938–2026', 'LRS 3x (original)'), 'p_SPA_c'])} y ninguna "
        f"sobrevive a Romano–Wolf; con la señal original y sin cortos, p "
        f"{peq(famS.loc[('tendencia: señal original, sin cortos', '1938–2026', 'LRS 3x (original)'), 'p_SPA_c'])}.",
    ]))

    # ── 0. comprobación ──
    rep.h("0. Comprobación de partida")
    tab("LRS 3x (original), 1x y 3x en el S&P 500 desde el 4-2-1938: R22 frente a lo publicado en R13",
        C["sanidad"], pct={"CAGR (R22)", "CAGR (R13)", "caída_máx (R22)", "caída_máx (R13)"}, digits=3)
    rep.p("R22 construye la LRS original con r13.run_weights y la misma señal que R13 (media de 200 sesiones del "
          "índice de rentabilidad total, orden al cierre siguiente) y reproduce sus cifras. Si no las reprodujera, "
          "compute() se detendría con un error.")

    # ── 1. fondos ──
    rep.h("1. Los fondos simulados frente a los ETFs reales")
    rep.p("**Modelo del fondo.** Un ETF diario de apalancamiento L rinde cada día\n\n"
          "    r = rf + L·(u − f) − |L|·s/252 − ER/252\n\n"
          "*Pie de fórmula: el fondo cobra la letra sobre su patrimonio, gana (o pierde, si L es negativo) L veces lo "
          "que el índice rinde por encima de su financiación, y paga el sobrecoste de sus swaps y su comisión.* "
          "Notación: rf, la letra del Tesoro de un día; u, la rentabilidad diaria del índice con dividendos; f, el "
          "tipo al que se financia la exposición (call money hasta 1954, fondos federales después, con un mes de "
          "retraso); s, el sobrecoste anual del swap; ER, la comisión anual del fondo. Con L < 0 el fondo cobra f "
          "sobre el nocional corto y paga los dividendos, como la pata corta de un swap de rentabilidad total. Es "
          "r12.letf con otros L, s y ER. Un día de −100% (en el S&P 500, cualquier L ≥ 4,9 el 19-10-1987) lo "
          "pierde quien lleva el fondo; al día siguiente la fórmula sigue con un fondo nuevo, en lugar de quedarse "
          "a cero para siempre como en sim.letf_returns.")
    pt = pd.DataFrame({tk: {"mercado": p["mercado"], "L": p["L"], "primer día real": p["lanzamiento"].date(),
                            "comisión (ER)": p["er"] * 100, "s usado": p["s"] * 100,
                            "s calibrado (vida real)": p["s_calibrado"] * 100, "s de 2010–2017": p["s_2010_2017"] * 100,
                            "gasto + |L|·s": fund_fee(P, tk) * 100} for tk, p in P.items()}).T
    tab("Parámetros de cada ETF (en % anual). El 3x usa el s del laboratorio (R3/R13); los demás, el s que "
        "reproduce la media del ETF real en toda su vida", pt, digits=2)
    vt = C["validación"].copy()
    for c in ("gasto", "s", "diferencia_media_anual", "error_de_seguimiento_anual"):
        vt[c] = vt[c].astype(float) * 100
    vt = vt.rename(columns={"gasto": "gasto (%)", "s": "s (%)", "diferencia_media_anual": "real − sintético (pp/año)",
                            "error_de_seguimiento_anual": "error de seguimiento (pp/año)"})
    tab(f"Validación: ETF real frente a su sintético desde {REAL_FROM} (con el s de toda su vida) y, fuera de "
        "muestra, en 2018–2026 con el s de 2010–2017", vt,
        pct={"CAGR_real", "CAGR_sintético", "caída_máx_real", "caída_máx_sintético"}, digits=3)
    v = C["validación"]
    ins = v.xs("2010–2026 (s de toda su vida)", level="prueba")
    oos = v.xs("2018–2026 (s de 2010–2017)", level="prueba")
    invs = [t for t in ins.index if P[t]["L"] < 0]
    rep.p(f"**Lectura.** Los inversos sintéticos siguen a los reales con una correlación diaria de "
          f"{num_es(ins.loc[invs, 'correlación_diaria'].min(), 3)} a {num_es(ins.loc[invs, 'correlación_diaria'].max(), 3)} "
          f"y una diferencia media de {num_es(ins.loc[invs, 'diferencia_media_anual'].abs().max() * 100, 2)} pp/año como "
          f"mucho (con el s de su propia vida, dentro de muestra). Fuera de muestra (s de 2010–2017, evaluado en "
          f"2018–2026) la diferencia de los inversos queda entre {num_es(oos.loc[invs, 'diferencia_media_anual'].min() * 100, 2)} "
          f"y {num_es(oos.loc[invs, 'diferencia_media_anual'].max() * 100, 2)} pp/año; la de los 3x alcistas llega a "
          f"{num_es(oos.loc[['UPRO', 'TQQQ'], 'diferencia_media_anual'].min() * 100, 2)} pp/año, porque su sobrecoste "
          "subió después de 2017. Antes de su lanzamiento todos los fondos son sintéticos: la historia larga de los "
          "inversos es una extrapolación de este modelo.")
    fig_validation(D, P, rep)

    # ── 2. inverso bajo la media ──
    rep.h("2. (a) Pasar al inverso bajo la media")
    rep.p("La LRS original (Gayed y Bilello 2016) lleva el 3x cuando el índice de rentabilidad total cierra sobre su "
          "media de 200 sesiones y letras cuando cierra debajo. Aquí, debajo de la media, compra un ETF inverso: "
          "3x/−1x (UPRO/SH en el S&P 500, TQQQ/PSQ en el Nasdaq-100), 3x/−2x (SDS, QID), 3x/−3x (SPXU, SQQQ), 2x/−2x "
          "(SSO/SDS, QLD/QID) y 1x/−1x (el índice y SH o PSQ); además, 3x/−3x con banda de ±2% y 3x/−3x solo si la "
          "media baja (media de hoy menor que la de hace 20 sesiones; si no, letras). Misma señal, misma ejecución "
          "y 5 pb por unidad negociada (pasar del 3x al inverso son dos unidades).")
    keys = [ONE, THREE, "LRS 1x (original)", "LRS 2x (original)", ORIGINAL] + inv_keys
    tab("S&P 500 desde el 4-2-1938: letras o inverso bajo la media. Δg: crecimiento logarítmico anual menos el de "
        "la referencia; IC90 y p (unilateral, H0: Δg ≤ 0) por bootstrap estacionario emparejado",
        view(T, keys, TREND_COLS), pct=PCT, digits=3)
    wins = list(SP_WINDOWS)
    tab("S&P 500: Δg frente a la LRS original por ventana (pp/año; * = p < 0,05 nominal, a favor de la variante)",
        pivot_dg(A, SP, inv_keys, wins), index=True)
    rep.p("Ventanas: " + "; ".join(f"{w}: {WINDOW_NOTE[w]}" for w in WINDOW_NOTE) + ".")
    rkeys = [ONE, "LRS 1x (original)", ORIGINAL] + inv_keys
    rt = rec.loc[rkeys].copy()
    for c in [c for c in rt.columns if c.startswith("Δg")]:
        rt[c] = rt[c] * 100
    tab(f"Nasdaq-100 desde 1985-10: {r13.N_NDX} reconstrucciones del periodo previo (solo calientan la media). "
        "CAGR y caída máxima; Δg en pp/año", rt.rename(columns=lambda c: c + " (pp)" if c.startswith("Δg") else c),
        pct={"CAGR_mediana", "CAGR_mín", "CAGR_máx", "maxDD_mediana"}, digits=3)
    tab("Nasdaq-100 (reconstrucción 0): Δg frente a la LRS original por ventana (pp/año; * = p < 0,05 nominal)",
        pivot_dg(A, NDX, inv_keys, list(NDX_WINDOWS)), index=True)
    fig_inverse(M[SP]["R"], rep)
    kt = pd.concat({SP: S["kelly"], NDX: N["kelly"]})
    kt2 = kt.copy()
    for c in ("sesiones", "μ_anual", "compuesta_anual", "σ_anual", "μ − f anual"):
        kt2[c] = kt2[c].astype(float)
    kt2["exacto"] = [kelly.fmt_lstar(v, 2, L_MAX_KELLY) for v in kt[kelly.LSTAR_EXACTO]]
    kt2 = kt2.drop(columns=[kelly.LSTAR_EXACTO]).rename(columns={"exacto": kelly.LSTAR_EXACTO})
    tab("El índice dentro y fuera de la tendencia (sesiones en que la LRS lleva el 3x o está fuera, con la señal "
        f"ejecutada). {kelly.LSTAR_APROX}: {kelly.LSTAR_DESC[kelly.LSTAR_APROX]}; IC90 por bootstrap estacionario de "
        f"bloque {BLOCK_KELLY}. {kelly.LSTAR_EXACTO}: el L constante que maximiza el crecimiento realizado de esas "
        f"sesiones (L/3 en el 3x con su comisión y, por encima de 3, préstamo a f + s; tope {num_es(L_MAX_KELLY, 0)}). "
        "Dentro de muestra. μ_anual: media aritmética de las rentabilidades diarias ×252 (la que usa L*); "
        "compuesta_anual: la rentabilidad anual compuesta de esas sesiones", kt2,
        pct={"sesiones", "μ_anual", "compuesta_anual", "σ_anual", "μ − f anual"}, digits=2)
    rep.p("*Pie de fórmula: L* = (μ − f − s)/σ² es el apalancamiento que maximiza el crecimiento esperado cuando las "
          "rentabilidades son aproximadamente normales: la prima sobre la financiación dividida por la varianza.* "
          "Notación: μ, rentabilidad media anual del índice en esas sesiones; f, tipo de financiación; s, sobrecoste "
          "del swap del 3x; σ², varianza anual del índice en esas sesiones.")
    fig_regimes(A, rep)
    lo_o, hi_o = K.loc[ks_out, "IC90_bajo"], K.loc[ks_out, "IC90_alto"]
    rep.p(f"**Lectura.** El inverso no mejora la regla salvo en la Gran Depresión (1928–1937), y ni allí de forma "
          f"significativa: la 3x/−1x resta "
          f"{spp(-T.loc['LRS 3x / −1x', 'Δg_vs_LRS'])[1:]} al año desde 1938 y cuanto más apalancado el inverso, peor "
          f"(3x/−2x {spp(T.loc['LRS 3x / −2x', 'Δg_vs_LRS'])}, 3x/−3x {spp(T.loc['LRS 3x / −3x', 'Δg_vs_LRS'])}). La "
          f"razón está en la tabla del índice: bajo la media, la media aritmética de las rentabilidades diarias del "
          f"S&P 500 equivale a un {pct_es(K.loc[ks_out, 'μ_anual'])} anual desde 1938 (compuesta, un "
          f"{pct_es(K.loc[ks_out, 'compuesta_anual'])}; el Nasdaq-100, "
          f"{pct_es(N['kelly'].loc[('1985–2026', 'fuera de la tendencia'), 'μ_anual'])} y "
          f"{pct_es(N['kelly'].loc[('1985–2026', 'fuera de la tendencia'), 'compuesta_anual'])} desde 1985) con una "
          f"volatilidad del {pct_es(K.loc[ks_out, 'σ_anual'])}. "
          + ("Bajo la media el índice también sube, en media aritmética y en compuesta: eso es lo que paga el inverso."
             if K.loc[ks_out, "compuesta_anual"] > 0 else
             "Bajo la media el índice sube en media aritmética, pero su rentabilidad compuesta es negativa.")
          + f" La media de 200 sesiones separa "
          f"sobre todo la volatilidad, no el signo de la rentabilidad (como en el artículo 1). El L* de Kelly fuera "
          f"de la tendencia es {num_es(K.loc[ks_out, kelly.LSTAR_APROX], 2)} con un IC90 de {num_es(lo_o, 1)} a "
          f"{num_es(hi_o, 1)}: estar fuera (L = 0) ya queda por debajo de su estimación puntual, y un inverso (L "
          f"negativo) se aleja todavía más. Además, un −3x paga su propio decaimiento justo en el tramo más volátil. "
          f"En 1928–1937, con una caída del índice del 83% en tres años, los inversos suavizan la regla; en el resto de "
          f"la historia restan en todas las ventanas.")

    # corto del −3x
    rep.h("La LRS con corto del −3x", 3)
    rep.p("Misma exposición alcista (+3) que la LRS original, pero en lugar de comprar el 3x se vende en corto el −3x "
          "(SPXU, SQQQ) por el 100% del patrimonio. El corto deriva: si el mercado cae, el −3x sube y el corto crece "
          "en proporción al patrimonio. Se reajusta cada mes (o cada día) y cuando cambia la señal. El capital cobra la "
          "letra, el producto del corto la letra − 0,5% y el préstamo de títulos cuesta 0, 2 o 5% anual. "
          f"Mantenimiento de FINRA para un ETF 3x: {pct_es(MAINT * 3, 0)} del valor del corto (30% × 3, regla 4210 "
          "tal como la aplica la exploración previa; no se ha verificado aquí contra el texto de la norma). Se "
          "comprueba al cierre; «restaurar» vuelve al objetivo en ese cierre y «liquidar» recompra y espera en caja "
          "al siguiente reajuste.")
    sht = view(T, sh_keys, TREND_COLS, {"incumplimientos_año": "incumplimientos/año", "ruina": "ruina"})
    tab("S&P 500 desde el 4-2-1938: LRS con corto del −3x", sht, pct=PCT, digits=3)
    shn = view(TN, sh_keys, TREND_COLS, {"incumplimientos_año": "incumplimientos/año", "ruina": "ruina"})
    tab("Nasdaq-100 desde 1985-10 (reconstrucción 0): LRS con corto del −3x", shn, pct=PCT, digits=3)
    b3 = pd.concat({SP: S["préstamo_nulo_corto3"], NDX: N["préstamo_nulo_corto3"]})
    b3 = b3.rename(columns={"préstamo que lo anula": "préstamo con el que deja de superar a la original"})
    for c in [c for c in b3.columns if c.startswith("Δg")]:
        b3[c] = b3[c].astype(float) * 100
    tab("Δg frente a la LRS original (pp/año) según el préstamo del −3x, S&P 500 desde 1938 y Nasdaq-100 desde 1985",
        b3.rename(columns=lambda c: c.replace("Δg con", "Δg (pp) con")), digits=2)
    r_sp, r_nd = T.loc[m_short, 'ruina'], TN.loc[short3_name('M', 0.02), 'ruina']
    when = (f"el {dmy(r_sp)} en los dos mercados" if r_sp == r_nd else
            f"el {dmy(r_sp)} en el S&P 500 y el {dmy(r_nd)} en el Nasdaq-100")
    rep.p(f"**Lectura.** Con reajuste mensual y sin regla de margen, el corto del −3x se arruina {when}: la regla estaba dentro, el índice ya había caído en octubre y el corto había crecido desde el "
          f"reajuste de fin de septiembre. Con «restaurar» sobrevive e incumple el mantenimiento "
          f"{num_es(T.loc[short3_name('M', 0.02, 'restaurar'), 'incumplimientos_año'], 1)} veces al año. Con reajuste "
          f"diario y un 2% de préstamo suma {spp(T.loc[d_short, 'Δg_vs_LRS'], 2)}/año en el S&P 500 y "
          f"{spp(TN.loc[d_short, 'Δg_vs_LRS'], 2)} en el Nasdaq-100: el inversor cobra la comisión y el sobrecoste del "
          f"inverso en lugar de pagar los del 3x, menos el préstamo, el haircut y el coste de reajustar cada día. Es "
          f"casi determinista (IC90 {T.loc[d_short, 'IC90_vs_LRS']} pp) y se anula con un préstamo del {be3} en el "
          f"S&P 500. No es una mejora de la regla, sino de su implementación, y exige cuatro cosas que los datos no "
          f"garantizan: poder vender en corto el −3x cada día, un préstamo por debajo de ese umbral también en las "
          f"crisis, un bróker que tolere {num_es(T.loc[d_short, 'incumplimientos_año'], 1)} incumplimientos del "
          f"mantenimiento al año al cierre, antes de reajustar, y que remunere el producto del corto. Sin esa "
          f"remuneración (sección 7) el corto diario con un 2% de préstamo queda en {spp(d_nr, 2)}/año frente a la "
          f"original en el S&P 500 y en {spp(rbv(NDX, d_short, '1985–2026'), 2)} en el Nasdaq-100; en el S&P 500 "
          f"{be_txt(be3_nr)}.")

    # ── 3. par corto ──
    rep.h("3. (b) El par corto: vender los dos lados")
    rep.p("La idea que circula: como el 3x y el −3x pierden valor con el tiempo por el decaimiento de volatilidad, "
          "venderlos en corto a la vez sería un arbitraje. Por cada dólar se vende 0,5 del alcista y 0,5 del bajista "
          "del mismo índice y |L| (SSO+SDS y UPRO+SPXU en el S&P 500 desde 1928; QLD+QID y TQQQ+SQQQ en el "
          "Nasdaq-100 desde 1985-10) y se reajusta cada día, semana, mes o trimestre (al cierre de la última sesión "
          "del periodo). Rejilla: préstamo de 0, 2, 5 y 10% anual; reglas de margen «ninguna», «restaurar» y "
          f"«liquidar» con el mantenimiento de FINRA de {pct_es(MAINT * 2, 0)} (2x) o {pct_es(MAINT * 3, 0)} (3x) del "
          "valor del corto.")
    rep.p("**El mecanismo.** Entre dos reajustes, sin costes, el par corto gana o pierde\n\n"
          "    P&L de camino ≈ (L²/2)·(Σx² − (Σx)²)\n\n"
          "*Pie de fórmula: el par gana la suma de los cuadrados de los movimientos diarios (el «decaimiento» que "
          "pierden los dos fondos) y pierde el cuadrado del movimiento acumulado del periodo; gana si el mercado va y "
          "vuelve, y pierde si tiene tendencia.* Notación: L, el apalancamiento de los fondos; x = u − f, la "
          "rentabilidad diaria del índice sobre su financiación; Σ, la suma sobre las sesiones del periodo. En "
          "media, (Σx)² = Σx² × VR, con VR el cociente de varianzas del periodo, así que el par gana (L²/2)·Σx²·(1 − VR): "
          "es una apuesta a que VR < 1 (reversión), no una prima segura. Con reajuste diario no hay efecto de camino y "
          "queda la diferencia entre las comisiones que cobran los fondos y el préstamo.")
    main_pairs = []
    for bull, bear, L in PAIRS[SP] + PAIRS[NDX]:
        main_pairs += [pair_name(bull, bear, fq, 0.02) for fq in FREQS]
        main_pairs += [pair_name(bull, bear, fq, 0.02, r) for fq in ("W", "M", "Q") for r in ("restaurar", "liquidar")]
    sp_main = [k for k in main_pairs if any(t in k for t in ("SSO", "UPRO"))]
    nd_main = [k for k in main_pairs if any(t in k for t in ("QLD", "TQQQ"))]
    tab("S&P 500 desde el 19-10-1928: par corto con un 2% de préstamo. Δg frente a las letras; incumplimientos del "
        "mantenimiento por año; ruina: primer día en que la cuenta pierde el 100%",
        view(NS, sp_main, NEUTRAL_COLS), pct=PCT, digits=3)
    tab("Nasdaq-100 desde 1985-10: par corto con un 2% de préstamo", view(NN, nd_main, NEUTRAL_COLS), pct=PCT, digits=3)
    ekeys_sp = [pair_name(b1, b2, fq, 0.02) for b1, b2, _ in PAIRS[SP] for fq in FREQS]
    ekeys_nd = [pair_name(b1, b2, fq, 0.02) for b1, b2, _ in PAIRS[NDX] for fq in FREQS]
    tab("S&P 500: Δg del par corto (2% de préstamo) frente a las letras por ventana (pp/año; * = p < 0,05 nominal)",
        pivot_dg(A, SP, ekeys_sp, wins, "neutral", "letras"), index=True)
    tab("Nasdaq-100: Δg del par corto (2% de préstamo) frente a las letras por ventana (pp/año; * = p < 0,05 nominal)",
        pivot_dg(A, NDX, ekeys_nd, list(NDX_WINDOWS), "neutral", "letras"), index=True)
    bp = pd.concat({SP: be_pair_s, NDX: be_pair_n})
    for c in [c for c in bp.columns if c.startswith("Δg")]:
        bp[c] = bp[c].astype(float) * 100
    tab("Par corto sin regla de margen: Δg frente a las letras (pp/año) en toda la muestra según el préstamo, y "
        "préstamo que lo anula (interpolado)", bp.rename(columns=lambda c: c.replace("Δg con", "Δg (pp) con")), digits=2)
    fig_pair_wealth(M, D, rep)
    fig_pair_eras(A, rep)
    mc = pd.concat([S["mecanismo"], N["mecanismo"]])
    mc.to_csv(rep.dir / "mecanismo_completo.csv", index=False)
    mshow = mc[(mc.L == 3) & mc.estrategia.str.startswith("par corto")].copy()
    for c in ("camino_anual", "aproximación_anual", "decaimiento_cosechado_anual", "pérdida_por_tendencia_anual"):
        mshow[c] = mshow[c] * 100
    mshow = mshow.drop(columns=["L", "t_pendiente_NW"]).rename(columns={
        "camino_anual": "camino (%/año)", "aproximación_anual": "aproximación (%/año)",
        "decaimiento_cosechado_anual": "(L²/2)·Σx² (%/año)", "pérdida_por_tendencia_anual": "(L²/2)·(Σx)² (%/año)"})
    tab("Mecanismo del par corto 3x por época: P&L de camino simulado por periodo (sin costes; cartera que deriva "
        "menos la reajustada a diario) frente a (L²/2)·(Σx² − (Σx)²). Cociente de varianzas = media de (Σx)² / media "
        "de Σx²; pendiente de la regresión del camino sobre la aproximación (errores de Newey-West de 12 retardos, "
        "periodos sin solape); peor periodo: P&L de la cartera sin cubrir en su peor periodo. Tabla completa (2x, "
        "lever-arb) en mecanismo_completo.csv", mshow.set_index(["mercado", "estrategia", "reajuste", "época"]),
        pct={"peor_periodo", "periodos_positivos"}, digits=3)
    fig_pair_formula(A, rep)
    vr = pd.concat([S["cociente_varianzas"], N["cociente_varianzas"]])
    vr["z medio anual (Σx² − (Σx)²)·252/h"] = vr["z medio anual (Σx² − (Σx)²)·252/h"] * 100
    tab("Cociente de varianzas VR(h) de x = u − f con ventanas solapadas de h sesiones (una por sesión) y contraste "
        "de z = Σx² − (Σx)² frente a 0: regresión sobre la constante con errores de Hansen-Hodrick de h − 1 retardos "
        "(stats.hac_overlap). z × (L²/2) es el camino de un par corto reajustado cada h sesiones; en %/año",
        vr.rename(columns={"z medio anual (Σx² − (Σx)²)·252/h": "z medio (%/año)"}).set_index(["mercado", "época", "h (sesiones)"]),
        digits=3)
    iid = pd.concat([S["iid"], N["iid"]])
    iid = iid[iid.L == 3].drop(columns=["L"])
    for c in ("media_anual", "mediana_anual"):
        iid[c] = iid[c] * 100
    tab(f"P&L de camino mensual (sin costes) del par corto y del lever-arb 3x: meses naturales de la historia, meses "
        f"de {T_IID} sesiones sorteadas una a una (días independientes, {num_es(N_IID, 0)} meses) y bootstrap "
        "estacionario de bloque 21; media y mediana en %/año",
        iid.rename(columns={"media_anual": "media (%/año)", "mediana_anual": "mediana (%/año)"}).set_index(
            ["mercado", "estrategia", "mundo"]),
        pct={"meses_positivos", "percentil_5_mensual", "percentil_1_mensual", "peor_mes"}, digits=3)
    ep = pd.concat([S["episodios"], N["episodios"]])
    ep_show = ep[ep.estrategia.str.contains("Par corto (?:UPRO|TQQQ)", regex=True)]
    tab("Episodios de crisis: resultado del par corto 3x con un 2% de préstamo (del primer al último día del "
        "episodio), su caída máxima dentro del episodio y su peor día",
        ep_show.set_index(["mercado", "episodio", "estrategia"]),
        pct={"índice", "resultado", "caída_máx", "peor_día"}, digits=3)
    ep.to_csv(rep.dir / "episodios_completo.csv", index=False)
    r38, r00 = mrow(mech, "par corto UPRO+SPXU", "mensual", "1938–1973"), mrow(mech, "par corto UPRO+SPXU", "mensual", "2000–2026")
    vr_sp = S["cociente_varianzas"].set_index(["época", "h (sesiones)"])
    rq = mrow(mech, "par corto UPRO+SPXU", "trimestral", "toda la muestra")
    rm = mrow(mech, "par corto UPRO+SPXU", "mensual", "toda la muestra")
    rep.p(f"**Lectura.** (1) La aproximación explica el camino del par 3x periodo a periodo (mensual: correlación "
          f"{num_es(rm.correlación, 3)} y pendiente {num_es(rm.pendiente, 2)} en el S&P 500 desde 1928; trimestral: "
          f"{num_es(rq.correlación, 3)} y {num_es(rq.pendiente, 2)}). El par trimestral sin cubrir perdió más del "
          f"100% en {int(rq.periodos_con_ruina)} trimestres de 1932 (peor trimestre: {pct_es(rq.peor_periodo)}) y "
          f"se arruina el {dmy(NS.loc[up_q, 'ruina'])}. (2) El «decaimiento» que se cosecha es grande "
          f"((L²/2)·Σx² = {pct_es(r38['decaimiento_cosechado_anual'])} al año en 1938–1973 y "
          f"{pct_es(r00['decaimiento_cosechado_anual'])} en 2000–2026, con reajuste mensual), pero las tendencias se "
          f"llevan {'más que todo' if r38['pérdida_por_tendencia_anual'] > r38['decaimiento_cosechado_anual'] else 'buena parte'} "
          f"en 1938–1973 ((L²/2)·(Σx)² = {pct_es(r38['pérdida_por_tendencia_anual'])}) y el "
          f"{pct_es(r00['pérdida_por_tendencia_anual'] / r00['decaimiento_cosechado_anual'], 0)} en 2000–2026 "
          f"({pct_es(r00['pérdida_por_tendencia_anual'])}). Lo que queda depende del cociente de varianzas: "
          f"{num_es(r38.cociente_de_varianzas, 2)} en 1938–1973 (tendencia: el par pierde "
          f"{pct_es(-r38.camino_anual)} al año de camino) y {num_es(r00.cociente_de_varianzas, 2)} en 2000–2026 "
          f"(reversión: gana {pct_es(r00.camino_anual)}). (3) Con ventanas solapadas, VR(5) fue "
          f"{num_es(vr_sp.loc[('1938–1973', 5), 'VR(h)'], 2)} en 1938–1973 (t = "
          f"{num_es(vr_sp.loc[('1938–1973', 5), 't (HAC)'], 1)}; la autocorrelación positiva de la época, en parte "
          f"precios desfasados) y VR(21) {num_es(vr_sp.loc[('2000–2026', 21), 'VR(h)'], 2)} en 2000–2026 (t = "
          f"{num_es(vr_sp.loc[('2000–2026', 21), 't (HAC)'], 1)}). En toda la muestra del S&P 500 VR(21) es "
          f"{num_es(vr_sp.loc[('toda la muestra', 21), 'VR(h)'], 2)} (t = {num_es(vr_sp.loc[('toda la muestra', 21), 't (HAC)'], 1)}): "
          f"no hay reversión media a un mes en la historia larga. (4) Con días independientes la media del par es "
          f"ligeramente negativa, como dice la teoría, y la mediana positiva: gana casi todos los meses y pierde mucho "
          f"en pocos (asimetría {num_es(iid_sp.loc[(3, 'par corto', 'días independientes'), 'asimetría'], 1)}). La ganancia "
          f"histórica del par sale de la autocorrelación negativa de una época, no de una propiedad de los fondos. "
          f"(5) El préstamo lo decide casi todo: con un 5% todos los pares del S&P 500 pierden frente a las letras, y "
          f"en el Nasdaq-100 el semanal 3x se anula hacia el "
          f"{be_pair_n.loc[('par corto TQQQ+SQQQ', 'semanal'), 'préstamo que lo anula']}. El coste real de pedir "
          f"prestados SPXU o SQQQ, y su disponibilidad en las crisis, no están en los datos.")

    # ── 4. par largo ──
    rep.h("4. (d) Comprar los dos lados (control)")
    long_keys = [pair_name(b1, b2, fq, side="largo") for b1, b2, _ in PAIRS[SP] for fq in FREQS]
    long_keys_n = [pair_name(b1, b2, fq, side="largo") for b1, b2, _ in PAIRS[NDX] for fq in FREQS]
    tab("Par largo (0,5 en cada fondo, sin préstamo): S&P 500 desde el 19-10-1928", view(NS, long_keys, NEUTRAL_COLS),
        pct=PCT, digits=3)
    tab("Par largo: Nasdaq-100 desde 1985-10", view(NN, long_keys_n, NEUTRAL_COLS), pct=PCT, digits=3)
    lg = pd.concat([NS.loc[long_keys, "Δg_vs_letras"], NN.loc[long_keys_n, "Δg_vs_letras"]])
    best_long = lg.idxmax()
    rep.p("**Lectura.** Es la apuesta contraria (gana si VR > 1) y además paga dos comisiones y dos sobrecostes. "
          + (f"Pierde frente a las letras en toda la muestra de los dos mercados con cualquier reajuste; el que menos "
             f"pierde es «{best_long}» ({spp(lg.max())}/año), porque el reajuste trimestral deja correr las "
             f"tendencias. " if lg.max() < 0 else
             f"Solo «{best_long}» gana a las letras en toda la muestra ({spp(lg.max())}/año). ")
          + "Es la comprobación de que el signo del par depende del cociente de varianzas y no de un «decaimiento» "
            "que se pueda cobrar gratis.")

    # ── 5. lever-arb ──
    rep.h("5. (c) Lever-arb: corto del ETF y L veces el índice")
    rep.p("Por cada dólar se vende 1 del ETF L× y se compran L veces el índice en futuros, que rinden u − f (la "
          "financiación implícita del futuro, sin base; 1 pb por unidad negociada). El capital cobra la letra y el "
          "producto del corto la letra − 0,5%; el préstamo del ETF cuesta 0, 1, 3 o 5% anual. Con cobertura diaria "
          "no hay efecto de camino y el resultado es\n\n"
          "    exceso sobre la letra ≈ ER + |L|·s − préstamo − haircut − coste de reajustar\n\n"
          "*Pie de fórmula: quien vende el fondo y replica su exposición con futuros cobra exactamente lo que el fondo "
          "cobra por encima de su réplica (su comisión y el sobrecoste de sus swaps) y paga el préstamo del título.* "
          "Con cobertura semanal, mensual o trimestral aparece el camino ≈ ((L² − L)/2)·(Σx² − (Σx)²), la misma "
          "apuesta que el par. A tamaño completo el corto (100% del patrimonio) incumple con facilidad el "
          f"mantenimiento de {pct_es(MAINT * 3, 0)}: solo se cuentan los incumplimientos. El tamaño viable vende 1/L "
          "del fondo y compra 1 vez el índice, con margin call «liquidar». Sensibilidad: el índice al contado en una "
          "cuenta de margen que compensa saldos (el producto del corto paga la compra; saldo negativo a f + 1,5%).")
    fee = pd.DataFrame({tk: {"gasto (ER)": P[tk]["er"] * 100, "|L|·s": abs(P[tk]["L"]) * P[tk]["s"] * 100,
                             "gasto + |L|·s": fund_fee(P, tk) * 100}
                        for tk in ("SSO", "UPRO", "QLD", "TQQQ")}).T
    tab("Lo que cada fondo cobra por encima de su réplica (% anual)", fee, digits=2)
    arb_sp = [k for k in NS.index if k.startswith("Lever-arb")]
    arb_nd = [k for k in NN.index if k.startswith("Lever-arb")]
    tab("S&P 500 desde el 19-10-1928: lever-arb", view(NS, arb_sp, NEUTRAL_COLS), pct=PCT, digits=3)
    tab("Nasdaq-100 desde 1985-10: lever-arb", view(NN, arb_nd, NEUTRAL_COLS), pct=PCT, digits=3)
    ba = pd.concat({SP: be_arb_s, NDX: be_arb_n})
    for c in [c for c in ba.columns if c.startswith("Δg")]:
        ba[c] = ba[c].astype(float) * 100
    tab("Lever-arb con futuros a tamaño completo: Δg frente a las letras (pp/año) en toda la muestra según el préstamo "
        "y préstamo que lo anula", ba.rename(columns=lambda c: c.replace("Δg con", "Δg (pp) con")), digits=2)
    akeys = [arb_d_u, arb_w_u, arb_v_u, arb_name("UPRO", 3, "M", 0.01)]
    tab("S&P 500: Δg del lever-arb de UPRO (1% de préstamo) frente a las letras por ventana (pp/año; * = p < 0,05)",
        pivot_dg(A, SP, akeys, wins, "neutral", "letras"), index=True)
    akeys_n = [arb_d_t, arb_w_t, arb_v_t, arb_name("TQQQ", 3, "M", 0.01)]
    tab("Nasdaq-100: Δg del lever-arb de TQQQ (1% de préstamo) frente a las letras por ventana (pp/año; * = p < 0,05)",
        pivot_dg(A, NDX, akeys_n, list(NDX_WINDOWS), "neutral", "letras"), index=True)
    fig_lever_arb(A, rep)
    mg_d = arb_name("SSO", 2, "D", 0.01, impl="al contado con margen")
    rep.p(f"**Lectura.** El único «arbitraje» casi limpio es el lever-arb con cobertura diaria: cobra "
          f"{pct_es(fund_fee(P, 'UPRO'), 2)} (UPRO) o {pct_es(fund_fee(P, 'TQQQ'), 2)} (TQQQ) al año y devuelve casi todo "
          f"al préstamo, al haircut y a los reajustes diarios. Con un 1% de préstamo quedan "
          f"{spp(NS.loc[arb_d_u, 'Δg_vs_letras'], 2)} y {spp(NN.loc[arb_d_t, 'Δg_vs_letras'], 2)} al año, con una "
          f"caída máxima del {pct_es(NS.loc[arb_d_u, 'maxDD'])} y del {pct_es(NN.loc[arb_d_t, 'maxDD'])}; con SSO y QLD, "
          f"que cobran menos, no queda nada. Esa cuenta supone que el producto del corto cobra la letra − 0,5%: sin "
          f"remuneración, el lever-arb diario de UPRO y el de TQQQ quedan en {spp(rbv(SP, arb_d_u, '1928–2026'), 2)} y "
          f"{spp(rbv(NDX, arb_d_t, '1985–2026'), 2)} al año"
          + (", y el «arbitraje» casi limpio desaparece. " if max(rbv(SP, arb_d_u, '1928–2026'),
                                                                 rbv(NDX, arb_d_t, '1985–2026')) <= 0 else ". ")
          + f"Con cobertura semanal gana más en los dos mercados, pero ya es la "
          f"apuesta del par: depende de la época (ver la tabla por ventanas) y a tamaño completo incumple el "
          f"mantenimiento varias veces al año. Con cobertura trimestral se arruina en 1932. La versión al contado con "
          f"margen parece mejor (p. ej. {spp(NS.loc[mg_d, 'Δg_vs_letras'], 2)} con SSO a diario), pero no por el "
          f"arbitraje: supone que el producto del corto paga la compra del índice sin haircut y, antes de 2010, cobra "
          f"de hecho la diferencia f − rf entre la financiación del fondo y la letra; queda fuera de la familia con "
          f"préstamo realista.")

    # ── 6. 2x frente a 3x y curva ──
    rep.h("6. (e) 2x frente a 3x y la curva de apalancamiento de la regla")
    x23 = pd.concat({SP: x23s, NDX: x23n})
    for c in ("Δg 3x − 2x", "IC90_bajo", "IC90_alto"):
        x23[c] = x23[c].astype(float) * 100
    tab("LRS original con el ETF 2x (SSO, QLD; real desde 2006) frente a la LRS 3x (original): Δg del 3x menos el 2x "
        "(pp/año) con IC90 y p unilateral", x23.rename(columns={"Δg 3x − 2x": "Δg 3x − 2x (pp)", "IC90_bajo": "IC90 bajo (pp)",
                                                               "IC90_alto": "IC90 alto (pp)"}),
        pct={"CAGR 2x (ETF 2x)", "CAGR 3x (original)", "caída_máx 2x", "caída_máx 3x"}, digits=3)
    cv = pd.concat([S["curva"].assign(mercado=SP), N["curva"].assign(mercado=NDX)])
    cv.to_csv(rep.dir / "curva_completa.csv", index=False)
    cshow = cv[cv.ventana.isin(["1938–2026", "1928–2026", "2000–2026", "ETFs reales", "1985–2026"])]
    cshow = cshow.set_index(["mercado", "ventana", "L"])
    tab("Curva de la regla: LRS con un fondo sintético de L× (s y comisión del 3x del laboratorio, fondo nuevo tras "
        "un día de −100%) y el mismo L× comprado y mantenido. En el S&P 500, también con la señal de precio "
        "(sensibilidad del laboratorio). Ruina: día en que la regla pierde el 100% dentro de la ventana", cshow,
        pct={"CAGR LRS", "caída_máx LRS", "CAGR comprar y mantener", "CAGR LRS (señal de precio)"}, digits=3)
    tab("El error del fondo muerto: la LRS 5x con el fondo de sim.letf_returns (a cero para siempre tras su primer "
        "−100%) y con un fondo nuevo, S&P 500 desde 1938", fm,
        pct={"CAGR con el fondo muerto (sim.letf_returns)", "CAGR con un fondo nuevo (este módulo)"}, digits=3)
    fig_curve(A, rep)
    fig_kelly_rt(A, rep)
    kin = K.loc[ks_in]
    kin_n = N["kelly"].loc[("1985–2026", "dentro de la tendencia")]
    rt_last = A["kelly_tiempo_real"].dropna().iloc[-1]
    crash = pd.Timestamp("1987-10-19")
    x_crash = float(D[SP]["u"].loc[crash] - D[SP]["f"].loc[crash])
    wall = -1 / x_crash
    rep.p(f"**Lectura.** (1) Con la regla original, el 3x creció más que el 2x en el S&P 500 desde 1938 "
          f"(p {peq(x23s.loc['1938–2026', 'p (3x > 2x)'])}), sobre todo en 1938–1973; desde 2000 y en el Nasdaq-100 "
          f"la diferencia no es significativa, y el 2x cae bastante menos. (2) La curva del S&P 500 tiene un máximo "
          f"dentro de muestra en {lx(Lbest)} y una pared en {num_es(wall, 2)}x: el 19-10-1987 el S&P 500 cayó "
          f"un {pct_es(-D[SP]['u'].loc[crash])} con la regla dentro, y un fondo de L × (u − f) = L × "
          f"({pct_es(x_crash, 2)}) pierde el 100% en cuanto L pasa de {num_es(wall, 2)}. El L* exacto "
          f"dentro de la tendencia ({num_es(kin[kelly.LSTAR_EXACTO], 1)}) coincide con ese máximo; el aproximado "
          f"({num_es(kin[kelly.LSTAR_APROX], 1)}) es mayor porque ignora las colas. (3) Con la señal de precio la regla "
          f"salió un día antes y la curva sube hasta {lx(Lpx)}; con el fondo muerto de sim.letf_returns, la LRS 5x de "
          f"precio rendía un {pct_es(fm.iloc[-1, 2])} en lugar de un {pct_es(fm.iloc[-1, 3])} (el error de la "
          f"exploración previa). Que el óptimo esté en 4x o en 5,5x lo decide un día. (4) En el Nasdaq-100 el máximo "
          f"está en {lx(Lbn)} y el L* dentro de la tendencia es {num_es(kin_n[kelly.LSTAR_APROX], 1)} (exacto "
          f"{num_es(kin_n[kelly.LSTAR_EXACTO], 1)}): allí el 3x ya es el Kelly condicionado. (5) Estimado día a día "
          f"con la historia disponible, el L* dentro de la tendencia del S&P 500 está por encima de 3 desde los años "
          f"cincuenta (hoy {num_es(rt_last['dentro de la tendencia'], 1)}); fuera de la tendencia rondó cero o fue "
          f"negativo hasta hace poco (hoy {num_es(rt_last['fuera de la tendencia'], 1)}). Nada de esto es operable: "
          f"no hay un ETF cotizado de más de 3x en la base, la curva está calculada con toda la muestra, y ningún L > 3 "
          f"sobrevive a la corrección por la búsqueda frente a la original (sección 7).")

    # ── 7. búsqueda ──
    rep.h("7. Cuánto descuenta la búsqueda")
    tab("Configuraciones simuladas por familia y series distintas (sin las idénticas entre sí ni a las referencias "
        "de R13)", fams, digits=0)
    fs = pd.concat([S["familias"], N["familias"]])
    fs = fs.copy()
    fs["Δg_mejor"] = fs["Δg_mejor"] * 100
    fs["sobreviven_RW"] = fs["sobreviven_RW"].str.slice(0, 160)
    tab("SPA de Hansen (versión consistente) por familia, con Holm y Romano–Wolf al 5% (mismos datos; bootstrap "
        f"estacionario de bloque {BLOCK}, {num_es(B, 0)} remuestreos). Δg_mejor en pp/año",
        fs.drop(columns=["t_SPA"]).rename(columns={"Δg_mejor": "Δg_mejor (pp)"}).set_index(
            ["mercado", "familia", "ventana", "frente a"]), digits=3)
    rw_t = S["rw_tendencia"].sort_values("p_RW").head(12).copy()
    for c in ("Δg", "IC90_bajo", "IC90_alto"):
        rw_t[c] = rw_t[c] * 100
    tab("S&P 500 desde 1938, familia de tendencia frente a la LRS original: las 12 reglas con menor p de Romano–Wolf "
        "(Δg e IC90 en pp/año)", rw_t, digits=4)
    rw_s = rwn_s.sort_values("p_RW").head(10).copy()
    rw_nn = rwn_n.sort_values("p_RW").head(10).copy()
    for df in (rw_s, rw_nn):
        for c in ("Δg", "IC90_bajo", "IC90_alto"):
            df[c] = df[c] * 100
    tab("S&P 500 desde el 19-10-1928, par corto y lever-arb con préstamo realista frente a las letras: las 10 con menor "
        "p de Romano–Wolf", rw_s, digits=4)
    tab("Nasdaq-100 desde 1985-10, par corto y lever-arb con préstamo realista frente a las letras: las 10 con menor p "
        "de Romano–Wolf", rw_nn, digits=4)
    surv_s = rwn_s.index[rwn_s.p_RW < 0.05].tolist()
    surv_1x = S["rw_tendencia_1x"].index[S["rw_tendencia_1x"].p_RW < 0.05].tolist()
    surv_n = rwn_n.index[rwn_n.p_RW < 0.05].tolist()
    er = famN.loc[(REALISTIC, "ETFs reales", "letras")]
    er_s = famS.loc[(REALISTIC, "ETFs reales", "letras")]
    rep.p(f"**Lectura.** (1) Tendencia frente a la LRS original: el SPA rechaza por el corto diario del −3x, la "
          f"diferencia casi determinista de comisiones de la sección 2; sin las reglas que venden en corto no sobrevive "
          f"nada (p_SPA {peq(famS.loc[('tendencia: sin vender en corto', '1938–2026', 'LRS 3x (original)'), 'p_SPA_c'])}; "
          f"la mejor es «{famS.loc[('tendencia: sin vender en corto', '1938–2026', 'LRS 3x (original)'), 'mejor']}», p_RW "
          f"{peq(famS.loc[('tendencia: sin vender en corto', '1938–2026', 'LRS 3x (original)'), 'p_RW_mejor'])}), y con la "
          f"señal original ninguna regla se acerca (p_SPA "
          f"{peq(famS.loc[('tendencia: señal original, sin cortos', '1938–2026', 'LRS 3x (original)'), 'p_SPA_c'])}). Desde "
          f"2016 y desde 2017, lo mismo. (2) Frente al 1x, la familia de tendencia rechaza (p_SPA "
          f"{peq(famS.loc[('tendencia: todas', '1938–2026', '1x'), 'p_SPA_c'])}), como la LRS original sola (p "
          f"{peq(T.loc[ORIGINAL, 'p_vs_1x'])}, tabla 4; el que cita el artículo es el del censo, p "
          f"{peq(C['censo_p']['p'])} en su ventana del {dmy(C['censo_p']['desde'])} al {dmy(C['censo_p']['hasta'])}, "
          f"recalculado aquí con los remuestreos de R19: la diferencia es de ventana y de remuestreos); de las "
          f"{len(surv_1x)} que sobreviven a Romano–Wolf, "
          f"{sum(1 for k in surv_1x if PX_TAG in k or 'corto' in k)} usan la señal de precio o venden en corto. "
          f"(3) Par corto y lever-arb frente a las letras con préstamo realista: en el S&P 500 sobreviven "
          f"{len(surv_s)} ({', '.join(surv_s) if surv_s else 'ninguna'}); en el Nasdaq-100, {len(surv_n)} "
          f"({', '.join(surv_n) if surv_n else 'ninguna'}). Son las coberturas semanales y el lever-arb diario, con "
          f"ventajas de décimas de punto a unos pocos puntos al año que dependen de un préstamo del 1–2% y, las "
          f"semanales, del cociente de varianzas de la época. Con ETFs reales (2010–2026) el SPA de esa familia da "
          f"p {peq(er_s['p_SPA_c'])} en el S&P 500 y p {peq(er['p_SPA_c'])} en el Nasdaq-100, y "
          + ("ninguna configuración sobrevive a Romano–Wolf: en su único periodo con fondos reales, la ventaja "
             "no se distingue de la suerte una por una."
             if int(er_s['rechazos_RW_5%']) + int(er['rechazos_RW_5%']) == 0 else
             f"sobreviven a Romano–Wolf {int(er_s['rechazos_RW_5%'])} y {int(er['rechazos_RW_5%'])} configuraciones."))

    # ── 7b. el producto del corto sin remunerar ──
    rep.h("El supuesto del producto del corto", 3)
    rep.p("Todo lo anterior supone que el producto de un corto cobra la letra − 0,5% (un bróker institucional). Muchas "
          "cuentas minoristas no remuneran ese saldo: aquí se repiten las reglas que venden en corto (LRS con corto "
          "del −3x, par corto y lever-arb) con el producto sin remunerar y sin compensar saldos (run_drift con "
          "rebate=False; el capital libre sigue cobrando la letra). Mismas reglas, otro supuesto de coste: no son "
          "configuraciones nuevas ni entran en el recuento ni en el censo (como las versiones de datos de R14 en R19). "
          "Remuestreos propios (semilla 72), emparejados entre los dos supuestos.")
    rt_ = pd.concat([RB[SP]["tabla"], RB[NDX]["tabla"]])
    rt_ = rt_.assign(Δg=rt_["Δg"] * 100)
    piv = rt_.pivot_table(index=["mercado", "regla", "frente a", "ventana"], columns="producto del corto",
                          values=["Δg", "p"], sort=False)
    piv.columns = [f"{a} ({b})" for a, b in piv.columns]
    piv = piv[[f"{a} ({b})" for b in REBATES for a in ("Δg", "p")]]
    piv = piv.rename(columns=lambda c: c.replace("Δg (", "Δg pp/año ("))
    tab("Las configuraciones citadas en el informe con el producto del corto remunerado (letra − 0,5%) y sin "
        "remunerar: Δg frente a la referencia (pp/año) y p unilateral", piv, digits=3)
    fr = pd.concat([RB[SP]["familias"].assign(mercado=SP), RB[NDX]["familias"].assign(mercado=NDX)])
    fr = fr.assign(Δg_mejor=fr["Δg_mejor"] * 100)
    fr["sobreviven_RW"] = fr["sobreviven_RW"].str.slice(0, 160)
    tab("Sin remunerar el producto de los cortos: SPA, Holm y Romano–Wolf de la familia de tendencia frente a la LRS "
        "original y de la familia con préstamo realista frente a las letras (compárese con la tabla anterior de "
        "familias). Δg_mejor en pp/año",
        fr.drop(columns=["t_SPA"]).rename(columns={"Δg_mejor": "Δg_mejor (pp)"}).set_index(
            ["mercado", "familia", "ventana", "frente a"]), digits=3)
    bn = pd.concat({(SP, "corto del −3x frente a la LRS"): RB[SP]["préstamo_nulo_corto3"],
                    (NDX, "corto del −3x frente a la LRS"): RB[NDX]["préstamo_nulo_corto3"],
                    (SP, "par corto y lever-arb frente a las letras"): RB[SP]["préstamo_nulo_neutral"],
                    (NDX, "par corto y lever-arb frente a las letras"): RB[NDX]["préstamo_nulo_neutral"]})
    for c in [c for c in bn.columns if c.startswith("Δg")]:
        bn[c] = bn[c].astype(float) * 100
    bn = bn.rename(columns=lambda c: c.replace("Δg con", "Δg (pp) con"))
    tab("Sin remunerar el producto de los cortos: Δg (pp/año) según el préstamo y préstamo que anula la ventaja "
        "(ventana principal de cada mercado)", bn, digits=2)
    f_tr = RB[SP]["familias"].set_index(["familia", "ventana", "frente a"])
    sp_tr = f_tr.loc[("tendencia: todas", "1938–2026", "LRS 3x (original)")]
    rep.p(f"**Lectura.** (1) La LRS con corto diario del −3x (2% de préstamo) pasa de "
          f"{spp(rbv(SP, d_short, '1938–2026', rb=REBATES[0]), 2)} a {spp(d_nr, 2)}/año frente a la original en el "
          f"S&P 500 (p_RW {peq(rw3_nr.loc[d_short, 'p_RW'])} en la familia de tendencia); el SPA de esa familia sin "
          f"remuneración da p {peq(sp_tr['p_SPA_c'])} (mejor: «{sp_tr['mejor']}»)"
          + (". La ganancia de implementación de la sección 2 existe solo si el bróker remunera el producto del corto."
             if not d_nr_ok else ". La ventaja sobrevive también sin remuneración.")
          + f" (2) Par corto y lever-arb con préstamo realista frente a las letras: con el producto remunerado "
          f"sobreviven a Romano–Wolf {len(surv_s)} en el S&P 500 y {len(surv_n)} en el Nasdaq-100; sin remunerar, "
          f"{len(surv_s_nr)} ({', '.join(surv_s_nr) if surv_s_nr else 'ninguna'}) y {len(surv_n_nr)} "
          f"({', '.join(surv_n_nr) if surv_n_nr else 'ninguna'}). "
          + ("Que «una cosecha sobreviva a la búsqueda» depende, por tanto, de ese supuesto."
             if (len(surv_s_nr) + len(surv_n_nr)) < (len(surv_s) + len(surv_n)) else
             "Las supervivientes no dependen de ese supuesto.")
          + " Las semanales son apuestas de cociente de varianzas; el supuesto les quita la letra sobre el producto "
            "del corto, del orden de la letra media de la muestra al año.")

    # ── 8. censo ──
    rep.h("8. Para el censo (R19 y R25)")
    rep.p("`r22_inverse.census_series(D=None, start=\"1938-02-04\", with_meta=False)` devuelve {«R22 · nombre»: serie} "
          "con la rentabilidad diaria neta de cada configuración probada en el S&P 500, sobre el índice de "
          "D['S&P 500']['u'] (el del censo), sin las referencias de R13 ni las series idénticas a ellas. Las series "
          "son NaN antes de `start` y, desde `start`, una ruina deja la riqueza a cero hasta el final (octubre de "
          "1987 en el corto mensual del −3x sin regla de margen y en la LRS de 5x o más). Con with_meta=True devuelve "
          "también la familia, si la regla vende en corto y si es operable con ETFs cotizados (≤ 3x). Las "
          "configuraciones con la misma serie (reglas de margen que no llegan a activarse) se devuelven todas; "
          "r19.dedupe las cuenta una vez. El censo ampliado (R25) las pide con start=None y absorbe la ruina desde "
          "el inicio de cada una de sus ventanas.")

    # ── 9. advertencias ──
    rep.h("9. Advertencias")
    rep.p("\n".join([
        "- **Préstamo de títulos.** Su coste y su disponibilidad no están en los datos: se usa una rejilla. Los ETFs "
        "inversos suelen ser más caros de pedir prestados y el coste sube en las crisis, justo cuando el par pierde.",
        "- **Margen.** El mantenimiento (30% × |L| del valor del corto) se comprueba solo al cierre; los brókeres "
        "pueden exigir más (hasta el 100% en ETFs apalancados) y avisar durante la sesión. La regla FINRA 4210 se "
        "aplica como la describió la exploración previa y no se ha verificado aquí contra el texto de la norma.",
        "- **Fondos sintéticos.** Antes de su lanzamiento (SH, SDS, SSO, PSQ, QID y QLD en 2006; UPRO y SPXU en "
        "2009; TQQQ y SQQQ en 2010) todos los fondos son sintéticos, con un s calibrado dentro de muestra. El inverso "
        "cobra f sobre el nocional corto; antes de 2010 f − rf fue de varias décimas, lo que favorece a los cortos "
        "sintéticos frente a la letra.",
        "- **Producto del corto.** Cobra la letra − 0,5%; muchos minoristas no cobran nada. La sensibilidad sin "
        "remuneración (sección 7) muestra que las conclusiones sobre el corto del −3x, el par corto y el lever-arb "
        "dependen de ese supuesto. No se modelan impuestos, la recompra forzosa del título (recall) ni el cierre "
        "de fondos.",
        "- **Calendario de reajuste.** El último día de cada semana, mes o trimestre se fija con las sesiones de la "
        "base, así que la semana del 11-S se reajusta el lunes 10-9-2001 (al cierre nadie sabía que sería la última "
        "sesión de la semana). Solo cambia el día del reajuste, no ninguna señal; R21 corrige lo mismo en su MACD "
        "semanal.",
        "- **Futuros.** Rinden u − f sin base y con 1 pb por unidad; los del S&P 500 existen desde 1982 y los del "
        "Nasdaq-100 desde 1996, así que el lever-arb anterior es hipotético. Los 5 pb por unidad son un coste actual.",
        "- **Kelly y la curva.** Todo L* y todo máximo de la curva usan la muestra completa: describen, no deciden. "
        "La señal de precio es una sensibilidad del laboratorio, no la regla de Gayed y Bilello.",
        "- **Semiconductores.** No se estudian aquí: el par SOXL/SOXS ya está en R5 (con reajuste trimestral se "
        "arruina) y la base no tiene inversos de −1x y −2x de semis para la familia (a). R21 y R23 sí los incluyen.",
        "- **Nasdaq-100.** Antes de 1985-10 solo sirve para calentar la media; los contrastes usan la reconstrucción 0 "
        "y las demás solo cambian el primer año (tabla de reconstrucciones).",
        "- **Mecanismo.** El camino se mide sin costes ni intereses y con los pesos del inicio de cada periodo; el "
        "resultado de las carteras sí los incluye.",
    ]))
    rep.save()


if __name__ == "__main__":
    run()
