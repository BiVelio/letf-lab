"""R32 — Vender el inverso en los repuntes (H3, H4 y H5 del artículo 6).

Pregunta (pre-registro del artículo 6, letf/preregistro/articulo6_inversos.md, commit 8012967).
Un ETF inverso apalancado pierde con el reajuste diario más que el alcista del mismo |L|: frente a
L veces el rendimiento logarítmico del índice, ½(L² − L)·Σr², 6 veces Σr² con L = −3 y 3 veces con
L = +3. ¿Se cobra esa pérdida vendiendo en corto el inverso tras un repunte del mercado y
manteniéndolo sin reajustar? Esta línea contrasta, tal como están escritas:

- **H3 · Cortos en repuntes.** La familia: vender en corto el −3x por una fracción s del capital
  (s = 1/3 y s = 1) tras un repunte del índice —subida de al menos k% en n sesiones, k ∈ {3, 5,
  10}, n ∈ {5, 10, 21}—, en tres estados (índice sobre su media de 200 sesiones, bajo ella, sin
  condición), y mantenerlo h ∈ {5, 21, 63} sesiones sin reajuste: 2 × 3 × 3 × 3 × 3 = 162 reglas
  por mercado. H3 dice que ninguna supera tras la búsqueda (p del SPA < 0,10), con el préstamo
  mediano de cada ticker, a (a) la caja, (b) el +3x comprado las mismas sesiones con la misma
  exposición inicial, (c) la LRS de Gayed y Bilello del artículo 2.
- **H4 · Dónde sale la ventaja.** La ventaja del corto sin reajuste frente al mismo corto
  reajustado cada día se explica por el cociente de varianzas del periodo: coeficiente positivo
  sobre (1 − VR) con errores HAC, y la ventaja desaparece cuando VR ≥ 1.
- **H5 · La cola.** La probabilidad de perder más del 50% del capital dentro del periodo de
  tenencia es mayor en las reglas de H3 que en el +3x comprado con la misma exposición.

H1 y H2 (mecánica y mediana frente a media) son de R31. Esta línea construye sus propios datos
(los fondos de R22 y los ETFs reales de la caché), así que no depende de R31.

Mercados y fondos:
- Historia larga: S&P 500 desde 1928 (primera decisión el 17-10-1928, cuando existe la media de
  200 sesiones) y Nasdaq-100 desde 1985-10 (decisión desde el 30-9-1985; la reconstrucción previa
  solo calienta la media), con los fondos de R22 (r22.hybrid): SPXU y SQQQ sintéticos antes de su
  lanzamiento (rf + L·(u − f) − |L|·s − gasto, s calibrado con su vida real, validado en R22) y
  reales después; el +3x es el del laboratorio (d["x3"]: sintético y UPRO/TQQQ reales).
- Pares reales −3x del pre-registro, cada uno con su vida real (2006–2026 como mucho): SQQQ/TQQQ,
  SPXU/UPRO, SOXS/SOXL, TZA/TNA, FAZ/FAS, LABD/LABU y SDOW/UDOW, con el +3x del mismo índice como
  alcista de (b). Con las dos historias largas son los 9 mercados en que se juzgan H3, H4 y H5 («vender
  en corto el −3x»). Además, una familia conjunta de los 6 pares −3x vivos desde el 12-3-2010 (todos
  salvo LABD, lanzado en 2015) en su ventana común, y Holm entre los 7 pares (los dos, no
  pre-registrados).
- Exploratorios, fuera del alcance de H3–H5 y de todo recuento: SDS/SSO y QID/QLD (−2x), SH y PSQ
  (−1x; el alcista es el índice del laboratorio, sin gasto de fondo) y TECS/TECL (no nombrado en el
  pre-registro), con el alcista del mismo |L| en (b). TMV, ERY y DUST quedan fuera: bonos y fondos que
  solo son −2x desde 2020.
- Señal y estado con el índice de rentabilidad total (r13.signal_index; en los sectores, el ETF
  1x empalmado con su índice largo, common.long_underlying) y r13.above_sma. La LRS 3x (original)
  de cada índice con r13.run_weights y el +3x de ese índice (UPRO o TQQQ en los pares de −2x y −1x).
- Revisión del 30-09-2026 (después de ver los resultados): la primera versión mezclaba los
  exploratorios en los recuentos de H3–H5, en Holm (12 pares) y en la familia conjunta (11 pares); se
  separaron. Ningún veredicto de H3 cambió (ver «Desviaciones y lecturas»).

Protocolo (el común del pre-registro y el laboratorio):
- Señal al cierre de t, orden al cierre de t+1, h sesiones en la posición y recompra al cierre de
  la sesión h; mientras hay posición no se miran señales, y la siguiente decisión es la del cierre
  de la recompra (entre dos operaciones hay al menos una sesión en caja: un corto nuevo se vende por
  s del capital de ese momento).
- Simulador de R29 (r29.drift_sim, el de R22 con «nunca»): 5 pb por unidad negociada; el capital
  cobra la letra; el préstamo se paga sobre el valor vivo del corto; el producto del corto cobra la
  letra − 0,5% o nada; mantenimiento FINRA de 30% × |L| del valor del corto (90% en un −3x): si se
  incumple al cierre, el corto se reduce ese cierre a s del capital; ruina si el capital llega a 0
  (absorbente en la ventana).
- Préstamo: la distribución diaria de IBKR en iBorrowDesk (~12 meses) de cada ticker, rejilla
  mínimo / mediana / p90; SPXU y SQQQ para la historia larga del S&P 500 y del Nasdaq-100.
- Igual exposición: el corto de s en el −L× tiene al entrar la exposición s·L al índice, la misma
  que s en el alcista del mismo |L|. El alcista se compra y se vende en las mismas sesiones, con
  los mismos costes y sin reajuste.
- Inferencia: Δg (crecimiento logarítmico anual de la regla menos el de la referencia) con
  bootstrap estacionario emparejado (bloque medio 63, 2.000 remuestreos, los mismos para toda la
  familia), p unilateral normal con el error del bootstrap (H0: Δg ≤ 0), Holm, step-down de Romano
  y Wolf (r19.romano_wolf) y SPA de Hansen (stats.spa_test) por familia. H3 se rechaza en un
  mercado y una comparación si el SPA da p < 0,10 con el préstamo mediano.
- H4: por operación (fechas de entrada únicas de las reglas de H3 con cada s y h), ganancia del
  corto sin reajuste menos la del mismo corto reajustado cada día (r29.drift_sim con reajuste
  diario, mismas entradas, préstamo mediano y producto remunerado), sobre 1 − VR del periodo,
  VR = (Σx)²/Σx² con x = u − f en las h sesiones; errores de Hansen-Hodrick con h − 1 retardos
  (stats.hac_overlap; las operaciones de reglas distintas se solapan). La ventaja «desaparece» con
  VR ≥ 1 si su media condicionada no es mayor que 0 (p unilateral ≥ 0,05). Como comprobación
  mecánica, lo mismo con todas las sesiones como entrada y sin costes ni intereses.
- H5: por operación, pérdida de más del 50% del capital en algún cierre de la tenencia (y al
  final), corto frente al alcista de las mismas sesiones; contraste de McNemar (binomial exacta)
  sobre las operaciones discordantes.
- Episodios: 1987, 2000–2002, 2008, 2020, 2022 y agosto de 2024 (fechas aproximadas del máximo
  previo a la recuperación), y Δg de la mejor regla sin cada episodio.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from letf_lab import metrics, stats
from letf_lab.data import history, vix
from letf_lab.report import FIG_W, Report, comma_fig, legend_band, num_es, pct_es

import matplotlib.dates as mdates  # noqa: E402  (después de report, que fija el motor Agg)
import matplotlib.pyplot as plt  # noqa: E402
from letf_lab.research import r12_long_ml as r12
from letf_lab.research import r13_lrs as r13
from letf_lab.research import r19_census as r19
from letf_lab.research import r22_inverse as r22
from letf_lab.research import r29_short_inverse as r29
from letf_lab.research.common import long_underlying
from letf_lab.research.r26_vix_data import Tables, dmy, pp
from letf_lab.universe import BY_TICKER

SEED = 32
B = 2000                         # remuestreos del bootstrap (como R19, R22 y R29)
BLOCK = 63                       # bloque medio del bootstrap estacionario (sesiones)
LAG = r13.EXEC_LAG               # señal al cierre de t, orden al cierre de t+1
COST_BPS = r13.COST_BPS          # 5 pb por unidad negociada
HAIRCUT = vix.PROCEEDS_SPREAD    # el producto del corto cobra la letra − 0,5% (o nada)
MAINT = vix.FINRA_MAINT          # 30% × |L| del valor del corto (tope 100%)
END = vix.END
CLIP = r22.CLIP
SPA_LEVEL = 0.10                 # umbral del pre-registro para H3
LOSS = 0.50                      # H5: perder más del 50% del capital

K_UP = (0.03, 0.05, 0.10)
N_UP = (5, 10, 21)
STATES = ("sobre", "bajo", "sin")
STATE_ES = {"sobre": "sobre la media", "bajo": "bajo la media", "sin": "sin condición"}
HOLDS = (5, 21, 63)
SIZES = (1.0 / 3.0, 1.0)
LEVELS = ("mín", "mediana", "p90")
REBATES = (True, False)
REBATE_ES = r29.REBATE_ES
COMPS = ("caja", "alcista", "LRS")
COMP_ES = {"caja": "(a) caja", "alcista": "(b) alcista las mismas sesiones", "LRS": "(c) LRS 3x"}
H3_ES = {"caja": "(a) caja", "alcista": "(b) +3x comprado las mismas sesiones", "LRS": "(c) LRS 3x"}

SP, NDX = r22.SP, r22.NDX
LONG = {SP: "SPXU", NDX: "SQQQ"}                    # inverso de la historia larga
# pares reales: (inverso, alcista de la comparación (b), índice)
REAL = (("SQQQ", "TQQQ", NDX), ("SPXU", "UPRO", SP), ("SDS", "SSO", SP), ("QID", "QLD", NDX),
        ("SH", "SPY", SP), ("PSQ", "QQQ", NDX), ("SOXS", "SOXL", "SOXX"), ("TZA", "TNA", "IWM"),
        ("FAZ", "FAS", "XLF"), ("LABD", "LABU", "XBI"), ("SDOW", "UDOW", "DIA"), ("TECS", "TECL", "XLK"))
PREREG = ("SQQQ", "SPXU", "SOXS", "TZA", "FAZ", "LABD", "SDOW")   # los −3x reales del pre-registro (H3–H5)
POOL_OUT = ("LABD",)             # fuera de la familia conjunta (vida desde 2015)
POOL = "6 pares −3x del pre-registro, ventana común"
INDEX_1X = {"SPY": SP, "QQQ": NDX}   # SH y PSQ se comparan con el índice del laboratorio (1x, sin gasto)
LRS3 = {SP: "UPRO", NDX: "TQQQ"}

EPISODES = {"1987": ("1987-08-25", "1987-12-31"), "2000–2002": ("2000-03-10", "2003-03-11"),
            "2008": ("2007-10-09", "2009-06-30"), "2020": ("2020-02-19", "2020-06-08"),
            "2022": ("2022-01-03", "2022-12-30"), "agosto de 2024": ("2024-07-16", "2024-08-30")}

INK, MUTED = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, VIOLET, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#9a9994", "#e34948"


# ───────────────────────────── reglas ─────────────────────────────

def rule_keys() -> list[tuple]:
    """Las 162 reglas: (s, k, n, estado, h)."""
    return [(s, k, n, st, h) for s in SIZES for k in K_UP for n in N_UP for st in STATES for h in HOLDS]


def s_label(s: float) -> str:
    return "1/3" if s < 1 else "1"


def rule_name(key: tuple) -> str:
    s, k, n, st, h = key
    return f"s = {s_label(s)} · +{round(k * 100)}% en {n} · {STATE_ES[st]} · {h} ses."


def rally(sx: pd.Series, n: int, k: float) -> pd.Series:
    """Repunte al cierre de t: el índice subió al menos k en las n sesiones anteriores."""
    return (sx / sx.shift(n) - 1 >= k - 1e-12) & sx.shift(n).notna()


def state_mask(state: pd.Series, st: str) -> pd.Series:
    """Estado al cierre de t (r13.above_sma: 1 sobre la media de 200, 0 bajo ella, NaN sin media)."""
    if st == "sobre":
        return state == 1
    if st == "bajo":
        return state == 0
    if st == "sin":
        return state.notna()
    raise ValueError(st)


def signal(sx: pd.Series, state: pd.Series, k: float, n: int, st: str, idx: pd.DatetimeIndex) -> np.ndarray:
    """Días de decisión (en idx) con repunte de k en n sesiones y el estado pedido."""
    s = (rally(sx, n, k) & state_mask(state, st)).reindex(idx)
    return s.fillna(False).to_numpy(bool)


def schedule(sig: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Posición tras cada cierre y cierres de entrada. Una señal al cierre i (en caja) vende al cierre
    e = i + 1 y mantiene tras los cierres e … e + h − 1 (rendimientos de e + 1 a e + h); recompra al
    cierre e + h. Las señales con posición abierta se ignoran; la siguiente decisión posible es la
    del cierre de la recompra."""
    n = len(sig)
    held = np.zeros(n, bool)
    entries = []
    nxt = 0
    for i in np.flatnonzero(sig):
        if i < nxt or i + 1 >= n:
            continue
        e = i + 1
        held[e:e + h] = True
        entries.append(e)
        nxt = e + h
    return held, np.asarray(entries, dtype=int)


def target(held: np.ndarray, idx: pd.DatetimeIndex, w: float) -> pd.Series:
    """Objetivo decidido al cierre de t para r29.drift_sim (lag 2): la posición tras el cierre t + 1."""
    return pd.Series(w * np.r_[held[1:], False].astype(float), index=idx)


# ───────────────────────────── datos ─────────────────────────────

def borrow_grid(tickers) -> pd.DataFrame:
    """Rejilla de préstamo de cada ticker (fracción anual): mínimo, mediana y p90 de la tasa diaria de
    IBKR en iBorrowDesk (vix.borrow_summary), con sus fechas."""
    bs = vix.borrow_summary(list(tickers))
    g = bs[["mín", "mediana", "p90", "máx"]] / 100.0
    return g.join(bs[["desde", "hasta", "días", "disponibles_mín"]])


def _lab_base(D: dict, mkt: str) -> dict:
    d = D[mkt]
    sx = r13.signal_index(d)
    state = r13.above_sma(sx)
    return {"u": d["u"], "rf": d["rf"], "f": d["f"], "sx": sx, "state": state, "lrs": r13.run_weights(d, state),
            "d": d}


def _sector_base(inv: str, bull: str) -> dict:
    """Índice del sector: ETF 1x empalmado con su índice largo (common.long_underlying), la letra y
    el tipo de financiación del laboratorio (history.rf_long, history.funding_long) y la LRS 3x con el
    alcista real."""
    u = long_underlying(inv)
    u = u[u.index <= END]
    idx = u.index
    rf, f = history.rf_long(idx), history.funding_long(idx)
    sx = (1 + u.fillna(0.0)).cumprod()
    state = r13.above_sma(sx)
    d = {"u": u, "rf": rf, "f": f, "x3": r12.real_letf(bull).reindex(idx)}
    return {"u": u, "rf": rf, "f": f, "sx": sx, "state": state, "lrs": r13.run_weights(d, state), "d": d}


def pair_name(inv: str, bull: str) -> str:
    """Nombre del par real; con SH y PSQ el alcista es el índice del laboratorio (sin gasto), no el ETF."""
    return f"{inv}/{INDEX_1X[bull]} 1x" if bull in INDEX_1X else f"{inv}/{bull}"


def bull_label(bull: str) -> str:
    return f"{INDEX_1X[bull]} 1x (índice del laboratorio, sin gasto)" if bull in INDEX_1X else bull


def in_scope(name: str) -> bool:
    """Mercados de H3–H5 tal como están escritas («vender en corto el −3x»): las dos historias largas y los
    siete −3x reales del pre-registro. SDS, QID (−2x), SH, PSQ (−1x) y TECS (ampliación) son exploratorios."""
    return is_long(name) or name.split("/")[0] in PREREG


def _market(name, kind, inv, bull, L, Lb, base, X, a, grid, index_name) -> dict:
    idx = base["u"].index
    idx = idx[(idx >= pd.Timestamp(a)) & (idx <= END)]
    X = X.reindex(idx)
    rf = base["rf"].reindex(idx)
    # la LRS decide por primera vez con la primera media de 200 sesiones: hasta su primera posición está en letras
    lrs = base["lrs"].reindex(idx)
    lrs = lrs.where(lrs.notna() | (idx > lrs.first_valid_index()), rf)
    return {"name": name, "kind": kind, "inv": inv, "bull": bull, "L": L, "Lb": Lb, "index": index_name,
            "idx": idx, "a": idx[0], "X": X, "rf": rf, "u": base["u"].reindex(idx),
            "f": base["f"].reindex(idx), "sx": base["sx"], "state": base["state"], "lrs": lrs,
            "borrow": {lv: float(grid[lv]) for lv in LEVELS}, "borrow_tk": grid.name,
            "maint": min(MAINT * abs(L), 1.0)}


def build_markets(D: dict | None = None, only: tuple | None = None) -> dict:
    """Los dos mercados de historia larga y los pares reales (only: nombres a construir)."""
    D = D or r13.load()
    P = r22.fund_params(D)
    grid = borrow_grid(sorted({r[0] for r in REAL}))
    out = {}
    for mkt, inv in LONG.items():
        name = f"{mkt} ({inv} sintético)"
        if only and name not in only:
            continue
        base = _lab_base(D, mkt)
        d = base["d"]
        a = base["state"].first_valid_index()
        if d.get("eval_from") is not None:
            ix = d["u"].index
            a = max(a, ix[ix < d["eval_from"]][-1])
        X = pd.DataFrame({inv: r22.hybrid(d, P, inv), "3x": d["x3"]})
        out[name] = _market(name, "historia larga", inv, "3x", -3.0, 3.0, base, X, a, grid.loc[inv], mkt)
    bases = {}
    for inv, bull, ix in REAL:
        name = pair_name(inv, bull)
        if only and name not in only:
            continue
        if ix in (SP, NDX):
            base = bases.setdefault(ix, _lab_base(D, ix))
        else:
            base = _sector_base(inv, bull)
        cal = base["u"].index
        r_inv = r12.real_letf(inv).reindex(cal)
        r_bull = base["u"] if bull in INDEX_1X else r12.real_letf(bull).reindex(cal)
        first = max(r_inv.first_valid_index(), r_bull.first_valid_index())
        a = cal[cal < first][-1]                      # decisión al cierre anterior al primer rendimiento
        X = pd.DataFrame({inv: r_inv, bull: r_bull})
        Lb = 1.0 if bull in INDEX_1X else float(BY_TICKER[bull].leverage)
        out[name] = _market(name, "par real", inv, bull, float(BY_TICKER[inv].leverage), Lb, base, X, a,
                            grid.loc[inv], ix)
    return out


# ───────────────────────────── simulación ─────────────────────────────

def absorb_np(R: np.ndarray) -> np.ndarray:
    """r22.absorb por columnas: tras el primer día con r ≤ −1, rentabilidad 0."""
    R = np.asarray(R, float)
    hit = np.maximum.accumulate(R <= -1.0, axis=0)
    dead = np.vstack([np.zeros((1,) + R.shape[1:], bool), hit[:-1]])
    return np.where(dead, 0.0, np.maximum(R, -1.0))


def logr_np(R: np.ndarray) -> np.ndarray:
    """log(1 + r) con la ruina absorbida y las pérdidas acotadas en CLIP (r22.logr)."""
    return np.log1p(np.maximum(absorb_np(R), CLIP))


def short_sim(m: dict, held: np.ndarray, s: float, borrow: float, rebate: bool, freq: str = "N",
              cost_bps: float = COST_BPS) -> tuple[np.ndarray, dict]:
    inv = m["inv"]
    r, info = r29.drift_sim(target(held, m["idx"], -s).to_frame(inv), m["X"][[inv]], m["rf"], freq, borrow=borrow,
                            maint={inv: m["maint"]}, margin_rule="restaurar", rebate=rebate, cost_bps=cost_bps)
    return r.to_numpy(), info


def long_sim(m: dict, held: np.ndarray, s: float) -> np.ndarray:
    bull = m["bull"]
    r, _ = r29.drift_sim(target(held, m["idx"], s).to_frame(bull), m["X"][[bull]], m["rf"], "N")
    return r.to_numpy()


def trade_table(r: np.ndarray, entries: np.ndarray, h: int) -> pd.DataFrame:
    """Por operación completa (entrada al cierre e, recompra al cierre e + h): rentabilidad sobre el
    capital anterior a la entrada (costes de entrada y salida incluidos) y capital mínimo a cierre
    dentro de la tenencia (una ruina lo deja en CLIP)."""
    n = len(r)
    e = entries[entries + h < n]
    if not len(e):
        return pd.DataFrame(columns=["e", "ret", "min_eq", "ruina"])
    lr = np.log1p(np.maximum(np.nan_to_num(r, nan=0.0), CLIP))
    M = lr[e[:, None] + np.arange(h + 1)[None, :]].cumsum(axis=1)
    ruin = (np.nan_to_num(r, nan=0.0)[e[:, None] + np.arange(h + 1)[None, :]] <= -1.0).any(axis=1)
    return pd.DataFrame({"e": e, "ret": np.expm1(M[:, -1]), "min_eq": np.exp(M.min(axis=1)), "ruina": ruin})


def simulate(m: dict, keys: list | None = None, levels=LEVELS, rebates=REBATES) -> dict:
    """Todas las reglas de un mercado: rendimientos diarios (filas = m["idx"]) del corto con cada
    préstamo y remuneración, del alcista de las mismas sesiones y del corto reajustado cada día
    (préstamo mediano, producto remunerado); entradas y márgenes de cada regla."""
    keys = keys or rule_keys()
    idx = m["idx"]
    n, K = len(idx), len(keys)
    sigs = {}
    S = {(lv, rb): np.full((n, K), np.nan) for lv in levels for rb in rebates}
    Lg, Dy = np.full((n, K), np.nan), np.full((n, K), np.nan)
    entries, held_all, margin = {}, np.zeros((n, K), bool), {}
    for j, key in enumerate(keys):
        s, k, nn, st, h = key
        sig = sigs.setdefault((k, nn, st), signal(m["sx"], m["state"], k, nn, st, idx))
        held, ent = schedule(sig, h)
        entries[key], held_all[:, j] = ent, held
        for lv in levels:
            for rb in rebates:
                S[(lv, rb)][:, j], info = short_sim(m, held, s, m["borrow"][lv], rb)
                if lv == "mediana" and rb:
                    margin[key] = {"incumplimientos": info["incumplimientos"], "ruinas": info["ruinas"]}
        Lg[:, j] = long_sim(m, held, s)
        Dy[:, j], _ = short_sim(m, held, s, m["borrow"]["mediana"], True, freq="D")
    return {"keys": keys, "names": [rule_name(k) for k in keys], "short": S, "long": Lg, "daily": Dy,
            "entries": entries, "held": held_all, "margin": margin}


# ───────────────────────────── inferencia ─────────────────────────────

def family(X: np.ndarray, names: list, rng, spa: bool = True, B_: int = B) -> pd.DataFrame:
    """r29.family_test sobre una matriz de diferencias diarias de log(1 + r) (filas = sesiones comunes,
    columnas = reglas): Δg anual, IC90 percentil, p unilateral normal con el error del bootstrap
    (H0: Δg ≤ 0), Holm, Romano–Wolf y, en attrs, el SPA de Hansen (stats.spa_test)."""
    X = np.asarray(X, float)
    m = X.mean(axis=0)
    draws = r19.boot_means(X, B_, BLOCK, rng)
    se = draws.std(axis=0, ddof=1)
    safe = np.where(se > 0, se, np.inf)
    t = m / safe
    p = norm.sf(t)
    lo, hi = np.percentile(draws, [5, 95], axis=0)
    rw = r19.romano_wolf(t, (draws - m) / safe)
    out = pd.DataFrame({"Δg": m * 252, "IC90_bajo": lo * 252, "IC90_alto": hi * 252, "t": t, "p": p,
                        "p_Holm": stats.holm(p), "p_RW": rw}, index=names)
    out.attrs.update(sesiones=len(X))
    if spa:
        res = stats.spa_test(X, BLOCK, B_, rng)
        j = int(m.argmax())
        out.attrs.update(p_SPA=res["p_SPA_c"], t_SPA=res["t_SPA"], mejor=names[j], Δg_mejor=float(m[j] * 252))
    return out


def comparisons(m: dict, sim: dict, lv: str, rb: bool, rows: slice | np.ndarray | None = None) -> dict:
    """Matrices de diferencias diarias de log(1 + r) de cada regla frente a (a) la caja, (b) el alcista
    de las mismas sesiones y (c) la LRS 3x, en (a, fin] (sin la fila de la decisión inicial)."""
    sl = slice(1, None) if rows is None else rows
    LS = logr_np(sim["short"][(lv, rb)][sl])
    cash = np.log1p(m["rf"].to_numpy()[sl])
    lrs = logr_np(m["lrs"].to_numpy()[sl][:, None])[:, 0]
    return {"caja": LS - cash[:, None], "alcista": LS - logr_np(sim["long"][sl]), "LRS": LS - lrs[:, None]}


def rule_stats(m: dict, sim: dict, rows=slice(1, None)) -> pd.DataFrame:
    """Operativa y riesgo de cada regla con el préstamo mediano y el producto remunerado."""
    out = {}
    R = sim["short"][("mediana", True)]
    rf = m["rf"]
    for j, key in enumerate(sim["keys"]):
        s, k, n, st, h = key
        r = pd.Series(absorb_np(R[rows, j][:, None])[:, 0], index=m["idx"][rows])
        tt = trade_table(R[:, j], sim["entries"][key], h)
        tl = trade_table(sim["long"][:, j], sim["entries"][key], h)
        out[sim["names"][j]] = {"s": s_label(s), "k": k, "n": n, "estado": STATE_ES[st], "h": h,
                                "operaciones": len(sim["entries"][key]),
                                "en posición": float(sim["held"][rows, j].mean()),
                                "CAGR": metrics.cagr(r), "maxDD": metrics.max_dd(r), "sharpe": metrics.sharpe(r, rf),
                                "ganancia media": float(tt.ret.mean()) if len(tt) else np.nan,
                                "aciertos": float((tt.ret > 0).mean()) if len(tt) else np.nan,
                                "peor operación": float(tt.ret.min()) if len(tt) else np.nan,
                                "P(pérdida > 50%)": float((tt.min_eq < 1 - LOSS).mean()) if len(tt) else np.nan,
                                "P(pérdida > 50%) alcista": float((tl.min_eq < 1 - LOSS).mean()) if len(tl) else np.nan,
                                "incumplimientos": sim["margin"][key]["incumplimientos"],
                                "ruinas": sim["margin"][key]["ruinas"]}
    return pd.DataFrame(out).T


def evaluate(m: dict, sim: dict, rng, levels=LEVELS, spa_levels=("mín", "mediana")) -> dict:
    """Familias de un mercado: por comparación, préstamo y remuneración (SPA con los préstamos de
    spa_levels; con el p90, solo Holm y Romano–Wolf: el préstamo solo resta)."""
    fam = {}
    for lv in levels:
        for rb in REBATES:
            C = comparisons(m, sim, lv, rb)
            for c in COMPS:
                fam[(c, lv, rb)] = family(C[c], sim["names"], rng, spa=lv in spa_levels)
    return fam


# ───────────────────────────── H4: cociente de varianzas ─────────────────────────────

def vr_of(x: np.ndarray, e: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """VR = (Σx)²/Σx² y Σx² en las sesiones e + 1 … e + h de cada entrada."""
    C1 = np.r_[0.0, np.cumsum(np.nan_to_num(x))]
    C2 = np.r_[0.0, np.cumsum(np.nan_to_num(x) ** 2)]
    S1 = C1[e + h + 1] - C1[e + 1]
    S2 = C2[e + h + 1] - C2[e + 1]
    return S1 ** 2 / np.where(S2 > 0, S2, np.nan), S2


def h4_trades(m: dict, sim: dict) -> dict:
    """Operaciones únicas de las reglas de H3 por (s, h): ganancia del corto sin reajuste (préstamo
    mediano, producto remunerado) y del mismo corto reajustado cada día, VR y Σx² del periodo, y el
    capital mínimo del corto y del alcista (H5)."""
    x = (m["u"] - m["f"]).to_numpy()
    R = sim["short"][("mediana", True)]
    out = {}
    for s in SIZES:
        for h in HOLDS:
            parts = []
            for j, key in enumerate(sim["keys"]):
                if key[0] != s or key[4] != h:
                    continue
                ent = sim["entries"][key]
                tn, td, tl = (trade_table(R[:, j], ent, h), trade_table(sim["daily"][:, j], ent, h),
                              trade_table(sim["long"][:, j], ent, h))
                if not len(tn):
                    continue
                parts.append(pd.DataFrame({"e": tn.e, "sin reajuste": tn.ret, "diario": td.ret.to_numpy(),
                                           "min_eq": tn.min_eq, "ruina": tn.ruina,
                                           "alcista": tl.ret.to_numpy(), "min_eq_alcista": tl.min_eq.to_numpy()}))
            if not parts:
                continue
            t = pd.concat(parts).drop_duplicates("e").sort_values("e").reset_index(drop=True)
            vr, s2 = vr_of(x, t.e.to_numpy(), h)
            t["VR"], t["Σx²"] = vr, s2
            t["Δ"] = t["sin reajuste"] - t["diario"]
            t["fecha"] = m["idx"][t.e.to_numpy()]
            out[(s, h)] = t
    return out


def h4_all_sessions(m: dict, s: float, h: int) -> pd.DataFrame:
    """Comprobación mecánica: cada sesión como entrada, sin costes ni intereses ni mantenimiento. Con el
    exceso del inverso sobre la letra y, sin reajuste: −s·(Π(1 + y) − 1); reajustado cada día:
    Π(1 − s·y) − 1."""
    y = (m["X"][m["inv"]] - m["rf"]).to_numpy()[1:]
    x = (m["u"] - m["f"]).to_numpy()[1:]
    ok = ~np.isnan(y)
    y, x = np.where(ok, y, 0.0), np.where(ok, x, 0.0)
    ly = np.r_[0.0, np.cumsum(np.log1p(np.maximum(y, -0.999999)))]
    ld = np.r_[0.0, np.cumsum(np.log1p(np.maximum(-s * y, -0.999999)))]
    n = len(y)
    st = np.arange(n - h + 1)
    N = -s * np.expm1(ly[st + h] - ly[st])
    Dd = np.expm1(ld[st + h] - ld[st])
    vr, s2 = vr_of(np.r_[0.0, x], st, h)
    return pd.DataFrame({"Δ": N - Dd, "VR": vr, "Σx²": s2}, index=m["idx"][st])


def h4_fit(t: pd.DataFrame, h: int) -> dict:
    """Δ sobre (1 − VR) con Hansen-Hodrick (h − 1 retardos; recurso Newey-West 2h) y las medias de Δ con
    VR < 1 y con VR ≥ 1 (regresión sobre las dos indicadoras, mismos errores)."""
    t = t.dropna(subset=["Δ", "VR"])
    y = t["Δ"].reset_index(drop=True)
    X = (1 - t["VR"]).rename("1 − VR").reset_index(drop=True).to_frame()
    fit = stats.ols_hac(y, X, **stats.hac_overlap(h))
    I = pd.DataFrame({"VR < 1": (t["VR"] < 1).astype(float).to_numpy(),
                      "VR ≥ 1": (t["VR"] >= 1).astype(float).to_numpy()})
    I = I.loc[:, I.sum() > 0]
    fm = stats.ols_hac(y, I, const=False, **stats.hac_overlap(h))
    cm = dict(zip(fm.names, zip(fm.beta, fm.t)))
    Z = (t["Σx²"] * (1 - t["VR"])).rename("Σx² − (Σx)²").reset_index(drop=True).to_frame()
    fz = stats.ols_hac(y, Z, **stats.hac_overlap(h))
    hi = int((t["VR"] >= 1).sum())
    return {"operaciones": len(t), "Δ medio": float(y.mean()), "β (1 − VR)": float(fit.beta[1]), "t β": float(fit.t[1]),
            "p β (> 0)": float(norm.sf(fit.t[1])), "R²": fit.r2,
            "Δ con VR < 1": float(cm.get("VR < 1", (np.nan,))[0]), "t (VR < 1)": float(cm.get("VR < 1", (np.nan, np.nan))[1]),
            "Δ con VR ≥ 1": float(cm.get("VR ≥ 1", (np.nan,))[0]), "t (VR ≥ 1)": float(cm.get("VR ≥ 1", (np.nan, np.nan))[1]),
            "p (VR ≥ 1, > 0)": float(norm.sf(cm["VR ≥ 1"][1])) if hi else np.nan, "con VR ≥ 1": hi,
            "R² sobre Σx² − (Σx)²": fz.r2, "estimador": fit.estimator}


def h4_verdict(r: dict) -> str:
    pos = r["β (1 − VR)"] > 0 and r["p β (> 0)"] < 0.05
    gone = not (r["Δ con VR ≥ 1"] > 0 and r["p (VR ≥ 1, > 0)"] < 0.05)
    return "se sostiene" if pos and gone else ("β no positivo" if not pos else "no desaparece con VR ≥ 1")


# ───────────────────────────── H5: la cola ─────────────────────────────

def h5_row(t: pd.DataFrame) -> dict:
    """P(pérdida > 50% en algún cierre de la tenencia) del corto y del alcista de las mismas sesiones,
    y McNemar (binomial exacta sobre las operaciones discordantes, unilateral: corto > alcista)."""
    a = (t["min_eq"] < 1 - LOSS).to_numpy()
    b = (t["min_eq_alcista"] < 1 - LOSS).to_numpy()
    only_s, only_l = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_s, only_s + only_l, 0.5, alternative="greater").pvalue if only_s + only_l else np.nan
    return {"operaciones": len(t), "P corto": float(a.mean()), "P alcista": float(b.mean()),
            "P corto al final": float((t["sin reajuste"] < -LOSS).mean()),
            "P alcista al final": float((t["alcista"] < -LOSS).mean()),
            "solo el corto": only_s, "solo el alcista": only_l, "p McNemar": p,
            "ruinas": int(t["ruina"].sum()), "peor corto": float(t["min_eq"].min() - 1),
            "peor alcista": float(t["min_eq_alcista"].min() - 1)}


def h5_verdict(r: dict) -> str:
    if r["P corto"] > r["P alcista"]:
        return "mayor en el corto" + (" (p < 0,05)" if r["p McNemar"] < 0.05 else "")
    if r["P corto"] == r["P alcista"]:
        return "igual" if r["P corto"] > 0 else "igual (ninguna)"
    return "mayor en el alcista"


# ───────────────────────────── episodios ─────────────────────────────

def episode_rows(m: dict, sim: dict) -> pd.DataFrame:
    """Rentabilidad en cada episodio: índice, LRS 3x y la distribución de las 162 reglas (corto con el
    préstamo mediano y el producto remunerado) y de su alcista de las mismas sesiones."""
    idx = m["idx"][1:]
    LS = logr_np(sim["short"][("mediana", True)][1:])
    LL = logr_np(sim["long"][1:])
    lrs = logr_np(m["lrs"].to_numpy()[1:][:, None])[:, 0]
    lu = np.log1p(m["u"].to_numpy()[1:])
    rows = {}
    for ep, (a, b) in EPISODES.items():
        sel = (idx > pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
        if sel.sum() < 5 or idx[0] > pd.Timestamp(a):
            continue
        rs, rl = np.expm1(LS[sel].sum(axis=0)), np.expm1(LL[sel].sum(axis=0))
        act = sim["held"][1:][sel].any(axis=0)
        rows[ep] = {"índice": float(np.expm1(lu[sel].sum())), "LRS 3x": float(np.expm1(lrs[sel].sum())),
                    "corto: mediana": float(np.median(rs)), "corto: peor": float(rs.min()), "corto: mejor": float(rs.max()),
                    "reglas que pierden": float((rs < 0).mean()), "reglas con posición": int(act.sum()),
                    "alcista: mediana": float(np.median(rl)), "alcista: peor": float(rl.min())}
    return pd.DataFrame(rows).T


def without_episodes(m: dict, fam_x: np.ndarray, j: int) -> dict:
    """Δg anual de la columna j de una matriz de diferencias con toda la ventana y sin cada episodio."""
    idx = m["idx"][1:]
    out = {"toda la ventana": float(fam_x[:, j].mean() * 252)}
    for ep, (a, b) in EPISODES.items():
        sel = (idx > pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
        if sel.sum() < 5 or idx[0] > pd.Timestamp(a):
            continue
        out[f"sin {ep}"] = float(fam_x[~sel, j].mean() * 252)
    return out


# ───────────────────────────── cálculo completo ─────────────────────────────

def market_results(m: dict, rng) -> dict:
    """Simulación, familias, operativa, H4, H5 y episodios de un mercado."""
    sim = simulate(m)
    fam = evaluate(m, sim, rng)
    trades = h4_trades(m, sim)
    h4 = {sh: h4_fit(t, sh[1]) for sh, t in trades.items()}
    h4m = {(s, h): h4_fit(h4_all_sessions(m, s, h), h) for s in SIZES for h in HOLDS} if m["kind"] == "historia larga" else {}
    h5 = {sh: h5_row(t) for sh, t in trades.items()}
    keep = {"short": {k: v for k, v in sim["short"].items() if k[0] == "mediana"}, "long": sim["long"],
            "held": sim["held"], "keys": sim["keys"], "names": sim["names"], "entries": sim["entries"]}
    return {"fam": fam, "rules": rule_stats(m, sim), "h4": h4, "h4_mech": h4m, "h5": h5,
            "trades": {k: v for k, v in trades.items() if k[1] == 21}, "episodes": episode_rows(m, sim), "sim": keep}


def pooled(M: dict, R: dict, rng) -> dict:
    """Familia conjunta de los pares −3x del pre-registro vivos desde 2010 (todos salvo LABD) en su
    ventana común, con el préstamo mediano de cada ticker y las dos remuneraciones."""
    names = [k for k, m in M.items() if m["kind"] == "par real" and in_scope(k) and m["inv"] not in POOL_OUT]
    a = max(M[k]["a"] for k in names)
    cal = None
    for k in names:
        ix = M[k]["idx"][1:]
        ix = ix[ix > a]
        cal = ix if cal is None else cal.intersection(ix)
    out = {"pares": names, "desde": a, "sesiones": len(cal)}
    for rb in REBATES:
        mats = {c: [] for c in COMPS}
        cols = []
        for k in names:
            m, sim = M[k], R[k]["sim"]
            rows = m["idx"].get_indexer(cal)
            C = comparisons(m, sim, "mediana", rb, rows=rows)
            for c in COMPS:
                mats[c].append(C[c])
            cols += [f"{m['inv']} · {nm}" for nm in sim["names"]]
        for c in COMPS:
            out[(c, rb)] = family(np.hstack(mats[c]), cols, rng)
    return out


def market_names() -> list[str]:
    """Mercados en orden fijo (la semilla de cada uno depende de su posición, no del orden de cálculo)."""
    return [f"{mkt} ({inv} sintético)" for mkt, inv in LONG.items()] + [pair_name(inv, bull) for inv, bull, _ in REAL]


def _one_market(name: str) -> tuple[str, dict, dict]:
    """Un mercado de principio a fin (para repartir el cálculo entre procesos)."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t1 = time.time()
    m = build_markets(only=(name,))[name]
    res = market_results(m, np.random.default_rng([SEED, market_names().index(name)]))
    res["segundos"] = time.time() - t1
    return name, m, res


def compute(only: tuple | None = None, workers: int = 8) -> dict:
    """Todo el cálculo de R32 (sin informe). Cada mercado tiene su semilla ([SEED, posición]), así que el
    resultado no depende de cuántos procesos se usen (workers = 1: todo en este proceso)."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    names = [n for n in market_names() if not only or n in only]
    if workers > 1 and len(names) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(workers, len(names)), mp_context=mp.get_context("spawn")) as ex:
            out = list(ex.map(_one_market, names))
    else:
        out = [_one_market(n) for n in names]
    M = {n: m for n, m, _ in out}
    R = {n: r for n, _, r in out}
    P = pooled(M, R, np.random.default_rng([SEED, 99])) if not only else None
    grid = borrow_grid(sorted({r[0] for r in REAL}))
    return {"M": M, "R": R, "pool": P, "grid": grid, "segundos": time.time() - t0}


# ───────────────────────────── tablas del informe ─────────────────────────────

def peq(p: float) -> str:
    return r22.peq(p)


def win_label(m: dict) -> str:
    return f"{dmy(m['idx'][1])} → {dmy(m['idx'][-1])}"


def is_long(name: str) -> bool:
    return "sintético" in name


def pool_window(C: dict) -> str:
    P = C["pool"]
    cal = C["M"]["SQQQ/TQQQ"]["idx"]
    return f"{dmy(cal[cal > P['desde']][0])} → {dmy(cal[-1])}"


def h3_verdict(p_rem: float, p_sin: float) -> str:
    a, b = p_rem < SPA_LEVEL, p_sin < SPA_LEVEL
    return "se rechaza" if a and b else ("se sostiene" if not (a or b) else "se rechaza solo remunerado"
                                         if a else "se rechaza solo sin remunerar")


def h3_rows(C: dict, comp: str, scope: bool = True) -> pd.DataFrame:
    """Veredicto de H3 por mercado frente a una referencia: SPA con el préstamo mediano y las dos
    remuneraciones, la mejor regla (remunerado) y los rechazos de Romano–Wolf. scope=True: los mercados del
    pre-registro y la familia conjunta; scope=False: los exploratorios."""
    rows = {}
    items = [(k, R["fam"][(comp, "mediana", True)], R["fam"][(comp, "mediana", False)], win_label(C["M"][k]),
              C["M"][k]["borrow"]["mediana"]) for k, R in C["R"].items() if in_scope(k) == scope]
    if scope and C.get("pool"):
        items.append((POOL, C["pool"][(comp, True)], C["pool"][(comp, False)], pool_window(C), np.nan))
    for k, fr, fn, w, b in items:
        j = fr["Δg"].idxmax()
        rows[k] = {"ventana": w, "préstamo mediano": b, "p_SPA (remunerado)": fr.attrs["p_SPA"],
                   "p_SPA (sin remunerar)": fn.attrs["p_SPA"], "mejor regla": j, "Δg de la mejor (pp)": 100 * fr.loc[j, "Δg"],
                   "IC90 (pp)": r29.ci_text(fr.loc[j, "IC90_bajo"], fr.loc[j, "IC90_alto"]),
                   "reglas con Δg > 0": int((fr["Δg"] > 0).sum()), "reglas": len(fr),
                   "Romano–Wolf 5%": int((fr.p_RW < 0.05).sum()),
                   "H3": h3_verdict(fr.attrs["p_SPA"], fn.attrs["p_SPA"])}
    return pd.DataFrame(rows).T


def holm_pairs(C: dict, comp: str, rb: bool) -> pd.Series:
    """Holm entre los 7 pares −3x reales del pre-registro sobre sus p del SPA (no pre-registrado: la
    búsqueda entre pares)."""
    ks = [k for k in C["R"] if not is_long(k) and in_scope(k)]
    p = np.array([C["R"][k]["fam"][(comp, "mediana", rb)].attrs["p_SPA"] for k in ks])
    return pd.Series(stats.holm(p), index=ks)


def borrow_rows(C: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sensibilidad al préstamo: (b) con detalle y (a) y (c) resumidas (producto remunerado salvo donde se dice)."""
    rb_, ac = {}, {}
    for k, R in C["R"].items():
        m, f = C["M"][k], R["fam"]
        rb_[k] = {"préstamo mín": m["borrow"]["mín"], "mediana": m["borrow"]["mediana"], "p90": m["borrow"]["p90"],
                  "p_SPA con el mín": f[("alcista", "mín", True)].attrs["p_SPA"],
                  "p_SPA con el mín, sin remunerar": f[("alcista", "mín", False)].attrs["p_SPA"],
                  "p_SPA con la mediana": f[("alcista", "mediana", True)].attrs["p_SPA"],
                  **{f"Δg mejor, {lv} (pp)": 100 * f[("alcista", lv, True)]["Δg"].max() for lv in LEVELS},
                  **{f"RW 5%, {lv}": int((f[("alcista", lv, True)].p_RW < 0.05).sum()) for lv in LEVELS}}
        ac[k] = {**{f"{c}: p_SPA con {'el mín' if lv == 'mín' else 'la mediana'}": f[(c, lv, True)].attrs["p_SPA"]
                    for c in ("caja", "LRS") for lv in ("mín", "mediana")},
                 **{f"{c}: Δg mejor con el p90 (pp)": 100 * f[(c, "p90", True)]["Δg"].max() for c in ("caja", "LRS")},
                 **{f"{c}: RW 5% con el p90": int((f[(c, "p90", True)].p_RW < 0.05).sum()) for c in ("caja", "LRS")}}
    return pd.DataFrame(rb_).T, pd.DataFrame(ac).T


REAL_MEAN = "pares −3x reales (media de 7)"
DIMS = (("s", "s = ", lambda v: v), ("h", "h = ", lambda v: f"{v} ses."), ("estado", "", lambda v: v),
        ("k", "k = ", lambda v: f"{round(float(v) * 100)}%"), ("n", "n = ", lambda v: f"{v} ses."))


def marginal(C: dict, comp: str = "alcista") -> pd.DataFrame:
    """Δg medio (pp/año, préstamo mediano, producto remunerado) de las reglas con cada valor de cada
    dimensión; los pares −3x reales del pre-registro, media de los 7 (los exploratorios, aparte)."""
    cols, reals = {}, []
    for k, R in C["R"].items():
        t = R["rules"].join(R["fam"][(comp, "mediana", True)]["Δg"])
        t["Δg"] = t["Δg"].astype(float) * 100
        g = {"todas": t["Δg"].mean()}
        for col, pre, f in DIMS:
            for v, x in t.groupby(col, sort=False)["Δg"].mean().items():
                g[f"{pre}{f(v)}"] = x
        if is_long(k):
            cols[k] = g
        elif in_scope(k):
            reals.append(pd.Series(g))
    order = list(next(iter(cols.values())))
    cols[REAL_MEAN] = pd.concat(reals, axis=1).mean(axis=1).to_dict()
    return pd.DataFrame(cols).reindex(order)


def fees_table(C: dict) -> pd.DataFrame:
    """Lo que la comparación (b) cobra por comisiones en los pares reales. Con fondos que rinden
    rf ± |L|·(u − f) − c (c = gasto + |L| × sobrecoste del swap calibrado con la vida real del ETF como en
    R22), el corto
    de s en el inverso menos s en el alcista del mismo |L| gana al día, a primer orden y mientras la
    posición está abierta, s·(c_inverso + c_alcista − haircut − préstamo): los dos fondos cobran por encima
    de su réplica y el corto se queda con ambas cosas. El resto del Δg es el camino (H4) y los costes de
    negociar."""
    rows = {}
    for k, m in C["M"].items():
        if is_long(k):
            continue
        d = {"u": m["u"], "rf": m["rf"], "f": m["f"]}

        def cost(tk, L):
            if tk in INDEX_1X:
                return 0.0
            er = BY_TICKER[tk].expense_ratio
            return abs(L) * r22.calibrate_spread(m["X"][tk].iloc[1:], d, L, er) + er

        ci, cb = cost(m["inv"], m["L"]), cost(m["bull"], m["Lb"])
        b = m["borrow"]["mediana"]
        R = C["R"][k]
        dg = R["fam"][("alcista", "mediana", True)]["Δg"].astype(float)
        r = R["rules"].loc[dg.index]
        sz = r["s"].map({"1/3": SIZES[0], "1": 1.0}).astype(float)
        pred = sz * r["en posición"].astype(float) * (ci + cb - HAIRCUT - b)
        rows[k] = {"alcance": "pre-registro" if in_scope(k) else "exploratorio",
                   "cobra el inverso (gasto + |L|·sobrecoste)": ci, "cobra el alcista": cb, "préstamo mediano": b,
                   "neto por unidad vendida": ci + cb - HAIRCUT - b, "Δg medio de las 162 (pp)": 100 * dg.mean(),
                   "comisiones − préstamo, media de las 162 (pp)": 100 * pred.mean(),
                   "resto: camino y costes de negociar (pp)": 100 * (dg.mean() - pred.mean()),
                   "correlación entre reglas": float(np.corrcoef(dg, pred)[0, 1])}
    return pd.DataFrame(rows).T


def top_rules(C: dict, k: str, comp: str = "alcista", n: int = 8) -> pd.DataFrame:
    R = C["R"][k]
    f = R["fam"][(comp, "mediana", True)].sort_values("Δg", ascending=False).head(n)
    r = R["rules"].loc[f.index]
    return pd.DataFrame({"Δg (pp)": 100 * f["Δg"], "IC90 (pp)": [r29.ci_text(a, b) for a, b in zip(f.IC90_bajo, f.IC90_alto)],
                         "p": f.p, "p_RW": f.p_RW, "operaciones": r["operaciones"], "en posición": r["en posición"],
                         "CAGR": r["CAGR"], "maxDD": r["maxDD"], "ganancia media por operación": r["ganancia media"],
                         "P(pérdida > 50%)": r["P(pérdida > 50%)"], "incumplimientos": r["incumplimientos"]})


def h4_rows(C: dict, k: str) -> pd.DataFrame:
    R = C["R"][k]
    rows = {}
    for (s, h), r in R["h4"].items():
        mech = R["h4_mech"].get((s, h), {})
        rows[(s_label(s), h)] = {"operaciones": r["operaciones"], "con VR ≥ 1": r["con VR ≥ 1"] / r["operaciones"],
                                 "Δ medio (pp)": 100 * r["Δ medio"], "β sobre 1 − VR (pp)": 100 * r["β (1 − VR)"],
                                 "t de β": r["t β"], "R²": r["R²"], "Δ con VR < 1 (pp)": 100 * r["Δ con VR < 1"],
                                 "Δ con VR ≥ 1 (pp)": 100 * r["Δ con VR ≥ 1"], "t (VR ≥ 1)": r["t (VR ≥ 1)"],
                                 "R² sobre Σx² − (Σx)²": r["R² sobre Σx² − (Σx)²"],
                                 "β, todas las sesiones sin costes (pp)": 100 * mech.get("β (1 − VR)", np.nan),
                                 "t, todas las sesiones": mech.get("t β", np.nan), "H4": h4_verdict(r)}
    out = pd.DataFrame(rows).T
    out.index.names = ["s", "h"]
    return out


def h4_real(C: dict) -> pd.DataFrame:
    rows = {}
    for k, R in C["R"].items():
        if is_long(k):
            continue
        rs = list(R["h4"].values())
        rows[k] = {"alcance": "pre-registro" if in_scope(k) else "exploratorio", "celdas (s, h)": len(rs), "H4 se sostiene": sum(h4_verdict(r) == "se sostiene" for r in rs),
                   "t de β mínimo": min(r["t β"] for r in rs), "t de β máximo": max(r["t β"] for r in rs),
                   "Δ con VR ≥ 1, el mayor (pp)": 100 * max(r["Δ con VR ≥ 1"] for r in rs),
                   "Δ con VR < 1, el menor (pp)": 100 * min(r["Δ con VR < 1"] for r in rs),
                   "operaciones (s = 1, h = 21)": R["h4"][(1.0, 21)]["operaciones"]}
    return pd.DataFrame(rows).T


def h5_rows(C: dict, k: str) -> pd.DataFrame:
    rows = {}
    for (s, h), r in C["R"][k]["h5"].items():
        rows[(s_label(s), h)] = {**{c: r[c] for c in ("operaciones", "P corto", "P alcista", "P corto al final",
                                                        "P alcista al final", "solo el corto", "solo el alcista", "p McNemar",
                                                        "peor corto", "peor alcista")}, "H5": h5_verdict(r)}
    out = pd.DataFrame(rows).T
    out.index.names = ["s", "h"]
    return out


def h5_real(C: dict) -> pd.DataFrame:
    rows = {}
    for k, R in C["R"].items():
        if is_long(k):
            continue
        v = {sh: h5_verdict(r) for sh, r in R["h5"].items()}
        rows[k] = {"alcance": "pre-registro" if in_scope(k) else "exploratorio",
                   "P corto (s = 1/3, h = 63)": R["h5"][(SIZES[0], 63)]["P corto"],
                   "P alcista (s = 1/3, h = 63)": R["h5"][(SIZES[0], 63)]["P alcista"],
                   "P corto (s = 1, h = 63)": R["h5"][(1.0, 63)]["P corto"],
                   "P alcista (s = 1, h = 63)": R["h5"][(1.0, 63)]["P alcista"],
                   "celdas: mayor en el corto": sum(x.startswith("mayor en el corto") for x in v.values()),
                   "celdas: mayor en el alcista": sum(x == "mayor en el alcista" for x in v.values()),
                   "celdas: igual": sum(x.startswith("igual") for x in v.values())}
    return pd.DataFrame(rows).T


def h5_counts(C: dict, scope: bool = True) -> dict:
    """Recuento de celdas (mercado, s, h) por veredicto de H5, por tamaño (mercados del pre-registro o
    exploratorios)."""
    out = {s: {"corto": 0, "alcista": 0, "igual": 0} for s in SIZES}
    for k, R in C["R"].items():
        if in_scope(k) != scope:
            continue
        for (s, h), r in R["h5"].items():
            v = h5_verdict(r)
            out[s]["corto" if v.startswith("mayor en el corto") else "alcista" if v == "mayor en el alcista" else "igual"] += 1
    return out


def episodes_long(C: dict) -> pd.DataFrame:
    parts = []
    for k, R in C["R"].items():
        if is_long(k):
            e = R["episodes"].copy()
            e.index = pd.MultiIndex.from_product([[k.split(" (")[0]], e.index], names=["mercado", "episodio"])
            parts.append(e)
    return pd.concat(parts)


def episodes_real(C: dict) -> pd.DataFrame:
    rows = {}
    for k, R in C["R"].items():
        if not is_long(k):
            e = R["episodes"]
            rows[k] = {"alcance": "pre-registro" if in_scope(k) else "exploratorio",
                       **{f"{ep}: mediana de las reglas": e.loc[ep, "corto: mediana"] if ep in e.index else np.nan for ep in EPISODES},
                       **{f"{ep}: índice": e.loc[ep, "índice"] if ep in e.index else np.nan for ep in ("2008", "2020", "2022", "agosto de 2024")}}
    t = pd.DataFrame(rows).T
    return t.dropna(axis=1, how="all")


def decisive(C: dict) -> pd.DataFrame:
    """La mejor regla frente a la caja y frente al alcista (préstamo mediano, producto remunerado) en la
    historia larga: Δg con toda la ventana y sin cada episodio."""
    rows = {}
    for k, R in C["R"].items():
        if not is_long(k):
            continue
        m, sim = C["M"][k], R["sim"]
        X = comparisons(m, sim, "mediana", True)
        for comp in ("caja", "alcista"):
            j = int(X[comp].mean(axis=0).argmax())
            rows[(k.split(" (")[0], COMP_ES[comp], sim["names"][j])] = {kk: 100 * v for kk, v in without_episodes(m, X[comp], j).items()}
    t = pd.DataFrame(rows).T
    t.index.names = ["mercado", "frente a", "regla"]
    return t


def best_in_episode(C: dict, k: str, ep: str, comp: str = "alcista") -> tuple[float, float]:
    """Rentabilidad en un episodio de la mejor regla de una comparación (préstamo mediano, producto
    remunerado): el corto y su alcista de las mismas sesiones."""
    R, m = C["R"][k], C["M"][k]
    sim = R["sim"]
    j = sim["names"].index(R["fam"][(comp, "mediana", True)]["Δg"].idxmax())
    a, b = EPISODES[ep]
    sel = (m["idx"] > pd.Timestamp(a)) & (m["idx"] <= pd.Timestamp(b))

    def ret(r):
        return float(np.expm1(np.log1p(np.maximum(np.nan_to_num(r[sel]), CLIP)).sum()))

    return ret(sim["short"][("mediana", True)][:, j]), ret(sim["long"][:, j])


def annex(C: dict, rep: Report) -> list[str]:
    """Una tabla CSV por mercado con las 162 reglas: operativa y, por referencia, préstamo y remuneración,
    Δg, IC90, p, Holm y Romano–Wolf."""
    files = []
    for k, R in C["R"].items():
        t = R["rules"].copy()
        for (c, lv, rb), f in R["fam"].items():
            tag = f"{c} · {lv} · {'remunerado' if rb else 'sin remunerar'}"
            t[f"Δg {tag}"] = f["Δg"]
            t[f"IC90_bajo {tag}"], t[f"IC90_alto {tag}"] = f["IC90_bajo"], f["IC90_alto"]
            t[f"p {tag}"], t[f"p_Holm {tag}"], t[f"p_RW {tag}"] = f["p"], f["p_Holm"], f["p_RW"]
        name = "familia_" + k.split(" (")[0].replace("&", "").replace(" ", "").replace("/", "_").replace("-", "")
        t.to_csv(rep.dir / f"{name}.csv")
        files.append(f"{name}.csv")
    if C.get("pool"):
        t = pd.concat({f"{c} · {'remunerado' if rb else 'sin remunerar'}": C["pool"][(c, rb)] for c in COMPS for rb in REBATES},
                      axis=1)
        t.to_csv(rep.dir / "familia_conjunta_pares.csv")
        files.append("familia_conjunta_pares.csv")
    return files


# ───────────────────────────── figuras ─────────────────────────────

def _style(ax):
    ax.grid(True, color="#e3e2de", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_family(C: dict, rep: Report):
    """Δg frente al alcista de las mismas sesiones de las 162 reglas de cada mercado."""
    ks = [k for k in C["R"] if in_scope(k)] + [k for k in C["R"] if not in_scope(k)]
    n_in = sum(in_scope(k) for k in ks)
    fig, ax = plt.subplots(figsize=(FIG_W, 8.6))
    rng = np.random.default_rng(0)
    for i, k in enumerate(ks):
        R = C["R"][k]
        f = R["fam"][("alcista", "mediana", True)]
        s = R["rules"]["s"].reindex(f.index)
        y = len(ks) - 1 - i
        for lab, col in (("1/3", BLUE), ("1", ORANGE)):
            v = 100 * f["Δg"][s == lab].astype(float)
            ax.scatter(v, y + rng.uniform(-0.28, 0.28, len(v)), s=10, color=col, alpha=0.7, lw=0)
        rw = f[f.p_RW < 0.05]
        if len(rw):
            ax.scatter(100 * rw["Δg"], [y] * len(rw), s=70, facecolor="none", edgecolor=INK, lw=1.2)
        p = f.attrs["p_SPA"]
        ax.text(1.01, y, f"p = {num_es(p, 3)}" if p >= 0.001 else "p < 0,001", transform=ax.get_yaxis_transform(),
                va="center", fontsize=10.5, color=INK if p < SPA_LEVEL else MUTED,
                fontweight="bold" if p < SPA_LEVEL else "normal")
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.axhline(len(ks) - n_in - 0.5, color=MUTED, lw=0.9, ls="--")
    ax.text(0.99, len(ks) - n_in - 0.62, "exploratorios, fuera de H3", transform=ax.get_yaxis_transform(), ha="right",
            va="top", fontsize=10.5, color=MUTED)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([k.replace(" sintético)", ")").replace(" (", "\n(") if is_long(k) else k for k in ks][::-1], fontsize=11)
    for lab, k in zip(ax.get_yticklabels(), ks[::-1]):
        if not in_scope(k):
            lab.set_color(MUTED)
    ax.set_xlabel("Δg frente al alcista de las mismas sesiones (pp/año)")
    ax.set_title("Las 162 reglas de cada mercado frente al alcista comprado", fontsize=13)
    _style(ax)
    ax.grid(False, axis="y")
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=6), plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=6),
         plt.Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=9)]
    legend_band(fig, h, ["s = 1/3", "s = 1", "Romano–Wolf < 0,05"], where="bottom", ncol=3, fontsize=11)
    comma_fig(fig)
    rep.fig(fig, "familia_frente_al_alcista",
            "Figura: cada punto es una regla (préstamo mediano de su ticker, producto remunerado): crecimiento logarítmico "
            "anual del corto del inverso tras el repunte menos el del alcista comprado en las mismas sesiones con la "
            "misma exposición (el +3x en los mercados del pre-registro, arriba; el alcista del mismo |L| o el índice en "
            "los exploratorios, abajo en gris, que no cuentan para H3). A la derecha, el p del SPA de la familia (en "
            "negrita si es < 0,10). Los círculos marcan las reglas que sobreviven a Romano–Wolf al 5%")


def fig_h4(C: dict, rep: Report, s: float = 1.0, h: int = 21):
    ks = [k for k in C["R"] if is_long(k)]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.8), sharey=True)
    for ax, k in zip(axes, ks):
        t = C["R"][k]["trades"][(s, h)].dropna(subset=["VR"])
        x, y = 1 - t["VR"], 100 * t["Δ"]
        keep = x >= -5
        ax.scatter(x[keep], y[keep], s=9, color=BLUE, alpha=0.35, lw=0)
        bins = np.array([-5, -3, -2, -1, -0.5, 0, 0.25, 0.5, 0.75, 1.0001])
        g = pd.cut(x, bins)
        mb = y.groupby(g, observed=True).mean()
        mx = x.groupby(g, observed=True).mean()
        ax.plot(mx, mb, color=ORANGE, lw=2.2, marker="o", ms=6)
        r = C["R"][k]["h4"][(s, h)]
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set(title=f"{k.split(' (')[0]}: β = {num_es(100 * r['β (1 − VR)'], 1)} pp (t = {num_es(r['t β'], 1)})",
               xlabel="1 − VR del periodo")
        ax.set_xlim(-5.2, 1.1)
        _style(ax)
    axes[0].set_ylabel("Sin reajuste − diario (%)")
    lo = min(np.nanpercentile(100 * C["R"][k]["trades"][(s, h)]["Δ"], 0.5) for k in ks)
    hi = max(np.nanpercentile(100 * C["R"][k]["trades"][(s, h)]["Δ"], 99.5) for k in ks)
    axes[0].set_ylim(lo, hi)
    hnd = [plt.Line2D([], [], marker="o", ls="", color=BLUE, alpha=0.5, ms=5), plt.Line2D([], [], color=ORANGE, lw=2.2, marker="o")]
    legend_band(fig, hnd, ["una operación", "media por tramo de 1 − VR"], where="bottom", ncol=2, fontsize=11)
    comma_fig(fig)
    rep.fig(fig, "ventaja_sin_reajuste",
            f"Figura: cada punto es una operación de las reglas de H3 con s = {s_label(s)} y h = {h} sesiones (fechas de "
            "entrada únicas; préstamo mediano y producto remunerado): lo que ganó el corto sin reajustar menos el mismo "
            "corto reajustado cada día, frente a 1 − VR de las sesiones de la tenencia. A la derecha del 0 el índice fue y "
            "volvió (VR < 1) y el corto sin reajuste ganó; a la izquierda tuvo tendencia y perdió. Se omiten las "
            "operaciones con 1 − VR < −5 (entran en la regresión)")


def fig_h5(C: dict, rep: Report):
    ks = [k for k in C["R"] if is_long(k)]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.6), sharey=True)
    cells = [(s, h) for s in SIZES for h in HOLDS]
    xs = np.arange(len(cells))
    for ax, k in zip(axes, ks):
        h5 = C["R"][k]["h5"]
        a = [100 * h5[c]["P corto"] for c in cells]
        b = [100 * h5[c]["P alcista"] for c in cells]
        ax.bar(xs - 0.2, a, 0.38, color=BLUE, label="corto del inverso")
        ax.bar(xs + 0.2, b, 0.38, color=ORANGE, label="+3x comprado")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{s_label(s)}\n{h}" for s, h in cells], fontsize=11)
        ax.set(title=k.split(" (")[0], xlabel="s y h (sesiones)")
        _style(ax)
        ax.grid(False, axis="x")
    axes[0].set_ylabel("Operaciones con pérdida > 50% (%)")
    hnd, lab = axes[0].get_legend_handles_labels()
    legend_band(fig, hnd, lab, where="bottom", ncol=2, fontsize=11)
    comma_fig(fig)
    rep.fig(fig, "cola_perdida_50",
            "Figura: porcentaje de operaciones (fechas de entrada únicas de las reglas de H3) en que el capital cayó más "
            "de un 50% en algún cierre de la tenencia: corto del inverso (préstamo mediano, producto remunerado, "
            "mantenimiento de FINRA) frente al +3x comprado en las mismas sesiones con la misma exposición inicial. "
            "Con s = 1/3 el +3x no puede perder más de un tercio")


def fig_episodes(C: dict, rep: Report):
    panels = [(k, ep) for k, ep in zip([k for k in C["R"] if is_long(k)], ("2008", "2020"))]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 8.2))
    for ax, (k, ep) in zip(axes, panels):
        m, R = C["M"][k], C["R"][k]
        sim = R["sim"]
        f = R["fam"][("alcista", "mediana", True)]
        j = sim["names"].index(f["Δg"].idxmax())
        a, b = EPISODES[ep]
        idx = m["idx"]
        sel = (idx > pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
        series = [("índice (1x)", m["u"].to_numpy(), INK, "-"), ("LRS 3x", m["lrs"].to_numpy(), GREY, "-"),
                  ("corto del inverso", sim["short"][("mediana", True)][:, j], BLUE, "-"),
                  ("+3x, mismas sesiones", sim["long"][:, j], ORANGE, "--")]
        for lab, r, col, ls in series:
            w = np.exp(np.cumsum(np.log1p(np.maximum(np.nan_to_num(r[sel]), CLIP))))
            ax.plot(idx[sel], w, color=col, ls=ls, lw=2.0 if col in (BLUE, ORANGE) else 1.5, label=lab)
        held = sim["held"][sel, j]
        ymin, ymax = ax.get_ylim()
        ax.fill_between(idx[sel], ymin, ymax, where=held, color=BLUE, alpha=0.08, lw=0, step="post")
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"{k.split(' (')[0]}, {ep}: {sim['names'][j]}", fontsize=12)
        ax.set_ylabel("Valor de 1 $")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)) if ep == "2008" else mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
        _style(ax)
    hnd, lab = axes[0].get_legend_handles_labels()
    hnd.append(plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.15))
    lab.append("con el corto abierto")
    legend_band(fig, hnd, lab, where="bottom", ncol=3, fontsize=11)
    comma_fig(fig)
    rep.fig(fig, "episodios",
            "Figura: la mejor regla frente al +3x (préstamo mediano, producto remunerado) de cada historia larga en "
            "un episodio: valor de 1 $ del corto del inverso, del +3x comprado en las mismas sesiones, de la LRS 3x y "
            "del índice. La franja azul marca las sesiones con el corto abierto")


# ───────────────────────────── informe ─────────────────────────────

def _y(items) -> str:
    items = list(items)
    return r29.y_join(items) if items else "ninguno"


def y_list(items) -> str:
    return r29.y_join(list(items))


def explo_table(C: dict) -> pd.DataFrame:
    """Mercados exploratorios (fuera del alcance de H3): p del SPA con el préstamo mediano en cada comparación y
    remuneración, y la mejor regla frente al alcista (remunerado)."""
    rows = {}
    for k, R in C["R"].items():
        if in_scope(k):
            continue
        m = C["M"][k]
        row = {"alcista de (b)": bull_label(m["bull"]), "L": m["L"]}
        for c in COMPS:
            fr, fn = R["fam"][(c, "mediana", True)], R["fam"][(c, "mediana", False)]
            row[f"{COMP_ES[c]}: p_SPA remunerado"] = fr.attrs["p_SPA"]
            row[f"{COMP_ES[c]}: p_SPA sin remunerar"] = fn.attrs["p_SPA"]
        fb = R["fam"][("alcista", "mediana", True)]
        row["(b): Δg de la mejor (pp)"] = 100 * float(fb["Δg"].max())
        rows[k] = row
    return pd.DataFrame(rows).T


def run(C: dict | None = None):
    warnings.filterwarnings("ignore")
    rep = Report("r32_short_rallies", "R32 — Vender el inverso en los repuntes (H3, H4 y H5 del artículo 6)")
    for f in list(rep.dir.glob("tabla_*.csv")) + list(rep.dir.glob("familia_*.csv")):
        f.unlink()
    C = C or compute()
    t0 = time.time()
    tab = Tables(rep)
    TN = {}
    Rk = C["R"]
    ks_in = [k for k in Rk if in_scope(k)]
    ks_ex = [k for k in Rk if not in_scope(k)]
    longs = [k for k in Rk if is_long(k)]
    reals_in = [k for k in ks_in if not is_long(k)]
    H = {c: h3_rows(C, c) for c in COMPS}
    hp = {(c, rb): holm_pairs(C, c, rb) for c in COMPS for rb in REBATES}
    bt, act = borrow_rows(C)
    grid = C["grid"]
    xt = explo_table(C)

    def ver(c, k):
        return H[c].loc[k, "H3"]

    def rej(c, pred=lambda v: v == "se rechaza"):
        return [k for k in ks_in if pred(ver(c, k))]

    def pspa(c, k, rb=True):
        return H[c].loc[k, "p_SPA (remunerado)" if rb else "p_SPA (sin remunerar)"]

    def short(k):
        return k.split(" (")[0] if is_long(k) else k

    def x_rej(c):
        a = xt[f"{COMP_ES[c]}: p_SPA remunerado"] < SPA_LEVEL
        b = xt[f"{COMP_ES[c]}: p_SPA sin remunerar"] < SPA_LEVEL
        return [k for k in xt.index if a[k] or b[k]]

    rej_a = rej("caja")
    part = {c: rej(c, lambda v: v.startswith("se rechaza solo")) for c in COMPS}
    rej_b = rej("alcista")
    rej_c = rej("LRS", lambda v: v != "se sostiene")
    h4_all = [(k, sh, h4_verdict(r)) for k in ks_in for sh, r in Rk[k]["h4"].items()]
    h4_ok = sum(v == "se sostiene" for *_, v in h4_all)
    h4_ex = [(k, sh, h4_verdict(r)) for k in ks_ex for sh, r in Rk[k]["h4"].items()]
    h4_ex_ok = sum(v == "se sostiene" for *_, v in h4_ex)
    tmin = min(r["t β"] for k in ks_in for r in Rk[k]["h4"].values())
    tmin_txt = num_es(np.floor(tmin * 100) / 100, 2)
    vr_hi_max = max(r["Δ con VR ≥ 1"] for k in ks_in for r in Rk[k]["h4"].values())
    h5c, h5x = h5_counts(C), h5_counts(C, scope=False)
    sp, nd = longs
    h5sp, h5nd = Rk[sp]["h5"], Rk[nd]["h5"]
    pool = C.get("pool")
    Hp = {c: H[c].loc[POOL] for c in COMPS} if pool else {}
    n_pool = len(pool["pares"]) if pool else 0
    mg = marginal(C)
    ep = episodes_long(C)
    dv = decisive(C)
    ft = fees_table(C)
    ft_in = ft[ft["alcance"] == "pre-registro"]
    r2z5 = [Rk[k]["h4"][(SIZES[0], 5)]["R² sobre Σx² − (Σx)²"] for k in ks_in]
    r2z1 = [r["R² sobre Σx² − (Σx)²"] for k in ks_in for (s_, h_), r in Rk[k]["h4"].items() if s_ == 1.0]

    def swing(mkt, comp):
        """El episodio cuya ausencia más cambia el Δg de la mejor regla, y el Δg sin él."""
        row = dv.xs((mkt, COMP_ES[comp]), level=(0, 1)).iloc[0]
        rest = row.drop("toda la ventana")
        e = (rest - row["toda la ventana"]).abs().idxmax()
        return e.replace("sin ", ""), float(row["toda la ventana"]), float(rest[e])

    bep_sp = best_in_episode(C, sp, swing("S&P 500", "alcista")[0])
    bep_nd = best_in_episode(C, nd, swing("Nasdaq-100", "alcista")[0])
    n_in = len(ks_in)

    rep.p(__doc__.split("\n", 2)[2])
    rep.h("Resumen")
    rep.p("\n".join([
        f"- **Alcance.** H3, H4 y H5 hablan de «vender en corto el −3x»: se juzgan en los {n_in} mercados −3x del "
        f"pre-registro, las dos historias largas y los {len(reals_in)} pares reales ({_y(reals_in)}), con el +3x como "
        f"alcista. SDS y QID (−2x), SH y PSQ (−1x) y TECS (fuera de la lista) se dan aparte como exploratorios y no "
        f"cuentan.",
        f"- **H3 (a), frente a la caja: se rechaza en {len(rej_a)} de los {n_in} mercados** con las dos remuneraciones "
        f"del producto del corto ({_y(short(k) for k in rej_a)})"
        + (f" y, solo con una de ellas, en {_y(short(k) for k in part['caja'])}" if part["caja"] else "")
        + (f"; también en la familia conjunta de los {n_pool} pares −3x vivos desde 2010 (p del SPA "
           f"{peq(Hp['caja']['p_SPA (remunerado)'])})" if pool else "")
        + f". No es el decaimiento: el corto del inverso es exposición alcista al índice (s·|L| al entrar) y cobra la "
        f"prima de la bolsa; la mejor regla del Nasdaq-100 desde 1985 le saca {pp(H['caja'].loc[nd, 'Δg de la mejor (pp)'] / 100)} "
        f"pp/año a la letra y la del S&P 500 desde 1928, {pp(H['caja'].loc[sp, 'Δg de la mejor (pp)'] / 100)} (p del SPA "
        f"{peq(pspa('caja', sp))} remunerado y {peq(pspa('caja', sp, False))} sin remunerar).",
        f"- **H3 (b), frente al +3x comprado en las mismas sesiones con la misma exposición: se sostiene en la historia "
        f"larga** (S&P 500 desde 1928: p del SPA {peq(pspa('alcista', sp))}, mejor regla {pp(H['alcista'].loc[sp, 'Δg de la mejor (pp)'] / 100, 2)} "
        f"pp/año; Nasdaq-100 desde 1985: p {peq(pspa('alcista', nd))}, {pp(H['alcista'].loc[nd, 'Δg de la mejor (pp)'] / 100)} pp)"
        + (f" y en la familia conjunta de los {n_pool} pares −3x (p {peq(Hp['alcista']['p_SPA (remunerado)'])} remunerado, "
           f"{peq(Hp['alcista']['p_SPA (sin remunerar)'])} sin remunerar)" if pool else "")
        + f". Par a par se rechaza en {_y(rej_b) if rej_b else 'ninguno'} con las dos remuneraciones"
        + (f" y solo con el producto remunerado en {_y(part['alcista'])}" if part["alcista"] else "")
        + f"; con Holm entre los {len(reals_in)} pares −3x (no pre-registrado) el menor p ajustado es "
        f"{num_es(hp[('alcista', True)].min(), 3)}. Lo que queda es pequeño y depende del préstamo: con el p90 de cada "
        f"ticker la mejor regla del S&P 500 gana {pp(bt.loc[sp, 'Δg mejor, p90 (pp)'] / 100, 2)} pp/año y la del "
        f"Nasdaq-100, {pp(bt.loc[nd, 'Δg mejor, p90 (pp)'] / 100)}; en la historia larga sobreviven a Romano–Wolf "
        f"{int(sum(bt.loc[k, f'RW 5%, {lv}'] for k in longs for lv in LEVELS))} reglas con alguno de los tres préstamos.",
        f"- **H3 (c), frente a la LRS 3x: se sostiene en {'todos los mercados' if not rej_c else f'todos salvo {_y(rej_c)}'}**"
        + (f" y en la familia conjunta (p {peq(Hp['LRS']['p_SPA (remunerado)'])})" if pool else "")
        + f": la mejor regla queda a {pp(H['LRS'].loc[sp, 'Δg de la mejor (pp)'] / 100)} pp/año de la LRS en el S&P 500 "
        f"y a {pp(H['LRS'].loc[nd, 'Δg de la mejor (pp)'] / 100)} en el Nasdaq-100.",
        f"- **Exploratorios** ({_y(ks_ex)}; en (b), el alcista del mismo |L| o el índice del laboratorio): el SPA da p < "
        f"0,10 con alguna remuneración frente a la caja en {_y(x_rej('caja'))}, frente al alcista en "
        f"{_y(x_rej('alcista'))} y frente a la LRS en {_y(x_rej('LRS'))}. No entran en ningún recuento.",
        f"- **H4 se sostiene en {h4_ok} de las {len(h4_all)} celdas** (mercado × s × h): la ventaja del corto sin reajuste "
        f"sobre el mismo corto reajustado cada día crece con 1 − VR (t de Hansen-Hodrick de al menos {tmin_txt}) y con "
        f"VR ≥ 1 es negativa en todas (la mayor, {pp(vr_hi_max, 2)} pp por operación). Con la escala del periodo, "
        f"Σx² − (Σx)², la aproximación de segundo orden explica del {pct_es(min(r2z5), 0)} al {pct_es(max(r2z5), 0)} de "
        f"la varianza con s = 1/3 y h = 5, y con s = 1 una mediana del {pct_es(float(np.median(r2z1)), 0)}: con el corto "
        "entero el mantenimiento lo reduce a mitad de camino y la aproximación deja de valer. Es lo mismo que R22 y R29: "
        f"sin reajuste, el corto es una apuesta a que el índice vuelva sobre sus pasos. En los exploratorios, "
        f"{h4_ex_ok} de {len(h4_ex)}.",
        f"- **H5 se cumple a medias.** Con s = 1/3 la probabilidad de perder más del 50% dentro de la tenencia es mayor en el "
        f"corto en {h5c[SIZES[0]]['corto']} de las {sum(h5c[SIZES[0]].values())} celdas (el +3x comprado por un tercio no "
        f"puede perder más de un tercio; el corto no tiene tope: S&P 500 con h = 63, {pct_es(h5sp[(SIZES[0], 63)]['P corto'], 2)} "
        f"de las operaciones, peor {pct_es(h5sp[(SIZES[0], 63)]['peor corto'], 0)}), e igual en el resto. Con s = 1 es "
        f"mayor en el +3x en {h5c[1.0]['alcista']} de las {sum(h5c[1.0].values())} celdas y en el corto en "
        f"{h5c[1.0]['corto']}: el mantenimiento de FINRA obliga a reducir el corto en cuanto el inverso sube un 5,3%, y "
        f"eso funciona como un stop (S&P 500, h = 63: {pct_es(h5sp[(1.0, 63)]['P corto'], 1)} frente a "
        f"{pct_es(h5sp[(1.0, 63)]['P alcista'], 1)}; Nasdaq-100: {pct_es(h5nd[(1.0, 63)]['P corto'], 1)} frente a "
        f"{pct_es(h5nd[(1.0, 63)]['P alcista'], 1)}).",
        f"- **Episodios.** El corto del inverso cae con el mercado igual que el +3x de las mismas sesiones: en "
        f"2000–2002 la mediana de las 162 reglas del Nasdaq-100 perdió un {pct_es(-ep.loc[('Nasdaq-100', '2000–2002'), 'corto: mediana'], 0)} "
        f"(el +3x, {pct_es(-ep.loc[('Nasdaq-100', '2000–2002'), 'alcista: mediana'], 0)}) y la peor, un "
        f"{pct_es(-ep.loc[('Nasdaq-100', '2000–2002'), 'corto: peor'], 0)}; en 1987 la peor regla del S&P 500 perdió un "
        f"{pct_es(-ep.loc[('S&P 500', '1987'), 'corto: peor'], 0)}. En el rebote en V de 2020 la mediana ganó un "
        f"{pct_es(ep.loc[('S&P 500', '2020'), 'corto: mediana'], 0)} en el S&P 500. La ventaja frente al +3x de la mejor "
        f"regla del Nasdaq-100 ({pp(swing('Nasdaq-100', 'alcista')[1] / 100)} pp/año) depende de {swing('Nasdaq-100', 'alcista')[0]}: "
        f"sin esas sesiones es {pp(swing('Nasdaq-100', 'alcista')[2] / 100)}; la del S&P 500 "
        f"({pp(swing('S&P 500', 'alcista')[1] / 100, 2)}), de {swing('S&P 500', 'alcista')[0]} "
        f"({pp(swing('S&P 500', 'alcista')[2] / 100, 2)} sin él). En esos episodios el corto pierde menos porque el "
        f"mantenimiento lo va reduciendo: en el S&P 500, {pct_es(bep_sp[0], 0)} frente a {pct_es(bep_sp[1], 0)} del +3x; en "
        f"el Nasdaq-100 los dos lo pierden casi todo ({pct_es(bep_nd[0], 1)} y {pct_es(bep_nd[1], 2)}) y esa diferencia, "
        f"grande en logaritmos, es toda la ventaja.",
    ]))

    rep.h("1. Datos, fondos y préstamo")
    rep.p("Cada mercado se simula con su calendario desde la primera decisión posible. La historia larga usa los fondos de "
          "R22 (sintéticos antes del lanzamiento, reales después); los pares reales, los ETFs de Yahoo con la limpieza de "
          "common.pair (SOXS: un día de desdoblamiento mal ajustado sustituido). El préstamo es la distribución diaria de "
          "IBKR en iBorrowDesk, un año de datos; se aplica constante a toda la historia.")
    mk = pd.DataFrame({k: {"alcance": "pre-registro (H3–H5)" if in_scope(k) else "exploratorio",
                           "tipo": C["M"][k]["kind"], "inverso": C["M"][k]["inv"], "L": C["M"][k]["L"],
                           "alcista de (b)": bull_label(C["M"][k]["bull"]), "ventana": win_label(C["M"][k]),
                           "sesiones": len(C["M"][k]["idx"]) - 1, "mantenimiento": C["M"][k]["maint"],
                           "préstamo de": C["M"][k]["borrow_tk"]} for k in ks_in + ks_ex}).T
    tab("Mercados: alcance, fondo inverso, alcista de la comparación (b), ventana de evaluación (primer rendimiento → "
        "último) y mantenimiento de FINRA (30% × |L| del valor del corto). «3x» en la historia larga es el +3x del "
        "laboratorio (sintético y UPRO o TQQQ reales); con SH y PSQ el alcista es el índice del laboratorio, sin gasto "
        "de fondo", mk, pct={"mantenimiento"}, digits=2)
    g = grid.copy()
    g = g.rename(columns={"mín": "mínimo", "máx": "máximo", "disponibles_mín": "acciones disponibles, mínimo"})
    tab("Préstamo de títulos en IBKR por ticker (iBorrowDesk, tasa anual diaria): la rejilla del pre-registro es mínimo / "
        "mediana / p90", g, pct={"mínimo", "mediana", "p90", "máximo"}, digits=3)
    TN["grid"] = tab.n

    rep.h("2. H3: la familia de reglas")
    rep.p("Cada mercado es una familia de 162 reglas; en cada una se contrastan las 162 frente a la misma referencia con "
          "los mismos remuestreos. H3 dice que ninguna supera tras la búsqueda: se rechaza si el SPA da p < 0,10 con el "
          "préstamo mediano. Como el pre-registro deja abiertas las dos remuneraciones del producto del corto, el veredicto "
          "se da con las dos («se rechaza solo remunerado» si solo una da p < 0,10). Con tipos cerca de cero la letra − 0,5% "
          "es negativa y «sin remunerar» es el caso favorable.")
    for c in COMPS:
        tab(f"H3 {H3_ES[c]} en los mercados del pre-registro: p del SPA de la familia de 162 reglas con el préstamo "
            f"mediano de cada ticker y las dos remuneraciones; la mejor regla con el producto remunerado, su Δg (pp/año) "
            f"e IC90, cuántas tienen Δg > 0 y cuántas sobreviven a Romano–Wolf al 5%. La última fila es la familia "
            f"conjunta de los {n_pool} pares −3x vivos desde 2010 ({num_es(162 * n_pool, 0)} reglas)", H[c],
            pct={"préstamo mediano"}, digits=3)
        TN[f"h3_{c}"] = tab.n
    tab("Exploratorios, fuera del alcance de H3 (no cuentan en ningún veredicto): p del SPA de cada comparación con el "
        "préstamo mediano y las dos remuneraciones, y el Δg de la mejor regla frente a su alcista (el del mismo |L|, o el "
        "índice del laboratorio sin gasto para SH y PSQ)", xt, digits=3)
    TN["explo"] = tab.n
    fig_family(C, rep)
    rep.p("En la comparación (b) los dos lados tienen la misma exposición al entrar y ninguno se reajusta. A segundo "
          "orden, una operación de h sesiones del corto de s en el −L× menos s en el +L× rinde")
    rep.p("Δ ≈ s·(c_inv + c_alc − haircut − b)·h/252 + s·L²·(Σx² − (Σx)²).")
    rep.p("*Pie de fórmula: el corto se queda con lo que los dos fondos rinden de menos que su réplica, menos el préstamo, y "
          "además cobra el decaimiento de los dos fondos (½(L² + L) del inverso más ½(L² − L) del alcista: L²·Σx²), pero "
          "paga L²·(Σx)² si el índice tiene tendencia; el decaimiento extra del inverso solo se cobra si el índice vuelve "
          "sobre sus pasos.* Notación: s, la parte del capital vendida o comprada; c, lo que cada fondo rinde de menos que "
          "su réplica al año (su gasto más |L| veces el sobrecoste del swap calibrado en R22); haircut, el 0,5% que el "
          "bróker descuenta a la letra sobre el producto del corto; b, el préstamo anual; L, el apalancamiento en valor "
          "absoluto; x = u − f, la rentabilidad diaria del índice menos su tipo de financiación; Σ, la suma sobre las h "
          "sesiones. Con L = 3, L² = 9 = 6 + 3.")
    tab("Pares reales: lo que cobra la comparación (b) a primer orden. Lo que cada fondo rinde de menos que su réplica "
        "(gasto + |L| × sobrecoste del swap, calibrado con su vida real como en R22), préstamo mediano y el neto por unidad "
        "vendida (menos el haircut de 0,5%); Δg medio de las 162 reglas frente al alcista y lo que predicen las comisiones "
        "con el tiempo que cada regla pasa en posición", ft,
        pct={"cobra el inverso (gasto + |L|·sobrecoste)", "cobra el alcista", "préstamo mediano", "neto por unidad vendida"},
        digits=3)
    TN["fees"] = tab.n
    hp_b = hp[("alcista", True)]
    neg_net = [k for k in reals_in if ft.loc[k, "neto por unidad vendida"] < 0]
    rej_bp = rej_b + part["alcista"]
    net_in = ft_in["neto por unidad vendida"].astype(float).sort_values(ascending=False)
    nets = y_list(f"{pct_es(net_in[k], 1)} en {k}" for k in rej_bp) if rej_bp else ""
    top_txt = (f"son los {len(rej_bp)} pares −3x con el neto por unidad vendida más alto ({nets}, al año): cobran las "
               "comisiones de los dos fondos y pagan poco préstamo" if set(net_in.index[:len(rej_bp)]) == set(rej_bp)
               else f"tienen un neto por unidad vendida de {nets} al año")
    rep.p(f"**Lectura.** Frente a la caja, las reglas ganan porque están compradas: vender el inverso es tener "
          f"exposición alcista y 2006–2026 fue alcista. Frente al +3x de las mismas sesiones —la comparación que aísla "
          f"el decaimiento— la historia larga no rechaza H3 y la ventaja de la mejor regla es de "
          f"{pp(H['alcista'].loc[sp, 'Δg de la mejor (pp)'] / 100, 2)} pp/año en el S&P 500 y "
          f"{pp(H['alcista'].loc[nd, 'Δg de la mejor (pp)'] / 100)} en el Nasdaq-100. En los {len(reals_in)} pares −3x "
          f"reales, la cuenta de comisiones y préstamo de la fórmula (tabla «T:fees») tiene el signo del Δg medio de las "
          f"162 reglas en {int(((ft_in['Δg medio de las 162 (pp)'] > 0) == (ft_in['comisiones − préstamo, media de las 162 (pp)'] > 0)).sum())} "
          f"de ellos, y lo que queda —el camino y los costes de negociar— es positivo en "
          f"{int((ft_in['resto: camino y costes de negociar (pp)'] > 0).sum())} (de "
          f"{pp(ft_in['resto: camino y costes de negociar (pp)'].min() / 100, 2)} a "
          f"{pp(ft_in['resto: camino y costes de negociar (pp)'].max() / 100, 1)} pp/año), coherente con que en 2006–2026 "
          f"los índices revirtieran dentro del mes (VR < 1, R29). "
          + (f"Los rechazos por par ({_y(rej_bp)}) {top_txt}"
             + (f"; con el préstamo mediano de {_y(neg_net)} el neto es negativo" if neg_net else "") + ". " if rej_bp else "")
          + f"Con {len(reals_in)} familias de pares al 10% se esperan {num_es(0.1 * len(reals_in), 1)} rechazos por azar si "
          f"ninguna regla tuviera ventaja: con Holm entre los {len(reals_in)} el menor p es {num_es(hp_b.min(), 3)} "
          f"({hp_b.idxmin()})"
          + (f", y la familia conjunta de {n_pool} pares da p {peq(Hp['alcista']['p_SPA (remunerado)'])}" if pool else "")
          + ". Frente a la LRS 3x ninguna familia rechaza: el corto está fuera del mercado casi todo el tiempo y la LRS no.")

    rep.h("3. H3: el préstamo")
    tab("H3 (b) según el préstamo (rejilla de cada ticker): p del SPA con el mínimo (las dos remuneraciones) y con la "
        "mediana, Δg de la mejor regla (pp/año, remunerado) y supervivientes de Romano–Wolf con cada préstamo. Los "
        "exploratorios, al final", bt.loc[ks_in + ks_ex], pct={"préstamo mín", "mediana", "p90"}, digits=3)
    TN["borrow"] = tab.n
    tab("H3 (a) y (c) según el préstamo (producto remunerado): p del SPA con el mínimo y con la mediana; con el p90, Δg de "
        "la mejor regla (pp/año) y supervivientes de Romano–Wolf. Los exploratorios, al final", act.loc[ks_in + ks_ex],
        digits=3)
    nb_min = [k for k in ks_in if bt.loc[k, "p_SPA con el mín"] < SPA_LEVEL]
    rep.p(f"**Lectura.** Con el préstamo mínimo de cada ticker, el SPA frente al +3x da p < 0,10 en {len(nb_min)} de los "
          f"{n_in} mercados del pre-registro ({_y(short(k) for k in nb_min)}); con la mediana, en {len(rej_bp)}. "
          "El mínimo es el día más barato de un año: nadie puede asegurarse ese coste durante décadas. Con el p90 la ventaja "
          "de la mejor regla se reduce en todos los mercados, y el préstamo sube justo cuando el corto pierde (R29).")

    rep.h("4. H3: qué reglas y qué dimensiones")
    tab(f"Δg medio frente al +3x de las mismas sesiones (pp/año, préstamo mediano, producto remunerado) de las reglas "
        f"con cada valor de cada dimensión; en los pares reales, la media de los {len(reals_in)} −3x del pre-registro",
        mg, digits=2)
    for k in longs:
        tab(f"{short(k)}: las 8 mejores reglas frente al +3x de las mismas sesiones (préstamo mediano, producto "
            f"remunerado); p unilateral, Romano–Wolf en la familia de 162; operativa y riesgo del corto",
            top_rules(C, k), pct={"en posición", "CAGR", "maxDD", "ganancia media por operación", "P(pérdida > 50%)"}, digits=3)
    rw_long = int(sum(H["alcista"].loc[k, "Romano–Wolf 5%"] for k in longs))
    rep.p(f"**Lectura.** En el S&P 500 el Δg medio frente al +3x es "
          + ("negativo con cualquier valor de cualquier dimensión" if (mg[sp] < 0).all() else "de signo variable")
          + f" (de {pp(mg[sp].min() / 100, 2)} a {pp(mg[sp].max() / 100, 2)} pp/año); en el Nasdaq-100 "
          + ("y en los pares −3x reales es positivo con cualquier valor" if (mg[nd] > 0).all() and (mg[REAL_MEAN] > 0).all()
             else "y en los pares −3x reales tiene signo variable")
          + f", mayor con s = 1 y sin condición de estado. Las mejores reglas pasan mucho tiempo vendidas (s = 1, h = 21 o 63): "
          f"la mejor del S&P 500 tuvo una caída máxima del {pct_es(-float(Rk[sp]['rules'].loc[H['alcista'].loc[sp, 'mejor regla'], 'maxDD']), 0)} "
          f"y la del Nasdaq-100, del {pct_es(-float(Rk[nd]['rules'].loc[H['alcista'].loc[nd, 'mejor regla'], 'maxDD']), 0)}. "
          + ("Ninguna sobrevive" if rw_long == 0 else f"{rw_long} sobreviven")
          + " a Romano–Wolf frente al +3x en la historia larga con el préstamo mediano. La tabla completa de cada mercado está en los CSV del anexo.")

    rep.h("5. H4: la ventaja sin reajuste y el cociente de varianzas")
    rep.p("Para cada operación (fechas de entrada únicas de las 27 reglas con los mismos s y h) se compara el corto sin "
          "reajuste con el mismo corto reajustado cada día, con las mismas entradas y salidas, el préstamo mediano y el "
          "producto remunerado. Hasta segundo orden y sin costes, la diferencia es")
    rep.p("Δ ≈ (s + s²)·(L²/2)·Σx²·(1 − VR),   con VR = (Σx)² / Σx²   y x = u − f en las h sesiones de la tenencia.")
    rep.p("*Pie de fórmula: el corto sin reajustar gana los cuadrados de los movimientos diarios y pierde el cuadrado del "
          "movimiento acumulado; gana si el índice va y vuelve (VR < 1) y pierde si tiene tendencia (VR > 1).* Notación: s, "
          "la parte del capital vendida; L = −3, el apalancamiento del fondo; u, la rentabilidad diaria del índice; f, su "
          "tipo de financiación; Σ, la suma sobre las sesiones de la tenencia. El pre-registro pide la regresión sobre "
          "(1 − VR); la escala Σx² cambia de un periodo a otro, así que se da también el R² sobre Σx² − (Σx)².")
    for k in longs:
        tab(f"{short(k)}: H4 por s y h. Operaciones únicas, parte con VR ≥ 1, Δ medio por operación, β de Δ sobre 1 − VR "
            f"(pp por unidad) con t de Hansen-Hodrick (h − 1 retardos), Δ medio con VR < 1 y con VR ≥ 1 (mismos errores) y, "
            f"como comprobación mecánica, β con todas las sesiones como entrada y sin costes", h4_rows(C, k),
            pct={"con VR ≥ 1"}, digits=3)
    tab("Pares reales: celdas (s, h) en que H4 se sostiene, t de β mínimo y máximo, y el mayor Δ medio con VR ≥ 1 (los "
        "exploratorios, al final)", h4_real(C).loc[reals_in + ks_ex], digits=2)
    fig_h4(C, rep)
    rep.p(f"**Lectura.** H4 se sostiene en {h4_ok} de {len(h4_all)} celdas. La ventaja existe cuando el índice revierte "
          "dentro de la tenencia y se vuelve pérdida cuando tiene tendencia; su media depende de cuántos periodos revierten, "
          "no de que el inverso «decaiga». Con s = 1 y h = 63 el Δ medio es negativo en la historia larga "
          f"({pp(Rk[sp]['h4'][(1.0, 63)]['Δ medio'], 1)} pp por operación en el S&P 500): solo el "
          f"{pct_es(Rk[sp]['h4'][(1.0, 63)]['con VR ≥ 1'] / Rk[sp]['h4'][(1.0, 63)]['operaciones'], 0)} de las operaciones "
          f"tiene VR ≥ 1, pero pierde {pp(-Rk[sp]['h4'][(1.0, 63)]['Δ con VR ≥ 1'], 1).lstrip('+')} pp de media frente a "
          f"{pp(Rk[sp]['h4'][(1.0, 63)]['Δ con VR < 1'], 1)} de las que revierten.")

    rep.h("6. H5: la cola")
    for k in longs:
        tab(f"{short(k)}: H5 por s y h. Parte de las operaciones con pérdida de más del 50% del capital en algún cierre de "
            f"la tenencia y al final, corto frente al +3x de las mismas sesiones; operaciones discordantes y McNemar "
            f"unilateral (corto > +3x); la peor caída de cada uno", h5_rows(C, k),
            pct={"P corto", "P alcista", "P corto al final", "P alcista al final", "peor corto", "peor alcista"}, digits=3)
    tab("Pares reales: P(pérdida > 50%) con h = 63 y recuento de celdas por veredicto (los exploratorios, al final)",
        h5_real(C).loc[reals_in + ks_ex],
        pct={"P corto (s = 1/3, h = 63)", "P alcista (s = 1/3, h = 63)", "P corto (s = 1, h = 63)", "P alcista (s = 1, h = 63)"},
        digits=3)
    fig_h5(C, rep)
    rep.p("**Lectura.** H5 compara dos colas con topes distintos. Con s = 1/3, el +3x comprado arriesga como mucho un "
          "tercio del capital y el corto no tiene tope: la cola es mayor en el corto cuando la hay. Con s = 1, el +3x "
          "comprado con todo el capital puede perderlo casi todo en una caída, y el corto tiene un freno que el "
          "pre-registro impone: el mantenimiento del 90% obliga a reducirlo a s del capital en cuanto el inverso sube un "
          "5,3% desde el último ajuste; en una caída larga el corto se comporta como el diario y en un rebote posterior "
          "tiene menos exposición. En la práctica ese freno es una recompra forzosa al peor precio del día, y el bróker "
          f"puede exigir más que FINRA. En los exploratorios, con s = 1/3 la cola es mayor en el corto en "
          f"{h5x[SIZES[0]]['corto']} de {sum(h5x[SIZES[0]].values())} celdas y con s = 1 en el alcista en "
          f"{h5x[1.0]['alcista']} de {sum(h5x[1.0].values())}.")

    rep.h("7. Episodios")
    tab("Rentabilidad en cada episodio (historia larga): índice, LRS 3x y distribución de las 162 reglas (corto con el "
        "préstamo mediano y el producto remunerado) y de sus +3x; «reglas con posición»: las que abrieron algún corto "
        "en el episodio. Fechas aproximadas, del máximo previo a la recuperación", ep,
        pct={"índice", "LRS 3x", "corto: mediana", "corto: peor", "corto: mejor", "reglas que pierden", "alcista: mediana",
             "alcista: peor"}, digits=3)
    TN["ep"] = tab.n
    tab("Qué episodios deciden: Δg (pp/año) de la mejor regla frente a la caja y frente al +3x en toda la ventana y "
        "quitando las sesiones de cada episodio", dv, digits=2)
    er = episodes_real(C).loc[reals_in + ks_ex]
    tab("Pares reales: mediana de la rentabilidad de las 162 reglas en cada episodio (vacío si el par no existía) y el "
        "índice del par (los exploratorios, al final)", er, pct=set(er.columns) - {"alcance"}, digits=3)
    fig_episodes(C, rep)
    rep.p("**Lectura.** El corto del inverso tras un repunte pierde en las caídas lo que pierde el +3x de las mismas "
          "sesiones: en 2000–2002 y 2008 los repuntes fueron de mercado bajista y el corto abierto después pagó la "
          "siguiente bajada. Lo que separa al corto del +3x en un episodio es el camino dentro de cada tenencia (H4) y "
          "el freno del mantenimiento (H5). "
          + " ".join(f"En el {mk}, el episodio que más mueve el Δg de la mejor regla frente "
                     f"{'a la caja' if c == 'caja' else 'al +3x'} ({pp(swing(mk, c)[1] / 100)} pp/año) es "
                     f"{swing(mk, c)[0]}: sin él, {pp(swing(mk, c)[2] / 100)}." for mk in ("S&P 500", "Nasdaq-100")
                     for c in ("caja", "alcista")))

    rep.h("Desviaciones y lecturas del pre-registro")
    rep.p("\n".join([
        "El pre-registro no se puede editar en este trabajo: estas notas completan su sección «Desviaciones», que dice "
        "«(ninguna por ahora)», y están también en `letf/preregistro/articulo6_inversos_desviaciones.md`.",
        "- **Corregido tras la revisión del 30-09-2026 (después de ver los resultados).** La primera versión de este "
        "informe aplicaba H3 también a SDS y QID (−2x), SH y PSQ (−1x) y TECS (fuera de la lista), con el alcista del "
        "mismo |L| en (b), y los mezclaba en los recuentos, en Holm entre 12 pares y en la familia conjunta de 11. Ahora "
        f"H3, H4 y H5 se juzgan en los {n_in} mercados −3x del pre-registro, con Holm entre los {len(reals_in)} pares y la "
        f"familia conjunta de {n_pool}; los otros cinco quedan como exploratorios. Ningún veredicto de H3 cambia.",
        "- **Lecturas que el pre-registro no fija.** Una operación no mira señales mientras está abierta y entre dos hay al "
        "menos una sesión en caja; la LRS de comparación es la 3x del índice de cada par; H3 se decide con las dos "
        "remuneraciones del producto; H4 usa las operaciones únicas de las reglas con los mismos s y h y errores de "
        "Hansen-Hodrick sobre ellas ordenadas por fecha; H5 mide la pérdida a cierre, dentro de la tenencia, con "
        "McNemar. Holm entre pares y la familia conjunta no están en el pre-registro: se añaden porque el SPA corrige la "
        "búsqueda dentro de cada familia, no entre mercados.",
        "- **Exploratorios.** SDS, QID, SH, PSQ y TECS se comparan en (b) con el alcista del mismo |L| (SSO, QLD, TECL) o "
        "con el índice del laboratorio sin gasto (SH, PSQ), no con el +3x que pide el pre-registro. Por eso no cuentan.",
    ]))

    rep.h("Salvedades y datos sin verificar")
    rep.p("\n".join([
        f"- **Préstamo con solo 12 meses de datos** (iBorrowDesk, del {dmy(grid['desde'].min())} al {dmy(grid['hasta'].max())}) "
        "y de un bróker; se aplica constante a 1928–2026. El coste en 1987, 2000–2002, 2008 o 2020 es desconocido y sube "
        "cuando el inverso sube (R29).",
        "- **Recompras forzosas.** Ni el recall de los títulos prestados ni la falta de disponibilidad se simulan: hubo días "
        f"con {int(grid.loc['SQQQ', 'disponibles_mín'])} acción de SQQQ, {int(grid.loc['SDOW', 'disponibles_mín'])} de SDOW y "
        f"{int(grid.loc['SH', 'disponibles_mín'])} de SH disponibles. Tampoco la regla 201 de la SEC: si el inverso cae un 10% "
        "en una sesión, ese día y el siguiente solo se puede vender en corto por encima del mejor precio de compra, y un "
        "−3x cae un 10% con una subida del índice de algo más del 3%, justo el día de la señal.",
        "- **Mantenimiento.** El 30% × |L| de FINRA 4210 es un mínimo; el bróker puede exigir más y cambiarlo sin aviso. La "
        "reducción del corto al incumplirlo se ejecuta al cierre y sin coste adicional al de 5 pb.",
        "- **Acceso minorista en la UE y marco regulatorio.** PRIIPs impide comprar ETFs de EE. UU. sin documento KID; si "
        "impide también abrir un corto depende del bróker y no está verificado. Las condiciones de remuneración del producto "
        "del corto en IBKR (tramos, saldo mínimo) tampoco.",
        "- **Fondos sintéticos** antes de 2009–2010 en la historia larga (R22, s calibrado con su vida real) y el Nasdaq-100 "
        "reconstruido antes de 1985-10 solo para calentar la media.",
        f"- **La búsqueda.** El SPA y Romano–Wolf corrigen la búsqueda dentro de cada familia de 162 reglas, no entre "
        f"{n_in} mercados × 3 referencias × 2 remuneraciones; por eso se da Holm entre pares y la familia conjunta.",
        "- **H1 y H2** son de R31. Esta línea no depende de sus datos.",
    ]))

    rep.h("Anexo: las familias completas")
    files = annex(C, rep)
    rep.p("Una tabla CSV por mercado con las 162 reglas (operativa con el préstamo mediano y el producto remunerado; Δg, "
          "IC90, p, Holm y Romano–Wolf frente a cada referencia, con cada préstamo y remuneración) y la familia conjunta: "
          + ", ".join(f"`{f}`" for f in files) + ".")
    rep.p(f"*Cálculo: {num_es(C['segundos'] / 60, 1)} minutos ({len(Rk)} mercados repartidos entre procesos, B = {B}); "
          f"tablas y figuras: {num_es(time.time() - t0, 0)} s.*")
    for k, n in TN.items():
        rep.lines = [ln.replace(f"«T:{k}»", str(n)) for ln in rep.lines]
    rep.save()
    return C


if __name__ == "__main__":
    run()
