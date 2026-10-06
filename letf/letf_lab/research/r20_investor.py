"""R20 — Qué puede esperar un inversor (I7).

Cinco estrategias sobre el S&P 500, con las reglas de R13 tal cual: comprar y
mantener el 1x, comprar y mantener el 3x, la LRS 3x original (Gayed y Bilello
2016: 3x sobre la media de 200 sesiones del índice de RENTABILIDAD TOTAL, letras
debajo), la LRS-Kelly y el ½ Kelly GJR sin tendencia. Se mide lo que viviría
quien las siguiera, no sólo su rentabilidad final:

1. P(ir por delante del 1x) a 1, 3, 5, 10 y 20 años, en la historia real (todas
   las fechas de inicio, con el número de ventanas independientes declarado) y
   en los tres mundos sintéticos de R13 (bootstrap, GJR-GARCH, regímenes), con
   la MISMA regla en los dos lados: señal sobre el índice de rentabilidad total.
   La LRS con la señal sobre un índice de precio (una sensibilidad del
   laboratorio) se evalúa también en los dos lados, para separar el efecto de la
   señal del de la muestra.
2. Años por detrás del 1x: caída máxima de la riqueza relativa y racha más larga
   por debajo de su máximo previo, y de dónde sale la pérdida relativa de las
   estrategias Kelly desde 1974.
3. Una regla de abandono fijada de antemano: el inversor pasa al 1x para
   siempre si en los últimos 3 años va más de un 20% por detrás del 1x (con la
   rejilla completa de 1/3/5 años × 10/20/30% y una regla de pánico por caída
   propia del 50%).
4. Crisis definidas por regla: caídas del S&P 500 de precio de al menos el 19%
   desde su máximo (la convención habitual de mercado bajista, que se mide con el
   índice de precio aunque la LRS decida con el de rentabilidad total); la crisis
   termina en el mínimo previo a recuperar la mitad de lo perdido. Se separan en
   largas y cortas con una rejilla de duración.
5. Peores y mejores días con la LRS dentro y fuera, marcando los que dependen
   de un cierre de ^GSPC que R14 considera una errata.
6. El único periodo fuera de muestra: desde 2017, el primer año completo tras
   la publicación de la LRS (SSRN, marzo de 2016), con ETFs reales.
7. Ficha de cada estrategia: rentabilidad a 30 años como mediana y P5–P95 de
   los mundos sintéticos, caída máxima típica, tiempo invertido, cambios de
   exposición al año y reglas exactas congeladas con el hash del commit.

Protocolo: nada de lo anterior entra en una señal; todas las reglas de
inversión son las de R13 (señal al cierre de t, orden al cierre de t+1). Los
horizontes, la regla de abandono, su rejilla, el umbral de crisis (19%, dado)
y la regla de fin de crisis se fijaron antes de ver resultados. Tras una
revisión se añadieron, y se declaran como tales: la misma señal en la historia y
en los mundos, las variantes con la señal de precio (sensibilidad), la lectura de
la riqueza relativa en los máximos del 1x, el reparto de la pérdida de las estrategias
Kelly, la rejilla de duración de las crisis (3–18 meses) y la sensibilidad a
los cierres dudosos.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter

from letf_lab import kelly, stats
from letf_lab.data import history
from letf_lab.report import FIG_W, Report, legend_band, num_es
from letf_lab.research import r12_long_ml as r12
from letf_lab.research import r13_lrs as r13

SEED = 20
HORIZONS = (1, 3, 5, 10, 20)             # años
N_WORLD = 4000                           # historias por mundo (R13 usa 1000)
BATCH = 500                              # historias por lote (memoria)
CRISIS_DROP = 0.19                       # caída mínima desde el máximo (dada)
OOS_FROM = "2017-01-03"                  # primer año completo tras la publicación (SSRN, marzo de 2016)
STRIDE = 21                              # inicios mensuales para la regla de abandono
B_BOOT = 2000
LAG = r13.EXEC_LAG

FOCUS = {  # nombre corto → clave en r13.strategies
    "1x": "1x comprar y mantener",
    "3x": "3x comprar y mantener",
    "LRS 3x": "LRS 3x (original)",
    "LRS-Kelly": "LRS-Kelly",
    "½ Kelly GJR": "½ Kelly GJR (sin tendencia)",
}
MC_KEY = {  # nombre corto → clave en r13.mc_strategies (en los mundos, Kelly con EWMA)
    "1x": "1x comprar y mantener",
    "3x": "3x comprar y mantener",
    "LRS 3x": "LRS 3x (original)",
    "LRS-Kelly": "LRS-Kelly",
    "½ Kelly GJR": "½ Kelly (sin tendencia)",
}
OTHERS = ["3x", "LRS 3x", "LRS-Kelly", "½ Kelly GJR"]
# Sensibilidad: las variantes con la señal sobre el índice de PRECIO (r13.price_variants en la
# historia; en los mundos, un índice de precio construido). Se evalúan en la historia y en los
# mundos para separar el efecto de la señal del de la muestra.
PX3, PXK = r13.PX_LRS, r13.PX_KELLY
PX_KEYS = [PX3, PXK]
REL_KEYS = OTHERS + PX_KEYS
WORLDS = (("bootstrap", "bootstrap"), ("gjr", "GJR"), ("regímenes", "regímenes"))
WORLD_STYLE = {"bootstrap": ("#c8600a", "s"), "GJR": ("#1b8a6b", "^"), "regímenes": ("#6b4fbf", "D")}
COLORS = {"1x": "k", "3x": "#6e6e6e", "LRS 3x": "#1f77b4", "LRS-Kelly": "#d62728",
          "½ Kelly GJR": "#9467bd", PX3: "#17becf", PXK: "#ff9896"}
# Cierres de ^GSPC con la firma de un error de transcripción, revisados a mano en R14
# (r14_data_audit.MANUAL_PAIRS) y que la base NO corrige. Afectan a la sesión del cierre y a la siguiente.
DISPLAY = {"½ Kelly GJR": "Medio Kelly (sin tendencia)"}   # nombres del artículo en las figuras
SUSPECT_CLOSES = ["1931-01-07", "1935-05-16", "1935-12-16", "1938-10-10", "1946-09-03",
                  "1948-02-04", "1949-01-24"]
CRISIS_MONTHS = (3, 6, 9, 12, 18)         # rejilla de duración máximo→mínimo (a posteriori, entera)

# Regla de abandono: (etiqueta, tipo, años de memoria, umbral). La principal va primero.
MAIN_RULE = "3 años, −20% frente al 1x"
ABANDON_RULES = [(f"{k} año{'s' if k > 1 else ''}, −{x:.0%} frente al 1x", "rel", k, x)
                 for k in (1, 3, 5) for x in (0.10, 0.20, 0.30)]
ABANDON_RULES += [("caída propia ≥ 50%", "dd", None, 0.50)]
assert MAIN_RULE in [r[0] for r in ABANDON_RULES]

RULES = {
    "1x": "Comprar y mantener el S&P 500 con dividendos reinvertidos (rentabilidad total). "
          "Sin operaciones.",
    "3x": "Comprar y mantener un ETF 3x diario del S&P 500 (UPRO real desde el 25-6-2009; antes, "
          "sintético rf + 3·(u − f − s) − 0,91% con f = call money/fed funds y s el sobrecoste "
          "del swap calibrado en la vida del ETF). Sin reequilibrio de la cartera.",
    "LRS 3x": "Al cierre de cada sesión t: si el índice de rentabilidad total del S&P 500 (^GSPC más "
              "el D/P de Shiller del mes anterior hasta 1988; el índice oficial ^SP500TR después) está por "
              "encima de su media simple de 200 sesiones, 100% en el 3x; si no, 100% en letras del Tesoro. La orden "
              "se ejecuta al cierre de t+1. 5 pb por unidad de rotación. Sensibilidad: la misma regla con "
              "el S&P 500 de precio (^GSPC, cierre).",
    "LRS-Kelly": "Fracción del 3x w = min(1, max(0, ½·(μ̂ − f̂ − s)/σ̂² / 3)) si el índice de rentabilidad "
                 "total del S&P 500 está "
                 "sobre su media de 200 sesiones y 0 si no; μ̂ − f̂ = media expansiva de u − f "
                 "(rentabilidad diaria menos financiación) hasta la sesión anterior ×252 (mínimo "
                 "10 años), σ̂² = varianza GJR-GARCH prevista a 21 sesiones "
                 "(ventana expansiva, reestimada cada 1008 sesiones). Sólo se reajusta si w cambia "
                 "más de 10 pp o toca 0 o 1; el resto en letras. Orden al cierre de t+1, 5 pb.",
    "½ Kelly GJR": "Como la LRS-Kelly pero sin la condición de la media de 200 sesiones: "
                   "w = min(1, max(0, ½·(μ̂ − f̂ − s)/σ̂² / 3)), mismos μ̂ − f̂, σ̂², umbral de 10 pp, "
                   "orden al cierre de t+1 y 5 pb.",
}


# ───────────────────────────── utilidades puras ─────────────────────────────

def log_growth(r) -> np.ndarray:
    """log(1 + r); una pérdida del 100% da log(1e-300): la cartera liquidada no revive."""
    return np.log(np.maximum(1.0 + np.asarray(r, float), 1e-300))


def cum_log(r) -> np.ndarray:
    """Riqueza logarítmica acumulada con un 0 inicial (el tiempo es el último eje)."""
    lg = log_growth(r)
    return np.concatenate([np.zeros(lg.shape[:-1] + (1,)), np.cumsum(lg, axis=-1)], axis=-1)


def window_diff(Cs: np.ndarray, Cb: np.ndarray, n: int) -> np.ndarray:
    """Diferencia de crecimiento log estrategia − referencia en todas las ventanas de n pasos."""
    return (Cs[..., n:] - Cs[..., :-n]) - (Cb[..., n:] - Cb[..., :-n])


def wilson(p: float, n: int, z: float = 1.645) -> tuple[float, float]:
    """Intervalo de Wilson para una proporción con n observaciones independientes."""
    if n <= 0:
        return np.nan, np.nan
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return float(c - h), float(c + h)


def underwater(D: np.ndarray) -> dict:
    """Caída máxima y racha más larga bajo el máximo previo de una riqueza log D.

    D: (filas, T+1) o (T+1,), con el tiempo en el último eje. Devuelve arrays por
    fila: maxdd (fracción, ≤ 0), peak/trough (índices de la caída máxima),
    longest (pasos seguidos por debajo del máximo previo), long_start (índice del
    máximo que abre esa racha), long_end (último paso por debajo) y recovered
    (False si la racha sigue abierta al final: está censurada)."""
    D = np.atleast_2d(np.asarray(D, float))
    T = D.shape[1]
    run = np.maximum.accumulate(D, axis=1)
    dd = D - run
    at_peak = dd >= -1e-12
    idx = np.broadcast_to(np.arange(T), D.shape)
    last = np.maximum.accumulate(np.where(at_peak, idx, 0), axis=1)
    length = idx - last
    rows = np.arange(len(D))
    trough = dd.argmin(axis=1)
    long_end = length.argmax(axis=1)
    return {"maxdd": np.expm1(dd.min(axis=1)), "trough": trough, "peak": last[rows, trough],
            "longest": length.max(axis=1), "long_start": last[rows, long_end], "long_end": long_end,
            "recovered": long_end < T - 1, "frac_under": (~at_peak).mean(axis=1)}


def date_crises(px: pd.Series, drop: float = CRISIS_DROP) -> pd.DataFrame:
    """Crisis por regla sobre un índice de precios.

    En fase alcista se sigue el máximo; si el cierre cae al menos `drop` por
    debajo, empieza una crisis. En la crisis se sigue el mínimo; la crisis acaba
    (y su valle es ese mínimo) cuando el índice recupera la mitad de lo perdido
    entre el máximo y el mínimo; entonces empieza otra fase alcista desde ese
    cierre. La recuperación es el primer cierre posterior al valle que iguala el
    máximo (NaT si aún no ha llegado)."""
    v = px.dropna()
    x, idx = v.values, v.index
    eps = []
    bull, peak, trough = True, 0, 0
    for i in range(1, len(x)):
        if bull:
            if x[i] > x[peak]:
                peak = i
            elif x[i] <= x[peak] * (1 - drop):
                bull, trough = False, i
        else:
            if x[i] < x[trough]:
                trough = i
            elif x[i] >= 0.5 * (x[peak] + x[trough]):
                eps.append((peak, trough, False))
                bull, peak = True, i
    if not bull:
        eps.append((peak, trough, True))
    rows = []
    for p, t, open_ in eps:
        rec = np.flatnonzero(x[t + 1:] >= x[p])
        rows.append({"pico": idx[p], "valle": idx[t],
                     "recuperación": idx[t + 1 + rec[0]] if len(rec) else pd.NaT,
                     "caída": x[t] / x[p] - 1, "en_curso": open_})
    return pd.DataFrame(rows)


def period_return(r: pd.Series, a, b) -> float:
    """Rentabilidad acumulada de r entre los cierres de a y b (sesiones a+1 … b)."""
    if pd.isna(a) or pd.isna(b):
        return np.nan
    s = r[(r.index > a) & (r.index <= b)]
    return np.nan if s.isna().any() or len(s) == 0 else float(np.prod(1 + s.values) - 1)


def years_to_regain(r: pd.Series, a, b) -> float:
    """Años desde el cierre de a (máximo del S&P) hasta que la riqueza vuelve a su nivel
    de ese día, buscando desde el cierre de b (mínimo del S&P); 0 si en b ya lo superaba."""
    s = r[r.index > a]
    if s.isna().any() or len(s) == 0:
        return np.nan
    w = np.cumprod(1 + s.values)
    k = int(np.searchsorted(s.index.values, np.datetime64(b), side="right")) - 1
    if k < 0:
        return np.nan
    if w[k] >= 1 - 1e-12:
        return 0.0
    hit = np.flatnonzero(w[k:] >= 1 - 1e-12)
    return (k + hit[0] + 1) / 252 if len(hit) else np.nan


def start_windows(C: np.ndarray, starts: np.ndarray, n: int) -> np.ndarray:
    """Matriz (len(starts), n+1) con la riqueza log de cada ventana, 0 al inicio."""
    W = sliding_window_view(C, n + 1)[starts]
    return W - W[:, :1]


def abandon(Cs: np.ndarray, C1: np.ndarray, kind: str, lookback: int | None, x: float,
            horizons: list[int]) -> tuple[np.ndarray, dict]:
    """Regla de abandono aplicada a ventanas que empiezan en 0 (filas = inversores).

    Cs, C1: riqueza log de la estrategia y del 1x, (n, T+1). En el paso j (tras j
    sesiones) el inversor mira sólo el pasado:
    - kind 'rel': si su riqueza relativa frente al 1x en las últimas `lookback`
      sesiones (o desde el inicio, si lleva menos) ha caído más de x, abandona;
    - kind 'dd': si su propia caída desde máximos supera x, abandona.
    Al abandonar da la orden al cierre siguiente (como EXEC_LAG = 2): la sesión
    j+1 es aún de la estrategia y desde la j+2 tiene el 1x. Devuelve el paso de
    decisión (−1 si nunca) y, para cada horizonte H (sesiones), la riqueza log final."""
    n, T1 = Cs.shape
    if kind == "rel":
        D = Cs - C1
        k = int(lookback)
        Dsh = np.concatenate([np.zeros((n, min(k, T1))), D[:, :max(T1 - k, 0)]], axis=1)
        cond = (D - Dsh) < np.log(1 - x)
    elif kind == "dd":
        cond = (Cs - np.maximum.accumulate(Cs, axis=1)) < np.log(1 - x)
    else:
        raise ValueError(kind)
    cond[:, 0] = False
    j = np.where(cond.any(axis=1), cond.argmax(axis=1), -1)
    rows = np.arange(n)
    out = {}
    for H in horizons:
        hit = (j >= 0) & (j < H)
        jj = np.clip(j + 1, 0, H)
        out[H] = np.where(hit, Cs[rows, jj] + C1[:, H] - C1[rows, jj], Cs[:, H])
    return j, out


def at_highs(D: np.ndarray, C1: np.ndarray) -> np.ndarray:
    """Riqueza relativa D leída sólo los días en que el 1x (riqueza log C1) marca un máximo
    histórico y mantenida hasta el siguiente. Evita que un mínimo del mercado, con la estrategia
    fuera, fabrique un máximo relativo."""
    C1 = np.asarray(C1, float)
    hi = C1 >= np.maximum.accumulate(C1) - 1e-12
    last = np.maximum.accumulate(np.where(hi, np.arange(len(C1)), 0))
    return np.asarray(D, float)[..., last]


def attribution(rel: pd.Series, cats: dict, bounds: list) -> pd.DataFrame:
    """Crecimiento log relativo de cada tramo (a, b] y su reparto entre categorías de sesión.

    rel: log(1 + r_estrategia) − log(1 + r_1x) por sesión; cats: {nombre: Serie booleana}
    que debe ser una partición de las sesiones (se comprueba)."""
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        s = (rel.index > a) & (rel.index <= b)
        masks = {k: v.reindex(rel.index, fill_value=False).values.astype(bool) & s for k, v in cats.items()}
        if not np.array_equal(np.sum(list(masks.values()), axis=0), s.astype(int)):
            raise ValueError("las categorías no son una partición del tramo")
        row = {"desde": a.date(), "hasta": b.date(), "factor_relativo": float(np.exp(rel[s].sum())),
               "log_relativo": float(rel[s].sum())}
        for k, mk in masks.items():
            row[f"log: {k}"] = float(rel[mk].sum())
        for k, mk in masks.items():
            row[f"sesiones: {k}"] = int(mk.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def fix_closes(px: pd.Series, dates) -> pd.Series:
    """Sustituye cada cierre de `dates` por la media geométrica de sus vecinos (el mismo
    arreglo que history.gspc_close aplica a GSPC_BAD_CLOSES)."""
    out = px.copy()
    for dte in pd.to_datetime(dates):
        i = out.index.get_loc(dte)
        out.iloc[i] = float(np.sqrt(out.iloc[i - 1] * out.iloc[i + 1]))
    return out


def suspect_sessions(index: pd.DatetimeIndex, dates=SUSPECT_CLOSES) -> pd.DatetimeIndex:
    """Sesiones cuya rentabilidad depende de un cierre dudoso: la del cierre y la siguiente."""
    pos = index.get_indexer(pd.to_datetime(dates))
    pos = pos[pos >= 0]
    return index[np.unique(np.r_[pos, pos + 1].clip(max=len(index) - 1))]


def crisis_by_duration(head: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, keep: list,
                       thresholds=CRISIS_MONTHS, k: str = "LRS 3x") -> pd.DataFrame:
    """Recuento de crisis largas (≥ X meses del máximo al mínimo) y cortas para cada umbral X:
    en cuántas la estrategia k perdió menos que el 1x del máximo al mínimo y en cuántas acabó por
    delante del máximo a la recuperación (sólo las ya recuperadas)."""
    rows = []
    for X in thresholds:
        for lab, sel in ((f"≥ {X} meses", lambda m: m >= X), (f"< {X} meses", lambda m: m < X)):
            names = [c for c in keep if sel(head.loc[c, "meses_pico_valle"])]
            rec = [c for c in names if pd.notna(b.loc[c, "1x"]) and pd.notna(b.loc[c, k])]
            ahead = [c for c in rec if b.loc[c, k] > b.loc[c, "1x"]]
            rows.append({"umbral": X, "clase": lab, "crisis": len(names),
                         "pierde menos (máx→mín)": int(sum(a.loc[c, k] > a.loc[c, "1x"] for c in names)),
                         "recuperadas": len(rec), "por delante (máx→recuperación)": len(ahead),
                         "por detrás": ", ".join(c for c in rec if c not in ahead) or "—"})
    return pd.DataFrame(rows).set_index(["umbral", "clase"])


# ───────────────────────────── mundos: índice de precio y estrategias ─────────────────────────────

def world_paths(method: str, u: np.ndarray, pr: np.ndarray, n: int, T: int, rng,
                dv: float) -> tuple[np.ndarray, np.ndarray]:
    """n historias de T días de rentabilidad total U y de rentabilidad de PRECIO PR.

    bootstrap: los mismos índices que r13.gen_paths (misma semilla ⇒ mismo U) aplicados a la
    vez a u y a pr, así que cada bloque lleva el dividendo de su época. gjr y regímenes:
    U de r13.gen_paths y PR = (1 + U)/(1 + dv) − 1, con dv la diferencia media diaria entre
    rentabilidad total y de precio en la muestra."""
    ok = ~np.isnan(u)
    u, pr = u[ok], pr[ok]
    if np.isnan(pr).any():
        raise ValueError("pr con huecos")
    if method == "bootstrap":
        idx = np.stack([stats.stationary_indices(len(u), 252, rng, T) for _ in range(n)])
        return u[idx], pr[idx]
    U = r13.gen_paths(method, u, n, T, rng)
    return U, (1 + U) / (1 + dv) - 1


def world_strategies(U: np.ndarray, PR: np.ndarray, rf_d: float, f_d: float, spread: float, er: float,
                     lag: int = LAG) -> tuple[dict, np.ndarray]:
    """Las estrategias de r13.mc_strategies (señal de la LRS sobre el índice de rentabilidad
    total, cumprod(1 + U), como la original en la historia) y, como sensibilidad, las variantes
    con la señal sobre el índice de PRECIO (cumprod(1 + PR)). El resto (3x, EWMA λ = 0,94, μ̂
    desde el año 10, ½ Kelly, umbral de 10 pp, costes) es copia literal de r13.mc_strategies;
    un test y una comprobación en ejecución verifican que se reproduce R13 exactamente (y, con
    PR = U, también las de precio). Devuelve las series y la señal de la original."""
    x3 = np.maximum(rf_d + 3 * (U - f_d) - (3 * spread + er) / 252, -1.0)
    sig_px = r13._sma_signal(np.cumprod(1 + PR, axis=1), 200)
    sig_tr = r13._sma_signal(np.cumprod(1 + U, axis=1), 200)
    lam = 0.94
    ewma = lfilter([1 - lam], [1.0, -lam], U ** 2, axis=1,
                   zi=np.full((len(U), 1), lam * np.var(U[:, :252])))[0]
    t_idx = np.arange(1, U.shape[1] + 1)
    mu = np.c_[np.full(len(U), np.nan), (np.cumsum(U, axis=1) / t_idx)[:, :-1]] * 252
    mu[:, : r13.BURN * 252] = np.nan
    kel = np.clip(r13.KELLY * (mu - f_d * 252 - spread) / (ewma * 252) / 3, 0, 1)

    def sticky_mat(target):
        out = np.full_like(target, np.nan)
        cur = np.full(len(target), np.nan)
        for t in range(target.shape[1]):
            v = target[:, t]
            move = np.isnan(cur) | (np.abs(v - cur) > r13.MIN_STEP) | (v == 0) | (v == 1)
            cur = np.where(np.isnan(v), np.nan, np.where(move, v, cur))
            out[:, t] = cur
        return out

    def run(w3):
        w3 = np.c_[np.zeros((len(w3), lag)), np.nan_to_num(w3)[:, :-lag]]
        turn = np.abs(np.diff(np.c_[np.zeros(len(w3)), w3], axis=1))
        return w3 * x3 + (1 - w3) * rf_d - turn * r13.COST_BPS / 1e4

    S = {"1x": U, "3x": x3, "LRS 3x": run(sig_tr),
         "LRS-Kelly": run(sticky_mat(kel * np.nan_to_num(sig_tr))),
         "½ Kelly GJR": run(sticky_mat(kel)),
         PX3: run(sig_px),
         PXK: run(sticky_mat(kel * np.nan_to_num(sig_px)))}
    return S, sig_tr


R13_TWIN = {"1x": "1x comprar y mantener", "3x": "3x comprar y mantener",
            "½ Kelly GJR": "½ Kelly (sin tendencia)", "LRS 3x": "LRS 3x (original)",
            "LRS-Kelly": "LRS-Kelly", PX3: "LRS 3x (original)", PXK: "LRS-Kelly"}


def check_against_r13(U, PR, rf_d, f_d, spread, er) -> float:
    """Máxima diferencia entre world_strategies y r13.mc_strategies en las claves comunes (las de
    precio solo coinciden con PR = U, que es como se llama en la comprobación)."""
    S, _ = world_strategies(U, PR, rf_d, f_d, spread, er)
    S13 = r13.mc_strategies(U, rf_d, f_d, spread, er, lag=LAG)
    return max(float(np.nanmax(np.abs(S[k] - S13[v]))) for k, v in R13_TWIN.items())


# ───────────────────────────── historia ─────────────────────────────

def history_frame(d: dict, R: dict, start) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Rentabilidades de las cinco estrategias en su muestra común, T-bill y fechas de C."""
    df = pd.DataFrame({s: R[s] for s in R}).loc[start:]
    df = df.dropna(how="all")
    full = d["u"].index
    pos = full.get_loc(df.index[0])
    dates = full[pos - 1: pos + len(df)]
    return df, d["rf"].reindex(df.index), dates


def ahead_history(df: pd.DataFrame, rf: pd.Series, keys: list[str]) -> pd.DataFrame:
    C = {k: cum_log(df[k].values) for k in keys}
    Crf = cum_log(rf.values)
    N = len(df)
    rows = []
    for k in keys:
        if df[k].isna().any():
            raise ValueError(f"{k} tiene huecos en la muestra común")
        bench = Crf if k == "1x" else C["1x"]
        for h in HORIZONS:
            n = 252 * h
            rel = window_diff(C[k], bench, n)
            p = float((rel > 0).mean())
            ni = N // n
            lo, hi = wilson(p, ni)
            rows.append({"estrategia": k if k != "1x" else "1x (frente a letras)", "años": h,
                         "P_historia": p, "ventanas": len(rel), "independientes": ni,
                         "IC90_bajo": lo, "IC90_alto": hi,
                         "Δ_mediana": float(np.median(rel)) / h,
                         "Δ_P10": float(np.percentile(rel, 10)) / h})
    return pd.DataFrame(rows)


def behind_history(df: pd.DataFrame, dates: pd.DatetimeIndex, keys: list[str]) -> pd.DataFrame:
    C1 = cum_log(df["1x"].values)
    rows = {}
    for k in keys:
        D = cum_log(df[k].values) - C1
        u = underwater(D)
        g = {f: v[0] for f, v in u.items()}
        dd = D - np.maximum.accumulate(D)
        rec = np.flatnonzero(D[g["trough"]:] >= D[g["peak"]] - 1e-12)
        rows[k] = {"caída_relativa_máx": g["maxdd"], "máximo_relativo": dates[g["peak"]].date(),
                   "mínimo_relativo": dates[g["trough"]].date(),
                   "recupera": dates[g["trough"] + rec[0]].date() if len(rec) else "no",
                   "racha_más_larga_años": g["longest"] / 252,
                   "racha_desde": dates[g["long_start"]].date(),
                   "racha_hasta": dates[g["long_end"] + 1].date() if g["recovered"] else "sigue",
                   "tiempo_>20%_bajo_máx": float((dd < np.log(0.8)).mean()),
                   "riqueza_relativa_final": float(np.exp(D[-1]))}
    return pd.DataFrame(rows).T.infer_objects()


def abandon_history(df: pd.DataFrame, years: int) -> pd.DataFrame:
    n = 252 * years
    C = {k: cum_log(df[k].values) for k in FOCUS}
    starts = np.arange(0, len(df) - n + 1, STRIDE)
    W1 = start_windows(C["1x"], starts, n)
    out = []
    for label, kind, lb, x in ABANDON_RULES:
        for k in OTHERS:
            Ws = start_windows(C[k], starts, n)
            j, fin = abandon(Ws, W1, kind, None if lb is None else 252 * lb, x, [n])
            out.append(abandon_row(label, k, j, Ws[:, n], fin[n], W1[:, n], n, years))
    return pd.DataFrame(out)


def abandon_row(label, k, j, held, fin, one, n, years) -> dict:
    """Resumen de una regla de abandono. El coste de quien no abandona es exactamente 0, así
    que la mediana sobre todos cae en esa masa cuando abandona poco más de la mitad: el coste
    se da condicionado a abandonar (mediana) y como media sobre todos (= P × coste medio
    condicionado)."""
    hit = (j >= 0) & (j < n)
    c = lambda v: np.expm1(v / years)  # noqa: E731
    cost = c(held) - c(fin)
    return {"regla": label, "estrategia": k, "inversores": len(j), "P_abandona": hit.mean(),
            "años_hasta_abandono": np.median(j[hit]) / 252 if hit.any() else np.nan,
            "CAGR_mantener": np.median(c(held)), "CAGR_con_abandono": np.median(c(fin)),
            "CAGR_1x": np.median(c(one)),
            "coste_si_abandona": np.median(cost[hit]) if hit.any() else np.nan,
            "coste_medio_todos": np.mean(cost),
            "P_detrás_mantener": np.mean(held < one), "P_detrás_con_abandono": np.mean(fin < one)}


# ───────────────────────────── mundos sintéticos ─────────────────────────────

def world_batch(U: np.ndarray, PR: np.ndarray, rf_d: float, f_d: float, spread: float, er: float) -> dict:
    """Reduce un lote de historias a lo que necesita el informe (evalúa los 30 años tras el calentamiento)."""
    S, sig = world_strategies(U, PR, rf_d, f_d, spread, er)
    b0 = r13.BURN * 252
    T = U.shape[1] - b0
    keys = list(FOCUS) + PX_KEYS
    C = {k: cum_log(S[k][:, b0:]) for k in keys}
    del S
    Crf = np.arange(T + 1) * np.log1p(rf_d)
    out = {"Cm": {k: C[k][:, ::STRIDE] for k in keys}, "Crf_m": Crf[::STRIDE]}
    out["final"] = {k: C[k][:, -1] for k in FOCUS}
    out["maxdd"] = {k: np.expm1((C[k] - np.maximum.accumulate(C[k], axis=1)).min(axis=1)) for k in FOCUS}
    out["rel"] = {}
    for k in REL_KEYS:
        u = underwater(C[k] - C["1x"])
        out["rel"][k] = {"maxdd": u["maxdd"], "longest": u["longest"], "recovered": u["recovered"]}
    hz = [252 * h for h in (10, 20, r13.YEARS)]
    out["aband"] = {}
    for label, kind, lb, x in ABANDON_RULES:
        for k in OTHERS:
            j, fin = abandon(C[k], C["1x"], kind, None if lb is None else 252 * lb, x, hz)
            out["aband"][(label, k)] = {"j": j, "fin": fin, "held": {H: C[k][:, H] for H in hz},
                                        "one": {H: C["1x"][:, H] for H in hz}}
    sig = sig[:, b0:]
    out["lrs_invested"] = np.nanmean(sig, axis=1)
    out["lrs_switches"] = (np.abs(np.diff(sig, axis=1)) > 0).sum(axis=1) / r13.YEARS
    return out


def _cat(parts: list, path: tuple):
    vals = []
    for p in parts:
        v = p
        for key in path:
            v = v[key]
        vals.append(v)
    return np.concatenate(vals) if np.ndim(vals[0]) else np.array(vals)


def price_returns(d: dict) -> pd.Series:
    """Rentabilidad diaria del índice de PRECIO en el calendario de u (la misma serie de
    ^GSPC que usa la señal de la historia; la primera sesión, desde el 30-12-1927)."""
    px = history.gspc_close()
    pr = px.pct_change(fill_method=None).reindex(d["u"].index)
    if pr.isna().any():
        raise ValueError("índice de precio con huecos en el calendario de u")
    if not np.allclose(pr.iloc[1:].values, d["px"].pct_change().iloc[1:].values, atol=1e-12):
        raise ValueError("price_returns no reproduce d['px']")
    return pr


def run_worlds(d: dict, rng) -> tuple[dict, dict]:
    u = d["u"].dropna().values
    pr = price_returns(d).reindex(d["u"].dropna().index).values
    dv = float(np.mean(u - pr))
    rf_d, f_d = float(d["rf"].mean()), float(d["f"].mean())
    T = (r13.YEARS + r13.BURN) * 252
    worlds, meta = {}, {"dv": dv, "check": {}}
    for method, label in WORLDS:
        parts = []
        for b in range(N_WORLD // BATCH):
            U, PR = world_paths(method, u, pr, BATCH, T, rng, dv)
            if not (np.isfinite(U).all() and np.isfinite(PR).all()):
                raise ValueError(f"{label}: historias con NaN")
            if b == 0:  # con PR = U, world_strategies debe reproducir r13.mc_strategies
                err = check_against_r13(U[:20], U[:20], rf_d, f_d, d["spread"], d["er"])
                meta["check"][label] = err
                if err > 1e-12:
                    raise ValueError(f"{label}: world_strategies se aparta de r13.mc_strategies ({err:.2e})")
            parts.append(world_batch(U, PR, rf_d, f_d, d["spread"], d["er"]))
            del U, PR
        keys = list(FOCUS) + PX_KEYS
        w = {"Cm": {k: np.concatenate([p["Cm"][k] for p in parts]) for k in keys},
             "Crf_m": parts[0]["Crf_m"],
             "final": {k: _cat(parts, ("final", k)) for k in FOCUS},
             "maxdd": {k: _cat(parts, ("maxdd", k)) for k in FOCUS},
             "rel": {k: {f: _cat(parts, ("rel", k, f)) for f in ("maxdd", "longest", "recovered")}
                     for k in REL_KEYS},
             "lrs_invested": _cat(parts, ("lrs_invested",)),
             "lrs_switches": _cat(parts, ("lrs_switches",))}
        hz = [252 * h for h in (10, 20, r13.YEARS)]
        w["aband"] = {}
        for label_r, *_ in ABANDON_RULES:
            for k in OTHERS:
                key = (label_r, k)
                w["aband"][key] = {"j": _cat(parts, ("aband", key, "j")),
                                   "fin": {H: _cat(parts, ("aband", key, "fin", H)) for H in hz},
                                   "held": {H: _cat(parts, ("aband", key, "held", H)) for H in hz},
                                   "one": {H: _cat(parts, ("aband", key, "one", H)) for H in hz}}
        worlds[label] = w
    return worlds, meta


def ahead_worlds(worlds: dict) -> pd.DataFrame:
    rows = []
    for wl, w in worlds.items():
        Cm = w["Cm"]
        for k in Cm:
            bench = w["Crf_m"] if k == "1x" else Cm["1x"]
            for h in HORIZONS:
                rel = window_diff(Cm[k], bench, 12 * h)
                rows.append({"mundo": wl, "estrategia": k if k != "1x" else "1x (frente a letras)",
                             "años": h, "P": float((rel > 0).mean())})
    return pd.DataFrame(rows).pivot_table(index=["estrategia", "años"], columns="mundo", values="P")


def world_card(worlds: dict) -> pd.DataFrame:
    rows = []
    for wl, w in worlds.items():
        for k in FOCUS:
            for yrs, col in ((10, 12 * 10), (r13.YEARS, -1)):
                lw = w["Cm"][k][:, col] if col != -1 else w["final"][k]
                cagr = np.expm1(lw / yrs)
                rows.append({"mundo": wl, "estrategia": k, "años": yrs,
                             "CAGR_mediana": np.median(cagr), "CAGR_P5": np.percentile(cagr, 5),
                             "CAGR_P95": np.percentile(cagr, 95)})
            dd = w["maxdd"][k]
            rows[-1].update({"maxDD_mediana": np.median(dd), "maxDD_P5": np.percentile(dd, 5),
                             "maxDD_P95": np.percentile(dd, 95),
                             "P(caída > 80%)": float((dd < -0.8).mean())})
    return pd.DataFrame(rows)


def _behind_row(label: str, k: str, maxdd: np.ndarray, longest: np.ndarray, recovered: np.ndarray) -> dict:
    yrs = longest / 252
    return {"mundo": label, "estrategia": k,
            "caída_relativa_mediana": np.median(maxdd), "caída_relativa_P10": np.percentile(maxdd, 10),
            "racha_mediana_años": np.median(yrs), "racha_P90_años": np.percentile(yrs, 90),
            "P(racha ≥ 5 años)": float((yrs >= 5).mean()), "P(racha ≥ 10 años)": float((yrs >= 10).mean()),
            "P(racha abierta al final)": float((~recovered).mean())}


def behind_worlds(worlds: dict) -> pd.DataFrame:
    rows = [_behind_row(wl, k, w["rel"][k]["maxdd"], w["rel"][k]["longest"], w["rel"][k]["recovered"])
            for wl, w in worlds.items() for k in REL_KEYS]
    return pd.DataFrame(rows).set_index(["estrategia", "mundo"]).sort_index()


def behind_windows(df: pd.DataFrame, keys: list[str], years: int = r13.YEARS) -> pd.DataFrame:
    """Lo mismo que behind_worlds en ventanas históricas de `years` años (inicios mensuales)."""
    n = 252 * years
    starts = np.arange(0, len(df) - n + 1, STRIDE)
    W1 = start_windows(cum_log(df["1x"].values), starts, n)
    rows = []
    for k in keys:
        u = underwater(start_windows(cum_log(df[k].values), starts, n) - W1)
        rows.append(_behind_row(f"historia ({years} años)", k, u["maxdd"], u["longest"], u["recovered"]))
    return pd.DataFrame(rows).set_index(["estrategia", "mundo"]).sort_index()


def abandon_worlds(worlds: dict, years: int) -> pd.DataFrame:
    H = 252 * years
    out = []
    for wl, w in worlds.items():
        for label, *_ in ABANDON_RULES:
            for k in OTHERS:
                a = w["aband"][(label, k)]
                row = abandon_row(label, k, a["j"], a["held"][H], a["fin"][H], a["one"][H], H, years)
                out.append({"mundo": wl, **row})
    return pd.DataFrame(out)


# ───────────────────────────── crisis, días, fuera de muestra ─────────────────────────────

def exposure_changes(e: pd.Series, a, b) -> int:
    """Sesiones de (a, b] en que cambia la exposición objetivo (b = NaT: hasta el final)."""
    s = e[(e.index > a) & ((e.index <= b) if pd.notna(b) else True)]
    prev = e[e.index <= a].iloc[-1:]
    x = pd.concat([prev, s]).values
    return int((np.abs(np.diff(x)) > 1e-9).sum())


def crisis_tables(px: pd.Series, R: dict, E: dict | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cr = date_crises(px)
    head, a, b, c = [], [], [], []
    seen = set()
    for _, e in cr.iterrows():
        lab = f"{e.pico.year}–{str(e.valle.year)[2:]}" if e.valle.year != e.pico.year else f"{e.pico.year}"
        while lab in seen:
            lab += "′"
        seen.add(lab)
        head.append({"crisis": lab, "pico": e.pico.date(), "valle": e.valle.date(),
                     "recuperación": e["recuperación"].date() if pd.notna(e["recuperación"]) else "no",
                     "caída_precio": e["caída"],
                     "meses_pico_valle": (e.valle - e.pico).days / 30.44,
                     "años_hasta_recuperar": (e["recuperación"] - e.pico).days / 365.25
                     if pd.notna(e["recuperación"]) else np.nan})
        if E is not None:
            head[-1]["cambios_LRS_máx→mín"] = exposure_changes(E["LRS 3x"], e.pico, e.valle)
            head[-1]["cambios_LRS_máx→recuperación"] = exposure_changes(E["LRS 3x"], e.pico, e["recuperación"])
        a.append({"crisis": lab, **{k: period_return(R[k], e.pico, e.valle) for k in FOCUS}})
        b.append({"crisis": lab, **{k: period_return(R[k], e.pico, e["recuperación"]) for k in FOCUS}})
        c.append({"crisis": lab, **{k: years_to_regain(R[k], e.pico, e.valle) for k in FOCUS}})
    f = lambda rows: pd.DataFrame(rows).set_index("crisis")  # noqa: E731
    return f(head), f(a), f(b), f(c)


def days_tables(R: dict, E: dict, u: pd.Series, start_lrs, start_all,
                suspect: pd.DatetimeIndex | None = None) -> dict:
    ex3 = E["LRS 3x"].shift(LAG)
    exk = E["LRS-Kelly"].shift(LAG)
    exh = E["½ Kelly GJR"].shift(LAG)
    base = pd.DataFrame({"S&P 500": u, "3x": R["3x"], "LRS 3x: exposición": ex3, "LRS 3x": R["LRS 3x"],
                         "LRS-Kelly: exposición": exk, "LRS-Kelly": R["LRS-Kelly"],
                         "½ Kelly GJR: exposición": exh}).loc[start_lrs:]
    base = base[base["LRS 3x"].notna()]
    base["cierre dudoso (R14)"] = base.index.isin(suspect if suspect is not None else [])
    worst = base.nsmallest(15, "S&P 500")
    best = base.nlargest(15, "S&P 500")
    own = base.nsmallest(10, "LRS 3x")
    agg = []
    for per, st in (("1928–2026", start_lrs), ("1938–2026", start_all)):
        b = base.loc[st:]
        if per == "1938–2026":
            b = b[b["LRS-Kelly"].notna()]
        sets = [(f"{n} peores", b.nsmallest(n, "S&P 500")) for n in (10, 20, 50, 100)]
        sets += [(f"{n} mejores", b.nlargest(n, "S&P 500")) for n in (10, 20, 50, 100)]
        sets += [("todas las sesiones", b)]
        for lab, s in sets:
            agg.append({"periodo": per, "días": lab, "n": len(s),
                        "LRS 3x dentro": (s["LRS 3x: exposición"] > 0).mean(),
                        "S&P medio": s["S&P 500"].mean(), "3x medio": s["3x"].mean(),
                        "LRS 3x medio": s["LRS 3x"].mean(),
                        "LRS-Kelly exposición media": s["LRS-Kelly: exposición"].mean()
                        if per == "1938–2026" else np.nan,
                        "LRS-Kelly medio": s["LRS-Kelly"].mean() if per == "1938–2026" else np.nan,
                        "dudosos (R14)": int(s["cierre dudoso (R14)"].sum())})
    return {"peores": worst, "mejores": best, "propios": own,
            "agregado": pd.DataFrame(agg).set_index(["periodo", "días"])}


def oos_tables(d: dict, R: dict, E: dict, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    Rs = {k: R[k] for k in FOCUS}
    t = r13.stats_table(Rs, d["rf"], OOS_FROM)
    t = t[["CAGR", "vol", "sharpe", "maxDD", "años_bajo_agua", "peor_año", "10k_final"]]
    tr = r13.trading_stats({k: E[k] for k in FOCUS}, OOS_FROM)
    t["tiempo_invertido"] = tr["tiempo_invertido"]
    df = pd.DataFrame(Rs).loc[OOS_FROM:].dropna()
    C1 = cum_log(df["1x"].values)
    rows = {}
    for k in OTHERS:
        diff = log_growth(df[k].values) - log_growth(df["1x"].values)
        est, lo, hi = stats.bootstrap_ci(diff, lambda z: z.mean() * 252, 63, B_BOOT, rng)
        u = underwater(cum_log(df[k].values) - C1)
        full = d["u"].index
        pos = full.get_loc(df.index[0])
        dts = full[pos - 1: pos + len(df)]
        rows[k] = {"Δ_crec_log_anual": est, "IC90_bajo": lo, "IC90_alto": hi,
                   "inicio_racha_por_detrás": dts[u["long_start"][0]].date(),
                   "riqueza_relativa_final": float(np.exp(cum_log(df[k].values)[-1] - C1[-1])),
                   "caída_relativa_máx": u["maxdd"][0], "racha_por_detrás_años": u["longest"][0] / 252,
                   "racha_abierta": not u["recovered"][0]}
    return t, pd.DataFrame(rows).T.infer_objects()


# ───────────────────────────── procedencia ─────────────────────────────

def provenance() -> dict:
    """Hash del commit y huella SHA-256 de los archivos que fijan las reglas (sólo lectura)."""
    root = Path(__file__).resolve().parents[3]
    files = ["letf/letf_lab/research/r20_investor.py", "letf/letf_lab/research/r13_lrs.py",
             "letf/letf_lab/research/r12_long_ml.py", "letf/letf_lab/data/history.py",
             "letf/letf_lab/research/r3_synthetic.py", "letf/letf_lab/research/common.py",
             "letf/letf_lab/sim.py", "letf/letf_lab/stats.py", "letf/letf_lab/metrics.py",
             "letf/letf_lab/kelly.py"]
    git = lambda *a: subprocess.run(["git", "--no-optional-locks", *a], cwd=root,  # noqa: E731
                                    capture_output=True, text=True).stdout
    status = {}
    for line in git("status", "--porcelain", "--", *files).splitlines():
        # formato «XY ruta»: no se recorta la línea, el primer carácter puede ser un espacio
        status[line[3:]] = {"M": "modificado sin commit", "??": "sin seguimiento"}.get(
            line[:2].strip(), line[:2].strip())
    rows = []
    for f in files:
        p = root / f
        rows.append({"archivo": f, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "",
                     "estado_git": status.get(f, "igual que el commit")})
    return {"head": git("rev-parse", "HEAD").strip(), "files": pd.DataFrame(rows).set_index("archivo")}


# ───────────────────────────── figuras ─────────────────────────────

def fig_ahead(ah: pd.DataFrame, aw: pd.DataFrame, rep: Report):
    panels = ["1x (frente a letras)", "3x", "LRS 3x", PX3, "LRS-Kelly", "½ Kelly GJR"]
    fig, axes = plt.subplots(len(panels), 1, figsize=(FIG_W, 2.9 * len(panels)), sharex=True)
    for ax, k in zip(axes, panels):
        h = ah[ah.estrategia == k].set_index("años")
        ax.errorbar(h.index, h.P_historia * 100,
                    yerr=[(h.P_historia - h.IC90_bajo) * 100, (h.IC90_alto - h.P_historia) * 100],
                    color="k", marker="o", ms=8, lw=2, capsize=4, label="historia 1938–2026 (intervalo del 90%)")
        for wl, (col, mk) in WORLD_STYLE.items():
            s = aw.loc[k][wl]
            ax.plot(s.index, s.values * 100, color=col, marker=mk, ms=8, lw=2, ls="--",
                    label=f"mundo {wl}")
        ax.axhline(50, color="0.6", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(list(HORIZONS), [str(x) for x in HORIZONS])
        ax.minorticks_off()
        ax.set_ylim(0, 102)
        ttl = ("1x frente a las letras del Tesoro" if k.startswith("1x") else
               "LRS 3x con la señal de precio frente al 1x" if k == PX3 else
               "LRS 3x (original, señal de rentabilidad total) frente al 1x" if k == "LRS 3x" else
               f"{DISPLAY.get(k, k)} frente al 1x")
        ax.set(title=ttl, ylabel="P(por delante) %")
    axes[0].legend(loc="lower right", fontsize=11)
    axes[-1].set_xlabel("Horizonte de inversión (años, escala log)")
    rep.fig(fig, "p_delante_horizonte",
            "Probabilidad de acabar con más riqueza que la referencia según el horizonte. En negro, la "
            "historia del S&P 500 desde 1938 con todas las fechas de inicio; las barras son un intervalo "
            "del 90% aproximado que usa sólo las ventanas independientes (88 a 1 año, 4 a 20 años). "
            "En color, los tres mundos sintéticos de R13 (4.000 historias de 30 años cada uno) con la MISMA "
            "regla que la historia: la señal de la LRS sobre el índice de rentabilidad total. El cuarto panel "
            "repite la LRS 3x con la señal sobre un índice de precio, en la historia y en los mundos (en el "
            "bootstrap, el ^GSPC remuestreado junto con la rentabilidad total; en los otros dos, la "
            "rentabilidad total menos un dividendo constante). Los Kelly de los mundos prevén la varianza "
            "con EWMA. El primer panel compara el 1x con las letras del Tesoro. En las figuras, el medio "
            "Kelly (sin tendencia) es el ½ Kelly GJR de las tablas.")


def fig_relative(df: pd.DataFrame, dates: pd.DatetimeIndex, rep: Report, min_years: float = 1.0):
    ks = ["LRS 3x", PX3, "LRS-Kelly"]
    fig, axes = plt.subplots(len(ks), 1, figsize=(FIG_W, 3.7 * len(ks)), sharex=True)
    C1 = cum_log(df["1x"].values)
    for ax, k in zip(axes, ks):
        D = cum_log(df[k].values) - C1
        rw = pd.Series(np.exp(D), index=dates)
        under = D < np.maximum.accumulate(D) - 1e-12
        # rachas por debajo del máximo relativo previo
        spells, i = [], 0
        while i < len(under):
            if under[i]:
                j = i
                while j + 1 < len(under) and under[j + 1]:
                    j += 1
                spells.append((i, j))
                i = j + 1
            else:
                i += 1
        longest = max(spells, key=lambda s: s[1] - s[0])
        for a, b in spells:
            yrs = (b - a + 1) / 252
            if yrs < min_years:
                continue
            strong = (a, b) == longest
            ax.axvspan(dates[a - 1], dates[b], color=COLORS[k], alpha=0.28 if strong else 0.10, lw=0)
            if yrs >= 5:
                ax.text(dates[a - 1] + (dates[b] - dates[a - 1]) / 2, 0.97, f"{yrs:.0f} a",
                        transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11)
        ax.semilogy(rw, color=COLORS[k], lw=1.6)
        ax.yaxis.set_major_locator(plt.LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.axhline(1, color="0.5", lw=0.8)
        ax.axvline(pd.Timestamp(OOS_FROM), color="k", lw=1, ls=":")
        ax.text(pd.Timestamp(OOS_FROM), 0.03, " 2017", transform=ax.get_xaxis_transform(), fontsize=11)
        name = {"LRS 3x": "LRS 3x (original)"}.get(k, k)
        ax.set(title=f"{name} frente al 1x", ylabel="Estrategia / 1x (log)")
    fig.suptitle("Riqueza relativa frente al 1x desde 1938")
    rep.fig(fig, "riqueza_relativa",
            "Riqueza de la estrategia dividida por la del 1x (1 = empate) desde febrero de 1938, en escala "
            "logarítmica. Sombreado: rachas de al menos un año por debajo del máximo relativo previo, "
            "es decir, años seguidos en los que el inversor habría ido perdiendo terreno frente al 1x "
            "respecto a su mejor momento; la más oscura es la más larga y la cifra, su duración en "
            "años cuando supera cinco. La línea de puntos marca 2017, inicio del periodo fuera de muestra. "
            "El panel central es la misma LRS 3x con la señal sobre el índice de precio, que el 19-10-1987 "
            "estaba fuera; la original estaba dentro.")


def fig_card(card: pd.DataFrame, hist: pd.DataFrame, ah: pd.DataFrame, aw: pd.DataFrame, rep: Report):
    ks = list(FOCUS)
    y = np.arange(len(ks))[::-1]
    off = {"bootstrap": 0.22, "GJR": 0.0, "regímenes": -0.22}
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 13.5))
    c30 = card[card["años"] == r13.YEARS]
    for wl, (col, mk) in WORLD_STYLE.items():
        s = c30[c30.mundo == wl].set_index("estrategia").loc[ks]
        axes[0].errorbar(s.CAGR_mediana * 100, y + off[wl],
                         xerr=[(s.CAGR_mediana - s.CAGR_P5) * 100, (s.CAGR_P95 - s.CAGR_mediana) * 100],
                         fmt=mk, color=col, ms=8, lw=2, capsize=3, label=f"mundo {wl}: mediana y P5–P95")
        axes[1].errorbar(s.maxDD_mediana * 100, y + off[wl],
                         xerr=[(s.maxDD_mediana - s.maxDD_P5) * 100, (s.maxDD_P95 - s.maxDD_mediana) * 100],
                         fmt=mk, color=col, ms=8, lw=2, capsize=3)
        p10 = [aw.loc[("1x (frente a letras)" if k == "1x" else k, 10), wl] for k in ks]
        axes[2].plot(np.array(p10) * 100, y + off[wl], mk, color=col, ms=9)
    axes[0].plot(hist.loc[ks, "CAGR"] * 100, y, "*", color="k", ms=15, label="historia 1938–2026 (88 años)")
    axes[1].plot(hist.loc[ks, "maxDD"] * 100, y, "*", color="k", ms=15)
    h10 = ah[ah["años"] == 10].set_index("estrategia")
    ph = [h10.loc["1x (frente a letras)" if k == "1x" else k] for k in ks]
    hist_ci = axes[2].errorbar([r.P_historia * 100 for r in ph], y,
                               xerr=[[(r.P_historia - r.IC90_bajo) * 100 for r in ph],
                                     [(r.IC90_alto - r.P_historia) * 100 for r in ph]],
                               fmt="*", color="k", ms=15, capsize=3)
    axes[0].axvline(0, color="0.6", lw=0.8)
    axes[2].axvline(50, color="0.6", lw=0.8)
    for ax in axes:
        ax.set_yticks(y, [DISPLAY.get(k, k).replace(" (", "\n(") for k in ks])
        ax.grid(axis="x", color="0.9")
    axes[0].set(title=f"Rentabilidad anual compuesta a {r13.YEARS} años", xlabel="% anual")
    axes[1].set(title=f"Caída máxima en {r13.YEARS} años", xlabel="% desde máximos")
    axes[2].set(title="P(por delante del 1x a 10 años); 1x: frente a letras", xlabel="%", xlim=(0, 102))
    h_, l_ = axes[0].get_legend_handles_labels()
    # la raya negra de las estrellas del panel inferior también va en la leyenda
    h_.insert(1, hist_ci)
    l_.insert(1, "historia en el panel inferior, con su intervalo del 90% aproximado")
    axes[2].legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=11)
    rep.fig(fig, "ficha",
            "Ficha visual de las cinco estrategias. Arriba, la rentabilidad anual a 30 años en los tres "
            "mundos sintéticos (marcador: mediana; barra: del percentil 5 al 95) y, con estrella, la de "
            "la historia 1938–2026, que es una sola trayectoria de 88 años. En medio, la caída máxima en "
            "30 años con la misma codificación (la estrella es la caída máxima histórica en 88 años). "
            "Abajo, la probabilidad de ir por delante del 1x a 10 años (el 1x, frente a las letras); en ese "
            "panel, la raya negra de cada estrella es el intervalo del 90% aproximado de la historia, con "
            "solo las ventanas independientes. En los mundos la señal de la LRS usa el índice de rentabilidad "
            "total, como la original en la historia, y el medio Kelly (sin tendencia) y la LRS-Kelly prevén "
            "la varianza con EWMA.")


def crisis_label(name: str, drop: float) -> str:
    """Rótulo de una crisis en crisis.png: «2018 (−19,8%)», «1938–39 (−25,1%)». La caída del
    índice va con un decimal (redondeada a entero, un −19,8% se leía como −20%) y con el
    signo «−» de num_es; el nombre no se toca."""
    return f"{name} ({num_es(drop * 100, 1)}%)"


def fig_crises(a: pd.DataFrame, b: pd.DataFrame, head: pd.DataFrame, rep: Report, since: int = 1938):
    keep = [c for c in a.index if int(c[:4]) >= since]
    lab = [crisis_label(c, head.loc[c, "caída_precio"]) for c in keep]
    y = np.arange(len(keep))[::-1]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 0.34 * len(keep) * 2 + 2.6), sharey=True)
    ks = ["1x", "LRS 3x", "LRS-Kelly", "½ Kelly GJR"]
    mk = {"1x": "o", "LRS 3x": "s", "LRS-Kelly": "D", "½ Kelly GJR": "^"}
    for ax, t, ttl in ((axes[0], a, "Del máximo al mínimo del S&P 500"),
                       (axes[1], b, "Del máximo a la recuperación del máximo del S&P 500")):
        for k in ks:
            ax.plot(t.loc[keep, k] * 100, y, mk[k], color=COLORS[k], ms=8, label=DISPLAY.get(k, k), alpha=0.9)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_yticks(y, lab)
        ax.grid(axis="x", color="0.9")
        ax.set(title=ttl, xlabel="Rentabilidad acumulada (%)")
    # la leyenda en su propia franja: dentro del panel tapaba puntos
    legend_band(fig, *axes[0].get_legend_handles_labels(), ncol=4, fontsize=11, columnspacing=1.2,
                handletextpad=0.3)
    rep.fig(fig, "crisis",
            "Rentabilidad de cada estrategia en las crisis del S&P 500 desde 1938 (caídas de al menos el "
            "19% desde el máximo; entre paréntesis, la caída del índice de precio). Arriba, del cierre "
            "del máximo al del mínimo; abajo, del máximo al primer cierre que lo recupera (la crisis de "
            "cada fila sin recuperación no tiene puntos abajo).")


OOS_EPISODES = [("2018 (regla)", "2018-09-20", "2018-12-24"),
                ("2020 (regla)", "2020-02-19", "2020-03-23"),
                ("2022, mínimo de junio (regla)", "2022-01-03", "2022-06-16"),
                ("2022, mínimo de octubre (datación habitual)", "2022-01-03", "2022-10-12"),
                ("2025 (fuera de la regla)", "2025-02-19", "2025-04-08")]


def episodes_table(px: pd.Series, R: dict, E: dict | None = None, eps=OOS_EPISODES) -> pd.DataFrame:
    """Episodios del periodo fuera de muestra con fechas fijas (descriptivo)."""
    rows = []
    for lab, a, b in eps:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        after = px.loc[b:]
        rec = after[after >= px.loc[a]].first_valid_index()
        head = {"episodio": lab, "caída_precio": px.loc[b] / px.loc[a] - 1,
                "meses_máx→mín": (b - a).days / 30.44,
                "recuperación": rec.date() if rec is not None else "no"}
        if E is not None:
            head["cambios_LRS_máx→recuperación"] = exposure_changes(E["LRS 3x"], a, rec if rec is not None else pd.NaT)
        rows.append({**head, "tramo": "máx→mín", **{k: period_return(R[k], a, b) for k in FOCUS}})
        rows.append({**head, "tramo": "máx→recuperación", **{k: period_return(R[k], a, rec) for k in FOCUS}})
    return pd.DataFrame(rows).set_index(["episodio", "tramo"])


def kelly_premium(d: dict) -> pd.Series:
    """Prima anual que usa r13.kelly_exposure: μ̂ − f̂ − s, con μ̂ − f̂ la prima compartida
    (kelly.premium: media creciente de u − f hasta la sesión anterior, mínimo 10 años)."""
    return kelly.premium(d["u"], d["f"], r12.MIN_OBS) - d["spread"]


def kelly_diagnosis(d: dict, R: dict, E: dict, var_fc: pd.Series, start, peak, trough) -> dict:
    """Por qué las estrategias Kelly no recuperan su máximo relativo frente al 1x.

    Cada sesión se clasifica por lo que se decidió LAG sesiones antes (la posición que cobra):
    μ̂ − f̂ − s ≤ 0 (Kelly = 0 sea cual sea la tendencia), prima > 0 con el S&P bajo su media
    (fuera por el filtro, sólo LRS-Kelly) y prima > 0 sobre la media. Los tramos intermedios
    (fin de 1978 y de 1984, los años de tipos altos) se eligieron tras ver el gráfico."""
    prem = kelly_premium(d)
    neg = (prem <= 0).loc[start:]
    share = neg.groupby(neg.index.year).mean()
    sig = r13.above_sma(r13.signal_index(d))   # la señal de la LRS original
    p_l, s_l = prem.shift(LAG), sig.shift(LAG)
    rel = {k: pd.Series(log_growth(R[k]) - log_growth(R["1x"]), index=R[k].index) for k in ("LRS-Kelly", "½ Kelly GJR")}
    cats = {"LRS-Kelly": {"μ̂ − f̂ − s ≤ 0": p_l <= 0, "prima > 0, bajo la media": (p_l > 0) & (s_l == 0),
                          "prima > 0, sobre la media": (p_l > 0) & (s_l == 1)},
            "½ Kelly GJR": {"μ̂ − f̂ − s ≤ 0": p_l <= 0, "prima > 0": p_l > 0}}
    end = R["1x"].dropna().index[-1]
    bounds = [pd.Timestamp(peak), pd.Timestamp("1978-12-31"), pd.Timestamp("1984-12-31"), pd.Timestamp(trough), end]
    tabs = {}
    for k in rel:
        parts = [attribution(rel[k], cats[k], bounds).assign(tramo="por tramos"),
                 attribution(rel[k], cats[k], [peak, trough]).assign(tramo="máximo → mínimo relativo"),
                 attribution(rel[k], cats[k], [peak, end]).assign(tramo="máximo → hoy")]
        tabs[k] = pd.concat(parts).set_index(["tramo", "desde", "hasta"])
    e = E["LRS-Kelly"].shift(LAG)
    u, r = R["1x"], R["LRS-Kelly"]
    yrs = slice(pd.Timestamp(peak).year, pd.Timestamp(trough).year)
    g = lambda x: x.groupby(x.index.year)  # noqa: E731
    yearly = pd.DataFrame({"LRS-Kelly": g(1 + r).prod() - 1, "1x": g(1 + u).prod() - 1,
                           "exposición media (x)": g(e).mean(), "sesiones μ̂−f̂−s ≤ 0": g(p_l <= 0).mean(),
                           "sesiones sobre la media": g(s_l == 1).mean(),
                           "f": g(d["f"]).mean() * 252, "μ̂ − f̂": g(prem + d["spread"]).mean(),
                           "σ̂ GJR": g(np.sqrt(var_fc)).mean()}).loc[yrs]
    yearly.insert(2, "relativa", (1 + yearly["LRS-Kelly"]) / (1 + yearly["1x"]) - 1)
    yearly.index.name = "año"
    per = slice("1980-01-01", "1984-12-31")
    band = (prem + d["spread"]).loc[pd.Timestamp(peak):pd.Timestamp(trough)]
    return {"sesiones": int(neg.sum()), "años": [int(y) for y, v in share.items() if v > 0.5],
            "prima_mín": float(prem.loc[start:].min()), "prima_rango": (float(band.min()), float(band.max())),
            "exp_lrsk": float(E["LRS-Kelly"].loc[per].mean()), "exp_half": float(E["½ Kelly GJR"].loc[per].mean()),
            "prima": float((prem + d["spread"]).loc[per].mean()), "f": float(d["f"].loc[per].mean() * 252),
            "tabs": tabs, "yearly": yearly}


def signal_gap(rp: pd.Series, rt: pd.Series, start) -> dict:
    """Diferencia de crecimiento log «precio − original» entre la LRS con señal de precio (rp) y
    con la señal original de rentabilidad total (rt): total anual, el día que más aporta y lo
    que queda sin él."""
    diff = pd.Series(log_growth(rp) - log_growth(rt), index=rp.index).loc[start:].dropna()
    yrs = len(diff) / 252
    top = diff.abs().idxmax()
    return {"anual": diff.sum() / yrs, "día": top, "aporte_día": float(diff[top]),
            "cuota_día": float(diff[top] / diff.sum()), "anual_sin_día": (diff.sum() - diff[top]) / yrs}


def behind_at_highs(df: pd.DataFrame, dates: pd.DatetimeIndex, keys: list[str]) -> pd.DataFrame:
    """Como behind_history, con la riqueza relativa leída sólo en los máximos históricos del 1x."""
    C1 = cum_log(df["1x"].values)
    hi = C1 >= np.maximum.accumulate(C1) - 1e-12
    src = np.maximum.accumulate(np.where(hi, np.arange(len(C1)), 0))  # día de máximo del 1x que se lee
    T1 = len(C1)
    rows = {}
    for k in keys:
        D = at_highs(cum_log(df[k].values) - C1, C1)
        g = {f: v[0] for f, v in underwater(D).items()}
        # la racha va del máximo del 1x en que se leyó el máximo relativo al primero que lo supera
        s0 = src[g["long_start"]]
        s1 = g["long_end"] + 1 if g["recovered"] else T1 - 1
        rows[k] = {"caída_relativa_máx": g["maxdd"], "máximo_relativo": dates[src[g["peak"]]].date(),
                   "racha_más_larga_años": (s1 - s0) / 252, "racha_desde": dates[s0].date(),
                   "racha_hasta": dates[s1].date() if g["recovered"] else "sigue"}
    return pd.DataFrame(rows).T.infer_objects()


def fixed_close_data(d: dict, dates=SUSPECT_CLOSES) -> dict:
    """d con los cierres dudosos de R14 corregidos (media geométrica de los vecinos) en el
    índice de precio y en la rentabilidad total (antes de 1988, u = precio + D/P de Shiller)."""
    px = history.gspc_close()
    px_fix = fix_closes(px, dates)
    delta = (px_fix.pct_change(fill_method=None) - px.pct_change(fill_method=None)).reindex(d["u"].index).fillna(0.0)
    if (delta.abs() > 0).any() and delta[delta.abs() > 0].index.max() > pd.Timestamp("1987-12-31"):
        raise ValueError("corrección posterior a 1988: u ya no es precio + dividendo")
    real = r12.real_letf("UPRO")
    chk = r12.make(d["u"], d["rf"], "UPRO", real)
    if not np.allclose(chk["x3"].values, d["x3"].values, equal_nan=True):
        raise ValueError("r12.make no reproduce el 3x de la base")
    d2 = r12.make(d["u"] + delta, d["rf"], "UPRO", real)
    d2["px"] = px_fix.reindex(d["u"].index).ffill()
    d2["px_real"] = True
    d2["u_sig"] = r13.sp500_signal_returns(d2["u"])   # la señal original sale de la u corregida
    return d2


# ───────────────────────────── informe ─────────────────────────────

def show(rep: Report, df: pd.DataFrame, **kw):
    """rep.table con los niveles de un MultiIndex como columnas (markdown legible)."""
    if isinstance(df.index, pd.MultiIndex):
        rep.table(df.reset_index(), index=False, **kw)
    else:
        rep.table(df, **kw)


WORDS_M = {1: "uno", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco"}
WORDS_F = {1: "una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco"}


def pl(n: int, one: str, many: str) -> str:
    """Singular o plural según n."""
    return one if n == 1 else many


def pc(x: float, d: int = 1, sign: bool = False) -> str:
    """Porcentaje con coma decimal para el texto."""
    if pd.isna(x):
        return "—"
    s = f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"
    return s.replace(".", ",").replace("-", "−")


def num(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def lst(xs: list[str]) -> str:
    return ", ".join(xs[:-1]) + " y " + xs[-1] if len(xs) > 1 else (xs[0] if xs else "ninguna")


def signal_worlds(worlds: dict) -> pd.DataFrame:
    """«precio − original»: la LRS 3x con señal de precio frente a la original (señal de
    rentabilidad total), en las mismas historias de cada mundo (30 años)."""
    rows = {}
    for wl, w in worlds.items():
        a, b = w["Cm"][PX3][:, -1], w["Cm"]["LRS 3x"][:, -1]
        yrs = r13.YEARS
        rows[wl] = {"Δ_log_anual_media": float(np.mean(a - b)) / yrs,
                    "Δ_CAGR_mediana": float(np.median(np.expm1(a / yrs) - np.expm1(b / yrs))),
                    "P(precio > original)": float(np.mean(a > b))}
    return pd.DataFrame(rows).T


def run():
    rep = Report("r20_investor", "R20 — Qué puede esperar un inversor (I7)")
    rng_w = np.random.default_rng(SEED)
    rng_b = np.random.default_rng(SEED + 1)
    prov = provenance()

    # ── datos y estrategias (R13 tal cual) ──
    D = r13.load()
    d = D["S&P 500"]
    var_fc = r13.var_forecast(d)
    R13, E13 = r13.strategies(d, var_fc)
    R = {s: R13[k] for s, k in FOCUS.items()}
    E = {s: E13[k] for s, k in FOCUS.items()}
    # sensibilidad: la LRS 3x y la LRS-Kelly con la señal sobre el índice de precio (R13)
    Rp13, Ep13 = r13.price_variants(d, var_fc)
    for k in PX_KEYS:
        R[k], E[k] = Rp13[k], Ep13[k]
    ALL = list(FOCUS) + PX_KEYS
    K28 = ["1x", "3x", "LRS 3x", PX3]
    start_all = R["LRS-Kelly"].first_valid_index()
    start_lrs = R["LRS 3x"].first_valid_index()
    df, rf, dates = history_frame(d, R, start_all)
    df28, rf28, dates28 = history_frame(d, {k: R[k] for k in K28}, start_lrs)
    end = df.index[-1]
    hist = r13.stats_table(R, d["rf"], start_all)
    trade = r13.trading_stats(E, start_all)
    yrs_hist = len(df) / 252

    # ── cálculos ──
    ah = ahead_history(df, rf, ALL)
    ah28 = ahead_history(df28, rf28, K28)
    worlds, wmeta = run_worlds(d, rng_w)
    aw = ahead_worlds(worlds)
    card = world_card(worlds)
    sgw = signal_worlds(worlds)
    sg = signal_gap(R[PX3], R["LRS 3x"], start_all)   # precio − original
    bh = behind_history(df, dates, REL_KEYS)
    bh_hi = behind_at_highs(df, dates, REL_KEYS)
    bh28 = behind_history(df28, dates28, ["3x", "LRS 3x", PX3])
    bw = pd.concat([behind_worlds(worlds), behind_windows(df, REL_KEYS)]).sort_index()
    ab_h = {y: abandon_history(df, y) for y in (10, 20)}
    ab_w = {y: abandon_worlds(worlds, y) for y in (10, 20, r13.YEARS)}
    head, cr_a, cr_b, cr_c = crisis_tables(d["px"].loc[:end], R, E)
    susp = suspect_sessions(d["u"].index)
    days = days_tables(R, E, d["u"], start_lrs, start_all, susp)
    d_fix = fixed_close_data(d)
    Rf13, Ef13 = r13.strategies(d_fix, r13.var_forecast(d_fix))
    R_fix = {s: Rf13[k] for s, k in FOCUS.items()}
    E_fix = {s: Ef13[k] for s, k in FOCUS.items()}
    days_fix = days_tables(R_fix, E_fix, d_fix["u"], start_lrs, start_all)
    hist_fix = r13.stats_table(R_fix, d_fix["rf"], start_all)
    oos_t, oos_rel = oos_tables(d, R, E, rng_b)
    eps = episodes_table(d["px"], R, E)
    kd = kelly_diagnosis(d, R, E, var_fc, start_all, bh.loc["LRS-Kelly", "máximo_relativo"],
                         bh.loc["LRS-Kelly", "mínimo_relativo"])

    # ── atajos para el texto ──
    AH = ah.set_index(["estrategia", "años"])
    P = lambda k, h: AH.loc[("1x (frente a letras)" if k == "1x" else k, h)]  # noqa: E731
    Wp = lambda k, h, w: aw.loc[("1x (frente a letras)" if k == "1x" else k, h), w]  # noqa: E731
    wl3 = [m for _, m in WORLDS]
    wtxt = lambda k, h: " / ".join(pc(Wp(k, h, w), 0) for w in wl3)  # noqa: E731
    wmin = lambda k, h: min(Wp(k, h, w) for w in wl3)  # noqa: E731
    wmax = lambda k, h: max(Wp(k, h, w) for w in wl3)  # noqa: E731
    bwt = lambda k, col, f: " / ".join(f(bw.loc[(k, w), col]) for w in wl3)  # noqa: E731
    HW = f"historia ({r13.YEARS} años)"
    c30 = card[card["años"] == r13.YEARS].set_index(["estrategia", "mundo"])
    c10 = card[card["años"] == 10].set_index(["estrategia", "mundo"])
    since38 = [c for c in head.index if int(c[:4]) >= 1938]
    rec_ok = [c for c in since38 if pd.notna(cr_b.loc[c, "1x"])]
    less = [c for c in since38 if cr_a.loc[c, "LRS 3x"] > cr_a.loc[c, "1x"]]
    ahead_rec = [c for c in rec_ok if cr_b.loc[c, "LRS 3x"] > cr_b.loc[c, "1x"]]
    behind_rec = [c for c in rec_ok if cr_b.loc[c, "LRS 3x"] <= cr_b.loc[c, "1x"]]
    cd = crisis_by_duration(head, cr_a, cr_b, since38)
    CL = 12  # umbral que se comenta en el texto (la rejilla completa va en la tabla)
    long_ = [c for c in since38 if head.loc[c, "meses_pico_valle"] >= CL]
    long_rec = [c for c in long_ if c in rec_ok]
    long_ahead = [c for c in long_rec if c in ahead_rec]
    long_behind = [c for c in long_rec if c not in ahead_rec]
    short_ = [c for c in since38 if head.loc[c, "meses_pico_valle"] < CL]
    short_rec = [c for c in short_ if c in rec_ok]
    short_ahead = [c for c in short_rec if c in ahead_rec]
    short_behind = [c for c in short_rec if c not in ahead_rec]
    fr = cd["por delante (máx→recuperación)"] / cd["recuperadas"].replace(0, np.nan)
    better = [fr.loc[(X, f"≥ {X} meses")] > fr.loc[(X, f"< {X} meses")] for X in CRISIS_MONTHS]
    grid_txt = ("En toda la rejilla de umbrales (3 a 18 meses), la proporción por delante es mayor en las largas "
                "que en las cortas (sección 4)." if all(better) else
                f"En la rejilla de umbrales (3 a 18 meses), la proporción por delante es mayor en las largas en "
                f"{sum(better)} de {len(better)} umbrales (sección 4).")
    mid = [c for c in since38 if 1942 <= int(c[:4]) < 2000]
    deep_7374 = head.loc[mid, "caída_precio"].idxmin() == "1973–74"
    long_7374 = head.loc[mid, "meses_pico_valle"].idxmax() == "1973–74"
    sup_7374 = ("el bajista más largo y profundo entre 1942 y 2000" if deep_7374 and long_7374 else
                "el bajista más profundo entre 1942 y 2000" if deep_7374 else
                "el bajista más largo entre 1942 y 2000" if long_7374 else "un bajista largo")
    mh = lambda y: ab_h[y][ab_h[y].regla == MAIN_RULE].set_index("estrategia")  # noqa: E731
    mw = lambda y, w: ab_w[y][(ab_w[y].regla == MAIN_RULE) & (ab_w[y].mundo == w)].set_index("estrategia")  # noqa: E731
    gh = lambda y, rule, k: ab_h[y].set_index(["regla", "estrategia"]).loc[(rule, k)]  # noqa: E731
    C1 = cum_log(df["1x"].values)
    rwk = pd.Series(np.exp(cum_log(df["LRS-Kelly"].values) - C1), index=dates)
    ag = days["agregado"]
    kt = kd["tabs"]["LRS-Kelly"]
    kt_h = kd["tabs"]["½ Kelly GJR"]
    kpt = kt.xs("máximo → mínimo relativo", level="tramo").iloc[0]
    kpe = kt.xs("máximo → hoy", level="tramo").iloc[0]
    kht = kt_h.xs("máximo → mínimo relativo", level="tramo").iloc[0]
    kseg = kt.xs("por tramos", level="tramo")
    neg_c = "log: μ̂ − f̂ − s ≤ 0"
    ky = kd["yearly"]
    k_bad = ky.nsmallest(6, "relativa").sort_index()
    peak_k = pd.Timestamp(bh.loc["LRS-Kelly", "máximo_relativo"])
    sp_trough = head[pd.to_datetime(head["valle"]) == peak_k].index
    e22 = [e for e in eps.index.get_level_values(0).unique() if "octubre" in e][0]
    sw22 = int(eps.loc[(e22, "máx→mín"), "cambios_LRS_máx→recuperación"])
    m22 = eps.loc[(e22, "máx→mín"), "meses_máx→mín"]
    px3 = PX3

    # ── introducción y resumen ──
    rep.p(__doc__.split("\n", 2)[2])
    rep.p(f"Datos: S&P 500 con dividendos desde {start_all.date()} (primera sesión con las cinco "
          f"estrategias) hasta {end.date()}, {num(yrs_hist)} años; el 1x, el 3x y la LRS 3x también desde "
          f"{start_lrs.date()}. Base de código: commit `{prov['head']}` con los cambios sin commit que había "
          "en el árbol de trabajo al ejecutar (sección 7). Con esa base, la muestra de R13 da 1x "
          f"{pc(hist.loc['1x', 'CAGR'])}, 3x {pc(hist.loc['3x', 'CAGR'])}, LRS 3x {pc(hist.loc['LRS 3x', 'CAGR'])}, "
          f"LRS-Kelly {pc(hist.loc['LRS-Kelly', 'CAGR'])} y ½ Kelly GJR {pc(hist.loc['½ Kelly GJR', 'CAGR'])} al año.")
    x87 = r13.distance_1987(d)
    c_tr, c_px = x87.columns
    cross = {c: x87.index[(x87[c] < 0).values][0] for c in x87}
    nxt = lambda day: d["u"].index[d["u"].index.get_loc(pd.Timestamp(day)) + 1]  # noqa: E731
    rep.p("**La señal original y la de precio.** La LRS original, la de Gayed y Bilello, decide con el índice "
          "de rentabilidad total del S&P 500 (con dividendos). Además de las cinco estrategias, el informe "
          "evalúa la LRS 3x y la LRS-Kelly con la señal sobre el índice de precio («señal de precio»), una "
          "sensibilidad del laboratorio. Los mundos aplican la MISMA regla que la historia (señal de "
          "rentabilidad total), y las dos señales se comparan en la historia y en los mundos.")
    rep.h("Resumen")
    rep.p(
        f"- **Ir por delante del 1x.** Con la regla original (señal de rentabilidad total) en la historia y en "
        f"los mundos, la LRS 3x fue por delante del 1x en el {pc(P('LRS 3x', 1).P_historia, 0)} de las ventanas "
        f"históricas de 1 año, el {pc(P('LRS 3x', 10).P_historia, 0)} de las de 10 (intervalo aproximado "
        f"{pc(P('LRS 3x', 10).IC90_bajo, 0)}–{pc(P('LRS 3x', 10).IC90_alto, 0)}; sólo "
        f"{int(P('LRS 3x', 10).independientes)} ventanas independientes) y el {pc(P('LRS 3x', 20).P_historia, 0)} de "
        f"las de 20 ({int(P('LRS 3x', 20).independientes)} independientes). En los mundos (bootstrap / GJR / "
        f"regímenes) es {wtxt('LRS 3x', 1)} a 1 año y {wtxt('LRS 3x', 10)} a 10. Con la señal de precio, la "
        f"historia da {pc(P(px3, 10).P_historia, 0)} a 10 años ({pc(P(px3, 10).IC90_bajo, 0)}–"
        f"{pc(P(px3, 10).IC90_alto, 0)}) y {pc(P(px3, 20).P_historia, 0)} a 20, frente a {wtxt(px3, 10)} en los "
        f"mundos. En la historia, la señal de precio crece {num(sg['anual'] * 100, 2)} puntos al año más que la "
        f"original (crecimiento log), y el {pc(sg['cuota_día'], 0)} de esa diferencia sale de un solo día, el "
        f"{sg['día'].strftime('%d-%m-%Y')}, que la señal de precio esquivó y la original no. En los mundos, que "
        f"conservan la diferencia de dividendos, la señal de precio aporta "
        f"{' / '.join(num(v * 100, 2) for v in sgw.loc[wl3, 'Δ_log_anual_media'])} puntos al año. A 10 años, la "
        f"historia queda {num((P('LRS 3x', 10).P_historia - wmax('LRS 3x', 10)) * 100, 0)} puntos por encima del "
        f"mejor mundo con la señal original y {num((P(px3, 10).P_historia - wmax(px3, 10)) * 100, 0)} con la de "
        f"precio. Quien empiece hoy con la regla original debería esperar algo entre el "
        f"{pc(wmin('LRS 3x', 10), 0)} y el {pc(wmax('LRS 3x', 10), 0)} a 10 años. El 3x no mejora con el plazo "
        f"(historia {pc(P('3x', 1).P_historia, 0)} a 1 año y {pc(P('3x', 20).P_historia, 0)} a 20). La LRS-Kelly y "
        f"el ½ Kelly GJR rondan el {pc(P('LRS-Kelly', 10).P_historia, 0)} a 10 años en la historia y en el mundo "
        f"bootstrap.\n"
        f"- **Años por detrás.** La riqueza relativa de la LRS 3x frente al 1x llegó a caer un "
        f"{pc(-bh.loc['LRS 3x', 'caída_relativa_máx'], 0)} ({bh.loc['LRS 3x', 'máximo_relativo']} → "
        f"{bh.loc['LRS 3x', 'mínimo_relativo']}), y su racha más larga sin superar el máximo relativo previo duró "
        f"{num(bh.loc['LRS 3x', 'racha_más_larga_años'])} años ({bh.loc['LRS 3x', 'racha_desde']} → "
        f"{bh.loc['LRS 3x', 'racha_hasta']}). Con la señal de precio, la caída es del "
        f"{pc(-bh.loc[px3, 'caída_relativa_máx'], 0)} y la racha dura {num(bh.loc[px3, 'racha_más_larga_años'])} años "
        f"({bh.loc[px3, 'racha_desde']} → {bh.loc[px3, 'racha_hasta']}). En 30 "
        f"años de un mundo, con la señal original, lo típico (mediana) es una caída relativa de "
        f"{bwt('LRS 3x', 'caída_relativa_mediana', lambda v: pc(v, 0))} y una racha de "
        f"{bwt('LRS 3x', 'racha_mediana_años', num)} años. La LRS-Kelly y el ½ Kelly GJR no han recuperado su "
        f"máximo relativo del {peak_k.strftime('%d-%m-%Y')}, el día del mínimo del S&P, con ellas fuera del "
        f"mercado; leída sólo en los máximos del 1x, la LRS-Kelly lleva por detrás desde "
        f"{pd.Timestamp(bh_hi.loc['LRS-Kelly', 'racha_desde']).year} ({num(bh_hi.loc['LRS-Kelly', 'racha_más_larga_años'])} "
        f"años). De la pérdida relativa de la LRS-Kelly hasta {pd.Timestamp(bh.loc['LRS-Kelly', 'mínimo_relativo']).year} "
        f"(riqueza relativa ×{num(kpt['factor_relativo'], 2)}), "
        + ("nada corresponde a sesiones con μ̂ − f̂ − s ≤ 0 (no hubo ninguna desde 1938). Toda se perdió con prima "
           "positiva: fuera por el filtro " if kd["sesiones"] == 0 else
           f"sólo el {pc(kpt[neg_c] / kpt['log_relativo'], 0)} corresponde a sesiones con μ̂ − f̂ − s ≤ 0. El resto "
           "se perdió con prima positiva: fuera por el filtro ") +
        f"de tendencia ({pc(kpt['log: prima > 0, bajo la media'] / kpt['log_relativo'], 0)}) o dentro con poca "
        f"exposición o con pérdidas apalancadas ({pc(kpt['log: prima > 0, sobre la media'] / kpt['log_relativo'], 0)}).\n"
        f"- **Abandono.** Con la regla «{MAIN_RULE}», a 20 años abandonaría la LRS 3x el "
        f"{pc(mh(20).loc['LRS 3x', 'P_abandona'], 0)} de los inversores históricos (mediana: a los "
        f"{num(mh(20).loc['LRS 3x', 'años_hasta_abandono'])} años) y al menos el "
        f"{pc(min(mw(r13.YEARS, w).loc['LRS 3x', 'P_abandona'] for w in wl3), 1)} en 30 años de cualquier mundo. "
        f"A quien abandona le cuesta {num(mh(20).loc['LRS 3x', 'coste_si_abandona'] * 100)} puntos al año en la "
        f"mediana histórica. En los mundos, a 30 años, le cuesta "
        f"{' / '.join(num(mw(r13.YEARS, w).loc['LRS 3x', 'coste_si_abandona'] * 100) for w in wl3)} puntos, y "
        "abandonar convierte la LRS 3x en algo parecido al 1x (mundo bootstrap: "
        f"{pc(mw(r13.YEARS, 'bootstrap').loc['LRS 3x', 'CAGR_con_abandono'])} con abandono, "
        f"{pc(mw(r13.YEARS, 'bootstrap').loc['LRS 3x', 'CAGR_mantener'])} manteniéndola, "
        f"{pc(mw(r13.YEARS, 'bootstrap').loc['LRS 3x', 'CAGR_1x'])} el 1x).\n"
        f"- **Crisis** (caídas del S&P ≥ 19%): desde 1938 hay {len(since38)}. Del máximo al mínimo, la LRS 3x "
        f"perdió menos que el 1x sólo en {len(less)}; del máximo a su recuperación acabó por delante en "
        f"{len(ahead_rec)} de {len(rec_ok)}. Tiende a ganar en los bajistas largos, pero no siempre: de las "
        f"{len(long_rec)} crisis ya recuperadas con al menos {CL} meses del máximo al mínimo, va por delante en "
        f"{len(long_ahead)}" + (f", y las excepciones son {lst(long_behind)}" if long_behind else "") +
        f" (1973–74: {pc(cr_b.loc['1973–74', 'LRS 3x'], 0, True)} frente a {pc(cr_b.loc['1973–74', '1x'], 0, True)}). "
        f"En las más cortas, va por delante en {len(short_ahead)} de {len(short_rec)}. {grid_txt} 2022 no fue una "
        f"caída en V: fueron {num(m22)} meses de bajista con repuntes hasta el mínimo de octubre, y la señal "
        f"cambió {sw22} veces entre el máximo y la recuperación.\n"
        f"- **Días extremos.** Estuvo fuera en {int(round((1 - ag.loc[('1928–2026', '10 peores'), 'LRS 3x dentro']) * 10))} "
        f"de los 10 peores días del S&P desde 1928, pero también en "
        f"{int(round((1 - ag.loc[('1928–2026', '10 mejores'), 'LRS 3x dentro']) * 10))} de los "
        "10 mejores: los días extremos llegan juntos, en las crisis, y la LRS esquiva las dos colas. Varios días "
        "extremos anteriores a 1950 dependen de cierres de ^GSPC que R14 marca como probables erratas (sección 5).\n"
        f"- **Fuera de muestra** ({OOS_FROM} → {end.date()}, UPRO real): LRS 3x "
        f"{pc(oos_t.loc['LRS 3x', 'CAGR'])}/año frente a {pc(oos_t.loc['1x', 'CAGR'])} del 1x y "
        f"{pc(oos_t.loc['3x', 'CAGR'])} del 3x; diferencia de crecimiento log frente al 1x "
        f"{pc(oos_rel.loc['LRS 3x', 'Δ_crec_log_anual'], 1, True)}/año con un intervalo del 90% de "
        f"{pc(oos_rel.loc['LRS 3x', 'IC90_bajo'], 1, True)} a {pc(oos_rel.loc['LRS 3x', 'IC90_alto'], 1, True)}: "
        "positiva pero no significativa, y por detrás del 3x sin reglas.\n"
        f"- **Ficha** (rentabilidad anual a 30 años, mediana de los mundos bootstrap / GJR / regímenes, con la "
        f"señal original): 1x {' / '.join(pc(c30.loc[('1x', w), 'CAGR_mediana']) for w in wl3)}; LRS 3x "
        f"{' / '.join(pc(c30.loc[('LRS 3x', w), 'CAGR_mediana']) for w in wl3)} (P5 entre "
        f"{pc(min(c30.loc[('LRS 3x', w), 'CAGR_P5'] for w in wl3))} y {pc(max(c30.loc[('LRS 3x', w), 'CAGR_P5'] for w in wl3))}); "
        f"LRS-Kelly {' / '.join(pc(c30.loc[('LRS-Kelly', w), 'CAGR_mediana']) for w in wl3)}. La caída máxima típica "
        f"en 30 años de la LRS 3x es {' / '.join(pc(c30.loc[('LRS 3x', w), 'maxDD_mediana'], 0) for w in wl3)}, "
        f"frente a {' / '.join(pc(c30.loc[('1x', w), 'maxDD_mediana'], 0) for w in wl3)} del 1x.")

    # ── 1. P(ir por delante) ──
    rep.h("1. P(ir por delante del 1x) según el horizonte")
    rep.p("Para cada fecha de inicio posible (todas las sesiones) se compara la riqueza de la estrategia "
          "con la del 1x al cabo de h años; P es la fracción de fechas en que la estrategia acaba por "
          f"delante. Las ventanas se solapan: con {num(yrs_hist)} años de historia sólo hay "
          f"{', '.join(str(int(yrs_hist // h)) for h in HORIZONS[:-1])} y {int(yrs_hist // 20)} ventanas "
          "independientes de 1, 3, 5, 10 y 20 años. El intervalo de Wilson del 90% usa ese número como tamaño "
          "muestral: es una aproximación, ni exacta ni seguro conservadora. Para el 1x se da P(ir por delante "
          "de las letras del Tesoro). Δ_log es la diferencia anualizada de crecimiento logarítmico frente a "
          "la referencia (mediana y percentil 10 de las ventanas). Las tres últimas columnas son los mundos.")
    rep.p("**Misma regla en la historia y en los mundos.** En los mundos, la señal de la LRS se calcula, como "
          "en la historia, sobre el índice de rentabilidad total simulado; las estrategias reproducen exactamente "
          "r13.mc_strategies (se comprueba en cada mundo al ejecutar: diferencia máxima "
          f"{max(wmeta['check'].values()):.0e}). Las filas «(señal de precio)» son la LRS 3x y la LRS-Kelly con la "
          "señal sobre un índice de precio construido así: en el bootstrap, cada bloque remuestrea a la vez la "
          "rentabilidad total y la de ^GSPC del mismo día, de modo que lleva el dividendo de su época; en los "
          "mundos GJR y de regímenes, que sólo generan rentabilidad total, el precio es (1 + u)/(1 + dv) − 1, con "
          f"dv = {pc(wmeta['dv'] * 252, 2)}/año, la diferencia media entre rentabilidad total y de precio en toda la "
          "muestra.")
    t = AH.join(aw).rename(columns={"Δ_mediana": "Δ_log_mediana", "Δ_P10": "Δ_log_P10"})
    show(rep, t[["P_historia", "IC90_bajo", "IC90_alto", "independientes", "ventanas", "Δ_log_mediana",
                 "Δ_log_P10", "bootstrap", "GJR", "regímenes"]],
              pct={"P_historia", "IC90_bajo", "IC90_alto", "Δ_log_mediana", "Δ_log_P10", "bootstrap", "GJR",
                   "regímenes"}, digits=0)
    rep.p(f"Las reglas sin parámetros estimados también pueden evaluarse desde {start_lrs.date()}, con la "
          "Gran Depresión dentro:")
    show(rep, ah28.set_index(["estrategia", "años"])[["P_historia", "IC90_bajo", "IC90_alto", "independientes",
                                                      "Δ_mediana", "Δ_P10"]]
              .rename(columns={"Δ_mediana": "Δ_log_mediana", "Δ_P10": "Δ_log_P10"}),
              pct={"P_historia", "IC90_bajo", "IC90_alto", "Δ_log_mediana", "Δ_log_P10"}, digits=0)
    rep.p(f"En los mundos se usan las {N_WORLD:,} historias de cada mundo".replace(",", ".") +
          " y todos los inicios mensuales dentro de sus 30 años evaluados (tras 10 de calentamiento). Las "
          "ventanas de una misma historia también se solapan, pero las historias son independientes: el "
          "error típico de cada P es inferior a un punto.")
    rep.h("Señal de precio frente a la original (rentabilidad total)", 3)
    sgt = sgw.copy()
    sgt.loc["historia 1938–2026"] = pd.Series({"Δ_log_anual_media": sg["anual"],
                                               "Δ_CAGR_mediana": hist.loc[px3, "CAGR"] - hist.loc["LRS 3x", "CAGR"],
                                               "P(precio > original)": np.nan})
    sgt.index.name = "fuente"
    rep.p("Diferencia «precio − original» de crecimiento log anual entre la LRS 3x con señal de precio y la "
          "original en las mismas historias (en la historia, Δ_CAGR es la diferencia de CAGR, una sola "
          "trayectoria):")
    show(rep, sgt, pct={"Δ_log_anual_media", "Δ_CAGR_mediana", "P(precio > original)"}, digits=2)
    rep.p(f"En la historia, la señal de precio crece {num(sg['anual'] * 100, 2)} puntos al año más que la original. "
          f"Un solo día, el {sg['día'].strftime('%d-%m-%Y')}, aporta {num(sg['aporte_día'], 2)} de log "
          f"({pc(sg['cuota_día'], 0)} del total): el índice de precio cruzó su media el "
          f"{pd.Timestamp(cross[c_px]).strftime('%d-%m')} ({pc(x87.loc[cross[c_px], c_px], 2)}) y el de rentabilidad "
          f"total el {pd.Timestamp(cross[c_tr]).strftime('%d-%m')} (la víspera estaba "
          f"{pc(x87[c_tr].shift(1).loc[cross[c_tr]], 2, True)} sobre su media). Con la orden al cierre siguiente, la "
          f"de precio salió al cierre del {nxt(cross[c_px]).strftime('%d-%m')} y la original al del "
          f"{nxt(cross[c_tr]).strftime('%d-%m')}: la original cobró 3 × {pc(d['u'].loc[sg['día']], 1)} "
          f"({pc(R['LRS 3x'].loc[sg['día']], 1)} en un día). Sin ese día, la diferencia es de "
          f"{num(sg['anual_sin_día'] * 100, 2)} puntos al año (en los mundos, "
          f"{' / '.join(num(v * 100, 2) for v in sgw.loc[wl3, 'Δ_log_anual_media'])}). El mecanismo (el índice de "
          "precio queda por debajo de su media antes que el de rentabilidad total, porque no incluye el "
          "dividendo) es sistemático; su tamaño en la historia, no.")
    fig_ahead(ah, aw, rep)
    rep.p(
        "**Lectura.** (1) Con la misma regla original en los dos lados, la LRS 3x va por delante del 1x a 10 años "
        f"en el {pc(P('LRS 3x', 10).P_historia, 0)} de las fechas de la historia (intervalo aproximado "
        f"{pc(P('LRS 3x', 10).IC90_bajo, 0)}–{pc(P('LRS 3x', 10).IC90_alto, 0)}) y en el {wtxt('LRS 3x', 10)} de las "
        f"historias de los mundos, "
        + ("los tres dentro de ese intervalo" if P('LRS 3x', 10).IC90_bajo <= wmin('LRS 3x', 10)
           and wmax('LRS 3x', 10) <= P('LRS 3x', 10).IC90_alto else "no todos dentro de ese intervalo")
        + f"; a 20 años, {pc(P('LRS 3x', 20).P_historia, 0)} frente a {wtxt('LRS 3x', 20)}. Con la señal de precio la "
        f"historia sube al {pc(P(px3, 10).P_historia, 0)} ({pc(P(px3, 10).IC90_bajo, 0)}–{pc(P(px3, 10).IC90_alto, 0)}), "
        f"{num((P(px3, 10).P_historia - wmax(px3, 10)) * 100, 0)} puntos sobre el mejor mundo, sobre todo por el 19 de "
        "octubre de 1987. Además, la regla se eligió mirando esta misma historia. "
        f"(2) En el mundo sin tendencias (GJR), la LRS 3x va por delante a 10 años en el "
        f"{pc(Wp('LRS 3x', 10, 'GJR'), 0)} de las historias: su ventaja depende de que haya rachas persistentes. El "
        "bootstrap, que las conserva sólo en parte (bloques medios de un año), la sitúa en el "
        f"{pc(Wp('LRS 3x', 10, 'bootstrap'), 0)} a 10 años. En el GJR, la señal de precio "
        f"{'resta' if Wp(px3, 10, 'GJR') < Wp('LRS 3x', 10, 'GJR') else 'no resta'} ({pc(Wp(px3, 10, 'GJR'), 0)} "
        f"frente a {pc(Wp('LRS 3x', 10, 'GJR'), 0)}): sin tendencias, salir antes sólo cuesta. (3) El 3x no mejora "
        "con el plazo: su ventaja mediana se diluye "
        f"({pc(P('3x', 1)['Δ_mediana'], 1, True)}/año a 1 año, {pc(P('3x', 20)['Δ_mediana'], 1, True)} a 20) y su "
        f"percentil 10 a 20 años sigue siendo {pc(P('3x', 20)['Δ_P10'], 1, True)}/año. (4) Las estrategias Kelly "
        f"están cerca del 50% a corto plazo y suben despacio ({pc(P('LRS-Kelly', 10).P_historia, 0)} a 10 años en "
        f"la historia, {pc(Wp('LRS-Kelly', 10, 'bootstrap'), 0)} en el bootstrap). Sólo el mundo de regímenes, que "
        "tiene medias distintas en calma y en crisis e innovaciones normales, les da una ventaja clara "
        f"({pc(Wp('LRS-Kelly', 10, 'regímenes'), 0)} y {pc(Wp('½ Kelly GJR', 10, 'regímenes'), 0)} a 10 años); ese "
        "mundo lleva incorporado lo que la gestión por volatilidad necesita para ganar. (5) Incluso el 1x queda "
        f"por detrás de las letras en el {pc(1 - P('1x', 10).P_historia, 0)} de las ventanas de 10 años.")

    # ── 2. años por detrás ──
    rep.h("2. Años por detrás del 1x")
    rep.p("Riqueza relativa = riqueza de la estrategia / riqueza del 1x desde el inicio de la muestra. La "
          "caída relativa máxima es su mayor caída desde un máximo previo. La racha es el mayor número de "
          "años seguidos sin superar el máximo relativo anterior; si sigue abierta al final está censurada y "
          "se marca «sigue». «Tiempo >20% bajo el máximo» es la fracción de sesiones en que la estrategia "
          "está más de un 20% por debajo de su mejor momento relativo.")
    show(rep, bh, pct={"caída_relativa_máx", "tiempo_>20%_bajo_máx"}, digits=2)
    rep.p("Un máximo relativo puede ser mecánico: si la estrategia está fuera en el mínimo del mercado, la "
          "riqueza relativa toca techo ese día. La tabla siguiente, añadida tras la revisión, lee la riqueza "
          "relativa sólo los días en que el 1x marca un máximo histórico y la mantiene hasta el siguiente:")
    show(rep, bh_hi, pct={"caída_relativa_máx"}, digits=2)
    rep.p(f"Desde {start_lrs.date()} (con 1929–1938):")
    show(rep, bh28, pct={"caída_relativa_máx", "tiempo_>20%_bajo_máx"}, digits=2)
    rep.p(f"En 30 años de cada mundo sintético (el percentil 10 de la caída relativa es el caso malo) y, como "
          f"comparación, en las ventanas históricas de 30 años desde 1938 (inicios mensuales; sólo "
          f"{int(yrs_hist // 30)} independientes). En los mundos, las filas sin «(señal de precio)» usan la "
          "señal original, como en la historia:")
    show(rep, bw, pct={"caída_relativa_mediana", "caída_relativa_P10", "P(racha ≥ 5 años)",
                       "P(racha ≥ 10 años)", "P(racha abierta al final)"}, digits=1)
    fig_relative(df, dates, rep)
    rep.p(
        "**Lectura.** Ir por delante al final no evita pasar años perdiendo terreno. La LRS 3x multiplicó "
        f"la riqueza del 1x por {num(bh.loc['LRS 3x', 'riqueza_relativa_final'], 0)} desde 1938 y aun así estuvo "
        f"{num(bh.loc['LRS 3x', 'racha_más_larga_años'])} años seguidos sin superar su mejor momento relativo "
        f"y un {pc(bh.loc['LRS 3x', 'tiempo_>20%_bajo_máx'], 0)} del tiempo más de un 20% por debajo de él. "
        "En los mundos, con la misma señal original, la mediana de 30 años da una racha de "
        f"{bwt('LRS 3x', 'racha_mediana_años', num)} años y una probabilidad de racha de al menos 10 años de "
        f"{bwt('LRS 3x', 'P(racha ≥ 10 años)', lambda v: pc(v, 0))}. Las ventanas históricas de 30 años dan una "
        f"caída relativa mediana de {pc(bw.loc[('LRS 3x', HW), 'caída_relativa_mediana'], 0)} y una racha mediana de "
        f"{num(bw.loc[('LRS 3x', HW), 'racha_mediana_años'])} años (mundos "
        f"{bwt('LRS 3x', 'caída_relativa_mediana', lambda v: pc(v, 0))}). Con la señal de precio, la misma historia da "
        f"{pc(bw.loc[(px3, HW), 'caída_relativa_mediana'], 0)} y {num(bw.loc[(px3, HW), 'racha_mediana_años'])} años "
        f"(mundos {bwt(px3, 'caída_relativa_mediana', lambda v: pc(v, 0))} y {bwt(px3, 'racha_mediana_años', num)} "
        f"años). En toda la muestra, la racha más larga de la original dura "
        f"{num(bh.loc['LRS 3x', 'racha_más_larga_años'])} años ({bh.loc['LRS 3x', 'racha_desde']} → "
        f"{bh.loc['LRS 3x', 'racha_hasta']}) y la de la señal de precio {num(bh.loc[px3, 'racha_más_larga_años'])}; "
        f"en las ventanas de 30 años, el percentil 90 de la racha es {num(bw.loc[('LRS 3x', HW), 'racha_P90_años'])} "
        f"años con la original y {num(bw.loc[(px3, HW), 'racha_P90_años'])} con la señal de precio. "
        + ("La historia parece más benigna con la señal de precio, sobre todo por el 19 de octubre de 1987."
           if bh.loc[px3, 'racha_más_larga_años'] < bh.loc['LRS 3x', 'racha_más_larga_años'] else "")
        + "\n\n"
        "**Las estrategias Kelly.** Su máximo relativo es del "
        f"{peak_k.strftime('%d-%m-%Y')}" + (f", el mínimo del S&P en la crisis {sp_trough[0]}" if len(sp_trough) else "") +
        ", con las dos estrategias fuera del mercado: es en parte un máximo mecánico. La riqueza de la "
        f"LRS-Kelly respecto al 1x pasó de {num(rwk.max())} entonces a {num(rwk.loc['1990':'2005'].min())} en "
        f"{rwk.loc['1990':'2005'].idxmin().year} y es {num(rwk.iloc[-1])} hoy. Leída sólo en los máximos del 1x, su "
        f"caída relativa máxima es del {pc(-bh_hi.loc['LRS-Kelly', 'caída_relativa_máx'], 0)} desde el "
        f"{bh_hi.loc['LRS-Kelly', 'máximo_relativo']} y su racha más larga, de "
        f"{num(bh_hi.loc['LRS-Kelly', 'racha_más_larga_años'])} años ({bh_hi.loc['LRS-Kelly', 'racha_desde']} → "
        f"{bh_hi.loc['LRS-Kelly', 'racha_hasta']}); la del ½ Kelly GJR, de "
        f"{num(bh_hi.loc['½ Kelly GJR', 'racha_más_larga_años'])} años. "
        + ("Así medida, la racha de la LRS-Kelly es incluso más larga: el máximo mecánico de 1974 no explica los "
           "años por detrás." if bh_hi.loc['LRS-Kelly', 'racha_más_larga_años'] >= bh.loc['LRS-Kelly', 'racha_más_larga_años']
           else "Así medida, la racha de la LRS-Kelly es más corta: parte de los años por detrás se debe al máximo "
           "mecánico de 1974.")
        + f" La de la LRS 3x, medida igual, {'se alarga' if bh_hi.loc['LRS 3x', 'racha_más_larga_años'] > bh.loc['LRS 3x', 'racha_más_larga_años'] else 'se acorta'} de "
        f"{num(bh.loc['LRS 3x', 'racha_más_larga_años'])} a {num(bh_hi.loc['LRS 3x', 'racha_más_larga_años'])} años "
        f"({bh_hi.loc['LRS 3x', 'racha_desde']} → {bh_hi.loc['LRS 3x', 'racha_hasta']}).\n\n"
        "¿De dónde sale la pérdida? La fracción de Kelly de R13 usa la prima μ̂ − f̂ (media histórica "
        "expansiva de la rentabilidad total menos la financiación, u − f) menos s. "
        + (f"Desde 1938, μ̂ − f̂ − s fue positiva en todas las sesiones (mínimo: {pc(kd['prima_mín'])}): la fracción de Kelly "
           "nunca se anuló por la prima, ni siquiera con los tipos altos, porque μ̂ − f̂ promedia toda la historia. "
           f"En 1980–84, con f = {pc(kd['f'])} de media, μ̂ − f̂ fue {pc(kd['prima'])} y la exposición media, "
           f"{num(kd['exp_lrsk'], 2)}x (LRS-Kelly) y {num(kd['exp_half'], 2)}x (½ Kelly GJR). "
           if kd["sesiones"] == 0 else
           f"μ̂ − f̂ − s ≤ 0 ocurrió en {kd['sesiones']} sesiones desde 1938"
           + (f", y en más de la mitad de las de {lst([str(y) for y in kd['años']])}" if kd["años"] else "")
           + f". En 1980–84 la exposición media fue de {num(kd['exp_lrsk'], 2)}x (LRS-Kelly) y "
           f"{num(kd['exp_half'], 2)}x (½ Kelly GJR), con μ̂ − f̂ = {pc(kd['prima'])} y f = {pc(kd['f'])}. ")
        + "La tabla reparte la pérdida relativa según lo que decidió la regla dos sesiones antes: prima ≤ 0 "
        "(Kelly a cero), prima > 0 con el S&P bajo su media (fuera por el filtro) y prima > 0 sobre la media "
        "(dentro, con la exposición que dé Kelly).")
    rep.h("LRS-Kelly: pérdida log relativa frente al 1x por tramos y por tipo de sesión", 3)
    show(rep, kt.drop(columns=[c for c in kt.columns if c.startswith("sesiones")]),
              pct={"log_relativo"} | {c for c in kt.columns if c.startswith("log:")}, digits=3)
    rep.p("«log» es la suma de log(1 + r_estrategia) − log(1 + r_1x) en el tramo (en tanto por uno); "
          "factor_relativo = exp(log). Los cortes de fin de 1978 y fin de 1984 se eligieron después de ver el "
          "gráfico, para aislar los años de tipos altos. Sesiones de cada tipo en el tramo máximo → hoy: "
          + "; ".join(f"{c.replace('sesiones: ', '')} {int(kt.xs('máximo → hoy', level='tramo').iloc[0][c])}"
                      for c in kt.columns if c.startswith("sesiones")) + ".")
    rep.h("½ Kelly GJR: el mismo reparto", 3)
    show(rep, kt_h.drop(columns=[c for c in kt_h.columns if c.startswith("sesiones")]),
              pct={"log_relativo"} | {c for c in kt_h.columns if c.startswith("log:")}, digits=3)
    rep.h("LRS-Kelly año a año, del máximo al mínimo relativo", 3)
    show(rep, ky, pct={"LRS-Kelly", "1x", "relativa", "sesiones μ̂−f̂−s ≤ 0", "sesiones sobre la media", "f", "μ̂ − f̂",
                       "σ̂ GJR"}, digits=2)
    rep.p(
        f"**Lectura.** Del máximo de {peak_k.year} al mínimo relativo de "
        f"{pd.Timestamp(bh.loc['LRS-Kelly', 'mínimo_relativo']).year}, la LRS-Kelly perdió "
        f"{num(-kpt['log_relativo'], 3)} de log frente al 1x (factor {num(kpt['factor_relativo'], 3)}). "
        + ("Ninguna sesión tuvo μ̂ − f̂ − s ≤ 0, así que toda la pérdida se produjo con prima positiva. "
           if kd["sesiones"] == 0 else
           f"Las sesiones con μ̂ − f̂ − s ≤ 0 aportan {num(kpt[neg_c], 3)} "
           f"({pc(kpt[neg_c] / kpt['log_relativo'], 0)}); hasta hoy, {num(kpe[neg_c], 3)} de "
           f"{num(kpe['log_relativo'], 3)} ({pc(kpe[neg_c] / kpe['log_relativo'], 0)}). En el ½ Kelly GJR, "
           f"{num(kht[neg_c], 3)} de {num(kht['log_relativo'], 3)}. ")
        + f"Por tramos, 1974–78 deja un factor {num(kseg.iloc[0]['factor_relativo'], 3)}, 1979–84 (los años de "
        f"tipos altos) {num(kseg.iloc[1]['factor_relativo'], 3)} y "
        f"1985–{pd.Timestamp(bh.loc['LRS-Kelly', 'mínimo_relativo']).year} {num(kseg.iloc[2]['factor_relativo'], 3)}. "
        "Una parte de la pérdida se produce fuera por el filtro de tendencia mientras el mercado sube, con "
        f"entradas tardías y salidas en falso ({num(kpt['log: prima > 0, bajo la media'], 3)}). Otra, dentro "
        "(sobre la media), con menos de 1x en años alcistas o más de 1x en años malos "
        f"({num(kpt['log: prima > 0, sobre la media'], 3)}). Dentro, la exposición la mueve sobre todo la σ̂² del "
        f"GJR: la prima estimada μ̂ − f̂ se quedó entre {pc(kd['prima_rango'][0])} y {pc(kd['prima_rango'][1])} "
        f"de {peak_k.year} a {pd.Timestamp(bh.loc['LRS-Kelly', 'mínimo_relativo']).year}. Los seis años que más "
        "restan: "
        + ", ".join(f"{y} ({pc(r_['relativa'], 0, True)}, exposición media {num(r_['exposición media (x)'], 2)}x)"
                    for y, r_ in k_bad.iterrows()) + "."
        + ("" if kd["sesiones"] == 0 else
           " Si en las sesiones con μ̂ − f̂ − s ≤ 0 hubiera empatado con el 1x, el factor hasta "
           f"{pd.Timestamp(bh.loc['LRS-Kelly', 'mínimo_relativo']).year} habría sido "
           f"{num(np.exp(kpt['log_relativo'] - kpt[neg_c]), 2)} en vez de {num(kpt['factor_relativo'], 2)}."))

    # ── 3. abandono ──
    n20 = len(np.arange(0, len(df) - 20 * 252 + 1, STRIDE))
    rep.h("3. Una regla de abandono realista")
    rep.p(f"Regla principal, fijada antes de mirar resultados: **{MAIN_RULE}**. El inversor compara cada "
          "día su cartera con el 1x en los últimos 3 años (o desde que empezó, si lleva menos). Si va más de "
          "un 20% por detrás (riqueza relativa < 0,8), vende al cierre siguiente y se queda en el 1x para "
          "siempre. Tres años es el plazo tras el que los planes de pensiones suelen despedir a un gestor "
          "que rinde por debajo de su referencia (Goyal y Wahal 2008). Más abajo se muestran enteras la rejilla "
          "(1, 3 o 5 años × 10, 20 o 30%) y una regla de pánico por caída propia ≥ 50%. Historia: un inversor "
          f"nuevo cada 21 sesiones desde 1938 (a 20 años son {n20} inicios, pero sólo {int(yrs_hist // 20)} "
          "ventanas independientes). Mundos: un inversor por historia, que empieza al final del calentamiento.")
    rep.p("«Coste» es la rentabilidad anual manteniendo menos la rentabilidad con abandono, en puntos por año; "
          "si es positivo, abandonar sale caro. Quien no abandona tiene coste exactamente 0, así que se da el "
          "coste mediano entre QUIENES ABANDONAN y la media sobre todos los inversores (que es P × el coste "
          "medio de quien abandona). Las rentabilidades son medianas entre todos los inversores.")
    cols = ["P_abandona", "años_hasta_abandono", "CAGR_mantener", "CAGR_con_abandono", "CAGR_1x",
            "coste_si_abandona", "coste_medio_todos", "P_detrás_mantener", "P_detrás_con_abandono"]
    pcts = {"P_abandona", "CAGR_mantener", "CAGR_con_abandono", "CAGR_1x", "coste_si_abandona",
            "coste_medio_todos", "P_detrás_mantener", "P_detrás_con_abandono"}
    main = [ab_h[y][ab_h[y].regla == MAIN_RULE].assign(fuente="historia", años=y) for y in (10, 20)]
    main += [ab_w[y][ab_w[y].regla == MAIN_RULE].rename(columns={"mundo": "fuente"}).assign(años=y)
             for y in (10, 20, r13.YEARS)]
    main = pd.concat(main).set_index(["estrategia", "fuente", "años"]).sort_index()
    show(rep, main[cols], pct=pcts, digits=1)
    kel_cost = [mh(20).loc[k, "coste_si_abandona"] for k in ("LRS-Kelly", "½ Kelly GJR")]
    kel_mean = [mh(20).loc[k, "coste_medio_todos"] for k in ("LRS-Kelly", "½ Kelly GJR")]
    rep.p(
        "**Lectura.** La desviación frente al 1x de cualquier estrategia con un 3x dentro es tan grande que "
        "quedarse un 20% por detrás en tres años es casi seguro en algún momento. En la historia, a 20 años, "
        f"abandonaría la LRS 3x el {pc(mh(20).loc['LRS 3x', 'P_abandona'], 0)} de los inversores; en los mundos, "
        f"a 30 años, el {' / '.join(pc(mw(r13.YEARS, w).loc['LRS 3x', 'P_abandona'], 1) for w in wl3)}, casi siempre "
        f"en los primeros años (mediana {' / '.join(num(mw(r13.YEARS, w).loc['LRS 3x', 'años_hasta_abandono']) for w in wl3)} "
        "años). Como se abandona justo después de ir mal, la pérdida relativa queda fijada: con abandono, la "
        "probabilidad de acabar por detrás del 1x a 30 años sube del "
        f"{pc(mw(r13.YEARS, 'bootstrap').loc['LRS 3x', 'P_detrás_mantener'], 0)} al "
        f"{pc(mw(r13.YEARS, 'bootstrap').loc['LRS 3x', 'P_detrás_con_abandono'], 0)} en el mundo bootstrap (historia a 20 "
        f"años: del {pc(mh(20).loc['LRS 3x', 'P_detrás_mantener'], 0)} al {pc(mh(20).loc['LRS 3x', 'P_detrás_con_abandono'], 0)}). "
        f"A quien abandona la LRS 3x le cuesta, en la mediana, {num(mh(20).loc['LRS 3x', 'coste_si_abandona'] * 100)} "
        f"puntos al año en la historia (20 años) y {' / '.join(num(mw(r13.YEARS, w).loc['LRS 3x', 'coste_si_abandona'] * 100) for w in wl3)} "
        f"en los mundos (30 años); en media sobre todos, {num(mh(20).loc['LRS 3x', 'coste_medio_todos'] * 100)} y "
        f"{' / '.join(num(mw(r13.YEARS, w).loc['LRS 3x', 'coste_medio_todos'] * 100) for w in wl3)}. En las estrategias "
        f"Kelly, el coste mediano de quien abandona es pequeño en la historia a 20 años ({num(kel_cost[0] * 100)} puntos "
        f"en la LRS-Kelly y {num(kel_cost[1] * 100)} en el ½ Kelly GJR; {num(kel_mean[0] * 100)} y "
        f"{num(kel_mean[1] * 100)} en media sobre todos), pero no en todos los mundos: a 30 años es de "
        f"{' / '.join(num(mw(r13.YEARS, w).loc['LRS-Kelly', 'coste_si_abandona'] * 100) for w in wl3)} (LRS-Kelly) y "
        f"{' / '.join(num(mw(r13.YEARS, w).loc['½ Kelly GJR', 'coste_si_abandona'] * 100) for w in wl3)} puntos (½ Kelly "
        "GJR). Es grande donde su ventaja lo es, en el mundo de regímenes. En el 3x abandonar apenas cuesta y en "
        "algunos mundos incluso ayuda (coste negativo): el 3x "
        "tiene poca ventaja que perder. Las reglas más pacientes no evitan el problema, sólo lo reparten. Con "
        f"«5 años, −30%», abandona la LRS 3x el {pc(gh(20, '5 años, −30% frente al 1x', 'LRS 3x').P_abandona, 0)} de los "
        f"inversores históricos a 20 años, con un coste mediano de "
        f"{num(gh(20, '5 años, −30% frente al 1x', 'LRS 3x').coste_si_abandona * 100)} puntos para quien lo hace. La "
        f"regla de pánico (caída propia ≥ 50%) salta para el {pc(gh(20, 'caída propia ≥ 50%', 'LRS 3x').P_abandona, 0)}, "
        f"con {num(gh(20, 'caída propia ≥ 50%', 'LRS 3x').coste_si_abandona * 100)} puntos, y casi siempre en los "
        "mundos, cuya caída máxima típica en 30 años supera el 70%.")
    rep.h("Rejilla completa de reglas de abandono", 3)
    rep.p("Filas: regla. Columnas: probabilidad de abandonar (P) y coste mediano, en puntos al año, entre "
          "quienes abandonan («coste si abandona»), para cada estrategia. Una celda de coste vacía significa "
          "que nadie abandona.")
    for src, tab in ([("historia, 10 años", ab_h[10]), ("historia, 20 años", ab_h[20])] +
                     [(f"mundo {wl}, {r13.YEARS} años", ab_w[r13.YEARS][ab_w[r13.YEARS].mundo == wl])
                      for _, wl in WORLDS]):
        rep.h(src, 4)
        pv = tab.pivot_table(index="regla", columns="estrategia", values=["P_abandona", "coste_si_abandona"],
                             sort=False, dropna=False)
        pv = pv.reindex([r[0] for r in ABANDON_RULES])
        pv.columns = [f"{s}: {'P' if m == 'P_abandona' else 'coste si abandona'}" for m, s in pv.columns]
        pv = pv[[f"{s}: {m}" for s in OTHERS for m in ("P", "coste si abandona")]]
        show(rep, pv, pct=set(pv.columns), digits=1)

    # ── 4. crisis ──
    rep.h("4. Crisis: caídas del S&P 500 de al menos el 19%")
    rep.p("Regla, fijada de antemano y aplicada al índice de PRECIO con cierres diarios desde 1928:\n\n"
          "- En fase alcista se sigue el máximo. Si el cierre cae al menos un 19% por debajo, empieza una crisis.\n"
          "- La crisis termina en su mínimo cuando el índice recupera la mitad de lo perdido entre máximo y "
          "mínimo. Desde ese cierre se vuelve a seguir el máximo.\n\n"
          "Así, repuntes como el de septiembre de 2001 a enero de 2002 (+21%) o el de noviembre de 2008 a enero "
          "de 2009 (+24%) no parten en dos 2000–02 ni 2007–09, como en las cronologías habituales. Pero la "
          "regla no reproduce todas: parte en dos 1929–32 y 1937–38, en tres el tramo 1938–42, y sitúa el "
          "mínimo de 2022 en junio (−23,6%) y no en octubre (−25,4%), porque el repunte de agosto de 2022 "
          "recuperó más de la mitad de lo perdido. No se cambió después de verlo; la sección 6 da 2022 con el "
          "mínimo de octubre. Las fechas se deciden con la serie completa: es una descripción, no una señal. "
          "Las rentabilidades de las estrategias incluyen dividendos y costes. Antes de 1938 las estrategias "
          "Kelly no existen (aún no tienen 10 años de datos). Las dos últimas columnas cuentan las sesiones en "
          "que la LRS 3x cambió de posición entre el máximo y el mínimo o la recuperación: muchas indican "
          "entradas y salidas en falso.")
    show(rep, head, pct={"caída_precio"}, digits=1)
    rep.h("Del máximo al mínimo del S&P 500", 3)
    show(rep, cr_a, pct=set(FOCUS), digits=1)
    rep.h("Del máximo a la recuperación del máximo del S&P 500 (precio)", 3)
    show(rep, cr_b, pct=set(FOCUS), digits=1)
    rep.h("Años hasta recuperar la riqueza del día del máximo", 3)
    rep.p("Desde el cierre del máximo del S&P, años hasta que la riqueza de cada estrategia vuelve a su nivel "
          "de ese día, buscando a partir del mínimo del S&P. Un 0 significa que en el mínimo ya estaba por "
          "encima (no perdió en la crisis).")
    show(rep, cr_c, digits=1)
    summ = {}
    for k in ["1x"] + OTHERS:
        a_, b_ = cr_a.loc[since38, k], cr_b.loc[rec_ok, k]
        summ[k] = {"mediana máx→mín": a_.median(), "peor máx→mín": a_.min(),
                   "mediana máx→recuperación": b_.median(),
                   "pierde menos que el 1x (máx→mín)": int((a_ > cr_a.loc[since38, "1x"]).sum())
                   if k != "1x" else np.nan,
                   "por delante del 1x (máx→recuperación)":
                       int((b_ > cr_b.loc[rec_ok, "1x"]).sum()) if k != "1x" else np.nan,
                   "años medianos hasta recuperar": cr_c.loc[since38, k].median()}
    rep.p(f"Resumen de las {len(since38)} crisis desde 1938 ({len(rec_ok)} con el máximo ya recuperado):")
    show(rep, pd.DataFrame(summ).T, pct={"mediana máx→mín", "peor máx→mín", "mediana máx→recuperación"},
              digits=1)
    rep.h("Crisis largas y cortas: rejilla de duración", 3)
    rep.p("Para no clasificar las crisis a mano, se separan por los meses del máximo al mínimo del S&P con "
          f"cada umbral de la rejilla {', '.join(str(x) for x in CRISIS_MONTHS)} meses. La rejilla se fijó tras la "
          "revisión, sin probar otras, y se muestra entera. La LRS 3x «pierde menos» si su rentabilidad del "
          "máximo al mínimo supera la del 1x; va «por delante» si también la supera del máximo a la "
          "recuperación. La última columna lista las crisis ya recuperadas en las que acabó por detrás.")
    show(rep, cd, digits=0)
    fig_crises(cr_a, cr_b, head, rep)
    hf, af, bf, _ = crisis_tables(d_fix["px"].loc[:end], R_fix, E_fix)
    moved = [(c, head.loc[c, "pico"], hf.index[i]) for c in since38
             for i in [int(np.argmin([abs((pd.Timestamp(x) - pd.Timestamp(head.loc[c, "valle"])).days)
                                     for x in hf["valle"]]))]
             if hf.iloc[i]["pico"] != head.loc[c, "pico"]]
    sens_cr = ""
    for c, p0, cf in moved:
        sens_cr += (f"Con los siete cierres dudosos corregidos (sección 5), el máximo de {c} pasa del "
                    f"{pd.Timestamp(p0).strftime('%d-%m-%Y')}, que es uno de ellos, al "
                    f"{pd.Timestamp(hf.loc[cf, 'pico']).strftime('%d-%m-%Y')} (caída {pc(hf.loc[cf, 'caída_precio'], 1)} "
                    f"en vez de {pc(head.loc[c, 'caída_precio'], 1)}). Del máximo al mínimo, la LRS 3x pierde un "
                    f"{pc(-af.loc[cf, 'LRS 3x'], 1)} en vez de un {pc(-cr_a.loc[c, 'LRS 3x'], 1)} (1x: {pc(af.loc[cf, '1x'], 1)}), y "
                    f"del máximo a la recuperación, {pc(bf.loc[cf, 'LRS 3x'], 1, True)} frente a {pc(bf.loc[cf, '1x'], 1, True)}. ")
    sens_cr += ("Las demás crisis no cambian de fechas." if moved else
                "Con los siete cierres dudosos corregidos, ninguna crisis cambia de fechas.")
    rep.p(sens_cr)
    rep.p(
        "**Lectura.** La LRS 3x no es un seguro contra las caídas. Del máximo al mínimo perdió menos que el 1x "
        f"sólo en {len(less)} de {len(since38)} crisis ({lst(less)}). En las caídas moderadas sigue con 3x hasta "
        "que el índice cruza su media, y el primer tramo de la caída, triplicado, cuesta más que toda la caída "
        f"del 1x. Del máximo a la recuperación acabó por delante en {len(ahead_rec)} ({lst(ahead_rec)}) y por "
        f"detrás en {len(behind_rec)} ({lst(behind_rec)}). La duración ayuda a ordenar, pero no separa limpiamente: "
        f"con al menos {CL} meses del máximo al mínimo, va por delante en {len(long_ahead)} de {len(long_rec)}, y con "
        f"menos de {CL} en {len(short_ahead)} de {len(short_rec)}. Las excepciones largas importan. En 1973–74, "
        f"{sup_7374} ({num(head.loc['1973–74', 'meses_pico_valle'])} meses, "
        f"{pc(head.loc['1973–74', 'caída_precio'], 0)}), perdió menos que el 1x hasta el mínimo; pero en los "
        f"{num(head.loc['1973–74', 'años_hasta_recuperar'])} años hasta recuperar el máximo, que incluyen el bajista de "
        f"1976–78, cambió de posición {int(head.loc['1973–74', 'cambios_LRS_máx→recuperación'])} veces y acabó en "
        f"{pc(cr_b.loc['1973–74', 'LRS 3x'], 0, True)} frente a {pc(cr_b.loc['1973–74', '1x'], 0, True)} del "
        f"1x. En 1976–78 ({num(head.loc['1976–78', 'meses_pico_valle'])} meses) acabó en "
        f"{pc(cr_b.loc['1976–78', 'LRS 3x'], 0, True)} frente a {pc(cr_b.loc['1976–78', '1x'], 0, True)}. Las ganancias "
        "grandes llegan en los bajistas largos con una tendencia limpia (2000–02 y 2007–09), en los que evita "
        "la mayor parte de la caída y vuelve a entrar con 3x en la subida. Las pérdidas se concentran en las "
        f"crisis de menos de {CL} meses, en las que acabó por detrás en {len(short_behind)} de {len(short_rec)} "
        f"({lst(short_behind)}). Unas son caídas rápidas con rebote en V, en las que vende tarde y recompra "
        "tarde; otras, bajistas con repuntes que la hacen entrar y salir. 2022 es de este segundo tipo, aunque "
        f"la regla lo cuente como corto: {num(head.loc['2022', 'meses_pico_valle'])} meses hasta el mínimo de la "
        "regla (en junio) y "
        f"{num(m22)} hasta el de octubre, {num(head.loc['2022', 'años_hasta_recuperar'])} años hasta recuperar y "
        f"{int(head.loc['2022', 'cambios_LRS_máx→recuperación'])} cambios de posición. Antes de 1938 la diferencia es "
        f"enorme a su favor (1930–32: {pc(cr_a.loc['1930–32', 'LRS 3x'], 0)} frente a {pc(cr_a.loc['1930–32', '1x'], 0)} "
        f"del máximo al mínimo). La LRS-Kelly pierde menos que el 1x en "
        f"{int(summ['LRS-Kelly']['pierde menos que el 1x (máx→mín)'])} de {len(since38)} (mediana "
        f"{pc(summ['LRS-Kelly']['mediana máx→mín'], 0)}), porque su exposición ya es baja cuando la volatilidad "
        f"sube. El 3x sin reglas pierde más que el 1x en todas (mediana {pc(summ['3x']['mediana máx→mín'], 0)}) y sólo "
        f"acaba por delante en {int(summ['3x']['por delante del 1x (máx→recuperación)'])} recuperaciones.")

    # ── 5. días extremos ──
    fmt_d = lambda ix: [t_.strftime("%d-%m-%Y") for t_ in ix]  # noqa: E731

    def own_summary(dd_: dict) -> tuple[int, int, pd.DataFrame]:
        o = dd_["propios"]
        dep = o[(o.index >= pd.Timestamp("1932-01-01")) & (o.index < pd.Timestamp("1936-01-01"))]
        return len(dep), int(dep["cierre dudoso (R14)"].sum()), o.drop(dep.index)
    n_dep, n_dep_sus, own_rest = own_summary(days)
    n_dep_f, _, own_rest_f = own_summary(days_fix)
    sus_worst = fmt_d(days["peores"].index[days["peores"]["cierre dudoso (R14)"]])
    sus_best = fmt_d(days["mejores"].index[days["mejores"]["cierre dudoso (R14)"]])
    rest_sus = fmt_d(own_rest.index[own_rest["cierre dudoso (R14)"]])
    clean_rest = fmt_d(own_rest.index[~own_rest["cierre dudoso (R14)"]])
    rep.h("5. Peores y mejores días, con la LRS dentro o fuera")
    rep.p("La exposición es la que de verdad cobra la estrategia ese día, decidida dos cierres antes (señal "
          "al cierre de t, orden al cierre de t+1). En la LRS 3x, 3 = dentro y 0 = en letras. «S&P 500» es la "
          f"rentabilidad total del día. Desde {start_lrs.date()}; la LRS-Kelly, desde 1938.")
    rep.p("**Cierres dudosos.** R14 revisó a mano siete pares de sesiones con la firma de un cierre mal escrito "
          f"en ^GSPC (un cierre fuera de línea con sus dos vecinos y CRSP sin ese movimiento): "
          f"{', '.join(pd.Timestamp(x).strftime('%d-%m-%Y') for x in SUSPECT_CLOSES)}. La base no los corrige "
          "(su regla fijada de antemano no los marca). La columna «cierre dudoso (R14)» señala la sesión de ese "
          "cierre y la siguiente, cuyas rentabilidades dependen de él. Al final de la sección, una sensibilidad "
          "los sustituye por la media geométrica de sus vecinos, como hace la base con los cuatro que sí corrige.")
    dpct = {"S&P 500", "3x", "LRS 3x", "LRS-Kelly"}
    for key, ttl in (("peores", "Los 15 peores días del S&P 500"), ("mejores", "Los 15 mejores días del S&P 500"),
                     ("propios", "Los 10 peores días de la propia LRS 3x")):
        rep.h(ttl, 3)
        t = days[key].copy()
        t["cierre dudoso (R14)"] = np.where(t["cierre dudoso (R14)"], "sí", "")
        t.index = t.index.date
        show(rep, t, pct=dpct, digits=2)
    rep.h("Agregado: días extremos frente a una sesión cualquiera", 3)
    show(rep, ag, pct={"LRS 3x dentro", "S&P medio", "3x medio", "LRS 3x medio", "LRS-Kelly medio"}, digits=2)
    rep.h("Sensibilidad: los siete cierres dudosos corregidos", 3)
    rep.p("Los 10 peores días de la LRS 3x con los cierres corregidos:")
    tf = days_fix["propios"].drop(columns=["cierre dudoso (R14)"]).copy()
    tf.index = tf.index.date
    show(rep, tf, pct=dpct, digits=2)
    sens = pd.DataFrame({"CAGR base": hist.loc[list(FOCUS), "CAGR"], "CAGR corregido": hist_fix.loc[list(FOCUS), "CAGR"],
                         "caída máx. base": hist.loc[list(FOCUS), "maxDD"],
                         "caída máx. corregida": hist_fix.loc[list(FOCUS), "maxDD"]})
    rep.p(f"Efecto en las cifras de toda la muestra desde {start_all.date()}:")
    show(rep, sens, pct=set(sens.columns), digits=2)
    rep.p(
        f"**Lectura.** Desde 1928 la LRS 3x estuvo dentro en el "
        f"{pc(ag.loc[('1928–2026', '10 peores'), 'LRS 3x dentro'], 0)} de los 10 peores días del S&P (el 19-10-1987, "
        f"{'dentro' if E['LRS 3x'].shift(LAG).loc['1987-10-19'] > 0 else 'fuera'}: la señal original cruzó su media el "
        f"viernes 16 y la orden salía al cierre del lunes; la de precio, "
        f"{'dentro' if E[px3].shift(LAG).loc['1987-10-19'] > 0 else 'fuera'}) y en el "
        f"{pc(ag.loc[('1928–2026', '20 peores'), 'LRS 3x dentro'], 0)} de los 20 peores. Pero también "
        f"estuvo fuera en {int(round((1 - ag.loc[('1928–2026', '10 mejores'), 'LRS 3x dentro']) * 10))} de los 10 mejores "
        f"y dentro sólo en el {pc(ag.loc[('1928–2026', '100 mejores'), 'LRS 3x dentro'], 0)} de los 100 mejores, frente "
        f"al {pc(ag.loc[('1928–2026', 'todas las sesiones'), 'LRS 3x dentro'], 0)} de una sesión cualquiera. Los días "
        "extremos de ambos signos se agrupan en las crisis, por debajo de la media de 200 sesiones. En los 100 "
        f"peores días desde 1938 la LRS 3x perdió de media {pc(ag.loc[('1938–2026', '100 peores'), 'LRS 3x medio'])} "
        f"(el 1x {pc(ag.loc[('1938–2026', '100 peores'), 'S&P medio'])}, el 3x {pc(ag.loc[('1938–2026', '100 peores'), '3x medio'])}), "
        f"y en los 100 mejores ganó {pc(ag.loc[('1938–2026', '100 mejores'), 'LRS 3x medio'])} (el 1x "
        f"{pc(ag.loc[('1938–2026', '100 mejores'), 'S&P medio'])}). "
        + (f"Entre los 15 peores días del S&P, {WORDS_M.get(len(sus_worst), len(sus_worst))} "
           f"{pl(len(sus_worst), 'depende', 'dependen')} de un "
           f"cierre dudoso ({lst(sus_worst)})" if sus_worst else
           "Ninguno de los 15 peores días del S&P depende de un cierre dudoso")
        + (f", y entre los 15 mejores, {WORDS_M.get(len(sus_best), len(sus_best))} ({lst(sus_best)}). " if sus_best else ". ")
        + f"De los 10 peores días de la propia LRS 3x, {n_dep} son de 1932–35, cuando la señal volvía a entrar "
        "en repuntes violentos que se deshacían"
        + (f" ({WORDS_M.get(n_dep_sus, n_dep_sus)} de ellos {pl(n_dep_sus, 'depende', 'dependen')} de un cierre dudoso)"
           if n_dep_sus else "")
        + (f". De los demás, {lst(rest_sus)} depende{'n' if len(rest_sus) > 1 else ''} de un cierre dudoso y no "
           f"es una caída real del mercado" if rest_sus else "")
        + (f"; {lst(clean_rest)} {'son caídas reales' if len(clean_rest) > 1 else 'es una caída real'} de un día "
           "con el índice sobre su media, que la regla no puede esquivar" if clean_rest else "")
        + f". Con los siete cierres corregidos, {n_dep_f} de los 10 peores días propios son de 1932–35, y los "
        f"demás son {lst(fmt_d(own_rest_f.index))}. El efecto en toda la muestra es "
        f"pequeño: la LRS 3x pasa de {pc(hist.loc['LRS 3x', 'CAGR'], 2)} a {pc(hist_fix.loc['LRS 3x', 'CAGR'], 2)} al año "
        f"y el 1x de {pc(hist.loc['1x', 'CAGR'], 2)} a {pc(hist_fix.loc['1x', 'CAGR'], 2)}.")

    # ── 6. fuera de muestra ──
    oos_y = (end - pd.Timestamp(OOS_FROM)).days / 365.25
    rep.h(f"6. El único periodo fuera de muestra: {OOS_FROM} → {end.date()}, con ETFs reales")
    rep.p(f"Tomamos como inicio la primera sesión de 2017, el primer año completo tras la publicación de Gayed y "
          f"Bilello (SSRN, marzo de 2016): son {num(oos_y)} años. En este tramo el 3x es UPRO real, con sus costes de swap reales; el 1x "
          "es el S&P 500 con dividendos y la caja son letras. Sólo la LRS 3x original es de verdad fuera de "
          "muestra. La fracción ½ de Kelly y el GJR se eligieron en R10–R12 mirando también estos años, y el "
          "sobrecoste del swap del sintético se calibra con toda la vida de UPRO.")
    show(rep, oos_t, pct={"CAGR", "vol", "maxDD", "peor_año", "tiempo_invertido"}, digits=2)
    rep.p("Diferencia de crecimiento log frente al 1x, con intervalo del 90% por bootstrap estacionario "
          f"(bloques medios de 63 sesiones, {B_BOOT} remuestreos), y riqueza relativa dentro del periodo. "
          "«inicio_racha_por_detrás» es el máximo relativo que abre la racha más larga sin superarlo:")
    show(rep, oos_rel, pct={"Δ_crec_log_anual", "IC90_bajo", "IC90_alto", "caída_relativa_máx"}, digits=2)
    rep.p("Episodios del periodo con fechas fijas: los de la regla de la sección 4, 2022 con el mínimo de "
          "octubre de la datación habitual y la caída de febrero a abril de 2025, que se quedó en el "
          f"{pc(-eps.loc[('2025 (fuera de la regla)', 'máx→mín'), 'caída_precio'])} y no entra en la regla. "
          "«cambios_LRS» cuenta las sesiones en que la LRS 3x cambió de posición entre el máximo y la "
          "recuperación:")
    show(rep, eps, pct={"caída_precio"} | set(FOCUS), digits=1)
    oos_mk = []
    for name in ("Nasdaq-100", "Semis"):
        dm = D[name]
        Rm, _ = r13.strategies(dm, r13.var_forecast(dm))
        tm = r13.stats_table({s: Rm[k] for s, k in FOCUS.items()}, dm["rf"], OOS_FROM)
        oos_mk.append(tm[["CAGR", "maxDD"]].assign(mercado=name))
    oos_mk = pd.concat(oos_mk).reset_index().rename(columns={"index": "estrategia"}).set_index(
        ["mercado", "estrategia"])
    rep.p("Como contraste, la misma regla con TQQQ y SOXL reales. No forman parte del artículo de Gayed y "
          "Bilello, y el Nasdaq-100 ya se exploró en R6, así que no son fuera de muestra en sentido estricto.")
    show(rep, oos_mk, pct={"CAGR", "maxDD"}, digits=2)
    oo = pd.DataFrame(R).loc[OOS_FROM:, list(FOCUS)].dropna()
    yr = (1 + oo).groupby(oo.index.year).prod() - 1
    yr["LRS 3x − 1x"] = yr["LRS 3x"] - yr["1x"]
    yr["LRS 3x invertida"] = (E["LRS 3x"].shift(LAG).loc[oo.index] > 0).groupby(oo.index.year).mean()
    yr.index.name = "año"
    rep.p(f"Año a año (el último, {end.year}, incompleto):")
    show(rep, yr, pct=set(yr.columns), digits=1)
    top = yr["LRS 3x − 1x"].sort_values(ascending=False)
    lrs_ep = eps["LRS 3x"].unstack("tramo")
    one_ep = eps["1x"].unstack("tramo")
    quick = [e for e in eps.index.get_level_values(0).unique() if "junio" not in e and "octubre" not in e]
    rep.p(
        f"**Lectura.** En los {num(oos_y)} años posteriores a la publicación la LRS 3x ganó "
        f"{pc(oos_t.loc['LRS 3x', 'CAGR'])} al año frente al {pc(oos_t.loc['1x', 'CAGR'])} del 1x, pero con un "
        f"intervalo que incluye el cero ({pc(oos_rel.loc['LRS 3x', 'IC90_bajo'], 1, True)} a "
        f"{pc(oos_rel.loc['LRS 3x', 'IC90_alto'], 1, True)} al año de crecimiento log). Ganó menos que el 3x sin "
        f"reglas ({pc(oos_t.loc['3x', 'CAGR'])}) y tuvo una caída máxima de {pc(oos_t.loc['LRS 3x', 'maxDD'], 0)}. "
        f"Frente al 1x lleva por detrás de su máximo relativo desde {oos_rel.loc['LRS 3x', 'inicio_racha_por_detrás']} "
        f"({num(oos_rel.loc['LRS 3x', 'racha_por_detrás_años'])} años, racha abierta). De las cuatro caídas del "
        f"periodo, {WORDS_F.get(len(quick), len(quick))} fueron rápidas: "
        + lst([num(eps.loc[(e, 'máx→mín'), 'meses_máx→mín']) for e in quick]) + " meses del máximo al mínimo en "
        + lst([e.split(' (')[0] for e in quick])
        + f". La de 2022 fue un bajista de {num(m22)} meses hasta octubre con repuntes, en el que la señal cambió "
        f"{sw22} veces antes de recuperar el máximo: no fue una V, pero tampoco la tendencia limpia de 2000–02 o "
        "2007–09 en la que la regla gana. Del máximo a la recuperación, la LRS 3x terminó en "
        + ", ".join(f"{pc(lrs_ep.loc[e, 'máx→recuperación'], 0)} ({e.split(' (')[0].split(',')[0]})"
                    for e in lrs_ep.index if pd.notna(lrs_ep.loc[e, 'máx→recuperación']) and "junio" not in e)
        + ", mientras el 1x ganaba entre "
        f"{pc(one_ep['máx→recuperación'].min(), 0)} y {pc(one_ep['máx→recuperación'].max(), 0)}. Su ventaja se "
        "concentra en pocos años alcistas con 3x casi todo el año: "
        + ", ".join(f"{y} ({pc(v, 0, True)})" for y, v in top.head(3).items())
        + ". Los peores fueron "
        + " y ".join(f"{y} ({pc(v, 0, True)})" for y, v in top.tail(2).iloc[::-1].items())
        + f", con idas y venidas alrededor de la media. La LRS-Kelly empató con el 1x ({pc(oos_t.loc['LRS-Kelly', 'CAGR'])}) y el ½ Kelly GJR lo superó "
        f"({pc(oos_t.loc['½ Kelly GJR', 'CAGR'])}), sin significación en ningún caso.")

    # ── 7. ficha ──
    rep.h("7. Ficha de cada estrategia")
    rep.p(f"La rentabilidad es la mediana y los percentiles 5 y 95 de las {N_WORLD:,}".replace(",", ".") +
          f" historias de cada mundo sintético, a 10 y {r13.YEARS} años, no la del backtest. La caída "
          f"máxima típica es la mediana de la caída máxima en {r13.YEARS} años. En los mundos la señal de la LRS "
          "usa el índice de rentabilidad total, como en la historia (sección 1), y las estrategias Kelly prevén "
          "la varianza con EWMA (λ = 0,94), no con GJR.")
    ct = card.copy()
    ct["rentabilidad"] = [f"{m:.1%} [{a:.1%}, {b:.1%}]" for m, a, b in
                          zip(ct.CAGR_mediana, ct.CAGR_P5, ct.CAGR_P95)]
    pv = ct.pivot_table(index="estrategia", columns=["años", "mundo"], values="rentabilidad", aggfunc="first")
    pv = pv.reindex(list(FOCUS))
    pv.columns = [f"{a} años · {m}" for a, m in pv.columns]
    pv = pv[[f"{a} años · {m}" for a in (10, r13.YEARS) for m in wl3]]
    show(rep, pv)
    dd = c30["maxDD_mediana"].unstack("mundo").reindex(list(FOCUS))[wl3]
    dd.columns = [f"caída máx. típica · {m}" for m in dd.columns]
    p80 = c30["P(caída > 80%)"].unstack("mundo").reindex(list(FOCUS))[wl3]
    p80.columns = [f"P(caída > 80%) · {m}" for m in p80.columns]
    show(rep, dd.join(p80), pct=set(dd.columns) | set(p80.columns), digits=2)
    op = pd.DataFrame({
        "CAGR historia": hist.loc[list(FOCUS), "CAGR"], "caída máx. historia": hist.loc[list(FOCUS), "maxDD"],
        "tiempo invertido": trade.loc[list(FOCUS), "tiempo_invertido"],
        "exposición media (x)": trade.loc[list(FOCUS), "exposición_media"],
        "cambios de exposición/año": trade.loc[list(FOCUS), "operaciones_año"],
        "P(delante del 1x, 10 años)": [P(k, 10).P_historia for k in FOCUS],
        "racha más larga por detrás (años)": [np.nan] + [bh.loc[k, "racha_más_larga_años"] for k in OTHERS]},
        index=list(FOCUS))
    rep.p(f"Operativa en la historia {start_all.date()} → {end.date()}; en la fila del 1x, P es frente a las "
          "letras. Un cambio de exposición es cada sesión en que cambia la posición objetivo: en la LRS 3x "
          "cuenta uno cada entrada y cada salida.")
    show(rep, op, pct={"CAGR historia", "caída máx. historia", "tiempo invertido", "P(delante del 1x, 10 años)"},
              digits=2)
    li = {wl: (np.mean(w["lrs_invested"]), np.mean(w["lrs_switches"])) for wl, w in worlds.items()}
    rep.p("En los mundos la LRS 3x original está invertida el " +
          " / ".join(pc(li[w][0], 0) for w in wl3) + " del tiempo (bootstrap / GJR / regímenes) y cambia de "
          "posición " + " / ".join(num(li[w][1]) for w in wl3) + " veces al año.")
    fig_card(card, hist, ah, aw, rep)
    rep.h("Fichas", 3)
    for k in FOCUS:
        lo5 = min(c30.loc[(k, w), "CAGR_P5"] for w in wl3)
        hi95 = max(c30.loc[(k, w), "CAGR_P95"] for w in wl3)
        extra = (f" Frente al 1x a 10 años: historia {pc(P(k, 10).P_historia, 0)}, mundos {wtxt(k, 10)}."
                 if k != "1x" else f" Frente a las letras a 10 años: historia {pc(P(k, 10).P_historia, 0)}, "
                 f"mundos {wtxt(k, 10)}.")
        if k == "LRS 3x":
            extra += (f" Con la señal de precio: historia {pc(P(px3, 10).P_historia, 0)}, mundos "
                      f"{wtxt(px3, 10)}.")
        rep.p(f"- **{k}.** Rentabilidad anual a {r13.YEARS} años (mediana bootstrap / GJR / regímenes): "
              f"{' / '.join(pc(c30.loc[(k, w), 'CAGR_mediana']) for w in wl3)}; del P5 al P95, entre {pc(lo5)} y "
              f"{pc(hi95)} según el mundo. A 10 años: "
              f"{' / '.join(pc(c10.loc[(k, w), 'CAGR_mediana']) for w in wl3)}. Caída máxima típica en "
              f"{r13.YEARS} años: {' / '.join(pc(c30.loc[(k, w), 'maxDD_mediana'], 0) for w in wl3)} (historia "
              f"1938–2026: {pc(hist.loc[k, 'maxDD'], 0)}). Tiempo invertido {pc(trade.loc[k, 'tiempo_invertido'], 0)}, "
              f"exposición media {num(trade.loc[k, 'exposición_media'], 2)}x, "
              f"{num(trade.loc[k, 'operaciones_año'])} cambios de exposición al año.{extra} Regla: {RULES[k]}")
    rep.h("Reglas congeladas y procedencia", 3)
    rep.p(f"Parámetros comunes: señal al cierre de t y orden al cierre de t+1 (EXEC_LAG = {LAG}); 5 pb por "
          f"unidad de rotación; caja en letras del Tesoro; el 3x paga sobre toda su exposición el tipo de call "
          f"money (hasta 1954) o de los fondos federales más el sobrecoste del swap s = {pc(d['spread'], 3)}/año, y "
          f"la comisión del ETF "
          f"{pc(d['er'], 2)}. Commit `{prov['head']}` (git rev-parse HEAD). El árbol de trabajo tenía "
          "cambios sin commit en archivos de la base, así que el hash no basta para reproducir. La tabla "
          "siguiente da la huella SHA-256 (16 primeros caracteres) de cada archivo tal como se ejecutó.")
    show(rep, prov["files"], index=True)
    prov2 = provenance()["files"]
    changed = [f for f in prov["files"].index if not f.endswith("r20_investor.py")
               and prov2.loc[f, "sha256"] != prov["files"].loc[f, "sha256"]]
    rep.p("Comprobación al terminar: " + (
        "ningún archivo de la base cambió durante la ejecución." if not changed else
        f"**{lst(changed)} cambió durante la ejecución**; las cifras pueden mezclar dos versiones."))

    # ── 8. salvedades ──
    rep.h("8. Salvedades")
    rep.p(
        f"- **Una sola historia.** Las probabilidades históricas vienen de ventanas solapadas: a 10 y 20 años hay "
        f"{int(yrs_hist // 10)} y {int(yrs_hist // 20)} ventanas independientes. El intervalo de Wilson con ese "
        "tamaño es sólo orientativo.\n"
        f"- **La comparación de las dos señales depende de un día.** El {pc(sg['cuota_día'], 0)} de la diferencia "
        f"entre la señal de precio y la original sale del {sg['día'].strftime('%d-%m-%Y')}, que la original pasó "
        "dentro; cualquier comparación entre las dos en la historia debe leerse con eso en mente (sección 1).\n"
        "- **Los mundos de R13 se calibran con toda la muestra 1928–2026.** El bootstrap la remuestrea, el GJR y "
        "el modelo de regímenes se estiman con ella, y rf, f y el dividendo dv de los mundos GJR y de "
        "regímenes son sus medias constantes. No miran el futuro de ninguna señal, pero no son independientes "
        "de la historia con la que se eligió la regla.\n"
        "- **En los mundos, el índice de precio de la sensibilidad es una aproximación.** En el bootstrap "
        "es el ^GSPC real remuestreado por bloques junto con la rentabilidad total; en los mundos GJR y de "
        f"regímenes, la rentabilidad total menos un dividendo constante del {pc(wmeta['dv'] * 252, 2)} anual, cuando "
        "en la historia varió del 1% al 7%. Las estrategias Kelly de los mundos prevén la varianza con EWMA y "
        "no con GJR; la ficha de rentabilidad del ½ Kelly GJR y de la LRS-Kelly corresponde a esa variante.\n"
        "- **Cada mundo favorece a unas estrategias.** El de regímenes usa innovaciones normales (casi sin días "
        "de −10%) y medias distintas por régimen, lo que favorece a las estrategias de volatilidad. El GJR "
        "mantiene constante la media logarítmica, con lo que la media aritmética sube con la varianza, y eso "
        "penaliza salir en los periodos turbulentos (R18). Ninguno es «el» mundo verdadero; el rango entre los "
        "tres es la incertidumbre de modelo.\n"
        "- **Regla de abandono.** El inversor pasa al 1x (no a caja), sin impuestos ni costes del cambio, y revisa "
        "cada día; con revisión mensual saltaría algo menos. Inicios cada 21 sesiones.\n"
        "- **Crisis.** Se datan con la serie completa y la regla de fin de crisis (recuperar la mitad de lo "
        "perdido) difiere de las cronologías habituales en 1929–32, 1937–42 y 2022 (sección 4). La rejilla de "
        "duración que separa crisis largas y cortas se añadió tras la revisión y se muestra entera.\n"
        "- **Cierres dudosos.** La base conserva siete cierres de ^GSPC anteriores a 1950 que R14 marca a mano "
        "como probables erratas. Cambian la lista de días extremos (sección 5), no las conclusiones generales.\n"
        "- **Diagnóstico de las estrategias Kelly.** El reparto de la pérdida relativa por tipo de sesión es "
        "contable, no causal: una sesión fuera por el filtro de tendencia también habría estado fuera, o con "
        "otra exposición, con otra fórmula de la prima. Los cortes de 1978 y 1984 se eligieron tras ver el "
        "gráfico.\n"
        f"- **Fuera de muestra.** Son {num(oos_y)} años, y sólo la LRS 3x original es fuera de muestra; las "
        "estrategias Kelly heredan parámetros elegidos con datos que incluyen este periodo.\n"
        "- **Costes.** 5 pb por unidad de rotación, sin impuestos. En una cuenta sujeta a impuestos, la LRS 3x "
        f"realizaría plusvalías en {num(trade.loc['LRS 3x', 'operaciones_año'] / 2)} viajes de ida y vuelta al año.\n"
        "- **Base de código.** Las cifras usan la base corregida (history.py, r12_long_ml.py, r13_lrs.py; ver "
        "BITACORA.md) y la LRS 3x original de Gayed y Bilello, con la señal sobre la rentabilidad total "
        f"({pc(hist.loc['LRS 3x', 'CAGR'])} al año desde {start_all.year}; sección 7).")
    rep.save()
    return {"ah": ah, "aw": aw, "card": card, "bh": bh, "bh_hi": bh_hi, "bw": bw, "ab_h": ab_h, "ab_w": ab_w,
            "crisis": (head, cr_a, cr_b, cr_c, cd), "days": days, "days_fix": days_fix, "hist_fix": hist_fix,
            "oos": (oos_t, oos_rel, oos_mk, eps), "hist": hist, "trade": trade, "prov": prov,
            "worlds_lrs": li, "kelly": kd, "signal": (sg, sgw), "wmeta": wmeta}


if __name__ == "__main__":
    run()
