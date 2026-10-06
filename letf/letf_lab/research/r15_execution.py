"""R15 — Ejecución y suerte del calendario en la LRS 3x y la LRS-Kelly.

Qué parte del resultado de una regla de tendencia apalancada depende de detalles
de ejecución que el inversor no controla o elige al azar. Para la LRS 3x
original (3x sobre la media de 200 sesiones del índice de rentabilidad total,
letras debajo: la regla de Gayed y Bilello 2016) y la LRS-Kelly de R13, en el
S&P 500, el Nasdaq-100 y los semiconductores:

1. Retraso de ejecución: la señal se calcula al cierre de t y la cartera cobra
   la rentabilidad a partir del cierre de t + retraso − 1. Retraso 1 = mismo
   cierre (MOC con la señal de las 15:50); 2 = cierre siguiente (base de R13);
   3 y 5 = dos y cuatro sesiones de demora.
2. Suerte del calendario: la señal sólo se mira una vez por semana (5 días de la
   semana posibles) o cada 21 sesiones (21 desfases posibles). El abanico de
   resultados entre desfases, que son igual de razonables a priori, es la suerte
   del calendario.
3. Señal de rentabilidad total (la original) frente a la de precio
   (sensibilidad del laboratorio).
4. Distancia del índice de la señal a su media los días en que cambia la
   señal y duración del nuevo estado: cuántos cruces son ruido, frente a un
   paseo aleatorio.
5. El medio Kelly de R12 se revisa cada 21 sesiones con `keep[::H]`, es decir,
   con un desfase fijado por el primer día de la serie: se miden los 21
   desfases posibles (reimplementado aquí, sin tocar R12).
6. Octubre de 1987: cuánto del CAGR de 1970–1999 depende del 19-10-1987 con
   cada retraso y cada índice de señal.

Todo está fijado de antemano (retrasos 1, 2, 3 y 5; los 5 días de la semana; los
21 desfases; umbrales de ruido de 5, 10 y 21 sesiones y de 0,5, 1 y 2%) y se
muestra entero. Ninguna señal usa datos posteriores a su fecha: la media móvil
(en el S&P 500, sobre la rentabilidad total: el índice oficial ^SP500TR desde
1988 y, antes, ^GSPC con el D/P de Shiller del mes anterior), la prima de Kelly (media expansiva de u − f) y el
GJR-GARCH (reestimado cada cuatro años con datos pasados) sólo ven el pasado. Los costes del swap de los
3x sintéticos se calibran en R12 con el periodo de los ETFs reales (muestra
completa), igual para todas las variantes, así que no afecta a las
comparaciones de este informe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from letf_lab import kelly, metrics, stats
from letf_lab.report import FIG_W, Report, comma_axis
from letf_lab.research import r12_long_ml as r12
from letf_lab.research import r13_lrs as r13

SEED = 15
B = 2000                       # remuestreos del bootstrap estacionario
BLOCK = 63                     # bloque medio del bootstrap (sesiones)
LAGS = (1, 2, 3, 5)
BASE_LAG = r13.EXEC_LAG        # 2: orden al cierre siguiente
H = r12.H                      # 21
WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes")
NOISE_DAYS = (5, 10, 21)       # un cruce que se deshace en ≤ k sesiones es ruido
NOISE_DIST = (0.005, 0.01, 0.02)
N_RW = 200                     # caminos del paseo aleatorio de referencia
STRATS = ("LRS 3x (original)", "LRS-Kelly")
BASE_SIGNAL = r13.SIGNAL       # la señal original (Gayed y Bilello): rentabilidad total
SIGNALS = (BASE_SIGNAL, r13.PRICE)
KIND_LABEL = {BASE_SIGNAL: "rentabilidad total (original)", r13.PRICE: "precio"}
CRASH = pd.Timestamp("1987-10-19")
CRASH_DAYS = (pd.Timestamp("1987-10-16"), CRASH)   # viernes y lunes del crac (−5,2% y −20,5%)
COVID = ("2020-02-15", "2020-04-30")                # caída y rebote de 2020
SUB = ("1970-01-01", "1999-12-31")
MARKETS = ("S&P 500", "Nasdaq-100", "Semis")
COLORS = {"LRS 3x (original)": "C0", "LRS-Kelly": "C3", "1x comprar y mantener": "k"}
SHOW = {"S&P 500": "S&P 500", "Nasdaq-100": "Nasdaq-100", "Semis": "Semiconductores"}
THE = {"S&P 500": "el S&P 500", "Nasdaq-100": "el Nasdaq-100", "Semis": "los semiconductores"}
MKT_COLORS = {"S&P 500": "C0", "Nasdaq-100": "C1", "Semis": "C2"}


def row_name(d: dict, strat: str, kind: str) -> str:
    """Nombre de fila de `strat` con el índice de señal `kind`: con la original, «LRS 3x
    (original) · rentabilidad total»; con la de precio, la gemela de R13 (r13.price_label),
    «LRS 3x (señal de precio)», o «… (señal de precio aproximado)» en los mercados sin un
    índice de precio real (Nasdaq-100 y semis). La señal de precio nunca es «original»."""
    if kind == BASE_SIGNAL:
        return f"{strat} · {kind}"
    return r13.price_label(strat, approx=not d.get("px_real", False))


# ───────────────────────────── calendario ─────────────────────────────

def weekly_schedule(index: pd.DatetimeIndex, weekday: int) -> np.ndarray:
    """Sesiones de revisión semanal: el día `weekday` (0 = lunes) de cada semana;
    si ese día no hay sesión (festivo), la primera sesión posterior."""
    idx = pd.DatetimeIndex(index)
    monday = (idx - pd.to_timedelta(idx.weekday, unit="D")).normalize()
    targets = np.unique(monday.values) + np.timedelta64(weekday, "D")
    pos = np.searchsorted(idx.values, targets.astype(idx.values.dtype), side="left")
    keep = np.zeros(len(idx), bool)
    keep[pos[pos < len(idx)]] = True
    return keep


def every_schedule(n: int, period: int, offset: int) -> np.ndarray:
    """Revisión cada `period` sesiones empezando en la sesión `offset` (keep[offset::period])."""
    keep = np.zeros(n, bool)
    keep[offset::period] = True
    return keep


def hold(target: pd.Series, keep: np.ndarray) -> pd.Series:
    """El objetivo sólo se observa en las sesiones de revisión y se mantiene entre
    ellas. La primera sesión con objetivo definido abre la cartera (también cuenta
    como revisión), para que todos los desfases empiecen el mismo día."""
    k = np.asarray(keep, bool).copy()
    fv = target.first_valid_index()
    if fv is not None:
        k[target.index.get_loc(fv)] = True
    return target.where(k).ffill()


# ───────────────────────────── estrategias ─────────────────────────────

# Índices de la señal: los de R13 (r13.tr_index, r13.signal_index), sin copias locales.
signal_index = r13.signal_index


def weights(d: dict, strat: str, kind: str = BASE_SIGNAL, keep: np.ndarray | None = None,
            kel: pd.Series | None = None) -> pd.Series:
    """Fracción de la cartera en el 3x decidida al cierre de cada sesión.

    LRS 3x: 1 sobre la media de 200 sesiones, 0 debajo. LRS-Kelly: medio Kelly
    con la varianza GJR prevista (fracción del 3x entre 0 y 1) si está sobre la
    media, 0 debajo, con el filtro de reajuste mínimo de R13 (sticky)."""
    sig = r13.above_sma(signal_index(d, kind))
    if strat == "LRS 3x (original)":
        return sig if keep is None else hold(sig, keep)
    tgt = kel * sig
    if keep is not None:
        tgt = hold(tgt, keep)
    return r13.sticky(tgt)


def lagged_exposure(w: pd.Series, lag: int) -> pd.Series:
    """Fracción en el 3x que soporta la rentabilidad de cada día (como en run_weights)."""
    return w.shift(lag)


def neutralize(r: pd.Series, rf_: pd.Series, day: pd.Timestamp) -> pd.Series:
    """La misma serie con la rentabilidad de `day` sustituida por la del T-bill."""
    out = r.copy()
    if day in out.index:
        out.loc[day] = rf_.loc[day]
    return out


def r12_strategy_offset(d: dict, var_fc: pd.Series, offset: int = 0, lag: int = BASE_LAG,
                        mu_fc: pd.Series | None = None) -> tuple[pd.Series, pd.Series]:
    """r12.strategy con la revisión cada H sesiones desplazada: keep[offset::H].

    Copia literal de la lógica de R12 salvo el desfase (offset = 0 la reproduce)."""
    u, r = d["u"], d["rf"]
    prem = mu_fc - d["f"] * 252 if mu_fc is not None else kelly.premium(u, d["f"], r12.MIN_OBS)
    L = r12.KELLY_FRACTION * (prem - d["spread"]) / var_fc
    w = (L / 3).clip(0, 1)
    keep = every_schedule(len(w), r12.H, offset)
    w = w.where(pd.Series(keep, index=w.index)).ffill()
    w = w.shift(lag).fillna(0.0)
    ret = w * d["x3"] + (1 - w) * r - w.diff().abs().fillna(0) * r12.COST_BPS / 1e4
    return ret, w


def r12_windows(d: dict) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Fechas que usa R12 en su tabla de estrategias, sin entrenar los modelos de ML.

    La tabla de varianzas de R12 empieza cuando el primer modelo de ML tiene
    MIN_OBS observaciones completas (rasgos y objetivo) y descarta, con dropna,
    las últimas H sesiones (varianza realizada aún desconocida). Devuelve
    (primera previsión, última previsión, inicio de la evaluación)."""
    u = d["u"].dropna()
    X = r12.features(u)
    target = (u ** 2).rolling(H).sum().shift(-H) * 252 / H
    ok = (X.notna().all(axis=1) & target.notna()).values
    first = u.index[int(np.argmax(np.cumsum(ok) >= r12.MIN_OBS))]
    last = target.last_valid_index()
    start = max(first, d["eval_from"]) if d["eval_from"] is not None else first
    return first, last, start


# ───────────────────────────── métricas ─────────────────────────────

def perf(r: pd.Series, rf_: pd.Series, start, end=None) -> dict:
    r = r.loc[start:end].dropna()
    return {"CAGR": metrics.cagr(r), "vol": metrics.vol(r), "sharpe": metrics.sharpe(r, rf_),
            "maxDD": metrics.max_dd(r)}


def trades_per_year(w: pd.Series, start, end=None) -> float:
    w = w.loc[start:end].dropna()
    return float((w.diff().abs() > 1e-9).sum() / (len(w) / 252))


def log_diff(a: pd.Series, b: pd.Series, start, end=None) -> pd.Series:
    x = pd.concat([a, b], axis=1).loc[start:end].dropna()
    return np.log1p(x.iloc[:, 0].clip(lower=-0.999)) - np.log1p(x.iloc[:, 1].clip(lower=-0.999))


def growth_diff_ci(a: pd.Series, b: pd.Series, start, rng) -> tuple[float, float, float]:
    """Diferencia de crecimiento logarítmico anual a − b e IC del 90% (bootstrap estacionario)."""
    x = log_diff(a, b, start).values
    est, lo, hi = stats.bootstrap_ci(x, np.mean, BLOCK, B, rng)
    return est * 252, lo * 252, hi * 252


def growth_se(r: pd.Series, start) -> float:
    """Error típico del crecimiento log anual con rentabilidades independientes (orientativo)."""
    x = np.log1p(r.loc[start:].dropna().clip(lower=-0.999))
    return float(x.std() * np.sqrt(252) / np.sqrt(len(x) / 252))


def yearly_contribution(x: pd.Series) -> pd.Series:
    """Aportación de cada año natural a la diferencia de crecimiento anual de toda la
    muestra: suma de las diferencias log del año dividida por los años de la muestra.
    Las aportaciones suman exactamente x.sum() / años (la Δcrec_anual)."""
    return x.groupby(x.index.year).sum() / (len(x) / 252)


def excluding(x: pd.Series, days=None, years=None) -> pd.Series:
    """La serie sin las fechas `days` ni los años naturales `years` (se quitan de la muestra)."""
    keep = np.ones(len(x), bool)
    if days is not None:
        keep &= ~x.index.isin(pd.DatetimeIndex(days))
    if years is not None:
        keep &= ~np.isin(x.index.year, list(years))
    return x[keep]


def annual_mean_ci(x: pd.Series, rng) -> tuple[float, float, float]:
    """Media diaria ×252 e IC del 90% por bootstrap estacionario (como growth_diff_ci)."""
    est, lo, hi = stats.bootstrap_ci(x.values, np.mean, BLOCK, B, rng)
    return est * 252, lo * 252, hi * 252


def ci_txt(lo: float, hi: float) -> str:
    return f"[{lo:+.1%}, {hi:+.1%}]"


def rank_str(base: float, v) -> str:
    """«k de n»: cuántas de las n variantes quedan por debajo de la base."""
    v = np.asarray(v, float)
    return f"{int((v < base - 1e-12).sum())} de {len(v)}"


def lag1_autocorr(r: pd.Series) -> tuple[float, float, int]:
    """Autocorrelación de primer orden y su error típico aproximado 1/√n (ruido blanco)."""
    r = r.dropna()
    return float(r.autocorr(1)), float(1 / np.sqrt(len(r))), len(r)


# ───────────────────────────── cruces de la media ─────────────────────────────

def crossings(px: pd.Series, n: int = 200) -> pd.DataFrame:
    """Días en que cambia la señal precio > media de n sesiones.

    distancia: px/media − 1 el día del cruce; duración: sesiones hasta el
    siguiente cruce (NaN si no lo hay: censurado); excursión: máxima |distancia|
    alcanzada durante el nuevo estado (incluido el día del cruce)."""
    ma = px.rolling(n).mean()
    dist = (px / ma - 1).values
    sig = np.where(np.isnan(dist), np.nan, (px > ma).values.astype(float))
    valid = ~np.isnan(sig)
    ch = np.zeros(len(sig), bool)
    ch[1:] = valid[1:] & valid[:-1] & (sig[1:] != sig[:-1])
    pos = np.flatnonzero(ch)
    if len(pos) == 0:
        return pd.DataFrame(columns=["dirección", "distancia", "duración", "excursión"])
    ad = np.abs(np.nan_to_num(dist))
    exc = np.maximum.reduceat(ad, pos)
    dur = np.r_[np.diff(pos), np.nan].astype(float)
    return pd.DataFrame({"dirección": np.where(sig[pos] == 1, 1, -1), "distancia": dist[pos],
                         "duración": dur, "excursión": exc}, index=px.index[pos])


def crossing_summary(c: pd.DataFrame, years: float) -> dict:
    a = c["distancia"].abs()
    out = {"cruces": len(c), "cruces_año": len(c) / years, "|dist|_mediana": a.median()}
    for x in NOISE_DIST:
        out[f"|dist|<{x:.1%}"] = (a < x).mean()
    for k in NOISE_DAYS:
        out[f"se_deshace_≤{k}"] = (c["duración"] <= k).mean()
    out["excursión<2%"] = (c["excursión"] < 0.02).mean()
    noise, lasting = c["duración"] <= 10, ~(c["duración"] <= 21)  # el último cruce cuenta como duradero
    out["|dist|_med_ruido(≤10)"] = a[noise].median()
    out["|dist|_med_duraderos(>21)"] = a[lasting].median()
    out["duraderos_con_|dist|<1%"] = (a[lasting] < 0.01).mean()
    return out


def random_walk_crossings(pr: np.ndarray, n_days: int, rng, n: int = 200,
                          paths: int = N_RW) -> dict:
    """Los mismos estadísticos en paseos aleatorios: rentabilidades diarias del índice
    de precio remuestreadas con reemplazo (misma deriva y distribución, sin
    dependencia temporal). Se descartan las n primeras sesiones de cada camino."""
    rows = []
    for _ in range(paths):
        r = rng.choice(pr, n_days + n)
        px = pd.Series(np.cumprod(1 + r))
        c = crossings(px, n)
        rows.append(crossing_summary(c, n_days / 252))
    return pd.DataFrame(rows).median().to_dict()


# ───────────────────────────── cálculo ─────────────────────────────

def market_start(d: dict, var_fc: pd.Series) -> pd.Timestamp:
    """Inicio de la evaluación, el mismo de R13: primera previsión de varianza (o eval_from)."""
    s = var_fc.first_valid_index()
    if d["eval_from"] is not None:
        s = max(s, d["eval_from"])
    idx = d["u"].index
    return idx[idx.get_indexer([s], method="bfill")[0]]


def run_from(d: dict, w: pd.Series, lag: int, start) -> pd.Series:
    """r13.run_weights; desde `start`, las sesiones anteriores a la primera orden
    ejecutada (las `lag` primeras de la LRS-Kelly) están en T-bills, para que todos
    los retrasos compartan exactamente la misma ventana."""
    r = r13.run_weights(d, w, lag=lag)
    after = r.index >= start
    return r.where(~after | r.notna(), d["rf"])


def compute_market(d: dict, var_fc: pd.Series, rng) -> dict:
    kel = r13.kelly_exposure(d, var_fc)
    start = market_start(d, var_fc)
    n = len(d["u"])
    out = {"start": start, "kel": kel, "lags": {}, "w": {}, "cal": {}, "cal_w": {}}
    # 1 y 3: retrasos con cada índice de señal
    for strat in STRATS:
        for kind in SIGNALS:
            w = weights(d, strat, kind, kel=kel)
            out["w"][(strat, kind)] = w
            for lag in LAGS:
                out["lags"][(strat, kind, lag)] = run_from(d, w, lag, start)
    # comprobación: la base reproduce R13 (salvo las sesiones iniciales en letras)
    R13, _ = r13.strategies(d, var_fc)
    for strat in STRATS:
        a, b = out["lags"][(strat, BASE_SIGNAL, BASE_LAG)].loc[start:], R13[strat].loc[start:]
        ok = b.notna()
        assert np.allclose(a[ok].values, b[ok].values), strat
    # 2: suerte del calendario (señal original, orden al cierre siguiente)
    for strat in STRATS:
        for k, day in enumerate(WEEKDAYS):
            w = weights(d, strat, BASE_SIGNAL, keep=weekly_schedule(d["u"].index, k), kel=kel)
            out["cal"][(strat, "semanal", day)] = run_from(d, w, BASE_LAG, start)
            out["cal_w"][(strat, "semanal", day)] = w
        for off in range(H):
            w = weights(d, strat, BASE_SIGNAL, keep=every_schedule(n, H, off), kel=kel)
            out["cal"][(strat, "mensual", off)] = run_from(d, w, BASE_LAG, start)
            out["cal_w"][(strat, "mensual", off)] = w
    # 4: cruces de la media (índice de la señal original) en la ventana de evaluación
    px = signal_index(d, BASE_SIGNAL)
    c = crossings(px)
    out["cruces"] = c.loc[start:]
    years = len(d["u"].loc[start:]) / 252
    pr = px.pct_change().loc[start:].dropna().values
    out["cruces_rw"] = random_walk_crossings(pr, len(d["u"].loc[start:]), rng)
    out["years"] = years
    # 5: R12 con sus 21 desfases
    first, last, s12 = r12_windows(d)
    v12 = var_fc.where((var_fc.index >= first) & (var_fc.index <= last))
    out["r12"], out["r12_w"] = {}, {}
    for off in range(H):
        out["r12"][off], out["r12_w"][off] = r12_strategy_offset(d, v12, off)
    out["r12_start"] = s12
    out["r12_daily"] = run_from(d, r13.sticky(kel), BASE_LAG, start)  # ½ Kelly GJR de R13 (diario)
    return out


# ───────────────────────────── tablas ─────────────────────────────

def lag_table(d: dict, res: dict, rng) -> pd.DataFrame:
    rows = {}
    st = res["start"]
    rows["1x comprar y mantener"] = {**perf(d["u"], d["rf"], st), "operaciones_año": 0.0}
    for strat in STRATS:
        base = res["lags"][(strat, BASE_SIGNAL, BASE_LAG)]
        for lag in LAGS:
            r = res["lags"][(strat, BASE_SIGNAL, lag)]
            row = perf(r, d["rf"], st)
            row["operaciones_año"] = trades_per_year(res["w"][(strat, BASE_SIGNAL)], st)
            if lag != BASE_LAG:
                est, lo, hi = growth_diff_ci(r, base, st, rng)
                row.update({"Δcrec_vs_2": est, "IC90_bajo": lo, "IC90_alto": hi})
            rows[f"{strat} · retraso {lag}"] = row
    return pd.DataFrame(rows).T


def lag_concentration(d: dict, res: dict, rng) -> pd.DataFrame:
    """¿De cuántos días (o años) sale la diferencia entre retrasos? (señal original).

    Δ_19oct87 y Δ_feb_abr_2020: aportación de ese día o tramo a la Δcrec_anual de toda la
    muestra (suma log / años). Δ_sin_2020: media anual del resto de la muestra, sin el año
    2020 entero, con su IC del 90%. años_a_favor: años naturales en que el retraso rinde más
    que el 2, entre los que difieren."""
    st = res["start"]
    rows = {}
    for strat in STRATS:
        base = res["lags"][(strat, BASE_SIGNAL, BASE_LAG)]
        for lag in (1, 3, 5):
            x = log_diff(res["lags"][(strat, BASE_SIGNAL, lag)], base, st)
            yrs = len(x) / 252
            top = x.abs().sort_values(ascending=False).index[:10]
            row = {"Δcrec_anual": x.sum() / yrs, "Δ_sin_10_mayores": x.drop(top).sum() / yrs,
                   "días_distintos": int((x.abs() > 1e-12).sum())}
            if st < CRASH:
                row["Δ_19oct87"] = x.get(CRASH, 0.0) / yrs
            row["Δ_feb_abr_2020"] = x.loc[COVID[0]:COVID[1]].sum() / yrs
            est, lo, hi = annual_mean_ci(excluding(x, years=[2020]), rng)
            row["Δ_sin_2020"] = est
            row["IC90_sin_2020"] = ci_txt(lo, hi)
            yc = yearly_contribution(x)
            yc = yc[yc.abs() > 1e-6]
            row["años_a_favor"] = f"{int((yc > 0).sum())} de {len(yc)}"
            best = x.loc[top[:3]]
            row["3 mayores días (Δ log)"] = ", ".join(f"{t.date()} ({v:+.1%})" for t, v in best.items())
            rows[f"{strat} · {lag} frente a 2"] = row
    return pd.DataFrame(rows).T


def pre_post_1990(d: dict, res: dict, rng) -> pd.DataFrame | None:
    """Diferencia de crecimiento anual frente al retraso 2 antes y después de 1990, con IC
    del 90% en los dos tramos, y el tramo anterior sin el crac (16 y 19-10-1987)."""
    st = res["start"]
    if st >= pd.Timestamp("1990-01-01"):
        return None
    rows, ci = {}, {}
    for strat in STRATS:
        base = res["lags"][(strat, BASE_SIGNAL, BASE_LAG)]
        for lag in (1, 3, 5):
            x = log_diff(res["lags"][(strat, BASE_SIGNAL, lag)], base, st)
            pre, post = x.loc[:"1989-12-31"], x.loc["1990-01-01":]
            e0, l0, h0 = annual_mean_ci(pre, rng)
            e1, l1, h1 = annual_mean_ci(excluding(pre, days=CRASH_DAYS), rng)
            e2, l2, h2 = annual_mean_ci(post, rng)
            rows[f"{strat} · {lag} frente a 2"] = {
                "años_hasta_1989": len(pre) / 252,
                "Δ_hasta_1989": e0, "IC90_hasta_1989": ci_txt(l0, h0),
                "Δ_hasta_1989_sin_crac": e1, "IC90_hasta_1989_sin_crac": ci_txt(l1, h1),
                "Δ_desde_1990": e2, "IC90_desde_1990": ci_txt(l2, h2)}
            ci[f"{strat} · {lag} frente a 2"] = {"hasta_1989": (l0, h0), "hasta_1989_sin_crac": (l1, h1),
                                                 "desde_1990": (l2, h2)}
    out = pd.DataFrame(rows).T.infer_objects()
    out.attrs["ci"] = ci   # intervalos numéricos para el texto
    return out


def autocorr_table(D: dict, RES: dict) -> pd.DataFrame:
    """Autocorrelación a un día del índice (rentabilidad total) en la ventana de evaluación."""
    rows = {}
    for name in MARKETS:
        u = D[name]["u"].loc[RES[name]["start"]:]
        tramos = [("todo", None, None)]
        if RES[name]["start"] < pd.Timestamp("1990-01-01"):
            tramos = [("hasta 1989", None, "1989-12-31"), ("desde 1990", "1990-01-01", None)]
        for lab, a, b in tramos:
            rho, se, n = lag1_autocorr(u.loc[a:b])
            rows[f"{SHOW[name]} · {lab}"] = {"ρ1": rho, "error_típico": se, "t": rho / se, "sesiones": n}
    return pd.DataFrame(rows).T.infer_objects()


def next_day_after_cross(d: dict, res: dict) -> pd.DataFrame:
    """Rentabilidad del índice (1x) la sesión siguiente a un cruce: es la que
    separa ejecutar en el mismo cierre (retraso 1) de hacerlo al siguiente (2)."""
    c = res["cruces"]
    u = d["u"]
    nxt = u.shift(-1).reindex(c.index)
    rows = {}
    periods = [("todo", None, None)]
    if res["start"] < pd.Timestamp("1990-01-01"):
        periods += [("hasta 1989", None, "1989-12-31"), ("desde 1990", "1990-01-01", None)]
    for lab, a, b in periods:
        for dname, sgn in (("cruce a la baja", -1), ("cruce al alza", 1)):
            m = (c["dirección"] == sgn)
            x = nxt[m].loc[a:b].dropna()
            se = x.std() / np.sqrt(len(x)) * 1e4 if len(x) > 1 else np.nan
            normal = u.loc[res["start"]:].loc[a:b].mean() * 1e4
            rows[f"{lab} · {dname}"] = {
                "cruces": len(x), "rent_día_siguiente_pb": x.mean() * 1e4, "error_típico_pb": se,
                "rent_media_todos_los_días_pb": normal,
                "t_frente_a_día_normal": (x.mean() * 1e4 - normal) / se}
    return pd.DataFrame(rows).T


def signal_table(d: dict, res: dict) -> tuple[pd.DataFrame, float, float]:
    st = res["start"]
    rows = {}
    for strat in STRATS:
        for kind in SIGNALS:
            row = {f"CAGR_retraso_{lag}": perf(res["lags"][(strat, kind, lag)], d["rf"], st)["CAGR"]
                   for lag in LAGS}
            r2 = res["lags"][(strat, kind, BASE_LAG)]
            if st < CRASH:
                row["CAGR_retraso_2_sin_19oct87"] = perf(neutralize(r2, d["rf"], CRASH), d["rf"], st)["CAGR"]
            row["maxDD_retraso_2"] = perf(r2, d["rf"], st)["maxDD"]
            row["operaciones_año"] = trades_per_year(res["w"][(strat, kind)], st)
            rows[row_name(d, strat, kind)] = row
    sp = r13.above_sma(signal_index(d, r13.PRICE)).loc[st:]
    stt = r13.above_sma(signal_index(d, BASE_SIGNAL)).loc[st:]
    return pd.DataFrame(rows).T, float((sp != stt).mean()), float(((sp == 1) & (stt == 0)).mean())


def calendar_table(d: dict, res: dict, drop_day: pd.Timestamp | None = None) -> pd.DataFrame:
    """Abanico de resultados entre desfases; con drop_day, ese día en T-bills en todas las series."""
    st = res["start"]
    fix = (lambda r: neutralize(r, d["rf"], drop_day)) if drop_day is not None else (lambda r: r)
    rows = {}
    for strat in STRATS:
        daily = perf(fix(res["lags"][(strat, BASE_SIGNAL, BASE_LAG)]), d["rf"], st)
        rows[f"{strat} · diaria"] = {"desfases": 1, "CAGR_mín": daily["CAGR"], "CAGR_mediana": daily["CAGR"],
                                     "CAGR_máx": daily["CAGR"], "amplitud_pp": 0.0,
                                     "maxDD_peor": daily["maxDD"], "maxDD_mejor": daily["maxDD"],
                                     "sharpe_mín": daily["sharpe"], "sharpe_máx": daily["sharpe"],
                                     "operaciones_año": trades_per_year(res["w"][(strat, BASE_SIGNAL)], st)}
        for freq in ("semanal", "mensual"):
            keys = [k for k in res["cal"] if k[0] == strat and k[1] == freq]
            P = pd.DataFrame([perf(fix(res["cal"][k]), d["rf"], st) for k in keys])
            rows[f"{strat} · {freq}"] = {
                "desfases": len(keys), "CAGR_mín": P.CAGR.min(), "CAGR_mediana": P.CAGR.median(),
                "CAGR_máx": P.CAGR.max(), "amplitud_pp": (P.CAGR.max() - P.CAGR.min()) * 100,
                "maxDD_peor": P.maxDD.min(), "maxDD_mejor": P.maxDD.max(),
                "sharpe_mín": P.sharpe.min(), "sharpe_máx": P.sharpe.max(),
                "operaciones_año": np.median([trades_per_year(res["cal_w"][k], st) for k in keys])}
    return pd.DataFrame(rows).T


def weekday_table(d: dict, res: dict) -> pd.DataFrame:
    st = res["start"]
    rows = {}
    for strat in STRATS:
        for day in WEEKDAYS:
            k = (strat, "semanal", day)
            p = perf(res["cal"][k], d["rf"], st)
            e = lagged_exposure(res["cal_w"][k], BASE_LAG)
            rows[f"{strat} · {day}"] = {"CAGR": p["CAGR"], "maxDD": p["maxDD"], "sharpe": p["sharpe"],
                                        "CAGR_1970_99": perf(res["cal"][k], d["rf"], *SUB)["CAGR"],
                                        "exposición_19oct87": 3 * e.get(CRASH, np.nan)}
    return pd.DataFrame(rows).T


def r12_table(D: dict, RES: dict) -> pd.DataFrame:
    rows = {}
    for name in MARKETS:
        d, res = D[name], RES[name]
        st = res["r12_start"]
        P = pd.DataFrame({off: perf(r, d["rf"], st) for off, r in res["r12"].items()}).T
        c0 = P.loc[0, "CAGR"]
        rows[name] = {"desde": st.date(), "CAGR_desfase_0 (R12)": c0, "CAGR_mín": P.CAGR.min(),
                      "CAGR_mediana": P.CAGR.median(), "CAGR_máx": P.CAGR.max(),
                      "amplitud_pp": (P.CAGR.max() - P.CAGR.min()) * 100,
                      "puesto_desfase_0": int((P.CAGR > c0).sum() + 1),
                      "maxDD_desfase_0": P.loc[0, "maxDD"], "maxDD_peor": P.maxDD.min(),
                      "maxDD_mejor": P.maxDD.max(),
                      "CAGR_½Kelly_diario_R13": perf(res["r12_daily"], d["rf"], st)["CAGR"]}
    return pd.DataFrame(rows).T.infer_objects()


def crash_table(d: dict, res: dict, sub: tuple[str, str] = SUB) -> pd.DataFrame:
    """Exposición y rentabilidad el 19-10-1987 y CAGR del subperiodo con y sin ese día."""
    rows = {}
    rf_ = d["rf"]
    u = d["u"]
    rows["1x comprar y mantener"] = {"exposición_19oct87": 1.0, "rent_19oct87": u.loc[CRASH],
                                     "CAGR_periodo": perf(u, rf_, *sub)["CAGR"],
                                     "CAGR_sin_19oct87": perf(neutralize(u, rf_, CRASH), rf_, *sub)["CAGR"]}
    for strat in STRATS:
        for kind in SIGNALS:
            w = res["w"][(strat, kind)]
            for lag in LAGS:
                r = res["lags"][(strat, kind, lag)]
                rows[f"{row_name(d, strat, kind)} · retraso {lag}"] = {
                    "exposición_19oct87": 3 * lagged_exposure(w, lag).loc[CRASH],
                    "rent_19oct87": r.loc[CRASH],
                    "CAGR_periodo": perf(r, rf_, *sub)["CAGR"],
                    "CAGR_sin_19oct87": perf(neutralize(r, rf_, CRASH), rf_, *sub)["CAGR"]}
    t = pd.DataFrame(rows).T
    t["depende_del_día_pp"] = (t["CAGR_sin_19oct87"] - t["CAGR_periodo"]) * 100
    base = f"{row_name(d, r13.ORIGINAL, BASE_SIGNAL)} · retraso {BASE_LAG}"
    t["es_la_original"] = ["sí" if i == base else "" for i in t.index]
    return t


def crossing_dates(d: dict, around: pd.Timestamp = CRASH, days: int = 10) -> dict:
    """Fecha del último cruce a la baja antes de `around` con cada índice de señal."""
    out = {}
    for kind in SIGNALS:
        c = crossings(signal_index(d, kind))
        c = c[(c.index <= around) & (c.index >= around - pd.Timedelta(days=days)) & (c["dirección"] == -1)]
        out[kind] = c.index[-1].date() if len(c) else None
    return out


FAMILIES = ("retrasos y señal", "semanales", "mensuales")


def variant_cagrs(d: dict, res: dict, strat: str, drop_day: pd.Timestamp | None = None
                  ) -> tuple[float, dict[str, np.ndarray]]:
    """CAGR de la base (señal original, retraso 2, revisión diaria) y de las 33 variantes
    alternativas por familias homogéneas: revisión diaria con otro retraso de la señal
    original o cualquier retraso de la de precio (7), semanal (5) y cada 21 sesiones (21),
    las dos últimas con la señal original. Con drop_day, ese día rinde la letra en todas
    las series."""
    st = res["start"]
    fix = (lambda r: neutralize(r, d["rf"], drop_day)) if drop_day is not None else (lambda r: r)
    c = lambda r: perf(fix(r), d["rf"], st)["CAGR"]
    base = c(res["lags"][(strat, BASE_SIGNAL, BASE_LAG)])
    fam = {"retrasos y señal": [c(r) for k, r in res["lags"].items()
                                if k[0] == strat and k != (strat, BASE_SIGNAL, BASE_LAG)],
           "semanales": [c(r) for k, r in res["cal"].items() if k[0] == strat and k[1] == "semanal"],
           "mensuales": [c(r) for k, r in res["cal"].items() if k[0] == strat and k[1] == "mensual"]}
    return base, {k: np.array(v) for k, v in fam.items()}


def sensitivity_summary(D: dict, RES: dict) -> pd.DataFrame:
    """La cifra base frente a las 33 variantes de ejecución alternativas: los otros 3
    retrasos con la señal original (rentabilidad total), los 4 con la de precio y los
    5 + 21 calendarios de revisión con la señal original. No son igual de probables ni de rápidas: 21 de las 33 son
    reglas revisadas cada 21 sesiones; el puesto por familias está en family_ranks."""
    rows = {}
    for name in MARKETS:
        d, res = D[name], RES[name]
        st = res["start"]
        for strat in STRATS:
            base, fam = variant_cagrs(d, res, strat)
            v = np.concatenate(list(fam.values()))
            row = {"CAGR_base": base, "variantes": len(v), "CAGR_mín": v.min(),
                   "CAGR_p25": np.percentile(v, 25), "CAGR_mediana": np.median(v),
                   "CAGR_p75": np.percentile(v, 75), "CAGR_máx": v.max(),
                   "base_supera_a": (v < base - 1e-12).mean()}
            if st < CRASH:
                b0, f0 = variant_cagrs(d, res, strat, CRASH)
                v0 = np.concatenate(list(f0.values()))
                row["CAGR_mediana_sin_19oct87"] = np.median(v0)
                row["base_supera_a_sin_19oct87"] = (v0 < b0 - 1e-12).mean()
            row["1x_comprar_y_mantener"] = metrics.cagr(d["u"].loc[st:])
            row["variantes_bajo_1x"] = int((v < row["1x_comprar_y_mantener"]).sum())
            rows[f"{name} · {strat}"] = row
    return pd.DataFrame(rows).T


def family_ranks(D: dict, RES: dict) -> pd.DataFrame:
    """Cuántas variantes de cada familia homogénea quedan por debajo de la base, con y sin
    el 19-10-1987 (este último, sólo donde la muestra lo incluye)."""
    rows = {}
    for name in MARKETS:
        d, res = D[name], RES[name]
        for strat in STRATS:
            for drop in ((None, CRASH) if res["start"] < CRASH else (None,)):
                base, fam = variant_cagrs(d, res, strat, drop)
                row = {"CAGR_base": base}
                row.update({f"{k} ({len(v)})": rank_str(base, v) for k, v in fam.items()})
                v = np.concatenate(list(fam.values()))
                row["todas (33)"] = rank_str(base, v)
                row.update({f"mediana_{k}": np.median(v_) for k, v_ in fam.items()})
                rows[f"{name} · {strat}" + (" · sin 19-10-1987" if drop is not None else "")] = row
    return pd.DataFrame(rows).T.infer_objects()


def crash_calendar(d: dict, res: dict) -> pd.DataFrame:
    rows = {}
    for strat in STRATS:
        for freq in ("semanal", "mensual"):
            keys = [k for k in res["cal"] if k[0] == strat and k[1] == freq]
            exp = np.array([3 * lagged_exposure(res["cal_w"][k], BASE_LAG).loc[CRASH] for k in keys])
            cg = np.array([perf(res["cal"][k], d["rf"], *SUB)["CAGR"] for k in keys])  # 1970–1999
            inside = exp > 0.05
            rows[f"{strat} · {freq}"] = {
                "desfases": len(keys), "expuestos_el_19oct87": int(inside.sum()),
                "CAGR_1970_99_expuestos": np.median(cg[inside]) if inside.any() else np.nan,
                "CAGR_1970_99_fuera": np.median(cg[~inside]) if (~inside).any() else np.nan,
                "CAGR_1970_99_mín": cg.min(), "CAGR_1970_99_máx": cg.max()}
    return pd.DataFrame(rows).T


# ───────────────────────────── figuras ─────────────────────────────

def fig_lag(D: dict, RES: dict, rep: Report):
    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    for ax, name in zip(axes, MARKETS):
        d, res = D[name], RES[name]
        st = res["start"]
        for strat in STRATS:
            for kind, ls, mk in ((BASE_SIGNAL, "-", "o"), (r13.PRICE, "--", "s")):
                y = [perf(res["lags"][(strat, kind, lag)], d["rf"], st)["CAGR"] * 100 for lag in LAGS]
                ax.plot(LAGS, y, ls=ls, marker=mk, ms=8, lw=2, color=COLORS[strat],
                        label=f"{strat.replace(' (original)', '')}, señal de {KIND_LABEL[kind]}")
        ax.axhline(metrics.cagr(d["u"].loc[st:]) * 100, color="k", lw=1.2, ls=":",
                   label="1x comprar y mantener")
        ax.set(title=f"{SHOW[name]} (desde {st.year})", ylabel="Rentabilidad anual (%)")
        ax.grid(alpha=0.3)
        comma_axis(ax)
    axes[-1].set(xlabel="Retraso de ejecución (1 = mismo cierre; 2 = cierre siguiente, la original; "
                        "3 y 5 = dos y cuatro sesiones después)", xticks=LAGS)
    axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10.5)
    fig.suptitle("Rentabilidad anual compuesta según el retraso de ejecución")
    rep.fig(fig, "cagr_retraso",
            "Rentabilidad anual compuesta de la LRS 3x (azul) y de la LRS-Kelly (rojo) con la orden "
            "ejecutada 1, 2, 3 o 5 sesiones después de la señal: línea continua, señal de rentabilidad "
            "total (la original); discontinua, señal de precio (en el Nasdaq-100 y los semiconductores, "
            "precio aproximado: la rentabilidad total menos un dividendo constante). La línea de puntos "
            "es comprar y mantener el 1x.")


def fig_lag_cum(D: dict, RES: dict, rep: Report):
    fig, axes = plt.subplots(3, 1, figsize=(10, 13))
    for ax, name in zip(axes, MARKETS):
        d, res = D[name], RES[name]
        st = res["start"]
        base = res["lags"][("LRS 3x (original)", BASE_SIGNAL, BASE_LAG)]
        for lag, c in ((1, "C2"), (3, "C1"), (5, "C3")):
            x = log_diff(res["lags"][("LRS 3x (original)", BASE_SIGNAL, lag)], base, st)
            ax.semilogy(np.exp(x.cumsum()), color=c, lw=1.5, label=f"retraso {lag} frente a 2")
        ax.axhline(1, color="0.5", lw=0.8)
        ax.yaxis.set_major_locator(plt.matplotlib.ticker.LogLocator(base=2))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×".replace(".", ",")))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        if st < CRASH:
            ax.axvline(CRASH, color="0.3", ls=":", lw=1)
            ax.text(CRASH, ax.get_ylim()[1], " 19-10-1987", va="top", fontsize=11, color="0.3")
        ax.set(title=f"{SHOW[name]}: LRS 3x original (señal de rentabilidad total)",
               ylabel="Riqueza relativa (log)")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=11)
    fig.suptitle("Riqueza acumulada de cada retraso respecto al cierre siguiente")
    rep.fig(fig, "coste_retraso_acumulado",
            "Riqueza de la LRS 3x original ejecutada con retraso 1, 3 o 5 dividida por la ejecutada "
            "al cierre siguiente (retraso 2), en escala logarítmica (1× = igual). Un salto vertical es un "
            "día suelto en que un retraso estaba dentro y el otro fuera, como el 19-10-1987; una "
            "pendiente sostenida es una diferencia repartida entre muchos cruces.")


def fig_calendar(D: dict, RES: dict, rep: Report):
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 11.5))
    groups = [(s, f) for s in STRATS for f in ("semanal", "mensual")]
    for ax, name in zip(axes, MARKETS):
        d, res = D[name], RES[name]
        st = res["start"]
        for i, (strat, freq) in enumerate(groups):
            keys = [k for k in res["cal"] if k[0] == strat and k[1] == freq]
            y = np.array([perf(res["cal"][k], d["rf"], st)["CAGR"] * 100 for k in keys])
            jitter = np.linspace(-0.18, 0.18, len(y))
            ax.scatter(i + jitter, y, s=50, color=COLORS[strat], alpha=0.75, edgecolor="white", lw=0.8,
                       zorder=3)
            daily = perf(res["lags"][(strat, BASE_SIGNAL, BASE_LAG)], d["rf"], st)["CAGR"] * 100
            ax.plot([i - 0.3, i + 0.3], [daily] * 2, color="k", lw=2, zorder=4,
                    label="revisión diaria" if i == 0 else None)
        ax.axhline(metrics.cagr(d["u"].loc[st:]) * 100, color="k", lw=1, ls=":",
                   label="1x comprar y mantener")
        ax.set_xticks(range(len(groups)),
                      [f"{s.replace(' (original)', '')}\n{f}" for s, f in groups])
        ax.set(title=f"{SHOW[name]} (desde {st.year})", ylabel="Rentabilidad anual (%)")
        ax.grid(alpha=0.3, axis="y")
        comma_axis(ax)
    axes[0].legend(loc="lower left")
    fig.suptitle("Suerte del calendario: un punto por día de revisión posible")
    rep.fig(fig, "abanico_calendario",
            "Rentabilidad anual compuesta de cada desfase posible del calendario de revisión: 5 para la "
            "revisión semanal (de lunes a viernes) y 21 para la revisión cada 21 sesiones; la raya negra "
            "es la revisión diaria y la línea de puntos comprar y mantener el 1x. Dentro de cada grupo los "
            "puntos van ordenados de izquierda a derecha por desfase (lunes a viernes; 0 a 20). Señal "
            "original (rentabilidad total) y orden siempre al cierre siguiente.")


def fig_r12(D: dict, RES: dict, rep: Report):
    fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=True)
    for ax, name in zip(axes, MARKETS):
        d, res = D[name], RES[name]
        st = res["r12_start"]
        y = np.array([perf(res["r12"][o], d["rf"], st)["CAGR"] * 100 for o in range(H)])
        ax.plot(range(H), y, color="0.6", lw=1, zorder=2)
        ax.scatter(range(1, H), y[1:], s=60, color="0.45", edgecolor="white", zorder=3,
                   label="otros desfases")
        ax.scatter([0], [y[0]], s=110, color="C3", edgecolor="k", zorder=4,
                   label="desfase 0 (el de R12)")
        daily = perf(res["r12_daily"], d["rf"], st)["CAGR"] * 100
        ax.axhline(daily, color="k", ls="--", lw=1.2, label="½ Kelly GJR revisado a diario (R13)")
        ax.set(title=f"{SHOW[name]} (desde {st.year})", ylabel="Rentabilidad anual (%)")
        ax.grid(alpha=0.3, axis="y")
        comma_axis(ax)
    axes[0].legend(loc="lower left", fontsize=10)
    axes[-1].set(xlabel="Desfase de la revisión (sesiones desde el primer día de la serie)",
                 xticks=range(0, H, 2))
    fig.suptitle("Medio Kelly GJR de R12: los 21 calendarios de revisión posibles")
    rep.fig(fig, "r12_desfases",
            "Rentabilidad anual compuesta del medio Kelly con varianza GJR-GARCH de R12 (revisión cada 21 sesiones) para cada "
            "uno de los 21 desfases posibles; en rojo el que usa R12 (keep[::21]). La línea discontinua "
            "es la misma regla revisada a diario con el filtro de reajuste mínimo de R13.")


def fig_crossings(RES: dict, rep: Report):
    fig, axes = plt.subplots(2, 1, figsize=(10, 11))
    for name in MARKETS:
        c = RES[name]["cruces"]
        a = np.sort(c["distancia"].abs().values) * 100
        axes[0].step(a, np.arange(1, len(a) + 1) / len(a) * 100, where="post", color=MKT_COLORS[name],
                     lw=2, label=f"{SHOW[name]} ({len(a)} cruces)")
        dur = np.sort(c["duración"].dropna().values)
        axes[1].step(dur, np.arange(1, len(dur) + 1) / len(dur) * 100, where="post",
                     color=MKT_COLORS[name], lw=2, label=SHOW[name])
        rw = RES[name]["cruces_rw"]
        axes[1].plot([10], [rw["se_deshace_≤10"] * 100], marker="D", ms=9, color=MKT_COLORS[name],
                     mec="k", ls="none", label=f"{SHOW[name]}: paseo aleatorio, ≤ 10 sesiones")
    axes[0].set(xlim=(0, 4), xlabel="|índice / media de 200 − 1| el día del cruce (%)",
                ylabel="% de cruces", title="Distancia a la media el día en que cambia la señal")
    for x in (0.5, 1, 2):
        axes[0].axvline(x, color="0.6", lw=0.8, ls=":")
    axes[1].set(xscale="log", xlabel="Sesiones hasta el siguiente cruce", ylabel="% de cruces",
                title="Cuánto dura el nuevo estado")
    for k in NOISE_DAYS:
        axes[1].axvline(k, color="0.6", lw=0.8, ls=":")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)
    fig.suptitle("Cruces de la media de 200 sesiones (índice de rentabilidad total)")
    rep.fig(fig, "cruces_ruido",
            "Arriba, distribución acumulada de la distancia del índice de rentabilidad total (la señal "
            "original) a su media el día en que cambia "
            "la señal; abajo, de las sesiones que tarda en llegar el cruce siguiente (los rombos marcan "
            "la fracción deshecha en 10 sesiones o menos en paseos aleatorios con las mismas "
            "rentabilidades diarias remuestreadas).")


def fig_1987(d: dict, res: dict, crash: pd.DataFrame, rep: Report):
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 10.5))
    short = {BASE_SIGNAL: "señal total (original)", r13.PRICE: "señal de precio"}
    a, b = "1987-09-21", "1987-11-06"
    xd = crossing_dates(d)
    ax = axes[0]
    for kind, c in ((BASE_SIGNAL, "C0"), (r13.PRICE, "C1")):
        s_ = signal_index(d, kind)
        dist = (s_ / s_.rolling(200).mean() - 1).loc[a:b] * 100
        ax.plot(dist, color=c, lw=2, marker="o", ms=4,
                label="señal de rentabilidad total (original)" if kind == BASE_SIGNAL else "señal de precio")
        cr = xd[kind]
        if cr is not None:
            t = pd.Timestamp(cr)
            ax.scatter([t], [dist.loc[t]], s=160, facecolor="none", edgecolor=c, lw=2.5, zorder=5)
    ax.axhline(0, color="k", lw=1)
    ax.axvline(CRASH, color="0.4", lw=1, ls=":")
    # arriba, a la derecha de la línea del crac, donde no hay datos (tras el 19 todo es negativo)
    ax.annotate("lunes 19-10-1987", (CRASH, 9), xytext=(6, 0), textcoords="offset points",
                fontsize=11, va="center", color="0.3")
    ax.set(ylabel="Distancia a la media de 200 (%)",
           title=f"S&P 500: la rentabilidad total cruza el {xd[BASE_SIGNAL]:%d-%m} y el precio el "
                 f"{xd[r13.PRICE]:%d-%m}")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02))
    ax.grid(alpha=0.3)
    comma_axis(ax)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    ax = axes[1]
    width = 0.2
    for j, (strat, kind) in enumerate([(s, k) for s in STRATS for k in SIGNALS]):
        rows = [f"{row_name(d, strat, kind)} · retraso {lag}" for lag in LAGS]
        y = crash.loc[rows, "CAGR_periodo"].values * 100
        y0 = crash.loc[rows, "CAGR_sin_19oct87"].values * 100
        x = np.arange(len(LAGS)) + (j - 1.5) * width
        col = COLORS[strat]
        bars = ax.bar(x, y, width=width * 0.92, color=col, alpha=1.0 if kind == BASE_SIGNAL else 0.45,
                      label=f"{strat.replace(' (original)', '')}, {short[kind]}")
        if kind == BASE_SIGNAL:   # la base: señal original, orden al cierre siguiente
            base = bars[LAGS.index(BASE_LAG)]
            base.set_edgecolor("k")
            base.set_linewidth(2.2)
        ax.scatter(x, y0, marker="_", s=250, color="k", lw=2.5, zorder=4,
                   label="con el 19-10-1987 en letras" if j == 0 else None)
    ax.axhline(crash.loc["1x comprar y mantener", "CAGR_periodo"] * 100, color="k", ls=":", lw=1.2,
               label="1x comprar y mantener")
    ax.set_xticks(range(len(LAGS)), ["1: mismo\ncierre", "2: cierre\nsiguiente\n(la original)",
                                     "3: dos\nsesiones\ndespués", "5: cuatro\nsesiones\ndespués"])
    ax.set(ylabel="Rentabilidad anual (%)", xlabel="Retraso de ejecución",
           title="1970–1999, con y sin el 19 de octubre de 1987", ylim=(0, None))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=11, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    comma_axis(ax)
    rep.fig(fig, "octubre_1987",
            "Arriba, distancia del S&P 500 a su media de 200 sesiones en el otoño de 1987 medida sobre "
            "el índice de rentabilidad total (la señal original) y sobre el de precio (círculos: día del "
            "cruce a la baja). Abajo, rentabilidad anual compuesta de 1970–1999 de cada regla según el "
            "retraso (barras; la de borde negro es la LRS 3x original con orden al cierre siguiente) y la "
            "que tendría si el 19-10-1987 hubiera estado en letras (rayas negras): la distancia entre "
            "ambas es lo que depende de ese día.")


# ───────────────────────────── informe ─────────────────────────────

def pct_cols(df: pd.DataFrame) -> set[str]:
    keys = ("CAGR", "maxDD", "Δ", "IC90", "vol")
    return {c for c in df.columns if c.startswith(keys)}


def num(x: float, k: int = 1) -> str:
    return f"{x:.{k}f}".replace(".", ",").replace("-", "−")


def txt(s_: str) -> str:
    """Cadena de tabla (p. ej. un intervalo) con coma decimal y signo menos tipográfico."""
    return s_.replace(".", ",").replace("-", "−")


def pc(x: float, k: int = 1) -> str:
    """Porcentaje con coma decimal para el texto."""
    return f"{x * 100:.{k}f}%".replace(".", ",").replace("-", "−")


def pp(x: float, k: int = 1, sign: bool = True) -> str:
    """Puntos porcentuales (x en fracción) con signo y coma decimal."""
    return f"{x * 100:{'+' if sign else ''}.{k}f} pp".replace(".", ",").replace("-", "−")


def run():
    rep = Report("r15_execution", "R15 — Ejecución y suerte del calendario")
    rng = np.random.default_rng(SEED)
    D = r13.load()
    D.pop("_ndx")
    RES = {name: compute_market(D[name], r13.var_forecast(D[name]), rng) for name in MARKETS}
    sp, rs = D["S&P 500"], RES["S&P 500"]

    # ── cálculo de todas las tablas ──
    LAG = {n: lag_table(D[n], RES[n], rng) for n in MARKETS}
    CONC = {n: lag_concentration(D[n], RES[n], rng) for n in MARKETS}
    PP90 = {n: pre_post_1990(D[n], RES[n], rng) for n in MARKETS}
    AC = autocorr_table(D, RES)
    NXT = {n: next_day_after_cross(D[n], RES[n]) for n in MARKETS}
    CAL = {n: calendar_table(D[n], RES[n]) for n in MARKETS}
    cal_sp_no87 = calendar_table(sp, rs, CRASH)
    wk = weekday_table(sp, rs)
    SIG = {n: signal_table(D[n], RES[n]) for n in MARKETS}
    cr = {}
    for n in MARKETS:
        real = crossing_summary(RES[n]["cruces"], RES[n]["years"])
        cr[f"{n} · real"] = real
        cr[f"{n} · paseo aleatorio"] = RES[n]["cruces_rw"]
    ct = pd.DataFrame(cr).T
    dcols = ["|dist|_mediana", "|dist|_med_ruido(≤10)", "|dist|_med_duraderos(>21)"]
    ct_show = ct.copy()
    ct_show[dcols] = ct_show[dcols] * 100
    ct_show = ct_show.rename(columns={c: c + " (%)" for c in dcols})
    real_rows = [f"{n} · real" for n in MARKETS]
    band = {n: perf(r13.run_weights(D[n], r13.above_sma(signal_index(D[n]), band=r13.BAND)), D[n]["rf"],
                    RES[n]["start"])["CAGR"] for n in MARKETS}
    t12 = r12_table(D, RES)
    crash = crash_table(sp, rs)
    crash_ndx = crash_table(D["Nasdaq-100"], RES["Nasdaq-100"], ("1985-10-01", "1999-12-31"))
    cc = crash_calendar(sp, rs)
    summ = sensitivity_summary(D, RES)
    FR = family_ranks(D, RES)
    xd = {n: crossing_dates(D[n]) for n in ("S&P 500", "Nasdaq-100")}

    # ── valores para el texto ──
    L3, K = "LRS 3x (original)", "LRS-Kelly"
    BS = BASE_SIGNAL
    PXS = row_name(sp, L3, r13.PRICE)     # «LRS 3x (señal de precio)»: ^GSPC real en el S&P 500
    g = lambda n, s_, lag, col="CAGR": LAG[n].loc[f"{s_} · retraso {lag}", col]
    cg = lambda n, s_, lag, col: CONC[n].loc[f"{s_} · {lag} frente a 2", col]
    ci_rows = [(n, i) for n in MARKETS for i in LAG[n].index if pd.notna(LAG[n].loc[i, "IC90_bajo"])]
    excl = [(n, i) for n, i in ci_rows if LAG[n].loc[i, "IC90_bajo"] > 0 or LAG[n].loc[i, "IC90_alto"] < 0]
    sp_excl = [i for n, i in excl if n == "S&P 500"]
    calr = lambda n, s_, f: CAL[n].loc[f"{s_} · {f}", "amplitud_pp"]
    calv = lambda n, s_, f, col: CAL[n].loc[f"{s_} · {f}", col]
    n_below = lambda n, s_, f: int(sum(perf(r, D[n]["rf"], RES[n]["start"])["CAGR"] <
                                       calv(n, s_, "diaria", "CAGR_mín")
                                       for k, r in RES[n]["cal"].items() if k[0] == s_ and k[1] == f))
    t_sp = SIG["S&P 500"][0]
    sm = lambda n, s_, col: summ.loc[f"{n} · {s_}", col]
    ex = crash.loc
    exn = crash_ndx.loc
    n1 = NXT["S&P 500"]
    flip_list = [(n, i) for n in MARKETS for i in CONC[n].index
                 if np.sign(CONC[n].loc[i, "Δcrec_anual"]) != np.sign(CONC[n].loc[i, "Δ_sin_10_mayores"])]
    flips = len(flip_list)
    n_conc = sum(len(CONC[n]) for n in MARKETS)
    flip_txt = "; ".join(f"{SHOW[n]}, {i.replace(' (original)', '')} ({pp(CONC[n].loc[i, 'Δcrec_anual'])} → "
                         f"{pp(CONC[n].loc[i, 'Δ_sin_10_mayores'])})" for n, i in flip_list)
    pq = lambda n, s_, lag, col: PP90[n].loc[f"{s_} · {lag} frente a 2", col]
    SPX, NDX, SMH = MARKETS
    arrow = lambda n, lag: (f"{pp(cg(n, L3, lag, 'Δcrec_anual'))} → "
                            f"{pp(cg(n, L3, lag, 'Δ_sin_10_mayores'))}")
    # aportación por año de la diferencia del retraso 3 en los semiconductores (LRS 3x)
    x_sm3 = log_diff(RES[SMH]["lags"][(L3, BS, 3)], RES[SMH]["lags"][(L3, BS, BASE_LAG)],
                     RES[SMH]["start"])
    yc_sm3 = yearly_contribution(x_sm3)
    big_years = "; ".join(f"{y} {pp(v)}" for y, v in yc_sm3[yc_sm3.abs() >= 0.005].items()) or "ninguno"
    # familias de variantes (S&P 500)
    fr = lambda row, col: FR.loc[row, col]
    fam_cols = [c for c in FR.columns if c.startswith(FAMILIES)]
    sp3, sp3_0 = f"{SPX} · {L3}", f"{SPX} · {L3} · sin 19-10-1987"
    up89, dn89 = n1.loc["hasta 1989 · cruce al alza"], n1.loc["hasta 1989 · cruce a la baja"]
    dn90 = n1.loc["desde 1990 · cruce a la baja"]
    ac = lambda row: AC.loc[row, "ρ1"]
    ndx_crash_log = log_diff(RES[NDX]["lags"][(L3, BS, 1)], RES[NDX]["lags"][(L3, BS, BASE_LAG)],
                             RES[NDX]["start"]).loc[CRASH]
    sp1_ci = [PP90[SPX].attrs["ci"][f"{s_} · 1 frente a 2"][t] for s_ in STRATS
              for t in ("hasta_1989", "desde_1990")]
    sp1_excl = sum(lo > 0 or hi < 0 for lo, hi in sp1_ci)
    pp_all = [(n, row, t_, lo, hi) for n in (SPX, NDX) for row, dct in PP90[n].attrs["ci"].items()
              for t_, (lo, hi) in dct.items()]
    pp_excl = [z for z in pp_all if z[3] > 0 or z[4] < 0]
    tramo_txt = {"hasta_1989": "hasta 1989", "hasta_1989_sin_crac": "hasta 1989 sin el crac",
                 "desde_1990": "desde 1990"}
    pp_excl_txt = "; ".join(f"{SHOW[n]}, {row.replace(' (original)', '')}, {tramo_txt[t_]}"
                            for n, row, t_, lo, hi in pp_excl)
    # el intervalo más alejado del cero (en unidades de su semianchura), para el texto
    far = max(pp_excl, key=lambda z: max(z[3], -z[4]) / (z[4] - z[3])) if pp_excl else None
    gains = lambda x: "gana" if x > 0 else "pierde"  # noqa: E731
    sig_ci = lambda lo, hi: lo > 0 or hi < 0  # noqa: E731
    l1_ci = PP90[SPX].attrs["ci"][f"{L3} · 1 frente a 2"]

    def continuation(up: pd.Series, dn: pd.Series) -> str:
        """Lectura del día siguiente al cruce: hay continuación si rinde más de lo normal tras un
        cruce al alza (t > 0) y menos tras uno a la baja (t < 0)."""
        cu, cd = up["t_frente_a_día_normal"] > 0, dn["t_frente_a_día_normal"] < 0
        if cu and cd:
            return "los dos lados apuntan a la continuación que cabría esperar de precios desfasados"
        if cu:
            return "sólo el lado alcista apunta a la continuación que cabría esperar de precios desfasados"
        if cd:
            return "sólo el lado bajista apunta a la continuación que cabría esperar de precios desfasados"
        return "ninguno de los dos lados apunta a una continuación"

    def exits(dd: dict, kind: str, lag: int) -> pd.Timestamp:
        """Sesión en cuyo cierre la regla deja el 3x tras el último cruce a la baja de octubre de 1987."""
        idx = D[dd]["u"].index
        return idx[idx.get_loc(pd.Timestamp(xd[dd][kind])) + lag - 1]

    wk_out = [day for day in WEEKDAYS if wk.loc[f"{L3} · {day}", "exposición_19oct87"] < 0.05]
    n_week_out = int(cc.loc[f"{L3} · semanal", "desfases"] - cc.loc[f"{L3} · semanal", "expuestos_el_19oct87"])
    n_month_out = int(cc.loc[f"{L3} · mensual", "desfases"] - cc.loc[f"{L3} · mensual", "expuestos_el_19oct87"])
    ndx_out = [lag for lag in LAGS if exn[f"{L3} · {BS} · retraso {lag}", "exposición_19oct87"] < 0.05]
    kelly_less = all(calr(n, K, "mensual") < calr(n, L3, "mensual") for n in MARKETS)
    base2, px2 = t_sp.loc[f"{L3} · {BS}", "CAGR_retraso_2"], t_sp.loc[PXS, "CAGR_retraso_2"]
    base2_0 = t_sp.loc[f"{L3} · {BS}", "CAGR_retraso_2_sin_19oct87"]
    px2_0 = t_sp.loc[PXS, "CAGR_retraso_2_sin_19oct87"]
    share87 = 1 - (px2_0 - base2_0) / (px2 - base2) if abs(px2 - base2) > 1e-9 else np.nan
    dist87 = r13.distance_1987(sp)
    # el 19-10-1987 en la diferencia de cada retraso frente al 2 (LRS 3x, S&P 500): solo se citan
    # los retrasos a los que ese día aporta algo (±0,05 pp o más); con la señal original, el 2 y
    # el 3 pueden haber pasado ese lunes igual de dentro
    exp87 = {lag: ex[f"{L3} · {BS} · retraso {lag}", "exposición_19oct87"] for lag in LAGS}
    lag_de = {1: "del mismo cierre", 3: "del retraso 3", 5: "del retraso 5"}
    lag_al = {1: "al mismo cierre", 3: "al retraso 3", 5: "al retraso 5"}
    c87 = [(lag, cg(SPX, L3, lag, "Δ_19oct87"), cg(SPX, L3, lag, "Δcrec_anual")) for lag in (1, 3, 5)]
    on87 = [z for z in c87 if abs(z[1]) >= 5e-4]
    off87 = [z[0] for z in c87 if abs(z[1]) < 5e-4]
    zero87 = all(abs(z[1]) < 1e-9 for z in c87 if z[0] in off87)   # misma exposición ese lunes
    if on87:
        read87 = "el 19-10-1987 aporta " + " y ".join(f"{pp(a)} de los {pp(b)} {lag_de[lag]}"
                                                      for lag, a, b in on87)
        if off87 and zero87:   # «y no aporta nada al retraso 3» / «ni al retraso 3 ni al retraso 5»
            read87 += (", y no aporta nada " + ("ni " if len(off87) > 1 else "")
                       + " ni ".join(lag_al[lag] for lag in off87))
        elif off87:
            read87 += ", y menos de 0,05 pp " + " y ".join(lag_al[lag] for lag in off87)
    else:
        read87 = ("el 19-10-1987 no aporta nada a la diferencia de ningún retraso frente al 2" if zero87 else
                  "el 19-10-1987 aporta menos de 0,05 pp a la diferencia de cada retraso frente al 2")
    rw_undo = ct.loc[[f"{n} · paseo aleatorio" for n in MARKETS], "se_deshace_≤10"]
    real_undo = ct.loc[real_rows, "se_deshace_≤10"]

    rep.p(__doc__.split("\n", 2)[2])
    rep.h("Lo esencial")
    bullets = [
        f"**Retraso.** En el S&P 500 (desde {rs['start'].year}) la LRS 3x original rinde "
        f"{pc(g(SPX, L3, 1))} con la orden en el mismo cierre y {pc(g(SPX, L3, 2))} al cierre siguiente "
        f"(la base); la LRS-Kelly, {pc(g(SPX, K, 1))} y {pc(g(SPX, K, 2))}. Retrasar la orden a 3 o 5 "
        f"sesiones cambia el crecimiento anual de la LRS 3x en {pp(cg(SPX, L3, 3, 'Δcrec_anual'))} y "
        f"{pp(cg(SPX, L3, 5, 'Δcrec_anual'))}. Del {pp(cg(SPX, L3, 1, 'Δcrec_anual'))} del mismo cierre, el "
        f"19-10-1987 aporta {pp(cg(SPX, L3, 1, 'Δ_19oct87'))}"
        + (": con la señal original, la orden al cierre siguiente soportó ese lunes y la del mismo cierre no"
           if exp87[BASE_LAG] > 0.05 and exp87[1] < 0.05 else "")
        + ". Sin los diez días en que más difieren, las "
        f"diferencias de los retrasos 1, 3 y 5 frente al 2 pasan a {pp(cg(SPX, L3, 1, 'Δ_sin_10_mayores'))}, "
        f"{pp(cg(SPX, L3, 3, 'Δ_sin_10_mayores'))} y {pp(cg(SPX, L3, 5, 'Δ_sin_10_mayores'))}. "
        + ("Ningún intervalo del 90% del S&P 500 excluye el cero." if not sp_excl else
           f"Intervalos del 90% que excluyen el cero en el S&P 500: {', '.join(sp_excl)}."),
        f"**El mismo cierre.** En la LRS 3x del S&P 500, ejecutar en el mismo cierre en vez de al "
        f"siguiente da {pp(pq(SPX, L3, 1, 'Δ_hasta_1989'))} al año hasta 1989 (IC90 {txt(pq(SPX, L3, 1, 'IC90_hasta_1989'))}; "
        f"sin el 16 y el 19-10-1987, {pp(pq(SPX, L3, 1, 'Δ_hasta_1989_sin_crac'))}, IC90 "
        f"{txt(pq(SPX, L3, 1, 'IC90_hasta_1989_sin_crac'))}) y "
        f"{pp(pq(SPX, L3, 1, 'Δ_desde_1990'))} desde 1990 (IC90 {txt(pq(SPX, L3, 1, 'IC90_desde_1990'))}). "
        + ("Ninguno de esos intervalos excluye el cero. " if not any(sig_ci(lo, hi) for lo, hi in l1_ci.values())
           else "Excluyen el cero: " + ", ".join(tramo_txt[t_] for t_, (lo, hi) in l1_ci.items() if sig_ci(lo, hi))
           + ". ")
        + f"La autocorrelación a un día del índice fue positiva "
        f"hasta 1989 (ρ₁ = {num(ac('S&P 500 · hasta 1989'), 2)}) y negativa después "
        f"({num(ac('S&P 500 · desde 1990'), 2)}). Eso es compatible con que los precios desfasados "
        f"favorecieran al mismo cierre antes de 1990, una ventaja que no sería operable: con precios "
        f"desfasados el cierre del índice no se podía negociar. El día siguiente a un cruce al alza rindió {num(up89['rent_día_siguiente_pb'])} pb frente a "
        f"{num(up89['rent_media_todos_los_días_pb'])} de un día normal (t = "
        f"{num(up89['t_frente_a_día_normal'])}) y tras un cruce a la baja "
        f"{num(dn89['rent_día_siguiente_pb'])} pb (t = {num(dn89['t_frente_a_día_normal'])}): "
        f"{continuation(up89, dn89)}. En el Nasdaq-100 el mismo cierre {gains(cg(NDX, L3, 1, 'Δcrec_anual'))} "
        f"{pp(abs(cg(NDX, L3, 1, 'Δcrec_anual')), sign=False)} al año. "
        f"De 1985-10 a 1989 la diferencia es de {pp(pq(NDX, L3, 1, 'Δ_hasta_1989'))} al año (IC90 "
        f"{txt(pq(NDX, L3, 1, 'IC90_hasta_1989'))}): "
        f"{pp(pq(NDX, L3, 1, 'Δ_hasta_1989') - pq(NDX, L3, 1, 'Δ_hasta_1989_sin_crac'))} "
        f"salen del crac y {pp(pq(NDX, L3, 1, 'Δ_hasta_1989_sin_crac'))} (IC90 "
        f"{txt(pq(NDX, L3, 1, 'IC90_hasta_1989_sin_crac'))}) del resto de esos cuatro años. Desde 1990 es "
        f"de {pp(pq(NDX, L3, 1, 'Δ_desde_1990'))} (IC90 {txt(pq(NDX, L3, 1, 'IC90_desde_1990'))}).",
        f"**Ningún retraso es sistemáticamente mejor.** De los {len(ci_rows)} intervalos del 90% "
        f"(3 mercados × 2 reglas × 3 retrasos frente al 2), {len(excl)} excluyen el cero "
        f"({'; '.join(f'{n}: {i}' for n, i in excl) or 'ninguno'}); por azar se esperarían "
        f"{num(0.1 * len(ci_rows))}. En los semiconductores el retraso 3 da {pp(cg(SMH, L3, 3, 'Δcrec_anual'))} "
        f"al año a la LRS 3x: del 15-02 al 30-04 de 2020 salen {pp(cg(SMH, L3, 3, 'Δ_feb_abr_2020'))}; sin 2020 "
        f"quedan {pp(cg(SMH, L3, 3, 'Δ_sin_2020'))} al año (IC90 {txt(cg(SMH, L3, 3, 'IC90_sin_2020'))}; "
        f"{cg(SMH, L3, 3, 'años_a_favor')} años a favor). En el mismo mercado el retraso 5 da "
        f"{pp(cg(SMH, L3, 5, 'Δcrec_anual'))} y el 1, {pp(cg(SMH, L3, 1, 'Δcrec_anual'))}. Quitar los diez "
        f"días de mayor diferencia cambia el signo en {flips} de las {n_conc} comparaciones"
        + (f": {flip_txt}. En esas, la diferencia la deciden unos pocos días." if flips else "."),
        f"**Suerte del calendario.** Revisando la LRS 3x cada 21 sesiones, la rentabilidad anual va de "
        f"{pc(calv(SPX, L3, 'mensual', 'CAGR_mín'))} a {pc(calv(SPX, L3, 'mensual', 'CAGR_máx'))} "
        f"según el desfase en el S&P 500 ({num(calr(SPX, L3, 'mensual'))} pp), de "
        f"{pc(calv(NDX, L3, 'mensual', 'CAGR_mín'))} a {pc(calv(NDX, L3, 'mensual', 'CAGR_máx'))} "
        f"en el Nasdaq-100 y de {pc(calv(SMH, L3, 'mensual', 'CAGR_mín'))} a "
        f"{pc(calv(SMH, L3, 'mensual', 'CAGR_máx'))} en los semiconductores. Con revisión semanal el "
        f"abanico es de {num(calr(SPX, L3, 'semanal'))}, {num(calr(NDX, L3, 'semanal'))} y "
        f"{num(calr(SMH, L3, 'semanal'))} pp. La LRS-Kelly {'es menos sensible' if kelly_less else 'no es siempre menos sensible'} "
        f"(mensual: {num(calr(SPX, K, 'mensual'))}, {num(calr(NDX, K, 'mensual'))} y "
        f"{num(calr(SMH, K, 'mensual'))} pp). En el S&P 500 la LRS 3x diaria supera a "
        f"{n_below(SPX, L3, 'semanal')} de los 5 calendarios semanales y a "
        f"{n_below(SPX, L3, 'mensual')} de los 21 mensuales.",
        f"**R12 eligió su calendario sin saberlo.** El medio Kelly GJR de R12 (revisión cada 21 sesiones "
        f"contadas desde el primer día de la serie) rinde {pc(t12.loc[SPX, 'CAGR_desfase_0 (R12)'])} "
        f"en el S&P 500 y los otros 20 calendarios dan entre {pc(t12.loc[SPX, 'CAGR_mín'])} y "
        f"{pc(t12.loc[SPX, 'CAGR_máx'])} (puesto {t12.loc[SPX, 'puesto_desfase_0']} de 21); "
        f"en el Nasdaq-100 queda en el puesto {t12.loc[NDX, 'puesto_desfase_0']} y en los "
        f"semiconductores en el {t12.loc[SMH, 'puesto_desfase_0']}, con abanicos de "
        f"{num(t12.loc[NDX, 'amplitud_pp'])} y {num(t12.loc[SMH, 'amplitud_pp'])} pp.",
        f"**Rentabilidad total o precio.** La señal original, sobre la rentabilidad total, está dentro más "
        f"a menudo que la de precio (discrepan el {pc(SIG[SPX][1])} de las sesiones en el S&P 500) y sale más "
        f"tarde. Con orden al cierre siguiente la LRS 3x original rinde {pc(base2)} y la de precio "
        f"{pc(px2)}; con el 19-10-1987 en letras, {pc(base2_0)} y {pc(px2_0)}"
        + (f": ese día explica el {pc(share87, 0)} de la diferencia." if np.isfinite(share87) else "."),
        f"**{'La mayoría de los cruces son ruido' if real_undo.min() > 0.5 else 'Muchos cruces son ruido'}.** "
        f"En el S&P 500 el "
        f"{pc(ct.loc['S&P 500 · real', '|dist|<0.5%'], 0)} de los cruces de la media se produce a menos "
        f"del 0,5% de ella y el {pc(ct.loc['S&P 500 · real', 'se_deshace_≤10'], 0)} se deshace en diez "
        f"sesiones o menos; en un paseo aleatorio con las mismas rentabilidades diarias, el "
        f"{pc(ct.loc['S&P 500 · paseo aleatorio', 'se_deshace_≤10'], 0)}. Los cruces que acaban durando "
        f"más de un mes empiezan a una distancia mediana de "
        f"{pc(ct.loc['S&P 500 · real', '|dist|_med_duraderos(>21)'], 2)}, frente a "
        f"{pc(ct.loc['S&P 500 · real', '|dist|_med_ruido(≤10)'], 2)} los que se deshacen (en el paseo "
        f"aleatorio, {pc(ct.loc['S&P 500 · paseo aleatorio', '|dist|_med_duraderos(>21)'], 2)} frente a "
        f"{pc(ct.loc['S&P 500 · paseo aleatorio', '|dist|_med_ruido(≤10)'], 2)}), y el "
        f"{pc(ct.loc['S&P 500 · real', 'duraderos_con_|dist|<1%'], 0)} de ellos también empieza a menos "
        "del 1%: el día del cruce apenas se distinguen. "
        f"El mercado real cruza {num(ct.loc['S&P 500 · real', 'cruces_año'])} veces al año, frente "
        f"a {num(ct.loc['S&P 500 · paseo aleatorio', 'cruces_año'])} en el paseo aleatorio.",
        f"**Octubre de 1987.** El índice de rentabilidad total del S&P 500 cruzó su media el "
        f"{xd[SPX][BS]:%d-%m} y el de precio el {xd[SPX][r13.PRICE]:%d-%m} (tabla del apartado 6). La LRS 3x "
        f"original (rentabilidad total, orden al cierre siguiente) dejó el 3x al cierre del "
        f"{exits(SPX, BS, 2):%d-%m}: rinde {pc(ex[f'{L3} · {BS} · retraso 2', 'rent_19oct87'])} ese lunes y "
        f"{pc(ex[f'{L3} · {BS} · retraso 2', 'CAGR_periodo'])} al año en 1970–1999, y el 19-10-1987 se lleva "
        f"{num(ex[f'{L3} · {BS} · retraso 2', 'depende_del_día_pp'])} pp de esos treinta años. En el mismo "
        f"cierre salió el {exits(SPX, BS, 1):%d-%m} y rinde {pc(ex[f'{L3} · {BS} · retraso 1', 'CAGR_periodo'])}; "
        f"con la señal de precio y orden al cierre siguiente salió el {exits(SPX, r13.PRICE, 2):%d-%m} y rinde "
        f"{pc(ex[f'{PXS} · retraso 2', 'CAGR_periodo'])}. Con la señal original, "
        f"{n_week_out} de los 5 calendarios semanales"
        + (f" ({', '.join(wk_out)})" if wk_out else "")
        + f" y {n_month_out} de los 21 mensuales lo esquivaron. En el Nasdaq-100 la señal original cruzó el "
        f"{xd[NDX][BS]:%d-%m} y la LRS 3x original soportó el lunes "
        f"({pc(exn[f'{L3} · {BS} · retraso 2', 'rent_19oct87'])}); "
        + (f"solo lo esquivó el retraso {ndx_out[0]}." if len(ndx_out) == 1 else
           f"lo esquivaron los retrasos {', '.join(map(str, ndx_out))}." if ndx_out else
           "no lo esquivó ningún retraso."),
        f"**La LRS 3x original frente a 33 variantes de ejecución.** Frente a 33 variantes alternativas "
        f"(los otros 3 retrasos con la señal original, los 4 con la de precio, 5 calendarios semanales y 21 "
        f"cada 21 sesiones), la LRS 3x del S&P 500 ({pc(sm(SPX, L3, 'CAGR_base'))}) supera a "
        f"{fr(sp3, 'todas (33)')} (mediana {pc(sm(SPX, L3, 'CAGR_mediana'))}). Las variantes no son "
        f"homogéneas: 26 de las 33 miran la señal con menos frecuencia (21 de ellas, sólo cada 21 sesiones). "
        f"Por familias, la base supera a {fr(sp3, fam_cols[0])} variantes con revisión diaria (otro retraso u "
        f"otra señal), a {fr(sp3, fam_cols[1])} semanales y a {fr(sp3, fam_cols[2])} mensuales. Con el "
        f"19-10-1987 en letras para todas, supera a {fr(sp3_0, 'todas (33)')} ({fr(sp3_0, fam_cols[0])}, "
        f"{fr(sp3_0, fam_cols[1])} y {fr(sp3_0, fam_cols[2])}) y la mediana pasa a "
        f"{pc(sm(SPX, L3, 'CAGR_mediana_sin_19oct87'))}. En el Nasdaq-100 y en los semiconductores la LRS 3x "
        f"diaria supera a {fr(f'{NDX} · {L3}', fam_cols[2])} y a {fr(f'{SMH} · {L3}', fam_cols[2])} "
        f"calendarios mensuales. La LRS-Kelly del S&P 500 ({pc(sm(SPX, K, 'CAGR_base'))}) supera a "
        f"{fr(f'{SPX} · {K}', 'todas (33)')} variantes (con el crac en letras, a "
        f"{fr(f'{SPX} · {K} · sin 19-10-1987', 'todas (33)')}). "
        f"La LRS 3x queda por debajo de comprar y mantener el 1x en "
        f"{int(sm(SPX, L3, 'variantes_bajo_1x'))} de las 33 variantes en el S&P 500 (peor caso "
        f"{pc(sm(SPX, L3, 'CAGR_mín'))} frente a {pc(sm(SPX, L3, '1x_comprar_y_mantener'))}), "
        f"en {int(sm(NDX, L3, 'variantes_bajo_1x'))} en el Nasdaq-100 y en "
        f"{int(sm(SMH, L3, 'variantes_bajo_1x'))} en los semiconductores; la LRS-Kelly, en "
        f"{int(sm(SPX, K, 'variantes_bajo_1x'))}, {int(sm(NDX, K, 'variantes_bajo_1x'))} y "
        f"{int(sm(SMH, K, 'variantes_bajo_1x'))}.",
    ]
    rep.p("\n".join(f"- {b}" for b in bullets))
    rep.table(summ, pct={c for c in summ.columns if c.startswith(("CAGR", "base_supera_a"))}
              | {"1x_comprar_y_mantener"}, digits=2)
    rep.p("CAGR_base: rentabilidad anual compuesta de la regla con la señal original (rentabilidad total), "
          "revisión diaria y orden al cierre siguiente (la de R13). base_supera_a: fracción de las 33 "
          "variantes alternativas con una rentabilidad menor que la base; «_sin_19oct87»: lo mismo con ese "
          "día en letras en todas las series (base incluida; los semiconductores empiezan en 2004). Los "
          "percentiles describen un abanico heterogéneo, no una distribución de probabilidad: las variantes "
          "no son igual de probables a priori (un retraso de 5 sesiones o mirar la señal cada 21 sesiones "
          "son elecciones más lentas que la base), por lo que el puesto de la base no mide sólo suerte. Por "
          "eso la tabla siguiente da el puesto dentro de cada familia homogénea.")
    rep.table(FR, pct={"CAGR_base"} | {c for c in FR.columns if c.startswith("mediana_")}, digits=2)
    rep.p("Cada celda «k de n»: variantes de esa familia con una rentabilidad menor que la base. Familias: "
          "«retrasos y señal», revisión diaria con retraso 1, 3 o 5 y la señal original, o retraso 1, 2, "
          "3 o 5 y la señal de precio; «semanales», revisión un día fijo de la semana; «mensuales», "
          "revisión cada 21 sesiones con cada uno de los 21 desfases (las dos, con la señal original). "
          "Dentro de una misma frecuencia, los desfases sí son igual de razonables a priori y su abanico es "
          "suerte del calendario; entre frecuencias, la diferencia es de regla.")

    rep.h("Ventanas de evaluación")
    rep.p("Cada mercado se evalúa desde la primera previsión de varianza GJR (el 1-10-1985 en el "
          "Nasdaq-100, cuando empiezan los datos reales), como en R13: "
          + "; ".join(f"{n}, {RES[n]['start'].date()} → {D[n]['u'].index[-1].date()}" for n in MARKETS)
          + ". Todos los retrasos y desfases comparten exactamente la misma ventana: hasta que se ejecuta "
            "la primera orden (las primeras 1–5 sesiones de la LRS-Kelly) la cartera está en letras. La "
            "base (retraso 2, revisión diaria, señal original de rentabilidad total) reproduce las series de "
            "R13 (se comprueba en el código). Costes: 5 pb por unidad de rotación; el 3x se financia a f "
            "(call money hasta 1954, fondos federales después) más el sobrecoste del swap y paga los gastos "
            "del fondo; caja en letras; 3x sintético hasta el lanzamiento del ETF real.")

    # 1. retrasos
    rep.h("1. Retraso de ejecución")
    rep.p("Retraso 1: la orden se ejecuta en el mismo cierre que genera la señal (hay que calcularla "
          "unos minutos antes con el precio casi final). Retraso 2: al cierre siguiente, la base. "
          "Retrasos 3 y 5: dos y cuatro sesiones más tarde. Δcrec_vs_2 es la diferencia de "
          "crecimiento logarítmico anual frente al retraso 2, con su intervalo del 90% por bootstrap "
          f"estacionario emparejado (bloques medios de {BLOCK} sesiones, {B} remuestreos). Las "
          "operaciones por año no dependen del retraso. Señal original (rentabilidad total).")
    for name in MARKETS:
        rep.h(f"{SHOW[name]} (desde {RES[name]['start'].date()})", 3)
        rep.table(LAG[name], pct=pct_cols(LAG[name]), digits=2)
    fig_lag(D, RES, rep)

    rep.h("¿De cuántos días sale la diferencia entre retrasos?", 3)
    rep.p("Diferencia de crecimiento logarítmico anual entre cada retraso y el retraso 2 "
          "(Δcrec_anual) y la misma sin los 10 días en que más difieren. Δ_19oct87 y Δ_feb_abr_2020: "
          "lo que aportan a Δcrec_anual el 19-10-1987 y el tramo del 15-02 al 30-04 de 2020. Δ_sin_2020: "
          "diferencia anual del resto de la muestra, sin el año 2020 entero, con su intervalo del 90% "
          "(mismo bootstrap). años_a_favor: años naturales en que el retraso rinde más que el 2, entre "
          "los que difieren. días_distintos: sesiones en que las dos versiones no rinden lo mismo. "
          "Última columna: los tres días de mayor diferencia (en log).")
    for name in MARKETS:
        rep.h(SHOW[name], 4)
        rep.table(CONC[name], pct={"Δcrec_anual", "Δ_19oct87", "Δ_sin_10_mayores", "Δ_feb_abr_2020",
                                   "Δ_sin_2020"}, digits=2)
    rep.p(f"Lectura: en la LRS 3x del S&P 500, {read87}. Sin los diez días de mayor diferencia: retraso 1, "
          f"{arrow(SPX, 1)}; retraso 3, {arrow(SPX, 3)}; retraso 5, {arrow(SPX, 5)}. En el Nasdaq-100: 1, "
          f"{arrow(NDX, 1)}; 3, {arrow(NDX, 3)}; 5, {arrow(NDX, 5)}. En los semiconductores: 1, {arrow(SMH, 1)}; "
          f"3, {arrow(SMH, 3)}; 5, {arrow(SMH, 5)}. En total el signo cambia en {flips} de las {n_conc} "
          f"comparaciones y se mantiene en {n_conc - flips}. En los semiconductores el retraso 3 da "
          f"{pp(cg(SMH, L3, 3, 'Δcrec_anual'))}: el tramo del 15-02 al 30-04 de 2020 aporta "
          f"{pp(cg(SMH, L3, 3, 'Δ_feb_abr_2020'))}, sin 2020 quedan {pp(cg(SMH, L3, 3, 'Δ_sin_2020'))} al "
          f"año (IC90 {txt(cg(SMH, L3, 3, 'IC90_sin_2020'))}), y los años que aportan medio punto o más (en "
          f"valor absoluto) son: {txt(big_years)}. La LRS-Kelly de los semiconductores da "
          f"{pp(cg(SMH, K, 3, 'Δcrec_anual'))} con el mismo retraso (sin 2020, {pp(cg(SMH, K, 3, 'Δ_sin_2020'))}, "
          f"IC90 {txt(cg(SMH, K, 3, 'IC90_sin_2020'))}); las dos reglas comparten los mismos cruces y no son "
          f"pruebas independientes. Con {len(ci_rows)} intervalos, que uno o dos excluyan el cero es "
          "compatible con el azar. Lo que el retraso cambia es qué cruces se aprovechan; si eso se decide en "
          "unos pocos días o a lo largo de muchos depende del mercado y del periodo.")
    fig_lag_cum(D, RES, rep)

    rep.h("Antes y después de 1990", 3)
    rep.p("Diferencia de crecimiento anual frente al retraso 2 en cada tramo, con su intervalo del 90% "
          "(bootstrap estacionario, bloques de 63 sesiones), y el tramo anterior a 1990 sin el crac "
          "(el viernes 16 y el lunes 19-10-1987, que se quitan de la muestra). Sólo el S&P 500 y el "
          "Nasdaq-100 tienen datos antes de 1990; en el Nasdaq-100 ese tramo son 4,3 años.")
    for name in (SPX, NDX):
        rep.h(SHOW[name], 4)
        rep.table(PP90[name], pct={c for c in PP90[name].columns if c.startswith("Δ")}, digits=2)
    rep.p("Autocorrelación a un día de la rentabilidad del índice en la ventana de evaluación (error "
          "típico aproximado 1/√n, el de un ruido blanco; con volatilidad agrupada es algo mayor):")
    rep.table(AC, digits=3)
    rep.p(f"Lectura: en la LRS 3x del S&P 500 el mismo cierre frente al siguiente da "
          f"{pp(pq(SPX, L3, 1, 'Δ_hasta_1989'))} al año hasta 1989 (IC90 {txt(pq(SPX, L3, 1, 'IC90_hasta_1989'))}), "
          f"{pp(pq(SPX, L3, 1, 'Δ_hasta_1989_sin_crac'))} sin el crac (IC90 "
          f"{txt(pq(SPX, L3, 1, 'IC90_hasta_1989_sin_crac'))}), y "
          f"{pp(pq(SPX, L3, 1, 'Δ_desde_1990'))} desde 1990 (IC90 {txt(pq(SPX, L3, 1, 'IC90_desde_1990'))}). "
          f"En la LRS-Kelly, {pp(pq(SPX, K, 1, 'Δ_hasta_1989'))} hasta 1989 (IC90 "
          f"{txt(pq(SPX, K, 1, 'IC90_hasta_1989'))}) y {pp(pq(SPX, K, 1, 'Δ_desde_1990'))} desde 1990. "
          + ("Ninguno de estos cuatro intervalos excluye el cero. " if sp1_excl == 0 else
             f"De estos cuatro intervalos, {sp1_excl} {'excluye' if sp1_excl == 1 else 'excluyen'} el cero. ")
          + "La autocorrelación a un día del índice cambia "
          f"de signo ({num(ac('S&P 500 · hasta 1989'), 2)} hasta 1989, "
          f"{num(ac('S&P 500 · desde 1990'), 2)} después), como cabe esperar de un índice con precios "
          "desfasados que dejó de tenerlos. Una ventaja del mismo cierre antes de 1990 por esa vía es "
          "una hipótesis compatible con los datos, no una explicación demostrada, y no sería operable: "
          "con precios desfasados el cierre del índice no se podía negociar. "
          f"En el Nasdaq-100 los {pp(pq(NDX, L3, 1, 'Δ_hasta_1989'))} al año de 1985-10 a 1989 (IC90 "
          f"{txt(pq(NDX, L3, 1, 'IC90_hasta_1989'))}) incluyen el calendario del crac: el "
          f"19-10-1987 aporta un {pc(ndx_crash_log)} "
          f"en log, unos {pp(pq(NDX, L3, 1, 'Δ_hasta_1989') - pq(NDX, L3, 1, 'Δ_hasta_1989_sin_crac'))} "
          f"al año de ese tramo. Sin el crac quedan {pp(pq(NDX, L3, 1, 'Δ_hasta_1989_sin_crac'))} (IC90 "
          f"{txt(pq(NDX, L3, 1, 'IC90_hasta_1989_sin_crac'))}), en poco más de cuatro años, con un "
          "índice reconstruido en su media de 200 sesiones hasta 1986 y una autocorrelación a un día "
          f"de {num(ac('Nasdaq-100 · hasta 1989'), 2)}. "
          f"De los {len(pp_all)} intervalos de estas dos tablas, que no se corrigen por comparaciones "
          f"múltiples (por azar se esperarían {num(0.1 * len(pp_all))}), {len(pp_excl)} excluyen el cero"
          + (f": {pp_excl_txt}. El más alejado del cero es {SHOW[far[0]]}, {far[1].replace(' (original)', '')}, "
             f"{tramo_txt[far[2]]} (IC90 [{pc(far[3])}, {pc(far[4])}]). No se buscaron de antemano y hay que "
             "leerlos como indicios." if pp_excl else "."))

    rep.h("El día siguiente al cruce", 3)
    rep.p("Lo que separa el retraso 1 del 2 es la sesión que sigue al cruce: con el 1 ya se ha "
          "cambiado de posición y con el 2 todavía no. Rentabilidad media del índice (1x) esa sesión, "
          "en puntos básicos, según el sentido del cruce de la señal original, frente a la media de todas "
          "las sesiones. t_frente_a_día_normal = (media del día siguiente − media de todos los días) / error "
          "típico. La continuación a un día favorece al mismo cierre si el día siguiente rinde más de lo "
          "normal tras un cruce al alza y menos tras uno a la baja.")
    for name in MARKETS:
        rep.h(SHOW[name], 4)
        rep.table(NXT[name], digits=1)
    rep.p(f"Lectura: en el S&P 500 antes de 1990, el día siguiente a un cruce al alza rinde de media "
          f"{num(up89['rent_día_siguiente_pb'])} pb (error típico {num(up89['error_típico_pb'])}), frente "
          f"a {num(up89['rent_media_todos_los_días_pb'])} de un día normal (t = "
          f"{num(up89['t_frente_a_día_normal'])}); tras un cruce a la baja, "
          f"{num(dn89['rent_día_siguiente_pb'])} pb (t = {num(dn89['t_frente_a_día_normal'])}): "
          f"{continuation(up89, dn89)}. Desde 1990 el día siguiente a un cruce a la "
          f"baja rinde {num(dn90['rent_día_siguiente_pb'])} pb (error típico {num(dn90['error_típico_pb'])}; "
          f"t = {num(dn90['t_frente_a_día_normal'])}). Con unos {min(NXT[SPX]['cruces'].iloc[2:]):.0f}–"
          f"{max(NXT[SPX]['cruces'].iloc[2:]):.0f} cruces por tramo y sentido, ninguna de estas medias es "
          "precisa.")

    # 2. calendario
    rep.h("2. Suerte del calendario: revisión semanal y mensual")
    rep.p("La señal (y, en la LRS-Kelly, también la exposición de Kelly) sólo se mira el día de "
          "revisión y se mantiene hasta el siguiente; la orden se ejecuta al cierre siguiente. Semanal: "
          "cada lunes, martes, miércoles, jueves o viernes (si es festivo, la sesión siguiente). "
          "Mensual: cada 21 sesiones con los 21 desfases posibles (no meses naturales). Ningún desfase "
          "es mejor a priori: el abanico mide cuánto del resultado es suerte del calendario. "
          "amplitud_pp = rentabilidad anual máxima − mínima en puntos porcentuales. Señal original.")
    for name in MARKETS:
        res = RES[name]
        rep.h(f"{SHOW[name]} (desde {res['start'].date()})", 3)
        rep.table(CAL[name], pct={"CAGR_mín", "CAGR_mediana", "CAGR_máx", "maxDD_peor", "maxDD_mejor"},
                  digits=2)
        se = {s_: growth_se(res["lags"][(s_, BS, BASE_LAG)], res["start"]) for s_ in STRATS}
        rep.p("Como escala, el error típico del crecimiento anual medio con rentabilidades "
              "independientes es " + " y ".join(f"{pp(se[s_], sign=False)} en la {s_}" for s_ in STRATS)
              + ". El abanico no es un intervalo de confianza: los desfases comparten casi todos los días "
                "y mide la sensibilidad a una elección arbitraria, no el error de muestreo.")
    rep.h("S&P 500: el mismo abanico con el 19-10-1987 en letras", 3)
    rep.table(cal_sp_no87, pct={"CAGR_mín", "CAGR_mediana", "CAGR_máx", "maxDD_peor", "maxDD_mejor"},
              digits=2)
    parts, cmp_ = [], []
    for n in MARKETS:
        t = cal_sp_no87 if n == SPX else CAL[n]
        parts.append(f"{n}{' (sin el 19-10-1987)' if n == SPX else ''}: diaria "
                     f"{pc(t.loc[f'{L3} · diaria', 'CAGR_mín'])}, mediana semanal "
                     f"{pc(t.loc[f'{L3} · semanal', 'CAGR_mediana'])}, mediana mensual "
                     f"{pc(t.loc[f'{L3} · mensual', 'CAGR_mediana'])}")
        above = t.loc[f"{L3} · mensual", "CAGR_mediana"] > t.loc[f"{L3} · diaria", "CAGR_mín"]
        cmp_.append(f"{'por encima' if above else 'por debajo'} en {THE[n]}")
    rep.p("LRS 3x — " + "; ".join(parts) + ". La mediana mensual queda frente a la diaria "
          + ", ".join(cmp_) + ". Lo que sí hace revisar con menos frecuencia es ensanchar el abanico: con "
          "una revisión al mes, el día elegido pesa tanto como la regla.")
    rep.h("S&P 500: revisión semanal por día de la semana", 3)
    rep.table(wk, pct={"CAGR", "maxDD", "CAGR_1970_99"}, digits=2)
    rep.p("exposición_19oct87: apalancamiento soportado el lunes 19-10-1987 con ese calendario. "
          + (f"Solo {' y '.join(wk_out)} {'esquiva' if len(wk_out) == 1 else 'esquivan'} el lunes con la "
             "señal original." if wk_out else
             "Con la señal original ningún día de revisión semanal esquiva el lunes: la señal cruzó el "
             f"{xd[SPX][BS]:%d-%m} y la orden más temprana posible se ejecuta al cierre del lunes."))
    fig_calendar(D, RES, rep)

    # 3. señal
    rep.h("3. Señal de rentabilidad total (la original) frente a la de precio (sensibilidad)")
    rep.p("La LRS de Gayed y Bilello (2016) calcula la media de 200 sesiones sobre el índice de "
          "rentabilidad total; la señal sobre el índice de precio es una sensibilidad del laboratorio. El "
          "índice de rentabilidad total crece además con los dividendos, así que queda por encima de su "
          "media más a menudo y sale más tarde. En el S&P 500 el índice de precio es ^GSPC y el de "
          "rentabilidad total, en la señal, el índice oficial ^SP500TR desde 1988 y, antes, ^GSPC con los "
          "dividendos de Shiller del mes anterior. En el Nasdaq-100 y los semiconductores el precio se aproxima restando un "
          "dividendo constante (0,8% y 1,2% anual) a la rentabilidad total: la diferencia entre ambas "
          "señales es una deriva determinista y pequeña.")
    for name in MARKETS:
        t, diff_share, px_only = SIG[name]
        rep.h(f"{SHOW[name]} (desde {RES[name]['start'].date()})", 3)
        rep.table(t, pct={c for c in t.columns if c.startswith(("CAGR", "maxDD"))}, digits=2)
        rep.p(f"Las dos señales discrepan el {pc(diff_share)} de las sesiones; "
              + (f"en el {pc(px_only, 3)} el precio está sobre su media y la rentabilidad total no."
                 if px_only > 0 else
                 "no hay ninguna sesión en que el precio esté sobre su media y la rentabilidad total no"
                 + ("." if D[name].get("px_real", False) else
                    " (el precio aproximado es la rentabilidad total menos un dividendo constante).")))
    rep.p("Lectura: el índice de señal no es un detalle neutro. Cambia qué días de crisis se pasan dentro "
          "y, combinado con el retraso, el orden entre las dos señales cambia de un mercado y de un retraso "
          f"a otro. En el S&P 500, con orden al cierre siguiente, la de precio rinde {pp(px2 - base2)} más al "
          f"año que la original; con el 19-10-1987 en letras, {pp(px2_0 - base2_0)}.")

    # 4. cruces
    rep.h("4. Distancia a la media en los días de cambio de señal")
    rep.p("Cruces de la media de 200 sesiones del índice de la señal original (rentabilidad total) dentro "
          "de la ventana de evaluación. "
          "«se_deshace_≤k»: fracción de cruces seguidos de otro en k sesiones o menos (la regla entra y "
          "sale, o sale y entra, pagando costes y el retraso). «excursión<2%»: fracción de estados en que "
          "el índice nunca se aleja más del 2% de su media antes del cruce siguiente. «|dist|_med_ruido» "
          "y «|dist|_med_duraderos»: distancia mediana (en %) el día del cruce de los que se deshacen en "
          "10 sesiones o menos y de los que duran más de 21; «duraderos_con_|dist|<1%»: fracción de estos "
          "últimos que empezaron a menos del 1% de la media. La referencia es un paseo aleatorio: las "
          "rentabilidades diarias del mismo índice y periodo remuestreadas con reemplazo "
          f"({N_RW} caminos; mediana de cada estadístico), con la misma deriva y distribución pero sin "
          "rachas ni agrupamiento de volatilidad.")
    rep.table(ct_show, pct={c for c in ct_show.columns
                             if c.startswith(("|dist|<", "se_deshace", "excursión", "duraderos"))}, digits=2)
    rep.p(f"Lectura: en los tres mercados entre el {pc(real_undo.min(), 0)} y el {pc(real_undo.max(), 0)} de "
          f"los cruces se deshacen en diez sesiones o menos (en el paseo aleatorio, entre el "
          f"{pc(rw_undo.min(), 0)} y el {pc(rw_undo.max(), 0)}). Entre el "
          f"{pc(ct.loc[real_rows, 'duraderos_con_|dist|<1%'].min(), 0)} "
          f"y el {pc(ct.loc[real_rows, 'duraderos_con_|dist|<1%'].max(), 0)} de los duraderos también empieza "
          "a menos del 1%: la distancia del día del cruce separa mal el ruido de la tendencia. Filtrar los "
          "cruces cercanos con la banda de ±2% de R13 da: "
          + "; ".join(f"{pc(band[n])} frente a {pc(g(n, L3, 2))} en {THE[n]}" for n in MARKETS) + ".")
    fig_crossings(RES, rep)

    # 5. R12
    rep.h("5. El medio Kelly de R12 y sus 21 calendarios de revisión")
    rep.p("R12 revisa la exposición cada 21 sesiones con keep[::H], de modo que el calendario lo fija "
          "el primer día de cada serie (3-1-1928 en el S&P 500). Aquí se repite la regla con los 21 "
          "desfases posibles (reimplementada en este módulo; el desfase 0 reproduce exactamente "
          "r12.strategy, comprobado en las pruebas), con la varianza GJR-GARCH y las mismas ventanas que "
          "la tabla E de R12 (que empieza cuando los modelos de ML tienen diez años de datos). "
          "«puesto_desfase_0»: posición del calendario de R12 entre los 21 (1 = el mejor). La última "
          "columna es la misma regla revisada a diario con el filtro de reajuste de R13.")
    rep.table(t12, pct={c for c in t12.columns if c.startswith(("CAGR", "maxDD"))}, digits=2)
    rep.p(f"Lectura: con una revisión cada 21 sesiones, entre {num(t12['amplitud_pp'].min())} y "
          f"{num(t12['amplitud_pp'].max())} puntos de rentabilidad anual dependen del día en que cae la "
          "revisión, algo que ningún modelo de varianza controla. Una diferencia entre dos modelos de la "
          "tabla E de R12 menor que este abanico no puede atribuirse al modelo sin comprobar que se "
          "mantiene con los otros calendarios.")
    fig_r12(D, RES, rep)

    # 6. octubre de 1987
    rep.h("6. Octubre de 1987 y la rentabilidad de 1970–1999 (S&P 500)")
    rep.p("Distancia de cada índice de señal a su media de 200 sesiones (índice/media − 1) del 13 al "
          "20-10-1987; el primer día negativo es el día en que la señal pasa a letras:")
    rep.table(dist87, pct=set(dist87.columns), digits=2)
    rep.p(f"El 15-10-1987 el índice de precio cerró un {pc(-dist87.loc[pd.Timestamp('1987-10-15').date(), dist87.columns[1]], 2)} "
          f"bajo su media y el de rentabilidad total un {pc(dist87.loc[pd.Timestamp('1987-10-15').date(), dist87.columns[0]], 2)} "
          f"por encima. Último cruce a la baja antes del 19-10-1987: rentabilidad total (la original), "
          f"{xd[SPX][BS]:%d-%m-%Y}; precio, {xd[SPX][r13.PRICE]:%d-%m-%Y}. Con retraso r la cartera deja el 3x "
          "al cierre de la r-ésima sesión desde el cruce (la del cruce es la primera): con la señal original "
          f"la venta se hace al cierre del {exits(SPX, BS, 2):%d-%m} con retraso 2 y al del "
          f"{exits(SPX, BS, 1):%d-%m} con retraso 1; con la de precio, al del {exits(SPX, r13.PRICE, 2):%d-%m} "
          f"con retraso 2 y al del {exits(SPX, r13.PRICE, 1):%d-%m} con retraso 1. exposición_19oct87: "
          "apalancamiento que soporta la regla ese lunes "
          f"({pc(ex['1x comprar y mantener', 'rent_19oct87'])} el índice con dividendos). CAGR_sin_19oct87: "
          "la rentabilidad anual compuesta de 1970–1999 si ese día la cartera hubiera ganado la letra; "
          "depende_del_día_pp es la diferencia, en puntos. es_la_original marca la LRS 3x original (señal "
          "de rentabilidad total, orden al cierre siguiente). Es una descomposición contable: no vuelve a "
          "simular las entradas posteriores. Las filas con exposición 0 rinden ese día la letra menos, si la "
          "venta se ejecutó al cierre anterior, el coste de 5 pb, que se carga el día en que cambia la "
          "exposición.")
    rep.table(crash, pct={"rent_19oct87", "CAGR_periodo", "CAGR_sin_19oct87"}, digits=2)
    rep.h("El mismo día con revisión semanal y mensual", 3)
    rep.table(cc, pct={c for c in cc.columns if c.startswith("CAGR")}, digits=2)
    rep.h("Nasdaq-100 (1985-10 a 1999)", 3)
    same_ndx = xd[NDX][BS] == xd[NDX][r13.PRICE]
    rep.p((f"En el Nasdaq-100 los dos índices cruzaron el {xd[NDX][BS]:%d-%m-%Y}" if same_ndx else
           f"En el Nasdaq-100 la señal original cruzó el {xd[NDX][BS]:%d-%m-%Y} y la de precio aproximado el "
           f"{xd[NDX][r13.PRICE]:%d-%m-%Y}")
          + f": con la señal original y retraso 2 la LRS 3x deja el 3x al cierre del "
          f"{exits(NDX, BS, 2):%d-%m}. El 19-10-1987 aporta {pp(cg(NDX, L3, 1, 'Δ_19oct87'))} de los "
          f"{pp(cg(NDX, L3, 1, 'Δcrec_anual'))} al año de diferencia del mismo cierre en el Nasdaq-100 "
          "(apartado 1).")
    rep.table(crash_ndx, pct={"rent_19oct87", "CAGR_periodo", "CAGR_sin_19oct87"}, digits=2)
    fig_1987(sp, rs, crash, rep)

    rep.h("Salvedades")
    rep.p("\n".join("- " + x for x in [
        "El retraso 1 supone que la señal calculada minutos antes del cierre coincide con la del cierre "
        "y que se opera a ese precio. Antes de 1990 la autocorrelación a un día del S&P 500 era "
        f"positiva (ρ₁ = {num(ac('S&P 500 · hasta 1989'), 2)}), rasgo de los índices con precios "
        "desfasados que no se podía aprovechar operando: si parte de la ventaja del retraso 1 antes de "
        "1990 procede de ahí, esas cifras son optimistas. La continuación tras los cruces antes de 1990 "
        f"da t = {num(up89['t_frente_a_día_normal'])} al alza y {num(dn89['t_frente_a_día_normal'])} a la baja.",
        "Los costes (5 pb por unidad de rotación) no dependen del retraso ni del día de la semana; no "
        "hay horquilla ni impacto. La exposición fraccionaria de la LRS-Kelly se mantiene constante "
        "entre reajustes, lo que implica reequilibrar a diario sin coste (como en R13).",
        "El 3x es sintético hasta el lanzamiento del ETF real y su sobrecoste se calibra con toda la "
        "muestra del ETF (R12); la fracción de Kelly y el GJR-GARCH se eligieron en R10–R12 con la misma "
        "muestra. Afectan por igual a todas las variantes de este informe.",
        "En el Nasdaq-100 la media de 200 sesiones de 1985–86 usa la reconstrucción de 1971–85 (la "
        "primera de R12); su índice de precio y el de los semiconductores se obtienen restando un "
        "dividendo constante.",
        f"Se calculan {len(ci_rows)} intervalos del 90% para los retrasos sin corrección múltiple: por "
        f"azar se esperan {num(0.1 * len(ci_rows))} que excluyan el cero; lo excluyen {len(excl)} "
        f"({'; '.join(f'{n}: {i}' for n, i in excl) or 'ninguno'}). Los intervalos por tramos (antes y "
        "después de 1990, sin 2020) son descriptivos y tampoco se corrigen. Los semiconductores sólo "
        f"tienen {num(RES[SMH]['years'], 0)} años y unos {len(RES[SMH]['cruces'])} cruces.",
        "El paseo aleatorio de referencia conserva la distribución y la deriva, pero no el agrupamiento "
        "de volatilidad; la comparación de los cruces es descriptiva, sin contraste formal.",
        "La revisión mensual es cada 21 sesiones, no el mes natural; la semanal usa el día de la semana "
        "(si es festivo, la sesión siguiente). La primera sesión con objetivo definido abre la cartera "
        "en todos los calendarios. Los abanicos no son intervalos de confianza.",
        "El desfase 0 reproduce r12.strategy con el código y los datos actuales. Si el README de R12 en "
        "disco procede de una ejecución anterior, sus cifras pueden diferir en décimas.",
    ]))
    rep.save()
    return {"D": D, "RES": RES, "lag": LAG, "conc": CONC, "next": NXT, "cal": CAL, "cal_no87": cal_sp_no87,
            "wk": wk, "sig": SIG, "cross": ct, "band": band, "r12": t12, "crash": crash,
            "crash_ndx": crash_ndx, "crash_cal": cc, "summary": summ, "cross_dates": xd}


if __name__ == "__main__":
    run()
