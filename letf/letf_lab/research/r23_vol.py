"""R23 — ¿Sirve prever la volatilidad para cronometrar el 3x?

Pregunta del artículo 3: la volatilidad se puede prever (R10–R12), y el 3x
pierde con la volatilidad (−½·L²·σ²). ¿Mejora a la LRS de Gayed y Bilello
(2016) una regla que use esa previsión para decidir cuándo estar en el 3x?

Previsiones de la varianza media de las 21 sesiones siguientes, todas causales
(con datos hasta el cierre de t) y fuera de muestra, con la reestimación de R12
(diez años mínimos; GARCH cada 1.008 sesiones, HAR cada 504 con embargo de 21):
- GARCH(1,1) y GJR-GARCH(1,1) gaussianos (el código del laboratorio,
  r12.garch_walk);
- EGARCH(1,1) de Nelson (1991), gaussiano, estimado aquí por máxima
  verosimilitud (no hay paquete arch en el entorno), con la previsión a h
  sesiones exacta bajo normalidad (egarch_horizon_var);
- HAR de Corsi (2009) en logaritmos sobre la varianza realizada diaria,
  semanal y mensual, devuelto a niveles con la corrección de Duan (1983)
  medida fuera de muestra (r12.walk_forward);
- EWMA de RiskMetrics (λ = 0,94), sin parámetros estimados.

Reglas: las de una exploración previa del laboratorio, con sus parámetros sin
tocar (véanse las salvedades del informe). Todas deciden al cierre de t y operan al cierre de t+1
con r13.run_weights (5 pb por unidad de rotación, el 3x sintético financiado
a f + s hasta el ETF real y el real después, las letras al T-bill); la
tendencia es la de la LRS original, la media de 200 sesiones del índice de
rentabilidad total (r13.tr_index):
(a) filtro de régimen: 3x si la volatilidad prevista está por debajo de su
    mediana expansiva (o de su percentil 75); si no, 1x o letras;
(b) LRS × volatilidad: sobre la media, 3x si la volatilidad prevista es baja y
    1x si es alta; bajo la media, letras. Se compara con la LRS con un
    apalancamiento fijo más bajo: el de su misma exposición media y el de su
    misma volatilidad;
(c) VIX (desde 1990): 3x si el VIX está por debajo de 20 o de su mediana
    expansiva; y la estructura temporal VIX/VIX3M (desde julio de 2006):
    3x en contango (VIX/VIX3M < 1). Solas o combinadas con la LRS;
(d) un clasificador en avance (logístico y gradient boosting) que predice si
    el 3x rinde más que las letras en las 21 sesiones que cobraría la posición,
    con embargo de 22 sesiones; 3x si la probabilidad es al menos 0,5.
Los umbrales de (a) y (b) usan solo previsiones fuera de muestra (la
exploración previa sembraba la mediana con los valores ajustados de la primera
estimación, 1928–1938, y eso movía las cifras).

Inferencia, como R13 y R19: Δg (crecimiento logarítmico anual) de cada regla
frente a la LRS 3x original, frente al 1x y frente al 3x comprar y mantener
con un bootstrap estacionario
emparejado (bloque medio de 63 sesiones, 2.000 remuestreos, los mismos para
todas las reglas), IC 90% percentil y p unilateral con el error estándar del
bootstrap (H0: Δg ≤ 0); SPA de Hansen (2005) sobre la familia completa de
reglas probadas, frente a las mismas referencias; ninguna regla se llama
mejora sin esa corrección. Mercados: S&P 500 (1939–), Nasdaq-100 (datos reales
desde 1985-10, con las 10 reconstrucciones de R13 para el calentamiento) y
semiconductores (2005–); ETFs reales desde 2010.
"""

from __future__ import annotations

import math
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize
from scipy.stats import norm

from letf_lab import data, kelly, metrics, stats
from letf_lab.data import history
from letf_lab.report import FIG_W, Report, comma_fig, legend_band, num_es, pct_es
from letf_lab.research import r12_long_ml as r12
from letf_lab.research import r13_lrs as r13

SEED = 23
B = 2000                   # remuestreos del bootstrap y del SPA (como R13 y R19)
B_MCS = 1000               # remuestreos del Model Confidence Set
B_AUC = 1000               # remuestreos del IC del AUC
BLOCK = 63                 # bloque medio del bootstrap estacionario (sesiones)
LAG = r13.EXEC_LAG         # señal al cierre de t, orden al cierre de t+1
H = r12.H                  # horizonte de la previsión (21 sesiones)
MIN_OBS = r12.MIN_OBS      # diez años antes de la primera previsión
REFIT = r12.REFIT          # reestimación: GARCH cada 2·REFIT, HAR cada REFIT (como R12)
MIN_THR = 252              # previsiones fuera de muestra mínimas del umbral expansivo
QS = {"mediana": 0.5, "p75": 0.75}
MODELS = ("GARCH", "GJR", "EGARCH", "HAR", "EWMA")
MAIN_MODEL = "GJR"         # el de R13 (LRS-Kelly) y el de los cuadrantes
VIX_LEVEL = 20.0           # umbral popular del VIX
CLF_REFIT = 252            # reentreno del clasificador (sesiones)
EMBARGO = H + 1            # la etiqueta de t se conoce al cierre de t + 22
CLF_P = 0.5                # 3x si la probabilidad es al menos 0,5
N_NDX = r13.N_NDX
REAL_FROM = r13.REAL_FROM

LRS = r13.ORIGINAL                     # «LRS 3x (original)»
ONE = "1x comprar y mantener"
THREE = "3x comprar y mantener"
ROT = "LRS rotación 3x/1x"
BENCH = (ONE, THREE, LRS, ROT)
CENSUS_PREFIX = "R23 · "

GROUPS = {"a": "(a) filtro de régimen", "b": "(b) LRS × volatilidad", "c": "(c) VIX",
          "t": "(c) VIX/VIX3M", "d": "(d) clasificador"}

# Ventanas del S&P 500 (las reglas de volatilidad empiezan en 1939: la etiqueta es la del
# laboratorio y la tabla da el primer día real)
SP_WINDOWS = {"completa": (None, None), "1938–1973": ("1938-01-01", "1973-12-31"),
              "1974–1999": ("1974-01-01", "1999-12-31"), "2000–2026": ("2000-01-01", None),
              "2016–2026 (tras la muestra de Gayed y Bilello)": ("2016-01-04", None),
              "2017–2026 (tras la publicación)": ("2017-01-03", None),
              "ETFs reales (2010–2026)": (REAL_FROM, None)}


# ───────────────────────────── EGARCH(1,1) ─────────────────────────────

C_ABS = math.sqrt(2 / math.pi)     # E|z| con z normal
LH_MAX, LH_MIN = 5.0, -30.0        # cotas del log de la varianza diaria (fuera de cualquier rango real)


def _egarch_path(om: float, a: float, g: float, b: float, r: list, lh0: float,
                 keep: bool = False) -> tuple[float, list | None]:
    """Recursión de Nelson (1991): log h_{t+1} = ω + β log h_t + α(|z_t| − √(2/π)) + γ z_t,
    con z_t = r_t/√h_t. Devuelve −log-verosimilitud gaussiana (sin constantes) y, con
    keep, log h_{t+1} para cada t (la varianza de t+1 conocida al cierre de t)."""
    lh, nll = lh0, 0.0
    out = [] if keep else None
    exp = math.exp
    for x in r:
        z = x * exp(-0.5 * lh)
        nll += lh + z * z
        lh = om + b * lh + a * (abs(z) - C_ABS) + g * z
        if lh > LH_MAX:
            lh = LH_MAX
        elif lh < LH_MIN:
            lh = LH_MIN
        if keep:
            out.append(lh)
    return 0.5 * nll, out


def _egarch_unpack(th) -> tuple[float, float, float, float]:
    return float(th[0]), float(th[1]), float(th[2]), 0.9999 / (1 + math.exp(-float(th[3])))


def egarch_fit(r: np.ndarray, x0=None) -> dict:
    """EGARCH(1,1) gaussiano por máxima verosimilitud (Nelder-Mead; β en (0; 0,9999)).
    r: rentabilidades diarias; se les resta su media. x0: parámetros internos de un
    ajuste anterior (arranque en caliente al reestimar)."""
    r = np.asarray(r, float)
    mean = float(np.nanmean(r))
    rr = (np.nan_to_num(r, nan=mean) - mean).tolist()
    lv = math.log(float(np.var(rr)))

    def nll(th):
        om, a, g, b = _egarch_unpack(th)
        v = _egarch_path(om, a, g, b, rr, lv)[0]
        return v if np.isfinite(v) else 1e300

    if x0 is None:
        b0 = 0.98
        x0 = [(1 - b0) * lv, 0.12, -0.08, math.log(b0 / (0.9999 - b0))]
    res = minimize(nll, x0, method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-4})
    om, a, g, b = _egarch_unpack(res.x)
    return {"omega": om, "alpha": a, "gamma": g, "beta": b, "mean": mean, "lv0": lv,
            "theta": [float(v) for v in res.x], "nll": float(res.fun), "ok": bool(res.success)}


def egarch_filter(r: np.ndarray, p: dict) -> np.ndarray:
    """log h_{t+1} conocido al cierre de t, con los parámetros p (la media de su ajuste)."""
    rr = (np.nan_to_num(np.asarray(r, float), nan=p["mean"]) - p["mean"]).tolist()
    return np.array(_egarch_path(p["omega"], p["alpha"], p["gamma"], p["beta"], rr, p["lv0"],
                                 keep=True)[1])


def egarch_mgf(s, alpha: float, gamma: float) -> np.ndarray:
    """E[exp(s·η)] con η = α(|z| − √(2/π)) + γz y z normal estándar:
    e^{−sα√(2/π)}·[e^{s²(α+γ)²/2}·Φ(s(α+γ)) + e^{s²(α−γ)²/2}·Φ(s(α−γ))]."""
    s = np.asarray(s, float)
    up, dn = s * (alpha + gamma), s * (alpha - gamma)
    return np.exp(-s * alpha * C_ABS) * (np.exp(up ** 2 / 2) * norm.cdf(up) + np.exp(dn ** 2 / 2) * norm.cdf(dn))


def egarch_horizon_var(lh_next: np.ndarray, p: dict, horizon: int) -> np.ndarray:
    """Varianza diaria media prevista para las `horizon` sesiones siguientes, exacta con z
    normal: log h_{t+1+k} = λ + β^k(log h_{t+1} − λ) + Σ_{j=1..k} β^{k−j}η_{t+j}, λ = ω/(1 − β),
    así que E[h_{t+1+k}] = exp(λ + β^k(log h_{t+1} − λ))·Π_{i<k} E[exp(β^i·η)]. La senda
    determinista del logaritmo (sin el producto) la subestima."""
    lr = p["omega"] / (1 - p["beta"])
    k = np.arange(horizon)
    m = egarch_mgf(p["beta"] ** np.arange(max(horizon - 1, 0)), p["alpha"], p["gamma"])
    P = np.r_[1.0, np.cumprod(m)]
    lh = np.asarray(lh_next, float)
    return (np.exp(lr + np.outer(lh - lr, p["beta"] ** k)) * P).mean(axis=1)


def egarch_walk(u: pd.Series) -> tuple[pd.Series, list[dict]]:
    """Varianza anual prevista a H sesiones fuera de muestra, reestimada cada 2·REFIT
    sesiones con datos hasta t (como r12.garch_walk), y los parámetros de cada ajuste."""
    u = u.dropna()
    r = u.to_numpy(float)
    out = np.full(len(u), np.nan)
    fits, x0 = [], None
    for start in range(MIN_OBS, len(u), REFIT * 2):
        p = egarch_fit(r[:start], x0)
        x0 = p["theta"]
        lh = egarch_filter(r, p)
        seg = slice(start, min(start + REFIT * 2, len(u)))
        out[seg] = egarch_horizon_var(lh[seg], p, H) * 252
        fits.append({"fecha": u.index[start].date(), **{k: p[k] for k in ("omega", "alpha", "gamma", "beta")},
                     "persistencia": p["beta"], "convergió": p["ok"]})
    return pd.Series(out, index=u.index), fits


# ───────────────────────────── previsiones ─────────────────────────────

def garch_hist(u: pd.Series, kind: str) -> pd.Series:
    """r12.garch_walk (fuera de muestra desde MIN_OBS) con un prefijo: antes de la primera
    previsión, los valores filtrados con los parámetros de la PRIMERA estimación, que ya se
    conocen en esa fecha. Solo lo usa el clasificador, para entrenar con los diez primeros
    años (todo lo que ve en T se conoce en T); las reglas de umbral usan solo el tramo
    fuera de muestra."""
    u = u.dropna()
    r = u.fillna(0).to_numpy(float)
    out = np.full(len(u), np.nan)
    for i, start in enumerate(range(MIN_OBS, len(u), REFIT * 2)):
        p = stats.garch_fit(r[:start], kind)
        hn = stats.garch_filter(r - r[:start].mean(), p)
        seg = slice(start, min(start + REFIT * 2, len(u)))
        out[seg] = stats.garch_horizon_var(hn[seg], p, H) * 252
        if i == 0:
            out[:start] = stats.garch_horizon_var(hn[:start], p, H) * 252
    return pd.Series(out, index=u.index)


def realized_target(u: pd.Series) -> pd.Series:
    """Varianza realizada anual de las H sesiones siguientes (t+1 … t+H), el objetivo de R12."""
    return (u ** 2).rolling(H).sum().shift(-H) * 252 / H


def har_log(u: pd.Series) -> pd.Series:
    """HAR en logaritmos (Corsi 2009): log de la varianza realizada de las 21 sesiones
    siguientes sobre el log de la diaria, la semanal y la mensual; reestimado cada REFIT
    sesiones con embargo H y devuelto a niveles con la corrección de Duan medida fuera de
    muestra (r12.walk_forward con log_target)."""
    from sklearn.linear_model import LinearRegression

    u = u.dropna()
    X = r12.features(u)[["lrv_d", "lrv_w", "lrv_m"]]
    ly = np.log(realized_target(u).clip(lower=1e-6))
    return r12.walk_forward(X, ly, LinearRegression, log_target=True)


def ewma(u: pd.Series) -> pd.Series:
    """EWMA de RiskMetrics (λ = 0,94) de la varianza anual: sin parámetros estimados."""
    u = u.dropna()
    return (u ** 2 * 252).ewm(alpha=0.06, adjust=False).mean()


def forecasts(d: dict) -> dict:
    """{modelo: varianza anual prevista para las 21 sesiones siguientes}, fuera de muestra,
    sobre d['u'].index; además 'GJR_prefijo' (garch_hist, solo para el clasificador) y
    'egarch_fits'."""
    u = d["u"].dropna()
    idx = d["u"].index
    eg, fits = egarch_walk(u)
    F = {"GARCH": r12.garch_walk(u, "garch"), "GJR": r12.garch_walk(u, "gjr"), "EGARCH": eg,
         "HAR": har_log(u), "EWMA": ewma(u)}
    out = {k: v.reindex(idx) for k, v in F.items()}
    out["GJR_prefijo"] = garch_hist(u, "gjr").reindex(idx)
    out["egarch_fits"] = fits
    return out


def oos_start(F: dict) -> pd.Timestamp:
    """Primer día en que las cinco previsiones son fuera de muestra (EWMA no estima nada:
    empieza con las demás para que todos los umbrales usen la misma historia)."""
    return max(F[m].first_valid_index() for m in MODELS if m != "EWMA")


def expanding_quantile(x: pd.Series, q: float, min_obs: int = MIN_THR) -> pd.Series:
    """Cuantil q de los valores de x hasta t incluido (causal), con min_obs como mínimo."""
    if q == 0.5:
        return x.expanding(min_obs).median()
    return x.expanding(min_obs).quantile(q)


def low_vol(fc: pd.Series, start, q: float = 0.5) -> pd.Series:
    """1 si la previsión de t está por debajo de su cuantil q expansivo, calculado solo con
    previsiones desde `start` (fuera de muestra); 0 si no; NaN antes de MIN_THR previsiones."""
    x = fc.where(fc.index >= pd.Timestamp(start))
    thr = expanding_quantile(x.dropna(), q).reindex(fc.index)
    return (x < thr).astype(float).where(thr.notna() & x.notna())


def vix_series(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """VIX (cierre) y VIX/VIX3M sobre `index`, sin rellenar antes de que existan (hasta 5
    sesiones de hueco se rellenan con el último valor conocido)."""
    v = data.panel(["^VIX", "^VIX3M"], "close")

    def align(s: pd.Series) -> pd.Series:
        s = s.dropna()
        out = s.reindex(s.index.union(index)).ffill(limit=5).reindex(index)
        return out.where(index >= s.index[0])

    vix = align(v["^VIX"])
    ts = align((v["^VIX"] / v["^VIX3M"]).dropna())
    return vix, ts


# ───────────────────────────── (d) clasificador ─────────────────────────────

def macd(px: pd.Series) -> tuple[pd.Series, pd.Series]:
    """MACD(12, 26) relativo al precio y su histograma (MACD menos su media exponencial de 9)."""
    m = (px.ewm(span=12, adjust=False).mean() - px.ewm(span=26, adjust=False).mean()) / px
    return m, m - m.ewm(span=9, adjust=False).mean()


def clf_features(d: dict, gjr: pd.Series, vix: pd.Series | None = None) -> pd.DataFrame:
    """Rasgos conocidos al cierre de t: distancia a la media de 200 sesiones del índice de
    rentabilidad total (la señal de la LRS), MACD y su histograma, la volatilidad prevista
    (GJR con prefijo, garch_hist) en nivel y relativa a su mediana expansiva y, con VIX, su
    logaritmo y el cociente entre la varianza implícita y la prevista."""
    px = r13.tr_index(d)
    idx = d["u"].index
    ma = px.rolling(200).mean()
    m, hist = macd(px)
    g = gjr.reindex(idx)
    X = pd.DataFrame({"distancia_media_200": px / ma - 1, "macd": m.where(ma.notna()),
                      "macd_histograma": hist.where(ma.notna()), "log_vol_prevista": 0.5 * np.log(g),
                      "vol_relativa": np.log(g / g.expanding(MIN_THR).median())}, index=idx)
    if vix is not None:
        X["log_vix"] = np.log(vix / 100)
        X["vix_sobre_prevista"] = np.log((vix / 100) ** 2 / g)
    return X


def clf_target(d: dict) -> pd.Series:
    """1 si Σ_{j=t+2}^{t+22} [log(1 + x3_j) − log(1 + rf_j)] > 0: el 3x rinde más que las
    letras en las 21 sesiones que cobra una posición decidida en t (orden al cierre de t+1).
    Se conoce al cierre de t + 22."""
    lx = np.log1p(d["x3"].clip(lower=-0.9999)) - np.log1p(d["rf"])
    fwd = lx.rolling(H).sum().shift(-EMBARGO)
    return (fwd > 0).astype(float).where(fwd.notna())


def clf_models() -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {"logístico": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
            "gradient boosting": lambda: HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=200,
                early_stopping=False, random_state=SEED)}


def clf_walk(X: pd.DataFrame, y: pd.Series, make_model, valid_from=None,
             min_obs: int = MIN_OBS, refit: int = CLF_REFIT, embargo: int = EMBARGO) -> pd.Series:
    """Probabilidad del 3x en t con un modelo entrenado en T ≤ t solo con muestras s ≤ T − embargo
    (etiquetas ya conocidas en T), reentrenado cada `refit` sesiones con ventana expansiva desde
    que hay min_obs etiquetas. valid_from: primera fecha en que se predice."""
    ok_x = X.notna().all(axis=1).to_numpy()
    ok_y = y.reindex(X.index).notna().to_numpy()
    Xa, ya = X.to_numpy(float), y.reindex(X.index).to_numpy(float)
    n = len(X)
    out = np.full(n, np.nan)
    lab = np.cumsum(ok_x & ok_y)
    known = np.r_[np.zeros(embargo), lab[:-embargo]] if n > embargo else np.zeros(n)
    if not (known >= min_obs).any():
        return pd.Series(out, index=X.index)
    first = int(np.argmax(known >= min_obs))
    if valid_from is not None:
        first = max(first, int(np.searchsorted(X.index, pd.Timestamp(valid_from))))
    for T in range(first, n, refit):
        train = np.zeros(n, bool)
        train[: max(T - embargo + 1, 0)] = True
        train &= ok_x & ok_y
        m = make_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xa[train], ya[train].astype(int))
        seg = np.arange(T, min(T + refit, n))
        seg = seg[ok_x[seg]]
        if len(seg):
            out[seg] = m.predict_proba(Xa[seg])[:, 1]
    return pd.Series(out, index=X.index)


def classifiers(d: dict, F: dict, vix: pd.Series | None) -> dict:
    """{nombre: probabilidad del 3x} de los dos modelos, sin VIX y (si hay VIX) con VIX.
    Predicen desde la primera previsión GJR fuera de muestra."""
    y = clf_target(d)
    gjr0 = F[MAIN_MODEL].first_valid_index()
    out = {}
    for tag, v in (("", None), (" + VIX", vix)):
        if tag and v is None:
            continue
        X = clf_features(d, F["GJR_prefijo"], v)
        if v is not None:
            X = X[X.index >= v.first_valid_index()]
        for name, mk in clf_models().items():
            out[f"Clasificador {name}{tag}"] = clf_walk(X, y, mk, valid_from=gjr0).reindex(d["u"].index)
    return out


def auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan


def auc_table(prob: dict, y: pd.Series, extra: dict, windows: dict, rng) -> pd.DataFrame:
    """AUC fuera de muestra de cada clasificador (y de puntuaciones sin ajuste: la distancia a la
    media y la volatilidad prevista cambiada de signo) en las fechas en que el clasificador
    predice, con IC 90% por bootstrap estacionario (bloque 63) de los pares (p, y); acierto con
    p ≥ 0,5 frente al de «siempre 3x» (la tasa base) y fracción del tiempo en el 3x."""
    rows = []
    for name, p in prob.items():
        for wname, (a, b) in windows.items():
            m = pd.concat([p, y], axis=1, keys=["p", "y"]).loc[a:b].dropna()
            if len(m) < 500:
                continue
            yy, pp = m.y.to_numpy(), m.p.to_numpy()
            draws = []
            for _ in range(B_AUC):
                i = stats.stationary_indices(len(m), BLOCK, rng)
                draws.append(auc(yy[i], pp[i]))
            lo, hi = np.nanpercentile(draws, [5, 95])
            row = {"clasificador": name, "ventana": wname, "desde": m.index[0].date(), "AUC": auc(yy, pp),
                   "IC90_bajo": lo, "IC90_alto": hi, "acierto": float(((pp >= CLF_P) == (yy > 0.5)).mean()),
                   "acierto_siempre_3x": float(yy.mean()), "fracción_en_3x": float((pp >= CLF_P).mean())}
            for lab, s in extra.items():
                z = s.reindex(m.index)
                ok = z.notna().to_numpy()
                row[f"AUC {lab}"] = auc(yy[ok], z.to_numpy()[ok])
            rows.append(row)
    return pd.DataFrame(rows)


# ───────────────────────────── reglas ─────────────────────────────

def bench_returns(d: dict) -> tuple[dict, dict]:
    """1x, 3x, LRS 3x (original) y LRS rotación 3x/1x con el protocolo de R13 (señal: media de
    200 sesiones del índice de rentabilidad total)."""
    sig = r13.above_sma(r13.tr_index(d))
    idx = d["u"].index
    R = {ONE: d["u"], THREE: d["x3"], LRS: r13.run_weights(d, sig), ROT: r13.run_weights(d, sig, 1 - sig)}
    E = {ONE: pd.Series(1.0, index=idx), THREE: pd.Series(3.0, index=idx), LRS: 3 * sig, ROT: 3 * sig + (1 - sig)}
    return R, E


def rule_weights(d: dict, F: dict, vix: pd.Series | None = None, ts: pd.Series | None = None,
                 prob: dict | None = None) -> tuple[dict, dict]:
    """Pesos (w3, w1) decididos al cierre de t de cada regla nueva y su grupo (GROUPS).

    Las reglas de (a) y (b) empiezan el mismo día (el primero con los cinco umbrales); las del
    VIX, cuando existe su mediana expansiva; las de VIX/VIX3M, con el VIX3M."""
    idx = d["u"].index
    sig = r13.above_sma(r13.tr_index(d))
    t0 = oos_start(F)
    lows = {(m, qn): low_vol(F[m], t0, q) for m in MODELS for qn, q in QS.items()}
    start = max(s.first_valid_index() for s in lows.values())
    W, G = {}, {}

    def add(name, w3, w1, g, s0=None):
        if s0 is not None:
            w3, w1 = w3.where(idx >= s0), w1.where(idx >= s0)
        W[name], G[name] = (w3, w1), g

    zero = pd.Series(0.0, index=idx)
    for (m, qn), low in lows.items():
        add(f"Filtro vol {m} ({qn}): 3x/letras", low, zero.where(low.notna()), "a", start)
        add(f"Filtro vol {m} ({qn}): 3x/1x", low, 1 - low, "a", start)
    for (m, qn), low in lows.items():
        add(f"LRS × vol {m} ({qn})", sig * low, sig * (1 - low), "b", start)
    if vix is not None:
        v20 = (vix < VIX_LEVEL).astype(float).where(vix.notna())
        vmed = vix.expanding(MIN_THR).median()
        vm = (vix < vmed).astype(float).where(vmed.notna())
        s_v = vmed.first_valid_index()
        v20s = f"{VIX_LEVEL:g}"
        add(f"VIX < {v20s}: 3x/letras", v20, zero.where(v20.notna()), "c", s_v)
        add(f"VIX < {v20s}: 3x/1x", v20, 1 - v20, "c", s_v)
        add("VIX < su mediana: 3x/letras", vm, zero.where(vm.notna()), "c", s_v)
        add("VIX < su mediana: 3x/1x", vm, 1 - vm, "c", s_v)
        add(f"LRS × VIX < {v20s}", sig * v20, sig * (1 - v20), "c", s_v)
        add("LRS × VIX < su mediana", sig * vm, sig * (1 - vm), "c", s_v)
    if ts is not None:
        c = (ts < 1).astype(float).where(ts.notna())
        add("VIX/VIX3M < 1: 3x/letras", c, zero.where(c.notna()), "t")
        add("VIX/VIX3M < 1: 3x/1x", c, 1 - c, "t")
        add("LRS × VIX/VIX3M < 1", sig * c, sig * (1 - c), "t")
    for name, p in (prob or {}).items():
        w = (p >= CLF_P).astype(float).where(p.notna())
        add(name, w, zero.where(w.notna()), "d")
    return W, G


def build(d: dict, F: dict, vix=None, ts=None, prob=None) -> tuple[dict, dict, dict]:
    """Rentabilidades diarias netas R, exposición decidida E (3·w3 + w1) y grupo de cada regla,
    con las cuatro referencias (1x, 3x, LRS original y rotación) en R y E."""
    R, E = bench_returns(d)
    W, G = rule_weights(d, F, vix, ts, prob)
    for k, (w3, w1) in W.items():
        R[k] = r13.run_weights(d, w3, w1, lag=LAG)
        E[k] = 3 * w3 + w1
    return R, E, G


def trading(e: pd.Series, a=None, b=None) -> dict:
    e = e.loc[a:b].dropna()
    ch = e.diff().abs() > 1e-9
    return {"exposición_media": float(e.mean()), "tiempo_en_3x": float((e >= 3 - 1e-9).mean()),
            "operaciones_año": float(ch.sum() / (len(e) / 252))}


# ───────────────────────────── inferencia ─────────────────────────────

def log_ret(r: pd.Series) -> pd.Series:
    return np.log1p(r.clip(lower=-0.9999))


def boot_draws(X: np.ndarray, n_boot: int, rng, mean_block: float = BLOCK) -> np.ndarray:
    """n_boot × K medias de un bootstrap estacionario (stats.stationary_indices, los mismos
    remuestreos para todas las columnas). Los NaN son fechas en que esa columna no existe: la
    media de cada remuestreo es la de sus fechas válidas (igual que np.nanmean(X[idx])). Se
    suma por bloques contiguos con la suma acumulada, así que cada remuestreo cuesta O(bloques·K)."""
    X = np.asarray(X, float)
    T, K = X.shape
    ok = np.isfinite(X)
    C = np.vstack([np.zeros((1, K)), np.cumsum(np.where(ok, X, 0.0), axis=0)])
    N = np.vstack([np.zeros((1, K)), np.cumsum(ok, axis=0)])
    out = np.empty((n_boot, K))
    for b in range(n_boot):
        idx = stats.stationary_indices(T, mean_block, rng)
        cut = np.r_[0, np.flatnonzero(np.diff(idx) != 1) + 1]
        s = idx[cut]
        e = s + np.diff(np.r_[cut, T])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[b] = (C[e] - C[s]).sum(axis=0) / (N[e] - N[s]).sum(axis=0)
    return out


def paired(R: dict, names: list[str], bench: str, a=None, b=None, rng=None,
           n_boot: int = B) -> pd.DataFrame:
    """Δg anual (regla − referencia) en las fechas comunes de [a, b] con IC 90% percentil y
    p unilateral normal con el error estándar del bootstrap (H0: Δg ≤ 0), como R19."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    lb = log_ret(R[bench]).loc[a:b]
    X = pd.DataFrame({k: log_ret(R[k]).loc[a:b] - lb for k in names})
    X = X.loc[X.notna().any(axis=1)]
    m = X.mean().to_numpy()
    draws = boot_draws(X.to_numpy(), n_boot, rng)
    se = np.nanstd(draws, axis=0, ddof=1)
    lo, hi = np.nanpercentile(draws, [5, 95], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(se > 0, norm.sf(m / se), np.nan)
    return pd.DataFrame({"Δg": m * 252, "IC90_bajo": lo * 252, "IC90_alto": hi * 252, "p": p},
                        index=names)


def spa(R: dict, cands: list[str], bench: str, a=None, b=None, rng=None) -> dict:
    """SPA de Hansen (stats.spa_test, recentrado consistente) de la familia `cands` frente a
    `bench` en la ventana común de [a, b] (fechas en que existen todas)."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    cc = [c for c in cands if c != bench]
    L = pd.DataFrame({k: log_ret(R[k]).loc[a:b] for k in cc + [bench]}).dropna()
    Dm = L[cc].sub(L[bench], axis=0).to_numpy()
    out = stats.spa_test(Dm, BLOCK, B, rng)
    j = int(Dm.mean(axis=0).argmax())
    return {"frente a": bench, "reglas": len(cc), "desde": L.index[0].date(), "hasta": L.index[-1].date(),
            "mejor": cc[j], "Δg_mejor": float(Dm.mean(axis=0)[j] * 252),
            "p_SPA": out["p_SPA_c"], "p_SPA_l": out["p_SPA_l"]}


def rule_table(R: dict, E: dict, d: dict, names: list[str], a=None, b=None, rng=None) -> pd.DataFrame:
    """Métricas de cada regla en [a, b] (desde su primer día) y Δg frente a la LRS original,
    frente al 1x y frente al 3x comprar y mantener, con su IC y su p."""
    rows = {}
    for k in names:
        r = R[k].loc[a:b].dropna()
        if len(r) < 250:
            continue
        m = metrics.summary(r, d["rf"])
        rows[k] = {"desde": r.index[0].date(), "CAGR": m["CAGR"], "vol": m["vol"], "sharpe": m["sharpe"],
                   "maxDD": m["maxDD"], **trading(E[k], r.index[0], b)}
    t = pd.DataFrame(rows).T
    keep = list(t.index)
    for tag, bench in (("LRS", LRS), ("1x", ONE), ("3x", THREE)):
        pk = [k for k in keep if k != bench]
        pt = paired(R, pk, bench, a, b, rng)
        for c in pt:
            t.loc[pk, f"{c}_vs_{tag}"] = pt[c]
    return t


# ───────────────────────────── (b) frente a bajar el apalancamiento ─────────────────────────────

def lrs_const(d: dict, sig: pd.Series, L: float) -> pd.Series:
    """LRS con un apalancamiento fijo L sobre la media y letras debajo, con los instrumentos de
    las reglas (b): L ≤ 1, L en el 1x y el resto en letras; 1 < L ≤ 3, (L − 1)/2 en el 3x y el
    resto en el 1x (reequilibrado a diario sin coste, como las referencias de R17)."""
    if L <= 1:
        return r13.run_weights(d, sig * 0, sig * L, lag=LAG)
    w3 = (L - 1) / 2
    return r13.run_weights(d, sig * w3, sig * (1 - w3), lag=LAG)


def equal_risk(d: dict, R: dict, E: dict, name: str, a=None, b=None, rng=None) -> dict:
    """La regla (b) `name` frente a la LRS con un apalancamiento fijo: el de su misma exposición
    media sobre la media (L_exp) y el de su misma volatilidad realizada (L_vol), los dos fijados
    con la ventana entera (son cotas: nadie los conocía de antemano)."""
    sig = r13.above_sma(r13.tr_index(d))
    r = R[name].loc[a:b].dropna()
    win = r.index
    applied = E[name].shift(LAG).reindex(win)
    above = sig.shift(LAG).reindex(win) == 1
    L_exp = float(applied[above].mean())
    vol_r = float(r.std())
    f = lambda L: float(lrs_const(d, sig, L).reindex(win).std()) - vol_r  # noqa: E731
    L_vol = float(brentq(f, 0.05, 3.0, xtol=1e-5)) if f(0.05) < 0 < f(3.0) else np.nan
    refs = {"L_exp": lrs_const(d, sig, L_exp), "L_vol": lrs_const(d, sig, L_vol) if np.isfinite(L_vol) else None}
    out = {"regla": name, "desde": win[0].date(), "L_exp": L_exp, "L_vol": L_vol,
           "CAGR_regla": metrics.cagr(r), "vol_regla": vol_r * np.sqrt(252), "maxDD_regla": metrics.max_dd(r)}
    for tag, ref in refs.items():
        if ref is None:
            continue
        x = ref.reindex(win)
        pt = paired({name: r, tag: x}, [name], tag, rng=rng).iloc[0]
        out.update({f"CAGR_{tag}": metrics.cagr(x), f"vol_{tag}": float(x.std() * np.sqrt(252)),
                    f"maxDD_{tag}": metrics.max_dd(x), f"Δg_vs_{tag}": pt["Δg"],
                    f"IC90_bajo_vs_{tag}": pt["IC90_bajo"], f"IC90_alto_vs_{tag}": pt["IC90_alto"],
                    f"p_vs_{tag}": pt["p"]})
    return out


def lrs_frontier(d: dict, win: pd.DatetimeIndex, Ls=np.arange(0.25, 3.001, 0.25)) -> pd.DataFrame:
    sig = r13.above_sma(r13.tr_index(d))
    rows = []
    for L in Ls:
        x = lrs_const(d, sig, float(L)).reindex(win)
        rows.append({"L": float(L), "CAGR": metrics.cagr(x), "vol": float(x.std() * np.sqrt(252)),
                     "maxDD": metrics.max_dd(x)})
    return pd.DataFrame(rows)


# ───────────────────────────── mecanismo: tendencia × volatilidad ─────────────────────────────

def quadrant_frame(d: dict, F: dict, model: str = MAIN_MODEL, a=None, b=None) -> pd.DataFrame:
    """Estado al cierre de t (sobre/bajo la media de 200 del índice de rentabilidad total;
    volatilidad prevista bajo/sobre su mediana expansiva fuera de muestra; su rango percentil
    expansivo) y rentabilidades del día t + 2, el que cobra una orden al cierre de t+1."""
    sig = r13.above_sma(r13.tr_index(d))
    t0 = oos_start(F)
    fc = F[model].where(F[model].index >= t0)
    low = low_vol(F[model], t0)
    rank = fc.dropna().expanding(MIN_THR).rank(pct=True).reindex(fc.index)
    prem = kelly.premium(d["u"], d["f"])
    lstar_ex_ante = (prem - d["spread"]) / fc
    df = pd.DataFrame({"sobre_media": sig.shift(LAG), "vol_baja": low.shift(LAG), "rango": rank.shift(LAG),
                       "L_ex_ante": lstar_ex_ante.shift(LAG), "u": d["u"], "x3": d["x3"], "rf": d["rf"],
                       "f": d["f"]}).loc[a:b]
    return df.dropna(subset=["sobre_media", "vol_baja", "u", "x3", "rf", "f"])


QUAD = {(1, 1): "sobre la media · vol. prevista baja", (1, 0): "sobre la media · vol. prevista alta",
        (0, 1): "bajo la media · vol. prevista baja", (0, 0): "bajo la media · vol. prevista alta"}


def cell_stats(df: pd.DataFrame, m: pd.Series, spread: float) -> dict:
    s = df[m]
    l1, l3, lr = log_ret(s.u), log_ret(s.x3), log_ret(s.rf)
    return {"días": float(m.mean()), "años": float(m.sum() / 252), "μ_1x": float(s.u.mean() * 252),
            "σ_1x": float(s.u.std() * np.sqrt(252)), "g_1x": float(l1.mean() * 252),
            "g_3x": float(l3.mean() * 252), "g_letras": float(lr.mean() * 252),
            "prima_3x_1x": float((l3 - l1).mean() * 252), "prima_3x_letras": float((l3 - lr).mean() * 252),
            kelly.LSTAR_APROX: kelly.lstar_approx(s.u, s.f, spread),
            "L_ex_ante_medio": float(s.L_ex_ante.mean())}


def quadrant_table(d: dict, df: pd.DataFrame, rng, n_boot: int = B) -> pd.DataFrame:
    """Prima del 3x (g_3x − g_1x) en cada cuadrante tendencia × volatilidad, con IC 90% por
    bootstrap estacionario de la ventana (cociente de medias de cada celda)."""
    masks = {lab: (df.sobre_media == a) & (df.vol_baja == v) for (a, v), lab in QUAD.items()}
    masks["todos los días"] = pd.Series(True, index=df.index)
    diff = (log_ret(df.x3) - log_ret(df.u)).to_numpy()
    cols = []
    for m in masks.values():
        mm = m.to_numpy(float)
        cols += [mm * diff, mm]
    draws = boot_draws(np.column_stack(cols), n_boot, rng)
    rows = {}
    for j, (lab, m) in enumerate(masks.items()):
        ratio = draws[:, 2 * j] / draws[:, 2 * j + 1] * 252
        lo, hi = np.nanpercentile(ratio, [5, 95])
        rows[lab] = {**cell_stats(df, m, d["spread"]), "IC90_bajo": lo, "IC90_alto": hi}
    return pd.DataFrame(rows).T


def quintile_table(d: dict, df: pd.DataFrame) -> pd.DataFrame:
    q = pd.cut(df.rango, [0, .2, .4, .6, .8, 1.0], labels=["Q1 (más baja)", "Q2", "Q3", "Q4", "Q5 (más alta)"],
               include_lowest=True)
    rows = {}
    for a, an in ((1, "sobre la media"), (0, "bajo la media")):
        for lab in q.cat.categories:
            m = (df.sobre_media == a) & (q == lab)
            if m.sum() > 100:
                rows[(an, lab)] = cell_stats(df, m, d["spread"])
    return pd.DataFrame(rows).T


def premium_regression(d: dict, F: dict, a=None, b=None) -> pd.DataFrame:
    """Regresión de la prima del 3x sobre el 1x de las 21 sesiones que cobra una posición
    decidida en t (Σ_{t+2}^{t+22} [log(1+x3) − log(1+u)]) sobre cuatro indicadores de estado:
    bajo la media, sobre la media, y la volatilidad alta dentro de cada uno. Sin constante: el
    coeficiente de «sobre la media» es la prima con volatilidad baja y el de «× vol. alta» la
    diferencia con volatilidad alta. Errores HAC para objetivos solapados (stats.hac_overlap)."""
    df = quadrant_frame(d, F, a=a, b=b)
    idx = d["u"].index
    lx = log_ret(d["x3"]) - log_ret(d["u"])
    y = lx.rolling(H).sum().shift(-EMBARGO).reindex(df.index)
    sig = r13.above_sma(r13.tr_index(d)).reindex(df.index)
    t0 = oos_start(F)
    low = low_vol(F[MAIN_MODEL], t0).reindex(df.index)
    hi = 1 - low
    X = pd.DataFrame({"bajo la media": 1 - sig, "sobre la media": sig,
                      "sobre la media × vol. alta": sig * hi, "bajo la media × vol. alta": (1 - sig) * hi},
                     index=df.index)
    del idx
    ok = y.notna()
    for base in ("bajo la media", "sobre la media"):
        # sin días con volatilidad baja en ese lado (p. ej. bajo la media desde 2016), la
        # interacción repetiría la columna: se quita y el coeficiente es el de la volatilidad alta
        inter = f"{base} × vol. alta"
        if ((X[base] - X[inter])[ok] > 0).sum() < 2 * H:
            X = X.drop(columns=inter).rename(columns={base: f"{base} (solo vol. alta)"})
    fit = stats.ols_hac(y, X, const=False, **stats.hac_overlap(H))
    t = fit.frame()
    t["coef"] *= 252 / H
    t["se"] *= 252 / H
    t.insert(0, "n", fit.n)
    return t


# ───────────────────────────── carrera de previsiones ─────────────────────────────

def forecast_race(d: dict, F: dict, a=None, b=None, rng=None) -> pd.DataFrame:
    """Calidad fuera de muestra de las cinco previsiones de la varianza de las 21 sesiones
    siguientes: pérdida QLIKE media (Patton 2011), Model Confidence Set al 10% (Hansen, Lunde
    y Nason 2011; bloque 63), R² en logaritmos y sesgo (media de la previsión / media de la
    realizada)."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    y = realized_target(d["u"])
    V = pd.DataFrame({m: F[m] for m in MODELS})
    V["realizada"] = y
    V = V.loc[max(oos_start(F), pd.Timestamp(a)) if a is not None else oos_start(F):b].dropna()
    yy = V.realizada.to_numpy()
    ql = pd.DataFrame({m: stats.qlike(yy, V[m].to_numpy()) for m in MODELS}, index=V.index)
    out = stats.model_confidence_set(ql, BLOCK, B_MCS, rng)
    ly = np.log(yy)
    for m in MODELS:
        lf = np.log(V[m].to_numpy())
        out.loc[m, "R2_log"] = 1 - np.mean((ly - lf) ** 2) / np.var(ly)
        out.loc[m, "sesgo"] = V[m].mean() / V.realizada.mean()
    out.insert(0, "desde", V.index[0].date())
    out.insert(1, "hasta", V.index[-1].date())
    return out


# ───────────────────────────── mercados ─────────────────────────────

def _cached(cache: str | None, key: str, fn):
    """Previsiones y probabilidades de un mercado, con una caché opcional en disco (pickle)."""
    if cache:
        p = Path(cache) / f"r23_{key}.pkl"
        if p.exists():
            with open(p, "rb") as fh:
                return pickle.load(fh)
        out = fn()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(out, fh)
        return out
    return fn()


def market_inputs(d: dict, key: str, cache: str | None = None) -> tuple[dict, dict, pd.Series, pd.Series]:
    """Previsiones F, probabilidades de los clasificadores, VIX y VIX/VIX3M de un mercado."""
    vix, ts = vix_series(d["u"].index)

    def fn():
        t0 = time.time()
        F = forecasts(d)
        t1 = time.time()
        prob = classifiers(d, F, vix)
        return {"F": F, "prob": prob, "segundos": {"previsiones": t1 - t0, "clasificadores": time.time() - t1}}

    x = _cached(cache, key, fn)
    return x["F"], x["prob"], vix, ts


def long_names(G: dict) -> list[str]:
    """Reglas que existen desde el principio (sin VIX): (a), (b) y los clasificadores sin VIX."""
    return [k for k, g in G.items() if g in ("a", "b") or (g == "d" and "VIX" not in k)]


def spa_block(R: dict, G: dict, windows: dict, rng) -> list[dict]:
    """SPA de la familia en cada ventana: frente a la LRS original, frente al 1x (solo las
    reglas nuevas, y además con la LRS original dentro) y frente al 3x comprar y mantener."""
    rows = []
    for wname, (fam, a) in windows.items():
        for bench, extra in ((LRS, []), (ONE, []), (ONE, [LRS]), (THREE, [])):
            s = spa(R, fam + extra, bench, a, None, rng)
            rows.append({"ventana": wname, "familia": "reglas nuevas" + (" + LRS original" if extra else ""), **s})
    return rows


def market_windows(name: str, d: dict, R: dict, G: dict) -> tuple[dict, dict]:
    """Ventanas de las tablas y del SPA de cada mercado."""
    long = long_names(G)
    start = max(R[k].first_valid_index() for k in long)
    if d.get("eval_from") is not None:
        start = max(start, d["eval_from"])
    last = str(d["u"].index[-1].year)
    if name == "S&P 500":
        # ninguna ventana empieza antes que las reglas de volatilidad (las referencias, tampoco)
        tw = {k: (max(pd.Timestamp(a), start) if a is not None else start, b) for k, (a, b) in SP_WINDOWS.items()}
    elif name == "Nasdaq-100":
        tw = {"completa": (start, None), "1985–1999": (start, "1999-12-31"), "2000–2026": ("2000-01-01", None),
              "2016–2026 (tras la muestra de Gayed y Bilello)": ("2016-01-04", None),
              "ETFs reales (2010–2026)": (REAL_FROM, None)}
    else:
        tw = {"completa": (start, None), "2016–2026 (tras la muestra de Gayed y Bilello)": ("2016-01-04", None),
              "ETFs reales (2010–2026)": (REAL_FROM, None)}
    no_ts = [k for k in G if G[k] != "t"]
    vix_start = max(R[k].first_valid_index() for k in no_ts)
    sw = {f"historia larga ({start.year}–{last})": (long, start),
          f"era del VIX ({max(vix_start, start).year}–{last})": (no_ts, max(vix_start, start)),
          f"ETFs reales ({REAL_FROM[:4]}–{last})": (list(G), pd.Timestamp(REAL_FROM)),
          f"2016–{last}": (list(G), pd.Timestamp("2016-01-04"))}
    return tw, sw


def market_eval(name: str, d: dict, F: dict, prob: dict, vix, ts, rng, full: bool = True) -> dict:
    """Todas las tablas de un mercado. full=False (reconstrucciones del Nasdaq-100 distintas de
    la primera) omite la carrera de previsiones, los IC del AUC y los cuadrantes."""
    R, E, G = build(d, F, vix, ts, prob)
    tw, sw = market_windows(name, d, R, G)
    names = list(G) + [THREE, ROT, LRS, ONE]
    tables = {}
    for wname, (a, b) in tw.items():
        t = rule_table(R, E, d, names, a, b, rng)
        t.insert(0, "grupo", [GROUPS.get(G.get(k), "referencia") for k in t.index])
        tables[wname] = t
    out = {"name": name, "R": R, "E": E, "G": G, "tablas": tables, "ventanas": tw,
           "spa": pd.DataFrame(spa_block(R, G, sw, rng)), "d_keys": {k: d[k] for k in ("spread", "er")}}
    y = clf_target(d)
    X = clf_features(d, F["GJR_prefijo"])
    extra = {"distancia a la media": X["distancia_media_200"], "−vol. relativa": -X["vol_relativa"]}
    aw = {"completa": (tw["completa"][0], None), "2000–2026": ("2000-01-01", None),
          "ETFs reales (2010–2026)": (REAL_FROM, None)}
    if full:
        out["auc"] = auc_table(prob, y, extra, aw, rng)
        out["carrera"] = forecast_race(d, F, tw["completa"][0], rng=rng)
        out["egarch"] = pd.DataFrame(F["egarch_fits"])
        df = quadrant_frame(d, F, a=tw["completa"][0])
        out["cuadrantes"] = quadrant_table(d, df, rng)
    else:
        rows = []
        for k, p in prob.items():
            for wname, (a, b) in aw.items():
                m = pd.concat([p, y], axis=1, keys=["p", "y"]).loc[a:b].dropna()
                if len(m) >= 500:
                    rows.append({"clasificador": k, "ventana": wname, "AUC": auc(m.y.to_numpy(), m.p.to_numpy()),
                                 "fracción_en_3x": float((m.p >= CLF_P).mean())})
        out["auc"] = pd.DataFrame(rows)
    return out


def _ndx_draw(args) -> dict:
    """Una reconstrucción del Nasdaq-100 (para el grupo de procesos): datos como
    r13.ndx_imputations, previsiones, clasificadores y tablas."""
    k, u, cache, seed = args
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        real = r12.real_letf("TQQQ")
        d = r12.make(u, history.rf_long(u.index), "TQQQ", real, eval_from=r12.NDX_EVAL_FROM)
        d["px"] = r13.price_index(d, "Nasdaq-100")
        F, prob, vix, ts = market_inputs(d, f"ndx{k}", cache)
        res = market_eval("Nasdaq-100", d, F, prob, vix, ts, np.random.default_rng(seed), full=(k == 0))
    if k != 0:
        res.pop("R")
        res.pop("E")
    res["k"] = k
    return res


def ndx_all(nd: dict, cache: str | None, n_jobs: int = 5) -> list[dict]:
    args = [(k, u, cache, SEED + 100 + k) for k, u in enumerate(nd["draws"][:N_NDX])]
    if n_jobs <= 1:
        return [_ndx_draw(a) for a in args]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(_ndx_draw, args))


def ndx_summary(res: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mediana (y rango) entre las reconstrucciones del Nasdaq-100 de las métricas de cada
    regla y ventana, de los p del SPA y de los AUC."""
    T = []
    for r in res:
        for w, t in r["tablas"].items():
            x = t.copy()
            x.insert(0, "ventana", w)
            x.insert(0, "k", r["k"])
            T.append(x)
    T = pd.concat(T)
    T.index.name = "regla"
    num = [c for c in ("CAGR", "maxDD", "tiempo_en_3x", "Δg_vs_LRS", "p_vs_LRS", "Δg_vs_1x", "p_vs_1x",
                       "Δg_vs_3x", "p_vs_3x") if c in T]
    T[num] = T[num].astype(float)
    agg = T.reset_index().groupby(["ventana", "regla"], sort=False)[num].median()
    rng_ = T.reset_index().groupby(["ventana", "regla"], sort=False)[["Δg_vs_LRS"]].agg(["min", "max"])
    agg["Δg_vs_LRS_mín"] = rng_[("Δg_vs_LRS", "min")]
    agg["Δg_vs_LRS_máx"] = rng_[("Δg_vs_LRS", "max")]
    S = pd.concat([r["spa"].assign(k=r["k"]) for r in res])
    Sg = S.groupby(["ventana", "familia", "frente a"], sort=False).agg(
        reglas=("reglas", "first"), p_SPA_mediana=("p_SPA", "median"), p_SPA_mín=("p_SPA", "min"),
        p_SPA_máx=("p_SPA", "max"), mejor_más_frecuente=("mejor", lambda x: x.mode().iloc[0]),
        Δg_mejor_mediana=("Δg_mejor", "median"))
    A = pd.concat([r["auc"].assign(k=r["k"]) for r in res])
    Ag = A.groupby(["clasificador", "ventana"], sort=False).agg(AUC_mediana=("AUC", "median"), AUC_mín=("AUC", "min"),
                                                                 AUC_máx=("AUC", "max"),
                                                                 fracción_en_3x=("fracción_en_3x", "median"))
    return agg, Sg, Ag


# ───────────────────────────── censo ─────────────────────────────

def census_rows(D: dict | None = None, cache: str | None = None) -> dict[str, pd.Series]:
    """S&P 500: rentabilidad diaria neta de cada regla nueva de R23 (43), para añadirlas al censo
    de R19. Mismo índice que el censo (d['u'].index del S&P 500 de r13.load(), 1928–2026), NaN
    antes del primer día de cada regla; nombres con el prefijo CENSUS_PREFIX. Cada serie lleva en
    .attrs su grupo ('grupo') y su primer día ('vivo_desde'): las del VIX empiezan en 1991, los
    clasificadores con VIX en 2000 y las de VIX/VIX3M en 2006, así que en la ventana común del
    censo (1939-04 →) solo caben las 32 de historia larga (grupos a, b y d sin VIX).
    cache: directorio de la caché de run() (evita recalcular previsiones y clasificadores)."""
    D = D if D is not None else r13.load()
    d = D["S&P 500"]
    F, prob, vix, ts = market_inputs(d, "sp500", cache)
    R, _, G = build(d, F, vix, ts, prob)
    out = {}
    for k, g in G.items():
        s = R[k].reindex(d["u"].index).rename(CENSUS_PREFIX + k)
        s.attrs = {"grupo": GROUPS[g], "vivo_desde": s.first_valid_index()}
        out[CENSUS_PREFIX + k] = s
    return out


def compute(cache: str | None = None, n_jobs: int = 5) -> dict:
    """Todos los cálculos del informe (sin figuras ni texto)."""
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    D = r13.load()
    nd = D.pop("_ndx")
    d = D["S&P 500"]
    res = {"segundos": {}}
    # comprobación inicial: R13 desde 1938
    R13, _ = r13.strategies(d, r13.var_forecast(d))
    res["sanity"] = r13.stats_table({k: R13[k] for k in (ONE, THREE, LRS)}, d["rf"], "1938-02-04")
    F, prob, vix, ts = market_inputs(d, "sp500", cache)
    sp = market_eval("S&P 500", d, F, prob, vix, ts, rng)
    Rb, _ = bench_returns(d)
    sp["sanity_diff"] = max(float((Rb[k] - R13[k]).abs().max()) for k in (ONE, THREE, LRS, ROT))
    # carrera de previsiones también desde 2000
    sp["carrera_2000"] = forecast_race(d, F, "2000-01-01", rng=rng)
    # mecanismo por subperiodos y regresión
    start = sp["ventanas"]["completa"][0]
    sub = {"1938–1973": (start, "1973-12-31"), "1974–1999": ("1974-01-01", "1999-12-31"),
           "2000–2026": ("2000-01-01", None), "2016–2026": ("2016-01-04", None)}
    sp["cuadrantes_sub"] = {k: quadrant_table(d, quadrant_frame(d, F, a=a, b=b), rng) for k, (a, b) in sub.items()}
    sp["cuadrantes_modelos"] = {m: quadrant_table(d, quadrant_frame(d, F, model=m, a=start), rng, 500)
                                for m in MODELS if m != MAIN_MODEL}
    sp["quintiles"] = quintile_table(d, quadrant_frame(d, F, a=start))
    sp["regresión"] = premium_regression(d, F, a=start)
    sp["regresión_sub"] = {k: premium_regression(d, F, a=a, b=b) for k, (a, b) in sub.items()}
    # (b) frente a bajar el apalancamiento
    bn = [k for k, g in sp["G"].items() if g == "b"]
    sp["igual_riesgo"] = pd.DataFrame([equal_risk(d, sp["R"], sp["E"], k, start, None, rng) for k in bn])
    sp["igual_riesgo_2016"] = pd.DataFrame([equal_risk(d, sp["R"], sp["E"], k, "2016-01-04", None, rng)
                                            for k in bn])
    sp["frontera"] = lrs_frontier(d, sp["R"][bn[0]].loc[start:].dropna().index)
    # SPA por grupos (S&P 500)
    G = sp["G"]
    grp_w = {"a": start, "b": start, "d": start, "c": sp["R"]["VIX < su mediana: 3x/letras"].first_valid_index(),
             "t": sp["R"]["VIX/VIX3M < 1: 3x/letras"].first_valid_index()}
    rows = []
    for g, a in grp_w.items():
        fam = [k for k in G if G[k] == g and (g != "d" or "VIX" not in k)]
        for bench in (LRS, ONE):
            rows.append({"grupo": GROUPS[g], **spa(sp["R"], fam, bench, a, None, rng)})
    sp["spa_grupos"] = pd.DataFrame(rows)
    # Holm sobre los p de la ventana completa
    t = sp["tablas"]["completa"]
    new = [k for k in G]
    sp["holm"] = {tag: int((stats.holm(t.loc[new, f"p_vs_{tag}"].astype(float).to_numpy()) < 0.05).sum())
                  for tag in ("LRS", "1x")}
    sp["nominal"] = {tag: int((t.loc[new, f"p_vs_{tag}"].astype(float) < 0.05).sum()) for tag in ("LRS", "1x")}
    sp["vix_on"] = {"2009": float((sp["E"]["VIX < 20: 3x/letras"].loc["2009-03-10":"2009-12-31"] >= 3).mean()),
                    "2020": float((sp["E"]["VIX < 20: 3x/letras"].loc["2020-03-24":"2020-12-31"] >= 3).mean()),
                    "contango": float((ts.dropna() < 1).mean())}
    res["sp"] = sp
    res["segundos"]["S&P 500"] = time.time() - t0
    # Nasdaq-100: 10 reconstrucciones
    t1 = time.time()
    nres = ndx_all(nd, cache, n_jobs)
    res["ndx0"] = next(r for r in nres if r["k"] == 0)
    res["ndx"] = ndx_summary(nres)
    res["segundos"]["Nasdaq-100"] = time.time() - t1
    # semis
    t2 = time.time()
    ds = D["Semis"]
    F_s, prob_s, vix_s, ts_s = market_inputs(ds, "semis", cache)
    res["semis"] = market_eval("Semis", ds, F_s, prob_s, vix_s, ts_s, np.random.default_rng(SEED + 7))
    res["segundos"]["Semis"] = time.time() - t2
    res["segundos"]["total"] = time.time() - t0
    return res


# ───────────────────────────── figuras ─────────────────────────────

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # paleta categórica validada (3 tonos)
INK, GREY = "#0b0b0b", "#8a8984"
GROUP_COLOR = {"a": GREY, "b": BLUE, "c": ORANGE, "t": ORANGE, "d": AQUA}
GROUP_MARK = {"a": "o", "b": "s", "c": "D", "t": "v", "d": "^"}


def _quad_label(lab: str) -> str:
    side, vol = lab.split(" · ")
    return f"{side}\n{vol.replace('vol. prevista ', 'volatilidad ')}"


def fig_mechanism(sp: dict, rep: Report):
    """Prima del 3x sobre el 1x (g_3x − g_1x) por cuadrante, con su IC 90%, y por quintil de la
    volatilidad GJR prevista, sobre y bajo la media."""
    qt, qq = sp["cuadrantes"], sp["quintiles"]
    start = sp["ventanas"]["completa"][0]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 10.5))
    ax = axes[0]
    labs = list(QUAD.values())
    y = qt.loc[labs, "prima_3x_1x"].astype(float) * 100
    lo = qt.loc[labs, "IC90_bajo"].astype(float) * 100
    hi = qt.loc[labs, "IC90_alto"].astype(float) * 100
    cols = [BLUE, BLUE, ORANGE, ORANGE]
    x = np.arange(4)
    bars = ax.bar(x, y, color=cols, width=0.62, edgecolor="white", linewidth=2)
    for b_, h in zip(bars, ("", "//", "", "//")):
        b_.set_hatch(h)
        b_.set_alpha(1.0 if h == "" else 0.75)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=INK, capsize=5, lw=1.2)
    for xi, lab in zip(x, labs):
        ax.text(xi, hi.loc[lab] + 1.5 if y.loc[lab] >= 0 else lo.loc[lab] - 1.5,
                f"{pct_es(qt.loc[lab, 'días'], 0)} de los días",
                ha="center", va="bottom" if y.loc[lab] >= 0 else "top", fontsize=11, color="#52514e")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, [_quad_label(l) for l in labs], fontsize=11)
    ax.set(ylabel="g del 3x − g del 1x (pp/año)",
           title=f"Prima del 3x sobre el 1x por cuadrante\n(S&P 500, {start.year}–{d_end(sp)}; IC 90%)")
    pad = (hi.max() - lo.min()) * 0.12
    ax.set_ylim(lo.min() - pad * 1.5, hi.max() + pad * 1.5)
    ax = axes[1]
    qn = qq.index.get_level_values(1).unique()
    xs = np.arange(len(qn))
    w = 0.38
    for j, (side, col) in enumerate((("sobre la media", BLUE), ("bajo la media", ORANGE))):
        vals = [float(qq.loc[(side, q), "prima_3x_1x"]) * 100 if (side, q) in qq.index else np.nan for q in qn]
        dias = [float(qq.loc[(side, q), "días"]) if (side, q) in qq.index else np.nan for q in qn]
        bx = xs + (j - 0.5) * w
        ax.bar(bx, vals, width=w, color=col, edgecolor="white", linewidth=2, label=side)
        for xi, v, dd in zip(bx, vals, dias):
            if np.isfinite(v):
                ax.text(xi, v + (1.2 if v >= 0 else -1.2), pct_es(dd, 0), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=11, color="#52514e")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(xs, [q.replace(" (", "\n(") for q in qn], fontsize=11)
    ax.set(ylabel="g del 3x − g del 1x (pp/año)", xlabel="Quintil de la volatilidad GJR prevista (frente a su historia)",
           title="Por quintil de la volatilidad prevista (sobre las barras, % de los días)")
    lo2, hi2 = ax.get_ylim()
    ax.set_ylim(lo2 - 4, hi2 + 4)
    h, l = ax.get_legend_handles_labels()
    legend_band(fig, h, l, where="bottom", ncol=2)
    comma_fig(fig)
    rep.fig(fig, "prima_por_estado", "Prima del 3x sobre el 1x (crecimiento logarítmico anual) según la tendencia "
            "(media de 200 sesiones) y la volatilidad GJR prevista en el S&P 500; estado al cierre de t y "
            "rentabilidad del día t + 2")


def d_end(res: dict) -> int:
    """Año de la última sesión del mercado."""
    return int(res["R"][LRS].dropna().index[-1].year)


def fig_frontier(sp: dict, rep: Report):
    """Rentabilidad frente a volatilidad y frente a caída máxima: la LRS con apalancamiento fijo
    (línea) y las reglas (a) y (b)."""
    t = sp["tablas"]["completa"]
    fr = sp["frontera"]
    G = sp["G"]
    start = sp["ventanas"]["completa"][0]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 10))
    for ax, xc, xl in ((axes[0], "vol", "Volatilidad anual (%)"), (axes[1], "maxDD", "Caída máxima (%)")):
        ax.plot(fr[xc] * 100, fr.CAGR * 100, color=INK, lw=1.6, zorder=2,
                label="LRS con apalancamiento fijo L (sobre la media; letras debajo)")
        for L in (1.0, 2.0, 3.0):
            r = fr[np.isclose(fr.L, L)].iloc[0]
            ax.plot(r[xc] * 100, r.CAGR * 100, "o", ms=5, color=INK, zorder=2.5)
            ax.annotate(f"L = {num_es(L, 0)}", (r[xc] * 100, r.CAGR * 100),
                        xytext=(-14, 12) if xc == "vol" else (10, 8), textcoords="offset points", fontsize=11,
                        color=INK, ha="right" if xc == "vol" else "left", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        for g, lab in (("a", "(a) filtro de régimen"), ("b", "(b) LRS × volatilidad")):
            ks = [k for k in G if G[k] == g and k in t.index]
            ax.scatter(t.loc[ks, xc].astype(float) * 100, t.loc[ks, "CAGR"].astype(float) * 100,
                       s=70, marker=GROUP_MARK[g], color=GROUP_COLOR[g], edgecolor="white", linewidth=1.5,
                       zorder=3, label=lab)
        for k, mk, col, lab in ((LRS, "*", ORANGE, "LRS 3x (original)"), (ONE, "o", INK, "1x")):
            ax.scatter(float(t.loc[k, xc]) * 100, float(t.loc[k, "CAGR"]) * 100, s=260 if mk == "*" else 90,
                       marker=mk, color=col, edgecolor="white", linewidth=1.5, zorder=4, label=lab)
        ax.set(xlabel=xl, ylabel="Rentabilidad anual compuesta (%)")
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 1.2)          # sitio para la etiqueta L = 3
        ax.grid(alpha=0.25)
    axes[0].set_title(f"S&P 500, {start.year}–{d_end(sp)}: reglas de volatilidad y LRS\ncon un apalancamiento fijo más bajo")
    h, l = axes[0].get_legend_handles_labels()
    legend_band(fig, h, l, where="bottom", ncol=1, fontsize=11)
    comma_fig(fig)
    rep.fig(fig, "frontera", "Rentabilidad frente a volatilidad y a caída máxima de las reglas de volatilidad (a) y (b), "
            "de la LRS original y del 1x, junto a la LRS con un apalancamiento fijo más bajo (L de 0,25 a 3)")


def fig_delta(sp: dict, rep: Report):
    """Δg de cada regla frente a la LRS original, con su IC 90%, desde su primer día."""
    t = sp["tablas"]["completa"]
    G = sp["G"]
    order = [k for g in ("a", "b", "c", "t", "d") for k in G if G[k] == g and k in t.index]
    n = len(order)
    fig, ax = plt.subplots(figsize=(FIG_W, 0.36 * n + 1.8))
    yy = np.arange(n)[::-1]
    for y_, k in zip(yy, order):
        g = G[k]
        m = float(t.loc[k, "Δg_vs_LRS"]) * 100
        lo, hi = float(t.loc[k, "IC90_bajo_vs_LRS"]) * 100, float(t.loc[k, "IC90_alto_vs_LRS"]) * 100
        ax.plot([lo, hi], [y_, y_], color=GROUP_COLOR[g], lw=1.6)
        ax.scatter(m, y_, marker=GROUP_MARK[g], s=40, color=GROUP_COLOR[g], zorder=3, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color=INK, lw=0.8)
    labels = [f"{k}  ({pd.Timestamp(t.loc[k, 'desde']).year}–)" for k in order]
    ax.set_yticks(yy, labels, fontsize=12)       # 12 pt: ~15 px en la columna de 730
    ax.set_ylim(-0.8, n - 0.2)
    ax.set(xlabel="Δg frente a la LRS 3x original\n(pp/año; IC 90%)")
    ax.set_title("S&P 500: Δg de cada regla frente a la LRS original", fontsize=13, loc="right")
    ax.grid(axis="x", alpha=0.25)
    comma_fig(fig)
    rep.fig(fig, "delta_frente_lrs", "Diferencia de crecimiento logarítmico anual de cada regla con la LRS 3x original "
            "en las fechas en que existen las dos, con su intervalo del 90% (bootstrap estacionario emparejado)")


def fig_vix(sp: dict, rep: Report):
    R = sp["R"]
    a = R["VIX < su mediana: 3x/letras"].first_valid_index()
    keys = [(ONE, INK, "-"), (LRS, BLUE, "-"), ("VIX < 20: 3x/letras", ORANGE, "-"), ("LRS × VIX < 20", ORANGE, "--")]
    fig, ax = plt.subplots(figsize=(FIG_W, 5.6))
    for k, col, ls in keys:
        r = R[k].loc[a:].dropna()
        ax.semilogy((1 + r).cumprod() * 10_000, color=col, ls=ls, lw=1.6,
                    label=f"{k}: {pct_es(metrics.cagr(r))}/año")
    ax.set(ylabel="Valor de 10.000 $ (log)",
           title=f"S&P 500 desde {a.year}: el filtro «VIX < 20» frente a la LRS y al 1x")
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    ax.grid(alpha=0.25)
    rep.fig(fig, "vix", "Evolución de 10.000 dólares con las reglas del VIX a 20, la LRS original y el 1x "
            "(3x sintético hasta UPRO, real después)")


# ───────────────────────────── informe ─────────────────────────────

def pp(x: float, k: int = 1) -> str:
    """Fracción anual como puntos con signo: pp(0.0123) → «+1,2»."""
    v = 100 * float(x)
    return ("+" if v > 0 and round(v, k) != 0 else "") + num_es(v, k)


def fp(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    return "< 0,001" if p < 0.001 else num_es(p, 3)


def ci(lo: float, hi: float, k: int = 1) -> str:
    return f"[{pp(lo, k)}; {pp(hi, k)}]"


def show_rules(t: pd.DataFrame, vs3x: bool = False) -> pd.DataFrame:
    """Tabla de reglas para el informe: Δg e IC en puntos por año (con vs3x, también frente al 3x
    comprar y mantener)."""
    cols = ["grupo", "desde", "CAGR", "vol", "sharpe", "maxDD", "exposición_media", "tiempo_en_3x",
            "operaciones_año", "Δg_vs_LRS", "IC90_bajo_vs_LRS", "IC90_alto_vs_LRS", "p_vs_LRS",
            "Δg_vs_1x", "IC90_bajo_vs_1x", "IC90_alto_vs_1x", "p_vs_1x"]
    if vs3x:
        cols += ["Δg_vs_3x", "IC90_bajo_vs_3x", "IC90_alto_vs_3x", "p_vs_3x"]
    out = t[[c for c in cols if c in t]].copy()
    for c in out:
        if c not in ("grupo", "desde"):
            out[c] = pd.to_numeric(out[c])
        if c.startswith(("Δg", "IC90")):
            out[c] = out[c] * 100
    ren = {c: f"{c} (pp)" for c in out if c.startswith(("Δg", "IC90"))}
    return out.rename(columns=ren)


def pct_cols(df: pd.DataFrame) -> set:
    return {c for c in df.columns if c in ("CAGR", "vol", "maxDD", "tiempo_en_3x", "días", "fracción_en_3x",
                                          "acierto", "acierto_siempre_3x") or c.startswith(("CAGR_", "maxDD_", "vol_"))}


def sub_delta_table(sp: dict, tag: str) -> pd.DataFrame:
    """Δg (pp/año) y p de cada regla frente a `tag` (LRS o 1x) en cada ventana."""
    cols = {}
    for w, t in sp["tablas"].items():
        cols[(w, "Δg (pp)")] = t[f"Δg_vs_{tag}"].astype(float) * 100
        cols[(w, "p")] = t[f"p_vs_{tag}"].astype(float)
    df = pd.DataFrame(cols)
    df = df.loc[[k for k in sp["G"] if k in df.index] + [k for k in (LRS, ROT, THREE) if k in df.index and k != (LRS if tag == "LRS" else None)]]
    df.columns = [f"{w} · {c}" for w, c in df.columns]
    return df


def quad_show(q: pd.DataFrame) -> pd.DataFrame:
    out = q[["días", "años", "μ_1x", "σ_1x", "g_1x", "g_3x", "g_letras", "prima_3x_1x", "IC90_bajo", "IC90_alto",
             "prima_3x_letras", kelly.LSTAR_APROX, "L_ex_ante_medio"]].copy()
    for c in ("μ_1x", "σ_1x", "g_1x", "g_3x", "g_letras", "prima_3x_1x", "IC90_bajo", "IC90_alto", "prima_3x_letras"):
        out[c] = out[c].astype(float) * 100
    return out.rename(columns={c: f"{c} (%)" for c in ("μ_1x", "σ_1x", "g_1x", "g_3x", "g_letras")} |
                      {c: f"{c} (pp)" for c in ("prima_3x_1x", "IC90_bajo", "IC90_alto", "prima_3x_letras")} |
                      {"L_ex_ante_medio": "L ex ante medio"})


def spa_show(s: pd.DataFrame) -> pd.DataFrame:
    out = s.copy()
    for c in [c for c in out if c.startswith("Δg")]:
        out[c] = out[c].astype(float) * 100
    return out.rename(columns={c: f"{c} (pp)" for c in out if c.startswith("Δg")})


class Tab:
    """Numeración de las tablas y figuras del informe."""

    def __init__(self, rep: Report):
        self.rep, self.n = rep, 0

    def __call__(self, title: str, df: pd.DataFrame, pct=None, index: bool = True, digits: int = 2):
        self.n += 1
        self.rep.p(f"**Tabla {self.n}. {title}**")
        if isinstance(df.index, pd.MultiIndex):      # columnas en vez de tuplas en el índice
            df, index = df.reset_index(), False
        self.rep.table(df, pct=pct if pct is not None else pct_cols(df), index=index, digits=digits)


def _rng_pp(x: pd.Series, k: int = 1) -> str:
    return f"entre {pp(x.min(), k)} y {pp(x.max(), k)} pp/año"


def _row(t: pd.DataFrame, k: str) -> pd.Series:
    return t.loc[k].astype(object)


def mechanism_text(sp: dict) -> str:
    q = sp["cuadrantes"]
    ah, al, bl, bh = QUAD[(1, 0)], QUAD[(1, 1)], QUAD[(0, 1)], QUAD[(0, 0)]
    reg = sp["regresión"]
    sub = sp["cuadrantes_sub"]
    pos = [k for k, qs in sub.items() if ah in qs.index and qs.loc[ah, "prima_3x_1x"] > 0]
    mods = {MAIN_MODEL: q, **sp["cuadrantes_modelos"]}
    mpos = [m for m, qs in mods.items() if qs.loc[ah, "prima_3x_1x"] > 0]
    qq = sp["quintiles"]
    up = qq.loc["sobre la media", "prima_3x_1x"]
    dn = qq.loc["bajo la media", "prima_3x_1x"]
    r73 = sp["regresión_sub"]["1938–1973"].loc["sobre la media × vol. alta"]
    inter = [reg.loc["sobre la media × vol. alta", "coef"]] + [r.loc["sobre la media × vol. alta", "coef"]
                                                             for r in sp["regresión_sub"].values()]
    n_neg = int(sum(c < 0 for c in inter))
    return (
        f"La tendencia separa la prima del 3x; la volatilidad prevista, no en la dirección que supone un filtro. "
        f"Sobre la media, el 3x creció más que el 1x tanto con volatilidad prevista baja "
        f"({pp(q.loc[al, 'prima_3x_1x'])} pp/año, {pct_es(q.loc[al, 'días'], 0)} de los días) como alta "
        f"({pp(q.loc[ah, 'prima_3x_1x'])} pp/año, {pct_es(q.loc[ah, 'días'], 0)}). Con volatilidad alta el índice "
        f"subió más (μ {pct_es(q.loc[ah, 'μ_1x'])} frente a {pct_es(q.loc[al, 'μ_1x'])}) con más riesgo "
        f"(σ {pct_es(q.loc[ah, 'σ_1x'])} frente a {pct_es(q.loc[al, 'σ_1x'])}); el {kelly.LSTAR_APROX} de esos días "
        f"fue {num_es(q.loc[ah, kelly.LSTAR_APROX], 1)}, por encima de 3, y el de los de volatilidad baja "
        f"{num_es(q.loc[al, kelly.LSTAR_APROX], 1)}. Un Kelly que escala por la varianza prevista (L ex ante) habría "
        f"bajado a {num_es(q.loc[ah, 'L_ex_ante_medio'], 1)} justo donde el 3x rendía más. Bajo la media, el 3x "
        f"perdió frente al 1x con las dos volatilidades ({pp(q.loc[bl, 'prima_3x_1x'])} y "
        f"{pp(q.loc[bh, 'prima_3x_1x'])} pp/año). La prima sobre la media con volatilidad alta es positiva en "
        f"{len(pos)} de los {len(sub)} subperiodos y con {len(mpos)} de los {len(mods)} modelos; por quintiles, "
        f"sobre la media es positiva en los {int((up > 0).sum())} de {len(up)} y bajo la media negativa en "
        f"{int((dn < 0).sum())} de {len(dn)}. Los intervalos son anchos (con volatilidad alta, "
        f"{ci(q.loc[ah, 'IC90_bajo'], q.loc[ah, 'IC90_alto'])}): la diferencia entre volatilidad alta y baja sobre "
        f"la media no es significativa en la muestra completa ({pp(reg.loc['sobre la media × vol. alta', 'coef'])} "
        f"pp/año, p = {fp(reg.loc['sobre la media × vol. alta', 'p'])}); solo lo es en 1938–1973, y a favor de la "
        f"volatilidad alta ({pp(r73['coef'])} pp/año, p = {fp(r73['p'])}). La señal que necesitaría un filtro "
        f"de volatilidad, una prima menor con volatilidad alta dentro de la tendencia, aparece en {n_neg} de las "
        f"{len(inter)} ventanas de la regresión. La "
        f"regresión mide 21 sesiones seguidas a partir del estado de t (que puede cambiar en ese tiempo) y la "
        f"tabla, el día t + 2; por eso sus cifras no coinciden.")


def rules_text(sp: dict) -> str:
    t = sp["tablas"]["completa"]
    G = sp["G"]
    by = {g: [k for k in G if G[k] == g] for g in "abctd"}
    d = t["Δg_vs_LRS"].astype(float)
    best_a = d[by["a"]].idxmax()
    best_b = d[by["b"]].idxmax()
    lrs = t.loc[LRS]
    dd_b = t.loc[by["b"], "maxDD"].astype(float)
    dd_a = t.loc[by["a"], "maxDD"].astype(float)
    ta = t.loc[best_a]
    med_b = [k for k in by["b"] if "(mediana)" in k]
    wins = [(w, k) for w, tt in sp["tablas"].items() for k in G if k in tt.index
            and float(tt.loc[k, "Δg_vs_LRS"]) > 0 and float(tt.loc[k, "p_vs_LRS"]) < 0.05]
    if wins:
        w0, k0 = wins[0]
        x0 = sp["tablas"][w0].loc[k0]
        win_txt = (f"Solo {len(wins)} combinación de regla y ventana supera a la LRS con p < 0,05 (sin corregir): "
                   f"«{k0}» en «{w0}» ({pp(x0['Δg_vs_LRS'])} pp/año, p = {fp(x0['p_vs_LRS'])}). Está en el 3x el "
                   f"{pct_es(x0['tiempo_en_3x'], 0)} del tiempo y frente al 3x comprar y mantener suma "
                   f"{pp(x0['Δg_vs_3x'])} pp/año (p = {fp(x0['p_vs_3x'])}): gana a la LRS por estar apalancado en un "
                   "mercado que subió, no por prever nada (sección (d)).") if len(wins) == 1 else (
                   f"{len(wins)} combinaciones de regla y ventana superan a la LRS con p < 0,05 (sin corregir): "
                   + "; ".join(f"«{k}» en «{w}»" for w, k in wins) + ".")
    else:
        win_txt = "Ninguna combinación de regla y ventana supera a la LRS con p < 0,05, ni siquiera sin corregir."
    return (
        f"(a) Los {len(by['a'])} filtros de régimen tienen un Δg frente a la LRS original {_rng_pp(d[by['a']])}. Los que "
        f"salen a letras con volatilidad alta pierden más que los que bajan al 1x, y los de la mediana más que los del "
        f"percentil 75, que están más tiempo en el 3x. El más cercano, «{best_a}», está en el 3x el "
        f"{pct_es(ta['tiempo_en_3x'], 0)} del tiempo y queda a {pp(ta['Δg_vs_LRS'], 2)} pp/año de la LRS, con una caída "
        f"máxima del {pct_es(ta['maxDD'])} frente al {pct_es(lrs['maxDD'])}; frente al 3x comprar y mantener, "
        f"{pp(ta['Δg_vs_3x'])} pp/año (p = {fp(ta['p_vs_3x'])}). Sus caídas máximas van del {pct_es(dd_a.max())} al "
        f"{pct_es(dd_a.min())}, frente al {pct_es(lrs['maxDD'])} de la LRS: sin la tendencia, la volatilidad prevista "
        f"no evita las caídas largas.\n\n"
        f"(b) Las {len(by['b'])} LRS × volatilidad tienen un Δg frente a la LRS {_rng_pp(d[by['b']])} y recortan su caída "
        f"máxima del {pct_es(lrs['maxDD'])} a entre el {pct_es(dd_b.min())} y el {pct_es(dd_b.max())}. La más cercana, "
        f"«{best_b}», queda a {pp(t.loc[best_b, 'Δg_vs_LRS'], 2)} pp/año. Con la mediana, la regla está en el 1x buena "
        f"parte del tiempo sobre la media (Δg {_rng_pp(t.loc[med_b, 'Δg_vs_LRS'].astype(float))}); con el percentil "
        f"75 se parece más a la LRS y pierde menos.\n\n"
        f"(c) y (d): las {len(by['c'])} reglas del VIX, las {len(by['t'])} de VIX/VIX3M y los {len(by['d'])} "
        f"clasificadores también crecen menos que la LRS en sus fechas (Δg {_rng_pp(d[by['c'] + by['t'] + by['d']])}); se "
        f"detallan abajo. Por ventanas (tablas siguientes), algunas reglas quedan por encima de la LRS en alguna, sin "
        f"que se repitan las mismas en todas. {win_txt}")


def equal_risk_text(sp: dict) -> str:
    eq = sp["igual_riesgo"].set_index("regla")
    e16 = sp["igual_riesgo_2016"].set_index("regla")
    k = "LRS × vol GJR (mediana)"
    r, r16 = eq.loc[k], e16.loc[k]
    nneg = int((eq["Δg_vs_L_vol"] < 0).sum())
    sig_neg = int((eq["IC90_alto_vs_L_vol"] < 0).sum())
    sig_pos = int((eq["IC90_bajo_vs_L_vol"] > 0).sum())
    return (
        f"Bajar al 1x cuando la volatilidad prevista es alta reduce el riesgo de la LRS, y bajar el apalancamiento "
        f"fijo lo reduce lo mismo. A igual volatilidad, {nneg} de las {len(eq)} reglas (b) crecen menos que la LRS "
        f"con apalancamiento fijo y el resto casi lo mismo (Δg {_rng_pp(eq['Δg_vs_L_vol'])}); "
        f"{'ningún IC 90% excluye el cero' if sig_neg + sig_pos == 0 else f'el IC 90% queda entero por debajo de cero en {sig_neg} y por encima en {sig_pos}'}. «{k}» rinde {pct_es(r['CAGR_regla'])} con caída máxima "
        f"{pct_es(r['maxDD_regla'])}; la LRS a L = {num_es(r['L_vol'], 2)} (su misma volatilidad) rinde "
        f"{pct_es(r['CAGR_L_vol'])} con {pct_es(r['maxDD_L_vol'])}, y la de su misma exposición media "
        f"(L = {num_es(r['L_exp'], 2)}), {pct_es(r['CAGR_L_exp'])} con {pct_es(r['maxDD_L_exp'])}. En 2016–2026 la "
        f"regla cayó menos ({pct_es(r16['maxDD_regla'])} frente a {pct_es(r16['maxDD_L_vol'])} de la LRS a "
        f"L = {num_es(r16['L_vol'], 2)}), pero creció menos ({pct_es(r16['CAGR_regla'])} frente a "
        f"{pct_es(r16['CAGR_L_vol'])}). En la figura, las reglas (b) quedan pegadas a la línea de la LRS con "
        f"apalancamiento fijo o por debajo de ella: no dan una combinación de rentabilidad y riesgo que no se consiga "
        f"eligiendo un L más bajo.")


def vix_text(sp: dict) -> str:
    R, t = sp["R"], sp["tablas"]["completa"]
    k = "VIX < 20: 3x/letras"
    a = R[k].first_valid_index()
    one = metrics.cagr(R[ONE].loc[a:].dropna())
    lrs = metrics.cagr(R[LRS].loc[a:].dropna())
    tr = sp["tablas"]["ETFs reales (2010–2026)"]
    kt = "VIX/VIX3M < 1: 3x/1x"
    lt = "LRS × VIX/VIX3M < 1"
    return (
        f"Desde {a.date()}, «{k}» rinde {pct_es(t.loc[k, 'CAGR'])} anual frente al {pct_es(one)} del 1x y el "
        f"{pct_es(lrs)} de la LRS, con una caída máxima del {pct_es(t.loc[k, 'maxDD'])}. El VIX sube después de las "
        f"caídas y tarda en bajar: la regla vende tarde y vuelve tarde. En el 3x estuvo el "
        f"{pct_es(sp['vix_on']['2009'], 0)} de las sesiones entre el 10 de marzo y diciembre de 2009 y el "
        f"{pct_es(sp['vix_on']['2020'], 0)} entre el 24 de marzo y diciembre de 2020, los dos rebotes más fuertes del "
        f"periodo. Combinado con la LRS («LRS × VIX < 20») tiene un Δg frente a ella de "
        f"{pp(t.loc['LRS × VIX < 20', 'Δg_vs_LRS'])} pp/año (IC "
        f"{ci(t.loc['LRS × VIX < 20', 'IC90_bajo_vs_LRS'], t.loc['LRS × VIX < 20', 'IC90_alto_vs_LRS'])}). Con la mediana expansiva del VIX en vez de 20 los resultados son parecidos.\n\n"
        f"La curva VIX/VIX3M estuvo en contango (VIX/VIX3M < 1) el {pct_es(sp['vix_on']['contango'], 0)} de los días "
        f"desde 2006, así que «{kt}» es casi un 3x comprar y mantener que sale en los picos de estrés. Con UPRO real "
        f"(2010–2026) rinde {pct_es(tr.loc[kt, 'CAGR'])} frente al {pct_es(tr.loc[LRS, 'CAGR'])} de la LRS "
        f"({pp(tr.loc[kt, 'Δg_vs_LRS'])} pp/año, p = {fp(tr.loc[kt, 'p_vs_LRS'])}) y al "
        f"{pct_es(tr.loc[THREE, 'CAGR'])} del 3x comprar y mantener ({pp(tr.loc[kt, 'Δg_vs_3x'])} pp/año, p = "
        f"{fp(tr.loc[kt, 'p_vs_3x'])}), con una caída máxima del {pct_es(tr.loc[kt, 'maxDD'])} frente al "
        f"{pct_es(tr.loc[LRS, 'maxDD'])} de la LRS. Combinada con la LRS («{lt}») tiene una caída máxima del "
        f"{pct_es(tr.loc[lt, 'maxDD'])} (la LRS, {pct_es(tr.loc[LRS, 'maxDD'])}) y un Δg frente a ella de "
        f"{pp(tr.loc[lt, 'Δg_vs_LRS'])} pp/año. Son 16 años de un mercado sobre todo alcista: poca muestra para "
        f"distinguir estas reglas del 3x.")


def clf_text(sp: dict) -> str:
    a = sp["auc"]
    full = a[a.ventana == "completa"].set_index("clasificador")
    tr = sp["tablas"]["ETFs reales (2010–2026)"]
    k = "Clasificador logístico"
    ar = a[(a.ventana == "ETFs reales (2010–2026)")].set_index("clasificador")
    a00 = a[(a.ventana == "2000–2026")].set_index("clasificador")
    return (
        f"Los clasificadores no separan los meses buenos del 3x de los malos. El AUC fuera de muestra va de "
        f"{num_es(full.AUC.min(), 2)} a {num_es(full.AUC.max(), 2)} en cada modelo desde su primer día, y su acierto "
        f"queda por debajo del de «siempre 3x» (la tasa base). Desde 2000, el AUC del logístico "
        f"({num_es(a00.loc[k, 'AUC'], 2)}) es el de puntuar cada día solo con la distancia a la media, sin ajustar nada "
        f"({num_es(a00.loc[k, 'AUC distancia a la media'], 2)}): lo poco que ordena viene de la tendencia, que ya usa "
        f"la LRS. Añadir el VIX no mejora el AUC. El logístico acaba en el 3x el {pct_es(full.loc[k, 'fracción_en_3x'], 0)} "
        f"del tiempo (el {pct_es(ar.loc[k, 'fracción_en_3x'], 0)} desde 2010): reaprende la tasa base, que es "
        f"favorable al 3x. Por eso con UPRO real rinde {pct_es(tr.loc[k, 'CAGR'])} anual, "
        f"{pp(tr.loc[k, 'Δg_vs_LRS'])} pp/año sobre la LRS, pero solo {pp(tr.loc[k, 'Δg_vs_3x'])} pp/año sobre el 3x "
        f"comprar y mantener (IC {ci(tr.loc[k, 'IC90_bajo_vs_3x'], tr.loc[k, 'IC90_alto_vs_3x'])}, p = "
        f"{fp(tr.loc[k, 'p_vs_3x'])}): es el apalancamiento en un mercado alcista, no una previsión.")


def spa_text(sp: dict) -> str:
    s = sp["spa"]
    get = lambda w, b, fam="reglas nuevas": s[s.ventana.str.startswith(w) & (s["frente a"] == b) & (s.familia == fam)].iloc[0]  # noqa: E731
    L, V, E_, P = (get(w, LRS) for w in ("historia larga", "era del VIX", "ETFs reales", "2016"))
    L1, E1, E3 = get("historia larga", ONE), get("ETFs reales", ONE), get("ETFs reales", THREE)
    g = sp["spa_grupos"].set_index(["grupo", "frente a"])
    gb = g.loc[(GROUPS["b"], ONE)]
    return (
        f"Frente a la LRS original ninguna familia pasa: p = {fp(L.p_SPA)} en la historia larga ({L.reglas} reglas), "
        f"{fp(V.p_SPA)} en la era del VIX ({V.reglas}), {fp(E_.p_SPA)} con ETFs reales ({E_.reglas}) y {fp(P.p_SPA)} "
        f"en 2016–2026. Frente al 1x, la historia larga da p = {fp(L1.p_SPA)} (mejor: «{L1.mejor}», "
        f"{pp(L1.Δg_mejor)} pp/año). Con ETFs reales el SPA frente al 1x da p = {fp(E1.p_SPA)}, con «{E1.mejor}» como "
        f"mejor ({pp(E1.Δg_mejor)} pp/año): es la regla que está casi siempre en el 3x, y frente al 3x comprar y "
        f"mantener la familia da p = {fp(E3.p_SPA)}. En 2010–2026 cualquier regla muy invertida en el 3x supera al "
        f"1x; no es timing. Por grupos, las LRS × volatilidad frente al 1x dan p = {fp(gb.p_SPA)}: lo que tienen de "
        f"bueno frente al 1x es la LRS que llevan dentro (frente a ella, p = "
        f"{fp(g.loc[(GROUPS['b'], LRS)].p_SPA)}). Holm sobre los p individuales de la ventana completa: "
        f"{sp['holm']['LRS']} rechazos frente a la LRS y {sp['holm']['1x']} frente al 1x.")


def other_text(res: dict) -> str:
    agg, Sg, Ag = res["ndx"]
    full = agg.loc["completa"]
    G = res["sp"]["G"]
    new = list(G)
    sg = Sg.reset_index()
    gl = lambda w, b, fam="reglas nuevas": sg[sg.ventana.str.startswith(w) & (sg["frente a"] == b) & (sg.familia == fam)].iloc[0]  # noqa: E731
    nl, n1, nr1, nr3 = gl("historia larga", LRS), gl("historia larga", ONE), gl("ETFs reales", ONE), gl("ETFs reales", THREE)
    b_names = [k for k in new if G[k] == "b"]
    c_names = [k for k in new if G[k] == "c"]
    above = int((full.loc[new, "Δg_vs_LRS"] > 0).sum())
    se = res["semis"]
    ss = se["spa"]
    sl = ss[ss.ventana.str.startswith("historia larga") & (ss["frente a"] == LRS)].iloc[0]
    s1 = ss[ss.ventana.str.startswith("historia larga") & (ss["frente a"] == ONE) & (ss.familia == "reglas nuevas")].iloc[0]
    ts_ = se["tablas"]["completa"]
    ag = Ag.reset_index()
    return (
        f"Nasdaq-100 (desde 1985-10, mediana de {N_NDX} reconstrucciones del calentamiento): {above} de las "
        f"{len(new)} reglas quedan por encima de la LRS original en la mediana de las reconstrucciones, sobre todo filtros de régimen del "
        f"percentil 75 (en el 3x la mayor parte del tiempo), pero el "
        f"SPA frente a la LRS da una mediana de p = {fp(nl.p_SPA_mediana)} (de {fp(nl.p_SPA_mín)} a "
        f"{fp(nl.p_SPA_máx)}) y frente al 1x de {fp(n1.p_SPA_mediana)}. Las LRS × volatilidad tienen un Δg frente a la "
        f"LRS {_rng_pp(full.loc[b_names, 'Δg_vs_LRS'])} y las del VIX {_rng_pp(full.loc[c_names, 'Δg_vs_LRS'])}. "
        f"Con TQQQ real (2010–2026) la familia supera al 1x (p = {fp(nr1.p_SPA_mediana)}), pero no al 3x comprar y "
        f"mantener (p = {fp(nr3.p_SPA_mediana)}): en esa ventana basta con estar en el 3x. Los AUC de los clasificadores van de {num_es(ag.AUC_mediana.min(), 2)} a "
        f"{num_es(ag.AUC_mediana.max(), 2)} (medianas por modelo y ventana, sin corregir por las {len(ag)} "
        f"combinaciones).\n\n"
        f"Semiconductores (desde {ts_.loc[new[0], 'desde']}): ninguna regla se acerca a una mejora descontada la búsqueda "
        f"(SPA frente a la LRS p = {fp(sl.p_SPA)}; frente al 1x p = {fp(s1.p_SPA)}). La mejor frente a la LRS, "
        f"«{sl.mejor}», suma {pp(sl.Δg_mejor)} pp/año.")


def reading_text(res: dict) -> str:
    sp = res["sp"]
    race = sp["carrera"]
    t = sp["tablas"]["completa"]
    new = list(sp["G"])
    q = sp["cuadrantes"]
    ah, al = QUAD[(1, 0)], QUAD[(1, 1)]
    s_all = [sp["spa"], res["semis"]["spa"]]
    p_min = min(float(x[x["frente a"] == LRS].p_SPA.min()) for x in s_all)
    p_min = min(p_min, float(res["ndx"][1].xs(LRS, level="frente a").p_SPA_mín.min()))
    return (
        f"La pregunta del artículo 3 era si la volatilidad, que se prevé bien, sirve para decidir cuándo estar en el "
        f"3x. La primera mitad se confirma: los cinco modelos prevén la varianza del mes siguiente con un R² en "
        f"logaritmos de {num_es(race.R2_log.min(), 2)} a {num_es(race.R2_log.max(), 2)} en el S&P 500, y los mejores "
        f"(GJR, EGARCH, HAR en logaritmos) son casi indistinguibles. La segunda, no: de {len(new)} reglas con esas "
        f"previsiones, el VIX o un clasificador, {int((t.loc[new, 'Δg_vs_LRS'].astype(float) < 0).sum())} crecen menos "
        f"que la LRS de Gayed y Bilello en el S&P 500, y el SPA frente a la LRS no baja de p = {fp(p_min)} en ninguna "
        f"ventana ni mercado (S&P 500, Nasdaq-100 con sus {N_NDX} reconstrucciones y semis).\n\n"
        f"El motivo está en los cuadrantes. El 3x pierde con la volatilidad, pero la volatilidad alta dentro de una "
        f"tendencia alcista vino con rentabilidades más altas: el apalancamiento óptimo de esos días, medido después, "
        f"siguió por encima de 3. Un filtro que baja al 1x o sale a letras con volatilidad alta renuncia justo a esos "
        f"días. Sobre la media, la volatilidad alta sí bajó algo el L* medido después (de "
        f"{num_es(q.loc[al, kelly.LSTAR_APROX], 1)} a {num_es(q.loc[ah, kelly.LSTAR_APROX], 1)}), pero sin acercarlo a "
        f"3. Y para reducir el riesgo no hace falta prever nada: bajar el apalancamiento fijo de la LRS consigue la "
        f"misma reducción con una rentabilidad parecida o mayor (sección 4, (b)). Y los filtros del VIX, populares en foros, restan: el VIX "
        f"alto llega después de la caída y se va después del rebote.\n\n"
        f"Para el artículo: la volatilidad prevista es una buena medida del riesgo, no una señal de cuándo entrar. "
        f"Quien quiera menos caída que la LRS 3x tiene una forma más sencilla y al menos igual de buena: menos "
        f"apalancamiento fijo (la línea de la figura de la frontera).")


def caveats_text(res: dict) -> str:
    return "\n".join([
        "- Umbrales fijados antes de esta ejecución (mediana y percentil 75 expansivos de la propia previsión, VIX 20 y "
        "su mediana expansiva, VIX/VIX3M = 1, probabilidad 0,5), pero escogidos en una exploración previa del "
        "laboratorio (sobre una base anterior, con otras cifras) que ya había visto resultados de reglas parecidas; el "
        "horizonte de 21 sesiones, λ = 0,94, las frecuencias de reestimación y los diez años mínimos vienen de R12, "
        "estimados con la misma muestra. El SPA cuenta las reglas de este informe, no las de esa exploración.",
        "- La mediana expansiva no es estacionaria: la fracción de días con «volatilidad baja» cambia de una época a "
        "otra (desde 2016, bajo la media no hubo ningún día con volatilidad prevista baja). Es lo que habría visto "
        "un inversor en tiempo real, pero hace que las celdas de los subperiodos tengan tamaños muy distintos.",
        "- El EGARCH es gaussiano y se estima con código propio (Nelder-Mead); su previsión a 21 sesiones es exacta "
        "bajo normalidad, pero los residuos reales tienen colas más gruesas. Los umbrales de las reglas son rangos de "
        "la propia previsión, así que un sesgo de nivel no las cambia.",
        "- El clasificador entrena sus diez primeros años con la varianza GJR filtrada con los parámetros de la "
        "primera estimación (conocidos en la primera fecha en que predice): todo lo que usa en T se conoce en T, pero "
        "los rasgos de entrenamiento anteriores a 1938 están dentro de muestra para esos parámetros.",
        "- El VIX es la volatilidad implícita del S&P 500; se usa también en el Nasdaq-100 y en semis (la base no "
        "tiene el VXN). El VIX de 1990–2003 es el que CBOE recalculó hacia atrás con la metodología de 2003. Las "
        "reglas del VIX empiezan en 1991 y las de VIX/VIX3M en 2006: sus CAGR no se comparan con las de 1939; los Δg, "
        "sí, porque se miden en las fechas comunes.",
        "- La regla «VIX < 20» es un umbral redondo muy repetido en foros de inversión; el laboratorio no ha podido "
        "verificar una fuente original, así que se trata como regla popular sin fuente.",
        "- Las referencias de igual exposición y de igual volatilidad fijan L con toda la ventana (cotas, no reglas). "
        "Cada cambio 3x ↔ 1x cuesta 10 pb (dos unidades de rotación); el 1x es el índice con dividendos sin comisión "
        "de ETF, como en todo el laboratorio.",
        "- Antes de 1990 los precios del S&P 500 tienen autocorrelación por precios desfasados (R15, R19): con la "
        "orden al cierre siguiente no se aprovecha, pero afecta a las reglas y a la LRS por igual.",
        "- Los p por ventana, mercado y comparación no se corrigen entre sí; la corrección por la búsqueda es el SPA de "
        "cada familia. El Nasdaq-100 no es un mercado virgen (R6 ya exploró TQQQ).",
    ])


SOURCES = "\n".join([
    "- Gayed, M. y Bilello, C. (2016). «Leverage for the Long Run». SSRN. La LRS original.",
    "- Bollerslev, T. (1986). «Generalized autoregressive conditional heteroskedasticity». Journal of Econometrics, 31.",
    "- Glosten, L., Jagannathan, R. y Runkle, D. (1993). «On the relation between the expected value and the "
    "volatility of the nominal excess return on stocks». Journal of Finance, 48.",
    "- Nelson, D. (1991). «Conditional heteroskedasticity in asset returns: a new approach». Econometrica, 59.",
    "- Corsi, F. (2009). «A simple approximate long-memory model of realized volatility». Journal of Financial "
    "Econometrics, 7.",
    "- Duan, N. (1983). «Smearing estimate: a nonparametric retransformation method». JASA, 78.",
    "- J. P. Morgan/Reuters (1996). RiskMetrics — Technical Document (EWMA con λ = 0,94).",
    "- Patton, A. (2011). «Volatility forecast comparison using imperfect volatility proxies». Journal of "
    "Econometrics, 160 (QLIKE).",
    "- Hansen, P., Lunde, A. y Nason, J. (2011). «The model confidence set». Econometrica, 79.",
    "- Hansen, P. (2005). «A test for superior predictive ability». Journal of Business & Economic Statistics, 23.",
    "- Politis, D. y Romano, J. (1994). «The stationary bootstrap». JASA, 89.",
    "- Hansen, L. y Hodrick, R. (1980). «Forward exchange rates as optimal predictors of future spot rates». "
    "Journal of Political Economy, 88 (errores para objetivos solapados).",
    "- Moreira, A. y Muir, T. (2017). «Volatility-managed portfolios». Journal of Finance, 72 (la cartera "
    "gestionada por volatilidad está en R17).",
    "- Appel, G. (2005). Technical Analysis: Power Tools for Active Investors (el MACD 12-26-9).",
    "- CBOE, «VIX White Paper» (metodología del VIX desde 2003) y el índice VIX3M (antes VXV). Cierres diarios de "
    "^VIX y ^VIX3M de la caché del laboratorio.",
    "- «VIX < 20» y «VIX/VIX3M < 1»: reglas populares sin fuente original verificada.",
])


def write_report(res: dict) -> Report:
    rep = Report("r23_vol", "R23 — ¿Sirve prever la volatilidad para cronometrar el 3x?")
    for f in rep.dir.glob("table_*.csv"):       # Report numera las tablas por línea: sin huérfanas
        f.unlink()
    tab = Tab(rep)
    sp = res["sp"]
    G = sp["G"]
    t = sp["tablas"]["completa"]
    start = sp["ventanas"]["completa"][0]
    end = sp["R"][LRS].dropna().index[-1]
    new = list(G)
    n_new = len(new)
    cnt = pd.Series(G).value_counts()
    long = long_names(G)

    # ── resumen
    rep.p(__doc__.split("\n", 2)[2])
    rep.h("Resumen")
    worse = int((t.loc[new, "Δg_vs_LRS"].astype(float) < 0).sum())
    best = t.loc[new, "Δg_vs_LRS"].astype(float).idxmax()
    spa_t = sp["spa"]
    s_long = spa_t[spa_t.ventana.str.startswith("historia larga")]
    p_long_lrs = float(s_long[(s_long["frente a"] == LRS)].p_SPA.iloc[0])
    p_long_1x = float(s_long[(s_long["frente a"] == ONE) & (s_long.familia == "reglas nuevas")].p_SPA.iloc[0])
    best_1x = s_long[(s_long["frente a"] == ONE) & (s_long.familia == "reglas nuevas")].iloc[0]
    q = sp["cuadrantes"]
    reg = sp["regresión"]
    ah, al = QUAD[(1, 0)], QUAD[(1, 1)]
    er = sp["igual_riesgo"].set_index("regla")
    n_sig_er = int(((er["IC90_bajo_vs_L_vol"] > 0) | (er["IC90_alto_vs_L_vol"] < 0)).sum())
    a0 = sp["auc"]
    a_full = a0[(a0.ventana == "completa") & ~a0.clasificador.str.contains("VIX")]
    rep.p("\n".join([
        f"- **Prever la volatilidad funciona; cronometrar el 3x con ella, no.** Se probaron {n_new} reglas "
        f"nuevas en el S&P 500 ({cnt.get('a', 0)} filtros de régimen, {cnt.get('b', 0)} LRS × volatilidad, "
        f"{cnt.get('c', 0) + cnt.get('t', 0)} con el VIX y {cnt.get('d', 0)} clasificadores). "
        f"{'Las ' + str(n_new) if worse == n_new else str(worse) + ' de las ' + str(n_new)} crecen menos que la LRS 3x original en sus propias fechas; la que más se acerca, «{best}», "
        f"queda en {pp(t.loc[best, 'Δg_vs_LRS'], 2)} pp/año (IC 90% {ci(t.loc[best, 'IC90_bajo_vs_LRS'], t.loc[best, 'IC90_alto_vs_LRS'])}). "
        f"Descontada la búsqueda (SPA de Hansen, {len(long)} reglas de historia larga, {start.date()} → {end.date()}), "
        f"p = {fp(p_long_lrs)} frente a la LRS original.",
        f"- **Frente al 1x tampoco sobrevive nada a la búsqueda.** La mejor de historia larga frente al 1x es "
        f"«{best_1x.mejor}» ({pp(best_1x.Δg_mejor)} pp/año); el SPA de la familia da p = {fp(p_long_1x)}. Esa regla "
        f"está en el 3x el {pct_es(t.loc[best_1x.mejor, 'tiempo_en_3x'], 0)} del tiempo y frente al 3x comprar y "
        f"mantener suma {pp(t.loc[best_1x.mejor, 'Δg_vs_3x'])} pp/año (p = {fp(t.loc[best_1x.mejor, 'p_vs_3x'])}).",
        f"- **Mecanismo.** Sobre la media de 200 sesiones, el 3x creció más que el 1x con volatilidad prevista "
        f"baja ({pp(q.loc[al, 'prima_3x_1x'])} pp/año, IC {ci(q.loc[al, 'IC90_bajo'], q.loc[al, 'IC90_alto'])}) "
        f"y también con volatilidad alta ({pp(q.loc[ah, 'prima_3x_1x'])} pp/año, IC "
        f"{ci(q.loc[ah, 'IC90_bajo'], q.loc[ah, 'IC90_alto'])}). Con volatilidad alta la media del índice fue "
        f"mayor y el {kelly.LSTAR_APROX} de esos días, {num_es(q.loc[ah, kelly.LSTAR_APROX], 1)}, siguió por "
        f"encima de 3, mientras que un Kelly que escala por la varianza prevista habría bajado a "
        f"{num_es(q.loc[ah, 'L_ex_ante_medio'], 1)} de media. La diferencia entre volatilidad alta y baja "
        f"sobre la media no es significativa (regresión con errores HAC solapados: "
        f"{pp(reg.loc['sobre la media × vol. alta', 'coef'])} pp/año, p = "
        f"{fp(reg.loc['sobre la media × vol. alta', 'p'])}). La información está en la tendencia.",
        f"- **Bajar a 1x con volatilidad alta es otra forma de bajar el apalancamiento.** A igual volatilidad, la "
        f"LRS con un apalancamiento fijo crece lo mismo o más (Δg de las {len(er)} reglas {_rng_pp(er['Δg_vs_L_vol'])}, "
        f"{'ninguno con el IC 90% fuera del cero' if n_sig_er == 0 else str(n_sig_er) + ' con el IC 90% fuera del cero'}). "
        f"«LRS × vol GJR (mediana)» rinde {pct_es(er.loc['LRS × vol GJR (mediana)', 'CAGR_regla'])} con caída máxima "
        f"{pct_es(er.loc['LRS × vol GJR (mediana)', 'maxDD_regla'])}; la LRS con un apalancamiento fijo de "
        f"su misma volatilidad (L = {num_es(er.loc['LRS × vol GJR (mediana)', 'L_vol'], 2)}) rinde "
        f"{pct_es(er.loc['LRS × vol GJR (mediana)', 'CAGR_L_vol'])} con "
        f"{pct_es(er.loc['LRS × vol GJR (mediana)', 'maxDD_L_vol'])}.",
        f"- **Los filtros del VIX restan.** «VIX < 20: 3x/letras» rinde "
        f"{pct_es(sp['tablas']['completa'].loc['VIX < 20: 3x/letras', 'CAGR'])} anual desde "
        f"{t.loc['VIX < 20: 3x/letras', 'desde']} ({pp(t.loc['VIX < 20: 3x/letras', 'Δg_vs_1x'])} pp/año de "
        f"crecimiento frente al 1x): tras un pico del VIX se queda fuera de los rebotes (en el 3x el "
        f"{pct_es(sp['vix_on']['2009'], 0)} de las sesiones entre el 10 de marzo y diciembre de 2009 y el "
        f"{pct_es(sp['vix_on']['2020'], 0)} entre el 24 de marzo y diciembre de 2020).",
        f"- **El clasificador no predice.** AUC fuera de muestra de "
        f"{num_es(a_full.AUC.min(), 3)}–{num_es(a_full.AUC.max(), 3)} en el S&P 500 desde "
        f"{a_full.desde.min()}, con un acierto por debajo de «siempre 3x».",
    ]))

    # ── 1. comprobación
    rep.h("1. Comprobación inicial")
    sn = res["sanity"]
    tab("R13 reproducido con la base actual: S&P 500 desde 1938-02-04, orden al cierre siguiente", sn[["CAGR", "vol", "maxDD"]])
    rep.p(f"Coincide con R13 (reports/r13_lrs): LRS 3x (original) {pct_es(sn.loc[LRS, 'CAGR'])} con caída máxima "
          f"{pct_es(sn.loc[LRS, 'maxDD'])}; 1x {pct_es(sn.loc[ONE, 'CAGR'])}; 3x comprar y mantener "
          f"{pct_es(sn.loc[THREE, 'CAGR'])}. Las referencias de este módulo (1x, 3x, LRS original y rotación "
          f"3x/1x) son las series de r13.strategies ({'idénticas' if sp['sanity_diff'] == 0 else 'diferencia máxima ' + num_es(sp['sanity_diff'] * 1e12, 2) + '·10⁻¹²'}). Todas las reglas "
          f"nuevas pasan por r13.run_weights con el mismo protocolo.")

    # ── 2. previsiones
    rep.h("2. ¿Se prevé la volatilidad?")
    race = sp["carrera"]
    tab(f"S&P 500: previsiones de la varianza de las 21 sesiones siguientes, fuera de muestra "
        f"({race.desde.iloc[0]} → {race.hasta.iloc[0]})", race.drop(columns=["desde", "hasta"]), pct=set(), digits=3)
    race2 = sp["carrera_2000"]
    tab(f"S&P 500 desde {race2.desde.iloc[0]}", race2.drop(columns=["desde", "hasta"]), pct=set(), digits=3)
    n0 = res["ndx0"]["carrera"]
    tab(f"Nasdaq-100 (reconstrucción 0; datos reales desde 1985-10), {n0.desde.iloc[0]} → {n0.hasta.iloc[0]}",
        n0.drop(columns=["desde", "hasta"]), pct=set(), digits=3)
    s0 = res["semis"]["carrera"]
    tab(f"Semiconductores, {s0.desde.iloc[0]} → {s0.hasta.iloc[0]}", s0.drop(columns=["desde", "hasta"]),
        pct=set(), digits=3)
    b1, b2 = race.index[0], race2.index[0]
    in_mcs = lambda r: ", ".join(r.index[r.en_MCS.astype(bool)])  # noqa: E731
    rep.p(f"pérdida_media: QLIKE (Patton 2011), robusta al ruido de la varianza realizada; p_MCS y en_MCS: Model "
          f"Confidence Set al 10% (Hansen, Lunde y Nason 2011) con bloque de {BLOCK} sesiones y {B_MCS} "
          f"remuestreos; R2_log: R² de la previsión en logaritmos; sesgo: media de la previsión entre media de "
          f"la realizada. En el S&P 500 completo la mejor es {b1} (en el MCS: {in_mcs(race)}); desde 2000, "
          f"{b2} (en el MCS: {in_mcs(race2)}); en el Nasdaq-100, {n0.index[0]}; en semis, {s0.index[0]}. Ningún "
          "modelo gana en todos los mercados y periodos. Los cinco se parecen más entre sí que a la realizada: "
          "todos siguen la misma agrupación de la volatilidad. El HAR va en logaritmos con la corrección de Duan; "
          "el de R12 (en niveles) era el peor de la carrera. El EGARCH se estimó con código propio y su previsión a "
          "21 sesiones es la media exacta con z normal (con la senda determinista del logaritmo quedaba sesgada a "
          "la baja).")
    eg = sp["egarch"]
    tab("EGARCH(1,1) del S&P 500: parámetros de cada reestimación (una de cada cuatro)",
        eg.iloc[::4].set_index("fecha")[["omega", "alpha", "gamma", "beta", "convergió"]], pct=set(), digits=4)
    rep.p("log h_{t+1} = ω + β·log h_t + α(|z_t| − √(2/π)) + γ·z_t, con z_t la rentabilidad del día dividida por "
          "su volatilidad condicional. γ < 0: las caídas suben la volatilidad más que las subidas (el efecto "
          "apalancamiento); β, la persistencia.")

    # ── 3. mecanismo
    rep.h("3. Mecanismo: la prima del 3x según la tendencia y la volatilidad prevista")
    rep.p(f"Cada sesión se clasifica al cierre de t por la tendencia (sobre o bajo la media de 200 sesiones del índice "
          f"de rentabilidad total, la señal de la LRS) y por la volatilidad GJR prevista (bajo o sobre su mediana "
          f"expansiva fuera de muestra), y se mide la rentabilidad del día t + 2, la que cobra una orden al cierre "
          f"de t+1. Es descriptivo, no una regla. prima_3x_1x: crecimiento logarítmico anual del 3x menos el del 1x "
          f"en esos días, con su IC 90% (bootstrap estacionario de la ventana, bloque {BLOCK}); {kelly.LSTAR_APROX}: "
          f"{kelly.LSTAR_DESC[kelly.LSTAR_APROX]}, con los días de cada celda (ex post); L ex ante medio: media en la "
          f"celda de (prima {kelly.PRIMA} − s)/varianza GJR prevista, el L de Kelly que habría fijado en tiempo real "
          f"quien escala por la volatilidad ({kelly.PRIMA}: {kelly.PRIMA_DESC}).")
    tab(f"S&P 500, {start.date()} → {end.date()}: cuadrantes tendencia × volatilidad GJR prevista", quad_show(q), pct={"días"})
    rows = []
    for w, qs in sp["cuadrantes_sub"].items():
        for lab in list(QUAD.values()):
            if lab in qs.index:
                r = qs.loc[lab]
                rows.append({"periodo": w, "cuadrante": lab, "días": r["días"],
                             "prima_3x_1x (pp)": float(r["prima_3x_1x"]) * 100,
                             "IC90_bajo (pp)": float(r["IC90_bajo"]) * 100, "IC90_alto (pp)": float(r["IC90_alto"]) * 100,
                             kelly.LSTAR_APROX: r[kelly.LSTAR_APROX], "L ex ante medio": r["L_ex_ante_medio"]})
    sub = pd.DataFrame(rows).set_index(["periodo", "cuadrante"])
    tab("Los cuadrantes por subperiodo (celdas con menos de 50 días: cifras poco fiables)", sub, pct={"días"})
    rows = []
    for m, qs in {MAIN_MODEL: q, **sp["cuadrantes_modelos"]}.items():
        for lab in QUAD.values():
            rows.append({"modelo": m, "cuadrante": lab, "días": qs.loc[lab, "días"],
                         "prima_3x_1x (pp)": float(qs.loc[lab, "prima_3x_1x"]) * 100,
                         "IC90_bajo (pp)": float(qs.loc[lab, "IC90_bajo"]) * 100,
                         "IC90_alto (pp)": float(qs.loc[lab, "IC90_alto"]) * 100})
    tab("La misma descomposición con cada modelo de volatilidad (S&P 500, ventana completa)",
        pd.DataFrame(rows).set_index(["modelo", "cuadrante"]), pct={"días"})
    tab("Por quintil de la volatilidad GJR prevista (rango frente a su historia fuera de muestra)",
        quad_show(sp["quintiles"].assign(IC90_bajo=np.nan, IC90_alto=np.nan)).drop(
            columns=["IC90_bajo (pp)", "IC90_alto (pp)"]), pct={"días"})
    fig_mechanism(sp, rep)
    rt = [reg.assign(periodo="completa")] + [r.assign(periodo=k) for k, r in sp["regresión_sub"].items()]
    rt = pd.concat(rt).reset_index().rename(columns={"index": "indicador"}).set_index(["periodo", "indicador"])
    rt = rt.rename(columns={"coef": "coef (pp/año)", "se": "ee (pp/año)"})
    rt[["coef (pp/año)", "ee (pp/año)"]] *= 100
    tab("Regresión de la prima del 3x sobre el 1x de las 21 sesiones siguientes a la orden sobre el estado "
        "(sin constante; errores HAC para objetivos solapados)", rt, pct=set(), digits=3)
    rep.p("«sobre la media» es la prima anual media con volatilidad baja y «sobre la media × vol. alta», cuánto cambia "
          "con volatilidad alta (igual bajo la media). Objetivo: Σ_{t+2}^{t+22} [log(1 + x3) − log(1 + u)]; errores de "
          "Hansen-Hodrick con 20 retardos (stats.hac_overlap; si su varianza no sale positiva, Newey-West con 42, "
          "indicado en la columna estimador). p bilateral.")
    rep.p(mechanism_text(sp))

    # ── 4. reglas
    rep.h(f"4. Las reglas en el S&P 500 ({start.year}–{end.year})")
    tab(f"S&P 500: cada regla desde su primer día (las de volatilidad, desde {start.date()}) hasta {end.date()}",
        show_rules(t.loc[new + [LRS, ROT, THREE, ONE]]), digits=3)
    rep.p("exposición_media: 3·(peso en el 3x) + (peso en el 1x) decididos en t; tiempo_en_3x: fracción de sesiones "
          "con todo en el 3x; operaciones_año: cambios de la exposición por año. Δg: crecimiento logarítmico anual de "
          "la regla menos el de la referencia en las fechas en que existen las dos; IC 90% percentil y p unilateral "
          f"(H0: Δg ≤ 0) con el error estándar de un bootstrap estacionario emparejado (bloque {BLOCK}, {B} "
          "remuestreos, los mismos para todas las reglas). Un p cercano a 1 indica que la regla es peor. "
          f"Sin corregir por la búsqueda: con p < 0,05 hay {sp['nominal']['LRS']} reglas frente a la LRS y "
          f"{sp['nominal']['1x']} frente al 1x; con Holm sobre las {n_new}, {sp['holm']['LRS']} y {sp['holm']['1x']}.")
    fig_delta(sp, rep)
    rep.p(rules_text(sp))
    tab("Δg frente a la LRS 3x original por ventana (pp/año y p unilateral)", sub_delta_table(sp, "LRS"), pct=set(),
        digits=3)
    tab("Δg frente al 1x por ventana (pp/año y p unilateral)", sub_delta_table(sp, "1x"), pct=set(), digits=3)
    rep.p("Cada regla se mide desde su primer día dentro de la ventana (las del VIX existen desde 1991, los "
          "clasificadores con VIX desde 2000 y las de VIX/VIX3M desde 2006). «1938–1973» empieza en realidad en "
          f"{start.date()}, cuando hay un año de previsiones fuera de muestra para fijar el umbral.")

    rep.h("(b) frente a bajar el apalancamiento", 3)
    eq = sp["igual_riesgo"].set_index("regla")
    show = eq.drop(columns=[c for c in eq if c.startswith(("IC90",))]).copy()
    for c in [c for c in show if c.startswith("Δg")]:
        show[c] = show[c].astype(float) * 100
    show = show.rename(columns={c: f"{c} (pp)" for c in show if c.startswith("Δg")})
    tab(f"LRS × volatilidad frente a la LRS con un apalancamiento fijo más bajo ({start.date()} → {end.date()})",
        show, pct={c for c in show if c.startswith(("CAGR", "vol", "maxDD"))})
    rep.p("L_exp: el apalancamiento fijo de la misma exposición media que la regla en los días sobre la media; L_vol: "
          "el de su misma volatilidad realizada. Las dos referencias son la LRS con esa L sobre la media (una mezcla "
          "fija de 3x y 1x, los instrumentos de la regla) y letras debajo; se fijan con toda la ventana, así que son "
          "cotas que nadie conocía de antemano, no reglas. Δg: la regla menos la referencia (p unilateral: H0, la "
          "regla no crece más).")
    eq16 = sp["igual_riesgo_2016"].set_index("regla")
    s16 = eq16[["L_exp", "L_vol", "CAGR_regla", "maxDD_regla", "CAGR_L_vol", "maxDD_L_vol", "Δg_vs_L_vol",
                "p_vs_L_vol"]].copy()
    s16["Δg_vs_L_vol"] = s16["Δg_vs_L_vol"].astype(float) * 100
    tab("Lo mismo en 2016–2026 (tras la muestra de Gayed y Bilello)", s16.rename(columns={"Δg_vs_L_vol": "Δg_vs_L_vol (pp)"}),
        pct={"CAGR_regla", "maxDD_regla", "CAGR_L_vol", "maxDD_L_vol"})
    fig_frontier(sp, rep)
    rep.p(equal_risk_text(sp))

    rep.h("(c) VIX y estructura temporal VIX/VIX3M", 3)
    cks = [k for k in G if G[k] in ("c", "t")]
    for w in ("completa", "ETFs reales (2010–2026)"):
        tt = sp["tablas"][w]
        tab(f"Reglas del VIX, {'desde su primer día' if w == 'completa' else 'con UPRO real (2010–2026)'}",
            show_rules(tt.loc[cks + [LRS, THREE, ONE]], vs3x=True).drop(columns=["grupo"]), digits=3)
    rep.p(vix_text(sp))
    fig_vix(sp, rep)

    rep.h("(d) Clasificador en avance", 3)
    at = sp["auc"].set_index(["clasificador", "ventana"])
    tab("S&P 500: capacidad predictiva fuera de muestra de los clasificadores", at, digits=3)
    rep.p("Objetivo: que el 3x rinda más que las letras en las 21 sesiones que cobra una posición decidida en t "
          "(de t + 2 a t + 22). Reentreno cada 252 sesiones con ventana expansiva, diez años de etiquetas como "
          "mínimo y embargo de 22 sesiones (solo se entrena con etiquetas ya conocidas). Rasgos: distancia a la media "
          "de 200 sesiones del índice de rentabilidad total, MACD(12, 26) relativo al precio y su histograma, la "
          "volatilidad GJR prevista (en nivel y frente a su mediana expansiva) y, con VIX, su logaritmo y el cociente "
          "entre la varianza implícita y la prevista. IC 90% del AUC por bootstrap estacionario de los pares "
          f"(probabilidad, resultado), bloque {BLOCK}, {B_AUC} remuestreos. «AUC distancia a la media» y «AUC −vol. "
          "relativa» son las mismas fechas puntuadas con un solo rasgo sin ajustar nada. acierto: con p ≥ 0,5; "
          "acierto_siempre_3x: la tasa base.")
    dk = [k for k in G if G[k] == "d"]
    for w in ("completa", "ETFs reales (2010–2026)"):
        tab(f"Los clasificadores como reglas, {'desde su primer día' if w == 'completa' else 'con UPRO real (2010–2026)'}",
            show_rules(sp["tablas"][w].loc[dk + [LRS, THREE, ONE]], vs3x=True).drop(columns=["grupo"]), digits=3)
    rep.p(clf_text(sp))

    # ── 5. búsqueda
    rep.h("5. Contraste de la búsqueda (SPA de Hansen)")
    tab("SPA de la familia de reglas nuevas en cada ventana", spa_show(sp["spa"]).set_index(["ventana", "familia", "frente a"]),
        pct=set(), digits=3)
    tab("SPA por grupo de reglas", spa_show(sp["spa_grupos"]).set_index(["grupo", "frente a"]), pct=set(), digits=3)
    rep.p(f"SPA de Hansen (2005) con recentrado consistente (p_SPA) e inferior (p_SPA_l), bootstrap estacionario "
          f"con bloque {BLOCK} y {B} remuestreos, sobre las diferencias diarias de log-rentabilidad en la ventana "
          "en que existen todas las reglas de la familia. «reglas nuevas + LRS original» añade la LRS a la familia "
          "frente al 1x. H0: ninguna regla de la familia supera a la referencia.")
    rep.p(spa_text(sp))

    # ── 6. otros mercados
    rep.h("6. Nasdaq-100 y semiconductores")
    agg, Sg, Ag = res["ndx"]
    full = agg.loc["completa"].copy()
    for c in [c for c in full if c.startswith("Δg")]:
        full[c] = full[c] * 100
    full = full.rename(columns={c: f"{c} (pp)" for c in full if c.startswith("Δg")})
    tab(f"Nasdaq-100 desde {r12.NDX_EVAL_FROM} (datos reales; mediana de {N_NDX} reconstrucciones del "
        "calentamiento 1971–85)", full, digits=3)
    tab(f"Nasdaq-100: SPA en las {N_NDX} reconstrucciones", Sg.assign(Δg_mejor_mediana=Sg.Δg_mejor_mediana * 100).rename(
        columns={"Δg_mejor_mediana": "Δg_mejor_mediana (pp)"}), pct=set(), digits=3)
    tab(f"Nasdaq-100: AUC de los clasificadores en las {N_NDX} reconstrucciones", Ag, pct={"fracción_en_3x"}, digits=3)
    rl = agg.loc["ETFs reales (2010–2026)"].copy()
    for c in [c for c in rl if c.startswith("Δg")]:
        rl[c] = rl[c] * 100
    tab("Nasdaq-100 con TQQQ real (2010–2026), mediana de las reconstrucciones",
        rl.rename(columns={c: f"{c} (pp)" for c in rl if c.startswith("Δg")}), digits=3)
    se = res["semis"]
    ts_ = se["tablas"]["completa"]
    tab(f"Semiconductores desde {ts_.loc[new[0], 'desde']}", show_rules(ts_.loc[new + [LRS, ROT, THREE, ONE]]),
        digits=3)
    tab("Semiconductores con SOXL real (2010–2026)",
        show_rules(se["tablas"]["ETFs reales (2010–2026)"].loc[new + [LRS, ROT, THREE, ONE]]), digits=3)
    tab("Semiconductores: SPA", spa_show(se["spa"]).set_index(["ventana", "familia", "frente a"]), pct=set(), digits=3)
    tab("Semiconductores: clasificadores", se["auc"].set_index(["clasificador", "ventana"]), digits=3)
    rep.p(other_text(res))

    # ── 7. lectura
    rep.h("7. Lectura")
    rep.p(reading_text(res))
    rep.h("Salvedades", 3)
    rep.p(caveats_text(res))
    rep.h("Fuentes", 3)
    rep.p(SOURCES)
    rep.h("Para el censo (R19 y R25)", 3)
    rep.p(f"census_rows(D=None, cache=None) devuelve las {n_new} series diarias netas del S&P 500 (nombre → "
          f"pd.Series con el índice del censo, NaN antes de su primer día; nombres con el prefijo «{CENSUS_PREFIX}»; "
          f"grupo y primer día en .attrs). Solo {len(long)} existen desde 1939 (grupos a, b y los clasificadores "
          "sin VIX); las del VIX empiezan en 1991, los clasificadores con VIX en 2000 y las de VIX/VIX3M en 2006, "
          "así que añadirlas todas a la ventana común del censo la acortaría a 2006. El censo ampliado (R25) "
          "mete las de historia larga en la ventana del artículo 2 y todas en una ventana desde 2006.")
    rep.p(f"Tiempo de cálculo: {num_es(res['segundos']['total'] / 60, 1)} minutos "
          f"(S&P 500 {num_es(res['segundos']['S&P 500'] / 60, 1)}; Nasdaq-100 "
          f"{num_es(res['segundos']['Nasdaq-100'] / 60, 1)}; semis {num_es(res['segundos']['Semis'] / 60, 1)}).")
    rep.save()
    return rep


def run(cache: str | None = None, n_jobs: int = 5):
    """Calcula todo y escribe reports/r23_vol/README.md con sus figuras. cache: directorio donde
    guardar (y reutilizar) las previsiones y los clasificadores de cada mercado."""
    return write_report(compute(cache, n_jobs))


if __name__ == "__main__":
    run()
