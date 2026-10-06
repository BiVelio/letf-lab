"""R12 — Historia larga (1928–) y momento óptimo con modelos de volatilidad y ML.

A. Reconstrucciones y su validación.
   - S&P 500 diario con dividendos desde 1928 (^GSPC + D/P de Shiller), frente
     al índice de rentabilidad total oficial en el periodo común.
   - Nasdaq-100 desde 1971 por imputación múltiple sobre el Nasdaq Composite,
     validado prediciendo 1991–1995 con un modelo estimado en 1985–1990.
B. El 3x en la historia larga: Gran Depresión, 1973–74 y L* con 98 años
   (aproximado y exacto, con y sin la comisión del fondo).
C. Previsión de la varianza a 21 días: ARCH, GARCH, GJR-GARCH, EWMA, HAR (en
   niveles, sobre las varianzas realizadas diaria, semanal y mensual) y cuatro
   modelos de machine learning (Ridge lineal, bosque aleatorio, gradient
   boosting y red neuronal) con 18 variables: el logaritmo de la varianza
   realizada de 1, 5, 21, 63 y 252 sesiones, la caída desde máximos, la
   distancia a la media de 200 sesiones, la rentabilidad de 21 sesiones y las
   rentabilidades de las cinco últimas sesiones y sus valores absolutos. Con
   reestimación periódica, embargo de 21 días y Model Confidence Set (Hansen,
   Lunde y Nason 2011).
D. Predicción de la rentabilidad a 21 días con los cuatro modelos de
   machine learning: R² fuera de muestra de Campbell y Thompson (2008)
   frente a la media histórica expansiva, Clark-West (2007) y acierto
   direccional. El objetivo es la rentabilidad de 21 sesiones evaluada cada
   día, así que el diferencial de Clark-West es un MA(20): su error estándar
   es el de Hansen y Hodrick (1980) con 20 retardos (con recurso a Newey-West
   de 42 si su varianza no sale positiva), y p_Holm corrige los 12 contrastes
   (4 modelos × 3 mercados).
E. Estrategias: exposición al 3x = medio Kelly con la varianza prevista por
   cada modelo, revisada cada 21 días, con costes y orden al cierre
   siguiente; SPA frente a comprar y mantener el 1x y el 3x.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm as st_norm

from letf_lab import kelly, metrics, sim, stats
from letf_lab.data import history
from letf_lab.report import FIG_W, Report, comma_fig, legend_band, num_es
from letf_lab.research import r10_predictability as r10
from letf_lab.research.common import long_underlying, rf
from letf_lab.research.r3_synthetic import synthetic
from letf_lab.universe import BY_TICKER

H = 21                 # horizonte de previsión y de revisión de la exposición
MIN_OBS = 252 * 10     # diez años antes de la primera previsión
REFIT = 504            # reestimación cada dos años
KELLY_FRACTION = 0.5   # fijado a priori (R10–R11: el L robusto es ≈ ½ L*)
COST_BPS = 5.0
SEED = 12
B = 500
N_SEEDS_MCS, B_SEEDS_MCS = 10, 1000   # sensibilidad del MCS de §C a la semilla
SEED_MCS_SEEDS = 1250
EXEC_LAG = 2           # señal al cierre de t, ejecución al cierre de t+1 (1 = mismo cierre, MOC)
NDX_EVAL_FROM = "1985-10-01"  # antes, Nasdaq-100 reconstruido: sólo para entrenar y calentar


# ───────────────────────────── datos ─────────────────────────────

def real_letf(ticker: str) -> pd.Series:
    """Rentabilidades del ETF REAL desde su lanzamiento (sin el sintético de R3,
    que antes se financiaba al T-bill: aquí el tramo previo lo pone letf())."""
    from letf_lab.research.common import pair
    return pair(ticker).letf.rename(ticker)


def letf(d: dict, L: float) -> pd.Series:
    """LETF sintético: rf + L·(u − f − s) − ER. Toda la exposición del swap paga f
    (call money hasta 1954, fondos federales después: history.funding_long) más el
    sobrecoste s; el colateral rinde la letra; ER es el gasto anual del fondo.

    Equivale a sim.letf_returns sobre u − (f − rf): la financiación es f, no la
    letra."""
    return sim.letf_returns(d["u"] - (d["f"] - d["rf"]), L, d["rf"], d["er"], d["spread"])


def make(u: pd.Series, r: pd.Series, ticker: str, real: pd.Series | None = None,
         eval_from: str | None = None) -> dict:
    """Serie del subyacente + 3x sintético (y real donde exista).

    El sobrecoste del swap se recalibra sobre el tipo de financiación f (call
    money hasta 1954, fondos federales después; no es un tipo interbancario en
    todo el periodo): s' = s − media(f − rf) en el periodo con el ETF real, para
    que el 3x sintético siga reproduciendo al real y antes pague la diferencia
    histórica f − rf."""
    e = BY_TICKER[ticker]
    f = history.funding_long(u.index)
    spread = synthetic(ticker)[1]
    if real is not None:
        # vida del ETF real (por si se pasa una serie que incluya un tramo sintético)
        from letf_lab import data as _data
        launch = _data.panel([ticker], "adj_close")[ticker].first_valid_index()
        live = real.dropna().index
        live = live[live >= launch]
        spread = spread - float((f - r).reindex(live).mean() * 252)
    d = {"u": u, "rf": r, "f": f, "spread": spread, "er": e.expense_ratio, "e": e,
         "eval_from": pd.Timestamp(eval_from) if eval_from else None}
    x3 = letf(d, 3.0)
    if real is not None:
        real = real.reindex(u.index)
        x3 = real.where(real.notna(), x3)
    d["x3"] = x3
    return d


def load_all() -> dict:
    u_sp = history.sp500_long()
    sp = make(u_sp, history.rf_long(u_sp.index), "UPRO", real_letf("UPRO"))
    nd = history.ndx_backcast()
    u_nd = nd["draws"][0]
    ndx = make(u_nd, history.rf_long(u_nd.index), "TQQQ", real_letf("TQQQ"),
               eval_from=NDX_EVAL_FROM)
    u_sx = long_underlying("SOXL")
    sox = make(u_sx, rf(u_sx.index), "SOXL", real_letf("SOXL"))
    return {"S&P 500": sp, "Nasdaq-100": ndx, "Semis": sox, "_ndx": nd}


# ───────────────────────────── A. validación ─────────────────────────────

def validate_sp() -> dict:
    from letf_lab import data
    tr = data.panel(["^SP500TR"], "adj_close")["^SP500TR"].pct_change(fill_method=None).dropna()
    px = data.panel(["^GSPC"], "close")["^GSPC"].pct_change(fill_method=None).dropna()
    dy = history.shiller_dividend_yield()
    rec = px + dy.reindex(px.index.union(dy.index)).ffill().reindex(px.index) / 252
    both = pd.concat([tr, rec], axis=1, keys=["oficial", "reconstruida"]).dropna()
    both = both[both.index <= dy.index[-1]]
    yr = (1 + both).groupby(both.index.year).prod() - 1
    return {"desde": both.index[0].date(), "hasta": both.index[-1].date(),
            "CAGR_oficial": metrics.cagr(both.oficial), "CAGR_reconstruida": metrics.cagr(both.reconstruida),
            "error_seguimiento_anual": (yr.oficial - yr.reconstruida).std(),
            "corr_anual": yr.corr().iloc[0, 1]}


def validate_ndx(M: int = 50, seed: int = 1) -> dict:
    """Pseudo fuera de muestra: se estima en 1985–1990 y se reconstruye 1991–1995."""
    from letf_lab import data
    ix = data.panel(["^IXIC"], "close")["^IXIC"].pct_change(fill_method=None).dropna()
    ndx = long_underlying("TQQQ")
    both = pd.concat([ndx, ix], axis=1, keys=["ndx", "ix"]).dropna()
    fit, test = both.loc["1985-10-01":"1990-12-31"], both.loc["1991-01-01":"1995-12-31"]
    s_fit = history.relative_drift_sd(ndx, ix, "1985-10-01", "1990-12-31")
    X = np.column_stack([np.ones(len(fit)), fit.ix])
    beta, *_ = np.linalg.lstsq(X, fit.ndx, rcond=None)
    res = fit.ndx.values - X @ beta
    rng = np.random.default_rng(seed)
    out = {}
    for variant in ("regresión libre", "deriva neutral"):
        cagrs, vols = [], []
        for _ in range(M):
            starts = rng.integers(0, len(res) - 21, len(test) // 21 + 1)
            e = np.concatenate([res[s:s + 21] for s in starts])[:len(test)]
            a = beta[0]
            if variant == "deriva neutral":
                e = e - e.mean()
                delta = rng.normal(0, s_fit / np.sqrt(len(test) / 252))
                a = (1 - beta[1]) * test.ix.mean() + (history.NDX_DIV + delta) / 252
            s = a + beta[1] * test.ix.values + e
            cagrs.append(metrics.cagr(pd.Series(s)))
            vols.append(np.std(s) * np.sqrt(252))
        lo, hi = np.percentile(cagrs, [5, 95])
        real = metrics.cagr(test.ndx)
        out[variant] = {"CAGR_real_1991_95": real, "CAGR_reconstruida": np.median(cagrs),
                        "IC90": f"[{lo:.1%}, {hi:.1%}]", "real_dentro_IC": bool(lo <= real <= hi),
                        "vol_real": test.ndx.std() * np.sqrt(252), "vol_reconstruida": np.median(vols)}
    out["_meta"] = {"beta": beta[1], "R2_diario": 1 - res.var() / fit.ndx.var(),
                    "CAGR_Composite_1991_95": metrics.cagr(test.ix), "s_rel_estimación": s_fit}
    return out


# ───────────────────────────── B. historia larga ─────────────────────────────

def long_history_stats(d: dict) -> pd.DataFrame:
    u, r = d["u"], d["rf"]
    x2 = letf(d, 2.0)
    rows = {}
    for name, s in (("1x", u), ("2x", x2), ("3x", d["x3"])):
        m = metrics.summary(s.dropna(), r)
        rows[name] = {"desde": m["start"], "CAGR": m["CAGR"], "vol": m["vol"], "maxDD": m["maxDD"],
                      "años_bajo_el_agua": m["max_underwater_d"] / 252}
    return pd.DataFrame(rows).T


def episode(d: dict, start: str, end: str) -> dict:
    out = {}
    for name, s in (("1x", d["u"]), ("3x", d["x3"])):
        w = (1 + s.loc[start:end].fillna(0)).cumprod()
        out[f"{name}_caída_máx"] = (w / w.cummax() - 1).min()
        out[f"{name}_final"] = w.iloc[-1] - 1
    return out


def lstar_long(d: dict, rng) -> dict:
    """L* aproximado (kelly.lstar_approx) = (μ − f − s)/σ² con el tipo de financiación f
    (el mismo con el que se recalibra s en make) e intervalo por bootstrap
    estacionario de bloques de un año."""
    df = pd.concat([d["u"], d["f"]], axis=1, keys=["u", "f"]).dropna()
    xy = df.values
    s = d["spread"]

    def f(a):
        return kelly.lstar_approx(a[:, 0], a[:, 1], s)

    est, lo, hi = stats.bootstrap_ci(xy, f, 252, B, rng)
    return {"años": len(xy) / 252, "L*": est, "IC90_bajo": lo, "IC90_alto": hi}


LSTAR_L_MAX = 3.0   # L* exacto con los instrumentos de la LRS: L/3 en el 3x y el resto en letras


def lstar_window(d: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Ventana de las estrategias de R13–R17 en un mercado: del primer mes completo
    tras la primera previsión (MIN_OBS sesiones, como garch_walk) al último mes
    cerrado. En el S&P 500 es la ventana de evaluación de R17 (1938-03 → último mes)."""
    idx = d["u"].dropna().index
    first = idx[MIN_OBS]
    if d["eval_from"] is not None:
        first = max(first, d["eval_from"])
    per = idx.to_period("M")
    p0 = first.to_period("M") + (0 if idx[per == first.to_period("M")][0] == first else 1)
    ends = pd.Series(idx, index=idx).groupby(per).max()
    return idx[per == p0][0], ends.iloc[-2]


def lstar_rows(d: dict, start=None, end=None) -> dict:
    """L* aproximado y exacto (con y sin la comisión del fondo) en [start, end], con
    los rótulos compartidos de kelly y el exacto limitado a LSTAR_L_MAX."""
    t = kelly.lstar_table(d, start, end, L_max=LSTAR_L_MAX)
    return {"desde": t["desde"], "hasta": t["hasta"],
            **{k: t[k] for k in (kelly.LSTAR_APROX, kelly.LSTAR_EXACTO_SIN, kelly.LSTAR_EXACTO)}}


# ───────────────────────────── C. varianza ─────────────────────────────

def features(u: pd.Series) -> pd.DataFrame:
    rv = u ** 2 * 252
    px = (1 + u.fillna(0)).cumprod()
    X = pd.DataFrame({
        "lrv_d": np.log(rv.clip(lower=1e-6)),
        "lrv_w": np.log(rv.rolling(5).mean().clip(lower=1e-6)),
        "lrv_m": np.log(rv.rolling(21).mean().clip(lower=1e-6)),
        "lrv_q": np.log(rv.rolling(63).mean().clip(lower=1e-6)),
        "lrv_y": np.log(rv.rolling(252).mean().clip(lower=1e-6)),
        "caída": px / px.cummax() - 1,
        "tendencia": px / px.rolling(200).mean() - 1,
        "ret_m": u.rolling(21).sum(),
    })
    for k in range(1, 6):
        X[f"r_{k}"] = u.shift(k - 1)
        X[f"abs_{k}"] = u.abs().shift(k - 1)
    return X


def _ml_models():
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "Ridge (lineal)": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Bosque aleatorio": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=200,
                                                          max_features=0.5, n_jobs=-1, random_state=SEED),
        "Gradient boosting": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                   min_samples_leaf=200, random_state=SEED),
        "Red neuronal (MLP)": lambda: make_pipeline(
            StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=300,
                                           early_stopping=True, random_state=SEED)),
    }


def walk_forward(X: pd.DataFrame, y: pd.Series, make_model, log_target: bool) -> pd.Series:
    """Previsión en t con un modelo entrenado sólo con objetivos ya observados
    (s ≤ t − H: embargo del horizonte), reentrenado cada REFIT días."""
    df = pd.concat([X, y.rename("__y")], axis=1)
    ok_x = df.drop(columns="__y").notna().all(axis=1).values
    ok_y = df["__y"].notna().values
    Xa, ya = df.drop(columns="__y").values, df["__y"].values
    out = np.full(len(df), np.nan)
    model, smear = None, 1.0
    first = np.argmax(np.cumsum(ok_x & ok_y) >= MIN_OBS)
    for start in range(first, len(df), REFIT):
        train = np.zeros(len(df), bool)
        train[: max(start - H + 1, 0)] = True
        train &= ok_x & ok_y
        model = make_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xa[train], ya[train])
            if log_target:
                # Corrección de sesgo al volver a niveles (Duan 1983) con residuos
                # FUERA de muestra: se reestima sin los dos últimos años del
                # entrenamiento y se mide el sesgo en ellos. Con residuos dentro de
                # muestra, los modelos que sobreajustan salían sesgados a la baja.
                idx_tr = np.flatnonzero(train)
                hold, core = idx_tr[-REFIT:], idx_tr[:-REFIT - H]
                aux = make_model()
                aux.fit(Xa[core], ya[core])
                smear = float(np.mean(np.exp(ya[hold] - aux.predict(Xa[hold]))))
        seg = np.arange(start, min(start + REFIT, len(df)))
        seg = seg[ok_x[seg]]
        if len(seg):
            p = model.predict(Xa[seg])
            out[seg] = np.exp(p) * smear if log_target else p
    return pd.Series(out, index=df.index)


def garch_walk(u: pd.Series, kind: str) -> pd.Series:
    r = u.fillna(0).values
    out = pd.Series(np.nan, index=u.index)
    for start in range(MIN_OBS, len(u), REFIT * 2):
        p = stats.garch_fit(r[:start], kind)
        hn = stats.garch_filter(r - r[:start].mean(), p)
        seg = slice(start, min(start + REFIT * 2, len(u)))
        out.iloc[seg] = stats.garch_horizon_var(hn[seg], p, H) * 252
    return out


def variance_race(d: dict) -> pd.DataFrame:
    u = d["u"].dropna()
    target = (u ** 2).rolling(H).sum().shift(-H) * 252 / H
    X = features(u)
    F = {
        "EWMA λ=0,94": (u ** 2 * 252).ewm(alpha=0.06, adjust=False).mean(),
        "ARCH(1)": garch_walk(u, "arch"),
        "GARCH(1,1)": garch_walk(u, "garch"),
        "GJR-GARCH": garch_walk(u, "gjr"),
        "HAR": walk_forward(X[["lrv_d", "lrv_w", "lrv_m"]].apply(np.exp), target,
                            lambda: __import__("sklearn.linear_model", fromlist=["LinearRegression"]).LinearRegression(),
                            log_target=False).clip(lower=1e-4),
    }
    ly = np.log(target.clip(lower=1e-6))
    for name, mk in _ml_models().items():
        F[name] = walk_forward(X, ly, mk, log_target=True)
    df = pd.DataFrame(F)
    df["realizada"] = target
    first = df.drop(columns="realizada").apply(lambda c: c.first_valid_index()).max()
    df = df[df.index >= first]
    # la realizada de las últimas H sesiones aún no se conoce: se conservan las
    # previsiones (las usan las estrategias) y la evaluación descarta esas filas
    return df.dropna(subset=[c for c in df.columns if c != "realizada"])


# ───────────────────────────── D. rentabilidad ─────────────────────────────

CW_NW = (H, 3 * H)   # Newey-West de 21 (la versión anterior) y de 63 retardos: trazabilidad


def cw_inference(y: pd.Series, bench: pd.Series, fc: pd.Series, h: int = H,
                 calendar: pd.DatetimeIndex | None = None) -> dict:
    """Clark-West de una previsión de objetivos solapados de h sesiones (inferencia pura,
    sin reentrenar nada).

    Principal: Hansen-Hodrick con h − 1 retardos y, si su varianza no sale positiva,
    Newey-West con 2h (stats.hac_overlap), con el estimador usado. Trazabilidad:
    Newey-West con h y 3h retardos y la submuestra mensual no solapada (una
    observación de cada h sesiones, los h desfases, Newey-West de stats.NONOV_LAGS):
    mediana, mínimo y máximo de su p. p unilaterales (H0: el modelo no mejora al
    benchmark anidado). y, bench y fc: series alineadas por fecha; calendar, las
    sesiones del mercado (por defecto las de y), para que la submuestra mensual avance
    de h en h sesiones aunque falte alguna fecha."""
    m = pd.concat([y, fc, bench], axis=1, keys=["y", "f", "b"]).dropna()
    fit = stats.clark_west_fit(m.y.values, m.b.values, m.f.values, **stats.hac_overlap(h))
    t = float(fit.t[0])
    out = {"CW_t": t, "CW_p": float(st_norm.sf(t)), "CW_estimador": fit.estimator}
    for lags in (h, 3 * h):
        out[f"CW_p_NW{lags}"] = stats.clark_west(m.y.values, m.b.values, m.f.values, lags=lags)[1]
    cal = y.index if calendar is None else calendar
    dcw = stats.clark_west_diff(m.y, m.b, m.f)
    dcw = dcw.reindex(cal[(cal >= dcw.index[0]) & (cal <= dcw.index[-1])])
    p_m = st_norm.sf(stats.nonoverlap(dcw, None, "const", h)["t"].astype(float))
    out.update({"CW_p_mensual_mediana": float(np.median(p_m)), "CW_p_mensual_mín": float(np.min(p_m)),
                "CW_p_mensual_máx": float(np.max(p_m))})
    return out


def return_race(d: dict, rng) -> pd.DataFrame:
    """Previsión de la rentabilidad de H sesiones con los modelos de ML (walk_forward)
    frente a la media histórica expansiva observada en t: R² fuera de muestra de
    Campbell y Thompson (2008), Clark-West (cw_inference) y acierto direccional."""
    u, r = d["u"].dropna(), d["rf"]
    start = d["eval_from"] or u.index[0]
    y = r10.window_sum(np.log1p(u) - np.log1p(r.reindex(u.index)), H)
    X = features(u)
    bench = y.shift(H).expanding(MIN_OBS).mean()  # media histórica observada en t
    rows = {}
    for name, mk in _ml_models().items():
        f = walk_forward(X, y, mk, log_target=False)
        m = pd.concat([y, f, bench], axis=1, keys=["y", "f", "b"]).dropna()
        m = m[m.index >= start]
        cw = cw_inference(m.y, m.b, m.f, calendar=u.index)
        hit = ((m.f > 0) == (m.y > 0)).values
        base = ((m.y > 0).mean())
        up = (m.y > 0).values
        idx = [stats.stationary_indices(len(hit), H * 3, rng) for _ in range(B)]
        ci = np.percentile([hit[i].mean() - up[i].mean() for i in idx], [5, 95])
        rows[name] = {"R2_OOS": stats.oos_r2(m.y.values, m.f.values, m.b.values), **cw,
                      "acierto_dirección": hit.mean(), "acierto_si_siempre_sube": base,
                      "exceso_acierto_IC90": f"[{ci[0]:+.1%}, {ci[1]:+.1%}]",
                      "desde": m.index[0].date()}
    return pd.DataFrame(rows).T


def cw_holm(races: dict) -> pd.DataFrame:
    """Los contrastes de Clark-West de todos los mercados (4 modelos × 3) con p_Holm
    sobre el p principal (Hansen-Hodrick)."""
    rows = []
    for name, t in races.items():
        for model, row in t.iterrows():
            rows.append({"mercado": name, "modelo": model, "CW_t": float(row.CW_t), "CW_p": float(row.CW_p),
                         "CW_estimador": row.CW_estimador})
    df = pd.DataFrame(rows)
    df["p_Holm"] = stats.holm(df.CW_p.values)
    return df


# ───────────────────────────── E. estrategias ─────────────────────────────

def strategy(d: dict, var_fc: pd.Series, mu_fc: pd.Series | None = None,
             lag: int = EXEC_LAG) -> tuple[pd.Series, pd.Series]:
    """Exposición al 3x = ½·L*_t/3, con L*_t = (μ̂_t − f̂_t − s)/σ²_t, revisada cada H días.

    μ̂_t − f̂_t es la prima compartida (kelly.premium): media creciente de u − f con
    datos hasta t − 1. Con una previsión mu_fc de la rentabilidad aritmética del
    índice, la prima es mu_fc − f_t."""
    u, r = d["u"], d["rf"]
    prem = mu_fc - d["f"] * 252 if mu_fc is not None else kelly.premium(u, d["f"], MIN_OBS)
    # Kelly con toda la exposición (L veces el capital) financiada a f: L* = (μ − f − s)/σ²
    L = KELLY_FRACTION * (prem - d["spread"]) / var_fc
    w = (L / 3).clip(0, 1)
    # revisión cada H días: se mantiene el peso del último día de revisión
    keep = np.zeros(len(w), bool)
    keep[::H] = True
    w = w.where(pd.Series(keep, index=w.index)).ffill()
    w = w.shift(lag).fillna(0.0)  # decidido al cierre de t, ejecutado según EXEC_LAG
    ret = w * d["x3"] + (1 - w) * r - w.diff().abs().fillna(0) * COST_BPS / 1e4
    return ret, w


def strategy_table(d: dict, V: pd.DataFrame, mu_ml: pd.Series, rng) -> tuple[pd.DataFrame, dict]:
    start = max(V.index[0], d["eval_from"]) if d["eval_from"] is not None else V.index[0]
    u, r = d["u"], d["rf"]
    S = {"1x comprar y mantener": u, "3x comprar y mantener": d["x3"]}
    for m in V.drop(columns="realizada").columns:
        full = V[m].reindex(u.index)
        S[f"½ Kelly · {m}"] = strategy(d, full)[0]
    # mu_ml es exceso LOGARÍTMICO sobre el T-bill a 21 días. Kelly pide la media
    # aritmética: μ ≈ E[exceso log] + r + σ²/2.
    mu_total = mu_ml * 252 / H + r * 252 + 0.5 * V["GJR-GARCH"].reindex(u.index)
    S["½ Kelly · GJR-GARCH + μ por gradient boosting"] = strategy(
        d, V["GJR-GARCH"].reindex(u.index), mu_total)[0]
    S = {k: v[v.index >= start].dropna() for k, v in S.items()}
    common = pd.concat(S, axis=1).dropna()
    rows = {}
    for k in common:
        m = metrics.summary(common[k], r)
        rows[k] = {"CAGR": m["CAGR"], "vol": m["vol"], "sharpe": m["sharpe"], "maxDD": m["maxDD"],
                   "calmar": m["calmar"]}
    logs = np.log1p(common.clip(lower=-0.999))
    strat = [c for c in logs if c.startswith("½ Kelly")]
    tests = {}
    for bench in ("1x comprar y mantener", "3x comprar y mantener"):
        D = logs[strat].sub(logs[bench], axis=0).values
        spa = stats.spa_test(D, 63, B, rng)
        tests[bench] = {"p_SPA_c": spa["p_SPA_c"], "mejor": strat[int(D.mean(axis=0).argmax())],
                        "Δ_crec_anual_mejor": D.mean(axis=0).max() * 252}
    return pd.DataFrame(rows).T, {"tests": tests, "desde": common.index[0].date(),
                                  "curvas": (1 + common).cumprod()}


# ───────────────────────────── figuras ─────────────────────────────

def fig_long(D: dict, nd: dict, rep: Report):
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 9))
    d = D["S&P 500"]
    x2 = letf(d, 2.0)
    for s, lab, c in ((d["u"], "1x", "k"), (x2, "2x", "C0"), (d["x3"], "3x", "C3")):
        axes[0].semilogy((1 + s.fillna(0)).cumprod(), color=c, lw=1, label=lab)
    axes[0].set(title="S&P 500 desde 1928: riqueza de 1 dólar", ylabel="Riqueza (escala log)")
    axes[0].legend()
    ax = axes[1]
    paths = []
    ndd = D["Nasdaq-100"]
    real = real_letf("TQQQ")
    for u in nd["draws"]:
        x3 = make(u, history.rf_long(u.index), "TQQQ", real)["x3"]
        paths.append(np.log((1 + x3.fillna(0)).cumprod()))
    P = pd.concat(paths, axis=1)
    lo, med, hi = np.exp(P.quantile(0.05, axis=1)), np.exp(P.median(axis=1)), np.exp(P.quantile(0.95, axis=1))
    ax.fill_between(P.index, lo, hi, color="C3", alpha=0.2, label="3x: banda del 90%")
    ax.semilogy(med, color="C3", lw=1, label="3x: mediana")
    ax.semilogy((1 + nd["draws"][0].fillna(0)).cumprod(), color="k", lw=1, label="1x (una reconstrucción)")
    ax.axvline(pd.Timestamp("1985-10-01"), color="0.4", ls="--", lw=1)
    ax.text(pd.Timestamp("1985-10-01"), 0.02, " datos reales →", transform=ax.get_xaxis_transform(),
            va="bottom", color="0.3")
    ax.set(title="Nasdaq-100 desde 1971 (reconstruido antes de 1985)",
           ylabel="Riqueza (escala log)")
    ax.legend(loc="upper left")
    fig.suptitle("La historia larga de un 3x")
    rep.fig(fig, "historia_larga", "Riqueza de 1x, 2x y 3x en el S&P 500 desde 1928 y en el Nasdaq-100 desde 1971")


def mcs_seed_table(ql: pd.DataFrame, n: int, B: int, seed0: int) -> pd.DataFrame:
    """MCS con QLIKE repetido con n semillas propias: nº de semillas dentro al 90% y rango del p."""
    res = [stats.model_confidence_set(ql, 63, B, np.random.default_rng(seed0 + k)) for k in range(n)]
    p = pd.concat([r.p_MCS for r in res], axis=1)
    inn = pd.concat([r.en_MCS for r in res], axis=1).astype(int)
    return pd.DataFrame({"dentro (semillas)": inn.sum(axis=1), "p_MCS_mín": p.min(axis=1),
                         "p_MCS_máx": p.max(axis=1)}).loc[res[0].index]


RACE_IN, RACE_SOME, RACE_OUT = "C2", "#a8dca8", "0.6"   # dentro con todas las semillas / con algunas / fuera


def race_class(t: pd.DataFrame, n_seeds: int = N_SEEDS_MCS) -> pd.Series:
    """Pertenencia al MCS al 90% con QLIKE según las n_seeds semillas de la sensibilidad
    («dentro (semillas)»): «todas», «algunas» o «ninguna». La corrida principal (B = 500) no
    decide el color: sus p cerca de 0,10 caen dentro o fuera según la semilla, y la figura no
    debe afirmar más que el texto. Sin la columna de semillas, la de la corrida principal."""
    if "dentro (semillas)" not in t:
        return t["en_MCS_QLIKE"].map({True: "todas", False: "ninguna"})
    k = t["dentro (semillas)"].astype(int)
    return pd.Series(np.where(k >= n_seeds, "todas", np.where(k > 0, "algunas", "ninguna")), index=t.index)


def fig_race(race: dict, rep: Report):
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(len(race), 1, figsize=(FIG_W, 3.4 * len(race) + 0.3))
    axes = np.atleast_1d(axes)
    color = {"todas": RACE_IN, "algunas": RACE_SOME, "ninguna": RACE_OUT}
    hatch = {"todas": None, "algunas": "///", "ninguna": None}
    seen = set()
    for ax, (name, t) in zip(axes, race.items()):
        t = t.sort_values("pérdida_media_QLIKE", ascending=False)
        cls = race_class(t)
        seen |= set(cls)
        y = np.arange(len(t))                    # posiciones numéricas: el orden de la tabla
        for c in color:
            m = (cls == c).values
            if m.any():
                ax.barh(y[m], t["pérdida_media_QLIKE"].values[m], color=color[c], hatch=hatch[c],
                        edgecolor=RACE_IN if c == "algunas" else color[c], lw=0)
        ax.set_yticks(y, t.index)
        ax.set_ylim(-0.6, len(t) - 0.4)
        ax.set(title=f"{r10.fig_name(name)}: pérdida QLIKE media", xlabel="QLIKE (menos es mejor)")
        ax.grid(axis="x", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    n = N_SEEDS_MCS
    lab = {"todas": f"dentro con las {n} semillas", "algunas": f"dentro según la semilla (1 a {n - 1} de {n})",
           "ninguna": "fuera con todas"}
    if not any("dentro (semillas)" in t for t in race.values()):
        lab = {"todas": "dentro", "ninguna": "fuera"}
    handles = [Patch(facecolor=color[c], hatch=hatch[c], edgecolor=RACE_IN if c == "algunas" else color[c],
                     lw=0, label=lab[c]) for c in color if c in lab and (c in seen or c != "algunas")]
    legend_band(fig, handles, [h.get_label() for h in handles], ncol=len(handles), fontsize=11,
                title="Conjunto de mejores modelos al 90% (MCS con QLIKE)", title_fontsize=11,
                columnspacing=1.2, handletextpad=0.5)
    fig.suptitle("Previsión de la volatilidad del mes siguiente:\nmodelos clásicos frente a machine learning")
    comma_fig(fig)
    rep.fig(fig, "carrera_volatilidad", "Pérdida de cada modelo al prever la volatilidad del mes siguiente; en "
            f"verde, los modelos dentro del conjunto de mejores modelos al 90% (Hansen, Lunde y Nason 2011) con "
            f"las {n} semillas de la sensibilidad (B = {num_es(B_SEEDS_MCS, 0)}); rayado en verde claro, los que "
            "quedan dentro con unas semillas y fuera con otras; en gris, los que quedan fuera con todas")


def fig_strategies(curves: dict, rep: Report):
    fig, axes = plt.subplots(len(curves), 1, figsize=(12, 5 * len(curves)))
    for ax, (name, c) in zip(np.atleast_1d(axes), curves.items()):
        keep = ["1x comprar y mantener", "3x comprar y mantener", "½ Kelly · GJR-GARCH",
                "½ Kelly · Gradient boosting", "½ Kelly · EWMA λ=0,94"]
        styles = {"1x comprar y mantener": ("k", 1.5), "3x comprar y mantener": ("0.6", 1.2)}
        for k in keep:
            if k in c:
                col, lw = styles.get(k, (None, 1.5))
                ax.semilogy(c[k], lw=lw, color=col, label=k)
        ax.set(title=name, ylabel="Riqueza (log)")
        ax.legend(fontsize=10, loc="upper left")
    fig.suptitle("Exposición al 3x según la volatilidad prevista, frente a comprar y mantener")
    rep.fig(fig, "estrategias", "Riqueza de las estrategias de exposición gestionada frente a comprar y mantener")


# ───────────────────────────── informe ─────────────────────────────

def run():
    rep = Report("r12_long_ml", "R12 — Historia larga y momento óptimo con volatilidad y ML")
    rep.p(__doc__.split("\n", 2)[2])
    rng = np.random.default_rng(SEED)
    D = load_all()
    nd = D.pop("_ndx")

    rep.h("A. Reconstrucciones y validación")
    rep.h("S&P 500 con dividendos de Shiller frente al índice oficial de rentabilidad total", 3)
    rep.table(pd.Series(validate_sp()).to_frame("valor"), digits=4)
    rep.h("Nasdaq-100 reconstruido: estimado en 1985–1990, evaluado en 1991–1995", 3)
    vn = validate_ndx()
    meta = vn.pop("_meta")
    rep.table(pd.DataFrame(vn).T, pct={"CAGR_real_1991_95", "CAGR_reconstruida", "vol_real",
                                       "vol_reconstruida"}, digits=3)
    rep.p(f"Pendiente sobre el Composite {meta['beta']:.2f}, R² diario {meta['R2_diario']:.2f}; "
          f"el Composite rindió {meta['CAGR_Composite_1991_95']:.1%} anual en 1991–95. "
          "La regresión libre reproduce la volatilidad pero no la media. La variante de "
          "deriva neutral fija la media en la del Composite más el dividendo del NDX y "
          f"añade un diferencial aleatorio con desviación típica {meta['s_rel_estimación']:.1%} "
          "anual (estimada en el tramo de ajuste) dividida por la raíz de los años.")
    rep.p(f"Modelo final (1985–1995): pendiente {nd['b']:.3f} sobre el Composite, R² diario "
          f"{nd['r2']:.3f}; diferencial NDX − Composite con desviación típica {nd['s_rel']:.1%} "
          f"anual (1986–2025). {len(nd['draws'])} reconstrucciones. El Nasdaq-100 sólo se "
          f"evalúa desde {NDX_EVAL_FROM}: la reconstrucción sirve para entrenar y calentar "
          "los modelos, y hereda la autocorrelación diaria del Composite (precios desfasados).")

    rep.h("B. El 3x en la historia larga")
    for name, d in D.items():
        rep.h(f"{name} (desde {d['u'].index[0].year})", 3)
        rep.table(long_history_stats(d), pct={"CAGR", "vol", "maxDD"}, digits=2)
    ep = {"S&P 500 · 1929–1932": episode(D["S&P 500"], "1929-09-03", "1932-07-08"),
          "S&P 500 · 1937–1942": episode(D["S&P 500"], "1937-03-06", "1942-04-28"),
          "S&P 500 · 1973–1974": episode(D["S&P 500"], "1973-01-11", "1974-10-03"),
          "Nasdaq-100 · 1973–1974": episode(D["Nasdaq-100"], "1973-01-11", "1974-10-03")}
    rep.h("Episodios", 3)
    ept = pd.DataFrame(ep).T
    rep.table(ept, pct=set(ept.columns), digits=4)
    rep.h("L* con toda la historia", 3)
    ls = {name: lstar_long(d, rng) for name, d in D.items()}
    nd_ls = []
    real_tq = real_letf("TQQQ")
    for u in nd["draws"][:20]:
        dd = make(u, history.rf_long(u.index), "TQQQ", real_tq)
        nd_ls.append(lstar_long(dd, rng)["L*"])
    # filas y columnas añadidas tras la revisión: sin sorteos del generador compartido
    rng_l = np.random.default_rng(SEED + 1)
    sp = D["S&P 500"]
    w0, w1 = lstar_window(sp)
    sp_w = {**sp, "u": sp["u"].loc[w0:w1], "f": sp["f"].loc[w0:w1]}
    lab_sp = f"S&P 500 ({w0:%Y-%m} → {w1:%Y-%m})"
    rows = {}
    for name, d in D.items():
        lab = f"{name} (desde {d['u'].dropna().index[0].year})"
        rows[lab] = {**ls[name], **lstar_rows(d)}
        if name == "S&P 500":
            rows[lab_sp] = {**lstar_long(sp_w, rng_l), **lstar_rows(sp, w0, w1)}
    lt = pd.DataFrame(rows).T.rename(columns={"L*": kelly.LSTAR_APROX})
    lt = lt.loc[:, ~lt.columns.duplicated()]
    for c in ("años", kelly.LSTAR_APROX, "IC90_bajo", "IC90_alto"):
        lt[c] = lt[c].astype(float)
    for c in (kelly.LSTAR_EXACTO_SIN, kelly.LSTAR_EXACTO):
        lt[c] = lt[c].map(lambda v: kelly.fmt_lstar(v, L_max=LSTAR_L_MAX))
    rep.table(lt[["desde", "hasta", "años", kelly.LSTAR_APROX, "IC90_bajo", "IC90_alto",
                  kelly.LSTAR_EXACTO_SIN, kelly.LSTAR_EXACTO]], digits=2)
    rep.p(f"{kelly.LSTAR_APROX}: {kelly.LSTAR_DESC[kelly.LSTAR_APROX]}, con μ la media de u, f la de la "
          "financiación, s el sobrecoste del swap y σ² la varianza de u; IC 90% por bootstrap estacionario "
          f"de bloques de un año ({B} remuestreos). {kelly.LSTAR_EXACTO}: el L de 0 a "
          f"{LSTAR_L_MAX:g} que maximiza el crecimiento logarítmico medio de L/3 en el 3x del laboratorio (con su "
          "comisión) y el resto en letras, reequilibrado a diario; sin comisión, lo mismo con el 3x más su "
          f"gasto anual. Con la comisión, el óptimo baja unos ER/(3σ²). La fila {lab_sp} es la ventana de "
          "evaluación de las estrategias de R13–R17, sin la Gran Depresión: su L* exacto con comisión es el "
          "máximo de la frontera «3x + letras» de R17 (allí en una rejilla de 0,05).")
    rep.p(f"Nasdaq-100: L* aproximado a lo largo de 20 reconstrucciones entre {num_es(min(nd_ls))} y "
          f"{num_es(max(nd_ls))} (mediana {num_es(np.median(nd_ls))}).")
    fig_long(D, nd, rep)

    rep.h("C. Previsión de la volatilidad del mes siguiente")
    race, V_all = {}, {}
    for name, d in D.items():
        V_full = variance_race(d)
        V_all[name] = V_full
        V = V_full[V_full.index >= d["eval_from"]] if d["eval_from"] is not None else V_full
        V = V.dropna()
        y = V.realizada.values
        ql = pd.DataFrame({c: stats.qlike(y, V[c].values) for c in V.drop(columns="realizada")},
                          index=V.index)
        mse = pd.DataFrame({c: (y - V[c].values) ** 2 * 100 for c in V.drop(columns="realizada")},
                           index=V.index)
        t = stats.model_confidence_set(ql, 63, B, rng).add_suffix("_QLIKE").join(
            stats.model_confidence_set(mse, 63, B, rng).add_suffix("_MSE"))
        race[name] = t
        rep.h(f"{name} ({V.index[0].date()} → {V.index[-1].date()})", 3)
        rep.table(t, digits=3)
        rep.p(f"Sensibilidad a la semilla (añadida tras una revisión externa; {N_SEEDS_MCS} semillas propias "
              f"de B = {num_es(B_SEEDS_MCS, 0)}, sin tocar el generador del resto del informe): en cuántas "
              "queda cada modelo dentro del conjunto al 90% con QLIKE y el rango de su p_MCS. La figura "
              "colorea con esta tabla, no con la corrida principal.")
        seeds = mcs_seed_table(ql, N_SEEDS_MCS, B_SEEDS_MCS, SEED_MCS_SEEDS + 100 * len(race))
        rep.table(seeds, digits=3)
        race[name] = t.join(seeds[["dentro (semillas)"]])
    fig_race(race, rep)

    rep.h("D. Predicción de la rentabilidad del mes siguiente")
    mu_ml, races = {}, {}
    for name, d in D.items():
        rep.h(name, 3)
        rr = races[name] = return_race(d, rng)
        rep.table(rr, pct={"R2_OOS", "acierto_dirección", "acierto_si_siempre_sube"}, digits=4)
        u = d["u"].dropna()
        y = r10.window_sum(np.log1p(u) - np.log1p(d["rf"].reindex(u.index)), H)
        mu_ml[name] = walk_forward(features(u), y, _ml_models()["Gradient boosting"], False)
    rep.p("R2_OOS: R² fuera de muestra de Campbell y Thompson (2008), 1 − ECM del modelo / ECM de la media "
          "histórica expansiva del exceso de 21 sesiones observada en t. CW_p: Clark-West (2007), p unilateral "
          f"con el error de Hansen y Hodrick (1980) de {H - 1} retardos (CW_estimador dice si hubo que recurrir "
          f"a Newey-West de {2 * H}). Trazabilidad: CW_p_NW{H} es la versión anterior (Newey-West de {H} "
          f"retardos, que con un MA({H - 1}) infravalora la varianza de largo plazo), CW_p_NW{3 * H} alarga la "
          f"ventana, y CW_p_mensual es el mismo contraste con una observación de cada {H} sesiones (objetivos "
          f"que no se solapan, Newey-West de {stats.NONOV_LAGS}): mediana, mínimo y máximo de los {H} "
          "desfases posibles.")
    rep.h("Clark-West: los 12 contrastes con corrección de Holm", 3)
    rep.table(cw_holm(races), index=False, digits=4)
    rep.p("p_Holm corrige los 12 contrastes (4 modelos × 3 mercados) sobre el p de Hansen-Hodrick.")

    rep.h("E. Estrategias: medio Kelly con la volatilidad prevista (revisión cada 21 días)")
    curves = {}
    for name, d in D.items():
        tab, info = strategy_table(d, V_all[name], mu_ml[name].reindex(d["u"].index), rng)
        curves[name] = info["curvas"]
        rep.h(f"{name} (fuera de muestra desde {info['desde']})", 3)
        rep.table(tab, pct={"CAGR", "vol", "maxDD"}, digits=2)
        rep.table(pd.DataFrame(info["tests"]).T, pct={"Δ_crec_anual_mejor"}, digits=3)
    fig_strategies(curves, rep)
    rep.save()


if __name__ == "__main__":
    run()
