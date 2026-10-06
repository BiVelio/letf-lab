"""Econometría para series financieras solapadas (sólo numpy/scipy).

- OLS con errores estándar HAC: Newey-West (1987), núcleo de Bartlett, o
  Hansen-Hodrick (1980), núcleo uniforme con h − 1 retardos, el natural para
  objetivos solapados de h días. Convención de la serie para esos objetivos
  (hac_overlap): Hansen-Hodrick y, si su varianza no sale positiva, Newey-West
  con 2h retardos, declarado en el resultado. Submuestras no solapadas.
- Bootstrap estacionario de Politis y Romano (1994).
- R² fuera de muestra de Campbell y Thompson (2008) y test MSPE-ajustado de
  Clark y West (2007) para modelos anidados.
- Reality Check de White (2000) sobre una familia de reglas.
- Correcciones por pruebas múltiples: Holm (1979) y Benjamini-Hochberg (1995).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.signal import lfilter


# ───────────────────────────── OLS con HAC ─────────────────────────────

KERNELS = {"bartlett": "NW", "uniform": "HH"}  # núcleo → sigla del estimador


def hac_label(lags: int, kernel: str = "bartlett") -> str:
    """Rótulo del estimador HAC: «NW 42» (Newey-West) o «HH 20» (Hansen-Hodrick)."""
    return f"{KERNELS[kernel]} {lags}"


@dataclass
class OLS:
    names: list[str]
    beta: np.ndarray
    se: np.ndarray
    r2: float
    n: int
    # estimador con el que se calculó `se` (el del recurso, si se usó)
    lags: int | None = None
    kernel: str = "bartlett"
    fallback: bool = False  # True: el estimador pedido no dio varianzas positivas
    requested: str | None = None  # núcleo del estimador pedido, si se recurrió a otro

    @property
    def t(self) -> np.ndarray:
        return self.beta / self.se

    @property
    def p(self) -> np.ndarray:
        return 2 * st.norm.sf(np.abs(self.t))

    @property
    def estimator(self) -> str:
        """Estimador usado, p. ej. «HH 20» o «NW 42 (recurso: HH no positiva)»."""
        lab = hac_label(self.lags, self.kernel) if self.lags is not None else ""
        if not self.fallback:
            return lab
        return f"{lab} (recurso: {KERNELS[self.requested or 'uniform']} no positiva)"

    def frame(self) -> pd.DataFrame:
        """Coeficientes con columnas fijas (coef, se, t, p) y el estimador de cada
        fila, de modo que una tabla que concatena varios ajustes dice cuál se usó."""
        return pd.DataFrame({"coef": self.beta, "se": self.se, "t": self.t, "p": self.p,
                             "estimador": self.estimator}, index=self.names)


def _xy(y: pd.Series, X: pd.DataFrame, const: bool = True):
    df = pd.concat([y.rename("__y"), X], axis=1).dropna()
    Xv = df.drop(columns="__y")
    if const:
        Xv.insert(0, "const", 1.0)
    return df["__y"].to_numpy(float), Xv.to_numpy(float), list(Xv.columns)


def hac_cov(X: np.ndarray, u: np.ndarray, lags: int, kernel: str = "bartlett") -> np.ndarray:
    """Covarianza HAC de los coeficientes MCO.

    kernel «bartlett»: Newey-West (pesos 1 − l/(lags + 1)); siempre semidefinida
    positiva, pero con lags = h infravalora la varianza de largo plazo de un
    MA(h − 1) (≈ 68% con h = 21). kernel «uniform»: Hansen-Hodrick (peso 1 hasta
    `lags`), el estimador natural para objetivos solapados de h días con
    lags = h − 1: recoge toda la varianza de largo plazo de un MA(h − 1), pero no
    garantiza una matriz semidefinida positiva."""
    if kernel not in KERNELS:
        raise ValueError(f"núcleo desconocido: {kernel}")
    X, u = np.asarray(X, float), np.asarray(u, float)
    n, k = X.shape
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1) if kernel == "bartlett" else 1.0
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    return XtX_inv @ S @ XtX_inv * n / (n - k)


def newey_west_cov(X: np.ndarray, u: np.ndarray, lags: int) -> np.ndarray:
    return hac_cov(X, u, lags, "bartlett")


def hac_overlap(h: int) -> dict:
    """Convención de la serie para objetivos solapados de h sesiones (Clark-West y
    t de MCO): Hansen-Hodrick con h − 1 retardos y, si alguna varianza no sale
    positiva, Newey-West con 2h retardos. Se pasa como argumentos con nombre:
    stats.ols_hac(y, X, **stats.hac_overlap(h))."""
    return {"lags": h - 1, "kernel": "uniform", "fallback": (2 * h, "bartlett")}


# Retardos de Newey-West en las submuestras mensuales no solapadas (nonoverlap con
# step = 21): 12, un año de observaciones mensuales. Bajo la nula esas
# observaciones apenas están autocorrelacionadas; los 12 retardos son un margen
# de seguridad. Es la convención de R16 y de la Clark-West mensual de R12.
NONOV_LAGS = 12


def ols_hac(y: pd.Series, X: pd.DataFrame, lags: int, const: bool = True,
            kernel: str = "bartlett", fallback: tuple[int, str] | None = None) -> OLS:
    """OLS con errores HAC (por defecto Newey-West, núcleo de Bartlett, con `lags`).

    Para objetivos de h días que se solapan, Bartlett con lags = h infravalora la
    varianza de largo plazo; lo adecuado es Hansen-Hodrick (kernel «uniform») con
    lags = h − 1 (ver hac_overlap). Como Hansen-Hodrick no garantiza varianzas
    positivas, `fallback` = (retardos, núcleo) indica con qué recalcular si alguna
    sale ≤ 0; sin recurso (por defecto), el error de esa variable queda en NaN.
    El estimador usado queda en OLS.estimator y OLS.fallback."""
    yv, Xv, names = _xy(y, X, const)
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    u = yv - Xv @ beta
    var = np.diag(hac_cov(Xv, u, lags, kernel))
    used = (lags, kernel, False, None)
    if fallback is not None and (var <= 0).any():
        var = np.diag(hac_cov(Xv, u, *fallback))
        used = (*fallback, True, kernel)
    r2 = 1 - (u @ u) / ((yv - yv.mean()) @ (yv - yv.mean()))
    se = np.sqrt(np.where(var > 0, var, np.nan))
    return OLS(names, beta, se, float(r2), len(yv), *used)


def nonoverlap(y: pd.Series, X: pd.DataFrame | None, var: str, step: int,
               lags: int = NONOV_LAGS, kernel: str = "bartlett") -> pd.DataFrame:
    """La misma regresión con una observación de cada `step` (objetivos de `step`
    días que no se solapan), para cada uno de los `step` desfases posibles:
    coeficiente de `var` y su t HAC (por defecto Newey-West con NONOV_LAGS).
    Con X = None, regresión sobre la constante (var = «const»), p. ej. el
    diferencial ajustado de Clark-West: stats.nonoverlap(clark_west_diff(...),
    None, "const", 21).

    La submuestra es una de cada `step` FILAS de y (un array se trata como una
    serie sin fechas). Si y omite sesiones, como la salida de clark_west_diff
    cuando hay huecos dentro de la muestra, conviene reindexarla antes sobre el
    calendario de sesiones para que el paso sea de `step` sesiones; las filas
    con NaN se descartan en cada submuestra."""
    if not isinstance(y, pd.Series):
        y = pd.Series(np.asarray(y, float), index=None if X is None else X.index)
    X = pd.DataFrame(index=y.index) if X is None else X
    rows = {}
    for o in range(step):
        fit = ols_hac(y.iloc[o::step], X.iloc[o::step], lags, kernel=kernel)
        j = fit.names.index(var)
        rows[o] = {"coef": fit.beta[j], "t": fit.t[j], "n": fit.n}
    return pd.DataFrame(rows).T


# ───────────────────────────── bootstrap ─────────────────────────────

def stationary_indices(n: int, mean_block: float, rng: np.random.Generator,
                       size: int | None = None) -> np.ndarray:
    """Índices de un remuestreo estacionario (bloques de longitud geométrica
    de media `mean_block`, que continúan circularmente)."""
    size = size or n
    new_block = rng.random(size) < 1.0 / mean_block
    new_block[0] = True
    pos = np.arange(size)
    block_start_pos = np.maximum.accumulate(np.where(new_block, pos, 0))
    start_val = rng.integers(0, n, size)[block_start_pos]
    return (start_val + pos - block_start_pos) % n


def bootstrap_ci(x: np.ndarray, stat, mean_block: float, B: int, rng,
                 alpha: float = 0.10) -> tuple[float, float, float]:
    """Estimación puntual e intervalo percentil (1−alpha) por bootstrap estacionario.

    `x` puede ser 1-D o 2-D (filas = tiempo); `stat` recibe el remuestreo."""
    x = np.asarray(x)
    n = len(x)
    draws = np.array([stat(x[stationary_indices(n, mean_block, rng)]) for _ in range(B)])
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(x)), float(lo), float(hi)


# ───────────────────────────── predicción fuera de muestra ─────────────────────────────

def expanding_forecast(y: pd.Series, X: pd.DataFrame, h: int, min_obs: int,
                       refit: int = 21) -> pd.DataFrame:
    """Previsión en tiempo real de y_t (un objetivo que se conoce en t+h).

    En la fecha t sólo se usan pares (X_s, y_s) con s ≤ t − h, que son los que
    ya se han observado. Devuelve la previsión del modelo y la de la media
    histórica (el benchmark de Campbell y Thompson)."""
    df = pd.concat([y.rename("y"), X], axis=1)
    Xa = np.column_stack([np.ones(len(df)), df.drop(columns="y").to_numpy(float)])
    ya = df["y"].to_numpy(float)
    ok = ~np.isnan(ya) & ~np.isnan(Xa).any(axis=1)
    f_model = np.full(len(df), np.nan)
    f_mean = np.full(len(df), np.nan)
    beta = None
    for t in range(len(df)):
        known = ok[: max(t - h + 1, 0)]
        if known.sum() < min_obs or np.isnan(Xa[t]).any():
            continue
        if beta is None or t % refit == 0:
            Xk, yk = Xa[: len(known)][known], ya[: len(known)][known]
            beta, *_ = np.linalg.lstsq(Xk, yk, rcond=None)
            mu = yk.mean()
        f_model[t] = Xa[t] @ beta
        f_mean[t] = mu
    return pd.DataFrame({"y": ya, "modelo": f_model, "media": f_mean}, index=df.index)


def oos_r2(y: np.ndarray, f_model: np.ndarray, f_bench: np.ndarray) -> float:
    m = ~np.isnan(y) & ~np.isnan(f_model) & ~np.isnan(f_bench)
    return float(1 - ((y[m] - f_model[m]) ** 2).sum() / ((y[m] - f_bench[m]) ** 2).sum())


def clark_west_diff(y, f_bench, f_model):
    """Diferencial MSPE-ajustado de Clark y West (2007) en las fechas con los tres datos:
    (y − f₁)² − [(y − f₂)² − (f₁ − f₂)²], con f₁ el benchmark anidado y f₂ el modelo.

    Las tres entradas se alinean por posición y se omiten las fechas en que falta
    alguna. Si y es una serie, devuelve una serie con sus fechas (la que necesita
    nonoverlap para la submuestra mensual: stats.nonoverlap(clark_west_diff(m.y,
    m.b, m.f), None, "const", 21)); si es un array, un array sin índice."""
    idx = y.index if isinstance(y, pd.Series) else None
    y, f1, f2 = (np.asarray(a, float) for a in (y, f_bench, f_model))
    m = ~np.isnan(y) & ~np.isnan(f2) & ~np.isnan(f1)
    y, f1, f2 = y[m], f1[m], f2[m]
    d = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    return d if idx is None else pd.Series(d, index=idx[m])


def clark_west_fit(y: np.ndarray, f_bench: np.ndarray, f_model: np.ndarray, lags: int,
                   kernel: str = "bartlett", fallback: tuple[int, str] | None = None) -> OLS:
    """Media del diferencial de Clark-West con su error HAC (regresión sobre la
    constante). fit.t[0] es el t; fit.estimator, el estimador usado."""
    f = pd.Series(np.asarray(clark_west_diff(y, f_bench, f_model)))
    return ols_hac(f, pd.DataFrame(index=f.index), lags, kernel=kernel, fallback=fallback)


def clark_west(y: np.ndarray, f_bench: np.ndarray, f_model: np.ndarray,
               lags: int, kernel: str = "bartlett",
               fallback: tuple[int, str] | None = None) -> tuple[float, float]:
    """Test MSPE-ajustado (H0: el modelo grande no mejora al anidado). t HAC y p unilateral.

    Por defecto, Newey-West con `lags`. Con objetivos solapados de h días, la
    convención de la serie es **hac_overlap(h) (Hansen-Hodrick con h − 1 retardos
    y recurso a Newey-West 2h); para saber qué estimador se usó, clark_west_fit."""
    fit = clark_west_fit(y, f_bench, f_model, lags, kernel, fallback)
    t = float(fit.t[0])
    return t, float(st.norm.sf(t))


# ───────────────────────────── pruebas múltiples ─────────────────────────────

def reality_check(d: np.ndarray, mean_block: float, B: int, rng) -> dict:
    """White (2000). d: T×K diferenciales de rendimiento (regla k − benchmark).

    H0: ninguna regla supera al benchmark en media. Devuelve el p-valor del
    máximo y el p-valor nominal de la mejor regla (sin corregir)."""
    d = np.asarray(d, float)
    T, K = d.shape
    dbar = d.mean(axis=0)
    V = np.sqrt(T) * dbar.max()
    best = int(dbar.argmax())
    # Con los mismos remuestreos: el máximo centrado (RC) y la media centrada de
    # la mejor regla por separado (su p-valor nominal, sin corregir por buscarla).
    Vb = np.empty(B)
    vb_best = np.empty(B)
    for b in range(B):
        centred = d[stationary_indices(T, mean_block, rng)].mean(axis=0) - dbar
        Vb[b] = np.sqrt(T) * centred.max()
        vb_best[b] = np.sqrt(T) * centred[best]
    return {"p_RC": float((Vb >= V).mean()), "mejor": best,
            "p_nominal_mejor": float((vb_best >= V).mean()),
            "media_diaria": dbar}


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)[::-1]
    q = np.empty(m)
    run = 1.0
    for rank, i in zip(range(m, 0, -1), order):
        run = min(run, p[i] * m / rank)
        q[i] = run
    return q


def spa_test(d: np.ndarray, mean_block: float, B: int, rng) -> dict:
    """Test de Habilidad Predictiva Superior de Hansen (2005), versiones
    consistente (c) e inferior (l). d: T×K diferenciales (regla − benchmark).

    Frente al Reality Check, estudentiza cada regla y recentra las que son
    claramente peores que el benchmark (t ≤ −√(2·log log T)), de modo que el test
    es menos sensible a las reglas malas (Hansen 2005): las que quedan por encima
    de ese umbral siguen restándole potencia."""
    d = np.asarray(d, float)
    T, K = d.shape
    dbar = d.mean(axis=0)
    draws = np.empty((B, K))
    for b in range(B):
        draws[b] = d[stationary_indices(T, mean_block, rng)].mean(axis=0)
    omega = np.sqrt(T) * draws.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.inf)
    t_obs = max((np.sqrt(T) * dbar / omega).max(), 0.0)
    # Recentrado bajo H0 (Hansen 2005, ec. 11): sólo las reglas significativamente
    # peores que el benchmark conservan su media negativa; el resto se lleva a 0.
    thr = -omega * np.sqrt(2 * np.log(np.log(T))) / np.sqrt(T)
    mu_c = np.where(dbar <= thr, dbar, 0.0)
    mu_l = np.minimum(dbar, 0.0)
    out = {}
    for name, mu in (("p_SPA_c", mu_c), ("p_SPA_l", mu_l)):
        z = draws - dbar + mu
        t_b = np.maximum((np.sqrt(T) * z / omega).max(axis=1), 0.0)
        out[name] = float((t_b >= t_obs).mean()) if t_obs > 0 else 1.0
    out["t_SPA"] = float(t_obs)
    return out


def model_confidence_set(losses: pd.DataFrame, mean_block: float, B: int, rng,
                         alpha: float = 0.10) -> pd.DataFrame:
    """Model Confidence Set de Hansen, Lunde y Nason (2011), estadístico T_max.

    Elimina iterativamente el peor modelo mientras se rechace la igualdad de
    capacidad predictiva. Devuelve la pérdida media, el p-valor MCS de cada
    modelo y si pertenece al conjunto al nivel 1 − alpha."""
    L = losses.dropna()
    names = list(L.columns)
    X = L.to_numpy(float)
    T = len(X)
    idx_draws = [stationary_indices(T, mean_block, rng) for _ in range(B)]
    alive = list(range(len(names)))
    pvals = {}
    running = 0.0
    while len(alive) > 1:
        sub = X[:, alive]
        di = sub - sub.mean(axis=1, keepdims=True)  # d_i. = L_i − media de los vivos
        dbar = di.mean(axis=0)
        boot = np.array([di[ix].mean(axis=0) for ix in idx_draws])
        se = boot.std(axis=0, ddof=1)
        t = dbar / se
        tb = ((boot - dbar) / se).max(axis=1)
        p = float((tb >= t.max()).mean())
        worst = alive[int(t.argmax())]
        running = max(running, p)
        pvals[names[worst]] = running
        alive.remove(worst)
    pvals[names[alive[0]]] = 1.0
    out = pd.DataFrame({"pérdida_media": L.mean(), "p_MCS": pd.Series(pvals)})
    out["en_MCS"] = out.p_MCS >= alpha
    return out.sort_values("pérdida_media")


def qlike(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Pérdida QLIKE (robusta al ruido del proxy de varianza, Patton 2011)."""
    ratio = realized / forecast
    return ratio - np.log(ratio) - 1


# ───────────────────────────── volatilidad ─────────────────────────────

def _garch_recursion(r, omega, a, b, h0):
    """h_t = ω + α r_{t−1}² + β h_{t−1}, h_0 dado (vectorizado con un filtro IIR)."""
    x = np.r_[h0, omega + a * r[:-1] ** 2]
    return lfilter([1.0], [1.0, -b], x, zi=[0.0])[0]


def garch11_fit(r: np.ndarray) -> dict:
    """GARCH(1,1) gaussiano por máxima verosimilitud (r en fracción diaria, media cero)."""
    from scipy.optimize import minimize

    r = np.asarray(r, float)
    r = r[~np.isnan(r)] - np.nanmean(r)
    v0 = r.var()

    def nll(theta):
        omega, a, b = np.exp(theta[0]), 1 / (1 + np.exp(-theta[1])), 1 / (1 + np.exp(-theta[2]))
        if a + b >= 0.9999:
            return 1e10
        h = _garch_recursion(r, omega, a, b, v0)
        return 0.5 * np.sum(np.log(h) + r ** 2 / h)

    x0 = [np.log(v0 * 0.05), np.log(0.08 / 0.92), np.log(0.9 / 0.1)]
    res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-6})
    omega, a, b = np.exp(res.x[0]), 1 / (1 + np.exp(-res.x[1])), 1 / (1 + np.exp(-res.x[2]))
    return {"omega": omega, "alpha": a, "beta": b, "mean": 0.0}


def garch11_filter(r: np.ndarray, p: dict) -> np.ndarray:
    """Varianza condicional h_{t+1} conocida al cierre de t."""
    r = np.nan_to_num(np.asarray(r, float))
    lr = p["omega"] / max(1 - p["alpha"] - p["beta"], 1e-6)
    # h_{t+1} = ω + α r_t² + β h_t, con h_0 en la varianza de largo plazo
    return lfilter([1.0], [1.0, -p["beta"]], p["omega"] + p["alpha"] * r ** 2,
                   zi=[p["beta"] * lr])[0]


def garch11_horizon(h_next: np.ndarray, p: dict, horizon: int) -> np.ndarray:
    """Varianza diaria media prevista en los próximos `horizon` días."""
    persist = p["alpha"] + p["beta"]
    lr = p["omega"] / max(1 - persist, 1e-6)
    k = np.arange(horizon)
    w = (persist ** k).mean()
    return lr + w * (h_next - lr)


# ───────────────────────────── cambios de régimen ─────────────────────────────

def hmm_fit(x: np.ndarray, K: int = 2, iters: int = 500, tol: float = 1e-8) -> dict:
    """HMM gaussiano (Hamilton 1989) por EM / Baum-Welch con escalado.

    Los estados se ordenan por volatilidad creciente (0 = calma)."""
    x = np.asarray(x, float)
    T = len(x)
    qs = np.quantile(np.abs(x - x.mean()), np.linspace(0.3, 0.9, K))
    mu = np.full(K, x.mean())
    sd = x.std() * qs / qs.mean()
    P = np.full((K, K), 0.02 / (K - 1))
    np.fill_diagonal(P, 0.98)
    pi = np.full(K, 1 / K)
    ll_old = -np.inf
    for _ in range(iters):
        dens = np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        dens = np.maximum(dens, 1e-300)
        a = np.empty((T, K))
        c = np.empty(T)
        a[0] = pi * dens[0]
        c[0] = a[0].sum()
        a[0] /= c[0]
        for t in range(1, T):
            a[t] = (a[t - 1] @ P) * dens[t]
            c[t] = a[t].sum()
            a[t] /= c[t]
        bta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            bta[t] = P @ (dens[t + 1] * bta[t + 1]) / c[t + 1]
        g = a * bta
        g /= g.sum(axis=1, keepdims=True)
        xi = (a[:-1, :, None] * P[None] * (dens[1:] * bta[1:])[:, None, :]
              / c[1:, None, None])
        pi = g[0]
        P = xi.sum(axis=0) / g[:-1].sum(axis=0)[:, None]
        P /= P.sum(axis=1, keepdims=True)
        w = g.sum(axis=0)
        mu = (g * x[:, None]).sum(axis=0) / w
        sd = np.sqrt((g * (x[:, None] - mu) ** 2).sum(axis=0) / w)
        ll = np.log(c).sum()
        if ll - ll_old < tol:
            break
        ll_old = ll
    order = np.argsort(sd)
    return {"mu": mu[order], "sd": sd[order], "P": P[np.ix_(order, order)],
            "pi": pi[order], "loglik": float(ll), "smoothed": g[:, order]}


def hmm_filter(x: np.ndarray, p: dict) -> np.ndarray:
    """Probabilidades de régimen para el periodo siguiente con información hasta t."""
    x = np.asarray(x, float)
    K = len(p["mu"])
    prob = p["pi"].copy()
    out = np.empty((len(x), K))
    for t, v in enumerate(x):
        if np.isnan(v):
            out[t] = prob @ p["P"]
            continue
        dens = np.exp(-0.5 * ((v - p["mu"]) / p["sd"]) ** 2) / p["sd"]
        post = prob * np.maximum(dens, 1e-300)
        post /= post.sum()
        prob = post @ p["P"]
        out[t] = prob
    return out


# ───────────────────────────── colas ─────────────────────────────

def gpd_tail(losses: np.ndarray, q: float = 0.95) -> dict:
    """Picos sobre umbral: ajuste GPD a las pérdidas que superan su cuantil q."""
    from scipy.stats import genpareto

    losses = np.asarray(losses, float)
    losses = losses[~np.isnan(losses)]
    u = np.quantile(losses, q)
    exc = losses[losses > u] - u
    xi, _, beta = genpareto.fit(exc, floc=0)
    return {"u": float(u), "xi": float(xi), "beta": float(beta),
            "zeta": len(exc) / len(losses), "n_exc": len(exc), "n": len(losses)}


def gpd_exceedance_prob(fit: dict, x: float) -> float:
    """P(pérdida diaria > x) según la cola GPD ajustada."""
    if x <= fit["u"]:
        raise ValueError("x debe estar por encima del umbral")
    z = 1 + fit["xi"] * (x - fit["u"]) / fit["beta"]
    if z <= 0:
        return 0.0
    return fit["zeta"] * z ** (-1 / fit["xi"]) if fit["xi"] != 0 else \
        fit["zeta"] * np.exp(-(x - fit["u"]) / fit["beta"])


def hill(losses: np.ndarray, k: int) -> float:
    """Estimador de Hill del índice de cola ξ con las k mayores pérdidas."""
    s = np.sort(np.asarray(losses, float)[~np.isnan(losses)])[::-1]
    s = s[s > 0]
    return float(np.mean(np.log(s[:k])) - np.log(s[k]))


# ───────────────────────────── familia GARCH ─────────────────────────────

def garch_fit(r: np.ndarray, kind: str = "garch") -> dict:
    """ARCH(1), GARCH(1,1) o GJR-GARCH(1,1) gaussianos por máxima verosimilitud.

    GJR (Glosten, Jagannathan y Runkle 1993) añade γ·1{r<0}·r²: las caídas
    elevan más la varianza que las subidas (efecto apalancamiento)."""
    from scipy.optimize import minimize

    r = np.asarray(r, float)
    r = r[~np.isnan(r)] - np.nanmean(r)
    v0 = r.var()
    neg = (r < 0).astype(float)
    sig = lambda z: 1 / (1 + np.exp(-z))

    def unpack(th):
        omega, a = np.exp(th[0]), sig(th[1])
        b = sig(th[2]) if kind != "arch" else 0.0
        g = sig(th[3]) if kind == "gjr" else 0.0
        return omega, a, b, g

    def nll(th):
        omega, a, b, g = unpack(th)
        if a + g / 2 + b >= 0.9999:
            return 1e10
        x = np.r_[v0, omega + (a + g * neg[:-1]) * r[:-1] ** 2]
        h = lfilter([1.0], [1.0, -b], x, zi=[0.0])[0]
        return 0.5 * np.sum(np.log(h) + r ** 2 / h)

    x0 = [np.log(v0 * 0.05), np.log(0.08 / 0.92), np.log(0.9 / 0.1), np.log(0.05 / 0.95)]
    x0 = x0[:{"arch": 2, "garch": 3, "gjr": 4}[kind]]
    if kind == "arch":
        x0 = [np.log(v0 * 0.5), 0.0]
    res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6})
    omega, a, b, g = unpack(res.x)
    return {"kind": kind, "omega": omega, "alpha": a, "beta": b, "gamma": g}


def garch_filter(r: np.ndarray, p: dict) -> np.ndarray:
    """h_{t+1} conocida al cierre de t, para cualquier modelo de garch_fit."""
    r = np.nan_to_num(np.asarray(r, float))
    persist = p["alpha"] + p.get("gamma", 0.0) / 2 + p["beta"]
    lr = p["omega"] / max(1 - persist, 1e-6)
    shock = (p["alpha"] + p.get("gamma", 0.0) * (r < 0)) * r ** 2
    return lfilter([1.0], [1.0, -p["beta"]], p["omega"] + shock, zi=[p["beta"] * lr])[0]


def garch_horizon_var(h_next: np.ndarray, p: dict, horizon: int) -> np.ndarray:
    """Varianza diaria media prevista en los próximos `horizon` días."""
    persist = p["alpha"] + p.get("gamma", 0.0) / 2 + p["beta"]
    lr = p["omega"] / max(1 - persist, 1e-6)
    w = (persist ** np.arange(horizon)).mean()
    return lr + w * (h_next - lr)
