"""R16 — Mecanismo: de dónde sale la ventaja de crecimiento de la exposición variable.

La LRS 3x original (la regla de Gayed y Bilello 2016: media de 200 sesiones
sobre el índice de rentabilidad total), la LRS-Kelly y el medio Kelly con GJR
mueven su exposición L_t al índice entre 0 y 3. Cada una se compara con la cartera que mantiene CONSTANTE
su misma exposición media L̄ (un 3x con peso L̄/3 y el resto en T-bills,
reajustado a diario) y la diferencia de crecimiento logarítmico Δg se reparte
en cuatro términos exactos de una expansión de segundo orden más un resto:
timing de la media, timing de la varianza, coste de dispersión y financiación
y costes. Después se pregunta si el reparto es estable (subperiodos, ventana
móvil de 10 años, jackknife de episodios, ruptura estructural de Andrews), si
la LRS gana sólo por esquivar volatilidad (placebo condicional), si amplía la
frontera media-varianza (spanning a lo Moreira–Muir) y si el estado de la media
de 200 sesiones anticipa la varianza más allá del GJR. Como sensibilidad, la
misma descomposición de la LRS 3x con la señal de precio (R13). Los p de
titular se exportan, con su procedencia, a p_titular.csv (los lee R19).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps

from letf_lab import provenance, stats
from letf_lab.report import FIG_W, Report, comma_axis, legend_band, num_es
from letf_lab.research import r12_long_ml as r12
from letf_lab.research import r13_lrs as r13

SEED = 16
ANN = 252
LAG = r13.EXEC_LAG          # señal al cierre de t, orden al cierre de t+1
H = r12.H
START = "1938"              # primer año con las tres estrategias (10 años de calentamiento)
STRATS = ("LRS 3x (original)", "LRS-Kelly", "½ Kelly GJR (sin tendencia)")
PX = r13.PX_LRS             # sensibilidad: la LRS 3x con la señal de precio
PERIODS = {"1938–2026": ("1938", None), "1938–1969": ("1938", "1969"),
           "1970–1991": ("1970", "1991"), "1992–2026": ("1992", None),
           "ETFs reales (2010–2026)": (r13.REAL_FROM, None)}
# Años naturales completos: el episodio incluye su rebote dentro del año.
EPISODES = {"1929–32": ("1929", "1932"), "1973–74": ("1973", "1974"), "1987": ("1987", "1987"),
            "2000–02": ("2000", "2002"), "2008–09": ("2008", "2009"), "2020": ("2020", "2020"),
            "2022": ("2022", "2022")}
ROLL, ROLL_STEP = 2520, 21  # ventana de 10 años, recalculada cada 21 sesiones
N_PERM = 2000               # permutaciones por esquema de placebo
N_DEC = 10                  # deciles de la varianza prevista por el GJR
NW_DAILY = 2 * H            # retardos Newey-West con objetivos solapados de 21 días
NW_MONTHLY = 6              # retardos Newey-West con datos mensuales
BOOT_BLOCKS = (12, 36)      # bloques medios (meses) del bootstrap del p de los términos
B_BOOT_TERM = 4000          # remuestreos de ese bootstrap
SEED_BOOT = 1600            # semilla propia (no mueve el resto del informe)
N_TITULAR = 40              # contrastes principales de R19 (Holm)
TRIM = 0.15                 # recorte de Andrews (1993): rupturas en [15%, 85%] de la muestra
N_NULL = 20_000             # simulaciones de la distribución asintótica del sup-Wald
# Añadidos tras la revisión externa (no forman parte del protocolo original):
HH_LAGS = H - 1             # Hansen-Hodrick: kernel uniforme con h − 1 retardos (objetivos de 21 días)
NW_LONG = ANN               # Bartlett con un año de retardos (residuos con memoria larga)
NW_NONOV = stats.NONOV_LAGS  # Bartlett con 12 retardos (un año) en muestras mensuales no solapadas
TRIM_N = (10, 20, 50, 100, 250)  # rejilla completa: sesiones de mayor |e| que se quitan
TOP_N = (10, 20)            # concentración: los N mejores y peores días de cada término
SMALL_CELL = ANN            # celdas con menos de un año de días (≈ 12 meses independientes)
RHO_LAGS = (H, 3 * H, 6 * H, ANN)  # autocorrelación de los residuos de las regresiones a 21 días

TERMS = ["timing_media", "timing_varianza", "dispersión", "financiación_y_costes", "resto"]
LABEL = {"timing_media": "(1) timing de la media", "timing_varianza": "(2) timing de la varianza",
         "dispersión": "(3) coste de dispersión", "financiación_y_costes": "(4) financiación y costes",
         "resto": "resto (orden ≥ 3)"}
# Paleta categórica validada (orden fijo; ver skill dataviz): azul, naranja, aguamarina, ámbar.
COL = {"timing_media": "#2a78d6", "timing_varianza": "#eb6834", "dispersión": "#1baf7a",
       "financiación_y_costes": "#eda100", "resto": "#9a9994"}
INK, INK2 = "#0b0b0b", "#52514e"
FIG_NAME = {"½ Kelly GJR (sin tendencia)": "Medio Kelly (sin tendencia)"}   # nombre del artículo en las figuras


# ───────────────────────────── matemáticas ─────────────────────────────

def decompose(r, L, e, rf, kap, ann: int = ANN) -> dict:
    """Δg de una exposición variable L_t frente a la exposición constante L̄ = media(L).

    r: rentabilidad de la estrategia; L: exposición aplicada ese día; e = u − rf:
    exceso del índice; kap: coste por unidad de exposición del instrumento
    (financiación, diferencial del swap, comisión, error de seguimiento), de
    modo que la referencia rinde rc = rf + L̄·(e − kap).

    Δg = Cov(L, e) − ½·Cov(L², e²) − ½·Var(L)·E[e²] − E[k − L̄·kap] + resto,
    con k = rf + L·e − r (todo lo que la estrategia paga) y momentos muestrales
    con divisor N, de modo que (1) es exactamente E[(L − L̄)·e]. El resto
    recoge los términos de orden ≥ 3 de log(1 + x)."""
    r, L, e, rf, kap = (np.asarray(a, float) for a in (r, L, e, rf, kap))
    Lb = L.mean()
    rc = rf + Lb * (e - kap)
    gs, gc = np.log1p(r).mean(), np.log1p(rc).mean()
    e2, L2 = e ** 2, L ** 2
    t1 = np.mean((L - Lb) * e)
    t2 = -0.5 * np.mean((L2 - L2.mean()) * e2)
    t3 = -0.5 * L.var() * e2.mean()
    k = rf + L * e - r
    t4 = -np.mean(k - Lb * kap)
    s = t1 + t2 + t3 + t4
    return {"L̄": Lb, "g_estrategia": gs * ann, "g_constante": gc * ann, "Δg": (gs - gc) * ann,
            "timing_media": t1 * ann, "timing_varianza": t2 * ann, "dispersión": t3 * ann,
            "financiación_y_costes": t4 * ann, "resto": (gs - gc - s) * ann, "suma_1_4": s * ann,
            "financiación": -np.mean((L - Lb) * kap) * ann, "rotación": -np.mean(k - L * kap) * ann}


def permute_within(values: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    """Permutación aleatoria de `values` que sólo mezcla posiciones del mismo grupo."""
    order = np.argsort(groups, kind="stable")
    shuffled = np.lexsort((rng.random(len(values)), groups))
    out = np.empty_like(values)
    out[order] = values[shuffled]
    return out


def sup_wald(y: np.ndarray, trim: float = TRIM, lags: int = NW_MONTHLY) -> dict:
    """Andrews (1993): sup-Wald de un cambio de media en fecha desconocida.

    Para cada ruptura candidata k en [trim·n, (1 − trim)·n] se estima
    y_t = a + b·1{t ≥ k} + ε_t y W_k = b̂² / Var_NW(b̂) con los residuos de ese
    modelo (errores Newey-West, kernel de Bartlett)."""
    y = np.asarray(y, float)
    n = len(y)
    lo, hi = int(np.ceil(trim * n)), int(np.floor((1 - trim) * n))
    path, diff, se = (np.full(n, np.nan) for _ in range(3))
    for k in range(lo, hi + 1):
        X = np.column_stack([np.ones(n), (np.arange(n) >= k).astype(float)])
        beta = np.array([y[:k].mean(), y[k:].mean() - y[:k].mean()])
        resid = y - X @ beta
        cov = stats.newey_west_cov(X, resid, lags)
        diff[k], se[k] = beta[1], np.sqrt(cov[1, 1])
        path[k] = beta[1] ** 2 / cov[1, 1]
    k = int(np.nanargmax(path))
    return {"stat": float(path[k]), "k": k, "path": path, "diff": diff, "se": se,
            "media_antes": float(y[:k].mean()), "media_después": float(y[k:].mean())}


def sup_wald_null(trim: float = TRIM, n_sim: int = N_NULL, n_steps: int = 1000, rng=None,
                  chunk: int = 1000) -> np.ndarray:
    """Distribución asintótica del sup-Wald con un parámetro (p = 1):
    sup_π [B(π) − π·B(1)]² / (π(1 − π)), π ∈ [trim, 1 − trim], B movimiento browniano."""
    rng = rng or np.random.default_rng(0)
    pi = np.arange(1, n_steps + 1) / n_steps
    keep = (pi >= trim) & (pi <= 1 - trim)
    out = []
    for s in range(0, n_sim, chunk):
        W = np.cumsum(rng.standard_normal((min(chunk, n_sim - s), n_steps)), axis=1) / np.sqrt(n_steps)
        BB = W - pi * W[:, -1:]
        out.append((BB[:, keep] ** 2 / (pi[keep] * (1 - pi[keep]))).max(axis=1))
    return np.concatenate(out)


# Covarianza HAC compartida (stats.hac_cov): «bartlett» = Newey-West, «uniform» =
# Hansen-Hodrick. Se conserva el nombre para los llamantes y las pruebas de R16.
hac_cov = stats.hac_cov


# La columna Hansen-Hodrick sigue la convención de la serie para objetivos solapados de H
# sesiones (stats.hac_overlap): h − 1 retardos y, si alguna varianza no sale positiva,
# Newey-West con 2h retardos (el estimador usado queda en fit["estimador"]).
_HH = stats.hac_overlap(H)
ERRORS = {"t NW 42 (versión anterior)": (NW_DAILY, "bartlett"),
          "t HH 20": (_HH["lags"], _HH["kernel"], _HH["fallback"]),
          "t NW 252": (NW_LONG, "bartlett")}


def ols_errors(y: pd.Series, X: pd.DataFrame, errors: dict = ERRORS) -> dict:
    """MCO con constante y el t de cada coeficiente con varios estimadores HAC.

    Cada estimador es (retardos, núcleo) o (retardos, núcleo, recurso): con recurso, si
    alguna varianza no sale positiva (Hansen-Hodrick no lo garantiza) se recalculan todas
    con el recurso = (retardos, núcleo), como stats.ols_hac. fit["estimador"][rótulo] dice
    cuál se usó («HH 20» o «NW 42 (recurso: HH no positiva)»); sin recurso, el t de una
    varianza no positiva queda en NaN."""
    names = ["const", *X.columns]
    Xv = np.column_stack([np.ones(len(X)), X.values.astype(float)])
    yv = np.asarray(y, float)
    beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
    resid = yv - Xv @ beta
    r2 = 1 - resid @ resid / ((yv - yv.mean()) @ (yv - yv.mean()))
    t, used = {}, {}
    for lab, (lags, kern, *fb) in errors.items():
        var = np.diag(hac_cov(Xv, resid, lags, kern))
        used[lab] = stats.hac_label(lags, kern)
        if fb and fb[0] is not None and (var <= 0).any():
            var = np.diag(hac_cov(Xv, resid, *fb[0]))
            used[lab] = f"{stats.hac_label(*fb[0])} (recurso: {stats.KERNELS[kern]} no positiva)"
        t[lab] = np.where(var > 0, beta / np.sqrt(np.abs(var)), np.nan)
    return {"names": names, "beta": beta, "t": t, "r2": float(r2), "n": len(yv), "resid": resid,
            "estimador": used}


def nonoverlap(y: pd.Series, X: pd.DataFrame, var: str, step: int = H,
               lags: int = NW_NONOV) -> pd.DataFrame:
    """La misma regresión con una observación de cada `step` (objetivos de `step`
    días que no se solapan), para cada uno de los `step` desfases posibles:
    coeficiente de `var` y su t de Newey-West (stats.nonoverlap)."""
    return stats.nonoverlap(y, X, var, step, lags)


def drop_extremes(P: pd.DataFrame, n: int) -> pd.DataFrame:
    """El panel sin las n sesiones de mayor |e| (caídas y subidas por igual)."""
    if n == 0:
        return P
    return P.drop(P.e.abs().nlargest(n).index)


def concentration(c: pd.Series, ks=TOP_N) -> dict:
    """Cuánto de la suma de unas contribuciones diarias aportan sus k mayores y
    sus k menores valores."""
    out = {"total": float(c.sum())}
    for k in ks:
        out[f"{k} mejores"] = float(c.nlargest(k).sum())
        out[f"{k} peores"] = float(c.nsmallest(k).sum())
    return out


# ───────────────────────────── datos ─────────────────────────────

def kappa(d: dict) -> pd.Series:
    """Coste diario por unidad de exposición del 3x: (rf + 3·e − x3)/3.

    En el sintético es (f − rf) + (3·s + ER)/(3·252); con el ETF real incluye
    además su error de seguimiento."""
    e = d["u"] - d["rf"]
    return (d["rf"] + 3 * e - d["x3"]) / 3


def panel(d: dict, R: dict, E: dict, name: str) -> pd.DataFrame:
    """Rentabilidad, exposición APLICADA (decidida LAG sesiones antes) e insumos."""
    e = d["u"] - d["rf"]
    df = pd.concat([R[name], E[name].shift(LAG), e, d["rf"], kappa(d)], axis=1,
                   keys=["r", "L", "e", "rf", "kap"])
    return df.dropna()


def _cols(P: pd.DataFrame):
    return (P[c].values for c in ("r", "L", "e", "rf", "kap"))


def _between(P: pd.DataFrame, a, b):
    return P.loc[a:(f"{b}-12-31" if b and len(str(b)) == 4 else b)]


# ───────────────────────────── análisis ─────────────────────────────

def period_table(panels: dict, periods: dict) -> pd.DataFrame:
    rows = {}
    for name, P in panels.items():
        for per, (a, b) in periods.items():
            sub = _between(P, a, b)
            if len(sub) < 252:
                continue
            rows[(name, per)] = {"desde": sub.index[0].date(), "años": len(sub) / ANN,
                                 **decompose(*_cols(sub))}
    return pd.DataFrame(rows).T.infer_objects()


def term_tstats(panels: dict, periods: dict) -> pd.DataFrame:
    """¿Es cada término distinto de cero? Media de la contribución mensual (con la L̄ del
    periodo) y su t de Newey-West. neto_varianza = timing de la varianza + coste de
    dispersión, la parte de segundo orden que queda neta (no entra en p_titular.csv)."""
    rows = {}
    for name, P in panels.items():
        for per, (a, b) in periods.items():
            sub = _between(P, a, b)
            row = {}
            for which in ("timing_media", "timing_varianza", "neto_varianza", "Δg"):
                c = monthly_contrib(sub, which)
                fit = stats.ols_hac(c, pd.DataFrame(index=c.index), NW_MONTHLY)
                row[f"{which} (pp)"] = c.mean() * 100
                row[f"t {which}"] = fit.t[0]
            rows[(name, per)] = row
    return pd.DataFrame(rows).T.infer_objects()


def rolling_decomposition(P: pd.DataFrame, window: int = ROLL, step: int = ROLL_STEP) -> pd.DataFrame:
    A = np.column_stack(list(_cols(P)))
    rows, idx = [], []
    for end in range(window, len(P) + 1, step):
        rows.append(decompose(*A[end - window:end].T))
        idx.append(P.index[end - 1])
    return pd.DataFrame(rows, index=idx)


def jackknife(P: pd.DataFrame) -> pd.DataFrame:
    rows = {"ninguno": {"días_quitados": 0, **decompose(*_cols(P))}}
    all_mask = np.zeros(len(P), bool)
    for ep, (a, b) in EPISODES.items():
        m = (P.index >= pd.Timestamp(a)) & (P.index <= pd.Timestamp(f"{b}-12-31"))
        if not m.any():
            continue
        all_mask |= m
        rows[ep] = {"días_quitados": int(m.sum()), **decompose(*_cols(P[~m]))}
    rows["todos a la vez"] = {"días_quitados": int(all_mask.sum()), **decompose(*_cols(P[~all_mask]))}
    return pd.DataFrame(rows).T.infer_objects()


def markov_fit(z: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """P(z_t = 1 | grupo_t, z_{t−1}) con frecuencias muestrales (grupos × 2)."""
    G = int(groups.max()) + 1
    p = np.empty((G, 2))
    for g in range(G):
        for s in (0, 1):
            m = (groups[1:] == g) & (z[:-1] == s)
            p[g, s] = z[1:][m].mean() if m.any() else z[groups == g].mean()
    return p


def markov_paths(p: np.ndarray, groups: np.ndarray, z0: float, n: int, rng) -> np.ndarray:
    """n caminos 0/1 de una cadena de Markov cuya probabilidad de estar en 1 depende
    del grupo de ese día y del estado del día anterior (independiente de las rentabilidades)."""
    T = len(groups)
    Z = np.empty((n, T))
    Z[:, 0] = z0
    for t in range(1, T):
        Z[:, t] = rng.random(n) < p[groups[t], Z[:, t - 1].astype(int)]
    return Z


PLACEBOS = ("sin condicionar", "deciles de varianza GJR", "deciles × década",
            "deciles de varianza GJR, persistente", "desplazamiento circular")


def placebo(d: dict, var_fc: pd.Series, which: str, rng, n_perm: int = N_PERM) -> dict:
    """Placebo condicional: se sustituye el estado de la media de 200 sesiones.

    La exposición aplicada es L_t = s_{t−LAG}·z_{t−LAG}, con z el estado (1 sobre
    la media, 0 debajo) y s = 3 (LRS 3x) o 3·(fracción de medio Kelly) (LRS-Kelly,
    sin la regla de reajuste mínimo). Esquemas:
    - «sin condicionar», «deciles de varianza GJR», «deciles × década»: z se baraja
      día a día entre días del mismo grupo (el de la varianza prevista en la
      fecha de la señal). Conserva cuánto esquiva la volatilidad prevista, pero
      destruye la persistencia del estado.
    - «deciles de varianza GJR, persistente»: cadena de Markov con
      P(z_t = 1 | decil_t, z_{t−1}) igual a la de la señal real: conserva la
      relación con la volatilidad prevista y la persistencia, y es independiente
      de las rentabilidades.
    - «desplazamiento circular»: la serie real de estados desplazada un número
      aleatorio de sesiones (al menos un año): persistencia y rachas exactas,
      sin condicionar a la volatilidad.
    Todo bruto de costes de rotación. La señal es la de la LRS original (r13.signal_index)."""
    sig = r13.above_sma(r13.signal_index(d))
    scale = pd.Series(3.0, index=sig.index) if which == "LRS 3x (original)" else \
        3 * r13.kelly_exposure(d, var_fc)
    F = pd.concat([sig.shift(LAG), var_fc.shift(LAG), scale.shift(LAG), d["u"] - d["rf"], d["rf"],
                   kappa(d)], axis=1, keys=["z", "v", "s", "e", "rf", "kap"]).loc[START:].dropna()
    z, s, e, rf, kap = (F[c].values for c in ("z", "s", "e", "rf", "kap"))
    net = e - kap
    dec = pd.qcut(F.v, N_DEC, labels=False).values
    decade = (F.index.year // 10).values
    groups = {"sin condicionar": np.zeros(len(F), int), "deciles de varianza GJR": dec,
              "deciles × década": dec * 1000 + decade}
    pm = markov_fit(z, dec)

    def draw(sch: str, n: int) -> np.ndarray:
        if sch in groups:
            return np.stack([permute_within(z, groups[sch], rng) for _ in range(n)])
        if sch == "deciles de varianza GJR, persistente":
            return markov_paths(pm, dec, z[0], n, rng)
        k = rng.integers(ANN, len(z) - ANN, n)
        return np.stack([np.roll(z, kk) for kk in k])

    def evaluate(Zm: np.ndarray) -> dict:
        Zm = np.atleast_2d(Zm)
        Lm = s * Zm
        Lb = Lm.mean(axis=1, keepdims=True)
        g = np.log1p(rf + Lm * net).mean(axis=1)
        gc = np.log1p(rf + Lb * net).mean(axis=1)
        L2 = Lm ** 2
        return {"Δg": (g - gc) * ANN, "g": g * ANN,
                "timing_media": ((Lm - Lb) * e).mean(axis=1) * ANN,
                "timing_varianza": -0.5 * ((L2 - L2.mean(axis=1, keepdims=True)) * e ** 2).mean(axis=1) * ANN,
                "dispersión": -0.5 * Lm.var(axis=1) * (e ** 2).mean() * ANN,
                "cambios_año": (np.diff(Zm, axis=1) != 0).sum(axis=1) / (Zm.shape[1] / ANN)}

    real = {k: float(v[0]) for k, v in evaluate(z).items()}
    out = {"real": real, "placebos": {}, "desde": F.index[0].date(), "n": len(F)}
    for sch in PLACEBOS:
        acc = {k: [] for k in real}
        for c0 in range(0, n_perm, 100):
            for k, v in evaluate(draw(sch, min(100, n_perm - c0))).items():
                acc[k].append(v)
        out["placebos"][sch] = {k: np.concatenate(v) for k, v in acc.items()}
    return out


def placebo_table(pl: dict) -> pd.DataFrame:
    rows = {}
    for which, res in pl.items():
        real = res["real"]
        for sch, P in res["placebos"].items():
            dg = P["Δg"]
            rows[(which, sch)] = {
                "Δg_real (pp)": real["Δg"] * 100, "Δg_placebo_media (pp)": dg.mean() * 100,
                "p5 (pp)": np.percentile(dg, 5) * 100, "p95 (pp)": np.percentile(dg, 95) * 100,
                "p_valor": (1 + (dg >= real["Δg"]).sum()) / (1 + len(dg)),
                "timing_media real": real["timing_media"] * 100,
                "timing_media placebo": P["timing_media"].mean() * 100,
                "timing_varianza real": real["timing_varianza"] * 100,
                "timing_varianza placebo": P["timing_varianza"].mean() * 100,
                "dispersión real": real["dispersión"] * 100,
                "dispersión placebo": P["dispersión"].mean() * 100,
                "cambios de estado/año real": real["cambios_año"],
                "cambios de estado/año placebo": P["cambios_año"].mean()}
    return pd.DataFrame(rows).T


def monthly_excess(r: pd.Series, rf: pd.Series) -> pd.Series:
    df = pd.concat([r, rf], axis=1, keys=["r", "rf"]).dropna()
    m = (1 + df).resample("ME").prod() - 1
    n = df.r.resample("ME").count()
    m = m[n >= 15]  # meses completos (el primero y el último pueden ir a medias)
    return m.r - m.rf


def spanning(y: pd.Series, X: pd.DataFrame, lags: int = NW_MONTHLY) -> dict:
    """Regresión de spanning (Huberman–Kandel; Moreira–Muir 2017): α > 0 significa
    que añadir y a las carteras de X mejora el Sharpe máximo alcanzable."""
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    fit = stats.ols_hac(df.y, df[X.columns], lags)
    Xv = np.column_stack([np.ones(len(df)), df[X.columns].values])
    resid = df.y.values - Xv @ fit.beta
    out = {"α anual (pp)": fit.beta[0] * 1200, "t_NW(α)": fit.t[0], "p(α)": fit.p[0]}
    for j, c in enumerate(X.columns, start=1):
        out[f"β {c}"] = fit.beta[j]
    out.update({"R²": fit.r2, "ratio_valoración": fit.beta[0] / resid.std(ddof=Xv.shape[1]) * np.sqrt(12),
                "meses": fit.n})
    return out


VSPECS = ("log RV ~ log GJR + sobre la media", "log(RV/GJR) ~ sobre la media (pendiente 1)",
          "RV ~ GJR + sobre la media (niveles)", "exceso 21 d (anual) ~ sobre la media + log GJR",
          "exceso 21 d (anual) ~ sobre la media (sin GJR)")


def variance_tests(d: dict, var_fc: pd.Series, start) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """¿El estado de la media de 200 sesiones anticipa la varianza realizada de
    las 21 sesiones siguientes más allá de la previsión GJR? (y la rentabilidad).

    Con la pendiente de log GJR libre, el coeficiente del estado compara con la
    previsión GJR RECALIBRADA dentro de muestra; con la pendiente fijada en 1
    (log(RV/GJR)), con la previsión GJR tal cual. Cada t se da con Newey-West de
    42 retardos (el de la versión anterior), Hansen-Hodrick de 20 y Newey-West de
    252, y con muestras mensuales no solapadas (los 21 desfases posibles)."""
    u = d["u"]
    sig = r13.above_sma(r13.signal_index(d)).rename("sobre_media")   # señal de la LRS original
    rv = ((u ** 2).rolling(H).sum().shift(-H) * ANN / H).rename("rv")
    fwd = ((np.log1p(u) - np.log1p(d["rf"])).rolling(H).sum().shift(-H) * ANN / H).rename("fwd")
    df = pd.concat([rv, fwd, var_fc.rename("gjr"), sig], axis=1).loc[start:].dropna()
    df = df[df.rv > 0]
    ly, lg = np.log(df.rv), np.log(df.gjr).rename("log_GJR")
    s = df[["sobre_media"]]
    specs = {  # (y, X, control, X sin el estado)
        VSPECS[0]: (ly, pd.concat([lg, s], axis=1), "log_GJR", lg.to_frame()),
        VSPECS[1]: (ly - lg, s, None, None),
        VSPECS[2]: (df.rv, df[["gjr", "sobre_media"]], "gjr", df[["gjr"]]),
        VSPECS[3]: (df.fwd, pd.concat([s, lg], axis=1), "log_GJR", lg.to_frame()),
        VSPECS[4]: (df.fwd, s, None, None),
    }
    rows = {}
    for name, (y, X, ctrl, X0) in specs.items():
        fit = ols_errors(y, X)
        j = fit["names"].index("sobre_media")
        nov = nonoverlap(y, X, "sobre_media")
        t_hh = fit["t"]["t HH 20"][j]
        row = {"coef_sobre_media": fit["beta"][j],
               **{lab: t[j] for lab, t in fit["t"].items()},
               "p HH": 2 * sps.norm.sf(abs(t_hh)), "estimador HH": fit["estimador"]["t HH 20"],
               "t mensual mín": nov.t.min(), "t mensual mediana": nov.t.median(),
               "t mensual máx": nov.t.max(),
               "coef mensual mín": nov.coef.min(), "coef mensual máx": nov.coef.max(),
               "coef_control": np.nan, "t_control (HH)": np.nan,
               "R²": fit["r2"], "R²_sin_estado": 0.0, "n": fit["n"], "media de y": float(np.mean(y)),
               **{f"ρ residuo {l}": pd.Series(fit["resid"]).autocorr(l) for l in RHO_LAGS}}
        if ctrl is not None:
            k = fit["names"].index(ctrl)
            row["coef_control"], row["t_control (HH)"] = fit["beta"][k], fit["t"]["t HH 20"][k]
            row["R²_sin_estado"] = ols_errors(y, X0, {})["r2"]
        rows[name] = row
    # descriptivo: cociente de medias RV / GJR (sin recalibrar) por celda, con su número de días
    df["decil"] = pd.qcut(df.gjr, N_DEC, labels=False) + 1
    grp = df.groupby(["decil", "sobre_media"])
    cells = {"cociente": (grp.rv.sum() / grp.gjr.sum()).unstack(), "n": grp.size().unstack()}
    summ = {}
    for st_, lab in ((1.0, "sobre la media"), (0.0, "bajo la media")):
        x = df[df.sobre_media == st_]
        hi, lo = x[x.decil >= 8], x[x.decil <= 7]
        summ[lab] = {"días": len(x), "volatilidad prevista mediana (%)": np.sqrt(x.gjr.median()) * 100,
                     "media geométrica RV/GJR": np.exp(np.log(x.rv / x.gjr).mean()),
                     "cociente de medias RV/GJR": x.rv.sum() / x.gjr.sum(),
                     "deciles 1–7: RV/GJR": lo.rv.sum() / lo.gjr.sum(), "deciles 1–7: días": len(lo),
                     "deciles 8–10: RV/GJR": hi.rv.sum() / hi.gjr.sum(), "deciles 8–10: días": len(hi)}
    return pd.DataFrame(rows).T.infer_objects(), cells, pd.DataFrame(summ).T.infer_objects()


def monthly_contrib(P: pd.DataFrame, which: str) -> pd.Series:
    """Contribución mensual a un término, escalada para que su media sea el término anual."""
    L, e = P.L, P.e
    if which == "timing_media":
        c = (L - L.mean()) * e
    elif which == "timing_varianza":
        c = -0.5 * (L ** 2 - (L ** 2).mean()) * e ** 2
    elif which == "neto_varianza":
        # timing de la varianza + dispersión: −½(L² − L̄²)e², cuya media es
        # −½Cov(L², e²) − ½Var(L)·E[e²] exactamente (momentos con divisor N)
        c = -0.5 * (L ** 2 - L.mean() ** 2) * e ** 2
    else:  # Δg
        rc = P.rf + L.mean() * (e - P.kap)
        c = np.log1p(P.r) - np.log1p(rc)
    m = c.resample("ME").sum()
    m = m[c.resample("ME").count() > 0]
    return m * ANN * len(m) / len(c)


def block_bootstrap_p(c: pd.Series, blocks=BOOT_BLOCKS, B: int = B_BOOT_TERM, seed: int = SEED_BOOT) -> dict:
    """p unilateral (media > 0) de la media de una contribución mensual con bootstrap
    estacionario por bloques: fracción de medias remuestreadas y centradas que igualan o
    superan la observada. Complementa la aproximación normal del t de Newey-West, con
    colas gruesas en los términos de rentabilidades al cuadrado."""
    x = c.dropna().to_numpy(float)
    m = x.mean()
    out = {}
    for b in blocks:
        rng = np.random.default_rng(seed + b)
        bm = np.array([x[stats.stationary_indices(len(x), b, rng)].mean() for _ in range(B)])
        out[f"p_bootstrap_{b}m"] = float(((bm - m) >= m).mean())
        out[f"t_bootstrap_{b}m"] = float(m / bm.std(ddof=1))
    return out


# ───────────────────────────── figuras ─────────────────────────────

def _pp(x: float, d: int = 2, sign: bool = True) -> str:
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace(".", ",").replace("-", "−")


def _sig_one(t: float, alpha: float = 0.05) -> bool:
    """¿Significativo con el p normal UNILATERAL (a favor de la estrategia, t > 0), el mismo que
    titular_rows exporta a R19? Un t NaN no es significativo."""
    return bool(sps.norm.sf(float(t)) < alpha)


def _sig_two(t: float, alpha: float = 0.05) -> bool:
    """¿Significativo con el p normal BILATERAL (coeficientes de signo libre)? Un t NaN no lo es."""
    return bool(2 * sps.norm.sf(abs(float(t))) < alpha)


def fig_bars(T: pd.DataFrame, rep: Report):
    fig, axes = plt.subplots(len(STRATS), 1, figsize=(FIG_W, 3.6 * len(STRATS) + 0.6), sharey=True)
    axes = np.atleast_1d(axes)
    pers = list(PERIODS)
    xl = [p.replace("ETFs reales (2010–2026)", "ETFs reales\n2010–2026") for p in pers]
    for ax, name in zip(axes, STRATS):
        sub = T.loc[name].reindex(pers)
        x = np.arange(len(pers))
        pos, neg = np.zeros(len(pers)), np.zeros(len(pers))
        for t in TERMS:
            v = sub[t].astype(float).values * 100
            base = np.where(v >= 0, pos, neg)
            ax.bar(x, v, 0.62, bottom=base, color=COL[t], edgecolor="white", lw=1.2,
                   label=LABEL[t])
            pos, neg = pos + np.where(v >= 0, v, 0), neg + np.where(v < 0, v, 0)
        dg = sub["Δg"].astype(float).values * 100
        ax.scatter(x, dg, marker="D", s=70, color=INK, zorder=5, label="Δg exacto")
        for xi, v in zip(x, dg):
            ax.text(xi + 0.34, v, _pp(v, 1), va="center", ha="left", fontsize=11, color=INK)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlim(-0.5, len(pers) - 0.2)
        ax.set_xticks(x, xl)
        ax.set(title=f"{FIG_NAME.get(name, name)}: L̄ = {_pp(float(sub['L̄'].iloc[0]), 2, False)} en 1938–2026",
               ylabel="Puntos al año")
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        comma_axis(ax)   # marcas con coma decimal, como las cifras rotuladas (_pp)
    h, lab = axes[0].get_legend_handles_labels()
    legend_band(fig, h, lab, ncol=2, fontsize=11.5)
    fig.suptitle("Descomposición de la diferencia frente al apalancamiento constante")
    rep.fig(fig, "terminos_subperiodos",
            "Cada barra reparte la diferencia de crecimiento logarítmico anual entre la estrategia y "
            "la misma exposición media mantenida constante (L̄ de cada periodo) en sus cuatro términos: "
            "timing de la media, timing de la varianza, coste de dispersión (siempre negativo) y "
            "financiación y costes; el rombo negro es la diferencia exacta, con su valor al lado.")


def fig_rolling(Rl: dict, rep: Report):
    fig, axes = plt.subplots(len(Rl), 1, figsize=(11, 4.3 * len(Rl) + 0.4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, df) in zip(axes, Rl.items()):
        for t in TERMS[:4]:
            ax.plot(df.index, df[t] * 100, color=COL[t], lw=1.8, label=LABEL[t])
        ax.plot(df.index, df["Δg"] * 100, color=INK, lw=2.4, label="Δg exacto")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set(title=name, ylabel="pp al año (10 años previos)")
        ax.grid(axis="y", color="0.9", lw=0.6)
    h, lab = axes[0].get_legend_handles_labels()
    legend_band(fig, h, lab, ncol=3, fontsize=11.5)
    fig.suptitle("Los términos en ventanas móviles de diez años (S&P 500)")
    rep.fig(fig, "terminos_ventana_movil",
            "Cada punto descompone la ventaja de crecimiento de la década que termina en esa fecha, con "
            "la exposición media de esa misma década como referencia constante; la línea negra es la "
            "diferencia exacta y las de color, sus términos.")


def fig_placebo(pl: dict, rep: Report):
    fig, axes = plt.subplots(len(pl), 1, figsize=(FIG_W, 3.9 * len(pl) + 0.8))
    axes = np.atleast_1d(axes)
    colors = {"deciles de varianza GJR": "#2a78d6", "deciles de varianza GJR, persistente": "#eb6834",
              "desplazamiento circular": "#1baf7a"}
    labels = {"deciles de varianza GJR": "barajado día a día dentro de cada decil de varianza",
              "deciles de varianza GJR, persistente": "persistente (Markov) dentro de cada decil",
              "desplazamiento circular": "estado real desplazado en el tiempo"}
    for ax, (which, res) in zip(np.atleast_1d(axes), pl.items()):
        shown = {k: res["placebos"][k] for k in colors}
        allv = np.concatenate([P["Δg"] for P in shown.values()] + [[res["real"]["Δg"]]]) * 100
        bins = np.linspace(allv.min() - 0.3, allv.max() + 0.3, 90)
        for sch, P in shown.items():
            ax.hist(P["Δg"] * 100, bins=bins, color=colors[sch], alpha=0.18, histtype="stepfilled")
            ax.hist(P["Δg"] * 100, bins=bins, color=colors[sch], lw=1.8, histtype="step",
                    label=labels[sch])
        rv = res["real"]["Δg"] * 100
        ax.axvline(rv, color=INK, lw=2.4, label="la señal real")
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.12)     # sitio para el rótulo por encima de las barras
        ax.annotate(f"real: {_pp(rv)} puntos", (rv, top * 1.06), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=11, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.axvline(0, color=INK2, lw=0.8, ls="--")
        ax.set(title=f"{which}: la señal real frente a {num_es(N_PERM, 0)} sin información",
               xlabel="Δg frente a la exposición media constante (puntos al año)",
               ylabel="Placebos")
        comma_axis(ax)
    # la leyenda, común a los dos paneles, en su propia franja: no tapa las barras
    h, lab = axes[0].get_legend_handles_labels()
    legend_band(fig, h, lab, ncol=1, fontsize=11.5)
    fig.suptitle("Placebo condicional: ¿gana la LRS solo por esquivar volatilidad?")
    rep.fig(fig, "placebo",
            "Distribución de la ventaja de crecimiento frente a la exposición media constante cuando el "
            "estado sobre/bajo la media de 200 sesiones se sustituye por uno sin información sobre las "
            "rentabilidades: barajado día a día dentro de cada decil de varianza prevista por el GJR "
            "(azul), generado con la misma persistencia y la misma relación con esos deciles (naranja) "
            "o la propia serie de estados desplazada en el tiempo (verde); la línea negra es la señal "
            "real. Si la LRS ganara solo por esquivar volatilidad, la línea caería dentro de las "
            "distribuciones azul y naranja.")


def fig_wald(B: dict, crit: float, rep: Report):
    fig, axes = plt.subplots(len(B), 1, figsize=(11, 3.9 * len(B)), sharex=True)
    for ax, (name, res) in zip(np.atleast_1d(axes), B.items()):
        path = pd.Series(res["path"], index=res["idx"]).dropna()
        ax.plot(path.index, path.values, color="#2a78d6", lw=1.8)
        top = max(crit, float(path.max())) * 1.3
        ax.set_ylim(0, top)
        ax.axhline(crit, color=INK2, lw=1.2, ls="--")
        ax.text(path.index[-1], crit, "valor crítico al 5% ", va="bottom", ha="right", fontsize=11,
                color=INK2)
        kd = res["idx"][res["k"]]
        ax.axvline(kd, color=INK, lw=1)
        right = kd > path.index[len(path) // 2]
        ax.text(kd, top * 0.6, f"máximo: {kd:%Y-%m} " if right else f" máximo: {kd:%Y-%m}", va="center",
                ha="right" if right else "left", fontsize=11, color=INK)
        ax.set(title=f"{name}: sup-Wald = {_pp(res['stat'], 1, False)} (p = {_pp(res['p'], 3, False)})",
               ylabel="Estadístico de Wald")
        ax.grid(axis="y", color="0.9", lw=0.6)
    fig.suptitle("¿Cuándo cambió el timing de la media? Ruptura en fecha desconocida")
    rep.fig(fig, "ruptura_timing_media",
            "Estadístico de Wald (Newey-West) de un cambio en la media de la contribución mensual al "
            "timing de la media, para cada fecha de ruptura candidata entre el 15% y el 85% de la "
            "muestra; la línea vertical marca el máximo y la discontinua, el valor crítico asintótico "
            "de Andrews (1993) al 5%.")


def fig_variance(V: dict, rep: Report):
    fig, axes = plt.subplots(len(V), 1, figsize=(11, 4.9 * len(V)), sharey=True)
    ymax = max(max(float(np.nanmax(c["cociente"].values)), float(c["resumen"].filter(like="RV/GJR").max().max()))
               for c in V.values()) * 1.32
    states = ((1.0, "sobre la media de 200", "#2a78d6", "sobre la media"),
              (0.0, "bajo la media de 200", "#eb6834", "bajo la media"))
    for ax, (name, c) in zip(np.atleast_1d(axes), V.items()):
        ratio, n, summ = c["cociente"], c["n"], c["resumen"]
        x = np.r_[ratio.index.values.astype(float), 11.9, 13.1]
        w = 0.38
        for j, (state, lab, col, key) in enumerate(states):
            vals = np.r_[ratio[state].values, summ.loc[key, "deciles 1–7: RV/GJR"],
                         summ.loc[key, "deciles 8–10: RV/GJR"]]
            cnt = np.r_[n[state].values, summ.loc[key, "deciles 1–7: días"],
                        summ.loc[key, "deciles 8–10: días"]]
            xs = x + (j - 0.5) * w
            small = cnt < SMALL_CELL
            ax.bar(xs[~small], vals[~small], w, color=col, edgecolor="white", lw=1, label=lab)
            ax.bar(xs[small], vals[small], w, color=col, alpha=0.35, edgecolor=col, lw=1, hatch="//")
            for xi, v, k in zip(xs, vals, cnt):
                ax.text(xi, v + 0.02, f"{int(k)}", rotation=90, ha="center", va="bottom", fontsize=10,
                        color=INK2)
        ax.axvline(11.0, color=INK2, lw=0.8, ls=":")
        ax.axhline(1, color=INK2, lw=0.8)
        ax.set(title=name, ylabel="Realizada / prevista", ylim=(0, max(1.3, ymax)))
        ax.set_xticks(x, [*(str(int(v)) for v in x[:-2]), "1–7", "8–10"])
        ax.set_xlabel("Decil de la varianza prevista por el GJR (1 = más baja); a la derecha, agrupados")
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].legend(loc="upper left", ncol=2, frameon=False)
    fig.suptitle("¿Sabe la media de 200 sesiones algo de la varianza que el GJR no sepa?")
    rep.fig(fig, "varianza_estado",
            "Varianza realizada de las 21 sesiones siguientes dividida por la prevista por el GJR-GARCH "
            "sin recalibrar (cociente de sumas dentro de cada celda), por decil de la previsión y según "
            "el índice cotice sobre (azul) o bajo (naranja) su media de 200 sesiones; los dos últimos "
            "grupos agrupan los deciles 1–7 y 8–10. Sobre cada barra, el número de días (solapados: "
            "unos 21 forman un mes independiente); las celdas rayadas tienen menos de 252. Por debajo "
            "de 1, el GJR previó más varianza de la que llegó. Una diferencia sistemática entre azul "
            "y naranja, en cualquiera de los dos sentidos, sería información que el GJR no tiene.")


# ───────────────────────────── informe ─────────────────────────────

DERIVATION = """
Sea r_t el T-bill del día, e_t = u_t − r_t el exceso del índice sobre él y
L_t la exposición al índice que la estrategia tiene ese día (decidida al
cierre de t−2 y ejecutada al de t−1). Las tres estrategias invierten un peso
L_t/3 en el 3x y el resto en T-bills, de modo que

    R_t = r_t + L_t·e_t − k_t,

donde k_t recoge todo lo que la estrategia paga: financiación de la exposición
del swap al tipo f_t (call money hasta 1954, fondos federales después; f_t − r_t
por unidad de exposición),
diferencial del swap y comisión del 3x, error de seguimiento del ETF real
desde 2009 y 5 pb por unidad de rotación. La referencia mantiene constante la
exposición media L̄ = E_T[L_t] con el mismo instrumento (peso L̄/3 en el 3x,
reajustado a diario sin coste de rotación):

    R^c_t = r_t + L̄·e_t − L̄·κ_t,    κ_t = (r_t + 3·e_t − x3_t)/3,

con κ_t el coste por unidad de exposición del 3x. Escribiendo
log(1 + R_t) = log(1 + r_t) + log(1 + x_t), con x_t = (L_t·e_t − k_t)/(1 + r_t),
y usando log(1 + x) = x − x²/2 + O(x³), el log(1 + r_t) común se cancela y, en
medias muestrales E_T (divisor N),

    Δg = E_T[log(1 + R_t)] − E_T[log(1 + R^c_t)]
       = E_T[(L_t − L̄)·e_t] − ½·E_T[L_t²·e_t² − L̄²·e_t²] − E_T[k_t − L̄·κ_t] + resto.

Como E_T[L_t − L̄] = 0, el primer término es exactamente la covarianza muestral.
En el segundo, E[L²e²] = Cov(L², e²) + E[L²]·E[e²] y E[L²] = L̄² + Var(L), así que

    Δg = Cov_T(L_t, e_t)              (1) timing de la media
       − ½·Cov_T(L_t², e_t²)          (2) timing de la varianza
       − ½·Var_T(L_t)·E_T[e_t²]       (3) coste de dispersión
       − E_T[k_t − L̄·κ_t]             (4) financiación y costes
       + resto                         (orden ≥ 3, factor 1/(1 + r_t) y productos con k_t)

(1) es positivo si la estrategia está más expuesta cuando el exceso es mayor;
(2), si está más expuesta cuando el cuadrado de la rentabilidad es menor; (3)
es siempre negativo: el crecimiento es cóncavo en L, y mover la exposición
alrededor de L̄ sin información cuesta ½·Var(L)·σ². (4) se separa a su vez en
−Cov_T(L_t, κ_t) (estar dentro cuando financiarse es caro o barato) y el coste
de rotación. Se usa e_t² y no u_t²: la expansión se hace alrededor de
log(1 + r_t) y la diferencia (2·u_t·r_t − r_t²) es del orden de 10⁻⁶ al día. Todo
se anualiza por 252 y se expresa en puntos porcentuales (pp) de crecimiento
logarítmico: una Δg de 4 pp convierte, por ejemplo, un 14% de crecimiento log
en un 18% (CAGR = e^g − 1).
""".strip("\n")

PROTOCOL = f"""
Protocolo y salvedades de diseño:

- Estrategias y datos de R13 sin cambios (`r13.strategies`: la LRS 3x original
  con la media sobre la rentabilidad total, orden al cierre siguiente,
  financiación a f más el sobrecoste del swap, 5 pb por rotación, 3x real desde
  su lanzamiento). Muestra común de las tres estrategias en el S&P 500 desde
  1938; la LRS 3x, que no estima nada, también desde 1928.
- L̄ es la media de la exposición en cada periodo analizado: la referencia
  constante es una herramienta contable ex post, no una estrategia que se
  pudiera elegir de antemano. Por eso los términos de subperiodos distintos no
  se suman: cada uno tiene su propia L̄.
- Jackknife: se quitan los años naturales completos de cada episodio (así
  entra su rebote dentro del año), se recalcula L̄ y se repite la descomposición.
- Placebo: los deciles de la varianza prevista por el GJR (que en cada fecha
  sólo usa datos hasta esa fecha) se cortan con toda la muestra 1938–2026; es
  un contraste de la hipótesis nula, no una regla. {N_PERM} permutaciones por
  esquema; p-valor unilateral (1 + #placebos ≥ real)/(1 + {N_PERM}).
- Spanning y ruptura: regresiones dentro de muestra con errores Newey-West
  ({NW_MONTHLY} retardos con datos mensuales). El sup-Wald de Andrews (1993)
  recorta el {TRIM:.0%} de cada extremo; su p-valor sale de {N_NULL:_} simulaciones
  de la distribución asintótica (puentes brownianos).
- Qué se fijó antes y qué se añadió después. Antes de ver resultados se fijaron
  las estrategias, la descomposición, los subperiodos, la ventana móvil, los
  episodios del jackknife, los placebos barajados (sin condicionar, deciles de
  varianza, deciles × década), el spanning, el sup-Wald y la regresión de la
  varianza siguiente con Newey-West de {NW_DAILY} retardos. Se añadieron DESPUÉS de
  ver el placebo barajado: el placebo persistente (Markov), el desplazamiento
  circular y las dos regresiones del exceso de rentabilidad a 21 días (con y sin
  el GJR como control). Se añadieron después de una revisión externa: la
  regresión con la pendiente de log GJR fijada en 1, los errores Hansen-Hodrick
  ({HH_LAGS} retardos, kernel uniforme) y Newey-West de {NW_LONG} retardos, las
  muestras mensuales no solapadas (los {H} desfases posibles, Newey-West de
  {NW_NONOV} retardos), los cocientes agrupados con su número de días, la
  concentración diaria de los términos y la descomposición sin las N sesiones
  de mayor |e| (rejilla N = {', '.join(map(str, TRIM_N))}, que se muestra entera). Todo
  lo añadido después es exploratorio; ninguna rejilla se ha usado para elegir
  parámetros.
""".replace("_", ".").strip("\n")




# ───────────────────────────── ejecución ─────────────────────────────

def _flat(df: pd.DataFrame) -> pd.DataFrame:
    """Índices compuestos como texto legible («a · b»)."""
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out.index = [" · ".join(map(str, t)) for t in out.index]
    return out


def _to_pp(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out:
            out[c] = out[c].astype(float) * 100
    return out.rename(columns={c: f"{c} (pp)" for c in cols if c in out})


SHOW = ["desde", "años", "L̄", "g_estrategia", "g_constante", "Δg", *TERMS[:4], "suma_1_4", "resto"]
PPCOLS = ["g_estrategia", "g_constante", "Δg", *TERMS, "suma_1_4", "financiación", "rotación"]


def analyse(rng) -> dict:
    D = r13.load()
    D.pop("_ndx")
    VAR, RE = {}, {}
    for name, d in D.items():
        VAR[name] = r13.var_forecast(d)
        RE[name] = r13.strategies(d, VAR[name])
    d = D["S&P 500"]
    R, E = RE["S&P 500"]
    panels = {k: panel(d, R, E, k) for k in STRATS}
    common = panels[STRATS[0]].loc[START:].index
    for k in STRATS[1:]:
        common = common.intersection(panels[k].index)
    P38 = {k: panels[k].loc[common] for k in STRATS}
    P28 = panels["LRS 3x (original)"]
    L3 = STRATS[0]
    A = {"P38": P38}

    # 1. subperiodos y otros mercados
    A["T"] = period_table(P38, PERIODS)
    A["T28"] = period_table({"LRS 3x (original)": P28}, {"1928–2026 (desde la señal)": (None, None)})
    A["TS"] = pd.concat([term_tstats(P38, PERIODS),
                         term_tstats({"LRS 3x (original)": P28}, {"1928–2026 (desde la señal)": (None, None)})])
    bt = {}
    for which in ("timing_varianza", "neto_varianza", "Δg"):
        c = monthly_contrib(P38[L3], which)
        fit = stats.ols_hac(c, pd.DataFrame(index=c.index), NW_MONTHLY)
        row = {"media (pp)": c.mean() * 100, "t NW": float(fit.t[0]),
               "p normal": float(sps.norm.sf(fit.t[0]))}
        row.update(block_bootstrap_p(c))
        for b in BOOT_BLOCKS:
            row[f"p_bootstrap_{b}m ×{N_TITULAR}"] = min(1.0, N_TITULAR * row[f"p_bootstrap_{b}m"])
        bt[which] = row
    A["boot_terms"] = pd.DataFrame(bt).T
    rows = []
    for name in ("Nasdaq-100", "Semis"):
        dm = D[name]
        Rm, Em = RE[name]
        st = VAR[name].first_valid_index()
        if dm["eval_from"] is not None:
            st = max(st, dm["eval_from"])
        pm = {k: panel(dm, Rm, Em, k).loc[st:] for k in STRATS}
        tm = period_table(pm, {f"desde {st.year}": (st, None),
                               "ETFs reales (2010–2026)": (r13.REAL_FROM, None)})
        tm.index = pd.MultiIndex.from_tuples([(name, a, b) for a, b in tm.index])
        rows.append(tm)
    A["To"] = pd.concat(rows)

    # 2. ventana móvil
    A["Rl"] = {k: rolling_decomposition(P38[k]) for k in STRATS}
    A["rl"] = pd.DataFrame({k: {
        "Δg > 0 (% de ventanas)": (v["Δg"] > 0).mean(),
        "timing_media > 0": (v["timing_media"] > 0).mean(),
        "timing_varianza > 0": (v["timing_varianza"] > 0).mean(),
        "Δg mín (pp)": v["Δg"].min() * 100, "Δg máx (pp)": v["Δg"].max() * 100,
        "timing_media mín (pp)": v["timing_media"].min() * 100,
        "timing_media máx (pp)": v["timing_media"].max() * 100,
        "timing_varianza mín (pp)": v["timing_varianza"].min() * 100,
        "timing_varianza máx (pp)": v["timing_varianza"].max() * 100,
        "ventanas": len(v)} for k, v in A["Rl"].items()}).T.infer_objects()

    # 3. jackknife
    J = {f"{k} (1938–)": jackknife(P38[k]) for k in STRATS}
    J["LRS 3x (original) (1928–)"] = jackknife(P28)
    A["J"] = pd.concat(J)

    # 3b. sensibilidad: la LRS 3x con la señal de precio (misma muestra común que las tres)
    Rp, Ep = r13.price_variants(d, VAR["S&P 500"])
    Ppx = panel(d, Rp, Ep, PX).reindex(common).dropna()
    px_periods = {"1938–2026": PERIODS["1938–2026"], "ETFs reales (2010–2026)": PERIODS["ETFs reales (2010–2026)"]}
    A["TPX"] = pd.concat([period_table({L3: P38[L3]}, px_periods), period_table({PX: Ppx}, px_periods)])
    A["TSPX"] = pd.concat([term_tstats({L3: P38[L3]}, px_periods), term_tstats({PX: Ppx}, px_periods)])
    A["JPX"] = jackknife(Ppx)

    # 4. placebo
    A["pl"] = {k: placebo(d, VAR["S&P 500"], k, rng) for k in ("LRS 3x (original)", "LRS-Kelly")}
    A["PT"] = placebo_table(A["pl"])

    # 5. spanning (excesos mensuales)
    ex = {"1x": monthly_excess(d["u"], d["rf"])}
    for k in STRATS:
        ex[k] = monthly_excess(R[k].loc[common], d["rf"])
    models = {
        "½ Kelly GJR ~ 1x": ("½ Kelly GJR (sin tendencia)", None),
        "LRS 3x ~ 1x": ("LRS 3x (original)", None),
        "LRS 3x ~ 1x + ½ Kelly GJR": ("LRS 3x (original)", "½ Kelly GJR (sin tendencia)"),
        "½ Kelly GJR ~ 1x + LRS 3x": ("½ Kelly GJR (sin tendencia)", "LRS 3x (original)"),
        "LRS-Kelly ~ 1x + ½ Kelly GJR": ("LRS-Kelly", "½ Kelly GJR (sin tendencia)"),
    }
    srows = {}
    for mname, (yk, x2) in models.items():
        X = pd.DataFrame({"1x": ex["1x"]})
        if x2 is not None:
            X["2º regresor"] = ex[x2]
        for per, (a, b) in PERIODS.items():
            y = _between(ex[yk].to_frame("y"), a, b)["y"]
            srows[(mname, per)] = spanning(y, X)
    A["ST"] = pd.DataFrame(srows).T.infer_objects()

    # 6. varianza siguiente
    VT, VF, VS = [], {}, []
    for name, dm in D.items():
        st = VAR[name].first_valid_index()
        if dm["eval_from"] is not None:
            st = max(st, dm["eval_from"])
        tab_v, cells, summ = variance_tests(dm, VAR[name], st)
        lab = f"{name} (desde {st.year})"
        VF[name] = {**cells, "resumen": summ}
        tab_v.index = pd.MultiIndex.from_product([[lab], tab_v.index])
        summ = summ.copy()
        summ.index = pd.MultiIndex.from_product([[lab], summ.index])
        VT.append(tab_v)
        VS.append(summ)
    A["VT"], A["VF"], A["VS"] = pd.concat(VT), VF, pd.concat(VS)

    # 7. ruptura estructural
    null = sup_wald_null(rng=rng)
    crit = {q: float(np.quantile(null, 1 - q)) for q in (0.10, 0.05, 0.01)}
    brows, Bfig = {}, {}
    targets = [(f"{k} (1938–)", P38[k]) for k in STRATS] + [("LRS 3x (original) (1928–)", P28)]
    for label, P in targets:
        for which in ("timing_media", "timing_varianza", "Δg"):
            c = monthly_contrib(P, which)
            res = sup_wald(c.values)
            res["p"] = float((null >= res["stat"]).mean())
            res["idx"] = c.index
            k92 = int(np.searchsorted(c.index, pd.Timestamp("1992-01-01")))
            brows[(label, which)] = {
                "sup_Wald": res["stat"], "p_asintótico": res["p"],
                "máximo en": f"{c.index[res['k']]:%Y-%m}",
                "media antes del máximo (pp)": res["media_antes"] * 100,
                "media después del máximo (pp)": res["media_después"] * 100,
                "media antes de 1992 (pp)": c.values[:k92].mean() * 100,
                "media desde 1992 (pp)": c.values[k92:].mean() * 100,
                "Wald en 1992-01": res["path"][k92],
                "cambio necesario en 1992 (pp)": np.sqrt(crit[0.05]) * res["se"][k92] * 100,
                "meses": len(c)}
            if which == "timing_media" and "1938" in label:
                Bfig[label.replace(" (1938–)", "")] = res
    A["BT"] = pd.DataFrame(brows).T.infer_objects()

    # apoyo al texto: timing de la media por año con ETFs reales y sesgo del GJR
    P = P38["LRS 3x (original)"].loc[r13.REAL_FROM:]
    yr = ((P.L - P.L.mean()) * P.e).groupby(P.index.year).sum() * ANN / len(P) * 100
    A["yr_etf"] = yr.sort_values().head(5).sort_index()
    # concentración diaria de los términos (1) y (2) y descomposición sin las sesiones extremas
    # (añadido tras la revisión externa; la rejilla TRIM_N se muestra entera)
    conc, trim, days = {}, {}, {}
    for k in STRATS:
        P = P38[k]
        Lb = P.L.mean()
        contrib = {"timing_media": (P.L - Lb) * P.e * ANN / len(P) * 100,
                   "timing_varianza": -0.5 * (P.L ** 2 - (P.L ** 2).mean()) * P.e ** 2 * ANN / len(P) * 100}
        for term, c in contrib.items():
            row = concentration(c)
            top, bot = c.nlargest(20).index, c.nsmallest(20).index
            row.update({
                "% en los 10 mejores": row["10 mejores"] / row["total"] * 100 if row["total"] > 0 else np.nan,
                "sin los 10 mejores": row["total"] - row["10 mejores"],
                "sin los 10 mejores ni los 10 peores": row["total"] - row["10 mejores"] - row["10 peores"],
                "20 mejores: L < L̄ y caída": int(((P.L[top] < Lb) & (P.e[top] < 0)).sum()),
                "20 mejores: L < L̄ y subida": int(((P.L[top] < Lb) & (P.e[top] > 0)).sum()),
                "20 peores: L < L̄ y subida": int(((P.L[bot] < Lb) & (P.e[bot] > 0)).sum()),
                "20 peores: L > L̄": int((P.L[bot] > Lb).sum())})
            conc[(k, term)] = row
            days[(k, term)] = c
        for nn in (0, *TRIM_N):
            trim[(k, nn)] = {"días_quitados": nn, **decompose(*_cols(drop_extremes(P, nn)))}
    A["CONC"], A["TRIM"] = pd.DataFrame(conc).T.infer_objects(), pd.DataFrame(trim).T.infer_objects()
    c1, c2 = days[(STRATS[0], "timing_media")], days[(STRATS[0], "timing_varianza")]
    A["top_days"] = {"t2_dias": [f"{t:%Y-%m-%d}" for t in c2.nlargest(3).index],
                     "t1_top": [(f"{t:%Y-%m-%d}", float(v)) for t, v in c1.nlargest(10).items()],
                     "t1_bot": [(f"{t:%Y-%m-%d}", float(v)) for t, v in c1.nsmallest(10).items()]}
    rv = ((d["u"] ** 2).rolling(H).sum().shift(-H) * ANN / H)
    B = pd.concat([rv, VAR["S&P 500"]], axis=1, keys=["rv", "g"]).loc[START:].dropna()
    A["gjr_bias"] = {"1938–1969": B.loc[:"1969"].rv.mean() / B.loc[:"1969"].g.mean(),
                     "1938–2026": B.rv.mean() / B.g.mean()}
    A["Bfig"], A["crit"] = Bfig, crit
    return A


def _summary(A: dict) -> list[str]:
    """Resumen con las cifras tal como salen de las tablas."""
    T, To, rl, J, PT, ST, VT, BT, ts = (A[k] for k in ("T", "To", "rl", "J", "PT", "ST", "VT", "BT", "TS"))
    CONC, TRIMT = A["CONC"], A["TRIM"]
    q = lambda s, p, c: float(T.loc[(s, p), c]) * 100
    L3, LK, KG = STRATS
    Tall = pd.concat([T, A["T28"]])
    rest_sp = float(Tall["resto"].abs().max() * 100)
    rest_o = float(To["resto"].abs().max() * 100)
    lr = T.loc[(L3, "1938–2026")]
    kg = T.loc[(KG, "1938–2026")]
    lk = T.loc[(LK, "1938–2026")]
    c1, c2 = CONC.loc[(L3, "timing_media")], CONC.loc[(L3, "timing_varianza")]
    tr = TRIMT.loc[L3]
    tr_s = lambda col: " / ".join(_pp(float(tr.loc[n, col]) * 100) for n in TRIM_N)
    trk = lambda n: _pp(float(TRIMT.loc[(LK, n), "timing_media"]) * 100)
    n_grid = ", ".join(str(n) for n in TRIM_N)
    out = []
    t_dg = float(ts.loc[(L3, '1938–2026'), 't Δg'])
    first_order = lr["timing_media"] > abs(lr["timing_varianza"] + lr["dispersión"])
    gana = lambda x: "gana" if x > 0 else "pierde"  # noqa: E731
    out.append(
        f"1. **La identidad cierra.** Los cuatro términos suman el Δg exacto: en el S&P 500 el resto "
        f"(orden ≥ 3) no pasa de {_pp(rest_sp, 2, False)} pp al año en ningún periodo; en el Nasdaq-100 "
        f"y los semiconductores, con días de ±10%, llega a {_pp(rest_o, 2, False)} pp.")
    out.append(
        f"2. **Contablemente, la LRS 3x {gana(lr['Δg'])} a su apalancamiento constante"
        + (" sobre todo por el término de primer orden, el «timing de la media»" if first_order else "")
        + (", y esa ventaja no es significativa.**" if lr["Δg"] > 0 and not _sig_one(t_dg) else ".**")
        + f" En 1938–2026 "
        f"crece {_pp(lr['Δg'] * 100)} pp al año más que un {_pp(lr['L̄'], 2, False)}x constante con el "
        f"mismo 3x ({_pp(lr['g_estrategia'] * 100, 2, False)}% frente a "
        f"{_pp(lr['g_constante'] * 100, 2, False)}% de crecimiento logarítmico): "
        f"{_pp(lr['timing_media'] * 100)} pp de timing de la media, {_pp(lr['timing_varianza'] * 100)} "
        f"de timing de la varianza, {_pp(lr['dispersión'] * 100)} de coste de dispersión y "
        f"{_pp(lr['financiación_y_costes'] * 100)} de financiación y costes. El timing de la varianza "
        f"y la dispersión suman {_pp((lr['timing_varianza'] + lr['dispersión']) * 100)} pp netos "
        f"(t = {_pp(ts.loc[(L3, '1938–2026'), 't neto_varianza'])}"
        + (", no significativo)" if not _sig_one(ts.loc[(L3, '1938–2026'), 't neto_varianza']) else ")")
        + ". "
        "El término de primer orden vive en los extremos: sus 20 mejores sesiones suman "
        f"{_pp(c1['20 mejores'])} pp ({_pp(c1['20 mejores'] / c1['total'] * 100, 0, False)}% del total) "
        f"y son {int(c1['20 mejores: L < L̄ y caída'])} caídas con la LRS fuera (sólo la sesión del "
        f"{A['top_days']['t1_top'][0][0]} aporta {_pp(A['top_days']['t1_top'][0][1])}); sus 20 peores suman {_pp(c1['20 peores'])} "
        f"y son {int(c1['20 peores: L < L̄ y subida'])} rebotes con la LRS también fuera. Las dos colas "
        "casi se compensan, y el término sobrevive a quitar por igual las sesiones de mayor |e| (punto "
        f"4). Pero su t de Newey-West (contribuciones mensuales) es "
        f"{_pp(ts.loc[(L3, '1938–2026'), 't timing_media'])}, y la de la ventaja total "
        f"{_pp(ts.loc[(L3, '1938–2026'), 't Δg'])}; el timing de la varianza, en cambio, tiene t = "
        f"{_pp(ts.loc[(L3, '1938–2026'), 't timing_varianza'])}. "
        + (f"En {len(A['P38'][L3]) / ANN:.0f} años, la ventaja de la LRS sobre su apalancamiento constante no "
           f"es estadísticamente significativa (p unilateral {_pp(sps.norm.sf(t_dg), 3, False)}). "
           if not _sig_one(t_dg) else "")
        + "Y «timing de la media» es "
        "un nombre contable: dice que la exposición diaria covarió con el exceso de ese día, no que la "
        "señal prediga la rentabilidad (punto 9). En un día de crash con la LRS fuera ganan a la vez "
        "el término de primer orden y el de segundo; la expansión reparte ese mismo día entre e y e².")
    out.append(
        f"3. **El ½ Kelly GJR es el artículo 1 escrito como contabilidad.** Frente a su constante "
        f"{gana(kg['Δg'])} {_pp(abs(kg['Δg']) * 100, 2, False)} pp: {_pp(kg['timing_varianza'] * 100)} de "
        f"timing de la varianza, {_pp(kg['dispersión'] * 100)} de dispersión y "
        f"{_pp(kg['timing_media'] * 100)} de timing de la media. La varianza es predecible y saberlo vale "
        f"{_pp(kg['timing_varianza'] * 100, 1, False)} pp al año, pero mover el apalancamiento para "
        f"aprovecharlo cuesta {_pp(-kg['dispersión'] * 100, 1, False)} en dispersión, y recortar tras los "
        f"sustos {'cuesta además' if kg['timing_media'] < 0 else 'no cuesta'} los rebotes. La LRS-Kelly: "
        f"{_pp(lk['Δg'] * 100)} pp, con {_pp(lk['timing_media'] * 100)} de timing de la media.")
    out.append(
        f"4. **El timing de la varianza es la parte persistente; el de la media, la ruidosa.** En "
        f"ventanas de diez años, el timing de la varianza de la LRS 3x es positivo en el "
        f"{_pp(rl.loc[L3, 'timing_varianza > 0'] * 100, 1, False)}% de ellas (en el "
        f"{_pp(rl.loc[KG, 'timing_varianza > 0'] * 100, 1, False)}% para el ½ Kelly); el de la media "
        f"oscila entre {_pp(rl.loc[L3, 'timing_media mín (pp)'])} y "
        f"{_pp(rl.loc[L3, 'timing_media máx (pp)'])} pp y es positivo en el "
        f"{_pp(rl.loc[L3, 'timing_media > 0'] * 100, 1, False)}%. En la tabla de significación, la t del "
        f"timing de la varianza va de {_pp(ts['t timing_varianza'].min())} a "
        f"{_pp(ts['t timing_varianza'].max())} según estrategia y periodo; la del timing de la media, de "
        f"{_pp(ts['t timing_media'].min())} a {_pp(ts['t timing_media'].max())}. Concentración diaria en "
        f"la LRS 3x: los diez días que más aportan al timing de la varianza suman {_pp(c2['10 mejores'])} "
        f"de sus {_pp(c2['total'])} pp ({_pp(c2['% en los 10 mejores'], 0, False)}%; los tres primeros, "
        f"{', '.join(A['top_days']['t2_dias'])}) y los diez que más restan, {_pp(c2['10 peores'])}. En "
        f"el timing de la media, los diez mejores suman {_pp(c1['10 mejores'])} de {_pp(c1['total'])} "
        f"({_pp(c1['% en los 10 mejores'], 0, False)}%), los veinte mejores {_pp(c1['20 mejores'])}, los "
        f"diez peores {_pp(c1['10 peores'])} y los veinte peores {_pp(c1['20 peores'])}. Sin los diez "
        f"mejores quedaría {_pp(c1['sin los 10 mejores'])}, pero quitar sólo la cola favorable es "
        f"asimétrico: sin los diez mejores ni los diez peores, {_pp(c1['sin los 10 mejores ni los 10 peores'])}. "
        f"Quitando las N sesiones de mayor |e| (N = {n_grid}, rejilla entera en la sección 1, con L̄ "
        f"recalculada), el timing de la media queda en {tr_s('timing_media')} pp y la Δg en "
        f"{tr_s('Δg')}; el de la varianza, {tr_s('timing_varianza')}, porque vive "
        "en los días de e² grande, aunque repartido entre muchos de ellos. "
        + ("En la LRS 3x ninguno de los dos términos cambia de signo al quitar esas sesiones"
           + ("; el de la media, sin embargo, no es significativo. " if not _sig_one(ts.loc[(L3, '1938–2026'), 't timing_media']) else ". ")
           if (tr[["timing_media", "timing_varianza"]].astype(float) > 0).all().all() else
           "En la LRS 3x algún término cambia de signo al quitar esas sesiones. ")
        + f"En la LRS-Kelly, su timing de la media, {trk(0)} pp, queda en "
        f"{trk(20)} sin las 20 sesiones de mayor |e| y en {trk(50)} sin las 50.")
    b = BT.loc[(f"{L3} (1938–)", "timing_media")]
    pmin = float(BT["p_asintótico"].min())
    out.append(
        f"5. **{'No hay ruptura que fechar' if pmin >= 0.05 else 'Una ruptura posible'}.** El timing de la media de la LRS 3x fue "
        f"{_pp(q(L3, '1938–1969', 'timing_media'))} pp en 1938–69, {_pp(q(L3, '1970–1991', 'timing_media'))} "
        f"en 1970–91, {_pp(q(L3, '1992–2026', 'timing_media'))} en 1992–2026 y "
        f"{_pp(q(L3, 'ETFs reales (2010–2026)', 'timing_media'))} con ETFs reales (2010–26). Parece una "
        f"ruptura en 1992, pero el sup-Wald de Andrews sobre la contribución mensual vale "
        f"{_pp(b['sup_Wald'], 2, False)} (p = {_pp(b['p_asintótico'], 3, False)}), con el máximo en "
        f"{b['máximo en']}, y el Wald en 1992-01 es {_pp(b['Wald en 1992-01'], 2, False)}. Para rechazar "
        f"una ruptura en 1992 haría falta un cambio de unos {_pp(b['cambio necesario en 1992 (pp)'], 1, False)} "
        f"pp en la media mensual de ese término (calculada con la L̄ de toda la muestra); el observado es "
        f"{_pp(b['media desde 1992 (pp)'] - b['media antes de 1992 (pp)'], 1)} pp. "
        + (f"Ningún término de ninguna estrategia tiene una ruptura significativa (p mínimo {_pp(pmin, 3, False)}). "
           "Con este ruido, «la LRS dejó de funcionar en 1992» y «la LRS sigue funcionando» son compatibles "
           "con los datos." if pmin >= 0.05 else
           f"El p mínimo de la tabla es {_pp(pmin, 3, False)}, sin corregir por los 12 contrastes."))
    jl = J.loc[f"{L3} (1938–)"]
    drops = jl["Δg"].drop(["ninguno", "todos a la vez"])
    worst = drops.idxmin()
    n_all = int(jl.loc["todos a la vez", "días_quitados"])
    out.append(
        f"6. **{'La ventaja vive en los grandes mercados bajistas' if jl.loc['todos a la vez', 'Δg'] < 0 else 'Los grandes mercados bajistas'}.** "
        + ("Ningún episodio aislado decide: el que " if (drops > 0).all() else "Un episodio basta para cambiar el signo: el que ")
        + f"más pesa, {worst}, baja la Δg de la LRS 3x de {_pp(jl.loc['ninguno', 'Δg'] * 100)} a "
        f"{_pp(jl.loc[worst, 'Δg'] * 100)} pp. Sin los seis a la vez ({n_all} sesiones, el "
        f"{_pp(n_all / len(A['P38'][L3]) * 100, 1, False)}% de la muestra) queda en "
        f"{_pp(jl.loc['todos a la vez', 'Δg'] * 100)} pp: timing de la media "
        f"{_pp(jl.loc['todos a la vez', 'timing_media'] * 100)}, de la varianza "
        f"{_pp(jl.loc['todos a la vez', 'timing_varianza'] * 100)}, dispersión "
        f"{_pp(jl.loc['todos a la vez', 'dispersión'] * 100)}. El contraste es asimétrico a propósito "
        "(quita justo los años en que un filtro de tendencia debe lucirse), así que no dice que la LRS "
        "no funcione: dice que toda su ventaja depende de que el horizonte del inversor contenga alguno "
        "de esos episodios. A diferencia del recorte por |e| del punto 4, que quita sesiones sueltas "
        "de los dos signos, aquí se quitan años enteros de mercado bajista, con sus días tranquilos.")
    p3 = PT.loc[(L3, "deciles de varianza GJR")]
    p3m = PT.loc[(L3, "deciles de varianza GJR, persistente")]
    p3c = PT.loc[(L3, "desplazamiento circular")]
    p3d = PT.loc[(L3, "deciles × década")]
    pk = PT.loc[(LK, "deciles de varianza GJR")]
    pkm = PT.loc[(LK, "deciles de varianza GJR, persistente")]
    pmax3 = float(PT.loc[L3, "p_valor"].max())
    out.append(
        ("7. **La LRS no gana solo por esquivar volatilidad, pero ganar a un placebo es un listón bajo.** "
         if max(p3["p_valor"], p3m["p_valor"]) < 0.05 else
         "7. **Frente a un placebo que esquiva la volatilidad igual, la ventaja no es clara.** ")
        + f"Barajando el estado dentro de cada decil de varianza prevista, el placebo esquiva la "
        f"volatilidad como la señal real y conserva buena parte de su timing de la varianza "
        f"({_pp(p3['timing_varianza placebo'])} frente a {_pp(p3['timing_varianza real'])} pp), pero su "
        f"timing de la media es {_pp(p3['timing_media placebo'])} pp: los días de varianza alta preceden "
        f"tanto caídas como rebotes, y salir en todos ellos cuesta rentabilidad. La señal real logra "
        f"{_pp(p3['timing_media real'])} pp. Δg bruto de rotación: {_pp(p3['Δg_real (pp)'])} pp real "
        f"frente a {_pp(p3['Δg_placebo_media (pp)'])} de media en los placebos (percentil 95: "
        f"{_pp(p3['p95 (pp)'])}), p = {_pp(p3['p_valor'], 4, False)}. El barajado diario destruye la "
        f"persistencia del estado ({_pp(p3['cambios de estado/año placebo'], 0, False)} cambios al año "
        f"frente a {_pp(p3['cambios de estado/año real'], 1, False)}); por eso, tras verlo, se añadieron "
        f"dos controles: con un placebo que conserva la persistencia y la relación con los deciles, p = "
        f"{_pp(p3m['p_valor'], 4, False)}; con la propia serie de estados desplazada en el tiempo, p = "
        f"{_pp(p3c['p_valor'], 4, False)}; conservando decil y década, p = {_pp(p3d['p_valor'], 4, False)} "
        f"(ningún esquema pasa de p = {_pp(pmax3, 4, False)}). LRS-Kelly: p = "
        f"{_pp(pk['p_valor'], 4, False)} y {_pp(pkm['p_valor'], 4, False)}. La clave está en la media de "
        f"los placebos, {_pp(p3m['Δg_placebo_media (pp)'])} pp: un timing sin información paga el coste "
        f"de dispersión, así que batir al azar no es lo mismo que batir al apalancamiento constante "
        f"(t = {_pp(ts.loc[(L3, '1938–2026'), 't Δg'])}, punto 2). Lo que el placebo sí dice es que, en "
        "esta muestra y dentro de un mismo nivel de volatilidad prevista, la señal estuvo fuera en "
        "sesiones de rentabilidad más baja que una señal al azar. El término en que se nota, el timing "
        "de la media, se juega en buena parte en unas pocas sesiones extremas de los dos signos (punto "
        "4), y eso no basta para afirmar que la señal prediga la rentabilidad (punto 9).")
    SP, NQ, SC = "S&P 500 (desde 1938)", "Nasdaq-100 (desde 1985)", "Semis (desde 2004)"
    V = lambda mkt, i: VT.loc[(mkt, VSPECS[i])]
    sv, sq, sl = V(SP, 0), V(SP, 1), V(SP, 2)
    vs = A["VS"].loc[SP]
    so, sb = vs.loc["sobre la media"], vs.loc["bajo la media"]
    cells = A["VF"]["S&P 500"]["n"]
    chg = lambda c: (f"{_pp(abs(np.exp(c) - 1) * 100, 1, False)}% "
                     + ("menor" if c < 0 else "mayor"))
    nov = lambda r: (f"entre {_pp(r['t mensual mín'])} y {_pp(r['t mensual máx'])}, con mediana "
                     f"{_pp(r['t mensual mediana'])}")
    pct = lambda v: f"{_pp(v * 100, 0, False)}%"
    raw_ok = not _sig_two(sq["t HH 20"])
    out.append(
        "8. **¿Sabe la media de 200 algo de la varianza que el GJR no sepa?"
        + (" Frente al GJR tal cual, nada; frente a un GJR recalibrado ex post, algo, sin que el R² se mueva.** "
           if raw_ok and _sig_two(sv["t HH 20"]) else "** ")
        + f"Frente a la previsión GJR tal cual{', nada' if raw_ok else ''}: en el S&P 500, log(RV/GJR) es un {chg(sq['coef_sobre_media'])} "
        f"sobre la media que bajo ella (t HH = {_pp(sq['t HH 20'])}; con muestras mensuales no "
        f"solapadas, {nov(sq)}). La media geométrica de RV/GJR es {_pp(so['media geométrica RV/GJR'], 3, False)} "
        f"sobre la media y {_pp(sb['media geométrica RV/GJR'], 3, False)} bajo ella; el cociente de medias, "
        f"{_pp(so['cociente de medias RV/GJR'], 3, False)} y {_pp(sb['cociente de medias RV/GJR'], 3, False)}"
        + (": sobre la media llega algo más de la varianza prevista, no menos. " if
           so['cociente de medias RV/GJR'] > sb['cociente de medias RV/GJR'] else ". ")
        + "La diferencia sólo aparece al recalibrar el GJR: la pendiente de log RV sobre log GJR es "
        f"{_pp(sv['coef_control'], 2, False)}, no 1 (la previsión exagera las subidas y bajadas de la "
        "varianza), y con esa pendiente, estimada con toda la muestra, a igual previsión la varianza "
        f"realizada es un {chg(sv['coef_sobre_media'])} (media geométrica) sobre la media que bajo ella. "
        f"Su t es {_pp(sv['t NW 42 (versión anterior)'])} con Newey-West de 42 retardos (la cifra de la "
        f"versión anterior de este informe), {_pp(sv['t HH 20'])} con Hansen-Hodrick y "
        f"{_pp(sv['t NW 252'])} con 252 retardos; con muestras mensuales no solapadas, {nov(sv)}: los "
        "residuos de log RV tienen memoria larga y 42 retardos no bastan. En niveles, donde la pendiente "
        f"recalibrada es aún menor ({_pp(sl['coef_control'], 2, False)}), el estado resta "
        f"{_pp(abs(sl['coef_sobre_media']), 4, False)} de varianza anual (el "
        f"{_pp(abs(sl['coef_sobre_media']) / sl['media de y'] * 100, 0, False)}% de la varianza realizada "
        f"media) y aguanta mejor: t HH {_pp(sl['t HH 20'])}, NW 252 {_pp(sl['t NW 252'])}, mensual "
        f"{nov(sl)}; pero en niveles pesan mucho los pocos meses de varianza extrema, con colas tan gruesas "
        "que ningún error HAC es del todo fiable. En logaritmos, "
        f"el R² sólo pasa de {_pp(sv['R²_sin_estado'], 3, False)} a {_pp(sv['R²'], 3, False)}; en niveles, "
        f"de {_pp(sl['R²_sin_estado'], 3, False)} a {_pp(sl['R²'], 3, False)}. Por celdas (cociente de sumas, "
        f"previsión sin recalibrar): en los deciles 8–10 de varianza prevista, {pct(so['deciles 8–10: RV/GJR'])} "
        f"sobre la media ({int(so['deciles 8–10: días'])} días) frente a {pct(sb['deciles 8–10: RV/GJR'])} "
        f"bajo ella ({int(sb['deciles 8–10: días'])}); en los deciles 1–7, {pct(so['deciles 1–7: RV/GJR'])} "
        f"({int(so['deciles 1–7: días'])}) frente a {pct(sb['deciles 1–7: RV/GJR'])} "
        f"({int(sb['deciles 1–7: días'])}). Las celdas de varianza baja bajo la media son pequeñas (decil "
        f"1: {int(cells.loc[1, 0.0])} días solapados, unos {int(cells.loc[1, 0.0] // H)} meses "
        f"independientes; decil 2: {int(cells.loc[2, 0.0])}). En el Nasdaq-100 y en semiconductores, "
        f"nada ni con la pendiente libre (t HH {_pp(V(NQ, 0)['t HH 20'])} y {_pp(V(SC, 0)['t HH 20'])}) "
        f"ni fijada en 1 ({_pp(V(NQ, 1)['t HH 20'])} y {_pp(V(SC, 1)['t HH 20'])}).")
    sr, sa = V(SP, 3), V(SP, 4)
    nr, xr = V(NQ, 3), V(SC, 3)
    na, xa = V(NQ, 4), V(SC, 4)
    n_reg = len(VT)
    bonf = min(1.0, float(sr["p HH"]) * n_reg)
    t_unc = float(np.nanmax([abs(float(r["t HH 20"])) for r in (sa, na, xa)]))
    so_g, sb_g = so["volatilidad prevista mediana (%)"], sb["volatilidad prevista mediana (%)"]
    out.append(
        f"9. **¿Predice el estado la rentabilidad del mes siguiente? {'En algún mercado' if _sig_two(t_unc) else 'No de forma robusta'}.** Contraste "
        "exploratorio, añadido después de ver el placebo. En el S&P 500, estar sobre la media anticipa "
        f"{_pp(sa['coef_sobre_media'] * 100, 1)} pp anualizados de exceso en las 21 sesiones siguientes "
        f"(t HH = {_pp(sa['t HH 20'])}; con muestras mensuales no solapadas, {nov(sa)}). Con log GJR "
        f"como control sube a {_pp(sr['coef_sobre_media'] * 100, 1)} pp: t = "
        f"{_pp(sr['t NW 42 (versión anterior)'])} con Newey-West de 42 retardos (la cifra anterior), "
        f"{_pp(sr['t HH 20'])} con Hansen-Hodrick y {_pp(sr['t NW 252'])} con 252 retardos; con muestras "
        f"mensuales no solapadas, {nov(sr)}, y un coeficiente de {_pp(sr['coef mensual mín'] * 100, 1)} a "
        f"{_pp(sr['coef mensual máx'] * 100, 1)} pp según el día de corte. La subida es un efecto de "
        "supresión: la previsión GJR predice por sí misma más rentabilidad (coeficiente de log GJR "
        f"{_pp(sr['coef_control'], 3)}, t HH = {_pp(sr['t_control (HH)'])}) y es más alta bajo la media "
        f"(volatilidad prevista mediana {_pp(sb_g, 1, False)}% frente a {_pp(so_g, 1, False)}%), así que "
        "controlar por ella destapa parte del coeficiente del estado. Con Bonferroni sobre las "
        f"{n_reg} regresiones de la tabla, p = {_pp(bonf, 2, False)}. Nasdaq-100: "
        f"{_pp(nr['coef_sobre_media'] * 100, 1)} pp con control (t HH = {_pp(nr['t HH 20'])}); "
        f"semiconductores: {_pp(xr['coef_sobre_media'] * 100, 1)} (t HH = {_pp(xr['t HH 20'])}). La LRS "
        "opera el estado sin control, y ese "
        + ("solo es significativo en algún mercado " if _sig_two(t_unc) else "no es significativo en ningún mercado ")
        + f"(|t HH| máximo {_pp(t_unc, 2, False)}).")
    s1 = ST.loc[("LRS 3x ~ 1x", "1938–2026")]
    s2 = ST.loc[("LRS 3x ~ 1x + ½ Kelly GJR", "1938–2026")]
    s3 = ST.loc[("½ Kelly GJR ~ 1x + LRS 3x", "1938–2026")]
    s4 = ST.loc[("LRS 3x ~ 1x", "1938–1969")]
    s5 = ST.loc[("½ Kelly GJR ~ 1x", "1938–2026")]
    s6 = ST.loc[("LRS-Kelly ~ 1x + ½ Kelly GJR", "1938–2026")]
    later = ST.loc[[("LRS 3x ~ 1x", p) for p in list(PERIODS)[2:]], "t_NW(α)"]
    out.append(
        f"10. **Spanning (mensual, después de costes).** En 1938–2026 la LRS 3x tiene α = "
        f"{_pp(s1['α anual (pp)'])} pp al año sobre el 1x (t = {_pp(s1['t_NW(α)'])}) y "
        f"{_pp(s2['α anual (pp)'])} (t = {_pp(s2['t_NW(α)'])}) sobre el 1x y el ½ Kelly juntos; al "
        f"revés, el ½ Kelly no añade nada a 1x + LRS (α = {_pp(s3['α anual (pp)'])}, t = "
        f"{_pp(s3['t_NW(α)'])}). La LRS-Kelly añade {_pp(s6['α anual (pp)'])} pp al 1x + ½ Kelly (t = "
        f"{_pp(s6['t_NW(α)'])}). Pero la α de la LRS 3x se concentra en 1938–69 "
        f"({_pp(s4['α anual (pp)'])}, t = {_pp(s4['t_NW(α)'])}); en 1970–91, 1992–2026 y 2010–26 su t "
        f"es {', '.join(_pp(v) for v in later.values)}. El ejercicio clásico de Moreira y Muir (½ Kelly "
        f"sobre el 1x) da α = {_pp(s5['α anual (pp)'])} (t = {_pp(s5['t_NW(α)'])}).")
    e_ = {s: T.loc[(s, "ETFs reales (2010–2026)")] for s in STRATS}
    all_lose = all(e_[s]["Δg"] < 0 for s in STRATS)
    tm_neg = all(e_[s]["timing_media"] < 0 for s in STRATS)
    out.append(
        "11. **Con ETFs reales (2010–2026)"
        + (", las tres pierden frente a su apalancamiento constante:** " if all_lose else
           ", frente a su apalancamiento constante:** ")
        + ", ".join(f"{s} {_pp(e_[s]['Δg'] * 100)}" for s in STRATS) + " pp al año (t = "
        + ", ".join(_pp(ts.loc[(s, "ETFs reales (2010–2026)"), "t Δg"]) for s in STRATS)
        + "). El timing de la varianza es " + ", ".join(_pp(e_[s]['timing_varianza'] * 100) for s in STRATS)
        + " pp y el de la media " + ", ".join(_pp(e_[s]['timing_media'] * 100) for s in STRATS)
        + " pp (t = " + ", ".join(_pp(ts.loc[(s, "ETFs reales (2010–2026)"), "t timing_media"]) for s in STRATS)
        + ")"
        + (": en un mercado alcista con correcciones que no llegaron a mercado bajista, el filtro (o el "
           "recorte por volatilidad del ½ Kelly) sale después de la caída y vuelve a entrar después del "
           "rebote" if tm_neg else "")
        + ". En la LRS 3x, los cinco años de peor timing de la media: "
        + "; ".join(f"{y} {_pp(v)}" for y, v in A["yr_etf"].items()) + " pp.")
    n3 = To.loc[("Nasdaq-100", L3, "desde 1985")]
    x3 = To.loc[("Semis", L3, "desde 2004")]
    out.append(
        f"12. **Otros mercados.** Nasdaq-100 desde 1985: la LRS 3x {gana(n3['Δg'])} "
        f"{_pp(abs(n3['Δg']) * 100, 2, False)} pp, con timing de la media {_pp(n3['timing_media'] * 100)}, "
        f"de la varianza {_pp(n3['timing_varianza'] * 100)} y dispersión {_pp(n3['dispersión'] * 100)}. "
        f"Semiconductores desde 2004: {_pp(x3['Δg'] * 100)} pp, con timing de la media "
        f"{_pp(x3['timing_media'] * 100)}, de la varianza {_pp(x3['timing_varianza'] * 100)} y una "
        f"dispersión de {_pp(x3['dispersión'] * 100)} pp (volatilidad del subyacente mucho mayor).")
    return out


TITULAR_PLACEBOS = (("deciles de volatilidad", "deciles de varianza GJR"),
                    ("Markov", "deciles de varianza GJR, persistente"),
                    ("desplazamiento circular", "desplazamiento circular"),
                    ("deciles × década", "deciles × década"))


def titular_rows(A: dict) -> list[dict]:
    """Los p de titular de R16 (lista cerrada; los lee R19 desde p_titular.csv), todos de la
    LRS 3x original en el S&P 500, 1938–2026: los cuatro placebos (p de aleatorización),
    Δg frente a su L̄ constante y los timings de la media y de la varianza (t de Newey-West
    de las contribuciones mensuales → p normal unilateral), la α de expansión sobre el 1x
    (spanning, unilateral) y el sup-Wald del timing de la media (acotado por 1/(N_NULL + 1))."""
    L3 = STRATS[0]
    PT, ts = A["PT"], A["TS"].loc[(L3, "1938–2026")]
    s1 = A["ST"].loc[("LRS 3x ~ 1x", "1938–2026")]
    b = A["BT"].loc[(f"{L3} (1938–)", "timing_media")]
    rows = [(f"placebo: {lab} (Δg LRS 3x)", PT.loc[(L3, sch), "p_valor"]) for lab, sch in TITULAR_PLACEBOS]
    rows += [("Δg LRS 3x frente a su L̄ constante (t NW mensual)", sps.norm.sf(ts["t Δg"])),
             ("timing de la media LRS 3x (t NW mensual)", sps.norm.sf(ts["t timing_media"])),
             ("timing de la varianza LRS 3x (t NW mensual)", sps.norm.sf(ts["t timing_varianza"])),
             ("α de expansión LRS 3x sobre el 1x (t NW)", sps.norm.sf(s1["t_NW(α)"])),
             ("ruptura del timing de la media LRS 3x (sup-Wald)", max(b["p_asintótico"], 1 / (N_NULL + 1)))]
    return [{"módulo": "R16", "mercado": "S&P 500", "contraste": c, "p": float(p)} for c, p in rows]


def alphas_1938_69(ST: pd.DataFrame, alpha: float = 0.05) -> str:
    """Frase de las salvedades con el rango de p de las α de spanning de 1938–1969 con p < alpha
    (bilateral, sin corregir), calculado a partir de la tabla."""
    p = ST.xs("1938–1969", level=1)["p(α)"].astype(float)
    p = p[p < alpha]
    if p.empty:
        return "ninguna α de 1938–69 tiene p < 0,05, y las demás α deben leerse con esa cautela"
    if len(p) == 1:
        return f"la α con p ≈ {_pp(p.iloc[0], 2, False)} de 1938–69 debe leerse con esa cautela"
    lo, hi = _pp(p.min(), 2, False), _pp(p.max(), 2, False)
    rng = lo if lo == hi else f"{lo}–{hi}"
    return f"las {len(p)} α con p ≈ {rng} de 1938–69 deben leerse con esa cautela"


def caveats(A: dict) -> str:
    sv = A["VT"].loc[("S&P 500 (desde 1938)", VSPECS[0])]
    rho = {f"rho{l}": _pp(sv[f"ρ residuo {l}"], 2, False) for l in RHO_LAGS}
    return CAVEATS.format(r38=_pp(A["gjr_bias"]["1938–1969"] * 100, 0, False),
                          rall=_pp(A["gjr_bias"]["1938–2026"] * 100, 0, False),
                          n_span=len(A["ST"]), n_var=len(A["VT"]), n_break=len(A["BT"]),
                          alphas=alphas_1938_69(A["ST"]), **rho)


CAVEATS = """
- **Todo es dentro de muestra.** La media de 200 sesiones sobre la rentabilidad
  total viene de una literatura que ya conocía esta historia (Gayed y Bilello
  2016 usan datos hasta 2015); la
  fracción ½ de Kelly y el GJR se eligieron en R10–R12 con el mismo S&P 500. El
  placebo, el spanning y las regresiones contrastan lo que pasó, no lo que se
  podía saber.
- **L̄ es ex post.** La gemela constante necesita conocer la exposición media
  futura de la estrategia; es una herramienta contable, no una alternativa
  invertible. Se reajusta a diario sin coste de rotación (como un fondo L̄x) y
  paga el mismo coste por unidad de exposición que el 3x.
- **La identidad es contabilidad, no causalidad.** Los términos son momentos
  diarios; leer (1) como «capacidad de predecir la media» supone que la
  covarianza no se debe a unos pocos días. Las 20 mejores sesiones de (1) en la
  LRS 3x suman más que todo el término (caídas con la LRS fuera) y las 20 peores
  casi lo mismo en negativo (rebotes con la LRS también fuera); el término
  resiste a quitar las sesiones de mayor |e| de los dos signos, pero el
  jackknife muestra que depende de unos pocos episodios bajistas, y un día de
  crash con la LRS fuera suma a la vez en (1) y en (2): la frontera entre
  «media» y «varianza» la pone la expansión, no la economía.
- **Placebo.** Deciles cortados con toda la muestra; la permutación diaria
  destruye la persistencia del estado, así que contrasta «sólo importa la
  volatilidad prevista», no «persistencia sin información»; bruto de rotación
  (el placebo cambiaría de posición casi a diario). En la LRS-Kelly, placebo y
  referencia se calculan sin la regla de reajuste mínimo.
- **Jackknife por años naturales**: incluye el rebote dentro del mismo año, no
  el de años siguientes (1975, 2003). Quita justo los años favorables a un
  filtro de tendencia.
- **Ruptura.** Las contribuciones mensuales tienen colas muy gruesas (1987,
  2008); la distribución asintótica del sup-Wald es una aproximación y el
  contraste tiene poca potencia: no encontrar ruptura no prueba estabilidad.
- **Muchos contrastes y algunos a posteriori.** {n_span} regresiones de spanning, {n_var}
  de varianza y rentabilidad y {n_break} de ruptura, sin corrección por pruebas
  múltiples salvo donde se indica: {alphas}. Los placebos persistentes, las regresiones de la
  rentabilidad y todas las comprobaciones de robustez de las secciones 1 y 6 se
  añadieron después de ver resultados (ver el protocolo) y son exploratorias.
- **Errores de las regresiones a 21 días.** Los objetivos se solapan y los
  residuos de log RV del S&P 500 tienen memoria larga (autocorrelación {rho21}
  a 21 sesiones, {rho63} a 63, {rho126} a 126 y {rho252} a 252): Newey-West con 42
  retardos infravalora la varianza.
  Se dan también Hansen-Hodrick, Newey-West con 252 retardos y muestras
  mensuales no solapadas; con estas últimas el resultado depende del día de
  corte, y se da el rango de los 21 posibles.
- **Recalibración.** En la regresión de log RV con la pendiente de log GJR libre,
  esa pendiente se estima con toda la muestra: el contraste pregunta si el estado
  añade algo a una previsión GJR corregida ex post, no a la que tenía el inversor.
- **El GJR sobrestima la varianza en el S&P 500**, sobre todo en 1938–69: la
  varianza realizada media de las 21 sesiones siguientes es el {r38}% de la
  prevista ({rall}% en 1938–2026), porque la estimación en ventana creciente
  arrastra la de los años treinta. No afecta a
  la identidad, pero sí a la exposición del ½ Kelly, que en esas décadas fue
  menor de la que el nombre sugiere.
- **Datos.** S&P 90 antes de 1957 y dividendos de Shiller hasta 1988; el 3x es
  sintético hasta 2009 y real (UPRO, TQQQ, SOXL) después, con su error de
  seguimiento dentro de κ; el Nasdaq-100 anterior a 1985 (una sola
  reconstrucción) sólo calienta los modelos.
""".strip("\n")


def run():
    rep = Report("r16_mechanism", "R16 — Mecanismo: de dónde sale la ventaja de la exposición variable")
    rep.p(__doc__.split("\n", 2)[2])
    rng = np.random.default_rng(SEED)
    A = analyse(rng)
    T, L3 = A["T"], STRATS[0]
    tit = provenance.write_titular(rep.dir, titular_rows(A), "r16_mechanism")

    rep.h("Resumen")
    rep.p("\n\n".join(_summary(A)))
    rep.h("Salvedades", 3)
    rep.p(caveats(A))

    rep.h("La identidad")
    rep.p(DERIVATION)
    rep.p(PROTOCOL)

    rep.h("1. La descomposición: S&P 500 por subperiodos")
    rep.p("Crecimiento logarítmico anual en puntos porcentuales. «g_constante» es el de la misma "
          "exposición media mantenida fija con el mismo 3x; «suma_1_4», la suma de los cuatro "
          "términos; «resto», lo que le falta para llegar al Δg exacto (la comprobación numérica). "
          "Cada fila usa la L̄ de su propio periodo, así que los subperiodos no suman el total.")
    Tall = pd.concat([T, A["T28"]])
    rep.table(_flat(_to_pp(Tall[SHOW], PPCOLS)), digits=2)
    rep.h("¿Son los términos distintos de cero?", 3)
    rep.p("Media de la contribución mensual de cada término (con la L̄ de su periodo; coincide con la "
          f"tabla anterior) y su t de Newey-West ({NW_MONTHLY} retardos). |t| > 1,96 equivale a "
          "p < 0,05 bilateral. Los p de titular de Δg y de los dos timings de la LRS 3x (los que lee "
          "R19) son unilaterales, a favor de la estrategia (t > 1,645 equivale a p < 0,05), y el "
          "resumen usa esa misma regla.")
    rep.table(_flat(A["TS"]), digits=2)
    rep.p("neto_varianza: timing de la varianza más coste de dispersión, con contribución diaria "
          "−½(L² − L̄²)e², cuya media es exactamente la suma de los dos términos. Es la parte de segundo "
          "orden que la estrategia gana en neto al moverse entre exposiciones; no entra en "
          "p_titular.csv, así que la familia de 40 contrastes de R19 no cambia.")
    bp = A["boot_terms"]
    rep.h("El p del timing de la varianza con bootstrap por bloques (LRS 3x, 1938–2026)", 3)
    rep.table(bp, digits=4)
    rep.p("Añadido tras una revisión externa. El p de titular de cada término (el que lee R19) es la "
          "aproximación normal unilateral de su t de Newey-West. Estos términos se construyen con "
          "rentabilidades diarias al cuadrado, de colas muy gruesas, así que aquí se da también el p "
          f"unilateral de un bootstrap estacionario de las contribuciones mensuales ({num_es(B_BOOT_TERM, 0)} "
          "remuestreos, bloques medios de "
          + " y ".join(f"{b}" for b in BOOT_BLOCKS)
          + " meses): la fracción de medias remuestreadas y centradas que igualan o superan la "
          f"observada. «×{N_TITULAR}»: ese p por {N_TITULAR}, el primer escalón de Holm si fuera el menor de "
          f"los {N_TITULAR} contrastes principales de R19 (topado en 1).")
    rep.h("Financiación y costes, separados", 3)
    rep.p("−Cov(L, κ): estar dentro cuando financiarse es barato (positivo) o caro (negativo); "
          "rotación: 5 pb por unidad de exposición que cambia de manos.")
    rep.table(_flat(_to_pp(Tall[["financiación", "rotación"]], ["financiación", "rotación"])), digits=3)
    fig_bars(T, rep)
    rep.h("¿Cuántas sesiones sostienen cada término?", 3)
    rep.p("Añadido tras una revisión externa (exploratorio). Contribución diaria de cada sesión al "
          "término, en pp al año (1938–2026, L̄ de toda la muestra): suma total, suma de los 10 y 20 "
          "días que más aportan y que más restan, y cuántos de esos 20 días tuvieron la exposición por "
          "debajo de su media (L < L̄, «fuera» en la LRS 3x) con el índice cayendo o subiendo. En la "
          "LRS 3x, los días extremos del timing de la media son caídas esquivadas (a favor) y rebotes "
          "perdidos (en contra), con la LRS fuera en ambos casos.")
    rep.table(_flat(A["CONC"]), digits=2)
    rep.p("Las diez mejores y las diez peores sesiones del timing de la media de la LRS 3x (pp al año "
          "que aporta cada una): a favor, " + "; ".join(f"{t} {_pp(v)}" for t, v in A["top_days"]["t1_top"])
          + ". En contra, " + "; ".join(f"{t} {_pp(v)}" for t, v in A["top_days"]["t1_bot"]) + ".")
    rep.p("Descomposición sin las N sesiones de mayor |e| (caídas y subidas por igual), con L̄ "
          f"recalculada; rejilla N = {', '.join(map(str, TRIM_N))} fijada antes de calcularla y mostrada "
          "entera. Quitar sesiones rompe la composición diaria de la estrategia, así que es una "
          "comprobación de sensibilidad, no una estrategia.")
    tc = ["días_quitados", "L̄", "Δg", *TERMS[:4], "resto"]
    rep.table(_flat(_to_pp(A["TRIM"][tc], ["Δg", *TERMS])), digits=2)
    rep.h("Otros mercados (misma descomposición)", 3)
    rep.p("Nasdaq-100 evaluado sólo con datos reales (desde 1985-10; la reconstrucción de 1971–85 "
          "sólo calienta la media y el GJR) y semiconductores desde 2004 (diez años de calentamiento "
          "desde 1994).")
    rep.table(_flat(_to_pp(A["To"][SHOW], PPCOLS)), digits=2)

    rep.h("Sensibilidad: la LRS 3x con la señal de precio", 3)
    rep.p(f"La misma descomposición para {PX} (la media sobre el índice de precio, variante del "
          "laboratorio) frente a la original, en la misma muestra común, con la L̄ de cada una:")
    rep.table(_flat(_to_pp(A["TPX"][SHOW], PPCOLS)), digits=2)
    rep.table(_flat(A["TSPX"]), digits=2)
    jp = A["JPX"]
    rep.p("Jackknife de la señal de precio (Δg y timing de la media sin cada episodio, pp): "
          + "; ".join(f"{ep} {_pp(jp.loc[ep, 'Δg'] * 100)} / {_pp(jp.loc[ep, 'timing_media'] * 100)}"
                      for ep in jp.index) + ".")
    a, b = A["TPX"].loc[(L3, "1938–2026")], A["TPX"].loc[(PX, "1938–2026")]
    rep.p(f"En 1938–2026 la señal de precio gana a su constante {_pp(b['Δg'] * 100)} pp al año, frente a "
          f"{_pp(a['Δg'] * 100)} de la original; la diferencia está sobre todo en el timing de la media "
          f"({_pp(b['timing_media'] * 100)} frente a {_pp(a['timing_media'] * 100)} pp).")

    rep.h("2. Ventana móvil de diez años")
    rep.p(f"Ventanas de {ROLL} sesiones que avanzan de {ROLL_STEP} en {ROLL_STEP}, cada una con su "
          "propia L̄; se solapan, así que las proporciones no son observaciones independientes.")
    rep.table(A["rl"], pct={"Δg > 0 (% de ventanas)", "timing_media > 0", "timing_varianza > 0"},
              digits=2)
    fig_rolling(A["Rl"], rep)

    rep.h("3. Jackknife: quitar cada gran episodio")
    rep.p("Se quitan los años naturales completos de cada episodio y se recalcula todo, incluida L̄. "
          "1929–32 sólo existe en la muestra de la LRS 3x desde 1928 (las otras dos necesitan diez "
          "años de calentamiento).")
    jc = ["días_quitados", "L̄", "Δg", *TERMS[:4]]
    rep.table(_flat(_to_pp(A["J"][jc], ["Δg", *TERMS[:4]])), digits=2)

    rep.h("4. Placebo condicional: ¿gana la LRS sólo por esquivar volatilidad?")
    rep.p("El estado sobre/bajo la media de 200 sesiones se sustituye por uno sin información sobre "
          "las rentabilidades. Barajado día a día entre días del mismo grupo: (a) «sin condicionar», "
          "todos los días (timing al azar con la misma fracción de tiempo dentro); (b) «deciles de "
          "varianza GJR», días del mismo decil de varianza prevista (el placebo esquiva la "
          "volatilidad prevista igual que la señal real): el esquema pedido; (c) «deciles × década», "
          "además en la misma década. Como el barajado diario destruye la persistencia del estado, dos "
          "controles más: (d) «persistente», una cadena de Markov con la misma probabilidad de estar "
          "dentro que la señal real según el decil de hoy y el estado de ayer; (e) «desplazamiento "
          "circular», la serie real de estados desplazada entre uno y 87 años (rachas idénticas, sin "
          "condicionar a la volatilidad). En la LRS-Kelly la exposición es 3·(½ Kelly)·estado, sin la "
          "regla de reajuste mínimo, tanto en el placebo como en su «real». Todo bruto de rotación "
          f"(desde {A['pl'][L3]['desde']}); Δg y términos en pp al año; p-valor unilateral con "
          f"{N_PERM} placebos por esquema (el mínimo posible es {_pp(1 / (N_PERM + 1), 4, False)}).")
    PT = A["PT"].copy()
    PT["p_valor"] = PT["p_valor"].map(lambda v: f"{v:.4f}")
    rep.table(_flat(PT), digits=2)
    fig_placebo(A["pl"], rep)

    rep.h("5. Spanning a lo Moreira–Muir")
    rep.p("Excesos mensuales sobre el T-bill (compuestos), después de costes; Newey-West con "
          f"{NW_MONTHLY} retardos. α > 0 y significativa: la estrategia de la izquierda amplía la "
          "frontera media-varianza de las de la derecha (Huberman y Kandel 1987; Moreira y Muir "
          "2017). Ratio de valoración = α/σ(ε)·√12. Es un criterio de Sharpe, no de crecimiento: una "
          "estrategia puede tener α positiva y crecer menos que otra si su volatilidad es mayor.")
    ST = A["ST"].copy()
    ST["β 2º regresor"] = ST["β 2º regresor"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    rep.table(_flat(ST), digits=3)

    rep.h("6. ¿Predice el estado de la media de 200 sesiones la varianza siguiente más allá del GJR?")
    rep.p("RV = varianza realizada anualizada de las 21 sesiones siguientes (t+1…t+21); GJR = su "
          "previsión con datos hasta t; «sobre la media» = 1 si el índice de rentabilidad total (la "
          "señal de la LRS original) cierra en t por encima de su media de 200 sesiones. Dos preguntas "
          "distintas. (a) Con la pendiente de log "
          "GJR libre («log RV ~ log GJR + sobre la media»), exp(coef) − 1 compara la varianza "
          "realizada (media geométrica) sobre y bajo la media A IGUAL PREVISIÓN GJR RECALIBRADA: la "
          "pendiente («coef_control») se estima con toda la muestra y no vale 1, así que no es la "
          "previsión que tenía el inversor. (b) Con la pendiente fijada en 1 («log(RV/GJR)»), compara "
          "el error de la previsión GJR tal cual en los dos estados. En niveles, el coeficiente está en "
          "varianza anual. Las dos últimas filas de cada mercado prueban lo mismo con el exceso de "
          "rentabilidad anualizado, con y sin el GJR como control; son exploratorias y se añadieron "
          "después de ver el placebo.")
    n_fb = int((A["VT"]["estimador HH"] != stats.hac_label(_HH["lags"], _HH["kernel"])).sum())
    rep.p(f"Cada t se da con cuatro errores: Newey-West de {NW_DAILY} retardos (el de la versión "
          f"anterior de este informe, que infravalora la varianza), Hansen-Hodrick de {HH_LAGS} "
          f"retardos (kernel uniforme; el natural para objetivos de {H} días que se solapan), "
          f"Newey-West de {NW_LONG} retardos (para la memoria larga de los residuos de log RV; ver "
          f"«ρ residuo») y regresiones con una observación de cada {H} sesiones (objetivos que no se "
          f"solapan, Newey-West de {NW_NONOV} retardos), de las que se da el mínimo, la mediana y el "
          f"máximo del t entre los {H} días de corte posibles. «p HH» es bilateral, con Hansen-Hodrick "
          f"y sin corregir por las {len(A['VT'])} regresiones de la tabla. Hansen-Hodrick no garantiza "
          f"una varianza positiva: si en una regresión no lo es, «t HH 20», «p HH» y «t_control (HH)» "
          f"usan Newey-West de {_HH['fallback'][0]} retardos (la convención de la serie para objetivos "
          f"solapados) y la columna «estimador HH» lo dice"
          + (f"; ocurre en {n_fb} de las {len(A['VT'])} regresiones. " if n_fb else
             "; aquí no ocurre en ninguna. ")
          + "Dentro de muestra.")
    VT = A["VT"]
    first = ["coef_sobre_media", *ERRORS, "p HH", "estimador HH", "t mensual mín", "t mensual mediana",
             "t mensual máx", "coef mensual mín", "coef mensual máx"]
    rep.table(_flat(VT[first]), digits=3)
    rep.p("Las mismas regresiones: el coeficiente del control y su t de Hansen-Hodrick, el R² con y sin "
          "el estado, el número de días y la autocorrelación de los residuos a 21, 63, 126 y 252 "
          "sesiones (con objetivos de 21 días, un residuo sin más memoria que el solapamiento tendría "
          "autocorrelación cero desde el retardo 21).")
    rep.table(_flat(VT[[c for c in VT.columns if c not in first]]), digits=3)
    rep.h("Cocientes agrupados", 3)
    rep.p("Varianza realizada entre varianza prevista por el GJR sin recalibrar: media geométrica del "
          "cociente diario y cociente de sumas (de medias), en toda la muestra y agrupando los deciles "
          "1–7 y 8–10 de la previsión (deciles cortados con toda la muestra de cada mercado), con el "
          "número de días solapados de cada celda (unos 21 forman un mes independiente).")
    rep.table(_flat(A["VS"]), digits=3)
    fig_variance(A["VF"], rep)

    rep.h("7. Ruptura estructural en el timing de la media (sup-Wald de Andrews)")
    c = A["crit"]
    rep.p("Contribución mensual de cada término, escalada para que su media sea el término anual; "
          "cambio de media en fecha desconocida entre el 15% y el 85% de la muestra, Newey-West con "
          f"{NW_MONTHLY} retardos. Valores críticos asintóticos simulados (p = 1): "
          f"{_pp(c[0.10], 2, False)} (10%), {_pp(c[0.05], 2, False)} (5%) y {_pp(c[0.01], 2, False)} "
          "(1%); la tabla 1 de Andrews (1993) da 7,12, 8,68 y 12,16 (la discretización en 1000 pasos "
          "rebaja un poco el cuantil del 5%, lo que hace el contraste algo menos conservador, no más). "
          "El Wald en 1992-01 no es un "
          "contraste válido (la fecha se eligió mirando los datos): se da para compararlo con el "
          "máximo. «Cambio necesario en 1992» es la diferencia de medias que habría hecho "
          "significativo al 5% el sup-Wald si la ruptura estuviera ahí (√valor crítico × error "
          "típico Newey-West de la diferencia).")
    rep.table(_flat(A["BT"]), digits=2)
    fig_wald(A["Bfig"], c[0.05], rep)

    rep.h("p de titular exportados (p_titular.csv)")
    rep.p("La lista cerrada de p de titular de este informe, que R19 lee para corregirlos junto a los de "
          "R17 y R18 (Holm, Benjamini–Hochberg y Benjamini–Yekutieli). Todos son de la LRS 3x original "
          f"en el S&P 500 (1938–2026) y unilaterales. Procedencia: commit {tit.commit.iloc[0][:12]}"
          + (" con cambios sin commit en el código del que depende" if tit.cambios_sin_commit.iloc[0] else "")
          + f"; huella del código {tit.huella_código.iloc[0]}.")
    rep.table(tit[["contraste", "p"]], index=False, digits=4)
    rep.save()
    return A


if __name__ == "__main__":
    run()
