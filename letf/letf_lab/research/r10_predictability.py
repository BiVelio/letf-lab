"""R10 — ¿Qué se puede predecir de un 3x? Rentabilidad esperada frente a varianza.

Marco. Con reset diario, el crecimiento log de un LETF de apalancamiento L en
una ventana es, a segundo orden,
    g(L) ≈ r + L·(μ − f − s) − ER − ½·L²·σ²
(μ, σ²: media aritmética y varianza anualizadas del índice con dividendos en
la ventana; r: letras del Tesoro (T-bill), donde está el efectivo del fondo;
f: tipo interbancario al que se financia toda la exposición, L veces el capital
(fondos federales con un mes de retraso; antes de 1954, call money); s: sobrecoste del swap
sobre f; ER: comisión). Acertar el momento de compra equivale a predecir μ y
σ² de la ventana siguiente. El L que maximiza g es el L* aproximado
    L* = (μ − f − s)/σ²
y, sin costes, g(2L*) = g(0) = r: a partir de 2L* el apalancamiento rinde
menos que las letras.

Base (la misma que R12–R20, r12.make): S&P 500 desde 1960 con los dividendos
de Shiller hasta 1988 y los cierres erróneos de ^GSPC corregidos; Nasdaq-100
real desde 1985-10 y semiconductores desde 1994, con un dividendo constante
antes de QQQ y SOXX. El 3x es el ETF real desde su lanzamiento y, antes, una
réplica financiada a f con s' = s_R3 − media(f − r) en la vida real del ETF.
Las reglas se deciden al cierre de t y se ejecutan al cierre de t+1
(EXEC_LAG = 2); el mismo cierre queda como sensibilidad (R11 §D).

Pruebas:
0. Base de datos y validación de la réplica del 3x (vida real de UPRO, TQQQ y
   SOXL, y la mitad de esa vida que no se usa para calibrar s').
A. La identidad: ¿cuánto del resultado realizado del 3x a 3 años explican μ y
   σ² de la propia ventana? Descomposición de la varianza de g entre ambos
   términos (con el rango que da asignar la covarianza a uno u otro), error por
   tramo (réplica o fondo real) y error que quitan los momentos 3.º y 4.º.
B. Predictibilidad: regresiones predictivas de (i) el exceso de rentabilidad
   futuro, (ii) el logaritmo de la varianza realizada futura (HAR, Corsi 2009,
   con rentabilidades diarias: el componente diario es la rentabilidad al
   cuadrado de un día, no una varianza realizada intradía) y (iii) el
   crecimiento del 3x, a 1 mes, 1 año y 3 años. Con objetivos solapados de h
   sesiones, errores de Hansen-Hodrick con h − 1 retardos (Newey-West con 2h
   si la varianza no sale positiva). R² fuera de muestra frente a la media
   histórica (Campbell-Thompson 2008) con ventana creciente, también con la
   previsión del exceso truncada en 0, y test de Clark-West (2007). La
   varianza, además, en niveles, la escala que entra en L*: HAR logarítmico
   con corrección de Duan (1983) en tiempo real y HAR en niveles.
C. Reglas de entrada/salida: 15 reglas de exposición dentro o fuera del 3x
   por caída, volatilidad y tendencia, con p nominal, Holm y
   Benjamini-Hochberg por regla. El Reality Check y el SPA de la familia están
   en R11 §D.
D. Incertidumbre de L*: bootstrap estacionario (bloques de un año) de
   (μ, σ², r, f); apalancamiento que maximiza la media y el percentil 5 del
   crecimiento, con su rango entre semillas.
E. L* ex ante: prima μ̂ − f̂ = media creciente de u − f hasta la sesión anterior
   (kelly.premium) y σ² con la previsión HAR logarítmica con corrección de
   Duan, usando en cada fecha solo la información disponible.
F. Diferencia de medianas a 3 años del 3x comprando con caída > 20% (orden al
   cierre siguiente) frente al resto: IC bootstrap (descriptivo), placebo
   circular del mismo estadístico y su distribución sin predictibilidad (H0
   simulada), que muestra que no está centrado en cero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

from letf_lab import kelly, metrics, sim, stats
from letf_lab.report import FIG_W, Report, comma_fig, legend_band, num_es, pct_es
from letf_lab.research.common import DIV_YIELD, long_underlying, rf
from letf_lab.research.r3_synthetic import synthetic

INDICES = [("S&P 500", "UPRO"), ("Nasdaq-100", "TQQQ"), ("Semis", "SOXL")]
HORIZONS = {"1 mes": 21, "1 año": 252, "3 años": 756}
MIN_OBS = 252 * 10  # diez años de historia antes de la primera previsión
B = 1000
B_PAIRS = 2000      # extracciones Monte Carlo de (μ, σ², r, f) en §D
SEEDS_LSTAR = 5     # semillas de la sensibilidad de §D
N_PLACEBO = 2000    # desplazamientos del placebo de §F
N_NULL = 400        # trayectorias sin predictibilidad (H0) de §F
N_NULL_SHIFTS = 200  # desplazamientos del placebo en cada trayectoria de H0 (tamaño del placebo)
SEED_NULL = 3010    # semilla propia de la simulación de H0 (no mueve el resto del informe)
COST_BPS = 5.0
SEED = 10
SEED_RULES = 1010   # semilla propia del p nominal por regla: R10 §C y R11 §D dan la misma cifra
SEED_SENS = 2010
HAR = ["log_rv_d", "log_rv_w", "log_rv_m"]
MU_VARS = ["caída", "tendencia", "log_rv_m"]
START_SP = "1960-01-01"
EXEC_LAG = 2               # = r12.EXEC_LAG: señal al cierre de t, orden al cierre de t+1
ENTRY_LAG = EXEC_LAG - 1   # desplazamiento de los resultados de comprar y mantener (R9, §F)
DUAN_MIN = 63              # mínimo de cocientes para el factor de Duan (y nunca menos de h)
VAR_FLOOR = 0.01 ** 2      # suelo de la varianza prevista por el HAR en niveles
DD_F = -0.20               # umbral de caída de §F
LGRID = np.round(np.arange(0, 5.01, 0.05), 2)


# ───────────────────────────── datos ─────────────────────────────

def window_sum(x: pd.Series, h: int) -> pd.Series:
    """Suma de x en (t, t+h]: lo que ocurre *después* del cierre de t."""
    cs = x.fillna(0).cumsum()
    return cs.shift(-h) - cs


def har_features(u: pd.Series) -> pd.DataFrame:
    """Log de la varianza realizada diaria, semanal (5) y mensual (21), anualizada."""
    rv_d = u ** 2 * 252
    return pd.DataFrame({
        "log_rv_d": np.log(rv_d.clip(lower=1e-6)),
        "log_rv_w": np.log(rv_d.rolling(5).mean().clip(lower=1e-6)),
        "log_rv_m": np.log(rv_d.rolling(21).mean().clip(lower=1e-6)),
    })


def base(t: str) -> dict:
    """Base corregida del artículo 2 (r12.make): u, rf, f, spread (s'), er, e y x3.

    S&P 500: history.sp500_long desde START_SP (^GSPC con los cierres de
    GSPC_BAD_CLOSES corregidos + D/P de Shiller hasta 1988). Nasdaq-100 y
    semiconductores: common.long_underlying (solo datos reales). r y el ETF
    real, como r12.load_all."""
    from letf_lab.data import history
    from letf_lab.research import r12_long_ml as r12  # perezoso: r12 importa r10

    if t == "UPRO":
        u = history.sp500_long()
        u = u[u.index >= START_SP]
    else:
        u = long_underlying(t)
    r = history.rf_long(u.index) if t in ("UPRO", "TQQQ") else rf(u.index)
    return r12.make(u, r, t, r12.real_letf(t))


def sp500_signal_returns(u: pd.Series, dy: pd.Series | None = None,
                         tr_start: pd.Timestamp | None = None,
                         tr_level: pd.Series | None = None) -> pd.Series:
    """Rentabilidad del S&P 500 para las SEÑALES: hasta 1988, u con el D/P de Shiller
    del mes anterior en lugar del mes en curso; desde el primer día de ^SP500TR, la
    rentabilidad del índice oficial ^SP500TR (no la del SPY, que es u desde 1993).
    Una sola implementación para toda la serie: la de r13 (la misma que usa la LRS
    original del artículo 2)."""
    from letf_lab.research import r13_lrs as r13  # perezoso: r13 importa r12, que importa r10

    return r13.sp500_signal_returns(u, dy, tr_start, tr_level)


def load(t: str) -> dict:
    """Base de base() más las variables de estado. caída, tendencia y vol21 (las
    señales de las reglas de §C y los estados de R9) salen de u_señal, que en el
    S&P 500 lleva antes de 1988 el D/P del mes anterior y desde 1988 es el índice
    oficial ^SP500TR (sp500_signal_returns); en Nasdaq-100 y semiconductores el
    dividendo añadido es constante y u_señal = u. Los regresores HAR y las
    previsiones de varianza usan u."""
    from letf_lab.research import r12_long_ml as r12  # perezoso: r12 importa r10

    d = base(t)
    u = d["u"]
    us = sp500_signal_returns(u) if t == "UPRO" else u
    px = (1 + us.fillna(0)).cumprod()
    X = har_features(u)
    X["caída"] = px / px.cummax() - 1
    X["tendencia"] = px / px.rolling(200).mean() - 1
    real = r12.real_letf(t).dropna()
    d.update(ticker=t, u_señal=us, X=X, vol21=np.sqrt((us ** 2 * 252).rolling(21).mean()),
             s_R3=synthetic(t)[1], real=real, lanzamiento=real.index[0])
    return d


def targets(d: dict, h: int) -> pd.DataFrame:
    """Momentos realizados en (t, t+h], anualizados."""
    u, r, f, x3 = d["u"], d["rf"], d["f"], d["x3"]
    m1 = window_sum(u, h) / h
    m2 = window_sum(u ** 2, h) / h
    var = (m2 - m1 ** 2) * 252
    return pd.DataFrame({
        "exceso": window_sum(np.log1p(u) - np.log1p(r), h) * 252 / h,
        "exceso_arit": window_sum(u - r, h) * 252 / h,
        "var": var,
        "log_var": np.log(var.clip(lower=1e-8)),
        "g3x": window_sum(np.log1p(x3.clip(lower=-0.999)), h) * 252 / h,
        "mu": m1 * 252,
        "r": window_sum(r, h) * 252 / h,
        "f": window_sum(f, h) * 252 / h,
    }).where(m1.notna())


def growth(L, mu, var, r, f, spread, er):
    """Crecimiento log anual de un fondo de reset diario con apalancamiento L
    (modelo de segundo orden): r + L(μ − f) − |L|·s − ER − ½L²σ². L = 0 es
    quedarse en letras, sin comisión ni swap."""
    L = np.asarray(L, float)
    return (r + L * (mu - f) - np.where(L != 0, np.abs(L) * spread + er, 0.0)
            - 0.5 * L ** 2 * var)


def fig_name(name: str) -> str:
    """Nombre del índice en las figuras (las claves de las tablas no cambian)."""
    return "Semiconductores" if name == "Semis" else name


def numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas de tipo object que solo tienen números → numéricas, para que
    report.fmt las redondee (las tablas hechas con dicts de tipos mezclados
    salen como object). Texto, fechas y booleanos no se tocan."""
    out = df.copy()
    for c in out.columns:
        col = out[c]
        if col.dtype != object:
            continue
        vals = col.dropna()
        if len(vals) and all(isinstance(v, (int, float, np.integer, np.floating))
                             and not isinstance(v, (bool, np.bool_)) for v in vals):
            out[c] = pd.to_numeric(col)
    return out


class Report1(Report):
    """Report de R9–R11: redondea también las columnas numéricas de tipo object."""

    def table(self, df: pd.DataFrame, *args, **kw):
        super().table(numeric(df), *args, **kw)


# ───────────────────────────── 0. base y réplica ─────────────────────────────

def r13_tr_start() -> pd.Timestamp:
    """Primer día de ^SP500TR en la base (el 4-1-1988)."""
    from letf_lab import data

    return data.panel(["^SP500TR"], "adj_close")["^SP500TR"].first_valid_index()


def etf_start(ticker: str) -> pd.Timestamp:
    """Primer cierre del ETF en la base (su primera rentabilidad es la del día siguiente)."""
    from letf_lab import data

    return data.panel([ticker], "adj_close")[ticker].first_valid_index()


def signal_gap_bp(u_signal: pd.Series, u: pd.Series) -> pd.Series:
    """Diferencia diaria CON SIGNO entre la rentabilidad de la señal y u, en pb. Su
    desviación típica es la de esta serie; el valor absoluto solo sirve para la media y el
    máximo del desfase (la desviación típica de |diferencia| es otra cifra, menor)."""
    return (u_signal - u) * 1e4


def source_text(d: dict) -> str:
    from letf_lab import data

    e = d["e"]
    etf0 = data.panel([e.underlying], "adj_close")[e.underlying].first_valid_index()
    if d["ticker"] == "UPRO":
        tr0 = data.panel(["^SP500TR"], "adj_close")["^SP500TR"].first_valid_index()
        return (f"^GSPC con cierres erróneos corregidos + D/P de Shiller hasta {tr0.year}; "
                f"^SP500TR hasta {etf0.year}; {e.underlying} después")
    return (f"{e.index_long} + dividendo constante del {pct_es(DIV_YIELD[e.index_long], 1)} "
            f"anual hasta {etf0.year}; {e.underlying} después")


def base_table(D: dict) -> pd.DataFrame:
    rows = {}
    for name, d in D.items():
        u = d["u"]
        rows[name] = {"desde": u.index[0].date(), "hasta": u.index[-1].date(),
                      "rentabilidad del índice": source_text(d),
                      "r_media": d["rf"].mean() * 252, "f_media": d["f"].mean() * 252,
                      "f−r_media": (d["f"] - d["rf"]).mean() * 252,
                      "s_R3 sobre r": d["s_R3"], "s' sobre f": d["spread"], "ER": d["er"],
                      "3x real desde": d["lanzamiento"].date()}
    return pd.DataFrame(rows).T


def to_pb(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Columnas en fracción anual → puntos básicos (×10⁴, « (pb)» en el nombre, en
    su sitio): report.fmt da un decimal a los porcentajes y s_R3 y s' difieren en
    décimas de pb. No cambia fmt."""
    out = df.copy()
    for c in cols:
        out[c] = (pd.to_numeric(out[c]) * 1e4).round(2)
    return out.rename(columns={c: f"{c} (pb)" for c in cols})


def replica_validation(d: dict) -> dict:
    """El 3x real frente a la réplica rf + 3(u − f − s') − ER en la vida del ETF.

    En el periodo de calibración de s' la deriva media coincide por
    construcción; lo informativo es la correlación, la caída máxima y la
    segunda mitad, con s' calibrado solo en la primera."""
    from letf_lab.research import r12_long_ml as r12

    real = d["real"]
    both = pd.concat([real, r12.letf(d, 3.0).reindex(real.index)], axis=1,
                     keys=["real", "réplica"]).dropna()
    half = both.index[len(both) // 2]
    first, second = both.index < half, both.index >= half
    ub = (d["u"] - (d["f"] - d["rf"])).reindex(both.index)
    s1 = sim.calibrate_spread(both.real[first], ub[first], 3, d["rf"], d["er"])
    rep2 = r12.letf(dict(d, spread=s1), 3.0).reindex(both.index)[second]
    real2 = both.real[second]
    return {"desde": both.index[0].date(), "hasta": both.index[-1].date(),
            "rent_anual_real": metrics.cagr(both.real),
            "rent_anual_réplica": metrics.cagr(both["réplica"]),
            "corr_diaria": both.corr().iloc[0, 1],
            "caída_máx_real": metrics.max_dd(both.real),
            "caída_máx_réplica": metrics.max_dd(both["réplica"]),
            "s'_1ª_mitad": s1, "2ª_mitad_desde": half.date(),
            "rent_anual_real_2ª": metrics.cagr(real2),
            "rent_anual_réplica_2ª": metrics.cagr(rep2),
            "caída_máx_real_2ª": metrics.max_dd(real2),
            "caída_máx_réplica_2ª": metrics.max_dd(rep2)}


# ───────────────────────────── A. identidad ─────────────────────────────

def identity(d: dict, h: int = 756) -> dict:
    T = targets(d, h)
    L = 3.0
    drift = T.r + L * (T.mu - T.f - d["spread"]) - d["er"]
    drag = -0.5 * L ** 2 * T["var"]
    # momentos brutos 3.º y 4.º del índice en la ventana (términos L³x³/3 − L⁴x⁴/4)
    m3 = window_sum(d["u"] ** 3, h) / h
    m4 = window_sum(d["u"] ** 4, h) / h
    mom = 252 * (L ** 3 / 3 * m3 - L ** 4 / 4 * m4)
    df = pd.DataFrame({"real": T.g3x, "drift": drift, "drag": drag, "mom": mom}).dropna()
    df["modelo"] = df.drift + df.drag
    df["modelo_mom"] = df.modelo + df.mom
    # tramo de la ventana (t, t+h]: solo réplica, solo fondo real o mixta
    idx = d["u"].index
    end = pd.Series(idx, index=idx).shift(-h).reindex(df.index)
    launch = d["lanzamiento"]
    df["tramo"] = np.where(end < launch, "réplica", np.where(df.index >= launch, "fondo real", "mixta"))
    res = df.real - df.modelo
    res4 = df.real - df.modelo_mom
    c = np.cov(df.drift, df.drag)
    vg = df.modelo.var()
    out = {"R2": 1 - res.var() / df.real.var(), "sesgo": res.mean(),
           "RMSE_residuo": np.sqrt((res ** 2).mean()),
           "cuota_μ": (c[0, 0] + c[0, 1]) / vg, "cuota_σ²": (c[1, 1] + c[0, 1]) / vg,
           "cuota_μ_sin_cov": c[0, 0] / vg, "cuota_μ_con_cov": (c[0, 0] + 2 * c[0, 1]) / vg,
           "RMSE_con_momentos": np.sqrt((res4 ** 2).mean()),
           "reducción_error_cuadrático_momentos": 1 - (res4 ** 2).sum() / (res ** 2).sum()}
    by = {}
    for tr in ("réplica", "mixta", "fondo real"):
        m = (df.tramo == tr).values
        if m.sum() < 2:
            continue
        r, r4 = res[m], res4[m]
        by[tr] = {"ventanas": int(m.sum()), "R2": 1 - r.var() / df.real[m].var(),
                  "RMSE_residuo": np.sqrt((r ** 2).mean()),
                  "RMSE_con_momentos": np.sqrt((r4 ** 2).mean()),
                  "reducción_error_cuadrático_momentos": 1 - (r4 ** 2).sum() / (r ** 2).sum()}
    return {"df": df, **out, "por_tramo": pd.DataFrame(by).T}


# ───────────────────────────── B. predictibilidad ─────────────────────────────

def r2_truncated(y, f_model, f_bench) -> float:
    """R² fuera de muestra con la previsión del modelo truncada en 0 (una prima
    esperada no negativa). Campbell y Thompson (2008) restringen además el signo
    de los coeficientes: no es exactamente su R²."""
    return stats.oos_r2(np.asarray(y, float), np.maximum(np.asarray(f_model, float), 0),
                        np.asarray(f_bench, float))


def predictive(d: dict, target: str, cols: list[str], h: int) -> dict:
    """Regresión predictiva a h sesiones: t de Hansen-Hodrick (h − 1 retardos;
    recurso Newey-West 2h), previsión fuera de muestra y Clark-West con el mismo
    estimador. R2_OOS_truncado: la previsión del exceso truncada en 0."""
    T = targets(d, h)
    y, X = T[target], d["X"][cols]
    hac = stats.hac_overlap(h)
    fit = stats.ols_hac(y, X, **hac)
    fit_nw = stats.ols_hac(y, X, lags=2 * h)
    fc = stats.expanding_forecast(y, X, h=h, min_obs=MIN_OBS)
    yv, fb, fm = fc.y.values, fc.media.values, fc.modelo.values
    r2 = stats.oos_r2(yv, fm, fb)
    r2_tr = r2_truncated(yv, fm, fb) if target in ("exceso", "exceso_arit") else np.nan
    cw = stats.clark_west_fit(yv, fb, fm, **hac)
    cw_t = float(cw.t[0])
    _, cw_p_nw = stats.clark_west(yv, fb, fm, lags=2 * h)
    return {"fit": fit, "fit_nw2h": fit_nw, "fc": fc, "R2_IS": fit.r2, "R2_OOS": r2,
            "R2_OOS_truncado": r2_tr, "CW_t": cw_t, "CW_p": float(norm.sf(cw_t)),
            "CW_estimador": cw.estimator, "CW_p_NW2h": cw_p_nw,
            "n": fit.n, "desde_OOS": fc.modelo.first_valid_index()}


def duan_oos(y_lvl: pd.Series, f_log: pd.Series, h: int, min_n: int | None = None) -> pd.Series:
    """Factor de Duan (1983) en tiempo real: media creciente de y_s/exp(f_s) con
    los objetivos ya observados.

    El objetivo de s se conoce en s + h, así que en t solo entran los cocientes
    con s ≤ t − h (desplazamiento de h sesiones). Hacen falta al menos min_n
    cocientes (por defecto, el mayor de DUAN_MIN y h); antes, NaN."""
    min_n = max(DUAN_MIN, h) if min_n is None else min_n
    ratio = (y_lvl / np.exp(f_log)).where(y_lvl.notna() & f_log.notna())
    return ratio.expanding(min_n).mean().shift(h)


def har_var_forecast(y_lvl: pd.Series, u: pd.Series, h: int, min_obs: int = MIN_OBS) -> pd.DataFrame:
    """Previsión en tiempo real de la varianza y_lvl (objetivo de h sesiones, en
    niveles) con el HAR logarítmico y la corrección de Duan fuera de muestra.

    log_raw: previsión de log y (media condicional del logaritmo); nivel_raw =
    exp(log_raw), la media geométrica condicional (la mediana solo si el error del
    logaritmo es simétrico), por debajo de la media; nivel = nivel_raw·factor.
    media_log: media creciente embargada de log y (benchmark en logaritmos)."""
    fc = stats.expanding_forecast(np.log(y_lvl.clip(lower=1e-8)), har_features(u), h=h,
                                  min_obs=min_obs)
    factor = duan_oos(y_lvl, fc.modelo, h)
    return pd.DataFrame({"realizada": y_lvl, "log_raw": fc.modelo, "nivel_raw": np.exp(fc.modelo),
                         "factor": factor, "nivel": np.exp(fc.modelo) * factor,
                         "media_log": fc.media})


def growth_loss(realized, forecast, premium: float) -> np.ndarray:
    """Crecimiento log anual que se pierde por fijar el apalancamiento con la varianza
    prevista, con una prima constante m = μ − f − s (la ecuación central):

        L̂ = m/σ̂², L* = m/σ², g(L*) − g(L̂) = ½σ²(L̂ − L*)² = ½·SR²·(ρ − 1)²,

    con ρ = σ²/σ̂² (realizada entre prevista) y SR² = m²/σ². Con m constante es
    proporcional a (ρ − 1)²/σ², la pérdida robusta de Patton (2011) con b = −3,
    ½(1/σ² − 2/σ̂² + σ²/σ̂⁴); QLIKE ≈ ½(ρ − 1)² solo para errores pequeños, y mide la
    fracción del crecimiento adicional que se pierde, no los puntos."""
    y = np.asarray(realized, float)
    ratio = y / np.asarray(forecast, float)
    return 0.5 * premium ** 2 * (ratio - 1) ** 2 / y


def premium_const(d: dict) -> float:
    """Prima neta constante μ − f − s' de toda la muestra (la de R11 §A)."""
    df = pd.concat([d["u"], d["f"]], axis=1, keys=["u", "f"]).dropna()
    return float((df.u - df.f).mean() * 252 - d["spread"])


def var_levels(d: dict, h: int) -> dict[str, dict]:
    """Precisión en niveles de varianza (la escala de L*) del HAR logarítmico con
    Duan y del HAR en niveles, frente a la media creciente embargada del objetivo
    en niveles, en la ventana común.

    R2_OOS_log puntúa la previsión del logaritmo de cada modelo frente a la media
    creciente del logaritmo: en el HAR logarítmico, log_raw (la media condicional
    de log y, la de la fila log_var de §B), sin el factor de Duan, que corrige la
    media en niveles y en logaritmos sería un sesgo; en el HAR en niveles, el
    logaritmo de su previsión, la única que tiene."""
    u = d["u"]
    y = targets(d, h)["var"]
    hv = har_var_forecast(y, u, h)
    Xh = har_features(u)
    bench = stats.expanding_forecast(y, Xh, h=h, min_obs=MIN_OBS).media
    lev = stats.expanding_forecast(y, np.exp(Xh), h=h, min_obs=MIN_OBS).modelo.clip(lower=VAR_FLOOR)
    F = pd.DataFrame({"y": y, "HAR-log + Duan": hv.nivel, "HAR en niveles": lev,
                      "media": bench, "media_log": hv.media_log, "log_raw": hv.log_raw}).dropna()
    ly = np.log(F.y.clip(lower=1e-8)).values
    log_fc = {"HAR-log + Duan": F.log_raw.values, "HAR en niveles": np.log(F["HAR en niveles"].values)}
    m_prem = premium_const(d)
    gl_media = float(growth_loss(F.y.values, F.media.values, m_prem).mean())
    out = {}
    for m in ("HAR-log + Duan", "HAR en niveles"):
        f = F[m].values
        gl = float(growth_loss(F.y.values, f, m_prem).mean())
        out[m] = {"n": len(F), "desde": F.index[0].date(),
                  "R2_OOS_niveles": stats.oos_r2(F.y.values, f, F.media.values),
                  "R2_OOS_log": stats.oos_r2(ly, log_fc[m], F.media_log.values),
                  "QLIKE_modelo": float(stats.qlike(F.y.values, f).mean()),
                  "QLIKE_media": float(stats.qlike(F.y.values, F.media.values).mean()),
                  "cociente_medio": float((F.y.values / f).mean()),
                  "pérdida_crec_modelo_pb": gl * 1e4, "pérdida_crec_media_pb": gl_media * 1e4,
                  "ventaja_crec_pb": (gl_media - gl) * 1e4}
    return out


# ───────────────────────────── C. reglas ─────────────────────────────

def rules(d: dict) -> dict[str, pd.Series]:
    X, vol = d["X"], d["vol21"]
    out = {}
    for k in (0.05, 0.10, 0.20, 0.30):
        out[f"dentro sólo con caída ≥ {k:.0%}"] = X["caída"] <= -k
    for k in (0.10, 0.20, 0.30):
        out[f"fuera con caída ≥ {k:.0%}"] = X["caída"] > -k
    for v in (0.12, 0.18, 0.25, 0.35):
        out[f"dentro sólo con vol21 < {v:.0%}"] = vol < v
    for v in (0.25, 0.35):
        out[f"dentro sólo con vol21 ≥ {v:.0%}"] = vol >= v
    out["dentro sólo sobre SMA200"] = X["tendencia"] > 0
    out["dentro sólo bajo SMA200"] = X["tendencia"] < 0
    return {k: v.astype(float) for k, v in out.items()}


def rule_label(k: str) -> str:
    """Etiqueta legible de una regla (las claves de rules() no cambian: r19 las usa)."""
    return (k.replace("sólo", "solo").replace("vol21", "volatilidad de 21 sesiones")
            .replace("SMA200", "la media de 200 sesiones"))


RULE_KEYS = [f"dentro sólo con caída ≥ {k:.0%}" for k in (0.05, 0.10, 0.20, 0.30)] + \
            [f"fuera con caída ≥ {k:.0%}" for k in (0.10, 0.20, 0.30)] + \
            [f"dentro sólo con vol21 < {v:.0%}" for v in (0.12, 0.18, 0.25, 0.35)] + \
            [f"dentro sólo con vol21 ≥ {v:.0%}" for v in (0.25, 0.35)] + \
            ["dentro sólo sobre SMA200", "dentro sólo bajo SMA200"]
RULE_LABELS = {k: rule_label(k) for k in RULE_KEYS}
LRS_RULE = "dentro sólo sobre SMA200"  # la señal de la LRS de Gayed y Bilello


def rule_returns(d: dict, w: pd.Series, lag: int = EXEC_LAG) -> pd.Series:
    """Rentabilidad log de la regla: peso decidido al cierre de t; con lag = 2
    cobra la rentabilidad desde el cierre de t+1 (orden al cierre siguiente),
    con lag = 1 desde el cierre de t (mismo cierre). Caja en letras; 5 pb sobre
    |Δw| del peso ya desplazado (mismo convenio que r13.run_weights)."""
    w = w.shift(lag).fillna(0.0)
    r = w * d["x3"] + (1 - w) * d["rf"] - w.diff().abs().fillna(0) * COST_BPS / 1e4
    return np.log1p(r.clip(lower=-0.999))


def rule_diffs(d: dict, lag: int = EXEC_LAG) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Matriz D (regla − mantener el 3x, rentabilidad log diaria) de las 15
    reglas, la misma para R10 §C y R11 §D.

    Empieza en la primera sesión en que todas las señales existen (la media de
    200 sesiones) más `lag`: rules() da 0 cuando la señal es NaN, así que el
    recorte se hace por posición. Devuelve también los pesos desplazados (1 =
    dentro ese día) y el crecimiento log diario del 3x en las mismas fechas."""
    bh = np.log1p(d["x3"].clip(lower=-0.999))
    R = rules(d)
    D = pd.DataFrame({k: rule_returns(d, w, lag) - bh for k, w in R.items()})
    W = pd.DataFrame({k: w.shift(lag) for k, w in R.items()})
    start = D.index.get_loc(d["X"]["tendencia"].first_valid_index()) + lag
    D = D.iloc[start:].dropna()
    return D, W.reindex(D.index), bh.reindex(D.index)


def rule_table(d: dict, lag: int = EXEC_LAG, B: int = B, seed: int = SEED_RULES) -> pd.DataFrame:
    """Tabla por regla: Δ de crecimiento anual frente a mantener el 3x, p nominal
    unilateral (bootstrap estacionario de su media, bloque 63), Holm, BH, tiempo
    dentro, crecimiento del 3x en los días dentro y coste mecánico de estar
    fuera, (1 − tiempo_dentro)·(r̄ − ḡ_3x): la Δ que tendría una regla que
    eligiera sus días al azar, sin costes.

    Fuente única de los p por regla: usa su propia semilla, así que R10 §C y
    R11 §D dan la misma cifra."""
    D, W, bh = rule_diffs(d, lag)
    rng = np.random.default_rng(seed)
    T = len(D)
    dv = D.values
    dbar = dv.mean(axis=0)
    exceed = np.zeros(D.shape[1])
    for _ in range(B):
        idx = stats.stationary_indices(T, 63, rng)
        exceed += (dv[idx].mean(axis=0) - dbar) >= dbar
    p = exceed / B
    g3 = bh.mean() * 252
    rbar = np.log1p(d["rf"]).reindex(D.index).mean() * 252
    tau = W.mean()
    out = pd.DataFrame({"Δ_crec_anual": dbar * 252, "p_nominal": p,
                        "p_Holm": stats.holm(p), "q_BH": stats.benjamini_hochberg(p),
                        "tiempo_dentro": tau.values,
                        "crec_3x_dentro": [bh[W[k] == 1].mean() * 252 for k in D.columns],
                        "coste_mecánico": ((1 - tau) * (rbar - g3)).values},
                       index=D.columns)
    out["selección_y_costes"] = out["Δ_crec_anual"] - out["coste_mecánico"]
    out.attrs.update(desde=D.index[0].date(), dias=T, crec_3x=g3, r_letras=rbar, lag=lag)
    return out


# ───────────────────────────── D. incertidumbre de L* ─────────────────────────────

def lstar_draws(d: dict, rng, B: int = B_PAIRS) -> pd.DataFrame:
    """B remuestreos estacionarios (bloques de 252 sesiones) de (u, rf, f): μ, σ²,
    r̄ y f̄ anuales de cada uno y su L* aproximado (kelly.lstar_approx)."""
    xy = pd.concat([d["u"], d["rf"], d["f"]], axis=1).dropna().values
    rows = []
    for _ in range(B):
        a = xy[stats.stationary_indices(len(xy), 252, rng)]
        rows.append((a[:, 0].mean() * 252, a[:, 0].var() * 252, a[:, 1].mean() * 252,
                     a[:, 2].mean() * 252, kelly.lstar_approx(a[:, 0], a[:, 2], d["spread"])))
    return pd.DataFrame(rows, columns=["mu", "var", "r", "f", "L*"])


def lstar_from_draws(P: pd.DataFrame, spread: float, er: float, Lg=LGRID) -> dict:
    """Sobre las mismas extracciones: L que maximiza el crecimiento medio, L que
    maximiza su percentil 5 y fracción de extracciones en que g(3x) < r̄."""
    g = growth(Lg[None, :], P.mu.values[:, None], P["var"].values[:, None],
               P.r.values[:, None], P.f.values[:, None], spread, er)
    return {"L_máx_media": float(Lg[g.mean(axis=0).argmax()]),
            "L_robusto_p5": float(Lg[np.percentile(g, 5, axis=0).argmax()]),
            "P(g(3x)<letras)": float((g[:, Lg == 3.0].ravel() < P.r.values).mean())}


def lstar_bootstrap(d: dict, rng) -> dict:
    df = pd.concat([d["u"], d["f"]], axis=1, keys=["u", "f"]).dropna()
    s, er = d["spread"], d["er"]
    P = lstar_draws(d, rng)
    lo, hi = np.percentile(P["L*"], [5, 95])
    out = {kelly.LSTAR_APROX: kelly.lstar_approx(df.u, df.f, s), "IC90_bajo": lo, "IC90_alto": hi,
           "P(L*<1,5)": float((P["L*"] < 1.5).mean()), "P(L*>3)": float((P["L*"] > 3).mean())}
    out.update(lstar_from_draws(P, s, er))
    sens = pd.DataFrame([lstar_from_draws(lstar_draws(d, np.random.default_rng(SEED_SENS + k)), s, er)
                         for k in range(SEEDS_LSTAR)])
    return {**out, "draws": P, "semillas": sens, "spread": s, "er": er}


# ───────────────────────────── E. L* ex ante ─────────────────────────────

EX_ANTE = {"base": f"prima {kelly.PRIMA}, σ² HAR-log + Duan",
           "(i) media creciente de u, menos f_t": "media creciente de u menos el f_t de ese día, "
                                                  "σ² HAR-log + Duan",
           "(ii) σ² HAR en niveles": f"prima {kelly.PRIMA}, σ² HAR en niveles"}


def lstar_ex_ante(d: dict) -> pd.DataFrame:
    """L*_t = (μ̂_t − f̂_t − s')/σ̂²_t con información hasta t (L* aproximado).

    base: prima compartida kelly.premium (media creciente de u − f con datos
    hasta t − 1, desde el año 10) y σ̂²_t = previsión del HAR logarítmico con
    corrección de Duan de la varianza del año siguiente (targets(·, 252).var).
    (i): la prima como media creciente de u menos el f de ese día; supone que la
    rentabilidad total no cambia cuando cambia f (como el Kelly anterior de R12).
    (ii): σ̂² del HAR en niveles (suelo VAR_FLOOR)."""
    h = 252
    u, f, s = d["u"], d["f"], d["spread"]
    y = targets(d, h)["var"]
    var_log = har_var_forecast(y, u, h)["nivel"]
    var_lev = stats.expanding_forecast(y, np.exp(har_features(u)), h=h,
                                       min_obs=MIN_OBS).modelo.clip(lower=VAR_FLOOR)
    prem = kelly.premium(u, f, MIN_OBS)
    prem_i = u.shift(1).expanding(MIN_OBS).mean() * 252 - f * 252
    keys = list(EX_ANTE)
    return pd.DataFrame({keys[0]: (prem - s) / var_log, keys[1]: (prem_i - s) / var_log,
                         keys[2]: (prem - s) / var_lev})


def ex_ante_summary(ex: pd.DataFrame) -> dict:
    """Resumen de cada variante. mediana y fracciones de días, en la ventana común
    a todas (ex.dropna(), desde ventana_común_desde): así una variante cambia solo
    el estimador y no la muestra (la (ii) no espera al factor de Duan y empieza
    antes). hoy y fecha: último dato de cada variante; primer_día: el primero de
    cada una, como información."""
    common = ex.dropna()
    out = {}
    for c in ex.columns:
        s, sc = ex[c].dropna(), common[c]
        out[c] = {"hoy": s.iloc[-1], "fecha": s.index[-1].date(), "mediana": sc.median(),
                  "fracción_días_L*≥3": (sc >= 3).mean(), "fracción_días_L*<1": (sc < 1).mean(),
                  "ventana_común_desde": common.index[0].date(), "primer_día": s.index[0].date()}
    return out


# ───────────────────────────── F. medianas con IC ─────────────────────────────

def episode_spans(mask, gap: int = 21) -> list[tuple[int, int]]:
    """(primera, última) posición de cada episodio: tramos del estado separados
    por más de `gap` sesiones."""
    idx = np.flatnonzero(np.asarray(mask, bool))
    if not len(idx):
        return []
    cut = np.flatnonzero(np.diff(idx) > gap)
    starts = np.r_[idx[0], idx[cut + 1]]
    ends = np.r_[idx[cut], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def episodes(mask, gap: int = 21) -> int:
    """Nº de episodios distintos: tramos del estado separados por más de `gap` días."""
    return len(episode_spans(mask, gap))


def _median_gap(y: np.ndarray, m: np.ndarray) -> float:
    if m.sum() < 20 or (~m).sum() < 20:
        return np.nan
    return float(np.median(y[m]) - np.median(y[~m]))


def median_gap_ci(d: dict, rng, h: int = 756, thr: float = DD_F, B: int = B,
                  n_placebo: int = N_PLACEBO, n_tests: int = 3) -> dict:
    """Mediana del crecimiento anual del 3x a h sesiones comprando al cierre
    siguiente cuando la caída al cierre de t es ≤ thr, menos la del resto.

    IC percentil 90% y de Bonferroni para n_tests índices (1 − 0,10/n_tests) con
    bootstrap estacionario de bloque h, y placebo circular del MISMO estadístico:
    el vector de estado se desplaza (al menos h sesiones) respecto a los
    resultados, lo que conserva lo agrupados que están los estados."""
    y = targets(d, h).g3x.shift(-ENTRY_LAG)
    df = pd.DataFrame({"y": y, "m": (d["X"]["caída"] <= thr).astype(float)}).dropna()
    yv, mv = df.y.to_numpy(), df.m.to_numpy() > 0.5
    n = len(yv)
    est = _median_gap(yv, mv)
    boots = []
    for _ in range(B):
        i = stats.stationary_indices(n, h, rng)
        boots.append(_median_gap(yv[i], mv[i]))
    a = 0.10 / n_tests
    lo, hi = np.nanpercentile(boots, [5, 95])
    lo_b, hi_b = np.nanpercentile(boots, [100 * a / 2, 100 * (1 - a / 2)])
    null = np.array([_median_gap(yv, np.roll(mv, k)) for k in rng.integers(h, n - h, n_placebo)])
    null = null[~np.isnan(null)]
    return {"días_en_estado": int(mv.sum()), "episodios": episodes(mv),
            "mediana_gap": est, "IC90_bajo_gap": lo, "IC90_alto_gap": hi,
            f"IC{100 * (1 - 0.10 / n_tests):.1f}_bajo_gap".replace(".", ","): lo_b,
            f"IC{100 * (1 - 0.10 / n_tests):.1f}_alto_gap".replace(".", ","): hi_b,
            "p_placebo_bilateral": float((np.abs(null) >= abs(est)).mean()),
            "p_placebo_unilateral": float((null >= est).mean())}


def median_gap_null(d: dict, obs: float, rng, h: int = 756, thr: float = DD_F, n_sim: int = N_NULL,
                    block: float = 63, n_shift: int = N_NULL_SHIFTS) -> dict:
    """Distribución del estadístico de median_gap_ci sin ninguna predictibilidad (H0).

    Se remuestrean días (u_señal, crecimiento log del 3x) juntos, con bootstrap
    estacionario de bloque medio `block` (1 = i.i.d.): la rentabilidad solo depende del
    pasado dentro de bloques cortos, no a 3 años. En cada trayectoria se recalculan la
    caída desde máximos (el estado) y el crecimiento del 3x a h sesiones comprando al
    cierre siguiente, y el mismo estadístico. Devuelve su media y desviación típica,
    P(estadístico ≥ obs) y, con n_shift desplazamientos por trayectoria, el tamaño del
    placebo circular al 5% (fracción de trayectorias sin efecto en que el placebo daría
    p ≤ 0,05), unilateral y bilateral, con los mismos p que median_gap_ci."""
    df = pd.concat([d["u_señal"], np.log1p(d["x3"].clip(lower=-0.999))], axis=1,
                   keys=["us", "lx"]).dropna()
    us, lx = df.us.to_numpy(), df.lx.to_numpy()
    n = len(df)
    T = n - ENTRY_LAG - h
    sims, rej, rej2 = [], [], []
    for _ in range(n_sim):
        i = rng.integers(0, n, n) if block <= 1 else stats.stationary_indices(n, block, rng)
        px = np.cumprod(1 + us[i])
        m = (px / np.maximum.accumulate(px) - 1 <= thr)[:T]
        cs = np.r_[0.0, np.cumsum(lx[i])]
        y = (cs[ENTRY_LAG + h + 1:ENTRY_LAG + h + 1 + T] - cs[ENTRY_LAG + 1:ENTRY_LAG + 1 + T]) * 252 / h
        g = _median_gap(y, m)
        sims.append(g)
        if n_shift and not np.isnan(g):
            null = np.array([_median_gap(y, np.roll(m, k)) for k in rng.integers(h, T - h, n_shift)])
            null = null[~np.isnan(null)]
            rej.append(float((null >= g).mean()) <= 0.05)
            rej2.append(float((np.abs(null) >= abs(g)).mean()) <= 0.05)
    sims = np.array(sims)
    ok = sims[~np.isnan(sims)]
    return {"bloque": block, "trayectorias": len(ok), "media_H0": float(ok.mean()),
            "sd_H0": float(ok.std(ddof=1)), "P(≥ observado)": float((ok >= obs).mean()),
            "tamaño_placebo_unilateral_5%": float(np.mean(rej)) if rej else np.nan,
            "tamaño_placebo_bilateral_5%": float(np.mean(rej2)) if rej2 else np.nan}


# ───────────────────────────── figuras ─────────────────────────────

def fig_identity(res: dict, rep: Report):
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 12.5))
    for ax, (name, r) in zip(axes, res.items()):
        df = r["df"].iloc[::5]
        ax.scatter(df.modelo * 100, df.real * 100, s=3, alpha=0.3, color="C3")
        lim = [min(df.modelo.min(), df.real.min()) * 100, max(df.modelo.max(), df.real.max()) * 100]
        ax.plot(lim, lim, "k--", lw=1)
        # R² en porcentaje con dos decimales, como la tabla de §A (con tres decimales, 0,9998
        # se leería «1,000», un ajuste perfecto)
        ax.set(title=f"{fig_name(name)}: R² = {pct_es(r['R2'], 2)}",
               xlabel="r + 3(μ − f − s) − ER − 4,5·σ² (%)",
               ylabel="Crecimiento realizado (%)")
    fig.suptitle("Crecimiento del 3x a 3 años frente al que dan μ y σ² de la ventana")
    comma_fig(fig)
    rep.fig(fig, "identidad", "El crecimiento realizado del 3x a 3 años frente al que predicen μ y σ² de la propia ventana")


OOS_LABELS = {"exceso": "exceso logarítmico sobre las letras",
              "log_var": "logaritmo de la varianza (HAR)",
              "g3x": "crecimiento del 3x"}


def fig_oos(tab: pd.DataFrame, rep: Report):
    # cada panel con su propia escala: con una común, los −275 puntos del Nasdaq-100 a 3 años
    # aplastan las barras del S&P 500 y de semiconductores
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 10.9))
    colors = {"exceso": "C0", "log_var": "C2", "g3x": "C3"}
    for ax, (name, g) in zip(axes, tab.groupby(level=0, sort=False)):
        g = g.droplevel(0)
        hs = list(HORIZONS)
        w = 0.26
        for k, tgt in enumerate(colors):
            vals = [g.loc[(h, tgt), "R2_OOS"] * 100 for h in hs]
            ax.bar(np.arange(len(hs)) + (k - 1) * w, vals, w, color=colors[tgt], label=OOS_LABELS[tgt])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(hs)), hs)
        ax.set(title=fig_name(name), ylabel="R² fuera de muestra (%)")
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    h, lab = axes[0].get_legend_handles_labels()
    legend_band(fig, h, lab, ncol=1)
    fig.suptitle("R² fuera de muestra en tiempo real, a 1 mes, 1 año y 3 años")
    comma_fig(fig)
    rep.fig(fig, "r2_oos", "R² fuera de muestra al predecir el exceso logarítmico sobre las letras, el "
                           "logaritmo de la varianza y el crecimiento del 3x")


def fig_har(d: dict, name: str, rep: Report):
    h = 252
    hv = har_var_forecast(targets(d, h)["var"], d["u"], h).dropna(subset=["nivel", "realizada"])
    fig, ax = plt.subplots(figsize=(FIG_W, 5.2))
    ax.plot(np.sqrt(hv.realizada) * 100, color="k", lw=1, label="volatilidad realizada del año siguiente")
    ax.plot(np.sqrt(hv.nivel) * 100, color="C2", lw=1,
            label="previsión en tiempo real (HAR logarítmico con corrección de Duan)")
    ax.set(title=f"{fig_name(name)}: volatilidad del año siguiente y su previsión",
           ylabel="Volatilidad anual (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=1, frameon=False)
    comma_fig(fig)
    rep.fig(fig, "har_sp500", "Volatilidad del año siguiente frente a su previsión HAR logarítmica "
                              "con corrección de Duan en tiempo real (S&P 500)")


def fig_rule_label(k: str) -> str:
    """Rótulo corto de una regla para la figura (el pie y el título dicen que la volatilidad
    es la de 21 sesiones y la media, la de 200)."""
    return (k.replace("sólo", "solo").replace("vol21", "volatilidad")
            .replace("sobre SMA200", "sobre la media").replace("bajo SMA200", "bajo la media"))


def fig_rules(rc: dict, rep: Report):
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 16), sharex=True)
    all_c = []
    for ax, (name, t) in zip(axes, rc.items()):
        t = t.sort_values("Δ_crec_anual")
        c = ["C2" if q < 0.05 else ("C1" if p < 0.05 else "0.6")
             for q, p in zip(t.q_BH, t.p_nominal)]
        all_c += c
        ax.barh(range(len(t)), t["Δ_crec_anual"] * 100, color=c)
        ax.set_yticks(range(len(t)), [fig_rule_label(k) for k in t.index])
        ax.axvline(0, color="k", lw=0.8)
        ax.set(title=fig_name(name))
        ax.grid(axis="x", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Crecimiento anual frente a mantener el 3x (puntos)")
    # la clave de colores solo si alguna barra la usa
    key = ("verde: significativa tras Benjamini-Hochberg; naranja: solo p nominal < 0,05"
           if any(x != "0.6" for x in all_c) else "ninguna con un p nominal menor que 0,05")
    fig.suptitle("Reglas de entrada y salida frente a mantener el 3x\n"
                 "(orden al cierre siguiente; volatilidad de 21 sesiones; media de 200)\n"
                 f"({key})", fontsize=13)
    comma_fig(fig)
    rep.fig(fig, "reglas", "Diferencia de crecimiento anual de cada regla de entrada y salida frente a mantener el 3x")


def fig_lstar(ls: dict, rep: Report):
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 10))
    Lg = np.linspace(0, 5, 101)
    for (name, b), c in zip(ls.items(), ["C0", "C1", "C2"]):
        P = b["draws"]
        axes[0].plot(Lg, gaussian_kde(P["L*"])(Lg), color=c,
                     label=f"{fig_name(name)}: {num_es(b[kelly.LSTAR_APROX], 2)} "
                           f"({num_es(b['IC90_bajo'], 2)}–{num_es(b['IC90_alto'], 2)})")
        g = growth(Lg[None, :], P.mu.values[:, None], P["var"].values[:, None],
                   P.r.values[:, None], P.f.values[:, None], b["spread"], b["er"])
        lo, mid, hi = np.percentile(g, [5, 50, 95], axis=0) * 100
        axes[1].fill_between(Lg, lo, hi, color=c, alpha=0.15)
        axes[1].plot(Lg, mid, color=c, label=fig_name(name))
    axes[0].axvline(3, color="r", ls="--", lw=1)
    axes[0].set(title="Distribución bootstrap de L* (bloques de un año)", xlabel="L*", yticks=[])
    axes[0].legend(title="L* (intervalo del 90%)")
    axes[1].axvline(3, color="r", ls="--", lw=1)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set(title="Crecimiento esperado g(L) con banda del 90%", xlabel="Apalancamiento L",
                ylabel="Crecimiento logarítmico anual (%)", ylim=(-30, 30))
    axes[1].legend(loc="lower left")
    fig.suptitle("El apalancamiento óptimo y su incertidumbre")
    comma_fig(fig)
    rep.fig(fig, "lstar_incertidumbre", "Distribución bootstrap del apalancamiento óptimo y curva de crecimiento con su banda")


CLIP_EX = (-2, 8)


def fig_ex_ante(ex: dict, rep: Report) -> dict:
    """Serie base de §E recortada a CLIP_EX para dibujarla; devuelve los días recortados."""
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 10.5), sharex=True)
    clipped = {}
    for ax, (name, df) in zip(axes, ex.items()):
        raw = df.iloc[:, 0].dropna()
        clipped[name] = int(((raw < CLIP_EX[0]) | (raw > CLIP_EX[1])).sum())
        s = raw.clip(*CLIP_EX)
        ax.plot(s, lw=0.8, color="C0")
        ax.axhline(3, color="r", ls="--", lw=1, label="3x")
        ax.axhline(1, color="k", lw=0.6, label="1x")
        ax.scatter([s.index[-1]], [s.iloc[-1]], color="C3", zorder=3,
                   label=f"hoy ({s.index[-1]:%d-%m-%Y}): {num_es(raw.iloc[-1], 2)}")
        ax.set(title=fig_name(name), ylabel="L* ex ante")
        ax.legend(loc="upper left", ncol=3, fontsize=11, columnspacing=1.0, handlelength=1.5)
    fig.suptitle("Apalancamiento óptimo estimado cada día con lo que se sabía ese día")
    comma_fig(fig)
    note = "; ".join(f"{fig_name(n)}: {num_es(k, 0)}" for n, k in clipped.items())
    rep.fig(fig, "lstar_ex_ante", "Apalancamiento óptimo estimado en tiempo real, con el valor actual. "
                                  f"Eje recortado a [{num_es(CLIP_EX[0], 0)}, {CLIP_EX[1]}]; días fuera "
                                  f"del recorte: {note}")
    return clipped


# ───────────────────────────── informe ─────────────────────────────

def run():
    rep = Report1("r10_predictability", "R10 — ¿Qué se puede predecir de un 3x?")
    rep.p(__doc__.split("\n", 2)[2])
    rng = np.random.default_rng(SEED)
    D = {name: load(t) for name, t in INDICES}

    rep.h("0. Base de datos y réplica sintética del 3x")
    rep.table(to_pb(base_table(D), ["f−r_media", "s_R3 sobre r", "s' sobre f", "ER"]),
              pct={"r_media", "f_media"}, digits=1)
    us = D["S&P 500"]["u_señal"]
    tr0 = r13_tr_start()
    spy0 = D["S&P 500"]["e"].underlying
    diff = signal_gap_bp(us, D["S&P 500"]["u"])
    gap_dp = float(diff[diff.index <= tr0].abs().max())
    gap_etf = diff[diff.index > etf_start(spy0)]
    rep.p("r: letras del Tesoro (^IRX desde 1960). f: fondos federales con un mes de retraso, "
          "el tipo al que se financia toda la exposición (L veces el capital); el efectivo del "
          "fondo rinde r. pb: puntos básicos anuales (0,01%). "
          "s_R3 es el sobrecoste del swap calibrado en R3 sobre las letras; s' = s_R3 − "
          "media(f − r) en la vida real del ETF, el sobrecoste sobre f con el que la réplica "
          "rf + 3(u − f − s') − ER reproduce al fondo real y, antes de su lanzamiento, paga la "
          "diferencia histórica entre f y r. Señales sin mirar el futuro: hasta 1988 el D/P de "
          "Shiller del mes en curso usa el precio medio del mes, que incluye cierres posteriores; "
          "por eso la caída, la tendencia y la volatilidad de 21 sesiones (las señales de §C y los "
          "estados de R9) se calculan hasta 1988 con el D/P del mes anterior y, desde el primer día "
          f"de ^SP500TR ({tr0.date()}), con la rentabilidad del índice oficial con dividendos, "
          f"también desde que u pasa a ser el ETF {spy0} (r10.sp500_signal_returns). "
          "Las rentabilidades con que se miden los resultados, los regresores HAR y las "
          "previsiones de varianza usan u. Hasta 1988 las dos series difieren como mucho "
          f"{num_es(gap_dp, 2)} pb en la rentabilidad de un día; desde {gap_etf.index[0].year} "
          f"({spy0} frente al índice), la diferencia diaria (con su signo) tiene una desviación típica de "
          f"{num_es(float(gap_etf.std()), 1)} pb y, en valor absoluto, una media de "
          f"{num_es(float(gap_etf.abs().mean()), 1)} pb y un máximo de {num_es(float(gap_etf.abs().max()), 0)} pb.")
    val = pd.DataFrame({name: replica_validation(d) for name, d in D.items()}).T
    rep.h("Validación de la réplica en la vida real de cada ETF", 3)
    rep.table(to_pb(val, ["s'_1ª_mitad"]),
              pct={"rent_anual_real", "rent_anual_réplica", "caída_máx_real",
                   "caída_máx_réplica", "rent_anual_real_2ª",
                   "rent_anual_réplica_2ª", "caída_máx_real_2ª", "caída_máx_réplica_2ª"},
              digits=4)
    y_live = sorted({x.year for x in val["desde"]})
    y_half = sorted({x.year for x in val["2ª_mitad_desde"]})
    span = lambda ys: str(ys[0]) if len(ys) == 1 else f"{ys[0]}–{ys[-1]}"  # noqa: E731
    rep.p("rent_anual: rentabilidad anual compuesta. Con s' calibrado en toda la vida del ETF, "
          "la deriva media de la réplica coincide con la del fondo por construcción: la "
          "validación informativa es la correlación diaria, la caída máxima y la segunda mitad, "
          "donde la réplica usa el s' calibrado solo con la primera. Las vidas de los ETF empiezan "
          f"en {span(y_live)} y sus segundas mitades en {span(y_half)}, hasta "
          f"{max(val['hasta']).year}, con tipos casi nulos durante años: no validan s' en los "
          "años setenta.")

    rep.h("A. La identidad: μ y σ² de la ventana explican el resultado del 3x (3 años)")
    ident = {name: identity(d) for name, d in D.items()}
    cols = ["R2", "sesgo", "RMSE_residuo", "cuota_μ", "cuota_σ²", "cuota_μ_sin_cov",
            "cuota_μ_con_cov", "RMSE_con_momentos", "reducción_error_cuadrático_momentos"]
    rep.table(pd.DataFrame({n: {k: r[k] for k in cols} for n, r in ident.items()}).T,
              pct=set(cols), digits=3)
    rep.p("Modelo: r + 3(μ − f − s') − ER − 4,5·σ², con los momentos realizados de la propia "
          "ventana. cuota_μ y cuota_σ²: parte de la varianza entre ventanas del crecimiento del "
          "3x que corresponde al término de rentabilidad y al de varianza, con la covarianza "
          "repartida a partes iguales; cuota_μ_sin_cov y cuota_μ_con_cov dan el rango de la "
          "cuota de μ si la covarianza se asigna entera a σ² o a μ. El término de rentabilidad es el "
          "de deriva completo, r + 3(μ − f − s') − ER: su cuota incluye también la variación entre "
          "ventanas de las letras y de la financiación, no solo la de μ. RMSE en puntos anuales. "
          "Momentos: el modelo más 252·(L³/3·E[u³] − L⁴/4·E[u⁴]), con los momentos brutos "
          "diarios del índice en la ventana; reducción_error_cuadrático_momentos = 1 − "
          "SSE(con momentos)/SSE(modelo). Es la ecuación central del artículo; la expansión de "
          "la rentabilidad diaria del propio fondo está en R11 §F.")
    rep.h("Por tramo: ventanas solo con la réplica, mixtas y solo con el fondo real", 3)
    rep.table(pd.concat({n: r["por_tramo"] for n, r in ident.items()}),
              pct={"R2", "RMSE_residuo", "RMSE_con_momentos", "reducción_error_cuadrático_momentos"},
              digits=3)
    fig_identity(ident, rep)

    rep.h("B. Predictibilidad en muestra y fuera de muestra")
    rows, coefs, vrows = {}, [], {}
    for name, d in D.items():
        for hn, h in HORIZONS.items():
            for tgt, cols in (("exceso", MU_VARS), ("log_var", HAR), ("g3x", MU_VARS)):
                p = predictive(d, tgt, cols, h)
                rows[(name, hn, tgt)] = {k: p[k] for k in ("n", "R2_IS", "R2_OOS", "R2_OOS_truncado",
                                                          "CW_t", "CW_p", "CW_estimador", "CW_p_NW2h")}
                if hn == "1 año":
                    f = p["fit"].frame()
                    f["t_NW2h"] = p["fit_nw2h"].t
                    f.insert(0, "modelo", f"{name} · {tgt}")
                    coefs.append(f)
            for m, v in var_levels(d, h).items():
                vrows[(name, hn, m)] = v
    tab = pd.DataFrame(rows).T
    tab.index.names = ["índice", "horizonte", "objetivo"]
    rep.table(tab, pct={"R2_IS", "R2_OOS", "R2_OOS_truncado"}, digits=3)
    rep.p(f"Previsión fuera de muestra con ventana creciente; la primera previsión usa "
          f"{MIN_OBS // 252} años de historia. Benchmark: la media histórica del objetivo con los "
          "datos ya observados. CW_t y CW_p: Clark-West con errores de Hansen-Hodrick de h − 1 "
          "retardos (h = sesiones del horizonte); si esa varianza no sale positiva, Newey-West "
          "con 2h retardos (CW_estimador dice cuál se usó). CW_p_NW2h: el mismo contraste con "
          "Newey-West de 2h retardos, como comprobación. R2_OOS_truncado: R² con la previsión "
          "del exceso truncada en 0 (una prima esperada no negativa); Campbell y Thompson (2008) "
          "restringen además el signo de los coeficientes, así que no es exactamente su R².")
    rep.h("Sensibilidad: el exceso aritmético y de dónde sale el R² del crecimiento del 3x", 3)
    srows = {}
    for name, d in D.items():
        for hn, h in HORIZONS.items():
            pa = predictive(d, "exceso_arit", MU_VARS, h)
            pv = predictive(d, "g3x", ["log_rv_m"], h)
            pc = predictive(d, "g3x", ["caída", "tendencia"], h)
            srows[(name, hn)] = {"R2_OOS_exceso_arit": pa["R2_OOS"],
                                 "R2_OOS_exceso_arit_truncado": pa["R2_OOS_truncado"],
                                 "CW_p_exceso_arit": pa["CW_p"],
                                 "R2_OOS_g3x_solo_volatilidad": pv["R2_OOS"],
                                 "R2_OOS_g3x_solo_caída_y_tendencia": pc["R2_OOS"]}
    st = pd.DataFrame(srows).T
    st.index.names = ["índice", "horizonte"]
    rep.table(st, pct={c for c in st.columns if c.startswith("R2")}, digits=3)
    rep.p("Añadido tras una revisión externa; no entra en los 18 contrastes de la tabla anterior. El "
          "objetivo «exceso» de §B es el exceso LOGARÍTMICO sobre las letras, ≈ μ − r − ½σ²: incluye la "
          "varianza de la ventana, que sí se puede prever. exceso_arit: el exceso aritmético, la media de "
          "u − r ×252 en la ventana (≈ μ − r, el que entra en L*), con las mismas tres variables, la misma "
          "ventana creciente y su versión truncada en 0. g3x_solo_volatilidad y g3x_solo_caída_y_tendencia: "
          "el R² fuera de muestra del crecimiento del 3x con solo la volatilidad reciente (log_rv_m) o solo "
          "la caída y la tendencia, para ver qué parte del R² viene del canal de la varianza.")
    rep.h("Coeficientes a 1 año (t de Hansen-Hodrick con 251 retardos; t_NW2h: Newey-West con 504)", 3)
    rep.table(pd.concat(coefs).reset_index(names="variable").set_index(["modelo", "variable"]),
              digits=3)
    rep.h("La varianza en niveles, la escala que entra en L*", 3)
    vt = pd.DataFrame(vrows).T
    vt.index.names = ["índice", "horizonte", "modelo"]
    rep.table(vt, pct={"R2_OOS_niveles", "R2_OOS_log"}, digits=3)
    rep.p("Objetivo: la varianza realizada anualizada de la ventana (la de L*). HAR-log + Duan: "
          "exponencial de la previsión del HAR logarítmico (la de la fila log_var) por el factor "
          "de Duan (1983) calculado en tiempo real con los cocientes realizada/prevista ya "
          f"observados (al menos max({DUAN_MIN}, h)); sin él, la exponencial da la media geométrica "
          "(la mediana solo si el error del logaritmo es simétrico), que queda por debajo de la media. Con h = 252 o 756 los cocientes se solapan mucho: el factor es la media "
          "de pocos años independientes. HAR en niveles: la misma regresión sobre las varianzas "
          "(la sensibilidad (ii) de §E). R2_OOS_niveles: la previsión en niveles frente a la "
          "media creciente embargada de la varianza. R2_OOS_log: la previsión del logaritmo "
          "frente a la media creciente del logaritmo; en el HAR logarítmico es la de la fila "
          "log_var, sin el factor de Duan (que corrige la media en niveles y en logaritmos sería "
          "un sesgo); en el HAR en niveles, el logaritmo de su previsión, que estima log E[y] y "
          "no E[log y] y por eso queda por encima. QLIKE: pérdida media del modelo y de la media "
          "histórica. Ventana común a los dos modelos, que empieza cuando existe el factor de "
          "Duan: por eso R2_OOS_log del HAR logarítmico no es exactamente el de la fila log_var. "
          "pérdida_crec_*: el crecimiento log anual que se pierde por fijar el apalancamiento con esa "
          "previsión y una prima constante m = μ − f − s' (la de toda la muestra), ½·SR²·(ρ − 1)² con "
          "ρ = realizada/prevista y SR² = m²/σ² (r10.growth_loss; R11 §C la usa a 21 sesiones), en puntos "
          "básicos, del modelo y de la media histórica, frente al apalancamiento que usaría la varianza "
          "realizada de cada ventana (una referencia que ningún modelo alcanza, así que los niveles son "
          "grandes y lo que informa es la comparación). ventaja_crec_pb = pérdida_crec_media_pb − "
          "pérdida_crec_modelo_pb: los puntos básicos de crecimiento log anual que gana el apalancamiento "
          "del modelo frente al de la media con esa prima, g(m/σ̂²_modelo) − g(m/σ̂²_media) exactamente "
          "(positiva: el modelo crece más). En la diferencia se cancela el término de la referencia, "
          "½m²·media(1/σ²), que no depende de la previsión y que el ruido de la varianza realizada infla; "
          "en un cociente de pérdidas no se cancela, y su tamaño dependería de esa referencia "
          "inalcanzable, así que se citan las diferencias y no cocientes. A diferencia de QLIKE, pesa más los periodos "
          "de calma y castiga más quedarse corto en la varianza prevista (ρ > 1: sobreapalancarse) que "
          "pasarse (ρ < 1: la pérdida no pasa de ½·SR²). Con h = 252 o 756 los errores no son "
          "pequeños y unos pocos años de calma con la varianza muy subestimada pueden dominar la media.")
    fig_oos(tab, rep)
    fig_har(D["S&P 500"], "S&P 500", rep)

    rep.h("C. Reglas de entrada y salida frente a mantener el 3x")
    rep.p("La familia son 15 reglas de exposición dentro o fuera del 3x, revisadas cada día "
          "con la información del cierre y ejecutadas al cierre siguiente; fuera, el dinero "
          "está en letras, y cada cambio cuesta 5 pb. Las reglas «dentro solo con caída ≥ k» "
          "salen del fondo en cuanto la caída se recupera por encima de k: no contrastan "
          "«comprar tras una caída y mantener», que se contrasta en R9 y en §F. La referencia "
          "es mantener el 3x, no el índice. La tendencia compara el índice con dividendos con su "
          "media de 200 sesiones, la regla de la LRS de Gayed y Bilello; en el S&P 500, desde 1988 "
          "con el índice oficial con dividendos (^SP500TR, la serie de los autores) y antes con el "
          "D/P de Shiller del mes anterior (§0), para no usar cierres posteriores a la señal. "
          "El Reality Check de White y el SPA de Hansen de esta familia están en R11 §D.")
    rc = {}
    for name, d in D.items():
        t = rule_table(d)
        rc[name] = t
        a = t.attrs
        rep.h(f"{name}: {num_es(a['dias'], 0)} días desde {a['desde']} · crecimiento del 3x "
              f"{pct_es(a['crec_3x'])} al año, letras {pct_es(a['r_letras'])}", 3)
        rep.table(t.rename(index=RULE_LABELS),
                  pct={"Δ_crec_anual", "tiempo_dentro", "crec_3x_dentro", "coste_mecánico",
                       "selección_y_costes"}, digits=3)
    rep.p("Δ_crec_anual: crecimiento log anual de la regla menos el de mantener el 3x. "
          "p_nominal: bootstrap estacionario (bloque de 63 sesiones) de la media de cada regla, "
          f"unilateral, con B = {num_es(B, 0)}; p_Holm y q_BH corrigen por las 15. tiempo_dentro: fracción "
          "de días dentro. crec_3x_dentro: crecimiento log anual del 3x en esos días. "
          "coste_mecánico = (1 − tiempo_dentro)·(r̄ − ḡ_3x): la Δ que tendría, sin costes, una "
          "regla que estuviera fuera la misma fracción de días elegidos al azar; "
          "selección_y_costes = Δ − coste_mecánico es lo que aporta elegir esos días, menos los "
          "costes de operar.")
    fig_rules(rc, rep)

    rep.h("D. Incertidumbre del apalancamiento óptimo (historia completa)")
    ls = {name: lstar_bootstrap(d, rng) for name, d in D.items()}
    keys = [kelly.LSTAR_APROX, "IC90_bajo", "IC90_alto", "P(L*<1,5)", "P(L*>3)",
            "P(g(3x)<letras)", "L_máx_media", "L_robusto_p5"]
    rep.table(pd.DataFrame({n: {k: b[k] for k in keys} for n, b in ls.items()}).T, digits=2)
    rep.p(f"{kelly.LSTAR_APROX} con toda la historia: {kelly.LSTAR_DESC[kelly.LSTAR_APROX]}. "
          f"IC 90% y probabilidades sobre {num_es(B_PAIRS, 0)} remuestreos estacionarios (bloques de un año) "
          "de (u, r, f). P(L* < 1,5): fracción de remuestreos en que el 3x queda por encima de "
          "2L*; sin la comisión, crecería menos que las letras a largo plazo. Con la comisión, "
          "g(3) < r ⇔ L* < 1,5 + ER/(3σ²): esa probabilidad es P(g(3x)<letras). En cada remuestreo "
          "se guardan (μ, σ², r̄, f̄) y se evalúa g(L) con costes: P(g(3x)<letras) es la fracción "
          "de remuestreos en que g(3) < r̄ (una probabilidad solo con un prior plano; la "
          "posterior con prior está en R11 §A), L_máx_media maximiza el crecimiento medio y "
          "L_robusto_p5 su percentil 5: el apalancamiento que mejor aguanta el error de "
          "estimación.")
    sens = pd.DataFrame({n: {f"{k} ({q})": getattr(b["semillas"][k], q)()
                             for k in ("L_robusto_p5", "L_máx_media", "P(g(3x)<letras)")
                             for q in ("min", "max")} for n, b in ls.items()}).T
    rep.h(f"Sensibilidad a la semilla ({SEEDS_LSTAR} semillas de {num_es(B_PAIRS, 0)} remuestreos)", 3)
    rep.table(sens, digits=2)
    fig_lstar(ls, rep)

    rep.h("E. L* ex ante (información disponible en cada fecha)")
    ex = {name: lstar_ex_ante(d) for name, d in D.items()}
    summ = {(name, c): v for name, df in ex.items() for c, v in ex_ante_summary(df).items()}
    et = pd.DataFrame(summ).T
    et.index.names = ["índice", "variante"]
    rep.table(et, pct={"fracción_días_L*≥3", "fracción_días_L*<1"}, digits=2)
    rep.p(f"L* aproximado ex ante: (μ̂ − f̂ − s')/σ̂². base: {EX_ANTE['base']}; la prima "
          f"{kelly.PRIMA} es la {kelly.PRIMA_DESC} (kelly.premium, la misma del medio Kelly de "
          "R12 y R13) y σ̂² la previsión del HAR logarítmico con corrección de Duan de la "
          "varianza del año siguiente. (i): la prima como media creciente de u menos el f_t de "
          "ese día, que supone que la rentabilidad total no cambia cuando cambia f. (ii): σ̂² del "
          "HAR en niveles, cuyo R² en niveles está en §B. mediana y fracciones de días: en la "
          "ventana común a las tres variantes (desde ventana_común_desde), para que cada "
          "variante cambie solo el estimador y no la muestra; primer_día: el primero de cada "
          "variante (la (ii) no espera al factor de Duan y empieza antes). hoy: último día con "
          "dato. La serie base empieza cuando existen la prima (10 años) y el factor de Duan con "
          "h = 252.")
    clipped = fig_ex_ante(ex, rep)
    rep.p("La figura recorta el eje a [−2, 8]; días fuera del recorte: "
          + ", ".join(f"{n} {k}" for n, k in clipped.items()) + ".")
    neg = []
    for name, df in ex.items():
        s0 = df.iloc[:, 0].dropna()
        spans = episode_spans((s0 < 0).to_numpy())
        txt = "; ".join(f"{s0.index[a]:%d-%m-%Y} a {s0.index[b]:%d-%m-%Y} "
                        f"({num_es(int((s0.iloc[a:b + 1] < 0).sum()), 0)} días)" for a, b in spans)
        neg.append(f"{name}: {txt or 'ninguno'}")
    rep.p("Tramos con L* ex ante negativo en la serie base (días con L* < 0 separados por más de 21 "
          "sesiones forman tramos distintos; entre paréntesis, los días negativos de cada tramo): "
          + ". ".join(neg) + ".")

    rep.h("F. Comprar con caída > 20% y mantener 3 años: diferencia de medianas del 3x")
    gaps = {name: median_gap_ci(d, rng) for name, d in D.items()}
    gt = pd.DataFrame(gaps).T
    rep.table(gt, pct={c for c in gt.columns if c.endswith("_gap")}, digits=3)
    rep.p("Estado: caída desde máximos ≤ −20% al cierre de t; compra al cierre de t+1 y "
          "crecimiento log anual del 3x en los 3 años siguientes, frente al resto de días. "
          f"IC por bootstrap estacionario con bloques de 3 años (B = {num_es(B, 0)}): el 90% y el de "
          "Bonferroni para tres índices, 1 − 0,10/3 = 96,7%. Placebo: el vector de estado se "
          f"desplaza circularmente ({num_es(N_PLACEBO, 0)} desplazamientos de al menos 3 años) y se "
          "recalcula el mismo estadístico; p bilateral y unilateral (caída mejor). episodios: "
          "tramos del estado separados por más de 21 sesiones. Las ventanas solapadas hacen que "
          "la muestra efectiva sea de unas pocas decenas de ventanas independientes; el "
          "bootstrap remuestrea esos pocos episodios y el placebo pregunta otra cosa (si el "
          "estado, colocado en otras fechas, da diferencias así de grandes), así que sus "
          "dispersiones no coinciden. Aunque la rentabilidad fuera impredecible, el estadístico no "
          "está centrado en cero: las subidas que ponen fin a una caída entran en la ventana de los "
          "días en caída, y las bajadas que la inician, en la de los días anteriores, que cuentan en el "
          "resto. Ni el bootstrap, que remuestrea pares de estado y resultado ya observados, ni el "
          "placebo, que separa cada caída de la recuperación que entra en su propia ventana, reproducen "
          "ese efecto: el IC es descriptivo y no se compara con el cero, y el p placebo exagera la "
          "evidencia a favor de comprar tras la caída. La tabla siguiente mide el estadístico sin "
          "ninguna predictibilidad.")
    rep.h("El mismo estadístico sin predictibilidad (H0 simulada)", 3)
    rng_null = np.random.default_rng(SEED_NULL)
    null_rows = {(name, "i.i.d." if b <= 1 else f"bloques de {b} sesiones"):
                 {"observado": gaps[name]["mediana_gap"],
                  **median_gap_null(d, gaps[name]["mediana_gap"], rng_null, block=b)}
                 for name, d in D.items() for b in (1, 63)}
    nt = pd.DataFrame(null_rows).T
    nt.index.names = ["índice", "remuestreo"]
    rep.table(nt, pct={"observado", "media_H0", "sd_H0"}, digits=3)
    rep.p(f"H0: se remuestrean los días (rentabilidad de la señal y del 3x, juntas) de toda la muestra de "
          f"cada índice, {num_es(N_NULL, 0)} trayectorias, i.i.d. o con bootstrap estacionario de bloque "
          "medio de 63 sesiones (conserva las rachas de volatilidad, no la predictibilidad a 3 años); en "
          "cada una se recalculan la caída desde máximos, el crecimiento del 3x a 3 años comprando al "
          "cierre siguiente y la diferencia de medianas. media_H0 y sd_H0: su media y su desviación "
          "típica; P(≥ observado): fracción de trayectorias sin efecto con una diferencia igual o mayor "
          "que la observada (el p unilateral bajo esta H0). tamaño_placebo_unilateral_5% y "
          f"tamaño_placebo_bilateral_5%: fracción de trayectorias sin efecto en que el placebo circular "
          f"({num_es(N_NULL_SHIFTS, 0)} desplazamientos) daría un p unilateral o bilateral ≤ 0,05, los dos p "
          "de la tabla anterior; si pasa de 0,05, el placebo rechaza de más. Cada tamaño vale solo para su "
          "p: el unilateral no dice cuánto rechaza el bilateral. Semilla propia (r10.SEED_NULL).")
    rep.save()


if __name__ == "__main__":
    run()
