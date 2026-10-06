"""R35 — El indicador del apalancamiento óptimo (artículo 7): muestra de desarrollo y configuración congelada.

Prerregistro: letf/preregistro/articulo7_indicador.md (commits aafe958 y 02faefc, con la ampliación a
bonos de «Desviaciones»). Cada día, para cada mercado (S&P 500 → SPY, SSO, UPRO; Nasdaq-100 → QQQ,
QLD, TQQQ; bono largo → TLT, UBT, TMF; bono medio → IEF, UST, TYD), el indicador estima el
crecimiento logarítmico esperado del propio fondo de reajuste diario para L ∈ {0, 1, 2, 3} en las
próximas h sesiones (h = 21 principal, h = 63 sensibilidad) y elige L.

Componentes (los del prerregistro; entre corchetes, lo que el prerregistro no fija y aquí se fija):
1. Objetivo. g(L) = rf + L·(μ − f − s_L) − ER_L − ½·L²·IV(h), anual. [L = 1 es el ETF 1x (o el
   índice antes de él): no paga financiación ni s, g(1) = μ − ½·IV; L = 0 son letras, g(0) = rf.]
2. Varianza: HAR-RI de R34 (r34.forecast): E[IV(h)] y su dispersión de(IV) en tiempo real.
3. Deriva con contracción. Prior: acciones, la media creciente de u − f con datos hasta t − 1
   (kelly.premium); bonos, el carry del bono sintético (rendimiento − financiación + rodadura) al
   cierre de t − 1. [Valor prudente: μ − f = 0 (ninguna prima sobre la financiación).] Peso de Kan y
   Zhou (2007, regla de dos fondos con N = 1 activo): w = (T − 5)/T · θ²_a/(θ²_a + 1/T), con θ̂² =
   (m/ee)²/T la razón de Sharpe diaria al cuadrado del prior frente a su error típico ee con T
   sesiones y θ²_a su estimador ajustado (Kan y Zhou, con la beta incompleta). [En bonos, ee es el
   error típico de la media de u − f − carry: cuánto se sabe de que el carry no está sesgado.]
   Tendencia: estado S_t = 1 si el índice con dividendos cierra por encima de su media de 200
   sesiones; su efecto β es la pendiente de la regresión de la media de las h sesiones siguientes
   de u − f (en bonos, de u − f − carry) sobre S − S̄, con los pares cuyo objetivo ya se conoce en
   t − 1, reestimada cada 21 sesiones con al menos 10 años de pares (t de Hansen y Hodrick con
   h − 1 retardos). μ − f = prior contraído + β·(S_t − S̄). [ee(μ)² = w·ee(prior)² + (S_t − S̄)²·
   ee(β)²: la varianza a posteriori del prior contraído más la del efecto de la tendencia.]
4. Kelly robusto con κ = 1: L* = (μ − f − s − κ·ee(μ))/(E[IV] + κ·de(IV)). [s del 3x; el L*
   continuo no descuenta la comisión, como kelly.LSTAR_APROX.] La recomendación discreta maximiza
   el crecimiento robusto g_κ(L) (la fórmula 1 con μ − f − κ·ee y E[IV] + κ·de), con los costes de
   cada fondo.
5. Ruina (EVT condicional de McNeil y Frey 2000): GJR-GARCH(1,1) gaussiano con ventana creciente
   (reestimado cada 252 sesiones, 10 años de mínimo) y Pareto generalizada sobre las pérdidas
   estandarizadas por encima de su cuantil 95%. p_t(L) = P(u_{t+1} ≤ −1/L | σ_{t+1}); se descarta
   L si 1 − (1 − p_t)^252 > 1/1.000. [L = 0 y L = 1 nunca se descartan.]
6. Histéresis: L_t = argmax_L [g_κ(L) − c(L_{t−1} → L)·252/h] entre los L permitidos, con c = 5 pb
   por unidad negociada (vender un fondo y comprar otro son dos unidades; las letras no pagan).
   Solo cambia si la mejora del crecimiento esperado en h sesiones supera el coste; si el L que se
   tiene deja de estar permitido, se cambia sin más.
7. Cartera de acciones y bonos (H8): S&P 500 y bono largo con el 3x de cada uno (UPRO y TMF, pesos
   L/3, resto en letras, L_acc + L_bono ≤ 3 como la HFEA), la combinación de la rejilla {0, 1, 2,
   3}² que maximiza el crecimiento robusto conjunto con la correlación prevista, cada L por debajo
   del máximo permitido de su activo, histéresis igual y reajuste trimestral y en cada cambio. La
   correlación prevista es un HAR sobre la z de Fisher de las correlaciones realizadas de 21, 63 y
   252 sesiones, estimado con ventana creciente y solo pares cuyo objetivo ya se conoce.

Protocolo del laboratorio: orden al cierre siguiente (EXEC_LAG = 2), 5 pb por unidad negociada,
letras al tipo del Tesoro, financiación interbancaria (history.funding_long), fondos sintéticos
calibrados con datos hasta 2018 antes de su lanzamiento y reales después (indicador.make_equity,
indicador.real_fund), bootstrap estacionario emparejado (bloque 63), SPA de Hansen y Romano–Wolf.

DISCIPLINA DE LA MUESTRA RESERVADA (1-1-2019 → 25-9-2026). Esta línea solo se ejecuta con datos
hasta el 31-12-2018: todas las series se recortan en `end` antes de calcular nada, run() rechaza
un end posterior y las pruebas lo comprueban. La prueba reservada la evalúa una sola vez
r36_indicador_holdout.run() con la configuración congelada (preregistro/articulo7_config_congelada.json).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import zlib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import norm

from letf_lab import kelly, metrics, provenance, stats
from letf_lab.data import history
from letf_lab.data import indicador as ind
from letf_lab.report import FIG_W, Report, comma_fig, legend_band, num_es, pct_es
from letf_lab.research import r10_predictability as r10
from letf_lab.research import r13_lrs as r13
from letf_lab.research import r24_folk as r24
from letf_lab.research import r34_harri as r34

log = logging.getLogger(__name__)

DEV_END, TEST_START, END = ind.DEV_END, ind.TEST_START, ind.END
DAYS = 252
LETF = Path(__file__).resolve().parents[2]                  # letf/
PREREG = LETF / "preregistro" / "articulo7_indicador.md"
FROZEN = LETF / "preregistro" / "articulo7_config_congelada.json"

EQUITY_MARKETS = ("S&P 500", "Nasdaq-100")
BOND_MARKETS = ("Bono largo", "Bono medio")
MARKETS = EQUITY_MARKETS + BOND_MARKETS
BOND_KIND = {"Bono largo": "largo", "Bono medio": "medio"}
EQUITY_START = {"S&P 500": pd.Timestamp(r34.START_SP), "Nasdaq-100": ind.NDX_FROM}

# Ruptura de serie de FRED: DGS20 salta +74 pb el 4-1-1973 con DGS10 en −2 pb (hasta entonces el
# 20 años rendía menos que el 10, efecto de los «bonos flor»; es un cambio de método, no un precio).
# Ese día la variación de la curva del bono sintético largo se pone a cero (queda carry + rodadura).
BOND_BREAKS = {"largo": ("1973-01-04",)}


@dataclass(frozen=True)
class Config:
    """Todos los parámetros del indicador y de su evaluación. Los valores por defecto son los
    congelados (articulo7_config_congelada.json); las pruebas usan versiones pequeñas."""

    horizons: tuple = (21, 63)          # h principal y de sensibilidad (prerregistro)
    h_main: int = 21
    kappa: float = 1.0                  # Kelly robusto (prerregistro: κ = 1 fijo)
    levels: tuple = (0, 1, 2, 3)
    ruin_p_annual: float = 1e-3         # restricción de ruina (prerregistro)
    evt_q: float = 0.95                 # umbral de la GPD: cuantil de las pérdidas estandarizadas
    evt_min: int = 2520                 # 10 años antes del primer ajuste GJR + GPD
    evt_refit: int = 252
    drift_min: int = kelly.MIN_OBS      # 10 años antes de la primera prima (kelly.premium)
    prudent: float = 0.0                # valor prudente de μ − f (anual)
    trend_n: int = 200                  # media de 200 sesiones (prerregistro)
    trend_min: int = 2520               # pares mínimos para estimar el efecto de la tendencia
    trend_refit: int = 21
    corr_windows: tuple = (21, 63, 252)
    corr_min: int = 2520
    corr_refit: int = 21
    corr_clip: float = 0.999
    cost_bps: float = 5.0
    exec_lag: int = 2                   # señal al cierre de t, orden al cierre de t + 1
    block: int = 63
    B: int = 2000
    B_mcs: int = r34.B_MCS
    seed: int = 35
    alpha: float = 0.10
    loss_p: float = 0.5                 # H4: probabilidad de una pérdida del 50%
    h5_share: float = 0.75              # H5: «casi toda» la variación de L* sale de la varianza
    hfea_w: float = 0.55
    sb_equity: str = "S&P 500"
    sb_bond: str = "Bono largo"
    sb_cap: int = 3                     # L_acc + L_bono ≤ 3 (pesos L/3 en los 3x, como la HFEA)
    harri_min_obs: int = r34.MIN_OBS
    harri_min_obs_extra: int = r34.MIN_OBS_EXTRA
    harri_refit: int = r34.REFIT
    harri_disp_min: int = r34.DISP_MIN
    bench_min_obs: int = r34.MIN_OBS    # HAR-log + Duan y GJR-GARCH de H1 y H6
    garch_refit: int = r34.GARCH_REFIT

    def harri_kw(self) -> dict:
        return {"min_obs": self.harri_min_obs, "min_obs_extra": self.harri_min_obs_extra,
                "refit": self.harri_refit, "disp_min": self.harri_disp_min}

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name for f in fields(cls)}
        extra = set(d) - names
        if extra:
            raise ValueError(f"parámetros desconocidos en la configuración: {sorted(extra)}")
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


PARAM_ES = {
    "horizons": "horizontes h (sesiones): principal y sensibilidad", "h_main": "horizonte principal",
    "kappa": "κ del Kelly robusto", "levels": "niveles de apalancamiento posibles",
    "ruin_p_annual": "probabilidad anual máxima de un día de ruina", "evt_q": "cuantil del umbral de la GPD",
    "evt_min": "sesiones mínimas antes del primer ajuste GJR + GPD", "evt_refit": "reestimación de GJR + GPD (sesiones)",
    "drift_min": "sesiones mínimas del prior de la deriva", "prudent": "valor prudente de μ − f (anual)",
    "trend_n": "media móvil del estado de tendencia (sesiones)", "trend_min": "pares mínimos del efecto de la tendencia",
    "trend_refit": "reestimación del efecto de la tendencia (sesiones)",
    "corr_windows": "ventanas de las correlaciones realizadas (sesiones)", "corr_min": "pares mínimos del HAR de correlación",
    "corr_refit": "reestimación del HAR de correlación (sesiones)", "corr_clip": "tope de |ρ| antes de la z de Fisher",
    "cost_bps": "coste por unidad negociada (pb)", "exec_lag": "retraso de ejecución (2: orden al cierre siguiente)",
    "block": "bloque medio del bootstrap estacionario", "B": "remuestreos del bootstrap y del SPA",
    "B_mcs": "remuestreos del MCS (H1, H6)", "seed": "semilla de la línea", "alpha": "nivel de los contrastes",
    "loss_p": "pérdida de H4", "h5_share": "parte mínima de la varianza en H5", "hfea_w": "peso de acciones de la HFEA",
    "sb_equity": "acciones de la cartera de H8", "sb_bond": "bono de la cartera de H8",
    "sb_cap": "tope de L_acc + L_bono en H8", "harri_min_obs": "HAR-RI: filas mínimas de la base",
    "harri_min_obs_extra": "HAR-RI: filas mínimas de cada bloque adicional", "harri_refit": "HAR-RI: reestimación (sesiones)",
    "harri_disp_min": "HAR-RI: cocientes mínimos de la dispersión",
    "bench_min_obs": "HAR-log y GJR-GARCH (H1, H6): filas mínimas", "garch_refit": "GJR-GARCH (H1, H6): reestimación",
}


def rng_for(cfg: Config, tag: str) -> np.random.Generator:
    """Generador propio de cada contraste (semilla de la línea + etiqueta), para que añadir uno no
    mueva los demás."""
    return np.random.default_rng([cfg.seed, zlib.crc32(tag.encode())])


# ───────────────────────────── mercados ─────────────────────────────

@dataclass
class Market:
    """Todo lo que el indicador necesita de un mercado, recortado a una fecha.

    u: rentabilidad total diaria del 1x (índice antes del ETF). x: {1: u, 2: fondo 2x, 3: fondo 3x}
    (sintéticos antes de su lanzamiento, reales después). spread y er: sobrecoste s y comisión de
    los fondos 2x y 3x. signal: rentabilidad con dividendos para la media de 200 sesiones (sin
    mirar el futuro). prior: prior anual de la deriva conocido al cierre de t (bonos: carry del
    bono sintético; acciones: None, se usa la media histórica). mi: entradas de HAR-RI."""

    name: str
    kind: str
    u: pd.Series
    rf: pd.Series
    f: pd.Series
    x: dict
    spread: dict
    er: dict
    tickers: dict
    signal: pd.Series
    prior: pd.Series | None
    mi: r34.MarketInputs
    launch: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    maturity: float | None = None       # bonos: vencimiento del sintético (indicador.maturity)

    def truncate(self, end) -> "Market":
        end = pd.Timestamp(end)
        c = (lambda s: None if s is None else s[s.index <= end].copy())
        return replace(self, u=c(self.u), rf=c(self.rf), f=c(self.f), x={k: c(v) for k, v in self.x.items()},
                       signal=c(self.signal), prior=c(self.prior), mi=self.mi.truncate(end))

    def last_date(self) -> pd.Timestamp:
        ss = [self.u, self.rf, self.f, self.signal, *self.x.values()] + ([self.prior] if self.prior is not None else [])
        return max([s.index.max() for s in ss if len(s)] + [self.mi.last_date()])


def equity_market(name: str, end=DEV_END) -> Market:
    """S&P 500 (desde 1960, como R10 y el prerregistro) o Nasdaq-100 (desde 1985-10): u, OHLC y
    VIX de r34.equity_inputs; fondos 2x y 3x de indicador.make_equity (s calibrado hasta 2018)."""
    end = ind.horizon(end)
    d = ind.make_equity(name, end)
    mi = r34.equity_inputs(name, end)
    u = mi.u.dropna()
    u = u[u.index >= EQUITY_START[name]]
    gap = (d["u"].reindex(u.index) - u).abs().max()
    assert gap < 1e-12, f"{name}: la u de R34 y la de indicador.make_equity no coinciden ({gap})"
    idx = u.index
    x = {1: u, 2: d["x2"].reindex(idx), 3: d["x3"].reindex(idx)}
    signal = r13.sp500_signal_returns(u) if name == "S&P 500" else u
    notes = [f"u desde el {r34.dmy(idx[0])} (S&P 500: history.sp500_long; Nasdaq-100: common.long_underlying)",
             "fondos: " + ", ".join(f"{d['tickers'][L]} real desde el {r34.dmy(d['launch'][L])}" for L in (2, 3))]
    if name == "S&P 500":
        notes.append("señal de la tendencia: r13.sp500_signal_returns (D/P del mes anterior hasta 1988, ^SP500TR después)")
    return Market(name, "acciones", u, d["rf"].reindex(idx), d["f"].reindex(idx), x, dict(d["spread"]),
                  dict(d["er"]), dict(d["tickers"]), signal, None, mi.truncate(end), dict(d["launch"]), notes)


def fix_breaks(u: pd.Series, bond: pd.DataFrame, kind: str) -> tuple[pd.Series, list]:
    """Rentabilidad del bono con las rupturas de serie de BOND_BREAKS corregidas: ese día solo
    carry + rodadura (sin la variación de la curva)."""
    u = u.copy()
    done = []
    for d in BOND_BREAKS.get(kind, ()):
        d = pd.Timestamp(d)
        if d in u.index and d in bond.index:
            old = float(u[d])
            u[d] = float(bond.at[d, "carry"] + bond.at[d, "roll"])
            done.append((d, old, float(u[d])))
    return u, done


def bond_market(name: str, end=DEV_END) -> Market:
    """Bono largo (TLT, UBT, TMF) o medio (IEF, UST, TYD) del agente de datos: 1x sintético desde
    1962 y ETF real desde 2002 (indicador.bond_underlying) con BOND_BREAKS corregidas; fondos 2x y
    3x sintéticos con ese 1x (r24.lev_fund, s de indicador.fund_spread) y reales desde su
    lanzamiento; prior = carry del bono sintético; HAR-RI con el rango del ETF y el MOVE."""
    kind = BOND_KIND[name]
    end = ind.horizon(end)
    M = ind.maturity(kind)
    syn = ind.cmt_bond(M, end)
    u, done = fix_breaks(ind.bond_underlying(kind, end), syn, kind)
    u = u.dropna()
    idx = u.index
    rf, f = history.rf_long(idx), history.funding_long(idx)
    tick = dict(ind.DECISION[kind])
    x = {1: u}
    spread, er, launch = {}, {}, {}
    for L in (2, 3):
        e = ind.BOND_FUNDS[tick[L]]
        spread[L], er[L] = ind.fund_spread(tick[L]), e.er
        real = ind.real_fund(tick[L], end)
        launch[L] = real.index[0]
        synth = r24.lev_fund(u, rf, f, L, e.er, spread[L])
        x[L] = pd.concat([synth[synth.index < launch[L]], real]).sort_index()
        x[L] = x[L][~x[L].index.duplicated()].reindex(idx).rename(tick[L])
    etf = ind.BOND_ETF[kind]
    ohlc = ind.ohlc(etf, end)
    mv = ind.move(end)["move"].dropna()
    notes = [f"u: bono sintético a la par de {num_es(M, 1)} años (FRED) hasta el primer rendimiento de {etf} y "
             f"{etf} real después", f"rango: {etf} desde el {r34.dmy(ohlc.index[0])}",
             f"MOVE desde el {r34.dmy(mv.index[0])} (× duración/100)" if len(mv) else "sin MOVE"]
    notes += [f"ruptura de serie el {r34.dmy(d)}: rentabilidad {pct_es(a, 2)} → {pct_es(b, 3)} (solo carry y rodadura)"
              for d, a, b in done]
    mi = r34.bond_inputs(name, u, ohlc, mv if len(mv) else None, f=f, dur=syn["dur"], notes=notes).truncate(end)
    return Market(name, "bonos", u, rf, f, x, spread, er, tick, u, syn["prior_ann"].reindex(idx), mi, launch, notes,
                  float(M))


def load_market(name: str, end=DEV_END) -> Market:
    return equity_market(name, end) if name in EQUITY_MARKETS else bond_market(name, end)


def load_all(end=DEV_END, names=MARKETS) -> dict[str, Market]:
    return {n: load_market(n, end) for n in names}


# ───────────────────────────── 3. deriva con contracción ─────────────────────────────

def kz_theta2_adjusted(theta2, T, N: int = 1) -> np.ndarray:
    """Estimador ajustado de θ² de Kan y Zhou (2007) a partir de θ̂² (razón de Sharpe por
    periodo al cuadrado) con T observaciones y N activos:
    θ²_a = ((T − N − 2)θ̂² − N)/T + 2(θ̂²)^{N/2}(1 + θ̂²)^{−(T−2)/2} / (T·B_{θ̂²/(1+θ̂²)}(N/2, (T − N)/2)),
    con B_x(a, b) la función beta incompleta (sin normalizar). Siempre positivo."""
    th = np.asarray(theta2, float)
    T = np.broadcast_to(np.asarray(T, float), th.shape)
    a, b = N / 2, (T - N) / 2
    out = np.full(th.shape, np.nan)
    ok = np.isfinite(th) & np.isfinite(T) & (T > N + 4) & (th >= 0)
    pos = ok & (th > 0)
    if pos.any():
        t2, TT, bb = th[pos], T[pos], b[pos]
        x = t2 / (1 + t2)
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            logB = np.log(special.betainc(a, bb, x)) + special.betaln(a, bb)
            lognum = np.log(2) + (N / 2) * np.log(t2) - (TT - 2) / 2 * np.log1p(t2)
            corr = np.exp(lognum - np.log(TT) - logB)
        out[pos] = ((TT - N - 2) * t2 - N) / TT + np.nan_to_num(corr, nan=0.0, posinf=0.0)
    zero = ok & (th == 0)
    out[zero] = 0.0          # límite de θ̂² → 0 (con N = 1: −1/T + 1/T)
    return out


def kz_weight(m, se, T, N: int = 1) -> np.ndarray:
    """Peso de Kan y Zhou (2007) sobre la media estimada m (su error típico se, T observaciones):
    c = (T − N − 1)(T − N − 4)/(T(T − 2)) · θ²_a/(θ²_a + N/T), con θ̂² = (m/se)²/T. Entre 0 y 1."""
    m, se, T = (np.asarray(v, float) for v in (m, se, T))
    with np.errstate(divide="ignore", invalid="ignore"):
        th = np.where(se > 0, (m / se) ** 2 / T, np.nan)
        ta = kz_theta2_adjusted(th, T, N)
        k = (T - N - 1) * (T - N - 4) / (T * (T - 2))
        w = k * ta / (ta + N / T)
    return np.clip(w, 0.0, 1.0)


def _fit_times(n: int, refit: int) -> range:
    return range(0, n, refit)


def trend_effect(e: pd.Series, S: pd.Series, h: int, cfg: Config) -> pd.DataFrame:
    """Efecto de la tendencia estimado con datos hasta t − 1: pendiente β de la regresión de la
    media anual de e en (s, s + h] sobre S_s − S̄, con los pares s + h ≤ t − 1; reestimación cada
    cfg.trend_refit sesiones (en las posiciones múltiplo de trend_refit), al menos cfg.trend_min
    pares; error típico de Hansen y Hodrick con h − 1 retardos (stats.hac_overlap)."""
    n = len(e)
    ev = e.to_numpy(float)
    ok_e = np.isfinite(ev)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok_e, ev, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok_e)])
    y = np.full(n, np.nan)
    j = np.arange(n - h)
    full = (cn[j + h + 1] - cn[j + 1]) == h
    y[j[full]] = (cs[j[full] + h + 1] - cs[j[full] + 1]) / h * DAYS
    Sv = S.reindex(e.index).to_numpy(float)
    out = np.full((n, 3), np.nan)                       # β, ee(β), S̄
    cur = None
    for t in _fit_times(n, cfg.trend_refit):
        k = t - 1 - h                                   # último s con objetivo conocido en t − 1
        if k >= 0:
            m = np.isfinite(y[: k + 1]) & np.isfinite(Sv[: k + 1])
            if m.sum() >= cfg.trend_min:
                ss = Sv[: k + 1][m]
                sbar = float(ss.mean())
                z = ss - sbar
                if z.std() > 0:
                    fit = stats.ols_hac(pd.Series(y[: k + 1][m]), pd.DataFrame({"z": z}), **stats.hac_overlap(h))
                    cur = (float(fit.beta[1]), float(fit.se[1]), sbar)
        if cur is not None:
            out[t: t + cfg.trend_refit] = cur
    return pd.DataFrame(out, index=e.index, columns=["beta", "se_beta", "sbar"])


def drift(mk: Market, h: int, cfg: Config) -> pd.DataFrame:
    """Deriva μ − f (anual) conocida al cierre de t, con su error típico ee(μ).

    Columnas: prior, se_prior, T (sesiones), w_kz, shrunk (prior contraído), S (estado de
    tendencia en t), beta, se_beta, sbar, trend = β·(S − S̄), mu_f = shrunk + trend y ee."""
    u = mk.u
    x = (u - mk.f.reindex(u.index)).astype(float)
    if mk.prior is None:                                     # acciones: media histórica hasta t − 1
        e = x
        m = kelly.premium(u, mk.f, cfg.drift_min)
    else:                                                    # bonos: carry al cierre de t − 1
        p = mk.prior.reindex(u.index).shift(1)
        m = p
        e = x - p / DAYS
    err = e.shift(1)
    T = err.expanding().count()
    se = err.expanding(cfg.drift_min).std() * DAYS / np.sqrt(T)
    m = m.where(se.notna())
    w = pd.Series(kz_weight(m - cfg.prudent, se, T), index=u.index).where(m.notna())
    shrunk = cfg.prudent + w * (m - cfg.prudent)
    level = (1 + mk.signal.reindex(u.index).fillna(0.0)).cumprod()
    S = r13.above_sma(level, cfg.trend_n)
    te = trend_effect(e, S, h, cfg)
    dev = S - te.sbar
    trend = te.beta * dev
    ee = np.sqrt(w * se ** 2 + (dev * te.se_beta) ** 2)
    return pd.DataFrame({"prior": m, "se_prior": se, "T": T, "w_kz": w, "shrunk": shrunk, "S": S,
                         "beta": te.beta, "se_beta": te.se_beta, "sbar": te.sbar, "trend": trend,
                         "mu_f": shrunk + trend, "ee": ee})


# ───────────────────────────── 5. restricción de ruina (EVT condicional) ─────────────────────────────

def gpd_sf(fit: dict, x: np.ndarray, losses: np.ndarray) -> np.ndarray:
    """P(pérdida estandarizada ≥ x) con la cola GPD por encima del umbral y la frecuencia empírica
    por debajo (stats.gpd_exceedance_prob, vectorizado)."""
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    above = np.isfinite(x) & (x > fit["u"])
    if above.any():
        xi, be, u0 = fit["xi"], fit["beta"], fit["u"]
        z = 1 + xi * (x[above] - u0) / be
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            p = np.where(z > 0, fit["zeta"] * z ** (-1 / xi), 0.0) if abs(xi) > 1e-12 else \
                fit["zeta"] * np.exp(-(x[above] - u0) / be)
        out[above] = p
    below = np.isfinite(x) & ~above
    if below.any():
        s = np.sort(losses)
        out[below] = 1 - np.searchsorted(s, x[below], side="left") / len(s)
    return out


def evt_ruin(u: pd.Series, cfg: Config, levels=(2, 3)) -> pd.DataFrame:
    """Probabilidad condicional de un día con u_{t+1} ≤ −1/L (el fondo L pierde todo), conocida al
    cierre de t: GJR-GARCH(1,1) con los datos hasta el inicio de cada tramo de cfg.evt_refit
    sesiones (como r34.gjr_forecast) y GPD sobre las pérdidas estandarizadas del ajuste por encima
    de su cuantil cfg.evt_q (McNeil y Frey 2000). Columnas: sigma (desviación diaria prevista),
    xi, beta_gpd y umbral de la GPD en vigor, p_dia_L, p_anual_L = 1 − (1 − p)^252, ok_L."""
    r = u.dropna().astype(float)
    rv = r.to_numpy()
    n = len(rv)
    cols = ["sigma", "xi", "beta_gpd", "umbral"] + [f"p_dia_{L}" for L in levels]
    out = np.full((n, len(cols)), np.nan)
    for start in range(cfg.evt_min, n, cfg.evt_refit):
        p = stats.garch_fit(rv[:start], "gjr")
        mu = float(rv[:start].mean())
        hn = stats.garch_filter(rv - mu, p)                  # varianza de t + 1 conocida al cierre de t
        z = (rv[1:start] - mu) / np.sqrt(hn[: start - 1])
        losses = -z
        fit = stats.gpd_tail(losses, cfg.evt_q)
        seg = slice(start, min(start + cfg.evt_refit, n))
        sig = np.sqrt(hn[seg])
        out[seg, 0] = sig
        out[seg, 1:4] = (fit["xi"], fit["beta"], fit["u"])
        for j, L in enumerate(levels):
            out[seg, 4 + j] = gpd_sf(fit, (1 / L + mu) / sig, losses)
    df = pd.DataFrame(out, index=r.index, columns=cols).reindex(u.index)
    for L in levels:
        pd_ = df[f"p_dia_{L}"].clip(0, 1)
        df[f"p_anual_{L}"] = -np.expm1(DAYS * np.log1p(-pd_.clip(upper=1 - 1e-15)))
        df[f"ok_{L}"] = (df[f"p_anual_{L}"] <= cfg.ruin_p_annual).where(df[f"p_anual_{L}"].notna())
    return df


# ───────────────────────────── 1, 4 y 6. crecimiento, Kelly robusto e histéresis ─────────────────────────────

def growth_table(mu_f, ee, V, de, rf, f, spread: dict, er: dict, kappa: float) -> pd.DataFrame:
    """Crecimiento esperado anual de cada L sobre las letras, g(L) − rf, con la deriva y la
    varianza robustas (μ − f − κ·ee y E[IV] + κ·de; κ = 0: el crecimiento sin recortes):
    L = 0: 0; L = 1: (μ − f − κ·ee) + (f − rf) − ½·D; L ≥ 2: L·(μ − f − s_L − κ·ee) − ER_L − ½·L²·D."""
    num = mu_f - kappa * ee
    D = V + kappa * de
    fr = (f - rf) * DAYS
    g = {0: pd.Series(0.0, index=mu_f.index).where(num.notna() & D.notna()),
         1: num + fr - 0.5 * D}
    for L in (2, 3):
        g[L] = L * (num - spread[L]) - er[L] - 0.5 * L ** 2 * D
    return pd.DataFrame({f"g{L}": g[L] for L in (0, 1, 2, 3)})


def switch_cost(prev: int, new: int, cost_bps: float) -> float:
    """Coste (fracción del capital) de pasar del fondo prev al new: 5 pb por unidad negociada; las
    letras (L = 0) no pagan."""
    if prev == new:
        return 0.0
    return cost_bps / 1e4 * ((prev != 0) + (new != 0))


def choose(G: np.ndarray, allowed: np.ndarray, h: int, cost_bps: float, costs=None,
           start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Elección diaria con histéresis. G: n × K crecimientos robustos (anuales) de las K opciones;
    allowed: n × K permitidas; costs[i][j]: coste de pasar de i a j (por defecto switch_cost entre
    niveles). Devuelve (objetivo sin histéresis, opción que se tiene con histéresis): argmax de
    G − coste·252/h entre las permitidas, empezando en la opción start. NaN (−1) mientras falte
    algún dato; los empates van a la opción de índice menor (menos apalancamiento)."""
    n, K = G.shape
    if costs is None:
        costs = np.array([[switch_cost(i, j, cost_bps) for j in range(K)] for i in range(K)])
    target = np.full(n, -1)
    held = np.full(n, -1)
    prev = start
    amort = DAYS / h
    for t in range(n):
        g = G[t]
        if not np.isfinite(g).all():
            continue
        ok = allowed[t]
        gm = np.where(ok, g, -np.inf)
        target[t] = int(np.argmax(gm))
        net = np.where(ok, g - costs[prev] * amort, -np.inf)
        prev = int(np.argmax(net))
        held[t] = prev
    return target, held


def indicator(mk: Market, h: int, cfg: Config, P: pd.DataFrame, evt: pd.DataFrame) -> pd.DataFrame:
    """El indicador diario de un mercado y horizonte con la previsión HAR-RI P (r34.forecast) y la
    restricción de ruina evt (evt_ruin). Columnas: la deriva (drift), V = E[IV(h)], de, D = V + κ·de,
    g0…g3 (crecimiento robusto sobre las letras, anual), gp0…gp3 (sin recortes, κ = 0), ok_2, ok_3,
    L_star (continuo, s del 3x), L_objetivo (discreto sin histéresis) y L (con histéresis)."""
    idx = mk.u.index
    dr = drift(mk, h, cfg)
    V = P["media"].reindex(idx)
    de = P["de"].reindex(idx)
    rf, f = mk.rf.reindex(idx), mk.f.reindex(idx)
    G = growth_table(dr.mu_f, dr.ee, V, de, rf, f, mk.spread, mk.er, cfg.kappa)
    Gp = growth_table(dr.mu_f, dr.ee, V, de, rf, f, mk.spread, mk.er, 0.0)
    Gp.columns = [c.replace("g", "gp") for c in Gp.columns]
    ev = evt.reindex(idx)
    ok = pd.DataFrame({"ok_0": True, "ok_1": True, "ok_2": ev["ok_2"], "ok_3": ev["ok_3"]}, index=idx)
    need = pd.concat([G, ev[["ok_2", "ok_3"]]], axis=1).notna().all(axis=1)
    Gv = G.where(need).to_numpy(float)
    okv = ok.fillna(False).astype(bool).to_numpy()
    target, held = choose(Gv, okv, h, cfg.cost_bps)
    D = V + cfg.kappa * de
    out = pd.concat([dr, pd.DataFrame({"V": V, "de": de, "D": D}, index=idx), G, Gp,
                     ev[["sigma", "p_anual_2", "p_anual_3"]], ok[["ok_2", "ok_3"]]], axis=1)
    out["num"] = dr.mu_f - cfg.kappa * dr.ee - mk.spread[3]
    out["L_star"] = out["num"] / D
    out["L_objetivo"] = pd.Series(target, index=idx).where(target >= 0).astype(float)
    out["L"] = pd.Series(held, index=idx).where(held >= 0).astype(float)
    out["modelo_varianza"] = P["modelo"].reindex(idx)
    return out


# ───────────────────────────── carteras ─────────────────────────────

def run_levels(mk: Market, L: pd.Series, cfg: Config) -> pd.Series:
    """Rentabilidad diaria de tener el fondo L (0 letras, 1 el 1x, 2 el 2x, 3 el 3x) decidido al
    cierre de t: orden al cierre de t + 1 (r24.run_multi con lag = exec_lag) y 5 pb por unidad."""
    a = {f"x{k}": mk.x[k] for k in (1, 2, 3)}
    a["rf"] = mk.rf
    w = pd.DataFrame({f"x{k}": (L == k).astype(float).where(L.notna()) for k in (1, 2, 3)}, index=mk.u.index)
    return r24.run_multi(a, w, cfg.exec_lag, cfg.cost_bps)


def lrs_state(mk: Market, cfg: Config) -> pd.Series:
    """Estado de la LRS de Gayed y Bilello: 1 si el índice con dividendos (signal) cierra por encima
    de su media de 200 sesiones (r13.above_sma sobre r13.tr_index)."""
    level = (1 + mk.signal.reindex(mk.u.index).fillna(0.0)).cumprod()
    return r13.above_sma(level, cfg.trend_n)


def run_lrs(mk: Market, cfg: Config) -> pd.Series:
    """LRS 3x original: el 3x sobre la media de 200 sesiones, letras debajo; orden al cierre siguiente."""
    a = {"x3": mk.x[3], "rf": mk.rf}
    return r24.run_multi(a, pd.DataFrame({"x3": lrs_state(mk, cfg)}), cfg.exec_lag, cfg.cost_bps)


def drift_portfolio(X: pd.DataFrame, rf: pd.Series, target: pd.DataFrame, rebalance: pd.Series,
                    lag: int, cost_bps: float) -> pd.Series:
    """Cartera de fondos más letras cuyas posiciones derivan entre reajustes. target: pesos de los
    fondos decididos al cierre de t (el resto, letras); rebalance: días en que se ordena volver al
    objetivo. La orden se ejecuta al cierre de t + lag − 1 con cost_bps por unidad negociada en los
    fondos (las letras no pagan). Empieza en letras el primer día con objetivo."""
    idx = X.index
    Xv = X.to_numpy(float)
    rfv = rf.reindex(idx).to_numpy(float)
    tv = target.reindex(idx).to_numpy(float)
    rb = rebalance.reindex(idx).fillna(False).to_numpy(bool)
    n, k = Xv.shape
    rets = np.full(n, np.nan)
    h = np.zeros(k)
    cash, V = 1.0, 1.0
    started = False
    pending: list[tuple[int, np.ndarray]] = []
    cost = cost_bps / 1e4
    for i in range(n):
        if started:
            xr = np.where(np.isfinite(Xv[i]), Xv[i], 0.0)
            h = h * (1 + xr)
            cash *= 1 + (rfv[i] if np.isfinite(rfv[i]) else 0.0)
            Vn = h.sum() + cash
            while pending and pending[0][0] == i:
                tg = pending.pop(0)[1]
                turn = np.abs(tg * Vn - h).sum() / Vn
                Vn *= 1 - turn * cost
                h = tg * Vn
                cash = Vn - h.sum()
            rets[i] = Vn / V - 1
            V = Vn
        if rb[i] and np.isfinite(tv[i]).all():
            started = True
            pending.append((i + lag - 1, tv[i].copy()))
    return pd.Series(rets, index=idx)


# ───────────────────────────── 7. correlación y cartera de acciones y bonos ─────────────────────────────

def corr_forecast(ue: pd.Series, ub: pd.Series, h: int, cfg: Config) -> pd.DataFrame:
    """Correlación prevista entre dos rentabilidades para las h sesiones siguientes: HAR sobre la z
    de Fisher de las correlaciones realizadas de cfg.corr_windows sesiones, con ventana creciente,
    reestimado cada cfg.corr_refit sesiones y solo con los pares cuyo objetivo ya se conoce en t
    (s ≤ t − h), al menos cfg.corr_min. Columnas: rho (prevista), realizada (en t + 1 … t + h) y
    rho_63 (la realizada de las 63 últimas, referencia ingenua)."""
    j = pd.concat([ue, ub], axis=1, keys=["a", "b"]).dropna()
    c = cfg.corr_clip
    fz = lambda r: np.arctanh(r.clip(-c, c))  # noqa: E731
    X = pd.DataFrame({f"z{w}": fz(j.a.rolling(w).corr(j.b)) for w in cfg.corr_windows})
    fwd = j.a.rolling(h).corr(j.b).shift(-h)
    y = fz(fwd)
    Xa, ya = X.to_numpy(float), y.to_numpy(float)
    n = len(j)
    zf = np.full(n, np.nan)
    beta = None
    okX = np.isfinite(Xa).all(axis=1)
    for t in range(n):
        if t % cfg.corr_refit == 0:
            kn = t - h + 1                                   # filas s ≤ t − h
            if kn > 0:
                m = okX[:kn] & np.isfinite(ya[:kn])
                if m.sum() >= cfg.corr_min:
                    beta = r34._ols(Xa[:kn][m], ya[:kn][m])
        if beta is not None and okX[t]:
            zf[t] = beta[0] + Xa[t] @ beta[1:]
    return pd.DataFrame({"rho": np.tanh(zf), "realizada": fwd, "rho_63": j.a.rolling(63).corr(j.b)},
                        index=j.index)


def sb_combos(cap: int = 3) -> list[tuple[int, int]]:
    return [(e, b) for e in range(4) for b in range(4) if e + b <= cap]


def stock_bond(Ie: pd.DataFrame, Ib: pd.DataFrame, rho: pd.Series, me: Market, mb: Market, h: int,
               cfg: Config) -> pd.DataFrame:
    """Combinación (L_acc, L_bono) de cada día para la cartera de acciones y bonos: pesos L/3 en el
    3x de cada activo y el resto en letras, L_acc + L_bono ≤ cfg.sb_cap. Crecimiento robusto sobre
    las letras: Σ L_i·(μ_i − f − s3_i − κ·ee_i) − Σ (L_i/3)·ER3_i − ½·L'ΣL, con Σ de varianzas
    D_i = E[IV_i] + κ·de_i y la correlación prevista; cada L_i ≤ el mayor L permitido en su activo por
    la restricción de ruina; histéresis con 5 pb por unidad de peso negociada."""
    idx = Ie.index.intersection(Ib.index)
    combos = sb_combos(cfg.sb_cap)
    Ne = (Ie.mu_f - cfg.kappa * Ie.ee - me.spread[3]).reindex(idx)
    Nb = (Ib.mu_f - cfg.kappa * Ib.ee - mb.spread[3]).reindex(idx)
    De, Db = Ie.D.reindex(idx), Ib.D.reindex(idx)
    r = rho.reindex(idx)
    cov = r * np.sqrt(De * Db)
    lmax = {}
    for k, I in (("e", Ie), ("b", Ib)):
        J = I.reindex(idx)
        lm = pd.Series(1.0, index=idx) + J.ok_2.fillna(False).astype(float) * (1 + J.ok_3.fillna(False).astype(float))
        lmax[k] = lm.where(J.ok_2.notna() & J.ok_3.notna())
    G = np.column_stack([(le * Ne + lb * Nb - le / 3 * me.er[3] - lb / 3 * mb.er[3]
                          - 0.5 * (le ** 2 * De + lb ** 2 * Db + 2 * le * lb * cov)).to_numpy(float)
                         for le, lb in combos])
    G[~(np.isfinite(lmax["e"].to_numpy()) & np.isfinite(lmax["b"].to_numpy()))] = np.nan
    allowed = np.column_stack([((le <= lmax["e"]) & (lb <= lmax["b"])).to_numpy() for le, lb in combos])
    c = cfg.cost_bps / 1e4
    costs = np.array([[c * (abs(a[0] - b[0]) + abs(a[1] - b[1])) / 3 for b in combos] for a in combos])
    target, held = choose(G, allowed, h, cfg.cost_bps, costs=costs, start=0)
    pick = lambda k, j: pd.Series([combos[i][j] if i >= 0 else np.nan for i in k], index=idx)  # noqa: E731
    out = pd.DataFrame({"L_acc": pick(held, 0), "L_bono": pick(held, 1),
                        "L_acc_objetivo": pick(target, 0), "L_bono_objetivo": pick(target, 1),
                        "rho": r, "lmax_acc": lmax["e"], "lmax_bono": lmax["b"]}, index=idx)
    out["g"] = [G[t, i] if i >= 0 else np.nan for t, i in enumerate(held)]
    return out


def run_stock_bond(sb: pd.DataFrame, me: Market, mb: Market, cfg: Config) -> pd.Series:
    """Rentabilidad de la cartera de acciones y bonos: pesos L/3 en el 3x de cada activo (los de la
    HFEA: UPRO y TMF), letras el resto; reajuste al objetivo cuando cambia la combinación y al final
    de cada trimestre (como la HFEA), orden al cierre siguiente, 5 pb por unidad negociada."""
    idx = sb.index
    X = pd.DataFrame({"acc": me.x[3].reindex(idx), "bono": mb.x[3].reindex(idx)})
    tgt = pd.DataFrame({"acc": sb.L_acc / 3, "bono": sb.L_bono / 3}, index=idx)
    key = sb.L_acc * 10 + sb.L_bono
    change = key.notna() & (key != key.shift(1))
    qe = r24.quarter_ends(idx)
    rb = change | pd.Series(idx.isin(list(qe)), index=idx)
    rb &= key.notna()
    return drift_portfolio(X, me.rf.reindex(idx), tgt, rb, cfg.exec_lag, cfg.cost_bps)


def run_hfea(me: Market, mb: Market, start, end, cfg: Config) -> pd.Series:
    """HFEA 55/45 (UPRO y TMF, reajuste trimestral; r24.hfea) comprada al cierre anterior a start."""
    idx = me.u.index
    a = {"x3": me.x[3], "tmf": mb.x[3].reindex(idx), "u": me.u, "rf": me.rf}
    return r24.hfea(a, cfg.hfea_w, start, end, lag=cfg.exec_lag)[0]


# ───────────────────────────── todo junto ─────────────────────────────

def compute(end=DEV_END, cfg: Config | None = None, markets: dict | None = None) -> dict:
    """El indicador de los cuatro mercados y los dos horizontes, sus carteras, la LRS, los fijos, la
    cartera de acciones y bonos y los benchmarks de H1 y H6, con datos ≤ end (markets: ya cargados,
    p. ej. sintéticos en las pruebas; se recortan igualmente a end). end = END solo lo usa r36."""
    cfg = cfg or Config()
    end = pd.Timestamp(end)
    M = markets if markets is not None else load_all(end)
    M = {k: v.truncate(end) for k, v in M.items()}
    for k, v in M.items():
        r34.assert_until(v.mi, end, k)
        assert v.last_date() <= end, f"{k}: datos posteriores a {end.date()}"
    R = {"cfg": cfg, "end": end, "markets": M, "var": {}, "evt": {}, "ind": {}, "ret": {}, "expo": {},
         "bench": {}, "corr": {}, "sb": {}, "sb_ret": {}}
    for name, mk in M.items():
        log.info("R35: %s", name)
        R["var"][name] = r34.forecast(mk.mi, horizons=cfg.horizons, end=end, **cfg.harri_kw())
        R["evt"][name] = evt_ruin(mk.u, cfg)
        ret, expo = {}, {}
        for h in cfg.horizons:
            I = indicator(mk, h, cfg, R["var"][name][h], R["evt"][name])
            R["ind"][(name, h)] = I
            ret[f"indicador h={h}"] = run_levels(mk, I.L, cfg)
            expo[f"indicador h={h}"] = I.L.shift(cfg.exec_lag)
        ret["0x"] = mk.rf
        for L in (1, 2, 3):
            ret[f"{L}x"] = mk.x[L]
            expo[f"{L}x"] = pd.Series(float(L), index=mk.u.index)
        ret["LRS"] = run_lrs(mk, cfg)
        expo["LRS"] = 3 * lrs_state(mk, cfg).shift(cfg.exec_lag)
        R["ret"][name] = pd.DataFrame(ret).reindex(mk.u.index)
        R["expo"][name] = pd.DataFrame(expo).reindex(mk.u.index)
        u = mk.mi.u.dropna().astype(float)
        G = r34.gjr_forecast(u, cfg.horizons, cfg.bench_min_obs, cfg.garch_refit)
        R["bench"][name] = {}
        for h in cfg.horizons:
            y = r34.realized_iv(u, h)
            hl = r10.har_var_forecast(y, u, h, min_obs=cfg.bench_min_obs)
            R["bench"][name][h] = pd.DataFrame({r34.M_HARRI: R["var"][name][h]["media"], r34.M_HARLOG: hl["nivel"],
                                                r34.M_GJR: G[h], "realizada": y})
    if cfg.sb_equity in M and cfg.sb_bond in M:
        me, mb = M[cfg.sb_equity], M[cfg.sb_bond]
        for h in cfg.horizons:
            R["corr"][h] = corr_forecast(me.u, mb.u.reindex(me.u.index), h, cfg)
            sb = stock_bond(R["ind"][(cfg.sb_equity, h)], R["ind"][(cfg.sb_bond, h)], R["corr"][h]["rho"], me, mb, h, cfg)
            R["sb"][h] = sb
            R["sb_ret"][f"acciones y bonos h={h}"] = run_stock_bond(sb, me, mb, cfg)
    return R


# ───────────────────────────── evaluación (desarrollo aquí; prueba reservada en r36) ─────────────────────────────

def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    return np.log1p(df.clip(lower=-0.9999))


def max_drawdown_cols(R: np.ndarray) -> np.ndarray:
    """Caída máxima (≤ 0) de cada columna de una matriz de rentabilidades diarias."""
    W = np.cumprod(1 + np.clip(R, -0.9999, None), axis=0)
    peak = np.maximum.accumulate(np.vstack([np.ones((1, R.shape[1])), W]), axis=0)[1:]
    return (W / peak - 1).min(axis=0)


def paired_bootstrap(R: pd.DataFrame, cfg: Config, rng, mdd: bool = False) -> dict:
    """Bootstrap estacionario emparejado (bloque cfg.block, cfg.B remuestreos, los mismos para todas
    las columnas): medias de log(1 + r) y, con mdd, caídas máximas de cada remuestreo."""
    X = R.to_numpy(float)
    L = np.log1p(np.clip(X, -0.9999, None))
    n, K = X.shape
    means = np.empty((cfg.B, K))
    dd = np.empty((cfg.B, K)) if mdd else None
    for b in range(cfg.B):
        ix = stats.stationary_indices(n, cfg.block, rng)
        means[b] = L[ix].mean(axis=0)
        if mdd:
            dd[b] = max_drawdown_cols(X[ix])
    return {"means": means, "mdd": dd, "cols": list(R.columns), "n": n}


def romano_wolf(t_obs, t_boot) -> np.ndarray:
    """p ajustados del step-down de Romano y Wolf (2005), H0_k: media_k ≤ 0 (una cola); es
    r19.romano_wolf (copiado para no arrastrar R19 a la configuración congelada)."""
    t_obs = np.asarray(t_obs, float)
    tb = np.asarray(t_boot, float)
    B, K = tb.shape
    order = np.argsort(-t_obs, kind="stable")
    tail = np.maximum.accumulate(tb[:, order][:, ::-1], axis=1)[:, ::-1]
    p = (1 + (tail >= t_obs[order]).sum(axis=0)) / (1 + B)
    p = np.minimum(np.maximum.accumulate(p), 1.0)
    out = np.empty(K)
    out[order] = p
    return out


def delta_family(R: pd.DataFrame, pairs: list[tuple[str, str]], cfg: Config, rng) -> pd.DataFrame:
    """Δg = crecimiento logarítmico anual de a menos el de b para cada par (a, b), con los mismos
    remuestreos: IC percentil del 90%, p unilateral (H0: Δg ≤ 0) con el error típico del bootstrap y
    p de Romano–Wolf en la familia."""
    bt = paired_bootstrap(R, cfg, rng)
    L = log_returns(R)
    cols = list(R.columns)
    m = L.mean().to_numpy(float)
    rows, tt, tb = [], [], []
    for a, b in pairs:
        i, j = cols.index(a), cols.index(b)
        d = bt["means"][:, i] - bt["means"][:, j]
        obs = m[i] - m[j]
        se = d.std(ddof=1)
        t = obs / se if se > 0 else np.nan
        lo, hi = np.percentile(d, [5, 95])
        rows.append({"estrategia": a, "frente a": b, "Δg": obs * DAYS, "IC90 inferior": lo * DAYS,
                     "IC90 superior": hi * DAYS, "t": t, "p (una cola)": float(norm.sf(t)) if np.isfinite(t) else np.nan})
        tt.append(t)
        tb.append((d - obs) / se if se > 0 else np.zeros_like(d))
    out = pd.DataFrame(rows)
    out["p Romano–Wolf"] = romano_wolf(tt, np.column_stack(tb))
    out.attrs.update(desde=R.index[0], hasta=R.index[-1], sesiones=len(R))
    return out


def spa(R: pd.DataFrame, variants: list[str], bench: str, cfg: Config, rng) -> dict:
    """SPA de Hansen (consistente, stats.spa_test) de las variantes frente a bench, en log(1 + r)."""
    L = log_returns(R)
    D = L[variants].sub(L[bench], axis=0).to_numpy(float)
    res = stats.spa_test(D, cfg.block, cfg.B, rng)
    dm = D.mean(axis=0) * DAYS
    j = int(dm.argmax())
    return {"frente a": bench, "variantes": len(variants), "mejor": variants[j], "Δg mejor": float(dm[j]),
            "p SPA": res["p_SPA_c"], "desde": R.index[0], "hasta": R.index[-1]}


def growth_stats(R: pd.DataFrame, E: pd.DataFrame | None, rf: pd.Series) -> pd.DataFrame:
    rows = {}
    for c in R.columns:
        r = R[c].dropna()
        rows[c] = {"crecimiento log": float(log_returns(r.to_frame()).mean().iloc[0] * DAYS),
                   "CAGR": metrics.cagr(r), "vol": metrics.vol(r), "maxDD": metrics.max_dd(r),
                   "exposición media": float(E[c].reindex(r.index).mean()) if E is not None and c in E else np.nan}
    return pd.DataFrame(rows).T


def equal_exposure(ra: pd.Series, ea: pd.Series, rb: pd.Series, eb: pd.Series, rf: pd.Series) -> tuple:
    """Iguala la exposición media: la de más exposición se mezcla a diario con letras,
    a·r + (1 − a)·rf, con a = exposición menor / mayor. Devuelve (ra', rb', a, cuál se escaló)."""
    ma, mb = float(ea.mean()), float(eb.mean())
    rf = rf.reindex(ra.index)
    if ma > mb and ma > 0:
        a = mb / ma
        return a * ra + (1 - a) * rf, rb, a, "a"
    if mb > ma and mb > 0:
        a = ma / mb
        return ra, a * rb + (1 - a) * rf, a, "b"
    return ra, rb, 1.0, "—"


def shapley_variance(N_parts: dict, D: pd.Series, fixed: str = "mean") -> dict:
    """Descomposición exacta de var(L*) con L* = (Σ partes del numerador)/D en la contribución de cada
    fuente (Shapley): para cada subconjunto S de fuentes, L_S es L* con las fuentes fuera de S fijas en
    su media; v(S) = var(L_S) y la parte de la fuente k es su contribución marginal media a v sobre
    todos los órdenes. Las partes suman var(L*) (sin resto), aunque L* no sea lineal en D."""
    from itertools import permutations

    names = list(N_parts) + ["varianza"]
    means = {k: float(v.mean()) for k, v in N_parts.items()}
    Dm = float(D.mean())

    def v(S: frozenset) -> float:
        num = sum((N_parts[k] if k in S else means[k]) for k in N_parts)
        den = D if "varianza" in S else Dm
        x = num / den
        return float(np.var(np.asarray(x, float) if not np.isscalar(x) else [x]))

    cache: dict = {}
    val = lambda S: cache.setdefault(S, v(S))  # noqa: E731
    phi = {k: 0.0 for k in names}
    perms = list(permutations(names))
    for order in perms:
        S = frozenset()
        for k in order:
            phi[k] += val(S | {k}) - val(S)
            S = S | {k}
    tot = val(frozenset(names))
    return {k: phi[k] / len(perms) / tot if tot > 0 else np.nan for k in names}


def lstar_decomposition(I: pd.DataFrame, cfg: Config, rng) -> dict:
    """Descomposición de la varianza del L* continuo (H5). L* = (P + T)/D con P = prior contraído −
    κ·ee − s (el numerador sin la tendencia), T = β·(S − S̄) (la tendencia) y D = E[IV] + κ·de (la
    varianza robusta): partes de Shapley de var(L*) (shapley_variance). Correlaciones de L* con E[IV]
    y con el estado S, y p unilateral de corr(L*, E[IV]) < 0 con el bootstrap estacionario."""
    J = I[["L_star", "num", "D", "trend", "V", "S"]].dropna()
    Ls, N, D, T = J.L_star, J.num, J.D, J.trend
    share = shapley_variance({"prior": N - T, "tendencia": T}, D)
    cV = float(Ls.corr(J.V))
    cS = float(Ls.corr(J.S)) if J.S.std() > 0 else np.nan
    X = J[["L_star", "V"]].to_numpy(float)
    n = len(X)
    draws = np.empty(cfg.B)
    for b in range(cfg.B):
        ix = stats.stationary_indices(n, cfg.block, rng)
        draws[b] = np.corrcoef(X[ix, 0], X[ix, 1])[0, 1]
    se = np.nanstd(draws, ddof=1)
    p = float(norm.cdf(cV / se)) if se > 0 else np.nan
    return {"días": n, "desde": J.index[0], "hasta": J.index[-1], "corr(L*, E[IV])": cV,
            "p corr < 0": p, "corr(L*, S)": cS, "parte de la varianza": share["varianza"],
            "parte del prior": share["prior"], "parte de la tendencia": share["tendencia"],
            "L* medio": float(Ls.mean()), "L* mediano": float(Ls.median())}


def dd_compare(R: pd.DataFrame, a: str, b: str, cfg: Config, rng) -> dict:
    """Caída máxima de a y b en la ventana, probabilidad (bootstrap estacionario emparejado de la
    ventana) de una caída máxima de al menos cfg.loss_p y p = fracción de remuestreos en que la caída
    de a es al menos tan profunda como la de b."""
    bt = paired_bootstrap(R[[a, b]], cfg, rng, mdd=True)
    X = R[[a, b]].to_numpy(float)
    mdd = max_drawdown_cols(X)
    return {"maxDD a": float(mdd[0]), "maxDD b": float(mdd[1]),
            "P(pérdida ≥ 50%) a": float((bt["mdd"][:, 0] <= -cfg.loss_p).mean()),
            "P(pérdida ≥ 50%) b": float((bt["mdd"][:, 1] <= -cfg.loss_p).mean()),
            "p (caída de a ≥ caída de b)": float((bt["mdd"][:, 0] <= bt["mdd"][:, 1]).mean())}


def forecast_window(R: dict, name: str, h: int, start, end) -> pd.DataFrame:
    """Previsiones de HAR-RI, HAR-log + Duan y GJR-GARCH hechas en [start, end] cuya ventana
    objetivo acaba dentro de los datos (realizada conocida)."""
    F = R["bench"][name][h]
    F = F[(F.index >= pd.Timestamp(start)) & (F.index <= pd.Timestamp(end))]
    return F.dropna()


def hypotheses(R: dict, start, end, cfg: Config, premium: dict, only=None) -> dict:
    """Los estadísticos de H1–H8 en la ventana [start, end] con las reglas de decisión congeladas.
    r36 la llama con la prueba reservada; aquí se aplica al desarrollo solo como descripción.
    premium: prima μ − f constante del desarrollo por mercado (escala la pérdida de crecimiento).
    only: subconjunto de {"H1", …, "H8"} (por defecto, todas)."""
    want = (lambda k: only is None or k in only)  # noqa: E731
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    hm = cfg.h_main
    a = cfg.alpha
    win = lambda df: df[(df.index >= start) & (df.index <= end)]  # noqa: E731
    out = {"ventana": (start, end)}

    # H1 y H6: HAR-RI frente a HAR-log + Duan y GJR-GARCH
    fc = {}
    for name in ("S&P 500", "Nasdaq-100", "Bono largo"):
        if name not in R["bench"] or not want("H6" if name == "Bono largo" else "H1"):
            continue
        for h in cfg.horizons:
            F = forecast_window(R, name, h, start, end)
            if len(F) <= 2 * cfg.block:
                continue
            c = r34.compare(F, h, premium[name], rng_for(cfg, f"H1 {name} {h}"), cfg.B_mcs)
            dm = c["dm"]
            dec = {}
            for loss in ("QLIKE", "crecimiento"):
                sub = dm[dm["pérdida"] == loss]
                dec[loss] = bool((sub["p (una cola)"] < a).all())
            fc[(name, h)] = {**c, "decisión": dec}
    out["H1"] = {k: v for k, v in fc.items() if k[0] in EQUITY_MARKETS}
    out["H6"] = {k: v for k, v in fc.items() if k[0] == "Bono largo"}

    # H2: el indicador (h principal) frente al 1x y al 2x fijos, familia de los dos mercados
    cols, pairs = {}, []
    for name in EQUITY_MARKETS:
        if name not in R["ret"]:
            continue
        Rm = R["ret"][name]
        for c in (f"indicador h={hm}", "1x", "2x"):
            cols[f"{name}: {c}"] = Rm[c]
        pairs += [(f"{name}: indicador h={hm}", f"{name}: 1x"), (f"{name}: indicador h={hm}", f"{name}: 2x")]
    if cols and want("H2"):
        F2 = win(pd.DataFrame(cols)).dropna()
        t2 = delta_family(F2, pairs, cfg, rng_for(cfg, "H2"))
        dec = {}
        for name in EQUITY_MARKETS:
            sub = t2[t2["estrategia"].str.startswith(name)]
            if len(sub):
                dec[name] = bool((sub["p Romano–Wolf"] < a).all() and (sub["Δg"] > 0).all())
        out["H2"] = {"tabla": t2, "decisión": dec}

    # H3 y H4: frente a la LRS
    h3, h4 = {}, {}
    for name in EQUITY_MARKETS:
        if name not in R["ret"] or not (want("H3") or want("H4")):
            continue
        Rm = win(R["ret"][name])
        var = [f"indicador h={h}" for h in cfg.horizons]
        F3 = Rm[var + ["LRS"]].dropna()
        s = spa(F3, var, "LRS", cfg, rng_for(cfg, f"H3 {name}"))
        h3[name] = {**s, "decisión (no supera)": bool(s["p SPA"] >= a)}
        E = win(R["expo"][name])
        j = Rm[[f"indicador h={hm}", "LRS"]].join(E[[f"indicador h={hm}", "LRS"]], rsuffix=" (exp)").dropna()
        ra, rb, k, which = equal_exposure(j[f"indicador h={hm}"], j[f"indicador h={hm} (exp)"],
                                          j["LRS"], j["LRS (exp)"], R["markets"][name].rf)
        F4 = pd.DataFrame({"indicador": ra, "LRS": rb})
        d = dd_compare(F4, "indicador", "LRS", cfg, rng_for(cfg, f"H4 {name}"))
        dec = d["maxDD a"] > d["maxDD b"] and d["P(pérdida ≥ 50%) a"] < d["P(pérdida ≥ 50%) b"]
        h4[name] = {**d, "exposición media indicador": float(j[f"indicador h={hm} (exp)"].mean()),
                    "exposición media LRS": float(j["LRS (exp)"].mean()), "factor": k,
                    "escalada": {"a": "indicador", "b": "LRS", "—": "ninguna"}[which], "decisión": bool(dec)}
    out["H3"], out["H4"] = h3, h4

    # H5: descomposición de L*
    h5 = {}
    for name in MARKETS:
        if (name, hm) not in R["ind"] or not want("H5"):
            continue
        dcp = lstar_decomposition(win(R["ind"][(name, hm)]), cfg, rng_for(cfg, f"H5 {name}"))
        dcp["decisión"] = bool(dcp["corr(L*, E[IV])"] < 0 and dcp["p corr < 0"] < a
                               and dcp["parte de la varianza"] >= cfg.h5_share
                               and abs(np.nan_to_num(dcp["corr(L*, S)"])) < abs(dcp["corr(L*, E[IV])"]))
        h5[name] = dcp
    out["H5"] = h5

    # H7: bono largo
    if "Bono largo" in R["ret"] and want("H7"):
        Rm = win(R["ret"]["Bono largo"])
        F7 = Rm[[f"indicador h={hm}", "1x", "2x", "3x"]].dropna()
        t7 = delta_family(F7[[f"indicador h={hm}", "1x", "2x"]],
                          [(f"indicador h={hm}", "1x"), (f"indicador h={hm}", "2x")], cfg, rng_for(cfg, "H7"))
        d7 = dd_compare(F7, f"indicador h={hm}", "3x", cfg, rng_for(cfg, "H7 dd"))
        g_ok = bool((t7["p Romano–Wolf"] < a).all() and (t7["Δg"] > 0).all())
        dd_ok = bool(d7["maxDD a"] > d7["maxDD b"])
        out["H7"] = {"tabla": t7, "caídas": d7, "decisión crecimiento": g_ok, "decisión caída": dd_ok,
                     "decisión": g_ok and dd_ok}

    # H8: cartera de acciones y bonos
    if R["sb_ret"] and want("H8"):
        me, mb = R["markets"][cfg.sb_equity], R["markets"][cfg.sb_bond]
        SB = pd.DataFrame(R["sb_ret"])
        SB["LRS"] = R["ret"][cfg.sb_equity]["LRS"]
        SB = win(SB).dropna()
        s0 = SB.index[0]
        SB["HFEA 55/45"] = run_hfea(me, mb, s0, end, cfg).reindex(SB.index)
        SB = SB.dropna()
        var = [f"acciones y bonos h={h}" for h in cfg.horizons]
        sp = {b: spa(SB[var + [b]], var, b, cfg, rng_for(cfg, f"H8 {b}")) for b in ("LRS", "HFEA 55/45")}
        d8 = dd_compare(SB, f"acciones y bonos h={hm}", "HFEA 55/45", cfg, rng_for(cfg, "H8 dd"))
        g_ok = bool(all(v["p SPA"] >= a for v in sp.values()))
        dd_ok = bool(d8["maxDD a"] > d8["maxDD b"])
        out["H8"] = {"spa": sp, "caídas": d8, "retornos": SB, "decisión no supera": g_ok,
                     "decisión caída": dd_ok, "decisión": g_ok and dd_ok}
    return out


# ───────────────────────────── configuración congelada ─────────────────────────────

HYPOTHESES = {
    "H1": {"texto": "HAR-RI prevé IV(21) mejor que HAR-log con Duan y que GJR-GARCH (S&P 500 y Nasdaq-100)",
           "contraste": "r34.compare en las previsiones hechas en la prueba con la ventana objetivo completa: MCS al 90% "
                        "(QLIKE y pérdida de crecimiento, bloque 63) y Diebold–Mariano con Hansen–Hodrick (h − 1 retardos)",
           "regla": "por mercado y pérdida: mejor si el p unilateral de Diebold–Mariano < 0,10 frente a los DOS benchmarks; "
                    "H1 se confirma en un mercado si pasa con QLIKE y con la pérdida de crecimiento (parcial si solo con "
                    "una); h = 63 es sensibilidad. La pérdida de crecimiento usa la prima constante del desarrollo."},
    "H2": {"texto": "La cartera del indicador (h = 21, con histéresis) crece más que el 1x y el 2x fijos",
           "contraste": "Δg de log(1 + r) con bootstrap estacionario emparejado (bloque 63, B = 2000), familia de 4 "
                        "(S&P 500 y Nasdaq-100 × 1x y 2x) con Romano–Wolf",
           "regla": "se confirma en un mercado si Δg > 0 y el p de Romano–Wolf < 0,10 frente al 1x y frente al 2x"},
    "H3": {"texto": "El indicador no supera a la LRS de Gayed y Bilello",
           "contraste": "SPA de Hansen (consistente) de las variantes h ∈ {21, 63} con κ = 1 frente a la LRS 3x, por mercado",
           "regla": "mejora solo si p SPA < 0,10; H3 (no supera) se confirma si p SPA ≥ 0,10"},
    "H4": {"texto": "A igual exposición media, el indicador tiene menor caída máxima y menor probabilidad de perder el 50% "
                    "que la LRS",
           "contraste": "la estrategia con más exposición media en la ventana se mezcla a diario con letras hasta igualar la "
                        "de la otra; caída máxima de la ventana y probabilidad de una caída máxima ≥ 50% en remuestreos "
                        "estacionarios emparejados de la ventana (bloque 63)",
           "regla": "se confirma si la caída máxima del indicador es menor (en valor absoluto) Y su probabilidad de perder "
                    "el 50% es estrictamente menor; se informa también el p del bootstrap de la caída"},
    "H5": {"texto": "L* está correlacionado negativamente con la varianza prevista y apenas con la tendencia",
           "contraste": "descomposición exacta (Shapley) de la varianza del L* continuo (h = 21) en varianza prevista, "
                        "prior y tendencia; correlaciones con E[IV] y con el estado S",
           "regla": "se confirma si corr(L*, E[IV]) < 0 con p unilateral < 0,10, la parte de la varianza ≥ 0,75 y "
                    "|corr(L*, S)| < |corr(L*, E[IV])| (S&P 500 y Nasdaq-100; los bonos se informan)"},
    "H6": {"texto": "HAR-RI prevé la IV(21) del bono largo mejor que HAR-log con Duan y que GJR-GARCH",
           "contraste": "el de H1", "regla": "la de H1"},
    "H7": {"texto": "El indicador del bono largo crece más que TLT (1x) y UBT (2x) y tiene menor caída máxima que TMF (3x)",
           "contraste": "el de H2 con la familia {1x, 2x} del bono largo; caída máxima de la ventana",
           "regla": "se confirma si Δg > 0 con p de Romano–Wolf < 0,10 frente a TLT y a UBT y la caída máxima es menor que "
                    "la de TMF"},
    "H8": {"texto": "La cartera de acciones y bonos no supera a la LRS ni a la HFEA 55/45 en crecimiento, pero tiene menor "
                    "caída máxima que la HFEA",
           "contraste": "SPA de las variantes h ∈ {21, 63} frente a la LRS 3x del S&P 500 y frente a la HFEA 55/45 "
                        "(comprada al cierre anterior al inicio de la ventana, reajuste trimestral); caída máxima (h = 21)",
           "regla": "«no supera» se confirma si p SPA ≥ 0,10 frente a las dos; «menor caída» si la caída máxima (h = 21) es "
                    "menor que la de la HFEA"},
}

CALLS = [
    "letf_lab.research.r36_indicador_holdout.run",
    "letf_lab.research.r35_indicador.Config.from_dict",
    "letf_lab.research.r35_indicador.load_all(end=END)",
    "letf_lab.research.r35_indicador.compute(end=END, cfg)",
    "letf_lab.research.r35_indicador.fingerprints(R, DEV_END)",
    "letf_lab.research.r35_indicador.hypotheses(R, TEST_START, END, cfg, premium)",
    "letf_lab.research.r34_harri.forecast / HARRI.predict_from / gjr_forecast / compare",
    "letf_lab.research.r10_predictability.har_var_forecast",
    "letf_lab.data.indicador.make_equity / bond_underlying / cmt_bond / real_fund / fund_spread / ohlc / move",
    "letf_lab.research.r24_folk.run_multi / hfea / lev_fund",
    "letf_lab.research.r13_lrs.above_sma / sp500_signal_returns",
    "letf_lab.stats.garch_fit / garch_filter / gpd_tail / spa_test / stationary_indices / ols_hac",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files() -> list[str]:
    """Archivos de letf_lab cargados en este proceso (rutas relativas a letf/), más r36."""
    import letf_lab.research.r36_indicador_holdout  # noqa: F401 — el código de la evaluación entra en la huella
    files = set()
    for name, mod in list(sys.modules.items()):
        if name.startswith("letf_lab") and getattr(mod, "__file__", None):
            p = Path(mod.__file__).resolve()
            try:
                files.add(str(p.relative_to(LETF)))
            except ValueError:
                continue
    return sorted(files)


def fingerprints(R: dict, until) -> dict:
    """Huella de las salidas del indicador con fechas ≤ until: r36 la recalcula con los datos hasta
    2026 y exige que coincida (sin mirar el futuro y con los mismos datos del desarrollo)."""
    until = pd.Timestamp(until)
    out = {}
    for (name, h), I in R["ind"].items():
        J = I[I.index <= until]
        L = J.L.dropna().astype(int)
        s = J.L_star.dropna()
        out[f"{name} h={h}"] = {"días": int(len(L)), "desde": str(L.index[0].date()) if len(L) else None,
                                "sha256_L": hashlib.sha256((",".join(map(str, L.values)) + "|" +
                                                            ",".join(str(d.date()) for d in L.index)).encode()).hexdigest(),
                                "suma_L_star": float(s.sum()), "L_final": int(L.iloc[-1]) if len(L) else None}
    for h, sb in R["sb"].items():
        J = sb[sb.index <= until].dropna(subset=["L_acc"])
        k = (J.L_acc * 10 + J.L_bono).astype(int)
        out[f"acciones y bonos h={h}"] = {"días": int(len(k)), "sha256_L": hashlib.sha256(",".join(map(str, k.values)).encode()).hexdigest()}
    return out


def dev_estimates(R: dict) -> dict:
    """Parámetros estimados con el desarrollo que r36 recalcula y compara."""
    M = R["markets"]
    out = {"s": {}, "gasto": {}, "vencimiento": {}, "prima_desarrollo": {}}
    for name, mk in M.items():
        for L in (2, 3):
            out["s"][mk.tickers[L]] = float(mk.spread[L])
            out["gasto"][mk.tickers[L]] = float(mk.er[L])
        if mk.maturity is not None:
            out["vencimiento"][name] = float(mk.maturity)
        out["prima_desarrollo"][name] = float(r34.premium_dev(mk.mi.truncate(DEV_END)))
    return out


def freeze(R: dict, cfg: Config, path: Path = FROZEN) -> dict:
    """Escribe la configuración congelada: parámetros, especificación, horizontes, κ, umbrales,
    semillas, reglas de decisión, funciones que llamará la evaluación, estimaciones del desarrollo,
    huellas de las salidas hasta 2018 y SHA-256 de cada archivo de código del que dependen."""
    files = source_files()
    doc = {
        "estudio": "Artículo 7 — indicador del apalancamiento óptimo (SSO, UPRO, QLD, TQQQ, UBT, TMF, UST, TYD)",
        "prerregistro": {"archivo": "letf/preregistro/articulo7_indicador.md", "commits": ["aafe958", "02faefc"],
                         "sha256": sha256(PREREG)},
        "congelado": {"fecha": pd.Timestamp.now().strftime("%Y-%m-%d"), "commit_base": provenance.git_head(),
                      "nota": "código sin commit en el momento de congelar: la huella SHA-256 de cada archivo es la "
                              "referencia, no el commit"},
        "muestras": {"desarrollo_fin": str(DEV_END.date()), "prueba_inicio": str(TEST_START.date()),
                     "prueba_fin": str(END.date()),
                     "inicio_por_mercado": {n: str(m.u.index[0].date()) for n, m in R["markets"].items()}},
        "parametros": cfg.to_dict(),
        "mercados": {n: {"tipo": m.kind, "fondos": {str(k): v for k, v in m.tickers.items()},
                         "lanzamiento_real": {str(k): str(v.date()) for k, v in m.launch.items()},
                         "prior_deriva": "carry del bono sintético (prior_ann) al cierre de t − 1" if m.prior is not None
                         else "kelly.premium: media creciente de u − f hasta t − 1"} for n, m in R["markets"].items()},
        "rupturas_de_serie": {k: list(v) for k, v in BOND_BREAKS.items()},
        "modelo_varianza": {"funcion": "letf_lab.research.r34_harri.forecast", "grupos": list(r34.ALL_GROUPS),
                            "min_obs": cfg.harri_min_obs, "min_obs_extra": cfg.harri_min_obs_extra,
                            "refit": cfg.harri_refit, "disp_min": cfg.harri_disp_min, "duan_min": r34.DUAN_MIN,
                            "cuantiles": list(r34.QUANTS), "vvix_desde": str(r34.VVIX_FROM.date())},
        "especificacion": [line.strip() for line in __doc__.split("Componentes", 1)[1].split("Protocolo")[0].splitlines()
                           if line.strip()],
        "hipotesis": HYPOTHESES,
        "codigo": {"entrada": "letf_lab.research.r36_indicador_holdout.run()", "llamadas": CALLS,
                   "marca_de_ejecucion": "letf/preregistro/articulo7_holdout_ejecutado.json",
                   "informe": "letf/reports/r36_indicador_holdout/README.md"},
        "estimados_en_desarrollo": dev_estimates(R),
        "huellas_desarrollo": fingerprints(R, DEV_END),
        "sha256": {f: sha256(LETF / f) for f in files},
        "decisiones": DECISIONS,
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, default=str) + "\n")
    return doc


DECISIONS = [
    "Rango del S&P 500 desde el 21-4-1982 (antes Yahoo trae el máximo y el mínimo «teóricos»), no desde 1960 (R33, R34).",
    "Muchas aperturas de ^GSPC (hasta 2006) y de ^NDX (hasta 2000) son el cierre anterior: sin término nocturno (R33).",
    "MOVE solo desde el 12-11-2002 (Yahoo ^MOVE); antes el bono va sin implícita (R33).",
    "s de TMF, TYD y TMV calibrado desde el 1-1-2011 (hasta 2010 seguían a otro índice) (R33).",
    "Ruptura de DGS20 el 4-1-1973 (+74 pb con DGS10 en −2 pb): ese día el bono largo sintético solo lleva carry y rodadura.",
    "Valor prudente de la deriva: μ − f = 0 (ninguna prima sobre la financiación).",
    "Peso de Kan y Zhou (2007) de la regla de dos fondos con N = 1 y el estimador ajustado de θ².",
    "Bonos: el error típico del prior (carry) es el de la media de u − f − carry hasta t − 1.",
    "ee(μ)² = w·ee(prior)² + (S − S̄)²·ee(β)² (varianza a posteriori del prior contraído más la del efecto de la tendencia).",
    "Efecto de la tendencia: pendiente de la media de las h sesiones siguientes sobre S − S̄, Hansen–Hodrick, cada 21 sesiones.",
    "L* continuo con el s del 3x y sin comisión; la recomendación discreta usa los costes de cada fondo; el 1x no paga "
    "financiación.",
    "Ruina: probabilidad diaria condicional anualizada, 1 − (1 − p)^252; GJR-GARCH y GPD (cuantil 95%) reestimados cada 252 "
    "sesiones con 10 años de mínimo; L = 0 y L = 1 siempre permitidos.",
    "Histéresis: argmax de g_κ(L) − coste·252/h; coste de 5 pb por unidad negociada (dos al cambiar de fondo); si el L que "
    "se tiene deja de estar permitido, se cambia sin histéresis; empates hacia el L menor.",
    "H3: el prerregistro fija κ = 1, así que las variantes del SPA son h ∈ {21, 63} con κ = 1.",
    "H8: S&P 500 y bono largo con los 3x (UPRO y TMF) a pesos L/3, L_acc + L_bono ≤ 3 como la HFEA; cada L por debajo del "
    "mayor L permitido en su activo; reajuste trimestral y en cada cambio; varianzas robustas con la correlación prevista "
    "(HAR sobre la z de Fisher de las correlaciones de 21, 63 y 252 sesiones).",
    "H4: la estrategia con más exposición media en la ventana se mezcla a diario con letras (sin coste) hasta igualar; "
    "la probabilidad de perder el 50% es la de una caída máxima ≥ 50% en remuestreos estacionarios de la ventana.",
    "H5: descomposición exacta (Shapley) de var(L*) en varianza prevista robusta, prior contraído (menos κ·ee y s) y "
    "tendencia; «casi toda» = parte de la varianza ≥ 0,75.",
    "H2 y H7 deciden con el p de Romano–Wolf de su familia (el protocolo nombra Romano–Wolf); se informa también el p sin "
    "ajustar.",
]


# ───────────────────────────── figuras ─────────────────────────────

INK, MUTED, GRID = r34.INK, r34.MUTED, r34.GRID
BLUE, ORANGE, AQUA, PURPLE, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#9a9994"


def _style(ax):
    r34._style(ax)


def fig_levels(R: dict, rep: Report, h: int):
    """L* continuo (recortado a [−1, 5]) y el L que se tiene, por mercado, en el desarrollo."""
    names = [n for n in MARKETS if (n, h) in R["ind"]]
    fig, axes = plt.subplots(len(names), 1, figsize=(FIG_W, 2.5 * len(names) + 1.0), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        I = R["ind"][(name, h)].dropna(subset=["L"])
        ax.plot(I.index, I.L_star.clip(-1, 5), color=GREY, lw=0.7)
        ax.step(I.index, I.L, where="post", color=BLUE, lw=1.4)
        ax.set_ylim(-1.1, 5.1)
        ax.set_yticks([-1, 0, 1, 2, 3, 4, 5])
        ax.set_title(f"{name} ({', '.join(R['markets'][name].tickers[k] for k in (1, 2, 3))})")
        _style(ax)
    axes[-1, 0].set_xlabel("")
    legend_band(fig, [plt.Line2D([], [], color=BLUE, lw=2), plt.Line2D([], [], color=GREY, lw=1.5)],
                ["L que se tiene (0 letras, 1, 2 o 3 veces)", "L* continuo (recortado a −1 … 5)"],
                where="bottom", ncol=2, fontsize=11)
    fig.suptitle(f"Apalancamiento del indicador, h = {h} (desarrollo)")
    comma_fig(fig)
    rep.fig(fig, f"apalancamiento_h{h}",
            f"Figura: apalancamiento del indicador con h = {h} en la muestra de desarrollo (hasta el 31-12-2018). En azul, "
            "el fondo que se tiene cada día (con la restricción de ruina y la histéresis); en gris, el L* continuo del "
            "Kelly robusto")


def fig_wealth(R: dict, rep: Report, D: dict):
    """Crecimiento de 1 € en escala logarítmica, por mercado, en su ventana de desarrollo."""
    names = list(D)
    fig, axes = plt.subplots(len(names), 1, figsize=(FIG_W, 3.2 * len(names) + 1.0), squeeze=False)
    keys = [(f"indicador h={R['cfg'].h_main}", BLUE, 1.8), ("1x", INK, 1.2), ("2x", ORANGE, 1.2), ("LRS", AQUA, 1.2)]
    for ax, name in zip(axes[:, 0], names):
        F = D[name]
        for k, c, lw in keys:
            if k in F:
                ax.plot(F.index, (1 + F[k]).cumprod(), color=c, lw=lw)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_ylabel("valor de 1 €")
        _style(ax)
    legend_band(fig, [plt.Line2D([], [], color=c, lw=2) for _, c, _ in keys],
                [f"indicador (h = {R['cfg'].h_main})", "1x fijo", "2x fijo", "LRS 3x"], where="bottom", ncol=4, fontsize=11)
    fig.suptitle("Valor de 1 € en la muestra de desarrollo (escala logarítmica)")
    rep.fig(fig, "valor_desarrollo",
            "Figura: valor de 1 € invertido al principio de la ventana común de cada mercado en la muestra de desarrollo "
            "(hasta el 31-12-2018), con orden al cierre siguiente y 5 pb por unidad negociada. Escala logarítmica")


def fig_drift(R: dict, rep: Report, h: int):
    """Prior, prior contraído y deriva con la tendencia, con la banda ± ee, en dos mercados."""
    names = [n for n in ("S&P 500", "Bono largo") if (n, h) in R["ind"]]
    fig, axes = plt.subplots(len(names), 1, figsize=(FIG_W, 3.3 * len(names) + 1.0), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        I = R["ind"][(name, h)].dropna(subset=["mu_f"])
        ax.fill_between(I.index, 100 * (I.mu_f - I.ee), 100 * (I.mu_f + I.ee), color=BLUE, alpha=0.15, lw=0)
        ax.plot(I.index, 100 * I.prior, color=GREY, lw=1.0)
        ax.plot(I.index, 100 * I.shrunk, color=ORANGE, lw=1.3)
        ax.plot(I.index, 100 * I.mu_f, color=BLUE, lw=1.0)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(name)
        ax.set_ylabel("% anual")
        _style(ax)
    legend_band(fig, [plt.Line2D([], [], color=GREY, lw=2), plt.Line2D([], [], color=ORANGE, lw=2),
                      plt.Line2D([], [], color=BLUE, lw=2), plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.15)],
                ["prior de μ − f", "prior contraído (Kan y Zhou)", "con la tendencia", "± ee(μ)"],
                where="bottom", ncol=2, fontsize=11)
    fig.suptitle(f"Deriva μ − f del indicador, h = {h} (desarrollo)")
    comma_fig(fig)
    rep.fig(fig, f"deriva_h{h}",
            "Figura: deriva anual sobre la financiación. Gris: el prior (acciones: media histórica hasta t − 1; bonos: carry "
            "del bono sintético); naranja: contraído hacia 0 con el peso de Kan y Zhou; azul: más el efecto de la tendencia "
            "estimado con datos pasados, con la banda de ± un error típico. Muestra de desarrollo")


def fig_stock_bond(R: dict, rep: Report, SB: pd.DataFrame):
    h = R["cfg"].h_main
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 7.4), gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
    ax = axes[0]
    keys = [(f"acciones y bonos h={h}", BLUE, 1.8), ("HFEA 55/45", ORANGE, 1.2), ("LRS", AQUA, 1.2)]
    for k, c, lw in keys:
        ax.plot(SB.index, (1 + SB[k]).cumprod(), color=c, lw=lw)
    ax.set_yscale("log")
    ax.set_ylabel("valor de 1 €")
    ax.set_title("Valor de 1 € (escala logarítmica)")
    _style(ax)
    sb = R["sb"][h].reindex(SB.index)
    axes[1].step(sb.index, sb.L_acc, where="post", color=BLUE, lw=1.2)
    axes[1].step(sb.index, sb.L_bono, where="post", color=PURPLE, lw=1.2)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_title("Apalancamiento de cada activo (peso L/3 en su 3x)")
    _style(axes[1])
    legend_band(fig, [plt.Line2D([], [], color=c, lw=2) for _, c, _ in keys] +
                [plt.Line2D([], [], color=PURPLE, lw=2)],
                [f"acciones y bonos (h = {h}); abajo, L del S&P 500", "HFEA 55/45", "LRS 3x (S&P 500)", "L del bono largo"],
                where="bottom", ncol=2, fontsize=11)
    fig.suptitle("Cartera de acciones y bonos del indicador (desarrollo)")
    rep.fig(fig, "acciones_y_bonos",
            "Figura: la cartera de acciones y bonos del indicador (S&P 500 y bono largo con UPRO y TMF a pesos L/3, "
            "correlación prevista, reajuste trimestral y en cada cambio) frente a la HFEA 55/45 y la LRS 3x, en la "
            "muestra de desarrollo")


# ───────────────────────────── informe ─────────────────────────────

def _d(x) -> str:
    return r34.dmy(x)


def _p(p: float) -> str:
    return "< 0,001" if p < 0.001 else num_es(p, 3)


def dev_window(R: dict, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    F = R["ret"][name].dropna()
    return F.index[0], F.index[-1]


def run(end=DEV_END, cfg: Config | None = None, write_frozen: bool = True) -> dict:
    """Indicador en la muestra de desarrollo (hasta el 31-12-2018), informe reports/r35_indicador y
    configuración congelada. end > DEV_END se rechaza."""
    from letf_lab.research.r26_vix_data import Tables

    end = pd.Timestamp(end)
    if end > DEV_END:
        raise ValueError("R35 es la etapa de desarrollo: end no puede pasar del 31-12-2018")
    cfg = cfg or Config()
    R = compute(end, cfg)
    for name, mk in R["markets"].items():
        assert mk.last_date() <= DEV_END, name
    for k, I in R["ind"].items():
        r34.assert_until(I, DEV_END, f"indicador {k}")
    est = dev_estimates(R)
    H = {}
    windows = {name: dev_window(R, name) for name in MARKETS}
    # la ventana del desarrollo común a todas las estrategias de cada contraste
    sb_start = pd.DataFrame(R["sb_ret"]).dropna().index[0]
    eq_start = max(windows[n][0] for n in EQUITY_MARKETS)
    bond_start = max(windows["Bono largo"][0], sb_start)
    H["acciones"] = hypotheses(R, eq_start, end, cfg, est["prima_desarrollo"], only={"H1", "H2", "H3", "H4", "H5"})
    H["bonos"] = hypotheses(R, bond_start, end, cfg, est["prima_desarrollo"], only={"H6", "H7", "H8"})

    rep = Report("r35_indicador", "R35 — El indicador del apalancamiento óptimo (artículo 7, desarrollo)")
    tab = Tables(rep)
    rep.p(__doc__.split("\n", 2)[2])
    rep.p(f"Generado con el commit {provenance.git_head()[:10]} (con cambios sin commit). **Todo lo que sigue es la "
          f"muestra de desarrollo**: datos hasta el {_d(end)}; la prueba reservada ({_d(TEST_START)} → {_d(END)}) no se "
          "ha cargado ni mirado. Ninguna cifra de este informe es un contraste de H1–H8: las reglas de decisión se "
          "aplican aquí solo para describir el desarrollo y comprobar el código que usará r36.")

    # ── resumen
    rep.h("Resumen (desarrollo)")
    lines = []
    for name in MARKETS:
        I = R["ind"][(name, cfg.h_main)].dropna(subset=["L"])
        share = I.L.value_counts(normalize=True).reindex([0, 1, 2, 3]).fillna(0)
        sw = int((I.L.diff().fillna(0) != 0).sum()) / (len(I) / DAYS)
        a, b = windows[name]
        g = growth_stats(R["ret"][name].loc[a:b], R["expo"][name].loc[a:b], R["markets"][name].rf)
        gi, g1 = g.loc[f"indicador h={cfg.h_main}"], g.loc["1x"]
        lines.append(f"- **{name}** ({_d(a)} → {_d(b)}): el indicador (h = {cfg.h_main}) pasa el "
                     + ", ".join(f"{pct_es(share[k], 0)} en {k}x" if k else f"{pct_es(share[k], 0)} en letras" for k in (0, 1, 2, 3))
                     + f"; {num_es(sw, 1)} cambios al año; crecimiento {pct_es(gi['crecimiento log'])} frente a "
                     f"{pct_es(g1['crecimiento log'])} del 1x; caída máxima {pct_es(gi['maxDD'], 0)} frente a "
                     f"{pct_es(g1['maxDD'], 0)}.")
    rep.p("\n".join(lines))

    # ── especificación
    rep.h("1. Especificación congelada")
    rep.p(f"Todos los parámetros están en `Config` y en {FROZEN.relative_to(LETF.parent)}; esta tabla los resume.")
    spec = pd.DataFrame({"qué es": PARAM_ES,
                         "valor": {k: (", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v))
                                   for k, v in cfg.to_dict().items()}})
    spec.index.name = "parámetro (Config)"
    tab("Parámetros del indicador y de su evaluación", spec)
    rep.p("Decisiones que el prerregistro no fija (tomadas antes de ver resultados de las carteras; ninguna usa la "
          "prueba):\n" + "\n".join(f"- {d}" for d in DECISIONS))

    # ── datos
    rep.h("2. Datos (todos recortados al 31-12-2018)")
    rows = []
    for name, mk in R["markets"].items():
        I = R["ind"][(name, cfg.h_main)].dropna(subset=["L"])
        rows.append({"mercado": name, "fondos": " / ".join(mk.tickers[k] for k in (1, 2, 3)),
                     "u desde": _d(mk.u.index[0]), "indicador desde": _d(I.index[0]),
                     "2x real desde": _d(mk.launch[2]), "3x real desde": _d(mk.launch[3]),
                     "s 2x (pb)": 1e4 * mk.spread[2], "s 3x (pb)": 1e4 * mk.spread[3],
                     "gasto 2x": mk.er[2], "gasto 3x": mk.er[3]})
    tab("Mercados, fondos y costes (s calibrado con datos hasta 2018)", pd.DataFrame(rows).set_index("mercado"),
        pct={"gasto 2x", "gasto 3x"})
    rep.p("\n".join(f"- {name}: " + "; ".join(mk.notes) for name, mk in R["markets"].items()))

    # ── componentes al final del desarrollo
    rep.h("3. Los componentes")
    rep.p("*Pie de fórmula: w = (T − 5)/T · θ²_a/(θ²_a + 1/T) es el peso de Kan y Zhou: la fracción del prior que "
          "sobrevive a la contracción hacia 0; con muchos años y una prima clara, w ≈ 1 − (ee/prior)².* Notación: T, "
          "sesiones con que se estima; θ²_a, razón de Sharpe diaria al cuadrado ajustada; ee, error típico del prior.")
    rows = []
    for (name, h), I in R["ind"].items():
        last = I.dropna(subset=["L"]).iloc[-1]
        J = I.dropna(subset=["L"])
        rows.append({"mercado": name, "h": h, "prior": last.prior, "ee prior": last.se_prior,
                     "años": last["T"] / DAYS, "w Kan–Zhou": last.w_kz, "prior contraído": last.shrunk,
                     "β tendencia": last.beta, "ee β": last.se_beta, "S": last.S, "μ − f": last.mu_f, "ee(μ)": last.ee,
                     "w medio": J.w_kz.mean(), "de/E[IV] medio": (J.de / J.V).mean()})
    tab("Deriva al cierre del desarrollo (31-12-2018) y medias de la ventana (anual)",
        pd.DataFrame(rows).set_index(["mercado", "h"]),
        pct={"prior", "ee prior", "prior contraído", "β tendencia", "ee β", "μ − f", "ee(μ)"}, digits=2)
    rows = []
    for name in MARKETS:
        E = R["evt"][name]
        I = R["ind"][(name, cfg.h_main)].dropna(subset=["L"])
        ev = E.reindex(I.index)
        bad3 = ev.ok_3 == False  # noqa: E712
        bad2 = ev.ok_2 == False  # noqa: E712
        rows.append({"mercado": name, "días": len(I), "3x descartado": bad3.mean(), "2x descartado": bad2.mean(),
                     "primer 3x descartado": _d(ev.index[bad3][0]) if bad3.any() else "—",
                     "último 3x descartado": _d(ev.index[bad3][-1]) if bad3.any() else "—",
                     "p anual 3x máx.": ev.p_anual_3.max(), "ξ de la GPD (final)": ev.xi.dropna().iloc[-1]})
    tab("Restricción de ruina (EVT condicional): fracción de días en que se descarta cada L",
        pd.DataFrame(rows).set_index("mercado"), pct={"3x descartado", "2x descartado"}, digits=4)
    rows = []
    for (name, h), I in R["ind"].items():
        J = I.dropna(subset=["L"])
        sh = J.L.value_counts(normalize=True).reindex([0, 1, 2, 3]).fillna(0)
        rows.append({"mercado": name, "h": h, "letras": sh[0], "1x": sh[1], "2x": sh[2], "3x": sh[3],
                     "L medio": J.L.mean(), "L* mediano": J.L_star.median(),
                     "cambios al año": (J.L.diff().fillna(0) != 0).sum() / (len(J) / DAYS),
                     "cambios sin histéresis al año": (J.L_objetivo.diff().fillna(0) != 0).sum() / (len(J) / DAYS),
                     "corr(L*, E[IV])": J.L_star.corr(J.V)})
    tab("Distribución del L que se tiene y cambios al año (desarrollo)", pd.DataFrame(rows).set_index(["mercado", "h"]),
        pct={"letras", "1x", "2x", "3x"})
    for h in cfg.horizons[:1]:
        fig_levels(R, rep, h)
        fig_drift(R, rep, h)

    # ── carteras
    rep.h("4. Carteras en la muestra de desarrollo (descriptivo, no es el contraste)")
    D = {}
    for name in MARKETS:
        a, b = windows[name]
        F = R["ret"][name].loc[a:b]
        D[name] = F
        g = growth_stats(F, R["expo"][name].loc[a:b], R["markets"][name].rf)
        tk = R["markets"][name].tickers
        g.index = [{"0x": "letras", "1x": f"1x ({tk[1]})", "2x": f"2x ({tk[2]})", "3x": f"3x ({tk[3]})",
                    "LRS": f"LRS 3x ({tk[3]})"}.get(k, k) for k in g.index]
        tab(f"{name}, {_d(a)} → {_d(b)}: crecimiento, caída máxima y exposición media (desarrollo)", g,
            pct={"crecimiento log", "CAGR", "vol", "maxDD"})
    fig_wealth(R, rep, {n: D[n] for n in ("S&P 500", "Nasdaq-100", "Bono largo")})

    # ── hipótesis en el desarrollo
    rep.h("5. Los estadísticos de H1–H8 en el desarrollo (orientativo)")
    rep.p("Las mismas funciones y reglas que aplicará r36 a la prueba reservada, aplicadas a la ventana común del "
          "desarrollo. Sirven para comprobar el código y para ver cómo se comportó el indicador antes de 2019; **no "
          "deciden ninguna hipótesis**. Las ventanas: acciones desde el "
          f"{_d(eq_start)}; bono largo y cartera de acciones y bonos desde el {_d(bond_start)}.")
    Ha, Hb = H["acciones"], H["bonos"]
    rows = []
    for key, src in (("H1", Ha["H1"]), ("H6", Hb["H6"])):
        for (name, h), c in src.items():
            t = c["tabla"]
            for bench in (r34.M_HARLOG, r34.M_GJR):
                dm = c["dm"][c["dm"]["frente a"] == bench].set_index("pérdida")["p (una cola)"]
                rows.append({"hipótesis": key, "mercado": name, "h": h, "frente a": bench,
                             "QLIKE HAR-RI / bench": t.loc[r34.M_HARRI, "QLIKE"] / t.loc[bench, "QLIKE"],
                             "p DM QLIKE": dm.get("QLIKE"), "p DM crecimiento": dm.get("crecimiento"),
                             "HAR-RI en MCS (QLIKE)": t.loc[r34.M_HARRI, "en MCS (QLIKE)"]})
    tab("H1 y H6 en el desarrollo: HAR-RI frente a los benchmarks (cociente de QLIKE: menos de 1, HAR-RI pierde menos)",
        pd.DataFrame(rows).set_index(["hipótesis", "mercado", "h", "frente a"]), digits=3)
    t2 = Ha["H2"]["tabla"].copy()
    tab(f"H2 en el desarrollo ({_d(t2.attrs['desde'])} → {_d(t2.attrs['hasta'])}): Δg del indicador (h = {cfg.h_main}) "
        "frente al 1x y al 2x fijos", t2.set_index(["estrategia", "frente a"]),
        pct={"Δg", "IC90 inferior", "IC90 superior"}, digits=3)
    yn = (lambda d: {k: ({True: "sí", False: "no"}[v] if isinstance(v, (bool, np.bool_)) else v)  # noqa: E731
                     for k, v in d.items()})
    rows = [{"mercado": n, **yn({k: v for k, v in s.items() if k not in ("desde", "hasta")})} for n, s in Ha["H3"].items()]
    tab("H3 en el desarrollo: SPA de las variantes h ∈ {21, 63} frente a la LRS 3x", pd.DataFrame(rows).set_index("mercado"),
        pct={"Δg mejor"}, digits=3)
    rows = [{"mercado": n, **yn(s)} for n, s in Ha["H4"].items()]
    tab("H4 en el desarrollo: caída máxima y probabilidad de perder el 50% a igual exposición media (a = indicador, "
        "b = LRS)", pd.DataFrame(rows).set_index("mercado"),
        pct={"maxDD a", "maxDD b", "P(pérdida ≥ 50%) a", "P(pérdida ≥ 50%) b"}, digits=3)
    rows = [{"mercado": n, **yn({k: v for k, v in s.items() if k not in ("desde", "hasta")})} for n, s in Ha["H5"].items()]
    tab(f"H5 en el desarrollo: descomposición de la varianza del L* continuo (h = {cfg.h_main})",
        pd.DataFrame(rows).set_index("mercado"), digits=3)
    if "H7" in Hb:
        t7 = Hb["H7"]["tabla"].copy()
        d7 = Hb["H7"]["caídas"]
        tab(f"H7 en el desarrollo ({_d(t7.attrs['desde'])} → {_d(t7.attrs['hasta'])}): indicador del bono largo frente "
            f"a TLT y UBT; caída máxima {pct_es(d7['maxDD a'], 0)} frente a {pct_es(d7['maxDD b'], 0)} de TMF",
            t7.set_index(["estrategia", "frente a"]), pct={"Δg", "IC90 inferior", "IC90 superior"}, digits=3)
    if "H8" in Hb:
        h8 = Hb["H8"]
        rows = [{"frente a": b, **{k: v for k, v in s.items() if k not in ("desde", "hasta", "frente a")}}
                for b, s in h8["spa"].items()]
        d8 = h8["caídas"]
        tab(f"H8 en el desarrollo: SPA de la cartera de acciones y bonos; caída máxima (h = {cfg.h_main}) "
            f"{pct_es(d8['maxDD a'], 0)} frente a {pct_es(d8['maxDD b'], 0)} de la HFEA 55/45",
            pd.DataFrame(rows).set_index("frente a"), pct={"Δg mejor"}, digits=3)
        SB = h8["retornos"]
        g = growth_stats(SB, None, R["markets"][cfg.sb_equity].rf)
        tab(f"Cartera de acciones y bonos, {_d(SB.index[0])} → {_d(SB.index[-1])} (desarrollo)",
            g.drop(columns="exposición media"), pct={"crecimiento log", "CAGR", "vol", "maxDD"})
        fig_stock_bond(R, rep, SB)

    # ── correlación prevista
    rows = []
    for h, C in R["corr"].items():
        J = C.dropna()
        exp_mean = C.realizada.expanding().mean().shift(h).reindex(J.index)
        rows.append({"h": h, "días": len(J), "desde": _d(J.index[0]),
                     "ECM HAR (z de Fisher)": float(((J.rho - J.realizada) ** 2).mean()),
                     "ECM realizada de 63 sesiones": float(((J.rho_63 - J.realizada) ** 2).mean()),
                     "ECM media creciente": float(((exp_mean - J.realizada) ** 2).mean()),
                     "correlación prevista media": float(J.rho.mean())})
    rep.h("6. La correlación prevista entre el S&P 500 y el bono largo")
    tab("Error cuadrático medio de la correlación prevista frente a la realizada en las h sesiones siguientes (desarrollo)",
        pd.DataFrame(rows).set_index("h"), digits=4)

    # ── configuración congelada
    rep.h("7. Configuración congelada y evaluación única de la prueba")
    doc = freeze(R, cfg) if write_frozen else None
    if doc is not None:
        rep.p(f"Escrita en `letf/preregistro/{FROZEN.name}`: parámetros, especificación, hipótesis con su regla de "
              f"decisión, funciones que llamará la evaluación, estimaciones del desarrollo (s de cada fondo, vencimientos, "
              f"prima de la pérdida de crecimiento), huellas de las salidas del indicador hasta el 31-12-2018 y el SHA-256 de "
              f"{len(doc['sha256'])} archivos de código. `letf_lab/research/r36_indicador_holdout.py` (run()) la carga, "
              "comprueba las huellas, se niega a correr si ya existe `letf/preregistro/articulo7_holdout_ejecutado.json` y "
              "evalúa H1–H8 en la prueba reservada una sola vez. No se ha ejecutado.")
        hy = pd.DataFrame({k: {"hipótesis": v["texto"], "regla": v["regla"]} for k, v in HYPOTHESES.items()}).T
        tab("Hipótesis y reglas de decisión congeladas", hy)
    rep.h("Salvedades")
    rep.p("\n".join([
        "- Solo desarrollo: ninguna cifra usa datos posteriores al 31-12-2018. Las carteras y los estadísticos de H1–H8 de "
        "este informe describen el desarrollo; las hipótesis se deciden en la prueba reservada, una vez, con r36.",
        "- La especificación se fijó antes de mirar las carteras del desarrollo y no se ha cambiado al verlas.",
        "- Los fondos 2x y 3x son sintéticos antes de su lanzamiento (2006–2010) y reales después; los de bonos de Direxion "
        "(TMF, TYD) siguieron a otro índice hasta 2010 (R33).",
        "- En los bonos, el s calibrado de UBT, UST y TYD es negativo (tabla 2): el sintético de esos fondos rinde más que "
        "su fórmula y el indicador los ve más baratos que gratis. En TYD (−46,5 pb) sale del ruido de sus precios (sesiones "
        "sin volumen y cierres erróneos, R33), y explica que el indicador del bono medio use el 3x el 14% de los días y "
        "crezca más que IEF en el desarrollo. El bono medio no entra en ninguna hipótesis.",
        "- Con κ = 1 en la deriva y en la varianza, el Kelly robusto es muy prudente: en el S&P 500 el indicador pasa casi "
        "el 60% de los días en letras. Es lo que fija el prerregistro; no se ha ajustado al verlo.",
        "- La restricción de ruina se apoya en unos pocos días extremos (1987) y en una Pareto generalizada estimada con "
        "los residuos de un GARCH gaussiano: es un orden de magnitud, no una probabilidad exacta.",
        "- El S&P 500 no tiene rango real antes de 1982, ni el Nasdaq-100 aperturas reales antes de 2000; el bono, "
        "rango solo desde 2002 y MOVE desde finales de 2002.",
    ]))
    rep.save()
    return R


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
