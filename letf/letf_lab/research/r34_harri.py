"""R34 — HAR-RI: el modelo de varianza del indicador del artículo 7 (muestra de desarrollo).

Prerregistro: letf/preregistro/articulo7_indicador.md (componente 2 y ampliación a bonos en
«Desviaciones»). HAR logarítmico sobre la varianza realizada diaria, semanal y mensual (Corsi
2009) más la varianza de rango diaria (estimador de Yang y Zhang con apertura, máximo, mínimo
y cierre), la semivarianza negativa, un componente de saltos, log VIX², log(VIX/VIX3M) y log
VVIX desde que existe cada uno (en bonos, el MOVE en el lugar del VIX); estimado con ventana
creciente y solo datos pasados, devuelto a niveles con el factor de Duan (1983) en tiempo real.
Para cada día da la MEDIA y la DISPERSIÓN previstas de la varianza integrada de las próximas h
sesiones (h = 21, principal; h = 63, sensibilidad): la dispersión sale de la distribución de
los errores fuera de muestra del propio modelo, también en tiempo real.

Disciplina de la muestra reservada. La prueba (1-1-2019 → 25-9-2026) no se usa aquí para nada:
todas las series se recortan al 31-12-2018 ANTES de calcular nada (MarketInputs.truncate), la
última previsión evaluada tiene su ventana objetivo entera dentro de 2018 y las pruebas
comprueban que cambiar cualquier dato posterior no cambia ninguna cifra.

Objetivo. IV_t(h) = (252/h)·Σ u²_s, s = t+1 … t+h: varianza integrada anualizada de las h
sesiones que siguen al cierre de t, con u la rentabilidad total diaria del índice (la misma
base que R10). Es el término ½·L²·IV(h) del crecimiento del fondo L (componente 1).

Especificación (fijada antes de ver resultados; constantes abajo):
- Grupos base (solo cierres, desde el primer día): HAR = log RV diaria, semanal (5) y mensual
  (21), la de r10.har_features; semivarianza negativa = fracción de la varianza de las 5
  últimas sesiones que viene de los días de caída; saltos = max(0, 1 − BV/RV) en 21 sesiones,
  con BV la variación bipotencia (π/2)·Σ|u_s||u_{s−1}| (Barndorff-Nielsen y Shephard 2004).
- Rango: log de la varianza diaria de Yang y Zhang o² + k·c² + (1 − k)·RS (o: apertura sobre
  el cierre anterior; c: cierre sobre apertura; RS: Rogers y Satchell), k = 0,34/(1,34 +
  (n + 1)/(n − 1)) con n = 21, y sus medias de 5 y 21 sesiones. Un día sin rango válido
  (máximo ≤ mínimo, incoherente o marcado por el cargador del agente de datos) usa u² en su
  lugar si el mes tiene ≥ 80% de días válidos; si no, el rango no existe ese día. En el S&P
  500 el rango empieza el 21-4-1982 (antes, Yahoo trae el máximo y el mínimo «teóricos»).
- Implícita: log (VIX/100)², log(VIX/VIX3M) y log(VVIX/100), alineados al calendario de u con
  un arrastre de hasta 5 sesiones.
- Ventana creciente, reestimación cada 21 sesiones (como r10). En t solo entran pares
  (X_s, log IV_s) con s ≤ t − h. Los grupos base necesitan 10 años (r10.MIN_OBS); cada grupo
  adicional entra, en el orden en que empiezan sus datos, cuando hay 5 años de filas conocidas
  con todos sus regresores; cada modelo anidado se estima en la muestra en la que existen
  sus regresores. Cada día se usa el modelo más rico cuyos regresores existen ese día.
- Niveles: E[IV] = exp(f)·media de los cocientes z_s = IV_s/exp(f_s) ya observados (Duan en
  tiempo real, r10.duan_oos: ≥ 63 cocientes). Dispersión: de(IV) = exp(f)·desviación típica de
  esos cocientes y cuantiles exp(f)·q_α(z), con ≥ 252 cocientes.

Comparación (solo desarrollo): HAR-log + Duan de R10 (r10.har_var_forecast) y GJR-GARCH(1,1)
gaussiano (stats.garch_fit, reestimado cada 252 sesiones con ventana creciente y 10 años de
mínimo). Pérdidas QLIKE y la de crecimiento ½·SR²·(ρ − 1)² de r10.growth_loss (ρ =
realizada/prevista; prima constante del desarrollo), conjunto de mejores modelos (MCS, Hansen,
Lunde y Nason 2011; bloque 63) y Diebold–Mariano con Hansen–Hodrick (h − 1 retardos).
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field, replace
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from letf_lab import provenance, stats
from letf_lab.report import FIG_W, Report, comma_fig, legend_band, num_es, pct_es
from letf_lab.research import r10_predictability as r10

log = logging.getLogger(__name__)

# ───────────────────────────── constantes (fijadas antes de estimar) ─────────────────────────────

DEV_END = pd.Timestamp("2018-12-31")      # fin de la muestra de desarrollo
TEST_START = pd.Timestamp("2019-01-01")   # prueba reservada: NO se usa en esta etapa
TEST_END = pd.Timestamp("2026-09-25")
HORIZONS = (21, 63)                       # principal y sensibilidad
MIN_OBS = r10.MIN_OBS                     # 10 años de filas conocidas antes de la primera previsión
MIN_OBS_EXTRA = 252 * 5                   # 5 años para que entre un grupo adicional
REFIT = 21                                # reestimación del HAR (como stats.expanding_forecast)
GARCH_REFIT = 252                         # reestimación del GJR-GARCH
DUAN_MIN = r10.DUAN_MIN                   # cocientes mínimos para el factor de Duan (y ≥ h)
DISP_MIN = 252                            # cocientes mínimos para la dispersión y los cuantiles
QUANTS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
YZ_N = 21
YZ_K = 0.34 / (1.34 + (YZ_N + 1) / (YZ_N - 1))   # peso de Yang y Zhang (≈ 0,139)
RANGE_MIN_FRAC = 0.8                      # fracción mínima de días con rango válido en el mes
IMPLIED_FFILL = 5                         # arrastre máximo de la implícita (sesiones)
FLOOR = 1e-6                              # suelo de las varianzas anualizadas antes del log (r10)
Y_FLOOR = 1e-8                            # suelo del objetivo antes del log (r10)
BLOCK = 63
B_MCS = 1000
ALPHA_MCS = 0.10
SEED = 34
START_SP = r10.START_SP                   # 1960-01-01, como R10 y el prerregistro
VVIX_FROM = pd.Timestamp("2007-01-01")    # 2006: huecos y cierres imposibles (15,71 el 15-3-2006)

GROUPS: dict[str, tuple[str, ...]] = {
    "har": ("log_rv_d", "log_rv_w", "log_rv_m"),
    "semivarianza": ("semi_neg_w",),
    "saltos": ("saltos_m",),
    "rango": ("log_rr_d", "log_rr_w", "log_rr_m"),
    "implícita": ("log_iv2",),
    "pendiente": ("log_iv_slope",),
    "volvol": ("log_vov",),
}
BASE_GROUPS = ("har", "semivarianza", "saltos")          # solo necesitan cierres
EXTRA_GROUPS = ("rango", "implícita", "pendiente", "volvol")
ALL_GROUPS = BASE_GROUPS + EXTRA_GROUPS
IMPLIED_GROUPS = ("implícita", "pendiente", "volvol")
GROUP_LABEL = {"har": "HAR (diaria, semanal, mensual)", "semivarianza": "semivarianza negativa",
               "saltos": "saltos (bipotencia)", "rango": "rango de Yang y Zhang",
               "implícita": "implícita al cuadrado", "pendiente": "pendiente de la implícita",
               "volvol": "volatilidad de la implícita"}
COL_LABEL = {"const": "constante", "log_rv_d": "log RV diaria", "log_rv_w": "log RV semanal",
             "log_rv_m": "log RV mensual", "semi_neg_w": "semivarianza negativa (fracción, 5 s.)",
             "saltos_m": "saltos (fracción, 21 s.)", "log_rr_d": "log rango diario",
             "log_rr_w": "log rango semanal", "log_rr_m": "log rango mensual",
             "log_iv2": "log implícita²", "log_iv_slope": "log pendiente de la implícita",
             "log_vov": "log volatilidad de la implícita"}

M_HARRI, M_HARLOG, M_GJR = "HAR-RI", "HAR-log + Duan", "GJR-GARCH"
MAIN_MODELS = (M_HARRI, M_HARLOG, M_GJR)
ABLATIONS = {                              # cadena descriptiva: qué aporta cada bloque
    "HAR + semivarianza + saltos": ("har", "semivarianza", "saltos"),
    "HAR + semivarianza + saltos + rango": ("har", "semivarianza", "saltos", "rango"),
}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9994"
MODEL_COLOR = {M_HARRI: BLUE, M_HARLOG: GREY, M_GJR: ORANGE}


# ───────────────────────────── entradas de un mercado ─────────────────────────────

@dataclass
class MarketInputs:
    """Todo lo que el modelo necesita de un mercado. Solo u es obligatoria.

    u: rentabilidad total diaria (fracción) del índice o del bono. ohlc: apertura, máximo,
    mínimo y cierre del índice de precio o del ETF (puede empezar más tarde que u o faltar).
    implied: columnas opcionales iv (implícita en puntos de volatilidad: VIX o MOVE), iv3m (la
    de 3 meses) y vov (su volatilidad: VVIX). f: financiación diaria (fracción), solo para la
    prima de la pérdida de crecimiento. labels: nombre de cada columna implícita en el
    informe. notes: procedencia, para el informe."""

    name: str
    u: pd.Series
    ohlc: pd.DataFrame | None = None
    implied: pd.DataFrame | None = None
    f: pd.Series | None = None
    labels: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def truncate(self, end: pd.Timestamp | str | None) -> "MarketInputs":
        """Copia con TODAS las series recortadas a fechas ≤ end (la disciplina de la prueba)."""
        if end is None:
            return self
        end = pd.Timestamp(end)
        cut = (lambda x: None if x is None else x.loc[x.index <= end].copy())
        return replace(self, u=cut(self.u), ohlc=cut(self.ohlc), implied=cut(self.implied), f=cut(self.f))

    def last_date(self) -> pd.Timestamp:
        ds = [x.index.max() for x in (self.u, self.ohlc, self.implied, self.f)
              if x is not None and len(x)]
        return max(ds)


def assert_until(obj, end: pd.Timestamp, what: str = "") -> None:
    """Falla si obj (serie, tabla o MarketInputs) tiene alguna fecha posterior a end."""
    end = pd.Timestamp(end)
    if isinstance(obj, MarketInputs):
        last = obj.last_date()
    else:
        last = obj.index.max() if len(obj) else None
    if last is not None and last > end:
        raise AssertionError(f"{what or 'serie'}: datos hasta {last.date()} > {end.date()} "
                             "(la muestra reservada no puede entrar)")


# ───────────────────────────── regresores ─────────────────────────────

def realized_iv(u: pd.Series, h: int) -> pd.Series:
    """IV_t(h) = (252/h)·Σ_{s=t+1}^{t+h} u_s²: NaN en las h últimas sesiones de la serie."""
    return (u.astype(float) ** 2).rolling(h).sum().shift(-h) * 252 / h


def range_variance(ohlc: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Varianza diaria de Yang y Zhang anualizada, o² + k·c² + (1 − k)·RS (× 252), y la marca
    de día válido. o = log(apertura/cierre anterior), c = log(cierre/apertura), RS = log(H/C)·
    log(H/O) + log(L/C)·log(L/O). Válido: precios positivos, H > L, H ≥ max(O, C) y L ≤ min(O,
    C) con tolerancia relativa 1e-6 (redondeos de Yahoo; se recortan antes de calcular RS) y,
    si la tabla trae la marca range_ok del cargador del agente de datos (indicador.ohlc), esa
    marca también (descarta, p. ej., el rango «teórico» de ^GSPC antes del 21-4-1982). Con la
    apertura igual al cierre anterior (Yahoo antes de ~2007 en ^GSPC), o = 0. Es la misma cifra
    que indicador.range_variances()["yz"] × 252 en los días válidos."""
    lower = ohlc.rename(columns=str.lower)
    x = lower[["open", "high", "low", "close"]].astype(float)
    O, H, L, C = x.open, x.high, x.low, x.close
    pc = C.shift(1)
    tol = 1e-6
    valid = ((O > 0) & (H > 0) & (L > 0) & (C > 0) & (pc > 0) & (H > L)
             & (H >= np.maximum(O, C) * (1 - tol)) & (L <= np.minimum(O, C) * (1 + tol)))
    if "range_ok" in lower:
        valid &= lower["range_ok"].fillna(False).astype(bool)
    Hc, Lc = np.maximum(H, np.maximum(O, C)), np.minimum(L, np.minimum(O, C))
    o = np.log(O / pc)
    c = np.log(C / O)
    rs = np.log(Hc / C) * np.log(Hc / O) + np.log(Lc / C) * np.log(Lc / O)
    yz = (o ** 2 + YZ_K * c ** 2 + (1 - YZ_K) * rs) * 252
    return yz.where(valid), valid.fillna(False).astype(bool)


def _align_implied(implied: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Implícita al calendario de u con arrastre de hasta IMPLIED_FFILL sesiones (solo pasado)."""
    imp = implied.sort_index()
    imp = imp[~imp.index.duplicated(keep="last")]
    return imp.reindex(imp.index.union(index)).ffill(limit=IMPLIED_FFILL).reindex(index)


def build_features(u: pd.Series, ohlc: pd.DataFrame | None = None,
                   implied: pd.DataFrame | None = None) -> pd.DataFrame:
    """Regresores de HAR-RI conocidos al cierre de cada t (todas las columnas de GROUPS; NaN
    donde un grupo no existe)."""
    u = u.astype(float)
    X = r10.har_features(u)                                   # log_rv_d, log_rv_w, log_rv_m
    r2 = u ** 2
    s5 = r2.rolling(5).sum()
    neg5 = r2.where(u < 0, 0.0).rolling(5).sum()
    # (las sumas móviles de pandas acumulan errores de redondeo: se recorta a [0, 1])
    X["semi_neg_w"] = np.clip(np.where(s5.isna(), np.nan, np.where(s5 > 0, neg5 / s5.where(s5 > 0), 0.5)), 0, 1)
    rv_m = r2.rolling(21).sum()
    bv_m = (np.pi / 2) * (u.abs() * u.abs().shift(1)).rolling(21).sum()
    X["saltos_m"] = np.where(rv_m.isna() | bv_m.isna(), np.nan,
                             np.where(rv_m > 0, np.clip(1 - bv_m / rv_m.where(rv_m > 0), 0, 1), 0.0))
    for c in GROUPS["rango"]:
        X[c] = np.nan
    if ohlc is not None and len(ohlc):
        yz, valid = range_variance(ohlc)
        yz = yz.reindex(u.index)
        valid = valid.reindex(u.index, fill_value=False).astype(float)
        ok = valid.rolling(21).mean() >= RANGE_MIN_FRAC
        filled = yz.fillna(r2 * 252).clip(lower=FLOOR)
        X["log_rr_d"] = np.log(filled).where(ok)
        X["log_rr_w"] = np.log(filled.rolling(5).mean()).where(ok)
        X["log_rr_m"] = np.log(filled.rolling(21).mean()).where(ok)
    for c in GROUPS["implícita"] + GROUPS["pendiente"] + GROUPS["volvol"]:
        X[c] = np.nan
    if implied is not None and len(implied):
        imp = _align_implied(implied, u.index)
        if "iv" in imp:
            X["log_iv2"] = 2 * np.log((imp["iv"] / 100).where(imp["iv"] > 0))
            if "iv3m" in imp:
                X["log_iv_slope"] = np.log((imp["iv"] / imp["iv3m"]).where((imp["iv"] > 0) & (imp["iv3m"] > 0)))
        if "vov" in imp:
            X["log_vov"] = np.log((imp["vov"] / 100).where(imp["vov"] > 0))
    return X[[c for g in ALL_GROUPS for c in GROUPS[g]]]


# ───────────────────────────── el modelo ─────────────────────────────

def _ols(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(A)), A]), y, rcond=None)
    return beta


@dataclass
class HARRI:
    """HAR-RI con ventana creciente, grupos anidados, Duan y dispersión en tiempo real.

    Uso: HARRI(h=21).predict(u, ohlc, implied, end=DEV_END) → DataFrame diario con
    realizada, log_prev, factor, media (E[IV]), de (desviación típica de IV), q05 … q95,
    modelo (grupos en uso) y n_ajuste. groups restringe los bloques (p. ej. ("har",) es el
    HAR-log + Duan de R10). Tras predict: chain_ (modelos anidados de la última reestimación:
    grupos, columnas, coeficientes y filas) y entries_ (primer día en que se usa cada grupo)."""

    h: int = 21
    groups: tuple = ALL_GROUPS
    min_obs: int = MIN_OBS
    min_obs_extra: int = MIN_OBS_EXTRA
    refit: int = REFIT
    duan_min: int | None = None
    disp_min: int = DISP_MIN
    quantiles: tuple = QUANTS

    def features(self, u, ohlc=None, implied=None) -> pd.DataFrame:
        return build_features(u, ohlc, implied)

    def predict(self, u: pd.Series, ohlc: pd.DataFrame | None = None,
                implied: pd.DataFrame | None = None, end=None) -> pd.DataFrame:
        """Previsión diaria en tiempo real; con end, todo se recorta antes a fechas ≤ end."""
        if end is not None:
            end = pd.Timestamp(end)
            u = u[u.index <= end]
            ohlc = None if ohlc is None else ohlc[ohlc.index <= end]
            implied = None if implied is None else implied[implied.index <= end]
        u = u.dropna().astype(float)
        return self.predict_from(build_features(u, ohlc, implied), realized_iv(u, self.h))

    def predict_from(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Igual que predict, con los regresores y el objetivo ya construidos (mismo índice)."""
        h = self.h
        groups = tuple(g for g in ALL_GROUPS if g in self.groups)
        base = [g for g in groups if g in BASE_GROUPS]
        if "har" not in base:
            raise ValueError("HAR-RI necesita el grupo «har»")
        extras = [g for g in groups if g in EXTRA_GROUPS]
        cols = [c for g in groups for c in GROUPS[g]]
        Xa = X[cols].to_numpy(float)
        ly = np.log(y.clip(lower=Y_FLOOR)).to_numpy(float)
        n = len(ly)
        pos = {c: i for i, c in enumerate(cols)}
        gidx = {g: np.array([pos[c] for c in GROUPS[g]]) for g in groups}
        avail = {g: ~np.isnan(Xa[:, gidx[g]]).any(axis=1) for g in groups}
        first = {g: int(np.argmax(avail[g])) if avail[g].any() else n for g in extras}
        extras = sorted(extras, key=lambda g: (first[g], EXTRA_GROUPS.index(g)))
        base_av = np.logical_and.reduce([avail[g] for g in base])
        ok_y = ~np.isnan(ly)
        f = np.full(n, np.nan)
        lab = np.full(n, None, dtype=object)
        nfit = np.full(n, np.nan)
        chain, history = None, []
        for t in range(n):
            kn = max(t - h + 1, 0)
            if chain is None or t % self.refit == 0:
                known = ok_y[:kn] & base_av[:kn]
                if known.sum() >= self.min_obs:
                    chain = self._chain(Xa, ly, known, kn, base, extras, gidx, avail)
                    history.append((X.index[t], chain))
            if chain is None:
                continue
            for gs, idx, beta, nobs in reversed(chain):
                xt = Xa[t, idx]
                if not np.isnan(xt).any():
                    f[t] = beta[0] + xt @ beta[1:]
                    lab[t] = "+".join(gs)
                    nfit[t] = nobs
                    break
        f_log = pd.Series(f, index=X.index)
        out = self.levels(y, f_log)
        out["modelo"] = lab
        out["n_ajuste"] = nfit
        self.chain_ = [(gs, [cols[i] for i in idx], beta, nobs) for gs, idx, beta, nobs in (chain or [])]
        self.history_ = history
        self.entries_ = {g: (out.index[np.flatnonzero([g in (s or "").split("+") for s in lab])[0]]
                             if any(g in (s or "").split("+") for s in lab) else None) for g in groups}
        return out

    def _chain(self, Xa, ly, known, kn, base, extras, gidx, avail):
        """Modelos anidados de una reestimación: la base y, en orden, cada grupo adicional que
        ya tiene min_obs_extra filas conocidas junto con los anteriores."""
        gs = list(base)
        idx = np.concatenate([gidx[g] for g in gs])
        mask = known.copy()
        chain = [(tuple(gs), idx, _ols(Xa[:kn][mask][:, idx], ly[:kn][mask]), int(mask.sum()))]
        for g in extras:
            m2 = mask & avail[g][:kn]
            if m2.sum() >= self.min_obs_extra:
                gs, mask = gs + [g], m2
                idx = np.concatenate([gidx[x] for x in gs])
                chain.append((tuple(gs), idx, _ols(Xa[:kn][mask][:, idx], ly[:kn][mask]), int(mask.sum())))
        return chain

    def levels(self, y: pd.Series, f_log: pd.Series) -> pd.DataFrame:
        """De la previsión del log a niveles con los cocientes z_s = y_s/exp(f_s) conocidos en t
        (s ≤ t − h): media (Duan, r10.duan_oos), desviación típica y cuantiles."""
        h = self.h
        dmin = max(DUAN_MIN, h) if self.duan_min is None else self.duan_min
        ef = np.exp(f_log)
        z = (y / ef).where(y.notna() & f_log.notna())
        factor = r10.duan_oos(y, f_log, h, dmin)
        out = pd.DataFrame({"realizada": y, "log_prev": f_log, "factor": factor,
                            "media": ef * factor,
                            "de": ef * z.expanding(self.disp_min).std().shift(h)})
        for q in self.quantiles:
            out[qcol(q)] = ef * z.expanding(self.disp_min).quantile(q).shift(h)
        return out


def qcol(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


# ───────────────────────────── benchmarks ─────────────────────────────

def gjr_forecast(u: pd.Series, horizons=HORIZONS, min_obs: int = MIN_OBS,
                 refit: int = GARCH_REFIT) -> pd.DataFrame:
    """GJR-GARCH(1,1) gaussiano, ventana creciente: los parámetros de cada tramo [k, k + refit)
    se estiman con u hasta k − 1; la varianza del día siguiente se filtra con los datos hasta t.
    Varianza media prevista de las h sesiones siguientes, anualizada, por horizonte."""
    r = u.fillna(0.0).to_numpy(float)
    n = len(r)
    out = {h: np.full(n, np.nan) for h in horizons}
    for start in range(min_obs, n, refit):
        p = stats.garch_fit(r[:start], "gjr")
        hn = stats.garch_filter(r - r[:start].mean(), p)
        seg = slice(start, min(start + refit, n))
        for h in horizons:
            out[h][seg] = stats.garch_horizon_var(hn[seg], p, h) * 252
    return pd.DataFrame(out, index=u.index)


# ───────────────────────────── evaluación ─────────────────────────────

def premium_dev(mi: MarketInputs) -> float:
    """Prima neta constante μ − f (anual) de la muestra que se pasa (ya recortada): escala la
    pérdida de crecimiento, igual para todos los modelos (el MCS y DM no dependen de ella)."""
    if mi.f is None:
        raise ValueError(f"{mi.name}: falta la financiación f para la pérdida de crecimiento")
    df = pd.concat([mi.u, mi.f], axis=1, keys=["u", "f"]).dropna()
    return float((df.u - df.f).mean() * 252)


def losses(F: pd.DataFrame, premium: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """QLIKE y pérdida de crecimiento (pb/año) diarias de cada modelo frente a «realizada»."""
    y = F["realizada"].to_numpy(float)
    M = F.drop(columns="realizada")
    ql = pd.DataFrame({c: stats.qlike(y, M[c].to_numpy(float)) for c in M}, index=F.index)
    gl = pd.DataFrame({c: r10.growth_loss(y, M[c].to_numpy(float), premium) * 1e4 for c in M}, index=F.index)
    return ql, gl


def diebold_mariano(l_bench: pd.Series, l_model: pd.Series, h: int) -> dict:
    """d = pérdida del benchmark − pérdida del modelo; t de su media con Hansen–Hodrick (h − 1
    retardos; Newey–West 2h si no sale positiva). p unilateral (H1: el modelo pierde menos)."""
    d = (l_bench - l_model).dropna()
    fit = stats.ols_hac(d, pd.DataFrame(index=d.index), **stats.hac_overlap(h))
    t = float(fit.t[0])
    return {"Δ pérdida media": float(d.mean()), "t": t, "p (una cola)": float(norm.sf(t)),
            "estimador": fit.estimator, "n": len(d)}


def compare(F: pd.DataFrame, h: int, premium: float, rng, B: int = B_MCS,
            models=MAIN_MODELS) -> dict:
    """Tabla de pérdidas, MCS (QLIKE y crecimiento) y DM de HAR-RI frente a cada benchmark."""
    F = F[list(models) + ["realizada"]].dropna()
    ql, gl = losses(F, premium)
    y = F["realizada"]
    tab = pd.DataFrame({"QLIKE": ql.mean(), "pérdida de crecimiento (pb/año)": gl.mean(),
                        "realizada/prevista (media)": (y.values[:, None] / F[list(models)]).mean()})
    mq = stats.model_confidence_set(ql, BLOCK, B, rng, ALPHA_MCS)
    mg = stats.model_confidence_set(gl, BLOCK, B, rng, ALPHA_MCS)
    tab["p MCS (QLIKE)"] = mq.p_MCS
    tab["en MCS (QLIKE)"] = mq.en_MCS.map({True: "sí", False: "no"})
    tab["p MCS (crecimiento)"] = mg.p_MCS
    tab["en MCS (crecimiento)"] = mg.en_MCS.map({True: "sí", False: "no"})
    dm = []
    for b in models:
        if b == M_HARRI:
            continue
        for lname, L in (("QLIKE", ql), ("crecimiento", gl)):
            dm.append({"frente a": b, "pérdida": lname, **diebold_mariano(L[b], L[M_HARRI], h)})
    return {"tabla": tab.loc[list(models)], "dm": pd.DataFrame(dm), "n": len(F),
            "desde": F.index[0], "hasta": F.index[-1], "ql": ql, "gl": gl}


def dispersion_check(P: pd.DataFrame, index: pd.DatetimeIndex) -> dict:
    """Calibración de la dispersión en tiempo real: cobertura del intervalo q05–q95, colas y
    E[((IV − media)/de)²] (≈ 1 si de está bien calibrada); frecuencia de IV ≤ q_α."""
    D = P.reindex(index)
    D = D[D[["realizada", "media", "de", "q05", "q95"]].notna().all(axis=1)]
    y = D["realizada"]
    rel = {q: float((y <= D[qcol(q)]).mean()) for q in QUANTS}
    return {"n": len(D), "desde": D.index[0] if len(D) else None,
            "cobertura 90%": float(((y >= D.q05) & (y <= D.q95)).mean()),
            "bajo q05": float((y < D.q05).mean()), "sobre q95": float((y > D.q95).mean()),
            "z² medio": float((z2 := ((y - D.media) / D.de) ** 2).mean()),
            "z²: peso de 5 meses": float(z2.groupby(z2.index.to_period("M")).sum().nlargest(5).sum() / z2.sum())
            if len(z2) else np.nan,
            "de/media (mediana)": float((D.de / D.media).median()), "fiabilidad": rel}


def coefficients(X: pd.DataFrame, y: pd.Series, cols: list[str], h: int) -> pd.DataFrame:
    """Coeficientes del modelo más rico con toda la muestra conocida (descriptivo, HAC
    Hansen–Hodrick h − 1)."""
    ly = np.log(y.clip(lower=Y_FLOOR))
    fit = stats.ols_hac(ly, X[cols], **stats.hac_overlap(h))
    fr = fit.frame()
    fr.index = [COL_LABEL.get(c, c) for c in fr.index]
    fr.attrs.update(n=fit.n, r2=fit.r2)
    return fr


@dataclass
class Config:
    """Parámetros de la evaluación (los de por defecto son los fijados; las pruebas los encogen)."""

    horizons: tuple = HORIZONS
    min_obs: int = MIN_OBS
    min_obs_extra: int = MIN_OBS_EXTRA
    garch_refit: int = GARCH_REFIT
    disp_min: int = DISP_MIN
    B: int = B_MCS
    seed: int = SEED
    ablations: bool = True


def evaluate_market(mi: MarketInputs, end=DEV_END, cfg: Config | None = None) -> dict:
    """Todo el análisis de un mercado con datos ≤ end: previsiones diarias (HAR-RI, HAR-log +
    Duan, GJR-GARCH y la cadena descriptiva), pérdidas, MCS, DM y calibración de la dispersión,
    por horizonte. Sirve igual para acciones y para los bonos del agente de datos."""
    cfg = cfg or Config()
    end = pd.Timestamp(end)
    mi = mi.truncate(end)
    assert_until(mi, end, mi.name)
    u = mi.u.dropna().astype(float)
    X = build_features(u, mi.ohlc, mi.implied)
    prem = premium_dev(mi)
    G = gjr_forecast(u, cfg.horizons, cfg.min_obs, cfg.garch_refit)
    out = {"name": mi.name, "premium": prem, "u_desde": u.index[0], "u_hasta": u.index[-1],
           "X": X, "labels": mi.labels, "notes": mi.notes, "h": {}}
    for k, h in enumerate(cfg.horizons):
        y = realized_iv(u, h)
        model = HARRI(h=h, min_obs=cfg.min_obs, min_obs_extra=cfg.min_obs_extra, disp_min=cfg.disp_min)
        P = model.predict_from(X, y)
        hl = r10.har_var_forecast(y, u, h, min_obs=cfg.min_obs)
        F = pd.DataFrame({M_HARRI: P["media"], M_HARLOG: hl["nivel"], M_GJR: G[h], "realizada": y})
        abl = {}
        if cfg.ablations:
            for name, gs in ABLATIONS.items():
                abl[name] = HARRI(h=h, groups=gs, min_obs=cfg.min_obs, min_obs_extra=cfg.min_obs_extra,
                                  disp_min=cfg.disp_min).predict_from(X, y)["media"]
        Fall = F.join(pd.DataFrame(abl)) if abl else F
        common = Fall.dropna().index
        for obj, what in ((P, "previsión"), (F, "tabla de previsiones")):
            assert_until(obj, end, what)
        rng = np.random.default_rng(cfg.seed + 100 * k)
        res = {"P": P, "F": F, "abl": abl, "model": model, "common": common}
        res["full"] = compare(F.loc[common], h, prem, rng, cfg.B)
        imp_days = common[P.loc[common, "modelo"].fillna("").str.contains("implícita").to_numpy()]
        res["imp"] = (compare(F.loc[imp_days], h, prem, np.random.default_rng(cfg.seed + 100 * k + 1), cfg.B)
                      if len(imp_days) > 2 * BLOCK else None)
        if abl:
            ql, gl = losses(Fall.loc[common], prem)
            chain = [M_HARLOG, *ABLATIONS, M_HARRI]
            res["ablation"] = pd.DataFrame({"QLIKE": ql[chain].mean(),
                                            "pérdida de crecimiento (pb/año)": gl[chain].mean(),
                                            "QLIKE frente a HAR-log (%)": 100 * ql[chain].mean() / ql[M_HARLOG].mean(),
                                            "crecimiento frente a HAR-log (%)": 100 * gl[chain].mean() / gl[M_HARLOG].mean()})
        res["disp"] = dispersion_check(P, common)
        richest = model.chain_[-1] if model.chain_ else None
        res["coef"] = coefficients(X, y, richest[1], h) if richest else None
        res["coef_groups"] = richest[0] if richest else None
        last_eval = common[-1] if len(common) else None
        if last_eval is not None:  # la ventana objetivo de la última previsión evaluada cabe en end
            k_last = u.index.get_loc(last_eval)
            assert k_last + h <= len(u) - 1 and u.index[k_last + h] <= end
        out["h"][h] = res
    return out


# ───────────────────────────── datos de acciones ─────────────────────────────

EQUITY = {"S&P 500": {"ohlc": "^GSPC", "letf": "UPRO"}, "Nasdaq-100": {"ohlc": "^NDX", "letf": "TQQQ"}}
IMPLIED_LABELS = {"iv": "VIX", "iv3m": "VIX3M", "vov": "VVIX"}
# Primer día con máximo y mínimo reales cuando no está el cargador del agente de datos (es su
# indicador.RANGE_START): en ^GSPC, Yahoo trae hasta el 20-4-1982 el rango «teórico» (suma de
# los máximos y mínimos de cada valor), 1,4–4,5 veces la varianza de cierre a cierre.
RANGE_FROM = {"^GSPC": pd.Timestamp("1982-04-21"), "^NDX": pd.Timestamp("1985-10-01"),
              "TLT": pd.Timestamp("2002-07-30"), "IEF": pd.Timestamp("2002-07-30")}


def _indicador():
    """El módulo del agente de datos (letf_lab/data/indicador.py) si existe; si no, None."""
    try:
        return importlib.import_module("letf_lab.data.indicador")
    except ImportError:
        return None


def _indicador_ohlc(ticker: str, end=DEV_END) -> tuple[pd.DataFrame | None, str | None]:
    """OHLC del cargador del agente de datos (indicador.ohlc, con sus marcas range_ok), si ya
    existe y funciona; si no, None (se usa la caché de Yahoo)."""
    mod = _indicador()
    if mod is None:
        return None, None
    for fn in ("ohlc", "index_ohlc", "load_ohlc", "ohlc_daily"):
        f = getattr(mod, fn, None)
        if not callable(f):
            continue
        try:
            try:
                df = f(ticker, end=end)
            except TypeError:
                df = f(ticker)
        except Exception as e:  # noqa: BLE001 — el cargador es opcional
            log.warning("indicador.%s(%s) falló (%s); uso la caché de Yahoo", fn, ticker, e)
            continue
        if isinstance(df, pd.DataFrame) and {"open", "high", "low", "close"} <= {str(c).lower() for c in df.columns}:
            return df.rename(columns=str.lower), f"letf_lab.data.indicador.{fn}"
    return None, None


def load_ohlc(ticker: str, end=DEV_END) -> tuple[pd.DataFrame, str]:
    """OHLC diario recortado a end: primero el cargador del agente de datos (import perezoso,
    con sus marcas range_ok); si no, la caché de Yahoo del laboratorio (sources.daily, que no
    refresca lo que ya está) con range_ok = falso antes de RANGE_FROM[ticker]."""
    df, src = _indicador_ohlc(ticker, end)
    if df is None:
        from letf_lab.data import sources

        df = sources.daily(ticker)[ticker][["open", "high", "low", "close"]].copy()
        start = RANGE_FROM.get(ticker)
        df["range_ok"] = True if start is None else df.index >= start
        src = f"Yahoo {ticker} (caché; rango desde {dmy(start) if start is not None else 'el primer día'})"
    return df[df.index <= pd.Timestamp(end)], src


def equity_implied(end=DEV_END) -> tuple[pd.DataFrame, list[str]]:
    """VIX (Cboe, 1990–), VIX3M (Cboe desde el 18-9-2009; antes, ^VIX3M de Yahoo desde el
    17-7-2006, idéntico a Cboe donde coinciden) y VVIX (Cboe, desde VVIX_FROM)."""
    from letf_lab.data import sources, vix

    v = vix.cboe_index("VIX", end=end)
    v3 = vix.cboe_index("VIX3M", end=end)
    try:
        y3 = sources.daily("^VIX3M")["^VIX3M"]["close"]
        y3 = y3[(y3.index < v3.index[0]) & (y3.index <= end)]
        v3 = pd.concat([y3, v3]).sort_index()
    except Exception as e:  # noqa: BLE001
        log.warning("sin ^VIX3M de Yahoo (%s): VIX3M solo desde Cboe", e)
    vv = vix.cboe_index("VVIX", end=end)
    vv = vv[vv.index >= VVIX_FROM]
    imp = pd.concat({"iv": v, "iv3m": v3, "vov": vv}, axis=1).sort_index()
    notes = [f"VIX de Cboe desde el {dmy(v.index[0])}",
             f"VIX3M desde el {dmy(v3.index[0])} (Yahoo hasta el {dmy(vix.cboe_index('VIX3M', end=end).index[0])}, "
             "Cboe después)",
             f"VVIX de Cboe desde el {dmy(vv.index[0])} (2006 descartado: huecos y cierres imposibles)"]
    return imp, notes


def dmy(x) -> str:
    return pd.Timestamp(x).strftime("%-d-%-m-%Y")


def equity_inputs(name: str, end=DEV_END, ohlc_loader: Callable | None = None) -> MarketInputs:
    """S&P 500 (history.sp500_long desde START_SP, la base de R10) o Nasdaq-100
    (common.long_underlying("TQQQ")), OHLC del índice de precio (load_ohlc u ohlc_loader(ticker,
    end)), VIX/VIX3M/VVIX y la financiación interbancaria (history.funding_long). Todo recortado
    a end (por defecto, el fin del desarrollo)."""
    from letf_lab.data import history
    from letf_lab.research.common import long_underlying

    spec = EQUITY[name]
    if name == "S&P 500":
        u = history.sp500_long()
        u = u[u.index >= START_SP]
    else:
        u = long_underlying(spec["letf"])
    ohlc, src = (ohlc_loader or load_ohlc)(spec["ohlc"], end)
    imp, notes = equity_implied(end)
    mi = MarketInputs(name=name, u=u.rename("u"), ohlc=ohlc, implied=imp, f=history.funding_long(u.index),
                      labels=dict(IMPLIED_LABELS), notes=[f"rango: {src}"] + notes)
    return mi.truncate(end)


def bond_inputs(name: str, u: pd.Series, ohlc: pd.DataFrame | None = None, move: pd.Series | None = None,
                f: pd.Series | None = None, dur: pd.Series | None = None,
                notes: list | None = None) -> MarketInputs:
    """Enchufe de los bonos del agente de datos: u = rentabilidad total diaria del bono (sintético
    desde 1962 y el ETF real después), ohlc = apertura, máximo, mínimo y cierre de TLT o IEF
    (2002–), move = índice MOVE si existe (va en el lugar del VIX; sin pendiente ni VVIX). El
    MOVE es volatilidad del RENDIMIENTO en pb anuales: con dur (duración modificada del bono,
    conocida al cierre de t) se pasa a volatilidad del precio en puntos, MOVE·dur/100, que es la
    escala de u; sin dur entra tal cual (en logaritmos solo cambia la constante). f por defecto:
    history.funding_long."""
    from letf_lab.data import history

    imp, labels = None, {}
    if move is not None:
        iv = move.astype(float)
        if dur is not None:
            iv = (iv * dur.reindex(iv.index).ffill() / 100).dropna()
            labels = {"iv": "MOVE × duración"}
        else:
            labels = {"iv": "MOVE"}
        imp = pd.DataFrame({"iv": iv})
    return MarketInputs(name=name, u=u.rename("u"), ohlc=ohlc, implied=imp,
                        f=history.funding_long(u.index) if f is None else f,
                        labels=labels, notes=list(notes or []))


BOND_NAMES = {"largo": "Bono largo (TLT)", "medio": "Bono medio (IEF)"}


def bond_markets(kinds=("largo", "medio"), end=DEV_END) -> dict[str, MarketInputs]:
    """Los bonos del agente de datos (letf_lab/data/indicador.py, import perezoso) listos para
    run(bonds=bond_markets) o evaluate_market: u = indicador.bond_underlying(kind), rango de
    indicador.ohlc(TLT o IEF), MOVE de indicador.move() pasado a precio con la duración del bono
    sintético (indicador.cmt_bond(maturity(kind)).dur). Todo recortado a end."""
    ind = _indicador()
    if ind is None:
        raise ImportError("falta letf_lab.data.indicador (los bonos los prepara el agente de datos)")
    out = {}
    for kind in kinds:
        etf = ind.BOND_ETF[kind]
        M = ind.maturity(kind)
        u = ind.bond_underlying(kind, end=end)
        ohlc = ind.ohlc(etf, end=end)
        try:
            mv = ind.move(end=end)["move"].dropna()
        except Exception as e:  # noqa: BLE001 — el MOVE es opcional (prerregistro)
            log.warning("sin MOVE (%s): el bono va sin implícita", e)
            mv = None
        dur = ind.cmt_bond(M, end=end)["dur"]
        notes = [f"u: bono sintético a la par de {num_es(M, 1)} años (FRED) hasta el primer rendimiento de {etf} "
                 f"({dmy(ind._etf_returns(etf).index[0])}) y {etf} real después (indicador.bond_underlying)",
                 f"rango: indicador.ohlc({etf}) desde el {dmy(ohlc.index[0])}"]
        if mv is not None:
            notes.append(f"MOVE de Yahoo desde el {dmy(mv.index[0])}, pasado a volatilidad del precio con la "
                         f"duración modificada del bono de {num_es(M, 1)} años")
        out[BOND_NAMES[kind]] = bond_inputs(BOND_NAMES[kind], u.dropna(), ohlc, mv, dur=dur, notes=notes).truncate(end)
    return out


# ───────────────────────────── API para el indicador ─────────────────────────────

def forecast(mi: MarketInputs, horizons=HORIZONS, end=DEV_END, **kw) -> dict[int, pd.DataFrame]:
    """HAR-RI en tiempo real de un mercado para cada horizonte: {h: tabla diaria} con media
    (E[IV(h)], anual), de (su desviación típica prevista), q05 … q95, factor (Duan), log_prev,
    modelo (bloques en uso), n_ajuste y realizada (NaN en las h últimas sesiones). Todo se
    recorta antes a end; kw pasa a HARRI (min_obs, refit, disp_min…). end=None usa toda la
    serie: solo para la evaluación única de la prueba reservada, con el código congelado."""
    mi = mi.truncate(end)
    u = mi.u.dropna().astype(float)
    X = build_features(u, mi.ohlc, mi.implied)
    return {h: HARRI(h=h, **kw).predict_from(X, realized_iv(u, h)) for h in horizons}


API = [
    ("r34.forecast(mi, horizons=(21, 63), end=DEV_END, **kw)",
     "{h: tabla diaria}: media = E[IV(h)] anual, de = su desviación típica prevista, q05…q95, factor, modelo"),
    ("r34.HARRI(h, groups, min_obs, min_obs_extra, refit, disp_min).predict(u, ohlc, implied, end)",
     "la clase; predict_from(X, y) con regresores ya hechos; tras predecir: chain_, history_, entries_"),
    ("r34.MarketInputs(name, u, ohlc, implied, f, labels, notes).truncate(end)",
     "entradas de un mercado (solo u obligatoria); implied: columnas iv, iv3m, vov"),
    ("r34.equity_inputs(«S&P 500» | «Nasdaq-100», end)", "rentabilidad total, OHLC, VIX/VIX3M/VVIX y f"),
    ("r34.bond_inputs(name, u, ohlc, move, f, dur) / r34.bond_markets(kinds, end)",
     "enchufe de los bonos (MOVE·dur/100 en el lugar del VIX); bond_markets lee letf_lab.data.indicador"),
    ("r34.build_features(u, ohlc, implied) / r34.realized_iv(u, h) / r34.range_variance(ohlc)",
     "regresores conocidos al cierre de t, objetivo IV_t(h) y varianza diaria de Yang y Zhang"),
    ("r34.gjr_forecast(u, horizons, min_obs, refit)", "benchmark GJR-GARCH(1,1) en tiempo real"),
    ("r34.evaluate_market(mi, end=DEV_END, cfg) / r34.run(bonds=None, end=DEV_END, cfg)",
     "evaluación de desarrollo (QLIKE, crecimiento, MCS, DM, calibración) e informe reports/r34_harri"),
]


# ───────────────────────────── figuras ─────────────────────────────

def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_forecasts(R: dict, rep: Report, h: int = 21):
    """Volatilidad prevista (raíz de E[IV]) con su banda del 90% frente a la realizada, una
    previsión cada h sesiones (ventanas que no se solapan)."""
    names = list(R)
    fig, axes = plt.subplots(len(names), 1, figsize=(FIG_W, 3.4 * len(names) + 0.9), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        P = R[name]["h"][h]["P"]
        common = R[name]["h"][h]["common"]
        D = P.loc[common].dropna(subset=["media", "q05", "q95", "realizada"]).iloc[::h]
        x = D.index
        ax.fill_between(x, 100 * np.sqrt(D.q05), 100 * np.sqrt(D.q95), color=BLUE, alpha=0.18, lw=0)
        ax.plot(x, 100 * np.sqrt(D.realizada), color=INK, lw=0, marker="o", ms=2.6, alpha=0.75)
        ax.plot(x, 100 * np.sqrt(D.media), color=BLUE, lw=1.6)
        ax.set_title(name)
        ax.set_ylabel("volatilidad anual (%)")
        top = float(np.nanpercentile(100 * np.sqrt(D.realizada), 99.5))
        ax.set_ylim(0, max(top, float(np.nanmax(100 * np.sqrt(D.q95)))) * 1.05)
        _style(ax)
    hnd = [plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.18, lw=0), plt.Line2D([], [], color=BLUE, lw=2),
           plt.Line2D([], [], color=INK, lw=0, marker="o", ms=5)]
    legend_band(fig, hnd, ["HAR-RI: banda del 90% (cuantiles 5 y 95)", "HAR-RI: raíz de la varianza prevista",
                           f"realizada en las {h} sesiones siguientes"], where="bottom", ncol=2, fontsize=11)
    fig.suptitle(f"Varianza de las próximas {h} sesiones: prevista y realizada (desarrollo)")
    comma_fig(fig)
    rep.fig(fig, f"prevision_h{h}",
            f"Figura: raíz de la varianza integrada prevista por HAR-RI para las {h} sesiones siguientes, con su banda "
            f"del 90% (cuantiles 5 y 95 de la distribución prevista en tiempo real), frente a la realizada. Una "
            f"previsión cada {h} sesiones, para que las ventanas no se solapen; solo la muestra de desarrollo "
            f"(hasta el 31-12-2018)")


def fig_relative(R: dict, rep: Report):
    """Pérdida media de HAR-RI y de GJR-GARCH en % de la del HAR-log + Duan (100 = igual)."""
    rows = [(f"{name} · h = {h}", R[name]["h"][h]["full"]["tabla"]) for name in R for h in R[name]["h"]]
    cols = (("QLIKE", "QLIKE"), ("pérdida de crecimiento (pb/año)", "Pérdida de crecimiento"))
    rel = {(col, m): [100 * t.loc[m, col] / t.loc[M_HARLOG, col] for _, t in rows]
           for col, _ in cols for m in (M_HARRI, M_GJR)}
    hi = max(max(v) for v in rel.values())
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 1.0 + 0.72 * len(rows) + 1.4), sharey=True, sharex=True)
    y = np.arange(len(rows))[::-1]
    for ax, (col, title) in zip(axes, cols):
        for j, (m, c) in enumerate(((M_HARRI, BLUE), (M_GJR, ORANGE))):
            off = 0.18 if j == 0 else -0.18
            v = rel[(col, m)]
            ax.barh(y + off, v, height=0.34, color=c, zorder=2)
            for yi, vi in zip(y, v):            # la cifra dentro de la barra, junto a su extremo
                ax.text(vi - 2, yi + off, num_es(vi, 0), va="center", ha="right", fontsize=11, color="white",
                        fontweight="bold", zorder=3)
        ax.axvline(100, color=INK, lw=1.2, ls="--", zorder=1)            # detrás de las barras
        ax.set_xlabel("% del HAR-log + Duan")
        ax.set_title(title)
        ax.set_xlim(0, max(hi * 1.05, 110))
        _style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=11)
    legend_band(fig, [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE)],
                [M_HARRI, M_GJR], where="bottom", ncol=2)
    fig.suptitle("Pérdida media frente al HAR-log + Duan (= 100; menos es mejor)")
    comma_fig(fig)
    rep.fig(fig, "perdidas_relativas",
            "Figura: pérdida media de HAR-RI y del GJR-GARCH en porcentaje de la del HAR-log + Duan de R10 (la línea "
            "de 100), en la ventana común de cada mercado y horizonte de la muestra de desarrollo. Menos es mejor")


def fig_reliability(R: dict, rep: Report):
    """Frecuencia con que la realizada queda por debajo de cada cuantil previsto."""
    fig, ax = plt.subplots(figsize=(FIG_W, 6.2))
    qs = np.array(QUANTS)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.0, ls=":")
    cols = [BLUE, ORANGE, AQUA, GREY]
    hnd, lab = [], []
    for i, name in enumerate(R):
        for h, ls in zip(R[name]["h"], ("-", "--")):
            rel = R[name]["h"][h]["disp"]["fiabilidad"]
            c = cols[i % len(cols)]
            (ln,) = ax.plot(qs, [rel[q] for q in QUANTS], color=c, ls=ls, lw=2.0, marker="o", ms=7)
            hnd.append(ln)
            lab.append(f"{name}, h = {h}")
    ax.set_xlabel("cuantil previsto (nivel nominal)")
    ax.set_ylabel("frecuencia con que la realizada\nqueda por debajo")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    tk = [q for q in qs if q not in (0.10, 0.90)]            # 0,05 y 0,10 (0,90 y 0,95) se pisan
    ax.set_xticks(tk)
    ax.set_xticklabels([num_es(q, 2) for q in tk])
    _style(ax)
    legend_band(fig, hnd, lab, where="bottom", ncol=2, fontsize=11)
    ax.set_title("Calibración de la dispersión prevista (desarrollo)")
    comma_fig(fig)
    rep.fig(fig, "calibracion",
            "Figura: calibración de la distribución prevista de la varianza. Para cada cuantil previsto (eje "
            "horizontal), la fracción de días en que la varianza realizada quedó por debajo (eje vertical); una "
            "previsión bien calibrada cae sobre la diagonal punteada")


# ───────────────────────────── informe ─────────────────────────────

def _fmt_p(p: float) -> str:
    return "< 0,001" if p < 0.001 else num_es(p, 3)


def _dm_view(dm: pd.DataFrame) -> pd.DataFrame:
    v = dm.copy()
    v["p (una cola)"] = v["p (una cola)"].map(_fmt_p)
    return v.set_index(["frente a", "pérdida"])


def _entries_table(R: dict) -> pd.DataFrame:
    """Primer día en que la previsión en tiempo real usa cada bloque, por mercado y horizonte."""
    rows = []
    for name, r in R.items():
        for h, res in r["h"].items():
            e = res["model"].entries_
            rows.append({"mercado": name, "h": h,
                         **{GROUP_LABEL[g]: dmy(e[g]) if e.get(g) is not None else "—" for g in ALL_GROUPS}})
    return pd.DataFrame(rows).set_index(["mercado", "h"])


def _open_stale(R: dict, end) -> dict:
    """Fracción de los días con rango válido del desarrollo en que la apertura es el cierre
    anterior (Yahoo no trae apertura real): el término nocturno o² vale 0 esos días."""
    out = {}
    for name, spec in EQUITY.items():
        if name not in R:
            continue
        o, _ = load_ohlc(spec["ohlc"], end)
        ok = o["range_ok"].astype(bool) if "range_ok" in o else pd.Series(True, index=o.index)
        o = o[ok]
        out[name] = float((np.abs(o.open / o.close.shift(1) - 1) < 1e-6).mean())
    return out


def run(bonds: Callable[[], dict] | dict | None = None, end=DEV_END, cfg: Config | None = None) -> dict:
    """Evalúa HAR-RI en el S&P 500 y el Nasdaq-100 (y en los bonos que se pasen: un dict
    {nombre: MarketInputs} o una función que lo devuelve) con datos ≤ end, y escribe
    reports/r34_harri. end > DEV_END se rechaza: esta etapa no puede tocar la prueba."""
    from letf_lab.research.r26_vix_data import Tables

    end = pd.Timestamp(end)
    if end > DEV_END:
        raise ValueError("R34 es la etapa de desarrollo: end no puede pasar del 31-12-2018")
    cfg = cfg or Config()
    markets = {name: equity_inputs(name, end) for name in EQUITY}
    if bonds is not None:
        extra = bonds() if callable(bonds) else bonds
        markets.update({k: v.truncate(end) for k, v in extra.items()})
    R = {}
    for name, mi in markets.items():
        log.info("R34: %s", name)
        R[name] = evaluate_market(mi, end, cfg)

    # comprobación: HAR-RI restringido al HAR reproduce el HAR-log + Duan de R10
    sp = R["S&P 500"]
    h0 = cfg.horizons[0]
    res0 = sp["h"][h0]
    only_har = HARRI(h=h0, groups=("har",), min_obs=cfg.min_obs,
                     disp_min=cfg.disp_min).predict_from(sp["X"], res0["F"]["realizada"])["media"]
    gap_r10 = float((only_har - res0["F"][M_HARLOG]).abs().max())

    rep = Report("r34_harri", "R34 — HAR-RI: el modelo de varianza del indicador (artículo 7, desarrollo)")
    tab = Tables(rep)
    rep.p(__doc__.split("\n", 2)[2])
    rep.p(f"Generado con el commit {provenance.git_head()[:10]}"
          f"{' (con cambios sin commit)' if provenance.dirty(('research/r34_harri.py',)) else ''}. "
          f"Todas las cifras usan datos hasta el {dmy(end)}; la prueba reservada "
          f"({dmy(TEST_START)} → {dmy(TEST_END)}) no se ha usado ni mirado.")

    # ── resumen
    rep.h("Resumen")
    lines = []
    for name, r in R.items():
        for h, res in r["h"].items():
            t = res["full"]["tabla"]
            sg = lambda x: ("+" if x > 0 else "") + num_es(x, 0)  # noqa: E731
            q = 100 * (t.loc[M_HARRI, "QLIKE"] / t.loc[M_HARLOG, "QLIKE"] - 1)
            g = 100 * (t.loc[M_HARRI, "pérdida de crecimiento (pb/año)"] / t.loc[M_HARLOG, "pérdida de crecimiento (pb/año)"] - 1)
            qg = 100 * (t.loc[M_HARRI, "QLIKE"] / t.loc[M_GJR, "QLIKE"] - 1)
            inq = [m for m in MAIN_MODELS if t.loc[m, "en MCS (QLIKE)"] == "sí"]
            ing = [m for m in MAIN_MODELS if t.loc[m, "en MCS (crecimiento)"] == "sí"]
            d = res["disp"]
            lines.append(
                f"- **{name}, h = {h}** ({dmy(res['full']['desde'])} → {dmy(res['full']['hasta'])}): QLIKE de "
                f"HAR-RI {sg(q)}% frente al HAR-log + Duan y {sg(qg)}% frente al GJR-GARCH; pérdida de "
                f"crecimiento {sg(g)}% frente al HAR-log + Duan. MCS (90%) con QLIKE: {', '.join(inq)}; con la "
                f"pérdida de crecimiento: {', '.join(ing)}. Banda del 90% de la dispersión: cubre el "
                f"{pct_es(d['cobertura 90%'], 0)} de los días.")
    rep.p("\n".join(lines))
    rep.p(f"Comprobación: HAR-RI restringido al bloque HAR reproduce el HAR-log + Duan de R10 (diferencia máxima de "
          f"la varianza prevista, S&P 500, h = {h0}: " + f"{gap_r10:.1e}".replace(".", ",").replace("-", "−") + ").")

    # ── datos
    rep.h("1. Datos (todos recortados al 31-12-2018)")
    rows = []
    for name, r in R.items():
        X = r["X"]
        first = {g: X[list(GROUPS[g])].dropna().index.min() for g in ALL_GROUPS}
        rows.append({"mercado": name, "rentabilidades desde": dmy(r["u_desde"]),
                     "hasta": dmy(r["u_hasta"]),
                     "rango desde": dmy(first["rango"]) if pd.notna(first["rango"]) else "—",
                     "implícita desde": dmy(first["implícita"]) if pd.notna(first["implícita"]) else "—",
                     "pendiente desde": dmy(first["pendiente"]) if pd.notna(first["pendiente"]) else "—",
                     "VVIX desde": dmy(first["volvol"]) if pd.notna(first["volvol"]) else "—",
                     "prima μ − f (anual)": r["premium"]})
    tab("Series de cada mercado y primer día en que existe cada bloque de regresores", pd.DataFrame(rows).set_index("mercado"),
        pct={"prima μ − f (anual)"})
    notes = [f"- {name}: " + "; ".join(r["notes"]) for name, r in R.items()]
    rep.p("\n".join(notes))
    stale = _open_stale(R, end)
    if stale:
        rep.p("Salvedad del rango: en la caché de Yahoo la apertura de ^GSPC es casi siempre el cierre anterior hasta "
              "2006 y la de ^NDX hasta 2000 (" + " y ".join(f"{pct_es(v, 0)} de los días con rango válido del "
                                                          f"{k}" for k, v in stale.items())
              + " en el desarrollo): esos días el término nocturno o² vale 0 y la varianza de Yang y Zhang se reduce a "
              "k·c² + (1 − k)·RS con c la rentabilidad de cierre a cierre. El máximo y el mínimo sí son del día; un día "
              "con el máximo por debajo de la apertura así registrada (o el mínimo por encima) se marca incoherente y usa "
              "u² en su lugar. Los coeficientes del bloque de rango se estiman con esa mezcla de calidades.")

    rep.h("Decisiones de esta etapa (el prerregistro no las fija; ninguna se eligió mirando la prueba)")
    rep.p("\n".join([
        "- **Rango del S&P 500 desde el 21-4-1982, no desde 1960.** Hasta el 20-4-1982 Yahoo trae para ^GSPC el "
        "máximo y el mínimo «teóricos» (suma de los de cada valor), de 1,4 a 4,5 veces la varianza de cierre a "
        "cierre; el cargador del agente de datos (indicador.ohlc) los marca como no válidos y aquí no entran. La "
        "base del modelo (HAR, semivarianza y saltos, solo cierres) sí empieza en 1960, como R10.",
        "- **Modelos anidados.** La base necesita 10 años de filas conocidas; cada bloque adicional (rango, "
        "implícita, pendiente, VVIX o MOVE) entra, en el orden en que empiezan sus datos, cuando hay 5 años de filas "
        "conocidas con todos sus regresores, y cada día se usa el modelo más rico cuyos regresores existen ese día.",
        "- **VIX3M** de Cboe desde el 18-9-2009 y, antes, el ^VIX3M de Yahoo desde el 17-7-2006. **VVIX** desde "
        "2007 (en 2006 hay huecos y cierres imposibles). El **Nasdaq-100** usa el VIX, como fija el prerregistro.",
        "- **Dispersión.** Una sola distribución creciente de los cocientes realizada/prevista fuera de muestra, "
        "sin separar por modelo: la desviación típica del logaritmo del error no es menor con los modelos más ricos "
        "(sale mayor en la década de 2010, con menos volatilidad y más sorpresas relativas), así que separar no "
        "estrecharía la banda.",
        "- **GJR-GARCH(1,1)** gaussiano reestimado cada 252 sesiones con ventana creciente y 10 años de mínimo (R12 "
        "lo reestimaba cada 504).",
        "- **Pérdida de crecimiento** con la prima neta constante μ − f de todo el desarrollo (tabla 1): solo escala "
        "la pérdida, igual para todos los modelos; el MCS y el Diebold–Mariano no dependen de ella.",
        "- **Bonos.** Se enchufan con run(bonds=bond_markets): el MOVE (volatilidad del rendimiento, pb) se pasa a "
        "volatilidad del precio con la duración modificada del bono sintético, MOVE·dur/100, antes de tomar "
        "logaritmos.",
    ]))

    # ── cuándo entra cada bloque
    rep.h("2. Cuándo entra cada bloque (primer día en que la previsión lo usa)")
    tab("Primer día en que la previsión en tiempo real usa cada bloque (base: 10 años de filas conocidas; "
        "cada bloque adicional: 5 años con todos sus regresores)", _entries_table(R))

    # ── comparación
    rep.h("3. HAR-RI frente a los benchmarks (H1 y H6 se contrastan en la prueba, no aquí)")
    rep.p("*Pie de fórmula: QLIKE = ρ − log ρ − 1 y pérdida de crecimiento = ½·SR²·(ρ − 1)², con ρ la varianza "
          "realizada entre la prevista; la segunda es el crecimiento anual que se pierde al fijar el apalancamiento "
          "de Kelly con la varianza prevista en vez de la verdadera.* Notación: SR² = m²/IV, con m la prima neta "
          "constante μ − f del desarrollo (tabla 1). MCS: conjunto de modelos que no se distinguen del mejor al 90% "
          "(bootstrap estacionario, bloque 63). DM: Diebold–Mariano de la pérdida del benchmark menos la de HAR-RI "
          "(positivo: HAR-RI pierde menos), p de una cola.")
    for name, r in R.items():
        for h, res in r["h"].items():
            for key, win in (("full", "ventana común"), ("imp", "días con volatilidad implícita en HAR-RI")):
                c = res.get(key)
                if c is None:
                    continue
                if key == "imp" and c["n"] == res["full"]["n"]:
                    continue                      # todos los días usan ya la implícita: sería la misma tabla
                tab(f"{name}, h = {h}, {win} ({dmy(c['desde'])} → {dmy(c['hasta'])}; {num_es(c['n'], 0)} días): "
                    "pérdida media, MCS al 90%", c["tabla"], digits=3)
                tab(f"{name}, h = {h}, {win}: Diebold–Mariano de HAR-RI frente a cada benchmark", _dm_view(c["dm"]),
                    digits=3)
    fig_relative(R, rep)

    # ── qué aporta cada bloque
    rep.h("4. Qué aporta cada bloque (descriptivo)")
    rep.p("Cadena anidada en la misma ventana común: el HAR-log + Duan de R10, más semivarianza negativa y saltos, "
          "más el rango, y HAR-RI completo (más la volatilidad implícita desde que existe). Descriptivo: el modelo "
          "prerregistrado es el completo, sin elegir bloques con estos datos.")
    for name, r in R.items():
        for h, res in r["h"].items():
            if "ablation" in res:
                tab(f"{name}, h = {h}: pérdida media de cada eslabón de la cadena", res["ablation"], digits=3)

    # ── coeficientes
    rep.h("5. Coeficientes del modelo más rico al cierre del desarrollo")
    rep.p("Regresión del log de IV(h) sobre los regresores del modelo más rico que usa la previsión a final de 2018, "
          "con todas las filas conocidas en las que existen (t de Hansen–Hodrick con h − 1 retardos). Es lo que el "
          "modelo sabe al empezar la prueba; los coeficientes se reestiman cada 21 sesiones con ventana creciente.")
    for name, r in R.items():
        for h, res in r["h"].items():
            cf = res["coef"]
            if cf is None:
                continue
            tab(f"{name}, h = {h}: coeficientes ({num_es(cf.attrs['n'], 0)} filas, R² {num_es(cf.attrs['r2'], 2)})",
                cf, digits=3)

    # ── dispersión
    rep.h("6. La dispersión prevista")
    rep.p("*Pie de fórmula: de(IV) = exp(f)·desviación típica de los cocientes IV/exp(f) ya observados*, con f la "
          "previsión del logaritmo: la incertidumbre de la varianza es proporcional a su nivel. Es el de(IV) del "
          "Kelly robusto (componente 4). z² medio = media de ((IV − E[IV])/de)²: 1 si de está bien calibrada, pero lo "
          "dominan unos pocos meses extremos (la columna siguiente da el peso de los cinco con más z² en la suma; en "
          "el S&P 500 son sobre todo agosto–octubre de 1987), así que la calibración se lee mejor en la cobertura de "
          "la banda y en la figura de los cuantiles.")
    rows = []
    for name, r in R.items():
        for h, res in r["h"].items():
            d = res["disp"]
            rows.append({"mercado": name, "h": h, "días": d["n"], "desde": dmy(d["desde"]),
                         "cobertura 90%": d["cobertura 90%"], "bajo q05": d["bajo q05"], "sobre q95": d["sobre q95"],
                         "z² medio": d["z² medio"], "z²: peso de 5 meses": d["z²: peso de 5 meses"],
                         "de/media (mediana)": d["de/media (mediana)"]})
    tab("Calibración de la distribución prevista de IV(h) (en tiempo real, muestra de desarrollo)",
        pd.DataFrame(rows).set_index(["mercado", "h"]),
        pct={"cobertura 90%", "bajo q05", "sobre q95", "z²: peso de 5 meses", "de/media (mediana)"}, digits=2)
    fig_reliability(R, rep)
    for h in cfg.horizons[:1]:
        fig_forecasts(R, rep, h)

    rep.h("Salvedades")
    rep.p("\n".join([
        "- Solo desarrollo: ninguna cifra usa datos posteriores al 31-12-2018. H1 y H6 se deciden en la prueba "
        "reservada, una sola vez, con este código congelado.",
        "- Los regresores de volatilidad implícita solo existen desde 1990 (VIX), 2006 (VIX3M) y 2007 (VVIX): el "
        "modelo que los usa se estima con menos años (desde 5) que la base (desde 10), y con h = 63 eso deja pocas "
        "ventanas independientes.",
        "- El Nasdaq-100 usa el VIX (del S&P 500), como fija el prerregistro; el VXN no entra.",
        "- La dispersión usa todos los errores fuera de muestra desde el principio (ventana creciente, como el factor "
        "de Duan): si la incertidumbre relativa cambia con el tiempo, la banda tarda en adaptarse.",
        "- Bonos: este informe los incluye solo si se ejecuta run(bonds=bond_markets) (datos de "
        "letf_lab/data/indicador.py: bono sintético, rango de TLT/IEF y MOVE). También sirve "
        "evaluate_market(bond_inputs(…)) con series propias.",
        "- La dispersión de(IV) es proporcional a la media prevista (de/E[IV] casi constante en el tiempo): en el "
        "Kelly robusto del componente 4 actúa como un recorte multiplicativo del apalancamiento, 1/(1 + κ·de/E[IV]).",
    ]))
    rep.save()
    return R


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
