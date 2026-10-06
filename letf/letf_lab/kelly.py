"""Apalancamiento de Kelly: la prima en tiempo real y L*, compartidos por R10–R20.

Un solo estimador y un solo rótulo para los dos artículos.

Prima (PRIMA = «μ̂ − f̂»): media creciente de u − f con datos hasta la sesión
anterior (t − 1), ×252, con un mínimo de min_obs sesiones (diez años). u es la
rentabilidad diaria del índice con dividendos y f el tipo diario con que se
financia toda la exposición (L veces el capital; el efectivo rinde las letras).
Es el exceso realizado sobre la financiación, con
el mismo tipo de estimador (media histórica) que el L* con toda la historia.
No es el benchmark de Campbell-Thompson de R10 y R12, que es la media creciente
del exceso LOGARÍTMICO de h sesiones sobre las letras. El estimador anterior
(media creciente de u menos el f del día) suponía que la rentabilidad total no
cambia cuando cambia f. Cada fórmula resta después el
sobrecoste s del swap (y la comisión, si la lleva).

L*, con estos rótulos en todos los informes:
- LSTAR_APROX, (μ − f − s)/σ²: el máximo de la aproximación de segundo orden
  del crecimiento, g(L) ≈ r + L(μ − f − s) − ER − ½L²σ² (Kelly o Merton en
  tiempo continuo). No descuenta la comisión del fondo.
- LSTAR_EXACTO y LSTAR_EXACTO_SIN: el L de [0, L_MAX] que maximiza el
  crecimiento logarítmico medio realizado de un apalancamiento constante L con
  los instrumentos de la LRS: L/3 en el 3x y el resto en letras, reequilibrado
  a diario; por encima de 3, el 3x más un préstamo al tipo f + s (como
  r17.const_leverage). Con la comisión del fondo (el 3x tal cual) o sin ella
  (x3 + ER/252). Con 3x y letras la comisión cuesta ER/3 por unidad de
  exposición, así que el exacto con comisión queda ≈ ER/(3σ²) por debajo.
  Si el crecimiento aún sube en L_MAX, el L* exacto está por encima del
  intervalo: sale como np.inf (censurado, no una estimación) y fmt_lstar lo
  muestra como «≥ 4».
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ANN = 252
MIN_OBS = ANN * 10   # diez años antes de la primera prima (como r10.MIN_OBS y r12.MIN_OBS)
L_MAX = 4.0          # como la frontera g(L) de R17

PRIMA = "μ̂ − f̂"
PRIMA_DESC = ("media creciente de u − f (rentabilidad del índice con dividendos menos la "
              "financiación) con datos hasta la sesión anterior, ×252, desde el año 10")
LSTAR_APROX = "L* aproximado"
LSTAR_EXACTO_SIN = "L* exacto sin comisión"
LSTAR_EXACTO = "L* exacto con comisión"
LSTAR_DESC = {
    LSTAR_APROX: "(μ − f − s)/σ², máximo de la aproximación de segundo orden del crecimiento; "
                 "no descuenta la comisión del fondo",
    LSTAR_EXACTO_SIN: "L que maximiza el crecimiento logarítmico medio de L/3 en el 3x y el resto "
                      "en letras, con el 3x sin su comisión (x3 + ER/252)",
    LSTAR_EXACTO: "L que maximiza el crecimiento logarítmico medio de L/3 en el 3x (con su "
                  "comisión) y el resto en letras",
}


# ───────────────────────────── prima ─────────────────────────────

def premium(u: pd.Series, f: pd.Series, min_obs: int = MIN_OBS, ann: int = ANN) -> pd.Series:
    """Prima anual de Kelly en tiempo real, μ̂_t − f̂_t (PRIMA).

    Media creciente de u − f con datos hasta t − 1 (al menos min_obs sesiones),
    ×ann. Sin el sobrecoste s ni la comisión: los resta cada fórmula."""
    x = u - f.reindex(u.index)
    return x.shift(1).expanding(min_obs).mean() * ann


# ───────────────────────────── L* ─────────────────────────────

def lstar_approx(u, f, spread: float, ann: int = ANN) -> float:
    """LSTAR_APROX = (μ − f − s)/σ² con las medias y la varianza (ddof = 0) de la muestra.

    u, f: rentabilidades diarias alineadas y sin huecos (también las columnas de
    un remuestreo); spread: s anual."""
    u, f = np.asarray(u, float), np.asarray(f, float)
    return float((u.mean() * ann - f.mean() * ann - spread) / (u.var() * ann))


def const_leverage_returns(L: float, x3, rf, f=None, spread: float = 0.0, ann: int = ANN):
    """Rentabilidad diaria de un apalancamiento constante L, reequilibrado a diario:
    L/3 en el 3x y el resto en letras; por encima de 3, el 3x más un préstamo de
    L/3 − 1 al tipo f + s (la misma cartera que r17.const_leverage)."""
    if L > 3:
        if f is None:
            raise ValueError("por encima de 3 hace falta el tipo de financiación f")
        return (L / 3) * x3 - (L / 3 - 1) * (f + spread / ann)
    return (L / 3) * x3 + (1 - L / 3) * rf


def growth(L: float, x3, rf, f=None, spread: float = 0.0, ann: int = ANN) -> float:
    """Crecimiento logarítmico anual realizado del apalancamiento constante L
    (media de log(1 + r) ×ann, con las pérdidas acotadas en −99,99% como en R17)."""
    r = const_leverage_returns(L, np.asarray(x3, float), np.asarray(rf, float),
                               None if f is None else np.asarray(f, float), spread, ann)
    return float(np.mean(np.log1p(np.maximum(r, -0.9999))) * ann)


def lstar_exact(x3, rf, f=None, spread: float = 0.0, L_max: float = L_MAX,
                ann: int = ANN) -> float:
    """L de [0, L_max] que maximiza growth (LSTAR_EXACTO con el 3x tal cual;
    LSTAR_EXACTO_SIN si se le pasa x3 + ER/252). La curva es cóncava en L, así
    que basta una búsqueda acotada. Sin f, L_max baja a 3 (no hay préstamo).

    Si el crecimiento en L_max es al menos el del óptimo hallado, la curva aún
    sube en el tope y el verdadero L* está fuera del intervalo: devuelve np.inf
    (censurado por arriba, «≥ L_max»), nunca el tope como si fuera un óptimo."""
    x3, rf = np.asarray(x3, float), np.asarray(rf, float)
    f = None if f is None else np.asarray(f, float)
    if L_max > 3 and f is None:
        L_max = 3.0
    g = lambda L: growth(L, x3, rf, f, spread, ann)  # noqa: E731
    res = minimize_scalar(lambda L: -g(L), bounds=(0.0, L_max), method="bounded",
                          options={"xatol": 1e-4})
    if g(L_max) >= -res.fun:
        return np.inf
    return float(res.x)


def fmt_lstar(x: float, digits: int = 2, L_max: float = L_MAX) -> str:
    """Texto de un L* exacto para tablas: «≥ 4» si está censurado (np.inf)."""
    return f"≥ {L_max:g}" if np.isinf(x) else f"{x:.{digits}f}"


def lstar_table(d: dict, start=None, end=None, L_max: float = L_MAX) -> dict:
    """Las tres versiones de L* en [start, end] para un mercado con las claves de
    r12.make (u, f, rf, x3, spread, er), con los rótulos compartidos.

    Un L* exacto por encima de L_max sale como np.inf; para mostrarlo en una
    tabla, fmt_lstar («≥ 4»). No aplica d["eval_from"]: la ventana es la que se
    pida (p. ej. start = d["eval_from"] para dejar fuera una reconstrucción).
    L_max = 3 limita el exacto a los instrumentos de la LRS sin préstamo (L/3 en
    el 3x y el resto en letras), el máximo de la frontera 3x + letras de R17."""
    df = pd.concat([d["u"], d["f"], d["rf"], d["x3"]], axis=1,
                   keys=["u", "f", "rf", "x3"]).loc[start:end].dropna()
    s, er = d["spread"], d["er"]
    return {"desde": df.index[0].date(), "hasta": df.index[-1].date(), "años": len(df) / ANN,
            LSTAR_APROX: lstar_approx(df.u, df.f, s),
            LSTAR_EXACTO_SIN: lstar_exact(df.x3 + er / ANN, df.rf, df.f, s, L_max),
            LSTAR_EXACTO: lstar_exact(df.x3, df.rf, df.f, s, L_max)}
