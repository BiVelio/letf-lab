"""R4 — Flujos de rebalanceo al cierre (huella intradía de los LETFs).

Un LETF de apalancamiento L con AUM A debe comprar/vender al cierre
    ΔExposición ≈ A · (L² − L) · r_día
en la *misma dirección* del movimiento del día (L²−L > 0 tanto para 3x como
para −3x). Si ese flujo mueve precios, esperamos:
  H1: r(15:30→16:00) correlacionado positivamente con r(cierre previo→15:30).
  H2: parte revierte al día siguiente (presión transitoria).
  H3: efecto concentrado en días de |r| grande (donde el flujo es enorme).
Placebo: la misma regresión una hora antes (14:30→15:30) no debería mostrar
un salto en el coeficiente al pasar de días tranquilos a días extremos.

Datos: Yahoo 1h (≈2 años). Con IB se amplía a años de barras de 1 min
(`data.intraday(..., interval="1m")`) → mucha más potencia estadística.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from letf_lab import data
from letf_lab.report import Report

TICKERS = ["SPY", "QQQ", "IWM", "SOXX", "XLK", "TLT"]


def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Pendiente y t-stat robusto (HC1) de y = a + b·x."""
    X = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = XtX_inv @ (X.T * e**2) @ X @ XtX_inv * n / (n - k)
    return float(beta[1]), float(beta[1] / np.sqrt(cov[1, 1]))


def daily_segments(bars: pd.DataFrame) -> pd.DataFrame:
    """Por día: precio a las 14:30, 15:30, cierre y apertura."""
    b = bars.copy()
    b["d"] = b.index.normalize()
    b["hm"] = b.index.strftime("%H:%M")
    piv = b.pivot_table(index="d", columns="hm", values="open")
    close = b.groupby("d")["close"].last()
    first = b.groupby("d")["open"].first()
    df = pd.DataFrame({"open": first, "p1430": piv.get("14:30"), "p1530": piv.get("15:30"),
                       "close": close}).dropna()
    prev = df.close.shift(1)
    df["r_to_1430"] = df.p1430 / prev - 1
    df["r_1430_1530"] = df.p1530 / df.p1430 - 1
    df["r_to_1530"] = df.p1530 / prev - 1
    df["r_last"] = df.close / df.p1530 - 1
    df["r_next_on"] = df.open.shift(-1) / df.close - 1          # overnight siguiente
    df["r_next_day"] = df.close.shift(-1) / df.close - 1
    return df.dropna()


def run():
    rep = Report("r4_close_flows", "R4 — Flujos de rebalanceo de LETFs al cierre")
    rep.p(__doc__.split("\n", 2)[2])
    rows, segs = [], {}
    for t in TICKERS:
        df = daily_segments(data.intraday(t, "1h"))
        segs[t] = df
        big = df.r_to_1530.abs() > df.r_to_1530.abs().quantile(0.8)
        b1, t1 = ols(df.r_last.values, df.r_to_1530.values)
        b2, t2 = ols(df.r_next_on.values, df.r_to_1530.values)
        bb, tb = ols(df.r_last[big].values, df.r_to_1530[big].values)
        bs, ts = ols(df.r_last[~big].values, df.r_to_1530[~big].values)
        bp, tp = ols(df.r_1430_1530.values, df.r_to_1430.values)
        rows.append({"ticker": t, "días": len(df), "desde": df.index[0].date(),
                     "β último 30m": b1, "t": t1,
                     "β días extremos (top20%)": bb, "t_ext": tb,
                     "β días normales": bs, "t_norm": ts,
                     "β overnight sig.": b2, "t_on": t2,
                     "β placebo 14:30→15:30": bp, "t_pl": tp})
    df = pd.DataFrame(rows).set_index("ticker")
    rep.h("Regresiones")
    rep.table(df, digits=4)
    rep.p("""Cómo leerlo:
- β>0 con t>2 en *último 30m* apoya H1 (momentum al cierre).
- Si β en días extremos ≫ β en días normales → H3 (efecto no lineal, compatible
  con flujos ∝ r de los LETFs).
- β overnight < 0 → reversión (H2).
- **Caveat de datos**: el "close" de la última barra horaria de Yahoo no es
  necesariamente el precio de la subasta de cierre, que es donde los LETFs
  ejecutan (MOC). Con IB se puede pedir el precio de subasta y barras de 1 min.
- Con ~700 días la potencia es baja; es una *sonda*. La prueba seria requiere
  barras de 1 min de IB durante 5-10 años y AUM diario de los LETFs (para
  escalar el flujo esperado en $ y compararlo con el volumen de cierre).""")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, t in zip(axes, ["QQQ", "SOXX"]):
        s = segs[t]
        q = pd.qcut(s.r_to_1530, 10, labels=False)
        g = s.groupby(q)[["r_last", "r_next_on"]].mean() * 1e4
        g.plot.bar(ax=ax)
        ax.set(title=f"{t}: retorno medio (bps) por decil de r(cierre previo→15:30)",
               xlabel="decil (0 = día más bajista)", ylabel="bps")
    rep.fig(fig, "deciles")
    rep.save()
    return df


if __name__ == "__main__":
    run()
