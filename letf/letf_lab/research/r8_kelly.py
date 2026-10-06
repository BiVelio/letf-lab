"""R8 — ¿Cuál es el apalancamiento óptimo? Curva de crecimiento g(L).

Bajo GBM, el crecimiento log de un fondo con reset diario es
    g(L) = rf + L·(μ − rf) − ½·L²·σ² − costes(L)
maximizado en L* = (μ − rf)/σ² (Kelly). Aquí se calcula g(L) *empírico*
simulando LETFs sintéticos con L ∈ [−1, 5] (incluye financiación y ER), por
década, y el L* que habría sido óptimo en cada una. El objetivo es ver si 3x
está dentro o fuera de la región razonable y cuánto varía L* entre regímenes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from letf_lab import metrics, sim
from letf_lab.report import Report
from letf_lab.research.common import long_underlying, rf

LS = np.round(np.arange(-1, 5.01, 0.25), 2)
SPREAD = 0.005  # coste de swap típico (R2)


def growth_curve(u: pd.Series, r: pd.Series) -> pd.Series:
    out = {}
    for L in LS:
        fund = L not in (0.0, 1.0)  # L=0 es caja (T-bill), L=1 el ETF índice
        x = sim.letf_returns(u, L, r, expense_ratio=0.009 if fund else 0.0,
                             spread=SPREAD if fund else 0.0)
        out[L] = metrics.cagr(x)
    return pd.Series(out)


def run():
    rep = Report("r8_kelly", "R8 — Apalancamiento óptimo por régimen")
    rep.p(__doc__.split("\n", 2)[2])
    for name, t in [("S&P 500", "UPRO"), ("Nasdaq-100", "TQQQ"), ("Semis", "SOXL")]:
        u = long_underlying(t)
        r = rf(u.index)
        dec = (u.index.year // 10) * 10
        curves = {f"{d}s": growth_curve(u[dec == d], r[dec == d]) for d in np.unique(dec)}
        curves["todo"] = growth_curve(u, r)
        C = pd.DataFrame(curves)
        best = pd.DataFrame({
            "L*_empírico": C.idxmax(),
            "CAGR en L*": C.max(),
            "CAGR 1x": C.loc[1.0], "CAGR 2x": C.loc[2.0], "CAGR 3x": C.loc[3.0],
        })
        rep.h(f"{name} (desde {u.index[0].year})")
        rep.table(best, pct={"CAGR en L*", "CAGR 1x", "CAGR 2x", "CAGR 3x"})
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for c in C.columns:
            ax.plot(C.index, C[c].clip(lower=-0.5) * 100, lw=2.5 if c == "todo" else 1,
                    color="k" if c == "todo" else None, label=c)
        ax.axvline(3, color="r", ls="--", lw=1)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set(title=f"{name}: CAGR (%) según apalancamiento diario L",
               xlabel="L", ylabel="CAGR % (recortado en −50%)")
        ax.legend(ncol=4, fontsize=8)
        rep.fig(fig, f"growth_{t.lower()}")
    rep.p("Lectura: L* varía enormemente por década (negativo en los 2000s, >3 en "
          "los 90s y 2010s). Ningún L fijo es 'correcto'; de ahí el interés de R6 "
          "(ajustar L a la volatilidad observada, que es lo único estimable ex ante "
          "con precisión — μ no lo es).")
    rep.save()


if __name__ == "__main__":
    run()
