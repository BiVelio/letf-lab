"""R2 — Tracking, financiación implícita y ruido de precio de cierre.

Modelo: r_letf = L·r_u + (1−L)·rf − ER/252 − |L|·spread/252 + ε

1. `spread` implícito por fondo (coste real del swap sobre rf) — dato clave
   para que las simulaciones sintéticas (R3, R6) no sean optimistas.
2. Tracking error diario (bps) y autocorrelación lag-1 de ε: una autocorrelación
   fuertemente negativa indica que el *precio de cierre* se desvía del NAV y
   revierte al día siguiente (premium/discount) → posible ineficiencia explotable
   o, como mínimo, coste oculto de ejecución al cierre.
3. Evolución anual del spread implícito en TQQQ/SQQQ (¿cambió con tipos altos?).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from letf_lab import sim
from letf_lab.report import Report
from letf_lab.research.common import pair
from letf_lab.universe import BY_TICKER, UNIVERSE


def residuals(t: str, spread: float = 0.0) -> pd.Series:
    e = BY_TICKER[t]
    p = pair(t)
    model = sim.letf_returns(p.u, e.leverage, p.rf, e.expense_ratio, spread)
    return (p.letf - model).rename(t)


def run():
    rep = Report("r2_tracking", "R2 — Tracking error, financiación y ruido de cierre")
    rep.p(__doc__.split("\n", 2)[2])

    rows = []
    for e in UNIVERSE:
        p = pair(e.ticker)
        spread = sim.calibrate_spread(p.letf, p.u, e.leverage, p.rf, e.expense_ratio)
        eps = residuals(e.ticker, spread)
        simr = sim.letf_returns(p.u, e.leverage, p.rf, e.expense_ratio, spread)
        big = eps.abs() > 5 * eps.std()
        rows.append({
            "ticker": e.ticker, "L": e.leverage, "subyacente": e.underlying,
            "desde": p.index[0].date(), "spread_implícito": spread,
            "TE_diario_bps": eps.std() * 1e4,
            "autocorr_ε_lag1": eps.autocorr(1),
            "corr": p.letf.corr(simr),
            "días_outlier": int(big.sum()),
            "ε_post_2022_bps": eps[eps.index >= "2022-01-01"].std() * 1e4,
        })
    df = pd.DataFrame(rows).set_index("ticker")

    rep.h("Resultados por fondo")
    rep.table(df, pct={"spread_implícito"})
    rep.p("""Lectura:
- **spread_implícito** > 0 es coste adicional al ER (swaps, rebalanceo, cash drag).
  Donde el proxy de subyacente es imperfecto (SOXX pre-2021, XBI, GDX) el spread
  mezcla coste real con error de proxy → hay que interpretarlo con cautela.
- **autocorr_ε_lag1** muy negativa ⇒ el cierre del LETF se desvía y revierte al
  día siguiente. Es la huella de premium/discount al cierre (o de desincronía de
  precios entre LETF y subyacente). Oportunidad a investigar con NAV y bid/ask de IB.""")

    # spread implícito por año para los más líquidos
    fig, ax = plt.subplots(figsize=(10, 4))
    yr = {}
    for t in ["TQQQ", "SQQQ", "UPRO", "SPXU", "SOXL", "TMF"]:
        e = BY_TICKER[t]
        p = pair(t)
        yr[t] = p.groupby(p.index.year).apply(
            lambda g, e=e: sim.calibrate_spread(g.letf, g.u, e.leverage, g.rf, e.expense_ratio))
        ax.plot(yr[t].index, yr[t].values * 100, marker="o", label=t)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(title="Spread implícito anual del swap sobre rf (%)", ylabel="% anual")
    ax.legend(ncol=3)
    rep.fig(fig, "spread_por_año")
    rep.h("Spread implícito por año")
    rep.table(pd.DataFrame(yr), pct=set(yr))

    # ε del día t vs ε del día t+1 (reversión) para TQQQ
    eps = residuals("TQQQ")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(eps.shift(1) * 1e4, eps * 1e4, s=4, alpha=0.4)
    ax.set(xlim=(-80, 80), ylim=(-80, 80), xlabel="ε_t (bps)", ylabel="ε_t+1 (bps)",
           title=f"TQQQ: reversión del residuo (ρ={eps.autocorr(1):.2f})")
    rep.fig(fig, "tqqq_eps_reversion")

    rep.h("¿Es explotable la reversión? Captura teórica (sin costes)")
    cap = []
    for e in UNIVERSE:
        eps = residuals(e.ticker)
        x = (-np.sign(eps.shift(1)) * eps).dropna()  # apostar a que ε revierte
        recent = x[x.index >= "2020-01-01"]
        cap.append({"ticker": e.ticker, "captura_bps_día": x.mean() * 1e4,
                    "t": x.mean() / x.std() * np.sqrt(len(x)),
                    "captura_post2020_bps": recent.mean() * 1e4,
                    "t_post2020": recent.mean() / recent.std() * np.sqrt(len(recent))})
    rep.table(pd.DataFrame(cap).set_index("ticker"))
    rep.p("""Estrategia de papel: al cierre de t, si el LETF cerró *caro* respecto a
L×subyacente (ε_t>0) vender LETF / comprar L× subyacente y deshacer al cierre
siguiente (y al revés). **No es operable tal cual**: ε_t se conoce con los precios
de cierre, pero las órdenes MOC se envían antes de las 15:50. Además el cierre de
Yahoo del LETF puede no ser el de la subasta. Es la hipótesis nº1 a verificar con
IB: precios de subasta, NAV intradía (iNAV) y ejecución a 15:50-15:59 con LOC.""")

    rep.save()
    return df


if __name__ == "__main__":
    run()
