"""R3 — Historia sintética larga (1960/1985 →) validada contra los LETFs reales.

Los LETFs existen desde 2006-2010: su historia real es casi toda mercado alcista
con tipos a cero. Para saber qué pasa en 1973-74, 1987, 2000-02 o 2008 hay que
sintetizarlos desde el índice, con el spread de swap calibrado en el periodo real.

1. Validación: sintético vs real en el periodo solapado (CAGR, correlación).
2. Resultados full-history y por década.
3. Peores drawdowns y tiempo de recuperación.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from letf_lab import metrics, sim
from letf_lab.report import Report
from letf_lab.research.common import long_underlying, pair, rf
from letf_lab.universe import BY_TICKER

FUNDS = ["SSO", "UPRO", "SDS", "SPXU", "QLD", "TQQQ", "SQQQ", "TNA", "TZA",
         "UDOW", "SDOW", "SOXL", "SOXS"]


def synthetic(t: str, spread: float | None = None) -> tuple[pd.Series, float]:
    """Retornos sintéticos del LETF `t` sobre la historia larga + spread usado."""
    e = BY_TICKER[t]
    u = long_underlying(t)
    r = rf(u.index)
    if spread is None:
        p = pair(t)
        spread = sim.calibrate_spread(p.letf, p.u, e.leverage, p.rf, e.expense_ratio)
    return sim.letf_returns(u, e.leverage, r, e.expense_ratio, spread).rename(t), spread


@lru_cache(maxsize=None)
def synthetic_actual(t: str) -> pd.Series:
    """Sintético antes del lanzamiento, real después (lo que usan R5/R6)."""
    s, _ = synthetic(t)
    a = pair(t).letf
    return pd.concat([s[s.index < a.index[0]], a]).rename(t)


def run():
    rep = Report("r3_synthetic", "R3 — Historia sintética larga de LETFs")
    rep.p(__doc__.split("\n", 2)[2])

    val, syn = [], {}
    for t in FUNDS:
        s, spread = synthetic(t)
        syn[t] = s
        a = pair(t).letf
        s_ov = s.reindex(a.index).dropna()
        a = a.reindex(s_ov.index)
        val.append({"ticker": t, "desde_real": a.index[0].date(), "spread": spread,
                    "CAGR_real": metrics.cagr(a), "CAGR_sint": metrics.cagr(s_ov),
                    "corr_diaria": a.corr(s_ov),
                    "maxDD_real": metrics.max_dd(a), "maxDD_sint": metrics.max_dd(s_ov)})
    rep.h("1. Validación en el periodo real")
    rep.table(pd.DataFrame(val).set_index("ticker"),
              pct={"spread", "CAGR_real", "CAGR_sint", "maxDD_real", "maxDD_sint"})
    rep.p("El spread se calibra para que el drift medio coincida, así que la "
          "prueba real es que maxDD y correlación también coincidan (forma del camino).")

    # full history, incluyendo el 1x como referencia
    full = {}
    for t in FUNDS:
        e = BY_TICKER[t]
        u = long_underlying(t)
        full[f"{e.underlying} (1x)"] = u
        full[t] = syn[t]
    rep.h("2. Historia completa")
    rep.table(metrics.table(full).drop(columns=["sortino"]))

    rep.h("Por década (CAGR)")
    dec = pd.DataFrame({k: v.groupby((v.index.year // 10) * 10).apply(metrics.cagr)
                        for k, v in full.items()}).T
    dec.columns = [f"{c}s" for c in dec.columns]
    rep.table(dec, pct=set(dec.columns))

    rep.h("3. Peores episodios")
    rows = []
    for t in ["UPRO", "TQQQ", "TNA", "SOXL"]:
        dd = metrics.drawdown(syn[t])
        w = (1 + syn[t]).cumprod()
        trough = dd.idxmin()
        peak = w[:trough].idxmax()
        rec = w[trough:][w[trough:] >= w[peak]]
        rows.append({"fondo": t, "pico": peak.date(), "valle": trough.date(),
                     "maxDD": dd.min(),
                     "recuperado": rec.index[0].date() if len(rec) else "no (a hoy)",
                     "años_bajo_agua": round(((rec.index[0] if len(rec) else w.index[-1]) - peak).days / 365.25, 1)})
    rep.table(pd.DataFrame(rows).set_index("fondo"))
    rep.p("Mensaje: TQQQ sintético perdió >99.9% en 2000-02 y no recuperó el pico "
          "de 2000 hasta más de una década después. El 'buy & hold' de un 3x "
          "funciona sólo si el periodo no contiene un crash de −50% en el índice.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, group in zip(axes, [["SSO", "UPRO", "SPXU"], ["QLD", "TQQQ", "SQQQ"]]):
        u = long_underlying(group[0])
        ax.semilogy((1 + u).cumprod(), label=f"{BY_TICKER[group[0]].underlying} 1x", color="k")
        for t in group:
            ax.semilogy((1 + syn[t]).cumprod(), label=t)
        ax.legend()
        ax.set(title="Riqueza (log) — sintético", ylim=(1e-4, None))
    rep.fig(fig, "wealth_long")
    rep.save()
    return syn


if __name__ == "__main__":
    run()
