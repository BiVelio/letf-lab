"""R5 — Cosechar el decay: vender en corto ambos lados (bull + bear).

Si ambos LETFs sufren −L(L−1)/2·σ², vender los dos y rebalancear captura ese
drag. Pero la posición es *short gamma*: pierde en tendencias fuertes (el lado
que va a favor del mercado crece sin límite) y paga préstamo de títulos (los
bear suelen ser caros de pedir prestados).

Variantes (por 1 $ de capital):
- PAIR: −0.5 en bull, −0.5 en bear; caja +2 (capital + producto del corto).
- HEDGE: −1/L en el LETF bull, +1 en el subyacente 1x → neutral en delta,
  aísla el decay del bull (y la financiación que paga el fondo).
Rebalanceo: semanal / mensual / trimestral (el delta deriva entre rebalanceos).
Caja: rf − 0.5% (IBKR remunera los saldos por debajo del benchmark).
Préstamo: rejilla 0–10% anual (conocer el borrow real vía IB es imprescindible).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from letf_lab import backtest, metrics
from letf_lab.report import Report
from letf_lab.research.common import pair, rf
from letf_lab.universe import BY_TICKER, PAIRS

REBAL = {"W": "W-FRI", "M": "ME", "Q": "QE"}
CASH_HAIRCUT = 0.005


def rebalance_mask(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """True en el último día hábil de cada periodo de rebalanceo."""
    last = pd.Series(index, index=index).resample(REBAL[freq]).last().dropna()
    return index.isin(last.values)


def run_drift(target: dict[str, float], R: pd.DataFrame, freq: str, borrow_bps: float,
              cash: pd.Series, cost_bps: float = 5.0) -> pd.Series:
    """Backtest con deriva real entre rebalanceos (pesos buy-and-hold intraperiodo)."""
    r = R[list(target)].fillna(0.0).to_numpy()
    c = cash.reindex(R.index).fillna(0.0).to_numpy()
    tgt = np.array(list(target.values()))
    reb = rebalance_mask(R.index, freq)
    held = np.zeros(len(tgt))
    out = np.empty(len(r))
    for i in range(len(r)):
        ret = held @ r[i] + (1 - held.sum()) * c[i] - np.abs(held.clip(max=0)).sum() * borrow_bps / 1e4 / 252
        held = held * (1 + r[i]) / (1 + ret) if ret > -1 else held * 0
        if reb[i]:
            ret -= np.abs(tgt - held).sum() * cost_bps / 1e4
            held = tgt.copy()
        out[i] = ret
    return pd.Series(out, index=R.index)


def run():
    rep = Report("r5_harvest", "R5 — Cosecha de decay (short bull + short bear)")
    rep.p(__doc__.split("\n", 2)[2])

    rows, curves = [], {}
    for bull, bear in PAIRS:
        # pair() respeta cambios de apalancamiento y limpia errores de datos
        R = pd.concat([pair(bull).letf, pair(bear).letf], axis=1, keys=[bull, bear]).dropna()
        cash = rf(R.index) - CASH_HAIRCUT / 252
        for freq in ["W", "M", "Q"]:
            for borrow in [0, 200, 500, 1000]:
                r = run_drift({bull: -0.5, bear: -0.5}, R, freq, borrow, cash)
                s = metrics.summary(r, rf(R.index))
                rows.append({"par": f"{bull}/{bear}", "rebal": freq, "borrow_bps": borrow,
                             "desde": R.index[0].date(), "CAGR": s["CAGR"], "vol": s["vol"],
                             "sharpe": s["sharpe"], "maxDD": s["maxDD"],
                             "peor_mes": r.resample("ME").apply(lambda x: (1 + x).prod() - 1).min()})
                if freq == "M" and borrow == 200:
                    curves[f"{bull}/{bear}"] = r
    df = pd.DataFrame(rows)

    rep.h("PAIR: resultados con préstamo 2% y rebalanceo mensual")
    base = df[(df.rebal == "M") & (df.borrow_bps == 200)].set_index("par").drop(columns=["rebal", "borrow_bps"])
    rep.table(base.sort_values("sharpe", ascending=False), pct={"peor_mes"})

    rep.h("Sensibilidad al coste de préstamo (Sharpe, rebalanceo mensual)")
    rep.table(df[df.rebal == "M"].pivot(index="par", columns="borrow_bps", values="sharpe"))
    rep.h("Sensibilidad a la frecuencia de rebalanceo (CAGR, préstamo 2%)")
    rep.table(df[df.borrow_bps == 200].pivot(index="par", columns="rebal", values="CAGR"),
              pct={"W", "M", "Q"})

    # HEDGE: short bull, long subyacente (delta-neutral)
    hrows = []
    for t in ["TQQQ", "UPRO", "SOXL", "TNA", "LABU", "TMF"]:
        e = BY_TICKER[t]
        p = pair(t)
        R = p[["letf", "u"]].rename(columns={"letf": t, "u": e.underlying})
        cash = p.rf - CASH_HAIRCUT / 252
        for borrow in [0, 200, 500]:
            r = run_drift({t: -1 / e.leverage, e.underlying: 1.0}, R, "M", borrow, cash)
            s = metrics.summary(r, p.rf)
            hrows.append({"hedge": f"−{t}/{e.leverage:g} + {e.underlying}", "borrow_bps": borrow,
                          "CAGR": s["CAGR"], "vol": s["vol"], "sharpe": s["sharpe"], "maxDD": s["maxDD"]})
    rep.h("HEDGE: −LETF/L + subyacente (rebalanceo mensual)")
    rep.table(pd.DataFrame(hrows).set_index(["hedge", "borrow_bps"]))
    rep.p("""Lectura:
- Sharpe medido en exceso de T-bill: la caja (producto del corto) cobra rf,
  así que un CAGR alto en 2023-2025 puede ser sobre todo tipo de interés.
- La cosecha es real y proporcional a σ² del subyacente, pero el perfil es de
  *vender volatilidad*: meses tranquilos positivos, pérdidas bruscas en rallies
  o crashes con tendencia (el `peor_mes` es la métrica a vigilar).
- Todo depende del borrow real. Los bear (SQQQ, SOXS, LABD…) suelen tener tasas
  altas y disponibilidad variable. **Siguiente paso con IB**: bajar el histórico
  de fee rate y shortable shares (tick 236) antes de dar esto por bueno.""")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for k in ["TQQQ/SQQQ", "UPRO/SPXU", "SOXL/SOXS", "TNA/TZA", "LABU/LABD", "TMF/TMV"]:
        ax.semilogy((1 + curves[k]).cumprod(), label=k)
    ax.set(title="PAIR short bull + short bear (rebal. mensual, préstamo 2%)")
    ax.legend(ncol=3)
    rep.fig(fig, "pair_curves")
    rep.save()
    return df


if __name__ == "__main__":
    run()
