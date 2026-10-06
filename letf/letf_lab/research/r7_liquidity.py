"""R7 — Liquidez y costes de ejecución del universo LETF.

"Menos volumen que las acciones modernas" no implica mala liquidez: un ETF se
puede crear/redimir y el market maker cubre con el subyacente. Lo relevante es:
1. Volumen en $ (mediana 60 días) y en % del AUM implícito.
2. Spread cotizado: sólo con IB. (Los estimadores desde OHLC diario —Roll,
   Corwin-Schultz, Abdi-Ranaldo— se probaron y dan 0-80 bps para ETFs cuyo
   spread real es ~1 tick: inservibles a este nivel de liquidez.)
3. Ruido de cierre vs NAV implícito: std del residuo de tracking (de R2) y su
   reversión → coste oculto de operar MOC en el LETF frente al subyacente.
4. Si IB está conectado: snapshot real de bid/ask (spread cotizado).
"""

from __future__ import annotations

import pandas as pd

from letf_lab import data, sim
from letf_lab.report import Report
from letf_lab.research.common import pair
from letf_lab.universe import UNIVERSE


def run():
    rep = Report("r7_liquidity", "R7 — Liquidez y costes de ejecución")
    rep.p(__doc__.split("\n", 2)[2])
    raw = data.daily([e.ticker for e in UNIVERSE] + sorted({e.underlying for e in UNIVERSE}))
    rows = []
    for e in UNIVERSE:
        d = raw[e.ticker].tail(250)
        p = pair(e.ticker)
        spread = sim.calibrate_spread(p.letf, p.u, e.leverage, p.rf, e.expense_ratio)
        eps = (p.letf - sim.letf_returns(p.u, e.leverage, p.rf, e.expense_ratio, spread)).tail(250)
        u = raw[e.underlying].tail(250)
        rows.append({
            "ticker": e.ticker, "L": e.leverage,
            "vol_$M_mediana": (d.close * d.volume).tail(60).median() / 1e6,
            "vol_$M_subyac": (u.close * u.volume).tail(60).median() / 1e6,
            "ruido_cierre_bps": eps.std() * 1e4,
            "reversión_ε": eps.autocorr(1),
        })
    df = pd.DataFrame(rows).set_index("ticker").sort_values("vol_$M_mediana", ascending=False)
    rep.h("Último año")
    rep.table(df, digits=1)
    rep.p("`ruido_cierre_bps` × |reversión| aproxima lo que se pierde (o se puede "
          "capturar) por operar el LETF al cierre en vez de replicar con el subyacente.")

    if data.ib_port():
        q = data.ib_quotes([e.ticker for e in UNIVERSE])
        rep.h("Snapshot IB (bid/ask real)")
        rep.table(q, digits=2)
    else:
        rep.p("_IB no conectado: sin snapshot de bid/ask. Arrancar IB Gateway y "
              "relanzar para añadir el spread cotizado real._")
    rep.save()
    return df


if __name__ == "__main__":
    run()
