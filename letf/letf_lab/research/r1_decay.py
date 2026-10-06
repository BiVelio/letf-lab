"""R1 — Anatomía del decay: ¿cuánto pierde realmente un LETF frente a "L× el subyacente"?

Preguntas:
1. Sobre ventanas de h días, ¿con qué frecuencia el LETF rinde menos que L× el
   retorno acumulado del subyacente? (lo que el inversor "espera").
2. ¿El gap log se explica por la varianza realizada como predice la teoría
   gap ≈ −L(L−1)/2 · σ²·h ?  Regresión del gap sobre la varianza realizada.
3. ¿Cuándo el LETF *gana* a L× (tendencias con baja volatilidad)?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from letf_lab.report import Report
from letf_lab.research.common import pair
from letf_lab.universe import UNIVERSE

HORIZONS = {"1w": 5, "1m": 21, "3m": 63, "1y": 252}


def window_stats(p: pd.DataFrame, L: float, h: int) -> pd.DataFrame:
    """Ventanas no solapadas de h días: gap de retorno y varianza realizada."""
    g = np.arange(len(p)) // h
    agg = p.groupby(g).agg(
        start=("u", lambda s: s.index[0]),
        R_letf=("letf", lambda s: (1 + s).prod() - 1),
        R_u=("u", lambda s: (1 + s).prod() - 1),
        var=("u", lambda s: (s**2).sum()),
        n=("u", "size"),
    )
    agg = agg[agg.n == h]
    agg["gap"] = agg.R_letf - L * agg.R_u  # vs expectativa ingenua
    agg["log_gap"] = np.log1p(agg.R_letf) - L * np.log1p(agg.R_u)
    return agg


def run():
    rep = Report("r1_decay", "R1 — Anatomía del decay de volatilidad")
    rep.p(__doc__.split("\n", 2)[2])

    rows, slopes = [], []
    for e in UNIVERSE:
        p = pair(e.ticker)
        for hname, h in HORIZONS.items():
            w = window_stats(p, e.leverage, h)
            if len(w) < 8:
                continue
            rows.append({
                "ticker": e.ticker, "L": e.leverage, "h": hname, "n_win": len(w),
                "P(LETF < L×u)": (w.gap < 0).mean(),
                "gap_mediano": w.gap.median(),
                "gap_p10": w.gap.quantile(0.10),
                "gap_p90": w.gap.quantile(0.90),
            })
        w = window_stats(p, e.leverage, 21)
        # log_gap = a + b·var ; teoría: b = −L(L−1)/2 ; a ≈ financiación + ER (por mes)
        X = np.c_[np.ones(len(w)), w["var"]]
        a, b = np.linalg.lstsq(X, w.log_gap.values, rcond=None)[0]
        resid = w.log_gap - (a + b * w["var"])
        slopes.append({
            "ticker": e.ticker, "L": e.leverage,
            "b_teórica": -e.leverage * (e.leverage - 1) / 2, "b_estimada": b,
            "intercepto_anual": a * 12,
            "R2": 1 - resid.var() / w.log_gap.var(),
        })

    df = pd.DataFrame(rows)
    sl = pd.DataFrame(slopes).set_index("ticker")

    rep.h("1. ¿Con qué frecuencia el LETF rinde menos que L× el subyacente?")
    piv = df.pivot_table(index=["ticker", "L"], columns="h", values="P(LETF < L×u)")
    rep.table(piv[list(HORIZONS)], pct=set(HORIZONS))
    rep.p("A 1 semana el LETF replica casi exactamente; a 1 año la probabilidad de "
          "quedarse por debajo sube, sobre todo en inversos y subyacentes volátiles.")

    rep.h("Distribución del gap (LETF − L×u) a 1 año")
    y = df[df.h == "1y"].set_index("ticker")[["L", "n_win", "gap_p10", "gap_mediano", "gap_p90"]]
    rep.table(y, pct={"gap_p10", "gap_mediano", "gap_p90"})

    rep.h("2. ¿La teoría −L(L−1)/2·σ² explica el gap mensual?")
    rep.table(sl, pct={"intercepto_anual"})
    rep.p("`b_estimada` ≈ `b_teórica` confirma que el decay es *mecánico* y "
          "predecible a partir de la varianza realizada. El intercepto anualizado "
          "captura ER + financiación + spread del swap (ver R2).")

    # figura: TQQQ, gap log mensual vs varianza realizada
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, t in zip(axes, ["TQQQ", "SQQQ"]):
        from letf_lab.universe import BY_TICKER
        L = BY_TICKER[t].leverage
        w = window_stats(pair(t), L, 21)
        ax.scatter(w["var"] * 12, w.log_gap * 12, s=10, alpha=0.6)
        xs = np.linspace(0, (w["var"] * 12).max(), 50)
        ax.plot(xs, -L * (L - 1) / 2 * xs + sl.loc[t, "intercepto_anual"], "r--",
                label=f"teoría b={-L*(L-1)/2:.1f}")
        ax.set(title=f"{t}: gap log mensual (anualiz.) vs varianza realizada",
               xlabel="varianza realizada anualizada del subyacente", ylabel="gap log anualizado")
        ax.legend()
    rep.fig(fig, "tqqq_sqqq_gap_vs_var")

    rep.h("3. Mapa de decay teórico: drag anual por nivel de volatilidad")
    vols = [0.10, 0.15, 0.20, 0.30, 0.45, 0.60]
    Ls = [-3, -2, -1, 2, 3]
    m = pd.DataFrame({f"L={L:+d}": [-L * (L - 1) / 2 * v**2 for v in vols] for L in Ls},
                     index=[f"σ={v:.0%}" for v in vols])
    rep.table(m, pct=set(m.columns))
    rep.p("Implicación: el decay no es 'magia negra' sino un coste cuadrático en σ. "
          "A σ=15% (SPY típico) un 3x pierde ~7%/año vs 3×; a σ=45% (semis, biotech) "
          "pierde ~60%/año → casi imposible de mantener a largo plazo. Un 3x inverso "
          "(L=−3) paga 6σ², el doble que el 3x largo.")
    rep.save()
    return df, sl


if __name__ == "__main__":
    run()
