"""R6 — ¿Se puede domar un 3x con filtros de tendencia y volatilidad?

Hipótesis: el decay es ∝ σ² y los crashes ocurren con σ alta y precio bajo su
media → estar fuera cuando el subyacente está bajo su SMA o con σ alta debería
evitar la ruina conservando buena parte del crecimiento.

Protocolo anti-sobreajuste:
- In-sample (IS): historia *sintética* hasta 2009 (incluye 1987, 2000-02, 2008).
- Out-of-sample (OOS): 2010 → hoy con precios *reales* del LETF.
- Se muestra la rejilla completa de parámetros, no el mejor punto.
- Señal al cierre de t, ejecución al cierre (MOC) → retorno de t+1; y un test
  con un día extra de retraso para medir la dependencia de la ejecución.
- Costes: 5 bps por unidad de turnover. Caja remunerada a T-bill.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from letf_lab import backtest, metrics
from letf_lab.report import Report
from letf_lab.research.common import long_underlying, rf
from letf_lab.research.r3_synthetic import synthetic_actual
from letf_lab.universe import BY_TICKER

SPLIT = "2010-01-01"
COST_BPS = 5.0


def signals(u: pd.Series, L: float, sma: int | None, vol_target: float | None,
            band: float = 0.0, vol_win: int = 20, min_step: float = 0.10) -> pd.Series:
    """Peso en el LETF (0..1) decidido al cierre de cada día."""
    px = (1 + u.fillna(0)).cumprod()
    w = pd.Series(1.0, index=u.index)
    if sma:
        ma = px.rolling(sma).mean()
        up, down = px > ma * (1 + band), px < ma * (1 - band)
        state = pd.Series(np.where(up, 1.0, np.where(down, 0.0, np.nan)), index=u.index)
        w = w * state.ffill().fillna(0.0)
    if vol_target:
        sig = u.rolling(vol_win).std() * np.sqrt(252)
        vt = (vol_target / (abs(L) * sig)).clip(upper=1.0).fillna(0.0)
        # sólo mover el peso si cambia > min_step (reduce turnover)
        vals, cur = [], 0.0
        for v in vt.values:
            if abs(v - cur) > min_step or v in (0.0, 1.0):
                cur = v
            vals.append(cur)
        w = w * pd.Series(vals, index=u.index)
    return w


def evaluate(letf: str, sma, vt, band=0.0, lag=0) -> dict[str, pd.Series]:
    u = long_underlying(letf)
    r = synthetic_actual(letf).reindex(u.index)
    w = signals(u, BY_TICKER[letf].leverage, sma, vt, band).shift(lag).fillna(0.0)
    res = backtest.run(w.to_frame(letf), r.to_frame(letf), COST_BPS, cash_return=rf(u.index))
    return {"strategy": res.returns, "turnover": res.turnover, "weight": w}


def split_stats(r: pd.Series) -> dict:
    """Métricas IS/OOS; Sharpe en exceso de T-bill."""
    r = r.dropna()
    a, b = r[r.index < SPLIT], r[r.index >= SPLIT]
    f = rf(r.index)
    return {"CAGR_IS": metrics.cagr(a), "maxDD_IS": metrics.max_dd(a), "Sharpe_IS": metrics.sharpe(a, f),
            "CAGR_OOS": metrics.cagr(b), "maxDD_OOS": metrics.max_dd(b), "Sharpe_OOS": metrics.sharpe(b, f)}


PCT = {"CAGR_IS", "maxDD_IS", "CAGR_OOS", "maxDD_OOS", "tiempo_invertido"}


def run():
    rep = Report("r6_regime", "R6 — Filtros de tendencia y volatilidad sobre LETFs")
    rep.p(__doc__.split("\n", 2)[2])

    for letf in ["TQQQ", "UPRO"]:
        e = BY_TICKER[letf]
        u = long_underlying(letf)
        rows = {f"{e.underlying} 1x B&H": split_stats(u),
                f"{letf} B&H": split_stats(synthetic_actual(letf).reindex(u.index))}
        a_priori = {"SMA200": (200, None, 0.0), "SMA200 ±2%": (200, None, 0.02),
                    "VolTarget 30%": (None, 0.30, 0.0), "SMA200 + VT30%": (200, 0.30, 0.0)}
        curves = {}
        for name, (sma, vt, band) in a_priori.items():
            ev = evaluate(letf, sma, vt, band)
            curves[name] = ev["strategy"]
            rows[f"{letf} {name}"] = split_stats(ev["strategy"]) | {
                "turnover_anual": ev["turnover"].mean() * 252,
                "tiempo_invertido": ev["weight"].mean()}
            lagged = evaluate(letf, sma, vt, band, lag=1)["strategy"]
            rows[f"{letf} {name} (+1d retraso)"] = split_stats(lagged)
        rep.h(f"{letf}: configuraciones fijadas a priori")
        rep.table(pd.DataFrame(rows).T, pct=PCT)

        # rejilla de robustez
        grid = []
        for sma, vt in itertools.product([None, 50, 100, 150, 200, 250], [None, 0.2, 0.3, 0.4, 0.5]):
            if sma is None and vt is None:
                continue
            s = split_stats(evaluate(letf, sma, vt)["strategy"])
            grid.append({"SMA": sma or "—", "VT": vt or "—", **s})
        g = pd.DataFrame(grid)
        rep.h(f"{letf}: rejilla completa (SMA × vol target)")
        rep.table(g.set_index(["SMA", "VT"]), pct=PCT)
        rep.p(f"Correlación de rango Sharpe IS vs OOS en la rejilla: "
              f"**{g.Sharpe_IS.rank().corr(g.Sharpe_OOS.rank()):.2f}** "
              "(alta ⇒ lo que funcionaba antes de 2010 siguió funcionando; baja ⇒ ruido).")

        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.semilogy((1 + u).cumprod(), color="k", label=f"{e.underlying} 1x")
        ax.semilogy((1 + synthetic_actual(letf).reindex(u.index).fillna(0)).cumprod(),
                    color="0.6", label=f"{letf} B&H")
        for name, c in curves.items():
            ax.semilogy((1 + c).cumprod(), label=name)
        ax.axvline(pd.Timestamp(SPLIT), color="r", ls="--", lw=1)
        ax.text(pd.Timestamp(SPLIT), ax.get_ylim()[1], " OOS real →", color="r", va="top")
        ax.set(title=f"{letf}: riqueza (log), sintético hasta 2009, real después")
        ax.legend(ncol=3, fontsize=8)
        rep.fig(fig, f"{letf.lower()}_curves")

    rep.save()


if __name__ == "__main__":
    run()
