"""Motor de backtest vectorizado, sin look-ahead.

Regla de oro: los pesos decididos con información hasta el cierre de t se
aplican al retorno de t+1 (`weights.shift(1)`). Los costes se cobran sobre el
turnover en el momento del rebalanceo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Result:
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    meta: dict = field(default_factory=dict)


def run(weights: pd.DataFrame, asset_returns: pd.DataFrame, cost_bps: float = 5.0,
        cash_return: pd.Series | None = None, borrow_bps: pd.Series | float = 0.0) -> Result:
    """
    weights: pesos objetivo decididos al cierre de cada día (index=fecha).
    asset_returns: retornos diarios de los activos (mismas columnas).
    cash_return: retorno de la caja no invertida (1 - suma de pesos, puede ser <0).
    borrow_bps: coste anual de préstamo para posiciones cortas (bps), escalar o por activo.
    """
    asset_returns = asset_returns.reindex(columns=weights.columns)
    w = weights.reindex(asset_returns.index).ffill().fillna(0.0)
    held = w.shift(1).fillna(0.0)  # posición vigente durante el día t
    gross = (held * asset_returns.fillna(0.0)).sum(axis=1)

    if cash_return is not None:
        cash_w = 1.0 - held.sum(axis=1)
        gross = gross + cash_w * cash_return.reindex(gross.index).fillna(0.0)

    if isinstance(borrow_bps, pd.Series):
        borrow = borrow_bps.reindex(held.columns).fillna(0.0)
    else:
        borrow = pd.Series(borrow_bps, index=held.columns)
    short_cost = (held.clip(upper=0).abs() * borrow / 1e4 / 252).sum(axis=1)

    # deriva de pesos intradía → turnover real al volver al objetivo
    drifted = held * (1 + asset_returns.fillna(0.0))
    drifted = drifted.div((1 + gross).replace(0, pd.NA), axis=0).fillna(0.0)
    turnover = (w - drifted).abs().sum(axis=1)
    costs = turnover * cost_bps / 1e4

    net = (gross - short_cost - costs).rename("strategy")
    return Result(net, w, turnover, {"cost_bps": cost_bps})
