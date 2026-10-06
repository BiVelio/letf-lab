"""Métricas de rendimiento/riesgo sobre series de retornos diarios."""

from __future__ import annotations

import numpy as np
import pandas as pd

D = 252


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    w = float((1 + r).prod())
    years = len(r) / D
    return w ** (1 / years) - 1 if w > 0 else -1.0


def vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(D))


def sharpe(r: pd.Series, rf: pd.Series | float = 0.0) -> float:
    ex = (r - (rf.reindex(r.index).fillna(0) if isinstance(rf, pd.Series) else rf)).dropna()
    sd = ex.std()
    return float(ex.mean() / sd * np.sqrt(D)) if sd > 0 else np.nan


def sortino(r: pd.Series) -> float:
    down = r[r < 0].std()
    return float(r.mean() / down * np.sqrt(D)) if down > 0 else np.nan


def drawdown(r: pd.Series) -> pd.Series:
    w = (1 + r.fillna(0)).cumprod()
    return w / w.cummax() - 1


def max_dd(r: pd.Series) -> float:
    return float(drawdown(r).min())


def ulcer(r: pd.Series) -> float:
    return float(np.sqrt((drawdown(r) ** 2).mean()))


def longest_underwater(r: pd.Series) -> int:
    """Días hábiles máximos bajo el máximo previo."""
    under = drawdown(r) < 0
    runs = under.groupby((~under).cumsum()).sum()
    return int(runs.max()) if len(runs) else 0


def summary(r: pd.Series, rf: pd.Series | float = 0.0) -> dict:
    r = r.dropna()
    c, m = cagr(r), max_dd(r)
    return {
        "start": r.index[0].date() if len(r) else None,
        "years": round(len(r) / D, 1),
        "CAGR": c,
        "vol": vol(r),
        "sharpe": sharpe(r, rf),
        "sortino": sortino(r),
        "maxDD": m,
        "calmar": c / abs(m) if m < 0 else np.nan,
        "ulcer": ulcer(r),
        "max_underwater_d": longest_underwater(r),
    }


def table(returns: dict[str, pd.Series] | pd.DataFrame, rf=0.0) -> pd.DataFrame:
    items = returns.items() if isinstance(returns, dict) else returns.items()
    return pd.DataFrame({k: summary(v, rf) for k, v in items}).T
