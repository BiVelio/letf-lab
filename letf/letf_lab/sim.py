"""Simulador de ETFs apalancados con reset diario.

Modelo (fondo con NAV=1 que mantiene caja y un swap de nocional L):
    r_letf = L * r_u + (1 - L) * rf - |L| * spread - ER/252

- L > 1: paga financiación sobre (L-1) → término (1-L)*rf negativo.
- L < 0: cobra rf sobre la caja más la del colateral del corto → (1-L)*rf > 0.
- spread: sobrecoste anual del swap sobre rf (se calibra en R2 con datos reales).

Resultado teórico (GBM, rebalanceo continuo):
    g_letf ≈ L*g_u + (L - L²)/2 * σ² + financiación
El término −L(L−1)/2·σ² es el "volatility drag" que se estudia en R1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def letf_returns(r_u: pd.Series, leverage: float | pd.Series, rf: pd.Series | float = 0.0,
                 expense_ratio: float = 0.0095, spread: float = 0.0) -> pd.Series:
    rf = rf if isinstance(rf, (int, float)) else rf.reindex(r_u.index).fillna(0.0)
    r = (leverage * r_u + (1 - leverage) * rf
         - abs(leverage) * spread / TRADING_DAYS - expense_ratio / TRADING_DAYS)
    # un fondo real no puede bajar de -100%: si ocurre, queda en cero para siempre
    r = r.clip(lower=-1.0)
    dead = (r <= -1.0).cummax().shift(1, fill_value=False)
    return r.mask(dead, 0.0).rename("sim")


def theoretical_drag(sigma_annual: float | np.ndarray, leverage: float) -> float | np.ndarray:
    """Drag anual de crecimiento geométrico vs L× (sin financiación): −L(L−1)/2·σ²."""
    return -leverage * (leverage - 1) / 2 * np.asarray(sigma_annual) ** 2


def growth_rate(mu: float, sigma: float, leverage: float, rf: float = 0.0,
                cost: float = 0.0) -> float:
    """Tasa de crecimiento log anual de un LETF bajo GBM (μ aritmético del subyacente)."""
    return rf + leverage * (mu - rf) - cost - 0.5 * leverage**2 * sigma**2


def kelly_leverage(mu: float, sigma: float, rf: float = 0.0) -> float:
    return (mu - rf) / sigma**2


def wealth(r: pd.Series | pd.DataFrame, start: float = 1.0):
    return start * (1 + r.fillna(0.0)).cumprod()


def calibrate_spread(r_letf: pd.Series, r_u: pd.Series, leverage: float,
                     rf: pd.Series, expense_ratio: float) -> float:
    """Spread anual implícito del swap: el residuo medio no explicado por L, rf y ER."""
    df = pd.concat([r_letf, r_u, rf.reindex(r_u.index)], axis=1, keys=["l", "u", "rf"]).dropna()
    base = leverage * df.u + (1 - leverage) * df.rf - expense_ratio / TRADING_DAYS
    return float(-(df.l - base).mean() * TRADING_DAYS / abs(leverage))
