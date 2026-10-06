"""Fuentes de datos con caché local en parquet.

Prioridad: Interactive Brokers (si hay TWS/IB Gateway escuchando) → yfinance.
Todo lo descargado se guarda en data/cache/ para que los experimentos sean
reproducibles y offline. Para refrescar, borrar el parquet o usar refresh=True.

Convención de columnas (diario): open, high, low, close, adj_close, volume.
`adj_close` incluye dividendos/distribuciones (retorno total).
"""

from __future__ import annotations

import logging
import os
import socket
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("LETF_CACHE", ROOT / "data" / "cache"))
CACHE.mkdir(parents=True, exist_ok=True)

IB_HOST = os.environ.get("IB_HOST", "127.0.0.1")
IB_PORTS = [int(p) for p in os.environ.get("IB_PORTS", "4001,7496,4002,7497").split(",")]
IB_CLIENT_ID = int(os.environ.get("IB_CLIENT_ID", "37"))


def _cache_path(source: str, ticker: str, interval: str) -> Path:
    safe = ticker.replace("^", "IDX_").replace("/", "_")
    return CACHE / f"{source}_{safe}_{interval}.parquet"


def ib_port() -> int | None:
    """Devuelve el primer puerto IB abierto, o None."""
    for port in IB_PORTS:
        with socket.socket() as s:
            s.settimeout(0.3)
            if s.connect_ex((IB_HOST, port)) == 0:
                return port
    return None


# ----------------------------------------------------------------- yfinance
def _yf_daily(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(
        tickers, period="max", interval="1d", auto_adjust=False,
        actions=False, progress=False, group_by="ticker", threads=True,
    )
    out = {}
    for t in tickers:
        df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        df = df.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
        df = df.dropna(subset=["close"])
        if df.empty:
            log.warning("yfinance sin datos para %s", t)
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        out[t] = df[["open", "high", "low", "close", "adj_close", "volume"]].astype(float)
    return out


def _yf_intraday(ticker: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    # Límites de Yahoo: 1m→7d, 5m/15m/30m→60d, 1h→730d.
    period = {"1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d"}[interval]
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                     progress=False, prepost=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    df.index = pd.to_datetime(df.index).tz_convert("America/New_York")
    df.index.name = "ts"
    return df[["open", "high", "low", "close", "volume"]].astype(float)


# ----------------------------------------------------------------------- IB
def _ib_contract(ticker: str):
    from ib_async import Index, Stock

    idx = {"^SPX": ("SPX", "CBOE"), "^NDX": ("NDX", "NASDAQ"), "^RUT": ("RUT", "RUSSELL"),
           "^VIX": ("VIX", "CBOE")}
    if ticker in idx:
        sym, exch = idx[ticker]
        return Index(sym, exch, "USD")
    return Stock(ticker, "SMART", "USD")


def _ib_connect():
    from ib_async import IB

    port = ib_port()
    if port is None:
        raise ConnectionError("No hay TWS/IB Gateway escuchando en " + str(IB_PORTS))
    ib = IB()
    ib.connect(IB_HOST, port, clientId=IB_CLIENT_ID, readonly=True, timeout=10)
    return ib


def _ib_daily(ticker: str, years: int = 30) -> pd.DataFrame:
    """Barras diarias ADJUSTED_LAST (ajustadas por splits y dividendos)."""
    from ib_async import util

    ib = _ib_connect()
    try:
        c = _ib_contract(ticker)
        ib.qualifyContracts(c)
        what = "TRADES" if c.secType == "IND" else "ADJUSTED_LAST"
        bars = ib.reqHistoricalData(c, endDateTime="", durationStr=f"{years} Y",
                                    barSizeSetting="1 day", whatToShow=what,
                                    useRTH=True, formatDate=1)
        df = util.df(bars)
    finally:
        ib.disconnect()
    if df is None or df.empty:
        raise ValueError(f"IB sin datos para {ticker}")
    df = df.set_index(pd.to_datetime(df["date"])).rename_axis("date")
    df["adj_close"] = df["close"]  # ADJUSTED_LAST ya viene ajustado
    return df[["open", "high", "low", "close", "adj_close", "volume"]].astype(float)


def ib_intraday(ticker: str, bar: str = "1 min", days: int = 120,
                what: str = "TRADES") -> pd.DataFrame:
    """Barras intradía de IB, paginando hacia atrás (1 min: ~1 día por request)."""
    from ib_async import util

    ib = _ib_connect()
    frames, end = [], ""
    chunk = {"1 min": "1 D", "5 mins": "1 W", "30 mins": "1 M", "1 hour": "1 M"}[bar]
    step = {"1 D": 1, "1 W": 5, "1 M": 21}[chunk]
    try:
        c = _ib_contract(ticker)
        ib.qualifyContracts(c)
        got = 0
        while got < days:
            bars = ib.reqHistoricalData(c, endDateTime=end, durationStr=chunk,
                                        barSizeSetting=bar, whatToShow=what,
                                        useRTH=True, formatDate=2)
            if not bars:
                break
            frames.append(util.df(bars))
            end = bars[0].date
            got += step
            ib.sleep(0.5)  # respetar pacing de IB
    finally:
        ib.disconnect()
    df = pd.concat(frames).drop_duplicates("date").set_index("date").sort_index()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    df.index.name = "ts"
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def ib_quotes(tickers: list[str]) -> pd.DataFrame:
    """Snapshot bid/ask/último + NAV intradía cuando IB lo provee (liquidez real)."""
    ib = _ib_connect()
    rows = []
    try:
        contracts = [_ib_contract(t) for t in tickers]
        ib.qualifyContracts(*contracts)
        tks = [ib.reqMktData(c, "", snapshot=True) for c in contracts]
        ib.sleep(3)
        for t, tk in zip(tickers, tks):
            mid = (tk.bid + tk.ask) / 2 if tk.bid and tk.ask else float("nan")
            rows.append({"ticker": t, "bid": tk.bid, "ask": tk.ask, "last": tk.last,
                         "bid_size": tk.bidSize, "ask_size": tk.askSize,
                         "spread_bps": (tk.ask - tk.bid) / mid * 1e4 if mid == mid else None})
    finally:
        ib.disconnect()
    return pd.DataFrame(rows).set_index("ticker")


# ------------------------------------------------------------------ público
def daily(tickers: list[str] | str, refresh: bool = False,
          source: str = "auto") -> dict[str, pd.DataFrame]:
    """OHLCV diario por ticker, con caché. source: auto | ib | yf."""
    if isinstance(tickers, str):
        tickers = [tickers]
    out, missing = {}, []
    for t in tickers:
        hits = [_cache_path(s, t, "1d") for s in ("ib", "yf")]
        hit = next((p for p in hits if p.exists()), None)
        if hit is not None and not refresh:
            out[t] = pd.read_parquet(hit)
        else:
            missing.append(t)
    if not missing:
        return out

    use_ib = source == "ib" or (source == "auto" and ib_port() is not None)
    if use_ib:
        for t in list(missing):
            try:
                df = _ib_daily(t)
                df.to_parquet(_cache_path("ib", t, "1d"))
                out[t] = df
                missing.remove(t)
            except Exception as e:  # noqa: BLE001 — caemos a yfinance
                log.warning("IB falló para %s (%s); uso yfinance", t, e)
    if missing and source != "ib":
        for t, df in _yf_daily(missing).items():
            df.to_parquet(_cache_path("yf", t, "1d"))
            out[t] = df
    return out


def intraday(ticker: str, interval: str = "1h", refresh: bool = False) -> pd.DataFrame:
    """Barras intradía (RTH). Con IB se obtiene mucha más historia que con Yahoo."""
    ib_bar = {"1m": "1 min", "5m": "5 mins", "30m": "30 mins", "1h": "1 hour"}[interval]
    for src in ("ib", "yf"):
        p = _cache_path(src, ticker, interval)
        if p.exists() and not refresh:
            return pd.read_parquet(p)
    if ib_port() is not None:
        df = ib_intraday(ticker, ib_bar, days=500 if interval != "1m" else 250)
        df.to_parquet(_cache_path("ib", ticker, interval))
        return df
    df = _yf_intraday(ticker, interval)
    df.to_parquet(_cache_path("yf", ticker, interval))
    return df


def panel(tickers: list[str], field: str = "adj_close", **kw) -> pd.DataFrame:
    d = daily(tickers, **kw)
    return pd.DataFrame({t: d[t][field] for t in tickers if t in d}).sort_index()


def risk_free_daily(index: pd.DatetimeIndex | None = None) -> pd.Series:
    """Tasa libre de riesgo diaria (fracción) desde ^IRX, forward-filled."""
    irx = daily("^IRX")["^IRX"]["close"] / 100.0
    rf = (irx / 252.0).rename("rf")
    if index is not None:
        rf = rf.reindex(rf.index.union(index)).ffill().reindex(index).fillna(0.0)
    return rf
