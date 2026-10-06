"""Datos de volatilidad del artículo 4: futuros VX de Cboe, índices de Cboe, índices S&P de
futuros del VIX reconstruidos, productos sintéticos de reajuste diario, ETPs de Yahoo y el
coste de préstamo de IBKR.

Todo se descarga una vez a data/cache/ (sources.CACHE) y no se vuelve a pedir salvo con
refresh=True; nada de aquí reescribe ni borra ficheros de caché que ya existían.

Fuentes (públicas, sin registro):
- Futuros VX de CFE, liquidación diaria por contrato mensual:
  * archivo antiguo (2004-03-26 → 2013-06): ARCHIVE_URL, un CSV por contrato;
  * ficheros actuales (2013-01-02 →): CDN_URL, un CSV por fecha de vencimiento.
  Arreglos conocidos: (1) antes del 26-3-2007 el archivo cotiza en escala VXB (10 × VIX) y
  se divide por 10; (2) en los ficheros actuales el settle vale 0 entre el 2-1-2013 y el
  17-5-2013: se toma el settle del archivo antiguo cuando existe (contratos F13–M13) y, si
  no, el cierre (px_src lo marca en cada fila).
- Índices de Cboe (VIX, VIX3M, VIX9D, VIX6M, VVIX, SHORTVOL, LONGVOL): CBOE_IDX_URL. El VIX
  de Yahoo (^VIX) tiene cierres erróneos (el 6-2-2026: 20,37 frente a 17,76 en Cboe); aquí
  el VIX es siempre el de Cboe. SHORTVOL y LONGVOL nacen el 22-11-2019 y el 8-10-2021
  (folleto de VS Trust, 28-4-2023); lo anterior es un recálculo hacia atrás de Cboe.
- ETPs y ETFs: Yahoo a través de sources.daily(source="yf") (mismo formato de caché del
  laboratorio: yf_<TICKER>_1d.parquet; los que ya estaban no se tocan).
- Préstamo de títulos: iBorrowDesk (BORROW_URL), que publica ~12 meses de la tasa diaria de
  Interactive Brokers; JSON en bruto en cache/ibkr_borrow/.
- Metodología S&P de los índices de futuros del VIX (copia de octubre de 2021 alojada por
  Global X; la web de S&P DJI devuelve 403 a los scripts):
  https://www.globalx.ca/wp-content/uploads/2024/08/VIX-Index-Methodology.pdf

Índices reconstruidos (futures_indices), metodología S&P:
  corto (ST): contratos 1.º y 2.º, CRW_1 = dr/dt y CRW_2 = (dt − dr)/dt;
  medio (MT): contratos 4.º–7.º, CRW_4 = dr/dt, CRW_5 = CRW_6 = 1, CRW_7 = (dt − dr)/dt;
  dt = sesiones del periodo de roll [vencimiento inicial, vencimiento siguiente); dr =
  sesiones desde la siguiente sesión hasta el vencimiento siguiente (excluido);
  CDR_t = Σ CRW_{i,t−1}·P_{i,t} / Σ CRW_{i,t−1}·P_{i,t−1} − 1 (ER) y
  TR_t = TR_{t−1}·(1 + CDR_t + TBR_t), TBR_t = [1/(1 − 91/360·TBAR_{t−1})]^(Δd/91) − 1 con
  TBAR = ^IRX (tasa de descuento de la letra a 13 semanas). Meses no listados (antes de
  2008): interpolación en varianza entre los contratos listados vecinos y extrapolación en
  varianza más allá del último («Historical Assumptions» de la metodología).
  Base 100.000 el 20-12-2005 (fecha base de S&P) en los dos.

Convenciones de hora (pre-registro): «settle» es la liquidación diaria de CFE, a las 16:15
ET hasta el cambio de la Rule Filing CFE 2020-028 (efectiva «el 7-10-2020 o después»; la
fecha exacta está sin verificar) y a las 16:00 ET después. «1600» es el TWAP de 15:45–16:00
ET de los índices SHORTVOL (−1x) y LONGVOL (+1x) de Cboe: como los dos cobran la misma
letra, el rendimiento a las 16:00 del índice corto es (r_LONGVOL − r_SHORTVOL)/2 y la
letra implícita de Cboe (r_LONGVOL + r_SHORTVOL)/2 (er_1600). No hay equivalente a las
16:00 del índice medio.

Productos sintéticos de reajuste diario (synthetic): L·ER + letra − gasto·Δd/365, con la
letra del laboratorio (^IRX/252 por sesión, sources.risk_free_daily) y el gasto por días
naturales; VYLD: −Δpuntos/100 + letra − gasto (el índice «points-change inverse» baja un 1%
por cada punto que sube el precio ponderado de los futuros). Un día de −100% o peor se
recorta a −100%.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from letf_lab.data import sources

log = logging.getLogger(__name__)

# ───────────────────────────── rutas y fuentes ─────────────────────────────

CACHE = sources.CACHE
VX_DIR = CACHE / "cboe_vx"               # archive/CFE_<c><aa>_VX.csv y cdn/VX_<vencimiento>.csv
IDX_DIR = CACHE / "cboe_idx"             # <NOMBRE>_History.csv
BORROW_DIR = CACHE / "ibkr_borrow"       # ibd_<TICKER>.json
SP_QUOTES = CACHE / "sp_vix_indices_quotes.csv"

ARCHIVE_URL = "https://cdn.cboe.com/resources/futures/archive/volume-and-price/CFE_{c}{yy}_VX.csv"
CDN_URL = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{d}.csv"
CBOE_IDX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
BORROW_URL = "https://www.iborrowdesk.com/api/ticker/{t}"
UA = {"User-Agent": "Mozilla/5.0 (letf-lab; research)"}
CODES = "FGHJKMNQUVXZ"
MONTHS = {m: i + 1 for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                                           "Sep", "Oct", "Nov", "Dec"))}

# ───────────────────────────── fechas del pre-registro ─────────────────────────────

END = pd.Timestamp("2026-09-25")            # último día de todas las ventanas
ST_START = pd.Timestamp("2005-12-20")       # base del índice corto (primer rendimiento: el día siguiente)
MT_START = pd.Timestamp("2006-04-24")       # primer día con 7 contratos listados (índice medio)
REAL_START = pd.Timestamp("2011-01-04")     # primer cierre de VIXY y VIXM (ventana de productos reales)
DESIGN_2018 = pd.Timestamp("2018-02-28")    # primer día con SVXY −0,5x y UVXY 1,5x (cambio al cierre del 27-2)
SETTLE_CHANGE = pd.Timestamp("2020-10-07")  # liquidación diaria de VX: 16:15 → 16:00 ET («el 7-10-2020 o después»)
FEB_2018 = (pd.Timestamp("2018-02-05"), pd.Timestamp("2018-02-06"))
VXB_UNTIL = pd.Timestamp("2007-03-26")      # antes, el archivo de CFE cotiza 10 × VIX
ZERO_SETTLE = (pd.Timestamp("2013-01-02"), pd.Timestamp("2013-05-17"))

# Festivos de NYSE/CFE después del último dato (solo para contar sesiones hasta el vencimiento
# siguiente cuando el índice se calcula hasta el final de los datos; regla estándar, sin verificar).
FUTURE_HOLIDAYS = ("2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15",
                   "2027-03-26", "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06",
                   "2027-11-25", "2027-12-24")

# ───────────────────────────── universo ─────────────────────────────

VIX_ETPS = ("SVIX", "SVXY", "UVIX", "UVXY", "VIXM", "VIXY", "VXX", "VXZ", "VYLD")
EQUITY_ETFS = ("SQQQ", "TQQQ", "SPXU", "UPRO", "QQQ", "SPY")
CBOE_INDICES = ("VIX", "VIX3M", "VIX9D", "VIX6M", "VVIX", "SHORTVOL", "LONGVOL")
BORROW_TICKERS = VIX_ETPS + ("SQQQ", "SPXU", "TQQQ", "SDS", "QID", "SOXL", "SOXS")
# Rejilla de préstamo anual del pre-registro: 0 y el mínimo, la mediana y el p90 observados en IBKR
BORROW_GRID = (0.0, 0.02, 0.052, 0.18)
PROCEEDS_SPREAD = 0.005                     # producto del corto remunerado a la letra − 0,5% (o a 0)
FINRA_MAINT = 0.30                          # mantenimiento 30% × |apalancamiento| (tope 100%)


@dataclass(frozen=True)
class ETP:
    """Producto real y su modelo diario sobre el índice reconstruido."""
    ticker: str
    index: str                 # "st" (corto) | "mt" (medio)
    kind: str                  # "fondo": L·ER + letra − gasto | "etn": TR − gasto | "puntos": VYLD
    leverage: float            # apalancamiento actual
    fee: float                 # gasto anual del modelo
    convention: str            # hora de valoración de su índice: "settle" (S&P) | "1600" (Cboe)
    leverage_before: float | None = None   # antes de DESIGN_2018 (SVXY −1x, UVXY 2x)
    note: str = ""


ETP_SPECS = {
    "VIXY": ETP("VIXY", "st", "fondo", 1.0, 0.0085, "settle", note="ProShares; ER corto 1x"),
    "VIXM": ETP("VIXM", "mt", "fondo", 1.0, 0.0085, "settle", note="ProShares; ER medio 1x"),
    "VXX": ETP("VXX", "st", "etn", 1.0, 0.0089, "settle", note="ETN de Barclays (serie B, 2018-01); TR corto"),
    "VXZ": ETP("VXZ", "mt", "etn", 1.0, 0.0089, "settle", note="ETN de Barclays (2018-01); TR medio"),
    "SVXY": ETP("SVXY", "st", "fondo", -0.5, 0.0095, "settle", -1.0, "ProShares; −1x hasta el 27-2-2018"),
    "UVXY": ETP("UVXY", "st", "fondo", 1.5, 0.0095, "settle", 2.0, "ProShares; 2x hasta el 27-2-2018"),
    "SVIX": ETP("SVIX", "st", "fondo", -1.0, 0.0135, "1600", note="VS Trust; −1x SHORTVOL (comisión de gestión 1,35%)"),
    "UVIX": ETP("UVIX", "st", "fondo", 2.0, 0.0165, "1600_long", note="VS Trust; 2x LONGVOL (comisión de gestión 1,65%)"),
    "VYLD": ETP("VYLD", "st", "puntos", -1.0, 0.0085, "settle", note="ETN de JPMorgan; −1% por punto (TR)"),
}

# Sintéticos de reajuste diario: clave → (apalancamiento o "puntos", índice, gasto, análogo)
SYNTHETIC = {
    "-1x": (-1.0, "st", 0.0095, "SVXY hasta el 27-2-2018 (XIV cobraba 1,35%)"),
    "-0.5x": (-0.5, "st", 0.0095, "SVXY desde el 28-2-2018"),
    "+1x": (1.0, "st", 0.0085, "VIXY"),
    "+1.5x": (1.5, "st", 0.0095, "UVXY desde el 28-2-2018"),
    "+2x": (2.0, "st", 0.0095, "UVXY hasta el 27-2-2018 (UVIX cobra 1,65%)"),
    "vyld": ("puntos", "st", 0.0085, "VYLD"),
    "+1x mt": (1.0, "mt", 0.0085, "VIXM"),
}
LABELS = {"-1x": "−1x", "-0.5x": "−0,5x", "+1x": "+1x", "+1.5x": "+1,5x", "+2x": "+2x",
          "vyld": "VYLD (puntos)", "+1x mt": "+1x medio plazo"}
CONVENTIONS = {"settle": "liquidación de CFE (16:15 ET hasta 10-2020, 16:00 después)",
               "1600": "TWAP 15:45–16:00 ET (SHORTVOL/LONGVOL de Cboe)"}


# ───────────────────────────── descargas ─────────────────────────────

def _get(url: str, timeout: int = 30, tries: int = 3) -> bytes | None:
    """GET con reintentos; None si el servidor dice que no existe (403/404)."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(1 + k)
        except Exception:  # noqa: BLE001 — red intermitente: reintentar
            time.sleep(1 + k)
    return None


def _third_friday(y: int, m: int) -> dt.date:
    d = dt.date(y, m, 15)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def _is_session(d: pd.Timestamp, sessions: pd.DatetimeIndex | None) -> bool:
    """¿Es d una sesión de CFE? Dentro del rango de `sessions`, si está en ella; fuera (o sin
    sessions), si es día hábil y no está en FUTURE_HOLIDAYS."""
    if sessions is not None and len(sessions) and sessions[0] <= d <= sessions[-1]:
        return d in sessions
    return d.weekday() < 5 and d.strftime("%Y-%m-%d") not in FUTURE_HOLIDAYS


def _previous_session(d: pd.Timestamp, sessions: pd.DatetimeIndex | None) -> pd.Timestamp:
    d = d - pd.Timedelta(days=1)
    while not _is_session(d, sessions):
        d = d - pd.Timedelta(days=1)
    return d


def rule_expiry(y: int, m: int, sessions: pd.DatetimeIndex | None = None) -> pd.Timestamp:
    """Vencimiento (liquidación final) del VX del mes (y, m) por la regla de CFE: el miércoles
    30 días antes del tercer viernes del mes siguiente.

    Con `sessions` (las sesiones de CFE), también las reglas de festivos de CFE: si ese tercer
    viernes es festivo, 30 días antes de la sesión anterior a ese viernes; si el día resultante
    es festivo, la sesión anterior. Así se reproducen los vencimientos observados en los
    ficheros actuales (p. ej. 18-3-2014 por el Viernes Santo del 18-4 y 18-6-2024 por el festivo
    del 19-6) y, antes de 2013, el de febrero de 2008: el 21-3-2008 fue Viernes Santo y el
    contrato liquidó el martes 19-2-2008 (el último precio del archivo es de ese día; la fecha
    no está comprobada contra un calendario publicado por CFE). Sin `sessions`, la regla simple
    (la usa download_vx, que prueba también las fechas vecinas).

    En 2004 algunos contratos (julio y octubre) vencieron el miércoles anterior al tercer viernes
    de su propio mes, según los datos del archivo (regla antigua sin verificar); solo afecta a
    fechas anteriores a ST_START."""
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    fri = pd.Timestamp(_third_friday(ny, nm))
    if sessions is None:
        return fri - pd.Timedelta(days=30)
    if not _is_session(fri, sessions):
        fri = _previous_session(fri, sessions)
    d = fri - pd.Timedelta(days=30)
    return d if _is_session(d, sessions) else _previous_session(d, sessions)


def _manifest_row(path, url: str) -> dict:
    b = path.read_bytes()
    return {"fichero": str(path.relative_to(CACHE)), "url": url, "bytes": len(b),
            "sha256": hashlib.sha256(b).hexdigest()[:16],
            "descargado": dt.datetime.now().isoformat(timespec="seconds")}


def download_vx(refresh: bool = False, last_year: int | None = None) -> pd.DataFrame:
    """Descarga los CSV de futuros VX a cache/cboe_vx/ (una vez). Devuelve el manifiesto
    (fichero, url, bytes, huella y hora de descarga)."""
    man_path = VX_DIR / "manifest.csv"
    if man_path.exists() and not refresh:
        return pd.read_csv(man_path)
    (VX_DIR / "archive").mkdir(parents=True, exist_ok=True)
    (VX_DIR / "cdn").mkdir(parents=True, exist_ok=True)
    last_year = last_year or dt.date.today().year + 1
    rows = []
    for y in range(2004, 2015):                       # el archivo antiguo acaba en 2013-06
        for m in range(1, 13):
            c, yy = CODES[m - 1], f"{y % 100:02d}"
            f = VX_DIR / "archive" / f"CFE_{c}{yy}_VX.csv"
            url = ARCHIVE_URL.format(c=c, yy=yy)
            if not f.exists() or refresh:
                raw = _get(url)
                if not raw or b"Trade Date" not in raw[:400]:
                    continue
                f.write_bytes(raw)
            rows.append(_manifest_row(f, url))
    for y in range(2013, last_year + 1):
        for m in range(1, 13):
            w = rule_expiry(y, m)
            cands = (w, w - pd.Timedelta(days=1), w - pd.Timedelta(days=2), w + pd.Timedelta(days=1))
            have = [d for d in cands if (VX_DIR / "cdn" / f"VX_{d.date()}.csv").exists()]
            if have and not refresh:
                rows.append(_manifest_row(VX_DIR / "cdn" / f"VX_{have[0].date()}.csv", CDN_URL.format(d=have[0].date())))
                continue
            for d in cands:
                f = VX_DIR / "cdn" / f"VX_{d.date()}.csv"
                url = CDN_URL.format(d=d.date())
                raw = _get(url)
                if raw and raw[:10].startswith(b"Trade Date"):
                    f.write_bytes(raw)
                    rows.append(_manifest_row(f, url))
                    break
    man = pd.DataFrame(rows)
    man.to_csv(man_path, index=False)
    return man


def download_cboe_indices(names=CBOE_INDICES, refresh: bool = False) -> None:
    """CSV públicos de Cboe a cache/cboe_idx/<NOMBRE>_History.csv (una vez)."""
    IDX_DIR.mkdir(parents=True, exist_ok=True)
    for n in names:
        f = IDX_DIR / f"{n}_History.csv"
        if f.exists() and not refresh:
            continue
        raw = _get(CBOE_IDX_URL.format(name=n), timeout=60)
        if raw is None or b"DATE" not in raw[:100]:
            raise ConnectionError(f"Cboe no devolvió {n}")
        f.write_bytes(raw)


def download_borrow(tickers=BORROW_TICKERS, refresh: bool = False) -> None:
    """JSON de iBorrowDesk (tasa diaria de préstamo de IBKR, ~12 meses) a cache/ibkr_borrow/."""
    BORROW_DIR.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        f = BORROW_DIR / f"ibd_{t}.json"
        if f.exists() and not refresh:
            continue
        raw = _get(BORROW_URL.format(t=t))
        if raw is None:
            log.warning("iBorrowDesk sin datos para %s", t)
            continue
        json.loads(raw)                                   # que sea JSON válido
        f.write_bytes(raw)
        time.sleep(0.5)


# ───────────────────────────── panel de futuros VX ─────────────────────────────

_PX = ["open", "high", "low", "close", "settle"]
_COLS = {"Trade Date": "date", "Futures": "futures", "Open": "open", "High": "high", "Low": "low",
         "Close": "close", "Settle": "settle", "Change": "change", "Total Volume": "volume",
         "EFP": "efp", "Open Interest": "oi"}


def _contract_month(label: str) -> str | None:
    """«K (May 04)» o «F (Jan 2013)» → «2004-05» / «2013-01»."""
    m = re.search(r"\(\s*([A-Za-z]{3})\s*(\d{2,4})\s*\)", str(label))
    if not m:
        return None
    y = int(m.group(2))
    y = 2000 + y if y < 100 else y
    return f"{y}-{MONTHS[m.group(1).title()]:02d}"


def _read_vx_csv(path) -> pd.DataFrame:
    lines = path.read_text(errors="replace").splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith("Trade Date"))
    head = [c.strip() for c in lines[k].split(",")]
    # algunas filas del archivo antiguo traen una coma de más al final: se leen con columnas de
    # sobra y se descartan
    names = head + [f"_extra{i}" for i in range(4)]
    df = pd.read_csv(io.StringIO("\n".join(lines[k + 1:])), header=None, names=names,
                     index_col=False, dtype=str)
    df = df[[c for c in _COLS if c in df.columns]].rename(columns=_COLS)
    df = df[df["date"].notna() & (df["date"].str.strip() != "")]
    df["date"] = pd.to_datetime(df["date"].str.strip(), format="mixed")
    for c in _PX + ["change", "volume", "oi"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    download_vx()
    arch, cdn = [], []
    for f in sorted((VX_DIR / "archive").glob("CFE_*_VX.csv")):
        d = _read_vx_csv(f)
        m = re.search(r"CFE_([A-Z])(\d\d)_VX", f.name)
        fallback = f"{2000 + int(m.group(2))}-{CODES.index(m.group(1)) + 1:02d}"
        d["contract"] = d["futures"].map(_contract_month).fillna(fallback)
        d["src"] = "archive"
        arch.append(d)
    for f in sorted((VX_DIR / "cdn").glob("VX_*.csv")):
        d = _read_vx_csv(f)
        exp = pd.Timestamp(f.stem.split("_")[1])
        d["contract"] = d["futures"].map(_contract_month).fillna(f"{exp.year}-{exp.month:02d}")
        d["expiry"] = exp
        d["src"] = "cdn"
        cdn.append(d)
    a, b = pd.concat(arch, ignore_index=True), pd.concat(cdn, ignore_index=True)
    vxb = a["date"] < VXB_UNTIL
    a.loc[vxb, _PX + ["change"]] = a.loc[vxb, _PX + ["change"]] / 10.0
    return a, b


@lru_cache(maxsize=None)
def _raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _load_raw()


@lru_cache(maxsize=None)
def sessions() -> pd.DatetimeIndex:
    """Sesiones de CFE: las fechas con alguna fila en los ficheros de VX (archivo y actuales)."""
    a, b = _raw()
    return pd.DatetimeIndex(sorted(set(a["date"]) | set(b["date"])))


@lru_cache(maxsize=None)
def _panel_all() -> pd.DataFrame:
    a, b = (x.copy() for x in _raw())
    exp_cdn = b.groupby("contract")["expiry"].first()
    ses = sessions()

    def expiry_of(c: str) -> pd.Timestamp:
        if c in exp_cdn.index:
            return exp_cdn[c]
        y, m = map(int, c.split("-"))
        return rule_expiry(y, m, ses)

    a["expiry"] = a["contract"].map(expiry_of)
    both = pd.concat([a, b], ignore_index=True)
    both["expiry"] = both["contract"].map(expiry_of)
    # por (fecha, contrato), el primer precio positivo en este orden (el origen queda en px_src)
    cand = []
    for pri, (src, col) in enumerate((("cdn", "settle"), ("archive", "settle"),
                                      ("cdn", "close"), ("archive", "close"))):
        x = both[(both["src"] == src) & (both[col] > 0)].copy()
        x["px"], x["px_src"], x["_pri"] = x[col], f"{src}_{col}", pri
        cand.append(x)
    p = (pd.concat(cand).sort_values(["date", "contract", "_pri"])
         .drop_duplicates(["date", "contract"], keep="first"))
    p = p[p["date"] <= p["expiry"]]
    cols = ["date", "contract", "expiry", "open", "high", "low", "close", "settle", "px", "px_src",
            "volume", "oi", "src"]
    return p[cols].sort_values(["date", "expiry"]).reset_index(drop=True)


def vx_panel(end: pd.Timestamp | None = END) -> pd.DataFrame:
    """Panel largo de futuros VX mensuales: date, contract («AAAA-MM»), expiry (liquidación
    final), open…settle (ya en escala VIX), px (precio usado: settle y, si falta, cierre),
    px_src (cdn_settle | archive_settle | cdn_close | archive_close), volume, oi, src."""
    p = _panel_all()
    return (p[p["date"] <= end] if end is not None else p).copy()


def vx_matrix(end: pd.Timestamp | None = END, field: str = "px") -> pd.DataFrame:
    """Matriz fecha × vencimiento de `field` (px por defecto)."""
    p = vx_panel(end)
    return p.pivot_table(index="date", columns="expiry", values=field, aggfunc="first").sort_index()


def expiry_schedule() -> pd.DatetimeIndex:
    """Vencimientos mensuales 2004–2027: los observados (ficheros actuales) y la regla de CFE con
    sus festivos (rule_expiry con las sesiones de CFE) para los meses sin fichero actual
    (incluidos los meses que no llegaron a listarse)."""
    obs = {(e.year, e.month): e for e in pd.DatetimeIndex(_panel_all()["expiry"].unique())}
    ses = sessions()
    return pd.DatetimeIndex(sorted({obs.get((y, m), rule_expiry(y, m, ses))
                                    for y in range(2004, 2028) for m in range(1, 13)}))


# ───────────────────────────── índices S&P reconstruidos ─────────────────────────────

def _calendar() -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Sesiones de CFE con algún precio y, detrás, días hábiles futuros (sin FUTURE_HOLIDAYS)
    hasta 2028: sirven para contar sesiones hasta un vencimiento posterior al último dato."""
    cal = pd.DatetimeIndex(sorted(_panel_all()["date"].unique()))
    fut = pd.bdate_range(cal[-1] + pd.Timedelta(days=1), "2028-12-31")
    fut = fut[~fut.isin(pd.to_datetime(FUTURE_HOLIDAYS))]
    return cal, cal.append(fut).values


def sp_index_arrays(P: np.ndarray, cal: pd.DatetimeIndex, sched: pd.DatetimeIndex,
                    ext: np.ndarray, CL: np.ndarray | None = None) -> pd.DataFrame:
    """Núcleo de la metodología S&P sobre una matriz de precios (función pura, sin caché).

    P: sesiones × vencimientos (sched, ordenados) con el precio de liquidación o NaN si el
    contrato no cotiza; cal: las sesiones de P; ext: cal seguido de las sesiones futuras
    (para contar sesiones hasta un vencimiento posterior al último dato); CL: True donde el
    precio es un cierre. Devuelve st_ret, st_pts, st_prev, st_cm, mt_ret, mt_cm y las marcas
    st_interp, mt_interp, st_close, mt_close (véase futures_indices)."""
    S = np.asarray(sched.values)
    ext = np.asarray(ext)
    CL = np.zeros(P.shape, bool) if CL is None else CL
    n = len(cal)

    def bdays(a, b) -> int:
        """Sesiones en [a, b)."""
        return int(np.searchsorted(ext, np.datetime64(b)) - np.searchsorted(ext, np.datetime64(a)))

    def price(i: int, j: int) -> tuple[float, bool]:
        if j >= P.shape[1]:                               # más allá del calendario de vencimientos
            return np.nan, True
        v = P[i, j]
        if np.isfinite(v):
            return float(v), False
        row = P[i]
        ok = np.isfinite(row) & (S > cal.values[i])
        lo = np.flatnonzero(ok[:j])
        hi = np.flatnonzero(ok[j + 1:]) + j + 1
        if len(lo) and len(hi):
            a, b = lo[-1], hi[0]
            w = bdays(S[a], S[j]) / bdays(S[a], S[b])
            return float(np.sqrt(row[a] ** 2 + w * (row[b] ** 2 - row[a] ** 2))), True
        if not len(hi) and len(lo) >= 2:
            a, m = lo[-1], lo[-2]
            w = bdays(S[a], S[j]) / bdays(S[m], S[a])
            v2 = row[a] ** 2 + w * (row[a] ** 2 - row[m] ** 2)
            return (float(np.sqrt(v2)) if v2 > 0 else np.nan), True
        return np.nan, True

    pos_ext = np.searchsorted(ext, cal.values)

    def weights(i: int) -> dict:
        nxt = ext[pos_ext[i] + 1]
        k1 = int(np.searchsorted(S, nxt, side="right"))
        dt_ = bdays(S[k1 - 1], S[k1])
        a = bdays(nxt, S[k1]) / dt_
        return {"st": {k1: a, k1 + 1: 1 - a},
                "mt": {k1 + 3: a, k1 + 4: 1.0, k1 + 5: 1.0, k1 + 6: 1 - a}}

    rec = {k: np.full(n, np.nan) for k in ("st_ret", "st_pts", "st_prev", "st_cm", "mt_ret", "mt_cm")}
    flags = {k: np.zeros(n, bool) for k in ("st_interp", "mt_interp", "st_close", "mt_close")}
    W = [weights(i) for i in range(n)]
    for i in range(n):
        for name in ("st", "mt"):
            # nivel ponderado al cierre de t con los pesos de t (vencimiento constante)
            num = den = 0.0
            for j, w in W[i][name].items():
                v, _ = price(i, j)
                num, den = num + w * v, den + w
            rec[f"{name}_cm"][i] = num / den if np.isfinite(num) else np.nan
            if i == 0:
                continue
            n1 = n0 = sw = 0.0
            interp = False
            for j, w in W[i - 1][name].items():
                if w == 0:
                    continue
                p1, f1 = price(i, j)
                p0, f0 = price(i - 1, j)
                n1, n0, sw = n1 + w * p1, n0 + w * p0, sw + w
                interp |= f0 or f1
                if j < CL.shape[1]:
                    flags[f"{name}_close"][i] |= bool(CL[i, j] or CL[i - 1, j])
            if np.isfinite(n1) and np.isfinite(n0) and n0 > 0:
                rec[f"{name}_ret"][i] = n1 / n0 - 1
                if name == "st":
                    rec["st_pts"][i] = (n1 - n0) / sw
                    rec["st_prev"][i] = n0 / sw
            flags[f"{name}_interp"][i] = interp
    out = pd.DataFrame({**rec, **flags}, index=cal)
    out.index.name = "date"
    return out


@lru_cache(maxsize=None)
def _indices_all() -> pd.DataFrame:
    p = _panel_all()
    cal, ext = _calendar()
    sched = expiry_schedule()
    M = p.pivot_table(index="date", columns="expiry", values="px", aggfunc="first")
    P = M.reindex(index=cal, columns=sched).to_numpy(float)
    CL = (p.assign(c=p["px_src"].str.endswith("close"))
          .pivot_table(index="date", columns="expiry", values="c", aggfunc="first")
          .reindex(index=cal, columns=sched).fillna(False).to_numpy(bool))
    out = sp_index_arrays(P, cal, sched, ext, CL)
    # letra de S&P (TBR) con la tasa de descuento ^IRX de la sesión anterior
    irx = sources.daily("^IRX")["^IRX"]["close"] / 100.0
    tbar = irx.reindex(irx.index.union(cal)).ffill().reindex(cal).shift(1)
    gap = pd.Series(cal, index=cal).diff().dt.days
    out["tbr"] = (1.0 / (1.0 - 91.0 / 360.0 * tbar)) ** (gap / 91.0) - 1.0
    for name in ("st", "mt"):
        r = out[f"{name}_ret"].where(out.index > ST_START)
        tr = (r + out["tbr"]).where(r.notna())
        base = out.index >= ST_START
        out[f"{name}_er"] = (100000.0 * (1 + r.fillna(0)).cumprod()).where(base)
        out[f"{name}_tr"] = (100000.0 * (1 + tr.fillna(0)).cumprod()).where(base)
    return out


def futures_indices(end: pd.Timestamp | None = END) -> pd.DataFrame:
    """Índices S&P 500 VIX Short-Term (st) y Mid-Term (mt) Futures reconstruidos, en el
    calendario de CFE (convención «settle»). Columnas:
      st_ret, mt_ret   rendimiento diario del índice ER (CDR_t), NaN si falta algún precio;
      st_er, st_tr, mt_er, mt_tr   niveles ER y TR, base 100.000 el 20-12-2005;
      st_pts           cambio diario en puntos del precio ponderado (pesos de t−1), para VYLD;
      st_prev          precio ponderado de t−1 con esos pesos (TDW_{t−1});
      st_cm, mt_cm     precio ponderado al cierre de t con los pesos de t (vencimiento constante
                       de ~30 días en el corto y ~5 meses en el medio);
      st_interp, mt_interp   True si algún precio del día es interpolado o extrapolado;
      st_close, mt_close     True si algún precio del día es un cierre (settle a 0 en 2013);
      tbr              letra de S&P del día (la del TR).
    Los rendimientos existen desde 2004; los niveles, desde ST_START. El índice medio no es
    defendible antes de MT_START."""
    f = _indices_all()
    return (f[f.index <= end] if end is not None else f).copy()


# ───────────────────────────── índices de Cboe ─────────────────────────────

@lru_cache(maxsize=None)
def _cboe_index_all(name: str) -> pd.Series:
    download_cboe_indices((name,))
    df = pd.read_csv(IDX_DIR / f"{name}_History.csv")
    df.columns = [c.strip().upper() for c in df.columns]
    col = "CLOSE" if "CLOSE" in df.columns else name.upper()
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").values,
                  index=pd.to_datetime(df["DATE"], format="mixed"), name=name).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.index.name = "date"
    return s


def cboe_index(name: str, end: pd.Timestamp | None = END) -> pd.Series:
    """Cierre diario de un índice de Cboe (VIX, VIX3M, VIX9D, VIX6M, VVIX, SHORTVOL, LONGVOL…)."""
    s = _cboe_index_all(name)
    return (s[s.index <= end] if end is not None else s).copy()


def cboe_indices(names=CBOE_INDICES, end: pd.Timestamp | None = END) -> pd.DataFrame:
    """Cierres de varios índices de Cboe en columnas (NaN antes de que exista cada uno)."""
    return pd.concat({n: cboe_index(n, end) for n in names}, axis=1).sort_index()


def er_1600(end: pd.Timestamp | None = END) -> pd.DataFrame:
    """Aproximación a las 16:00 ET del índice corto con los índices de Cboe (TWAP 15:45–16:00
    de los contratos 1.º y 2.º, sin letra: los dos son de exceso de rentabilidad, porque
    r_LONGVOL + r_SHORTVOL ≈ 0 en media). st_ret_1600 = −r_SHORTVOL (el índice de SVIX, en
    vivo desde el 22-11-2019); st_ret_1600_long = r_LONGVOL (el de UVIX, en vivo desde el
    8-10-2021); gap = (r_LONGVOL + r_SHORTVOL)/2, su discrepancia. Calendario de Cboe."""
    s = cboe_index("SHORTVOL", end).pct_change(fill_method=None)
    lg = cboe_index("LONGVOL", end).pct_change(fill_method=None)
    j = pd.concat([lg, s], axis=1, keys=["longvol_ret", "shortvol_ret"], sort=True).dropna()
    j["st_ret_1600"] = -j.shortvol_ret
    j["st_ret_1600_long"] = j.longvol_ret
    j["gap"] = (j.longvol_ret + j.shortvol_ret) / 2
    return j


# ───────────────────────────── caja, calendarios y utilidades ─────────────────────────────

def cash(index: pd.DatetimeIndex, kind: str = "lab") -> pd.Series:
    """Rendimiento diario de la letra sobre `index`: «lab» = ^IRX/252 por sesión (la caja del
    laboratorio, sources.risk_free_daily); «sp» = la TBR de S&P (días naturales)."""
    if kind == "lab":
        return sources.risk_free_daily(pd.DatetimeIndex(index))
    if kind == "sp":
        # una fecha que no es sesión de CFE recibe 0: la sesión siguiente ya devenga esos días
        return _indices_all()["tbr"].reindex(pd.DatetimeIndex(index)).fillna(0.0)
    raise ValueError(kind)


def _cal_days(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index, index=index).diff().dt.days.fillna(1.0)


def to_calendar(r: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Lleva rendimientos diarios a otro calendario componiéndolos: cada fecha de `index`
    recibe el producto de (1 + r) de las fechas de r que caen desde la fecha anterior de
    `index` (excluida) hasta ella (incluida); 0 si no cae ninguna; NaN fuera del rango de r."""
    r = r.dropna()
    index = pd.DatetimeIndex(index)
    out = pd.Series(np.nan, index=index, name=r.name)
    if r.empty:
        return out
    pos = index.searchsorted(r.index, side="left")
    keep = pos < len(index)
    g = (1.0 + r[keep]).groupby(pos[keep]).prod() - 1.0
    live = (index >= r.index[0]) & (index <= r.index[-1])
    out[live] = 0.0
    out.iloc[g.index.to_numpy()] = g.to_numpy()
    return out


def drop_days(r: pd.Series | pd.DataFrame, days=FEB_2018):
    """Quita esas sesiones de la serie (por defecto el 5 y el 6-2-2018), como si no hubieran
    existido: el resto se compone igual."""
    return r[~r.index.isin(pd.DatetimeIndex(days))]


def absorb(r: pd.Series) -> pd.Series:
    """Ruina absorbente: tras el primer día de −100% la riqueza queda en 0 (rendimiento 0)."""
    r = r.clip(lower=-1.0)
    dead = (r <= -1.0).cumsum().shift(1, fill_value=0) > 0
    return r.where(~dead, 0.0)


# ───────────────────────────── sintéticos ─────────────────────────────

def _er(convention: str, index_name: str, end) -> tuple[pd.Series, pd.Series]:
    """(rendimiento ER, cambio en puntos) del índice en la convención pedida."""
    F = futures_indices(end)
    if convention == "settle":
        return F[f"{index_name}_ret"], F["st_pts"]
    if convention in ("1600", "1600_long"):
        if index_name != "st":
            raise ValueError("no hay aproximación a las 16:00 del índice medio")
        # «1600»: −SHORTVOL (índice de SVIX); «1600_long»: LONGVOL (índice de UVIX, para validar)
        col = "st_ret_1600" if convention == "1600" else "st_ret_1600_long"
        e = er_1600(end)[col]
        prev = F["st_prev"]                               # aproximación: nivel de t−1 del settle
        prev = prev.reindex(prev.index.union(e.index)).ffill().reindex(e.index)
        return e, e * prev
    raise ValueError(convention)


def synthetic(key: str = "-1x", convention: str = "settle", fee: float | None = None,
              end: pd.Timestamp | None = END, cash_kind: str = "lab",
              leverage: float | None = None) -> pd.Series:
    """Rendimiento diario de un producto sintético de reajuste diario (claves de SYNTHETIC, o
    `leverage` para cualquier L sobre el índice corto con el gasto de `fee`, 0,95% si no se
    da). L·ER + letra − gasto·Δd/365; VYLD: −Δpuntos/100 + letra − gasto. Empieza el día
    siguiente a ST_START (índice corto) o a MT_START (medio)."""
    if leverage is not None:
        L, idx, fee0 = float(leverage), "st", 0.0095
    else:
        L, idx, fee0, _ = SYNTHETIC[key]
    fee = fee0 if fee is None else fee
    er, pts = _er(convention, idx, end)
    er = er.dropna()
    days = _cal_days(er.index)
    c = cash(er.index, cash_kind)
    core = -pts.reindex(er.index) / 100.0 if L == "puntos" else L * er
    r = (core + c - fee * days / 365.0).clip(lower=-1.0)
    start = ST_START if idx == "st" else MT_START
    r = r[r.index > start]
    r.name = key if leverage is None else f"{leverage:+g}x"
    return r


def synthetic_products(convention: str = "settle", end: pd.Timestamp | None = END,
                       keys=None) -> pd.DataFrame:
    """Todos los sintéticos de SYNTHETIC en columnas (el medio plazo solo con «settle»)."""
    keys = keys or [k for k in SYNTHETIC if convention == "settle" or SYNTHETIC[k][1] == "st"]
    return pd.concat({k: synthetic(k, convention, end=end) for k in keys}, axis=1)


# ───────────────────────────── ETPs reales ─────────────────────────────

def etp_prices(tickers=VIX_ETPS + EQUITY_ETFS, field: str = "adj_close",
               end: pd.Timestamp | None = END) -> pd.DataFrame:
    """Precios diarios de Yahoo (ajustados por splits; adj_close incluye distribuciones),
    descargados con sources.daily(source="yf") a la caché del laboratorio si faltan."""
    tickers = list(tickers)
    d = sources.daily(tickers, source="yf")
    px = pd.DataFrame({t: d[t][field] for t in tickers if t in d}).sort_index()
    return px[px.index <= end] if end is not None else px


def etp_returns(tickers=VIX_ETPS + EQUITY_ETFS, end: pd.Timestamp | None = END) -> pd.DataFrame:
    """Rendimientos diarios (adj_close) sin rellenar huecos."""
    return etp_prices(tickers, "adj_close", end).pct_change(fill_method=None)


def etp_leverage(ticker: str, index: pd.DatetimeIndex) -> pd.Series:
    e = ETP_SPECS[ticker]
    lev = pd.Series(e.leverage, index=index, dtype=float)
    if e.leverage_before is not None:
        lev[index < DESIGN_2018] = e.leverage_before
    return lev


def etp_model(ticker: str, convention: str | None = None, end: pd.Timestamp | None = END,
              fee: float | None = None) -> pd.Series:
    """Modelo diario de un ETP real sobre el índice reconstruido, en el calendario del índice:
    fondo L·ER + letra (laboratorio) − gasto; ETN: ER + TBR de S&P − gasto (sigue el TR);
    VYLD: −Δpuntos/100 + TBR − gasto. convention: la del producto si no se da."""
    e = ETP_SPECS[ticker]
    conv = convention or e.convention
    er, pts = _er(conv, e.index, end)
    er = er.dropna()
    days = _cal_days(er.index)
    fee = e.fee if fee is None else fee
    if e.kind == "fondo":
        core = etp_leverage(ticker, er.index) * er + cash(er.index, "lab")
    elif e.kind == "etn":
        core = er + cash(er.index, "sp")
    else:
        core = -pts.reindex(er.index) / 100.0 + cash(er.index, "sp")
    r = (core - fee * days / 365.0).clip(lower=-1.0)
    r.name = ticker
    return r


def minus_half_spliced(convention: str = "settle", end: pd.Timestamp | None = END) -> pd.Series:
    """El −0,5x del pre-registro (H2, H4, H5): sintético hasta el 27-2-2018 (calendario de CFE)
    y SVXY real (cierre de Yahoo) desde el 28-2-2018."""
    syn = synthetic("-0.5x", convention, end=end)
    real = etp_returns(["SVXY"], end)["SVXY"].dropna()
    out = pd.concat([syn[syn.index < DESIGN_2018], real[real.index >= DESIGN_2018]])
    out.name = "-0.5x"
    return out


# ───────────────────────────── estructura temporal ─────────────────────────────

def term_structure(end: pd.Timestamp | None = END, n: int = 4) -> pd.DataFrame:
    """Por sesión de CFE: f1…fn (settle de los n contratos listados con vencimiento posterior a
    t, en orden), e1…en (sus vencimientos), bd1/bd2 (sesiones de t, excluida, al vencimiento,
    incluido) y cd1 (días naturales); st_cm y mt_cm (precio ponderado de los índices); vix,
    vix3m (Cboe, rellenado como mucho 5 sesiones)."""
    p = vx_panel(end)
    p = p[p["expiry"] > p["date"]]
    p = p.assign(k=p.groupby("date").cumcount() + 1)
    p = p[p["k"] <= n]
    f = p.pivot(index="date", columns="k", values="px")
    e = p.pivot(index="date", columns="k", values="expiry")
    out = pd.concat([f.add_prefix("f"), e.add_prefix("e")], axis=1)
    out.columns = [str(c) for c in out.columns]
    _, ext = _calendar()
    t = out.index.values
    for k in (1, 2):
        ek = out[f"e{k}"].values.astype("datetime64[ns]")
        bd = (np.searchsorted(ext, ek, side="right") - np.searchsorted(ext, t, side="right")).astype(float)
        out[f"bd{k}"] = np.where(pd.isna(out[f"e{k}"]).values, np.nan, bd)
    out["cd1"] = (out["e1"] - out.index.to_series()).dt.days
    F = futures_indices(end)
    out["st_cm"], out["mt_cm"] = F["st_cm"].reindex(out.index), F["mt_cm"].reindex(out.index)
    for name in ("VIX", "VIX3M"):
        s = cboe_index(name, end)
        out[name.lower()] = s.reindex(s.index.union(out.index)).ffill(limit=5).reindex(out.index)
    return out


# ───────────────────────────── préstamo de títulos (IBKR) ─────────────────────────────

def ibkr_borrow(tickers=BORROW_TICKERS) -> pd.DataFrame:
    """Tasa diaria de préstamo de IBKR según iBorrowDesk: ticker, date, fee (% anual), rebate
    (% anual), available (acciones disponibles), low_fee, high_fee."""
    download_borrow(tickers)
    rows = []
    for t in tickers:
        f = BORROW_DIR / f"ibd_{t}.json"
        if not f.exists():
            continue
        d = pd.DataFrame(json.loads(f.read_text()).get("daily", []))
        if d.empty:
            continue
        d["date"] = pd.to_datetime(d["date"])
        d["ticker"] = t
        rows.append(d[["ticker", "date", "fee", "rebate", "available", "low_fee", "high_fee"]])
    return pd.concat(rows, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def borrow_summary(tickers=BORROW_TICKERS) -> pd.DataFrame:
    """Distribución de la tasa diaria de préstamo por ticker (% anual)."""
    b = ibkr_borrow(tickers)
    g = b.groupby("ticker")
    out = pd.DataFrame({
        "desde": g.date.min().dt.date, "hasta": g.date.max().dt.date, "días": g.size(),
        "mín": g.fee.min(), "p10": g.fee.quantile(0.10), "mediana": g.fee.median(),
        "p90": g.fee.quantile(0.90), "máx": g.fee.max(), "máx_intradía": g.high_fee.max(),
        "último": g.fee.last(), "rebate_mediana": g.rebate.median(),
        "disponibles_mediana": g.available.median(), "disponibles_mín": g.available.min()})
    return out.reindex([t for t in tickers if t in out.index])


# ───────────────────────────── cotizaciones de S&P (comprobación de nivel) ─────────────────────────────

def sp_index_quotes(refresh: bool = False) -> pd.DataFrame:
    """Último cierre que Yahoo da para los índices S&P de futuros del VIX (no tiene su historia):
    ^SPVXSP (ER corto), ^SPVXMP (ER medio) y ^SPVIXMT (TR medio). Se guarda con la hora de la
    consulta; la fecha del cierre se supone la sesión anterior a la consulta (sin verificar)."""
    if SP_QUOTES.exists() and not refresh:
        return pd.read_csv(SP_QUOTES, parse_dates=["consulta"])
    import yfinance as yf

    rows = []
    now = pd.Timestamp.now()
    for t, what in (("^SPVXSP", "st_er"), ("^SPVXMP", "mt_er"), ("^SPVIXMT", "mt_tr")):
        fi = yf.Ticker(t).fast_info
        rows.append({"ticker": t, "serie": what, "cierre_anterior": float(fi["previousClose"]),
                     "consulta": now})
    df = pd.DataFrame(rows)
    df.to_csv(SP_QUOTES, index=False)
    return df
