"""R26 — Los datos del artículo 4: futuros del VIX, índices reconstruidos y productos sintéticos.

Pregunta. ¿Se pueden reconstruir con datos públicos los índices de futuros del VIX de S&P y
los productos de reajuste diario (−1x, −0,5x, +1,5x, +2x y el VYLD de puntos) con fidelidad
suficiente para contrastar las hipótesis H1–H5 del pre-registro
(letf/preregistro/articulo4_volatilidad.md)? Esta línea no contrasta ninguna hipótesis: solo
construye y valida los datos que usan las demás (módulo letf_lab.data.vix).

Datos (todo en data/cache/, descargado una vez):
- futuros VX de CFE (liquidación diaria; archivo antiguo 2004–2013 y ficheros actuales
  2013–2026), con dos arreglos: escala 10 × VIX antes del 26-3-2007 y settle a 0 entre el
  2-1-2013 y el 17-5-2013 (se usa el settle del archivo antiguo o, si no hay, el cierre);
- índices de Cboe (VIX, VIX3M, VIX9D, VIX6M, VVIX, SHORTVOL y LONGVOL) de sus CSV públicos;
- ETPs de volatilidad (SVIX, SVXY, UVIX, UVXY, VIXM, VIXY, VXX, VXZ, VYLD) y ETFs de
  acciones (SQQQ, TQQQ, SPXU, UPRO, QQQ, SPY) de Yahoo, cierre de mercado;
- préstamo de títulos de IBKR (iBorrowDesk, ~12 meses).

Método. Índices corto (contratos 1.º y 2.º) y medio (4.º a 7.º) con la metodología de S&P
(pesos dr/dt; interpolación en varianza de los meses no listados antes de 2008), ER y TR
(letra de S&P con ^IRX), base 100.000 el 20-12-2005. Dos convenciones de hora, como pide el
pre-registro: «liquidación» (el settle de CFE: 16:15 ET hasta octubre de 2020 y 16:00 ET
después) y «16:00» (el TWAP de 15:45–16:00 del índice SHORTVOL de Cboe). Sintéticos: L·ER +
letra − gasto (VYLD: −Δpuntos/100 + letra − gasto). Validación frente a cada ETP real: Δg
(crecimiento logarítmico anual real − modelo) con IC del 90% por bootstrap estacionario
emparejado (bloque medio 63, 2.000 remuestreos), correlación, beta y error de seguimiento
diarios y semanales; el 5–6-2-2018; el cambio de hora de la liquidación en 2020. Y el error
del ^VIX de Yahoo del 6-2-2026, que también está en la caché antigua del laboratorio: qué
resultados anteriores usan ^VIX (R9, R23 y R25) y cuánto les afecta, sin cambiarlos.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from letf_lab import data, metrics, stats
from letf_lab.data import vix
from letf_lab.report import FIG_W, PCT_COLS, Report, comma_fig, fmt, legend_band, num_es, pct_es

SEED = 26
B = 2000                   # remuestreos del bootstrap (como R13 y R19)
BLOCK = 63                 # bloque medio del bootstrap estacionario
LAG = 2                    # señal al cierre de t, orden al cierre de t+1 (r13.EXEC_LAG)
END = vix.END
LRS = "LRS 3x (original)"
CENSUS = vix.sources.ROOT / "reports" / "r19_census" / "censo_rentabilidades_diarias.csv.gz"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#9a9994"

LOG_TICKS = (0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0)
CONV_ES = {"settle": "liquidación", "1600": "16:00", "1600_long": "16:00 (LONGVOL)"}
WINDOWS = {"principal": (vix.ST_START, END), "2011–2026": (vix.REAL_START, END),
           "desde 28-2-2018": (vix.DESIGN_2018, END)}


# ───────────────────────────── utilidades ─────────────────────────────

def log_growth(r: pd.Series) -> float:
    """Crecimiento logarítmico anual: media de log(1 + r) × 252 (como r13.log_growth)."""
    return float(np.log1p(r.dropna().clip(lower=-0.9999)).mean() * 252)


def dmy(x) -> str:
    return pd.Timestamp(x).strftime("%-d-%-m-%Y")


def pp(x: float, k: int = 1) -> str:
    v = 100 * float(x)
    return ("+" if v > 0 and round(v, k) != 0 else "") + num_es(v, k)


def span(s: pd.Series) -> tuple:
    s = s.dropna()
    return (s.index[0].date(), s.index[-1].date(), len(s)) if len(s) else (None, None, 0)


def window(r: pd.Series, a, b) -> pd.Series:
    return r[(r.index > pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))].dropna()


def paired_dg(p: pd.Series, m: pd.Series, rng, n_boot: int = B) -> tuple[float, float, float]:
    """Δg = g(p) − g(m) y su IC del 90% (bootstrap estacionario emparejado, bloque BLOCK)."""
    d = (np.log1p(p.clip(lower=-0.9999)) - np.log1p(m.clip(lower=-0.9999))).to_numpy()
    est, lo, hi = stats.bootstrap_ci(d, lambda x: x.mean() * 252, BLOCK, n_boot, rng)
    return est, lo, hi


class Tables:
    """Tablas numeradas con título (como R22): CSV sin redondear y markdown redondeado."""

    def __init__(self, rep: Report):
        self.rep, self.n = rep, 0

    def __call__(self, title: str, df: pd.DataFrame, pct: set | None = None, digits: int = 2,
                 index: bool = True):
        self.n += 1
        self.rep.p(f"**Tabla {self.n}. {title}**")
        dv = df.copy()
        if isinstance(dv.index, pd.MultiIndex):
            dv, index = dv.reset_index(), False
        pct = set(pct or ()) | PCT_COLS
        for c in dv.columns:
            num = pd.to_numeric(dv[c], errors="coerce")
            if num.notna().sum() != dv[c].notna().sum() or not num.notna().any() or dv[c].dtype == bool:
                continue
            if c in pct:
                dv[c] = num
            elif np.allclose(num.dropna(), num.dropna().round()):
                dv[c] = num.map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
            else:
                dv[c] = num.map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
        self.rep.lines += [fmt(dv, pct, digits).to_markdown(index=index), ""]
        df.to_csv(self.rep.dir / f"tabla_{self.n:02d}.csv")


# ───────────────────────────── 1. disponibilidad ─────────────────────────────

def availability() -> pd.DataFrame:
    rows = []
    raw = dict(zip(("archive", "cdn"), vix._load_raw()))
    p = vix.vx_panel()
    for src, name, url in (("archive", "Futuros VX: archivo antiguo de CFE", "CFE_<c><aa>_VX.csv"),
                           ("cdn", "Futuros VX: ficheros actuales de Cboe", "VX_<vencimiento>.csv")):
        x = raw[src]
        x = x[((x.settle > 0) | (x.close > 0)) & (x.date <= END)]
        used = int((p.src == src).sum())
        rows.append({"serie": name, "fuente": url, "desde": x.date.min().date(), "hasta": x.date.max().date(),
                     "sesiones": x.date.nunique(),
                     "nota": f"{x.contract.nunique()} contratos mensuales; {used} filas usadas en el panel"})
    F = vix.futures_indices()
    for k, name, a in (("st", "Índice corto reconstruido (ER y TR)", vix.ST_START),
                       ("mt", "Índice medio reconstruido (ER y TR)", vix.MT_START)):
        r = F[f"{k}_ret"][F.index > a].dropna()
        rows.append({"serie": name, "fuente": "futuros VX + metodología S&P", "desde": r.index[0].date(),
                     "hasta": r.index[-1].date(), "sesiones": len(r),
                     "nota": "base 20-12-2005" if k == "st" else "defendible desde el 24-4-2006 (7 contratos)"})
    live = {"SHORTVOL": "recalculado por Cboe antes del 22-11-2019", "LONGVOL": "recalculado por Cboe antes del 8-10-2021"}
    for n in vix.CBOE_INDICES:
        a, b, k = span(vix.cboe_index(n))
        rows.append({"serie": n, "fuente": "CSV público de Cboe", "desde": a, "hasta": b, "sesiones": k,
                     "nota": live.get(n, "")})
    px = vix.etp_prices(vix.VIX_ETPS + vix.EQUITY_ETFS)
    for t in px:
        a, b, k = span(px[t])
        note = vix.ETP_SPECS[t].note if t in vix.ETP_SPECS else ""
        rows.append({"serie": t, "fuente": "Yahoo (adj_close)", "desde": a, "hasta": b, "sesiones": k, "nota": note})
    irx = data.panel(["^IRX"], "close")["^IRX"]
    a, b, k = span(irx[irx.index <= END])
    rows.append({"serie": "^IRX (letra a 13 semanas)", "fuente": "Yahoo (caché del laboratorio)", "desde": a,
                 "hasta": b, "sesiones": k, "nota": "caja de los sintéticos y TBR de S&P"})
    bw = vix.ibkr_borrow()
    rows.append({"serie": "Préstamo de títulos (IBKR)", "fuente": "iBorrowDesk", "desde": bw.date.min().date(),
                 "hasta": bw.date.max().date(), "sesiones": int(bw.groupby("ticker").size().median()),
                 "nota": f"{bw.ticker.nunique()} tickers; sesiones = mediana por ticker"})
    return pd.DataFrame(rows).set_index("serie")


# ───────────────────────────── 2. futuros VX ─────────────────────────────

def vx_by_year() -> pd.DataFrame:
    p = vix.vx_panel()
    day = p.groupby("date").agg(contratos=("contract", "nunique"), volumen=("volume", "sum"), oi=("oi", "sum"))
    y = day.groupby(day.index.year).agg(sesiones=("contratos", "size"), contratos_mediana=("contratos", "median"),
                                        volumen_diario_mediano=("volumen", "median"),
                                        posición_abierta_mediana=("oi", "median"))
    src = p.groupby([p.date.dt.year, "px_src"]).size().unstack(fill_value=0)
    y = y.join(src.rename(columns=lambda c: f"filas_{c}"))
    T = vix.term_structure()
    ratio = (T.f1 / T.vix).dropna()
    y["F1/VIX_mediana"] = ratio.groupby(ratio.index.year).median()
    y.index.name = "año"
    return y


def vx_checks() -> dict:
    """Arreglos del panel: continuidad de la escala en 2007, solape archivo/actuales en 2013,
    error del cierre frente al settle y vencimientos del archivo distintos de la regla."""
    a, b = vix._load_raw()
    ov = a.merge(b, on=["date", "contract"], suffixes=("_a", "_b"))
    ov = ov[(ov.settle_a > 0) & (ov.settle_b > 0)]
    both = b[(b.settle > 0) & (b.close > 0)]
    rel = ((both.close - both.settle) / both.settle).abs()
    T = vix.term_structure()
    sc = T.loc["2007-03-20":"2007-03-28", ["f1", "vix"]]
    p = vix.vx_panel(end=None)
    last = p.groupby("contract").date.max()
    exp = p.groupby("contract").expiry.first()
    dev = pd.DataFrame({"último_día_con_precio": last.dt.date, "vencimiento_regla": exp.dt.date})
    dev = dev[(last != exp) & (last < END - pd.Timedelta(days=40))]
    dev.index.name = "contrato"
    zero = p[(p.date >= vix.ZERO_SETTLE[0]) & (p.date <= vix.ZERO_SETTLE[1] + pd.Timedelta(days=3))]
    # la regla con los festivos de CFE frente a los vencimientos de los ficheros actuales
    obs = b.groupby("contract")["expiry"].first()
    ses = vix.sessions()
    rule = pd.Series({c: vix.rule_expiry(*map(int, c.split("-")), ses) for c in obs.index})
    plain = pd.Series({c: vix.rule_expiry(*map(int, c.split("-"))) for c in obs.index})
    return {"solape_filas": len(ov), "solape_max_dif": float((ov.settle_a - ov.settle_b).abs().max()),
            "cierre_vs_settle": rel.describe(percentiles=[0.5, 0.9, 0.99]), "escala": sc,
            "vencimientos": dev, "fuente_2013": zero.px_src.value_counts(),
            "regla_observados": len(obs), "regla_aciertos": int((rule == obs).sum()),
            "regla_simple_aciertos": int((plain == obs).sum())}


# ───────────────────────────── 3. índices ─────────────────────────────

def index_checks() -> tuple[pd.DataFrame, pd.DataFrame]:
    F = vix.futures_indices(end=None)
    rows = []
    for k, a in (("st", vix.ST_START), ("mt", vix.MT_START)):
        x = F[F.index > a]
        rows.append({"índice": {"st": "corto", "mt": "medio"}[k],
                     "primer rendimiento": x.index[0].date(), "sesiones": int(x[f"{k}_ret"].notna().sum()),
                     "sin rendimiento": int(x[f"{k}_ret"].isna().sum()),
                     "con precio interpolado": int(x[f"{k}_interp"].sum()),
                     "último interpolado": x.index[x[f"{k}_interp"]].max().date() if x[f"{k}_interp"].any() else None,
                     "con cierre (2013)": int(x[f"{k}_close"].sum())})
    q = vix.sp_index_quotes()
    last = F.index[F.index < pd.Timestamp(q.consulta.iloc[0]).normalize()][-1]
    lv = []
    for _, r in q.iterrows():
        mine = float(F.loc[last, r.serie])
        lv.append({"índice S&P": r.ticker, "serie": r.serie, "fecha": last.date(),
                   "S&P (Yahoo, cierre anterior)": r.cierre_anterior, "reconstruido": mine,
                   "reconstruido ÷ S&P − 1": mine / r.cierre_anterior - 1})
    lv = pd.DataFrame(lv).set_index("índice S&P")
    lv.attrs["st_ratio"] = float(lv.loc["^SPVXSP", "S&P (Yahoo, cierre anterior)"] / lv.loc["^SPVXSP", "reconstruido"])
    return pd.DataFrame(rows).set_index("índice"), lv.drop(index="^SPVXSP")


# ───────────────────────────── 4. validación ─────────────────────────────

def conventions_of(t: str) -> list[str]:
    e = vix.ETP_SPECS[t]
    if e.index == "mt":
        return ["settle"]
    return ["settle", "1600_long" if t == "UVIX" else "1600"]


def validation(rng) -> tuple[pd.DataFrame, dict]:
    R = vix.etp_returns(vix.VIX_ETPS)
    rows, paths = [], {}
    for t in vix.VIX_ETPS:
        real = R[t].dropna()
        for conv in conventions_of(t):
            m = vix.to_calendar(vix.etp_model(t, conv), real.index)
            j = pd.concat([real, m], axis=1, keys=["p", "m"]).dropna().iloc[1:]
            dg, lo, hi = paired_dg(j.p, j.m, rng)
            wk = (1 + j).resample("W-FRI").prod() - 1
            own = conv == vix.ETP_SPECS[t].convention
            rows.append({"ETP": t, "convención": CONV_ES[conv] + (" (la suya)" if own else ""),
                         "desde": j.index[0].date(), "sesiones": len(j),
                         "g real": log_growth(j.p), "g modelo": log_growth(j.m), "Δg (pp)": 100 * dg,
                         "IC90 Δg": f"[{pp(lo)}; {pp(hi)}]",
                         "corr. diaria": j.p.corr(j.m), "beta diaria": np.polyfit(j.m, j.p, 1)[0],
                         "TE diario": (j.p - j.m).std() * np.sqrt(252),
                         "corr. semanal": wk.p.corr(wk.m), "TE semanal": (wk.p - wk.m).std() * np.sqrt(52)})
            if own:
                paths[t] = (np.log1p(j.p) - np.log1p(j.m)).cumsum()
            extra = []
            if t == "SVXY":                                 # el pre-registro pide el −1x sin el 5–6-2-2018
                extra.append((" (sin 5-6-2-2018)", vix.drop_days(j)))
            if t == "VXZ":                                  # en 2018 casi no cotizó (precio sin cambios)
                extra.append((" (desde 2019)", j.loc["2019-01-01":]))
            for tag, x in extra:
                dg, lo, hi = paired_dg(x.p, x.m, rng)
                wk = (1 + x).resample("W-FRI").prod() - 1
                rows.append({"ETP": t, "convención": CONV_ES[conv] + tag, "desde": x.index[0].date(),
                             "sesiones": len(x), "g real": log_growth(x.p), "g modelo": log_growth(x.m),
                             "Δg (pp)": 100 * dg, "IC90 Δg": f"[{pp(lo)}; {pp(hi)}]",
                             "corr. diaria": x.p.corr(x.m), "beta diaria": np.polyfit(x.m, x.p, 1)[0],
                             "TE diario": (x.p - x.m).std() * np.sqrt(252),
                             "corr. semanal": wk.p.corr(wk.m), "TE semanal": (wk.p - wk.m).std() * np.sqrt(52)})
    return pd.DataFrame(rows).set_index(["ETP", "convención"]), paths


# ───────────────────────────── 5. febrero de 2018 ─────────────────────────────

def feb2018() -> tuple[pd.DataFrame, pd.DataFrame]:
    F = vix.futures_indices()
    e = vix.er_1600()
    R = vix.etp_returns(["SVXY", "UVXY", "VIXY", "VXX", "VIXM", "VXZ"])
    cols = {"ER corto (liquidación)": F.st_ret, "ER corto (16:00)": e.st_ret_1600,
            "ER medio (liquidación)": F.mt_ret,
            "−1x sintético (liquidación)": vix.synthetic("-1x", "settle"),
            "−1x sintético (16:00)": vix.synthetic("-1x", "1600"),
            **{f"{t} (Yahoo)": R[t] for t in R}}
    d = pd.concat(cols, axis=1, sort=True).loc["2018-02-01":"2018-02-09"]
    d.index = d.index.date
    d.index.name = "fecha"
    mult = []
    series = {"−1x sintético (liquidación)": vix.synthetic("-1x", "settle"),
              "−1x sintético (16:00)": vix.synthetic("-1x", "1600"),
              "SHORTVOL de Cboe (−1x, 16:00, sin letra ni gasto)": vix.cboe_index("SHORTVOL").pct_change(),
              "SVXY real (Yahoo; entonces −1x)": R["SVXY"],
              "−0,5x sintético (liquidación)": vix.synthetic("-0.5x", "settle"),
              "−0,5x sintético (16:00)": vix.synthetic("-0.5x", "1600"),
              "+1x sintético (liquidación)": vix.synthetic("+1x", "settle"),
              "VIXY real (Yahoo)": R["VIXY"],
              "+2x sintético (liquidación)": vix.synthetic("+2x", "settle"),
              "UVXY real (Yahoo; entonces 2x)": R["UVXY"]}
    for k, r in series.items():
        two = float((1 + r.loc["2018-02-05":"2018-02-06"]).prod())
        wk = float((1 + r.loc["2018-02-01":"2018-02-09"]).prod())
        mult.append({"serie": k, "5 y 6-2-2018 (× capital)": two, "1 a 9-2-2018 (× capital)": wk})
    return d, pd.DataFrame(mult).set_index("serie")


# ───────────────────────────── 6. cambio de hora de 2020 ─────────────────────────────

def settle_change() -> tuple[pd.DataFrame, pd.DataFrame]:
    R = vix.etp_returns(["VIXY", "VXX", "UVXY", "SVXY", "VIXM", "VXZ"])
    rows = []
    for t in R:
        real = R[t].dropna()
        for conv in (["settle", "1600"] if vix.ETP_SPECS[t].index == "st" else ["settle"]):
            m = vix.to_calendar(vix.etp_model(t, conv), real.index)
            j = vix.drop_days(pd.concat([real, m], axis=1, keys=["p", "m"]).dropna().iloc[1:])
            for per, (a, b) in (("antes del 7-10-2020", (None, vix.SETTLE_CHANGE - pd.Timedelta(days=1))),
                                ("desde el 7-10-2020", (vix.SETTLE_CHANGE, None))):
                x = j.loc[a:b]
                rows.append({"ETP": t, "periodo": per, "modelo": CONV_ES[conv], "sesiones": len(x),
                             "corr. diaria": x.p.corr(x.m), "TE diario": (x.p - x.m).std() * np.sqrt(252)})
    tab = pd.DataFrame(rows).pivot_table(index=["ETP", "periodo"], columns="modelo",
                                         values=["corr. diaria", "TE diario"], sort=False)
    tab.columns = [f"{a} · {b}" for a, b in tab.columns]
    tab = tab.reindex(pd.MultiIndex.from_tuples([(t, p) for t in R for p in ("antes del 7-10-2020", "desde el 7-10-2020")],
                                                names=["ETP", "periodo"]))
    # correlación mensual de VIXY con cada ER
    F, e = vix.futures_indices(), vix.er_1600()
    real = R["VIXY"].dropna()
    j = pd.concat([real, vix.to_calendar(F.st_ret, real.index), vix.to_calendar(e.st_ret_1600, real.index)],
                  axis=1, keys=["p", "s", "h"]).dropna().iloc[1:]
    j = vix.drop_days(j)
    mon = j.groupby(j.index.to_period("M")).apply(lambda x: pd.Series({"liquidación": x.p.corr(x.s),
                                                                        "16:00": x.p.corr(x.h)}))
    mon.index = mon.index.to_timestamp()
    return tab, mon


# ───────────────────────────── 7. sintéticos ─────────────────────────────

def reference_series() -> dict:
    out = {"SPY (1x, Yahoo)": vix.etp_returns(["SPY"])["SPY"]}
    if CENSUS.exists():
        head = pd.read_csv(CENSUS, nrows=0).columns
        if LRS in head:
            c = pd.read_csv(CENSUS, index_col=0, parse_dates=True, usecols=[head[0], LRS])
            out[f"{LRS} (S&P 500, censo de R19)"] = c[LRS]
    return out


def synthetic_table() -> pd.DataFrame:
    rows = []
    for conv in ("settle", "1600"):
        S = vix.synthetic_products(conv)
        for k in S:
            r = S[k]
            a, b = WINDOWS["principal"]
            if vix.SYNTHETIC[k][1] == "mt":
                a = vix.MT_START
            x = window(r, a, b)
            rows.append({"producto": vix.LABELS[k], "convención": CONV_ES[conv], "desde": x.index[0].date(),
                         "CAGR": metrics.cagr(x), "CAGR sin 5-6-2-2018": metrics.cagr(vix.drop_days(x)),
                         "maxDD": metrics.max_dd(x), "peor día": x.min(), "fecha peor día": x.idxmin().date(),
                         "CAGR 2011–2026": metrics.cagr(window(r, *WINDOWS["2011–2026"])),
                         "CAGR desde 28-2-2018": metrics.cagr(window(r, *WINDOWS["desde 28-2-2018"]))})
    for k, r in reference_series().items():
        x = window(r, *WINDOWS["principal"])
        rows.append({"producto": k, "convención": "referencia", "desde": x.index[0].date(),
                     "CAGR": metrics.cagr(x), "CAGR sin 5-6-2-2018": metrics.cagr(vix.drop_days(x)),
                     "maxDD": metrics.max_dd(x), "peor día": x.min(), "fecha peor día": x.idxmin().date(),
                     "CAGR 2011–2026": metrics.cagr(window(r, *WINDOWS["2011–2026"])),
                     "CAGR desde 28-2-2018": metrics.cagr(window(r, *WINDOWS["desde 28-2-2018"]))})
    return pd.DataFrame(rows).set_index(["producto", "convención"])


# ───────────────────────────── 8. ^VIX de Yahoo ─────────────────────────────

def yahoo_vix() -> dict:
    y = data.panel(["^VIX", "^VIX3M"], "close")
    y = y[y.index <= END]
    c = vix.cboe_indices(["VIX", "VIX3M"])
    diffs, cover = [], []
    for yy, cc in (("^VIX", "VIX"), ("^VIX3M", "VIX3M")):
        j = pd.concat([y[yy], c[cc]], axis=1, keys=["y", "c"]).dropna()
        rel = j.y / j.c - 1
        big = j[rel.abs() > 0.005].assign(dif=rel[rel.abs() > 0.005])
        for d, r in big.iterrows():
            diffs.append({"índice": cc, "fecha": d.date(), "Yahoo": r.y, "Cboe": r.c, "Yahoo ÷ Cboe − 1": r.dif})
        ys, cs = y[yy].dropna(), c[cc].dropna()
        cover.append({"índice": cc, "Yahoo desde": ys.index[0].date(), "Cboe desde": cs.index[0].date(),
                      "sesiones comunes": len(j), "solo en Yahoo": len(ys.index.difference(cs.index)),
                      "solo en Cboe": len(cs.index.difference(ys.index)),
                      "días con |dif| > 0,5%": int((rel.abs() > 0.005).sum()), "máx |dif|": rel.abs().max()})
    # señales de R23 con uno y otro VIX, alineados como en r23.vix_series (calendario del S&P 500,
    # huecos de hasta 5 sesiones rellenados) y con su mediana expansiva de 252 observaciones
    idx = data.panel(["^GSPC"], "close")["^GSPC"].dropna().index
    idx = idx[(idx >= "1990-01-02") & (idx <= END)]

    def align(x: pd.Series) -> pd.Series:
        x = x.dropna()
        out = x.reindex(x.index.union(idx)).ffill(limit=5).reindex(idx)
        return out.where(idx >= x.index[0])

    vy, vc = align(y["^VIX"]), align(c["VIX"])
    ty, tc = align((y["^VIX"] / y["^VIX3M"]).dropna()), align((c["VIX"] / c["VIX3M"]).dropna())

    def below(x, thr):
        return (x < thr).astype(float).where(x.notna() & pd.Series(thr, index=x.index).notna())

    sig = {"VIX < 20": (below(vy, 20.0), below(vc, 20.0)),
           "VIX < su mediana expansiva": (below(vy, vy.expanding(252).median()),
                                         below(vc, vc.expanding(252).median())),
           "VIX/VIX3M < 1": (below(ty, 1.0), below(tc, 1.0))}
    spy = vix.etp_returns(["SPY"])["SPY"].reindex(idx)
    flips, net = [], []
    for k, (a, b) in sig.items():
        both = pd.concat([a, b], axis=1, keys=["y", "c"]).dropna()
        dif = both[both.y != both.c]
        eff = []
        for d in dif.index:
            pos = idx.get_loc(d)
            sess = idx[pos + LAG] if pos + LAG < len(idx) else None
            x3 = 3 * spy.loc[sess] if sess is not None else np.nan
            e = (1 if dif.loc[d, "y"] else -1) * x3
            eff.append(e)
            flips.append({"señal de R23": k, "día de la señal": d.date(),
                          "Yahoo": "dentro" if dif.loc[d, "y"] else "fuera",
                          "Cboe": "dentro" if dif.loc[d, "c"] else "fuera",
                          "sesión afectada": sess.date() if sess is not None else None,
                          "3 × SPY esa sesión": x3, "Yahoo − Cboe (≈)": e})
        yrs = len(both) / 252
        net.append({"señal de R23": k, "comparable desde": both.index[0].date(), "sesiones": len(both),
                    "señales distintas": len(dif), "de ellas, el 6-2-2026": int(pd.Timestamp("2026-02-06") in dif.index),
                    "efecto acumulado Yahoo − Cboe (≈)": float(np.nansum(eff)),
                    "pp/año (≈)": 100 * float(np.nansum(eff)) / yrs})
    # R9: tramo del VIX al comprar (S&P 500, horizonte de 3 años = 756 sesiones)
    edges, labels = [0, 15, 20, 30, 200], ["VIX < 15", "15–20", "20–30", "> 30"]
    by = pd.cut(vy, edges, labels=labels)
    bc = pd.cut(vc, edges, labels=labels)
    usable = idx[: max(0, len(idx) - 756 - 1)]
    r9 = pd.DataFrame({"Yahoo": by, "Cboe": bc}).reindex(usable).dropna()
    r9 = r9[r9.Yahoo.astype(str) != r9.Cboe.astype(str)]
    r9.index = r9.index.date
    r9.index.name = "día de compra"
    return {"diffs": pd.DataFrame(diffs).set_index(["índice", "fecha"]),
            "cover": pd.DataFrame(cover).set_index("índice"),
            "flips": pd.DataFrame(flips).set_index("señal de R23"),
            "net": pd.DataFrame(net).set_index("señal de R23"),
            "r9": r9.astype(str), "r9_n": len(usable)}


def vix_users(yv: dict) -> pd.DataFrame:
    net = yv["net"]
    n_r9 = len(yv["r9"])
    r9_days = ", ".join(dmy(d) for d in yv["r9"].index)

    def fl(k):
        r = net.loc[k]
        n, n6 = int(r["señales distintas"]), int(r["de ellas, el 6-2-2026"])
        if n == 0:
            return "ninguna"
        return f"{n} sesión{'es' if n > 1 else ''}" + (f" ({n6} el 6-2-2026)" if n6 else "")

    rows = [
        {"línea": "R9 (entrada óptima)", "qué usa": "^VIX: tramo al comprar (horizonte de 3 años) y el último VIX",
         "sesiones con otro valor": ((f"{n_r9} compra{'s' if n_r9 != 1 else ''} en otro tramo ({r9_days}), por otra "
                                      "discrepancia de Yahoo" if n_r9 else "ninguna compra en otro tramo")
                                     + "; el 6-2-2026 no entra (no tiene 3 años por delante) y el último VIX "
                                       "(25-9-2026) coincide con Cboe")},
        {"línea": "R23 (volatilidad)", "qué usa": "^VIX y ^VIX3M: reglas (c) VIX < 20, VIX < su mediana y "
                                                    "VIX/VIX3M < 1; clasificadores con el VIX como rasgo",
         "sesiones con otro valor": (f"señal distinta en «VIX < 20»: {fl('VIX < 20')}; en «VIX < su mediana»: "
                                     f"{fl('VIX < su mediana expansiva')}; en «VIX/VIX3M < 1»: {fl('VIX/VIX3M < 1')}; "
                                     "los clasificadores con VIX ven un rasgo distinto el 6-2-2026")},
        {"línea": "R25 (censo ampliado)", "qué usa": "las reglas de VIX de R23",
         "sesiones con otro valor": "las mismas que en R23"},
    ]
    return pd.DataFrame(rows).set_index("línea")


# ───────────────────────────── API para las demás líneas ─────────────────────────────

API = [
    ("vix.vx_panel(end)", "panel largo de futuros VX (settle, origen del precio, volumen, posición abierta)"),
    ("vix.vx_matrix(end, field)", "matriz fecha × vencimiento"),
    ("vix.term_structure(end)", "f1…f4, vencimientos, sesiones al vencimiento (bd1, bd2), st_cm, mt_cm, VIX y VIX3M de Cboe"),
    ("vix.futures_indices(end)", "índices corto y medio: st_ret/mt_ret (ER diario), niveles ER y TR, st_pts, marcas"),
    ("vix.er_1600(end)", "ER corto a las 16:00 (−SHORTVOL) y LONGVOL"),
    ("vix.synthetic(key, convention, fee, end)", "sintético de reajuste diario: '-1x', '-0.5x', '+1x', '+1.5x', '+2x', 'vyld', '+1x mt'; convention 'settle' | '1600'"),
    ("vix.synthetic_products(convention, end)", "todos los sintéticos en columnas"),
    ("vix.minus_half_spliced(convention, end)", "el −0,5x de H2, H4 y H5: sintético hasta el 27-2-2018 y SVXY real después"),
    ("vix.etp_returns(tickers, end) / vix.etp_prices(...)", "ETPs y ETFs de Yahoo (adj_close)"),
    ("vix.etp_model(ticker, convention, end)", "modelo diario de cada ETP real"),
    ("vix.cboe_index(name, end) / vix.cboe_indices(names, end)", "VIX, VIX3M, VIX9D, VIX6M, VVIX, SHORTVOL, LONGVOL de Cboe"),
    ("vix.cash(index, kind)", "letra diaria: 'lab' (^IRX/252, la del laboratorio) o 'sp' (TBR de S&P)"),
    ("vix.to_calendar(r, index)", "compone rendimientos de un calendario (CFE, Cboe) en otro (NYSE)"),
    ("vix.drop_days(r, days) / vix.FEB_2018", "quita el 5 y el 6-2-2018 (u otras sesiones)"),
    ("vix.absorb(r)", "ruina absorbente tras un −100%"),
    ("vix.ibkr_borrow() / vix.borrow_summary()", "préstamo diario de IBKR y su distribución"),
    ("vix.BORROW_GRID, PROCEEDS_SPREAD, FINRA_MAINT", "0/2/5,2/18%; letra − 0,5%; 30% × |L|"),
    ("vix.END, ST_START, MT_START, REAL_START, DESIGN_2018, SETTLE_CHANGE", "fechas del pre-registro"),
]


# ───────────────────────────── figuras ─────────────────────────────

def fig_validation(paths: dict, V: pd.DataFrame, rep: Report):
    order = ["VIXY", "VXX", "UVXY", "VIXM", "VXZ", "SVXY", "SVIX", "UVIX", "VYLD"]
    fig, axes = plt.subplots(3, 3, figsize=(FIG_W, 8.2), sharex=False)
    for ax, t in zip(axes.flat, order):
        c = paths[t]
        ratio = np.exp(c)
        ax.plot(ratio.index, ratio.values, color=BLUE, lw=1.4)
        ax.axhline(1, color=MUTED, lw=0.8)
        own = V.loc[t]
        row = own[own.index.str.contains("la suya")].iloc[0]
        ax.set_title(f"{t}: Δg {pp(row['Δg (pp)'] / 100)} pp", fontsize=12)
        ax.set_yscale("log")
        lo, hi = ratio.min(), ratio.max()
        pad = max(1.15, (hi / lo) ** 0.08)
        y0, y1 = min(lo, 1) / pad, max(hi, 1) * pad
        ax.set_ylim(y0, y1)
        ticks = [t for t in LOG_TICKS if y0 <= t <= y1]
        while len(ticks) > 5:
            ticks = ticks[::2] if 1.0 in ticks[::2] else ticks[1::2]
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num_es(v, 2 if v < 2 else 0)))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.yaxis.set_minor_locator(plt.NullLocator())
        yrs = (c.index[-1] - c.index[0]).days / 365.25
        if yrs < 3:
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%-m-%Y"))
        else:
            ax.xaxis.set_major_locator(mdates.YearLocator(base=4 if yrs > 10 else 2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.25)
    fig.suptitle("Valor del ETP real ÷ valor de su modelo (1 = seguimiento perfecto)")
    comma_fig(fig)
    rep.fig(fig, "validacion_etps",
            "Figura: cociente acumulado entre cada ETP real (cierre de Yahoo) y su modelo sobre el índice "
            "reconstruido, en la convención de hora del producto (liquidación de CFE para los de S&P; 16:00 para "
            "SVIX y UVIX); escala logarítmica")


def fig_feb2018(rep: Report):
    F = vix.futures_indices()
    series = {"−1x a la liquidación (16:15)": (vix.synthetic("-1x", "settle"), BLUE, "-"),
              "−1x a las 16:00 (SHORTVOL)": (vix.synthetic("-1x", "1600"), ORANGE, "-"),
              "SVXY real (cierre de Yahoo)": (vix.etp_returns(["SVXY"])["SVXY"], INK, "--")}
    a, b = pd.Timestamp("2018-01-31"), pd.Timestamp("2018-02-09")
    fig, ax = plt.subplots(figsize=(FIG_W, 5.0))
    lo_all, hi_all = 1.0, 1.0
    for k, (r, col, ls) in series.items():
        x = r.loc[a:b].copy()
        x.iloc[0] = 0.0
        w = (1 + x).cumprod()
        lo_all, hi_all = min(lo_all, w.min()), max(hi_all, w.max())
        ax.plot(w.index, w.values, color=col, ls=ls, lw=1.8, marker="o", ms=5, label=k)
        ax.annotate(num_es(w.iloc[-1], 2), (w.index[-1], w.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=11, color=MUTED)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num_es(v, 2)))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks([t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2) if lo_all / 1.3 <= t <= hi_all * 1.3])
    ax.set_xlim(a - pd.Timedelta(days=0.5), b + pd.Timedelta(days=1.2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d-%-m"))
    ax.set(ylabel="Valor de 1 $ (log)", title="Un −1x del 31-1 al 9-2-2018 según la hora de valoración")
    ax.grid(alpha=0.25)
    h, lab = ax.get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=1)
    comma_fig(fig)
    del F
    rep.fig(fig, "febrero_2018",
            "Figura: valor de 1 dólar en un −1x sobre el índice corto (con letra y gasto de 0,95%) valorado con la "
            "liquidación de CFE de las 16:15 y con el TWAP de las 16:00, y el SVXY real, del 31-1 al 9-2-2018")


def fig_settle_change(mon: pd.DataFrame, rep: Report):
    fig, ax = plt.subplots(figsize=(FIG_W, 4.8))
    ax.plot(mon.index, mon["liquidación"], color=BLUE, lw=1.5, label="con el ER de la liquidación de CFE")
    ax.plot(mon.index, mon["16:00"], color=ORANGE, lw=1.5, label="con el ER de las 16:00 (SHORTVOL)")
    ax.axvline(vix.SETTLE_CHANGE, color=MUTED, ls="--", lw=1)
    ax.text(vix.SETTLE_CHANGE, np.floor(mon.min().min() * 20) / 20 + 0.01, " liquidación a las 16:00\n (7-10-2020)",
            fontsize=11, color=MUTED, va="bottom")
    ax.set_ylim(max(0.0, np.floor(mon.min().min() * 20) / 20), 1.005)
    ax.set(ylabel="Correlación diaria en el mes",
           title="VIXY (cierre de las 16:00) frente a cada versión del índice corto")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.25)
    h, lab = ax.get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=1)
    comma_fig(fig)
    rep.fig(fig, "cambio_hora_2020",
            "Figura: correlación mensual de los rendimientos diarios de VIXY con el índice corto reconstruido con la "
            "liquidación de CFE y con el de las 16:00, 2011–2026 (sin el 5 y el 6-2-2018)")


def fig_synthetic(rep: Report):
    keys = [("-1x", "settle", BLUE, "-"), ("-1x", "1600", BLUE, ":"), ("-0.5x", "settle", ORANGE, "-"),
            ("-0.5x", "1600", ORANGE, ":"), ("vyld", "settle", AQUA, "-")]
    fig, ax = plt.subplots(figsize=(FIG_W, 5.6))
    for k, conv, col, ls in keys:
        r = vix.synthetic(k, conv)
        w = (1 + r).cumprod()
        name = "VYLD, puntos" if k == "vyld" else vix.LABELS[k]
        ax.semilogy(w.index, w.values, color=col, ls=ls, lw=1.6,
                    label=f"{name} ({CONV_ES[conv]}): {pct_es(metrics.cagr(r))}/año")
    spy = window(vix.etp_returns(["SPY"])["SPY"], *WINDOWS["principal"])
    w = (1 + spy).cumprod()
    ax.semilogy(w.index, w.values, color=GREY, lw=1.4, label=f"SPY (1x): {pct_es(metrics.cagr(spy))}/año")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num_es(v, 2 if v < 1 else 0)))
    ax.set(ylabel="Valor de 1 $ (log)", title="Sintéticos sobre el índice corto, 21-12-2005 → 25-9-2026")
    ax.grid(alpha=0.25)
    h, lab = ax.get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=1, fontsize=11)
    comma_fig(fig)
    rep.fig(fig, "sinteticos",
            "Figura: valor de 1 dólar en los sintéticos de reajuste diario (con letra y gasto) con las dos "
            "convenciones de hora, y en el SPY; escala logarítmica. Sin préstamo, sin impuestos y sin la liquidación "
            "anticipada que sufrió el XIV")


# ───────────────────────────── informe ─────────────────────────────

def compute() -> dict:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rng = np.random.default_rng(SEED)
    V, paths = validation(rng)
    return {"avail": availability(), "vx_year": vx_by_year(), "vx_checks": vx_checks(),
            "idx": index_checks(), "V": V, "paths": paths, "feb": feb2018(), "chg": settle_change(),
            "syn": synthetic_table(), "borrow": vix.borrow_summary(), "yvix": (yv := yahoo_vix()),
            "users": vix_users(yv)}


def run(C: dict | None = None):
    warnings.filterwarnings("ignore")
    rep = Report("r26_vix_data", "R26 — Los datos del artículo 4: futuros del VIX, índices reconstruidos y "
                                 "productos sintéticos")
    for f in rep.dir.glob("tabla_*.csv"):
        f.unlink()
    C = C or compute()
    tab = Tables(rep)
    V, syn, chk = C["V"], C["syn"], C["vx_checks"]
    feb_d, feb_m = C["feb"]
    chg, mon = C["chg"]
    idx_t, lvl = C["idx"]
    yv = C["yvix"]
    bs = C["borrow"]

    def v(t, conv):
        s = V.loc[t]
        return s[s.index.str.startswith(CONV_ES[conv])].iloc[0]

    m2 = feb_m["5 y 6-2-2018 (× capital)"]
    sv_ex = V.loc[("SVXY", "liquidación (sin 5-6-2-2018)")]
    sv_own, svix, uvix, vyld = v("SVXY", "settle"), v("SVIX", "1600"), v("UVIX", "1600_long"), v("VYLD", "settle")
    s1 = syn.loc[("−1x", "liquidación")]
    s1h = syn.loc[("−1x", "16:00")]
    s05 = syn.loc[("−0,5x", "liquidación")]
    s05h = syn.loc[("−0,5x", "16:00")]
    lrs_key = [k for k in syn.index.get_level_values(0) if k.startswith(LRS)]
    uvxy_b = bs.loc["UVXY"]
    vxx22 = float(np.exp(C["paths"]["VXX"].loc["2022"]).max())
    rz = vix.etp_returns(["VXZ"])["VXZ"]
    vxz_zero = int(((rz == 0) & (rz.index.year == 2018)).sum())
    vix_row = yv["diffs"].loc[("VIX", pd.Timestamp("2026-02-06").date())]
    own = V[V.index.get_level_values(1).str.contains("la suya")]
    low_wk = [t for t, _ in own.index[own["corr. semanal"] < 0.98]]
    mt_l = lvl.loc["^SPVXMP", "reconstruido ÷ S&P − 1"]
    mttr_l = lvl.loc["^SPVIXMT", "reconstruido ÷ S&P − 1"]

    rep.p(__doc__.split("\n", 2)[2])
    rep.h("Resumen")
    rep.p("\n".join([
        f"- **Los índices se reconstruyen bien.** Tras 20,8 años, el índice medio ER reconstruido acaba a "
        f"{pp(mt_l, 2)}% del valor que publica S&P y el TR a {pp(mttr_l, 2)}%. "
        f"Frente a los ETPs que siguen a S&P, la diferencia de crecimiento anual real − modelo es de "
        f"{num_es(v('VIXY', 'settle')['Δg (pp)'], 2)} pp en VIXY, {num_es(v('VXX', 'settle')['Δg (pp)'], 2)} pp en VXX, "
        f"{num_es(v('VIXM', 'settle')['Δg (pp)'], 2)} pp en VIXM, {num_es(v('VXZ', 'settle')['Δg (pp)'], 2)} pp en VXZ, "
        f"{num_es(v('UVXY', 'settle')['Δg (pp)'], 2)} pp en UVXY y {num_es(vyld['Δg (pp)'], 2)} pp en VYLD "
        f"(tabla 6). Semana a semana la correlación supera 0,98 en todos salvo " + " y ".join(low_wk) + ".",
        f"- **Los productos de VS Trust rinden bastante menos que su índice.** SVIX crece "
        f"{num_es(-svix['Δg (pp)'], 1)} pp al año menos que −1x SHORTVOL + letra − 1,35% y UVIX "
        f"{num_es(-uvix['Δg (pp)'], 1)} pp menos que 2x LONGVOL + letra − 1,65%. La diferencia es mayor que la "
        f"comisión de gestión; su desglose (gastos del folleto, ejecución) está sin verificar.",
        f"- **El 5 y el 6-2-2018 dependen de la hora.** Un −1x mantenido esos dos días queda multiplicado por "
        f"{num_es(m2.iloc[0], 3)} si se valora con la liquidación de CFE de las 16:15 (como el índice de S&P y el XIV), "
        f"por {num_es(m2.iloc[1], 3)} con el TWAP de las 16:00 (SHORTVOL) y por {num_es(m2.iloc[3], 3)} en el SVXY "
        f"real. Por eso el SVXY real crece {num_es(sv_own['Δg (pp)'], 1)} pp al año más que su modelo de liquidación "
        f"desde 2011 (tabla 6) y todo resultado con −1x se da con las dos horas y sin esos dos días.",
        f"- **El cambio de hora de octubre de 2020 se ve en los datos.** Antes, el cierre de VIXY (16:00) "
        f"correlaciona más con el índice de las 16:00 que con el de la liquidación (16:15); después, al revés "
        f"(tabla 9 y figura). La fecha exacta del cambio («el 7-10-2020 o después», CFE 2020-028) está sin verificar.",
        f"- **Sintéticos, 21-12-2005 → 25-9-2026 (sin préstamo ni impuestos):** el −1x crece "
        f"{pct_es(s1['CAGR'])} al año con la liquidación y {pct_es(s1h['CAGR'])} con las 16:00 "
        f"({pct_es(s1['CAGR sin 5-6-2-2018'])} y {pct_es(s1h['CAGR sin 5-6-2-2018'])} sin el 5 y el 6-2-2018); "
        f"el −0,5x, {pct_es(s05['CAGR'])} y {pct_es(s05h['CAGR'])}. "
        + (f"En la misma ventana la LRS 3x (original) del artículo 2 creció {pct_es(syn.loc[(lrs_key[0], 'referencia'), 'CAGR'])} "
           f"al año. Son descripciones, no contrastes: H1–H5 se contrastan en otras líneas." if lrs_key else ""),
        f"- **Préstamo en IBKR (iBorrowDesk, {bs['desde'].min()} → {bs['hasta'].max()}):** UVXY tuvo una mediana de "
        f"{num_es(uvxy_b['mediana'], 2)}% y un p90 de {num_es(uvxy_b['p90'], 2)}% anual, los 5,2% y 18% de la rejilla "
        f"del pre-registro (tabla 11).",
        f"- **El ^VIX de Yahoo tiene un cierre erróneo el 6-2-2026** ({num_es(vix_row['Yahoo'], 2)} frente a "
        f"{num_es(vix_row['Cboe'], 2)} en Cboe), que también está en la caché del laboratorio. R9, R23 y R25 usan ese "
        f"^VIX: con el de Cboe, R23 tendría {int(yv['net']['señales distintas'].sum())} sesiones de señal distintas en sus "
        f"reglas de VIX ({int(yv['net']['de ellas, el 6-2-2026'].sum())} por ese día) y R9 "
        f"{len(yv['r9'])} compra{'s' if len(yv['r9']) != 1 else ''} en otro tramo (por otra discrepancia); el efecto es "
        f"como mucho de {num_es(yv['net']['pp/año (≈)'].abs().max(), 3)} pp al año (sección 9). "
        "Esos resultados no se cambian aquí.",
    ]))

    rep.h("1. Fuentes y disponibilidad")
    tab(f"Series del artículo 4, recortadas al {dmy(END)} (fin de las ventanas del pre-registro)", C["avail"])
    rep.p("Las URL exactas y la huella de cada fichero de futuros están en data/cache/cboe_vx/manifest.csv; "
          "la fuente de cada función, en la cabecera de letf_lab/data/vix.py. Yahoo no tiene la serie A de VXX "
          "(2009–2019), ni XIV, ni la historia de los índices de S&P (solo su última cotización); le faltan las "
          "primeras sesiones de VXX y VXZ (serie B desde el 17-1-2018; Yahoo, desde el 25-1-2018).")

    rep.h("2. Futuros VX: arreglos y calidad")
    sc = chk["escala"]
    c2013 = chk["fuente_2013"]
    q = chk["cierre_vs_settle"]
    rep.p(
        f"- Escala: antes del 26-3-2007 el archivo de CFE cotiza 10 × VIX y se divide por 10. La serie queda "
        f"continua en el cambio: primer contrato {num_es(sc.loc['2007-03-23', 'f1'], 2)} el 23-3-2007 y "
        f"{num_es(sc.loc['2007-03-26', 'f1'], 2)} el 26-3-2007, con el VIX en {num_es(sc.loc['2007-03-23', 'vix'], 2)} y "
        f"{num_es(sc.loc['2007-03-26', 'vix'], 2)}; la mediana de F1/VIX por año está entre "
        f"{num_es(C['vx_year']['F1/VIX_mediana'].min(), 2)} y {num_es(C['vx_year']['F1/VIX_mediana'].max(), 2)} (tabla 2).\n"
        f"- Settle a cero en 2013: en los ficheros actuales el settle vale 0 del 2-1 al 17-5-2013. Se usa el settle del "
        f"archivo antiguo, que coincide con el actual en las {chk['solape_filas']} filas en que existen los dos (diferencia "
        f"máxima {num_es(chk['solape_max_dif'], 4)}), y, para los contratos que el archivo no tiene (julio de 2013 en "
        f"adelante), el cierre: {int(c2013.get('cdn_close', 0))} filas. Cuando existen los dos (desde mayo de 2013), el "
        f"cierre se aparta del settle una mediana de {pct_es(q['50%'], 2)} y un p99 de {pct_es(q['99%'], 1)}. El índice "
        f"corto no usa ningún cierre; el medio, en {idx_t.loc['medio', 'con cierre (2013)']} sesiones (tabla 4).\n"
        f"- Vencimientos: los de 2013 en adelante son los de los ficheros de Cboe; antes, la regla de CFE: el miércoles "
        f"30 días antes del tercer viernes del mes siguiente y, si ese viernes es festivo, 30 días antes de la sesión "
        f"anterior (si el día resultante es festivo, la sesión anterior). Con los festivos, la regla reproduce los "
        f"{chk['regla_aciertos']} de {chk['regla_observados']} vencimientos de los ficheros actuales (sin ellos, "
        f"{chk['regla_simple_aciertos']}). Antes de 2013 solo cambia uno: el 21-3-2008 fue Viernes Santo, así que el "
        f"contrato de febrero de 2008 liquidó el martes 19-2-2008, el día de su último precio en el archivo (fecha no "
        f"comprobada contra un calendario publicado por CFE). En el archivo, el último precio de cada contrato cae en "
        f"su vencimiento salvo en {len(chk['vencimientos'])} contratos (tabla 3): los dos de 2004 vencieron el miércoles "
        "anterior al tercer viernes de su propio mes (en 2004 regía otra regla, deducida aquí de los datos y sin "
        "verificar) y quedan antes de la ventana principal; el índice de antes del 20-12-2005 no se usa.")
    tab("Futuros VX mensuales por año: contratos con precio, volumen y posición abierta diarios (medianas), filas "
        "según el origen del precio y mediana de F1/VIX", C["vx_year"])
    tab("Contratos del archivo antiguo cuyo último precio no cae en el vencimiento de la regla", chk["vencimientos"])
    rep.p("En 2004–2005 solo había 3 o 4 contratos listados y un volumen mediano de unos cientos de contratos al día: "
          "por eso la ventana principal empieza el 20-12-2005 (fecha base de S&P) en el índice corto y el 24-4-2006 "
          "(primer día con 7 contratos) en el medio.")

    rep.h("3. Índices reconstruidos")
    tab("Índices reconstruidos: sesiones, precios interpolados y cierres usados en lugar del settle", idx_t)
    rep.p("La interpolación en varianza solo afecta al índice corto en 18 sesiones de 2006; en el medio, a los meses "
          "4.º–7.º que aún no se listaban (hasta abril de 2008) y a 8 sesiones de diciembre de 2012 (los ficheros "
          "actuales empiezan el 2-1-2013, así que al séptimo mes le faltan esos días).")
    tab("Comprobación de nivel: índice reconstruido (base 100.000 el 20-12-2005) frente a la última cotización de "
        "S&P en Yahoo (cierre anterior a la consulta; que sea el de esa fecha está sin verificar)", lvl,
        pct={"reconstruido ÷ S&P − 1"})
    rep.p(f"El índice corto no sirve de comprobación de nivel: el ^SPVXSP de Yahoo vale {num_es(lvl.attrs['st_ratio'], 0)} "
          "veces el reconstruido, lo que apunta a un rebase del índice de S&P por el camino (factor y fecha sin "
          "verificar). El medio, con la misma base de 100.000, "
          f"se aparta del de S&P un {pct_es(mt_l, 2)} (ER) y un {pct_es(mttr_l, 2)} (TR; S&P usa la subasta semanal de "
          "la letra y aquí se usa ^IRX) tras 20,8 años.")

    rep.h("4. Validación contra los ETPs reales")
    rep.p("Modelos: fondo L·ER + letra − gasto (VIXY, VIXM, SVXY, UVXY, SVIX, UVIX; SVXY −1x y UVXY 2x hasta el "
          "27-2-2018); ETN: ER + TBR de S&P − gasto (VXX, VXZ); VYLD: −Δpuntos/100 + TBR − 0,85%. Gastos: 0,85% (VIXY, "
          "VIXM, VYLD), 0,89% (VXX, VXZ), 0,95% (SVXY, UVXY), 1,35% (SVIX) y 1,65% (UVIX), las comisiones de gestión de "
          "los folletos. El modelo se lleva al calendario de NYSE componiendo sus rendimientos (vix.to_calendar). g: "
          "crecimiento logarítmico anual; Δg = g real − g modelo, con su IC del 90% por bootstrap estacionario "
          "emparejado; TE: error de seguimiento anualizado. Los precios de Yahoo son el cierre de mercado de las "
          "16:00, no el valor liquidativo.")
    tab("ETP real (Yahoo) frente a su modelo sobre el índice reconstruido, con cada convención de hora", V,
        pct={"g real", "g modelo", "TE diario", "TE semanal"}, digits=3)
    fig_validation(C["paths"], V, rep)
    rep.p(
        "- Los de S&P (VIXY, VXX, VIXM, VXZ, UVXY, VYLD) siguen al modelo de la liquidación con Δg de décimas de "
        "punto; la correlación diaria es más baja antes de 2020 por el desfase de 15 minutos entre el cierre de "
        "mercado y la liquidación (sección 6), y la semanal lo corrige.\n"
        f"- SVXY: su ventaja sobre el modelo de la liquidación sale del 5–6-2-2018 (sección 5): sin esos dos días, Δg es "
        f"de {num_es(sv_ex['Δg (pp)'], 2)} pp al año.\n"
        "- SVIX y UVIX: el cociente cae de forma sostenida, no por saltos: es un coste continuo por encima de la "
        "comisión de gestión.\n"
        f"- VXZ apenas cotizó en 2018: Yahoo repite el precio en {vxz_zero} sesiones de ese año (todo el 5–6-2-2018 "
        "incluido). Su validación útil es la de 2019 en adelante (fila «desde 2019»).\n"
        f"- VXX: Barclays suspendió la emisión el 14-3-2022 y la reanudó ese año (notas de prensa; fechas sin verificar "
        f"aquí). En 2022 el cociente con el modelo llega a {num_es(vxx22, 2)}: el ETN cotizó con prima sobre su valor "
        "indicativo mientras no se emitían participaciones nuevas, y la prima desapareció después.")

    rep.h("5. El 5 y el 6-2-2018")
    tab("Rendimientos diarios del 1 al 9-2-2018", feb_d, pct=set(feb_d.columns))
    tab("Multiplicador del capital en esos días", feb_m, digits=3)
    fig_feb2018(rep)
    er5 = feb_d.loc[pd.Timestamp("2018-02-05").date()]
    er6 = feb_d.loc[pd.Timestamp("2018-02-06").date()]
    rep.p(f"El 5-2-2018 el índice corto subió un {pct_es(er5['ER corto (liquidación)'], 0)} con la liquidación de las "
          f"16:15 y un {pct_es(er5['ER corto (16:00)'], 0)} con el TWAP de las 16:00: la mayor parte de la subida llegó "
          "en esos 15 minutos (la explicación habitual es el reajuste de los fondos inversos y apalancados; sin "
          f"verificar aquí). Con la liquidación, el −1x pierde un {pct_es(-er5['−1x sintético (liquidación)'], 0)} el día "
          f"5 y el rebote del día 6 ({pct_es(er6['−1x sintético (liquidación)'], 0)}) se aplica sobre casi nada; con las "
          f"16:00 pierde un {pct_es(-er5['−1x sintético (16:00)'], 0)}. El cierre de mercado del SVXY cayó un "
          f"{pct_es(-er5['SVXY (Yahoo)'], 0)} el día 5 y un {pct_es(-er6['SVXY (Yahoo)'], 0)} el día 6: quedó entre las "
          "dos convenciones (por qué exactamente, sin verificar). "
          "El XIV (Credit Suisse, −1x sobre el índice de S&P) sufrió un evento de aceleración el 5-2-2018 y se liquidó "
          "(última cotización el 15-2-2018); ninguno de los sintéticos de este informe lo simula: el que lo necesite "
          "debe cortar la serie. El valor final pagado por el XIV está sin verificar.")

    rep.h("6. El cambio de hora de la liquidación en 2020")
    tab("Correlación y error de seguimiento diarios de cada ETP con el modelo de la liquidación y con el de las 16:00, "
        "antes y después del 7-10-2020 (sin el 5 y el 6-2-2018)", chg, digits=3, pct={c for c in chg.columns if "TE" in c})
    fig_settle_change(mon, rep)
    rep.p("La Rule Filing CFE 2020-028 adelantó la liquidación diaria del VX de las 15:15 a las 15:00 de Chicago (16:15 → "
          "16:00 ET), «efectiva el 7-10-2020 o después». Los datos son compatibles con un cambio en octubre de 2020: "
          "desde entonces el cierre de mercado de los ETPs (16:00) sigue mejor a la liquidación que al TWAP de "
          "15:45–16:00. Consecuencia para las demás líneas: antes de octubre de 2020 la convención de la liquidación va "
          "15 minutos por detrás del cierre de las acciones: una señal que use la liquidación de ese día no se puede "
          "ejecutar al cierre de las 16:00 del mismo día; la orden al cierre siguiente del laboratorio evita ese "
          "problema.")

    rep.h("7. Productos sintéticos")
    rep.p("L·ER + letra (^IRX/252) − gasto por días naturales; VYLD: −Δpuntos/100 + letra − 0,85%. La convención de las "
          "16:00 usa −SHORTVOL como rendimiento del índice corto (no hay versión de las 16:00 del medio). Sin préstamo, "
          "sin impuestos y sin la liquidación anticipada de un ETN tras una caída grande.")
    tab(f"Sintéticos por convención de hora; ventana principal {dmy(vix.ST_START + pd.Timedelta(days=1))} → {dmy(END)} "
        "(medio plazo desde el 25-4-2006). Referencias: SPY y la LRS 3x (original) del artículo 2 en la misma ventana",
        syn, pct={"CAGR", "CAGR sin 5-6-2-2018", "maxDD", "peor día", "CAGR 2011–2026", "CAGR desde 28-2-2018"})
    fig_synthetic(rep)

    rep.h("8. Préstamo de títulos (IBKR)")
    tab("Tasa diaria de préstamo de IBKR por ticker (% anual; iBorrowDesk; disponibles en acciones)", bs,
        digits=2)
    vol_b = bs.loc[[t for t in vix.VIX_ETPS if t in bs.index]]
    rep.p(f"La rejilla del pre-registro (0 / 2 / 5,2 / 18% anual) toma la mediana ({num_es(uvxy_b['mediana'], 2)}%) y "
          f"el p90 ({num_es(uvxy_b['p90'], 2)}%) de UVXY; el 2% está entre el mínimo ({num_es(uvxy_b['mín'], 2)}%) y el "
          f"p10 ({num_es(uvxy_b['p10'], 2)}%) de UVXY, y la mediana de los nueve ETPs de volatilidad va de "
          f"{num_es(vol_b['mediana'].min(), 2)}% ({vol_b['mediana'].idxmin()}) a {num_es(vol_b['mediana'].max(), 2)}% "
          f"({vol_b['mediana'].idxmax()}). Son 12 meses de un solo bróker: no hay "
          "historia de préstamo antes de septiembre de 2025, así que cualquier resultado anterior aplica la rejilla "
          "constante. Fuente: https://www.iborrowdesk.com/api/ticker/<TICKER> (datos que publica Interactive Brokers), "
          "descargada el 29-9-2026 a data/cache/ibkr_borrow/.")

    rep.h("9. El ^VIX de Yahoo y los resultados anteriores del laboratorio")
    tab("Cobertura y discrepancias del VIX y del VIX3M de Yahoo (caché del laboratorio) frente a Cboe", yv["cover"],
        pct={"máx |dif|"})
    tab("Sesiones en que Yahoo y Cboe difieren más de un 0,5%", yv["diffs"], pct={"Yahoo ÷ Cboe − 1"})
    tab("Líneas del laboratorio que usan ^VIX o ^VIX3M de Yahoo y sesiones en que el dato de Cboe daría otro valor",
        C["users"])
    net = yv["net"]
    tab("Señales de R23 calculadas con el VIX de Yahoo (lo publicado) y con el de Cboe, alineadas como en R23: sesiones "
        "con señal distinta y efecto aproximado en la regla «3x/letras» (± 3 × SPY la sesión en que se nota; la regla "
        "con la LRS solo lo nota si la LRS está dentro)", net, pct={"efecto acumulado Yahoo − Cboe (≈)"}, digits=3)
    tab("Detalle: día de la señal, posición con cada fuente y sesión en que se nota (orden al cierre siguiente)",
        yv["flips"], pct={"3 × SPY esa sesión", "Yahoo − Cboe (≈)"})
    if len(yv["r9"]):
        tab(f"R9: compras (de {yv['r9_n']} con 3 años por delante) cuyo tramo del VIX cambia con el dato de Cboe",
            yv["r9"])
    f6 = yv["flips"][yv["flips"]["día de la señal"] == pd.Timestamp("2026-02-06").date()]
    worst = net["pp/año (≈)"].abs().max()
    rep.p(f"El error del 6-2-2026 (Yahoo {num_es(vix_row['Yahoo'], 2)}; Cboe {num_es(vix_row['Cboe'], 2)}) cambia la "
          f"señal de {len(f6)} regla(s) ({', '.join('«' + k + '»' for k in f6.index)}) en una sesión, la del "
          f"{dmy(f6['sesión afectada'].iloc[0])}, en la que 3 × SPY rindió {pct_es(f6['3 × SPY esa sesión'].iloc[0], 1)}: "
          "con el dato de Cboe esas reglas habrían estado en el 3x y habrían perdido eso. "
          + ("Las demás señales distintas salen de las discrepancias pequeñas de Yahoo en 2008–2021 (tabla 13) y de los "
             "días que solo tiene una de las fuentes. " if len(yv["flips"]) > len(f6) else
             "Ninguna otra discrepancia de Yahoo (tabla 13) cambia una señal de R23. ")
          + ("En R9, la del 9-12-2014 cambia una compra de tramo (tabla 17). " if len(yv["r9"]) else "")
          + "El efecto aproximado sobre lo publicado es como mucho de "
          f"{num_es(worst, 3)} pp al año: no cambia ninguna conclusión de R9, R23 ni R25, y sus cifras no se tocan aquí, "
          "solo se anotan. Además, el ^VIX3M de Yahoo empieza el 17-7-2006 y el CSV de Cboe el 18-9-2009: las 800 sesiones "
          "anteriores que usa R23 no se pueden contrastar con Cboe (origen sin verificar). En el artículo 4 el VIX y el "
          "VIX3M vienen siempre de Cboe (vix.cboe_index).")

    rep.h("10. Funciones para las demás líneas (letf_lab.data.vix)")
    rep.lines += [pd.DataFrame(API, columns=["función", "qué devuelve"]).to_markdown(index=False), ""]
    rep.p("Calendarios: los índices reconstruidos van en sesiones de CFE; los de Cboe en las suyas (en 2022–2026 el CSV "
          "del VIX trae filas en festivos de NYSE); los ETPs en sesiones de NYSE. Hay que alinear con vix.to_calendar, "
          "que compone los rendimientos (CFE abrió el 3-4-2015, el 5-12-2018 y el 9-1-2025, días en que NYSE cerró o no "
          "consta en Cboe; y Cboe calculó SHORTVOL el 2-4-2021, sin fichero de CFE). Los ETPs descargados hoy traen "
          "filas posteriores al 25-9-2026 (quizá una sesión incompleta): todas las funciones recortan en vix.END.")

    rep.h("Salvedades y datos sin verificar")
    rep.p("\n".join([
        "- La fecha exacta del cambio de hora de la liquidación (CFE 2020-028: «el 7-10-2020 o después»).",
        "- El rebase del índice corto de S&P (cociente ~10⁴) y que la cotización de Yahoo usada sea el cierre del "
        "28-9-2026.",
        "- SHORTVOL y LONGVOL antes de su lanzamiento (22-11-2019 y 8-10-2021) son recálculos de Cboe; en 2021 LONGVOL "
        "y −SHORTVOL discrepan algunos días (hasta un 0,5% diario): la convención de las 16:00 usa SHORTVOL.",
        "- La convención de las 16:00 de VYLD aproxima el cambio en puntos con el nivel de la liquidación del día "
        "anterior.",
        "- El desglose de la brecha de SVIX y UVIX con su índice, el pago final del XIV y el origen del tramo "
        "2006–2009 del ^VIX3M de Yahoo.",
        "- Los festivos futuros de FUTURE_HOLIDAYS (solo cuentan si el índice se calcula más allá del último dato).",
        "- El préstamo: 12 meses de un bróker (IBKR vía iBorrowDesk).",
    ]))
    rep.save()
    return C


if __name__ == "__main__":
    run()
