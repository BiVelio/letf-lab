"""R14 — Auditoría de la historia larga: ^GSPC + Shiller frente a CRSP (Fama–French).

La historia diaria del S&P 500 desde 1928 que usan R12 y R13 sale de ^GSPC en
Yahoo (el índice de 90 valores hasta el 4-3-1957) más el dividendo mensual de
Shiller. Aquí se contrasta con una fuente independiente: el mercado CRSP
ponderado por capitalización de la Ken French Data Library (Mkt-RF + RF,
diario desde 1926-07-01, con dividendos). CRSP no es el S&P 500 (todo el NYSE
hasta 1962, luego AMEX y, desde 1972, Nasdaq), así que una diferencia no es por
sí misma un error: la referencia de lo que es «normal» la dan las épocas con
datos limpios (1988–, ^SP500TR y SPY).

1. Correlación diaria, semanal y mensual, error de seguimiento y diferencias
   anuales por épocas de la serie de Yahoo.
2. Días sospechosos, con reglas FIJADAS ANTES de comparar las series (abajo).
3. Sesiones de sábado (NYSE hasta 1952): Yahoo no las tiene y su lunes acumula
   dos sesiones. Se acota el error que eso introduce en un fondo de reajuste
   diario con las sesiones reales de CRSP.
4. El RF de French (letra del Tesoro a 1 mes) frente a history.rf_long
   (3 meses, con un mes de retraso antes de 1960).
5. Robustez: las cifras clave de R12–R13 recalculadas con CRSP como índice, por qué
   las apalancadas rinden más sobre CRSP (media aritmética frente a ½σ², varianza
   diaria frente a semanal, autocorrelación) y la descomposición de la base anterior
   de R13 a CRSP paso a paso (3x de 1960–2009, señal, índice, tipo sin riesgo y
   cierres erróneos). Salvo que se diga, la LRS usa la señal original de Gayed y
   Bilello: la media de 200 sesiones de la rentabilidad total.

Reglas de días sospechosos (parámetros fijados antes de mirar los resultados).
r = rentabilidad de PRECIO de ^GSPC en Yahoo; c = rentabilidad de CRSP llevada
al calendario de Yahoo; σ = desviación típica de las 63 sesiones ANTERIORES.
- R1, par de reversión: |r_t| > 4σ_t y |r_t+1| > 4σ_t con signos opuestos,
  mientras CRSP queda plano: |c_t| < 2σᶜ_t y |c_t+1| < 2σᶜ_t.
- R2, discrepancia aislada: |(r_t − c_t) − mediana| > 6 desviaciones robustas
  (1,4826·MAD) de la diferencia en las 252 sesiones anteriores.
- R3, cierre congelado: r_t = 0 exactamente mientras |c_t| > 2σᶜ_t.
En la sensibilidad «días corregidos» se sustituyen por CRSP sólo los días de R1
y R3 (errores de dato casi seguros); los de R2 se listan pero no se corrigen,
porque pueden ser diferencias genuinas de composición.
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from letf_lab import data, kelly, metrics, stats
from letf_lab.data import history, sources
from letf_lab.report import FIG_W, Report, comma_fig
from letf_lab.research import r12_long_ml as r12
from letf_lab.research import r13_lrs as r13
from letf_lab.research.common import pair, rets
from letf_lab.research.r3_synthetic import synthetic, synthetic_actual

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"
SEED = 14
B = 500                     # remuestreos del IC de L* (como R12)
S500_FROM = "1957-03-04"    # primer día del S&P 500 de 500 valores
SAT_UNTIL = "1952-12-31"    # el NYSE dejó de abrir los sábados en 1952
GD = (r13.FIRST_RULE_DAY, "1938-12-31")   # Gran Depresión (como R13: primera sesión de la LRS)
CRASH = ("1929-09-03", "1932-07-08")   # del máximo al mínimo (como R12)

# Reglas de días sospechosos: fijadas antes de comparar las series
WIN = 63
REV_Z = 4.0
FLAT_Z = 2.0
ISO_WIN = 252
ISO_Z = 6.0
FROZEN_Z = 2.0

# Pares que R1 no marca pero que, revisados A MANO tras ver la lista de R2, tienen la
# firma de un cierre mal escrito (un cierre fuera de línea con sus dos vecinos y CRSP
# sin ese movimiento). Sólo se usan en la sensibilidad I, declarada a posteriori.
MANUAL_PAIRS = ["1931-01-07", "1935-05-16", "1935-12-16", "1938-10-10", "1946-09-03",
                "1948-02-04", "1949-01-24"]

KEY = ["1x comprar y mantener", "3x comprar y mantener", "LRS 3x (original)",
       "½ Kelly GJR (sin tendencia)", "LRS-Kelly"]
RULES = ["1x comprar y mantener", "3x comprar y mantener", "LRS 1x (original)", "LRS 3x (original)"]
SHORT = {"1x comprar y mantener": "1x", "3x comprar y mantener": "3x", "LRS 1x (original)": "LRS 1x",
         "LRS 3x (original)": "LRS 3x", "½ Kelly GJR (sin tendencia)": "½ Kelly GJR",
         "LRS-Kelly": "LRS-Kelly"}


def num(x: float, d: int = 2) -> str:
    """Número con coma decimal y signo menos tipográfico (sin «−0,00»)."""
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace(".", ",").replace("-", "−")


def pct(x: float, d: int = 1) -> str:
    """Porcentaje con coma decimal para el texto."""
    return num(x * 100, d) + "%"


def fplus(x: float, d: int = 2) -> str:
    """«f + 0,39%» o «f − 0,14%» para un diferencial sobre f dado en fracción."""
    return f"f {'−' if x < 0 else '+'} {num(abs(x) * 100, d)}%"


def sgn(x: float, d: int = 2) -> str:
    """Como num, con «+» explícito en las diferencias positivas."""
    s = num(x, d)
    return s if s.startswith("−") or not s.strip("0,") else "+" + s


# ───────────────────────────── datos de French ─────────────────────────────

def _french_text(name: str) -> str:
    raw = urllib.request.urlopen(FRENCH_URL.format(name), timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def parse_french(txt: str, digits: int) -> pd.DataFrame:
    """Primer bloque de un CSV de French (antes de «Annual Factors»), en fracciones.

    digits = 8 para ficheros diarios (AAAAMMDD) y 6 para mensuales (AAAAMM)."""
    rows = []
    for line in txt.splitlines():
        if "Annual" in line:
            break
        if re.match(rf"^\s*\d{{{digits}}}\s*,", line):
            rows.append(line)
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None,
                     names=["fecha", "mkt_rf", "smb", "hml", "rf"])
    fmt = "%Y%m%d" if digits == 8 else "%Y%m"
    df.index = pd.to_datetime(df.pop("fecha").astype(str).str.strip(), format=fmt)
    df.index.name = "fecha"
    return df.astype(float) / 100


def _cached(name: str, fetch) -> pd.DataFrame:
    path = sources.CACHE / f"french_{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = fetch()
    df.to_parquet(path)
    return df


def french_daily() -> pd.DataFrame:
    return _cached("ff3_daily", lambda: parse_french(_french_text("F-F_Research_Data_Factors_daily"), 8))


def french_monthly() -> pd.DataFrame:
    return _cached("ff3_monthly", lambda: parse_french(_french_text("F-F_Research_Data_Factors"), 6))


def daily_rf_from_monthly(dates: pd.DatetimeIndex, rf_m: pd.Series) -> pd.Series:
    """Tipo diario simple que, compuesto sobre las sesiones de cada mes, da el tipo
    mensual (la definición de French). rf_m va fechado el día 1 de cada mes."""
    per = pd.Series(dates.to_period("M"), index=dates)
    n = per.map(per.value_counts()).values.astype(float)
    m = pd.Series(rf_m.values, index=rf_m.index.to_period("M"))
    rm = m.reindex(per.values).values
    return pd.Series((1 + rm) ** (1 / n) - 1, index=dates, name="rf")


def crsp_market() -> pd.DataFrame:
    """Mercado CRSP-VW diario con dividendos en su calendario (con sábados).

    El RF del fichero diario está redondeado a 0,01% por sesión (hasta ±1,26%
    anual de sesgo si se suma tal cual); se reconstruye el diario exacto a partir
    del RF mensual, que es como French lo define."""
    d, m = french_daily(), french_monthly()
    rf_x = daily_rf_from_monthly(d.index, m["rf"]).fillna(d["rf"])
    return pd.DataFrame({"u": d["mkt_rf"] + rf_x, "rf": rf_x, "rf_redondeado": d["rf"],
                         "mkt_rf": d["mkt_rf"]})


# ───────────────────────────── utilidades ─────────────────────────────

def to_calendar(r: pd.Series, cal: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Compone r en los periodos (cal[i−1], cal[i]]; cal[0] es sólo la fecha base.

    Devuelve la rentabilidad compuesta en cal[1:] y cuántas sesiones de r caen en
    cada periodo (0 = fecha del calendario sin dato en r: rentabilidad 0)."""
    r = r.dropna()
    r = r[(r.index > cal[0]) & (r.index <= cal[-1])]
    pos = cal.searchsorted(r.index, side="left")
    g = pd.Series(np.log1p(r.values)).groupby(pos)
    s, k = g.sum(), g.size()
    lr, cnt = np.zeros(len(cal)), np.zeros(len(cal), int)
    lr[s.index.values] = s.values
    cnt[k.index.values] = k.values
    return (pd.Series(np.expm1(lr[1:]), index=cal[1:]),
            pd.Series(cnt[1:], index=cal[1:]))


def merge_gap(r: pd.Series, cal: pd.DatetimeIndex, L: float) -> pd.Series:
    """Por periodo de cal: log-riqueza de un fondo L× que reajusta tras cada sesión
    de r menos la de uno que sólo reajusta en las fechas de cal (sin costes).

    Con dos sesiones s y m, en niveles: (1+Ls)(1+Lm) − [1 + L((1+s)(1+m)−1)] = L(L−1)·s·m."""
    r = r.dropna()
    r = r[(r.index > cal[0]) & (r.index <= cal[-1])]
    pos = cal.searchsorted(r.index, side="left")
    native = pd.Series(np.log1p(L * r.values)).groupby(pos).sum()
    merged = np.expm1(pd.Series(np.log1p(r.values)).groupby(pos).sum())
    gap = native - np.log1p(L * merged)
    return pd.Series(gap.values, index=cal[gap.index.values])


def trailing_sd(x: pd.Series, win: int) -> pd.Series:
    return x.rolling(win).std().shift(1)


def reversal_pairs(r: pd.Series, c: pd.Series, win: int = WIN, z: float = REV_Z,
                   flat: float = FLAT_Z) -> pd.Series:
    """R1: True en t si (t, t+1) es un par de reversión > zσ en r con c plano (< flat·σᶜ)."""
    s_r, s_c = trailing_sd(r, win), trailing_sd(c, win)
    r1, c1 = r.shift(-1), c.shift(-1)
    flag = ((r.abs() > z * s_r) & (r1.abs() > z * s_r) & (np.sign(r) == -np.sign(r1))
            & (c.abs() < flat * s_c) & (c1.abs() < flat * s_c))
    return flag.fillna(False).astype(bool)


def discrepancy_score(r: pd.Series, c: pd.Series, win: int = ISO_WIN) -> pd.Series:
    """R2: (r − c − mediana) / (1,4826·MAD), ambos de las `win` sesiones anteriores."""
    d = r - c
    med = d.rolling(win).median().shift(1)
    mad = d.rolling(win).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True).shift(1)
    return (d - med) / (1.4826 * mad)


def frozen_closes(r: pd.Series, c: pd.Series, win: int = WIN, z: float = FROZEN_Z) -> pd.Series:
    """R3: cierre sin cambio en r mientras c se mueve más de zσᶜ."""
    return ((r == 0) & (c.abs() > z * trailing_sd(c, win))).astype(bool)


def lstar(u: pd.Series, fin: pd.Series, spread: float, rng, n_boot: int = B) -> tuple[float, float, float]:
    """L* aproximado (kelly.lstar_approx) = (μ − fin − s)/σ² con IC90 por bootstrap
    estacionario (bloques de un año).

    Con fin = f es la fórmula de r12.lstar_long, coherente con la financiación que
    usan las estrategias."""
    xy = pd.concat([u, fin], axis=1).dropna().values

    def f(a):
        return kelly.lstar_approx(a[:, 0], a[:, 1], spread)

    return stats.bootstrap_ci(xy, f, 252, n_boot, rng)


def growth_decomposition(r: pd.Series, periods: int = 252) -> dict:
    """Media log anual = media aritmética − ½σ² − resto, exacto por construcción.

    ½σ² es la penalización por varianza de las rentabilidades de cada periodo (ddof = 0);
    el resto recoge ½·media² y los momentos de orden superior de log(1 + r)."""
    x = r.dropna().values
    ar = x.mean() * periods
    hv = 0.5 * x.var() * periods
    lg = np.log1p(x).mean() * periods
    return {"media_aritm": ar, "½σ²": hv, "resto": ar - hv - lg, "media_log": lg}


def index_props(u: pd.Series) -> dict:
    """Volatilidad diaria y semanal anualizadas (rentabilidades log), razón de varianzas y
    autocorrelación de orden 1 de las rentabilidades diarias.

    VR = var(semana) / (sesiones por semana · var(día)): 1 sin autocorrelación y > 1 con
    autocorrelación positiva, p. ej., por precios desfasados (negociación no sincrónica).
    var_sesión_semanal = var(semana) / sesiones por semana es la varianza por sesión que
    implica la escala semanal; la diaria es var_sesión_semanal / VR."""
    lg = np.log1p(u.dropna())
    g = lg.resample("W-FRI")
    wk = g.sum()[g.count() > 0]
    n = len(lg) / len(wk)
    return {"vol_diaria": lg.std() * np.sqrt(252), "vol_semanal": wk.std() * np.sqrt(52),
            "VR": wk.var() / (n * lg.var()), "AR1": u.dropna().autocorr(1),
            "VR_implícita": vr_from_acf([lg.autocorr(k) for k in range(1, 5)], 5),
            "var_sesión_semanal": wk.var() / n, "var_diaria": lg.var()}


def vr_from_acf(rho, q: int = 5) -> float:
    """Razón de varianzas a q sesiones implícita en las autocorrelaciones ρ_1…ρ_{q−1} de una
    serie estacionaria: 1 + 2·Σ_{k<q} (1 − k/q)·ρ_k."""
    rho = np.asarray(rho, float)[: q - 1]
    k = np.arange(1, len(rho) + 1)
    return float(1 + 2 * np.sum((1 - k / q) * rho))


def vr_ar1(rho: float, q: int = 5) -> float:
    """Caso AR(1) de vr_from_acf: ρ_k = ρ^k."""
    return vr_from_acf(rho ** np.arange(1, q), q)


# ───────────────────────────── construcción de versiones ─────────────────────────────

def build(u: pd.Series, px: pd.Series, rf_: pd.Series, mode: str, spread: float | None = None,
          signal: str | None = None, yahoo: bool = True) -> dict:
    """Diccionario d de R12–R13 para un índice.

    mode = 'r13': la base ANTERIOR de R13 (real = synthetic_actual: UPRO real desde
               2009-06 y sintético de R3 desde 1960);
    mode = 'r13_s': la base actual de R13, real = sólo el UPRO real (el diferencial se
               recalibra en el periodo del ETF, que es lo que dice make(), y el 3x de
               1960–2009 es el sintético de R12 financiado a f);
    mode = 'sint': 3x sintético uniforme r12.letf(d, 3) en toda la muestra con el
               diferencial de 'r13_s' (el único válido para un índice que no es el S&P).
    spread: si se da, sustituye al diferencial que calcula make() (también en el Kelly)
               y el 3x se rehace con él donde no hay serie «real»; sirve para separar
               las dos correcciones de B (versiones A′ y B′).
    signal: señal de la LRS (r13.signal_index). Sin ella, la original de rentabilidad
               total; 'precio' usa px. yahoo=True: u es la serie de Yahoo + Shiller y la
               señal de rentabilidad total usa el D/P del mes anterior hasta 1988 y
               ^SP500TR después (r13.sp500_signal_returns, sin mirar el futuro); con
               CRSP (yahoo=False)
               los dividendos ya son los pagados y la señal usa u."""
    real = synthetic_actual("UPRO") if mode == "r13" else pair("UPRO").letf
    d = r12.make(u, rf_, "UPRO", real)
    if spread is not None:
        d["spread"] = spread
        rr = real.reindex(u.index)
        d["x3"] = rr.where(rr.notna(), r12.letf(d, 3.0))
    if mode == "sint":
        d["x3"] = r12.letf(d, 3.0)
    d["px"] = px
    if signal is not None:
        d["signal"] = signal
    if yahoo:
        d["u_sig"] = r13.sp500_signal_returns(u)
    return d


def run_version(d: dict) -> dict:
    var_fc = r13.var_forecast(d)
    R, E = r13.strategies(d, var_fc)
    return {"d": d, "R": R, "E": E, "start": var_fc.first_valid_index()}


def load(end=None) -> dict:
    """Series alineadas en el calendario de Yahoo (sp500_long) y CRSP nativo."""
    u_y = history.sp500_long(clean=False)  # la auditoría parte de la serie SIN corregir
    gspc = sources.panel(["^GSPC"], "close")["^GSPC"].dropna()
    crsp = crsp_market()
    end = min(u_y.index[-1], crsp.index[-1]) if end is None else pd.Timestamp(end)
    u_y = u_y.loc[:end]
    base = gspc.index[gspc.index < u_y.index[0]][-1]
    cal = pd.DatetimeIndex([base]).append(u_y.index)
    c, n_sess = to_calendar(crsp["u"], cal)
    r_px = gspc.pct_change(fill_method=None).reindex(u_y.index)
    tr0 = data.panel(["^SP500TR"], "adj_close")["^SP500TR"].first_valid_index()
    spy0 = rets("SPY")["SPY"].first_valid_index()
    return {"u_y": u_y, "gspc": gspc.loc[:end], "r_px": r_px, "crsp": crsp.loc[:end], "c": c,
            "n_sess": n_sess, "cal": cal, "end": end, "tr0": tr0, "spy0": spy0}


def eras(X: dict) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    u = X["u_y"]
    s90_end = u.index[u.index < S500_FROM][-1]
    tr_end = X["tr0"]
    spy0 = X["spy0"]
    return [("1928–1957 · S&P 90 + D/P Shiller", u.index[0], s90_end),
            ("1957–1988 · S&P 500 + D/P Shiller", pd.Timestamp(S500_FROM), tr_end),
            ("1988–1993 · ^SP500TR", u.index[u.index > tr_end][0], u.index[u.index < spy0][-1]),
            ("1993–2026 · SPY", spy0, X["end"]),
            ("1928–2026 · todo", u.index[0], X["end"])]


# ───────────────────────────── 1. comparación ─────────────────────────────

def era_table(X: dict, flagged: pd.Series | None = None) -> pd.DataFrame:
    df = pd.concat([X["u_y"], X["c"]], axis=1, keys=["y", "c"]).dropna()
    flagged = pd.Series(False, index=df.index) if flagged is None else flagged.reindex(df.index, fill_value=False)
    rows = {}
    for lab, a, b in eras(X):
        s = df.loc[a:b]
        lg = np.log1p(s)
        wk = lg.resample("W-FRI").sum()
        mo = lg.resample("ME").sum()
        yr = np.expm1(lg.groupby(lg.index.year).sum())
        full = s.groupby(s.index.year).size() > 200
        dy = (yr.y - yr.c)[full]
        beta = np.cov(s.y, s.c)[0, 1] / s.c.var()
        rows[lab] = {"sesiones": len(s), "corr_diaria": s.corr().iloc[0, 1],
                     "corr_semanal": wk.corr().iloc[0, 1], "corr_mensual": mo.corr().iloc[0, 1],
                     "beta_diaria": beta, "vol_Yahoo": s.y.std() * np.sqrt(252),
                     "vol_CRSP": s.c.std() * np.sqrt(252),
                     "error_seguimiento": (s.y - s.c).std() * np.sqrt(252),
                     "error_seg_sin_R1R2": (s.y - s.c)[~flagged.loc[a:b]].std() * np.sqrt(252),
                     "CAGR_Yahoo": metrics.cagr(s.y), "CAGR_CRSP": metrics.cagr(s.c),
                     "Δ_CAGR_pp": (metrics.cagr(s.y) - metrics.cagr(s.c)) * 100,
                     "dt_dif_anual_pp": dy.std() * 100}
    return pd.DataFrame(rows).T


def annual_table(X: dict) -> pd.DataFrame:
    df = pd.concat([X["u_y"], X["c"]], axis=1, keys=["y", "c"]).dropna()
    lg = np.log1p(df)
    yr = np.expm1(lg.groupby(lg.index.year).sum())
    yr["dif_pp"] = (yr.y - yr.c) * 100
    yr["sesiones"] = df.groupby(df.index.year).size()
    return yr.rename(columns={"y": "Yahoo", "c": "CRSP"})


def decade_table(yr: pd.DataFrame) -> pd.DataFrame:
    full = yr[yr.sesiones > 200]
    g = full.groupby((full.index // 10) * 10)
    out = pd.DataFrame({
        "años": g.size(),
        "dif_media_pp": g.dif_pp.mean(),
        "dt_dif_pp": g.dif_pp.std(),
        "máx_|dif|_pp": g.dif_pp.apply(lambda s: s.abs().max()),
        "año_máx": g.dif_pp.apply(lambda s: int(s.abs().idxmax())),
    })
    out.index = [f"{d}s" for d in out.index]
    return out


def calendar_audit(X: dict) -> pd.DataFrame:
    """Sesiones de CRSP que caen en cada fecha de Yahoo, por épocas."""
    n = X["n_sess"]
    fr = X["crsp"].index
    fr = fr[(fr > X["cal"][0]) & (fr <= X["end"])]
    sat = fr[fr.dayofweek == 5]
    rows = {}
    for lab, a, b in eras(X):
        s = n.loc[a:b]
        fr_ab = fr[(fr >= a) & (fr <= b)]
        rows[lab] = {"fechas_Yahoo": len(s), "sin_sesión_CRSP": int((s == 0).sum()),
                     "con_2+_sesiones_CRSP": int((s >= 2).sum()),
                     "sábados_CRSP": int(((sat >= a) & (sat <= b)).sum()),
                     "laborables_CRSP_sin_fecha_Yahoo": int(len(fr_ab[fr_ab.dayofweek < 5].difference(s.index)))}
    return pd.DataFrame(rows).T


# ───────────────────────────── 2. días sospechosos ─────────────────────────────

def suspicious(X: dict) -> dict:
    r, c = X["r_px"], X["c"].reindex(X["r_px"].index)
    ok = r.notna() & c.notna()
    r, c = r[ok], c[ok]
    rev = reversal_pairs(r, c)
    score = discrepancy_score(r, c)
    iso = score.abs() > ISO_Z
    frz = frozen_closes(r, c)
    s_r, s_c = trailing_sd(r, WIN), trailing_sd(c, WIN)
    pairs = []
    idx = r.index
    for t in idx[rev.values]:
        i = idx.get_loc(t)
        t1 = idx[i + 1]
        pairs.append({"fecha": t.date(), "fecha_siguiente": t1.date(),
                      "Yahoo_t": r.iloc[i], "Yahoo_t+1": r.iloc[i + 1],
                      "CRSP_t": c.iloc[i], "CRSP_t+1": c.iloc[i + 1],
                      "z_Yahoo_t": r.iloc[i] / s_r.iloc[i], "z_CRSP_t": c.iloc[i] / s_c.iloc[i]})
    iso_df = pd.DataFrame({"Yahoo": r[iso], "CRSP": c[iso], "dif_pp": (r - c)[iso] * 100,
                           "puntuación": score[iso]})
    frz_df = pd.DataFrame({"Yahoo": r[frz], "CRSP": c[frz], "z_CRSP": (c / s_c)[frz]})
    fix = pd.Series(False, index=r.index)
    fix[rev] = True
    fix[rev.shift(1, fill_value=False)] = True  # el segundo día del par
    fix[frz] = True
    flagged = fix | iso
    return {"flagged": flagged, "r": r, "c": c, "rev": rev, "iso": iso, "frz": frz, "score": score,
            "pairs": pd.DataFrame(pairs), "iso_df": iso_df, "frz_df": frz_df, "fix": fix,
            "z_r": r / s_r, "z_c": c / s_c}


def r2_pairs(S: dict) -> pd.DataFrame:
    """Días R2 consecutivos con discrepancias de signo opuesto: la firma de un cierre
    mal escrito que R1 no recoge (se listan, no se corrigen: incluyen días genuinos)."""
    iso = S["iso_df"].sort_index()
    idx = S["r"].index
    pos = idx.get_indexer(iso.index)
    rows = []
    for i in range(len(iso) - 1):
        a, b = iso.index[i], iso.index[i + 1]
        if pos[i + 1] == pos[i] + 1 and np.sign(iso.dif_pp.iloc[i]) != np.sign(iso.dif_pp.iloc[i + 1]):
            rows.append({"fecha": a.date(), "fecha_siguiente": b.date(),
                         "Yahoo_t": S["r"][a], "Yahoo_t+1": S["r"][b], "CRSP_t": S["c"][a], "CRSP_t+1": S["c"][b],
                         "en_R1": bool(S["rev"][a])})
    return pd.DataFrame(rows)


def manual_fix(S: dict) -> pd.Series:
    """R1/R3 más los pares revisados a mano (t y t+1)."""
    fix = S["fix"].copy()
    idx = fix.index
    for dte in MANUAL_PAIRS:
        i = idx.get_loc(pd.Timestamp(dte))
        fix.iloc[i] = True
        fix.iloc[i + 1] = True
    return fix


def suspicious_counts(S: dict, X: dict) -> pd.DataFrame:
    rows = {}
    for lab, a, b in eras(X):
        rows[lab] = {"R1_pares_reversión": int(S["rev"].loc[a:b].sum()),
                     "R2_discrepancias_aisladas": int(S["iso"].loc[a:b].sum()),
                     "R3_cierres_congelados": int(S["frz"].loc[a:b].sum()),
                     "días_con_r=0": int((S["r"].loc[a:b] == 0).sum())}
    return pd.DataFrame(rows).T


def corrected_yahoo(X: dict, S: dict, fix: pd.Series | None = None) -> tuple[pd.Series, pd.Series]:
    """u y px de Yahoo con los días marcados (R1/R3 por defecto) sustituidos por CRSP
    (en u, sólo donde u viene de ^GSPC, es decir, hasta el inicio de ^SP500TR)."""
    fix = S["fix"] if fix is None else fix
    days = fix.index[fix.values]
    u = X["u_y"].copy()
    early = days[days <= X["tr0"]]
    u.loc[early] = X["c"].loc[early]
    r = X["r_px"].copy()
    r.loc[days] = X["c"].loc[days]
    px0 = X["gspc"].loc[X["cal"][0]]
    px = px0 * (1 + r.fillna(0)).cumprod()
    return u, px


# ───────────────────────────── 3. sábados ─────────────────────────────

def saturday_analysis(X: dict, sig_w: pd.Series | None = None) -> dict:
    """Error de fusionar el sábado con el lunes en un fondo de reajuste diario.

    Se mide con las sesiones reales de CRSP: fondo L× que reajusta cada sesión
    (incluidos sábados) frente al que sólo reajusta en el calendario de Yahoo.
    Sin financiación ni costes (que en un fondo real se devengan por día natural)."""
    crsp = X["crsp"]
    cal = X["cal"]
    u = crsp["u"]
    fr = u[(u.index > cal[0]) & (u.index <= pd.Timestamp(SAT_UNTIL))]
    has_sat = pd.Series(fr.index.dayofweek == 5, index=fr.index)
    pos = cal.searchsorted(fr.index, side="left")
    sat_period = pd.Series(has_sat.values).groupby(pos).any()
    sat_period = pd.Series(sat_period.values, index=cal[sat_period.index.values])
    out = {"n_sábados": int(has_sat.sum())}
    gaps = {}
    for L in (2.0, 3.0):
        g = merge_gap(fr, cal, L)
        g = g.where(sat_period.reindex(g.index, fill_value=False), 0.0)
        gaps[L] = g
    out["gaps"] = gaps
    out["sat_period"] = sat_period
    # ¿el lunes de Yahoo incluye el sábado? correlación con CRSP compuesto y con el lunes solo
    per = sat_period[sat_period].index
    last_only = fr.groupby(pos).last()
    last_only = pd.Series(last_only.values, index=cal[last_only.index.values])
    y = X["u_y"].reindex(per)
    out["corr_lunes_con_sábado"] = float(np.corrcoef(y, X["c"].reindex(per))[0, 1])
    out["corr_lunes_solo"] = float(np.corrcoef(y, last_only.reindex(per))[0, 1])
    sat_r = fr[has_sat.values]
    out["vol_sábado_anual"] = float(sat_r.std() * np.sqrt(252))
    wk = fr[~has_sat.values]
    out["vol_laborable_anual"] = float(wk.std() * np.sqrt(252))
    if sig_w is not None:
        out["gap_LRS3x"] = gaps[3.0] * sig_w.reindex(gaps[3.0].index).fillna(0)
    return out


def saturday_table(sa: dict) -> pd.DataFrame:
    rows = {}
    for L, g in sa["gaps"].items():
        yrs = g.groupby(g.index.year)
        ann = yrs.sum()
        worst = g.abs().groupby(g.index.year).sum()
        n_years = len(g) / 252
        rows[f"{L:.0f}x"] = {"efecto_total_log_pp": g.sum() * 100,
                             "efecto_anual_medio_pp": g.sum() / n_years * 100,
                             "máx_|efecto|_en_un_año_pp": ann.abs().max() * 100,
                             "año_del_máximo": int(ann.abs().idxmax()),
                             "cota_peor_caso_anual_media_pp": worst.sum() / n_years * 100,
                             "cota_peor_caso_máx_año_pp": worst.max() * 100}
    if "gap_LRS3x" in sa:
        g = sa["gap_LRS3x"]
        ann = g.groupby(g.index.year).sum()
        worst = g.abs().groupby(g.index.year).sum()
        n_years = len(g) / 252
        rows["LRS 3x (sesiones invertidas)"] = {
            "efecto_total_log_pp": g.sum() * 100, "efecto_anual_medio_pp": g.sum() / n_years * 100,
            "máx_|efecto|_en_un_año_pp": ann.abs().max() * 100, "año_del_máximo": int(ann.abs().idxmax()),
            "cota_peor_caso_anual_media_pp": worst.sum() / n_years * 100,
            "cota_peor_caso_máx_año_pp": worst.max() * 100}
    return pd.DataFrame(rows).T


# ───────────────────────────── 4. tipo sin riesgo ─────────────────────────────

def rf_comparison(X: dict) -> dict:
    fm = french_monthly()["rf"]
    cal = X["cal"]
    rf_l = history.rf_long(X["u_y"].index)
    f_l = history.funding_long(X["u_y"].index)
    crsp = X["crsp"]
    per_idx = X["u_y"].index
    m_l = rf_l.groupby(per_idx.to_period("M")).sum()
    m_f = f_l.groupby(per_idx.to_period("M")).sum()
    fr = crsp.loc[(crsp.index > cal[0])]
    m_round = fr["rf_redondeado"].groupby(fr.index.to_period("M")).sum()
    m_fr = pd.Series(fm.values, index=fm.index.to_period("M"))
    M = pd.concat([m_fr, m_l, m_f, m_round], axis=1, keys=["french", "rf_long", "f", "french_diario"]).dropna()
    Y = M.groupby(M.index.year).sum()
    Y["n_meses"] = M.groupby(M.index.year).size()
    ann = Y[Y.n_meses == 12].copy()
    dec = ann.groupby((ann.index // 10) * 10)
    tab = pd.DataFrame({
        "French_1m": dec.french.mean(), "rf_long": dec.rf_long.mean(),
        "Δ_rf_long−French_pp": (dec.rf_long.mean() - dec.french.mean()) * 100,
        "máx_|Δ|_anual_pp": dec.apply(lambda s: (s.rf_long - s.french).abs().max()) * 100,
        "sesgo_RF_diario_redondeado_pp": (dec.french_diario.mean() - dec.french.mean()) * 100,
        "f_financiación": dec.f.mean(),
    })
    tab.index = [f"{d}s" for d in tab.index]
    full = {"corr_mensual": float(M.french.corr(M.rf_long)),
            "Δ_medio_pp": float((ann.rf_long - ann.french).mean() * 100),
            "sesgo_redondeo_medio_pp": float((ann.french_diario - ann.french).mean() * 100),
            "sesgo_redondeo_máx_pp": float((ann.french_diario - ann.french).abs().max() * 100),
            "año_sesgo_máx": int((ann.french_diario - ann.french).abs().idxmax())}
    return {"tabla": tab, "anual": ann, "resumen": full}


# ───────────────────────────── 5. robustez ─────────────────────────────

# Columnas de la tabla de L*: el rótulo compartido de R10, R12 y kelly (kelly.LSTAR_APROX).
LSTAR = kelly.LSTAR_APROX                                      # «L* aproximado»
LSTAR_A = f"{kelly.LSTAR_APROX} (diferencial de A)"             # el de la base anterior
LSTAR_B = f"{kelly.LSTAR_APROX} (diferencial corregido)"        # recalibrado solo con UPRO real


def lstar_table(V: dict, X: dict, rng) -> pd.DataFrame:
    """L* aproximado (kelly.LSTAR_APROX) = (μ − f − s)/σ² por fuente y ventana. Si la base
    anterior (A) y la actual (B) usan diferenciales distintos, con los dos (LSTAR_A, el de
    A, y LSTAR_B, el recalibrado solo con el UPRO real); si coinciden, una sola columna
    LSTAR (no se muestran pares iguales)."""
    rows = {}
    s_r13 = V["A"]["d"]["spread"]
    s_ok = V["D"]["d"]["spread"]
    both = not np.isclose(s_r13, s_ok, rtol=0, atol=1e-6)
    wins = [("1928–2026", None, None), ("1938–2026 (sin la Gran Depresión)", "1938-01-01", None),
            ("1928–1938 (Gran Depresión)", None, "1938-12-31")]

    def row(u, f):
        if not both:
            e, lo, hi = lstar(u, f, s_ok, rng)
            return {"años": len(u) / 252, LSTAR: e, "IC90": f"[{lo:.2f}, {hi:.2f}]"}
        e1, lo1, hi1 = lstar(u, f, s_r13, rng)
        e2, lo2, hi2 = lstar(u, f, s_ok, rng)
        return {"años": len(u) / 252, LSTAR_A: e1, "IC90 (diferencial de A)": f"[{lo1:.2f}, {hi1:.2f}]",
                LSTAR_B: e2, "IC90 (diferencial corregido)": f"[{lo2:.2f}, {hi2:.2f}]"}

    for src, key in (("Yahoo", "D"), ("CRSP", "E")):
        d = V[key]["d"]
        for wl, a, b in wins:
            rows[(src, wl)] = row(d["u"].loc[a:b], d["f"].loc[a:b])
    # CRSP desde 1926 en su propio calendario de laborables (sábados fundidos)
    crsp = X["crsp"]
    wk = crsp.index[crsp.index.dayofweek < 5]
    cal = pd.DatetimeIndex([wk[0] - pd.Timedelta(days=1)]).append(wk)
    u26, _ = to_calendar(crsp["u"], cal)
    u26 = u26.loc[:X["end"]]
    rows[("CRSP", "1926–2026")] = row(u26, history.funding_long(u26.index))
    t = pd.DataFrame(rows).T.infer_objects()
    t.index = [f"{a} · {b}" for a, b in t.index]
    return t


def compare_table(Ry: dict, Rc: dict, rf_: pd.Series, start, end, keys) -> pd.DataFrame:
    ty = r13.stats_table({k: Ry[k] for k in keys}, rf_, start, end)
    tc = r13.stats_table({k: Rc[k] for k in keys}, rf_, start, end)
    return pd.DataFrame({"CAGR_Yahoo": ty.CAGR, "CAGR_CRSP": tc.CAGR,
                         "Δ_CAGR_pp": (tc.CAGR - ty.CAGR) * 100,
                         "maxDD_Yahoo": ty.maxDD, "maxDD_CRSP": tc.maxDD,
                         "sharpe_Yahoo": ty.sharpe, "sharpe_CRSP": tc.sharpe})


# Todas las versiones usan la señal de la LRS original (media sobre la rentabilidad
# total) salvo las marcadas «señal de precio» (C y F).
VERSIONS = {
    "A": "A · base anterior de R13: 3x de R3 en 1960–2009 (Yahoo sin corregir)",
    "A′": "A′ · A con sólo el diferencial corregido (3x de R3 en 1960–2009)",
    "B′": "B′ · A con sólo el 3x de R12 a f en 1960–2009 (diferencial de A)",
    "B": "B · base actual de R13: make() sólo con UPRO real (Yahoo sin corregir)",
    "C": "C · Yahoo, 3x sintético uniforme, señal de precio",
    "D": "D · Yahoo, 3x sintético uniforme",
    "E": "E · CRSP-VW, construcción de D",
    "F": "F · CRSP-VW, señal de precio aprox. (TR − D/P)",
    "G": "G · A con el RF de French",
    "H": "H · D con los días sospechosos R1/R3 corregidos",
    "I": "I · H más 7 pares revisados a mano (a posteriori)",
}


def same_spread(dA: dict, dB: dict) -> bool:
    """¿Da make() el mismo diferencial con las dos construcciones? Desde que make()
    recalibra solo con el periodo del ETF real, sí: A′ y B′ coincidirían con A y B."""
    return bool(np.isclose(dA["spread"], dB["spread"], rtol=0, atol=1e-6))


def versions(X: dict, S: dict) -> dict:
    u_y, c = X["u_y"], X["c"]
    idx = u_y.index
    rf_l = history.rf_long(idx)
    px_y = X["gspc"].reindex(idx).ffill()
    # precio aproximado de CRSP: se resta el D/P de Shiller del MES ANTERIOR (sin mirar el futuro)
    dy_prev = history._monthly_to_daily(history.shiller_dividend_yield(), idx).ffill() / 252
    px_c_px = (1 + (c - dy_prev).fillna(0)).cumprod()
    rf_fr, _ = to_calendar(X["crsp"]["rf"], X["cal"])
    dA, dB = build(u_y, px_y, rf_l, "r13"), build(u_y, px_y, rf_l, "r13_s")
    V = {"A": run_version(dA)}
    if not same_spread(dA, dB):   # si coinciden, A′ = A y B′ = B: no se muestran pares iguales
        V["A′"] = run_version(build(u_y, px_y, rf_l, "r13", spread=dB["spread"]))
        V["B′"] = run_version(build(u_y, px_y, rf_l, "r13_s", spread=dA["spread"]))
    V.update({
        "B": run_version(dB),
        "C": run_version(build(u_y, px_y, rf_l, "sint", signal=r13.PRICE)),
        "D": run_version(build(u_y, px_y, rf_l, "sint", signal=r13.SIGNAL)),
        "E": run_version(build(c, px_c_px, rf_l, "sint", signal=r13.SIGNAL, yahoo=False)),
        "F": run_version(build(c, px_c_px, rf_l, "sint", signal=r13.PRICE, yahoo=False)),
        "G": run_version(build(u_y, px_y, rf_fr, "r13"))})
    if S["fix"].any():
        u_fix, px_fix = corrected_yahoo(X, S)
        V["H"] = run_version(build(u_fix, px_fix.reindex(idx), rf_l, "sint", signal=r13.SIGNAL))
    u_m, px_m = corrected_yahoo(X, S, manual_fix(S))
    V["I"] = run_version(build(u_m, px_m.reindex(idx), rf_l, "sint", signal=r13.SIGNAL))
    return V


ETF_OLD = "make() con synthetic_actual (base anterior)"
ETF_NEW = "sólo ETF real (base actual)"


def etf_tracking(V: dict) -> pd.DataFrame:
    """3x sintético de r12.make frente al ETF real en su periodo, con el diferencial
    que daba make() en la base anterior (real = synthetic_actual) y con el de la base
    actual (sólo el ETF real). La calibración iguala medias ARITMÉTICAS."""
    from letf_lab.research.common import long_underlying, rf as rf_irx
    rows = {}
    for t in ("UPRO", "TQQQ", "SOXL"):
        if t == "UPRO":
            u = V["A"]["d"]["u"]
            r = V["A"]["d"]["rf"]
        elif t == "TQQQ":
            u = history.ndx_backcast(M=1)["draws"][0]
            r = history.rf_long(u.index)
        else:
            u = long_underlying(t)
            r = rf_irx(u.index)
        real = pair(t).letf
        for lab, arg in ((ETF_OLD, synthetic_actual(t)), (ETF_NEW, real)):
            d = r12.make(u, r, t, arg)
            both = pd.concat([real, r12.letf(d, 3.0)], axis=1, keys=["real", "sint"]).dropna()
            lg = np.log1p(both)
            rows[(t, lab)] = {"diferencial_s_pp": d["spread"] * 100, "desde": both.index[0].date(),
                              "Δ_aritm_anual_pp": (both.sint - both.real).mean() * 252 * 100,
                              "Δ_log_anual_pp": (lg.sint - lg.real).mean() * 252 * 100,
                              "CAGR_real": metrics.cagr(both.real), "CAGR_sintético": metrics.cagr(both.sint)}
    t = pd.DataFrame(rows).T.infer_objects()
    t.index = [f"{a} · {b}" for a, b in t.index]
    return t


def version_table(V: dict, col: str, start, end, keys) -> pd.DataFrame:
    rows = {}
    rf_ = V["A"]["d"]["rf"]
    for k, v in V.items():
        t = r13.stats_table({s: v["R"][s] for s in keys}, rf_, start, end)
        rows[VERSIONS[k]] = {SHORT[s]: t.loc[s, col] for s in keys}
    return pd.DataFrame(rows).T


def crash_table(V: dict) -> pd.DataFrame:
    rows = {}
    for k in ("A", "D", "E"):
        e = r12.episode(V[k]["d"], *CRASH)
        rows[VERSIONS[k]] = e
    return pd.DataFrame(rows).T


LVL_KEYS = ["1x comprar y mantener", "3x comprar y mantener", "LRS 3x (original)"]
L3_DRAG = 0.5 * 9 * 252 * 100   # ½σ² anual (pp) de un 3x por unidad de varianza diaria del índice


def crsp_gap(V: dict, Dc: dict, start, end) -> dict:
    """Por qué las estrategias apalancadas rinden más con CRSP (E) que con Yahoo (D).

    Descompone la media log anual de cada estrategia en media aritmética − ½σ² − resto
    con las dos fuentes y con Yahoo corregido (Dc: días R1/R3 y pares revisados a mano,
    señal sobre rentabilidad total, a posteriori), y compara volatilidad diaria, semanal,
    razón de varianzas y autocorrelación de los tres índices por épocas. La brecha de ½σ²
    del 3x (≈ ½·9·Δσ²_diaria del índice) se reparte, de forma exacta en el índice, en
    errores de Yahoo (σ²_Y − σ²_Yc), varianza semanal ((W_Yc − W_C)/VR_Yc) y
    autocorrelación (W_C·(1/VR_Yc − 1/VR_C)), con W = varianza por sesión a escala
    semanal y σ²_diaria = W/VR."""
    srcs = (("Yahoo", V["D"]), ("Yahoo corregido", Dc), ("CRSP", V["E"]))
    lvl, dif = {}, {}
    for k in KEY:
        g = {s: growth_decomposition(v["R"][k].loc[start:end]) for s, v in srcs}
        cg = {s: metrics.cagr(v["R"][k].loc[start:end]) for s, v in srcs}
        if k in LVL_KEYS:
            for s in g:
                lvl[f"{SHORT[k]} · {s}"] = {**g[s], "CAGR": cg[s]}
        y, c = g["Yahoo"], g["CRSP"]
        dif[SHORT[k]] = {"Δ_CAGR_pp": (cg["CRSP"] - cg["Yahoo"]) * 100,
                         "Δ_media_log_pp": (c["media_log"] - y["media_log"]) * 100,
                         "por_media_aritm_pp": (c["media_aritm"] - y["media_aritm"]) * 100,
                         "por_½σ²_pp": -(c["½σ²"] - y["½σ²"]) * 100,
                         "por_resto_pp": -(c["resto"] - y["resto"]) * 100,
                         "Δ_CAGR_frente_a_Yahoo_corregido_pp": (cg["CRSP"] - cg["Yahoo corregido"]) * 100}
    periods = [(f"{start.year}–{end.year}", start, end), (f"{start.year}–1957", start, pd.Timestamp("1957-03-01")),
               ("1957–1987", pd.Timestamp(S500_FROM), pd.Timestamp("1987-12-31")),
               (f"1988–{end.year}", pd.Timestamp("1988-01-01"), end)]
    props, era3 = {}, {}
    k3 = "3x comprar y mantener"
    for lab, a, b in periods:
        for s, v in srcs:
            props[(lab, s)] = index_props(v["d"]["u"].loc[a:b])
        y, c = (growth_decomposition(V[x]["R"][k3].loc[a:b]) for x in ("D", "E"))
        era3[lab] = {"Δ_media_log_pp": (c["media_log"] - y["media_log"]) * 100,
                     "por_media_aritm_pp": (c["media_aritm"] - y["media_aritm"]) * 100,
                     "por_½σ²_pp": -(c["½σ²"] - y["½σ²"]) * 100,
                     "por_resto_pp": -(c["resto"] - y["resto"]) * 100}
    P = pd.DataFrame(props).T
    full = periods[0][0]
    py, pyc, pc = P.loc[(full, "Yahoo")], P.loc[(full, "Yahoo corregido")], P.loc[(full, "CRSP")]
    split = {"errores_Yahoo": (py.var_diaria - pyc.var_diaria) * L3_DRAG,
             "varianza_semanal": (pyc["var_sesión_semanal"] - pc["var_sesión_semanal"]) / pyc.VR * L3_DRAG,
             "autocorrelación": pc["var_sesión_semanal"] * (1 / pyc.VR - 1 / pc.VR) * L3_DRAG}
    split["suma"] = sum(split.values())
    split["brecha_½σ²_3x"] = -dif["3x"]["por_½σ²_pp"]
    # trayectoria: diferencia acumulada del 3x y sus dos términos principales
    b = pd.concat([V["D"]["R"][k3], V["E"]["R"][k3]], axis=1, keys=["y", "c"]).loc[start:end].dropna()
    cum = pd.DataFrame({"total": (np.log1p(b.c) - np.log1p(b.y)).cumsum(),
                        "aritm": (b.c - b.y).cumsum(),
                        "varianza": -0.5 * (b.c ** 2 - b.y ** 2).cumsum()})
    ar1 = pd.DataFrame({s: v["d"]["u"].rolling(756).corr(v["d"]["u"].shift(1)) for s, v in srcs}).loc[start:end]
    tab_p = P[["vol_diaria", "vol_semanal", "VR", "VR_implícita", "AR1"]].copy()
    tab_p.index = [f"{a} · {s}" for a, s in tab_p.index]
    ratio = pd.DataFrame({"observada": P.xs("CRSP", level=1).VR / P.xs("Yahoo", level=1).VR,
                          "implícita": P.xs("CRSP", level=1)["VR_implícita"] / P.xs("Yahoo", level=1)["VR_implícita"]})
    return {"niveles": pd.DataFrame(lvl).T, "diferencias": pd.DataFrame(dif).T, "props": P,
            "tabla_props": tab_p, "reparto": split, "acumulado": cum, "ar1": ar1, "full": full,
            "épocas_3x": pd.DataFrame(era3).T, "cociente_VR": ratio}


def ab_split(V: dict) -> dict:
    """Datos para separar las dos correcciones de B: media aritmética del 3x en 1960 →
    lanzamiento de UPRO según la versión, f − T-bill en ese tramo y en el del ETF real, y
    el diferencial de R3 sobre el T-bill."""
    live = pair("UPRO").letf.dropna().index[0]
    a, b = pd.Timestamp("1960-01-01"), live - pd.Timedelta(days=1)
    out = {f"x3_{k}": V[k]["d"]["x3"].loc[a:b].mean() * 252 for k in ("A", "A′", "B′", "B") if k in V}
    f, r = V["A"]["d"]["f"], V["A"]["d"]["rf"]
    out.update({"f_rf_1960": (f - r).loc[a:b].mean() * 252, "f_rf_live": (f - r).loc[live:].mean() * 252,
                "s_R3": synthetic("UPRO")[1], "s_A": V["A"]["d"]["spread"], "s_B": V["B"]["d"]["spread"],
                "lanzamiento": live, "hasta": b})
    return out


# ───────────────────────────── figuras ─────────────────────────────

# rótulos de los cambios de fuente de la serie de Yahoo (sin tickers que el lector no conoce)
ERA_LINES = [("1957-03-04", "S&P 500"), ("1988-01-04", "índice oficial"), ("1993-02-01", "SPY")]
# encima del panel superior, cada rótulo empieza en su línea («1988: índice oficial»), en dos
# alturas para que los de 1988 y 1993, a cinco años, no se pisen
ERA_TIERS = (0, 1, 0)


def era_label(dte: str, lab: str) -> str:
    """«1988: índice oficial»: el año del cambio de fuente y la fuente desde entonces."""
    return f"{pd.Timestamp(dte).year}: {lab}"


def _era_marks(ax, y_text=True):
    y0, y1 = ax.get_ylim()
    for dte, lab in ERA_LINES:
        ax.axvline(pd.Timestamp(dte), color="0.55", lw=0.9, ls=":")
        if y_text:
            ax.text(pd.Timestamp(dte) - pd.Timedelta(days=200), y0 + 0.03 * (y1 - y0), lab, va="bottom",
                    ha="right", rotation=90, fontsize=11, color="0.35")


def fig_difference(X: dict, yr: pd.DataFrame, rep: Report):
    df = pd.concat([X["u_y"], X["c"]], axis=1, keys=["y", "c"]).dropna()
    lg = np.log1p(df)
    cum = (lg.y - lg.c).cumsum() * 100
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 10.5), sharex=True)
    ax = axes[0]
    ax.plot(cum, color="k", lw=1.4)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_title("Diferencia acumulada Yahoo − CRSP (rentabilidad total)", pad=38)
    ax.set(ylabel="Puntos log acumulados")
    _era_marks(ax, y_text=False)
    # rótulos de los cambios de fuente encima del panel, en horizontal, sin tocar los datos: todos
    # empiezan en su línea (ha="left"), que sigue punteada hasta el rótulo
    for (dte, lab), tier in zip(ERA_LINES, ERA_TIERS):
        ax.annotate(era_label(dte, lab), xy=(mdates.date2num(pd.Timestamp(dte)), 1.0),
                    xycoords=ax.get_xaxis_transform(),
                    xytext=(0, 3 + 15 * tier), textcoords="offset points", ha="left", va="bottom",
                    fontsize=11, color="0.35", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=0.9, ls=":", shrinkA=0, shrinkB=0,
                                    relpos=(0, 0)))
    ax = axes[1]
    full = yr[yr.sesiones > 200]
    x = pd.to_datetime(full.index.astype(str) + "-07-01")
    ax.bar(x, full.dif_pp, width=300, color="0.35")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set(title="Diferencia de rentabilidad de cada año natural", ylabel="Yahoo − CRSP (pp)")
    _era_marks(ax, y_text=False)
    ax = axes[2]
    wk = lg.resample("W-FRI").sum()
    cd, cw = df.y.rolling(252).corr(df.c), wk.y.rolling(52).corr(wk.c)
    ax.plot(cd, color="C0", lw=1.2, label="diaria (ventana de 252 sesiones)")
    ax.plot(cw, color="C3", lw=1.2, label="semanal (ventana de 52 semanas)")
    lo = np.floor(min(cd.min(), cw.min()) * 20) / 20
    ax.set(title="Correlación móvil entre ambas series", ylabel="Correlación", ylim=(lo, 1.005))
    # abajo a la derecha, donde no hay datos: abajo a la izquierda tapaba el mínimo de 1936
    ax.legend(loc="lower right")
    _era_marks(ax, y_text=False)
    comma_fig(fig)
    rep.fig(fig, "diferencia_acumulada",
            "Diferencia acumulada y anual entre el S&P 500 de Yahoo/Shiller y el mercado CRSP, y su correlación móvil")


def fig_scatter(S: dict, rep: Report):
    fig, axes = plt.subplots(2, 1, figsize=(11, 13))
    zr, zc = S["z_r"], S["z_c"]
    fixd = S["fix"]
    for ax, (lab, a, b) in zip(axes, (("1928–1957 (S&P 90)", None, "1957-03-01"),
                                     ("1957–2026", "1957-03-04", None))):
        x, y = zc.loc[a:b], zr.loc[a:b]
        ok = x.notna() & y.notna()
        ax.scatter(x[ok].clip(-15, 15), y[ok].clip(-15, 15), s=4, color="0.6", alpha=0.35,
                   rasterized=True, label="sesiones")
        rev = (S["rev"] | S["rev"].shift(1, fill_value=False)).loc[a:b]
        iso = S["iso"].loc[a:b]
        frz = S["frz"].loc[a:b]
        if iso.any():
            ax.scatter(x[iso].clip(-15, 15), y[iso].clip(-15, 15), s=40, color="C0", edgecolor="w",
                       label=f"R2 discrepancia aislada ({int(iso.sum())})")
        if rev.any():
            ax.scatter(x[rev].clip(-15, 15), y[rev].clip(-15, 15), s=70, color="C3", edgecolor="w",
                       marker="D", label=f"R1 par de reversión ({int(rev.sum())} días)")
        if frz.any():
            ax.scatter(x[frz].clip(-15, 15), y[frz].clip(-15, 15), s=60, color="C1", edgecolor="w",
                       marker="s", label=f"R3 cierre congelado ({int(frz.sum())})")
        ax.plot([-15, 15], [-15, 15], color="0.3", lw=0.8, ls="--")
        ax.set(title=f"{lab}: movimiento de cada sesión en desviaciones típicas",
               xlabel="CRSP (z, σ de las 63 sesiones previas)", ylabel="^GSPC de Yahoo (z)",
               xlim=(-15.5, 15.5), ylim=(-15.5, 15.5))
        ax.legend(loc="upper left", fontsize=10)
    rep.fig(fig, "dias_sospechosos",
            "Cada sesión: movimiento de ^GSPC frente al de CRSP, en desviaciones típicas, con los días que marcan las reglas")


def fig_saturdays(sa: dict, rep: Report):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    g3, g2 = sa["gaps"][3.0], sa["gaps"][2.0]
    ann3 = g3.groupby(g3.index.year).sum() * 100
    worst3 = g3.abs().groupby(g3.index.year).sum() * 100
    ax = axes[0]
    ax.bar(ann3.index, ann3.values, color="C3", width=0.7, label="efecto real en el año (3x)")
    ax.plot(worst3.index, worst3.values, color="0.3", lw=1.2, marker="o", ms=4,
            label="cota de peor caso: Σ|efecto| (3x)")
    ax.plot(worst3.index, -worst3.values, color="0.3", lw=1.2, marker="o", ms=4)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set(title="Reajustar también el sábado frente a fundir sábado y lunes",
           ylabel="Diferencia de rentabilidad log (pp/año)")
    ax.legend(loc="lower left", fontsize=10)
    ax = axes[1]
    for g, col, lab in ((g3, "C3", "3x"), (g2, "C0", "2x")):
        ax.plot(g.cumsum() * 100, color=col, lw=1.5, label=lab)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set(title="Efecto acumulado 1928–1952", ylabel="Puntos log acumulados (%)")
    ax.legend(loc="upper left")
    rep.fig(fig, "sabados",
            "Efecto de las sesiones de sábado anteriores a 1952 sobre un fondo apalancado de reajuste diario")


def fig_rf(rc: dict, rep: Report):
    ann = rc["anual"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax = axes[0]
    ax.plot(ann.index, ann.french * 100, color="k", lw=1.5, label="French: letra a 1 mes")
    ax.plot(ann.index, ann.rf_long * 100, color="C0", lw=1.5, label="rf_long: letra a 3 meses (R12–R13)")
    ax.plot(ann.index, ann.f * 100, color="C1", lw=1.2, ls="--", label="f: call money / fed funds")
    ax.set(title="Tipo sin riesgo y de financiación: rentabilidad de cada año", ylabel="% anual")
    ax.legend(loc="upper left", fontsize=10)
    ax = axes[1]
    ax.bar(ann.index, (ann.rf_long - ann.french) * 100, color="C0", width=0.8,
           label="rf_long − French (mensual)")
    ax.plot(ann.index, (ann.french_diario - ann.french) * 100, color="C3", lw=1.4,
            label="RF diario redondeado − RF mensual (French)")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set(title="Diferencias", ylabel="pp/año")
    ax.legend(loc="lower left", fontsize=10)
    rep.fig(fig, "tipo_sin_riesgo",
            "Tipo sin riesgo de French frente al del laboratorio, y sesgo del RF diario redondeado")


STRAT_COLORS = {"1x comprar y mantener": "k", "3x comprar y mantener": "0.55", "LRS 3x (original)": "C0",
                "½ Kelly GJR (sin tendencia)": "C4", "LRS-Kelly": "C3"}


def fig_robust(V: dict, start, end, rep: Report):
    Ry, Rc = V["D"]["R"], V["E"]["R"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for k in KEY:
        y, c = Ry[k].loc[start:end].dropna(), Rc[k].loc[start:end].dropna()
        col = STRAT_COLORS[k]
        axes[0].semilogy((1 + y).cumprod() * 1e4, color=col, lw=1.5, label=f"{SHORT[k]}: Yahoo")
        axes[0].semilogy((1 + c).cumprod() * 1e4, color=col, lw=1.5, ls="--", label=f"{SHORT[k]}: CRSP")
        both = pd.concat([y, c], axis=1).dropna()
        lr = (np.log1p(both.iloc[:, 1].clip(lower=-0.999)) - np.log1p(both.iloc[:, 0].clip(lower=-0.999))).cumsum()
        axes[1].plot(lr * 100, color=col, lw=1.5, label=SHORT[k])
    axes[0].set(title="10.000 $ desde 1938: Yahoo (continua) frente a CRSP (discontinua)",
                ylabel="Valor (log)")
    axes[0].legend(loc="upper left", fontsize=10, ncol=2)
    axes[1].axhline(0, color="0.6", lw=0.7)
    axes[1].set(title="Diferencia acumulada CRSP − Yahoo de cada estrategia", ylabel="Puntos log (%)")
    axes[1].legend(loc="upper left", fontsize=10, ncol=3)
    rep.fig(fig, "robustez_crsp",
            "Las estrategias de R13 con el S&P 500 de Yahoo y con el mercado CRSP como índice, desde 1938")


def fig_mechanism(G: dict, rep: Report):
    fig, axes = plt.subplots(2, 1, figsize=(12, 11), sharex=True)
    ax = axes[0]
    cum = G["acumulado"] * 100
    ax.plot(cum["total"], color="k", lw=1.8, label="diferencia total (rentabilidad log)")
    ax.plot(cum["varianza"], color="C3", lw=1.4, label="por la penalización por varianza: −½·Δ(r²)")
    ax.plot(cum["aritm"], color="C0", lw=1.4, label="por la media aritmética: Δr")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set(title="3x: diferencia acumulada CRSP − Yahoo y de dónde sale", ylabel="Puntos log acumulados (%)")
    ax.legend(loc="upper left", fontsize=10)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.3 * (hi - lo), hi)
    _era_marks(ax)
    ax = axes[1]
    for s, col, ls in (("Yahoo", "k", "-"), ("Yahoo corregido", "C1", "--"), ("CRSP", "C0", "-")):
        ax.plot(G["ar1"][s], color=col, lw=1.2, ls=ls, label=s)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set(title="Autocorrelación de orden 1 del índice (ventana de 3 años)", ylabel="AR(1) diaria")
    ax.legend(loc="upper right", fontsize=10)
    _era_marks(ax, y_text=False)
    rep.fig(fig, "mecanismo_crsp",
            "Por qué el 3x rinde más sobre CRSP: menor varianza diaria por la autocorrelación del índice amplio")


def fig_depression(V: dict, rep: Report):
    Ry, Rc = V["D"]["R"], V["E"]["R"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 11), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for k in ("1x comprar y mantener", "3x comprar y mantener", "LRS 3x (original)"):
        col = STRAT_COLORS[k]
        for R, ls, src in ((Ry, "-", "Yahoo"), (Rc, "--", "CRSP")):
            r = R[k].loc[GD[0]:GD[1]].dropna()
            w = (1 + r).cumprod() * 1e4
            axes[0].semilogy(w, color=col, lw=1.5, ls=ls, label=f"{SHORT[k]}: {src}")
            axes[1].plot((w / w.cummax() - 1) * 100, color=col, lw=1.2, ls=ls)
    axes[0].set(title="La Gran Depresión con cada fuente: 10.000 $ desde octubre de 1928",
                ylabel="Valor (log)")
    axes[0].set_ylim(bottom=4)
    axes[0].legend(loc="lower left", fontsize=10, ncol=3)
    axes[1].set(title="Caída desde máximos", ylabel="%")
    rep.fig(fig, "gran_depresion_crsp",
            "1x, 3x y LRS 3x durante la Gran Depresión con el S&P 90 de Yahoo y con el mercado CRSP")


# ───────────────────────────── cálculo ─────────────────────────────

def compute() -> dict:
    rng = np.random.default_rng(SEED)
    X = load()
    S = suspicious(X)
    V = versions(X, S)
    R = {"X": X, "S": S, "V": V}
    R["eras"] = era_table(X, S["flagged"])
    R["calendario"] = calendar_audit(X)
    R["anual"] = annual_table(X)
    R["decadas"] = decade_table(R["anual"])
    R["sospechosos"] = suspicious_counts(S, X)
    R["pares_R2"] = r2_pairs(S)
    sig_w = r13.above_sma(r13.signal_index(V["A"]["d"])).shift(r13.EXEC_LAG)   # señal original
    R["sabados"] = saturday_analysis(X, sig_w)
    R["sabados_tabla"] = saturday_table(R["sabados"])
    R["rf"] = rf_comparison(X)
    start, end = V["A"]["start"], X["end"]
    R["start"], R["end"] = start, end
    R["periodos"] = [(f"{start.year}–{end.year}", start, end),
                     (f"{start.year}–1957 (S&P 90)", start, pd.Timestamp("1957-03-01")),
                     (f"1957–{end.year}", pd.Timestamp(S500_FROM), end)]
    R["comparación"] = {lab: compare_table(V["D"]["R"], V["E"]["R"], V["D"]["d"]["rf"], a, b, KEY)
                        for lab, a, b in R["periodos"]}
    R["comparación"]["Gran Depresión"] = compare_table(V["D"]["R"], V["E"]["R"], V["D"]["d"]["rf"], *GD, RULES)
    R["crash"] = crash_table(V)
    R["versiones_CAGR"] = version_table(V, "CAGR", start, end, KEY)
    R["versiones_maxDD"] = version_table(V, "maxDD", start, end, KEY)
    R["versiones_GD"] = version_table(V, "CAGR", *GD, RULES)
    R["lstar"] = lstar_table(V, X, rng)
    R["etf"] = etf_tracking(V)
    u_fix, px_fix = corrected_yahoo(X, S)
    idx = X["u_y"].index
    # base actual de R13 (B) con los días R1/R3 corregidos; señal original
    HB = run_version(build(u_fix, px_fix.reindex(idx), history.rf_long(idx), "r13_s", signal=r13.SIGNAL))
    R["GD_R13_corregido"] = r13.stats_table({k: HB["R"][k] for k in RULES}, HB["d"]["rf"], *GD)
    # Yahoo corregido (R1/R3 + pares a mano, a posteriori) con la construcción de D
    u_m, px_m = corrected_yahoo(X, S, manual_fix(S))
    Dc = run_version(build(u_m, px_m.reindex(idx), history.rf_long(idx), "sint", signal=r13.SIGNAL))
    R["brecha_crsp"] = crsp_gap(V, Dc, start, end)
    R["ab"] = ab_split(V)
    return R


# ───────────────────────────── informe ─────────────────────────────

def _date(d) -> str:
    return pd.Timestamp(d).strftime("%-d-%-m-%Y")


def _ci(s: str) -> tuple[float, float]:
    lo, hi = s.strip("[]").split(",")
    return float(lo), float(hi)


def run():
    R = compute()
    X, S, V = R["X"], R["S"], R["V"]
    et, sc, st = R["eras"], R["sospechosos"], R["sabados_tabla"]
    vc, vd, vg, lt, etf = R["versiones_CAGR"], R["versiones_maxDD"], R["versiones_GD"], R["lstar"], R["etf"]
    rc, sa, cmp_ = R["rf"], R["sabados"], R["comparación"]
    start, end = R["start"], R["end"]
    lab_all = R["periodos"][0][0]
    E90, EALL = et.index[0], et.index[-1]
    later = et.iloc[1:4]
    v = VERSIONS
    g = X["gspc"]
    rep = Report("r14_data_audit", "R14 — Auditoría de la historia larga: Yahoo/Shiller frente a CRSP")
    rep.p(__doc__.split("\n", 2)[2])
    rep.p(f"Todas las comparaciones van del {_date(X['u_y'].index[0])} al {_date(end)} (último dato de "
          "French) y usan el calendario de Yahoo: las sesiones de CRSP que no tienen fecha en Yahoo "
          "(los sábados) se componen con la sesión siguiente. Ejecución como en R13: señal al cierre "
          "de t, orden al cierre de t+1, 5 pb por unidad de rotación.")

    # ── resumen
    s90 = et.loc[E90]
    pr = S["pairs"]
    d0 = pd.Timestamp("1935-04-16")
    i0 = g.index.get_loc(d0)
    typo = (f"el {_date(d0)} Yahoo da {num(g.iloc[i0], 2)} entre {num(g.iloc[i0 - 1], 2)} y "
            f"{num(g.iloc[i0 + 1], 2)}, mientras CRSP se mueve un {pct(X['c'].loc[d0], 1)}")
    rev_dates = ", ".join(_date(x) for x in pr.fecha) if len(pr) else "ninguno"
    C, D_, H, I = v["C"], v["D"], v["H"], v["I"]
    dH = ((vc.loc[H] - vc.loc[D_]) * 100)   # H e I son D (señal original) con los días corregidos
    dI = ((vc.loc[I] - vc.loc[D_]) * 100)
    ex = {k: V[k]["E"]["LRS 3x (original)"].shift(r13.EXEC_LAG) for k in ("C", "D")}
    state = lambda k, day: "dentro" if ex[k].loc[day] > 0 else "fuera"  # noqa: E731
    d1 = g.index[i0 + 1]
    both_days = lambda k: (f"{state(k, d0)} los dos días" if state(k, d0) == state(k, d1)  # noqa: E731
                           else f"{state(k, d0)} el primero y {state(k, d1)} el segundo")
    g3 = st.loc["3x"]
    sign3 = "habría sumado" if g3["efecto_anual_medio_pp"] > 0 else "habría restado"
    rs = rc["resumen"]
    dG = ((vc.loc[v["G"]] - vc.loc[v["A"]]) * 100).abs().max()
    cA = cmp_[lab_all]
    rank_y = list(cA.CAGR_Yahoo.sort_values(ascending=False).index)
    rank_c = list(cA.CAGR_CRSP.sort_values(ascending=False).index)
    same_rank = rank_y == rank_c
    gd = cmp_["Gran Depresión"]
    A, B = v["A"], v["B"]
    lrs_dd = vd["LRS 3x"]
    ly = lt.loc["Yahoo · 1928–2026"]
    lc = lt.loc["CRSP · 1928–2026"]
    l26 = lt.loc["CRSP · 1926–2026"]
    lno = lt[lt.index.str.contains("sin la Gran")]
    lgd = lt[lt.index.str.contains("1928–1938")]
    full_rows = lt.loc[["Yahoo · 1928–2026", "CRSP · 1928–2026", "CRSP · 1926–2026"]]
    two_s = LSTAR_B in lt.columns                 # diferenciales distintos en A y en B
    L_hi, L_lo = (LSTAR_A, LSTAR_B) if two_s else (LSTAR, LSTAR)
    cis = [_ci(x) for c in lt.columns if c.startswith("IC90") for x in full_rows[c]]
    ci_lo, ci_hi = min(c[0] for c in cis), max(c[1] for c in cis)
    gd_r13c = R["GD_R13_corregido"]
    lev = ["3x", "LRS 3x", "½ Kelly GJR", "LRS-Kelly"]
    dAB = (vc.loc[A, lev] - vc.loc[B, lev]) * 100          # de la base anterior a la actual
    split_AB = "A′" in V                                  # si el diferencial de A y B coincide, no hay A′ ni B′
    if split_AB:
        Ap, Bp = v["A′"], v["B′"]
        dAAp = (vc.loc[A, lev] - vc.loc[Ap, lev]) * 100    # sólo el diferencial
        dApB = (vc.loc[Ap, lev] - vc.loc[B, lev]) * 100    # después, sólo la serie de 1960–2009
        dBpA = (vc.loc[Bp, lev] - vc.loc[A, lev]) * 100    # sólo la serie, con el diferencial de A
    fall = "bajan" if dAB.min() > 0 else "cambian"
    ab = R["ab"]
    gap_R3 = (ab["s_B"] - (ab["s_R3"] - ab["f_rf_1960"])) * 100   # coste del swap que no paga R3 en 1960–2009
    gap_s = (ab["s_B"] - ab["s_A"]) * 100
    G = R["brecha_crsp"]
    gd_, gl_, gp_, rp_ = G["diferencias"], G["niveles"], G["props"], G["reparto"]
    full_ = G["full"]

    def prop(era, src, col):
        return gp_.loc[(era, src), col]

    e57 = "1957–1987"
    e38 = [e for e, _ in gp_.index if e.endswith("–1957")][0]
    kel = cA.loc["½ Kelly GJR (sin tendencia)"]
    exp_k = {s: V[s]["E"]["½ Kelly GJR (sin tendencia)"].loc[start:end].mean() for s in ("D", "E")}
    vq, e3 = G["cociente_VR"], G["épocas_3x"]
    e88 = [e for e, _ in gp_.index if e.startswith("1988")][0]
    lab_l = f"{X['u_y'].index[0].year}–{end.year}"
    lq = {s: ((V[s]["d"]["u"].loc[:end].mean() - V[s]["d"]["f"].loc[:end].mean()) * 252,
              V[s]["d"]["u"].loc[:end].var() * 252) for s in ("D", "E")}

    rep.h("Resumen")
    bullets = [
        f"**A escala anual, la serie larga es fiable.** En 1928–2026 el S&P 500 de Yahoo/Shiller rinde "
        f"{pct(et.loc[EALL, 'CAGR_Yahoo'], 2)} anual y el mercado CRSP {pct(et.loc[EALL, 'CAGR_CRSP'], 2)}, "
        f"con correlación mensual {num(et.loc[EALL, 'corr_mensual'], 3)}. La época del S&P 90 (1928–1957) "
        f"es la más ruidosa: correlación diaria {num(s90.corr_diaria, 3)} (entre {num(later.corr_diaria.min(), 3)} "
        f"y {num(later.corr_diaria.max(), 3)} después), error de seguimiento {pct(s90.error_seguimiento)} "
        f"anual (entre {pct(later.error_seguimiento.min())} y {pct(later.error_seguimiento.max())} después), "
        f"volatilidad de {pct(s90.vol_Yahoo)} en Yahoo frente a {pct(s90.vol_CRSP)} en CRSP, y Yahoo rinde "
        f"{num(s90['Δ_CAGR_pp'], 2)} pp/año más.",
        f"**^GSPC tiene cierres mal escritos.** La regla R1, fijada antes de mirar, marca {len(pr)} pares "
        f"de reversión ({rev_dates}); p. ej., {typo}. Revisando a mano la lista de R2 aparecen al menos "
        f"{len(MANUAL_PAIRS)} pares más con la misma firma que R1 no recoge (la regla es conservadora); "
        "varios cierres erróneos son números redondos (15,00 el 16-9-1936, el 3-9-1946 y el 24-1-1949; "
        f"14,00 el 4-2-1948). R3 (cierres congelados) no marca ninguno.",
        f"**Esos errores {'inflan' if vg.loc[D_, 'LRS 3x'] > vg.loc[H, 'LRS 3x'] else 'mueven'} la LRS en la "
        f"Gran Depresión.** El {_date(d0)} (caída falsa) y el {_date(d1)} (rebote falso de "
        f"{pct(X['r_px'].loc[d1], 1)}) la LRS 3x original estaba {both_days('D')} (en el mercado); "
        f"la de precio, {both_days('C')}. Corregir los días R1/R3 (H frente a D, señal "
        f"original) lleva la LRS 3x de 1928–1938 de {pct(vg.loc[D_, 'LRS 3x'])} a {pct(vg.loc[H, 'LRS 3x'])} "
        f"anual ({pct(vg.loc[I, 'LRS 3x'])} con los pares revisados a mano) y la LRS 1x de "
        f"{pct(vg.loc[D_, 'LRS 1x'])} a {pct(vg.loc[H, 'LRS 1x'])}; en la base actual de R13 (B, sobre la "
        f"serie sin corregir), la LRS 3x de la Gran Depresión pasa de {pct(vg.loc[B, 'LRS 3x'])} a "
        f"{pct(gd_r13c.loc['LRS 3x (original)', 'CAGR'])}. R13 ya corrige los cuatro cierres de R1 "
        f"(history.GSPC_BAD_CLOSES). En {lab_all} el efecto es "
        f"pequeño: entre {num(min(dH.min(), dI.min()), 2)} y {num(max(dH.max(), dI.max()), 2)} pp/año según "
        "la estrategia.",
        f"**Sábados.** Yahoo no tiene las {sa['n_sábados']} sesiones de sábado anteriores a 1952 y su lunes "
        f"acumula dos sesiones (correlación del lunes de Yahoo con sábado + lunes de CRSP "
        f"{num(sa['corr_lunes_con_sábado'], 3)}; con el lunes solo, {num(sa['corr_lunes_solo'], 3)}). En un "
        f"fondo 3x de reajuste diario, reajustar también el sábado {sign3} "
        f"{num(abs(g3['efecto_anual_medio_pp']), 2)} pp/año de media en 1928–1952 con las sesiones de CRSP "
        f"(máximo {num(g3['máx_|efecto|_en_un_año_pp'], 2)} pp en {int(g3['año_del_máximo'])}; "
        f"{num(st.loc['LRS 3x (sesiones invertidas)', 'efecto_anual_medio_pp'], 2)} pp/año en la LRS 3x). "
        f"La cota de peor caso, Σ|L(L−1)·s·m|, es {num(g3['cota_peor_caso_anual_media_pp'], 2)} pp/año de "
        f"media y {num(g3['cota_peor_caso_máx_año_pp'], 2)} pp en el peor año.",
        f"**Tipo sin riesgo.** history.rf_long (letra a 3 meses) queda {num(rs['Δ_medio_pp'], 2)} pp/año por "
        f"encima del RF de French (1 mes) de media, con correlación mensual {num(rs['corr_mensual'], 3)}; "
        f"usar el de French mueve las cifras clave de R13 como mucho {num(dG, 2)} pp/año. El RF del fichero "
        f"diario de French está redondeado a 0,01% por sesión: sumado tal cual sesga el tipo "
        f"{num(rs['sesgo_redondeo_medio_pp'], 2)} pp/año de media y hasta {num(rs['sesgo_redondeo_máx_pp'], 2)} "
        f"pp en {rs['año_sesgo_máx']}; aquí se reconstruye a partir del RF mensual.",
        f"**Con CRSP como índice, el orden de las estrategias {'no cambia' if same_rank else 'cambia'}.** "
        f"Con la misma construcción (versiones D y E), las apalancadas rinden más con CRSP en {lab_all}: "
        f"3x {sgn(cA.loc['3x comprar y mantener', 'Δ_CAGR_pp'])} pp/año, LRS 3x "
        f"{sgn(cA.loc['LRS 3x (original)', 'Δ_CAGR_pp'])}, ½ Kelly GJR "
        f"{sgn(cA.loc['½ Kelly GJR (sin tendencia)', 'Δ_CAGR_pp'])} y LRS-Kelly "
        f"{sgn(cA.loc['LRS-Kelly', 'Δ_CAGR_pp'])}, frente a {sgn(cA.loc['1x comprar y mantener', 'Δ_CAGR_pp'])} "
        f"el 1x (el motivo, en el punto siguiente). La caída máxima de la LRS 3x (original) es "
        f"{pct(cA.loc['LRS 3x (original)', 'maxDD_CRSP'])} con CRSP frente a "
        f"{pct(cA.loc['LRS 3x (original)', 'maxDD_Yahoo'])} con Yahoo (con la señal de precio, "
        f"{pct(vd.loc[v['F'], 'LRS 3x'])} con CRSP frente a {pct(vd.loc[v['C'], 'LRS 3x'])} con Yahoo, y "
        f"{pct(vc.loc[v['F'], 'LRS 3x'])} frente a {pct(vc.loc[v['C'], 'LRS 3x'])} anual). En la Gran Depresión el 3x queda en "
        f"ruina con las dos fuentes (caída de {pct(gd.loc['3x comprar y mantener', 'maxDD_Yahoo'], 2)} y "
        f"{pct(gd.loc['3x comprar y mantener', 'maxDD_CRSP'], 2)}).",
        f"**La ventaja del apalancado con CRSP es un artefacto del índice amplio, no incertidumbre de la "
        f"fuente.** En {full_} la media aritmética del 3x es la misma con las dos fuentes "
        f"({pct(gl_.loc['3x · Yahoo', 'media_aritm'], 2)} con Yahoo y {pct(gl_.loc['3x · CRSP', 'media_aritm'], 2)} "
        f"con CRSP; por épocas difiere por composición, pero se compensa); la brecha sale de la penalización por varianza diaria, ½σ² de "
        f"{num(gl_.loc['3x · Yahoo', '½σ²'] * 100, 2)} frente a {num(gl_.loc['3x · CRSP', '½σ²'] * 100, 2)} pp, "
        f"que explica {num(gd_.loc['3x', 'por_½σ²_pp'], 2)} de los {num(gd_.loc['3x', 'Δ_media_log_pp'], 2)} pp "
        f"de rentabilidad log. CRSP tiene menos varianza diaria que Yahoo (volatilidad "
        f"{pct(prop(full_, 'CRSP', 'vol_diaria'), 2)} frente a {pct(prop(full_, 'Yahoo', 'vol_diaria'), 2)}) "
        f"pero casi la misma semanal ({pct(prop(full_, 'CRSP', 'vol_semanal'), 2)} frente a "
        f"{pct(prop(full_, 'Yahoo', 'vol_semanal'), 2)}), porque sus rentabilidades diarias están más "
        f"autocorrelacionadas (orden 1: {num(prop(full_, 'CRSP', 'AR1'), 3)} frente a "
        f"{num(prop(full_, 'Yahoo', 'AR1'), 3)}; en {e57}, {num(prop(e57, 'CRSP', 'AR1'), 3)} frente a "
        f"{num(prop(e57, 'Yahoo', 'AR1'), 3)} con la misma volatilidad semanal, "
        f"{pct(prop(e57, 'CRSP', 'vol_semanal'), 2)} y {pct(prop(e57, 'Yahoo', 'vol_semanal'), 2)}). Es la "
        f"firma de la negociación no sincrónica: los valores poco líquidos del índice amplio cierran con "
        f"precios desfasados que ningún fondo podría negociar. Repartida sobre la varianza del índice "
        f"(aproximación que suma {num(rp_['suma'], 2)} pp), la brecha de ½σ² del 3x son unos "
        f"{num(rp_['autocorrelación'], 2)} pp de esa autocorrelación, {num(rp_['varianza_semanal'], 2)} una "
        f"varianza semanal algo mayor del S&P de Yahoo (sobre todo el S&P 90 de {e38}: volatilidad semanal "
        f"{pct(prop(e38, 'Yahoo', 'vol_semanal'))} frente a {pct(prop(e38, 'CRSP', 'vol_semanal'))}) y "
        f"{num(rp_['errores_Yahoo'], 2)} los cierres erróneos de Yahoo, que inflan su varianza diaria (en "
        f"{e38} su volatilidad diaria, {pct(prop(e38, 'Yahoo', 'vol_diaria'), 2)}, supera a la semanal, "
        f"{pct(prop(e38, 'Yahoo', 'vol_semanal'), 2)}; corregidos los pares, "
        f"{pct(prop(e38, 'Yahoo corregido', 'vol_diaria'), 2)} y "
        f"{pct(prop(e38, 'Yahoo corregido', 'vol_semanal'), 2)}). La LRS 3x sigue el mismo patrón "
        f"({num(gd_.loc['LRS 3x', 'por_½σ²_pp'], 2)} de {num(gd_.loc['LRS 3x', 'Δ_media_log_pp'], 2)} pp por "
        f"varianza). No indica que Yahoo infravalore los LETF: para un fondo sobre el S&P 500 la referencia "
        f"es Yahoo (corregido), y la brecha D−E no debe leerse como margen de error de la fuente. En las "
        f"estrategias Kelly la brecha es menor y no pasa por la penalización: en ½ Kelly GJR entra por la "
        f"media aritmética ({sgn(gd_.loc['½ Kelly GJR', 'por_media_aritm_pp'])} pp con la misma "
        f"exposición media, {num(exp_k['D'], 2)} y {num(exp_k['E'], 2)}), y en LRS-Kelly casi desaparece "
        f"frente a Yahoo corregido ({sgn(gd_.loc['LRS-Kelly', 'Δ_CAGR_frente_a_Yahoo_corregido_pp'])} pp).",
        f"**Dos decisiones de construcción pesan tanto o más que la fuente.** (i) El 3x de 1960–2009. La "
        f"base anterior de R13 (A) pasaba a r12.make como «ETF real» synthetic_actual, que desde 1960 ya es "
        f"el sintético de R3: financiado al T-bill + {num(ab['s_R3'] * 100, 2)}% cuando f superaba al T-bill "
        f"en {num(ab['f_rf_1960'] * 100, 2)} pp/año, así que dejaba de pagar {num(gap_R3, 2)} pp/año de coste "
        "del swap por unidad de exposición. La base actual (B) pasa a make() sólo el UPRO real: el 3x de "
        f"1960–2009 es el sintético de R12, financiado a {fplus(ab['s_B'])}. "
        + (f"El diferencial ya no cambia: make() lo recalibra sólo con el periodo del ETF real, así que A y B "
           f"usan el mismo (s = {num(ab['s_B'] * 100, 3)}%) y esa corrección ya está en la base. "
           if not split_AB else
           f"El diferencial también cambia (s = {num(ab['s_A'] * 100, 3)}% en A y {num(ab['s_B'] * 100, 3)}% "
           f"en B; {num(gap_s, 2)} pp/año por unidad de exposición). ")
        + f"De A a B, el 3x va de {pct(vc.loc[A, '3x'])} a {pct(vc.loc[B, '3x'])}, la LRS 3x de "
        f"{pct(vc.loc[A, 'LRS 3x'])} a {pct(vc.loc[B, 'LRS 3x'])}, ½ Kelly GJR de {pct(vc.loc[A, '½ Kelly GJR'])} "
        f"a {pct(vc.loc[B, '½ Kelly GJR'])} y LRS-Kelly de {pct(vc.loc[A, 'LRS-Kelly'])} a "
        f"{pct(vc.loc[B, 'LRS-Kelly'])} (media aritmética del 3x en 1960–2009: {pct(ab['x3_A'])} en A y "
        f"{pct(ab['x3_B'])} en B). (ii) Calcular la media de 200 sesiones sobre la rentabilidad total (la "
        f"señal original de Gayed y Bilello) o sobre el precio: la LRS 3x original rinde "
        f"{num((vc.loc[v['C'], 'LRS 3x'] - vc.loc[v['D'], 'LRS 3x']) * 100, 2)} pp/año menos que la de precio "
        f"con Yahoo y {num((vc.loc[v['F'], 'LRS 3x'] - vc.loc[v['E'], 'LRS 3x']) * 100, 2)} pp menos con CRSP, "
        f"y la caída máxima de la LRS 3x va de {pct(lrs_dd.max())} a {pct(lrs_dd.min())} según la versión.",
        f"**L\\* de Kelly (aproximado, financiado a f).** Con toda la historia 1928–2026: Yahoo "
        f"{num(ly[L_hi])} y CRSP {num(lc[L_hi])}; CRSP desde 1926, {num(l26[L_hi])}"
        + (f" (con el diferencial corregido, {num(ly[L_lo])}, {num(lc[L_lo])} y {num(l26[L_lo])})" if two_s else "")
        + f". Sin la Gran Depresión (1938–2026) sube a {num(lno[L_lo].min())}–{num(lno[L_hi].max())}; "
        f"en 1928–1938 es {num(lgd[L_lo].min())}–{num(lgd[L_hi].max())}, con un IC90 que "
        f"incluye el cero. Los IC90 de la historia completa son anchos: sus extremos van de "
        f"{num(ci_lo)} a {num(ci_hi)}.",
    ]
    rep.p("\n".join(f"- {b}" for b in bullets))

    # ── 1
    rep.h("1. Yahoo/Shiller frente a CRSP: correlación y diferencias por épocas")
    rep.table(et, pct={"error_seguimiento", "error_seg_sin_R1R2", "CAGR_Yahoo", "CAGR_CRSP",
                       "vol_Yahoo", "vol_CRSP"}, digits=3)
    rep.p(f"error_seg_sin_R1R2 excluye los días que marcan R1 y R2 ({int(S['flagged'].sum())} en total). "
          f"En 1928–1957 el error de seguimiento baja de {pct(s90.error_seguimiento)} a "
          f"{pct(s90.error_seg_sin_R1R2)}: buena parte del exceso de ruido de esa época está en unos pocos "
          f"días. Lo que queda es del orden del de 1993–2026 con SPY ({pct(et.iloc[3].error_seguimiento)}), "
          "época de datos limpios en la que la diferencia es sólo de composición (500 valores frente a todo "
          "el mercado, Nasdaq incluido). La correlación semanal es más alta que la diaria en todas las épocas: "
          "parte de la discrepancia diaria son precios desfasados de los valores menos líquidos de CRSP, no "
          "errores.")
    rep.h("Calendario", 3)
    rep.table(R["calendario"], digits=0)
    rep.p("Salvo los sábados, los calendarios coinciden exactamente: no hay fechas de Yahoo sin sesión en "
          "CRSP ni laborables de CRSP sin fecha en Yahoo. El periodo con dos o más sesiones de más es el "
          "3-1-1928, que recoge también el sábado 31-12-1927.")
    rep.h("Diferencias anuales por décadas (años completos)", 3)
    rep.table(R["decadas"], digits=2)
    rep.h("Los diez años con más diferencia", 3)
    yr = R["anual"]
    yr.to_csv(rep.dir / "anual_yahoo_crsp.csv")
    full = yr[yr.sesiones > 200]
    top = full.reindex(full.dif_pp.abs().sort_values(ascending=False).index[:10])
    rep.table(top[["Yahoo", "CRSP", "dif_pp"]], pct={"Yahoo", "CRSP"}, digits=2)
    dec = R["decadas"]
    dec_rest = dec.drop(index=[i for i in dec.index if i.startswith("192")])
    sd_eras = et.iloc[:4]["dt_dif_anual_pp"]
    rep.p(f"La desviación típica de la diferencia anual va de {num(sd_eras.min(), 1)} a {num(sd_eras.max(), 1)} "
          "pp en todas las épocas, también con SPY: año a año el S&P 500 y el mercado total se separan por "
          "composición tanto como por errores. Fuera de 1928–1929 (Yahoo "
          f"{num(dec.loc['1920s', 'dif_media_pp'], 1)} pp/año por encima, dos años), las medias por década van "
          f"de {num(dec_rest.dif_media_pp.min(), 2)} a {num(dec_rest.dif_media_pp.max(), 2)} pp, sin signo "
          "dominante. La serie anual completa está en anual_yahoo_crsp.csv.")
    fig_difference(X, yr, rep)
    rep.p("*Figura 1.* Arriba, diferencia acumulada entre la rentabilidad total del S&P 500 de "
          "Yahoo/Shiller y la del mercado CRSP, en puntos logarítmicos (+10 significa que Yahoo acumula "
          "un 10% más). Las líneas de puntos marcan los cambios de fuente de la serie de Yahoo: S&P 500 de "
          "500 valores desde el 4-3-1957, ^SP500TR desde 1988 y SPY desde 1993. En medio, la diferencia "
          "de cada año natural. Abajo, la correlación móvil de las rentabilidades diarias (ventana de un "
          "año) y semanales (52 semanas).")

    # ── 2
    rep.h("2. Días sospechosos (reglas fijadas de antemano)")
    rep.table(sc, digits=0)
    if len(pr):
        rep.h("R1: pares de reversión", 3)
        rep.table(pr.set_index("fecha"), pct={"Yahoo_t", "Yahoo_t+1", "CRSP_t", "CRSP_t+1"}, digits=1)
        rep.p(f"Los {len(pr)} pares son cierres aislados fuera de línea con sus dos vecinos: {typo}. "
              "Son errores de transcripción de ^GSPC en Yahoo, no movimientos del mercado.")
    if len(S["frz_df"]):
        rep.h("R3: cierres congelados", 3)
        rep.table(S["frz_df"], pct={"Yahoo", "CRSP"}, digits=1)
    else:
        n0, n0_90 = int(sc.iloc[-1]["días_con_r=0"]), int(sc.iloc[0]["días_con_r=0"])
        rep.p(f"R3 no marca ningún día. De los {n0} días con cierre sin cambio en ^GSPC, {n0_90} son de "
              f"1928–1957, cuando el índice valía entre {g.loc[:'1957-03-01'].min():.0f} y "
              f"{g.loc[:'1957-03-01'].max():.0f} puntos y se publicaba con dos decimales; todos "
              "coinciden con días tranquilos en CRSP.")
    rep.h("R2: discrepancias aisladas (las 20 mayores)", 3)
    top_iso = S["iso_df"].reindex(S["iso_df"].puntuación.abs().sort_values(ascending=False).index[:20])
    top_iso.index = top_iso.index.date
    rep.table(top_iso, pct={"Yahoo", "CRSP"}, digits=2)
    S["iso_df"].to_csv(rep.dir / "discrepancias_R2.csv")
    p2 = R["pares_R2"]
    p2.to_csv(rep.dir / "pares_R2.csv", index=False)
    rep.p(f"R2 marca {int(sc.iloc[-1]['R2_discrepancias_aisladas'])} días (lista completa en "
          f"discrepancias_R2.csv). No todos son errores: el 20-10-1987 el S&P 500 subió un 5,3% mientras el "
          f"mercado amplio apenas se movió, y el 28/29-10-1929 son días genuinos de pánico. Entre ellos hay "
          f"{len(p2)} pares de días consecutivos con discrepancias de signo opuesto (pares_R2.csv), la firma "
          f"de un cierre mal escrito; {int(p2.en_R1.sum())} ya están en R1. Revisados a mano los cierres de "
          f"los de más de 3%, al menos {len(MANUAL_PAIRS)} tienen esa firma y R1 no los recoge porque la σ de "
          f"63 sesiones era alta o CRSP se movió más de 2σ: "
          f"{', '.join(_date(x) for x in MANUAL_PAIRS)}. Por la regla fijada de antemano no entran en la "
          "corrección principal (H); sólo en la sensibilidad I, declarada a posteriori. Hay más pares "
          "pequeños (±1–2%) con la misma forma en 1961–1965 que no se han revisado uno a uno.")
    fig_scatter(S, rep)
    rep.p("*Figura 2.* Cada punto es una sesión: en horizontal, el movimiento de CRSP y en vertical el "
          "de ^GSPC, ambos en desviaciones típicas de sus 63 sesiones anteriores (recortados a ±15). Los "
          "puntos sobre la diagonal son días en que las dos fuentes coinciden. Los rombos rojos son los "
          "pares de reversión de R1 (^GSPC se mueve más de 4σ y CRSP menos de 2σ) y los puntos azules, "
          "las discrepancias aisladas de R2. Arriba la época del S&P 90; abajo, desde 1957.")

    # ── 3
    rep.h("3. Sesiones de sábado antes de 1952")
    rep.table(st, digits=2)
    rep.p(f"El NYSE abrió los sábados (sesiones de dos horas) hasta 1952; CRSP las recoge y Yahoo no. "
          f"La volatilidad del sábado fue de {pct(sa['vol_sábado_anual'])} anualizada, frente a "
          f"{pct(sa['vol_laborable_anual'])} en los laborables. Para un fondo L× que reajusta cada sesión, "
          "fundir el sábado s con el lunes m cambia la riqueza en L(L−1)·s·m (6·s·m para el 3x): el "
          "signo depende de si sábado y lunes se mueven en el mismo sentido. La tabla mide el efecto real "
          "con las sesiones de CRSP (efecto positivo: el fondo que reajusta el sábado gana más, así que la "
          "simulación sobre el calendario de Yahoo lo infravalora) y la cota de peor caso, que supone "
          "que todas las diferencias del año tienen el mismo signo. No incluye financiación ni costes: un "
          "fondo real devenga la financiación por día natural, y reajustar el sábado le costaría una "
          "operación más por semana. La autocorrelación sábado–lunes de CRSP incluye precios desfasados "
          "de valores poco líquidos; en un índice de 90 grandes valores sería menor, y el efecto también.")
    fig_saturdays(sa, rep)
    rep.p("*Figura 3.* Arriba, para cada año de 1928 a 1952, diferencia de rentabilidad (logarítmica, en "
          "puntos porcentuales) entre un fondo 3x que reajusta tras cada sesión, sábado incluido, y uno "
          "que funde sábado y lunes como hace la serie de Yahoo. Las barras son el efecto real con las "
          "sesiones de CRSP y la línea gris, la cota de peor caso (todas las diferencias del año con el "
          "mismo signo). Abajo, el efecto acumulado para un 2x y un 3x.")

    # ── 4
    rep.h("4. Tipo sin riesgo: French frente a history.rf_long")
    tb = rc["tabla"]
    rep.table(tb, pct={"French_1m", "rf_long", "f_financiación"}, digits=2)
    rep.p(f"Rentabilidad anual media por década (años completos). rf_long usa letras a 3 meses (NBER "
          f"hasta 1934, TB3MS de FRED hasta 1960, ^IRX después, con un mes de retraso antes de 1960) y "
          f"French, letras a 1 mes: la diferencia media, {num(rs['Δ_medio_pp'], 2)} pp/año, es del orden "
          "de la prima por plazo entre ambas. f es el tipo al que R12–R13 financian el apalancamiento "
          f"(call money hasta 1954, fed funds después): en los años treinta y cuarenta pagaba "
          f"{num((tb.loc['1930s', 'f_financiación'] - tb.loc['1930s', 'French_1m']) * 100, 2)} y "
          f"{num((tb.loc['1940s', 'f_financiación'] - tb.loc['1940s', 'French_1m']) * 100, 2)} pp más que la "
          "letra, lo que encarece el 3x sintético de esa época. La última columna es el sesgo de sumar el RF "
          "diario de French, redondeado a 0,01% por sesión: en los años cuarenta el tipo diario era de "
          "0,002% y el redondeo lo borra entero "
          f"({num(tb.loc['1940s', 'sesgo_RF_diario_redondeado_pp'], 2)} pp sobre "
          f"{num(tb.loc['1940s', 'French_1m'] * 100, 2)}%).")
    fig_rf(rc, rep)
    rep.p("*Figura 4.* Arriba, rentabilidad de cada año de la letra del Tesoro a 1 mes (French), de la "
          "letra a 3 meses que usa el laboratorio (rf_long) y del tipo de financiación del apalancamiento "
          "(f). Abajo, diferencia anual entre rf_long y French (barras) y sesgo de sumar el RF diario "
          "redondeado del fichero de French frente a su RF mensual (línea roja).")

    # ── 5
    rep.h("5. Robustez: las cifras clave con CRSP como índice")
    rep.p("Para comparar sólo el índice, las versiones D (Yahoo) y E (CRSP) se construyen igual: 3x "
          "sintético en toda la muestra (no hay ETF real sobre CRSP), financiado a f con el diferencial "
          "recalibrado sólo con UPRO real, T-bill de rf_long, y la señal de la LRS original, sobre el índice "
          "de rentabilidad total (con CRSP no hay otra: French no publica el mercado diario sin dividendos). "
          "El GJR-GARCH y la prima de Kelly se estiman en cada serie con sus propios datos pasados.")
    for lab, t in cmp_.items():
        title = ("Yahoo frente a CRSP en la Gran Depresión (1928-10 → 1938)" if lab == "Gran Depresión"
                 else f"Yahoo frente a CRSP, {lab}")
        rep.h(title, 3)
        rep.table(t, pct={"CAGR_Yahoo", "CAGR_CRSP", "maxDD_Yahoo", "maxDD_CRSP"}, digits=2)
    rep.h("Del máximo de 1929 al mínimo de 1932", 3)
    rep.table(R["crash"], pct=set(R["crash"].columns), digits=4)
    fig_robust(V, start, end, rep)
    rep.p(f"*Figura 5.* Arriba, evolución de 10.000 $ desde el {_date(start)} en cada estrategia, con el "
          "S&P 500 de Yahoo (línea continua) y con el mercado CRSP (discontinua) como índice; misma "
          "construcción en ambos (versiones D y E). Abajo, diferencia acumulada CRSP − Yahoo de cada "
          "estrategia en puntos logarítmicos: una línea que sube indica que la estrategia rinde más con "
          "CRSP. Que suban las apalancadas y no el 1x no es una ventaja real del apalancamiento sobre CRSP: "
          "se explica en el apartado siguiente y en la figura 7.")
    fig_depression(V, rep)
    rep.p(f"*Figura 6.* Arriba, 10.000 $ invertidos del {_date(GD[0])} al {_date(GD[1])} en el índice (1x), "
          "en el 3x y en la LRS 3x, con el S&P 90 de Yahoo (continua) y con el mercado CRSP (discontinua). "
          "Abajo, caída desde máximos de cada una. Versiones D y E: señal original (rentabilidad total) y 3x "
          "sintético en toda la muestra, así que la LRS 3x no coincide exactamente con la de R13, que usa los "
          "cierres corregidos.")

    rep.h("Por qué el apalancado rinde más con CRSP", 3)
    rep.p("La rentabilidad log media de cada estrategia es exactamente su media aritmética menos la "
          "penalización por varianza ½σ² (de las rentabilidades diarias) menos un resto de orden superior. "
          "Si la brecha CRSP − Yahoo viniera de la fuente (un índice que sube más o menos), aparecería en la "
          "media aritmética; si viene de cuánto se mueve el índice de un día para otro, aparece en ½σ². "
          f"Tabla en % anual, {_date(start)} → {_date(end)}; «Yahoo corregido» es D con los días R1/R3 y "
          "los 7 pares revisados a mano sustituidos por CRSP (a posteriori):")
    rep.table(gl_, pct=set(gl_.columns), digits=2)
    rep.p("Diferencia CRSP − Yahoo (versión E menos D) y de dónde sale, en pp/año; las tres columnas «por_» "
          "suman Δ_media_log_pp:")
    rep.table(gd_, digits=2)
    rep.p("Propiedades de cada índice por épocas: volatilidad diaria y semanal anualizadas (rentabilidades "
          "log), razón de varianzas VR = var(semana)/(sesiones por semana · var(día)), que vale 1 sin "
          "autocorrelación, y autocorrelación de orden 1 de las rentabilidades diarias:")
    rep.table(G["tabla_props"], pct={"vol_diaria", "vol_semanal"}, digits=3)
    rep.p(f"En {full_} la media aritmética del 3x es prácticamente la misma con las dos fuentes; lo que "
          f"cambia es ½σ². "
          f"Un fondo que reajusta cada día paga la varianza DIARIA del índice (≈ ½·9·σ²_diaria), y la de "
          f"CRSP es menor aunque su varianza semanal sea casi igual: su VR es mayor que 1 "
          f"({num(prop(full_, 'CRSP', 'VR'), 3)} en {full_}, {num(prop(e57, 'CRSP', 'VR'), 3)} en {e57}) "
          f"porque sus rentabilidades diarias están autocorrelacionadas. La VR implícita en las "
          f"autocorrelaciones de los retardos 1 a 4, 1 + 2·Σ(1 − k/5)·ρ_k, queda por encima de la observada "
          f"antes de 1988 en las dos series por igual (en torno a un 10%, por motivos que no he aislado), "
          f"pero el cociente CRSP/Yahoo "
          f"coincide en todas las épocas: {num(vq.loc[e57, 'implícita'], 3)} implícito frente a "
          f"{num(vq.loc[e57, 'observada'], 3)} observado en {e57}, {num(vq.loc[e38, 'implícita'], 3)} frente a "
          f"{num(vq.loc[e38, 'observada'], 3)} en {e38} y {num(vq.loc[full_, 'implícita'], 3)} frente a "
          f"{num(vq.loc[full_, 'observada'], 3)} en {full_}. La menor varianza diaria de CRSP es, por tanto, "
          "la que corresponde a su mayor autocorrelación. La autocorrelación positiva de un índice amplio ponderado por capitalización es la "
          "firma clásica de la negociación no sincrónica (Scholes y Williams, 1977; Lo y MacKinlay, 1990): "
          "muchos valores de CRSP no cotizan al cierre y su último precio es de horas o días antes, así que "
          "el índice diario se suaviza y la noticia de hoy se reparte entre hoy y mañana. Un fondo real no "
          "puede negociar a esos precios desfasados: la menor penalización del 3x sobre CRSP es un artefacto "
          "de medición, no un rendimiento alcanzable, y tampoco indica que Yahoo infravalore los LETF sobre "
          "el S&P 500. Por eso la brecha D−E de las estrategias apalancadas no debe leerse como incertidumbre "
          "de la fuente.")
    rep.p(f"Reparto aproximado de la brecha de ½σ² del 3x ({num(gd_.loc['3x', 'por_½σ²_pp'], 2)} pp: ½σ² "
          f"con Yahoo menos ½σ² con CRSP) sobre la varianza del índice, σ²_diaria = W/VR con W la varianza por "
          f"sesión a escala semanal: cierres erróneos de Yahoo, {num(rp_['errores_Yahoo'], 2)} pp (σ² de "
          f"Yahoo menos la de Yahoo corregido); varianza semanal, {num(rp_['varianza_semanal'], 2)} pp (el "
          f"S&P de Yahoo es algo más volátil también por semanas, sobre todo el S&P 90 de {e38}); "
          f"autocorrelación de CRSP, {num(rp_['autocorrelación'], 2)} pp. Suman {num(rp_['suma'], 2)} pp. Los "
          f"cierres erróneos de Yahoo empujan en el mismo sentido: en {e38} su volatilidad diaria "
          f"({pct(prop(e38, 'Yahoo', 'vol_diaria'), 2)}) supera a la semanal "
          f"({pct(prop(e38, 'Yahoo', 'vol_semanal'), 2)}) y su VR queda por debajo de 1, la huella de los "
          f"pares de reversión; corregidos, las dos valen {pct(prop(e38, 'Yahoo corregido', 'vol_diaria'), 2)} "
          f"y {pct(prop(e38, 'Yahoo corregido', 'vol_semanal'), 2)}. Aun así pesan poco: el 3x con Yahoo "
          f"corregido rinde {pct(gl_.loc['3x · Yahoo corregido', 'CAGR'], 2)} frente a "
          f"{pct(gl_.loc['3x · Yahoo', 'CAGR'], 2)}.")
    rep.p("La misma descomposición del 3x por épocas (CRSP − Yahoo, pp/año):")
    rep.table(e3, digits=2)
    rep.p(f"La parte de varianza es positiva en todas las épocas y mayor donde las series se separan más: "
          f"{num(e3.loc[e38, 'por_½σ²_pp'], 2)} pp/año en {e38} (donde además el S&P 90 de Yahoo es más "
          f"volátil también por semanas y tiene los cierres erróneos), {num(e3.loc[e57, 'por_½σ²_pp'], 2)} en "
          f"{e57} y {num(e3.loc[e88, 'por_½σ²_pp'], 2)} desde 1988, cuando la autocorrelación es negativa en "
          f"las dos series pero menos en CRSP ({num(prop(e88, 'CRSP', 'AR1'), 3)} frente a "
          f"{num(prop(e88, 'Yahoo', 'AR1'), 3)}). La parte aritmética, en cambio, sí depende de la época "
          f"(de {sgn(e3.por_media_aritm_pp.min())} a {sgn(e3.por_media_aritm_pp.max())} pp/año): es la "
          "diferencia de composición entre el S&P y el mercado entero, que se compensa en el total. La "
          "brecha no se cierra después de 1988, sólo se estrecha.")
    rep.p(f"La LRS 3x sigue el mismo patrón ({num(gd_.loc['LRS 3x', 'por_½σ²_pp'], 2)} de "
          f"{num(gd_.loc['LRS 3x', 'Δ_media_log_pp'], 2)} pp por varianza). En las estrategias Kelly la brecha "
          f"es menor y no pasa por la penalización: en ½ Kelly GJR entra por la media aritmética "
          f"({sgn(gd_.loc['½ Kelly GJR', 'por_media_aritm_pp'])} pp) con la misma exposición media "
          f"({num(exp_k['D'], 2)} con Yahoo y {num(exp_k['E'], 2)} con CRSP), es decir, por el calendario de la "
          "exposición (la covarianza entre el peso y la rentabilidad), que no he descompuesto más; en "
          "LRS-Kelly casi desaparece frente a Yahoo corregido "
          f"({sgn(gd_.loc['LRS-Kelly', 'Δ_CAGR_frente_a_Yahoo_corregido_pp'])} pp).")
    fig_mechanism(G, rep)
    rep.p("*Figura 7.* Arriba, diferencia acumulada entre el 3x sobre CRSP y el 3x sobre Yahoo (versiones E "
          "y D) en puntos logarítmicos (negro), y sus dos términos principales: la parte debida a la media "
          "aritmética, Σ(r_CRSP − r_Yahoo) (azul), y la debida a la penalización por varianza, "
          "−½·Σ(r²_CRSP − r²_Yahoo) (rojo). Al final del periodo toda la brecha es penalización: la parte "
          "aritmética oscila entre unos −50 y +25 puntos con la composición (S&P frente a mercado entero) y acaba cerca de "
          "cero; el resto, de orden superior, es pequeño. Abajo, autocorrelación de orden 1 de las "
          "rentabilidades diarias de cada índice en ventanas móviles de tres años: la de CRSP (azul) queda "
          "por encima de la de Yahoo hasta mediados de los años 2000, que es cuando crece la parte de "
          "varianza de arriba; después las dos coinciden y esa parte deja de crecer.")

    rep.h("Descomposición: de R13 a CRSP paso a paso", 3)
    rep.p("A es la base anterior de R13 (hasta el último dato de French), reproducida a propósito para "
          "medir cada corrección: su 3x es UPRO real desde 2009, el sintético de R3 en 1960–2009 (financiado "
          f"al T-bill + {num(ab['s_R3'] * 100, 2)}%, sobre ^GSPC con un dividendo fijo) y el de R12 antes. B es "
          "la base actual: make() sólo con el UPRO real, así que el 3x de 1960–2009 es el sintético de R12 "
          "financiado a f + s. "
          + (f"Las dos usan el mismo diferencial (s = {num(ab['s_B'] * 100, 3)}%), porque make() lo recalibra "
             "sólo con el periodo del ETF real: esa corrección ya está en la base, y las versiones intermedias "
             "A′ y B′ coincidirían con A y B, así que no se muestran. "
             if not split_AB else
             f"B cambia también el diferencial (de {num(ab['s_A'] * 100, 3)}% a {num(ab['s_B'] * 100, 3)}%). A′ "
             "y B′ separan las dos correcciones: A′ es A con sólo el diferencial corregido y B′ es A con sólo "
             "el 3x de R12 en 1960–2009. ")
          + "C usa el sintético de R12 en toda la muestra con la señal de precio, y D lo mismo con la señal "
          "original (la media sobre la rentabilidad total, con el D/P de Shiller del mes anterior hasta 1988 "
          "para no mirar el futuro y ^SP500TR después). E aplica la construcción de D al índice CRSP, y F usa para la señal un "
          "precio aproximado de CRSP: rentabilidad total menos el D/P de Shiller del mes anterior (el D/P del "
          "S&P, no el de CRSP). G es A con el RF de French. H es D con los días R1/R3 sustituidos por CRSP, e "
          "I añade los pares revisados a mano. Todas las versiones salvo C y F usan la señal original.")
    rep.p(f"Rentabilidad anual compuesta, {_date(start)} → {_date(end)}:")
    rep.table(vc, pct=set(vc.columns), digits=2)
    rep.p("Caída máxima en el mismo periodo:")
    rep.table(vd, pct=set(vd.columns), digits=2)
    rep.p(f"Gran Depresión ({_date(GD[0])} → {_date(GD[1])}), rentabilidad anual compuesta (sólo reglas "
          "sin parámetros estimados):")
    rep.table(vg, pct=set(vg.columns), digits=2)
    rep.p(f"B es la base actual de R13 sobre la serie de Yahoo sin corregir (R13 usa los cierres corregidos): "
          f"1x {pct(vc.loc[B, '1x'])}, 3x {pct(vc.loc[B, '3x'])}, LRS 3x {pct(vc.loc[B, 'LRS 3x'])}, LRS-Kelly "
          f"{pct(vc.loc[B, 'LRS-Kelly'])} y ½ Kelly GJR {pct(vc.loc[B, '½ Kelly GJR'])}. De A a B las cifras de "
          f"las estrategias apalancadas {fall} entre {num(dAB.min(), 2)} y {num(dAB.max(), 2)} pp/año. "
          + (f"Es una sola corrección, la del 3x de 1960–2009: R3 lo financiaba a "
             f"{fplus(ab['s_R3'] - ab['f_rf_1960'])} y R12 lo financia a {fplus(ab['s_B'])} (en 1960–2009 f "
             f"superó al T-bill en {num(ab['f_rf_1960'] * 100, 2)} pp/año; en el periodo de UPRO, "
             f"{num(ab['f_rf_live'] * 100, 2)}). La media aritmética del 3x en 1960–2009 es "
             f"{pct(ab['x3_A'], 2)} en A y {pct(ab['x3_B'], 2)} en B: el 3x de la base anterior no pagaba en "
             "ese tramo parte del coste del swap que el UPRO real sí paga."
             if not split_AB else
             "Son dos correcciones y su reparto depende del orden, porque el diferencial sólo actúa donde el "
             "3x se construye con r12.letf: en A, sólo antes de 1960 (y en el Kelly). En el orden A → A′ → B, "
             f"sólo el diferencial resta entre {num(dAAp.min(), 2)} y {num(dAAp.max(), 2)} pp/año y el resto "
             f"({num(dApB.min(), 2)} a {num(dApB.max(), 2)}) es financiar el 3x de 1960–2009 a "
             f"{fplus(ab['s_B'])} en vez de a {fplus(ab['s_R3'] - ab['f_rf_1960'])}. En el orden A → B′ → B, "
             f"cambiar sólo la serie mueve {sgn(dBpA.min())} a {sgn(dBpA.max())} pp/año. La media aritmética del "
             f"3x en 1960–2009 es {pct(ab['x3_A'], 2)} en A, {pct(ab['x3_B′'], 2)} en B′ y {pct(ab['x3_B'], 2)} "
             "en B."))
    rep.p("La mayor sensibilidad de la LRS 3x no es a la fuente "
          "sino a la definición de la señal: con la media sobre rentabilidad total el índice queda por "
          "encima de su media más a menudo (los dividendos, de un 3–6% anual antes de 1960, la empujan), la "
          "estrategia sale más tarde y su caída máxima empeora. La LRS 3x es, además, la estrategia que más "
          "se mueve con cualquier cambio (entre "
          f"{pct(vc['LRS 3x'].min())} y {pct(vc['LRS 3x'].max())} anual y entre {pct(lrs_dd.max())} y "
          f"{pct(lrs_dd.min())} de caída máxima en las {len(vc)} versiones); las estrategias Kelly, las que menos "
          f"(½ Kelly GJR entre {pct(vc['½ Kelly GJR'].min())} y {pct(vc['½ Kelly GJR'].max())}; LRS-Kelly "
          f"entre {pct(vc['LRS-Kelly'].min())} y {pct(vc['LRS-Kelly'].max())}).")

    rep.h("El diferencial del swap: 3x sintético frente al ETF real", 3)
    rep.table(etf, pct={"CAGR_real", "CAGR_sintético"}, digits=3)
    d_old = etf.loc[[f"{t} · {ETF_OLD}" for t in ("UPRO", "TQQQ", "SOXL")], "Δ_aritm_anual_pp"].abs().max()
    rep.p("La calibración iguala medias aritméticas diarias, así que la columna que la valida es "
          "Δ_aritm_anual_pp: con el diferencial recalibrado sólo con el ETF real es cero en los tres "
          "fondos"
          + ("; con el de la base anterior, el sintético rinde de más. " if d_old > 0.005 else
             f", y con la construcción de la base anterior también (como mucho {num(d_old, 3)} pp): make() ya "
             "recalibra sólo con la vida del ETF real, así que las dos filas de cada fondo coinciden. ")
          + "En rentabilidad logarítmica el sintético de TQQQ y SOXL queda por debajo del real incluso bien "
          "calibrado: el ruido de seguimiento no es simétrico en fondos tan volátiles. Esto no afecta a "
          "las secciones anteriores, que sólo usan el S&P 500.")

    rep.h("L* con toda la historia", 3)
    rep.table(lt, digits=2)
    rep.p(f"{LSTAR} = (μ − f − s)/σ² con la muestra completa de cada ventana (estadístico ex post, no "
          "una regla operable) e IC90 por bootstrap estacionario con bloques de un año. "
          + (f"«{LSTAR_A}» usa el diferencial de la base anterior (A) y «{LSTAR_B}» el recalibrado con "
             "UPRO real. "
             if two_s else
             f"El diferencial es el de la base actual ({fplus(ab['s_B'])}; A y B lo comparten). ")
          + "CRSP desde 1926 se calcula sobre su propio "
          "calendario de laborables, con los sábados compuestos con la sesión siguiente. El L* de CRSP es "
          f"mayor que el de Yahoo sólo por el denominador: en {lab_l} su μ − f es incluso algo menor "
          f"({num(lq['E'][0] * 100, 2)} frente a {num(lq['D'][0] * 100, 2)} pp/año), pero su σ² diaria es "
          f"{num(lq['E'][1] * 100, 2)} frente a {num(lq['D'][1] * 100, 2)} pp, por la misma autocorrelación que "
          "abarata el 3x sobre CRSP; no es un L* alcanzable por un fondo real. El artículo cita el L* de R12 "
          "(aproximado y exacto, con la comisión del fondo), no éste.")

    rep.h("Salvedades")
    rep.p("\n".join(f"- {x}" for x in [
        "CRSP no es el S&P 500: hasta 1962 es todo el NYSE, luego añade AMEX y desde 1972 el Nasdaq. Una "
        "diferencia entre ambas series no es por sí sola un error; la referencia de lo normal la dan las "
        "épocas limpias (1988–2026).",
        "El mercado de French (Mkt-RF + RF) tiene dos decimales en porcentaje: el redondeo de Mkt-RF añade "
        "un ruido de ±0,005% por sesión, unos 5 pb anuales sin sesgo. El RF diario sí sesga y se ha "
        "sustituido por el reconstruido a partir del RF mensual.",
        "El precio de CRSP para la señal es aproximado: French no publica el mercado diario sin "
        "dividendos, así que E usa la rentabilidad total y F le resta el D/P mensual de Shiller (el del "
        "S&P, con un mes de retraso para no mirar el futuro).",
        "El 3x sobre CRSP es sintético en toda la muestra, con el diferencial calibrado con UPRO en "
        "2009–2026: usa información posterior a casi todo el periodo, como R12–R13.",
        "La fracción ½ de Kelly, el GJR-GARCH, la media de 200 sesiones y la banda salen de R10–R13, "
        "elegidos con la misma historia del S&P 500: aquí no se reoptimiza nada, pero tampoco son "
        "elecciones fuera de muestra. L* usa la muestra completa de cada ventana.",
        "La cota de los sábados no incluye financiación ni costes y usa la autocorrelación de CRSP, que "
        "exagera la de un índice de grandes valores.",
        "La regla R1 es conservadora (4 pares); la revisión manual de R2 añade 7 pares y hay más candidatos "
        "pequeños sin revisar. La sensibilidad I es a posteriori.",
        "«Yahoo corregido» en el análisis de la brecha con CRSP usa también los pares revisados a mano: es "
        "un diagnóstico a posteriori. El reparto de la brecha de ½σ² del 3x en errores, varianza semanal y "
        "autocorrelación es aproximado: usa la varianza log del índice por 4,5 en lugar de la del 3x y un "
        "orden fijo de los términos.",
        "Atribuir la autocorrelación de CRSP a la negociación no sincrónica es la lectura estándar, pero "
        "aquí no se contrasta directamente (haría falta un índice CRSP sólo de grandes valores); parte "
        "puede ser composición. La conclusión práctica no depende de ello: la ventaja del 3x con CRSP está "
        "en la varianza diaria, no en la media, y no se trasladaría a un fondo sobre el S&P 500.",
        "Durante el trabajo se corrigió la base a raíz de esta auditoría (cierres erróneos, sobrecoste del "
        "swap y 3x de 1960–2009; ver BITACORA.md). La versión «A» de este informe reproduce la base anterior "
        "a propósito, para medir el efecto de cada corrección.",
    ]))

    rep.h("Problemas del código base que detectó esta auditoría (ya corregidos)")
    rep.p("\n".join(f"- {x}" for x in [
        "**r12.make recibía real = synthetic_actual(...)** (en r12.load_all y, a través de él, en r13.load). "
        "synthetic_actual no es sólo el ETF real: incluye el sintético de R3 desde 1960 (UPRO), 1985 (TQQQ) "
        "o 1994 (SOXL). Por eso el 3x de 1960 a 2009 era el sintético de R3, financiado al T-bill y construido "
        "sobre ^GSPC más un dividendo fijo del 3,5% hasta 1988, no sobre la u con dividendos de Shiller ni a "
        "f: el 3x y el 1x de R12–R13 no eran coherentes entre sí en ese tramo. Ahora r12.load_all pasa sólo "
        f"el ETF real (pair(t).letf) y make() recalibra el diferencial sólo con su vida (UPRO: s = "
        f"{num(etf.loc[f'UPRO · {ETF_NEW}', 'diferencial_s_pp'], 3)}%); con ese cambio las cifras apalancadas "
        f"{fall} entre {num(dAB.min(), 2)} y {num(dAB.max(), 2)} pp/año (versión B frente a A).",
        "Cierres erróneos de ^GSPC en Yahoo (16-4-1935, 16-8-1935, 16-9-1936, 17-4-1961 y al menos otros "
        "siete antes de 1950): los cuatro de R1 se corrigen ahora en history.gspc_close (media geométrica de "
        "sus vecinos), que alimenta history.sp500_long hasta 1988 y la señal de precio de R13; los siete pares "
        "revisados a mano no se corrigen. common.long_underlying (^GSPC desde 1960, que usa el sintético de "
        "R3) sigue sin corregir; la base de R12–R13 ya no lo usa antes de 1988.",
        "history.sp500_long no tiene las sesiones de sábado anteriores a 1952: el lunes suma dos sesiones "
        "(efecto pequeño, sección 3), sin documentar en el módulo.",
        "La señal de la LRS original sobre la rentabilidad total de sp500_long miraba el futuro antes de "
        "1988: el D/P de Shiller del mes en curso usa la media de los cierres del mes. R13 (y aquí las "
        "versiones con Yahoo) calcula la señal con el D/P del mes anterior y, desde 1988, con ^SP500TR "
        "(r13.sp500_signal_returns); las rentabilidades no cambian.",
    ]))
    rep.save()
    return R


if __name__ == "__main__":
    run()
