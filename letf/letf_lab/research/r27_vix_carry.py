"""R27 — La prima y cómo cobrarla: el mecanismo de la volatilidad corta y H3 en los productos del VIX.

Pregunta. ¿De dónde sale lo que pierden los ETPs largos del VIX y lo que gana quien está al
otro lado, y cuál es la forma de cobrarlo: comprar un inverso (SVXY, SVIX, VYLD o un −0,5x/−1x
sintético) o vender en corto, con la misma exposición, el ETP largo que se deprecia (VXX, VIXY,
UVXY, UVIX)? Contrasta H3 del pre-registro (letf/preregistro/articulo4_volatilidad.md, commit
7cded63) para los productos del VIX; la parte de SQQQ frente a QQQ/TQQQ está en R29
(r29_short_inverse).

Mecanismo (descriptivo; no hay hipótesis pre-registradas sobre él):
- estructura de plazos (F1 > VIX, F2 > F1, VIX < VIX3M) y descomposición del rendimiento
  logarítmico del índice corto en cambio de nivel del futuro a 30 días y roll;
- prima del futuro frente al VIX: el futuro 21 sesiones antes de vencer frente a su
  liquidación final, y el futuro a 30 días frente al VIX de 21 sesiones después;
- prima de varianza: VIX de Cboe frente a la volatilidad realizada del S&P 500 en las 21
  sesiones siguientes;
- arrastre del reajuste diario: log(1 + L·x) = L·log(1 + x) − ½(L² − L)·x² + resto, para
  L = −1, −0,5, +1, +1,5 y +2 (sintéticos) y para los ETPs reales;
- colas: picos sobre umbral (GPD) y Hill en las subidas diarias del índice corto, con las
  dos convenciones de hora, frente a las pérdidas del SPY y de la LRS 3x (original).

H3 («Vender el ETF que se deprecia»), como está escrita: (a) a igual exposición, el corto de
UVXY (y de UVIX) supera a comprar SVXY (SVIX) antes de préstamo solo por el tamaño del corto y
la frecuencia de reajuste; un corto sin reajuste diario gana frente al diario si y solo si el
cociente de varianzas del periodo es menor que 1. (b) Con el préstamo mediano y sin remunerar
el producto del corto, Δg ≤ 0 frente al inverso comprado.

Concreción de H3 en esta línea (el pre-registro no fija el estadístico; se escribió al
programar esta línea, con una primera ejecución de prueba de los cortos reales a la vista, y
se declara como interpretación, no como desviación):
- (a-i) «supera antes de préstamo»: Δg del corto con préstamo 0 (producto a la letra − 0,5%)
  frente al inverso comprado, con p unilateral y Romano–Wolf en su familia.
- (a-ii) «solo por el tamaño y la frecuencia» (lectura común con R29, r29_short_inverse): el
  mismo corto y el mismo inverso sobre fondos ideales, los modelos de los ETPs (índice
  reconstruido y gasto del folleto; R26), dan la parte mecánica. «Tamaño» es lo que da un corto
  de peso s con fondos ideales: el gasto del inverso que no se compra, el gasto del largo que el
  corto cobra en proporción a s y el diferencial del producto del corto sobre s; «frecuencia»
  es el camino del reajuste no diario. La diferencia real − modelo es la fricción de los fondos
  (lo que cada ETP real rinde de más o de menos que su índice menos el gasto del folleto), que
  no es ni tamaño ni frecuencia. (a-ii) se sostiene si el IC90 de la fricción (bootstrap
  estacionario emparejado) contiene el 0.
- (a-iii) «si y solo si»: por periodo (semana o mes), el signo del P&L de camino exacto (corto
  que deriva menos corto reajustado cada día, sin costes) frente al de 1 − VR, con
  VR = (Σy)²/Σy² e y el exceso diario del fondo sobre la letra; y en el agregado de cada
  ventana. Aproximación de segundo orden: s(1 + s)/2 · (Σy² − (Σy)²) (se deriva abajo).
- (b): Δg del corto con préstamo del 5,2% (mediana de UVXY en IBKR) y el producto del corto
  sin remunerar. Se sostiene si Δg ≤ 0; se falsa si Δg > 0 con p de Romano–Wolf < 0,05; con
  Δg > 0 sin significación, «no concluyente». El veredicto se da en cada familia de productos
  reales y con todos sus cortos (no solo UVXY y UVIX), con y sin el 5–6-2-2018.

Protocolo (el del pre-registro y el laboratorio). Orden al cierre de t+1 de lo decidido al
cierre de t; 5 pb por unidad negociada; la caja cobra la letra (^IRX/252). Cortos con
r22_inverse.run_drift: préstamo de la rejilla 0 / 2 / 5,2 / 18% anual sobre el valor vivo del
corto; producto del corto a la letra − 0,5% (negativo con tipos cero: el corto paga por su
producto y «sin remunerar» es el caso favorable; que IBKR lo aplique así está sin verificar) o
sin remunerar; mantenimiento FINRA de 30% × |L| del valor del corto (tope 100%) y, si se
incumple al cierre, el corto se reduce a su objetivo ese mismo cierre; ruina si el capital
llega a cero (absorbente en la ventana). Reajuste diario, semanal (último día de la semana)
o mensual. Igual exposición: corto de peso s = E/L en un ETP de apalancamiento L para una
exposición −E al índice corto (E = 0,5 frente a SVXY desde el 28-2-2018, 1 frente a SVXY
antes y frente a SVIX; en VYLD, E = precio ponderado de los futuros/100, fijado en cada
reajuste). UVXY era 2x y SVXY −1x hasta el 27-2-2018: en la ventana 2011–2026 el corto
cambia de tamaño en ese cierre, decidido al cierre del 26-2-2018 con el cambio ya conocido (que
ProShares lo hubiera anunciado antes de ese cierre está sin verificar; el efecto de un día es
despreciable). Cada ventana se simula desde cero.

Familias (todas las variantes cuentan): cada familia es un inverso comprado y todos los cortos
{ETP largo} × {diario, semanal, mensual} × {0, 2, 5,2, 18%} × {letra − 0,5%, sin
remuneración} en una ventana común. Productos reales: −0,5x desde el 28-2-2018 (SVXY), −0,5x
y −1x desde el 31-3-2022 (SVXY, SVIX), −1x en 2011–2018 (SVXY, con y sin el 5–6-2-2018),
2011–2026 (SVXY, con y sin el 5–6-2-2018) y VYLD desde el 21-3-2025. Las familias desde el
31-3-2022 no llevan VXX: Barclays suspendió su emisión en marzo de 2022 y el cierre de Yahoo
empezó esas ventanas un 30% por encima de su modelo, una prima sobre el valor indicativo que un
corto cobra al desaparecer y que no es la prima de la volatilidad; con VXX se dan como
sensibilidad (R2v, R3v). Sintéticos en la ventana principal (20-12-2005 →
25-9-2026; antes y después del 28-2-2018): −0,5x y −1x con las dos convenciones de hora
(liquidación de CFE y 16:00), el −1x con y sin el 5–6-2-2018, y VYLD, frente a los cortos de
los sintéticos +1x, +1,5x y +2x. Inferencia: Δg con bootstrap estacionario emparejado (bloque
medio 63, 2.000 remuestreos), p unilateral (H0: Δg ≤ 0), Holm, Romano–Wolf y SPA de Hansen
en cada familia (también las sintéticas de antes y después de 2018); Hansen–Hodrick para los
cocientes de varianzas con ventanas solapadas.

Episodios (ventanas del informe exploratorio, fijadas antes de sus resultados): 2008 (29-8 →
20-11-2008), agosto de 2015 (18 → 25-8-2015), febrero de 2018 (31-1 → 9-2-2018), marzo de
2020 (21-2 → 18-3-2020) y agosto de 2024 (31-7 → 7-8-2024).

Tamaño y Kelly: crecimiento de una exposición constante k al −1x (k·(−1x) + (1 − k)·letra,
reajuste diario; por encima de 1, préstamo a la letra), L* aproximado (μ/σ²), L* exacto
(máximo del crecimiento realizado, IC90 por bootstrap de bloque 252) y L* con cola GPD
(cuerpo empírico y cola ajustada por encima del percentil 95 de las pérdidas). Colas, L* y
tamaños del −1x con y sin el 5–6-2-2018 y con las dos convenciones de hora.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import genpareto, norm

from letf_lab import data, kelly, metrics, stats
from letf_lab.data import vix
from letf_lab.report import FIG_W, Report, comma_fig, legend_band, num_es, pct_es
from letf_lab.research import r19_census as r19
from letf_lab.research import r22_inverse as r22
from letf_lab.research import r26_vix_data as r26

SEED = 27
B = 2000                   # remuestreos (Δg, IC, Romano–Wolf, SPA)
BLOCK = 63
B_KELLY = 1000             # remuestreos del IC de L*
BLOCK_KELLY = 252
B_EVT = 300                # remuestreos del IC de ξ
LAG = r22.LAG              # 2: señal al cierre de t, orden al cierre de t+1
COST_BPS = r22.COST_BPS    # 5 pb por unidad negociada
END = vix.END
FREQS = ("D", "W", "M")
FREQ_ES = {"D": "diario", "W": "semanal", "M": "mensual"}
BORROW = vix.BORROW_GRID   # 0 / 2 / 5,2 / 18%
B_MED = 0.052
HAIRCUT = vix.PROCEEDS_SPREAD
REB = {"letra − 0,5%": True, "sin remuneración": False}
REB_YES, REB_NO = list(REB)
LRS = "LRS 3x (original)"
CENSUS = r26.CENSUS

POST = vix.DESIGN_2018                                   # 28-2-2018
PRE_END = pd.Timestamp("2018-02-27")
REAL_START = pd.Timestamp("2011-10-04")                  # primer cierre de SVXY y UVXY
VS_START = pd.Timestamp("2022-03-30")                    # primer cierre de SVIX y UVIX
VYLD_START = pd.Timestamp("2025-03-21")                  # primer cierre de VYLD en Yahoo
PRINCIPAL = (vix.ST_START, END)
SYN_WINDOWS = {"principal": (vix.ST_START, END), "hasta 27-2-2018": (vix.ST_START, PRE_END),
               "desde 28-2-2018": (POST, END)}
EPISODES = {"2008": ("2008-08-29", "2008-11-20"), "ago-2015": ("2015-08-18", "2015-08-25"),
            "feb-2018": ("2018-01-31", "2018-02-09"), "mar-2020": ("2020-02-21", "2020-03-18"),
            "ago-2024": ("2024-07-31", "2024-08-07")}
EPISODE_ES = {"2008": "2008 (29-8 → 20-11)", "ago-2015": "ago. 2015 (18 → 25-8)",
              "feb-2018": "feb. 2018 (31-1 → 9-2)", "mar-2020": "mar. 2020 (21-2 → 18-3)",
              "ago-2024": "ago. 2024 (31-7 → 7-8)"}
K_MAX = 3.0                # tope de la búsqueda de L* (exposición al índice)
K_GRID = np.round(np.arange(0.0, 2.001, 0.02), 3)
Q_EVT = (0.90, 0.95, 0.975, 0.99)
CLIP = r22.CLIP            # −99,99%: pérdidas acotadas al tomar logaritmos

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA, YELLOW, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#9a9994"
CONV_ES = {"settle": "liquidación", "1600": "16:00", "1600_long": "16:00 (LONGVOL)"}

log_growth, dmy, pp, window = r26.log_growth, r26.dmy, r26.pp, r26.window


# ───────────────────────────── utilidades ─────────────────────────────

def maint(L: float) -> float:
    """Mantenimiento FINRA de un corto en un ETF apalancado: 30% × |L| de su valor, tope 100%."""
    return min(vix.FINRA_MAINT * abs(float(L)), 1.0)


def lx(L: float) -> str:
    return r22.lx(L)


def borrow_es(b: float) -> str:
    return f"préstamo {num_es(100 * b, 1).replace(',0', '')}%"


def ci_es(lo: float, hi: float, k: int = 1) -> str:
    """IC de una fracción anual en pp con coma decimal: «[−1,6; +0,5]»."""
    return f"[{pp(lo, k)}; {pp(hi, k)}]" if np.isfinite(lo) and np.isfinite(hi) else ""


def lg_series(r: pd.Series) -> pd.Series:
    return np.log1p(r.clip(lower=CLIP))


def ann_g(r: pd.Series) -> float:
    """Crecimiento logarítmico anual con la ruina absorbida (r22.absorb) y pérdidas acotadas."""
    return float(lg_series(r22.absorb(r.dropna())).mean() * 252)


def between(r, a, b):
    return r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]


def paired_boot(d: np.ndarray, rng, B_: int = B) -> tuple[float, float, float, float]:
    """Media anual de una serie diaria de diferencias, IC90 percentil y p bilateral (normal con
    el error del bootstrap estacionario emparejado, bloque BLOCK)."""
    d = np.asarray(d, float)[:, None]
    m = float(d.mean())
    draws = r19.boot_means(d, B_, BLOCK, rng)[:, 0]
    se = draws.std(ddof=1)
    lo, hi = np.percentile(draws, [5, 95])
    p = float(2 * norm.sf(abs(m) / se)) if se > 0 else np.nan
    return m * 252, lo * 252, hi * 252, p


_CACHE: dict = {}


def real_returns() -> pd.DataFrame:
    """ETPs de volatilidad y SPY (Yahoo, adj_close) recortados en vix.END."""
    if "R" not in _CACHE:
        _CACHE["R"] = vix.etp_returns(vix.VIX_ETPS + ("SPY",))
    return _CACHE["R"]


def lrs_series() -> pd.Series | None:
    """La LRS 3x (original) del artículo 2 (serie diaria del censo de R19), si está."""
    if "lrs" not in _CACHE:
        s = None
        if CENSUS.exists():
            head = pd.read_csv(CENSUS, nrows=0).columns
            if LRS in head:
                s = pd.read_csv(CENSUS, index_col=0, parse_dates=True, usecols=[head[0], LRS])[LRS]
                s = s[s.index <= END]
        _CACHE["lrs"] = s
    return _CACHE["lrs"]


# ───────────────────────────── 1. estructura de plazos y roll ─────────────────────────────

def term_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Contango por ventana y descomposición del rendimiento logarítmico de los índices en
    cambio de nivel del futuro de vencimiento constante y roll: log(1 + r_t) =
    Δlog(cm_t) + roll_t, con cm_t el precio ponderado con los pesos de t y r_t el rendimiento
    del índice con los pesos de t − 1 (roll_t = log(Σw_{t−1}P_t / Σw_t P_t))."""
    T = vix.term_structure()
    F = vix.futures_indices()
    dec = {}
    for k in ("st", "mt"):
        tot = np.log1p(F[f"{k}_ret"])
        lvl = np.log(F[f"{k}_cm"]).diff()
        dec[k] = pd.DataFrame({"total": tot, "nivel": lvl, "roll": tot - lvl})
    wins = {"principal (21-12-2005 →)": PRINCIPAL, "hasta 27-2-2018": (vix.ST_START, PRE_END),
            "desde 28-2-2018": (POST, END)}
    rows = []
    for w, (a, b) in wins.items():
        x = T[(T.index > a) & (T.index <= b)].dropna(subset=["f1", "f2", "vix", "st_cm"])
        v3 = x.dropna(subset=["vix3m"])
        row = {"ventana": w, "sesiones": len(x), "F1 > VIX": (x.f1 > x.vix).mean(),
               "F2 > F1": (x.f2 > x.f1).mean(), "CM30 > VIX": (x.st_cm > x.vix).mean(),
               "VIX < VIX3M": (v3.vix < v3.vix3m).mean(), "sesiones con VIX3M": len(v3),
               "F1/VIX − 1 (mediana)": (x.f1 / x.vix - 1).median(),
               "F2/F1 − 1 (media)": (x.f2 / x.f1 - 1).mean()}
        for k, lab in (("st", "corto"), ("mt", "medio")):
            a2 = max(pd.Timestamp(a), vix.MT_START) if k == "mt" else a
            d = dec[k][(dec[k].index > a2) & (dec[k].index <= b)].dropna()
            row[f"g índice {lab}"] = d.total.mean() * 252
            row[f"nivel {lab}"] = d.nivel.mean() * 252
            row[f"roll {lab}"] = d.roll.mean() * 252
        rows.append(row)
    st = dec["st"][(dec["st"].index > vix.ST_START)].dropna()
    return pd.DataFrame(rows).set_index("ventana"), st


# ───────────────────────────── 2. prima del futuro ─────────────────────────────

def futures_premium(h: int = 21) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(a) Por contrato: precio h sesiones de CFE antes del vencimiento frente a la liquidación
    final (el precio del día del vencimiento en los ficheros de CFE; que sea exactamente la
    liquidación especial de apertura está sin verificar). (b) Diario: futuro de 30 días (st_cm)
    menos el VIX de Cboe h sesiones después (t de Hansen–Hodrick con h − 1 retardos)."""
    p = vix.vx_panel()
    cal = pd.DatetimeIndex(sorted(p.date.unique()))
    vx = vix.cboe_index("VIX")
    rows = []
    for c, g in p.groupby("contract"):
        g = g.set_index("date").sort_index()
        e = g.expiry.iloc[0]
        if e > END or e not in g.index:
            continue
        pos = cal.searchsorted(e)
        if pos - h < 0:
            continue
        d0 = cal[pos - h]
        if d0 not in g.index or d0 <= vix.ST_START:
            continue
        f0, fT = float(g.px.loc[d0]), float(g.px.loc[e])
        rows.append({"contrato": c, "entrada": d0, "vencimiento": e, "F entrada": f0,
                     "VIX entrada": float(vx.get(d0, np.nan)), "liquidación": fT,
                     "prima (pts)": f0 - fT, "rend. largo": fT / f0 - 1})
    pc = pd.DataFrame(rows).set_index("contrato")
    T = vix.term_structure()
    cm, v = T.st_cm, T.vix
    fut_v = v.shift(-h)
    wins = {"principal": PRINCIPAL, "hasta 27-2-2018": (vix.ST_START, PRE_END), "desde 28-2-2018": (POST, END)}
    out = []
    for w, (a, b) in wins.items():
        x = pc[(pc.entrada > a) & (pc.entrada <= b)]
        fit = stats.ols_hac(x["prima (pts)"].reset_index(drop=True), pd.DataFrame(index=range(len(x))), 3)
        dd = pd.DataFrame({"base": cm - v, "ex": cm - fut_v})
        dd = dd[(dd.index > a) & (dd.index <= b)].dropna()
        fx = stats.ols_hac(dd.ex, pd.DataFrame(index=dd.index), **stats.hac_overlap(h))
        out.append({"ventana": w, "contratos": len(x), "prima media (pts)": x["prima (pts)"].mean(),
                    "prima mediana (pts)": x["prima (pts)"].median(),
                    "F > liquidación": (x["prima (pts)"] > 0).mean(), "t (NW 3)": float(fit.t[0]),
                    "rend. medio del largo": x["rend. largo"].mean(),
                    "peor rend. del corto": -x["rend. largo"].max(),
                    "sesiones": len(dd), "CM30 − VIX (pts)": dd.base.mean(), "CM30 > VIX": (dd.base > 0).mean(),
                    f"CM30 − VIX a {h} sesiones (pts)": dd.ex.mean(), f"CM30 > VIX a {h} sesiones": (dd.ex > 0).mean(),
                    "t (HH)": float(fx.t[0])})
    return pd.DataFrame(out).set_index("ventana"), pc


# ───────────────────────────── 3. prima de varianza ─────────────────────────────

def variance_premium(h: int = 21) -> pd.DataFrame:
    """VIX de Cboe al cierre de t frente a la volatilidad realizada anualizada del S&P 500
    (^GSPC, rendimientos logarítmicos) en las h sesiones siguientes."""
    g = data.panel(["^GSPC"], "close")["^GSPC"].dropna()
    g = g[g.index <= END]
    lr = np.log(g).diff()
    rv = np.sqrt((lr ** 2).rolling(h).sum().shift(-h) * 252 / h) * 100
    v = vix.cboe_index("VIX").reindex(g.index)
    df = pd.DataFrame({"VIX": v, "RV": rv}).dropna()
    wins = {"1990–2026": (df.index[0], END), "principal": PRINCIPAL, "hasta 27-2-2018": (vix.ST_START, PRE_END),
            "desde 28-2-2018": (POST, END)}
    rows = []
    for w, (a, b) in wins.items():
        x = df[(df.index >= a) & (df.index <= b)]
        dv = x.VIX - x.RV
        fit = stats.ols_hac(dv, pd.DataFrame(index=x.index), **stats.hac_overlap(h))
        dvar = (x.VIX ** 2 - x.RV ** 2) / 1e4
        fv = stats.ols_hac(dvar, pd.DataFrame(index=x.index), **stats.hac_overlap(h))
        rows.append({"ventana": w, "desde": x.index[0].date(), "sesiones": len(x), "VIX medio": x.VIX.mean(),
                     "RV media": x.RV.mean(), "VIX − RV (pts)": dv.mean(), "t (HH)": float(fit.t[0]),
                     "VIX > RV": (dv > 0).mean(), "VIX² − RV² (varianza)": dvar.mean(), "t varianza (HH)": float(fv.t[0]),
                     "p5 de VIX − RV": dv.quantile(0.05), "mínimo": dv.min(), "fecha del mínimo": dv.idxmin().date()})
    return pd.DataFrame(rows).set_index("ventana")


# ───────────────────────────── 4. arrastre del reajuste diario ─────────────────────────────

def drag_decomposition(r: pd.Series, x: pd.Series, L: float, c: pd.Series) -> dict:
    """log(1 + r) = L·log(1 + x) − ½(L² − L)·x² + c + resto, en medias anuales (×252).
    r: fondo; x: índice (ER) en el mismo calendario; c: letra − gasto del día."""
    j = pd.concat([r, x, c], axis=1, keys=["r", "x", "c"]).dropna()
    g = np.log1p(j.r.clip(lower=CLIP)).mean() * 252
    lgx = L * np.log1p(j.x.clip(lower=CLIP)).mean() * 252
    drag = -0.5 * (L * L - L) * (j.x ** 2).mean() * 252
    cc = j.c.mean() * 252
    return {"sesiones": len(j), "L": L, "½(L² − L)": 0.5 * (L * L - L), "g fondo": g, "L × g índice": lgx,
            "arrastre ½(L²−L)Σx²": drag, "letra − gasto": cc, "resto (orden ≥ 3 y seguimiento)": g - lgx - drag - cc}


def drag_table() -> pd.DataFrame:
    rows = []
    F = vix.futures_indices()
    e16 = vix.er_1600()
    for conv in ("settle", "1600"):
        x = F.st_ret if conv == "settle" else e16.st_ret_1600
        x = x[(x.index > vix.ST_START)].dropna()
        for key in ("-1x", "-0.5x", "+1x", "+1.5x", "+2x"):
            L = vix.SYNTHETIC[key][0]
            r = vix.synthetic(key, conv)
            c = vix.cash(r.index) - vix.SYNTHETIC[key][2] * vix._cal_days(r.index) / 365.0
            for tag, drop in (("", False), (" sin 5–6-2-2018", True)):
                if drop and conv == "1600" and key != "-1x":
                    continue
                rr, xx, ccc = (vix.drop_days(s) for s in (r, x, c)) if drop else (r, x, c)
                rows.append({"serie": f"{vix.LABELS[key]} sintético{tag}", "convención": CONV_ES[conv],
                             "ventana": "principal", **drag_decomposition(rr, xx, L, ccc)})
    R = real_returns()
    idx_ret = {"settle": F.st_ret, "1600": e16.st_ret_1600, "1600_long": e16.st_ret_1600_long}
    cases = [("SVXY", POST, END, None, "desde 28-2-2018"), ("UVXY", POST, END, None, "desde 28-2-2018"),
             ("VIXY", POST, END, None, "desde 28-2-2018"), ("SVXY", REAL_START, PRE_END, None, "2011–2018"),
             ("UVXY", REAL_START, PRE_END, None, "2011–2018"), ("SVXY", REAL_START, PRE_END, "sin", "2011–2018"),
             ("SVIX", VS_START, END, None, "desde 31-3-2022"), ("UVIX", VS_START, END, None, "desde 31-3-2022")]
    for t, a, b, tag, per in cases:
        spec = vix.ETP_SPECS[t]
        r = between(R[t], a, b).dropna()
        x = vix.to_calendar(idx_ret[spec.convention].dropna(), r.index).iloc[1:]
        r = r.iloc[1:]
        L = float(vix.etp_leverage(t, r.index).iloc[0])
        c = vix.cash(r.index) - spec.fee * vix._cal_days(r.index) / 365.0
        if tag:
            r, x, c = (vix.drop_days(s) for s in (r, x, c))
        w = f"{dmy(r.index[0])} → {dmy(r.index[-1])}"
        rows.append({"serie": f"{t} real, {per}{' sin 5–6-2-2018' if tag else ''}", "convención": CONV_ES[spec.convention],
                     "ventana": w, **drag_decomposition(r, x, L, c)})
    return pd.DataFrame(rows).set_index(["serie", "convención"])


# ───────────────────────────── 5. colas ─────────────────────────────

def gpd_fit(loss: np.ndarray, q: float) -> dict:
    return stats.gpd_tail(np.asarray(loss, float), q)


def gpd_rate(fit: dict, x: float) -> float:
    """Probabilidad anual (×252) de una pérdida diaria mayor que x según la cola GPD."""
    if x <= fit["u"]:
        return np.nan
    return 252 * stats.gpd_exceedance_prob(fit, x)


SIN_FEB = ", sin el 5–6-2-2018"
UP_SET = "Subida diaria del índice corto (liquidación)"
UP_16 = "Subida diaria del índice corto (16:00)"
INDEX_TAILS = (UP_SET, UP_16, UP_SET + SIN_FEB, UP_16 + SIN_FEB)


def tail_series() -> dict:
    """Series de colas en la ventana principal. Las subidas del índice corto (las que arruinan un
    −1x) van con las dos convenciones de hora y con y sin el 5–6-2-2018, como pide el
    pre-registro para todo resultado con exposición −1x."""
    F = vix.futures_indices()
    e16 = vix.er_1600()
    R = real_returns()
    up = between(F.st_ret, vix.ST_START + pd.Timedelta(days=1), END)
    up16 = between(e16.st_ret_1600, vix.ST_START + pd.Timedelta(days=1), END)
    out = {UP_SET: up, UP_16: up16, UP_SET + SIN_FEB: vix.drop_days(up), UP_16 + SIN_FEB: vix.drop_days(up16),
           "Pérdida diaria del −0,5x sintético (liquidación)": -vix.synthetic("-0.5x", "settle"),
           "Pérdida diaria del SPY": -between(R.SPY, vix.ST_START + pd.Timedelta(days=1), END)}
    lrs = lrs_series()
    if lrs is not None:
        out[f"Pérdida diaria de la {LRS}"] = -between(lrs, vix.ST_START + pd.Timedelta(days=1), END)
    return {k: v.dropna() for k, v in out.items()}


def tails(rng) -> dict:
    S = tail_series()
    rows = []
    fits = {}
    for k, s in S.items():
        fit = gpd_fit(s.to_numpy(), 0.95)
        fits[k] = fit
        xs = []
        for _ in range(B_EVT):
            ii = stats.stationary_indices(len(s), BLOCK, rng)
            try:
                xs.append(gpd_fit(s.to_numpy()[ii], 0.95)["xi"])
            except Exception:  # noqa: BLE001 — un remuestreo degenerado no invalida los demás
                continue
        lo, hi = np.percentile(xs, [5, 95])
        rows.append({"serie": k, "sesiones": len(s), "máximo": s.max(), "fecha del máximo": s.idxmax().date(),
                     "percentil 99": s.quantile(0.99), "umbral (p95)": fit["u"], "excesos": fit["n_exc"],
                     "ξ (GPD)": fit["xi"], "IC90 ξ": f"[{num_es(lo, 2)}; {num_es(hi, 2)}]", "β": fit["beta"],
                     "ξ Hill (k = 100)": stats.hill(s.to_numpy(), 100),
                     "P(> 20%) por año": gpd_rate(fit, 0.20), "P(> 50%) por año": gpd_rate(fit, 0.50),
                     "P(> 100%) por año": gpd_rate(fit, 1.00)})
    main = pd.DataFrame(rows).set_index("serie")
    sens = []
    for k in INDEX_TAILS:
        s = S[k].to_numpy()
        for q in Q_EVT:
            f = gpd_fit(s, q)
            p100 = gpd_rate(f, 1.0)
            sens.append({"serie": k.replace("Subida diaria del índice corto ", "índice corto "), "cuantil del umbral": q,
                         "umbral": f["u"], "excesos": f["n_exc"], "ξ": f["xi"], "β": f["beta"],
                         "P(> 50%) por año": gpd_rate(f, 0.5), "P(> 100%) por año": p100,
                         "años entre días > 100%": 1 / p100 if p100 and p100 > 0 else np.inf,
                         "P(> 200%) por año": gpd_rate(f, 2.0)})
    sens = pd.DataFrame(sens).set_index(["serie", "cuantil del umbral"])
    F = vix.futures_indices()
    e16 = vix.er_1600()
    vx = vix.cboe_index("VIX")
    s1 = vix.synthetic("-1x", "settle")
    top = S[UP_SET].nlargest(10)
    tt = pd.DataFrame({"índice corto (liquidación)": top,
                       "índice corto (16:00)": e16.st_ret_1600.reindex(top.index),
                       "−1x sintético (liquidación)": s1.reindex(top.index),
                       "VIX al cierre": vx.reindex(top.index),
                       "VIX: cambio del día": (vx / vx.shift(1) - 1).reindex(top.index)})
    tt.index = tt.index.date
    tt.index.name = "fecha"
    del F
    return {"main": main, "sens": sens, "top": tt, "fits": fits, "series": S}


# ───────────────────────────── H3: carteras ─────────────────────────────

def rule_name(veh: str, freq: str, b: float, reb: str) -> str:
    return f"corto {veh} {FREQ_ES[freq]} · {borrow_es(b)} · {reb}"


def run_bought(r: pd.Series, name: str) -> pd.Series:
    idx = r.index
    out, _ = r22.run_drift(pd.DataFrame({name: 1.0}, index=idx), r.to_frame(name), vix.cash(idx), freq="D",
                           cost_bps=COST_BPS, lag=LAG)
    return out


def legs(r: pd.Series, lev: pd.Series, expo: pd.Series, freq: str, name: str,
         dyn: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Columnas, pesos objetivo y mantenimiento del corto de igual exposición en un ETP largo.

    lev y expo: apalancamiento del ETP y exposición −E buscada en la sesión de cada rendimiento;
    el objetivo del cierre de t se ejecuta al cierre de t+1 y gana desde t+2, así que usa lev y
    expo de t+2 (un cambio de régimen conocido de antemano: el de 2018 se decide al cierre del
    26-2-2018; que ProShares lo hubiera anunciado antes está sin verificar). dyn (VYLD): multiplicador con
    información de t, fijado en la sesión anterior a cada reajuste (así la orden coincide con el
    reajuste programado). Si el apalancamiento cambia dentro de la ventana (UVXY el 28-2-2018),
    cada régimen es una columna con su mantenimiento."""
    idx = r.index
    fl = lev.shift(-LAG).ffill()
    fe = expo.shift(-LAG).ffill()
    base = fe / fl
    if dyn is not None:
        d = dyn.reindex(idx).ffill()
        pre = pd.Series(r22.rebalance_days(idx, freq), index=idx).shift(-1, fill_value=True)
        dd = d.where(pre)
        dd.iloc[0] = d.iloc[0]
        base = base * dd.ffill()
    Ls = sorted(pd.unique(lev.to_numpy()))
    if len(Ls) == 1:
        return r.to_frame(name), (-base).to_frame(name), {name: maint(Ls[0])}
    Rv, W, mt = {}, {}, {}
    for L in Ls:
        c = f"{name}·{lx(L)}"
        Rv[c] = r.where(lev == L, 0.0)
        W[c] = (-base).where(fl == L, 0.0)
        mt[c] = maint(L)
    return pd.DataFrame(Rv), pd.DataFrame(W), mt


def run_short(Rv: pd.DataFrame, W: pd.DataFrame, freq: str, borrow: float, rebate: bool,
              mt: dict) -> tuple[pd.Series, dict]:
    idx = Rv.index
    return r22.run_drift(W, Rv, vix.cash(idx), freq=freq, borrow=borrow, haircut=HAIRCUT, cost_bps=COST_BPS,
                         lag=LAG, maint=mt, margin_rule="restaurar", rebate=rebate)


@dataclass
class Vehicle:
    label: str                 # nombre en las reglas
    col: str                   # columna de rentabilidades
    lev: float | str           # apalancamiento o "etp" (vix.etp_leverage)
    fee: float


@dataclass
class Family:
    key: str
    title: str
    kind: str                  # "real" | "sintético"
    window: str
    bench_label: str
    bench: pd.Series
    rules: dict = field(default_factory=dict)
    meta: pd.DataFrame | None = None
    R: pd.DataFrame | None = None
    expo: pd.Series | None = None
    dyn: pd.Series | None = None
    vehicles: list = field(default_factory=list)


def build_family(key: str, title: str, kind: str, wlabel: str, R: pd.DataFrame, bench_col: str, bench_label: str,
                 vehicles: list, expo: pd.Series, dyn: pd.Series | None = None) -> Family:
    idx = R.index
    fam = Family(key, title, kind, wlabel, bench_label, run_bought(R[bench_col], bench_col), R=R, expo=expo, dyn=dyn,
                 vehicles=vehicles)
    meta = []
    for v in vehicles:
        lev = vix.etp_leverage(v.col, idx) if v.lev == "etp" else pd.Series(float(v.lev), index=idx)
        for fq in FREQS:
            Rv, W, mt = legs(R[v.col], lev, expo, fq, v.label, dyn)
            for b in BORROW:
                for rb_lab, rb in REB.items():
                    out, info = run_short(Rv, W, fq, b, rb, mt)
                    name = rule_name(v.label, fq, b, rb_lab)
                    fam.rules[name] = out
                    w0 = W.iloc[LAG - 1:].abs().sum(axis=1)
                    meta.append({"regla": name, "vehículo": v.label, "reajuste": FREQ_ES[fq], "préstamo": b,
                                 "producto del corto": rb_lab, "peso inicial del corto": float(w0.iloc[0]),
                                 "incumplimientos/año": info["incumplimientos_año"],
                                 "incumplimientos": info["incumplimientos"], "ruinas": info["ruinas"]})
    fam.meta = pd.DataFrame(meta).set_index("regla")
    return fam


def expo_regime(idx: pd.DatetimeIndex) -> pd.Series:
    """Exposición de SVXY: −1 hasta el 27-2-2018 y −0,5 desde el 28-2-2018."""
    return pd.Series(np.where(idx < POST, 1.0, 0.5), index=idx)


def real_frame(cols, a, b, drop: bool = False) -> pd.DataFrame:
    R = real_returns()[list(cols)]
    R = R[(R.index >= pd.Timestamp(a)) & (R.index <= pd.Timestamp(b))].dropna()
    return vix.drop_days(R) if drop else R


def vyld_level(index: pd.DatetimeIndex) -> pd.Series:
    """Multiplicador de VYLD: precio ponderado de los futuros (índice corto, pesos de t)/100,
    llevado al calendario pedido con el último valor conocido (es un nivel, no un rendimiento)."""
    cm = vix.futures_indices().st_cm.dropna()
    return cm.reindex(cm.index.union(index)).ffill().reindex(index) / 100.0


def real_families() -> list[Family]:
    V = {"VXX": Vehicle("VXX", "VXX", "etp", vix.ETP_SPECS["VXX"].fee),
         "VIXY": Vehicle("VIXY", "VIXY", "etp", vix.ETP_SPECS["VIXY"].fee),
         "UVXY": Vehicle("UVXY", "UVXY", "etp", vix.ETP_SPECS["UVXY"].fee),
         "UVIX": Vehicle("UVIX", "UVIX", "etp", vix.ETP_SPECS["UVIX"].fee)}
    fams = []
    specs = [
        ("R1", "−0,5x frente a SVXY, diseño actual", f"{dmy(POST)} → {dmy(END)}", "SVXY", ("VXX", "VIXY", "UVXY"),
         POST, END, False, lambda i: pd.Series(0.5, index=i), None),
        ("R2v", "−0,5x frente a SVXY, con UVIX y con VXX (sensibilidad)", f"{dmy(VS_START)} → {dmy(END)}", "SVXY",
         ("VXX", "VIXY", "UVXY", "UVIX"), VS_START, END, False, lambda i: pd.Series(0.5, index=i), None),
        ("R3v", "−1x frente a SVIX, con VXX (sensibilidad)", f"{dmy(VS_START)} → {dmy(END)}", "SVIX",
         ("VXX", "VIXY", "UVXY", "UVIX"), VS_START, END, False, lambda i: pd.Series(1.0, index=i), None),
        ("R4", "−1x frente a SVXY (entonces −1x)", f"{dmy(REAL_START)} → {dmy(PRE_END)}", "SVXY", ("VIXY", "UVXY"),
         REAL_START, PRE_END, False, lambda i: pd.Series(1.0, index=i), None),
        ("R4s", "−1x frente a SVXY (entonces −1x), sin el 5–6-2-2018", f"{dmy(REAL_START)} → {dmy(PRE_END)}", "SVXY",
         ("VIXY", "UVXY"), REAL_START, PRE_END, True, lambda i: pd.Series(1.0, index=i), None),
        ("R5", "SVXY 2011–2026 (−1x y, desde el 28-2-2018, −0,5x)", f"{dmy(REAL_START)} → {dmy(END)}", "SVXY",
         ("VIXY", "UVXY"), REAL_START, END, False, expo_regime, None),
        ("R5s", "SVXY 2011–2026, sin el 5–6-2-2018", f"{dmy(REAL_START)} → {dmy(END)}", "SVXY", ("VIXY", "UVXY"),
         REAL_START, END, True, expo_regime, None),
        ("R6", "VYLD (exposición −F/100)", f"{dmy(VYLD_START)} → {dmy(END)}", "VYLD", ("VXX", "VIXY", "UVXY", "UVIX"),
         VYLD_START, END, False, lambda i: pd.Series(1.0, index=i), "vyld"),
    ]
    for key, title, wl, bench, longs, a, b, drop, ex, dyn in specs:
        R = real_frame((bench,) + longs, a, b, drop)
        d = vyld_level(R.index) if dyn == "vyld" else None
        kind = SENS if key.endswith("v") else "real"
        fams.append(build_family(key, title, kind, wl, R, bench, f"{bench} comprado", [V[t] for t in longs],
                                 ex(R.index), d))
    # R2 y R3 (las familias del contraste): las mismas simulaciones sin los cortos de VXX
    out = []
    for f in fams:
        if f.key in ("R2v", "R3v"):
            out.append(without_vehicle(f, f.key[:-1], f.title.replace(" y con VXX (sensibilidad)", "")
                                       .replace(", con VXX (sensibilidad)", ""), "VXX"))
        out.append(f)
    return sorted(out, key=lambda f: REAL_ORDER.index(f.key))


SENS = "real, con VXX (sensibilidad)"
REAL_ORDER = ("R1", "R2", "R3", "R4", "R4s", "R5", "R5s", "R6", "R2v", "R3v")


def without_vehicle(fam: Family, key: str, title: str, veh: str) -> Family:
    """La familia sin los cortos de un vehículo: las mismas reglas simuladas, sin recalcular nada.
    La ventana común no cambia (VXX cotiza todas las sesiones de R2v y R3v)."""
    rules = {k: r for k, r in fam.rules.items() if fam.meta.loc[k, "vehículo"] != veh}
    return Family(key, title, "real", fam.window, fam.bench_label, fam.bench, rules=rules,
                  meta=fam.meta.loc[list(rules)], R=fam.R.drop(columns=[veh]), expo=fam.expo, dyn=fam.dyn,
                  vehicles=[v for v in fam.vehicles if v.label != veh])


SYN_LONGS = {"+1x": ("+1x (tipo VIXY)", 1.0), "+1.5x": ("+1,5x (tipo UVXY actual)", 1.5),
             "+2x": ("+2x (tipo UVXY hasta 2018 y UVIX)", 2.0)}


def synthetic_frame(keys, conv: str, a, b, drop: bool = False) -> pd.DataFrame:
    S = pd.concat({k: vix.synthetic(k, conv) for k in keys}, axis=1)
    S = S[(S.index >= pd.Timestamp(a)) & (S.index <= pd.Timestamp(b))].dropna()
    return vix.drop_days(S) if drop else S


def synthetic_families() -> list[Family]:
    fams = []
    V = [Vehicle(lab, k, L, vix.SYNTHETIC[k][2]) for k, (lab, L) in SYN_LONGS.items()]
    n = 0
    for conv in ("settle", "1600"):
        for inv, E in (("-0.5x", 0.5), ("-1x", 1.0)):
            for drop in (False, True):
                if drop and inv != "-1x":
                    continue
                for wl, (a, b) in SYN_WINDOWS.items():
                    if drop and wl == "desde 28-2-2018":
                        continue
                    n += 1
                    R = synthetic_frame([inv] + list(SYN_LONGS), conv, a, b, drop)
                    title = (f"{vix.LABELS[inv]} sintético ({CONV_ES[conv]})"
                             + (", sin el 5–6-2-2018" if drop else ""))
                    fams.append(build_family(f"S{n}", title, "sintético", wl, R, inv, f"{vix.LABELS[inv]} sintético comprado",
                                             V, pd.Series(E, index=R.index)))
    for wl, (a, b) in SYN_WINDOWS.items():
        n += 1
        R = synthetic_frame(["vyld"] + list(SYN_LONGS), "settle", a, b)
        fams.append(build_family(f"S{n}", "VYLD sintético (liquidación)", "sintético", wl, R, "vyld",
                                 "VYLD sintético comprado", V, pd.Series(1.0, index=R.index), vyld_level(R.index)))
    return fams


# ───────────────────────────── VXX en 2022 ─────────────────────────────

VXX_DATES = {"11-3-2022 (última sesión antes de la suspensión)": "2022-03-11",
             "30-3-2022 (inicio de R2 y R3)": "2022-03-30", "30-9-2022": "2022-09-30", "25-9-2026": "2026-09-25"}


def vxx_premium() -> tuple[pd.DataFrame, dict]:
    """Cociente acumulado entre el VXX de Yahoo y su modelo (TR corto − gasto, liquidación de CFE;
    el mismo de R26, normalizado en el primer cierre común) en las fechas de VXX_DATES y su
    máximo de 2022, y lo que ese cociente regala a un corto de VXX de peso 1 desde el inicio de
    R2 y R3: −(log cociente final − log cociente inicial) / años."""
    real = real_returns()["VXX"].dropna()
    m = vix.to_calendar(vix.etp_model("VXX"), real.index)
    j = pd.concat([real, m], axis=1, keys=["p", "m"]).dropna().iloc[1:]
    ratio = np.exp((np.log1p(j.p) - np.log1p(j.m)).cumsum())
    rows = {k: {"fecha": dmy(pd.Timestamp(d)), "cociente VXX real / modelo": float(ratio.asof(pd.Timestamp(d)))}
            for k, d in VXX_DATES.items()}
    y22 = ratio.loc["2022"]
    rows["máximo de 2022"] = {"fecha": dmy(y22.idxmax()), "cociente VXX real / modelo": float(y22.max())}
    t = pd.DataFrame(rows).T
    r0, r1 = float(ratio.asof(VS_START)), float(ratio.asof(END))
    yrs = (END - VS_START).days / 365.25
    return t, {"r0": r0, "r1": r1, "gift": -np.log(r1 / r0) / yrs, "peak": float(y22.max()),
               "peak_date": y22.idxmax(), "sep": float(ratio.asof(pd.Timestamp("2022-09-30")))}


# ───────────────────────────── H3: inferencia por familia ─────────────────────────────

def evaluate(fam: Family, rng, with_spa: bool = True) -> tuple[pd.DataFrame, dict]:
    bt = r22.boot_family(fam.rules, fam.bench, rng=rng)
    rf = vix.cash(fam.bench.index)
    rows = {}
    for k, r in fam.rules.items():
        s = r.dropna()
        pf = r22.perf(s, rf)
        rd = r22.ruin_date(r22.absorb(s))
        rows[k] = {"CAGR": pf["CAGR"], "maxDD": pf["maxDD"], "g": ann_g(s),
                   "ruina": str(rd.date()) if rd is not None else ""}
    t = fam.meta.join(pd.DataFrame.from_dict(rows, orient="index")).join(bt)
    t["IC90"] = [ci_es(lo, hi) for lo, hi in zip(t.IC90_bajo, t.IC90_alto)]
    bb = fam.bench.dropna()
    summ = {"familia": fam.key, "título": fam.title, "tipo": fam.kind, "ventana": fam.window,
            "desde": bt.attrs.get("desde"), "hasta": bt.attrs.get("hasta"), "sesiones": bt.attrs.get("sesiones"),
            "inverso comprado": fam.bench_label, "CAGR del inverso": metrics.cagr(r22.absorb(bb)),
            "maxDD del inverso": metrics.max_dd(r22.absorb(bb)), "g del inverso": ann_g(bb),
            "reglas": len(fam.rules), "Δg > 0": int((t["Δg"] > 0).sum()),
            "p_RW < 0,05 (a favor del corto)": int((t.p_RW < 0.05).sum()),
            "con ruina": int((t.ruina != "").sum())}
    if with_spa:
        sp = r22.spa(fam.rules, fam.bench, rng=rng)
        summ.update({"mejor regla": sp["mejor"], "Δg de la mejor": sp["Δg_mejor"], "p_SPA": sp["p_SPA_c"]})
    return t, summ


# ───────────────────────────── H3a-ii: tamaño, frecuencia y fricción ─────────────────────────────

def model_frame(fam: Family, drop: bool = False) -> pd.DataFrame:
    """Modelos de los ETPs de la familia (índice reconstruido en la convención de cada producto,
    vix.etp_model) en el calendario de NYSE de la familia. Con drop, el 5–6-2-2018 se quita del
    modelo ANTES de llevarlo al calendario (si no, to_calendar los compondría en el día 7). La
    primera fila compone toda la historia anterior: run_drift no la usa (orden al cierre siguiente)."""
    out = {}
    for c in fam.R.columns:
        m = vix.etp_model(c)
        out[c] = vix.to_calendar(vix.drop_days(m) if drop else m, fam.R.index)
    return pd.DataFrame(out)


def decomposition(fams: dict, rng) -> pd.DataFrame:
    """Por par (corto de un ETP largo frente al inverso comprado), con préstamo 0 y producto a la
    letra − 0,5%: Δg real, Δg de los modelos (mecánica: tamaño y frecuencia), la parte analítica
    del tamaño (gastos y producto del corto) y la fricción real − modelo con su IC90."""
    rows = []
    for key in ("R1", "R3", "R4", "R4s"):
        fam = fams[key]
        M = model_frame(fam, drop=key.endswith("s"))
        inv_col = [c for c in fam.R.columns if c not in [v.col for v in fam.vehicles]][0]
        inv_real = fam.bench
        inv_model = run_bought(M[inv_col], inv_col)
        idx = fam.R.index
        rbar = vix.cash(idx).mean() * 252
        fee_inv = vix.ETP_SPECS[inv_col].fee
        for v in fam.vehicles:
            lev = vix.etp_leverage(v.col, idx)
            gd = {}
            for fq in FREQS:
                name = rule_name(v.label, fq, 0.0, REB_YES)
                s_real = fam.rules[name]
                Rv, W, mt = legs(M[v.col], lev, fam.expo, fq, v.label)
                s_model, _ = run_short(Rv, W, fq, 0.0, True, mt)
                j = pd.concat([s_real, inv_real, s_model, inv_model], axis=1, keys=["sr", "ir", "sm", "im"]).dropna()
                lj = j.apply(lambda c: lg_series(r22.absorb(c)))
                dg_real = (lj.sr - lj.ir).mean() * 252
                dg_model = (lj.sm - lj.im).mean() * 252
                dd = (lj.sr - lj.ir) - (lj.sm - lj.im)
                fr, lo, hi, p = paired_boot(dd.to_numpy(), rng)
                late = dd[dd.index >= vix.SETTLE_CHANGE]
                if key == "R1" and len(late) > 250:
                    fr2, lo2, hi2, _ = paired_boot(late.to_numpy(), rng)
                    late_txt = f"{pp(fr2)} {ci_es(lo2, hi2)}"
                else:
                    late_txt = ""
                s = float(fam.expo.iloc[-1] / lev.iloc[-1])
                carry = fee_inv + s * v.fee - s * HAIRCUT
                gd[fq] = dg_model
                rows.append({"familia": key, "par": f"corto {v.label} frente a {inv_col}", "reajuste": FREQ_ES[fq],
                             "_lo": lo, "_hi": hi,
                             "peso del corto (final)": s, "Δg real": dg_real, "Δg modelos": dg_model,
                             "tamaño: gastos y producto (analítico)": carry,
                             "frecuencia (modelos, frente al diario)": dg_model - gd["D"],
                             "fricción (real − modelos)": fr, "IC90 fricción": ci_es(lo, hi), "p fricción (bilateral)": p,
                             "fricción desde 7-10-2020 (pp, IC90)": late_txt,
                             "del largo: g corto real − modelo": (lj.sr - lj.sm).mean() * 252,
                             "del inverso: −(g real − modelo)": -(lj.ir - lj.im).mean() * 252,
                             "letra media": rbar})
    return pd.DataFrame(rows).set_index(["familia", "par", "reajuste"])


# ───────────────────────────── H3a-iii: frecuencia y cociente de varianzas ─────────────────────────────

def path_cases(fams: dict) -> list[tuple]:
    """(etiqueta, exceso diario del fondo sobre la letra, peso del corto s, familia, vehículo)."""
    R = real_returns()
    out = []
    for key, t in (("R1", "UVXY"), ("R1", "VIXY"), ("R1", "VXX"), ("R4", "UVXY"), ("R4", "VIXY"),
                   ("R3", "UVIX"), ("R3", "UVXY")):
        fam = fams[key]
        idx = fam.R.index
        y = (fam.R[t] - vix.cash(idx)).iloc[1:]
        s = float(fam.expo.iloc[0] / vix.etp_leverage(t, idx).iloc[0])
        out.append((f"{t} ({fam.window})", y, s, key, t))
    for key, conv, lab in (("+1.5x", "settle", "liquidación"), ("+1.5x", "1600", "16:00"),
                           ("+1x", "settle", "liquidación"), ("+2x", "settle", "liquidación")):
        r = vix.synthetic(key, conv)
        r = r[r.index > vix.ST_START]
        y = r - vix.cash(r.index)
        L = SYN_LONGS[key][1]
        E = 0.5 if key in ("+1x", "+1.5x") else 1.0
        fk = next(f.key for f in fams.values() if f.kind == "sintético" and f.window == "principal"
                  and f.title == f"{vix.LABELS['-0.5x' if E == 0.5 else '-1x']} sintético ({CONV_ES[conv]})")
        out.append((f"{SYN_LONGS[key][0]} sintético, {lab} (principal)", y, E / L, fk, SYN_LONGS[key][0]))
    return out


def path_test(fams: dict, res: dict) -> tuple[pd.DataFrame, dict]:
    """Camino exacto por periodo del corto que deriva frente al reajustado cada día (sin costes
    ni intereses; r22.period_paths con w = −s) y su aproximación s(1 + s)/2·(Σy² − (Σy)²).

    Derivación: con Y = Σy y Q = Σy², el corto sin reajuste acaba con 1 − s(Π(1 + y) − 1) ≈
    1 − sY − s(Y² − Q)/2 y el diario con Π(1 − sy) ≈ 1 − sY + s²(Y² − Q)/2; la diferencia es
    s(1 + s)/2·(Q − Y²), positiva si y solo si VR = Y²/Q < 1 (a segundo orden)."""
    rows, periods = [], {}
    for lab, y, s, fk, veh in path_cases(fams):
        ex = y.to_frame("f")
        coef = s * (1 + s) / 2
        t = res[fk]
        for fq in ("W", "M"):
            per = r22.period_paths(ex, {"f": -s}, y, fq, coef)
            per["VR"] = per.x ** 2 / per.x2
            # en logaritmos (crecimiento): log(riqueza sin reajuste) − log(riqueza con reajuste diario)
            daily = per.deriva - per.camino
            per["camino_log"] = np.log1p(per.deriva.clip(lower=CLIP)) - np.log1p(daily.clip(lower=CLIP))
            periods[(lab, fq)] = per
            agree = np.sign(per.camino) == np.sign(1 - per.VR)
            agree_log = np.sign(per.camino_log) == np.sign(1 - per.VR)
            fit = stats.ols_hac(per.camino, per.aprox.to_frame("aprox"), stats.NONOV_LAGS)
            vr_agg = float((per.x ** 2).mean() / per.x2.mean())
            k = 52 if fq == "W" else 12
            dg = (t.loc[rule_name(veh, fq, 0.0, REB_YES), "g"] - t.loc[rule_name(veh, "D", 0.0, REB_YES), "g"])
            rows.append({"caso": lab, "peso del corto s": s, "reajuste": FREQ_ES[fq], "periodos": len(per),
                         "camino medio anual": per.camino.mean() * k, "aproximación anual": per.aprox.mean() * k,
                         "VR agregado": vr_agg, "periodos con VR < 1": (per.VR < 1).mean(),
                         "signo de acuerdo (camino, 1 − VR)": agree.mean(),
                         "acuerdo en el agregado": "sí" if np.sign(per.camino.mean()) == np.sign(1 - vr_agg) else "no",
                         "camino log medio anual": per.camino_log.mean() * k,
                         "signo de acuerdo en log": agree_log.mean(),
                         "acuerdo en el agregado (log)": "sí" if np.sign(per.camino_log.mean()) == np.sign(1 - vr_agg) else "no",
                         "correlación": float(np.corrcoef(per.camino, per.aprox)[0, 1]),
                         "pendiente": float(fit.beta[1]), "t pendiente (NW)": float(fit.t[1]),
                         "Δg simulado (frente al diario, sin préstamo)": dg,
                         "peor periodo sin reajuste": float(per.deriva.min())})
    return pd.DataFrame(rows).set_index(["caso", "reajuste"]), periods


def vr_test() -> pd.DataFrame:
    """VR(h) del índice corto (ER) con ventanas solapadas y contraste de z = Σx² − (Σx)² frente a
    0 con Hansen–Hodrick (h − 1 retardos). z > 0 en media: el corto sin reajuste gana al diario."""
    F = vix.futures_indices()
    e16 = vix.er_1600()
    rows = []
    for conv, x in (("liquidación", F.st_ret), ("16:00", e16.st_ret_1600)):
        for w, (a, b) in SYN_WINDOWS.items():
            xe = x[(x.index > a) & (x.index <= b)].dropna()
            for h in (5, 21):
                s1 = xe.rolling(h).sum().shift(-(h - 1))
                s2 = (xe ** 2).rolling(h).sum().shift(-(h - 1))
                z = (s2 - s1 ** 2).dropna()
                fit = stats.ols_hac(z, pd.DataFrame(index=z.index), **stats.hac_overlap(h))
                rows.append({"convención": conv, "ventana": w, "h (sesiones)": h, "ventanas": len(z),
                             "VR(h)": float((s1 ** 2).mean() / s2.mean()), "z medio anual": float(z.mean() * 252 / h),
                             "t (HH)": float(fit.t[0])})
                if conv == "liquidación" and w == "principal":
                    xd = vix.drop_days(xe)
                    s1d = xd.rolling(h).sum().shift(-(h - 1))
                    s2d = (xd ** 2).rolling(h).sum().shift(-(h - 1))
                    zz = (s2d - s1d ** 2).dropna()
                    fit = stats.ols_hac(zz, pd.DataFrame(index=zz.index), **stats.hac_overlap(h))
                    rows.append({"convención": conv, "ventana": "principal, sin el 5–6-2-2018", "h (sesiones)": h,
                                 "ventanas": len(zz), "VR(h)": float((s1d ** 2).mean() / s2d.mean()),
                                 "z medio anual": float(zz.mean() * 252 / h), "t (HH)": float(fit.t[0])})
    return pd.DataFrame(rows).set_index(["convención", "ventana", "h (sesiones)"])


# ───────────────────────────── H3: tablas de lectura ─────────────────────────────

def key_table(t: pd.DataFrame) -> pd.DataFrame:
    """Por vehículo y reajuste: Δg antes de préstamo (letra − 0,5%) y con el 5,2% sin remunerar."""
    rows = []
    for veh in t["vehículo"].unique():
        for fq in FREQS:
            a = t.loc[rule_name(veh, fq, 0.0, REB_YES)]
            b = t.loc[rule_name(veh, fq, B_MED, REB_NO)]
            rows.append({"vehículo": veh, "reajuste": FREQ_ES[fq], "peso": a["peso inicial del corto"],
                         "Δg sin préstamo": a["Δg"], "IC90 (a)": a["IC90"], "p_RW (a)": a["p_RW"],
                         "Δg 5,2% sin remunerar": b["Δg"], "IC90 (b)": b["IC90"], "p_RW (b)": b["p_RW"],
                         "CAGR (b)": b["CAGR"], "maxDD (b)": b["maxDD"],
                         "incumpl./año": b["incumplimientos/año"], "ruina (b)": b["ruina"]})
    return pd.DataFrame(rows).set_index(["vehículo", "reajuste"])


def breakeven(t: pd.DataFrame) -> pd.DataFrame:
    """Préstamo anual que iguala el corto al inverso comprado (interpolación lineal de Δg en la
    rejilla; «> 18%» si aún gana con el 18%; «< 0» si pierde sin préstamo)."""
    rows = []
    for veh in t["vehículo"].unique():
        for fq in FREQS:
            row = {"vehículo": veh, "reajuste": FREQ_ES[fq]}
            for rb in REB:
                y = np.array([t.loc[rule_name(veh, fq, b, rb), "Δg"] for b in BORROW])
                x = np.array(BORROW)
                if y[0] <= 0:
                    v = "< 0"
                elif y[-1] > 0:
                    v = "> 18%"
                else:
                    i = int(np.argmax(y <= 0))
                    b0 = x[i - 1] + (x[i] - x[i - 1]) * y[i - 1] / (y[i - 1] - y[i])
                    v = f"{num_es(100 * b0, 1)}%"
                row[f"préstamo que lo iguala ({rb})"] = v
            rows.append(row)
    return pd.DataFrame(rows).set_index(["vehículo", "reajuste"])


def synth_summary(fams: list[Family], res: dict) -> pd.DataFrame:
    """Sintéticos: el corto del +1,5x (−0,5x) o del +2x (−1x) o del +1x (VYLD) por ventana."""
    rows = []
    for f in fams:
        if f.kind != "sintético":
            continue
        t = res[f.key]
        veh = SYN_LONGS["+1x"][0] if "VYLD" in f.title else SYN_LONGS["+1.5x" if "0,5x" in f.title else "+2x"][0]
        row = {"familia": f.title, "ventana": f.window, "corto de": veh,
               "CAGR del inverso": metrics.cagr(r22.absorb(f.bench.dropna()))}
        for fq in FREQS:
            row[f"Δg {FREQ_ES[fq]} sin préstamo"] = t.loc[rule_name(veh, fq, 0.0, REB_YES), "Δg"]
        for fq in FREQS:
            row[f"Δg {FREQ_ES[fq]} 5,2% sin rem."] = t.loc[rule_name(veh, fq, B_MED, REB_NO), "Δg"]
        row["ruinas (de 72)"] = int((t.ruina != "").sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index(["familia", "ventana"])


def y_list(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


def fmt_years(v: float) -> str:
    """Años entre sucesos: «más de 100.000» si la cola GPD casi no llega (o no llega: ξ < 0)."""
    return "más de 100.000" if not np.isfinite(v) or v > 1e5 else num_es(v, 0)


def fmt_p(p: float) -> str:
    return "< 0,001" if p < 0.001 else num_es(p, 3)


def _cells(t: pd.DataFrame, veh: str, b: float, rb: str) -> pd.DataFrame:
    return t.loc[[rule_name(veh, fq, b, rb) for fq in FREQS]].assign(fq=list(FREQS))


def _parts(sub: pd.DataFrame) -> str:
    out = []
    for _, r in sub.iterrows():
        if r.ruina:
            out.append(f"{FREQ_ES[r.fq]}, ruina el {dmy(r.ruina)}")
        else:
            out.append(f"{FREQ_ES[r.fq]} {pp(r['Δg'])} pp, p_RW {fmt_p(r.p_RW)}")
    return "; ".join(out)


def _others(sig: list) -> str:
    rest = [FREQ_ES[f] for f in FREQS if f not in sig]
    return ("no con el " + " ni con el ".join(rest)) if rest else ""


def verdict_a1(t: pd.DataFrame, veh: str) -> tuple[str, str]:
    """H3 (a-i): el corto sin préstamo (producto a la letra − 0,5%) frente al inverso comprado."""
    sub = _cells(t, veh, 0.0, REB_YES)
    sig = [r.fq for _, r in sub.iterrows() if r["Δg"] > 0 and r.p_RW < 0.05]
    if len(sig) == len(FREQS):
        v = "se cumple con los tres reajustes"
    elif sig:
        v = "se cumple con reajuste " + " y ".join(FREQ_ES[f] for f in sig) + "; " + _others(sig)
    elif (sub["Δg"] > 0).all():
        v = "Δg > 0 en los tres reajustes, sin significación tras la búsqueda"
    else:
        v = "no se cumple"
    return v, _parts(sub)


def verdict_b(t: pd.DataFrame, veh: str) -> tuple[str, str]:
    """H3 (b) para un vehículo en su familia: préstamo 5,2% y producto sin remunerar."""
    sub = _cells(t, veh, B_MED, REB_NO)
    sig = [r.fq for _, r in sub.iterrows() if r["Δg"] > 0 and r.p_RW < 0.05]
    if (sub["Δg"] <= 0).all():
        v = "se sostiene (Δg ≤ 0 con los tres reajustes)"
    elif sig:
        edge = all(sub.set_index("fq").loc[f, "p_RW"] >= 0.01 for f in sig)
        v = ("se falsa con reajuste " + " y ".join(FREQ_ES[f] for f in sig) + " (Δg > 0 con p_RW < 0,05"
             + (", al límite)" if edge else ")"))
    else:
        v = "no concluyente (Δg > 0 sin significación tras la búsqueda)"
    return v, _parts(sub)


# familias de productos reales del contraste y vehículos de H3 en cada una (el pre-registro nombra
# UVXY y UVIX; H3 (b) se juzga con todos los cortos de la familia)
H3_REAL = {"R1": ("UVXY",), "R2": ("UVXY", "UVIX"), "R3": ("UVIX", "UVXY"), "R4": ("UVXY",), "R4s": ("UVXY",),
           "R5": ("UVXY",), "R5s": ("UVXY",), "R6": ("UVXY", "UVIX")}


def verdict_b_family(t: pd.DataFrame) -> tuple[str, str, list]:
    """H3 (b) en una familia con TODOS sus cortos (préstamo 5,2%, producto sin remunerar):
    (estado, texto, celdas que la falsan). Estado: «se sostiene» si ningún Δg > 0; «se falsa» si
    alguno tiene Δg > 0 con p_RW < 0,05; si no, «no concluyente»."""
    sub = t[(t["préstamo"] == B_MED) & (t["producto del corto"] == REB_NO)]
    pos = sub[sub["Δg"] > 0]
    sig = pos[pos.p_RW < 0.05]
    cells = [f"{r['vehículo']} {r['reajuste']} {pp(r['Δg'])} pp (p_RW {fmt_p(r.p_RW)})" for _, r in sig.iterrows()]
    if pos.empty:
        return "se sostiene", f"se sostiene (Δg ≤ 0 en los {len(sub)} cortos)", cells
    if cells:
        return "se falsa", "se falsa: " + "; ".join(cells), cells
    return ("no concluyente", f"no concluyente ({len(pos)} de {len(sub)} cortos con Δg > 0, ninguno con p_RW < 0,05)",
            cells)


def fam_label(fam: Family) -> str:
    return f"{fam.key} · {fam.bench_label}, {fam.window}"


def h3_verdict_lines(res: dict, fams: dict) -> tuple[list, list, dict]:
    """Veredicto escrito de H3 (a-i) y (b) en CADA familia de productos reales (H3_REAL, con y sin
    el 5–6-2-2018) y, aparte, en las de sensibilidad con VXX. Devuelve (líneas de a-i, líneas de
    b, estado de b por familia)."""
    la, lb, st = [], [], {}
    for k, vehs in H3_REAL.items():
        if k not in res:
            continue
        f = fams[k]
        parts = []
        for v in vehs:
            if v in set(res[k]["vehículo"]):
                a1 = verdict_a1(res[k], v)
                parts.append(f"{v}: {a1[0]} ({a1[1]})")
        la.append(f"  - {fam_label(f)}: " + "; ".join(parts) + ".")
        state, txt, _ = verdict_b_family(res[k])
        st[k] = state
        lb.append(f"  - {fam_label(f)}: {txt}.")
    for k in sorted(x for x in res if fams[x].kind == SENS):
        state, txt, _ = verdict_b_family(res[k])
        st[k] = state
        lb.append(f"  - {fam_label(fams[k])} (sensibilidad; VXX con prima en 2022): {txt}.")
    return la, lb, st


# ───────────────────────────── episodios ─────────────────────────────

def episode_returns(r: pd.Series, a, b) -> tuple[float, float, float]:
    """Resultado del cierre de a al cierre de b, peor día y caída máxima dentro (ruina absorbente)."""
    x = r[(r.index > pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))].dropna()
    if len(x) < 2 or r.dropna().index[0] > pd.Timestamp(a):
        return np.nan, np.nan, np.nan
    x = r22.absorb(x)
    w = (1 + x).cumprod()
    return float(w.iloc[-1] - 1), float(x.min()), float((w / w.cummax().clip(lower=1.0) - 1).min())


def episodes(fams: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    s_main = {f.title: f for f in fams.values() if f.kind == "sintético" and f.window == "principal"}
    S = {}
    for conv in ("settle", "1600"):
        S[f"−1x sintético ({CONV_ES[conv]})"] = vix.synthetic("-1x", conv)
        S[f"−0,5x sintético ({CONV_ES[conv]})"] = vix.synthetic("-0.5x", conv)
    S["VYLD sintético (liquidación)"] = vix.synthetic("vyld", "settle")
    f05 = s_main["−0,5x sintético (liquidación)"]
    for fq in FREQS:
        S[f"corto +1,5x sintético {FREQ_ES[fq]} (−0,5x; 5,2%, sin rem.)"] = f05.rules[rule_name(SYN_LONGS["+1.5x"][0], fq, B_MED, REB_NO)]
    f1 = s_main["−1x sintético (liquidación)"]
    S["corto +2x sintético diario (−1x; 5,2%, sin rem.)"] = f1.rules[rule_name(SYN_LONGS["+2x"][0], "D", B_MED, REB_NO)]
    R = real_returns()
    S["SVXY real"] = R.SVXY
    r5 = fams["R5"]
    for fq in FREQS:
        S[f"corto UVXY real {FREQ_ES[fq]} (igual a SVXY; 5,2%, sin rem.)"] = r5.rules[rule_name("UVXY", fq, B_MED, REB_NO)]
    S["SVIX real"] = R.SVIX
    r3 = fams["R3"]
    for fq in ("D", "W"):
        S[f"corto UVIX real {FREQ_ES[fq]} (−1x; 5,2%, sin rem.)"] = r3.rules[rule_name("UVIX", fq, B_MED, REB_NO)]
    S["SPY"] = R.SPY
    lrs = lrs_series()
    if lrs is not None:
        S[LRS] = lrs
    res, worst = {}, {}
    for k, r in S.items():
        res[k], worst[k] = {}, {}
        for ep, (a, b) in EPISODES.items():
            tot, wd, _ = episode_returns(r, a, b)
            res[k][EPISODE_ES[ep]] = tot
            worst[k][EPISODE_ES[ep]] = wd
    return pd.DataFrame(res).T, pd.DataFrame(worst).T


# ───────────────────────────── tamaño y Kelly con colas gordas ─────────────────────────────

def growth_k(u: np.ndarray, rf: np.ndarray, k: float) -> float:
    """Crecimiento logarítmico anual de k unidades de exposición: r = rf + k·u (u: exceso del −1x
    sobre la letra por unidad), pérdidas acotadas en −99,99% (convención del laboratorio)."""
    return float(np.mean(np.log1p(np.maximum(rf + k * u, CLIP))) * 252)


def fmt_k(x: float, d: int = 2) -> str:
    """L* con coma decimal; «≥ 3» si está censurado en el tope (como kelly.fmt_lstar)."""
    return f"≥ {num_es(K_MAX, 0)}" if not np.isfinite(x) else num_es(x, d)


def lstar_exact(u: np.ndarray, rf: np.ndarray, k_max: float = K_MAX) -> float:
    """k de [0, k_max] que maximiza growth_k; np.inf si la curva aún sube en el tope (censurado)."""
    res = minimize_scalar(lambda k: -growth_k(u, rf, k), bounds=(0.0, k_max), method="bounded",
                          options={"xatol": 1e-4})
    if growth_k(u, rf, k_max) >= -res.fun:
        return np.inf
    return float(res.x)


def gpd_growth(u: np.ndarray, rf: np.ndarray, k: float, fit: dict, grid: np.ndarray) -> float:
    """Crecimiento con la cola de pérdidas GPD: días del cuerpo (pérdida ≤ umbral) tal cual y,
    con probabilidad ζ, una pérdida u_umbral + Y con Y ~ GPD (integral por cuantiles)."""
    loss = -u
    body = loss <= fit["u"]
    zeta = 1 - body.mean()
    gb = np.mean(np.where(body, np.log1p(np.maximum(rf + k * u, CLIP)), 0.0))
    Y = fit["u"] + genpareto.ppf(grid, fit["xi"], scale=fit["beta"])
    gt = np.mean(np.log1p(np.maximum(rf.mean() - k * Y, CLIP)))
    return float((gb + zeta * gt) * 252)


KELLY_SERIES = ("−1x sintético (liquidación)", "−1x sintético (16:00)",
                "−1x sintético (liquidación), sin el 5–6-2-2018", "−1x sintético (16:00), sin el 5–6-2-2018")


def kelly_inputs() -> dict:
    """Exceso por unidad de exposición corta al índice (u) y letra (rf) de cada serie."""
    if "kin" in _CACHE:
        return _CACHE["kin"]
    out = {}
    for conv in ("settle", "1600"):
        r = vix.synthetic("-1x", conv)
        rf = vix.cash(r.index)
        u = r - rf
        for w, (a, b) in SYN_WINDOWS.items():
            m = (u.index > a) & (u.index <= b)
            out[(f"−1x sintético ({CONV_ES[conv]})", w)] = (u[m], rf[m])
            if w != "desde 28-2-2018":
                uu, rr = vix.drop_days(u[m]), vix.drop_days(rf[m])
                out[(f"−1x sintético ({CONV_ES[conv]}), sin el 5–6-2-2018", w)] = (uu, rr)
    R = real_returns()
    for t, E, a, lab in (("SVXY", 0.5, POST, "SVXY real (por unidad: ÷ 0,5)"), ("SVIX", 1.0, VS_START, "SVIX real")):
        r = R[t][R.index >= a].dropna().iloc[1:]
        rf = vix.cash(r.index)
        out[(lab, f"{dmy(r.index[0])} → {dmy(r.index[-1])}")] = ((r - rf) / E, rf)
    _CACHE["kin"] = out
    return out


def kelly_table(rng) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    grid = (np.arange(20000) + 0.5) / 20000
    rows, curves = [], {}
    for (name, w), (u, rf) in kelly_inputs().items():
        uv, rv = u.to_numpy(), rf.to_numpy()
        mu, sd = uv.mean() * 252, uv.std() * np.sqrt(252)
        lx_ = lstar_exact(uv, rv)
        ks = []
        n = len(uv)
        for _ in range(B_KELLY):
            ii = stats.stationary_indices(n, BLOCK_KELLY, rng)
            ks.append(lstar_exact(uv[ii], rv[ii]))
        ks = np.array(ks)
        lo, hi = np.percentile(np.where(np.isinf(ks), K_MAX * 10, ks), [5, 95])
        row = {"serie": name, "ventana": w, "sesiones": n, "μ exceso anual": mu, "σ anual": sd,
               "asimetría diaria": float(pd.Series(uv).skew()), "curtosis diaria": float(pd.Series(uv).kurt()),
               "peor día (k = 1)": float(uv.min()), "k que arruina el peor día": 1 / float(-uv.min()),
               kelly.LSTAR_APROX: mu / sd ** 2, kelly.LSTAR_EXACTO: lx_,
               "IC90 L* exacto": f"[{fmt_k(lo)}; {fmt_k(hi if hi < K_MAX else np.inf)}]",
               "g en L* exacto": growth_k(uv, rv, lx_) if np.isfinite(lx_) else np.nan,
               "g con k = 0,5": growth_k(uv, rv, 0.5), "g con k = 1": growth_k(uv, rv, 1.0)}
        if w == "principal":
            fit = gpd_fit(-uv, 0.95)
            gk = np.array([gpd_growth(uv, rv, k, fit, grid) for k in K_GRID])
            kk = K_GRID[int(np.argmax(gk))]
            res = minimize_scalar(lambda k: -gpd_growth(uv, rv, k, fit, grid), bounds=(max(kk - 0.05, 0), kk + 0.05),
                                  method="bounded")
            row["L* con cola GPD"] = float(res.x)
            row["ξ de las pérdidas"] = fit["xi"]
            curves[(name, "GPD")] = pd.Series(gk, index=K_GRID)
            _CACHE[("fit", name)] = (fit, uv, rv)
        if w == "principal" or name.startswith("SV"):
            curves[(name, "historia")] = pd.Series([growth_k(uv, rv, k) for k in K_GRID], index=K_GRID)
        rows.append(row)
    kt = pd.DataFrame(rows).set_index(["serie", "ventana"])
    # tamaño: probabilidades de cola por exposición (principal, cola GPD de las pérdidas del −1x), con
    # las dos convenciones y con y sin el 5–6-2-2018
    size = []
    for name in KELLY_SERIES:
        fit, uv, rv = _CACHE[("fit", name)]
        u, rf = kelly_inputs()[(name, "principal")]
        for k in (0.25, 0.5, 0.75, 1.0):
            r = rf + k * u
            p50 = gpd_rate(fit, 0.5 / k)
            pr = gpd_rate(fit, 1.0 / k)
            size.append({"serie": name, "k": k, "CAGR": metrics.cagr(r), "g": growth_k(uv, rv, k),
                         "maxDD": metrics.max_dd(r), "peor día": float(r.min()),
                         "años entre días ≤ −50% (GPD)": round(1 / p50) if p50 and p50 > 0 else np.inf,
                         "años entre ruinas en un día (GPD)": round(1 / pr) if pr and pr > 0 else np.inf})
    lrs = lrs_series()
    if lrs is not None:
        x = between(lrs, vix.ST_START + pd.Timedelta(days=1), END).dropna()
        size.append({"serie": f"{LRS} (referencia, artículo 2)", "k": np.nan, "CAGR": metrics.cagr(x), "g": ann_g(x),
                     "maxDD": metrics.max_dd(x), "peor día": float(x.min())})
    spy = between(real_returns().SPY, vix.ST_START + pd.Timedelta(days=1), END).dropna()
    size.append({"serie": "SPY (referencia)", "k": np.nan, "CAGR": metrics.cagr(spy), "g": ann_g(spy),
                 "maxDD": metrics.max_dd(spy), "peor día": float(spy.min())})
    return kt, curves, pd.DataFrame(size).set_index(["serie", "k"])


# ───────────────────────────── figuras ─────────────────────────────

def _year_axis(ax, base: int = 4):
    ax.xaxis.set_major_locator(mdates.YearLocator(base=base))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def fig_roll(dec: pd.DataFrame, rep: Report):
    c = dec.cumsum()
    fig, ax = plt.subplots(figsize=(FIG_W, 5.0))
    yrs = len(dec) / 252
    for col, lab, colr, ls in (("total", "índice corto ER (log)", INK, "-"),
                               ("roll", "roll acumulado", ORANGE, "-"),
                               ("nivel", "cambio de nivel del futuro a 30 días", BLUE, "-")):
        ax.plot(c.index, c[col], color=colr, lw=1.8 if col != "total" else 1.4, ls=ls,
                label=f"{lab}: {pp(c[col].iloc[-1] / yrs)} puntos log/año")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(ylabel="Suma de rendimientos logarítmicos", title="El índice corto pierde por el roll, no por el VIX")
    _year_axis(ax)
    ax.grid(alpha=0.25)
    h, lab = ax.get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=1)
    comma_fig(fig)
    rep.fig(fig, "roll", "Figura: rendimiento logarítmico acumulado del índice corto reconstruido (ER, liquidación de "
                         "CFE) descompuesto en el cambio de nivel del futuro de vencimiento constante a 30 días y el roll, "
                         f"{dmy(dec.index[0])} → {dmy(dec.index[-1])}")


def fig_drag(D: pd.DataFrame, rep: Report):
    keys = ["−1x sintético", "−0,5x sintético", "+1x sintético", "+1,5x sintético", "+2x sintético"]
    sub = D.xs("liquidación", level="convención").loc[keys]
    comps = [("L × g índice", "L × índice", BLUE), ("arrastre ½(L²−L)Σx²", "arrastre ½(L² − L)·Σx²", ORANGE),
             ("letra − gasto", "letra − gasto", AQUA), ("resto (orden ≥ 3 y seguimiento)", "resto (orden ≥ 3)", VIOLET)]
    fig, ax = plt.subplots(figsize=(FIG_W, 5.6))
    n = len(keys)
    wbar = 0.19
    y = np.arange(n)
    for j, (c, lab, col) in enumerate(comps):
        ax.barh(y + (j - 1.5) * wbar, 100 * sub[c].to_numpy(), height=wbar * 0.9, color=col, label=lab)
    ax.scatter(100 * sub["g fondo"], y, marker="D", s=70, color=INK, zorder=5, label="crecimiento del fondo (suma)")
    for yi, g in zip(y, sub["g fondo"]):
        ax.annotate(num_es(100 * g, 1), (100 * g, yi), xytext=(-9, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=11, color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(100 * min(sub["g fondo"].min(), sub["L × g índice"].min()) - 35, None)
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace(" sintético", "") for k in keys])
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set(xlabel="Puntos log al año", title="Crecimiento de cada sintético y sus componentes, 2005–2026")
    ax.grid(axis="x", alpha=0.25)
    h, lab = ax.get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=2, fontsize=11)
    comma_fig(fig)
    rep.fig(fig, "arrastre", "Figura: descomposición del crecimiento logarítmico anual de cada sintético sobre el índice "
                             "corto (liquidación de CFE, ventana principal): L por el crecimiento del índice, arrastre de "
                             "segundo orden ½(L² − L)·Σx², letra − gasto y resto de orden superior")


def fig_tails(T: dict, rep: Report):
    S, fits = T["series"], T["fits"]
    keys = [UP_SET, UP_16, UP_SET + SIN_FEB, "Pérdida diaria del SPY"]
    cols = [BLUE, ORANGE, BLUE, GREY]
    fig, ax = plt.subplots(figsize=(FIG_W, 5.9))
    for k, col in zip(keys, cols):
        lab = (k.replace("Subida diaria del índice corto", "subidas del índice corto")
                .replace("Pérdida diaria del SPY", "caídas del SPY"))
        sin = k.endswith(SIN_FEB)
        if not sin:                          # sin el 5–6-2-2018 los puntos son los mismos menos dos
            s = np.sort(S[k][S[k] > 0].to_numpy())[::-1]
            rate = np.arange(1, len(s) + 1) / len(S[k]) * 252
            ax.loglog(100 * s, rate, ".", color=col, ms=4, alpha=0.8, label=f"{lab} (observado)")
        f = fits[k]
        xx = np.geomspace(f["u"] * 1.001, 3.0, 200)
        ax.loglog(100 * xx, [gpd_rate(f, v) for v in xx], ":" if sin else "-", color=col, lw=2.0 if sin else 1.6,
                  label=f"{lab}: cola GPD, ξ = {num_es(f['xi'], 2)}")
    for v, t in ((100, "−1x a cero"), (200, "−0,5x a cero")):
        ax.axvline(v, color=MUTED, ls="--", lw=1)
        ax.text(v * 1.04, 3e-3, t, fontsize=11, color=MUTED, rotation=90, va="bottom")
    ax.set_xlim(2, 320)
    ax.set_ylim(1e-3, 300)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num_es(v, 0) + "%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num_es(v, 3 if v < 0.01 else 2 if v < 1 else 0)))
    ax.set(xlabel="Movimiento diario (escala log)", ylabel="Días al año con un movimiento mayor",
           title="Colas: subidas diarias del índice corto frente a caídas del SPY")
    ax.grid(alpha=0.25, which="major")
    h, lab = ax.get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=1, fontsize=11)
    rep.fig(fig, "colas", "Figura: frecuencia anual de subidas diarias del índice corto mayores que cada tamaño (liquidación "
                          "y 16:00) y de caídas diarias del SPY, 2005–2026, con la cola GPD ajustada por encima del "
                          "percentil 95 (de puntos, la de la liquidación sin el 5 y el 6-2-2018); escalas logarítmicas. "
                          "Las líneas verticales marcan la subida que deja a cero un −1x (100%) y un −0,5x (200%)")


def fig_h3(fams: dict, rep: Report):
    panels = [("R1", "SVXY comprado", "UVXY", "−0,5x desde el 28-2-2018: SVXY frente al corto de UVXY"),
              ("R3", "SVIX comprado", "UVIX", "−1x desde el 31-3-2022: SVIX frente al corto de UVIX")]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 8.6))
    handles = {}
    for ax, (key, blab, veh, title) in zip(axes, panels):
        f = fams[key]
        series = [(blab, f.bench, INK, "-")]
        for fq, col in zip(FREQS, (BLUE, ORANGE, AQUA)):
            series.append((f"corto {FREQ_ES[fq]}, préstamo 5,2%, sin remunerar",
                           f.rules[rule_name(veh, fq, B_MED, REB_NO)], col, "-"))
        series.append(("corto diario, sin préstamo, letra − 0,5%", f.rules[rule_name(veh, "D", 0.0, REB_YES)], BLUE, ":"))
        lo, hi = 1.0, 1.0
        for lab, r, col, ls in series:
            x = r22.absorb(r.dropna())
            w = (1 + x).cumprod()
            alive = w[w > 0.05]
            lo, hi = min(lo, alive.min()), max(hi, alive.max())
            ln, = ax.semilogy(w.index, w.clip(lower=1e-3).values, color=col, ls=ls, lw=1.6)
            handles.setdefault(lab.replace(blab, "inverso comprado (SVXY o SVIX)"), ln)
            rd = r22.ruin_date(x)
            if rd is not None:
                ax.annotate(f"ruina el {dmy(rd)}", (rd, lo * 1.02), xytext=(6, 4), textcoords="offset points",
                            fontsize=11, color=MUTED)
        ax.set_ylim(lo / 1.15, hi * 1.15)
        ticks = [t for t in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10) if lo / 1.15 <= t <= hi * 1.15]
        ax.set_yticks(ticks)
        ax.set_title(title, fontsize=13)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num_es(v, 2 if v < 1 else 1 if v < 2 else 0)))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylabel("Valor de 1 $ (log)")
        yrs = (f.bench.index[-1] - f.bench.index[0]).days / 365.25
        _year_axis(ax, 2 if yrs > 5 else 1)
        ax.grid(alpha=0.25)
    legend_band(fig, list(handles.values()), list(handles), where="bottom", ncol=1, fontsize=11)
    fig.suptitle("Comprar el inverso o vender el ETP largo a igual exposición")
    comma_fig(fig)
    rep.fig(fig, "comprar_o_vender", "Figura: valor de 1 dólar en el inverso comprado y en el corto de igual exposición del "
                                     "ETP largo (reajuste diario, semanal o mensual; préstamo 5,2% y producto sin remunerar, "
                                     "y la versión diaria sin préstamo); cierres de Yahoo, escala logarítmica")


def fig_paths(periods: dict, rep: Report):
    cases = [k for k in periods if k[0].startswith("UVXY (28")]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.6))
    for ax, key in zip(axes, cases):
        per = periods[key]
        ok = np.sign(per.camino) == np.sign(1 - per.VR)
        ax.scatter(100 * per.aprox[ok], 100 * per.camino[ok], s=14, color=BLUE, alpha=0.7, label="mismo signo que 1 − VR")
        ax.scatter(100 * per.aprox[~ok], 100 * per.camino[~ok], s=22, color=ORANGE, marker="x",
                   label="signo distinto")
        lim = 100 * np.nanpercentile(np.abs(np.r_[per.aprox, per.camino]), 99.5)
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.axvline(0, color=GRID, lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"UVXY, {FREQ_ES[key[1]]} ({num_es(100 * ok.mean(), 0)}% de acuerdo)", fontsize=12)
        ax.set_xlabel("s(1 + s)/2 · (Σy² − (Σy)²), %")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Camino exacto, % del capital")
    h, lab = axes[0].get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=2, fontsize=11)
    fig.suptitle("Corto de UVXY sin reajuste frente al diario (s = 1/3)")
    comma_fig(fig)
    rep.fig(fig, "camino_uvxy", "Figura: por periodo, ventaja del corto de UVXY que no se reajusta frente al reajustado cada "
                                "día (sin costes ni intereses) y su aproximación de segundo orden, 28-2-2018 → 25-9-2026; en "
                                "naranja, los periodos en que el signo no coincide con el de 1 − VR")


def fig_kelly(curves: dict, kt: pd.DataFrame, rep: Report):
    fig, ax = plt.subplots(figsize=(FIG_W, 5.4))
    spec = [(("−1x sintético (liquidación)", "historia"), BLUE, "-", "liquidación (16:15 hasta 2020)"),
            (("−1x sintético (liquidación), sin el 5–6-2-2018", "historia"), BLUE, ":", "liquidación, sin el 5–6-2-2018"),
            (("−1x sintético (16:00)", "historia"), ORANGE, "-", "16:00"),
            (("−1x sintético (liquidación)", "GPD"), BLUE, "--", "liquidación con cola GPD"),
            (("−1x sintético (liquidación), sin el 5–6-2-2018", "GPD"), BLUE, "-.", "liquidación sin el 5–6-2-2018, cola GPD"),
            (("−1x sintético (16:00)", "GPD"), ORANGE, "--", "16:00 con cola GPD")]
    for key, col, ls, lab in spec:
        if key not in curves:
            continue
        c = curves[key]
        ax.plot(c.index, 100 * c.clip(lower=-0.6).values, color=col, ls=ls, lw=1.8, label=lab)
        j = int(np.argmax(c.values))
        ax.plot(c.index[j], 100 * c.iloc[j], "o", color=col, ms=7)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(-60, None)
    ax.set(xlabel="Exposición corta al índice k (1 = un −1x con todo el capital)",
           ylabel="Crecimiento logarítmico, %/año", title="Crecimiento según el tamaño, 2005–2026")
    ax.grid(alpha=0.25)
    h, lab = ax.get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=2, fontsize=11)
    comma_fig(fig)
    rep.fig(fig, "kelly", "Figura: crecimiento logarítmico anual de k·(−1x) + (1 − k)·letra, reajustado a diario, con la "
                          "historia y con la cola de pérdidas GPD (percentil 95); el punto marca el máximo (L*). Curvas "
                          "recortadas en −60%/año")


# ───────────────────────────── cálculo e informe ─────────────────────────────

def compute() -> dict:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rng = np.random.default_rng(SEED)
    term, dec = term_stats()
    prem, pc = futures_premium()
    fams_l = real_families() + synthetic_families()
    fams = {f.key: f for f in fams_l}
    res, summ = {}, []
    for f in fams_l:
        t, s = evaluate(f, rng, with_spa=True)             # SPA en todas las familias (pre-registro)
        res[f.key] = t
        summ.append(s)
    paths, periods = path_test(fams, res)
    return {"term": term, "dec": dec, "prem": prem, "pc": pc, "vrp": variance_premium(), "drag": drag_table(),
            "tails": tails(rng), "fams": fams, "res": res, "summ": pd.DataFrame(summ).set_index("familia"),
            "dec_h3": decomposition(fams, rng), "paths": paths, "periods": periods, "vr": vr_test(),
            "episodes": episodes(fams), "kelly": kelly_table(rng), "vxx": vxx_premium()}


def run(C: dict | None = None):
    warnings.filterwarnings("ignore")
    rep = Report("r27_vix_carry", "R27 — La prima y cómo cobrarla: volatilidad corta y venta de los ETPs del VIX")
    for f in list(rep.dir.glob("tabla_*.csv")) + list(rep.dir.glob("familia_*.csv")):
        f.unlink()
    C = C or compute()
    tab = r26.Tables(rep)
    N = {}

    def T_(key, *a, **k):
        tab(*a, **k)
        N[key] = tab.n

    fams, res, summ = C["fams"], C["res"], C["summ"]
    term, prem, vrp, drag = C["term"], C["prem"], C["vrp"], C["drag"]
    Tl = C["tails"]
    kt, curves, size = C["kelly"]
    ep_res, ep_worst = C["episodes"]
    dh3, paths, vr = C["dec_h3"], C["paths"], C["vr"]
    for k, t in res.items():
        t.to_csv(rep.dir / f"familia_{k}.csv")

    # ── cifras para el texto
    tp = term.loc["principal (21-12-2005 →)"]
    pr = prem.loc["principal"]
    pc = C["pc"]
    worst_c = pc["rend. largo"].idxmax()
    vp = vrp.loc["1990–2026"]
    KT = {k: key_table(res[k]) for k in ("R1", "R2", "R3", "R4", "R4s", "R5", "R5s", "R6", "R2v", "R3v")}
    a1 = {k: verdict_a1(res[k], v) for k, v in (("R1", "UVXY"), ("R3", "UVIX"), ("R4", "UVXY"), ("R4s", "UVXY"),
                                                  ("R5", "UVXY"), ("R5s", "UVXY"), ("R2", "UVIX"), ("R2", "UVXY"))}
    vb = {k: verdict_b(res[k], v) for k, v in (("R1", "UVXY"), ("R3", "UVIX"), ("R4", "UVXY"), ("R4s", "UVXY"),
                                                 ("R5", "UVXY"), ("R5s", "UVXY"), ("R2", "UVIX"))}
    la_lines, lb_lines, vb_state = h3_verdict_lines(res, fams)
    b_fals = [k for k in H3_REAL if vb_state.get(k) == "se falsa"]
    b_hold = [k for k in H3_REAL if vb_state.get(k) == "se sostiene"]
    b_inc = [k for k in H3_REAL if vb_state.get(k) == "no concluyente"]
    vxx_t, vxx = C["vxx"]
    b_text = "; ".join(x for x in (
        ("se falsa en " + y_list(b_fals)) if b_fals else "",
        ("no es concluyente en " + y_list(b_inc)) if b_inc else "",
        ("se sostiene en " + y_list(b_hold)) if b_hold else "") if x)
    fr = dh3.xs("diario", level="reajuste")
    fr_uvxy = fr.loc[("R1", "corto UVXY frente a SVXY")]
    fr_vixy = fr.loc[("R1", "corto VIXY frente a SVXY")]
    fr_uvix = fr.loc[("R3", "corto UVIX frente a SVIX")]
    fr_pre = fr.loc[("R4", "corto UVXY frente a SVXY")]
    fr_pre_s = fr.loc[("R4s", "corto UVXY frente a SVXY")]
    ex0 = lambda r: r["_lo"] > 0 or r["_hi"] < 0  # noqa: E731 — IC90 que excluye el 0
    a2_rej = [lab for lab, r in (("UVXY/SVXY desde 2018", fr_uvxy), ("UVIX/SVIX desde 2022", fr_uvix)) if ex0(r)]
    pw = paths.xs("semanal", level="reajuste")
    pm = paths.xs("mensual", level="reajuste")
    n_cases = len(paths)
    ag_w = int(paths["acuerdo en el agregado"].eq("sí").sum())
    ag_l = int(paths["acuerdo en el agregado (log)"].eq("sí").sum())
    m_main = Tl["main"]
    st_row = m_main.loc["Subida diaria del índice corto (liquidación)"]
    h16_row = m_main.loc["Subida diaria del índice corto (16:00)"]
    spy_row = m_main.loc["Pérdida diaria del SPY"]
    sens = Tl["sens"]
    s_set = sens.xs("índice corto (liquidación)", level="serie")
    s_sin = sens.xs("índice corto (liquidación), sin el 5–6-2-2018", level="serie")
    sin_row = m_main.loc[UP_SET + SIN_FEB]
    k_sin16 = kt.loc[("−1x sintético (16:00), sin el 5–6-2-2018", "principal")]
    k_set = kt.loc[("−1x sintético (liquidación)", "principal")]
    k_16 = kt.loc[("−1x sintético (16:00)", "principal")]
    k_sin = kt.loc[("−1x sintético (liquidación), sin el 5–6-2-2018", "principal")]
    real_fams = summ[summ.tipo == "real"]
    sens_fams = summ[summ.tipo == SENS]
    syn_main = summ[(summ.tipo == "sintético") & (summ.ventana == "principal")]
    syn_other = summ[(summ.tipo == "sintético") & (summ.ventana != "principal")]
    n_rules = int(summ[summ.tipo != SENS].reglas.sum())
    n_fams = int((summ.tipo != SENS).sum())
    d_uv = drag.loc[("UVXY real, desde 28-2-2018", "liquidación")]
    d_sv = drag.loc[("SVXY real, desde 28-2-2018", "liquidación")]
    d1 = drag.loc[("−1x sintético", "liquidación")]
    d05 = drag.loc[("−0,5x sintético", "liquidación")]
    lrs_rows = [i for i in size.index if str(i[0]).startswith(LRS)]
    lrs_cagr = size.loc[lrs_rows[0], "CAGR"] if lrs_rows else np.nan
    syn05 = synth_summary(list(fams.values()), res)
    ep = ep_res
    e18 = EPISODE_ES["feb-2018"]
    e20 = EPISODE_ES["mar-2020"]
    e08 = EPISODE_ES["2008"]
    Rr = real_returns()
    um = vix.to_calendar(vix.etp_model("UVXY"), Rr.index)
    uv_d = {d: (Rr.UVXY.loc[d], um.loc[d]) for d in ("2018-02-05", "2018-02-07", "2018-02-08")}
    syn_spa = int((syn_main.p_SPA < 0.05).sum())
    syn_spa_o = int((syn_other.p_SPA < 0.05).sum())
    log_no = paths[paths["acuerdo en el agregado (log)"] == "no"]
    log_no_fq = set(log_no.index.get_level_values("reajuste"))
    worst_path = float(-paths["peor periodo sin reajuste"].min())

    rep.p(__doc__.split("\n", 2)[2])
    pos_summary = len(rep.lines)

    # ── 1. mecanismo
    rep.h("1. Mecanismo: la prima está en el futuro")
    rep.h("1.1 Estructura de plazos y roll", 3)
    T_("term", "Estructura de plazos por ventana (fracción de sesiones) y descomposición del crecimiento logarítmico anual "
       "del índice corto (contratos 1.º y 2.º) y del medio (4.º a 7.º, desde el 25-4-2006) en cambio de nivel del futuro "
       "de vencimiento constante y roll. VIX y VIX3M de Cboe (VIX3M desde el 18-9-2009)", term,
       pct={"F1 > VIX", "F2 > F1", "CM30 > VIX", "VIX < VIX3M", "F1/VIX − 1 (mediana)", "F2/F1 − 1 (media)",
            "g índice corto", "nivel corto", "roll corto", "g índice medio", "nivel medio", "roll medio"})
    fig_roll(C["dec"], rep)
    rep.p(f"El roll es lo que cuesta mantener la exposición a vencimiento constante: cada día el índice vende parte del "
          f"primer contrato y compra el segundo, que en contango es más caro. El nivel del futuro a 30 días vuelve a su "
          f"media y en 20 años aporta {pp(tp['nivel corto'])} puntos log al año; el roll resta "
          f"{num_es(-100 * tp['roll corto'], 1)}. La pérdida de VXX o VIXY no exige que el VIX baje. El medio plazo "
          f"(VIXM, VXZ) pierde menos ({num_es(-100 * tp['roll medio'], 1)} puntos de roll) porque su curva es más plana.")
    rep.h("1.2 La prima del futuro frente al VIX", 3)
    T_("prem", "Prima del futuro: (a) por contrato, precio 21 sesiones de CFE antes del vencimiento menos su liquidación "
       "final (t con Newey-West de 3 retardos; los contratos casi no se solapan); (b) diario, futuro de 30 días (CM30) "
       "menos el VIX de Cboe del día y de 21 sesiones después (t de Hansen–Hodrick con 20 retardos)", prem,
       pct={"F > liquidación", "rend. medio del largo", "peor rend. del corto", "CM30 > VIX", "CM30 > VIX a 21 sesiones"})
    rep.p(f"El futuro no es una previsión insesgada del VIX: cotiza con prima y la prima se cobra de media "
          f"({num_es(pr['prima media (pts)'], 2)} puntos por contrato; mediana {num_es(pr['prima mediana (pts)'], 2)}). La "
          f"distribución es muy asimétrica: el peor contrato para un corto fue el de vencimiento {dmy(pc.loc[worst_c, 'vencimiento'])}, "
          f"que subió un {pct_es(pc.loc[worst_c, 'rend. largo'], 0)} en sus últimas 21 sesiones (de "
          f"{num_es(pc.loc[worst_c, 'F entrada'], 2)} a {num_es(pc.loc[worst_c, 'liquidación'], 2)}). Antes de 2018 la "
          f"prima por contrato no es significativa (t = {num_es(prem.loc['hasta 27-2-2018', 't (NW 3)'], 2)}). La "
          "liquidación final se toma del precio del día del vencimiento en los ficheros de CFE (que coincida con la "
          "liquidación especial de apertura está sin verificar).")
    rep.h("1.3 La prima de varianza", 3)
    T_("vrp", "VIX de Cboe frente a la volatilidad realizada del S&P 500 (^GSPC, rendimientos logarítmicos) en las 21 "
       "sesiones siguientes; t de Hansen–Hodrick con 20 retardos", vrp, pct={"VIX > RV"}, digits=3)
    rep.p("La prima de varianza (Carr y Wu 2009; Bollerslev, Tauchen y Zhou 2009) es el motivo económico: el seguro contra "
          "la volatilidad se paga por encima de su coste esperado. La prima del futuro (1.2) es la forma en que la cobra un "
          f"producto de futuros del VIX. Las dos son positivas de media y muy negativas en los días que importan: el VIX "
          f"quedó {num_es(-vp['mínimo'], 0)} puntos por debajo de la volatilidad realizada de las 21 sesiones siguientes el "
          f"{dmy(vp['fecha del mínimo'])}.")
    rep.h("1.4 El arrastre del reajuste diario", 3)
    T_("drag", "Descomposición del crecimiento logarítmico anual: log(1 + r) = L·log(1 + x) − ½(L² − L)·x² + (letra − "
       "gasto) + resto, con x el rendimiento diario del índice corto (ER) en la convención de cada producto y en su "
       "calendario. En los ETPs reales el resto incluye su seguimiento del índice", drag,
       pct={"g fondo", "L × g índice", "arrastre ½(L²−L)Σx²", "letra − gasto", "resto (orden ≥ 3 y seguimiento)"}, digits=3)
    fig_drag(drag, rep)
    rep.p(f"El término de segundo orden depende de L² − L: vale lo mismo para UVXY (1,5) y SVXY (−0,5), y para UVIX (2) y "
          f"SVIX (−1). Comprar el inverso no evita el arrastre: el −0,5x sintético lo paga "
          f"({num_es(-100 * d05['arrastre ½(L²−L)Σx²'], 1)} puntos al año) igual que el +1,5x. En el −1x con la "
          f"liquidación, el resto de orden superior ({num_es(100 * d1['resto (orden ≥ 3 y seguimiento)'], 1)} puntos al "
          f"año) sale de pocos días: log(1 − x) no es −x − x²/2 cuando x se acerca a 1 (el 5-2-2018 el índice subió un "
          f"{pct_es(st_row['máximo'], 0)}). En SVIX y UVIX el resto de "
          f"{num_es(100 * drag.loc[('SVIX real, desde 31-3-2022', '16:00'), 'resto (orden ≥ 3 y seguimiento)'], 1)} y "
          f"{num_es(100 * drag.loc[('UVIX real, desde 31-3-2022', '16:00 (LONGVOL)'), 'resto (orden ≥ 3 y seguimiento)'], 1)} puntos es "
          "sobre todo seguimiento: lo que esos fondos rinden por debajo de su índice (R26).")
    rep.h("1.5 Colas", 3)
    T_("tails", "Colas diarias en la ventana principal: picos sobre el umbral del percentil 95 (GPD; IC90 de ξ por "
       f"bootstrap estacionario de bloque 63 y {B_EVT} remuestreos), Hill con las 100 mayores y frecuencia anual de un "
       "movimiento mayor que 20, 50 y 100% según la GPD (vacío si el tamaño queda por debajo del umbral). Las subidas "
       "del índice corto, con las dos convenciones de hora y con y sin el 5–6-2-2018", m_main, digits=4)
    T_("sens", "Sensibilidad al umbral de la cola GPD de las subidas del índice corto, con las dos convenciones y con y "
       "sin el 5–6-2-2018: con una subida del 100% un −1x queda a cero y con una del 200% un −0,5x", sens, digits=4)
    T_("top", "Las 10 mayores subidas diarias del índice corto con la liquidación de CFE, la misma sesión a las 16:00, el "
       "−1x sintético y el VIX de Cboe", Tl["top"],
       pct={"índice corto (liquidación)", "índice corto (16:00)", "−1x sintético (liquidación)", "VIX: cambio del día"})
    fig_tails(Tl, rep)
    rep.p(f"ξ > 0 es una cola de tipo potencia, sin techo: la historia no acota la peor subida. Con 20 años y un solo día "
          f"cerca del 100%, las frecuencias de la GPD por encima de lo observado son extrapolaciones: con la liquidación, "
          f"un día de más del 100% sale una vez cada {fmt_years(s_set.loc[0.95, 'años entre días > 100%'])} años con el "
          f"umbral del percentil 95 y una vez cada {fmt_years(s_set.loc[0.99, 'años entre días > 100%'])} con el del 99 "
          f"(tabla {N['sens']}); a las 16:00, mucho menos. Sin el 5–6-2-2018 la mayor subida es del "
          f"{pct_es(sin_row['máximo'], 0)} ({dmy(sin_row['fecha del máximo'])}), ξ = {num_es(sin_row['ξ (GPD)'], 2)} "
          f"(IC90 {sin_row['IC90 ξ']}) y el día de más del 100% pasa a una vez cada "
          f"{fmt_years(s_sin.loc[0.95, 'años entre días > 100%'])} años (percentil 95): la frecuencia de la ruina de un "
          "−1x la decide en buena parte ese día. Las excedencias llegan en racimos (2008, 2018, 2020), así que "
          "tratarlas como independientes infravalora el riesgo de varios días seguidos.")

    # ── 2. H3
    rep.h("2. H3: comprar el inverso o vender el ETP largo")
    fam_tab = summ[["título", "tipo", "ventana", "sesiones", "inverso comprado", "reglas"]]
    T_("fams", "Familias de H3: un inverso comprado y todos los cortos de igual exposición (ETP largo × reajuste diario, "
       "semanal o mensual × préstamo 0, 2, 5,2 y 18% × producto del corto a la letra − 0,5% o sin remunerar). En los "
       "sintéticos, las ventanas «hasta 27-2-2018» y «desde 28-2-2018» se simulan desde cero. SPA y Romano–Wolf en cada "
       f"familia. En total, {num_es(n_rules, 0)} configuraciones en {n_fams} familias; R2v y R3v (con VXX) son "
       "sensibilidades y no cuentan en el total", fam_tab)
    rep.p("Pesos de igual exposición: frente a SVXY (−0,5x) desde el 28-2-2018, VXX y VIXY a 0,5, UVXY a 1/3 y UVIX a 0,25; "
          "frente a un −1x (SVIX, o SVXY antes del 28-2-2018), VXX y VIXY a 1, UVXY a 2/3 (a 0,5 cuando era 2x) y UVIX a "
          "0,5; frente a VYLD, F/(100·L). Mantenimiento FINRA: 30% (VXX, VIXY), 45% (UVXY 1,5x), 60% (UVXY 2x, UVIX). Las "
          "tablas completas de cada familia (las 48, 72 o 96 reglas con CAGR, caída máxima, Δg, IC90, p, Holm, "
          "Romano–Wolf, incumplimientos y ruina) están en familia_<clave>.csv.")
    rep.h("2.1 Resultados con productos reales", 3)
    cols_pct = {"Δg sin préstamo", "Δg 5,2% sin remunerar", "CAGR (b)", "maxDD (b)"}
    T_("R1", f"R1 · −0,5x, {fams['R1'].window}: corto de igual exposición frente a SVXY comprado (CAGR "
       f"{pct_es(summ.loc['R1', 'CAGR del inverso'])}). (a): sin préstamo, producto a la letra − 0,5%; (b): préstamo 5,2% y "
       f"producto sin remunerar. Δg en %/año (pp); IC90 en pp; p_RW: Romano–Wolf en la familia de {len(fams['R1'].rules)}; "
       "CAGR, caída máxima, incumplimientos del mantenimiento por año y ruina con (b)", KT["R1"], pct=cols_pct, digits=3)
    T_("R3", f"R3 · −1x, {fams['R3'].window}: frente a SVIX comprado (CAGR {pct_es(summ.loc['R3', 'CAGR del inverso'])}); "
       f"familia de {len(fams['R3'].rules)}, sin VXX", KT["R3"], pct=cols_pct, digits=3)
    T_("R4", f"R4 · −1x, {fams['R4'].window}: frente a SVXY comprado, entonces −1x (CAGR "
       f"{pct_es(summ.loc['R4', 'CAGR del inverso'])}); y sin el 5–6-2-2018 (CAGR {pct_es(summ.loc['R4s', 'CAGR del inverso'])}). "
       "Familias de 48", pd.concat({"con el 5–6-2-2018": KT["R4"], "sin el 5–6-2-2018": KT["R4s"]}, names=["muestra"]),
       pct=cols_pct, digits=3)
    T_("R5", f"R5 · SVXY en {fams['R5'].window} (−1x hasta el 27-2-2018 y −0,5x después; el corto cambia de tamaño ese "
       f"cierre), con y sin el 5–6-2-2018; CAGR de SVXY {pct_es(summ.loc['R5', 'CAGR del inverso'])} y "
       f"{pct_es(summ.loc['R5s', 'CAGR del inverso'])}", pd.concat({"con el 5–6-2-2018": KT["R5"], "sin el 5–6-2-2018": KT["R5s"]},
                                                                 names=["muestra"]), pct=cols_pct, digits=3)
    T_("R2R6", f"R2 · −0,5x frente a SVXY desde el {dmy(VS_START)} (con UVIX a 0,25, sin VXX; familia de "
       f"{len(fams['R2'].rules)}; CAGR de SVXY {pct_es(summ.loc['R2', 'CAGR del inverso'])}) y R6 · VYLD desde el "
       f"{dmy(VYLD_START)} (corto de peso F/(100·L) fijado en cada reajuste; CAGR de VYLD "
       f"{pct_es(summ.loc['R6', 'CAGR del inverso'])}; {int(summ.loc['R6', 'sesiones'])} sesiones: muestra muy corta)",
       pd.concat({"R2 (SVXY)": KT["R2"], "R6 (VYLD)": KT["R6"]}, names=["familia"]), pct=cols_pct, digits=3)
    T_("vxx", "VXX en 2022: cociente acumulado entre el cierre de Yahoo y su modelo (índice corto TR reconstruido menos "
       "el gasto, liquidación de CFE; normalizado en el primer cierre común, como en R26)", vxx_t, digits=3)
    vxx_rows = pd.concat({"R2v (SVXY)": KT["R2v"].loc[["VXX"]], "R3v (SVIX)": KT["R3v"].loc[["VXX"]]},
                         names=["familia"])
    T_("vxxr", "Sensibilidad: los cortos de VXX que se quitaron de R2 y R3 (mismas columnas que las tablas anteriores; "
       f"p_RW en las familias con VXX, de {len(fams['R2v'].rules)} reglas)", vxx_rows, pct=cols_pct, digits=3)
    rep.p(f"R2 y R3 empiezan el {dmy(VS_START)}, el primer cierre de SVIX y UVIX. Ese día el VXX de Yahoo cotizaba un "
          f"{pct_es(vxx['r0'] - 1, 0)} por encima de su modelo (tabla {N['vxx']}): Barclays había suspendido la emisión "
          f"del ETN en marzo (R26; fechas de notas de prensa, sin verificar aquí) y el precio de mercado se separó de su "
          f"valor indicativo; el cociente llegó a "
          f"{num_es(vxx['peak'], 2)} el {dmy(vxx['peak_date'])} y en septiembre ya era {num_es(vxx['sep'], 2)}. Un corto "
          f"de VXX de peso 1 abierto ese día cobra la desaparición de esa prima, unos {num_es(100 * vxx['gift'], 1)} pp al año en la "
          "ventana, que no es la prima de la volatilidad. Por eso R2 y R3 no llevan VXX y las familias con VXX (R2v, R3v) "
          f"son una sensibilidad (tabla {N['vxxr']}): con VXX, la mejor regla del SPA de R2v es "
          f"«{sens_fams.loc['R2v', 'mejor regla']}» ({pp(sens_fams.loc['R2v', 'Δg de la mejor'])} pp).")
    fig_h3(fams, rep)
    rep.p(f"Antes de 2018 el resultado lo decide el 5–6-2-2018. SVXY, entonces −1x, reajustaba con su valor liquidativo de "
          f"las 16:15, en el pico: perdió un {pct_es(-ep.loc['SVXY real', e18], 0)} entre el 31-1 y el 9-2-2018. El corto de "
          f"UVXY se reajusta con el cierre de mercado de las 16:00: el 5-2-2018 UVXY cerró con una subida del "
          f"{pct_es(uv_d['2018-02-05'][0], 0)}, frente al {pct_es(uv_d['2018-02-05'][1], 0)} que da su modelo con la "
          f"liquidación de las 16:15 (su valor liquidativo no está en los datos), y el corto diario perdió un "
          f"{pct_es(-ep.loc['corto UVXY real diario (igual a SVXY; 5,2%, sin rem.)', e18], 0)} en el episodio. Es un efecto "
          "del precio de mercado y de la hora a la que se reajusta cada uno, no del préstamo; sin esos dos días la ventaja se "
          f"reduce mucho (tabla {N['R4']}).")

    rep.h("2.2 ¿Solo tamaño y frecuencia? (H3 a)", 3)
    show = dh3.drop(columns=["_lo", "_hi", "letra media"])
    T_("dec", "Descomposición del Δg del corto (sin préstamo, producto a la letra − 0,5%) frente al inverso comprado: Δg real; "
       "Δg de los mismos cortos y compras sobre los modelos de los ETPs (índice reconstruido y gasto del folleto; R26), que "
       "es la parte mecánica; de ella, la parte analítica del tamaño (gasto del inverso + s × gasto del largo − s × 0,5%) y "
       "la de la frecuencia (modelos, frente al diario); y la fricción de los fondos (real − modelos) con IC90 y p bilateral "
       "por bootstrap estacionario emparejado, también desde el cambio de hora de la liquidación (R1). Fracciones anuales",
       show, pct={"Δg real", "Δg modelos", "tamaño: gastos y producto (analítico)", "frecuencia (modelos, frente al diario)",
                  "fricción (real − modelos)", "del largo: g corto real − modelo", "del inverso: −(g real − modelo)"},
       digits=3)
    rep.p("«Solo por el tamaño y la frecuencia» se lee aquí y en R29 (acciones) del mismo modo: el tamaño es lo que da un "
          "corto de peso s con fondos ideales (índice y gasto del folleto): el gasto del inverso que no se compra, el del "
          "largo que el corto cobra en proporción a s y el diferencial del producto del corto sobre s; la frecuencia es el "
          "camino del reajuste no diario. La fricción de los fondos, lo que rinden de menos que su índice más allá del "
          "gasto del folleto, no es ni tamaño ni frecuencia: si es distinta de 0, H3 (a) no se sostiene tal como está "
          "escrita.")
    rep.p(f"Con reajuste diario el corto y el inverso tienen la misma exposición diaria al índice: la mecánica solo aporta "
          f"los gastos y el producto del corto ({pp(fr_uvxy['tamaño: gastos y producto (analítico)'])} pp al año en "
          f"UVXY/SVXY, algo menos tras los costes de reajustar a diario: {pp(fr_uvxy['Δg modelos'])} pp en los modelos). "
          f"El Δg real es {pp(fr_uvxy['Δg real'])} pp. La diferencia, {pp(fr_uvxy['fricción (real − modelos)'])} pp "
          f"(IC90 {fr_uvxy['IC90 fricción']}), es la fricción de los fondos: lo que UVXY y SVXY rinden de menos que un "
          f"producto ideal sobre su índice, que el vendedor de UVXY cobra ({pp(fr_uvxy['del largo: g corto real − modelo'])} "
          f"pp) y el comprador de SVXY paga ({pp(fr_uvxy['del inverso: −(g real − modelo)'])} pp). Desde el 7-10-2020, "
          f"cuando la liquidación pasa a las 16:00 y ya no hay desfase de hora con el cierre de mercado, sigue siendo "
          f"{fr_uvxy['fricción desde 7-10-2020 (pp, IC90)']} pp. En UVIX/SVIX la fricción es mucho mayor, "
          f"{pp(fr_uvix['fricción (real − modelos)'])} pp (IC90 {fr_uvix['IC90 fricción']}), y en VIXY/SVXY "
          f"{pp(fr_vixy['fricción (real − modelos)'])} pp. Antes de 2018 (−1x) es {pp(fr_pre['fricción (real − modelos)'])} "
          f"pp con el 5–6-2-2018 y {pp(fr_pre_s['fricción (real − modelos)'])} sin él; en esa cifra siguen el 7 y el "
          f"8-2-2018, en que UVXY subió al cierre un {pct_es(uv_d['2018-02-07'][0], 0)} y un {pct_es(uv_d['2018-02-08'][0], 0)} "
          f"frente al {pct_es(uv_d['2018-02-07'][1], 0)} y el {pct_es(uv_d['2018-02-08'][1], 0)} de su modelo. Qué compone la fricción (costes de reajuste de los "
          "fondos, ejecución, gastos fuera de la comisión de gestión, primas y descuentos del precio de mercado) está sin "
          "verificar. Con reajuste semanal o mensual la parte mecánica añade el camino (sección 2.3), que puede ser de "
          "cualquier signo.")

    rep.h("2.3 Reajuste no diario y cociente de varianzas (H3 a)", 3)
    order = ["peso del corto s", "periodos", "camino medio anual", "aproximación anual", "VR agregado",
             "periodos con VR < 1", "signo de acuerdo (camino, 1 − VR)", "acuerdo en el agregado",
             "camino log medio anual", "signo de acuerdo en log", "acuerdo en el agregado (log)",
             "Δg simulado (frente al diario, sin préstamo)", "pendiente", "t pendiente (NW)", "peor periodo sin reajuste"]
    T_("paths", "Por periodo, el corto que no se reajusta frente al reajustado cada día (sin costes ni intereses): media anual "
       "del camino exacto (diferencia de riqueza al final del periodo) y de su aproximación s(1 + s)/2·(Σy² − (Σy)²), con y "
       "el exceso diario del fondo sobre la letra; VR agregado = media de (Σy)² / media de Σy²; acuerdo de signo entre el "
       "camino y 1 − VR, por periodo y en el agregado; lo mismo con la diferencia de logaritmos (crecimiento); Δg simulado: "
       "el corto semanal o mensual menos el diario en su familia, sin préstamo, con costes e intereses; pendiente del camino "
       "sobre la aproximación (Newey-West de 12 retardos)", paths[order],
       pct={"camino medio anual", "aproximación anual", "periodos con VR < 1", "signo de acuerdo (camino, 1 − VR)",
            "camino log medio anual", "signo de acuerdo en log", "Δg simulado (frente al diario, sin préstamo)",
            "peor periodo sin reajuste"}, digits=3)
    fig_paths(C["periods"], rep)
    T_("vr", "Cociente de varianzas VR(h) del índice corto (ER) con ventanas solapadas y contraste de z = Σx² − (Σx)² frente "
       "a 0 (Hansen–Hodrick, h − 1 retardos). z > 0 en media: un corto reajustado cada h sesiones gana, antes de costes, al "
       "diario", vr, digits=3, pct={"z medio anual"})
    n_log_bad = int((paths["acuerdo en el agregado (log)"] == "no").sum())
    rep.p(f"A segundo orden, «el corto sin reajuste diario gana si y solo si VR < 1» es una identidad (derivación en "
          f"r27_vix_carry.path_test). En riqueza al final de cada periodo se cumple: el signo coincide con el de 1 − VR "
          f"entre el {pct_es(paths['signo de acuerdo (camino, 1 − VR)'].min(), 0)} y el "
          f"{pct_es(paths['signo de acuerdo (camino, 1 − VR)'].max(), 0)} de los periodos y en el agregado de "
          f"{ag_w} de {n_cases} casos. En crecimiento no siempre: el corto que no se reajusta pierde mucho más en los "
          f"periodos malos (en el peor, un {pct_es(worst_path, 0)} del capital del inicio del periodo"
          + (", más que todo el capital" if worst_path > 1 else "") + "), y la media de los logaritmos contradice el "
          f"signo de 1 − VR en {n_log_bad} de {n_cases} casos"
          + (f", {'todos' if len(log_no_fq) == 1 else 'entre ellos'} con reajuste {' y '.join(sorted(log_no_fq))}" if n_log_bad else "")
          + ". El Δg simulado se aparta además del camino por los costes, los intereses y el mantenimiento: cuando se "
          "incumple, el corto se recompra en el pico y no recupera el rebote del mes. El VR del "
          "índice es menor que 1 en casi todas las ventanas y horizontes (el futuro revierte algo), pero la reversión no es "
          f"significativa a 21 sesiones (t de HH entre {num_es(vr.xs(21, level='h (sesiones)')['t (HH)'].min(), 2)} y "
          f"{num_es(vr.xs(21, level='h (sesiones)')['t (HH)'].max(), 2)}) y a las 16:00 antes de 2018 fue mayor que 1. Que un "
          "corto semanal gane al diario en una ventana no es una propiedad del producto sino de esa ventana.")

    rep.h("2.4 Préstamo y producto del corto (H3 b)", 3)
    be = pd.concat({"R1 (−0,5x, SVXY, desde 2018)": breakeven(res["R1"]), "R3 (−1x, SVIX, desde 2022)": breakeven(res["R3"]),
                    "R4 (−1x, SVXY, 2011–2018)": breakeven(res["R4"]),
                    "R4 sin el 5–6-2-2018": breakeven(res["R4s"])}, names=["familia"])
    T_("be", "Préstamo anual con el que el corto rinde lo mismo que el inverso comprado (interpolación lineal de Δg en la "
       "rejilla 0 / 2 / 5,2 / 18%)", be)
    rep.p("H3 (b) con préstamo del 5,2% y el producto sin remunerar, en cada familia de productos reales y con todos sus "
          "cortos (se falsa si alguno tiene Δg > 0 con p de Romano–Wolf < 0,05 en su familia):\n" + "\n".join(lb_lines))
    rep.p(f"Los pares del pre-registro, uno a uno: UVXY frente a SVXY desde el 28-2-2018, {vb['R1'][0]} ({vb['R1'][1]}); "
          f"UVIX frente a SVIX desde el 30-3-2022, {vb['R3'][0]} ({vb['R3'][1]}); UVXY frente a SVXY (−1x) en 2011–2018, "
          f"{vb['R4'][0]} y, sin el 5–6-2-2018, {vb['R4s'][0]}; UVXY frente a SVXY en 2011–2026, {vb['R5'][0]} y, sin el "
          f"5–6-2-2018, {vb['R5s'][0]} ({vb['R5s'][1]}). El préstamo de la rejilla es constante; en IBKR sube con el VIX "
          "(R26), justo cuando el corto más lo necesita, hay días casi sin títulos disponibles y el bróker puede "
          "recomprarlos. Nada de eso está en la simulación.")

    rep.h("2.5 Sintéticos: 2005–2026, antes y después de 2018", 3)
    T_("syn", "Sintéticos (sin fricción de fondo): corto del +1,5x (frente al −0,5x), del +2x (frente al −1x) o del +1x "
       "(frente a VYLD, peso F/100) frente al inverso sintético comprado, por ventana; Δg en fracción anual. Cada ventana "
       "se simula desde cero", syn05,
       pct={"CAGR del inverso"} | {f"Δg {FREQ_ES[fq]} sin préstamo" for fq in FREQS}
       | {f"Δg {FREQ_ES[fq]} 5,2% sin rem." for fq in FREQS}, digits=3)
    s05 = syn05.loc[("−0,5x sintético (liquidación)", "principal")]
    nw = int((syn05["Δg semanal sin préstamo"] > 0).sum())
    nm = int((syn05["Δg mensual sin préstamo"] > 0).sum())
    rep.p(f"Sin fricción de fondo, el corto diario sin préstamo frente al inverso solo gana los gastos (frente al −0,5x con "
          f"la liquidación: {pp(s05['Δg diario sin préstamo'])} pp al año en la ventana principal) y con el 5,2% sin "
          f"remunerar pierde ({pp(s05['Δg diario 5,2% sin rem.'])} pp). El reajuste semanal o mensual añade el camino de la "
          f"sección 2.3: sin préstamo, el semanal supera al inverso en {nw} de {len(syn05)} filas y el mensual en {nm}. "
          "La ventaja de los productos reales que viene de la fricción (sección 2.2) no aparece aquí, porque los sintéticos "
          "no tienen fricción.")

    rep.h("2.6 SPA y Romano–Wolf por familia", 3)
    cols = ["título", "ventana", "reglas", "CAGR del inverso", "maxDD del inverso", "Δg > 0",
            "p_RW < 0,05 (a favor del corto)", "con ruina", "mejor regla", "Δg de la mejor", "p_SPA"]
    T_("spa", "Por familia (todas las ventanas; R2v y R3v, con VXX, son sensibilidades): reglas con Δg > 0, con p de "
       "Romano–Wolf < 0,05 a favor del corto, con ruina dentro de la ventana, la mejor regla y el SPA de Hansen (H0: "
       "ningún corto supera al inverso comprado; 2.000 remuestreos, bloque 63)", summ[cols],
       pct={"CAGR del inverso", "maxDD del inverso", "Δg de la mejor"}, digits=3)

    rep.h("2.7 Veredicto de H3 para los productos del VIX", 3)
    verdict_lines = [
        "- (a) «El corto de UVXY (y de UVIX) supera a comprar SVXY (SVIX) antes de préstamo», en cada familia de "
        "productos reales (sin préstamo, producto a la letra − 0,5%):",
        *la_lines,
        "- (a) «…solo por el tamaño del corto y la frecuencia de reajuste» (lectura común con R29: la fricción de los "
        "fondos no es ni tamaño ni frecuencia): "
        + (f"no se sostiene: la fricción de los fondos (real − modelos) es distinta de 0 con IC90 en {' y en '.join(a2_rej)} "
           f"(sección 2.2)." if a2_rej else "se sostiene: la fricción de los fondos no es distinta de 0 (sección 2.2)."),
        f"- (a) «Un corto sin reajuste diario gana frente al diario si y solo si VR < 1»: en riqueza por periodo, se cumple "
        f"(acuerdo de signo en el {pct_es(paths['signo de acuerdo (camino, 1 − VR)'].min(), 0)}–"
        f"{pct_es(paths['signo de acuerdo (camino, 1 − VR)'].max(), 0)} de los periodos; {ag_w} de {n_cases} agregados). En "
        f"crecimiento logarítmico, {ag_l} de {n_cases} agregados: con reajuste mensual el corto que deriva puede perder "
        "aunque VR < 1.",
        "- (b) «Con el préstamo mediano y sin remunerar el producto del corto, Δg ≤ 0», con todos los cortos de cada "
        f"familia: {b_text} (sección 2.4). Con VXX (R2v, R3v): "
        + "; ".join(f"{k} {vb_state[k]}" for k in ("R2v", "R3v") if k in vb_state) + ".",
        "- La parte de SQQQ frente a QQQ/TQQQ de H3 está en R29 (r29_short_inverse).",
    ]
    rep.p("\n".join(verdict_lines))

    # ── 3. episodios
    rep.h("3. Episodios")
    T_("ep", "Resultado en cada episodio (del cierre del primer día al del último; ruina absorbente). Los cortos se toman de "
       "su simulación completa (ventana principal los sintéticos; 2011–2026 los de UVXY; desde 2022 los de UVIX), con "
       "préstamo 5,2% y producto sin remunerar; vacío si el producto no existía", ep_res, pct=set(ep_res.columns))
    T_("epw", "Peor día dentro de cada episodio", ep_worst, pct=set(ep_worst.columns))
    rep.p(f"En marzo de 2020 todas las formas de cobrar la prima perdieron lo mismo a igual exposición (−0,5x sintético "
          f"{pct_es(ep.loc['−0,5x sintético (liquidación)', e20], 0)}, SVXY {pct_es(ep.loc['SVXY real', e20], 0)}, corto "
          f"diario de UVXY {pct_es(ep.loc['corto UVXY real diario (igual a SVXY; 5,2%, sin rem.)', e20], 0)}): no hay "
          "vehículo que evite una subida sostenida de la volatilidad. El reajuste mensual lo empeora "
          f"({pct_es(ep.loc['corto UVXY real mensual (igual a SVXY; 5,2%, sin rem.)', e20], 0)}): el corto crece mientras "
          f"el ETP sube. En febrero de 2018 decide la hora: −1x con la liquidación {pct_es(ep.loc['−1x sintético (liquidación)', e18], 0)}, "
          f"a las 16:00 {pct_es(ep.loc['−1x sintético (16:00)', e18], 0)}. En 2008 el −0,5x perdió un "
          f"{pct_es(-ep.loc['−0,5x sintético (liquidación)', e08], 0)} y el SPY un {pct_es(-ep.loc['SPY', e08], 0)}"
          + (f"; la {LRS} ganó un {pct_es(ep.loc[LRS, e08], 1)}: estuvo en letras." if LRS in ep.index else "."))

    # ── 4. Kelly
    rep.h("4. Tamaño y Kelly con colas gordas")
    T_("kelly", f"Exposición corta al índice k (k·(−1x) + (1 − k)·letra, reajuste diario; los reales, por unidad de "
       f"exposición: SVXY ÷ 0,5). {kelly.LSTAR_APROX}: μ/σ² del exceso sobre la letra; {kelly.LSTAR_EXACTO}: el k de "
       f"[0, {num_es(K_MAX, 0)}] que maximiza el crecimiento logarítmico realizado (con el gasto del producto; pérdidas "
       f"acotadas en −99,99%), IC90 por bootstrap estacionario de bloque {BLOCK_KELLY} y {B_KELLY} remuestreos; L* con "
       "cola GPD: el mismo máximo con la cola de pérdidas por encima del percentil 95 sustituida por la GPD ajustada. "
       "Dentro de muestra", kt,
       pct={"μ exceso anual", "σ anual", "peor día (k = 1)", "g en L* exacto", "g con k = 0,5", "g con k = 1"}, digits=2)
    T_("size", "Por tamaño, ventana principal: CAGR, crecimiento y caída máxima con la historia, y años entre días de −50% o "
       "de ruina (pérdida del 100%) según la cola GPD de las pérdidas del −1x (umbral del percentil 95: con otro umbral "
       f"cambian mucho, tabla {N['sens']}), con las dos convenciones y con y sin el 5–6-2-2018; referencias en la misma "
       "ventana", size, pct={"CAGR", "g", "maxDD", "peor día"}, digits=4)
    fig_kelly(curves, kt, rep)
    rep.p(f"Con la liquidación de las 16:15, el 5-2-2018 hace que cualquier exposición por encima de "
          f"{num_es(k_set['k que arruina el peor día'], 2)} pierda todo en un día; sin ese día, la exposición que el peor "
          f"día deja a cero sube a {num_es(k_sin['k que arruina el peor día'], 2)}, y a las 16:00 es "
          f"{num_es(k_16['k que arruina el peor día'], 2)}. El L* exacto pasa de {fmt_k(k_set[kelly.LSTAR_EXACTO])} "
          f"(IC90 {k_set['IC90 L* exacto']}) a {fmt_k(k_sin[kelly.LSTAR_EXACTO])} quitando esos dos días: una estimación "
          f"que depende de un día es una medida de la suerte de la muestra. Con la cola GPD, el L* es "
          f"{fmt_k(k_set['L* con cola GPD'])} con la liquidación y {fmt_k(k_16['L* con cola GPD'])} a las 16:00, frente a "
          f"{fmt_k(k_set[kelly.LSTAR_EXACTO])} y {fmt_k(k_16[kelly.LSTAR_EXACTO])} con la historia; sin el 5–6-2-2018, "
          f"{fmt_k(k_sin['L* con cola GPD'])} y {fmt_k(k_sin16['L* con cola GPD'])} (con la historia, "
          f"{fmt_k(k_sin[kelly.LSTAR_EXACTO])} y {fmt_k(k_sin16[kelly.LSTAR_EXACTO])}). Con k = 1 y la liquidación, la cola "
          f"GPD da una ruina en un día cada {fmt_years(size.loc[('−1x sintético (liquidación)', 1.0), 'años entre ruinas en un día (GPD)'])} "
          f"años con el 5–6-2-2018 y cada "
          f"{fmt_years(size.loc[('−1x sintético (liquidación), sin el 5–6-2-2018', 1.0), 'años entre ruinas en un día (GPD)'])} "
          f"sin él (tabla {N['size']}). Con una ruina de verdad "
          "(log 0 = −∞) y una cola sin techo (ξ > 0), cualquier k > 0 tiene probabilidad positiva de ruina y crecimiento "
          "esperado −∞: el L* con cola GPD depende del recorte del laboratorio en −99,99% y es un techo, no un objetivo."
          + (f" En la misma ventana la {LRS} del artículo 2 creció {pct_es(lrs_cagr)} al año con una caída máxima de "
             f"{pct_es(size.loc[lrs_rows[0], 'maxDD'])}; el −0,5x (k = 0,5) con la liquidación, "
             f"{pct_es(size.loc[('−1x sintético (liquidación)', 0.5), 'CAGR'])} con "
             f"{pct_es(size.loc[('−1x sintético (liquidación)', 0.5), 'maxDD'])}." if lrs_rows else ""))

    rep.h("Límites y datos sin verificar")
    rep.p("\n".join([
        "- Muestra: 20 años de futuros y pocos episodios extremos (2008, 2015, 2018, 2020, 2024); las conclusiones sobre el "
        "tamaño y sobre la ventaja del corto antes de 2018 dependen de uno o dos días.",
        "- Productos que cambiaron: SVXY (−1x → −0,5x) y UVXY (2x → 1,5x) el 27-2-2018; SVIX y UVIX desde el 30-3-2022; VYLD "
        "desde marzo de 2025 (1,5 años). Los precios son cierres de mercado de Yahoo, no valores liquidativos.",
        "- Préstamo: rejilla constante (0 / 2 / 5,2 / 18%) tomada de 12 meses de un bróker (IBKR vía iBorrowDesk); sin "
        "recompras forzosas ni días sin títulos. Que IBKR remunere el producto del corto a la letra − 0,5% solo en cuentas "
        "grandes está sin verificar (tabla por tramos).",
        "- Margen: mantenimiento FINRA 30% × |L| al cierre; un bróker puede exigir más y liquidar dentro de la sesión.",
        "- La fricción de los fondos (sección 2.2) se mide frente a los modelos de R26 y su composición está sin verificar; "
        "antes de octubre de 2020 incluye el desfase entre la liquidación de las 16:15 y el cierre de las 16:00.",
        "- La liquidación final de cada contrato es el precio del día del vencimiento en los ficheros de CFE (sin verificar "
        "que sea exactamente la liquidación especial de apertura).",
        "- Los sintéticos no simulan la liquidación anticipada de un ETN (XIV) ni la reducción del apalancamiento de "
        "ProShares; el −1x con la liquidación sigue vivo tras el 5-2-2018 con un 4% del capital.",
        "- Acceso minorista en la UE: que un minorista no pueda comprar ETPs de EE. UU. sin KID (PRIIPs) viene de un "
        "extracto de IBKR y está sin verificar aquí (R29 lo marca igual); las cifras no son recomendaciones.",
        "- Producto del corto con tipos cero: que IBKR cobre al corto cuando la letra − 0,5% es negativa está sin "
        "verificar; afecta a (a) en 2011–2015 y 2020–2021 (con «sin remunerar» no hay ese cargo).",
        f"- VXX: en 2022 el cierre de Yahoo llegó a {num_es(vxx['peak'], 2)} veces su modelo (tabla {N['vxx']}); R2 y R3 no "
        "llevan VXX por eso. Las fechas de la suspensión y de la reanudación de la emisión son las de R26 (notas de "
        "prensa, sin verificar aquí).",
        "- El cambio de apalancamiento de 2018 entra en el corto al cierre del 26-2-2018 (orden al cierre siguiente); "
        "que ProShares lo hubiera anunciado antes de ese cierre está sin verificar (efecto de un día).",
        "- La parte de SQQQ frente a QQQ/TQQQ de H3 se contrasta en R29 (r29_short_inverse).",
    ]))
    rep.h("Referencias")
    rep.p("\n".join([
        "- Avellaneda, M. y Zhang, S. (2010). Path-Dependence of Leveraged ETF Returns. SIAM Journal on Financial "
        "Mathematics 1(1), 586–603. doi:10.1137/090760805",
        "- Bollerslev, T., Tauchen, G. y Zhou, H. (2009). Expected Stock Returns and Variance Risk Premia. Review of "
        "Financial Studies 22(11), 4463–4492. doi:10.1093/rfs/hhp008",
        "- Carr, P. y Wu, L. (2009). Variance Risk Premiums. Review of Financial Studies 22(3), 1311–1341. "
        "doi:10.1093/rfs/hhn038",
        "- Eraker, B. y Wu, Y. (2017). Explaining the negative returns to volatility claims: An equilibrium approach. "
        "Journal of Financial Economics 125(1), 72–98. doi:10.1016/j.jfineco.2017.04.007",
        "- Whaley, R. E. (2013). Trading Volatility: At What Cost? Journal of Portfolio Management 40(1), 95–108. "
        "doi:10.3905/jpm.2013.40.1.095",
        "- FINRA, Regulatory Notice 09-53 (mantenimiento de los ETFs apalancados e inversos).",
        "Los DOI se comprobaron en Crossref durante la exploración previa del artículo 4.",
    ]))
    rep.h("Interpretaciones del pre-registro")
    rep.p("El pre-registro no fija el estadístico de H3. Esta línea lo concreta como se describe en la cabecera (a-i, a-ii, "
          "a-iii y b); la concreción se escribió al programar la línea, con una primera ejecución de prueba de los cortos "
          "reales ya vista, y se declara aquí. Se añadió, también a la vista de esa prueba, la versión en logaritmos de "
          "(a-iii) y la fricción desde el 7-10-2020. Las ventanas de los episodios son las del informe exploratorio.")
    rep.p("Cambios tras la revisión del laboratorio (con todos los resultados ya vistos): (1) «tamaño y frecuencia» se "
          "lee igual aquí y en R29 (la fricción de los fondos no es ni lo uno ni lo otro); (2) el veredicto escrito de H3 "
          "(b) cubre todas las familias de productos reales y todos sus cortos, también R2 y R5s (R5 sin el 5–6-2-2018), "
          "que antes solo estaban en las tablas; (3) R2 y R3 dejan fuera los cortos de VXX por la prima de 2022 (R2v y R3v, con "
          "VXX, quedan como sensibilidad); (4) el SPA se calcula en todas las familias, también en las sintéticas de antes "
          "y después de 2018; (5) las colas, el L* con cola GPD y los tamaños del −1x se dan también sin el 5–6-2-2018; y "
          "(6) el vencimiento de febrero de 2008 sigue la regla de festivos de CFE (R26). Ninguna otra desviación.")

    # ── resumen (se escribe al final para citar las tablas por su número)
    fam_sig = int(real_fams["p_RW < 0,05 (a favor del corto)"].sum())
    summary = [
        f"- **La pérdida de los ETPs largos es el roll del futuro, no el VIX.** En la ventana principal el índice corto ER "
        f"crece {num_es(100 * tp['g índice corto'], 1)} puntos log al año: {num_es(100 * tp['roll corto'], 1)} de roll y "
        f"{pp(tp['nivel corto'])} de cambio de nivel del futuro a 30 días. El primer futuro cotiza por encima del VIX el "
        f"{pct_es(tp['F1 > VIX'], 0)} de las sesiones y el segundo por encima del primero el {pct_es(tp['F2 > F1'], 0)} "
        f"(tabla {N['term']}).",
        f"- **Hay prima, pequeña y asimétrica.** Veintiuna sesiones antes de vencer, el futuro supera a su liquidación final "
        f"en {num_es(pr['prima media (pts)'], 2)} puntos de media (t = {num_es(pr['t (NW 3)'], 2)}; en el "
        f"{pct_es(pr['F > liquidación'], 0)} de {int(pr['contratos'])} contratos), pero el peor contrato subió un "
        f"{pct_es(pc.loc[worst_c, 'rend. largo'], 0)} en ese mes (tabla {N['prem']}). La prima de varianza del S&P 500 va en "
        f"el mismo sentido: el VIX supera a la volatilidad realizada de las 21 sesiones siguientes en "
        f"{num_es(vp['VIX − RV (pts)'], 2)} puntos de media desde 1990 (tabla {N['vrp']}).",
        f"- **El reajuste diario cuesta lo mismo a UVXY que a SVXY.** El arrastre ½(L² − L)·Σx² tiene el mismo coeficiente "
        f"con L = 1,5 y L = −0,5 (0,375), y con L = 2 y L = −1 (1): desde el 28-2-2018 resta "
        f"{num_es(-100 * d_uv['arrastre ½(L²−L)Σx²'], 1)} puntos log al año a UVXY y a SVXY. En el −1x con la liquidación, "
        f"los términos de orden superior restan {num_es(-100 * d1['resto (orden ≥ 3 y seguimiento)'], 1)} puntos más, casi "
        f"todo el 5-2-2018 (tabla {N['drag']}).",
        f"- **Colas.** Las subidas diarias del índice corto tienen ξ = {num_es(st_row['ξ (GPD)'], 2)} (IC90 {st_row['IC90 ξ']}) "
        f"con la liquidación y {num_es(h16_row['ξ (GPD)'], 2)} a las 16:00; las caídas del SPY, {num_es(spy_row['ξ (GPD)'], 2)}. "
        f"Un día de +100% (el que deja a cero un −1x) sale entre una vez cada {fmt_years(s_set['años entre días > 100%'].min())} "
        f"y una vez cada {fmt_years(s_set['años entre días > 100%'].max())} años según el umbral: una extrapolación con un solo día "
        f"observado cerca (+{pct_es(st_row['máximo'], 0)} el {dmy(st_row['fecha del máximo'])}). Sin el 5–6-2-2018, "
        f"ξ = {num_es(sin_row['ξ (GPD)'], 2)} y ese día sale entre una vez cada {fmt_years(s_sin['años entre días > 100%'].min())} "
        f"y una vez cada {fmt_years(s_sin['años entre días > 100%'].max())} años (tablas {N['tails']}–{N['sens']}).",
        f"- **H3 (a), vender antes de préstamo.** Desde el 28-2-2018, el corto de UVXY de igual exposición que SVXY (peso "
        f"1/3), sin préstamo, frente a comprar SVXY: {a1['R1'][1]}; {a1['R1'][0]}. Desde el 31-3-2022, UVIX frente a SVIX: "
        f"{a1['R3'][1]}; {a1['R3'][0]} (tablas {N['R1']}–{N['R3']}). En 2011–2018 (−1x) la ventaja la decide el 5–6-2-2018: SVXY se "
        f"reajustaba en el pico de las 16:15 y el corto se reajusta al cierre de las 16:00 (tabla {N['R4']}).",
        f"- **…y no solo por tamaño y frecuencia** (lectura común con R29: tamaño y frecuencia son la mecánica con fondos "
        f"ideales; la fricción de los fondos no es ninguna de las dos). Con reajuste diario la mecánica (gastos y producto "
        f"del corto) da "
        f"{pp(fr_uvxy['Δg modelos'])} pp en UVXY/SVXY; el resto, {pp(fr_uvxy['fricción (real − modelos)'])} pp (IC90 "
        f"{fr_uvxy['IC90 fricción']}), es fricción de los fondos: el comprador del inverso paga lo que su fondo rinde por "
        f"debajo del índice y el vendedor del largo cobra lo que el suyo rinde por debajo. En UVIX/SVIX, "
        f"{pp(fr_uvix['fricción (real − modelos)'])} pp (IC90 {fr_uvix['IC90 fricción']}). Su composición está sin "
        f"verificar (tabla {N['dec']}).",
        f"- **«Sin reajuste diario gana si y solo si VR < 1».** En riqueza por periodo se cumple ("
        f"{pct_es(paths['signo de acuerdo (camino, 1 − VR)'].min(), 0)}–{pct_es(paths['signo de acuerdo (camino, 1 − VR)'].max(), 0)} "
        f"de las semanas o meses; {ag_w} de {n_cases} agregados); en crecimiento, solo {ag_l} de {n_cases}: con reajuste "
        f"mensual el corto que deriva pierde en los meses malos lo que gana en los demás (tabla {N['paths']}). El VR del "
        f"índice a 21 sesiones no es distinto de 1 con significación (tabla {N['vr']}).",
        f"- **H3 (b), préstamo del 5,2% y producto sin remunerar**, con todos los cortos de cada familia real: {b_text}. "
        f"UVXY/SVXY desde 2018: {vb['R1'][0]}: {vb['R1'][1]}. UVIX/SVIX desde 2022: {vb['R3'][0]}. Sin el 5–6-2-2018, "
        f"2011–2026: {verdict_b_family(res['R5s'])[1]}. 2011–2018: {vb['R4'][0]} (tablas {N['R1']}–{N['be']}).",
        f"- **Familias completas.** {num_es(n_rules, 0)} configuraciones en {n_fams} familias (sin contar R2v y R3v, las "
        f"sensibilidades con VXX). En las de productos reales, {fam_sig} de {int(real_fams.reglas.sum())} cortos superan al "
        f"inverso con p de Romano–Wolf < 0,05 y el SPA rechaza «ningún corto supera al inverso» (p < 0,05) en "
        f"{int((real_fams.p_SPA < 0.05).sum())} de {len(real_fams)} familias. En los sintéticos, sin fricción de fondo, el "
        f"corto diario sin préstamo frente al −0,5x gana {pp(s05['Δg diario sin préstamo'])} pp y con el 5,2% sin "
        f"remunerar {pp(s05['Δg diario 5,2% sin rem.'])}; el SPA rechaza en {syn_spa} de {len(syn_main)} familias de la "
        f"ventana principal y en {syn_spa_o} de {len(syn_other)} de las de antes y después de 2018, y la mejor regla es un "
        f"corto semanal en {int(syn_main['mejor regla'].str.contains('semanal').sum())} de las {len(syn_main)} principales "
        f"(tablas {N['syn']}–{N['spa']}). R2 y R3 no llevan VXX: un corto de VXX abierto el {dmy(VS_START)} cobra además "
        f"la desaparición de la prima con que cotizaba en 2022 (unos {num_es(100 * vxx['gift'], 1)} pp al año; tabla "
        f"{N['vxx']}).",
        f"- **Ningún vehículo evita los episodios.** En marzo de 2020, a −0,5x, el sintético perdió "
        f"{pct_es(-ep.loc['−0,5x sintético (liquidación)', e20], 0)}, SVXY {pct_es(-ep.loc['SVXY real', e20], 0)} y el corto "
        f"diario de UVXY {pct_es(-ep.loc['corto UVXY real diario (igual a SVXY; 5,2%, sin rem.)', e20], 0)}; el mensual, "
        f"{pct_es(-ep.loc['corto UVXY real mensual (igual a SVXY; 5,2%, sin rem.)', e20], 0)} (tabla {N['ep']}).",
        f"- **Tamaño.** El L* exacto del −1x sintético es {fmt_k(k_set[kelly.LSTAR_EXACTO])} con la liquidación (IC90 "
        f"{k_set['IC90 L* exacto']}), {fmt_k(k_sin[kelly.LSTAR_EXACTO])} sin el 5–6-2-2018 y "
        f"{fmt_k(k_16[kelly.LSTAR_EXACTO])} a las 16:00; μ/σ² da {num_es(k_set[kelly.LSTAR_APROX], 2)}, "
        f"{num_es(k_sin[kelly.LSTAR_APROX], 2)} y {num_es(k_16[kelly.LSTAR_APROX], 2)}. Con la cola GPD, "
        f"{fmt_k(k_set['L* con cola GPD'])} con la liquidación y {fmt_k(k_sin['L* con cola GPD'])} sin el 5–6-2-2018. Es "
        f"dentro de muestra y lo decide un día (tablas {N['kelly']}–{N['size']})."
        + (f" En la misma ventana, la {LRS} del artículo 2 creció {pct_es(lrs_cagr)} al año y el −0,5x con la liquidación "
           f"{pct_es(size.loc[('−1x sintético (liquidación)', 0.5), 'CAGR'])}." if lrs_rows else ""),
    ]
    rep.lines[pos_summary:pos_summary] = ["## Resumen", "", "\n".join(summary), ""]
    rep.save()
    return C


if __name__ == "__main__":
    run()
