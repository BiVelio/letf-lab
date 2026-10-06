"""R9 — Óptimo de compra: ¿importa *cuándo* se entra en un 3x?

Para cada día de la historia larga se registra el estado del mercado al cierre
(caída desde máximos, volatilidad de 21 días, tendencia frente a la media de
200 sesiones y VIX) y se mide qué habría pasado comprando al cierre siguiente
y manteniendo 1, 3 y 5 años, con 1x (el índice con dividendos) y con el 3x:
el ETF real desde su lanzamiento y, antes, una réplica financiada al tipo
interbancario (fondos federales) con el sobrecoste del swap recalibrado en la
vida real del ETF (la base de R10 y R12, r10.load).

Comprobaciones:
- Placebo: la serie de estados se desplaza circularmente respecto a los
  resultados (conserva lo agrupados que están los estados en el tiempo). El
  p-valor, bilateral, es la fracción de desplazamientos con un efecto igual o
  mayor en valor absoluto. Con resultados solapados de 3 y 5 años hay pocas
  colocaciones de verdad distintas: el p no mide bien probabilidades muy
  pequeñas (sección 9).
- Fuera de muestra: réplica antes de 2010 frente a fondos reales después.
- Tres índices: S&P 500 (desde 1960), Nasdaq-100 (1985) y semiconductores (1994).
- Ejecución: la señal se lee al cierre de t y la compra se hace al cierre de
  t+1 (r10.ENTRY_LAG); comprar en el mismo cierre de la señal queda como
  sensibilidad (sección 6).
- Episodios: las caídas profundas son pocas; se listan una a una, por el
  primer día del cruce y con todos los días del estado agrupados en episodios.
- Apalancamiento de rejilla condicionado (L_rejilla): el L de una rejilla de
  réplicas con los mismos costes que maximiza el crecimiento log realizado a 3
  años según el estado de entrada.

Todos los estados usan solo información hasta el cierre del día de la señal: en
el S&P 500, antes de 1988, se calculan con el D/P de Shiller del mes anterior,
porque el del mes en curso usa el precio medio del mes, y desde 1988 con el
índice oficial con dividendos ^SP500TR, también después de 1993, cuando u es el
ETF SPY (r10.sp500_signal_returns); los resultados se miden con la rentabilidad u.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from letf_lab import data, kelly
from letf_lab.report import FIG_W, Report, comma_fig, num_es, pct_es
from letf_lab.research import r10_predictability as r10
from letf_lab.research import r12_long_ml as r12

INDICES = r10.INDICES
H = {"1a": 252, "3a": 756, "5a": 1260}
SPLIT = "2010-01-01"
LS = np.round(np.arange(0, 5.01, 0.25), 2)
LGRID = "L_rejilla"  # rótulo del L de la rejilla (no es el L* aproximado ni el exacto)
N_PLACEBO = 2000  # resolución nominal 0,0005 (por debajo del umbral de Bonferroni), pero pocas colocaciones distintas
DD_EDGES = [-1.0, -0.30, -0.20, -0.10, -0.05, 0.0001]
DD_LABELS = ["< −30%", "−20 a −30%", "−10 a −20%", "−5 a −10%", "máximos (0 a −5%)"]
VOL_EDGES = [0, 0.12, 0.18, 0.25, 0.35, 10]
VOL_LABELS = ["< 12%", "12–18%", "18–25%", "25–35%", "> 35%"]
VIX_EDGES = [0, 15, 20, 30, 200]
VIX_LABELS = ["VIX < 15", "15–20", "20–30", "> 30"]
PCT = {"mediana_1x", "mediana_3x", "p5_3x", "P(3x<0)", "P(3x>1x)", "Δ_mediana_3x",
       "IS_mediana_3x", "OOS_mediana_3x", "3x", "1x", "3x_1a", "3x_3a", "3x_5a",
       "1x_1a", "1x_3a", "1x_5a", "caída", "mediana_2x", "P(pérdida)_1x", "P(pérdida)_2x",
       "P(pérdida)_3x"}


# ───────────────────────────── estados y resultados ─────────────────────────────

def states(u: pd.Series) -> pd.DataFrame:
    px = (1 + u.fillna(0)).cumprod()
    dd = px / px.cummax() - 1
    vol = u.rolling(21).std() * np.sqrt(252)
    s = pd.DataFrame({
        "dd": dd,
        "vol": vol,
        "caída": pd.cut(dd, DD_EDGES, labels=DD_LABELS),
        "volatilidad": pd.cut(vol, VOL_EDGES, labels=VOL_LABELS),
        "tendencia": np.where(px > px.rolling(200).mean(), "sobre SMA200", "bajo SMA200"),
    }, index=u.index)
    s.loc[px.rolling(200).mean().isna(), "tendencia"] = None
    return s


def fwd_log(r: pd.Series, h: int) -> pd.Series:
    """Retorno log de comprar al cierre de t y vender al cierre de t+h."""
    lw = np.log1p(r.fillna(0).clip(lower=-0.999)).cumsum()
    return lw.shift(-h) - lw


def fwd_entry(r: pd.Series, h: int, lag: int = r10.ENTRY_LAG) -> pd.Series:
    """Retorno log de h sesiones comprando al cierre de t+lag lo que señala el
    cierre de t (con lag = 1, orden al cierre siguiente): en t, la suma de los
    retornos de t+lag+1 a t+lag+h."""
    return fwd_log(r, h).shift(-lag)


def ann(x, h: int):
    return np.expm1(np.asarray(x) * 252 / h)


def outcomes(t: str) -> tuple[pd.Series, pd.DataFrame, pd.Series, dict[int, pd.DataFrame]]:
    """u, estados, retorno 3x y, por horizonte, retornos log futuros de la rejilla
    L comprando al cierre siguiente. L = 0: letras, sin costes; L > 0, también
    L = 1: la réplica de apalancamiento L financiada a f con s' y la comisión
    (r12.letf), la misma base de costes en toda la rejilla. Los estados salen de
    u_señal (r10.load); el 1x de las tablas es el índice u (fwd_entry(u, h))."""
    d = r10.load(t)
    u, x3 = d["u"], d["x3"]
    fwd = {h: pd.DataFrame({L: fwd_entry(g, h) for L, g in grid_returns(d).items()})
           for h in H.values()}
    return u, states(d["u_señal"]), x3, fwd


def grid_returns(d: dict) -> dict[float, pd.Series]:
    """Rentabilidad diaria de cada L de la rejilla: letras en L = 0 y, para todo
    L > 0 (también L = 1), la réplica r12.letf con los costes del 3x."""
    return {L: d["rf"] if L == 0 else r12.letf(d, L) for L in LS}


# ───────────────────────────── estadística ─────────────────────────────

def placebo_p(mask: np.ndarray, y: np.ndarray, h: int, rng) -> float:
    """p-valor de «la mediana de y en el estado difiere de la incondicional».

    La máscara se desplaza circularmente (mínimo h días) para romper la relación
    temporal con y sin romper su agrupamiento."""
    ok = ~np.isnan(y)
    base = np.median(y[ok])
    obs = abs(np.median(y[mask & ok]) - base)
    n = len(y)
    shifts = rng.integers(h, n - h, N_PLACEBO)
    null = np.array([abs(np.median(y[np.roll(mask, k) & ok]) - base)
                     if (np.roll(mask, k) & ok).sum() > 20 else 0.0 for k in shifts])
    return float((null >= obs).mean())


episodes = r10.episodes  # tramos del estado separados por más de 21 días (vive en r10)


def lstar(fwd_h: pd.DataFrame, mask: pd.Series) -> float:
    """L_rejilla: el L de la rejilla que maximiza el crecimiento log medio a h
    comprando en el estado. Si el máximo cae en el tope de la rejilla, el óptimo
    está por encima: np.inf (censurado, «≥ 5»), nunca el tope como si fuera un
    óptimo (como kelly.lstar_exact)."""
    g = fwd_h[mask].dropna().mean()
    if not len(g):
        return np.nan
    L = float(g.idxmax())
    return np.inf if len(g) > 1 and L == float(max(g.index)) else L


def fmt_lgrid(x: float) -> str:
    """Texto de L_rejilla para las tablas: «≥ 5» si está censurado."""
    return "" if pd.isna(x) else kelly.fmt_lstar(x, 2, float(LS[-1]))


def lgrid_es(x: float) -> str:
    """L_rejilla en el texto: coma decimal, «≥ 5» si está censurado."""
    return f"≥ {num_es(float(LS[-1]), 0)}" if np.isinf(x) else num_es(x, 2)


def show(t: pd.DataFrame) -> pd.DataFrame:
    """Tabla de state_table para el README: L_rejilla como texto («≥ 5»)."""
    return t.assign(**{LGRID: t[LGRID].map(fmt_lgrid)}) if LGRID in t else t


PVALS: list[dict] = []  # una fila por comparación con placebo (sección 9)


def state_table(col: pd.Series, labels: list[str], x3f: pd.Series, u1f: pd.Series,
                fwd_h: pd.DataFrame, h: int, rng, name: str = "") -> pd.DataFrame:
    y3, y1 = ann(x3f, h), ann(u1f, h)
    # la referencia «todos» y el placebo solo usan días en que el estado existe
    # (el VIX empieza en 1990; la SMA200 necesita 200 días)
    valid = ~np.isnan(y3) & ~np.isnan(y1) & col.notna().values
    rows = {}
    for lab in labels:
        m = (col == lab).values & valid
        if m.sum() < 40:
            continue
        rows[lab] = {
            "días": int(m.sum()),
            "episodios": episodes(m),
            "mediana_1x": np.median(y1[m]),
            "mediana_3x": np.median(y3[m]),
            "p5_3x": np.percentile(y3[m], 5),
            "P(3x<0)": (y3[m] < 0).mean(),
            "P(3x>1x)": (y3[m] > y1[m]).mean(),
            "Δ_mediana_3x": np.median(y3[m]) - np.median(y3[valid]),
            "p_placebo": placebo_p(m, np.where(valid, y3, np.nan), h, rng),
            LGRID: lstar(fwd_h, pd.Series(m, index=col.index)),
        }
        idx = col.index[m]
        PVALS.append({"prueba": name, "estado": lab, "p_placebo": rows[lab]["p_placebo"],
                      "Δ_mediana_3x": rows[lab]["Δ_mediana_3x"], "días": int(m.sum()),
                      "episodios": rows[lab]["episodios"], "primer_día": idx[0].date(),
                      "último_día": idx[-1].date()})
    rows["todos"] = {
        "días": int(valid.sum()), "episodios": np.nan,
        "mediana_1x": np.median(y1[valid]), "mediana_3x": np.median(y3[valid]),
        "p5_3x": np.percentile(y3[valid], 5), "P(3x<0)": (y3[valid] < 0).mean(),
        "P(3x>1x)": (y3[valid] > y1[valid]).mean(), "Δ_mediana_3x": 0.0,
        "p_placebo": np.nan, LGRID: lstar(fwd_h, pd.Series(valid, index=col.index)),
    }
    return pd.DataFrame(rows).T


def any_day_table(u: pd.Series, x3: pd.Series, fwd: dict, h: int) -> dict:
    """Comprar cualquier día (al cierre siguiente) y mantener h sesiones: mediana
    anual y P(pérdida) con 1x (el índice), 2x (réplica) y 3x, en los días con los
    tres datos."""
    df = pd.DataFrame({"1x": ann(fwd_entry(u, h), h), "2x": ann(fwd[h][2.0], h),
                       "3x": ann(fwd_entry(x3, h), h)}, index=u.index).dropna()
    out = {"días": len(df), "desde": df.index[0].date(), "hasta": df.index[-1].date()}
    for c in df:
        out[f"mediana_{c}"] = df[c].median()
        out[f"P(pérdida)_{c}"] = (df[c] < 0).mean()
    return out


def is_oos(col: pd.Series, labels: list[str], x3f: pd.Series, h: int) -> pd.DataFrame:
    y3 = pd.Series(ann(x3f, h), index=x3f.index)
    out = {}
    for lab in labels:
        m = col == lab
        a = y3[m & (y3.index < SPLIT)].dropna()
        b = y3[m & (y3.index >= SPLIT)].dropna()
        out[lab] = {"IS_días": len(a), "IS_mediana_3x": a.median() if len(a) > 40 else np.nan,
                    "OOS_días": len(b), "OOS_mediana_3x": b.median() if len(b) > 40 else np.nan}
    return pd.DataFrame(out).T


def drawdown_episodes(u: pd.Series, x3: pd.Series, thr: float, lag: int = r10.ENTRY_LAG,
                      dd: pd.Series | None = None) -> pd.DataFrame:
    """Primer día de cada caída que cruza `thr` desde un máximo histórico nuevo.

    «señal»: el cierre en que la caída cruza el umbral; «compra»: el cierre en
    que se ejecuta la orden (lag sesiones después). Cuentan los retornos de
    compra+1 a compra+h; si no hay h sesiones completas, la celda queda vacía.
    dd: la caída desde máximos de la señal (states(u_señal).dd); por defecto, la de u."""
    if dd is None:
        px = (1 + u.fillna(0)).cumprod()
        dd = px / px.cummax() - 1
    rows, armed = [], True
    for d, v in dd.items():
        if v >= -0.0001:
            armed = True
        elif armed and v <= thr:
            armed = False
            i = u.index.get_loc(d)
            row = {"señal": d.date(), "compra": u.index[i + lag].date() if i + lag < len(u) else None,
                   "caída": v}
            for k, h in H.items():
                if i + h + lag < len(u):
                    w = slice(i + 1 + lag, i + h + 1 + lag)
                    row[f"1x_{k}"] = ann(np.log1p(u.iloc[w]).sum(), h)
                    row[f"3x_{k}"] = ann(np.log1p(x3.iloc[w].clip(lower=-0.999)).sum(), h)
            rows.append(row)
    return pd.DataFrame(rows).set_index("señal")


def state_episodes(col: pd.Series, lab: str, x3f: pd.Series, u1f: pd.Series, h: int) -> pd.DataFrame:
    """Todos los días del estado con resultado a h sesiones, agrupados en
    episodios como r10.episodes (hueco de más de 21 sesiones), con la fila total.
    La suma de días es la de la columna «días» de state_table."""
    y3, y1 = ann(x3f, h), ann(u1f, h)
    valid = ~np.isnan(y3) & ~np.isnan(y1) & col.notna().values
    m = (col == lab).values & valid
    rows = []
    for a, b in r10.episode_spans(m):
        mm = np.zeros(len(m), bool)
        mm[a:b + 1] = True
        mm &= m
        rows.append({"primer_día": col.index[a].date(), "último_día": col.index[b].date(),
                     "días": int(mm.sum()), "mediana_3x": np.median(y3[mm]),
                     "mediana_1x": np.median(y1[mm]), "P(3x>1x)": (y3[mm] > y1[mm]).mean()})
    out = pd.DataFrame(rows)
    if m.any():
        out.loc[len(out)] = {"primer_día": "total", "último_día": "", "días": int(m.sum()),
                             "mediana_3x": np.median(y3[m]), "mediana_1x": np.median(y1[m]),
                             "P(3x>1x)": (y3[m] > y1[m]).mean()}
    return out


# ───────────────────────────── figuras ─────────────────────────────

def fig_curve(results: dict, rep: Report):
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 11.5), sharey=True)
    edges = np.r_[-1.0, np.arange(-0.50, 0.001, 0.05)]
    edges[-1] = 0.0001
    centers = np.r_[-0.525, (edges[1:-1] + edges[2:]) / 2]  # «< −50%» se dibuja en −52,5%
    for ax, (name, (u, s, x3, fwd)) in zip(axes, results.items()):
        h = H["3a"]
        y3 = pd.Series(ann(fwd_entry(x3, h), h), index=u.index)
        y1 = pd.Series(ann(fwd_entry(u, h), h), index=u.index)
        b = pd.Series(pd.cut(s.dd, edges, labels=centers), index=u.index).astype(float)
        g3 = y3.groupby(b).quantile([0.25, 0.5, 0.75]).unstack()
        g1 = y1.groupby(b).median()
        n = y3.dropna().groupby(b).size()
        keep = n[n > 60].index
        xs = np.asarray(keep) * 100
        ax.fill_between(xs, g3.loc[keep, 0.25] * 100, g3.loc[keep, 0.75] * 100,
                        color="C3", alpha=0.18, label="3x: rango intercuartílico")
        ax.plot(xs, g3.loc[keep, 0.5] * 100, "o-", color="C3", label="3x: mediana")
        ax.plot(xs, g1.loc[keep] * 100, "s--", color="k", label="1x: mediana")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set(title=f"{r10.fig_name(name)} (desde {u.index[0].year})",
               xlabel="Caída desde máximos al comprar (%)",
               ylabel="Rentabilidad anual (%)")
        ax.invert_xaxis()
    axes[0].legend(loc="upper left")
    fig.suptitle("Rentabilidad anual a 3 años según la caída al comprar")
    comma_fig(fig)
    rep.fig(fig, "curva_caida_3a", "Rentabilidad anualizada a 3 años según la caída desde máximos al comprar")


def fig_density(results: dict, rep: Report):
    fig, axes = plt.subplots(3, 1, figsize=(11, 14))
    xs = np.linspace(-1, 1.5, 400)
    for ax, (name, (u, s, x3, fwd)) in zip(axes, results.items()):
        h = H["3a"]
        y3 = pd.Series(ann(fwd_entry(x3, h), h), index=u.index)
        y1 = pd.Series(ann(fwd_entry(u, h), h), index=u.index)
        for lab, c in [("máximos (0 a −5%)", "C0"), ("−10 a −20%", "C1"), ("< −30%", "C3")]:
            v = y3[s["caída"] == lab].dropna().clip(-0.99, 1.5)
            if len(v) > 60:
                ax.plot(xs * 100, gaussian_kde(v)(xs), color=c, lw=2,
                        label=f"3x, entrada {lab} (n = {num_es(len(v), 0)})")
        v1 = y1.dropna().clip(-0.99, 1.5)
        ax.plot(xs * 100, gaussian_kde(v1)(xs), "k--", lw=1.2, label="1x, cualquier día")
        ax.axvline(0, color="0.5", lw=0.6)
        ax.set(title=r10.fig_name(name), xlabel="Rentabilidad anual siguientes 3 años (%)", yticks=[],
               ylabel="Densidad de probabilidad")
        ax.legend(fontsize=11)
    fig.suptitle("Distribución de resultados a 3 años del 3x según el momento de entrada")
    comma_fig(fig)
    rep.fig(fig, "densidad_3a", "Densidad de la rentabilidad a 3 años del 3x según la caída al comprar")


def fig_leverage_density(results: dict, rep: Report):
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 11.5))
    xs = np.linspace(-1, 1.5, 400)
    h = H["3a"]
    for ax, (name, (u, s, x3, fwd)) in zip(axes, results.items()):
        series = {"1x": fwd_entry(u, h), "2x": fwd[h][2.0], "3x": fwd_entry(x3, h)}
        for (lab, f), c in zip(series.items(), ["k", "C0", "C3"]):
            v = pd.Series(ann(f, h), index=f.index).dropna().clip(-0.99, 1.5)
            ax.plot(xs * 100, gaussian_kde(v)(xs), color=c, lw=2,
                    label=f"{lab}: mediana {pct_es(v.median(), 0)}, P(pérdida) {pct_es((v < 0).mean(), 0)}")
        ax.axvline(0, color="0.5", lw=0.6)
        ax.set(title=f"{r10.fig_name(name)} (desde {u.index[0].year})",
               xlabel="Rentabilidad anual de los 3 años siguientes (%)", yticks=[],
               ylabel="Densidad")
        ax.legend(loc="upper right")
    fig.suptitle("Comprar cualquier día y mantener 3 años: 1x, 2x y 3x")
    comma_fig(fig)
    rep.fig(fig, "densidad_apalancamiento", "Densidad de la rentabilidad a 3 años con 1x, 2x y 3x comprando cualquier día")


def fig_heatmap(results: dict, rep: Report):
    fig, axes = plt.subplots(3, 1, figsize=(11, 15))
    for ax, (name, (u, s, x3, fwd)) in zip(axes, results.items()):
        h = H["3a"]
        y3 = pd.Series(ann(fwd_entry(x3, h), h), index=u.index)
        df = pd.DataFrame({"y": y3, "c": s["caída"], "v": s["volatilidad"]}).dropna()
        med = df.pivot_table(index="c", columns="v", values="y", aggfunc="median", observed=False)
        n = df.pivot_table(index="c", columns="v", values="y", aggfunc="size", observed=False)
        med = med.where(n > 60)
        im = ax.imshow(med.values * 100, cmap="RdYlGn", vmin=-40, vmax=60, aspect="auto")
        for i in range(med.shape[0]):
            for j in range(med.shape[1]):
                v = med.values[i, j]
                ax.text(j, i, "—" if np.isnan(v) else pct_es(v, 0), ha="center", va="center",
                        fontsize=11)
        ax.set_xticks(range(len(med.columns)), med.columns, fontsize=11)
        ax.set_yticks(range(len(med.index)), med.index, fontsize=11)
        ax.set(title=r10.fig_name(name), xlabel="Volatilidad 21 días al comprar")
    fig.colorbar(im, ax=axes[-1], label="Mediana anual 3 años del 3x (%)")
    fig.suptitle("3x: mediana a 3 años por caída desde máximos × volatilidad al comprar (— = menos de 60 días)")
    rep.fig(fig, "mapa_caida_vol", "Mediana a 3 años del 3x según caída y volatilidad al entrar")


def fig_lstar(tables: dict, rep: Report):
    fig, ax = plt.subplots(figsize=(11, 6))
    w = 0.26
    top = float(LS[-1])
    censored = False
    for k, (name, t) in enumerate(tables.items()):
        t = t.drop(index="todos")
        v = t[LGRID].astype(float)
        bars = ax.bar(np.arange(len(t)) + (k - 1) * w, v.clip(upper=top), w, label=r10.fig_name(name))
        censored |= bool(np.isinf(v).any())
        for b, c in zip(bars, np.isinf(v)):
            if c:  # censurado: el máximo está en el tope de la rejilla
                b.set_hatch("//")
                ax.text(b.get_x() + b.get_width() / 2, top, f"≥ {top:g}", ha="center", va="bottom",
                        fontsize=10)
    ax.set_xticks(range(len(t)), t.index)
    ax.axhline(3, color="r", ls="--", lw=1, label="3x")
    ax.set(ylabel="L de la rejilla (máx. crecimiento log a 3 años)",
           xlabel="Caída desde máximos al comprar", ylim=(0, top + 0.6),
           title="Apalancamiento de la rejilla que más crece a 3 años según la caída al entrar\n"
                 "(réplicas con costes" + (f"; rayado: tope de la rejilla, ≥ {top:g})" if censored else ")"))
    ax.legend(fontsize=11)
    comma_fig(fig)
    rep.fig(fig, "lstar_por_caida", "Apalancamiento de la rejilla que maximiza el crecimiento a 3 años "
                                    "según la caída al entrar")


# ───────────────────────────── informe ─────────────────────────────

def run():
    rep = r10.Report1("r9_entry", "R9 — Óptimo de compra: ¿importa cuándo se entra en un 3x?")
    PVALS.clear()
    rep.p(__doc__.split("\n", 2)[2])
    rng = np.random.default_rng(9)
    results = {name: outcomes(t) for name, t in INDICES}
    vix = data.panel(["^VIX"], "close")["^VIX"]

    rep.h("0. Punto de partida: comprar cualquier día")
    rep.table(pd.DataFrame({name: any_day_table(u, x3, fwd, H["3a"])
                            for name, (u, s, x3, fwd) in results.items()}).T, pct=PCT)
    rep.p("Mediana de la rentabilidad anual a 3 años y fracción de días con pérdida comprando "
          "al cierre siguiente de cualquier día, con el índice (1x), la réplica del 2x financiada "
          "a f y el 3x.")
    fig_leverage_density(results, rep)

    rep.h("1. Caída desde máximos al comprar")
    dd_tables = {}
    for name, (u, s, x3, fwd) in results.items():
        for k, h in H.items():
            t = state_table(s["caída"], DD_LABELS, fwd_entry(x3, h), fwd_entry(u, h), fwd[h], h, rng,
                            name=f"{name} · caída · {k}")
            if k == "3a":
                dd_tables[name] = t
            rep.h(f"{name} · horizonte {k}", 3)
            rep.table(show(t), pct=PCT)
    rep.p("mediana_1x y mediana_3x: rentabilidad anual comprando al cierre siguiente a la señal "
          "(1x: el índice). L_rejilla: el apalancamiento de la rejilla (0 a 5 en pasos de 0,25) "
          "que maximiza el crecimiento log medio realizado en el horizonte comprando en ese "
          "estado. 0 son las letras, sin costes; todo L > 0, también el 1, es una réplica de "
          "apalancamiento L con los costes del 3x (r12.letf: financiada a f, con s' y la "
          "comisión), para que toda la rejilla tenga la misma base de costes. «≥ 5»: el máximo "
          "cae en el tope de la rejilla, así que el óptimo está por encima (censurado). No es el "
          f"{kelly.LSTAR_APROX} de R10, (μ − f − s)/σ², ni el {kelly.LSTAR_EXACTO} de R12–R20.")
    fig_curve(results, rep)
    fig_density(results, rep)
    fig_lstar(dd_tables, rep)

    rep.h("2. Volatilidad de 21 días al comprar (horizonte 3 años)")
    for name, (u, s, x3, fwd) in results.items():
        h = H["3a"]
        rep.h(name, 3)
        rep.table(show(state_table(s["volatilidad"], VOL_LABELS, fwd_entry(x3, h), fwd_entry(u, h),
                                   fwd[h], h, rng, name=f"{name} · volatilidad · 3a")), pct=PCT)
    fig_heatmap(results, rep)

    rep.h("3. Tendencia: sobre o bajo la media de 200 días (horizonte 3 años)")
    for name, (u, s, x3, fwd) in results.items():
        h = H["3a"]
        rep.h(name, 3)
        rep.table(show(state_table(s["tendencia"], ["sobre SMA200", "bajo SMA200"], fwd_entry(x3, h),
                                   fwd_entry(u, h), fwd[h], h, rng, name=f"{name} · tendencia · 3a")),
                  pct=PCT)

    rep.h("4. VIX al comprar (S&P 500, desde 1990, horizonte 3 años)")
    u, s, x3, fwd = results["S&P 500"]
    vcol = pd.Series(pd.cut(vix.reindex(u.index).ffill(), VIX_EDGES, labels=VIX_LABELS),
                     index=u.index)
    rep.table(show(state_table(vcol, VIX_LABELS, fwd_entry(x3, H["3a"]), fwd_entry(u, H["3a"]),
                               fwd[H["3a"]], H["3a"], rng, name="S&P 500 · VIX · 3a")), pct=PCT)

    rep.h("5. Fuera de muestra: réplica < 2010 frente a fondo real ≥ 2010 (3x, horizonte 3 años)")
    rep.p("La parte fuera de muestra solo llega hasta señales de 2023: hacen falta 3 años de futuro.")
    for name, (u, s, x3, fwd) in results.items():
        rep.h(name, 3)
        rep.table(is_oos(s["caída"], DD_LABELS, fwd_entry(x3, H["3a"]), H["3a"]), pct=PCT)

    rep.h("6. Ejecución: comprar al cierre siguiente (base) o en el mismo cierre de la señal (3x, 3 años)")
    rows = {}
    for name, (u, s, x3, fwd) in results.items():
        h = H["3a"]
        f1 = pd.Series(ann(fwd_entry(x3, h), h), index=u.index)
        f0 = pd.Series(ann(fwd_log(x3, h), h), index=u.index)
        for lab in ["máximos (0 a −5%)", "−20 a −30%", "< −30%"]:
            m = s["caída"] == lab
            rows[f"{name} · {lab}"] = {"cierre siguiente (base)": f1[m].median(),
                                       "mismo cierre": f0[m].median()}
    rep.table(pd.DataFrame(rows).T, pct={"cierre siguiente (base)", "mismo cierre"})

    rep.h("7. Episodios: primer día en que la caída cruza −20% y −30% desde un máximo nuevo")
    rep.p("Las entradas tras grandes caídas no son miles de observaciones independientes: "
          "son unos pocos episodios. Esta tabla los muestra uno a uno: «señal» es el cierre en "
          "que la caída cruza el umbral y «compra», el cierre siguiente, en que se ejecuta.")
    for name, (u, s, x3, fwd) in results.items():
        for thr in (-0.20, -0.30):
            rep.h(f"{name} · cruce de {pct_es(thr, 0)}", 3)
            rep.table(drawdown_episodes(u, x3, thr, dd=s.dd), pct=PCT)

    rep.h("7b. Todos los días en caída < −30% y −20 a −30%, por episodio (3x, horizonte 3 años)")
    rep.p("La sección 7 mira solo el primer día de cada cruce; la sección 1 cuenta todos los días "
          "del estado. Aquí esos días se agrupan en episodios (tramos separados por más de 21 "
          "sesiones, como la columna «episodios»): la suma de días es la de la sección 1.")
    for name, (u, s, x3, fwd) in results.items():
        h = H["3a"]
        for lab in ("< −30%", "−20 a −30%"):
            rep.h(f"{name} · {lab}", 3)
            rep.table(state_episodes(s["caída"], lab, fwd_entry(x3, h), fwd_entry(u, h), h),
                      pct=PCT, index=False)

    rep.h("8. Robustez: la volatilidad > 35% sin la burbuja puntocom (3x, horizonte 3 años)")
    rows8 = {}
    for name, (u, s, x3, fwd) in results.items():
        h = H["3a"]
        y = pd.Series(ann(fwd_entry(x3, h), h), index=u.index)
        hv = s["volatilidad"] == "> 35%"
        dotcom = (u.index >= "1998-01-01") & (u.index < "2003-01-01")
        rows8[name] = {"todos_los_días": y.median(), "vol>35%": y[hv].median(),
                       "vol>35% sin 1998–2002": y[hv & ~dotcom].median(),
                       "días_sin_1998–2002": int(y[hv & ~dotcom].notna().sum())}
    rep.table(pd.DataFrame(rows8).T,
              pct={"todos_los_días", "vol>35%", "vol>35% sin 1998–2002"})

    pv = pd.DataFrame(PVALS)
    p = pv["p_placebo"].to_numpy(float)
    rep.h("9. Pruebas múltiples")
    bonf = 0.05 / len(p)
    surv = pv[pv.p_placebo <= bonf]
    zero = int((surv.p_placebo == 0).sum())
    rep.p(f"Se han hecho **{len(p)}** comparaciones con placebo (p bilateral: la diferencia de medianas "
          f"en valor absoluto). Con p ≤ 0,05 salen **{(p <= 0.05).sum()}**; si los p fueran exactos, por "
          f"puro azar se esperarían **{num_es(0.05 * len(p), 1)}**. El umbral de Bonferroni para "
          f"{len(p)} pruebas es p ≤ {num_es(bonf, 4)}, y el p placebo queda por debajo en "
          f"**{len(surv)}** ({zero} con p = 0: ninguno de los {num_es(N_PLACEBO, 0)} desplazamientos da "
          f"una diferencia igual o mayor; resolución {num_es(1 / N_PLACEBO, 4)}). Eso no permite afirmar "
          "que superen la corrección de Bonferroni: con 0 de "
          f"{num_es(N_PLACEBO, 0)}, la cota superior al 95% del p es de unos 3/{num_es(N_PLACEBO, 0)} = "
          f"{num_es(3 / N_PLACEBO, 4)}, por encima del umbral, y el placebo desplaza en bloque la serie de "
          "estados sobre resultados solapados de 3 y 5 años, así que ofrece pocas colocaciones de verdad "
          "distintas y no mide bien probabilidades tan pequeñas.")
    if len(surv):
        rep.h("Comparaciones con p placebo por debajo del umbral de Bonferroni (no concluyente)", 3)
        rep.table(surv.set_index(["prueba", "estado"]), pct={"Δ_mediana_3x"}, digits=4)
    nom = pv[pv.p_placebo <= 0.05]
    lst = "; ".join(f"{n} {lgrid_es(float(t.loc['todos', LGRID]))}" for n, t in dd_tables.items())
    sign = "; ".join(f"{n}: {pct_es(r['vol>35%'])} con todos los días y {pct_es(r['vol>35% sin 1998–2002'])} "
                     "sin 1998–2002" for n, r in rows8.items())
    rep.p(f"Lectura, con las cifras de las tablas: de las {len(nom)} comparaciones con p ≤ 0,05, "
          f"{int((nom['Δ_mediana_3x'] < 0).sum())} tienen Δ negativa; de las {len(surv)} con p placebo "
          f"por debajo del umbral de Bonferroni, {int((surv['Δ_mediana_3x'] < 0).sum())}. Mediana del 3x con "
          f"volatilidad > 35% (sección 8): {sign}. L_rejilla a 3 años comprando cualquier día "
          f"(fila «todos» de la sección 1, réplicas con costes): {lst}.")

    rep.h(f"10. Estado actual (cierre {results['S&P 500'][0].index[-1].date()})")
    now = {}
    for name, (u, s, x3, fwd) in results.items():
        last = s.iloc[-1]
        now[name] = {"caída": last.dd, "vol_21d": last.vol, "tramo_caída": last["caída"],
                     "tramo_vol": last["volatilidad"], "tendencia": last["tendencia"]}
    now["S&P 500"]["VIX"] = float(vix.dropna().iloc[-1])
    rep.table(pd.DataFrame(now).T, pct={"caída", "vol_21d"})

    rep.save()


if __name__ == "__main__":
    run()
