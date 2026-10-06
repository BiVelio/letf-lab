"""R30 — Cartera: ¿prima o beta? (H5) y la volatilidad corta junto a la LRS del artículo 2.

Preguntas (pre-registro del artículo 4, letf/preregistro/articulo4_volatilidad.md):
- H5 · ¿Prima o beta? El −0,5x tiene alfa CAPM positiva frente al SPY (HAC, p < 0,05, una
  cola) en la ventana principal y con SVXY real desde 2018.
- «Qué se publica»: las cifras frente a la LRS del artículo 2. ¿Diversifica la volatilidad
  corta a la LRS 3x (original) de Gayed y Bilello, o es la misma apuesta contra una caída de la
  bolsa? ¿Mejora alguna mezcla fija el crecimiento de la LRS sola?

Series:
- El −0,5x del pre-registro (vix.minus_half_spliced): sintético sobre el índice corto
  reconstruido hasta el 27-2-2018 y SVXY real (cierre de Yahoo) desde el 28-2-2018, con las
  dos convenciones de hora del tramo sintético (liquidación de CFE y 16:00 de Cboe). Como
  tercera pata, el VYLD sintético (−Δpuntos/100 + letra − 0,85%; convención de la
  liquidación): el VYLD real solo cotiza desde el 20-3-2025.
- La LRS 3x (original) del artículo 2 (r13): UPRO (sintético antes del 26-6-2009) cuando el
  S&P 500 de rentabilidad total cierra sobre su media de 200 sesiones, letras si no.
- SPY (el 1x del laboratorio en estas fechas) y la letra del laboratorio (^IRX).
Las patas de volatilidad se llevan al calendario de NYSE componiendo sus rendimientos
(vix.to_calendar).

H5 (especificación primaria fijada aquí, antes de estimar): rendimientos semanales
(viernes a viernes) en exceso de la letra, y = α + β·(SPY − letra), errores de Newey-West con
13 retardos (un trimestre); α anualizado × 52; p unilateral (H0: α ≤ 0). Ventanas: principal
21-12-2005 → 25-9-2026 (con las dos convenciones) y SVXY real 28-2-2018 → 25-9-2026. H5 se
sostiene si α > 0 con p < 0,05 en las dos ventanas (y en las dos convenciones de la
principal). Sensibilidades, todas en la tabla con su Holm: diaria (con SPY de t − 1, t y t + 1,
el beta de Dimson, por el desfase de 15 minutos de la liquidación hasta 2020; Newey-West 21),
mensual (Newey-West 12), el tramo anterior a 2018 y sin el 5 y el 6-2-2018.

Mezclas (una familia): LRS con 10, 20, 30, 50 y 100% en la pata de volatilidad (pesos
fijos reajustados al último cierre de cada mes) y con pesos inversos a la volatilidad de los
dos productos (σ de 63 sesiones del 3x y de la pata, fijados a fin de mes; 50% hasta tener
21 sesiones), con cada una de las tres patas: 18 configuraciones frente a la LRS sola.
Protocolo: objetivo al cierre de t, orden al cierre de t+1, 5 pb por unidad negociada entre
las dos patas (simulador de R29, r29.drift_sim). Inferencia: Δg frente a la LRS sola con
bootstrap estacionario emparejado (bloque 63, 2.000 remuestreos), Holm, Romano–Wolf y SPA.
Correlación en las caídas: semanal y en las peores semanas del SPY y de la LRS; episodios.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import norm

from letf_lab import metrics, stats
from letf_lab.data import vix
from letf_lab.report import FIG_W, Report, comma_fig, legend_band, num_es, pct_es
from letf_lab.research import r13_lrs as r13
from letf_lab.research import r22_inverse as r22
from letf_lab.research import r29_short_inverse as r29
from letf_lab.research.r26_vix_data import Tables, dmy, pp

SEED = 30
B = 2000
BLOCK = 63
LAG = r13.EXEC_LAG
END = vix.END
LRS = r13.ORIGINAL
SPY = "SPY"

LEGS = {"−0,5x (liquidación)": ("-0.5x", "settle"), "−0,5x (16:00)": ("-0.5x", "1600"),
        "VYLD sintético": ("vyld", "settle")}
H5_LEGS = ("−0,5x (liquidación)", "−0,5x (16:00)")
WEIGHTS = (0.1, 0.2, 0.3, 0.5, 1.0)
INVVOL = "inversa a la volatilidad"
VOL_WIN = 63

# decisión (el primer rendimiento con posición es dos sesiones después) y ventanas (a, b]
SIM_STARTS = {"principal": "2005-12-19", "desde 2018": "2018-02-26"}
WINDOWS = {"principal": ("2005-12-19", END), "antes del 28-2-2018": ("2005-12-19", "2018-02-27"),
           "desde el 28-2-2018": ("2018-02-26", END)}
SIM_OF = {"principal": "principal", "antes del 28-2-2018": "principal", "desde el 28-2-2018": "desde 2018"}
# H5: ventanas de rendimientos (a, b]
H5_WINDOWS = {"principal": ("2005-12-20", END), "antes del 28-2-2018": ("2005-12-20", "2018-02-27"),
              "SVXY real desde el 28-2-2018": ("2018-02-27", END)}
FREQ_SPEC = {"W": ("semanal", 52, 13), "D": ("diaria (Dimson)", 252, 21), "M": ("mensual", 12, 12)}
PRIMARY = "W"

EPISODES = {"crisis de 2008": ("2008-09-01", "2009-03-09"), "agosto de 2011": ("2011-07-22", "2011-10-03"),
            "agosto de 2015": ("2015-08-17", "2015-08-25"), "febrero de 2018": ("2018-01-26", "2018-02-09"),
            "4.º trimestre de 2018": ("2018-10-01", "2018-12-24"), "covid 2020": ("2020-02-19", "2020-03-23"),
            "2022": ("2022-01-03", "2022-10-12"), "agosto de 2024": ("2024-07-31", "2024-08-07"),
            "abril de 2025": ("2025-04-02", "2025-04-08")}

INK, MUTED = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, VIOLET, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#9a9994", "#e34948"


# ───────────────────────────── datos ─────────────────────────────

def load(D: dict | None = None) -> dict:
    """LRS 3x (original), su exposición, el 3x, SPY, la letra y las patas de volatilidad en el
    calendario del S&P 500 (sesiones de NYSE), hasta vix.END."""
    D = D or r13.load()
    d = D["S&P 500"]
    idx = d["u"].index[(d["u"].index <= END) & (d["u"].index >= "2004-01-01")]
    sig = r13.above_sma(r13.signal_index(d))
    lrs = r13.run_weights(d, sig).reindex(idx)
    legs = {}
    for name, (key, conv) in LEGS.items():
        raw = vix.minus_half_spliced(conv) if key == "-0.5x" else vix.synthetic(key, conv)
        legs[name] = vix.to_calendar(raw, idx)
    return {"d": d, "idx": idx, "lrs": lrs, "held": sig.shift(LAG).reindex(idx), "x3": d["x3"].reindex(idx),
            "spy": d["u"].reindex(idx), "rf": d["rf"].reindex(idx), "legs": legs}


def win(r: pd.Series, a, b) -> pd.Series:
    return r29.win(r, a, b)


# ───────────────────────────── H5: alfa CAPM ─────────────────────────────

def _agg(j: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "D":
        return j
    rule = "W-FRI" if freq == "W" else "ME"
    g = (1 + j).resample(rule).prod() - 1
    n = j.iloc[:, 0].resample(rule).count()
    return g[n > 0]


def capm(y: pd.Series, m: pd.Series, rf: pd.Series, a, b, freq: str = PRIMARY, drop=None) -> dict:
    """Alfa CAPM de y frente a m en (a, b] con rendimientos de la frecuencia pedida en exceso de la
    letra y errores de Newey-West (FREQ_SPEC). Diaria: con m de t − 1, t y t + 1 (beta de Dimson =
    suma de los tres). drop: sesiones que se quitan antes de agregar."""
    j = pd.concat([y, m, rf], axis=1, keys=["y", "m", "rf"])
    j = j[(j.index > pd.Timestamp(a)) & (j.index <= pd.Timestamp(b))].dropna()
    if drop is not None:
        j = j[~j.index.isin(pd.DatetimeIndex(drop))]
    g = _agg(j, freq)
    ye, me = g.y - g.rf, g.m - g.rf
    name, per, lags = FREQ_SPEC[freq]
    X = pd.DataFrame({"SPY": me})
    if freq == "D":
        X["SPY t−1"], X["SPY t+1"] = me.shift(1), me.shift(-1)
    fit = stats.ols_hac(ye, X, lags)
    beta = float(np.sum(fit.beta[1:]))
    return {"frecuencia": name, "desde": g.index[0].date(), "hasta": g.index[-1].date(), "n": fit.n,
            "α anual": float(fit.beta[0] * per), "t(α)": float(fit.t[0]), "p (α > 0)": float(norm.sf(fit.t[0])),
            "β": beta, "R²": fit.r2, "rendimiento en exceso anual": float(ye.mean() * per),
            "β × prima del SPY": float(beta * me.mean() * per), "estimador": fit.estimator,
            "se_α": float(fit.se[0] * per)}


def h5_table(L: dict) -> pd.DataFrame:
    """Todas las especificaciones de H5 (y la LRS como referencia), con Holm sobre las del −0,5x."""
    rows = []
    for wn, (a, b) in H5_WINDOWS.items():
        legs = H5_LEGS if wn != "SVXY real desde el 28-2-2018" else H5_LEGS[:1]
        for leg in legs:
            lab = leg if wn != "SVXY real desde el 28-2-2018" else "SVXY real (−0,5x)"
            for fq in (PRIMARY, "D", "M"):
                for drop in ((None, vix.FEB_2018) if a < "2018-02-05" < str(b) else (None,)):
                    r = capm(L["legs"][leg], L["spy"], L["rf"], a, b, fq, drop)
                    rows.append({"ventana": wn, "serie": lab, "sin 5-6-2-2018": "sí" if drop is not None else "no",
                                 "primaria": "sí" if (fq == PRIMARY and drop is None) else "", **r})
    t = pd.DataFrame(rows)
    t["p_Holm (todas)"] = stats.holm(t["p (α > 0)"].to_numpy())
    ref = []
    for wn, (a, b) in H5_WINDOWS.items():
        for lab, s in (("LRS 3x (original)", L["lrs"]), ("VYLD sintético", L["legs"]["VYLD sintético"])):
            if lab == "VYLD sintético" and wn.startswith("SVXY"):
                continue
            ref.append({"ventana": wn, "serie": lab, "sin 5-6-2-2018": "no", "primaria": "",
                        **capm(s, L["spy"], L["rf"], a, b, PRIMARY)})
    return pd.concat([t, pd.DataFrame(ref)], ignore_index=True)


def h5_verdict(t: pd.DataFrame) -> dict:
    prim = t[(t.primaria == "sí")]
    main = prim[prim.ventana == "principal"]
    post = prim[prim.ventana == "SVXY real desde el 28-2-2018"]
    ok_main = bool(((main["α anual"] > 0) & (main["p (α > 0)"] < 0.05)).all())
    ok_post = bool(((post["α anual"] > 0) & (post["p (α > 0)"] < 0.05)).all())
    return {"principal": ok_main, "desde 2018": ok_post, "H5": ok_main and ok_post}


def updown_betas(L: dict) -> pd.DataFrame:
    """Descriptivo: beta semanal del −0,5x en las semanas de subida y de bajada del SPY."""
    rows = []
    for wn, (a, b) in H5_WINDOWS.items():
        for leg in (H5_LEGS if not wn.startswith("SVXY") else H5_LEGS[:1]):
            j = pd.concat([L["legs"][leg], L["spy"], L["rf"]], axis=1, keys=["y", "m", "rf"])
            j = _agg(j[(j.index > pd.Timestamp(a)) & (j.index <= pd.Timestamp(b))].dropna(), "W")
            ye, me = j.y - j.rf, j.m - j.rf
            X = pd.DataFrame({"sube": me.clip(lower=0), "baja": me.clip(upper=0)})
            fit = stats.ols_hac(ye, X, 13)
            rows.append({"ventana": wn, "serie": leg if not wn.startswith("SVXY") else "SVXY real (−0,5x)",
                         "β en semanas de subida": float(fit.beta[1]), "β en semanas de bajada": float(fit.beta[2]),
                         "t de la diferencia (≈)": float((fit.beta[2] - fit.beta[1]) / np.hypot(fit.se[1], fit.se[2]))})
    return pd.DataFrame(rows).set_index(["ventana", "serie"])


# ───────────────────────────── mezclas ─────────────────────────────

def mix_name(leg: str, v) -> str:
    if v == INVVOL:
        return f"LRS + {leg}, pesos inversos a la volatilidad"
    if v == 1.0:
        return f"{leg} solo"
    return f"LRS {pct_es(1 - v, 0)} + {leg} {pct_es(v, 0)}"


def invvol_weight(L: dict, leg: str) -> pd.Series:
    """Peso de la pata con pesos inversos a la volatilidad de los dos productos (σ de VOL_WIN sesiones
    del 3x y de la pata hasta el cierre de t), fijado al último cierre de cada mes; 0,5 hasta tener 21
    sesiones de la pata."""
    s3 = L["x3"].rolling(VOL_WIN, min_periods=21).std()
    sl = L["legs"][leg].rolling(VOL_WIN, min_periods=21).std()
    w = (1 / sl) / (1 / sl + 1 / s3)
    idx = L["idx"]
    month_end = r22.rebalance_days(idx, "M").copy()
    month_end[-1] = False                     # la última sesión de la muestra no cierra un mes
    w = w.where(month_end).ffill()
    return w.fillna(0.5)


def mix_series(L: dict, leg: str, v, start, drop=None) -> pd.Series:
    """LRS + pata con pesos fijos (reajuste mensual) o inversos a la volatilidad (reajuste cuando
    cambian, a fin de mes), desde la decisión del cierre de `start`."""
    idx = L["idx"]
    X = pd.DataFrame({"LRS": L["lrs"], "pata": L["legs"][leg]}).reindex(idx)
    live = pd.Series(np.where(idx >= pd.Timestamp(start), 1.0, np.nan), index=idx)
    if v == INVVOL:
        wl = invvol_weight(L, leg)
        tgt = pd.DataFrame({"LRS": (1 - wl) * live, "pata": wl * live})
        freq = "N"
    else:
        tgt = pd.DataFrame({"LRS": (1 - v) * live, "pata": v * live})
        freq = "M"
    rf = L["rf"]
    if drop is not None:
        keep = ~idx.isin(pd.DatetimeIndex(drop))
        tgt, X, rf = tgt[keep], X[keep], rf[keep]
    return r29.drift_sim(tgt, X, rf, freq, margin_rule="ninguna")[0]


def simulate_mixes(L: dict, start, drop=None) -> tuple[dict, dict]:
    R, meta = {}, {}
    R[LRS] = mix_series(L, next(iter(LEGS)), 0.0, start, drop)
    idx = L["idx"] if drop is None else L["idx"][~L["idx"].isin(pd.DatetimeIndex(drop))]
    live = pd.Series(np.where(idx >= pd.Timestamp(start), 1.0, np.nan), index=idx)
    R["SPY comprado"] = r29.drift_sim(pd.DataFrame({"SPY": live}), L["spy"].to_frame("SPY").reindex(idx),
                                      L["rf"].reindex(idx), "M")[0]
    for leg in LEGS:
        for v in WEIGHTS + (INVVOL,):
            k = mix_name(leg, v)
            R[k] = mix_series(L, leg, v, start, drop)
            meta[k] = {"pata": leg, "peso de la pata": v if v != INVVOL else INVVOL}
    return R, meta


def mix_results(L: dict, S: dict, rng) -> tuple[dict, pd.DataFrame]:
    out, fams = {}, []
    for wn, (a, b) in WINDOWS.items():
        R, meta = S[SIM_OF[wn]]
        pairs = {k: (R[k], R[LRS]) for k in meta}
        ft = r29.family_test(pairs, a, b, rng)
        rows = {}
        for k in [LRS] + list(meta):
            p = r29.perf(R[k], L["rf"], a, b)
            rows[k] = {"pata": meta.get(k, {}).get("pata", ""), "peso de la pata": meta.get(k, {}).get("peso de la pata", 0.0),
                       **p}
        t = pd.DataFrame(rows).T
        t.loc["SPY comprado"] = {"pata": "", "peso de la pata": "", **r29.perf(R["SPY comprado"], L["rf"], a, b)}
        t = t.join(ft)
        t.attrs.update(ft.attrs)
        out[wn] = t
        fams.append({"ventana": wn, "configuraciones": len(ft), "p_SPA": ft.attrs["p_SPA"], "mejor": ft.attrs["mejor"],
                     "Δg_mejor": ft.attrs["Δg_mejor"], "rechazos_Holm_5%": int((ft.p_Holm < 0.05).sum()),
                     "rechazos_RW_5%": int((ft.p_RW < 0.05).sum())})
    return out, pd.DataFrame(fams)


# ───────────────────────────── correlación en las caídas ─────────────────────────────

def weekly(r: pd.Series) -> pd.Series:
    r = r.dropna()
    g = (1 + r).resample("W-FRI").prod() - 1
    return g[r.resample("W-FRI").count() > 0]


def crash_corr(L: dict, S: dict) -> pd.DataFrame:
    """Correlación de la LRS con cada pata: diaria y semanal, en las peores semanas del SPY y de la LRS,
    con la LRS dentro o fuera toda la semana, y coincidencia de colas (semanal)."""
    rows = []
    for wn, (a, b) in WINDOWS.items():
        base = pd.DataFrame({"LRS": win(L["lrs"], a, b), "SPY": win(L["spy"], a, b),
                             "held": win(L["held"], a, b), **{k: win(v, a, b) for k, v in L["legs"].items()}}).dropna()
        W = pd.DataFrame({c: weekly(base[c]) for c in ["LRS", "SPY", *L["legs"]]})
        held_w = base.held.resample("W-FRI").mean().reindex(W.index)
        q_spy, q_lrs = W.SPY.quantile(0.05), W.LRS.quantile(0.05)
        bad_spy, bad_lrs = W.SPY <= q_spy, W.LRS <= q_lrs
        for leg in L["legs"]:
            q_leg = W[leg].quantile(0.05)
            bad_leg = W[leg] <= q_leg
            rows.append({"ventana": wn, "pata": leg, "semanas": len(W),
                         "corr. diaria": base.LRS.corr(base[leg]), "corr. semanal": W.LRS.corr(W[leg]),
                         "corr. semanal SPY–pata": W.SPY.corr(W[leg]),
                         "corr. en el 5% peor del SPY": W.LRS[bad_spy].corr(W[leg][bad_spy]),
                         "corr. en el 5% peor de la LRS": W.LRS[bad_lrs].corr(W[leg][bad_lrs]),
                         "corr. con la LRS dentro": W.LRS[held_w == 1].corr(W[leg][held_w == 1]),
                         "corr. con la LRS fuera": W.LRS[held_w == 0].corr(W[leg][held_w == 0]),
                         "P(pata en su 5% peor | LRS en su 5% peor)": float(bad_leg[bad_lrs].mean()),
                         "P(pata en su 5% peor | SPY en su 5% peor)": float(bad_leg[bad_spy].mean()),
                         "pata media en el 5% peor de la LRS": float(W[leg][bad_lrs].mean()),
                         "LRS media en el 5% peor de la pata": float(W.LRS[bad_leg].mean())})
    return pd.DataFrame(rows).set_index(["ventana", "pata"])


def episodes(L: dict, S: dict) -> pd.DataFrame:
    R, _ = S["principal"]
    leg = "−0,5x (liquidación)"
    series = {"SPY": L["spy"], LRS: R[LRS], **{k: L["legs"][k] for k in LEGS},
              mix_name(leg, 0.2): R[mix_name(leg, 0.2)], mix_name(leg, 0.5): R[mix_name(leg, 0.5)]}
    rows = {}
    for ep, (a, b) in EPISODES.items():
        rows[ep] = {k: float((1 + r22.absorb(s.loc[a:b].dropna())).prod() - 1) for k, s in series.items()}
    t = pd.DataFrame(rows).T
    t.index.name = "episodio"
    return t


# ───────────────────────────── cálculo completo ─────────────────────────────

def compute(D: dict | None = None) -> dict:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    L = load(D)
    rng = np.random.default_rng(SEED)
    S = {s: simulate_mixes(L, a) for s, a in SIM_STARTS.items()}
    mixes, fams = mix_results(L, S, rng)
    # sensibilidad: la familia de la ventana principal sin el 5 y el 6-2-2018
    Sd = simulate_mixes(L, SIM_STARTS["principal"], vix.FEB_2018)
    a, b = WINDOWS["principal"]
    fd = r29.family_test({k: (Sd[0][k], Sd[0][LRS]) for k in Sd[1]}, a, b, np.random.default_rng(SEED + 1))
    h5 = h5_table(L)
    return {"L": L, "S": S, "mixes": mixes, "fams": fams, "sin_feb": fd, "h5": h5, "h5_verdict": h5_verdict(h5),
            "updown": updown_betas(L), "crash": crash_corr(L, S), "episodes": episodes(L, S)}


# ───────────────────────────── figuras ─────────────────────────────

def _style(ax):
    ax.grid(True, color="#e3e2de", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_h5(h5: pd.DataFrame, rep: Report):
    """α anual (pp) con su IC del 90% (α ± 1,645·EE) en cada especificación del −0,5x."""
    t = h5[~h5.serie.isin([LRS, "VYLD sintético"])].copy()
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(FIG_W, 9.0))
    y = np.arange(len(t))
    cols = {"principal": BLUE, "antes del 28-2-2018": AQUA, "SVXY real desde el 28-2-2018": ORANGE}
    for i, (_, r) in enumerate(t.iterrows()):
        a, se = r["α anual"] * 100, r["se_α"] * 100
        col = cols[r.ventana]
        ax.plot([a - 1.645 * se, a + 1.645 * se], [i, i], color=col, lw=2.0)
        ax.plot(a, i, "o", ms=9 if r.primaria == "sí" else 7, color=col if r.primaria == "sí" else "white",
                markeredgecolor=col, markeredgewidth=2.0)
    labels = []
    for _, r in t.iterrows():
        conv = "" if r.serie.startswith("SVXY") else (" · 16:00" if "16:00" in r.serie else " · liquidación")
        nf = " · sin 5-6-2-2018" if r["sin 5-6-2-2018"] == "sí" else ""
        labels.append(f"{r.frecuencia.split(' ')[0]}{conv}{nf}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color=MUTED, lw=1.0)
    ax.set_xlabel("α CAPM frente al SPY (pp/año) e intervalo del 90%")
    ax.set_title("¿Prima o beta? Alfa del −0,5x frente al SPY")
    _style(ax)
    h = [plt.Line2D([], [], color=c, lw=2.0, marker="o", ms=8) for c in cols.values()]
    h.append(plt.Line2D([], [], color=INK, lw=0, marker="o", ms=8, markerfacecolor="white", markeredgewidth=2.0))
    legend_band(fig, h, ["21-12-2005 → 25-9-2026", "21-12-2005 → 27-2-2018", "SVXY real, 28-2-2018 → 25-9-2026",
                         "sensibilidad (relleno: especificación primaria)"], where="bottom", ncol=2, fontsize=11)
    comma_fig(fig)
    rep.fig(fig, "alfa_capm",
            "Figura: alfa CAPM anual del −0,5x frente al SPY con su intervalo del 90% (el extremo izquierdo es la cota "
            "unilateral del 95%: H5 pedía que quedara por encima de 0) en cada ventana, convención de hora y frecuencia. "
            "Relleno: la especificación primaria (semanal, Newey-West 13)")


def fig_episodes(ep: pd.DataFrame, rep: Report):
    keys = [("SPY", GREY), (LRS, BLUE), ("−0,5x (liquidación)", ORANGE), ("−0,5x (16:00)", VIOLET)]
    t = ep[[k for k, _ in keys]].iloc[::-1] * 100
    fig, ax = plt.subplots(figsize=(FIG_W, 7.4))
    y = np.arange(len(t))
    hgt = 0.2
    for j, (k, c) in enumerate(keys):
        ax.barh(y + (1.5 - j) * hgt, t[k], height=hgt * 0.9, color=c, label=k)
    ax.set_yticks(y)
    ax.set_yticklabels(t.index, fontsize=11)
    ax.axvline(0, color=MUTED, lw=1.0)
    ax.set_xlabel("Rentabilidad en el episodio (%)")
    ax.set_title("Las caídas: LRS 3x (original) y −0,5x")
    _style(ax)
    h, lab = ax.get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=2, fontsize=11)
    comma_fig(fig)
    rep.fig(fig, "episodios",
            "Figura: rentabilidad de SPY, de la LRS 3x (original) del artículo 2 y del −0,5x (liquidación y 16:00; SVXY "
            "real desde el 28-2-2018) en cada episodio de caída del S&P 500 (fechas aproximadas del máximo al mínimo)")


def fig_wealth(L: dict, S: dict, rep: Report):
    R, _ = S["principal"]
    leg = "−0,5x (liquidación)"
    a, b = WINDOWS["principal"]
    items = [("SPY comprado", R["SPY comprado"], GREY, "-"), (LRS, R[LRS], BLUE, "-"), (f"{leg} solo", R[mix_name(leg, 1.0)], ORANGE, "-"),
             (mix_name(leg, 0.2), R[mix_name(leg, 0.2)], AQUA, "-"), (mix_name(leg, 0.5), R[mix_name(leg, 0.5)], VIOLET, "-")]
    fig, ax = plt.subplots(figsize=(FIG_W, 5.8))
    for lab, r, col, ls in items:
        x = r22.absorb(win(r, a, b))
        w = (1 + x).cumprod()
        ax.semilogy(w.index, w.values, color=col, ls=ls, lw=1.7, label=f"{lab}: {pct_es(metrics.cagr(x))}/año")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num_es(v, 2 if v < 1 else 0)))
    ax.set(ylabel="Valor de 1 $ (log)", title="LRS 3x (original), −0,5x y sus mezclas, 2005–2026")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style(ax)
    h, lab = ax.get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=1, fontsize=11)
    comma_fig(fig)
    rep.fig(fig, "mezclas",
            "Figura: valor de 1 dólar en SPY, la LRS 3x (original), el −0,5x (liquidación hasta el 27-2-2018 y SVXY real "
            "después) y dos mezclas con reajuste mensual, 21-12-2005 → 25-9-2026; escala logarítmica. Sin préstamo ni "
            "impuestos")


# ───────────────────────────── informe ─────────────────────────────

def run(C: dict | None = None):
    warnings.filterwarnings("ignore")
    rep = Report("r30_vol_portfolio", "R30 — Cartera: ¿prima o beta? (H5) y la volatilidad corta junto a la LRS")
    for f in rep.dir.glob("tabla_*.csv"):
        f.unlink()
    C = C or compute()
    tab = Tables(rep)
    TN = {}
    L, h5, mixes, fams, cr, ep = C["L"], C["h5"], C["mixes"], C["fams"], C["crash"], C["episodes"]
    ud, sfeb, verdict = C["updown"], C["sin_feb"], C["h5_verdict"]
    main = mixes["principal"]
    prim = h5[h5.primaria == "sí"].set_index(["ventana", "serie"])
    pm_s = prim.loc[("principal", "−0,5x (liquidación)")]
    pm_h = prim.loc[("principal", "−0,5x (16:00)")]
    pp_s = prim.loc[("SVXY real desde el 28-2-2018", "SVXY real (−0,5x)")]
    h5m = h5[~h5.serie.isin([LRS, "VYLD sintético"])]
    lrs_a = h5[(h5.serie == LRS) & (h5.ventana == "principal")].iloc[0]
    crm = cr.loc["principal"]
    c_s = crm.loc["−0,5x (liquidación)"]
    fam = fams.set_index("ventana")
    best = main.loc[fam.loc["principal", "mejor"]]
    post = mixes["desde el 28-2-2018"]
    post_mix = post[post.index.str.startswith("LRS ") & (post.index != LRS)]
    udm = ud.loc[("principal", "−0,5x (liquidación)")]

    def ex(r):
        return f"{pp(r['α anual'])} pp/año (t = {num_es(r['t(α)'], 2)}, p {r22.peq(r['p (α > 0)'])})"

    rep.p(__doc__.split("\n", 2)[2])
    rep.h("Resumen")
    rep.p("\n".join([
        f"- **H5 no se sostiene: es sobre todo beta.** Con la especificación primaria (semanal, Newey-West 13), el alfa CAPM "
        f"del −0,5x frente al SPY es de {ex(pm_s)} en la ventana principal con la liquidación de CFE, {ex(pm_h)} con las "
        f"16:00 y {ex(pp_s)} con SVXY real desde el 28-2-2018. Ninguna de las {len(h5m)} especificaciones llega a p < 0,05 "
        f"(la menor, {num_es(h5m['p (α > 0)'].min(), 3)}; con Holm, {num_es(h5m['p_Holm (todas)'].min(), 2)}). Su beta es "
        f"{num_es(pm_s['β'], 2)}: de los {num_es(pm_s['rendimiento en exceso anual'] * 100, 1)} pp anuales sobre la letra, "
        f"{num_es(pm_s['β × prima del SPY'] * 100, 1)} son beta × la prima del SPY.",
        f"- **La beta es asimétrica.** Semana a semana el −0,5x se mueve {num_es(udm['β en semanas de subida'], 2)} veces "
        f"el SPY cuando sube y {num_es(udm['β en semanas de bajada'], 2)} veces cuando baja (t de la diferencia ≈ "
        f"{num_es(udm['t de la diferencia (≈)'], 1)}; con SVXY real desde 2018, "
        f"{num_es(ud.loc[('SVXY real desde el 28-2-2018', 'SVXY real (−0,5x)'), 'β en semanas de subida'], 2)} y "
        f"{num_es(ud.loc[('SVXY real desde el 28-2-2018', 'SVXY real (−0,5x)'), 'β en semanas de bajada'], 2)}). Lo que "
        "cobra es sobre todo el pago por cargar con las caídas, una prima que el CAPM no ve como alfa.",
        f"- **No diversifica a la LRS cuando hace falta.** La correlación semanal de la LRS 3x (original) con el −0,5x es "
        f"{num_es(c_s['corr. semanal'], 2)}: {num_es(c_s['corr. con la LRS dentro'], 2)} en las semanas en que la LRS está "
        f"dentro y {num_es(c_s['corr. con la LRS fuera'], 2)} cuando está en letras. En el 5% de peores semanas de la LRS, "
        f"el −0,5x está también en su 5% peor el {pct_es(c_s['P(pata en su 5% peor | LRS en su 5% peor)'], 0)} de las veces "
        f"(el azar daría un 5%) y pierde de media un {pct_es(-c_s['pata media en el 5% peor de la LRS'], 1)} por semana. "
        f"La LRS se libra de algunas caídas largas porque sale del mercado (2008: LRS {pct_es(ep.loc['crisis de 2008', LRS], 1)}, "
        f"−0,5x {pct_es(ep.loc['crisis de 2008', '−0,5x (liquidación)'], 1)}); en las rápidas pierden los dos (covid: "
        f"{pct_es(ep.loc['covid 2020', LRS], 1)} y {pct_es(ep.loc['covid 2020', '−0,5x (liquidación)'], 1)}; febrero de 2018: "
        f"{pct_es(ep.loc['febrero de 2018', LRS], 1)} y {pct_es(ep.loc['febrero de 2018', '−0,5x (liquidación)'], 1)}).",
        f"- **Ninguna mezcla mejora a la LRS sola tras la búsqueda.** De las 18 (10–100% en la pata o pesos inversos a la "
        f"volatilidad, con tres patas), el SPA da p = {num_es(fam.loc['principal', 'p_SPA'], 2)} en la ventana principal, "
        f"{num_es(fam.loc['antes del 28-2-2018', 'p_SPA'], 2)} antes del 28-2-2018 y "
        f"{num_es(fam.loc['desde el 28-2-2018', 'p_SPA'], 2)} después; la mejor en la principal es «{best.name}» "
        f"({pp(best['Δg'])} pp/año, IC90 {r29.ci_text(best.IC90_bajo, best.IC90_alto)}, p_RW {r22.peq(best.p_RW)}). "
        f"Desde 2018 todas las mezclas crecen menos que la LRS sola (Δg de {pp(post_mix['Δg'].max())} a "
        f"{pp(post_mix['Δg'].min())} pp). Las mezclas con la pata de volatilidad bajan algo la volatilidad y la caída "
        "máxima, no el riesgo de las caídas rápidas.",
        f"- **Referencias en la ventana principal (21-12-2005 → 25-9-2026):** LRS 3x (original) "
        f"{pct_es(main.loc[LRS, 'CAGR'])} al año (caída máxima {pct_es(main.loc[LRS, 'maxDD'], 0)}; su alfa CAPM semanal, "
        f"{pp(lrs_a['α anual'])} pp, t = {num_es(lrs_a['t(α)'], 2)}), −0,5x {pct_es(main.loc[mix_name('−0,5x (liquidación)', 1.0), 'CAGR'])} "
        f"({pct_es(main.loc[mix_name('−0,5x (liquidación)', 1.0), 'maxDD'], 0)}), SPY {pct_es(main.loc['SPY comprado', 'CAGR'])} "
        f"({pct_es(main.loc['SPY comprado', 'maxDD'], 0)}).",
    ]))

    rep.h("1. H5: ¿prima o beta?")
    rep.p("Regresión y − letra = α + β·(SPY − letra) + ε, con errores de Newey-West; α y el rendimiento en exceso, "
          "anualizados. En la diaria, β es la suma de los coeficientes de SPY en t − 1, t y t + 1 (beta de Dimson): hasta "
          "octubre de 2020 la liquidación de CFE llega 15 minutos después del cierre del SPY. p es unilateral (H0: α ≤ 0); "
          "p_Holm, sobre las 27 especificaciones del −0,5x. La LRS y el VYLD sintético van como referencia, fuera de H5.")
    rep.p("*Pie de fórmula: el alfa es lo que el −0,5x rinde por encima de la letra y de lo que explica su exposición a la "
          "bolsa (β veces el exceso del SPY).* Notación: y, el rendimiento del −0,5x en la semana; β, su sensibilidad al "
          "SPY; ε, lo que no explica el SPY.")
    hv = h5.drop(columns=["se_α"]).copy()
    for c in ("α anual", "rendimiento en exceso anual", "β × prima del SPY"):
        hv[c] = hv[c] * 100
    hv = hv.rename(columns={"α anual": "α (pp/año)", "rendimiento en exceso anual": "exceso sobre la letra (pp/año)",
                            "β × prima del SPY": "β × prima del SPY (pp/año)"}).set_index(["ventana", "serie"])
    tab("Alfa CAPM del −0,5x frente al SPY en cada especificación (primaria: semanal, sin quitar días) y, al final, la LRS "
        "3x (original) y el VYLD sintético como referencia", hv, digits=3)
    TN["h5"] = tab.n
    tab("Beta semanal del −0,5x en las semanas de subida y de bajada del SPY (descriptivo; Newey-West 13)", ud, digits=2)
    TN["ud"] = tab.n
    fig_h5(h5, rep)
    rep.p(f"**Veredicto.** H5 pedía α > 0 con p < 0,05 en la ventana principal y con SVXY real desde 2018. Ventana "
          f"principal: {'sí' if verdict['principal'] else 'no'}; desde 2018: {'sí' if verdict['desde 2018'] else 'no'}. "
          + ("H5 no se sostiene (pedía un α significativo en las dos). Un α no significativo no la rechaza en sentido "
             "estricto, pero tampoco la apoya, y el pre-registro la da por falsada si α ≤ 0 o no es significativo. "
             if not verdict["H5"] else "H5 se sostiene. ") +
          "El alfa estimado es pequeño o negativo frente a su error (unos 5 pp al año "
          "de error estándar): con 20 años de datos no se distingue de 0, y desde 2018 es negativo en las tres "
          "frecuencias. La convención de las 16:00 y quitar el 5 y el 6-2-2018 lo suben unos pocos puntos, sin acercarlo a "
          "la significación. Lo que sí se ve es la asimetría (tabla «T:ud»): el −0,5x amplifica las bajadas del SPY más "
          "que las subidas, el perfil de quien vende seguros contra caídas.")

    rep.h("2. La volatilidad corta junto a la LRS: correlación en las caídas")
    cv = cr.copy()
    tab("Correlación de la LRS 3x (original) con cada pata de volatilidad: diaria, semanal, en las peores semanas del SPY "
        "y de la LRS (5%), según la LRS esté dentro o fuera toda la semana, y coincidencia de colas", cv,
        pct={"P(pata en su 5% peor | LRS en su 5% peor)", "P(pata en su 5% peor | SPY en su 5% peor)",
             "pata media en el 5% peor de la LRS", "LRS media en el 5% peor de la pata"}, digits=2)
    TN["cr"] = tab.n
    tab("Rentabilidad en cada episodio de caída del S&P 500 (fechas aproximadas del máximo al mínimo)", ep,
        pct=set(ep.columns), digits=3)
    TN["ep"] = tab.n
    fig_episodes(ep, rep)
    rep.p("**Lectura.** La LRS y la volatilidad corta son la misma apuesta mientras la LRS está dentro del mercado "
          f"(correlación semanal {num_es(c_s['corr. con la LRS dentro'], 2)}); la diversificación solo aparece cuando la "
          "LRS ya ha salido a letras, y entonces es la LRS la que protege, no la pata de volatilidad. En las caídas rápidas "
          "(agosto de 2015, febrero de 2018, la covid, agosto de 2024) la LRS no tiene tiempo de salir y las dos pierden a "
          f"la vez. En 2008 la LRS salió a tiempo ({pct_es(ep.loc['crisis de 2008', LRS], 1)}) y el −0,5x perdió la mitad; "
          f"en abril de 2025 la LRS ya estaba fuera ({pct_es(ep.loc['abril de 2025', LRS], 1)}). En 2022 y en el cuarto "
          f"trimestre de 2018, en cambio, la LRS perdió más que el −0,5x ({pct_es(ep.loc['2022', LRS], 1)} frente a "
          f"{pct_es(ep.loc['2022', '−0,5x (liquidación)'], 1)} en 2022): caídas lentas, con entradas y salidas en falso "
          "de la LRS. El VYLD sintético pierde menos en las caídas (pierde un 1% por cada punto que "
          "sube el futuro: con el futuro en 20 equivale a un −0,2x) y apenas se mueve con la LRS en las peores semanas del "
          "SPY, pero crece menos (sección 3).")

    rep.h("3. Mezclas con la LRS")
    cols = {"pata": "pata", "peso de la pata": "peso de la pata", "CAGR": "CAGR", "vol": "vol", "sharpe": "sharpe",
            "maxDD": "maxDD", "peor_mes": "peor mes", "Δg": "Δg (pp)", "p": "p", "p_Holm": "p_Holm", "p_RW": "p_RW"}

    def mv(t):
        return r29.dg_view(t, cols)

    tab("Ventana principal (21-12-2005 → 25-9-2026): LRS 3x (original), mezclas y patas solas; Δg frente a la LRS sola "
        "(18 configuraciones, una familia)", mv(main), pct={"CAGR", "vol", "maxDD", "peor mes"}, digits=3)
    TN["main"] = tab.n
    piv = pd.DataFrame({f"Δg (pp) · {w}": t["Δg"] * 100 for w, t in mixes.items()})
    piv = piv.join(pd.DataFrame({f"CAGR · {w}": t["CAGR"] for w, t in mixes.items()}))
    piv = piv.join(pd.DataFrame({f"p_RW · {w}": t["p_RW"] for w, t in mixes.items()}))
    tab("Antes y después del 28-2-2018: CAGR, Δg frente a la LRS sola y p_RW de cada configuración (desde 2018 las dos "
        "convenciones del −0,5x son la misma serie, SVXY real)", piv, pct={c for c in piv.columns if c.startswith("CAGR")},
        digits=3)
    tab("SPA de Hansen y rechazos de Holm y Romano–Wolf al 5% de la familia de mezclas por ventana", fam.assign(
        **{"Δg_mejor": fam["Δg_mejor"] * 100}).rename(columns={"Δg_mejor": "Δg de la mejor (pp)"}), digits=3)
    sv = sfeb[["Δg", "IC90_bajo", "IC90_alto", "p", "p_RW"]].copy()
    sv["IC90 (pp)"] = [r29.ci_text(a, b) for a, b in zip(sv.IC90_bajo, sv.IC90_alto)]
    sv = sv.drop(columns=["IC90_bajo", "IC90_alto"]).assign(Δg=sv["Δg"] * 100).rename(columns={"Δg": "Δg (pp)"})
    tab(f"Sensibilidad del pre-registro: la familia de la ventana principal sin el 5 y el 6-2-2018 (SPA p = "
        f"{num_es(sfeb.attrs['p_SPA'], 3)})", sv, digits=3)
    fig_wealth(L, C["S"], rep)
    rep.p(f"**Lectura.** Ninguna mezcla supera a la LRS sola tras la búsqueda en ninguna ventana (tabla «T:main» y "
          f"siguientes). Añadir hasta un 30% de −0,5x apenas cambia el crecimiento (Δg de décimas, con intervalos de "
          f"varios puntos) y baja algo la volatilidad y la caída máxima; más peso no ayuda, y desde 2018 cualquier peso "
          f"resta. Las mezclas con el VYLD sintético bajan más la volatilidad y la caída máxima "
          f"(50%: {pct_es(main.loc[mix_name('VYLD sintético', 0.5), 'maxDD'], 0)} frente a "
          f"{pct_es(main.loc[LRS, 'maxDD'], 0)}), a costa de crecer menos. Mejorar la ratio de Sharpe no era la hipótesis "
          "y no se contrasta aquí; el pre-registro mide Δg.")

    rep.h("Salvedades y datos sin verificar")
    rep.p("\n".join([
        "- El −0,5x anterior al 28-2-2018 es sintético (índice reconstruido; SVXY era entonces −1x). Las dos convenciones de "
        "hora dan el rango de lo que habría rendido; ningún producto −0,5x cotizaba entonces.",
        "- El VYLD real solo existe desde el 20-3-2025: el de la tabla es el sintético de R26 (índice de puntos).",
        "- Pocos episodios extremos en 20 años (2008, 2011, 2015, 2018, 2020, 2022, 2024, 2025): el alfa y las mezclas "
        "dependen de ellos. Fechas de los episodios aproximadas.",
        "- Sin impuestos ni préstamo (aquí no hay cortos). Que el minorista de la UE no pueda comprar SVXY ni otros ETPs "
        "de EE. UU. sin documento KID (PRIIPs) viene de un extracto de IBKR y está sin verificar aquí; tampoco hay un "
        "equivalente europeo verificado del −0,5x.",
        "- El alfa CAPM no es la única medida de «prima»: un modelo que valore el riesgo de caída (opciones de venta) daría "
        "otra respuesta; no se estima aquí.",
    ]))
    for k, n in TN.items():
        rep.lines = [ln.replace(f"«T:{k}»", str(n)) for ln in rep.lines]
    rep.save()
    return C


if __name__ == "__main__":
    run()
