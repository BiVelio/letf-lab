"""R33 — Los datos del artículo 7: rango diario, bonos del Tesoro sintéticos, fondos apalancados
de bonos y el índice MOVE.

Pregunta. ¿Hay datos públicos con la historia y la calidad que pide el pre-registro del artículo 7
(letf/preregistro/articulo7_indicador.md, commits aafe958 y 02faefc, con la ampliación a bonos de
«Desviaciones»)? Esta línea no contrasta ninguna hipótesis ni muestra ningún resultado del
indicador ni de ninguna cartera: solo construye, describe y valida los datos que usan las demás
(módulo letf_lab.data.indicador).

Disciplina de la muestra reservada (1-1-2019 → 25-9-2026). Todo parámetro de los datos (el
diferencial con que se cubren los huecos de FRED, el vencimiento del bono que imita a TLT e IEF y
el sobrecoste s de cada fondo apalancado) se estima con datos hasta el 31-12-2018 como mucho, y
las pruebas comprueban que cambiar cualquier dato posterior no lo mueve. Las tablas de calidad y
de disponibilidad cubren la muestra entera, como permite el pre-registro para los datos; en la
prueba reservada solo se dan medidas de seguimiento (correlación, error de seguimiento y la
diferencia de crecimiento real − sintético), nunca el nivel de rentabilidad de un activo o de un
fondo, para no enseñar el resultado de ninguna cartera antes de la evaluación única. Las
estadísticas de los estimadores de varianza y del MOVE usan solo el desarrollo.

Datos (todo en data/cache/, descargado una vez; nada se refresca):
- OHLC diario de Yahoo: ^GSPC, ^NDX, TLT e IEF (sources.daily).
- Rendimientos diarios del Tesoro a vencimiento constante de FRED: DGS5, DGS7, DGS10, DGS20 y
  DGS30 (history.fred).
- Fondos apalancados de bonos (UBT, TMF, UST, TYD; TBT, TMV, TTT y PST como referencia) y de
  acciones (SSO, UPRO, QLD, TQQQ) de Yahoo (adj_close).
- Índice MOVE: Yahoo, símbolo ^MOVE.
- Letra (history.rf_long) y financiación interbancaria (history.funding_long) del laboratorio.
"""

from __future__ import annotations

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from letf_lab.data import history, indicador as I
from letf_lab.report import FIG_W, Report, comma_fig, legend_band, num_es, pct_es
from letf_lab.research.r26_vix_data import dmy, pp

DEV_END, TEST_START, END = I.DEV_END, I.TEST_START, I.END
DEV, TEST = "desarrollo", "prueba reservada"
# columnas que se pueden mostrar en la prueba reservada: solo seguimiento, ningún nivel
TRACK_COLS = ["desde", "hasta", "sesiones", "corr_diaria", "corr_semanal", "beta_semanal",
              "TE_diario", "TE_semanal", "Δg"]

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA, VIOLET, GREY, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#9a9994", "#efeeea"
KIND_SHORT = {"largo": "bono largo", "medio": "bono medio"}
ESTIMATORS = {"pk": "Parkinson", "gk": "Garman–Klass", "rs": "Rogers–Satchell", "yz": "Yang–Zhang (diario)",
              "oc2": "apertura → cierre²", "on2": "nocturno²"}


# ───────────────────────────── utilidades ─────────────────────────────

def dev(x):
    """Tramo de desarrollo (hasta DEV_END)."""
    return x[x.index <= DEV_END]


def test(x):
    """Tramo de la prueba reservada (TEST_START → END)."""
    return x[(x.index >= TEST_START) & (x.index <= END)]


def n_valid(s: pd.Series) -> int:
    return int(s.notna().sum()) if s.dtype != bool else int(s.sum())


def first_last(s: pd.Series) -> tuple:
    s = s.dropna() if s.dtype != bool else s[s]
    return (s.index[0].date(), s.index[-1].date()) if len(s) else (None, None)


def tracking_rows(real: pd.Series, synth: pd.Series, label: dict) -> list[dict]:
    """tracking_stats en el desarrollo (completo) y en la prueba reservada (solo TRACK_COLS)."""
    rows = []
    for per, f in ((DEV, dev), (TEST, test)):
        r, s = f(real.dropna()), f(synth.dropna())
        if len(r.index.intersection(s.index)) < 30:
            continue
        st = I.tracking_stats(r, s)
        if per == TEST:
            st = {k: st[k] for k in TRACK_COLS}
        rows.append({**label, "tramo": per, **st})
    return rows


# ───────────────────────────── 1. disponibilidad ─────────────────────────────

FRED_NOTES = {"DGS5": "solo para interpolar", "DGS7": "desde 1969-07; solo para interpolar",
              "DGS10": "define los días del mercado de bonos",
              "DGS20": "hueco 1987–1993-09 (cubierto con DGS30)",
              "DGS30": "desde 1977-02; 2002-02-19 → 2006-02-08 estimado (tratado como hueco)"}


def _avail_row(serie, fuente, s, nota=""):
    a, b = first_last(s)
    return {"serie": serie, "fuente": fuente, "desde": a, "hasta": b,
            "sesiones hasta 2018": n_valid(dev(s)), "sesiones 2019 → 25-9-2026": n_valid(test(s)), "nota": nota}


def availability() -> pd.DataFrame:
    rows = []
    for t in I.OHLC_TICKERS:
        x = I.ohlc(t, end=None)
        rows.append(_avail_row(f"{t}: cierre", "Yahoo", x["close"]))
        rows.append(_avail_row(f"{t}: rango válido", "Yahoo", x["range_ok"],
                               {"^GSPC": "máximo y mínimo desde 1962; reales desde el 21-4-1982"}.get(t, "")))
        rows.append(_avail_row(f"{t}: apertura real", "Yahoo", x["open_ok"]))
    for m, s in I.FRED_TENORS.items():
        y = history.fred(s)
        rows.append(_avail_row(f"{s} ({m:g} años)", "FRED", y[y.index <= END], FRED_NOTES[s]))
    for k in ("largo", "medio"):
        M = I.maturity(k)
        rows.append(_avail_row(f"{KIND_SHORT[k].capitalize()} sintético ({num_es(M, 1 if M % 1 else 0)} años)",
                               "FRED (construido)", I.cmt_bond(M, end=None)["ret"],
                               f"imita a {I.BOND_ETF[k]}; vencimiento elegido con datos hasta 2018"))
    for t, e in I.BOND_FUNDS.items():
        rows.append(_avail_row(f"{t} ({num_es(e.leverage, 0)}x {KIND_SHORT[e.kind]})", "Yahoo (adj_close)",
                               I.real_fund(t, end=None), "referencia" if e.role == "referencia" else ""))
    for t, (name, L) in I.EQUITY_FUNDS.items():
        rows.append(_avail_row(f"{t} ({num_es(L, 0)}x {name})", "Yahoo (adj_close)", I._equity_real(t)))
    mv = I.move(end=None)["move"]
    rows.append(_avail_row("MOVE", "Yahoo (^MOVE)", mv, "solo cierre"))
    ses = I.sessions(I.BOND_START, end=None)
    rows.append(_avail_row("Letra (rf)", "NBER, FRED TB3MS, ^IRX", history.rf_long(ses), "history.rf_long"))
    rows.append(_avail_row("Financiación (f)", "FRED FEDFUNDS", history.funding_long(ses),
                           "history.funding_long (mensual, un mes de retraso)"))
    return pd.DataFrame(rows).set_index("serie")


# ───────────────────────────── 2. OHLC y varianza de rango ─────────────────────────────

FLAG_ES = {"flag_missing": "falta algún precio", "flag_open_eq_close": "apertura = cierre",
           "flag_high_eq_close": "máximo = cierre", "flag_low_eq_close": "mínimo = cierre",
           "flag_open_prev_close": "apertura = cierre anterior", "flag_no_range": "sin rango",
           "flag_inconsistent": "incoherente", "range_ok": "rango válido", "open_ok": "apertura real"}


def ohlc_segments(t: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    rs = I.RANGE_START[t]
    segs = []
    if t == "^GSPC":
        segs.append((f"1962 → {dmy(rs - pd.Timedelta(days=1))} (rango teórico)", pd.Timestamp("1962-01-02"),
                     rs - pd.Timedelta(days=1)))
    segs.append((f"{dmy(rs)} → 31-12-2018", rs, DEV_END))
    segs.append(("1-1-2019 → 25-9-2026", TEST_START, END))
    return segs


def ohlc_quality() -> pd.DataFrame:
    """Sesiones y marcas de calidad por índice o ETF y tramo (recuentos)."""
    rows = []
    for t in I.OHLC_TICKERS:
        x = I.ohlc(t, end=None)
        for lab, a, b in ohlc_segments(t):
            s = x[(x.index >= a) & (x.index <= b)]
            rows.append({"serie": t, "tramo": lab, "sesiones": len(s),
                         **{FLAG_ES[c]: int(s[c].sum()) for c in FLAG_ES}})
    return pd.DataFrame(rows).set_index(["serie", "tramo"])


def estimator_ratios() -> pd.DataFrame:
    """Desarrollo, días con rango válido: media de cada estimador ÷ media de r² (cierre a cierre),
    peso de la noche y correlación de los logaritmos de las sumas mensuales con la de r²."""
    rows = []
    for t in I.OHLC_TICKERS:
        v = I.variance_inputs(t)                          # recortado en DEV_END
        v = v[v.range_ok & v.r2.notna() & (v.index >= I.RANGE_START[t])]
        m = v.groupby(v.index.to_period("M"))
        n = m.size()
        ms = m[["r2", *ESTIMATORS]].sum()[n >= 15]
        row = {"serie": t, "desde": v.index[0].date(), "sesiones": len(v)}
        for c, lab in ESTIMATORS.items():
            row[f"{lab} ÷ r²"] = v[c].mean() / v.r2.mean()
        row["corr. mensual log YZ–log r²"] = np.log(ms.yz).corr(np.log(ms.r2))
        row["corr. mensual log PK–log r²"] = np.log(ms.pk).corr(np.log(ms.r2))
        rows.append(row)
    return pd.DataFrame(rows).set_index("serie")


def range_break() -> pd.DataFrame:
    """Media de Parkinson ÷ media de r² por año (desarrollo) en ^GSPC y ^NDX, con el rango de
    Yahoo tal cual (también el teórico de ^GSPC antes de 1982)."""
    out = {}
    for t, a in (("^GSPC", "1962-01-01"), ("^NDX", "1986-01-01")):
        x = I.ohlc(t).copy()
        ok = ~x.flag_missing & ~x.flag_no_range & ~x.flag_inconsistent & (x.index >= a)
        x["range_ok"] = ok                                 # sin descartar el rango teórico
        v = I.range_variances(x)
        ok &= v.r2.notna()
        g = v[ok].groupby(v[ok].index.year)
        out[t] = g.pk.mean() / g.r2.mean()
    t = pd.DataFrame(out)
    t.index.name = "año"
    return t


# ───────────────────────────── 3. bonos sintéticos ─────────────────────────────

def r24_long() -> pd.Series:
    """El bono largo de R24 (a la par de 25 años sobre DGS20, sin rodadura) en sesiones de NYSE."""
    from letf_lab.research.r24_folk import bond_on
    idx = I.sessions(I.BOND_START, end=None)
    return bond_on("largo", idx).rename("R24")


def maturity_grid() -> pd.DataFrame:
    t = pd.concat({I.BOND_ETF[k]: I.maturity_table(k) for k in ("largo", "medio")}, names=["ETF"])
    t["elegido"] = [M == I.maturity("largo" if e == "TLT" else "medio") for e, M in t.index]
    return t


def bond_validation() -> pd.DataFrame:
    rows = []
    for k in ("largo", "medio"):
        etf = I.BOND_ETF[k]
        real = I._etf_returns(etf)
        M = I.maturity(k)
        syn = I.cmt_bond(M, end=None)["ret"]
        rows += tracking_rows(real, syn, {"ETF": etf, "construcción": f"a la par de {num_es(M, 1 if M % 1 else 0)} "
                                                                     "años con rodadura (R33)"})
        if k == "largo":
            rows += tracking_rows(real, r24_long(), {"ETF": etf, "construcción": "a la par de 25 años sobre DGS20, "
                                                                               "sin rodadura (R24)"})
    return pd.DataFrame(rows).set_index(["ETF", "construcción", "tramo"])


def bond_yearly() -> pd.DataFrame:
    """Rentabilidad de cada año del desarrollo (2002 desde el 31-7): ETF, sintético y R24."""
    cols = {}
    for k in ("largo", "medio"):
        etf = I.BOND_ETF[k]
        real = dev(I._etf_returns(etf))
        syn = dev(I.cmt_bond(I.maturity(k))["ret"]).reindex(real.index)
        cols[etf] = real
        cols[f"sintético {etf}"] = syn
        if k == "largo":
            cols["R24 (TLT)"] = dev(r24_long()).reindex(real.index)
    df = pd.DataFrame(cols)
    y = (1 + df).groupby(df.index.year).prod() - 1
    y.index.name = "año"
    y["TLT − sintético"] = y["TLT"] - y["sintético TLT"]
    y["IEF − sintético"] = y["IEF"] - y["sintético IEF"]
    return y


def bond_decomposition() -> pd.DataFrame:
    """Desarrollo, por década: media anual (× 252) de carry, rodadura y curva, y medias de las
    magnitudes ex ante (rendimiento, financiación, carry_ann, roll_ann y prior_ann)."""
    rows = []
    for k in ("largo", "medio"):
        b = I.cmt_bond(I.maturity(k)).dropna(subset=["ret"])
        dec = (b.index.year // 10) * 10
        g = b.groupby(dec)
        t = pd.DataFrame({"carry": g.carry.mean() * 252, "rodadura": g.roll.mean() * 252,
                          "curva": g.curve.mean() * 252, "total": g.ret.mean() * 252,
                          "y": g.y.mean(), "f": g.f_ann.mean(), "carry_ann": g.carry_ann.mean(),
                          "roll_ann": g.roll_ann.mean(), "prior_ann": g.prior_ann.mean(),
                          "sesiones": g.size()})
        t.index = [f"{d}s" if d < 2010 else "2010–2018" for d in t.index]
        t.index = pd.MultiIndex.from_product([[KIND_SHORT[k]], t.index], names=["bono", "década"])
        rows.append(t)
    return pd.concat(rows)


def bond_flags() -> pd.DataFrame:
    """Sesiones de NYSE con la curva mantenida (mercado de bonos cerrado) y con un plazo cubierto."""
    rows = []
    for k in ("largo", "medio"):
        b = I.cmt_bond(I.maturity(k), end=None)
        for per, f in ((DEV, dev), (TEST, test)):
            x = f(b)
            rows.append({"bono": KIND_SHORT[k], "tramo": per, "sesiones": len(x),
                         "curva mantenida (bonos cerrados)": int(x.stale.sum()),
                         "con un plazo cubierto": int(x.filled.sum())})
    return pd.DataFrame(rows).set_index(["bono", "tramo"])


# ───────────────────────────── 4. fondos apalancados ─────────────────────────────

def _fund_row(t, L, index_es, launch, er, s, real, flags, calib_from=None, s_log=np.nan):
    return {"fondo": t, "L": L, "índice": index_es, "primer rendimiento real": launch.date(),
            "gasto (%)": 100 * er, "s calibrado (pb/año)": 1e4 * s,
            "s calibrado con": f"{dmy(calib_from or launch)} → 31-12-2018",
            "s igualando el crecimiento log (pb, sensibilidad)": 1e4 * s_log,
            "sesiones reales hasta 2018": int(dev(real).notna().sum()),
            "días sustituidos (20 pp)": len(real.attrs.get("bad_days", [])),
            "sin volumen hasta 2018": int(dev(flags).zero_volume.sum()),
            "sin volumen desde 2019": int(test(flags).zero_volume.sum()),
            "cierre repetido hasta 2018": int(dev(flags).unchanged.sum()),
            "cierre repetido desde 2019": int(test(flags).unchanged.sum()),
            "desvío > 5 pp hasta 2018": int(dev(flags).big_dev.sum()),
            "desvío > 5 pp desde 2019": int(test(flags).big_dev.sum())}


def fund_table() -> pd.DataFrame:
    from letf_lab.universe import BY_TICKER
    rows = []
    rf, f = I._rates()
    for t, e in I.BOND_FUNDS.items():
        real = I.real_fund(t, end=None)
        cal = I._fit(real)
        if e.calib_from:
            cal = cal[cal.index >= pd.Timestamp(e.calib_from)]
        sl = log_spread(I._fit(I.bond_underlying(e.kind, end=None)), I._fit(rf), I._fit(f), e.leverage, e.er, cal)
        rows.append(_fund_row(t, e.leverage, f"{KIND_SHORT[e.kind]} ({I.BOND_ETF[e.kind]})" +
                              (" · referencia" if e.role == "referencia" else ""), real.index[0], e.er,
                              I.fund_spread(t), real, I.fund_flags(t, end=None), e.calib_from, sl))
    for t, (name, L) in I.EQUITY_FUNDS.items():
        real = I._equity_real(t)
        u, r_, f_ = I._equity_base(name)
        sl = log_spread(I._fit(u), I._fit(r_), I._fit(f_), L, BY_TICKER[t].expense_ratio, I._fit(real))
        rows.append(_fund_row(t, L, name, real.index[0], BY_TICKER[t].expense_ratio, I.equity_spread(t), real,
                              I.fund_flags(t, end=None), None, sl))
    return pd.DataFrame(rows).set_index("fondo")


def s_full_life() -> dict:
    """s (pb) de TMF, TYD y TMV calibrado con toda su vida hasta 2018 (sin CALIB_DIREXION) y el Δg
    real − sintético que daría en la prueba reservada, para declarar cuánto cambia."""
    from letf_lab.research.r24_folk import calibrate_spread, lev_fund
    rf, f = I._rates()
    out = {}
    for t in ("TMF", "TYD", "TMV"):
        e = I.BOND_FUNDS[t]
        u = I.bond_underlying(e.kind, end=None)
        sf = calibrate_spread(I._fit(u), I._fit(rf), I._fit(f), e.leverage, e.er, I._fit(I.real_fund(t, end=None)))
        syn = lev_fund(u, rf.reindex(u.index), f.reindex(u.index), e.leverage, e.er, sf)
        out[t] = {"s": 1e4 * sf, "dg_test": I.tracking_stats(test(I.real_fund(t, end=None)), test(syn))["Δg"]}
    return out


def log_spread(u, rf, f, L, er, real) -> float:
    """Sensibilidad: s que iguala el crecimiento logarítmico (el valor final) del sintético al del
    fondo real en la misma ventana, en lugar de la media diaria. Una suma de log-rendimientos no se
    altera con precios viejos o un cierre erróneo que se deshace; una media aritmética, sí."""
    from scipy.optimize import brentq
    from letf_lab.research.r24_folk import lev_fund
    real = real.dropna()
    idx = real.index.intersection(u.dropna().index)
    g = np.log1p(real.reindex(idx)).sum()
    return brentq(lambda s: np.log1p(lev_fund(u.reindex(idx), rf.reindex(idx), f.reindex(idx), L, er, s)).sum() - g,
                  -0.5, 0.5)


def fund_validation() -> pd.DataFrame:
    """Real frente a sintético (mismo s) por fondo y tramo; en los fondos de bonos, también el
    sintético construido sobre el bono sintético (como se construye el tramo anterior a 2002)."""
    from letf_lab.research.r24_folk import lev_fund
    rows = []
    rf, f = I._rates()
    for t, e in I.BOND_FUNDS.items():
        real = I.real_fund(t, end=None)
        rows += tracking_rows(real, I.bond_fund(t, end=None, synthetic=True),
                              {"fondo": t, "sintético sobre": I.BOND_ETF[e.kind]})
        cmt = I.cmt_bond(I.maturity(e.kind), end=None)["ret"]
        deep = lev_fund(cmt, rf.reindex(cmt.index), f.reindex(cmt.index), e.leverage, e.er, I.fund_spread(t))
        rows += tracking_rows(real, deep, {"fondo": t, "sintético sobre": "bono sintético"})
    for t in I.EQUITY_FUNDS:
        rows += tracking_rows(I._equity_real(t), I.equity_fund(t, end=None, synthetic=True),
                              {"fondo": t, "sintético sobre": "índice (rentabilidad total)"})
    return pd.DataFrame(rows).set_index(["fondo", "sintético sobre", "tramo"])


def fund_ratio_paths() -> dict:
    """Cociente acumulado real ÷ sintético (sobre el ETF) de los fondos de decisión de bonos."""
    out = {}
    for t in ("UBT", "TMF", "UST", "TYD"):
        real = I.real_fund(t, end=None)
        syn = I.bond_fund(t, end=None, synthetic=True).reindex(real.index)
        out[t] = (np.log1p(real) - np.log1p(syn)).cumsum()
    return out


# ───────────────────────────── 5. MOVE ─────────────────────────────

def move_quality() -> pd.DataFrame:
    mv = I.move(end=None)
    ses = I.sessions("2002-11-12", end=None)
    rows = []
    for per, f in ((DEV, dev), (TEST, test)):
        x = f(mv)
        s = f(pd.Series(0, index=ses))
        row = {"tramo": per, "desde": x.index[0].date(), "hasta": x.index[-1].date(), "cierres": len(x),
               "sesiones de NYSE sin MOVE": int((~s.index.isin(x.index)).sum()),
               "días con MOVE y NYSE cerrada": int((~x.index.isin(ses)).sum()),
               "cierre repetido": int(x.repeated.sum()), "mayor salto (días naturales)": int(x.gap_days.max())}
        if per == DEV:
            row.update({"mínimo": x.move.min(), "mediana": x.move.median(), "máximo": x.move.max()})
        rows.append(row)
    return pd.DataFrame(rows).set_index("tramo")


def move_vs_realized() -> tuple[pd.DataFrame, dict]:
    """Desarrollo: MOVE (media del mes) frente a la volatilidad realizada del mismo mes de los
    cambios diarios del Tesoro a 10 años (pb × √252). Comparación contemporánea: no es una
    previsión."""
    mv = dev(I.move()["move"])
    y10 = dev(I.cmt_curve()["y10"])
    dy = (1e4 * y10.diff()).dropna()
    dy = dy[dy.index >= mv.index[0]]
    m = pd.DataFrame({"MOVE": mv.groupby(mv.index.to_period("M")).mean(),
                      "realizada 10 años": dy.groupby(dy.index.to_period("M")).std() * np.sqrt(252)}).dropna()
    st = {"meses": len(m), "corr": m.MOVE.corr(m["realizada 10 años"]),
          "corr_log": np.log(m.MOVE).corr(np.log(m["realizada 10 años"])),
          "cociente_mediano": float((m.MOVE / m["realizada 10 años"]).median())}
    return m, st


# ───────────────────────────── figuras ─────────────────────────────

def _shade_test(ax, label=True):
    ax.axvspan(TEST_START, END, color=LIGHT, zorder=0)
    if label:
        ax.text(TEST_START + pd.Timedelta(days=60), 0.98, "prueba\nreservada", transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=10, color=MUTED)


def fig_availability(rep: Report):
    T = pd.Timestamp
    g = I.gap_table()
    g20 = g[g.plazo.str.startswith("20")].iloc[0]
    g30 = g[g.plazo.str.startswith("30")].sort_values("desde")
    start = T("1960-01-01")
    real_c, syn_c, bad_c = BLUE, ORANGE, GREY
    launch = {t: I.real_fund(t, end=None).index[0] for t in ("UBT", "TMF", "UST", "TYD")}
    launch.update({t: I._equity_real(t).index[0] for t in I.EQUITY_FUNDS})
    tl0 = I._etf_returns("TLT").index[0]
    rows = [
        ("^GSPC: rango", [(T("1962-01-02"), I.RANGE_START["^GSPC"], bad_c), (I.RANGE_START["^GSPC"], END, real_c)]),
        ("^NDX: rango", [(I.RANGE_START["^NDX"], END, real_c)]),
        ("TLT e IEF: rango", [(I.RANGE_START["TLT"], END, real_c)]),
        ("Tesoro 10 años", [(T("1962-01-02"), END, real_c)]),
        ("Tesoro 20 años", [(T("1962-01-02"), T(g20.desde), real_c), (T(g20.desde), T(g20.hasta), syn_c),
                            (T(g20.hasta), END, real_c)]),
        ("Tesoro 30 años", [(T("1962-01-02"), T(g30.iloc[0].hasta), syn_c), (T(g30.iloc[0].hasta), T(g30.iloc[1].desde), real_c),
                            (T(g30.iloc[1].desde), T(g30.iloc[1].hasta), syn_c), (T(g30.iloc[1].hasta), END, real_c)]),
        ("Bono largo 1x (TLT)", [(T("1962-01-02"), tl0, syn_c), (tl0, END, real_c)]),
        ("Bono medio 1x (IEF)", [(T("1962-01-02"), tl0, syn_c), (tl0, END, real_c)]),
    ]
    for t in ("UBT", "TMF", "UST", "TYD"):
        e = I.BOND_FUNDS[t]
        rows.append((f"{t} ({num_es(e.leverage, 0)}x {KIND_SHORT[e.kind]})",
                     [(T("1962-01-02"), launch[t], syn_c), (launch[t], END, real_c)]))
    for t, (name, L) in I.EQUITY_FUNDS.items():
        a = start if name == "S&P 500" else I.NDX_FROM
        rows.append((f"{t} ({num_es(L, 0)}x {name})", [(a, launch[t], syn_c), (launch[t], END, real_c)]))
    rows.append(("MOVE", [(I.move(end=None).index[0], END, real_c)]))
    fig, ax = plt.subplots(figsize=(FIG_W, 7.4))
    for i, (lab, segs) in enumerate(rows):
        for a, b, c in segs:
            a = max(a, start)
            ax.barh(i, (b - a).days, left=a, height=0.62, color=c, edgecolor="white", linewidth=1.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(start, END + pd.Timedelta(days=200))
    _shade_test(ax, label=False)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Qué dato hay en cada tramo", fontsize=14)
    legend_band(fig, [Patch(color=real_c), Patch(color=syn_c), Patch(color=bad_c), Patch(color=LIGHT)],
                ["dato real", "sintético o hueco cubierto", "rango «teórico» (no se usa)", "prueba reservada"],
                where="bottom", ncol=2, fontsize=11)
    comma_fig(fig)
    rep.fig(fig, "disponibilidad",
            "Figura 1: disponibilidad de cada serie del artículo 7 hasta el 25-9-2026; en gris claro, la prueba "
            "reservada (1-1-2019 → 25-9-2026). Los fondos del S&P 500 son sintéticos desde 1928; el eje empieza en "
            "1960")


def fig_range_break(rb: pd.DataFrame, rep: Report):
    fig, ax = plt.subplots(figsize=(FIG_W, 4.6))
    ax.plot(rb.index, rb["^GSPC"], color=BLUE, lw=2, marker="o", ms=4, label="S&P 500 (^GSPC)")
    ax.plot(rb.index, rb["^NDX"], color=ORANGE, lw=2, marker="o", ms=4, label="Nasdaq-100 (^NDX)")
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.axvline(1982.3, color=MUTED, ls="--", lw=1)
    ax.text(1982.8, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] < 6 else 4.4,
            "21-4-1982: primer día\ncon rango real", va="top", fontsize=11, color=MUTED)
    ax.set(ylabel="Parkinson ÷ r² (media del año)", title="El rango de ^GSPC en Yahoo es «teórico» hasta 1982")
    ax.grid(color=GRID, lw=0.6)
    h, lab = ax.get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=2)
    comma_fig(fig)
    rep.fig(fig, "rango_1982",
            "Figura 2: media anual de la varianza de Parkinson dividida por la media del rendimiento de cierre a "
            "cierre al cuadrado, 1962–2018 (solo desarrollo). Con un rango real el cociente queda por debajo de 1")


def fig_bond_tracking(rep: Report):
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 6.6), sharex=True)
    r24 = r24_long()
    for ax, k in zip(axes, ("largo", "medio")):
        etf = I.BOND_ETF[k]
        real = I._etf_returns(etf)
        M = I.maturity(k)
        syn = I.cmt_bond(M, end=None)["ret"].reindex(real.index)
        c = np.exp((np.log1p(real) - np.log1p(syn)).cumsum())
        ax.plot(c.index, c.values, color=BLUE, lw=1.6)
        if k == "largo":
            c2 = np.exp((np.log1p(real) - np.log1p(r24.reindex(real.index))).cumsum())
            ax.plot(c2.index, c2.values, color=ORANGE, lw=1.4)
        ax.axhline(1, color=MUTED, lw=0.8)
        _shade_test(ax, label=False)
        ax.set_title(f"{etf} ÷ bono sintético de {num_es(M, 1 if M % 1 else 0)} años", fontsize=13)
        ax.grid(color=GRID, lw=0.6)
        ax.set_ylabel("Real ÷ sintético")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(base=4))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Valor del ETF real ÷ valor del bono sintético (1 = seguimiento perfecto)")
    from matplotlib.lines import Line2D
    legend_band(fig, [Line2D([], [], color=BLUE, lw=2), Line2D([], [], color=ORANGE, lw=2), Patch(color=LIGHT)],
                ["bono de R33 (con rodadura)", "bono de R24 (25 años, sin rodadura)", "prueba reservada"],
                where="bottom", ncol=2, fontsize=11)
    comma_fig(fig)
    rep.fig(fig, "seguimiento_bonos",
            "Figura 3: cociente acumulado entre TLT o IEF (Yahoo, con dividendos) y el bono sintético a vencimiento "
            "constante, del 31-7-2002 al 25-9-2026; es una medida de seguimiento, no de rentabilidad")


def fig_fund_tracking(paths: dict, rep: Report):
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 6.4), sharex=True)
    for ax, (t, c) in zip(axes.flat, paths.items()):
        e = I.BOND_FUNDS[t]
        w = np.exp(c)
        ax.plot(w.index, w.values, color=BLUE, lw=1.4)
        ax.axhline(1, color=MUTED, lw=0.8)
        _shade_test(ax, label=False)
        ax.set_title(f"{t} ({num_es(e.leverage, 0)}x {I.BOND_ETF[e.kind]})", fontsize=12)
        ax.grid(color=GRID, lw=0.6)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(labelsize=11)
    fig.suptitle("Fondo real ÷ sintético con el s calibrado hasta 2018")
    comma_fig(fig)
    rep.fig(fig, "seguimiento_fondos",
            "Figura 4: cociente acumulado entre cada fondo apalancado de bonos real y su sintético sobre el ETF 1x "
            "(rf + L·(u − f) − L·s − gasto), desde su lanzamiento; en gris claro, la prueba reservada, donde el "
            "cociente es un seguimiento fuera de muestra del s calibrado. La caída de TMF y TYD en 2009–2010 es el "
            "índice distinto que seguían entonces")


def fig_move(m: pd.DataFrame, rep: Report):
    fig, ax = plt.subplots(figsize=(FIG_W, 4.6))
    x = m.index.to_timestamp()
    ax.plot(x, m.MOVE, color=BLUE, lw=1.8, label="MOVE (media del mes)")
    ax.plot(x, m["realizada 10 años"], color=ORANGE, lw=1.4,
            label="volatilidad realizada del 10 años (desviación diaria × √252)")
    ax.set(ylabel="Puntos básicos anuales", title="MOVE y volatilidad realizada del Tesoro, 2002–2018")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(color=GRID, lw=0.6)
    h, lab = ax.get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=1)
    comma_fig(fig)
    rep.fig(fig, "move",
            "Figura 5: MOVE de Yahoo y volatilidad realizada del mismo mes de los cambios diarios del rendimiento a 10 "
            "años de FRED, noviembre de 2002 a diciembre de 2018 (solo desarrollo; comparación contemporánea)")


# ───────────────────────────── informe ─────────────────────────────

def compute() -> dict:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    mv, mv_st = move_vs_realized()
    return {"avail": availability(), "ohlc_q": ohlc_quality(), "est": estimator_ratios(), "rb": range_break(),
            "gaps": I.gap_table(), "grid": maturity_grid(), "bval": bond_validation(), "byear": bond_yearly(),
            "bdec": bond_decomposition(), "bflags": bond_flags(), "funds": fund_table(),
            "fval": fund_validation(), "fpaths": fund_ratio_paths(), "move_q": move_quality(),
            "move_m": mv, "move_st": mv_st, "s_full": s_full_life()}


class Tab:
    """Tablas numeradas con título (como r26.Tables) y decimales por columna: CSV sin redondear y
    markdown redondeado; los booleanos salen como «sí» o vacío."""

    def __init__(self, rep: Report):
        self.rep, self.n = rep, 0

    def __call__(self, title: str, df: pd.DataFrame, dec: dict | None = None, default: int = 2,
                 index: bool = True):
        self.n += 1
        self.rep.p(f"**Tabla {self.n}. {title}**")
        dv = df.copy()
        if isinstance(dv.index, pd.MultiIndex):
            dv, index = dv.reset_index(), False
        for c in dv.columns:
            if dv[c].dtype == bool:
                dv[c] = dv[c].map({True: "sí", False: ""})
                continue
            num = pd.to_numeric(dv[c], errors="coerce")
            if num.notna().sum() != dv[c].notna().sum() or not num.notna().any():
                continue
            d = (dec or {}).get(c)
            if d is None:
                d = 0 if np.allclose(num.dropna(), num.dropna().round()) else default
            dv[c] = num.map(lambda v, d=d: "" if pd.isna(v) else f"{v:.{d}f}")
        self.rep.lines += [dv.to_markdown(index=index), ""]
        df.to_csv(self.rep.dir / f"tabla_{self.n:02d}.csv")


TRACK_ES = {"corr_diaria": "corr. diaria", "corr_semanal": "corr. semanal", "beta_semanal": "beta semanal",
            "vol_real": "vol. real (%)", "vol_sint": "vol. sintético (%)", "TE_diario": "error de seguimiento diario (%)",
            "TE_semanal": "error de seguimiento semanal (%)", "g_real": "crecimiento real (%)",
            "g_sint": "crecimiento sintético (%)", "Δg": "Δg real − sintético (pp)"}
TRACK_DEC = {"corr. diaria": 3, "corr. semanal": 3, "beta semanal": 3, "vol. real (%)": 1, "vol. sintético (%)": 1,
             "error de seguimiento diario (%)": 2, "error de seguimiento semanal (%)": 2, "crecimiento real (%)": 2,
             "crecimiento sintético (%)": 2, "Δg real − sintético (pp)": 2}


def track_view(t: pd.DataFrame) -> pd.DataFrame:
    """Tabla de seguimiento para el informe: porcentajes en número y nombres en castellano."""
    v = t.copy()
    for c in ("vol_real", "vol_sint", "TE_diario", "TE_semanal", "g_real", "g_sint", "Δg"):
        if c in v:
            v[c] = 100 * pd.to_numeric(v[c])
    return v.rename(columns=TRACK_ES)


def run(C: dict | None = None):
    warnings.filterwarnings("ignore")
    rep = Report("r33_indicador_datos", "R33 — Los datos del artículo 7: rango diario, bonos del Tesoro sintéticos, "
                                        "fondos apalancados y el índice MOVE")
    for f in rep.dir.glob("tabla_*.csv"):
        f.unlink()
    C = C or compute()
    tab = Tab(rep)
    rb, est, oq, bv, grid, by = C["rb"], C["est"], C["ohlc_q"], C["bval"], C["grid"], C["byear"]
    funds, fv, mq, mst, sf = C["funds"], C["fval"], C["move_q"], C["move_st"], C["s_full"]
    gaps = C["gaps"]
    ML, MM = I.maturity("largo"), I.maturity("medio")
    mlab = lambda M: num_es(M, 1 if M % 1 else 0)  # noqa: E731
    pre = rb.loc[1962:1981, "^GSPC"]
    post = rb.loc[1983:2018, "^GSPC"]
    g_seg = oq.loc["^GSPC"]
    g_dev = g_seg[g_seg.index.str.startswith("21-4-1982")].iloc[0]
    n_dev = oq.loc["^NDX"].iloc[0]

    def bvr(etf, per, constr="R33"):
        s = bv.loc[etf]
        return s[s.index.get_level_values(0).str.contains(constr)].xs(per, level=1).iloc[0]

    tl_d, tl_t, if_d, if_t = bvr("TLT", DEV), bvr("TLT", TEST), bvr("IEF", DEV), bvr("IEF", TEST)
    r24_d = bvr("TLT", DEV, "R24")

    def fvr(t, over, per):
        return fv.loc[(t, over, per)]

    dec_funds = ["UBT", "TMF", "UST", "TYD"]
    dg_test = {t: fvr(t, I.BOND_ETF[I.BOND_FUNDS[t].kind], TEST)["Δg"] for t in dec_funds}
    eq_dg_test = {t: fvr(t, "índice (rentabilidad total)", TEST)["Δg"] for t in I.EQUITY_FUNDS}
    tyd_flags = I.fund_flags("TYD", end=None)
    tyd_zero = int(dev(tyd_flags).zero_volume.loc["2013":"2018"].sum())
    ief03 = by.loc[2003, "IEF − sintético"]

    vg = I.variance_inputs("^GSPC").loc["2014":"2018"]
    night_gspc = vg.on2.mean() / vg.r2.mean()

    rep.p(__doc__.split("\n", 2)[2])
    rep.h("Resumen")
    rep.p("\n".join([
        f"- **El rango del S&P 500 solo es real desde el 21-4-1982, no desde 1960.** Yahoo trae máximo y mínimo "
        f"de ^GSPC desde el 2-1-1962, pero hasta el 20-4-1982 son «teóricos»: la varianza de Parkinson vale entre "
        f"{num_es(pre.min(), 1)} y {num_es(pre.max(), 1)} veces el rendimiento al cuadrado cada año de 1962–1981 y "
        f"entre {num_es(post.min(), 2)} y {num_es(post.max(), 2)} desde 1983, y el rango diario cae de golpe del 2% a "
        f"alrededor del 1% el 21-4-1982, el día en que empezaron a cotizar los futuros del S&P 500 (figura 2). Es una "
        f"desviación del pre-registro, que preveía rango desde 1960: antes de 1982 el modelo de varianza solo tiene "
        f"cierres.",
        f"- **Muchas «aperturas» son el cierre anterior.** En ^GSPC, {int(g_dev['apertura = cierre anterior'])} de "
        f"{int(g_dev['sesiones'])} sesiones de 1982–2018 (en 1996–2005, el 95–98% de cada año) y en ^NDX "
        f"{int(n_dev['apertura = cierre anterior'])} de {int(n_dev['sesiones'])} (1985–2000) no tienen "
        f"apertura real; esos días el estimador de Yang y Zhang no tiene tramo nocturno y su parte diurna va de "
        f"cierre a cierre (tabla 2). Desde 2014 en ^GSPC y desde 2001 en ^NDX, las aperturas son reales salvo en un puñado de sesiones.",
        f"- **Bonos sintéticos a vencimiento constante desde 1962**, con carry, rodadura y cambio de la curva de "
        f"FRED y tres huecos cubiertos y declarados (tabla 4). Con datos hasta 2018, el que mejor sigue a TLT es el "
        f"bono a la par de {mlab(ML)} años y a IEF el de {mlab(MM)} años (tabla 5). En el desarrollo, la correlación "
        f"semanal es {num_es(tl_d['corr_semanal'], 3)} con TLT y {num_es(if_d['corr_semanal'], 3)} con IEF, el error "
        f"de seguimiento semanal {pct_es(tl_d['TE_semanal'], 1)} y {pct_es(if_d['TE_semanal'], 1)} al año, y el "
        f"crecimiento anual del ETF supera al del sintético en {pp(tl_d['Δg'], 2)} y {pp(if_d['Δg'], 2)} pp. Fuera "
        f"de muestra, en la prueba reservada: correlación semanal {num_es(tl_t['corr_semanal'], 3)} y "
        f"{num_es(if_t['corr_semanal'], 3)}, error de seguimiento {pct_es(tl_t['TE_semanal'], 1)} y "
        f"{pct_es(if_t['TE_semanal'], 1)} y Δg {pp(tl_t['Δg'], 2)} y {pp(if_t['Δg'], 2)} pp (tabla 6). Sigue a TLT "
        f"mejor que el bono de R24, que no envejece el bono (error semanal {pct_es(r24_d['TE_semanal'], 1)} en el "
        f"desarrollo).",
        f"- **Fondos apalancados de bonos:** el sobrecoste s, calibrado hasta 2018, es de "
        + ", ".join(f"{num_es(1e4 * I.fund_spread(t), 0)} pb en {t}" for t in dec_funds)
        + f" (tabla 10; en TMF y TYD, con datos desde 2011, porque antes seguían otro índice). Fuera de muestra, la "
          "diferencia de crecimiento anual real − sintético es de "
        + ", ".join(f"{pp(dg_test[t], 2)} pp en {t}" for t in dec_funds)
        + " (tabla 11). TYD es el más sucio: en 2013–2018 cotizó sin volumen en "
          f"{tyd_zero} sesiones y trae un cierre erróneo el 26-11-2014 (−12,7% con IEF casi plano, deshecho el "
          "1-12-2014); se marca, no se corrige.",
        f"- **Fondos de acciones (SSO, UPRO, QLD y TQQQ)** con s calibrado solo hasta 2018: "
        + ", ".join(f"{num_es(1e4 * I.equity_spread(t), 0)} pb en {t}" for t in I.EQUITY_FUNDS)
        + ". Fuera de muestra, la diferencia real − sintético es de "
        + ", ".join(f"{pp(eq_dg_test[t], 2)} pp en {t}" for t in I.EQUITY_FUNDS) + " al año. r12.make calibra s con toda la vida del fondo, prueba reservada incluida, "
          "y no debe usarse para el artículo 7.",
        f"- **MOVE solo desde el 12-11-2002** (Yahoo, ^MOVE; solo el cierre). FRED no lo publica y no hay otra "
          f"historia diaria pública descargable. Hasta 2018 faltan {int(mq.loc[DEV, 'sesiones de NYSE sin MOVE'])} "
          f"sesiones de NYSE; su media mensual se mueve con la volatilidad realizada del 10 años del mismo mes "
          f"(correlación {num_es(mst['corr'], 2)}; tabla 12 y figura 5).",
        "- **Ningún resultado del indicador ni de ninguna cartera aparece aquí.** En la prueba reservada solo se dan "
        "medidas de seguimiento (diferencias entre una serie real y su sintético), nunca el nivel de rentabilidad.",
    ]))

    rep.h("1. Fuentes y disponibilidad")
    tab(f"Series del artículo 7 hasta el {dmy(END)}: primer y último dato y sesiones con dato en el desarrollo y en "
        "la prueba reservada", C["avail"])
    fig_availability(rep)
    rep.p("Rango válido: sin precios que falten, con máximo mayor que mínimo, coherente con la apertura y el cierre y "
          "posterior a RANGE_START. Apertura real: distinta del cierre anterior (y no igual a la vez a máximo, mínimo y "
          "cierre). Los fondos cuentan desde su primer rendimiento real; antes de esa fecha, el laboratorio usa el "
          "sintético.")

    rep.h("2. OHLC diario y varianza de rango")
    rep.p("Precios de Yahoo; en TLT e IEF, apertura, máximo y mínimo se multiplican por adj_close/close, así que el "
          "rendimiento de cierre a cierre y el nocturno incluyen los cupones. Estimadores diarios, en log-rendimiento² "
          "sin anualizar, con o = ln(A/C₋₁), c = ln(C/A), u = ln(M/A) y d = ln(m/A): Parkinson ln(M/m)²/(4 ln 2); "
          "Garman–Klass ½ ln(M/m)² − (2 ln 2 − 1)·c²; Rogers–Satchell u(u − c) + d(d − c); Yang–Zhang diario "
          "o² + k·c² + (1 − k)·RS con k = 0,34/(1,34 + (n + 1)/(n − 1)) y n = 21 (k = "
          f"{num_es(I.yz_k(), 3)}). Semivarianzas de cierre a cierre: r²·1{{r < 0}} y r²·1{{r > 0}}. Los "
          "estimadores de rango solo existen los días con rango válido.")
    tab("Marcas de calidad del OHLC de Yahoo por serie y tramo (número de sesiones)", oq)
    fig_range_break(rb, rep)
    rep.p(f"Antes del 21-4-1982, la media anual de Parkinson ÷ r² de ^GSPC está entre {num_es(pre.min(), 2)} y "
          f"{num_es(pre.max(), 2)}; desde 1983, entre {num_es(post.min(), 2)} y {num_es(post.max(), 2)}, como en ^NDX. "
          "Con un rango verdadero el cociente es menor que 1: el tramo de apertura a cierre no recoge la noche y el "
          "máximo y el mínimo se observan con un muestreo discreto. Un cociente de 2 a 4 solo es posible si el máximo "
          "y el mínimo no son los del índice: es el rango «teórico», que suma los máximos y los mínimos de cada "
          "valor. El 20-4-1982 el rango aún es del 2,0% y el 21-4-1982 del 0,5% (futuros del S&P 500 en el CME desde "
          "ese día); la fecha exacta del cambio de método en la fuente original está sin verificar.")
    tab("Estimadores de varianza en el desarrollo (días con rango válido): media de cada estimador ÷ media de r² de "
        "cierre a cierre y correlación de los logaritmos de las sumas mensuales", est, default=3)
    rep.p(f"En los bonos la noche pesa: el nocturno² es el {pct_es(est.loc['TLT', 'nocturno² ÷ r²'], 0)} de r² en "
          f"TLT y el {pct_es(est.loc['IEF', 'nocturno² ÷ r²'], 0)} en IEF (el Tesoro cotiza casi 24 horas), así que "
          "solo el Yang–Zhang, que la suma, se acerca a r². En ^GSPC la noche casi no aparece "
          f"({pct_es(est.loc['^GSPC', 'nocturno² ÷ r²'], 1)} de r²) porque la mitad de las aperturas son el cierre "
          f"anterior; en 2014–2018, con todas las aperturas reales, es el {pct_es(night_gspc, 0)}. Ningún estimador "
          "se compara aquí como previsión: eso es R34.")

    rep.h("3. Bonos del Tesoro sintéticos a vencimiento constante")
    rep.p("Curva diaria de FRED (DGS5, DGS7, DGS10, DGS20 y DGS30; días del mercado de bonos = días con DGS10), "
          "interpolada linealmente entre plazos, sin extrapolar. Cada sesión de NYSE se compra al cierre anterior un "
          "bono a la par de M años (cupón semestral igual al rendimiento a M de ese día) y al cierre se valora, ya con "
          "M − Δ años de vida (Δ: días naturales/365,25), con la curva del día. La rentabilidad total se descompone en "
          "carry (cupón devengado c·Δ), rodadura (lo que sube el precio al envejecer sobre la curva de ayer) y cambio "
          "de la curva. Es la construcción de R24 (r24.par_bond_returns) con el envejecimiento del bono y la "
          "interpolación de la curva, que R24 no tenía. Las sesiones de NYSE sin mercado de bonos mantienen la curva y "
          "solo devengan carry y rodadura. Ex ante, cada día: carry_ann = y − f, roll_ann = rentabilidad a un año "
          "de rodar por la curva de hoy y prior_ann = carry_ann + roll_ann, el prior de la deriva del pre-registro.")
    gv = gaps.copy()
    gv.index = range(1, len(gv) + 1)
    gv.index.name = "hueco"
    tab("Huecos de FRED cubiertos con el plazo más cercano más el diferencial medio de 250 sesiones comunes contiguas "
        "(estimado, como todo parámetro, con datos hasta 2018)", gv, dec={"diferencial (pb)": 1})
    rep.p("El tercer hueco no lo es en FRED: del 19-2-2002 al 8-2-2006 el Tesoro no emitía el bono a 30 años y FRED "
          "publica un valor estimado cuyo diferencial con el 20 años salta unos 14 pb en los dos extremos (de −0,21 a "
          "−0,07 puntos el 19-2-2002 y de −0,08 a −0,21 el 9-2-2006). Se sustituye por el 20 años más el diferencial "
          "de antes del hueco. El 30 años anterior a 1977 se cubre con el diferencial de las 250 sesiones "
          "POSTERIORES (no hay anteriores): mira hacia delante dentro del desarrollo, en un dato de la curva, no en "
          "una señal. Antes de julio de 1969 no hay DGS7 y el bono medio interpola entre 5 y 10 años.")
    gview = track_view(grid.drop(columns=["desde", "hasta"]))
    tab("Vencimiento del bono sintético frente a TLT e IEF en una rejilla fijada de antemano, 31-7-2002 → 31-12-2018: "
        "se elige el de menor error de seguimiento semanal", gview, dec=TRACK_DEC)
    tab("Validación del bono sintético frente al ETF: desarrollo completo y, en la prueba reservada, solo el "
        "seguimiento (fuera de muestra: el vencimiento se eligió con datos hasta 2018)", track_view(bv), dec=TRACK_DEC)
    rep.p("La correlación diaria (≈ 0,96) es menor que la semanal (≈ 0,99) por la hora: los rendimientos de FRED son "
          "cotizaciones de hacia las 15:30 de Nueva York y el cierre de TLT e IEF es a las 16:00. Por eso el "
          "seguimiento se mide semana a semana.")
    yv = 100 * by
    tab("Rentabilidad de cada año del desarrollo (%): ETF, bono sintético y, para TLT, el bono de R24 (2002: desde "
        "el 31-7)", yv, default=2)
    rep.p(f"El peor año es 2003 en IEF: el ETF gana {pp(ief03, 1)} pp más que el sintético, casi todo de julio a "
          "diciembre (IEF cae menos que el bono de 8,5 años en la subida de tipos del verano) y en diciembre, con la "
          "distribución del 16-12-2003. La explicación (la duración del índice de IEF de entonces o el ajuste de esa "
          "distribución en Yahoo) está sin verificar.")
    fig_bond_tracking(rep)
    dv = C["bdec"].copy()
    for c in ("carry", "rodadura", "curva", "total", "y", "f", "carry_ann", "roll_ann", "prior_ann"):
        dv[c] = 100 * dv[c]
    tab("Descomposición del bono sintético por década en el desarrollo (%, media anual): carry, rodadura y cambio de "
        "la curva, y medias ex ante del rendimiento (y), la financiación (f), carry_ann, roll_ann y prior_ann", dv,
        default=2)
    tab("Sesiones de NYSE con la curva mantenida (mercado de bonos cerrado) y con un plazo cubierto", C["bflags"])

    rep.h("4. Fondos apalancados")
    rep.p("Sintético = rf + L·(u − f) − |L|·s − gasto (r24.lev_fund, la fórmula de r12.make con la financiación "
          "interbancaria f), con u el ETF 1x del fondo (TLT o IEF; el bono sintético antes de 2002) y s calibrado "
          "para que la media diaria del sintético iguale a la del fondo real en su vida HASTA EL 31-12-2018. Antes de "
          "su primer rendimiento real, el sintético; después, el real. Los rendimientos reales pasan la regla del "
          "laboratorio (common.pair): un día a más de 20 pp de L·u se sustituye por L·u; no hubo ninguno. Los gastos de "
          "los fondos de bonos son aproximados y sin verificar; s absorbe la diferencia.")
    rep.p(f"Excepción: TMF, TYD y TMV (Direxion, abril de 2009) no seguían al principio un índice como el de TLT o "
          f"IEF. Su beta diaria frente a 3 × TLT o 3 × IEF (en valor absoluto) fue de 3,0 a 3,6 en cada bimestre de "
          f"mayo de 2009 a octubre de 2010 y de 2,85 a 3,03 desde noviembre de 2010; se lanzaron como «30-Year» y «10-Year Treasury» (bono corriente a 30 y 10 "
          f"años; la fecha exacta del cambio de índice está sin verificar). En su primer año, TMF perdió cerca de un 10% "
          f"frente a su sintético sobre TLT (figura 4). Su s se calibra desde el {dmy(I.CALIB_DIREXION)}: con toda la vida sería de "
          f"{num_es(sf['TMF']['s'], 0)} pb en TMF, {num_es(sf['TYD']['s'], 0)} en TYD y "
          f"{num_es(sf['TMV']['s'], 0)} en TMV, en lugar de {num_es(1e4 * I.fund_spread('TMF'), 0)}, "
          f"{num_es(1e4 * I.fund_spread('TYD'), 0)} y {num_es(1e4 * I.fund_spread('TMV'), 0)}, y el TMF sintético de "
          f"1962–2009 crecería cerca de {num_es(3 * (sf['TMF']['s'] - 1e4 * I.fund_spread('TMF')) / 100, 1)} pp "
          "menos al año. El fondo real se usa igual desde su lanzamiento: es lo que se podía comprar. La decisión "
          "se toma por el índice, no por el ajuste, y no lo mejora fuera de muestra: en la prueba reservada la "
          "diferencia real − sintético es de "
          + ", ".join(f"{pp(fvr(t, I.BOND_ETF[I.BOND_FUNDS[t].kind], TEST)['Δg'], 2)} pp al año en {t} "
                      f"({pp(sf[t]['dg_test'], 2)} con el s de toda la vida)" for t in ("TMF", "TYD", "TMV")) + ".")
    rep.p("El s se calibra como en r12.make: igualando la media aritmética diaria. Con precios viejos o un cierre "
          "erróneo que se deshace, esa media sube (la suma de rendimientos simples crece con el ruido; la de "
          "log-rendimientos no), así que el s de TYD sale demasiado bajo; la tabla 10 da, como sensibilidad, el s que "
          "iguala el crecimiento logarítmico en la misma ventana. En los fondos limpios la diferencia viene de que el "
          "fondo real y su sintético no tienen exactamente la misma volatilidad diaria.")
    tab("Fondos apalancados: primer rendimiento real, gasto, s calibrado hasta 2018 y marcas de calidad del precio",
        funds, dec={"gasto (%)": 2, "s calibrado (pb/año)": 1, "s igualando el crecimiento log (pb, sensibilidad)": 1,
                    "L": 0})
    tab("Seguimiento del fondo real por su sintético con el mismo s: sobre el ETF 1x (el empalme que se usa) y, en los "
        "de bonos, sobre el bono sintético (como se construye el tramo anterior a 2002). En la prueba reservada, solo "
        "seguimiento", track_view(fv), dec=TRACK_DEC)
    fig_fund_tracking(C["fpaths"], rep)
    tb = fvr("TMF", "bono sintético", DEV)
    rep.p(
        f"- El s de los fondos de bonos 2x es negativo ({num_es(1e4 * I.fund_spread('UBT'), 0)} pb en UBT y "
        f"{num_es(1e4 * I.fund_spread('UST'), 0)} en UST): se financian por debajo de los fondos federales "
        "(probablemente con repos o futuros; sin verificar). En los inversos es positivo y grande: cobran por su "
        "posición corta menos que la financiación f.\n"
        f"- Sobre el bono sintético (la historia de antes de 2002) el TMF sintético sigue peor: error semanal "
        f"{pct_es(tb['TE_semanal'], 1)} al año en el desarrollo, frente a {pct_es(fvr('TMF', 'TLT', DEV)['TE_semanal'], 1)} "
        "sobre TLT; es el error del bono multiplicado por tres.\n"
        f"- TYD cotizó sin volumen en {int(funds.loc['TYD', 'sin volumen hasta 2018'])} sesiones hasta 2018 y su "
        f"correlación diaria con el sintético es de solo {num_es(fvr('TYD', 'IEF', DEV)['corr_diaria'], 2)} en el "
        f"desarrollo ({num_es(fvr('TYD', 'IEF', TEST)['corr_diaria'], 2)} desde 2019). El 26-11-2014 cerró a 37,90 "
        "(−12,7%, 3.200 títulos, en el mínimo del día) con IEF subiendo un 0,2%, y el 1-12-2014 volvió a 43,90 "
        "(+15,8%). fund_flags marca estos días (big_dev: desvío > 5 pp frente a L·u) y los precios viejos; no se "
        "corrigen, porque la regla del laboratorio es la de 20 pp y la mayoría de los desvíos > 5 pp son de mercado "
        "(marzo de 2020, con TLT lejos de su valor liquidativo). Quien use el TYD real en el desarrollo debe saberlo.\n"
        "- Fuera de muestra, los fondos de acciones reales crecen algo menos que su sintético (su sobrecoste efectivo "
        "fue mayor después de 2018) y casi todos los inversos de bonos algo más: es la estabilidad del s calibrado, no un resultado de ninguna cartera.")

    rep.h("5. Índice MOVE")
    rep.p(f"Fuente: {I.MOVE_SOURCE}. FRED no lo publica (es un índice con licencia de ICE) y no se ha encontrado otra "
          "historia diaria pública descargable sin registro; el índice se calcula desde 1988 (según la bibliografía; "
          "sin verificar), pero antes del 12-11-2002 no hay datos públicos. Se usa solo el cierre de Yahoo; su "
          "apertura, máximo y mínimo no se usan y el volumen es siempre cero.")
    tab("MOVE de Yahoo: cobertura y marcas (mínimo, mediana y máximo solo en el desarrollo)", mq, default=1)
    fig_move(C["move_m"], rep)
    rep.p(f"En {mst['meses']} meses del desarrollo, la correlación entre la media mensual del MOVE y la volatilidad "
          f"realizada del 10 años del mismo mes es de {num_es(mst['corr'], 2)} ({num_es(mst['corr_log'], 2)} en "
          f"logaritmos) y la mediana del cociente es {num_es(mst['cociente_mediano'], 2)}. Es una comprobación de que la "
          "serie mide lo que dice (volatilidad del Tesoro en pb), no una previsión: si el MOVE ayuda a prever la "
          "varianza del bono lo dirá R34 con H6.")

    rep.h("6. Desviaciones y decisiones que se declaran")
    rep.p("\n".join([
        "1. **Rango del S&P 500 desde el 21-4-1982**, no desde 1960 (sección 2): antes, solo cierres. El Nasdaq-100 "
        "tiene rango desde el 1-10-1985, como preveía el pre-registro.",
        "2. **Aperturas que son el cierre anterior** en ^GSPC (1982–2013; casi todas las de 1996–2005) y ^NDX (hasta "
        "2000): se marcan "
        "(open_ok) y el Yang–Zhang de esos días no tiene noche.",
        "3. **Huecos de FRED**: los tres de la tabla 4, con el valor estimado del 30 años de 2002–2006 tratado como "
        "hueco.",
        f"4. **Vencimientos {mlab(ML)} y {mlab(MM)} años**, elegidos en una rejilla fija con datos hasta 2018.",
        "5. **El 1x de cada bono es el ETF real desde el 31-7-2002** (TLT o IEF) y el sintético antes, como hace el "
        "laboratorio con los fondos (bond_underlying).",
        "6. **MOVE solo desde el 12-11-2002**: la varianza del bono no tiene volatilidad implícita antes.",
        "7. **s de todos los fondos (de bonos y de acciones) calibrado solo con datos hasta 2018.** Los fondos de "
        "acciones se construyen aquí (equity_fund, make_equity) porque r12.make usa toda la vida del fondo real.",
        "8. **TYD y otros fondos con precios viejos o desvíos**: marcados (fund_flags), no corregidos.",
        f"9. **s de TMF, TYD y TMV calibrado desde el {dmy(I.CALIB_DIREXION)}** (antes seguían a otro índice); R24 "
        "calibraba el de TMF con toda su vida.",
        f"10. **IEF en 2003**: {num_es(100 * ief03, 1)} pp por encima del sintético, sin explicar (sección 3).",
    ]))

    rep.h("7. Cómo usan estos datos las demás líneas")
    rep.p("Todas las funciones de letf_lab.data.indicador recortan por defecto en el 31-12-2018 (end=DEV_END); "
          "end=None llega al 25-9-2026 y solo debe usarse en la evaluación única de la prueba reservada. Las pruebas "
          "(tests/test_r33_indicador_datos.py) comprueban que ninguna salida por defecto pasa del 31-12-2018 y que "
          "alterar los datos posteriores (precios, curva y fondos) no cambia ningún parámetro ni ningún dato del "
          "desarrollo.")
    api = pd.DataFrame(I.API, columns=["función", "qué da"]).set_index("función")
    tab("API de letf_lab.data.indicador para las líneas del artículo 7", api)
    rep.save()
