"""R29 — Vender los ETFs inversos de renta variable (H3 del artículo 4 con SQQQ y SPXU).

Pregunta (pre-registro del artículo 4, letf/preregistro/articulo4_volatilidad.md, H3: «lo
mismo para SQQQ frente a QQQ/TQQQ»; SPXU frente a SPY/UPRO se añade con el mismo diseño).
¿Rinde más vender en corto un ETF inverso 3x de renta variable que comprar el fondo alcista
con la misma exposición? H3 dice: (a) a igual exposición y antes de préstamo, el corto supera
al fondo comprado solo por el tamaño del corto y la frecuencia de reajuste, y un corto sin
reajuste diario gana frente al diario si y solo si el cociente de varianzas del periodo es
menor que 1; (b) con el préstamo mediano (5,2%) y sin remunerar el producto del corto,
Δg ≤ 0 frente al fondo comprado.

Diseño (fijado antes de mirar resultados):
- Igual exposición: un corto del −3x por 1/3 del patrimonio es +1x en cada reajuste (frente
  a QQQ o SPY comprados); por el 100%, +3x (frente a TQQQ o UPRO comprados).
- Reajuste diario, semanal, mensual o nunca («nunca»: el corto solo se reduce cuando incumple
  el mantenimiento). Rejilla de préstamo 0 / 2 / 5,2 / 18% anual (vix.BORROW_GRID); producto
  del corto remunerado a la letra − 0,5% o sin remunerar; mantenimiento FINRA de 30% × 3 = 90%
  del valor del corto: si al cierre el patrimonio no lo cubre, el corto vuelve a su objetivo
  ese mismo cierre; ruina si el patrimonio llega a 0 (queda a cero).
- Familia del pre-registro («lo mismo para SQQQ frente a QQQ/TQQQ»): SQQQ a +1x y +3x ×
  reajuste diario, semanal o mensual (los de R27) × 4 préstamos × 2 remuneraciones = 48
  configuraciones; el veredicto de H3(b) sale de ella. Ampliación, fuera del pre-registro: SPXU
  frente a SPY/UPRO y el reajuste «nunca», 2 mercados × 2 exposiciones × 4 frecuencias × 4
  préstamos × 2 remuneraciones = 128 configuraciones, una familia (y cuatro subfamilias por
  mercado y exposición); al ser mayor, su Romano–Wolf tiene menos potencia para falsar H3(b).
- «Solo por el tamaño y la frecuencia» (H3(a), primera parte) se lee como en R27: el tamaño es
  lo que da un corto de peso w con fondos ideales (índice y gasto del folleto: los gastos, el
  haircut y la letra − f); la frecuencia, el camino. El sobrecoste de los fondos sobre su índice
  más allá del gasto del folleto (el s de R22) es fricción, que no es ni lo uno ni lo otro. Se
  mide con los ETFs reales en 2011–2026 frente a fondos ideales (friction_table); se sostiene si
  el IC90 de la fricción contiene el 0.
- Protocolo común: objetivo decidido al cierre de t, orden al cierre de t+1, 5 pb por unidad
  negociada, el capital libre cobra la letra, el préstamo se paga sobre el valor vivo del
  corto (simulador de R22, run_drift, ampliado aquí con «nunca»: drift_sim).
- Fondos: los de R22 (r22.hybrid): SQQQ y SPXU reales desde su lanzamiento (12-2-2010 y
  26-6-2009) y antes el sintético rf + L·(u − f) − |L|·s − gasto con el s calibrado con su
  vida real; TQQQ y UPRO, el 3x del laboratorio (d["x3"]); QQQ y SPY, el índice del
  laboratorio (sus rentabilidades de Yahoo en estas fechas).
- Ventanas (primer rendimiento): principal 21-12-2005 → 25-9-2026 (la del pre-registro);
  antes y después del 28-2-2018; productos reales 2011–2026 (desde el 4-1-2011). Cada
  ventana empieza de cero salvo «antes de 2018», que es el tramo inicial de la principal.
- Inferencia: Δg (crecimiento logarítmico anual del corto menos el del fondo comprado de la
  misma exposición) con bootstrap estacionario emparejado (bloque medio 63, 2.000
  remuestreos, los mismos para toda la familia), p unilateral normal con el error del
  bootstrap (H0: Δg ≤ 0), Holm, step-down de Romano y Wolf y SPA de Hansen.
- Mecanismo (enlace con R22): con x = u − f, el P&L de camino de un periodo (corto que deriva
  menos corto reajustado cada día, sin costes) es ≈ (|w| + w²)·(L²/2)·(Σx² − (Σx)²): 2·(Σx² −
  (Σx)²) a +1x y 9·(Σx² − (Σx)²) a +3x. Gana si y solo si (Σx)² < Σx², es decir, si el
  cociente de varianzas del periodo es menor que 1 (R22 lo derivó para el par corto, con
  coeficiente L²/2).
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
from letf_lab.research import r19_census as r19
from letf_lab.research import r22_inverse as r22
from letf_lab.research.r26_vix_data import Tables, dmy, pp

SEED = 29
B = 2000                         # remuestreos del bootstrap (como R13, R19 y R22)
BLOCK = 63                       # bloque medio del bootstrap estacionario (sesiones)
LAG = r13.EXEC_LAG               # señal al cierre de t, orden al cierre de t+1
COST_BPS = r13.COST_BPS          # 5 pb por unidad negociada
HAIRCUT = vix.PROCEEDS_SPREAD    # el producto del corto cobra la letra − 0,5% (o nada)
MAINT = vix.FINRA_MAINT          # 30% × |L| del valor del corto (tope 100%)
BORROW = vix.BORROW_GRID         # 0 / 2 / 5,2 / 18% anual
MEDIAN_BORROW = 0.052            # la mediana de IBKR del pre-registro (UVXY)
END = vix.END
L_INV = -3.0
SP, NDX = r22.SP, r22.NDX
LRS = r13.ORIGINAL

# mercado → (inverso −3x, 1x, 3x)
MARKETS = {NDX: ("SQQQ", "QQQ", "TQQQ"), SP: ("SPXU", "SPY", "UPRO")}
EXPOSURES = {"+1x": 1.0 / 3.0, "+3x": 1.0}           # peso del corto en el −3x
FREQS = ("D", "W", "M", "N")
FREQ_ES = {"D": "diario", "W": "semanal", "M": "mensual", "N": "nunca", "Q": "trimestral"}
PER_YEAR = {"W": 52, "M": 12}
REBATES = (True, False)
REBATE_ES = {True: "letra − 0,5%", False: "sin remunerar"}

# fechas de decisión (el primer rendimiento con posición es dos sesiones después) y ventanas
SIM_STARTS = {"principal": "2005-12-19", "2011–2026": "2010-12-31", "desde 2018": "2018-02-26"}
WINDOWS = {"principal": ("2005-12-19", END), "antes del 28-2-2018": ("2005-12-19", "2018-02-27"),
           "desde el 28-2-2018": ("2018-02-26", END), "productos reales 2011–2026": ("2010-12-31", END)}
SIM_OF = {"principal": "principal", "antes del 28-2-2018": "principal", "desde el 28-2-2018": "desde 2018",
          "productos reales 2011–2026": "2011–2026"}
WIN_LABEL = {"principal": "21-12-2005 → 25-9-2026", "antes del 28-2-2018": "21-12-2005 → 27-2-2018",
             "desde el 28-2-2018": "28-2-2018 → 25-9-2026", "productos reales 2011–2026": "4-1-2011 → 25-9-2026"}
LONG_STARTS = {SP: "1938-02-03", NDX: "1985-09-30"}   # historia larga (fuera del pre-registro)

EPISODES = {"crisis de 2008": ("2008-09-01", "2009-03-09"), "agosto de 2011": ("2011-07-22", "2011-10-03"),
            "agosto de 2015": ("2015-08-17", "2015-08-25"), "4.º trimestre de 2018": ("2018-10-01", "2018-12-24"),
            "covid 2020": ("2020-02-19", "2020-03-23"), "2022": ("2022-01-03", "2022-12-28"),
            "abril de 2025": ("2025-04-02", "2025-04-08")}

INK, MUTED = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, VIOLET, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#9a9994", "#e34948"
FREQ_COL = {"D": BLUE, "W": AQUA, "M": ORANGE, "N": VIOLET}


# ───────────────────────────── utilidades ─────────────────────────────

def bl(b: float) -> str:
    """Rótulo de un préstamo: 0%, 2%, 5,2%, 18%."""
    return pct_es(b, 0 if abs(b * 100 - round(b * 100)) < 1e-9 else 1)


def win(r: pd.Series, a, b) -> pd.Series:
    """Rendimientos de (a, b] sin huecos."""
    return r[(r.index > pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))].dropna()


def config_name(mkt: str, exp: str, freq: str, b: float, rebate: bool) -> str:
    return f"Corto de {MARKETS[mkt][0]} ({exp}), {FREQ_ES[freq]}, préstamo {bl(b)}, producto {REBATE_ES[rebate]}"


def long_name(mkt: str, exp: str) -> str:
    inv, one, three = MARKETS[mkt]
    return f"{one if exp == '+1x' else three} comprado"


def perf(r: pd.Series, rf: pd.Series, a, b) -> dict:
    """Métricas en (a, b]; una ruina deja la riqueza a cero (absorb)."""
    x = r22.absorb(win(r, a, b))
    mo = (1 + x).resample("ME").prod() - 1
    return {"CAGR": metrics.cagr(x), "vol": metrics.vol(x), "sharpe": metrics.sharpe(x, rf), "maxDD": metrics.max_dd(x),
            "peor_día": float(x.min()), "peor_mes": float(mo.min())}


# ───────────────────────────── datos ─────────────────────────────

def market_data(D: dict | None = None) -> dict:
    """Por mercado: los tres fondos (inverso −3x de R22, 1x y 3x del laboratorio) en el calendario
    del índice, la letra, el tipo de financiación f y los parámetros de R22."""
    D = D or r13.load()
    P = r22.fund_params(D)
    out = {}
    for mkt, (inv, one, three) in MARKETS.items():
        d = D[mkt]
        idx = d["u"].index[d["u"].index <= END]
        X = pd.DataFrame({inv: r22.hybrid(d, P, inv), one: d["u"], three: d["x3"]}).reindex(idx)
        out[mkt] = {"d": d, "X": X, "rf": d["rf"].reindex(idx), "f": d["f"].reindex(idx), "P": P}
    out["_D"] = D
    return out


# ───────────────────────────── simulador ─────────────────────────────

def rebalance_mask(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Sesiones de reajuste: las de r22.rebalance_days y, con «N», ninguna."""
    if freq == "N":
        return np.zeros(len(index), bool)
    return r22.rebalance_days(index, freq)


def drift_sim(target: pd.DataFrame, R: pd.DataFrame, rf: pd.Series, freq: str = "M", borrow: float | dict = 0.0,
              haircut: float = HAIRCUT, cost_bps: float = COST_BPS, lag: int = LAG, maint: dict | None = None,
              margin_rule: str = "restaurar", rebate: bool = True) -> tuple[pd.Series, dict]:
    """r22.run_drift para fondos (sin futuros ni cuenta de margen neta; el saldo negativo paga la
    letra), con una frecuencia más: «N» (nunca se reajusta por calendario; solo cuando cambia el
    objetivo o, con margin_rule = «restaurar», cuando se incumple el mantenimiento). Con D, W, M
    y Q da exactamente lo mismo que r22.run_drift (tests/test_r29_short_inverse.py).

    Por 1 $ de patrimonio: los largos consumen caja; los cortos pagan `borrow` anual sobre su valor
    vivo y su producto cobra la letra − haircut (rebate=True) o nada (rebate=False); el capital
    libre cobra la letra. Objetivo decidido al cierre de t y ejecutado al cierre de t+1 (lag = 2).
    Mantenimiento: si Σ maint·|corto| supera al patrimonio al cierre, es un incumplimiento y
    «restaurar» vuelve al objetivo ese cierre. Ruina: el día rinde −100% y empieza una cuenta nueva
    (quien evalúa una ventana aplica absorb desde su inicio)."""
    cols = list(target.columns)
    K = len(cols)
    idx = target.index
    Rv = R.reindex(idx)[cols].fillna(0.0).clip(lower=-1.0).to_numpy().tolist()
    T = target.shift(lag - 1).to_numpy(dtype=float)
    valid = ~np.isnan(T).any(axis=1)
    Tl = np.nan_to_num(T).tolist()
    rfv = rf.reindex(idx).fillna(0.0).to_numpy().tolist()
    bor = [(borrow.get(c, 0.0) if isinstance(borrow, dict) else borrow) / 252 for c in cols]
    mnt = [(maint or {}).get(c, 0.0) for c in cols]
    check = any(m > 0 for m in mnt)
    cst = cost_bps / 1e4
    reb = rebalance_mask(idx, freq).tolist()
    hc = haircut / 252
    held = [0.0] * K
    out = np.full(len(idx), np.nan)
    flag = np.zeros(len(idx), bool)
    started = parked = False
    prev = None
    n_calls = n_reb = n_ruin = 0
    days = 0
    for i in range(len(idx)):
        if not started:
            if not valid[i]:
                continue
            started = True
        days += 1
        r = Rv[i]
        longs = shorts = ret = 0.0
        for k in range(K):
            h = held[k]
            if h > 0:
                ret += h * r[k]
                longs += h
            elif h < 0:
                ret += h * r[k] + h * bor[k]
                shorts -= h
        cash = 1.0 - longs
        if rebate:
            ret += cash * rfv[i] + shorts * (rfv[i] - hc)
        else:
            ret += cash * rfv[i]
        tgt = Tl[i]
        if ret <= -1.0:
            out[i] = -1.0
            n_ruin += 1
            held, prev, parked = list(tgt), tgt, False
            continue
        g = 1.0 + ret
        held = [held[k] * (1.0 + r[k]) / g for k in range(K)]
        changed = prev is None or any(abs(a - b) > 1e-12 for a, b in zip(tgt, prev))
        prev = tgt
        do = reb[i] or changed
        if parked and not do:
            out[i] = ret
            continue
        breach = check and sum(mnt[k] * -held[k] for k in range(K) if held[k] < 0) > 1.0
        if breach:
            n_calls += 1
            flag[i] = True
            if not do:
                if margin_rule == "restaurar":
                    do = True
                elif margin_rule == "liquidar":
                    tc = sum(abs(held[k]) * cst for k in range(K))
                    ret = g * (1.0 - tc) - 1.0
                    held = [0.0] * K
                    parked = True
                    out[i] = ret
                    continue
        if do:
            parked = False
            tc = sum(abs(tgt[k] - held[k]) * cst for k in range(K))
            ret = g * (1.0 - tc) - 1.0
            held = list(tgt)
            n_reb += 1
        out[i] = ret
    years = max(days, 1) / 252
    return pd.Series(out, index=idx), {"incumplimientos_año": n_calls / years, "reajustes_año": n_reb / years,
                                       "incumplimientos": n_calls, "ruinas": n_ruin,
                                       "días_incumplimiento": pd.Series(flag, index=idx)}


def short_series(m: dict, mkt: str, w: float, freq: str, b: float, rebate: bool, start, end=END,
                 cost_bps: float = COST_BPS) -> tuple[pd.Series, dict]:
    """Corto del −3x del mercado por w del patrimonio desde la decisión del cierre de `start`."""
    inv = MARKETS[mkt][0]
    X = m["X"].loc[:end]
    live = pd.Series(np.where(X.index >= pd.Timestamp(start), 1.0, np.nan), index=X.index)
    return drift_sim(pd.DataFrame({inv: -w * live}), X[[inv]], m["rf"].loc[:end], freq, borrow=b,
                     maint={inv: min(MAINT * abs(L_INV), 1.0)}, margin_rule="restaurar", rebate=rebate,
                     cost_bps=cost_bps)


def long_series(m: dict, tk: str, start, end=END, cost_bps: float = COST_BPS) -> pd.Series:
    """Comprar y mantener el fondo tk con la misma convención de entrada (orden al cierre de t+1)."""
    X = m["X"].loc[:end]
    live = pd.Series(np.where(X.index >= pd.Timestamp(start), 1.0, np.nan), index=X.index)
    return drift_sim(pd.DataFrame({tk: live}), X[[tk]], m["rf"].loc[:end], "M", cost_bps=cost_bps)[0]


def simulate(M: dict, start, freqs=FREQS, borrows=BORROW, rebates=REBATES, exps=None,
             markets=None, end=END) -> tuple[dict, dict, dict]:
    """Todas las configuraciones desde `start`: (series, info de margen, ficha de cada una)."""
    exps = exps or EXPOSURES
    R, info, meta = {}, {}, {}
    for mkt in markets or MARKETS:
        m = M[mkt]
        inv, one, three = MARKETS[mkt]
        st = pd.Timestamp(start) if not isinstance(start, dict) else pd.Timestamp(start[mkt])
        for exp, w in exps.items():
            ln = long_name(mkt, exp)
            R[ln] = long_series(m, one if exp == "+1x" else three, st, end)
            for freq in freqs:
                for b in borrows:
                    for rb in rebates:
                        k = config_name(mkt, exp, freq, b, rb)
                        R[k], info[k] = short_series(m, mkt, w, freq, b, rb, st, end)
                        meta[k] = {"mercado": mkt, "exposición": exp, "reajuste": freq, "préstamo": b,
                                   "remunerado": rb, "referencia": ln}
    return R, info, meta


# ───────────────────────────── inferencia ─────────────────────────────

def diff_frame(pairs: dict, a, b) -> pd.DataFrame:
    """log(1 + r) − log(1 + referencia) de cada par en (a, b], con la ruina absorbida desde a y
    solo las sesiones comunes a toda la familia."""
    cols = {}
    for k, (r, bn) in pairs.items():
        j = pd.concat([win(r, a, b), win(bn, a, b)], axis=1, keys=["r", "b"]).dropna()
        cols[k] = r22.logr(r22.absorb(j.r)) - r22.logr(r22.absorb(j.b))
    return pd.DataFrame(cols).dropna()


def family_test(pairs: dict, a, b, rng, B_: int = B, spa: bool = True) -> pd.DataFrame:
    """Δg anual de cada par (regla, su referencia) con los mismos remuestreos para toda la familia:
    IC90 percentil, p unilateral normal con el error del bootstrap (H0: Δg ≤ 0), p bilateral,
    Holm, Romano–Wolf (r19.romano_wolf) y, en attrs, el SPA de Hansen de la familia."""
    X = diff_frame(pairs, a, b)
    names = list(X.columns)
    V = X.to_numpy()
    m = V.mean(axis=0)
    draws = r19.boot_means(V, B_, BLOCK, rng)
    se = draws.std(axis=0, ddof=1)
    safe = np.where(se > 0, se, np.inf)
    t = m / safe
    p = norm.sf(t)
    lo, hi = np.percentile(draws, [5, 95], axis=0)
    rw = r19.romano_wolf(t, (draws - m) / safe)
    out = pd.DataFrame({"Δg": m * 252, "IC90_bajo": lo * 252, "IC90_alto": hi * 252, "t": t, "p": p,
                        "p_bilateral": 2 * norm.sf(np.abs(t)), "p_Holm": stats.holm(p), "p_RW": rw}, index=names)
    out.attrs.update(desde=X.index[0].date(), hasta=X.index[-1].date(), sesiones=len(X))
    if spa:
        res = stats.spa_test(V, BLOCK, B_, rng)
        j = int(m.argmax())
        out.attrs.update(p_SPA=res["p_SPA_c"], mejor=names[j], Δg_mejor=float(m[j] * 252))
    return out


def ci_text(lo: float, hi: float, k: int = 1) -> str:
    return f"[{pp(lo, k)}; {pp(hi, k)}]"


# ───────────────────────────── resultados por ventana ─────────────────────────────

PREREG_FAMILY = "familia del pre-registro: SQQQ frente a QQQ/TQQQ (diario, semanal, mensual)"
P_RW_PRE = "p_RW (familia del pre-registro)"


def is_prereg(meta: dict) -> bool:
    """¿Está la configuración en la familia que nombra el pre-registro (SQQQ frente a QQQ o TQQQ,
    con los reajustes de R27: diario, semanal o mensual)? SPXU/SPY/UPRO y «nunca» son ampliaciones."""
    return meta["mercado"] == NDX and meta["reajuste"] in ("D", "W", "M")


def window_results(S: dict, M: dict, rng) -> tuple[dict, pd.DataFrame]:
    """Por ventana: la tabla de las 128 configuraciones (métricas, Δg y contrastes en la familia
    completa) y el resumen de familias (completa y por mercado y exposición)."""
    res, fams = {}, []
    for w, (a, b) in WINDOWS.items():
        R, info, meta = S[SIM_OF[w]]
        pairs = {k: (R[k], R[v["referencia"]]) for k, v in meta.items()}
        ft = family_test(pairs, a, b, rng)
        rows = {}
        for k, v in meta.items():
            mkt = v["mercado"]
            rf = M[mkt]["rf"]
            pr, pb = perf(R[k], rf, a, b), perf(R[v["referencia"]], rf, a, b)
            live = win(R[k], a, b).index
            rows[k] = {**v, "CAGR": pr["CAGR"], "CAGR_referencia": pb["CAGR"], "maxDD": pr["maxDD"],
                       "maxDD_referencia": pb["maxDD"], "peor_día": pr["peor_día"],
                       "incumplimientos_año": float(info[k]["días_incumplimiento"].reindex(live).sum() / (len(live) / 252)),
                       "ruina": (lambda x: "" if x is None else str(x.date()))(r22.ruin_date(win(R[k], a, b)))}
        # la familia del pre-registro («SQQQ frente a QQQ/TQQQ»): Nasdaq-100, reajustes diario, semanal y mensual
        pre = {k: p for k, p in pairs.items() if is_prereg(meta[k])}
        fp_ = family_test(pre, a, b, rng)
        t = pd.DataFrame(rows).T.join(ft)
        t[P_RW_PRE] = fp_["p_RW"].reindex(t.index)
        t["p_Holm (familia del pre-registro)"] = fp_["p_Holm"].reindex(t.index)
        t.attrs.update(ft.attrs)
        res[w] = t
        fams.append({"ventana": w, "familia": PREREG_FAMILY, "configuraciones": len(fp_),
                     "p_SPA": fp_.attrs["p_SPA"], "mejor": fp_.attrs["mejor"], "Δg_mejor": fp_.attrs["Δg_mejor"],
                     "rechazos_Holm_5%": int((fp_.p_Holm < 0.05).sum()), "rechazos_RW_5%": int((fp_.p_RW < 0.05).sum())})
        fams.append({"ventana": w, "familia": "las 128 configuraciones", "configuraciones": len(ft),
                     "p_SPA": ft.attrs["p_SPA"], "mejor": ft.attrs["mejor"], "Δg_mejor": ft.attrs["Δg_mejor"],
                     "rechazos_Holm_5%": int((ft.p_Holm < 0.05).sum()), "rechazos_RW_5%": int((ft.p_RW < 0.05).sum())})
        for mkt in MARKETS:
            for exp in EXPOSURES:
                sub = {k: p for k, p in pairs.items() if meta[k]["mercado"] == mkt and meta[k]["exposición"] == exp}
                fs = family_test(sub, a, b, rng)
                fams.append({"ventana": w, "familia": f"corto de {MARKETS[mkt][0]} ({exp}) frente a {long_name(mkt, exp)}",
                             "configuraciones": len(fs), "p_SPA": fs.attrs["p_SPA"], "mejor": fs.attrs["mejor"],
                             "Δg_mejor": fs.attrs["Δg_mejor"], "rechazos_Holm_5%": int((fs.p_Holm < 0.05).sum()),
                             "rechazos_RW_5%": int((fs.p_RW < 0.05).sum())})
    return res, pd.DataFrame(fams)


# ───────────────────────────── H3(a): comisiones, costes y frecuencia ─────────────────────────────

def fee_prediction(M: dict, mkt: str, exp: str, a, b) -> dict:
    """Lo que el corto diario cobra frente al fondo comprado a igual exposición, antes de préstamo y
    sin costes de negociar, según el modelo de los fondos (r22): +1x: w·(|L|·s + gasto) − w·haircut +
    (letra − f); +3x: (|L|·s + gasto) del −3x + (|L|·s + gasto) del 3x − haircut.

    Con la lectura de «solo por el tamaño y la frecuencia» común con R27, se separa en «tamaño» (lo
    que daría el corto con fondos ideales, índice y gasto del folleto: los gastos, el haircut y la
    letra − f, que escalan con el peso w) y «fricción» (el sobrecoste s de los fondos sobre su
    índice, más allá del gasto del folleto), que no es ni tamaño ni frecuencia."""
    m = M[mkt]
    d, P = m["d"], m["P"]
    inv = MARKETS[mkt][0]
    er_inv, s_inv = P[inv]["er"], abs(P[inv]["L"]) * P[inv]["s"]
    carry = float(win(m["rf"] - m["f"], a, b).mean() * 252)
    if exp == "+1x":
        w = EXPOSURES[exp]
        size, fric = w * er_inv - w * HAIRCUT + carry, w * s_inv
        return {"cobra del −3x": w * (er_inv + s_inv), "cobra del fondo largo": 0.0, "haircut": -w * HAIRCUT,
                "letra − f": carry, "tamaño": size, "fricción": fric, "total": size + fric}
    size = er_inv + d["er"] - HAIRCUT
    fric = s_inv + 3 * d["spread"]
    return {"cobra del −3x": er_inv + s_inv, "cobra del fondo largo": d["er"] + 3 * d["spread"], "haircut": -HAIRCUT,
            "letra − f": 0.0, "tamaño": size, "fricción": fric, "total": size + fric}


def friction_table(M: dict, rng) -> pd.DataFrame:
    """H3(a), primera parte, con la lectura común con R27 y fondos reales: en la ventana de productos
    reales (2011–2026), el corto diario sin préstamo y con el producto remunerado frente al fondo
    comprado de la misma exposición, con los ETFs reales (SQQQ, SPXU, TQQQ, UPRO de Yahoo; QQQ y SPY,
    el índice del laboratorio) y con fondos ideales (rf + L·(u − f) − gasto del folleto, sin
    sobrecoste s). Fricción = Δg real − Δg ideal, con IC90 y p bilateral por bootstrap estacionario
    emparejado (bloque 63); se reparte entre el −3x vendido y el fondo comprado. Mismos costes (5 pb)
    en los dos."""
    a, b = WINDOWS["productos reales 2011–2026"]
    st = SIM_STARTS["2011–2026"]
    rows = []
    for mkt in MARKETS:
        m = M[mkt]
        d, P, X = m["d"], m["P"], m["X"]
        inv, one, three = MARKETS[mkt]
        real = X.copy()
        real[three] = P[three]["real"].reindex(X.index)
        ideal = X.copy()
        ideal[inv] = r22.synthetic(d, P[inv]["L"], spread=0.0, er=P[inv]["er"]).reindex(X.index)
        ideal[three] = r22.synthetic(d, 3.0, spread=0.0, er=P[three]["er"]).reindex(X.index)
        for exp, w in EXPOSURES.items():
            tk = one if exp == "+1x" else three
            sims = {}
            for lab, XX in (("real", real), ("ideal", ideal)):
                mm = dict(m, X=XX)
                sims[lab] = (short_series(mm, mkt, w, "D", 0.0, True, st)[0], long_series(mm, tk, st))
            L = diff_frame({"sr": (sims["real"][0], sims["ideal"][0]), "lr": (sims["real"][1], sims["ideal"][1]),
                            "r": sims["real"], "i": sims["ideal"]}, a, b)
            fr = (L.r - L.i).to_numpy()[:, None]
            draws = r19.boot_means(fr, B, BLOCK, rng)[:, 0]
            se = draws.std(ddof=1)
            lo, hi = np.percentile(draws, [5, 95])
            rows.append({"mercado": mkt, "exposición": exp, "desde": L.index[0].date(), "sesiones": len(L),
                         "Δg real": L.r.mean() * 252, "Δg fondos ideales (tamaño)": L.i.mean() * 252,
                         "fricción (real − ideal)": fr.mean() * 252, "IC90": ci_text(lo * 252, hi * 252, 2),
                         "p (≠ 0)": float(2 * norm.sf(abs(fr.mean()) / se)) if se > 0 else np.nan,
                         "del −3x vendido": L.sr.mean() * 252, "del fondo comprado": -L.lr.mean() * 252,
                         "_lo": lo * 252, "_hi": hi * 252})
    return pd.DataFrame(rows).set_index(["mercado", "exposición"])


def decomposition(M: dict, S: dict, rng) -> pd.DataFrame:
    """Δg del corto frente al fondo comprado, paso a paso (ventana principal): (1) reajuste diario,
    sin préstamo, producto remunerado y sin costes de negociar (lo que cobran los fondos); (2) los
    5 pb del reajuste diario; (3) la frecuencia; (4) el préstamo del 5,2%; (5) sin remunerar el
    producto. La suma es el Δg de la configuración de H3(b)."""
    a, b = WINDOWS["principal"]
    R, _, meta = S["principal"]
    rows = []
    for mkt in MARKETS:
        m = M[mkt]
        inv, one, three = MARKETS[mkt]
        for exp, w in EXPOSURES.items():
            tk = one if exp == "+1x" else three
            ref0 = long_series(m, tk, a, cost_bps=0.0)
            s0, _ = short_series(m, mkt, w, "D", 0.0, True, a, cost_bps=0.0)
            g0 = diff_frame({"x": (s0, ref0)}, a, b)["x"].mean() * 252
            ref = R[long_name(mkt, exp)]

            def dg(freq, bb, rb):
                return diff_frame({"x": (R[config_name(mkt, exp, freq, bb, rb)], ref)}, a, b)["x"].mean() * 252

            pred = fee_prediction(M, mkt, exp, a, b)
            base_d = dg("D", 0.0, True)
            for freq in FREQS:
                f0 = dg(freq, 0.0, True)
                f1 = dg(freq, MEDIAN_BORROW, True)
                f2 = dg(freq, MEDIAN_BORROW, False)
                rows.append({"mercado": mkt, "exposición": exp, "reajuste": FREQ_ES[freq],
                             "(1) comisiones de los fondos (medido)": g0, "(1) según el modelo": pred["total"],
                             "(1a) tamaño: gastos del folleto": pred["tamaño"],
                             "(1b) fricción: sobrecoste s": pred["fricción"],
                             "(2) costes del reajuste diario": base_d - g0, "(3) frecuencia": f0 - base_d,
                             "(4) préstamo 5,2%": f1 - f0, "(5) sin remunerar": f2 - f1, "Δg total (H3b)": f2})
    return pd.DataFrame(rows).set_index(["mercado", "exposición", "reajuste"])


def fee_table(M: dict, S: dict, rng) -> pd.DataFrame:
    """H3(a), primera parte: el corto diario, sin préstamo y con el producto remunerado, frente al
    fondo comprado de la misma exposición (p bilateral: ¿es distinto de 0?), con lo que predice el
    modelo de comisiones, en cada ventana. Una familia por ventana (4 pares)."""
    rows = []
    for w, (a, b) in WINDOWS.items():
        R, _, _ = S[SIM_OF[w]]
        pairs = {(mkt, exp): (R[config_name(mkt, exp, "D", 0.0, True)], R[long_name(mkt, exp)])
                 for mkt in MARKETS for exp in EXPOSURES}
        ft = family_test({f"{k[0]}|{k[1]}": v for k, v in pairs.items()}, a, b, rng, spa=False)
        for (mkt, exp) in pairs:
            r = ft.loc[f"{mkt}|{exp}"]
            fp_ = fee_prediction(M, mkt, exp, a, b)
            rows.append({"ventana": w, "mercado": mkt, "exposición": exp, "Δg medido": r["Δg"],
                         "IC90": ci_text(r["IC90_bajo"], r["IC90_alto"], 2), "p (≠ 0)": r["p_bilateral"],
                         "p (> 0)": r["p"], "p_Holm (> 0)": stats.holm(ft["p"].to_numpy())[list(ft.index).index(f"{mkt}|{exp}")],
                         "comisiones según el modelo (sin costes de negociar)": fp_["total"],
                         "de ellas, tamaño (gastos del folleto)": fp_["tamaño"],
                         "de ellas, fricción (sobrecoste s)": fp_["fricción"]})
    return pd.DataFrame(rows).set_index(["ventana", "mercado", "exposición"])


# ───────────────────────────── H3(a): cociente de varianzas ─────────────────────────────

def excess_inputs(M: dict, mkt: str, a, b) -> tuple[pd.DataFrame, pd.Series]:
    """Rentabilidad en exceso del −3x (fondo − letra) y x = u − f en (a, b]."""
    m = M[mkt]
    inv = MARKETS[mkt][0]
    ex = pd.DataFrame({inv: m["X"][inv] - m["rf"]})
    x = (m["d"]["u"] - m["d"]["f"]).reindex(m["X"].index)
    j = pd.concat([ex, x.rename("x")], axis=1)
    j = j[(j.index > pd.Timestamp(a)) & (j.index <= pd.Timestamp(b))].dropna()
    return j[[inv]], j["x"]


def whole_window_path(ex: pd.Series, x: pd.Series, w: float, coef: float) -> dict:
    """«Nunca»: un solo periodo, la ventana entera. Corto que deriva − corto diario, sin costes."""
    drift = -w * np.expm1(np.log1p(ex.clip(lower=-0.999999)).sum())
    daily = np.expm1(np.log1p(np.maximum(-w * ex.to_numpy(), -0.999999)).sum())
    S_, Q = float(x.sum()), float((x ** 2).sum())
    return {"camino": float(drift - daily), "aprox": coef * (Q - S_ ** 2), "VR": S_ ** 2 / Q, "deriva": float(drift)}


def mechanism(M: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """P&L de camino por periodo (semana o mes naturales) del corto que deriva frente al reajustado
    cada día, sin costes ni intereses (r22.period_paths), su aproximación (|w| + w²)·(L²/2)·(Σx² −
    (Σx)²) y el cociente de varianzas; por ventana, mercado y exposición. «Coherente»: el camino
    medio es positivo si y solo si VR < 1 (VR = media de (Σx)² / media de Σx²); «periodos
    coherentes»: la fracción de periodos en que el signo del camino es el de Σx² − (Σx)².
    Aparte, «nunca»: un solo periodo, la ventana entera (por 1 $ de patrimonio inicial)."""
    rows, nrows, periods = [], [], {}
    for w_name, (a, b) in WINDOWS.items():
        for mkt in MARKETS:
            inv = MARKETS[mkt][0]
            ex, x = excess_inputs(M, mkt, a, b)
            for exp, w in EXPOSURES.items():
                coef = (w + w ** 2) * L_INV ** 2 / 2
                for fq in ("W", "M"):
                    per = r22.period_paths(ex, {inv: -w}, x, fq, coef)
                    periods[(w_name, mkt, exp, fq)] = per
                    vr = float((per.x ** 2).mean() / per.x2.mean())
                    cam = float(per.camino.mean() * PER_YEAR[fq])
                    fit = stats.ols_hac(per.camino, per.aprox.to_frame("aprox"), stats.NONOV_LAGS)
                    sgn = np.sign(per.camino) == np.sign(per.x2 - per.x ** 2)
                    rows.append({"ventana": w_name, "mercado": mkt, "exposición": exp, "reajuste": FREQ_ES[fq],
                                 "periodos": len(per), "camino (%/año)": cam,
                                 "aproximación (%/año)": float(per.aprox.mean() * PER_YEAR[fq]),
                                 "VR del periodo": vr, "coherente": "sí" if (cam > 0) == (vr < 1) else "NO",
                                 "periodos coherentes": float(sgn.mean()),
                                 "correlación": float(np.corrcoef(per.camino, per.aprox)[0, 1]),
                                 "pendiente": float(fit.beta[1])})
                wp = whole_window_path(ex[inv], x, w, coef)
                nrows.append({"ventana": w_name, "mercado": mkt, "exposición": exp, "sesiones": len(x),
                              "corto sin reajustar (bruto)": wp["deriva"], "corto diario (bruto)": wp["deriva"] - wp["camino"],
                              "camino": wp["camino"], "aproximación de 2.º orden": wp["aprox"], "VR de la ventana": wp["VR"],
                              "coherente": "sí" if (wp["camino"] > 0) == (wp["VR"] < 1) else "NO"})
    key = ["ventana", "mercado", "exposición"]
    return (pd.DataFrame(rows).set_index(key + ["reajuste"]), pd.DataFrame(nrows).set_index(key), periods)


def frequency_table(S: dict, M: dict, mech: pd.DataFrame, mech_n: pd.DataFrame, rng) -> pd.DataFrame:
    """Δg simulado (neto de costes) de cada frecuencia frente al diario, sin préstamo y con el
    producto remunerado, junto al VR de la ventana: ¿gana el que no se reajusta a diario si y solo
    si VR < 1? Familia por ventana (2 mercados × 2 exposiciones × 3 frecuencias)."""
    rows = []
    for w, (a, b) in WINDOWS.items():
        R, _, _ = S[SIM_OF[w]]
        pairs = {f"{mkt}|{exp}|{fq}": (R[config_name(mkt, exp, fq, 0.0, True)], R[config_name(mkt, exp, "D", 0.0, True)])
                 for mkt in MARKETS for exp in EXPOSURES for fq in ("W", "M", "N")}
        ft = family_test(pairs, a, b, rng)
        for k in pairs:
            mkt, exp, fq = k.split("|")
            r = ft.loc[k]
            vr = (mech_n.loc[(w, mkt, exp), "VR de la ventana"] if fq == "N"
                  else mech.loc[(w, mkt, exp, FREQ_ES[fq]), "VR del periodo"])
            rows.append({"ventana": w, "mercado": mkt, "exposición": exp, "reajuste": FREQ_ES[fq],
                         "Δg frente al diario": r["Δg"], "IC90": ci_text(r["IC90_bajo"], r["IC90_alto"]),
                         "p (≠ 0)": r["p_bilateral"], "p_RW (> 0)": r["p_RW"], "VR": vr,
                         "coherente": "sí" if (r["Δg"] > 0) == (vr < 1) else "NO"})
    return pd.DataFrame(rows).set_index(["ventana", "mercado", "exposición", "reajuste"])


def vr_table(M: dict) -> pd.DataFrame:
    """Cociente de varianzas VR(h) de x = u − f con ventanas solapadas de h sesiones en las ventanas del
    pre-registro, y contraste de z = Σx² − (Σx)² frente a 0 (Hansen-Hodrick, stats.hac_overlap), como
    r22.variance_ratio_test."""
    rows = []
    for w, (a, b) in WINDOWS.items():
        for mkt in MARKETS:
            _, x = excess_inputs(M, mkt, a, b)
            for h in (5, 21, 63):
                s1 = x.rolling(h).sum().shift(-(h - 1))
                s2 = (x ** 2).rolling(h).sum().shift(-(h - 1))
                z = (s2 - s1 ** 2).dropna()
                fit = stats.ols_hac(z, pd.DataFrame(index=z.index), **stats.hac_overlap(h))
                rows.append({"ventana": w, "mercado": mkt, "h (sesiones)": h, "ventanas": len(z),
                             "VR(h)": float((s1 ** 2).mean() / s2.mean()), "z medio (%/año)": float(z.mean() * 252 / h),
                             "t (HAC)": float(fit.t[0]), "estimador": fit.estimator})
    return pd.DataFrame(rows)


def vr_long(M: dict) -> pd.DataFrame:
    """VR(h) de la historia larga con la función de R22 (artículo 3), para enlazar con sus cifras."""
    out = []
    for mkt, start in ((SP, M[SP]["d"]["u"].first_valid_index()), (NDX, pd.Timestamp(r22.NDX_FROM))):
        t = r22.variance_ratio_test(M[mkt]["d"], mkt, start)
        out.append(t)
    return pd.concat(out, ignore_index=True)


# ───────────────────────────── préstamo que anula la ventaja ─────────────────────────────

def breakeven(res: dict) -> pd.DataFrame:
    """Préstamo anual que deja a 0 el Δg frente al fondo comprado, interpolando en la rejilla (ventana
    principal y productos reales)."""
    rows = {}
    for w in ("principal", "productos reales 2011–2026"):
        t = res[w]
        for mkt in MARKETS:
            for exp in EXPOSURES:
                for fq in FREQS:
                    for rb in REBATES:
                        g = np.array([t.loc[config_name(mkt, exp, fq, b, rb), "Δg"] for b in BORROW], float)
                        if g[0] <= 0:
                            be = "no gana ni sin préstamo"
                        elif g[-1] > 0:
                            be = f"> {bl(BORROW[-1])}"
                        else:
                            j = int(np.argmax(g <= 0))
                            be = pct_es(BORROW[j - 1] + (BORROW[j] - BORROW[j - 1]) * g[j - 1] / (g[j - 1] - g[j]), 1)
                        rows[(w, mkt, exp, FREQ_ES[fq], REBATE_ES[rb])] = {
                            **{f"Δg con {bl(b)}": v for b, v in zip(BORROW, g)}, "préstamo que la anula": be}
    out = pd.DataFrame(rows).T
    out.index.names = ["ventana", "mercado", "exposición", "reajuste", "producto del corto"]
    return out


# ───────────────────────────── episodios, historia larga y disponibilidad ─────────────────────────────

def episodes(S: dict, M: dict) -> pd.DataFrame:
    """Rentabilidad en cada episodio (ventana principal): fondo comprado y cortos de H3(b)."""
    R, _, _ = S["principal"]
    rows = []
    for ep, (a, b) in EPISODES.items():
        for mkt in MARKETS:
            for exp in EXPOSURES:
                keys = [long_name(mkt, exp)] + [config_name(mkt, exp, fq, MEDIAN_BORROW, False) for fq in FREQS]
                for k in keys:
                    r = R[k].loc[a:b].dropna()
                    w = (1 + r22.absorb(r)).cumprod()
                    rows.append({"episodio": ep, "mercado": mkt, "exposición": exp,
                                 "estrategia": k.split(", préstamo")[0], "rentabilidad": float(w.iloc[-1] - 1),
                                 "caída_máx": float((w / w.cummax() - 1).min())})
    t = pd.DataFrame(rows)
    return t.pivot_table(index=["mercado", "exposición", "estrategia"], columns="episodio", values="rentabilidad",
                         sort=False)[list(EPISODES)]


def long_history(M: dict, rng) -> pd.DataFrame:
    """Fuera del pre-registro (descriptivo): la misma comparación desde 1938 (S&P 500) y desde
    1985-10 (Nasdaq-100) con los fondos sintéticos de R22 antes de su lanzamiento; préstamo 0 y
    5,2%, producto remunerado y sin remunerar."""
    R, info, meta = simulate(M, LONG_STARTS, borrows=(0.0, MEDIAN_BORROW))
    rows = []
    for mkt in MARKETS:
        a = LONG_STARTS[mkt]
        sub = {k: (R[k], R[v["referencia"]]) for k, v in meta.items() if v["mercado"] == mkt}
        ft = family_test(sub, a, END, rng, spa=False)
        for k in sub:
            v = meta[k]
            rows.append({"mercado": mkt, "desde": ft.attrs["desde"], "exposición": v["exposición"],
                         "reajuste": FREQ_ES[v["reajuste"]], "préstamo": bl(v["préstamo"]),
                         "producto": REBATE_ES[v["remunerado"]],
                         "CAGR": perf(R[k], M[mkt]["rf"], a, END)["CAGR"],
                         "CAGR referencia": perf(R[v["referencia"]], M[mkt]["rf"], a, END)["CAGR"],
                         "Δg": ft.loc[k, "Δg"], "IC90": ci_text(ft.loc[k, "IC90_bajo"], ft.loc[k, "IC90_alto"]),
                         "ruina": (lambda x: "" if x is None else str(x.date()))(r22.ruin_date(win(R[k], a, END)))})
    return pd.DataFrame(rows).set_index(["mercado", "exposición", "reajuste", "préstamo", "producto"])


def availability(M: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Préstamo y disponibilidad en IBKR (iBorrowDesk, ~12 meses) y su relación con el precio del
    ETF: ¿sube el préstamo cuando el corto pierde?"""
    tick = ["SQQQ", "SPXU", "TQQQ", "UVXY"]
    bs = vix.borrow_summary(tick)
    bw = vix.ibkr_borrow(tick)
    px = vix.etp_prices(["SQQQ", "SPXU", "TQQQ"], "adj_close", end=None)
    rows = []
    for t in ("SQQQ", "SPXU", "TQQQ"):
        b = bw[bw.ticker == t].set_index("date")
        fee = b.fee.groupby(level=0).last()
        p = px[t].dropna()
        r21 = p.pct_change(21, fill_method=None).reindex(fee.index, method="ffill")
        j = pd.concat([fee, r21], axis=1, keys=["fee", "r21"]).dropna()
        top = j.r21 >= j.r21.quantile(0.9)
        rows.append({"ETF": t, "días": len(j), "correlación préstamo – rentabilidad del ETF a 21 sesiones": j.fee.corr(j.r21),
                     "préstamo mediano (%)": float(j.fee.median()),
                     "préstamo mediano con el ETF en su 10% de subidas (%)": float(j.fee[top].median()),
                     "días con < 10.000 acciones disponibles": int((b.available < 10_000).sum()),
                     "días con < 1.000": int((b.available < 1_000).sum())})
    return bs, pd.DataFrame(rows).set_index("ETF")


def validation(M: dict) -> pd.DataFrame:
    """Fondos sintéticos de R22 frente a los ETFs reales (r22.validation_table), solo los de R29."""
    D = M["_D"]
    t = r22.validation_table(D, M[SP]["P"])
    keep = [tk for tk in ("SQQQ", "SPXU", "TQQQ", "UPRO")]
    return t.loc[t.index.get_level_values(0).isin(keep)]


def lrs_reference(M: dict) -> pd.DataFrame:
    """La LRS 3x (original) del artículo 2 (S&P 500, r13) en las ventanas de R29."""
    d = M[SP]["d"]
    sig = r13.above_sma(r13.signal_index(d))
    lrs = r13.run_weights(d, sig)
    rows = {}
    for w, (a, b) in WINDOWS.items():
        order_day = lrs.index[lrs.index > pd.Timestamp(a)][0]      # como los cortos: posición desde ese cierre
        p = perf(lrs, d["rf"], order_day, b)
        rows[w] = {"CAGR": p["CAGR"], "maxDD": p["maxDD"], "sharpe": p["sharpe"]}
    return pd.DataFrame(rows).T


# ───────────────────────────── cálculo completo ─────────────────────────────

def compute(D: dict | None = None) -> dict:
    """Todo el cálculo de R29 (sin informe). Semillas fijas."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    M = market_data(D)
    rng = np.random.default_rng(SEED)
    S = {s: simulate(M, a) for s, a in SIM_STARTS.items()}
    res, fams = window_results(S, M, rng)
    mech, mech_n, periods = mechanism(M)
    C = {"M": M, "S": S, "res": res, "fams": fams, "mech": mech, "mech_n": mech_n, "periods": periods,
         "freq": frequency_table(S, M, mech, mech_n, np.random.default_rng(SEED + 1)),
         "fees": fee_table(M, S, np.random.default_rng(SEED + 2)),
         "friction": friction_table(M, np.random.default_rng(SEED + 4)),
         "decomp": decomposition(M, S, None), "vr": vr_table(M), "vr_long": vr_long(M),
         "breakeven": breakeven(res), "episodes": episodes(S, M),
         "long": long_history(M, np.random.default_rng(SEED + 3)),
         "avail": availability(M), "valid": validation(M), "lrs": lrs_reference(M)}
    return C


# ───────────────────────────── figuras ─────────────────────────────

def _style(ax):
    ax.grid(True, color="#e3e2de", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_borrow(res: dict, bs: pd.DataFrame, rep: Report):
    """Δg frente al fondo comprado según el préstamo (ventana principal), por mercado y exposición."""
    t = res["principal"]
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 7.6), sharex=True)
    xs = np.array(BORROW) * 100
    for i, mkt in enumerate(MARKETS):
        inv = MARKETS[mkt][0]
        for j, exp in enumerate(EXPOSURES):
            ax = axes[i, j]
            for fq in ("D", "W", "M"):
                for rb, ls in ((True, "-"), (False, "--")):
                    g = [t.loc[config_name(mkt, exp, fq, b, rb), "Δg"] * 100 for b in BORROW]
                    ax.plot(xs, g, color=FREQ_COL[fq], ls=ls, lw=2.0, marker="o", ms=4)
            ax.axhline(0, color=MUTED, lw=0.9)
            med = float(bs.loc[inv, "mediana"])
            ax.axvline(med, color=GREY, lw=1.0, ls=":")
            ax.text(med + 0.3, 0.97, f"mediana IBKR\nde {inv}: {num_es(med, 1)}%", transform=ax.get_xaxis_transform(),
                    fontsize=10, color=MUTED, va="top")
            nv = t.loc[config_name(mkt, exp, "N", 0.0, True), "Δg"] * 100
            ax.set_title(f"{inv} {exp} frente a {long_name(mkt, exp).split()[0]} (nunca: {num_es(nv, 0)} pp)", fontsize=12)
            if i == 1:
                ax.set_xlabel("Préstamo anual (%)")
            if j == 0:
                ax.set_ylabel("Δg (pp/año)")
            ax.set_xticks(xs)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: bl(v / 100).rstrip("%")))
            _style(ax)
    # dos filas: frecuencias arriba y remuneración abajo (la leyenda se llena por columnas)
    blank = plt.Line2D([], [], color="none")
    h = [plt.Line2D([], [], color=FREQ_COL["D"], lw=2.0), plt.Line2D([], [], color=INK, lw=2.0, ls="-"),
         plt.Line2D([], [], color=FREQ_COL["W"], lw=2.0), plt.Line2D([], [], color=INK, lw=2.0, ls="--"),
         plt.Line2D([], [], color=FREQ_COL["M"], lw=2.0), blank]
    lab = ["reajuste diario", "producto a la letra − 0,5%", "semanal", "producto sin remunerar", "mensual", ""]
    legend_band(fig, h, lab, where="bottom", ncol=3, fontsize=11)
    fig.suptitle("Ventaja del corto frente al fondo comprado, 2005–2026")
    comma_fig(fig)
    rep.fig(fig, "ventaja_segun_prestamo",
            "Figura: crecimiento logarítmico anual del corto del −3x menos el del fondo comprado con la misma exposición "
            "(pp/año), según el préstamo de títulos, en la ventana principal (21-12-2005 → 25-9-2026). Línea continua: el "
            "producto del corto cobra la letra − 0,5%; discontinua: no cobra nada. La línea de puntos es la mediana del "
            "préstamo en IBKR en los últimos 12 meses. «Nunca» (sin reajuste) queda fuera de escala y se indica en el título")


def fig_wealth(S: dict, rep: Report):
    R, _, _ = S["principal"]
    mkt = NDX
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 8.6), sharex=True)
    for ax, exp in zip(axes, EXPOSURES):
        ln = long_name(mkt, exp)
        items = [(ln, R[ln], INK, "-")] + [(f"corto {FREQ_ES[fq]}", R[config_name(mkt, exp, fq, MEDIAN_BORROW, False)],
                                             FREQ_COL[fq], "-") for fq in FREQS]
        for lab, r, col, ls in items:
            x = r22.absorb(win(r, *WINDOWS["principal"]))
            w = (1 + x).cumprod()
            ax.semilogy(w.index, w.values, color=col, ls=ls, lw=1.6 if col != INK else 2.0,
                        label=f"{lab}: {pct_es(metrics.cagr(x))}/año")
        ax.set(ylabel="Valor de 1 $ (log)", title=f"Nasdaq-100, exposición {exp}: SQQQ vendido frente a "
                                                  f"{ln.split()[0]} comprado")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num_es(v, 2 if v < 1 else 0)))
        ax.legend(loc="upper left", fontsize=10, frameon=False)
        _style(ax)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(base=4))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    comma_fig(fig)
    rep.fig(fig, "riqueza_nasdaq",
            "Figura: valor de 1 dólar en el corto de SQQQ (préstamo del 5,2% y producto sin remunerar, la configuración "
            "de H3b) con cada reajuste, frente a QQQ (arriba, +1x) y TQQQ (abajo, +3x) comprados; escala logarítmica. "
            "SQQQ y TQQQ son sintéticos (R22) hasta el 11-2-2010")


def fig_mechanism(periods: dict, rep: Report):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.6), sharey=False)
    for ax, mkt in zip(axes, MARKETS):
        per = periods[("principal", mkt, "+3x", "M")]
        lo = min(per.aprox.min(), per.camino.min()) * 100
        hi = max(per.aprox.max(), per.camino.max()) * 100
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=1.0)
        pre = per.index < vix.DESIGN_2018
        ax.scatter(per.aprox[pre] * 100, per.camino[pre] * 100, s=14, color=BLUE, alpha=0.75, label="antes del 28-2-2018")
        ax.scatter(per.aprox[~pre] * 100, per.camino[~pre] * 100, s=14, color=ORANGE, alpha=0.75, label="desde el 28-2-2018")
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.axvline(0, color=MUTED, lw=0.7)
        ax.set(title=f"{MARKETS[mkt][0]} +3x, mensual", xlabel="9·(Σx² − (Σx)²) (%)")
        if mkt == NDX:
            ax.set_ylabel("Camino del mes (%)")
        _style(ax)
    h, lab = axes[0].get_legend_handles_labels()
    legend_band(fig, h, lab, where="bottom", ncol=2, markerscale=2.0)
    comma_fig(fig)
    rep.fig(fig, "camino_mensual",
            "Figura: cada punto es un mes de 2006–2026. Horizontal: la aproximación 9·(Σx² − (Σx)²), con x = u − f; "
            "vertical: lo que el corto de SQQQ o SPXU del 100% reajustado a fin de mes ganó o perdió frente al mismo "
            "corto reajustado cada día (sin costes ni intereses). Gana a la derecha del 0, cuando el mercado va y "
            "vuelve dentro del mes; pierde a la izquierda, cuando tiene tendencia")


# ───────────────────────────── informe ─────────────────────────────

def peq(p: float) -> str:
    return r22.peq(p)


def y_join(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


def dg_view(t: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Columnas elegidas, Δg en puntos porcentuales e IC90 como texto."""
    df = t[[c for c in cols if c in t.columns]].copy()
    if "IC90_bajo" in t.columns:
        df["IC90 (pp)"] = [ci_text(lo, hi) if np.isfinite(lo) else "" for lo, hi in
                           zip(t.IC90_bajo.astype(float), t.IC90_alto.astype(float))]
    for c in df.columns:
        if c.startswith("Δg"):
            df[c] = df[c].astype(float) * 100
    return df.rename(columns=cols)


def run(C: dict | None = None):
    warnings.filterwarnings("ignore")
    rep = Report("r29_short_inverse", "R29 — Vender los ETFs inversos de renta variable (H3 con SQQQ y SPXU)")
    for f in rep.dir.glob("tabla_*.csv"):
        f.unlink()
    C = C or compute()
    tab = Tables(rep)
    TN = {}
    res, fams, mech, mech_n, fr = C["res"], C["fams"], C["mech"], C["mech_n"], C["freq"]
    fees, dec, be, lrs = C["fees"], C["decomp"], C["breakeven"], C["lrs"]
    bs, av = C["avail"]
    main = res["principal"]

    def cfg(mkt, exp, fq, b, rb, w="principal", col="Δg"):
        return res[w].loc[config_name(mkt, exp, fq, b, rb), col]

    # H3(b): préstamo 5,2% y producto sin remunerar
    h3b = {w: t[(t["préstamo"] == MEDIAN_BORROW) & (~t["remunerado"].astype(bool))] for w, t in res.items()}
    n_b = sum(len(t) for t in h3b.values())
    sig_b = sum(int((t.p_RW < 0.05).sum()) for t in h3b.values())
    pos_b = sum(int((t["Δg"] > 0).sum()) for t in h3b.values())
    best_b = main.loc[h3b["principal"]["Δg"].astype(float).idxmax()]
    h3b_pre = {w: t[t[P_RW_PRE].notna()] for w, t in h3b.items()}
    n_bp = sum(len(t) for t in h3b_pre.values())
    sig_bp = sum(int(((t[P_RW_PRE] < 0.05) & (t["Δg"].astype(float) > 0)).sum()) for t in h3b_pre.values())
    pos_bp = sum(int((t["Δg"].astype(float) > 0).sum()) for t in h3b_pre.values())
    best_bp = main.loc[h3b_pre["principal"]["Δg"].astype(float).idxmax()]
    fam_pre = fams[fams.familia == PREREG_FAMILY].set_index("ventana")
    frt = C["friction"]
    fr_rej = [f"{MARKETS[mk][0]} {ex}" for (mk, ex), r in frt.iterrows() if r["_lo"] > 0 or r["_hi"] < 0]
    # H3(a)
    f_main = fees.loc["principal"]
    n_coh = int((fr["coherente"] == "sí").sum())
    n_coh_path = int((mech["coherente"] == "sí").sum()) + int((mech_n["coherente"] == "sí").sum())
    n_path = len(mech) + len(mech_n)
    pc_min = float(mech["periodos coherentes"].min())
    wm = fr[fr.index.get_level_values("reajuste") != FREQ_ES["N"]]
    n_wm, n_sig_f = len(wm), int((wm["p_RW (> 0)"] < 0.05).sum())
    vr_m = C["vr"]
    vr21 = vr_m[(vr_m.ventana == "principal") & (vr_m["h (sesiones)"] == 21)].set_index("mercado")
    vrl = C["vr_long"]
    vrl21 = vrl[(vrl["época"] == "toda la muestra") & (vrl["h (sesiones)"] == 21)].set_index("mercado")
    vr3873 = float(vrl[(vrl["época"] == "1938–1973") & (vrl["h (sesiones)"] == 21)]["VR(h)"].iloc[0])
    rf_mean = float(win(C["M"][SP]["rf"], *WINDOWS["principal"]).mean() * 252)
    fam_all = fams[fams.familia == "las 128 configuraciones"].set_index("ventana")
    surv = main[main.p_RW < 0.05]
    surv_b = sorted(set(bl(x) for x in surv["préstamo"]))

    rep.p(__doc__.split("\n", 2)[2])
    rep.h("Resumen")
    rep.p("\n".join([
        f"- **H3(b) " + ("no se rechaza" if sig_bp == 0 else "se rechaza") + ".** En la familia del pre-registro (SQQQ "
        f"frente a QQQ y TQQQ, reajuste diario, semanal o mensual: 48 configuraciones), con el préstamo mediano (5,2%) y "
        f"el producto del corto sin remunerar, {'ninguna' if sig_bp == 0 else sig_bp} de las {n_bp} comparaciones (6 "
        f"configuraciones × 4 ventanas) supera al fondo comprado de la misma exposición tras la corrección por la "
        f"búsqueda (Romano–Wolf en esa familia). El Δg es positivo en {pos_bp} de ellas; la mejor en la ventana principal "
        f"es «{best_bp.name}» ({pp(best_bp['Δg'])} pp/año, IC90 {ci_text(best_bp.IC90_bajo, best_bp.IC90_alto)}, p_RW "
        f"{peq(best_bp[P_RW_PRE])}). En la ampliación a 128 (con SPXU y «nunca»), {sig_b} rechazos en {n_b} "
        f"comparaciones; la mejor es "
        + ("la misma" if best_b.name == best_bp.name else f"«{best_b.name}» ({pp(best_b['Δg'])} pp, IC90 "
           f"{ci_text(best_b.IC90_bajo, best_b.IC90_alto)})") + ".",
        f"- **H3(a), primera parte, {'no se sostiene tal como está escrita' if fr_rej else 'se sostiene'}** (lectura común "
        f"con R27: el tamaño es lo que da "
        f"el corto con fondos ideales, índice y gasto del folleto; la fricción de los fondos no es ni tamaño ni "
        f"frecuencia). A igual exposición y con reajuste diario, sin préstamo y con el producto remunerado, el corto "
        f"supera al fondo comprado: a +3x, {pp(f_main.loc[(NDX, '+3x'), 'Δg medido'])} pp/año frente a TQQQ (IC90 "
        f"{f_main.loc[(NDX, '+3x'), 'IC90']}) y {pp(f_main.loc[(SP, '+3x'), 'Δg medido'])} pp frente a UPRO "
        f"({f_main.loc[(SP, '+3x'), 'IC90']}) en la ventana principal. De lo que predice el modelo de los fondos "
        f"({pp(dec.loc[(NDX, '+3x', 'diario'), '(1) según el modelo'])} y {pp(dec.loc[(SP, '+3x', 'diario'), '(1) según el modelo'])} "
        f"pp antes de costes), el tamaño (gastos del folleto y haircut) es {pp(dec.loc[(NDX, '+3x', 'diario'), '(1a) tamaño: gastos del folleto'])} "
        f"y {pp(dec.loc[(SP, '+3x', 'diario'), '(1a) tamaño: gastos del folleto'])} pp y el resto, "
        f"{pp(dec.loc[(NDX, '+3x', 'diario'), '(1b) fricción: sobrecoste s'])} y {pp(dec.loc[(SP, '+3x', 'diario'), '(1b) fricción: sobrecoste s'])} "
        f"pp, es el sobrecoste de los fondos (tabla «T:decomp»). Con los ETFs reales en 2011–2026, la fricción medida "
        f"(real − fondos ideales) es {pp(frt.loc[(NDX, '+3x'), 'fricción (real − ideal)'])} pp a +3x en el Nasdaq-100 (IC90 "
        f"{frt.loc[(NDX, '+3x'), 'IC90']}) y {pp(frt.loc[(SP, '+3x'), 'fricción (real − ideal)'])} en el S&P 500 "
        f"({frt.loc[(SP, '+3x'), 'IC90']}); su IC90 excluye el 0 en "
        + (y_join(fr_rej) if fr_rej else "ninguna combinación") + f" (tabla «T:fric»). A +1x la ventaja es de décimas "
        f"({pp(f_main.loc[(NDX, '+1x'), 'Δg medido'], 2)} pp frente a QQQ y {pp(f_main.loc[(SP, '+1x'), 'Δg medido'], 2)} "
        f"frente a SPY).",
        f"- **H3(a), segunda parte, se sostiene.** El corto que no se reajusta a diario gana al diario si y solo si el "
        f"cociente de varianzas del periodo es menor que 1 en las {n_coh} combinaciones de ventana, mercado, exposición y "
        f"frecuencia (Δg simulado y neto, tabla «T:freq») y en las {n_coh_path} de {n_path} del camino bruto (tablas «T:mech» y «T:mechn»); "
        f"mes a mes o semana a semana el signo coincide en al menos el {pct_es(pc_min, 0)} de los periodos. Que el signo "
        f"coincida no quiere decir que la diferencia sea segura: la ventaja del semanal o el mensual sobre el diario "
        f"solo pasa Romano–Wolf en {n_sig_f} de {n_wm} casos. En "
        f"2005–2026 los dos índices revirtieron: VR(21) = {num_es(vr21.loc[NDX, 'VR(h)'], 2)} en el Nasdaq-100 y "
        f"{num_es(vr21.loc[SP, 'VR(h)'], 2)} en el S&P 500 (t = {num_es(vr21.loc[NDX, 't (HAC)'], 1)} y "
        f"{num_es(vr21.loc[SP, 't (HAC)'], 1)}); en toda la historia del S&P 500 desde 1928, "
        f"{num_es(vrl21.loc[SP, 'VR(h)'], 2)} (la cifra de R22). Sin reajuste nunca, el VR de la ventana es 5–12 y el "
        f"corto pierde entre {num_es(-cfg(SP, '+1x', 'N', 0.0, True) * 100, 0)} y "
        f"{num_es(-cfg(NDX, '+3x', 'N', 0.0, True) * 100, 0)} pp al año: el −3x se hunde, el corto se queda sin "
        f"exposición y el dinero espera en letras.",
        f"- **Las familias y la búsqueda.** En la del pre-registro (48), el SPA da p {peq(fam_pre.loc['principal', 'p_SPA'])} "
        f"en la principal y Romano–Wolf deja {int(fam_pre.loc['principal', 'rechazos_RW_5%'])} supervivientes. Con las 128 "
        f"configuraciones, el SPA rechaza en {int((fam_all['p_SPA'] < 0.05).sum())} de las {len(fam_all)} ventanas (p "
        f"{peq(fam_all.loc['principal', 'p_SPA'])} en la principal) y Romano–Wolf deja "
        f"{int(fam_all.loc['principal', 'rechazos_RW_5%'])} supervivientes en la principal, todas con un préstamo de "
        f"{' o '.join(surv_b)}: el corto diario a +3x (las comisiones) y los semanales o mensuales sin préstamo (el "
        f"cociente de varianzas de 2005–2026). Con el 5,2% o el 18% no sobrevive ninguna.",
        f"- **El préstamo lo decide.** En IBKR (iBorrowDesk, 12 meses) SQQQ costó una mediana de "
        f"{num_es(bs.loc['SQQQ', 'mediana'], 1)}% (p90 {num_es(bs.loc['SQQQ', 'p90'], 1)}%) y SPXU "
        f"{num_es(bs.loc['SPXU', 'mediana'], 1)}% (p90 {num_es(bs.loc['SPXU', 'p90'], 1)}%). El corto mensual de SQQQ a +1x "
        f"sin remunerar se anula con un {be.loc[('principal', NDX, '+1x', 'mensual', 'sin remunerar'), 'préstamo que la anula']} "
        f"de préstamo y el diario a +3x con un {be.loc[('principal', NDX, '+3x', 'diario', 'sin remunerar'), 'préstamo que la anula']} "
        f"(tabla «T:be»). Hubo días con 1 acción de SQQQ y 100 de SPXU disponibles, y el préstamo de SPXU sube cuando el ETF "
        f"sube (cuando el corto pierde).",
        f"- **No es otra cosa que la misma exposición.** En las caídas el corto pierde lo mismo que el fondo comprado "
        f"(tabla «T:ep»): es +1x o +3x a la bolsa por otro camino. Como referencia, en la ventana principal la LRS 3x (original) "
        f"del artículo 2 creció {pct_es(lrs.loc['principal', 'CAGR'])} al año, QQQ {pct_es(main.loc[config_name(NDX, '+1x', 'D', 0.0, True), 'CAGR_referencia'])} "
        f"y SPY {pct_es(main.loc[config_name(SP, '+1x', 'D', 0.0, True), 'CAGR_referencia'])}.",
    ]))

    rep.h("1. Fondos, ventanas y referencias")
    rep.p("SQQQ y SPXU son los de R22: reales desde su lanzamiento y, antes, el sintético rf + L·(u − f) − |L|·s − gasto "
          "con el s calibrado con la vida del ETF. TQQQ y UPRO son el 3x del laboratorio (sintético antes del "
          "lanzamiento); QQQ y SPY, el índice del laboratorio, que en estas fechas es su rentabilidad total de Yahoo. "
          "Ventana principal: 21-12-2005 → 25-9-2026 (la del pre-registro); SQQQ y TQQQ son sintéticos sus primeros "
          "4,1 años y SPXU y UPRO los primeros 3,5. La validación fuera de muestra de R22 (s de 2010–2017 evaluado en "
          "2018–2026) es la de la tabla «T:valid».")
    tab("Fondos sintéticos de R22 frente a los ETFs reales (r22.validation_table)", C["valid"],
        pct={"CAGR_real", "CAGR_sintético", "caída_máx_real", "caída_máx_sintético", "gasto", "s",
             "diferencia_media_anual", "error_de_seguimiento_anual"}, digits=3)
    TN["valid"] = tab.n
    tab("LRS 3x (original) del artículo 2 (S&P 500, r13) en las ventanas de R29", lrs.rename(index=WIN_LABEL | {}),
        digits=2)

    rep.h("2. H3(b): préstamo mediano y producto sin remunerar")
    rep.p("H3(b) dice que con el préstamo mediano (5,2%) y sin remunerar el producto del corto, Δg ≤ 0 frente al "
          "fondo comprado de la misma exposición. Se contrasta H0: Δg ≤ 0 en cada configuración; H3(b) queda rechazada "
          "si alguna la rechaza tras Romano–Wolf en la familia del pre-registro (SQQQ frente a QQQ y TQQQ, reajuste "
          "diario, semanal o mensual: 48 configuraciones). La familia de 128 (con SPXU y «nunca») es una ampliación: al "
          "ser mayor, su Romano–Wolf pide más para rechazar, y se da al lado.")
    cols = {"CAGR": "CAGR", "CAGR_referencia": "CAGR del fondo comprado", "Δg": "Δg (pp)", "p": "p",
            "p_Holm": "p_Holm", "p_RW": "p_RW (familia de 128)", P_RW_PRE: P_RW_PRE,
            "incumplimientos_año": "incumplimientos/año", "ruina": "ruina"}
    v = dg_view(h3b["principal"], cols)
    v.index = [f"{MARKETS[h3b['principal'].loc[k, 'mercado']][0]} {h3b['principal'].loc[k, 'exposición']}, "
               f"{FREQ_ES[h3b['principal'].loc[k, 'reajuste']]}" for k in v.index]
    tab("Ventana principal (21-12-2005 → 25-9-2026): corto con préstamo del 5,2% y producto sin remunerar frente al "
        "fondo comprado de la misma exposición; p_Holm y p_RW en la familia de 128 y p_RW en la del pre-registro (48; "
        "vacío fuera de ella)", v, pct={"CAGR", "CAGR del fondo comprado"}, digits=3)
    piv = []
    for w, t in h3b.items():
        for k, r in t.iterrows():
            piv.append({"configuración": f"{MARKETS[r['mercado']][0]} {r['exposición']}, {FREQ_ES[r['reajuste']]}",
                        "ventana": WIN_LABEL[w], "Δg (pp)": 100 * r["Δg"], "p_RW (128)": r["p_RW"],
                        "p_RW (pre-registro)": r[P_RW_PRE]})
    piv = pd.DataFrame(piv)
    pv = piv.pivot_table(index="configuración", columns="ventana", values=["Δg (pp)", "p_RW (pre-registro)", "p_RW (128)"],
                         sort=False, dropna=False)
    pv.columns = [f"{a} · {b}" for a, b in pv.columns]
    tab("H3(b) por ventana: Δg (pp/año) y p_RW en la familia del pre-registro (48; vacío fuera de ella) y en la de 128",
        pv, digits=2)
    rep.p(f"**Lectura.** H3(b) {'no se rechaza en ninguna ventana' if sig_bp == 0 else 'se rechaza'} en la familia del "
          f"pre-registro (p_RW mínimo de sus configuraciones de H3(b) "
          f"{peq(min(float(t[P_RW_PRE].min()) for t in h3b_pre.values()))}), ni en la de 128. El reajuste diario pierde siempre (paga el préstamo "
          "sobre todo el corto y no cobra la letra sobre el producto); «nunca» pierde mucho (sección 4); los semanales y "
          "mensuales quedan cerca de 0 con intervalos anchos. Los CAGR del corto y del fondo son los de la ventana; una "
          "ruina deja la riqueza a cero hasta el final. «Incumplimientos/año» cuenta los cierres en que, antes de "
          "reajustar, el patrimonio no cubría el 90% del valor del corto (a +3x con reajuste diario pasa cada vez que el "
          "−3x sube más de un 5,3% en un día; el corto vuelve a su objetivo ese mismo cierre).")

    rep.h("3. H3(a), primera parte: igual exposición y reajuste diario")
    rep.p("Con reajuste diario, el corto del −3x por w del patrimonio tiene exactamente la exposición 3w al índice "
          "cada día, la misma que el fondo comprado, así que no hay efecto de la frecuencia. «Solo por el tamaño y la "
          "frecuencia» se lee como en R27: el tamaño es lo que daría el corto de peso w con fondos ideales (índice y gasto "
          "del folleto): los gastos que cobra o se ahorra, el haircut y la letra − f; lo que los fondos rinden de menos que "
          "su índice más allá del gasto del folleto (el sobrecoste s de R22) es fricción, que no es ni tamaño ni "
          "frecuencia. Según el modelo de los fondos de R22, el corto gana:")
    rep.p("- a +1x (w = 1/3, frente a QQQ o SPY): (|L|·s + gasto)/3 del −3x − haircut/3 + (letra − f);\n"
          "- a +3x (w = 1, frente a TQQQ o UPRO): (|L|·s + gasto) del −3x + (|L|·s + gasto) del 3x − haircut.")
    rep.p("*Pie de fórmula: quien vende el −3x cobra, en lugar de pagarlos, los gastos y sobrecostes de ese fondo, y a +3x "
          "además se ahorra los del 3x que no compra; solo cede el diferencial del producto del corto. Los términos con "
          "gasto, haircut y letra − f son el tamaño; los términos con s, la fricción.* Notación: L = −3, el "
          "apalancamiento del fondo; s, su sobrecoste anual calibrado (R22); gasto, su comisión anual; haircut, lo que el "
          "bróker descuenta a la letra sobre el producto del corto (0,5%); f, el tipo al que se financia la exposición.")
    fv = fees.copy()
    for c in ("Δg medido", "comisiones según el modelo (sin costes de negociar)", "de ellas, tamaño (gastos del folleto)",
              "de ellas, fricción (sobrecoste s)"):
        fv[c] = fv[c] * 100
    fv = fv.rename(columns={"Δg medido": "Δg medido (pp)", "IC90": "IC90 (pp)",
                            "comisiones según el modelo (sin costes de negociar)": "modelo, sin costes de negociar (pp)",
                            "de ellas, tamaño (gastos del folleto)": "de ellas, tamaño (pp)",
                            "de ellas, fricción (sobrecoste s)": "de ellas, fricción (pp)"})
    fv.index = fv.index.set_levels([[WIN_LABEL[w] for w in fv.index.levels[0]], fv.index.levels[1], fv.index.levels[2]])
    tab("Corto diario sin préstamo y con el producto remunerado frente al fondo comprado de la misma exposición, por "
        "ventana; p (≠ 0) bilateral y p_Holm unilateral entre las cuatro comparaciones de cada ventana", fv, digits=3)
    dv = dec.copy() * 100
    tab("Descomposición del Δg (pp/año) del corto frente al fondo comprado, ventana principal: (1) lo que cobran los "
        "fondos (reajuste diario, sin préstamo, producto remunerado y sin costes de negociar), medido y según el modelo; "
        "(2) los 5 pb del reajuste diario; (3) la frecuencia de reajuste; (4) un préstamo del 5,2%; (5) no remunerar el "
        "producto. (1) medido + (2) + (3) + (4) + (5) = Δg de la configuración de H3(b). El modelo (1) se reparte en "
        "(1a), el tamaño (gastos del folleto, haircut y letra − f), y (1b), la fricción (sobrecoste s de los fondos)",
        dv, digits=2)
    TN["decomp"] = tab.n
    fv2 = frt.drop(columns=["_lo", "_hi"]).copy()
    for c in ("Δg real", "Δg fondos ideales (tamaño)", "fricción (real − ideal)", "del −3x vendido", "del fondo comprado"):
        fv2[c] = fv2[c] * 100
    fv2 = fv2.rename(columns={c: f"{c} (pp)" for c in ("Δg real", "Δg fondos ideales (tamaño)", "fricción (real − ideal)",
                                                         "del −3x vendido", "del fondo comprado")} | {"IC90": "IC90 (pp)"})
    tab("La fricción medida con los ETFs reales (SQQQ, SPXU, TQQQ y UPRO de Yahoo; QQQ y SPY, el índice del "
        "laboratorio), 2011–2026: corto diario sin préstamo y producto remunerado frente al fondo comprado, con los "
        "fondos reales y con fondos ideales (índice y gasto del folleto, sin sobrecoste); fricción = real − ideal, con "
        "IC90 y p bilateral por bootstrap estacionario emparejado (bloque 63), y su reparto entre el −3x vendido y el "
        "fondo comprado. Mismos costes de negociar en los dos", fv2, digits=3)
    TN["fric"] = tab.n
    rep.p(f"**Lectura.** A +3x la ventaja del corto diario es grande y casi determinista: "
          f"{pp(dec.loc[(NDX, '+3x', 'diario'), '(1) comisiones de los fondos (medido)'])} pp/año en el Nasdaq-100 y "
          f"{pp(dec.loc[(SP, '+3x', 'diario'), '(1) comisiones de los fondos (medido)'])} en el S&P 500 antes de costes, "
          "lo que predice el modelo de los fondos con una diferencia de décimas. De eso, el tamaño (gastos del folleto y "
          f"haircut) es {pp(dec.loc[(NDX, '+3x', 'diario'), '(1a) tamaño: gastos del folleto'])} y "
          f"{pp(dec.loc[(SP, '+3x', 'diario'), '(1a) tamaño: gastos del folleto'])} pp; el resto es el sobrecoste s de los "
          "fondos, lo mismo que R22 encontró en el lever-arb con cobertura diaria y en la «LRS con corto del −3x» "
          "(artículo 3): el corto cobra lo que el fondo cobra por encima de su réplica. Con los ETFs reales de "
          f"2011–2026 (tabla «T:fric»), la fricción es {pp(frt.loc[(NDX, '+3x'), 'fricción (real − ideal)'])} pp "
          f"(IC90 {frt.loc[(NDX, '+3x'), 'IC90']}) a +3x en el Nasdaq-100, {pp(frt.loc[(SP, '+3x'), 'fricción (real − ideal)'])} "
          f"({frt.loc[(SP, '+3x'), 'IC90']}) en el S&P 500, {pp(frt.loc[(NDX, '+1x'), 'fricción (real − ideal)'])} "
          f"({frt.loc[(NDX, '+1x'), 'IC90']}) a +1x en el Nasdaq-100 y {pp(frt.loc[(SP, '+1x'), 'fricción (real − ideal)'])} "
          f"({frt.loc[(SP, '+1x'), 'IC90']}) en el S&P 500. Con la lectura de R27, la primera parte de H3(a) («solo por "
          "el tamaño y la frecuencia») " + (f"no se sostiene: la fricción de los fondos es distinta de 0 en "
          f"{y_join(fr_rej)}. " if fr_rej else "se sostiene: la fricción de los fondos no es distinta de 0. ")
          + "La ventaja cabe en el préstamo: con un 5,2% el diario pierde en las cuatro combinaciones.")

    rep.h("4. H3(a), segunda parte: el cociente de varianzas")
    rep.p("El corto que no se reajusta cada día deriva: si el −3x sube (el mercado cae), el corto crece en proporción al "
          "patrimonio; si baja, encoge. Frente al mismo corto reajustado cada día, su P&L de camino en un periodo es, "
          "sin costes ni intereses y hasta segundo orden,")
    rep.p("camino ≈ (|w| + w²)·(L²/2)·(Σx² − (Σx)²),   con x = u − f;   a +1x (w = 1/3) el coeficiente es 2 y a "
          "+3x (w = 1), 9.")
    rep.p("*Pie de fórmula: el corto sin reajustar gana la suma de los cuadrados de los movimientos diarios del periodo y "
          "pierde el cuadrado del movimiento acumulado; gana si el mercado va y vuelve y pierde si tiene tendencia.* "
          "Notación: w, la parte del patrimonio vendida en corto; L = −3, el apalancamiento del fondo; u, la rentabilidad "
          "diaria del índice; f, su tipo de financiación; Σ, la suma sobre las sesiones del periodo. En media, "
          "(Σx)² = Σx² × VR, con VR el cociente de varianzas del periodo, así que el camino medio es positivo si y solo si "
          "VR < 1. Es la fórmula del par corto de R22 (coeficiente L²/2) para un solo fondo: comprar QQQ o TQQQ equivale "
          "al corto reajustado cada día (la exposición diaria es la misma), así que el camino es también la diferencia "
          "frente al fondo comprado, antes de comisiones.")
    mv = mech.copy()
    for c in ("camino (%/año)", "aproximación (%/año)"):
        mv[c] = mv[c] * 100
    mv = mv.rename(columns={"camino (%/año)": "camino (pp/año)", "aproximación (%/año)": "aproximación (pp/año)"})
    tab("Camino bruto por semana o mes naturales del corto que deriva frente al diario (r22.period_paths, sin costes ni "
        "intereses), su aproximación, el VR de esos periodos (media de (Σx)² / media de Σx²) y si el signo del camino "
        "medio es el que predice VR; «periodos coherentes»: fracción de periodos en que el signo del camino es el de "
        "Σx² − (Σx)²; pendiente de la regresión del camino sobre la aproximación (Newey-West 12)", mv,
        pct={"periodos coherentes"}, digits=3)
    TN["mech"] = tab.n
    tab("«Nunca»: la ventana entera como un solo periodo, por 1 $ de patrimonio inicial, sin costes ni intereses ni "
        "mantenimiento: P&L del corto que deriva, del corto diario, su diferencia (camino), la aproximación de segundo "
        "orden (mala con movimientos tan grandes) y el VR de la ventana, (Σx)²/Σx²", mech_n, digits=2)
    TN["mechn"] = tab.n
    fv2 = fr.copy()
    fv2["Δg frente al diario"] = fv2["Δg frente al diario"] * 100
    fv2 = fv2.rename(columns={"Δg frente al diario": "Δg frente al diario (pp)", "IC90": "IC90 (pp)"})
    tab("Δg simulado (neto de costes, sin préstamo, producto remunerado, con el mantenimiento) de cada frecuencia frente "
        "al reajuste diario, y el VR de sus periodos (el de la ventana en «nunca»); p_RW en la familia de las 12 "
        "comparaciones de cada ventana", fv2, digits=3)
    TN["freq"] = tab.n
    vt = vr_m.set_index(["ventana", "mercado", "h (sesiones)"]).copy()
    vt["z medio (%/año)"] = vt["z medio (%/año)"] * 100
    vt = vt.rename(columns={"z medio (%/año)": "z medio (pp/año)"})
    tab("Cociente de varianzas VR(h) de x = u − f con ventanas solapadas de h sesiones (como R22, tabla 18) y contraste "
        "de z = Σx² − (Σx)² frente a 0 (Hansen-Hodrick, h − 1 retardos): VR < 1 y t > 0 es reversión", vt, digits=3)
    lv = vrl.set_index(["mercado", "época", "h (sesiones)"]).copy()
    zc = [c for c in lv.columns if c.startswith("z medio")][0]
    lv[zc] = lv[zc] * 100
    lv = lv.rename(columns={zc: "z medio (pp/año)"})
    tab("Enlace con el artículo 3: VR(h) de la historia larga con la función de R22 (r22.variance_ratio_test; S&P 500 "
        "desde 1928, Nasdaq-100 desde 1985-10)", lv, digits=3)
    TN["vrl"] = tab.n
    fig_mechanism(C["periods"], rep)
    rep.p(f"**Lectura.** La segunda parte de H3(a) se cumple en todas las combinaciones: en 2005–2026 los periodos "
          f"semanales y mensuales tuvieron VR < 1 (el mercado revirtió dentro de la semana y del mes) y los cortos "
          f"semanales y mensuales ganaron al diario; en la ventana entera VR fue de 5 a 12 (veinte años de tendencia) y el "
          f"corto sin reajustar perdió. La aproximación explica el camino mes a mes (correlación de "
          f"{num_es(mech['correlación'].min(), 3)} a {num_es(mech['correlación'].max(), 3)}). Lo que no garantiza es el "
          f"futuro: la reversión de 2000–2026 es la excepción de la historia larga (VR(21) del S&P 500 desde 1928 = "
          f"{num_es(vrl21.loc[SP, 'VR(h)'], 2)}; {num_es(vr3873, 2)} en 1938–1973, tabla «T:vrl»). Vender el −3x sin reajuste diario es una "
          f"apuesta a que VR siga por debajo de 1, como el par corto de R22, no una prima por el «decaimiento».")

    rep.h("5. La familia completa y la búsqueda")
    tab("SPA de Hansen y rechazos de Holm y Romano–Wolf al 5% por ventana: la familia del pre-registro (48: SQQQ frente a "
        "QQQ y TQQQ, diario, semanal y mensual), las 128 configuraciones juntas (cada una frente a su fondo comprado) y "
        "por mercado y exposición (32)", fams.assign(**{"Δg_mejor": fams["Δg_mejor"] * 100})
        .rename(columns={"Δg_mejor": "Δg de la mejor (pp)"}).set_index(["ventana", "familia"]), digits=4)
    sv = dg_view(surv, {"Δg": "Δg (pp)", "p": "p", "p_RW": "p_RW"})
    tab("Configuraciones que sobreviven a Romano–Wolf en la ventana principal (familia de 128)", sv, digits=3)
    fig_borrow(res, bs, rep)
    rep.p("**Lectura.** El SPA rechaza porque hay configuraciones que ganan de verdad dentro de la muestra, pero todas "
          "tienen un préstamo de como mucho el 2%: el corto diario a +3x cobra las comisiones de los fondos y los "
          "semanales y mensuales sin préstamo cobran el cociente de varianzas de 2005–2026. Ninguna con el 5,2% (la "
          f"mediana del pre-registro) o el 18% sobrevive, y el producto sin remunerar resta "
          f"{num_es(-dec.loc[(NDX, '+1x', 'diario'), '(5) sin remunerar'] * 100, 1)} pp (+1x) y "
          f"{num_es(-dec.loc[(NDX, '+3x', 'diario'), '(5) sin remunerar'] * 100, 1)} pp (+3x) al año con una letra media "
          f"del {pct_es(rf_mean, 1)}. La tabla completa está en el anexo. El mensual del SPXU a +1x "
          "cae más de lo lineal entre el 2% y el 5,2% (figura): con el préstamo más alto el patrimonio es algo menor, el "
          "corto incumple el mantenimiento el 10-10-2008 (con el 2% no llega a hacerlo), se reduce a su objetivo ese cierre "
          "y se pierde parte del rebote del lunes 13-10-2008. La regla de margen hace el resultado sensible al camino.")

    rep.h("6. Préstamo que anula la ventaja y disponibilidad")
    bv = be.copy()
    for c in bv.columns:
        if c.startswith("Δg"):
            bv[c] = bv[c].astype(float) * 100
    bv = bv.rename(columns={c: c.replace("Δg con", "Δg (pp) con") for c in bv.columns})
    bv.index = bv.index.set_levels([[WIN_LABEL[w] for w in bv.index.levels[0]]] + list(bv.index.levels[1:]))
    tab("Δg frente al fondo comprado con cada préstamo de la rejilla y préstamo que lo anula (interpolación lineal; "
        "ventana principal y productos reales)", bv, digits=2)
    TN["be"] = tab.n
    tab("Préstamo de títulos en IBKR (iBorrowDesk, % anual; disponibles en acciones)", bs, digits=2)
    tab("Préstamo y precio: correlación de la tasa diaria con la rentabilidad del ETF en las 21 sesiones anteriores, "
        "tasa mediana en los días en que el ETF está en su 10% de mayores subidas (cuando el corto pierde) y días con "
        "pocas acciones disponibles", av, digits=2)
    rep.p(f"**Lectura.** Los 12 meses de iBorrowDesk (del {dmy(bs['desde'].min())} al {dmy(bs['hasta'].max())}) dan a "
          f"SQQQ un préstamo mediano del {num_es(bs.loc['SQQQ', 'mediana'], 1)}% y a SPXU del "
          f"{num_es(bs.loc['SPXU', 'mediana'], 1)}%, entre el 2% y el 5,2% de la rejilla. Con eso, a precios del pasado, los "
          "cortos semanales y mensuales quedan cerca de su punto de equilibrio (tabla «T:be») y el diario pierde salvo el de "
          "SQQQ a +3x con el producto remunerado. Hay tres límites que la simulación no recoge: (1) el préstamo sube cuando el −3x "
          f"sube (en SPXU la mediana pasa del {num_es(av.loc['SPXU', 'préstamo mediano (%)'], 1)}% al "
          f"{num_es(av.loc['SPXU', 'préstamo mediano con el ETF en su 10% de subidas (%)'], 1)}% en sus días de mayores "
          "subidas), justo cuando el corto pierde; (2) la disponibilidad cae a casi cero algunos días (1 acción de SQQQ y "
          "100 de SPXU como mínimo), y el bróker puede recomprar los títulos prestados (recall) sin aviso; (3) muchas "
          "cuentas minoristas no remuneran el producto del corto, y quien lo remunera suele hacerlo solo por encima de "
          "un saldo (condiciones de IBKR sin verificar). Además, para un minorista de la UE el corto de un ETF de EE. UU. "
          "depende del bróker: la normativa PRIIPs impide comprar ETFs sin documento KID, pero no está verificado aquí si "
          "impide abrir un corto; y en EE. UU. el mantenimiento de FINRA para un ETF 3x es del 90% del valor del corto "
          "(el bróker puede exigir más).")

    rep.h("7. Episodios")
    ev = C["episodes"]
    tab("Rentabilidad en cada episodio (ventana principal): fondo comprado y cortos de H3(b) (préstamo 5,2%, producto "
        "sin remunerar); fechas aproximadas del máximo al mínimo del índice", ev, pct=set(ev.columns), digits=3)
    TN["ep"] = tab.n
    fig_wealth(C["S"], rep)
    rep.p("**Lectura.** El corto del −3x es la misma exposición alcista: con reajuste diario pierde en cada episodio lo "
          "mismo que QQQ o TQQQ; con reajuste semanal o mensual, algo menos o algo más según el camino dentro del periodo "
          "(menos en 2008 y en la covid, más en agosto de 2015 y en el cuarto trimestre de 2018). «Nunca» "
          "no pierde en los episodios posteriores a 2009 porque para entonces el corto ya casi no existe: el −3x cayó más "
          "de un 99% y el patrimonio quedó en letras.")

    rep.h("8. Historia larga (fuera del pre-registro)")
    lg = C["long"].copy()
    lg["Δg"] = lg["Δg"] * 100
    lg = lg.rename(columns={"Δg": "Δg (pp)", "IC90": "IC90 (pp)"})
    tab("Descriptivo, fuera del pre-registro: la misma comparación desde el 4-2-1938 (S&P 500) y desde el 1-10-1985 "
        "(Nasdaq-100) con los fondos sintéticos de R22 antes de su lanzamiento; préstamo 0 y 5,2%", lg,
        pct={"CAGR", "CAGR referencia"}, digits=2)
    rep.p("**Lectura.** En la historia larga del S&P 500 (VR(21) ≈ 1) los cortos semanales y mensuales a +1x no ganan a SPY "
          "ni sin préstamo; lo que queda a +3x es sobre todo el cobro de comisiones, que un préstamo del 5,2% se come. En "
          "el Nasdaq-100 desde 1985 (VR(21) = 0,86) la ventaja semanal y mensual a +3x sin préstamo es grande, pero su "
          "intervalo con el 5,2% incluye el 0. Los fondos anteriores a 2009–2010 son sintéticos y el préstamo histórico es "
          "desconocido.")

    rep.h("Salvedades y datos sin verificar")
    rep.p("\n".join([
        "- Préstamo: 12 meses de un bróker (IBKR vía iBorrowDesk); antes, la rejilla constante del pre-registro. El "
        "coste real en las crisis de 2008, 2020 o 2022, desconocido.",
        "- SQQQ, SPXU, TQQQ y UPRO son sintéticos antes de 2009–2010 (s calibrado con su vida real; la validación fuera "
        "de muestra de R22 está en la tabla 1).",
        "- Recall, disponibilidad nula, cambios de margen del bróker y el tratamiento fiscal del corto no se simulan.",
        "- Acceso minorista en la UE (PRIIPs) y las condiciones de remuneración del producto del corto en IBKR: sin "
        "verificar.",
        "- «Antes/desde el 28-2-2018» es el corte del artículo (el cambio de SVXY y UVXY); en renta variable no hubo "
        "ningún cambio de producto en esa fecha.",
        "- Dentro de muestra: la reversión de 2000–2026 no es la de la historia larga (sección 4).",
        "- Cambios tras la revisión del laboratorio, con los resultados ya vistos: H3(b) se decide en la familia del "
        "pre-registro (48: SQQQ frente a QQQ y TQQQ, diario, semanal y mensual) y la de 128 queda como ampliación; y "
        "«solo por el tamaño y la frecuencia» se lee igual que en R27 (el sobrecoste s de los fondos es fricción, no "
        "tamaño), con la fricción medida sobre los ETFs reales de 2011–2026 (tabla «T:fric»). Antes, esta línea contaba "
        "todo lo que cobra el corto diario (gastos y sobrecoste) como «ni tamaño ni frecuencia».",
    ]))

    rep.h("Anexo: las 128 configuraciones")
    cols = {"CAGR": "CAGR", "CAGR_referencia": "CAGR del fondo comprado", "maxDD": "maxDD", "Δg": "Δg (pp)", "p": "p",
            "p_Holm": "p_Holm", "p_RW": "p_RW", "incumplimientos_año": "incumplimientos/año", "ruina": "ruina"}
    tab("Ventana principal (21-12-2005 → 25-9-2026): todas las configuraciones frente a su fondo comprado; Holm y "
        "Romano–Wolf en la familia de 128", dg_view(main, cols), pct={"CAGR", "CAGR del fondo comprado"}, digits=3)
    allw = pd.DataFrame({WIN_LABEL[w]: t["Δg"] * 100 for w, t in res.items()})
    allw = allw.join(pd.DataFrame({f"p_RW · {WIN_LABEL[w]}": t["p_RW"] for w, t in res.items()}))
    tab("Δg (pp/año) y p_RW de las 128 configuraciones en cada ventana", allw, digits=2)
    for k, n in TN.items():
        rep.lines = [ln.replace(f"«T:{k}»", str(n)) for ln in rep.lines]
    rep.save()
    return C


if __name__ == "__main__":
    run()
