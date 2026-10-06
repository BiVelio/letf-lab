"""R31 — El decaimiento del inverso (H1 y H2 del artículo 6).

Pregunta (pre-registro del artículo 6, letf/preregistro/articulo6_inversos.md, commit 8012967).
Un ETF inverso apalancado pierde valor por el reajuste diario con más violencia que el alcista:
frente a L veces el rendimiento logarítmico del índice, un fondo diario pierde ≈ ½(L² − L)·Σr²,
6·Σr² con L = −3 y 3·Σr² con L = +3. ¿Se cumple con los fondos reales (H1) y se cobra vendiendo
el −3x (H2)?

- H1 · Mecánica. En cada par real, el arrastre realizado del inverso frente a L veces el
  rendimiento logarítmico del índice coincide con ½(L² − L)·Σr² (pendiente entre 0,9 y 1,1 y
  R² > 0,9 en ventanas anuales), y es aproximadamente el doble que el del alcista con el mismo |L|.
- H2 · Mediana frente a media. Mantener en corto el −3x h sesiones (h = 5, 21, 63, 252) gana en la
  mediana más que mantener el +3x las mismas sesiones, pero no en la media: la diferencia de medias
  no es significativamente mayor que cero en un bootstrap por bloques de la historia. El
  «decaimiento» se cobra vendiendo la cola derecha del inverso.

Cómo se contrasta (la concreción se escribió al programar esta línea; la revisión del 30-09-2026, hecha
después de ver los resultados, la acercó al texto del pre-registro: ver «Desviaciones y lecturas»).

H1.
- Arrastre literal, el del pre-registro («frente a L veces el rendimiento logarítmico del índice»):
  L·Σlog(1 + u) − Σlog(1 + R), de un fondo de apalancamiento L con rendimiento diario R sobre el índice
  u. El veredicto de H1 se da con él. Por la identidad log-crecimiento del fondo = L·log-crecimiento del
  índice + Σ(letra − L·f) − Σφ − D, el arrastre literal es D + Σφ − Σ(letra − L·f): lleva dentro el
  carry (letra − L·f) y la fricción φ = letra + L·(u − f) − R (gasto, sobrecoste del swap y error de
  seguimiento), con f el tipo de financiación (fed funds con un mes de retraso).
- Dos lecturas que el pre-registro no fija, dadas aparte: neto del carry, D + φ, y neto del carry y de
  la fricción, D = Σ[(R − log(1 + R)) − L·(u − log(1 + u))], la convexidad del fondo menos L veces la
  del índice, que no depende de f ni del gasto. Con R = L·u, D = ½(L² − L)·Σu² más términos de orden 3
  y superiores; el ruido de seguimiento suma ½Σe².
- «Ventanas anuales»: años naturales con al menos 63 sesiones del par. Por fondo, regresión MCO del
  arrastre de cada año sobre ½(L² − L)·Σu²_año con constante (errores de White); se cumple si la
  pendiente está en [0,9; 1,1] y R² > 0,9. «En cada par real» se juzga con el inverso de cada par del
  pre-registro (SH y PSQ incluidos); el alcista y los pares de la ampliación se dan igual.
- «Aproximadamente el doble», tal como está escrito: ΣD_inverso/ΣD_alcista en los años comunes entre
  1,8 y 2,2 (±10% de 2; IC90 remuestreando años, 2.000). Se da también frente al cociente de la
  fórmula, (|L| + 1)/(|L| − 1), que es 2 con |L| = 3 y 3 con |L| = 2, y con las otras dos medidas del
  arrastre. Con |L| = 1 (SH, PSQ) el alcista es el índice, sin arrastre: no se puede contrastar.
- Historia larga sintética (descriptiva): los fondos de R22 (inverse_pairs.synthetic_history) sobre el
  S&P 500 desde 1928 y el Nasdaq-100 desde el 1-10-1985, L = −3, −2, −1, 2 y 3.

H2.
- «Mantener en corto el −3x»: el corto sin reajuste del protocolo común, s = 1 (se vende en el −3x el
  100% del capital: exposición inicial +3, la del +3x comprado con el 100%). Entrada al cierre de t y
  salida al cierre de t + h, una ventana por sesión (solapadas); 5 pb por unidad negociada al entrar,
  al salir y en cada reducción; el capital cobra la letra; préstamo sobre el valor vivo del corto;
  mantenimiento FINRA del 90% del valor del corto: si al cierre el patrimonio no lo cubre, el corto se
  reduce ese cierre a s veces el patrimonio (la regla «restaurar» de R29); ruina si el patrimonio
  llega a 0 (queda a 0). El +3x: comprar y mantener el 100% (5 pb al entrar y al salir). Rentabilidad
  simple de la ventana, por 1 $.
- Préstamo y producto: la rejilla del pre-registro, préstamo mínimo, mediano o p90 de cada ticker en
  IBKR (iBorrowDesk; SPXU en el S&P 500 sintético y SQQQ en el Nasdaq-100) × producto a la letra −
  0,5% o sin remunerar. Referencia: préstamo mediano y producto remunerado. Además, con el préstamo
  mediano y el producto remunerado, sin mantenimiento (el pago puro, solo con la ruina) y s = 1/3
  frente a 1/3 del +3x y 2/3 en letras; y, fuera de la rejilla y solo como referencia, sin préstamo.
- Datos: los pares 3x reales del pre-registro (SQQQ/TQQQ, SPXU/UPRO, SOXS/SOXL, TZA/TNA, FAZ/FAS,
  LABD/LABU, SDOW/UDOW), cada uno en su vida común, y la historia larga sintética (−3x frente a +3x
  del S&P 500 desde 1928 y del Nasdaq-100 desde 1985-10). Los 3x de la ampliación, aparte.
- Inferencia: diferencia de medianas (mediana del corto − mediana del +3x) y de medias, con un
  bootstrap estacionario de las ventanas (bloque medio 63 sesiones, el del protocolo; 2.000
  remuestreos, los mismos para todas las variantes), IC90 percentil y p unilateral normal con el error
  del bootstrap (H0: diferencia ≤ 0). La media se contrasta también con errores de Hansen-Hodrick
  (stats.hac_overlap(h)). H2 se cumple en una (serie, h) si la mediana del corto supera a la del +3x
  con p de Romano–Wolf < 0,05 en la familia de las 9 series × 4 horizontes (36) de cada variante y la
  diferencia de medias NO es significativamente mayor que 0 (p unilateral sin corregir ≥ 0,05; sin
  corregir es lo que más fácil pone H2 en aprietos). Sensibilidad: bloque max(63, 2h) y Holm.
- «Se cobra vendiendo la cola derecha del inverso»: cuantiles, asimetría, probabilidad de perder
  más del 50%, y la parte de la ganancia media del corto que se devuelve en su 5% de peores
  ventanas; el pago de opción se explica con la aproximación log F ≈ −3X − 6Q (y log G ≈ 3X − 3Q),
  X el log-rendimiento del índice en la ventana y Q = Σu².

Protocolo del laboratorio: 5 pb por unidad negociada, la caja a la letra (^IRX), semillas fijas.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, skew

from letf_lab import stats
from letf_lab.data import vix
from letf_lab.report import FIG_W, Report, comma_fig, legend_band, num_es, pct_es
from letf_lab.research import inverse_pairs as ip
from letf_lab.research import r13_lrs as r13
from letf_lab.research import r19_census as r19
from letf_lab.research import r22_inverse as r22
from letf_lab.research.r26_vix_data import Tables, pp

SEED = 31
B = 2000                          # remuestreos (como R13, R19, R22 y R29)
BLOCK = 63                        # bloque medio del bootstrap estacionario (sesiones)
H_LIST = (5, 21, 63, 252)         # horizontes de H2 (sesiones)
MIN_YEAR = 63                     # sesiones mínimas de un año natural en H1
SLOPE_BAND = (0.9, 1.1)           # H1: pendiente
R2_MIN = 0.9                      # H1: R²
RATIO_TOL = 0.10                  # H1: «aproximadamente», ±10%
DOUBLE = 2.0                      # H1: «el doble», tal como está escrito (1,8–2,2)
ALPHA = 0.05
COST_BPS = r13.COST_BPS           # 5 pb por unidad negociada
HAIRCUT = vix.PROCEEDS_SPREAD     # producto del corto a la letra − 0,5%
MAINT = min(vix.FINRA_MAINT * 3, 1.0)   # 90% del valor del corto de un −3x
CLIP = r22.CLIP
END = vix.END
SP, NDX = ip.SP, ip.NDX
SYN = {SP: "S&P 500 sintético (1928–2026)", NDX: "Nasdaq-100 sintético (1985–2026)"}
SYN_BORROW = {SP: "SPXU", NDX: "SQQQ"}

LEVELS = ("mín", "mediana", "p90")                     # rejilla de préstamo del pre-registro
LEVEL_ES = {"mín": "mínimo", "mediana": "mediano", "p90": "p90"}


def grid_name(lv: str, rebate: bool) -> str:
    return f"préstamo {LEVEL_ES[lv]}, producto {'remunerado' if rebate else 'sin remunerar'}"


GRID_V = [grid_name(lv, rb) for lv in LEVELS for rb in (True, False)]
MAIN = grid_name("mediana", True)
PURE, THIRD, NOBORROW = "sin mantenimiento", "s = 1/3", "sin préstamo (fuera de la rejilla)"
VARIANTS = {**{grid_name(lv, rb): dict(s=1.0, maint=MAINT, borrow=lv, rebate=rb) for lv in LEVELS for rb in (True, False)},
            PURE: dict(s=1.0, maint=None, borrow="mediana", rebate=True),
            THIRD: dict(s=1 / 3, maint=MAINT, borrow="mediana", rebate=True),
            NOBORROW: dict(s=1.0, maint=MAINT, borrow=None, rebate=True)}
YEARS_SHOWN = (1929, 1931, 1932, 1933, 1974, 1987, 2000, 2001, 2002, 2008, 2020, 2022, 2024)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2de"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"
L_COL = {-3: BLUE, 3: ORANGE, -2: AQUA, 2: YELLOW, -1: MAGENTA}
L_MARK = {-3: "o", -2: "o", -1: "o", 2: "s", 3: "s"}


# ───────────────────────────── utilidades ─────────────────────────────

def lx(L: float) -> str:
    return ("+" if L > 0 else "") + r22.lx(L)


def peq(p: float) -> str:
    return r22.peq(p)


def ci_text(lo: float, hi: float, k: int = 1) -> str:
    return f"[{pp(lo, k)}; {pp(hi, k)}]"


# ───────────────────────────── H1: el arrastre ─────────────────────────────

def drag_daily(R: pd.Series, u: pd.Series, rf: pd.Series, f: pd.Series, L: float) -> pd.DataFrame:
    """Términos diarios de la identidad log(1 + R) = L·log(1 + u) + (letra − L·f) − φ − D, con
    φ = letra + L·(u − f) − R y D = (R − log(1 + R)) − L·(u − log(1 + u)) (exacta)."""
    lr = np.log1p(R.clip(lower=CLIP))
    lu = np.log1p(u.clip(lower=CLIP))
    return pd.DataFrame({"g": lr, "Lg": L * lu, "carry": rf - L * f, "phi": rf + L * (u - f) - R,
                         "D": (R - lr) - L * (u - lu), "u2": u ** 2})


def annual(dd: pd.DataFrame, L: float, min_sessions: int = MIN_YEAR) -> pd.DataFrame:
    """Sumas por año natural (los de al menos min_sessions sesiones) y la fórmula ½(L² − L)·Σu²."""
    g = dd.groupby(dd.index.year)
    a = g.sum()
    a["sesiones"] = g.size()
    a["P"] = 0.5 * (L * L - L) * a["u2"]
    a["vol"] = np.sqrt(252 * a["u2"] / a["sesiones"])
    a.index.name = "año"
    return a[a["sesiones"] >= min_sessions]


def fit_drag(a: pd.DataFrame, col: str = "D") -> dict:
    """MCO de col_año sobre la fórmula ½(L² − L)·Σu²_año con constante (White); pendiente por el origen."""
    y, x = a[col].astype(float), a["P"].astype(float)
    fit = stats.ols_hac(y, x.to_frame("P"), 0)
    b, se = float(fit.beta[1]), float(fit.se[1])
    return {"años": len(a), "pendiente": b, "EE": se, "IC90_bajo": b - 1.645 * se, "IC90_alto": b + 1.645 * se,
            "constante": float(fit.beta[0]), "R2": fit.r2, "pendiente_origen": float((y * x).sum() / (x * x).sum()),
            "cumple": bool(SLOPE_BAND[0] <= b <= SLOPE_BAND[1] and fit.r2 > R2_MIN)}


def theory_ratio(L: int) -> float:
    """Cociente de los arrastres del −L y del +L según la fórmula: (L² + L)/(L² − L)."""
    return (L * L + L) / (L * L - L) if L > 1 else np.nan


def ratio_boot(di: pd.Series, dl: pd.Series, rng, B_: int = B) -> tuple[float, float, float]:
    """ΣD_inverso/ΣD_alcista en los años comunes e IC90 remuestreando años."""
    j = pd.concat([di, dl], axis=1, keys=["i", "l"]).dropna()
    r = float(j.i.sum() / j.l.sum())
    k = rng.integers(0, len(j), (B_, len(j)))
    draws = j.i.to_numpy()[k].sum(1) / j.l.to_numpy()[k].sum(1)
    lo, hi = np.percentile(draws, [5, 95])
    return r, float(lo), float(hi)


def fund_side(df: pd.DataFrame, col: str, L: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dd = drag_daily(df[col], df["u"], df["rf"], df["f"], L)
    return dd, annual(dd, L)


def literal_drag(a: pd.DataFrame) -> pd.Series:
    """Arrastre literal del pre-registro: L veces el log-rendimiento del índice menos el del fondo,
    L·Σlog(1 + u) − Σlog(1 + R) = D + φ − (letra − L·f): lleva dentro el carry y la fricción."""
    return a["Lg"] - a["g"]


def h1_real(frames: dict, rng) -> dict:
    """Por par real: ajustes de H1 (inverso y alcista) con las tres medidas del arrastre (literal,
    neta del carry y neta del carry y la fricción), cocientes, fricción y descomposición."""
    fits, ratios, decomp, years = [], [], [], []
    for key, df in frames.items():
        p = ip.BY_KEY[key]
        sides = [("inverso", p.inverse, "inv", -p.L)] + ([("alcista", p.long, "long", p.L)] if p.long else [])
        ann = {}
        for side, tk, col, L in sides:
            dd, a = fund_side(df, col, L)
            ann[side] = a
            f0 = fit_drag(a, "D")
            f1 = fit_drag(a.assign(DF=a.D + a.phi), "DF")
            f2 = fit_drag(a.assign(LIT=literal_drag(a)), "LIT")
            fits.append({"par": key, "pre-registro": p.prereg, "fondo": tk, "lado": side, "L": L, **f0,
                         "pendiente literal": f2["pendiente"], "IC90 literal bajo": f2["IC90_bajo"],
                         "IC90 literal alto": f2["IC90_alto"], "constante literal": f2["constante"],
                         "R2 literal": f2["R2"], "cumple literal": f2["cumple"],
                         "pendiente neta del carry": f1["pendiente"], "constante neta del carry": f1["constante"],
                         "R2 neto del carry": f1["R2"], "cumple neto del carry": f1["cumple"]})
            n = len(dd)
            decomp.append({"par": key, "pre-registro": p.prereg, "fondo": tk, "L": L, "desde": df.index[0].date(),
                           "g fondo": dd.g.sum() * 252 / n, "L × g índice": dd.Lg.sum() * 252 / n,
                           "letra − L·f": dd.carry.sum() * 252 / n, "− fricción φ": -dd.phi.sum() * 252 / n,
                           "− arrastre D": -dd.D.sum() * 252 / n,
                           "fórmula ½(L²−L)Σu²": 0.5 * (L * L - L) * dd.u2.sum() * 252 / n,
                           "gasto del folleto": ip.expense_ratio(tk)})
            for yr, r in a.iterrows():
                years.append({"par": key, "fondo": tk, "L": L, "año": yr, "sesiones": int(r.sesiones), "vol índice": r.vol,
                              "Σu²": r.u2, "fórmula": r.P, "arrastre D": r.D, "fricción φ": r.phi, "carry": r.carry,
                              "arrastre literal": r.Lg - r.g, "g fondo": r.g, "L × g índice": r.Lg})
        if p.long:
            common = ann["inverso"].index.intersection(ann["alcista"].index)
            ai, al = ann["inverso"].loc[common], ann["alcista"].loc[common]
            rr, lo, hi = ratio_boot(ai.D, al.D, rng)
            rl = float(literal_drag(ai).sum() / literal_drag(al).sum())
            rn = float((ai.D + ai.phi).sum() / (al.D + al.phi).sum())
            th = theory_ratio(p.L)
            ratios.append({"par": key, "pre-registro": p.prereg, "|L|": p.L, "años": len(common),
                           "ΣD inverso": float(ai.D.sum()), "ΣD alcista": float(al.D.sum()),
                           "cociente": rr, "IC90_bajo": lo, "IC90_alto": hi,
                           "cumple el doble": bool(abs(rr / DOUBLE - 1) <= RATIO_TOL),
                           "fórmula": th, "cociente/fórmula": rr / th, "cumple": bool(abs(rr / th - 1) <= RATIO_TOL),
                           "cociente neto del carry": rn, "cumple el doble neto del carry": bool(abs(rn / DOUBLE - 1) <= RATIO_TOL),
                           "cociente literal": rl, "cumple el doble literal": bool(abs(rl / DOUBLE - 1) <= RATIO_TOL)})
    return {"fits": pd.DataFrame(fits), "ratios": pd.DataFrame(ratios), "decomp": pd.DataFrame(decomp),
            "years": pd.DataFrame(years)}


def h1_synthetic(hist: dict, rng) -> dict:
    """Lo mismo con la historia larga sintética (descriptivo)."""
    fits, ratios, years, ann_all = [], [], [], {}
    for mkt, df in hist.items():
        ann = {}
        for L in (-3, -2, -1, 2, 3):
            dd = drag_daily(df[L], df["u"], df["rf"], df["f"], L)
            a = annual(dd, L)
            ann[L] = a
            ann_all[(mkt, L)] = a
            f0 = fit_drag(a, "D")
            fits.append({"historia": SYN[mkt], "L": L, **f0})
            for yr, r in a.iterrows():
                years.append({"historia": SYN[mkt], "L": L, "año": yr, "vol índice": r.vol, "Σu²": r.u2,
                              "fórmula": r.P, "arrastre D": r.D, "fricción φ": r.phi})
        for Lp in (3, 2):
            rr, lo, hi = ratio_boot(ann[-Lp].D, ann[Lp].D, rng)
            th = theory_ratio(Lp)
            ratios.append({"historia": SYN[mkt], "|L|": Lp, "años": len(ann[Lp]), "cociente": rr, "IC90_bajo": lo,
                           "IC90_alto": hi, "cumple el doble": bool(abs(rr / DOUBLE - 1) <= RATIO_TOL), "fórmula": th,
                           "cociente/fórmula": rr / th, "cumple": bool(abs(rr / th - 1) <= RATIO_TOL)})
    return {"fits": pd.DataFrame(fits), "ratios": pd.DataFrame(ratios), "years": pd.DataFrame(years), "ann": ann_all}


# ───────────────────────────── H2: mantener h sesiones ─────────────────────────────

def hold_short(Rs: np.ndarray, rf: np.ndarray, h: int, s: float = 1.0, borrow: float = 0.0, rebate: bool = True,
               maint: float | None = MAINT, cost_bps: float = COST_BPS, haircut: float = HAIRCUT) -> tuple[np.ndarray, dict]:
    """Corto sin reajuste del −3x por s del capital, de cada cierre i al cierre i + h (una ventana por
    sesión, vectorizado). Rs[k], rf[k]: rendimiento del fondo y letra diaria del día k (cierre k − 1 →
    cierre k). Por 1 $: el patrimonio E cobra la letra; el corto de valor V paga su rendimiento y el
    préstamo, y su producto cobra la letra − haircut (rebate) o nada; si maint·V > E al cierre, el corto
    se reduce a s·E (5 pb de lo recomprado); ruina si E ≤ 0 (queda a 0); 5 pb al entrar y al salir.
    Mismas cuentas que r29.drift_sim con «nunca» y «restaurar» (tests/test_r31_inverse_drag.py)."""
    n = len(Rs)
    N = n - h
    c = cost_bps / 1e4
    base = np.arange(N)
    E = np.full(N, 1.0 - c * s)
    V = np.full(N, float(s))
    ruined = np.zeros(N, bool)
    cuts = np.zeros(N, int)
    for k in range(1, h + 1):
        r, q = Rs[base + k], rf[base + k]
        inc = E * q - V * r - V * borrow / 252
        if rebate:
            inc = inc + V * (q - haircut / 252)
        E = E + inc
        V = V * (1.0 + r)
        dead = E <= 0
        if dead.any():
            E = np.where(dead, 0.0, E)
            V = np.where(dead, 0.0, V)
            ruined |= dead
        if maint:
            br = (maint * V > E) & ~dead
            if br.any():
                newV = s * E
                E = np.where(br, E - c * np.abs(V - newV), E)
                V = np.where(br, newV, V)
                cuts += br
    E = np.maximum(E - c * V, 0.0)
    return E - 1.0, {"ruina": ruined, "reducciones": cuts}


def _window_sum(x: np.ndarray, h: int) -> np.ndarray:
    """Suma de x[i + 1 … i + h] para cada i = 0 … n − h − 1."""
    C = np.r_[0.0, np.cumsum(x)]
    n = len(x)
    i = np.arange(n - h)
    return C[i + h + 1] - C[i + 1]


def hold_long(Rl: np.ndarray, rf: np.ndarray, h: int, s: float = 1.0, cost_bps: float = COST_BPS) -> np.ndarray:
    """Comprar s del +3x (y 1 − s en letras) al cierre i y vender al cierre i + h; 5 pb al entrar y salir."""
    c = cost_bps / 1e4
    G = np.exp(_window_sum(np.log1p(np.maximum(Rl, CLIP)), h))
    C = np.exp(_window_sum(np.log1p(rf), h))
    return s * G * (1 - c) + (1 - s - c * s) * C - 1.0


def h2_inputs(frames: dict, hist: dict) -> dict:
    """Series de H2: nombre → inverso −3x, +3x, u, letra, rejilla de préstamo del inverso (fracción anual:
    mínimo, mediana y p90), familia, comisiones diarias de los dos fondos y letra media."""
    bg = ip.borrow_grid()

    def grid(tk):
        return {lv: (float(bg.loc[tk, lv]) / 100 if tk in bg.index else np.nan) for lv in LEVELS}

    out = {}
    for key, df in frames.items():
        p = ip.BY_KEY[key]
        if p.L != 3 or p.long is None:
            continue
        out[key] = {"inv": df["inv"], "long": df["long"], "u": df["u"], "rf": df["rf"],
                    "borrow": grid(p.inverse), "familia": p.prereg, "inverso": p.inverse,
                    "comisiones": fees_daily(df["inv"], df["long"], df["u"], df["rf"], df["f"]),
                    "letra": float(df["rf"].mean())}
    for mkt, df in hist.items():
        out[SYN[mkt]] = {"inv": df[-3], "long": df[3], "u": df["u"], "rf": df["rf"],
                         "borrow": grid(SYN_BORROW[mkt]), "familia": True, "inverso": SYN_BORROW[mkt],
                         "comisiones": fees_daily(df[-3], df[3], df["u"], df["rf"], df["f"]),
                         "letra": float(df["rf"].mean())}
    return out


def fees_daily(inv: pd.Series, lng: pd.Series, u: pd.Series, rf: pd.Series, f: pd.Series) -> float:
    """Lo que el −3x y el +3x rinden de menos que su réplica cada día (φ del −3x + φ del +3x, con
    φ = letra + L·(u − f) − R la fricción aritmética de cada fondo; media diaria de la serie). El corto
    de s en el −3x menos s en el +3x comprado gana a primer orden s·(φ_inv + φ_alc − b − haircut) al día
    con el producto remunerado, y s·(φ_inv + φ_alc − b − letra) sin remunerar (net_carry)."""
    phi_i = (rf - 3 * (u - f) - inv).mean()
    phi_l = (rf + 3 * (u - f) - lng).mean()
    return float(phi_i + phi_l)


def borrow_of(x: dict, v: str) -> float:
    """Préstamo anual (fracción) de la variante v en la serie x (0 en la referencia sin préstamo)."""
    lv = VARIANTS[v]["borrow"]
    if lv is None:
        return 0.0
    bw = x["borrow"]
    return float(bw[lv]) if isinstance(bw, dict) else float(bw)


def net_carry(x: dict, v: str, h: int) -> float:
    """A primer orden, lo que el corto de la variante v gana al +3x en h sesiones por las comisiones de los
    dos fondos, menos el préstamo y lo que no cobra del producto (el haircut, o toda la letra)."""
    cfg = VARIANTS[v]
    lost = HAIRCUT / 252 if cfg["rebate"] else x.get("letra", 0.0)
    return cfg["s"] * h * (x["comisiones"] - borrow_of(x, v) / 252 - lost)


def h2_series(x: dict, h: int) -> dict:
    """Rentabilidades de las ventanas de h sesiones: corto en cada variante, +3x, fondos brutos y el índice."""
    Rs, Rl = x["inv"].to_numpy(float), x["long"].to_numpy(float)
    rf, u = x["rf"].to_numpy(float), x["u"].to_numpy(float)
    out = {"fechas": x["inv"].index[: len(Rs) - h]}
    longs = {}
    for v, cfg in VARIANTS.items():
        sh, info = hold_short(Rs, rf, h, cfg["s"], borrow_of(x, v), cfg["rebate"], cfg["maint"])
        if cfg["s"] not in longs:
            longs[cfg["s"]] = hold_long(Rl, rf, h, cfg["s"])
        out[v] = {"corto": sh, "largo": longs[cfg["s"]], "ruina": info["ruina"], "reducciones": info["reducciones"]}
    out["F"] = np.exp(_window_sum(np.log1p(np.maximum(Rs, CLIP)), h))
    out["G"] = np.exp(_window_sum(np.log1p(np.maximum(Rl, CLIP)), h))
    out["X"] = _window_sum(np.log1p(u), h)
    out["Q"] = _window_sum(u ** 2, h)
    out["var_u"] = float(np.mean(u ** 2))
    return out


def dist(x: np.ndarray) -> dict:
    q = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    return {"media": float(x.mean()), "mediana": float(q[3]), "p1": q[0], "p5": q[1], "p25": q[2], "p75": q[4],
            "p95": q[5], "p99": q[6], "asimetría": float(skew(x)), "P(< −50%)": float((x < -0.5).mean()),
            "peor": float(x.min()), "mejor": float(x.max())}


def long_block(h: int) -> int:
    """Bloque medio de la comprobación de sensibilidad: max(63, 2h) (a 252 sesiones las ventanas se solapan
    251 sesiones y un bloque de 63 no recoge toda su dependencia)."""
    return max(BLOCK, 2 * h)


def boot_h2(S: dict, h: int, rng, B_: int = B, chunk: int = 100, block: int = BLOCK, variants=None) -> dict:
    """Diferencias de medias y de medianas (corto − +3x) en B remuestreos estacionarios de las ventanas
    (bloque medio `block`; el del protocolo, 63), los mismos para todas las variantes."""
    variants = list(variants or VARIANTS)
    N = len(S["fechas"])
    out = {v: {"media": [], "mediana": []} for v in variants}
    done = 0
    while done < B_:
        m = min(chunk, B_ - done)
        idx = np.stack([stats.stationary_indices(N, block, rng) for _ in range(m)])
        med_l = {}
        for v in variants:
            s, l = S[v]["corto"][idx], S[v]["largo"][idx]
            key = id(S[v]["largo"])
            if key not in med_l:
                med_l[key] = np.median(l, axis=1)
            out[v]["media"].append((s - l).mean(axis=1))
            out[v]["mediana"].append(np.median(s, axis=1) - med_l[key])
        done += m
    return {v: {k: np.concatenate(a) for k, a in d.items()} for v, d in out.items()}


def _p_and_tboot(obs: float, draws: np.ndarray) -> tuple[float, float, np.ndarray]:
    """p unilateral normal con el error del bootstrap (H0: diferencia ≤ 0), t observado y t del bootstrap
    centrados en la nula, (draw − obs)/EE, para Romano–Wolf."""
    se = draws.std(ddof=1)
    if not se > 0:
        return np.nan, np.nan, np.zeros_like(draws)
    return float(norm.sf(obs / se)), obs / se, (draws - obs) / se


def _rw(t: pd.DataFrame, sel: pd.Series, tcol: str, tb: dict) -> np.ndarray:
    """Romano–Wolf (r19_census.romano_wolf) en las filas sel: t observados y B × K t del bootstrap. Las
    series y horizontes se remuestrean por separado (calendarios distintos), así que las columnas son
    independientes entre sí: el máximo de la nula es el de columnas independientes, conservador frente a
    la dependencia positiva real."""
    ix = t.index[sel]
    return r19.romano_wolf(t.loc[ix, tcol].to_numpy(float), np.column_stack([tb[i] for i in ix]))


def verdict(med_ok: pd.Series, mean_ok: pd.Series) -> np.ndarray:
    return np.where(med_ok & mean_ok, "se cumple",
                    np.where(~med_ok & mean_ok, "falla: mediana", np.where(med_ok, "falla: media", "falla: las dos")))


def h2_table(inputs: dict, B_: int = B) -> tuple[pd.DataFrame, dict]:
    """Por serie, h y variante: distribución del corto y del +3x, diferencias de medias y medianas con
    su IC90 y p del bootstrap (bloque 63), p de Romano–Wolf y de Holm de la mediana en la familia de
    cada variante, t de Hansen-Hodrick de la media y, en la variante principal, las mismas p con el
    bloque max(63, 2h)."""
    rows, series, tb, tb_long = [], {}, {}, {}
    for j, (name, x) in enumerate(inputs.items()):
        for h in H_LIST:
            S = h2_series(x, h)
            series[(name, h)] = S
            bt = boot_h2(S, h, np.random.default_rng([SEED, j, h]), B_)
            bl = boot_h2(S, h, np.random.default_rng([SEED, j, h, 1]), B_, block=long_block(h), variants=[MAIN])
            for v in VARIANTS:
                s, l = S[v]["corto"], S[v]["largo"]
                d = s - l
                md, mdd = float(d.mean()), float(np.median(s) - np.median(l))
                bm, bd = bt[v]["media"], bt[v]["mediana"]
                p_m, _, _ = _p_and_tboot(md, bm)
                p_d, t_d, tb_d = _p_and_tboot(mdd, bd)
                tb[len(rows)] = tb_d
                extra = {}
                if v == MAIN:
                    p_ml, _, _ = _p_and_tboot(md, bl[v]["media"])
                    p_dl, t_dl, tb_dl = _p_and_tboot(mdd, bl[v]["mediana"])
                    tb_long[len(rows)] = tb_dl
                    extra = {"p mediana (bloque largo)": p_dl, "t mediana (bloque largo)": t_dl,
                             "p media (bloque largo)": p_ml}
                hh = stats.ols_hac(pd.Series(d), pd.DataFrame(index=range(len(d))), **stats.hac_overlap(h))
                ds, dl = dist(s), dist(l)
                worst = np.sort(s)[: max(1, int(round(0.05 * len(s))))]
                rows.append({"serie": name, "familia": x["familia"], "h": h, "variante": v, "ventanas": len(s),
                             "desde": S["fechas"][0].date(), "hasta": S["fechas"][-1].date(),
                             "préstamo": borrow_of(x, v), "producto remunerado": VARIANTS[v]["rebate"],
                             "mediana corto": ds["mediana"], "mediana +3x": dl["mediana"], "Δ mediana": mdd,
                             "Δ mediana IC90_bajo": float(np.percentile(bd, 5)), "Δ mediana IC90_alto": float(np.percentile(bd, 95)),
                             "p mediana": p_d, "t mediana": t_d,
                             "media corto": ds["media"], "media +3x": dl["media"], "Δ media": md,
                             "Δ media IC90_bajo": float(np.percentile(bm, 5)), "Δ media IC90_alto": float(np.percentile(bm, 95)),
                             "p media": p_m, "t media (HH)": float(hh.t[0]),
                             "comisiones − préstamo en h": net_carry(x, v, h),
                             "corto gana": float((d > 0).mean()),
                             **{f"{k} corto": ds[k] for k in ("p1", "p5", "p95", "p99", "asimetría", "P(< −50%)", "peor")},
                             **{f"{k} +3x": dl[k] for k in ("p1", "p5", "p95", "p99", "asimetría", "P(< −50%)", "mejor")},
                             "cola 5% del corto": float(worst.sum() / len(s)),
                             "media corto sin su peor 5%": float(np.sort(s)[len(worst):].sum() / len(s)),
                             "ruinas": int(S[v]["ruina"].sum()), "ventanas con reducción": float((S[v]["reducciones"] > 0).mean()),
                             **extra})
    t = pd.DataFrame(rows)
    for c in ("p mediana RW", "p mediana Holm", "p mediana RW (bloque largo)"):
        t[c] = np.nan
    for grp in (t.familia, ~t.familia):      # la familia del pre-registro y la ampliación, cada variante aparte
        for v in VARIANTS:
            fv = grp & (t.variante == v)
            if not fv.any():
                continue
            t.loc[fv, "p mediana RW"] = _rw(t, fv, "t mediana", tb)
            t.loc[fv, "p mediana Holm"] = stats.holm(t.loc[fv, "p mediana"].to_numpy())
            if v == MAIN:
                t.loc[fv, "p mediana RW (bloque largo)"] = _rw(t, fv, "t mediana (bloque largo)", tb_long)
    med_ok = (t["Δ mediana"] > 0) & (t["p mediana RW"] < ALPHA)
    mean_ok = ~((t["Δ media"] > 0) & (t["p media"] < ALPHA))
    t["mediana a favor del corto"] = med_ok
    t["media no mayor que 0"] = mean_ok
    t["H2"] = verdict(med_ok, mean_ok)
    ml = t.variante == MAIN
    t["H2 (bloque largo)"] = None
    t.loc[ml, "H2 (bloque largo)"] = verdict((t["Δ mediana"] > 0) & (t["p mediana RW (bloque largo)"] < ALPHA),
                                             ~((t["Δ media"] > 0) & (t["p media (bloque largo)"] < ALPHA)))[ml.to_numpy()]
    t["H2 (Holm)"] = verdict((t["Δ mediana"] > 0) & (t["p mediana Holm"] < ALPHA), mean_ok)
    return t, series


def fund_moments(inputs: dict, series: dict) -> pd.DataFrame:
    """Media y mediana del múltiplo bruto de cada fondo en h sesiones (F del −3x y G del +3x). Con
    rendimientos normales e independientes, mediana/media = exp(−(L²/2)·σ²·h), el mismo con L = −3 y
    L = +3; la asimetría del índice los separa (su cola izquierda es la cola derecha del −3x)."""
    rows = []
    for name, x in inputs.items():
        for h in H_LIST:
            S = series[(name, h)]
            F, G = S["F"], S["G"]
            rows.append({"serie": name, "h": h, "media F − 1": F.mean() - 1, "mediana F − 1": np.median(F) - 1,
                         "media G − 1": G.mean() - 1, "mediana G − 1": np.median(G) - 1,
                         "mediana/media F": np.median(F) / F.mean(), "mediana/media G": np.median(G) / G.mean(),
                         "exp(−4,5·σ²·h)": float(np.exp(-4.5 * S["var_u"] * h)),
                         "media del índice (e^X − 1)": float(np.expm1(S["X"]).mean())})
    return pd.DataFrame(rows).set_index(["serie", "h"])


# ───────────────────────────── cálculo completo ─────────────────────────────

def compute(D: dict | None = None, B_: int = B) -> dict:
    """Todo el cálculo de R31 (sin informe). Lee la caché; no descarga nada (ip.download lo hace)."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    frames = ip.load_pairs()
    hist = ip.synthetic_history(D)
    rng = np.random.default_rng(SEED)
    H1 = h1_real(frames, rng)
    H1s = h1_synthetic(hist, np.random.default_rng(SEED + 1))
    inputs = h2_inputs(frames, hist)
    t2, series = h2_table(inputs, B_)
    return {"frames": frames, "hist": hist, "H1": H1, "H1s": H1s, "inputs": inputs, "H2": t2, "series": series,
            "moments": fund_moments(inputs, series), "dataset": ip.dataset_table(frames),
            "borrow": ip.borrow_grid()}


# ───────────────────────────── figuras ─────────────────────────────

def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _num_axis(ax, axis: str = "y", d: int | None = None):
    def f(v, _):
        k = d if d is not None else (0 if abs(v) >= 1 or v == 0 else (1 if abs(v) >= 0.1 else 2))
        return num_es(v, k)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(plt.FuncFormatter(f))


def short_name(name: str) -> str:
    return {SYN[SP]: "S&P 500 sintético", SYN[NDX]: "Nasdaq-100 sintético"}.get(name, name)


def fig_h1_scatter(C: dict, rep: Report):
    yr = C["H1"]["years"]
    pre = set(p.key for p in ip.prereg_pairs())
    yr = yr[yr.par.isin(pre)]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.8), sharex=True, sharey=True)
    lost = {}
    for ax, col, title in ((axes[0], "arrastre literal", "Tal como está escrita: con el carry"),
                           (axes[1], "arrastre D", "Neto del carry y la fricción (D)")):
        n_out = 0
        for L in (-3, -2, -1, 2, 3):
            q = yr[yr.L == L]
            x, y = q["fórmula"].to_numpy() * 100, q[col].to_numpy() * 100
            ok = (x > 0) & (y > 0)
            n_out += int((~ok).sum())
            ax.scatter(x[ok], y[ok], s=18, color=L_COL[L], marker=L_MARK[L], alpha=0.8, edgecolor="white", lw=0.4)
        lost[col] = (n_out, len(yr))
        lim = (0.2, 500)
        ax.plot(lim, lim, color=MUTED, lw=1.0)
        ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("½(L² − L)·Σr² (puntos log)")
        _num_axis(ax, "x")
        _num_axis(ax, "y")
        _style(ax)
    axes[0].set_ylabel("Arrastre realizado (puntos log)")
    h = [plt.Line2D([], [], color=L_COL[L], marker=L_MARK[L], ls="", ms=7) for L in (-3, -2, -1, 2, 3)]
    legend_band(fig, h, [f"L = {lx(L)}" for L in (-3, -2, -1, 2, 3)], where="bottom", ncol=5, fontsize=11)
    n_lit, n_all = lost["arrastre literal"]
    rep.fig(fig, "arrastre_formula",
            "Figura: cada punto es un año natural de un fondo real de los pares del pre-registro (2006–2026). "
            "Horizontal: la fórmula ½(L² − L)·Σr² con los rendimientos diarios del índice; vertical: el arrastre "
            "realizado. Izquierda, tal como lo escribe el pre-registro: L veces el log-rendimiento del índice menos el "
            "del fondo, que lleva dentro el carry (letra − L·f) y la fricción; "
            + (f"{n_lit} de los {n_all} años-fondo tienen un arrastre literal ≤ 0 y no caben en la escala logarítmica "
               "(años de tipos altos y poca volatilidad, en los que el carry del inverso supera a su arrastre). "
               if n_lit else "")
            + "Derecha: el arrastre D, neto del carry y de la fricción. La diagonal es la igualdad; escalas "
            "logarítmicas. Los puntos de la derecha claramente por debajo de la diagonal son sobre todo de FAS y FAZ, "
            "medidos contra XLF: probablemente su índice de referencia no fue el de XLF durante parte de su vida (sin "
            "verificar)")


def fig_h1_years(C: dict, rep: Report):
    ann = C["H1s"]["ann"]
    yr = C["H1"]["years"]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 7.8), sharex=True)
    ax = axes[0]
    for L in (-3, 3):
        a = ann[(SP, L)]
        ax.plot(a.index, a.D * 100, color=L_COL[L], lw=1.6, marker=L_MARK[L], ms=3.5, label=f"{lx(L)} sintético")
    for tk, L in (("SPXU", -3), ("UPRO", 3)):
        q = yr[yr.fondo == tk]
        ax.scatter(q["año"], q["arrastre D"] * 100, s=46, facecolor="white", edgecolor=L_COL[L], lw=1.6, zorder=3,
                   label=f"{tk} real")
    ax.set_yscale("log")
    _num_axis(ax, "y")
    ax.set(ylabel="Arrastre anual D (puntos log)", title="Arrastre de cada año: −3x y +3x del S&P 500")
    ax.legend(loc="upper right", fontsize=10, frameon=False, ncol=2, columnspacing=1.2)
    _style(ax)
    ax = axes[1]
    YL = (1.2, 3.3)
    out = []
    for mkt, col in ((SP, INK), (NDX, MUTED)):
        r = (ann[(mkt, -3)].D / ann[(mkt, 3)].D).dropna()
        ax.plot(r.index, r.values, color=col, lw=1.4, label=f"{short_name(SYN[mkt])}")
    pre3 = [p for p in ip.prereg_pairs() if p.L == 3]
    first = True
    for p in pre3:
        q = yr[yr.par == p.key].pivot_table(index="año", columns="L", values="arrastre D")
        if -3 in q and 3 in q:
            rr = (q[-3] / q[3]).dropna()
            ax.scatter(rr.index, rr.clip(upper=YL[1]), s=14, color=ORANGE, alpha=0.8, zorder=3,
                       label="pares 3x reales del pre-registro" if first else None)
            out += [f"{p.key} en {y_} ({num_es(v_, 1)})" for y_, v_ in rr[rr > YL[1]].items()]
            first = False
    ax.axhline(2.0, color=BLUE, lw=1.2, ls="--", label="fórmula: 6/3 = 2")
    for mkt in (SP, NDX):
        r = (ann[(mkt, -3)].D / ann[(mkt, 3)].D).dropna()
        out += [f"{short_name(SYN[mkt])} en {y_} ({num_es(v_, 2)})" for y_, v_ in r[r < YL[0]].items()]
    ax.set(ylim=YL, ylabel="D(−3x) / D(+3x)", title="Cociente de los arrastres del mismo año")
    ax.set_yticks(np.arange(1.2, 3.31, 0.2))
    _num_axis(ax, "y", 1)
    ax.legend(loc="upper left", fontsize=10, frameon=False, ncol=2)
    _style(ax)
    rep.fig(fig, "arrastre_por_año",
            "Figura: arriba, el arrastre de cada año natural (puntos logarítmicos, escala logarítmica) del −3x y del +3x "
            "sintéticos del S&P 500 desde 1928, con SPXU y UPRO reales (círculos) desde 2009. Abajo, el cociente entre "
            "el arrastre del −3x y el del +3x en el mismo año: la fórmula da 2; las líneas son los sintéticos del S&P 500 "
            "y del Nasdaq-100 y los puntos, los siete pares 3x reales del pre-registro, con el arrastre neto D (los más "
            "alejados de 2 son de FAZ/FAS, medidos contra XLF, que probablemente no es su índice de referencia en todo el "
            "periodo; sin verificar). Fuera de escala: " + y_join(out) + ".")


def fig_h2_diffs(C: dict, rep: Report):
    t = C["H2"]
    t = t[t.familia & (t.variante == MAIN)]
    names = list(dict.fromkeys(t.serie))
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 8.4), sharey=True)
    for ax, h in zip(axes.flat, H_LIST):
        q = t[t.h == h].set_index("serie").reindex(names)
        y = np.arange(len(names))[::-1]
        for col, lo, hi, off, c, mk, face in (("Δ mediana", "Δ mediana IC90_bajo", "Δ mediana IC90_alto", 0.17, BLUE, "o", BLUE),
                                              ("Δ media", "Δ media IC90_bajo", "Δ media IC90_alto", -0.17, ORANGE, "s", "white")):
            ax.hlines(y + off, q[lo] * 100, q[hi] * 100, color=c, lw=2.0)
            ax.scatter(q[col] * 100, y + off, s=42, marker=mk, facecolor=face, edgecolor=c, lw=1.6, zorder=3)
        ax.axvline(0, color=MUTED, lw=0.9)
        ax.set_yticks(np.arange(len(names))[::-1])
        ax.set_yticklabels([short_name(n) for n in names], fontsize=11)
        ax.set(title=f"h = {h} sesiones", xlabel="corto − +3x (pp)")
        _num_axis(ax, "x")
        _style(ax)
    h_ = [plt.Line2D([], [], color=BLUE, marker="o", lw=2.0, ms=7),
          plt.Line2D([], [], color=ORANGE, marker="s", markerfacecolor="white", lw=2.0, ms=7)]
    legend_band(fig, h_, ["diferencia de medianas, con su IC90", "diferencia de medias, con su IC90"], where="bottom",
                ncol=2, fontsize=11)
    fig.suptitle("Corto del −3x frente al +3x comprado, s = 1")
    rep.fig(fig, "mediana_frente_a_media",
            "Figura: rentabilidad de mantener h sesiones el corto del −3x (s = 1, con el mantenimiento FINRA, el "
            "préstamo mediano de cada ticker en IBKR y el producto a la letra − 0,5%) menos la del +3x comprado con el "
            "100%, en todas las ventanas de cada serie. Círculos: la mediana del corto menos la mediana del +3x; "
            "cuadrados: la diferencia de medias. Las barras son el IC90 del bootstrap estacionario de las ventanas "
            "(bloque medio 63)")


def fig_payoff(C: dict, rep: Report):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.9))
    v = PURE
    S = C["series"][(SYN[SP], 63)]
    X, sh, lg = S["X"], S[v]["corto"], S[v]["largo"]
    Qm = float(np.median(S["Q"]))
    k = slice(None, None, 5)
    ax = axes[0]
    ax.scatter(X[k] * 100, lg[k] * 100, s=5, color=ORANGE, alpha=0.30, lw=0)
    ax.scatter(X[k] * 100, sh[k] * 100, s=5, color=BLUE, alpha=0.30, lw=0)
    xs = np.linspace(np.percentile(X, 0.1), np.percentile(X, 99.9), 200)
    ax.plot(xs * 100, np.expm1(3 * xs - 3 * Qm) * 100, color=ORANGE, lw=2.2)
    ax.plot(xs * 100, -np.expm1(-3 * xs - 6 * Qm) * 100, color=BLUE, lw=2.2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set(ylim=(-105, 230), xlabel="Log-rendimiento del índice, X (%)", ylabel="Rentabilidad de la ventana (%)",
           title="63 sesiones: el pago")
    _style(ax)
    ax = axes[1]
    S2 = C["series"][(SYN[SP], 252)]
    bins = np.linspace(-100, 300, 81)
    for arr, c in ((S2[v]["largo"], ORANGE), (S2[v]["corto"], BLUE)):
        ax.hist(np.clip(arr * 100, -100, 300), bins=bins, density=True, histtype="step", color=c, lw=1.8)
        ax.axvline(np.median(arr) * 100, color=c, lw=1.6)
        ax.axvline(arr.mean() * 100, color=c, lw=1.6, ls="--")
    sh2, lg2 = S2[v]["corto"], S2[v]["largo"]
    ax.set(xlabel="Rentabilidad de la ventana (%)", ylabel="Densidad", title="252 sesiones: la distribución")
    ax.set_yticks([])
    _style(ax)
    for a in axes:
        _num_axis(a, "x")
    _num_axis(axes[0], "y")
    h = [plt.Line2D([], [], color=BLUE, lw=2.2), plt.Line2D([], [], color=ORANGE, lw=2.2),
         plt.Line2D([], [], color=INK, lw=1.6), plt.Line2D([], [], color=INK, lw=1.6, ls="--")]
    legend_band(fig, h, ["corto del −3x (s = 1)", "+3x comprado", "mediana", "media"], where="bottom", ncol=4,
                fontsize=11)
    rep.fig(fig, "pago_de_opcion",
            "Figura: S&P 500 sintético, 1928–2026, corto sin mantenimiento (el pago puro, con el préstamo mediano de "
            "SPXU, el producto a la letra − 0,5% y la ruina) y +3x comprado. "
            "Izquierda: cada punto es una ventana de 63 sesiones (una de cada cinco); las curvas son las aproximaciones "
            "1 − exp(−3X − 6Q) del corto y exp(3X − 3Q) − 1 del +3x con la Q mediana. El corto gana como mucho el "
            "100% y su pérdida no tiene techo (en la simulación la ruina la deja en −100%, porque el capital no baja de "
            "cero; en una cuenta real la pérdida puede superar el capital); el +3x pierde como mucho el 100% y gana sin "
            "límite. Derecha: la distribución de "
            "las ventanas de 252 sesiones (acotada en 300%): la mediana del corto (" + pct_es(np.median(sh2), 0) + ") queda "
            "a la derecha de la del +3x (" + pct_es(np.median(lg2), 0) + ") y su media (" + pct_es(sh2.mean(), 0) + "), a la "
            "izquierda de la del +3x (" + pct_es(lg2.mean(), 0) + "); la barra de −100% son las ventanas en ruina del corto ("
            + pct_es(S2[v]["ruina"].mean(), 0) + ")")


# ───────────────────────────── informe ─────────────────────────────

def y_join(items) -> str:
    items = list(items)
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


def _pp_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype(float) * 100
    return df


def run(C: dict | None = None):
    warnings.filterwarnings("ignore")
    rep = Report("r31_inverse_drag", "R31 — El decaimiento del inverso (H1 y H2 del artículo 6)")
    for f in list(rep.dir.glob("tabla_*.csv")) + list(rep.dir.glob("*_completo.csv")):
        f.unlink()
    C = C or compute()
    tab = Tables(rep)
    TN = {}
    H1, H1s, t2, mom = C["H1"], C["H1s"], C["H2"], C["moments"]
    fits, ratios, dec, yrs = H1["fits"], H1["ratios"], H1["decomp"], H1["years"]
    bg = C["borrow"]

    # ── H1: cifras del resumen
    inv_pre = fits[fits["pre-registro"] & (fits.lado == "inverso")].set_index("fondo")
    n_inv = len(inv_pre)
    lit_ok, lit_bad = inv_pre[inv_pre["cumple literal"]], inv_pre[~inv_pre["cumple literal"]]
    net_ok, net_bad = inv_pre[inv_pre["cumple neto del carry"]], inv_pre[~inv_pre["cumple neto del carry"]]
    ok_inv = int(inv_pre.cumple.sum())
    bad_inv, good = inv_pre[~inv_pre.cumple], inv_pre[inv_pre.cumple]
    rat_pre = ratios[ratios["pre-registro"]].set_index("par")
    n_rat = len(rat_pre)
    dbl_ok, dbl_bad = rat_pre[rat_pre["cumple el doble"]], rat_pre[~rat_pre["cumple el doble"]]
    rat2 = rat_pre[rat_pre["|L|"] == 2]
    n_dbl_lit = int(rat_pre["cumple el doble literal"].sum())
    n_dbl_net = int(rat_pre["cumple el doble neto del carry"].sum())
    no_ratio = [p.key for p in ip.prereg_pairs() if p.long is None]
    lng_pre = fits[fits["pre-registro"] & (fits.lado == "alcista")].set_index("fondo")
    sq = dec.set_index("fondo")
    sfit = H1s["fits"]
    srat = H1s["ratios"].set_index(["historia", "|L|"])
    ann = H1s["ann"]
    r87 = float(ann[(SP, -3)].D.loc[1987] / ann[(SP, 3)].D.loc[1987])
    fz = yrs[yrs.par == "FAZ/FAS"]
    fz_old, fz_new = fz[fz["año"] <= 2020]["fricción φ"], fz[fz["año"] >= 2021]["fricción φ"]
    fzi = fz[fz.fondo == "FAZ"].set_index("año")
    fz_ratio = fzi["arrastre D"] / fzi["fórmula"]
    fz_r_old, fz_r_new = fz_ratio[fz_ratio.index <= 2020], fz_ratio[fz_ratio.index >= 2021]
    carry_sq = yrs[yrs.fondo == "SQQQ"].set_index("año")["carry"]
    slope_net = inv_pre["pendiente neta del carry"]
    net_bad_txt = y_join(f"{k} (pendiente {num_es(v, 2)})" for k, v in net_bad["pendiente neta del carry"].items())
    phi_inv = dec[dec["pre-registro"] & (dec.L < 0)]["− fricción φ"].mul(-1)
    phi_lng = dec[dec["pre-registro"] & (dec.L > 0)]["− fricción φ"].mul(-1)

    # ── H2: cifras del resumen
    fam = t2[t2.familia]
    main = fam[fam.variante == MAIN]
    cnt = fam.groupby(["variante", "h"]).H2.value_counts().unstack(fill_value=0)
    for c in ("se cumple", "falla: media", "falla: mediana", "falla: las dos"):
        if c not in cnt:
            cnt[c] = 0

    def n_ok(v, h=None):
        q = fam[fam.variante == v]
        return int((q[q.h == h] if h is not None else q).H2.eq("se cumple").sum())

    def mean_up(v, h):
        q = fam[(fam.variante == v) & (fam.h == h)]
        return q[(q["Δ media"] > 0) & (q["p media"] < ALPHA)]

    n_ok2 = n_ok(MAIN)
    by_h = {h: main[main.h == h] for h in H_LIST}
    ok_h = {h: n_ok(MAIN, h) for h in H_LIST}
    ns = len(by_h[5])
    grid_tot = {v: n_ok(v) for v in GRID_V}
    lo_n, hi_n = min(grid_tot.values()), max(grid_tot.values())

    def short_v(v):
        return v.replace("préstamo ", "").replace("producto ", "")

    lo_txt = y_join(short_v(v) for v in GRID_V if grid_tot[v] == lo_n)
    hi_txt = y_join(short_v(v) for v in GRID_V if grid_tot[v] == hi_n)
    m_up = main[(main["Δ media"] > 0) & (main["p media"] < ALPHA)]
    m_up_txt = y_join(f"{short_name(r.serie)} a {r.h} sesiones" for _, r in m_up.iterrows()) or "ninguna"
    n_med_fail = int(main.H2.isin(["falla: mediana", "falla: las dos"]).sum())
    nob5 = mean_up(NOBORROW, 5)
    q_nob5 = fam[(fam.variante == NOBORROW) & (fam.h == 5)]
    fee_ratio = float((q_nob5["Δ media"] / q_nob5["comisiones − préstamo en h"]).median())
    ok_nob = {h: n_ok(NOBORROW, h) for h in H_LIST}
    ok_pure = {h: n_ok(PURE, h) for h in H_LIST}
    ok_third = {h: n_ok(THIRD, h) for h in H_LIST}
    pure = fam[fam.variante == PURE].set_index(["serie", "h"])
    mainx = main.set_index(["serie", "h"])
    spn, ndn = SYN[SP], SYN[NDX]
    p_sp = pure.loc[(spn, 252)]
    p_sp21 = pure.loc[(spn, 21)]
    mo = mom.loc[spn]
    d252 = by_h[252]["Δ media"]
    s252 = (f"; la media del corto queda entre {num_es(-d252.max() * 100, 0)} y {num_es(-d252.min() * 100, 0)} pp por "
            "debajo de la del +3x") if (d252 < 0).all() else ""
    same_long = bool((main["H2"] == main["H2 (bloque largo)"]).all())
    same_holm = bool((main["H2"] == main["H2 (Holm)"]).all())
    diff_long = main[main["H2"] != main["H2 (bloque largo)"]]
    pure_up = {h: len(mean_up(PURE, h)) for h in H_LIST}

    def med_up_short(v):
        q = fam[(fam.variante == v) & (fam.h <= 63)]
        return int(q["mediana a favor del corto"].sum())

    med63_pure, med63_third = med_up_short(PURE), med_up_short(THIRD)

    rep.p(__doc__.split("\n", 2)[2])
    rep.h("Resumen")
    rep.p("\n".join([
        f"- **H1, primera parte (la fórmula), no se cumple tal como está escrita: con el arrastre literal cumple en "
        f"{len(lit_ok)} de los {n_inv} inversos del pre-registro ({y_join(lit_ok.index) or 'ninguno'}).** El arrastre "
        "literal es lo que el fondo rinde de menos que L veces el log-rendimiento del índice, L·Σlog(1 + u) − Σlog(1 + R). "
        "Lleva dentro el carry (letra − L·f; en un −3x, la letra más tres veces el tipo de financiación), que no depende de "
        f"la volatilidad y se mueve con los tipos de interés: el de SQQQ fue de {pp(carry_sq.min(), 1)} a "
        f"{pp(carry_sq.max(), 1)} pp al año. En los otros {len(lit_bad)} inversos la pendiente va de "
        f"{num_es(lit_bad['pendiente literal'].min(), 2)} a {num_es(lit_bad['pendiente literal'].max(), 2)} y el R², de "
        f"{num_es(lit_bad['R2 literal'].min(), 2)} a {num_es(lit_bad['R2 literal'].max(), 2)} (tabla «T:fits»).",
        f"- **La mecánica sí se cumple, pero solo neta del carry y de la fricción**, una lectura que el pre-registro no "
        f"fija. Con el arrastre neto de los dos, D = Σ[(R − log(1 + R)) − L·(u − log(1 + u))], cumplen {ok_inv} de "
        f"{n_inv}: pendiente de {num_es(good.pendiente.min(), 2)} a {num_es(good.pendiente.max(), 2)} y R² de al menos "
        f"{num_es(good.R2.min(), 3)}"
        + (f"; falla {y_join(f'{k} (pendiente {num_es(r.pendiente, 2)}, R² {num_es(r.R2, 3)})' for k, r in bad_inv.iterrows())}"
           if len(bad_inv) else "")
        + f". Neta solo del carry (D + φ, con la fricción dentro) cumplen {len(net_ok)} de {n_inv}"
        + (f"; fallan {net_bad_txt}" if len(net_bad) else "")
        + ". Con la identidad, el log-crecimiento de SQQQ fue de "
        f"{pp(sq.loc['SQQQ', 'g fondo'])} puntos al año en 2010–2026: {pp(sq.loc['SQQQ', 'L × g índice'])} de −3 veces el "
        f"Nasdaq-100, {pp(sq.loc['SQQQ', 'letra − L·f'])} de carry, {pp(sq.loc['SQQQ', '− fricción φ'])} de fricción y "
        f"{pp(sq.loc['SQQQ', '− arrastre D'])} de arrastre (la fórmula da {pp(-sq.loc['SQQQ', 'fórmula ½(L²−L)Σu²'])}); "
        f"TQQQ, {pp(sq.loc['TQQQ', '− arrastre D'])} de arrastre.",
        f"- **H1, segunda parte («aproximadamente el doble»), tal como está escrita (cociente de 1,8 a 2,2), se cumple en "
        f"{len(dbl_ok)} de los {n_rat} pares con alcista del mismo |L|** con el arrastre neto D "
        f"({y_join(dbl_ok.index) or 'ninguno'}) y falla en "
        f"{y_join(f'{k} ({num_es(r.cociente, 2)})' for k, r in dbl_bad.iterrows()) or 'ninguno'}"
        + (". Los pares 2x fallan porque la fórmula no da el doble sino el triple, (|L| + 1)/(|L| − 1) = 3, y el triple es "
           "lo que sale" if len(rat2) and rat2.cumple.all() and not rat2["cumple el doble"].any() else "")
        + (". FAZ/FAS se aparta del doble y de la fórmula" if "FAZ/FAS" in dbl_bad.index else "")
        + f". {y_join(no_ratio)} no se pueden contrastar: su alcista es el propio índice, que no tiene arrastre. Con el "
        f"arrastre literal el doble se cumple en {n_dbl_lit} de {n_rat} (cocientes de "
        f"{num_es(rat_pre['cociente literal'].min(), 2)} a {num_es(rat_pre['cociente literal'].max(), 2)}: el carry resta "
        f"al del inverso y suma al del alcista), y neto solo del carry, en {n_dbl_net}. En la historia sintética el "
        f"cociente de D es {num_es(srat.loc[(spn, 3), 'cociente'], 2)} (S&P 500, 1928–2026) y "
        f"{num_es(srat.loc[(ndn, 3), 'cociente'], 2)} (Nasdaq-100); en un año con un crac se aleja: {num_es(r87, 2)} en "
        "1987, porque el término de orden 3 castiga al +3x en la caída y alivia al −3x.",
        f"- **La fricción es del mismo orden en los dos lados.** La de los inversos del pre-registro va de "
        f"{pp(phi_inv.min())} a {pp(phi_inv.max())} pp al año y la de los alcistas de {pp(phi_lng.min())} a "
        f"{pp(phi_lng.max())}, mayor en los años de más volatilidad: netas solo del carry, las pendientes de los inversos "
        f"van de {num_es(slope_net.min(), 2)} a {num_es(slope_net.max(), 2)} (tabla «T:fits»).",
        "- **Contra su propia media, el −3x no decae más que el +3x.** El 6 frente al 3 sale de medir contra L veces el "
        "log-rendimiento del índice, que ya lleva dentro el −σ²/2 del índice multiplicado por L: con L < 0 ese término "
        "suma. Frente a su propia rentabilidad esperada, en la aproximación normal los dos fondos pierden en la mediana "
        "½L²σ²h = 4,5σ²h en h sesiones (σ², la varianza diaria del índice). Por eso el «decaimiento» del inverso no es una "
        "prima que el corto cobre en la media (sección 3.3).",
        f"- **H2 no se cumple como está escrita: se cumple en {n_ok2} de las {len(main)} combinaciones** de la familia "
        f"(los 7 pares 3x del pre-registro y las dos historias sintéticas × 4 horizontes) con el corto de referencia: "
        f"s = 1, mantenimiento FINRA, préstamo mediano de cada ticker en IBKR y producto a la letra − 0,5%. Por "
        f"horizonte, en {ok_h[5]}, {ok_h[21]}, {ok_h[63]} y {ok_h[252]} de {ns} series a 5, 21, 63 y 252 sesiones. En los "
        f"seis puntos de la rejilla del pre-registro (préstamo mínimo, mediano o p90; producto remunerado o no) el "
        f"recuento va de {lo_n} ({lo_txt}) a {hi_n} ({hi_txt}) de 36 (tabla «T:counts»). Falla casi "
        f"siempre por la mediana ({n_med_fail} de las {len(main) - n_ok2} que fallan): la del corto no supera a la del +3x "
        f"con p de Romano–Wolf < 0,05. La media del corto es significativamente mayor que la del +3x solo en {m_up_txt}. "
        f"A 252 sesiones la mediana del corto no supera de forma significativa a la del +3x en ninguna serie (en "
        f"{int((by_h[252]['Δ mediana'] < 0).sum())} es menor){s252}. Con s = 1 el mantenimiento obliga a recomprar tras "
        "cada subida del −3x de un 5,3%, y en las series con tendencia fuerte la convexidad del +3x hace el resto.",
        f"- **Sin préstamo** (fuera de la rejilla; se da solo como referencia, porque aísla las comisiones) el corto gana "
        f"también en la media a 5 sesiones en {len(nob5)} de {ns} series: es lo que cobran de comisiones los dos fondos "
        f"(la mediana del cociente entre la ventaja media y esas comisiones es {num_es(fee_ratio, 2)}). El préstamo "
        f"mediano se come esa ventaja: con él, la media es significativamente mayor a 5 sesiones en "
        f"{len(mean_up(MAIN, 5))} de {ns}. Sin préstamo, H2 se cumpliría en {ok_nob[5]}, {ok_nob[21]}, {ok_nob[63]} y "
        f"{ok_nob[252]} de {ns} series.",
        f"- **El pago puro, sin mantenimiento** (con el préstamo mediano y el producto remunerado), cumple H2 en "
        f"{ok_pure[5]}, {ok_pure[21]}, {ok_pure[63]} y {ok_pure[252]} de {ns} series a 5, 21, 63 y 252 sesiones; donde "
        f"falla a horizontes cortos es por la media ({pure_up[5]} y {pure_up[21]} series a 5 y 21 sesiones), que cobra "
        "además la reversión del índice desde 2000 (el cociente de varianzas menor que 1 de R29). "
        f"En el S&P 500 sintético (1928–2026) a 252 sesiones la mediana del corto supera a la del +3x en "
        f"{pp(p_sp['Δ mediana'])} pp (p de Romano–Wolf {peq(p_sp['p mediana RW'])}) y su media queda "
        f"{num_es(-p_sp['Δ media'] * 100, 1)} pp por debajo; a cambio, {pct_es(p_sp['ruinas'] / p_sp['ventanas'], 0)} de "
        f"las ventanas acaban en ruina y la asimetría del corto es {num_es(p_sp['asimetría corto'], 2)} frente a "
        f"{num_es(p_sp['asimetría +3x'], 2)} del +3x. Es vender la cola derecha del inverso: en su 5% de peores ventanas "
        f"el corto devuelve {num_es(-p_sp['cola 5% del corto'] * 100, 1)} pp de su media de "
        f"{num_es(p_sp['media corto'] * 100, 1)} pp.",
    ]))

    rep.h("1. Datos: los pares y la historia larga")
    dsv = C["dataset"].copy()
    b = bg[["mín", "mediana", "p90"]].rename(columns=lambda c: f"préstamo {c} (%)")
    dsv = dsv.join(b, on=dsv.index.map(lambda k: ip.BY_KEY[k].inverse)) if len(b) else dsv
    dsv = dsv.drop(columns=["key_0"], errors="ignore")
    tab("Pares reales (inverse_pairs.PAIRS): vida común de los fondos y del ETF 1x que hace de índice (Yahoo, "
        "adj_close, hasta el 25-9-2026), días con dato erróneo sustituidos por L·u, gasto del folleto cuando el "
        "laboratorio lo tiene (universe.py) y préstamo del inverso en IBKR (iBorrowDesk, ~12 meses: mínimo, mediana "
        "y p90 de la tasa diaria, % anual)", dsv, pct={"gasto inverso", "gasto alcista"}, digits=2)
    hs = []
    for mkt, df in C["hist"].items():
        for L in (-3, -2, -1, 2, 3):
            hs.append({"historia": SYN[mkt], "L": L, "desde": df.index[0].date(), "hasta": df.index[-1].date(),
                       "sesiones": len(df), "s (sobrecoste anual)": df.attrs["s"][L], "gasto": df.attrs["gasto"][L]})
    tab("Historia larga sintética (inverse_pairs.synthetic_history): fondos de R22, rf + L·(u − f) − |L|·s − gasto, "
        "con el s calibrado con la vida de cada ETF real (el 3x, el del laboratorio)", pd.DataFrame(hs).set_index(["historia", "L"]),
        pct={"s (sobrecoste anual)", "gasto"}, digits=4)
    rep.p("La ampliación («los que la caché y Yahoo permitan») añade SPXS/SPXL, SRTY/URTY, SMDD/UMDD, TECS/TECL, "
          "TWM/UWM, DXD/DDM, MZZ/MVV, DOG, RWM, los bonos TMV/TMF y TBT/UBT, y ERY/ERX y DUST/NUGT partidos en su tramo "
          "3x (hasta el 31-3-2020) y 2x (desde el 1-4-2020). Se dan aparte y no entran en los veredictos. El ETF 1x es "
          "una aproximación del índice de cada fondo: donde no es el mismo índice (probablemente FAS y FAZ frente a "
          "XLF durante parte de su vida; los semiconductores antes de 2021), su diferencia entra en el arrastre como "
          "ruido.")

    rep.h("2. H1: el arrastre del reajuste diario")
    rep.p("Un fondo de reajuste diario con apalancamiento L rinde cada día R = letra + L·(u − f) − φ, con u el "
          "rendimiento del índice, f el tipo al que se financia la exposición y φ su fricción (gasto, sobrecoste del "
          "swap y error de seguimiento). Sumando los logaritmos de un año:")
    rep.p("Σ log(1 + R) = L·Σ log(1 + u) + Σ(letra − L·f) − Σφ − D,   con   D = Σ[(R − log(1 + R)) − L·(u − log(1 + u))] "
          "≈ ½(L² − L)·Σu².")
    rep.p("*Pie de fórmula: lo que el fondo crece en logaritmos es L veces lo que crece el índice, más el carry, menos la "
          "fricción, menos el arrastre D, que es lo que se pierde por reajustar cada día y crece con el cuadrado de los "
          "movimientos diarios.* Notación: R, rendimiento diario del fondo; u, del índice (su ETF 1x); L, el "
          "apalancamiento (−3 en SQQQ, +3 en TQQQ); f, el tipo de financiación (fed funds); φ, la fricción aritmética; "
          "Σ, la suma sobre las sesiones del año. ½(L² − L) vale 6 con L = −3, 3 con L = 3, 3 con L = −2, 1 con L = 2 y "
          "1 con L = −1. D no depende de f ni del gasto: se mide solo con R y u.")
    rep.p("El pre-registro mide el arrastre «frente a L veces el rendimiento logarítmico del índice». Tal como está "
          "escrito, es L·Σlog(1 + u) − Σlog(1 + R) = D + Σφ − Σ(letra − L·f): el arrastre literal, con el carry y la "
          "fricción dentro. H1 se juzga con él. Se dan además dos lecturas que el pre-registro no fija: neto del carry "
          "(D + φ) y neto del carry y de la fricción (D, la que mide solo el reajuste).")
    fv = fits[fits["pre-registro"]].set_index(["par", "fondo"]).copy()
    fv["IC90 literal"] = [f"[{num_es(a, 3)}; {num_es(b_, 3)}]" for a, b_ in zip(fv["IC90 literal bajo"], fv["IC90 literal alto"])]
    fv["IC90"] = [f"[{num_es(a, 3)}; {num_es(b_, 3)}]" for a, b_ in zip(fv.IC90_bajo, fv.IC90_alto)]
    fv = _pp_cols(fv, ["constante"])
    for c in ("cumple literal", "cumple neto del carry", "cumple"):
        fv[c] = fv[c].map({True: "sí", False: "NO"})
    fv = fv[["L", "años", "pendiente literal", "IC90 literal", "R2 literal", "cumple literal", "pendiente neta del carry",
             "R2 neto del carry", "cumple neto del carry", "pendiente", "IC90", "constante", "R2", "cumple"]].rename(columns={
        "pendiente literal": "literal: pendiente", "IC90 literal": "literal: IC90", "R2 literal": "literal: R²",
        "cumple literal": "literal: cumple (H1 como está escrita)", "pendiente neta del carry": "neto del carry: pendiente",
        "R2 neto del carry": "neto del carry: R²", "cumple neto del carry": "neto del carry: cumple",
        "pendiente": "D: pendiente", "IC90": "D: IC90", "constante": "D: constante (pp)", "R2": "D: R²", "cumple": "D: cumple"})
    tab("H1 en los pares del pre-registro: regresión del arrastre anual sobre ½(L² − L)·Σu² (años naturales con al "
        "menos 63 sesiones; MCO con constante y errores de White) con las tres medidas: literal, L·Σlog(1 + u) − "
        "Σlog(1 + R) (la del pre-registro); neta del carry, D + φ; y neta del carry y de la fricción, D. Cumple si la "
        "pendiente está entre 0,9 y 1,1 y R² > 0,9. El veredicto de H1 se da con el inverso", fv, digits=3)
    TN["fits"] = tab.n
    rv = rat_pre.copy()
    rv["IC90"] = [f"[{num_es(a, 2)}; {num_es(b_, 2)}]" for a, b_ in zip(rv.IC90_bajo, rv.IC90_alto)]
    rv = _pp_cols(rv, ["ΣD inverso", "ΣD alcista"])
    for c in ("cumple el doble", "cumple", "cumple el doble literal", "cumple el doble neto del carry"):
        rv[c] = rv[c].map({True: "sí", False: "NO"})
    rv = rv[["|L|", "años", "ΣD inverso", "ΣD alcista", "cociente", "IC90", "cumple el doble", "fórmula", "cumple",
             "cociente neto del carry", "cumple el doble neto del carry", "cociente literal", "cumple el doble literal"]].rename(
        columns={"ΣD inverso": "ΣD inverso (puntos log)", "ΣD alcista": "ΣD alcista (puntos log)", "cociente": "cociente de D",
                 "cumple el doble": "D: el doble (1,8–2,2)", "fórmula": "fórmula (|L| + 1)/(|L| − 1)",
                 "cumple": "D: ±10% de la fórmula", "cumple el doble neto del carry": "neto del carry: el doble",
                 "cumple el doble literal": "literal: el doble"})
    tab("H1, segunda parte: arrastre acumulado del inverso y del alcista con el mismo |L| en los mismos años y su "
        "cociente (IC90 remuestreando años). Tal como está escrita, «aproximadamente el doble»: entre 1,8 y 2,2. Se da "
        "también frente a la fórmula, (|L| + 1)/(|L| − 1), que vale 2 a 3x y 3 a 2x, y con las otras dos medidas. SH y "
        "PSQ no tienen alcista con el mismo |L| distinto del índice", rv, digits=3)
    TN["ratios"] = tab.n
    dv = dec[dec["pre-registro"]].set_index(["par", "fondo"]).drop(columns=["pre-registro"])
    cols = ["g fondo", "L × g índice", "letra − L·f", "− fricción φ", "− arrastre D", "fórmula ½(L²−L)Σu²"]
    dv = _pp_cols(dv, cols).rename(columns={c: f"{c} (pp/año)" for c in cols})
    tab("Descomposición del log-crecimiento anual medio de cada fondo en toda su vida común con el par (pp de log al "
        "año): L veces el del índice, carry (letra − L·f), fricción aritmética y arrastre; suman exactamente el del "
        "fondo. La última columna es la fórmula del arrastre", dv, pct={"gasto del folleto"}, digits=2)
    yv = yrs[yrs.par.isin(["SQQQ/TQQQ", "SPXU/UPRO", "FAZ/FAS"])].copy()
    yp = yv.pivot_table(index=["par", "año"], columns="L", values=["arrastre literal", "arrastre D", "fórmula", "carry",
                                                                     "fricción φ"])
    yp.columns = [f"{a} {lx(b_)}" for a, b_ in yp.columns]
    vol = yv[yv.L > 0].set_index(["par", "año"])["vol índice"]
    yp.insert(0, "vol del índice", vol)
    yp["D(−3x)/D(+3x)"] = yp["arrastre D −3x"] / yp["arrastre D +3x"]
    pcols = [c for c in yp.columns if c.startswith(("arrastre", "fórmula", "fricción", "carry"))]
    yp = _pp_cols(yp, pcols).rename(columns=lambda c: c + " (pp)" if c in pcols else c)
    order_y = ["vol del índice"] + [f"{a} {lx(L)} (pp)" for L in (-3, 3) for a in
                                    ("fórmula", "arrastre D", "arrastre literal", "carry", "fricción φ")] + ["D(−3x)/D(+3x)"]
    tab("Por año: fórmula, arrastre D, arrastre literal, carry y fricción aritmética (puntos log) de SQQQ/TQQQ, "
        "SPXU/UPRO y FAZ/FAS, con la volatilidad anual del ETF 1x y el cociente de los arrastres D del mismo año. El "
        "arrastre literal es D + fricción − carry", yp[order_y], pct={"vol del índice"}, digits=1)
    TN["yrs"] = tab.n
    fig_h1_scatter(C, rep)
    why = {"SOXS": f"SOXS, cuyo arrastre es tan grande ({pp(-sq.loc['SOXS', '− arrastre D'], 0)} pp al año de media) "
                   "que el carry apenas pesa", "FAZ": "FAZ, por compensación (abajo)"}
    rep.p(f"**Lectura.** Tal como está escrita, H1 no se cumple: el arrastre literal solo sigue la fórmula en "
          f"{y_join(why.get(k, k) for k in lit_ok.index) or 'ninguno'}. En el resto, el carry, que en un −3x son unos cuatro tipos de interés (la letra más tres veces el de "
          "financiación), resta cada año al arrastre literal una cantidad que depende de los tipos y no de la "
          "volatilidad: casi nada en 2010–2015 y en 2020–2021, y unos 20 pp al año en 2023–2024 (tabla «T:yrs»). Neto "
          f"del carry y de la fricción, D sigue la fórmula año a año: en los inversos que cumplen, la pendiente va de "
          f"{num_es(good.pendiente.min(), 2)} a {num_es(good.pendiente.max(), 2)}, con R² ≥ {num_es(good.R2.min(), 3)}; la "
          "constante es de décimas de punto. "
          + (f"FAZ no cumple con D (pendiente {num_es(bad_inv.pendiente.iloc[0], 2)}): con XLF como índice, la fórmula "
             f"sobrestima su arrastre todos los años hasta 2020 (D/fórmula de {num_es(fz_r_old.min(), 2)} a "
             f"{num_es(fz_r_old.max(), 2)}) y lo acierta desde 2021 ({num_es(fz_r_new.min(), 2)} a "
             f"{num_es(fz_r_new.max(), 2)}); la fricción aritmética de FAS y FAZ va de {pp(fz_old.min(), 0)} a "
             f"{pp(fz_old.max(), 0)} pp al año hasta 2020 y de {pp(fz_new.min(), 1)} a {pp(fz_new.max(), 1)} desde 2021. "
             "El patrón es el de un fondo que seguía otro índice financiero, que se movía menos que XLF, hasta 2020; la "
             "fecha y el índice anterior no están verificados aquí. Con el arrastre literal FAZ sí cumple, porque su "
             "arrastre D queda por debajo de la fórmula y la fricción lo compensa. " if len(bad_inv) and "FAZ" in bad_inv.index else "")
          + f"En los alcistas del pre-registro la pendiente de D va de {num_es(lng_pre.pendiente.min(), 2)} a "
          f"{num_es(lng_pre.pendiente.max(), 2)}: los 2x tienen un arrastre pequeño (coeficiente 1) y el ruido pesa más, y "
          f"FAS tiene el mismo problema de índice que FAZ.")

    rep.h("2.1 La historia larga sintética", 3)
    sv = sfit.set_index(["historia", "L"])[["años", "pendiente", "IC90_bajo", "IC90_alto", "constante", "R2"]].copy()
    sv["IC90"] = [f"[{num_es(a, 3)}; {num_es(b_, 3)}]" for a, b_ in zip(sv.IC90_bajo, sv.IC90_alto)]
    sv = _pp_cols(sv.drop(columns=["IC90_bajo", "IC90_alto"]), ["constante"]).rename(
        columns={"constante": "constante (pp)", "R2": "R²"})
    sr = H1s["ratios"].set_index(["historia", "|L|"])
    sv2 = sr[["años", "cociente", "IC90_bajo", "IC90_alto", "cumple el doble", "fórmula", "cumple"]].copy()
    for c in ("cumple el doble", "cumple"):
        sv2[c] = sv2[c].map({True: "sí", False: "NO"})
    sv2 = sv2.rename(columns={"cumple el doble": "el doble (1,8–2,2)", "cumple": "±10% de la fórmula"})
    tab("Historia larga sintética: la regresión de D por L (descriptivo; los fondos siguen el modelo, así que solo "
        "mide los términos de orden 3 y superiores)", sv, digits=3)
    tab("Historia larga sintética: cociente del arrastre D del −L y del +L en los mismos años, con IC90 remuestreando "
        "años", sv2, digits=3)
    sel = []
    for yr_ in YEARS_SHOWN:
        row = {"año": yr_}
        for L in (-3, 3, -2, 2, -1):
            a = ann[(SP, L)]
            if yr_ in a.index:
                row[f"D {lx(L)} (pp)"] = a.D.loc[yr_] * 100
                row["vol del índice"] = a.vol.loc[yr_]
        row["D(−3x)/D(+3x)"] = row["D −3x (pp)"] / row["D +3x (pp)"]
        sel.append(row)
    tab("S&P 500 sintético: arrastre D de los años de los episodios del pre-registro y de los cracs (puntos log)",
        pd.DataFrame(sel).set_index("año"), pct={"vol del índice"}, digits=1)
    fig_h1_years(C, rep)
    rep.p(f"**Lectura.** En 1932 el −3x sintético del S&P 500 perdió {num_es(ann[(SP, -3)].D.loc[1932] * 100, 0)} puntos "
          f"log por el reajuste diario y el +3x {num_es(ann[(SP, 3)].D.loc[1932] * 100, 0)}; en 2008, "
          f"{num_es(ann[(SP, -3)].D.loc[2008] * 100, 0)} y {num_es(ann[(SP, 3)].D.loc[2008] * 100, 0)}; en 2024, "
          f"{num_es(ann[(SP, -3)].D.loc[2024] * 100, 1)} y {num_es(ann[(SP, 3)].D.loc[2024] * 100, 1)}. El cociente es 2 "
          f"en casi todos los años. Se aparta en los años con una sesión extrema: {num_es(r87, 2)} en 1987. El término "
          "siguiente de la serie es −(L³ − L)/3·Σu³: con u = −20% (19-10-1987) suma al arrastre del +3x y resta al del "
          "−3x. En las caídas bruscas el −3x decae menos de lo que dice la fórmula de segundo orden, justo cuando el corto "
          "pierde.")

    rep.h("2.2 Ampliación (fuera del pre-registro)", 3)
    ev = fits[~fits["pre-registro"]].set_index(["par", "fondo"])[["L", "años", "pendiente literal", "R2 literal",
                                                                  "cumple literal", "pendiente", "R2", "cumple",
                                                                  "constante neta del carry"]].copy()
    ev = _pp_cols(ev, ["constante neta del carry"]).rename(columns={
        "pendiente literal": "literal: pendiente", "R2 literal": "literal: R²", "cumple literal": "literal: cumple",
        "pendiente": "D: pendiente", "R2": "D: R²", "cumple": "D: cumple", "constante neta del carry": "fricción anual (pp)"})
    for c in ("literal: cumple", "D: cumple"):
        ev[c] = ev[c].map({True: "sí", False: "NO"})
    er = ratios[~ratios["pre-registro"]].set_index("par")[["|L|", "años", "cociente", "cumple el doble", "fórmula",
                                                           "cumple"]].copy()
    for c in ("cumple el doble", "cumple"):
        er[c] = er[c].map({True: "sí", False: "NO"})
    er = er.rename(columns={"cociente": "cociente de D", "cumple el doble": "el doble (1,8–2,2)",
                            "cumple": "±10% de la fórmula"})
    tab("Ampliación: la regresión de H1 en cada fondo, con el arrastre literal y con D (la fricción anual es la "
        "constante de D + φ)", ev, digits=3)
    tab("Ampliación: cociente de los arrastres D del inverso y del alcista", er, digits=3)
    n_ev_inv = fits[~fits["pre-registro"] & (fits.lado == "inverso")]
    bad_ev = n_ev_inv[~n_ev_inv.cumple]
    bad_el = fits[~fits["pre-registro"] & (fits.lado == "alcista") & ~fits.cumple]
    rep.p(f"**Lectura.** En la ampliación, con el arrastre literal cumplen {int(n_ev_inv['cumple literal'].sum())} de "
          f"{len(n_ev_inv)} inversos y con D, {int(n_ev_inv.cumple.sum())}"
          + (f" (falla {y_join(f'{r.fondo} en {r.par} (pendiente {num_es(r.pendiente, 2)})' for _, r in bad_ev.iterrows())})"
             if len(bad_ev) else "")
          + f"; el cociente de D está a menos de un 10% de la fórmula en {int(er['±10% de la fórmula'].eq('sí').sum())} de "
          f"{len(er)} pares. Entre los alcistas fallan con D {len(bad_el)}, sobre todo 2x, cuyo arrastre (coeficiente 1) es "
          "pequeño frente al ruido de seguimiento y a los términos de orden 3.")

    rep.h("3. H2: mediana frente a media")
    rep.p("Cada sesión abre una ventana: se vende en corto el −3x por el 100% del capital (exposición inicial +3) o se "
          "compra el +3x por el 100%, y se cierra h sesiones después. El corto no se reajusta, cobra la letra sobre el "
          "capital, paga el préstamo sobre su valor vivo y se reduce a s veces el patrimonio si incumple el mantenimiento "
          "FINRA (90% del valor del corto), lo que con s = 1 pasa en cuanto el −3x sube un 5,3% desde la última "
          "reducción. El préstamo y la remuneración del producto recorren la rejilla del pre-registro: mínimo, mediana y "
          "p90 de cada ticker en IBKR, con el producto a la letra − 0,5% o sin remunerar. La variante de referencia es "
          "la del préstamo mediano con el producto remunerado. La mediana y la media son las de todas las ventanas de "
          "la serie. H2 se cumple en una serie y horizonte si la mediana del corto supera a la del +3x (p de "
          "Romano–Wolf < 0,05 en la familia de 36 de cada variante; bootstrap estacionario con bloque medio 63) y la "
          "media NO es significativamente mayor (p ≥ 0,05 sin corregir).")
    cols = {"ventanas": "ventanas", "mediana corto": "mediana corto", "mediana +3x": "mediana +3x", "Δ mediana": "Δ mediana (pp)",
            "p mediana RW": "p RW (mediana)", "media corto": "media corto", "media +3x": "media +3x",
            "Δ media": "Δ media (pp)", "p media": "p (media > 0)", "t media (HH)": "t HH (media)",
            "comisiones − préstamo en h": "comisiones − préstamo en h (pp)", "H2": "H2"}
    mv = main.set_index(["serie", "h"]).copy()
    mv["IC90 Δ mediana"] = [ci_text(a, b_, 2) for a, b_ in zip(mv["Δ mediana IC90_bajo"], mv["Δ mediana IC90_alto"])]
    mv["IC90 Δ media"] = [ci_text(a, b_, 2) for a, b_ in zip(mv["Δ media IC90_bajo"], mv["Δ media IC90_alto"])]
    mv = _pp_cols(mv, ["Δ mediana", "Δ media", "comisiones − préstamo en h"])
    order = ["ventanas", "mediana corto", "mediana +3x", "Δ mediana", "IC90 Δ mediana", "p mediana RW", "media corto",
             "media +3x", "Δ media", "IC90 Δ media", "p media", "t media (HH)", "comisiones − préstamo en h", "H2"]
    mv = mv[order].rename(columns=cols)
    mv.index = mv.index.set_levels([[short_name(s) for s in mv.index.levels[0]], mv.index.levels[1]])
    tab("H2 con el corto de referencia (s = 1, mantenimiento FINRA, préstamo mediano de cada ticker, producto a la "
        "letra − 0,5%) frente al +3x comprado, por serie y horizonte: medianas y medias de la rentabilidad de la "
        "ventana, diferencias (corto − +3x) con IC90 y p del bootstrap estacionario de las ventanas (bloque medio 63, "
        "2.000 remuestreos; la de la mediana, de Romano–Wolf en la familia de 36), t de Hansen-Hodrick de la media y "
        "lo que el corto gana a primer orden en h sesiones por las comisiones de los dos fondos menos el préstamo y el "
        "haircut del producto", mv, pct={"mediana corto", "mediana +3x", "media corto", "media +3x"}, digits=3)
    TN["main"] = tab.n
    cv = cnt[["se cumple", "falla: media", "falla: mediana", "falla: las dos"]]
    cv = cv.reindex(pd.MultiIndex.from_product([list(VARIANTS), H_LIST], names=["variante", "h"]))
    tab("Recuento del veredicto de H2 en la familia (9 series) por variante y horizonte. Las seis primeras son la "
        "rejilla del pre-registro con s = 1 y el mantenimiento; después, sin mantenimiento y s = 1/3 (préstamo mediano, "
        "producto remunerado) y, fuera de la rejilla, sin préstamo. «Falla: media» es que la media del corto es "
        "significativamente mayor que la del +3x; «falla: mediana», que su mediana no lo es", cv, digits=0)
    TN["counts"] = tab.n
    gv = fam[fam.variante.isin(GRID_V)].pivot_table(index=["serie", "h"], columns="variante", values="H2", aggfunc="first")
    gv = gv[GRID_V].replace({"se cumple": "sí", "falla: media": "media", "falla: mediana": "mediana",
                             "falla: las dos": "las dos"})
    gv = gv.reindex(pd.MultiIndex.from_tuples([k for k in mainx.index], names=["serie", "h"]))
    gv.index = gv.index.set_levels([[short_name(s) for s in gv.index.levels[0]], gv.index.levels[1]])
    gv.columns = [c.replace("préstamo ", "").replace("producto ", "") for c in gv.columns]
    tab("H2 en cada punto de la rejilla del pre-registro (préstamo mínimo, mediano o p90; producto remunerado o sin "
        "remunerar): «sí» si se cumple; si no, qué falla", gv, digits=0)
    fig_h2_diffs(C, rep)
    rep.p(f"**Lectura.** Con el préstamo mediano, a 5 sesiones H2 se cumple en {ok_h[5]} de {ns} series y a 21 en "
          f"{ok_h[21]}: la mediana favorece al corto y la media no se distingue de 0. A 63 se cumple en {ok_h[63]} y a 252 "
          f"en {ok_h[252]}: la mediana deja de favorecer al corto. Con s = 1 el mantenimiento FINRA le obliga a recomprar "
          "en cuanto el −3x sube un 5,3%, lo que recorta sus ganancias cuando el índice rebota y lo encoge cuando el "
          "índice sube después, y en 2009–2026 la tendencia alcista de los pares reales favoreció al +3x (su convexidad: "
          "e^(3X) crece sin techo y 1 − e^(−3X) no pasa del 100%). El préstamo decide los horizontes cortos: con el "
          f"mínimo de cada ticker y el producto remunerado H2 se cumple en {grid_tot[grid_name('mín', True)]} de 36 y con "
          f"el p90, en {grid_tot[grid_name('p90', True)]}. "
          + ("Con el bloque max(63, 2h) en lugar de 63, o con Holm en lugar de Romano–Wolf, ningún veredicto de la "
             "variante de referencia cambia." if same_long and same_holm else
             f"Con el bloque max(63, 2h) cambian {len(diff_long)} veredictos de la variante de referencia"
             + ("" if same_holm else "; con Holm en lugar de Romano–Wolf también cambia alguno") + "."))

    rep.h("3.1 Las variantes", 3)
    keyv = [MAIN, PURE, THIRD, NOBORROW]
    rows = []
    for (sname, h), r in mainx.iterrows():
        if h not in (21, 252):
            continue
        row = {"serie": short_name(sname), "h": h}
        for v in keyv:
            rr = fam[(fam.serie == sname) & (fam.h == h) & (fam.variante == v)].iloc[0]
            lab = "referencia" if v == MAIN else v
            row[f"Δ mediana · {lab} (pp)"] = rr["Δ mediana"] * 100
            row[f"Δ media · {lab} (pp)"] = rr["Δ media"] * 100
        rows.append(row)
    vv = pd.DataFrame(rows).set_index(["serie", "h"])
    tab("Diferencias corto − +3x (pp) a 21 y 252 sesiones: la referencia (s = 1, mantenimiento, préstamo mediano, "
        "producto remunerado); sin mantenimiento (el pago puro, solo con la ruina); s = 1/3 frente a 1/3 del +3x y 2/3 "
        "en letras; y, fuera de la rejilla, la referencia sin préstamo", vv, digits=2)
    rep.p(f"**Lectura.** Sin mantenimiento H2 se cumple en {ok_pure[5]}, {ok_pure[21]}, {ok_pure[63]} y {ok_pure[252]} "
          f"de {ns} series a 5, 21, 63 y 252 sesiones; con s = 1/3, en {ok_third[5]}, {ok_third[21]}, {ok_third[63]} y "
          f"{ok_third[252]}. Sin el mantenimiento, o con un tercio del capital (que lo incumple mucho menos), la mediana "
          f"del corto supera a la del +3x (p de Romano–Wolf < 0,05) hasta 63 sesiones en {med63_pure} de las "
          f"{3 * ns} combinaciones de 5, 21 y 63 sesiones sin mantenimiento y en {med63_third} con s = 1/3. Cuando H2 "
          "falla a horizontes cortos es por la media: el corto que no recompra cobra además la reversión del índice desde "
          "2000 (el cociente de varianzas menor que 1 de R29), que pesa en los pares reales y en el Nasdaq-100 "
          "sintético. En la historia larga del S&P 500, donde esa reversión no existe "
          f"(VR(21) = 0,99 desde 1928, R22), el pago puro cumple H2 a "
          f"{y_join(str(h) for h in H_LIST if pure.loc[(spn, h), 'H2'] == 'se cumple') or 'ningún horizonte'} sesiones "
          f"(a 21: mediana {pp(p_sp21['Δ mediana'])} pp, media {pp(p_sp21['Δ media'])}).")

    rep.h("3.2 La cola", 3)
    tl = []
    for v in (MAIN, PURE):
        for sname in ("SQQQ/TQQQ", "SPXU/UPRO", spn, ndn):
            for h in (21, 252):
                r = fam[(fam.serie == sname) & (fam.h == h) & (fam.variante == v)].iloc[0]
                tl.append({"variante": "referencia" if v == MAIN else v, "serie": short_name(sname), "h": h,
                           "p1 corto": r["p1 corto"], "p1 +3x": r["p1 +3x"],
                           "p5 corto": r["p5 corto"], "mediana corto": r["mediana corto"], "p95 corto": r["p95 corto"],
                           "p99 corto": r["p99 corto"], "p99 +3x": r["p99 +3x"], "asimetría corto": r["asimetría corto"],
                           "asimetría +3x": r["asimetría +3x"], "P(< −50%) corto": r["P(< −50%) corto"],
                           "P(< −50%) +3x": r["P(< −50%) +3x"], "ruinas": r["ruinas"],
                           "cola 5% del corto (pp)": r["cola 5% del corto"] * 100, "media corto (pp)": r["media corto"] * 100})
    tv = pd.DataFrame(tl).set_index(["variante", "serie", "h"])
    tab("Distribución de la rentabilidad de las ventanas (préstamo mediano, producto remunerado): cuantiles, "
        "asimetría, probabilidad de perder más del 50%, ventanas en ruina, y lo que aporta a la media del corto su 5% "
        "de peores ventanas (suma de esas ventanas / número de ventanas)",
        tv, pct={"p1 corto", "p1 +3x", "p5 corto", "mediana corto", "p95 corto", "p99 corto",
                 "p99 +3x", "P(< −50%) corto", "P(< −50%) +3x"}, digits=2)
    pu252 = fam[(fam.variante == PURE) & (fam.h == 252)]
    pr252 = fam[(fam.variante == MAIN) & (fam.h == 252)]
    pure_all = fam[fam.variante == PURE]
    ruin_rate = pu252.ruinas / pu252.ventanas
    rep.p(f"**Lectura.** Sin mantenimiento, el corto del −3x gana poco muchas veces y pierde mucho pocas: su asimetría "
          f"es negativa en todas las series y horizontes de la familia (de {num_es(pure_all['asimetría corto'].min(), 1)} "
          f"a {num_es(pure_all['asimetría corto'].max(), 1)}), mientras la del +3x a un año es positiva (desde "
          f"{num_es(pr252['asimetría +3x'].min(), 1)}); su mejor ventana está acotada en +100% y la peor, solo por la "
          f"ruina. A un año, entre el {pct_es(ruin_rate.min(), 0)} y el {pct_es(ruin_rate.max(), 0)} de las ventanas acaban "
          f"en ruina según la serie (el máximo es {short_name(pu252.loc[ruin_rate.idxmax(), 'serie'])}). El mantenimiento "
          "FINRA corta esa cola: con él, la probabilidad de perder más del 50% en un año queda entre el "
          f"{pct_es(pr252['P(< −50%) corto'].min(), 1)} y el {pct_es(pr252['P(< −50%) corto'].max(), 1)}, por debajo de la "
          f"del +3x en {int((pr252['P(< −50%) corto'] < pr252['P(< −50%) +3x']).sum())} de {len(pr252)} series; el precio "
          "es recomprar tras las subidas del −3x, que es lo que se come su mediana a horizontes largos.")

    rep.h("3.3 El pago de opción", 3)
    rep.p("Sea X el log-rendimiento del índice en la ventana y Q = Σu² la suma de los cuadrados de sus rendimientos "
          "diarios. Con R = L·u cada día (sin carry ni comisiones), el múltiplo del −3x es F y el del +3x, G:")
    rep.p("log F ≈ −3X − 6Q,   log G ≈ 3X − 3Q;   corto = 1 − F,   +3x = G − 1.")
    rep.p("*Pie de fórmula: el −3x pierde 6 veces la suma de cuadrados frente a −3 veces el índice y el +3x 3 veces; el "
          "corto cobra lo que el −3x cae, que como mucho es el 100%, y el +3x cobra lo que sube, sin techo.* Notación: X, "
          "el log-rendimiento del índice en las h sesiones; Q, la suma de sus rendimientos diarios al cuadrado (≈ σ²·h); "
          "F y G, lo que vale al final 1 $ en el −3x y en el +3x.")
    rep.p("Con rendimientos diarios independientes de media α y varianza σ², la media de un fondo diario no tiene "
          "arrastre y su mediana sí, el mismo para los dos:")
    rep.p("E[F] ≈ e^(−3αh),  mediana(F) ≈ e^(−3αh − 4,5σ²h);   E[G] ≈ e^(3αh),  mediana(G) ≈ e^(3αh − 4,5σ²h).")
    rep.p("*Pie de fórmula: en media cada fondo da −3 o +3 veces lo que da el índice, sin pérdida por el reajuste; en la "
          "mediana, los dos pierden el mismo ½L²σ² = 4,5σ² por sesión.* Notación: α, la rentabilidad media diaria del "
          "índice; σ², su varianza diaria; h, las sesiones de la ventana; E[·], la media; e^(·), la exponencial.")
    rep.p("De ahí H2: el corto se queda con la mediana baja del −3x (1 − mediana(F) es grande) mientras el +3x la sufre, "
          "así que en la mediana gana el corto; en la media, E[corto] − E[+3x] = 2 − 2·cosh(3αh) ≤ 0 antes de "
          "comisiones: la media del −3x, que sí sube cuando el índice se hunde, es lo que el corto vende. El «decaimiento» "
          "del inverso es la distancia entre su media y su mediana, y cobrarlo es vender la cola derecha del −3x, que es la "
          "cola izquierda del índice.")
    mv2 = mom.copy()
    mv2.index = mv2.index.set_levels([[short_name(s) for s in mv2.index.levels[0]], mv2.index.levels[1]])
    fam_names = [short_name(s) for s in dict.fromkeys(main.serie)]
    mv2 = mv2[mv2.index.get_level_values(0).isin(fam_names)]
    tab("Media y mediana de F y de G (múltiplos brutos del −3x y del +3x, con su carry y sus comisiones) en h sesiones, "
        "su cociente mediana/media y el de la aproximación normal, exp(−4,5·σ²·h) con la varianza diaria del índice de "
        "cada serie", mv2, pct={"media F − 1", "mediana F − 1", "media G − 1", "mediana G − 1", "media del índice (e^X − 1)"},
        digits=3)
    fig_payoff(C, rep)
    rep.p(f"**Lectura.** En el S&P 500 sintético a un año, la mediana de F es un {pct_es(1 - mo.loc[252, 'mediana/media F'], 0)} "
          f"menor que su media y la de G un {pct_es(1 - mo.loc[252, 'mediana/media G'], 0)} (la aproximación normal da un "
          f"{pct_es(1 - mo.loc[252, 'exp(−4,5·σ²·h)'], 0)} para los dos). La diferencia sale de que el log-rendimiento "
          "del índice no es normal: su cola izquierda, las caídas bruscas, es la cola derecha del −3x, que sube su media "
          "sin mover su mediana y baja la media del +3x; en los pares reales de 2009–2026, además, la varianza a un año "
          "fue menor que σ²·h (la reversión de R29) y los dos cocientes quedan más cerca de 1. En la aproximación normal "
          "el −3x no decae más que el +3x respecto de su propia media; decae más respecto de −3 veces el log-rendimiento "
          "del índice, que es lo que mide H1.")

    rep.h("3.4 Ampliación (fuera del pre-registro)", 3)
    ex = t2[~t2.familia & (t2.variante == MAIN)].set_index(["serie", "h"])[
        ["Δ mediana", "p mediana RW", "Δ media", "p media", "comisiones − préstamo en h", "H2"]]
    ex = _pp_cols(ex, ["Δ mediana", "Δ media", "comisiones − préstamo en h"]).rename(columns={
        "Δ mediana": "Δ mediana (pp)", "p mediana RW": "p RW (mediana, en la ampliación)", "Δ media": "Δ media (pp)",
        "p media": "p (media > 0)", "comisiones − préstamo en h": "comisiones − préstamo en h (pp)"})
    tab("Ampliación: H2 con el corto de referencia (préstamo mediano, producto remunerado) en los pares 3x fuera del "
        "pre-registro (Romano–Wolf en su propia familia)", ex, digits=3)

    rep.h("Desviaciones y lecturas del pre-registro")
    rep.p("\n".join([
        "El pre-registro no se puede editar en este trabajo: estas notas completan su sección «Desviaciones», que dice "
        "«(ninguna por ahora)», y están también en `letf/preregistro/articulo6_inversos_desviaciones.md`.",
        "- **Corregido tras la revisión del 30-09-2026 (después de ver los resultados).** La primera versión de este "
        "informe daba H2 sin préstamo (fuera de la rejilla), juzgaba H1 con D en lugar del arrastre literal, cambiaba "
        "«el doble» por el cociente de la fórmula, usaba un bloque max(63, 2h) y corregía la mediana con Holm. Ahora H1 "
        "se juzga como está escrita, H2 recorre la rejilla con el préstamo mediano como referencia, el bloque es 63 y la "
        "corrección es Romano–Wolf; lo anterior queda como sensibilidad o lectura, y la referencia sin préstamo, fuera de "
        "la rejilla y marcada.",
        "- **Lecturas que el pre-registro no fija.** H1: «ventanas anuales» son años naturales con al menos 63 sesiones "
        "del par; el arrastre neto D y D + φ se dan como lecturas, no como veredicto. H2: «mantener en corto el −3x» es "
        "el corto sin reajuste del protocolo con s = 1; la variante de referencia es el préstamo mediano con el producto "
        "remunerado (el pre-registro no dice qué punto de la rejilla); la mediana se juzga con α = 0,05 y Romano–Wolf en "
        "la familia de 36 de cada variante, y la media, con p unilateral sin corregir ≥ 0,05 (sin corregir es lo que "
        "más fácil pone H2 en aprietos).",
        "- **Desviación de aplicación.** Romano–Wolf necesita los remuestreos de toda la familia a la vez; las series "
        "tienen calendarios distintos, así que cada (serie, h) se remuestrea por separado y el máximo de la nula es el "
        "de columnas independientes. Eso hace la corrección algo conservadora, no más permisiva.",
    ]))

    rep.h("Salvedades y datos sin verificar")
    rep.p("\n".join([
        "- El índice de cada par es su ETF 1x de Yahoo (adj_close), no el índice que sigue el fondo; donde no coinciden "
        "(probablemente FAS y FAZ frente a XLF durante parte de su vida; SOXL y SOXS antes de 2021), la diferencia entra "
        "en D y en φ. Qué índice siguieron FAS y FAZ y cuándo cambió no está verificado aquí.",
        "- Precios de cierre de Yahoo, no valores liquidativos: las primas y descuentos diarios suman ruido a D (½Σe²).",
        "- Préstamo: 12 meses de IBKR (iBorrowDesk), aplicado constante a toda la historia, también a 1928–2008. "
        "Recompras forzosas (recall), falta de títulos y cambios de margen del bróker no se simulan; el mantenimiento "
        "FINRA es el mínimo (el bróker puede exigir más).",
        "- Las ventanas de H2 se solapan: el bootstrap por bloques y los errores de Hansen-Hodrick lo recogen solo en "
        "parte (a 252 sesiones las ventanas comparten 251 sesiones y el bloque del protocolo es 63; con max(63, 2h) se da "
        "la comprobación). La muestra de 2009–2026 tiene unos 17 años independientes y los intervalos son anchos.",
        "- Los fondos sintéticos siguen el modelo del laboratorio (R22) con un s constante; antes de 1954 se financian al "
        "call money. Antes del 1-10-1985 no hay Nasdaq-100 real.",
        "- Acceso minorista en la UE (PRIIPs) y marco regulatorio del corto de ETFs de EE. UU.: sin verificar.",
    ]))

    rep.h("Anexo")
    rep.p("Tablas completas en CSV: `anual_completo.csv` (arrastre literal, D, fórmula, carry y fricción de cada fondo "
          "real y año), `anual_sintetico_completo.csv` (lo mismo en la historia sintética) y `h2_completo.csv` (todas las "
          "series, horizontes y variantes de H2, con cuantiles, colas y las p de Romano–Wolf, Holm y del bloque "
          "max(63, 2h)).")
    yrs.to_csv(rep.dir / "anual_completo.csv", index=False)
    H1s["years"].to_csv(rep.dir / "anual_sintetico_completo.csv", index=False)
    t2.to_csv(rep.dir / "h2_completo.csv", index=False)
    for k, n in TN.items():
        rep.lines = [ln.replace(f"«T:{k}»", str(n)) for ln in rep.lines]
    rep.save()
    return C


if __name__ == "__main__":
    run()
