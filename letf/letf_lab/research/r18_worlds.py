"""R18 — Mundos sintéticos: ablación y potencia estadística (I5).

Pregunta. La LRS 3x original (Gayed y Bilello 2016: media de 200 sesiones sobre
el índice de rentabilidad total) y la LRS-Kelly de R13 ganan al 1x en la historia
del S&P 500, pero con un solo pasado no se distingue «no hay
efecto» de «no hay potencia». Aquí se generan mercados sintéticos en los que se
sabe qué rasgo existe y cuál se ha quitado, y en cada uno se mide (a) la
ventaja verdadera de la estrategia y (b) la probabilidad de que un contraste la
detecte con 10, 30 o 98 años de datos.

Mundos (1.000 historias de 108 años cada uno: 10 de calentamiento + 98):
- iid: bootstrap de días sueltos de la historia 1928–2026 (bloque medio 1 día):
  mismas colas diarias, sin agrupamiento de volatilidad ni tendencias.
- bootstrap estacionario con bloques medios de 21, 252 y 756 días: conserva la
  dependencia de la historia hasta la escala del bloque.
- GJR-GARCH «corregido» (r13.gen_paths): log-rentabilidades con media
  constante, residuos estandarizados reales remuestreados (con sus partes
  negativa y positiva reescaladas para no inflar la persistencia) y varianza
  de largo plazo igual a la histórica. Agrupamiento y efecto apalancamiento,
  sin tendencias en la media.
- GARCH simétrico: el mismo generador reestimado sin el término asimétrico:
  agrupamiento sin efecto apalancamiento.
- HMM completo (r13.gen_paths, «regímenes»): dos regímenes de Markov
  (calma/crisis) estimados con rentabilidades semanales, cada uno con su media
  y su volatilidad; innovaciones normales.
- HMM con medias iguales: los mismos regímenes y las mismas volatilidades,
  pero con la misma rentabilidad esperada ARITMÉTICA en los dos (se conserva
  la media incondicional). Sólo la varianza depende del régimen.
- Sensibilidad añadida (no estaba en el plan): GJR-GARCH con media ARITMÉTICA
  constante. En el GJR de r13 la media constante es la logarítmica, así que la
  rentabilidad esperada simple sube con la varianza (≈ m + h/2) y salir en
  las rachas turbulentas cuesta rentabilidad; esta variante (log-media
  m + (σ² − h_t)/2, mismas innovaciones) separa ese supuesto del canal de
  volatilidad. Va en las tablas, no en las figuras.
En GJR, GARCH y HMM los parámetros no son fijos: cada bloque de 5 historias usa
parámetros reestimados en un remuestreo por bloques (bootstrap estacionario de
bloque medio 1.260 días) de la historia, de modo que la dispersión de los
resultados incluye la incertidumbre de estimación. Los tres mundos GARCH y los
dos HMM comparten remuestreo y números aleatorios (comparaciones emparejadas).

Estadístico decisivo, fijado antes de simular: Δg = 252 × media diaria de
log(1 + r_estrategia) − log(1 + r_referencia), en % anual, para la LRS 3x y la
LRS-Kelly frente a (i) el 1x y (ii) una cartera constante con la MISMA
exposición media al 3x que la estrategia (fracción w̄ del 3x y 1 − w̄ en
T-bills, reequilibrada a diario). (ii) aísla el valor de la sincronización; la
diferencia entre (i) y (ii) es el efecto «nivel» de apalancamiento. Contraste:
t con errores Newey-West (Bartlett, 63 retardos), unilateral (H1: Δg > 0) al
5%. Potencia = frecuencia de rechazo entre historias. Las estrategias y costes
son los de r13.mc_strategies (señal: media de 200 sesiones del índice simulado,
que es de rentabilidad total, la misma señal que la LRS original en la historia;
al cierre de t, orden al cierre de t+1, 5 pb por rotación; Kelly con EWMA
λ = 0,94 y la prima aprendida en cada historia desde su año 10). Los p de titular
de la historia real se exportan, con su procedencia, a p_titular.csv (los lee R19).

Cambio tras la revisión (segunda versión): en la primera, w̄ era la exposición
media de la estrategia en la PROPIA ventana evaluada. Se conoce sólo al final
y está muy correlacionada con la rentabilidad del mercado en esa ventana (la
LRS está más tiempo dentro en las ventanas buenas), así que la referencia
ponía más apalancamiento justo donde convenía y sesgaba Δg a la baja, tanto más
cuanto más corta la ventana. Ahora w̄ es la exposición media a largo plazo de
la estrategia en ese mundo, fijada ex ante: la media en los 98 años evaluados
de las DEMÁS historias con los mismos parámetros (las otras 249 del lote en los
bootstrap; las otras 4 del mismo sorteo en GJR, GARCH y HMM). No depende de la
historia a la que se aplica. La versión anterior («ex post») se conserva para
medir ese sesgo por horizonte y para comparar con la historia real a 88 años
con el mismo estimador.
"""

from __future__ import annotations

import pickle
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fft as sfft
from scipy.signal import lfilter
from scipy.stats import norm

from letf_lab import provenance, stats
from letf_lab.report import FIG_W, Report, comma_fig, legend_band
from letf_lab.research import r13_lrs as r13

SEED = 18
N_PATHS = 1000
CHUNK = 250                 # historias por tarea en paralelo
PATHS_PER_DRAW = 5          # historias por cada juego de parámetros (GJR/GARCH/HMM)
PAR_BLOCK = 1260            # bloque medio del remuestreo para reestimar parámetros
BURN = r13.BURN             # 10 años de calentamiento (media de Kelly y medias móviles)
YEARS_MAX = 98
T = (YEARS_MAX + BURN) * 252
E0 = BURN * 252
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 75, 88, 98)
KEY_H = (10, 30, 98)
NW_LAGS = 63
ALPHA = 0.05
Z_A, Z_POW = norm.ppf(1 - ALPHA), norm.ppf(0.80)
LAG = r13.EXEC_LAG
N_JOBS = 4
CLIP = -0.999               # como r13.spa_tests: un −100% cuenta como −99,9% en log
P_EFF_MAX = 0.9999          # persistencia efectiva máxima admitida (la de stats.garch_fit)

# nombre: (generador, parámetro, familia)
WORLDS = {
    "iid (bloque 1 d)": ("bootstrap", 1, "bootstrap"),
    "bootstrap 21 d": ("bootstrap", 21, "bootstrap"),
    "bootstrap 252 d": ("bootstrap", 252, "bootstrap"),
    "bootstrap 756 d": ("bootstrap", 756, "bootstrap"),
    "GJR-GARCH": ("gjr", None, "garch"),
    "GARCH simétrico": ("garch", None, "garch"),
    "HMM medias iguales": ("hmm_eq", None, "hmm"),
    "HMM completo": ("hmm", None, "hmm"),
    "GJR, media aritmética constante": ("gjr_arith", None, "garch"),   # sensibilidad
}
FIG_WORLDS = list(WORLDS)[:8]   # las figuras muestran los ocho mundos del plan (8 colores)
STRATS = ("LRS 3x (original)", "LRS-Kelly")
BENCH = ("1x", "constante")          # «constante» = exposición de referencia ex ante (a largo plazo)
POST_B = "constante ex post"         # exposición media de la propia ventana (primera versión)
REF_DAYS = YEARS_MAX * 252           # la exposición a largo plazo se mide en los 98 años evaluados
COMP = [(s, b) for s in STRATS for b in BENCH]
COMP_LABEL = {("LRS 3x (original)", "1x"): "LRS 3x − 1x",
              ("LRS 3x (original)", "constante"): "LRS 3x − constante",
              ("LRS-Kelly", "1x"): "LRS-Kelly − 1x",
              ("LRS-Kelly", "constante"): "LRS-Kelly − constante"}
# paleta categórica de referencia, en orden fijo; marcadores y trazo como segunda codificación
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COLOR = dict(zip(FIG_WORLDS, PALETTE))
MARKER = {"bootstrap": "o", "garch": "s", "hmm": "^"}
DASHED = {"iid (bloque 1 d)", "GARCH simétrico", "HMM medias iguales"}  # mundos sin el rasgo
INK = "#0b0b0b"


# ───────────────────────────── generadores ─────────────────────────────

def boot_paths(u: np.ndarray, n: int, T: int, block: float, rng) -> np.ndarray:
    """Bootstrap estacionario con bloque medio `block` (1 = días independientes).

    Con block = 252 es idéntico a r13.gen_paths('bootstrap')."""
    u = u[~np.isnan(u)]
    idx = np.stack([stats.stationary_indices(len(u), block, rng, T) for _ in range(n)])
    return u[idx]


def garch_setup(u: np.ndarray, kind: str = "gjr") -> dict:
    """Estimación y residuos del generador GJR de r13.gen_paths, sin simular.

    p_eff es la persistencia efectiva con los residuos que se remuestrean; si
    llega a 1, ω = var·(1 − p_eff) ≤ 0 y la varianza simulada puede hacerse
    negativa (r13.gen_paths devolvería NaN)."""
    u = u[~np.isnan(u)]
    ell = np.log1p(u)
    p = stats.garch_fit(ell, kind)
    m = ell.mean()
    h = np.r_[np.var(ell), stats.garch_filter(ell - m, p)[:-1]]
    z = (ell - m) / np.sqrt(h)
    z = z / z.std()
    # corrección de r13: partes negativa y positiva reescaladas a la mitad de la varianza cada una
    z[z < 0] *= np.sqrt(0.5 / np.mean(z ** 2 * (z < 0)))
    z[z >= 0] *= np.sqrt(0.5 / np.mean(z ** 2 * (z >= 0)))
    z = (z - z.mean()) / z.std()
    p_eff = p["alpha"] * np.mean(z ** 2) + p["gamma"] * np.mean(z ** 2 * (z < 0)) + p["beta"]
    return {"p": p, "m": m, "z": z, "p_eff": p_eff, "v0": np.var(ell), "omega": np.var(ell) * (1 - p_eff)}


def garch_paths(u: np.ndarray, n: int, T: int, rng, kind: str = "gjr",
                setup: dict | None = None, arith: bool = False) -> tuple[np.ndarray, dict]:
    """El generador GJR de r13.gen_paths con el modelo como parámetro.

    kind = 'gjr' reproduce r13.gen_paths('gjr') (mismos números aleatorios);
    kind = 'garch' reestima imponiendo γ = 0: igual agrupamiento, sin asimetría.
    arith = True: log-media m + (v0 − h_t)/2, con lo que la media aritmética
    (≈ m + v0/2) no depende de la varianza; mismas innovaciones y varianzas."""
    s = setup or garch_setup(u, kind)
    p, m, z, omega, v0 = s["p"], s["m"], s["z"], s["omega"], s["v0"]
    hv = np.full(n, v0)
    out = np.empty((n, T))
    for t in range(T):
        e = rng.choice(z, n) * np.sqrt(hv)
        out[:, t] = m + e + ((v0 - hv) / 2 if arith else 0.0)
        hv = omega + (p["alpha"] + p["gamma"] * (e < 0)) * e ** 2 + p["beta"] * hv
    return np.expm1(out), {"alpha": p["alpha"], "beta": p["beta"], "gamma": p["gamma"],
                           "p_eff": s["p_eff"], "m_anual": m * 252, "vol_anual": np.sqrt(s["v0"] * 252)}


def hmm_equal_means(mu_d: np.ndarray, sd_d: np.ndarray, stay: np.ndarray) -> np.ndarray:
    """Medias logarítmicas diarias que igualan la rentabilidad esperada aritmética
    de los dos regímenes y conservan la incondicional: E[e^ℓ | k] = constante."""
    q = 1 - stay
    pi = np.array([q[1], q[0]]) / q.sum()          # distribución estacionaria de la cadena diaria
    a = np.log(pi @ np.exp(mu_d + sd_d ** 2 / 2))
    return a - sd_d ** 2 / 2


def hmm_paths(u: np.ndarray, n: int, T: int, rng, equal_means: bool = False) -> tuple[np.ndarray, dict]:
    """El generador de regímenes de r13.gen_paths; equal_means=False lo reproduce
    exactamente (mismos números aleatorios)."""
    u = u[~np.isnan(u)]
    ell = np.log1p(u)
    wk = pd.Series(ell).groupby(np.arange(len(ell)) // 5).sum().values
    f = stats.hmm_fit(wk)
    mu_d, sd_d = f["mu"] / 5, f["sd"] / np.sqrt(5)
    stay = np.diag(f["P"]) ** (1 / 5)
    par = {"mu_calma": mu_d[0] * 252, "mu_crisis": mu_d[1] * 252, "vol_calma": sd_d[0] * np.sqrt(252),
           "vol_crisis": sd_d[1] * np.sqrt(252), "dur_calma_años": 1 / (1 - stay[0]) / 252,
           "dur_crisis_años": 1 / (1 - stay[1]) / 252}
    if equal_means:
        mu_d = hmm_equal_means(mu_d, sd_d, stay)
    state = rng.choice(2, n, p=f["smoothed"].mean(axis=0))
    out = np.empty((n, T))
    for t in range(T):
        out[:, t] = mu_d[state] + sd_d[state] * rng.standard_normal(n)
        switch = rng.random(n) > stay[state]
        state = np.where(switch, 1 - state, state)
    return np.expm1(out), par


def gen_chunk(world: str, u: np.ndarray, chunk: int) -> tuple[np.ndarray, list, np.ndarray]:
    """CHUNK historias del mundo; devuelve (U, parámetros por sorteo, sorteo de cada historia)."""
    gen, par, fam = WORLDS[world]
    if gen == "bootstrap":
        rng = np.random.default_rng([SEED, int(par), chunk])
        U = r13.gen_paths("bootstrap", u, CHUNK, T, rng) if par == 252 else boot_paths(u, CHUNK, T, par, rng)
        return U, [], np.full(CHUNK, -1)
    group = {"garch": 1, "hmm": 2}[fam]
    per = CHUNK // PATHS_PER_DRAW
    Us, pars, draw = [], [], []
    for k in range(chunk * per, (chunk + 1) * per):
        rng_res = np.random.default_rng([SEED, 100 + group, k])
        ur = u[stats.stationary_indices(len(u), PAR_BLOCK, rng_res)]
        rechazos = 0
        if fam == "garch":
            # sólo parámetros con varianza estacionaria en los DOS modelos del par
            while True:
                su = {kd: garch_setup(ur, kd) for kd in ("gjr", "garch")}
                if max(v["p_eff"] for v in su.values()) < P_EFF_MAX:
                    break
                rechazos += 1
                ur = u[stats.stationary_indices(len(u), PAR_BLOCK, rng_res)]
        rng = np.random.default_rng([SEED, 200 + group, k])   # mismos números en los dos mundos del par
        if fam == "garch":
            if gen == "gjr":
                Uk = r13.gen_paths("gjr", ur, PATHS_PER_DRAW, T, rng)
            elif gen == "gjr_arith":
                Uk, _ = garch_paths(ur, PATHS_PER_DRAW, T, rng, "gjr", setup=su["gjr"], arith=True)
            else:
                Uk, _ = garch_paths(ur, PATHS_PER_DRAW, T, rng, "garch", setup=su["garch"])
            sv = su["garch" if gen == "garch" else "gjr"]
            p = sv["p"]
            pk = {"alpha": p["alpha"], "beta": p["beta"], "gamma": p["gamma"],
                  "α+γ/2+β": p["alpha"] + p["gamma"] / 2 + p["beta"], "p_eff": sv["p_eff"],
                  "m_anual": sv["m"] * 252, "rechazos": rechazos}
        elif gen == "hmm":
            Uk, pk = r13.gen_paths("regímenes", ur, PATHS_PER_DRAW, T, rng), {}
        else:
            Uk, pk = hmm_paths(ur, PATHS_PER_DRAW, T, rng, equal_means=True)
        Us.append(Uk)
        pars.append(pk)
        draw += [k] * PATHS_PER_DRAW
    return np.vstack(Us), pars, np.array(draw)


# ───────────────────────────── estrategias ─────────────────────────────

def sticky_mat(target: np.ndarray) -> np.ndarray:
    """Copia de la regla de r13.mc_strategies: reajusta si el objetivo se mueve > MIN_STEP."""
    out = np.full_like(target, np.nan)
    cur = np.full(len(target), np.nan)
    for t in range(target.shape[1]):
        v = target[:, t]
        move = np.isnan(cur) | (np.abs(v - cur) > r13.MIN_STEP) | (v == 0) | (v == 1)
        cur = np.where(np.isnan(v), np.nan, np.where(move, v, cur))
        out[:, t] = cur
    return out


def mc_core(U: np.ndarray, rf_d: float, f_d: float, spread: float, er: float,
            lag: int = LAG) -> tuple[dict, dict, np.ndarray]:
    """Las estrategias de r13.mc_strategies que aquí se usan (mismo código), más las
    fracciones EJECUTADAS en el 3x que hacen falta para la cartera de exposición
    constante. Devuelve (rentabilidades, pesos ejecutados, señal sobre la media)."""
    x3 = np.maximum(rf_d + 3 * (U - f_d) - (3 * spread + er) / 252, -1.0)
    px = np.cumprod(1 + U, axis=1)
    sig = r13._sma_signal(px, 200)
    lam = 0.94
    ewma = lfilter([1 - lam], [1.0, -lam], U ** 2, axis=1,
                   zi=np.full((len(U), 1), lam * np.var(U[:, :252])))[0]
    var_fc = ewma * 252
    t_idx = np.arange(1, U.shape[1] + 1)
    mu = np.c_[np.full(len(U), np.nan), (np.cumsum(U, axis=1) / t_idx)[:, :-1]] * 252
    mu[:, : BURN * 252] = np.nan
    kel = np.clip(r13.KELLY * (mu - f_d * 252 - spread) / var_fc / 3, 0, 1)
    w_kel = sticky_mat(kel * np.nan_to_num(sig))

    def shift(w):
        return np.c_[np.zeros((len(w), lag)), np.nan_to_num(w)[:, :-lag]]

    def run(w3):
        w3 = shift(w3)
        turn = np.abs(np.diff(np.c_[np.zeros(len(w3)), w3], axis=1))
        return w3 * x3 + (1 - w3) * rf_d - turn * r13.COST_BPS / 1e4, w3

    R = {"1x comprar y mantener": U, "3x comprar y mantener": x3}
    W = {}
    R["LRS 3x (original)"], W["LRS 3x (original)"] = run(sig)
    R["LRS-Kelly"], W["LRS-Kelly"] = run(w_kel)
    return R, W, sig


# ───────────────────────────── contrastes ─────────────────────────────

def nw_mean_se(D: np.ndarray, lags: int = NW_LAGS) -> tuple[np.ndarray, np.ndarray]:
    """Media por fila y su error estándar Newey-West (Bartlett, `lags` retardos).

    Idéntico a stats.ols_hac(y, sin regresores, const=True) fila a fila, pero
    vectorizado con la FFT: Σ_t x_t x_{t−l} para todas las filas a la vez."""
    D = np.atleast_2d(np.asarray(D, float))
    n = D.shape[1]
    m = D.mean(axis=1)
    X = D - m[:, None]
    nfft = sfft.next_fast_len(n + lags + 1)
    F = sfft.rfft(X, nfft, axis=1)
    ac = sfft.irfft(F.real ** 2 + F.imag ** 2, nfft, axis=1)[:, : lags + 1]
    w = 1 - np.arange(1, lags + 1) / (lags + 1)
    S = ac[:, 0] + 2 * (ac[:, 1:] * w).sum(axis=1)
    var = S / n ** 2 * n / (n - 1)
    return m, np.sqrt(np.maximum(var, 0))


def const_mix(wbar: np.ndarray, x3: np.ndarray, rf) -> np.ndarray:
    """Fracción constante w̄ del 3x y 1 − w̄ en T-bills, reequilibrada a diario (sin costes)."""
    return wbar[:, None] * x3 + (1 - wbar[:, None]) * rf


def loo_mean(x: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Media de x en las DEMÁS filas de su grupo (deja fuera la propia).

    Da a cada historia una referencia ex ante: no depende de sus rentabilidades.
    groups = None: todas las filas forman un grupo."""
    x = np.asarray(x, float)
    g = np.zeros(len(x), int) if groups is None else np.asarray(groups)
    s = pd.Series(x).groupby(g)
    tot, cnt = s.transform("sum").values, s.transform("count").values
    if (cnt < 2).any():
        raise ValueError("cada grupo necesita al menos dos historias")
    return (tot - x) / (cnt - 1)


def path_tests(R: dict, W: dict, rf_d: float, e0: int = E0, horizons=HORIZONS,
               groups: np.ndarray | None = None, ref_days: int = REF_DAYS) -> dict:
    """Δg estimado, error NW y exposición media en las ventanas [e0, e0 + Y años).

    Dos carteras constantes de referencia: «constante» usa la exposición media a
    largo plazo (ref_days) de las demás historias del mismo grupo (ex ante);
    «constante ex post», la media de la estrategia en la propia ventana."""
    out = {}
    l1 = np.log1p(R["1x comprar y mantener"][:, e0:])
    x3 = R["3x comprar y mantener"][:, e0:]
    for Y in horizons:
        out[("mercado", Y)] = l1[:, : Y * 252].mean(axis=1) * 252
    for s in STRATS:
        ls = np.log1p(np.maximum(R[s][:, e0:], CLIP))
        w = W[s][:, e0:]
        wref = loo_mean(w[:, :ref_days].mean(axis=1), groups)
        out[(s, "exp_ref")] = 3 * wref
        for Y in horizons:
            n = Y * 252
            wbar = w[:, :n].mean(axis=1)
            for b, lb in (("1x", l1[:, :n]),
                          ("constante", np.log1p(np.maximum(const_mix(wref, x3[:, :n], rf_d), CLIP))),
                          (POST_B, np.log1p(np.maximum(const_mix(wbar, x3[:, :n], rf_d), CLIP)))):
                g, se = nw_mean_se(ls[:, :n] - lb)
                out[(s, b, Y, "g")], out[(s, b, Y, "se")] = g * 252, se * 252
            out[(s, "exp", Y)] = 3 * wbar
    return out


def path_metrics(S: dict, eval_from: int, n_days: int) -> dict:
    """Por historia, lo que r13.mc_summary agrega: CAGR, caída máxima y log-riqueza final."""
    out = {}
    years = n_days / 252
    for k, r in S.items():
        r = r[:, eval_from:eval_from + n_days].copy()
        if k == "3x comprar y mantener":  # tras liquidarse, a cero para siempre
            dead = np.cumsum(r <= -1, axis=1) > 0
            first = dead & ~np.c_[np.zeros((len(r), 1), bool), dead[:, :-1]]
            r = np.where(dead & ~first, 0.0, r)
        w = np.cumprod(1 + r, axis=1)
        out[k] = {"lw": np.log(np.maximum(w[:, -1], 1e-300)),
                  "cagr": np.maximum(w[:, -1], 0) ** (1 / years) - 1,
                  "dd": (w / np.maximum.accumulate(w, axis=1) - 1).min(axis=1)}
    return out


def summarize(M: dict) -> pd.DataFrame:
    """Tabla con las columnas de r13.mc_summary a partir de path_metrics."""
    rows = {}
    base = M["1x comprar y mantener"]["lw"]
    for k, m in M.items():
        rows[k] = {"CAGR_mediana": np.median(m["cagr"]), "CAGR_p5": np.percentile(m["cagr"], 5),
                   "CAGR_p95": np.percentile(m["cagr"], 95), "maxDD_mediana": np.median(m["dd"]),
                   "P(caída > 90%)": (m["dd"] < -0.9).mean(), "P(pierde dinero)": (m["lw"] < 0).mean(),
                   "P(gana al 1x)": float((m["lw"] > base).mean())}
    return pd.DataFrame(rows).T


# ───────────────────────────── diagnóstico ─────────────────────────────

def _rowcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))


def path_features(U: np.ndarray, sig: np.ndarray, e0: int, lag: int = LAG) -> dict:
    """Rasgos de cada historia que explican la LRS, en la ventana [e0, fin).

    La señal es la de la estrategia (índice de rentabilidad total sobre su media
    de 200 sesiones) tal como se ejecuta: la del cierre de t − lag."""
    u = U[:, e0:]
    s = np.c_[np.full((len(U), lag), np.nan), sig[:, :-lag]][:, e0:]
    up, dn = s == 1, s == 0
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_up = np.nanmean(np.where(up, u, np.nan), axis=1) * 252
        mu_dn = np.nanmean(np.where(dn, u, np.nan), axis=1) * 252
        sd_up = np.nanstd(np.where(up, u, np.nan), axis=1) * np.sqrt(252)
        sd_dn = np.nanstd(np.where(dn, u, np.nan), axis=1) * np.sqrt(252)
    n_m = u.shape[1] // 21
    M = u[:, : n_m * 21].reshape(len(u), n_m, 21)
    Rm = np.log1p(M).sum(axis=2)
    RV = (M ** 2).sum(axis=2)
    c12 = np.cumsum(np.c_[np.zeros(len(Rm)), Rm], axis=1)
    mom = c12[:, 12:-1] - c12[:, :-13]          # 12 meses que acaban en m
    return {"crec_log_anual": np.log1p(u).mean(axis=1) * 252,
            "vol_anual": u.std(axis=1) * np.sqrt(252),
            "fracción_sobre_media": up.sum(axis=1) / (up | dn).sum(axis=1),
            "μ_sobre − μ_bajo": mu_up - mu_dn,
            "σ_bajo / σ_sobre": sd_dn / sd_up,
            "agrupamiento ρ(RV_m, RV_m+1)": _rowcorr(RV[:, :-1], RV[:, 1:]),
            "apalancamiento ρ(R_m, RV_m+1)": _rowcorr(Rm[:, :-1], RV[:, 1:]),
            "tendencia ρ(R_12m, R_m+1)": _rowcorr(mom, Rm[:, 12:])}


# ───────────────────────────── una tarea ─────────────────────────────

def run_chunk(world: str, chunk: int, u: np.ndarray, rf_d: float, f_d: float,
              spread: float, er: float) -> dict:
    t0 = time.time()
    U, pars, draw = gen_chunk(world, u, chunk)
    R, W, sig = mc_core(U, rf_d, f_d, spread, er)
    x3 = R["3x comprar y mantener"]
    ruin = ((x3[:, E0:] <= -1) & (W["LRS 3x (original)"][:, E0:] > 0)).any(axis=1)
    n88 = E0 + 88 * 252
    out = {"world": world, "chunk": chunk, "draw": draw, "params": pars,
           "tests": path_tests(R, W, rf_d, groups=draw),
           "m30": path_metrics(R, E0, 30 * 252), "m98": path_metrics(R, E0, YEARS_MAX * 252),
           "features": path_features(U, sig, E0),
           "features88": path_features(U[:, :n88], sig[:, :n88], E0), "ruina_LRS": ruin,
           "U40": U[:, : (BURN + r13.YEARS) * 252].astype(np.float32)}
    out["segundos"] = time.time() - t0
    return out


def simulate(u: np.ndarray, rf_d: float, f_d: float, spread: float, er: float) -> dict:
    from joblib import Parallel, delayed

    tasks = [(w, c) for w in WORLDS for c in range(N_PATHS // CHUNK)]
    res = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(run_chunk)(w, c, u, rf_d, f_d, spread, er) for w, c in tasks)
    worlds = {}
    for w in WORLDS:
        parts = sorted([r for r in res if r["world"] == w], key=lambda r: r["chunk"])
        cat = lambda key: {k: np.concatenate([p[key][k] for p in parts]) for k in parts[0][key]}
        worlds[w] = {
            "tests": cat("tests"), "features": cat("features"), "features88": cat("features88"),
            "draw": np.concatenate([p["draw"] for p in parts]),
            "params": [q for p in parts for q in p["params"]],
            "m30": {k: {m: np.concatenate([p["m30"][k][m] for p in parts]) for m in ("lw", "cagr", "dd")}
                    for k in parts[0]["m30"]},
            "m98": {k: {m: np.concatenate([p["m98"][k][m] for p in parts]) for m in ("lw", "cagr", "dd")}
                    for k in parts[0]["m98"]},
            "ruina_LRS": np.concatenate([p["ruina_LRS"] for p in parts]),
            "diagnóstico": r13.world_diagnostics(np.vstack([p["U40"] for p in parts]).astype(float), u),
            "segundos": sum(p["segundos"] for p in parts),
        }
    return worlds


# ───────────────────────────── historia real ─────────────────────────────

# Variantes históricas. La LRS 3x (original) de R13 y la LRS-Kelly con varianza EWMA usan
# la misma señal que los mundos y que path_features (índice de rentabilidad total sobre su
# media de 200 sesiones, la de Gayed y Bilello) y son las cifras de comparación; la LRS-Kelly
# de R13 prevé la varianza con GJR. Las de «señal de precio» (r13.price_variants) son la
# sensibilidad del laboratorio, que los mundos, sin dividendos, no pueden reproducir.
REAL_KG = "LRS-Kelly (GJR, señal total, R13)"
REAL_STRATS = ("LRS 3x (original)", r13.PX_LRS, "LRS-Kelly (EWMA, señal total)", REAL_KG, r13.PX_KELLY)
REAL_LIKE = {"LRS 3x (original)": "LRS 3x (original)", "LRS-Kelly": "LRS-Kelly (EWMA, señal total)"}
REAL_PX = {"LRS 3x (original)": r13.PX_LRS, "LRS-Kelly": r13.PX_KELLY}
POST = "constante (media de la ventana)"     # mismo estimador que POST_B en los mundos
ANTE = "constante (media previa)"            # exposición media de la estrategia ANTES de la ventana
MIN_PRIOR = 10 * 252                         # años previos mínimos para fijar la referencia ex ante


def tr_signal(u: pd.Series, n: int = 200) -> pd.Series:
    """Señal de la LRS sobre el índice de rentabilidad total, con el mismo código que los mundos.
    u: la rentabilidad con dividendos de la señal (en el S&P 500, d['u_sig'], sin mirar el futuro)."""
    uu = u.dropna()
    s = r13._sma_signal(np.cumprod(1 + uu.values)[None], n)[0]
    return pd.Series(s, index=uu.index).reindex(u.index)


def real_history(D: dict) -> dict:
    """Las mismas cifras en el S&P 500 real: la LRS 3x original de R13 (señal de rentabilidad
    total, la de los mundos), la LRS-Kelly con varianza EWMA y la de R13 (GJR), y como
    sensibilidad las dos con la señal de precio (r13.price_variants)."""
    d = D["S&P 500"]
    var_fc = r13.var_forecast(d)
    R, E = r13.strategies(d, var_fc)
    Rp, Ep = r13.price_variants(d, var_fc)
    sig_tr = r13.above_sma(r13.signal_index(d))
    # la señal de la LRS original es la de los mundos: el código de los mundos sobre la misma
    # rentabilidad de la señal da los mismos estados salvo empates numéricos (índice ≈ media)
    sig_w = tr_signal(d.get("u_sig", d["u"]))
    ok = sig_tr.notna() & sig_w.notna()
    assert int((sig_tr[ok] != sig_w[ok]).sum()) <= 3, "la señal de R13 no es la de los mundos"
    ew = (d["u"] ** 2 * 252).ewm(alpha=0.06, adjust=False).mean()
    w_ew = r13.sticky(r13.kelly_exposure(d, ew) * sig_tr)
    RS = {"LRS 3x (original)": R["LRS 3x (original)"], r13.PX_LRS: Rp[r13.PX_LRS],
          "LRS-Kelly (EWMA, señal total)": r13.run_weights(d, w_ew), REAL_KG: R["LRS-Kelly"],
          r13.PX_KELLY: Rp[r13.PX_KELLY]}
    # fracción EJECUTADA en el 3x (la decisión de t − 2)
    WX = {"LRS 3x (original)": (E["LRS 3x (original)"] / 3).shift(LAG),
          r13.PX_LRS: (Ep[r13.PX_LRS] / 3).shift(LAG),
          "LRS-Kelly (EWMA, señal total)": w_ew.shift(LAG),
          REAL_KG: (E["LRS-Kelly"] / 3).shift(LAG),
          r13.PX_KELLY: (Ep[r13.PX_KELLY] / 3).shift(LAG)}
    uu = d["u"].dropna()
    last, idx = uu.index[-1], uu.index
    a28 = max(RS[k].first_valid_index() for k in REAL_STRATS[:2])
    a38 = max(RS[k].first_valid_index() for k in REAL_STRATS[2:])
    # orden fijo: [0] toda la historia, [1] la ventana común con Kelly, [2] 30 años, [3] 10 años
    windows = {f"{a28.year}–{last.year}": a28, f"{a38.year}–{last.year}": a38,
               "últimos 30 años": idx[-30 * 252], "últimos 10 años": idx[-10 * 252]}
    rows = []
    for wname, a in windows.items():
        u_, x3, rf_ = d["u"].loc[a:last], d["x3"].loc[a:last], d["rf"].loc[a:last]
        for s in REAL_STRATS:
            r = RS[s].loc[a:last]
            if r.isna().any() or r.empty:
                continue
            ls = np.log1p(np.maximum(r, CLIP))
            wbar = float(WX[s].loc[a:last].mean())
            prior = WX[s].loc[:a].iloc[:-1].dropna()
            refs = {"1x": np.nan, POST: wbar}
            if len(prior) >= MIN_PRIOR:
                refs[ANTE] = float(prior.mean())
            for b, wref in refs.items():
                lb = np.log1p(u_) if b == "1x" else np.log1p(np.maximum(wref * x3 + (1 - wref) * rf_, CLIP))
                g, se = nw_mean_se((ls - lb).values[None])
                g, se = g[0] * 252, se[0] * 252
                rows.append({"ventana": wname, "desde": r.index[0].date(), "años": len(r) / 252,
                             "estrategia": s, "frente a": b, "exposición_estrategia": 3 * wbar,
                             "exposición_referencia": 3 * wref, "Δg": g, "error_NW": se, "t": g / se,
                             "p_unilateral": norm.sf(g / se)})
    check = r13.stats_table({k: R[k] for k in ("1x comprar y mantener", "3x comprar y mantener")} | RS,
                            d["rf"], var_fc.first_valid_index())[["CAGR", "vol", "maxDD"]]
    # rasgos de la historia con la misma señal que los mundos, en las dos ventanas largas
    us = d.get("u_sig", d["u"]).reindex(idx)
    sig_np = r13._sma_signal(np.cumprod(1 + us.values)[None], 200)
    feats = {}
    for wname in list(windows)[:2]:
        f = path_features(uu.values[None], sig_np, idx.get_loc(windows[wname]))
        feats[wname] = {k: float(v[0]) for k, v in f.items()}
    in_mkt = {k: float(WX[k].loc[a28:last].mean()) for k in REAL_STRATS[:2]}
    return {"tests": pd.DataFrame(rows), "check": check, "features_w": feats,
            "features": feats[list(windows)[0]], "windows": list(windows), "dentro": in_mkt}


# ───────────────────────────── tablas ─────────────────────────────

def pvals(g: np.ndarray, se: np.ndarray) -> np.ndarray:
    """p unilateral del t NW; si el error es 0 (exposición nula: Δg ≡ 0) no hay evidencia, t = 0."""
    return norm.sf(np.divide(g, se, out=np.zeros_like(g), where=se > 0))


def real_row(real_t: pd.DataFrame, strat: str, bench: str, win: str, col: str = "Δg") -> float:
    """Cifra de la historia real; bench 'constante' = la de la media de la ventana (POST)."""
    bench = POST if bench == "constante" else bench
    rr = real_t[(real_t.ventana == win) & (real_t.estrategia == strat) & (real_t["frente a"] == bench)]
    return float(rr[col].iloc[0]) if len(rr) else np.nan


def effect_table(S: dict, real_t: pd.DataFrame, win: str) -> pd.DataFrame:
    """Ventaja verdadera de cada mundo: media (y mediana) de Δg en historias de 98 años con la
    referencia ex ante, y el percentil de la historia real (ventana común de 88 años, señal
    total) entre las historias de 88 años del mundo, con el MISMO estimador que la cifra real
    (frente a la constante: exposición media de la propia ventana)."""
    rows = []
    for s in STRATS:
        for w, res in S.items():
            t = res["tests"]
            g1, gc = t[(s, "1x", 98, "g")], t[(s, "constante", 98, "g")]
            r1, rc = (real_row(real_t, REAL_LIKE[s], b, win) for b in BENCH)
            rows.append({"mundo": w, "estrategia": s, "exposición_media": t[(s, "exp_ref")].mean(),
                         "Δg frente al 1x": g1.mean(), "± EE MC (1x)": g1.std() / np.sqrt(len(g1)),
                         "mediana (1x)": np.median(g1),
                         "Δg frente a constante (sincronización)": gc.mean(),
                         "± EE MC (constante)": gc.std() / np.sqrt(len(gc)),
                         "mediana (constante)": np.median(gc),
                         "efecto nivel (constante − 1x)": (g1 - gc).mean(),
                         "P(Δg > 0) frente a constante": (gc > 0).mean(),
                         "percentil historia (1x)": (t[(s, "1x", 88, "g")] < r1).mean(),
                         "percentil historia (constante)": (t[(s, POST_B, 88, "g")] < rc).mean()})
    return pd.DataFrame(rows).set_index(["estrategia", "mundo"])


BIAS_H = (5, 10, 30, 88, 98)


def bias_table(S: dict) -> pd.DataFrame:
    """Sesgo de la referencia ex post: Δg medio frente a la constante con la exposición media de
    la propia ventana menos el Δg medio con la exposición a largo plazo fijada ex ante, por
    horizonte (misma historia, diferencia emparejada); el Δg ex ante a 10 y 98 años; y la
    correlación entre la exposición media de la ventana y la rentabilidad logarítmica del
    mercado en ella, que es lo que produce el sesgo."""
    rows = []
    for s in STRATS:
        for w, res in S.items():
            t = res["tests"]
            r = {"estrategia": s, "mundo": w}
            for Y in BIAS_H:
                r[f"sesgo {Y} a"] = (t[(s, POST_B, Y, "g")] - t[(s, "constante", Y, "g")]).mean()
            for Y in (10, 98):
                r[f"Δg ex ante {Y} a"] = t[(s, "constante", Y, "g")].mean()
            e, m = t[(s, "exp", 10)], t[("mercado", 10)]
            r["corr(w̄, mercado) 10 a"] = _rowcorr(e[None], m[None])[0] if e.std() > 0 else np.nan
            rows.append(r)
    return pd.DataFrame(rows).set_index(["estrategia", "mundo"])


def _robust_sd(x: np.ndarray) -> float:
    q1, q3 = np.percentile(x, [25, 75])
    return (q3 - q1) / 1.349


def power_table(S: dict, horizons=KEY_H) -> pd.DataFrame:
    """Potencia del t NW y su calibración en cada mundo, comparación y horizonte.

    k = dispersión robusta (RIC/1,349) de Δg entre historias con los MISMOS
    parámetros / error NW mediano. Con k = 1 el error del test es correcto. La
    «potencia calibrada» divide cada t entre k (un test con el tamaño correcto)."""
    rows = []
    for w, res in S.items():
        t, draw = res["tests"], res["draw"]
        par = (draw >= 0).any()
        for (s, b) in COMP:
            for Y in horizons:
                g, se = t[(s, b, Y, "g")], t[(s, b, Y, "se")]
                p = pvals(g, se)
                if par:   # desviaciones respecto de la media de su juego de parámetros
                    dev = g - pd.Series(g).groupby(draw).transform("mean").values
                    rsd = _robust_sd(dev) * np.sqrt(PATHS_PER_DRAW / (PATHS_PER_DRAW - 1))
                else:
                    rsd = _robust_sd(g)
                k = rsd / np.median(se)
                p0 = pvals(g - g.mean(), se)     # el mismo mundo desplazado a Δg medio = 0
                rows.append({"mundo": w, "comparación": COMP_LABEL[(s, b)], "años": Y,
                             "Δg_medio": g.mean(), "Δg_mediano": np.median(g),
                             "potencia": (p < ALPHA).mean(), "potencia_calibrada": (pvals(g, se * k) < ALPHA).mean(),
                             "p_mediano": np.median(p), "P(0,2 ≤ p ≤ 0,4)": ((p >= 0.2) & (p <= 0.4)).mean(),
                             "P(0,2 ≤ p ≤ 0,4) con Δg = 0": ((p0 >= 0.2) & (p0 <= 0.4)).mean(),
                             "EE_NW_mediano": np.median(se), "DE_robusta_misma_param": rsd, "k": k,
                             "DE/EE (con colas)": g.std(ddof=1) / np.sqrt(np.mean(se ** 2)),
                             "efecto_mín_detectable_80%": (Z_A + Z_POW) * rsd})
    return pd.DataFrame(rows)


def features_table(S: dict, real: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Medianas de los rasgos en cada mundo (98 años) y en las dos ventanas históricas, y el
    percentil de la historia: 1928–2026 entre historias de 98 años, 1938–2026 entre las de 88."""
    w28, w38 = real["windows"][:2]
    f28, f38 = real["features_w"][w28], real["features_w"][w38]
    cols = list(f28)
    rows = {f"historia {w28}": f28, f"historia {w38}": f38}
    pct, pct88 = {}, {}
    for w, res in S.items():
        f, f88 = res["features"], res["features88"]
        rows[w] = {k: float(np.nanmedian(f[k])) for k in cols}
        pct[w] = {k: float(np.nanmean(f[k] < f28[k])) for k in cols}
        pct88[w] = {k: float(np.nanmean(f88[k] < f38[k])) for k in cols}
    return pd.DataFrame(rows).T[cols], pd.DataFrame(pct).T[cols], pd.DataFrame(pct88).T[cols]


def param_table(S: dict) -> tuple[pd.DataFrame, float]:
    rows = {}
    for w in ("GJR-GARCH", "GARCH simétrico", "HMM medias iguales"):
        P = pd.DataFrame(S[w]["params"]).drop(columns="rechazos", errors="ignore")
        for c in P:
            q = P[c].quantile([0.05, 0.5, 0.95])
            rows[(w if w != "HMM medias iguales" else "HMM (ambos)", c)] = {
                "p5": q[0.05], "mediana": q[0.5], "p95": q[0.95]}
    rej = pd.DataFrame(S["GJR-GARCH"]["params"])["rechazos"]
    return pd.DataFrame(rows).T, float(rej.sum() / (rej.sum() + len(rej)))


def fixed_fits(u: np.ndarray) -> pd.DataFrame:
    """Parámetros estimados con la historia completa, como referencia de los sorteos."""
    rows = {}
    for kind, w in (("gjr", "GJR-GARCH"), ("garch", "GARCH simétrico")):
        sv = garch_setup(u, kind)
        p = sv["p"]
        for c in ("alpha", "beta", "gamma"):
            rows[(w, c)] = p[c]
        rows[(w, "α+γ/2+β")] = p["alpha"] + p["gamma"] / 2 + p["beta"]
        rows[(w, "p_eff")] = sv["p_eff"]
        rows[(w, "m_anual")] = sv["m"] * 252
    _, par = hmm_paths(u, 1, 1, np.random.default_rng(0))
    for c, v in par.items():
        rows[("HMM (ambos)", c)] = v
    return pd.Series(rows).to_frame("historia completa")


MECH = ["μ_sobre − μ_bajo", "σ_bajo / σ_sobre", "vol_anual"]
DEC_COLS = ["coste sin información (a + d·(vol − v̄))", "canal de medias (b·brecha)",
            "canal de volatilidad (c·ln cociente)", "coste + sólo medias", "coste + sólo volatilidad",
            "predicho", "real (señal total)"]


def mechanism_table(S: dict, real: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descomposición descriptiva del valor de sincronizar (Δg frente a constante, 98 años).

    MCO sobre todas las historias de todos los mundos (sin el 1% de cada cola):
    Δg = a + b·(μ_sobre − μ_bajo) + c·ln(σ_bajo/σ_sobre) + d·(vol − v̄), con v̄ la
    volatilidad media de esas historias; a es el Δg previsto cuando la señal no
    separa ni medias ni volatilidades, a una volatilidad típica (dentro del
    soporte). La variable dependiente usa la referencia ex post (media de la
    propia historia), el mismo estimador que la cifra real con la que se compara.
    Se aplica a los rasgos reales de cada ventana (misma señal sobre el índice
    total y misma ventana que el Δg real). No es causal: resume qué rasgos de una
    historia acompañan a una LRS que gana."""
    rows, dec = {}, {}
    rt = real["tests"]
    for s in STRATS:
        X = np.vstack([np.c_[S[w]["features"][MECH[0]], np.log(S[w]["features"][MECH[1]]),
                             S[w]["features"][MECH[2]]] for w in WORLDS])
        y = np.concatenate([S[w]["tests"][(s, POST_B, 98, "g")] for w in WORLDS])
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        lo, hi = np.percentile(y[ok], [1, 99])
        ok &= (y > lo) & (y < hi)
        vbar = X[ok, 2].mean()
        A = np.c_[np.ones(ok.sum()), X[ok, :2], X[ok, 2] - vbar]
        b, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        r2 = 1 - ((y[ok] - A @ b) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
        vq = np.percentile(X[ok, 2], [0.5, 99.5])
        rows[s] = {"a: Δg sin información, a vol v̄": b[0], "b: por unidad de μ_sobre − μ_bajo": b[1],
                   "c: por unidad de ln(σ_bajo/σ_sobre)": b[2], "d: por unidad de vol": b[3],
                   "v̄ (vol media)": vbar, "vol p0,5": vq[0], "vol p99,5": vq[1],
                   "R²": r2, "historias": int(ok.sum())}
        for win, fr in real["features_w"].items():
            g_real = real_row(rt, REAL_LIKE[s], POST, win)
            if np.isnan(g_real):
                continue
            cost = b[0] + b[3] * (fr[MECH[2]] - vbar)
            cm, cv = b[1] * fr[MECH[0]], b[2] * np.log(fr[MECH[1]])
            dec[(s, win)] = {"brecha de medias": fr[MECH[0]], "σ fuera / σ dentro": fr[MECH[1]],
                             "vol": fr[MECH[2]], DEC_COLS[0]: cost, DEC_COLS[1]: cm, DEC_COLS[2]: cv,
                             DEC_COLS[3]: cost + cm, DEC_COLS[4]: cost + cv, DEC_COLS[5]: cost + cm + cv,
                             DEC_COLS[6]: g_real}
    md = pd.DataFrame(dec).T
    md.index.names = ["estrategia", "ventana"]
    return pd.DataFrame(rows).T, md


# ───────────────────────────── figuras ─────────────────────────────

VS = {"1x": "frente al 1x", "constante": "frente a su exposición media constante"}


def _style_world(w):
    return dict(color=COLOR[w], marker=MARKER[WORLDS[w][2]], ls="--" if w in DASHED else "-")


def fig_effects(S: dict, real_t: pd.DataFrame, strat: str, rep: Report, fname: str, win: str,
                bias88: tuple[float, float]):
    from matplotlib.lines import Line2D

    worlds = FIG_WORLDS[::-1]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 10.5))
    if strat == "LRS-Kelly":
        reals = [(REAL_LIKE[strat], ":", "historia, EWMA y señal total"),
                 (REAL_PX[strat], "-.", "historia, señal de precio")]
    else:
        reals = [(REAL_LIKE[strat], ":", "historia, señal total (original)"),
                 (REAL_PX[strat], "-.", "historia, señal de precio")]
    for ax, b in zip(axes, BENCH):
        for i, w in enumerate(worlds):
            t = S[w]["tests"]
            g10, g30, g98 = (t[(strat, b, Y, "g")] * 100 for Y in (10, 30, 98))
            lo10, hi10 = np.percentile(g10, [5, 95])
            lo30, hi30 = np.percentile(g30, [5, 95])
            ax.plot([lo10, hi10], [i, i], color=COLOR[w], lw=1.5, alpha=0.55, solid_capstyle="round")
            ax.plot([lo30, hi30], [i, i], color=COLOR[w], lw=6, alpha=0.9, solid_capstyle="round")
            ax.plot(g98.mean(), i, marker=MARKER[WORLDS[w][2]], color="white", mec=INK, mew=1.5, ms=10,
                    zorder=5)
        ax.axvline(0, color="0.55", lw=1)
        vals = [(real_row(real_t, k, b, win) * 100, ls, lab) for k, ls, lab in reals]
        for j, (v, ls, lab) in enumerate(vals):
            ax.axvline(v, color=INK, ls=ls, lw=1.8)
            right = j == 0 and (len(vals) == 1 or v >= vals[1][0])
            right = right or (j == 1 and v > vals[0][0])
            # solo la cifra junto a su línea: qué es cada línea lo dice la leyenda
            ax.annotate(_pp(v / 100), (v, len(worlds) - 0.35),
                        xytext=(5 if right else -5, 0), textcoords="offset points", fontsize=11, color=INK,
                        va="bottom", ha="left" if right else "right")
        ax.set_yticks(range(len(worlds)), worlds)
        ax.set_ylim(-0.7, len(worlds) + 0.5)
        ax.set(xlabel="Δg (puntos de crecimiento logarítmico al año)", title=f"{strat} {VS[b]}")
        ax.grid(axis="x", color="0.9")
    handles = [Line2D([], [], color="0.4", lw=6, label="5–95% de las historias de 30 años"),
               Line2D([], [], color="0.4", lw=1.5, alpha=0.6, label="5–95% de las historias de 10 años"),
               Line2D([], [], marker="o", color="white", mec=INK, mew=1.5, ms=9, ls="",
                      label="media a 98 años (ventaja verdadera)")]
    handles += [Line2D([], [], color=INK, ls=ls, lw=1.8, label=f"{lab} ({win[:9]})") for _, ls, lab in reals]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=11,
                   frameon=False)
    comma_fig(fig)
    extra = ("varianza EWMA y señal sobre el índice total, como en los mundos (comparable), y la misma "
             "regla con la señal de precio (sensibilidad)") if strat == "LRS-Kelly" else (
             "la señal original sobre el índice de rentabilidad total, como en los mundos y en Gayed y "
             "Bilello (comparable), y la señal de precio (sensibilidad del laboratorio)")
    rep.fig(fig, fname, f"{strat}: ventaja de crecimiento logarítmico (Δg) frente al 1x (arriba) y frente a "
            "una cartera constante con la misma exposición media (abajo), en cada mundo. El símbolo blanco es "
            "la ventaja verdadera (media de 1.000 historias de 98 años); la barra gruesa y la fina, el rango "
            "del 5 al 95% de lo que mediría un investigador con 30 y con 10 años de datos. En los mundos la "
            "exposición de la cartera constante es la media a largo plazo, fijada ex ante. Las líneas negras "
            f"son la historia real del S&P 500 ({win}) con {extra}; su cartera constante usa la exposición "
            f"media de esa misma ventana de 88 años, cuyo sesgo a ese horizonte es de {_pp(bias88[0], 2)} a "
            f"{_pp(bias88[1], 2)} pp de media en los mundos (sección 3).")


def fig_power(S: dict, strat: str, rep: Report, fname: str):
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 10.0), sharex=True)
    xt = [5, 10, 20, 30, 40, 50, 75, 98]
    for ax, b in zip(axes, BENCH):
        for w in FIG_WORLDS:
            t = S[w]["tests"]
            pw = [(pvals(t[(strat, b, Y, "g")], t[(strat, b, Y, "se")]) < ALPHA).mean() * 100 for Y in HORIZONS]
            ax.plot(HORIZONS, pw, lw=2, ms=7, label=w, **_style_world(w))
        # líneas de referencia rotuladas fuera del gráfico, a la derecha: dentro las cruzaban las curvas
        for yv in (5, 80):
            ax.axhline(yv, color="0.55", lw=1, ls=":")
            ax.text(1.01, yv, f"{yv}%", transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=11, color="0.35", clip_on=False)
        ax.set(ylim=(0, 100), ylabel="Probabilidad de rechazar H₀ (%)",
               title=f"Potencia: {strat} {VS[b]}")
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(color="0.92")
    axes[0].tick_params(labelbottom=False)
    axes[1].set(xlabel="Años de datos del investigador", xticks=xt)
    legend_band(fig, *axes[0].get_legend_handles_labels(), ncol=3, fontsize=11)
    rep.fig(fig, fname, f"{strat}: probabilidad de que un test t con errores Newey-West (63 retardos, "
            "unilateral al 5%) declare significativa la ventaja, según los años de datos, en cada mundo. "
            "Abajo, la cartera constante tiene la exposición media a largo plazo de la estrategia en ese "
            "mundo, fijada ex ante. Trazo discontinuo: mundos a los que se ha quitado un rasgo (dependencia, "
            "asimetría o diferencia de medias). Donde la ventaja verdadera es nula o negativa la curva mide "
            "falsas alarmas; en los mundos GARCH y HMM el test es algo optimista (k > 1 en la tabla de "
            "calibración).")


def fig_mde(P: pd.DataFrame, real_t: pd.DataFrame, rep: Report, win: str):
    from matplotlib.ticker import FuncFormatter, FixedLocator

    fig, axes = plt.subplots(2, 1, figsize=(11, 11), sharex=True)
    Pf = P[P.mundo.isin(FIG_WORLDS)]
    for ax, s in zip(axes, STRATS):
        for b, col in zip(BENCH, (PALETTE[0], PALETTE[1])):
            q = Pf[Pf["comparación"] == COMP_LABEL[(s, b)]].groupby("años")["efecto_mín_detectable_80%"]
            med, lo, hi = q.median() * 100, q.min() * 100, q.max() * 100
            ax.fill_between(med.index, lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(med.index, med, color=col, lw=2.2, marker="o", ms=6,
                    label=f"{COMP_LABEL[(s, b)]}: efecto detectable al 80%")
            v = real_row(real_t, REAL_LIKE[s], b, win) * 100
            ax.axhline(abs(v), color=col, ls=":", lw=2)
            ax.annotate(f"{COMP_LABEL[(s, b)]}, historia: {_pp(v / 100)} pp",
                        (5, abs(v)), xytext=(0, 4), textcoords="offset points", ha="left", fontsize=11,
                        color=INK)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 7, 10, 15, 20, 30, 50]))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.set(ylabel="pp de crecimiento al año (escala log)", title=f"Efecto mínimo detectable: {s}")
        ax.grid(color="0.92")
        ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    axes[1].set(xlabel="Años de datos", xticks=[5, 10, 20, 30, 40, 50, 75, 98])
    rep.fig(fig, "efecto_minimo_detectable", "Ventaja verdadera mínima que un test unilateral al 5% bien "
            "calibrado detectaría con un 80% de probabilidad (2,49 veces la dispersión real de Δg entre "
            "historias), según los años de datos. Línea: mediana de los ocho mundos; sombra: del mundo más "
            f"favorable al menos favorable. La cartera constante tiene la exposición media a largo plazo, "
            f"fijada ex ante. Las líneas de puntos marcan, en valor absoluto, la ventaja estimada en la "
            f"historia real ({win}) con la misma señal que los mundos (índice de rentabilidad total; "
            "LRS-Kelly con varianza EWMA): si queda por debajo de la curva, una ventaja verdadera de ese "
            "tamaño no se detectaría con fiabilidad.")


def fig_features(S: dict, real: dict, rep: Report):
    keys = [("μ_sobre − μ_bajo", 100, "pp/año", "Rentabilidad anual sobre la media − bajo la media"),
            ("σ_bajo / σ_sobre", 1, "cociente", "Volatilidad bajo la media / sobre la media"),
            ("apalancamiento ρ(R_m, RV_m+1)", 1, "correlación", "Apalancamiento: ρ(mes, varianza del mes siguiente)"),
            ("tendencia ρ(R_12m, R_m+1)", 1, "correlación", "Tendencia: ρ(12 meses, mes siguiente)")]
    worlds = FIG_WORLDS[::-1]
    fig, axes = plt.subplots(4, 1, figsize=(11, 16))
    for ax, (k, sc, unit, title) in zip(axes, keys):
        for i, w in enumerate(worlds):
            v = S[w]["features"][k] * sc
            lo, med, hi = np.nanpercentile(v, [5, 50, 95])
            ax.plot([lo, hi], [i, i], color=COLOR[w], lw=5, alpha=0.85, solid_capstyle="round")
            ax.plot(med, i, marker=MARKER[WORLDS[w][2]], color="white", mec=INK, mew=1.4, ms=9, zorder=5)
        rv = real["features"][k] * sc
        ax.axvline(rv, color=INK, ls=":", lw=1.8)
        ax.annotate(f"historia: {_n(rv)}", (rv, len(worlds) - 0.4),
                    xytext=(4, 0), textcoords="offset points", fontsize=11, va="bottom")
        ax.set_yticks(range(len(worlds)), worlds)
        ax.set_ylim(-0.7, len(worlds) + 0.3)
        ax.set(title=title, xlabel=unit)
        ax.grid(axis="x", color="0.9")
    rep.fig(fig, "rasgos_mundos", "Cuatro rasgos del mercado que determinan la ventaja de la LRS, medidos en "
            "las historias de 98 años de cada mundo (barra: 5–95% de las historias; símbolo: mediana) y en "
            f"el S&P 500 de {real['windows'][0]} (línea de puntos). «Sobre/bajo la media» se refiere a la señal "
            "de la LRS sobre el índice de rentabilidad total tal como se ejecuta (la del cierre de dos sesiones "
            "antes); los meses son bloques de 21 sesiones.")


def fig_mechanism(S: dict, real: dict, real_t: pd.DataFrame, rep: Report):
    fig, axes = plt.subplots(2, 1, figsize=(11, 11))
    xs = np.concatenate([S[w]["features"]["μ_sobre − μ_bajo"] for w in FIG_WORLDS]) * 100
    xr = [real["features_w"][w]["μ_sobre − μ_bajo"] * 100 for w in real["windows"][:2]]
    xlim = np.nanpercentile(xs, [0.5, 99.5])
    xlim = (min(xlim[0], min(xr) - 1), max(xlim[1], max(xr) + 1))
    stars = {real["windows"][0]: dict(marker="*", color=INK, s=300),
             real["windows"][1]: dict(marker="*", color="white", edgecolor=INK, linewidth=1.6, s=300)}
    for ax, s in zip(axes, STRATS):
        ys = np.concatenate([S[w]["tests"][(s, POST_B, 98, "g")] for w in FIG_WORLDS]) * 100
        for w in FIG_WORLDS:
            f, t = S[w]["features"], S[w]["tests"]
            x = f["μ_sobre − μ_bajo"] * 100
            y = t[(s, POST_B, 98, "g")] * 100
            ax.scatter(x, y, s=6, color=COLOR[w], alpha=0.25, lw=0)
            ax.scatter(np.nanmedian(x), np.mean(y), s=120, color=COLOR[w], marker=MARKER[WORLDS[w][2]],
                       edgecolor="white", linewidth=1.5, zorder=5, label=w)
        for win, st in stars.items():
            v = real_row(real_t, REAL_LIKE[s], POST, win)
            if np.isfinite(v):
                ax.scatter(real["features_w"][win]["μ_sobre − μ_bajo"] * 100, v * 100, zorder=6,
                           label=f"historia {win}", **st)
        ax.axhline(0, color="0.55", lw=1)
        ax.axvline(0, color="0.55", lw=1)
        ax.set(xlabel="Rentabilidad anual sobre la media − bajo la media (pp)",
               ylabel="Δg frente a constante (pp/año)", title=f"{s}: valor de sincronizar frente a tendencia",
               xlim=xlim, ylim=np.nanpercentile(ys, [0.5, 99.5]))
        ax.grid(color="0.92")
    axes[0].legend(loc="upper left", fontsize=9.5, ncol=2, framealpha=0.95)
    rep.fig(fig, "mecanismo", "Cada punto pequeño es una historia de 98 años: en horizontal, cuánto más rinde "
            "el índice cuando la señal de la LRS está dentro que cuando está fuera; en vertical, lo que "
            "sincronizar añade al crecimiento frente a una cartera constante con la exposición media de esa "
            "misma historia. Los símbolos grandes son la media de cada mundo. Las estrellas son la historia "
            f"real con la misma señal (índice total) y rasgos y Δg medidos en la misma ventana: negra "
            f"{real['windows'][0]}, blanca {real['windows'][1]} (la LRS-Kelly, con varianza EWMA, sólo existe "
            "desde 1938). Ejes recortados al 1% central de las historias.")


# ───────────────────────────── informe ─────────────────────────────

def compute(cache: str | None = None) -> tuple[dict, dict, float]:
    if cache and Path(cache).exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    D = r13.load()
    d = D["S&P 500"]
    u = d["u"].dropna().values
    rf_d, f_d = float(d["rf"].mean()), float(d["f"].mean())
    real = real_history(D)
    S = simulate(u, rf_d, f_d, d["spread"], d["er"])
    real["u"] = u
    real["fijos"] = fixed_fits(u)
    real["rf_f"] = (rf_d * 252, f_d * 252, d["spread"], d["er"])
    out = (S, real, time.time() - t0)
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump(out, fh)
    return out


def _table(rep: Report, df: pd.DataFrame, pct: set | None = None, digits: int = 2):
    """Tabla con índice múltiple aplanado en columnas (markdown legible)."""
    if isinstance(df.index, pd.MultiIndex):
        rep.table(df.reset_index(), pct=pct, index=False, digits=digits)
    else:
        rep.table(df, pct=pct, digits=digits)


def run(cache: str | None = None):
    S, real, secs = compute(cache)
    rep = Report("r18_worlds", "R18 — Mundos sintéticos: ablación y potencia estadística")
    rt = real["tests"]
    w28, win, w30, w10 = real["windows"]   # win: ventana común con la LRS-Kelly (desde 1938)
    E = effect_table(S, rt, win)
    P = power_table(S, HORIZONS)
    PK = P[P["años"].isin(KEY_H)]
    ft, fp, fp88 = features_table(S, real)
    Bt = bias_table(S)
    mt, md = mechanism_table(S, real)
    rep.h("Resumen")
    for line in summary(S, real, E, P, ft, fp, fp88, Bt, mt, md):
        rep.p(line)
    rep.h("Diseño")
    rep.p(__doc__.split("\n", 2)[2])
    rf_a, f_a, spread, er = real["rf_f"]
    rep.p(f"Ejecución: {N_PATHS:,} historias por mundo, {len(WORLDS)} mundos, "
          f"{f'{secs / 60:.1f}'.replace('.', ',')} minutos con {N_JOBS} procesos. T-bill {_pc(rf_a, 2)} y "
          f"financiación {_pc(f_a, 2)} anuales constantes (medias 1928–2026), sobrecoste del swap "
          f"{_pc(spread, 3).replace('-', '−')} y gastos {_pc(er, 2)}. Las historias de 10, 30 y 88 años son "
          "los primeros años evaluados de cada historia de 98: el proceso es estacionario y las estrategias "
          "sólo usan el pasado, así que su distribución es la de una historia de esa longitud. Nada mira el "
          "futuro dentro de una historia. Lo que usa la muestra completa 1928–2026 son los propios generadores "
          "(el bootstrap remuestrea la historia entera y los modelos se estiman con remuestreos de ella). La "
          "exposición de la cartera constante de referencia sale de OTRAS historias del mismo mundo: es ex "
          "ante para cada historia, pero es un dato que un investigador real no tendría (un oráculo). La "
          "versión ex post (media de la propia ventana) se usa sólo para medir su sesgo, para la regresión "
          "de la descomposición y para situar la historia real entre las historias de 88 años con el mismo "
          "estimador.".replace("1,000", "1.000"))

    # 1. historia real
    rep.h("1. La historia real con el mismo estadístico")
    rep.p("Las series de la historia real son las de R13 (r13.strategies y r13.price_variants, más la "
          "LRS-Kelly con varianza EWMA): S&P 500 desde el primer día con previsión GJR, orden al cierre "
          "siguiente.")
    rep.table(real["check"], pct={"CAGR", "vol", "maxDD"}, digits=2)
    _table(rep, rt.set_index(["ventana", "estrategia", "frente a"]), pct={"Δg", "error_NW"}, digits=3)
    dn = real["dentro"]
    rep.p("Δg y su error en % anual; p unilateral (H1: la estrategia crece más). Exposiciones en veces el "
          "índice. «constante (media de la ventana)» = fracción media del 3x de la estrategia en esa misma "
          "ventana, reequilibrada a diario, sin la señal: es el estimador ex post de la primera versión y el "
          "que se compara con los mundos a 88 años. «constante (media previa)» = la fracción media ejecutada "
          "por la estrategia desde su inicio hasta el día anterior a la ventana. Se conoce al empezar la "
          "ventana (ex ante) y sólo se da cuando hay al menos 10 años previos. Es la versión operable de la "
          "referencia a largo plazo de los mundos.")
    rep.p("«LRS 3x (original)» = la LRS de Gayed y Bilello: la media de 200 sesiones del índice de "
          "rentabilidad total (con dividendos: el índice oficial ^SP500TR desde 1988 y, antes, ^GSPC con el D/P "
          "de Shiller del mes anterior para no mirar el futuro), la misma señal que en los mundos y en los rasgos. «señal total» en la LRS-Kelly: "
          "la misma señal, con la varianza prevista por EWMA como en los mundos (comparable); la de R13 la "
          "prevé con GJR-GARCH. «señal de precio» = la misma regla sobre el índice de precio (^GSPC), una "
          "sensibilidad del laboratorio. Como el índice de precio va por detrás del total en el rendimiento "
          "por dividendo δ, estar por encima de su media de 200 sesiones equivale aproximadamente a exigir al "
          "índice total un margen de δ × 0,4 años sobre la suya (≈ 1,6% con δ = 4%). Por eso la señal de "
          f"precio está dentro menos tiempo: el {_pc(dn[r13.PX_LRS])} de los días desde 1928, frente al "
          f"{_pc(dn['LRS 3x (original)'])} de la original. Los mundos no tienen dividendos y no pueden "
          "reproducir esa variante.")

    # 2. diagnóstico de los mundos
    rep.h("2. ¿Se parecen los mundos a la historia?")
    rep.h("Diagnóstico de r13.world_diagnostics (primeros 30 años evaluados)", 3)
    rep.table(pd.DataFrame({w: S[w]["diagnóstico"] for w in WORLDS}).T, digits=3)
    rep.h("Rasgos que alimentan la LRS (mediana de las historias de 98 años)", 3)
    rep.table(ft, pct={"crec_log_anual", "vol_anual", "fracción_sobre_media", "μ_sobre − μ_bajo"}, digits=3)
    rep.p(f"Percentil de la historia {w28} dentro de cada mundo (fracción de historias de 98 años con un "
          "valor menor; cerca de 0% o de 100% = la historia sería rara en ese mundo):")
    rep.table(fp, pct=set(fp.columns), digits=2)
    rep.p(f"Lo mismo para la historia {win} entre las historias de 88 años de cada mundo:")
    rep.table(fp88, pct=set(fp88.columns), digits=2)
    fig_features(S, real, rep)
    rep.h("Parámetros sorteados (reestimados en remuestreos por bloques de la historia)", 3)
    ptab, rej = param_table(S)
    ptab = ptab.join(real["fijos"])
    ptab.index = pd.MultiIndex.from_tuples(ptab.index, names=["modelo", "parámetro"])
    _table(rep, ptab, digits=4)
    rep.p("La columna «historia completa» es la estimación con los 98 años sin remuestrear. p_eff es la "
          "persistencia efectiva de la varianza con los residuos remuestreados (α·E[z²] + γ·E[z²·1(z<0)] + β) "
          "tras la corrección de r13, que reescala las partes negativa y positiva de los residuos. HMM: "
          "medias y volatilidades anualizadas de las log-rentabilidades de cada régimen (antes de igualar "
          "las medias) y duración media esperada de cada régimen.")
    rep.p(f"Se descartaron y se volvieron a sortear los remuestreos con p_eff ≥ 0,9999 en el GJR o en "
          f"el GARCH (varianza no estacionaria: ω ≤ 0 y r13.gen_paths devolvería NaN): {_pc(rej, 1)} de los "
          "intentos. La distribución de parámetros queda truncada a la región estacionaria.")

    # 3. ventaja verdadera
    rep.h("3. La ventaja verdadera en cada mundo (98 años)")
    _table(rep, E, pct={"Δg frente al 1x", "± EE MC (1x)", "mediana (1x)", "Δg frente a constante (sincronización)",
                      "± EE MC (constante)", "mediana (constante)", "efecto nivel (constante − 1x)",
                      "P(Δg > 0) frente a constante", "percentil historia (1x)",
                      "percentil historia (constante)"}, digits=2)
    rep.p("Media y mediana entre historias de Δg a 98 años; «± EE MC» es el error de Monte Carlo de la "
          "media. La cartera constante tiene la exposición media a largo plazo de la estrategia en ese mundo "
          "(columna exposición_media), fijada ex ante. Δg frente al 1x = efecto nivel + sincronización. "
          "«percentil historia» = fracción de historias de 88 años del mundo con un Δg menor que el de la "
          f"historia real ({win}), con la misma señal (índice total; LRS-Kelly con varianza EWMA) y el mismo "
          "estimador: frente a la constante, la exposición media de la propia ventana en los dos lados. Las "
          "historias en que la LRS estaba dentro en un día de caída ≥ 33,3% (el 3x pierde el 100%) cuentan ese "
          "día como −99,9%, igual que el SPA de R13; en los mundos GARCH eso arrastra la media por debajo de "
          "la mediana:")
    rep.table(pd.Series({w: S[w]["ruina_LRS"].mean() for w in WORLDS}).to_frame("P(ruina del 3x con la LRS dentro)"),
              pct={"P(ruina del 3x con la LRS dentro)"}, digits=3)
    for s, fname in (("LRS 3x (original)", "efecto_lrs3x"), ("LRS-Kelly", "efecto_lrs_kelly")):
        b88 = Bt.loc[s].loc[FIG_WORLDS, "sesgo 88 a"]
        fig_effects(S, rt, s, rep, fname, win, (b88.min(), b88.max()))
    rep.h("El sesgo de la referencia ex post", 3)
    _table(rep, Bt, pct={c for c in Bt.columns if not c.startswith("corr")}, digits=2)
    rep.p("«sesgo Y a» = Δg medio frente a la constante con la exposición media de la propia ventana de Y "
          "años (ex post, la primera versión) menos el Δg medio con la exposición a largo plazo fijada ex "
          "ante, en las mismas historias. corr(w̄, mercado) = correlación entre historias de la exposición "
          "media de la estrategia en una ventana de 10 años y la rentabilidad logarítmica del índice en esa "
          "ventana. Como la estrategia está más tiempo dentro cuando el mercado sube, la referencia ex post "
          "lleva más apalancamiento en las ventanas buenas y le quita a la estrategia parte de su ventaja. "
          "Con la referencia ex ante el Δg medio apenas depende del horizonte.")
    rep.h("Descomposición: ¿tendencia o volatilidad?", 3)
    rep.table(mt, pct={"a: Δg sin información, a vol v̄", "v̄ (vol media)", "vol p0,5", "vol p99,5"}, digits=3)
    rep.p("Regresión descriptiva, historia a historia, del valor de sincronizar a 98 años (referencia ex "
          "post, como la cifra real) sobre tres rasgos de esa misma historia: la brecha de rentabilidad anual "
          "del índice entre los días con la señal dentro y fuera (en fracción), el logaritmo del cociente de "
          "volatilidades fuera/dentro y la volatilidad anual centrada en su media v̄. Así, a es el Δg que se "
          "espera cuando la señal no separa ni medias ni volatilidades, a una volatilidad típica, dentro del "
          "soporte de los datos (la primera versión evaluaba el intercepto con volatilidad 0, fuera de él). "
          "Aplicada a los rasgos reales de cada ventana (misma señal sobre el índice total y misma ventana que "
          "el Δg real), reparte ese Δg así (en % anual; «coste» = a + d·(vol − v̄), el Δg previsto sin "
          "información a la volatilidad de esa ventana):")
    _table(rep, md, pct={"brecha de medias", "vol"} | set(DEC_COLS), digits=2)
    fig_mechanism(S, real, rt, rep)

    # 4. potencia
    rep.h("4. Potencia: ¿con qué probabilidad se detectaría?")
    cols = {"potencia": "potencia", "potencia_calibrada": "pot. calibrada", "p_mediano": "p mediano"}
    tabs = []
    for c, lab in cols.items():
        x = PK.pivot_table(index=["comparación", "mundo"], columns="años", values=c, sort=False)
        x.columns = [f"{lab} {y} a" for y in x.columns]
        tabs.append(x)
    tab = pd.concat(tabs, axis=1)
    _table(rep, tab, pct={c for c in tab.columns if c.startswith("pot")}, digits=3)
    rep.p("Potencia = fracción de historias en que el t Newey-West unilateral rechaza al 5%; frente a la "
          "constante, con la exposición a largo plazo fijada ex ante. «pot. calibrada» = lo mismo con el error "
          "multiplicado por k (siguiente tabla), es decir, con un test del tamaño correcto. p mediano = el "
          "p-valor típico que vería un investigador en ese mundo.")
    rep.h("¿Qué significa un p ≈ 0,3?", 3)
    P30 = PK[PK["años"] == 30]
    pz = P30.pivot_table(index="mundo", columns="comparación", values="P(0,2 ≤ p ≤ 0,4)", sort=False)
    pz.loc["Δg = 0 exacto, test bien calibrado (p uniforme)"] = 0.20
    rep.table(pz, pct=set(pz.columns), digits=2)
    rep.p("Probabilidad de obtener un p-valor entre 0,2 y 0,4 con 30 años de datos en cada mundo. La última "
          "fila es la referencia de efecto nulo exacto: si Δg = 0 y el test tiene el tamaño correcto, el p es "
          "uniforme y cae en ese intervalo el 20% de las veces. El cociente entre dos filas es el factor de "
          "Bayes que aporta observar «p ≈ 0,3» a favor de un mundo frente al otro.")
    tc = "LRS 3x − constante"
    q = P30[P30["comparación"] == tc].set_index("mundo")
    bf = pd.DataFrame({"Δg medio 30 a": q["Δg_medio"], "p mediano": q["p_mediano"],
                       "P(0,2 ≤ p ≤ 0,4)": q["P(0,2 ≤ p ≤ 0,4)"],
                       "ídem, mismo mundo con Δg = 0": q["P(0,2 ≤ p ≤ 0,4) con Δg = 0"],
                       "FB frente a p uniforme": q["P(0,2 ≤ p ≤ 0,4)"] / 0.20,
                       "FB frente al mismo mundo con Δg = 0":
                           q["P(0,2 ≤ p ≤ 0,4)"] / q["P(0,2 ≤ p ≤ 0,4) con Δg = 0"]})
    rep.table(bf, pct={"Δg medio 30 a", "P(0,2 ≤ p ≤ 0,4)", "ídem, mismo mundo con Δg = 0"}, digits=3)
    rep.p(f"{tc} con 30 años. «mismo mundo con Δg = 0» desplaza todas las historias del mundo para que su "
          "Δg medio sea 0, con el mismo ruido y el mismo error Newey-West: es el efecto nulo exacto sin "
          "suponer que el test está bien calibrado. FB = factor de Bayes a favor del mundo de la fila.")
    rep.h("Efecto mínimo detectable y calibración del error Newey-West", 3)
    cal = PK.set_index(["comparación", "mundo", "años"])[
        ["Δg_medio", "Δg_mediano", "EE_NW_mediano", "DE_robusta_misma_param", "k", "DE/EE (con colas)",
         "efecto_mín_detectable_80%"]]
    _table(rep, cal, pct={"Δg_medio", "Δg_mediano", "EE_NW_mediano", "DE_robusta_misma_param",
                        "efecto_mín_detectable_80%"}, digits=2)
    rep.p("k = dispersión robusta de Δg (rango intercuartílico / 1,349) entre historias con los mismos "
          "parámetros, dividida por el error NW mediano: k ≈ 1 significa que el test sabe cuánto ruido "
          "tiene; k > 1, que se cree más preciso de lo que es. «DE/EE (con colas)» usa la desviación "
          "típica ordinaria de todas las historias (incluye la incertidumbre de parámetros y las historias "
          "con ruina). Efecto mínimo detectable = (z₀,₉₅ + z₀,₈₀) × dispersión robusta: la ventaja "
          "verdadera que un test bien calibrado detectaría 8 de cada 10 veces.")
    fig_power(S, "LRS 3x (original)", rep, "potencia_lrs3x")
    fig_power(S, "LRS-Kelly", rep, "potencia_lrs_kelly")
    fig_mde(P, rt, rep, win)

    # 5. distribución de resultados
    rep.h("5. Resultados a 30 y a 98 años (formato de r13.mc_summary)")
    for w in WORLDS:
        rep.h(w, 3)
        t = pd.concat({"30 años": summarize(S[w]["m30"]), "98 años": summarize(S[w]["m98"])},
                      names=["horizonte", "estrategia"])
        _table(rep, t, pct=set(t.columns), digits=3)

    tit = provenance.write_titular(rep.dir, titular_rows(real), "r18_worlds")
    rep.h("p de titular exportados (p_titular.csv)")
    rep.p("La lista cerrada de p de titular de la historia real de este informe (t Newey-West unilateral), "
          "que R19 lee para corregirlos junto a los de R16 y R17. Procedencia: commit " + tit.commit.iloc[0][:12]
          + (" con cambios sin commit en el código del que depende" if tit.cambios_sin_commit.iloc[0] else "")
          + f"; huella del código {tit.huella_código.iloc[0]}.")
    rep.table(tit[["contraste", "p"]], index=False, digits=4)

    rep.h("Salvedades")
    ruin = [S[w]["ruina_LRS"].mean() for w in WORLDS if WORLDS[w][2] == "garch"]
    rf_a, f_a = real["rf_f"][:2]
    for line in CAVEATS:
        rep.p(line.replace("§REJ§", _pc(rej, 1)).replace("§RUIN§", f"{_pc(min(ruin), 1)[:-1]}–{_pc(max(ruin), 1)}")
              .replace("§RF§", _pc(rf_a, 2)).replace("§F§", _pc(f_a, 2)))
    rep.save()
    return S, real


def titular_rows(real: dict) -> list[dict]:
    """Los p de titular de R18 (lista cerrada; los lee R19 desde p_titular.csv): la historia real
    del S&P 500 frente al 1x y frente a la cartera constante de su exposición media (desde 1938 y,
    la LRS 3x original, también desde 1928), para la LRS 3x original, la de precio, la LRS-Kelly
    con varianza EWMA y la de R13, y la LRS 3x original en los últimos 30 y 10 años (frente al 1x
    y sincronización con la referencia ex ante)."""
    rt = real["tests"]
    w28, win, w30, w10 = real["windows"]

    def q(s, b, w):
        return float(rt[(rt.ventana == w) & (rt.estrategia == s) & (rt["frente a"] == b)]["p_unilateral"].iloc[0])

    O, P, KE = "LRS 3x (original)", r13.PX_LRS, "LRS-Kelly (EWMA, señal total)"
    rows = [(f"LRS 3x original frente al 1x, {win}", q(O, "1x", win)),
            (f"LRS 3x original frente a su constante, {win}", q(O, POST, win)),
            (f"LRS 3x original frente a su constante, {w28}", q(O, POST, w28)),
            (f"LRS 3x señal de precio frente al 1x, {win}", q(P, "1x", win)),
            (f"LRS 3x señal de precio frente a su constante, {win}", q(P, POST, win)),
            (f"LRS-Kelly EWMA señal total frente al 1x, {win}", q(KE, "1x", win)),
            (f"LRS-Kelly EWMA señal total frente a su constante, {win}", q(KE, POST, win)),
            (f"LRS-Kelly de R13 (GJR) frente al 1x, {win}", q(REAL_KG, "1x", win)),
            (f"LRS-Kelly de R13 (GJR) frente a su constante, {win}", q(REAL_KG, POST, win)),
            ("LRS 3x original: sincronización, últimos 30 años (referencia ex ante)", q(O, ANTE, w30)),
            ("LRS 3x original: sincronización, últimos 10 años (referencia ex ante)", q(O, ANTE, w10)),
            ("LRS 3x original frente al 1x, últimos 30 años", q(O, "1x", w30)),
            ("LRS 3x original frente al 1x, últimos 10 años", q(O, "1x", w10))]
    return [{"módulo": "R18", "mercado": "S&P 500", "contraste": c, "p": p} for c, p in rows]


def _pp(x: float, d: int = 1, sign: bool = True) -> str:
    """Puntos porcentuales con coma decimal."""
    return (f"{x * 100:+.{d}f}" if sign else f"{x * 100:.{d}f}").replace(".", ",").replace("-", "−")


def _pc(x: float, d: int = 0) -> str:
    return f"{x * 100:.{d}f}%".replace(".", ",")


def _p(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def _n(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def _rng(vals, fmt) -> str:
    """«a–b» (o «a» si coinciden al redondear)."""
    lo, hi = fmt(min(vals)), fmt(max(vals))
    return lo if lo == hi else f"{lo} a {hi}"


SHORT = {"iid (bloque 1 d)": "el iid", "bootstrap 21 d": "el de bloques de un mes",
         "HMM medias iguales": "el HMM de medias iguales", "GARCH simétrico": "el GARCH simétrico",
         "GJR-GARCH": "el GJR", "GJR, media aritmética constante": "el GJR de media aritmética constante"}


def _join(ws) -> str:
    names = [SHORT.get(w, w) for w in ws]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " y " + names[-1]


def summary(S, real, E, P, ft, fp, fp88, Bt, mt, md) -> list[str]:
    rt = real["tests"]
    w28, win, w30, w10 = real["windows"]

    def rr(strat, b, w=win):
        q = rt[(rt.ventana == w) & (rt.estrategia == strat) & (rt["frente a"] == b)].iloc[0]
        return q["Δg"], q["p_unilateral"], q["exposición_referencia"]

    def eff(strat, w, col="Δg frente a constante (sincronización)"):
        return E.loc[(strat, w), col]

    def pw(w, comp, Y, col="potencia"):
        return float(P[(P.mundo == w) & (P["comparación"] == comp) & (P["años"] == Y)][col].iloc[0])

    T3, G3 = REAL_LIKE["LRS 3x (original)"], REAL_PX["LRS 3x (original)"]
    TK, GK, PK = REAL_LIKE["LRS-Kelly"], REAL_KG, REAL_PX["LRS-Kelly"]
    L = "LRS 3x (original)"
    trend = ["bootstrap 252 d", "bootstrap 756 d", "HMM completo"]
    notrend = ["iid (bloque 1 d)", "bootstrap 21 d", "GJR-GARCH", "GARCH simétrico", "HMM medias iguales"]
    tc = "LRS 3x − constante"
    x2 = lambda v: str(round(v, 2)).replace(".", ",")

    # historia
    g1, p1, _ = rr(T3, "1x")
    gc, pc, xc = rr(T3, POST)
    g28, p28, _ = rr(T3, POST, w28)
    gg1, pg1, _ = rr(G3, "1x")
    ggc, pgc, _ = rr(G3, POST)
    gg28, pg28, _ = rr(G3, POST, w28)
    gk1, pk1, _ = rr(TK, "1x")
    gkc, pkc, _ = rr(TK, POST)
    gr1, pr1, _ = rr(GK, "1x")
    grc, prc, _ = rr(GK, POST)
    gq1, pq1, _ = rr(PK, "1x")
    gqc, pqc, _ = rr(PK, POST)
    dn = real["dentro"]
    a30, pa30, xa30 = rr(T3, ANTE, w30)
    e30, pe30, _ = rr(T3, POST, w30)
    a10, pa10, xa10 = rr(T3, ANTE, w10)
    e10, pe10, _ = rr(T3, POST, w10)
    o30, po30, _ = rr(T3, "1x", w30)
    o10, po10, _ = rr(T3, "1x", w10)
    ga30, pga30, _ = rr(G3, ANTE, w30)
    ga10, pga10, _ = rr(G3, ANTE, w10)
    b3 = Bt.loc[L]
    # potencia y p
    Pf = P[P.mundo.isin(FIG_WORLDS) & (P["comparación"] == tc)]
    mde = Pf.groupby("años")["efecto_mín_detectable_80%"].median()
    q30 = P[(P["años"] == 30) & (P["comparación"] == tc)].set_index("mundo")
    pz30, pm30, pz0 = q30["P(0,2 ≤ p ≤ 0,4)"], q30["p_mediano"], q30["P(0,2 ≤ p ≤ 0,4) con Δg = 0"]
    rng_ = lambda ws, Y, comp=tc: _rng([pw(w, comp, Y) for w in ws], _pc)
    pct = E.loc[L]["percentil historia (constante)"]
    pct1 = E.loc[L]["percentil historia (1x)"]
    # descomposición
    d28, d38 = md.loc[(L, w28)], md.loc[(L, win)]
    fr28, fr38 = real["features_w"][w28], real["features_w"][win]
    gap28, gap38 = fr28["μ_sobre − μ_bajo"], fr38["μ_sobre − μ_bajo"]
    q = fr28["fracción_sobre_media"]
    r2d = (gap28 / 252) ** 2 * q * (1 - q) / (fr28["vol_anual"] ** 2 / 252)
    r2s = f"{r2d * 100:.3f}%".replace(".", ",")

    def alone(d):
        cm, cv = d["coste + sólo medias"], d["coste + sólo volatilidad"]
        if cm > 0 and cv > 0:
            return (f"cada canal cubriría por sí solo el coste (coste + medias {_pp(cm)}; coste + volatilidad "
                    f"{_pp(cv)})")
        if cv > 0:
            return (f"el canal de volatilidad cubre por sí solo el coste ({_pp(cv)}) y el de medias no "
                    f"({_pp(cm)})")
        if cm > 0:
            return (f"el canal de medias cubre por sí solo el coste ({_pp(cm)}) y el de volatilidad no "
                    f"({_pp(cv)})")
        return f"ninguno cubre por sí solo el coste ({_pp(cm)} y {_pp(cv)})"

    keyf = ["μ_sobre − μ_bajo", "σ_bajo / σ_sobre", "apalancamiento ρ(R_m, RV_m+1)", "tendencia ρ(R_12m, R_m+1)"]
    fb = fp.loc[["bootstrap 252 d", "bootstrap 756 d"], keyf]
    ks = P[P.mundo.str.startswith("bootstrap") | P.mundo.str.startswith("iid")]["k"]
    kp = P[~(P.mundo.str.startswith("bootstrap") | P.mundo.str.startswith("iid"))]["k"]
    rare = [w for w in notrend + ["GJR, media aritmética constante"] if pct[w] >= 0.95]
    not_rare = [w for w in notrend + ["GJR, media aritmética constante"] if pct[w] < 0.95]
    hk = f"historia {w28}"
    lines = [
        f"**La historia, con la misma señal que los mundos.** La LRS 3x original (la de Gayed y Bilello: "
        f"media de 200 sesiones del índice de rentabilidad total, la señal de los mundos y de los rasgos) "
        f"crece en el S&P 500 de {win.replace('–', ' a ')} "
        f"{_pp(g1)} pp al año más que el 1x (p = {_p(p1)}) y {_pp(gc)} pp más que una cartera con su misma "
        f"exposición media ({x2(xc)}x) mantenida constante (p = {_p(pc)}); desde 1928, {_pp(g28)} pp "
        f"(p = {_p(p28)}). La diferencia entre ambas cifras, {_pp(g1 - gc)} pp, es efecto nivel: más "
        f"apalancamiento medio. Con la señal de precio (una sensibilidad del laboratorio): {_pp(gg1)} frente "
        f"al 1x (p = {_p(pg1)}) y {_pp(ggc)} frente a la constante (p = {_p(pgc)}; desde 1928, {_pp(gg28)} y "
        f"p = {_p(pg28)}). La señal de precio exige al índice total un margen de unos 0,4 años de dividendo "
        f"sobre su media y está dentro el {_pc(dn[G3])} de los días, frente al {_pc(dn[T3])}; la diferencia "
        f"de {_pp(ggc - gc)} pp frente a la constante es de una sola trayectoria y no se contrasta aquí. La "
        f"LRS-Kelly comparable (varianza EWMA, señal total) gana {_pp(gk1)} pp al 1x (p = {_p(pk1)}) y "
        f"{_pp(gkc)} a su exposición constante (p = {_p(pkc)}); la de R13 (GJR, señal total), {_pp(gr1)} "
        f"(p = {_p(pr1)}) y {_pp(grc)} (p = {_p(prc)}); con la señal de precio, {_pp(gq1)} (p = {_p(pq1)}) y "
        f"{_pp(gqc)} (p = {_p(pqc)}).",
        f"**Los últimos 30 y 10 años.** Con la cartera constante fijada ex ante (la exposición media de la "
        f"LRS 3x antes de la ventana: {x2(xa30)}x y {x2(xa10)}x), sincronizar da {_pp(a30)} pp al año en los "
        f"últimos 30 años (p = {_p(pa30)}) y {_pp(a10)} en los últimos 10 (p = {_p(pa10)}); con la señal de "
        f"precio, {_pp(ga30)} (p = {_p(pga30)}) y {_pp(ga10)} (p = {_p(pga10)}). Frente al 1x: {_pp(o30)} "
        f"(p = {_p(po30)}) y {_pp(o10)} (p = {_p(po10)}). Con la exposición media de la propia ventana, el "
        f"estimador de la primera versión, salían {_pp(e30)} (p = {_p(pe30)}) y {_pp(e10)} (p = {_p(pe10)}). "
        f"Esa referencia está sesgada a la baja en ventanas cortas: en los ocho mundos, el sesgo medio es "
        f"de {_rng([b3.loc[w, 'sesgo 10 a'] for w in FIG_WORLDS], _pp)} pp a 10 años y de "
        f"{_rng([b3.loc[w, 'sesgo 30 a'] for w in FIG_WORLDS], _pp)} a 30. La diferencia real entre ambas "
        f"referencias en los últimos 10 años ({_pp(e10 - a10)} pp) es de ese orden.",
        f"**Ablación: de dónde sale la ventaja de sincronizar.** Frente a la exposición constante a largo "
        f"plazo, la LRS 3x pierde {_pp(-eff(L, 'iid (bloque 1 d)'), sign=False)} pp al año en el mundo iid y "
        f"{_pp(-eff(L, 'bootstrap 21 d'), sign=False)} con bloques de un mes, y gana "
        f"{_pp(eff(L, 'bootstrap 252 d'))} con bloques de un año y {_pp(eff(L, 'bootstrap 756 d'))} con "
        f"bloques de tres: la ventaja vive en la dependencia a escalas de entre un mes y varios años. En los "
        f"mundos paramétricos: GJR {_pp(eff(L, 'GJR-GARCH'))}, GARCH sin asimetría "
        f"{_pp(eff(L, 'GARCH simétrico'))} (el efecto apalancamiento vale "
        f"{_pp(eff(L, 'GJR-GARCH') - eff(L, 'GARCH simétrico'), sign=False)} pp), HMM con medias iguales "
        f"{_pp(eff(L, 'HMM medias iguales'))} y HMM completo {_pp(eff(L, 'HMM completo'))} (las medias de "
        f"régimen valen {_pp(eff(L, 'HMM completo') - eff(L, 'HMM medias iguales'), sign=False)} pp). "
        f"Sin tendencia la LRS 3x no gana a la exposición constante; con volatilidad agrupada y asimétrica "
        f"pero media aritmética constante (sensibilidad añadida) queda en "
        f"{_pp(eff(L, 'GJR, media aritmética constante'))} de media y "
        f"{_pp(eff(L, 'GJR, media aritmética constante', 'mediana (constante)'))} de mediana.",
        f"**El sesgo de la referencia ex post.** En la primera versión la cartera constante tenía la "
        f"exposición media de la propia ventana. Esa exposición y la rentabilidad del mercado en la ventana "
        f"tienen una correlación de {_rng([b3.loc[w, 'corr(w̄, mercado) 10 a'] for w in FIG_WORLDS], x2)} "
        f"entre historias de 10 años de los mundos: la referencia llevaba más apalancamiento justo en las "
        f"ventanas buenas. Por eso el Δg medio subía con el horizonte. En el bootstrap de 252 días, el sesgo "
        f"es de {_pp(b3.loc['bootstrap 252 d', 'sesgo 5 a'])}, {_pp(b3.loc['bootstrap 252 d', 'sesgo 10 a'])}, "
        f"{_pp(b3.loc['bootstrap 252 d', 'sesgo 30 a'])} y {_pp(b3.loc['bootstrap 252 d', 'sesgo 98 a'])} pp "
        f"a 5, 10, 30 y 98 años. Con la referencia ex ante, el Δg medio de ese mundo es "
        f"{_pp(b3.loc['bootstrap 252 d', 'Δg ex ante 10 a'])} a 10 años y "
        f"{_pp(b3.loc['bootstrap 252 d', 'Δg ex ante 98 a'])} a 98. La potencia, el efecto mínimo "
        f"detectable y la ventaja verdadera de este informe usan la referencia ex ante.",
        f"**Descomposición.** Historia a historia, el valor de sincronizar de la LRS 3x se explica casi por "
        f"completo (R² = {x2(mt.loc[L, 'R²'])}) por la brecha de rentabilidad del índice con la señal dentro "
        f"y fuera y por el cociente de volatilidades fuera/dentro (la volatilidad anual entra como control). Si la "
        f"señal no separase ni medias ni volatilidades, sincronizar costaría "
        f"{_pp(d28[DEC_COLS[0]])} pp al año a la volatilidad de {w28} ({_pc(fr28['vol_anual'], 1)}) y "
        f"{_pp(d38[DEC_COLS[0]])} a la de {win} ({_pc(fr38['vol_anual'], 1)}): es el arrastre de entrar y "
        f"salir del 3x más los costes. Con los rasgos de {w28}, medidos con la misma señal y en la misma "
        f"ventana que el Δg real, el canal de medias aporta {_pp(d28[DEC_COLS[1]])} y el de volatilidad "
        f"{_pp(d28[DEC_COLS[2]])}; predicho {_pp(d28['predicho'])} frente a un real de "
        f"{_pp(d28[DEC_COLS[6]])}, y {alone(d28)}. Con los de {win}: {_pp(d38[DEC_COLS[1]])} y "
        f"{_pp(d38[DEC_COLS[2]])}; predicho {_pp(d38['predicho'])} frente a {_pp(d38[DEC_COLS[6]])}, y "
        f"{alone(d38)}. El canal de volatilidad es casi igual en las dos ventanas, pero el de medias no: "
        f"la brecha de rentabilidad con la señal dentro y fuera es de {_pp(gap28, sign=False)} pp al año "
        f"desde 1928 y de sólo {_pp(gap38, sign=False)} desde 1938. Más de la mitad de la brecha de "
        f"{w28} sale de 1928–1938, es decir, de la Gran Depresión. Sin ella, la LRS 3x es sobre todo "
        f"una apuesta por la volatilidad previsible, la conclusión del artículo 1. La brecha de {w28} es un "
        f"R² diario de apenas {r2s}, compatible con que el artículo 1 no encontrara rentabilidad previsible. "
        f"Queda en el percentil {_pc(fp.loc['iid (bloque 1 d)', 'μ_sobre − μ_bajo'])} del mundo iid y en el "
        f"{_pc(fp.loc['bootstrap 21 d', 'μ_sobre − μ_bajo'])} del de bloques de un mes. La de {win} queda "
        f"en el {_pc(fp88.loc['iid (bloque 1 d)', 'μ_sobre − μ_bajo'])} y el "
        f"{_pc(fp88.loc['bootstrap 21 d', 'μ_sobre − μ_bajo'])} (historias de 88 años).",
        f"**Potencia.** En los mundos donde la ventaja existe (bootstrap de 252 y 756 días y HMM completo, "
        f"{_pp(min(eff(L, w) for w in trend))} a {_pp(max(eff(L, w) for w in trend))} pp), el t Newey-West "
        f"la detecta con una probabilidad del {rng_(trend, 10)} con 10 años, {rng_(trend, 30)} con 30 y "
        f"{rng_(trend, 98)} con 98. Para detectarla 8 de cada 10 veces tendría que ser de unos "
        f"{_pp(mde[10], 0, False)} pp con 10 años, {_pp(mde[30], 0, False)} con 30 y {_pp(mde[98], 0, False)} "
        f"con 98 (mediana de los ocho mundos). Frente al 1x, con 98 años, la potencia sube al "
        f"{rng_(trend, 98, 'LRS 3x − 1x')}. La LRS-Kelly frente a su exposición constante es casi "
        f"indetectable: {_pc(pw('bootstrap 252 d', 'LRS-Kelly − constante', 98))} y "
        f"{_pc(pw('bootstrap 756 d', 'LRS-Kelly − constante', 98))} con 98 años en los bootstrap de 252 y "
        f"756 días.",
        f"**¿Qué dice un p ≈ 0,3 con 30 años?** Casi nada sobre si el efecto existe. Si Δg = 0 exactamente y "
        f"el test está bien calibrado, el p es uniforme y cae entre 0,2 y 0,4 el 20% de las veces. En los "
        f"mundos con ventaja ocurre el {_rng([pz30[w] for w in trend], _pc)}: el factor de Bayes a favor "
        f"del efecto frente a un efecto nulo exacto es de sólo "
        f"{_rng([pz30[w] / 0.2 for w in trend], x2)}. Si el nulo exacto es el mismo mundo desplazado a "
        f"Δg = 0, con su ruido y su error Newey-West, el factor queda en "
        f"{_rng([pz30[w] / pz0[w] for w in trend], x2)}. Observar p ≈ 0,3 sólo pesa contra los modelos en "
        f"que sincronizar es costoso: en los mundos sin tendencia (Δg medio a 30 años de "
        f"{_rng([q30.loc[w, 'Δg_medio'] for w in notrend], _pp)} pp) ocurre en el "
        f"{_rng([pz30[w] for w in notrend], _pc)} de las historias. El p mediano es "
        f"{_rng([pm30[w] for w in trend], _p)} en los mundos con ventaja y "
        f"{_rng([pm30[w] for w in notrend], _p)} en los que no la tienen. Los últimos 30 años reales dan "
        f"p = {_p(pa30)} con la referencia ex ante: no distinguen «no hay efecto» de «hay un efecto de "
        f"{_rng([eff(L, w) for w in trend], _pp)} pp que 30 años no bastan para detectar».",
        f"**Contra qué hipótesis nula.** «Δg ≤ 0» es un nulo generoso, porque sin tendencia la sincronización "
        f"de la LRS 3x tiene un valor esperado negativo. La historia real con la señal comparable "
        f"({_pp(gc)} pp desde 1938) queda en el percentil {_pc(pct['iid (bloque 1 d)'], 1)} del mundo iid, "
        f"{_pc(pct['bootstrap 21 d'], 1)} del de bloques de un mes, {_pc(pct['HMM medias iguales'], 1)} del HMM "
        f"de medias iguales, {_pc(pct['GARCH simétrico'], 1)} del GARCH simétrico, {_pc(pct['GJR-GARCH'], 1)} del "
        f"GJR y {_pc(pct['GJR, media aritmética constante'], 1)} del GJR de media aritmética constante. En los "
        f"mundos con tendencia queda en el {_rng([pct[w] for w in trend], _pc)}. Frente al 1x ({_pp(g1)} pp), "
        f"en el {_rng([pct1[w] for w in notrend], _pc)} de los mundos sin tendencia y el "
        f"{_rng([pct1[w] for w in trend], _pc)} de los que la tienen. "
        + (f"Frente a la constante, la historia sería rara (percentil ≥ 95%) en {_join(rare)}" if rare else
           "Frente a la constante, la historia no sería rara (percentil < 95%) en ningún mundo sin tendencia")
        + (f"; no lo es en {_join(not_rare)}. " if rare and not_rare else ". ")
        + "Como prueba contra algunos modelos concretos sin tendencia, la historia informa algo más de lo "
        "que dice su t frente a cero; como prueba de «la LRS gana», no llega. La primera versión comparaba "
        "con la cifra de la señal de precio y exageraba estos percentiles.",
        f"**Mundos frente a la historia.** Los bootstrap de 252 y 756 días reproducen los rasgos que "
        f"importan (la historia {w28} cae entre sus percentiles {fb.values.min() * 100:.0f} y "
        f"{fb.values.max() * 100:.0f}); el iid y el de un mes no generan ni su brecha de rentabilidad ni su "
        f"cociente de volatilidades. El GJR acierta el cociente de volatilidades "
        f"({x2(ft.loc['GJR-GARCH', 'σ_bajo / σ_sobre'])} frente a {x2(ft.loc[hk, 'σ_bajo / σ_sobre'])}) pero, "
        f"con media logarítmica constante, invierte el signo de la brecha "
        f"({_pp(ft.loc['GJR-GARCH', 'μ_sobre − μ_bajo'])} pp), tiene una volatilidad típica menor "
        f"(mediana {_pc(ft.loc['GJR-GARCH', 'vol_anual'], 1)} frente a {_pc(ft.loc[hk, 'vol_anual'], 1)}) y "
        f"más del doble de días de −10%; el HMM no tiene caídas diarias de −10% y subestima el cociente de "
        f"volatilidades (la historia queda en su percentil "
        f"{fp.loc['HMM completo', 'σ_bajo / σ_sobre'] * 100:.0f}). El error Newey-West está bien calibrado en "
        f"los bootstrap (k = {x2(ks.min())}–{x2(ks.max())}) y es algo optimista en los paramétricos (hasta "
        f"k = {x2(kp.max())}); la tabla da también la potencia calibrada.",
    ]
    return lines


CAVEATS = [
    "Los mundos se construyen con la misma historia 1928–2026 en la que se eligieron la LRS (media de "
    "200 sesiones, 3x) y el medio Kelly (R10–R12): no son evidencia independiente. Los bootstrap heredan "
    "sus episodios concretos (1929–32, 1973–74, 2008) y los modelos se estiman con ella.",
    "Referencia de exposición constante. En los mundos es la exposición media a largo plazo de la "
    "estrategia, estimada con las demás historias del mismo mundo o del mismo sorteo de parámetros. Es "
    "ex ante para cada historia, pero es un oráculo que un investigador real no tiene. Su versión "
    "operable en la historia es la media previa a la ventana, y sólo existe para los últimos 30 y 10 "
    "años. Para las ventanas históricas completas se usa la media de la propia ventana, con el mismo "
    "estimador ex post en los mundos; su sesgo a 88–98 años es de décimas (tabla de la sección 3). La "
    "cartera se reequilibra a diario sin costes. La referencia ex ante sustituyó a la ex post después de "
    "la revisión de la primera ejecución. Ninguna de las dos se ajustó para mejorar los resultados, pero "
    "el cambio se hizo después de verlos.",
    "En los mundos el T-bill (§RF§) y la financiación (§F§) son constantes; en la historia varían y "
    "están correlacionados con los regímenes. Las cifras históricas comparables usan, como los mundos y "
    "como Gayed y Bilello, la señal sobre el índice de rentabilidad total. La LRS con la señal de precio "
    "solo se da como sensibilidad: los mundos no tienen dividendos y no pueden reproducirla. La LRS-Kelly "
    "de los mundos prevé la varianza con EWMA, no con GJR.",
    "Un día en que el 3x pierde el 100% con la LRS dentro cuenta como −99,9% (log −6,9), como en el SPA "
    "de R13; con −100% la riqueza sería cero y Δg, −∞. Afecta al §RUIN§ de las historias de los mundos "
    "GARCH, cuyas medias quedan por eso por debajo de sus medianas: se dan ambas.",
    "La distribución de parámetros del GJR/GARCH se truncó a la región estacionaria (p_eff < 0,9999, se "
    "repitió el sorteo en el §REJ§ de los intentos) y el bloque de 1.260 días del remuestreo se eligió "
    "tras comprobar que con 252 días la persistencia salía sesgada a la baja (la comprobación miró sólo "
    "los parámetros, no las estrategias).",
    "La definición de «medias iguales» del HMM (misma media aritmética en los dos regímenes) se fijó antes "
    "de simular. El mundo «GJR, media aritmética constante» NO estaba en el plan: se añadió después de la "
    "primera ejecución, al ver que el GJR de r13 tiene por construcción una brecha de medias negativa. Se "
    "presenta como sensibilidad y queda fuera de las figuras.",
    "El test es un t con errores Newey-West de 63 retardos, unilateral al 5%; no se repitió con bootstrap "
    "por historia (serían 9 × 1.000 × 4 × 10 remuestreos). Con 1.000 historias, el error de Monte Carlo "
    "de una potencia es de ±1,6 puntos como mucho. Las historias de 10 y 30 años son los primeros años "
    "de las de 98 (anidadas, no independientes).",
    "El código base se corrigió durante el trabajo (ver BITACORA.md): cierres de ^GSPC, sobrecoste del "
    "swap calibrado solo con la vida del ETF real, señal de la LRS original sobre la rentabilidad total sin "
    "mirar el futuro y prima de Kelly compartida. Las cifras de este informe usan ese código.",
    "El GJR «corregido» es el de r13.gen_paths (residuos con las partes negativa y positiva reescaladas). "
    "Las variantes de este módulo (GARCH sin asimetría, media aritmética constante, HMM de medias iguales, "
    "estrategias) copian el código de r13 y tests/test_r18_worlds.py comprueba que, sin la modificación, "
    "reproducen exactamente r13.gen_paths, r13.mc_strategies y r13.mc_summary: si r13 cambia, esos tests "
    "fallan.",
    "La regresión de descomposición es descriptiva (asociación entre rasgos y resultado a través de "
    "historias de mundos distintos), no un efecto causal, y su R² alto refleja en parte que la brecha de "
    "rentabilidad y el Δg se miden en la misma historia. Se ajusta con historias de 98 años y se aplica "
    "también a la ventana de 88 años (1938–2026). La brecha de medias histórica depende mucho de que la "
    "ventana incluya la Gran Depresión.",
    "El factor de Bayes de «p ≈ 0,3» compara la probabilidad de ese intervalo de p en dos mundos concretos: "
    "no es una probabilidad posterior de que el efecto exista, que dependería de la probabilidad previa "
    "de cada mundo.",
]


if __name__ == "__main__":
    run()
