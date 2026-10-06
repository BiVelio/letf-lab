"""R19 — Censo de pruebas: cuántas variantes se han probado y cuánto descuenta eso (I6).

Pregunta. El laboratorio ha probado muchas variantes de la misma idea (R6, R10–R13,
R15, R17, R18). La LRS 3x y sus parientes ganan al 1x en el S&P 500, pero la cifra
que se publica es la de una regla elegida entre muchas: ¿cuánto sobrevive cuando se
descuenta la búsqueda?

Censo. Se reconstruyen en el S&P 500, con la misma ventana común, las rentabilidades
diarias de todas las variantes probadas en el laboratorio, con el protocolo común de
R13: señal al cierre de t, orden al cierre de t+1 (retraso 2), 3x sintético a f + s
(real desde UPRO), T-bills, 5 pb por unidad de rotación del objetivo:
- R6: filtro de media móvil (50–250 sesiones) × volatilidad objetivo (20–50%), la
  rejilla completa (29) más la variante a priori con banda ±2%, sobre el índice de
  rentabilidad total. R6 las probó con ejecución al cierre (MOC) y el motor de
  backtest.run; aquí van con el protocolo común.
- R10/R11: las 15 reglas de entrada y salida (caída, volatilidad, media de 200).
  R10 las ejecutaba en el mismo cierre; aquí, en el siguiente.
- R12: ½ Kelly revisado cada 21 sesiones con cada modelo de varianza (EWMA, ARCH,
  GARCH, GJR, HAR, Ridge, bosque aleatorio, gradient boosting, red neuronal) y el GJR
  con la media prevista por gradient boosting. Los modelos de ML se reentrenan aquí
  con el mismo código de R12 (walk_forward), limitados a 4 hilos.
- R13: la familia completa, la rejilla de 30 (media × banda) y la variante
  «mismo cierre» de cada regla, con la señal original (rentabilidad total) y con
  la de precio: el laboratorio probó y publicó las dos.
- R15: retrasos 1, 2, 3 y 5 con señal de precio y de rentabilidad total, 5 días de
  revisión semanal y 21 desfases de revisión cada 21 sesiones (con la señal
  original), para la LRS 3x y la LRS-Kelly; los 21 desfases del ½ Kelly GJR de R12.
- R17: Faber 10 meses (1x y 3x), TSMOM 12 meses 3x, Moreira–Muir con c en tiempo
  real y ½ Kelly ex ante anual.
- R18: LRS-Kelly con EWMA y señal de rentabilidad total (la LRS 3x de R18 es la
  original de R13).
Las señales de rentabilidad total (R6, R10, R13, R15, R18) usan en el S&P 500 el
índice oficial ^SP500TR desde 1988 y, antes, ^GSPC con el D/P de Shiller del mes
anterior (r13.sp500_signal_returns).
Fuera del censo, y dicho en el informe: las versiones de datos de R14 (mismas
reglas con otros datos), las referencias de L constante de R16 y R17 (se fijan con
la exposición o la volatilidad de toda la muestra), Moreira–Muir con c de muestra
completa (mira el futuro) y todo lo hecho en otros mercados.
Las series idénticas en la ventana común (la misma regla construida en dos módulos,
p. ej. la LRS 3x original en R13, R15 y R18) cuentan una vez.

Medidas, fijadas antes de mirar resultados:
- Estadístico: ratio de información (IR) diario de las log-rentabilidades en exceso
  del 1x (a_t = ln(1+r_t) − ln(1+u_t)); para las alternativas, en exceso de la LRS 3x
  original. Se anualiza ×√252 sólo para mostrarlo.
- Número efectivo de pruebas con los autovalores de la matriz de correlación de las
  a_t: principal Li y Ji (2005); además cociente de participación, rango efectivo por
  entropía (Roy y Vetterli 2007), Nyholt (2004), Galwey (2009) y autovalores para el
  95% y el 99% de la varianza.
- Deflated Sharpe Ratio (Bailey y López de Prado 2014): PSR frente al máximo esperado
  de N IR sin habilidad, SR₀ = √V·[(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))], con V (a) la
  varianza de los IR observada entre las pruebas y (b) la que daría el puro azar,
  1/T. Se muestra en función de N (1 a 1.000). Es descriptivo: compara con el máximo
  ESPERADO más 1,645 errores típicos de un solo IR, no con el cuantil 95% del máximo,
  así que «DSR ≥ 0,95» no es un contraste al 5% (su error familiar es mucho menor).
- Minimum Track Record Length (Bailey y López de Prado 2012) al 95% frente a IR* = 0
  y frente a SR₀(N_eff), que depende de la propia longitud n de la historia:
  SR₀(n) = z_N·√(V_disp + 1/n), con V_disp = 0 (azar) o máx(V − 1/T, 0) (observada).
- p-valores unilaterales (H0: Δg ≤ 0) con el error estándar del bootstrap
  estacionario (bloque medio 63 sesiones, 2.000 remuestreos, los mismos para todas
  las pruebas), con Holm y Benjamini–Yekutieli (BHY), Šidák a N_eff y el step-down de
  Romano y Wolf (2005) con los mismos remuestreos; SPA de Hansen sobre todo el censo;
  además Holm y BHY sobre los p-valores de titular de R16–R18, leídos de sus
  p_titular.csv (con la huella del código que los generó, comprobada aquí).
- La LRS 3x original no sale del censo: la fijaron Gayed y Bilello (2016), con la
  media sobre el índice de rentabilidad total (su nota 15), y es la hipótesis
  principal. Se contrasta sola (y en su único tramo limpio, desde 2016); el DSR y
  las correcciones por la búsqueda del censo se aplican a la MEJOR del censo y a la
  LRS 3x con la señal de precio, una variante del laboratorio que es un miembro
  más del censo.
- Potencia: años de historia para detectar el IR observado con un 80% de
  probabilidad al 5% (unilateral), ((z₀,₉₅ + z₀,₈₀)/IR)²·(1 − γ₃·IR + (γ₄ − 1)/4·IR²);
  con T = MinTRL la potencia es del 50% por construcción.
- Probability of Backtest Overfitting por CSCV (Bailey, Borwein, López de Prado y
  Zhu 2017) con 16 bloques contiguos (12.870 particiones), IR y Δg.
- Correlación de rangos (Spearman y Kendall) entre 1939–1991 y 1992–2026, frente a
  una nula en la que la segunda época no tiene diferencias reales entre pruebas.
Cambios tras la primera ejecución (por defectos del método, no por los resultados):
el p empírico del bootstrap no baja de 1/2.001, lo que impedía a Holm rechazar nada
con casi doscientas pruebas, y se sustituyó por el normal con el error estándar bootstrap; el
intervalo percentil del Spearman estaba sesgado a la baja (el remuestreo añade ruido
a las dos épocas) y se sustituyó por la nula; la regresión de degradación de CSCV
(pendiente ≈ −1 por construcción con mitades complementarias) se quitó.
Cambios tras la revisión externa (también de método): el DSR deja de leerse como
contraste y se añaden, como contraste al 5% con pruebas múltiples, Šidák a N_eff y
Romano–Wolf; el DSR ya no se aplica como veredicto a la LRS 3x preespecificada, que se
contrasta sola y desde 2016; el MinTRL frente a SR₀ se resuelve con SR₀(n) (antes se
trataba el umbral como fijo); SPA y Romano–Wolf también sin las reglas de mismo
cierre, como sensibilidad. Tras la segunda revisión: la LRS 3x original es la de
Gayed y Bilello (señal de rentabilidad total); la de precio entra en el censo con su
familia; el titular «descontada la búsqueda» se da solo con reglas operables; y los
p de titular de R16–R18 se leen de sus ficheros en lugar de transcribirse.
Nada de esto elige parámetros de ninguna regla: todas las señales son las de sus
módulos (sólo pasado). La matriz de correlación, V y los bloques de CSCV usan la
muestra completa, como corresponde a un diagnóstico ex post del proceso de búsqueda.
"""

from __future__ import annotations

import itertools
import pickle
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats as sst
from scipy.optimize import brentq
from scipy.stats import norm

from letf_lab import kelly, metrics, provenance, stats
from letf_lab.report import FIG_W, ROOT as REPORTS, Report, comma_axis, legend_band, num_es
from letf_lab.research import r10_predictability as r10
from letf_lab.research import r12_long_ml as r12
from letf_lab.research import r13_lrs as r13

SEED = 19
LAG = r13.EXEC_LAG            # señal al cierre de t, orden al cierre de t+1
B_BOOT = 2000                 # remuestreos del bootstrap de p-valores
B_SPA = 2000
B_RANK = 1000                 # remuestreos para el intervalo de la correlación de rangos
BLOCK = 63                    # bloque medio del bootstrap estacionario (sesiones)
S_CSCV = 16                   # bloques de CSCV → C(16, 8) = 12.870 particiones
SPLIT = pd.Timestamp("1992-01-01")
ALPHA = 0.05
EULER = 0.5772156649015329
N_GRID = np.unique(np.round(np.logspace(0, 3, 121)).astype(int))
ML_THREADS = 4
TOP_Q = 0.2                   # quintil superior para la persistencia entre épocas

BENCH = "1x comprar y mantener"
LRS = "LRS 3x (original)"
PX_LRS = r13.PX_LRS          # la LRS 3x con la señal de precio: un miembro más del censo
FOCAL = (LRS, PX_LRS, "LRS-Kelly", "½ Kelly GJR (sin tendencia)", "LRS rotación 3x/1x",
         "3x comprar y mantener")
NEFF_MAIN = "Li y Ji (2005)"
ROT = "LRS rotación 3x/1x"
FAM_PX = "R13: señal de precio"
TITULAR = ("r16_mechanism", "r17_benchmarks", "r18_worlds")   # módulos cuyos p de titular se leen
N_R13_SPA = 9                 # comparaciones del SPA principal de R13 (3 mercados × 3 referencias)
GB_END = pd.Timestamp("2016-01-01")   # la muestra de Gayed y Bilello (2016) acaba en 2015


def is_same_close(name: str) -> bool:
    """Reglas ejecutadas en el mismo cierre de la señal (no operables antes de ~1990)."""
    return "mismo cierre" in name or re.search(r"retraso 1(\D|$)", name) is not None

# grupos para colorear (tres como máximo en un diagrama de dispersión) y familias
GROUP_COLOR = {"LRS 3x: parámetros y ejecución": "#2a78d6",
               "Kelly y volatilidad": "#eb6834",
               "Otras reglas y referencias": "#1baf7a"}
GROUP_MARKER = {"LRS 3x: parámetros y ejecución": "o", "Kelly y volatilidad": "s",
                "Otras reglas y referencias": "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
LINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"]


# ───────────────────────── matemáticas (probadas en tests) ─────────────────────────

def sr_moments(X: np.ndarray) -> dict:
    """IR diario (media / desviación típica muestral), asimetría y curtosis (no en
    exceso: la normal da 3) de cada columna de X."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] == 1:
        X = X.T
    m = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    z = (X - m) / X.std(axis=0, ddof=0)
    return {"sr": m / sd, "skew": (z ** 3).mean(axis=0), "kurt": (z ** 4).mean(axis=0),
            "n": X.shape[0]}


def psr(sr, sr_star, n_obs, skew, kurt):
    """Probabilistic Sharpe Ratio (Bailey y López de Prado 2012): P(SR verdadero > SR*)
    con el SR estimado en n_obs periodos y la corrección por asimetría y curtosis."""
    sr, sr_star = np.asarray(sr, float), np.asarray(sr_star, float)
    den = np.sqrt(np.maximum(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return norm.cdf((sr - sr_star) * np.sqrt(n_obs - 1) / den)


def expected_max_sr(var_trials, n_trials):
    """Máximo esperado de N SR estimados sin habilidad (Bailey y López de Prado 2014):
    √V·[(1−γ)Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))]; 0 si N ≤ 1."""
    n = np.asarray(n_trials, float)
    safe = np.maximum(n, 1.0 + 1e-9)
    z = (1 - EULER) * norm.ppf(1 - 1 / safe) + EULER * norm.ppf(1 - 1 / (safe * np.e))
    return np.where(n <= 1, 0.0, np.sqrt(var_trials) * z)


def deflated_sr(sr, n_obs, skew, kurt, var_trials, n_trials):
    """Deflated Sharpe Ratio: el PSR frente a SR₀ = máximo esperado bajo N pruebas."""
    return psr(sr, expected_max_sr(var_trials, n_trials), n_obs, skew, kurt)


def power_trl(sr, skew, kurt, alpha: float = ALPHA, power: float = 0.80):
    """Periodos de historia para que un contraste unilateral al nivel alpha detecte el SR
    observado con probabilidad `power` (con la varianza del SR de Mertens, como el PSR):
    1 + (1 − γ₃·SR + (γ₄ − 1)/4·SR²)·((z_{1−α} + z_power)/SR)². Infinito si SR ≤ 0.
    Con power = 0,5 es el MinTRL frente a SR* = 0."""
    sr = np.asarray(sr, float)
    z = norm.ppf(1 - alpha) + norm.ppf(power)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1 + (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) * (z / sr) ** 2
    return np.where(sr > 0, out, np.inf)


def min_trl(sr, sr_star, skew, kurt, alpha: float = ALPHA):
    """Minimum Track Record Length (en periodos): los que hacen falta para que el
    PSR frente a SR* alcance 1 − alpha con el SR, asimetría y curtosis observados.
    Infinito si SR ≤ SR*. SR* es un umbral FIJO: no sirve para SR₀, que depende de n."""
    sr, sr_star = np.asarray(sr, float), np.asarray(sr_star, float)
    z = norm.ppf(1 - alpha)
    gap = sr - sr_star
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1 + (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) * (z / gap) ** 2
    return np.where(gap > 0, out, np.inf)


def z_max(n_trials):
    """Máximo esperado de N normales estándar independientes (fórmula de Bailey y López
    de Prado): SR₀ = z_max(N)·√V."""
    return expected_max_sr(1.0, n_trials)


def min_trl_deflated(sr: float, skew: float, kurt: float, z_n: float, v_disp: float = 0.0,
                     alpha: float = ALPHA) -> float:
    """MinTRL (en periodos) frente al máximo esperado sin habilidad SR₀(n), que depende de
    la propia longitud n de la historia: SR₀(n) = z_n·√(v_disp + 1/n). v_disp es la
    dispersión entre pruebas que no es ruido de muestreo (no encoge con n); con v_disp = 0
    (dispersión del puro azar, V = 1/n) el umbral tiende a 0. Resuelve
    (SR − SR₀(n))·√(n − 1)/√(1 − γ₃SR + (γ₄−1)/4·SR²) = Φ⁻¹(1 − alpha). Infinito si SR no
    supera el límite z_n·√v_disp."""
    zq = norm.ppf(1 - alpha)
    den = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    if z_n <= 0:                                  # N = 1: SR₀ = 0
        return float(min_trl(sr, 0.0, skew, kurt, alpha))
    if sr <= z_n * np.sqrt(v_disp):
        return np.inf

    def g(n):
        return (sr - z_n * np.sqrt(v_disp + 1 / n)) * np.sqrt(n - 1) / den - zq

    # por debajo de n0 el primer factor es negativo; desde n0 la función es creciente
    n0 = max(1 / ((sr / z_n) ** 2 - v_disp), 1 + 1e-9)
    hi = 2 * n0 + 2
    while g(hi) < 0:
        hi *= 2
    return float(brentq(g, n0, hi, xtol=1e-6, rtol=1e-12))


def sidak(p, n_tests):
    """p ajustado de Šidák para n_tests contrastes independientes: 1 − (1 − p)^n."""
    return 1 - (1 - np.asarray(p, float)) ** n_tests


def sidak_n_star(p, alpha: float = ALPHA):
    """Número de contrastes independientes a partir del cual el p de Šidák supera alpha."""
    p = np.asarray(p, float)
    with np.errstate(divide="ignore"):
        return np.log(1 - alpha) / np.log1p(-p)


def dsr_rule_size(n_trials, level: float = 0.95):
    """Tamaño de la regla «DSR del máximo ≥ level» bajo la nula global con N pruebas
    independientes y SR estimados normales: P(máx ≥ SR₀ + Φ⁻¹(level)·σ) =
    1 − Φ(z_max(N) + Φ⁻¹(level))^N. Con N = 35 y level 0,95 es ≈ 0,003, no 0,05."""
    n = np.asarray(n_trials, float)
    return 1 - norm.cdf(z_max(n) + norm.ppf(level)) ** n


def max_quantile(n_trials, q: float = 0.95):
    """Cuantil q del máximo de N normales estándar independientes: Φ⁻¹(q^(1/N)). Es el
    umbral de un contraste del máximo con error familiar 1 − q."""
    return norm.ppf(q ** (1 / np.asarray(n_trials, float)))


def romano_wolf(t_obs, t_boot) -> np.ndarray:
    """p ajustados del step-down de Romano y Wolf (2005) para H0_k: media_k ≤ 0 (una cola).

    t_obs: K estadísticos estudentizados observados; t_boot: B × K estadísticos bootstrap
    centrados en la nula (mismos remuestreos para todas las pruebas, lo que conserva su
    dependencia). Con las pruebas ordenadas de mayor a menor t, p_(j) = máx_{i ≤ j}
    (1 + #{b: máx_{k ∈ S_i} t*_bk ≥ t_(i)})/(1 + B), con S_i = {(i), …, (K)}. Controla el
    error familiar en sentido fuerte."""
    t_obs = np.asarray(t_obs, float)
    tb = np.asarray(t_boot, float)
    B, K = tb.shape
    order = np.argsort(-t_obs, kind="stable")
    tail = np.maximum.accumulate(tb[:, order][:, ::-1], axis=1)[:, ::-1]   # máx sobre S_j
    p = (1 + (tail >= t_obs[order]).sum(axis=0)) / (1 + B)
    p = np.minimum(np.maximum.accumulate(p), 1.0)
    out = np.empty(K)
    out[order] = p
    return out


def n_eff(eigvals) -> dict:
    """Número efectivo de pruebas independientes con los autovalores λ de una matriz
    de correlación M × M (los negativos por redondeo se llevan a 0)."""
    lam = np.sort(np.clip(np.asarray(eigvals, float), 0, None))[::-1]
    M = len(lam)
    lam = lam * M / lam.sum()                     # traza = M
    p = lam / M
    nz = p[p > 0]
    share = np.cumsum(lam) / M
    # Li y Ji: I(λ ≥ 1) + (λ − ⌊λ⌋); la tolerancia evita que un λ = 4 calculado como
    # 3,9999999999 cuente casi 2 en lugar de 1
    fl = np.floor(lam + 1e-9)
    li_ji = np.sum((lam >= 1 - 1e-9) + np.clip(lam - fl, 0, None))
    return {
        "Li y Ji (2005)": float(li_ji),
        "Nyholt (2004)": float(1 + (M - 1) * (1 - np.var(lam, ddof=1) / M)),
        "Galwey (2009)": float(np.sum(np.sqrt(lam)) ** 2 / M),
        "cociente de participación": float(M ** 2 / np.sum(lam ** 2)),
        "rango efectivo (entropía)": float(np.exp(-np.sum(nz * np.log(nz)))),
        "autovalores para el 95%": int(np.searchsorted(share, 0.95 - 1e-12) + 1),
        "autovalores para el 99%": int(np.searchsorted(share, 0.99 - 1e-12) + 1),
    }


def bhy(p) -> np.ndarray:
    """Benjamini–Yekutieli (2001): q de BH multiplicado por c(m) = Σ 1/i (control del
    FDR con cualquier dependencia), acotado en 1."""
    p = np.asarray(p, float)
    c = np.sum(1.0 / np.arange(1, len(p) + 1))
    return np.minimum(stats.benjamini_hochberg(p) * c, 1.0)


def stationary_blocks(n: int, mean_block: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Un remuestreo estacionario (Politis y Romano 1994) de longitud n como bloques:
    comienzos uniformes y longitudes geométricas de media mean_block, circulares, con
    el último recortado para sumar exactamente n."""
    m = int(3 * n / mean_block) + 16
    lengths = rng.geometric(1.0 / mean_block, m)
    while lengths.sum() < n:
        lengths = np.r_[lengths, rng.geometric(1.0 / mean_block, m)]
    cut = int(np.searchsorted(np.cumsum(lengths), n))
    lengths = lengths[:cut + 1].copy()
    lengths[-1] -= lengths.sum() - n
    starts = rng.integers(0, n, len(lengths))
    return starts, lengths


def block_indices(starts: np.ndarray, lengths: np.ndarray, n: int) -> np.ndarray:
    """Índices explícitos del remuestreo (para comprobar block_sums)."""
    return np.concatenate([(s + np.arange(l)) % n for s, l in zip(starts, lengths)])


def block_sums(C: np.ndarray, starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Suma de cada columna sobre el remuestreo, con la suma acumulada C ((n+1) × K,
    C[0] = 0): cada bloque circular cuesta dos lecturas."""
    n = C.shape[0] - 1
    end = starts + lengths
    wrap = end > n
    s = C[np.minimum(end, n)] - C[starts] + C[np.where(wrap, end - n, 0)]
    return s.sum(axis=0)


def cumsum0(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)])


def boot_means(X: np.ndarray, B: int, mean_block: float, rng) -> np.ndarray:
    """B × K medias bootstrap (mismos remuestreos para todas las columnas)."""
    n = len(X)
    C = cumsum0(X)
    return np.array([block_sums(C, *stationary_blocks(n, mean_block, rng)) / n for _ in range(B)])


def boot_test(X: np.ndarray, B: int, mean_block: float, rng) -> dict:
    """Contraste unilateral de H0: media ≤ 0 en cada columna con el bootstrap estacionario
    (mismos remuestreos para todas). p = 1 − Φ(m̄ / EE*), continuo (con B remuestreos el p
    empírico no baja de 1/(B+1), lo que impediría a Holm rechazar nada con cientos de
    pruebas); p empírico recentrado (1 + #{m* − m̄ ≥ m̄}) / (1 + B); t = m̄/EE* y los
    B × K estadísticos centrados t* = (m* − m̄)/EE* para Romano–Wolf."""
    m = np.asarray(X, float).mean(axis=0)
    draws = boot_means(X, B, mean_block, rng)
    se = draws.std(axis=0, ddof=1)
    safe = np.where(se > 0, se, np.inf)
    t = m / safe
    return {"p": norm.sf(t), "p_emp": (1 + np.sum(draws - m >= m, axis=0)) / (1 + B),
            "t": t, "t_boot": (draws - m) / safe}


def boot_pvalues(X: np.ndarray, B: int, mean_block: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """p normal con el EE del bootstrap y p empírico (ver boot_test)."""
    r = boot_test(X, B, mean_block, rng)
    return r["p"], r["p_emp"]


def boot_ir(X: np.ndarray, B: int, mean_block: float, rng) -> np.ndarray:
    """B × K IR bootstrap (media / desviación típica sobre el mismo remuestreo)."""
    n = len(X)
    C1, C2 = cumsum0(X), cumsum0(np.asarray(X, float) ** 2)
    out = np.empty((B, X.shape[1]))
    for b in range(B):
        st_, ln = stationary_blocks(n, mean_block, rng)
        m1 = block_sums(C1, st_, ln) / n
        m2 = block_sums(C2, st_, ln) / n
        out[b] = m1 / np.sqrt(np.maximum(m2 - m1 ** 2, 1e-300) * n / (n - 1))
    return out


def cscv(X: np.ndarray, S: int = S_CSCV, metric: str = "ir") -> dict:
    """Combinatorially Symmetric Cross-Validation (Bailey et al. 2017).

    X: T × N rendimientos de las N pruebas. Se parte en S bloques contiguos; en cada
    una de las C(S, S/2) particiones se elige la prueba con mejor métrica dentro de
    muestra (IS) y se mira su rango relativo ω fuera (OOS). λ = ln(ω/(1−ω));
    PBO = P(λ ≤ 0). metric: "ir" (media/desviación) o "mean" (media)."""
    X = np.asarray(X, float)
    T, N = X.shape
    blocks = np.array_split(np.arange(T), S)
    s1 = np.stack([X[b].sum(axis=0) for b in blocks])
    s2 = np.stack([(X[b] ** 2).sum(axis=0) for b in blocks])
    nb = np.array([len(b) for b in blocks], float)
    combos = np.array(list(itertools.combinations(range(S), S // 2)))
    Min = np.zeros((len(combos), S))
    Min[np.arange(len(combos))[:, None], combos] = 1.0

    def perf(mask):
        n = mask @ nb
        m1 = mask @ s1 / n[:, None]
        if metric == "mean":
            return m1
        m2 = mask @ s2 / n[:, None]
        return m1 / np.sqrt(np.maximum(m2 - m1 ** 2, 1e-300) * (n / (n - 1))[:, None])

    IS, OOS = perf(Min), perf(1 - Min)
    best = IS.argmax(axis=1)
    rows = np.arange(len(combos))
    is_b, oos_b = IS[rows, best], OOS[rows, best]
    below = (OOS < oos_b[:, None]).sum(axis=1)
    ties = (OOS == oos_b[:, None]).sum(axis=1)
    rank = below + (ties + 1) / 2                 # rango medio, 1..N
    w = rank / (N + 1)
    logit = np.log(w / (1 - w))
    slope, intercept = np.polyfit(is_b, oos_b, 1)
    med_oos = np.median(OOS, axis=1)
    return {"logits": logit, "pbo": float(np.mean(logit <= 0)), "n_particiones": len(combos),
            "n_pruebas": N, "is_best": is_b, "oos_best": oos_b, "pendiente": float(slope),
            "ordenada": float(intercept), "p_pérdida": float(np.mean(oos_b < 0)),
            "oos_mejor_menos_mediana": float(np.mean(oos_b - med_oos)),
            "percentil_oos_mediano": float(np.median(w)),
            "elegida": pd.Series(best).value_counts()}


def rank_null(ir1: np.ndarray, X2: np.ndarray, B: int, mean_block: float, rng) -> np.ndarray:
    """Distribución nula del Spearman entre ir1 (época 1) y el IR de la época 2 cuando
    en la época 2 ninguna prueba tiene exceso medio (se resta la media de cada columna):
    el orden de la época 2 es sólo ruido, con la dependencia real entre pruebas."""
    Xc = np.asarray(X2, float) - np.asarray(X2, float).mean(axis=0)
    draws = boot_ir(Xc, B, mean_block, rng)
    return np.array([sst.spearmanr(ir1, draws[i])[0] for i in range(B)])


def top_persistence(a: np.ndarray, b: np.ndarray, q: float = TOP_Q) -> float:
    """Fracción de las pruebas del cuantil superior q en a que siguen en él en b."""
    k = max(int(round(q * len(a))), 1)
    ta = set(np.argsort(-a)[:k])
    tb = set(np.argsort(-b)[:k])
    return len(ta & tb) / k


# ───────────────────────────── censo ─────────────────────────────

class Census:
    """Series de rentabilidad diaria de cada prueba y su ficha."""

    def __init__(self, d: dict):
        self.d = d
        self.R: dict[str, pd.Series] = {}
        self.meta: dict[str, dict] = {}
        self.live: dict[str, pd.Timestamp] = {}

    def add(self, name: str, r: pd.Series, module: str, family: str, group: str,
            live: pd.Timestamp | None = None, lrs_param: bool = False, note: str = ""):
        assert name not in self.R, name
        self.R[name] = r
        self.live[name] = live if live is not None else r.first_valid_index()
        self.meta[name] = {"módulo": module, "familia": family, "grupo": group,
                           "parametrización_LRS3x": lrs_param, "nota": note}


def idx_after(index: pd.DatetimeIndex, pos: int) -> pd.Timestamp:
    return index[min(pos, len(index) - 1)]


def r12_live(d: dict, var_fc: pd.Series, mu: pd.Series | None = None, offset: int = 0) -> pd.Timestamp:
    """Primera sesión en que r12.strategy tiene un peso decidido con previsión (antes
    lleva el peso a 0 con fillna): primera revisión (keep[offset::H]) con previsión,
    más el retraso de ejecución."""
    u = d["u"]
    m = mu if mu is not None else kelly.premium(u, d["f"], r12.MIN_OBS)
    ok = (var_fc.notna() & m.notna()).values
    keep = np.zeros(len(u), bool)
    keep[offset::r12.H] = True
    first = int(np.argmax(ok & keep))
    return idx_after(u.index, first + LAG)


def ml_forecasts(d: dict, cache: str | None = None) -> dict:
    """Previsiones de varianza de R12 (HAR y los cuatro modelos de ML) y la media de
    gradient boosting, con r12.walk_forward (mismo código, embargo y reentrenamiento).
    El bosque aleatorio usa ML_THREADS núcleos en vez de todos (mismo resultado)."""
    if cache and Path(cache).exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from threadpoolctl import threadpool_limits

    u = d["u"].dropna()
    target = (u ** 2).rolling(r12.H).sum().shift(-r12.H) * 252 / r12.H
    X = r12.features(u)
    ly = np.log(target.clip(lower=1e-6))
    models = r12._ml_models()
    models["Bosque aleatorio"] = lambda: RandomForestRegressor(
        n_estimators=200, min_samples_leaf=200, max_features=0.5, n_jobs=ML_THREADS,
        random_state=r12.SEED)
    out, secs = {}, {}
    with threadpool_limits(limits=ML_THREADS):
        t0 = time.time()
        out["HAR"] = r12.walk_forward(X[["lrv_d", "lrv_w", "lrv_m"]].apply(np.exp), target,
                                      LinearRegression, log_target=False).clip(lower=1e-4)
        secs["HAR"] = time.time() - t0
        for name, mk in models.items():
            t0 = time.time()
            out[name] = r12.walk_forward(X, ly, mk, log_target=True)
            secs[name] = time.time() - t0
        t0 = time.time()
        y = r10.window_sum(np.log1p(u) - np.log1p(d["rf"].reindex(u.index)), r12.H)
        out["μ gradient boosting"] = r12.walk_forward(X, y, models["Gradient boosting"], False)
        secs["μ gradient boosting"] = time.time() - t0
    out = {k: v.reindex(d["u"].index) for k, v in out.items()}
    out["_segundos"] = secs
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump(out, fh)
    return out


def build_census(d: dict, var_fc: pd.Series, ml: dict) -> Census:
    from letf_lab.research import r6_regime as r6
    from letf_lab.research import r15_execution as r15
    from letf_lab.research import r17_benchmarks as r17
    from letf_lab.research import r18_worlds as r18

    C = Census(d)
    u, rf = d["u"], d["rf"]
    G_LRS, G_K, G_O = list(GROUP_COLOR)

    # R13: la familia principal y el 3x
    R13, _ = r13.strategies(d, var_fc, lag=LAG)
    C.R[BENCH] = R13[BENCH]
    fam13 = {"3x comprar y mantener": ("comprar y mantener", G_O, False),
             "LRS 1x (original)": ("R13: LRS con otro apalancamiento", G_O, False),
             "LRS 2x (original)": ("R13: LRS con otro apalancamiento", G_O, False),
             LRS: ("R13: LRS 3x media × banda", G_LRS, True),
             "LRS 3x + banda ±2%": ("R13: LRS 3x media × banda", G_LRS, True),
             "LRS rotación 3x/1x": ("R13: LRS con otro apalancamiento", G_O, False),
             "½ Kelly GJR (sin tendencia)": ("R12–R13: ½ Kelly por volatilidad", G_K, False),
             "LRS-Kelly": ("R13–R18: LRS-Kelly", G_K, False),
             "LRS-Kelly suavizado": ("R13–R18: LRS-Kelly", G_K, False)}
    for k, (fam, grp, lp) in fam13.items():
        C.add(k, R13[k], "R13", fam, grp, lrs_param=lp)
    # R13 con la señal de precio: sus reglas, su rejilla de 30 y su mismo cierre (antes de
    # que r13.family meta la LRS 3x de precio en otra familia)
    Rp, _ = r13.price_variants(d, var_fc, lag=LAG)
    for k, r in Rp.items():
        grp = G_K if "Kelly" in k else G_O if k.startswith(("LRS 1x", "LRS 2x", "LRS rotación")) else G_LRS
        C.add(k, r, "R13", FAM_PX, grp, lrs_param=grp == G_LRS)
    for n in r13.GRID_N:
        for band in r13.GRID_BANDS:
            C.add(f"LRS 3x media {n} banda {band:.0%} · señal de precio",
                  r13.run_weights(d, r13.above_sma(d["px"], n, band), lag=LAG), "R13", FAM_PX, G_LRS,
                  lrs_param=True)
    Rpm, _ = r13.price_variants(d, var_fc, lag=1)
    for k, r in Rpm.items():
        C.add(f"{k} · mismo cierre", r, "R13", FAM_PX, C.meta[k]["grupo"],
              lrs_param=C.meta[k]["parametrización_LRS3x"])

    # R12: ½ Kelly cada 21 sesiones con cada modelo de varianza
    ew = (u.dropna() ** 2 * 252).ewm(alpha=0.06, adjust=False).mean().reindex(u.index)
    V12 = {"EWMA λ=0,94": ew, "ARCH(1)": r13.var_forecast(d, "arch"),
           "GARCH(1,1)": r13.var_forecast(d, "garch"), "GJR-GARCH": var_fc,
           "HAR": ml["HAR"]}
    for k in ("Ridge (lineal)", "Bosque aleatorio", "Gradient boosting", "Red neuronal (MLP)"):
        V12[k] = ml[k]
    for k, v in V12.items():
        C.add(f"½ Kelly R12 · {k}", r12.strategy(d, v, lag=LAG)[0], "R12",
              "R12–R13: ½ Kelly por volatilidad", G_K, live=r12_live(d, v))
    mu_total = ml["μ gradient boosting"] * 252 / r12.H + rf * 252 + 0.5 * var_fc
    C.add("½ Kelly R12 · GJR-GARCH + μ por gradient boosting", r12.strategy(d, var_fc, mu_total, lag=LAG)[0],
          "R12", "R12–R13: ½ Kelly por volatilidad", G_K, live=r12_live(d, var_fc, mu_total))

    # R13: la familia del SPA (repite los ½ Kelly de R12) y la rejilla de 30
    main, grid = r13.family(d, R13, lag=LAG)
    for k, r in main.items():
        if k not in C.R:
            C.add(k, r, "R13", "R12–R13: ½ Kelly por volatilidad", G_K, live=r12_live(d, var_fc))
    for k, r in grid.items():
        if k.startswith("LRS 3x media"):
            C.add(k, r, "R13", "R13: LRS 3x media × banda", G_LRS, lrs_param=True)
    # R13: mismo cierre (retraso 1)
    Rm, _ = r13.strategies(d, var_fc, lag=1)
    for k, r in Rm.items():
        if "comprar y mantener" in k:
            continue
        g = C.meta[k]["grupo"]
        C.add(f"{k} · mismo cierre", r, "R13", "R13: mismo cierre", g,
              lrs_param=C.meta[k]["parametrización_LRS3x"])

    # R15: retrasos, señal y calendario; desfases del ½ Kelly de R12. Con retraso 2, la
    # señal original y la de precio repiten la LRS 3x (original) y la de precio de R13:
    # dedupe las cuenta una vez
    res = r15.compute_market(d, var_fc, np.random.default_rng(r15.SEED))
    for (strat, kind, lag), r in res["lags"].items():
        grp = G_LRS if strat == LRS else G_K
        fam = "R15: LRS 3x ejecución y calendario" if strat == LRS else "R13–R18: LRS-Kelly"
        base = strat if kind == r15.BASE_SIGNAL else ("LRS 3x" if strat == LRS else strat) + " · señal de precio"
        C.add(f"{base} · retraso {lag}", r, "R15", fam, grp, live=res["start"], lrs_param=strat == LRS)
    for (strat, freq, day), r in res["cal"].items():
        grp = G_LRS if strat == LRS else G_K
        fam = "R15: LRS 3x ejecución y calendario" if strat == LRS else "R13–R18: LRS-Kelly"
        lab = f"revisión semanal ({day})" if freq == "semanal" else f"revisión cada 21 sesiones (desfase {day})"
        C.add(f"{strat} · {lab}", r, "R15", fam, grp, live=res["start"], lrs_param=strat == LRS)
    first, last, _ = r15.r12_windows(d)
    v12 = var_fc.where((var_fc.index >= first) & (var_fc.index <= last))
    for off, r in res["r12"].items():
        C.add(f"½ Kelly GJR R12 · desfase {off} (R15)", r, "R15", "R12–R13: ½ Kelly por volatilidad", G_K,
              live=r12_live(d, v12, offset=off))
    C.add("½ Kelly GJR diario (R15)", res["r12_daily"], "R15", "R12–R13: ½ Kelly por volatilidad", G_K,
          live=res["start"])

    # R17: referencias de la literatura
    R17 = r17.build(dict(d), var_fc)[0]
    for k in ("Faber 10 meses 1x", "Faber 10 meses 3x", "TSMOM 12 meses 3x",
              "Moreira–Muir (c en tiempo real)", r17.KELLY_EX_ANTE):
        grp = G_K if ("Moreira" in k or "Kelly" in k) else G_O
        C.add(k, R17[k], "R17", "R17: referencias de la literatura", grp)

    # R18: la LRS 3x de su historia real es la original de R13 (señal de rentabilidad total, la
    # de los mundos) y la LRS-Kelly con varianza EWMA
    u_sig = d.get("u_sig", u)
    sig_tr = r13.above_sma(r13.signal_index(d))
    C.add("LRS 3x (señal total, R18)", r13.run_weights(d, sig_tr, lag=LAG), "R18",
          "R15: LRS 3x ejecución y calendario", G_LRS, lrs_param=True)
    w_ew = r13.sticky(r13.kelly_exposure(d, ew) * sig_tr)
    C.add("LRS-Kelly (EWMA, señal total, R18)", r13.run_weights(d, w_ew, lag=LAG), "R18",
          "R13–R18: LRS-Kelly", G_K)
    # R6: media × volatilidad objetivo sobre el 3x (índice de rentabilidad total, sin mirar el futuro)
    for sma, vt in itertools.product([None, 50, 100, 150, 200, 250], [None, 0.2, 0.3, 0.4, 0.5]):
        if sma is None and vt is None:
            continue
        w = r6.signals(u_sig, 3.0, sma, vt)
        name = f"R6 · media {sma or '—'} × vol. objetivo {f'{vt:.0%}' if vt else '—'}"
        grp = G_LRS if vt is None else G_K
        C.add(name, r13.run_weights(d, w, lag=LAG), "R6", "R6: media × volatilidad objetivo", grp,
              live=idx_after(u.index, max(sma or 0, 21) + LAG), lrs_param=vt is None)
    w = r6.signals(u_sig, 3.0, 200, None, band=0.02)
    C.add("R6 · media 200 ±2% (rent. total)", r13.run_weights(d, w, lag=LAG), "R6",
          "R6: media × volatilidad objetivo", G_LRS, live=idx_after(u.index, 200 + LAG), lrs_param=True)

    # R10/R11: las 15 reglas de entrada y salida (sobre el índice de rentabilidad total de la señal)
    px = (1 + u_sig.fillna(0)).cumprod()
    dd = {"X": pd.DataFrame({"caída": px / px.cummax() - 1, "tendencia": px / px.rolling(200).mean() - 1}),
          "vol21": np.sqrt((u_sig ** 2 * 252).rolling(21).mean())}   # como R10 §C
    for k, w in r10.rules(dd).items():
        lp = k == "dentro sólo sobre SMA200"
        C.add(f"R10 · {k}", r13.run_weights(d, w, lag=LAG), "R10–R11", "R10–R11: reglas de entrada",
              G_LRS if lp else G_O, live=idx_after(u.index, 200 + LAG), lrs_param=lp)

    return C


def window(C: Census) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Del primer mes completo en que TODAS las pruebas están vivas a la última sesión común."""
    idx = C.d["u"].index
    last_live = max(C.live.values())
    per = idx.to_period("M")
    p0 = last_live.to_period("M") + 1
    start = idx[per >= p0][0]
    end = min(r.last_valid_index() for r in C.R.values())
    return start, end


def dedupe(Rw: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """Columnas únicas (la primera de cada grupo de series idénticas en la ventana)."""
    keep, dup = [], {}
    fp = {}
    for c in Rw.columns:
        v = Rw[c].values
        key = hash(np.round(v, 12).tobytes())
        if key in fp and np.max(np.abs(Rw[fp[key]].values - v)) < 1e-12:
            dup.setdefault(fp[key], []).append(c)
            continue
        fp[key] = c
        keep.append(c)
    return keep, dup


# ───────────────────────────── análisis ─────────────────────────────

def active(Rw: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
    lr = np.log1p(Rw.clip(lower=-0.9999))
    return lr.sub(np.log1p(ref.clip(lower=-0.9999)), axis=0)


def corr_eigs(A: pd.DataFrame) -> np.ndarray:
    return np.linalg.eigvalsh(np.corrcoef(A.values, rowvar=False))[::-1]


def trial_table(A: pd.DataFrame, V: float, n_trials: float, T: int) -> pd.DataFrame:
    """IR, momentos, PSR (frente a 0), DSR con n_trials (descriptivo) y MinTRL de cada
    prueba (A: excesos). El MinTRL frente a SR₀ se resuelve con SR₀(n) = z·√(V_disp + 1/n):
    con V observada, V_disp = máx(V − 1/T, 0) (la parte de la dispersión entre pruebas que
    no es ruido de muestreo); con la del azar, V_disp = 0."""
    m = sr_moments(A.values)
    sr0 = float(expected_max_sr(V, n_trials))
    zn = float(z_max(n_trials))
    v_disp = max(V - 1.0 / T, 0.0)
    return pd.DataFrame({
        "Δg": A.mean().values * 252, "IR": m["sr"] * np.sqrt(252), "asimetría": m["skew"],
        "curtosis": m["kurt"], "PSR": psr(m["sr"], 0.0, T, m["skew"], m["kurt"]),
        "DSR": psr(m["sr"], sr0, T, m["skew"], m["kurt"]),
        "MinTRL_años_IR*=0": min_trl(m["sr"], 0.0, m["skew"], m["kurt"]) / 252,
        "años_80%_potencia": power_trl(m["sr"], m["skew"], m["kurt"]) / 252,
        "MinTRL_años_SR0(n)_V_obs": [min_trl_deflated(s, sk, ku, zn, v_disp) / 252 for s, sk, ku in
                                     zip(m["sr"], m["skew"], m["kurt"])],
        "MinTRL_años_SR0(n)_V_azar": [min_trl_deflated(s, sk, ku, zn, 0.0) / 252 for s, sk, ku in
                                      zip(m["sr"], m["skew"], m["kurt"])],
    }, index=A.columns)


def nw_ratio(x: np.ndarray, lags: int = BLOCK) -> float:
    """Varianza de largo plazo (Bartlett) / varianza: T efectivo = T / cociente."""
    x = np.asarray(x, float) - np.mean(x)
    g0 = x @ x / len(x)
    s = g0
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * (x[l:] @ x[:-l]) / len(x)
    return float(s / g0)


STALE_UNTIL = pd.Timestamp("1989-12-31")   # precios desfasados: autocorrelación positiva hasta 1989 (R15)
STALE_LAGS = {"ρ₁": (1,), "ρ₁ + ρ₂": (1, 2)}
STALE_PERIODS = (("1939–1989", "1939", "1989"), ("1970–1999", "1970", "1999"), ("1970–1989", "1970", "1989"))


def yearly_autocorr(u: pd.Series, lags=(1,)) -> pd.Series:
    """Suma de las autocorrelaciones de orden k ∈ lags de u dentro de cada año natural,
    asignada a cada día de ese año."""
    x = u.dropna()
    rho = {y: sum(float(np.corrcoef(g.values[:-k], g.values[k:])[0, 1]) for k in lags)
           for y, g in x.groupby(x.index.year) if len(g) > 20}
    return pd.Series(u.index.year.map(rho).astype(float), index=u.index)


def stale_bias(u: pd.Series, L: pd.Series | float, lags=(1,), until=STALE_UNTIL) -> pd.Series:
    """Parte del crecimiento log diario de un fondo reajustado cada día con exposición L que
    solo existe por la autocorrelación del índice medido (precios desfasados): con la misma
    varianza a largo plazo, un índice sin autocorrelación daría L(L − 1)·ρ·u² menos cada día
    (0 para el índice, L = 1). ρ: autocorrelación de cada año (yearly_autocorr); solo hasta
    `until`. Aproximación de primer orden en ρ."""
    rho = yearly_autocorr(u, lags)
    L = L if isinstance(L, pd.Series) else pd.Series(float(L), index=u.index)
    L = L.reindex(u.index).fillna(0.0)
    b = L * (L - 1) * rho * u ** 2
    return b.where(u.index <= until, 0.0).fillna(0.0)


def etf_gap(d: dict) -> tuple[float, pd.Timestamp]:
    """δ: lo que el índice oficial con dividendos (^SP500TR) rinde de más al año que el ETF 1x
    (SPY) en la vida del 3x real, donde se calibra el sobrecoste del swap; y el primer día en
    que u es el ETF (antes, u es el índice y la réplica del 3x sale 3δ más barata)."""
    from letf_lab import data
    tr = data.panel(["^SP500TR"], "adj_close")["^SP500TR"].pct_change(fill_method=None)
    etf = d["e"].underlying
    px = data.panel([etf], "adj_close")[etf]
    both = pd.concat([tr, px.pct_change(fill_method=None)], axis=1, keys=["tr", "etf"]).dropna()
    live = r12.real_letf("UPRO").first_valid_index()
    return float((both.tr - both.etf).loc[live:].mean() * 252), px.first_valid_index()


def primary_sensitivity(d: dict, R: dict, E: dict, start, end, seed: int = SEED + 3) -> dict:
    """Hipótesis principal (LRS 3x original frente al 1x), la LRS frente a su L̄ constante y el 3x
    frente al 1x, sin corregir y descontando dos sesgos a favor del apalancamiento anteriores a
    los ETFs:
    - precios desfasados antes de 1990 (stale_bias con la exposición APLICADA, ρ₁ o ρ₁ + ρ₂);
    - sobrecoste del swap calibrado frente al ETF 1x y aplicado al índice antes del ETF: la
      réplica sale 3δ al año más barata (etf_gap); se resta L_t·δ/252 cada día anterior al ETF.
    Δg y p con el bootstrap del censo (bloque 63, B_BOOT, los mismos remuestreos para todas las
    columnas, generador propio); Δg y sesgo medio por periodos."""
    from letf_lab.research import r17_benchmarks as r17
    u = d["u"]
    bench = R[BENCH]
    L = E[LRS].shift(LAG).reindex(u.index).fillna(0.0)
    win = slice(start, end)
    Lbar = float(L.loc[win].mean())
    delta, etf0 = etf_gap(d)
    pre_etf = pd.Series((u.index <= etf0).astype(float), index=u.index)
    lg = lambda r: np.log1p(r.clip(lower=-0.9999))  # noqa: E731
    base = {"LRS 3x frente al 1x": (lg(R[LRS]) - lg(bench)).loc[win],
            "LRS 3x frente a su L̄ constante": (lg(R[LRS]) - lg(r17.const_leverage(d, Lbar))).loc[win],
            "3x comprar y mantener frente al 1x": (lg(R["3x comprar y mantener"]) - lg(bench)).loc[win]}

    def biases(lags, calib: bool) -> dict:
        z = pd.Series(0.0, index=u.index)
        sl, sc, s3 = z, z, z
        if lags is not None:
            sl, sc, s3 = stale_bias(u, L, lags), stale_bias(u, Lbar, lags), stale_bias(u, 3.0, lags)
        if calib:
            sl = sl + L * delta / 252 * pre_etf
            sc = sc + Lbar * delta / 252 * pre_etf
            s3 = s3 + 3 * delta / 252 * pre_etf
        return {"LRS 3x frente al 1x": sl, "LRS 3x frente a su L̄ constante": sl - sc,
                "3x comprar y mantener frente al 1x": s3}

    corr = {"sin corregir": biases(None, False), "precios desfasados (ρ₁)": biases((1,), False),
            "precios desfasados (ρ₁ + ρ₂)": biases((1, 2), False),
            "sobrecoste frente al índice": biases(None, True),
            "las dos (ρ₁ y sobrecoste)": biases((1,), True)}
    cols = {(cmp_, tag): base[cmp_] - b[cmp_].loc[win] for tag, b in corr.items() for cmp_ in base}
    X = pd.DataFrame(cols)
    bt = boot_test(X.values, B_BOOT, BLOCK, np.random.default_rng(seed))
    rows = {}
    for j, (cmp_, tag) in enumerate(X.columns):
        row = {"Δg": X.iloc[:, j].mean() * 252, "p_boot": float(bt["p"][j])}
        for lab, a, b in STALE_PERIODS:
            row[f"Δg {lab}"] = X.iloc[:, j].loc[a:f"{b}-12-31"].mean() * 252
        bb = corr[tag][cmp_].loc[win]
        row["sesgo"] = bb.mean() * 252
        for lab, a, b in STALE_PERIODS:
            row[f"sesgo {lab}"] = bb.loc[a:f"{b}-12-31"].mean() * 252
        rows[(cmp_, tag)] = row
    out = pd.DataFrame(rows).T.astype(float)
    pp_cols = [c for c in out.columns if c.startswith(("Δg", "sesgo"))]
    out[pp_cols] = out[pp_cols] * 100        # en puntos, con dos decimales: el sesgo es de décimas
    out = out.rename(columns={c: f"{c} (pp)" for c in pp_cols})
    out.index.names = ["comparación", "corrección"]
    return {"tabla": out, "L̄": Lbar, "δ": delta, "etf0": etf0.date(),
            "desde": X.index[0].date(), "hasta": X.index[-1].date()}


def analyse(C: Census, rng) -> dict:
    d = C.d
    start, end = window(C)
    names_all = [k for k in C.R if k != BENCH]
    Rw_all = pd.DataFrame({k: C.R[k].loc[start:end] for k in names_all})
    assert not Rw_all.isna().any().any(), Rw_all.columns[Rw_all.isna().any()].tolist()
    keep, dup = dedupe(Rw_all)
    Rw = Rw_all[keep]
    bench = C.R[BENCH].loc[start:end]
    rf = d["rf"].loc[start:end]
    T, K = Rw.shape
    meta = pd.DataFrame(C.meta).T.loc[keep]
    meta["duplicados"] = [len(dup.get(k, [])) for k in keep]
    meta["también_como"] = ["; ".join(dup.get(k, [])) for k in keep]

    out = {"start": start, "end": end, "T": T, "K": K, "K_bruto": len(names_all), "dup": dup,
           "meta": meta, "Rw": Rw, "bench": bench, "rf": rf}

    # rendimiento descriptivo
    desc = {}
    for k in keep:
        r = Rw[k]
        desc[k] = {"CAGR": metrics.cagr(r), "vol": metrics.vol(r), "maxDD": metrics.max_dd(r),
                   "Sharpe": metrics.sharpe(r, rf)}
    out["desc"] = pd.DataFrame(desc).T
    out["bench_desc"] = {"CAGR": metrics.cagr(bench), "vol": metrics.vol(bench),
                         "maxDD": metrics.max_dd(bench), "Sharpe": metrics.sharpe(bench, rf)}

    # excesos frente al 1x (todas) y frente a la LRS original (las alternativas)
    A1 = active(Rw, bench)
    alt = [k for k in keep if k != LRS]
    A2 = active(Rw[alt], Rw[LRS])
    out["A1"], out["A2"] = A1, A2

    # referencias sin sincronización: L constante (fracción L/3 en el 3x, resto en letras)
    from letf_lab.research import r17_benchmarks as r17
    stat = {}
    for L in (1.5, 2.0, 2.5, 3.0):
        r = r17.const_leverage(d, L).loc[start:end]
        a = np.log1p(r.clip(lower=-0.9999)) - np.log1p(bench.clip(lower=-0.9999))
        m = sr_moments(a.values[:, None])
        stat[f"L constante {L:g}"] = {"CAGR": metrics.cagr(r), "Δg": a.mean() * 252,
                                      "IR": float(m["sr"][0]) * np.sqrt(252),
                                      "PSR": float(psr(m["sr"][0], 0.0, T, m["skew"][0], m["kurt"][0]))}
    out["static"] = pd.DataFrame(stat).T

    # número efectivo de pruebas
    eig = {"excesos sobre el 1x": corr_eigs(A1), "rentabilidades": corr_eigs(np.log1p(Rw)),
           "alternativas: excesos sobre la LRS 3x": corr_eigs(A2)}
    out["eig"] = eig
    out["neff"] = pd.DataFrame({k: n_eff(v) for k, v in eig.items()})
    out["rho_bar"] = {}
    for tag, A in (("1x", A1), ("LRS", A2)):
        Rc = np.corrcoef(A.values, rowvar=False)
        out["rho_bar"][tag] = float((Rc.sum() - len(Rc)) / (len(Rc) * (len(Rc) - 1)))
    ne1 = out["neff"].loc[NEFF_MAIN, "excesos sobre el 1x"]
    ne2 = out["neff"].loc[NEFF_MAIN, "alternativas: excesos sobre la LRS 3x"]

    # DSR: dispersión observada de los IR entre pruebas (V) o la del puro azar (1/T)
    m1, m2 = sr_moments(A1.values), sr_moments(A2.values)
    V1, V2 = float(np.var(m1["sr"], ddof=1)), float(np.var(m2["sr"], ddof=1))
    V0 = 1.0 / T
    out["V"] = {"V1": V1, "V2": V2, "V0": V0}
    out["ne"] = {"ne1": ne1, "ne2": ne2}
    t1 = trial_table(A1, V1, ne1, T)
    t1["DSR_V_azar"] = psr(m1["sr"], expected_max_sr(V0, ne1), T, m1["skew"], m1["kurt"])
    t1["DSR_N_total"] = psr(m1["sr"], expected_max_sr(V1, K), T, m1["skew"], m1["kurt"])
    t2 = trial_table(A2, V2, ne2, T)
    t2["DSR_V_azar"] = psr(m2["sr"], expected_max_sr(V0, ne2), T, m2["skew"], m2["kurt"])
    t2["DSR_N_total"] = psr(m2["sr"], expected_max_sr(V2, len(alt)), T, m2["skew"], m2["kurt"])
    # T efectivo por autocorrelación de los excesos (Newey-West, 63 retardos)
    t1["T_efectivo/T"] = [1 / nw_ratio(A1[k].values) for k in keep]
    t2["T_efectivo/T"] = [1 / nw_ratio(A2[k].values) for k in alt]
    t1["DSR_T_efectivo"] = psr(m1["sr"], expected_max_sr(V1, ne1), T * t1["T_efectivo/T"].values,
                               m1["skew"], m1["kurt"])
    t2["DSR_T_efectivo"] = psr(m2["sr"], expected_max_sr(V2, ne2), T * t2["T_efectivo/T"].values,
                               m2["skew"], m2["kurt"])

    # p-valores por bootstrap (mismos remuestreos para todas): Holm, BH y BHY (acotan la
    # dependencia), Šidák a N_eff (la corrige con el número efectivo) y el step-down de
    # Romano–Wolf con los mismos remuestreos (la usa directamente); este último también
    # sólo con las reglas operables (sin las de mismo cierre)
    op = {}
    for t, A, ne, cols in ((t1, A1, ne1, keep), (t2, A2, ne2, alt)):
        bt = boot_test(A.values, B_BOOT, BLOCK, rng)
        t["p_boot"], t["p_empírico"], t["t_EE_boot"] = bt["p"], bt["p_emp"], bt["t"]
        t["p_Holm"] = stats.holm(t.p_boot.values)
        t["q_BH"] = stats.benjamini_hochberg(t.p_boot.values)
        t["q_BHY"] = bhy(t.p_boot.values)
        t["p_Šidák_Neff"] = sidak(t.p_boot.values, ne)
        t["p_RW"] = romano_wolf(bt["t"], bt["t_boot"])
        io = [i for i, k in enumerate(cols) if not is_same_close(k)]
        t["p_RW_operables"] = np.nan
        t.iloc[io, t.columns.get_loc("p_RW_operables")] = romano_wolf(bt["t"][io], bt["t_boot"][:, io])
        op[id(t)] = [cols[i] for i in io]
    out["t1"], out["t2"] = t1, t2
    op1, op2 = op[id(t1)], op[id(t2)]
    out["operables"] = {"1x": op1, "LRS": op2}

    # SPA de Hansen sobre todo el censo (Δg diario)
    out["spa"] = {"frente al 1x": {**stats.spa_test(A1.values, BLOCK, B_SPA, rng), "reglas": K,
                                   "mejor": t1["Δg"].idxmax(), "Δg_mejor": t1["Δg"].max()},
                  "frente a la LRS 3x original": {**stats.spa_test(A2.values, BLOCK, B_SPA, rng),
                                                  "reglas": len(alt), "mejor": t2["Δg"].idxmax(),
                                                  "Δg_mejor": t2["Δg"].max()}}
    # sensibilidad (añadida tras la revisión, con su propio generador para no alterar el
    # resto): SPA sólo con las reglas operables, sin las de mismo cierre
    rng_s = np.random.default_rng(SEED + 1)
    out["spa_op"] = {"frente al 1x": {**stats.spa_test(A1[op1].values, BLOCK, B_SPA, rng_s),
                                      "reglas": len(op1), "mejor": t1.loc[op1, "Δg"].idxmax()},
                     "frente a la LRS 3x original": {**stats.spa_test(A2[op2].values, BLOCK, B_SPA, rng_s),
                                                     "reglas": len(op2), "mejor": t2.loc[op2, "Δg"].idxmax()}}

    # la LRS 3x preespecificada fuera de la muestra de Gayed y Bilello (termina en 2015), con la
    # de precio y la mejor operable del censo como comparación
    post = A1.index >= GB_END
    rng_o = np.random.default_rng(SEED + 2)
    best_op = t1.loc[op1, "IR"].idxmax()
    pk = list(dict.fromkeys([LRS, PX_LRS, best_op]))
    out["best_op"] = best_op
    Xo = A1.loc[post, pk]
    bo = boot_test(Xo.values, B_BOOT, BLOCK, rng_o)
    mo = sr_moments(Xo.values)
    out["post"] = pd.DataFrame({
        "CAGR": [metrics.cagr(Rw.loc[post, k]) for k in pk], "CAGR 1x": metrics.cagr(bench[post]),
        "Δg": Xo.mean().values * 252, "IR": mo["sr"] * np.sqrt(252),
        "PSR": psr(mo["sr"], 0.0, int(post.sum()), mo["skew"], mo["kurt"]), "p_boot": bo["p"]},
        index=pk)
    out["post_n"] = int(post.sum())
    out["post_span"] = (A1.index[post][0], A1.index[post][-1])
    out["gb_overlap"] = float((A1.index < GB_END).mean())

    # DSR en función de N para las pruebas que se dibujan
    out["curves"] = {}
    for tag, A, m, V in (("1x", A1, m1, V1), ("LRS", A2, m2, V2)):
        cols = list(A.columns)
        cur = {}
        for k in cols:
            i = cols.index(k)
            cur[k] = {"V": psr(m["sr"][i], expected_max_sr(V, N_GRID), T, m["skew"][i], m["kurt"][i]),
                      "V0": psr(m["sr"][i], expected_max_sr(V0, N_GRID), T, m["skew"][i], m["kurt"][i])}
        out["curves"][tag] = cur
    out["sr0"] = pd.DataFrame({
        "N": N_GRID,
        "SR0_anual_V_1x": expected_max_sr(V1, N_GRID) * np.sqrt(252),
        "SR0_anual_V_LRS": expected_max_sr(V2, N_GRID) * np.sqrt(252),
        "SR0_anual_azar": expected_max_sr(V0, N_GRID) * np.sqrt(252)}).set_index("N")

    # CSCV / PBO
    lrs_fam = [k for k in keep if bool(meta.loc[k, "parametrización_LRS3x"])]
    cs = {}
    for lab, X, metric in (
            ("todo el censo · IR frente al 1x", A1, "ir"),
            ("todo el censo · Δg frente al 1x (= orden por CAGR)", A1, "mean"),
            ("parametrizaciones de la LRS 3x · IR frente al 1x", A1[lrs_fam], "ir"),
            ("parametrizaciones de la LRS 3x · Δg frente al 1x", A1[lrs_fam], "mean"),
            ("Kelly y volatilidad · IR frente al 1x",
             A1[[k for k in keep if meta.loc[k, "grupo"] == "Kelly y volatilidad"]], "ir"),
            ("alternativas · IR frente a la LRS 3x original", A2, "ir"),
            ("alternativas · Δg frente a la LRS 3x original", A2, "mean")):
        res = cscv(X.values, S_CSCV, metric)
        res["nombres"] = list(X.columns)
        cs[lab] = res
    out["cscv"] = cs
    out["lrs_fam"] = lrs_fam
    # puesto de la original y de la de precio entre las parametrizaciones de la LRS 3x (IR, ventana completa)
    ir_fam = t1.loc[lrs_fam, "IR"].sort_values(ascending=False)
    out["rank_lrs"] = {k: (int(list(ir_fam.index).index(k)) + 1, len(ir_fam)) for k in (LRS, PX_LRS)
                       if k in ir_fam.index}

    # rangos entre épocas
    e1 = A1.index < SPLIT
    ep = {}
    for tag, A in (("1x", A1), ("LRS", A2)):
        X1, X2 = A[e1].values, A[~e1].values
        ir1, ir2 = sr_moments(X1)["sr"] * np.sqrt(252), sr_moments(X2)["sr"] * np.sqrt(252)
        g1, g2 = X1.mean(axis=0) * 252, X2.mean(axis=0) * 252
        rho = sst.spearmanr(ir1, ir2)[0]
        null = rank_null(ir1, X2, B_RANK, BLOCK, rng)
        ep[tag] = {"names": list(A.columns), "ir1": ir1, "ir2": ir2, "g1": g1, "g2": g2,
                   "rho_ir": rho, "tau_ir": sst.kendalltau(ir1, ir2)[0],
                   "rho_g": sst.spearmanr(g1, g2)[0], "tau_g": sst.kendalltau(g1, g2)[0],
                   "null_p95": float(np.percentile(null, 95)), "null_mean": float(null.mean()),
                   "p_null": float((1 + np.sum(null >= rho)) / (1 + len(null))),
                   "top_ir": top_persistence(ir1, ir2),
                   "top_g": top_persistence(g1, g2), "n1": int(e1.sum()), "n2": int((~e1).sum())}
    out["epochs"] = ep
    # por familias (IR frente al 1x)
    fam_rows = {}
    for fam, g in meta.groupby("familia"):
        ks = list(g.index)
        if len(ks) < 5:
            continue
        i = [ep["1x"]["names"].index(k) for k in ks]
        a, b = ep["1x"]["ir1"][i], ep["1x"]["ir2"][i]
        fam_rows[fam] = {"pruebas": len(ks), "Spearman_IR": sst.spearmanr(a, b)[0],
                         "Kendall_IR": sst.kendalltau(a, b)[0],
                         "mejor_1939_1991": ks[int(np.argmax(a))], "su_puesto_1992_2026":
                         f"{int(np.sum(b > b[int(np.argmax(a))])) + 1} de {len(ks)}"}
    out["epochs_fam"] = pd.DataFrame(fam_rows).T
    return out


# ───────────────── p-valores de titular publicados (R16–R18) ─────────────────

def published_table(base: Path | None = None) -> pd.DataFrame:
    """Los p de titular de R16, R17 y R18 (lista cerrada de cada módulo), leídos de
    <base>/<módulo>/p_titular.csv con provenance.read_titular, que exige que la huella del
    código con que se generaron coincida con la del código actual. Holm, BH y BHY sobre todos."""
    base = REPORTS if base is None else Path(base)
    df = pd.concat([provenance.read_titular(base / m, m) for m in TITULAR], ignore_index=True)
    df = df[["módulo", "mercado", "contraste", "p", "commit", "huella_código"]].copy()
    df["p_Holm"] = stats.holm(df.p.values)
    df["q_BH"] = stats.benjamini_hochberg(df.p.values)
    df["q_BHY"] = bhy(df.p.values)
    return df.sort_values("p").reset_index(drop=True)


# ───────────────────────────── formato ─────────────────────────────

def fp(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fyears(x: float) -> str:
    if not np.isfinite(x):
        return "∞"
    return f"{x:,.0f}".replace(",", " ") if x >= 100 else f"{x:.1f}"


def pc(x: float, k: int = 1) -> str:
    return f"{x * 100:.{k}f}%".replace(".", ",")


def pp(x: float, k: int = 1, sign: bool = True) -> str:
    return (f"{x * 100:+.{k}f}" if sign else f"{x * 100:.{k}f}").replace(".", ",").replace("-", "−") + " pp"


def num(x: float, k: int = 2) -> str:
    return f"{x:.{k}f}".replace(".", ",").replace("-", "−")


def label(name: str) -> str:
    """Nombre corto para leyendas y anotaciones: «LRS 3x (original)» → «LRS 3x»,
    «LRS 3x (señal de precio)» → «LRS 3x · precio», «… · retraso 1» → «… · mismo cierre»."""
    return (name.replace(" (señal de precio)", " · precio").replace(" · señal de precio", " · precio")
            .replace(" · retraso 1", " · mismo cierre").replace(" · retraso 2", "")
            .replace(" (original)", ""))


def short(name: str, n: int = 44) -> str:
    return name if len(name) <= n else name[:n - 1] + "…"


# ───────────────────────────── figuras ─────────────────────────────

def _axes_style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _n_marks(ax, ne_tab, col, K, y_text=1.01):
    lo, hi = ne_tab[col].iloc[:5].min(), ne_tab[col].iloc[:5].max()
    ax.axvspan(lo, hi, color="#ecebe7", zorder=0)
    ne = ne_tab.loc[NEFF_MAIN, col]
    ax.axvline(ne, color=MUTED, lw=1.2, ls="-.")
    ax.axvline(K, color=MUTED, lw=1.2, ls=":")
    ax.text(ne, y_text, f" N_eff = {ne:.0f}", color=MUTED, fontsize=11, va="bottom",
            transform=ax.get_xaxis_transform())
    ax.text(K, y_text, f" N = {K}", color=MUTED, fontsize=11, va="bottom",
            transform=ax.get_xaxis_transform())


def fig_dsr(A: dict, rep: Report, picks1: list[str], picks2: list[str]):
    ne_tab = A["neff"]
    t1 = A["t1"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    # arriba: el contraste al 5% con pruebas múltiples, p de Šidák en función de N
    ax = axes[0]
    _axes_style(ax)
    _n_marks(ax, ne_tab, "excesos sobre el 1x", A["K"])
    ax.axhline(ALPHA, color=INK, lw=0.9, ls="--")
    ax.text(1.05, ALPHA * 1.08, "5%", color=INK, fontsize=11, va="bottom")
    for k, c in zip(picks1, LINE_COLORS):
        p = float(t1.loc[k, "p_boot"])
        ax.loglog(N_GRID, np.clip(sidak(p, N_GRID), 1e-4, 1), color=c, lw=2,
                  label=f"{label(k)} (Romano–Wolf {fpc(float(t1.loc[k, 'p_RW']))})")
    ax.set(xlim=(1, 1000), ylim=(5e-4, 1.05), xlabel="Número de pruebas independientes N",
           ylabel="p ajustado de Šidák, 1 − (1 − p)^N")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.legend(loc="lower right", fontsize=10.5)
    ax.set_title("Contraste al 5% frente al 1x: p de Šidák según N", pad=22)
    # centro y abajo: el DSR, descriptivo
    for ax, tag, picks, col, K, ttl in (
            (axes[1], "1x", picks1, "excesos sobre el 1x", A["K"],
             "DSR frente al 1x (descriptivo, no es un contraste)"),
            (axes[2], "LRS", picks2, "alternativas: excesos sobre la LRS 3x", A["K"] - 1,
             "DSR de las alternativas frente a la LRS 3x original")):
        _axes_style(ax)
        _n_marks(ax, ne_tab, col, K)
        for k, c in zip(picks, LINE_COLORS):
            cur = A["curves"][tag][k]
            ax.semilogx(N_GRID, cur["V"], color=c, lw=2, label=label(k))
            ax.semilogx(N_GRID, cur["V0"], color=c, lw=1.4, ls="--")
        ax.set(xlim=(1, 1000), ylim=(0, 1.0), xlabel="Número de pruebas independientes N",
               ylabel="DSR = PSR frente a SR₀(N)")
        ax.legend(loc="lower left" if tag == "1x" else "upper right", fontsize=10.5,
                  title="continua: V observada · discontinua: V = 1/T", title_fontsize=10)
        ax.set_title(ttl, pad=22)
    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ne1 = ne_tab.loc[NEFF_MAIN, "excesos sobre el 1x"]
    rep.fig(fig, "dsr_frente_a_n",
            "Arriba, el contraste con pruebas múltiples: p unilateral (bootstrap) de cada regla frente "
            "al 1x ajustado por Šidák como si fuera la mejor de N pruebas independientes; la regla "
            "sobrevive a la búsqueda al 5% mientras su curva queda bajo la discontinua. En la leyenda, "
            f"el p del step-down de Romano–Wolf sobre las {A['K']} reglas, que no necesita fijar N (para la "
            "LRS 3x original, preespecificada, sólo como referencia: se contrasta sola). En el "
            "centro, el Deflated Sharpe Ratio frente al 1x; abajo, el de las alternativas frente a la "
            "LRS 3x original: la confianza en que el IR supera al máximo esperado de N pruebas sin "
            "habilidad, con la dispersión de los IR observada (continua) o la del puro azar, 1/T "
            "(discontinua). Es una medida descriptiva: «DSR ≥ 0,95» no es un contraste al 5% (con "
            f"N = {ne1:.0f} su error familiar bajo la nula es del "
            f"{pc(float(dsr_rule_size(ne1)), 1)}). La franja gris es el rango de los cinco estimadores "
            "de N efectivo; la raya y punto, el principal (Li y Ji); la punteada, el total de pruebas "
            "únicas.")


def fig_pbo(A: dict, rep: Report, labels: list[str]):
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 4.2 * len(labels)))
    bins = np.linspace(-7, 7, 57)
    for ax, lab in zip(np.atleast_1d(axes), labels):
        _axes_style(ax)
        res = A["cscv"][lab]
        lg = np.clip(res["logits"], -6.99, 6.99)
        h, e = np.histogram(lg, bins=bins)
        h = h / len(lg)
        ctr = (e[:-1] + e[1:]) / 2
        ax.bar(ctr, h, width=np.diff(e) * 0.9, color=np.where(ctr <= 0, "#e34948", "#2a78d6"))
        ax.axvline(0, color=INK, lw=1)
        ax.set(xlim=(-7, 7), ylabel="Fracción de particiones",
               title=f"{lab[0].upper() + lab[1:]}\nPBO = {pc(res['pbo'])} · {res['n_pruebas']} pruebas")
    np.atleast_1d(axes)[-1].set_xlabel("Logit del rango fuera de muestra de la mejor dentro de muestra")
    rep.fig(fig, "pbo_logits",
            "Distribución de los logits λ = ln(ω/(1−ω)) de CSCV con 16 bloques (12.870 particiones): ω es "
            "el rango relativo, fuera de muestra, de la prueba con mejor IR dentro de muestra. En rojo, las "
            "particiones con λ ≤ 0, en las que la elegida queda por debajo de la mediana fuera de muestra; "
            "su fracción es la probabilidad de sobreajuste (PBO). Los logits se recortan a ±7 para dibujar.")


FIG_LABEL = {"½ Kelly GJR (sin tendencia)": "Medio Kelly (sin tendencia)"}   # nombres del artículo


def _side_labels(ax, pts: list, x_text: float, gap: float):
    """Rótulos en una columna a la derecha de los datos (x_text, en unidades de datos), con líneas
    guía hasta cada punto: no pisan ningún punto ni se pisan entre sí (separación mínima `gap`)."""
    pts = sorted(pts, key=lambda p: p[2])
    ys = []
    for _, _, y in pts:
        ys.append(max(y, ys[-1] + gap) if ys else y)
    lo_, hi_ = ax.get_ylim()
    over = ys[-1] - (hi_ - gap / 2)
    if over > 0:                          # si la columna se sale por arriba, se baja entera
        ys = [y - over for y in ys]
    for (lab, x, y), yt in zip(pts, ys):
        ax.annotate(lab, (x, y), xytext=(x_text, yt), textcoords="data", fontsize=11, color=INK,
                    va="center", ha="left", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                                            shrinkA=0, shrinkB=5), zorder=5)


def fig_ranks(A: dict, rep: Report):
    ep = A["epochs"]["1x"]
    names = ep["names"]
    meta = A["meta"]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 12.6))
    n = len(names)
    r1 = sst.rankdata(ep["ir1"]) / n * 100
    r2 = sst.rankdata(ep["ir2"]) / n * 100
    focus = [LRS, "3x comprar y mantener", "LRS-Kelly", "½ Kelly GJR (sin tendencia)"]
    lo = min(ep["ir1"].min(), ep["ir2"].min()) - 0.05
    hi = max(ep["ir1"].max(), ep["ir2"].max()) + 0.05
    for ax, (x, y, xl, yl, ttl) in zip(axes, (
            (r1, r2, "Percentil del IR en 1939–1991", "Percentil del IR en 1992–2026",
             f"Rangos del IR frente al 1x: Spearman {num(ep['rho_ir'])}\n"
             f"(sin persistencia, percentil 95: {num(ep['null_p95'])})"),
            (ep["ir1"], ep["ir2"], "IR anual frente al 1x en 1939–1991",
             "IR anual frente al 1x en 1992–2026", "Valores del IR frente al 1x en cada época"))):
        _axes_style(ax)
        for g, c in GROUP_COLOR.items():
            m = (meta.loc[names, "grupo"] == g).values
            ax.scatter(x[m], y[m], s=42, color=c, marker=GROUP_MARKER[g], edgecolor="white",
                       linewidth=0.8, label=f"{g} ({m.sum()})", zorder=3)
        top = ax is axes[0]
        # la columna de rótulos va a la derecha de todos los puntos (x ≤ 100 arriba, ≤ hi abajo)
        x0, x1 = (-3, 100) if top else (lo, hi)
        span = x1 - x0
        ax.set_xlim(x0, x1 + 0.62 * span)
        ax.set_ylim(*((-3, 103) if top else (lo, hi)))
        for k in focus:
            i = names.index(k)
            ax.scatter([x[i]], [y[i]], s=90, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=4)
        _side_labels(ax, [(FIG_LABEL.get(k, label(k)), x[names.index(k)], y[names.index(k)]) for k in focus],
                     x1 + 0.06 * span, 0.07 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
        ax.set_xticks([t for t in ax.get_xticks() if x0 <= t <= x1])
        ax.set(xlabel=xl, ylabel=yl, title=ttl)
    # la leyenda en su propia franja: dentro de los paneles tapaba puntos
    legend_band(fig, *axes[0].get_legend_handles_labels(), ncol=2, fontsize=11)
    axes[1].plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    axes[1].axhline(0, color=GRID, lw=1)
    axes[1].axvline(0, color=GRID, lw=1)
    for ax in axes:   # marcas con coma decimal, como el título (num)
        comma_axis(ax)
    rep.fig(fig, "rangos_epocas",
            "Cada punto es una prueba del censo. Arriba, su percentil de IR frente al 1x en 1939–1991 "
            "(eje horizontal) y en 1992–2026 (vertical): si el orden se conservara, los puntos irían por "
            "la diagonal. Abajo, los valores del IR anual en cada época con la diagonal de igualdad. "
            "Color y forma: grupo de reglas; con aro negro, las cuatro reglas rotuladas.")


def fig_spectrum(A: dict, rep: Report):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    for ax in axes:
        _axes_style(ax)
    for (k, lam), c in zip(A["eig"].items(), LINE_COLORS):
        lam = np.clip(lam, 1e-8, None)
        i = np.arange(1, len(lam) + 1)
        ne = A["neff"].loc[NEFF_MAIN, k]
        axes[0].semilogy(i, lam, color=c, lw=2, label=f"{k} (N_eff Li–Ji = {ne:.0f})")
        axes[1].plot(i, np.cumsum(lam) / lam.sum() * 100, color=c, lw=2, label=k)
    axes[0].axhline(1, color=MUTED, lw=1, ls="--")
    axes[0].set(xscale="log", xlabel="Orden del autovalor", ylabel="Autovalor (escala log)",
                title="Autovalores de la matriz de correlación de las pruebas")
    axes[0].legend(loc="lower left", fontsize=10.5)
    axes[1].set(xscale="log", xlabel="Número de componentes", ylabel="% de la varianza total",
                title="Varianza acumulada", ylim=(0, 101))
    axes[1].axhline(95, color=MUTED, lw=1, ls="--")
    axes[1].legend(loc="lower right", fontsize=10.5)
    rep.fig(fig, "espectro",
            "Autovalores ordenados de la matriz de correlación diaria de las pruebas únicas del censo "
            "(arriba, escala log; la discontinua marca λ = 1) y la varianza acumulada que explican "
            "(abajo). Tres matrices: excesos de log-rentabilidad sobre el 1x, log-rentabilidades en "
            "bruto y excesos de las alternativas sobre la LRS 3x original.")


# ───────────────────────────── informe ─────────────────────────────

def census_table(A: dict) -> pd.DataFrame:
    meta = A["meta"]
    raw = pd.DataFrame({"módulo": [A["dup_meta"][k]["módulo"] for k in A["names_all"]],
                        "familia": [A["dup_meta"][k]["familia"] for k in A["names_all"]]},
                       index=A["names_all"])
    raw["única"] = raw.index.isin(meta.index)
    g = raw.groupby(["módulo", "familia"], sort=False).agg(construidas=("única", "size"),
                                                           únicas=("única", "sum"))
    g = g.astype(int).reset_index()
    g.loc[len(g)] = ["total", "", len(raw), int(raw["única"].sum())]
    return g


def focus_rows(A: dict, t: pd.DataFrame, extra: list[str]) -> list[str]:
    rows = [k for k in FOCAL if k in t.index] + [k for k in extra if k in t.index and k not in FOCAL]
    return list(dict.fromkeys(rows))


def show_trials(t: pd.DataFrame, rows: list[str], A: dict, vs: str) -> pd.DataFrame:
    df = t.loc[rows].copy()
    out = pd.DataFrame(index=[short(k, 60) for k in rows])
    out["CAGR"] = A["desc"].loc[rows, "CAGR"].values
    out["Δg"] = df["Δg"].values
    out["IR"] = df["IR"].values
    out["PSR"] = df["PSR"].values
    out["DSR (N_eff, V obs.)"] = df["DSR"].values
    out["DSR (N_eff, V azar)"] = df["DSR_V_azar"].values
    out["DSR (N total)"] = df["DSR_N_total"].values
    out["p_boot"] = [fp(p) for p in df["p_boot"]]
    out["p Šidák (N_eff)"] = [fp(p) for p in df["p_Šidák_Neff"]]
    out["p Romano–Wolf"] = [fp(p) for p in df["p_RW"]]
    out["p_Holm"] = [fp(p) for p in df["p_Holm"]]
    return out


def fpc(p: float) -> str:
    """p para el texto, con coma decimal."""
    return fp(p).replace(".", ",")


def mc_se(p: float = ALPHA, B: int = B_BOOT) -> float:
    """Error típico de Monte Carlo de un p bootstrap con B remuestreos: √(p(1 − p)/B)
    (≈ 0,005 cerca de 0,05 con B = 2.000)."""
    return float(np.sqrt(p * (1 - p) / B))


def at_alpha(p: float, alpha: float = ALPHA, B: int = B_BOOT) -> bool:
    """¿Está p a menos de un error de Monte Carlo del umbral? Entonces «pasa» o «no pasa»
    depende de los remuestreos, no de los datos."""
    return bool(np.isfinite(p) and abs(p - alpha) < mc_se(alpha, B))


def fp_edge(p: float) -> str:
    """Como fp, pero con cuatro decimales cerca del umbral: 0,05006 no se lee «0.050»."""
    return f"{p:.4f}" if at_alpha(p) else fp(p)


def p_edge_text(p: float) -> str:
    """«p = 0,0501, en el umbral del 5%: la distancia está dentro del error de Monte Carlo
    (±0,005)» o, lejos del umbral, «p = 0,044»."""
    if not at_alpha(p):
        return f"p = {fpc(p)}"
    return (f"p = {fp_edge(p).replace('.', ',')}, en el umbral del 5%: la distancia está dentro del error "
            f"de Monte Carlo (±{num_es(mc_se(), 3)})")


def peq(p: float) -> str:
    """«p = 0,017» o «p < 0,001» para el texto."""
    return "p < 0,001" if p < 0.001 else f"p = {fpc(p)}"


def fyc(x: float) -> str:
    """Años para el texto, con coma decimal."""
    return fyears(x).replace(".", ",")


def names_list(ks) -> str:
    ks = [f"«{label(k)}»" for k in ks]
    return "ninguna" if not ks else ks[0] if len(ks) == 1 else ", ".join(ks[:-1]) + " y " + ks[-1]


def run(cache: str | None = None):
    t_start = time.time()
    rep = Report("r19_census", "R19 — Censo de pruebas: cuánto descuenta haber probado tantas variantes")
    for f in rep.dir.glob("table_*.csv"):       # Report numera las tablas por línea: sin huérfanas
        f.unlink()
    rng = np.random.default_rng(SEED)
    D = r13.load()
    d = D["S&P 500"]
    var_fc = r13.var_forecast(d)
    ml = ml_forecasts(d, cache)
    ml_secs = ml.pop("_segundos", {})
    C = build_census(d, var_fc, ml)
    A = analyse(C, rng)
    R13, E13 = r13.strategies(d, var_fc)
    A["stale"] = primary_sensitivity(d, R13, E13, A["start"], A["end"])
    A["r13_check"] = r13.stats_table({k: R13[k] for k in (BENCH, "3x comprar y mantener", LRS,
                                                          "LRS-Kelly", "½ Kelly GJR (sin tendencia)")},
                                     d["rf"], var_fc.first_valid_index())
    A["names_all"] = [k for k in C.R if k != BENCH]
    A["dup_meta"] = C.meta
    t1, t2 = A["t1"], A["t2"]
    K, T = A["K"], A["T"]
    ne1, ne2 = A["ne"]["ne1"], A["ne"]["ne2"]
    V1, V2, V0 = A["V"]["V1"], A["V"]["V2"], A["V"]["V0"]
    op1, op2 = A["operables"]["1x"], A["operables"]["LRS"]
    best1 = t1["IR"].idxmax()
    same_close = [k for k in t1.index if is_same_close(k)]
    best1n = t1.loc[op1, "IR"].idxmax()
    best2 = t2["IR"].idxmax()
    bestg1 = t1["Δg"].idxmax()
    years = T / 252
    ep1, ep2 = A["epochs"]["1x"], A["epochs"]["LRS"]
    cs = A["cscv"]
    pub = published_table()
    ef = A["epochs_fam"]
    fam_g = "R13: LRS 3x media × banda"
    if fam_g in ef.index:
        A["epochs_msg"] = (f"Dentro de la rejilla media × banda de R13 ({int(ef.loc[fam_g, 'pruebas'])} "
                           f"reglas), Spearman {num(float(ef.loc[fam_g, 'Spearman_IR']))}: la mejor de "
                           f"1939–1991 («{ef.loc[fam_g, 'mejor_1939_1991']}») queda "
                           f"{ef.loc[fam_g, 'su_puesto_1992_2026']} en 1992–2026.")
    else:
        A["epochs_msg"] = ""

    # MinTRL del contraste con pruebas múltiples: años para que el p (iid, con asimetría y
    # curtosis) quede bajo el nivel de Šidák por prueba, 1 − 0,95^(1/N_eff)
    a_sid1 = 1 - (1 - ALPHA) ** (1 / ne1)
    t1["MinTRL_años_Šidák_Neff"] = min_trl(t1["IR"].values / np.sqrt(252), 0.0, t1["asimetría"].values,
                                           t1["curtosis"].values, alpha=a_sid1) / 252
    t1["N_Šidák_5%"] = sidak_n_star(t1["p_boot"].values)
    lim_obs1 = float(z_max(ne1) * np.sqrt(max(V1 - V0, 0)) * np.sqrt(252))
    lim_obs2 = float(z_max(ne2) * np.sqrt(max(V2 - V0, 0)) * np.sqrt(252))
    # sensibilidad: con reglas correlacionadas el ruido en la dispersión entre pruebas es (1 − ρ̄)/T
    vd_rho = max(V1 - (1 - A["rho_bar"]["1x"]) / T, 0.0)
    A["lim_rho"] = float(z_max(ne1) * np.sqrt(vd_rho) * np.sqrt(252))
    best_op = A["best_op"]
    px_row = t1.loc[PX_LRS]
    rk = A["rank_lrs"]
    for tag, k in (("lrs", LRS), ("rot", best1n)):
        A[f"mintrl_rho_{tag}"] = min_trl_deflated(t1.loc[k, "IR"] / np.sqrt(252), t1.loc[k, "asimetría"],
                                                  t1.loc[k, "curtosis"], float(z_max(ne1)), vd_rho) / 252

    # ── resumen ──
    rep.p(__doc__.split("\n", 2)[2].split("\nMedidas")[0])
    lrs_row, kel_row, hk_row = t1.loc[LRS], t1.loc["LRS-Kelly"], t1.loc["½ Kelly GJR (sin tendencia)"]
    rot = t1.loc[best1n]
    sr0_1 = float(expected_max_sr(V1, ne1)) * np.sqrt(252)
    sr0_0 = float(expected_max_sr(V0, ne1)) * np.sqrt(252)
    n_holm1 = int((t1.p_Holm < ALPHA).sum())
    n_bhy1 = int((t1.q_BHY < ALPHA).sum())
    n_bh1 = int((t1.q_BH < ALPHA).sum())
    n_p1 = int((t1.p_boot < ALPHA).sum())
    n_holm2 = int((t2.p_Holm < ALPHA).sum())
    n_bhy2 = int((t2.q_BHY < ALPHA).sum())
    n_p2 = int((t2.p_boot < ALPHA).sum())
    rw1 = list(t1.index[t1.p_RW < ALPHA][np.argsort(t1.p_RW[t1.p_RW < ALPHA].values, kind="stable")])
    rw1op = list(t1.index[t1.p_RW_operables < ALPHA])
    sid1 = list(t1.index[t1["p_Šidák_Neff"] < ALPHA])
    sid1op = [k for k in sid1 if k in op1]
    n_rw2 = int((t2.p_RW < ALPHA).sum())
    n_sid2 = int((t2["p_Šidák_Neff"] < ALPHA).sum())
    spa1, spa2 = A["spa"]["frente al 1x"], A["spa"]["frente a la LRS 3x original"]
    spa1o, spa2o = A["spa_op"]["frente al 1x"], A["spa_op"]["frente a la LRS 3x original"]
    ne_rng1 = A["neff"]["excesos sobre el 1x"].iloc[:5]
    ne_rng2 = A["neff"]["alternativas: excesos sobre la LRS 3x"].iloc[:5]
    pbo_all = cs["todo el censo · IR frente al 1x"]
    pbo_lrs = cs["parametrizaciones de la LRS 3x · IR frente al 1x"]
    pbo_alt = cs["alternativas · IR frente a la LRS 3x original"]
    post, (p0, p1) = A["post"], A["post_span"]
    size1 = float(dsr_rule_size(ne1))
    edge = [k for k in t1.index if 0.04 <= t1.loc[k, "p_RW"] <= 0.06]
    edge_txt = "; ".join(f"«{label(k)}» ({fpc(t1.loc[k, 'p_RW'])})" for k in edge) or "ninguna"
    edge_in = [k for k in edge if t1.loc[k, "p_RW"] < ALPHA]
    edge_sum = ((f"; la última" if len(edge_in) == 1 else f"; las {len(edge_in)} últimas") + ", con p = " +
                ", ".join(fpc(t1.loc[k, 'p_RW']) for k in edge_in) +
                ", en el filo del error de Monte Carlo (±0,005)") if edge_in else ""
    st_tab = A["stale"]["tabla"]

    def stale_row(tag: str) -> str:
        r = st_tab.loc[("LRS 3x frente al 1x", tag)]
        return f"{pp(r['Δg (pp)'] / 100)}/año y {p_edge_text(r['p_boot'])}"

    rep.h("Resumen")
    lines = [
        f"- **Censo.** Se reconstruyeron {A['K_bruto']} series; {K} son reglas distintas (el resto, la "
        f"misma regla construida en dos o más módulos). Ventana común del S&P 500: "
        f"{A['start'].date()} → {A['end'].date()} ({num(years, 1)} años, {T} sesiones), orden al cierre "
        f"siguiente. El 1x rinde {pc(A['bench_desc']['CAGR'])} al año en ella.",
        f"- **Pruebas efectivas.** Las {K} reglas equivalen a entre {ne_rng1.min():.0f} y "
        f"{ne_rng1.max():.0f} pruebas independientes según el estimador (principal, Li y Ji: "
        f"**{ne1:.0f}**) con los excesos sobre el 1x; las {K - 1} alternativas a la LRS, frente a ella, "
        f"a entre {ne_rng2.min():.0f} y {ne_rng2.max():.0f} (Li y Ji: **{ne2:.0f}**).",
        f"- **Dos contrastes distintos.** La LRS 3x original no salió de este censo: la fijaron Gayed y "
        f"Bilello (2016) y es la hipótesis principal, así que se contrasta sola. La búsqueda del censo "
        f"se descuenta a la regla que el censo elige, la mejor. Contrastar primero la principal al 5% y, "
        f"sólo si pasa, el resto con una corrección familiar al 5% mantiene el error familiar total en "
        f"el 5% (procedimiento de secuencia fija).",
        f"- **LRS 3x original (preespecificada) frente al 1x.** La regla de Gayed y Bilello (media de 200 "
        f"sesiones del índice de rentabilidad total, orden al cierre siguiente): IR {num(lrs_row.IR)} anual, "
        f"Δg {pp(lrs_row['Δg'])}/año, {peq(lrs_row.p_boot)} (bootstrap), PSR {num(lrs_row.PSR, 3)}: "
        + ("pasa al 5% sola. " if lrs_row.p_boot < ALPHA else "no pasa al 5% ni sola. ")
        + f"Con una salvedad seria: su propia búsqueda es desconocida y su muestra (1928–2015) "
        f"cubre el {pc(A['gb_overlap'], 0)} de esta ventana, así que este p no es confirmatorio. El único "
        f"tramo limpio es el posterior ({p0.date()} → {p1.date()}, {A['post_n']} sesiones): Δg "
        f"{pp(post.loc[LRS, 'Δg'])}/año, IR {num(post.loc[LRS, 'IR'])}, p = {fpc(post.loc[LRS, 'p_boot'])} "
        f"(CAGR {pc(post.loc[LRS, 'CAGR'])} frente a {pc(post.loc[LRS, 'CAGR 1x'])} del 1x), con sólo "
        f"{num(A['post_n'] / 252, 1)} años. Si se la tratara como una regla más del censo, su p ajustado "
        f"sería {fpc(lrs_row['p_Šidák_Neff'])} (Šidák, N_eff = {ne1:.0f}) o {fpc(lrs_row.p_RW)} "
        f"(Romano–Wolf; {fpc(lrs_row.p_RW_operables)} solo con las operables), pero ése no es su caso. Es "
        f"la {rk[LRS][0]}.ª de las {rk[LRS][1]} parametrizaciones de la LRS 3x del censo por IR en toda la "
        "ventana. Descontando dos sesgos a favor del apalancamiento anteriores a los ETFs (sección 5, "
        "añadido tras una revisión externa): los precios desfasados anteriores a 1990, Δg "
        f"{stale_row('precios desfasados (ρ₁)')}; el sobrecoste del swap calibrado frente al SPY y no "
        f"frente al índice, {stale_row('sobrecoste frente al índice')}; los dos, "
        f"{stale_row('las dos (ρ₁ y sobrecoste)')}.",
        f"- **La señal de precio, una regla más del censo.** La LRS 3x con la media sobre el índice de precio "
        f"es una variante del laboratorio, no la de los autores: IR {num(px_row.IR)}, Δg "
        f"{pp(px_row['Δg'])}/año, {peq(px_row.p_boot)} sin corregir. Descontada la búsqueda: "
        f"{fpc(px_row['p_Šidák_Neff'])} con Šidák a N_eff, {fpc(px_row.p_RW)} con Romano–Wolf "
        f"({fpc(px_row.p_RW_operables)} solo con las operables): "
        + ("**sobrevive a la búsqueda al 5%**. " if max(px_row.p_RW, px_row['p_Šidák_Neff']) < ALPHA else
           "**no sobrevive a la búsqueda al 5%**. ")
        + f"Es la {rk[PX_LRS][0]}.ª de las {rk[PX_LRS][1]} parametrizaciones de la LRS 3x por IR.",
        f"- **La mejor del censo con orden al cierre siguiente** es «{best1n}» (IR {num(rot.IR)}, Δg "
        f"{pp(rot['Δg'])}/año, {peq(rot.p_boot)}). Descontada la búsqueda: p = "
        f"{fpc(rot['p_Šidák_Neff'])} con Šidák a N_eff = {ne1:.0f}, {fpc(rot.p_RW)} con el step-down de "
        f"Romano–Wolf sobre las {K} reglas ({fpc(rot.p_RW_operables)} sólo con las {len(op1)} operables) "
        f"y SPA de Hansen p = {fpc(spa1['p_SPA_c'])} ({fpc(spa1o['p_SPA_c'])} sin las de mismo cierre). "
        + ("**Sobrevive a la búsqueda al 5%.** " if rot.p_RW < ALPHA and rot['p_Šidák_Neff'] < ALPHA else
           "**No sobrevive a la búsqueda con todas las correcciones.** ")
        + f"Con Šidák sobrevive mientras el censo equivalga a menos de {rot['N_Šidák_5%']:.0f} pruebas "
        f"independientes. La mejor de todas por IR es «{best1}» (IR {num(t1.loc[best1, 'IR'])}) y por Δg "
        f"«{bestg1}» ({pp(t1.loc[bestg1, 'Δg'])}/año)"
        + (", las dos en el mismo cierre, que antes de 1990 no era operable (R13, R15)."
           if best1 in same_close and bestg1 in same_close else ".")
        + f" Como vara de medir, un 2x constante sin ninguna sincronización tiene IR "
        f"{num(A['static'].loc['L constante 2', 'IR'])} frente al 1x.",
        f"- **Titular «descontada la búsqueda», solo con reglas operables.** Para la mejor operable "
        f"(«{label(best1n)}»): SPA operable {fpc(spa1o['p_SPA_c'])}, Romano–Wolf operable "
        f"{fpc(rot.p_RW_operables)}, Šidák a N_eff {fpc(rot['p_Šidák_Neff'])}; con las reglas de mismo "
        f"cierre, que antes de 1990 no se podían seguir, SPA {fpc(spa1['p_SPA_c'])} y Romano–Wolf "
        f"{fpc(rot.p_RW)}. Si además se cuentan las {N_R13_SPA} comparaciones del SPA de R13 (3 mercados × "
        f"3 referencias), la cota de Holm (Bonferroni) del Romano–Wolf operable es "
        f"{fpc(min(1.0, N_R13_SPA * rot.p_RW_operables))}.",
        f"- **Qué reglas sobreviven a la búsqueda (5%, frente al 1x).** Romano–Wolf: {len(rw1)} de {K} "
        f"({names_list(rw1)}){edge_sum}; con sólo las operables, {len(rw1op)} ({names_list(rw1op)}). "
        f"Šidák a N_eff: {len(sid1)} ({names_list(sid1)}). Holm, BH y BHY sobre las {K} no rechazan "
        f"ninguna ({n_holm1}, {n_bh1} y {n_bhy1}): tratan las {K} pruebas como si fueran casi "
        f"independientes. Frente a la LRS 3x original no sobrevive ninguna alternativa: {n_p2} de "
        f"{K - 1} con p < 5% sin corregir, {n_rw2} con Romano–Wolf; SPA p = {fpc(spa2['p_SPA_c'])}.",
        f"- **El DSR es descriptivo.** Compara el IR con el máximo esperado de N pruebas sin habilidad "
        f"más 1,645 errores típicos de un solo IR, no con el cuantil 95% del máximo. «DSR ≥ 0,95» es "
        f"por eso mucho más exigente que un contraste al 5%: bajo la nula, con N = {ne1:.0f} pruebas "
        f"independientes, el máximo lo alcanza sólo con probabilidad {pc(size1)}. Con N = {ne1:.0f}, "
        f"SR₀ = {num(sr0_1)} (dispersión observada) o {num(sr0_0)} (azar); el DSR de «{label(best1n)}» es "
        f"{num(rot.DSR, 3)} y {num(rot.DSR_V_azar, 3)}; el de la LRS 3x, que no es la elegida, sería "
        f"{num(lrs_row.DSR, 3)} y {num(lrs_row.DSR_V_azar, 3)}.",
        f"- **LRS-Kelly** (IR {num(kel_row.IR)}, p = {fpc(kel_row.p_boot)}) y **½ Kelly GJR** (IR "
        f"{num(hk_row.IR)}, p = {fpc(hk_row.p_boot)}) no superan al 1x ni sin corregir.",
        f"- **Tiempo mínimo de historia (95%) y potencia.** Para afirmar que la LRS 3x original supera al "
        f"1x (IR* = 0) harían falta {fyc(lrs_row['MinTRL_años_IR*=0'])} años con su IR, y "
        f"{fyc(lrs_row['años_80%_potencia'])} para detectarlo con un 80% de probabilidad (con T = MinTRL, la "
        f"potencia es del 50% por construcción); la muestra tiene {years:.0f}. La señal de precio: "
        f"{fyc(px_row['MinTRL_años_IR*=0'])} y {fyc(px_row['años_80%_potencia'])}; «{label(best1n)}»: "
        f"{fyc(rot['MinTRL_años_IR*=0'])} y {fyc(rot['años_80%_potencia'])}. Para superar la búsqueda al 5% "
        f"(Šidák a N_eff), con la aproximación iid del PSR: «{label(best1n)}», "
        f"{fyc(rot['MinTRL_años_Šidák_Neff'])} años "
        f"({'los tiene' if rot['MinTRL_años_Šidák_Neff'] < years else 'no los tiene'}); la señal de precio, "
        f"{fyc(px_row['MinTRL_años_Šidák_Neff'])} "
        f"({'los tiene' if px_row['MinTRL_años_Šidák_Neff'] < years else 'no los tiene'}); la LRS 3x "
        f"original, {fyc(lrs_row['MinTRL_años_Šidák_Neff'])} (como hipótesis preespecificada no necesita ese "
        f"descuento). Con el criterio del DSR, más severo, y el umbral SR₀(n), que baja al alargarse la "
        f"historia: «{label(best1n)}» {fyc(rot['MinTRL_años_SR0(n)_V_azar'])} años con la dispersión del azar "
        f"y {fyc(rot['MinTRL_años_SR0(n)_V_obs'])} con la observada; señal de precio "
        f"{fyc(px_row['MinTRL_años_SR0(n)_V_azar'])} y {fyc(px_row['MinTRL_años_SR0(n)_V_obs'])}; LRS 3x "
        f"original {fyc(lrs_row['MinTRL_años_SR0(n)_V_azar'])} y {fyc(lrs_row['MinTRL_años_SR0(n)_V_obs'])}.",
        f"- **p-valores de titular de R16–R18** ({len(pub)}, leídos de sus p_titular.csv): "
        f"{int((pub.p < ALPHA).sum())} por debajo "
        f"del 5%; {int((pub.p_Holm < ALPHA).sum())} con Holm, {int((pub.q_BH < ALPHA).sum())} con BH y "
        f"{int((pub.q_BHY < ALPHA).sum())} con BHY.",
        f"- **Sobreajuste de la selección (CSCV, 16 bloques).** PBO {pc(pbo_all['pbo'])} al elegir por IR "
        f"entre todo el censo, {pc(pbo_lrs['pbo'])} entre las {pbo_lrs['n_pruebas']} parametrizaciones "
        f"de la LRS 3x y {pc(pbo_alt['pbo'])} entre las alternativas frente a la LRS original. Una PBO baja "
        f"no salva a las alternativas: la mejor dentro de muestra pierde contra la LRS original fuera de "
        f"muestra en el {pc(pbo_alt['p_pérdida'])} de las particiones. La comparación favorecería a la "
        f"original si fuera, a posteriori, de las mejores parametrizaciones; es la {rk[LRS][0]}.ª de "
        f"{rk[LRS][1]} por IR en toda la ventana (la señal de precio, la {rk[PX_LRS][0]}.ª).",
        f"- **Épocas.** El orden por IR frente al 1x de 1939–1991 se conserva en 1992–2026 con Spearman "
        f"{num(ep1['rho_ir'])} (p = {fpc(ep1['p_null'])} frente a una época 2 sin diferencias reales; su "
        f"percentil 95 es {num(ep1['null_p95'])}); del quintil superior sigue en él el "
        f"{pc(ep1['top_ir'], 0)}. Buena parte es estructural: el IR frente al 1x premia el apalancamiento "
        f"medio en un mercado que sube en las dos épocas. Frente a la LRS original, Spearman "
        f"{num(ep2['rho_ir'])} (p = {fpc(ep2['p_null'])}). {A['epochs_msg']}",
    ]
    rep.p("\n".join(lines))
    rep.p("Con la base actual, R13 da desde 1938: " +
          ", ".join(f"{k} {pc(A['r13_check'].loc[k, 'CAGR'])}" for k in A["r13_check"].index) +
          ". La ventana común de este censo empieza más tarde (cuando los modelos de ML de R12 "
          "tienen diez años de datos), de ahí las pequeñas diferencias con las tablas siguientes.")

    rep.h("Protocolo")
    rep.p("Medidas" + __doc__.split("\nMedidas", 1)[1])
    rep.p(f"Tiempo de los modelos de ML de R12 (s): " +
          ", ".join(f"{k} {v:.0f}" for k, v in ml_secs.items()) + ".")

    # ── 1. censo ──
    rep.h("1. El censo")
    rep.table(census_table(A), index=False, digits=0)
    rep.p("«construidas» son las series que se reconstruyeron para cada módulo; «únicas», las que no "
          "repiten exactamente (en la ventana común) una serie ya contada. Duplicados más frecuentes: " +
          "; ".join(f"«{k}» = {len(v) + 1} series" for k, v in
                    sorted(A["dup"].items(), key=lambda kv: -len(kv[1]))[:4]) + ".")
    rep.p("Fuera del censo: las versiones de datos de R14 (mismas reglas, otros datos), las carteras "
          "de L constante de R16 y R17 (referencias fijadas con la exposición o la volatilidad de toda "
          "la muestra), Moreira–Muir con c de muestra completa (mira el futuro), los placebos y "
          "mundos sintéticos (R16–R18), el Nasdaq-100 y los semiconductores, y los análisis de R1–R9 "
          "que no son reglas de inversión. Todo eso también es búsqueda: el N de esta página es una "
          "cota inferior del número de caminos que ha recorrido el laboratorio.")
    rep.h("Rendimiento del censo (resumen)", 3)
    dsum = A["desc"].describe(percentiles=[0.1, 0.5, 0.9]).T[["min", "10%", "50%", "90%", "max"]]
    dsum.loc["IR frente al 1x"] = t1["IR"].describe(percentiles=[0.1, 0.5, 0.9])[["min", "10%", "50%", "90%", "max"]].values
    dsum.loc["Δg frente al 1x"] = t1["Δg"].describe(percentiles=[0.1, 0.5, 0.9])[["min", "10%", "50%", "90%", "max"]].values
    rep.table(dsum, digits=3)

    # ── 2. N efectivo ──
    rep.h("2. Número efectivo de pruebas")
    rep.table(A["neff"], digits=1)
    ev = A["eig"]["excesos sobre el 1x"]
    rep.p(f"Con los excesos sobre el 1x, el primer autovalor recoge el {pc(ev[0] / ev.sum())} de la "
          f"varianza y los cinco primeros el {pc(ev[:5].sum() / ev.sum())}. Con las rentabilidades en "
          f"bruto el mercado domina aún más (primer autovalor {pc(A['eig']['rentabilidades'][0] / K)}), "
          "por eso ese N_eff es menor: dos reglas que siguen al mismo mercado parecen redundantes aunque "
          "sus apuestas frente al 1x sean distintas. Para las correcciones importa la correlación del "
          "estadístico que se selecciona (el exceso sobre el 1x o sobre la LRS), no la del mercado. Li y "
          "Ji (2005), fijado como principal, da el número de contrastes independientes de una corrección "
          "de Šidák, que se aplica en la sección 5; el cociente de participación y la entropía miden "
          "dimensión y dan cifras menores. El step-down de Romano–Wolf no necesita N_eff: usa la "
          "dependencia real a través de los remuestreos comunes.")
    fig_spectrum(A, rep)

    # ── 3. DSR ──
    rep.h("3. Deflated Sharpe Ratio (descriptivo)")
    rep.p(f"Estadístico: IR diario de a_t = ln(1+r_t) − ln(1+u_t). T = {T} sesiones. V observada = "
          f"varianza de los IR diarios de las {K} pruebas ({V1:.2e}; en IR anual, desviación típica "
          f"{num(np.sqrt(V1 * 252))}); V del azar = 1/T ({V0:.2e}; {num(np.sqrt(V0 * 252))} anual). "
          f"Con V observada, SR₀ incluye la dispersión real entre reglas distintas (unas son 3x y otras "
          f"1x): exagera el ruido y es la versión severa; con 1/T supone que todas las diferencias son "
          f"ruido de muestreo de reglas independientes y es la benévola. El DSR «T efectivo» (en el CSV) "
          f"corrige además la autocorrelación de los excesos (T·γ₀/varianza de largo plazo, Bartlett con "
          f"63 retardos) y apenas cambia nada: LRS 3x {num(lrs_row.DSR_T_efectivo, 3)} frente a "
          f"{num(lrs_row.DSR, 3)}.")
    rep.h("Por qué «DSR ≥ 0,95» no es un contraste al 5%", 3)
    marks_s = sorted(set([2, 5, 10, int(round(ne1)), int(round(ne2)), 100, K, 1000]))
    sz = pd.DataFrame({
        "N": marks_s,
        "E[máx] (errores típicos)": z_max(np.array(marks_s)),
        "umbral del DSR: E[máx] + 1,645": z_max(np.array(marks_s)) + norm.ppf(0.95),
        "umbral al 5% familiar: cuantil 95% del máx": max_quantile(np.array(marks_s)),
        "P(DSR del máx ≥ 0,95) bajo la nula": dsr_rule_size(np.array(marks_s))}).set_index("N")
    rep.table(sz, pct={"P(DSR del máx ≥ 0,95) bajo la nula"}, digits=2)
    rep.p("N pruebas independientes sin habilidad, con IR estimados normales. El DSR es el PSR frente al "
          "máximo ESPERADO de N IR sin habilidad: exige que el IR supere ese máximo esperado en 1,645 "
          "errores típicos de UN IR. Pero el máximo de N normales se concentra (su dispersión es menor "
          "que la de una sola), así que su cuantil 95%, que es el umbral de un contraste con un 5% de "
          "error familiar, queda muy por debajo. La última columna es la probabilidad, bajo la nula "
          "global, de que la mejor de las N alcance DSR ≥ 0,95: el tamaño real de esa regla. El DSR se da "
          "aquí como medida descriptiva (cuánto supera el IR a lo que la búsqueda daría por azar); el "
          "contraste al 5% con pruebas múltiples está en la sección 5. Con estas fórmulas cerradas "
          "coincide una simulación con T = 21.992 y 200.000 réplicas (tests del módulo).")
    rep.h("Máximo esperado del IR anual sin habilidad, SR₀, según N", 3)
    marks = sorted(set([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, int(round(ne1)), int(round(ne2)), K]))
    s0 = pd.DataFrame({"N": marks,
                       "SR₀ con V observada (1x)": expected_max_sr(V1, marks) * np.sqrt(252),
                       "SR₀ con V observada (LRS)": expected_max_sr(V2, marks) * np.sqrt(252),
                       "SR₀ con V = 1/T": expected_max_sr(V0, marks) * np.sqrt(252)}).set_index("N")
    rep.table(s0, digits=3)
    rep.h("Referencia: apalancamiento constante, sin ninguna sincronización", 3)
    rep.table(A["static"], pct={"CAGR", "Δg"}, digits=3)
    rep.p("No son pruebas del censo sino la vara de medir: una fracción fija L/3 en el 3x y el resto en "
          "letras, reequilibrada a diario. Frente al 1x, cualquier exposición media mayor que 1 gana IR "
          "positivo en un mercado con prima de riesgo positiva, sin habilidad alguna para elegir el "
          "momento. Las correcciones frente al 1x responden a «¿supera al 1x después de la búsqueda?», no "
          "a «¿tiene habilidad de sincronización?» (esa pregunta es la de R16–R18).")
    rep.h(f"Frente al 1x (N_eff = {ne1:.0f}; N total = {K})", 3)
    top1 = list(t1.sort_values("IR", ascending=False).index[:8])
    rows1 = focus_rows(A, t1, top1)
    rep.table(show_trials(t1, rows1, A, "1x"), pct={"CAGR", "Δg"}, digits=3)
    rep.p("El DSR de cada fila responde a «¿cuánto supera el IR al máximo esperado de N pruebas sin "
          "habilidad, si ésta hubiera sido la elegida?». Sólo tiene sentido para la regla que el censo "
          "selecciona (la primera por IR); para la LRS 3x original, fijada antes del censo, se da como "
          "referencia. p Šidák (N_eff) = 1 − (1 − p)^N_eff; p Romano–Wolf: step-down sobre las "
          f"{K} reglas con los mismos {B_BOOT} remuestreos (error de Monte Carlo ±0,004 hacia p = 0,03); "
          f"p_Holm, sobre las {K} sin usar su dependencia.")
    rep.h(f"Alternativas frente a la LRS 3x original (N_eff = {ne2:.0f}; N total = {K - 1})", 3)
    top2 = list(t2.sort_values("IR", ascending=False).index[:8])
    rows2 = [k for k in (PX_LRS, ROT, "LRS-Kelly", "LRS 3x + banda ±2%", "LRS-Kelly suavizado")
             if k in t2.index] + top2
    rows2 = list(dict.fromkeys(rows2))
    rep.table(show_trials(t2, rows2, A, "LRS"), pct={"CAGR", "Δg"}, digits=3)
    picks1 = list(dict.fromkeys([LRS, PX_LRS, "LRS-Kelly", best1n]))
    picks2 = list(dict.fromkeys([best2, ROT, "LRS-Kelly"]))
    fig_dsr(A, rep, picks1[:4], picks2[:3])
    rep.p("N a partir del cual el p de Šidák frente al 1x supera el 5% (la regla deja de sobrevivir a la "
          "búsqueda si el censo equivale a más pruebas independientes): " +
          "; ".join(f"{label(k)}: " + (num(t1.loc[k, 'N_Šidák_5%'], 1) if t1.loc[k, 'N_Šidák_5%'] >= 1 else
                                       "ninguno (p > 5% ya sin corregir)") for k in picks1) +
          f". El N_eff principal es {ne1:.0f} y el rango de estimadores, {ne_rng1.min():.0f}–"
          f"{ne_rng1.max():.0f}: la conclusión sobre «{label(best1n)}» depende de que el censo no equivalga a "
          f"más de unas {t1.loc[best1n, 'N_Šidák_5%']:.0f} pruebas independientes (Nyholt da "
          f"{A['neff'].loc['Nyholt (2004)', 'excesos sobre el 1x']:.0f}); Romano–Wolf, que no necesita "
          f"elegir N, da {fpc(t1.loc[best1n, 'p_RW'])}.")

    # ── 4. MinTRL ──
    rep.h("4. Longitud mínima de la historia (MinTRL, 95%)")
    mt = pd.DataFrame({
        "IR anual": t1.loc[rows1, "IR"].values,
        "años para IR* = 0": [fyears(x) for x in t1.loc[rows1, "MinTRL_años_IR*=0"]],
        "años para 80% de potencia": [fyears(x) for x in t1.loc[rows1, "años_80%_potencia"]],
        "años para superar la búsqueda al 5% (Šidák, N_eff)":
            [fyears(x) for x in t1.loc[rows1, "MinTRL_años_Šidák_Neff"]],
        "años para IR* = SR₀(n), V azar": [fyears(x) for x in t1.loc[rows1, "MinTRL_años_SR0(n)_V_azar"]],
        "años para IR* = SR₀(n), V obs.": [fyears(x) for x in t1.loc[rows1, "MinTRL_años_SR0(n)_V_obs"]],
        "asimetría": t1.loc[rows1, "asimetría"].values, "curtosis": t1.loc[rows1, "curtosis"].values,
    }, index=[short(k, 60) for k in rows1])
    rep.table(mt, digits=2)
    rep.p(f"Frente al 1x, con {years:.0f} años disponibles. MinTRL = 1 + (1 − γ₃·IR + (γ₄ − 1)/4·IR²)·"
          f"(z/(IR − IR*))², con IR diario, γ₃ y γ₄ la asimetría y la curtosis de los excesos diarios. "
          f"Con una historia de MinTRL años, el PSR del IR observado es el 95% exacto y la potencia de "
          f"detectar un IR verdadero igual al observado es del 50%: «años para 80% de potencia» usa "
          f"(z₀,₉₅ + z₀,₈₀)² en lugar de z₀,₉₅². "
          f"«Superar la búsqueda al 5%»: IR* = 0 con z al nivel de Šidák por prueba, "
          f"1 − 0,95^(1/{ne1:.0f}) (z = {num(float(max_quantile(ne1)))} en lugar de 1,645). «SR₀(n)»: el "
          f"criterio del DSR, con el umbral que depende de la propia longitud n de la historia, "
          f"SR₀(n) = z_N·√(V_disp + 1/n), z_{ne1:.0f} = {num(float(z_max(ne1)))}: con la dispersión del "
          f"azar V_disp = 0 y el umbral tiende a cero; con la observada, V_disp = V − 1/T (el "
          f"{pc(1 - V0 / V1, 0)} de V no es ruido de muestreo y no encoge) y el umbral tiende a "
          f"{num(lim_obs1, 3)} en IR anual. ∞: el IR no supera ese límite. Esta columna hereda la severidad "
          f"del DSR (sección 3) y exige más que la de Šidák. La curtosis diaria de los excesos es alta "
          f"(el 19-10-1987 pesa mucho), pero con IR diarios del orden de 0,02 su término apenas cambia el "
          f"resultado: lo que manda es (z/IR)².")
    mt2 = pd.DataFrame({
        "IR anual frente a la LRS": t2.loc[rows2, "IR"].values,
        "años para IR* = 0": [fyears(x) for x in t2.loc[rows2, "MinTRL_años_IR*=0"]],
        "años para IR* = SR₀(n), V azar": [fyears(x) for x in t2.loc[rows2, "MinTRL_años_SR0(n)_V_azar"]],
    }, index=[short(k, 60) for k in rows2])
    if V2 > V0:
        mt2["años para IR* = SR₀(n), V obs."] = [fyears(x) for x in t2.loc[rows2, "MinTRL_años_SR0(n)_V_obs"]]
    rep.h("Alternativas frente a la LRS 3x original", 3)
    rep.table(mt2, digits=2)
    rep.p((f"Con V observada el umbral tiende a {num(lim_obs2, 3)} en IR anual frente a la LRS."
           if V2 > V0 else
           f"Frente a la LRS, la varianza observada de los IR ({V2:.2e}) es menor que la del azar para "
           f"pruebas independientes (1/T = {V0:.2e}): las alternativas están tan correlacionadas con la "
           f"LRS que su ruido de muestreo común no se ve en la dispersión entre ellas. V_disp = 0 y las "
           f"dos versiones coinciden.") +
          " Ninguna alternativa se acerca a una historia disponible.")

    # ── 5. p-valores ──
    rep.h("5. p-valores con pruebas múltiples: Holm, BHY, Šidák y Romano–Wolf")
    both = pd.concat([t1[["p_boot", "p_empírico"]], t2[["p_boot", "p_empírico"]]])
    both = both[both["p_empírico"] >= 0.01]
    p_gap = float((both["p_boot"] - both["p_empírico"]).abs().max())
    pt = pd.DataFrame({
        "pruebas": [K, K - 1],
        "p < 5% sin corregir": [n_p1, n_p2],
        "esperadas por azar (5%)": [0.05 * K, 0.05 * (K - 1)],
        "Holm < 5%": [n_holm1, n_holm2],
        "BH < 5%": [n_bh1, int((t2.q_BH < ALPHA).sum())],
        "BHY < 5%": [n_bhy1, n_bhy2],
        "Šidák (N_eff) < 5%": [len(sid1), n_sid2],
        "Romano–Wolf < 5%": [len(rw1), n_rw2],
        "Romano–Wolf, operables < 5%": [len(rw1op), int((t2.p_RW_operables < ALPHA).sum())],
        "p mínimo": [fp(t1.p_boot.min()), fp(t2.p_boot.min())],
        "p_SPA_c": [fp(spa1["p_SPA_c"]), fp(spa2["p_SPA_c"])],
        "p_SPA_c, operables": [fp(spa1o["p_SPA_c"]), fp(spa2o["p_SPA_c"])],
    }, index=["frente al 1x", "alternativas frente a la LRS 3x original"])
    rep.table(pt, digits=1)
    rep.p(f"p unilateral (H0: Δg ≤ 0) = 1 − Φ(Δg/EE), con EE el error estándar del bootstrap estacionario "
          f"de las sesiones (bloque medio {BLOCK}, {B_BOOT} remuestreos comunes a todas las pruebas). El p "
          f"empírico del bootstrap (en el CSV) no baja de 1/{B_BOOT + 1} y con {K} pruebas Holm no podría "
          f"rechazar nada; donde el empírico vale 0,01 o más, los dos difieren como mucho en "
          f"{num(p_gap, 3)}. Holm controla la probabilidad de algún falso positivo con cualquier "
          f"dependencia, pero sin usarla: con {K} pruebas que equivalen a unas {ne1:.0f} es muy "
          f"conservador. BH controla la proporción esperada de falsos descubrimientos con dependencia "
          f"positiva; BHY (Benjamini y Yekutieli 2001) con cualquier dependencia, multiplicando por "
          f"c(m) = Σ1/i ({num(np.sum(1 / np.arange(1, K + 1)))} con {K} pruebas). Šidák a N_eff corrige "
          f"con el número efectivo de pruebas de la sección 2. El step-down de Romano y Wolf (2005) "
          f"compara cada t = Δg/EE con el máximo de los t bootstrap centrados de las reglas que quedan "
          f"(mismos remuestreos, que conservan la dependencia) y controla el error familiar en sentido "
          f"fuerte; su primer paso es el Reality Check de White estudentizado. El SPA (Hansen 2005, "
          f"{B_SPA} remuestreos) contrasta si la mejor regla supera a la referencia recentrando las "
          f"reglas claramente peores, por eso es algo más potente. «Operables»: sin las "
          f"{K - len(op1)} reglas ejecutadas en el mismo cierre de la señal, que antes de 1990 no se "
          f"podían seguir (sensibilidad añadida tras la revisión; el SPA operable usa otro generador).")
    rep.p(f"**Qué sobrevive.** Frente al 1x, con Romano–Wolf al 5%: {names_list(rw1)}; sólo con las "
          f"operables: {names_list(rw1op)}. Con Šidák a N_eff: {names_list(sid1)}. La LRS 3x original "
          + ("estaría entre ellas" if lrs_row.p_RW < ALPHA else "no estaría entre ellas")
          + f" si se la tratara como una regla más (Romano–Wolf {fpc(lrs_row.p_RW)}, "
          f"Šidák {fpc(lrs_row['p_Šidák_Neff'])}), pero es la hipótesis preespecificada y se contrasta "
          f"sola (siguiente apartado). La LRS 3x con la señal de precio, que sí es una variante del "
          f"laboratorio, da Romano–Wolf {fpc(px_row.p_RW)} y Šidák {fpc(px_row['p_Šidák_Neff'])}. La mejor "
          f"operable del censo es «{label(best1n)}» (Romano–Wolf operable {fpc(rot.p_RW_operables)}, SPA "
          f"operable {fpc(spa1o['p_SPA_c'])}). En el filo (Romano–Wolf entre 0,04 y "
          f"0,06, error de Monte Carlo ±0,005): {edge_txt}.")
    rep.h("La LRS 3x original, hipótesis preespecificada", 3)
    po = post.copy()
    po["p_boot"] = [fp(x) for x in po["p_boot"]]
    full_row = pd.DataFrame({"CAGR": [A["desc"].loc[k, "CAGR"] for k in po.index],
                             "CAGR 1x": A["bench_desc"]["CAGR"],
                             "Δg": t1.loc[po.index, "Δg"].values, "IR": t1.loc[po.index, "IR"].values,
                             "PSR": t1.loc[po.index, "PSR"].values,
                             "p_boot": [fp(x) for x in t1.loc[po.index, "p_boot"]]},
                            index=po.index)
    po.index = [f"{k} · {p0.year}–{p1.year}" for k in po.index]
    full_row.index = [f"{k} · {A['start'].year}–{A['end'].year}" for k in full_row.index]
    rep.table(pd.concat([full_row, po]), pct={"CAGR", "CAGR 1x", "Δg"}, digits=3)
    rep.p(f"Gayed y Bilello (2016) fijaron la regla (3x sobre la media de 200 sesiones del índice de "
          f"rentabilidad total, letras debajo; su nota 15) con datos de 1928–2015; el laboratorio la "
          f"recalcula sin tocarla, con su protocolo de ejecución (orden al cierre siguiente). El censo del "
          f"laboratorio no puede deflactar una regla que no eligió: se contrasta sola, p = "
          f"{fpc(lrs_row.p_boot)}. Pero la búsqueda que la produjo es la de sus autores, desconocida, y "
          f"su muestra cubre el {pc(A['gb_overlap'], 0)} de esta ventana: el p en la ventana completa es "
          f"en su mayor parte dentro de muestra para ellos. El único tramo limpio empieza en 2016: "
          f"{num(A['post_n'] / 252, 1)} años con Δg {pp(post.loc[LRS, 'Δg'])}/año frente al 1x (IR "
          f"{num(post.loc[LRS, 'IR'])}) y p = {fpc(post.loc[LRS, 'p_boot'])} (el MinTRL frente a IR* = 0 es de "
          f"{fyc(lrs_row['MinTRL_años_IR*=0'])} años con el IR de toda la ventana). La señal de precio y "
          f"«{label(best_op)}» se muestran en el mismo tramo sólo como comparación: para ellas no es fuera de "
          f"muestra, porque el laboratorio las probó con datos que incluyen este tramo (bootstrap del tramo "
          f"con su propio generador).")
    rep.h("Sensibilidad: dos sesgos a favor del apalancamiento antes de los ETFs", 3)
    st_ = A["stale"]
    st_show = st_["tabla"].copy()
    # cuatro decimales cerca del 5%: con tres, 0,05006 se leía «0.05», justo tras «pasa al 5% sola»
    st_show["p_boot"] = [fp_edge(p) for p in st_show["p_boot"]]
    rep.table(st_show, digits=3)
    rep.p("Añadido tras una revisión externa. Ventana del censo "
          f"({st_['desde']} → {st_['hasta']}). «sesgo»: la parte de la ventaja que se descuenta (media anual "
          "en la ventana y por periodos); «Δg»: el crecimiento log anual frente a la referencia después de "
          "descontarla; p_boot: el mismo contraste unilateral que la hipótesis principal (bootstrap "
          f"estacionario, bloque {BLOCK}, B = {num_es(B_BOOT, 0)}, los mismos remuestreos para todas las filas), "
          "con su propio generador, así que el «sin corregir» puede diferir del p principal en el error "
          f"de Monte Carlo (≈ ±{num_es(mc_se(), 3)} cerca de 0,05); los p a menos de ese error del 5% se "
          "muestran con cuatro decimales y no deciden nada por sí solos. (1) Precios desfasados: hasta 1989 el índice diario tiene autocorrelación "
          "positiva (R15), la huella de cierres que ningún fondo podía negociar. Con la misma varianza a "
          "largo plazo, un fondo que se reajusta cada día con exposición L crece sobre esos cierres "
          "L(L − 1)·ρ·σ² al año más que sobre precios negociables (en primera aproximación); el índice "
          "(L = 1) no gana nada. Se descuenta Σ L_t(L_t − 1)·ρ_año·u_t², con la exposición aplicada de cada "
          "día y ρ la autocorrelación de orden 1 (o la suma de las de orden 1 y 2) de las rentabilidades "
          f"diarias de cada año natural, solo hasta 1989; frente al L̄ constante ({num(st_['L̄'])}x, la "
          "exposición media de la LRS en la ventana, reajustado a diario) se resta también la parte que "
          "infla al constante. Es una cota de orden de magnitud: la autocorrelación de un año mezcla "
          "precios desfasados con dependencia genuina. (2) Sobrecoste frente al índice: el sobrecoste del "
          "swap se calibra frente al ETF 1x (SPY), que ya descuenta su comisión y el efectivo de sus "
          f"dividendos, pero antes del ETF (hasta el {st_['etf0']}) se aplica al índice, que no los paga: "
          "la réplica del 3x sale 3δ al año más barata, con δ = "
          f"{num(st_['δ'] * 100, 3)} puntos, lo que ^SP500TR rinde de más que el SPY en la vida del 3x real. "
          "Se descuenta L_t·δ cada día anterior al ETF.")
    rep.h("La LRS 3x original y la de precio tratadas como reglas del censo", 3)
    cm = pd.DataFrame({k: {"IR": t1.loc[k, "IR"], "Δg": t1.loc[k, "Δg"], "p_boot": fp(t1.loc[k, "p_boot"]),
                           "p Romano–Wolf": fp(t1.loc[k, "p_RW"]),
                           "p Romano–Wolf operables": fp(t1.loc[k, "p_RW_operables"]),
                           "p Šidák (N_eff)": fp(t1.loc[k, "p_Šidák_Neff"]),
                           "puesto entre las parametrizaciones de la LRS 3x": f"{rk[k][0]} de {rk[k][1]}"}
                       for k in (PX_LRS, LRS)}).T
    rep.table(cm, pct={"Δg"}, digits=3)
    rep.p("La señal de precio es un miembro natural del censo: la probó y publicó el laboratorio (R13–R15), "
          "así que su p debe descontar la búsqueda. La original se da con las mismas correcciones solo como "
          "sensibilidad: como hipótesis preespecificada se contrasta sola (arriba).")
    rep.h("Las pruebas con menor p frente al 1x", 3)
    tp = t1.sort_values("p_boot").head(15)
    rep.table(pd.DataFrame({"Δg": tp["Δg"].values, "IR": tp["IR"].values,
                            "p_boot": [fp(p) for p in tp.p_boot],
                            "p Šidák (N_eff)": [fp(p) for p in tp["p_Šidák_Neff"]],
                            "p Romano–Wolf": [fp(p) for p in tp.p_RW],
                            "p RW operables": [fp(p) if np.isfinite(p) else "—" for p in tp.p_RW_operables],
                            "p_Holm": [fp(p) for p in tp.p_Holm],
                            "q_BH": [fp(p) for p in tp.q_BH], "q_BHY": [fp(p) for p in tp.q_BHY]},
                           index=[short(k, 60) for k in tp.index]), pct={"Δg"}, digits=3)
    rep.p("«—»: regla de mismo cierre, fuera de la familia operable.")
    rep.h("Las alternativas con menor p frente a la LRS 3x original", 3)
    tp = t2.sort_values("p_boot").head(10)
    rep.table(pd.DataFrame({"Δg": tp["Δg"].values, "IR": tp["IR"].values,
                            "p_boot": [fp(p) for p in tp.p_boot],
                            "p Šidák (N_eff)": [fp(p) for p in tp["p_Šidák_Neff"]],
                            "p Romano–Wolf": [fp(p) for p in tp.p_RW],
                            "p_Holm": [fp(p) for p in tp.p_Holm],
                            "q_BH": [fp(p) for p in tp.q_BH], "q_BHY": [fp(p) for p in tp.q_BHY]},
                           index=[short(k, 60) for k in tp.index]), pct={"Δg"}, digits=3)
    rep.h(f"p-valores de titular publicados por R16–R18 ({len(pub)})", 3)
    pubs = pub.copy()
    for c in ("p", "p_Holm", "q_BH", "q_BHY"):
        pubs[c] = [fp(x) for x in pubs[c]]
    rep.table(pubs, index=False, digits=3)
    rep.p("Leídos de reports/r16_mechanism, reports/r17_benchmarks y reports/r18_worlds (p_titular.csv), "
          "comprobados con la huella del código actual (commit y huella en p_valores_publicados.csv). Las t "
          "de Newey-West se pasan a p unilateral normal en su módulo; el SPA de R17 entra como el menor p ya "
          "corregido con Holm sobre sus comparaciones (3 mercados × 4 γ), porque el mínimo sin corregir no es "
          "un p válido. Son contrastes heterogéneos (placebos, "
          "bootstrap, t de Newey-West, SPA), muchos sobre la misma regla y la misma historia, así que están "
          "correlacionados: Holm y BHY son válidos con cualquier dependencia; BH sólo con dependencia "
          "positiva. La familia no incluye los intervalos de R15 (antes y después de 1990) ni las "
          "comparaciones de R14, que no dan p-valores.")

    # ── 6. PBO ──
    rep.h("6. Probabilidad de sobreajuste del backtest (CSCV)")
    pb = pd.DataFrame({lab: {"pruebas": r["n_pruebas"], "PBO": r["pbo"],
                             "percentil fuera (mediana)": r["percentil_oos_mediano"],
                             "P(pérdida fuera)": r["p_pérdida"],
                             "elegida − mediana, fuera (anual)": r["oos_mejor_menos_mediana"] *
                             (np.sqrt(252) if "IR" in lab else 252),
                             "elegida más a menudo": label(r["nombres"][int(r["elegida"].index[0])]),
                             "su frecuencia": r["elegida"].iloc[0] / r["n_particiones"]}
                       for lab, r in cs.items()}).T.infer_objects()
    for c in ("pruebas",):
        pb[c] = pb[c].astype(int)
    for c in ("PBO", "percentil fuera (mediana)", "P(pérdida fuera)", "elegida − mediana, fuera (anual)",
              "su frecuencia"):
        pb[c] = pb[c].astype(float)
    rep.table(pb, pct={"PBO", "P(pérdida fuera)", "su frecuencia"}, digits=3)
    rep.p(f"{S_CSCV} bloques contiguos de unos {years / S_CSCV:.1f} años; en cada una de las "
          f"{pbo_all['n_particiones']} particiones en dos mitades se elige la mejor prueba en una mitad y se "
          "mira su puesto en la otra. PBO = fracción de particiones en que la elegida queda por debajo de la "
          "mediana fuera de muestra; «percentil fuera» es la mediana de su rango relativo ω. P(pérdida "
          "fuera) = fracción en que su IR (o Δg) fuera de muestra es negativo: frente al 1x, perder contra "
          "el 1x; frente a la LRS, perder contra la LRS. No se da la regresión de degradación de Bailey et "
          "al.: con mitades complementarias y casi siempre la misma elegida, el rendimiento fuera es el "
          "total menos el de dentro y la pendiente sale ≈ −1 por construcción. La PBO depende de qué "
          "se meta en la familia: con muchas reglas malas, elegir la mejor casi siempre deja algo por "
          "encima de la mediana aunque no sea la mejor de verdad; por eso se da también dentro de familias "
          "homogéneas.")
    fig_pbo(A, rep, ["todo el censo · IR frente al 1x", "parametrizaciones de la LRS 3x · IR frente al 1x",
                     "alternativas · IR frente a la LRS 3x original"])

    # ── 7. épocas ──
    rep.h("7. ¿Se conserva el orden entre épocas?")
    et = pd.DataFrame({tag: {"pruebas": len(e["names"]), "sesiones 1939–1991": e["n1"],
                             "sesiones 1992–2026": e["n2"], "Spearman IR": e["rho_ir"],
                             "nula: media": e["null_mean"], "nula: p95": e["null_p95"],
                             "p (nula)": fp(e["p_null"]),
                             "Kendall IR": e["tau_ir"], "Spearman Δg": e["rho_g"], "Kendall Δg": e["tau_g"],
                             "quintil superior que repite (IR)": e["top_ir"],
                             "quintil superior que repite (Δg)": e["top_g"]}
                       for tag, e in (("frente al 1x", ep1), ("alternativas frente a la LRS", ep2))}).T
    et = et.infer_objects()
    rep.table(et, pct={"quintil superior que repite (IR)", "quintil superior que repite (Δg)"}, digits=3)
    rep.p(f"Épocas: {A['start'].date()} → 1991-12-31 y 1992-01-01 → {A['end'].date()}. Nula: a cada prueba "
          f"se le resta en 1992–2026 su exceso medio (ninguna difiere de verdad en la segunda época) y se "
          f"remuestrea esa época ({B_RANK} veces, bootstrap estacionario de bloque {BLOCK}, comunes a todas las "
          "pruebas, lo que conserva su dependencia); el p es la fracción de Spearman nulos ≥ el observado. "
          "Rechazar la nula dice que el orden de la segunda época no es sólo ruido, no que sea el mismo por "
          "habilidad: con el 1x como referencia, una regla con más exposición media gana en cualquier época "
          f"con prima de riesgo positiva. Con independencia, el quintil superior repetiría un {pc(TOP_Q, 0)}.")
    rep.h("Por familias (IR frente al 1x)", 3)
    rep.table(A["epochs_fam"].infer_objects(), digits=2)
    epn = A["epochs"]["1x"]
    er = pd.DataFrame({"IR 1939–1991": epn["ir1"], "IR 1992–2026": epn["ir2"],
                       "Δg 1939–1991": epn["g1"], "Δg 1992–2026": epn["g2"]}, index=epn["names"])
    rep.h("Las reglas focales en cada época (frente al 1x)", 3)
    rep.table(er.loc[[k for k in FOCAL if k in er.index] + [best1]].rename(index=lambda s: short(s, 60)),
              pct={"Δg 1939–1991", "Δg 1992–2026"}, digits=3)
    fig_ranks(A, rep)

    # ── salvedades ──
    rep.h("Salvedades")
    rep.p("\n".join([
        "- El censo reconstruye las variantes con el protocolo común (retraso 2, costes de R13). R6 se "
        "probó con ejecución en el mismo cierre y con backtest.run (rotación por deriva de pesos) y R10 "
        "en el mismo cierre: aquí son las mismas señales con otra ejecución, no las cifras de sus informes.",
        "- Los modelos de ML de R12 se reentrenan aquí con el mismo código (walk_forward) y 4 hilos; el "
        "bosque aleatorio con n_jobs = 4 en lugar de todos los núcleos (mismo resultado con semilla fija). "
        "El gradient boosting con menos hilos puede diferir de R12 en el último decimal.",
        "- El N efectivo por autovalores usa la correlación de toda la muestra (diagnóstico ex post). Los "
        "estimadores discrepan entre sí por un factor de varios; por eso Šidák y el DSR se dan en función "
        "de N.",
        "- El DSR es descriptivo: «DSR ≥ 0,95» exige superar el máximo esperado en 1,645 errores típicos "
        f"de un solo IR y, bajo la nula, con N = {ne1:.0f} pruebas independientes, el máximo lo consigue "
        f"con probabilidad {pc(size1)}, no del 5%. El contraste al 5% con pruebas múltiples es Šidák a "
        "N_eff o Romano–Wolf. El DSR supone además que las pruebas son intercambiables bajo la nula: V "
        "observada mezcla dispersión real (reglas con apalancamientos distintos) con ruido y exagera SR₀; "
        "V = 1/T supone reglas independientes sin habilidad.",
        f"- Šidák a N_eff depende del estimador de N_eff (de {ne_rng1.min():.0f} a {ne_rng1.max():.0f}): "
        f"«{label(best1n)}» deja de sobrevivir por encima de unas {t1.loc[best1n, 'N_Šidák_5%']:.0f} pruebas "
        "independientes. Romano–Wolf no necesita N_eff, pero es conservador en la configuración menos "
        "favorable (cuenta todas las reglas, también las claramente peores) y sus p salen de "
        f"{B_BOOT} remuestreos: error de Monte Carlo de ±0,005 hacia p = 0,05. En el filo (p de "
        f"Romano–Wolf entre 0,04 y 0,06): {edge_txt}.",
        f"- MinTRL frente a SR₀(n) con V observada: V_disp = V − 1/T trata como ruido de muestreo toda la "
        f"varianza que tendrían pruebas independientes. Con reglas correlacionadas, el ruido que aparece en "
        f"la dispersión entre ellas es (1 − ρ̄)/T, con ρ̄ = {num(A['rho_bar']['1x'])} la correlación media "
        f"de los excesos: la dispersión real sería mayor, el límite de SR₀ subiría de {num(lim_obs1, 3)} a "
        f"{num(A['lim_rho'], 3)} y el MinTRL de la LRS 3x de {fyears(lrs_row['MinTRL_años_SR0(n)_V_obs'])} a "
        f"{fyears(A['mintrl_rho_lrs'])} años («{label(best1n)}»: {fyears(A['mintrl_rho_rot'])}). La columna «V obs.» "
        "es la versión benévola de ese criterio severo.",
        "- La LRS 3x original (la de Gayed y Bilello, con la media sobre la rentabilidad total) se contrasta "
        "sola porque la fijaron sus autores antes de este censo. Esa defensa vale frente a la búsqueda del "
        "laboratorio, no frente a la suya, que no se conoce (eligieron la media de 200 entre varias con los "
        "mismos datos); y su "
        f"muestra cubre el {pc(A['gb_overlap'], 0)} de esta ventana. El único tramo limpio (desde 2016) es "
        "corto y su p no es significativo.",
        "- El PSR/DSR y el MinTRL suponen observaciones independientes (con asimetría y curtosis). Los "
        "excesos diarios tienen algo de autocorrelación: el DSR con «T efectivo» (Newey-West) apenas "
        "cambia.",
        "- Los p-valores bootstrap y los de titular son unilaterales y no anidan exactamente los contrastes "
        "de sus módulos (otras ventanas, otras referencias). Los p-valores de R16–R18 se leen de sus "
        "p_titular.csv (con la huella del código comprobada), no se recalculan aquí.",
        "- La PBO no es la probabilidad de que la LRS sea falsa: mide si el proceso de elegir la mejor de "
        "la familia elige algo que fuera de muestra queda en la mitad buena. Depende de la composición de "
        "la familia y de que 16 bloques de ~5,5 años sean intercambiables.",
        "- El censo es una cota inferior de la búsqueda: no cuenta las versiones de datos de R14, los otros "
        "mercados, las decisiones previas (½ Kelly, 200 sesiones, GJR) ni lo que se miró y no se escribió. "
        "La señal de precio, que antes quedaba fuera como decisión previa, está dentro con su familia.",
        "- Las señales de rentabilidad total de R6 y R10 se calculan aquí con la misma rentabilidad sin "
        "mirar el futuro que la LRS original (D/P del mes anterior antes de 1988; ^SP500TR desde 1988); las "
        "reglas no cambian.",
        "- Este informe usa la base corregida (history.py, r12_long_ml.py, r13_lrs.py; ver BITACORA.md). "
        "Si la base cambia, cambian las cifras, y los p_titular.csv de R16–R18 deben regenerarse antes.",
    ]))

    # ── apéndice ──
    rep.h("Apéndice: todas las pruebas del censo")
    rep.p("<details><summary>Tabla completa (una fila por regla única)</summary>")
    full = pd.DataFrame({"módulo": A["meta"]["módulo"], "familia": A["meta"]["familia"],
                         "CAGR": A["desc"]["CAGR"], "maxDD": A["desc"]["maxDD"],
                         "Δg_1x": t1["Δg"], "IR_1x": t1["IR"], "PSR_1x": t1["PSR"], "DSR_1x": t1["DSR"],
                         "p_1x": [fp(p) for p in t1.p_boot], "p_RW_1x": [fp(p) for p in t1.p_RW],
                         "Δg_LRS": t2["Δg"].reindex(t1.index), "IR_LRS": t2["IR"].reindex(t1.index),
                         "p_LRS": [fp(p) if np.isfinite(p) else "" for p in t2.p_boot.reindex(t1.index)],
                         "duplicados": A["meta"]["duplicados"]})
    full = full.sort_values("IR_1x", ascending=False)
    rep.table(full, pct={"CAGR", "maxDD", "Δg_1x", "Δg_LRS"}, digits=3)
    rep.p("</details>")
    rep.p(f"Tiempo total de ejecución: {(time.time() - t_start) / 60:.1f} min.")
    rep.save()
    # series y fichas completas para quien quiera rehacer el análisis
    A["Rw"].to_csv(rep.dir / "censo_rentabilidades_diarias.csv.gz", compression="gzip")
    A["meta"].join(A["desc"]).join(t1.add_suffix("_1x")).join(t2.add_suffix("_LRS")).to_csv(
        rep.dir / "censo_fichas.csv")
    pub.to_csv(rep.dir / "p_valores_publicados.csv", index=False)
    return A


if __name__ == "__main__":
    run()
