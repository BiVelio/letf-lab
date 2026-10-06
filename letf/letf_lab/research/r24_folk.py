"""R24 — Las estrategias de los foros, a examen.

Pregunta. En Reddit (r/LETFs, r/TQQQ), Bogleheads, TradingView y Composer circulan
reglas para operar ETFs apalancados con historiales espectaculares. ¿Alguna mejora la
regla de Gayed y Bilello (2016), la «LRS 3x (original)» de R13, cuando se ejecuta con
el protocolo del laboratorio y se descuenta la búsqueda?

Reglas (fuente y fecha de cada una en la ficha de la sección 1; se implementan tal como
se publicaron, sin tocar sus parámetros):
- LRS reempaquetada: banda +5%/−3% sobre la media de 200 sesiones del índice propio
  (QQQ → TQQQ); banda +4%/−3% sobre el SPY para operar TQQQ, con la salida publicada
  (vender y comprar el 1x poco a poco en 6–12 meses) y los «guardaespaldas» (1x si el
  índice propio está más de un 30% sobre su media; caja si más de un 40%), y también
  con salida a caja; la señal del SPY (SPY > media de 200) para operar TQQQ; y la media
  de 200 calculada sobre el precio del propio 3x, una variante del laboratorio que no
  procede de ninguna fuente publicada (la página de Composer más próxima usa un cruce
  50/200 del propio TQQQ, otra regla).
- 3x con adornos: RSI(10) del 3x > 79 → caja; 9Sig (60/40 3x/bonos con una línea de
  señal trimestral del +9%); apalancarse (1x → 3x) tras caídas del 20% o del 30% hasta
  el máximo siguiente; aportar cada mes a un 3x (DCA).
- Paridad de riesgo: HFEA, 3x de acciones y 3x de bonos largos (TMF), 55/45 y 40/60,
  con reajuste trimestral.
- Reversión a corto: RSI(2) de Connors en el 3x; TQQQ For The Long Term (FTLT), un
  árbol de Composer con tronco LRS y ramas de RSI(10) que usa UVXY, SQQQ y TLT.

Protocolo (el de R13 y R19): señal al cierre de t, orden al cierre de t+1
(r13.EXEC_LAG = 2), 5 pb por unidad de rotación, lo no invertido en letras del Tesoro y
el apalancamiento financiado a f + s. Las reglas que solo usan el 3x, el 1x y letras
pasan por r13.run_weights; las que usan más fondos (FTLT, RSI → UVXY), por run_multi,
que es r13.run_weights con más columnas (se comprueba en cada ejecución que da lo mismo
con el 3x y el 1x); las carteras que derivan entre reajustes (HFEA, 9Sig), por
rebalance_sim, con la misma orden al cierre siguiente y el mismo coste. En el S&P 500
todas las señales usan el índice de RENTABILIDAD TOTAL de la LRS original
(r13.tr_index: ^SP500TR desde 1988; antes ^GSPC con el D/P de Shiller del mes
anterior); también el «SPY» con que el Nasdaq-100 decide en las reglas que lo miran.
La señal de precio (^GSPC; en la banda sobre el índice propio del Nasdaq-100, su precio
aproximado de R13) es una sensibilidad. El Nasdaq-100 se evalúa desde
1985-10-01 (el tramo 1971–85 reconstruido solo calienta las señales, con 10
reconstrucciones como sensibilidad) y los ETFs reales desde r13.REAL_FROM.

Fondos que no están en la base de R13, construidos aquí:
- Bonos del Tesoro sintéticos a partir de los rendimientos diarios de FRED (desde
  1962): un bono a la par de 25 años (≈ el vencimiento medio del índice ICE 20+ que
  replica TLT; DGS20, con el hueco de 1987–1993 relleno con DGS30 más el diferencial
  medio previo) y otro de 5 años (proxy de AGG). Se empalman con TLT y AGG reales.
- TMF (3x del bono largo) y los −3x del índice (SPXU, SQQQ) con la fórmula del 3x del
  laboratorio, rf + L·(r − f) − |L|·s − gasto, con s calibrado en la vida del fondo
  real, y empalmados con el real. Un fondo sintético que pierde el 100% un día se
  sustituye al día siguiente por uno nuevo (no se queda a cero para siempre, como en
  sim.letf_returns): solo pierde la estrategia que lo tenía ese día.
- UVXY y TECL solo existen como fondos reales (UVXY desde octubre de 2011).

Inferencia: Δg = diferencia de crecimiento logarítmico anual frente a la LRS 3x
(original) y frente al 1x, con bootstrap estacionario emparejado (stats.
stationary_indices, bloque medio de 63 sesiones, 2.000 remuestreos, los mismos para
todas las reglas de una ventana), intervalo percentil del 90% y p unilateral con el
error típico del bootstrap; Holm dentro de cada familia y SPA de Hansen
(stats.spa_test, con su recentrado) sobre la familia completa de reglas probadas.
Nada se llama mejora sin descontar la búsqueda.
"""

from __future__ import annotations

import logging
import re
import warnings
import zlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy.optimize import brentq
from scipy.stats import norm

from letf_lab import data, kelly, metrics, stats
from letf_lab.data import history
from letf_lab.report import FIG_W, ROOT as REPORTS, Report, comma_fig, legend_band, num_es, pct_es
from letf_lab.research import r12_long_ml as r12
from letf_lab.research import r13_lrs as r13
from letf_lab.universe import BY_TICKER

log = logging.getLogger(__name__)

SEED = 24
B = 2000
BLOCK = 63
LAG = r13.EXEC_LAG
COST_BPS = r13.COST_BPS
ORIGINAL = r13.ORIGINAL
ONE, THREE = "1x comprar y mantener", "3x comprar y mantener"
SP, NDX = "S&P 500", "Nasdaq-100"
SP_START = "1938-02-04"            # primera sesión de la tabla de R13 (primer día con previsión GJR)
BOND_START = "1962-02-01"          # primer mes completo con rendimientos diarios del Tesoro (FRED)
NDX_START = r12.NDX_EVAL_FROM      # 1985-10-01
REAL_FROM = r13.REAL_FROM          # 2010-02-12
SPLIT = pd.Timestamp(REAL_FROM)    # «antes / desde 2010»
POST_GB = "2016-01-04"             # tras la muestra de Gayed y Bilello (acaba en octubre de 2015)
LONG_M, MID_M = 25.0, 5.0          # vencimiento (años) del bono largo y del medio sintéticos
RAMP, RAMPS = 189, (126, 252)      # salida publicada: DCA en el 1x en 6–12 meses (9 = el punto medio)
SANITY = {"LRS 3x (original)": (0.174, -0.800), ONE: (0.115, -0.552), THREE: (0.149, -0.984)}

# ───────────────────────────── nombres y ficha ─────────────────────────────

BAND = "Banda +5%/−3% (índice propio)"
CORE = "SPY +4%/−3% (salida a caja)"
PUB = "SPY +4%/−3% publicada (salida a DCA)"
PUB6 = "SPY +4%/−3% publicada · DCA 6 meses"
PUB12 = "SPY +4%/−3% publicada · DCA 12 meses"
PUBG = "SPY +4%/−3% publicada · guardaespaldas fijos"
SPYSIG = "Señal del SPY para el 3x"
OWN3X = "Media de 200 del propio 3x"
RSI79 = "RSI(10) > 79 → caja"
RSI79U = "RSI(10) > 79 → UVXY"
NINE = "9Sig"
DIP20 = "Caída ≥ 20% → 3x"
DIP30 = "Caída ≥ 30% → 3x"
HFEA55 = "HFEA 55/45"
HFEA40 = "HFEA 40/60"
CONNORS = "RSI(2) de Connors"
FTLT_L = "FTLT versión larga (BIL por UVXY)"
FTLT_X = "FTLT original (UVXY, TECL)"
DCA = "DCA en el 3x"

C_LRS, C_EXTRA, C_RP, C_REV = "LRS reempaquetada", "3x con adornos", "paridad de riesgo", "reversión a corto"
CLASS_COLOR = {C_LRS: "#2a78d6", C_EXTRA: "#eb6834", C_RP: "#1baf7a", C_REV: "#4a3aa7"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SC, PX = " · mismo cierre", " · señal de precio"
LRS62 = "LRS 3x (original), desde el 1-2-1962"      # fila de comparación con las reglas con bonos

# Ficha de cada regla: la regla publicada, su fuente y fecha (de la exploración previa,
# reglas_fuentes.csv, con las URL tal cual; no se han vuelto a consultar), cómo se
# implementa aquí y la fecha desde la que se mide su fuera de muestra natural.
FICHA = {
    BAND: dict(clase=C_LRS, pub="2025-01-02", aprox=True, fecha="c. 2024–2025 (métricas «since 2010»)",
               regla="Compra TQQQ cuando QQQ cierra un 5% sobre su media de 200 sesiones; vende cuando cierra "
                     "un 3% por debajo; entre medias, SGOV (letras)",
               fuente="tradingview.com/script/VfrwgjA6-TQQQ-200-SMA-5-Entry-3-Exit-since-2010-Metrics-by-DE/ ; "
                      "tradingview.com/script/0S8jQcA5-200-SMA-5-3-Buffer-for-SPY-QQQ/ (r/LETFs)",
               impl="banda asimétrica sobre el índice propio con dividendos; fuera, letras"),
    CORE: dict(clase=C_LRS, pub="2025-01-02", aprox=True, fecha="c. 2025",
               regla="Compra TQQQ cuando el SPY cierra un 4% sobre su media de 200; vende cuando cierra un 3% "
                     "por debajo",
               fuente="tradingview.com/script/QEVAQHl6-SPY-200SMA-4-Entry-3-Exit-Strategy-QQQ-TQQQ/ "
                      "(freefighter07, r/TQQQ)",
               impl="núcleo de la regla con salida a letras (sin la rampa ni los guardaespaldas)"),
    PUB: dict(clase=C_LRS, pub="2025-01-02", aprox=True, fecha="c. 2025",
              regla="Lo anterior, pero al vender «slowly DCA into QQQ over the next 6–12 months» y vuelta al 100% "
                    "TQQQ con la siguiente compra; guardaespaldas: QQQ > 30% sobre su media → QQQ; > 40% → vender",
              fuente="misma página de freefighter07",
              impl="al vender, rampa lineal de letras al 1x en 9 meses (189 sesiones); guardaespaldas mientras "
                   "dure la condición (6 y 12 meses y guardaespaldas fijos hasta la nueva compra: sensibilidad)"),
    SPYSIG: dict(clase=C_LRS, pub="2023-01-25", aprox=True, fecha="2021–2023",
                 regla="TQQQ si el SPY cierra sobre su media de 200 (tronco de FTLT y consejo habitual en r/LETFs)",
                 fuente="composer.trade/trading-strategies/tqqq-for-the-long-term-reddit-post-link-"
                        "HukRwDJLlYPLMbrQbua5",
                 impl="solo en el Nasdaq-100 (en el S&P 500 es la LRS original)"),
    OWN3X: dict(clase=C_LRS, pub=None, aprox=True, fecha="—",
                regla="La media de 200 calculada sobre el precio del propio ETF apalancado (variante del "
                      "laboratorio; no hay una fuente publicada con esta regla exacta)",
                fuente="sin fuente verificable: la exploración previa la anotó como «práctica habitual». La página "
                       "de Composer más próxima (…tqqq-for-the-long-term-reddit-post-link-50d-tqqq200d-…-"
                       "gj4k6luYNV0LPIxDKNW3) es otra regla (un cruce de las medias de 50 y 200 del propio TQQQ) y "
                       "no se le atribuye",
                impl="3x si su precio cierra sobre su media de 200; sin fecha de publicación, no entra en la "
                     "tabla del fuera de muestra tras la publicación"),
    RSI79: dict(clase=C_EXTRA, pub="2024-10-16", aprox=False, fecha="15-10-2024",
                regla="TQQQ salvo que el RSI(10) de TQQQ pase de 79; entonces UVXY (diario)",
                fuente="composer.trade/trading-strategies/simple-tqqq-rsi-mean-reversion-4hcYKZBIjhZo3Yg0NTQk",
                impl="versión caja en toda la historia; versión UVXY solo desde octubre de 2011"),
    RSI79U: dict(clase=C_EXTRA, pub="2024-10-16", aprox=False, fecha="15-10-2024",
                 regla="La misma, con UVXY", fuente="la misma", impl="solo Nasdaq-100 con ETFs reales, desde 2011-10"),
    NINE: dict(clase=C_EXTRA, pub="2017-01-03", aprox=True, fecha="c. 2017",
               regla="60/40 TQQQ/AGG; cada trimestre el TQQQ vuelve a su línea de señal (+9% sobre el saldo tras el "
                     "último reajuste); compras de hasta el 90% del saldo en bonos; «30 down»: si el 3x cierra el "
                     "trimestre ≤ 70% de su máximo de 8 trimestres, se ignoran las 2 ventas siguientes; «spike»: "
                     "+100% en el trimestre → 60%",
               fuente="jasonkelly.com/resources/strategies/ ; jasonkelly.com/2023/02/best-3x-leveraged-etfs/ ; "
                      "bestfolio.app/blog/kelly-signal-danger-v2",
               impl="reconstrucción; AGG sintético (Tesoro a 5 años) antes de 2003; sin el «base reset» que "
                    "BestFolio declara añadido suyo"),
    DIP20: dict(clase=C_EXTRA, pub=None, aprox=True, fecha="—",
                regla="«Comprar la caída con apalancamiento»: 1x normalmente; 3x desde que el índice cae un 20% "
                      "desde máximos hasta que marca un máximo nuevo",
                fuente="sin regla canónica: estilización de lo que se describe en los foros",
                impl="caída sobre el índice con dividendos de la señal"),
    DIP30: dict(clase=C_EXTRA, pub=None, aprox=True, fecha="—", regla="Lo mismo con un 30%",
                fuente="ídem", impl="ídem"),
    HFEA55: dict(clase=C_RP, pub="2019-08-01", aprox=False, fecha="agosto de 2019",
                 regla="UPRO/TMF 55/45 con reajuste trimestral (HEDGEFUNDIE)",
                 fuente="bogleheads.org/forum/viewtopic.php?t=272007 ; optimizedportfolio.com/hedgefundie-adventure/",
                 impl="3x de acciones + TMF (sintético desde 1962, real desde 2009-04); en el Nasdaq-100, TQQQ/TMF"),
    HFEA40: dict(clase=C_RP, pub="2019-02-01", aprox=False, fecha="febrero de 2019",
                 regla="UPRO/TMF 40/60, la versión inicial", fuente="ídem", impl="ídem"),
    CONNORS: dict(clase=C_REV, pub="2009-01-02", aprox=False, fecha="2008",
                  regla="Compra si el precio está sobre su media de 200 y el RSI(2) baja de 10; vende al cerrar "
                        "sobre la media de 5",
                  fuente="chartschool.stockcharts.com/…/trading-strategies/rsi-2 (Connors y Alvarez, 2008)",
                  impl="señal sobre el índice con dividendos, posición en el 3x"),
    FTLT_L: dict(clase=C_REV, pub="2024-09-03", aprox=False, fecha="31-8-2024",
                 regla="El árbol de FTLT con BIL (letras) en lugar de UVXY, TQQQ en lugar de TECL y RSI(10) del SPY "
                       "> 80 en lugar del de SPXL",
                 fuente="composer.trade/trading-strategies/tqqq-for-the-long-term-reddit-post-link-bil-instead-of-"
                        "uvxy-and-tqqq-intsead-of-tcel-for-longterm-backtest-compare-oT9dyDsvBpCUkJWCwoom",
                 impl="historia larga con el 3x, el −3x y TLT sintéticos; en el S&P 500, UPRO/SPXU"),
    FTLT_X: dict(clase=C_REV, pub="2023-01-25", aprox=False, fecha="última edición 24-1-2023",
                 regla="Si SPY > media de 200: UVXY si RSI10(TQQQ) > 79 o RSI10(SPXL) > 80; si no, TQQQ. Si SPY ≤ "
                       "media: TECL si RSI10(TQQQ) < 31; UPRO si RSI10(SPY) < 30; si TQQQ < su media de 20, el de "
                       "mayor RSI10 entre SQQQ y TLT; si no, SQQQ si RSI10(SQQQ) < 31, o TQQQ. Diario.",
                 fuente="composer.trade/trading-strategies/tqqq-for-the-long-term-reddit-post-link-"
                        "HukRwDJLlYPLMbrQbua5 (atribuida a Dereck Nielsen; la versión «Original» ampliada dice "
                        "inspirarse en «Leverage For The Long Run»)",
                 impl="árbol exacto con ETFs reales, solo desde octubre de 2011 (UVXY)"),
    DCA: dict(clase=C_EXTRA, pub=None, aprox=True, fecha="2021–2024",
              regla="Aportación mensual fija al 3x en lugar del 1x",
              fuente="seekingalpha.com/article/4416406 ; bogleheads.org/forum/viewtopic.php?t=378882 ; "
                     "czxttkl.com/2024/01/13/dollar-cost-average-on-tqqq-vs-qqq-real-data/",
              impl="ventanas móviles de 10 y 20 años; no es una serie de rentabilidades (es el 3x)"),
}
for _k in (PUB6, PUB12, PUBG):
    FICHA[_k] = dict(FICHA[PUB], impl="sensibilidad de la salida publicada")

# Composer publica, para cada «symphony», el backtest desde su última edición: su fuera de
# muestra. Cifras leídas en su API pública durante la exploración (rentabilidad anual,
# acumulada y caída máxima, hasta el 25-9-2026); no se han vuelto a comprobar aquí.
COMPOSER = [(FTLT_X, "2023-01-24", 0.8910, 9.2885, -0.5019),
            (FTLT_L, "2024-09-03", 0.7459, 2.1441, -0.5076),
            (RSI79U, "2024-10-15", 0.3705, 0.8410, -0.5804)]

EXPO = {"x3": 3.0, "u": 1.0, "x_3": -3.0, "sp_x3": 3.0, "tecl": 3.0,
        "tlt": 0.0, "tmf": 0.0, "agg": 0.0, "uvxy": 0.0}


def rng_for(tag: str) -> np.random.Generator:
    """Generador determinista por bloque (SEED y la etiqueta): el resultado de un bloque no
    depende del orden en que se ejecuten los demás."""
    return np.random.default_rng([SEED, zlib.crc32(tag.encode())])


# ───────────────────────────── bonos y fondos sintéticos ─────────────────────────────

def par_bond_returns(y: pd.Series, M: float) -> pd.Series:
    """Rentabilidad diaria de un bono a la par de M años con cupón semestral: cada día se
    compra a la par (cupón y_{t−1}), se valora con y_t y devenga el cupón de los días
    naturales transcurridos (como mucho 5)."""
    c = y.shift(1)
    n = 2 * M
    yt, cc = y / 2, c / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        price = cc / yt * (1 - (1 + yt) ** (-n)) + (1 + yt) ** (-n)
    dt = y.index.to_series().diff().dt.days.clip(upper=5) / 365.25
    return (price - 1 + c * dt).rename(f"bono{M:g}")


def long_bond_yield() -> pd.Series:
    """Rendimiento a 20 años (DGS20); el hueco de 1987–1993 se rellena con DGS30 más el
    diferencial medio 20 − 30 de las 250 sesiones anteriores al hueco."""
    y20, y30 = history.fred("DGS20") / 100, history.fred("DGS30") / 100
    y = y20.reindex(y20.index.union(y30.index))
    y30 = y30.reindex(y.index)
    gap = y.isna() & y30.notna() & (y.index > "1986-12-31") & (y.index < "1993-10-01")
    pre = (y - y30).loc[:"1986-12-31"].dropna().iloc[-250:].mean()
    y[gap] = y30[gap] + pre
    return y.dropna()


def bond_on(kind: str, idx: pd.DatetimeIndex) -> pd.Series:
    """Bono sintético en las sesiones de bolsa. Los días con la bolsa abierta y el mercado
    de bonos cerrado conservan el rendimiento anterior y solo devengan cupón."""
    y = long_bond_yield() if kind == "largo" else history.fred("DGS5").dropna() / 100
    y = y.reindex(y.index.union(idx)).ffill().reindex(idx)
    return par_bond_returns(y, LONG_M if kind == "largo" else MID_M)


def real_returns(ticker: str) -> pd.Series:
    """Rentabilidad diaria del ETF real. Los apalancados pasan por common.pair (vía
    r12.real_letf), que corrige los días con un contrasplit mal ajustado."""
    if ticker in BY_TICKER:
        return r12.real_letf(ticker)
    return data.panel([ticker], "adj_close")[ticker].pct_change(fill_method=None).dropna().rename(ticker)


def splice(synth: pd.Series, real: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Sintético hasta la primera rentabilidad del real y el real después, en idx."""
    real = real.dropna()
    return pd.concat([synth[synth.index < real.index[0]], real]).sort_index().reindex(idx)


def lev_fund(u: pd.Series, rf: pd.Series, f: pd.Series, L: float, er: float, s: float) -> pd.Series:
    """Fondo apalancado sintético del laboratorio (r12.letf): rf + L·(u − f) − |L|·s − gasto.

    A diferencia de sim.letf_returns, un día de −100% no deja el fondo a cero para siempre:
    al día siguiente se compra un fondo nuevo (solo pierde quien lo tenía ese día)."""
    r = rf + L * (u - f) - abs(L) * s / 252 - er / 252
    return r.clip(lower=-1.0)


def calibrate_spread(u: pd.Series, rf: pd.Series, f: pd.Series, L: float, er: float,
                     real: pd.Series) -> float:
    """s tal que la media diaria del sintético iguala a la del fondo real en su vida."""
    real = real.dropna()
    idx = real.index.intersection(u.dropna().index)
    base = rf.reindex(idx) + L * (u.reindex(idx) - f.reindex(idx)) - er / 252
    return float((base - real.reindex(idx)).mean() * 252 / abs(L))


def assets(d: dict, market: str, sp: dict | None = None) -> dict:
    """Todo lo que usan las reglas en un mercado (claves de r12.make más fondos y señales).

    Fondos: x3 (el 3x de R13), u (1x), x_3 (−3x: SPXU o SQQQ), tlt, agg, tmf, uvxy,
    tecl y sp_x3 (el 3x del S&P 500, para las ramas de FTLT que compran UPRO).
    Señales (niveles): sig (índice con dividendos de la señal de la LRS, r13.tr_index),
    sp_sig (el del S&P 500), px_price (índice de precio del mercado, sensibilidad: ^GSPC en el
    S&P 500; en el Nasdaq-100, el precio aproximado de R13), sp_px (el precio del S&P 500,
    ^GSPC), px3, px_3, px_tlt y px_spxl (precio de cada fondo)."""
    idx = d["u"].index
    rf, f, u = d["rf"], d["f"], d["u"]
    a = {k: d[k] for k in ("u", "rf", "f", "x3", "spread", "er", "eval_from")}
    a["market"] = market
    real = {t: real_returns(t) for t in ("TLT", "AGG", "TMF", "UVXY", "TECL", "SPXL")}
    a["tlt"] = splice(bond_on("largo", idx), real["TLT"], idx)
    a["agg"] = splice(bond_on("medio", idx), real["AGG"], idx)
    e = BY_TICKER["TMF"]
    a["s_tmf"] = calibrate_spread(a["tlt"], rf, f, 3.0, e.expense_ratio, real["TMF"])
    a["tmf"] = splice(lev_fund(a["tlt"], rf, f, 3.0, e.expense_ratio, a["s_tmf"]), real["TMF"], idx)
    inv = "SQQQ" if market == NDX else "SPXU"
    e = BY_TICKER[inv]
    r_inv = real_returns(inv)
    a["s_inv"] = calibrate_spread(u, rf, f, -3.0, e.expense_ratio, r_inv)
    a["x_3"] = splice(lev_fund(u, rf, f, -3.0, e.expense_ratio, a["s_inv"]), r_inv, idx)
    a["uvxy"] = real["UVXY"].reindex(idx)
    a["tecl"] = real["TECL"].reindex(idx)
    a["sig"] = r13.tr_index(d)
    if sp is None:
        a["sp_sig"], a["sp_x3"] = a["sig"], d["x3"]
    else:
        a["sp_sig"] = r13.tr_index(sp).reindex(idx).ffill()
        a["sp_x3"] = sp["x3"].reindex(idx)
    a["px_price"] = d["px"]
    a["sp_px"] = d["px"] if sp is None else sp["px"].reindex(idx).ffill()
    a["px3"] = level(d["x3"])
    a["px_3"] = level(a["x_3"])
    a["px_tlt"] = level(a["tlt"])
    a["px_spxl"] = level(real["SPXL"].reindex(idx))
    a["inv_ticker"] = inv
    return a


def level(r: pd.Series) -> pd.Series:
    """Nivel (base 1) de una serie de rentabilidades, NaN antes de su primer dato."""
    first = r.first_valid_index()
    out = (1 + r.loc[first:].fillna(0)).cumprod()
    return out.reindex(r.index)


# ───────────────────────────── indicadores ─────────────────────────────

def sma(px: pd.Series, n: int) -> pd.Series:
    return px.rolling(n).mean()


def rsi(px: pd.Series, n: int) -> pd.Series:
    """RSI de Wilder como ta.rsi de TradingView: medias RMA (α = 1/n) sembradas con la media
    simple de las n primeras variaciones; 100 si no hay bajadas y 0 si no hay subidas."""
    p = px.dropna()
    dlt = np.diff(p.to_numpy(float))
    out = np.full(len(p), np.nan)
    if len(dlt) >= n:
        g, lo = np.clip(dlt, 0, None), np.clip(-dlt, 0, None)
        ag, al = g[:n].mean(), lo[:n].mean()

        def val(up, dn):
            return 100.0 if dn == 0 else 0.0 if up == 0 else 100 - 100 / (1 + up / dn)

        out[n] = val(ag, al)
        for i in range(n, len(dlt)):
            ag = (ag * (n - 1) + g[i]) / n
            al = (al * (n - 1) + lo[i]) / n
            out[i + 1] = val(ag, al)
    return pd.Series(out, index=p.index).reindex(px.index)


def band_state(px: pd.Series, n: int, up: float, dn: float) -> pd.Series:
    """1 tras cerrar sobre media·(1 + up); 0 tras cerrar bajo media·(1 − dn); entre medias
    conserva el estado. Antes del primer cruce, fuera (0); NaN sin media."""
    ma = sma(px, n)
    st = np.where(px > ma * (1 + up), 1.0, np.where(px < ma * (1 - dn), 0.0, np.nan))
    return pd.Series(st, index=px.index).ffill().fillna(0.0).where(ma.notna())


# ───────────────────────────── reglas (pesos decididos al cierre de t) ─────────────────────────────

def W(a: dict, **cols) -> pd.DataFrame:
    idx = a["u"].index
    return pd.DataFrame({k: v.reindex(idx) for k, v in cols.items()})


def lrs(a: dict) -> pd.DataFrame:
    """La LRS 3x (original): 3x sobre la media de 200 del índice con dividendos."""
    return W(a, x3=r13.above_sma(a["sig"]))


def band_own(a: dict, sig: str = "sig") -> pd.DataFrame:
    return W(a, x3=band_state(a[sig], 200, 0.05, 0.03))


def spy_core(a: dict, sig: str = "sp_sig") -> pd.DataFrame:
    return W(a, x3=band_state(a[sig], 200, 0.04, 0.03))


def spy_published(a: dict, ramp: int = RAMP, sticky: bool = False, sig: str = "sp_sig",
                  own: str = "sig") -> pd.DataFrame:
    """SPY +4%/−3% según el texto publicado: en compra, 3x; al vender, rampa lineal de
    letras al 1x en `ramp` sesiones; guardaespaldas: índice propio > 30% sobre su media →
    1x, > 40% → letras, mientras dure (sticky: hasta la siguiente señal de compra)."""
    core = band_state(a[sig], 200, 0.04, 0.03).to_numpy(float)
    s = a[own]
    ext = (s / sma(s, 200) - 1).to_numpy(float)
    n = len(core)
    w3, w1 = np.full(n, np.nan), np.full(n, np.nan)
    since, guard, prev = None, 0, np.nan
    for i in range(n):
        c, e = core[i], ext[i]
        if np.isnan(c) or np.isnan(e):
            continue
        if c == 1:
            if prev != 1:
                guard = 0
            g = 2 if e >= 0.40 else 1 if e >= 0.30 else 0
            guard = max(guard, g) if sticky else g
            w3[i], w1[i] = (0.0, 0.0) if guard == 2 else (0.0, 1.0) if guard == 1 else (1.0, 0.0)
            since = None
        else:
            since = 0 if since is None else since + 1
            w3[i], w1[i] = 0.0, min(1.0, (since + 1) / ramp)
        prev = c
    idx = a["u"].index
    return W(a, x3=pd.Series(w3, idx), u=pd.Series(w1, idx))


def spy_signal(a: dict) -> pd.DataFrame:
    return W(a, x3=r13.above_sma(a["sp_sig"]))


def own3x(a: dict) -> pd.DataFrame:
    return W(a, x3=r13.above_sma(a["px3"]))


def rsi79_cash(a: dict) -> pd.DataFrame:
    r = rsi(a["px3"], 10)
    return W(a, x3=(r <= 79).astype(float).where(r.notna()))


def rsi79_uvxy(a: dict) -> pd.DataFrame:
    r = rsi(a["px3"], 10)
    hot = (r > 79).astype(float).where(r.notna())
    return W(a, x3=1 - hot, uvxy=hot)


def connors(a: dict, sig: str = "sig", entry: float = 10.0) -> pd.DataFrame:
    """RSI(2) de Connors: compra si el índice está sobre su media de 200 y RSI(2) < 10;
    vende al cerrar sobre la media de 5. Señal sobre el índice, posición en el 3x."""
    px = a[sig]
    r2, m200, m5 = rsi(px, 2).to_numpy(float), sma(px, 200).to_numpy(float), sma(px, 5).to_numpy(float)
    p = px.to_numpy(float)
    st = np.full(len(p), np.nan)
    cur = 0.0
    for i in range(len(p)):
        if np.isnan(m200[i]) or np.isnan(r2[i]) or np.isnan(m5[i]):
            continue
        if cur == 0.0 and p[i] > m200[i] and r2[i] < entry:
            cur = 1.0
        elif cur == 1.0 and p[i] > m5[i]:
            cur = 0.0
        st[i] = cur
    return W(a, x3=pd.Series(st, px.index))


def dip_leverage(a: dict, k: float, sig: str = "sig") -> pd.DataFrame:
    """1x normalmente; 3x desde que el índice cae un k desde su máximo hasta el siguiente
    máximo."""
    px = a[sig].to_numpy(float)
    st = np.full(len(px), np.nan)
    peak, cur = -np.inf, 0.0
    for i, p in enumerate(px):
        if np.isnan(p):
            continue
        if p >= peak:
            peak, cur = p, 0.0
        elif p <= peak * (1 - k):
            cur = 1.0
        st[i] = cur
    s = pd.Series(st, a[sig].index)
    return W(a, x3=s, u=1 - s)


def ftlt(a: dict, exact: bool, hot_to: str = "uvxy") -> pd.DataFrame:
    """TQQQ For The Long Term. exact=True: el árbol de Composer (UVXY, TECL, RSI de SPXL);
    exact=False: la versión larga publicada (BIL por UVXY, TQQQ por TECL, RSI del SPY por
    el de SPXL). hot_to cambia el destino de la rama UVXY en el exacto: «uvxy», «cash»
    (letras) o «x3» (seguir en el 3x)."""
    sp, p3, p_3, ptl = a["sp_sig"], a["px3"], a["px_3"], a["px_tlt"]
    up_ma = sma(sp, 200)
    up = sp > up_ma
    r3, r_3, rtl, rsp = rsi(p3, 10), rsi(p_3, 10), rsi(ptl, 10), rsi(sp, 10)
    r_hot2 = rsi(a["px_spxl"], 10) if exact else rsp
    below20 = p3 < sma(p3, 20)
    cols = ["x3", "sp_x3", "x_3", "tlt"] + (["uvxy", "tecl"] if exact else [])
    out = pd.DataFrame(0.0, index=a["u"].index, columns=cols)
    hot1 = up & (r3 > 79)
    hot2 = up & ~hot1 & (r_hot2 > 80)
    ok_up = up & ~hot1 & ~hot2
    dn = ~up
    dip1 = dn & (r3 < 31)
    dip2 = dn & ~dip1 & (rsp < 30)
    weak = dn & ~dip1 & ~dip2 & below20
    rest = dn & ~dip1 & ~dip2 & ~below20
    choose_inv = r_3 >= rtl                      # «select top 1» por RSI(10) entre SQQQ y TLT
    if exact:
        out.loc[hot1 | hot2, "uvxy"] = 1.0
    out.loc[ok_up, "x3"] = 1.0
    out.loc[dip1, "tecl" if exact else "x3"] = 1.0
    out.loc[dip2, "sp_x3"] = 1.0
    out.loc[weak & choose_inv, "x_3"] = 1.0
    out.loc[weak & ~choose_inv, "tlt"] = 1.0
    out.loc[rest & (r_3 < 31), "x_3"] = 1.0
    out.loc[rest & ~(r_3 < 31), "x3"] = 1.0
    ok = up_ma.notna() & r3.notna() & r_3.notna() & rtl.notna() & rsp.notna() & r_hot2.notna() \
        & sma(p3, 20).notna()
    out = out.where(ok)
    if exact and hot_to != "uvxy":
        if hot_to == "x3":
            out["x3"] = out["x3"] + out["uvxy"]
        out["uvxy"] = out["uvxy"] * 0.0
    return merge_same(a, out)


def merge_same(a: dict, w: pd.DataFrame) -> pd.DataFrame:
    """En el S&P 500 el «UPRO» de las ramas de FTLT es el propio 3x: una sola columna, para
    no cobrar una rotación entre dos columnas que son el mismo fondo."""
    if a["market"] == SP and "sp_x3" in w:
        w = w.copy()
        w["x3"] = w["x3"] + w.pop("sp_x3")
    return w


# ───────────────────────────── motor ─────────────────────────────

def run_multi(a: dict, w: pd.DataFrame, lag: int = LAG, cost_bps: float = COST_BPS) -> pd.Series:
    """r13.run_weights con más fondos: pesos w (columnas = fondos de `a`) decididos al
    cierre de t, cobran desde el cierre de t + lag − 1; el resto en letras; cost_bps por
    unidad de rotación. Un peso distinto de cero sobre un fondo sin dato anula el día."""
    X = pd.DataFrame({c: a[c] for c in w.columns}).reindex(w.index)
    ws = w.shift(lag)
    valid = ws.notna().all(axis=1)
    ws = ws.fillna(0.0)
    turn = ws.diff().abs().fillna(0).sum(axis=1)
    r = (ws * X.fillna(0.0)).sum(axis=1) + (1 - ws.sum(axis=1)) * a["rf"] - turn * cost_bps / 1e4
    bad = ((ws != 0) & X.isna()).any(axis=1)
    return r.where(valid & ~bad)


def run_rule(a: dict, w: pd.DataFrame, lag: int = LAG, cost_bps: float = COST_BPS) -> pd.Series:
    """Con solo el 3x y el 1x, r13.run_weights tal cual; con más fondos, run_multi."""
    if set(w.columns) <= {"x3", "u"}:
        zero = pd.Series(0.0, index=w.index)
        return r13.run_weights(a, w["x3"] if "x3" in w else zero, w["u"] if "u" in w else None,
                               lag=lag, cost_bps=cost_bps)
    return run_multi(a, w, lag, cost_bps)


def exposure(w: pd.DataFrame) -> pd.Series:
    """Exposición a la bolsa (3 = un 3x entero; el −3x cuenta −3; bonos y UVXY, 0)."""
    return sum(w[c] * EXPO[c] for c in w.columns).where(w.notna().all(axis=1))


def quarter_ends(idx: pd.DatetimeIndex) -> set:
    s = pd.Series(idx, index=idx)
    return set(s.groupby([idx.year, idx.quarter]).max().values)


def rebalance_sim(a: dict, cols: list[str], decide, start, end=None, lag: int = LAG,
                  cost_bps: float = COST_BPS) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Cartera cuyas posiciones derivan entre reajustes (HFEA, 9Sig).

    Se compra al cierre de la sesión anterior a `start` (el coste de esa compra entra en la
    primera rentabilidad, que es la de `start`, como en las reglas de señal de la misma
    ventana); si no hay sesión anterior con datos, al cierre de la primera y la primera
    rentabilidad es la del día siguiente. decide(i, t, h, state) se llama al cierre de cada sesión t con
    las posiciones h (en $) y devuelve None o los pesos objetivo; la orden se ejecuta
    lag − 1 sesiones después, al cierre, con cost_bps por unidad de rotación (Σ|Δ$|/valor).
    Devuelve las rentabilidades, los pesos al cierre (tras operar) y los días con orden
    ejecutada, en el índice completo de `a`."""
    assert lag >= 2, "rebalance_sim ejecuta al cierre siguiente o más tarde"
    allx = pd.DataFrame({c: a[c] for c in cols}).loc[:end].dropna()
    i0 = max(int(allx.index.searchsorted(pd.Timestamp(start))) - 1, 0)   # la compra, al cierre anterior
    X = allx.iloc[i0:]
    idx, Xv = X.index, X.to_numpy(float)
    n, k = Xv.shape
    rets, wts, trades = np.full(n, np.nan), np.full((n, k), np.nan), np.zeros(n)
    state: dict = {}
    w0 = np.asarray(decide(-1, idx[0], None, state), float)
    h = w0 * (1 - np.abs(w0).sum() * cost_bps / 1e4)
    V_prev = 1.0
    wts[0] = h / h.sum()
    trades[0] = 1.0
    pending: list[tuple[int, np.ndarray]] = []
    tgt = decide(0, idx[0], h.copy(), state)
    if tgt is not None:
        pending.append((lag - 1, np.asarray(tgt, float)))
    for i in range(1, n):
        h = h * (1 + Xv[i])
        V = h.sum()
        while pending and pending[0][0] == i:
            _, tg = pending.pop(0)
            turn = np.abs(tg * V - h).sum() / V
            V *= 1 - turn * cost_bps / 1e4
            h = tg * V
            trades[i] = 1.0
        rets[i] = V / V_prev - 1
        V_prev = V
        wts[i] = h / V
        tgt = decide(i, idx[i], h.copy(), state)
        if tgt is not None:
            pending.append((i + lag - 1, np.asarray(tgt, float)))
    full = a["u"].index
    return (pd.Series(rets, idx).reindex(full), pd.DataFrame(wts, idx, cols).reindex(full),
            pd.Series(trades, idx).reindex(full))


def hfea(a: dict, w_eq: float, start, end=None, lag: int = LAG):
    """HEDGEFUNDIE: w_eq en el 3x de acciones y el resto en TMF, reajuste trimestral."""
    tgt = np.array([w_eq, 1 - w_eq])
    qe = quarter_ends(a["u"].index)

    def decide(i, t, h, state):
        if h is None:
            return tgt
        return tgt if t in qe else None

    return rebalance_sim(a, ["x3", "tmf"], decide, start, end, lag)


def nine_sig(a: dict, start, end=None, lag: int = LAG, g: float = 0.09, skips: int = 2):
    """9Sig de Jason Kelly (reconstrucción): 60/40 3x/AGG; cada trimestre el 3x vuelve a
    su línea de señal (saldo del 3x tras el último reajuste × 1,09); las compras se limitan
    al 90% del saldo en bonos; «30 down»: si el 3x cierra el trimestre en el 70% o menos de
    su máximo de los 8 cierres trimestrales anteriores, se ignoran las 2 ventas siguientes;
    «spike»: si el 3x gana un 100% o más en el trimestre y tras el reajuste pesaría entre
    el 60% y el 100%, fuera de un periodo «30 down», vuelve al 60%."""
    qe = quarter_ends(a["u"].index)
    p3 = level(a["x3"])

    def decide(i, t, h, state):
        if h is None:
            state.update(skip=0, qcloses=[], last_p=None, T_after=0.6)
            return np.array([0.6, 0.4])
        if t not in qe:
            return None
        T, Bd = h
        V = T + Bd
        p = float(p3.loc[t])
        hist = state["qcloses"][-8:]
        state["qcloses"].append(p)
        if hist and p <= 0.70 * max(hist):
            state["skip"] = skips
        qret = p / state["last_p"] - 1 if state["last_p"] else 0.0
        state["last_p"] = p
        signal = state["T_after"] * (1 + g)
        if T > signal:
            if state["skip"] > 0:
                state["skip"] -= 1
                newT = T
            else:
                newT = signal
        else:
            newT = T + min(signal - T, 0.9 * Bd)
        if qret >= 1.0 and state["skip"] == 0 and 0.6 <= newT / V <= 1.0:
            newT = 0.6 * V
        state["T_after"] = newT
        return np.array([newT / V, 1 - newT / V])

    return rebalance_sim(a, ["x3", "agg"], decide, start, end, lag)


# ───────────────────────────── libro de reglas ─────────────────────────────

class Book:
    """Rentabilidades (R), exposición decidida (E), días con operación (T) y clase de cada
    regla de un mercado y una ventana."""

    def __init__(self, market: str):
        self.market = market
        self.R: dict[str, pd.Series] = {}
        self.E: dict[str, pd.Series] = {}
        self.T: dict[str, pd.Series] = {}
        self.main: list[str] = []      # la familia principal (sin referencias ni sensibilidades)
        self.sens: list[str] = []      # sensibilidades (salidas, guardaespaldas, señal de precio)
        self.refs: list[str] = []      # referencias: 1x, 3x y LRS original
        self.dups: dict[str, str] = {}  # regla idéntica a otra ya construida → la que se conserva

    def add(self, name: str, r: pd.Series, e: pd.Series, t: pd.Series, main: bool = True, ref: bool = False):
        """Añade una regla; si su serie es idéntica a la de una regla ya añadida (misma
        rentabilidad todos los días), no se añade y se anota en dups: cuenta una vez."""
        assert name not in self.R and name not in self.dups, name
        for k in self.refs + self.main + self.sens:
            other = self.R[k]
            if other.isna().equals(r.isna()) and float((other - r).abs().max()) < 1e-12:
                self.dups[name] = k
                return
        self.R[name], self.E[name], self.T[name] = r, e, t
        (self.refs if ref else self.main if main else self.sens).append(name)


def weight_rules(a: dict, uvxy: bool = False) -> dict[str, tuple[pd.DataFrame, bool]]:
    """Reglas de pesos del mercado de `a`: {nombre: (pesos, principal)}."""
    nd = a["market"] == NDX
    out = {BAND: (band_own(a), True), CORE: (spy_core(a), True), PUB: (spy_published(a), True)}
    for ramp, name in zip(RAMPS, (PUB6, PUB12)):
        out[name] = (spy_published(a, ramp=ramp), False)
    out[PUBG] = (spy_published(a, sticky=True), False)
    if nd:
        out[SPYSIG] = (spy_signal(a), True)
    out[OWN3X] = (own3x(a), True)
    out[RSI79] = (rsi79_cash(a), True)
    out[DIP20] = (dip_leverage(a, 0.20), True)
    out[DIP30] = (dip_leverage(a, 0.30), True)
    out[CONNORS] = (connors(a), True)
    out[FTLT_L] = (ftlt(a, exact=False), True)
    if uvxy:
        out[FTLT_X] = (ftlt(a, exact=True), True)
        out[RSI79U] = (rsi79_uvxy(a), True)
    # sensibilidad: las bandas con la señal de precio (en el Nasdaq-100, el SPY de precio para las reglas
    # que miran el SPY y el precio aproximado del índice propio, como en R13)
    for k, w in ((BAND, band_own(a, "px_price")), (CORE, spy_core(a, "sp_px")),
                 (PUB, spy_published(a, sig="sp_px", own="px_price"))):
        out[k + PX] = (w, False)
    return out


def trades_of(w: pd.DataFrame) -> pd.Series:
    """Días en que cambia la posición. En la salida publicada de SPY +4%/−3% la rampa hacia
    el 1x sube el peso un poco cada día: cuenta como una operación (la venta), no como una
    al día (el coste sí se cobra entero, 5 pb por unidad movida)."""
    s = w
    if "u" in w and ((w["u"] > 0) & (w["u"] < 1)).any():
        s = w.assign(u=(w["u"] > 0).astype(float).where(w["u"].notna()))
    return (s.diff().abs().sum(axis=1) > 1e-9).astype(float).where(w.notna().all(axis=1))


def build(a: dict, lag: int = LAG, rebal_start=None, uvxy: bool = False) -> Book:
    """Referencias (1x, 3x, LRS original), reglas de pesos y, con rebal_start y orden al
    cierre siguiente, las carteras con reajuste periódico (desde rebal_start)."""
    bk = Book(a["market"])
    idx = a["u"].index
    one = pd.Series(1.0, idx)
    zero = pd.Series(0.0, idx)
    bk.add(ONE, a["u"], one, zero, ref=True)
    bk.add(THREE, a["x3"], 3 * one, zero, ref=True)
    w = lrs(a)
    bk.add(ORIGINAL, run_rule(a, w, lag), exposure(w), trades_of(w), ref=True)
    for name, (w, main) in weight_rules(a, uvxy).items():
        bk.add(name, run_rule(a, w, lag), exposure(w), trades_of(w), main)
    if rebal_start is not None and lag >= 2:
        for name, fn in ((NINE, lambda: nine_sig(a, rebal_start, lag=lag)),
                         (HFEA55, lambda: hfea(a, 0.55, rebal_start, lag=lag)),
                         (HFEA40, lambda: hfea(a, 0.40, rebal_start, lag=lag))):
            r, wts, tr = fn()
            bk.add(name, r, 3 * wts["x3"], tr)
    return bk


def same_close(bk: Book) -> list[str]:
    return [k for k in bk.R if k not in (ONE, THREE)]


# ───────────────────────────── evaluación ─────────────────────────────

def frame(R: dict, keys, start=None, end=None) -> pd.DataFrame:
    """Rentabilidades de `keys` en la ventana común (desde la última primera sesión válida
    hasta la primera última), sin huecos."""
    df = pd.DataFrame({k: R[k] for k in keys}).loc[start:end]
    first = df.apply(lambda c: c.first_valid_index()).max()
    last = df.apply(lambda c: c.last_valid_index()).min()
    df = df.loc[first:last]
    holes = df.isna().sum()
    assert holes.sum() == 0, f"huecos en la ventana: {holes[holes > 0].to_dict()}"
    return df


def logs(df: pd.DataFrame) -> pd.DataFrame:
    return np.log1p(df.clip(lower=-0.9999))


def boot_means(X: np.ndarray, rng, B: int = B) -> np.ndarray:
    """Medias de las columnas de X en B remuestreos estacionarios (los mismos para todas)."""
    n = len(X)
    out = np.empty((B, X.shape[1]))
    for b in range(B):
        out[b] = X[stats.stationary_indices(n, BLOCK, rng)].mean(axis=0)
    return out


def delta_table(df: pd.DataFrame, benches: tuple, rng, skip=()) -> pd.DataFrame:
    """Δg (log anual) de cada columna frente a cada referencia, con intervalo percentil del
    90% y p unilateral (H0: Δg ≤ 0) con el error típico del bootstrap emparejado."""
    L = logs(df)
    bm = boot_means(L.to_numpy(float), rng)
    m = L.mean().to_numpy(float)
    cols = list(df.columns)
    out = {}
    for bench in benches:
        j = cols.index(bench)
        for i, c in enumerate(cols):
            if c == bench or c in skip:
                continue
            d = bm[:, i] - bm[:, j]
            se = d.std(ddof=1)
            mu = m[i] - m[j]
            lo, hi = np.percentile(d, [5, 95])
            tag = short_bench(bench)
            row = out.setdefault(c, {})
            row[f"Δg_{tag}"] = mu * 252
            row[f"IC90_inf_{tag}"] = lo * 252
            row[f"IC90_sup_{tag}"] = hi * 252
            row[f"p_{tag}"] = float(norm.sf(mu / se)) if se > 0 else np.nan
    return pd.DataFrame(out).T


def short_bench(bench: str) -> str:
    return {ORIGINAL: "LRS", ONE: "1x", THREE: "3x"}[bench]


def perf_table(bk: Book, df: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """Métricas de R13 (stats_table) en la ventana de df, más exposición y operaciones."""
    t = r13.stats_table({k: df[k] for k in df}, rf, df.index[0], df.index[-1])
    yrs = len(df) / 252
    for k in df:
        e = bk.E[k].reindex(df.index)
        t.loc[k, "exposición_media"] = e.mean()
        t.loc[k, "tiempo_invertido"] = (e > 0).mean()
        t.loc[k, "operaciones_año"] = bk.T[k].reindex(df.index).sum() / yrs
    return t[["CAGR", "vol", "sharpe", "maxDD", "peor_año", "exposición_media", "tiempo_invertido",
              "operaciones_año"]]


def unique_cols(df: pd.DataFrame, keys) -> tuple[list[str], dict[str, str]]:
    """Reglas de `keys` sin las que repiten exactamente otra en la ventana de df (cuentan una
    vez, como en r19.dedupe): (únicas, {repetida: la que se conserva})."""
    keep, dup = [], {}
    for k in keys:
        same = next((j for j in keep + [c for c in (ORIGINAL, ONE, THREE) if c in df]
                     if float((df[j] - df[k]).abs().max()) < 1e-12), None)
        if same is None:
            keep.append(k)
        else:
            dup[k] = same
    return keep, dup


def spa_family(df: pd.DataFrame, family: list[str], rng, label: str) -> list[dict]:
    """SPA de Hansen (versión consistente, con el recentrado de stats.spa_test) de la
    familia frente a la LRS original, al 1x y al 3x; frente al 1x y al 3x, la LRS original
    entra como una regla más. Las reglas idénticas a otra en la ventana cuentan una vez."""
    family, dup = unique_cols(df, family)
    L = logs(df)
    out = []
    for bench in (ORIGINAL, ONE, THREE):
        cc = [c for c in family + ([ORIGINAL] if bench != ORIGINAL else []) if c != bench and c in L]
        D = L[cc].sub(L[bench], axis=0).to_numpy(float)
        res = stats.spa_test(D, BLOCK, B, rng)
        best = int(D.mean(axis=0).argmax())
        out.append({"familia": label, "frente_a": bench, "reglas": len(cc), "mejor": cc[best],
                    "Δg_mejor": D.mean(axis=0)[best] * 252, "p_SPA_c": res["p_SPA_c"],
                    "desde": df.index[0].date(), "hasta": df.index[-1].date(),
                    "repetidas": "; ".join(f"{k} = {v}" for k, v in dup.items()) or "—"})
    return out


def pairwise_delta(R: dict, keys, bench: str, start, end, rng) -> pd.DataFrame:
    """Δg de cada regla frente a `bench` en [start, end], sobre los días en que las dos
    tienen dato (una regla con bonos empieza en 1962 aunque la ventana empiece antes)."""
    rows = {}
    groups: dict[tuple, list[str]] = {}
    for k in keys:
        both = pd.concat([R[k], R[bench]], axis=1).loc[start:end].dropna()
        if len(both) < 252:
            continue
        groups.setdefault((both.index[0], both.index[-1]), []).append(k)
    for (a_, b_), ks in groups.items():
        df = frame(R, ks + [bench], a_, b_)
        t = delta_table(df, (bench,), rng)
        tag = short_bench(bench)
        for k in ks:
            rows[k] = {"desde": df.index[0].date(), "años": len(df) / 252,
                       "CAGR": metrics.cagr(df[k]), f"CAGR_{tag}": metrics.cagr(df[bench]),
                       "Δg": t.loc[k, f"Δg_{tag}"], "p": t.loc[k, f"p_{tag}"]}
    return pd.DataFrame.from_dict(rows, orient="index")


def similarity(bk: Book, df: pd.DataFrame, keys) -> pd.DataFrame:
    """Parecido con la LRS original: correlación diaria, días en la misma posición (en bolsa
    sí/no) y correlación con el 3x; exposición media."""
    e_l = bk.E[ORIGINAL].reindex(df.index)
    rows = {}
    for k in keys:
        e = bk.E[k].reindex(df.index)
        rows[k] = {"clase": FICHA[k]["clase"], "corr_con_LRS": df[k].corr(df[ORIGINAL]),
                   "misma_posición_que_LRS": float(((e > 0) == (e_l > 0)).mean()),
                   "corr_con_3x": df[k].corr(df[THREE]), "exposición_media": e.mean()}
    return pd.DataFrame.from_dict(rows, orient="index")


# ───────────────────────────── DCA ─────────────────────────────

def irr(t_years: np.ndarray, tw: float) -> float:
    """Tasa anual i con Σ (1 + i)^t_c = tw (aportaciones de 1 $ que crecen t_c años)."""
    f = lambda i: np.sum((1 + i) ** t_years) - tw  # noqa: E731
    try:
        return float(brentq(f, -0.99, 5.0))
    except ValueError:
        return np.nan


def dca_windows(R: pd.DataFrame, years: int) -> pd.DataFrame:
    """Aportación de 1 $ al cierre de la primera sesión de cada mes durante `years` años;
    se valora al cierre de la sesión anterior al primer día del mes siguiente al último
    aporte. Por cada ventana (una por mes de inicio): múltiplo del DCA (riqueza final /
    aportado), su TIR y la rentabilidad anual de invertir la primera suma de golpe."""
    W_ = (1 + R).cumprod()
    idx = R.index
    firsts = pd.DatetimeIndex(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).min().values)
    n_m = years * 12
    rows = []
    for i0 in range(len(firsts) - n_m):
        contrib = firsts[i0:i0 + n_m]
        end_day = idx[idx.get_loc(firsts[i0 + n_m]) - 1]
        t_years = (end_day - contrib).days.to_numpy() / 365.25
        row = {"inicio": contrib[0], "fin": end_day}
        for c in R.columns:
            wT = W_[c].loc[end_day]
            tw = float(np.sum(wT / W_[c].loc[contrib].to_numpy()))
            row[f"{c}|múltiplo"] = tw / n_m
            row[f"{c}|TIR"] = irr(t_years, tw)
            row[f"{c}|única"] = (wT / W_[c].loc[contrib[0]]) ** (1 / t_years[0]) - 1
        rows.append(row)
    return pd.DataFrame(rows)


def dca_summary(win: pd.DataFrame, label: str, years: int) -> list[dict]:
    out = []
    m1 = win[f"{ONE}|múltiplo"]
    for c in (THREE, ORIGINAL):
        m = win[f"{c}|múltiplo"]
        ratio = m / m1
        after = win.inicio >= SPLIT
        out.append({"mercado": label, "años": years, "estrategia": "3x" if c == THREE else "LRS 3x (original)",
                    "ventanas": len(win), "primera": win.inicio.iloc[0].date(), "última": win.inicio.iloc[-1].date(),
                    "P(DCA pierde dinero)": float((m < 1).mean()), "P(DCA 1x pierde dinero)": float((m1 < 1).mean()),
                    "P(DCA por debajo del DCA 1x)": float((ratio < 1).mean()),
                    "P(suma única por debajo del 1x)": float((win[f"{c}|única"] < win[f"{ONE}|única"]).mean()),
                    "P(por debajo del 1x), inicios desde 2010": float((ratio[after] < 1).mean()) if after.any() else np.nan,
                    "TIR_mediana": float(win[f"{c}|TIR"].median()), "TIR_1x_mediana": float(win[f"{ONE}|TIR"].median()),
                    "TIR_P5": float(win[f"{c}|TIR"].quantile(0.05)), "TIR_1x_P5": float(win[f"{ONE}|TIR"].quantile(0.05)),
                    "peor_cociente_frente_1x": float(ratio.min()),
                    "inicio_peor": win.inicio.loc[ratio.idxmin()].date()})
    return out


# ───────────────────────────── sensibilidades ─────────────────────────────

def ndx_imputations(D: dict, keys: list[str]) -> pd.DataFrame:
    """CAGR y Δg frente a la LRS original de cada regla en las r13.N_NDX reconstrucciones
    del Nasdaq-100 (evaluado desde 1985-10; el pasado reconstruido solo entra en el
    calentamiento de las señales)."""
    nd, sp = D["_ndx"], D[SP]
    real = r12.real_letf("TQQQ")
    rows = []
    for k, u in enumerate(nd["draws"][:r13.N_NDX]):
        d = r12.make(u, history.rf_long(u.index), "TQQQ", real, eval_from=NDX_START)
        d["px"] = r13.price_index(d, NDX)
        a = assets(d, NDX, sp=sp)
        bk = build(a, rebal_start=NDX_START)
        df = frame(bk.R, [ONE, THREE, ORIGINAL] + keys, NDX_START)
        L = logs(df)
        for s in df:
            rows.append({"reconstrucción": k, "regla": s, "CAGR": metrics.cagr(df[s]),
                         "Δg_LRS": float((L[s] - L[ORIGINAL]).mean() * 252)})
    x = pd.DataFrame(rows)
    return x.groupby("regla", sort=False).agg(CAGR_mediana=("CAGR", "median"), CAGR_mín=("CAGR", "min"),
                                              CAGR_máx=("CAGR", "max"), Δg_LRS_mediana=("Δg_LRS", "median"),
                                              Δg_LRS_mín=("Δg_LRS", "min"), Δg_LRS_máx=("Δg_LRS", "max"))


def validation_tables(a_sp: dict, a_nd: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bonos sintéticos frente a TLT y AGG reales, y fondos sintéticos frente a los reales
    en la vida común (el sobrecoste s se calibra ahí, así que la media coincide por
    construcción: lo que valida es la correlación y la caída)."""
    idx = a_sp["u"].index
    rows = []
    for name, kind in (("TLT", "largo"), ("AGG", "medio")):
        rr = real_returns(name)
        s = bond_on(kind, idx).reindex(rr.index)
        both = pd.concat([rr, s], axis=1, keys=["real", "sint"]).dropna()
        mo = (1 + both).groupby([both.index.year, both.index.month]).prod() - 1
        rows.append({"fondo": name, "sintético": f"bono a la par de {LONG_M if kind == 'largo' else MID_M:g} años",
                     "desde": both.index[0].date(), "CAGR_real": metrics.cagr(both.real),
                     "CAGR_sint": metrics.cagr(both.sint), "vol_real": metrics.vol(both.real),
                     "vol_sint": metrics.vol(both.sint), "corr_diaria": both.corr().iloc[0, 1],
                     "corr_mensual": mo.corr().iloc[0, 1]})
    bonds = pd.DataFrame(rows).set_index("fondo")
    rows = []
    for a, t, s_key, base, L in ((a_sp, "TMF", "s_tmf", "tlt", 3.0), (a_sp, "SPXU", "s_inv", "u", -3.0),
                                 (a_nd, "SQQQ", "s_inv", "u", -3.0)):
        rr = real_returns(t)
        e = BY_TICKER[t]
        syn = lev_fund(a[base], a["rf"], a["f"], L, e.expense_ratio, a[s_key]).reindex(rr.index)
        both = pd.concat([rr, syn], axis=1, keys=["real", "sint"]).dropna()
        rows.append({"fondo": t, "desde": both.index[0].date(), "s_anual": a[s_key],
                     "CAGR_real": metrics.cagr(both.real), "CAGR_sint": metrics.cagr(both.sint),
                     "maxDD_real": metrics.max_dd(both.real), "maxDD_sint": metrics.max_dd(both.sint),
                     "corr_diaria": both.corr().iloc[0, 1]})
    return bonds, pd.DataFrame(rows).set_index("fondo")


def sanity(bk: Book, a: dict) -> tuple[pd.DataFrame, bool, float]:
    """R13 desde 1938-02-04: la LRS 3x (original), el 1x y el 3x deben dar las cifras del
    informe de R13 (leídas de reports/r13_lrs/README.md; si no se puede, SANITY). Además,
    run_multi con los pesos de la LRS debe dar exactamente r13.run_weights."""
    ref = dict(SANITY)
    path = REPORTS / "r13_lrs" / "README.md"
    if path.exists():
        txt = path.read_text()
        sec = txt.split("## S&P 500: desde 1938-02-04")[1].split("\n\n|")[1] if "## S&P 500: desde 1938-02-04" in txt else ""
        for k in ref:
            m = re.search(rf"\|\s*{re.escape(k)}\s*\|\s*([-\d.]+)%\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([-\d.]+)%", sec)
            if m:
                ref[k] = (float(m.group(1)) / 100, float(m.group(2)) / 100)
    t = r13.stats_table({k: bk.R[k] for k in ref}, a["rf"], SP_START)
    out = pd.DataFrame({"CAGR_R24": t.CAGR, "CAGR_R13": [ref[k][0] for k in t.index],
                        "maxDD_R24": t.maxDD, "maxDD_R13": [ref[k][1] for k in t.index]})
    ok = bool(((out.CAGR_R24 - out.CAGR_R13).abs() < 0.0006).all()
              and ((out.maxDD_R24 - out.maxDD_R13).abs() < 0.0006).all())
    w = lrs(a)
    diff = float((run_multi(a, W(a, x3=w["x3"], u=0 * w["x3"])) - bk.R[ORIGINAL]).abs().max())
    return out, ok and diff < 1e-12, diff


def lstar_rows(d: dict, market: str, windows) -> pd.DataFrame:
    rows = []
    for lab, a_, b_ in windows:
        t = kelly.lstar_table(d, a_, b_)
        rows.append({"mercado": market, "ventana": lab, "desde": t["desde"], "hasta": t["hasta"],
                     kelly.LSTAR_APROX: t[kelly.LSTAR_APROX],
                     kelly.LSTAR_EXACTO: kelly.fmt_lstar(t[kelly.LSTAR_EXACTO]),
                     kelly.LSTAR_EXACTO_SIN: kelly.fmt_lstar(t[kelly.LSTAR_EXACTO_SIN])})
    return pd.DataFrame(rows)


# ───────────────────────────── censo ─────────────────────────────

def census_returns(D: dict | None = None, same_close_rules: bool = True, price_signal: bool = True,
                   bonds: bool = True) -> dict[str, pd.Series]:
    """Rentabilidad diaria neta, en el S&P 500, de cada regla probada en R24, para añadirla
    al censo de R19: {"R24 · <regla>": serie}, en el índice de D["S&P 500"]["u"] (el del
    censo), con el protocolo común (orden al cierre siguiente, 5 pb, señal de rentabilidad
    total). Sin la LRS original, el 1x y el 3x (ya están en el censo) ni el DCA en el 3x (su
    serie es la del 3x). same_close_rules añade las variantes «· mismo cierre» (r19.
    is_same_close las reconoce); price_signal, las «· señal de precio»; bonds, las reglas
    con bonos (9Sig, HFEA y FTLT), que empiezan en 1962: con ellas la ventana común del
    censo empezaría en 1962."""
    D = r13.load() if D is None else D
    a = assets(D[SP], SP)
    bk = build(a, rebal_start=BOND_START if bonds else None)
    out = {}
    for k in bk.main + bk.sens:
        if (not price_signal and k.endswith(PX)) or (not bonds and k in (FTLT_L, NINE, HFEA55, HFEA40)):
            continue
        out[f"R24 · {k}"] = bk.R[k]
    if same_close_rules:
        bm = build(a, lag=1)
        for k in bm.main + bm.sens:
            if (not price_signal and k.endswith(PX)) or (not bonds and k == FTLT_L):
                continue
            out[f"R24 · {k}{SC}"] = bm.R[k]
    idx = D[SP]["u"].index
    return {k: v.reindex(idx) for k, v in out.items()}


def census_meta(names) -> dict[str, dict]:
    """Ficha de cada serie de census_returns con los campos de r19.Census.add."""
    out = {}
    for n in names:
        base = n.removeprefix("R24 · ").replace(SC, "").replace(PX, "")
        cls = FICHA[base]["clase"]
        out[n] = {"módulo": "R24", "familia": f"R24: {cls}",
                  "grupo": "LRS 3x: parámetros y ejecución" if cls == C_LRS else "Otras reglas y referencias",
                  "parametrización_LRS3x": cls == C_LRS}
    return out


# ───────────────────────────── figuras ─────────────────────────────

def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def class_handles():
    return [Line2D([], [], marker="o", ls="", color=c, markersize=9, label=k) for k, c in CLASS_COLOR.items()]


def fig_forest(panels: list[tuple[str, pd.DataFrame]], rep: Report):
    """Δg frente a la LRS original con su intervalo del 90%, una fila por regla."""
    n = [len(t) for _, t in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(FIG_W, 1.3 + 0.36 * sum(n) + 0.9 * len(panels)),
                             gridspec_kw={"height_ratios": n})
    lim = max(max(t.IC90_sup.abs().max(), t.IC90_inf.abs().max()) for _, t in panels) * 100 + 1
    for ax, (title, t) in zip(np.atleast_1d(axes), panels):
        t = t.iloc[::-1]
        y = np.arange(len(t))
        for yi, (k, row) in zip(y, t.iterrows()):
            col = CLASS_COLOR[row.clase]
            ax.plot([row.IC90_inf * 100, row.IC90_sup * 100], [yi, yi], color=col, lw=2, solid_capstyle="round")
            ax.plot(row.Δg * 100, yi, "o", color=col, markersize=8, markeredgecolor="white", markeredgewidth=1.5)
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(y, t.etiqueta)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.6, len(t) - 0.4)
        ax.set_title(title, color=INK)
        _style(ax)
    np.atleast_1d(axes)[-1].set_xlabel("Δg frente a la LRS original\n(puntos de crecimiento log al año)")
    legend_band(fig, class_handles(), list(CLASS_COLOR), ncol=2)
    fig.suptitle("Diferencia de crecimiento de cada regla frente a la LRS original\n"
                 "(punto: estimación; barra: intervalo del 90%)")
    comma_fig(fig)
    rep.fig(fig, "frente_a_lrs", "Diferencia de crecimiento logarítmico anual de cada regla frente a la LRS 3x "
                                 "original, con su intervalo del 90% por bootstrap estacionario emparejado")


def fig_2010(panels: list[tuple[str, pd.DataFrame]], rep: Report):
    """Δg frente a la LRS antes y después del 12-2-2010, una fila por regla."""
    n = [len(t) for _, t in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(FIG_W, 1.4 + 0.36 * sum(n) + 0.9 * len(panels)),
                             gridspec_kw={"height_ratios": n})
    for ax, (title, t) in zip(np.atleast_1d(axes), panels):
        t = t.iloc[::-1]
        y = np.arange(len(t))
        for yi, (k, row) in zip(y, t.iterrows()):
            col = CLASS_COLOR[row.clase]
            ax.plot([row.antes * 100, row.después * 100], [yi, yi], color=col, lw=1.6, alpha=0.7)
            ax.plot(row.antes * 100, yi, "o", markerfacecolor="white", markeredgecolor=col, markersize=8,
                    markeredgewidth=2)
            ax.plot(row.después * 100, yi, "o", color=col, markersize=8, markeredgecolor="white",
                    markeredgewidth=1.5)
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(y, t.etiqueta)
        ax.set_ylim(-0.6, len(t) - 0.4)
        ax.set_title(title, color=INK)
        _style(ax)
    np.atleast_1d(axes)[-1].set_xlabel("Δg frente a la LRS original\n(puntos de crecimiento log al año)")
    h = [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=MUTED, markeredgewidth=2,
                markersize=9, label="hasta el 11-2-2010"),
         Line2D([], [], marker="o", ls="", color=MUTED, markersize=9, label="desde el 12-2-2010")]
    hs = h + class_handles()
    legend_band(fig, hs, [x.get_label() for x in hs], ncol=3)
    fig.suptitle("Frente a la LRS original, antes y después de 2010")
    comma_fig(fig)
    rep.fig(fig, "antes_despues_2010", "Diferencia de crecimiento frente a la LRS original de cada regla hasta el "
                                       "11-2-2010 (círculo hueco) y desde el 12-2-2010 (círculo lleno)")


def fig_ftlt(tabs: list[tuple[str, pd.Series]], rep: Report):
    """Rentabilidad anual de FTLT según la ejecución y el destino de la rama UVXY."""
    fig, axes = plt.subplots(len(tabs), 1, figsize=(FIG_W, 1.6 + 0.42 * sum(len(s) for _, s in tabs)),
                             gridspec_kw={"height_ratios": [len(s) for _, s in tabs]})
    for ax, (title, s) in zip(np.atleast_1d(axes), tabs):
        s = s.iloc[::-1]
        y = np.arange(len(s))
        cols = [CLASS_COLOR[C_REV] if k.startswith("FTLT") else "#9a9994" for k in s.index]
        ax.barh(y, s.values * 100, color=cols, height=0.62)
        for yi, v in zip(y, s.values):
            ax.text(v * 100 + (1 if v >= 0 else -1), yi, pct_es(v), va="center",
                    ha="left" if v >= 0 else "right", color=INK, fontsize=11)
        ax.set_yticks(y, s.index)
        ax.set_xlim(min(0, s.min() * 100 - 10), s.max() * 100 * 1.22)
        ax.axvline(0, color=INK, lw=1)
        ax.set_title(title, loc="left", color=INK)
        _style(ax)
    np.atleast_1d(axes)[-1].set_xlabel("Rentabilidad anual compuesta (%)")
    fig.suptitle("FTLT con ETFs reales: cuánto aportan UVXY y operar en el mismo cierre")
    comma_fig(fig)
    rep.fig(fig, "ftlt", "Rentabilidad anual de FTLT con ETFs reales según la ejecución (mismo cierre o cierre "
                         "siguiente) y el destino de su rama UVXY, frente a sus referencias")


def fig_dca(wins: list[tuple[str, pd.DataFrame]], rep: Report):
    """Cociente entre la riqueza final del DCA en el 3x (y en la LRS) y la del DCA en el 1x."""
    fig, axes = plt.subplots(len(wins), 1, figsize=(FIG_W, 3.4 * len(wins) + 1.2))
    for ax, (title, w) in zip(np.atleast_1d(axes), wins):
        m1 = w[f"{ONE}|múltiplo"]
        for c, col, lab in ((THREE, "#eb6834", "DCA en el 3x"), (ORIGINAL, "#2a78d6", "DCA en la LRS 3x original")):
            ax.semilogy(w.inicio, w[f"{c}|múltiplo"] / m1, color=col, lw=2, label=lab)
        ax.axhline(1, color=INK, lw=1)
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylabel("Riqueza final / DCA en el 1x")
        ticks = [v for v in (0.125, 0.25, 0.5, 1, 2, 4, 8, 16) if ax.get_ylim()[0] <= v <= ax.get_ylim()[1]]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num_es(v, 3 if v < 0.25 else 2 if v < 1 else 0)))
        _style(ax)
        ax.grid(axis="y", color=GRID, lw=0.6)
    np.atleast_1d(axes)[-1].set_xlabel("Mes de la primera aportación (ventanas de 10 años)")
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    legend_band(fig, h, lab, ncol=2)
    fig.suptitle("Aportar cada mes durante 10 años: riqueza final frente a aportar al 1x")
    comma_fig(fig)
    rep.fig(fig, "dca", "Cociente entre la riqueza final de aportar cada mes al 3x (o a la LRS original) y la de "
                        "aportar al 1x, en ventanas de 10 años según el mes de inicio")


# ───────────────────────────── informe ─────────────────────────────

PCTS = {"CAGR", "vol", "maxDD", "peor_año", "tiempo_invertido"}


def flat(df: pd.DataFrame) -> pd.DataFrame:
    """Índice múltiple → columnas (para que la tabla no muestre tuplas)."""
    return df.reset_index() if isinstance(df.index, pd.MultiIndex) else df


def attribution(R: dict, keys, bench: str, start, end=None, top: int = 3) -> pd.DataFrame:
    """De dónde sale la Δg frente a `bench`: suma de la diferencia de log-rentabilidades por
    año natural, total y los `top` años que más aportan (con su parte del total)."""
    rows = []
    for k in keys:
        df = frame(R, [k, bench], start, end)
        L = logs(df)
        by = (L[k] - L[bench]).groupby(df.index.year).sum()
        tot = float(by.sum())
        best = by.sort_values(ascending=tot < 0).iloc[:top]
        rows.append({"regla": k, "desde": df.index[0].date(), "Δg_anual": tot / (len(df) / 252),
                     "diferencia_log_total": tot,
                     "años_que_más_aportan": ", ".join(f"{y} ({num_es(v, 2)})" for y, v in best.items()),
                     "parte_de_esos_años": float(best.sum() / tot) if tot else np.nan})
    return pd.DataFrame(rows).set_index("regla")


def pp(x: float, k: int = 1) -> str:
    """Puntos porcentuales con signo: 0,012 → «+1,2 pp»."""
    return ("+" if x > 0 else "") + num_es(100 * x, k) + " pp"


def fp(p: float) -> str:
    return "< 0,001" if p < 0.001 else num_es(p, 3)


def pp_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype(float) * 100
    return out.rename(columns={c: f"{c} (pp)" for c in cols})


def _date(x) -> str:
    return pd.Timestamp(x).strftime("%-d-%-m-%Y")


def label(k: str, bond_since: bool = False) -> str:
    return k + (" (desde 1962)" if bond_since else "")


def run():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rep = Report("r24_folk", "R24 — Las estrategias de los foros, a examen")
    for f in rep.dir.glob("table_*.csv"):       # Report numera las tablas por línea: sin huérfanas
        f.unlink()
    rep.p(__doc__.split("\n", 2)[2])
    D = r13.load()
    sp_d, nd_d = D[SP], D[NDX]
    a_sp = assets(sp_d, SP)
    a_nd = assets(nd_d, NDX, sp=sp_d)

    # ── libros: S&P 500 (reajustes desde 1962), Nasdaq-100 (desde 1985), ETFs reales (desde 2010)
    bk_sp = build(a_sp, rebal_start=BOND_START)
    bk_nd = build(a_nd, rebal_start=NDX_START)
    bk_sp_r = build(a_sp, rebal_start=REAL_FROM)
    bk_nd_r = build(a_nd, rebal_start=REAL_FROM, uvxy=True)
    sc_sp, sc_nd = build(a_sp, lag=1), build(a_nd, lag=1, uvxy=True)

    # ── 0. cordura
    san, ok, diff = sanity(bk_sp, a_sp)
    if not ok:
        raise RuntimeError(f"R24: no se reproduce R13 (diferencia run_multi {diff:.2e}):\n{san}")
    SUM = []            # frases del resumen, se escriben al final (arriba del informe)
    body_start = len(rep.lines)

    rep.h("0. Cordura: la base es la de R13")
    rep.p("**Tabla 0.** S&P 500 desde el 4-2-1938, orden al cierre siguiente: R24 frente al informe de R13.")
    rep.table(san, pct=set(san.columns), digits=2)
    rep.p(f"Las tres series coinciden con las del informe de R13 (tabla «S&P 500: desde 1938-02-04»). El motor de "
          f"varios fondos de este informe (run_multi) da, con los pesos de la LRS, exactamente la serie de "
          f"r13.run_weights (diferencia máxima {diff:.0e}).")

    # ── 1. ficha
    rep.h("1. Ficha de las reglas")
    ficha = pd.DataFrame({k: {"clase": v["clase"], "regla publicada": v["regla"], "fuente": v["fuente"],
                              "fecha": v["fecha"], "implementación en R24": v["impl"]}
                          for k, v in FICHA.items() if k not in (PUB6, PUB12, PUBG)}).T
    ficha.index.name = "regla"
    rep.p("**Tabla 1.** Regla publicada, fuente, fecha, clase e implementación. Las fuentes se leyeron en una "
          "exploración previa (sin acceso a la web en esta ejecución); las fechas marcadas «c.» son aproximadas.")
    rep.table(ficha)
    rep.p("Clases. **LRS reempaquetada**: la misma idea que Gayed y Bilello (fuera del mercado bajo una media de "
          "200 sesiones) con otra banda, otra señal u otro índice. **3x con adornos**: casi siempre en el 3x, con "
          "reglas que cambian poco la exposición (o, en las reglas de caídas, un 1x que se apalanca tras una caída: "
          "la clase más cercana, aunque no es de corto plazo). **Paridad de riesgo**: 3x de acciones con 3x de "
          "bonos largos. **Reversión a corto**: reglas de RSI que compran tras bajadas de días y venden tras "
          "subidas.")

    # ── 2. datos añadidos
    rep.h("2. Fondos que no están en la base de R13")
    vb, vf = validation_tables(a_sp, a_nd)
    rep.p("**Tabla 2a.** Bonos sintéticos frente a los ETFs reales en su vida común.")
    rep.table(vb, pct={"CAGR_real", "CAGR_sint", "vol_real", "vol_sint"}, digits=3)
    rep.p("**Tabla 2b.** Fondos apalancados sintéticos frente a los reales en su vida común (s: sobrecoste "
          "anual calibrado sobre f).")
    rep.table(vf, pct={"s_anual", "CAGR_real", "CAGR_sint", "maxDD_real", "maxDD_sint"}, digits=3)
    rep.p("El vencimiento de 25 años del bono largo es el medio del índice que replica TLT (ICE 20+), no un "
          "ajuste; el proxy de AGG (Tesoro a 5 años) no recoge la prima de crédito ni la hipotecaria de AGG y su "
          "correlación diaria es menor. El sobrecoste s de TMF, SPXU y SQQQ se calibra en la vida del fondo real, "
          "así que la rentabilidad media coincide por construcción; la correlación y la caída máxima sí validan. "
          "Los rendimientos diarios del Tesoro empiezan en 1962: las reglas con bonos (9Sig, HFEA y FTLT) solo se "
          "evalúan desde entonces en el S&P 500.")

    # ── 3–5. ventanas completas
    fams, spa_rows, count_rows = {}, [], []
    sp_nb = [k for k in bk_sp.main if k not in (FTLT_L, NINE, HFEA55, HFEA40)]
    sp_b = list(bk_sp.main)
    nd_m = list(bk_nd.main)
    nd_rm = [k for k in bk_nd_r.main if k not in (FTLT_X, RSI79U)]
    sp_rm = list(bk_sp_r.main)
    nd_ux = list(bk_nd_r.main)
    REFS = [ONE, THREE, ORIGINAL]
    bases = [("A", "S&P 500 desde 1938 (reglas sin bonos)", bk_sp, sp_nb, SP_START, a_sp),
             ("B", "S&P 500 desde 1962 (todas)", bk_sp, sp_b, BOND_START, a_sp),
             ("N", "Nasdaq-100 desde 1985", bk_nd, nd_m, NDX_START, a_nd),
             ("RS", "S&P 500 con ETFs reales (desde 2010)", bk_sp_r, sp_rm, REAL_FROM, a_sp),
             ("RN", "Nasdaq-100 con ETFs reales (desde 2010)", bk_nd_r, nd_rm, REAL_FROM, a_nd),
             ("UX", "Nasdaq-100 con ETFs reales y UVXY (desde 2011-10)", bk_nd_r, nd_ux,
              a_nd["uvxy"].first_valid_index(), a_nd)]
    INF = {}
    sec = 3
    for code, title, bk, fam, start, a in bases:
        df = frame(bk.R, REFS + fam, start)
        perf = perf_table(bk, df, a["rf"])
        inf_all = delta_table(df, (ORIGINAL, ONE), rng_for(f"delta {code}"), skip=(ONE,))
        inf = inf_all.loc[list(fam)].copy()
        for tag in ("LRS", "1x"):
            inf[f"p_Holm_{tag}"] = stats.holm(inf[f"p_{tag}"].to_numpy(float))
        INF[code] = (df, perf, inf)
        inf_show = pd.concat([inf_all.loc[[THREE, ORIGINAL]], inf])
        fams[code] = fam
        spa_rows += spa_family(df, fam, rng_for(f"spa {code}"), title)
        count_rows.append({"ventana": title, "reglas_familia": len(unique_cols(df, fam)[0])})
        if code != "B":
            rep.h(f"{sec}. {title}")
            sec += 1
        else:
            rep.h(title, 3)
        rep.p(f"**Tabla {code}1.** Rentabilidad y riesgo, del {_date(df.index[0])} al {_date(df.index[-1])} "
              f"({num_es(len(df) / 252, 1)} años), orden al cierre siguiente. exposición_media: veces el índice "
              "(3 = un 3x entero); tiempo_invertido: días con exposición positiva.")
        rep.table(perf, pct=PCTS, digits=2)
        dcols = [c for c in inf_show.columns if c.startswith(("Δg", "IC90"))]
        show = pd.concat([pp_cols(inf_show[dcols], dcols),
                          inf_show[[c for c in inf_show.columns if c.startswith("p_")]]], axis=1)
        rep.p(f"**Tabla {code}2.** Δg frente a la LRS 3x (original) y frente al 1x: diferencia de crecimiento "
              "logarítmico anual, intervalo del 90% y p unilateral (H0: Δg ≤ 0) con bootstrap estacionario "
              f"emparejado; p_Holm corrige las {len(fam)} reglas de la familia.")
        rep.table(show, digits=3)

    # ── similitud con la LRS
    rep.h(f"{sec}. Cuáles son la LRS con otro nombre")
    sec += 1
    sim_sp = similarity(bk_sp, INF["B"][0], sp_b)
    sim_nd = similarity(bk_nd, INF["N"][0], nd_m)
    rep.p("**Tabla S1.** Parecido de cada regla con la LRS 3x (original) en el S&P 500 desde 1962: correlación de "
          "las rentabilidades diarias, fracción de días en la misma posición (en bolsa o fuera) y correlación con "
          "el 3x comprar y mantener.")
    rep.table(sim_sp, pct={"misma_posición_que_LRS"}, digits=2)
    rep.p("**Tabla S2.** Lo mismo en el Nasdaq-100 desde 1985.")
    rep.table(sim_nd, pct={"misma_posición_que_LRS"}, digits=2)

    # ── subperiodos y antes / después de 2010
    rep.h(f"{sec}. Subperiodos: antes y después de 2010 y tras la publicación")
    sec += 1
    SP_W = [("1938–1973", SP_START, "1973-12-31"), ("1974–1999", "1974-01-01", "1999-12-31"),
            ("2000–2026", "2000-01-01", None), ("hasta el 11-2-2010", SP_START, SPLIT - pd.Timedelta(days=1)),
            ("desde el 12-2-2010", SPLIT, None), ("2016–2026", POST_GB, None)]
    ND_W = [("1985–1999", NDX_START, "1999-12-31"), ("2000–2026", "2000-01-01", None),
            ("hasta el 11-2-2010", NDX_START, SPLIT - pd.Timedelta(days=1)), ("desde el 12-2-2010", SPLIT, None),
            ("2016–2026", POST_GB, None)]
    SUB = {}
    for mk, bk, keys, wins in ((SP, bk_sp, sp_b, SP_W), (NDX, bk_nd, nd_m, ND_W)):
        dg, pv, cg = {}, {}, {}
        for lab, a_, b_ in wins:
            t = pairwise_delta(bk.R, keys, ORIGINAL, a_, b_, rng_for(f"sub {mk} {lab}"))
            dg[lab], pv[lab], cg[lab] = t["Δg"], t["p"], t["CAGR"]
            lrs_c = metrics.cagr(bk.R[ORIGINAL].loc[a_:b_].dropna())
            head = {ORIGINAL: lrs_c}
            if mk == SP:     # la misma LRS en la ventana de las reglas con bonos (desde 1962)
                head[LRS62] = (metrics.cagr(bk.R[ORIGINAL].loc[BOND_START:b_].dropna())
                               if pd.Timestamp(a_) < pd.Timestamp(BOND_START) else np.nan)
            cg[lab] = pd.concat([pd.Series(head), cg[lab]])
        SUB[mk] = (pd.DataFrame(dg), pd.DataFrame(pv), pd.DataFrame(cg))
        note = (" Las reglas con bonos (9Sig, HFEA, FTLT) empiezan en 1962: en la columna 1938–1973 y en «hasta el "
                "11-2-2010» su ventana empieza ese año. Para compararlas, la segunda fila es la misma LRS desde el "
                "1-2-1962 en esas columnas (las Δg de las tablas siguientes son siempre sobre los mismos días)."
                if mk == SP else "")
        rep.p(f"**Tabla {mk}-a.** {mk}: rentabilidad anual compuesta por subperiodo (la primera fila, la LRS 3x "
              f"original).{note}")
        rep.table(SUB[mk][2], pct=set(SUB[mk][2].columns), digits=2)
        rep.p(f"**Tabla {mk}-b.** {mk}: Δg frente a la LRS 3x (original) por subperiodo, en puntos de crecimiento "
              "log al año.")
        rep.table(SUB[mk][0] * 100, digits=2)
        rep.p(f"**Tabla {mk}-c.** {mk}: p unilateral de esa Δg (sin corregir por la búsqueda).")
        rep.table(SUB[mk][1], digits=3)

    ATT = {}
    for mk, bk, keys, start in ((SP, bk_sp, sp_b, BOND_START), (NDX, bk_nd, nd_m, NDX_START)):
        t = INF["B" if mk == SP else "N"][2]
        big = [k for k in keys if t.loc[k, "Δg_LRS"] > 0.01]
        ATT[mk] = attribution(bk.R, big, ORIGINAL, start)
        rep.p(f"**Tabla {mk}-d.** {mk}: de dónde sale la ventaja de las reglas que superan a la LRS original en más de "
              "1 punto al año en la ventana completa: diferencia de log-rentabilidades acumulada por año natural y "
              "los tres años que más aportan (entre paréntesis, su diferencia log).")
        rep.table(ATT[mk], pct={"Δg_anual", "parte_de_esos_años"}, digits=2)

    # ── tras la publicación de cada regla
    rep.h(f"{sec}. Tras la publicación de cada regla: el fuera de muestra natural")
    sec += 1
    post = []
    rp = rng_for("post")
    for mk, a, bk in ((NDX, a_nd, bk_nd_r), (SP, a_sp, bk_sp_r)):
        for k, v in FICHA.items():
            if v["pub"] is None or k in (PUB6, PUB12, PUBG) or (mk == SP and k in (SPYSIG, FTLT_X, RSI79U)):
                continue
            since = pd.Timestamp(v["pub"])
            if k in (NINE, HFEA55, HFEA40):
                r = (nine_sig(a, since) if k == NINE else hfea(a, 0.55 if k == HFEA55 else 0.40, since))[0]
            else:
                r = bk.R[k]
            R = {k: r, ORIGINAL: bk.R[ORIGINAL], ONE: a["u"], THREE: a["x3"]}
            df = frame(R, [k, ORIGINAL, ONE, THREE], since)
            t = delta_table(df, (ORIGINAL, THREE), rp, skip=(ONE,))
            post.append({"regla": k, "mercado": mk, "publicada": v["fecha"], "desde": df.index[0].date(),
                         "años": len(df) / 252, "CAGR": metrics.cagr(df[k]),
                         "CAGR_LRS": metrics.cagr(df[ORIGINAL]), "CAGR_1x": metrics.cagr(df[ONE]),
                         "CAGR_3x": metrics.cagr(df[THREE]), "maxDD": metrics.max_dd(df[k]),
                         "Δg_LRS": t.loc[k, "Δg_LRS"], "p_LRS": t.loc[k, "p_LRS"], "Δg_3x": t.loc[k, "Δg_3x"]})
    for mk, a in ((NDX, a_nd), (SP, a_sp)):
        df = frame({ORIGINAL: (bk_nd_r if mk == NDX else bk_sp_r).R[ORIGINAL], ONE: a["u"], THREE: a["x3"]},
                   [ORIGINAL, ONE, THREE], POST_GB)
        t = delta_table(df, (ONE, THREE), rp)
        post.append({"regla": ORIGINAL, "mercado": mk, "publicada": "2016 (SSRN); muestra hasta 10-2015",
                     "desde": df.index[0].date(), "años": len(df) / 252, "CAGR": metrics.cagr(df[ORIGINAL]),
                     "CAGR_LRS": metrics.cagr(df[ORIGINAL]), "CAGR_1x": metrics.cagr(df[ONE]),
                     "CAGR_3x": metrics.cagr(df[THREE]), "maxDD": metrics.max_dd(df[ORIGINAL]),
                     "Δg_LRS": 0.0, "p_LRS": np.nan, "Δg_3x": t.loc[ORIGINAL, "Δg_3x"]})
    post = pd.DataFrame(post).set_index(["regla", "mercado"])
    rep.p("**Tabla P.** Desde la primera sesión tras la publicación (columna «desde»), con ETFs reales. Las "
          "carteras con reajuste (9Sig, HFEA) compran al cierre anterior y cobran desde ese día; las reglas de señal "
          "traen su estado de antes. Δg y p "
          "frente a la LRS 3x (original) en los mismos días; Δg_3x, frente al 3x comprar y mantener.")
    rep.table(flat(pp_cols(post, ["Δg_LRS", "Δg_3x"])), pct={"CAGR", "CAGR_LRS", "CAGR_1x", "CAGR_3x", "maxDD"},
              index=False, digits=2)
    rep.p("Casi todas las fechas son recientes: con menos de tres años (buena parte de la tabla) la diferencia "
          "con la LRS es ruido, y la tabla solo sirve para ver si la regla siguió haciendo lo que prometía.")

    # ── FTLT
    rep.h(f"{sec}. FTLT: UVXY y la hora de ejecución")
    sec += 1
    ux0 = a_nd["uvxy"].first_valid_index()
    wx = ftlt(a_nd, exact=True)
    variants = {"FTLT original, mismo cierre": run_multi(a_nd, wx, 1),
                "FTLT original, cierre siguiente": run_multi(a_nd, wx, LAG),
                "FTLT, rama UVXY → letras": run_multi(a_nd, ftlt(a_nd, True, "cash"), LAG),
                "FTLT, rama UVXY → TQQQ": run_multi(a_nd, ftlt(a_nd, True, "x3"), LAG),
                "FTLT versión larga (BIL)": bk_nd_r.R[FTLT_L],
                "LRS 3x (original)": bk_nd_r.R[ORIGINAL], "TQQQ (3x)": a_nd["x3"], "QQQ (1x)": a_nd["u"]}
    ft_rows, ft_bars, ft_years = [], [], {}
    for lab, since in (("desde octubre de 2011 (UVXY)", ux0), ("tras la última edición (24-1-2023), desde el 25-1-2023", "2023-01-25")):
        df = frame(variants, list(variants), since)
        # desde 2011-10, los mismos remuestreos que la tabla UX2 (misma ventana): la misma p para FTLT
        t = delta_table(df, ("LRS 3x (original)",), rng_for("delta UX" if since == ux0 else f"ftlt {lab}"))
        for k in df:
            ft_rows.append({"ventana": lab, "serie": k, "desde": df.index[0].date(), "CAGR": metrics.cagr(df[k]),
                            "maxDD": metrics.max_dd(df[k]),
                            "Δg_LRS": t.loc[k, "Δg_LRS"] if k in t.index else 0.0,
                            "p_LRS": t.loc[k, "p_LRS"] if k in t.index else np.nan})
        ft_bars.append((f"FTLT y referencias, {lab}", df.apply(metrics.cagr)))
        ft_years[lab] = len(df) / 252
    ft = pd.DataFrame(ft_rows).set_index(["ventana", "serie"])
    rep.p("**Tabla F1.** FTLT con ETFs reales en el Nasdaq-100: el árbol exacto en el mismo cierre y en el "
          "siguiente, con la rama UVXY cambiada por letras o por TQQQ, y la versión larga. Δg y p frente a la LRS 3x "
          "(original), sin corregir por la búsqueda.")
    rep.table(flat(pp_cols(ft, ["Δg_LRS"])), pct={"CAGR", "maxDD"}, index=False, digits=3)
    held = wx["uvxy"].shift(LAG).loc[ux0:] > 0
    ret_u = a_nd["uvxy"].loc[ux0:][held]
    lr = np.log1p(ret_u.dropna()).sort_values(ascending=False)
    r_ex = variants["FTLT original, cierre siguiente"].copy()
    for day in lr.index[:5]:
        r_ex.loc[day] = a_nd["rf"].loc[day]
    uv = {"días en UVXY": int(held.sum()), "fracción de días": float(held.mean()),
          "suma log de UVXY en esos días": float(lr.sum()), "de ella, los 5 mejores días": float(lr.iloc[:5].sum()),
          "CAGR con UVXY": metrics.cagr(variants["FTLT original, cierre siguiente"].loc[ux0:].dropna()),
          "CAGR sin los 5 mejores días de UVXY (letras)": metrics.cagr(r_ex.loc[ux0:].dropna())}
    top5 = ", ".join(f"{_date(k)} ({pct_es(np.expm1(v), 0)})" for k, v in lr.iloc[:5].items())
    rep.p("**Tabla F2.** Los días en UVXY del árbol exacto (orden al cierre siguiente), desde octubre de 2011.")
    rep.table(pd.Series(uv).to_frame("valor"), digits=3)
    rep.p(f"Los cinco mejores días de UVXY con la regla dentro: {top5}.")
    comp = []
    for k, since, c_cagr, c_cum, c_dd in COMPOSER:
        w = {FTLT_X: wx, FTLT_L: ftlt(a_nd, exact=False), RSI79U: rsi79_uvxy(a_nd)}[k]
        for lag_, cost, ex in ((1, 0.0, "mismo cierre, sin costes (como Composer)"), (1, COST_BPS, "mismo cierre, 5 pb"),
                               (LAG, COST_BPS, "cierre siguiente, 5 pb (laboratorio)")):
            r = run_multi(a_nd, w, lag_, cost).loc[since:].iloc[1:]
            wlt = (1 + r).cumprod()
            comp.append({"regla": k, "desde": since, "ejecución": ex, "CAGR": metrics.cagr(r),
                         "acumulada": float(wlt.iloc[-1] - 1), "maxDD": metrics.max_dd(r), "CAGR_Composer": c_cagr,
                         "acumulada_Composer": c_cum, "maxDD_Composer": c_dd})
    comp = pd.DataFrame(comp).set_index(["regla", "ejecución"])
    rep.p("**Tabla F3.** Réplica de las cifras que Composer publica desde la última edición de cada regla (su "
          "fuera de muestra; leídas en su web durante la exploración previa, no verificadas en esta ejecución). "
          "Composer opera en el cierre de la señal y sin costes; su primer día compra y la rentabilidad cuenta desde "
          "el siguiente.")
    rep.table(flat(comp), index=False,
              pct={"CAGR", "acumulada", "maxDD", "CAGR_Composer", "acumulada_Composer", "maxDD_Composer"},
              digits=3)
    ex_rows = []
    for mk, a, bk, bm in ((NDX, a_nd, bk_nd, sc_nd), (SP, a_sp, bk_sp, sc_sp)):
        u = a["u"]
        for a_, b_, lab in ((NDX_START if mk == NDX else "1962-02-01", "1999-12-31", None),
                            ("2000-01-01", "2009-12-31", "2000–2009"), ("2010-01-01", None, "2010–2026")):
            lab = lab or f"{pd.Timestamp(a_).year}–1999"
            row = {"mercado": mk, "tramo": lab, "autocorr_diaria_índice": float(u.loc[a_:b_].autocorr(1))}
            for k in (ORIGINAL, FTLT_L, CONNORS, RSI79):
                row[f"{k} · siguiente"] = metrics.cagr(bk.R[k].loc[a_:b_].dropna())
                row[f"{k} · mismo"] = metrics.cagr(bm.R[k].loc[a_:b_].dropna())
            ex_rows.append(row)
    ex = pd.DataFrame(ex_rows).set_index(["mercado", "tramo"])
    rep.p("**Tabla F4.** Rentabilidad anual con orden al cierre siguiente y en el mismo cierre, por tramo, y "
          "autocorrelación diaria del índice (negativa: el día siguiente tiende a revertir). En el S&P 500 el "
          "primer tramo empieza en 1962 (FTLT usa bonos).")
    rep.table(flat(ex), pct={c for c in ex.columns if c != "autocorr_diaria_índice"}, index=False, digits=3)

    # ── DCA
    rep.h(f"{sec}. DCA en un 3x")
    sec += 1
    dca_rows, dca_fig = [], []
    for lab, bk, start in (("S&P 500 desde 1938", bk_sp, SP_START), ("Nasdaq-100 desde 1985", bk_nd, NDX_START),
                           ("S&P 500, ETFs reales", bk_sp_r, REAL_FROM), ("Nasdaq-100, ETFs reales", bk_nd_r, REAL_FROM)):
        R = frame(bk.R, [ONE, THREE, ORIGINAL], start)
        for years in (10, 20):
            if len(R) / 252 < years + 1:
                continue
            w = dca_windows(R, years)
            dca_rows += dca_summary(w, lab, years)
            if years == 10 and "reales" not in lab:
                dca_fig.append((f"{lab.split(' desde')[0]}: 10 años de aportaciones", w))
    dca = pd.DataFrame(dca_rows).set_index(["mercado", "años", "estrategia"])
    rep.p("**Tabla D.** Aportar 1 $ cada mes (primera sesión) durante 10 o 20 años y valorar al final; una ventana "
          "por mes de inicio (muy solapadas: 10 años de historia dan una ventana de 10 años independiente). "
          "«Suma única»: invertir de golpe al principio. TIR: tasa anual que iguala las aportaciones con la riqueza "
          "final.")
    rep.table(flat(dca), pct={c for c in dca.columns if c.startswith(("P(", "TIR"))}, index=False, digits=3)

    # ── L*
    rep.h(f"{sec}. Por qué los 3x con adornos ganan desde 2010: el apalancamiento óptimo de cada tramo")
    sec += 1
    ls = pd.concat([lstar_rows(sp_d, SP, [("1938–2026", SP_START, None), ("hasta el 11-2-2010", SP_START, "2010-02-11"),
                                          ("desde el 12-2-2010", REAL_FROM, None)]),
                    lstar_rows(nd_d, NDX, [("1985–2026", NDX_START, None), ("hasta el 11-2-2010", NDX_START, "2010-02-11"),
                                           ("desde el 12-2-2010", REAL_FROM, None)])])
    rep.p(f"**Tabla K.** L* de cada tramo con los rótulos de kelly.py: {kelly.LSTAR_APROX}, "
          f"{kelly.LSTAR_DESC[kelly.LSTAR_APROX]}; {kelly.LSTAR_EXACTO}, {kelly.LSTAR_DESC[kelly.LSTAR_EXACTO]}; "
          f"«≥ {kelly.L_MAX:g}» si el máximo queda por encima del intervalo.")
    rep.table(ls, index=False, digits=2)

    # ── sensibilidades
    rep.h(f"{sec}. Sensibilidades: la salida publicada, la señal de precio y el mismo cierre")
    sec += 1
    sens_rows = []
    for mk, bk, start in ((SP, bk_sp, SP_START), (NDX, bk_nd, NDX_START)):
        keys = [k for k in bk.sens]
        df = frame(bk.R, REFS + [PUB, CORE, BAND] + keys, start)
        t = delta_table(df, (ORIGINAL,), rng_for(f"sens {mk}"), skip=(ONE, THREE))
        for k in [BAND, CORE, PUB] + keys:
            sens_rows.append({"mercado": mk, "regla": k, "CAGR": metrics.cagr(df[k]), "maxDD": metrics.max_dd(df[k]),
                              "Δg_LRS": t.loc[k, "Δg_LRS"], "p_LRS": t.loc[k, "p_LRS"]})
    sens = pd.DataFrame(sens_rows).set_index(["mercado", "regla"])
    rep.p("**Tabla X1.** La regla SPY +4%/−3% con otras salidas y las bandas con la señal de precio en lugar de la "
          "de rentabilidad total: el SPY de precio (^GSPC) en las que miran el SPY y, en la banda sobre el índice "
          "propio, ^GSPC en el S&P 500 y en el Nasdaq-100 su precio aproximado (la rentabilidad total menos un "
          "dividendo constante, como en R13: la base no tiene el precio diario del Nasdaq-100 en toda la historia). "
          "Ventanas completas (S&P 500 desde 1938, Nasdaq-100 desde 1985).")
    rep.table(flat(pp_cols(sens, ["Δg_LRS"])), pct={"CAGR", "maxDD"}, index=False, digits=3)
    mc_rows = []
    for mk, bk, bm, start in ((SP, bk_sp, sc_sp, SP_START), (NDX, bk_nd, sc_nd, NDX_START),
                              ("S&P 500 reales", bk_sp_r, sc_sp, REAL_FROM), ("Nasdaq-100 reales", bk_nd_r, sc_nd, REAL_FROM)):
        for k in [ORIGINAL] + [x for x in bm.main if x in bk.R]:
            if k in (FTLT_X, RSI79U) and "reales" not in mk:
                continue
            s_ = ux0 if k in (FTLT_X, RSI79U) else start
            both = pd.concat([bk.R[k], bm.R[k]], axis=1, keys=["sig", "mismo"]).loc[s_:].dropna()
            mc_rows.append({"ventana": mk, "regla": k, "desde": both.index[0].date(),
                            "CAGR_cierre_siguiente": metrics.cagr(both.sig), "CAGR_mismo_cierre": metrics.cagr(both.mismo),
                            "Δg_mismo_menos_siguiente": float((np.log1p(both.mismo) - np.log1p(both.sig)).mean() * 252)})
    mc = pd.DataFrame(mc_rows).set_index(["ventana", "regla"])
    rep.p("**Tabla X2.** Orden en el mismo cierre de la señal (el supuesto de Composer y de muchos backtests de los "
          "foros) frente al cierre siguiente (el del laboratorio). Antes de 1990 el mismo cierre aprovecha "
          "precios desfasados que no eran operables (R13, R15).")
    rep.table(flat(pp_cols(mc, ["Δg_mismo_menos_siguiente"])), pct={"CAGR_cierre_siguiente", "CAGR_mismo_cierre"},
              index=False, digits=3)

    # SPA de las familias ampliadas: principal + salidas alternativas + señal de precio (operables) y,
    # además, todas en el mismo cierre (incluida la LRS original)
    for code, title, bk, bm, fam, start in (("A+", "S&P 500 desde 1938", bk_sp, sc_sp, sp_nb, SP_START),
                                            ("B+", "S&P 500 desde 1962", bk_sp, sc_sp, sp_b, BOND_START),
                                            ("N+", "Nasdaq-100 desde 1985", bk_nd, sc_nd, nd_m, NDX_START)):
        R = dict(bk.R)
        ext = list(fam) + [k for k in bk.sens]
        df = frame(R, REFS + ext, start)
        spa_rows += spa_family(df, ext, rng_for(f"spa {code} op"), f"{title}, ampliada (operables)")
        count_rows.append({"ventana": f"{title}, ampliada (operables)", "reglas_familia": len(unique_cols(df, ext)[0])})
        for k in [ORIGINAL] + bm.main + bm.sens:
            if k in (FTLT_X, RSI79U) or (code == "A+" and k == FTLT_L):
                continue
            R[k + SC] = bm.R[k]
            ext.append(k + SC)
        df = frame(R, REFS + ext, start)
        spa_rows += spa_family(df, ext, rng_for(f"spa {code}"), f"{title}, ampliada con mismo cierre")
        count_rows.append({"ventana": f"{title}, ampliada con mismo cierre",
                           "reglas_familia": len(unique_cols(df, ext)[0])})

    # ── búsqueda
    rep.h(f"{sec}. Descontar la búsqueda: SPA de Hansen sobre cada familia")
    sec += 1
    spa = pd.DataFrame(spa_rows).set_index(["familia", "frente_a"])
    rep.p("**Tabla SPA.** Para cada familia y referencia: número de reglas, la mejor (por Δg), su Δg y el p del SPA "
          "de Hansen (versión consistente; bootstrap estacionario de bloque 63 y 2.000 remuestreos). Frente al 1x y "
          "al 3x la LRS original entra como una regla más. La familia ampliada suma las salidas alternativas, la "
          "señal de precio y todas las reglas en el mismo cierre (incluida la LRS).")
    rep.table(flat(pp_cols(spa, ["Δg_mejor"])), index=False, digits=3)

    # ── recuento
    n_built = {}
    census_rows = []
    for mk, bk, bm in ((SP, bk_sp, sc_sp), (NDX, bk_nd, sc_nd)):
        ux = [k for k in (FTLT_X, RSI79U) if mk == NDX]
        row = {"mercado": mk, "principales": len(bk.main) + len(ux),
               "salidas y señal alternativas": len(bk.sens),
               "mismo cierre (con la LRS)": len(bm.refs) - 2 + len(bm.main) + len(bm.sens),
               "diagnósticos de FTLT (UVXY → letras, → TQQQ)": 2 if mk == NDX else 0,
               "idénticas a otra (cuentan una vez)": ", ".join(f"{k} = {v}" for k, v in {**bk.dups, **bm.dups}.items())
               or "—"}
        row["total"] = sum(v for k, v in row.items() if isinstance(v, int))
        n_built[mk] = row["total"]
        census_rows.append(row)
    rep.h(f"{sec}. Recuento de variantes")
    sec += 1
    cnt = pd.DataFrame(count_rows).set_index("ventana")
    rep.p("**Tabla C1.** Series de reglas construidas en cada mercado (sin el 1x, el 3x ni la LRS original con orden "
          "al cierre siguiente; más el DCA, que no es una serie). Las de UVXY solo existen con ETFs reales.")
    rep.table(pd.DataFrame(census_rows), index=False)
    rep.p("**Tabla C2.** Reglas de cada familia contrastada con el SPA (sin las referencias).")
    rep.table(cnt)
    imp = ndx_imputations(D, nd_m)
    rep.p(f"En total, {n_built[SP]} series de reglas en el S&P 500 y {n_built[NDX]} en el Nasdaq-100. La función "
          "census_returns() devuelve las del S&P 500 para el censo ampliado (R25). Los semiconductores, que R13, R21 "
          "y R23 sí estudian, quedan fuera: estas reglas se publicaron para SPY, QQQ y TQQQ, y trasladarlas a SOXL "
          "sería elegir otra vez sus parámetros.")
    rep.h(f"Nasdaq-100: {r13.N_NDX} reconstrucciones del periodo 1971–85", 3)
    rep.p("**Tabla I.** CAGR y Δg frente a la LRS original desde 1985-10 con cada reconstrucción (el pasado "
          "reconstruido solo calienta las señales y las medias).")
    rep.table(pp_cols(imp, ["Δg_LRS_mediana", "Δg_LRS_mín", "Δg_LRS_máx"]),
              pct={"CAGR_mediana", "CAGR_mín", "CAGR_máx"}, digits=2)

    # ── figuras
    def forest_panel(code_main, code_bond, fam_main, fam_bond):
        rows = []
        for code, fam, bond in ((code_main, fam_main, False), (code_bond, fam_bond, True)):
            if code is None:
                continue
            inf = INF[code][2]
            for k in fam:
                rows.append({"regla": k, "etiqueta": label(k, bond), "clase": FICHA[k]["clase"],
                             "Δg": inf.loc[k, "Δg_LRS"], "IC90_inf": inf.loc[k, "IC90_inf_LRS"],
                             "IC90_sup": inf.loc[k, "IC90_sup_LRS"]})
        return pd.DataFrame(rows).set_index("regla")

    order = list(CLASS_COLOR)

    def sort_cls(t):
        return t.assign(_o=t.clase.map(order.index)).sort_values(["_o"], kind="stable").drop(columns="_o")

    f_sp = sort_cls(forest_panel("A", "B", sp_nb, [k for k in sp_b if k not in sp_nb]))
    f_nd = sort_cls(forest_panel("N", None, nd_m, []))
    fig_forest([("S&P 500 desde 1938", f_sp), ("Nasdaq-100 desde 1985", f_nd)], rep)

    def split_panel(mk, keys, bond_keys):
        dg = SUB[mk][0]
        t = pd.DataFrame({"antes": dg["hasta el 11-2-2010"], "después": dg["desde el 12-2-2010"]}).loc[keys]
        t["clase"] = [FICHA[k]["clase"] for k in t.index]
        t["etiqueta"] = [label(k, k in bond_keys) for k in t.index]
        return sort_cls(t)

    s_sp = split_panel(SP, sp_b, [k for k in sp_b if k not in sp_nb])
    s_nd = split_panel(NDX, nd_m, [])
    fig_2010([("S&P 500 desde 1938", s_sp), ("Nasdaq-100 desde 1985", s_nd)], rep)
    fig_ftlt(ft_bars, rep)
    fig_dca(dca_fig, rep)

    # ── resumen (se inserta arriba) y apéndice
    SUM = summary_lines(dict(INF=INF, SUB=SUB, spa=spa, post=post, ft=ft, uv=uv, comp=comp, dca=dca, ls=ls,
                             sim_sp=sim_sp, sim_nd=sim_nd, mc=mc, imp=imp, sens=sens, n_built=n_built, ATT=ATT,
                             top5=lr.iloc[:5], ex=ex, san=san,
                             f2_años=ft_years["tras la última edición (24-1-2023), desde el 25-1-2023"]))
    rep.h("Apéndice: series para el censo (R19 y R25)")
    rep.p("`r24_folk.census_returns(D=None, same_close_rules=True, price_signal=True, bonds=True) -> dict[str, "
          "pd.Series]` devuelve la rentabilidad diaria neta en el S&P 500 de cada regla probada aquí, con el nombre "
          "«R24 · <regla>» (y «· mismo cierre» o «· señal de precio»), en el índice de r13.load()[\"S&P 500\"][\"u\"]; "
          "`census_meta(nombres)` da la ficha de cada una para r19.Census.add. Las reglas con bonos empiezan en "
          "1962, lo que movería la ventana común del censo; con bonds=False quedan fuera.")
    head = rep.lines[:body_start]
    body = rep.lines[body_start:]
    rep.lines = head + ["## Resumen", ""] + [f"- {s}" for s in SUM] + [""] + body
    rep.save()
    return {"INF": INF, "SUB": SUB, "spa": spa, "post": post, "ft": ft, "dca": dca, "ls": ls, "n_built": n_built}


def summary_lines(X: dict) -> list[str]:
    """Frases del resumen con las cifras de esta ejecución."""
    INF, SUB, post, ft = X["INF"], X["SUB"], X["post"], X["ft"]
    spa = X["spa"].reset_index()
    perf = {c: INF[c][1] for c in INF}
    inf = {c: INF[c][2] for c in INF}
    san = X["san"]

    def P(code, k, col="CAGR"):
        return perf[code].loc[k, col]

    def I(code, k, col):
        return inf[code].loc[k, col]

    def S(fam, bench=ORIGINAL):
        return spa.loc[(spa.familia == fam) & (spa.frente_a == bench)].iloc[0]

    def rng_pp(vals):
        return f"{pp(min(vals))} a {pp(max(vals))}".replace(" pp a", " a")

    out = []
    out.append(f"**Cordura.** La base reproduce R13: desde el 4-2-1938, LRS 3x (original) {pct_es(san.loc[ORIGINAL, 'CAGR_R24'])} "
               f"anual con una caída máxima del {pct_es(san.loc[ORIGINAL, 'maxDD_R24'])}; 1x {pct_es(san.loc[ONE, 'CAGR_R24'])}; "
               f"3x comprar y mantener {pct_es(san.loc[THREE, 'CAGR_R24'])}. Todas las reglas se ejecutan con ese protocolo "
               "(orden al cierre siguiente, 5 pb, señal de rentabilidad total en el S&P 500).")
    out.append(f"**Qué se probó.** Once reglas de los foros, en {len(inf['B'])} variantes principales en el S&P 500 y "
               f"{len(inf['UX'])} en el Nasdaq-100 (dos de ellas, con UVXY, solo con ETFs reales), más sus salidas "
               f"alternativas, la señal de precio y la ejecución en el mismo cierre: {X['n_built'][SP]} series en el S&P 500 "
               f"y {X['n_built'][NDX]} en el Nasdaq-100, y el DCA. Cada familia se contrasta con el SPA de Hansen (sección «Descontar la búsqueda»).")
    # LRS reempaquetada
    rl = [BAND, CORE, PUB, OWN3X]
    pos = [BAND, CORE, OWN3X]
    sims = pd.concat([X["sim_sp"].loc[pos], X["sim_nd"].loc[pos + [SPYSIG]]])
    dA = [I("A", k, "Δg_LRS") for k in rl]
    best = max(rl, key=lambda k: I("A", k, "Δg_LRS"))
    att = X["ATT"][SP]
    own = att.loc[OWN3X] if OWN3X in att.index else None
    if own is not None:
        y1, v1 = re.findall(r"(\d{4}) \(([−\-\d,]+)\)", own.años_que_más_aportan)[0]
    out.append(f"**La LRS con otro nombre.** Las bandas +5%/−3% y +4%/−3%, la señal del SPY y la media del propio 3x están "
               f"en la misma posición que la LRS original el {pct_es(sims.misma_posición_que_LRS.min(), 0)}–"
               f"{pct_es(sims.misma_posición_que_LRS.max(), 0)} de los días (la salida publicada de SPY +4%/−3%, que compra el "
               f"1x poco a poco, el {pct_es(X['sim_sp'].loc[PUB, 'misma_posición_que_LRS'], 0)}), con una correlación diaria de {num_es(sims.corr_con_LRS.min(), 2)}–"
               f"{num_es(sims.corr_con_LRS.max(), 2)}. En el S&P 500 desde 1938 su Δg frente a la original va de "
               f"{rng_pp(dA)} al año; la mejor, «{best}»"
               + (" (una variante del laboratorio: ninguna fuente publicada usa esta regla exacta)" if best == OWN3X else "")
               + f", {pp(I('A', best, 'Δg_LRS'))} (p = {fp(I('A', best, 'p_LRS'))}), y el SPA de "
               f"la familia frente a la original da p = {fp(S('S&P 500 desde 1938 (reglas sin bonos)')['p_SPA_c'])}. Las bandas "
               f"operan {num_es(max(P('A', BAND, 'operaciones_año'), P('A', CORE, 'operaciones_año')), 1)} veces al año o menos, "
               f"frente a {num_es(P('A', ORIGINAL, 'operaciones_año'), 1)}, pero caen más ({pct_es(P('A', BAND, 'maxDD'), 0)} y "
               f"{pct_es(P('A', CORE, 'maxDD'), 0)} frente a {pct_es(P('A', ORIGINAL, 'maxDD'), 0)}); la media del propio 3x cae "
               f"menos ({pct_es(P('A', OWN3X, 'maxDD'), 0)})"
               + (f", y desde 1962 casi toda su ventaja sale de {y1} ({v1} de {num_es(own.diferencia_log_total, 2)} "
                  "puntos log): su media, calculada sobre el 3x, decidió salir el 14-10-1987 y evitó el lunes 19, que la "
                  "original encajó." if own is not None else "."))
    sn = X["sens"].loc[SP]
    kx = [k + PX for k in (BAND, CORE, PUB)]
    snn = X["sens"].loc[NDX]
    out.append(f"**Señal de precio.** Con la media sobre el índice de precio (^GSPC), como en los gráficos de TradingView, "
               f"las tres reglas de bandas del S&P 500 caen entre el {pct_es(sn.loc[kx, 'maxDD'].max(), 0)} y el "
               f"{pct_es(sn.loc[kx, 'maxDD'].min(), 0)} y su Δg frente a la original va de "
               f"{rng_pp(list(sn.loc[kx, 'Δg_LRS']))} (p nominal ≥ {fp(sn.loc[kx, 'p_LRS'].min())}); en el Nasdaq-100, con "
               f"el SPY de precio y el precio aproximado del índice, de {rng_pp(list(snn.loc[kx, 'Δg_LRS']))} "
               f"(p nominal ≥ {fp(snn.loc[kx, 'p_LRS'].min())}; tabla X1).")
    # Nasdaq-100 con el SPY
    ks = [SPYSIG, CORE, PUB]
    dN = [I("N", k, "Δg_LRS") for k in ks]
    pN = [I("N", k, "p_LRS") for k in ks]
    hN = [I("N", k, "p_Holm_LRS") for k in ks]
    attN = X["ATT"][NDX]
    d16 = SUB[NDX][0]["2016–2026"]
    out.append(f"**Nasdaq-100: decidir con el SPY.** Desde 1985 las reglas que miran el SPY superan a la LRS original en "
               f"{rng_pp(dN)} al año (p nominal {fp(min(pN))}–{fp(max(pN))}); con Holm, p ≥ {fp(min(hN))}, y el SPA de la "
               f"familia da p = {fp(S('Nasdaq-100 desde 1985')['p_SPA_c'])}. La ventaja se concentra en pocos años: "
               f"{pct_es(attN.loc[PUB, 'parte_de_esos_años'], 0)} de la de SPY +4%/−3% publicada sale de "
               f"{attN.loc[PUB, 'años_que_más_aportan']} y {pct_es(attN.loc[SPYSIG, 'parte_de_esos_años'], 0)} de la de la "
               f"señal del SPY de {attN.loc[SPYSIG, 'años_que_más_aportan']} (diferencias log). Desde 2016: "
               f"{pp(d16[SPYSIG])} (señal del SPY), {pp(d16[CORE])} y {pp(d16[PUB])} (bandas del SPY). Con las 10 "
               "reconstrucciones del Nasdaq-100 anterior a 1985 las cifras apenas cambian (tabla I).")
    # FTLT
    f1 = ft.loc["desde octubre de 2011 (UVXY)"]
    f2 = ft.loc["tras la última edición (24-1-2023), desde el 25-1-2023"]
    uv = X["uv"]
    t5 = X["top5"]
    comp = X["comp"]
    c_ft = comp.loc[(FTLT_X, "mismo cierre, sin costes (como Composer)")]
    out.append(f"**FTLT.** El árbol exacto con ETFs reales rinde {pct_es(f1.loc['FTLT original, cierre siguiente', 'CAGR'])} "
               f"anual desde octubre de 2011, frente al {pct_es(f1.loc['LRS 3x (original)', 'CAGR'])} de la LRS y el "
               f"{pct_es(f1.loc['TQQQ (3x)', 'CAGR'])} de TQQQ (Δg {pp(I('UX', FTLT_X, 'Δg_LRS'))} frente a la LRS, p nominal "
               f"{fp(I('UX', FTLT_X, 'p_LRS'))}, Holm {fp(I('UX', FTLT_X, 'p_Holm_LRS'))}; SPA de la familia con UVXY "
               f"{fp(S('Nasdaq-100 con ETFs reales y UVXY (desde 2011-10)')['p_SPA_c'])}). Con la rama UVXY en letras rinde "
               f"{pct_es(f1.loc['FTLT, rama UVXY → letras', 'CAGR'])}: los {uv['días en UVXY']} días en UVXY suman "
               f"{num_es(uv['suma log de UVXY en esos días'], 2)} puntos log, y {num_es(uv['de ella, los 5 mejores días'], 2)} "
               f"salen de cinco días (el mejor, el {_date(t5.index[0])}, +{pct_es(np.expm1(t5.iloc[0]), 0)}). En el mismo cierre "
               f"rinde {pct_es(f1.loc['FTLT original, mismo cierre', 'CAGR'], 0)}; así calcula Composer, y con esa ejecución "
               f"los datos del laboratorio reproducen su cifra publicada desde la última edición "
               f"({pct_es(c_ft.CAGR)} frente a {pct_es(c_ft.CAGR_Composer)}). Tras esa edición, con orden al cierre "
               f"siguiente, rinde {pct_es(f2.loc['FTLT original, cierre siguiente', 'CAGR'])} frente al "
               f"{pct_es(f2.loc['TQQQ (3x)', 'CAGR'])} de TQQQ y el {pct_es(f2.loc['LRS 3x (original)', 'CAGR'])} de la LRS "
               f"({num_es(X['f2_años'], 1)} años; Δg {pp(f2.loc['FTLT original, cierre siguiente', 'Δg_LRS'])} frente a la LRS, p = "
               f"{fp(f2.loc['FTLT original, cierre siguiente', 'p_LRS'])}).")
    # mismo cierre
    mc = X["mc"]
    fl = mc.loc[(NDX, FTLT_L)]
    cn = mc.loc[(SP, CONNORS)]
    ext = S("Nasdaq-100 desde 1985, ampliada con mismo cierre")
    exo = S("Nasdaq-100 desde 1985, ampliada (operables)")
    out.append(f"**La hora de ejecución.** Las ramas de RSI(10) de FTLT viven de la reversión del día siguiente: la "
               f"autocorrelación diaria del índice es negativa desde 2000 (tabla F4), y la versión larga pasa de "
               f"{pct_es(fl.CAGR_cierre_siguiente)} a {pct_es(fl.CAGR_mismo_cierre)} en el Nasdaq-100 desde 1985 si se opera "
               f"en el mismo cierre. Es lo único que supera a la LRS tras descontar la búsqueda (familia ampliada del "
               f"Nasdaq-100 con mismo cierre, SPA p = {fp(ext.p_SPA_c)}; sin el mismo cierre, p = {fp(exo.p_SPA_c)}), y exige "
               "decidir con un precio de cierre que aún no se conoce cuando hay que mandar la orden (R15: la señal se "
               "calcula minutos antes). No vale para toda regla de RSI: RSI(2) de Connors rinde menos en el mismo cierre "
               f"({pct_es(cn.CAGR_mismo_cierre)} frente a {pct_es(cn.CAGR_cierre_siguiente)} en el S&P 500 desde 1938).")
    # 3x con adornos
    s_sp = X["sim_sp"]
    out.append(f"**3x con adornos.** RSI(10) > 79 → caja es casi un 3x: invertida el {pct_es(P('A', RSI79, 'tiempo_invertido'), 0)} "
               f"de los días, correlación con el 3x de {num_es(s_sp.loc[RSI79, 'corr_con_3x'], 2)}; en el S&P 500 desde 1938 "
               f"rinde {pct_es(P('A', RSI79))} frente al {pct_es(P('A', THREE))} del 3x. 9Sig también (correlación "
               f"{num_es(s_sp.loc[NINE, 'corr_con_3x'], 2)}): es la mejor con ETFs reales en el S&P 500 "
               f"({pct_es(P('RS', NINE))} frente al {pct_es(P('RS', ORIGINAL))} de la LRS; Δg {pp(I('RS', NINE, 'Δg_LRS'))}, p "
               f"nominal {fp(I('RS', NINE, 'p_LRS'))}, Holm {fp(I('RS', NINE, 'p_Holm_LRS'))}, SPA "
               f"{fp(S('S&P 500 con ETFs reales (desde 2010)')['p_SPA_c'])}), pero de 1962 a febrero de 2010 rindió "
               f"{pct_es(SUB[SP][2].loc[NINE, 'hasta el 11-2-2010'])} anual y en el Nasdaq-100 su caída máxima es del "
               f"{pct_es(P('N', NINE, 'maxDD'))}.")
    # antes / después de 2010 y L*
    ls = X["ls"].set_index(["mercado", "ventana"])

    def lx(mk, w):
        return str(ls.loc[(mk, w), kelly.LSTAR_EXACTO]).replace(".", ",")
    flips = {}
    for mk in (SP, NDX):
        dg = SUB[mk][0]
        flips[mk] = int(((dg["hasta el 11-2-2010"] < 0) & (dg["desde el 12-2-2010"] > 0)).sum()), len(dg)
    rs3 = S("S&P 500 con ETFs reales (desde 2010)", THREE)
    rn3 = S("Nasdaq-100 con ETFs reales (desde 2010)", THREE)
    ux3 = S("Nasdaq-100 con ETFs reales y UVXY (desde 2011-10)", THREE)
    out.append(f"**Antes y después de 2010.** {flips[SP][0]} de las {flips[SP][1]} reglas del S&P 500 y "
               f"{flips[NDX][0]} de las {flips[NDX][1]} del Nasdaq-100 rinden menos que la LRS original hasta el 11-2-2010 "
               "y más desde el 12-2-2010 (figura «Frente a la LRS original, antes y después de 2010»). Lo que cambia entre los dos tramos es el apalancamiento que "
               f"convenía: el {kelly.LSTAR_EXACTO} fue {lx(SP, 'hasta el 11-2-2010')} en el S&P 500 hasta 2010 y "
               f"{lx(SP, 'desde el 12-2-2010')} después; en el Nasdaq-100, {lx(NDX, 'hasta el 11-2-2010')} y "
               f"{lx(NDX, 'desde el 12-2-2010')}. "
               "Desde 2010 lo óptimo era estar siempre en el 3x o más, y las reglas que casi nunca salen de él ganan a la LRS; "
               f"frente al propio 3x comprar y mantener, ninguna familia con ETFs reales es significativa (SPA p = "
               f"{fp(rs3.p_SPA_c)}, {fp(rn3.p_SPA_c)} y {fp(ux3.p_SPA_c)}).")
    # HFEA
    pst = post
    h55 = pst.loc[(HFEA55, SP)]
    h40 = pst.loc[(HFEA40, SP)]
    out.append(f"**HFEA es una apuesta a que bajen los tipos.** En el S&P 500 de 1962 a 2026 rinde {pct_es(P('B', HFEA55))} "
               f"(55/45) frente al {pct_es(P('B', ONE))} del 1x y el {pct_es(P('B', ORIGINAL))} de la LRS, con una caída máxima del "
               f"{pct_es(P('B', HFEA55, 'maxDD'), 0)}; de 1962 a 1973, {pct_es(SUB[SP][2].loc[HFEA55, '1938–1973'])} anual. Tras "
               f"su publicación rindió {pct_es(h55.CAGR)} (55/45, desde agosto de 2019) frente al {pct_es(h55.CAGR_1x)} del 1x, y "
               f"{pct_es(h40.CAGR)} (40/60, desde febrero de 2019) frente al {pct_es(h40.CAGR_1x)}, con caídas del "
               f"{pct_es(h55.maxDD, 0)} y el {pct_es(h40.maxDD, 0)}. En el Nasdaq-100 supera a la LRS "
               f"({pp(I('N', HFEA55, 'Δg_LRS'))}, p = {fp(I('N', HFEA55, 'p_LRS'))}) gracias a 1985–1999 "
               f"({pp(SUB[NDX][0].loc[HFEA55, '1985–1999'])}).")
    out.append(f"**Comprar caídas con apalancamiento y la RSI(2) de Connors fallan.** Pasar al 3x tras una caída del 20% o "
               f"del 30% rinde {pct_es(P('N', DIP20))} y {pct_es(P('N', DIP30))} en el Nasdaq-100 desde 1985, frente al "
               f"{pct_es(P('N', ONE))} del 1x, con caídas del {pct_es(P('N', DIP20, 'maxDD'), 2)}. La RSI(2) de Connors en el 3x "
               f"está invertida el {pct_es(P('A', CONNORS, 'tiempo_invertido'), 0)} del tiempo, opera "
               f"{num_es(P('A', CONNORS, 'operaciones_año'), 0)} veces al año y rinde {pct_es(P('A', CONNORS))} en el S&P 500 "
               f"desde 1938, frente al {pct_es(P('A', ONE))} del 1x.")
    dca = X["dca"]
    d_sp = dca.loc[("S&P 500 desde 1938", 10, "3x")]
    d_nd = dca.loc[("Nasdaq-100 desde 1985", 10, "3x")]
    d_nd20 = dca.loc[("Nasdaq-100 desde 1985", 20, "3x")]
    d_l = dca.loc[("S&P 500 desde 1938", 10, "LRS 3x (original)")]
    out.append(f"**El DCA no arregla el 3x.** Aportando cada mes durante 10 años, el 3x del S&P 500 acaba perdiendo dinero "
               f"en el {pct_es(d_sp['P(DCA pierde dinero)'])} de las ventanas (el 1x, en el {pct_es(d_sp['P(DCA 1x pierde dinero)'])}) "
               f"y por debajo del DCA en el 1x en el {pct_es(d_sp['P(DCA por debajo del DCA 1x)'])}; en el Nasdaq-100, "
               f"{pct_es(d_nd['P(DCA pierde dinero)'])} y {pct_es(d_nd['P(DCA por debajo del DCA 1x)'])} (a 20 años, "
               f"{pct_es(d_nd20['P(DCA pierde dinero)'])} y {pct_es(d_nd20['P(DCA por debajo del DCA 1x)'])}). En las ventanas "
               f"que empiezan desde 2010 nunca queda por debajo del 1x: de ahí sale la afirmación de los foros. El DCA en la "
               f"LRS original pierde dinero en el {pct_es(d_l['P(DCA pierde dinero)'])} de las ventanas de 10 años del S&P 500.")
    pb = post.loc[(BAND, NDX)]
    n9 = post.loc[(NINE, NDX)]
    recent = post[[pd.Timestamp(d) >= pd.Timestamp("2019-01-01") for d in post.desde]]
    recent = recent[recent.index.get_level_values(0) != ORIGINAL]
    out.append(f"**Tras la publicación.** Salvo la RSI(2) de Connors (2008) y 9Sig (hacia 2017), las reglas se publicaron "
               f"desde 2019 y su fuera de muestra dura entre {num_es(recent.años.min(), 1)} y {num_es(recent.años.max(), 1)} "
               f"años: no basta para distinguirlas de la LRS. Las bandas, publicadas hacia 2025, rinden {pct_es(pb.CAGR)} en el "
               f"Nasdaq-100 desde enero de 2025 frente al {pct_es(pb.CAGR_LRS)} de la LRS: salieron más tarde en las caídas de "
               "marzo de 2025 y de marzo de 2026 y volvieron más tarde en abril de 2026. 9Sig, desde 2017, rinde "
               f"{pct_es(n9.CAGR)} en el Nasdaq-100 frente al {pct_es(n9.CAGR_3x)} de TQQQ y el {pct_es(n9.CAGR_LRS)} de la LRS.")
    ops = spa[spa.familia.str.contains("operables") | ~spa.familia.str.contains("ampliada")]
    lo, hi = ops[ops.frente_a == ORIGINAL].p_SPA_c.min(), ops[ops.frente_a == ORIGINAL].p_SPA_c.max()
    out.append(f"**Veredicto.** Descontada la búsqueda, ninguna regla de los foros ejecutable con orden al cierre "
               f"siguiente supera a la LRS original en ninguna ventana (SPA frente a la original, p entre {fp(lo)} y "
               f"{fp(hi)}); ninguna p de Holm individual baja de "
               f"{fp(min(inf[c]['p_Holm_LRS'].min() for c in inf))}. Las que más prometen en el Nasdaq-100 (el SPY como "
               "señal, HFEA, FTLT) ganan por unos pocos años o con una ejecución que no es operable tal como se simula, y las que ganan "
               "desde 2010 ganan porque estaban casi siempre en el 3x.")
    return out


if __name__ == "__main__":
    run()
