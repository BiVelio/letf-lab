"""R25 — Censo ampliado: el censo del artículo 2 con las reglas del artículo 3.

Pregunta. El artículo 2 descontó la búsqueda con el censo de R19: 222 reglas distintas en
el S&P 500, entre las que ninguna alternativa superaba a la LRS 3x original de Gayed y
Bilello (2016) tras la corrección. El artículo 3 probó muchas más (R21–R24). Con todas
dentro, ¿sobrevive a la búsqueda alguna regla nueva frente a la LRS original o frente al
1x? ¿A cuántas pruebas independientes equivale ahora el censo y qué puesto ocupa la
original?

Censo. r19.build_census (importado, sin tocar R19) reconstruye las series del censo del
artículo 2 en el S&P 500, y se añaden, con las funciones que exporta cada módulo, todas las
reglas del S&P 500 de:
- R21 (census_rows, census_meta): las 25 señales de tendencia nuevas y sus 25 gemelas al
  mismo cierre;
- R22 (census_series): inverso bajo la media, LRS con corto del −3x, 2x con el ETF 2x, la
  curva de apalancamiento de la regla (1,5x a 6x) con la señal original y con la de precio,
  par corto, par largo y lever-arb;
- R23 (census_rows): filtros de régimen de volatilidad, LRS × volatilidad, VIX, VIX/VIX3M
  y clasificadores;
- R24 (census_returns, census_meta): las reglas de los foros, sus salidas y señal de precio
  alternativas y sus gemelas al mismo cierre.
Todas con el protocolo común (señal al cierre de t, orden al cierre de t+1, 5 pb por unidad
negociada) y las señales del S&P 500 sobre el índice de rentabilidad total. Las series
idénticas en una ventana (la misma regla construida en dos módulos) cuentan una vez
(r19.dedupe, que conserva la primera: una regla de R21–R24 idéntica a una de R19 cuenta como
de R19). Las de R22 empiezan una cuenta nueva tras un día de −100%; aquí se absorbe la ruina
desde el inicio de cada ventana (quien empezara ese día lo perdería todo), como recomienda
r22.census_series. Quedan fuera, como en R19, las versiones de una misma regla con otros
supuestos de datos o de costes (en R22, el producto del corto sin remunerar) y todo lo
probado en otros mercados.

Ventanas. Cada regla entra en las ventanas en que está viva, y cada ventana empieza el
primer mes completo en que lo están todas las suyas (la regla de r19.window):
- principal: la del censo del artículo 2 (desde 1939-04), con todas las reglas nuevas que
  ya existen entonces (todas salvo las de VIX de R23 y las de bonos de R24);
- 1962: añade las reglas con bonos de R24 (9Sig, HFEA y FTLT, desde 1962);
- todas: añade las de VIX (desde 1991) y VIX/VIX3M (desde 2006) de R23.

Medidas, las de R19 con sus funciones y su semilla:
- exceso logarítmico diario frente al 1x, a_t = ln(1 + r_t) − ln(1 + u_t), y, para las
  alternativas, frente a la LRS 3x original;
- número efectivo de pruebas con los autovalores de la matriz de correlación de los excesos
  (r19.n_eff; principal, Li y Ji), con el censo de R19, con las nuevas y con todas;
- p unilateral (H0: Δg ≤ 0) con el error típico del bootstrap estacionario (r19.boot_test:
  bloque medio de 63 sesiones, 2.000 remuestreos, los mismos para todas las reglas) y el
  step-down de Romano y Wolf (r19.romano_wolf) sobre toda la familia y solo sobre las nuevas;
- SPA de Hansen (stats.spa_test, con su recentrado) frente a la LRS original y frente al 1x,
  con todas las reglas, solo con las nuevas y solo con las operables (sin las de mismo
  cierre, como R19);
- puesto de la LRS original por IR (y por Δg) frente al 1x, entre todas las reglas y entre
  las parametrizaciones de la LRS 3x.
El orden en que se usan los remuestreos es el de r19.analyse (boot_test frente al 1x, frente
a la LRS, SPA frente al 1x y frente a la LRS, con el generador de R19): con solo las reglas de
R19, la ventana principal reproduce sus cifras, y como los remuestreos de r19.boot_test
dependen solo del número de sesiones, el p de la LRS original frente al 1x (la hipótesis
principal del artículo 2, que se contrasta sola) es el mismo con el censo ampliado.
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from letf_lab import stats
from letf_lab.report import FIG_W, Report, comma_fig, num_es
from letf_lab.research import r13_lrs as r13
from letf_lab.research import r19_census as r19

log = logging.getLogger(__name__)

SEED = r19.SEED               # el generador de R19: reproduce su censo en la ventana principal
SEED_EXTRA = 2500             # SPA de subconjuntos (nuevas, operables): generadores propios
B_BOOT, B_SPA, BLOCK = r19.B_BOOT, r19.B_SPA, r19.BLOCK
ALPHA = r19.ALPHA
BENCH, LRS = r19.BENCH, r19.LRS
SP = "S&P 500"
MODULES = ("R21", "R22", "R23", "R24")
SOURCES = ("R19",) + MODULES
BOND_CUTOFF = pd.Timestamp("1963-01-01")    # la ventana «1962» admite lo que vive antes de 1963
WINDOWS = ("principal", "1962", "todas")
WINDOW_LABEL = {"principal": "principal (la del artículo 2)", "1962": "desde 1962 (con bonos)",
                "todas": "todas las reglas"}

# Cifras publicadas por R19 (reports/r19_census/README.md, ventana 1939-04-03 → 2026-09-25) que la
# parte de R19 de este censo debe reproducir: se comprueban en cada ejecución
R19_PUBLISHED = {"reglas distintas": 222, "N_eff frente al 1x (Li y Ji)": 40, "N_eff frente a la LRS (Li y Ji)": 50,
                 "SPA frente al 1x": 0.017, "SPA frente a la LRS": 0.306, "p LRS frente al 1x": 0.034,
                 "Romano–Wolf < 5% frente al 1x": 4, "Romano–Wolf < 5% frente a la LRS": 0}

GROUP_R22 = "Otras reglas y referencias"
SHORT_FAMS = ("(a) LRS con corto del −3x", "(b) par corto", "(c) lever-arb")   # familias de R22 que venden en corto
GROUP_R23 = "Kelly y volatilidad"
COLORS = {"R19": "#9a9994", "R21": "#2a78d6", "R22": "#eb6834", "R23": "#1baf7a", "R24": "#4a3aa7"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


# ───────────────────────────── construcción ─────────────────────────────

def source_of(name: str) -> str:
    """Módulo de origen de una serie del censo: «R21 · …» → R21; el resto, R19."""
    for m in MODULES:
        if name.startswith(f"{m} · "):
            return m
    return "R19"


def absorb_from(r: pd.Series, start) -> pd.Series:
    """Una serie de R22 (cuenta nueva tras un −100%) vista por quien empieza en `start`: NaN antes
    y, desde `start`, la ruina absorbida (r22.absorb: rentabilidad 0 hasta el final). Es lo que hace
    r22.census_series(start=…)."""
    from letf_lab.research import r22_inverse as r22
    out = r.copy()
    out[out.index < pd.Timestamp(start)] = np.nan
    live = out.notna()
    out[live] = r22.absorb(out[live])
    return out


def new_rules(D: dict, cache: str | None = None) -> tuple[dict, pd.DataFrame]:
    """Rentabilidades diarias netas en el S&P 500 de todas las reglas de R21–R24 que exportan sus
    módulos y sus fichas (módulo, familia, grupo, parametrización_LRS3x, vivo_desde, nota). Las de R22
    son las series en bruto (cuenta nueva tras un −100%): la ruina se absorbe en cada ventana."""
    from letf_lab.research import r21_trend as r21
    from letf_lab.research import r22_inverse as r22
    from letf_lab.research import r23_vol as r23
    from letf_lab.research import r24_folk as r24

    R, meta = {}, {}
    rows = r21.census_rows(D)
    m21 = r21.census_meta()
    for k, r in rows.items():
        R[k] = r
        meta[k] = {"módulo": "R21", "familia": m21.loc[k, "familia"], "grupo": m21.loc[k, "grupo"],
                   "parametrización_LRS3x": bool(m21.loc[k, "parametrización_LRS3x"]), "nota": ""}
    raw, m22 = r22.census_series(D, start=None, with_meta=True)
    for k, r in raw.items():
        R[k] = r
        meta[k] = {"módulo": "R22", "familia": f"R22: {m22.loc[k, 'familia']}", "grupo": GROUP_R22,
                   "parametrización_LRS3x": False,
                   "nota": "" if m22.loc[k, "operable_con_ETFs"] else "más de 3x: sin ETF cotizado"}
    rows23 = r23.census_rows(D, cache)
    for k, r in rows23.items():
        R[k] = r
        meta[k] = {"módulo": "R23", "familia": f"R23: {r.attrs.get('grupo', '')}", "grupo": GROUP_R23,
                   "parametrización_LRS3x": False, "nota": ""}
    rows24 = r24.census_returns(D)
    m24 = r24.census_meta(list(rows24))
    for k, r in rows24.items():
        R[k] = r
        meta[k] = {**m24[k], "nota": ""}
    idx = D[SP]["u"].index
    R = {k: v.reindex(idx) for k, v in R.items()}
    M = pd.DataFrame(meta).T
    M["vivo_desde"] = [R[k].first_valid_index() for k in M.index]
    M["operable"] = [not r19.is_same_close(k) for k in M.index]
    return R, M


def month_after(t: pd.Timestamp, idx: pd.DatetimeIndex) -> pd.Timestamp:
    """Primera sesión del mes siguiente al de t (la regla de r19.window)."""
    per = idx.to_period("M")
    return idx[per >= t.to_period("M") + 1][0]


def window_members(live: pd.Series, base: list[str], idx: pd.DatetimeIndex) -> dict:
    """Reglas de cada ventana y su primera sesión. live: primer día de cada regla (todas: R19 y
    nuevas); base: las del censo de R19, que fijan la ventana principal."""
    start_a = month_after(live[base].max(), idx)
    first_a = start_a.to_period("M").start_time
    out = {"principal": (start_a, [k for k in live.index if k in base or live[k] < first_a])}
    b = [k for k in live.index if live[k] < BOND_CUTOFF]
    out["1962"] = (month_after(live[b].max(), idx), b)
    out["todas"] = (month_after(live.max(), idx), list(live.index))
    return out


# ───────────────────────────── análisis ─────────────────────────────

def ir_of(A: pd.DataFrame) -> pd.Series:
    """IR anual (media/desviación ×√252) de cada columna de excesos."""
    return pd.Series(r19.sr_moments(A.to_numpy(float))["sr"] * np.sqrt(252), index=A.columns)


def neff_of(A: pd.DataFrame) -> dict:
    """Estimadores de r19.n_eff con los autovalores de la matriz de correlación de A."""
    if A.shape[1] < 2:
        return {r19.NEFF_MAIN: float(A.shape[1])}
    return r19.n_eff(r19.corr_eigs(A))


def spa_row(A: pd.DataFrame, cols: list[str], rng, label: str, ref: str, t: pd.Series | None = None) -> dict:
    """SPA de Hansen (stats.spa_test) de `cols` de A frente a la referencia `ref`: la regla con más Δg
    («mejor», como en R19) y, con t (estadísticos estudentizados de r19.boot_test), la de mayor t, que es
    la que decide el SPA."""
    X = A[cols].to_numpy(float)
    res = stats.spa_test(X, BLOCK, B_SPA, rng)
    m = X.mean(axis=0)
    j = int(m.argmax())
    out = {"familia": label, "frente a": ref, "reglas": len(cols), "mejor": cols[j],
           "Δg_mejor": float(m[j] * 252), "p_SPA": res["p_SPA_c"]}
    if t is not None:
        out["mayor t"] = t[cols].idxmax()
    return out


def analyse(Rw: pd.DataFrame, bench: pd.Series, meta: pd.DataFrame, new: set[str], short: set[str] = frozenset(),
            extra_spa: bool = True, seed: int = SEED) -> dict:
    """Pruebas de un censo (columnas de Rw: una por regla distinta, sin NaN) en su ventana.

    Con el generador de R19 y su orden: boot_test de los excesos sobre el 1x y de las alternativas
    sobre la LRS, SPA frente al 1x y frente a la LRS (todas las reglas). Después, con generadores
    propios, el SPA de cada subconjunto (SUBSETS: las nuevas, las operables y las que no venden en
    corto). Romano–Wolf con los remuestreos de boot_test, sobre toda la familia y sobre cada
    subconjunto (una familia más pequeña: p ajustados menores o iguales)."""
    cols = list(Rw.columns)
    alt = [k for k in cols if k != LRS]
    A1 = r19.active(Rw, bench)
    A2 = r19.active(Rw[alt], Rw[LRS])
    rng = np.random.default_rng(seed)
    bt1 = r19.boot_test(A1.to_numpy(float), B_BOOT, BLOCK, rng)
    bt2 = r19.boot_test(A2.to_numpy(float), B_BOOT, BLOCK, rng)
    tt1, tt2 = pd.Series(bt1["t"], index=cols), pd.Series(bt2["t"], index=alt)
    spa = [spa_row(A1, cols, rng, "todas", "1x", tt1), spa_row(A2, alt, rng, "todas", "LRS 3x (original)", tt2)]
    t1 = pd.DataFrame({"Δg": A1.mean() * 252, "IR": ir_of(A1), "p": bt1["p"],
                       "p_RW": r19.romano_wolf(bt1["t"], bt1["t_boot"])}, index=cols)
    t2 = pd.DataFrame({"Δg": A2.mean() * 252, "IR": ir_of(A2), "p": bt2["p"],
                       "p_RW": r19.romano_wolf(bt2["t"], bt2["t_boot"])}, index=alt)
    member = {"nuevas": lambda k: k in new, "operables": lambda k: not r19.is_same_close(k),
              "sin cortos": lambda k: k not in short}
    for t, bt, cc in ((t1, bt1, cols), (t2, bt2, alt)):
        t["fuente"] = [source_of(k) for k in cc]
        t["operable"] = [not r19.is_same_close(k) for k in cc]
        for tag, f in member.items():
            ii = [i for i, k in enumerate(cc) if f(k)]
            col = f"p_RW_{tag.replace(' ', '_')}"
            t[col] = np.nan
            if ii and (tag != "nuevas" or new) and (tag != "sin cortos" or short):
                t.iloc[ii, t.columns.get_loc(col)] = r19.romano_wolf(bt["t"][ii], bt["t_boot"][:, ii])
    if extra_spa:
        for j, (tag, f) in enumerate(member.items()):
            c1 = [k for k in cols if f(k)]
            c2 = [k for k in c1 if k != LRS]
            if tag == "nuevas" and not new or tag == "sin cortos" and not short or not c2:
                continue
            rng_s = np.random.default_rng([seed, SEED_EXTRA, j + 1])
            spa += [spa_row(A1, c1, rng_s, tag, "1x", tt1), spa_row(A2, c2, rng_s, tag, "LRS 3x (original)", tt2)]
    n1 = [k for k in cols if k in new]
    neff = {}
    for lab, keep in (("todas", cols), ("R19", [k for k in cols if k not in new]), ("nuevas", n1)):
        if len(keep) >= 2:
            neff[("1x", lab)] = neff_of(A1[keep])
            ka = [k for k in keep if k != LRS]
            neff[("LRS 3x (original)", lab)] = neff_of(A2[ka]) if len(ka) >= 2 else {r19.NEFF_MAIN: float(len(ka))}
    lrs_fam = [k for k in cols if bool(meta.loc[k, "parametrización_LRS3x"])]
    rank = {"IR, todas": (int((t1["IR"] > t1.loc[LRS, "IR"]).sum()) + 1, len(t1)),
            "Δg, todas": (int((t1["Δg"] > t1.loc[LRS, "Δg"]).sum()) + 1, len(t1)),
            "IR, parametrizaciones de la LRS 3x": (int((t1.loc[lrs_fam, "IR"] > t1.loc[LRS, "IR"]).sum()) + 1,
                                                   len(lrs_fam))}
    return {"t1": t1, "t2": t2, "spa": pd.DataFrame(spa), "neff": neff, "rank": rank,
            "T": len(Rw), "K": len(cols), "K_nuevas": len(n1), "desde": Rw.index[0], "hasta": Rw.index[-1]}


def reproduce_r19(res: dict) -> pd.DataFrame:
    """La parte de R19 en la ventana principal frente a lo que publica R19 (R19_PUBLISHED): los p con
    tres decimales redondeados como en R19 (f"{p:.3f}"), los N_eff al entero."""
    t1, t2, sp = res["t1"], res["t2"], res["spa"].set_index(["familia", "frente a"])
    got = {"reglas distintas": res["K"],
           "N_eff frente al 1x (Li y Ji)": round(res["neff"][("1x", "todas")][r19.NEFF_MAIN]),
           "N_eff frente a la LRS (Li y Ji)": round(res["neff"][("LRS 3x (original)", "todas")][r19.NEFF_MAIN]),
           "SPA frente al 1x": float(sp.loc[("todas", "1x"), "p_SPA"]),
           "SPA frente a la LRS": float(sp.loc[("todas", "LRS 3x (original)"), "p_SPA"]),
           "p LRS frente al 1x": float(t1.loc[LRS, "p"]),
           "Romano–Wolf < 5% frente al 1x": int((t1.p_RW < ALPHA).sum()),
           "Romano–Wolf < 5% frente a la LRS": int((t2.p_RW < ALPHA).sum())}
    shown = {k: (f"{v:.3f}" if isinstance(v, float) else str(v)) for k, v in got.items()}
    pub = {k: (f"{v:.3f}" if isinstance(v, float) else str(v)) for k, v in R19_PUBLISHED.items()}
    out = pd.DataFrame({"R25 (solo las reglas de R19)": shown, "publicado por R19": pub})
    out["valor sin redondear"] = [f"{v:.4f}" if isinstance(v, float) else "" for v in got.values()]
    out["coincide"] = np.where(out.iloc[:, 0] == out.iloc[:, 1], "sí", "NO")
    return out


# ───────────────────────────── cálculo completo ─────────────────────────────

def ml_forecasts(d: dict, cache: str | None) -> dict:
    """Previsiones de ML de R19 (r19.ml_forecasts), guardadas en cache/r19_ml.pkl si hay caché."""
    path = None
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        path = str(Path(cache) / "r19_ml.pkl")
    return r19.ml_forecasts(d, path)


def inputs(cache: str | None = None) -> dict:
    """Las series del censo: las de R19 (r19.build_census) y las de R21–R24 (new_rules), con sus fichas,
    su primer día y los segundos que costó cada parte."""
    t0 = time.time()
    secs = {}
    D = r13.load()
    d = D[SP]
    var_fc = r13.var_forecast(d)
    ml = ml_forecasts(d, cache)
    ml.pop("_segundos", None)
    C = r19.build_census(d, var_fc, ml)
    secs["censo de R19"] = time.time() - t0
    base = [k for k in C.R if k != BENCH]
    t1 = time.time()
    Rn, Mn = new_rules(D, cache)
    secs["reglas de R21–R24"] = time.time() - t1
    live = pd.Series({**{k: C.live[k] for k in base}, **Mn["vivo_desde"].to_dict()})
    meta = pd.concat([pd.DataFrame(C.meta).T.assign(vivo_desde=[C.live[k] for k in base],
                                                    operable=[not r19.is_same_close(k) for k in base]),
                      Mn])
    meta["fuente"] = [source_of(k) for k in meta.index]
    meta["vende_en_corto"] = [str(f).startswith(tuple(f"R22: {x}" for x in SHORT_FAMS)) for f in meta.familia]
    R = {**{k: C.R[k] for k in C.R}, **Rn}
    return {"R": R, "base": base, "meta": meta, "live": live, "idx": d["u"].index, "segundos": secs}


def compute(cache: str | None = None, inp: dict | None = None) -> dict:
    """Todo el cálculo de R25 (sin informe). cache: directorio para las previsiones de ML de R19 y
    las de R23 (se reutilizan si existen). inp: el resultado de inputs(), si ya se tiene."""
    inp = inp or inputs(cache)
    t0 = time.time()
    R, base, meta, live = inp["R"], inp["base"], inp["meta"], inp["live"]
    secs = dict(inp["segundos"])
    W = window_members(live, base, inp["idx"])
    end = min(R[k].last_valid_index() for k in base)
    res = {"meta": meta, "live": live, "ventanas": {}, "segundos": secs}
    t2 = time.time()
    for w in WINDOWS:
        start, names = W[w]
        cols = {}
        for k in names:
            r = R[k]
            if source_of(k) == "R22":
                r = absorb_from(r, start)
            cols[k] = r.loc[start:end]
        Rw_all = pd.DataFrame(cols)
        bad = Rw_all.columns[Rw_all.isna().any()].tolist()
        assert not bad, f"{w}: huecos en {bad[:5]}"
        keep, dup = r19.dedupe(Rw_all)
        Rw = Rw_all[keep]
        new = {k for k in keep if source_of(k) != "R19"}
        short = {k for k in keep if bool(meta.loc[k, "vende_en_corto"])}
        bench = R[BENCH].loc[start:end]
        out = analyse(Rw, bench, meta, new, short)
        out.update(construidas=len(names), dup=dup, excluidas=[k for k in live.index if k not in names])
        if w == "principal":
            r19_keep = [k for k in keep if source_of(k) == "R19"]
            base_res = analyse(Rw[r19_keep], bench, meta, set(), set(), extra_spa=False)
            out["solo_R19"] = base_res
            out["reproduce"] = reproduce_r19(base_res)
            # las series de R19 frente a su CSV publicado (si existe)
            path = r19.REPORTS / "r19_census" / "censo_rentabilidades_diarias.csv.gz"
            if path.exists():
                cen = pd.read_csv(path, index_col=0, parse_dates=True)
                com = [c for c in cen.columns if c in Rw.columns]
                out["csv_R19"] = {"columnas": cen.shape[1], "comunes": len(com),
                                  "dif_máx": float((cen[com] - Rw[com].reindex(cen.index)).abs().max().max())}
        res["ventanas"][w] = out
        secs[f"ventana {w}"] = time.time() - t2
        t2 = time.time()
    secs["análisis (las tres ventanas)"] = time.time() - t0
    secs["total"] = secs["censo de R19"] + secs["reglas de R21–R24"] + secs["análisis (las tres ventanas)"]
    return res


# ───────────────────────────── informe ─────────────────────────────

def fp(p: float) -> str:
    """p con tres decimales y coma, redondeado como en R19 (f"{p:.3f}"): el mismo p se escribe igual
    en los dos informes. Por debajo de 0,001, «< 0,001»."""
    if p is None or not np.isfinite(p):
        return "—"
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def peq(p: float) -> str:
    """«= 0,034» o «< 0,001», para escribir «p {peq(x)}»."""
    s_ = fp(p)
    return s_ if s_.startswith("<") else f"= {s_}"


def ptab(p: float) -> str:
    """p para las tablas (punto decimal, como el resto de las tablas del laboratorio), con el mismo
    redondeo que el texto."""
    if p is None or not np.isfinite(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def pp(x: float, d: int = 1) -> str:
    v = 100 * float(x)
    return ("+" if round(v, d) > 0 else "") + num_es(v, d) + " pp"


def fdate(x) -> str:
    return pd.Timestamp(x).strftime("%-d-%-m-%Y")


def short(name: str, n: int = 70) -> str:
    return name if len(name) <= n else name[:n - 1] + "…"


def names_txt(ks, n: int = 6) -> str:
    ks = list(ks)
    if not ks:
        return "ninguna"
    txt = "; ".join(f"«{k}»" for k in ks[:n])
    return txt + (f" y {len(ks) - n} más" if len(ks) > n else "")


def count_table(res: dict) -> pd.DataFrame:
    """Series por origen: construidas, en cada ventana y distintas tras quitar duplicados."""
    meta = res["meta"]
    rows = {}
    for s_ in SOURCES:
        ks = meta.index[meta.fuente == s_]
        row = {"construidas": len(ks)}
        for w in WINDOWS:
            V = res["ventanas"][w]
            row[f"{w}: en la ventana"] = len([k for k in ks if k not in V["excluidas"]])
            row[f"{w}: distintas"] = int((V["t1"].fuente == s_).sum())
        rows[s_] = row
    t = pd.DataFrame.from_dict(rows, orient="index")
    t.loc["total"] = t.sum()
    return t.astype(int)


def dup_table(res: dict) -> pd.DataFrame:
    """Reglas nuevas idénticas a otra en la ventana principal (y a cuál)."""
    V = res["ventanas"]["principal"]
    rows = []
    for k, ds in V["dup"].items():
        for x in ds:
            if source_of(x) != "R19":
                rows.append({"regla nueva": x, "idéntica a": k, "origen de la conservada": source_of(k)})
    return pd.DataFrame(rows)


def window_table(res: dict) -> pd.DataFrame:
    rows = {}
    for w in WINDOWS:
        V = res["ventanas"][w]
        ne = V["neff"]
        rng1 = [v for k, v in ne[("1x", "todas")].items() if not k.startswith("autovalores")]
        rows[WINDOW_LABEL[w]] = {
            "desde": V["desde"].date(), "hasta": V["hasta"].date(), "años": V["T"] / 252,
            "reglas distintas": V["K"], "de R19": V["K"] - V["K_nuevas"], "nuevas": V["K_nuevas"],
            "N_eff 1x: todas": ne[("1x", "todas")][r19.NEFF_MAIN],
            "N_eff 1x: R19": ne[("1x", "R19")][r19.NEFF_MAIN],
            "N_eff 1x: nuevas": ne[("1x", "nuevas")][r19.NEFF_MAIN],
            "N_eff 1x: otros estimadores": f"{num_es(min(rng1), 0)}–{num_es(max(rng1), 0)}",
            "N_eff LRS: todas": ne[("LRS 3x (original)", "todas")][r19.NEFF_MAIN],
            "N_eff LRS: R19": ne[("LRS 3x (original)", "R19")][r19.NEFF_MAIN],
            "N_eff LRS: nuevas": ne[("LRS 3x (original)", "nuevas")][r19.NEFF_MAIN]}
    return pd.DataFrame.from_dict(rows, orient="index")


def spa_table(res: dict) -> pd.DataFrame:
    out = []
    for w in WINDOWS:
        s_ = res["ventanas"][w]["spa"].copy()
        s_.insert(0, "ventana", WINDOW_LABEL[w])
        out.append(s_)
    t = pd.concat(out)
    t["Δg_mejor"] = t["Δg_mejor"] * 100
    t["mejor"] = t["mejor"].map(lambda k: short(k, 56))
    t["mayor t"] = t["mayor t"].map(lambda k: short(k, 56))
    t["p_SPA"] = t["p_SPA"].map(ptab)
    return t.rename(columns={"Δg_mejor": "Δg_mejor (pp)"})


def survivors(t: pd.DataFrame, col: str = "p_RW", only_new: bool = False) -> list[str]:
    """Reglas con p ajustado (columna col) por debajo del 5%, ordenadas por ese p."""
    x = t[t[col] < ALPHA].sort_values(col)
    if only_new:
        x = x[x.fuente != "R19"]
    return list(x.index)


def rw_table(res: dict) -> pd.DataFrame:
    rows = []
    for w in WINDOWS:
        V = res["ventanas"][w]
        for ref, t in (("1x", V["t1"]), ("LRS 3x (original)", V["t2"])):
            tn = t[t.fuente != "R19"]
            rows.append({"ventana": WINDOW_LABEL[w], "frente a": ref, "reglas": len(t),
                         "p < 5% sin corregir": int((t.p < ALPHA).sum()),
                         "nuevas con p < 5%": int((tn.p < ALPHA).sum()),
                         "RW < 5%: todas": len(survivors(t)),
                         "de ellas nuevas": len(survivors(t, only_new=True)),
                         "RW < 5%: familia de las nuevas": len(survivors(t, "p_RW_nuevas")),
                         "RW < 5%: familia de las operables": len(survivors(t, "p_RW_operables")),
                         "RW < 5%: familia sin cortos": len(survivors(t, "p_RW_sin_cortos"))})
    return pd.DataFrame(rows)


def rw_names(res: dict) -> pd.DataFrame:
    """Las reglas que sobreviven a Romano–Wolf en alguna familia (ventana principal)."""
    V = res["ventanas"]["principal"]
    rows = []
    for ref, t in (("1x", V["t1"]), ("LRS 3x (original)", V["t2"])):
        cols = [c for c in ("p_RW", "p_RW_nuevas", "p_RW_operables", "p_RW_sin_cortos") if c in t]
        hit = t[(t[cols] < ALPHA).any(axis=1)].sort_values("p")
        for k, r in hit.iterrows():
            rows.append({"frente a": ref, "regla": short(k, 80), "origen": r.fuente, "Δg (pp)": r["Δg"] * 100,
                         "IR": r["IR"], "p": ptab(r["p"]), **{c.replace("p_RW", "RW"): ptab(r[c]) for c in cols}})
    return pd.DataFrame(rows)


def rank_table(res: dict) -> pd.DataFrame:
    rows = {}
    for w in WINDOWS:
        V = res["ventanas"][w]
        t = V["t1"]
        row = {"IR frente al 1x": t.loc[LRS, "IR"], "Δg frente al 1x (pp)": t.loc[LRS, "Δg"] * 100,
               "p frente al 1x (sola)": ptab(t.loc[LRS, "p"])}
        for k, (r, n) in V["rank"].items():
            row[f"puesto por {k}"] = f"{r} de {n}"
        row["mejor por IR"] = short(t["IR"].idxmax(), 60)
        rows[WINDOW_LABEL[w]] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def top_new(res: dict, w: str = "principal", ref: str = "1x", n: int = 12) -> pd.DataFrame:
    V = res["ventanas"][w]
    t = V["t1"] if ref == "1x" else V["t2"]
    tn = t[t.fuente != "R19"].sort_values("p").head(n).copy()
    out = pd.DataFrame({"origen": tn.fuente, "Δg (pp)": tn["Δg"] * 100, "IR": tn["IR"],
                        "p": tn["p"].map(ptab), "RW todas": tn["p_RW"].map(ptab),
                        "RW nuevas": tn["p_RW_nuevas"].map(ptab),
                        "operable": np.where(tn["operable"].astype(bool), "sí", "no (mismo cierre)")})
    out.index = [short(k, 72) for k in tn.index]
    return out


def fig_strip(res: dict, rep: Report):
    """Δg frente a la LRS original de cada regla distinta de la ventana principal, por origen; con un
    círculo, las que sobreviven a Romano–Wolf frente a la LRS en la familia completa."""
    V = res["ventanas"]["principal"]
    t = V["t2"]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(FIG_W, 5.6))
    lo = max(float(np.nanpercentile(t["Δg"] * 100, 2)) - 1, -25.0)
    hi = max(float(t["Δg"].max() * 100) + 1.5, 4.0)
    for j, s_ in enumerate(SOURCES):
        x = np.clip(t.loc[t.fuente == s_, "Δg"].to_numpy(float) * 100, lo, None)
        jit = rng.uniform(-0.28, 0.28, len(x))
        jit[(t.loc[t.fuente == s_, "p_RW"] < ALPHA).to_numpy()] = 0.0    # las que sobreviven, en su círculo
        y = np.full(len(x), -j) + jit
        ax.scatter(x, y, s=22, color=COLORS[s_], alpha=0.75, edgecolor="white", linewidth=0.4, zorder=3)
        sv = t.index[(t.fuente == s_) & (t.p_RW < ALPHA)]
        if len(sv):
            xs = t.loc[sv, "Δg"].to_numpy(float) * 100
            ax.scatter(xs, np.full(len(xs), -j), s=110, facecolor="none", edgecolor=INK, linewidth=1.4, zorder=4)
    ax.axvline(0, color=INK, lw=1)
    ax.text(-0.3, 0.62, "← crece menos que la LRS original", color=MUTED, fontsize=12, va="center", ha="right")
    ax.text(0.3, 0.62, "más →", color=MUTED, fontsize=12, va="center", ha="left")
    ax.set_yticks([-j for j in range(len(SOURCES))],
                  [f"{s_} ({int((t.fuente == s_).sum())})" for s_ in SOURCES], fontsize=12)
    ax.set_ylim(-len(SOURCES) + 0.4, 0.95)
    ax.set_xlim(lo - 0.5, hi)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    n_out = int((t["Δg"] * 100 < lo).sum())
    ax.set(xlabel="Δg frente a la LRS 3x original (pp/año)" +
           (f"\n({n_out} reglas por debajo de {num_es(lo, 0)} se dibujan en el borde)" if n_out else ""),
           title=f"S&P 500, {V['desde'].year}–{V['hasta'].year}: cada regla del censo ampliado\n"
                 "frente a la LRS original, por módulo de origen")
    comma_fig(fig)
    rep.fig(fig, "censo_frente_a_lrs",
            "Cada punto es una regla distinta del censo ampliado en la ventana principal: su diferencia de crecimiento "
            "logarítmico anual con la LRS 3x original. Filas: el módulo de origen (entre paréntesis, cuántas reglas "
            "distintas). Con un círculo negro, las que superan a la original tras la corrección de Romano–Wolf sobre "
            "toda la familia")


def write_report(res: dict) -> Report:
    rep = Report("r25_census_plus", "R25 — Censo ampliado: el censo del artículo 2 con las reglas del artículo 3")
    for f in rep.dir.glob("table_*.csv"):       # Report numera las tablas por línea: sin huérfanas
        f.unlink()
    rep.p(__doc__.split("\n", 2)[2].split("\nMedidas")[0])
    VA, VB, VC = (res["ventanas"][w] for w in WINDOWS)
    rp = VA["reproduce"]
    ok = (rp["coincide"] == "sí").all()
    spa = {w: res["ventanas"][w]["spa"].set_index(["familia", "frente a"]) for w in WINDOWS}
    L0 = "LRS 3x (original)"

    def S(w, fam, ref):
        return spa[w].loc[(fam, ref)]

    t1A, t2A = VA["t1"], VA["t2"]
    rkA, neA = VA["rank"], VA["neff"]
    base = VA["solo_R19"]
    sv_all = survivors(t2A)
    sv_short = [k for k in sv_all if bool(res["meta"].loc[k, "vende_en_corto"])]
    sv_ns = survivors(t2A, "p_RW_sin_cortos")
    sv_new = survivors(t2A, "p_RW_nuevas")
    ns = t2A[~t2A.index.isin(res["meta"].index[res["meta"].vende_en_corto.astype(bool)])]
    best_ns = ns["Δg"].idxmax()
    best1_new = t1A[t1A.fuente != "R19"]["IR"].idxmax()
    sv1, sv1_op = survivors(t1A), survivors(t1A, "p_RW_operables")
    sv1_new = survivors(t1A, "p_RW_nuevas")

    n_s = len(sv_short)
    who_short = ("la única vende" if n_s == 1 == len(sv_all) else "una vende" if n_s == 1 else
                 "las dos venden" if n_s == 2 == len(sv_all) else "todas venden" if n_s == len(sv_all) else
                 f"{n_s} venden")

    def win_txt(V, w):
        a = survivors(V["t2"], "p_RW_sin_cortos")
        return (f"desde {V['desde'].year} ({V['K']} reglas), SPA p {peq(S(w, 'todas', L0)['p_SPA'])} con todas y "
                f"p {peq(S(w, 'sin cortos', L0)['p_SPA'])} sin las que venden en corto, con "
                f"{len(a)} supervivientes de Romano–Wolf sin cortos")

    # ── resumen ──
    rep.h("Resumen")
    lines = [
        f"**Comprobación.** Con solo las reglas de R19, la ventana principal reproduce el censo del artículo 2: "
        f"{base['K']} reglas distintas, N_eff (Li y Ji) de {rp.iloc[1, 0]} frente al 1x y {rp.iloc[2, 0]} frente a "
        f"la LRS, SPA p {peq(float(rp.iloc[3, 2]))} frente al 1x y p {peq(float(rp.iloc[4, 2]))} frente a la LRS "
        f"original, y p {peq(float(rp.iloc[5, 2]))} para la LRS original frente al 1x "
        + ("(todo coincide con lo que publica R19)." if ok else "(algo NO coincide con R19: tabla 1).")
        + (f" Las {VA['csv_R19']['comunes']} series coinciden con el CSV de R19 (diferencia máxima "
           f"{num_es(VA['csv_R19']['dif_máx'], 0)})." if "csv_R19" in VA else ""),
        f"**Censo ampliado.** Se reconstruyeron {len(res['meta'])} series en el S&P 500 "
        f"({int((res['meta'].fuente == 'R19').sum())} del censo de R19 y {int((res['meta'].fuente != 'R19').sum())} "
        f"de R21–R24). En la ventana principal, del {fdate(VA['desde'])} al {fdate(VA['hasta'])} "
        f"({num_es(VA['T'] / 252, 1)} años), hay {VA['K']} reglas distintas: {VA['K'] - VA['K_nuevas']} de R19 y "
        f"{VA['K_nuevas']} nuevas. {len(VA['excluidas'])} series empiezan más tarde (las de bonos de R24 y las de "
        f"VIX de R23) y entran en las ventanas desde {VB['desde'].year} ({VB['K']} reglas) y desde "
        f"{VC['desde'].year} ({VC['K']}).",
        f"**Pruebas efectivas.** El número efectivo de pruebas independientes (Li y Ji) pasa de "
        f"{num_es(neA[('1x', 'R19')][r19.NEFF_MAIN], 0)} a {num_es(neA[('1x', 'todas')][r19.NEFF_MAIN], 0)} frente "
        f"al 1x y de {num_es(neA[('LRS 3x (original)', 'R19')][r19.NEFF_MAIN], 0)} a "
        f"{num_es(neA[('LRS 3x (original)', 'todas')][r19.NEFF_MAIN], 0)} frente a la LRS original; las "
        f"{VA['K_nuevas']} nuevas solas equivalen a {num_es(neA[('1x', 'nuevas')][r19.NEFF_MAIN], 0)} y "
        f"{num_es(neA[('LRS 3x (original)', 'nuevas')][r19.NEFF_MAIN], 0)}.",
        f"**Frente a la LRS original.** Con las {int(S('principal', 'todas', L0)['reglas'])} alternativas, SPA p "
        f"{peq(S('principal', 'todas', L0)['p_SPA'])} y Romano–Wolf deja {len(sv_all)} ({names_txt(sv_all)})"
        + (f"; {who_short} en corto el −3x a diario (R22): su ventaja es una diferencia de comisiones casi determinista que depende "
           "del préstamo de títulos y de que el bróker remunere el producto del corto; sin esa remuneración "
           "desaparece (R22, sección 7)" if sv_short else "")
        + f". Sin las reglas que venden en corto, SPA p {peq(S('principal', 'sin cortos', L0)['p_SPA'])} (la que "
        f"más crece, «{best_ns}», {pp(t2A.loc[best_ns, 'Δg'])}/año, p nominal {fp(t2A.loc[best_ns, 'p'])}) y "
        + ("Romano–Wolf no deja ninguna" if not sv_ns else f"Romano–Wolf deja {len(sv_ns)} ({names_txt(sv_ns)})")
        + f". Solo con las {int(S('principal', 'nuevas', L0)['reglas'])} nuevas, SPA p "
        f"{peq(S('principal', 'nuevas', L0)['p_SPA'])} y {len(sv_new)} supervivientes "
        f"({names_txt(sv_new)}). En las otras ventanas: {win_txt(VB, '1962')}; {win_txt(VC, 'todas')}.",
        f"**Frente al 1x.** SPA p {peq(S('principal', 'todas', '1x')['p_SPA'])} con todas, "
        f"p {peq(S('principal', 'operables', '1x')['p_SPA'])} solo con las operables y "
        f"p {peq(S('principal', 'nuevas', '1x')['p_SPA'])} solo con las nuevas. Romano–Wolf deja {len(sv1)} reglas "
        f"({names_txt(sv1, 8)}); en la familia de las operables, {len(sv1_op)} ({names_txt(sv1_op, 8)}); en la de "
        f"las nuevas, {len(sv1_new)} ({names_txt(sv1_new, 8)}). La mejor nueva por IR es «{best1_new}» (IR "
        f"{num_es(t1A.loc[best1_new, 'IR'], 2)}, Δg {pp(t1A.loc[best1_new, 'Δg'])}/año, p_RW "
        f"{fp(t1A.loc[best1_new, 'p_RW'])}"
        + (", al mismo cierre, que antes de 1990 no era operable)." if r19.is_same_close(best1_new) else ")."),
        f"**La LRS original.** Contrastada sola, como hipótesis fijada antes (Gayed y Bilello 2016), frente al 1x: "
        f"Δg {pp(t1A.loc[LRS, 'Δg'])}/año, IR {num_es(t1A.loc[LRS, 'IR'], 2)}, p {peq(t1A.loc[LRS, 'p'])}, el mismo "
        f"p que en R19 y en el artículo 2 (los remuestreos no cambian al añadir reglas). En el censo ampliado es la "
        f"{rkA['IR, todas'][0]}.ª de {rkA['IR, todas'][1]} por IR frente al 1x (la {rkA['Δg, todas'][0]}.ª por Δg) "
        f"y la {rkA['IR, parametrizaciones de la LRS 3x'][0]}.ª de las "
        f"{rkA['IR, parametrizaciones de la LRS 3x'][1]} parametrizaciones de la LRS 3x (en R19, la "
        f"{base['rank']['IR, todas'][0]}.ª de {base['rank']['IR, todas'][1]} y la "
        f"{base['rank']['IR, parametrizaciones de la LRS 3x'][0]}.ª de "
        f"{base['rank']['IR, parametrizaciones de la LRS 3x'][1]}).",
    ]
    for s_ in lines:
        rep.p(f"- {s_}")

    # ── 1. comprobación ──
    rep.h("1. Comprobación: la parte de R19")
    rep.p("**Tabla 1. Censo de R19 reconstruido aquí (solo sus reglas, ventana principal, generador y orden de "
          "remuestreo de r19.analyse) frente a las cifras que publica R19, con el mismo redondeo.**")
    rep.table(rp)
    if "csv_R19" in VA:
        c = VA["csv_R19"]
        rep.p(f"De las {c['columnas']} series del CSV de rentabilidades diarias de R19, {c['comunes']} están entre las "
              f"distintas de aquí y coinciden día a día (diferencia máxima {num_es(c['dif_máx'], 0)}).")
    if not ok:
        rep.p("**Atención: alguna cifra no coincide con R19.**")

    # ── 2. el censo ──
    rep.h("2. El censo ampliado")
    rep.p("**Tabla 2. Series por origen: construidas, dentro de cada ventana y distintas (sin las idénticas a otra "
          "ya contada en esa ventana).**")
    rep.table(count_table(res))
    fams = res["meta"][res["meta"].fuente != "R19"].groupby(["fuente", "familia"]).size().rename("series")
    rep.p("**Tabla 3. Familias nuevas y número de series construidas.**")
    rep.table(fams.reset_index(), index=False)
    dt = dup_table(res)
    if len(dt):
        rep.p(f"**Tabla 4. Reglas nuevas idénticas a otra en la ventana principal ({len(dt)}).** Cuentan una vez, con "
              "el nombre de la primera que se construyó (reglas de margen que no llegan a activarse, la guarda fija "
              "de la salida publicada de R24 y la LRS 2x sintética de precio de R22, que es la de R13).")
        rep.table(dt, index=False)
    exc = res["meta"].loc[VA["excluidas"]]
    exc = exc.groupby(["fuente", "familia"]).agg(series=("vivo_desde", "size"), desde=("vivo_desde", "min"))
    exc["desde"] = [x.date() for x in exc["desde"]]
    rep.p("**Tabla 5. Series que quedan fuera de la ventana principal porque empiezan después (entran en las "
          "ventanas siguientes).**")
    rep.table(exc.reset_index(), index=False)

    # ── 3. ventanas y pruebas efectivas ──
    rep.h("3. Ventanas y número efectivo de pruebas")
    rep.p("**Tabla 6. Cada ventana: fechas, reglas distintas y número efectivo de pruebas independientes (Li y Ji, "
          "con los excesos sobre el 1x o, para las alternativas, sobre la LRS original), con el censo de R19, con "
          "las nuevas y con todas; y el rango de los demás estimadores de r19.n_eff frente al 1x.**")
    rep.table(window_table(res), digits=1)
    rep.p("El número efectivo mide a cuántas pruebas independientes equivalen las reglas probadas, dada su "
          "correlación: añadir reglas muy parecidas a las que ya había lo sube poco. Es una cota inferior del "
          "número de caminos que ha recorrido el laboratorio: lo probado en otros mercados, con otros supuestos de "
          "datos o de costes, o descartado antes de programarlo, no está aquí.")

    # ── 4. SPA y Romano–Wolf ──
    rep.h("4. Contrastes que descuentan la búsqueda")
    rep.p(f"**Tabla 7. SPA de Hansen (versión consistente, stats.spa_test; bootstrap estacionario de bloque {BLOCK}, "
          f"{num_es(B_SPA, 0)} remuestreos) por ventana, referencia y familia: todas las reglas, solo las nuevas, "
          "solo las operables (sin las de mismo cierre) y sin las que venden en corto (el corto del −3x, el par "
          "corto y el lever-arb de R22). «mejor»: la de más Δg (pp/año); «mayor t»: la de mayor estadístico "
          "estudentizado, la que decide el SPA.**")
    rep.table(spa_table(res), index=False, digits=3)
    rep.p("**Tabla 8. Romano–Wolf al 5% (step-down con los remuestreos de r19.boot_test): cuántas reglas sobreviven "
          "en toda la familia, cuántas de ellas son nuevas y cuántas sobreviven si la familia se limita a las nuevas, "
          "a las operables o a las que no venden en corto (una familia menor descuenta menos).**")
    rep.table(rw_table(res), index=False, digits=3)
    rn = rw_names(res)
    if len(rn):
        rep.p("**Tabla 9. Ventana principal: las reglas que sobreviven a Romano–Wolf en alguna de esas familias "
              "(Δg en pp/año; p nominal y p ajustado en cada familia; vacío si la regla no pertenece a ella).**")
        rep.table(rn, index=False, digits=3)
    rep.p(f"**Tabla 10. Ventana principal: las 12 reglas nuevas con menor p frente al 1x (Δg en pp/año; p unilateral "
          f"con el error típico del bootstrap; Romano–Wolf sobre toda la familia de {VA['K']} y solo sobre las "
          f"{VA['K_nuevas']} nuevas).**")
    rep.table(top_new(res, "principal", "1x"), digits=2)
    rep.p("**Tabla 11. Lo mismo frente a la LRS 3x original.**")
    rep.table(top_new(res, "principal", "LRS"), digits=2)
    fig_strip(res, rep)
    rep.p("Una regla con p nominal pequeño no es una mejora: con cientos de pruebas muy correlacionadas, alguna "
          "sale por azar. Romano–Wolf controla la probabilidad de declarar alguna falsa (el error familiar) con la "
          "dependencia real entre las reglas; el SPA contrasta si la mejor de la familia supera a la referencia. Las "
          "reglas de mismo cierre no eran operables antes de 1990 (R13, R15) y entran en el censo como en R19. El "
          "corto diario del −3x de R22 sobrevive frente a la original por una diferencia de comisiones casi sin "
          "varianza (un t enorme con una Δg de décimas de punto), no por una señal mejor, y depende de supuestos de "
          "coste que R22 no puede comprobar.")

    # ── 5. la original ──
    rep.h("5. El puesto de la LRS original")
    rep.p("**Tabla 12. La LRS 3x original frente al 1x en cada ventana y su puesto entre las reglas distintas (por IR "
          "y por Δg frente al 1x) y entre las parametrizaciones de la LRS 3x (las que marca R19 y las «LRS "
          "reempaquetadas» de R24).**")
    rep.table(rank_table(res), digits=3)
    rep.p("El p de la LRS original frente al 1x es el de una hipótesis fijada antes de ver los datos (la de Gayed y "
          "Bilello), que se contrasta sola; no se corrige por la búsqueda del censo, que se descuenta a la regla "
          "que el censo elige. En la ventana principal coincide con el de R19 y el del artículo 2 porque los "
          "remuestreos de r19.boot_test solo dependen del número de sesiones y del generador. En las ventanas más "
          "cortas es otro p, con menos historia.")

    # ── 6. límites ──
    rep.h("6. Límites")
    for s_ in (
        "Todo es dentro de muestra y en un solo mercado. Las reglas de R21–R24 llegan con sus propias familias y "
        "contrastes en sus informes; aquí se juntan para descontar la búsqueda completa en el S&P 500.",
        "Las ventanas más cortas (desde 1962 y desde 2006) tienen menos potencia: que nada sobreviva en ellas dice "
        "poco por sí solo.",
        "Las reglas de R22 por encima de 3x (la curva de apalancamiento) no tienen un ETF cotizado; las del corto "
        "del −3x, el par corto y el lever-arb dependen del préstamo de títulos y de la remuneración del producto "
        "del corto (R22, sección 7). Se cuentan como pruebas porque se probaron.",
        f"Con {num_es(B_BOOT, 0)} remuestreos el error de Monte Carlo de un p cercano a 0,05 es de unos ±0,005.",
    ):
        rep.p(f"- {s_}")
    secs = res["segundos"]
    rep.p("Tiempo de cálculo: " + "; ".join(f"{k} {num_es(v / 60, 1)} min" for k, v in secs.items()) + ".")
    rep.save()
    return rep


def run(cache: str | None = None):
    """Calcula el censo ampliado y escribe reports/r25_census_plus/README.md. cache: directorio donde
    guardar (y reutilizar) las previsiones de ML de R19 y las de R23."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    res = compute(cache)
    write_report(res)
    return res


if __name__ == "__main__":
    run()
