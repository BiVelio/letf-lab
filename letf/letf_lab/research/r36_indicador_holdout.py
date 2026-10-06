"""R36 — Evaluación única de la prueba reservada del artículo 7 (1-1-2019 → 25-9-2026).

ESTE MÓDULO EVALÚA LA PRUEBA RESERVADA. run() solo debe llamarse una vez, a propósito, cuando se
decida mirar el resultado: deja una marca (letf/preregistro/articulo7_holdout_ejecutado.json) y
se niega a volver a correr si la marca existe, termine como termine.

Qué hace run(), en este orden:
1. Carga la configuración congelada (letf/preregistro/articulo7_config_congelada.json, escrita por
   r35_indicador.freeze con datos hasta el 31-12-2018) y comprueba el SHA-256 de cada archivo de
   código que lista: si alguno cambió, se niega a correr.
2. Comprueba que los parámetros del código (r35_indicador.Config, fechas, rupturas de serie,
   hipótesis y reglas) son los congelados.
3. Crea la marca de ejecución (en exclusiva: si ya existe, se niega). A partir de aquí la prueba
   cuenta como mirada.
4. Carga los datos hasta el 25-9-2026 con los mismos cargadores (r35_indicador.load_all) y calcula
   el indicador en tiempo real con el mismo código (r35_indicador.compute): los modelos se
   reestiman con ventana creciente como en el desarrollo; ningún parámetro se elige de nuevo.
5. Comprueba que lo estimado con el desarrollo (s de cada fondo, vencimientos de los bonos, prima
   de la pérdida de crecimiento) y las salidas del indicador hasta el 31-12-2018 coinciden con las
   congeladas: si no, aborta sin evaluar (los datos o el código del desarrollo cambiaron).
6. Evalúa H1–H8 en la ventana 1-1-2019 → 25-9-2026 con r35_indicador.hypotheses y las reglas de
   decisión congeladas, escribe reports/r36_indicador_holdout (README.md y resultados.json) y
   cierra la marca como «completado».

Si run() aborta en las fases 4–5 (carga, cálculo o comprobaciones), la marca queda como «abortado»
con la fase y estadisticos_de_la_prueba_calculados = false: no se ha calculado ningún estadístico
de la prueba y es decisión humana, tras corregir la causa, borrar la marca y volver a correr.

Las pruebas de este módulo usan solo datos sintéticos y nunca llaman a run() con la configuración
y la marca reales sin que antes falle una comprobación.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from letf_lab.report import Report
from letf_lab.research import r34_harri as r34
from letf_lab.research import r35_indicador as r35

log = logging.getLogger(__name__)

CONFIG = r35.FROZEN
MARKER = r35.LETF / "preregistro" / "articulo7_holdout_ejecutado.json"
OUT = "r36_indicador_holdout"
TEST_START, END, DEV_END = r35.TEST_START, r35.END, r35.DEV_END
REL_TOL = 1e-9


class FrozenError(RuntimeError):
    """El código, los parámetros o los datos del desarrollo no son los congelados."""


class AlreadyRun(RuntimeError):
    """La prueba reservada ya se evaluó (existe la marca de ejecución)."""


# ───────────────────────────── comprobaciones ─────────────────────────────

def load_config(path: Path = CONFIG) -> dict:
    path = Path(path)
    if not path.exists():
        raise FrozenError(f"falta la configuración congelada {path}: ejecuta r35_indicador.run() antes")
    return json.loads(path.read_text())


def verify_hashes(doc: dict, root: Path = r35.LETF) -> None:
    """Falla si algún archivo de doc["sha256"] (rutas relativas a root) falta o cambió."""
    bad = []
    for rel, h in doc["sha256"].items():
        p = Path(root) / rel
        if not p.exists():
            bad.append(f"{rel}: no existe")
        elif r35.sha256(p) != h:
            bad.append(f"{rel}: SHA-256 distinto")
    if bad:
        raise FrozenError("el código no es el congelado; no se evalúa nada:\n  " + "\n  ".join(bad))


def prereg_changed(doc: dict, root: Path = r35.LETF) -> bool | None:
    """¿Cambió el prerregistro después de congelar? (p. ej., al anotar «Desviaciones»). No bloquea:
    se informa en el informe y en la marca."""
    pre = doc.get("prerregistro", {})
    p = Path(root) / Path(pre.get("archivo", "letf/x")).relative_to("letf")
    return None if not p.exists() or "sha256" not in pre else r35.sha256(p) != pre["sha256"]


def config_from(doc: dict) -> r35.Config:
    return r35.Config.from_dict(doc["parametros"])


def verify_constants(doc: dict) -> None:
    """Los parámetros, fechas, rupturas e hipótesis del código son los congelados."""
    bad = []
    if config_from(doc) != r35.Config():
        diff = {k: (v, getattr(r35.Config(), k)) for k, v in config_from(doc).to_dict().items()
                if r35.Config().to_dict()[k] != v}
        bad.append(f"parámetros distintos (congelado, código): {diff}")
    m = doc["muestras"]
    if (m["desarrollo_fin"], m["prueba_inicio"], m["prueba_fin"]) != (str(DEV_END.date()), str(TEST_START.date()),
                                                                     str(END.date())):
        bad.append(f"fechas de las muestras distintas: {m}")
    if doc["rupturas_de_serie"] != {k: list(v) for k, v in r35.BOND_BREAKS.items()}:
        bad.append("rupturas de serie distintas")
    if doc["hipotesis"] != r35.HYPOTHESES:
        bad.append("hipótesis o reglas de decisión distintas")
    if bad:
        raise FrozenError("; ".join(bad))


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(1.0, abs(a), abs(b))


def verify_estimates(doc: dict, R: dict) -> None:
    """s, comisiones, vencimientos y primas del desarrollo recalculados con los datos cargados."""
    now = r35.dev_estimates(R)
    bad = []
    for grp, vals in doc["estimados_en_desarrollo"].items():
        for k, v in vals.items():
            w = now.get(grp, {}).get(k)
            if w is None or not _close(float(v), float(w)):
                bad.append(f"{grp}[{k}]: congelado {v}, ahora {w}")
    if bad:
        raise FrozenError("lo estimado con el desarrollo no coincide:\n  " + "\n  ".join(bad))


def verify_fingerprints(doc: dict, R: dict) -> None:
    """Las salidas del indicador hasta el 31-12-2018, calculadas ahora con los datos hasta 2026, son
    las del desarrollo (el cálculo no mira el futuro y los datos del desarrollo no cambiaron)."""
    now = r35.fingerprints(R, DEV_END)
    bad = []
    for k, v in doc["huellas_desarrollo"].items():
        w = now.get(k)
        if w is None:
            bad.append(f"{k}: falta")
            continue
        for f in ("días", "sha256_L", "desde", "L_final"):
            if f in v and v[f] != w.get(f):
                bad.append(f"{k}.{f}: congelado {v[f]}, ahora {w.get(f)}")
        if "suma_L_star" in v and not _close(v["suma_L_star"], w["suma_L_star"]):
            bad.append(f"{k}.suma_L_star: congelado {v['suma_L_star']}, ahora {w['suma_L_star']}")
    if bad:
        raise FrozenError("las salidas del desarrollo no coinciden con las congeladas:\n  " + "\n  ".join(bad))


# ───────────────────────────── marca de ejecución única ─────────────────────────────

def claim(marker: Path, doc: dict) -> None:
    """Crea la marca en exclusiva (falla si ya existe): la prueba cuenta como mirada desde aquí."""
    marker = Path(marker)
    rec = {"estado": "en curso", "inicio": pd.Timestamp.now().isoformat(timespec="seconds"),
           "configuracion_sha256": hashlib_json(doc)}
    try:
        with open(marker, "x") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=2)
    except FileExistsError:
        prev = marker.read_text()
        raise AlreadyRun(f"la prueba reservada ya se evaluó (o se intentó): {marker}\n{prev}") from None


def finish(marker: Path, state: str, **extra) -> None:
    marker = Path(marker)
    rec = json.loads(marker.read_text())
    rec.update(estado=state, fin=pd.Timestamp.now().isoformat(timespec="seconds"), **extra)
    marker.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str) + "\n")


def hashlib_json(doc: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


# ───────────────────────────── evaluación ─────────────────────────────

def window_end(R: dict) -> pd.Timestamp:
    return min(END, min(mk.u.index.max() for mk in R["markets"].values()))


def evaluate(R: dict, cfg: r35.Config, doc: dict, start=TEST_START) -> dict:
    """H1–H8 en [start, fin de los datos ≤ END] con las reglas congeladas (r35.hypotheses) más la
    descripción de las carteras en la ventana."""
    end = window_end(R)
    H = r35.hypotheses(R, start, end, cfg, doc["estimados_en_desarrollo"]["prima_desarrollo"])
    desc = {}
    for name, F in R["ret"].items():
        W = F[(F.index >= pd.Timestamp(start)) & (F.index <= end)]
        E = R["expo"][name]
        desc[name] = r35.growth_stats(W, E.reindex(W.index), R["markets"][name].rf)
    H["descripción"] = desc
    H["resumen"] = summary(H, cfg.h_main)
    return H


def summary(H: dict, h_main: int) -> dict:
    """Una línea por hipótesis: confirmada, no confirmada o parcial (h = h_main decide; h = 63 es
    sensibilidad y se ve en las tablas)."""
    out = {}
    h1 = {}
    for key in ("H1", "H6"):
        for (name, h), c in H.get(key, {}).items():
            if h != h_main:
                continue
            d = c["decisión"]
            state = "confirmada" if all(d.values()) else ("parcial" if any(d.values()) else "no confirmada")
            h1.setdefault(key, {})[name] = {"estado": state, **{f"mejor con {k}": v for k, v in d.items()}}
    out.update(h1)
    if "H2" in H:
        out["H2"] = {k: "confirmada" if v else "no confirmada" for k, v in H["H2"]["decisión"].items()}
    out["H3"] = {k: ("confirmada (no supera a la LRS)" if v["decisión (no supera)"] else "no confirmada (supera a la LRS)")
                 + f", p SPA = {v['p SPA']:.3f}" for k, v in H.get("H3", {}).items()}
    out["H4"] = {k: "confirmada" if v["decisión"] else "no confirmada" for k, v in H.get("H4", {}).items()}
    out["H5"] = {k: "confirmada" if v["decisión"] else "no confirmada" for k, v in H.get("H5", {}).items()
                 if k in r35.EQUITY_MARKETS}
    if "H7" in H:
        out["H7"] = {"crecimiento": H["H7"]["decisión crecimiento"], "caída frente a TMF": H["H7"]["decisión caída"],
                     "estado": "confirmada" if H["H7"]["decisión"] else "no confirmada"}
    if "H8" in H:
        out["H8"] = {"no supera a la LRS ni a la HFEA": H["H8"]["decisión no supera"],
                     "menor caída que la HFEA": H["H8"]["decisión caída"],
                     "estado": "confirmada" if H["H8"]["decisión"] else "no confirmada"}
    return out


def summary_lines(H: dict, h_main: int) -> list[str]:
    """El resumen en frases, con la cifra que decide cada hipótesis."""
    from letf_lab.report import num_es, pct_es

    yn = {True: "sí", False: "no"}
    st = {True: "confirmada", False: "no confirmada"}
    p3 = lambda p: "< 0,001" if p < 0.001 else "= " + num_es(p, 3)  # noqa: E731
    out = []
    for key in ("H1", "H6"):
        for (name, h), c in H.get(key, {}).items():
            if h != h_main:
                continue
            d = c["decisión"]
            state = "confirmada" if all(d.values()) else ("parcial" if any(d.values()) else "no confirmada")
            dm = c["dm"].set_index(["frente a", "pérdida"])["p (una cola)"]
            ps = "; ".join(f"{b} con {l}: p {p3(v)}" for (b, l), v in dm.items())
            out.append(f"**{key} · {name}**: {state} (mejor con QLIKE: {yn[d['QLIKE']]}; con la pérdida de "
                       f"crecimiento: {yn[d['crecimiento']]}; Diebold–Mariano: {ps}).")
    if "H2" in H:
        t = H["H2"]["tabla"]
        for name, ok in H["H2"]["decisión"].items():
            sub = t[t["estrategia"].str.startswith(name)]
            ps = "; ".join(f"frente al {r['frente a'].split(': ')[-1]}: Δg {pct_es(r['Δg'])}, p RW {p3(r['p Romano–Wolf'])}"
                           for _, r in sub.iterrows())
            out.append(f"**H2 · {name}**: {st[ok]} ({ps}).")
    for name, v in H.get("H3", {}).items():
        out.append(f"**H3 · {name}**: {'confirmada: no supera' if v['decisión (no supera)'] else 'no confirmada: supera'} "
                   f"a la LRS (p SPA {p3(v['p SPA'])}; mejor variante {v['mejor']}, Δg {pct_es(v['Δg mejor'])}).")
    for name, v in H.get("H4", {}).items():
        out.append(f"**H4 · {name}**: {st[v['decisión']]} (a igual exposición: caída máxima {pct_es(v['maxDD a'], 1)} "
                   f"frente a {pct_es(v['maxDD b'], 1)} de la LRS; probabilidad de perder el 50% "
                   f"{pct_es(v['P(pérdida ≥ 50%) a'], 1)} frente a {pct_es(v['P(pérdida ≥ 50%) b'], 1)}).")
    for name, v in H.get("H5", {}).items():
        if name not in r35.EQUITY_MARKETS:
            continue
        out.append(f"**H5 · {name}**: {st[v['decisión']]} (corr(L*, E[IV]) = {num_es(v['corr(L*, E[IV])'], 2)}, "
                   f"p {p3(v['p corr < 0'])}; corr(L*, S) = {num_es(v['corr(L*, S)'], 2)}; parte de la varianza "
                   f"{num_es(v['parte de la varianza'], 2)}).")
    if "H7" in H:
        t, d = H["H7"]["tabla"], H["H7"]["caídas"]
        ps = "; ".join(f"frente al {r['frente a']}: Δg {pct_es(r['Δg'])}, p RW {p3(r['p Romano–Wolf'])}"
                       for _, r in t.iterrows())
        out.append(f"**H7**: {st[H['H7']['decisión']]} (crecimiento: {yn[H['H7']['decisión crecimiento']]}, {ps}; "
                   f"caída máxima {pct_es(d['maxDD a'], 1)} frente a {pct_es(d['maxDD b'], 1)} de TMF).")
    if "H8" in H:
        v = H["H8"]
        ps = "; ".join(f"frente a {b}: p SPA {p3(x['p SPA'])}" for b, x in v["spa"].items())
        out.append(f"**H8**: {st[v['decisión']]} (no supera a la LRS ni a la HFEA: {yn[v['decisión no supera']]}, {ps}; "
                   f"caída máxima {pct_es(v['caídas']['maxDD a'], 1)} frente a {pct_es(v['caídas']['maxDD b'], 1)} "
                   "de la HFEA 55/45).")
    return out


# ───────────────────────────── informe ─────────────────────────────

def yes_no(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas de decisión (booleanas) como «sí» / «no»."""
    df = df.copy()
    for c in df.columns:
        if str(c).startswith("decisión"):
            df[c] = df[c].map(lambda v: {True: "sí", False: "no"}.get(bool(v), v) if isinstance(v, (bool, np.bool_))
                              else v)
    return df


class _Report(Report):
    """Report en un directorio cualquiera (las pruebas escriben en uno temporal)."""

    def __init__(self, out_dir: Path, title: str):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lines = [f"# {title}", ""]


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, pd.DataFrame):
        return json.loads(x.to_json(orient="split", date_format="iso", force_ascii=False))
    if isinstance(x, pd.Series):
        return None                                        # series diarias: no van al JSON
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def write_report(H: dict, R: dict, cfg: r35.Config, doc: dict, out_dir: Path) -> Path:
    from letf_lab.research.r26_vix_data import Tables

    start, end = H["ventana"]
    rep = _Report(out_dir, "R36 — Prueba reservada del artículo 7 (evaluación única)")
    tab = Tables(rep)
    d = r34.dmy
    rep.p(f"Ventana: {d(start)} → {d(end)}. Configuración congelada: `{Path(CONFIG).name}` "
          f"(commit base {doc['congelado']['commit_base'][:10]}, congelada el {doc['congelado']['fecha']}); "
          f"{len(doc['sha256'])} archivos de código verificados por SHA-256; salidas del desarrollo verificadas. Evaluación "
          "única: la marca de ejecución impide repetirla. ¿Cambió el prerregistro después de congelar (p. ej., para "
          f"anotar desviaciones)? {({True: 'sí', False: 'no', None: 'no se pudo comprobar'})[prereg_changed(doc)]}.")
    rep.h("Resumen")
    rep.p("\n".join(f"- {line}" for line in summary_lines(H, cfg.h_main)))
    rep.h("Hipótesis y reglas congeladas")
    tab("Hipótesis y reglas de decisión (congeladas antes de la prueba)",
        pd.DataFrame({k: {"hipótesis": v["texto"], "regla": v["regla"]} for k, v in doc["hipotesis"].items()}).T)
    rep.h("H1 y H6: previsión de la varianza")
    for key in ("H1", "H6"):
        for (name, h), c in H.get(key, {}).items():
            tab(f"{key} · {name}, h = {h} ({d(c['desde'])} → {d(c['hasta'])}; {c['n']} días): pérdida media y MCS al 90%",
                c["tabla"], digits=4)
            tab(f"{key} · {name}, h = {h}: Diebold–Mariano de HAR-RI frente a cada benchmark", r34._dm_view(c["dm"]), digits=4)
    if "H2" in H:
        rep.h("H2: el indicador frente al 1x y al 2x fijos")
        tab("H2: Δg anual del indicador (h = 21), IC del 90%, p unilateral y p de Romano–Wolf",
            H["H2"]["tabla"].set_index(["estrategia", "frente a"]), pct={"Δg", "IC90 inferior", "IC90 superior"}, digits=4)
    rep.h("H3 y H4: frente a la LRS")
    tab("H3: SPA de las variantes h ∈ {21, 63} frente a la LRS 3x",
        yes_no(pd.DataFrame(H.get("H3", {})).T.drop(columns=["desde", "hasta"], errors="ignore")), pct={"Δg mejor"},
        digits=4)
    tab("H4: a igual exposición media (a = indicador, b = LRS)", yes_no(pd.DataFrame(H.get("H4", {})).T),
        pct={"maxDD a", "maxDD b", "P(pérdida ≥ 50%) a", "P(pérdida ≥ 50%) b"}, digits=4)
    rep.h("H5: de dónde sale la variación de L*")
    tab("H5: descomposición de la varianza del L* continuo (h = 21)",
        yes_no(pd.DataFrame(H.get("H5", {})).T.drop(columns=["desde", "hasta"], errors="ignore")), digits=4)
    if "H7" in H:
        rep.h("H7: el bono largo")
        tab("H7: Δg del indicador del bono largo frente a TLT y UBT", H["H7"]["tabla"].set_index(["estrategia", "frente a"]),
            pct={"Δg", "IC90 inferior", "IC90 superior"}, digits=4)
        tab("H7: caída máxima del indicador (a) y de TMF (b)", pd.DataFrame([H["H7"]["caídas"]]),
            pct={"maxDD a", "maxDD b"}, digits=4)
    if "H8" in H:
        rep.h("H8: la cartera de acciones y bonos")
        tab("H8: SPA frente a la LRS y a la HFEA 55/45", pd.DataFrame(H["H8"]["spa"]).T.drop(columns=["desde", "hasta"],
                                                                                          errors="ignore"),
            pct={"Δg mejor"}, digits=4)
        tab("H8: caída máxima de la cartera (a, h = 21) y de la HFEA 55/45 (b)", pd.DataFrame([H["H8"]["caídas"]]),
            pct={"maxDD a", "maxDD b"}, digits=4)
    rep.h("Carteras en la prueba reservada (descriptivo)")
    for name, g in H["descripción"].items():
        tab(f"{name}: crecimiento, caída máxima y exposición media en la ventana", g,
            pct={"crecimiento log", "CAGR", "vol", "maxDD"})
    if "H8" in H:
        tab("Cartera de acciones y bonos, HFEA y LRS en la ventana",
            r35.growth_stats(H["H8"]["retornos"], None, R["markets"][cfg.sb_equity].rf).drop(columns="exposición media"),
            pct={"crecimiento log", "CAGR", "vol", "maxDD"})
    rep.save()
    res = {k: v for k, v in H.items() if k not in ("descripción",)}
    if "H8" in res:
        res["H8"] = {k: v for k, v in res["H8"].items() if k != "retornos"}
    (Path(out_dir) / "resultados.json").write_text(
        json.dumps(_jsonable(res), ensure_ascii=False, indent=2, default=str) + "\n")
    return Path(out_dir) / "README.md"


# ───────────────────────────── entrada ─────────────────────────────

def run(config_path: Path = CONFIG, marker_path: Path = MARKER, out_dir: Path | None = None,
        loader=None) -> dict:
    """Evalúa H1–H8 en la prueba reservada UNA vez (ver el docstring del módulo). loader(end) →
    {nombre: r35.Market}; por defecto r35.load_all."""
    doc = load_config(config_path)
    verify_hashes(doc)
    verify_constants(doc)
    cfg = config_from(doc)
    claim(marker_path, doc)
    stage = "carga de los datos hasta 2026"
    try:
        markets = (loader or r35.load_all)(end=END)
        stage = "cálculo del indicador en tiempo real"
        R = r35.compute(END, cfg, markets)
        stage = "comprobación de lo estimado y de las salidas del desarrollo"
        verify_estimates(doc, R)
        verify_fingerprints(doc, R)
        stage = "evaluación de H1–H8"
        H = evaluate(R, cfg, doc)
        stage = "informe"
        path = write_report(H, R, cfg, doc, Path(out_dir) if out_dir else r35.LETF / "reports" / OUT)
        finish(marker_path, "completado", informe=str(path), resumen=_jsonable(H["resumen"]),
               prerregistro_cambiado_tras_congelar=prereg_changed(doc))
    except Exception as e:
        # si aborta antes de «evaluación de H1–H8», no se ha calculado ningún estadístico de la prueba
        finish(marker_path, "abortado", fase=stage, error=repr(e),
               estadisticos_de_la_prueba_calculados=stage in ("evaluación de H1–H8", "informe"))
        raise
    return H


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
