"""R37 — Las figuras del artículo 7 hasta el 25-9-2026 (sin segunda evaluación).

R37 NO EVALÚA NADA. La prueba reservada (1-1-2019 → 25-9-2026) se evaluó una vez con R36 y la
marca letf/preregistro/articulo7_holdout_ejecutado.json lo deja escrito. R37 vuelve a calcular el
indicador congelado con el mismo código y los mismos datos que R36 (configuración congelada →
r35.load_all(end=END) → r35.compute(END, cfg)), comprueba que cada cifra que también está en R36
coincide con la evaluada (tolerancia relativa 1e-9) y que las del desarrollo coinciden con R33–R35, y
solo entonces dibuja las figuras hasta el 25-9-2026,
con la prueba reservada sombreada.

Qué hace run(), en este orden:
1. Lee la configuración congelada y comprueba el SHA-256 de cada archivo de código que lista, los
   parámetros y la marca (estado «completado» y la misma configuración). Guarda el SHA-256 de los
   tres archivos del prerregistro para exigir al final que no hayan cambiado.
2. Con la red bloqueada (si falta un dato de data/cache, para con error: no descarga nada), carga
   los datos hasta el 25-9-2026 y calcula el indicador con r35.compute; el resultado se guarda en
   una caché FUERA del repositorio (R37_CACHE) para no recalcular en cada paso.
3. Guardas, todas antes de dibujar (si una falla, aborta): ninguna serie pasa del 25-9-2026; lo
   estimado y las salidas del desarrollo son las congeladas (r36.verify_estimates y
   r36.verify_fingerprints); las cifras de la prueba son las de las tablas de R36 (tolerancia
   relativa 1e-9); las del desarrollo, las de R34 y R35; los datos de R33, los de R33.
4. Escribe reports/r37_indicador_figuras (README.md, tablas y figuras).

Nunca llama a r36.run, r36.claim, r36.finish, r35.freeze ni a los run() de R33–R35 (reescribirían
la marca, la configuración congelada o sus informes): mientras corre, esas funciones lanzan
AssertionError. Lo que R37 calcula sobre 2019–2026 que no estaba en R36 (cobertura de la banda,
días de ruina, reparto de L, cambios al año, correlación del MOVE, valor de 1 € de 1971–2026) es
DESCRIPTIVO: no es una hipótesis ni cambia ninguna decisión.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import pickle
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from letf_lab.data import indicador as ind
from letf_lab.data import sources
from letf_lab.report import num_es, pct_es
from letf_lab.research import r33_indicador_datos as r33
from letf_lab.research import r34_harri as r34
from letf_lab.research import r35_indicador as r35
from letf_lab.research import r36_indicador_holdout as r36

log = logging.getLogger(__name__)

END, TEST_START, DEV_END = r35.END, r35.TEST_START, r35.DEV_END
OUT = "r37_indicador_figuras"
LETF = r35.LETF
REPORTS = LETF / "reports"
PREREG_DIR = LETF / "preregistro"
PREREG_FILES = ("articulo7_holdout_ejecutado.json", "articulo7_config_congelada.json", "articulo7_indicador.md")
REL_TOL = 1e-9
H_MAIN = 21

# Funciones que reescribirían la marca, la configuración congelada o los informes de R33–R36.
FORBIDDEN = ((r36, "run"), (r36, "claim"), (r36, "finish"), (r35, "freeze"), (r35, "run"), (r34, "run"),
             (r33, "run"))

# Archivos que R37 no puede cambiar (además de los de la configuración congelada): git status vacío.
PROTECTED = ("letf/letf_lab/research/r33_indicador_datos.py", "letf/letf_lab/research/r34_harri.py",
             "letf/letf_lab/research/r35_indicador.py", "letf/letf_lab/research/r36_indicador_holdout.py",
             "letf/letf_lab/data/indicador.py", "letf/letf_lab/kelly.py", "letf/letf_lab/metrics.py",
             "letf/reports/r33_indicador_datos", "letf/reports/r34_harri", "letf/reports/r35_indicador",
             "letf/reports/r36_indicador_holdout", "letf/preregistro")

CACHE = Path(os.environ.get("R37_CACHE") or Path(tempfile.gettempdir()) / "r37_cache.pkl")


class GuardError(RuntimeError):
    """Una guarda de R37 falló: no se dibuja nada."""


# ───────────────────────────── protecciones del proceso ─────────────────────────────

def _no_network(*a, **k):
    raise RuntimeError("R37: red bloqueada; falta un dato en data/cache y no se descarga nada")


@contextlib.contextmanager
def network_blocked():
    """urllib.request.urlopen, yfinance.download y yfinance.Ticker fallan mientras dura el bloque."""
    import urllib.request

    import yfinance

    saved = [(urllib.request, "urlopen"), (yfinance, "download"), (yfinance, "Ticker")]
    old = [getattr(m, a) for m, a in saved]
    try:
        for m, a in saved:
            setattr(m, a, _no_network)
        yield
    finally:
        for (m, a), f in zip(saved, old):
            setattr(m, a, f)


@contextlib.contextmanager
def writers_forbidden():
    """r36.run/claim/finish, r35.freeze y los run() de R33–R35 lanzan AssertionError en el bloque."""
    old = [getattr(m, a) for m, a in FORBIDDEN]

    def _make(name):
        def _forbidden(*a, **k):
            raise AssertionError(f"R37 no puede llamar a {name}: reescribiría lo congelado o un informe")
        return _forbidden

    try:
        for m, a in FORBIDDEN:
            setattr(m, a, _make(f"{m.__name__.rsplit('.', 1)[-1]}.{a}"))
        yield
    finally:
        for (m, a), f in zip(FORBIDDEN, old):
            setattr(m, a, f)


def _check_cache_path(path: Path) -> Path:
    path = Path(path)
    root = LETF.parent.resolve()
    p = path.resolve()
    if p == root or root in p.parents:
        raise GuardError(f"la caché de R37 no puede estar dentro del repositorio: {p}")
    return path


# ───────────────────────────── lo congelado: lectura y comprobación ─────────────────────────────

def prereg_hashes(prereg_dir: Path = PREREG_DIR) -> dict:
    """SHA-256 de los tres archivos del prerregistro (None si falta alguno)."""
    return {f: (r35.sha256(Path(prereg_dir) / f) if (Path(prereg_dir) / f).exists() else None) for f in PREREG_FILES}


def check_marker(doc: dict, marker: Path = r36.MARKER) -> dict:
    """La marca existe, está «completado» y se hizo con esta configuración congelada. Solo lectura."""
    marker = Path(marker)
    if not marker.exists():
        raise GuardError(f"no existe la marca {marker}: R37 solo dibuja una prueba ya evaluada por R36")
    rec = json.loads(marker.read_text())
    if rec.get("estado") != "completado":
        raise GuardError(f"la marca no está «completado»: {rec.get('estado')}")
    if rec.get("configuracion_sha256") != r36.hashlib_json(doc):
        raise GuardError("la marca se hizo con otra configuración congelada")
    return rec


def data_fingerprint(cache_dir: Path | None = None) -> str:
    """SHA-256 del contenido de data/cache (nombres y bytes): la caché de R37 vale si no cambió."""
    d = Path(cache_dir or sources.CACHE)
    h = hashlib.sha256()
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(d)).encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


# ───────────────────────────── recálculo con el código congelado ─────────────────────────────

def recompute(doc: dict, loader=None, cache: Path | None = CACHE) -> tuple[dict, bool]:
    """El indicador congelado hasta END: r35.compute(END, cfg, loader(end=END)), como r36.run.
    loader: por defecto r35.load_all (las pruebas inyectan un mundo sintético). cache: pickle fuera
    del repositorio con clave (configuración congelada, contenido de data/cache); None, sin caché.
    Devuelve (R, si salió de la caché)."""
    cfg = r36.config_from(doc)
    key = None
    if cache is not None:
        cache = _check_cache_path(cache)
        key = {"configuracion": r36.hashlib_json(doc), "datos": data_fingerprint(), "END": str(END.date()),
               "cargador": "r35.load_all" if loader is None else f"{loader.__module__}.{loader.__qualname__}"}
        if cache.exists():
            try:
                with open(cache, "rb") as fh:
                    obj = pickle.load(fh)
                if obj.get("clave") == key:
                    log.info("R37: indicador de la caché %s", cache)
                    return obj["R"], True
            except Exception as e:                       # caché rota o de otra versión: se recalcula
                log.warning("R37: caché inutilizable (%r); se recalcula", e)
    t = time.time()
    markets = (loader or r35.load_all)(end=END)
    log.info("R37: datos cargados en %.1f s; calculando el indicador hasta el %s", time.time() - t, r34.dmy(END))
    t = time.time()
    R = r35.compute(END, cfg, markets)
    log.info("R37: indicador calculado en %.1f s", time.time() - t)
    if cache is not None:
        tmp = cache.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"clave": key, "R": R}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache)
    return R, False


def dev_cut(R: dict, name: str, h: int) -> pd.Timestamp:
    """Última previsión del desarrollo: su ventana objetivo (las h sesiones siguientes) acaba en el
    31-12-2018 o antes. Con datos hasta 2026 las previsiones de diciembre de 2018 ya tienen
    realizada, pero su resultado es de la prueba y no entran en el desarrollo."""
    u = R["markets"][name].mi.u.dropna()
    idx = u.index[u.index <= DEV_END]
    return idx[-h - 1]


# ───────────────────────────── datos de R33 hasta END (F6 y F9) ─────────────────────────────

def range_ratio(end=END) -> pd.DataFrame:
    """La lógica de r33.range_break con datos hasta end: media anual de Parkinson ÷ media de r² en
    ^GSPC (desde 1962) y ^NDX (desde 1986), con el rango de Yahoo tal cual."""
    out, last = {}, {}
    for t, a in (("^GSPC", "1962-01-01"), ("^NDX", "1986-01-01")):
        x = ind.ohlc(t, end=end).copy()
        last[t] = x.index.max()
        ok = ~x.flag_missing & ~x.flag_no_range & ~x.flag_inconsistent & (x.index >= a)
        x["range_ok"] = ok
        v = ind.range_variances(x)
        ok &= v.r2.notna()
        g = v[ok].groupby(v[ok].index.year)
        out[t] = g.pk.mean() / g.r2.mean()
    t = pd.DataFrame(out)
    t.index.name = "año"
    t.attrs["último día"] = last
    return t


def move_monthly(end=END) -> pd.DataFrame:
    """La lógica de r33.move_vs_realized con datos hasta end: MOVE (media del mes) y volatilidad
    realizada del mismo mes de los cambios diarios del Tesoro a 10 años (pb × √252)."""
    mv = ind.move(end=end)["move"]
    y10 = ind.cmt_curve(end=end)["y10"]
    dy = (1e4 * y10.diff()).dropna()
    dy = dy[dy.index >= mv.index[0]]
    m = pd.DataFrame({"MOVE": mv.groupby(mv.index.to_period("M")).mean(),
                      "realizada 10 años": dy.groupby(dy.index.to_period("M")).std() * np.sqrt(252)}).dropna()
    m.attrs["último día"] = {"MOVE": mv.dropna().index.max(), "DGS10": y10.dropna().index.max()}
    return m


def load_data_series(end=END) -> dict:
    """Los datos de R33 que se dibujan hasta end (F6 y F9) y sus referencias del desarrollo, que
    r33 calcula con su recorte por defecto (31-12-2018)."""
    return {"rango": range_ratio(end), "move": move_monthly(end),
            "ref_rango": r33.range_break(), "ref_move": r33.move_vs_realized()[1]}


# ───────────────────────────── referencias congeladas (tablas de R34–R36) ─────────────────────────────

_TITLE = re.compile(r"^\*\*Tabla (\d+)\. (.*)\*\*\s*$")
_WINDOW = re.compile(r"(\d{1,2}-\d{1,2}-\d{4}) → (\d{1,2}-\d{1,2}-\d{4})(?:; ([\d.]+) días)?")


def parse_window(title: str) -> dict | None:
    """(desde, hasta, días) del título de una tabla: «(2-1-2019 → 26-8-2026; 1923 días)»."""
    m = _WINDOW.search(title)
    if not m:
        return None
    d = lambda s: pd.Timestamp(pd.to_datetime(s, format="%d-%m-%Y"))  # noqa: E731
    return {"desde": d(m.group(1)), "hasta": d(m.group(2)),
            "días": int(m.group(3).replace(".", "")) if m.group(3) else None}


class Refs:
    """Las tablas de un informe congelado: CSV a precisión completa (tabla_NN.csv) y su título
    del README. Solo lectura."""

    def __init__(self, root: Path = REPORTS):
        self.root = Path(root)
        self._titles: dict = {}

    def table(self, report: str, n: int, index) -> pd.DataFrame:
        df = pd.read_csv(self.root / report / f"tabla_{n:02d}.csv")
        df = df.rename(columns={"Unnamed: 0": "_fila"})
        return df.set_index(index if index else "_fila")

    def title(self, report: str, n: int) -> str:
        if report not in self._titles:
            t = {}
            for line in (self.root / report / "README.md").read_text().splitlines():
                m = _TITLE.match(line)
                if m:
                    t[int(m.group(1))] = m.group(2)
            self._titles[report] = t
        return self._titles[report][n]


class DictRefs:
    """Las mismas referencias en memoria (las pruebas sintéticas las construyen con refs_from)."""

    def __init__(self, tables: dict, titles: dict):
        self.tables, self.titles = tables, titles

    def table(self, report: str, n: int, index) -> pd.DataFrame:
        return self.tables[(report, n)].copy()

    def title(self, report: str, n: int) -> str:
        return self.titles[(report, n)]


def _close(a, b, tol: float = REL_TOL) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    a, b = float(a), float(b)
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b))
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _rel(a, b) -> float:
    if isinstance(a, str) or isinstance(b, str):
        return 0.0 if str(a) == str(b) else np.inf
    a, b = float(a), float(b)
    if np.isnan(a) or np.isnan(b):
        return 0.0 if np.isnan(a) and np.isnan(b) else np.inf
    return abs(a - b) / max(1.0, abs(a), abs(b))


class Check:
    """Una tabla calculada ahora (got) que debe ser la tabla n del informe report: mismas filas,
    mismas columnas (las de got) y, si el título da la ventana, mismas fechas y días."""

    def __init__(self, report: str, n: int, got: pd.DataFrame, index=None, window: dict | None = None,
                 what: str = "", subset: bool = False):
        self.report, self.n, self.got, self.index, self.window, self.what = report, n, got, index, window, what
        self.subset = subset          # True: solo las filas de got (una figura dibuja parte de la tabla)

    @property
    def name(self) -> str:
        return f"{self.report} tabla {self.n}" + (f" ({self.what})" if self.what else "")

    def compare(self, refs) -> dict:
        bad, worst = [], 0.0
        try:
            ref = refs.table(self.report, self.n, self.index)
        except (FileNotFoundError, KeyError) as e:
            return {"guarda": self.name, "ok": False, "peor": np.inf, "resultado": f"falta la referencia: {e!r}"}
        missing_c = [c for c in self.got.columns if c not in ref.columns]
        rows = list(self.got.index) if self.subset else list(ref.index)
        missing_r = [i for i in rows if i not in self.got.index or i not in ref.index]
        if missing_c or missing_r or not rows:
            bad.append(f"faltan columnas {missing_c} o filas {missing_r}")
        else:
            for i in rows:
                for c in self.got.columns:
                    r = _rel(ref.at[i, c], self.got.at[i, c])
                    worst = max(worst, r)
                    if not _close(ref.at[i, c], self.got.at[i, c]):
                        bad.append(f"{i} · {c}: congelado {ref.at[i, c]}, ahora {self.got.at[i, c]}")
        if self.window is not None:
            w = parse_window(refs.title(self.report, self.n))
            if w is None:
                bad.append("el título no da la ventana")
            else:
                for k in ("desde", "hasta", "días"):
                    if w[k] is not None and w[k] != self.window.get(k):
                        bad.append(f"ventana {k}: congelado {w[k]}, ahora {self.window.get(k)}")
        res = (f"{len(rows)} {'fila' if len(rows) == 1 else 'filas'} × {len(self.got.columns)} "
               f"{'columna' if len(self.got.columns) == 1 else 'columnas'}"
               + (f"; ventana {r34.dmy(self.window['desde'])} → {r34.dmy(self.window['hasta'])}" if self.window else "")
               if not bad else "; ".join(bad[:6]))
        return {"guarda": self.name, "ok": not bad, "peor": worst, "resultado": res}


def refs_from(checks: list[Check], perturb: dict | None = None) -> DictRefs:
    """Referencias que coinciden con checks (para las pruebas sintéticas); perturb = {(report, n):
    función que cambia la tabla} para comprobar que la guarda falla."""
    tables, titles = {}, {}
    for c in checks:
        t = c.got.copy()
        if perturb and (c.report, c.n) in perturb:
            t = perturb[(c.report, c.n)](t)
        tables[(c.report, c.n)] = t
        w = c.window or {}
        titles[(c.report, c.n)] = (f"x ({r34.dmy(w['desde'])} → {r34.dmy(w['hasta'])}"
                                   + (f"; {w['días']} días)" if w.get("días") is not None else ")")) if w else "x"
    return DictRefs(tables, titles)


# ───────────────────────────── las series: prueba y desarrollo ─────────────────────────────

def _win(df, a, b):
    return df[(df.index >= pd.Timestamp(a)) & (df.index <= pd.Timestamp(b))]


def test_frames(R: dict, cfg: r35.Config) -> dict:
    """Las series de la prueba tal como las evaluó R36 (las que se dibujan en N1–N5): rentabilidades
    de cada mercado en [TEST_START, fin], la cartera de acciones y bonos con la HFEA comprada al
    cierre del 31-12-2018 (como r35.hypotheses) y las previsiones de H1 y H6."""
    end = r36.window_end(R)
    out = {"fin": end, "ret": {}, "expo": {}, "prev": {}}
    for name, F in R["ret"].items():
        W = _win(F, TEST_START, end)
        out["ret"][name] = W
        out["expo"][name] = R["expo"][name].reindex(W.index)
    if R["sb_ret"]:
        me, mb = R["markets"][cfg.sb_equity], R["markets"][cfg.sb_bond]
        SB = pd.DataFrame(R["sb_ret"])
        SB["LRS"] = R["ret"][cfg.sb_equity]["LRS"]
        SB = _win(SB, TEST_START, end).dropna()
        SB["HFEA 55/45"] = r35.run_hfea(me, mb, SB.index[0], end, cfg).reindex(SB.index)
        out["SB"] = SB.dropna()
    for name in ("S&P 500", "Nasdaq-100", "Bono largo"):
        for h in cfg.horizons:
            if name in R["bench"]:
                out["prev"][(name, h)] = r35.forecast_window(R, name, h, TEST_START, end)[
                    list(r34.MAIN_MODELS) + ["realizada"]].dropna()
    return out


def dev_frames(R: dict, cfg: r35.Config) -> dict:
    """Las del desarrollo con las ventanas de R34 y R35, recortadas aunque haya datos hasta 2026:
    carteras en r35.dev_window (≤ 31-12-2018), cartera de acciones y bonos con la HFEA de R35
    (comprada al inicio de la ventana, hasta el 31-12-2018) y previsiones cuya ventana objetivo
    acaba el 31-12-2018 o antes (dev_cut)."""
    out = {"ret": {}, "expo": {}, "ventana": {}, "prev": {}, "corte": {}}
    for name, F in R["ret"].items():
        J = F[F.index <= DEV_END].dropna()
        a, b = J.index[0], J.index[-1]
        out["ventana"][name] = (a, b)
        out["ret"][name] = F.loc[a:b]
        out["expo"][name] = R["expo"][name].loc[a:b]
    eq = [n for n in r35.EQUITY_MARKETS if n in R["ret"]]
    out["inicio acciones"] = max(out["ventana"][n][0] for n in eq) if eq else None
    if R["sb_ret"]:
        me, mb = R["markets"][cfg.sb_equity], R["markets"][cfg.sb_bond]
        SB0 = pd.DataFrame(R["sb_ret"])
        sb_start = SB0[SB0.index <= DEV_END].dropna().index[0]
        bond_start = max(out["ventana"][cfg.sb_bond][0], sb_start)
        out["inicio bonos"] = bond_start
        SB = SB0.copy()
        SB["LRS"] = R["ret"][cfg.sb_equity]["LRS"]
        SB = _win(SB, bond_start, DEV_END).dropna()
        SB["HFEA 55/45"] = r35.run_hfea(me, mb, SB.index[0], DEV_END, cfg).reindex(SB.index)
        out["SB"] = SB.dropna()
    for name in ("S&P 500", "Nasdaq-100", "Bono largo"):
        if name not in R["bench"]:
            continue
        for h in cfg.horizons:
            cut = dev_cut(R, name, h)
            out["corte"][(name, h)] = cut
            F = R["bench"][name][h][list(r34.MAIN_MODELS) + ["realizada"]].dropna()
            out["prev"][(name, h)] = F[F.index <= cut]
    return out


def loss_table(F: pd.DataFrame, premium: float) -> pd.DataFrame:
    """Pérdida media de cada modelo (como r34.compare, sin MCS ni bootstrap)."""
    ql, gl = r34.losses(F, premium)
    y = F["realizada"]
    M = list(r34.MAIN_MODELS)
    return pd.DataFrame({"QLIKE": ql.mean(), "pérdida de crecimiento (pb/año)": gl.mean(),
                         "realizada/prevista (media)": (y.values[:, None] / F[M]).mean()})


def calibration(R: dict, name: str, h: int, idx: pd.DatetimeIndex) -> dict:
    """r34.dispersion_check de HAR-RI en los días idx (la de R34, tabla 23, en el desarrollo)."""
    return r34.dispersion_check(R["var"][name][h], idx)


def _window(F) -> dict:
    return {"desde": F.index[0], "hasta": F.index[-1], "días": len(F)}


# ───────────────────────────── las tablas que deben coincidir ─────────────────────────────

R36_GROWTH = {"S&P 500": 22, "Nasdaq-100": 23, "Bono largo": 24, "Bono medio": 25}
R36_LOSS = {("S&P 500", 21): 2, ("S&P 500", 63): 4, ("Nasdaq-100", 21): 6, ("Nasdaq-100", 63): 8,
            ("Bono largo", 21): 10, ("Bono largo", 63): 12}
R34_LOSS = {("S&P 500", 21): 3, ("S&P 500", 63): 7, ("Nasdaq-100", 21): 11, ("Nasdaq-100", 63): 13}
R35_GROWTH = {"S&P 500": 6, "Nasdaq-100": 7, "Bono largo": 8, "Bono medio": 9}
R36_DIR, R35_DIR, R34_DIR = r36.OUT, "r35_indicador", "r34_harri"


def test_checks(R: dict, cfg: r35.Config, doc: dict, T: dict) -> list[Check]:
    """Las cifras de la prueba recalculadas sobre las mismas series que se dibujan (T = test_frames)
    frente a las tablas de R36 (sin p-valores, bootstrap, MCS ni SPA)."""
    prem = doc["estimados_en_desarrollo"]["prima_desarrollo"]
    hm = cfg.h_main
    out = []
    for name, n in R36_GROWTH.items():
        if name in T["ret"]:
            W = T["ret"][name]
            out.append(Check(R36_DIR, n, r35.growth_stats(W, T["expo"][name], R["markets"][name].rf), what=name))
    if "SB" in T:
        g = r35.growth_stats(T["SB"], None, R["markets"][cfg.sb_equity].rf).drop(columns="exposición media")
        out.append(Check(R36_DIR, 26, g, what="cartera de acciones y bonos, LRS y HFEA"))
        mdd = r35.max_drawdown_cols(T["SB"][[f"acciones y bonos h={hm}", "HFEA 55/45"]].to_numpy(float))
        out.append(Check(R36_DIR, 21, pd.DataFrame([{"maxDD a": mdd[0], "maxDD b": mdd[1]}]), what="H8, caídas"))
    for key, n in R36_LOSS.items():
        if key in T["prev"]:
            F = T["prev"][key]
            out.append(Check(R36_DIR, n, loss_table(F, prem[key[0]]), window=_window(F),
                             what=f"{key[0]}, h = {key[1]}"))
    h4 = {}
    for name in r35.EQUITY_MARKETS:
        if name not in T["ret"]:
            continue
        Rm, E = T["ret"][name], T["expo"][name]
        j = Rm[[f"indicador h={hm}", "LRS"]].join(E[[f"indicador h={hm}", "LRS"]], rsuffix=" (exp)").dropna()
        ra, rb, k, _ = r35.equal_exposure(j[f"indicador h={hm}"], j[f"indicador h={hm} (exp)"], j["LRS"],
                                          j["LRS (exp)"], R["markets"][name].rf)
        mdd = r35.max_drawdown_cols(pd.DataFrame({"a": ra, "b": rb}).to_numpy(float))
        h4[name] = {"maxDD a": mdd[0], "maxDD b": mdd[1], "exposición media indicador": float(j[f"indicador h={hm} (exp)"].mean()),
                    "exposición media LRS": float(j["LRS (exp)"].mean()), "factor": k}
    if h4:
        out.append(Check(R36_DIR, 16, pd.DataFrame(h4).T, what="H4, sin el bootstrap"))
    h5 = {}
    for name in r35.MARKETS:
        if (name, hm) not in R["ind"]:
            continue
        J = _win(R["ind"][(name, hm)], TEST_START, T["fin"])[["L_star", "num", "D", "trend", "V", "S"]].dropna()
        share = r35.shapley_variance({"prior": J.num - J.trend, "tendencia": J.trend}, J.D)
        h5[name] = {"días": len(J), "corr(L*, E[IV])": float(J.L_star.corr(J.V)),
                    "corr(L*, S)": float(J.L_star.corr(J.S)) if J.S.std() > 0 else np.nan,
                    "parte de la varianza": share["varianza"], "parte del prior": share["prior"],
                    "parte de la tendencia": share["tendencia"], "L* medio": float(J.L_star.mean()),
                    "L* mediano": float(J.L_star.median())}
    if h5:
        out.append(Check(R36_DIR, 17, pd.DataFrame(h5).T, what="H5, sin el bootstrap"))
    if "Bono largo" in T["ret"]:
        F7 = T["ret"]["Bono largo"][[f"indicador h={hm}", "1x", "2x", "3x"]].dropna()
        mdd = r35.max_drawdown_cols(F7[[f"indicador h={hm}", "3x"]].to_numpy(float))
        out.append(Check(R36_DIR, 19, pd.DataFrame([{"maxDD a": mdd[0], "maxDD b": mdd[1]}]), what="H7, caídas"))
    return out


def dev_checks(R: dict, cfg: r35.Config, doc: dict, D: dict) -> list[Check]:
    """Las cifras del desarrollo de las figuras extendidas (D = dev_frames) frente a R34 y R35."""
    prem = doc["estimados_en_desarrollo"]["prima_desarrollo"]
    out = []
    for key, n in R34_LOSS.items():
        if key in D["prev"]:
            F = D["prev"][key]
            out.append(Check(R34_DIR, n, loss_table(F, prem[key[0]]), window=_window(F),
                             what=f"{key[0]}, h = {key[1]}, ventana común"))
    rows = {}
    for name in r35.EQUITY_MARKETS:
        for h in cfg.horizons:
            if (name, h) not in D["prev"]:
                continue
            c = calibration(R, name, h, D["prev"][(name, h)].index)
            rows[(name, h)] = {"días": c["n"], "desde": r34.dmy(c["desde"]), "cobertura 90%": c["cobertura 90%"],
                               "bajo q05": c["bajo q05"], "sobre q95": c["sobre q95"], "z² medio": c["z² medio"],
                               "z²: peso de 5 meses": c["z²: peso de 5 meses"],
                               "de/media (mediana)": c["de/media (mediana)"]}
    if rows:
        t = pd.DataFrame(rows).T
        t.index = t.index.set_names(["mercado", "h"])
        out.append(Check(R34_DIR, 23, t, index=["mercado", "h"], what="calibración del desarrollo"))
    for name, n in R35_GROWTH.items():
        if name not in D["ret"]:
            continue
        g = r35.growth_stats(D["ret"][name], D["expo"][name], R["markets"][name].rf)
        tk = R["markets"][name].tickers
        g.index = [{"0x": "letras", "1x": f"1x ({tk[1]})", "2x": f"2x ({tk[2]})", "3x": f"3x ({tk[3]})",
                    "LRS": f"LRS 3x ({tk[3]})"}.get(k, k) for k in g.index]
        a, b = D["ventana"][name]
        out.append(Check(R35_DIR, n, g, window={"desde": a, "hasta": b, "días": None}, what=name))
    rows = {}
    for name in ("S&P 500", "Nasdaq-100", "Bono largo"):
        start = D["inicio bonos"] if name == "Bono largo" else D["inicio acciones"]
        for h in cfg.horizons:
            if (name, h) not in D["prev"] or start is None:
                continue
            F = D["prev"][(name, h)]
            t = loss_table(F[F.index >= start], prem[name])
            for bench in (r34.M_HARLOG, r34.M_GJR):
                key = ("H6" if name == "Bono largo" else "H1", name, h, bench)
                rows[key] = {"QLIKE HAR-RI / bench": t.loc[r34.M_HARRI, "QLIKE"] / t.loc[bench, "QLIKE"]}
    if rows:
        t = pd.DataFrame(rows).T
        t.index = t.index.set_names(["hipótesis", "mercado", "h", "frente a"])
        out.append(Check(R35_DIR, 10, t, index=["hipótesis", "mercado", "h", "frente a"],
                         what="cociente de QLIKE de H1 y H6 en el desarrollo"))
    if "SB" in D:
        SB = D["SB"]
        g = r35.growth_stats(SB, None, R["markets"][cfg.sb_equity].rf).drop(columns="exposición media")
        out.append(Check(R35_DIR, 17, g, window={"desde": SB.index[0], "hasta": SB.index[-1], "días": None},
                         what="cartera de acciones y bonos con la HFEA del desarrollo"))
    return out


def drawn_check(n: int, W: pd.DataFrame, E: pd.DataFrame | None, rf: pd.Series, what: str, refs=None) -> dict:
    """Guarda de una figura nueva (N1, N3, N4): r35.growth_stats sobre EXACTAMENTE el DataFrame que se
    dibuja frente a la tabla n de R36 (solo las filas dibujadas) y valor final de 1 € = (1 + r).prod().
    Lanza GuardError si no coincide; devuelve la fila de la guarda con los valores finales."""
    g = r35.growth_stats(W, E, rf)
    if E is None:
        g = g.drop(columns="exposición media")
    row = Check(R36_DIR, n, g, what=what, subset=True).compare(Refs(REPORTS) if refs is None else refs)
    final = {c: float((1 + W[c].dropna()).prod()) for c in W.columns}
    row["valor final"] = final
    row["resultado"] += "; valor final de 1 €: " + ", ".join(f"{c} {num_es(v, 3)}" for c, v in final.items())
    check_all([row])
    return row


# ───────────────────────────── guardas de los datos de R33 y del final de las series ─────────────────────────────

def _ok(name: str, ok: bool, res: str, worst: float = np.nan) -> dict:
    return {"guarda": name, "ok": bool(ok), "peor": worst, "resultado": res}


def data_checks(data: dict) -> list[dict]:
    """Los datos de R33 que se dibujan hasta END: en el desarrollo, los mismos que R33 (cociente de
    Parkinson de cada año ≤ 2018 y correlación mensual del MOVE con la realizada del 10 años)."""
    out = []
    rb, ref = data["rango"], data["ref_rango"]
    bad, worst = [], 0.0
    if ref.index.max() > DEV_END.year:
        bad.append(f"la referencia de R33 llega a {ref.index.max()}")
    for y in ref.index:
        for c in ref.columns:
            a = ref.at[y, c]
            b = rb.at[y, c] if (y in rb.index and c in rb.columns) else np.nan
            worst = max(worst, _rel(a, b))
            if not _close(a, b):
                bad.append(f"{y} · {c}: R33 {a}, ahora {b}")
    out.append(_ok("R33: cociente de Parkinson, años ≤ 2018", not bad,
                   f"{len(ref)} años ({ref.index.min()}–{ref.index.max()}) × {ref.shape[1]} series" if not bad
                   else "; ".join(bad[:6]), worst))
    m, st = data["move"], data["ref_move"]
    md = m[m.index <= pd.Period(DEV_END, "M")]
    corr = float(md["MOVE"].corr(md["realizada 10 años"]))
    w = _rel(corr, st["corr"])
    ok = _close(corr, st["corr"]) and len(md) == st["meses"]
    out.append(_ok("R33: correlación MOVE–realizada del desarrollo", ok,
                   f"{len(md)} meses, correlación {num_es(corr, 6)} (R33: {st['meses']} meses, "
                   f"{num_es(st['corr'], 6)})", w))
    return out


def _series_after(obj, path: str, end: pd.Timestamp, bad: list) -> int:
    """Cuenta las series (DataFrame o Series con índice de fechas) de obj y anota las que pasan de end."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        if isinstance(obj.index, pd.DatetimeIndex) and len(obj.index) and obj.index.max() > end:
            bad.append(f"{path}: hasta {obj.index.max().date()}")
        return 1
    if isinstance(obj, dict):
        return sum(_series_after(v, f"{path}[{k}]", end, bad) for k, v in obj.items())
    return 0


def extent_checks(R: dict, data: dict | None, end: pd.Timestamp = END) -> list[dict]:
    """Ninguna serie pasa del 25-9-2026: los mercados, todo lo calculado por r35.compute y los datos
    de R33 que se dibujan; y el fin de la ventana de la prueba es el de R36 (25-9-2026)."""
    out, bad = [], []
    for k, v in R["markets"].items():
        if v.last_date() > end:
            bad.append(f"mercado {k}: hasta {v.last_date().date()}")
    n = sum(_series_after(R[k], k, end, bad) for k in ("ret", "expo", "ind", "var", "bench", "evt", "sb", "sb_ret", "corr"))
    out.append(_ok("ninguna serie del indicador pasa del 25-9-2026", not bad,
                   f"{len(R['markets'])} mercados y {n} series, la última del {r34.dmy(max(v.last_date() for v in R['markets'].values()))}"
                   if not bad else "; ".join(bad[:6])))
    we = r36.window_end(R)
    out.append(_ok("fin de la ventana de la prueba (r36.window_end)", we == end, r34.dmy(we)))
    if data is not None:
        bad = []
        last = {**{f"OHLC {k}": v for k, v in data["rango"].attrs.get("último día", {}).items()},
                **data["move"].attrs.get("último día", {})}
        if len(last) < 4:
            bad.append(f"faltan fechas finales: {sorted(last)}")
        for k, v in last.items():
            if pd.Timestamp(v) > end:
                bad.append(f"{k}: hasta {pd.Timestamp(v).date()}")
        if data["rango"].index.max() > end.year or data["move"].index.max() > pd.Period(end, "M"):
            bad.append("tablas anuales o mensuales que pasan de 2026-09")
        out.append(_ok("ninguna serie de R33 (OHLC, MOVE, DGS10) pasa del 25-9-2026", not bad,
                       "; ".join(f"{k} hasta el {r34.dmy(v)}" for k, v in last.items()) if not bad else "; ".join(bad)))
    return out


def run_guards(R: dict, cfg: r35.Config, doc: dict, refs, data: dict | None, T: dict, D: dict) -> list[dict]:
    """Todas las guardas que comparan cifras (prueba contra R36, desarrollo contra R34 y R35, datos
    contra R33, final de las series). refs: objeto con table/title (Refs) o función checks → refs
    (las pruebas sintéticas). Devuelve una fila por guarda; no lanza: lo decide check_all."""
    checks = test_checks(R, cfg, doc, T) + dev_checks(R, cfg, doc, D)
    if callable(refs) and not hasattr(refs, "table"):
        refs = refs(checks)
    out = [c.compare(refs) for c in checks]
    if data is not None:
        out += data_checks(data)
    out += extent_checks(R, data)
    return out


def check_all(rows: list[dict]) -> None:
    """Aborta (GuardError) si alguna guarda falló: no se dibuja nada."""
    bad = [r for r in rows if not r["ok"]]
    if bad:
        raise GuardError("R37: fallan guardas; no se dibuja nada:\n  "
                         + "\n  ".join(f"{r['guarda']}: {r['resultado']}" for r in bad))


# ───────────────────────────── archivos protegidos (git, solo lectura) ─────────────────────────────

ALLOWED_CHANGES = ("letf/letf_lab/research/r37_indicador_figuras.py", "letf/letf_lab/research/r37_figuras.py",
                   "letf/tests/test_r37_indicador_figuras.py", "letf/reports/r37_indicador_figuras/",
                   "posts/indicador-apalancamiento/")


def git_guard(repo: Path = LETF.parent) -> dict:
    """git status y git diff (solo lectura): nada de PROTECTED cambia y lo único nuevo o cambiado
    del árbol son los archivos de R37 y el post."""
    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout
    try:
        prot = git("status", "--porcelain", "--untracked-files=all", "--", *PROTECTED).strip()
        diff = git("diff", "--name-only", "HEAD", "--", *PROTECTED).strip()
        other = [ln[3:] for ln in git("status", "--porcelain", "--untracked-files=all").splitlines()
                 if not ln[3:].startswith(ALLOWED_CHANGES)]
    except (OSError, subprocess.CalledProcessError) as e:
        return _ok("git: lo congelado sin cambios", False, f"no se pudo leer git: {e!r}")
    bad = ([f"cambiado: {x}" for x in (prot + "\n" + diff).split("\n") if x.strip()]
           + [f"fuera de R37: {x}" for x in other])
    return _ok("git: r33–r36, data/indicador.py, kelly.py, metrics.py, sus informes y el prerregistro sin cambios",
               not bad, "git status y git diff vacíos en lo protegido; solo cambian archivos de R37 y el post"
               if not bad else "; ".join(bad[:8]))


# ───────────────────────────── todo junto ─────────────────────────────

def prepare(config_path: Path = r36.CONFIG, marker_path: Path = r36.MARKER, prereg_dir: Path = PREREG_DIR,
            loader=None, cache: Path | None = CACHE, refs=None, data_loader=load_data_series) -> dict:
    """Comprobaciones de lo congelado, recálculo (red bloqueada y escritores prohibidos) y TODAS las
    guardas. Si una falla, lanza GuardError o r36.FrozenError antes de dibujar nada. Devuelve el
    contexto con el que se dibujan las figuras. loader, refs y data_loader: las pruebas sintéticas
    los inyectan (data_loader=None omite las guardas de los datos de R33)."""
    t0 = time.time()
    before = prereg_hashes(prereg_dir)
    doc = r36.load_config(config_path)
    r36.verify_hashes(doc)
    r36.verify_constants(doc)
    marker = check_marker(doc, marker_path)
    cfg = r36.config_from(doc)
    rows = [_ok("código congelado (r36.verify_hashes)", True, f"{len(doc['sha256'])} archivos con su SHA-256"),
            _ok("parámetros, fechas, rupturas e hipótesis (r36.verify_constants)", True, "los congelados"),
            _ok("marca de la evaluación única", True,
                f"estado «{marker['estado']}», configuración {r36.hashlib_json(doc)[:12]}…; solo lectura")]
    with network_blocked(), writers_forbidden():
        R, cached = recompute(doc, loader, cache)
        data = data_loader(END) if data_loader is not None else None
    r36.verify_estimates(doc, R)
    r36.verify_fingerprints(doc, R)
    rows += [_ok("lo estimado en el desarrollo (r36.verify_estimates)", True,
                 f"{sum(len(v) for v in doc['estimados_en_desarrollo'].values())} valores, tolerancia 1e-9"),
             _ok("huellas del desarrollo (r36.verify_fingerprints)", True,
                 f"{len(doc['huellas_desarrollo'])} huellas hasta el 31-12-2018")]
    T, D = test_frames(R, cfg), dev_frames(R, cfg)
    rows += run_guards(R, cfg, doc, Refs(REPORTS) if refs is None else refs, data, T, D)
    check_all(rows)
    if before != prereg_hashes(prereg_dir):
        raise GuardError("R37: el prerregistro (marca, configuración o texto) cambió durante el cálculo")
    log.info("R37: %d guardas superadas en %.1f s%s", len(rows), time.time() - t0, " (indicador de la caché)" if cached else "")
    return {"doc": doc, "cfg": cfg, "R": R, "T": T, "D": D, "data": data, "guardas": rows, "prereg": before,
            "cache": cached, "config_path": Path(config_path), "prereg_dir": Path(prereg_dir), "refs_arg": refs}


def refs_for(ctx: dict, checks: list[Check]):
    """Las referencias con las que se comparan las guardas de una figura: las tablas reales de los
    informes (Refs) o, en las pruebas sintéticas, las que construye su función a partir de checks."""
    r = ctx.get("refs_arg")
    if r is None:
        return Refs(REPORTS)
    if callable(r) and not hasattr(r, "table"):
        return r(checks)
    return r


def finish_checks(ctx: dict, git: bool = True) -> list[dict]:
    """Al terminar: el código congelado y el prerregistro siguen igual (y git, con el repositorio real)."""
    doc = r36.load_config(ctx["config_path"])
    r36.verify_hashes(doc)
    rows = [_ok("al terminar: código congelado", True, "r36.verify_hashes otra vez")]
    now = prereg_hashes(ctx["prereg_dir"])
    rows.append(_ok("al terminar: marca, configuración congelada y prerregistro", now == ctx["prereg"],
                    "SHA-256 iguales que al empezar: " + ", ".join(f"{k} {(v or '—')[:12]}…" for k, v in now.items())))
    if git:
        rows.append(git_guard())
    check_all(rows)
    return rows


# ───────────────────────────── informe ─────────────────────────────

INTRO = (
    "R37 **no es una segunda evaluación**. La prueba reservada (1-1-2019 → 25-9-2026) se evaluó una sola vez con R36 "
    "y la marca `letf/preregistro/articulo7_holdout_ejecutado.json` lo deja escrito. R37 vuelve a calcular el "
    "indicador congelado con el mismo código y los mismos datos (configuración congelada → `r35.load_all(end=END)` → "
    "`r35.compute(END, cfg)`, como `r36.run`), sin tocar la marca, y comprueba que cada cifra que también está en R36 "
    "coincide con la evaluada (tolerancia relativa 1e-9) y que las del desarrollo coinciden con R33–R35 antes de dibujar las figuras del artículo 7 hasta el "
    "25-9-2026, con la prueba reservada sombreada en gris claro. No se reestima, no se reajusta ni se cambia "
    "ningún parámetro.")

DESCRIPTIVE_NOTE = (
    "Lo que R37 calcula sobre 2019–2026 que no estaba en R36 (cobertura de la banda, días de ruina, reparto de L, "
    "cambios al año, correlación del MOVE, valor de 1 € de 1971–2026) es **descriptivo: no es una hipótesis ni cambia "
    "ninguna decisión**.")


def guard_table(rows: list[dict]) -> pd.DataFrame:
    t = pd.DataFrame(rows)
    t["ok"] = t["ok"].map({True: "sí", False: "no"})
    t["peor"] = t["peor"].map(lambda v: "" if not np.isfinite(v) else ("0" if v == 0 else f"{v:.1e}".replace(".", ",")))
    return t.rename(columns={"ok": "pasa", "peor": "peor diferencia relativa"}).set_index("guarda")


def write_report(ctx: dict, out_dir: Path, sections=(), final=None) -> Path:
    """README.md del informe con las guardas. sections: funciones (rep, tab, ctx) → None que añaden
    las figuras y las tablas descriptivas. final: función sin argumentos que se llama después de
    las secciones (con las figuras ya escritas) y devuelve las guardas del final."""
    from letf_lab.research.r26_vix_data import Tables

    doc = ctx["doc"]
    rep = r36._Report(out_dir, "R37 — Las figuras del artículo 7 hasta el 25-9-2026")
    tab = Tables(rep)
    rep.p(INTRO)
    rep.p(f"Configuración congelada: `{Path(ctx['config_path']).name}` (commit base "
          f"{doc['congelado']['commit_base'][:10]}, congelada el {doc['congelado']['fecha']}); "
          f"{len(doc['sha256'])} archivos de código verificados por SHA-256. Datos: `letf/data/cache`, con la red "
          f"bloqueada (no se descarga nada). Ventana de la prueba: {r34.dmy(TEST_START)} → "
          f"{r34.dmy(ctx['T']['fin'])}.")
    rep.p(DESCRIPTIVE_NOTE)
    for s in sections:
        s(rep, tab, ctx)
    ctx["guardas_finales"] = final() if final is not None else []
    rep.h("Guardas")
    rep.p("Todas se comprueban antes de dibujar; si una falla, R37 aborta sin escribir figuras. Tolerancia relativa "
          "1e-9 frente a las tablas CSV de R34–R36 (precisión completa). No se recalcula ningún p-valor, bootstrap, MCS "
          "ni SPA: las figuras no los necesitan. Las del final se comprueban después de escribir las figuras.")
    tab("Guardas de R37: lo congelado, la prueba de R36, el desarrollo de R34 y R35, los datos de R33 y el final de "
        "las series", guard_table(ctx["guardas"] + ctx["guardas_finales"]))
    rep.save()
    return Path(out_dir) / "README.md"


# ───────────────────────────── entrada ─────────────────────────────

def run(out_dir: Path | None = None, sections=None, git: bool = True, **kw) -> dict:
    """Comprueba, recalcula, pasa las guardas y escribe reports/r37_indicador_figuras. Mientras corre,
    la red está bloqueada y las funciones que reescriben lo congelado lanzan AssertionError. sections:
    por defecto r37_figuras.SECTIONS (todas las figuras); () escribe solo las guardas. kw: los de
    prepare (las pruebas inyectan configuración, marca, cargador y referencias)."""
    if sections is None:
        from letf_lab.research import r37_figuras

        sections = r37_figuras.SECTIONS
    out_dir = Path(out_dir) if out_dir else REPORTS / OUT
    with network_blocked(), writers_forbidden():
        ctx = prepare(**kw)
        ctx["informe"] = write_report(ctx, out_dir, sections, final=lambda: finish_checks(ctx, git=git))
    return ctx


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
