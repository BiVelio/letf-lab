"""Procedencia de los p de titular que R16–R18 exportan a p_titular.csv y R19 lee.

Cada fila lleva el commit (HEAD) con que se generó, si los archivos de los que
depende tenían cambios sin commit y una huella SHA-256 de esos archivos. R19
recalcula la huella con el código actual y rechaza un p_titular.csv escrito con
otro código: la fecha de modificación no sirve (un checkout la reescribe).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pandas as pd

PKG = Path(__file__).resolve().parent        # letf/letf_lab
REPO = PKG.parents[1]                        # raíz del repositorio (letf-lab)
TITULAR = "p_titular.csv"
COLUMNS = ["módulo", "mercado", "contraste", "p"]

# Archivos de los que dependen las cifras de R12–R13 (datos, 3x sintético, señales, Kelly).
BASE = ("data/history.py", "data/sources.py", "research/common.py", "research/r3_synthetic.py",
        "research/r10_predictability.py", "research/r12_long_ml.py", "research/r13_lrs.py",
        "sim.py", "stats.py", "kelly.py", "metrics.py", "universe.py")
DEPS = {m: BASE + (f"research/{m}.py",) for m in ("r16_mechanism", "r17_benchmarks", "r18_worlds")}


def code_hash(files) -> str:
    """Huella SHA-256 (16 cifras) del contenido de `files` (rutas relativas a letf_lab), en orden."""
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode())
        h.update((PKG / f).read_bytes())
    return h.hexdigest()[:16]


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "--no-optional-locks", *args], cwd=REPO, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_head() -> str:
    return _git("rev-parse", "HEAD") or "desconocido"


def dirty(files) -> bool:
    """¿Tiene alguno de `files` cambios sin commit (o está sin seguimiento)?"""
    paths = [str((PKG / f).relative_to(REPO)) for f in files]
    return bool(_git("status", "--porcelain", "--", *paths))


def write_titular(out_dir: Path, rows: list[dict], module: str) -> pd.DataFrame:
    """Escribe out_dir/p_titular.csv con las filas (módulo, mercado, contraste, p) y el sello
    de procedencia del módulo (commit, cambios sin commit y huella del código)."""
    files = DEPS[module]
    df = pd.DataFrame(rows, columns=COLUMNS)
    assert df.p.between(0, 1, inclusive="right").all(), "p fuera de (0, 1]"
    df["commit"] = git_head()
    df["cambios_sin_commit"] = dirty(files)
    df["huella_código"] = code_hash(files)
    df["archivos"] = ";".join(files)
    df.to_csv(Path(out_dir) / TITULAR, index=False)
    return df


def read_titular(out_dir: Path, module: str) -> pd.DataFrame:
    """Lee out_dir/p_titular.csv y comprueba que se generó con el código actual del módulo."""
    path = Path(out_dir) / TITULAR
    assert path.exists(), f"falta {path}: ejecuta {module} antes que R19"
    # la huella y el commit son texto: sin dtype, «0123…» se leería como entero (sin el
    # cero inicial) y «1234e5…» como número en coma flotante
    df = pd.read_csv(path, dtype={"huella_código": str, "commit": str})
    files = tuple(df["archivos"].iloc[0].split(";"))
    assert files == DEPS[module], f"{path}: lista de archivos distinta de la actual; regenera {module}"
    now = code_hash(files)
    old = set(df["huella_código"].astype(str))
    assert old == {now}, (f"{path} se generó con otro código ({', '.join(sorted(old))} ≠ {now}): "
                          f"regenera {module} antes que R19")
    return df
