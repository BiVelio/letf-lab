"""R37 — Una función por figura del artículo 7 hasta el 25-9-2026.

Cada fig_<id>(R, M, out_dir, ...) dibuja una figura con las series que r37_indicador_figuras.prepare
ya comprobó (R = r35.compute hasta el 25-9-2026, M = R["markets"]), la guarda en out_dir/<nombre>.png
y devuelve (nombre, pie). No recalcula nada que decida: dibuja la salida del indicador congelado.
Estilo de R33–R35 (r34._style, la franja r33._shade_test y sus colores); la prueba reservada va
sombreada en gris claro y rotulada, y la frontera del 31-12-2018 se marca con una línea discontinua.

Las secciones sec_<parte>(rep, tab, ctx) enlazan las figuras en el README de R37 y anotan las cifras
DESCRIPTIVAS de 2019–2026 (no estaban en R36: no son una hipótesis ni cambian ninguna decisión) en
ctx["descriptivo"]; sec_descriptivo, la última de SECTIONS, las escribe en una tabla. Las guardas
propias de una figura (los puntos del desarrollo son los de R34, la banda es la de la previsión
evaluada) se añaden a ctx["guardas"] antes de dibujar; si una falla, se lanza GuardError.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from letf_lab.report import FIG_W, comma_fig, layout, legend_band, num_es, pct_es
from letf_lab.research import r33_indicador_datos as r33
from letf_lab.research import r34_harri as r34
from letf_lab.research import r35_indicador as r35

END, TEST_START, DEV_END = r35.END, r35.TEST_START, r35.DEV_END
INK, MUTED, GRID = r34.INK, r34.MUTED, r34.GRID
BLUE, ORANGE, AQUA, GREY, LIGHT = r33.BLUE, r33.ORANGE, r33.AQUA, r33.GREY, r33.LIGHT
MODEL_COLOR = r34.MODEL_COLOR
VAR_MARKETS = ("S&P 500", "Nasdaq-100", "Bono largo")      # los de H1 y H6
EQUITY = ("S&P 500", "Nasdaq-100")
DPI = 200


# ───────────────────────────── utilidades ─────────────────────────────

def _save(fig, out_dir: Path, name: str) -> Path:
    """Como Report.fig: layout() de letf_lab.report, 200 ppp, y cierra la figura."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layout(fig)
    p = out_dir / f"{name}.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def _link(rep, name: str, caption: str) -> None:
    rep.lines += [f"![{caption}]({name}.png)", ""]


def _shade(ax, label: bool = True) -> None:
    """La prueba reservada en gris claro (r33._shade_test) y la frontera del 31-12-2018."""
    r33._shade_test(ax, label=label)
    ax.axvline(DEV_END, color=MUTED, ls="--", lw=0.9, zorder=1)


def _shade_years(ax, last_year: int, label: bool = True) -> None:
    """Lo mismo en un eje de años (medias anuales): la prueba son los años 2019 → last_year."""
    ax.axvspan(DEV_END.year + 0.5, last_year + 0.5, color=LIGHT, zorder=0)
    ax.axvline(DEV_END.year + 0.5, color=MUTED, ls="--", lw=0.9, zorder=1)
    if label:
        ax.text(DEV_END.year + 0.8, 0.98, "prueba\nreservada", transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=10, color=MUTED)


def _test_panel(ax) -> None:
    """Un panel que solo muestra la prueba reservada: fondo gris claro, como la franja."""
    ax.set_facecolor(LIGHT)


def _shade_handles() -> tuple[list, list]:
    return ([Patch(color=LIGHT), Line2D([], [], color=MUTED, ls="--", lw=1.2)],
            [f"prueba reservada ({r34.dmy(TEST_START)} → {r34.dmy(END)})", "31-12-2018: fin del desarrollo"])


def _vol(x) -> np.ndarray:
    """Varianza anual → volatilidad anual en %."""
    return 100 * np.sqrt(np.asarray(x, dtype=float))


def _ok(name: str, ok: bool, res: str, worst: float = np.nan) -> dict:
    return {"guarda": name, "ok": bool(ok), "peor": worst, "resultado": res}


def _guard(ctx: dict, row: dict) -> None:
    """Añade la guarda de una figura a ctx["guardas"]; si falla, aborta antes de dibujar."""
    from letf_lab.research import r37_indicador_figuras as core

    ctx.setdefault("guardas", []).append(row)
    core.check_all([row])


def _desc(ctx: dict, cifra: str, mercado: str, desarrollo: str, prueba: str, fuente: str) -> None:
    """Una cifra descriptiva (desarrollo y 2019–2026) para la tabla de sec_descriptivo."""
    ctx.setdefault("descriptivo", []).append({"cifra": cifra, "mercado": mercado, "desarrollo": desarrollo,
                                              "prueba reservada (2019–2026)": prueba, "figura": fuente})


# ───────────────────────────── datos (R33): F6 rango_1982 y F9 move ─────────────────────────────

CAP_RANGO = ("Media anual de la varianza de Parkinson dividida por el rendimiento al cuadrado, 1962–2026 (2026 "
             "hasta el 25-9): con un rango real el cociente queda por debajo de 1; antes del 21-4-1982 el de ^GSPC "
             "no lo es; en gris claro, la prueba reservada (R33 y R37)")
CAP_MOVE = ("MOVE (media mensual) y volatilidad realizada del rendimiento a 10 años, de noviembre de 2002 a "
            "septiembre de 2026, en puntos básicos anuales; en gris claro, la prueba reservada (R33 y R37)")


def fig_rango_1982(R, M, out_dir: Path, rb: pd.DataFrame) -> tuple[str, str]:
    """F6: r33.fig_range_break con los años hasta 2026 (rb = r37_indicador_figuras.range_ratio(END));
    2026 es un año parcial (hasta el 25-9) y se marca con un punto hueco."""
    last = int(rb.index.max())
    fig, ax = plt.subplots(figsize=(FIG_W, 4.8))
    _shade_years(ax, last)
    for col, c, lab in (("^GSPC", BLUE, "S&P 500 (^GSPC)"), ("^NDX", ORANGE, "Nasdaq-100 (^NDX)")):
        s = rb[col].dropna()
        ax.plot(s.index, s.values, color=c, lw=2, marker="o", ms=4, label=lab, zorder=3)
        if last in s.index:
            ax.plot([last], [s.loc[last]], color=c, marker="o", ms=7, mfc="white", mew=1.6, zorder=4)
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.axvline(1982.3, color=MUTED, ls="--", lw=1)
    ax.text(1982.8, 4.4, "21-4-1982: primer día\ncon rango real", va="top", fontsize=11, color=MUTED)
    ax.set_xlim(1960, last + 1.5)
    ax.set(ylabel="Parkinson ÷ r² (media del año)", title=f"El rango de ^GSPC en Yahoo es «teórico» hasta 1982 "
           f"(1962–{last})")
    ax.grid(color=GRID, lw=0.6)
    h, lab = ax.get_legend_handles_labels()
    sh, sl = _shade_handles()
    h += [Line2D([], [], color=MUTED, lw=0, marker="o", ms=7, mfc="white", mew=1.6)] + sh
    lab += [f"{last}: hasta el {r34.dmy(END)} (año parcial)", sl[0],
            f"{sl[1]}\n(entre las medias de 2018 y 2019)"]
    legend_band(fig, h, lab, where="bottom", ncol=2, fontsize=11)
    comma_fig(fig)
    _save(fig, out_dir, "rango_1982")
    return "rango_1982", CAP_RANGO


def fig_move(R, M, out_dir: Path, mv: pd.DataFrame) -> tuple[str, str]:
    """F9: r33.fig_move con los meses hasta septiembre de 2026 (mv = move_monthly(END)) y la correlación
    del mismo mes en el desarrollo (la de R33) y en la prueba (descriptiva)."""
    fig, ax = plt.subplots(figsize=(FIG_W, 4.8))
    x = mv.index.to_timestamp()
    _shade(ax, label=False)               # la franja se explica en la leyenda
    ax.plot(x, mv.MOVE, color=BLUE, lw=1.8, label="MOVE (media del mes)")
    ax.plot(x, mv["realizada 10 años"], color=ORANGE, lw=1.4,
            label="volatilidad realizada del 10 años (desviación diaria × √252)")
    c = move_correlations(mv)
    ax.text(0.01, 0.98, f"correlación del mismo mes:\n{num_es(c['desarrollo'], 2)} en el desarrollo, "
            f"{num_es(c['prueba'], 2)} en la prueba (descriptivo)", transform=ax.transAxes, va="top", ha="left", fontsize=11,
            color=MUTED)
    a, b = mv.index.min(), mv.index.max()
    ax.set(ylabel="Puntos básicos anuales", title=f"MOVE y volatilidad realizada del Tesoro, {a.year}–{b.year}")
    ax.set_ylim(0, float(mv.max().max()) * 1.25)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(color=GRID, lw=0.6)
    h, lab = ax.get_legend_handles_labels()
    sh, sl = _shade_handles()
    legend_band(fig, h + sh, lab + sl, where="bottom", ncol=1, fontsize=11)
    comma_fig(fig)
    _save(fig, out_dir, "move")
    return "move", CAP_MOVE


def move_correlations(mv: pd.DataFrame) -> dict:
    """Correlación MOVE–realizada del mismo mes: desarrollo (≤ 12-2018, la de R33) y prueba (descriptiva)."""
    cut = pd.Period(DEV_END, "M")
    out = {}
    for k, part in (("desarrollo", mv[mv.index <= cut]), ("prueba", mv[mv.index > cut])):
        out[k] = float(part.MOVE.corr(part["realizada 10 años"]))
        out[f"meses {k}"] = len(part)
    return out


def sec_datos(rep, tab, ctx: dict) -> None:
    """F6 y F9 (si ctx["data"] existe; las pruebas sintéticas no cargan los datos de R33)."""
    data = ctx.get("data")
    if data is None:
        return
    out = Path(rep.dir)
    rep.h("Datos (R33) hasta el 25-9-2026")
    rep.p("Las dos figuras de R33 que se cortaban en 2018, con la misma lógica de `r33.range_break` y "
          "`r33.move_vs_realized` y los datos cargados con `end = 25-9-2026`. Sus años y meses del desarrollo son "
          "los de R33 (guardas). Las demás figuras de R33 (disponibilidad y seguimiento) ya llegaban al 25-9-2026.")
    _link(rep, *fig_rango_1982(ctx["R"], ctx["R"]["markets"], out, data["rango"]))
    rb = data["rango"]
    t = rb[rb.index > DEV_END.year]
    for col, name in (("^GSPC", "S&P 500"), ("^NDX", "Nasdaq-100")):
        d = rb.loc[(rb.index >= 1983) & (rb.index <= DEV_END.year), col].dropna()
        _desc(ctx, "Parkinson ÷ r², media anual (mín.–máx.)", name,
              f"{num_es(d.min(), 3)}–{num_es(d.max(), 3)} (1983–2018)" if col == "^GSPC" else
              f"{num_es(d.min(), 3)}–{num_es(d.max(), 3)} ({d.index.min()}–2018)",
              f"{num_es(t[col].min(), 3)}–{num_es(t[col].max(), 3)} (2019–{t.index.max()})", "rango_1982")
    _link(rep, *fig_move(ctx["R"], ctx["R"]["markets"], out, data["move"]))
    c = move_correlations(data["move"])
    _desc(ctx, "correlación MOVE–realizada del 10 años (mensual)", "Tesoro 10 años",
          f"{num_es(c['desarrollo'], 3)} ({c['meses desarrollo']} meses)",
          f"{num_es(c['prueba'], 3)} ({c['meses prueba']} meses)", "move")



# ───────────────────────────── varianza (R34): F1 prevision_h21 y N5 prevision_prueba ─────────────────────────────

CAP_PREV = ("Volatilidad prevista por HAR-RI para las 21 sesiones siguientes (línea), con su banda del 90%, y la "
            "realizada (puntos), una previsión cada 21 sesiones hasta agosto de 2026 (la última que se puede evaluar es "
            "la del 26-8-2026, cuyo resultado se conoce el 25-9-2026); en gris claro, la prueba reservada. Es el modelo congelado con datos hasta 2018, "
            "reestimado cada 21 sesiones con ventana creciente (R34 y R37)")
CAP_PREV_TEST = ("Prueba reservada: volatilidad prevista para las 21 sesiones siguientes por HAR-RI (azul, con su banda "
                 "del 90%), por el HAR-log con Duan y por el GJR-GARCH, y la realizada (puntos), una previsión cada 21 "
                 "sesiones desde el 2-1-2019 hasta agosto de 2026, en el S&P 500, el Nasdaq-100 y el bono largo. Las "
                 "QLIKE medias de H1 y H6 que figuran en cada panel usan las 1.923 previsiones diarias del 2-1-2019 "
                 "al 26-8-2026 (R36 y R37)")
_PCOLS = ["media", "q05", "q95", "realizada"]


def forecast_points(R: dict, name: str, h: int, dev_index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Los puntos de F1: la previsión de HAR-RI (media, q05, q95) y la realizada en la ventana común de los
    tres modelos (R["bench"], como el «common» de R34) hasta la última previsión con resultado conocido,
    una cada h sesiones desde el mismo primer día que R34. dev_index: la ventana común del desarrollo
    (dev_frames, cortada en el 28-11-2018 o el 28-9-2018). Devuelve (puntos, guarda)."""
    P = R["var"][name][h]
    idx = R["bench"][name][h].dropna().index
    pts = P.loc[idx].dropna(subset=_PCOLS).iloc[::h]
    dev_pts = P.loc[dev_index].dropna(subset=_PCOLS).iloc[::h]
    head = pts[pts.index <= dev_index.max()]
    same = head.index.equals(dev_pts.index)
    harri = R["bench"][name][h][r34.M_HARRI].reindex(pts.index)
    worst = float(((pts.media - harri).abs() / harri.abs().clip(lower=1.0)).max())
    ok = same and worst <= 1e-12 and pts.index.max() <= END
    res = (f"{len(pts)} puntos del {r34.dmy(pts.index[0])} al {r34.dmy(pts.index[-1])}; los {len(dev_pts)} del "
           f"desarrollo, los de la ventana común de R34 ({num_es(len(dev_index), 0)} días hasta el {r34.dmy(dev_index.max())}); "
           f"la línea es la previsión de HAR-RI evaluada")
    return pts, _ok(f"F1 prevision_h{h}: {name}", ok, res if ok else f"puntos distintos o banda ajena: {res}", worst)


def _forecast_panel(ax, pts: pd.DataFrame, extra: pd.DataFrame | None = None, headroom: float = 1.06) -> float:
    x = pts.index
    ax.fill_between(x, _vol(pts.q05), _vol(pts.q95), color=BLUE, alpha=0.18, lw=0, zorder=2)
    if extra is not None:
        for mname in (r34.M_HARLOG, r34.M_GJR):
            ax.plot(x, _vol(extra[mname]), color=MODEL_COLOR[mname], lw=1.3, zorder=3)
    ax.plot(x, _vol(pts.realizada), color=INK, lw=0, marker="o", ms=2.6, alpha=0.75, zorder=4)
    ax.plot(x, _vol(pts.media), color=BLUE, lw=1.6, zorder=5)
    ax.set_ylabel("volatilidad anual (%)")
    r34._style(ax)
    top = max(float(np.nanmax(_vol(pts.realizada))), float(np.nanmax(_vol(pts.q95))))
    ax.set_ylim(0, top * headroom)        # sin recorte: el pico de marzo de 2020 entra entero
    return top


def _forecast_handles(h: int, models: bool) -> tuple[list, list]:
    hnd = [plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.18, lw=0), Line2D([], [], color=BLUE, lw=2)]
    lab = ["HAR-RI: banda del 90% (cuantiles 5 y 95)", "HAR-RI: raíz de la varianza prevista"]
    if models:
        hnd += [Line2D([], [], color=MODEL_COLOR[m], lw=2) for m in (r34.M_HARLOG, r34.M_GJR)]
        lab += [r34.M_HARLOG, r34.M_GJR]
    hnd.append(Line2D([], [], color=INK, lw=0, marker="o", ms=5))
    lab.append(f"realizada en las {h} sesiones siguientes")
    return hnd, lab


def fig_prevision_h21(R, M, out_dir: Path, pts: dict, h: int = 21) -> tuple[str, str]:
    """F1: r34.fig_forecasts hasta la última previsión con resultado conocido (26-8-2026), con la prueba
    sombreada; pts = {mercado: forecast_points(...)[0]}. La escala vertical llega al punto más alto."""
    names = list(pts)
    fig, axes = plt.subplots(len(names), 1, figsize=(FIG_W, 3.4 * len(names) + 1.2), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        _shade(ax, label=False)           # la franja se explica en la leyenda: el rótulo pisaría 2020
        _forecast_panel(ax, pts[name])
        ax.set_xlim(pts[name].index[0] - pd.Timedelta(days=200), END + pd.Timedelta(days=200))
        ax.set_title(f"{name} ({pts[name].index[0].year}–{pts[name].index[-1].year})")
    hnd, lab = _forecast_handles(h, False)
    sh, sl = _shade_handles()
    legend_band(fig, hnd + sh, lab + sl, where="bottom", ncol=2, fontsize=10.5)
    fig.suptitle(f"Varianza de las próximas {h} sesiones: prevista y realizada")
    comma_fig(fig)
    _save(fig, out_dir, f"prevision_h{h}")
    return f"prevision_h{h}", CAP_PREV


def holdout_forecast_points(R: dict, name: str, h: int, F: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Los puntos de N5: las previsiones de la prueba que evaluó R36 (F = T["prev"][(name, h)], 1.923 días
    con h = 21) con la banda de HAR-RI de R["var"], una cada h sesiones desde el 2-1-2019. Guarda: la
    media de la banda es la previsión de HAR-RI evaluada, y la banda existe todos los días."""
    Q = R["var"][name][h][["media", "q05", "q95"]].reindex(F.index)
    worst = float(((Q.media - F[r34.M_HARRI]).abs() / F[r34.M_HARRI].abs().clip(lower=1.0)).max())
    ok = worst <= 1e-12 and Q[["q05", "q95"]].notna().all().all() and F.index.max() <= END
    pts = F.join(Q[["q05", "q95"]]).rename(columns={r34.M_HARRI: "media"}).iloc[::h]
    res = (f"{len(F)} previsiones del {r34.dmy(F.index[0])} al {r34.dmy(F.index[-1])} (las de R36), "
           f"{len(pts)} dibujadas; banda y media de la misma previsión")
    return pts, _ok(f"N5 prevision_prueba: {name}", ok, res, worst)


def fig_prevision_prueba(R, M, out_dir: Path, pts: dict, losses: dict, h: int = 21) -> tuple[str, str]:
    """N5: HAR-RI con su banda, HAR-log + Duan y GJR-GARCH frente a la realizada en la prueba reservada.
    losses = {mercado: r37_indicador_figuras.loss_table(F entero)} (las de R36 tablas 2, 6 y 10)."""
    names = list(pts)
    fig, axes = plt.subplots(len(names), 1, figsize=(FIG_W, 3.0 * len(names) + 1.5), squeeze=False, sharex=True)
    for ax, name in zip(axes[:, 0], names):
        _test_panel(ax)
        _forecast_panel(ax, pts[name], pts[name], headroom=1.22)
        ql = losses[name]["QLIKE"]
        ax.set_title(name)
        ax.text(0.99, 0.96, "QLIKE media: " + " · ".join(f"{s} {num_es(ql[m], 3)}" for m, s in
                (( r34.M_HARRI, "HAR-RI"), (r34.M_HARLOG, "HAR-log"), (r34.M_GJR, "GJR"))),
                transform=ax.transAxes, ha="right", va="top", fontsize=10, color=MUTED)
    ax = axes[-1, 0]
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(TEST_START - pd.Timedelta(days=20), END + pd.Timedelta(days=20))
    hnd, lab = _forecast_handles(h, True)
    legend_band(fig, hnd, lab, where="bottom", ncol=2, fontsize=10.5)
    fig.suptitle(f"Prueba reservada: varianza de las próximas {h} sesiones")
    comma_fig(fig)
    _save(fig, out_dir, "prevision_prueba")
    return "prevision_prueba", CAP_PREV_TEST


# ───────────────────────────── varianza (R34): F10 perdidas_relativas y F11 calibracion ─────────────────────────────

CAP_REL = ("Pérdida media de HAR-RI y del GJR-GARCH en porcentaje de la del HAR-log con Duan (100): QLIKE (arriba) y "
           "crecimiento perdido (abajo), en el S&P 500, el Nasdaq-100 y el bono largo con h = 21 y h = 63 (el bono "
           "largo, con las pérdidas del desarrollo de R35, no estaba en la figura de R34), en el desarrollo (izquierda) y en la prueba reservada (derecha; previsiones del 2-1-2019 al "
           "26-8-2026 con h = 21 y al 26-6-2026 con h = 63); menos es mejor. Las de la prueba son las de las tablas de "
           "H1 y H6 (R34, R36 y R37)")
CAP_CAL = ("Calibración de la distribución prevista de la varianza: para cada cuantil previsto, la fracción de días en "
           "que la realizada quedó por debajo, en el desarrollo (izquierda) y en la prueba reservada (derecha, "
           "descriptivo); la diagonal es la calibración perfecta (R34 y R37)")
LOSS_COLS = (("QLIKE", "QLIKE"), ("pérdida de crecimiento (pb/año)", "Crecimiento perdido"))


def relative_losses(tables: dict) -> pd.DataFrame:
    """100 × pérdida media de HAR-RI y GJR-GARCH ÷ la del HAR-log + Duan, por (mercado, h) y pérdida.
    tables = {(mercado, h): loss_table(F)} (como r34.fig_relative con la tabla «full»)."""
    rows = {}
    for key, t in tables.items():
        for col, _ in LOSS_COLS:
            for mname in (r34.M_HARRI, r34.M_GJR):
                rows[(*key, col, mname)] = 100 * t.loc[mname, col] / t.loc[r34.M_HARLOG, col]
    s = pd.Series(rows, name="% del HAR-log + Duan")
    s.index = s.index.set_names(["mercado", "h", "pérdida", "modelo"])
    return s.to_frame()


def fig_perdidas_relativas(R, M, out_dir: Path, dev: pd.DataFrame, test: pd.DataFrame) -> tuple[str, str]:
    """F10: r34.fig_relative en una rejilla 2 × 2 (filas QLIKE y crecimiento perdido; columnas desarrollo y
    prueba reservada), con el S&P 500, el Nasdaq-100 y el bono largo × h. dev y test: relative_losses."""
    keys = list(dict.fromkeys((k[0], k[1]) for k in dev.index))
    labels = [f"{n} · h = {h}" for n, h in keys]
    vals = pd.concat([dev.iloc[:, 0], test.iloc[:, 0]])
    hi = float(vals.max())
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 2.0 + 2 * (0.5 * len(keys) + 0.9)), sharey=True, sharex=True)
    y = np.arange(len(keys))[::-1]
    for j, (part, frame) in enumerate((("desarrollo (hasta 2018)", dev), ("prueba reservada (2019–2026)", test))):
        for i, (col, title) in enumerate(LOSS_COLS):
            ax = axes[i, j]
            if j == 1:
                _test_panel(ax)
            for k, (mname, c) in enumerate(((r34.M_HARRI, BLUE), (r34.M_GJR, ORANGE))):
                off = 0.19 if k == 0 else -0.19
                v = [float(frame.loc[(n, h, col, mname)].iloc[0]) for n, h in keys]
                ax.barh(y + off, v, height=0.36, color=c, zorder=2)
                for yi, vi in zip(y, v):
                    ax.text(vi - 2, yi + off, num_es(vi, 0), va="center", ha="right", fontsize=9.5,
                            color="white", fontweight="bold", zorder=3)
            ax.axvline(100, color=INK, lw=1.2, ls="--", zorder=1)
            ax.set_title(f"{title}\n{part}", fontsize=12.5)
            ax.set_xlim(0, max(hi * 1.05, 110))
            r34._style(ax)
            if i == 1:
                ax.set_xlabel("% del HAR-log + Duan")
    axes[0, 0].set_yticks(y)
    axes[0, 0].set_yticklabels(labels, fontsize=10.5)
    axes[1, 0].set_yticklabels(labels, fontsize=10.5)
    legend_band(fig, [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE)] + [Patch(color=LIGHT)],
                [r34.M_HARRI, r34.M_GJR, "prueba reservada (2019–2026)"], where="bottom", ncol=3, fontsize=11)
    fig.suptitle("Pérdida media frente al HAR-log + Duan (= 100; menos es mejor)")
    comma_fig(fig)
    _save(fig, out_dir, "perdidas_relativas")
    return "perdidas_relativas", CAP_REL


CAL_STYLE = {("S&P 500", 21): (BLUE, "-"), ("S&P 500", 63): (BLUE, "--"), ("Nasdaq-100", 21): (ORANGE, "-"),
             ("Nasdaq-100", 63): (ORANGE, "--"), ("Bono largo", 21): (AQUA, "-"), ("Bono largo", 63): (AQUA, "--")}


def fig_calibracion(R, M, out_dir: Path, dev: dict, test: dict) -> tuple[str, str]:
    """F11: r34.fig_reliability en dos paneles, desarrollo (ventanas de R34, tabla 23) y prueba reservada
    (descriptivo). dev y test: {(mercado, h): r34.dispersion_check(...)}."""
    qs = np.array(r34.QUANTS)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 5.4), sharey=True)
    hnd, lab = [], []
    for ax, (part, D) in zip(axes, (("Desarrollo", dev), ("Prueba reservada (descriptivo)", test))):
        if D is test:
            _test_panel(ax)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1.0, ls=":")
        for key, c in D.items():
            col, ls = CAL_STYLE[key]
            (ln,) = ax.plot(qs, [c["fiabilidad"][q] for q in r34.QUANTS], color=col, ls=ls, lw=1.8, marker="o", ms=5)
            if D is dev:
                hnd.append(ln)
                lab.append(f"{key[0]}, h = {key[1]}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        tk = [q for q in qs if q not in (0.10, 0.90)]
        ax.set_xticks(tk)
        ax.set_xticklabels([num_es(q, 2) for q in tk], fontsize=10.5)
        ax.set_xlabel("cuantil previsto (nivel nominal)")
        ax.set_title(part, fontsize=13)
        r34._style(ax)
    axes[0].set_ylabel("frecuencia con que la realizada\nqueda por debajo")
    legend_band(fig, hnd, lab, where="bottom", ncol=2, fontsize=11)
    fig.suptitle("Calibración de la dispersión prevista de HAR-RI")
    comma_fig(fig)
    _save(fig, out_dir, "calibracion")
    return "calibracion", CAP_CAL


def sec_varianza(rep, tab, ctx: dict) -> None:
    """F1, N5, F10 y F11 con las previsiones que prepare() ya comparó con R34 (desarrollo) y R36 (prueba)."""
    from letf_lab.research import r37_indicador_figuras as core

    R, T, D, cfg = ctx["R"], ctx["T"], ctx["D"], ctx["cfg"]
    prem = ctx["doc"]["estimados_en_desarrollo"]["prima_desarrollo"]
    hm, out, Mk = cfg.h_main, Path(rep.dir), R["markets"]
    rep.h("Varianza (R34 y H1/H6 de R36) hasta el 25-9-2026")
    rep.p("Las previsiones son las de `r35.compute` (HAR-RI de R34 congelado, reestimado cada 21 sesiones con "
          "ventana creciente). En el desarrollo solo entran las previsiones cuya ventana objetivo acaba el 31-12-2018 "
          "o antes (hasta el 28-11-2018 con h = 21 y el 28-9-2018 con h = 63), como en R34; las de diciembre de 2018 "
          "ya tienen realizada, pero su resultado es de la prueba. En la prueba, las de R36: del 2-1-2019 al "
          "26-8-2026 (h = 21) o al 26-6-2026 (h = 63). Las pérdidas de la prueba son las de las tablas de H1 y H6 "
          "de R36; la calibración de la prueba es descriptiva (R36 no la informa).")
    pts = {}
    for name in EQUITY:
        if (name, hm) in D["prev"]:
            pts[name], g = forecast_points(R, name, hm, D["prev"][(name, hm)].index)
            _guard(ctx, g)
    _link(rep, *fig_prevision_h21(R, Mk, out, pts, hm))
    tp, losses = {}, {}
    for name in VAR_MARKETS:
        if (name, hm) in T["prev"]:
            F = T["prev"][(name, hm)]
            tp[name], g = holdout_forecast_points(R, name, hm, F)
            _guard(ctx, g)
            losses[name] = core.loss_table(F, prem[name])
    _link(rep, *fig_prevision_prueba(R, Mk, out, tp, losses, hm))
    dev_t, test_t = {}, {}
    for name in VAR_MARKETS:
        for h in cfg.horizons:
            if (name, h) not in D["prev"]:
                continue
            F = D["prev"][(name, h)]
            if name == "Bono largo":                  # la ventana de H6 en R35 (desde el 12-9-1973)
                F = F[F.index >= D["inicio bonos"]]
            dev_t[(name, h)] = core.loss_table(F, prem[name])
            test_t[(name, h)] = core.loss_table(T["prev"][(name, h)], prem[name])
    rd, rt = relative_losses(dev_t), relative_losses(test_t)
    _link(rep, *fig_perdidas_relativas(R, Mk, out, rd, rt))
    tab("Pérdida media de HAR-RI y del GJR-GARCH en % de la del HAR-log + Duan (figura perdidas_relativas): "
        "desarrollo (ventanas de R34; bono largo desde el 12-9-1973, la de H6 en R35) y prueba reservada (R36)",
        rd.rename(columns={rd.columns[0]: "desarrollo"}).join(rt.rename(columns={rt.columns[0]: "prueba reservada"})))
    cd, ct = {}, {}
    for name in VAR_MARKETS:
        for h in cfg.horizons:
            if (name, h) not in D["prev"]:
                continue
            Fd = D["prev"][(name, h)]
            if name == "Bono largo":
                Fd = Fd[Fd.index >= D["inicio bonos"]]
            cd[(name, h)] = core.calibration(R, name, h, Fd.index)
            ct[(name, h)] = core.calibration(R, name, h, T["prev"][(name, h)].index)
            _desc(ctx, f"cobertura de la banda del 90% de HAR-RI, h = {h}", name,
                  f"{pct_es(cd[(name, h)]['cobertura 90%'])} ({num_es(cd[(name, h)]['n'], 0)} días"
                  + (f" desde el {r34.dmy(Fd.index[0])})" if name == "Bono largo" else ")"),
                  f"{pct_es(ct[(name, h)]['cobertura 90%'])} ({num_es(ct[(name, h)]['n'], 0)} días)", "calibracion")
    eq = [k for k in cd if k[0] in EQUITY]
    _link(rep, *fig_calibracion(R, Mk, out, {k: cd[k] for k in eq}, {k: ct[k] for k in eq}))


# ───────────────────────────── utilidades de las figuras de R35 y de la prueba ─────────────────────────────

PURPLE = r35.PURPLE
IND = f"indicador h={r35.Config().h_main}"


def _check(ctx: dict, checks: list) -> list[dict]:
    """Compara cada Check (r37_indicador_figuras.Check) con su tabla congelada y lo añade a las guardas
    (aborta con GuardError si alguno falla)."""
    from letf_lab.research import r37_indicador_figuras as core

    refs = core.refs_for(ctx, checks)
    rows = [c.compare(refs) for c in checks]
    for r in rows:
        _guard(ctx, r)
    return rows


def _drawn(ctx: dict, n: int, W: pd.DataFrame, E, rf: pd.Series, what: str) -> dict:
    """core.drawn_check (tabla n de R36 sobre exactamente el DataFrame W que se dibuja), con las
    referencias de ctx; la fila va a las guardas."""
    from letf_lab.research import r37_indicador_figuras as core

    g = r35.growth_stats(W, E, rf)
    if E is None:
        g = g.drop(columns="exposición media")
    refs = core.refs_for(ctx, [core.Check(core.R36_DIR, n, g, what=what, subset=True)])
    row = core.drawn_check(n, W, E, rf, what, refs)
    row["guarda"] = f"{row['guarda']}, la serie dibujada"
    _guard(ctx, {k: v for k, v in row.items() if k != "valor final"})
    return row


def _wealth(W: pd.DataFrame, start=None) -> pd.DataFrame:
    """Valor de 1 €: (1 + r).cumprod(), con el 1 inicial en el cierre anterior (start) si se da."""
    V = (1 + W).cumprod()
    if start is not None:
        V = pd.concat([pd.DataFrame(1.0, index=[pd.Timestamp(start)], columns=W.columns), V])
    return V


def _log_euro(ax, subs=(1.0,)) -> None:
    """Eje logarítmico de valor en euros con números normales (1, 2, 5, 10, 1.000; coma decimal)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: num_es(v, 0) if v >= 1 else num_es(v, 2 if v < 0.1 else 1)))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _years(ax, base: int = 1) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(base=base))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _test_xlim(ax) -> None:
    ax.set_xlim(DEV_END - pd.Timedelta(days=25), END + pd.Timedelta(days=25))
    _years(ax)


# ───────────────────────────── indicador (R35): F3 apalancamiento_h21, F12 deriva_h21, N2 apalancamiento_prueba ─────────────────────────────

CAP_LEVELS = ("Apalancamiento del indicador con h = 21: el nivel que se tiene cada día (azul) y el L* continuo del Kelly "
              "robusto (gris), desde 1971 (S&P 500), 1996 (Nasdaq-100) y 1973 (bonos) hasta el 25-9-2026; en gris claro, "
              "la prueba reservada (R35 y R37)")
CAP_DRIFT = ("Deriva anual sobre la financiación: prima histórica (gris), contraída con el peso de Kan y Zhou (naranja) y "
             "con el efecto de la tendencia (azul), con la banda de ± un error típico, hasta el 25-9-2026; en gris claro, "
             "la prueba reservada. S&P 500 y bono largo (R35 y R37)")
CAP_LEVELS_TEST = ("Prueba reservada: apalancamiento que tuvo el indicador (h = 21) cada día (azul) y su L* continuo (gris) "
                   "en el S&P 500, el Nasdaq-100 y el bono largo, del 2-1-2019 al 25-9-2026; las marcas rojas de arriba, "
                   "los días en que la restricción de ruina descartó el 3x. Su exposición media fue de 0,68, 0,83 y 0 "
                   "(R36 y R37)")
RUIN = "#c0392b"


def level_stats(R: dict, name: str, h: int, a, b) -> dict:
    """Las cifras de R35 tablas 4 y 5 (restricción de ruina y reparto del L que se tiene) con la misma
    lógica que r35.run, en los días [a, b]. Los cambios se cuentan sobre la serie entera, así que el
    primer día de la prueba cuenta si cambió frente al 31-12-2018."""
    I = R["ind"][(name, h)].dropna(subset=["L"])
    chg = (I.L.diff().fillna(0) != 0)
    chg0 = (I.L_objetivo.diff().fillna(0) != 0)
    w = (I.index >= pd.Timestamp(a)) & (I.index <= pd.Timestamp(b))
    I, chg, chg0 = I[w], chg[w], chg0[w]
    ev = R["evt"][name].reindex(I.index)
    bad3, bad2 = ev.ok_3 == False, ev.ok_2 == False  # noqa: E712
    sh = I.L.value_counts(normalize=True).reindex([0, 1, 2, 3]).fillna(0)
    yrs = len(I) / r35.DAYS
    return {"días": len(I), "3x descartado": float(bad3.mean()), "2x descartado": float(bad2.mean()),
            "primer 3x descartado": r34.dmy(ev.index[bad3][0]) if bad3.any() else "—",
            "último 3x descartado": r34.dmy(ev.index[bad3][-1]) if bad3.any() else "—",
            "días 3x descartado": int(bad3.sum()), "días 2x descartado": int(bad2.sum()),
            "letras": float(sh[0]), "1x": float(sh[1]), "2x": float(sh[2]), "3x": float(sh[3]),
            "L medio": float(I.L.mean()), "L* mediano": float(I.L_star.median()),
            "cambios al año": float(chg.sum() / yrs), "cambios sin histéresis al año": float(chg0.sum() / yrs)}


def level_checks(R: dict, cfg, ctx: dict) -> dict:
    """Guarda: level_stats en el desarrollo reproduce R35 tablas 4 y 5 (h = 21); así la misma lógica
    aplicada a la prueba (descriptiva) es comparable. Devuelve {mercado: (desarrollo, prueba)}."""
    from letf_lab.research import r37_indicador_figuras as core

    h, out, t4, t5 = cfg.h_main, {}, {}, {}
    for name in r35.MARKETS:
        if (name, h) not in R["ind"]:
            continue
        I = R["ind"][(name, h)].dropna(subset=["L"])
        d = level_stats(R, name, h, I.index[0], DEV_END)
        t = level_stats(R, name, h, TEST_START, ctx["T"]["fin"])
        out[name] = (d, t)
        t4[name] = {k: d[k] for k in ("días", "3x descartado", "2x descartado", "primer 3x descartado",
                                      "último 3x descartado")}
        t5[(name, h)] = {k: d[k] for k in ("letras", "1x", "2x", "3x", "L medio", "L* mediano", "cambios al año",
                                           "cambios sin histéresis al año")}
    g4 = pd.DataFrame(t4).T
    g4.index.name = "mercado"
    g5 = pd.DataFrame(t5).T
    g5.index = g5.index.set_names(["mercado", "h"])
    _check(ctx, [core.Check(core.R35_DIR, 4, g4, index="mercado", what="ruina en el desarrollo, lógica de N2"),
                 core.Check(core.R35_DIR, 5, g5, index=["mercado", "h"], subset=True,
                            what="reparto de L y cambios al año en el desarrollo, lógica de N2")])
    return out


def fig_apalancamiento_h21(R, M, out_dir: Path, h: int = 21) -> tuple[str, str]:
    """F3: r35.fig_levels hasta el 25-9-2026, con la prueba sombreada (cuatro paneles, L* en [−1, 5])."""
    names = [n for n in r35.MARKETS if (n, h) in R["ind"]]
    fig, axes = plt.subplots(len(names), 1, figsize=(FIG_W, 2.5 * len(names) + 1.2), squeeze=False)
    for i, (ax, name) in enumerate(zip(axes[:, 0], names)):
        I = R["ind"][(name, h)].dropna(subset=["L"])
        _shade(ax, label=False)
        ax.plot(I.index, I.L_star.clip(-1, 5), color=GREY, lw=0.7)
        ax.step(I.index, I.L, where="post", color=BLUE, lw=1.2)
        ax.set_ylim(-1.1, 5.1)
        ax.set_yticks([-1, 0, 1, 2, 3, 4, 5])
        ax.set_xlim(I.index[0] - pd.Timedelta(days=200), END + pd.Timedelta(days=200))
        ax.set_title(f"{name} ({', '.join(M[name].tickers[k] for k in (1, 2, 3))}), "
                     f"{I.index[0].year}–{I.index[-1].year}")
        r34._style(ax)
    sh, sl = _shade_handles()
    legend_band(fig, [Line2D([], [], color=BLUE, lw=2), Line2D([], [], color=GREY, lw=1.5)] + sh,
                ["L que se tiene (0 letras, 1, 2 o 3 veces)", "L* continuo (recortado a −1 … 5)"] + sl,
                where="bottom", ncol=2, fontsize=10.5)
    fig.suptitle(f"Apalancamiento del indicador, h = {h}, hasta el {r34.dmy(END)}")
    comma_fig(fig)
    _save(fig, out_dir, f"apalancamiento_h{h}")
    return f"apalancamiento_h{h}", CAP_LEVELS


def fig_deriva_h21(R, M, out_dir: Path, h: int = 21) -> tuple[str, str]:
    """F12: r35.fig_drift hasta el 25-9-2026 (S&P 500 y bono largo), con la prueba sombreada."""
    names = [n for n in ("S&P 500", "Bono largo") if (n, h) in R["ind"]]
    fig, axes = plt.subplots(len(names), 1, figsize=(FIG_W, 3.3 * len(names) + 1.4), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        I = R["ind"][(name, h)].dropna(subset=["mu_f"])
        _shade(ax, label=False)
        ax.fill_between(I.index, 100 * (I.mu_f - I.ee), 100 * (I.mu_f + I.ee), color=BLUE, alpha=0.15, lw=0)
        ax.plot(I.index, 100 * I.prior, color=GREY, lw=1.0)
        ax.plot(I.index, 100 * I.shrunk, color=ORANGE, lw=1.3)
        ax.plot(I.index, 100 * I.mu_f, color=BLUE, lw=1.0)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlim(I.index[0] - pd.Timedelta(days=200), END + pd.Timedelta(days=200))
        ax.set_title(f"{name}, {I.index[0].year}–{I.index[-1].year}")
        ax.set_ylabel("% anual")
        r34._style(ax)
    sh, sl = _shade_handles()
    legend_band(fig, [Line2D([], [], color=GREY, lw=2), Line2D([], [], color=ORANGE, lw=2),
                      Line2D([], [], color=BLUE, lw=2), plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.15)] + sh,
                ["prior de μ − f", "prior contraído (Kan y Zhou)", "con la tendencia", "± ee(μ)"] + sl,
                where="bottom", ncol=2, fontsize=10.5)
    fig.suptitle(f"Deriva μ − f del indicador, h = {h}, hasta el {r34.dmy(END)}")
    comma_fig(fig)
    _save(fig, out_dir, f"deriva_h{h}")
    return f"deriva_h{h}", CAP_DRIFT


def fig_apalancamiento_prueba(R, M, out_dir: Path, end, h: int = 21) -> tuple[str, str]:
    """N2: L que tuvo el indicador cada día y su L* continuo en la prueba reservada (S&P 500,
    Nasdaq-100 y bono largo), con los días en que la ruina descartó el 3x marcados arriba."""
    names = [n for n in VAR_MARKETS if (n, h) in R["ind"]]
    fig, axes = plt.subplots(len(names), 1, figsize=(FIG_W, 2.6 * len(names) + 1.4), squeeze=False, sharex=True)
    for ax, name in zip(axes[:, 0], names):
        I = R["ind"][(name, h)].dropna(subset=["L"])
        I = I[(I.index >= TEST_START) & (I.index <= end)]
        _test_panel(ax)
        ax.plot(I.index, I.L_star.clip(-1, 5), color=GREY, lw=0.8)
        ax.step(I.index, I.L, where="post", color=BLUE, lw=1.4)
        bad = R["evt"][name].ok_3.reindex(I.index) == False  # noqa: E712
        if bad.any():
            ax.plot(I.index[bad], np.full(int(bad.sum()), 4.75), color=RUIN, lw=0, marker="|", ms=9, mew=1.2)
        e = float(R["expo"][name][f"indicador h={h}"].reindex(I.index).mean())
        ax.set_ylim(-1.1, 5.1)
        ax.set_yticks([-1, 0, 1, 2, 3, 4, 5])
        ax.set_title(f"{name} ({', '.join(M[name].tickers[k] for k in (1, 2, 3))}): exposición media "
                     f"{num_es(e, 2)}")
        r34._style(ax)
    _test_xlim(axes[-1, 0])
    legend_band(fig, [Line2D([], [], color=BLUE, lw=2), Line2D([], [], color=GREY, lw=1.5),
                      Line2D([], [], color=RUIN, lw=0, marker="|", ms=9, mew=1.5)],
                ["L que se tiene (0 letras, 1, 2 o 3 veces)", "L* continuo (recortado a −1 … 5)",
                 "día en que la ruina descartó el 3x"], where="bottom", ncol=2, fontsize=10.5)
    fig.suptitle(f"Prueba reservada: apalancamiento del indicador (h = {h})")
    comma_fig(fig)
    _save(fig, out_dir, "apalancamiento_prueba")
    return "apalancamiento_prueba", CAP_LEVELS_TEST


def levels_test_checks(R: dict, cfg, ctx: dict) -> None:
    """Guardas de N2 sobre las series dibujadas: la exposición media (L desplazado exec_lag sesiones)
    es la de R36 tablas 22–24 y el L* medio y mediano el de la tabla 17 (H5)."""
    from letf_lab.research import r37_indicador_figuras as core

    h, end, checks, h5 = cfg.h_main, ctx["T"]["fin"], [], {}
    for name in VAR_MARKETS:
        if (name, h) not in R["ind"]:
            continue
        I = R["ind"][(name, h)].dropna(subset=["L"])
        I = I[(I.index >= TEST_START) & (I.index <= end)]
        lagged = R["ind"][(name, h)].L.shift(cfg.exec_lag).reindex(I.index).astype(float)
        ex = R["expo"][name][f"indicador h={h}"].reindex(I.index).astype(float)
        e = float(ex.mean())
        ok = bool(np.allclose(ex.to_numpy(), lagged.to_numpy(), rtol=0, atol=0, equal_nan=True)) and ex.notna().all()
        _guard(ctx, _ok(f"N2 apalancamiento_prueba: {name}, la exposición es el L dibujado {cfg.exec_lag} sesiones "
                        "después", ok, f"{len(I)} días; exposición media {num_es(e, 4)}", 0.0 if ok else np.inf))
        checks.append(core.Check(core.R36_DIR, core.R36_GROWTH[name],
                                 pd.DataFrame({"exposición media": [e]}, index=[f"indicador h={h}"]), subset=True,
                                 what=f"N2, {name}: exposición media"))
        h5[name] = {"L* medio": float(I.L_star.mean()), "L* mediano": float(I.L_star.median())}
    checks.append(core.Check(core.R36_DIR, 17, pd.DataFrame(h5).T, subset=True, what="N2: L* medio y mediano"))
    _check(ctx, checks)


def sec_indicador(rep, tab, ctx: dict) -> None:
    """F3, F12 y N2, con las guardas de su lógica (R35 tablas 4 y 5) y de la prueba (R36 tablas 17 y 22–24)."""
    R, cfg, out = ctx["R"], ctx["cfg"], Path(rep.dir)
    Mk, h = R["markets"], cfg.h_main
    rep.h("El indicador (R35) hasta el 25-9-2026")
    rep.p("El L que se tiene cada día, su L* continuo y la deriva, tal como los calculó el indicador congelado para la "
          "prueba (`r35.compute` hasta el 25-9-2026). Hasta el 31-12-2018 son exactamente los del desarrollo (las 10 "
          "huellas congeladas). La exposición media y el L* medio y mediano de la prueba son los de R36 (tablas 17 y "
          "22–24); el reparto de L, los cambios al año y los días en que la ruina descartó un nivel son descriptivos, "
          "con la misma lógica que R35 tablas 4 y 5 (que reproduce en el desarrollo: guarda).")
    stats = level_checks(R, cfg, ctx)
    levels_test_checks(R, cfg, ctx)
    _link(rep, *fig_apalancamiento_h21(R, Mk, out, h))
    _link(rep, *fig_deriva_h21(R, Mk, out, h))
    _link(rep, *fig_apalancamiento_prueba(R, Mk, out, ctx["T"]["fin"], h))
    for name, (d, t) in stats.items():
        _desc(ctx, f"reparto del L que se tiene, h = {h} (letras / 1x / 2x / 3x)", name,
              " / ".join(pct_es(d[k]) for k in ("letras", "1x", "2x", "3x")),
              " / ".join(pct_es(t[k]) for k in ("letras", "1x", "2x", "3x")),
              "apalancamiento_prueba" if name in VAR_MARKETS else "apalancamiento_h21")
        _desc(ctx, f"cambios de L al año, h = {h} (con histéresis; sin ella)", name,
              f"{num_es(d['cambios al año'], 1)}; {num_es(d['cambios sin histéresis al año'], 1)}",
              f"{num_es(t['cambios al año'], 1)}; {num_es(t['cambios sin histéresis al año'], 1)}", "apalancamiento_h21")
        for lv in (3, 2):
            dd, tt = d[f"días {lv}x descartado"], t[f"días {lv}x descartado"]
            _desc(ctx, f"días en que la ruina descartó el {lv}x", name,
                  f"{pct_es(d[f'{lv}x descartado'])} ({num_es(dd, 0)} días"
                  + (f"; el primero, {d['primer 3x descartado']})" if lv == 3 and dd else ")"),
                  f"{pct_es(t[f'{lv}x descartado'])} ({num_es(tt, 0)} días"
                  + (f"; el primero, {t['primer 3x descartado']})" if lv == 3 and tt else ")"),
                  "apalancamiento_prueba" if name in VAR_MARKETS else "apalancamiento_h21")


# ───────────────────────────── carteras (R35 y R36): F4 valor_h21, N1 y N3 ─────────────────────────────

CAP_WEALTH = ("Valor de 1 € con el indicador (h = 21), el 1x y el 2x fijos y la LRS 3x, en escala logarítmica, desde el "
              "inicio de cada ventana del desarrollo (S&P 500, 14-9-1971; Nasdaq-100, 18-4-1997; bono largo, 12-9-1973) "
              "hasta el 25-9-2026; en gris claro, la prueba reservada (R35 y R37)")
CAP_WEALTH_TEST = ("Prueba reservada, 1-1-2019 → 25-9-2026: valor de 1 € con el indicador (h = 21), el 1x, el 2x y la LRS "
                   "3x en el S&P 500 y el Nasdaq-100 (izquierda, escala logarítmica) y su caída desde el máximo "
                   "(derecha), con orden al cierre siguiente y 5 puntos básicos por unidad negociada. Son cuatro de las "
                   "series de la tabla anterior: faltan las letras y el 3x (R36 y R37)")
CAP_BOND_TEST = ("Bono largo en la prueba reservada, 1-1-2019 → 25-9-2026: valor de 1 € con el indicador (h = 21), que "
                 "estuvo en letras todo el periodo, TLT, UBT, TMF y la LRS del bono, en escala logarítmica (arriba), y su "
                 "caída desde el máximo (abajo) (R36 y R37)")
WEALTH_KEYS = ((IND, BLUE, 2.0), ("1x", INK, 1.2), ("2x", ORANGE, 1.2), ("LRS", AQUA, 2.6))   # LRS gruesa:
# sin color, el naranja del 2x y el verde de la LRS tienen casi la misma luminancia
BOND_KEYS = ((IND, BLUE, 2.0), ("1x", INK, 1.2), ("2x", ORANGE, 1.2), ("3x", PURPLE, 1.2), ("LRS", AQUA, 2.6))
WEALTH_MARKETS = ("S&P 500", "Nasdaq-100", "Bono largo")


def _z(k: str) -> int:
    """La LRS (gruesa) debajo, el indicador encima y el resto en medio."""
    return {IND: 5, SBI: 5, "LRS": 2}.get(k, 3)


def _key_handles(keys) -> list:
    """Las líneas de la leyenda con el grosor relativo de cada serie (la LRS, más gruesa)."""
    return [Line2D([], [], color=c, lw=3.6 if lw >= 2.5 else 2) for _, c, lw in keys]


def _labels(M, name: str, keys) -> list[str]:
    tk = M[name].tickers
    lab = {IND: "indicador (h = 21)", "1x": f"1x ({tk[1]})", "2x": f"2x ({tk[2]})", "3x": f"3x ({tk[3]})",
           "LRS": f"LRS 3x ({tk[3]})"}
    return [lab[k] for k, *_ in keys]


def wealth_frames(R: dict, D: dict, end) -> dict:
    """Las series de F4: R["ret"][m] con el indicador (h = 21), 1x, 2x y LRS, del primer día de la
    ventana del desarrollo (r35.dev_window) al 25-9-2026, continuas (sin reinicio)."""
    out = {}
    for name in WEALTH_MARKETS:
        if name in R["ret"]:
            a = D["ventana"][name][0]
            out[name] = R["ret"][name].loc[a:end, [k for k, *_ in WEALTH_KEYS]]
    return out


def wealth_checks(R: dict, M: dict, ctx: dict, X: dict) -> None:
    """Guardas de F4: sin huecos; el tramo ≤ 31-12-2018 de lo dibujado da R35 tablas 6–8 y el de la prueba
    R36 tablas 22–24 (r35.growth_stats, tolerancia 1e-9)."""
    from letf_lab.research import r37_indicador_figuras as core

    checks = []
    for name, W in X.items():
        gaps = int(W.isna().sum().sum())
        _guard(ctx, _ok(f"F4 valor_h21: {name}, serie continua", gaps == 0 and W.index.max() <= END,
                        f"{len(W)} días del {r34.dmy(W.index[0])} al {r34.dmy(W.index[-1])}, sin huecos"
                        if gaps == 0 else f"{gaps} valores que faltan", 0.0))
        dev, test = W[W.index <= DEV_END], W[W.index >= TEST_START]
        E = R["expo"][name]
        g = r35.growth_stats(dev, E, M[name].rf)
        tk = M[name].tickers
        g.index = [{"1x": f"1x ({tk[1]})", "2x": f"2x ({tk[2]})", "LRS": f"LRS 3x ({tk[3]})"}.get(k, k) for k in g.index]
        checks.append(core.Check(core.R35_DIR, core.R35_GROWTH[name], g, subset=True,
                                 window={"desde": dev.index[0], "hasta": dev.index[-1], "días": None},
                                 what=f"F4, {name}: tramo del desarrollo dibujado"))
        checks.append(core.Check(core.R36_DIR, core.R36_GROWTH[name], r35.growth_stats(test, E, M[name].rf),
                                 subset=True, what=f"F4, {name}: tramo de la prueba dibujado"))
    _check(ctx, checks)


def fig_valor_h21(R, M, out_dir: Path, X: dict) -> tuple[str, str]:
    """F4: r35.fig_wealth sin cortar en 2018 (valor de 1 € continuo hasta el 25-9-2026)."""
    names = list(X)
    fig, axes = plt.subplots(len(names), 1, figsize=(FIG_W, 3.2 * len(names) + 1.2), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        V = _wealth(X[name])
        _shade(ax, label=False)
        for k, c, lw in WEALTH_KEYS:
            ax.plot(V.index, V[k], color=c, lw=lw, zorder=_z(k))
        ax.set_yscale("log")
        _log_euro(ax)
        ax.set_xlim(V.index[0] - pd.Timedelta(days=200), END + pd.Timedelta(days=200))
        ax.set_title(f"{name} ({r34.dmy(V.index[0])} → {r34.dmy(V.index[-1])})")
        ax.set_ylabel("valor de 1 €")
        r34._style(ax)
    sh, sl = _shade_handles()
    legend_band(fig, _key_handles(WEALTH_KEYS) + sh,
                ["indicador (h = 21)", "1x fijo", "2x fijo", "LRS 3x"] + sl, where="bottom", ncol=3, fontsize=10.5)
    fig.suptitle(f"Valor de 1 € hasta el {r34.dmy(END)} (escala logarítmica)")
    comma_fig(fig)
    _save(fig, out_dir, "valor_h21")
    return "valor_h21", CAP_WEALTH


def _value_and_drawdown(ax_v, ax_d, W: pd.DataFrame, keys, title: str) -> pd.DataFrame:
    """Valor de 1 € desde el cierre del 31-12-2018 (escala log.) y caída desde el máximo, en %."""
    V = _wealth(W, DEV_END)
    for k, c, lw in keys:               # la LRS debajo (cuando está dentro es el 3x) y el indicador encima
        z = {IND: 5, "LRS": 2}.get(k, 3)
        ax_v.plot(V.index, V[k], color=c, lw=lw, zorder=z)
        ax_d.plot(V.index, 100 * (V[k] / V[k].cummax() - 1), color=c, lw=lw * 0.7, zorder=z)
    ax_v.set_yscale("log")
    _log_euro(ax_v, (1.0, 2.0, 5.0))
    ax_v.axhline(1, color=MUTED, lw=0.8)
    ax_d.axhline(0, color=MUTED, lw=0.8)
    ax_v.set_title(title, fontsize=12.5)
    for ax in (ax_v, ax_d):
        _test_panel(ax)
        r34._style(ax)
    return V


def fig_valor_prueba_acciones(R, M, out_dir: Path, W: dict) -> tuple[str, str]:
    """N1: valor de 1 € (izquierda) y caída desde el máximo (derecha) en la prueba reservada, S&P 500 y
    Nasdaq-100; W = {mercado: el DataFrame que comprobó drawn_check (tablas 22 y 23 de R36)}."""
    names = list(W)
    fig, axes = plt.subplots(len(names), 2, figsize=(FIG_W, 3.4 * len(names) + 1.4), squeeze=False, sharex=True)
    for i, name in enumerate(names):
        V = _value_and_drawdown(axes[i, 0], axes[i, 1], W[name], WEALTH_KEYS, f"{name}: valor de 1 €")
        axes[i, 1].set_title(f"{name}: caída desde el máximo (%)", fontsize=12.5)
        axes[i, 0].set_ylim(top=float(V.max().max()) * 1.15)
    for ax in axes[-1]:
        _test_xlim(ax)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    legend_band(fig, _key_handles(WEALTH_KEYS),
                ["indicador (h = 21)", "1x fijo", "2x fijo", "LRS 3x"], where="bottom", ncol=4, fontsize=10.5)
    fig.suptitle(f"Prueba reservada: valor de 1 € del {r34.dmy(TEST_START)} al {r34.dmy(END)}")
    comma_fig(fig)
    _save(fig, out_dir, "valor_prueba_acciones")
    return "valor_prueba_acciones", CAP_WEALTH_TEST


def fig_valor_prueba_bono(R, M, out_dir: Path, W: pd.DataFrame, name: str = "Bono largo") -> tuple[str, str]:
    """N3: bono largo en la prueba: indicador (en letras), TLT, UBT, TMF y LRS; valor y caída."""
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 7.6), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.2]})
    _value_and_drawdown(axes[0], axes[1], W, BOND_KEYS, f"{name}: valor de 1 € (escala logarítmica)")
    axes[1].set_title("Caída desde el máximo (%)", fontsize=12.5)
    _test_xlim(axes[1])
    legend_band(fig, _key_handles(BOND_KEYS), _labels(M, name, BOND_KEYS),
                where="bottom", ncol=3, fontsize=10.5)
    fig.suptitle(f"Prueba reservada: bono largo del {r34.dmy(TEST_START)} al {r34.dmy(END)}")
    comma_fig(fig)
    _save(fig, out_dir, "valor_prueba_bono")
    return "valor_prueba_bono", CAP_BOND_TEST


# ───────────────────────────── cartera de acciones y bonos: F2 acciones_y_bonos y N4 cartera_prueba ─────────────────────────────

CAP_SB = ("Cartera de acciones y bonos del indicador frente a la HFEA 55/45 y la LRS 3x (arriba, valor de 1 € en escala "
          "logarítmica) y apalancamiento de cada activo (abajo), del 12-9-1973 al 25-9-2026; en gris claro, la prueba "
          "reservada, en la que la HFEA se recompra a sus pesos al cierre del 31-12-2018, como en R36. Los dos niveles de la cartera se eligen a la vez (L_acc + L_bono ≤ 3), así que no "
          "coinciden con los de los indicadores solos (R35 y R37)")
CAP_SB_TEST = ("Prueba reservada, 1-1-2019 → 25-9-2026: cartera de acciones y bonos del indicador (h = 21) frente a la "
               "HFEA 55/45, comprada al cierre del 31-12-2018, y la LRS 3x del S&P 500 (arriba, valor de 1 € en escala "
               "logarítmica; en trazo discontinuo, la misma cartera con h = 63), y apalancamiento de cada activo (abajo). Los dos niveles de la cartera se eligen a la vez (L_acc + "
               "L_bono ≤ 3), así que no coinciden con los de los indicadores solos. La cartera tuvo TMF el 2,6% de los "
               "días (descriptivo, R37); el indicador del bono largo solo, ninguno (R36 y R37)")
SBI = f"acciones y bonos h={r35.Config().h_main}"
SB_KEYS = ((SBI, BLUE, 2.0), ("HFEA 55/45", ORANGE, 1.2), ("LRS", AQUA, 2.6))
SB_LABELS = ["acciones y bonos (h = 21); abajo, L del S&P 500", "HFEA 55/45", "LRS 3x (S&P 500)", "L del bono largo"]


def stock_bond_frame(R: dict, cfg, D: dict, T: dict) -> pd.DataFrame:
    """F2: la cartera del indicador y la LRS, continuas, y la HFEA encadenada: la del desarrollo
    (r35.run_hfea desde el 12-9-1973 hasta el 31-12-2018, R35 tabla 17) y la de R36 (comprada al
    cierre del 31-12-2018). Encadenar los rendimientos = multiplicar por el valor al 31-12-2018."""
    cols = [k for k, *_ in SB_KEYS]
    return pd.concat([D["SB"][cols], T["SB"][cols]])


def stock_bond_checks(R: dict, cfg, ctx: dict, X: pd.DataFrame) -> None:
    """Guardas de F2: los días son todos los de la cartera (sin huecos ni repetidos); la cartera y la
    LRS son las series continuas del indicador; cada tramo da R35 tabla 17 y R36 tabla 26."""
    from letf_lab.research import r37_indicador_figuras as core

    S = pd.DataFrame(R["sb_ret"])
    S["LRS"] = R["ret"][cfg.sb_equity]["LRS"]
    S = S.loc[X.index[0]:ctx["T"]["fin"]].dropna()
    same_days = X.index.equals(S.index) and X.index.is_unique
    worst = max(float((X[c] - S[c].reindex(X.index)).abs().max()) for c in (SBI, "LRS")) if same_days else np.inf
    _guard(ctx, _ok("F2 acciones_y_bonos: la cartera y la LRS son continuas", same_days and worst == 0,
                    f"{len(X)} días del {r34.dmy(X.index[0])} al {r34.dmy(X.index[-1])}, los de r35.compute; "
                    "HFEA encadenada al 31-12-2018", worst))
    rf = R["markets"][cfg.sb_equity].rf
    dev, test = X[X.index <= DEV_END], X[X.index >= TEST_START]
    g = lambda W: r35.growth_stats(W, None, rf).drop(columns="exposición media")  # noqa: E731
    _check(ctx, [core.Check(core.R35_DIR, 17, g(dev), subset=True,
                            window={"desde": dev.index[0], "hasta": dev.index[-1], "días": None},
                            what="F2: tramo del desarrollo dibujado"),
                 core.Check(core.R36_DIR, 26, g(test), subset=True, what="F2: tramo de la prueba dibujado")])


def _levels_panel(ax, sb: pd.DataFrame) -> None:
    ax.step(sb.index, sb.L_acc, where="post", color=BLUE, lw=1.2)
    ax.step(sb.index, sb.L_bono, where="post", color=PURPLE, lw=1.2)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylim(-0.25, 3.25)
    ax.set_title("Apalancamiento de cada activo (peso L/3 en su 3x)", fontsize=12.5)
    r34._style(ax)


def fig_acciones_y_bonos(R, M, out_dir: Path, X: pd.DataFrame, h: int = 21) -> tuple[str, str]:
    """F2: r35.fig_stock_bond del 12-9-1973 al 25-9-2026 con la prueba sombreada."""
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 7.8), gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
    V = _wealth(X)
    for ax in axes:
        _shade(ax, label=False)
    for k, c, lw in SB_KEYS:
        axes[0].plot(V.index, V[k], color=c, lw=lw, zorder=_z(k))
    axes[0].set_yscale("log")
    _log_euro(axes[0])
    axes[0].set_ylabel("valor de 1 €")
    axes[0].set_title(f"Valor de 1 € (escala logarítmica), {r34.dmy(X.index[0])} → {r34.dmy(X.index[-1])}",
                      fontsize=12.5)
    r34._style(axes[0])
    _levels_panel(axes[1], R["sb"][h].reindex(X.index))
    axes[1].set_xlim(X.index[0] - pd.Timedelta(days=200), END + pd.Timedelta(days=200))
    sh, sl = _shade_handles()
    legend_band(fig, _key_handles(SB_KEYS) + [Line2D([], [], color=PURPLE, lw=2)] + sh,
                SB_LABELS + sl, where="bottom", ncol=2, fontsize=10.5)
    fig.suptitle(f"Cartera de acciones y bonos del indicador hasta el {r34.dmy(END)}")
    comma_fig(fig)
    _save(fig, out_dir, "acciones_y_bonos")
    return "acciones_y_bonos", CAP_SB


def fig_cartera_prueba(R, M, out_dir: Path, W: pd.DataFrame, h: int = 21) -> tuple[str, str]:
    """N4: la cartera de R36 (T["SB"], con la HFEA comprada al cierre del 31-12-2018) en la prueba."""
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 7.4), gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
    V = _wealth(W, DEV_END)
    _test_panel(axes[0])
    _test_panel(axes[1])
    alt = f"acciones y bonos h={max(R['cfg'].horizons)}"
    if alt in V and alt != SBI:
        axes[0].plot(V.index, V[alt], color=BLUE, lw=0.9, ls="--")
    for k, c, lw in SB_KEYS:
        axes[0].plot(V.index, V[k], color=c, lw=lw, zorder=_z(k))
    axes[0].axhline(1, color=MUTED, lw=0.8)
    axes[0].set_yscale("log")
    _log_euro(axes[0], (1.0, 2.0, 5.0))
    axes[0].set_ylabel("valor de 1 €")
    axes[0].set_title("Valor de 1 € desde el cierre del 31-12-2018 (escala logarítmica)", fontsize=12.5)
    r34._style(axes[0])
    _levels_panel(axes[1], R["sb"][h].reindex(W.index))
    _test_xlim(axes[1])
    hnd = _key_handles(SB_KEYS) + [Line2D([], [], color=PURPLE, lw=2)]
    lab = list(SB_LABELS)
    if alt in V and alt != SBI:
        hnd.append(Line2D([], [], color=BLUE, lw=1.2, ls="--"))
        lab.append(f"acciones y bonos (h = {max(R['cfg'].horizons)})")
    legend_band(fig, hnd, lab, where="bottom", ncol=2, fontsize=10.5)
    fig.suptitle("Prueba reservada: cartera de acciones y bonos del indicador")
    comma_fig(fig)
    _save(fig, out_dir, "cartera_prueba")
    return "cartera_prueba", CAP_SB_TEST


def _final_rows(fig: str, market: str, row: dict) -> list[dict]:
    return [{"figura": fig, "mercado": market, "serie": k, "valor final de 1 €": v} for k, v in row["valor final"].items()]


def sec_carteras(rep, tab, ctx: dict) -> None:
    """F4, N1, N3, F2 y N4. Las series de la prueba son las de R36 (tablas 22–24 y 26) y se comprueban
    sobre exactamente el DataFrame que se dibuja; los tramos del desarrollo, con R35 (tablas 6–8 y 17)."""
    R, cfg, T, D = ctx["R"], ctx["cfg"], ctx["T"], ctx["D"]
    Mk, out, end = R["markets"], Path(rep.dir), T["fin"]
    rep.h("Carteras (R35 y R36) hasta el 25-9-2026")
    rep.p("Rentabilidades diarias del indicador congelado, del 1x, 2x y 3x fijos y de la LRS, con orden al cierre "
          "siguiente y 5 pb por unidad negociada, sin reinicio en 2019: el tramo del desarrollo de cada figura da las "
          "tablas de R35 y el de la prueba las de R36 (guardas). En las figuras solo de la prueba, el valor de 1 € "
          "empieza en el cierre del 31-12-2018.")
    X = wealth_frames(R, D, end)
    wealth_checks(R, Mk, ctx, X)
    _link(rep, *fig_valor_h21(R, Mk, out, X))
    finals = []
    for name, W in X.items():
        v = (1 + W).prod()
        _desc(ctx, "valor de 1 € invertido al inicio del desarrollo (indicador / 1x / 2x / LRS)", name,
              f"desde el {r34.dmy(W.index[0])}: " + " / ".join(num_es(float((1 + W.loc[:DEV_END, k]).prod()), 2)
                                                             for k, *_ in WEALTH_KEYS) + " € al 31-12-2018",
              " / ".join(num_es(float(v[k]), 2) for k, *_ in WEALTH_KEYS) + f" € al {r34.dmy(end)}", "valor_h21")
    cols = [k for k, *_ in WEALTH_KEYS]
    WN = {}
    for name in EQUITY:
        if name in T["ret"]:
            WN[name] = T["ret"][name][cols]
            row = _drawn(ctx, core_growth(name), WN[name], T["expo"][name], Mk[name].rf, f"N1, {name}")
            finals += _final_rows("valor_prueba_acciones", name, row)
    _link(rep, *fig_valor_prueba_acciones(R, Mk, out, WN))
    if "Bono largo" in T["ret"]:
        WB = T["ret"]["Bono largo"][[k for k, *_ in BOND_KEYS]]
        row = _drawn(ctx, core_growth("Bono largo"), WB, T["expo"]["Bono largo"], Mk["Bono largo"].rf, "N3, bono largo")
        finals += _final_rows("valor_prueba_bono", "Bono largo", row)
        _link(rep, *fig_valor_prueba_bono(R, Mk, out, WB))
    if "SB" in T and "SB" in D:
        XS = stock_bond_frame(R, cfg, D, T)
        stock_bond_checks(R, cfg, ctx, XS)
        cont = r35.run_hfea(Mk[cfg.sb_equity], Mk[cfg.sb_bond], XS.index[0], end, cfg).reindex(T["SB"].index)
        g = r35.growth_stats(cont.to_frame("HFEA"), None, Mk[cfg.sb_equity].rf).iloc[0]
        rep.p("En la cartera de acciones y bonos, la HFEA 55/45 se encadena: hasta el 31-12-2018 es la del desarrollo "
              f"(R35, comprada el {r34.dmy(XS.index[0])}) y desde el 2-1-2019 la de R36, recomprada a sus pesos al cierre "
              "del 31-12-2018. Una HFEA continua desde 1973 daría en la prueba "
              f"{num_es(float((1 + cont).prod()), 4)} € y un {pct_es(g['CAGR'])} anual, frente a "
              f"{num_es(float((1 + T['SB']['HFEA 55/45']).prod()), 4)} € y "
              f"{pct_es(float(r35.growth_stats(T['SB'][['HFEA 55/45']], None, Mk[cfg.sb_equity].rf).iloc[0]['CAGR']))} "
              "de R36 (descriptivo: la de R36 es la evaluada).")
        _link(rep, *fig_acciones_y_bonos(R, Mk, out, XS, cfg.h_main))
        WS = T["SB"][[c for c in T["SB"].columns]]
        row = _drawn(ctx, 26, WS, None, Mk[cfg.sb_equity].rf, "N4, cartera de acciones y bonos")
        finals += _final_rows("cartera_prueba", "S&P 500 y bono largo", row)
        _link(rep, *fig_cartera_prueba(R, Mk, out, WS, cfg.h_main))
        sbd = {}
        for part, idx in (("desarrollo", D["SB"].index), ("prueba", T["SB"].index)):
            sb = R["sb"][cfg.h_main].reindex(idx)
            sbd[part] = (f"TMF (L del bono ≥ 1) {pct_es(float((sb.L_bono >= 1).mean()))}; L del S&P 500 = 0 / 1 / 2 / 3: "
                         + " / ".join(pct_es(float((sb.L_acc == k).mean())) for k in (0, 1, 2, 3)))
        _desc(ctx, "cartera de acciones y bonos (h = 21): días con TMF y reparto del L del S&P 500",
              "S&P 500 y bono largo", sbd["desarrollo"] + f" (desde el {r34.dmy(D['SB'].index[0])})",
              sbd["prueba"], "cartera_prueba")
    if finals:
        tab("Valor final de 1 € en las figuras de la prueba (del cierre del 31-12-2018 al 25-9-2026): (1 + r).prod() "
            "de las series dibujadas, las de R36", pd.DataFrame(finals).set_index(["figura", "mercado", "serie"]))


def core_growth(name: str) -> int:
    from letf_lab.research import r37_indicador_figuras as core

    return core.R36_GROWTH[name]

def sec_descriptivo(rep, tab, ctx: dict) -> None:
    """La tabla de las cifras descriptivas que han anotado las secciones anteriores (va la última)."""
    rows = ctx.get("descriptivo") or []
    if not rows:
        return
    rep.h("Desarrollo y prueba reservada: cifras descriptivas, no deciden nada")
    rep.p("Descriptivo: no es una hipótesis ni cambia ninguna decisión. Son cifras de 2019–2026 que R36 no informa, "
          "calculadas con el indicador congelado y las mismas series que las figuras, junto a su valor en el "
          "desarrollo (que es el de R33–R35 cuando estos lo daban).")
    tab("Desarrollo y prueba reservada: cifras descriptivas, no deciden nada",
        pd.DataFrame(rows).set_index(["cifra", "mercado"]))


# Orden del README: sec_descriptivo va la última (escribe lo que anotaron las demás).
SECTIONS = [sec_datos, sec_varianza, sec_indicador, sec_carteras, sec_descriptivo]
