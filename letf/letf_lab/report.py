"""Salida de cada línea de investigación: reports/<nombre>/README.md + figuras."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Figuras pensadas para leerse en una columna de ~730 px y ampliarse al hacer clic:
# 200 ppp y FIG_W pulgadas de ancho (1.600 px), así que la columna las reduce a ~0,46 y
# 12 pt salen a ~15 px (el cuerpo del texto va a ~19 px). Las figuras de los artículos
# usan FIG_W; con 11–12 pulgadas la misma letra saldría a ~10 px.
FIG_W = 8.0
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "figure.titlesize": 15,
    "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
})
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "reports"

PCT_COLS = {"CAGR", "vol", "maxDD", "ulcer"}


def fmt(df: pd.DataFrame, pct: set[str] | None = None, digits: int = 2) -> pd.DataFrame:
    """Formatea columnas porcentuales y redondea el resto (para markdown)."""
    pct = PCT_COLS | (pct or set())
    out = df.copy()
    for c in out.columns:
        if c in pct:
            # entre 99,9% y 100% se muestran dos decimales: «−100,0%» haría creer que el
            # fondo quedó exactamente a cero
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else
                                f"{v:.2%}" if 0.999 < abs(v) < 1 else f"{v:.1%}")
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(digits)
    return out


class Report:
    def __init__(self, name: str, title: str):
        self.dir = ROOT / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lines = [f"# {title}", ""]

    def h(self, text: str, level: int = 2):
        self.lines += [f"{'#' * level} {text}", ""]

    def p(self, text: str):
        self.lines += [text.strip(), ""]

    def table(self, df: pd.DataFrame, pct: set[str] | None = None, index: bool = True,
              digits: int = 2):
        self.lines += [fmt(df, pct, digits).to_markdown(index=index), ""]
        df.to_csv(self.dir / f"table_{len(self.lines)}.csv")

    def fig(self, fig, name: str, caption: str = ""):
        layout(fig)
        fig.savefig(self.dir / f"{name}.png", dpi=200)
        plt.close(fig)
        self.lines += [f"![{caption or name}]({name}.png)", ""]

    def save(self):
        (self.dir / "README.md").write_text("\n".join(self.lines))
        print(f"→ {self.dir / 'README.md'}")


# ───────────────── maquetación: paneles y leyenda de toda la figura ─────────────────

BAND_GAP_IN = 0.08   # pulgadas entre el título general y una leyenda de arriba, o entre el borde y una de abajo
BAND_TUCK_IN = 0.07  # se descuentan del margen de tight_layout (~0,2 pulgadas) bajo una leyenda de arriba


def legend_band(fig, handles, labels, where: str = "top", **kw):
    """Leyenda de toda la figura en una franja propia, encima de los paneles (bajo el título
    general) o debajo de ellos; layout() le reserva exactamente su alto.

    Sustituye a la fila de la rejilla dedicada a la leyenda: tight_layout da a todas las filas
    el mismo hueco vertical, el que necesitan las marcas y la etiqueta del eje x de un panel
    más el título del siguiente, así que entre esa fila y el primer panel quedaba una franja en
    blanco de 2–4 líneas de texto a 730 px."""
    if where not in ("top", "bottom"):
        raise ValueError(where)
    kw.setdefault("frameon", False)
    kw.setdefault("borderaxespad", 0.0)
    leg = fig.legend(handles, labels, loc="upper center" if where == "top" else "lower center", **kw)
    fig._legend_band = (leg, where)
    return leg


def layout(fig) -> None:
    """tight_layout con sitio para el título general y, si la hay, para la leyenda de
    legend_band. Con rect distinto de 1 arriba, tight_layout ya descuenta el título general
    (matplotlib ≥ 3.6): reservarlo otra vez lo contaba dos veces y dejaba un hueco grande, así
    que solo se añade un respiro de 0,1 pulgadas."""
    H = fig.get_figheight()
    sup = fig._suptitle
    band = getattr(fig, "_legend_band", None)
    top = 1 - 0.1 / H if sup is not None else 1
    if band is None:
        fig.tight_layout(rect=(0, 0, 1, top))
        return
    leg, where = band
    r = fig.canvas.get_renderer()
    px = fig.dpi * H                               # alto de la figura en píxeles del lienzo
    gap = BAND_GAP_IN / H
    if where == "top":
        y = (sup.get_window_extent(r).y0 / px if sup is not None else 1) - gap
        leg.set_bbox_to_anchor((0.5, y), transform=fig.transFigure)
        # los paneles, con sus títulos, empiezan justo bajo la leyenda: tight_layout añade su margen
        # de siempre (~0,2 pulgadas), del que se descuenta BAND_TUCK_IN porque la caja de la leyenda
        # ya trae su propio relleno (~15 px a 730 en total); el título general queda por encima
        top = leg.get_window_extent(r).y0 / px + BAND_TUCK_IN / H
        was = sup.get_in_layout() if sup is not None else None
        if sup is not None:
            sup.set_in_layout(False)
        try:
            fig.tight_layout(rect=(0, 0, 1, top))
        finally:
            if sup is not None:
                sup.set_in_layout(was)
    else:
        leg.set_bbox_to_anchor((0.5, gap), transform=fig.transFigure)
        fig.tight_layout(rect=(0, leg.get_window_extent(r).y1 / px, 1, top))


# ───────────────── números en castellano para rótulos de figuras y textos ─────────────────
# Ayudantes aditivos: no cambian fmt() ni las tablas. Coma decimal, punto de
# miles y signo menos tipográfico (U+2212), como el texto de los artículos.

MINUS = "−"


def num_es(x: float, d: int = 2, miles: bool = True) -> str:
    """Número con coma decimal y «−»: num_es(−0.0234, 2) → «−0,02»; num_es(1998, 0) → «1.998»."""
    s = f"{x:,.{d}f}" if miles else f"{x:.{d}f}"
    s = s.replace(",", "\0").replace(".", ",").replace("\0", ".")
    return s.replace("-", MINUS)


def pct_es(x: float, d: int = 1) -> str:
    """Fracción como porcentaje sin espacio: pct_es(0.1234, 1) → «12,3%»."""
    return num_es(100 * x, d) + "%"


def comma_axis(ax) -> None:
    """Coma decimal en las marcas de los ejes LINEALES con el formateador numérico
    por defecto. Los ejes logarítmicos (10^k), de fechas o de categorías no se
    tocan. Se llama después de dibujar, para no pisar el formateador de fechas."""
    from matplotlib.ticker import ScalarFormatter

    class _Coma(ScalarFormatter):
        def __call__(self, x, pos=None):
            return super().__call__(x, pos).replace(".", ",")

    for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
        if scale == "linear" and type(axis.get_major_formatter()) is ScalarFormatter:
            axis.set_major_formatter(_Coma())


def comma_fig(fig) -> None:
    """comma_axis en todos los ejes de la figura (incluidos los gemelos)."""
    for ax in fig.axes:
        comma_axis(ax)
