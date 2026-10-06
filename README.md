# letf-lab · Research by BiVelio

[Español](#español) · [English](#english)

## Español

Este es el motor con el que **Research by BiVelio** calcula sus artículos sobre ETFs apalancados
(<https://research.bivelio.com>, publicados en <https://bivelio.substack.com>), junto con los
prerregistros que fijamos antes de las pruebas. Es una instantánea: el código tal como está hoy, para
que puedas leer cómo se obtiene cada cifra y comprobarla tú.

> Herramienta educativa; no es una recomendación de inversión.

### Qué hay

| Ruta | Qué es |
|---|---|
| `letf/letf_lab/sim.py` | Fondo apalancado de reajuste diario: `L·r + (1 − L)·rf − |L|·sobrecoste del swap − comisión`, con suelo en −100% |
| `letf/letf_lab/backtest.py` | Backtest vectorizado: la señal al cierre de t se ejecuta al cierre de t+1, con costes por rotación |
| `letf/letf_lab/kelly.py`, `stats.py`, `metrics.py` | Apalancamiento de Kelly, bootstrap estacionario, SPA de Hansen, Romano–Wolf, métricas |
| `letf/letf_lab/data/` | Código que **descarga** los datos de sus fuentes públicas (Yahoo Finance, FRED, Cboe, Kenneth French, Shiller…) y los guarda en una caché local. No incluye ningún dato |
| `letf/letf_lab/research/rN_*.py` | Una línea de investigación por fichero: cada una escribe su informe en `letf/reports/<línea>/` |
| `letf/preregistro/`, `hyperliquid/preregistro/` | Los prerregistros, byte a byte los mismos que publicamos en <https://research.bivelio.com/prerregistros/> |

### Instalación

Necesitas Python 3.11 o posterior.

```bash
cd letf
python -m venv .venv && . .venv/bin/activate
pip install -e .            # o: uv pip install -e .
```

### Cómo reproduce cada artículo

Cada línea se ejecuta con `python run_all.py <línea>` desde `letf/`. La primera vez descarga los datos
que necesita (la caché va a `letf/data/cache/`, o a la carpeta de la variable `LETF_CACHE`) y escribe el
informe, con sus tablas y figuras, en `letf/reports/<línea>/README.md`.

| Artículo | Líneas |
|---|---|
| 1. ¿Existe el momento óptimo para comprar un ETF 3x? | `r9_entry`, `r10_predictability`, `r11_doctoral` |
| 2. Apalancar con reglas: lo que gana la LRS | `r12_long_ml` … `r20_investor` |
| 3. ¿Hay algo mejor que la media de 200? | `r21_trend` … `r25_census_plus` |
| 4. ¿Se puede cobrar la caída de UVXY y SQQQ? | `r26_vix_data` … `r30_vol_portfolio` |
| 6. ¿Se gana vendiendo SQQQ en los repuntes? | `r31_inverse_drag`, `r32_short_rallies` |
| 7. Un indicador de apalancamiento para ETFs 2x y 3x | `r33_indicador_datos` … `r37_indicador_figuras` |

Por ejemplo: `python run_all.py r13_lrs`. Los artículos de Hyperliquid (5 y 8) salen de otra línea de
trabajo con sus propios datos; aquí están sus prerregistros.

Las cifras pueden moverse en el último decimal si las fuentes corrigen datos antiguos, y las series más
recientes no estarán en los artículos ya publicados: cada informe dice hasta qué fecha llegan sus datos.
El prerregistro del artículo 7 se evalúa **una sola vez** (`r36_indicador_holdout` deja una marca,
`letf/preregistro/articulo7_holdout_ejecutado.json`, y se niega a repetir la prueba). Lo único que hemos
cambiado en esa marca es la ruta del informe, que ahora es relativa al repositorio.

### Comprobar un prerregistro

Cada página de <https://research.bivelio.com/prerregistros/> da el commit que introdujo el documento y
su huella SHA-256. Compruébala con:

```bash
shasum -a 256 letf/preregistro/articulo7_indicador.md
```

### Licencia

**Todos los derechos reservados (BiVelio Inc).** El código está a la vista para que puedas verificarlo;
no tiene licencia de código abierto ni concede ningún permiso de uso, copia, modificación o distribución.
Ver `LICENSE`.

## English

This is the engine **Research by BiVelio** uses to compute its articles on leveraged ETFs
(<https://research.bivelio.com/en/>, published in Spanish at <https://bivelio.substack.com>), together
with the preregistrations we fixed before running the tests. It is a snapshot: the code as it stands
today, so you can read how every figure is obtained and check it yourself.

> Educational tool; not investment advice.

### What's here

| Path | What it is |
|---|---|
| `letf/letf_lab/sim.py` | Daily-reset leveraged fund: `L·r + (1 − L)·rf − |L|·swap spread − expense ratio`, floored at −100% |
| `letf/letf_lab/backtest.py` | Vectorised backtest: a signal at the close of t is executed at the close of t+1, with turnover costs |
| `letf/letf_lab/kelly.py`, `stats.py`, `metrics.py` | Kelly leverage, stationary bootstrap, Hansen's SPA, Romano–Wolf, metrics |
| `letf/letf_lab/data/` | Code that **downloads** the data from its public sources (Yahoo Finance, FRED, Cboe, Kenneth French, Shiller…) into a local cache. No data is included |
| `letf/letf_lab/research/rN_*.py` | One research line per file: each writes its report to `letf/reports/<line>/` |
| `letf/preregistro/`, `hyperliquid/preregistro/` | The preregistrations, byte for byte the ones published at <https://research.bivelio.com/en/preregistrations/> (written in Spanish) |

### Installation

You need Python 3.11 or later.

```bash
cd letf
python -m venv .venv && . .venv/bin/activate
pip install -e .            # or: uv pip install -e .
```

### How it reproduces each article

Run each line with `python run_all.py <line>` from `letf/`. The first run downloads the data it needs
(cached in `letf/data/cache/`, or in the folder set by `LETF_CACHE`) and writes the report, with its
tables and figures, to `letf/reports/<line>/README.md` (in Spanish).

| Article | Lines |
|---|---|
| 1. Is there a best time to buy a 3x ETF? | `r9_entry`, `r10_predictability`, `r11_doctoral` |
| 2. Leverage with rules: what the LRS earns | `r12_long_ml` … `r20_investor` |
| 3. Is there anything better than the 200-day average? | `r21_trend` … `r25_census_plus` |
| 4. Can you collect the decay of UVXY and SQQQ? | `r26_vix_data` … `r30_vol_portfolio` |
| 6. Does shorting SQQQ after rallies pay? | `r31_inverse_drag`, `r32_short_rallies` |
| 7. A leverage indicator for 2x and 3x ETFs | `r33_indicador_datos` … `r37_indicador_figuras` |

For example: `python run_all.py r13_lrs`. The Hyperliquid articles (5 and 8) come from a separate line
of work with its own data; their preregistrations are here.

Figures can move in the last decimal if the sources revise old data, and the most recent data won't be
in articles already published: each report says how far its data goes. The article 7 preregistration is
evaluated **only once** (`r36_indicador_holdout` leaves a marker,
`letf/preregistro/articulo7_holdout_ejecutado.json`, and refuses to run the test again). The only change
we made to that marker is the report path, now relative to the repository.

### Checking a preregistration

Every page at <https://research.bivelio.com/en/preregistrations/> gives the commit that introduced the
document and its SHA-256 fingerprint. Check it with:

```bash
shasum -a 256 letf/preregistro/articulo7_indicador.md
```

### Licence

**All rights reserved (BiVelio Inc).** The code is visible so you can verify it; it has no open-source
licence and grants no permission to use, copy, modify or distribute it. See `LICENSE`.
