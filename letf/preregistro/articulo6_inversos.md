# Prerregistro — Artículo 6: vender en corto los ETFs inversos apalancados en los repuntes

Fijado el 30-09-2026, antes de las pruebas confirmatorias. El commit que introduce este fichero es su sello de tiempo; cualquier cambio posterior se anota al final, en «Desviaciones», con su fecha y motivo.

## La idea que se contrasta

La propuesta de la dirección: un ETF inverso apalancado pierde valor por el reajuste diario con más violencia que el apalancado alcista, así que ponerse corto en el inverso, sobre todo en los repuntes del mercado, debería dar dinero.

La parte mecánica tiene base: frente a L veces el rendimiento logarítmico del índice, un fondo de reajuste diario pierde aproximadamente ½(L² − L) veces la suma de los rendimientos diarios al cuadrado. Con L = 3 son 3 veces; con L = −3, 6 veces. El estudio pregunta si esa pérdida adicional se puede cobrar, cuándo y a qué precio.

## Lo que ya se ha visto (contaminación declarada)

Los artículos 3 y 4 ya midieron cortos de SQQQ y SPXU (R22, R29), el par corto TQQQ + SQQQ y la descomposición de la caída de SQQQ: −75,5 puntos logarítmicos al año, de los que −53,9 son −3 veces el Nasdaq-100 y −25,7 el arrastre por volatilidad. Un corto que se reajusta cada día no cobra ese arrastre; uno que no se reajusta lo cobra como una apuesta a que el cociente de varianzas del periodo sea menor que 1. También se sabe que la autocorrelación diaria del índice es negativa desde 2000. Las reglas nuevas de abajo no se han probado.

## Datos y ventanas

- Pares reales inverso/alcista: SQQQ/TQQQ, SPXU/UPRO, SDS/SSO, QID/QLD, SH, PSQ, SOXS/SOXL, TZA/TNA, FAZ/FAS, LABD/LABU, SDOW/UDOW y los que la caché y Yahoo permitan, con su vida real (2006–2026 como mucho).
- Historia larga sintética con el simulador del laboratorio (validado para L < 0 en R22): S&P 500 desde 1928 y Nasdaq-100 desde 1985-10.
- Coste de préstamo: distribución observada en iBorrowDesk (IBKR) por ticker; rejilla mínimo / mediana / p90 de cada uno. Producto del corto remunerado a la letra − 0,5% o a 0.

## Protocolo común

- Señal al cierre de t y orden al cierre de t+1. 5 pb por unidad negociada. La caja cobra la letra.
- Corto sin reajuste: se vende una fracción s del capital en el inverso y no se toca hasta la salida. Mantenimiento FINRA del 30% × |L| (90% para un −3x); si se incumple, el corto se reduce ese cierre. Ruina si el capital llega a cero.
- Inferencia: Δg con bootstrap estacionario emparejado (bloque medio 63); SPA de Hansen y Romano–Wolf sobre la familia completa; regresiones con errores de Hansen–Hodrick.

## Hipótesis

- **H1 · Mecánica.** En cada par real, el arrastre realizado del inverso frente a L veces el rendimiento logarítmico del índice coincide con ½(L² − L)Σr² (pendiente entre 0,9 y 1,1 y R² > 0,9 en ventanas anuales), y es aproximadamente el doble que el del alcista con el mismo |L|.
- **H2 · Mediana frente a media.** Mantener en corto el −3x h sesiones (h = 5, 21, 63, 252) gana en la mediana más que mantener el +3x las mismas sesiones, pero no en la media: la diferencia de medias no es significativamente mayor que cero en un bootstrap por bloques de la historia. El «decaimiento» se cobra vendiendo la cola derecha del inverso.
- **H3 · Cortos en repuntes.** La familia de reglas: vender en corto el −3x (s = 1/3 y s = 1) tras un repunte del índice — subida de al menos k% en n sesiones (k ∈ {3, 5, 10}, n ∈ {5, 10, 21}) —, en tres estados: por encima de la media de 200 sesiones, por debajo (repunte en mercado bajista) y sin condición; mantenerlo h ∈ {5, 21, 63} sesiones sin reajuste. Ninguna regla supera tras la búsqueda (p del SPA < 0,10), con el préstamo mediano de cada ticker, a: (a) la caja, (b) el +3x comprado las mismas sesiones con la misma exposición inicial, (c) la LRS de Gayed y Bilello del artículo 2.
- **H4 · Dónde sale la ventaja.** La ventaja del corto sin reajuste frente al equivalente con reajuste diario se explica por el cociente de varianzas del periodo: coeficiente positivo sobre (1 − VR) con errores HAC, y la ventaja desaparece cuando VR ≥ 1.
- **H5 · La cola.** La probabilidad de perder más del 50% del capital dentro del periodo de tenencia es mayor en las reglas de H3 que en el +3x comprado con la misma exposición.

## Qué se publica pase lo que pase

H1–H5 tal como salgan, con la familia completa, los episodios que deciden los resultados (1987, 2000–2002, 2008, 2020, 2022, agosto de 2024) y los límites: préstamo con solo 12 meses de datos, recompras forzosas, acceso minorista en la UE y marco regulatorio.

## Desviaciones

(ninguna por ahora)
