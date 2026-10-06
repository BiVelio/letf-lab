# Prerregistro del artículo 6: desviaciones y lecturas

Anexo a `letf/preregistro/articulo6_inversos.md` (commit 8012967). En este trabajo no se puede editar el prerregistro, así que este fichero hace de su sección «Desviaciones», que todavía dice «(ninguna por ahora)». Las entradas llevan la fecha del 30-09-2026. Las del apartado 1 se decidieron **después de ver los resultados**: salieron de la revisión de la primera ejecución completa de R31 y R32. Todas acercan el análisis al texto del prerregistro, y lo que se hacía antes queda en los informes como sensibilidad o como lectura.

## 1. Corregido tras la revisión (30-09-2026, después de ver los resultados)

| # | Qué hacía la primera versión | Qué dice el prerregistro | Cómo queda |
|---|---|---|---|
| 1 | R31 · H2: la variante principal era un corto **sin préstamo** con el producto remunerado. | El préstamo es una rejilla (mínimo / mediana / p90 de cada ticker) y el producto se remunera a la letra − 0,5% o a 0. | H2 recorre los seis puntos de la rejilla. La referencia es el préstamo mediano con el producto remunerado. El corto sin préstamo solo aparece como referencia, marcado «fuera de la rejilla». Resultado: 11 de 36 con la referencia; entre 4 (p90, con el producto remunerado o sin remunerar) y 13 (mínimo, remunerado) en la rejilla; 12 sin préstamo. |
| 2 | R31 · H1, primera parte: el veredicto usaba D, el arrastre neto del carry y de la fricción. Con él se cumplía en 10 de 11. | «El arrastre realizado del inverso frente a L veces el rendimiento logarítmico del índice», es decir, L·Σlog(1 + u) − Σlog(1 + R). | El veredicto usa el arrastre literal: se cumple en 2 de 11 (SOXS y FAZ). D (10 de 11) y D + φ (8 de 11) se dan como lecturas. El resumen dice que H1 solo se cumple neta del carry y de la fricción. |
| 3 | R31 · H1, segunda parte: «aproximadamente el doble» se había cambiado por «±10% del cociente de la fórmula», que vale 3 con \|L\| = 2. | «Aproximadamente el doble». | El veredicto usa el doble, entre 1,8 y 2,2. Con D se cumple en 6 de 9; fallan SDS/SSO (3,12), QID/QLD (3,03) y FAZ/FAS (2,44). SH y PSQ no se pueden contrastar. El cociente de la fórmula se da aparte. |
| 4 | R31 · H2: bootstrap con bloque medio max(63, 2h). | «Bloque medio 63». | Bloque 63. max(63, 2h) queda como sensibilidad y no cambia ningún veredicto de la referencia. |
| 5 | R31 · H2: la mediana se corregía con Holm. | «SPA de Hansen y Romano–Wolf sobre la familia completa». | Romano–Wolf en la familia de 36 de cada variante. Holm queda como sensibilidad y no cambia ningún veredicto de la referencia. |
| 6 | R32 · H3: incluía SDS y QID (−2x), SH y PSQ (−1x) y TECS, un fondo que el prerregistro no nombra. En (b) usaba el alcista del mismo \|L\|. Todos entraban en los recuentos, en Holm entre 12 pares y en la familia conjunta de 11. | «Vender en corto el −3x», y (b) es «el +3x comprado». | H3, H4 y H5 se juzgan en los 9 mercados −3x: las dos historias largas más SQQQ, SPXU, SOXS, TZA, FAZ, LABD y SDOW. Holm se aplica entre esos 7 pares, y la familia conjunta tiene 6 (todos menos LABD). Los otros cinco quedan como exploratorios, no cuentan y conservan su alcista del mismo \|L\|. Ningún veredicto de H3 cambia. |
| 7 | R31 · textos: se daba como hecho que el índice de FAS y FAZ «no fue el de XLF hasta 2020». | — | Se presenta como hipótesis sin verificar. |

## 2. Lecturas que el prerregistro no fija (escritas al programar, antes de ver resultados)

- **H1.** «Ventanas anuales» son años naturales con al menos 63 sesiones del par. «En cada par real» se juzga con el inverso de cada par. La regresión es MCO con constante y errores de White.
- **H2.** «Mantener en corto el −3x» se concreta como el corto sin reajuste del protocolo con s = 1 (exposición inicial +3, la misma que el +3x comprado con el 100%). La mediana se juzga con α = 0,05. La media «no es significativamente mayor que cero» si su p unilateral, sin corregir, es de al menos 0,05.
- **H3.** El prerregistro deja abiertas las dos formas de remunerar el producto, así que el veredicto se da con las dos. Mientras una operación está abierta no se leen señales, y entre dos operaciones hay al menos una sesión en caja. La LRS de comparación es la 3x del índice de cada par.
- **H4.** Se usan las operaciones con fecha de entrada única de las reglas que comparten s y h. Los errores son de Hansen-Hodrick con h − 1 retardos, con las operaciones ordenadas por fecha. «Desaparece con VR ≥ 1» significa que la media condicionada no es significativamente mayor que 0.
- **H5.** Pérdida de más del 50% al cierre de alguna sesión de la tenencia, contrastada con McNemar sobre las operaciones discordantes.
- **Añadidos que no están en el prerregistro.** Holm entre pares y la familia conjunta de R32, porque el SPA corrige la búsqueda dentro de cada familia pero no entre mercados.

## 3. Desviación de aplicación que se mantiene

- **R31 · H2, Romano–Wolf.** El procedimiento necesita los remuestreos de toda la familia a la vez. Las 9 series tienen calendarios distintos, así que cada combinación de serie y h se remuestrea por separado y la distribución nula del máximo se calcula como si las columnas fueran independientes. Frente a la dependencia positiva real, la corrección sale algo conservadora, no más permisiva.
