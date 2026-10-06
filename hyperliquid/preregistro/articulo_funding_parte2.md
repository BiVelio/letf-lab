# Prerregistro — Hyperliquid, parte 2: cómo capturar el funding

Fijado el 30-09-2026, antes de las pruebas confirmatorias. El commit que introduce este fichero es su sello de tiempo; los cambios posteriores se anotan en «Desviaciones».

**Es un estudio de datos públicos y solo de lectura.** No hay claves, carteras, firmas ni órdenes. No es una recomendación de inversión ni una propuesta de invertir dinero de la Fundación.

## Lo que ya se ha visto (contaminación declarada)

El artículo 5 (informe `hyperliquid/reports/suelo_funding`) midió: el suelo de la fórmula, el carry contado contra perpetuo en BTC, ETH, SOL y HYPE, su compresión 2024–2026, el diferencial frente a Binance y Bybit, la base, la probabilidad de liquidación de un corto aislado, el 10-10-2025 y reglas de entrada y salida por umbral (peores que quedarse dentro). Esas cifras no se repiten como hallazgos nuevos.

## Formas de capturar el funding que se estudian

1. **Carry transversal:** cartera neutral que cada día (o cada semana) se pone corta en los perpetuos con mayor funding previsto y se cubre en contado donde existe, o solo con perpetuos opuestos.
2. **Diferencial entre plataformas accesibles:** Hyperliquid frente a otros perpetuos descentralizados con API pública (por ejemplo dYdX, Lighter, Aster, Paradex, Drift; los que tengan historial descargable), y frente a Binance y Bybit solo como referencia.
3. **«Cazar» el pago horario:** abrir justo antes del cobro y cerrar después. Se contrasta con la regla de cálculo documentada (a qué posiciones se paga y sobre qué precio) y con los costes.
4. **Market making con reembolsos:** lo que aportan las comisiones maker y los reembolsos frente al funding, con los niveles documentados.
5. **Previsión del funding:** si el funding previsto y la prima de la hora anterior predicen el de la hora siguiente mejor que el último valor, y si eso permite mejorar la entrada.

## Hipótesis

- **H1:** el carry transversal, neto de comisiones taker y del coste ejecutable medido en el libro, no supera al carry estático en BTC con la misma exposición (Δg con bootstrap por bloques, SPA sobre la familia de reglas; p < 0,10 para declarar mejora).
- **H2:** el diferencial de funding entre Hyperliquid y al menos una plataforma descentralizada accesible es positivo de media en BTC y ETH (IC 90% por bloques excluye 0), pero no cubre el coste de mover capital entre plataformas y de dos patas de comisiones con capital pequeño.
- **H3:** cazar el pago horario no es rentable con comisiones de nivel 0: el cobro esperado de una hora es inferior al coste de ida y vuelta en más del 99% de las horas.
- **H4:** el funding previsto mejora la previsión de la hora siguiente frente al último valor (pérdida cuadrática menor, contraste de Diebold–Mariano), pero las reglas de entrada basadas en él no superan a quedarse dentro tras la búsqueda.
- **H5:** el market making con reembolsos solo compensa por volumen de nivel alto; con el nivel 0 no existe reembolso maker.

## Qué se publica pase lo que pase

H1–H5 con su familia completa, los costes ejecutables, la accesibilidad desde España de cada plataforma y el marco legal ya descrito en el artículo 5. Los primeros resultados del estudio en papel de 26 semanas se citan solo como estado, sin conclusiones.

## Desviaciones

Anotadas el 30-9-2026, al completar el estudio, y revisadas el mismo día tras la revisión del estudio (detalle en `hyperliquid/reports/captura_funding/README.md`, sección 8). Todas se decidieron después de ver resultados, parciales o completos.

1. **H1, familia.** El prerregistro fija las coberturas («en contado donde existe, o solo con perpetuos opuestos») y la frecuencia diaria o semanal; no fija k, las señales ni el universo, que elige el módulo (k ∈ {1, 3, 5}; último pago, media de 24 h y de 7 días). «En contado donde existe» se prueba en sus dos lecturas: solo los perps con contado, y contado donde existe con perps opuestos en el resto (mixta). Con los perps opuestos, 54 reglas. Una versión parcial incluía en la familia una cobertura larga en el perp de BTC, que no está en el prerregistro y contenía la mejor regla de la muestra (+11,3 pp al año); ya tenía resultados cuando se sacó de la familia. Se da aparte, como informativa, con el SPA de todas las reglas como sensibilidad.
2. **H1, «la misma exposición».** El nocional corto es siempre el capital, como en la referencia: con n < k candidatos, 1/n por nombre. La versión anterior (1/k por nombre, exposición menor cuando faltan candidatos) se da como sensibilidad.
3. **H4, primera parte.** `predictedFundings` no tiene historial. Se contrasta con un registro en vivo propio (cada minuto). El contraste decisivo: una entrada por hora de reloj (media de d entre monedas y sondeos), solo horas completas (sondeos antes del minuto 10 y después del 50), Diebold–Mariano sin retardos, p < 0,10 y al menos 3 horas. Una versión anterior sumaba el registro del estudio en papel, contaba dos veces las horas comunes y horas parciales, y su regla se fijó después de ver 17 horas del estudio en papel; se retiró. El registro propio tiene 2 horas completas: la primera parte queda **sin evaluar**. Si se evalúa con los datos del estudio en papel cuando publique en su semana 13, será con esta misma regla (BTC y HYPE). Como prueba secundaria, con el historial: regresión del pago siguiente sobre el último pago y la prima de la hora anterior.
4. **H4, segunda parte.** Las reglas de entrada del prerregistro se basan en el funding previsto, que no tiene historial; se usan sus sustitutos con historial: el último pago y la regresión con el último pago y la prima.
5. **H3.** El «cobro esperado» se mide con el pago realizado (cota superior: al final de la hora el previsto coincide casi con él). Que quien está dentro en el momento del pago cobre la hora entera se infiere de que la documentación no habla de prorrateo; con prorrateo, cazar rendiría menos.
6. **H2.** «Positivo en BTC y ETH frente a al menos una plataforma» se lee como la misma plataforma en las dos monedas, y la segunda parte («no cubre el coste») se lee con esa misma plataforma; la lectura moneda a moneda se da como secundaria. «Accesible», como términos de uso que no excluyen a España. La posición dura toda la ventana (caso principal), con la sensibilidad a un año y las ventanas de 365 días. Costes: mediana de varias fotos de libros; Paradex en el régimen por API (libro por API y la comisión por API de `markets`, 2 pb), con el de la web (libro «interactive» y comisión 0) como sensibilidad; una versión anterior mezclaba el libro por API con la comisión 0.
7. **H5.** «Compensa» se concreta como: el diferencial cotizado de los perps líquidos supera dos comisiones maker en la mayoría del volumen (foto de libros, sin selección adversa). Solo informan los niveles 0 a 3: desde el nivel 4 la comisión maker es 0 y cualquier diferencial cubre por construcción.
8. **Estudio en papel.** Sus datos no se usan en ninguna hipótesis, solo su estado. Una versión anterior los usaba en H4 y en H5.
