# Prerregistro — Artículo 4: la prima de la volatilidad y la venta de ETFs que se deprecian

Fijado el 29-09-2026, antes de las pruebas confirmatorias. El commit que introduce este fichero es su sello de tiempo; cualquier cambio posterior se anota al final, en «Desviaciones», con su fecha y motivo.

## Lo que ya se ha visto (contaminación declarada)

Una exploración previa, fuera del laboratorio y sin corrección por búsqueda, ya miró: comprar y mantener SVXY, SVIX y VYLD; cortos de VXX, VIXY, UVXY, UVIX y VIXM con distintos préstamos; los filtros «VIX/VIX3M < 1» y «VIX por debajo de su media de 10 sesiones»; el filtro «F1 > VIX» y su fragilidad al quitar días; la regla de Simon y Campasano antes y después de 2012; cortos de SQQQ y SPXU; y mezclas con la LRS. Esas cifras no se citan; las pruebas de abajo se repiten en el laboratorio con su protocolo y la familia completa se corrige por búsqueda.

## Datos y ventanas

- Futuros VX de Cboe (liquidaciones diarias), índices corto y medio plazo reconstruidos con la metodología de S&P y validados contra VIXY, VXX, VIXM, VXZ, UVXY y UVIX. VIX, VIX3M, VIX9D, VIX6M y VVIX de los CSV públicos de Cboe (no de Yahoo, que tiene al menos un cierre erróneo del VIX, el 6-2-2026).
- Ventana principal: índice corto sintético desde el 20-12-2005 (medio plazo desde el 24-4-2006) hasta el 25-9-2026.
- Robustez: productos reales 2011–2026; diseño actual de SVXY y UVXY (−0,5x y 1,5x) desde el 28-2-2018.
- Todo resultado con exposición −1x se da con las dos convenciones de hora (liquidación de las 16:15 y cierre de las 16:00) y con y sin el 5 y el 6-2-2018.

## Protocolo común

- Señal al cierre de t y orden al cierre de t+1. 5 pb por unidad negociada. La caja cobra la letra del Tesoro.
- Cortos: rejilla de préstamo anual 0 / 2 / 5,2 / 18% (mínimo, mediana y p90 observados en IBKR) y el producto del corto remunerado a la letra − 0,5% o a 0. Mantenimiento FINRA de 30% × |apalancamiento| (tope 100%); si se incumple, el corto se reduce ese mismo cierre. Ruina si el capital llega a cero.
- Inferencia: Δg (diferencia de crecimiento logarítmico anual) con bootstrap estacionario emparejado de bloque medio 63; SPA de Hansen y Romano–Wolf sobre cada familia; regresiones con errores de Hansen–Hodrick para objetivos solapados.

## Hipótesis

- **H1 · La prima predice el rendimiento del futuro.** En 2016-01 → 2026-09, fuera de la muestra de Cheng (2019), su prima ex ante (futuro menos una previsión del VIX estimada solo con datos pasados) predice el rendimiento a un mes del futuro con β > 0 (HAC, 5%, una cola). Se falsa si β ≤ 0 o no es significativo.
- **H2 · Ningún filtro de estructura de plazos sobrevive a la búsqueda.** Familia: {F1 > VIX, VIX < VIX3M, F2 > F1, roll diario > 0,10 (Simon y Campasano), prima de Cheng > 0}, aplicada a un −0,5x (sintético hasta el 27-2-2018, SVXY real después). Ningún filtro supera a estar siempre −0,5x con p_SPA < 0,10. Si alguno lo hace, se repite quitando sus 3 mejores días; si entonces Δg ≤ 0, se informa como frágil.
- **H3 · Vender el ETF que se deprecia.** (a) A igual exposición, el corto de UVXY (y de UVIX) supera a comprar SVXY (SVIX) antes de préstamo solo por el tamaño del corto y la frecuencia de reajuste; un corto sin reajuste diario gana frente al diario si y solo si el cociente de varianzas del periodo es menor que 1. (b) Con el préstamo mediano y sin remunerar el producto del corto, Δg ≤ 0 frente al inverso comprado. Lo mismo para SQQQ frente a QQQ/TQQQ.
- **H4 · Gestión por volatilidad.** Escalar el −0,5x por la varianza prevista, a igual exposición media, no mejora Δg (p_SPA > 0,10).
- **H5 · ¿Prima o beta?** El −0,5x tiene alfa CAPM positiva frente al SPY (HAC, p < 0,05, una cola) en la ventana principal y con SVXY real desde 2018.

## Qué se publica pase lo que pase

Los resultados de H1–H5 tal como salgan, con su familia completa, las cifras frente a la LRS del artículo 2 y los límites (muestra con pocos episodios extremos, productos que cambiaron en 2018, préstamo con solo 12 meses de datos de un bróker, acceso minorista en la UE).

## Desviaciones

(ninguna por ahora)
