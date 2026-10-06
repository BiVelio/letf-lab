# Prerregistro — Estudio en papel: carry BTC y HYPE en Hyperliquid

Fijado el 29-09-2026. El commit que introduce este fichero es su sello de tiempo; el estudio empieza con la primera foto registrada después de ese commit y dura **26 semanas**. Cualquier cambio posterior se anota en «Desviaciones» con su fecha y motivo.

**Es un estudio en papel y de solo lectura.** No hay claves, carteras, firmas ni órdenes. No invierte dinero de la Fundación ni de nadie. Antes de cualquier uso de capital real hace falta el dictamen legal y fiscal del punto K6.

## Estrategia simulada

- Largo al contado y corto en el perpetuo, mismo nocional, en BTC (spot UBTC, par @142) y HYPE (spot nativo, par @107).
- Tamaño de referencia: 25.000 USD por pata.
- Modalidad de capital: margen de cartera (capital = nocional, préstamo en stablecoins según la fórmula documentada) y, como sensibilidad, modo estándar a 2x y 1x.
- Costes: comisiones de nivel 0 (perp 0,045% taker, spot 0,070% taker), coste ejecutable de entrada y salida medido en el libro, préstamo del margen de cartera y coste de cubrir el EUR/USD (€STR y funding del perp xyz:EUR, registrados).

## Datos que se registran

- `fundingHistory` de BTC y HYPE cada hora (con relleno de huecos por la API).
- `predictedFundings` cada 5 minutos (Hyperliquid, Binance, Bybit).
- Libro L2 de BTC, @142, HYPE y @107 cada 5 minutos: coste ejecutable de una orden de 25.000 USD por pata.
- Precio del oráculo, mark y spot de UBTC frente a BTC (desanclaje de Unit).
- Precio de USDC frente al USD desde una fuente pública.
- Eventos: ADL que afecten a cortos de BTC o HYPE, pausas, delistados, cambios de parámetros.

## Hipótesis

- **H1:** funding bruto realizado ≥ 6% APR en BTC y ≥ 8% APR en HYPE a lo largo de las 26 semanas.
- **H2:** rendimiento neto de comisiones, préstamo y cobertura EUR ≥ €STR + 3 puntos anuales.
- **H3:** coste ejecutable de ida y vuelta ≤ 30 pb en al menos el 95% de las fotos.

## Criterios de parada (basta uno)

- **K1:** APR móvil de 30 días < 0 durante 14 días seguidos.
- **K2:** exceso neto sobre el €STR < 0 a las 13 semanas.
- **K3:** un ADL que afecte a cortos de BTC o HYPE, o un desanclaje de UBTC de más del 1% durante más de 1 hora.
- **K4:** USDC por debajo de 0,99 USD durante más de 1 hora.
- **K5:** cambio en el tipo de interés del funding o en los LTV del margen de cartera.
- **K6:** dictamen legal o fiscal negativo (código de conducta de la CNMV para entidades sin ánimo de lucro, perímetro CFD de ESMA, Ley 49/2002). Detiene el estudio sean cuales sean los números.
- **K7:** España o la UE pasan a ser jurisdicción restringida en los términos de Hyperliquid.

## Qué se publica pase lo que pase

El resultado de H1–H3, los criterios de parada que se hayan activado y los huecos de datos, a las 13 y a las 26 semanas.

## Desviaciones

**29-09-2026, antes de ver resultados (el estudio llevaba menos de 3 horas).** Aclaraciones de implementación de puntos que el texto no fija. No cambian hipótesis, umbrales ni criterios. Motivo: la revisión del código (`hl-paper`) mostró que eran decisiones no declaradas.

1. **Inicio.** La primera foto registrada tras el commit es la de 2026-09-29T20:00:02.406Z y tiene las cuatro patas ejecutables. Queda fijada en el código; la evaluación no se hace si esa foto falta en los datos. El sello de tiempo es el commit 9c271b6, aunque la rama se integre después con otro hash.
2. **Cobertura EUR (H2).** El coste de cubrir el EUR/USD es el funding que paga un largo en xyz:EUR por el importe del capital. No se imputan comisiones, margen ni préstamo de esa pata, porque sus comisiones dependen de cómo configure el mercado HIP-3 quien lo despliega. El €STR se registra y es la referencia de H2; no se cobra además como coste, porque ya está en el funding del perp. Sin ningún dato de xyz:EUR, H2 queda sin evaluar.
3. **Huecos de funding.** Una hora cuenta cuando ya ha habido tiempo de descargarla (70 minutos). En BTC y HYPE, una hora sin dato cuenta como 0 en H1 y en el P&L. En xyz:EUR se imputa la media de las horas recibidas. Si falta más del 2% de las horas de alguna serie, H1 y H2 se publican marcadas como «datos incompletos».
4. **K1.** El APR es el del funding bruto (suma diaria de F × 365), no el neto de la estrategia. Se evalúa por separado en BTC, en HYPE y en la cartera (media de ambos) y basta con que se active en una. Se usan días UTC completos del estudio. Una hora sin dato se imputa con la media de su día. Un día sin ninguna hora queda vacío. La ventana de 30 días necesita 27 con dato, y un día sin APR móvil corta la racha.
5. **Margen de cartera.** El corto usa un margen inicial de nocional / 10, como el ejemplo de la documentación. El préstamo cubre ese margen, los costes y las pérdidas no realizadas del corto. Se calcula por activo y no se abona rendimiento al USDC ocioso.
6. **Coste de salida.** Si el libro de la última foto no alcanza para 25.000 USD, se toma el mayor de dos valores: el peor coste completo visto hasta entonces, o el tramo que sí se llena más la comisión.
7. **K5 y K7 a revisión manual.** K5 se activa solo si cambia el r implícito del funding o un LTV. Se revisa a mano si hay horas fuera de la fórmula, si UBTC o HYPE dejan de aparecer entre las garantías o si no hay datos de las reservas en 2 horas. En K7, cualquier mención de España o la UE en los términos pide revisión manual; K7 solo se activa tras esa revisión. Lo mismo ocurre si cambia la fecha o el texto de los términos.
