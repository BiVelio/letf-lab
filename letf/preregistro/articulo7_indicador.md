# Prerregistro — Artículo 7: un indicador del apalancamiento óptimo para SSO, UPRO, QLD y TQQQ

Fijado el 30-09-2026, antes de construir el indicador. El commit que introduce este fichero es su sello de tiempo; cualquier cambio posterior se anota en «Desviaciones» con fecha y motivo.

## Idea

No se busca el «día óptimo» (el artículo 1 no lo encontró), sino **cuánto apalancamiento conviene llevar cada día**: 0 (letras), 1x, 2x (SSO, QLD) o 3x (UPRO, TQQQ). El indicador estima, para cada L, el crecimiento logarítmico esperado del propio fondo de reajuste diario en las próximas h sesiones, con su incertidumbre, y elige el L que maximiza un crecimiento robusto con una restricción de ruina.

## Componentes (fijados de antemano)

1. **Objetivo propio del fondo.** Crecimiento esperado del fondo L en h sesiones: L·(μ − f − s) − ER − ½·L²·IV(h), con μ la rentabilidad media del índice, f la financiación, s el sobrecoste del swap, ER la comisión y IV(h) la varianza integrada de las próximas h sesiones.
2. **Modelo de varianza «HAR-RI».** HAR logarítmico con: varianza de rango diaria (estimador de Yang y Zhang con máximo, mínimo, apertura y cierre), semivarianza negativa, componente de saltos, VIX², pendiente VIX/VIX3M (desde que existe) y VVIX (desde que existe); devuelto a niveles con el factor de Duan en tiempo real. Da la media y la dispersión prevista de IV(h).
3. **Deriva con contracción.** μ = una prima a priori (la media histórica del exceso sobre f con datos hasta t − 1) contraída hacia un valor prudente con el peso de Kan y Zhou (2007), más el estado de tendencia (por encima o por debajo de la media de 200 sesiones de la serie con dividendos) solo si su efecto se estima con datos hasta t − 1.
4. **Kelly robusto.** L* = (μ − f − s − κ·ee(μ)) / (E[IV] + κ·de(IV)), con ee(μ) el error típico de la deriva y de(IV) la dispersión prevista de la varianza; κ = 1 fijo.
5. **Restricción de ruina.** Probabilidad condicional de un día con caída de al menos 1/L (EVT condicional: Pareto generalizada sobre los residuos estandarizados, al estilo de McNeil y Frey, 2000); se descarta cualquier L con probabilidad anual mayor que 1/1.000.
6. **Histéresis.** El indicador solo cambia de L si la mejora del crecimiento esperado en h sesiones supera el coste de la operación.
7. **Horizontes:** h = 21 (principal) y h = 63 (sensibilidad). Salida diaria: L* continuo y la recomendación discreta en {0, 1, 2, 3}.

## Muestras

- Estimación y desarrollo: S&P 500 y Nasdaq-100 hasta el 31-12-2018 (S&P desde 1960 con datos de rango; Nasdaq-100 desde 1985-10).
- **Prueba reservada: 1-1-2019 → 25-9-2026**, que se evalúa una sola vez con los parámetros congelados. Contaminación declarada: los artículos 1–6 ya usaron esos años con otras reglas; el indicador no.
- Tras publicarse, seguimiento en vivo del indicador durante 6 meses (lectura diaria registrada).

## Hipótesis

- **H1:** HAR-RI prevé IV(21) mejor que HAR-log con Duan y que GJR-GARCH en la prueba reservada (QLIKE y pérdida de crecimiento; conjunto de mejores modelos y Diebold–Mariano, p < 0,10).
- **H2:** La cartera que sigue al indicador (con histéresis) crece más que el 1x y que el 2x fijos en la prueba reservada (Δg > 0 con bootstrap por bloques, p < 0,10, una cola).
- **H3:** El indicador no supera a la LRS de Gayed y Bilello (misma prueba reservada, SPA sobre las variantes h y κ previstas arriba; mejora solo si p < 0,10).
- **H4:** A igual exposición media, el indicador tiene menor caída máxima y menor probabilidad de pérdida del 50% que la LRS.
- **H5:** El L* del indicador está correlacionado negativamente con la varianza prevista y apenas con la tendencia: casi toda su variación sale de la varianza (descomposición de la varianza de L*).

## Protocolo

Orden al cierre siguiente; 5 pb por unidad negociada; fondos sintéticos calibrados antes de su lanzamiento y reales después; financiación interbancaria; inferencia con bootstrap estacionario emparejado (bloque 63), SPA y Romano–Wolf.

## Qué se publica pase lo que pase

H1–H5 en la prueba reservada, tal como salgan, el código y la lectura diaria del indicador.

## Desviaciones

**30-09-2026, antes de construir nada: ampliación a bonos apalancados** (petición de la dirección). Se añade, con las mismas reglas:

- **Instrumentos:** bono del Tesoro de EE. UU. a largo plazo (TLT como 1x; UBT 2x; TMF 3x) y a medio plazo (IEF como 1x; UST 2x; TYD 3x). Los inversos (TBT −2x, TMV −3x, PST, TTT) solo como referencia, sin decisiones.
- **Datos:** rentabilidad total diaria de un bono sintético a vencimiento constante construido con los rendimientos diarios de FRED (DGS10, DGS20, DGS30, con los huecos cubiertos por el vencimiento más cercano y declarados), validado contra TLT e IEF desde 2002 (el laboratorio ya construyó un bono largo sintético que reproduce TLT en R24). Varianza de rango con los máximos y mínimos de TLT e IEF desde 2002; antes, solo con cierres. Índice MOVE si se puede descargar con historia; si no, se declara y se omite.
- **Modelo de varianza:** la misma estructura HAR-RI adaptada (varianza de rango, semivarianza, saltos, y el MOVE en lugar del VIX cuando exista).
- **Deriva:** carry del bono (rendimiento menos financiación, más el efecto de rodar por la curva) como prior, con la misma contracción.
- **Muestras:** desarrollo hasta el 31-12-2018 (bono sintético desde 1962); prueba reservada 1-1-2019 → 25-9-2026, que incluye la caída de 2022.
- **Hipótesis añadidas:**
  - **H6:** HAR-RI prevé la varianza integrada a 21 sesiones del bono largo mejor que HAR-log con Duan y que GJR-GARCH en la prueba reservada (mismos contrastes que H1).
  - **H7:** la cartera que sigue al indicador en el bono largo crece más que el 1x y el 2x fijos del bono largo en la prueba reservada (mismo contraste que H2), y tiene menor caída máxima que el 3x fijo (TMF).
  - **H8:** una cartera de acciones y bonos que aplica el indicador a cada activo, con la correlación prevista entre los dos, no supera a la LRS de Gayed y Bilello ni a la HFEA 55/45 en crecimiento en la prueba reservada (mejora solo si p del SPA < 0,10), pero sí tiene menor caída máxima que la HFEA.

**30-09-2026 / 01-10-2026, al construir los datos y el indicador con datos solo hasta el 31-12-2018 y antes de evaluar la prueba reservada.** Decisiones que el texto no fijaba o que los datos obligaron a cambiar (detalle en `letf/reports/r33_indicador_datos`, `r34_harri` y `r35_indicador`, y en `articulo7_config_congelada.json`):

- **Datos de rango.** El máximo y el mínimo diarios del S&P 500 en Yahoo solo son reales desde el 21-4-1982; antes son «teóricos», así que hasta esa fecha el modelo de varianza usa solo cierres. El Nasdaq-100 tiene rango desde 1985-10. Muchas aperturas repiten el cierre anterior; en esos días el estimador de Yang y Zhang no tiene parte nocturna (marca `open_ok`).
- **MOVE.** Solo existe desde el 12-11-2002 (Yahoo, ^MOVE); antes, el modelo de bonos no lo usa.
- **Bonos sintéticos.** Vencimientos de 25 años (imita a TLT) y 8,5 años (imita a IEF), con carry y efecto de rodar por la curva. El salto de +74 pb del rendimiento a 20 años de FRED el 4-1-1973 es un cambio de método: ese día el bono solo cobra carry y rodadura.
- **Fondos de Direxion (TMF, TYD, TMV).** Su sobrecoste se calibra desde el 1-1-2011, porque hasta finales de 2010 seguían otro índice (beta diaria de 3,0–3,6). Los datos de precio de TYD son malos (294 sesiones sin volumen y un cierre erróneo el 26-11-2014); el bono medio no entra en ninguna hipótesis.
- **Especificación del indicador.** La deriva se contrae hacia μ − f = 0 con el peso de Kan y Zhou; en bonos, el prior es el carry con un día de retraso. El efecto de la tendencia se reestima cada 21 sesiones con errores de Hansen y Hodrick. La probabilidad de ruina es la diaria condicional anualizada.
- **Reglas de decisión.** H2 y H7 con p de Romano y Wolf; el SPA de H3 sobre h ∈ {21, 63} (κ está fijo en 1); H4 a igual exposición media; H5 con una descomposición de Shapley exacta y umbral 0,75; H8 con UPRO y TMF en pesos L/3 y L_acciones + L_bonos ≤ 3.
- **Observación del periodo de desarrollo (sin ningún cambio posterior):** con κ = 1 el Kelly robusto es muy conservador; en 1971–2018 el indicador del S&P 500 está en letras el 59% de los días y crece menos que el 1x. No se ajustó nada tras verlo.

**05-10-2026, después de evaluar la prueba reservada (anotación tardía).** Desviación que la configuración congelada (commit 74ff502) ya aplicaba antes de la prueba, pero que este apartado no recogía:

- **Contraste de H1 y H6.** El texto registrado nombra «conjunto de mejores modelos y Diebold–Mariano, p < 0,10». La regla congelada decide solo con el Diebold–Mariano de una cola frente a los dos benchmarks en cada pérdida (confirmada con las dos, parcial con una) y deja el MCS como información. Es una desviación del texto, no una concreción. Motivo de anotarla ahora: la revisión del artículo 7 señaló que faltaba aquí. Veredictos en la prueba (R36, tablas 2–3, 6–7 y 10–11): con la regla congelada, H1 parcial en el S&P 500, no en el Nasdaq-100, y H6 confirmada; si además el MCS al 90% tuviera que descartar a los dos benchmarks, H1 no en el S&P 500 (QLIKE: p MCS 0,262 de los dos; crecimiento: HAR-log dentro, p 0,63), no en el Nasdaq-100, y H6 parcial (crecimiento: fuera los dos, p 0,088 y 0,064; QLIKE: dentro los dos, p 0,153). Lo registrado arriba no se modifica.
