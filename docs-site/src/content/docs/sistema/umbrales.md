---
title: Umbrales de confianza
description: Cómo se calibran y por qué no se eligen a ojo.
---

## Estado actual

`AI_THRESHOLDS_CALIBRATED=false`. Mientras siga así, **el servicio marca todas
las respuestas para confirmación manual**, con el motivo
`thresholds_not_calibrated`.

No es un descuido. Los valores que aparecen en la configuración (0.90 y 0.60)
son marcadores de posición sin respaldo empírico, y ponerlos en marcha sin
medirlos primero sería peor que no tener umbral.

## Por qué no se eligen a ojo

Un umbral de confianza decide cuándo el sistema acepta una identificación sin
preguntar. Equivocarse produce dos fallos muy distintos:

- **Umbral demasiado alto**: el operario confirma a mano casi todo. Molesto,
  pero visible y corregible.
- **Umbral demasiado bajo**: el sistema acepta lecturas erróneas sin avisar.
  Un movimiento queda registrado en el cilindro equivocado, el inventario se
  descuadra y nadie se entera hasta la siguiente auditoría.

El segundo es el fallo grave, y es silencioso. Por eso la calibración parte de
fijar un techo de error silencioso tolerable y busca el umbral más bajo que lo
respeta, en vez de buscar la "mejor precisión".

## Cómo calibrar

Necesita el conjunto de test (cilindros que el modelo no vio entrenando) y su
verdad de campo:

```bash
python training/calibrate_thresholds.py \
    --images datasets/raw \
    --ground-truth datasets/ground_truth.csv \
    --split datasets/processed/test.txt \
    --max-silent-error-rate 0.01
```

El script recorre todos los umbrales posibles y muestra, para cada uno, qué
porcentaje automatiza, con qué precisión y cuántos errores silenciosos comete.
Después recomienda el umbral más bajo que respeta el techo.

Con el resultado, actualice `.env`:

```
AI_SERIAL_AUTO_ACCEPT_CONFIDENCE=<recomendado>
AI_SERIAL_REVIEW_CONFIDENCE=<recomendado - 0.30>
AI_THRESHOLDS_CALIBRATED=true
```

## Qué techo elegir

`--max-silent-error-rate` es una decisión de negocio, no técnica. Depende de
cuánto cuesta un cilindro mal registrado frente a cuánto cuesta una
confirmación manual.

Como referencia: el propio informe del proyecto cifra en más de S/24.000 la
pérdida acumulada por 20 cilindros extraviados. A ese precio por activo, un
techo del **1%** es un punto de partida defendible, y conviene empezar más
estricto e ir relajando con datos de operación reales.

Si ningún umbral respeta el techo, el script lo dice y devuelve error. En ese
caso la respuesta correcta no es bajar el listón, sino mejorar la captura o el
modelo, y seguir con confirmación manual mientras tanto.

## Los otros dos umbrales

- `AI_CATALOG_FUZZY_MAX_DISTANCE` (por defecto 2): cuántas ediciones se
  toleran entre lo leído y un serial del inventario. Subirlo recupera lecturas
  peores pero aumenta el riesgo de confundir dos cilindros parecidos.
- `AI_CATALOG_FUZZY_MARGIN` (por defecto 1): cuánto debe separarse el mejor
  candidato del segundo para no declarar el resultado ambiguo. Bajarlo hace
  que el sistema se decida más veces, y se equivoque más.

Ambos deben revisarse cuando el catálogo tenga los 100 cilindros: cuantos más
seriales haya, más probable es que dos se parezcan, y más importa el margen.