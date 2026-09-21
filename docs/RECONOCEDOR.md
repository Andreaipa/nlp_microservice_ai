# Reconocedor especializado: experimento y resultado

**Resultado: no funcionó con los datos disponibles. El componente queda
desactivado.** Este documento explica qué se intentó, por qué falló y qué
haría falta, para que nadie repita el intento por el mismo camino.

## El problema que intentaba resolver

La evaluación de extremo a extremo dejó el diagnóstico claro: el detector
localiza el troquelado con confianza ~0.83, pero el OCR genérico lo lee con un
**CER de 0.537** — más de la mitad de los caracteres mal. Ése es el techo del
sistema, y no se sube ajustando umbrales.

La hipótesis era razonable: un reconocedor genérico está entrenado para leer
miles de caracteres en cualquier tipografía, mientras que aquí hacen falta
**21 símbolos** en un estilo único, y en la práctica son los diez dígitos más
la `K`. Un modelo pequeño especializado debería tener ventaja.

## Qué se hizo

1. **Dataset real sin capturar nada nuevo** (`build_ocr_dataset.py`): 318
   recortes del troquelado con su texto correcto, cruzando las cajas que
   localizó la pre-anotación con la ficha de cada cilindro.
2. **Datos sintéticos** (`synth_stamped.py`): 8.000 recortes que imitan el
   troquelado —relieve con luz direccional, fondo metálico veteado, óxido,
   espaciado irregular de punzón— sobrerrepresentando los caracteres que
   apenas aparecen en el inventario.
3. **CRNN + CTC** (`train_recognizer.py`): la arquitectura estándar para texto
   en línea, entrenada desde cero.

## Qué salió

| Intento                         | CER sobre fotos reales | Seriales exactos |
| ------------------------------- | ---------------------- | ---------------- |
| Sólo sintéticos                 | 0.596                  | 0%               |
| + sobremuestreo de reales ×16   | 0.664                  | 0%               |
| + afinado final sólo con reales | **0.597**              | **0%**           |
| _OCR genérico (referencia)_     | _0.537_                | _—_              |

Peor que el punto de partida, y con lecturas seguras de sí mismas pero
equivocadas:

```
esperado 21S062189  ->  "202139"     conf 0.68
esperado 21S062189  ->  "21K43"      conf 0.66
esperado Y4523012   ->  "1462612"    conf 0.77
```

## Por qué falló

**El error de fondo fue entrenar desde cero.** Con 243 recortes reales de
entrenamiento no hay material para que una red aprenda desde la nada qué es un
carácter, y los 8.000 sintéticos no suplen esa carencia: la pérdida bajó a
0.08, señal de que el modelo aprendió muy bien la textura sintética y no la
del metal real. Sobremuestrear los reales y afinar al final mitigó la brecha
—de 0.664 a 0.597— pero no la cerró.

El alfabeto reducido, que parecía la gran ventaja, no compensa la falta de
datos: el problema no es cuántos símbolos hay que distinguir, sino cuánta
variedad de aspecto real ha visto el modelo de cada uno.

## Qué haría falta

1. **Afinar un reconocedor preentrenado en lugar de entrenar desde cero.** Es
   la corrección principal. Los modelos PP-OCR que usa RapidOCR ya han visto
   millones de imágenes de texto real; sólo hay que adaptarlos a este dominio,
   no enseñarles a leer. Con 318 recortes eso es viable, mientras que entrenar
   de cero no lo es.
2. **Más recortes reales.** Entre 2.000 y 5.000 darían margen incluso desde
   cero. Se consiguen con más fotografías cercanas por cilindro: cada toma
   `3.Cerca` produce un recorte utilizable, y ahora hay una por cilindro.
3. **Sintéticos más fieles.** Los actuales imitan la geometría del relieve
   pero no la textura del metal oxidado. Partir de fondos recortados de
   fotografías reales, en lugar de generarlos, acercaría los dominios.

## Qué queda aprovechable

Todo el andamiaje, que es la parte lenta:

- `training/build_ocr_dataset.py` — extrae recortes etiquetados de lo ya anotado
- `training/synth_stamped.py` — generador de troquelado sintético
- `training/train_recognizer.py` — entrenamiento CRNN+CTC con afinado en dos fases
- `app/ocr/charset.py` — alfabeto cerrado y codificación CTC
- `app/ocr/serial_recognizer.py` — inferencia ONNX, lista para cuando haya un
  modelo que merezca la pena

Cambiar a la estrategia de afinado sobre un modelo preentrenado reutiliza los
tres primeros sin tocarlos.

## Estado en el servicio

`AI_SERIAL_RECOGNIZER_PATH` viene sin definir, así que el componente está
**inactivo** y el pipeline usa el OCR general. Activarlo con el modelo actual
empeoraría el sistema.

El componente incluye una salvaguarda que conviene conservar: si el alfabeto
guardado junto a los pesos no coincide con el del servicio, se desactiva solo.
Un desajuste ahí no produce ningún error, sólo caracteres equivocados en
silencio.
