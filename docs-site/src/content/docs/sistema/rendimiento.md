---
title: Rendimiento y tecnología
description: Dónde se va el tiempo y cómo ajustar el servicio a cada equipo.
---

El despliegue previsto no tiene GPU y las pruebas se harán en un equipo
Windows de gama media. Ese es el criterio con el que se han tomado las
decisiones de abajo: no la cifra máxima alcanzable, sino que el servicio sea
usable en una máquina modesta.

## Dónde se va el tiempo

Perfilado sobre una fotografía real (`Cilindro_002/3.Cerca`, 2296×4080
reducida a 1153×2048), en CPU:

| Etapa                           | Tiempo            |
| ------------------------------- | ----------------- |
| Decodificación y reescalado     | 32 ms             |
| **Detección YOLO (ONNX)**       | **15 ms**         |
| Realce para OCR                 | 9 ms              |
| **OCR (PaddleOCR, por pasada)** | **900 – 2800 ms** |

La conclusión es inequívoca: **el detector no es el problema, el OCR es el
99% del coste**. Optimizar YOLO no habría servido de nada.

## PaddleOCR frente a RapidOCR

Ambos ejecutan los mismos modelos PP-OCR; la diferencia es el motor de
inferencia. Medido sobre un recorte real del troquelado:

| Motor                       | Tiempo     | Lectura del serial   | Dependencias              |
| --------------------------- | ---------- | -------------------- | ------------------------- |
| **RapidOCR** (ONNX Runtime) | **127 ms** | `19s206055` ✅       | onnxruntime (ya presente) |
| PaddleOCR (PaddlePaddle)    | 1224 ms    | `198206055` ❌ (S→8) | paddlepaddle, ~600 MB     |

Diez veces más rápido, más acertado en esa muestra y sin añadir dependencias:
el servicio ya usa onnxruntime para el detector. Por eso **`rapid` es el motor
por omisión**.

`paddle` se mantiene disponible (`AI_OCR_BACKEND=paddle`) porque reconoce más
texto secundario en algunas fotografías, lo que ayuda a rellenar campos
distintos del número de serie. Cambiar de uno a otro es una variable de
entorno: la abstracción `OcrEngine` aísla esa decisión.

Este cambio no es sólo cuestión de latencia. Con el OCR diez veces más barato,
la pre-anotación del dataset pasó de ~100 minutos a ~10, lo que permitió
añadir **búsqueda multi-escala** (buscar también sobre la imagen ampliada al
doble, para el troquelado diminuto de las tomas lejanas). Eso subió la
cobertura de anotación del 23% al 30%. Una decisión de rendimiento acabó
mejorando la calidad del modelo.

## Configuración recomendada por equipo

### Equipo modesto (Windows de gama media, sin GPU)

```
AI_OCR_BACKEND=rapid
AI_OCR_PREPROCESS_VARIANTS=2     # cada variante es una pasada de OCR
AI_OCR_THREADS=2                 # evita que una petición acapare la máquina
AI_MAX_IMAGE_SIDE=1600           # menos píxeles que recorrer
AI_DETECTOR_BACKEND=onnx
AI_DETECTOR_INPUT_SIZE=768
```

Con esto una fotografía debería resolverse en menos de un segundo. La pérdida
frente a la configuración completa es de cobertura del consenso, no de
corrección: las lecturas siguen resolviéndose contra el catálogo.

### Servidor con margen

```
AI_OCR_BACKEND=rapid
AI_OCR_PREPROCESS_VARIANTS=3
AI_OCR_THREADS=0                 # sin límite
AI_MAX_IMAGE_SIDE=2048
AI_DETECTOR_INPUT_SIZE=960
```

### Qué se beneficiaría de GPU

Sólo el reconocimiento OCR y, marginalmente, el detector. El preprocesado y el
parser son irrelevantes a este respecto. Con RapidOCR en CPU el servicio ya es
utilizable, así que **la GPU no es un requisito** sino una mejora si aparece.

## Palancas de ajuste, por impacto

1. **`AI_OCR_PREPROCESS_VARIANTS`** — cada variante es una pasada completa de
   OCR. Es la palanca más directa: bajar de 3 a 1 recorta el tiempo a un
   tercio, a cambio de perder el consenso entre variantes, que es lo que
   protege de lecturas erróneas con alta confianza.
2. **`AI_MAX_IMAGE_SIDE`** — el coste del OCR crece con el área. Bajar de 2048
   a 1600 ahorra un 40%, pero el troquelado es texto de bajo contraste y
   pierde legibilidad enseguida: no conviene bajar de 1400.
3. **`AI_DETECTOR_INPUT_SIZE`** — apenas afecta (15 ms), no merece tocarlo.
4. **Número de trabajadores de uvicorn** — el OCR mantiene su modelo en
   memoria por proceso. Más trabajadores multiplican la memoria; en un equipo
   modesto conviene dejar uno.

## Tamaño del despliegue

|                     | Con PaddleOCR | Con RapidOCR |
| ------------------- | ------------- | ------------ |
| Imagen Docker       | ~1,6 GB       | ~900 MB      |
| Memoria por proceso | ~1,5 GB       | ~400 MB      |

En un equipo de gama media esa diferencia decide si se puede ejecutar el
servicio junto a otras cosas o no.