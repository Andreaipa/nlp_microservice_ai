---
title: Arquitectura
description: Pipeline, decisiones de diseño y por qué se tomaron.
---

## Frontera de responsabilidades

El microservicio responde a una sola pregunta: **"¿qué información puedo
identificar en esta imagen?"**. No decide qué hacer con ella.

```
App móvil  ──►  Backend principal  ──►  Microservicio IA
                       │                        │
                       │                   detección
                       │                   OCR
                       │                   normalización
                       │                   resolución contra catálogo
                       │                        │
                       ◄────────── JSON ────────┘
                       │
                  Base de datos
                  Trazabilidad
```

El backend es dueño del dato maestro y de la lógica de negocio. La IA informa
de lo que observó y con cuánta confianza; si no está segura, lo dice en lugar
de resolver por su cuenta.

## Pipeline

```
imagen (multipart)
   │
   ├─ decodificación y límite de tamaño ......... app/preprocessing/image_io.py
   ├─ medición de calidad (nitidez, luz) ........ app/preprocessing/image_io.py
   │
   ├─ DETECCIÓN de regiones .................... app/vision/
   │     onnx | ultralytics | heuristic
   │
   ├─ recorte por región ....................... app/preprocessing/image_io.py
   ├─ realce para metal troquelado ............. app/preprocessing/enhance.py
   │     CLAHE · relieve · enfoque · black-hat · original
   │
   ├─ OCR sobre cada variante .................. app/ocr/
   │
   ├─ consenso entre variantes (serial) ........ app/parsing/serial.py
   ├─ extracción de campos ..................... app/parsing/fields.py
   ├─ normalización ............................ app/parsing/normalizers.py
   │
   ├─ resolución contra inventario ............. app/services/catalog.py
   ├─ enriquecimiento desde backend ............ app/services/backend_client.py
   │
   └─ respuesta JSON + auditoría ............... app/services/pipeline.py
```

## Decisiones de diseño

### 1. El serial se resuelve contra un catálogo cerrado

El inventario tiene 100 cilindros. No hay que reconocer una cadena arbitraria,
sino decidir cuál de 100 seriales conocidos aparece en la foto.

La consecuencia es grande: un OCR que falle uno o dos caracteres sigue
identificando bien el cilindro, porque la distancia ponderada al serial real
es mucho menor que a cualquier otro del inventario. Comprobado sobre el
cilindro 002: `195206055`, `19s2O6O55` y `I9S2060S5` resuelven los tres a
`19S206055`.

Sin esta etapa, un solo carácter mal leído produciría un identificador
inexistente.

La salvaguarda es que si dos candidatos quedan igual de cerca, el resultado se
marca ambiguo y decide una persona.

### 2. Varias variantes de realce y consenso, no la más confiada

Qué transformación hace legible un troquelado depende de la luz de cada foto y
no se puede saber de antemano. Se prueban varias y se combinan por acuerdo.

Esto no es una precaución teórica. Al pasar PaddleOCR por las cinco variantes
sobre una placa de prueba:

| variante | lectura         | confianza |
| -------- | --------------- | --------- |
| clahe    | `P5 19S206055`  | 0.99      |
| relief   | `P59S206055`    | **0.99**  |
| sharpen  | `JP5 19S206055` | 0.96      |
| blackhat | `5.9S206055`    | 0.96      |
| identity | `JP5 19S206055` | 0.97      |

La lectura **más confiada era incorrecta**. Quedarse con el máximo elige mal;
el acuerdo entre transformaciones independientes lleva a `19S206055`, que es
el correcto.

### 3. El detector se abstrae detrás de una interfaz

`app/vision/base.py` define el contrato. Hay tres implementaciones: ONNX para
producción, ultralytics para depurar y una heurística de OpenCV que permite
que el servicio funcione de extremo a extremo **antes** de que exista un
modelo entrenado, de modo que backend y app móvil puedan integrarse ya.

El modo heurístico se declara en `/version`, en `/ready` y en cada respuesta:
nunca se hace pasar por un modelo entrenado.

### 4. Entrenamiento e inferencia separados

El contenedor de producción sólo lleva onnxruntime. PyTorch y ultralytics
viven en `requirements/train.txt` y no entran en la imagen: más de 1 GB menos
y un arranque bastante más rápido.

### 5. Cada dato lleva su procedencia

Todo campo se devuelve como `{value, confidence, source}` con `source` en
`ai | catalog | backend | manual | unknown`. Permite saber después si un dato
lo leyó la cámara o lo puso la base de datos, que es información necesaria
para depurar una discrepancia.

Un campo que no se pudo determinar se queda en `null` con `source: unknown`.
El servicio no rellena huecos por verosimilitud.

### 6. La confianza del OCR no basta: también se comprueba la geometría

Al revisar las lecturas sobre el dataset real apareció un caso que obliga a
desconfiar de la confianza declarada por el motor OCR. Sobre el cuerpo de un
cilindro cubierto de pintura descascarada, PaddleOCR devolvió una cadena muy
parecida a un número de serie **con 0.96 de confianza**, en una zona donde al
ampliar la imagen no hay texto alguno.

Lo que distingue esa lectura de una buena no es la confianza, sino la forma:
la caja era vertical (41 × 139 px), mientras que un troquelado se lee en
horizontal. El pipeline descarta las líneas cuya caja no tiene forma de línea
de texto, lo que elimina ese tipo de falso positivo sin perder lecturas
legítimas.

Es también la razón por la que la resolución contra catálogo no se puede
sustituir por "aceptar si la confianza es alta".

### 7. Los umbrales se calibran, no se eligen

Mientras `AI_THRESHOLDS_CALIBRATED=false`, toda respuesta se marca para
confirmación manual. Un umbral elegido a ojo produce errores silenciosos, que
en trazabilidad es el peor fallo: registra el movimiento en el cilindro
equivocado sin avisar a nadie. Ver [THRESHOLDS.md](/sistema/umbrales/).

### 8. Los defectos del inventario se declaran, no se disimulan

El maestro real contiene un serial repetido en dos cilindros (`K5738166`) y
uno de tres caracteres (`804`). Ninguna lectura, por buena que sea, permite
identificar con certeza en esos casos. El servicio lo detecta al cargar el
catálogo y fuerza confirmación manual, aunque los umbrales estén calibrados y
el OCR haya acertado.

## Clases del detector

Tres, deliberadamente pocas:

| Clase          | Para qué                                       |
| -------------- | ---------------------------------------------- |
| `cylinder`     | Encuadre y descarte de fondo                   |
| `marking_area` | Collarín troquelado: contiene todos los campos |
| `serial_text`  | Línea concreta del número de serie             |

Cada clase adicional multiplica el trabajo de anotación y el riesgo de
confusión entre categorías parecidas, mientras que el OCR y el parser ya
separan los campos dentro de una región. Regla aplicada: una clase existe sólo
si implica un recorte distinto o un tratamiento distinto aguas abajo.

Esta lista es una propuesta razonada, no una conclusión: debe revisarse al
mirar las primeras fotografías reales. Si el troquelado aparece siempre en un
bloque compacto, `serial_text` podría sobrar; si hay cilindros con etiqueta
adhesiva además del troquelado, podría hacer falta una clase para ella.