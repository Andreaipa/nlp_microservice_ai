# AI Cylinder Service

Microservicio de visión artificial que identifica cilindros industriales a
partir de una fotografía del troquelado de su collarín.

Detecta las regiones relevantes, lee el texto grabado en el metal, lo
normaliza y resuelve el número de serie contra el inventario conocido. Devuelve
información estructurada con el nivel de confianza y la procedencia de cada
dato, para que el backend principal decida qué hacer con ella.

```
imagen ──► detección ──► recorte ──► realce ──► OCR ──► parser
                                                          │
                                       catálogo ◄──────────┤
                                       backend  ◄──────────┘
                                                          │
                                                       JSON
```

## Estado

| Componente                    | Estado                                                             |
| ----------------------------- | ------------------------------------------------------------------ |
| Microservicio, API y pipeline | Funcional y probado                                                |
| Contrato JSON e integración   | Definido y verificado — el backend puede integrarse ya             |
| Docker                        | Imagen construida y verificada en ejecución                        |
| Dataset                       | 100 cilindros, 599 fotografías (verificado contra el ZIP original) |
| Catálogo del inventario       | 99 seriales                                                        |
| Motor OCR                     | RapidOCR sobre ONNX Runtime                                        |
| Detector YOLO                 | Entrenado, **insuficiente todavía**                                |
| Umbrales calibrados           | No — todo se marca para confirmación manual                        |

### Métricas reales

Detector, sobre cilindros no vistos en entrenamiento:

| Clase          | mAP50 |
| -------------- | ----- |
| Global         | 0.301 |
| `marking_area` | 0.447 |
| `serial_text`  | 0.155 |

Sistema completo, sobre las 90 imágenes de test (15 cilindros nunca vistos):

```
identificación correcta ........ 31.1%   (60% en tomas cercanas)
correcto o entre las 4 opciones  40.0%
errores silenciosos ............  0.0%
requiere confirmación manual ... 100.0%
CER medio del serial ...........  0.254
latencia mediana / p95 ......... 857 ms / 1682 ms
```

| Condición                             | Acierto |
| ------------------------------------- | ------- |
| **3.Cerca**                           | **60%** |
| 6.Poca luz                            | 33%     |
| 1.Frontal · 2.Diagonal · 5.Luz normal | 27%     |
| 4.Lejos                               | 13%     |

**Cómo leer estas cifras.** El sistema no identifica de forma autónoma: sirve
como asistente con confirmación humana, que es el modo en que está
configurado. Ninguna lectura errónea llegaría a la base de datos, porque la
tasa de errores silenciosos es cero.

La diferencia entre el 60% de las tomas cercanas y el 13% de las lejanas es el
dato más accionable del proyecto: **que la aplicación móvil obligue a
acercarse al collarín multiplica el acierto**, sin tocar una línea del modelo.
En las fotos frontales, de lejos o con poca luz, el troquelado del hombro ni
siquiera es visible: ningún algoritmo puede leerlo.

### Lo que enseñaron las pruebas con teléfono

Las primeras pruebas con la app sobre cilindros reales (serie `K57…`, el 31%
del inventario) mostraron que buena parte del fallo no era del OCR sino de
cómo se interpretaba lo leído:

- El troquelado lleva un espacio en medio ("K5738 060") y el OCR lo devolvía
  en dos trozos que nadie volvía a unir.
- Se proponía el fragmento leído con más confianza, que solía ser otra marca
  ("PW200", la presión de trabajo), en lugar del que existía en el inventario.
- Con un umbral de error fijo, "300" se daba por el cilindro "804".
- De las tres variantes de realce, sólo contaba de verdad la "mejor": las
  lecturas de las otras se descartaban.

La corrección busca cada serial del inventario **dentro** de todo lo leído,
une los trozos de una misma línea, tolera errores en proporción a la longitud
del serial y descarta las marcas que no son seriales (presiones, fechas,
número ONU). Con ella, ofrecer alternativas sí sirve:

|                                       | Antes | Ahora   |
| ------------------------------------- | ----- | ------- |
| El correcto es el propuesto           | 22%   | **31%** |
| El correcto está entre las 4 opciones | 24%   | **40%** |
| Propone un número equivocado          | 53%   | **27%** |

Con menos propuestas equivocadas, cuando el sistema no está seguro deja el
campo vacío y pide otra foto en lugar de sugerir un número que confunde.

### Qué movió y qué no

| Cambio                                  | Efecto                                    |
| --------------------------------------- | ----------------------------------------- |
| RapidOCR en lugar de PaddleOCR          | latencia 8,8 s → 0,67 s                   |
| Corregir el filtro de posición vertical | anotaciones 92 → 326                      |
| Detector entrenado (72 → 247 imágenes)  | mAP50 0.147 → 0.301                       |
| Emparejamiento por fragmento            | acierto 18,9% → 22,2%; cercanas 47% → 60% |
| Buscar el serial dentro de lo leído     | acierto 22,2% → 31,1%; CER 0.537 → 0.254  |

Entrenar el detector mejoró la latencia y la calidad de los recortes, pero no
la precisión de identificación. Lo que sí la movió fue aprovechar mejor lo que
el OCR ya leía, contrastándolo con el inventario cerrado.

## Qué incluye el repositorio y qué no

|                                                | En el repositorio | Cómo obtenerlo                    |
| ---------------------------------------------- | ----------------- | --------------------------------- |
| Código, pruebas y documentación                | Sí                | —                                 |
| Anotaciones de caja (326)                      | Sí                | cuestan horas de cómputo          |
| Catálogo del inventario y verdad de campo      | Sí                | `make catalog`                    |
| Informes de métricas                           | Sí                | `make evaluate`, `make calibrate` |
| **Fotografías originales** (1,5 GB)            | No                | `make import SOURCE=...`          |
| **Pesos del detector**                         | No                | `make train && make export`       |
| Derivados (dataset YOLO, recortes, sintéticos) | No                | se regeneran con un comando       |

Sin los pesos el servicio arranca igualmente: cae al detector heurístico y lo
declara en `/version` y en `/ready`. Para reconstruirlo todo desde cero basta
con el flujo de `make` descrito más abajo.

## Arranque rápido

```bash
make install
cp .env.example .env
make run
```

Documentación interactiva en http://localhost:8000/docs

Con Docker:

```bash
docker compose up -d --build
curl http://localhost:8000/api/v1/health
```

## Uso

```bash
curl -X POST http://localhost:8000/api/v1/cylinders/analyze \
     -F "image=@cilindro.jpg"
```

```json
{
  "success": true,
  "requires_manual_confirmation": false,
  "data": {
    "numero_serie": {
      "value": "19S206055",
      "confidence": 0.94,
      "source": "ai"
    },
    "marca": { "value": "JP5", "confidence": 0.88, "source": "ai" },
    "tipo_gas": { "value": "CO2", "confidence": null, "source": "backend" },
    "m3": { "value": null, "confidence": null, "source": "unknown" }
  },
  "match": { "status": "matched", "resolved_serial": "19S206055" }
}
```

Cada campo lleva su procedencia: `ai` si se leyó de la imagen, `catalog` o
`backend` si vino de un maestro, `unknown` si no se pudo determinar. **El
servicio no rellena huecos por verosimilitud.**

## Endpoints

| Método | Ruta                        | Para qué                               |
| ------ | --------------------------- | -------------------------------------- |
| POST   | `/api/v1/cylinders/analyze` | Analiza una fotografía                 |
| GET    | `/api/v1/health`            | Vivacidad (`livenessProbe`)            |
| GET    | `/api/v1/ready`             | Disponibilidad real (`readinessProbe`) |
| GET    | `/api/v1/version`           | Versiones de servicio y modelo         |
| GET    | `/docs`                     | Documentación OpenAPI                  |

## Estructura

```
app/
├── api/            endpoints y dependencias
├── core/           configuración, logging, errores
├── schemas/        contrato tipado (Pydantic)
├── vision/         detección: onnx | ultralytics | heurístico
├── ocr/            motores OCR: paddle | null
├── preprocessing/  carga, calidad y realce para metal troquelado
├── parsing/        normalización, campos y número de serie
├── services/       pipeline, catálogo, cliente del backend
└── observability/  auditoría de inferencias

training/           auditoría de dataset, split, entrenamiento,
                    evaluación y calibración de umbrales
tests/              78 pruebas
docs/               diagnóstico, arquitectura, protocolos
```

### Desviaciones respecto a la estructura inicialmente propuesta

- **`training/` en lugar de `scripts/`**: separa la pila de entrenamiento
  (PyTorch, ultralytics, ~1,5 GB) de la de inferencia. El contenedor de
  producción no la instala, y `requirements/` está dividido en consecuencia
  (`base`, `ocr`, `train`, `dev`).
- **`app/parsing/` como módulo propio**: normalizar texto de OCR resultó ser
  una parte sustancial del problema (unidades mezcladas, dos significados para
  el campo PH, códigos de país en tres formatos), no un detalle del pipeline.
- **Sin `notebooks/`**: cada paso del flujo es un script con argumentos,
  reproducible y ejecutable en CI. Un cuaderno con estado oculto no ofrece esa
  garantía. Para explorar puntualmente sigue estando `python -i`.
- **`app/observability/`**: la auditoría de inferencias es un requisito
  operativo (poder reconstruir por qué el sistema dijo lo que dijo), no parte
  de la lógica de negocio.

## Flujo de trabajo del modelo

```bash
# 1. Importar el dataset normalizando nombres de carpeta
python training/import_dataset.py --source ~/Downloads/"Cilindros Electrametal"

# 2. Auditar antes de anotar nada
python training/audit_dataset.py --root datasets/raw

# 3. Construir el catálogo del inventario y la verdad de campo
python training/build_catalog.py --root datasets/raw

# 4. Separar por identidad de cilindro (nunca aleatoriamente)
python training/split_by_cylinder.py --root datasets/raw --out datasets/processed

# 5. Proponer anotaciones automáticamente, y REVISARLAS
python training/preannotate.py --root datasets/raw

# 6. Ensamblar la estructura que espera ultralytics
python training/prepare_yolo_dataset.py

# 7. Entrenar
python training/train_detector.py --data datasets/yolo/data.yaml

# 7b. Ampliar la cobertura de anotación con el modelo recién entrenado,
#     revisar sus propuestas y reentrenar (repetible)
python training/pseudo_label.py --weights models/runs/detector/weights/best.pt \
    --only-split datasets/processed/train.txt

# 8. Evaluar el detector sobre cilindros no vistos
python training/evaluate_detector.py --weights models/runs/detector/weights/best.pt \
    --data datasets/yolo/data.yaml --split test

# 9. Exportar a ONNX para producción
python training/export_onnx.py --weights models/runs/detector/weights/best.pt

# 10. Medir el sistema completo: la métrica que de verdad importa
python training/evaluate_pipeline.py

# 11. Calibrar los umbrales con datos, no a ojo
python training/calibrate_thresholds.py --max-silent-error-rate 0.01
```

## Configuración

Todo por variables de entorno con prefijo `AI_`. Ver [.env.example](.env.example).

Las dos que más importan:

- `AI_DETECTOR_BACKEND`: `heuristic` (sin modelo), `onnx` (producción),
  `ultralytics` (desarrollo).
- `AI_THRESHOLDS_CALIBRATED`: mientras sea `false`, **todas** las respuestas se
  marcan para confirmación manual. Es intencionado. Ver
  [docs/THRESHOLDS.md](docs/THRESHOLDS.md).

## Rendimiento

Perfilado sobre una fotografía real en CPU:

| Etapa                       | Tiempo                                      |
| --------------------------- | ------------------------------------------- |
| Decodificación y reescalado | 32 ms                                       |
| Detección YOLO (ONNX)       | 15 ms                                       |
| Realce para OCR             | 9 ms                                        |
| OCR, por pasada             | **127 ms** (RapidOCR) · 1224 ms (PaddleOCR) |

El OCR domina el coste; el detector es irrelevante a este respecto. Por eso el
motor por omisión es **RapidOCR sobre ONNX Runtime**: diez veces más rápido
que PaddleOCR en CPU, más acertado en las muestras probadas y sin añadir
dependencias, ya que onnxruntime ya se usa para el detector.

No se asume GPU y no hace falta. Para equipos modestos hay una configuración
recomendada en [docs/RENDIMIENTO.md](docs/RENDIMIENTO.md).

## Documentación

Hay tres formas de consultarla, y las tres salen de la misma fuente:

|               | Cómo                                    | Para qué                                      |
| ------------- | --------------------------------------- | --------------------------------------------- |
| **Sitio web** | `make docs-dev` → http://localhost:4321 | Navegable, con búsqueda. Lo más cómodo.       |
| **Swagger**   | `make run` → http://localhost:8000/docs | Probar la API desde el navegador.             |
| **Markdown**  | `docs/`                                 | Legible en el repositorio, sin compilar nada. |

El sitio se construye con [Starlight](https://starlight.astro.build/es/) y su
sección de API se genera desde `docs/openapi.json`, así que **no puede
desviarse del contrato real del servicio**. `make docs-sync` mantiene las
páginas alineadas con `docs/`.

### Índice

- [Puesta en marcha](docs/PUESTA_EN_MARCHA.md) — arrancar, integrar, reproducir
- [Arquitectura](docs/ARQUITECTURA.md) — decisiones de diseño y por qué
- [Rendimiento](docs/RENDIMIENTO.md) — perfilado, elección de OCR, ajuste por equipo
- [Umbrales](docs/THRESHOLDS.md) — cómo se calibran
- [Integración con el backend](docs/INTEGRACION_BACKEND.md) — contrato y errores
- [El dataset](docs/DATASET.md) — contenido real, hallazgos e inconsistencias
- [Protocolo de captura](docs/PROTOCOLO_CAPTURA.md) — cómo fotografiar
- [Estrategia de anotación](docs/ANOTACION.md) — clases y criterios
- [Reconocedor especializado](docs/RECONOCEDOR.md) — un experimento que falló
- [Diagnóstico inicial](docs/DIAGNOSTICO.md) — qué se encontró al empezar

## Pruebas

```bash
make test    # 198 pruebas
make lint    # estilo y tipos
```

## Documentación en local

```bash
make docs-install   # sólo la primera vez
make docs-dev       # http://localhost:4321
```
