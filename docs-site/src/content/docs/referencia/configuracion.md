---
title: Configuración
description: Todas las variables de entorno del servicio, con sus valores por defecto y cuándo tocarlas.
---

Toda la configuración se resuelve desde variables de entorno con prefijo
`AI_`, o desde un archivo `.env`. No hay valores operativos repartidos por el
código: todos viven en `app/core/config.py`.

## Servicio

| Variable              | Por defecto        | Para qué                                                                 |
| --------------------- | ------------------ | ------------------------------------------------------------------------ |
| `AI_ENVIRONMENT`      | `local`            | `local`, `dev`, `staging` o `prod`. En `prod` se desactiva CORS abierto. |
| `AI_HOST` / `AI_PORT` | `0.0.0.0` / `8000` | Dirección de escucha.                                                    |
| `AI_API_PREFIX`       | `/api/v1`          | Prefijo de las rutas.                                                    |
| `AI_SERVICE_VERSION`  | `1.0.0`            | Viaja en cada respuesta.                                                 |

## Detección

| Variable                          | Por defecto            | Para qué                                                                                     |
| --------------------------------- | ---------------------- | -------------------------------------------------------------------------------------------- |
| `AI_DETECTOR_BACKEND`             | `rapid`→`onnx`         | `onnx` (producción), `ultralytics` (desarrollo) o `heuristic` (sin modelo).                  |
| `AI_DETECTOR_MODEL_PATH`          | `models/detector.onnx` | Modelo exportado.                                                                            |
| `AI_DETECTOR_INPUT_SIZE`          | `768`                  | **Se ignora si el modelo declara otro**: manda el modelo.                                    |
| `AI_DETECTOR_CONF_THRESHOLD`      | `0.25`                 | Confianza mínima de una detección.                                                           |
| `AI_DETECTOR_CROP_PADDING`        | `0.15`                 | Margen al recortar una región.                                                               |
| `AI_DETECTOR_CROP_PADDING_BOTTOM` | `0.90`                 | Margen **inferior**, mayor a propósito: el troquelado está siempre por debajo de la válvula. |

## OCR

| Variable                     | Por defecto | Para qué                                                                                                          |
| ---------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------- |
| `AI_OCR_BACKEND`             | `rapid`     | `rapid` (recomendado), `paddle` o `null`.                                                                         |
| `AI_OCR_PREPROCESS_VARIANTS` | `3`         | Variantes de realce por recorte. **Es la palanca principal de latencia**: cada una es una pasada completa de OCR. |
| `AI_OCR_THREADS`             | `0`         | `0` = automático. En equipos modestos, `2`.                                                                       |
| `AI_OCR_MIN_CONFIDENCE`      | `0.30`      | Descarta lecturas por debajo.                                                                                     |

## Umbrales de decisión

| Variable                           | Por defecto | Para qué                                                                                                                  |
| ---------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------- |
| `AI_THRESHOLDS_CALIBRATED`         | `true`      | Si es `false`, **todo** se marca para confirmación manual.                                                                |
| `AI_SERIAL_AUTO_ACCEPT_CONFIDENCE` | `1.00`      | Confianza a partir de la cual se acepta sin confirmar. En `1.00` porque la calibración concluyó que no hay umbral seguro. |
| `AI_SERIAL_REVIEW_CONFIDENCE`      | `0.70`      | Por debajo, la lectura se considera dudosa.                                                                               |
| `AI_CATALOG_FUZZY_MAX_DISTANCE`    | `2`         | Ediciones toleradas contra el inventario.                                                                                 |
| `AI_CATALOG_FUZZY_MARGIN`          | `1`         | Separación mínima entre el mejor candidato y el segundo para no declarar ambigüedad.                                      |

Ver [Umbrales de confianza](/sistema/umbrales/) antes de tocar estos valores.

## Catálogo e integración

| Variable                     | Por defecto             | Para qué                                    |
| ---------------------------- | ----------------------- | ------------------------------------------- |
| `AI_CATALOG_ENABLED`         | `true`                  | Resolución del serial contra el inventario. |
| `AI_CATALOG_PATH`            | `datasets/catalog.json` | Maestro generado por `make catalog`.        |
| `AI_BACKEND_ENABLED`         | `false`                 | Consultar el maestro del backend principal. |
| `AI_BACKEND_BASE_URL`        | —                       | Base de la API del backend.                 |
| `AI_BACKEND_API_KEY`         | —                       | Se envía como `Authorization: Bearer`.      |
| `AI_BACKEND_TIMEOUT_SECONDS` | `5.0`                   | La consulta es best-effort.                 |

## Entrada y observabilidad

| Variable                          | Por defecto      | Para qué                                                                                                            |
| --------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------- |
| `AI_MAX_UPLOAD_BYTES`             | `15728640`       | 15 MB por fotografía.                                                                                               |
| `AI_MAX_IMAGE_SIDE`               | `2048`           | Lado mayor tras reescalar. Bajarlo acelera el OCR pero el troquelado pierde legibilidad: no conviene bajar de 1400. |
| `AI_LOG_LEVEL` / `AI_LOG_JSON`    | `INFO` / `true`  | JSON por línea en producción.                                                                                       |
| `AI_AUDIT_ENABLED`                | `true`           | Registro por inferencia.                                                                                            |
| `AI_AUDIT_DIR`                    | `datasets/audit` | Dónde se guarda.                                                                                                    |
| `AI_AUDIT_STORE_IMAGES_ON_REVIEW` | `true`           | Guarda la imagen sólo cuando hubo que confirmar.                                                                    |
| `AI_AUDIT_RETENTION_DAYS`         | `30`             | Purga automática.                                                                                                   |

## Reconocedor especializado

| Variable                    | Por defecto | Para qué                                      |
| --------------------------- | ----------- | --------------------------------------------- |
| `AI_SERIAL_RECOGNIZER_PATH` | —           | Sin definir: el componente está **inactivo**. |

El modelo entrenado quedó por debajo del OCR genérico. Ver
[Reconocedor: experimento](/datos/reconocedor/).
