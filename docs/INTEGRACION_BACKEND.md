# Integración con el backend principal

## Reparto de responsabilidades

|                       | Microservicio IA             | Backend principal     |
| --------------------- | ---------------------------- | --------------------- |
| Pregunta que responde | "¿Qué veo en esta imagen?"   | "¿Qué hago con esto?" |
| Conoce el inventario  | Sólo el catálogo de seriales | Sí, es su dueño       |
| Decide movimientos    | No                           | Sí                    |
| Guarda trazabilidad   | No                           | Sí                    |
| Estado                | Sin estado                   | Con estado            |

El microservicio nunca decide si un cilindro entra, sale o se despacha. Informa
de lo que leyó y de su confianza.

## Flujo previsto

```
App móvil ──foto──► Backend ──multipart──► Microservicio IA
                       │                          │
                       │◄────── JSON ─────────────┘
                       │
                       ├─ requires_manual_confirmation = false
                       │     └─ registra el movimiento
                       │
                       └─ requires_manual_confirmation = true
                             └─ muestra al operario los candidatos
                                y pide confirmación
```

## Endpoint principal

```
POST /api/v1/cylinders/analyze
Content-Type: multipart/form-data
Campo: image (JPEG, PNG o WEBP, hasta 15 MB)
```

Cabecera recomendada: `X-Request-ID`. Si la envía, el servicio la propaga a
sus logs y la devuelve; así una incidencia se puede rastrear de punta a punta.

### Respuesta

```json
{
  "success": true,
  "request_id": "8f3a1c9d2b7e4a10",
  "model_version": "detector:onnx:detector.onnx|ocr:paddle:v3:en|parser:1.0.0",
  "processing_time_ms": 842,
  "requires_manual_confirmation": false,
  "review_reasons": [],
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
  "match": {
    "status": "matched",
    "resolved_serial": "19S206055",
    "candidates": []
  },
  "backend_match": { "found": true, "queried": true }
}
```

### Cómo interpretarla

1. **`requires_manual_confirmation`** es el campo que gobierna la decisión.
   Si es `true`, no registre nada automáticamente.
2. **`review_reasons`** dice por qué, y permite dar un mensaje útil al
   operario en vez de un "no se pudo leer":

   | Motivo                        | Qué mostrar                                                        |
   | ----------------------------- | ------------------------------------------------------------------ |
   | `no_serial_detected`          | "No se distingue el número. Acérquese al collarín."                |
   | `low_ocr_confidence`          | "Lectura poco clara. Confirme el número."                          |
   | `ambiguous_catalog_match`     | Mostrar los candidatos de `match.candidates`                       |
   | `serial_not_in_catalog`       | "Ese cilindro no está en el inventario."                           |
   | `duplicate_serial_in_catalog` | "Ese número está registrado en más de un cilindro." Mostrar cuáles |
   | `serial_too_short`            | "El número grabado es demasiado corto. Verifíquelo."               |
   | `poor_image_quality`          | "Foto movida u oscura. Repítala."                                  |
   | `no_detector_model`           | Estado del sistema, no del usuario                                 |
   | `thresholds_not_calibrated`   | Estado del sistema, no del usuario                                 |

   `duplicate_serial_in_catalog` y `serial_too_short` son defectos del dato
   maestro, no de la lectura: el sistema los señala aunque el OCR haya
   acertado, y fuerzan confirmación manual **aunque los umbrales estén
   calibrados**. En el inventario actual afectan al serial `K5738166`
   (registrado en dos cilindros) y al `804` (tres caracteres).

3. **`match.candidates`** son los seriales del inventario compatibles con lo
   leído, ordenados. Ante ambigüedad, es la lista que debe ofrecerse al
   operario: elegir de tres opciones es mucho más rápido que teclear nueve
   caracteres.
4. **`source`** de cada campo indica su procedencia (`ai`, `catalog`,
   `backend`, `manual`, `unknown`). Sirve para resolver discrepancias después.
5. Un campo con `value: null` y `source: "unknown"` significa que no se pudo
   determinar. El servicio no lo rellena por verosimilitud.

## Endpoint que debe ofrecer el backend

Si activa `AI_BACKEND_ENABLED=true`, el microservicio consultará:

```
GET {AI_BACKEND_BASE_URL}/cylinders/{numero_serie}
Authorization: Bearer {AI_BACKEND_API_KEY}
```

Se espera un JSON con los campos del cilindro (directamente o envuelto en
`data`). Un `404` se interpreta como "no registrado", que es información
válida, no un error.

La consulta es best-effort: si el backend no responde, el análisis se devuelve
igualmente con lo leído de la imagen y `backend_match.error` explicando qué
pasó. Una caída del backend no deja ciega a la aplicación móvil.

## Errores

Todos comparten forma:

```json
{
  "success": false,
  "request_id": "...",
  "error": { "code": "invalid_image", "message": "...", "details": {} }
}
```

| HTTP | `code`                   | Causa                               |
| ---- | ------------------------ | ----------------------------------- |
| 413  | `image_too_large`        | Supera `AI_MAX_UPLOAD_BYTES`        |
| 415  | `unsupported_media_type` | Tipo de archivo no admitido         |
| 422  | `invalid_image`          | No se pudo decodificar              |
| 422  | `validation_error`       | Falta el campo `image`              |
| 502  | `backend_unavailable`    | El backend principal no respondió   |
| 503  | `model_not_available`    | El detector no tiene modelo cargado |

Use `code`, no el mensaje: los textos están en castellano y pueden cambiar.

## Endpoints operativos

- `GET /api/v1/health`: vivacidad del proceso. Para `livenessProbe`.
- `GET /api/v1/ready`: disponibilidad real. Devuelve **503** si falta el
  modelo entrenado o los umbrales sin calibrar. Para `readinessProbe`.
- `GET /api/v1/version`: versiones de servicio y modelo, tamaño del catálogo.
  Conviene registrar `model_version` junto a cada movimiento: permite saber
  después qué versión produjo una lectura dudosa.
