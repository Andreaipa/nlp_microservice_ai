---
title: Integración con el backend
description: Contrato, errores y reparto de responsabilidades.
---

## Cómo está conectado

La aplicación móvil **IdentCil** (Ionic + Angular + Capacitor) no tiene un
servidor propio: su backend es **Supabase**, con el que habla directamente. El
microservicio de IA se integra así:

```
                 ┌──────────── foto (multipart) ───────────┐
                 │                                          ▼
   App IdentCil ─┤                               Microservicio de IA
   (Ionic)       │                                          │
                 │◄──── serie propuesta + candidatos ───────┘
                 │                                          │
                 │                         lee la tabla `cilindros`
                 │                         cada 60 s (catálogo)
                 ▼                                          ▼
              Supabase ◄────────────────────────────────────┘
   (inventario, movimientos, trazabilidad)
```

- La **app** llama al microservicio con la foto y recibe la serie propuesta.
- El **operario confirma** o corrige el número en un modal: la IA nunca
  registra nada por su cuenta.
- A partir de ahí la app sigue su flujo de siempre: consulta el estado del
  cilindro y su ficha en Supabase y guarda el movimiento.
- El **microservicio lee la tabla `cilindros`** de Supabase para su catálogo,
  así que un cilindro dado de alta en la app pasa a reconocerse en las fotos
  sin reiniciar nada.

## Reparto de responsabilidades

|                       | Microservicio IA            | App + Supabase                 |
| --------------------- | --------------------------- | ------------------------------ |
| Pregunta que responde | "¿Qué número veo?"          | "¿Qué cilindro es y qué hago?" |
| Conoce el inventario  | Lo lee de Supabase          | Es su dueño                    |
| Decide movimientos    | No                          | Sí                             |
| Guarda trazabilidad   | Su auditoría de inferencias | Los movimientos                |

## Qué hace la aplicación

En las pantallas de **ingreso, salida y recojo** hay un tercer botón junto a la
búsqueda y al escáner QR: fotografiar el troquelado. El flujo, compartido por
las tres pantallas (`IdentificacionFotoService`):

1. comprueba que el servicio responde (`GET /api/v1/health`), para no hacer
   una foto que luego no se pueda analizar;
2. la primera vez, explica cómo encuadrar: es lo que más influye en el
   resultado (60% de acierto de cerca frente a 13% de lejos);
3. abre la cámara o la galería y envía la foto;
4. muestra la foto, la serie propuesta, las alternativas del inventario y el
   motivo por el que conviene revisarla;
5. el operario confirma, corrige o repite la foto.

## Trazabilidad

La app genera un `X-Request-ID` por petición. El microservicio lo registra en su
auditoría (con las detecciones, la lectura y, si hubo que revisar, la foto) y la
app lo guarda con el movimiento. Así cada ingreso se puede enlazar con la
inferencia exacta que lo produjo.

Cada cilindro de un ingreso, una salida o el reingreso de un recojo guarda en
su JSONB:

```json
{
  "numero_serie": "19S206055",
  "metodo_identificacion": "foto",
  "ia": {
    "request_id": "app-e6a40d7b56f945a9",
    "model_version": "detector:onnx:detector.onnx|ocr:rapid:onnxruntime|parser:1.0.0",
    "serie_propuesta": "19S206055",
    "confianza": 0.676,
    "corregido": false
  }
}
```

`metodo_identificacion` vale `foto`, `qr`, `lista` o `manual`, y `corregido`
indica si el operario cambió el número que propuso la IA. Con eso se puede
medir en operación real cuántas identificaciones se hacen por foto y cuántas
lecturas hubo que corregir. Va dentro de las columnas JSONB existentes, así que
**no requiere migrar el esquema**.

La tabla `recojos` no se toca: tiene columnas fijas y un campo desconocido
haría fallar el guardado. La traza del recojo va en el reingreso que el propio
recojo deja en `ingreso_cilindros`.

## Configuración

En el microservicio:

```
AI_SUPABASE_URL=https://<proyecto>.supabase.co
AI_SUPABASE_KEY=<clave anon>
AI_CORS_ALLOW_ORIGINS=http://localhost:8100,http://localhost,capacitor://localhost
```

El servicio sólo **lee** la tabla `cilindros`, así que le basta la clave
pública (anon) siempre que las políticas RLS permitan esa lectura sin sesión.
Las políticas propuestas para la app (`IdentCil-main/supabase/politicas_rls.sql`)
la permiten sólo en `cilindros`, que no contiene datos personales, y cierran el
resto de tablas. Si la lectura falla, el servicio no se detiene: sigue con el
catálogo estático y `/api/v1/ready` muestra el error de sincronización.

En la app, `src/environments/environment.ts`:

```ts
aiServiceUrl: 'http://localhost:8000';
```

Para probar desde un móvil real en la misma red, sustituya `localhost` por la
IP del equipo que ejecuta el servicio: desde el móvil, "localhost" es el
propio móvil.

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
   | `confirmation_required`       | "Confirme que el número es correcto."                              |
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