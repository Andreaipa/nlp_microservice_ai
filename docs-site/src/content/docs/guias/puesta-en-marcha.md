---
title: Puesta en marcha
description: Arrancar el servicio, integrarlo y reproducir los resultados.
---

Guía para dejar el microservicio funcionando, integrarlo con el backend y
reproducir los resultados.

## 1. Arranque en local

```bash
make install          # dependencias de inferencia + OCR
cp .env.example .env
make run
```

Comprobación:

```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/version
```

Documentación interactiva: http://localhost:8000/docs

## 2. Arranque con Docker

```bash
docker compose up -d --build
curl http://localhost:8000/api/v1/health
```

El modelo y el catálogo se montan como volumen, así que sustituir un modelo
entrenado no obliga a reconstruir la imagen.

## 3. Analizar una fotografía

```bash
curl -X POST http://localhost:8000/api/v1/cylinders/analyze \
     -H "X-Request-ID: prueba-001" \
     -F "image=@datasets/raw/Cilindro_002/3.Cerca/IMG_20260903_150814.jpg"
```

## 4. Qué debe hacer la aplicación móvil

Esto no es un detalle de interfaz: **es lo que decide si el sistema sirve**.

| Condición de la foto          | Acierto |
| ----------------------------- | ------- |
| Cerca del collarín            | **60%** |
| De frente, lejos, en penumbra | 13-20%  |

La aplicación debe **guiar al operario a encuadrar el collarín de cerca**, con
un marco en pantalla, y no permitir el disparo hasta que el cilindro ocupe
buena parte del encuadre. Sin eso el sistema rinde al 22% y la experiencia es
mala; con ello, al 60%.

Recomendaciones adicionales para la captura:

- no usar flash de frente: el reflejo borra el troquelado;
- si hay poca luz, iluminar **desde un lado**, en ángulo rasante, para que el
  relieve proyecte sombra;
- enfocar el collarín, no el cuerpo del cilindro.

## 5. Flujo de trabajo previsto

```
La app captura  ->  Backend  ->  POST /api/v1/cylinders/analyze
                                        |
                        requires_manual_confirmation = false
                                        |-> el backend registra el movimiento
                                        |
                        requires_manual_confirmation = true
                                        |-> la app muestra el número propuesto
                                            y los candidatos; el operario
                                            confirma o corrige
```

El operario nunca teclea a ciegas: o confirma lo propuesto, o escribe un
número que el backend puede validar contra el inventario antes de aceptarlo.

## 6. Integración con el backend

El contrato está en [INTEGRACION_BACKEND.md](/sistema/integracion/). Lo
mínimo que debe hacer el backend:

1. Enviar la fotografía a `/api/v1/cylinders/analyze`.
2. Mirar `requires_manual_confirmation`.
3. Si es `true`, usar `review_reasons` para dar un mensaje útil y
   `match.candidates` para ofrecer alternativas.
4. Guardar `model_version` junto al movimiento: permite saber después qué
   versión produjo cada lectura.

Para que el microservicio consulte el maestro del backend:

```
AI_BACKEND_ENABLED=true
AI_BACKEND_BASE_URL=http://backend:3000/api
AI_BACKEND_API_KEY=...
```

El backend debe exponer `GET /cylinders/{numero_serie}`. La consulta es
best-effort: si no responde, el análisis se devuelve igualmente con lo leído
de la imagen.

## 7. Reproducir los resultados

```bash
make import        # importa el dataset normalizando nombres
make audit         # audita: conteos, duplicados, calidad
make catalog       # catálogo del inventario y verdad de campo
make split         # reparto por identidad de cilindro
make preannotate   # propone anotaciones (revisar antes de entrenar)
make yolo-dataset  # ensambla la estructura para ultralytics
make train         # entrena el detector
make export        # exporta a ONNX
make evaluate      # mide el sistema de extremo a extremo
make calibrate     # calibra los umbrales de confianza
```

Cada paso escribe su informe en `models/`, de modo que las cifras del README
se pueden rehacer y contrastar.

## 8. Ajuste según el equipo

Ver [RENDIMIENTO.md](/sistema/rendimiento/). Para un equipo modesto sin GPU:

```
AI_OCR_BACKEND=rapid
AI_OCR_PREPROCESS_VARIANTS=2
AI_OCR_THREADS=2
AI_MAX_IMAGE_SIDE=1600
```

## 9. Qué vigilar en operación

- `GET /api/v1/ready` devuelve 503 si falta el modelo o los umbrales no están
  calibrados: sirve como `readinessProbe`.
- `datasets/audit/` guarda un registro por inferencia y la imagen de los casos
  que requirieron confirmación. Revisarlos periódicamente indica si el sistema
  se está degradando y alimenta el siguiente reentrenamiento.
- La tasa de `requires_manual_confirmation` es el indicador operativo más
  útil: si sube, algo ha cambiado en las condiciones de captura.