---
title: Scripts y comandos
description: Qué hace cada script del proyecto y en qué orden usarlos.
---

Todo el flujo, desde importar fotografías hasta calibrar umbrales, son
scripts con argumentos: reproducibles y ejecutables en CI. No hay cuadernos
con estado oculto.

## Flujo completo

```bash
make import        # importa el dataset normalizando nombres de carpeta
make audit         # audita: conteos, duplicados, calidad, bloqueantes
make catalog       # catálogo del inventario y verdad de campo
make split         # reparto train/val/test por identidad de cilindro
make preannotate   # propone anotaciones de caja (revisar antes de entrenar)
make yolo-dataset  # ensambla la estructura que espera ultralytics
make train         # entrena el detector
make export        # exporta el detector a ONNX
make evaluate      # mide el sistema de extremo a extremo
make calibrate     # calibra los umbrales de confianza
```

Cada paso deja su informe en `models/`.

## Preparación de datos

### `training/import_dataset.py`

Copia el dataset normalizando las variaciones de nombre que trae la captura
real (`1. Frontal ` → `1.Frontal`, `Descripción 85` → `Descripción`). Los
nombres de archivo no se tocan: son la referencia de la verdad de campo.

```bash
python training/import_dataset.py --source ~/Downloads/"Cilindros Electrametal"
python training/import_dataset.py --source ... --dry-run
```

### `training/audit_dataset.py`

Responde a las preguntas que condicionan todo lo demás: cuántos cilindros hay,
cuántas fotos por cilindro, si hay corruptas o duplicadas, cómo se reparten
las condiciones. Devuelve código de salida distinto de cero si encuentra
bloqueantes, de modo que sirve en CI.

### `training/build_catalog.py`

Lee las fichas `.docx` de cada cilindro y produce el catálogo del inventario y
la verdad de campo. Normaliza unidades mezcladas, códigos de país en varios
formatos y el campo PH, e **informa de las incidencias en lugar de
silenciarlas**.

### `training/split_by_cylinder.py`

Reparte por **identidad de cilindro**, nunca por imagen. Todas las fotos de un
cilindro van íntegras al mismo subconjunto: un reparto aleatorio metería fotos
del mismo cilindro en entrenamiento y test a la vez, y a partir de ahí todas
las métricas mienten.

## Anotación

### `training/preannotate.py`

Localiza automáticamente la línea del serial aprovechando que ya se sabe qué
número lleva cada fotografía. Ejecuta el OCR sobre la franja superior, a dos
escalas, y acepta la región cuyo texto coincide con el serial esperado.

```bash
python training/preannotate.py --root datasets/raw --workers 8 --variants 5
```

Cobertura sobre este dataset: **54%**. Las cajas son propuestas y hay que
revisarlas: una caja mal puesta enseña al detector a mirar donde no debe.

### `training/prepare_yolo_dataset.py`

Ensambla `datasets/yolo/` con la estructura que espera ultralytics, enlazando
las imágenes en lugar de copiarlas para no duplicar 1,5 GB. Si la validación
queda con muy pocas imágenes anotadas, traslada **cilindros completos** desde
entrenamiento.

### `training/pseudo_label.py`

Amplía las anotaciones usando el detector ya entrenado, que reconoce el hombro
del cilindro también donde el texto no se lee.

```bash
python training/pseudo_label.py --weights models/runs/detector/weights/best.pt \
    --only-split datasets/processed/train.txt --conf 0.70
```

:::caution
Nunca pseudo-etiquetar el conjunto de test: las cajas vendrían del propio
modelo que se va a evaluar y la métrica dejaría de medir nada.
:::

## Modelo

### `training/train_detector.py`

Entrena el detector YOLO. Detecta el acelerador disponible (CUDA, MPS o CPU) y
usa aumentos elegidos para este dominio: mucha variación de luz, sin espejado
(un serial reflejado no existe).

### `training/export_onnx.py`

Exporta a ONNX para que el contenedor de producción no necesite PyTorch. Los
nombres de clase viajan junto al modelo.

### `training/evaluate_detector.py` y `training/evaluate_pipeline.py`

El primero mide el detector (precisión, exhaustividad, mAP). El segundo mide
lo que de verdad importa: **qué porcentaje de fotografías permite identificar
correctamente el cilindro**, separando aciertos, derivaciones a revisión y
errores silenciosos.

### `training/calibrate_thresholds.py`

Recorre todos los umbrales posibles y busca el más bajo que mantiene el error
silencioso por debajo del techo indicado.

```bash
python training/calibrate_thresholds.py --max-silent-error-rate 0.01
```

## Reconocedor especializado

Estos tres scripts corresponden a un experimento que **no dio resultado**
(ver [Reconocedor](/datos/reconocedor/)), pero el andamiaje sirve para el
siguiente intento:

- `training/build_ocr_dataset.py` — extrae recortes etiquetados de lo ya anotado
- `training/synth_stamped.py` — genera recortes sintéticos de troquelado
- `training/train_recognizer.py` — entrena un CRNN+CTC en dos fases

## Utilidades

### `training/demo_report.py`

Genera evidencia visual: un montaje con las regiones detectadas, lo leído y lo
esperado, más las respuestas JSON completas. Útil para documentar el
comportamiento con ejemplos reales.

```bash
python training/demo_report.py --count 6 --condition 3.Cerca
```

### `training/export_openapi.py`

Exporta el esquema OpenAPI a `docs/openapi.json`, que es de donde esta
documentación genera la referencia de la API. Conviene reejecutarlo cuando
cambie el contrato.

## Calidad

```bash
make test    # 198 pruebas
make lint    # estilo y tipos
```
