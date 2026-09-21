# Modelos

Los pesos no se versionan: son binarios grandes que cambian con cada
entrenamiento y el repositorio guarda cómo reproducirlos.

## Reconstruir el detector

```bash
make preannotate     # anotaciones (o usar las versionadas en datasets/labels)
make yolo-dataset
make train
make export          # deja models/detector.onnx
```

## Qué sí está versionado

| Archivo | Qué es |
|---|---|
| `dataset_audit.json` | auditoría del dataset |
| `preannotation_report.json` | cobertura de la pre-anotación |
| `pipeline_metrics.json` | evaluación de extremo a extremo |
| `threshold_calibration.json` | curva completa de umbrales |
| `threshold_calibration_cerca.json` | ídem, sólo tomas cercanas |
| `demo/` | evidencia visual sobre imágenes de test |

Son la evidencia de las cifras publicadas en la documentación: permiten
contrastarlas sin volver a ejecutar nada.

## `recognizer/`

Experimento descartado. El reconocedor especializado quedó por debajo del OCR
genérico (CER 0.597 frente a 0.537). Ver `docs/RECONOCEDOR.md`.
