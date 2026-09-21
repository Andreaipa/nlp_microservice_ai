#!/usr/bin/env python3
"""Entrenamiento del detector YOLO de regiones del cilindro.

Requiere `pip install -r requirements/train.txt` (arrastra PyTorch; no es
necesario en el contenedor de inferencia).

El dataset debe estar en formato YOLO y descrito por un data.yaml cuyos
listados de train/val provengan de training/split_by_cylinder.py, para
garantizar que ninguna imagen del mismo cilindro cruza entre subconjuntos.

Uso:
    python training/train_detector.py --data datasets/processed/data.yaml
    python training/train_detector.py --data ... --model yolo26n.pt --epochs 150
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.vision.base import DetectionClass  # noqa: E402

# Modelo por defecto: la variante más pequeña de la familia más reciente.
# Se prioriza tamaño y latencia en CPU porque el despliegue previsto no
# garantiza GPU. `--model yolo11n.pt` sirve de alternativa estable.
DEFAULT_MODEL = "yolo26n.pt"


def pick_device() -> str:
    """Elige el acelerador disponible.

    El orden es CUDA, luego MPS (la GPU integrada de Apple Silicon, que acorta
    mucho el entrenamiento frente a la CPU) y por último CPU. El despliegue en
    producción no asume GPU: esto sólo afecta al entrenamiento.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_augmentation() -> dict:
    """Aumentos elegidos para las condiciones reales de captura.

    Justificación de cada ajuste, frente a los valores por omisión:

    * hsv_v alto y hsv_s moderado: el dataset incluye tomas con poca luz y el
      metal cambia mucho de aspecto según la iluminación. Es la variación
      dominante del problema.
    * degrees y perspective generosos: las fotos se toman a pulso desde
      ángulos distintos, y el collarín es curvo.
    * fliplr desactivado: un número de serie reflejado no existe en la
      realidad y enseñaría al modelo a aceptar texto invertido.
    * mosaic moderado y close_mosaic: ayuda al principio, pero recorta
      contexto y conviene apagarlo antes del final para que las últimas épocas
      vean cilindros completos.
    * erasing bajo: simula óxido y suciedad parcial sobre el troquelado.
    """
    return {
        "hsv_h": 0.010,
        "hsv_s": 0.50,
        "hsv_v": 0.55,
        "degrees": 12.0,
        "translate": 0.12,
        "scale": 0.45,
        "shear": 4.0,
        "perspective": 0.0008,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.5,
        "close_mosaic": 15,
        "mixup": 0.0,
        "erasing": 0.2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Ruta al data.yaml.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Pesos de partida (por defecto {DEFAULT_MODEL}).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=768,
                        help="El troquelado es texto pequeño y agradece resolución, "
                             "pero el coste crece con el cuadrado: a 960 px cada "
                             "época sobre este dataset ronda el minuto. 768 es el "
                             "equilibrio; suba a 960 si dispone de GPU dedicada.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default=None,
                        help="cpu, mps, 0, 0,1... Por omisión se detecta.")
    parser.add_argument("--project", type=Path, default=Path("models/runs"))
    parser.add_argument("--name", default="detector")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"ERROR: no existe {args.data}.")
        print("       Genere antes el dataset YOLO y la separación por cilindro:")
        print("         python training/split_by_cylinder.py --root datasets/raw")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: falta ultralytics. Instale: pip install -r requirements/train.txt")
        return 1

    device = args.device or pick_device()

    # Ultralytics antepone su propio `runs_dir` a cualquier `project` relativo,
    # de modo que "models/runs" acaba en "runs/detect/models/runs". Con una
    # ruta absoluta lo respeta tal cual y los pesos quedan donde se pidieron.
    project = args.project.resolve()

    print(f"Modelo base .......... {args.model}")
    print(f"Dispositivo .......... {device}")
    print(f"Clases ............... {', '.join(DetectionClass.ALL)}")
    print(f"Resolución ........... {args.imgsz}")
    print(f"Épocas ............... {args.epochs}")

    # Si los pesos ya están descargados en models/, se usan de ahí.
    local_weights = Path("models") / args.model
    source = str(local_weights) if local_weights.exists() else args.model

    try:
        model = YOLO(source)
    except Exception as exc:
        print(f"ERROR al cargar {args.model}: {exc}")
        print("       Alternativa estable: --model yolo11n.pt")
        return 1

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        project=str(project),
        name=args.name,
        seed=args.seed,
        exist_ok=True,
        plots=True,
        **build_augmentation(),
    )

    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else project / args.name
    best = run_dir / "weights" / "best.pt"
    print(f"\nEntrenamiento terminado. Mejores pesos: {best}")

    # El mapa de clases viaja junto al modelo para que la inferencia no
    # dependa del orden configurado en el servicio.
    classes_path = run_dir / "classes.json"
    classes_path.write_text(json.dumps(list(DetectionClass.ALL), ensure_ascii=False),
                            encoding="utf-8")

    print("\nSiguientes pasos:")
    print(f"  1. Evaluar:  python training/evaluate_detector.py --weights {best} "
          f"--data {args.data}")
    print(f"  2. Exportar: python training/export_onnx.py --weights {best}")
    print("  3. Calibrar umbrales sobre el conjunto de test antes de producción.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
