#!/usr/bin/env python3
"""Exportación del detector entrenado a ONNX para inferencia en producción.

Separar entrenamiento de inferencia es lo que permite que el contenedor de
producción no incluya PyTorch ni ultralytics: sólo onnxruntime. La diferencia
son más de 1 GB de imagen y un arranque notablemente más rápido.

Uso:
    python training/export_onnx.py --weights models/runs/detector/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.vision.base import DetectionClass  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/detector.onnx"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--simplify", action="store_true", default=True)
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"ERROR: no existe {args.weights}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: falta ultralytics. Instale: pip install -r requirements/train.txt")
        return 1

    model = YOLO(str(args.weights))
    print(f"Exportando {args.weights} a ONNX (imgsz={args.imgsz}, opset={args.opset})...")

    exported = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                            simplify=args.simplify, dynamic=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(exported), args.output)

    # Los nombres de clase se guardan junto al modelo: el servicio los lee de
    # aquí en lugar de asumir un orden.
    names = getattr(model, "names", None)
    classes = (
        [names[i] for i in sorted(names)] if isinstance(names, dict)
        else list(DetectionClass.ALL)
    )
    args.output.with_suffix(".classes.json").write_text(
        json.dumps(classes, ensure_ascii=False), encoding="utf-8"
    )

    size_mb = args.output.stat().st_size / 1e6
    print(f"\nModelo exportado: {args.output} ({size_mb:.1f} MB)")
    print(f"Clases: {classes}")
    print("\nPara activarlo en el servicio:")
    print("  AI_DETECTOR_BACKEND=onnx")
    print(f"  AI_DETECTOR_MODEL_PATH={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
