#!/usr/bin/env python3
"""Evaluación del detector: precisión, exhaustividad, mAP e IoU por clase.

Se evalúa sobre el subconjunto de test, formado por cilindros que el modelo no
vio durante el entrenamiento. Medir sobre validación daría una cifra optimista
porque esas imágenes guiaron la parada temprana.

Uso:
    python training/evaluate_detector.py --weights models/runs/detector/weights/best.pt \
        --data datasets/processed/data.yaml --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Umbral bajo a propósito: el mAP se calcula sobre toda la curva.")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--json", type=Path, default=Path("models/detector_metrics.json"))
    args = parser.parse_args()

    for path in (args.weights, args.data):
        if not path.exists():
            print(f"ERROR: no existe {path}")
            return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: falta ultralytics. Instale: pip install -r requirements/train.txt")
        return 1

    model = YOLO(str(args.weights))
    metrics = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz,
                        conf=args.conf, iou=args.iou, plots=True)

    summary = {
        "weights": str(args.weights),
        "split": args.split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class": {},
    }

    names = getattr(metrics, "names", {}) or {}
    for index, class_name in names.items():
        try:
            precision, recall, ap50, ap = metrics.box.class_result(index)
            summary["per_class"][class_name] = {
                "precision": float(precision), "recall": float(recall),
                "mAP50": float(ap50), "mAP50_95": float(ap),
            }
        except (IndexError, AttributeError):
            continue

    print("=" * 62)
    print(f"DETECTOR · subconjunto {args.split}")
    print("=" * 62)
    print(f"  Precisión .......... {summary['precision']:.4f}")
    print(f"  Exhaustividad ...... {summary['recall']:.4f}")
    print(f"  mAP@50 ............. {summary['mAP50']:.4f}")
    print(f"  mAP@50-95 .......... {summary['mAP50_95']:.4f}")
    if summary["per_class"]:
        print("\n  Por clase:")
        for class_name, values in summary["per_class"].items():
            print(f"    {class_name:<16} P={values['precision']:.3f} "
                  f"R={values['recall']:.3f} mAP50={values['mAP50']:.3f}")
    print("=" * 62)
    print("\nRecordatorio: el detector sólo localiza. La métrica que decide si el")
    print("sistema sirve es la identificación correcta del cilindro de extremo a")
    print("extremo: training/evaluate_pipeline.py")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMétricas guardadas en {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
