"""Pruebas del ensamblado del dataset YOLO."""
from __future__ import annotations

from pathlib import Path

import pytest

from training.prepare_yolo_dataset import flatten_name, rebalance_validation


def build_dataset(tmp_path: Path, annotated: dict[str, int]) -> tuple[Path, Path, dict]:
    """Crea una estructura mínima cilindro/condición con etiquetas."""
    root = tmp_path / "raw"
    labels = tmp_path / "labels"

    for cylinder, count in annotated.items():
        for index in range(6):
            directory = root / cylinder / f"{index + 1}.Cond"
            directory.mkdir(parents=True, exist_ok=True)
            image = directory / f"img_{index}.jpg"
            image.write_bytes(b"x")
            if index < count:
                label = labels / image.relative_to(root).with_suffix(".txt")
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    manifest = {"splits": {
        "train": {"cylinders": [c for c in annotated if c.startswith("T")]},
        "val": {"cylinders": [c for c in annotated if c.startswith("V")]},
        "test": {"cylinders": [c for c in annotated if c.startswith("X")]},
    }}
    return root, labels, manifest


def test_only_annotated_images_are_included(tmp_path: Path):
    """Una imagen sin cajas le enseñaría al detector que ahí no hay nada."""
    root, labels, manifest = build_dataset(
        tmp_path, {"T1": 2, "V1": 1, "X1": 1}
    )
    assignment = rebalance_validation(manifest, root, labels, min_val=0)
    assert len(assignment["train"]) == 2
    assert len(assignment["val"]) == 1
    assert len(assignment["test"]) == 1


def test_validation_is_topped_up_from_training(tmp_path: Path):
    root, labels, manifest = build_dataset(
        tmp_path, {"T1": 4, "T2": 2, "T3": 2, "V1": 1, "X1": 3}
    )
    assignment = rebalance_validation(manifest, root, labels, min_val=3)
    assert len(assignment["val"]) >= 3
    # Lo que entra en validación sale de entrenamiento: nada se duplica.
    assert len(assignment["train"]) + len(assignment["val"]) == 4 + 2 + 2 + 1


def test_rebalancing_moves_whole_cylinders(tmp_path: Path):
    """Mover fotos sueltas rompería la separación por identidad."""
    root, labels, manifest = build_dataset(
        tmp_path, {"T1": 4, "T2": 3, "V1": 1, "X1": 2}
    )
    assignment = rebalance_validation(manifest, root, labels, min_val=3)

    def cylinders_of(split: str) -> set[str]:
        return {path.parts[0] for path in assignment[split]}

    assert not cylinders_of("train") & cylinders_of("val")
    assert not cylinders_of("train") & cylinders_of("test")
    assert not cylinders_of("val") & cylinders_of("test")


def test_test_split_is_never_touched(tmp_path: Path):
    """El test es la referencia de la evaluación: no se toca al rebalancear."""
    root, labels, manifest = build_dataset(
        tmp_path, {"T1": 4, "T2": 3, "V1": 0, "X1": 2}
    )
    before = rebalance_validation(manifest, root, labels, min_val=0)["test"]
    after = rebalance_validation(manifest, root, labels, min_val=5)["test"]
    assert before == after


def test_flatten_name_keeps_cylinder_and_condition():
    flat = flatten_name(Path("Cilindro_007/3.Cerca/IMG_1234.jpg"))
    assert flat == "Cilindro_007__3.Cerca__IMG_1234.jpg"
    # El origen debe seguir siendo legible al analizar los fallos.
    assert "Cilindro_007" in flat and "3.Cerca" in flat


@pytest.mark.parametrize("min_val", [0, 3, 100])
def test_rebalance_is_stable_for_any_minimum(tmp_path: Path, min_val: int):
    root, labels, manifest = build_dataset(tmp_path, {"T1": 2, "T2": 2, "V1": 1, "X1": 1})
    assignment = rebalance_validation(manifest, root, labels, min_val=min_val)
    total = sum(len(v) for v in assignment.values())
    assert total == 6, "ninguna imagen anotada se pierde ni se duplica"
