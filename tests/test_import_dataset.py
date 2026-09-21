"""Pruebas de la normalización de nombres al importar el dataset."""
from __future__ import annotations

import pytest

from training.import_dataset import canonical_condition, canonical_cylinder


@pytest.mark.parametrize("raw,expected", [
    ("1. Frontal ", "1.Frontal"),
    ("1.Frontal", "1.Frontal"),
    ("6. Poca luz", "6.Poca luz"),
    ("5. Luz normal", "5.Luz normal"),
    ("Descripción", "Descripción"),
    ("Descripción 85", "Descripción"),   # variante real del cilindro 085
    ("Descripcion ", "Descripción"),
])
def test_condition_names_are_normalized(raw, expected):
    assert canonical_condition(raw) == expected


def test_unknown_condition_is_reported_not_guessed():
    assert canonical_condition("Fotos varias") is None


@pytest.mark.parametrize("raw,expected", [
    ("Cilindro_002", "Cilindro_002"),
    ("Cilindro 7", "Cilindro_007"),
    ("cilindro_100", "Cilindro_100"),
])
def test_cylinder_names_are_zero_padded(raw, expected):
    assert canonical_cylinder(raw) == expected


def test_non_cylinder_folder_is_ignored():
    assert canonical_cylinder("Documentos") is None
