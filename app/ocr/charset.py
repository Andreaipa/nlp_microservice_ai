"""Alfabeto del reconocedor especializado de números de serie.

Los seriales de los 100 cilindros del inventario usan sólo estos caracteres.
El alfabeto es deliberadamente cerrado: reconocer 21 símbolos en un estilo
concreto es un problema mucho más pequeño que el OCR general, y es la razón
por la que un modelo propio y pequeño puede superar a uno genérico en este
dominio.

Si el inventario incorpora cilindros con caracteres nuevos hay que ampliar
esta cadena y reentrenar; el servicio avisa al cargar el modelo si el alfabeto
guardado junto a los pesos no coincide con éste.
"""
from __future__ import annotations

ALPHABET = "0123456789EGHKLNPSTXY"

# El índice 0 queda reservado para el símbolo en blanco que necesita CTC.
BLANK_INDEX = 0


def encode(text: str) -> list[int]:
    return [ALPHABET.index(c) + 1 for c in text if c in ALPHABET]


def decode(indices: list[int]) -> str:
    """Decodifica una secuencia CTC: colapsa repeticiones y quita los blancos."""
    out: list[str] = []
    previous = BLANK_INDEX
    for index in indices:
        if index != previous and index != BLANK_INDEX:
            out.append(ALPHABET[index - 1])
        previous = index
    return "".join(out)


NUM_CLASSES = len(ALPHABET) + 1
