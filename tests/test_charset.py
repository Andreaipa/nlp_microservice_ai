"""Pruebas del alfabeto del reconocedor especializado."""
from __future__ import annotations

import pytest

from app.ocr.charset import ALPHABET, BLANK_INDEX, NUM_CLASSES, decode, encode


def test_alphabet_covers_the_real_inventory():
    """Los seriales de los 100 cilindros usan sólo estos caracteres."""
    assert len(ALPHABET) == 21
    for char in "0123456789K":
        assert char in ALPHABET


def test_blank_is_reserved_for_ctc():
    assert BLANK_INDEX == 0
    assert len(ALPHABET) + 1 == NUM_CLASSES
    # Ningún carácter puede codificarse como el blanco.
    assert all(index > BLANK_INDEX for index in encode(ALPHABET))


def test_encode_decode_round_trip():
    """Un texto sin caracteres repetidos seguidos se recupera tal cual."""
    text = "19S2060"
    assert decode(encode(text)) == text


def test_repeated_characters_need_a_blank_between_them():
    """Así funciona CTC: sin blanco intermedio, dos iguales colapsan en uno.

    Es el comportamiento correcto del algoritmo, no un defecto: la red aprende
    a emitir un blanco entre caracteres repetidos.
    """
    indices = encode("55")
    assert decode(indices) == "5", "sin blanco, se colapsan"

    with_blank = [indices[0], BLANK_INDEX, indices[1]]
    assert decode(with_blank) == "55"


def test_decode_ignores_blanks_and_repetitions():
    serial = encode("19S206055")
    # Cada símbolo repetido varias veces, como emite realmente la red.
    noisy: list[int] = []
    for index in serial:
        noisy.extend([BLANK_INDEX, index, index, index])
    assert decode(noisy) == "19S206055"


def test_unknown_characters_are_dropped_not_guessed():
    assert encode("19S-206/055") == encode("19S206055")


@pytest.mark.parametrize("serial", ["19S206055", "K5738028", "21S491074", "804"])
def test_real_serials_are_representable(serial: str):
    assert all(c in ALPHABET for c in serial)


def test_recognizer_stays_disabled_without_a_model():
    """El reconocedor experimental no debe activarse solo.

    El modelo entrenado quedó por debajo del OCR genérico (CER 0.597 frente a
    0.537). Activarlo empeoraría el sistema, así que la configuración lo deja
    inactivo mientras no se apunte explícitamente a un modelo.
    """
    from app.core.config import Settings

    assert Settings(_env_file=None).serial_recognizer_path is None


def test_recognizer_refuses_a_mismatched_alphabet(tmp_path):
    """Un alfabeto distinto al del servicio devolvería caracteres erróneos.

    Y lo haría sin lanzar ningún error, que es lo peligroso: los índices que
    emite la red apuntarían a otras letras.
    """
    from app.ocr.serial_recognizer import SerialRecognizer

    model = tmp_path / "recognizer.onnx"
    model.write_bytes(b"no es un modelo")
    (tmp_path / "alphabet.txt").write_text("ABC", encoding="utf-8")

    recognizer = SerialRecognizer(model)
    assert recognizer.is_ready() is False
