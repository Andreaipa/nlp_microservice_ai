"""Pruebas del pipeline completo con un OCR controlado."""
from __future__ import annotations

import pytest

from app.api.deps import build_container
from app.core.config import Settings
from app.ocr.null_engine import NullOcrEngine
from app.parsing.serial import consensus_candidates
from app.schemas.analyze import MatchStatus, ReviewReason
from app.schemas.common import FieldSource, OcrLine
from app.services.pipeline import AnalysisPipeline

REAL = "19S206055"

# Lecturas observadas al pasar PaddleOCR por las cinco variantes de realce
# sobre una placa troquelada. Se conservan literalmente porque reproducen el
# modo de fallo que motivó el consenso.
VARIANT_READINGS = [
    [("P5 19S206055", 0.99), ("AB 09/19 C02 45.2KG", 0.96)],
    [("P59S206055", 0.99), ("FAB 09/1.9 C02 45.2KG", 0.94)],
    [("JP5 19S206055", 0.96), ("FAB 09/19 C02 45.2KG", 0.99)],
    [("5.9S206055", 0.96), ("AB 09/19 802 45.2KG", 0.93)],
    [("JP5 19S206055", 0.97), ("FAB 09/19 C02 45.2KG", 0.99)],
]


def test_consensus_beats_the_single_most_confident_reading():
    """La variante más segura leía 'P59S206055'; el acuerdo corrige el error."""
    winner = consensus_candidates(VARIANT_READINGS)[0]
    assert winner.text == REAL


def test_consensus_of_a_single_variant_still_works():
    candidates = consensus_candidates([[("N° DE SERIE: 19S206055", 0.9)]])
    assert candidates[0].text == REAL


def test_consensus_with_no_readings_is_empty():
    assert consensus_candidates([]) == []
    assert consensus_candidates([[]]) == []


def build_pipeline(settings: Settings, lines: list[OcrLine]) -> AnalysisPipeline:
    container = build_container(settings)
    container.pipeline._ocr = NullOcrEngine(lines)
    return container.pipeline


@pytest.mark.asyncio
async def test_pipeline_reads_serial_and_enriches_from_catalog(
    settings: Settings, stamped_jpeg: bytes
):
    lines = [
        OcrLine(text="JP5 19S206055", confidence=0.93),
        OcrLine(text="FAB 09/19 C02 45.2KG", confidence=0.90),
    ]
    pipeline = build_pipeline(settings, lines)
    result = await pipeline.analyze(stamped_jpeg, request_id="test-1")

    assert result.data.numero_serie.value == REAL
    assert result.data.numero_serie.source is FieldSource.AI
    assert result.match.status is MatchStatus.MATCHED

    # El catálogo completa lo que la foto no da, marcando su procedencia.
    assert result.data.ancho_diametro.value == pytest.approx(22.0)
    assert result.data.ancho_diametro.source is FieldSource.CATALOG


@pytest.mark.asyncio
async def test_ocr_confusion_is_recovered_through_the_catalog(
    settings: Settings, stamped_jpeg: bytes
):
    """'C02' por 'CO2' y '195206055' por '19S206055' son errores esperables."""
    lines = [
        OcrLine(text="195206055", confidence=0.82),
        OcrLine(text="C02 45.2KG", confidence=0.80),
    ]
    pipeline = build_pipeline(settings, lines)
    result = await pipeline.analyze(stamped_jpeg, request_id="test-2")

    assert result.data.numero_serie.value == REAL
    assert result.data.tipo_gas.value == "CO2"


@pytest.mark.asyncio
async def test_unknown_serial_is_flagged_not_invented(
    settings: Settings, stamped_jpeg: bytes
):
    lines = [OcrLine(text="99Z999999", confidence=0.95)]
    pipeline = build_pipeline(settings, lines)
    result = await pipeline.analyze(stamped_jpeg, request_id="test-3")

    assert result.match.status is MatchStatus.NOT_FOUND
    assert result.match.resolved_serial is None
    assert result.requires_manual_confirmation is True
    assert ReviewReason.SERIAL_NOT_IN_CATALOG in result.review_reasons


@pytest.mark.asyncio
async def test_no_text_produces_empty_data_and_review(
    settings: Settings, stamped_jpeg: bytes
):
    pipeline = build_pipeline(settings, [])
    result = await pipeline.analyze(stamped_jpeg, request_id="test-4")

    assert result.data.numero_serie.value is None
    assert result.requires_manual_confirmation is True
    assert ReviewReason.NO_SERIAL_DETECTED in result.review_reasons


@pytest.mark.asyncio
async def test_untrained_detector_is_declared(settings: Settings, stamped_jpeg: bytes):
    pipeline = build_pipeline(settings, [])
    result = await pipeline.analyze(stamped_jpeg, request_id="test-5")
    assert ReviewReason.NO_DETECTOR_MODEL in result.review_reasons


@pytest.mark.asyncio
async def test_audit_entry_is_written(settings: Settings, stamped_jpeg: bytes):
    lines = [OcrLine(text="19S206055", confidence=0.9)]
    pipeline = build_pipeline(settings, lines)
    await pipeline.analyze(stamped_jpeg, request_id="test-6")

    records = list(settings.audit_dir.rglob("inferences.jsonl"))
    assert records, "el análisis debe dejar traza de auditoría"
    assert "19S206055" in records[0].read_text(encoding="utf-8")


class SizeAwareOcr:
    """OCR de prueba que sólo reconoce texto en la imagen completa.

    Reproduce el caso real que motiva el respaldo: el recorte de la región
    detectada corta el número de serie y lo deja ilegible, mientras que la
    imagen entera sí permite leerlo.
    """

    name = "size-aware"

    def __init__(self, full_height: int, lines: list[OcrLine]) -> None:
        self._full_height = full_height
        self._lines = lines

    def is_ready(self) -> bool:
        return True

    def describe(self) -> str:
        return "size-aware-test-ocr"

    def read(self, image):
        # Las variantes de realce reescalan el recorte, así que se compara la
        # proporción en lugar del alto exacto.
        return list(self._lines) if image.shape[0] >= self._full_height else []


@pytest.mark.asyncio
async def test_whole_image_rescues_a_bad_crop(settings: Settings, stamped_jpeg: bytes):
    container = build_container(settings)
    pipeline = container.pipeline
    pipeline._ocr = SizeAwareOcr(300, [OcrLine(text="19S206055", confidence=0.9)])

    result = await pipeline.analyze(stamped_jpeg, request_id="test-fallback")

    assert result.data.numero_serie.value == REAL
    assert result.match.status is MatchStatus.MATCHED
    # La razón del primer intento fallido no debe sobrevivir al reintento.
    assert ReviewReason.NO_SERIAL_DETECTED not in result.review_reasons


@pytest.mark.asyncio
async def test_duplicate_serial_forces_manual_confirmation(
    settings: Settings, stamped_jpeg: bytes, tmp_path
):
    """Aunque la lectura sea perfecta, un serial duplicado no identifica."""
    import json

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"cylinders": [
        {"numero_serie": "K5738166", "cilindro": "Cilindro_051", "marca": "JD"},
        {"numero_serie": "K5738166", "cilindro": "Cilindro_053", "marca": "JD"},
    ]}), encoding="utf-8")

    scoped = settings.model_copy(update={
        "catalog_path": catalog_path, "thresholds_calibrated": True,
        "serial_auto_accept_confidence": 0.1,
    })
    pipeline = build_pipeline(scoped, [OcrLine(text="K5738166", confidence=0.99)])
    result = await pipeline.analyze(stamped_jpeg, request_id="dup")

    assert result.requires_manual_confirmation is True
    assert ReviewReason.DUPLICATE_SERIAL_IN_CATALOG in result.review_reasons
    assert any("Cilindro_051" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_short_serial_forces_manual_confirmation(
    settings: Settings, stamped_jpeg: bytes, tmp_path
):
    import json

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"cylinders": [
        {"numero_serie": "804", "cilindro": "Cilindro_081"},
    ]}), encoding="utf-8")

    scoped = settings.model_copy(update={
        "catalog_path": catalog_path, "thresholds_calibrated": True,
        "serial_auto_accept_confidence": 0.1,
    })
    pipeline = build_pipeline(scoped, [OcrLine(text="804", confidence=0.99)])
    result = await pipeline.analyze(stamped_jpeg, request_id="short")

    assert result.requires_manual_confirmation is True
    assert ReviewReason.SERIAL_TOO_SHORT in result.review_reasons


class BoxedOcr:
    """OCR de prueba que devuelve líneas con geometría controlada."""

    name = "boxed"

    def __init__(self, lines: list[OcrLine]) -> None:
        self._lines = lines

    def is_ready(self) -> bool:
        return True

    def describe(self) -> str:
        return "boxed-test-ocr"

    def read(self, image):
        return list(self._lines)


@pytest.mark.asyncio
async def test_vertical_text_on_whole_image_is_rejected(
    settings: Settings, stamped_jpeg: bytes
):
    """Reproduce la alucinación observada en el dataset real.

    Sobre pintura descascarada el OCR devolvió una cadena parecida a un serial
    con 0.96 de confianza en una caja vertical, donde no había texto. La
    confianza no la descarta; la forma de la caja sí.
    """
    from app.schemas.common import BoundingBox

    container = build_container(settings)
    pipeline = container.pipeline
    # 41 px de ancho por 139 de alto: las medidas del caso real.
    pipeline._ocr = BoxedOcr([
        OcrLine(text="19S206055", confidence=0.96,
                box=BoundingBox(x1=401, y1=688, x2=442, y2=827)),
    ])

    result = await pipeline.analyze(stamped_jpeg, request_id="halluc")

    assert result.data.numero_serie.value is None
    assert result.match.status is not MatchStatus.MATCHED


@pytest.mark.asyncio
async def test_horizontal_text_on_whole_image_is_accepted(
    settings: Settings, stamped_jpeg: bytes
):
    from app.schemas.common import BoundingBox

    container = build_container(settings)
    pipeline = container.pipeline
    pipeline._ocr = BoxedOcr([
        OcrLine(text="19S206055", confidence=0.90,
                box=BoundingBox(x1=100, y1=200, x2=500, y2=260)),
    ])

    result = await pipeline.analyze(stamped_jpeg, request_id="ok")
    assert result.data.numero_serie.value == REAL


@pytest.mark.asyncio
async def test_partial_reading_matching_a_short_serial_is_not_auto_accepted(
    settings: Settings, stamped_jpeg: bytes, tmp_path
):
    """Una lectura parcial que coincide con un serial corto no basta.

    Reproduce el fallo detectado sobre el test real: el OCR lee '20640', que
    es el serial completo de otro cilindro del inventario. Aunque el catálogo
    lo resuelva, el resultado debe quedar en manos de una persona.
    """
    import json

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"cylinders": [
        {"numero_serie": "20640", "cilindro": "Cilindro_015"},
        {"numero_serie": "21S062189", "cilindro": "Cilindro_090"},
    ]}), encoding="utf-8")

    scoped = settings.model_copy(update={
        "catalog_path": catalog_path,
        "thresholds_calibrated": True,
        "serial_auto_accept_confidence": 0.1,
    })
    pipeline = build_pipeline(scoped, [OcrLine(text="20640", confidence=0.99)])
    result = await pipeline.analyze(stamped_jpeg, request_id="partial")

    assert result.requires_manual_confirmation is True
    assert ReviewReason.SERIAL_TOO_SHORT in result.review_reasons
