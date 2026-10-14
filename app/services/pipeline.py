"""Orquestación del análisis de una fotografía de cilindro.

    imagen -> preprocesado -> detección -> recortes -> OCR -> parser
           -> resolución contra catálogo -> enriquecimiento -> JSON

Principio rector: el servicio declara lo que observó y con cuánta confianza.
No decide nada de negocio y no rellena campos por verosimilitud. Un campo que
no se pudo leer ni consultar se queda vacío con `source: unknown`, porque un
hueco explícito es mucho menos dañino que un dato inventado.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger
from app.observability.audit import AuditLog
from app.ocr.base import OcrEngine
from app.ocr.layout import is_plausible_text_box, join_split_lines
from app.parsing.fields import parse_fields
from app.parsing.serial import (
    MIN_UNMATCHED_PROPOSAL_LENGTH,
    SerialCandidate,
    allowed_errors,
    consensus_candidates,
    fuzzy_substring_distance,
    normalize_serial_charset,
)
from app.preprocessing import enhance, image_io
from app.schemas.analyze import (
    AnalyzeResponse,
    BackendMatch,
    ImageQuality,
    MatchStatus,
    ReviewReason,
    SerialMatch,
)
from app.schemas.common import Detection, FieldSource, OcrLine, TracedValue
from app.schemas.cylinder import CylinderData
from app.services.backend_client import BackendClient
from app.services.catalog import CylinderCatalog
from app.vision.base import DetectionClass, Detector

logger = get_logger(__name__)

# Cuántos candidatos a número de serie se contrastan contra el inventario.
MAX_CANDIDATES_CHECKED = 25


# Razones de revisión que se refieren al intento de leer el serial. Si un
# reintento lo resuelve, dejan de aplicar y se retiran.
# Motivos que fuerzan confirmación manual por sí solos, independientemente de
# la confianza del OCR y de si los umbrales están calibrados.
_BLOCKING_REASONS = frozenset({
    ReviewReason.DUPLICATE_SERIAL_IN_CATALOG,
    ReviewReason.SERIAL_TOO_SHORT,
})

_SERIAL_REASONS = frozenset({
    ReviewReason.NO_SERIAL_DETECTED,
    ReviewReason.SERIAL_FORMAT_INVALID,
    ReviewReason.SERIAL_NOT_IN_CATALOG,
    ReviewReason.AMBIGUOUS_CATALOG_MATCH,
})

@dataclass
class _ImageStage:
    """Resultado de la parte de CPU del análisis (ver `_analyze_image`)."""

    image: np.ndarray
    quality: ImageQuality
    detections: list[Detection]
    ocr_lines: list[OcrLine]
    data: CylinderData
    serial_candidate: SerialCandidate | None
    match: SerialMatch
    warnings: list[str]
    review_reasons: list[ReviewReason]


class AnalysisPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        detector: Detector,
        ocr_engine: OcrEngine,
        catalog: CylinderCatalog,
        backend: BackendClient,
        audit: AuditLog,
    ) -> None:
        self._settings = settings
        self._detector = detector
        self._ocr = ocr_engine
        self._catalog = catalog
        self._backend = backend
        self._audit = audit
        # Análisis simultáneos. Cada uno ocupa CPU y memoria (la imagen y sus
        # variantes); los que excedan el límite esperan su turno en vez de
        # competir por la CPU y alargar todos.
        self._cpu_slots = asyncio.Semaphore(settings.max_concurrent_analyses)

    # -- descripción del montaje --------------------------------------------
    @property
    def model_version(self) -> str:
        """Versión efectiva del conjunto que produjo el resultado."""
        return f"detector:{self._detector.describe()}|ocr:{self._ocr.describe()}|parser:1.0.0"

    def describe_components(self) -> dict[str, str]:
        return {
            "detector": self._detector.describe(),
            "ocr": self._ocr.describe(),
            "catalog": f"{len(self._catalog)} cilindros",
        }

    # -- ejecución -----------------------------------------------------------
    async def analyze(self, payload: bytes, *, request_id: str) -> AnalyzeResponse:
        started = time.perf_counter()

        # Decodificar, detectar y leer cuesta uno o varios segundos de CPU. Se
        # hace en un hilo para no bloquear el bucle de eventos: si no, mientras
        # se analiza una foto el servicio no responde a nada más, ni siquiera a
        # /health, y un balanceador lo daría por caído. ONNX Runtime y OpenCV
        # liberan el GIL durante el cálculo.
        async with self._cpu_slots:
            stage = await asyncio.to_thread(self._analyze_image, payload)
        image, quality, detections = stage.image, stage.quality, stage.detections
        ocr_lines, data, match = stage.ocr_lines, stage.data, stage.match
        serial_candidate = stage.serial_candidate
        warnings, review_reasons = stage.warnings, stage.review_reasons

        backend_match = await self._enrich_from_backend(data, match)

        needs_review = self._decide_review(
            serial_candidate, match, review_reasons,
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        response = AnalyzeResponse(
            success=True,
            request_id=request_id,
            service_version=self._settings.service_version,
            model_version=self.model_version,
            processing_time_ms=elapsed_ms,
            requires_manual_confirmation=needs_review,
            review_reasons=sorted(set(review_reasons), key=lambda r: r.value),
            data=data,
            match=match,
            backend_match=backend_match,
            detections=detections,
            ocr_lines=ocr_lines,
            image_quality=quality,
            warnings=warnings,
        )

        # Escribe a disco (y codifica la imagen si hay que revisarla).
        await asyncio.to_thread(
            self._audit.record,
            request_id=request_id,
            payload=self._audit_payload(response),
            image=image,
            needs_review=needs_review,
        )
        logger.info(
            "analisis completado",
            extra={
                "processing_time_ms": elapsed_ms,
                "detections": len(detections),
                "ocr_lines": len(ocr_lines),
                "serial": data.numero_serie.value,
                "match_status": match.status.value,
                "requires_manual_confirmation": needs_review,
            },
        )
        return response

    def _analyze_image(self, payload: bytes) -> _ImageStage:
        """Parte de CPU del análisis. Se ejecuta fuera del bucle de eventos."""
        image = image_io.decode_image(payload, max_bytes=self._settings.max_upload_bytes)
        image, _ = image_io.limit_size(image, self._settings.max_image_side)
        quality = image_io.assess_quality(image)

        warnings: list[str] = []
        review_reasons: list[ReviewReason] = []

        if not quality.is_acceptable:
            review_reasons.append(ReviewReason.POOR_IMAGE_QUALITY)
            warnings.append(
                f"Calidad de imagen baja (nitidez={quality.blur_score}, "
                f"brillo={quality.brightness}, contraste={quality.contrast})."
            )

        detections = self._detect(image, warnings, review_reasons)
        ocr_lines, variant_readings = self._read_regions(image, detections)

        if not ocr_lines:
            warnings.append("El OCR no reconoció texto en la imagen.")

        pairs = [(line.text, line.confidence) for line in ocr_lines]
        parse_result = parse_fields(pairs)
        warnings.extend(parse_result.warnings)
        data = parse_result.data

        serial_candidate, match = self._resolve_serial(variant_readings, review_reasons)

        # Respaldo: si las regiones detectadas no permitieron resolver el
        # serial, se reintenta leyendo la imagen completa. Un recorte mal
        # ajustado puede partir el número por la mitad y dejarlo ilegible.
        #
        # Se hace sólo en ese caso y no siempre: sobre una fotografía de 9 MP
        # cada pasada de OCR cuesta segundos, y cuando el recorte ya funcionó
        # no aporta nada. Las razones de revisión del primer intento se
        # descartan si el reintento tiene éxito, para no arrastrar un
        # diagnóstico que dejó de ser cierto.
        if detections and match.status is not MatchStatus.MATCHED:
            whole_lines, whole_readings = self._read_single(image, region_label=None)
            if whole_readings:
                combined = variant_readings + whole_readings
                retry_reasons: list[ReviewReason] = []
                retry_candidate, retry_match = self._resolve_serial(
                    combined, retry_reasons
                )
                if retry_match.status is MatchStatus.MATCHED:
                    review_reasons[:] = [
                        reason for reason in review_reasons
                        if reason not in _SERIAL_REASONS
                    ]
                    review_reasons.extend(retry_reasons)
                    serial_candidate, match = retry_candidate, retry_match
                    variant_readings = combined
                    ocr_lines.extend(whole_lines)
                    pairs = [(line.text, line.confidence) for line in ocr_lines]
                    parse_result = parse_fields(pairs)
                    data = parse_result.data
                    # El re-análisis puede descubrir campos nuevos y, con
                    # ellos, avisos que antes no existían.
                    warnings.extend(
                        w for w in parse_result.warnings if w not in warnings
                    )

        if serial_candidate is not None:
            data.numero_serie = TracedValue(
                value=match.resolved_serial or serial_candidate.text,
                confidence=round(serial_candidate.confidence, 4),
                source=FieldSource.AI,
                raw=serial_candidate.raw,
            )
            if serial_candidate.was_corrected:
                warnings.append(
                    "Se corrigieron caracteres del serial según el patrón esperado: "
                    + "; ".join(serial_candidate.corrections)
                )

        # Defectos del maestro que impiden identificar con certeza aunque la
        # lectura haya sido correcta.
        if match.resolved_serial:
            entry = self._catalog.get(match.resolved_serial)
            if entry is not None:
                warnings.extend(self._catalog.integrity_warnings(match.resolved_serial))
                if entry.is_ambiguous:
                    review_reasons.append(ReviewReason.DUPLICATE_SERIAL_IN_CATALOG)
                if entry.is_too_short:
                    review_reasons.append(ReviewReason.SERIAL_TOO_SHORT)

        self._enrich_from_catalog(data, match)
        return _ImageStage(
            image=image,
            quality=quality,
            detections=detections,
            ocr_lines=ocr_lines,
            data=data,
            serial_candidate=serial_candidate,
            match=match,
            warnings=warnings,
            review_reasons=review_reasons,
        )

    # -- etapas --------------------------------------------------------------
    def _detect(
        self,
        image: np.ndarray,
        warnings: list[str],
        review_reasons: list[ReviewReason],
    ) -> list[Detection]:
        if self._detector.name == "heuristic":
            review_reasons.append(ReviewReason.NO_DETECTOR_MODEL)
            warnings.append(
                "No hay modelo de detección entrenado: se está usando el detector "
                "heurístico. Los resultados requieren verificación."
            )
        try:
            return self._detector.detect(image)
        except Exception:
            logger.exception("fallo en la detección")
            warnings.append("La detección de regiones falló; se analizó la imagen completa.")
            return []

    def _read_regions(
        self, image: np.ndarray, detections: list[Detection]
    ) -> tuple[list[OcrLine], list[list[tuple[str, float]]]]:
        """Ejecuta OCR sobre las regiones con texto, o sobre toda la imagen.

        Devuelve la mejor lectura de cada región (para el parser de campos) y
        las lecturas de todas las variantes de realce (para el consenso del
        número de serie).
        """
        text_regions = [
            detection for detection in detections
            if detection.label in DetectionClass.TEXT_BEARING
        ]

        # Sin regiones de texto no se descarta la imagen: se intenta leerla
        # entera. Es peor no leer nada que leer con más ruido.
        if not text_regions:
            return self._read_single(image, region_label=None)

        # Las regiones más específicas primero: serial_text antes que el
        # collarín completo.
        text_regions.sort(
            key=lambda d: (d.label != DetectionClass.SERIAL_TEXT, -d.confidence)
        )

        lines: list[OcrLine] = []
        readings: list[list[tuple[str, float]]] = []
        for detection in text_regions:
            box = (detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2)
            patch = image_io.crop(
                image, box,
                padding=self._settings.detector_crop_padding,
                padding_bottom=self._settings.detector_crop_padding_bottom,
            )
            if patch.size == 0:
                continue
            region_lines, region_readings = self._read_single(
                patch, region_label=detection.label,
                offset=(detection.box.x1, detection.box.y1),
            )
            lines.extend(region_lines)
            readings.extend(region_readings)
        return lines, readings

    def _read_single(
        self,
        patch: np.ndarray,
        *,
        region_label: str | None,
        offset: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[list[OcrLine], list[list[tuple[str, float]]]]:
        """Prueba varias variantes de realce sobre un mismo recorte.

        Cuál variante funciona depende de la iluminación de cada fotografía y
        no es predecible de antemano, así que se ejecutan todas las
        configuradas y se conservan dos cosas: la mejor lectura, que alimenta
        el parser de campos, y todas las lecturas, que alimentan el consenso
        del número de serie.
        """
        if not self._ocr.is_ready():
            return [], []

        corrected = enhance.deskew(patch)
        variants = enhance.build_ocr_variants(
            corrected, self._settings.ocr_preprocess_variants
        )

        best: list[OcrLine] = []
        best_score = 0.0
        readings: list[list[tuple[str, float]]] = []

        for variant_name, prepared in variants:
            try:
                lines = self._ocr.read(prepared)
            except Exception:
                logger.exception("fallo del OCR", extra={"variante": variant_name})
                continue
            # La geometría se filtra en cada variante. Antes se filtraba sólo
            # la mejor y se descartaba de las demás todo texto que no leyera
            # ella, lo que anulaba el consenso: si otra variante leía mejor el
            # serial, su lectura se perdía.
            ignored = self._settings.ignored_reading_set
            lines = [
                line for line in lines
                if is_plausible_text_box(line)
                and normalize_serial_charset(line.text) not in ignored
            ]
            if not lines:
                continue

            # Las uniones sólo alimentan la búsqueda del serial; las líneas
            # originales siguen siendo las que se devuelven.
            joined = join_split_lines(lines)
            readings.append(
                [(line.text, line.confidence) for line in [*lines, *joined]]
            )

            score = max(line.confidence for line in lines)
            if score > best_score:
                best_score = score
                best = [
                    OcrLine(
                        text=line.text,
                        confidence=line.confidence,
                        box=self._shift_box(line.box, offset, prepared, patch),
                        source_region=region_label,
                    )
                    for line in lines
                ]

        return best, readings

    @staticmethod
    def _shift_box(box, offset, prepared: np.ndarray, original: np.ndarray):
        """Reproyecta una caja del recorte realzado a la imagen original."""
        if box is None:
            return None
        scale_x = original.shape[1] / max(prepared.shape[1], 1)
        scale_y = original.shape[0] / max(prepared.shape[0], 1)
        offset_x, offset_y = offset
        return box.__class__(
            x1=box.x1 * scale_x + offset_x, y1=box.y1 * scale_y + offset_y,
            x2=box.x2 * scale_x + offset_x, y2=box.y2 * scale_y + offset_y,
        )

    def _resolve_serial(
        self,
        readings: list[list[tuple[str, float]]],
        review_reasons: list[ReviewReason],
    ) -> tuple[SerialCandidate | None, SerialMatch]:
        pattern = self._catalog.pattern
        candidates = consensus_candidates(readings, pattern)
        texts = [text for reading in readings for text, _ in reading]

        if not texts or (not candidates and not self._settings.catalog_enabled):
            review_reasons.append(ReviewReason.NO_SERIAL_DETECTED)
            return None, SerialMatch(status=MatchStatus.NOT_ATTEMPTED)

        if not self._settings.catalog_enabled:
            best = candidates[0]
            if not best.pattern_valid:
                review_reasons.append(ReviewReason.SERIAL_FORMAT_INVALID)
            return best, SerialMatch(status=MatchStatus.NO_CATALOG,
                                     resolved_serial=best.text)

        # 1. Buscar cada serial del inventario dentro de todo lo leído. Es la
        #    evidencia más fuerte: no depende de que el OCR haya aislado el
        #    número ni de qué fragmento leyó con más confianza (en las pruebas
        #    con teléfono, "PW200" se leía con 100% y el serial con menos).
        match = self._catalog.search(texts)
        if match.status is MatchStatus.MATCHED and match.resolved_serial:
            return self._supporting_candidate(match.resolved_serial, readings), match

        # 2. Candidatos aislados que sean un fragmento de un serial conocido:
        #    el OCR suele perder los primeros caracteres, los más gastados
        #    ("206055" por "19S206055"), y eso la búsqueda anterior no lo cubre.
        for candidate in candidates[:MAX_CANDIDATES_CHECKED]:
            resolved = self._catalog.resolve(candidate.text)
            if resolved.status is MatchStatus.MATCHED:
                return candidate, resolved

        # 3. Sin identificación. Se propone una lectura sólo si tiene forma de
        #    serial: un fragmento como "060" o "200" confunde más de lo que
        #    ayuda. Las alternativas del inventario se ofrecen igualmente.
        proposal = next(
            (c for c in candidates if len(c.text) >= MIN_UNMATCHED_PROPOSAL_LENGTH), None
        )
        if match.status is MatchStatus.AMBIGUOUS:
            review_reasons.append(ReviewReason.AMBIGUOUS_CATALOG_MATCH)
        elif proposal is not None:
            review_reasons.append(ReviewReason.SERIAL_NOT_IN_CATALOG)
            if not proposal.pattern_valid:
                review_reasons.append(ReviewReason.SERIAL_FORMAT_INVALID)
        else:
            review_reasons.append(ReviewReason.NO_SERIAL_DETECTED)
        if match.status is MatchStatus.NOT_ATTEMPTED:
            match = SerialMatch(status=MatchStatus.NOT_FOUND)
        return proposal, match

    @staticmethod
    def _supporting_candidate(
        serial: str, readings: list[list[tuple[str, float]]]
    ) -> SerialCandidate:
        """La lectura que respalda un serial encontrado en el inventario.

        Su confianza combina, como el consenso, en cuántas variantes de realce
        aparece el serial y con qué confianza las leyó el OCR.
        """
        budget = allowed_errors(serial)
        supporting: list[tuple[float, str, float]] = []  # distancia, texto, confianza
        variants_with_support = 0
        for reading in readings:
            hits = [
                (fuzzy_substring_distance(serial, normalize_serial_charset(text)), text, conf)
                for text, conf in reading
            ]
            hits = [hit for hit in hits if hit[0] <= budget]
            if hits:
                variants_with_support += 1
                supporting.extend(hits)
        if not supporting:  # serial corto, encontrado como número exacto
            return SerialCandidate(text=serial, confidence=0.0, raw=serial, pattern_valid=True)

        _, raw, _ = min(supporting, key=lambda hit: (hit[0], -hit[2]))
        agreement = variants_with_support / max(len([r for r in readings if r]), 1)
        mean_confidence = sum(hit[2] for hit in supporting) / len(supporting)
        return SerialCandidate(
            text=normalize_serial_charset(raw),
            confidence=round(min(1.0, 0.5 * agreement + 0.5 * mean_confidence), 4),
            raw=raw.strip(),
            pattern_valid=True,
        )

    def _enrich_from_catalog(self, data: CylinderData, match: SerialMatch) -> None:
        """Completa huecos con el maestro local, sin pisar lo leído por IA."""
        if match.status is not MatchStatus.MATCHED or not match.resolved_serial:
            return
        entry = self._catalog.get(match.resolved_serial)
        if entry is None:
            return

        for name, value in entry.attributes.items():
            if not hasattr(data, name) or value is None:
                continue
            current: TracedValue = getattr(data, name)
            if current.is_present:
                continue
            setattr(data, name, TracedValue(
                value=value, confidence=None, source=FieldSource.CATALOG,
            ))

    async def _enrich_from_backend(
        self, data: CylinderData, match: SerialMatch
    ) -> BackendMatch:
        if not self._backend.enabled:
            return BackendMatch(found=False, queried=False)
        if match.status is not MatchStatus.MATCHED or not match.resolved_serial:
            return BackendMatch(found=False, queried=False)

        payload, error = await self._backend.fetch_cylinder(match.resolved_serial)
        if error:
            return BackendMatch(found=False, queried=True, error=error)
        if payload is None:
            return BackendMatch(found=False, queried=True)

        for name, value in payload.items():
            if not hasattr(data, name) or value is None:
                continue
            current: TracedValue = getattr(data, name)
            # El backend es la fuente de verdad de los datos maestros: prevalece
            # sobre una lectura por OCR de un campo que no es identificador.
            if name != "numero_serie" or not current.is_present:
                setattr(data, name, TracedValue(
                    value=value, confidence=None, source=FieldSource.BACKEND,
                ))

        return BackendMatch(found=True, queried=True)

    def _decide_review(
        self,
        candidate: SerialCandidate | None,
        match: SerialMatch,
        review_reasons: list[ReviewReason],
    ) -> bool:
        """Decide si el resultado necesita confirmación humana.

        Los umbrales son provisionales mientras no se calibren con el conjunto
        de test real; hasta entonces el servicio marca TODO para revisión, para
        no dar por buena una identificación con un umbral sin fundamento
        empírico. Ver docs/THRESHOLDS.md.
        """
        # Un defecto del maestro no lo arregla ninguna confianza: si el serial
        # está duplicado o es demasiado corto, decide una persona.
        if _BLOCKING_REASONS.intersection(review_reasons):
            return True

        if not self._settings.thresholds_calibrated:
            review_reasons.append(ReviewReason.THRESHOLDS_NOT_CALIBRATED)
            return True

        if candidate is None:
            return True

        confidence = candidate.confidence
        if confidence < self._settings.serial_review_confidence:
            review_reasons.append(ReviewReason.LOW_OCR_CONFIDENCE)
            return True

        if match.status is not MatchStatus.MATCHED:
            return True

        # Un umbral de 1.00 significa "no aceptar nunca de forma automática",
        # que es lo que concluyó la calibración. Hay que tratarlo como tal de
        # forma explícita: la confianza por consenso puede valer exactamente
        # 1.0 en una foto nítida, y con una comparación estricta ese caso
        # pasaría como aceptado sin que ninguna persona lo viera.
        auto_accept = self._settings.serial_auto_accept_confidence
        if auto_accept >= 1.0:
            review_reasons.append(ReviewReason.CONFIRMATION_REQUIRED)
            return True
        if confidence < auto_accept:
            review_reasons.append(ReviewReason.LOW_OCR_CONFIDENCE)
            return True

        return bool(review_reasons)

    @staticmethod
    def _audit_payload(response: AnalyzeResponse) -> dict[str, Any]:
        return {
            "model_version": response.model_version,
            "processing_time_ms": response.processing_time_ms,
            "requires_manual_confirmation": response.requires_manual_confirmation,
            "review_reasons": [reason.value for reason in response.review_reasons],
            "serial": response.data.numero_serie.value,
            "serial_confidence": response.data.numero_serie.confidence,
            "serial_raw": response.data.numero_serie.raw,
            "match_status": response.match.status.value,
            "detections": [
                {"label": d.label, "confidence": d.confidence,
                 "box": [d.box.x1, d.box.y1, d.box.x2, d.box.y2]}
                for d in response.detections
            ],
            "ocr": [{"text": line.text, "confidence": line.confidence}
                    for line in response.ocr_lines],
            "image_quality": (
                response.image_quality.model_dump() if response.image_quality else None
            ),
        }
