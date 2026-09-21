"""Extracción de campos estructurados a partir del texto reconocido.

El troquelado del collarín no viene rotulado: en el metal se lee algo parecido
a `JP5 19S206055 09/2019 CO2 45.2KG 40.4L CHN PH 02/2025`, sin la palabra
"MARCA" ni "PESO" delante. Por eso el parser trabaja en dos pasadas:

1. Busca etiquetas explícitas ("MARCA: JP5"), que aparecen cuando el cilindro
   lleva además una placa o calcomanía impresa.
2. Para lo que quede sin resolver, infiere el campo por la forma del token
   (una magnitud con "KG" es el peso, "40.4 L" son litros, un código de tres
   letras conocido es la procedencia, etc.).

Cuando un campo no se puede determinar se deja como desconocido. El servicio
no rellena huecos por verosimilitud: ese es trabajo del backend con su dato
maestro.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.parsing import normalizers as norm
from app.schemas.common import FieldSource, TracedValue
from app.schemas.cylinder import CylinderData

# Etiquetas admitidas por campo, ya sin acentos y en mayúsculas.
FIELD_LABELS: dict[str, tuple[str, ...]] = {
    "numero_serie": ("N DE SERIE", "NO DE SERIE", "NRO DE SERIE", "NUMERO DE SERIE",
                     "SERIE", "SERIAL", "S/N", "SN"),
    "marca": ("MARCA", "BRAND", "FABRICANTE"),
    "fecha_fabricacion": ("ANIO DE FABRICACION", "ANO DE FABRICACION",
                          "ANO FABRICACION", "FABRICACION", "FAB", "FECHA DE FABRICACION"),
    "tipo_gas": ("TIPO DE GAS", "GAS", "CONTENIDO"),
    "unidad_medida": ("UNIDAD DE MEDIDA", "UNIDAD", "UM"),
    "peso": ("PESO", "TARA", "WEIGHT", "TW"),
    "ancho_diametro": ("ANCHO DIAMETRO", "DIAMETRO", "ANCHO", "DIAM"),
    "largo_altura": ("LARGO ALTURA", "ALTURA", "LARGO", "ALTO"),
    "litros": ("LITROS", "CAPACIDAD", "WC", "VOLUMEN"),
    "color": ("COLOR",),
    "ph": ("PH", "PRUEBA HIDROSTATICA", "P H", "TEST", "RETEST"),
    "m3": ("M3", "METROS CUBICOS"),
    "procedencia": ("PROCEDENCIA", "ORIGEN", "PAIS", "MADE IN"),
}

_LABEL_SPLIT = re.compile(r"[:=]")

# Formas reconocibles sin etiqueta.
_RE_WEIGHT = re.compile(r"\b(\d{1,3}(?:[.,]\d{1,2})?)\s*(KG|KGS|LB|LBS)\b")
_RE_LITRES = re.compile(r"\b(\d{1,3}(?:[.,]\d{1,2})?)\s*(L|LT|LTS|LITROS)\b")
_RE_CUBIC = re.compile(r"\b(\d{1,3}(?:[.,]\d{1,3})?)\s*(M3|M\^3)\b")
_RE_MONTH_YEAR = re.compile(r"\b(0?[1-9]|1[0-2])\s*[/\-.]\s*((?:19|20)\d{2})\b")
# "PH 02/25", "TEST 02-2025": la marca de prueba hidrostática rara vez lleva
# dos puntos cuando está troquelada en el metal.
_RE_PH_MARKED = re.compile(
    r"\b(?:PH|TEST|RETEST|HT)\b[\s.:\-]*(\d{1,2}\s*[/\-.]\s*\d{2,4})"
)
# "FAB 09/2019", "MFG 09/19": lo mismo para la fecha de fabricación.
_RE_FAB_MARKED = re.compile(
    r"\b(?:FAB|MFG|MFD|MANUF\w*)\b[\s.:\-]*(\d{1,2}\s*[/\-.]\s*\d{2,4})"
)
_RE_BRAND = re.compile(r"\b([A-Z]{2,6}\d{0,3})\b")

# Tokens que nunca son una marca aunque encajen con la forma.
_BRAND_STOPWORDS = frozenset({
    # Unidades
    "KG", "KGS", "LB", "LBS", "CM", "MM", "M", "L", "LT", "LTS", "M3",
    # Gases
    "CO2", "O2", "AR", "N2", "H2", "HE", "AIR", "MIX", "ACET",
    # Marcadores de campo que preceden a un valor y no son marcas. "FAB" es
    # el caso más frecuente: aparece justo antes de la fecha de fabricación y
    # sin esta lista se confunde con el fabricante.
    "FAB", "MFG", "MFD", "PH", "HT", "TEST", "RETEST", "TW", "WC", "MADE", "IN",
    "SERIE", "SERIAL", "MARCA", "GAS", "COLOR", "PESO", "TARA", "LITROS",
    "CAPACIDAD", "UNIDAD", "MEDIDA", "PROCEDENCIA", "ORIGEN", "TIPO", "DE",
    # Códigos de país: son procedencia, no marca.
    "CHN", "PER", "USA", "BRA", "ARG", "ESP", "ITA", "DEU", "IND", "JPN",
    "KOR", "MEX", "CHL", "COL", "FRA", "GBR", "TUR", "THA", "VNM",
})


# Confianza mínima que debe tener la LECTURA OCR para deducir de ella un
# campo puramente textual (marca, color, procedencia) sin etiqueta que lo
# respalde.
#
# Viene de un fallo observado sobre una fotografía real: una pegatina leída
# como "Linca pgp" con baja confianza acabó asignada como marca del cilindro,
# cuya marca real es JP5. Un dato así estorba más de lo que ayuda.
#
# El umbral se aplica a la confianza del OCR, no al valor ya penalizado, para
# que siga siendo interpretable: "no deduzco una marca de un texto que el
# propio OCR no está seguro de haber leído".
#
# No se aplica a las magnitudes con unidad ("45.2 KG", "40.4 L") ni a los
# gases: su forma ya las valida, y exigirles alta confianza perdería lecturas
# correctas sin evitar ningún falso positivo.
MIN_TEXTUAL_OCR_CONFIDENCE = 0.55


@dataclass
class ParseResult:
    data: CylinderData = field(default_factory=CylinderData)
    warnings: list[str] = field(default_factory=list)


def _label_of(text: str) -> tuple[str, str] | None:
    """Si la línea trae 'ETIQUETA: valor', devuelve (campo, valor)."""
    parts = _LABEL_SPLIT.split(text, maxsplit=1)
    if len(parts) != 2:
        return None

    label = norm.clean_text(parts[0])
    value = parts[1].strip()
    if not label or not value:
        return None

    # Coincidencia por etiqueta más larga primero, para que "LARGO ALTURA"
    # gane sobre "LARGO".
    best: tuple[str, int] | None = None
    for field_name, aliases in FIELD_LABELS.items():
        for alias in aliases:
            matches = label == alias or label.endswith(alias) or alias in label
            if matches and (best is None or len(alias) > best[1]):
                best = (field_name, len(alias))
    return (best[0], value) if best else None


def _set_from_ai(
    data: CylinderData, name: str, value, confidence: float, raw: str,
    *, minimum: float = 0.0,
) -> None:
    """Asigna un valor sólo si el campo sigue vacío o si mejora la confianza."""
    if value is None or confidence < minimum:
        return
    current: TracedValue = getattr(data, name)
    if current.is_present and (current.confidence or 0.0) >= confidence:
        return
    setattr(data, name, TracedValue(
        value=value, confidence=round(confidence, 4), source=FieldSource.AI, raw=raw,
    ))


def _assign_labelled(result: ParseResult, name: str, value: str, confidence: float) -> None:
    data = result.data

    if name == "marca":
        cleaned = norm.clean_text(value)
        if cleaned and not norm.is_absent(cleaned) and cleaned not in _BRAND_STOPWORDS:
            _set_from_ai(data, "marca", cleaned, confidence, value)

    elif name == "tipo_gas":
        _set_from_ai(data, "tipo_gas", norm.normalize_gas(value), confidence, value)

    elif name == "color":
        _set_from_ai(data, "color", norm.normalize_color(value), confidence, value)

    elif name == "fecha_fabricacion":
        month_year = norm.normalize_month_year(value)
        if month_year:
            _set_from_ai(data, "fecha_fabricacion", month_year, confidence, value)
        year = norm.normalize_year(value)
        _set_from_ai(data, "anio_fabricacion", year, confidence, value)

    elif name == "ph":
        # PH es la prueba hidrostática: puede venir como fecha o como
        # periodicidad, y son datos distintos.
        date, period = norm.normalize_hydrostatic_test(value)
        if date:
            _set_from_ai(data, "ph", date, confidence, value)
        if period is not None:
            _set_from_ai(data, "ph_periodicidad_anios", period, confidence, value)
        if not date and period is None and not norm.is_absent(value):
            result.warnings.append(
                f"El campo PH no se pudo interpretar: '{value}'. Se espera una "
                "fecha (mes/año) o una periodicidad ('CADA 10 AÑOS')."
            )

    elif name == "peso":
        _set_from_ai(data, "peso", norm.normalize_weight_kg(value), confidence, value)

    elif name == "litros":
        _set_from_ai(data, "litros", norm.normalize_litres(value), confidence, value)

    elif name == "m3":
        _set_from_ai(data, "m3", norm.normalize_cubic_metres(value), confidence, value)

    elif name == "unidad_medida":
        _set_from_ai(data, "unidad_medida", norm.normalize_measure_unit(value), confidence, value)

    elif name == "procedencia":
        _set_from_ai(data, "procedencia", norm.normalize_country(value), confidence, value)

    elif name in ("ancho_diametro", "largo_altura"):
        expected = (norm.DIAMETER_RANGE_CM if name == "ancho_diametro"
                    else norm.HEIGHT_RANGE_CM)
        converted, warning = norm.normalize_length_cm(value, expected_range=expected)
        _set_from_ai(data, name, converted, confidence, value)
        if warning:
            result.warnings.append(f"{name}: {warning}")


def _infer_unlabelled(result: ParseResult, text: str, confidence: float) -> None:
    """Deduce campos por la forma del token cuando no hay etiqueta."""
    data = result.data
    cleaned = norm.clean_text(text)
    # Sólo los campos textuales ambiguos exigen confianza alta.
    textual_ok = confidence >= MIN_TEXTUAL_OCR_CONFIDENCE

    match = _RE_WEIGHT.search(cleaned)
    if match:
        _set_from_ai(data, "peso", norm.normalize_weight_kg(match.group(0)),
                     confidence * 0.9, text)
        _set_from_ai(data, "unidad_medida", "KG", confidence * 0.8, text)

    match = _RE_LITRES.search(cleaned)
    if match:
        _set_from_ai(data, "litros", norm.normalize_litres(match.group(1)),
                     confidence * 0.9, text)

    match = _RE_CUBIC.search(cleaned)
    if match:
        _set_from_ai(data, "m3", norm.normalize_cubic_metres(match.group(1)),
                     confidence * 0.9, text)
        _set_from_ai(data, "unidad_medida", "M3", confidence * 0.8, text)

    gas = norm.normalize_gas(cleaned)
    if gas:
        _set_from_ai(data, "tipo_gas", gas, confidence * 0.9, text)

    color = norm.normalize_color(cleaned) if textual_ok else None
    if color:
        _set_from_ai(data, "color", color, confidence * 0.85, text)

    country = norm.normalize_country(cleaned, strict=True) if textual_ok else None
    if country:
        _set_from_ai(data, "procedencia", country, confidence * 0.85, text)

    # Marcas explícitas sin dos puntos: son la señal más fiable disponible.
    match = _RE_PH_MARKED.search(cleaned)
    if match:
        hydro = norm.normalize_month_year(match.group(1))
        if hydro:
            _set_from_ai(data, "ph", hydro, confidence * 0.9, text)

    match = _RE_FAB_MARKED.search(cleaned)
    if match:
        made = norm.normalize_month_year(match.group(1))
        if made:
            _set_from_ai(data, "fecha_fabricacion", made, confidence * 0.9, text)
            _set_from_ai(data, "anio_fabricacion", int(made[:4]), confidence * 0.9, text)

    # Fechas sin etiqueta: la primera se toma como fabricación y la segunda
    # como prueba hidrostática, que es el orden en que se estampan. Es una
    # convención, así que la confianza se reduce y se deja constancia.
    dates = _RE_MONTH_YEAR.findall(cleaned)
    if dates:
        ordered = sorted(f"{year}-{int(month):02d}" for month, year in dates)
        if not data.fecha_fabricacion.is_present:
            _set_from_ai(data, "fecha_fabricacion", ordered[0],
                         confidence * 0.7, text)
            _set_from_ai(data, "anio_fabricacion", int(ordered[0][:4]),
                         confidence * 0.7, text)
        if len(ordered) > 1 and not data.ph.is_present:
            _set_from_ai(data, "ph", ordered[-1], confidence * 0.6, text)
            result.warnings.append(
                "La fecha de prueba hidrostática se dedujo por orden de aparición, "
                "sin etiqueta explícita."
            )

    if not data.marca.is_present and textual_ok:
        for token in _RE_BRAND.findall(cleaned):
            if token in _BRAND_STOPWORDS or token.isdigit():
                continue
            if len(token) < 2:
                continue
            _set_from_ai(data, "marca", token, confidence * 0.55, text)
            break


def parse_fields(lines: list[tuple[str, float]]) -> ParseResult:
    """Convierte las líneas OCR en un CylinderData parcial.

    `lines` son pares (texto, confianza). El número de serie no se resuelve
    aquí: lo hace el pipeline con app.parsing.serial, que además lo contrasta
    contra el catálogo.
    """
    result = ParseResult()

    labelled_lines: list[tuple[str, float]] = []
    for text, confidence in lines:
        if not text or not text.strip():
            continue
        labelled = _label_of(text)
        if labelled:
            name, value = labelled
            if name != "numero_serie":
                _assign_labelled(result, name, value, confidence)
            labelled_lines.append((text, confidence))

    # Segunda pasada sobre todo el texto, para los campos aún vacíos.
    for text, confidence in lines:
        if text and text.strip():
            _infer_unlabelled(result, text, confidence)

    return result
