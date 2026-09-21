"""Normalización de los campos leídos por OCR.

Objetivo: convertir texto crudo en valores tipados y comparables, sin inventar
información. Cuando un texto no se puede interpretar con seguridad, la función
devuelve None y el campo queda como desconocido.
"""
from __future__ import annotations

import re
import unicodedata

# --- Texto ------------------------------------------------------------------

def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def clean_text(text: str) -> str:
    """Normaliza espacios, acentos y mayúsculas para comparar de forma estable."""
    return re.sub(r"\s+", " ", strip_accents(text)).strip().upper()


# --- Dato ausente -----------------------------------------------------------
# En las fichas del inventario "SN" ("sin número"/"sin dato") se usa como
# marcador de campo vacío, no como valor. Aparece en marca, PH y procedencia.
ABSENT_MARKERS: frozenset[str] = frozenset({
    "SN", "S/N", "S N", "NA", "N/A", "NO", "NINGUNO", "SIN DATO", "-", "--", "",
})


def is_absent(text: str) -> bool:
    return clean_text(text) in ABSENT_MARKERS


# --- Gas --------------------------------------------------------------------
# Se normaliza a un código corto y estable. Las variantes cubren tanto el
# nombre completo estampado como la fórmula.
GAS_ALIASES: dict[str, tuple[str, ...]] = {
    "CO2": ("CO2", "DIOXIDO DE CARBONO", "ANHIDRIDO CARBONICO", "C02", "CARBONICO"),
    "O2": ("O2", "OXIGENO", "OXIGENO MEDICINAL", "02"),
    "AR": ("AR", "ARGON"),
    "N2": ("N2", "NITROGENO"),
    "ACET": ("ACETILENO", "C2H2", "ACET"),
    "H2": ("H2", "HIDROGENO"),
    "HE": ("HE", "HELIO"),
    "AIR": ("AIRE", "AIRE COMPRIMIDO", "AIR"),
    "MIX": ("MEZCLA", "MIX", "ARGAMIX", "ARGOMIX"),
}


# Una mezcla nombra sus componentes ("80% ARGON 20% DIOXIDO DE CARBONO"), de
# modo que una búsqueda por subcadena la clasificaría como argón puro. Se
# detecta antes que nada.
_MIXTURE = re.compile(r"\d{1,3}\s*%|MEZCLA|MIX")


def normalize_gas(text: str) -> str | None:
    cleaned = clean_text(text)
    if not cleaned or is_absent(cleaned):
        return None

    if _MIXTURE.search(cleaned):
        return "MIX"

    # Coincidencia exacta antes que parcial: evita que "AIRE" case con "AR".
    for code, aliases in GAS_ALIASES.items():
        if cleaned in aliases:
            return code

    # Si el texto nombra más de un gas y no se marcó como mezcla, sigue siendo
    # una mezcla: devolver uno de los dos sería inventar.
    matched = {
        code for code, aliases in GAS_ALIASES.items()
        if any(len(alias) >= 3 and alias in cleaned for alias in aliases)
    }
    if len(matched) > 1:
        return "MIX"
    if len(matched) == 1:
        return matched.pop()
    return None


# --- Color ------------------------------------------------------------------
COLOR_ALIASES: dict[str, tuple[str, ...]] = {
    "gris": ("GRIS", "PLOMO", "GREY", "GRAY"),
    "negro": ("NEGRO", "BLACK"),
    "verde": ("VERDE", "GREEN"),
    "azul": ("AZUL", "BLUE"),
    "blanco": ("BLANCO", "WHITE"),
    "rojo": ("ROJO", "RED"),
    "amarillo": ("AMARILLO", "YELLOW"),
    "naranja": ("NARANJA", "ANARANJADO", "ORANGE"),
    "marron": ("MARRON", "MAARON", "MARON", "CAFE", "BROWN"),
}

# Separadores de un color compuesto, p. ej. "MARRON/GRIS".
_COLOR_SEPARATOR = re.compile(r"\s*(?:/|-|\bY\b|\bCON\b)\s*")


def normalize_color(text: str) -> str | None:
    """Normaliza el color, conservando los compuestos como 'marron/gris'.

    Un cilindro bicolor es información real (suele indicar repintado o cambio
    de servicio), así que se conserva en lugar de quedarse con el primero.
    """
    cleaned = clean_text(text)
    if not cleaned or is_absent(cleaned):
        return None

    parts: list[str] = []
    for chunk in _COLOR_SEPARATOR.split(cleaned):
        for canonical, aliases in COLOR_ALIASES.items():
            if any(alias in chunk for alias in aliases) and canonical not in parts:
                parts.append(canonical)
                break
    if parts:
        return "/".join(parts)
    return None


# --- Fechas -----------------------------------------------------------------
_MONTH_YEAR = re.compile(r"\b(0?[1-9]|1[0-2])\s*[/\-.]\s*((?:19|20)\d{2})\b")
_YEAR_MONTH = re.compile(r"\b((?:19|20)\d{2})\s*[/\-.]\s*(0?[1-9]|1[0-2])\b")
_YEAR_ONLY = re.compile(r"\b((?:19|20)\d{2})\b")
# El troquelado suele abreviar el año a dos dígitos: "02/25".
_MONTH_SHORT_YEAR = re.compile(r"\b(0?[1-9]|1[0-2])\s*[/\-.]\s*(\d{2})\b")

# Ventana de interpretación para años de dos dígitos. Un cilindro en servicio
# no es anterior a los años 70, y una prueba hidrostática puede estar fechada
# unos años por delante de hoy si se registró la próxima revisión.
SHORT_YEAR_PIVOT = 70


def _expand_short_year(value: int) -> int:
    return 1900 + value if value >= SHORT_YEAR_PIVOT else 2000 + value


def normalize_month_year(text: str) -> str | None:
    """Extrae mes/año y lo devuelve como 'AAAA-MM'.

    Los cilindros estampan tanto la fecha de fabricación como la de la prueba
    hidrostática en formato mes/año. Se acepta también año/mes porque el orden
    varía entre fabricantes.
    """
    cleaned = clean_text(text)

    match = _MONTH_YEAR.search(cleaned)
    if match:
        return f"{match.group(2)}-{int(match.group(1)):02d}"

    match = _YEAR_MONTH.search(cleaned)
    if match:
        return f"{match.group(1)}-{int(match.group(2)):02d}"

    match = _MONTH_SHORT_YEAR.search(cleaned)
    if match:
        year = _expand_short_year(int(match.group(2)))
        return f"{year}-{int(match.group(1)):02d}"
    return None


def normalize_year(text: str) -> int | None:
    month_year = normalize_month_year(text)
    if month_year:
        return int(month_year.split("-")[0])

    match = _YEAR_ONLY.search(clean_text(text))
    if match:
        return int(match.group(1))
    return None


# --- Magnitudes -------------------------------------------------------------
_NUMBER = re.compile(r"(\d+(?:[.,]\d+)?)")


def _parse_number(text: str) -> float | None:
    match = _NUMBER.search(text.replace(" ", ""))
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def normalize_weight_kg(text: str) -> float | None:
    """Peso en kilogramos. Convierte desde libras si la unidad lo indica."""
    cleaned = clean_text(text)
    value = _parse_number(cleaned)
    if value is None:
        return None
    if "LB" in cleaned or "POUND" in cleaned:
        value *= 0.45359237
    return round(value, 2) if 0 < value < 500 else None


def normalize_litres(text: str) -> float | None:
    value = _parse_number(clean_text(text))
    if value is None:
        return None
    return round(value, 2) if 0 < value <= 500 else None


def normalize_cubic_metres(text: str) -> float | None:
    value = _parse_number(clean_text(text))
    if value is None:
        return None
    return round(value, 3) if 0 < value <= 100 else None


# Rangos físicos admisibles para un cilindro industrial. Sirven para decidir si
# un número leído es plausible y para corregir unidades, no para rechazar datos
# del backend.
DIAMETER_RANGE_CM = (10.0, 60.0)
HEIGHT_RANGE_CM = (30.0, 200.0)


def normalize_length_cm(
    text: str, *, expected_range: tuple[float, float]
) -> tuple[float | None, str | None]:
    """Convierte una longitud a centímetros y corrige unidades incoherentes.

    El documento de descripción del cilindro 002 registra "LARGO ALTURA: 1.21
    CM", que para un envase de 40 litros es imposible: el valor real son 1.21
    metros. Cuando el número queda fuera del rango físico se prueban las
    conversiones habituales (m -> cm, mm -> cm, pulgadas -> cm) y se devuelve
    un aviso describiendo la corrección aplicada.
    """
    cleaned = clean_text(text)
    value = _parse_number(cleaned)
    if value is None:
        return None, None

    low, high = expected_range
    if low <= value <= high:
        return round(value, 2), None

    candidates: list[tuple[float, str]] = [
        (value * 100.0, "se interpretó como metros"),
        (value / 10.0, "se interpretó como milímetros"),
        (value * 2.54, "se interpretó como pulgadas"),
    ]
    for converted, explanation in candidates:
        if low <= converted <= high:
            return (
                round(converted, 2),
                f"valor fuera de rango ({value}); {explanation} y se convirtió a cm",
            )

    return None, f"valor fuera de rango físico y sin conversión plausible: {value}"


# --- Otros ------------------------------------------------------------------
KNOWN_COUNTRY_CODES = frozenset({
    "CHN", "PER", "USA", "BRA", "ARG", "ESP", "ITA", "DEU", "IND", "JPN",
    "KOR", "MEX", "CHL", "COL", "FRA", "GBR", "TUR", "THA", "VNM",
})

# Las fichas alternan el código ISO de tres letras, el de dos y el nombre del
# país en castellano o en inglés. Todo se unifica al código de tres letras.
COUNTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "CHN": ("CHN", "CN", "CHINA"),
    "USA": ("USA", "US", "EEUU", "EE UU", "ESTADOS UNIDOS", "UNITED STATES"),
    "BRA": ("BRA", "BR", "BRASIL", "BRAZIL"),
    "ARG": ("ARG", "AR", "ARGENTINA"),
    "PER": ("PER", "PE", "PERU"),
    "ESP": ("ESP", "ES", "ESPANA"),
    "ITA": ("ITA", "IT", "ITALIA", "ITALY"),
    "DEU": ("DEU", "DE", "ALEMANIA", "GERMANY"),
    "IND": ("IND", "INDIA"),
    "JPN": ("JPN", "JP", "JAPON", "JAPAN"),
    "KOR": ("KOR", "KR", "COREA", "KOREA"),
    "MEX": ("MEX", "MX", "MEXICO"),
    "CHL": ("CHL", "CL", "CHILE"),
    "COL": ("COL", "CO", "COLOMBIA"),
}

_WORD = re.compile(r"[A-Z]+")


def normalize_country(text: str, *, strict: bool = False) -> str | None:
    """Normaliza la procedencia a un código ISO de tres letras.

    `strict` excluye los códigos de dos letras. Es necesario cuando el texto
    no viene rotulado como procedencia: en castellano "DE" es una preposición,
    y sin esta restricción la lectura "DIÓXIDO DE CARBONO" acaba clasificada
    como procedencia Alemania. Se observó exactamente ese fallo sobre una
    fotografía del dataset.

    Con la etiqueta "PROCEDENCIA:" delante no hay ambigüedad y se aceptan
    también los códigos de dos letras, que es como los escriben 35 de las 100
    fichas del inventario.
    """
    cleaned = clean_text(text)
    if not cleaned or is_absent(cleaned):
        return None

    def allowed(alias: str) -> bool:
        return len(alias) > 2 or not strict

    # Coincidencia exacta con el campo completo: el caso habitual.
    for code, aliases in COUNTRY_ALIASES.items():
        if cleaned in aliases and allowed(cleaned):
            return code

    # Token a token, exigiendo palabra completa para que "CO" no case dentro
    # de "COLOR".
    tokens = _WORD.findall(cleaned)
    for code, aliases in COUNTRY_ALIASES.items():
        for token in tokens:
            if token in aliases and allowed(token):
                return code
    return None


# En las fichas reales el campo PH contiene una fecha ("02/2025") en unos
# cilindros y la periodicidad de la prueba ("CADA 10 AÑOS") en la mayoría.
# Son dos informaciones distintas y se modelan por separado en lugar de
# forzarlas al mismo tipo.
_PH_PERIOD = re.compile(r"CADA\s*(\d{1,2})\s*(?:ANOS?|ANIOS?|YEARS?)")


def normalize_hydrostatic_test(text: str) -> tuple[str | None, int | None]:
    """Interpreta el campo PH.

    Devuelve (fecha AAAA-MM, periodicidad en años). Cualquiera de los dos
    puede ser None: una ficha que sólo indica "CADA 10 AÑOS" no dice cuándo se
    hizo la última prueba, y afirmar una fecha a partir de ahí sería inventar.
    """
    if not text or is_absent(text):
        return None, None

    cleaned = clean_text(text)
    date = normalize_month_year(cleaned)

    period_match = _PH_PERIOD.search(cleaned)
    period = int(period_match.group(1)) if period_match else None

    return date, period


def normalize_measure_unit(text: str) -> str | None:
    cleaned = clean_text(text)
    if is_absent(cleaned):
        return None
    if re.search(r"\bM3\b|\bM\^?3\b|METRO.?S? CUBICO", cleaned):
        return "M3"
    if re.search(r"\bKG\b|KILO", cleaned):
        return "KG"
    return None
