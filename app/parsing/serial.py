"""Lectura, corrección y validación del número de serie.

El número de serie es el identificador del activo: si se lee mal, todo el
sistema de trazabilidad registra el movimiento en el cilindro equivocado, que
es peor que no registrarlo. De ahí que este módulo sea deliberadamente
conservador.

Única muestra confirmada del inventario (Descripción.docx, Cilindro_002):

    19S206055   ->  2 dígitos + 1 letra + 6 dígitos

Los dos primeros dígitos coinciden con el año de fabricación (09/2019), lo que
sugiere que el prefijo lo codifica. Con UNA sola muestra eso es una hipótesis,
no una regla: por eso el patrón es configurable y el validador acepta por
defecto una familia amplia de formatos. Cuando el catálogo completo de los 100
cilindros esté cargado, `SerialPattern.infer_from_catalog` deriva el patrón
real de los datos en lugar de mantenerlo escrito a mano.

Sobre la corrección de confusiones OCR: no se aplica a ciegas. Un carácter
sólo se sustituye cuando el patrón esperado exige un tipo distinto en esa
posición concreta. Cambiar "S" por "5" en una posición que admite letras
destruiría un serial correcto.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Confusiones frecuentes del OCR sobre caracteres troquelados en metal.
# Se leen como: al esperar un dígito, esta letra probablemente era ese dígito.
LETTER_TO_DIGIT: dict[str, str] = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1", "T": "7",
    "Z": "2", "E": "3", "A": "4",
    "S": "5", "G": "6", "B": "8",
}

DIGIT_TO_LETTER: dict[str, str] = {
    "0": "O", "1": "I", "2": "Z", "3": "E",
    "4": "A", "5": "S", "6": "G", "7": "T", "8": "B",
}

# Pares mutuamente confundibles, usados para ponderar la distancia de edición
# al resolver contra el catálogo.
CONFUSABLE_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(pair) for pair in
    [("O", "0"), ("Q", "0"), ("D", "0"), ("I", "1"), ("L", "1"), ("T", "7"),
     ("Z", "2"), ("E", "3"), ("A", "4"), ("S", "5"), ("G", "6"), ("B", "8"),
     ("U", "V"), ("M", "N"), ("C", "G"), ("5", "6"), ("8", "0")]
)

# Número mínimo de seriales necesarios para fijar una máscara posicional.
# Con pocas muestras, deducir que "la posición 2 es siempre una letra" es
# sobreajuste: bastaría un cilindro de otro lote para invalidarlo, y la máscara
# se usa para CORREGIR caracteres, así que equivocarse ahí corrompe lecturas
# que eran correctas.
MIN_SAMPLES_FOR_MASK = 8

# Ruido habitual alrededor del serial. La etiqueta puede venir como
# "N° DE SERIE:", "NRO SERIE" o "S/N", así que se elimina por pasadas
# sucesivas en lugar de con un único patrón.
_LABEL_WORD = re.compile(
    r"^\s*(?:N[°º]?|NO|NRO|NUM(?:ERO)?|DE|DEL|SERIE|SERIAL|SER|S\s*/\s*N|SN)"
    r"\b[\s.:\-]*",
    re.IGNORECASE,
)
_ALLOWED_CHARS = re.compile(r"[^A-Z0-9]")


@dataclass(frozen=True)
class SerialPattern:
    """Formato esperado del número de serie.

    `mask` describe el tipo de cada posición: 'D' dígito, 'A' letra,
    '?' cualquiera. Si es None no se asume estructura posicional y sólo se
    aplican las restricciones de longitud.
    """

    mask: str | None = None
    min_length: int = 5
    max_length: int = 14
    regex: str | None = None

    @property
    def compiled(self) -> re.Pattern[str] | None:
        return re.compile(self.regex) if self.regex else None

    def expected_kind(self, position: int) -> str:
        if not self.mask or position >= len(self.mask):
            return "?"
        return self.mask[position]

    def matches(self, serial: str) -> bool:
        if not (self.min_length <= len(serial) <= self.max_length):
            return False
        compiled = self.compiled
        if compiled and not compiled.fullmatch(serial):
            return False
        if self.mask:
            if len(serial) != len(self.mask):
                return False
            for char, kind in zip(serial, self.mask, strict=True):
                if kind == "D" and not char.isdigit():
                    return False
                if kind == "A" and not char.isalpha():
                    return False
        return True

    @classmethod
    def infer_from_catalog(cls, serials: list[str]) -> SerialPattern:
        """Deriva el patrón a partir de los seriales reales del inventario.

        Sólo fija una máscara posicional si hay muestras suficientes y TODOS
        los seriales comparten la misma longitud y el mismo tipo en cada
        posición. Ante pocas muestras o cualquier variación se queda con los
        límites de longitud, que es lo único que los datos respaldan.
        """
        cleaned = [normalize_serial_charset(s) for s in serials if s]
        cleaned = [s for s in cleaned if s]
        if not cleaned:
            return cls()

        lengths = Counter(len(s) for s in cleaned)
        min_len, max_len = min(lengths), max(lengths)

        mask: str | None = None
        if len(lengths) == 1 and len(cleaned) >= MIN_SAMPLES_FOR_MASK:
            length = min_len
            kinds: list[str] = []
            for index in range(length):
                column = {s[index] for s in cleaned}
                if all(c.isdigit() for c in column):
                    kinds.append("D")
                elif all(c.isalpha() for c in column):
                    kinds.append("A")
                else:
                    kinds.append("?")
            mask = "".join(kinds)

        return cls(mask=mask, min_length=min_len, max_length=max_len)


# Patrón por defecto: sin máscara posicional. Deliberadamente permisivo,
# porque una sola muestra no justifica imponer una estructura rígida.
DEFAULT_PATTERN = SerialPattern(min_length=6, max_length=12)


def normalize_serial_charset(text: str) -> str:
    """Deja sólo caracteres válidos, en mayúsculas y sin la etiqueta previa."""
    if not text:
        return ""

    upper = text.strip().upper()

    # Cuando hay dos puntos, el serial está a la derecha: "N° DE SERIE: 19S...".
    if ":" in upper:
        upper = upper.rsplit(":", 1)[1]

    # Restos de etiqueta sin dos puntos, p. ej. "SERIE 19S206055".
    previous = None
    while previous != upper:
        previous = upper
        upper = _LABEL_WORD.sub("", upper)

    return _ALLOWED_CHARS.sub("", upper)


@dataclass
class SerialCandidate:
    """Serial propuesto a partir de una línea OCR."""

    text: str
    confidence: float
    raw: str
    corrections: list[str] = field(default_factory=list)
    pattern_valid: bool = False

    @property
    def was_corrected(self) -> bool:
        return bool(self.corrections)


def apply_pattern_corrections(serial: str, pattern: SerialPattern) -> tuple[str, list[str]]:
    """Corrige confusiones OCR guiándose por el tipo esperado de cada posición.

    Si el patrón no define máscara, no se corrige nada: sin saber qué tipo se
    espera, cualquier sustitución sería una suposición.
    """
    if not pattern.mask or len(serial) != len(pattern.mask):
        return serial, []

    corrected: list[str] = []
    notes: list[str] = []

    for index, char in enumerate(serial):
        kind = pattern.expected_kind(index)
        if kind == "D" and char.isalpha() and char in LETTER_TO_DIGIT:
            replacement = LETTER_TO_DIGIT[char]
            notes.append(f"pos {index}: '{char}' -> '{replacement}' (se esperaba dígito)")
            corrected.append(replacement)
        elif kind == "A" and char.isdigit() and char in DIGIT_TO_LETTER:
            replacement = DIGIT_TO_LETTER[char]
            notes.append(f"pos {index}: '{char}' -> '{replacement}' (se esperaba letra)")
            corrected.append(replacement)
        else:
            corrected.append(char)

    return "".join(corrected), notes


def extract_serial_candidates(
    lines: list[tuple[str, float]],
    pattern: SerialPattern = DEFAULT_PATTERN,
) -> list[SerialCandidate]:
    """Obtiene candidatos a número de serie de las líneas reconocidas.

    Se consideran tanto la línea completa como sus tokens, porque el OCR puede
    devolver "SERIE 19S206055" en una sola línea o partido en varias.
    """
    candidates: list[SerialCandidate] = []
    seen: set[str] = set()

    for raw_text, confidence in lines:
        if not raw_text:
            continue

        fragments = [raw_text, *re.split(r"[\s|/\\]+", raw_text)]
        for fragment in fragments:
            normalized = normalize_serial_charset(fragment)
            if not normalized or len(normalized) < pattern.min_length:
                continue
            if len(normalized) > pattern.max_length:
                continue
            # Un serial mezcla dígitos; una palabra suelta no lo es.
            if not any(ch.isdigit() for ch in normalized):
                continue

            corrected, notes = apply_pattern_corrections(normalized, pattern)
            if corrected in seen:
                continue
            seen.add(corrected)

            candidates.append(
                SerialCandidate(
                    text=corrected,
                    confidence=confidence,
                    raw=raw_text.strip(),
                    corrections=notes,
                    pattern_valid=pattern.matches(corrected),
                )
            )

    # Prioriza los que encajan con el patrón; a igualdad, la mayor confianza.
    candidates.sort(key=lambda c: (c.pattern_valid, c.confidence), reverse=True)
    return candidates


def weighted_edit_distance(left: str, right: str) -> float:
    """Distancia de edición que penaliza menos las confusiones típicas del OCR.

    Sustituir 'S' por '5' cuesta 0.5 en lugar de 1.0, porque es un error de
    lectura esperable y no evidencia de que sean cilindros distintos. Así, al
    resolver contra el catálogo, un serial mal leído se acerca a su original
    más que a otro serial genuinamente diferente.
    """
    if left == right:
        return 0.0
    if not left:
        return float(len(right))
    if not right:
        return float(len(left))

    # Los costes son fraccionarios (una confusión típica cuesta 0.5), así que
    # la fila acumulada es de floats desde el principio.
    previous: list[float] = [float(index) for index in range(len(right) + 1)]
    for i, char_left in enumerate(left, start=1):
        current: list[float] = [float(i)]
        for j, char_right in enumerate(right, start=1):
            if char_left == char_right:
                substitution_cost = 0.0
            elif frozenset((char_left, char_right)) in CONFUSABLE_PAIRS:
                substitution_cost = 0.5
            else:
                substitution_cost = 1.0
            current.append(min(
                previous[j] + 1.0,              # eliminación
                current[j - 1] + 1.0,           # inserción
                previous[j - 1] + substitution_cost,
            ))
        previous = current
    return previous[-1]


def consensus_candidates(
    readings: list[list[tuple[str, float]]],
    pattern: SerialPattern = DEFAULT_PATTERN,
) -> list[SerialCandidate]:
    """Combina las lecturas de varias variantes de realce en un solo ranking.

    Motivo: la confianza que devuelve el motor OCR no es comparable entre
    variantes de preprocesado. En pruebas sobre troquelado, una variante llegó
    a puntuar 0.99 una lectura con un carácter perdido, mientras otra acertaba
    con 0.97. Quedarse con el máximo elige la lectura equivocada.

    El acuerdo entre transformaciones independientes es una señal más fiable:
    si tres realces distintos leen lo mismo, es mucho más probable que sea
    correcto que si uno solo lo afirma con seguridad. La puntuación final pesa
    ese acuerdo junto a la confianza media y la validez del formato.
    """
    if not readings:
        return []

    total_variants = len([r for r in readings if r]) or 1
    votes: dict[str, list[SerialCandidate]] = {}

    for variant_lines in readings:
        if not variant_lines:
            continue
        seen_in_variant: set[str] = set()
        for candidate in extract_serial_candidates(variant_lines, pattern):
            # Un candidato cuenta como un voto por variante, no por línea.
            if candidate.text in seen_in_variant:
                continue
            seen_in_variant.add(candidate.text)
            votes.setdefault(candidate.text, []).append(candidate)

    aggregated: list[SerialCandidate] = []
    for text, occurrences in votes.items():
        agreement = len(occurrences) / total_variants
        mean_confidence = sum(c.confidence for c in occurrences) / len(occurrences)
        pattern_valid = any(c.pattern_valid for c in occurrences)

        score = 0.5 * agreement + 0.5 * mean_confidence
        if pattern_valid:
            score = min(1.0, score * 1.05)

        best_occurrence = max(occurrences, key=lambda c: c.confidence)
        aggregated.append(
            SerialCandidate(
                text=text,
                confidence=round(min(1.0, score), 4),
                raw=best_occurrence.raw,
                corrections=best_occurrence.corrections,
                pattern_valid=pattern_valid,
            )
        )

    aggregated.sort(key=lambda c: (c.pattern_valid, c.confidence), reverse=True)
    return aggregated
