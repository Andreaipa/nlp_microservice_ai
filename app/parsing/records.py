"""Normalización de registros de cilindro.

Un registro de cilindro puede llegar de dos sitios:

* las fichas `Descripción.docx` del dataset, leídas por
  `training/build_catalog.py`;
* la tabla `cilindros` de Supabase, que la aplicación móvil alimenta cuando
  el operario registra un envase (ver app/services/catalog_sync.py).

Las dos fuentes escriben los mismos campos de formas distintas: el gas como
"DIÓXIDO DE CARBONO" o "CO2", la altura en metros o en centímetros, la prueba
hidrostática como fecha o como periodicidad. Esta función los lleva todos al
mismo formato, para que el catálogo que usa el servicio sea uniforme venga de
donde venga cada cilindro.
"""
from __future__ import annotations

from app.parsing import normalizers as norm
from app.parsing.serial import MIN_RELIABLE_SERIAL_LENGTH, normalize_serial_charset


def normalize_record(cylinder: str, raw: dict[str, str]) -> tuple[dict, list[str]]:
    """Aplica los normalizadores del servicio y recoge las incidencias."""
    issues: list[str] = []
    record: dict = {}

    serial = normalize_serial_charset(raw.get("numero_serie", ""))
    if not serial:
        issues.append(f"{cylinder}: ficha sin número de serie legible.")
        return {}, issues
    record["numero_serie"] = serial
    if len(serial) < MIN_RELIABLE_SERIAL_LENGTH:
        issues.append(
            f"{cylinder}: el serial '{serial}' tiene sólo {len(serial)} caracteres; "
            "es demasiado corto para identificar el cilindro de forma fiable."
        )

    if (marca := raw.get("marca")) and not norm.is_absent(marca):
        record["marca"] = norm.clean_text(marca)

    if fab := raw.get("fecha_fabricacion"):
        if month_year := norm.normalize_month_year(fab):
            record["fecha_fabricacion"] = month_year
            record["anio_fabricacion"] = int(month_year[:4])
        elif year := norm.normalize_year(fab):
            record["anio_fabricacion"] = year
        elif not norm.is_absent(fab):
            issues.append(f"{cylinder}: fecha de fabricación no interpretable: '{fab}'.")

    if gas := raw.get("tipo_gas"):
        if code := norm.normalize_gas(gas):
            record["tipo_gas"] = code
        elif not norm.is_absent(gas):
            issues.append(f"{cylinder}: tipo de gas no reconocido: '{gas}'.")

    if (unit := raw.get("unidad_medida")) and (
        code := norm.normalize_measure_unit(unit)
    ):
        record["unidad_medida"] = code

    if peso := raw.get("peso"):
        if value := norm.normalize_weight_kg(peso):
            record["peso"] = value
        elif not norm.is_absent(peso):
            issues.append(f"{cylinder}: peso fuera de rango o ilegible: '{peso}'.")

    if (litros := raw.get("litros")) and (value := norm.normalize_litres(litros)):
        record["litros"] = value

    if (m3 := raw.get("m3")) and (value := norm.normalize_cubic_metres(m3)):
        record["m3"] = value

    for field_name, expected in (
        ("ancho_diametro", norm.DIAMETER_RANGE_CM),
        ("largo_altura", norm.HEIGHT_RANGE_CM),
    ):
        if not (text := raw.get(field_name)):
            continue
        value, warning = norm.normalize_length_cm(text, expected_range=expected)
        if value is not None:
            record[field_name] = value
        if warning:
            issues.append(f"{cylinder}: {field_name}: {warning}")

    if color := raw.get("color"):
        if colour := norm.normalize_color(color):
            record["color"] = colour
        elif not norm.is_absent(color):
            issues.append(f"{cylinder}: color no reconocido: '{color}'.")

    if ph := raw.get("ph"):
        date, period = norm.normalize_hydrostatic_test(ph)
        if date:
            record["ph"] = date
        if period is not None:
            record["ph_periodicidad_anios"] = period
        if not date and period is None and not norm.is_absent(ph):
            issues.append(f"{cylinder}: campo PH no interpretable: '{ph}'.")

    if origin := raw.get("procedencia"):
        if code := norm.normalize_country(origin):
            record["procedencia"] = code
        elif not norm.is_absent(origin):
            issues.append(f"{cylinder}: procedencia no reconocida: '{origin}'.")

    return record, issues
