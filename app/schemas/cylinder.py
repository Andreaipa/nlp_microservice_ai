"""Modelo de datos del cilindro.

Los campos reflejan exactamente la información estampada en el collarín de los
cilindros de Electrametal, según `Descripción.docx` del dataset:

    N° DE SERIE: 19S206055        MARCA: JP5
    AÑO DE FABRICACIÓN: 09/2019   TIPO DE GAS: DIÓXIDO DE CARBONO CO2
    UNIDAD DE MEDIDA: KG          PESO: 45.2 KG
    ANCHO DIÁMETRO: 22 CM         LARGO ALTURA: 1.21 CM (sic)
    LITROS: 40.4 L                COLOR: GRIS
    PH: 02/2025                   PROCEDENCIA: CHN

Al revisar las 100 fichas del inventario aparecieron variantes que la ficha
del cilindro 002 no anticipaba: el campo PH contiene una periodicidad ("CADA
10 AÑOS") en 64 de los 100 cilindros en lugar de una fecha, la procedencia
alterna "CHN" y "CN", y "SN" se usa como marcador de dato ausente.

Notas de modelado que se apartan de una lectura literal:

* `ph` NO es acidez química. En un cilindro a presión "PH" es la Prueba
  Hidrostática y su valor es una FECHA (mes/año) de la última prueba superada.
  Se modela como cadena ISO parcial "AAAA-MM".
* `anio_fabricacion` se expone como entero (el año) para respetar el contrato
  pedido, y se añade `fecha_fabricacion` ("AAAA-MM") para no perder el mes.
* `largo_altura` viene en el documento como "1.21 CM", que es físicamente
  imposible para un cilindro de 40 L. El normalizador lo corrige a centímetros
  y deja constancia en `warnings`. Ver app/parsing/normalizers.py.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import TracedValue


class CylinderData(BaseModel):
    """Conjunto completo de atributos de un cilindro, cada uno trazado."""

    # --- Identificación -----------------------------------------------------
    numero_serie: TracedValue[str] = Field(default_factory=TracedValue[str].unknown)
    marca: TracedValue[str] = Field(default_factory=TracedValue[str].unknown)

    # --- Fabricación --------------------------------------------------------
    anio_fabricacion: TracedValue[int] = Field(default_factory=TracedValue[int].unknown)
    fecha_fabricacion: TracedValue[str] = Field(
        default_factory=TracedValue[str].unknown,
        description="Mes y año de fabricación en formato AAAA-MM.",
    )
    procedencia: TracedValue[str] = Field(
        default_factory=TracedValue[str].unknown,
        description="Código de país de origen, p. ej. CHN.",
    )

    # --- Contenido ----------------------------------------------------------
    tipo_gas: TracedValue[str] = Field(
        default_factory=TracedValue[str].unknown,
        description="Código normalizado del gas: CO2, O2, AR, N2, ACET, MIX.",
    )
    unidad_medida: TracedValue[str] = Field(
        default_factory=TracedValue[str].unknown,
        description="Unidad en que se comercializa el contenido: KG o M3.",
    )
    m3: TracedValue[float] = Field(default_factory=TracedValue[float].unknown)
    litros: TracedValue[float] = Field(
        default_factory=TracedValue[float].unknown,
        description="Capacidad hídrica del envase, en litros.",
    )

    # --- Físicos ------------------------------------------------------------
    peso: TracedValue[float] = Field(
        default_factory=TracedValue[float].unknown, description="Peso en kilogramos."
    )
    ancho_diametro: TracedValue[float] = Field(
        default_factory=TracedValue[float].unknown, description="Diámetro en centímetros."
    )
    largo_altura: TracedValue[float] = Field(
        default_factory=TracedValue[float].unknown, description="Altura en centímetros."
    )
    color: TracedValue[str] = Field(default_factory=TracedValue[str].unknown)

    # --- Seguridad ----------------------------------------------------------
    ph: TracedValue[str] = Field(
        default_factory=TracedValue[str].unknown,
        description=(
            "Fecha de la última Prueba Hidrostática en formato AAAA-MM. "
            "No es acidez química."
        ),
    )
    ph_periodicidad_anios: TracedValue[int] = Field(
        default_factory=TracedValue[int].unknown,
        description=(
            "Cada cuántos años debe repetirse la Prueba Hidrostática. En las "
            "fichas del inventario la mayoría de cilindros declara la "
            "periodicidad ('CADA 10 AÑOS') en lugar de la fecha concreta, y "
            "son dos datos distintos: de la periodicidad no se deduce cuándo "
            "se hizo la última prueba."
        ),
    )


# Campos que razonablemente pueden leerse de una fotografía del collarín.
AI_READABLE_FIELDS: frozenset[str] = frozenset({
    "numero_serie",
    "marca",
    "anio_fabricacion",
    "fecha_fabricacion",
    "tipo_gas",
    "procedencia",
    "ph",
    "ph_periodicidad_anios",
    "peso",
    "litros",
    "unidad_medida",
    "color",
})

# Campos que en la práctica deben venir del backend o del catálogo: o no están
# estampados, o su lectura por OCR no es fiable frente a un dato maestro.
BACKEND_OWNED_FIELDS: frozenset[str] = frozenset({
    "m3",
    "ancho_diametro",
    "largo_altura",
})
