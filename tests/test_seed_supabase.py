"""Pruebas del alta del dataset en la tabla `cilindros` de la aplicación."""
from __future__ import annotations

import pytest

from training.seed_supabase import to_app_payload


def test_gas_codes_become_the_names_the_app_form_uses():
    """La app guarda 'DIÓXIDO DE CARBONO', no 'CO2'."""
    for code, name in [("CO2", "DIÓXIDO DE CARBONO"), ("O2", "OXÍGENO"),
                       ("AR", "ARGÓN"), ("N2", "NITRÓGENO"), ("MIX", "MEZCLA")]:
        assert to_app_payload({"numero_serie": "X1", "tipo_gas": code}, 1)["tipo_gas"] == name


def test_acetylene_is_registered_by_content_like_the_form():
    payload = to_app_payload({"numero_serie": "X1", "tipo_gas": "ACET", "litros": 6.0}, 1)
    assert payload["litros"] is None
    assert payload["contenido"] == pytest.approx(6.0)


@pytest.mark.parametrize("record,expected", [
    ({"ph_periodicidad_anios": 10}, "CADA 10 AÑOS PH"),
    ({"ph": "2025-02"}, "02/2025"),
    ({"tipo_gas": "CO2"}, "CADA 5 AÑOS PH"),     # regla por defecto de la app
    ({"tipo_gas": "O2"}, "CADA 10 AÑOS PH"),
])
def test_hydrostatic_test_follows_the_app_rules(record, expected):
    assert to_app_payload({"numero_serie": "X1", **record}, 1)["ph"] == expected


def test_manufacturing_date_uses_month_and_year():
    payload = to_app_payload({"numero_serie": "X1", "fecha_fabricacion": "2019-09"}, 1)
    assert payload["anio_fabricacion"] == "09/2019"


def test_serial_is_normalized_and_owner_is_kept():
    payload = to_app_payload({"numero_serie": " 19s206055 ", "color": "gris"}, 12)
    assert payload["numero_serie"] == "19S206055"
    assert payload["cliente_id"] == 12
    assert payload["color"] == "GRIS"
