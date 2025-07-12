# tests/test_planner_calculations.py

import pytest
from datetime import datetime
import pytz
import numpy as np

# Importamos el módulo completo para probar sus funciones
from astrakairos.planner import calculations

# --- Fixtures de Prueba ---

@pytest.fixture(scope="module")
def paris_observatory():
    """Una fixture de ubicación de Skyfield reutilizable para las pruebas."""
    # Usamos una ubicación conocida para que los resultados sean predecibles.
    # Latitud, Longitud, Altitud para el Observatorio de París.
    return calculations.get_observer_location(48.8368, 2.3358, 67)

@pytest.fixture(scope="module")
def test_date_new_moon():
    """Una fecha de prueba donde hay luna nueva, simplificando algunos tests."""
    return datetime(2024, 1, 11)

@pytest.fixture(scope="module")
def test_date_full_moon():
    """Una fecha de prueba donde hay luna llena, para probar el brillo."""
    return datetime(2024, 1, 25)

# --- Pruebas para get_nightly_events ---

def test_get_nightly_events_returns_correct_structure(paris_observatory, test_date_new_moon):
    """
    Verifies that the function returns a dictionary with all the expected keys
    and that the event times are logical if they exist.
    """
    events = calculations.get_nightly_events(paris_observatory, test_date_new_moon, 'Europe/Paris')
    
    assert isinstance(events, dict)
    expected_keys = [
        'sunset_utc', 'sunset_local', 'sunrise_utc', 'sunrise_local',
        'astronomical_twilight_end', 'astronomical_twilight_start',
        'moonrise_utc', 'moonrise_local', 'moonset_utc', 'moonset_local'
    ]
    for key in expected_keys:
        assert key in events
    
    # --- Assertión Robusta ---
    # Solo comprueba la lógica de los tiempos si ambos eventos (atardecer y amanecer) ocurren.
    # En latitudes altas durante el verano, el atardecer o el amanecer (o los crepúsculos)
    # podrían no ocurrir, y la función devolvería None correctamente.
    if events['sunset_utc'] and events['sunrise_utc']:
        assert events['sunset_utc'] < events['sunrise_utc']
    
    if events['astronomical_twilight_end'] and events['astronomical_twilight_start']:
        assert events['astronomical_twilight_end'] < events['astronomical_twilight_start']

# --- Pruebas para calculate_sky_conditions_at_time ---

def test_sky_conditions_at_time_new_moon(paris_observatory, test_date_new_moon):
    """Verifica las condiciones del cielo durante una luna nueva."""
    # Tomamos la medianoche local en París para la prueba
    local_tz = pytz.timezone('Europe/Paris')
    midnight_local = local_tz.localize(datetime(2024, 1, 12, 0, 0))
    midnight_utc = midnight_local.astimezone(pytz.utc)
    
    conditions = calculations.calculate_sky_conditions_at_time(paris_observatory, midnight_utc)
    
    assert isinstance(conditions, dict)
    # Durante la luna nueva, la fase debe ser muy cercana a 0%
    assert conditions['moon_phase_percent'] < 2.0 
    assert 'zenith_ra_hours' in conditions

def test_sky_conditions_at_time_full_moon(paris_observatory, test_date_full_moon):
    """Verifica las condiciones del cielo durante una luna llena."""
    local_tz = pytz.timezone('Europe/Paris')
    midnight_local = local_tz.localize(datetime(2024, 1, 26, 0, 0))
    midnight_utc = midnight_local.astimezone(pytz.utc)
    
    conditions = calculations.calculate_sky_conditions_at_time(paris_observatory, midnight_utc)
    
    # Durante la luna llena, la fase debe ser muy cercana a 100%
    assert conditions['moon_phase_percent'] > 98.0
    # La luna llena suele estar alta en el cielo a medianoche
    assert conditions['moon_alt_deg'] > 30.0

# --- Pruebas para generate_sky_quality_map ---

def test_sky_quality_map_no_moon(paris_observatory, test_date_new_moon):
    """
    Prueba el mapa de calidad en una noche sin luna. El mejor punto debe ser el cénit.
    """
    local_tz = pytz.timezone('Europe/Paris')
    midnight_local = local_tz.localize(datetime(2024, 1, 12, 0, 0))
    midnight_utc = midnight_local.astimezone(pytz.utc)
    
    # Obtenemos las coordenadas del cénit para comparar
    zenith_conditions = calculations.calculate_sky_conditions_at_time(paris_observatory, midnight_utc)
    
    result_map = calculations.generate_sky_quality_map(
        paris_observatory, midnight_utc, min_altitude_deg=30.0
    )
    
    assert isinstance(result_map, dict)
    # Sin luna, el mejor punto debe estar muy cerca del cénit (la mayor altitud)
    assert np.isclose(result_map['best_alt_deg'], 85.0, atol=5.0) # El grid es de 5 grados
    assert np.isclose(result_map['best_ra_hours'], zenith_conditions['zenith_ra_hours'], atol=0.5)

def test_sky_quality_map_with_moon(paris_observatory, test_date_full_moon):
    """
    Prueba el mapa de calidad en una noche con luna. El mejor punto debe estar lejos de la luna.
    """
    local_tz = pytz.timezone('Europe/Paris')
    midnight_local = local_tz.localize(datetime(2024, 1, 26, 0, 0))
    midnight_utc = midnight_local.astimezone(pytz.utc)
    
    conditions = calculations.calculate_sky_conditions_at_time(paris_observatory, midnight_utc)
    
    result_map = calculations.generate_sky_quality_map(
        paris_observatory, midnight_utc, min_altitude_deg=30.0
    )
    
    # El mejor punto (best_az) NO debe estar cerca del azimut de la luna.
    azimuth_difference = abs(result_map['best_az_deg'] - conditions['moon_az_deg'])
    # La diferencia debería ser grande, idealmente cercana a 180 grados, pero comprobamos que no sea pequeña.
    assert min(azimuth_difference, 360 - azimuth_difference) > 90.0

# --- Pruebas para Funciones Auxiliares ---

def test_airmass_calculation():
    """Verifica el cálculo de la masa de aire en puntos clave."""
    assert np.isclose(calculations.calculate_airmass(90), 1.0) # En el cénit
    assert np.isclose(calculations.calculate_airmass(60), 1.1547, atol=1e-4)
    assert np.isclose(calculations.calculate_airmass(30), 2.0) # A 30 grados de altitud
    assert calculations.calculate_airmass(0) == np.inf
    
    # Prueba con un array
    altitudes = np.array([90, 30, 0])
    expected = np.array([1.0, 2.0, np.inf])
    result = calculations.calculate_airmass(altitudes)
    assert np.allclose(result, expected, equal_nan=False)