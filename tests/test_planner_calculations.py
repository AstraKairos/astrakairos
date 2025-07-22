# tests/test_planner_calculations.py

import pytest
from datetime import datetime
import pytz
import numpy as np

# Import the complete module to test its functions
from astrakairos.planner import calculations

# Test Fixtures

@pytest.fixture(scope="module")
def paris_observatory():
    """Reusable Skyfield location fixture for tests."""
    # Using a known location for predictable results.
    # Latitude, Longitude, Altitude for Paris Observatory.
    return calculations.get_observer_location(48.8368, 2.3358, 67)

@pytest.fixture(scope="module")
def test_date_new_moon():
    """Una fecha de prueba donde hay luna nueva, simplificando algunos tests."""
    return datetime(2024, 1, 11)

@pytest.fixture(scope="module")
def test_date_full_moon():
    """Una fecha de prueba donde hay luna llena, para probar el brillo."""
    return datetime(2024, 1, 25)

# --- Tests for get_nightly_events ---

def test_get_nightly_events_returns_correct_structure(paris_observatory, test_date_new_moon):
    """
    Verifies that the function returns a dictionary with all the expected keys
    and that the event times are logical if they exist.
    """
    events = calculations.get_nightly_events(paris_observatory, test_date_new_moon, 'Europe/Paris')
    
    assert isinstance(events, dict)
    expected_keys = [
        'sunset_utc', 'sunset_local', 'sunrise_utc', 'sunrise_local',
        'astronomical_twilight_end_utc', 'astronomical_twilight_start_utc',
        'moonrise_utc', 'moonrise_local', 'moonset_utc', 'moonset_local'
    ]
    for key in expected_keys:
        assert key in events
    
    # Robust assertion
    # Only check time logic if both events (sunset and sunrise) occur.
    # At high latitudes during summer, sunset or sunrise (or twilights)
    # might not occur, and the function would correctly return None.
    if events['sunset_utc'] and events['sunrise_utc']:
        assert events['sunset_utc'] < events['sunrise_utc']
    
    if events['astronomical_twilight_end_utc'] and events['astronomical_twilight_start_utc']:
        assert events['astronomical_twilight_end_utc'] < events['astronomical_twilight_start_utc']

# Tests for calculate_sky_conditions_at_time

def test_sky_conditions_at_time_new_moon(paris_observatory, test_date_new_moon):
    """Test sky conditions during new moon."""
    # Take local midnight in Paris for the test
    local_tz = pytz.timezone('Europe/Paris')
    midnight_local = local_tz.localize(datetime(2024, 1, 12, 0, 0))
    midnight_utc = midnight_local.astimezone(pytz.utc)
    
    conditions = calculations.calculate_sky_conditions_at_time(paris_observatory, midnight_utc)
    
    assert isinstance(conditions, dict)
    # During new moon, the phase should be very close to 0%
    assert conditions['moon_phase_percent'] < 2.0 
    assert 'zenith_ra_hours' in conditions

def test_sky_conditions_at_time_full_moon(paris_observatory, test_date_full_moon):
    """Verifica las condiciones del cielo durante una luna llena."""
    local_tz = pytz.timezone('Europe/Paris')
    midnight_local = local_tz.localize(datetime(2024, 1, 26, 0, 0))
    midnight_utc = midnight_local.astimezone(pytz.utc)
    
    conditions = calculations.calculate_sky_conditions_at_time(paris_observatory, midnight_utc)
    
    # During full moon, the phase should be very close to 100%
    assert conditions['moon_phase_percent'] > 98.0
    # Full moon is usually high in the sky at midnight
    assert conditions['moon_alt_deg'] > 30.0

# --- Tests for generate_sky_quality_map ---

def test_sky_quality_map_no_moon(paris_observatory, test_date_new_moon):
    """
    Prueba el mapa de calidad en una noche sin luna. El mejor punto debe ser el cénit.
    """
    local_tz = pytz.timezone('Europe/Paris')
    midnight_local = local_tz.localize(datetime(2024, 1, 12, 0, 0))
    midnight_utc = midnight_local.astimezone(pytz.utc)
    
    # Get zenith coordinates for comparison
    zenith_conditions = calculations.calculate_sky_conditions_at_time(paris_observatory, midnight_utc)
    
    result_map = calculations.generate_sky_quality_map(
        paris_observatory, midnight_utc, min_altitude_deg=30.0
    )
    
    assert isinstance(result_map, dict)
    # Without moon, the best point should be very close to zenith (highest altitude)
    assert np.isclose(result_map['best_alt_deg'], 85.0, atol=5.0) # Grid is 5 degrees
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
    
    # The best point (best_az) should NOT be close to the moon's azimuth.
    azimuth_difference = abs(result_map['best_az_deg'] - conditions['moon_az_deg'])
    # The difference should be large, ideally close to 180 degrees, but we check it's not small.
    assert min(azimuth_difference, 360 - azimuth_difference) > 90.0

# Tests for auxiliary functions

def test_airmass_calculation():
    """Test airmass calculation at key points."""
    assert np.isclose(calculations.calculate_airmass(90), 1.0) # At zenith
    assert np.isclose(calculations.calculate_airmass(60), 1.1547, atol=1e-4)
    assert np.isclose(calculations.calculate_airmass(30), 2.0) # At 30 degrees altitude
    assert calculations.calculate_airmass(0) == np.inf
    
    # Prueba con un array
    altitudes = np.array([90, 30, 0])
    expected = np.array([1.0, 2.0, np.inf])
    result = calculations.calculate_airmass(altitudes)
    assert np.allclose(result, expected, equal_nan=False)