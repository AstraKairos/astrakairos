# tests/test_physics_dynamics.py
import pytest
import numpy as np

# Importa todas las funciones del módulo a probar
from astrakairos.physics.dynamics import (
    calculate_velocity_vector,
    calculate_angular_velocity,
    calculate_radial_velocity,
    estimate_period_from_motion,
    calculate_orbit_coverage,
)

# --- Pruebas para calculate_velocity_vector ---

def test_velocity_vector_simple_linear_motion():
    """
    Prueba un caso de movimiento lineal simple y fácil de verificar manualmente.
    Movimiento del punto (x=0, y=1) al punto (x=1, y=1) en 10 años.
    Esto corresponde a un desplazamiento de 1.0 arcsec hacia el Este (dx=1, dy=0).
    La velocidad debe ser 0.1 arcsec/año en dirección Este (PA_v = 90°).
    """
    # Pos 1: (x=0, y=1) -> pa=0°, sep=1.0
    # Pos 2: (x=1, y=1) -> pa=45°, sep=sqrt(2)
    data = {
        'pa_first': 0.0, 'sep_first': 1.0, 'date_first': 2000.0,
        'pa_last': 45.0, 'sep_last': np.sqrt(2), 'date_last': 2010.0,
    }
    v_total, pa_v = calculate_velocity_vector(data)
    assert np.isclose(v_total, 0.1)
    assert np.isclose(pa_v, 90.0)

def test_velocity_vector_no_motion():
    """Verifica que la velocidad es cero si no hay movimiento."""
    data = {
        'pa_first': 30.0, 'sep_first': 5.0, 'date_first': 2000.0,
        'pa_last': 30.0, 'sep_last': 5.0, 'date_last': 2010.0,
    }
    v_total, pa_v = calculate_velocity_vector(data)
    assert np.isclose(v_total, 0.0)

def test_velocity_vector_pure_radial_motion():
    """Prueba el movimiento puramente radial (a lo largo de la línea de visión)."""
    # Movimiento hacia afuera en PA 90°
    data = {
        'pa_first': 90.0, 'sep_first': 1.0, 'date_first': 2000.0,
        'pa_last': 90.0, 'sep_last': 2.0, 'date_last': 2001.0,
    }
    v_total, pa_v = calculate_velocity_vector(data)
    assert np.isclose(v_total, 1.0)
    assert np.isclose(pa_v, 90.0)

def test_velocity_vector_pure_tangential_motion():
    """Prueba el movimiento puramente tangencial (en un círculo)."""
    # Movimiento de PA 0° a PA 90° en 1 año, a una separación constante de 1 arcsec.
    # Desplazamiento de (x=0, y=1) a (x=1, y=0). dx=1, dy=-1.
    data = {
        'pa_first': 0.0, 'sep_first': 1.0, 'date_first': 2000.0,
        'pa_last': 90.0, 'sep_last': 1.0, 'date_last': 2001.0,
    }
    v_total, pa_v = calculate_velocity_vector(data)
    # v_total = sqrt(vx^2 + vy^2) = sqrt(1^2 + (-1)^2) = sqrt(2)
    assert np.isclose(v_total, np.sqrt(2))
    # pa_v = atan2(dx, dy) = atan2(1, -1) = 135°
    assert np.isclose(pa_v, 135.0)
    
def test_velocity_vector_raises_error_on_same_date():
    """Verifica que se lanza un ValueError si las fechas son idénticas."""
    data = {
        'pa_first': 10.0, 'sep_first': 1.0, 'date_first': 2000.0,
        'pa_last': 20.0, 'sep_last': 1.1, 'date_last': 2000.0,
    }
    with pytest.raises(ValueError, match="identical dates"):
        calculate_velocity_vector(data)


# --- Pruebas para calculate_angular_velocity ---

def test_angular_velocity_simple():
    """Prueba un cálculo simple de velocidad angular."""
    data = {'pa_first': 10, 'pa_last': 20, 'date_first': 2000, 'date_last': 2010}
    vel = calculate_angular_velocity(data)
    assert np.isclose(vel, 1.0) # (20-10) / 10

def test_angular_velocity_wraparound_positive():
    """Prueba el manejo del cruce por 360° (ej. de 350° a 10°)."""
    data = {'pa_first': 350, 'pa_last': 10, 'date_first': 2000, 'date_last': 2010}
    vel = calculate_angular_velocity(data)
    # El cambio es de +20°, no de -340°
    assert np.isclose(vel, 2.0) # 20 / 10

def test_angular_velocity_wraparound_negative():
    """Prueba el manejo del cruce por 360° en dirección contraria."""
    data = {'pa_first': 10, 'pa_last': 350, 'date_first': 2000, 'date_last': 2005}
    vel = calculate_angular_velocity(data)
    # El cambio es de -20°, no de +340°
    assert np.isclose(vel, -4.0) # -20 / 5


# --- Pruebas para calculate_radial_velocity ---

def test_radial_velocity_increasing():
    """Prueba una separación que aumenta."""
    data = {'sep_first': 1.0, 'sep_last': 1.5, 'date_first': 2000, 'date_last': 2005}
    vel = calculate_radial_velocity(data)
    assert np.isclose(vel, 0.1) # 0.5 / 5

def test_radial_velocity_decreasing():
    """Prueba una separación que disminuye."""
    data = {'sep_first': 2.0, 'sep_last': 1.8, 'date_first': 2010, 'date_last': 2012}
    vel = calculate_radial_velocity(data)
    assert np.isclose(vel, -0.1) # -0.2 / 2


# --- Pruebas para estimate_period_from_motion ---

def test_estimate_period():
    """Prueba la estimación de período con una velocidad angular dada."""
    assert np.isclose(estimate_period_from_motion(1.0), 360.0)
    assert np.isclose(estimate_period_from_motion(2.0), 180.0)
    # Debe usar el valor absoluto de la velocidad
    assert np.isclose(estimate_period_from_motion(-0.5), 720.0)

def test_estimate_period_returns_none_for_small_motion():
    """Verifica que devuelve None si el movimiento es demasiado lento."""
    assert estimate_period_from_motion(0.001) is None
    assert estimate_period_from_motion(0) is None


# --- Pruebas para calculate_orbit_coverage ---

def test_orbit_coverage_with_period():
    """Prueba el cálculo de cobertura con un período conocido."""
    data = {'date_first': 2000, 'date_last': 2050}
    period = 100.0
    coverage = calculate_orbit_coverage(data, period)
    assert np.isclose(coverage, 0.5) # 50 / 100

def test_orbit_coverage_caps_at_one():
    """Verifica que la cobertura no puede superar el 100% (1.0)."""
    data = {'date_first': 1900, 'date_last': 2050}
    period = 100.0
    coverage = calculate_orbit_coverage(data, period)
    assert np.isclose(coverage, 1.0) # 150 / 100 -> capped at 1.0

def test_orbit_coverage_without_period():
    """Prueba que devuelve el lapso de tiempo si el período es desconocido."""
    data = {'date_first': 1985.5, 'date_last': 2020.0}
    coverage = calculate_orbit_coverage(data, None)
    assert np.isclose(coverage, 34.5)