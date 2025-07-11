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
    calculate_observation_priority_index
)

from astrakairos.physics.kepler import predict_position

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

# Implement tests for the OPI

@pytest.fixture
def sample_orbital_elements():
    """Fixture que proporciona un conjunto de elementos orbitales realistas y completos."""
    return {
        'P': 100.0,      # Periodo: 100 años
        'T': 2000.0,     # Periastron: año 2000.0
        'e': 0.5,        # Excentricidad
        'a': 2.0,        # Semieje mayor en arcosegundos
        'i': 45.0,       # Inclinación en grados
        'Omega': 90.0,   # Longitud del nodo ascendente en grados
        'omega': 30.0    # Argumento del periastron en grados
    }

def test_opi_ideal_case(sample_orbital_elements):
    """
    Tests an ideal case with a clear, programmatically generated deviation.
    This test is robust and does not rely on hard-coded intermediate values.
    """
    t_last_obs = 2010.0
    current_date = 2020.0

    # Step 1: Get the TRUE predicted position from the orbital model.
    # This avoids any incorrect hard-coded assumptions.
    theta_pred_deg, rho_pred = predict_position(sample_orbital_elements, t_last_obs)

    # Step 2: Create a fictional "observed" position by introducing a known offset.
    # This makes the deviation deterministic and testable.
    observed_pa = theta_pred_deg + 5.0  # Simulate a +5 degree error in PA
    observed_sep = rho_pred + 0.1      # Simulate a +0.1" error in separation
    
    last_observation = {
        'date_last': t_last_obs, 
        'pa_last': observed_pa, 
        'sep_last': observed_sep
    }

    # Step 3: Calculate the expected deviation based on our known values.
    # This calculation mimics the internal logic of the function we are testing.
    theta_pred_rad = np.radians(theta_pred_deg)
    observed_rad = np.radians(observed_pa)
    
    x_pred = rho_pred * np.sin(theta_pred_rad)
    y_pred = rho_pred * np.cos(theta_pred_rad)
    x_obs = observed_sep * np.sin(observed_rad)
    y_obs = observed_sep * np.cos(observed_rad)
    
    expected_deviation = np.sqrt((x_pred - x_obs)**2 + (y_pred - y_obs)**2)
    expected_opi = expected_deviation / (current_date - t_last_obs)

    # Step 4: Call the function under test.
    opi, deviation = calculate_observation_priority_index(
        sample_orbital_elements, last_observation, current_date
    )

    # Step 5: Assert that the function's results match our robustly calculated expected values.
    assert np.isclose(deviation, expected_deviation)
    assert np.isclose(opi, expected_opi)

def test_opi_no_deviation(sample_orbital_elements):
    """
    Verifica que el OPI es (casi) cero si la observación coincide perfectamente con la predicción.
    """
    # Predecimos la posición para 2015.0 para crear una observación "perfecta"
    t_obs = 2015.0
    predicted_pa, predicted_sep = predict_position(sample_orbital_elements, t_obs)
    
    last_observation = {'date_last': t_obs, 'pa_last': predicted_pa, 'sep_last': predicted_sep}
    current_date = 2025.0
    
    opi, deviation = calculate_observation_priority_index(
        sample_orbital_elements, last_observation, current_date
    )
    
    assert np.isclose(deviation, 0.0, atol=1e-9)
    assert np.isclose(opi, 0.0, atol=1e-9)

def test_opi_returns_none_for_incomplete_orbit_data(sample_orbital_elements):
    """
    Verifica que la función devuelve None si faltan elementos orbitales clave.
    """
    incomplete_elements = sample_orbital_elements.copy()
    del incomplete_elements['P']  # Eliminamos el período, un elemento esencial

    last_observation = {'date_last': 2010.0, 'pa_last': 105.0, 'sep_last': 1.6}
    current_date = 2020.0
    
    result = calculate_observation_priority_index(
        incomplete_elements, last_observation, current_date
    )
    
    assert result is None

def test_opi_returns_none_for_incomplete_observation_data(sample_orbital_elements):
    """
    Verifica que la función devuelve None si faltan datos de la última observación.
    """
    last_observation = {'date_last': 2010.0, 'pa_last': None, 'sep_last': 1.6} # pa_last es None
    current_date = 2020.0
    
    result = calculate_observation_priority_index(
        sample_orbital_elements, last_observation, current_date
    )
    
    assert result is None

def test_opi_handles_prediction_failure(sample_orbital_elements):
    """
    Verifica que la función maneja correctamente un fallo en predict_position (ej. P=0).
    """
    bad_elements = sample_orbital_elements.copy()
    bad_elements['P'] = 0 # El período cero causará un error de división en la predicción
    
    last_observation = {'date_last': 2010.0, 'pa_last': 105.0, 'sep_last': 1.6}
    current_date = 2020.0
    
    result = calculate_observation_priority_index(
        bad_elements, last_observation, current_date
    )
    
    assert result is None