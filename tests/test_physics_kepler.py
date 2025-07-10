# tests/test_physics_kepler.py
import pytest
import math
import numpy as np
from astrakairos.physics.kepler import solve_kepler, predict_position

# --- Test para solve_kepler ---
def test_solve_kepler_circular_orbit():
    """Verifies that for a circular orbit (e=0), E equals M."""
    mean_anomaly_rad = np.radians(45.0)  # M in radians
    eccentricity = 0.0
    eccentric_anomaly_rad = solve_kepler(mean_anomaly_rad, eccentricity)
    assert np.isclose(eccentric_anomaly_rad, mean_anomaly_rad, atol=1e-9)

# tests/test_physics_kepler.py
import pytest
import numpy as np
from astrakairos.physics.kepler import solve_kepler

def test_solve_kepler_known_case():
    """
    Verifies the solution of solve_kepler for a specific standard case (M=0.5 rad, e=0.2).
    The expected value is derived from the function's consistent and mathematically
    verified output for these inputs.
    """
    mean_anomaly_rad = 0.5  # Input M in RADIANS
    eccentricity = 0.2

    # This is the actual, mathematically consistent value calculated by our solve_kepler
    # for M=0.5 radians and e=0.2.
    expected_E_rad = 0.6154681694899653 

    eccentric_anomaly_rad = solve_kepler(mean_anomaly_rad, eccentricity, tol=1e-12)
    
    # Assert using numpy.isclose for robust floating-point comparison.
    # atol=1e-9 is a good absolute tolerance for this level of precision.
    assert np.isclose(eccentric_anomaly_rad, expected_E_rad, atol=1e-9)

# --- Test para predict_position ---
def test_predict_position_circular_face_on_orbit():
    """Tests position prediction for a simple orbit: circular and face-on."""
    orbital_elements = {
        'P': 10.0, 'T': 2000.0, 'e': 0.0, 'a': 2.0,
        'i': 0.0, 'Omega': 0.0, 'omega': 0.0
    }
    # At a quarter period (2.5 years after T), the star should be at PA=90°
    date = 2002.5
    pa_deg, sep_arcsec = predict_position(orbital_elements, date)

    assert np.isclose(pa_deg, 90.0, atol=1e-6)
    assert np.isclose(sep_arcsec, 2.0, atol=1e-6)

# --- Puedes añadir más tests aquí ---