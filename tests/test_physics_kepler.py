# tests/test_physics_kepler.py
import pytest
import math
import numpy as np
from astrakairos.physics.kepler import solve_kepler, predict_position, compute_orbital_anomalies

# Basic existing tests
def test_solve_kepler_circular_orbit():
    """Verifies that for a circular orbit (e=0), E equals M."""
    mean_anomaly_rad = np.radians(45.0)  # M in radians
    eccentricity = 0.0
    eccentric_anomaly_rad = solve_kepler(mean_anomaly_rad, eccentricity)
    assert np.isclose(eccentric_anomaly_rad, mean_anomaly_rad, atol=1e-9)

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

# Additional scientific tests

def test_solve_kepler_vectorized_consistency():
    """Test that vectorized and scalar calls give consistent results."""
    M_values = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    e = 0.3
    
    # Vectorized call
    E_vector = solve_kepler(M_values, e)
    
    # Individual scalar calls
    E_scalars = np.array([solve_kepler(float(M), e) for M in M_values])
    
    assert np.allclose(E_vector, E_scalars, atol=1e-12)

def test_solve_kepler_high_eccentricity():
    """Test solver stability for high eccentricity orbits."""
    M = 1.0
    e_high = 0.9  # High eccentricity
    
    E = solve_kepler(M, e_high)
    
    # Verify Kepler's equation is satisfied
    residual = E - e_high * np.sin(E) - M
    assert abs(residual) < 1e-10

def test_solve_kepler_invalid_eccentricity():
    """Test that invalid eccentricity values raise appropriate errors."""
    M = 1.0
    
    # Test e >= 1 (parabolic/hyperbolic)
    with pytest.raises(ValueError, match="outside stable range"):
        solve_kepler(M, 1.0)
    
    # Test negative eccentricity
    with pytest.raises(ValueError, match="outside stable range"):
        solve_kepler(M, -0.1)

def test_predict_position_validation():
    """Test orbital element validation in predict_position."""
    base_elements = {
        'P': 10.0, 'T': 2000.0, 'e': 0.1, 'a': 1.0,
        'i': 45.0, 'Omega': 0.0, 'omega': 0.0
    }
    
    # Test invalid period
    invalid_elements = base_elements.copy()
    invalid_elements['P'] = -5.0
    with pytest.raises(ValueError, match="outside valid range"):
        predict_position(invalid_elements, 2020.0)
    
    # Test invalid eccentricity
    invalid_elements = base_elements.copy()
    invalid_elements['e'] = 1.5
    with pytest.raises(ValueError, match="outside valid range"):
        predict_position(invalid_elements, 2020.0)
    
    # Test invalid inclination
    invalid_elements = base_elements.copy()
    invalid_elements['i'] = 200.0
    with pytest.raises(ValueError, match="outside valid range"):
        predict_position(invalid_elements, 2020.0)

def test_compute_orbital_anomalies_array():
    """Test computation of orbital anomalies for array of dates."""
    orbital_elements = {'P': 10.0, 'T': 2000.0, 'e': 0.2}
    dates = np.array([2000.0, 2002.5, 2005.0, 2007.5, 2010.0])
    
    anomalies = compute_orbital_anomalies(orbital_elements, dates)
    
    # Check that all keys are present
    assert 'M' in anomalies
    assert 'E' in anomalies  
    assert 'nu' in anomalies
    
    # Check array shapes
    assert anomalies['M'].shape == dates.shape
    assert anomalies['E'].shape == dates.shape
    assert anomalies['nu'].shape == dates.shape
    
    # At periastron (T=2000), M should be 0
    assert np.isclose(anomalies['M'][0], 0.0, atol=1e-10)

def test_solve_kepler_scalar_return_type():
    """Test that scalar input returns scalar output."""
    M_scalar = 1.0
    e = 0.3
    
    E_result = solve_kepler(M_scalar, e)
    
    # Should return a Python float, not numpy array
    assert isinstance(E_result, float)
    assert not isinstance(E_result, np.ndarray)

def test_solve_kepler_array_return_type():
    """Test that array input returns array output."""
    M_array = np.array([1.0, 2.0])
    e = 0.3
    
    E_result = solve_kepler(M_array, e)
    
    # Should return numpy array
    assert isinstance(E_result, np.ndarray)
    assert E_result.shape == M_array.shape

def test_configuration_usage():
    """Test that centralized configuration is being used."""
    # This test ensures the refactored functions use config values
    from astrakairos.config import DEFAULT_KEPLER_TOLERANCE
    
    M = 1.0
    e = 0.3
    
    # Call without explicit tolerance - should use config default
    E1 = solve_kepler(M, e)
    
    # Call with explicit tolerance matching config
    E2 = solve_kepler(M, e, tol=DEFAULT_KEPLER_TOLERANCE)
    
    # Results should be identical
    assert E1 == E2

# --- Tests de casos extremos ---

def test_solve_kepler_near_parabolic():
    """Test solver behavior near parabolic eccentricity."""
    M = 0.1
    e = 0.93  # High eccentricity but within stable range
    
    # Should not raise error but may log warning
    E = solve_kepler(M, e)
    
    # Verify solution
    residual = E - e * np.sin(E) - M
    assert abs(residual) < 1e-8  # Slightly relaxed tolerance for extreme case

def test_missing_orbital_elements():
    """Test error handling for missing orbital elements."""
    incomplete_elements = {'P': 10.0, 'T': 2000.0}  # Missing e, a, i, etc.
    
    with pytest.raises(ValueError, match="Missing required orbital element"):
        predict_position(incomplete_elements, 2020.0)

def test_compute_anomalies_validation():
    """Test validation in compute_orbital_anomalies."""
    dates = np.array([2000.0, 2010.0])
    
    # Missing elements
    incomplete_elements = {'P': 10.0}  # Missing T, e
    with pytest.raises(ValueError, match="Missing required orbital element"):
        compute_orbital_anomalies(incomplete_elements, dates)
    
    # Invalid period
    invalid_elements = {'P': -5.0, 'T': 2000.0, 'e': 0.1}
    with pytest.raises(ValueError, match="outside valid range"):
        compute_orbital_anomalies(invalid_elements, dates)