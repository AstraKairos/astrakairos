"""
Keplerian orbital mechanics calculations for binary star systems.

This module implements numerical solutions to Kepler's equation and orbital
position calculations with proper vectorization and numerical stability.

Functions:
    solve_kepler: Solves Kepler's equation using Newton-Raphson method
    predict_position: Calculates position angle and separation for given epoch
    compute_orbital_anomalies: Computes mean, eccentric, and true anomalies

Dependencies:
    numpy: Vectorized numerical operations
    logging: Convergence information and warnings
"""

import logging
import numpy as np
from typing import Dict, Tuple, Union, Optional

# Centralized configuration imports for scientific consistency
from astrakairos.config import (
    DEFAULT_KEPLER_TOLERANCE,
    DEFAULT_KEPLER_MAX_ITERATIONS,
    HIGH_ECCENTRICITY_THRESHOLD,
    HIGH_E_COEFFICIENT,
    MIN_ECCENTRICITY,
    MAX_ECCENTRICITY_STABLE,
    MAX_ECCENTRICITY,
    DANGEROUS_ECCENTRICITY_WARNING,
    MIN_PERIOD_YEARS,
    MAX_PERIOD_YEARS,
    MIN_SEMIMAJOR_AXIS_ARCSEC,
    MAX_SEMIMAJOR_AXIS_ARCSEC,
    MIN_INCLINATION_DEG,
    MAX_INCLINATION_DEG,
    KEPLER_CONVERGENCE_WARNING_THRESHOLD,
    KEPLER_LOGGING_PRECISION,
    MIN_LONGITUDE_ASCENDING_NODE_DEG,
    MAX_LONGITUDE_ASCENDING_NODE_DEG,
    MIN_ARGUMENT_PERIASTRON_DEG,
    MAX_ARGUMENT_PERIASTRON_DEG,
    MIN_EPOCH_PERIASTRON_YEAR,
    MAX_EPOCH_PERIASTRON_YEAR
)

# Configure scientific logging
logger = logging.getLogger(__name__)

def solve_kepler(M_rad: Union[float, np.ndarray],
                 e: float,
                 tol: Optional[float] = None,
                 max_iter: Optional[int] = None,
                 e_threshold: Optional[float] = None,
                 coeff_high_e: Optional[float] = None) -> Union[float, np.ndarray]:
    """
    Solves Kepler's equation (M = E - e*sin(E)) for Eccentric Anomaly (E)
    using the Newton-Raphson method with a robust, hybrid initial guess strategy.

    This implementation is optimized for stability and speed across all
    elliptical eccentricities (0 <= e < 1) and operates vectorized on arrays.
    Uses centralized configuration from config.py for scientific consistency.

    Args:
        M_rad: Mean anomaly in radians. Can be a scalar or numpy array.
        e: Eccentricity of the orbit (0 <= e < 1).
        tol: The desired precision for convergence. Uses DEFAULT_KEPLER_TOLERANCE if None.
        max_iter: Maximum number of iterations. Uses DEFAULT_KEPLER_MAX_ITER if None.
        e_threshold: Eccentricity threshold for robust initial guess. Uses HIGH_ECCENTRICITY_THRESHOLD if None.
        coeff_high_e: Coefficient for high-eccentricity initial guess. Uses HIGH_E_COEFFICIENT if None.

    Returns:
        The Eccentric Anomaly (E) in radians. Same shape as input M_rad.
        
    Raises:
        ValueError: If eccentricity is outside valid range or potentially unstable.
        
    Notes:
        - For e > 0.95, issues a warning about potential numerical instability
        - Based on Napier (2024) for high-eccentricity initial guess strategy
    """
    # Use centralized configuration with fallback to provided values
    tol = tol if tol is not None else DEFAULT_KEPLER_TOLERANCE
    max_iter = max_iter if max_iter is not None else DEFAULT_KEPLER_MAX_ITERATIONS
    e_threshold = e_threshold if e_threshold is not None else HIGH_ECCENTRICITY_THRESHOLD
    coeff_high_e = coeff_high_e if coeff_high_e is not None else HIGH_E_COEFFICIENT
    
    # Scientific validation using centralized config
    if not (MIN_ECCENTRICITY <= e <= MAX_ECCENTRICITY_STABLE):
        raise ValueError(f"Eccentricity {e:.6f} outside stable range [{MIN_ECCENTRICITY}, {MAX_ECCENTRICITY_STABLE}]")
    
    # Validate eccentricity range for numerical stability
    if e > DANGEROUS_ECCENTRICITY_WARNING:
        logger.warning(f"High eccentricity {e:.{KEPLER_LOGGING_PRECISION}f} may cause instability")
    
    # Determine if input is scalar
    input_is_scalar = np.isscalar(M_rad)
    
    # Ensure M_rad is a numpy array for vectorized operations
    M_rad = np.asarray(M_rad)
    original_shape = M_rad.shape
    M_rad = M_rad.flatten()
    
    # Normalize Mean Anomaly to [-pi, pi] for better convergence
    M_norm = M_rad % (2 * np.pi)
    M_norm = np.where(M_norm > np.pi, M_norm - 2 * np.pi, M_norm)
    M_norm = np.where(M_norm < -np.pi, M_norm + 2 * np.pi, M_norm)

    # Initial guess strategy based on eccentricity
    if e < e_threshold:
        E = M_norm + e * np.sin(M_norm) * (1.0 + e * np.cos(M_norm))
    else:
        E = M_norm + coeff_high_e * e

    # Newton-Raphson iteration
    converged = np.zeros(E.shape, dtype=bool)
    
    for _ in range(max_iter):
        mask = ~converged
        
        if not np.any(mask):
            break
            
        f_E = E[mask] - e * np.sin(E[mask]) - M_norm[mask]
        f_prime_E = 1.0 - e * np.cos(E[mask])

        # The correction step
        delta = f_E / f_prime_E
        
        E[mask] -= delta

        # Check for convergence: if the correction step is smaller than the tolerance.
        converged[mask] = np.abs(delta) < tol

    # Performance warning for low convergence
    convergence_rate = np.sum(converged) / len(converged) * 100
    if convergence_rate < KEPLER_CONVERGENCE_WARNING_THRESHOLD:
        logger.warning(f"Low convergence rate: {convergence_rate:.1f}% (e={e:.{KEPLER_LOGGING_PRECISION}f})")
    
    # Reshape result and handle scalar vs array return type
    result = E.reshape(original_shape)
    
    # Return scalar if input was scalar (consistent API)
    if input_is_scalar:
        return float(result.item())
    else:
        return result

def predict_position(orbital_elements: Dict[str, float], date: float) -> Tuple[float, float]:
    """
    Predicts the sky position (Position Angle and Separation) of a binary star 
    for a given observation date, based on its Keplerian orbital elements.
    
    Uses centralized configuration for scientific validation of orbital elements.

    Args:
        orbital_elements: A dictionary containing the 7 Keplerian elements:
                          'P' (Period, in years), 'T' (Time of periastron passage, in years),
                          'e' (Eccentricity), 'a' (Semi-major axis, in arcseconds),
                          'i' (Inclination, in degrees), 'Omega' (Longitude of Ascending Node, in degrees),
                          'omega' (Argument of Periastron, in degrees).
        date: The observation date in decimal years.

    Returns:
        A tuple containing (position_angle_deg, separation_arcsec).
        
    Raises:
        ValueError: If any orbital element is outside scientifically valid ranges.
        KeyError: If required orbital elements are missing.
    """
    # 1. Extract and validate orbital elements
    try:
        P = orbital_elements['P']
        T = orbital_elements['T']
        e = orbital_elements['e']
        a = orbital_elements['a']
        i_deg = orbital_elements['i']
        Omega_deg = orbital_elements['Omega']
        omega_deg = orbital_elements['omega']
    except KeyError as exc:
        raise ValueError(f"Missing required orbital element in dictionary: {exc}")

    # Scientific validation using centralized configuration
    if not (MIN_PERIOD_YEARS <= P <= MAX_PERIOD_YEARS):
        raise ValueError(f"Orbital period {P:.3f} years outside valid range [{MIN_PERIOD_YEARS}, {MAX_PERIOD_YEARS}]")
    
    if not (MIN_ECCENTRICITY <= e <= MAX_ECCENTRICITY):
        raise ValueError(f"Eccentricity {e:.{KEPLER_LOGGING_PRECISION}f} outside valid range [{MIN_ECCENTRICITY}, {MAX_ECCENTRICITY}]")
    
    if not (MIN_SEMIMAJOR_AXIS_ARCSEC <= a <= MAX_SEMIMAJOR_AXIS_ARCSEC):
        raise ValueError(f"Semi-major axis {a:.3f} arcsec outside valid range [{MIN_SEMIMAJOR_AXIS_ARCSEC}, {MAX_SEMIMAJOR_AXIS_ARCSEC}]")
    
    if not (MIN_INCLINATION_DEG <= i_deg <= MAX_INCLINATION_DEG):
        raise ValueError(f"Inclination {i_deg:.1f}° outside valid range [{MIN_INCLINATION_DEG}°, {MAX_INCLINATION_DEG}°]")

    # Additional validation for angular elements
    if not (MIN_LONGITUDE_ASCENDING_NODE_DEG <= Omega_deg <= MAX_LONGITUDE_ASCENDING_NODE_DEG):
        raise ValueError(f"Longitude of ascending node {Omega_deg:.1f}° outside valid range [{MIN_LONGITUDE_ASCENDING_NODE_DEG}°, {MAX_LONGITUDE_ASCENDING_NODE_DEG}°]")
    
    if not (MIN_ARGUMENT_PERIASTRON_DEG <= omega_deg <= MAX_ARGUMENT_PERIASTRON_DEG):
        raise ValueError(f"Argument of periastron {omega_deg:.1f}° outside valid range [{MIN_ARGUMENT_PERIASTRON_DEG}°, {MAX_ARGUMENT_PERIASTRON_DEG}°]")
    
    if not (MIN_EPOCH_PERIASTRON_YEAR <= T <= MAX_EPOCH_PERIASTRON_YEAR):
        raise ValueError(f"Epoch of periastron {T:.1f} outside reasonable range [{MIN_EPOCH_PERIASTRON_YEAR}, {MAX_EPOCH_PERIASTRON_YEAR}]")

    # Conditionally log high-eccentricity cases
    if e > DANGEROUS_ECCENTRICITY_WARNING and P > 1000:
        logger.info(f"Computing high-eccentricity ({e:.3f}) long-period ({P:.0f}y) orbit")

    # Convert angles to radians
    i_rad = np.radians(i_deg)
    Omega_rad = np.radians(Omega_deg)
    omega_rad = np.radians(omega_deg)
    
    # Calculate Mean Anomaly
    mean_motion = 2 * np.pi / P
    M = mean_motion * (date - T)
    
    # Solve Kepler's equation for Eccentric Anomaly
    E = solve_kepler(M, e)
    
    # Calculate orbital plane coordinates
    x_prime = a * (np.cos(E) - e)
    y_prime = a * np.sqrt(1 - e**2) * np.sin(E)
    
    # Apply orbital rotations to project to sky plane
    cos_omega = np.cos(omega_rad)
    sin_omega = np.sin(omega_rad)
    x = x_prime * cos_omega - y_prime * sin_omega
    y = x_prime * sin_omega + y_prime * cos_omega
    
    # Apply inclination rotation
    cos_i = np.cos(i_rad)
    y_inclined = y * cos_i
    
    # Apply ascending node rotation
    cos_Omega = np.cos(Omega_rad)
    sin_Omega = np.sin(Omega_rad)
    x_sky = x * sin_Omega + y_inclined * cos_Omega
    y_sky = -x * cos_Omega + y_inclined * sin_Omega
    
    # Convert to observables
    separation_arcsec = np.sqrt(x_sky**2 + y_sky**2)
    position_angle_rad = np.arctan2(x_sky, y_sky) 
    position_angle_deg = np.degrees(position_angle_rad)
    
    # Normalize to [0, 360) range
    position_angle_deg = position_angle_deg % 360.0
    
    return (position_angle_deg, separation_arcsec)

def compute_orbital_anomalies(orbital_elements: Dict[str, float], dates: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes Mean Anomaly (M), Eccentric Anomaly (E), and True Anomaly (nu)
    for an array of dates, useful for plotting orbits or detailed analysis.
    
    Uses the vectorized solve_kepler function directly (no np.vectorize needed).

    Args:
        orbital_elements: Dictionary with orbital elements.
        dates: A numpy array of dates in decimal years.
        
    Returns:
        A dictionary with numpy arrays for M, E, and nu (all in radians).
        
    Notes:
        - All anomalies are returned in radians for mathematical consistency
        - Mean anomaly is normalized to [0, 2π] range
        - Uses numerically stable atan2 formulation for true anomaly
    """
    # Extract and validate required elements
    try:
        P = orbital_elements['P']
        T = orbital_elements['T']
        e = orbital_elements['e']
    except KeyError as exc:
        raise ValueError(f"Missing required orbital element for anomaly calculation: {exc}")
    
    # Scientific validation
    if not (MIN_PERIOD_YEARS <= P <= MAX_PERIOD_YEARS):
        raise ValueError(f"Orbital period {P:.3f} years outside valid range")
    
    if not (MIN_ECCENTRICITY <= e <= MAX_ECCENTRICITY):
        raise ValueError(f"Eccentricity {e:.6f} outside valid range")
    
    # Calculate Mean Anomaly for the array of dates
    mean_motion = 2 * np.pi / P
    M_array = mean_motion * (dates - T)
    
    # Calculate anomalies for all epochs
    E_array = solve_kepler(M_array, e)
    
    # Convert Eccentric to True Anomaly using stable formula
    tan_nu_half_array = np.sqrt((1 + e) / (1 - e)) * np.tan(E_array / 2)
    nu_array = 2 * np.arctan(tan_nu_half_array)
    
    return {
        'M': M_array % (2 * np.pi), # Normalize M to [0, 2π]
        'E': E_array,
        'nu': nu_array
    }