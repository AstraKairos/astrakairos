import numpy as np
from typing import Dict, Tuple

def solve_kepler(M_rad: np.ndarray,
                 e: float,
                 tol: float = 1e-12,
                 max_iter: int = 15,
                 e_threshold: float = 0.8,
                 coeff_high_e: float = 0.71) -> np.ndarray:
    """
    Solves Kepler's equation (M = E - e*sin(E)) for Eccentric Anomaly (E)
    using the Newton-Raphson method with a robust, hybrid initial guess strategy.

    This implementation is optimized for stability and speed across all
    elliptical eccentricities (0 <= e < 1) and operates vectorized on arrays.

    Args:
        M_rad: Mean anomaly in radians. Can be a scalar or numpy array.
        e: Eccentricity of the orbit (0 <= e < 1).
        tol: The desired precision for convergence (aka. "Tolerance").
            - float, default 1e-12
        max_iter: Maximum number of iterations for the Newton-Raphson method.
            - int, default 15
        e_threshold: Eccentricity threshold above which to switch to a more robust
            high-eccentricity initial guess. Backed by Napier (2024).
            - float, default 0.8
        coeff_high_e: Coefficient used in the high-eccentricity initial guess:
            E ≈ M + coeff_high_e * e, where 0.71 was suggested by Napier (2024).
            - float, default 0.71

    Returns:
        The Eccentric Anomaly (E) in radians. Same shape as input M_rad.
    """
    # Ensure M_rad is a numpy array for vectorized operations
    M_rad = np.asarray(M_rad)
    original_shape = M_rad.shape
    M_rad = M_rad.flatten()
    
    # 1. Normalize Mean Anomaly to be within [-pi, pi] for better convergence.
    M_norm = M_rad % (2 * np.pi)
    M_norm = np.where(M_norm > np.pi, M_norm - 2 * np.pi, M_norm)
    M_norm = np.where(M_norm < -np.pi, M_norm + 2 * np.pi, M_norm)

    # 2. Provide a robust initial guess for Eccentric Anomaly (E) using a
    #    hybrid strategy based on eccentricity.
    if e < e_threshold:
        # For low to moderate eccentricities, a second-order approximation is very effective.
        E = M_norm + e * np.sin(M_norm) * (1.0 + e * np.cos(M_norm))
    else:
        # For high eccentricities (e -> 1), this simpler guess is more stable.
        # Based on Napier (2024): E ≈ M + 0.71 * e
        E = M_norm + coeff_high_e * e

    # 3. Newton-Raphson iteration loop - This is the core of the solver.
    # The function to find the root of is f(E) = E - e*sin(E) - M_norm = 0
    # Its derivative is f'(E) = 1 - e*cos(E)
    # The iteration step is E_next = E - f(E) / f'(E)

    # Track convergence for each element
    converged = np.zeros(E.shape, dtype=bool)
    
    for _ in range(max_iter):
        # Only compute for elements that haven't converged yet
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

    # Reshape back to original shape
    return E.reshape(original_shape)

def predict_position(orbital_elements: Dict[str, float], date: float) -> Tuple[float, float]:
    """
    Predicts the sky position (Position Angle and Separation) of a binary star 
    for a given observation date, based on its Keplerian orbital elements.

    Args:
        orbital_elements: A dictionary containing the 7 Keplerian elements:
                          'P' (Period, in years), 'T' (Time of periastron passage, in years),
                          'e' (Eccentricity), 'a' (Semi-major axis, in arcseconds),
                          'i' (Inclination, in degrees), 'Omega' (Longitude of Ascending Node, in degrees),
                          'omega' (Argument of Periastron, in degrees).
        date: The observation date in decimal years.

    Returns:
        A tuple containing (position_angle_deg, separation_arcsec).
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

    if P <= 0:
        raise ValueError("Orbital period (P) must be positive.")
    if not (0 <= e < 1): # Checking for non-physical eccentricities
        raise ValueError("Eccentricity (e) must be in the range [0, 1).")
    if a <= 0:
        raise ValueError("Semi-major axis (a) must be positive.")
    if not (0 <= i_deg <= 180):
        raise ValueError("Inclination (i) must be in the range [0, 180] degrees.")

    # 2. Convert all input angles from degrees to radians for numpy functions
    i_rad = np.radians(i_deg)
    Omega_rad = np.radians(Omega_deg)
    omega_rad = np.radians(omega_deg)
    
    # 3. Calculate Mean Anomaly (M)
    # M represents the fraction of the orbital period that has elapsed since periastron.
    mean_motion = 2 * np.pi / P
    M = mean_motion * (date - T)
    
    # 4. Solve Kepler's Equation for Eccentric Anomaly (E)
    # This is the crucial step linking time to orbital position.
    E = solve_kepler(M, e)
    
    # 5. Calculate rectangular coordinates (x_prime, y_prime) in the orbital plane.
    # The primary star is at the origin (0,0). x' is along the major axis towards periastron.
    # y' is perpendicular to x' in the orbital plane.
    x_prime = a * (np.cos(E) - e)
    y_prime = a * np.sqrt(1 - e**2) * np.sin(E)
    
    # 6. Apply three successive rotations to project to the plane of the sky.
    # This is the standard formulation for visual binary orbits.

    # Rotate by the argument of periastron (omega) to align the orbit with the line of nodes.
    cos_omega = np.cos(omega_rad)
    sin_omega = np.sin(omega_rad)
    x = x_prime * cos_omega - y_prime * sin_omega
    y = x_prime * sin_omega + y_prime * cos_omega
    
    # Rotate by the inclination (i) around the line of nodes (the new x-axis).
    # This projects the orbit onto the sky plane.
    cos_i = np.cos(i_rad)
    y_inclined = y * cos_i
    
    # Rotate by the longitude of the ascending node (Omega) in the plane of the sky.
    # This aligns the orbit with celestial coordinates (North and East).
    cos_Omega = np.cos(Omega_rad)
    sin_Omega = np.sin(Omega_rad)

    # x_sky corresponds to the East-West direction (+East)
    # y_sky corresponds to the North-South direction (+North)
    x_sky = x * sin_Omega + y_inclined * cos_Omega
    y_sky = -x * cos_Omega + y_inclined * sin_Omega
    
    # 7. Convert the sky-plane cartesian coordinates to observables.
    # Separation (rho) is the distance from the primary star (origin).
    separation_arcsec = np.sqrt(x_sky**2 + y_sky**2)
    
    # Position Angle (theta) is measured from North (positive Y-axis) towards East (positive X-axis).
    # np.arctan2(x, y) follows this astronomical convention.
    position_angle_rad = np.arctan2(x_sky, y_sky) 
    position_angle_deg = np.degrees(position_angle_rad)
    
    # 8. Normalize the Position Angle to the conventional [0, 360) degree range.
    if position_angle_deg < 0:
        position_angle_deg += 360
    
    return (position_angle_deg, separation_arcsec)

def compute_orbital_anomalies(orbital_elements: Dict[str, float], dates: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes Mean Anomaly (M), Eccentric Anomaly (E), and True Anomaly (nu)
    for an array of dates, useful for plotting orbits or detailed analysis.

    Args:
        orbital_elements: Dictionary with orbital elements.
        dates: A numpy array of dates in decimal years.
        
    Returns:
        A dictionary with numpy arrays for M, E, and nu (all in radians).
    """
    P = orbital_elements['P']
    T = orbital_elements['T']
    e = orbital_elements['e']
    
    # Calculate Mean Anomaly for the array of dates
    mean_motion = 2 * np.pi / P
    M_array = mean_motion * (dates - T)
    
    # Use np.vectorize to apply the scalar solve_kepler function to each element of the M_array.
    solve_kepler_vec = np.vectorize(solve_kepler)
    E_array = solve_kepler_vec(M_array, e)
    
    # Calculate True Anomaly for the array of E values
    # This uses the numerically stable atan2 formulation.
    tan_nu_half_array = np.sqrt((1 + e) / (1 - e)) * np.tan(E_array / 2)
    nu_array = 2 * np.arctan(tan_nu_half_array)
    
    return {
        'M': M_array % (2 * np.pi), # Normalize M to [0, 2pi]
        'E': E_array,
        'nu': nu_array
    }