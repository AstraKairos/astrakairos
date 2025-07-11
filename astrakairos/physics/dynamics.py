import numpy as np
from typing import Dict, Tuple, Optional
from .kepler import predict_position

def calculate_velocity_vector(data: Dict[str, float]) -> Tuple[float, float]:
    """
    Calculate the total velocity and velocity position angle from observations.
    
    Args:
        data: Dictionary containing:
            - pa_first: First position angle in degrees
            - sep_first: First separation in arcseconds
            - date_first: First observation date in years
            - pa_last: Last position angle in degrees
            - sep_last: Last separation in arcseconds
            - date_last: Last observation date in years
            
    Returns:
        (v_total, pa_v_deg): Total velocity in arcsec/year and velocity PA in degrees
    """
    # Extract values
    pa1 = data['pa_first']    # θ1 in degrees
    pa2 = data['pa_last']     # θ2 in degrees
    sep1 = data['sep_first']  # ρ1 in arcseconds
    sep2 = data['sep_last']   # ρ2 in arcseconds
    t1 = data['date_first']   # First date in years
    t2 = data['date_last']    # Last date in years
    
    # Check for valid data
    if None in [pa1, pa2, sep1, sep2, t1, t2]:
        raise ValueError("Missing required data for velocity calculation")
    
    if t2 == t1:
        raise ValueError("Cannot calculate velocity with identical dates")
    
    # Convert angles to radians
    theta1_rad = np.radians(pa1)
    theta2_rad = np.radians(pa2)
    
    # Calculate Cartesian components
    # In astronomy, PA is measured from North through East
    # So x = ρ * sin(θ) (East) and y = ρ * cos(θ) (North)
    x1 = sep1 * np.sin(theta1_rad)
    y1 = sep1 * np.cos(theta1_rad)
    x2 = sep2 * np.sin(theta2_rad)
    y2 = sep2 * np.cos(theta2_rad)
    
    # Calculate displacement
    dx = x2 - x1
    dy = y2 - y1
    
    # Calculate time interval
    dt = t2 - t1
    
    # Calculate velocity components
    vx = dx / dt  # East velocity component
    vy = dy / dt  # North velocity component
    
    # Calculate total velocity
    v_total = np.sqrt(vx**2 + vy**2)
    
    # Calculate velocity position angle
    # arctan2(vx, vy) because PA is from North through East - Regular arctan wouldn't handle quadrant correctly
    # Note: vx is East component, vy is North component
    pa_v_rad = np.arctan2(vx, vy)
    pa_v_deg = np.degrees(pa_v_rad)
    
    # Ensure PA is in range [0, 360)
    if pa_v_deg < 0:
        pa_v_deg += 360
    
    return (v_total, pa_v_deg)

def calculate_angular_velocity(data: Dict[str, float]) -> float:
    """
    Calculate the angular velocity (dθ/dt) in degrees/year.
    
    Args:
        data: Dictionary with pa_first, pa_last, date_first, date_last
        
    Returns:
        Angular velocity in degrees/year
    """
    pa1 = data['pa_first']
    pa2 = data['pa_last']
    t1 = data['date_first']
    t2 = data['date_last']
    
    if t2 == t1:
        raise ValueError("Cannot calculate angular velocity with identical dates")
    
    # Handle angle wrap-around
    dpa = pa2 - pa1
    
    # If the angle change is > 180°, assume it went the other way
    if dpa > 180:
        dpa -= 360
    elif dpa < -180:
        dpa += 360
    
    dt = t2 - t1
    angular_velocity = dpa / dt
    
    return angular_velocity

def calculate_radial_velocity(data: Dict[str, float]) -> float:
    """
    Calculate the radial velocity (dρ/dt) in arcsec/year.
    
    Args:
        data: Dictionary with sep_first, sep_last, date_first, date_last
        
    Returns:
        Radial velocity in arcsec/year (positive = increasing separation)
    """
    sep1 = data['sep_first']
    sep2 = data['sep_last']
    t1 = data['date_first']
    t2 = data['date_last']
    
    if t2 == t1:
        raise ValueError("Cannot calculate radial velocity with identical dates")
    
    dsep = sep2 - sep1
    dt = t2 - t1
    radial_velocity = dsep / dt
    
    return radial_velocity

def estimate_period_from_motion(angular_velocity_deg_per_year: float, min_angular_velocity_threshold: float = 0.01) -> Optional[float]:
    """
    Estimate orbital period from angular velocity, assuming circular motion.
    
    Args:
        angular_velocity_deg_per_year: Angular velocity in degrees/year
        min_angular_velocity_threshold: The minimum absolute angular velocity
                                        (in degrees/year) below which an orbital
                                        period estimation is considered unreliable
                                        or not practically useful. Defaults to 0.01.
        
    Returns:
        Estimated period in years, or None if velocity is too small
    """
    if abs(angular_velocity_deg_per_year) < min_angular_velocity_threshold:
        """
        Small insight on this seemigly arbitrary threshold,
        A value of 0.01 in the angular velocity corresponds to an orbital period
        of 36.000 years, which makes it not very useful for practical purposes.
        """
        return None  # Motion too slow to estimate
    
    # Period = 360° / angular_velocity
    period_years = 360.0 / abs(angular_velocity_deg_per_year)
    
    return period_years

def calculate_orbit_coverage(data: Dict[str, float], period_years: Optional[float] = None) -> float:
    """
    Calculate what fraction of the orbit has been observed.
    
    Args:
        data: Dictionary with date_first and date_last
        period_years: Orbital period in years (if known)
        
    Returns:
        Fraction of orbit covered (0 to 1), or observation span in years if period unknown
    """
    t1 = data['date_first']
    t2 = data['date_last']
    
    observation_span = t2 - t1
    
    if period_years and period_years > 0:
        coverage = observation_span / period_years
        return min(coverage, 1.0)  # Cap at 100%
    else:
        return observation_span  # Just return years if no period

def calculate_observation_priority_index(
    orbital_elements: Dict[str, float],
    last_observation: Dict[str, float],
    current_date: float
) -> Optional[Tuple[float, float]]:
    """
    Calculates the Observation Priority Index (OPI).

    This index quantifies the rate of deviation between the position predicted
    by an orbital model and the last recorded observation. It serves as a powerful
    indicator of how "outdated" or "incorrect" a published orbit might be,
    prioritizing targets that require new measurements.

    Args:
        orbital_elements: Dictionary containing the 7 Keplerian orbital elements
                          (P, T, e, a, i, Omega, omega).
        last_observation: Dictionary containing data from the last observation:
                          'date_last' (year), 'pa_last' (PA in degrees),
                          'sep_last' (separation in arcseconds).
        current_date: The current date (as a decimal year) for which the
                      observation is being planned.

    Returns:
        A tuple containing (opi, deviation_arcsec), where:
        - opi: The Observation Priority Index in arcseconds per year.
        - deviation_arcsec: The total on-sky deviation in arcseconds at the
                            time of the last observation.
        Returns None if essential data for the calculation is missing.
    """
    # 1. Validate that all required data is present and valid
    required_orbit_keys = {'P', 'T', 'e', 'a', 'i', 'Omega', 'omega'}
    required_obs_keys = {'date_last', 'pa_last', 'sep_last'}

    if not all(k in orbital_elements and orbital_elements[k] is not None for k in required_orbit_keys) or \
       not all(k in last_observation and last_observation[k] is not None for k in required_obs_keys):
        return None

    t_last_obs = last_observation['date_last']
    rho_last_obs = last_observation['sep_last']
    theta_last_obs_deg = last_observation['pa_last']

    # 2. Predict the THEORETICAL position for the exact date of the LAST observation
    try:
        theta_pred_deg, rho_pred = predict_position(orbital_elements, t_last_obs)
    except (ValueError, KeyError) as e:
        # If prediction fails (e.g., P=0), the OPI cannot be calculated.
        print(f"  [Warning] Could not predict position for OPI calculation: {e}")
        return None

    # 3. Calculate the deviation vector (Δ) magnitude
    # Convert angles to radians for trigonometric functions
    theta_pred_rad = np.radians(theta_pred_deg)
    theta_last_obs_rad = np.radians(theta_last_obs_deg)

    # Convert both positions (predicted and observed) to Cartesian coordinates
    # x corresponds to East (+), y corresponds to North (+)
    x_pred = rho_pred * np.sin(theta_pred_rad)
    y_pred = rho_pred * np.cos(theta_pred_rad)

    x_obs = rho_last_obs * np.sin(theta_last_obs_rad)
    y_obs = rho_last_obs * np.cos(theta_last_obs_rad)

    # Calculate the magnitude of the deviation (Euclidean distance on the sky plane)
    deviation_arcsec = np.sqrt((x_pred - x_obs)**2 + (y_pred - y_obs)**2)

    # 4. Calculate the Observation Priority Index (OPI)
    time_since_last_obs = current_date - t_last_obs

    if time_since_last_obs == 0:
        # If the observation date is exactly the current date, the rate is undefined.
        # In this case, the deviation itself is the most direct measure of discrepancy.
        # We assign an "infinite" OPI if there's a deviation, indicating immediate priority.
        opi = np.inf if deviation_arcsec > 0 else 0.0
    elif time_since_last_obs < 0:
        # If current_date is BEFORE last_observation, this indicates a data anomaly
        # or a misinterpretation of 'current_date'.
        # The OPI as a 'rate since last observation' isn't meaningful here.
        # It's better to return 0 or None, or raise a ValueError depending on strictness.
        # For simplicity, we'll just return 0.0, indicating no future observation priority.
        opi = 0.0
    else:
        # For any positive time difference, calculate the rate directly.
        opi = deviation_arcsec / time_since_last_obs
        
    return opi, deviation_arcsec