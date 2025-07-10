import numpy as np
from typing import Dict, Tuple, Optional

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
    # arctan2(vx, vy) because PA is from North through East
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

def estimate_period_from_motion(angular_velocity_deg_per_year: float) -> Optional[float]:
    """
    Estimate orbital period from angular velocity, assuming circular motion.
    
    Args:
        angular_velocity_deg_per_year: Angular velocity in degrees/year
        
    Returns:
        Estimated period in years, or None if velocity is too small
    """
    if abs(angular_velocity_deg_per_year) < 0.01:
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