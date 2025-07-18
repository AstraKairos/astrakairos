"""
Orbital dynamics and motion analysis for binary star systems.

Provides tools for:
- Observation Priority Index (OPI): Quantifies orbital solution degradation rate
- Robust linear fitting: Theil-Sen regression for outlier-resistant motion analysis  
- Curvature index: Measures deviation between orbital and linear motion models

Functions return None for invalid inputs rather than raising exceptions.
All validation thresholds are sourced from centralized configuration.
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional, Any
from astropy.table import Table 
from sklearn.linear_model import TheilSenRegressor

# Local imports
from .kepler import predict_position
from ..data.source import WdsSummary, OrbitalElements

# Centralized configuration imports for consistency
from ..config import (
    OPI_DEVIATION_THRESHOLD_ARCSEC,
    OPI_INFINITE_THRESHOLD,
    MIN_POINTS_FOR_ROBUST_FIT,
    ROBUST_REGRESSION_RANDOM_STATE,
    MAX_ASTROMETRIC_VELOCITY_ARCSEC_PER_YEAR,
    MIN_TIME_BASELINE_YEARS,
    MAX_CURVATURE_INDEX_ARCSEC,
    MIN_PREDICTION_DATE_OFFSET_YEARS,
    MAX_PREDICTION_DATE_OFFSET_YEARS,
    MAX_RMSE_FOR_LINEAR_FIT_ARCSEC,
    MIN_RESIDUAL_SIGNIFICANCE,
    MAX_EXTRAPOLATION_FACTOR,
    MIN_EPOCH_YEAR,
    MAX_EPOCH_YEAR,
    MIN_SEPARATION_ARCSEC,
    MAX_SEPARATION_ARCSEC,
    MAX_DEVIATION_WARNING_ARCSEC,
    MAX_OLD_OBSERVATION_WARNING_YEARS
)

# Configure logging
logger = logging.getLogger(__name__)

def calculate_observation_priority_index(
    orbital_elements: OrbitalElements,
    wds_summary: WdsSummary,
    current_date: float
) -> Optional[Tuple[float, float]]:
    """
    Calculates the Observation Priority Index (OPI) using centralized configuration.

    This index quantifies the rate of deviation between the position predicted
    by an orbital model and the last recorded observation. It serves as an
    indicator of how outdated or incorrect a published orbit might be,
    prioritizing targets that require new measurements.

    Args:
        orbital_elements: Dictionary containing the 7 Keplerian orbital elements.
        wds_summary: Dictionary of WDS summary data, must include 'date_last',
                     'pa_last', and 'sep_last'.
        current_date: The current date (as a decimal year).

    Returns:
        A tuple containing (OPI, deviation_in_arcsec), or None if calculation fails.
    """
    # Validate that all required data is present and valid
    if not orbital_elements or not wds_summary:
        return None
        
    t_last_obs = wds_summary.get('date_last')
    theta_last_obs_deg = wds_summary.get('pa_last')
    rho_last_obs = wds_summary.get('sep_last')

    if None in [t_last_obs, theta_last_obs_deg, rho_last_obs]:
        return None

    # Validation of input ranges using centralized configuration
    if not (MIN_EPOCH_YEAR <= t_last_obs <= MAX_EPOCH_YEAR):
        return None
    
    if not (0.0 <= theta_last_obs_deg <= 360.0):
        return None
    
    if not (MIN_SEPARATION_ARCSEC <= rho_last_obs <= MAX_SEPARATION_ARCSEC):
        return None

    # Predict the position for the exact date of the last observation
    try:
        predicted_pos = predict_position(orbital_elements, t_last_obs)
        if predicted_pos is None:
            return None
        theta_pred_deg, rho_pred = predicted_pos
    except (ValueError, KeyError):
        return None

    # Convert polar to Cartesian coordinates to calculate Euclidean distance
    theta_pred_rad = np.radians(theta_pred_deg)
    theta_last_obs_rad = np.radians(theta_last_obs_deg)
    x_pred = rho_pred * np.sin(theta_pred_rad)
    y_pred = rho_pred * np.cos(theta_pred_rad)
    x_obs = rho_last_obs * np.sin(theta_last_obs_rad)
    y_obs = rho_last_obs * np.cos(theta_last_obs_rad)

    deviation_arcsec = np.sqrt((x_pred - x_obs)**2 + (y_pred - y_obs)**2)
    time_since_last_obs = current_date - t_last_obs

    # Calculate the rate of deviation (OPI) using centralized configuration
    if time_since_last_obs <= 0:
        # Zero or negative time baseline - use configured threshold
        opi = np.inf if deviation_arcsec > OPI_DEVIATION_THRESHOLD_ARCSEC else 0.0
    else:
        opi = deviation_arcsec / time_since_last_obs
        
    # Log warnings only for extreme cases to avoid performance impact
    if deviation_arcsec > MAX_DEVIATION_WARNING_ARCSEC:
        logger.warning(f"Large positional deviation detected: {deviation_arcsec:.3f}\" > {MAX_DEVIATION_WARNING_ARCSEC}\"")
        
    return opi, deviation_arcsec

def calculate_robust_linear_fit(measurements: Table) -> Optional[Dict[str, Any]]:
    """
    Performs a robust linear fit on historical astrometric data using Theil-Sen regression.
    
    Implements temporal centering for numerical stability and validates results 
    against physical bounds for astrometric motion.
    
    Args:
        measurements: Astropy Table with columns 'epoch', 'theta', 'rho'
                     containing astrometric measurements
    
    Returns:
        Dictionary containing velocity components, statistics, and fitting parameters,
        or None if insufficient data or fitting fails
    """
    # Use centralized configuration for minimum points requirement
    if not measurements or len(measurements) < MIN_POINTS_FOR_ROBUST_FIT:
        return None

    try:
        # Prepare data for regression with proper array handling
        t = np.array(measurements['epoch']).reshape(-1, 1)
        theta_rad = np.radians(np.array(measurements['theta']))
        rho = np.array(measurements['rho'])

        # Validate input data ranges
        epoch_range = np.max(t) - np.min(t)
        if epoch_range < MIN_TIME_BASELINE_YEARS:
            return None

        # Convert polar coordinates to Cartesian (x, y)
        x = rho * np.sin(theta_rad)
        y = rho * np.cos(theta_rad)

        # Center time data for numerical stability
        t_mean = np.mean(t)
        t_centered = t - t_mean

        # Perform two independent robust regressions for x(t) and y(t) with centered time
        theil_sen_x = TheilSenRegressor(random_state=ROBUST_REGRESSION_RANDOM_STATE)
        theil_sen_y = TheilSenRegressor(random_state=ROBUST_REGRESSION_RANDOM_STATE)

        theil_sen_x.fit(t_centered, x)
        theil_sen_y.fit(t_centered, y)

        # The slopes of the fits are the velocity components (vx, vy)
        vx = theil_sen_x.coef_[0]  # arcsec/year
        vy = theil_sen_y.coef_[0]  # arcsec/year

        # Validate velocity results
        v_total_robust = np.sqrt(vx**2 + vy**2)
        if v_total_robust > MAX_ASTROMETRIC_VELOCITY_ARCSEC_PER_YEAR:
            logger.warning(f"High astrometric velocity detected: {v_total_robust:.4f}\"/yr")

        pa_v_robust = np.degrees(np.arctan2(vx, vy)) % 360

        # Calculate the Root Mean Square Error (RMSE) of the linear fit
        x_pred = theil_sen_x.predict(t_centered)
        y_pred = theil_sen_y.predict(t_centered)
        
        # The residuals are the Euclidean distances between observed and predicted points
        residuals = np.sqrt((x - x_pred)**2 + (y - y_pred)**2)
        rmse = np.sqrt(np.mean(residuals**2))

        # Calculate additional useful statistics
        max_residual = np.max(residuals)
        median_residual = np.median(residuals)
        
        # Time baseline for context with proper scalar extraction
        time_baseline = float(np.max(t) - np.min(t))

        # Extract intercepts for accurate linear predictions (now centered at mean epoch)
        intercept_x = theil_sen_x.intercept_
        intercept_y = theil_sen_y.intercept_
        mean_epoch_fit = float(t_mean)

        return {
            'vx_arcsec_per_year': vx,
            'vy_arcsec_per_year': vy,
            'v_total_robust': v_total_robust,
            'pa_v_robust': pa_v_robust,
            'rmse': rmse,
            'max_residual': max_residual,
            'median_residual': median_residual,
            'time_baseline_years': time_baseline,
            'n_points_fit': len(t),
            'intercept_x': intercept_x,
            'intercept_y': intercept_y,
            'mean_epoch_fit': mean_epoch_fit
        }

    except Exception as e:
        logger.error(f"Robust linear fit failed: {e}")
        return None

def calculate_curvature_index(
    orbital_elements: OrbitalElements,
    linear_fit_results: Dict[str, Any],
    current_date: float
) -> Optional[float]:
    """
    Calculates the Curvature Index using centralized configuration.
    
    This index quantifies the deviation between an orbital model and a robust linear 
    fit at a specific date, providing insight into orbital motion significance.

    Args:
        orbital_elements: The 7 Keplerian orbital elements.
        linear_fit_results: Complete results from calculate_robust_linear_fit()
                           including vx, vy, intercept_x, intercept_y, and mean_epoch_fit.
        current_date: The date (in decimal years) to evaluate the deviation.

    Returns:
        The Curvature Index in arcseconds, or None if calculation fails.
    """
    if not orbital_elements or not linear_fit_results:
        return None

    # Validate required keys in linear fit results
    required_keys = ['vx_arcsec_per_year', 'vy_arcsec_per_year', 'intercept_x', 'intercept_y', 'mean_epoch_fit']
    missing_keys = [key for key in required_keys if key not in linear_fit_results]
    if missing_keys:
        return None

    try:
        # Extract linear fit parameters
        vx = linear_fit_results['vx_arcsec_per_year']
        vy = linear_fit_results['vy_arcsec_per_year']
        intercept_x = linear_fit_results['intercept_x']
        intercept_y = linear_fit_results['intercept_y']
        mean_epoch = linear_fit_results['mean_epoch_fit']
        
        # Validate prediction date using centralized configuration
        time_offset = current_date - mean_epoch
        if not (MIN_PREDICTION_DATE_OFFSET_YEARS <= time_offset <= MAX_PREDICTION_DATE_OFFSET_YEARS):
            return None
        
        # Check for safe extrapolation using centralized configuration
        if 'time_baseline_years' in linear_fit_results:
            baseline = linear_fit_results['time_baseline_years']
            if baseline > 0:
                extrapolation_factor = abs(time_offset) / baseline
                if extrapolation_factor > MAX_EXTRAPOLATION_FACTOR:
                    return None

        # Predict position using the orbital model
        orbital_pos = predict_position(orbital_elements, current_date)
        if not orbital_pos:
            return None
        theta_orb_deg, rho_orb = orbital_pos

        # Validate orbital prediction using centralized configuration
        if not (MIN_SEPARATION_ARCSEC <= rho_orb <= MAX_SEPARATION_ARCSEC):
            return None

        # Calculate the position from the linear model using centered intercepts
        x_lin = intercept_x + vx * (current_date - mean_epoch)
        y_lin = intercept_y + vy * (current_date - mean_epoch)

        # Convert orbital position to Cartesian and calculate deviation
        theta_orb_rad = np.radians(theta_orb_deg)
        x_orb = rho_orb * np.sin(theta_orb_rad)
        y_orb = rho_orb * np.cos(theta_orb_rad)
        
        # The Curvature Index is the Euclidean distance between the two predictions
        curvature_index = np.sqrt((x_orb - x_lin)**2 + (y_orb - y_lin)**2)
        
        # Log warning only for extreme cases
        if curvature_index > MAX_CURVATURE_INDEX_ARCSEC:
            logger.warning(f"Large curvature index {curvature_index:.3f}\" > {MAX_CURVATURE_INDEX_ARCSEC}\"")
        
        return curvature_index

    except Exception as e:
        logger.error(f"Curvature index calculation failed: {e}")
        return None

def estimate_velocity_from_endpoints(
    wds_summary: WdsSummary
) -> Optional[Dict[str, Any]]:
    """
    Calculates velocity estimate using only the first and last observations.
    
    This is a fallback method for when insufficient measurements are available
    for robust analysis. It provides a simple two-point velocity estimate
    with proper validation using framework configuration.
    
    Args:
        wds_summary: WDS summary data containing first and last observations.
        
    Returns:
        Dictionary with velocity components and derived quantities, or None if
        insufficient data.
    """
    
    if not wds_summary:
        return None
        
    required_fields = ['date_first', 'date_last', 'pa_first', 'pa_last', 'sep_first', 'sep_last']
    if not all(field in wds_summary and wds_summary[field] is not None for field in required_fields):
        return None
    
    try:
        # Extract data with validation
        t1, t2 = wds_summary['date_first'], wds_summary['date_last']
        theta1_deg, theta2_deg = wds_summary['pa_first'], wds_summary['pa_last']
        rho1, rho2 = wds_summary['sep_first'], wds_summary['sep_last']
        
        # Validate time baseline using centralized configuration
        dt = t2 - t1
        if dt <= 0:  # No time baseline or negative
            return None
            
        if dt < MIN_TIME_BASELINE_YEARS:
            return None  # Too short baseline for reliable estimate
            
        # Convert to Cartesian coordinates
        theta1_rad, theta2_rad = np.radians(theta1_deg), np.radians(theta2_deg)
        x1, y1 = rho1 * np.sin(theta1_rad), rho1 * np.cos(theta1_rad)
        x2, y2 = rho2 * np.sin(theta2_rad), rho2 * np.cos(theta2_rad)
        
        # Calculate velocity components
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        
        v_total = np.sqrt(vx**2 + vy**2)
        pa_v = np.degrees(np.arctan2(vx, vy)) % 360
        
        # Validate velocity results using centralized configuration
        if v_total > MAX_ASTROMETRIC_VELOCITY_ARCSEC_PER_YEAR:
            logger.warning(f"High velocity from endpoint calculation: {v_total:.4f}\"/yr > {MAX_ASTROMETRIC_VELOCITY_ARCSEC_PER_YEAR}")
        
        return {
            'vx_arcsec_per_year': vx,
            'vy_arcsec_per_year': vy,
            'v_total_estimate': v_total,
            'pa_v_estimate': pa_v,
            'time_baseline_years': dt,
            'n_points_fit': 2,
            'method': 'two_point_estimate'
        }
        
    except Exception as e:
        logger.error(f"Endpoint velocity calculation failed: {e}")
        return None