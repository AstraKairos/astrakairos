"""
Orbital Dynamics and Motion Analysis Module

This module implements scientifically rigorous astrometric analysis algorithms
for binary star systems, with emphasis on observational priority assessment,
linear motion characterization, and orbital curvature quantification.

Scientific Framework:
--------------------
- Observation Priority Index (OPI): Quantifies orbital solution degradation rate
- Robust Linear Fitting: Theil-Sen regression for outlier-resistant motion analysis  
- Curvature Index: Measures deviation between orbital and linear motion models
- Statistical Validation: Comprehensive bounds checking and significance testing

Key Algorithms:
---------------
1. **OPI Calculation**: Rate of positional deviation between orbital predictions
   and last observed positions, indicating urgency for new observations

2. **Theil-Sen Regression**: Robust linear fitting resistant to outliers,
   with proper numerical centering for epoch stability

3. **Curvature Quantification**: Euclidean distance between orbital and linear
   predictions, revealing orbital motion signature strength

4. **Scientific Validation**: Physical bounds checking for velocities, time
   baselines, and prediction extrapolation limits

Mathematical Background:
-----------------------
- Cartesian coordinate transformations: x = ρ*sin(θ), y = ρ*cos(θ)
- Theil-Sen estimator: Median of slopes between all point pairs
- Centered regression: Temporal centering for numerical stability
- Euclidean metric: L2 norm for positional deviations

Dependencies:
-------------
- numpy: Vectorized numerical operations and statistical functions
- astropy.table: Structured astronomical data handling
- sklearn.linear_model: Theil-Sen robust regression implementation
- config: Centralized scientific constants and validation ranges
- logging: Scientific debugging and validation monitoring

Examples:
---------
>>> # Calculate observation priority for orbital system
>>> from astrakairos.physics.dynamics import calculate_observation_priority_index
>>> opi, deviation = calculate_observation_priority_index(
...     orbital_elements, wds_summary, current_date=2025.0)

>>> # Robust linear motion analysis
>>> from astrakairos.physics.dynamics import calculate_robust_linear_fit
>>> fit_results = calculate_robust_linear_fit(measurements_table)
>>> velocity = fit_results['v_total_robust']

>>> # Orbital vs linear curvature comparison
>>> from astrakairos.physics.dynamics import calculate_curvature_index
>>> curvature = calculate_curvature_index(
...     orbital_elements, fit_results, prediction_date=2025.0)

Notes:
------
All functions implement comprehensive validation using configurable astronomical
ranges and return None for invalid inputs rather than raising exceptions,
following the framework's defensive programming philosophy.

This implementation prioritizes scientific rigor and reproducibility, with
all thresholds and parameters sourced from centralized configuration for
framework consistency and publication-ready reproducibility.

Authors: AstraKairos Development Team
License: MIT
Version: 2.0 (Refactored for scientific publication standards)
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional, Any
from astropy.table import Table 
from sklearn.linear_model import TheilSenRegressor

# Local imports
from .kepler import predict_position
from ..data.source import WdsSummary, OrbitalElements

# Centralized configuration imports for scientific consistency
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
    MAX_EXTRAPOLATION_FACTOR
)

# Configure scientific logging
logger = logging.getLogger(__name__)

def calculate_observation_priority_index(
    orbital_elements: OrbitalElements,
    wds_summary: WdsSummary,
    current_date: float
) -> Optional[Tuple[float, float]]:
    """
    Calculates the Observation Priority Index (OPI) using centralized configuration.

    This index quantifies the rate of deviation between the position predicted
    by an orbital model and the last recorded observation. It serves as a powerful
    indicator of how "outdated" or "incorrect" a published orbit might be,
    prioritizing targets that require new measurements.

    Uses scientifically validated thresholds from centralized configuration
    for consistent and reproducible results across the framework.

    Args:
        orbital_elements: Dictionary containing the 7 Keplerian orbital elements.
        wds_summary: Dictionary of WDS summary data, must include 'date_last',
                     'pa_last', and 'sep_last'.
        current_date: The current date (as a decimal year).

    Returns:
        A tuple containing (OPI, deviation_in_arcsec), or None if calculation fails.
        
    Raises:
        None: Function uses defensive programming, returning None for invalid inputs.
        
    Notes:
        - Uses OPI_DEVIATION_THRESHOLD_ARCSEC for zero-time-baseline handling
        - Implements proper Cartesian coordinate transformations
        - Logs scientific warnings for edge cases
    """
    # Validate that all required data is present and valid
    if not orbital_elements or not wds_summary:
        logger.debug("OPI calculation failed: Missing orbital elements or WDS summary")
        return None
        
    t_last_obs = wds_summary.get('date_last')
    theta_last_obs_deg = wds_summary.get('pa_last')
    rho_last_obs = wds_summary.get('sep_last')

    if None in [t_last_obs, theta_last_obs_deg, rho_last_obs]:
        logger.debug("OPI calculation failed: Missing required WDS summary data")
        return None

    # Scientific validation of input ranges
    if not (1800.0 <= t_last_obs <= 2100.0):  # Reasonable historical range
        logger.warning(f"Last observation date {t_last_obs} outside typical range [1800, 2100]")
    
    if not (0.0 <= theta_last_obs_deg <= 360.0):
        logger.warning(f"Position angle {theta_last_obs_deg}° outside valid range [0°, 360°]")
    
    if not (0.001 <= rho_last_obs <= 100.0):  # Reasonable separation range
        logger.warning(f"Separation {rho_last_obs}\" outside typical range [0.001\", 100\"]")

    # Predict the THEORETICAL position for the exact date of the LAST observation
    try:
        predicted_pos = predict_position(orbital_elements, t_last_obs)
        if predicted_pos is None:
            logger.debug("OPI calculation failed: Orbital prediction returned None")
            return None
        theta_pred_deg, rho_pred = predicted_pos
    except (ValueError, KeyError) as e:
        logger.debug(f"OPI calculation failed: Orbital prediction error - {e}")
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
        logger.debug(f"Zero time baseline: deviation={deviation_arcsec:.6f}\", OPI={'∞' if opi == np.inf else '0'}")
    else:
        opi = deviation_arcsec / time_since_last_obs
        logger.debug(f"OPI calculation: deviation={deviation_arcsec:.4f}\", time_baseline={time_since_last_obs:.2f}yr, OPI={opi:.6f}")
        
    # Scientific validation of results
    if deviation_arcsec > 10.0:  # Unusually large deviation
        logger.warning(f"Large positional deviation detected: {deviation_arcsec:.3f}\" - orbit may be incorrect")
    
    if time_since_last_obs > 50.0:  # Very old observation
        logger.warning(f"Very old last observation: {time_since_last_obs:.1f} years - high observation priority")
        
    return opi, deviation_arcsec

def calculate_robust_linear_fit(measurements: Table) -> Optional[Dict[str, Any]]:
    """
    Performs a robust linear fit on historical astrometric data using centralized
    scientific configuration to determine the linear motion vector and assess 
    the quality of the linear fit.
    
    Implements Theil-Sen regression with proper numerical centering for stability
    and comprehensive scientific validation of results using configurable thresholds.
    
    Args:
        measurements: Astropy Table with columns 'epoch', 'theta', 'rho'
                     containing astrometric measurements
    
    Returns:
        Dictionary containing velocity components, statistics, and fitting parameters,
        or None if insufficient data or fitting fails
        
    Notes:
        - Uses MIN_POINTS_FOR_ROBUST_FIT from centralized configuration
        - Implements temporal centering for numerical stability
        - Validates results against physical bounds for astrometric motion
        - Returns comprehensive statistics for scientific analysis
    """
    # Use centralized configuration for minimum points requirement
    if not measurements or len(measurements) < MIN_POINTS_FOR_ROBUST_FIT:
        logger.debug(f"Insufficient data for robust fit: {len(measurements) if measurements else 0} < {MIN_POINTS_FOR_ROBUST_FIT}")
        return None

    try:
        # Prepare data for regression with proper array handling
        t = np.array(measurements['epoch']).reshape(-1, 1)
        theta_rad = np.radians(np.array(measurements['theta']))
        rho = np.array(measurements['rho'])

        # Scientific validation of input data ranges
        epoch_range = np.max(t) - np.min(t)
        if epoch_range < MIN_TIME_BASELINE_YEARS:
            logger.warning(f"Short time baseline: {epoch_range:.2f} years < {MIN_TIME_BASELINE_YEARS}")
        
        if np.any(rho <= 0):
            logger.warning("Non-positive separations detected in measurements")
            
        if np.any((theta_rad < 0) | (theta_rad > 2*np.pi)):
            logger.warning("Position angles outside [0°, 360°] range detected")

        # Convert polar coordinates to Cartesian (x, y)
        x = rho * np.sin(theta_rad)
        y = rho * np.cos(theta_rad)

        # CRITICAL: Center time data for numerical stability
        # This prevents precision issues with large epoch values (e.g., 1950.0, 2020.0)
        t_mean = np.mean(t)
        t_centered = t - t_mean  # Now "time zero" is the mean observation epoch

        # Perform two independent robust regressions for x(t) and y(t) with centered time
        # Use centralized configuration for reproducible random state
        theil_sen_x = TheilSenRegressor(random_state=ROBUST_REGRESSION_RANDOM_STATE)
        theil_sen_y = TheilSenRegressor(random_state=ROBUST_REGRESSION_RANDOM_STATE)

        theil_sen_x.fit(t_centered, x)
        theil_sen_y.fit(t_centered, y)

        # The slopes of the fits are the velocity components (vx, vy)
        vx = theil_sen_x.coef_[0]  # arcsec/year
        vy = theil_sen_y.coef_[0]  # arcsec/year

        # Scientific validation of velocity results
        v_total_robust = np.sqrt(vx**2 + vy**2)
        if v_total_robust > MAX_ASTROMETRIC_VELOCITY_ARCSEC_PER_YEAR:
            logger.warning(f"High astrometric velocity detected: {v_total_robust:.4f}\"/yr > {MAX_ASTROMETRIC_VELOCITY_ARCSEC_PER_YEAR}")

        pa_v_robust = np.degrees(np.arctan2(vx, vy)) % 360

        # Calculate the Root Mean Square Error (RMSE) of the linear fit
        x_pred = theil_sen_x.predict(t_centered)
        y_pred = theil_sen_y.predict(t_centered)
        
        # The residuals are the Euclidean distances between observed and predicted points
        residuals = np.sqrt((x - x_pred)**2 + (y - y_pred)**2)
        rmse = np.sqrt(np.mean(residuals**2))

        # Scientific validation of fit quality
        if rmse > MAX_RMSE_FOR_LINEAR_FIT_ARCSEC:
            logger.warning(f"High RMSE detected: {rmse:.4f}\" > {MAX_RMSE_FOR_LINEAR_FIT_ARCSEC}\" - linear model may be inappropriate")

        # Calculate additional useful statistics
        max_residual = np.max(residuals)
        median_residual = np.median(residuals)
        
        # Time baseline for context with proper scalar extraction
        time_baseline = float(np.max(t) - np.min(t))

        # Extract intercepts for accurate linear predictions (now centered at mean epoch)
        intercept_x = theil_sen_x.intercept_
        intercept_y = theil_sen_y.intercept_
        mean_epoch_fit = float(t_mean)  # The centering epoch for proper predictions

        # Log scientific summary
        logger.debug(f"Robust fit completed: v_total={v_total_robust:.4f}\"/yr, RMSE={rmse:.4f}\", baseline={time_baseline:.1f}yr")

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
            'intercept_x': intercept_x,  # X-intercept at mean epoch (not year 0)
            'intercept_y': intercept_y,  # Y-intercept at mean epoch (not year 0)
            'mean_epoch_fit': mean_epoch_fit  # CRITICAL: Centering epoch for predictions
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
    Calculates the Curvature Index using centralized configuration and scientific validation.
    
    This index quantifies the deviation between an orbital model and a robust linear 
    fit at a specific date, providing insight into orbital motion significance.

    The index measures how much the known orbital solution deviates from
    a simple linear motion model. A high value suggests the orbital solution
    accurately captures the observed curvature in the motion, while a low value
    might indicate issues with the orbital solution or suggest nearly linear motion.

    This metric complements the OPI by providing insight into the quality of
    the orbital fit relative to the observed motion pattern.

    Args:
        orbital_elements: The 7 Keplerian orbital elements.
        linear_fit_results: Complete results from calculate_robust_linear_fit()
                           including vx, vy, intercept_x, intercept_y, and mean_epoch_fit.
        current_date: The date (in decimal years) to evaluate the deviation.

    Returns:
        The Curvature Index in arcseconds, or None if calculation fails.
        
    Notes:
        - Uses centralized configuration for prediction limits and validation
        - Implements proper centered regression formula for linear predictions
        - Validates extrapolation safety using configurable factors
        - Logs scientific warnings for edge cases and unusual results
    """
    if not orbital_elements or not linear_fit_results:
        logger.debug("Curvature index calculation failed: Missing orbital elements or linear fit results")
        return None

    # Validate required keys in linear fit results
    required_keys = ['vx_arcsec_per_year', 'vy_arcsec_per_year', 'intercept_x', 'intercept_y', 'mean_epoch_fit']
    missing_keys = [key for key in required_keys if key not in linear_fit_results]
    if missing_keys:
        logger.debug(f"Curvature index calculation failed: Missing linear fit keys: {missing_keys}")
        return None

    try:
        # Extract linear fit parameters with scientific validation
        vx = linear_fit_results['vx_arcsec_per_year']
        vy = linear_fit_results['vy_arcsec_per_year']
        intercept_x = linear_fit_results['intercept_x']
        intercept_y = linear_fit_results['intercept_y']
        mean_epoch = linear_fit_results['mean_epoch_fit']
        
        # Scientific validation of prediction date
        time_offset = current_date - mean_epoch
        if not (MIN_PREDICTION_DATE_OFFSET_YEARS <= time_offset <= MAX_PREDICTION_DATE_OFFSET_YEARS):
            logger.warning(f"Prediction date offset {time_offset:.1f} years outside safe range [{MIN_PREDICTION_DATE_OFFSET_YEARS}, {MAX_PREDICTION_DATE_OFFSET_YEARS}]")
        
        # Check for safe extrapolation if baseline data available
        if 'time_baseline_years' in linear_fit_results:
            baseline = linear_fit_results['time_baseline_years']
            extrapolation_factor = abs(time_offset) / baseline if baseline > 0 else float('inf')
            if extrapolation_factor > MAX_EXTRAPOLATION_FACTOR:
                logger.warning(f"Large extrapolation factor {extrapolation_factor:.2f} > {MAX_EXTRAPOLATION_FACTOR} - results may be unreliable")

        # Predict position using the orbital model
        orbital_pos = predict_position(orbital_elements, current_date)
        if not orbital_pos:
            logger.debug("Curvature index calculation failed: Orbital prediction returned None")
            return None
        theta_orb_deg, rho_orb = orbital_pos

        # Scientific validation of orbital prediction
        if not (0.001 <= rho_orb <= 100.0):
            logger.warning(f"Orbital separation prediction {rho_orb:.4f}\" outside typical range [0.001\", 100\"]")

        # Calculate the position from the linear model using CORRECT centered intercepts
        # Linear prediction using centered time: position = intercept + slope * (time - mean_epoch)
        # This is the scientifically correct formula for centered regression
        x_lin = intercept_x + vx * (current_date - mean_epoch)
        y_lin = intercept_y + vy * (current_date - mean_epoch)

        # Convert orbital position to Cartesian and calculate deviation
        theta_orb_rad = np.radians(theta_orb_deg)
        x_orb = rho_orb * np.sin(theta_orb_rad)
        y_orb = rho_orb * np.cos(theta_orb_rad)
        
        # The Curvature Index is the Euclidean distance between the two predictions
        curvature_index = np.sqrt((x_orb - x_lin)**2 + (y_orb - y_lin)**2)
        
        # Scientific validation of result
        if curvature_index > MAX_CURVATURE_INDEX_ARCSEC:
            logger.warning(f"Large curvature index {curvature_index:.3f}\" > {MAX_CURVATURE_INDEX_ARCSEC}\" - orbital or linear model may be incorrect")
        
        if curvature_index < MIN_RESIDUAL_SIGNIFICANCE:
            logger.debug(f"Very small curvature index {curvature_index:.6f}\" - motion appears nearly linear")
        
        logger.debug(f"Curvature index calculated: {curvature_index:.4f}\" at epoch {current_date:.1f}")
        
        return curvature_index

    except Exception as e:
        logger.error(f"Curvature index calculation failed: {e}")
        return None

def calculate_mean_velocity_from_endpoints(
    wds_summary: WdsSummary
) -> Optional[Dict[str, Any]]:
    """
    Calculates mean velocity using only the first and last observations with
    centralized configuration and scientific validation.
    
    This is a fallback method for when insufficient measurements are available
    for robust analysis. It provides a simple two-point velocity estimate
    with proper validation using framework configuration.
    
    DEPRECATED: This method is maintained for backward compatibility but
    is superseded by calculate_robust_linear_fit() for systems with sufficient
    historical data.
    
    Args:
        wds_summary: WDS summary data containing first and last observations.
        
    Returns:
        Dictionary with velocity components and derived quantities, or None if
        insufficient data.
        
    Notes:
        - Uses MIN_TIME_BASELINE_YEARS for temporal validation
        - Validates velocities against MAX_ASTROMETRIC_VELOCITY_ARCSEC_PER_YEAR
        - Logs scientific warnings for deprecated usage and edge cases
    """
    logger.warning("Using deprecated two-point velocity calculation. Consider using robust linear fit for better accuracy.")
    
    if not wds_summary:
        logger.debug("Endpoint velocity calculation failed: Missing WDS summary")
        return None
        
    required_fields = ['date_first', 'date_last', 'pa_first', 'pa_last', 'sep_first', 'sep_last']
    if not all(field in wds_summary and wds_summary[field] is not None for field in required_fields):
        logger.debug("Endpoint velocity calculation failed: Missing required WDS summary fields")
        return None
    
    try:
        # Extract data with validation
        t1, t2 = wds_summary['date_first'], wds_summary['date_last']
        theta1_deg, theta2_deg = wds_summary['pa_first'], wds_summary['pa_last']
        rho1, rho2 = wds_summary['sep_first'], wds_summary['sep_last']
        
        # Scientific validation of time baseline using centralized configuration
        dt = t2 - t1
        if dt <= 0:  # No time baseline or negative
            logger.debug("Endpoint velocity calculation failed: Non-positive time baseline")
            return None
            
        if dt < MIN_TIME_BASELINE_YEARS:
            logger.warning(f"Short time baseline for endpoint calculation: {dt:.2f} years < {MIN_TIME_BASELINE_YEARS}")
        
        # Scientific validation of input ranges
        if not (0.0 <= theta1_deg <= 360.0) or not (0.0 <= theta2_deg <= 360.0):
            logger.warning("Position angles outside [0°, 360°] range in endpoint calculation")
        
        if rho1 <= 0 or rho2 <= 0:
            logger.warning("Non-positive separations in endpoint calculation")
            
        # Convert to Cartesian coordinates
        theta1_rad, theta2_rad = np.radians(theta1_deg), np.radians(theta2_deg)
        x1, y1 = rho1 * np.sin(theta1_rad), rho1 * np.cos(theta1_rad)
        x2, y2 = rho2 * np.sin(theta2_rad), rho2 * np.cos(theta2_rad)
        
        # Calculate velocity components
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        
        v_total = np.sqrt(vx**2 + vy**2)
        pa_v = np.degrees(np.arctan2(vx, vy)) % 360
        
        # Scientific validation of velocity results using centralized configuration
        if v_total > MAX_ASTROMETRIC_VELOCITY_ARCSEC_PER_YEAR:
            logger.warning(f"High velocity from endpoint calculation: {v_total:.4f}\"/yr > {MAX_ASTROMETRIC_VELOCITY_ARCSEC_PER_YEAR}")
        
        logger.debug(f"Endpoint velocity calculated: {v_total:.4f}\"/yr over {dt:.1f} years")
        
        return {
            'vx_arcsec_per_year': vx,
            'vy_arcsec_per_year': vy,
            'v_total_endpoint': v_total,
            'pa_v_endpoint': pa_v,
            'time_baseline_years': dt,
            'n_points_fit': 2,
            'method': 'two_point_endpoint'
        }
        
    except Exception as e:
        logger.error(f"Endpoint velocity calculation failed: {e}")
        return None