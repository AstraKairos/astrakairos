"""
Orbital dynamics and motion analysis for binary star systems.

Provides tools for:
- Observation Priority Index (OPI): Quantifies orbital solution degradation rate
- Robust linear fitting: Theil-Sen regression for outlier-resistant motion analysis  
- Curvature index: Measures deviation between orbital and linear motion models
- Total observed angular velocity: Two-point velocity estimate for sparse data scenarios

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
    MIN_POINTS_FOR_ROBUST_FIT,
    ROBUST_REGRESSION_RANDOM_STATE,
    MAX_ASTROMETRIC_VELOCITY_ARCSEC_PER_YEAR,
    MIN_TIME_BASELINE_YEARS,
    MAX_CURVATURE_INDEX_ARCSEC,
    MIN_PREDICTION_DATE_OFFSET_YEARS,
    MAX_PREDICTION_DATE_OFFSET_YEARS,
    MAX_EXTRAPOLATION_FACTOR,
    MIN_EPOCH_YEAR,
    MAX_EPOCH_YEAR,
    MIN_SEPARATION_ARCSEC,
    MAX_SEPARATION_ARCSEC,
    MAX_DEVIATION_WARNING_ARCSEC,
    # New Monte Carlo and fallback error constants
    ORB6_FALLBACK_ERRORS,
    WDS_FALLBACK_ERRORS,
    DEFAULT_MC_SAMPLES,
    MC_CONFIDENCE_LEVEL,
    MC_RANDOM_SEED
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
        
    # Get observation data with fallback to first observation if last is not available
    t_last_obs = wds_summary.get('date_last') or wds_summary.get('date_first')
    theta_last_obs_deg = wds_summary.get('pa_last') or wds_summary.get('pa_first')
    rho_last_obs = wds_summary.get('sep_last') or wds_summary.get('sep_first')

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
        if 'epoch' not in measurements.colnames or 'theta' not in measurements.colnames or 'rho' not in measurements.colnames:
            logger.error(f"Missing required columns in measurements. Available: {measurements.colnames}")
            return None
            
        # Convert to numpy arrays and handle potential None/masked values
        epoch_data = measurements['epoch']
        theta_data = measurements['theta']
        rho_data = measurements['rho']
        
        # Filter out None values and convert to numpy arrays
        valid_indices = []
        for i in range(len(epoch_data)):
            if (epoch_data[i] is not None and theta_data[i] is not None and rho_data[i] is not None):
                valid_indices.append(i)
        
        if len(valid_indices) < MIN_POINTS_FOR_ROBUST_FIT:
            logger.warning(f"Insufficient valid measurements: {len(valid_indices)} < {MIN_POINTS_FOR_ROBUST_FIT}")
            return None
            
        # Extract valid data
        t = np.array([epoch_data[i] for i in valid_indices]).reshape(-1, 1)
        theta_values = np.array([theta_data[i] for i in valid_indices])
        rho = np.array([rho_data[i] for i in valid_indices])
        
        # Convert theta to radians
        theta_rad = np.radians(theta_values)

        # Validate input data ranges
        epoch_range = np.max(t) - np.min(t)
        if epoch_range < MIN_TIME_BASELINE_YEARS:
            logger.warning(f"Time baseline too short: {epoch_range:.2f} years < {MIN_TIME_BASELINE_YEARS}")
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


def calculate_robust_linear_fit_bootstrap(
    measurements: Table, 
    num_bootstrap: int = 500
) -> Optional[Dict[str, Any]]:
    """
    Perform robust linear fit with bootstrap uncertainty estimation.
    
    This function extends the standard robust linear fit by using bootstrap
    resampling to estimate uncertainties in the velocity parameters. The bootstrap
    approach is more appropriate than Monte Carlo for this case since the
    uncertainties come from the scatter in the data rather than measurement errors.
    
    Args:
        measurements: Astropy Table with 'epoch', 'theta', 'rho' columns
        num_bootstrap: Number of bootstrap samples for uncertainty estimation
        
    Returns:
        Dictionary with velocity estimates, uncertainties, and fit statistics
    """
    # First, get the main fit result
    main_result = calculate_robust_linear_fit(measurements)
    if main_result is None:
        return None
    
    try:
        # Prepare data (same as in main function)
        if 'epoch' not in measurements.colnames or 'theta' not in measurements.colnames or 'rho' not in measurements.colnames:
            return None
            
        epoch_data = measurements['epoch']
        theta_data = measurements['theta']
        rho_data = measurements['rho']
        
        # Filter valid data
        valid_indices = []
        for i in range(len(epoch_data)):
            if (epoch_data[i] is not None and theta_data[i] is not None and rho_data[i] is not None):
                valid_indices.append(i)
        
        if len(valid_indices) < MIN_POINTS_FOR_ROBUST_FIT:
            return None
            
        # Extract valid data
        t = np.array([epoch_data[i] for i in valid_indices])
        theta_values = np.array([theta_data[i] for i in valid_indices])
        rho = np.array([rho_data[i] for i in valid_indices])
        
        # Convert to Cartesian
        theta_rad = np.radians(theta_values)
        x = rho * np.sin(theta_rad)
        y = rho * np.cos(theta_rad)
        
        # Center time data
        t_mean = np.mean(t)
        t_centered = t - t_mean
        
        # Bootstrap sampling
        n_points = len(t)
        vx_bootstrap = []
        vy_bootstrap = []
        
        np.random.seed(MC_RANDOM_SEED)  # For reproducibility
        
        for _ in range(num_bootstrap):
            # Resample with replacement
            bootstrap_indices = np.random.choice(n_points, size=n_points, replace=True)
            
            t_boot = t_centered[bootstrap_indices].reshape(-1, 1)
            x_boot = x[bootstrap_indices]
            y_boot = y[bootstrap_indices]
            
            try:
                # Fit on bootstrap sample
                theil_sen_x_boot = TheilSenRegressor(random_state=ROBUST_REGRESSION_RANDOM_STATE)
                theil_sen_y_boot = TheilSenRegressor(random_state=ROBUST_REGRESSION_RANDOM_STATE)
                
                theil_sen_x_boot.fit(t_boot, x_boot)
                theil_sen_y_boot.fit(t_boot, y_boot)
                
                vx_bootstrap.append(theil_sen_x_boot.coef_[0])
                vy_bootstrap.append(theil_sen_y_boot.coef_[0])
                
            except Exception:
                continue  # Skip failed fits
        
        if len(vx_bootstrap) == 0:
            # If bootstrap failed, return main result without uncertainties
            main_result.update({
                'vx_uncertainty': None,
                'vy_uncertainty': None,
                'v_total_uncertainty': None,
                'pa_v_uncertainty': None,
                'uncertainty_method': 'none'
            })
            return main_result
        
        # Calculate bootstrap statistics
        vx_stats = _calculate_mc_statistics(np.array(vx_bootstrap))
        vy_stats = _calculate_mc_statistics(np.array(vy_bootstrap))
        
        # Calculate derived quantities (v_total, pa_v) for each bootstrap sample
        v_total_bootstrap = []
        pa_v_bootstrap = []
        
        for vx_b, vy_b in zip(vx_bootstrap, vy_bootstrap):
            v_total_b = np.sqrt(vx_b**2 + vy_b**2)
            pa_v_b = np.degrees(np.arctan2(vx_b, vy_b)) % 360
            v_total_bootstrap.append(v_total_b)
            pa_v_bootstrap.append(pa_v_b)
        
        v_total_stats = _calculate_mc_statistics(np.array(v_total_bootstrap))
        pa_v_stats = _calculate_mc_statistics(np.array(pa_v_bootstrap))
        
        # Add uncertainty information to main result
        main_result.update({
            'vx_uncertainty': vx_stats['uncertainty'],
            'vy_uncertainty': vy_stats['uncertainty'],
            'v_total_uncertainty': v_total_stats['uncertainty'],
            'pa_v_uncertainty': pa_v_stats['uncertainty'],
            'uncertainty_method': 'bootstrap',
            'num_bootstrap_samples': len(vx_bootstrap),
            'bootstrap_success_rate': len(vx_bootstrap) / num_bootstrap
        })
        
        return main_result
        
    except Exception as e:
        logger.error(f"Bootstrap linear fit failed: {e}")
        return main_result  # Return main result if bootstrap fails


def calculate_prediction_divergence(
    orbital_elements: OrbitalElements,
    linear_fit_results: Dict[str, Any],
    current_date: float
) -> Optional[float]:
    """
    Calculates the Prediction Divergence using centralized configuration.
    
    This metric quantifies the deviation between an orbital model prediction and a robust 
    linear fit prediction at a specific date, providing insight into orbital motion significance.

    Args:
        orbital_elements: The 7 Keplerian orbital elements.
        linear_fit_results: Complete results from calculate_robust_linear_fit()
                           including vx, vy, intercept_x, intercept_y, and mean_epoch_fit.
        current_date: The date (in decimal years) to evaluate the deviation.

    Returns:
        The Prediction Divergence in arcseconds, or None if calculation fails.
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
        
        # The Prediction Divergence is the Euclidean distance between the two predictions
        prediction_divergence = np.sqrt((x_orb - x_lin)**2 + (y_orb - y_lin)**2)
        
        # Log warning only for extreme cases
        if prediction_divergence > MAX_CURVATURE_INDEX_ARCSEC:
            logger.warning(f"Large prediction divergence {prediction_divergence:.3f}\" > {MAX_CURVATURE_INDEX_ARCSEC}\"")
        
        return prediction_divergence

    except Exception as e:
        logger.error(f"Prediction divergence calculation failed: {e}")
        return None

def estimate_velocity_from_endpoints(
    wds_summary: WdsSummary
) -> Optional[Dict[str, Any]]:
    """
    Calculates velocity estimate (aka. total observed angular velocity) using 
    only the first and last observations.
    
    This is a method for when insufficient measurements are available
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


# --- Monte Carlo Error Propagation Functions ---

def _calculate_mc_statistics(samples: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics from Monte Carlo samples.
    
    Args:
        samples: Array of samples from Monte Carlo simulation
        
    Returns:
        Dictionary with median, uncertainty, and percentiles
    """
    if len(samples) == 0:
        return {'median': np.nan, 'uncertainty': np.nan}
    
    # Calculate percentiles for confidence interval
    lower_percentile = (100.0 - MC_CONFIDENCE_LEVEL) / 2.0
    upper_percentile = 100.0 - lower_percentile
    
    median = np.median(samples)
    p_lower = np.percentile(samples, lower_percentile)
    p_upper = np.percentile(samples, upper_percentile)
    
    # Uncertainty is half the confidence interval width
    uncertainty = (p_upper - p_lower) / 2.0
    
    return {
        'median': median,
        'uncertainty': uncertainty,
        'p_lower': p_lower,
        'p_upper': p_upper
    }


def estimate_velocity_from_endpoints_mc(
    wds_summary: WdsSummary,
    num_samples: int = DEFAULT_MC_SAMPLES
) -> Optional[Dict[str, Any]]:
    """
    Calculate velocity estimate with uncertainty propagation using Monte Carlo.
    
    This function performs uncertainty propagation for the two-point velocity
    estimate by sampling from the measurement error distributions.
    
    Args:
        wds_summary: WDS summary data with error fields
        num_samples: Number of Monte Carlo samples
        
    Returns:
        Dictionary with velocity estimates, uncertainties, and quality metrics
    """
    if not wds_summary:
        return None
        
    # Check for required basic fields
    required_fields = ['date_first', 'date_last', 'pa_first', 'pa_last', 'sep_first', 'sep_last']
    if not all(field in wds_summary and wds_summary[field] is not None for field in required_fields):
        return None
    
    # Check for error fields and determine if Monte Carlo is possible
    error_fields = ['pa_first_error', 'pa_last_error', 'sep_first_error', 'sep_last_error']
    has_errors = [wds_summary.get(field) is not None and wds_summary[field] > 0 
                  for field in error_fields]
    
    # If no measured errors are available, fall back to point estimate
    if not any(has_errors):
        logger.debug("No measurement errors available, using point estimate")
        result = estimate_velocity_from_endpoints(wds_summary)
        if result:
            result.update({
                'quality': 'point_estimate',
                'uncertainty_source': 'none',
                'v_total_uncertainty': None,
                'pa_v_uncertainty': None
            })
        return result
    
    # Set up error values (measured or fallback)
    pa_first_error = wds_summary.get('pa_first_error') or WDS_FALLBACK_ERRORS['pa_error']
    pa_last_error = wds_summary.get('pa_last_error') or WDS_FALLBACK_ERRORS['pa_error']
    sep_first_error = wds_summary.get('sep_first_error') or WDS_FALLBACK_ERRORS['sep_error']
    sep_last_error = wds_summary.get('sep_last_error') or WDS_FALLBACK_ERRORS['sep_error']
    
    # Count measured vs fallback errors for quality score
    num_measured = sum(has_errors)
    quality_score = num_measured / len(error_fields)
    
    try:
        # Set random seed for reproducibility
        np.random.seed(MC_RANDOM_SEED)
        
        # Extract base values
        t1, t2 = wds_summary['date_first'], wds_summary['date_last']
        dt = t2 - t1
        
        # Validate time baseline
        if dt <= 0 or dt < MIN_TIME_BASELINE_YEARS:
            return None
            
        # Monte Carlo sampling - vectorized for better performance
        np.random.seed(MC_RANDOM_SEED)
        
        # Generate all samples at once (vectorized)
        pa1_samples = np.random.normal(wds_summary['pa_first'], pa_first_error, size=num_samples)
        pa2_samples = np.random.normal(wds_summary['pa_last'], pa_last_error, size=num_samples)
        sep1_samples = np.random.normal(wds_summary['sep_first'], sep_first_error, size=num_samples)
        sep2_samples = np.random.normal(wds_summary['sep_last'], sep_last_error, size=num_samples)
        
        # Ensure positive separations (vectorized)
        sep1_samples = np.maximum(sep1_samples, 0.001)
        sep2_samples = np.maximum(sep2_samples, 0.001)
        
        # Convert to Cartesian coordinates (vectorized)
        theta1_rad = np.radians(pa1_samples % 360)
        theta2_rad = np.radians(pa2_samples % 360)
        
        x1 = sep1_samples * np.sin(theta1_rad)
        y1 = sep1_samples * np.cos(theta1_rad)
        x2 = sep2_samples * np.sin(theta2_rad)
        y2 = sep2_samples * np.cos(theta2_rad)
        
        # Calculate velocities (vectorized)
        dt = wds_summary['date_last'] - wds_summary['date_first']
        vx_samples = (x2 - x1) / dt
        vy_samples = (y2 - y1) / dt
        
        # Calculate derived quantities (vectorized)
        v_total_samples = np.sqrt(vx_samples**2 + vy_samples**2)
        pa_v_samples = np.degrees(np.arctan2(vx_samples, vy_samples)) % 360
        
        # Calculate statistics
        v_total_stats = _calculate_mc_statistics(v_total_samples)
        pa_v_stats = _calculate_mc_statistics(pa_v_samples)
        
        # Determine uncertainty source
        uncertainty_source = 'mixed' if 0 < num_measured < len(error_fields) else (
            'measured' if num_measured == len(error_fields) else 'fallback')
        
        return {
            'v_total_median': v_total_stats['median'],
            'v_total_uncertainty': v_total_stats['uncertainty'],
            'pa_v_median': pa_v_stats['median'],
            'pa_v_uncertainty': pa_v_stats['uncertainty'],
            'time_baseline_years': dt,
            'n_points_fit': 2,
            'quality': 'monte_carlo',
            'uncertainty_source': uncertainty_source,
            'quality_score': quality_score,
            'num_samples': num_samples,
            'method': 'two_point_mc'
        }
        
    except Exception as e:
        logger.error(f"Monte Carlo velocity calculation failed: {e}")
        return None


def calculate_observation_priority_index_mc(
    orbital_elements: OrbitalElements,
    wds_summary: WdsSummary,
    current_date: float,
    num_samples: int = DEFAULT_MC_SAMPLES
) -> Optional[Dict[str, Any]]:
    """
    Calculate Observation Priority Index with uncertainty propagation using Monte Carlo.
    
    This function propagates uncertainties from both orbital elements and 
    the last measurement to provide OPI with confidence intervals.
    
    Args:
        orbital_elements: Orbital elements with uncertainties
        wds_summary: WDS summary with measurement errors
        current_date: Current date for OPI calculation
        num_samples: Number of Monte Carlo samples
        
    Returns:
        Dictionary with OPI statistics, uncertainties, and quality metrics
    """
    if not orbital_elements or not wds_summary:
        return None
    
    # Check for required observation data with fallback
    t_last_obs = wds_summary.get('date_last') or wds_summary.get('date_first')
    theta_last_obs_deg = wds_summary.get('pa_last') or wds_summary.get('pa_first')
    rho_last_obs = wds_summary.get('sep_last') or wds_summary.get('sep_first')
    
    if None in [t_last_obs, theta_last_obs_deg, rho_last_obs]:
        return None
    
    # Basic validation using centralized configuration
    if not (MIN_EPOCH_YEAR <= t_last_obs <= MAX_EPOCH_YEAR):
        return None
    if not (0.0 <= theta_last_obs_deg <= 360.0):
        return None
    if not (MIN_SEPARATION_ARCSEC <= rho_last_obs <= MAX_SEPARATION_ARCSEC):
        return None
    
    # Collect all 9 parameters with their errors
    orbital_params = ['P', 'T', 'e', 'a', 'i', 'Omega', 'omega']
    measurement_params = ['pa_last', 'sep_last']  # Use last measurement for OPI
    
    # Build parameter dictionary with error handling
    params = {}
    errors = {}
    measured_error_count = 0
    total_param_count = len(orbital_params) + len(measurement_params)
    
    # Handle orbital parameters
    for param in orbital_params:
        if param in orbital_elements and orbital_elements[param] is not None:
            params[param] = orbital_elements[param]
            
            # Check for measured error
            error_key = f'e_{param}'
            if error_key in orbital_elements and orbital_elements[error_key] is not None and orbital_elements[error_key] > 0:
                errors[param] = orbital_elements[error_key]
                measured_error_count += 1
            else:
                # Use fallback error
                if param == 'P':
                    # Period fallback: 10% of the period value
                    errors[param] = params[param] * ORB6_FALLBACK_ERRORS['e_P']
                else:
                    errors[param] = ORB6_FALLBACK_ERRORS[f'e_{param}']
        else:
            return None  # Missing required orbital parameter
    
    # Handle measurement parameters
    if theta_last_obs_deg is not None:
        params['pa_last'] = theta_last_obs_deg
        pa_error_key = 'pa_last_error'
        if pa_error_key in wds_summary and wds_summary[pa_error_key] is not None and wds_summary[pa_error_key] > 0:
            errors['pa_last'] = wds_summary[pa_error_key]
            measured_error_count += 1
        else:
            errors['pa_last'] = WDS_FALLBACK_ERRORS['pa_error']
    
    if rho_last_obs is not None:
        params['sep_last'] = rho_last_obs
        sep_error_key = 'sep_last_error'
        if sep_error_key in wds_summary and wds_summary[sep_error_key] is not None and wds_summary[sep_error_key] > 0:
            errors['sep_last'] = wds_summary[sep_error_key]
            measured_error_count += 1
        else:
            errors['sep_last'] = WDS_FALLBACK_ERRORS['sep_error']
    
    # Calculate quality score
    quality_score = measured_error_count / total_param_count
    
    try:
        # Set random seed for reproducibility
        np.random.seed(MC_RANDOM_SEED)
        
        # Monte Carlo sampling
        opi_samples = []
        deviation_samples = []
        
        for _ in range(num_samples):
            # Sample orbital elements
            sampled_elements = {}
            for param in orbital_params:
                sampled_elements[param] = np.random.normal(params[param], errors[param])
            
            # Sample measurement values
            pa_sample = np.random.normal(params['pa_last'], errors['pa_last']) % 360
            rho_sample = max(np.random.normal(params['sep_last'], errors['sep_last']), 0.001)
            
            # Validate sampled orbital elements (basic constraints)
            if sampled_elements['e'] < 0 or sampled_elements['e'] >= 1:
                continue  # Skip invalid eccentricity
            if sampled_elements['a'] <= 0:
                continue  # Skip invalid semi-major axis
            if sampled_elements['P'] <= 0:
                continue  # Skip invalid period
            
            try:
                # Predict position using sampled orbital elements
                predicted_pos = predict_position(sampled_elements, t_last_obs)
                if predicted_pos is None:
                    continue
                
                theta_pred_deg, rho_pred = predicted_pos
                
                # Calculate deviation in Cartesian coordinates
                theta_pred_rad = np.radians(theta_pred_deg)
                theta_obs_rad = np.radians(pa_sample)
                
                x_pred = rho_pred * np.sin(theta_pred_rad)
                y_pred = rho_pred * np.cos(theta_pred_rad)
                x_obs = rho_sample * np.sin(theta_obs_rad)
                y_obs = rho_sample * np.cos(theta_obs_rad)
                
                deviation = np.sqrt((x_pred - x_obs)**2 + (y_pred - y_obs)**2)
                time_since_obs = current_date - t_last_obs
                
                # Calculate OPI
                if time_since_obs <= 0:
                    opi = np.inf if deviation > OPI_DEVIATION_THRESHOLD_ARCSEC else 0.0
                else:
                    opi = deviation / time_since_obs
                
                opi_samples.append(opi)
                deviation_samples.append(deviation)
                
            except Exception:
                continue  # Skip failed predictions
        
        if len(opi_samples) == 0:
            return None
        
        # Calculate statistics
        opi_stats = _calculate_mc_statistics(np.array(opi_samples))
        deviation_stats = _calculate_mc_statistics(np.array(deviation_samples))
        
        # Determine uncertainty source
        uncertainty_source = ('mixed' if 0 < measured_error_count < total_param_count else
                            'measured' if measured_error_count == total_param_count else 'fallback')
        
        return {
            'opi_median': opi_stats['median'],
            'opi_uncertainty': opi_stats['uncertainty'],
            'deviation_median': deviation_stats['median'],
            'deviation_uncertainty': deviation_stats['uncertainty'],
            'quality': 'monte_carlo',
            'uncertainty_source': uncertainty_source,
            'quality_score': quality_score,
            'num_samples': len(opi_samples),
            'num_valid_samples': len(opi_samples),
            'method': 'opi_mc'
        }
        
    except Exception as e:
        logger.error(f"Monte Carlo OPI calculation failed: {e}")
        return None