# astrakairos/analyzer/workflows.py
"""
Scientific Analysis Workflows for Binary Star Research.

This module contains the pure algorithmic functions that implement the scientific
workflows for discovery, characterization, and orbital analysis. These functions
are completely independent of CLI or UI concerns.
"""

import logging
import random
from typing import Dict, Any, Optional
from astropy.time import Time

from ..data.source import DataSource, WdsSummary, OrbitalElements
from ..data.validators import HybridValidator
from ..exceptions import AnalysisError, ValidationError, PhysicalityValidationError
from ..physics.dynamics import (
    estimate_velocity_from_endpoints, 
    estimate_velocity_from_endpoints_mc,
    calculate_observation_priority_index,
    calculate_observation_priority_index_mc,
    calculate_robust_linear_fit,
    calculate_robust_linear_fit_bootstrap,
    calculate_prediction_divergence
)
from ..config import (
    MIN_EPOCH_YEAR, MAX_EPOCH_YEAR, MIN_SEPARATION_ARCSEC, MAX_SEPARATION_ARCSEC,
    MIN_POSITION_ANGLE_DEG, MAX_POSITION_ANGLE_DEG,
    ALLOW_SINGLE_EPOCH_SYSTEMS, ENABLE_VALIDATION_WARNINGS, 
    VALIDATION_WARNING_SAMPLE_RATE, DEFAULT_GAIA_MIN_RADIUS,
    CLI_VALUE_NOT_AVAILABLE, CLI_DEFAULT_PHYSICALITY_VALUES
)

log = logging.getLogger(__name__)


def _get_current_decimal_year() -> float:
    """
    Get current time as decimal year using astropy.time.Time.
    
    Returns:
        float: Current decimal year
    """
    return Time.now().decimalyear


def _calculate_search_radius(wds_summary: WdsSummary, analysis_config: Dict[str, Any]) -> float:
    """
    Calculate Gaia search radius based on system separation.
    
    Calculates the appropriate search radius for Gaia catalog queries
    based on the last separation measurement and configuration parameters.
    
    Args:
        wds_summary: WDS summary data
        analysis_config: Analysis configuration with radius parameters
        
    Returns:
        float: Search radius in arcseconds
    """
    if wds_summary.get('sep_last') is not None:
        base_radius = wds_summary['sep_last'] * analysis_config['gaia_radius_factor']
        return max(analysis_config['gaia_min_radius'], 
                  min(base_radius, analysis_config['gaia_max_radius']))
    
    return DEFAULT_GAIA_MIN_RADIUS


def _should_log_validation_warning() -> bool:
    """Determine if a validation warning should be logged based on sampling rate."""
    return ENABLE_VALIDATION_WARNINGS and random.random() < VALIDATION_WARNING_SAMPLE_RATE


async def _perform_discovery_analysis(wds_id: str, wds_summary: WdsSummary, data_source: DataSource) -> Dict[str, Any]:
    """
    Perform discovery mode analysis for basic motion estimation with uncertainty propagation.
    
    Args:
        wds_id: WDS identifier
        wds_summary: WDS summary data
        data_source: Data source for retrieving data
        
    Returns:
        Dict containing discovery analysis results with uncertainties
        
    Raises:
        AnalysisError: When analysis computation fails
    """
    log.debug(f"Running discovery analysis for {wds_id}")
    
    try:
        # Check if we have sufficient data for endpoint analysis
        required_fields = ['date_first', 'pa_first', 'sep_first']
        missing_fields = [f for f in required_fields if not wds_summary.get(f)]
        
        if missing_fields:
            raise AnalysisError(f"Missing required fields for discovery analysis: {missing_fields}")
        
        # For single-epoch systems, we can still do basic analysis
        if not wds_summary.get('date_last') and not ALLOW_SINGLE_EPOCH_SYSTEMS:
            raise AnalysisError(f"Single-epoch system not allowed: {wds_id}")
        
        # Perform endpoint velocity estimation with Monte Carlo uncertainties
        if wds_summary.get('date_last'):
            # Multi-epoch analysis
            velocity_result = estimate_velocity_from_endpoints_mc(wds_summary)
        else:
            # Single-epoch analysis (limited information)
            velocity_result = estimate_velocity_from_endpoints(wds_summary)
            velocity_result.update({
                'v_total_uncertainty': None,
                'uncertainty_quality': 0.0
            })
        
        log.debug(f"Discovery analysis successful for {wds_id}")
        return velocity_result
        
    except Exception as e:
        raise AnalysisError(f"Discovery analysis failed for {wds_id}") from e


async def _perform_characterize_analysis(wds_id: str, data_source: DataSource) -> Dict[str, Any]:
    """
    Perform characterize mode analysis with robust fitting and bootstrap uncertainties.
    
    Args:
        wds_id: WDS identifier
        data_source: Data source for measurements
        
    Returns:
        Dict containing characterization results with uncertainties
        
    Raises:
        AnalysisError: When analysis computation fails
    """
    log.debug(f"Running characterization analysis for {wds_id}")
    
    try:
        # Get all measurements for robust fitting
        measurements = await data_source.get_measurements(wds_id)
        if not measurements or len(measurements) < 3:
            raise AnalysisError(f"Insufficient measurements for characterization: {wds_id}")
        
        # Perform robust linear fit with bootstrap uncertainties
        robust_result = calculate_robust_linear_fit_bootstrap(measurements)
        
        log.debug(f"Characterization analysis successful for {wds_id}")
        return robust_result
        
    except Exception as e:
        raise AnalysisError(f"Characterization analysis failed for {wds_id}") from e


async def _calculate_system_masses(
    wds_id: str,
    orbital_elements: OrbitalElements,
    wds_summary: WdsSummary,
    gaia_validator: Optional[HybridValidator],
    parallax_source: str
) -> Optional[Dict[str, Any]]:
    """
    Calculate system masses using Kepler's Third Law and available parallax data.
    
    Args:
        wds_id: WDS identifier
        orbital_elements: Orbital elements dictionary
        wds_summary: WDS summary data
        gaia_validator: Gaia validator for parallax queries
        parallax_source: Preferred parallax source
        
    Returns:
        Dictionary with mass calculation results or None if failed
    """
    try:
        from ..physics.masses import calculate_total_mass_kepler3
        
        # Check if we have the required orbital elements
        period = orbital_elements.get('P')
        semimajor_axis = orbital_elements.get('a')
        
        if not period or not semimajor_axis:
            log.debug(f"Missing orbital elements for mass calculation: {wds_id}")
            return None
        
        # Get uncertainties if available
        period_error = orbital_elements.get('e_P', 0.0) or 0.0
        semimajor_axis_error = orbital_elements.get('e_a', 0.0) or 0.0
        
        # Get parallax data based on source preference
        parallax_data = None
        
        if parallax_source == 'auto' or parallax_source == 'gaia':
            if gaia_validator:
                try:
                    parallax_data = await gaia_validator.get_parallax_data(wds_summary)
                    if parallax_data:
                        log.debug(f"Using Gaia parallax for {wds_id}: {parallax_data['parallax']:.3f} ± {parallax_data['parallax_error']:.3f} mas")
                except Exception as e:
                    log.warning(f"Failed to get Gaia parallax for {wds_id}: {e}")
        
        # Could add other parallax sources here (Hipparcos, literature, etc.)
        # For now, only Gaia is implemented
        
        if not parallax_data:
            log.debug(f"No parallax data available for mass calculation: {wds_id}")
            return None
        
        # Calculate masses
        try:
            mass_result = calculate_total_mass_kepler3(
                period_years=period,
                semimajor_axis_arcsec=semimajor_axis,
                parallax_mas=parallax_data['parallax'],
                period_error=period_error,
                semimajor_axis_error=semimajor_axis_error,
                parallax_error=parallax_data['parallax_error'],
                parallax_source=parallax_data['source']
            )
            
            log.debug(f"Mass calculation successful for {wds_id}: {mass_result.total_mass_solar:.2f} ± {mass_result.total_mass_error:.2f} M☉")
            
            # Convert to dictionary for JSON serialization
            return {
                'total_mass_solar': mass_result.total_mass_solar,
                'total_mass_error': mass_result.total_mass_error,
                'distance_pc': mass_result.distance_used_pc,
                'parallax_mas': mass_result.parallax_used_mas,
                'parallax_source': mass_result.parallax_source,
                'quality_score': mass_result.quality_score,
                'mc_samples': mass_result.mc_samples,
                'warnings': mass_result.warnings
            }
            
        except Exception as e:
            raise AnalysisError(f"Mass calculation failed for {wds_id}") from e
            
    except Exception as e:
        raise AnalysisError(f"Error in mass calculation for {wds_id}") from e


async def _perform_orbital_analysis(wds_id: str, wds_summary: WdsSummary, data_source: DataSource, 
                                   gaia_validator: Optional[HybridValidator] = None,
                                   calculate_masses: bool = False,
                                   parallax_source: str = 'auto') -> Dict[str, Any]:
    """
    Perform orbital mode analysis with OPI calculation and optional mass calculation.
    
    Args:
        wds_id: WDS identifier
        wds_summary: WDS summary data
        data_source: Data source for orbital elements
        gaia_validator: Optional Gaia validator for parallax data
        calculate_masses: Whether to calculate system masses
        parallax_source: Source preference for parallax ('auto', 'gaia', 'none')
        
    Returns:
        Dict containing orbital analysis results
        
    Raises:
        AnalysisError: When analysis computation fails
    """
    log.debug(f"Running orbital analysis for {wds_id}")
    
    try:
        # Get orbital elements
        orbital_elements = await data_source.get_orbital_elements(wds_id)
        if not orbital_elements:
            raise AnalysisError(f"No orbital elements found for {wds_id}")
        
        # Calculate current epoch for OPI calculation
        current_epoch = _get_current_decimal_year()
        
        # Perform OPI calculation with Monte Carlo uncertainties
        opi_result = calculate_observation_priority_index_mc(
            orbital_elements, current_epoch
        )
        
        # Calculate prediction divergence for the orbital solution
        prediction_div = calculate_prediction_divergence(
            orbital_elements, wds_summary
        )
        opi_result['prediction_divergence_arcsec'] = prediction_div
        
        # Optional mass calculation
        if calculate_masses and parallax_source != 'none':
            try:
                mass_result = await _calculate_system_masses(
                    wds_id, orbital_elements, wds_summary, 
                    gaia_validator, parallax_source
                )
                if mass_result:
                    opi_result.update(mass_result)
            except AnalysisError as e:
                # Mass calculation failure is not fatal for orbital analysis
                log.warning(f"Mass calculation failed for {wds_id}: {e}")
                # Continue without masses
        
        log.debug(f"Orbital analysis successful for {wds_id}")
        return opi_result
        
    except Exception as e:
        raise AnalysisError(f"Orbital analysis failed for {wds_id}") from e


async def _perform_gaia_validation(wds_id: str, wds_summary: WdsSummary, 
                                   gaia_validator, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform Gaia validation for physicality assessment.
    
    Args:
        wds_id: WDS identifier
        wds_summary: WDS summary data
        gaia_validator: Gaia validator instance
        analysis_config: Analysis configuration for search radius calculation
        
    Returns:
        Dict containing Gaia validation results
        
    Raises:
        PhysicalityValidationError: When validation fails
    """
    log.debug(f"Running Gaia validation for {wds_id}")
    
    try:
        search_radius = _calculate_search_radius(wds_summary, analysis_config)
        log.debug(f"Calculated search radius: {search_radius}")
        
        # Create system_data dict as expected by HybridValidator
        system_data = {
            'wds_id': wds_id,
            'component_pair': 'AB',  # Default component pair
            **wds_summary  # Include all summary data
        }
        
        log.debug(f"Calling validator.validate_physicality with system_data keys: {list(system_data.keys())}")
        physicality_assessment = await gaia_validator.validate_physicality(
            system_data, search_radius_arcsec=search_radius
        )
        log.debug(f"Physicality assessment result: {physicality_assessment}")
        
        if physicality_assessment:
            result = {
                'physicality_label': physicality_assessment.get('label').value if hasattr(physicality_assessment.get('label'), 'value') else str(physicality_assessment.get('label')),
                'physicality_p_value': physicality_assessment.get('p_value'),
                'physicality_method': physicality_assessment.get('method').value if hasattr(physicality_assessment.get('method'), 'value') else str(physicality_assessment.get('method')),
                'physicality_confidence': physicality_assessment.get('confidence')
            }
            log.debug(f"Returning result: {result}")
            return result
        else:
            # Return default values for failed validation
            return CLI_DEFAULT_PHYSICALITY_VALUES['ERROR'].copy()
            
    except Exception as e:
        if isinstance(e, PhysicalityValidationError):
            raise
        else:
            raise PhysicalityValidationError(f"Gaia validation failed for {wds_id}") from e
