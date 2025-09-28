"""
Gaia Data Utilities - Error Correction and Quality Assessment

This module provides utilities for working with Gaia astrometric data,
including RUWE-based error correction and quality assessment functions.

The error correction implements a conservative approach where errors are
inflated based on the Renormalised Unit Weight Error (RUWE) to account
for systematic uncertainties in poor astrometric solutions.
"""

import numpy as np
import logging
from typing import Optional, Union, Dict, Any

from ..config import (
    RUWE_CORRECTION_ENABLED,
    RUWE_CORRECTION_MAX_FACTOR,
    RUWE_CORRECTION_THRESHOLD,  # Updated name
    RUWE_CORRECTION_APPLY_TO_ALL_DIMENSIONS,
    # Fallback error constants
    FALLBACK_PARALLAX_ERROR_MAS,
    FALLBACK_PMRA_ERROR_MAS_PER_YEAR,
    FALLBACK_PMDEC_ERROR_MAS_PER_YEAR,
    FALLBACK_RA_ERROR_MAS,
    FALLBACK_DEC_ERROR_MAS
)

log = logging.getLogger(__name__)

def correct_gaia_error(
    measured_error: float,
    ruwe: Optional[float] = None,
    error_type: str = "generic",
    apply_correction: bool = None
) -> float:
    """
    Apply RUWE-based error correction to Gaia astrometric measurements.
    
    This function implements a conservative error inflation strategy based on
    the Renormalised Unit Weight Error (RUWE). Poor astrometric solutions
    (high RUWE) indicate systematic uncertainties that are not captured in
    the formal statistical errors, so we inflate these errors accordingly.
    
    Philosophy: "When in doubt, be more conservative with uncertainties"
    
    Args:
        measured_error: The formal statistical error from Gaia (mas, mas/yr, etc.)
        ruwe: Renormalised Unit Weight Error (dimensionless)
        error_type: Type of error for logging ('parallax', 'pmra', 'pmdec', etc.)
        apply_correction: Override global correction setting
        
    Returns:
        Corrected error (same units as input)
        
    Notes:
        - If RUWE < RUWE_CORRECTION_MIN_RUWE, no correction is applied
        - Correction factor is capped at RUWE_CORRECTION_MAX_FACTOR
        - If RUWE is None/missing, returns original error unchanged
        
    Example:
        >>> # Good astrometric solution
        >>> correct_gaia_error(0.1, ruwe=1.0)  # Returns 0.1 (no correction)
        >>> 
        >>> # Poor astrometric solution  
        >>> correct_gaia_error(0.1, ruwe=2.5)  # Returns 0.25 (2.5x inflation)
        >>> 
        >>> # Very poor solution (capped)
        >>> correct_gaia_error(0.1, ruwe=5.0)  # Returns 0.3 (3x cap applied)
    """
    # Check if correction is enabled
    correction_enabled = apply_correction if apply_correction is not None else RUWE_CORRECTION_ENABLED
    if not correction_enabled:
        log.debug(f"RUWE correction disabled, returning original {error_type} error: {measured_error}")
        return measured_error
    
    # Validate inputs
    if measured_error <= 0:
        log.warning(f"Invalid {error_type} error: {measured_error} <= 0, returning as-is")
        return measured_error
    
    # If RUWE is missing or invalid, return original error
    if ruwe is None or not np.isfinite(ruwe) or ruwe <= 0:
        log.debug(f"RUWE unavailable for {error_type} error correction, using original error: {measured_error}")
        return measured_error
    
    # Apply correction threshold
    if ruwe < RUWE_CORRECTION_THRESHOLD:
        log.debug(f"RUWE ({ruwe:.2f}) below threshold ({RUWE_CORRECTION_THRESHOLD}), "
                 f"no correction applied to {error_type} error")
        return measured_error
    
    # Calculate correction factor (capped at maximum)
    correction_factor = min(ruwe, RUWE_CORRECTION_MAX_FACTOR)
    corrected_error = measured_error * correction_factor
    
    log.debug(f"RUWE {error_type} correction: {measured_error:.4f} → {corrected_error:.4f} "
             f"(RUWE={ruwe:.2f}, factor={correction_factor:.2f})")
    
    return corrected_error


def get_safe_error_with_fallback(
    measured_error: Optional[float],
    fallback_error: float,
    ruwe: Optional[float] = None,
    error_type: str = "generic"
) -> float:
    """
    Get a safe error value with conservative fallback for missing data.
    
    This function implements the "Safe Error Fallback" strategy:
    1. If measured error is available, apply RUWE correction and return
    2. If measured error is missing, return conservative fixed fallback
    
    Args:
        measured_error: The measured error from Gaia (may be None/masked)
        fallback_error: Conservative fixed fallback error value
        ruwe: RUWE for error correction (if available)
        error_type: Type of error for logging
        
    Returns:
        Safe error value ready for statistical calculations
        
    Example:
        >>> # Normal case with measured error
        >>> get_safe_error_with_fallback(0.1, 0.5, ruwe=1.2, error_type="parallax")
        0.12  # 0.1 * 1.2 RUWE correction
        
        >>> # Missing error case
        >>> get_safe_error_with_fallback(None, 0.5, error_type="parallax") 
        0.5   # Conservative fallback
    """
    # Check if measured error is available and valid
    if measured_error is not None and np.isfinite(measured_error) and measured_error > 0:
        # Apply RUWE correction to measured error
        corrected_error = correct_gaia_error(measured_error, ruwe, error_type)
        log.debug(f"Using measured {error_type} error: {corrected_error:.4f}")
        return corrected_error
    else:
        # Use conservative fallback
        log.info(f"Missing {error_type} error, using conservative fallback: {fallback_error}")
        return fallback_error


def get_gaia_parallax_error_safe(parallax_error: Optional[float], ruwe: Optional[float] = None) -> float:
    """Get safe parallax error with fallback."""
    return get_safe_error_with_fallback(
        parallax_error, 
        FALLBACK_PARALLAX_ERROR_MAS, 
        ruwe, 
        "parallax"
    )


def get_gaia_pmra_error_safe(pmra_error: Optional[float], ruwe: Optional[float] = None) -> float:
    """Get safe proper motion RA error with fallback."""
    return get_safe_error_with_fallback(
        pmra_error, 
        FALLBACK_PMRA_ERROR_MAS_PER_YEAR, 
        ruwe, 
        "pmra"
    )


def get_gaia_pmdec_error_safe(pmdec_error: Optional[float], ruwe: Optional[float] = None) -> float:
    """Get safe proper motion Dec error with fallback."""
    return get_safe_error_with_fallback(
        pmdec_error, 
        FALLBACK_PMDEC_ERROR_MAS_PER_YEAR, 
        ruwe, 
        "pmdec"
    )


def get_gaia_ra_error_safe(ra_error: Optional[float], ruwe: Optional[float] = None) -> float:
    """Get safe RA error with fallback."""
    return get_safe_error_with_fallback(
        ra_error, 
        FALLBACK_RA_ERROR_MAS, 
        ruwe, 
        "ra"
    )


def get_gaia_dec_error_safe(dec_error: Optional[float], ruwe: Optional[float] = None) -> float:
    """Get safe Dec error with fallback."""
    return get_safe_error_with_fallback(
        dec_error, 
        FALLBACK_DEC_ERROR_MAS, 
        ruwe, 
        "dec"
    )


def assess_gaia_data_quality(star: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive assessment of Gaia data quality for a single source.
    
    Args:
        star: Gaia source data dictionary
        
    Returns:
        Dict with quality assessment:
        - ruwe_quality: bool (True if RUWE acceptable)
        - parallax_quality: bool (True if parallax significant)
        - pm_quality: bool (True if proper motion significant)
        - overall_quality: str ('excellent', 'good', 'fair', 'poor')
        - quality_flags: List of quality issues
    """
    quality = {
        'ruwe_quality': False,
        'parallax_quality': False,
        'pm_quality': False,
        'overall_quality': 'poor',
        'quality_flags': []
    }
    
    # RUWE quality assessment
    ruwe = star.get('ruwe')
    if ruwe is not None and np.isfinite(ruwe):
        if ruwe <= 1.4:  # Lindegren et al. 2018 threshold
            quality['ruwe_quality'] = True
        else:
            quality['quality_flags'].append(f'high_ruwe_{ruwe:.2f}')
    else:
        quality['quality_flags'].append('missing_ruwe')
    
    # Parallax quality assessment
    parallax = star.get('parallax')
    parallax_error = star.get('parallax_error')
    if parallax is not None and parallax_error is not None:
        if parallax_error > 0:
            parallax_significance = abs(parallax) / parallax_error
            if parallax_significance >= 3.0:
                quality['parallax_quality'] = True
            else:
                quality['quality_flags'].append(f'low_parallax_significance_{parallax_significance:.1f}')
        else:
            quality['quality_flags'].append('zero_parallax_error')
    else:
        quality['quality_flags'].append('missing_parallax_data')
    
    # Proper motion quality assessment
    pmra = star.get('pmra')
    pmra_error = star.get('pmra_error')
    pmdec = star.get('pmdec')
    pmdec_error = star.get('pmdec_error')
    
    if all(x is not None for x in [pmra, pmra_error, pmdec, pmdec_error]):
        if pmra_error > 0 and pmdec_error > 0:
            pm_total = np.sqrt(pmra**2 + pmdec**2)
            pm_error_total = np.sqrt(pmra_error**2 + pmdec_error**2)
            pm_significance = pm_total / pm_error_total
            if pm_significance >= 3.0:
                quality['pm_quality'] = True
            else:
                quality['quality_flags'].append(f'low_pm_significance_{pm_significance:.1f}')
        else:
            quality['quality_flags'].append('zero_pm_errors')
    else:
        quality['quality_flags'].append('missing_pm_data')
    
    # Overall quality assessment
    quality_count = sum([quality['ruwe_quality'], quality['parallax_quality'], quality['pm_quality']])
    
    if quality_count == 3:
        quality['overall_quality'] = 'excellent'
    elif quality_count == 2:
        quality['overall_quality'] = 'good'
    elif quality_count == 1:
        quality['overall_quality'] = 'fair'
    else:
        quality['overall_quality'] = 'poor'
    
    return quality


def validate_gaia_source_completeness(star: Dict[str, Any]) -> Dict[str, bool]:
    """
    Check completeness of essential Gaia astrometric data.
    
    Args:
        star: Gaia source data dictionary
        
    Returns:
        Dict indicating which data types are available:
        - has_position: RA/Dec available
        - has_parallax: Parallax measurement available
        - has_proper_motion: Proper motion available
        - has_errors: Error estimates available
        - has_correlations: Correlation coefficients available
    """
    completeness = {
        'has_position': False,
        'has_parallax': False,
        'has_proper_motion': False,
        'has_errors': False,
        'has_correlations': False
    }
    
    # Position check
    if all(key in star and star[key] is not None for key in ['ra', 'dec']):
        completeness['has_position'] = True
    
    # Parallax check
    if 'parallax' in star and star['parallax'] is not None:
        completeness['has_parallax'] = True
    
    # Proper motion check
    if all(key in star and star[key] is not None for key in ['pmra', 'pmdec']):
        completeness['has_proper_motion'] = True
    
    # Error check
    error_keys = ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']
    if any(key in star and star[key] is not None for key in error_keys):
        completeness['has_errors'] = True
    
    # Correlation check
    corr_keys = ['ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
                'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
                'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr']
    if any(key in star and star[key] is not None for key in corr_keys):
        completeness['has_correlations'] = True
    
    return completeness


def build_covariance_matrix(star: Dict[str, Any], dimensions: int = 3) -> Optional[np.ndarray]:
    """
    Build covariance matrix for astrometric parameters with RUWE correction.
    
    This function constructs the full covariance matrix including correlations
    and applies RUWE-based error correction to account for systematic uncertainties.
    
    Args:
        star: Gaia source data dictionary
        dimensions: 1 (parallax), 2 (proper motion), or 3 (parallax + proper motion)
        
    Returns:
        Covariance matrix or None if data is insufficient
        
    Raises:
        ValueError: If dimensions is not 1, 2, or 3
    """
    from ..config import GAIA_DEFAULT_CORRELATION_MISSING
    
    def get_correlation_safe(star: Dict, key: str) -> float:
        """Safe correlation retrieval with fallback default."""
        corr = star.get(key)
        if corr is None:
            log.debug(f"Missing correlation {key}, using default {GAIA_DEFAULT_CORRELATION_MISSING}")
            return GAIA_DEFAULT_CORRELATION_MISSING
        return float(corr)
    
    try:
        # Get RUWE for error correction (default to 1.0 if missing)
        ruwe = star.get('ruwe', 1.0)
        if ruwe is None or not np.isfinite(ruwe):
            ruwe = 1.0
            
        if dimensions == 1:
            # 1D: parallax only
            err = star.get('parallax_error')
            if err is None or err <= 0:
                return None
                
            # Apply RUWE correction
            corrected_err = get_gaia_parallax_error_safe(err, ruwe)
            return np.array([[corrected_err**2]])
            
        elif dimensions == 2:
            # 2D: proper motion only
            pmra_err = star.get('pmra_error')
            pmdec_err = star.get('pmdec_error')
            
            if pmra_err is None or pmdec_err is None or pmra_err <= 0 or pmdec_err <= 0:
                return None
            
            # Apply RUWE correction
            corrected_pmra_err = get_gaia_pmra_error_safe(pmra_err, ruwe)
            corrected_pmdec_err = get_gaia_pmdec_error_safe(pmdec_err, ruwe)
                
            # Get correlation
            pmra_pmdec_corr = get_correlation_safe(star, 'pmra_pmdec_corr')
            
            C = np.zeros((2, 2))
            C[0, 0] = corrected_pmra_err**2
            C[1, 1] = corrected_pmdec_err**2
            C[0, 1] = C[1, 0] = corrected_pmra_err * corrected_pmdec_err * pmra_pmdec_corr
            return C
            
        elif dimensions == 3:
            # 3D: parallax + proper motion
            plx_err = star.get('parallax_error')
            pmra_err = star.get('pmra_error') 
            pmdec_err = star.get('pmdec_error')
            
            if (plx_err is None or pmra_err is None or pmdec_err is None or
                plx_err <= 0 or pmra_err <= 0 or pmdec_err <= 0):
                return None
            
            # Apply RUWE correction to all astrometric errors
            corrected_plx_err = correct_gaia_error(plx_err, ruwe, 'parallax')
            corrected_pmra_err = correct_gaia_error(pmra_err, ruwe, 'pmra')
            corrected_pmdec_err = correct_gaia_error(pmdec_err, ruwe, 'pmdec')
            
            # Get correlations
            plx_pmra_corr = get_correlation_safe(star, 'parallax_pmra_corr')
            plx_pmdec_corr = get_correlation_safe(star, 'parallax_pmdec_corr')
            pmra_pmdec_corr = get_correlation_safe(star, 'pmra_pmdec_corr')
            
            C = np.zeros((3, 3))
            C[0, 0] = corrected_plx_err**2
            C[1, 1] = corrected_pmra_err**2
            C[2, 2] = corrected_pmdec_err**2
            C[0, 1] = C[1, 0] = corrected_plx_err * corrected_pmra_err * plx_pmra_corr
            C[0, 2] = C[2, 0] = corrected_plx_err * corrected_pmdec_err * plx_pmdec_corr
            C[1, 2] = C[2, 1] = corrected_pmra_err * corrected_pmdec_err * pmra_pmdec_corr
            return C
            
        else:
            raise ValueError(f"Unsupported dimensions: {dimensions}")
            
    except Exception as e:
        log.debug(f"Error building {dimensions}D covariance matrix: {e}")
        return None
