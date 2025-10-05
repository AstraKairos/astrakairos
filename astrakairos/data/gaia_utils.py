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
from typing import Optional, Union, Dict, Any, Tuple

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
    """Get safe parallax error with fallback. No additional correction applied."""
    return get_safe_error_with_fallback(
        parallax_error, 
        FALLBACK_PARALLAX_ERROR_MAS, 
        None,
        "parallax"
    )


def get_gaia_pmra_error_safe(pmra_error: Optional[float], ruwe: Optional[float] = None) -> float:
    """Get safe proper motion RA error with fallback. No additional correction applied."""
    return get_safe_error_with_fallback(
        pmra_error, 
        FALLBACK_PMRA_ERROR_MAS_PER_YEAR, 
        None,
        "pmra"
    )


def get_gaia_pmdec_error_safe(pmdec_error: Optional[float], ruwe: Optional[float] = None) -> float:
    """Get safe proper motion Dec error with fallback. No additional correction applied."""
    return get_safe_error_with_fallback(
        pmdec_error, 
        FALLBACK_PMDEC_ERROR_MAS_PER_YEAR, 
        None,
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
    Build covariance matrix for astrometric parameters.
    
    This function constructs the full covariance matrix including correlations.
    If errors have already been inflated (via inflate_gaia_uncertainties),
    no additional correction is applied.
    
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
        errors_already_inflated = star.get('_errors_inflated', False)
            
        if dimensions == 1:
            err = star.get('parallax_error')
            if err is None or err <= 0:
                return None
                
            if not errors_already_inflated:
                err = get_gaia_parallax_error_safe(err, star.get('ruwe'))
                
            return np.array([[err**2]])
            
        elif dimensions == 2:
            pmra_err = star.get('pmra_error')
            pmdec_err = star.get('pmdec_error')
            
            if pmra_err is None or pmdec_err is None or pmra_err <= 0 or pmdec_err <= 0:
                return None
            
            if not errors_already_inflated:
                pmra_err = get_gaia_pmra_error_safe(pmra_err, star.get('ruwe'))
                pmdec_err = get_gaia_pmdec_error_safe(pmdec_err, star.get('ruwe'))
                
            pmra_pmdec_corr = get_correlation_safe(star, 'pmra_pmdec_corr')
            
            C = np.zeros((2, 2))
            C[0, 0] = pmra_err**2
            C[1, 1] = pmdec_err**2
            C[0, 1] = C[1, 0] = pmra_err * pmdec_err * pmra_pmdec_corr
            return C
            
        elif dimensions == 3:
            plx_err = star.get('parallax_error')
            pmra_err = star.get('pmra_error') 
            pmdec_err = star.get('pmdec_error')
            
            if (plx_err is None or pmra_err is None or pmdec_err is None or
                plx_err <= 0 or pmra_err <= 0 or pmdec_err <= 0):
                return None
            
            if not errors_already_inflated:
                ruwe = star.get('ruwe')
                plx_err = correct_gaia_error(plx_err, ruwe, 'parallax')
                pmra_err = correct_gaia_error(pmra_err, ruwe, 'pmra')
                pmdec_err = correct_gaia_error(pmdec_err, ruwe, 'pmdec')
            
            plx_pmra_corr = get_correlation_safe(star, 'parallax_pmra_corr')
            plx_pmdec_corr = get_correlation_safe(star, 'parallax_pmdec_corr')
            pmra_pmdec_corr = get_correlation_safe(star, 'pmra_pmdec_corr')
            
            C = np.zeros((3, 3))
            C[0, 0] = plx_err**2
            C[1, 1] = pmra_err**2
            C[2, 2] = pmdec_err**2
            C[0, 1] = C[1, 0] = plx_err * pmra_err * plx_pmra_corr
            C[0, 2] = C[2, 0] = plx_err * pmdec_err * plx_pmdec_corr
            C[1, 2] = C[2, 1] = pmra_err * pmdec_err * pmra_pmdec_corr
            return C
            
        else:
            raise ValueError(f"Unsupported dimensions: {dimensions}")
            
    except Exception as e:
        log.debug(f"Error building {dimensions}D covariance matrix: {e}")
        return None


def apply_magnitude_dependent_error_inflation(g_mag: float) -> float:
    """
    Calculate systematic parallax error inflation factor based on G magnitude.
    
    Implements El-Badry et al. (2021) Equation 16, which corrects for
    magnitude-dependent underestimation of Gaia parallax uncertainties.
    At G ≈ 13, σ_π is underestimated by ~30%. The correction is largest
    for bright (G < 13) and faint (G > 17) sources.
    
    Args:
        g_mag: Gaia G-band magnitude
        
    Returns:
        Multiplicative inflation factor (≥ 1.0)
        
    Reference:
        El-Badry et al. (2021), MNRAS 506, 2269-2295
        https://doi.org/10.1093/mnras/stab323
    """
    from ..config import GAIA_PARALLAX_INFLATION_PARAMS
    
    if g_mag < 7 or g_mag > 21:
        return 1.0
    
    A = GAIA_PARALLAX_INFLATION_PARAMS['A']
    G0 = GAIA_PARALLAX_INFLATION_PARAMS['G0']
    b = GAIA_PARALLAX_INFLATION_PARAMS['b']
    p0 = GAIA_PARALLAX_INFLATION_PARAMS['p0']
    p1 = GAIA_PARALLAX_INFLATION_PARAMS['p1']
    p2 = GAIA_PARALLAX_INFLATION_PARAMS['p2']
    
    gaussian_bump = A * np.exp(-((g_mag - G0)**2) / (b**2))
    polynomial = p0 + p1*g_mag + p2*(g_mag**2)
    
    return gaussian_bump + polynomial


def apply_separation_dependent_error_inflation(separation_arcsec: float) -> float:
    """
    Calculate additional error inflation for close angular separations.
    
    For sources with companions at θ < 4 arcsec, Gaia astrometric
    uncertainties are underestimated by ~40-80% due to blending effects.
    This function implements an empirical correction based on El-Badry
    et al. (2021) Fig. 16-17.
    
    Args:
        separation_arcsec: Angular separation in arcseconds
        
    Returns:
        Additional multiplicative inflation factor (≥ 1.0)
        
    Reference:
        El-Badry et al. (2021), MNRAS 506, 2269-2295, Fig. 16-17
    """
    from ..config import GAIA_ADAPTIVE_THRESHOLDS
    
    threshold = GAIA_ADAPTIVE_THRESHOLDS['close_separation']['angle_threshold_arcsec']
    
    if separation_arcsec >= threshold:
        return 1.0
    
    max_inflation = 1.6
    inflation = 1.0 + (max_inflation - 1.0) * (1.0 - separation_arcsec / threshold)
    
    return inflation


def apply_ruwe_dependent_error_inflation(ruwe: float) -> float:
    """
    Calculate additional error inflation for high RUWE values.
    
    RUWE primarily serves as a quality flag in El-Badry et al. (2021).
    Sources with RUWE > 1.4 have potentially problematic astrometric solutions
    and show larger error underestimates (Fig. 16), but the paper provides
    no explicit functional form for RUWE-dependent inflation.
    
    From Fig. 16, sources with RUWE > 1.4 show approximately 1.3-1.5x larger
    inflation factors compared to RUWE < 1.4 at similar magnitudes/separations.
    We use a conservative step function based on these empirical values.
    
    Args:
        ruwe: Renormalized Unit Weight Error
        
    Returns:
        Additional multiplicative inflation factor (≥ 1.0)
        
    Reference:
        El-Badry et al. (2021), MNRAS 506, 2269-2295, Fig. 16, Section 5.3
    """
    from ..config import GAIA_ADAPTIVE_THRESHOLDS
    
    threshold = GAIA_ADAPTIVE_THRESHOLDS['ruwe_threshold']
    
    if ruwe <= threshold:
        # Good astrometry: no additional inflation
        return 1.0
    elif ruwe <= 2.0:
        # Moderately problematic: ~1.3x additional factor (Fig. 16)
        return 1.3
    else:
        # Severely problematic: ~1.5x additional factor
        return 1.5


def inflate_gaia_uncertainties(gaia_data: Dict[str, Any],
                               separation_arcsec: Optional[float] = None) -> Dict[str, Any]:
    """
    Apply all systematic error corrections to Gaia astrometric data.
    
    Combines three independent sources of error underestimation:
    1. Magnitude-dependent (El-Badry Eq. 16)
    2. Separation-dependent (for θ < 4 arcsec)
    3. RUWE-dependent (for poor astrometric solutions)
    
    Args:
        gaia_data: Dictionary containing Gaia astrometric measurements
        separation_arcsec: Angular separation to companion (if known)
        
    Returns:
        Copy of gaia_data with inflated error values and correction flag
        
    Reference:
        El-Badry et al. (2021), MNRAS 506, 2269-2295
    """
    if gaia_data.get('_errors_inflated'):
        return gaia_data
    
    inflated = gaia_data.copy()
    
    g_mag = gaia_data.get('phot_g_mean_mag')
    ruwe = gaia_data.get('ruwe')
    
    parallax_inflation = 1.0
    pm_inflation = 1.0
    
    if g_mag is not None and np.isfinite(g_mag):
        mag_factor = apply_magnitude_dependent_error_inflation(g_mag)
        parallax_inflation *= mag_factor
        pm_inflation *= np.sqrt(mag_factor)
    
    if separation_arcsec is not None and separation_arcsec < 4.0:
        sep_factor = apply_separation_dependent_error_inflation(separation_arcsec)
        parallax_inflation *= sep_factor
        pm_inflation *= np.sqrt(sep_factor)
    
    if ruwe is not None and ruwe > 1.4:
        ruwe_factor = apply_ruwe_dependent_error_inflation(ruwe)
        parallax_inflation *= ruwe_factor
        pm_inflation *= ruwe_factor
    
    if 'parallax_error' in inflated and inflated['parallax_error'] is not None:
        inflated['parallax_error'] *= parallax_inflation
    
    if 'pmra_error' in inflated and inflated['pmra_error'] is not None:
        inflated['pmra_error'] *= pm_inflation
    
    if 'pmdec_error' in inflated and inflated['pmdec_error'] is not None:
        inflated['pmdec_error'] *= pm_inflation
    
    inflated['_errors_inflated'] = True
    inflated['_parallax_inflation_factor'] = parallax_inflation
    inflated['_pm_inflation_factor'] = pm_inflation
    
    return inflated


def get_adaptive_delta_mu_thresholds(separation_arcsec: float,
                                     ruwe_primary: float,
                                     ruwe_secondary: float) -> Tuple[float, float]:
    """
    Calculate adaptive classification thresholds for delta_mu_orbit significance.
    
    For close separations (θ < 4 arcsec) and poor astrometry (RUWE > 1.4),
    the standard thresholds (2.5σ physical, 5.0σ optical) are too strict
    due to systematic error underestimation. This function adjusts thresholds
    to maintain consistent classification rates.
    
    Args:
        separation_arcsec: Angular separation in arcseconds
        ruwe_primary: RUWE of primary component
        ruwe_secondary: RUWE of secondary component
        
    Returns:
        Tuple of (physical_threshold, ambiguous_threshold) in units of σ
        
    Reference:
        El-Badry et al. (2021), MNRAS 506, 2269-2295, Fig. 15-17
    """
    from ..config import GAIA_ADAPTIVE_THRESHOLDS
    
    close_config = GAIA_ADAPTIVE_THRESHOLDS['close_separation']
    wide_config = GAIA_ADAPTIVE_THRESHOLDS['wide_separation']
    
    if separation_arcsec < close_config['angle_threshold_arcsec']:
        physical_threshold = close_config['delta_mu_physical']
        ambiguous_threshold = close_config['delta_mu_ambiguous']
    else:
        physical_threshold = wide_config['delta_mu_physical']
        ambiguous_threshold = wide_config['delta_mu_ambiguous']
    
    # Note: We don't adjust thresholds for RUWE here because error inflation
    # already accounts for RUWE effects via apply_ruwe_dependent_error_inflation()
    # The 5σ and 8σ thresholds are designed to work with inflated errors.
    
    return physical_threshold, ambiguous_threshold


def check_astrometric_quality_flags(gaia_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate Image Parameter Determination (IPD) quality flags.
    
    IPD flags detect astrometric problems not captured by RUWE alone,
    particularly for unresolved binaries or sources with companions at
    θ < 2 arcsec. These flags detect ~40% of problems missed by RUWE.
    
    Args:
        gaia_data: Dictionary containing Gaia astrometric data
        
    Returns:
        Dictionary with quality assessment:
        - 'quality': 'excellent', 'good', 'marginal', or 'poor'
        - 'flags': Individual flag status
        - 'n_bad_flags': Count of problematic flags
        
    Reference:
        El-Badry et al. (2021), MNRAS 506, 2269-2295, Fig. 18
        Lindegren et al. (2021), A&A 649, A2
    """
    from ..config import GAIA_QUALITY_FLAGS
    
    ruwe = gaia_data.get('ruwe', 1.0)
    ipd_harmonic = gaia_data.get('ipd_gof_harmonic_amplitude', 0.0)
    ipd_multi_peak = gaia_data.get('ipd_frac_multi_peak', 0.0)
    
    flags = {
        'ruwe_excellent': ruwe < GAIA_QUALITY_FLAGS['ruwe_excellent'],
        'ruwe_good': ruwe < GAIA_QUALITY_FLAGS['ruwe_good'],
        'ruwe_acceptable': ruwe < GAIA_QUALITY_FLAGS['ruwe_marginal'],
        'ipd_harmonic_good': ipd_harmonic < GAIA_QUALITY_FLAGS['ipd_gof_harmonic_amplitude_threshold'],
        'ipd_multi_peak_good': ipd_multi_peak < GAIA_QUALITY_FLAGS['ipd_frac_multi_peak_threshold']
    }
    
    n_bad_flags = sum([
        not flags['ruwe_acceptable'],
        not flags['ipd_harmonic_good'],
        not flags['ipd_multi_peak_good']
    ])
    
    if n_bad_flags == 0 and flags['ruwe_excellent']:
        quality = 'excellent'
    elif n_bad_flags == 0:
        quality = 'good'
    elif n_bad_flags == 1:
        quality = 'marginal'
    else:
        quality = 'poor'
    
    return {
        'quality': quality,
        'flags': flags,
        'n_bad_flags': n_bad_flags
    }


def check_tangential_velocity_consistency(gaia_primary: Dict[str, Any],
                                         gaia_secondary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if tangential velocities are consistent between components.
    
    Physically bound systems should have similar space velocities. Large
    differences in v_tan suggest an optical alignment. This is used as
    a tie-breaker for ambiguous cases, not as a primary discriminator.
    
    Args:
        gaia_primary: Primary component Gaia data
        gaia_secondary: Secondary component Gaia data
        
    Returns:
        Dictionary with consistency assessment:
        - 'consistent': Boolean
        - 'v_tan_primary': Tangential velocity of primary (km/s)
        - 'v_tan_secondary': Tangential velocity of secondary (km/s)
        - 'fractional_difference': |v1 - v2| / mean(v1, v2)
        
    Reference:
        El-Badry et al. (2021), MNRAS 506, 2269-2295, Table A1
    """
    from ..config import GAIA_VELOCITY_CONSISTENCY
    
    conversion_factor = GAIA_VELOCITY_CONSISTENCY['velocity_conversion_factor']
    max_diff_fraction = GAIA_VELOCITY_CONSISTENCY['tangential_velocity_max_diff_fraction']
    
    pm_total_p = gaia_primary.get('pm_total')
    pm_total_s = gaia_secondary.get('pm_total')
    parallax_p = gaia_primary.get('parallax')
    parallax_s = gaia_secondary.get('parallax')
    
    if None in [pm_total_p, pm_total_s, parallax_p, parallax_s]:
        return {'consistent': None, 'error': 'missing_data'}
    
    if parallax_p <= 0 or parallax_s <= 0:
        return {'consistent': None, 'error': 'invalid_parallax'}
    
    v_tan_p = conversion_factor * pm_total_p / parallax_p
    v_tan_s = conversion_factor * pm_total_s / parallax_s
    
    v_diff = abs(v_tan_p - v_tan_s)
    v_mean = (v_tan_p + v_tan_s) / 2.0
    
    if v_mean <= 0:
        return {'consistent': None, 'error': 'zero_mean_velocity'}
    
    fractional_diff = v_diff / v_mean
    consistent = fractional_diff <= max_diff_fraction
    
    return {
        'consistent': consistent,
        'v_tan_primary': v_tan_p,
        'v_tan_secondary': v_tan_s,
        'fractional_difference': fractional_diff
    }

