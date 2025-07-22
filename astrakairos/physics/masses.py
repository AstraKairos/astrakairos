"""
Mass calculation for binary star systems using Kepler's Third Law.

This module implements mass calculations with Monte Carlo error propagation,
following the established patterns in dynamics.py for robust statistical analysis.

The implementation uses Kepler's Third Law: M_total = a³ / P² (in solar masses, AU, years)
where a_AU = a_arcsec × d_pc, and includes comprehensive uncertainty propagation
through Monte Carlo sampling.

Functions:
    calculate_total_mass_kepler3: Main mass calculation with error propagation
    calculate_individual_masses: Estimate individual masses if mass ratio available
    _validate_mass_inputs: Input validation with physical constraints
    _calculate_mc_statistics: Statistics from Monte Carlo samples

Dependencies:
    numpy: Vectorized numerical operations
    logging: Convergence information and warnings
    dataclasses: Result structure
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass

# Centralized configuration imports for consistency
from ..config import (
    # Monte Carlo configuration
    DEFAULT_MC_SAMPLES, MC_CONFIDENCE_LEVEL, MC_RANDOM_SEED,
    # Physical constants and validation ranges
    MIN_PERIOD_YEARS, MAX_PERIOD_YEARS, MIN_SEMIMAJOR_AXIS_ARCSEC,
    MIN_PARALLAX_MAS, MAX_PARALLAX_MAS, MIN_DISTANCE_PC, MAX_DISTANCE_PC,
    MIN_STELLAR_MASS_SOLAR, MAX_STELLAR_MASS_SOLAR, 
    MIN_TOTAL_MASS_SOLAR, MAX_TOTAL_MASS_SOLAR,
    MIN_MASS_RATIO, MAX_MASS_RATIO,
    # Warning thresholds
    LARGE_MASS_ERROR_THRESHOLD, EXTREME_MASS_WARNING_SOLAR, 
    LOW_PARALLAX_WARNING_MAS,
    # Parallax source priority
    PARALLAX_SOURCE_PRIORITY,
    # Monte Carlo limits
    MASS_MC_MIN_SAMPLES, MASS_MC_MAX_SAMPLES, MASS_MC_CONVERGENCE_THRESHOLD
)

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MassResult:
    """Results from mass calculation with uncertainties."""
    total_mass_solar: float
    total_mass_error: float
    individual_masses_solar: Optional[Tuple[float, float]]
    individual_mass_errors: Optional[Tuple[float, float]]
    mass_ratio: Optional[float]
    mass_ratio_error: Optional[float]
    parallax_used_mas: float
    distance_used_pc: float
    parallax_source: str
    mc_samples: int
    quality_score: float
    warnings: List[str]

def _validate_mass_inputs(
    period_years: float,
    semimajor_axis_arcsec: float,
    parallax_mas: float,
    period_error: float = 0.0,
    semimajor_axis_error: float = 0.0,
    parallax_error: float = 0.0
) -> Tuple[bool, List[str]]:
    """
    Validate inputs for mass calculation.
    
    Args:
        period_years: Orbital period in years
        semimajor_axis_arcsec: Semi-major axis in arcseconds
        parallax_mas: Parallax in milliarcseconds
        *_error: Uncertainties (1-sigma)
        
    Returns:
        Tuple of (is_valid, warning_list)
    """
    warnings = []
    is_valid = True
    
    # Basic range validation
    if not (MIN_PERIOD_YEARS <= period_years <= MAX_PERIOD_YEARS):
        logger.error(f"Period {period_years:.3f} years outside valid range [{MIN_PERIOD_YEARS}, {MAX_PERIOD_YEARS}]")
        is_valid = False
        
    if not (MIN_SEMIMAJOR_AXIS_ARCSEC <= semimajor_axis_arcsec):
        logger.error(f"Semi-major axis {semimajor_axis_arcsec:.3f} arcsec below minimum {MIN_SEMIMAJOR_AXIS_ARCSEC}")
        is_valid = False
        
    if not (MIN_PARALLAX_MAS <= parallax_mas <= MAX_PARALLAX_MAS):
        logger.error(f"Parallax {parallax_mas:.3f} mas outside valid range [{MIN_PARALLAX_MAS}, {MAX_PARALLAX_MAS}]")
        is_valid = False
    
    # Distance calculation and validation
    distance_pc = 1000.0 / parallax_mas  # Convert mas to pc
    if not (MIN_DISTANCE_PC <= distance_pc <= MAX_DISTANCE_PC):
        logger.error(f"Distance {distance_pc:.1f} pc outside valid range [{MIN_DISTANCE_PC}, {MAX_DISTANCE_PC}]")
        is_valid = False
    
    # Warning conditions
    if parallax_mas < LOW_PARALLAX_WARNING_MAS:
        warnings.append(f"Low parallax precision ({parallax_mas:.2f} mas) may affect mass accuracy")
    
    if parallax_error > 0 and parallax_error / parallax_mas > 0.2:
        warnings.append(f"Large parallax uncertainty ({parallax_error/parallax_mas*100:.1f}%)")
    
    # Error validation
    if period_error < 0 or semimajor_axis_error < 0 or parallax_error < 0:
        logger.error("Negative uncertainties are not allowed")
        is_valid = False
    
    return is_valid, warnings

def _calculate_mc_statistics(samples: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics from Monte Carlo samples.
    
    Args:
        samples: Array of samples from Monte Carlo simulation
        
    Returns:
        Dictionary with median, uncertainty, and percentiles
    """
    if len(samples) == 0:
        return {'median': 0.0, 'uncertainty': 0.0, 'p_lower': 0.0, 'p_upper': 0.0}
    
    # Remove any invalid samples (NaN, infinite, or negative masses)
    valid_samples = samples[np.isfinite(samples) & (samples > 0)]
    
    if len(valid_samples) == 0:
        logger.warning("No valid samples in Monte Carlo simulation")
        return {'median': 0.0, 'uncertainty': np.inf, 'p_lower': 0.0, 'p_upper': 0.0}
    
    # Calculate percentiles for confidence interval
    lower_percentile = (100.0 - MC_CONFIDENCE_LEVEL) / 2.0
    upper_percentile = 100.0 - lower_percentile
    
    median = np.median(valid_samples)
    p_lower = np.percentile(valid_samples, lower_percentile)
    p_upper = np.percentile(valid_samples, upper_percentile)
    
    # Uncertainty is half the confidence interval width
    uncertainty = (p_upper - p_lower) / 2.0
    
    return {
        'median': median,
        'uncertainty': uncertainty,
        'p_lower': p_lower,
        'p_upper': p_upper
    }

def calculate_total_mass_kepler3(
    period_years: float,
    semimajor_axis_arcsec: float,
    parallax_mas: float,
    period_error: float = 0.0,
    semimajor_axis_error: float = 0.0,
    parallax_error: float = 0.0,
    parallax_source: str = 'unknown',
    mc_samples: int = DEFAULT_MC_SAMPLES
) -> Optional[MassResult]:
    """
    Calculate total system mass using Kepler's Third Law.
    
    M_total = a³ / P² (in solar masses, AU, years)
    where a_AU = a_arcsec × d_pc
    
    Parameters
    ----------
    period_years : float
        Orbital period in years
    semimajor_axis_arcsec : float
        Semi-major axis in arcseconds
    parallax_mas : float
        Parallax in milliarcseconds
    period_error : float, optional
        1-sigma uncertainty in period (years)
    semimajor_axis_error : float, optional
        1-sigma uncertainty in semi-major axis (arcseconds)
    parallax_error : float, optional
        1-sigma uncertainty in parallax (mas)
    parallax_source : str, optional
        Source of parallax data for quality assessment
    mc_samples : int, optional
        Number of Monte Carlo samples for error propagation
        
    Returns
    -------
    MassResult or None
        Complete mass analysis with uncertainties, or None if calculation fails
        
    Notes
    -----
    Uses Kepler's Third Law in the form:
    M_total [M☉] = (a [AU])³ / (P [years])²
    
    Distance conversion: d [pc] = 1000 / π [mas]
    Semi-major axis: a [AU] = a [arcsec] × d [pc]
    """
    
    # Validate inputs
    is_valid, warnings = _validate_mass_inputs(
        period_years, semimajor_axis_arcsec, parallax_mas,
        period_error, semimajor_axis_error, parallax_error
    )
    
    if not is_valid:
        logger.error("Mass calculation failed due to invalid inputs")
        return None
    
    # Constrain Monte Carlo samples to reasonable range
    mc_samples = max(MASS_MC_MIN_SAMPLES, min(mc_samples, MASS_MC_MAX_SAMPLES))
    
    # Calculate point estimate first
    distance_pc = 1000.0 / parallax_mas
    semimajor_axis_au = semimajor_axis_arcsec * distance_pc
    total_mass_solar = (semimajor_axis_au ** 3) / (period_years ** 2)
    
    # Check for extreme mass values
    if total_mass_solar > EXTREME_MASS_WARNING_SOLAR:
        warnings.append(f"Very high total mass ({total_mass_solar:.1f} M☉) - check inputs")
    
    if not (MIN_TOTAL_MASS_SOLAR <= total_mass_solar <= MAX_TOTAL_MASS_SOLAR):
        warnings.append(f"Total mass ({total_mass_solar:.2f} M☉) outside typical range")
    
    # Monte Carlo error propagation if uncertainties are provided
    total_mass_error = 0.0
    quality_score = 1.0  # Start with perfect quality
    
    if any(err > 0 for err in [period_error, semimajor_axis_error, parallax_error]):
        logger.debug(f"Running Monte Carlo with {mc_samples} samples for mass uncertainty")
        
        # Set random seed for reproducibility
        np.random.seed(MC_RANDOM_SEED)
        
        # Generate random samples
        period_samples = np.random.normal(period_years, period_error, mc_samples) if period_error > 0 else np.full(mc_samples, period_years)
        semimajor_samples = np.random.normal(semimajor_axis_arcsec, semimajor_axis_error, mc_samples) if semimajor_axis_error > 0 else np.full(mc_samples, semimajor_axis_arcsec)
        parallax_samples = np.random.normal(parallax_mas, parallax_error, mc_samples) if parallax_error > 0 else np.full(mc_samples, parallax_mas)
        
        # Ensure physical constraints on samples
        period_samples = np.clip(period_samples, MIN_PERIOD_YEARS, MAX_PERIOD_YEARS)
        semimajor_samples = np.clip(semimajor_samples, MIN_SEMIMAJOR_AXIS_ARCSEC, np.inf)
        parallax_samples = np.clip(parallax_samples, MIN_PARALLAX_MAS, MAX_PARALLAX_MAS)
        
        # Calculate mass for each sample
        distance_samples = 1000.0 / parallax_samples
        semimajor_au_samples = semimajor_samples * distance_samples
        mass_samples = (semimajor_au_samples ** 3) / (period_samples ** 2)
        
        # Calculate statistics
        mc_stats = _calculate_mc_statistics(mass_samples)
        total_mass_solar = mc_stats['median']  # Use median as more robust estimate
        total_mass_error = mc_stats['uncertainty']
        
        # Quality assessment based on uncertainty
        if total_mass_error > 0:
            relative_error = total_mass_error / total_mass_solar
            if relative_error > LARGE_MASS_ERROR_THRESHOLD:
                warnings.append(f"Large mass uncertainty ({relative_error*100:.1f}%)")
                quality_score *= (1.0 - min(relative_error, 0.9))  # Reduce quality for large errors
    
    # Assess quality based on parallax source
    source_quality = {
        'gaia_dr3': 1.0,
        'gaia': 0.9,
        'hipparcos': 0.7,
        'literature': 0.5,
        'estimated': 0.3,
        'unknown': 0.1
    }
    quality_score *= source_quality.get(parallax_source.lower(), 0.1)
    
    return MassResult(
        total_mass_solar=total_mass_solar,
        total_mass_error=total_mass_error,
        individual_masses_solar=None,  # Will be calculated if mass ratio is available
        individual_mass_errors=None,
        mass_ratio=None,
        mass_ratio_error=None,
        parallax_used_mas=parallax_mas,
        distance_used_pc=distance_pc,
        parallax_source=parallax_source,
        mc_samples=mc_samples,
        quality_score=quality_score,
        warnings=warnings
    )

def calculate_individual_masses(
    mass_result: MassResult,
    mass_ratio: float,
    mass_ratio_error: float = 0.0,
    mc_samples: int = DEFAULT_MC_SAMPLES
) -> MassResult:
    """
    Calculate individual component masses given total mass and mass ratio.
    
    Uses the definitions:
    - q = M2 / M1 (mass ratio, secondary/primary)
    - M1 + M2 = M_total
    - M1 = M_total / (1 + q)
    - M2 = M_total * q / (1 + q)
    
    Parameters
    ----------
    mass_result : MassResult
        Result from calculate_total_mass_kepler3
    mass_ratio : float
        Mass ratio q = M2/M1 (secondary/primary)
    mass_ratio_error : float, optional
        1-sigma uncertainty in mass ratio
    mc_samples : int, optional
        Number of Monte Carlo samples for error propagation
        
    Returns
    -------
    MassResult
        Updated mass result with individual masses
    """
    
    # Validate mass ratio
    if not (MIN_MASS_RATIO <= mass_ratio <= MAX_MASS_RATIO):
        logger.error(f"Mass ratio {mass_ratio:.3f} outside valid range [{MIN_MASS_RATIO}, {MAX_MASS_RATIO}]")
        return mass_result
    
    # Calculate point estimates
    m1_solar = mass_result.total_mass_solar / (1.0 + mass_ratio)
    m2_solar = mass_result.total_mass_solar * mass_ratio / (1.0 + mass_ratio)
    
    # Validate individual masses
    for mass, label in [(m1_solar, "Primary"), (m2_solar, "Secondary")]:
        if not (MIN_STELLAR_MASS_SOLAR <= mass <= MAX_STELLAR_MASS_SOLAR):
            mass_result.warnings.append(f"{label} mass ({mass:.2f} M☉) outside typical stellar range")
    
    # Monte Carlo error propagation if uncertainties available
    m1_error, m2_error = 0.0, 0.0
    
    if mass_result.total_mass_error > 0 or mass_ratio_error > 0:
        logger.debug(f"Running Monte Carlo for individual mass uncertainties")
        
        np.random.seed(MC_RANDOM_SEED)
        
        # Generate samples
        total_mass_samples = np.random.normal(
            mass_result.total_mass_solar, 
            mass_result.total_mass_error, 
            mc_samples
        ) if mass_result.total_mass_error > 0 else np.full(mc_samples, mass_result.total_mass_solar)
        
        mass_ratio_samples = np.random.normal(
            mass_ratio, 
            mass_ratio_error, 
            mc_samples
        ) if mass_ratio_error > 0 else np.full(mc_samples, mass_ratio)
        
        # Ensure physical constraints
        total_mass_samples = np.clip(total_mass_samples, MIN_TOTAL_MASS_SOLAR, MAX_TOTAL_MASS_SOLAR)
        mass_ratio_samples = np.clip(mass_ratio_samples, MIN_MASS_RATIO, MAX_MASS_RATIO)
        
        # Calculate individual mass samples
        m1_samples = total_mass_samples / (1.0 + mass_ratio_samples)
        m2_samples = total_mass_samples * mass_ratio_samples / (1.0 + mass_ratio_samples)
        
        # Calculate uncertainties
        m1_stats = _calculate_mc_statistics(m1_samples)
        m2_stats = _calculate_mc_statistics(m2_samples)
        
        m1_solar = m1_stats['median']
        m2_solar = m2_stats['median']
        m1_error = m1_stats['uncertainty']
        m2_error = m2_stats['uncertainty']
    
    # Update the mass result
    mass_result.individual_masses_solar = (m1_solar, m2_solar)
    mass_result.individual_mass_errors = (m1_error, m2_error)
    mass_result.mass_ratio = mass_ratio
    mass_result.mass_ratio_error = mass_ratio_error
    
    return mass_result

def format_mass_results(mass_result: MassResult, include_warnings: bool = True) -> str:
    """
    Format mass calculation results for display.
    
    Parameters
    ----------
    mass_result : MassResult
        Mass calculation results
    include_warnings : bool, optional
        Whether to include warnings in output
        
    Returns
    -------
    str
        Formatted string with mass information
    """
    if not mass_result:
        return "No mass data available"
    
    lines = []
    
    # Total mass
    if mass_result.total_mass_error > 0:
        lines.append(f"Total Mass: {mass_result.total_mass_solar:.2f} ± {mass_result.total_mass_error:.2f} M☉")
    else:
        lines.append(f"Total Mass: {mass_result.total_mass_solar:.2f} M☉")
    
    # Distance and parallax info
    lines.append(f"Distance: {mass_result.distance_used_pc:.1f} pc (π = {mass_result.parallax_used_mas:.2f} mas)")
    lines.append(f"Parallax Source: {mass_result.parallax_source}")
    
    # Individual masses if available
    if mass_result.individual_masses_solar:
        m1, m2 = mass_result.individual_masses_solar
        if mass_result.individual_mass_errors:
            e1, e2 = mass_result.individual_mass_errors
            lines.append(f"Primary Mass: {m1:.2f} ± {e1:.2f} M☉")
            lines.append(f"Secondary Mass: {m2:.2f} ± {e2:.2f} M☉")
        else:
            lines.append(f"Primary Mass: {m1:.2f} M☉")
            lines.append(f"Secondary Mass: {m2:.2f} M☉")
        
        if mass_result.mass_ratio:
            if mass_result.mass_ratio_error:
                lines.append(f"Mass Ratio (q): {mass_result.mass_ratio:.3f} ± {mass_result.mass_ratio_error:.3f}")
            else:
                lines.append(f"Mass Ratio (q): {mass_result.mass_ratio:.3f}")
    
    # Quality and Monte Carlo info
    lines.append(f"Quality Score: {mass_result.quality_score:.2f}")
    if mass_result.total_mass_error > 0:
        lines.append(f"Monte Carlo Samples: {mass_result.mc_samples}")
    
    # Warnings
    if include_warnings and mass_result.warnings:
        lines.append("Warnings:")
        for warning in mass_result.warnings:
            lines.append(f"  - {warning}")
    
    return "\n".join(lines)
