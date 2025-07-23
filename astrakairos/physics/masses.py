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
from dataclasses import dataclass, replace

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
    MASS_MC_MIN_SAMPLES, MASS_MC_MAX_SAMPLES, MASS_MC_CONVERGENCE_THRESHOLD,
    # Quality assessment parameters
    MAX_QUALITY_PENALTY_FACTOR, LARGE_PARALLAX_UNCERTAINTY_THRESHOLD,
    PARALLAX_SOURCE_QUALITY
)

# Import custom exceptions
from ..exceptions import InvalidMassInputError, NumericalInstabilityError

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
) -> List[str]:
    """
    Validate inputs for mass calculation.
    
    Args:
        period_years: Orbital period in years
        semimajor_axis_arcsec: Semi-major axis in arcseconds
        parallax_mas: Parallax in milliarcseconds
        *_error: Uncertainties (1-sigma)
        
    Returns:
        List of warning messages
        
    Raises:
        InvalidMassInputError: If any input is outside valid ranges
    """
    warnings = []
    
    # Basic range validation
    if not (MIN_PERIOD_YEARS <= period_years <= MAX_PERIOD_YEARS):
        raise InvalidMassInputError(f"Period {period_years:.3f} years outside valid range [{MIN_PERIOD_YEARS}, {MAX_PERIOD_YEARS}]")
        
    if not (MIN_SEMIMAJOR_AXIS_ARCSEC <= semimajor_axis_arcsec):
        raise InvalidMassInputError(f"Semi-major axis {semimajor_axis_arcsec:.3f} arcsec below minimum {MIN_SEMIMAJOR_AXIS_ARCSEC}")
        
    if not (MIN_PARALLAX_MAS <= parallax_mas <= MAX_PARALLAX_MAS):
        raise InvalidMassInputError(f"Parallax {parallax_mas:.3f} mas outside valid range [{MIN_PARALLAX_MAS}, {MAX_PARALLAX_MAS}]")
    
    # Distance calculation and validation
    distance_pc = 1000.0 / parallax_mas  # Convert mas to pc
    if not (MIN_DISTANCE_PC <= distance_pc <= MAX_DISTANCE_PC):
        raise InvalidMassInputError(f"Distance {distance_pc:.1f} pc outside valid range [{MIN_DISTANCE_PC}, {MAX_DISTANCE_PC}]")
    
    # Warning conditions
    if parallax_mas < LOW_PARALLAX_WARNING_MAS:
        warnings.append(f"Low parallax precision ({parallax_mas:.2f} mas) may affect mass accuracy")
    
    if parallax_error > 0 and parallax_error / parallax_mas > LARGE_PARALLAX_UNCERTAINTY_THRESHOLD:
        warnings.append(f"Large parallax uncertainty ({parallax_error/parallax_mas*100:.1f}%)")
    
    # Error validation
    if period_error < 0 or semimajor_axis_error < 0 or parallax_error < 0:
        raise InvalidMassInputError("Negative uncertainties are not allowed")
    
    return warnings

def _calculate_mc_statistics(samples: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics from Monte Carlo samples.
    
    Args:
        samples: Array of samples from Monte Carlo simulation
        
    Returns:
        Dictionary with median, uncertainty, and percentiles
        
    Raises:
        NumericalInstabilityError: If no valid samples are available
    """
    if len(samples) == 0:
        raise NumericalInstabilityError("No Monte Carlo samples provided for statistical analysis")
    
    # Remove any invalid samples (NaN, infinite, or negative masses)
    valid_samples = samples[np.isfinite(samples) & (samples > 0)]
    
    if len(valid_samples) == 0:
        raise NumericalInstabilityError("No valid samples in Monte Carlo simulation - all samples are NaN, infinite, or negative")
    
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
) -> MassResult:
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
    MassResult
        Complete mass analysis with uncertainties
        
    Raises
    ------
    InvalidMassInputError
        If period, semi-major axis, or parallax values are outside valid ranges,
        or if any uncertainty values are negative
    NumericalInstabilityError
        If Monte Carlo simulation fails to produce valid samples for statistical analysis
        
    Notes
    -----
    Uses Kepler's Third Law in the form:
    M_total [M☉] = (a [AU])³ / (P [years])²
    
    Distance conversion: d [pc] = 1000 / π [mas]
    Semi-major axis: a [AU] = a [arcsec] × d [pc]
    
    Error propagation assumes Gaussian (normal) distributions for input uncertainties.
    This is standard practice for astronomical measurements and is appropriate when
    uncertainties represent 1-sigma confidence intervals from least-squares fits
    or similar statistical analyses.
    """
    
    # Validate inputs (raises InvalidMassInputError if invalid)
    warnings = _validate_mass_inputs(
        period_years, semimajor_axis_arcsec, parallax_mas,
        period_error, semimajor_axis_error, parallax_error
    )
    
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
        
        # Calculate statistics (raises NumericalInstabilityError if no valid samples)
        mc_stats = _calculate_mc_statistics(mass_samples)
        total_mass_solar = mc_stats['median']  # Use median as more robust estimate
        total_mass_error = mc_stats['uncertainty']
        
        # Quality assessment based on uncertainty
        if total_mass_error > 0:
            relative_error = total_mass_error / total_mass_solar
            if relative_error > LARGE_MASS_ERROR_THRESHOLD:
                warnings.append(f"Large mass uncertainty ({relative_error*100:.1f}%)")
                quality_score *= (1.0 - min(relative_error, MAX_QUALITY_PENALTY_FACTOR))
    
    # Assess quality based on parallax source
    quality_score *= PARALLAX_SOURCE_QUALITY.get(parallax_source.lower(), 0.1)
    
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
    
    This is a pure function that does not modify the input MassResult.
    It creates and returns a new MassResult with individual masses calculated.
    
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
        New mass result with individual masses calculated
        
    Raises
    ------
    InvalidMassInputError
        If mass ratio is outside valid range [MIN_MASS_RATIO, MAX_MASS_RATIO]
        or if any individual mass falls outside typical stellar mass range
    NumericalInstabilityError
        If Monte Carlo simulation fails to produce valid samples for error estimation
    """
    
    # Validate mass ratio
    if not (MIN_MASS_RATIO <= mass_ratio <= MAX_MASS_RATIO):
        raise InvalidMassInputError(f"Mass ratio {mass_ratio:.3f} outside valid range [{MIN_MASS_RATIO}, {MAX_MASS_RATIO}]")
    
    # Copy warnings from original result
    warnings = mass_result.warnings.copy()
    
    # Calculate point estimates
    m1_solar = mass_result.total_mass_solar / (1.0 + mass_ratio)
    m2_solar = mass_result.total_mass_solar * mass_ratio / (1.0 + mass_ratio)
    
    # Validate individual masses
    for mass, label in [(m1_solar, "Primary"), (m2_solar, "Secondary")]:
        if not (MIN_STELLAR_MASS_SOLAR <= mass <= MAX_STELLAR_MASS_SOLAR):
            warnings.append(f"{label} mass ({mass:.2f} M☉) outside typical stellar range")
    
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
        
        # Calculate uncertainties (raises NumericalInstabilityError if no valid samples)
        m1_stats = _calculate_mc_statistics(m1_samples)
        m2_stats = _calculate_mc_statistics(m2_samples)
        
        m1_solar = m1_stats['median']
        m2_solar = m2_stats['median']
        m1_error = m1_stats['uncertainty']
        m2_error = m2_stats['uncertainty']
    
    # Create new MassResult with individual masses (immutable pattern)
    return replace(
        mass_result,
        individual_masses_solar=(m1_solar, m2_solar),
        individual_mass_errors=(m1_error, m2_error),
        mass_ratio=mass_ratio,
        mass_ratio_error=mass_ratio_error,
        warnings=warnings
    )
