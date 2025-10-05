"""
El-Badry & Rix (2018) MNRAS 480, 4884-4902 
Pure proper motion-based binary classification following Equation (7).

This module implements the EXACT methodology from the 2018 paper:
- Uses ONLY Δμ (proper motion difference) as classification criterion
- NO χ² test as primary veto
- 3σ threshold (not 5σ)
- Includes μ_orbit calculation based on separation
- Absolute cut at 2×μ_orbit for high-uncertainty pairs
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_mu_orbit(separation_arcsec: float) -> float:
    """
    Calculate expected orbital proper motion based on separation.
    
    From El-Badry & Rix (2018) Equation (4):
    μ_orbit = 0.44 × (θ/arcsec)^(-1/2) mas/yr
    
    This assumes a circular orbit with the characteristic velocity dispersion
    of a typical binary star system.
    
    Parameters
    ----------
    separation_arcsec : float
        Angular separation in arcseconds
        
    Returns
    -------
    float
        Expected orbital proper motion in mas/yr
    """
    if separation_arcsec <= 0:
        logger.warning(f"Invalid separation {separation_arcsec} arcsec, using 0.1 arcsec")
        separation_arcsec = 0.1
    
    mu_orbit = 0.44 * (separation_arcsec ** -0.5)
    return mu_orbit


def calculate_delta_mu_threshold(
    separation_arcsec: float,
    delta_mu_uncertainty: float,
    sigma_multiplier: float = 3.0
) -> Tuple[float, float]:
    """
    Calculate proper motion difference threshold following El-Badry & Rix (2018).
    
    From Equation (7):
    μ ≤ μ_orbit + 3σ_μ
    
    With additional absolute cut:
    μ ≤ 2 × μ_orbit
    
    Parameters
    ----------
    separation_arcsec : float
        Angular separation in arcseconds
    delta_mu_uncertainty : float
        Uncertainty in proper motion difference (mas/yr)
    sigma_multiplier : float, optional
        Number of sigma for threshold (default 3.0 per paper)
        
    Returns
    -------
    tuple of (float, float)
        (relative_threshold, absolute_threshold) in mas/yr
        Use the MINIMUM of these two values
    """
    mu_orbit = calculate_mu_orbit(separation_arcsec)
    
    # Equation (7): relative threshold based on orbital motion + uncertainty
    relative_threshold = mu_orbit + sigma_multiplier * delta_mu_uncertainty
    
    # Absolute cut: reject if Δμ > 2×μ_orbit regardless of uncertainty
    absolute_threshold = 2.0 * mu_orbit
    
    return relative_threshold, absolute_threshold


def classify_pair_el_badry_rix_2018(
    delta_mu: float,
    delta_mu_uncertainty: float,
    separation_arcsec: float,
    sigma_multiplier: float = 3.0,
    ambiguous_sigma_multiplier: float = 5.0
) -> str:
    """
    Classify a binary pair using pure Δμ criterion from El-Badry & Rix (2018).
    
    Classification logic:
    1. Physical: Δμ ≤ min(μ_orbit + 3σ_Δμ, 2×μ_orbit)
    2. Ambiguous: 3σ < Δμ ≤ 5σ (conservative extension)
    3. Optical: Δμ > 5σ
    
    Parameters
    ----------
    delta_mu : float
        Proper motion difference magnitude (mas/yr)
    delta_mu_uncertainty : float
        Uncertainty in Δμ (mas/yr)
    separation_arcsec : float
        Angular separation (arcsec)
    sigma_multiplier : float, optional
        Sigma threshold for Physical classification (default 3.0)
    ambiguous_sigma_multiplier : float, optional
        Sigma threshold for Ambiguous/Optical boundary (default 5.0)
        
    Returns
    -------
    str
        "Physical", "Ambiguous", or "Optical"
    """
    if delta_mu_uncertainty <= 0:
        logger.warning(f"Invalid Δμ uncertainty {delta_mu_uncertainty}, using 0.1 mas/yr")
        delta_mu_uncertainty = 0.1
    
    # Calculate thresholds per El-Badry & Rix (2018)
    relative_threshold, absolute_threshold = calculate_delta_mu_threshold(
        separation_arcsec, delta_mu_uncertainty, sigma_multiplier
    )
    
    # Use MINIMUM of relative and absolute thresholds
    physical_threshold = min(relative_threshold, absolute_threshold)
    
    # Calculate ambiguous threshold (conservative extension beyond paper)
    mu_orbit = calculate_mu_orbit(separation_arcsec)
    ambiguous_relative = mu_orbit + ambiguous_sigma_multiplier * delta_mu_uncertainty
    ambiguous_absolute = 3.0 * mu_orbit  # More lenient absolute cut for ambiguous
    ambiguous_threshold = min(ambiguous_relative, ambiguous_absolute)
    
    # Classification
    if delta_mu <= physical_threshold:
        return "Physical"
    elif delta_mu <= ambiguous_threshold:
        return "Ambiguous"
    else:
        return "Optical"


def get_classification_details(
    delta_mu: float,
    delta_mu_uncertainty: float,
    separation_arcsec: float,
    sigma_multiplier: float = 3.0
) -> dict:
    """
    Get detailed classification information for debugging/analysis.
    
    Returns
    -------
    dict
        Dictionary with classification details including:
        - classification: str (Physical/Ambiguous/Optical)
        - delta_mu: float
        - delta_mu_uncertainty: float
        - separation_arcsec: float
        - mu_orbit: float (expected orbital PM)
        - relative_threshold: float (μ_orbit + 3σ)
        - absolute_threshold: float (2×μ_orbit)
        - physical_threshold: float (min of relative/absolute)
        - sigma_ratio: float (Δμ / σ_Δμ)
        - orbit_ratio: float (Δμ / μ_orbit)
    """
    mu_orbit = calculate_mu_orbit(separation_arcsec)
    relative_threshold, absolute_threshold = calculate_delta_mu_threshold(
        separation_arcsec, delta_mu_uncertainty, sigma_multiplier
    )
    physical_threshold = min(relative_threshold, absolute_threshold)
    
    classification = classify_pair_el_badry_rix_2018(
        delta_mu, delta_mu_uncertainty, separation_arcsec, sigma_multiplier
    )
    
    return {
        "classification": classification,
        "delta_mu": delta_mu,
        "delta_mu_uncertainty": delta_mu_uncertainty,
        "separation_arcsec": separation_arcsec,
        "mu_orbit": mu_orbit,
        "relative_threshold": relative_threshold,
        "absolute_threshold": absolute_threshold,
        "physical_threshold": physical_threshold,
        "sigma_ratio": delta_mu / delta_mu_uncertainty if delta_mu_uncertainty > 0 else np.inf,
        "orbit_ratio": delta_mu / mu_orbit if mu_orbit > 0 else np.inf,
    }
