"""
Expert Hierarchical Validator for Binary Star Physicality Assessment

This module implements the "Expert Hierarchical Validator" - a rule-based decision
system that emulates the reasoning process of an expert astronomer when evaluating
whether a binary star system is physically bound or optical.

Philosophy: "Formalized Expert Reasoning"
- Each decision follows an explicit chain of reasoning
- Poor quality data triggers conservative behavior  
- Conflicting evidence is handled through hierarchical rules
- Every conclusion includes transparent justification

The validator implements a veto-based hierarchy:
1. VETO ÓPTICO #1: Astrometría inconsistente con datos de alta calidad
2. VETO ÓPTICO #2: Paralaje absolutamente incompatible (>10σ)
3. CASO DE ALTA CONFIANZA FÍSICA: Toda la evidencia apoya
4. MANEJO DE RUWE ALTO: Datos de mala calidad
5. MANEJO DE MOVIMIENTO ORBITAL: Paralaje OK, PM no
6. FALLBACK: Cualquier otro caso
"""

import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from scipy.stats import chi2

# Import configuration
from ..config import (
    DEFAULT_PHYSICAL_P_VALUE_THRESHOLD,
    DEFAULT_AMBIGUOUS_P_VALUE_THRESHOLD,
    EXPERT_PARALLAX_EXTREME_SIGMA_THRESHOLD,
    EXPERT_PM_EXTREME_SIGMA_THRESHOLD,
    EXPERT_RUWE_GOOD_THRESHOLD,
    EXPERT_PARALLAX_COMPATIBILITY_SIGMA_THRESHOLD,
    EXPERT_PM_COMPATIBILITY_SIGMA_THRESHOLD,
    EXPERT_CONFIDENCE_VERY_HIGH,
    EXPERT_CONFIDENCE_HIGH,
    EXPERT_CONFIDENCE_MEDIUM,
    EXPERT_CONFIDENCE_LOW,
    EXPERT_METHOD_OPTICAL_VETO_ASTROMETRY,
    EXPERT_METHOD_OPTICAL_VETO_PARALLAX,
    EXPERT_METHOD_OPTICAL_VETO_PROPER_MOTION,
    EXPERT_METHOD_PHYSICAL_HIGH_CONFIDENCE,
    EXPERT_METHOD_PHYSICAL_PM_ORBITAL,
    EXPERT_METHOD_PHYSICAL_BAD_RUWE,
    EXPERT_METHOD_AMBIGUOUS_FALLBACK
)

from ..data.gaia_utils import get_adaptive_delta_mu_thresholds

# Import Gaia utilities
from ..data.gaia_utils import (
    correct_gaia_error,
    get_gaia_parallax_error_safe,
    get_gaia_pmra_error_safe,
    get_gaia_pmdec_error_safe,
    assess_gaia_data_quality
)

# Import data types
from ..data.source import PhysicalityLabel, ValidationMethod

log = logging.getLogger(__name__)

class EvidencePriority(Enum):
    """Priority levels for different types of evidence in the decision hierarchy."""
    CRITICAL = "critical"      # Veto-level evidence (extreme parallax differences)
    HIGH = "high"             # Strong evidence (consistent astrometry with good data)
    MEDIUM = "medium"         # Moderate evidence (some conflicting indicators)
    LOW = "low"              # Weak evidence (poor quality or ambiguous data)

class ValidationEvidence(Enum):
    """Types of evidence that can be evaluated."""
    SUPPORTS_PHYSICAL = "supports_physical"
    SUPPORTS_OPTICAL = "supports_optical"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"

@dataclass
class ValidationCriterion:
    """Result of evaluating a single validation criterion."""
    passed: Optional[bool]
    evidence: ValidationEvidence
    priority: EvidencePriority
    evidence_strength: float  # Renamed from confidence for clarity
    p_value: Optional[float]  # Explicitly track p-value reliability
    details: Dict[str, Any]
    reasoning: str

@dataclass 
class ExpertDecisionResult:
    """Final result from the Expert Hierarchical Validator."""
    label: PhysicalityLabel
    evidence_strength: float
    method: str
    p_value: Optional[float]
    reasoning: str
    evidence_chain: List[ValidationCriterion]
    data_quality_assessment: Dict[str, Any]

class ExpertHierarchicalValidator:
    """
    Expert-level physicality validator using hierarchical decision rules.
    
    This validator implements the decision-making process of an experienced
    astronomer, using explicit rules and vetos to handle complex cases
    with conflicting evidence or poor data quality.
    
    Implements the decision hierarchy with explicit vetos and
    conservative fallback behavior.
    """
    
    def __init__(self,
                 physical_threshold: float = DEFAULT_PHYSICAL_P_VALUE_THRESHOLD,
                 ambiguous_threshold: float = DEFAULT_AMBIGUOUS_P_VALUE_THRESHOLD,
                 enable_ruwe_correction: bool = True):
        """
        Initialize the Expert Hierarchical Validator.
        
        Args:
            physical_threshold: P-value threshold for physical classification
            ambiguous_threshold: P-value threshold for ambiguous classification
            enable_ruwe_correction: Whether to apply RUWE-based error correction
        """
        self.physical_threshold = physical_threshold
        self.ambiguous_threshold = ambiguous_threshold
        self.enable_ruwe_correction = enable_ruwe_correction
        
        log.info(f"Expert Hierarchical Validator initialized: "
                f"thresholds=({physical_threshold:.3f}, {ambiguous_threshold:.3f}), "
                f"RUWE_correction={enable_ruwe_correction}")

    def validate_pair(self,
                     primary_gaia: Dict[str, Any],
                     secondary_gaia: Dict[str, Any],
                     separation_arcsec: Optional[float] = None) -> ExpertDecisionResult:
        """
        Expert validation of a binary star pair using hierarchical decision rules.
        
        Args:
            primary_gaia: Primary component Gaia data
            secondary_gaia: Secondary component Gaia data
            separation_arcsec: Angular separation between components (optional)
            
        Returns:
            ExpertDecisionResult with classification and reasoning
        """
        from ..data.gaia_utils import inflate_gaia_uncertainties
        
        primary_gaia = inflate_gaia_uncertainties(primary_gaia, separation_arcsec)
        secondary_gaia = inflate_gaia_uncertainties(secondary_gaia, separation_arcsec)
        
        primary_quality = assess_gaia_data_quality(primary_gaia)
        secondary_quality = assess_gaia_data_quality(secondary_gaia)
        
        # Step 1.5: Check for problematic correlations
        correlation_warning = self._check_correlation_quality(primary_gaia, secondary_gaia)
        if correlation_warning:
            log.warning(f"Correlation quality issue detected: {correlation_warning}")
            primary_quality['correlation_warning'] = correlation_warning
            secondary_quality['correlation_warning'] = correlation_warning
        
        evidence_chain = []
        
        astrometric_result = self._evaluate_astrometric_consistency(
            primary_gaia, secondary_gaia, primary_quality, secondary_quality
        )
        evidence_chain.append(astrometric_result)
        
        parallax_result = self._evaluate_parallax_compatibility(
            primary_gaia, secondary_gaia, primary_quality, secondary_quality,
            separation_arcsec
        )
        evidence_chain.append(parallax_result)
        
        # Proper motion alignment (with adaptive thresholds)
        pm_result = self._evaluate_proper_motion_alignment(
            primary_gaia, secondary_gaia, primary_quality, secondary_quality,
            separation_arcsec
        )
        evidence_chain.append(pm_result)
        
        # RUWE quality assessment
        ruwe_result = self._evaluate_ruwe_quality(
            primary_gaia, secondary_gaia, primary_quality, secondary_quality
        )
        evidence_chain.append(ruwe_result)
        
        # Step 3: Apply expert decision hierarchy
        decision = self._make_expert_decision(
            evidence_chain, primary_quality, secondary_quality
        )
        
        return ExpertDecisionResult(
            label=decision['label'],
            evidence_strength=decision['evidence_strength'],
            method=decision['method'],
            p_value=decision.get('p_value'),
            reasoning=decision['reasoning'],
            evidence_chain=evidence_chain,
            data_quality_assessment={
                'primary': primary_quality,
                'secondary': secondary_quality
            }
        )

    def _evaluate_astrometric_consistency(self,
                                       primary: Dict[str, Any],
                                       secondary: Dict[str, Any],
                                       primary_quality: Dict[str, Any],
                                       secondary_quality: Dict[str, Any]) -> ValidationCriterion:
        """
        Evaluate global astrometric consistency using chi-squared test.
        
        This is the most comprehensive test - if all astrometric parameters
        (parallax + proper motion) are consistent, it's strong evidence for
        a physical pair.
        
        Returns the calculated p-value regardless of data quality. Quality
        interpretation is handled by _make_expert_decision.
        """
        try:
            # Check for correlation warnings that might affect chi-squared reliability
            correlation_warning = primary_quality.get('correlation_warning') or secondary_quality.get('correlation_warning')
            
            # Attempt 3D chi-squared test (parallax + proper motion)
            chi2_result = self._calculate_chi2_3d_corrected(primary, secondary)
            
            if chi2_result is not None:
                chi2_val, dof = chi2_result
                p_value = chi2.sf(chi2_val, df=dof)
                
                # Adjust interpretation if correlations are problematic
                effective_quality = (
                    primary_quality.get('ruwe_quality', False) and 
                    secondary_quality.get('ruwe_quality', False) and
                    not correlation_warning  # Downgrade if correlations are problematic
                )
                
                # Determine evidence based on p-value (data quality handled separately)
                if p_value > self.physical_threshold:
                    evidence = ValidationEvidence.SUPPORTS_PHYSICAL
                    passed = True
                    priority = EvidencePriority.HIGH if effective_quality else EvidencePriority.MEDIUM
                    evidence_strength = p_value
                    reasoning = f"Astrometric parameters are statistically consistent (χ²={chi2_val:.2f}, dof={dof}, p={p_value:.4f})"
                    if correlation_warning:
                        reasoning += " [correlation warning: interpret with caution]"
                elif p_value < self.ambiguous_threshold:
                    evidence = ValidationEvidence.SUPPORTS_OPTICAL
                    passed = False
                    priority = EvidencePriority.HIGH if effective_quality else EvidencePriority.MEDIUM
                    evidence_strength = 1.0 - p_value
                    reasoning = f"Astrometric parameters are statistically inconsistent (χ²={chi2_val:.2f}, dof={dof}, p={p_value:.4f})"
                    if correlation_warning:
                        reasoning += " [correlation warning: interpret with caution]"
                else:
                    evidence = ValidationEvidence.AMBIGUOUS
                    passed = None
                    priority = EvidencePriority.MEDIUM
                    evidence_strength = 0.5
                    reasoning = f"Astrometric consistency is ambiguous (χ²={chi2_val:.2f}, dof={dof}, p={p_value:.4f})"
                    if correlation_warning:
                        reasoning += " [correlation warning: interpret with caution]"
                
                return ValidationCriterion(
                    passed=passed,
                    evidence=evidence,
                    priority=priority,
                    evidence_strength=evidence_strength,
                    p_value=p_value,  # Always return the calculated p-value
                    details={'chi2': chi2_val, 'dof': dof, 'p_value': p_value},
                    reasoning=reasoning
                )
                
        except Exception as e:
            log.warning(f"Failed to evaluate astrometric consistency: {e}")
        
        # Fallback for missing/invalid data
        return ValidationCriterion(
            passed=None,
            evidence=ValidationEvidence.UNAVAILABLE,
            priority=EvidencePriority.LOW,
            evidence_strength=0.0,
            p_value=None,
            details={'error': 'insufficient_astrometric_data'},
            reasoning="Insufficient astrometric data for global consistency test"
        )

    def _evaluate_parallax_compatibility(self,
                                      primary: Dict[str, Any],
                                      secondary: Dict[str, Any],
                                      primary_quality: Dict[str, Any],
                                      secondary_quality: Dict[str, Any],
                                      separation_arcsec: Optional[float] = None) -> ValidationCriterion:
        """
        Evaluate parallax compatibility between components.
        
        This is often the most reliable indicator for wide binaries where
        proper motion may be dominated by orbital motion.
        """
        try:
            plx1 = primary.get('parallax')
            plx2 = secondary.get('parallax')
            
            if plx1 is None or plx2 is None:
                return ValidationCriterion(
                    passed=None,
                    evidence=ValidationEvidence.UNAVAILABLE,
                    priority=EvidencePriority.LOW,
                    evidence_strength=0.0,
                    p_value=None,
                    details={'error': 'missing_parallax'},
                    reasoning="Parallax data unavailable for one or both components"
                )
            
            plx_err1 = get_gaia_parallax_error_safe(
                primary.get('parallax_error'), 
                primary.get('ruwe')
            )
            plx_err2 = get_gaia_parallax_error_safe(
                secondary.get('parallax_error'),
                secondary.get('ruwe')
            )
            
            plx_diff = abs(plx1 - plx2)
            combined_error = np.sqrt(plx_err1**2 + plx_err2**2)
            
            if combined_error > 0:
                sigma_difference = plx_diff / combined_error
                
                parallax_threshold = EXPERT_PARALLAX_COMPATIBILITY_SIGMA_THRESHOLD
                if separation_arcsec is not None and separation_arcsec < 4.0:
                    from ..config import GAIA_ADAPTIVE_THRESHOLDS
                    multiplier = GAIA_ADAPTIVE_THRESHOLDS['close_separation']['parallax_sigma_multiplier']
                    parallax_threshold *= multiplier
                
                if sigma_difference > EXPERT_PARALLAX_EXTREME_SIGMA_THRESHOLD:
                    return ValidationCriterion(
                        passed=False,
                        evidence=ValidationEvidence.SUPPORTS_OPTICAL,
                        priority=EvidencePriority.CRITICAL,
                        evidence_strength=EXPERT_CONFIDENCE_VERY_HIGH,
                        p_value=None,
                        details={
                            'plx1': plx1, 'plx2': plx2, 'difference': plx_diff,
                            'sigma_difference': sigma_difference, 'threshold': EXPERT_PARALLAX_EXTREME_SIGMA_THRESHOLD
                        },
                        reasoning=f"VETO ÓPTICO #2: Extreme parallax difference ({sigma_difference:.1f}σ > {EXPERT_PARALLAX_EXTREME_SIGMA_THRESHOLD}σ) indicates different distances"
                    )
                elif sigma_difference < parallax_threshold:
                    priority = EvidencePriority.HIGH if (
                        primary_quality.get('parallax_quality', False) and 
                        secondary_quality.get('parallax_quality', False)
                    ) else EvidencePriority.MEDIUM
                    
                    return ValidationCriterion(
                        passed=True,
                        evidence=ValidationEvidence.SUPPORTS_PHYSICAL,
                        priority=priority,
                        evidence_strength=1.0 - (sigma_difference / parallax_threshold),
                        p_value=None,
                        details={
                            'plx1': plx1, 'plx2': plx2, 'difference': plx_diff,
                            'sigma_difference': sigma_difference, 'threshold': parallax_threshold
                        },
                        reasoning=f"Parallax values are compatible ({sigma_difference:.1f}σ < {parallax_threshold:.1f}σ)"
                    )
                else:
                    # Ambiguous parallax difference
                    return ValidationCriterion(
                        passed=None,
                        evidence=ValidationEvidence.AMBIGUOUS,
                        priority=EvidencePriority.MEDIUM,
                        evidence_strength=0.5,
                        p_value=None,
                        details={
                            'plx1': plx1, 'plx2': plx2, 'difference': plx_diff,
                            'sigma_difference': sigma_difference
                        },
                        reasoning=f"Parallax compatibility is ambiguous ({sigma_difference:.1f}σ)"
                    )
                    
        except Exception as e:
            log.warning(f"Failed to evaluate parallax compatibility: {e}")
        
        return ValidationCriterion(
            passed=None,
            evidence=ValidationEvidence.UNAVAILABLE,
            priority=EvidencePriority.LOW,
            evidence_strength=0.0,
            p_value=None,
            details={'error': 'parallax_calculation_failed'},
            reasoning="Failed to calculate parallax compatibility"
        )

    def _evaluate_proper_motion_alignment(self,
                                       primary: Dict[str, Any],
                                       secondary: Dict[str, Any],
                                       primary_quality: Dict[str, Any],
                                       secondary_quality: Dict[str, Any],
                                       separation_arcsec: Optional[float] = None) -> ValidationCriterion:
        """
        Evaluate proper motion alignment between components.
        
        Uses adaptive thresholds based on separation and RUWE following
        El-Badry et al. (2021) methodology: 5σ for Δμ physical threshold,
        8σ for ambiguous threshold.
        
        For physical pairs, proper motions should be similar unless
        orbital motion is significant.
        """
        try:
            # Get proper motion values
            pmra1, pmdec1 = primary.get('pmra'), primary.get('pmdec')
            pmra2, pmdec2 = secondary.get('pmra'), secondary.get('pmdec')
            
            if any(x is None for x in [pmra1, pmdec1, pmra2, pmdec2]):
                return ValidationCriterion(
                    passed=None,
                    evidence=ValidationEvidence.UNAVAILABLE,
                    priority=EvidencePriority.LOW,
                    evidence_strength=0.0,
                    p_value=None,
                    details={'error': 'missing_proper_motion'},
                    reasoning="Proper motion data unavailable for one or both components"
                )
            
            # Get safe errors with RUWE correction
            pmra_err1 = get_gaia_pmra_error_safe(
                primary.get('pmra_error'),
                primary.get('ruwe')
            )
            pmdec_err1 = get_gaia_pmdec_error_safe(
                primary.get('pmdec_error'),
                primary.get('ruwe')
            )
            pmra_err2 = get_gaia_pmra_error_safe(
                secondary.get('pmra_error'),
                secondary.get('ruwe')
            )
            pmdec_err2 = get_gaia_pmdec_error_safe(
                secondary.get('pmdec_error'),
                secondary.get('ruwe')
            )
            
            # Calculate 2D proper motion difference
            dpmra = pmra1 - pmra2
            dpmdec = pmdec1 - pmdec2
            pm_diff_magnitude = np.sqrt(dpmra**2 + dpmdec**2)
            
            # Combined error
            combined_pmra_err = np.sqrt(pmra_err1**2 + pmra_err2**2)
            combined_pmdec_err = np.sqrt(pmdec_err1**2 + pmdec_err2**2)
            combined_pm_err = np.sqrt(combined_pmra_err**2 + combined_pmdec_err**2)
            
            # =====================================================================
            # CRITICAL: Calculate Δμ_orbit (El-Badry & Rix 2018)
            # This is the tangential component of PM difference, which is the 
            # PRIMARY criterion for physicality (not the 2D magnitude)
            # =====================================================================
            delta_mu_orbit = None
            delta_mu_orbit_sigma = None
            
            if separation_arcsec is not None and separation_arcsec > 0:
                try:
                    from astropy.coordinates import SkyCoord
                    import astropy.units as u
                    
                    # Get sky coordinates
                    ra1 = primary.get('ra')
                    dec1 = primary.get('dec')
                    ra2 = secondary.get('ra')
                    dec2 = secondary.get('dec')
                    
                    if all(x is not None for x in [ra1, dec1, ra2, dec2]):
                        coord1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
                        coord2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
                        
                        # Get position angle and calculate tangential unit vector
                        position_angle = coord1.position_angle(coord2)
                        pa_rad = position_angle.to(u.radian).value
                        
                        # Tangential unit vector (perpendicular to radial)
                        # Following El-Badry: e_t = (-sin(PA), cos(PA)) rotated 90°
                        delta_ra = separation_arcsec * np.sin(pa_rad)
                        delta_dec = separation_arcsec * np.cos(pa_rad)
                        r_norm = np.hypot(delta_ra, delta_dec)
                        
                        if r_norm > 0:
                            e_r_ra = delta_ra / r_norm
                            e_r_dec = delta_dec / r_norm
                            # Tangential is perpendicular to radial
                            e_t_ra = -e_r_dec
                            e_t_dec = e_r_ra
                            
                            # Project PM difference onto tangential direction
                            delta_mu_orbit = float(dpmra * e_t_ra + dpmdec * e_t_dec)
                            
                            # Calculate error in tangential component using covariance
                            corr1 = float(primary.get('pmra_pmdec_corr', 0.0) or 0.0)
                            corr2 = float(secondary.get('pmra_pmdec_corr', 0.0) or 0.0)
                            
                            cov1 = np.array([
                                [pmra_err1**2, pmra_err1 * pmdec_err1 * corr1],
                                [pmra_err1 * pmdec_err1 * corr1, pmdec_err1**2]
                            ])
                            cov2 = np.array([
                                [pmra_err2**2, pmra_err2 * pmdec_err2 * corr2],
                                [pmra_err2 * pmdec_err2 * corr2, pmdec_err2**2]
                            ])
                            
                            cov_delta = cov1 + cov2
                            e_t = np.array([e_t_ra, e_t_dec])
                            variance = float(e_t.T @ cov_delta @ e_t)
                            variance = max(variance, 0.0)
                            delta_mu_orbit_error = float(np.sqrt(variance)) if variance > 0 else float('inf')
                            
                            if np.isfinite(delta_mu_orbit_error) and delta_mu_orbit_error > 0:
                                delta_mu_orbit_sigma = abs(delta_mu_orbit) / delta_mu_orbit_error
                                log.debug(f"✅ Δμ_orbit calculated: {delta_mu_orbit_sigma:.2f}σ (tangential)")
                        else:
                            log.warning(f"Failed to calculate Δμ_orbit: r_norm = 0")
                    else:
                        missing = [k for k, v in {'ra1': ra1, 'dec1': dec1, 'ra2': ra2, 'dec2': dec2}.items() if v is None]
                        log.warning(f"Failed to calculate Δμ_orbit: missing coordinates {missing}")
                            
                except Exception as e:
                    log.warning(f"Failed to calculate Δμ_orbit: {e}")
            
            if combined_pm_err > 0:
                sigma_difference = pm_diff_magnitude / combined_pm_err
                
                # Get adaptive thresholds based on separation and RUWE
                ruwe_p = primary.get('ruwe')
                ruwe_s = secondary.get('ruwe')
                
                if separation_arcsec is not None and ruwe_p is not None and ruwe_s is not None:
                    physical_thresh, ambiguous_thresh = get_adaptive_delta_mu_thresholds(
                        separation_arcsec, ruwe_p, ruwe_s
                    )
                else:
                    # Fallback to old hardcoded values if data unavailable
                    physical_thresh = EXPERT_PM_COMPATIBILITY_SIGMA_THRESHOLD
                    ambiguous_thresh = EXPERT_PM_EXTREME_SIGMA_THRESHOLD
                
                if sigma_difference < physical_thresh:
                    # Aligned proper motion
                    priority = EvidencePriority.HIGH if (
                        primary_quality.get('pm_quality', False) and 
                        secondary_quality.get('pm_quality', False)
                    ) else EvidencePriority.MEDIUM
                    
                    return ValidationCriterion(
                        passed=True,
                        evidence=ValidationEvidence.SUPPORTS_PHYSICAL,
                        priority=priority,
                        evidence_strength=1.0 - (sigma_difference / physical_thresh),
                        p_value=None,
                        details={
                            'pm_diff_magnitude': pm_diff_magnitude,
                            'sigma_difference': sigma_difference,
                            'threshold': physical_thresh,
                            'separation_arcsec': separation_arcsec,
                            'delta_mu_orbit': delta_mu_orbit,
                            'delta_mu_orbit_sigma': delta_mu_orbit_sigma
                        },
                        reasoning=f"Proper motions are aligned ({sigma_difference:.1f}σ < {physical_thresh:.1f}σ)"
                    )
                else:
                    # Misaligned proper motion - could be orbital motion or optical pair
                    # Apply extreme veto if difference exceeds ambiguous threshold
                    if sigma_difference > ambiguous_thresh:
                        return ValidationCriterion(
                            passed=False,
                            evidence=ValidationEvidence.SUPPORTS_OPTICAL,
                            priority=EvidencePriority.HIGH,
                            evidence_strength=0.9,
                            p_value=None,
                            details={
                                'pm_diff_magnitude': pm_diff_magnitude,
                                'sigma_difference': sigma_difference,
                                'extreme_threshold': ambiguous_thresh,
                                'separation_arcsec': separation_arcsec,
                                'delta_mu_orbit': delta_mu_orbit,
                                'delta_mu_orbit_sigma': delta_mu_orbit_sigma
                            },
                            reasoning=f"VETO ÓPTICO PM: Extreme proper motion difference ({sigma_difference:.1f}σ > {ambiguous_thresh:.1f}σ) indicates unrelated motion"
                        )
                    else:
                        return ValidationCriterion(
                            passed=False,
                            evidence=ValidationEvidence.AMBIGUOUS,
                            priority=EvidencePriority.MEDIUM,
                            evidence_strength=0.5,
                            p_value=None,
                            details={
                                'pm_diff_magnitude': pm_diff_magnitude,
                                'sigma_difference': sigma_difference,
                                'physical_threshold': physical_thresh,
                                'ambiguous_threshold': ambiguous_thresh,
                                'separation_arcsec': separation_arcsec,
                                'delta_mu_orbit': delta_mu_orbit,
                                'delta_mu_orbit_sigma': delta_mu_orbit_sigma
                            },
                            reasoning=f"Proper motions are misaligned ({sigma_difference:.1f}σ > {physical_thresh:.1f}σ) - could indicate orbital motion or optical pair"
                        )
                    
        except Exception as e:
            log.warning(f"Failed to evaluate proper motion alignment: {e}")
        
        return ValidationCriterion(
            passed=None,
            evidence=ValidationEvidence.UNAVAILABLE,
            priority=EvidencePriority.LOW,
            evidence_strength=0.0,
            p_value=None,
            details={'error': 'pm_calculation_failed'},
            reasoning="Failed to calculate proper motion alignment"
        )

    def _evaluate_ruwe_quality(self,
                            primary: Dict[str, Any],
                            secondary: Dict[str, Any],
                            primary_quality: Dict[str, Any],
                            secondary_quality: Dict[str, Any]) -> ValidationCriterion:
        """
        Evaluate RUWE quality for both components.
        
        Good RUWE indicates reliable astrometric solutions, while poor RUWE
        suggests systematic issues that could affect physicality assessment.
        """
        ruwe1 = primary.get('ruwe')
        ruwe2 = secondary.get('ruwe')
        
        if ruwe1 is None and ruwe2 is None:
            return ValidationCriterion(
                passed=None,
                evidence=ValidationEvidence.UNAVAILABLE,
                priority=EvidencePriority.LOW,
                evidence_strength=0.0,
                p_value=None,
                details={'error': 'missing_ruwe'},
                reasoning="RUWE data unavailable for both components"
            )
        
        # Check if at least one component has good RUWE
        good_ruwe_count = 0
        if ruwe1 is not None and ruwe1 <= EXPERT_RUWE_GOOD_THRESHOLD:
            good_ruwe_count += 1
        if ruwe2 is not None and ruwe2 <= EXPERT_RUWE_GOOD_THRESHOLD:
            good_ruwe_count += 1
        
        max_ruwe = max(x for x in [ruwe1, ruwe2] if x is not None)
        
        if good_ruwe_count == 2:
            return ValidationCriterion(
                passed=True,
                evidence=ValidationEvidence.SUPPORTS_PHYSICAL,
                priority=EvidencePriority.HIGH,
                evidence_strength=0.8,
                p_value=None,
                details={'ruwe1': ruwe1, 'ruwe2': ruwe2, 'threshold': EXPERT_RUWE_GOOD_THRESHOLD},
                reasoning=f"Both components have good RUWE (≤{EXPERT_RUWE_GOOD_THRESHOLD})"
            )
        elif good_ruwe_count == 1:
            return ValidationCriterion(
                passed=True,
                evidence=ValidationEvidence.SUPPORTS_PHYSICAL,
                priority=EvidencePriority.MEDIUM,
                evidence_strength=0.6,
                p_value=None,
                details={'ruwe1': ruwe1, 'ruwe2': ruwe2, 'threshold': EXPERT_RUWE_GOOD_THRESHOLD},
                reasoning=f"One component has good RUWE (≤{EXPERT_RUWE_GOOD_THRESHOLD})"
            )
        else:
            return ValidationCriterion(
                passed=False,
                evidence=ValidationEvidence.AMBIGUOUS,
                priority=EvidencePriority.MEDIUM,
                evidence_strength=0.3,
                p_value=None,
                details={'ruwe1': ruwe1, 'ruwe2': ruwe2, 'threshold': EXPERT_RUWE_GOOD_THRESHOLD},
                reasoning=f"Both components have poor RUWE (>{EXPERT_RUWE_GOOD_THRESHOLD}) indicating unreliable astrometry"
            )

    def _make_expert_decision(self,
                            evidence_chain: List[ValidationCriterion],
                            primary_quality: Dict[str, Any],
                            secondary_quality: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply expert hierarchical decision rules.
        
        UPDATED HIERARCHY (El-Badry 2021 aligned, Oct 2025):
        Following El-Badry et al. (2021) Section 4.4, we prioritize Δμ check
        over chi-squared test. El-Badry uses ΔV < V_orb + 2σ_V as PRIMARY criterion.
        
        New hierarchy:
        0. ΔΜORBIT CHECK (NEW): Proper motion consistency with orbital motion
        1. VETO ÓPTICO #1: Paralaje absolutamente incompatible (>5σ)
        2. VETO ÓPTICO #2: Movimiento Propio Extremo (>12σ)
        3. CASO DE ALTA CONFIANZA FÍSICA: Δμ_orbit + astrometry + RUWE all good
        4. MANEJO DE RUWE ALTO: Datos de mala calidad
        5. MANEJO DE MOVIMIENTO ORBITAL: Paralaje OK, PM marginally inconsistent
        6. FALLBACK: Cualquier otro caso
        """
        from ..config import (EXPERT_DELTA_MU_PRIMARY_VETO, ORBIT_EXCESS_SIGMA_PHYSICAL_MAX,
                             ORBIT_EXCESS_SIGMA_AMBIGUOUS_MAX)
        
        # Extract evidence by type
        astrometric_evidence = evidence_chain[0]  # Global chi-squared
        parallax_evidence = evidence_chain[1]     # Parallax compatibility
        pm_evidence = evidence_chain[2]           # Proper motion alignment
        ruwe_evidence = evidence_chain[3]         # RUWE quality
        
        # NEW: Δμ_orbit Primary Check (El-Badry 2021 methodology)
        # Check proper motion consistency FIRST, override chi-squared if consistent
        # CRITICAL: Use delta_mu_orbit_sigma (tangential component), NOT sigma_difference (2D magnitude)
        if EXPERT_DELTA_MU_PRIMARY_VETO:
            # Try to get Δμ_orbit first (El-Badry metric), fallback to standard PM if unavailable
            delta_mu_sigma = pm_evidence.details.get('delta_mu_orbit_sigma')
            pm_sigma_2d = pm_evidence.details.get('sigma_difference', float('inf'))
            
            # Prefer Δμ_orbit if available, otherwise use standard PM
            pm_sigma = delta_mu_sigma if delta_mu_sigma is not None else pm_sigma_2d
            
            # Log which metric is being used
            if delta_mu_sigma is not None:
                log.debug(f"Using Δμ_orbit_sigma = {delta_mu_sigma:.2f}σ for primary check")
            else:
                log.debug(f"Δμ_orbit unavailable, using PM 2D sigma = {pm_sigma_2d:.2f}σ")
            
            # Physical if Δμ < 5σ (El-Badry-inspired threshold)
            if pm_sigma < ORBIT_EXCESS_SIGMA_PHYSICAL_MAX:
                # Additional check: parallax must not be extreme
                if parallax_evidence.priority != EvidencePriority.CRITICAL:
                    metric_used = "Δμ_orbit" if delta_mu_sigma is not None else "PM difference"
                    return {
                        'label': PhysicalityLabel.LIKELY_PHYSICAL,
                        'evidence_strength': EXPERT_CONFIDENCE_HIGH,
                        'method': EXPERT_METHOD_PHYSICAL_HIGH_CONFIDENCE,
                        'p_value': astrometric_evidence.details.get('p_value'),
                        'reasoning': f"ΔΜORBIT PHYSICAL: {metric_used} = {pm_sigma:.1f}σ < {ORBIT_EXCESS_SIGMA_PHYSICAL_MAX}σ - consistent with orbital motion (El-Badry 2021 criterion)"
                    }
            
            # Ambiguous if 5σ < Δμ < 10σ
            elif pm_sigma < ORBIT_EXCESS_SIGMA_AMBIGUOUS_MAX:
                if parallax_evidence.passed != False:
                    return {
                        'label': PhysicalityLabel.AMBIGUOUS,
                        'evidence_strength': EXPERT_CONFIDENCE_MEDIUM,
                        'method': EXPERT_METHOD_AMBIGUOUS_FALLBACK,
                        'p_value': astrometric_evidence.details.get('p_value'),
                        'reasoning': f"ΔΜORBIT AMBIGUOUS: Moderate proper motion difference ({pm_sigma:.1f}σ) - could be orbital or optical"
                    }
        
        # VETO ÓPTICO #1: Paralaje Absolutamente Incompatible
        if parallax_evidence.priority == EvidencePriority.CRITICAL:
            return {
                'label': PhysicalityLabel.LIKELY_OPTICAL,
                'evidence_strength': EXPERT_CONFIDENCE_VERY_HIGH,
                'method': EXPERT_METHOD_OPTICAL_VETO_PARALLAX,
                'p_value': None,
                'reasoning': f"VETO ÓPTICO PARALLAX: {parallax_evidence.reasoning} - distance difference too large to be physical"
            }
        
        # VETO ÓPTICO #2: Movimiento Propio Extremo (now at 12σ threshold)
        # CRITICAL FIX: Only veto if Δμ_orbit > 12σ, not just PM 2D magnitude
        # This prevents false optical classifications for wide pairs with high PM but consistent orbits
        if (pm_evidence.passed == False and 
            pm_evidence.evidence == ValidationEvidence.SUPPORTS_OPTICAL and
            pm_evidence.priority == EvidencePriority.HIGH):
            
            # Check if Δμ_orbit is available and use it for veto decision
            delta_mu_sigma = pm_evidence.details.get('delta_mu_orbit_sigma')
            pm_sigma_2d = pm_evidence.details.get('sigma_difference', float('inf'))
            
            # Use Δμ_orbit if available, otherwise fallback to PM 2D
            effective_sigma = delta_mu_sigma if delta_mu_sigma is not None else pm_sigma_2d
            
            # Log which metric is being used for transparency
            if delta_mu_sigma is not None:
                log.debug(f"PM veto check: Using Δμ_orbit = {delta_mu_sigma:.2f}σ (tangential component)")
            else:
                log.warning(f"PM veto check: Δμ_orbit unavailable, using PM 2D = {pm_sigma_2d:.2f}σ (FALLBACK)")
            
            # Only apply veto if Δμ_orbit (or PM 2D as fallback) truly exceeds threshold
            if effective_sigma > EXPERT_PM_EXTREME_SIGMA_THRESHOLD:
                metric_used = "Δμ_orbit" if delta_mu_sigma is not None else "PM 2D (FALLBACK)"
                return {
                    'label': PhysicalityLabel.LIKELY_OPTICAL,
                    'evidence_strength': EXPERT_CONFIDENCE_HIGH,
                    'method': EXPERT_METHOD_OPTICAL_VETO_PROPER_MOTION,
                    'p_value': None,
                    'reasoning': f"VETO ÓPTICO PM: {metric_used} = {effective_sigma:.1f}σ > {EXPERT_PM_EXTREME_SIGMA_THRESHOLD}σ - proper motion difference too extreme for orbital motion"
                }
            else:
                # Sigma is below veto threshold - reclassify as ambiguous instead
                metric_used = "Δμ_orbit" if delta_mu_sigma is not None else "PM 2D"
                log.info(f"PM veto SUPPRESSED: {metric_used} = {effective_sigma:.1f}σ < {EXPERT_PM_EXTREME_SIGMA_THRESHOLD}σ threshold")
                return {
                    'label': PhysicalityLabel.AMBIGUOUS,
                    'evidence_strength': EXPERT_CONFIDENCE_MEDIUM,
                    'method': EXPERT_METHOD_AMBIGUOUS_FALLBACK,
                    'p_value': None,
                    'reasoning': f"Moderate {metric_used} ({effective_sigma:.1f}σ) - insufficient for optical veto, could be orbital motion"
                }
        
        # VETO ÓPTICO #3 (DOWNGRADED): Astrometría Inconsistente
        # Now AFTER Δμ_orbit check and only if both chi² AND PM fail
        # CRITICAL: Suppress this veto if Δμ_orbit < 12σ (orbital motion possible)
        if (astrometric_evidence.passed == False and 
            astrometric_evidence.priority == EvidencePriority.HIGH and
            ruwe_evidence.passed == True and
            pm_evidence.passed == False):
            
            # Check Δμ_orbit before applying veto
            delta_mu_sigma = pm_evidence.details.get('delta_mu_orbit_sigma')
            pm_sigma_2d = pm_evidence.details.get('sigma_difference', float('inf'))
            effective_sigma = delta_mu_sigma if delta_mu_sigma is not None else pm_sigma_2d
            
            # Only veto if Δμ_orbit truly exceeds threshold
            if effective_sigma > EXPERT_PM_EXTREME_SIGMA_THRESHOLD:
                metric_used = "Δμ_orbit" if delta_mu_sigma is not None else "PM 2D (FALLBACK)"
                return {
                    'label': PhysicalityLabel.LIKELY_OPTICAL,
                    'evidence_strength': EXPERT_CONFIDENCE_MEDIUM,  # Reduced from HIGH
                    'method': EXPERT_METHOD_OPTICAL_VETO_ASTROMETRY,
                    'p_value': astrometric_evidence.details.get('p_value'),
                    'reasoning': f"VETO ÓPTICO ASTROMETRY: {astrometric_evidence.reasoning} with good RUWE - both chi-squared and PM fail ({metric_used} = {effective_sigma:.1f}σ > {EXPERT_PM_EXTREME_SIGMA_THRESHOLD}σ)"
                }
            else:
                # Suppress veto - Δμ_orbit below threshold suggests possible orbital motion
                metric_used = "Δμ_orbit" if delta_mu_sigma is not None else "PM 2D"
                log.info(f"Astrometry veto SUPPRESSED: {metric_used} = {effective_sigma:.1f}σ < {EXPERT_PM_EXTREME_SIGMA_THRESHOLD}σ threshold")
                # Fall through to evaluate other evidence
        
        # CASO DE ALTA CONFIANZA FÍSICA: Toda la Evidencia Apoya
        if (astrometric_evidence.passed == True and
            ruwe_evidence.passed == True and
            parallax_evidence.passed != False and
            pm_evidence.passed != False):
            
            return {
                'label': PhysicalityLabel.LIKELY_PHYSICAL,
                'evidence_strength': EXPERT_CONFIDENCE_HIGH,
                'method': EXPERT_METHOD_PHYSICAL_HIGH_CONFIDENCE,
                'p_value': astrometric_evidence.details.get('p_value'),
                'reasoning': f"HIGH CONFIDENCE PHYSICAL: {astrometric_evidence.reasoning} with good data quality and no contradictory evidence"
            }
        
        # MANEJO DE RUWE ALTO: Datos de Mala Calidad
        if ruwe_evidence.passed == False:
            if pm_evidence.passed == True:
                return {
                    'label': PhysicalityLabel.LIKELY_PHYSICAL,
                    'evidence_strength': EXPERT_CONFIDENCE_MEDIUM,
                    'method': EXPERT_METHOD_PHYSICAL_BAD_RUWE,
                    'p_value': None,
                    'reasoning': f"BAD RUWE PHYSICAL: {pm_evidence.reasoning} despite poor RUWE - proper motion alignment is most reliable indicator"
                }
            else:
                return {
                    'label': PhysicalityLabel.AMBIGUOUS,
                    'evidence_strength': EXPERT_CONFIDENCE_LOW,
                    'method': EXPERT_METHOD_AMBIGUOUS_FALLBACK,
                    'p_value': None,
                    'reasoning': f"BAD RUWE AMBIGUOUS: {ruwe_evidence.reasoning} and inconsistent proper motion - cannot make reliable determination"
                }
        
        # MANEJO DE MOVIMIENTO ORBITAL: Paralaje OK, PM no
        if (parallax_evidence.passed == True and
            pm_evidence.passed == False and
            ruwe_evidence.passed == True):
            
            return {
                'label': PhysicalityLabel.LIKELY_PHYSICAL,
                'evidence_strength': EXPERT_CONFIDENCE_MEDIUM,
                'method': EXPERT_METHOD_PHYSICAL_PM_ORBITAL,
                'p_value': None,
                'reasoning': f"ORBITAL MOTION PHYSICAL: {parallax_evidence.reasoning} but {pm_evidence.reasoning} - likely orbital motion detectable"
            }
        
        # FALLBACK: Cualquier otro caso
        physical_count = len([e for e in evidence_chain if e.passed == True])
        optical_count = len([e for e in evidence_chain if e.passed == False])
        
        return {
            'label': PhysicalityLabel.AMBIGUOUS,
            'evidence_strength': EXPERT_CONFIDENCE_LOW,
            'method': EXPERT_METHOD_AMBIGUOUS_FALLBACK,
            'p_value': None,
            'reasoning': f"AMBIGUOUS FALLBACK: Mixed evidence - {physical_count} criteria support physical, {optical_count} oppose - no clear pattern"
        }

    def _check_correlation_quality(self, star1: Dict[str, Any], star2: Dict[str, Any]) -> Optional[str]:
        """
        Check for problematic correlations that can invalidate chi-squared tests.
        
        Strong correlations between parallax and proper motion errors can lead to
        inflated chi-squared values that don't reflect true statistical significance.
        
        Returns:
            Warning message if correlations are problematic, None otherwise
        """
        try:
            warnings = []
            
            # Check correlation between parallax and proper motion for each star
            for star_name, star in [("primary", star1), ("secondary", star2)]:
                # Get correlation coefficients
                parallax_pmra_corr = star.get('parallax_pmra_corr')
                parallax_pmdec_corr = star.get('parallax_pmdec_corr')
                pmra_pmdec_corr = star.get('pmra_pmdec_corr')
                
                # Check for extreme correlations (>0.9 absolute value)
                extreme_corr_threshold = 0.9
                
                if parallax_pmra_corr is not None and abs(parallax_pmra_corr) > extreme_corr_threshold:
                    warnings.append(f"{star_name}: extreme parallax-pmRA correlation ({parallax_pmra_corr:.3f})")
                
                if parallax_pmdec_corr is not None and abs(parallax_pmdec_corr) > extreme_corr_threshold:
                    warnings.append(f"{star_name}: extreme parallax-pmDEC correlation ({parallax_pmdec_corr:.3f})")
                
                if pmra_pmdec_corr is not None and abs(pmra_pmdec_corr) > extreme_corr_threshold:
                    warnings.append(f"{star_name}: extreme pmRA-pmDEC correlation ({pmra_pmdec_corr:.3f})")
            
            # Return combined warning if any found
            if warnings:
                return "; ".join(warnings) + " - chi-squared tests may be unreliable"
            
            return None
            
        except Exception as e:
            log.debug(f"Error checking correlation quality: {e}")
            return None

    def _calculate_chi2_3d_corrected(self, star1: Dict[str, Any], star2: Dict[str, Any]) -> Optional[Tuple[float, int]]:
        """
        Calculate 3D chi-squared with full covariance matrix and RUWE correction.
        
        This implementation uses the proper matrix formulation: 
        χ² = Δp^T @ C_total^(-1) @ Δp
        
        Where C_total is the combined covariance matrix including correlations
        and RUWE-corrected errors.
        """
        try:
            # Check for required parameters
            required_keys = ['parallax', 'pmra', 'pmdec']
            for key in required_keys:
                if star1.get(key) is None or star2.get(key) is None:
                    return None
            
            # Get parameter vectors
            params1 = np.array([star1['parallax'], star1['pmra'], star1['pmdec']])
            params2 = np.array([star2['parallax'], star2['pmra'], star2['pmdec']])
            
            # Check for missing values
            if np.any(np.isnan(params1)) or np.any(np.isnan(params2)):
                return None
            
            # Build covariance matrices using the improved function from gaia_utils
            from ..data.gaia_utils import build_covariance_matrix
            
            C1 = build_covariance_matrix(star1, dimensions=3)
            C2 = build_covariance_matrix(star2, dimensions=3)
            
            if C1 is None or C2 is None:
                log.debug("Cannot build covariance matrices - falling back to diagonal approximation")
                return self._calculate_chi2_3d_diagonal_fallback(star1, star2)
            
            # Combined covariance matrix: C_total = C1 + C2
            C_total = C1 + C2
            
            # Check for positive definiteness
            try:
                eigenvals = np.linalg.eigvals(C_total)
                if np.any(eigenvals <= 0):
                    log.warning("Non-positive definite covariance matrix - using diagonal fallback")
                    return self._calculate_chi2_3d_diagonal_fallback(star1, star2)
            except np.linalg.LinAlgError:
                log.warning("Cannot compute eigenvalues - using diagonal fallback")
                return self._calculate_chi2_3d_diagonal_fallback(star1, star2)
            
            # Parameter difference vector
            delta_params = params1 - params2
            
            # Proper matrix chi-squared: χ² = Δp^T @ C_total^(-1) @ Δp
            try:
                C_total_inv = np.linalg.inv(C_total)
                chi2_total = delta_params.T @ C_total_inv @ delta_params
            except np.linalg.LinAlgError:
                log.warning("Cannot invert covariance matrix - using pseudoinverse")
                try:
                    C_total_inv = np.linalg.pinv(C_total)
                    chi2_total = delta_params.T @ C_total_inv @ delta_params
                except:
                    log.warning("Matrix operations failed - using diagonal fallback")
                    return self._calculate_chi2_3d_diagonal_fallback(star1, star2)
            
            dof = 3  # parallax + pmra + pmdec
            
            log.debug(f"3D χ² calculation (full covariance): χ²={chi2_total:.3f}, dof={dof}")
            
            return float(chi2_total), dof
            
        except Exception as e:
            log.warning(f"Failed to calculate 3D chi-squared with full covariance: {e}")
            return self._calculate_chi2_3d_diagonal_fallback(star1, star2)

    def _calculate_chi2_3d_diagonal_fallback(self, star1: Dict[str, Any], star2: Dict[str, Any]) -> Optional[Tuple[float, int]]:
        """
        Fallback diagonal chi-squared calculation for when covariance matrices fail.
        
        This is the simpler diagonal approximation that ignores correlations.
        Used only when the full matrix approach fails.
        """
        try:
            # Import the error correction functions
            from ..data.gaia_utils import correct_gaia_error
            
            # Get parameter values
            params1 = np.array([star1['parallax'], star1['pmra'], star1['pmdec']])
            params2 = np.array([star2['parallax'], star2['pmra'], star2['pmdec']])
            
            # Get RUWE-corrected errors
            errors1 = np.array([
                correct_gaia_error(star1['parallax_error'], star1.get('ruwe'), 'parallax'),
                correct_gaia_error(star1['pmra_error'], star1.get('ruwe'), 'pmra'),
                correct_gaia_error(star1['pmdec_error'], star1.get('ruwe'), 'pmdec')
            ])
            errors2 = np.array([
                correct_gaia_error(star2['parallax_error'], star2.get('ruwe'), 'parallax'),
                correct_gaia_error(star2['pmra_error'], star2.get('ruwe'), 'pmra'),
                correct_gaia_error(star2['pmdec_error'], star2.get('ruwe'), 'pmdec')
            ])
            
            # Calculate differences and combined errors
            param_diff = params1 - params2
            combined_errors = np.sqrt(errors1**2 + errors2**2)
            
            # Check for valid errors
            if np.any(combined_errors <= 0):
                log.warning("Zero or negative combined errors in diagonal chi-squared calculation")
                return None
            
            # Calculate diagonal chi-squared
            chi2_contributions = (param_diff / combined_errors) ** 2
            chi2_total = np.sum(chi2_contributions)
            dof = 3
            
            log.debug(f"3D χ² calculation (diagonal fallback): χ²={chi2_total:.3f}, dof={dof}")
            
            return float(chi2_total), dof
            
        except Exception as e:
            log.warning(f"Failed diagonal chi-squared calculation: {e}")
            return None


# Create a convenience instance for backward compatibility
PhysicalityDecisionTree = ExpertHierarchicalValidator
