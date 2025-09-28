from astroquery.gaia import Gaia
import numpy as np
from scipy.stats import chi2
from typing import Tuple, Optional, Dict, Any
from enum import Enum
import asyncio
import logging
import functools
from datetime import datetime
from astropy import units as u
from astropy.coordinates import SkyCoord

# Import configuration constants
from ..config import (
    DEFAULT_GAIA_TABLE, DEFAULT_GAIA_SEARCH_RADIUS_ARCSEC, DEFAULT_GAIA_MAG_LIMIT,
    DEFAULT_GAIA_MAX_ROWS, DEFAULT_PHYSICAL_P_VALUE_THRESHOLD, 
    DEFAULT_AMBIGUOUS_P_VALUE_THRESHOLD,
    GAIA_MAX_RETRY_ATTEMPTS, GAIA_RETRY_DELAY_SECONDS,
    MIN_PARALLAX_SIGNIFICANCE,
    GAIA_MAX_RUWE, GAIA_DEFAULT_CORRELATION_MISSING,
    GAIA_MATCHING_MAG_TOLERANCE, GAIA_MATCHING_SPATIAL_PENALTY_FACTOR,
    GAIA_MATCHING_MAX_SPATIAL_PENALTY, GAIA_MATCHING_MIN_PARALLAX_SIGNIFICANCE,
    QUALITY_SCORE_RUWE_WEIGHT, QUALITY_SCORE_SIGNIFICANCE_WEIGHT,
    QUALITY_SCORE_MAGNITUDE_WEIGHT, QUALITY_SCORE_RUWE_THRESHOLD,
    QUALITY_SCORE_SIGNIFICANCE_NORMALIZATION, QUALITY_SCORE_MAG_REFERENCE,
    GAIA_SEARCH_RADIUS_MULTIPLIER_SECOND, GAIA_SEARCH_RADIUS_MULTIPLIER_THIRD,
    GAIA_WIDE_BINARY_SEARCH_RADIUS_ARCSEC, GAIA_MIN_SOURCES_REQUIRED,
    GAIA_PARALLAX_MIN_SIGNIFICANCE, GAIA_PARALLAX_NORMALIZATION_FACTOR,
    GAIA_MAGNITUDE_NORMALIZATION_LIMIT, GAIA_RUWE_PERMISSIVE_MULTIPLIER,
    GAIA_RUWE_QUERY_MULTIPLIER, GAIA_PARALLAX_SIGNIFICANCE_DIVISOR,
    GAIA_MAG_LIMIT_BUFFER, GAIA_WDS_SEPARATION_TOLERANCE_FRACTION,
    # RUWE correction configuration
    RUWE_CORRECTION_ENABLED, RUWE_CORRECTION_APPLY_TO_ALL_DIMENSIONS
)

# Import RUWE error correction function from gaia_utils
from ..data.gaia_utils import (
    build_covariance_matrix,
    get_gaia_parallax_error_safe,
    get_gaia_pmra_error_safe,
    get_gaia_pmdec_error_safe,
    assess_gaia_data_quality
)

# Import decision tree for advanced validation
from ..analyzer.decision_tree import ExpertHierarchicalValidator

# Import source types and enums
from ..data.source import PhysicalityValidator, PhysicalityLabel, ValidationMethod, PhysicalityAssessment

# Import exceptions
from ..exceptions import (
    PhysicalityValidationError, ParallaxDataUnavailableError, 
    GaiaQueryError, InsufficientAstrometricDataError
)

log = logging.getLogger(__name__)

class GaiaValidator(PhysicalityValidator):
    """
    Validator for physical binary systems using Gaia data.

    This class implements an adaptive chi-squared (χ²) testing strategy.
    It attempts to use the most complete astrometric data available for a pair
    of stars (3D: parallax + proper motion, 2D: proper motion only, 1D: parallax only)
    to assess physicality of binary systems.
    """
    
    def __init__(self,
                 gaia_table: str = DEFAULT_GAIA_TABLE,
                 default_search_radius_arcsec: float = DEFAULT_GAIA_SEARCH_RADIUS_ARCSEC,
                 physical_p_value_threshold: float = DEFAULT_PHYSICAL_P_VALUE_THRESHOLD,
                 ambiguous_p_value_threshold: float = DEFAULT_AMBIGUOUS_P_VALUE_THRESHOLD,
                 mag_limit: float = DEFAULT_GAIA_MAG_LIMIT,
                 max_sources: int = DEFAULT_GAIA_MAX_ROWS,
                 gaia_client=None):
        """
        Initializes the Gaia validator with configuration parameters.

        Args:
            gaia_table: The Gaia data release table to query
            default_search_radius_arcsec: Default search radius for Gaia queries
            physical_p_value_threshold: The p-value above which a pair is considered
                                        'Likely Physical'. Default: 0.045 (4.5%)
            ambiguous_p_value_threshold: The p-value above which a pair is 'Ambiguous'.
                                         Below this: 'Likely Optical'. Default: 0.005 (0.5%)
            mag_limit: G-band magnitude limit for Gaia queries
            max_sources: Maximum number of sources to retrieve
            gaia_client: Optional Gaia client for dependency injection (testing)

        Raises:
            ValueError: If thresholds are not in valid order
        """
        self.gaia_table = gaia_table
        self.default_search_radius_arcsec = default_search_radius_arcsec
        self.mag_limit = mag_limit
        self.max_sources = max_sources
        self.gaia = gaia_client or Gaia

        self.physical_threshold = physical_p_value_threshold
        self.ambiguous_threshold = ambiguous_p_value_threshold
        
        # Always initialize Expert Hierarchical Validator
        self.decision_tree = ExpertHierarchicalValidator(
            physical_threshold=physical_p_value_threshold,
            ambiguous_threshold=ambiguous_p_value_threshold,
            enable_ruwe_correction=True
        )
        
        # Validate thresholds
        if self.physical_threshold <= self.ambiguous_threshold:
            raise ValueError(
                f"physical_p_value_threshold ({self.physical_threshold}) must be greater than "
                f"ambiguous_p_value_threshold ({self.ambiguous_threshold}). "
                "A p-value must cross the 'ambiguous' threshold before it can be considered 'physical'."
            )
            
        if not (0.0 < self.physical_threshold <= 1.0):
            raise ValueError(f"physical_p_value_threshold must be in (0, 1], got {self.physical_threshold}")
            
        if not (0.0 < self.ambiguous_threshold <= 1.0):
            raise ValueError(f"ambiguous_p_value_threshold must be in (0, 1], got {self.ambiguous_threshold}")
            
        # Initialize query cache for Gaia queries
        self._query_cache = {}
            
        log.info(f"GaiaValidator initialized: table={gaia_table}, "
                f"thresholds=({self.physical_threshold:.3f}, {self.ambiguous_threshold:.3f})")
    
    
    async def validate_physicality(self,
                                 wds_summary: Dict[str, Any],
                                 search_radius_arcsec: Optional[float] = None) -> PhysicalityAssessment:
        """
        Validates if a binary system is physically bound using Gaia data.
        Now intelligently uses Gaia source IDs when available to bypass coordinate search.
        
        Args:
            wds_summary: WDS summary data containing coordinates, magnitudes, and optionally Gaia IDs
            search_radius_arcsec: Optional override for search radius
            
        Returns:
            PhysicalityAssessment object
            
        Raises:
            PhysicalityValidationError: When validation cannot be completed
            InsufficientAstrometricDataError: When insufficient data is available
            GaiaQueryError: When Gaia queries fail
        """
        # Check if we have Gaia source IDs available - much more reliable!
        gaia_source_ids = self._extract_gaia_source_ids(wds_summary)
        
        if gaia_source_ids and len(gaia_source_ids) >= 2:
            log.debug("Using direct Gaia source IDs: %s", list(gaia_source_ids.values()))
            try:
                gaia_results = await self._query_gaia_by_source_ids_async(gaia_source_ids)
                if gaia_results and len(gaia_results) >= GAIA_MIN_SOURCES_REQUIRED:
                    wds_magnitudes = (wds_summary.get('mag_pri'), wds_summary.get('mag_sec'))
                    result = self._validate_physicality_sync(gaia_results, wds_magnitudes, wds_summary)
                    return self._create_final_assessment(result, gaia_results, wds_magnitudes, None, 
                                                       direct_source_query=True)
            except Exception as e:
                log.warning(f"Direct Gaia source ID query failed: {e}. Falling back to coordinate search.")
        
        # Fallback to coordinate-based search
        return await self._validate_by_coordinates(wds_summary, search_radius_arcsec)
    
    def _extract_gaia_source_ids(self, wds_summary: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Extract Gaia source IDs from WDS summary if available."""
        log.debug("Extracting Gaia source IDs from WDS metadata")
        log.debug("WDS summary keys: %s", list(wds_summary.keys()))
        
        gaia_ids = {}
        
        # Check for direct Gaia source IDs in the summary
        gaia_source_ids_field = wds_summary.get('gaia_source_ids')
        log.debug("Raw gaia_source_ids field: %s (type=%s)", gaia_source_ids_field, type(gaia_source_ids_field))
        
        if gaia_source_ids_field:
            import json
            try:
                if isinstance(gaia_source_ids_field, str):
                    parsed_ids = json.loads(gaia_source_ids_field)
                else:
                    parsed_ids = gaia_source_ids_field
                    
                log.debug("Parsed gaia_source_ids payload: %s", parsed_ids)
                
                if isinstance(parsed_ids, dict):
                    # Extract component A and B (ignore 'component' field which seems to be a different star)
                    if 'A' in parsed_ids and 'B' in parsed_ids:
                        gaia_ids['A'] = str(parsed_ids['A'])
                        gaia_ids['B'] = str(parsed_ids['B'])
                        log.debug("Extracted Gaia IDs: %s", gaia_ids)
                        return gaia_ids
                    else:
                        log.debug("Missing component A or B in gaia_source_ids payload")
                else:
                    log.debug("gaia_source_ids payload is not a dictionary; ignoring")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                log.debug("Failed to parse gaia_source_ids: %s", e)
        else:
            log.debug("No gaia_source_ids field present")
        
        # Check for Gaia IDs in component names
        name_primary = wds_summary.get('name_primary', '')
        name_secondary = wds_summary.get('name_secondary', '')
        
        if name_primary and 'Gaia DR3' in name_primary:
            gaia_id = name_primary.split()[-1]
            if gaia_id.isdigit() and len(gaia_id) >= 18:
                gaia_ids['A'] = gaia_id
        
        if name_secondary and 'Gaia DR3' in name_secondary:
            gaia_id = name_secondary.split()[-1]
            if gaia_id.isdigit() and len(gaia_id) >= 18:
                gaia_ids['B'] = gaia_id
        
        return gaia_ids if len(gaia_ids) >= 2 else None
    
    async def _query_gaia_by_source_ids_async(self, gaia_source_ids: Dict[str, str]):
        """Query Gaia directly by source IDs - much more reliable than coordinate search."""
        from astroquery.gaia import Gaia
        
        source_ids = list(gaia_source_ids.values())
        source_ids_str = ','.join(source_ids)
        
        query = f"""
        SELECT source_id, ra, dec, parallax, parallax_error,
               pmra, pmra_error, pmdec, pmdec_error,
               ra_error, dec_error, 
               ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr,
               dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr,
               parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr,
               phot_g_mean_mag, bp_rp, ruwe
        FROM {self.gaia_table} 
        WHERE source_id IN ({source_ids_str})
        ORDER BY source_id
        """
        
        log.debug(f"Querying Gaia by source IDs: {source_ids}")
        job = Gaia.launch_job_async(query)
        results = job.get_results()
        
        if len(results) < 2:
            raise InsufficientAstrometricDataError(f"Only found {len(results)} of {len(source_ids)} requested Gaia sources")
        
        log.info(f"Successfully retrieved {len(results)} Gaia sources by ID")
        return results
    
    async def _validate_by_coordinates(self, wds_summary: Dict[str, Any], 
                                     search_radius_arcsec: Optional[float]) -> PhysicalityAssessment:
        """Fallback validation using coordinate-based search with improved wide binary support."""
        # Extract coordinates and magnitudes from WDS summary
        ra_deg = wds_summary.get('ra_deg')
        dec_deg = wds_summary.get('dec_deg')
        mag_pri = wds_summary.get('mag_pri')
        mag_sec = wds_summary.get('mag_sec')
        
        if ra_deg is None or dec_deg is None:
            raise PhysicalityValidationError("Missing coordinates in WDS summary")
        
        final_radius = search_radius_arcsec if search_radius_arcsec is not None else self.default_search_radius_arcsec
        wds_magnitudes = (mag_pri, mag_sec)
        
        # Improved search strategy for wide binaries - use much larger radii
        search_radii = [
            final_radius, 
            final_radius * GAIA_SEARCH_RADIUS_MULTIPLIER_SECOND,
            final_radius * GAIA_SEARCH_RADIUS_MULTIPLIER_THIRD,
            GAIA_WIDE_BINARY_SEARCH_RADIUS_ARCSEC  # Try very wide search as last resort
        ]
        
        for attempt, radius in enumerate(search_radii):
            try:
                gaia_results = await self._query_gaia_for_pair_async(ra_deg, dec_deg, radius)
                
                if gaia_results is not None and len(gaia_results) >= GAIA_MIN_SOURCES_REQUIRED:
                    result = self._validate_physicality_sync(gaia_results, wds_magnitudes, wds_summary)
                    return self._create_final_assessment(result, gaia_results, wds_magnitudes, radius, 
                                                       direct_source_query=False)
                
                if attempt < len(search_radii) - 1:
                    log.debug(f"Search attempt {attempt + 1} with radius {radius:.1f}\" found {len(gaia_results) if gaia_results else 0} sources, trying larger radius")
                    
            except Exception as e:
                if attempt == len(search_radii) - 1:  # Last attempt
                    raise PhysicalityValidationError(f"Gaia validation failed after multiple attempts: {e}") from e
                else:
                    log.debug(f"Gaia query attempt {attempt + 1} failed: {e}, trying larger radius")
                    continue
        
        # If all attempts failed
        radii_str = ", ".join([f"{r:.1f}\"" for r in search_radii])
        raise InsufficientAstrometricDataError(
            f"Insufficient Gaia sources found after trying radii: {radii_str} (need ≥{GAIA_MIN_SOURCES_REQUIRED})"
        )
    
    def _create_assessment(self, label: PhysicalityLabel, search_radius_arcsec: Optional[float] = None, 
                          p_value: Optional[float] = None, method: ValidationMethod = None,
                          gaia_primary: Optional[str] = None, gaia_secondary: Optional[str] = None,
                          decision_confidence: Optional[float] = None) -> PhysicalityAssessment:
        """
        Create a PhysicalityAssessment object with correct confidence interpretation.
        
        Args:
            decision_confidence: Direct confidence in the classification (0-1), independent of p_value
        """
        # Calculate decision confidence based on the type of evidence
        if decision_confidence is None:
            if p_value is not None:
                # For chi-squared tests, higher p-value indicates stronger evidence for physical pair
                if label == PhysicalityLabel.LIKELY_PHYSICAL:
                    decision_confidence = min(p_value / self.physical_threshold, 1.0)
                elif label == PhysicalityLabel.LIKELY_OPTICAL:
                    decision_confidence = min((self.ambiguous_threshold - p_value) / self.ambiguous_threshold, 1.0) if p_value < self.ambiguous_threshold else 0.5
                else:  # AMBIGUOUS
                    decision_confidence = 0.5
            else:
                # For rule-based decisions, use moderate confidence
                decision_confidence = 0.7 if label != PhysicalityLabel.AMBIGUOUS else 0.3

        return {
            'label': label,
            'confidence': decision_confidence,  # Now correctly represents confidence in the classification
            'p_value': p_value,  # Statistical p-value from chi-squared test (may be None for rule-based decisions)
            'method': method,
            'parallax_consistency': None,
            'proper_motion_consistency': None,
            'gaia_source_id_primary': gaia_primary,
            'gaia_source_id_secondary': gaia_secondary,
            'validation_date': datetime.now().isoformat(),
            'search_radius_arcsec': search_radius_arcsec or self.default_search_radius_arcsec,
            'significance_thresholds': {
                'physical': self.physical_threshold,
                'ambiguous': self.ambiguous_threshold
            },
            'retry_attempts': 1
        }
    
    def _create_final_assessment(self, result: Dict[str, Any], gaia_results, wds_magnitudes, 
                              search_radius_arcsec: Optional[float], direct_source_query: bool = False) -> PhysicalityAssessment:
        """Convert validation result to final PhysicalityAssessment."""
        primary_gaia, secondary_gaia = self._identify_components_by_mag(gaia_results, wds_magnitudes)
        gaia_primary_id = primary_gaia.get('source_id') if primary_gaia else None
        gaia_secondary_id = secondary_gaia.get('source_id') if secondary_gaia else None
        
        # --- Provisional Calibration Logic (2025-09) ---
        # If Δμ_orbit significance is available, adjust label according to new orbit excess sigma bands
        # before constructing assessment. This provides an orthogonal decision axis to the p-value.
        delta_mu_sig = result.get('delta_mu_orbit_significance')
        if delta_mu_sig is not None and isinstance(delta_mu_sig, (int, float)) and np.isfinite(delta_mu_sig):
            try:
                from astrakairos.config import (
                    ORBIT_EXCESS_SIGMA_PHYSICAL_MAX, ORBIT_EXCESS_SIGMA_AMBIGUOUS_MAX
                )
                # Only override if original method not an explicit optical veto to avoid undoing strong expert decisions
                if result.get('label') != PhysicalityLabel.LIKELY_OPTICAL:
                    if delta_mu_sig < ORBIT_EXCESS_SIGMA_PHYSICAL_MAX and result.get('label') != PhysicalityLabel.LIKELY_OPTICAL:
                        result['label'] = PhysicalityLabel.LIKELY_PHYSICAL
                        result.setdefault('expert_method', 'expert_delta_mu_consistent')
                    elif delta_mu_sig >= ORBIT_EXCESS_SIGMA_AMBIGUOUS_MAX and result.get('label') == PhysicalityLabel.LIKELY_PHYSICAL:
                        # High orbital excess can indicate mismatched co-motion -> downgrade to ambiguous/optical
                        result['label'] = PhysicalityLabel.LIKELY_OPTICAL
                        result.setdefault('expert_method', 'expert_delta_mu_excess')
                    elif ORBIT_EXCESS_SIGMA_PHYSICAL_MAX <= delta_mu_sig < ORBIT_EXCESS_SIGMA_AMBIGUOUS_MAX and result.get('label') == PhysicalityLabel.LIKELY_PHYSICAL:
                        # Transitional: if initially physical only by marginal p-value, degrade to ambiguous
                        if result.get('p_value') is not None and result['p_value'] <= self.physical_threshold:
                            result['label'] = PhysicalityLabel.AMBIGUOUS
                            result.setdefault('expert_method', 'expert_delta_mu_transitional')
            except Exception:
                # Fail quietly if constants not found (should not happen)
                pass

        assessment = self._create_assessment(
            label=result['label'],
            p_value=result['p_value'],
            method=result['method'],
            gaia_primary=gaia_primary_id,
            gaia_secondary=gaia_secondary_id,
            search_radius_arcsec=search_radius_arcsec,
            decision_confidence=result.get('expert_confidence')  # Use expert confidence if available
        )
        
        # Add information about query method
        if direct_source_query:
            assessment['query_method'] = 'direct_source_id'
            assessment['note'] = 'Used direct Gaia source IDs for high precision validation'
        else:
            assessment['query_method'] = 'coordinate_search'
        
        # Add method type classification with enum-safe handling
        def _normalize_method_identifier(value: Any) -> str:
            if isinstance(value, ValidationMethod):
                return value.value
            if isinstance(value, Enum):
                enum_val = getattr(value, 'value', None)
                return enum_val if isinstance(enum_val, str) else str(value)
            return '' if value is None else str(value)

        method_raw = result.get('method')
        expert_method_raw = result.get('expert_method')

        method_str = _normalize_method_identifier(method_raw)
        expert_method_str = _normalize_method_identifier(expert_method_raw)

        if expert_method_str:
            assessment['method_type'] = 'expert_rule'
            assessment['expert_method'] = expert_method_str
        elif isinstance(method_raw, ValidationMethod) and method_raw in {
            ValidationMethod.GAIA_3D_PARALLAX_PM,
            ValidationMethod.PROPER_MOTION_ONLY,
            ValidationMethod.GAIA_PARALLAX_ONLY,
        }:
            assessment['method_type'] = 'chi2_statistical'
        elif isinstance(method_raw, ValidationMethod) and method_raw == ValidationMethod.EXPERT_EL_BADRY:
            assessment['method_type'] = 'expert_rule'
        elif isinstance(method_raw, ValidationMethod) and method_raw == ValidationMethod.STATISTICAL_ANALYSIS:
            assessment['method_type'] = 'statistical_analysis'
        elif method_str.lower().startswith('expert_'):
            assessment['method_type'] = 'expert_rule'
        elif 'chi2' in method_str.lower():
            assessment['method_type'] = 'chi2_statistical'
        elif method_str:
            assessment['method_type'] = 'statistical'
        else:
            assessment['method_type'] = 'unknown'

        for key in (
            'delta_mu_orbit',
            'delta_mu_orbit_error',
            'delta_mu_orbit_significance',
            'separation_arcsec',
            'position_angle_deg',
            'proper_motion_difference'
        ):
            if key in result:
                assessment[key] = result[key]
        
        return assessment
    
    def _validate_physicality_sync(self,
                                  gaia_results,
                                  wds_mags: Tuple[Optional[float], Optional[float]],
                                  wds_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Simplified physicality validation that delegates to the appropriate validator.
        
        This function is now a clean orchestrator that:
        1. Identifies the correct binary components 
        2. Chooses between traditional p-value method and Expert Hierarchical Validator
        3. Returns structured results
        
        Args:
            gaia_results: Pre-fetched Gaia query results
            wds_mags: WDS magnitudes for component matching
        """
        # Filter by astrometric quality first
        quality_filtered = [star for star in gaia_results if self._validate_astrometric_quality(star)]
        
        if len(quality_filtered) < GAIA_MIN_SOURCES_REQUIRED:
            log.warning(f"Only {len(quality_filtered)} quality sources after filtering from {len(gaia_results)} total")
            raise InsufficientAstrometricDataError(f"Only {len(quality_filtered)} quality sources available")
        
        # Identify components
        primary_gaia, secondary_gaia = self._identify_components_by_mag(quality_filtered, wds_mags)
        if primary_gaia is None or secondary_gaia is None:
            if len(quality_filtered) < GAIA_MIN_SOURCES_REQUIRED:
                raise InsufficientAstrometricDataError("Insufficient quality sources for component identification")
            else:
                raise InsufficientAstrometricDataError("Cannot identify binary components from available sources")
        
        # Verify separation consistency with catalog expectations when available
        if wds_summary and not self._verify_separation_consistency(primary_gaia, secondary_gaia, wds_summary):
            log.warning("Selected Gaia pair has inconsistent separation with WDS catalog; rejecting match")
            raise InsufficientAstrometricDataError("Gaia pair separation inconsistent with WDS catalog")

        # Primary assessment via expert hierarchical reasoning
        expert_result = self._validate_with_expert_tree(primary_gaia, secondary_gaia)

        # Secondary statistical assessment using classical chi-squared tests
        statistical_result = self._calculate_statistical_consistency(primary_gaia, secondary_gaia)

        # Compute El-Badry Δμ_orbit metrics for reporting
        el_badry_metrics = self._calculate_el_badry_metrics(primary_gaia, secondary_gaia)
        if el_badry_metrics:
            expert_result.update(el_badry_metrics)
            if statistical_result:
                statistical_result.update(el_badry_metrics)

        # If expert reasoning failed or remained ambiguous, fall back to statistical evidence
        expert_failed = expert_result.get('expert_method') == "fallback_error"
        expert_ambiguous = expert_result['label'] == PhysicalityLabel.AMBIGUOUS

        if statistical_result and (expert_failed or expert_ambiguous):
            statistical_result['expert_confidence'] = expert_result.get('expert_confidence', 0.0)
            statistical_result['expert_reasoning'] = expert_result.get('expert_reasoning')
            statistical_result['expert_method'] = expert_result.get('expert_method')
            return statistical_result

        # Otherwise, enrich expert result with statistical context when available
        if statistical_result:
            if expert_result.get('p_value') is None:
                expert_result['p_value'] = statistical_result['p_value']
            expert_result['statistical_method'] = statistical_result['method']
            expert_result['statistical_p_value'] = statistical_result['p_value']
            expert_result['statistical_label'] = statistical_result['label']

        return expert_result

    def _validate_with_expert_tree(self, primary_gaia: Dict, secondary_gaia: Dict) -> Dict[str, Any]:
        """
        Use Expert Hierarchical Validator for physicality assessment.
        
        This is the new, sophisticated approach that uses expert reasoning.
        """
        try:
            # Prepare data for expert validator
            log.debug("🔬 Preparing data for Expert Hierarchical Validator...")
            primary_data = self._prepare_expert_data(primary_gaia)
            secondary_data = self._prepare_expert_data(secondary_gaia)
            
            # Run expert validation
            log.debug("🔬 Running expert validation...")
            expert_result = self.decision_tree.validate_pair(primary_data, secondary_data)
            
            # Log expert decision
            log.info(f"Expert Hierarchical Validator: {expert_result.label.value} "
                    f"(evidence_strength: {expert_result.evidence_strength:.2f}, method: {expert_result.method})")
            log.info(f"Expert reasoning: {expert_result.reasoning}")
            
            # Return expert decision
            return {
                'label': expert_result.label,
                'p_value': expert_result.p_value,  # May be None if unreliable
                'method': ValidationMethod.EXPERT_EL_BADRY,
                'expert_method': expert_result.method,
                'expert_confidence': expert_result.evidence_strength,
                'expert_reasoning': expert_result.reasoning
            }
        except Exception as e:
            log.error(f"Expert Hierarchical Validator failed: {e}")
            # Return AMBIGUOUS instead of falling back to traditional method
            return {
                'label': PhysicalityLabel.AMBIGUOUS,
                'p_value': None,
                'method': ValidationMethod.EXPERT_EL_BADRY,
                'expert_method': "fallback_error",
                'expert_confidence': 0.0,
                'expert_reasoning': f"Expert validator failed: {e}"
            }

    def _prepare_expert_data(self, gaia_star: Dict) -> Dict[str, Any]:
        """
        Prepare Gaia star data for Expert Hierarchical Validator.
        
        Args:
            gaia_star: Gaia query result for one star
            
        Returns:
            Dictionary formatted for expert validator
        """
        # Safely extract values and convert to float
        def safe_float(value, default=0.0):
            try:
                if value is None:
                    return default
                if hasattr(value, 'mask') and np.ma.is_masked(value):
                    return default
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Get base values
        parallax = safe_float(gaia_star.get('parallax', 0.0))
        pmra = safe_float(gaia_star.get('pmra', 0.0))
        pmdec = safe_float(gaia_star.get('pmdec', 0.0))
        ruwe = safe_float(gaia_star.get('ruwe', 1.0), 1.0)
        
        # Get errors using safe functions (these already apply RUWE correction internally)
        parallax_error_raw = safe_float(gaia_star.get('parallax_error', None))
        pmra_error_raw = safe_float(gaia_star.get('pmra_error', None))
        pmdec_error_raw = safe_float(gaia_star.get('pmdec_error', None))
        
        # Use safe error functions that handle RUWE correction internally
        # No additional RUWE correction needed here to avoid double inflation
        parallax_error = get_gaia_parallax_error_safe(parallax_error_raw, ruwe)
        pmra_error = get_gaia_pmra_error_safe(pmra_error_raw, ruwe)
        pmdec_error = get_gaia_pmdec_error_safe(pmdec_error_raw, ruwe)
        
        return {
            'parallax': parallax,
            'parallax_error': parallax_error,
            'pmra': pmra,
            'pmra_error': pmra_error,
            'pmdec': pmdec,
            'pmdec_error': pmdec_error,
            'ruwe': ruwe,
            'source_id': str(gaia_star.get('source_id', 'unknown'))
        }

    def _calculate_el_badry_metrics(self, primary_gaia: Dict, secondary_gaia: Dict) -> Dict[str, float]:
        """Compute Δμ_orbit metrics following El-Badry & Rix (2018)."""
        def get_value(star: Dict, key: str, default=None):
            value = star.get(key, default) if hasattr(star, 'get') else getattr(star, key, default)
            if value is None:
                return default
            if hasattr(value, 'mask') and np.ma.is_masked(value):
                return default
            return value

        try:
            ra_a = float(get_value(primary_gaia, 'ra'))
            dec_a = float(get_value(primary_gaia, 'dec'))
            ra_b = float(get_value(secondary_gaia, 'ra'))
            dec_b = float(get_value(secondary_gaia, 'dec'))

            pmra_a = float(get_value(primary_gaia, 'pmra', 0.0))
            pmdec_a = float(get_value(primary_gaia, 'pmdec', 0.0))
            pmra_b = float(get_value(secondary_gaia, 'pmra', 0.0))
            pmdec_b = float(get_value(secondary_gaia, 'pmdec', 0.0))

            ruwe_a = float(get_value(primary_gaia, 'ruwe', 1.0) or 1.0)
            ruwe_b = float(get_value(secondary_gaia, 'ruwe', 1.0) or 1.0)

            pmra_err_a_raw = get_value(primary_gaia, 'pmra_error')
            pmdec_err_a_raw = get_value(primary_gaia, 'pmdec_error')
            pmra_err_b_raw = get_value(secondary_gaia, 'pmra_error')
            pmdec_err_b_raw = get_value(secondary_gaia, 'pmdec_error')

            pmra_err_a = get_gaia_pmra_error_safe(pmra_err_a_raw, ruwe_a)
            pmdec_err_a = get_gaia_pmdec_error_safe(pmdec_err_a_raw, ruwe_a)
            pmra_err_b = get_gaia_pmra_error_safe(pmra_err_b_raw, ruwe_b)
            pmdec_err_b = get_gaia_pmdec_error_safe(pmdec_err_b_raw, ruwe_b)

            corr_a = float(get_value(primary_gaia, 'pmra_pmdec_corr', 0.0) or 0.0)
            corr_b = float(get_value(secondary_gaia, 'pmra_pmdec_corr', 0.0) or 0.0)

            coord_a = SkyCoord(ra=ra_a * u.deg, dec=dec_a * u.deg)
            coord_b = SkyCoord(ra=ra_b * u.deg, dec=dec_b * u.deg)

            separation = coord_a.separation(coord_b)
            position_angle = coord_a.position_angle(coord_b)

            sep_arcsec = separation.to(u.arcsec).value
            if sep_arcsec == 0:
                return {}

            pa_rad = position_angle.to(u.radian).value
            delta_ra = sep_arcsec * np.sin(pa_rad)
            delta_dec = sep_arcsec * np.cos(pa_rad)
            r_norm = np.hypot(delta_ra, delta_dec)
            if r_norm == 0:
                return {}

            e_r_ra = delta_ra / r_norm
            e_r_dec = delta_dec / r_norm
            e_t_ra = -e_r_dec
            e_t_dec = e_r_ra

            delta_pmra = pmra_b - pmra_a
            delta_pmdec = pmdec_b - pmdec_a
            proper_motion_difference = float(np.hypot(delta_pmra, delta_pmdec))

            delta_mu_orbit = float(delta_pmra * e_t_ra + delta_pmdec * e_t_dec)

            cov_a = np.array([
                [pmra_err_a ** 2, pmra_err_a * pmdec_err_a * corr_a],
                [pmra_err_a * pmdec_err_a * corr_a, pmdec_err_a ** 2]
            ])
            cov_b = np.array([
                [pmra_err_b ** 2, pmra_err_b * pmdec_err_b * corr_b],
                [pmra_err_b * pmdec_err_b * corr_b, pmdec_err_b ** 2]
            ])

            cov_delta = cov_a + cov_b
            e_t = np.array([e_t_ra, e_t_dec])
            variance = float(e_t.T @ cov_delta @ e_t)
            variance = max(variance, 0.0)
            delta_mu_error = float(np.sqrt(variance)) if variance > 0 else float('inf')

            if not np.isfinite(delta_mu_error) or delta_mu_error == 0:
                delta_mu_significance = float('inf') if delta_mu_orbit != 0 else 0.0
            else:
                delta_mu_significance = abs(delta_mu_orbit) / delta_mu_error

            return {
                'delta_mu_orbit': delta_mu_orbit,
                'delta_mu_orbit_error': delta_mu_error,
                'delta_mu_orbit_significance': delta_mu_significance,
                'separation_arcsec': sep_arcsec,
                'position_angle_deg': position_angle.to(u.deg).value,
                'proper_motion_difference': proper_motion_difference
            }
        except Exception as exc:
            log.warning(f"Failed to compute El-Badry Δμ_orbit metrics: {exc}")
            return {}

    def _calculate_statistical_consistency(self, primary_gaia: Dict, secondary_gaia: Dict) -> Optional[Dict[str, Any]]:
        """Run classical chi-squared consistency tests as statistical backup evidence."""
        test_sequence = [
            (self._calculate_chi2_3d, ValidationMethod.GAIA_3D_PARALLAX_PM, 3),
            (self._calculate_chi2_2d, ValidationMethod.PROPER_MOTION_ONLY, 2),
            (self._calculate_chi2_1d, ValidationMethod.GAIA_PARALLAX_ONLY, 1),
        ]

        for calculator, method_enum, dof in test_sequence:
            chi2_result = calculator(primary_gaia, secondary_gaia)
            if chi2_result is None:
                continue

            chi2_val = chi2_result
            p_value = chi2.sf(chi2_val, df=dof)

            if p_value > self.physical_threshold:
                label = PhysicalityLabel.LIKELY_PHYSICAL
            elif p_value > self.ambiguous_threshold:
                label = PhysicalityLabel.AMBIGUOUS
            else:
                label = PhysicalityLabel.LIKELY_OPTICAL

            return {
                'label': label,
                'p_value': p_value,
                'method': method_enum,
                'chi2': chi2_val,
                'degrees_of_freedom': dof,
                'expert_confidence': None,
            }

        return None

    def _calculate_chi2_3d(self, star1: Dict, star2: Dict) -> Optional[float]:
        """3D chi-squared using full covariance (parallax + proper motion)."""
        required = ['parallax', 'pmra', 'pmdec']
        if any(star1.get(k) is None or star2.get(k) is None for k in required):
            return None

        params1 = np.array([float(star1['parallax']), float(star1['pmra']), float(star1['pmdec'])])
        params2 = np.array([float(star2['parallax']), float(star2['pmra']), float(star2['pmdec'])])

        C1 = build_covariance_matrix(star1, dimensions=3)
        C2 = build_covariance_matrix(star2, dimensions=3)
        if C1 is None or C2 is None:
            return self._calculate_chi2_3d_diagonal(star1, star2)

        C_total = C1 + C2
        try:
            C_inv = np.linalg.inv(C_total)
        except np.linalg.LinAlgError:
            return self._calculate_chi2_3d_diagonal(star1, star2)

        delta = params1 - params2
        chi2_val = float(delta.T @ C_inv @ delta)
        return chi2_val

    def _calculate_chi2_3d_diagonal(self, star1: Dict, star2: Dict) -> Optional[float]:
        """Diagonal fallback when full covariance cannot be constructed."""
        try:
            errors1 = np.array([
                get_gaia_parallax_error_safe(star1.get('parallax_error'), star1.get('ruwe')),
                get_gaia_pmra_error_safe(star1.get('pmra_error'), star1.get('ruwe')),
                get_gaia_pmdec_error_safe(star1.get('pmdec_error'), star1.get('ruwe')),
            ])
            errors2 = np.array([
                get_gaia_parallax_error_safe(star2.get('parallax_error'), star2.get('ruwe')),
                get_gaia_pmra_error_safe(star2.get('pmra_error'), star2.get('ruwe')),
                get_gaia_pmdec_error_safe(star2.get('pmdec_error'), star2.get('ruwe')),
            ])

            if np.any(errors1 <= 0) or np.any(errors2 <= 0):
                return None

            params1 = np.array([float(star1['parallax']), float(star1['pmra']), float(star1['pmdec'])])
            params2 = np.array([float(star2['parallax']), float(star2['pmra']), float(star2['pmdec'])])

            combined_errors = np.sqrt(errors1**2 + errors2**2)
            delta = params1 - params2
            chi2_components = (delta / combined_errors) ** 2
            return float(np.sum(chi2_components))
        except Exception:
            return None

    def _calculate_chi2_2d(self, star1: Dict, star2: Dict) -> Optional[float]:
        """2D chi-squared for proper motion alignment."""
        required = ['pmra', 'pmdec']
        if any(star1.get(k) is None or star2.get(k) is None for k in required):
            return None

        params1 = np.array([float(star1['pmra']), float(star1['pmdec'])])
        params2 = np.array([float(star2['pmra']), float(star2['pmdec'])])

        C1 = build_covariance_matrix(star1, dimensions=2)
        C2 = build_covariance_matrix(star2, dimensions=2)
        if C1 is None or C2 is None:
            return self._calculate_chi2_2d_diagonal(star1, star2)

        C_total = C1 + C2
        try:
            C_inv = np.linalg.inv(C_total)
        except np.linalg.LinAlgError:
            return self._calculate_chi2_2d_diagonal(star1, star2)

        delta = params1 - params2
        return float(delta.T @ C_inv @ delta)

    def _calculate_chi2_2d_diagonal(self, star1: Dict, star2: Dict) -> Optional[float]:
        try:
            err1_ra = get_gaia_pmra_error_safe(star1.get('pmra_error'), star1.get('ruwe'))
            err1_dec = get_gaia_pmdec_error_safe(star1.get('pmdec_error'), star1.get('ruwe'))
            err2_ra = get_gaia_pmra_error_safe(star2.get('pmra_error'), star2.get('ruwe'))
            err2_dec = get_gaia_pmdec_error_safe(star2.get('pmdec_error'), star2.get('ruwe'))

            if min(err1_ra, err1_dec, err2_ra, err2_dec) <= 0:
                return None

            delta_ra = float(star1['pmra']) - float(star2['pmra'])
            delta_dec = float(star1['pmdec']) - float(star2['pmdec'])

            combined_ra_err = np.sqrt(err1_ra**2 + err2_ra**2)
            combined_dec_err = np.sqrt(err1_dec**2 + err2_dec**2)

            chi2_ra = (delta_ra / combined_ra_err) ** 2
            chi2_dec = (delta_dec / combined_dec_err) ** 2
            return float(chi2_ra + chi2_dec)
        except Exception:
            return None

    def _calculate_chi2_1d(self, star1: Dict, star2: Dict) -> Optional[float]:
        """1D chi-squared for parallax consistency."""
        if star1.get('parallax') is None or star2.get('parallax') is None:
            return None

        err1 = get_gaia_parallax_error_safe(star1.get('parallax_error'), star1.get('ruwe'))
        err2 = get_gaia_parallax_error_safe(star2.get('parallax_error'), star2.get('ruwe'))
        if err1 <= 0 or err2 <= 0:
            return None

        delta = float(star1['parallax']) - float(star2['parallax'])
        combined_err = np.sqrt(err1**2 + err2**2)
        if combined_err <= 0:
            return None

        return float((delta / combined_err) ** 2)
    
    def _verify_separation_consistency(self, primary_gaia: Dict, secondary_gaia: Dict, wds_summary: Dict[str, Any]) -> bool:
        """Ensure the Gaia-selected pair matches the catalog separation."""
        try:
            wds_sep = wds_summary.get('sep_last')
            if wds_sep is None or wds_sep <= 0:
                log.debug("No WDS separation available for verification; skipping check")
                return True

            gaia_sep = self._calculate_angular_separation(primary_gaia, secondary_gaia)
            sep_diff_fraction = abs(gaia_sep - wds_sep) / wds_sep
            is_consistent = sep_diff_fraction <= GAIA_WDS_SEPARATION_TOLERANCE_FRACTION

            log.debug(
                "Separation verification: Gaia=%.2f\", WDS=%.2f\", fractional diff=%.2f (%s)",
                gaia_sep,
                wds_sep,
                sep_diff_fraction,
                "OK" if is_consistent else "FAIL"
            )

            return is_consistent
        except Exception as exc:
            log.warning("Error verifying separation consistency: %s", exc)
            return True

    def _calculate_angular_separation(self, star1: Dict, star2: Dict) -> float:
        """Calculate angular separation between two Gaia detections in arcseconds."""
        ra1, dec1 = float(star1['ra']), float(star1['dec'])
        ra2, dec2 = float(star2['ra']), float(star2['dec'])

        dra = (ra1 - ra2) * np.cos(np.radians((dec1 + dec2) / 2.0))
        ddec = dec1 - dec2
        separation_deg = np.sqrt(dra**2 + ddec**2)

        return separation_deg * 3600.0

    def _get_params_and_check_validity(self, star: Dict, keys: list) -> bool:
        """Helper to check if all necessary keys exist and are valid for a star."""
        for key in keys:
            if key not in star.colnames:
                return False
            value = star[key]
            if value is None:
                return False
            if hasattr(value, 'mask') and np.ma.is_masked(value):
                return False
        return True

    def _identify_components_by_mag(self, gaia_results, wds_mags: Tuple[Optional[float], Optional[float]]):
        """
        Identifies the primary and secondary components from a list of Gaia results
        using algorithm that considers both magnitude and spatial proximity.
        
        Returns:
            Tuple of (primary_gaia, secondary_gaia) or (None, None) if insufficient sources
        """
        if len(gaia_results) < 2:
            log.warning(f"Only {len(gaia_results)} Gaia source(s) found - need at least 2")
            return None, None
            
        mag_pri_wds, mag_sec_wds = wds_mags

        if mag_pri_wds is None or mag_sec_wds is None:
            # If WDS magnitudes are not available, return the two closest sources
            # (they are already sorted by brightness from the query)
            return self._select_closest_pair(gaia_results)

        return self._match_components(gaia_results, mag_pri_wds, mag_sec_wds)
    
    def _select_closest_pair(self, gaia_results):
        """Select the two spatially closest sources from Gaia results."""
        if len(gaia_results) < 2:
            return None, None
            
        # Calculate all pairwise distances
        min_distance = np.inf
        best_pair = (gaia_results[0], gaia_results[1])
        
        for i in range(len(gaia_results)):
            for j in range(i + 1, len(gaia_results)):
                source1, source2 = gaia_results[i], gaia_results[j]
                
                # Calculate angular distance in arcseconds
                ra1, dec1 = source1['ra'], source1['dec']
                ra2, dec2 = source2['ra'], source2['dec']
                
                # Simple angular distance approximation for small separations
                delta_ra = (ra1 - ra2) * np.cos(np.radians((dec1 + dec2) / 2))
                delta_dec = dec1 - dec2
                distance_arcsec = np.sqrt(delta_ra**2 + delta_dec**2) * 3600
                
                if distance_arcsec < min_distance:
                    min_distance = distance_arcsec
                    best_pair = (source1, source2)
        
        # Return brightest first (primary)
        if best_pair[0]['phot_g_mean_mag'] <= best_pair[1]['phot_g_mean_mag']:
            return best_pair[0], best_pair[1]
        else:
            return best_pair[1], best_pair[0]
    
    def _match_components(self, gaia_results, mag_pri_wds, mag_sec_wds):
        """Component matching using magnitude + spatial proximity."""
        mag_tolerance = GAIA_MATCHING_MAG_TOLERANCE  # magnitudes
        
        pri_candidates = []
        sec_candidates = []
        
        for source in gaia_results:
            g_mag = source['phot_g_mean_mag']
            if g_mag is None or (hasattr(g_mag, 'mask') and np.ma.is_masked(g_mag)):
                continue
                
            pri_diff = abs(g_mag - mag_pri_wds)
            sec_diff = abs(g_mag - mag_sec_wds)
            
            if pri_diff <= mag_tolerance:
                pri_candidates.append((source, pri_diff))
            if sec_diff <= mag_tolerance:
                sec_candidates.append((source, sec_diff))
        
        # If no good magnitude matches, fall back to closest pair
        if not pri_candidates or not sec_candidates:
            log.warning(f"Poor magnitude matching for WDS mags {mag_pri_wds:.2f}, {mag_sec_wds:.2f}. Using closest pair.")
            return self._select_closest_pair(gaia_results)
        
        # Find the best pair considering both magnitude and spatial separation
        best_score = np.inf
        best_pri_match = None
        best_sec_match = None
        
        for pri_source, pri_mag_diff in pri_candidates:
            for sec_source, sec_mag_diff in sec_candidates:
                
                # Skip if same source
                if pri_source['source_id'] == sec_source['source_id']:
                    continue
                
                # Calculate spatial separation
                ra1, dec1 = pri_source['ra'], pri_source['dec']
                ra2, dec2 = sec_source['ra'], sec_source['dec']
                
                delta_ra = (ra1 - ra2) * np.cos(np.radians((dec1 + dec2) / 2))
                delta_dec = dec1 - dec2
                sep_arcsec = np.sqrt(delta_ra**2 + delta_dec**2) * 3600
                
                # Combined score: magnitude difference + spatial penalty
                # Prefer closer pairs with good magnitude matches
                mag_score = pri_mag_diff + sec_mag_diff
                spatial_penalty = min(sep_arcsec / GAIA_MATCHING_SPATIAL_PENALTY_FACTOR, GAIA_MATCHING_MAX_SPATIAL_PENALTY)
                combined_score = mag_score + spatial_penalty
                
                if combined_score < best_score:
                    best_score = combined_score
                    best_pri_match = pri_source
                    best_sec_match = sec_source
        
        if best_pri_match is not None and best_sec_match is not None:
            # Calculate final separation for logging
            ra1, dec1 = best_pri_match['ra'], best_pri_match['dec']
            ra2, dec2 = best_sec_match['ra'], best_sec_match['dec']
            delta_ra = (ra1 - ra2) * np.cos(np.radians((dec1 + dec2) / 2))
            delta_dec = dec1 - dec2
            final_sep = np.sqrt(delta_ra**2 + delta_dec**2) * 3600
            
            log.debug(f"Matched components: sep={final_sep:.2f}\", "
                     f"mags={best_pri_match['phot_g_mean_mag']:.2f}/{best_sec_match['phot_g_mean_mag']:.2f} "
                     f"vs WDS {mag_pri_wds:.2f}/{mag_sec_wds:.2f}")
            
            return best_pri_match, best_sec_match
        
        # Final fallback to brightness order
        log.warning("Component matching failed, using brightness order")
        return gaia_results[0], gaia_results[1] if len(gaia_results) >= 2 else (None, None)

    async def _query_gaia_for_pair_async(self, ra_deg: float, dec_deg: float, radius_arcsec: float):
        """
        Async wrapper for Gaia queries with retry logic and simple caching.
        """
        # Simple cache key based on coordinates and radius
        cache_key = f"{ra_deg:.4f}_{dec_deg:.4f}_{radius_arcsec:.1f}"
        
        # Check cache first
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        for attempt in range(GAIA_MAX_RETRY_ATTEMPTS):
            try:
                # Run the actual Gaia query in executor (it's inherently blocking)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    self._query_gaia_for_pair_sync, 
                    ra_deg, dec_deg, radius_arcsec, attempt + 1
                )
                
                if result is not None:
                    # Cache successful results
                    self._query_cache[cache_key] = result
                    return result
                    
                # If no results and attempts remain, wait before retry
                if attempt < GAIA_MAX_RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(GAIA_RETRY_DELAY_SECONDS)
                
            except Exception as e:
                log.warning(f"Gaia query attempt {attempt + 1} failed: {e}")
                if attempt < GAIA_MAX_RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(GAIA_RETRY_DELAY_SECONDS)
                else:
                    log.error(f"All {GAIA_MAX_RETRY_ATTEMPTS} Gaia query attempts failed")
        
        return None

    def _query_gaia_for_pair_sync(self, ra_deg: float, dec_deg: float, radius_arcsec: float, attempt: int = 1):
        """
        Synchronous Gaia query - called from async executor.
        Uses instance configuration to avoid modifying global state.
        Permissive query to capture binary systems.
        """
        try:
            query = f"""
            SELECT
                source_id, ra, dec, parallax, parallax_error,
                pmra, pmra_error, pmdec, pmdec_error, phot_g_mean_mag,
                parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr,
                astrometric_chi2_al, astrometric_n_good_obs_al,
                ruwe
            FROM {self.gaia_table}
            WHERE 1=CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_arcsec / 3600.0})
            ) AND phot_g_mean_mag < {self.mag_limit + GAIA_MAG_LIMIT_BUFFER}
              AND astrometric_chi2_al IS NOT NULL
              AND (ruwe < {GAIA_MAX_RUWE * GAIA_RUWE_QUERY_MULTIPLIER} OR ruwe IS NULL)
            ORDER BY phot_g_mean_mag ASC
            """
            
            log.debug(f"Querying Gaia at ({ra_deg:.4f}, {dec_deg:.4f}) "
                     f"radius {radius_arcsec:.1f}\" (attempt {attempt}) with relaxed filters")
            
            job = Gaia.launch_job(query)
            results = job.get_results()
            
            if len(results) == 0:
                log.debug(f"No Gaia sources found in {radius_arcsec:.1f}\" radius")
                return None
            elif len(results) < 2:
                log.debug(f"Only {len(results)} Gaia source found - need at least 2")
                return None
            else:
                log.debug(f"Found {len(results)} Gaia sources, returning top {min(len(results), self.max_sources)}")
                return results[:self.max_sources]
                
        except Exception as e:
            log.debug(f"Gaia query failed: {e}")
            raise e

    def _validate_astrometric_quality(self, star: Dict) -> bool:
        """
        Validates basic astrometric quality criteria for Gaia sources.
        Filters out sources with poor astrometric solutions.
        Updated with more stringent parallax requirements for better discrimination.
        
        Args:
            star: Gaia source data dictionary
            
        Returns:
            True if the source meets minimum quality criteria for analysis
        """
        try:
            # Relaxed RUWE threshold for wide binaries
            ruwe = star.get('ruwe')
            if ruwe is not None and ruwe > GAIA_MAX_RUWE * GAIA_RUWE_PERMISSIVE_MULTIPLIER:
                log.debug(f"Source {star.get('source_id', 'unknown')} rejected: RUWE {ruwe:.2f} > {GAIA_MAX_RUWE * GAIA_RUWE_PERMISSIVE_MULTIPLIER:.1f}")
                return False
            
            # More stringent parallax significance requirement (increased from 0.33σ to 1.5σ minimum)
            if 'parallax' in star.colnames and 'parallax_error' in star.colnames:
                parallax = star['parallax']
                parallax_error = star['parallax_error']
                if (parallax is not None and parallax_error is not None and 
                    parallax_error > 0 and abs(parallax / parallax_error) < MIN_PARALLAX_SIGNIFICANCE):
                    log.debug(f"Source {star.get('source_id', 'unknown')} rejected: Poor parallax SNR {abs(parallax / parallax_error):.1f} < {MIN_PARALLAX_SIGNIFICANCE}")
                    return False
            
            # Check for completely missing essential data
            has_parallax = (star.get('parallax') is not None and star.get('parallax_error') is not None and 
                           star.get('parallax_error', 0) > 0)
            has_pm = (star.get('pmra') is not None and star.get('pmdec') is not None and
                     star.get('pmra_error') is not None and star.get('pmdec_error') is not None and
                     star.get('pmra_error', 0) > 0 and star.get('pmdec_error', 0) > 0)
            
            if not has_parallax and not has_pm:
                log.debug(f"Source {star.get('source_id', 'unknown')} rejected: No usable astrometric data")
                return False
            
            return True
            
        except Exception as e:
            log.warning(f"Error validating astrometric quality: {e}")
            return True  # Conservative approach - don't reject on validation errors

    async def get_parallax_data(
        self,
        wds_summary: Dict[str, Any],
        search_radius_arcsec: float = DEFAULT_GAIA_SEARCH_RADIUS_ARCSEC
    ) -> Dict[str, Any]:
        """
        Retrieve parallax data for mass calculations.
        
        This method queries Gaia around the system position and returns
        the best available parallax measurement for mass calculations.
        
        Args:
            wds_summary: WDS summary data containing coordinates
            search_radius_arcsec: Search radius around system position
            
        Returns:
            Dict containing parallax information:
                - parallax: value in mas
                - parallax_error: uncertainty in mas
                - source: 'gaia_dr3'
                - gaia_source_id: Gaia source identifier
                - ruwe: Renormalised Unit Weight Error
                - significance: parallax/parallax_error ratio
                - g_mag: G-band magnitude
                
        Raises:
            ParallaxDataUnavailableError: When no suitable parallax data found
            GaiaQueryError: When Gaia query fails
        """
        try:
            # Get system coordinates
            ra_deg = wds_summary.get('ra_deg')
            dec_deg = wds_summary.get('dec_deg')
            
            if ra_deg is None or dec_deg is None:
                raise ParallaxDataUnavailableError("Missing coordinates for parallax query")
                
            # Query Gaia around the position
            gaia_data = await self._query_gaia_for_pair_async(ra_deg, dec_deg, search_radius_arcsec)
            
            if not gaia_data or len(gaia_data) == 0:
                raise ParallaxDataUnavailableError("No Gaia sources found for parallax query")
                
            # Select best parallax source
            best_star = self._select_best_parallax_source(gaia_data)
            
            if not best_star:
                raise ParallaxDataUnavailableError("No suitable parallax source found")
                
            # Extract parallax data
            parallax = best_star.get('parallax')
            parallax_error = best_star.get('parallax_error')
            
            if parallax is None or parallax_error is None:
                raise ParallaxDataUnavailableError("Missing parallax or parallax_error in Gaia data")
                
            # Check parallax significance (at least 3-sigma detection)
            significance = parallax / parallax_error if parallax_error > 0 else 0.0
            
            if significance < GAIA_PARALLAX_MIN_SIGNIFICANCE:
                raise ParallaxDataUnavailableError(f"Low parallax significance: {significance:.2f}")
                
            return {
                'parallax': float(parallax),
                'parallax_error': float(parallax_error),
                'source': 'gaia_dr3',
                'gaia_source_id': str(best_star.get('source_id', '')),
                'ruwe': float(best_star.get('ruwe', np.nan)),
                'significance': float(significance),
                'g_mag': float(best_star.get('phot_g_mean_mag', np.nan)),
                'search_radius_used': float(search_radius_arcsec)
            }
            
        except (ParallaxDataUnavailableError, GaiaQueryError):
            raise
        except Exception as e:
            raise GaiaQueryError(f"Error retrieving parallax data: {e}")
    
    def _select_best_parallax_source(self, gaia_data) -> Optional[Dict[str, Any]]:
        """
        Select the best parallax source from multiple Gaia detections.
        
        Selection criteria (in order of priority):
        1. Valid parallax and parallax_error
        2. Parallax significance >= 3.0 
        3. RUWE <= 1.4 (good astrometric solution)
        4. Brightest in G-band (most reliable astrometry)
        
        Args:
            gaia_data: Astropy Table with Gaia sources
            
        Returns:
            Best source as dict, or None if no suitable source
        """
        if not gaia_data or len(gaia_data) == 0:
            return None
            
        # Filter for valid parallax measurements
        valid_sources = []
        
        for star in gaia_data:
            parallax = star.get('parallax')
            parallax_error = star.get('parallax_error')
            
            if (parallax is not None and parallax_error is not None and 
                parallax_error > 0):
                
                significance = parallax / parallax_error
                
                if significance >= GAIA_PARALLAX_MIN_SIGNIFICANCE:  # At least 3-sigma detection
                    valid_sources.append({
                        'source_id': star.get('source_id'),
                        'parallax': parallax,
                        'parallax_error': parallax_error,
                        'significance': significance,
                        'ruwe': star.get('ruwe', 999.0),  # Default high RUWE if missing
                        'phot_g_mean_mag': star.get('phot_g_mean_mag', 99.0),  # Default faint if missing
                        'ra': star.get('ra'),
                        'dec': star.get('dec'),
                        'pmra': star.get('pmra'),
                        'pmdec': star.get('pmdec'),
                        'pmra_error': star.get('pmra_error'),
                        'pmdec_error': star.get('pmdec_error')
                    })
        
        if not valid_sources:
            return None
            
        # Sort by quality criteria:
        # 1. Good RUWE first (< 1.4)
        # 2. Higher parallax significance
        # 3. Brighter magnitude (reliable)
        def quality_score(source):
            ruwe_score = 1.0 if source['ruwe'] <= QUALITY_SCORE_RUWE_THRESHOLD else 0.5
            sig_score = min(source['significance'] / QUALITY_SCORE_SIGNIFICANCE_NORMALIZATION, 1.0)  # Normalize to [0,1]
            mag_score = max(0.0, (QUALITY_SCORE_MAG_REFERENCE - source['phot_g_mean_mag']) / QUALITY_SCORE_MAG_REFERENCE)  # Brighter stars preferred
            
            return ruwe_score * QUALITY_SCORE_RUWE_WEIGHT + sig_score * QUALITY_SCORE_SIGNIFICANCE_WEIGHT + mag_score * QUALITY_SCORE_MAGNITUDE_WEIGHT
        
        # Select source with highest quality score
        best_source = max(valid_sources, key=quality_score)
        
        log.debug(f"Selected parallax source: ID={best_source['source_id']}, "
                 f"π={best_source['parallax']:.3f}±{best_source['parallax_error']:.3f} mas, "
                 f"significance={best_source['significance']:.1f}, "
                 f"RUWE={best_source['ruwe']:.2f}")
        
        return best_source