from astroquery.gaia import Gaia
import numpy as np
from scipy.stats import chi2
from typing import Tuple, Optional, Dict, Any
from enum import Enum
import asyncio
import logging
import functools
import re
import json
from datetime import datetime
from astropy import units as u
from astropy.coordinates import SkyCoord

# Import configuration constants
from ..config import (
    DEFAULT_GAIA_TABLE,
    DEFAULT_PHYSICAL_P_VALUE_THRESHOLD,
    DEFAULT_AMBIGUOUS_P_VALUE_THRESHOLD,
    MIN_PARALLAX_SIGNIFICANCE,
    GAIA_MAX_RUWE,
    GAIA_DEFAULT_CORRELATION_MISSING,
    QUALITY_SCORE_RUWE_WEIGHT,
    QUALITY_SCORE_SIGNIFICANCE_WEIGHT,
    QUALITY_SCORE_MAGNITUDE_WEIGHT,
    QUALITY_SCORE_RUWE_THRESHOLD,
    QUALITY_SCORE_SIGNIFICANCE_NORMALIZATION,
    QUALITY_SCORE_MAG_REFERENCE,
    GAIA_MIN_SOURCES_REQUIRED,
    GAIA_PARALLAX_MIN_SIGNIFICANCE,
    GAIA_RUWE_PERMISSIVE_MULTIPLIER,
    GAIA_WDS_SEPARATION_TOLERANCE_FRACTION,
    GAIA_RUWE_RELAX_THRESHOLD,
    GAIA_RUWE_TOLERANCE_MULTIPLIER,
    GAIA_REFERENCE_EPOCH,
    GAIA_WDS_MAX_EPOCH_DIFFERENCE_YEARS,
    EXPERT_PARALLAX_COMPATIBILITY_SIGMA_THRESHOLD,
    # RUWE correction configuration
    RUWE_CORRECTION_ENABLED,
    RUWE_CORRECTION_APPLY_TO_ALL_DIMENSIONS,
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
                 physical_p_value_threshold: float = DEFAULT_PHYSICAL_P_VALUE_THRESHOLD,
                 ambiguous_p_value_threshold: float = DEFAULT_AMBIGUOUS_P_VALUE_THRESHOLD,
                 gaia_client=None):
        """
        Initializes the Gaia validator with configuration parameters.

        Args:
            gaia_table: The Gaia data release table to query
            physical_p_value_threshold: The p-value above which a pair is considered
                                        'Likely Physical'. Default: 0.045 (4.5%)
            ambiguous_p_value_threshold: The p-value above which a pair is 'Ambiguous'.
                                         Below this: 'Likely Optical'. Default: 0.005 (0.5%)
            gaia_client: Optional Gaia client for dependency injection (testing)

        Raises:
            ValueError: If thresholds are not in valid order
        """
        self.gaia_table = gaia_table
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
            
        
        log.info(f"GaiaValidator initialized: table={gaia_table}, "
                f"thresholds=({self.physical_threshold:.3f}, {self.ambiguous_threshold:.3f})")
    
    
    async def validate_physicality(self,
                                   wds_summary: Dict[str, Any]) -> PhysicalityAssessment:
        """
        Validates if a binary system is physically bound using Gaia data.
        Prioritizes Gaia source IDs stored in the summary to avoid heuristic matching.
        
        Args:
            wds_summary: WDS summary data containing coordinates, magnitudes, and optionally Gaia IDs
            
        Returns:
            PhysicalityAssessment object
            
        Raises:
            PhysicalityValidationError: When validation cannot be completed
            InsufficientAstrometricDataError: When insufficient data is available
            GaiaQueryError: When Gaia queries fail
        """
        gaia_source_ids = self._extract_gaia_source_ids(wds_summary)
        if not gaia_source_ids or len(gaia_source_ids) < 2:
            raise InsufficientAstrometricDataError("Gaia source IDs required for physicality validation")

        log.debug("Using Gaia source IDs for validation: %s", gaia_source_ids)
        try:
            gaia_results = await self._query_gaia_by_source_ids_async(gaia_source_ids)
        except InsufficientAstrometricDataError:
            raise
        except Exception as exc:
            raise GaiaQueryError(f"Direct Gaia source ID query failed: {exc}") from exc

        result, primary_gaia, secondary_gaia = self._validate_physicality_sync(
            gaia_results,
            wds_summary=wds_summary,
            gaia_source_ids=gaia_source_ids
        )
        return self._create_final_assessment(
            result,
            primary_gaia,
            secondary_gaia,
            search_radius_arcsec=None,
            direct_source_query=True
        )
    
    def _extract_gaia_source_ids(self, wds_summary: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Extract Gaia source IDs using the supported WDS string formats."""
        log.debug("Extracting Gaia source IDs from WDS metadata")
        log.debug("WDS summary keys: %s", list(wds_summary.keys()))

        id_prefixes = {
            "DR1", "DR2", "DR3", "DR4", "DR5", "DR6", "DR7", "DR8", "DR9",
            "EDR3", "ER3"
        }
        wds_pattern = re.compile(r"^\d{6,7}[+-]\d{6}$")

        def _normalize_identifier(raw: Any) -> Optional[str]:
            if raw is None:
                return None
            candidate = str(raw).strip()
            if not candidate:
                return None

            digits_only = ''.join(ch for ch in candidate if ch.isdigit())
            if digits_only:
                return digits_only

            return candidate if candidate.isdigit() else None

        def _extract_id_from_tokens(tokens: list[str]) -> Optional[str]:
            for index, token in enumerate(tokens):
                normalized = token.strip()
                upper = normalized.upper()

                if upper in id_prefixes and index + 1 < len(tokens):
                    digits = ''.join(ch for ch in tokens[index + 1] if ch.isdigit())
                    if digits:
                        return digits

                if normalized.startswith('[') and ']' in normalized:
                    next_part = tokens[index + 1] if index + 1 < len(tokens) else ''
                    identifier = normalized if not next_part else f"{normalized} {next_part.strip()}"
                    return identifier.strip()

            return None

        def _parse_component_record(text: str) -> Optional[Tuple[str, str]]:
            tokens = text.split()
            if len(tokens) < 3:
                return None
            if not wds_pattern.match(tokens[0]):
                return None

            component_token = tokens[1]
            if not component_token.isalpha():
                return None

            component = component_token.upper()
            identifier = _extract_id_from_tokens(tokens)
            if identifier:
                return component, identifier
            return None

        gaia_ids: Dict[str, str] = {}

        def _normalize_component_letter(raw_value: Any) -> Optional[str]:
            if raw_value is None:
                return None
            text = str(raw_value).strip()
            if not text:
                return None
            upper_text = text.upper()
            if upper_text in {'PRIMARY', 'SECONDARY'}:
                return None
            for char in text:
                if char.isalpha():
                    return char.upper()
            return None

        def _collect_letters_from_string(raw_value: Any) -> list[str]:
            if raw_value is None:
                return []
            text = str(raw_value).strip()
            if not text:
                return []

            letters: list[str] = []
            cleaned = re.sub(r"[^A-Za-z]", " ", text)
            tokens = [token for token in cleaned.split() if token]
            for token in tokens:
                normalized = ''.join(ch for ch in token.upper() if ch.isalpha())
                if not normalized:
                    continue
                if normalized in {'PRIMARY', 'SECONDARY'}:
                    continue
                if len(normalized) == 1:
                    if normalized not in letters:
                        letters.append(normalized)
                else:
                    for char in normalized:
                        if char.isalpha():
                            upper = char.upper()
                            if upper not in letters:
                                letters.append(upper)
                        if len(letters) >= 2:
                            break
                if len(letters) >= 2:
                    break

            if not letters:
                for char in text:
                    if char.isalpha():
                        upper = char.upper()
                        if upper not in letters:
                            letters.append(upper)
                        if len(letters) >= 2:
                            break

            return letters

        primary_letter = None
        secondary_letter = None

        for field in (
            'pair_primary_component',
            'primary_component',
            'primary_component_letter',
            'component_letter_primary',
            'primary_component_name',
            'component_primary'
        ):
            candidate = _normalize_component_letter(wds_summary.get(field))
            if candidate:
                primary_letter = candidate
                break

        for field in (
            'pair_secondary_component',
            'secondary_component',
            'secondary_component_letter',
            'component_letter_secondary',
            'secondary_component_name',
            'component_secondary'
        ):
            candidate = _normalize_component_letter(wds_summary.get(field))
            if candidate:
                secondary_letter = candidate
                break

        pair_letters: list[str] = []
        if primary_letter:
            pair_letters.append(primary_letter)
        if secondary_letter and secondary_letter not in pair_letters:
            pair_letters.append(secondary_letter)

        for field in ('component_pair', 'components', 'pair', 'pair_label', 'pair_name'):
            if len(pair_letters) >= 2:
                break
            for letter in _collect_letters_from_string(wds_summary.get(field)):
                if letter not in pair_letters:
                    pair_letters.append(letter)
                if len(pair_letters) >= 2:
                    break

        if not pair_letters:
            ordered_letters: list[str] = ['A', 'B']
            primary_letter = ordered_letters[0]
            secondary_letter = ordered_letters[1]
        else:
            ordered_letters = pair_letters.copy()
            primary_letter = primary_letter or ordered_letters[0]
            secondary_letter = secondary_letter or (ordered_letters[1] if len(ordered_letters) > 1 else None)

        # First, scan all textual fields for the supported WDS record formats
        for value in wds_summary.values():
            if not isinstance(value, str):
                continue
            parsed = _parse_component_record(value.strip())
            if parsed is None:
                continue
            component, identifier = parsed
            gaia_ids.setdefault(component, identifier)
            if component not in ordered_letters:
                ordered_letters.append(component)

        if not ordered_letters:
            ordered_letters = ['A', 'B']

        primary_letter = ordered_letters[0]
        secondary_letter = ordered_letters[1] if len(ordered_letters) > 1 else None

        def _record_structured_id(component_hint: Optional[str], raw_identifier: Any) -> None:
            identifier = _normalize_identifier(raw_identifier)
            if not identifier:
                return

            if component_hint:
                normalized_hint = component_hint.strip().upper()
                if len(normalized_hint) == 1 and normalized_hint.isalpha():
                    gaia_ids.setdefault(normalized_hint, identifier)
                    if normalized_hint not in ordered_letters:
                        ordered_letters.append(normalized_hint)
                    return

                if normalized_hint in {'PRIMARY', 'SECONDARY'}:
                    target_letter = primary_letter if normalized_hint == 'PRIMARY' else secondary_letter
                    if target_letter:
                        upper_target = target_letter.upper()
                        gaia_ids.setdefault(upper_target, identifier)
                        if upper_target not in ordered_letters:
                            ordered_letters.append(upper_target)
                    return

            for letter in ordered_letters:
                if letter not in gaia_ids:
                    gaia_ids[letter] = identifier
                    return

            for fallback_letter in ('A', 'B', 'C', 'D'):
                if fallback_letter not in gaia_ids:
                    gaia_ids[fallback_letter] = identifier
                    return

        structured_ids_raw = wds_summary.get('gaia_source_ids')
        structured_mapping: Dict[str, Any] = {}

        if isinstance(structured_ids_raw, dict):
            structured_mapping = structured_ids_raw
        elif isinstance(structured_ids_raw, str):
            payload = structured_ids_raw.strip()
            if payload:
                try:
                    parsed_payload = json.loads(payload)
                except json.JSONDecodeError:
                    log.debug("Failed to decode gaia_source_ids JSON: %s", payload)
                else:
                    if isinstance(parsed_payload, dict):
                        structured_mapping = parsed_payload
                    elif isinstance(parsed_payload, list):
                        for entry in parsed_payload:
                            if not isinstance(entry, dict):
                                continue
                            component_key = (entry.get('component') or
                                             entry.get('component_letter') or
                                             entry.get('component_name'))
                            value = (entry.get('source_id') or
                                     entry.get('gaia_id') or
                                     entry.get('value'))
                            if component_key is None:
                                continue
                            structured_mapping[str(component_key)] = value
        elif isinstance(structured_ids_raw, list):
            for entry in structured_ids_raw:
                if not isinstance(entry, dict):
                    continue
                component_key = (entry.get('component') or
                                 entry.get('component_letter') or
                                 entry.get('component_name'))
                value = (entry.get('source_id') or
                         entry.get('gaia_id') or
                         entry.get('value'))
                if component_key is None:
                    continue
                structured_mapping[str(component_key)] = value

        for key, value in structured_mapping.items():
            if key is None:
                continue
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            if normalized_key.lower() == 'component':
                continue
            _record_structured_id(normalized_key, value)

        def _parse_simple_id(field_value: Any) -> Optional[str]:
            if field_value is None:
                return None
            tokens = str(field_value).split()
            return _extract_id_from_tokens(tokens)

        primary_simple = _parse_simple_id(wds_summary.get('gaia_id_primary'))
        if primary_simple:
            gaia_ids.setdefault(primary_letter, primary_simple)

        secondary_simple = _parse_simple_id(wds_summary.get('gaia_id_secondary'))
        if secondary_simple and secondary_letter:
            gaia_ids.setdefault(secondary_letter, secondary_simple)

        if ordered_letters:
            filtered_ids = {}
            for letter in ordered_letters:
                value = gaia_ids.get(letter)
                if value:
                    filtered_ids[letter] = value
            if len(filtered_ids) >= 2:
                return filtered_ids

        if len(gaia_ids) >= 2:
            filtered_items = list(gaia_ids.items())[:2]
            return {key: value for key, value in filtered_items}
        return None
    
    async def _query_gaia_by_source_ids_async(self, gaia_source_ids: Dict[str, str]):
        """Query Gaia directly by source IDs - much more reliable than coordinate search."""
        from astroquery.gaia import Gaia
        
        source_ids: list[str] = []
        for component, raw_id in gaia_source_ids.items():
            candidate = str(raw_id).strip()
            if not candidate.isdigit():
                raise InsufficientAstrometricDataError(
                    f"Gaia source ID for component {component} is not numeric: {candidate}"
                )
            source_ids.append(candidate)
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
            'search_radius_arcsec': search_radius_arcsec,
            'significance_thresholds': {
                'physical': self.physical_threshold,
                'ambiguous': self.ambiguous_threshold
            },
            'retry_attempts': 1
        }
    
    def _create_final_assessment(
        self,
        result: Dict[str, Any],
        primary_gaia,
        secondary_gaia,
        search_radius_arcsec: Optional[float],
        direct_source_query: bool = False
    ) -> PhysicalityAssessment:
        """Convert validation result to final PhysicalityAssessment."""
        expected_sep = result.get('expected_separation_arcsec')
        expected_pa = result.get('expected_position_angle_deg')

        gaia_primary_id = None
        gaia_secondary_id = None

        if primary_gaia is not None:
            try:
                gaia_primary_id = primary_gaia['source_id']
            except Exception:
                gaia_primary_id = primary_gaia.get('source_id') if hasattr(primary_gaia, 'get') else None

        if secondary_gaia is not None:
            try:
                gaia_secondary_id = secondary_gaia['source_id']
            except Exception:
                gaia_secondary_id = secondary_gaia.get('source_id') if hasattr(secondary_gaia, 'get') else None

        parallax_sigma_diff = None
        pm_sigma_diff = None
        if primary_gaia and secondary_gaia:
            parallax_sigma_diff = self._compute_parallax_sigma_difference(primary_gaia, secondary_gaia)
            pm_sigma_diff = self._compute_pm_sigma_difference(primary_gaia, secondary_gaia)
            if parallax_sigma_diff is not None:
                result['gaia_parallax_sigma_difference'] = parallax_sigma_diff
            if pm_sigma_diff is not None:
                result['gaia_pm_sigma_difference'] = pm_sigma_diff
        
        result = self._apply_orbital_evidence_corrections(result, parallax_sigma_diff)

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

        if parallax_sigma_diff is not None:
            assessment['gaia_parallax_sigma_difference'] = parallax_sigma_diff
        if pm_sigma_diff is not None:
            assessment['gaia_pm_sigma_difference'] = pm_sigma_diff

        for key in (
            'delta_mu_orbit',
            'delta_mu_orbit_error',
            'delta_mu_orbit_significance',
            'separation_arcsec',
            'position_angle_deg',
            'proper_motion_difference',
            'override_reason'
        ):
            if key in result:
                assessment[key] = result[key]
        
        return assessment
    
    def _extract_expected_geometry(self, wds_summary: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
        """Extract expected angular separation and position angle from WDS metadata."""
        if not wds_summary:
            return None, None

        def _as_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            if hasattr(value, 'mask') and np.ma.is_masked(value):
                return None
            try:
                float_val = float(value)
            except (TypeError, ValueError):
                return None
            return float_val

        sep_last = _as_float(wds_summary.get('sep_last'))
        sep_first = _as_float(wds_summary.get('sep_first'))
        pa_last = _as_float(wds_summary.get('pa_last'))
        pa_first = _as_float(wds_summary.get('pa_first'))

        expected_sep = None
        if sep_last is not None and sep_last > 0:
            expected_sep = sep_last
        elif sep_first is not None and sep_first > 0:
            expected_sep = sep_first

        expected_pa = None
        if pa_last is not None:
            expected_pa = pa_last
        elif pa_first is not None:
            expected_pa = pa_first

        return expected_sep, expected_pa

    def _attach_expected_geometry(
        self,
        result: Optional[Dict[str, Any]],
        expected_sep: Optional[float],
        expected_pa: Optional[float]
    ) -> None:
        """Persist expected geometry on result dictionaries for downstream use."""
        if result is None:
            return

        if expected_sep is not None and expected_sep > 0:
            result.setdefault('expected_separation_arcsec', expected_sep)

        if expected_pa is not None and np.isfinite(expected_pa):
            normalized_pa = expected_pa % 360.0
            result.setdefault('expected_position_angle_deg', normalized_pa)

    def _resolve_component_letters(
        self,
        wds_summary: Optional[Dict[str, Any]],
        normalized_ids: Dict[str, str],
        ordered_components: Optional[list] = None
    ) -> Tuple[str, str]:
        """Determine which component letters correspond to primary and secondary stars."""
        pair_label = ''
        if wds_summary:
            pair_label = (
                wds_summary.get('component_pair')
                or wds_summary.get('components')
                or ''
            )

        letters = [char.upper() for char in pair_label if char.isalpha()]
        if len(letters) >= 2:
            return letters[0], letters[1]

        fallback: list[str] = []
        if ordered_components:
            for component in ordered_components:
                normalized = str(component).strip().upper()
                if normalized and normalized not in fallback:
                    fallback.append(normalized)
        if len(fallback) >= 2:
            return fallback[0], fallback[1]

        remaining = [key for key in normalized_ids.keys() if key]
        if len(remaining) >= 2:
            return remaining[0], remaining[1]

        return 'A', 'B'

    def _validate_physicality_sync(self,
                                  gaia_results,
                                  wds_summary: Optional[Dict[str, Any]] = None,
                                  gaia_source_ids: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Any], Any, Any]:
        """
        Simplified physicality validation that delegates to the appropriate validator.
        
        This function is now a clean orchestrator that:
        1. Identifies the correct binary components 
        2. Chooses between traditional p-value method and Expert Hierarchical Validator
        3. Returns structured results
        
        Args:
            gaia_results: Pre-fetched Gaia query results
            wds_summary: Contextual WDS data for geometry checks
            gaia_source_ids: Mapping of component letters to Gaia source IDs
        """
        if gaia_source_ids is None or len(gaia_source_ids) < 2:
            raise InsufficientAstrometricDataError("Gaia source IDs required for component identification")

        normalized_ids: Dict[str, str] = {}
        ordered_components: list[str] = []
        for component, value in gaia_source_ids.items():
            if value is None:
                continue
            component_key = str(component).strip()
            if not component_key:
                continue
            if component_key not in ordered_components:
                ordered_components.append(component_key)
            normalized_ids[component_key.upper()] = str(value)

        if len(normalized_ids) < 2:
            raise InsufficientAstrometricDataError("Gaia source ID mapping incomplete for component pair")

        primary_letter, secondary_letter = self._resolve_component_letters(
            wds_summary,
            normalized_ids,
            ordered_components
        )

        primary_id = normalized_ids.get(primary_letter)
        secondary_id = normalized_ids.get(secondary_letter)

        if not primary_id or not secondary_id:
            raise InsufficientAstrometricDataError(
                f"Missing Gaia source IDs for components {primary_letter}/{secondary_letter}"
            )

        indexed_results: Dict[str, Any] = {}
        total_sources = 0
        for star in gaia_results:
            total_sources += 1
            source_value = None
            if hasattr(star, 'colnames') and 'source_id' in getattr(star, 'colnames', []):
                source_value = star['source_id']
            elif hasattr(star, 'get'):
                source_value = star.get('source_id')
            elif isinstance(star, dict):
                source_value = star.get('source_id')
            if source_value is None:
                continue
            indexed_results[str(source_value)] = star

        if total_sources < GAIA_MIN_SOURCES_REQUIRED:
            raise InsufficientAstrometricDataError(
                f"Only {total_sources} Gaia sources returned; need ≥{GAIA_MIN_SOURCES_REQUIRED}"
            )

        primary_gaia = indexed_results.get(primary_id)
        secondary_gaia = indexed_results.get(secondary_id)

        if primary_gaia is None or secondary_gaia is None:
            missing_components = []
            if primary_gaia is None:
                missing_components.append(primary_letter)
            if secondary_gaia is None:
                missing_components.append(secondary_letter)
            raise InsufficientAstrometricDataError(
                f"Gaia query did not return expected components: {', '.join(missing_components)}"
            )

        for component_label, star in ((primary_letter, primary_gaia), (secondary_letter, secondary_gaia)):
            if not self._validate_astrometric_quality(star):
                raise InsufficientAstrometricDataError(
                    f"Component {component_label} fails Gaia astrometric quality filters"
                )

        expected_sep, expected_pa = self._extract_expected_geometry(wds_summary)

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

        # Preserve the expected geometry context for downstream consumers
        self._attach_expected_geometry(expert_result, expected_sep, expected_pa)
        if statistical_result:
            self._attach_expected_geometry(statistical_result, expected_sep, expected_pa)

        # If expert reasoning failed or remained ambiguous, fall back to statistical evidence
        expert_failed = expert_result.get('expert_method') == "fallback_error"
        expert_ambiguous = expert_result['label'] == PhysicalityLabel.AMBIGUOUS

        if statistical_result and (expert_failed or expert_ambiguous):
            statistical_result['expert_confidence'] = expert_result.get('expert_confidence', 0.0)
            statistical_result['expert_reasoning'] = expert_result.get('expert_reasoning')
            statistical_result['expert_method'] = expert_result.get('expert_method')
            return statistical_result, primary_gaia, secondary_gaia

        # Otherwise, enrich expert result with statistical context when available
        if statistical_result:
            if expert_result.get('p_value') is None:
                expert_result['p_value'] = statistical_result['p_value']
            expert_result['statistical_method'] = statistical_result['method']
            expert_result['statistical_p_value'] = statistical_result['p_value']
            expert_result['statistical_label'] = statistical_result['label']

        return expert_result, primary_gaia, secondary_gaia

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
            # Prefer the most recent WDS separation measurement
            wds_sep = wds_summary.get('sep_last')
            wds_epoch = wds_summary.get('date_last')

            if (wds_sep is None or wds_sep <= 0) and wds_summary.get('sep_first'):
                # Fall back to the first measurement only if the latest one is unavailable
                wds_sep = wds_summary.get('sep_first')
                wds_epoch = wds_summary.get('date_first')

            if wds_sep is None or wds_sep <= 0:
                log.debug("No WDS separation available for verification; skipping check")
                return True

            try:
                wds_sep = float(wds_sep)
            except (TypeError, ValueError):
                log.debug("WDS separation value %r is not numeric; skipping check", wds_sep)
                return True

            def _as_float(value: Any) -> Optional[float]:
                if value is None:
                    return None
                if hasattr(value, 'mask') and np.ma.is_masked(value):
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            wds_epoch_val = _as_float(wds_epoch)
            if wds_epoch_val is not None:
                epoch_diff = abs(wds_epoch_val - GAIA_REFERENCE_EPOCH)
                if epoch_diff > GAIA_WDS_MAX_EPOCH_DIFFERENCE_YEARS:
                    log.debug(
                        "Latest WDS epoch %.3f is %.2f years away from Gaia reference %.1f; skipping strict separation check",
                        wds_epoch_val,
                        epoch_diff,
                        GAIA_REFERENCE_EPOCH
                    )
                    return True

            gaia_sep = self._calculate_angular_separation(primary_gaia, secondary_gaia)

            tolerance_fraction = GAIA_WDS_SEPARATION_TOLERANCE_FRACTION
            ruwe_primary = _as_float(primary_gaia.get('ruwe'))
            ruwe_secondary = _as_float(secondary_gaia.get('ruwe'))
            if (
                (ruwe_primary is not None and ruwe_primary > GAIA_RUWE_RELAX_THRESHOLD)
                or (ruwe_secondary is not None and ruwe_secondary > GAIA_RUWE_RELAX_THRESHOLD)
            ):
                tolerance_fraction *= GAIA_RUWE_TOLERANCE_MULTIPLIER
                log.debug(
                    "Relaxing separation tolerance due to RUWE (primary=%.2f, secondary=%.2f) -> %.2f",
                    ruwe_primary if ruwe_primary is not None else float('nan'),
                    ruwe_secondary if ruwe_secondary is not None else float('nan'),
                    tolerance_fraction
                )

            sep_diff_fraction = abs(gaia_sep - wds_sep) / wds_sep
            is_consistent = sep_diff_fraction <= tolerance_fraction

            log.debug(
                "Separation verification: Gaia=%.2f\" (epoch %.1f), WDS=%.2f\" (epoch %s), fractional diff=%.2f <= %.2f (%s)",
                gaia_sep,
                GAIA_REFERENCE_EPOCH,
                wds_sep,
                f"{wds_epoch_val:.3f}" if wds_epoch_val is not None else "unknown",
                sep_diff_fraction,
                tolerance_fraction,
                "OK" if is_consistent else "FAIL"
            )

            return is_consistent
        except Exception as exc:
            log.warning("Error verifying separation consistency: %s", exc)
            return True

    def _calculate_angular_separation(self, star1: Dict, star2: Dict) -> float:
        """Calculate angular separation between two Gaia detections in arcseconds."""
        separation, _ = self._calculate_pair_geometry(star1, star2)
        if separation is None:
            raise ValueError("Missing coordinates for Gaia separation computation")
        return separation

    def _safe_gaia_value(self, star: Dict[str, Any], key: str) -> Optional[float]:
        value = star.get(key) if hasattr(star, 'get') else getattr(star, key, None)
        if value is None:
            return None
        if hasattr(value, 'mask') and np.ma.is_masked(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _compute_parallax_sigma_difference(self, star1: Dict, star2: Dict) -> Optional[float]:
        plx1 = self._safe_gaia_value(star1, 'parallax')
        plx2 = self._safe_gaia_value(star2, 'parallax')
        if plx1 is None or plx2 is None:
            return None

        ruwe1 = self._safe_gaia_value(star1, 'ruwe')
        ruwe2 = self._safe_gaia_value(star2, 'ruwe')

        err1_raw = self._safe_gaia_value(star1, 'parallax_error')
        err2_raw = self._safe_gaia_value(star2, 'parallax_error')
        if err1_raw is None or err2_raw is None:
            return None

        err1 = get_gaia_parallax_error_safe(err1_raw, ruwe1)
        err2 = get_gaia_parallax_error_safe(err2_raw, ruwe2)
        combined = np.sqrt(err1**2 + err2**2)
        if not np.isfinite(combined) or combined <= 0:
            return None

        return abs(plx1 - plx2) / combined

    def _compute_pm_sigma_difference(self, star1: Dict, star2: Dict) -> Optional[float]:
        pmra1 = self._safe_gaia_value(star1, 'pmra')
        pmra2 = self._safe_gaia_value(star2, 'pmra')
        pmdec1 = self._safe_gaia_value(star1, 'pmdec')
        pmdec2 = self._safe_gaia_value(star2, 'pmdec')
        if None in (pmra1, pmra2, pmdec1, pmdec2):
            return None

        ruwe1 = self._safe_gaia_value(star1, 'ruwe')
        ruwe2 = self._safe_gaia_value(star2, 'ruwe')

        pmra_err1_raw = self._safe_gaia_value(star1, 'pmra_error')
        pmra_err2_raw = self._safe_gaia_value(star2, 'pmra_error')
        pmdec_err1_raw = self._safe_gaia_value(star1, 'pmdec_error')
        pmdec_err2_raw = self._safe_gaia_value(star2, 'pmdec_error')
        if None in (pmra_err1_raw, pmra_err2_raw, pmdec_err1_raw, pmdec_err2_raw):
            return None

        pmra_err1 = get_gaia_pmra_error_safe(pmra_err1_raw, ruwe1)
        pmra_err2 = get_gaia_pmra_error_safe(pmra_err2_raw, ruwe2)
        pmdec_err1 = get_gaia_pmdec_error_safe(pmdec_err1_raw, ruwe1)
        pmdec_err2 = get_gaia_pmdec_error_safe(pmdec_err2_raw, ruwe2)

        combined_pmra_err = np.sqrt(pmra_err1**2 + pmra_err2**2)
        combined_pmdec_err = np.sqrt(pmdec_err1**2 + pmdec_err2**2)
        combined = np.sqrt(combined_pmra_err**2 + combined_pmdec_err**2)
        if not np.isfinite(combined) or combined <= 0:
            return None

        diff = np.hypot(pmra1 - pmra2, pmdec1 - pmdec2)
        return diff / combined

    def _apply_orbital_evidence_corrections(
        self,
        result: Dict[str, Any],
        parallax_sigma_diff: Optional[float]
    ) -> Dict[str, Any]:
        """Apply orbital-evidence-based overrides in a testable, modular fashion."""
        delta_mu_sig = result.get('delta_mu_orbit_significance')

        if not self._has_valid_orbital_evidence(delta_mu_sig):
            return result

        correction_type = self._determine_orbital_correction_type(
            result,
            parallax_sigma_diff,
            float(delta_mu_sig)
        )

        if correction_type == 'promote_to_physical':
            return self._promote_veto_to_physical(result, float(delta_mu_sig), parallax_sigma_diff)
        if correction_type == 'downgrade_to_ambiguous':
            return self._downgrade_veto_to_ambiguous(result, float(delta_mu_sig), parallax_sigma_diff)
        if correction_type == 'downgrade_excess_orbital':
            return self._downgrade_excess_orbital(result, float(delta_mu_sig))

        return result

    def _has_valid_orbital_evidence(self, delta_mu_sig: Any) -> bool:
        """Return True when orbital evidence significance is usable."""
        return delta_mu_sig is not None and isinstance(delta_mu_sig, (int, float)) and np.isfinite(delta_mu_sig)

    def _determine_orbital_correction_type(
        self,
        result: Dict[str, Any],
        parallax_sigma_diff: Optional[float],
        delta_mu_sig: float
    ) -> str:
        """Classify the orbital correction that should be applied."""
        try:
            from astrakairos.config import (
                ORBIT_EXCESS_SIGMA_PHYSICAL_MAX,
                ORBIT_EXCESS_SIGMA_AMBIGUOUS_MAX,
                EXPERT_PARALLAX_COMPATIBILITY_SIGMA_THRESHOLD,
            )
        except ImportError as exc:
            log.warning(f"Config constants unavailable: {exc}")
            return 'no_correction'

        current_label = result.get('label')
        method_tag = self._normalize_method_tag(result.get('expert_method'))

        parallax_consistent = (
            parallax_sigma_diff is not None
            and parallax_sigma_diff < EXPERT_PARALLAX_COMPATIBILITY_SIGMA_THRESHOLD
        )

        if (
            current_label == PhysicalityLabel.LIKELY_OPTICAL
            and method_tag == 'optical_veto_astrometry'
            and parallax_consistent
        ):
            if delta_mu_sig < ORBIT_EXCESS_SIGMA_PHYSICAL_MAX:
                return 'promote_to_physical'
            if delta_mu_sig < ORBIT_EXCESS_SIGMA_AMBIGUOUS_MAX:
                return 'downgrade_to_ambiguous'

        if current_label == PhysicalityLabel.LIKELY_PHYSICAL:
            if delta_mu_sig >= ORBIT_EXCESS_SIGMA_AMBIGUOUS_MAX:
                return 'downgrade_excess_orbital'

        return 'no_correction'

    def _promote_veto_to_physical(
        self,
        result: Dict[str, Any],
        delta_mu_sig: float,
        parallax_sigma_diff: Optional[float]
    ) -> Dict[str, Any]:
        """Promote chi²-vetoed systems to Likely Physical when orbital evidence supports it."""
        log.info(
            "Δμ override: promoting chi² vetoed system to Likely Physical (Δμ=%.2fσ, Δϖ=%sσ)",
            delta_mu_sig,
            f"{parallax_sigma_diff:.2f}" if parallax_sigma_diff is not None else "nan"
        )

        result['label'] = PhysicalityLabel.LIKELY_PHYSICAL
        result['expert_method'] = 'expert_delta_mu_override'
        result['override_reason'] = 'delta_mu_consistent'

        current_conf = result.get('expert_confidence', 0.6)
        if not isinstance(current_conf, (int, float)) or not np.isfinite(current_conf):
            current_conf = 0.6
        result['expert_confidence'] = max(current_conf, 0.65)

        return result

    def _downgrade_veto_to_ambiguous(
        self,
        result: Dict[str, Any],
        delta_mu_sig: float,
        parallax_sigma_diff: Optional[float]
    ) -> Dict[str, Any]:
        """Downgrade chi² vetoed systems to Ambiguous when orbital evidence is transitional."""
        log.info(
            "Δμ override: downgrading chi² vetoed system to Ambiguous (Δμ=%.2fσ, Δϖ=%sσ)",
            delta_mu_sig,
            f"{parallax_sigma_diff:.2f}" if parallax_sigma_diff is not None else "nan"
        )

        result['label'] = PhysicalityLabel.AMBIGUOUS
        result['expert_method'] = 'expert_delta_mu_override_ambiguous'
        result['override_reason'] = 'delta_mu_transitional'

        current_conf = result.get('expert_confidence', 0.5)
        if not isinstance(current_conf, (int, float)) or not np.isfinite(current_conf):
            current_conf = 0.5
        result['expert_confidence'] = max(current_conf, 0.55)

        return result

    def _downgrade_excess_orbital(
        self,
        result: Dict[str, Any],
        delta_mu_sig: float
    ) -> Dict[str, Any]:
        """Downgrade systems with excessive orbital motion incompatible with co-motion."""
        result['label'] = PhysicalityLabel.LIKELY_OPTICAL
        result['expert_method'] = 'expert_delta_mu_excess'
        result['override_reason'] = 'delta_mu_excess'
        return result

    def _normalize_method_tag(self, method_value: Any) -> str:
        """Normalize expert method identifiers into plain strings."""
        if isinstance(method_value, Enum):
            raw_val = getattr(method_value, 'value', None)
            return raw_val if isinstance(raw_val, str) else str(method_value)
        return '' if method_value is None else str(method_value)

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

    def _calculate_pair_geometry(
        self,
        star1: Dict[str, Any],
        star2: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float]]:
        ra1 = self._safe_gaia_value(star1, 'ra')
        dec1 = self._safe_gaia_value(star1, 'dec')
        ra2 = self._safe_gaia_value(star2, 'ra')
        dec2 = self._safe_gaia_value(star2, 'dec')

        if None in (ra1, dec1, ra2, dec2):
            return None, None

        try:
            coord1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
            coord2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
            separation = coord1.separation(coord2).to(u.arcsec).value
            position_angle = coord1.position_angle(coord2).to(u.deg).value
            return separation, position_angle
        except Exception as exc:
            log.debug("Failed to compute pair geometry via SkyCoord: %s", exc)

        # Fallback to small-angle approximation if astropy evaluation failed
        delta_ra = (ra1 - ra2) * np.cos(np.radians((dec1 + dec2) / 2.0))
        delta_dec = dec1 - dec2
        separation_deg = np.sqrt(delta_ra**2 + delta_dec**2)
        separation_arcsec = separation_deg * 3600.0

        # Position angle fallback using arctangent
        pa_rad = np.arctan2(
            np.sin(np.radians(ra2 - ra1)) * np.cos(np.radians(dec2)),
            np.cos(np.radians(dec1)) * np.sin(np.radians(dec2))
            - np.sin(np.radians(dec1)) * np.cos(np.radians(dec2)) * np.cos(np.radians(ra2 - ra1))
        )
        pa_deg = (np.degrees(pa_rad) + 360.0) % 360.0

        return separation_arcsec, pa_deg

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
            # Relaxed RUWE handling for wide binaries: warn but keep moderately poor solutions
            ruwe = star.get('ruwe')
            if ruwe is not None:
                hard_ruwe_limit = GAIA_RUWE_RELAX_THRESHOLD * GAIA_RUWE_TOLERANCE_MULTIPLIER * 3.0
                if ruwe > hard_ruwe_limit:
                    log.debug(
                        f"Source {star.get('source_id', 'unknown')} rejected: RUWE {ruwe:.2f} exceeds hard limit {hard_ruwe_limit:.1f}"
                    )
                    return False
                if ruwe > GAIA_MAX_RUWE * GAIA_RUWE_PERMISSIVE_MULTIPLIER:
                    log.debug(
                        f"Source {star.get('source_id', 'unknown')} has high RUWE {ruwe:.2f}; allowing with relaxed separation tolerance"
                    )
            
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
        wds_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Retrieve parallax data for mass calculations.
        
        This method queries Gaia using the stored source identifiers and returns
        the best available parallax measurement for mass calculations.
        
        Args:
            wds_summary: WDS summary data containing coordinates
            
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
            gaia_source_ids = self._extract_gaia_source_ids(wds_summary)
            if not gaia_source_ids:
                raise ParallaxDataUnavailableError("Gaia source IDs required for parallax query")

            try:
                gaia_data = await self._query_gaia_by_source_ids_async(gaia_source_ids)
            except InsufficientAstrometricDataError as exc:
                raise ParallaxDataUnavailableError(str(exc)) from exc

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
                'g_mag': float(best_star.get('phot_g_mean_mag', np.nan))
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