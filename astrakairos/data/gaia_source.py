from astroquery.gaia import Gaia
import numpy as np
from scipy.stats import chi2
from typing import Tuple, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime

# Import configuration constants
from ..config import (
    DEFAULT_GAIA_TABLE, DEFAULT_GAIA_SEARCH_RADIUS_ARCSEC, DEFAULT_GAIA_MAG_LIMIT,
    DEFAULT_GAIA_MAX_ROWS, DEFAULT_PHYSICAL_P_VALUE_THRESHOLD, 
    DEFAULT_AMBIGUOUS_P_VALUE_THRESHOLD,
    GAIA_MAX_RETRY_ATTEMPTS, GAIA_RETRY_DELAY_SECONDS,
    MIN_PARALLAX_SIGNIFICANCE,
    GAIA_MAX_RUWE, GAIA_DEFAULT_CORRELATION_MISSING
)

# Import source types and enums
from ..data.source import PhysicalityValidator, PhysicalityLabel, ValidationMethod, PhysicalityAssessment

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
                 max_sources: int = DEFAULT_GAIA_MAX_ROWS):
        """
        Initializes the Gaia validator with configuration parameters.

        Args:
            gaia_table: The Gaia data release table to query (configurable for future releases)
            default_search_radius_arcsec: Default search radius for Gaia queries
            physical_p_value_threshold: The p-value above which a pair is considered
                                        'Likely Physical'. Standard: 0.05 (5%)
            ambiguous_p_value_threshold: The p-value above which a pair is 'Ambiguous'.
                                         Below this: 'Likely Optical'. Standard: 0.001 (0.1%)
            mag_limit: G-band magnitude limit for Gaia queries (avoids faint noise)
            max_sources: Maximum number of sources to retrieve (prevents memory issues)
            
        Raises:
            ValueError: If thresholds are not in valid order
            
        Notes:
            - Configuration is instance-specific for thread safety
        """
        # Store configuration instance-specific instead of modifying global state
        self.gaia_table = gaia_table
        self.default_search_radius_arcsec = default_search_radius_arcsec
        self.mag_limit = mag_limit
        self.max_sources = max_sources

        # Global astroquery configuration kept minimal - only table specified per query
        self.physical_threshold = physical_p_value_threshold
        self.ambiguous_threshold = ambiguous_p_value_threshold
        
        # Simple in-memory cache for repeated queries within same session
        self._query_cache = {}
        
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
                                 wds_summary: Dict[str, Any],
                                 search_radius_arcsec: Optional[float] = None) -> Optional[PhysicalityAssessment]:
        """
        Validates if a binary system is physically bound using Gaia data.
        
        Args:
            wds_summary: WDS summary data containing coordinates and magnitudes
            search_radius_arcsec: Optional override for search radius
            
        Returns:
            PhysicalityAssessment object or None if validation fails
        """
        try:
            # Extract coordinates and magnitudes from WDS summary
            ra_deg = wds_summary.get('ra_deg')
            dec_deg = wds_summary.get('dec_deg')
            mag_pri = wds_summary.get('mag_pri')
            mag_sec = wds_summary.get('mag_sec')
            
            if ra_deg is None or dec_deg is None:
                return self._create_assessment(PhysicalityLabel.UNKNOWN, search_radius_arcsec)
            
            final_radius = search_radius_arcsec if search_radius_arcsec is not None else self.default_search_radius_arcsec
            wds_magnitudes = (mag_pri, mag_sec)
            
            # Query Gaia async for I/O operations 
            gaia_results = await self._query_gaia_for_pair_async(ra_deg, dec_deg, final_radius)
            if gaia_results is None or len(gaia_results) < 2:
                return self._create_assessment(PhysicalityLabel.UNKNOWN, search_radius_arcsec)
            
            # CPU work: run directly without executor (GIL prevents threading benefits)
            result = self._validate_physicality_sync(gaia_results, wds_magnitudes)
            
            # Convert result to PhysicalityAssessment
            return self._create_final_assessment(result, gaia_results, wds_magnitudes, search_radius_arcsec)
            
        except Exception as e:
            log.error(f"Error during Gaia validation for coordinates ({ra_deg:.4f}, {dec_deg:.4f}): {e}")
            return self._create_assessment(PhysicalityLabel.UNKNOWN, search_radius_arcsec)
    
    def _create_assessment(self, label: PhysicalityLabel, search_radius_arcsec: Optional[float] = None, 
                          p_value: Optional[float] = None, method: ValidationMethod = None,
                          gaia_primary: Optional[str] = None, gaia_secondary: Optional[str] = None) -> PhysicalityAssessment:
        """Create a PhysicalityAssessment object."""
        if method is None:
            method = ValidationMethod.INSUFFICIENT_DATA
            
        return {
            'label': label,
            'confidence': 1.0 - p_value if p_value else 0.0,
            'p_value': p_value,
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
                              search_radius_arcsec: Optional[float]) -> PhysicalityAssessment:
        """Convert validation result to final PhysicalityAssessment."""
        # Get Gaia source IDs if available
        primary_gaia, secondary_gaia = self._identify_components_by_mag(gaia_results, wds_magnitudes)
        gaia_primary_id = primary_gaia.get('source_id') if primary_gaia else None
        gaia_secondary_id = secondary_gaia.get('source_id') if secondary_gaia else None
        
        return self._create_assessment(
            label=result['label'],  # Already an enum
            p_value=result['p_value'],
            method=result['method'],  # Already an enum
            gaia_primary=gaia_primary_id,
            gaia_secondary=gaia_secondary_id,
            search_radius_arcsec=search_radius_arcsec
        )
    
    def _validate_physicality_sync(self,
                                  gaia_results,
                                  wds_mags: Tuple[Optional[float], Optional[float]]) -> Dict[str, Any]:
        """
        Orchestrates the adaptive physicality validation logic and returns a
        structured dictionary with the results.
        
        Args:
            gaia_results: Pre-fetched Gaia query results
            wds_mags: WDS magnitudes for component matching
        """
        # Filter by astrometric quality first
        quality_filtered = [star for star in gaia_results if self._validate_astrometric_quality(star)]
        
        if len(quality_filtered) < 2:
            log.warning(f"Only {len(quality_filtered)} quality sources after filtering from {len(gaia_results)} total")
            return {'label': PhysicalityLabel.UNKNOWN, 'p_value': None, 'method': ValidationMethod.INSUFFICIENT_DATA}
        
        primary_gaia, secondary_gaia = self._identify_components_by_mag(quality_filtered, wds_mags)
        if primary_gaia is None or secondary_gaia is None:
            # Check if it's due to insufficient sources or genuine component matching failure
            if len(quality_filtered) < 2:
                return {'label': PhysicalityLabel.UNKNOWN, 'p_value': None, 'method': ValidationMethod.INSUFFICIENT_DATA}
            else:
                return {'label': PhysicalityLabel.AMBIGUOUS, 'p_value': None, 'method': ValidationMethod.INSUFFICIENT_DATA}
        
        chi2_result = self._calculate_chi2_3d(primary_gaia, secondary_gaia)
        if chi2_result:
            test_type = ValidationMethod.GAIA_3D_PARALLAX_PM
        else:
            chi2_result = self._calculate_chi2_2d_pm(primary_gaia, secondary_gaia)
            if chi2_result:
                test_type = ValidationMethod.PROPER_MOTION_ONLY
            else:
                chi2_result = self._calculate_chi2_1d_plx(primary_gaia, secondary_gaia)
                if chi2_result:
                    test_type = ValidationMethod.GAIA_PARALLAX_ONLY

        if not chi2_result:
            return {'label': PhysicalityLabel.UNKNOWN, 'p_value': None, 'method': ValidationMethod.INSUFFICIENT_DATA}

        chi_squared_val, dof = chi2_result
        p_value = 1.0 - chi2.cdf(chi_squared_val, df=dof)
        
        if p_value > self.physical_threshold:
            label = PhysicalityLabel.LIKELY_PHYSICAL
        elif p_value > self.ambiguous_threshold:
            label = PhysicalityLabel.AMBIGUOUS
        else:
            label = PhysicalityLabel.LIKELY_OPTICAL
        
        return {'label': label, 'p_value': p_value, 'method': test_type}

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

    def _calculate_chi2_3d(self, star1: Dict, star2: Dict) -> Optional[Tuple[float, int]]:
        """Calculates 3D chi-squared (plx, pmra, pmdec) using unified covariance builder."""
        required_keys = ['parallax', 'pmra', 'pmdec']
        if not (self._get_params_and_check_validity(star1, required_keys) and 
                self._get_params_and_check_validity(star2, required_keys)):
            return None
        
        try:
            params1 = np.array([star1['parallax'], star1['pmra'], star1['pmdec']])
            params2 = np.array([star2['parallax'], star2['pmra'], star2['pmdec']])
            
            # Use unified covariance matrix builder
            C1 = self._build_covariance_matrix(star1, dimensions=3)
            C2 = self._build_covariance_matrix(star2, dimensions=3)
            
            if C1 is None or C2 is None:
                return None
            
            C_total_inv = np.linalg.inv(C1 + C2)
            delta_params = params1 - params2
            chi_squared = delta_params.T @ C_total_inv @ delta_params
            return chi_squared, 3
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            return None

    def _calculate_chi2_2d_pm(self, star1: Dict, star2: Dict) -> Optional[Tuple[float, int]]:
        """Calculates 2D chi-squared for proper motion using unified covariance builder."""
        required_keys = ['pmra', 'pmdec']
        if not (self._get_params_and_check_validity(star1, required_keys) and 
                self._get_params_and_check_validity(star2, required_keys)):
            return None

        try:
            params1 = np.array([star1['pmra'], star1['pmdec']])
            params2 = np.array([star2['pmra'], star2['pmdec']])
            
            # Use unified covariance matrix builder
            C1 = self._build_covariance_matrix(star1, dimensions=2)
            C2 = self._build_covariance_matrix(star2, dimensions=2)
            
            if C1 is None or C2 is None:
                return None

            C_total_inv = np.linalg.inv(C1 + C2)
            delta_params = params1 - params2
            chi_squared = delta_params.T @ C_total_inv @ delta_params
            return chi_squared, 2
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            return None

    def _calculate_chi2_1d_plx(self, star1: Dict, star2: Dict) -> Optional[Tuple[float, int]]:
        """Calculates 1D chi-squared for parallax using unified approach."""
        required_keys = ['parallax']
        if not (self._get_params_and_check_validity(star1, required_keys) and 
                self._get_params_and_check_validity(star2, required_keys)):
            return None
        
        try:
            # Use unified covariance matrix builder for consistency
            C1 = self._build_covariance_matrix(star1, dimensions=1)
            C2 = self._build_covariance_matrix(star2, dimensions=1)
            
            if C1 is None or C2 is None:
                return None
                
            plx1, plx2 = star1['parallax'], star2['parallax']
            combined_variance = C1[0, 0] + C2[0, 0]  # Sum of variances
            
            if combined_variance <= 0:
                return None
            
            chi_squared = ((plx1 - plx2)**2) / combined_variance
            return chi_squared, 1
        except (ValueError, ZeroDivisionError, IndexError):
            return None

    def _identify_components_by_mag(self, gaia_results, wds_mags: Tuple[Optional[float], Optional[float]]):
        """
        Identifies the primary and secondary components from a list of Gaia results
        using improved algorithm that considers both magnitude and spatial proximity.
        
        Returns:
            Tuple of (primary_gaia, secondary_gaia) or (None, None) if insufficient sources
        """
        # Check if we have enough sources
        if len(gaia_results) < 2:
            log.warning(f"Only {len(gaia_results)} Gaia source(s) found - need at least 2")
            return None, None
            
        mag_pri_wds, mag_sec_wds = wds_mags

        if mag_pri_wds is None or mag_sec_wds is None:
            # If WDS magnitudes are not available, return the two closest sources
            # (they are already sorted by brightness from the query)
            return self._select_closest_pair(gaia_results)

        # Use improved matching algorithm
        return self._match_components_improved(gaia_results, mag_pri_wds, mag_sec_wds)
    
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
    
    def _match_components_improved(self, gaia_results, mag_pri_wds, mag_sec_wds):
        """Improved component matching using magnitude + spatial proximity."""
        # First pass: Find candidates within reasonable magnitude tolerance
        mag_tolerance = 1.5  # magnitudes
        
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
                spatial_penalty = min(sep_arcsec / 10.0, 5.0)  # Cap at 5 for very wide pairs
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
                    
                # If no results and more attempts remain, wait before retry
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
        """
        try:
            # Query using instance configuration to avoid modifying global state
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
            ) AND phot_g_mean_mag < {self.mag_limit}
              AND astrometric_chi2_al IS NOT NULL
              AND ruwe < {GAIA_MAX_RUWE}
            ORDER BY phot_g_mean_mag ASC
            """
            
            log.debug(f"Querying Gaia at ({ra_deg:.4f}, {dec_deg:.4f}) "
                     f"radius {radius_arcsec:.1f}\" (attempt {attempt})")
            
            # Use instance-specific limits to avoid global state modification
            job = Gaia.launch_job(query)
            results = job.get_results()
            
            if len(results) == 0:
                return None
            elif len(results) < 2:
                log.warning(f"Only {len(results)} Gaia source found - need at least 2")
                return None
            else:
                return results[:self.max_sources]  # Limit to configured maximum
                
        except Exception as e:
            # Don't handle retries here - let the async wrapper handle them
            raise e

    def _validate_astrometric_quality(self, star: Dict) -> bool:
        """
        Validates basic astrometric quality criteria for Gaia sources.
        Filters out sources with poor astrometric solutions.
        
        Args:
            star: Gaia source data dictionary
            
        Returns:
            True if the source meets minimum quality criteria for analysis
        """
        try:
            # Check RUWE (Renormalised Unit Weight Error) - primary filter
            ruwe = star.get('ruwe')
            if ruwe is not None and ruwe > GAIA_MAX_RUWE:
                return False
            
            # Check for extremely poor parallax measurements (filter only extreme cases)
            if 'parallax' in star.colnames and 'parallax_error' in star.colnames:
                parallax = star['parallax']
                parallax_error = star['parallax_error']
                if (parallax is not None and parallax_error is not None and 
                    parallax_error > 0 and abs(parallax / parallax_error) < MIN_PARALLAX_SIGNIFICANCE / 2):
                    # Only reject very poor parallax measurements to avoid rejecting distant binaries
                    return False
            
            # Accept all other cases - proper motion filtering too aggressive for wide binaries
            return True
            
        except Exception as e:
            log.warning(f"Error validating astrometric quality: {e}")
            return True  # Conservative approach - don't reject on validation errors

    def _build_covariance_matrix(self, star: Dict, dimensions: int = 3) -> Optional[np.ndarray]:
        """
        Builds covariance matrix for astrometric parameters with proper error handling.
        
        Args:
            star: Gaia source data
            dimensions: 1 (parallax), 2 (proper motion), or 3 (parallax + proper motion)
            
        Returns:
            Covariance matrix or None if data is insufficient
        """
        def get_correlation_safe(star: Dict, key: str) -> float:
            """
            Safe correlation retrieval with fallback default.
            
            Args:
                star: Gaia source data
                key: Correlation coefficient key
                
            Returns:
                Correlation coefficient or configured default
            """
            if hasattr(star, 'get'):
                corr = star.get(key)
            else:
                corr = star[key] if key in star.colnames else None
                
            if corr is None:
                log.debug(f"Missing correlation {key} for source {star.get('source_id', 'unknown')}, "
                         f"using default {GAIA_DEFAULT_CORRELATION_MISSING}")
                return GAIA_DEFAULT_CORRELATION_MISSING
            
            return float(corr)
        
        try:
            if dimensions == 1:
                # 1D: parallax only
                if 'parallax_error' not in star.colnames:
                    return None
                err = star['parallax_error']
                if err is None or err <= 0:
                    return None
                return np.array([[err**2]])
                
            elif dimensions == 2:
                # 2D: proper motion only
                required_keys = ['pmra_error', 'pmdec_error']
                if not all(key in star.colnames for key in required_keys):
                    return None
                    
                pmra_err = star['pmra_error']
                pmdec_err = star['pmdec_error']
                
                if pmra_err is None or pmdec_err is None or pmra_err <= 0 or pmdec_err <= 0:
                    return None
                    
                # Use safe correlation retrieval
                pmra_pmdec_corr = get_correlation_safe(star, 'pmra_pmdec_corr')
                
                C = np.zeros((2, 2))
                C[0, 0] = pmra_err**2
                C[1, 1] = pmdec_err**2
                C[0, 1] = C[1, 0] = pmra_err * pmdec_err * pmra_pmdec_corr
                return C
                
            elif dimensions == 3:
                # 3D: parallax + proper motion
                required_keys = ['parallax_error', 'pmra_error', 'pmdec_error']
                if not all(key in star.colnames for key in required_keys):
                    return None
                    
                plx_err = star['parallax_error']
                pmra_err = star['pmra_error'] 
                pmdec_err = star['pmdec_error']
                
                if (plx_err is None or pmra_err is None or pmdec_err is None or
                    plx_err <= 0 or pmra_err <= 0 or pmdec_err <= 0):
                    return None
                
                # Use safe correlation retrieval
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

    async def get_parallax_data(
        self,
        wds_summary: Dict[str, Any],
        search_radius_arcsec: float = 10.0
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve parallax data for mass calculations.
        
        This method queries Gaia around the system position and returns
        the best available parallax measurement for mass calculations.
        
        Args:
            wds_summary: WDS summary data containing coordinates
            search_radius_arcsec: Search radius around system position
            
        Returns:
            Dict or None
                Parallax information including:
                - parallax: value in mas
                - parallax_error: uncertainty in mas
                - source: 'gaia_dr3'
                - gaia_source_id: Gaia source identifier
                - ruwe: Renormalised Unit Weight Error
                - significance: parallax/parallax_error ratio
                - g_mag: G-band magnitude
        """
        try:
            # Get system coordinates
            ra_deg = wds_summary.get('ra_deg')
            dec_deg = wds_summary.get('dec_deg')
            
            if ra_deg is None or dec_deg is None:
                log.debug("Missing coordinates for parallax query")
                return None
                
            # Query Gaia around the position
            gaia_data = await self._query_gaia_for_pair_async(ra_deg, dec_deg, search_radius_arcsec)
            
            if not gaia_data or len(gaia_data) == 0:
                log.debug("No Gaia sources found for parallax query")
                return None
                
            # Select best parallax source
            best_star = self._select_best_parallax_source(gaia_data)
            
            if not best_star:
                log.debug("No suitable parallax source found")
                return None
                
            # Extract parallax data
            parallax = best_star.get('parallax')
            parallax_error = best_star.get('parallax_error')
            
            if parallax is None or parallax_error is None:
                log.debug("Missing parallax or parallax_error in Gaia data")
                return None
                
            # Check parallax significance (at least 3-sigma detection)
            significance = parallax / parallax_error if parallax_error > 0 else 0.0
            
            if significance < 3.0:
                log.debug(f"Low parallax significance: {significance:.2f}")
                return None
                
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
            
        except Exception as e:
            log.error(f"Error retrieving parallax data: {e}")
            return None
    
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
                
                if significance >= 3.0:  # At least 3-sigma detection
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
        # 3. Brighter magnitude (more reliable)
        def quality_score(source):
            ruwe_score = 1.0 if source['ruwe'] <= 1.4 else 0.5
            sig_score = min(source['significance'] / 10.0, 1.0)  # Normalize to [0,1]
            mag_score = max(0.0, (20.0 - source['phot_g_mean_mag']) / 20.0)  # Brighter is better
            
            return ruwe_score * 0.4 + sig_score * 0.4 + mag_score * 0.2
        
        # Select source with highest quality score
        best_source = max(valid_sources, key=quality_score)
        
        log.debug(f"Selected parallax source: ID={best_source['source_id']}, "
                 f"π={best_source['parallax']:.3f}±{best_source['parallax_error']:.3f} mas, "
                 f"significance={best_source['significance']:.1f}, "
                 f"RUWE={best_source['ruwe']:.2f}")
        
        return best_source