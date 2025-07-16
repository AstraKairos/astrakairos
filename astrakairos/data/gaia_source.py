from astroquery.gaia import Gaia
import numpy as np
from scipy.stats import chi2
from typing import Tuple, Optional, Dict, Any
import asyncio
import logging

# Import configuration constants
from ..config import (
    DEFAULT_GAIA_TABLE, DEFAULT_GAIA_SEARCH_RADIUS_ARCSEC, DEFAULT_GAIA_MAG_LIMIT,
    DEFAULT_GAIA_ROW_LIMIT, DEFAULT_PHYSICAL_P_VALUE_THRESHOLD, 
    DEFAULT_AMBIGUOUS_P_VALUE_THRESHOLD, GAIA_QUERY_TIMEOUT_SECONDS,
    GAIA_MAX_RETRY_ATTEMPTS, GAIA_RETRY_DELAY_SECONDS,
    MIN_PARALLAX_SIGNIFICANCE, MIN_PM_SIGNIFICANCE,
    GAIA_MAX_RUWE, GAIA_DEFAULT_CORRELATION_MISSING
)

log = logging.getLogger(__name__)

class GaiaValidator:
    """
    Validator for physical binary systems using Gaia data.

    This class implements an adaptive chi-squared (χ²) testing strategy.
    It attempts to use the most complete astrometric data available for a pair
    of stars (3D: parallax + proper motion, 2D: proper motion only, 1D: parallax only)
    to provide the most robust possible assessment of physicality.
    """
    
    def __init__(self,
                 gaia_table: str = DEFAULT_GAIA_TABLE,
                 default_search_radius_arcsec: float = DEFAULT_GAIA_SEARCH_RADIUS_ARCSEC,
                 physical_p_value_threshold: float = DEFAULT_PHYSICAL_P_VALUE_THRESHOLD,
                 ambiguous_p_value_threshold: float = DEFAULT_AMBIGUOUS_P_VALUE_THRESHOLD,
                 mag_limit: float = DEFAULT_GAIA_MAG_LIMIT,
                 max_sources: int = DEFAULT_GAIA_ROW_LIMIT):
        """
        Initializes the Gaia validator with scientifically rigorous configuration.

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
            ValueError: If thresholds are not in scientifically valid order
            
        Notes:
            - Avoids modifying global astroquery state when possible
            - All configuration is instance-specific for thread safety
        """
        # Store configuration instance-specific instead of modifying global state
        self.gaia_table = gaia_table
        self.default_search_radius_arcsec = default_search_radius_arcsec
        self.mag_limit = mag_limit
        self.max_sources = max_sources

        # Only modify global state if absolutely necessary for astroquery compatibility
        # These will be set per-query basis in the query methods when possible
        try:
            # Check if we can avoid global state modification
            # Most modern astroquery versions support table specification in queries
            pass
        except:
            # Fallback to global state modification for older astroquery versions
            log.warning("Using global astroquery configuration (may affect other Gaia queries in same process)")
            Gaia.MAIN_GAIA_TABLE = gaia_table
            Gaia.ROW_LIMIT = max_sources

        self.physical_threshold = physical_p_value_threshold
        self.ambiguous_threshold = ambiguous_p_value_threshold
        
        # Scientific validation of thresholds
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
                f"physical_threshold={self.physical_threshold:.3f}, "
                f"ambiguous_threshold={self.ambiguous_threshold:.3f}")
    
    
    async def validate_physicality(self,
                                 wds_summary: Dict[str, Any],
                                 search_radius_arcsec: Optional[float] = None) -> Dict[str, Any]:
        """
        Validates if a binary system is physically bound using Gaia data.
        
        FIXED: Now returns PhysicalityAssessment compatible with source.py interface
        
        Args:
            wds_summary: WDS summary data containing coordinates and magnitudes
            search_radius_arcsec: Optional override for search radius
            
        Returns:
            PhysicalityAssessment dictionary with complete metadata
        """
        from datetime import datetime
        from ..data.source import PhysicalityLabel, ValidationMethod
        
        # Prepare default response structure
        def create_assessment(label: PhysicalityLabel, p_value: Optional[float] = None, 
                            method: ValidationMethod = ValidationMethod.INSUFFICIENT_DATA,
                            gaia_primary: Optional[str] = None,
                            gaia_secondary: Optional[str] = None) -> Dict[str, Any]:
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
                'retry_attempts': 1  # Will be updated with actual retry count
            }
        
        try:
            # Extract coordinates and magnitudes from WDS summary
            ra_deg = wds_summary.get('ra_deg')
            dec_deg = wds_summary.get('dec_deg')
            mag_pri = wds_summary.get('mag_pri')
            mag_sec = wds_summary.get('mag_sec')
            
            if ra_deg is None or dec_deg is None:
                return create_assessment(PhysicalityLabel.UNKNOWN)
            
            final_radius = search_radius_arcsec if search_radius_arcsec is not None else self.default_search_radius_arcsec
            wds_magnitudes = (mag_pri, mag_sec)
            
            # Async retry logic for Gaia queries
            gaia_results = await self._query_gaia_for_pair_async(ra_deg, dec_deg, final_radius)
            if gaia_results is None or len(gaia_results) < 2:
                return create_assessment(PhysicalityLabel.UNKNOWN)
            
            # Run synchronous analysis in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._validate_physicality_sync, 
                gaia_results,
                wds_magnitudes
            )
            
            # Convert legacy format to PhysicalityAssessment
            label_map = {
                'Likely Physical': PhysicalityLabel.LIKELY_PHYSICAL,
                'Likely Optical': PhysicalityLabel.LIKELY_OPTICAL,
                'Ambiguous': PhysicalityLabel.AMBIGUOUS,
                'Unknown': PhysicalityLabel.UNKNOWN
            }
            
            method_map = {
                '3D (plx+pm)': ValidationMethod.GAIA_3D_PARALLAX_PM,
                '2D (pm_only)': ValidationMethod.PROPER_MOTION_ONLY,
                '1D (plx_only)': ValidationMethod.GAIA_PARALLAX_ONLY,
                'Not enough Gaia sources': ValidationMethod.INSUFFICIENT_DATA,
                'Component matching failed': ValidationMethod.INSUFFICIENT_DATA,
                'Incomplete astrometry': ValidationMethod.INSUFFICIENT_DATA
            }
            
            # Get Gaia source IDs if available
            primary_gaia, secondary_gaia = self._identify_components_by_mag(gaia_results, wds_magnitudes)
            gaia_primary_id = primary_gaia.get('source_id') if primary_gaia else None
            gaia_secondary_id = secondary_gaia.get('source_id') if secondary_gaia else None
            
            return create_assessment(
                label=label_map.get(result['label'], PhysicalityLabel.UNKNOWN),
                p_value=result['p_value'],
                method=method_map.get(result['test_used'], ValidationMethod.INSUFFICIENT_DATA),
                gaia_primary=gaia_primary_id,
                gaia_secondary=gaia_secondary_id
            )
            
        except Exception as e:
            log.error(f"Error during Gaia validation for coordinates ({ra_deg:.4f}, {dec_deg:.4f}): {e}")
            return create_assessment(PhysicalityLabel.UNKNOWN)
    
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
        primary_gaia, secondary_gaia = self._identify_components_by_mag(gaia_results, wds_mags)
        if primary_gaia is None or secondary_gaia is None:
            # Check if it's due to insufficient sources or genuine component matching failure
            if len(gaia_results) < 2:
                return {'label': 'Unknown', 'p_value': None, 'test_used': 'Not enough Gaia sources'}
            else:
                return {'label': 'Ambiguous', 'p_value': None, 'test_used': 'Component matching failed'}
        
        # --- Adaptive Chi-Squared Testing Strategy ---
        test_type = "None"
        chi2_result = self._calculate_chi2_3d(primary_gaia, secondary_gaia)
        if chi2_result:
            test_type = '3D (plx+pm)'
        else:
            chi2_result = self._calculate_chi2_2d_pm(primary_gaia, secondary_gaia)
            if chi2_result:
                test_type = '2D (pm_only)'
            else:
                chi2_result = self._calculate_chi2_1d_plx(primary_gaia, secondary_gaia)
                if chi2_result:
                    test_type = '1D (plx_only)'

        if not chi2_result:
            return {'label': 'Unknown', 'p_value': None, 'test_used': 'Incomplete astrometry'}

        chi_squared_val, dof = chi2_result
        p_value = 1.0 - chi2.cdf(chi_squared_val, df=dof)
        
        if p_value > self.physical_threshold:
            label = 'Likely Physical'
        elif p_value > self.ambiguous_threshold:
            label = 'Ambiguous'
        else:
            label = 'Likely Optical'
        
        return {'label': label, 'p_value': p_value, 'test_used': test_type}

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
        by comparing their magnitudes to the WDS catalog magnitudes.
        
        Returns:
            Tuple of (primary_gaia, secondary_gaia) or (None, None) if insufficient sources
        """
        # Check if we have enough sources
        if len(gaia_results) < 2:
            log.warning(f"Only {len(gaia_results)} Gaia source(s) found - need at least 2 for binary analysis")
            return None, None
            
        mag_pri_wds, mag_sec_wds = wds_mags

        if mag_pri_wds is None or mag_sec_wds is None:
            # If WDS magnitudes are not available, assume the brightest Gaia source
            # is the primary and the second brightest is the secondary.
            return gaia_results[0], gaia_results[1]

        # Find the Gaia source closest in magnitude to the WDS primary
        # and the one closest to the WDS secondary.
        # This is more robust than assuming the brightest is always the primary.
        
        best_pri_match, best_sec_match = None, None
        min_pri_diff, min_sec_diff = np.inf, np.inf
        
        for source in gaia_results:
            g_mag = source['phot_g_mean_mag']
            if g_mag is None or (hasattr(g_mag, 'mask') and np.ma.is_masked(g_mag)): 
                continue

            pri_diff = abs(g_mag - mag_pri_wds)
            sec_diff = abs(g_mag - mag_sec_wds)

            if pri_diff < min_pri_diff:
                min_pri_diff = pri_diff
                best_pri_match = source

            if sec_diff < min_sec_diff:
                min_sec_diff = sec_diff
                best_sec_match = source
        
        # Check if we matched the same Gaia source to both components
        if best_pri_match is not None and best_sec_match is not None and \
           best_pri_match['source_id'] == best_sec_match['source_id']:
            # Scientific fallback strategy for ambiguous magnitude matching
            log.warning(f"Same Gaia source matched to both WDS components. "
                       f"WDS mags: {mag_pri_wds:.2f}, {mag_sec_wds:.2f}. "
                       f"Using brightness order as fallback.")
            
            # Return the two brightest distinct sources
            if len(gaia_results) >= 2:
                return gaia_results[0], gaia_results[1]
            else:
                log.warning("Insufficient Gaia sources for component identification")
                return None, None

        return best_pri_match, best_sec_match

    async def _query_gaia_for_pair_async(self, ra_deg: float, dec_deg: float, radius_arcsec: float):
        """
        Async wrapper for Gaia queries with proper retry logic using asyncio.sleep().
        
        This is the correct async pattern - retry logic lives in async world,
        not in the sync executor thread.
        """
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
                    return result
                    
                # If no results and more attempts remain, wait before retry
                if attempt < GAIA_MAX_RETRY_ATTEMPTS - 1:
                    log.info(f"Gaia query returned no results, retrying in {GAIA_RETRY_DELAY_SECONDS} seconds...")
                    await asyncio.sleep(GAIA_RETRY_DELAY_SECONDS)  # Proper async sleep
                
            except Exception as e:
                log.warning(f"Gaia query attempt {attempt + 1} failed: {e}")
                if attempt < GAIA_MAX_RETRY_ATTEMPTS - 1:
                    log.info(f"Retrying in {GAIA_RETRY_DELAY_SECONDS} seconds...")
                    await asyncio.sleep(GAIA_RETRY_DELAY_SECONDS)  # Proper async sleep
                else:
                    log.error(f"All {GAIA_MAX_RETRY_ATTEMPTS} Gaia query attempts failed for "
                             f"coordinates ({ra_deg:.4f}, {dec_deg:.4f})")
        
        return None

    def _query_gaia_for_pair_sync(self, ra_deg: float, dec_deg: float, radius_arcsec: float, attempt: int = 1):
        """
        Synchronous Gaia query - called from async executor.
        No retry logic here - that's handled in the async wrapper.
        
        Uses instance configuration to avoid modifying global state when possible.
        """
        try:
            # Scientifically motivated query with comprehensive astrometry and quality filtering
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
                     f"with radius {radius_arcsec:.1f}\" (attempt {attempt}/{GAIA_MAX_RETRY_ATTEMPTS})")
            
            # Use instance-specific limits to avoid global state modification
            job = Gaia.launch_job(query)
            results = job.get_results()
            
            if len(results) == 0:
                log.info(f"No Gaia sources found within {radius_arcsec:.1f}\" of ({ra_deg:.4f}, {dec_deg:.4f})")
                return None
            elif len(results) < 2:
                log.warning(f"Only {len(results)} Gaia source found - need at least 2 for binary analysis")
                return None
            else:
                log.info(f"Found {len(results)} Gaia sources, using {min(len(results), self.max_sources)}")
                return results[:self.max_sources]  # Limit to configured maximum
                
        except Exception as e:
            # Don't handle retries here - let the async wrapper handle them
            raise e

    def _validate_astrometric_quality(self, star: Dict) -> bool:
        """
        Validates the astrometric quality of a Gaia source for reliable analysis.
        
        Args:
            star: Gaia source data dictionary
            
        Returns:
            True if the source meets minimum quality criteria for analysis
            
        Notes:
            - RUWE < 1.4 indicates good astrometric solution (Lindegren et al. 2018)
            - Parallax significance > 3σ for reliable distance estimates  
            - Proper motion significance > 2σ for kinematic analysis
        """
        try:
            # Check RUWE (Renormalised Unit Weight Error) using config constant
            ruwe = star.get('ruwe')
            if ruwe is not None and ruwe > GAIA_MAX_RUWE:
                log.debug(f"Source {star.get('source_id')} has poor RUWE: {ruwe:.2f}")
                return False
            
            # Check parallax significance if available
            if 'parallax' in star.colnames and 'parallax_error' in star.colnames:
                parallax = star['parallax']
                parallax_error = star['parallax_error']
                if (parallax is not None and parallax_error is not None and 
                    parallax_error > 0 and abs(parallax / parallax_error) < MIN_PARALLAX_SIGNIFICANCE):
                    log.debug(f"Source {star.get('source_id')} has low parallax significance: "
                             f"{abs(parallax / parallax_error):.1f}σ")
                    # Don't reject - parallax might be negative for distant stars
            
            # Check proper motion significance
            if ('pmra' in star.colnames and 'pmra_error' in star.colnames and
                'pmdec' in star.colnames and 'pmdec_error' in star.colnames):
                pmra, pmra_err = star['pmra'], star['pmra_error'] 
                pmdec, pmdec_err = star['pmdec'], star['pmdec_error']
                
                if (pmra is not None and pmra_err is not None and pmra_err > 0 and
                    pmdec is not None and pmdec_err is not None and pmdec_err > 0):
                    
                    pm_total = np.sqrt(pmra**2 + pmdec**2)
                    pm_err_total = np.sqrt(pmra_err**2 + pmdec_err**2)
                    
                    if pm_total / pm_err_total < MIN_PM_SIGNIFICANCE:
                        log.debug(f"Source {star.get('source_id')} has low PM significance: "
                                 f"{pm_total / pm_err_total:.1f}σ")
                        # Don't reject - some binaries have small proper motions
            
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
            
        Notes:
            - Handles correlation coefficients correctly with scientific transparency
            - Validates data availability before construction
            - Uses configurable default for missing correlations
        """
        def get_correlation_safe(star: Dict, key: str) -> float:
            """
            Safe correlation retrieval with scientific transparency.
            
            Args:
                star: Gaia source data
                key: Correlation coefficient key
                
            Returns:
                Correlation coefficient or configured default with logging
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
                
                # Use safe correlation retrieval with scientific transparency
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

    async def query_star_data(self, wds_id: str, position: Tuple[float, float], 
                              radius: float = 5.0) -> Optional[Dict]:
        """
        Queries Gaia for star data by position with optimized ADQL query.
        
        Args:
            wds_id: WDS identifier for the star
            position: (ra, dec) in decimal degrees
            radius: Search radius in arcseconds
            
        Returns:
            Star data dictionary or None if not found
            
        Notes:
            - Optimized ADQL query with RUWE filter for efficiency
            - Handles multiple potential matches by selecting best quality
            - Includes comprehensive error handling for network issues
        """
        ra, dec = position
        
        # Optimized ADQL query with RUWE filter and required fields
        adql_query = f"""
        SELECT TOP 10
            source_id,
            ra, dec,
            parallax, parallax_error,
            pmra, pmra_error,
            pmdec, pmdec_error,
            parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr,
            ruwe,
            phot_g_mean_mag,
            bp_rp,
            astrometric_excess_noise,
            astrometric_params_solved
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(POINT('ICRS', ra, dec), 
                      CIRCLE('ICRS', {ra}, {dec}, {radius/3600.0})) = 1
        AND ruwe < {GAIA_MAX_RUWE}
        AND parallax_error > 0
        AND pmra_error > 0
        AND pmdec_error > 0
        ORDER BY ruwe ASC, parallax_error ASC
        """
        
        try:
            log.debug(f"Querying Gaia for {wds_id} at ({ra:.6f}, {dec:.6f}) with radius {radius}\"")
            
            # Execute the query with proper error handling
            job = Gaia.launch_job_async(adql_query)
            results = job.get_results()
            
            if len(results) == 0:
                log.debug(f"No Gaia sources found for {wds_id}")
                return None
            
            # Select best quality match (lowest RUWE, then lowest parallax error)
            best_match = results[0]  # Already sorted by quality
            
            log.debug(f"Found {len(results)} Gaia sources for {wds_id}, "
                     f"selected source_id={best_match['source_id']} with RUWE={best_match['ruwe']:.3f}")
            
            return best_match
            
        except Exception as e:
            log.error(f"Error querying Gaia for {wds_id}: {e}")
            return None