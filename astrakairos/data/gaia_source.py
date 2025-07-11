from astroquery.gaia import Gaia
import numpy as np
from scipy.stats import chi2
from typing import Tuple, Optional, Dict, Any
import asyncio

class GaiaValidator:
    """
    Validator for physical binary systems using Gaia data.

    This class implements an adaptive chi-squared (χ²) testing strategy.
    It attempts to use the most complete astrometric data available for a pair
    of stars (3D: parallax + proper motion, 2D: proper motion only, 1D: parallax only)
    to provide the most robust possible assessment of physicality.
    """
    
    def __init__(self,
                 gaia_table: str = "gaiadr3.gaia_source",
                 default_search_radius_arcsec: float = 10.0,
                 physical_p_value_threshold: float = 0.05,
                 ambiguous_p_value_threshold: float = 0.001):
        """
        Initializes the Gaia validator with configurable statistical thresholds.

        Args:
            gaia_table: The Gaia data release table to query.
            default_search_radius_arcsec: Default search radius.
            physical_p_value_threshold: The p-value above which a pair is considered
                                        'Likely Physical'. A common choice is 0.05 (5%).
            ambiguous_p_value_threshold: The p-value above which a pair is 'Ambiguous'.
                                         Below this, it is 'Likely Optical'. A common choice is 0.001 (0.1%).
        """
        Gaia.MAIN_GAIA_TABLE = gaia_table
        Gaia.ROW_LIMIT = 10
        self.default_search_radius_arcsec = default_search_radius_arcsec
        self.physical_threshold = physical_p_value_threshold
        self.ambiguous_threshold = ambiguous_p_value_threshold
    
    async def validate_physicality(self,
                                 primary_coords_deg: Tuple[float, float],
                                 wds_magnitudes: Tuple[Optional[float], Optional[float]],
                                 search_radius_arcsec: Optional[float] = None) -> Dict[str, Any]:
        """
        Validates if a binary system is physically bound using Gaia data.
        Returns a dictionary with the assessment details.
        """
        try:
            loop = asyncio.get_event_loop()
            final_radius = search_radius_arcsec if search_radius_arcsec is not None else self.default_search_radius_arcsec
            
            result = await loop.run_in_executor(
                None, 
                self._validate_physicality_sync, 
                primary_coords_deg,
                wds_magnitudes,
                final_radius
            )
            return result
        except Exception as e:
            print(f"Error during Gaia validation: {e}")
            return {'label': 'Error', 'p_value': None, 'test_used': 'None'}
    
    def _validate_physicality_sync(self,
                                  primary_coords: Tuple[float, float],
                                  wds_mags: Tuple[Optional[float], Optional[float]],
                                  radius: float) -> Dict[str, Any]:
        """
        Orchestrates the adaptive physicality validation logic and returns a
        structured dictionary with the results.
        """
        gaia_results = self._query_gaia_for_pair(primary_coords[0], primary_coords[1], radius)
        if gaia_results is None or len(gaia_results) < 2:
            return {'label': 'Unknown', 'p_value': None, 'test_used': 'Not enough Gaia sources'}
        
        primary_gaia, secondary_gaia = self._identify_components_by_mag(gaia_results, wds_mags)
        if primary_gaia is None or secondary_gaia is None:
            return {'label': 'Ambiguous', 'p_value': None, 'test_used': 'Component matching failed'}
        
        # --- Adaptive Chi-Squared Testing Strategy ---
        test_map = {3: '3D (plx+pm)', 2: '2D (pm_only)', 1: '1D (plx_only)'}
        chi2_result = None
        
        # 1. Attempt 3D test
        chi2_result = self._calculate_chi2_3d(primary_gaia, secondary_gaia)
        if chi2_result:
            test_type = test_map[3]
        
        # 2. Attempt 2D test if 3D failed
        if not chi2_result:
            chi2_result = self._calculate_chi2_2d_pm(primary_gaia, secondary_gaia)
            if chi2_result:
                test_type = test_map[2]
        
        # 3. Attempt 1D test if 2D also failed
        if not chi2_result:
            chi2_result = self._calculate_chi2_1d_plx(primary_gaia, secondary_gaia)
            if chi2_result:
                test_type = test_map[1]
        
        # 4. If no test could be performed, return Unknown
        if not chi2_result:
            return {'label': 'Unknown', 'p_value': None, 'test_used': 'Incomplete astrometry'}

        # --- Calculate p-value and determine label ---
        chi_squared_val, dof = chi2_result
        p_value = 1.0 - chi2.cdf(chi_squared_val, df=dof)
        
        # Determine the label based on configurable thresholds
        if p_value > self.physical_threshold:
            label = 'Likely Physical'
        elif p_value > self.ambiguous_threshold:
            label = 'Ambiguous'
        else:
            label = 'Likely Optical'
        
        return {
            'label': label,
            'p_value': p_value,
            'test_used': test_type
        }

    def _get_params_and_check_validity(self, star: Dict, keys: list) -> bool:
        """Helper to check if all necessary keys exist and are valid for a star."""
        return all(key in star and star[key] is not None and not np.ma.is_masked(star[key]) for key in keys)

    def _calculate_chi2_3d(self, star1: Dict, star2: Dict) -> Optional[Tuple[float, int]]:
        """Calculates 3D chi-squared (plx, pmra, pmdec)."""
        required_keys = ['parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
                         'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr']
        if not (self._get_params_and_check_validity(star1, required_keys) and self._get_params_and_check_validity(star2, required_keys)):
            return None
        
        try:
            params1 = np.array([star1['parallax'], star1['pmra'], star1['pmdec']])
            params2 = np.array([star2['parallax'], star2['pmra'], star2['pmdec']])
            C1 = np.zeros((3, 3))
            C1[0, 0] = star1['parallax_error']**2; C1[1, 1] = star1['pmra_error']**2; C1[2, 2] = star1['pmdec_error']**2
            C1[0, 1] = C1[1, 0] = star1['parallax_error'] * star1['pmra_error'] * star1['parallax_pmra_corr']
            C1[0, 2] = C1[2, 0] = star1['parallax_error'] * star1['pmdec_error'] * star1['parallax_pmdec_corr']
            C1[1, 2] = C1[2, 1] = star1['pmra_error'] * star1['pmdec_error'] * star1['pmra_pmdec_corr']
            C2 = np.zeros((3, 3))
            C2[0, 0] = star2['parallax_error']**2; C2[1, 1] = star2['pmra_error']**2; C2[2, 2] = star2['pmdec_error']**2
            C2[0, 1] = C2[1, 0] = star2['parallax_error'] * star2['pmra_error'] * star2['parallax_pmra_corr']
            C2[0, 2] = C2[2, 0] = star2['parallax_error'] * star2['pmdec_error'] * star2['parallax_pmdec_corr']
            C2[1, 2] = C2[2, 1] = star2['pmra_error'] * star2['pmdec_error'] * star2['pmra_pmdec_corr']
            
            C_total_inv = np.linalg.inv(C1 + C2)
            delta_params = params1 - params2
            chi_squared = delta_params.T @ C_total_inv @ delta_params
            return chi_squared, 3
        except (np.linalg.LinAlgError, ValueError):
            return None

    def _calculate_chi2_2d_pm(self, star1: Dict, star2: Dict) -> Optional[Tuple[float, int]]:
        """Calculates 2D chi-squared for proper motion (pmra, pmdec)."""
        required_keys = ['pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'pmra_pmdec_corr']
        if not (self._get_params_and_check_validity(star1, required_keys) and self._get_params_and_check_validity(star2, required_keys)):
            return None

        try:
            params1 = np.array([star1['pmra'], star1['pmdec']])
            params2 = np.array([star2['pmra'], star2['pmdec']])
            C1 = np.zeros((2, 2))
            C1[0, 0] = star1['pmra_error']**2; C1[1, 1] = star1['pmdec_error']**2
            C1[0, 1] = C1[1, 0] = star1['pmra_error'] * star1['pmdec_error'] * star1['pmra_pmdec_corr']
            C2 = np.zeros((2, 2))
            C2[0, 0] = star2['pmra_error']**2; C2[1, 1] = star2['pmdec_error']**2
            C2[0, 1] = C2[1, 0] = star2['pmra_error'] * star2['pmdec_error'] * star2['pmra_pmdec_corr']

            C_total_inv = np.linalg.inv(C1 + C2)
            delta_params = params1 - params2
            chi_squared = delta_params.T @ C_total_inv @ delta_params
            return chi_squared, 2
        except (np.linalg.LinAlgError, ValueError):
            return None

    def _calculate_chi2_1d_plx(self, star1: Dict, star2: Dict) -> Optional[Tuple[float, int]]:
        """Calculates 1D chi-squared for parallax (plx)."""
        required_keys = ['parallax', 'parallax_error']
        if not (self._get_params_and_check_validity(star1, required_keys) and self._get_params_and_check_validity(star2, required_keys)):
            return None
        
        try:
            plx1, plx1_err = star1['parallax'], star1['parallax_error']
            plx2, plx2_err = star2['parallax'], star2['parallax_error']
            
            # Simple sigma comparison is equivalent to 1D chi-squared
            combined_err = np.sqrt(plx1_err**2 + plx2_err**2)
            if combined_err == 0: return None
            
            chi_squared = ((plx1 - plx2) / combined_err)**2
            return chi_squared, 1
        except (ValueError, ZeroDivisionError):
            return None

    def _identify_components_by_mag(self, gaia_results, wds_mags: Tuple[Optional[float], Optional[float]]):
        """
        Identifies the primary and secondary components from a list of Gaia results
        by comparing their magnitudes to the WDS catalog magnitudes.
        """
        mag_pri_wds, mag_sec_wds = wds_mags

        if mag_pri_wds is None or mag_sec_wds is None:
            # If WDS magnitudes are not available, assume the brightest Gaia source
            # is the primary and the second brightest is the secondary.
            return gaia_results[0], gaia_results[1]

        # Find the Gaia source closest in magnitude to the WDS primary
        # and the one closest to the WDS secondary.
        # This is more robust than assuming the brightest is always the primary. - It's not perfect however,
        # as many things can happen with magnitude data, but it should be reliable enough for most cases.
        
        # Calculate the magnitude difference for each Gaia source to the WDS primary and secondary
        best_pri_match, best_sec_match = None, None
        min_pri_diff, min_sec_diff = np.inf, np.inf
        
        for source in gaia_results:
            g_mag = source['phot_g_mean_mag']
            if g_mag is None or np.ma.is_masked(g_mag): continue

            pri_diff = abs(g_mag - mag_pri_wds)
            sec_diff = abs(g_mag - mag_sec_wds)

            if pri_diff < min_pri_diff:
                min_pri_diff = pri_diff
                best_pri_match = source

            if sec_diff < min_sec_diff:
                min_sec_diff = sec_diff
                best_sec_match = source
        
        # Check if we didn't match the same Gaia source to both components
        if best_pri_match is not None and best_sec_match is not None and \
           best_pri_match['source_id'] == best_sec_match['source_id']:
            # Ambiguous case: the same Gaia star is the best match for both WDS components.
            # This can happen if magnitudes are very similar. Fallback to brightness order.
            return gaia_results[0], gaia_results[1]

        return best_pri_match, best_sec_match

    def _query_gaia_for_pair(self, ra_deg: float, dec_deg: float, radius_arcsec: float):
        """
        Queries Gaia for the 2 brightest stars within a given search radius.
        """
        try:
            query = f"""
            SELECT TOP 2
                source_id, ra, dec, parallax, parallax_error,
                pmra, pmra_error, pmdec, pmdec_error, phot_g_mean_mag
            FROM {Gaia.MAIN_GAIA_TABLE}
            WHERE 1=CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_arcsec / 3600.0})
            )
            ORDER BY phot_g_mean_mag ASC
            """
            
            job = Gaia.launch_job(query)
            results = job.get_results()
            
            return results if len(results) > 0 else None
                
        except Exception as e:
            print(f"Gaia query error: {e}")
            return None