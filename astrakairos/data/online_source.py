import logging
import asyncio
from typing import Optional
import numpy as np

from astropy.table import Table
from astroquery.vizier import Vizier
from astroquery.exceptions import TimeoutError as AstroqueryTimeoutError

from .source import DataSource, WdsSummary, OrbitalElements, PhysicalityAssessment
from ..utils.io import parse_wds_designation
from ..config import (
    DEFAULT_VIZIER_ROW_LIMIT, DEFAULT_VIZIER_TIMEOUT, DEFAULT_VIZIER_RETRY_ATTEMPTS,
    DEFAULT_VIZIER_RETRY_DELAY, VIZIER_WDS_CATALOG, VIZIER_ORBITAL_CATALOG,
    VIZIER_BACKOFF_BASE, VIZIER_BACKOFF_MAX_DELAY, VIZIER_WDS_COLUMNS, 
    VIZIER_ORBITAL_COLUMNS
)

log = logging.getLogger(__name__)

class OnlineDataSource(DataSource):
    """
    Online data source implementation using VizieR service.
    
    This implementation uses the VizieR service to access official astronomical
    catalogs (WDS, ORB6) programmatically. It provides a robust interface for
    retrieving canonical data with proper error handling and validation.
    
    Note: This source does not support complete measurement history retrieval
    due to VizieR limitations. Use LocalDataSource for comprehensive
    historical analysis.
    """

    def __init__(self, 
                 vizier_row_limit: int = DEFAULT_VIZIER_ROW_LIMIT,
                 vizier_timeout: int = DEFAULT_VIZIER_TIMEOUT):
        """
        Initialize the online data source.

        Args:
            vizier_row_limit: Maximum number of rows to retrieve from VizieR queries.
            vizier_timeout: Timeout in seconds for VizieR queries.
        """
        self.vizier_row_limit = vizier_row_limit
        self.vizier_timeout = vizier_timeout

    async def get_wds_summary(self, wds_id: str) -> Optional[WdsSummary]:
        """
        Fetches WDS summary data for a star from the official WDS catalog.
        """
        log.info(f"Querying VizieR ({VIZIER_WDS_CATALOG}) for summary of {wds_id}...")
        try:
            # Create thread-safe VizieR instance for this specific query
            vizier = Vizier(
                row_limit=self.vizier_row_limit,
                timeout=self.vizier_timeout,
                catalog=VIZIER_WDS_CATALOG,
                columns=VIZIER_WDS_COLUMNS
            )
            
            result_tables = await self._retry_vizier_query(
                vizier.query_constraints, WDS=f"={wds_id}"
            )
            
            if result_tables and len(result_tables[0]) > 0:
                star_data = result_tables[0][0]
                log.debug(f"Found WDS data for {wds_id}: {len(result_tables[0])} entries")
                
                # Safely extract data with type conversion and validation
                try:
                    # Use the canonical WDS identifier from VizieR and validate format
                    canonical_wds_id = str(star_data['WDS']).strip()
                    
                    # Validate WDS ID format using centralized function
                    if not parse_wds_designation(canonical_wds_id):
                        log.warning(f"Invalid WDS ID format from VizieR: {canonical_wds_id}")
                        return None
                    
                    summary = WdsSummary(
                        wds_id=canonical_wds_id,
                        ra_deg=extract_float_value(star_data['RAJ2000']),
                        dec_deg=extract_float_value(star_data['DEJ2000']),
                        date_first=extract_float_value(star_data['Obs1']),
                        date_last=extract_float_value(star_data['Obs2']),
                        obs=extract_int_value(star_data['Nobs']) or 0,
                        pa_first=extract_float_value(star_data['pa1']),
                        pa_last=extract_float_value(star_data['pa2']),
                        sep_first=extract_float_value(star_data['sep1']),
                        sep_last=extract_float_value(star_data['sep2']),
                        mag_pri=extract_float_value(star_data['mag1']),
                        mag_sec=extract_float_value(star_data['mag2']),
                        spec_type=extract_string_value(star_data['SpType'])
                    )
                    log.info(f"Successfully extracted WDS summary for {canonical_wds_id}")
                    return summary
                    
                except (ValueError, TypeError) as e:
                    log.error(f"Error converting WDS data for {wds_id}: {e}")
                    log.debug(f"Problematic data fields: {dict(star_data)}")
                    return None

            log.info(f"No WDS summary found on VizieR for {wds_id}.")
            return None
        
        except Exception as e:
            log.error(f"Failed to fetch WDS summary for {wds_id} from VizieR: {e}")
            return None

    async def get_all_measurements(self, wds_id: str) -> Optional[Table]:
        """
        Fetches all historical measurements for a star from the VizieR service.
        
        Note: VizieR does not provide access to complete historical measurements
        for double stars programmatically. This method returns None and logs
        the limitation. For comprehensive measurement analysis, use LocalDataSource
        with full catalog files.
        """
        log.info(f"Measurement retrieval for {wds_id} not supported via VizieR.")
        log.info("For complete historical measurements, use LocalDataSource with full catalog files.")
        return None

    async def get_orbital_elements(self, wds_id: str) -> Optional[OrbitalElements]:
        """
        Fetches orbital elements for a star from the most recent orbital catalog 
        (configurable ORB6 catalog) on VizieR.
        """
        log.info(f"Querying VizieR ({VIZIER_ORBITAL_CATALOG}) for orbital elements of {wds_id}...")
        try:
            # Create thread-safe VizieR instance for this specific query
            vizier = Vizier(
                row_limit=self.vizier_row_limit,
                timeout=self.vizier_timeout,
                catalog=VIZIER_ORBITAL_CATALOG,
                columns=VIZIER_ORBITAL_COLUMNS
            )
            
            result_tables = await self._retry_vizier_query(
                vizier.query_constraints, WDS=f"={wds_id}"
            )
            
            if not result_tables or len(result_tables[0]) == 0:
                log.info(f"No published orbit found in orbital catalog for {wds_id}.")
                return None
            
            log.debug(f"Found orbital data for {wds_id}: {len(result_tables[0])} entries")
            # Multiple entries may exist, we take the first (usually highest grade)
            orbit_data = result_tables[0][0]

            # Safely extract data with type conversion and unit handling
            try:
                # Use the canonical WDS identifier from VizieR and validate format
                canonical_wds_id = str(orbit_data['WDS']).strip()
                
                # Validate WDS ID format using centralized function
                if not parse_wds_designation(canonical_wds_id):
                    log.warning(f"Invalid WDS ID format in orbital catalog: {canonical_wds_id}")
                    return None

                # Safely extract and convert period with validation
                if np.ma.is_masked(orbit_data['P']) or orbit_data['P'] <= 0:
                    log.warning(f"Invalid or missing period for {wds_id}")
                    return None
                
                # Safely extract and validate semimajor axis
                if np.ma.is_masked(orbit_data['Axis']) or orbit_data['Axis'] <= 0:
                    log.warning(f"Invalid or missing semimajor axis for {wds_id}")
                    return None
                
                # Extract values directly - they're already in correct units (yr, arcsec)
                period = float(orbit_data['P'])
                semimajor_axis = float(orbit_data['Axis'])

                elements = OrbitalElements(
                    wds_id=canonical_wds_id,
                    P=period,
                    a=semimajor_axis,
                    i=extract_float_value(orbit_data['i']),
                    Omega=extract_float_value(orbit_data['Node']),
                    T=extract_float_value(orbit_data['T']),
                    e=extract_float_value(orbit_data['e']),
                    omega=extract_float_value(orbit_data['omega']),
                    
                    # Errors
                    e_P=extract_float_value(orbit_data['e_P']),
                    e_a=extract_float_value(orbit_data['e_Axis']),
                )
                log.info(f"Successfully extracted orbital elements for {canonical_wds_id}")
                return elements
                
            except (ValueError, TypeError, KeyError) as e:
                log.error(f"Error converting orbital data for {wds_id}: {e}")
                log.debug(f"Problematic data fields: {dict(orbit_data)}")
                return None
                
        except Exception as e:
            log.error(f"Failed to fetch orbital elements for {wds_id} from VizieR: {e}")
            return None

    async def validate_physicality(self, system_data: WdsSummary) -> Optional[PhysicalityAssessment]:
        """
        This implementation does not validate physicality.
        
        Returns a neutral assessment indicating insufficient data for validation.
        """
        from .source import PhysicalityLabel, ValidationMethod
        
        return PhysicalityAssessment(
            label=PhysicalityLabel.UNKNOWN,
            p_value=None,
            method=ValidationMethod.INSUFFICIENT_DATA
        )


    
    async def _retry_vizier_query(self, query_func, *args, **kwargs):
        """
        Execute a VizieR query with exponential backoff retry logic.
        
        Args:
            query_func: The VizieR query function to execute
            *args, **kwargs: Arguments to pass to the query function
            
        Returns:
            Query results or None if all retries failed
        """
        for attempt in range(DEFAULT_VIZIER_RETRY_ATTEMPTS):
            try:
                result = await asyncio.to_thread(query_func, *args, **kwargs)
                return result
            except AstroqueryTimeoutError:
                log.warning(f"VizieR query timeout (attempt {attempt + 1}/{DEFAULT_VIZIER_RETRY_ATTEMPTS})")
                if attempt < DEFAULT_VIZIER_RETRY_ATTEMPTS - 1:
                    # Exponential backoff with configurable base and max delay
                    backoff_delay = min(
                        VIZIER_BACKOFF_BASE * (2 ** attempt),
                        VIZIER_BACKOFF_MAX_DELAY
                    )
                    await asyncio.sleep(backoff_delay)
            except Exception as e:
                log.error(f"VizieR query failed (attempt {attempt + 1}/{DEFAULT_VIZIER_RETRY_ATTEMPTS}): {type(e).__name__}: {e}")
                if attempt < DEFAULT_VIZIER_RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(DEFAULT_VIZIER_RETRY_DELAY)
        
        return None

# Helper functions for VizieR data extraction
def extract_float_value(masked_value) -> Optional[float]:
    """Extract float value from VizieR data."""
    if np.ma.is_masked(masked_value):
        return None
    try:
        return float(masked_value)
    except (ValueError, TypeError):
        return None

def extract_int_value(masked_value) -> Optional[int]:
    """Extract int value from VizieR data."""
    if np.ma.is_masked(masked_value):
        return None
    try:
        return int(masked_value)
    except (ValueError, TypeError):
        return None

def extract_string_value(masked_value) -> Optional[str]:
    """Extract string value from VizieR data."""
    if np.ma.is_masked(masked_value):
        return None
    try:
        return str(masked_value).strip()
    except (ValueError, TypeError):
        return None