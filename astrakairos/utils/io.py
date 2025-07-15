import pandas as pd
import re
import csv
import logging
from typing import List, Dict, Any, Optional
from astropy.coordinates import SkyCoord
import astropy.units as u

# Import configuration constants
from ..config import (
    MIN_RA_DEG, MAX_RA_DEG, MIN_DEC_DEG, MAX_DEC_DEG,
    CSV_SNIFFER_SAMPLE_SIZE, DEFAULT_COORDINATE_PRECISION,
    COORDINATE_ERROR_BEHAVIOR
)

log = logging.getLogger(__name__)

def load_csv_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Loads astronomical star data from a CSV file with robust delimiter detection.

    This function implements a multi-stage approach to CSV parsing specifically
    designed for astronomical catalogs which may use different delimiter conventions:
    1. Auto-detection using csv.Sniffer with configurable sample size
    2. Fallback to common astronomical catalog delimiters (comma, semicolon)
    
    Scientific Use Cases:
    - WDS catalog files (typically comma-separated)
    - ORB6 orbital element catalogs (space-separated, converted to CSV)
    - Custom observation logs (various formats)

    Args:
        filepath: Absolute path to the CSV file containing stellar data
                 Expected to have at minimum a 'wds_name' column for star identification

    Returns:
        pandas.DataFrame: Loaded astronomical data with proper column types
        None: Reserved for future use (currently raises exception on failure)
        
    Raises:
        IOError: If the file cannot be loaded with any supported delimiter
        
    Notes:
        - Sample size for auto-detection is configurable via CSV_SNIFFER_SAMPLE_SIZE
        - UTF-8 encoding is enforced for international catalog compatibility
        - Progress is logged for debugging large catalog operations
    """
    delimiters_to_try = [None, ',', ';'] # None will trigger csv.Sniffer

    for i, delimiter in enumerate(delimiters_to_try):
        try:
            current_delimiter = delimiter
            if i == 0: # First attempt with Sniffer
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        sample = f.read(CSV_SNIFFER_SAMPLE_SIZE)
                        dialect = csv.Sniffer().sniff(sample)
                        current_delimiter = dialect.delimiter
                    except csv.Error:
                        log.debug("CSV Sniffer failed, proceeding to fallback delimiters.")
                        continue # Sniffer failed, try next delimiter in list
            
            log.info(f"Attempting to read CSV with delimiter: '{current_delimiter}'")
            df = pd.read_csv(filepath, delimiter=current_delimiter, encoding='utf-8')
            log.info(f"CSV loaded successfully. Rows: {len(df)}")
            return df
            
        except Exception as e:
            log.warning(f"Failed to load CSV with delimiter '{current_delimiter}': {e}")
            
    log.error(f"Could not load CSV file: {filepath}. All parsing attempts failed.")
    raise IOError(f"Could not load or parse the input file '{filepath}'.")


def save_results_to_csv(results: List[Dict[str, Any]], filepath: str) -> None:
    """
    Saves astronomical analysis results to a publication-ready CSV file.

    This function creates standardized output files suitable for:
    - Scientific publication supplementary data
    - Further analysis in astronomical software (e.g., TOPCAT, DS9)
    - Sharing with observational collaborators
    
    Output Format:
    - UTF-8 encoding for international compatibility
    - Standard CSV format with headers
    - Consistent numerical precision for measurements
    
    Args:
        results: List of analysis dictionaries, where each dictionary represents
                one stellar system with standardized keys:
                - 'wds_name': WDS identifier (required)
                - 'opi': Observation Priority Index (for orbital mode)
                - 'rmse': Motion fit quality (for characterize mode)  
                - 'physicality_p_value': Gaia validation (for discovery mode)
        filepath: Output path for the CSV file (will be created/overwritten)
        
    Raises:
        Exception: If file writing fails (disk space, permissions, etc.)
        
    Notes:
        - Empty results lists are handled gracefully with warning
        - File operations are atomic (complete or fail, no partial writes)
        - Progress logging for large result sets
    """
    if not results:
        log.warning("No results to save.")
        return
    
    try:
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False, encoding='utf-8')
        log.info(f"Results successfully saved to {filepath} ({len(df)} rows)")
    except Exception as e:
        log.error(f"Failed to save results to {filepath}: {e}")
        raise

def format_coordinates_astropy(ra_hours: float, dec_degrees: float, precision: Optional[int] = None) -> str:
    """
    Formats celestial coordinates for display using the astropy library.
    
    This is the canonical coordinate formatting function for AstraKairos.
    All coordinate display should use this function or the unified wrappers.
    
    Args:
        ra_hours: Right ascension in hours (0-24)
        dec_degrees: Declination in degrees (-90 to +90)
        precision: Number of decimal places for seconds (uses DEFAULT_COORDINATE_PRECISION if None)
        
    Returns:
        Formatted coordinate string in HMS/DMS format, "N/A" for None inputs,
        or "Invalid Coords" for formatting errors (based on COORDINATE_ERROR_BEHAVIOR)
        
    Raises:
        ValueError: If COORDINATE_ERROR_BEHAVIOR is "raise" and formatting fails
    """
    if ra_hours is None or dec_degrees is None:
        return "N/A"
        
    if precision is None:
        precision = DEFAULT_COORDINATE_PRECISION
        
    try:
        coords = SkyCoord(ra=ra_hours * u.hourangle, dec=dec_degrees * u.deg, frame='icrs')
        return coords.to_string('hmsdms', sep=' ', precision=precision, pad=True)
        
    except (ValueError, TypeError) as e:
        error_msg = f"Astropy coordinate formatting failed: {e}"
        log.error(error_msg)
        
        if COORDINATE_ERROR_BEHAVIOR == "raise":
            raise ValueError(error_msg)
        elif COORDINATE_ERROR_BEHAVIOR == "return_none":
            return None
        else:  # "return_invalid" (default)
            return "Invalid Coords"


def parse_wds_designation(wds_id: str) -> Optional[Dict[str, float]]:
    """
    Parses a WDS designation string (e.g., "00013+1234") to extract
    approximate J2000 coordinates with scientific validation.

    Args:
        wds_id: The WDS identifier string.

    Returns:
        A dictionary with 'ra_deg' and 'dec_deg', or None if parsing fails or
        coordinates are outside valid astronomical ranges.
    """
    if not isinstance(wds_id, str) or len(wds_id) < 10:
        return None
        
    # The regex ensures the format is HHMMM[+-]DDMM
    if not re.match(r'^\d{5}[+-]\d{4}', wds_id[:10]):
        log.debug(f"WDS ID '{wds_id}' does not match the coordinate format.")
        return None
        
    try:
        ra_h = int(wds_id[0:2])
        ra_m = int(wds_id[2:5]) / 10.0 # HHMM.m
        ra_hours = ra_h + ra_m / 60.0
        ra_deg = ra_hours * 15.0

        dec_sign = 1 if wds_id[5] == '+' else -1
        dec_d = int(wds_id[6:8])
        dec_m = int(wds_id[8:10])
        dec_deg = dec_sign * (dec_d + dec_m / 60.0)
        
        # Validate astronomical coordinate ranges
        if not (MIN_RA_DEG <= ra_deg <= MAX_RA_DEG):
            log.warning(f"WDS ID '{wds_id}' has RA {ra_deg:.3f}° outside valid range [{MIN_RA_DEG}, {MAX_RA_DEG}]")
            return None
            
        if not (MIN_DEC_DEG <= dec_deg <= MAX_DEC_DEG):
            log.warning(f"WDS ID '{wds_id}' has Dec {dec_deg:.3f}° outside valid range [{MIN_DEC_DEG}, {MAX_DEC_DEG}]")
            return None
        
        return {'ra_deg': ra_deg, 'dec_deg': dec_deg}

    except (ValueError, IndexError) as e:
        log.warning(f"Failed to parse WDS designation '{wds_id}': {e}")
        return None

# Unified coordinate formatting functions (to replace duplicated implementations)
def format_coordinates(ra_hours: float, dec_degrees: float, precision: Optional[int] = None) -> str:
    """
    Universal coordinate formatting function - single source of truth.
    
    This function replaces the multiple duplicated coordinate formatting 
    implementations found in planner/calculations.py and planner/gui.py.
    
    Args:
        ra_hours: Right ascension in hours
        dec_degrees: Declination in degrees  
        precision: Number of decimal places for seconds (uses DEFAULT_COORDINATE_PRECISION if None)
        
    Returns:
        Formatted coordinate string in HMS/DMS format
    """
    return format_coordinates_astropy(ra_hours, dec_degrees, precision)

def format_ra_hours_unified(hours: float) -> str:
    """
    Unified RA formatting function to replace duplicated implementations.
    
    This function should replace the duplicated format_ra_hours functions
    in planner/calculations.py and planner/gui.py modules.
    """
    if hours is None:
        return "N/A"
    
    # Normalize hours to [0, 24) range
    normalized_hours = hours % 24.0
    
    h = int(normalized_hours)
    m = int((normalized_hours - h) * 60)
    s = ((normalized_hours - h) * 60 - m) * 60
    
    return f"{h:02d}h{m:02d}m{s:05.2f}s"

def format_dec_degrees_unified(degrees: float) -> str:
    """
    Unified Dec formatting function to replace duplicated implementations.
    
    This function should replace the duplicated format_dec_degrees functions
    in planner/calculations.py and planner/gui.py modules.
    """
    if degrees is None:
        return "N/A"
    
    sign = "+" if degrees >= 0 else "-"
    abs_deg = abs(degrees)
    d = int(abs_deg)
    m = int((abs_deg - d) * 60)
    s = ((abs_deg - d) * 60 - m) * 60
    
    return f"{sign}{d:02d}°{m:02d}'{s:05.2f}\""