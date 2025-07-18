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
    COORDINATE_ERROR_BEHAVIOR, MIN_WDS_ID_LENGTH, WDS_COORDINATE_PATTERN
)

log = logging.getLogger(__name__)

class DataLoadError(Exception):
    """Exception raised when data cannot be loaded from a file."""
    pass

def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Loads astronomical star data from a CSV file with robust delimiter detection.

    This function uses pandas' built-in delimiter detection capabilities
    designed for astronomical catalogs which may use different delimiter conventions.
    
    Use Cases:
    - WDS catalog files (typically comma-separated)
    - ORB6 orbital element catalogs (space-separated, converted to CSV)
    - Custom observation logs (various formats)

    Args:
        filepath: Absolute path to the CSV file containing stellar data
                 Expected to have at minimum a 'wds_id' column for star identification

    Returns:
        pandas.DataFrame: Loaded astronomical data with proper column types
        
    Raises:
        DataLoadError: If the file cannot be loaded or parsed
        
    Notes:
        - Uses pandas' automatic delimiter detection (sep=None)
        - UTF-8 encoding with fallback to latin-1 for older catalogs
    """
    encodings_to_try = ['utf-8', 'latin-1']
    
    for encoding in encodings_to_try:
        try:
            log.info(f"Attempting to read CSV with encoding: {encoding}")
            # Let pandas handle delimiter detection automatically
            df = pd.read_csv(filepath, sep=None, engine='python', encoding=encoding)
            
            # Validate that required 'wds_id' column exists
            if 'wds_id' not in df.columns:
                raise DataLoadError(f"Required 'wds_id' column not found in {filepath}")
            
            log.info(f"CSV loaded successfully. Rows: {len(df)}, Encoding: {encoding}")
            return df
            
        except UnicodeDecodeError:
            log.debug(f"Encoding {encoding} failed, trying next...")
            continue
        except (FileNotFoundError, PermissionError) as e:
            log.error(f"File access error: {e}")
            raise DataLoadError(f"Could not access file '{filepath}': {e}")
        except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
            log.error(f"Data parsing error: {e}")
            raise DataLoadError(f"Could not parse file '{filepath}': {e}")
            
    log.error(f"Could not decode file with any supported encoding: {filepath}")
    raise DataLoadError(f"Could not decode file '{filepath}' with any supported encoding")


def save_results_to_csv(results: List[Dict[str, Any]], filepath: str) -> None:
    """
    Saves astronomical analysis results to a publication-ready CSV file.

    This function creates standardized output files suitable for:
    - Publication supplementary data
    - Further analysis in astronomical software (e.g., TOPCAT, DS9)
    - Sharing with observational collaborators
    
    Output Format:
    - UTF-8 encoding for international compatibility
    - Standard CSV format with headers
    - Consistent numerical precision for measurements
    
    Args:
        results: List of analysis dictionaries, where each dictionary represents
                one stellar system with standardized keys:
                - 'wds_id': WDS identifier (required)
                - 'opi': Observation Priority Index (for orbital mode)
                - 'rmse': Motion fit quality (for characterize mode)  
                - 'physicality_p_value': Gaia validation (for discovery mode)
        filepath: Output path for the CSV file (will be created/overwritten)
        
    Raises:
        IOError: If file writing fails (disk space, permissions, etc.)
        ValueError: If results structure is invalid
        
    Notes:
        - Empty results lists are handled gracefully with warning
        - File operations are atomic (complete or fail, no partial writes)
    """
    if not results:
        log.warning("No results to save.")
        return
    
    try:
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False, encoding='utf-8')
        log.info(f"Results successfully saved to {filepath} ({len(df)} rows)")
    except (FileNotFoundError, PermissionError) as e:
        log.error(f"File access error when saving to {filepath}: {e}")
        raise IOError(f"Could not access file '{filepath}': {e}")
    except UnicodeEncodeError as e:
        log.error(f"Encoding error when saving to {filepath}: {e}")
        raise IOError(f"Encoding error when saving to '{filepath}': {e}")
    except ValueError as e:
        log.error(f"Data structure error when saving to {filepath}: {e}")
        raise ValueError(f"Invalid results structure: {e}")
    except Exception as e:
        log.error(f"Unexpected error when saving to {filepath}: {e}")
        raise IOError(f"Unexpected error when saving to '{filepath}': {e}")

def format_coordinates_astropy(ra_hours: float, dec_degrees: float, precision: Optional[int] = None) -> str:
    """
    Formats celestial coordinates for display using the astropy library.
    
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
    approximate J2000 coordinates with validation.

    Args:
        wds_id: The WDS identifier string.

    Returns:
        A dictionary with 'ra_deg' and 'dec_deg', or None if parsing fails
        or coordinates are outside valid astronomical ranges.
    """
    if not isinstance(wds_id, str) or len(wds_id) < MIN_WDS_ID_LENGTH:
        return None
        
    # WDS designation format validation: HHMMM[+-]DDMM[AB-like components]
    # Components after coordinates are optional (e.g., AB, AC, ABC)
    if not re.match(WDS_COORDINATE_PATTERN, wds_id[:12]):
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
        
        # Validate astronomical coordinate ranges immediately
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

# Coordinate formatting functions
def format_coordinates(ra_hours: float, dec_degrees: float, precision: Optional[int] = None) -> str:
    """
    Universal coordinate formatting function.
    
    Args:
        ra_hours: Right ascension in hours
        dec_degrees: Declination in degrees  
        precision: Number of decimal places for seconds (uses DEFAULT_COORDINATE_PRECISION if None)
        
    Returns:
        Formatted coordinate string in HMS/DMS format
    """
    return format_coordinates_astropy(ra_hours, dec_degrees, precision)