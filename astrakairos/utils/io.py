import pandas as pd
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from astropy.coordinates import SkyCoord
import astropy.units as u

# Import configuration constants
from ..config import (
    MIN_RA_DEG, MAX_RA_DEG, MIN_DEC_DEG, MAX_DEC_DEG,
    CSV_SNIFFER_SAMPLE_SIZE, DEFAULT_COORDINATE_PRECISION,
    COORDINATE_ERROR_BEHAVIOR, MIN_WDS_ID_LENGTH, WDS_COORDINATE_PATTERN,
    DEGREES_PER_HOUR, MINUTES_PER_HOUR, ASTROPY_FRAME, ASTROPY_FORMAT,
    ENCODING_FALLBACK_ORDER
)

log = logging.getLogger(__name__)

class DataLoadError(Exception):
    """Exception raised when data cannot be loaded from a file."""
    pass


class DataSaveError(Exception):
    """Exception raised when data cannot be saved to a file."""
    pass


class InvalidWdsFormatError(Exception):
    """Exception raised when WDS designation format is invalid."""
    pass


class CoordinateOutOfRangeError(Exception):
    """Exception raised when coordinates are outside valid astronomical ranges."""
    pass


class CoordinateOutOfRangeError(Exception):
    """Exception raised when coordinates are outside valid astronomical ranges."""
    pass

def load_csv_data(filepath: str) -> pd.DataFrame:
    """Loads astronomical star data from a CSV file with robust delimiter detection.

    This function uses pandas' built-in delimiter detection capabilities
    designed for astronomical catalogs which may use different delimiter conventions.
    
    Args:
        filepath: Absolute path to the CSV file containing stellar data.
                 Expected to have at minimum a 'wds_id' column for star identification.

    Returns:
        Loaded astronomical data with proper column types.
        
    Raises:
        DataLoadError: If the file cannot be loaded, parsed, or lacks required columns.
    """
    for encoding in ENCODING_FALLBACK_ORDER:
        try:
            log.info(f"Attempting to read CSV with encoding: {encoding}")
            
            # Try default CSV reading first (comma-separated)
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                
                # Validate that required 'wds_id' column exists
                if 'wds_id' not in df.columns:
                    # Try semicolon delimiter fallback
                    log.debug("wds_id column not found with comma delimiter, trying semicolon...")
                    df = pd.read_csv(filepath, encoding=encoding, sep=';')
                    
                    if 'wds_id' not in df.columns:
                        raise DataLoadError(f"Required 'wds_id' column not found in {filepath}")
                
                log.info(f"CSV loaded successfully. Rows: {len(df)}, Encoding: {encoding}")
                return df
                
            except pd.errors.ParserError:
                # If comma parsing fails, try semicolon
                log.debug("CSV parsing failed with comma delimiter, trying semicolon...")
                try:
                    df = pd.read_csv(filepath, encoding=encoding, sep=';')
                    
                    # Validate that required 'wds_id' column exists
                    if 'wds_id' not in df.columns:
                        raise DataLoadError(f"Required 'wds_id' column not found in {filepath}")
                    
                    log.info(f"CSV loaded successfully with semicolon delimiter. Rows: {len(df)}, Encoding: {encoding}")
                    return df
                except pd.errors.ParserError:
                    # Both comma and semicolon parsing failed - this is a real parser error
                    log.error(f"Data parsing error: failed to parse CSV format with both delimiters")
                    raise DataLoadError(f"Could not parse CSV format in file: {filepath}")
            
        except UnicodeDecodeError:
            log.debug(f"Encoding {encoding} failed, trying next...")
            continue
        except FileNotFoundError as e:
            log.error(f"File not found: {e}")
            raise DataLoadError(f"File not found: {filepath}")
        except PermissionError as e:
            log.error(f"Permission denied: {e}")
            raise DataLoadError(f"Permission denied accessing file: {filepath}")
        except pd.errors.EmptyDataError as e:
            log.error(f"Empty data file: {e}")
            raise DataLoadError(f"File contains no data: {filepath}")
        except DataLoadError:
            # Re-raise our own DataLoadError (like missing wds_id column)
            raise
        except Exception as e:
            # Handle CSV Sniffer errors for empty files or other CSV parsing issues
            if "Could not determine delimiter" in str(e):
                log.error(f"Empty file or invalid CSV format: {e}")
                raise DataLoadError(f"File contains no data or invalid CSV format: {filepath}")
            continue
            
    log.error(f"Could not decode file with any supported encoding: {filepath}")
    raise DataLoadError(f"Could not decode file '{filepath}' with any supported encoding")


def save_results_to_csv(results: List[Dict[str, Any]], filepath: str) -> None:
    """Saves astronomical analysis results to a publication-ready CSV file.

    This function creates standardized output files suitable for publication
    supplementary data, further analysis in astronomical software, and 
    sharing with observational collaborators.
    
    Args:
        results: List of analysis dictionaries, where each dictionary represents
                one stellar system with standardized keys:
                - 'wds_id': WDS identifier (required)
                - 'opi': Observation Priority Index (for orbital mode)
                - 'rmse': Motion fit quality (for characterize mode)  
                - 'physicality_p_value': Gaia validation (for discovery mode)
        filepath: Output path for the CSV file (will be created/overwritten).
        
    Raises:
        DataSaveError: If file writing fails due to permissions, disk space, 
                      encoding issues, or invalid data structure.
    """
    if not results:
        log.warning("No results to save.")
        return
    
    try:
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False, encoding='utf-8')
        log.info(f"Results successfully saved to {filepath} ({len(df)} rows)")
    except FileNotFoundError as e:
        log.error(f"Directory not found when saving to {filepath}: {e}")
        raise DataSaveError(f"Directory not found: {filepath}")
    except PermissionError as e:
        log.error(f"Permission denied when saving to {filepath}: {e}")
        raise DataSaveError(f"Permission denied: {filepath}")
    except UnicodeEncodeError as e:
        log.error(f"Unicode encoding error when saving to {filepath}: {e}")
        raise DataSaveError(f"Unicode encoding error in data: {e}")
    except ValueError as e:
        log.error(f"Invalid data structure when saving to {filepath}: {e}")
        raise DataSaveError(f"Invalid results structure: {e}")
    except OSError as e:
        log.error(f"OS error when saving to {filepath}: {e}")
        raise DataSaveError(f"OS error (disk space, path length, etc.): {e}")

def format_coordinates_astropy(ra_hours: float, dec_degrees: float, precision: Optional[int] = None) -> str:
    """Formats celestial coordinates for display using the astropy library.
    
    Args:
        ra_hours: Right ascension in hours (0-24).
        dec_degrees: Declination in degrees (-90 to +90).
        precision: Number of decimal places for seconds (uses DEFAULT_COORDINATE_PRECISION if None).
        
    Returns:
        Formatted coordinate string in HMS/DMS format, "N/A" for None inputs,
        or "Invalid Coords" for formatting errors (based on COORDINATE_ERROR_BEHAVIOR).
        
    Raises:
        ValueError: If COORDINATE_ERROR_BEHAVIOR is "raise" and coordinates are invalid.
        CoordinateOutOfRangeError: If coordinates are outside valid astronomical ranges.
    """
    if ra_hours is None or dec_degrees is None:
        return "N/A"
        
    if precision is None:
        precision = DEFAULT_COORDINATE_PRECISION
    
    # Type validation before range checking
    try:
        ra_hours = float(ra_hours)
        dec_degrees = float(dec_degrees)
    except (TypeError, ValueError):
        error_msg = f"Coordinates must be numeric (RA: {ra_hours}, Dec: {dec_degrees})"
        log.error(error_msg)
        if COORDINATE_ERROR_BEHAVIOR == "raise":
            raise ValueError(error_msg)
        elif COORDINATE_ERROR_BEHAVIOR == "return_none":
            return None
        else:  # "return_invalid" (default)
            return "Invalid Coords"
    
    # Explicit coordinate range validation
    if not (0.0 <= ra_hours <= 24.0):
        error_msg = f"RA hours {ra_hours} outside valid range [0, 24]"
        log.error(error_msg)
        if COORDINATE_ERROR_BEHAVIOR == "raise":
            raise CoordinateOutOfRangeError(error_msg)
        elif COORDINATE_ERROR_BEHAVIOR == "return_none":
            return None
        else:  # "return_invalid" (default)
            return "Invalid Coords"
    
    if not (-90.0 <= dec_degrees <= 90.0):
        error_msg = f"Dec degrees {dec_degrees} outside valid range [-90, 90]"
        log.error(error_msg)
        if COORDINATE_ERROR_BEHAVIOR == "raise":
            raise CoordinateOutOfRangeError(error_msg)
        elif COORDINATE_ERROR_BEHAVIOR == "return_none":
            return None
        else:  # "return_invalid" (default)
            return "Invalid Coords"
        
    try:
        coords = SkyCoord(ra=ra_hours * u.hourangle, dec=dec_degrees * u.deg, frame=ASTROPY_FRAME)
        return coords.to_string(ASTROPY_FORMAT, sep=' ', precision=precision, pad=True)
        
    except (ValueError, TypeError) as e:
        error_msg = f"Astropy coordinate formatting failed: {e}"
        log.error(error_msg)
        
        if COORDINATE_ERROR_BEHAVIOR == "raise":
            raise ValueError(error_msg)
        elif COORDINATE_ERROR_BEHAVIOR == "return_none":
            return None
        else:  # "return_invalid" (default)
            return "Invalid Coords"


def parse_wds_designation(wds_id: str) -> Dict[str, float]:
    """Parses a WDS designation string to extract approximate J2000 coordinates with validation.

    Args:
        wds_id: The WDS identifier string (e.g., "00013+1234").

    Returns:
        A dictionary with 'ra_deg' and 'dec_deg' keys containing coordinate values.
        
    Raises:
        InvalidWdsFormatError: If WDS designation format is invalid.
        CoordinateOutOfRangeError: If parsed coordinates are outside valid ranges.
    """
    if not isinstance(wds_id, str) or len(wds_id) < MIN_WDS_ID_LENGTH:
        raise InvalidWdsFormatError(f"WDS ID '{wds_id}' is too short (minimum {MIN_WDS_ID_LENGTH} characters)")
        
    # WDS designation format validation: HHMMM[+-]DDMM[components]
    if not re.match(WDS_COORDINATE_PATTERN, wds_id[:12]):
        raise InvalidWdsFormatError(f"WDS ID '{wds_id}' does not match coordinate format HHMMM±DDMM")
        
    try:
        ra_h = int(wds_id[0:2])
        ra_m = int(wds_id[2:5]) / 10.0  # HHMM.m
        ra_hours = ra_h + ra_m / MINUTES_PER_HOUR
        ra_deg = ra_hours * DEGREES_PER_HOUR

        dec_sign = 1 if wds_id[5] == '+' else -1
        dec_d = int(wds_id[6:8])
        dec_m = int(wds_id[8:10])
        dec_deg = dec_sign * (dec_d + dec_m / MINUTES_PER_HOUR)
        
        # Validate astronomical coordinate ranges
        if not (MIN_RA_DEG <= ra_deg <= MAX_RA_DEG):
            raise CoordinateOutOfRangeError(
                f"WDS ID '{wds_id}' has RA {ra_deg:.3f}° outside valid range [{MIN_RA_DEG}, {MAX_RA_DEG}]"
            )
            
        if not (MIN_DEC_DEG <= dec_deg <= MAX_DEC_DEG):
            raise CoordinateOutOfRangeError(
                f"WDS ID '{wds_id}' has Dec {dec_deg:.3f}° outside valid range [{MIN_DEC_DEG}, {MAX_DEC_DEG}]"
            )
        
        return {'ra_deg': ra_deg, 'dec_deg': dec_deg}

    except (ValueError, IndexError) as e:
        raise InvalidWdsFormatError(f"Failed to parse WDS designation '{wds_id}': {e}")


# === Safe Parsing Utilities ===
# These functions provide robust parsing for astronomical catalog data

def safe_int(s: str) -> Optional[int]:
    """
    Safely convert string to int, returning None on error.
    
    Args:
        s: String to convert
        
    Returns:
        Integer value or None if conversion fails
    """
    try:
        return int(s.strip()) if s.strip() else None
    except (ValueError, AttributeError):
        return None


def safe_float(s: str) -> Optional[float]:
    """
    Safely convert string to float, returning None on error.
    
    Args:
        s: String to convert
        
    Returns:
        Float value or None if conversion fails
    """
    try:
        return float(s.strip()) if s.strip() else None
    except (ValueError, AttributeError):
        return None


def parse_wdss_coordinates(wdss_id: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse coordinates from WDSS identifier (first 14 chars).
    
    Args:
        wdss_id: WDSS identifier string
        
    Returns:
        Tuple of (ra_deg, dec_deg) or (None, None) if parsing fails
    """
    if not wdss_id or len(wdss_id) < 14:
        return None, None
    
    try:
        # Format: HHMMSss+DDMMss or HHMMSss-DDMMss
        coord_part = wdss_id[:14]
        
        # Find the sign position
        sign_pos = -1
        for i, char in enumerate(coord_part):
            if char in ['+', '-']:
                sign_pos = i
                break
        
        if sign_pos == -1:
            return None, None
        
        # Parse RA part (before sign)
        ra_str = coord_part[:sign_pos]
        if len(ra_str) >= 6:
            hours = int(ra_str[0:2])
            minutes = int(ra_str[2:4])
            seconds = int(ra_str[4:6]) if len(ra_str) >= 6 else 0
            ra_deg = (hours + minutes/60.0 + seconds/3600.0) * 15.0
        else:
            ra_deg = None
        
        # Parse Dec part (after sign)
        dec_str = coord_part[sign_pos:]
        if len(dec_str) >= 6:
            sign = -1 if dec_str[0] == '-' else 1
            degrees = int(dec_str[1:3])
            minutes = int(dec_str[3:5])
            seconds = int(dec_str[5:7]) if len(dec_str) >= 7 else 0
            dec_deg = sign * (degrees + minutes/60.0 + seconds/3600.0)
        else:
            dec_deg = None
        
        return ra_deg, dec_deg
        
    except Exception:
        return None, None


def parse_wdss_coordinate_string(coord_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse WDSS coordinate string to decimal degrees.
    
    Args:
        coord_str: Coordinate string in WDSS format
        
    Returns:
        Tuple of (ra_deg, dec_deg) or (None, None) if parsing fails
    """
    if not coord_str or len(coord_str) < 17:
        return None, None
    
    try:
        # RA: hhmmss.ss (positions 0-8)
        ra_str = coord_str[0:9].strip()
        if len(ra_str) >= 6:
            hours = int(ra_str[0:2])
            minutes = int(ra_str[2:4])
            seconds = float(ra_str[4:]) if len(ra_str) > 4 else 0.0
            ra_deg = (hours + minutes/60.0 + seconds/3600.0) * 15.0
        else:
            ra_deg = None
        
        # Dec: +ddmmss.s (positions 9-17)
        dec_str = coord_str[9:18].strip()
        if len(dec_str) >= 6:
            sign = -1 if dec_str[0] == '-' else 1
            degrees = int(dec_str[1:3])
            minutes = int(dec_str[3:5])
            seconds = float(dec_str[5:]) if len(dec_str) > 5 else 0.0
            dec_deg = sign * (degrees + minutes/60.0 + seconds/3600.0)
        else:
            dec_deg = None
        
        return ra_deg, dec_deg
        
    except Exception:
        return None, None

