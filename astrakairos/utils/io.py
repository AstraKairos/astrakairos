import pandas as pd
import re
import csv
import logging
from typing import List, Dict, Any, Optional
from astropy.coordinates import SkyCoord
import astropy.units as u

log = logging.getLogger(__name__)

def load_csv_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Loads star data from a CSV file, robustly detecting the delimiter.

    This function attempts to use csv.Sniffer to auto-detect the delimiter,
    then falls back to a list of common delimiters (comma, semicolon).

    Args:
        filepath: Path to the CSV file.

    Returns:
        A pandas DataFrame with the loaded data, or None if loading fails.
    """
    delimiters_to_try = [None, ',', ';'] # None will trigger csv.Sniffer

    for i, delimiter in enumerate(delimiters_to_try):
        try:
            current_delimiter = delimiter
            if i == 0: # First attempt with Sniffer
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        sample = f.read(2048) # Read a sample
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
    Saves analysis results to a CSV file.

    Args:
        results: A list of dictionaries, where each dictionary is a row.
        filepath: The path for the output CSV file.
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

def format_coordinates_astropy(ra_hours: float, dec_degrees: float, precision: int = 1) -> str:
    """
    Formats celestial coordinates for display using the astropy library.
    """
    if ra_hours is None or dec_degrees is None:
        return "N/A"
    try:
        coords = SkyCoord(ra=ra_hours * u.hourangle, dec=dec_degrees * u.deg, frame='icrs')
        return coords.to_string('hmsdms', sep=' ', precision=precision, pad=True)
        
    except (ValueError, TypeError) as e:
        log.error(f"Astropy coordinate formatting failed: {e}")
        return "Invalid Coords"


def parse_wds_designation(wds_id: str) -> Optional[Dict[str, float]]:
    """
    Parses a WDS designation string (e.g., "00013+1234") to extract
    approximate J2000 coordinates.

    Args:
        wds_id: The WDS identifier string.

    Returns:
        A dictionary with 'ra_deg' and 'dec_deg', or None if parsing fails.
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
        ra_deg = (ra_h + ra_m / 60.0) * 15.0

        dec_sign = 1 if wds_id[5] == '+' else -1
        dec_d = int(wds_id[6:8])
        dec_m = int(wds_id[8:10])
        dec_deg = dec_sign * (dec_d + dec_m / 60.0)
        
        return {'ra_deg': ra_deg, 'dec_deg': dec_deg}

    except (ValueError, IndexError) as e:
        log.warning(f"Failed to parse WDS designation '{wds_id}': {e}")
        return None