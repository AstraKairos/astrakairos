import pandas as pd
import re
from typing import List, Dict, Any, Optional
import csv

def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV file with star data, attempting to auto-detect the delimiter.
    If auto-detection fails, it falls back to common delimiters (semicolon, comma).
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        DataFrame with the loaded data.
    
    Raises:
        Exception: If the CSV file cannot be loaded with any tried delimiter.
    """
    try:
        # Try to auto-detect the delimiter using csv.Sniffer
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read a small sample to sniff the dialect
            sample = f.read(2048) # Read the first 2KB
            try:
                dialect = csv.Sniffer().sniff(sample)
                detected_delimiter = dialect.delimiter
            except csv.Error:
                # If sniffing fails (e.g., file too small, no common delimiters found),
                # fall back to default behavior
                detected_delimiter = None # Will trigger the fallback
            
            f.seek(0) # Rewind the file pointer to the beginning
        
        if detected_delimiter:
            df = pd.read_csv(filepath, delimiter=detected_delimiter, encoding='utf-8')
            print(f"CSV loaded successfully with detected delimiter '{detected_delimiter}'. Rows: {len(df)}")
            return df
        
        # Fallback if auto-detection didn't work or detected_delimiter is None
        # Try semicolon first (common in some regions)
        print("Auto-detection failed or no delimiter detected. Trying fallback delimiters...")
        df = pd.read_csv(filepath, delimiter=';', encoding='utf-8')
        print(f"CSV loaded successfully with semicolon delimiter. Rows: {len(df)}")
        return df

    except Exception as e_semicolon:
        print(f"Error loading CSV with semicolon delimiter: {e_semicolon}")
        try:
            # Then try comma (most common globally)
            df = pd.read_csv(filepath, delimiter=',', encoding='utf-8')
            print(f"CSV loaded successfully with comma delimiter. Rows: {len(df)}")
            return df
        except Exception as e_comma:
            print(f"Error loading CSV with comma delimiter: {e_comma}")
            raise Exception(f"Could not load CSV file: {filepath}. Tried auto-detection, semicolon, and comma delimiters.")

def save_results_to_csv(results: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save analysis results to a CSV file.
    
    Args:
        results: List of dictionaries containing analysis results
        filepath: Output file path
    """
    if not results:
        print("No results to save")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"Results saved to {filepath} ({len(df)} rows)")

def format_coordinates(ra_hours: float, dec_degrees: float) -> str:
    """
    Format celestial coordinates for display.
    
    Args:
        ra_hours: Right ascension in hours
        dec_degrees: Declination in degrees
        
    Returns:
        Formatted string like "12h34m56s +12°34'56"
    """
    # Format RA
    ra_h = int(ra_hours)
    ra_m = int((ra_hours - ra_h) * 60)
    ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60
    
    # Format Dec
    dec_sign = "+" if dec_degrees >= 0 else "-"
    dec_abs = abs(dec_degrees)
    dec_d = int(dec_abs)
    dec_m = int((dec_abs - dec_d) * 60)
    dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60
    
    return f"{ra_h:02d}h{ra_m:02d}m{ra_s:05.2f}s {dec_sign}{dec_d:02d}°{dec_m:02d}'{dec_s:05.2f}\""

def parse_wds_designation(wds_id: str) -> Optional[Dict[str, float]]:
    """
    Parse WDS designation to extract approximate coordinates.
    
    Args:
        wds_id: WDS identifier like "00013+1234"
        
    Returns:
        Dictionary with 'ra_hours' and 'dec_degrees', or None if parsing fails
    """
    try:
        # WDS format: HHMMM±DDMM
        # First 5 chars: RA (HHMMm where m is tenths of minutes)
        # Next 5 chars: Dec (±DDmm where mm is minutes)
        
        if not re.match(r'^\d{5}[+-]\d{4}', wds_id[:10]):
            return None # J2000 WDS format

        if len(wds_id) < 10:
            return None
        
        # Extract RA
        ra_h = int(wds_id[0:2])
        ra_m = int(wds_id[2:4])
        ra_m_decimal = int(wds_id[4]) / 10.0
        ra_hours = ra_h + (ra_m + ra_m_decimal) / 60.0
        
        # Extract Dec
        dec_sign = 1 if wds_id[5] == '+' else -1
        dec_d = int(wds_id[6:8])
        dec_m = int(wds_id[8:10])
        dec_degrees = dec_sign * (dec_d + dec_m / 60.0)
        
        return {
            'ra_hours': ra_hours,
            'dec_degrees': dec_degrees
        }
        
    except (ValueError, IndexError):
        return None

def filter_valid_observations(df: pd.DataFrame, 
                            min_obs: int = 2,
                            min_timespan: float = 1.0) -> pd.DataFrame:
    """
    Filter DataFrame to include only valid observations.
    
    Args:
        df: Input DataFrame
        min_obs: Minimum number of observations
        min_timespan: Minimum timespan in years
        
    Returns:
        Filtered DataFrame
    """
    # Filter by number of observations
    if 'obs' in df.columns:
        df = df[df['obs'] >= min_obs]
    
    # Filter by timespan if date columns exist
    if 'date_first' in df.columns and 'date_last' in df.columns:
        df = df[(df['date_last'] - df['date_first']) >= min_timespan]
    
    # Remove rows with missing critical data
    critical_columns = ['wds_name', 'pa_first', 'sep_first', 'pa_last', 'sep_last']
    for col in critical_columns:
        if col in df.columns:
            df = df[df[col].notna()]
    
    return df.reset_index(drop=True)