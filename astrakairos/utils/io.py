import pandas as pd
from typing import List, Dict, Any, Optional

def load_csv_data(filepath: str, delimiter: str = ';') -> pd.DataFrame:
    """
    Load CSV file with star data.
    
    Args:
        filepath: Path to the CSV file
        delimiter: CSV delimiter (default: ';')
        
    Returns:
        DataFrame with the loaded data
    """
    try:
        # Try with specified delimiter first
        df = pd.read_csv(filepath, delimiter=delimiter, encoding='utf-8')
        print(f"CSV loaded successfully. Rows: {len(df)}")
        return df
    except Exception as e:
        print(f"Error loading CSV with delimiter '{delimiter}': {e}")
        
        # Try with comma delimiter
        if delimiter != ',':
            try:
                df = pd.read_csv(filepath, delimiter=',', encoding='utf-8')
                print(f"CSV loaded with comma delimiter. Rows: {len(df)}")
                return df
            except Exception as e2:
                print(f"Error with comma delimiter: {e2}")
        
        raise Exception(f"Could not load CSV file: {filepath}")

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