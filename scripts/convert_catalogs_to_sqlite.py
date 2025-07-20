#!/usr/bin/env python3
"""
Convert WDS text catalogs to SQLite for efficient querying.

This script converts the WDS summary, ORB6, and measurements catalogs
from fixed-width text format to SQLite database with proper indexing.
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from astrakairos.config import (
    DAYS_PER_JULIAN_YEAR, CENTURIES_PER_YEAR, MILLIARCSEC_PER_ARCSEC,
    MIN_PERIOD_YEARS, MAX_PERIOD_YEARS,
    MIN_SEMIMAJOR_AXIS_ARCSEC, MAX_SEMIMAJOR_AXIS_ARCSEC,
    MIN_ECCENTRICITY, MAX_ECCENTRICITY,
    MIN_INCLINATION_DEG, MAX_INCLINATION_DEG
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def detect_wds_format(filepath: str) -> dict:
    """Auto-detect WDS file format by analyzing first few lines."""
    log.info(f"Auto-detecting format for: {filepath}")
    
    with open(filepath, 'r') as f:
        sample_lines = [f.readline().rstrip() for _ in range(10)]
    
    # Remove empty lines
    sample_lines = [line for line in sample_lines if line.strip()]
    
    if not sample_lines:
        raise ValueError("File appears to be empty")
    
    # Analyze line length and content patterns
    line_lengths = [len(line) for line in sample_lines]
    avg_length = sum(line_lengths) / len(line_lengths)
    
    log.info(f"Detected average line length: {avg_length:.1f}")
    log.info(f"Sample line: {repr(sample_lines[0][:50])}")
    
    # Basic format detection based on typical WDS catalog characteristics
    format_info = {
        'avg_line_length': avg_length,
        'sample_line': sample_lines[0],
        'format_detected': 'unknown'
    }
    
    if 130 <= avg_length <= 150:
        format_info['format_detected'] = 'wds_summary'
    elif 80 <= avg_length <= 120:
        format_info['format_detected'] = 'measurements'  
    elif avg_length > 200:
        format_info['format_detected'] = 'orb6'
    
    log.info(f"Detected format: {format_info['format_detected']}")
    return format_info


def parse_wdss_master_catalog(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse WDSS master catalog into three clean DataFrames using optimized approach.
    
    This function reads the master WDSS file once and extracts all data types efficiently.
    
    Args:
        filepath: Path to the master WDSS catalog file
        
    Returns:
        Tuple of (df_components, df_measurements, df_correspondence)
    """
    log.info(f"Parsing WDSS master catalog: {filepath}")
    
    components_data = []
    measurements_data = []
    correspondence_data = []
    
    # Use a set for fast correspondence lookup instead of list search
    correspondence_seen = set()
    
    total_lines = 0
    systems_processed = set()
    
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                if total_lines % 100000 == 0:
                    log.info(f"Processed {total_lines} lines, {len(systems_processed)} systems")
                
                # Skip empty lines and headers
                if not line.strip() or line.startswith('=') or len(line) < 160:
                    continue
                
                # Extract basic identifiers
                wdss_id = line[0:14].strip()
                component = line[15:18].strip()
                
                if not wdss_id:
                    continue
                
                systems_processed.add(wdss_id)
                
                # Parse WDS correspondence efficiently (once per system)
                wds_correspondence = None
                discoverer_designation = None
                if len(line) >= 147:
                    wds_corr = line[137:147].strip()
                    if wds_corr and wds_corr != '' and not wds_corr.isspace():
                        wds_correspondence = wds_corr
                
                if len(line) >= 155:
                    disc_des = line[148:155].strip()
                    if disc_des and disc_des != '' and not disc_des.isspace():
                        discoverer_designation = disc_des
                
                # Add correspondence mapping using set for O(1) lookup instead of O(N) list search
                if wds_correspondence and wdss_id not in correspondence_seen:
                    correspondence_seen.add(wdss_id)
                    correspondence_data.append({
                        'wdss_id': wdss_id,
                        'wds_id': wds_correspondence,
                        'discoverer_designation': discoverer_designation
                    })
                
                # Determine line type and parse accordingly
                if component and len(component) == 1 and component.isalpha():
                    # Component line (A, B, C, etc.)
                    comp_data = _parse_component_line_data(line, wdss_id, component)
                    if comp_data:
                        components_data.append(comp_data)
                        
                elif component and len(component) >= 2:
                    # Measurement line (AB, AC, BC, etc.)
                    meas_data = _parse_measurement_line_data(line, wdss_id, component)
                    if meas_data:
                        measurements_data.append(meas_data)
        
        log.info(f"Extracted {len(components_data)} components, {len(measurements_data)} measurements, {len(correspondence_data)} correspondences")
        
        # Convert to DataFrames efficiently
        df_components = pd.DataFrame(components_data)
        df_measurements = pd.DataFrame(measurements_data)
        df_correspondence = pd.DataFrame(correspondence_data)
        
        log.info(f"Created DataFrames from {total_lines} input lines")
        return df_components, df_measurements, df_correspondence
        
    except Exception as e:
        log.error(f"Failed to parse WDSS master catalog: {e}")
        raise


def _parse_component_line_data(line: str, wdss_id: str, component: str) -> dict:
    """Extract component data from a WDSS component line."""
    try:
        # Parse coordinates from line (cols 119-136)
        coordinates = line[118:136].strip() if len(line) > 136 else ''
        ra_deg, dec_deg = parse_coordinates(coordinates)
        
        # Fallback to WDSS ID coordinates if needed
        if ra_deg is None or dec_deg is None:
            ra_deg, dec_deg = parse_wdss_coordinates(wdss_id)
        
        return {
            'wdss_id': wdss_id,
            'component': component,
            'date_first': safe_int(line[24:28]),
            'n_obs': safe_int(line[29:32]),
            'pa_first': safe_float(line[33:36]),
            'sep_first': safe_float(line[37:43]),
            'vmag': safe_float(line[45:50]),
            'kmag': safe_float(line[52:57]),
            'spectral_type': line[59:64].strip() if len(line) > 64 else '',
            'pm_ra': safe_float(line[65:73]) if len(line) > 73 else None,
            'pm_dec': safe_float(line[73:81]) if len(line) > 81 else None,
            'parallax': safe_float(line[82:89]) if len(line) > 89 else None,
            'name': line[90:114].strip() if len(line) > 114 else '',
            'ra_deg': ra_deg,
            'dec_deg': dec_deg
        }
    except Exception as e:
        log.debug(f"Error parsing component line: {e}")
        return None


def _parse_measurement_line_data(line: str, wdss_id: str, pair: str) -> dict:
    """Extract measurement data from a WDSS measurement line."""
    try:
        return {
            'wdss_id': wdss_id,
            'pair': pair,
            'epoch': safe_float(line[24:34]),
            'theta': safe_float(line[36:43]),
            'rho': safe_float(line[52:61]),
            'mag1': safe_float(line[72:78]),
            'mag2': safe_float(line[86:92]),
            'reference': line[119:127].strip() if len(line) > 127 else '',
            'technique': line[128:130].strip() if len(line) > 130 else ''
        }
    except Exception as e:
        log.debug(f"Error parsing measurement line: {e}")
        return None





def generate_summary_table(df_components: pd.DataFrame, df_measurements: pd.DataFrame, df_correspondence: pd.DataFrame) -> pd.DataFrame:
    log.info("Generating summary table using vectorized operations")

    # 1. Aggregate measurements using groupby
    if not df_measurements.empty:
        agg_measurements = (
            df_measurements.sort_values(['wdss_id', 'epoch'])
            .groupby('wdss_id')
            .agg(
                date_first=('epoch', 'first'),
                date_last=('epoch', 'last'),
                n_obs=('epoch', 'count'),
                pa_first=('theta', 'first'),
                pa_last=('theta', 'last'),
                sep_first=('rho', 'first'),
                sep_last=('rho', 'last')
            )
        )
    else:
        agg_measurements = pd.DataFrame(columns=['wdss_id']).set_index('wdss_id')

    # 2. Pivot component data from long to wide format
    if not df_components.empty:
        # Select only primary (A) and secondary (B) for simplicity
        df_ab = df_components[df_components['component'].isin(['A', 'B'])]
        
        # Remove duplicates - keep first occurrence of each component per system
        df_ab = df_ab.drop_duplicates(subset=['wdss_id', 'component'], keep='first')
        
        df_wide_components = df_ab.pivot(
            index='wdss_id', 
            columns='component', 
            values=['vmag', 'kmag', 'spectral_type', 'ra_deg', 'dec_deg', 'pm_ra', 'pm_dec', 'parallax', 'name']
        )
        # Flatten the multi-level column index
        df_wide_components.columns = [f'{val}_{comp}' for val, comp in df_wide_components.columns]
    else:
        df_wide_components = pd.DataFrame(columns=['wdss_id']).set_index('wdss_id')

    # 3. Merge all data sources together
    # Start with the aggregated measurements, then join everything else
    df_summary = agg_measurements
    df_summary = df_summary.join(df_wide_components, how='outer')
    df_summary = df_summary.join(df_correspondence.set_index('wdss_id'), how='left')

    df_summary.reset_index(inplace=True)

    # 4. Finalize columns and rename for SQLite schema
    df_summary.rename(columns={
        'wds_id': 'wds_correspondence', # from correspondence merge
        'vmag_A': 'vmag',              # LocalDataSource expects 'vmag as mag_pri'
        'vmag_B': 'kmag',              # LocalDataSource expects 'kmag as mag_sec'
        'ra_deg_A': 'ra_deg',
        'dec_deg_A': 'dec_deg',
        'spectral_type_A': 'spectral_type',
        'parallax_A': 'parallax',
        'pm_ra_A': 'pm_ra',
        'pm_dec_A': 'pm_dec',
        'name_A': 'name'
    }, inplace=True)

    # If wds_correspondence is null, use wdss_id as the primary wds_id
    df_summary['wds_id'] = df_summary['wds_correspondence'].fillna(df_summary['wdss_id'])
    
    # Select final columns to ensure a clean schema
    final_cols = [
        'wds_id', 'wdss_id', 'discoverer_designation', 'date_first', 'date_last', 'n_obs',
        'pa_first', 'pa_last', 'sep_first', 'sep_last', 'vmag', 'kmag',
        'ra_deg', 'dec_deg', 'spectral_type', 'parallax', 'pm_ra', 'pm_dec', 'name'
    ]
    
    # Add missing columns with NaN if they don't exist
    for col in final_cols:
        if col not in df_summary.columns:
            df_summary[col] = np.nan
    
    log.info(f"Generated summary table with {len(df_summary)} systems")
    return df_summary[final_cols]

def parse_coordinates(coord_str: str) -> tuple:
    """Parse WDSS coordinate string to decimal degrees."""
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


def parse_wdss_coordinates(wdss_id: str) -> tuple:
    """Parse coordinates from WDSS identifier (first 14 chars)."""
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


def safe_int(s: str) -> Optional[int]:
    """Safely convert string to int, returning None on error."""
    try:
        return int(s.strip()) if s.strip() else None
    except (ValueError, AttributeError):
        return None


def safe_float(s: str) -> Optional[float]:
    """Safely convert string to float, returning None on error."""
    try:
        return float(s.strip()) if s.strip() else None
    except (ValueError, AttributeError):
        return None


def parse_orb6_catalog(filepath: str) -> pd.DataFrame:
    """Parse ORB6 catalog with robust error handling."""
    log.info(f"Parsing ORB6 catalog: {filepath}")
    
    # Configurable column specifications  
    colspecs = [
        (19, 29),   # wds_id
        (81, 93),   # P_str (Period with unit flag)
        (105, 116), # a_str (Semi-major axis with unit flag)
        (125, 134), # i_str (Inclination)
        (143, 154), # Omega_str (Node)
        (162, 177), # T_str (Time of periastron with unit flag)
        (187, 196), # e_str (Eccentricity)
        (205, 215)  # omega_str (Longitude of periastron)
    ]
    
    names = ['wds_id', 'P_str', 'a_str', 'i_str', 'Omega_str', 
             'T_str', 'e_str', 'omega_str']
    
    try:
        df = pd.read_fwf(filepath, colspecs=colspecs, names=names, comment='R', dtype=str)
        log.info(f"Read {len(df)} raw entries from ORB6")
        
        df.dropna(subset=['wds_id'], inplace=True)
        df['wds_id'] = df['wds_id'].str.strip()
        
        # Parse values and unit flags with validation
        df['P'] = pd.to_numeric(df['P_str'].str[:-1], errors='coerce')
        df.loc[df['P_str'].str.endswith('d', na=False), 'P'] /= DAYS_PER_JULIAN_YEAR
        df.loc[df['P_str'].str.endswith('c', na=False), 'P'] *= CENTURIES_PER_YEAR
        
        df['a'] = pd.to_numeric(df['a_str'].str[:-1], errors='coerce')
        df.loc[df['a_str'].str.endswith('m', na=False), 'a'] /= MILLIARCSEC_PER_ARCSEC
        
        df['i'] = pd.to_numeric(df['i_str'], errors='coerce')
        df['Omega'] = pd.to_numeric(df['Omega_str'].str.rstrip('*q'), errors='coerce')
        df['T'] = pd.to_numeric(df['T_str'].str[:-1], errors='coerce')
        df['e'] = pd.to_numeric(df['e_str'], errors='coerce')
        df['omega'] = pd.to_numeric(df['omega_str'].str.rstrip('q'), errors='coerce')
        
        # Apply physical validation
        apply_physical_validation(df)
        
        # Remove duplicates (keep first)
        df.drop_duplicates(subset=['wds_id'], keep='first', inplace=True)
        
        log.info(f"Processed {len(df)} valid ORB6 entries")
        
        # Debug: Print all columns to see if there are duplicates
        log.info(f"ORB6 columns before return: {df.columns.tolist()}")
        log.info(f"ORB6 column types: {df.dtypes.to_dict()}")
        
        return df[['wds_id', 'P', 'a', 'i', 'Omega', 'T', 'e', 'omega']].rename(columns={'omega': 'omega_arg'})
        
    except Exception as e:
        log.error(f"Failed to parse ORB6 catalog: {e}")
        raise


# Removed parse_wds_measurements_catalog - functionality integrated into parse_wdss_master_catalog

def apply_physical_validation(df: pd.DataFrame) -> None:
    """Apply physical validation to orbital elements."""
    
    # Period validation
    if 'P' in df.columns:
        invalid_P = (df['P'] < MIN_PERIOD_YEARS) | (df['P'] > MAX_PERIOD_YEARS)
        if invalid_P.any():
            n_invalid = invalid_P.sum()
            log.warning(f"Setting {n_invalid} invalid periods to NaN")
            df.loc[invalid_P, 'P'] = np.nan
    
    # Semi-major axis validation
    if 'a' in df.columns:
        invalid_a = (df['a'] < MIN_SEMIMAJOR_AXIS_ARCSEC) | (df['a'] > MAX_SEMIMAJOR_AXIS_ARCSEC)
        if invalid_a.any():
            n_invalid = invalid_a.sum()
            log.warning(f"Setting {n_invalid} invalid semi-major axes to NaN")
            df.loc[invalid_a, 'a'] = np.nan
    
    # Eccentricity validation
    if 'e' in df.columns:
        invalid_e = (df['e'] < MIN_ECCENTRICITY) | (df['e'] > MAX_ECCENTRICITY)
        if invalid_e.any():
            n_invalid = invalid_e.sum()
            log.warning(f"Setting {n_invalid} invalid eccentricities to NaN")
            df.loc[invalid_e, 'e'] = np.nan
    
    # Inclination validation
    if 'i' in df.columns:
        invalid_i = (df['i'] < MIN_INCLINATION_DEG) | (df['i'] > MAX_INCLINATION_DEG)
        if invalid_i.any():
            n_invalid = invalid_i.sum()
            log.warning(f"Setting {n_invalid} invalid inclinations to NaN")
            df.loc[invalid_i, 'i'] = np.nan
    
    # Angular elements validation (0-360°)
    for angle_col in ['Omega', 'omega']:
        if angle_col in df.columns:
            invalid_angle = (df[angle_col] < 0.0) | (df[angle_col] > 360.0)
            if invalid_angle.any():
                n_invalid = invalid_angle.sum()
                log.warning(f"Setting {n_invalid} invalid {angle_col} values to NaN")
                df.loc[invalid_angle, angle_col] = np.nan


def create_sqlite_database(df_wds: pd.DataFrame, df_orb6: pd.DataFrame, 
                          df_measurements: Optional[pd.DataFrame], 
                          output_path: str) -> None:
    """Create SQLite database with proper indexing."""
    log.info(f"Creating SQLite database: {output_path}")
    
    conn = sqlite3.connect(output_path)
    
    try:
        # Explicitly drop existing tables
        conn.execute('DROP TABLE IF EXISTS wdss_summary')
        conn.execute('DROP TABLE IF EXISTS orbital_elements')  
        conn.execute('DROP TABLE IF EXISTS measurements')
        log.info("Dropped existing tables")
        
        # Create WDSS summary table
        df_wds.to_sql('wdss_summary', conn, if_exists='replace', index=False)
        conn.execute('CREATE INDEX idx_wdss_summary_id ON wdss_summary(wds_id)')
        conn.execute('CREATE INDEX idx_wdss_summary_wdss_id ON wdss_summary(wdss_id)')
        # Note: coordinate index will be added when coordinates are properly extracted
        # conn.execute('CREATE INDEX idx_wdss_summary_coords ON wdss_summary(ra_deg, dec_deg)')
        log.info(f"Created wdss_summary table with {len(df_wds)} entries")
        
        # Debug ORB6 DataFrame
        log.info(f"ORB6 DataFrame columns: {df_orb6.columns.tolist()}")
        log.info(f"ORB6 DataFrame shape: {df_orb6.shape}")
        log.info(f"ORB6 duplicated columns: {df_orb6.columns.duplicated().any()}")
        if df_orb6.columns.duplicated().any():
            log.error(f"Duplicated column names found: {df_orb6.columns[df_orb6.columns.duplicated()]}")
        
        # Create ORB6 table
        df_orb6.to_sql('orbital_elements', conn, if_exists='replace', index=False)
        conn.execute('CREATE INDEX idx_orbital_elements_id ON orbital_elements(wds_id)')
        log.info(f"Created orbital_elements table with {len(df_orb6)} entries")
        
        # Create measurements table if available
        if df_measurements is not None:
            df_measurements.to_sql('measurements', conn, if_exists='replace', index=False)
            conn.execute('CREATE INDEX idx_measurements_wdss_id ON measurements(wdss_id)')
            conn.execute('CREATE INDEX idx_measurements_epoch ON measurements(epoch)')
            log.info(f"Created measurements table with {len(df_measurements)} entries")
        
        # Vacuum database for optimal performance
        conn.execute('VACUUM')
        
        log.info("SQLite database created successfully")
        
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Convert WDSS catalogs to SQLite')
    parser.add_argument('--wdss-master-file', required=True, help='Path to WDSS master catalog file')
    parser.add_argument('--orb6', required=True, help='Path to ORB6 catalog')
    parser.add_argument('--output', required=True, help='Output SQLite database path')
    parser.add_argument('--force', action='store_true', help='Overwrite existing database')
    
    args = parser.parse_args()
    
    # Check if output exists
    if Path(args.output).exists() and not args.force:
        log.error(f"Output file {args.output} already exists. Use --force to overwrite.")
        sys.exit(1)
    
    try:
        # Parse WDSS master file - single pass reading
        log.info("Parsing WDSS master catalog...")
        df_components, df_measurements, df_correspondence = parse_wdss_master_catalog(args.wdss_master_file)
        
        # Generate summary table from components and measurements
        log.info("Generating summary table...")
        df_wds = generate_summary_table(df_components, df_measurements, df_correspondence)
        
        # Parse ORB6 catalog
        log.info("Parsing ORB6 catalog...")
        df_orb6 = parse_orb6_catalog(args.orb6)
        
        # Create SQLite database
        log.info("Creating SQLite database...")
        create_sqlite_database(df_wds, df_orb6, df_measurements, args.output)
        
        log.info("Conversion completed successfully!")
        
    except Exception as e:
        log.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
