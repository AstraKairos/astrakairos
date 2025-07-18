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


def parse_wds_summary_catalog(filepath: str) -> pd.DataFrame:
    """Parse WDS summary catalog with robust error handling."""
    log.info(f"Parsing WDS summary catalog: {filepath}")
    
    # Configurable column specifications
    colspecs = [
        (0, 10),    # wds_id
        (10, 17),   # discoverer
        (17, 22),   # components
        (23, 27),   # date_first
        (28, 32),   # date_last
        (33, 37),   # obs
        (38, 41),   # pa_first
        (42, 45),   # pa_last
        (46, 51),   # sep_first
        (52, 57),   # sep_last
        (58, 63),   # mag_pri
        (64, 69),   # mag_sec
        (70, 79),   # spec_type
        (112, 130)  # precise_coords_str
    ]
    
    names = [
        'wds_id', 'discoverer', 'components', 'date_first', 'date_last', 'obs',
        'pa_first', 'pa_last', 'sep_first', 'sep_last', 'mag_pri', 'mag_sec',
        'spec_type', 'precise_coords_str'
    ]
    
    try:
        df = pd.read_fwf(filepath, colspecs=colspecs, names=names, dtype=str)
        log.info(f"Read {len(df)} raw entries from WDS summary")
        
        # Clean and validate data
        df.dropna(subset=['wds_id'], inplace=True)
        df['wds_id'] = df['wds_id'].str.strip()
        
        # Convert numeric columns with error handling
        numeric_cols = ['date_first', 'date_last', 'obs', 'pa_first', 'pa_last',
                        'sep_first', 'sep_last', 'mag_pri', 'mag_sec']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Parse coordinates with validation
        coords_str = df['precise_coords_str'].fillna('')
        
        # Parse RA: HHMMSS.SS
        ra_str = df['precise_coords_str'].str[:9].str.strip()
        ra_h = pd.to_numeric(ra_str.str[:2], errors='coerce')
        ra_m = pd.to_numeric(ra_str.str[2:4], errors='coerce')
        ra_s = pd.to_numeric(ra_str.str[4:], errors='coerce')
        df['ra_deg'] = (ra_h + ra_m / 60.0 + ra_s / 3600.0) * 15.0
        
        # Parse Dec: +DDMMSS.S or -DDMMSS.S
        dec_str = df['precise_coords_str'].str[9:].str.strip()
        dec_sign = np.where(dec_str.str.startswith('-'), -1.0, 1.0)
        dec_d = pd.to_numeric(dec_str.str[1:3], errors='coerce')
        dec_m = pd.to_numeric(dec_str.str[3:5], errors='coerce')
        dec_s = pd.to_numeric(dec_str.str[5:], errors='coerce')
        df['dec_deg'] = dec_sign * (dec_d + dec_m / 60.0 + dec_s / 3600.0)
        
        # Remove duplicate entries (keep most observations)
        df.sort_values('obs', ascending=False, inplace=True)
        df.drop_duplicates(subset=['wds_id'], keep='first', inplace=True)
        
        log.info(f"Processed {len(df)} valid WDS summary entries")
        return df
        
    except Exception as e:
        log.error(f"Failed to parse WDS summary catalog: {e}")
        raise


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
        return df[['wds_id', 'P', 'a', 'i', 'Omega', 'T', 'e', 'omega']]
        
    except Exception as e:
        log.error(f"Failed to parse ORB6 catalog: {e}")
        raise


def parse_wds_measurements_catalog(filepath: str) -> pd.DataFrame:
    """Parse WDS measurements catalog with robust error handling."""
    log.info(f"Parsing WDS measurements catalog: {filepath}")
    
    # Configurable column specifications
    colspecs = [
        (0, 14),    # wds_id
        (24, 34),   # epoch
        (36, 43),   # theta
        (51, 52),   # rho_flag
        (52, 61)    # rho
    ]
    
    names = ['wds_id', 'epoch', 'theta', 'rho_flag', 'rho']
    
    try:
        # Process in chunks for large files
        chunk_size = 50000
        chunks = []
        
        for chunk in pd.read_fwf(filepath, colspecs=colspecs, names=names, 
                                 header=None, dtype=str, chunksize=chunk_size):
            chunk['wds_id'] = chunk['wds_id'].str.strip()
            
            # Convert numeric columns
            numeric_cols = ['epoch', 'theta', 'rho']
            for col in numeric_cols:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            
            # Remove invalid entries
            chunk.dropna(subset=['wds_id', 'epoch', 'theta', 'rho'], inplace=True)
            
            # Handle unit conversion for rho if specified as milliarcseconds
            chunk.loc[chunk['rho_flag'] == 'm', 'rho'] /= MILLIARCSEC_PER_ARCSEC
            
            chunks.append(chunk[['wds_id', 'epoch', 'theta', 'rho']])
            log.info(f"Processed chunk with {len(chunk)} measurements")
        
        df = pd.concat(chunks, ignore_index=True)
        log.info(f"Processed {len(df)} total measurements")
        return df
        
    except Exception as e:
        log.error(f"Failed to parse WDS measurements catalog: {e}")
        raise


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
        # Create WDS summary table
        df_wds.to_sql('wds_summary', conn, if_exists='replace', index=False)
        conn.execute('CREATE INDEX idx_wds_summary_id ON wds_summary(wds_id)')
        conn.execute('CREATE INDEX idx_wds_summary_coords ON wds_summary(ra_deg, dec_deg)')
        log.info(f"Created wds_summary table with {len(df_wds)} entries")
        
        # Create ORB6 table
        df_orb6.to_sql('orbital_elements', conn, if_exists='replace', index=False)
        conn.execute('CREATE INDEX idx_orbital_elements_id ON orbital_elements(wds_id)')
        log.info(f"Created orbital_elements table with {len(df_orb6)} entries")
        
        # Create measurements table if available
        if df_measurements is not None:
            df_measurements.to_sql('measurements', conn, if_exists='replace', index=False)
            conn.execute('CREATE INDEX idx_measurements_id ON measurements(wds_id)')
            conn.execute('CREATE INDEX idx_measurements_epoch ON measurements(epoch)')
            log.info(f"Created measurements table with {len(df_measurements)} entries")
        
        # Vacuum database for optimal performance
        conn.execute('VACUUM')
        
        log.info("SQLite database created successfully")
        
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Convert WDS catalogs to SQLite')
    parser.add_argument('--wds-summary', required=True, help='Path to WDS summary catalog')
    parser.add_argument('--orb6', required=True, help='Path to ORB6 catalog')
    parser.add_argument('--measurements', help='Path to WDS measurements catalog (optional)')
    parser.add_argument('--output', required=True, help='Output SQLite database path')
    parser.add_argument('--force', action='store_true', help='Overwrite existing database')
    
    args = parser.parse_args()
    
    # Check if output exists
    if Path(args.output).exists() and not args.force:
        log.error(f"Output file {args.output} already exists. Use --force to overwrite.")
        sys.exit(1)
    
    try:
        # Parse catalogs
        df_wds = parse_wds_summary_catalog(args.wds_summary)
        df_orb6 = parse_orb6_catalog(args.orb6)
        
        df_measurements = None
        if args.measurements:
            df_measurements = parse_wds_measurements_catalog(args.measurements)
        
        # Create SQLite database
        create_sqlite_database(df_wds, df_orb6, df_measurements, args.output)
        
        log.info("Conversion completed successfully!")
        
    except Exception as e:
        log.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
