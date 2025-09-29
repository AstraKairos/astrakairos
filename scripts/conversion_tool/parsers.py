"""
Functions for parsing different astronomical catalog formats.

This module contains specialized functions for parsing:
- WDSS master catalogs (components, measurements, correspondence)
- ORB6 orbital elements catalog
- El-Badry physical binaries catalog
"""

import logging
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from astropy.table import Table

from astrakairos.config import (
    DAYS_PER_JULIAN_YEAR, CENTURIES_PER_YEAR, MILLIARCSEC_PER_ARCSEC,
    MIN_PERIOD_YEARS, MAX_PERIOD_YEARS,
    MIN_SEMIMAJOR_AXIS_ARCSEC, MAX_SEMIMAJOR_AXIS_ARCSEC,
    MIN_ECCENTRICITY, MAX_ECCENTRICITY,
    MIN_INCLINATION_DEG, MAX_INCLINATION_DEG,
    TECHNIQUE_ERROR_MODEL, WDSS_MEASUREMENT_COLSPECS, WDSS_COMPONENT_COLSPECS,
    ORB6_ERROR_VALIDATION_THRESHOLDS, ORB6_COLSPECS, ORB6_COLUMN_NAMES,
    GAIA_ID_PATTERN
)
from astrakairos.exceptions import (
    CatalogParsingError, FileFormatError, DataValidationError
)
from astrakairos.utils.io import (
    safe_int, safe_float, parse_wdss_coordinates, parse_wdss_coordinate_string
)

log = logging.getLogger(__name__)


def extract_gaia_ids_from_name_field(name_field: str) -> Dict[str, str]:
    """
    Extract Gaia DRx/ERx source IDs from the WDSS 'name' field.
    
    The name field may contain Gaia IDs in various formats, often associated
    with component letters. This function attempts to parse and match them.
    Supports any component letter from A-Z and beyond.
    
    Args:
        name_field: Content of the WDSS 'name' field
        
    Returns:
        Dictionary mapping component letters to Gaia source IDs
        Example: {'A': '5853498713190525568', 'B': '5853498713190678912'}
        
    Examples:
        "Gaia DR3 5853498713190525568 A, Gaia DR3 5853498713190678912 B"
        "Gaia ER3 3907774724056384512 B"
        "5853498713190525568 A 5853498713190678912 B"
        "A:5853498713190525568 B:5853498713190678912"
        "6.65 DR2 3925443088835673600"  # Single ID without component -> assumed 'A'
        "D:1234567890123456789 E:9876543210987654321"  # For higher-order systems
    """
    if not name_field or pd.isna(name_field):
        return {}
    
    # Find all Gaia IDs (18-20 digit numbers)
    gaia_ids = re.findall(GAIA_ID_PATTERN, str(name_field))
    
    # Initialize result dictionary
    result = {}
    
    if not gaia_ids:
        return result
    
    name_upper = str(name_field).upper()
    
    # Strategy 1: Find all component-ID pairs using more specific patterns
    # Look for patterns like "A:123", "123 A", "Gaia DR3 123 A", etc.
    
    # Pattern for colon-separated format: A:123456...
    colon_matches = re.findall(r'\b([A-Z]):(\d{18,20})\b', name_upper)
    for component, gaia_id in colon_matches:
        result[component] = gaia_id
    
    # Pattern for Gaia DRx/ERx formats: Gaia DR3 123456... A, Gaia ER3 123... B
    gaia_release_matches = re.findall(r'GAIA\s+(?:DR|ER)\d\s+(\d{18,20})\s+([A-Z])\b', name_upper)
    for gaia_id, component in gaia_release_matches:
        if component not in result:  # Don't overwrite colon matches
            result[component] = gaia_id

    # Pattern for release tokens without explicit "Gaia" prefix (e.g., DR2 123... A)
    release_matches = re.findall(r'\b(?:DR|ER)\d\s+(\d{18,20})\s+([A-Z])\b', name_upper)
    for gaia_id, component in release_matches:
        if component not in result:
            result[component] = gaia_id
    
    # Pattern for number-space-letter: 123456... A (but not if already matched)
    # We need to be careful to match the closest number to each letter
    remaining_components = set(re.findall(r'\b([A-Z])\b', name_upper)) - set(result.keys())
    
    for component in remaining_components:
        # Look for number immediately before this component letter
        pattern = rf'(\d{{18,20}})\s+{component}\b'
        match = re.search(pattern, name_upper)
        if match:
            # Check if this ID is already assigned
            candidate_id = match.group(1)
            if candidate_id not in result.values():
                result[component] = candidate_id
    
    # Strategy 2: Handle cases with Gaia IDs but no explicit component letters
    # If we found Gaia IDs but no component assignments, assign them to A, B, C, etc.
    assigned_ids = set(result.values())
    unassigned_ids = [gaia_id for gaia_id in gaia_ids if gaia_id not in assigned_ids]
    
    if unassigned_ids:
        # Find the first available component letters starting from 'A'
        used_components = set(result.keys())
        available_components = [chr(ord('A') + i) for i in range(26) if chr(ord('A') + i) not in used_components]
        
        for i, gaia_id in enumerate(unassigned_ids):
            if i < len(available_components):
                component = available_components[i]
                result[component] = gaia_id
                log.debug(f"Assigned unassigned Gaia ID {gaia_id} to component {component}")
    
    # Log successful extractions
    if result:
        log.debug(f"Extracted Gaia IDs from '{name_field[:50]}...': {result}")
    
    return result


def parse_el_badry_catalog(filepath: str) -> pd.DataFrame:
    """
    Parse the El-Badry et al. (2021) catalog from FITS format.
    
    Args:
        filepath: Path to the El-Badry FITS catalog
        
    Returns:
        DataFrame with consistent column naming for cross-matching
        
    Raises:
        CatalogParsingError: If catalog cannot be parsed
        FileFormatError: If file format is invalid
    """
    try:
        log.info(f"Loading El-Badry catalog: {filepath}")
        
        el_badry_table = Table.read(filepath)
        df = el_badry_table.to_pandas()
        log.info(f"Loaded El-Badry catalog with {len(df)} binary systems")
        
        # Validate required columns
        required_cols = ['source_id1', 'source_id2']
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            raise CatalogParsingError(f"El-Badry catalog missing required columns: {missing_required}")
        
        # Add missing optional columns with default values
        optional_cols = ['R_chance_align', 'binary_type']
        for col in optional_cols:
            if col not in df.columns:
                df[col] = None
                log.warning(f"Optional column '{col}' not found in El-Badry catalog, using None")
        
        # Rename columns for consistency
        df.rename(columns={
            'source_id1': 'gaia_source_id_1', 
            'source_id2': 'gaia_source_id_2'
        }, inplace=True)
        
        # Create bidirectional pair IDs for robust matching
        df['gaia_source_id_1'] = df['gaia_source_id_1'].astype(str)
        df['gaia_source_id_2'] = df['gaia_source_id_2'].astype(str)
        
        sorted_ids = np.sort(df[['gaia_source_id_1', 'gaia_source_id_2']].values, axis=1)
        df['pair_id'] = [f"{id1}_{id2}" for id1, id2 in sorted_ids]
        
        # Remove duplicate pairs
        df_cleaned = df.drop_duplicates(subset=['pair_id'], keep='first')
        log.info(f"Found {len(df_cleaned)} unique binary pairs in El-Badry catalog")
        
        return df_cleaned
        
    except ImportError as e:
        raise CatalogParsingError("astropy is required to read FITS files. Install with: pip install astropy") from e
    except Exception as e:
        raise CatalogParsingError(f"Failed to load El-Badry catalog: {e}") from e


def parse_orb6_catalog(filepath: str) -> pd.DataFrame:
    """
    Parse the ORB6 catalog from its fixed-width text format.
    
    Args:
        filepath: Path to the ORB6 catalog file
        
    Returns:
        DataFrame with processed orbital elements
        
    Raises:
        CatalogParsingError: If parsing fails
        FileFormatError: If file format is invalid
    """
    log.info(f"Parsing ORB6 catalog: {filepath}")

    try:
        # Use centralized column specifications from config
        df = pd.read_fwf(filepath, colspecs=ORB6_COLSPECS, names=ORB6_COLUMN_NAMES, 
                         comment='R',  # Ignore header/ruler lines
                         dtype=str)   # Read everything as text for robust manual control
        
        log.info(f"Read {len(df)} raw entries from ORB6")
        
        # Initial cleanup
        df.dropna(subset=['wds_id'], inplace=True)
        df['wds_id'] = df['wds_id'].str.strip()

        # Convert numeric columns and error columns to float
        numeric_cols = ['i_str', 'e_i', 'e_str', 'e_e', 'e_P', 'e_a', 'e_Omega', 'e_T', 'e_omega_arg', 'grade']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Processing fields with units/flags
        
        # Period (P)
        df['P'] = pd.to_numeric(df['P_str'].str.rstrip('ydc'), errors='coerce')
        df.loc[df['P_str'].str.endswith('d', na=False), 'P'] /= DAYS_PER_JULIAN_YEAR
        df.loc[df['P_str'].str.endswith('c', na=False), 'P'] *= CENTURIES_PER_YEAR
        # Apply same conversion to errors
        df.loc[df['P_str'].str.endswith('d', na=False), 'e_P'] /= DAYS_PER_JULIAN_YEAR
        df.loc[df['P_str'].str.endswith('c', na=False), 'e_P'] *= CENTURIES_PER_YEAR

        # Semi-major axis (a)
        df['a'] = pd.to_numeric(df['a_str'].str.rstrip('amM'), errors='coerce')
        df.loc[df['a_str'].str.endswith('m', na=False), 'a'] /= MILLIARCSEC_PER_ARCSEC
        df.loc[df['a_str'].str.endswith('m', na=False), 'e_a'] /= MILLIARCSEC_PER_ARCSEC

        # Inclination (i) - already processed as numeric
        df['i'] = df['i_str']

        # Node (Omega) - preserve flags in separate column
        df['Omega_flag'] = df['Omega_str'].str.extract(r'([*q]+)$')
        df['Omega'] = pd.to_numeric(df['Omega_str'].str.rstrip('*q'), errors='coerce')

        # Time of periastron (T) - preserve unit flags and ignore conversion for now
        df['T_flag'] = df['T_str'].str.extract(r'([cdmy]+)$')
        df['T'] = pd.to_numeric(df['T_str'].str.rstrip('cdmy'), errors='coerce')
        
        # Eccentricity (e) - already processed as numeric
        df['e'] = df['e_str']

        # Argument of periastron (omega) - preserve flags
        df['omega_flag'] = df['omega_str'].str.extract(r'([q]+)$')
        df['omega_arg'] = pd.to_numeric(df['omega_str'].str.rstrip('q'), errors='coerce')
        
        # Validate that errors are positive and reasonable using centralized thresholds
        error_cols = ['e_P', 'e_a', 'e_i', 'e_Omega', 'e_T', 'e_e', 'e_omega_arg']
        for col in error_cols:
            # Set negative or zero errors to NaN (they're meaningless)
            df.loc[df[col] <= 0, col] = np.nan
            # Set unreasonably large errors to NaN using centralized thresholds
            if col in ORB6_ERROR_VALIDATION_THRESHOLDS:
                threshold = ORB6_ERROR_VALIDATION_THRESHOLDS[col]
                df.loc[df[col] > threshold, col] = np.nan

        # Apply physical validation to main orbital elements
        _apply_physical_validation(df)
        
        # Remove duplicates, keeping first orbit (typically most recent/best)
        df.drop_duplicates(subset=['wds_id'], keep='first', inplace=True)
        
        log.info(f"Processed {len(df)} valid and unique ORB6 entries")
        
        # Select final columns for database
        final_cols = [
            'wds_id', 'P', 'e_P', 'a', 'e_a', 'i', 'e_i', 'Omega', 'e_Omega',
            'T', 'e_T', 'e', 'e_e', 'omega_arg', 'e_omega_arg', 'grade'
        ]
        return df[final_cols]

    except Exception as e:
        raise CatalogParsingError(f"Failed to parse ORB6 catalog: {e}") from e


def parse_wdss_master_catalog(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the WDSS master catalog file to extract all data components.
    
    Args:
        filepath: Path to the WDSS master catalog file
        
    Returns:
        A tuple containing three DataFrames:
            - df_components: Stellar component data
            - df_measurements: Orbital measurement data  
            - df_correspondence: WDS-WDSS designation correspondence mapping
              
    Raises:
        FileNotFoundError: If the catalog file doesn't exist
        CatalogParsingError: If parsing fails or file format is invalid
    """
    log.info(f"Parsing WDSS master catalog using pandas.read_fwf: {filepath}")
    
    try:
        # Read the entire file using pandas.read_fwf with minimal column specification
        basic_colspecs = [
            (0, 14),    # wdss_id
            (15, 18),   # component/pair  
            (18, 160)   # rest of data
        ]
        
        df_raw = pd.read_fwf(
            filepath, 
            colspecs=basic_colspecs, 
            names=['wdss_id', 'component_pair', 'data_rest'],
            dtype=str,
            encoding='latin-1',
            comment='=',  # Skip header lines
            skip_blank_lines=True
        )
        
        # Filter out invalid lines
        df_raw = df_raw.dropna(subset=['wdss_id'])
        df_raw = df_raw[df_raw['wdss_id'].str.strip() != '']
        df_raw = df_raw[df_raw['data_rest'].str.len() >= 100]  # Minimum line length
        
        log.info(f"Read {len(df_raw)} valid lines from WDSS catalog")
        
        # Separate component lines (single letter) from measurement lines (2+ letters)
        df_raw['component_clean'] = df_raw['component_pair'].str.strip()
        
        component_mask = (df_raw['component_clean'].str.len() == 1) & df_raw['component_clean'].str.isalpha()
        measurement_mask = (df_raw['component_clean'].str.len() >= 2) & df_raw['component_clean'].str.isalpha()
        
        df_component_lines = df_raw[component_mask].copy()
        df_measurement_lines = df_raw[measurement_mask].copy()
        
        log.info(f"Separated {len(df_component_lines)} component lines and {len(df_measurement_lines)} measurement lines")
        
        # Parse component lines using centralized column specifications
        df_components = _parse_component_lines_vectorized(df_component_lines)
        
        # Parse measurement lines using centralized column specifications  
        df_measurements = _parse_measurement_lines_vectorized(df_measurement_lines)
        
        # Extract correspondence data from component lines (one per system)
        df_correspondence = _extract_correspondence_vectorized(df_component_lines)
        
        log.info(f"Extracted {len(df_components)} components, {len(df_measurements)} measurements, {len(df_correspondence)} correspondences")
        
        return df_components, df_measurements, df_correspondence
        
    except pd.errors.EmptyDataError:
        raise FileFormatError(f"File {filepath} is empty or contains no valid data")
    except Exception as e:
        raise CatalogParsingError(f"Failed to parse WDSS master catalog {filepath}: {e}") from e


def estimate_uncertainty_from_technique(technique: str) -> Tuple[float, float]:
    """
    Estimate positional uncertainties based on observation technique.
    
    Args:
        technique: Observation technique code (e.g., 'S', 'Hg', 'M')
    
    Returns:
        tuple: (theta_uncertainty_deg, rho_uncertainty_arcsec)
    """
    technique = technique.strip().upper()
    
    # Get uncertainty model from config (literature-based)
    if technique in TECHNIQUE_ERROR_MODEL:
        model = TECHNIQUE_ERROR_MODEL[technique]
        return (model['pa_error'], model['rho_error'])
    else:
        # Conservative default for unknown techniques
        default = TECHNIQUE_ERROR_MODEL['DEFAULT']
        return (default['pa_error'], default['rho_error'])


def _apply_physical_validation(df: pd.DataFrame) -> None:
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


def _parse_component_lines_vectorized(df_lines: pd.DataFrame) -> pd.DataFrame:
    """Parse component lines using vectorized operations with centralized column specs."""
    # Reconstruct full lines for parsing - preserve original spacing
    # Original WDSS format: wdss_id (14 chars) + 2 spaces + component + padding + data  
    df_lines['full_line'] = df_lines['wdss_id'] + '  ' + df_lines['component_pair'].str.strip() + '      ' + df_lines['data_rest'].str.lstrip()
    
    # Use centralized column specifications
    cols = WDSS_COMPONENT_COLSPECS
    
    # Extract all fields using vectorized string operations
    df_components = pd.DataFrame({
        'wdss_id': df_lines['wdss_id'].str.strip(),
        'component': df_lines['component_clean'],
        'date_first': df_lines['full_line'].str[cols['date_first'][0]:cols['date_first'][1]].apply(safe_float),
        'n_obs': df_lines['full_line'].str[cols['n_obs'][0]:cols['n_obs'][1]].apply(safe_int),
        'pa_first': df_lines['full_line'].str[cols['pa_first'][0]:cols['pa_first'][1]].apply(safe_float),
        'sep_first': df_lines['full_line'].str[cols['sep_first'][0]:cols['sep_first'][1]].apply(safe_float),
        'vmag': df_lines['full_line'].str[cols['vmag'][0]:cols['vmag'][1]].apply(safe_float),
        'kmag': df_lines['full_line'].str[cols['kmag'][0]:cols['kmag'][1]].apply(safe_float),
        'spectral_type': df_lines['full_line'].str[cols['spectral_type'][0]:cols['spectral_type'][1]].str.strip(),
        'pm_ra': df_lines['full_line'].str[cols['pm_ra'][0]:cols['pm_ra'][1]].apply(safe_float),
        'pm_dec': df_lines['full_line'].str[cols['pm_dec'][0]:cols['pm_dec'][1]].apply(safe_float),
        'parallax': df_lines['full_line'].str[cols['parallax'][0]:cols['parallax'][1]].apply(safe_float),
        'name': df_lines['full_line'].str[cols['name'][0]:cols['name'][1]].str.strip()
    })
    
    # Parse coordinates vectorized
    coordinates_str = df_lines['full_line'].str[cols['coordinates'][0]:cols['coordinates'][1]]
    coords_parsed = coordinates_str.apply(parse_wdss_coordinate_string)
    df_components['ra_deg'] = coords_parsed.apply(lambda x: x[0] if x else None)
    df_components['dec_deg'] = coords_parsed.apply(lambda x: x[1] if x else None)
    
    # Fallback to WDSS ID coordinates if needed
    missing_coords = (df_components['ra_deg'].isna()) | (df_components['dec_deg'].isna())
    if missing_coords.any():
        wdss_coords = df_components.loc[missing_coords, 'wdss_id'].apply(parse_wdss_coordinates)
        df_components.loc[missing_coords, 'ra_deg'] = wdss_coords.apply(lambda x: x[0] if x else None)
        df_components.loc[missing_coords, 'dec_deg'] = wdss_coords.apply(lambda x: x[1] if x else None)
    
    # Extract Gaia IDs from the name field for each component
    gaia_extractions = df_components['name'].apply(extract_gaia_ids_from_name_field)
    
    # Create dynamic columns for all found Gaia IDs
    all_components = set()
    for extraction in gaia_extractions:
        all_components.update(extraction.keys())
    
    # Initialize all possible Gaia ID columns
    for component in sorted(all_components):
        df_components[f'gaia_id_{component}'] = gaia_extractions.apply(lambda x: x.get(component))
    
    # For backward compatibility, ensure A, B, C columns exist
    for component in ['A', 'B', 'C']:
        if f'gaia_id_{component}' not in df_components.columns:
            df_components[f'gaia_id_{component}'] = None
    
    # For individual components, try to match the Gaia ID to the component letter
    df_components['gaia_id_component'] = None
    for idx, row in df_components.iterrows():
        component = row['component']
        extraction = gaia_extractions.iloc[idx] if idx < len(gaia_extractions) else {}
        
        # First priority: exact match with component letter
        if component in extraction:
            df_components.loc[idx, 'gaia_id_component'] = extraction[component]
        elif f'gaia_id_{component}' in df_components.columns and row[f'gaia_id_{component}']:
            df_components.loc[idx, 'gaia_id_component'] = row[f'gaia_id_{component}']
        else:
            # Fallback: use any available Gaia ID from the extraction
            if extraction:
                df_components.loc[idx, 'gaia_id_component'] = next(iter(extraction.values()))
    
    return df_components.dropna(subset=['wdss_id', 'component'])


def _parse_measurement_lines_vectorized(df_lines: pd.DataFrame) -> pd.DataFrame:
    """Parse measurement lines using vectorized operations with centralized column specs."""
    # Reconstruct full lines for parsing - preserve original spacing
    # Original WDSS format: wdss_id (14 chars) + 2 spaces + component_pair + padding + data
    df_lines['full_line'] = df_lines['wdss_id'] + '  ' + df_lines['component_pair'].str.strip() + '      ' + df_lines['data_rest'].str.lstrip()
    
    # Use centralized column specifications
    cols = WDSS_MEASUREMENT_COLSPECS
    
    # Extract technique first for error estimation
    technique_series = df_lines['full_line'].str[cols['technique'][0]:cols['technique'][1]].str.strip()
    
    # Extract all measurement fields
    df_measurements = pd.DataFrame({
        'wdss_id': df_lines['wdss_id'].str.strip(),
        'pair': df_lines['component_clean'],
        'epoch': df_lines['full_line'].str[cols['epoch'][0]:cols['epoch'][1]].apply(safe_float),
        'theta': df_lines['full_line'].str[cols['theta'][0]:cols['theta'][1]].apply(safe_float),
        'rho': df_lines['full_line'].str[cols['rho'][0]:cols['rho'][1]].apply(safe_float),
        'mag1': df_lines['full_line'].str[cols['mag1'][0]:cols['mag1'][1]].apply(safe_float),
        'mag2': df_lines['full_line'].str[cols['mag2'][0]:cols['mag2'][1]].apply(safe_float),
        'reference': df_lines['full_line'].str[cols['reference'][0]:cols['reference'][1]].str.strip(),
        'technique': technique_series
    })
    
    # Extract explicit error columns
    theta_error = df_lines['full_line'].str[cols['theta_error'][0]:cols['theta_error'][1]].apply(safe_float)
    rho_error = df_lines['full_line'].str[cols['rho_error'][0]:cols['rho_error'][1]].apply(safe_float)
    
    # Set zero errors to None
    theta_error = theta_error.replace(0.0, None)
    rho_error = rho_error.replace(0.0, None)
    
    # Estimate errors from technique for missing values
    technique_errors = technique_series.apply(estimate_uncertainty_from_technique)
    est_theta_errors = technique_errors.apply(lambda x: x[0])
    est_rho_errors = technique_errors.apply(lambda x: x[1])
    
    # Use explicit errors if available, otherwise use estimates
    df_measurements['theta_error'] = theta_error.fillna(est_theta_errors).infer_objects(copy=False)
    df_measurements['rho_error'] = rho_error.fillna(est_rho_errors).infer_objects(copy=False)
    
    # Flag error source
    df_measurements['error_source'] = 'estimated'
    df_measurements.loc[theta_error.notna() | rho_error.notna(), 'error_source'] = 'mixed'
    df_measurements.loc[theta_error.notna() & rho_error.notna(), 'error_source'] = 'measured'
    
    return df_measurements.dropna(subset=['wdss_id', 'pair'])


def _extract_correspondence_vectorized(df_lines: pd.DataFrame) -> pd.DataFrame:
    """Extract WDS correspondence data from component lines using vectorized operations."""
    # Reconstruct full lines for parsing - preserve original spacing
    df_lines['full_line'] = df_lines['wdss_id'] + '  ' + df_lines['component_pair'].str.strip() + '      ' + df_lines['data_rest'].str.lstrip()
    
    # Use centralized column specifications
    cols = WDSS_COMPONENT_COLSPECS
    
    # Extract correspondence fields
    wds_correspondence = df_lines['full_line'].str[cols['wds_correspondence'][0]:cols['wds_correspondence'][1]].str.strip()
    discoverer_designation = df_lines['full_line'].str[cols['discoverer_designation'][0]:cols['discoverer_designation'][1]].str.strip()
    
    # Create correspondence DataFrame
    df_correspondence = pd.DataFrame({
        'wdss_id': df_lines['wdss_id'].str.strip(),
        'wds_id': wds_correspondence,
        'discoverer_designation': discoverer_designation
    })
    
    # Filter out empty correspondences and keep only one per system
    df_correspondence = df_correspondence[
        (df_correspondence['wds_id'].notna()) & 
        (df_correspondence['wds_id'] != '') & 
        (~df_correspondence['wds_id'].str.isspace())
    ]
    
    # Keep first occurrence per system
    df_correspondence = df_correspondence.drop_duplicates(subset=['wdss_id'], keep='first')
    
    return df_correspondence
