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
from typing import Optional, Tuple

import numpy as np
import pandas as pd
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
    CatalogParsingError, FileFormatError, DataValidationError, ElBadryCrossmatchError, ConversionProcessError
)
from astrakairos.utils.io import (
    safe_int, safe_float, parse_wdss_coordinates, parse_wdss_coordinate_string
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


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


def perform_el_badry_crossmatch(df_components: pd.DataFrame, df_el_badry: pd.DataFrame) -> pd.DataFrame:
    """
    Perform pair-wise cross-match with El-Badry catalog using efficient pandas operations.
    
    Note: This cross-match specifically targets pairs with components labeled 'A' and 'B',
    as this is the standard for primary binary systems in the WDSS catalog. Systems with
    other component labels (C, D, etc.) are not included in the cross-match.
    
    Args:
        df_components: DataFrame of all components parsed from WDSS
        df_el_badry: Pre-processed El-Badry catalog DataFrame
    
    Returns:
        DataFrame with matched systems including El-Badry physicality data
        
    Raises:
        ElBadryCrossmatchError: If cross-matching fails
    """
    log.info("Performing pair-wise cross-match with El-Badry catalog...")
    
    try:
        # 1. Prepare components: extract Gaia IDs from A and B components only
        df_comps = df_components[df_components['component'].isin(['A', 'B'])].copy()
        # Robust regex to extract Gaia IDs from various formats
        df_comps['gaia_id'] = df_comps['name'].str.extract(GAIA_ID_PATTERN)[0]
        df_comps.dropna(subset=['gaia_id'], inplace=True)
        
        # 2. Pivot to have A and B components in the same row - much cleaner than groupby
        df_pairs = df_comps.pivot(index='wdss_id', columns='component', values='gaia_id').reset_index()
        
        # Handle case where only A or B components exist (return empty result gracefully)
        if 'A' not in df_pairs.columns or 'B' not in df_pairs.columns:
            log.info("Insufficient component pairs for cross-matching (need both A and B components)")
            return pd.DataFrame(columns=['wdss_id', 'R_chance_align', 'binary_type', 'in_el_badry_catalog'])
        
        df_pairs.dropna(subset=['A', 'B'], inplace=True)
        df_pairs.rename(columns={'A': 'gaia_id_A', 'B': 'gaia_id_B'}, inplace=True)

        # 3. Create sorted pair key for both DataFrames
        sorted_ids_wdss = np.sort(df_pairs[['gaia_id_A', 'gaia_id_B']].values, axis=1)
        df_pairs['pair_id'] = [f"{id1}_{id2}" for id1, id2 in sorted_ids_wdss]

        # 4. Perform the cross-match using inner join
        df_matched = pd.merge(
            df_pairs[['wdss_id', 'pair_id']],
            df_el_badry[['pair_id', 'R_chance_align', 'binary_type']],
            on='pair_id',
            how='inner'  # Only keep matches
        )
        
        df_matched['in_el_badry_catalog'] = True
        log.info(f"Cross-match complete. Found {len(df_matched)} WDSS systems in El-Badry catalog.")
        
        return df_matched[['wdss_id', 'R_chance_align', 'binary_type', 'in_el_badry_catalog']]
        
    except Exception as e:
        raise ElBadryCrossmatchError(f"Failed to perform El-Badry cross-match: {e}") from e



def _parse_component_lines_vectorized(df_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Parse component lines using vectorized operations with centralized column specs.
    
    Args:
        df_lines: DataFrame with component lines
        
    Returns:
        DataFrame with parsed component data
    """
    # Reconstruct full lines for parsing
    df_lines['full_line'] = df_lines['wdss_id'] + df_lines['component_pair'].str.ljust(3) + df_lines['data_rest']
    
    # Use centralized column specifications
    cols = WDSS_COMPONENT_COLSPECS
    
    # Extract all fields using vectorized string operations
    df_components = pd.DataFrame({
        'wdss_id': df_lines['wdss_id'].str.strip(),
        'component': df_lines['component_clean'],
        'date_first': df_lines['full_line'].str[cols['date_first'][0]:cols['date_first'][1]].apply(safe_int),
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
    
    return df_components.dropna(subset=['wdss_id', 'component'])


def _parse_measurement_lines_vectorized(df_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Parse measurement lines using vectorized operations with centralized column specs.
    
    Args:
        df_lines: DataFrame with measurement lines
        
    Returns:
        DataFrame with parsed measurement data
    """
    # Reconstruct full lines for parsing
    df_lines['full_line'] = df_lines['wdss_id'] + df_lines['component_pair'].str.ljust(8) + df_lines['data_rest']
    
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
    df_measurements['theta_error'] = theta_error.fillna(est_theta_errors)
    df_measurements['rho_error'] = rho_error.fillna(est_rho_errors)
    
    # Flag error source
    df_measurements['error_source'] = 'estimated'
    df_measurements.loc[theta_error.notna() | rho_error.notna(), 'error_source'] = 'mixed'
    df_measurements.loc[theta_error.notna() & rho_error.notna(), 'error_source'] = 'measured'
    
    return df_measurements.dropna(subset=['wdss_id', 'pair'])


def _extract_correspondence_vectorized(df_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Extract WDS correspondence data from component lines using vectorized operations.
    
    Args:
        df_lines: DataFrame with component lines
        
    Returns:
        DataFrame with correspondence data (one per system)
    """
    # Reconstruct full lines for parsing
    df_lines['full_line'] = df_lines['wdss_id'] + df_lines['component_pair'].str.ljust(3) + df_lines['data_rest']
    
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


def parse_wdss_master_catalog(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the WDSS master catalog file to extract all data components.
    
    This function processes the complete WDSS master catalog file using
    vectorized operations to efficiently extract stellar components,
    orbital measurements, and WDS correspondence data.
    
    Args:
        filepath: Path to the WDSS master catalog file
        
    Returns:
        A tuple containing three DataFrames:
            - df_components: Stellar component data (coordinates, magnitudes, 
              spectral types, proper motions, parallax) indexed by wdss_id
            - df_measurements: Orbital measurement data (epochs, position angles,
              separations with uncertainties) indexed by wdss_id  
            - df_correspondence: WDS-WDSS designation correspondence mapping
              
    Raises:
        FileNotFoundError: If the catalog file doesn't exist
        CatalogParsingError: If parsing fails or file format is invalid
        
    Note:
        Uses vectorized string operations for optimal performance on large
        catalog files. Assumes components A and B represent primary and
        secondary stars for cross-matching purposes.
    """
    log.info(f"Parsing WDSS master catalog using pandas.read_fwf: {filepath}")
    
    try:
        # Read the entire file using pandas.read_fwf with minimal column specification
        # We use very broad columns to capture all data, then post-process
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


def _aggregate_measurements(df_measurements: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate measurement data with error propagation.
    
    Args:
        df_measurements: DataFrame with measurement data
        
    Returns:
        DataFrame with aggregated measurements indexed by wdss_id
    """
    if df_measurements.empty:
        return pd.DataFrame(columns=['wdss_id']).set_index('wdss_id')
    
    return (
        df_measurements.sort_values(['wdss_id', 'epoch'])
        .groupby('wdss_id')
        .agg(
            date_first=('epoch', 'first'),
            date_last=('epoch', 'last'),
            n_obs=('epoch', 'count'),
            pa_first=('theta', 'first'),
            pa_last=('theta', 'last'),
            sep_first=('rho', 'first'),
            sep_last=('rho', 'last'),
            pa_first_error=('theta_error', 'first'),
            pa_last_error=('theta_error', 'last'),
            sep_first_error=('rho_error', 'first'),
            sep_last_error=('rho_error', 'last')
        )
    )


def _pivot_components(df_components: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot component data from long to wide format.
    
    Args:
        df_components: DataFrame with component data
        
    Returns:
        DataFrame with pivoted component data indexed by wdss_id
    """
    if df_components.empty:
        return pd.DataFrame(columns=['wdss_id']).set_index('wdss_id')
    
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
    
    return df_wide_components


def _merge_data_sources(agg_measurements: pd.DataFrame, wide_components: pd.DataFrame, 
                       df_correspondence: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all data sources together.
    
    Args:
        agg_measurements: Aggregated measurement data
        wide_components: Pivoted component data
        df_correspondence: WDS correspondence data
        
    Returns:
        DataFrame with merged data
    """
    df_summary = agg_measurements
    df_summary = df_summary.join(wide_components, how='outer')
    df_summary = df_summary.join(df_correspondence.set_index('wdss_id'), how='left')
    
    return df_summary


def _enrich_with_el_badry_data(df_summary: pd.DataFrame, df_el_badry: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Enrich summary table with El-Badry physicality data.
    
    Args:
        df_summary: Summary DataFrame to enrich
        df_el_badry: El-Badry catalog data (optional)
        
    Returns:
        DataFrame enriched with El-Badry data
    """
    log.info("Enriching summary table with El-Badry physicality data...")
    
    if df_el_badry is not None and not df_el_badry.empty:
        # Reset index to access wdss_id as a column for merging
        df_summary_enriched = df_summary.reset_index()
        
        # Simple left merge with the clean El-Badry data
        df_summary_enriched = pd.merge(
            df_summary_enriched,
            df_el_badry[['wdss_id', 'R_chance_align', 'binary_type', 'in_el_badry_catalog']],
            on='wdss_id',
            how='left'
        )
        
        # Set index back
        df_summary = df_summary_enriched.set_index('wdss_id')
        
        matches = df_summary['in_el_badry_catalog'].fillna(False).sum()
        log.info(f"Successfully cross-matched {matches} systems with El-Badry catalog using efficient pair-wise matching.")
        
    else:
        df_summary['R_chance_align'] = np.nan
        df_summary['binary_type'] = None
        df_summary['in_el_badry_catalog'] = False
    
    return df_summary


def _finalize_and_rename_columns(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Finalize columns and rename for SQLite schema compatibility.
    
    Args:
        df_summary: Summary DataFrame to finalize
        
    Returns:
        DataFrame with finalized column structure
    """
    df_summary.reset_index(inplace=True)

    # The LocalDataSource expects specific column names for compatibility:
    # - 'vmag' for primary magnitude (from pivoted 'vmag_A')
    # - 'kmag' for secondary magnitude (from pivoted 'vmag_B')
    # - Primary component data (A) becomes the default for coordinates and stellar properties
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
    
    # Select final columns to ensure a clean schema with error columns and El-Badry enrichment
    final_cols = [
        'wds_id', 'wdss_id', 'discoverer_designation', 'date_first', 'date_last', 'n_obs',
        'pa_first', 'pa_last', 'sep_first', 'sep_last', 
        'pa_first_error', 'pa_last_error', 'sep_first_error', 'sep_last_error',
        'vmag', 'kmag', 'ra_deg', 'dec_deg', 'spectral_type', 'parallax', 'pm_ra', 'pm_dec', 'name',
        'in_el_badry_catalog', 'R_chance_align', 'binary_type'  # Add El-Badry enrichment columns
    ]
    
    # Add missing columns with NaN if they don't exist
    for col in final_cols:
        if col not in df_summary.columns:
            df_summary[col] = np.nan
    
    return df_summary[final_cols]


def generate_summary_table(df_components: pd.DataFrame, df_measurements: pd.DataFrame, 
                          df_correspondence: pd.DataFrame, df_el_badry: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Generate the summary table with integrated El-Badry cross-matching.
    
    Now performs pair-wise cross-matching with the El-Badry catalog for maximum accuracy,
    rather than individual component matching. This ensures 100% precision in identifying
    systems that are truly in the gold-standard catalog.
    
    Args:
        df_components: Component data from WDSS catalogs
        df_measurements: Measurement data from WDSS catalogs
        df_correspondence: WDS-WDSS correspondence mapping
        df_el_badry: El-Badry catalog data with matched pairs (optional)
    
    Returns:
        DataFrame with summary data and El-Badry enrichment
    """
    log.info("Generating summary table using vectorized operations with error propagation")

    # 1. Aggregate measurements using groupby with error propagation
    agg_measurements = _aggregate_measurements(df_measurements)

    # 2. Pivot component data from long to wide format
    wide_components = _pivot_components(df_components)

    # 3. Merge all data sources together
    df_summary = _merge_data_sources(agg_measurements, wide_components, df_correspondence)

    # 4. Enrich with El-Badry physicality data using pair-wise matching
    df_summary = _enrich_with_el_badry_data(df_summary, df_el_badry)

    # 5. Finalize and rename columns for schema compatibility
    df_summary = _finalize_and_rename_columns(df_summary)
    
    log.info(f"Generated summary table with {len(df_summary)} systems")
    return df_summary

def estimate_uncertainty_from_technique(technique: str) -> Tuple[float, float]:
    """
    Estimate positional uncertainties based on observation technique.
    
    Uses the centralized TECHNIQUE_ERROR_MODEL from config.py, which provides
    robust, literature-based uncertainty estimates that are citable and generalizable.
    
    Args:
        technique: Observation technique code (e.g., 'S', 'Hg', 'M')
    
    Returns:
        tuple: (theta_uncertainty_deg, rho_uncertainty_arcsec)
    
    References:
        Values based on established literature (Heintz 1978, Lindegren et al. 2021, etc.)
        as documented in config.py TECHNIQUE_ERROR_MODEL.
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


def parse_orb6_catalog(filepath: str) -> pd.DataFrame:
    """
    Parse the ORB6 catalog from its fixed-width text format.
    This function reads orbital elements and their formal errors, which are
    interleaved in a single line per record.
    
    Based on orb6format.txt documentation with centralized column specifications.
    
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
        apply_physical_validation(df)
        
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
        
        # Critical spatial index for coordinate queries (dec_deg first for better selectivity)
        conn.execute('CREATE INDEX idx_wdss_summary_coords ON wdss_summary(dec_deg, ra_deg)')
        
        # Additional performance indexes for common query patterns (only if columns exist)
        if 'date_last' in df_wds.columns:
            conn.execute('CREATE INDEX idx_wdss_summary_date_last ON wdss_summary(date_last)')
        if 'n_obs' in df_wds.columns:
            conn.execute('CREATE INDEX idx_wdss_summary_n_obs ON wdss_summary(n_obs)')
        
        log.info(f"Created wdss_summary table with {len(df_wds)} entries and performance indexes")
        
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
    """
    Main entry point for converting WDSS catalogs to SQLite database.
    
    This function coordinates the complete conversion pipeline:
    1. Parses command line arguments for input/output paths
    2. Loads and processes WDSS catalog files using vectorized operations
    3. Optionally cross-matches with El-Badry binary catalog for physicality
    4. Parses ORB6 orbital elements catalog
    5. Applies physical validation and data quality checks
    6. Creates optimized SQLite database with proper indexing
    
    The resulting database contains three main tables:
    - summary: Aggregated stellar component and measurement data
    - orb6: Orbital elements from published orbits
    - measurements: Individual epoch measurements with uncertainties
    
    Command line arguments:
        --wdss-files: Paths to WDSS catalog files (required)
        --orb6: Path to ORB6 orbital elements catalog (required)
        --output: Output SQLite database path (required)
        --force: Overwrite existing database (optional)
        --el-badry-file: El-Badry binary catalog for cross-matching (optional)
        
    Raises:
        SystemExit: On argument parsing errors or file processing failures
        FileNotFoundError: If required input files don't exist
        PermissionError: If output database cannot be created
    """
    parser = argparse.ArgumentParser(description='Convert WDSS catalogs to SQLite')
    parser.add_argument('--wdss-files', required=True, nargs='+', help='Paths to WDSS catalog files (e.g., wdss1.txt wdss2.txt wdss3.txt wdss4.txt)')
    parser.add_argument('--orb6', required=True, help='Path to ORB6 catalog')
    parser.add_argument('--output', required=True, help='Output SQLite database path')
    parser.add_argument('--force', action='store_true', help='Overwrite existing database')
    parser.add_argument('--el-badry-file', help='Path to El-Badry et al. (2021) binary catalog FITS file for cross-matching')
    
    args = parser.parse_args()
    
    # Check if output exists
    if Path(args.output).exists() and not args.force:
        raise ConversionProcessError(f"Output file {args.output} already exists. Use --force to overwrite.")
    
    try:
        # Parse multiple WDSS files and combine them
        log.info(f"Parsing {len(args.wdss_files)} WDSS catalog files...")
        all_components = []
        all_measurements = []
        all_correspondence = []
        
        for wdss_file in args.wdss_files:
            log.info(f"Processing {wdss_file}...")
            df_components, df_measurements, df_correspondence = parse_wdss_master_catalog(wdss_file)
            all_components.append(df_components)
            all_measurements.append(df_measurements)
            all_correspondence.append(df_correspondence)
        
        # Combine all data
        combined_components = pd.concat(all_components, ignore_index=True) if all_components else pd.DataFrame()
        combined_measurements = pd.concat(all_measurements, ignore_index=True) if all_measurements else pd.DataFrame()
        combined_correspondence = pd.concat(all_correspondence, ignore_index=True) if all_correspondence else pd.DataFrame()
        
        # Remove duplicates across files
        if not combined_components.empty:
            combined_components = combined_components.drop_duplicates(subset=['wdss_id', 'component'], keep='first')
        if not combined_measurements.empty:
            combined_measurements = combined_measurements.drop_duplicates(subset=['wdss_id', 'pair', 'epoch'], keep='first')
        if not combined_correspondence.empty:
            combined_correspondence = combined_correspondence.drop_duplicates(subset=['wdss_id'], keep='first')
        
        log.info(f"Combined: {len(combined_components)} components, {len(combined_measurements)} measurements, {len(combined_correspondence)} correspondences")
        
        # Cross-match with El-Badry catalog if provided
        if args.el_badry_file:
            log.info("Cross-matching with El-Badry catalog using improved pair-wise matching...")
            df_el_badry = parse_el_badry_catalog(args.el_badry_file)
            df_el_badry_data = perform_el_badry_crossmatch(combined_components, df_el_badry)
        else:
            log.info("No El-Badry catalog provided, skipping cross-match")
            df_el_badry_data = None
        
        # Generate summary table from components and measurements with El-Badry integration
        log.info("Generating summary table...")
        df_wds = generate_summary_table(combined_components, combined_measurements, combined_correspondence, df_el_badry_data)
        
        # Parse ORB6 catalog
        log.info("Parsing ORB6 catalog...")
        df_orb6 = parse_orb6_catalog(args.orb6)
        
        # Create SQLite database
        log.info("Creating SQLite database...")
        create_sqlite_database(df_wds, df_orb6, combined_measurements, args.output)
        
        log.info("Conversion completed successfully!")
        
    except (CatalogParsingError, FileFormatError, DataValidationError, ElBadryCrossmatchError) as e:
        raise ConversionProcessError(f"Catalog processing failed: {e}") from e
    except Exception as e:
        raise ConversionProcessError(f"Unexpected error during conversion: {e}") from e


if __name__ == '__main__':
    try:
        main()
    except ConversionProcessError as e:
        log.error(f"Conversion failed: {e}")
        sys.exit(1)
