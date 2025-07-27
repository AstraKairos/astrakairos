"""
Functions for generating summary tables of astronomical data.

This module handles the aggregation and pivoting of component
and measurement data to create optimized summary tables.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)


def generate_summary_table(df_components: pd.DataFrame, df_measurements: pd.DataFrame, 
                          df_correspondence: pd.DataFrame, df_el_badry: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Generate the summary table with integrated El-Badry cross-matching.
    
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
        
        df_summary['in_el_badry_catalog'] = df_summary['in_el_badry_catalog'].fillna(False)
        matches = df_summary['in_el_badry_catalog'].sum()
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
