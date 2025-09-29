"""
Functions for generating summary tables of astronomical data.

This module handles the aggregation and pivoting of component
and measurement data to create optimized summary tables.
"""

import json
import logging
import re
import pandas as pd
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)


def generate_summary_table(df_components: pd.DataFrame, df_measurements: pd.DataFrame, 
                          df_correspondence: pd.DataFrame, df_el_badry: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Generate the summary table treating each component pair as independent system.
    
    Each component pair (AC, BD, CD, CE, etc.) becomes a separate row/system.
    
    Args:
        df_components: Component data from WDSS catalogs
        df_measurements: Measurement data from WDSS catalogs
        df_correspondence: WDS-WDSS correspondence mapping
        df_el_badry: El-Badry catalog data with matched pairs (optional)
    
    Returns:
        DataFrame with summary data where each row represents one component pair
    """
    log.info("Generating summary table - each component pair as independent system")

    total_measurements = len(df_measurements)
    total_pairs = df_measurements['pair'].nunique() if not df_measurements.empty else 0
    log.info(f"Processing {total_measurements:,} measurements across {total_pairs} component pairs")

    log.info("Step 1/5: Aggregating measurements by component pair")
    agg_measurements = _aggregate_measurements(df_measurements)

    log.info("Step 2/5: Creating component data for each pair")
    wide_components = _pivot_components(df_components, df_measurements)

    log.info("Step 3/5: Merging data sources")
    df_summary = _merge_data_sources(agg_measurements, wide_components, df_correspondence)

    log.info("Step 4/5: Enriching with El-Badry catalog")
    df_summary = _enrich_with_el_badry_data(df_summary, df_el_badry)

    log.info("Step 5/5: Finalizing schema")
    df_summary = _finalize_and_rename_columns(df_summary)
    
    log.info(f"Summary complete: {len(df_summary):,} independent component pair systems")
    return df_summary


def _aggregate_measurements(df_measurements: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate measurement data by component pair with error propagation.
    
    Each component pair (AC, BD, CD, CE, etc.) is treated as a completely 
    independent system.
    
    Args:
        df_measurements: DataFrame with measurement data
        
    Returns:
        DataFrame with aggregated measurements indexed by (wdss_id, pair)
    """
    if df_measurements.empty:
        return pd.DataFrame(columns=['wdss_id', 'component_pair']).set_index(['wdss_id', 'component_pair'])
    
    df_measurements = df_measurements.copy()
    df_measurements['wdss_id'] = df_measurements['wdss_id'].astype('category')
    df_measurements['pair'] = df_measurements['pair'].astype('category')
    
    df_measurements['system_pair_id'] = df_measurements['wdss_id'].astype(str) + '-' + df_measurements['pair'].astype(str)
    
    result = (
        df_measurements.sort_values(['wdss_id', 'pair', 'epoch'])
        .groupby(['wdss_id', 'pair'], observed=True)
        .agg(
            system_pair_id=('system_pair_id', 'first'),
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
        .reset_index()
        .set_index('system_pair_id')
    )
    
    return result


def _pivot_components(df_components: pd.DataFrame, df_measurements: pd.DataFrame) -> pd.DataFrame:
    """
    Create component data matched to each component pair system.
    
    For each measurement pair (AC, BD, etc.), we extract the relevant 
    component data and create a system entry.
    
    Args:
        df_components: DataFrame with component data
        df_measurements: DataFrame with measurement data (for pair information)
        
    Returns:
        DataFrame with component data indexed by system_pair_id
    """
    if df_components.empty or df_measurements.empty:
        return pd.DataFrame(columns=['system_pair_id']).set_index('system_pair_id')

    unique_pairs = df_measurements[['wdss_id', 'pair']].drop_duplicates()
    unique_pairs['system_pair_id'] = unique_pairs['wdss_id'] + '-' + unique_pairs['pair']
    
    # Instead of taking only the first component, consolidate all components per system
    # This preserves Gaia IDs from all components (A, B, C, etc.)
    consolidated_components = _consolidate_all_components(df_components)
    
    df_pair_components = unique_pairs.merge(
        consolidated_components, 
        on='wdss_id', 
        how='left'
    )
    
    # Select and rename columns for final output
    component_columns = {
        'system_pair_id': 'system_pair_id',
        'wdss_id': 'wdss_id', 
        'pair': 'component_pair',
        'ra_deg': 'ra_deg',
        'dec_deg': 'dec_deg',
        'vmag': 'vmag',
        'kmag': 'kmag',
        'spectral_type': 'spectral_type',
        'pm_ra': 'pm_ra',
        'pm_dec': 'pm_dec',
        'parallax': 'parallax',
        'name': 'name'
    }
    
    # Dynamically include all Gaia ID columns
    gaia_columns = [col for col in df_pair_components.columns if col.startswith('gaia_id_')]
    for col in gaia_columns:
        component_columns[col] = col
    
    # Keep only required columns and rename
    available_cols = {k: v for k, v in component_columns.items() if k in df_pair_components.columns}
    df_pair_components = df_pair_components[list(available_cols.keys())].rename(columns=available_cols)
    
    # For each component pair, extract the appropriate Gaia IDs based on the pair letters
    # e.g., for pair "AB", we want gaia_id_A and gaia_id_B
    gaia_id_columns = [col for col in df_pair_components.columns if col.startswith('gaia_id_')]
    if gaia_id_columns:
        df_pair_components['gaia_id_primary'] = None
        df_pair_components['gaia_id_secondary'] = None
        
        for idx, row in df_pair_components.iterrows():
            pair = row.get('component_pair', '')
            if len(pair) >= 2:
                primary_component = pair[0]  # e.g., 'A' from 'AB'
                secondary_component = pair[1]  # e.g., 'B' from 'AB'
                
                # Map component letters to Gaia IDs
                gaia_id_primary = row.get(f'gaia_id_{primary_component}')
                gaia_id_secondary = row.get(f'gaia_id_{secondary_component}')
                
                df_pair_components.loc[idx, 'gaia_id_primary'] = gaia_id_primary
                df_pair_components.loc[idx, 'gaia_id_secondary'] = gaia_id_secondary
    
    # Set index and return
    return df_pair_components.set_index('system_pair_id')


def _consolidate_all_components(df_components: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate all components per system into a single row.
    
    This function merges all individual components (A, B, C, etc.) of each system
    into a single row, preserving all Gaia IDs and taking representative values
    for other fields (from the primary component, typically A).
    
    Args:
        df_components: DataFrame with individual component data
        
    Returns:
        DataFrame with one row per system containing consolidated data
    """
    if df_components.empty:
        return pd.DataFrame(columns=['wdss_id'])
    
    consolidated_rows = []
    
    for wdss_id, group in df_components.groupby('wdss_id'):
        # Use the first component (usually A) as the base for most fields
        base_row = group.sort_values('component').iloc[0].copy()
        
        # Consolidate all Gaia IDs from all components
        gaia_id_columns = [col for col in group.columns if col.startswith('gaia_id_')]
        
        # For each gaia_id_X column, take the first non-null value across all components
        for gaia_col in gaia_id_columns:
            values = group[gaia_col].dropna()
            if not values.empty:
                base_row[gaia_col] = values.iloc[0]
            else:
                base_row[gaia_col] = None
        
        # For the 'name' field, try to consolidate Gaia IDs from all components
        all_names = group['name'].dropna().tolist()
        if all_names:
            # Extract Gaia IDs from all name fields and combine them
            from .parsers import extract_gaia_ids_from_name_field
            combined_gaia_ids = {}
            
            for i, name_field in enumerate(all_names):
                component_letter = group.iloc[i]['component']
                extracted_ids = extract_gaia_ids_from_name_field(str(name_field))
                
                # If extraction found IDs, map them to the actual component letter
                if extracted_ids:
                    # If the extraction didn't include component letter, assign it
                    if 'A' in extracted_ids and len(extracted_ids) == 1:
                        # Single ID extracted as 'A', reassign to actual component
                        gaia_id = extracted_ids['A']
                        combined_gaia_ids[component_letter] = gaia_id
                        # Update the corresponding gaia_id_X column
                        base_row[f'gaia_id_{component_letter}'] = gaia_id
                    else:
                        # Multiple IDs or correctly assigned, merge them
                        for comp, gaia_id in extracted_ids.items():
                            combined_gaia_ids[comp] = gaia_id
                            base_row[f'gaia_id_{comp}'] = gaia_id
            
            # Keep the first name field as representative
            base_row['name'] = all_names[0]
        
        consolidated_rows.append(base_row)
    
    return pd.DataFrame(consolidated_rows)


def _merge_data_sources(agg_measurements: pd.DataFrame, wide_components: pd.DataFrame, 
                       df_correspondence: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all data sources together for the new multi-pair approach.
    
    Args:
        agg_measurements: Aggregated measurement data (by pair)
        wide_components: Component data (by pair)
        df_correspondence: WDS correspondence data
        
    Returns:
        DataFrame with merged data indexed by system_pair_id
    """
    # Reset index to avoid column conflicts during merge
    df_summary_reset = agg_measurements.reset_index()
    wide_components_reset = wide_components.reset_index()
    
    # Rename 'pair' to 'component_pair' in agg_measurements for consistent merge
    if 'pair' in df_summary_reset.columns:
        df_summary_reset = df_summary_reset.rename(columns={'pair': 'component_pair'})
    
    # Merge on wdss_id and component_pair
    df_summary = df_summary_reset.merge(
        wide_components_reset, 
        on=['wdss_id', 'component_pair'], 
        how='outer',
        suffixes=('', '_comp')
    )
    
    # Handle duplicate system_pair_id columns - keep the one from measurements
    if 'system_pair_id_comp' in df_summary.columns:
        # Use system_pair_id from measurements, fill missing with component version
        df_summary['system_pair_id'] = df_summary['system_pair_id'].fillna(df_summary['system_pair_id_comp'])
        df_summary = df_summary.drop(columns=['system_pair_id_comp'])
    
    # For correspondence data, we need to match on the base wdss_id
    if not df_correspondence.empty:
        df_correspondence_indexed = df_correspondence.set_index('wdss_id')
        
        # Merge correspondence data based on wdss_id
        df_summary = df_summary.merge(
            df_correspondence_indexed, 
            left_on='wdss_id', 
            right_index=True, 
            how='left'
        )
    
    # Set index to system_pair_id - use the proper string IDs
    if 'system_pair_id' in df_summary.columns:
        df_summary = df_summary.set_index('system_pair_id')
    elif df_summary.index.name != 'system_pair_id':
        # If we lost the system_pair_id, regenerate it from wdss_id and component_pair
        if 'wdss_id' in df_summary.columns and 'component_pair' in df_summary.columns:
            df_summary['system_pair_id'] = df_summary['wdss_id'] + '-' + df_summary['component_pair']
            df_summary = df_summary.set_index('system_pair_id')
    
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
        
        # Set index back to system_pair_id (NOT wdss_id!)
        if 'system_pair_id' in df_summary_enriched.columns:
            df_summary = df_summary_enriched.set_index('system_pair_id')
        else:
            df_summary = df_summary_enriched
        
        df_summary['in_el_badry_catalog'] = df_summary['in_el_badry_catalog'].fillna(False).astype(bool)
        matches = df_summary['in_el_badry_catalog'].sum()
        log.info(f"Successfully cross-matched {matches} systems with El-Badry catalog using efficient pair-wise matching.")
        
    else:
        df_summary['R_chance_align'] = np.nan
        df_summary['binary_type'] = None
        df_summary['in_el_badry_catalog'] = False
    
    return df_summary


def _finalize_and_rename_columns(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Finalize columns and rename for SQLite schema.
    
    Args:
        df_summary: Summary DataFrame to finalize
        
    Returns:
        DataFrame with finalized column structure including component_pair
    """
    # Import here to avoid circular imports
    from .parsers import extract_gaia_ids_from_name_field
    from astrakairos.config import GAIA_ID_PATTERN
    
    if df_summary.index.name == 'system_pair_id':
        df_summary.reset_index(inplace=True)

    log.info(f"Finalizing schema - Available columns: {list(df_summary.columns)[:10]}")

    df_summary.rename(columns={
        'wds_correspondence': 'wds_id_original',
    }, inplace=True)

    # Create clean wds_id: use correspondence if available, otherwise wdss_id
    if 'wds_id_original' in df_summary.columns:
        if 'wdss_id' in df_summary.columns:
            df_summary['wds_id'] = df_summary['wds_id_original'].fillna(df_summary['wdss_id'])
        else:
            log.warning("wdss_id column missing, using wds_id_original only")
            df_summary['wds_id'] = df_summary['wds_id_original']
    else:
        if 'wdss_id' in df_summary.columns:
            df_summary['wds_id'] = df_summary['wdss_id']
        else:
            log.error(f"Neither wds_id_original nor wdss_id found. Available columns: {list(df_summary.columns)}")
            raise KeyError("Missing required wdss_id or wds_id_original columns")
    
    gaia_regex = re.compile(GAIA_ID_PATTERN)

    def _normalize_gaia_value(value):
        if pd.isna(value) or value is None:
            return None
        candidate = str(value).strip()
        if not candidate:
            return None
        match = gaia_regex.search(candidate)
        if match:
            return match.group(1)
        return None

    def _coalesce_gaia_columns(df: pd.DataFrame, base_name: str) -> None:
        candidate_cols = [col for col in df.columns if col.startswith(base_name)]
        if not candidate_cols:
            df[base_name] = None
            return

        candidate_cols.sort(key=lambda col: (0 if col == base_name else 1, col))

        coalesced = df[candidate_cols].apply(
            lambda row: next((normalized for normalized in (_normalize_gaia_value(val) for val in row) if normalized), None),
            axis=1
        )

        df[base_name] = coalesced

        for col in candidate_cols:
            if col != base_name:
                df.drop(columns=col, inplace=True)

    # Normalize all Gaia ID columns to string digits
    all_gaia_columns = [col for col in df_summary.columns if col.startswith('gaia_id_')]
    for col in all_gaia_columns:
        df_summary[col] = df_summary[col].apply(_normalize_gaia_value)

    # Coalesce primary and secondary identifiers
    _coalesce_gaia_columns(df_summary, 'gaia_id_primary')
    _coalesce_gaia_columns(df_summary, 'gaia_id_secondary')

    # Require both primary and secondary Gaia IDs for downstream validation logic
    if {'gaia_id_primary', 'gaia_id_secondary'}.issubset(df_summary.columns):
        valid_gaia_mask = df_summary['gaia_id_primary'].notna() & df_summary['gaia_id_secondary'].notna()
        dropped_pairs = (~valid_gaia_mask).sum()
        if dropped_pairs:
            log.info(f"Dropping {dropped_pairs} component pairs without complete Gaia IDs")
        df_summary = df_summary[valid_gaia_mask].copy()

    # Extract Gaia source IDs from the 'name' field
    log.info("Extracting Gaia source IDs from 'name' field...")
    df_summary['gaia_source_ids'] = None
    
    # First try to consolidate existing individual Gaia ID columns (if they exist)
    gaia_id_columns = [col for col in df_summary.columns if col.startswith('gaia_id_') and col not in ['gaia_id_primary', 'gaia_id_secondary']]
    
    gaia_extraction_count = 0
    for idx, row in df_summary.iterrows():
        gaia_dict = {}
        
        # Method 1: Use existing gaia_id_ columns if available
        if gaia_id_columns:
            for col in gaia_id_columns:
                component = col.replace('gaia_id_', '')
                gaia_id = row[col]
                if pd.notna(gaia_id) and gaia_id is not None and str(gaia_id).strip():
                    gaia_dict[component] = str(gaia_id).strip()

        # Ensure primary/secondary IDs are represented in the component mapping
        pair_label = row.get('component_pair', '')
        if len(pair_label) >= 2:
            primary_component = pair_label[0]
            secondary_component = pair_label[1]
            primary_id = row.get('gaia_id_primary')
            secondary_id = row.get('gaia_id_secondary')
            if primary_id and primary_component not in gaia_dict:
                gaia_dict[primary_component] = str(primary_id)
            if secondary_id and secondary_component not in gaia_dict:
                gaia_dict[secondary_component] = str(secondary_id)
        
        # Method 2: Extract from name field if no existing columns or if they're incomplete
        if (not gaia_dict or len(gaia_dict) < 2) and 'name' in df_summary.columns:
            name_field = row.get('name')
            if pd.notna(name_field) and name_field is not None:
                extracted_ids = extract_gaia_ids_from_name_field(str(name_field))
                if extracted_ids:
                    for component, gaia_id in extracted_ids.items():
                        if component not in gaia_dict:
                            gaia_dict[component] = gaia_id
                    gaia_extraction_count += 1
        
        if gaia_dict:
            # Convert to JSON string for storage in SQLite
            df_summary.loc[idx, 'gaia_source_ids'] = json.dumps(gaia_dict)
    
    if gaia_extraction_count > 0:
        log.info(f"Successfully extracted Gaia source IDs from 'name' field for {gaia_extraction_count} systems")
    elif gaia_id_columns:
        log.info(f"Using pre-existing Gaia ID columns: {gaia_id_columns}")
    else:
        log.warning("No Gaia source IDs found in either individual columns or 'name' field")
    
    # Select final columns including the CRITICAL component_pair field
    final_cols = [
        'system_pair_id', 'wds_id', 'wdss_id', 'component_pair',  # ← KEY ADDITIONS
        'discoverer_designation', 'date_first', 'date_last', 'n_obs',
        'pa_first', 'pa_last', 'sep_first', 'sep_last', 
        'pa_first_error', 'pa_last_error', 'sep_first_error', 'sep_last_error',
        'vmag', 'kmag', 'ra_deg', 'dec_deg', 'spectral_type', 'parallax', 'pm_ra', 'pm_dec', 'name',
        'in_el_badry_catalog', 'R_chance_align', 'binary_type',
        'gaia_id_primary', 'gaia_id_secondary', 'gaia_source_ids'  # Add consolidated Gaia identifiers
    ]
    
    # Add missing columns with NaN if they don't exist
    for col in final_cols:
        if col not in df_summary.columns:
            df_summary[col] = np.nan
    
    return df_summary[final_cols]
