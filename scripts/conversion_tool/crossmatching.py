"""
Functions for cross-matching between astronomical catalogs.

This module handles the cross-matching between the WDSS catalog and the
high-confidence binary catalog from El-Badry et al. (2021) to identify
confirmed physical binary systems.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict

from astrakairos.config import GAIA_ID_PATTERN
from astrakairos.exceptions import ElBadryCrossmatchError

log = logging.getLogger(__name__)


def _process_system_group(group: pd.DataFrame) -> Optional[Dict]:
    """
    Processes a group of components for a single WDSS system to extract an A/B pair.
    
    Returns a dictionary with the Gaia ID pair if both 'A' and 'B' components
    are present in the group; otherwise, returns None.
    """
    components = group.set_index('component')['gaia_id'].to_dict()
    if 'A' in components and 'B' in components:
        return {
            'wdss_id': group['wdss_id'].iloc[0],
            'gaia_id_A': components['A'],
            'gaia_id_B': components['B']
        }
    return None


def perform_el_badry_crossmatch(df_components: pd.DataFrame, df_el_badry: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a pair-wise cross-match with the El-Badry catalog using a robust method.

    This method groups components by system to handle inconsistencies in the WDSS
    catalog and performs the match based on a canonical pair identifier generated
    from the components' Gaia IDs.
    
    Args:
        df_components: DataFrame containing all components from the WDSS catalog.
        df_el_badry: Pre-processed DataFrame from the El-Badry catalog.
    
    Returns:
        A DataFrame with the matched systems and their physicality data.
        
    Raises:
        ElBadryCrossmatchError: If an error occurs during the cross-matching process.
    """
    log.info("Performing pair-wise cross-match with El-Badry catalog...")
    
    if df_components.empty:
        log.warning("Components DataFrame is empty. Cannot perform cross-match.")
        return pd.DataFrame(columns=['wdss_id', 'R_chance_align', 'binary_type', 'in_el_badry_catalog'])

    try:
        # 1. Prepare components: extract Gaia IDs from A and B components.
        df_comps = df_components[df_components['component'].isin(['A', 'B'])].copy()
        df_comps['gaia_id'] = df_comps['name'].str.extract(GAIA_ID_PATTERN)[0]
        
        df_comps.dropna(subset=['gaia_id'], inplace=True)
        log.info(f"Found {len(df_comps)} 'A'/'B' components with a valid Gaia ID.")
        
        if df_comps.empty:
            log.warning("No 'A' or 'B' components with Gaia IDs found to form pairs.")
            return pd.DataFrame(columns=['wdss_id', 'R_chance_align', 'binary_type', 'in_el_badry_catalog'])

        # 2. Group by wdss_id to robustly form A/B pairs.
        log.info("Grouping components by system to form A/B pairs...")
        grouped = df_comps.groupby('wdss_id')
        
        pair_list = [_process_system_group(group) for _, group in grouped]
        valid_pairs = [p for p in pair_list if p is not None]
        
        if not valid_pairs:
            log.warning("No valid A/B pairs were formed from WDSS. El-Badry cross-match will yield no results.")
            return pd.DataFrame(columns=['wdss_id', 'R_chance_align', 'binary_type', 'in_el_badry_catalog'])

        df_pairs = pd.DataFrame(valid_pairs)
        log.info(f"Formed {len(df_pairs)} unique A/B pairs from the WDSS catalog for cross-matching.")

        # 3. Create a canonical pair ID ('pair_id') for both DataFrames.
        sorted_ids_wdss = np.sort(df_pairs[['gaia_id_A', 'gaia_id_B']].values, axis=1)
        df_pairs['pair_id'] = [f"{id1}_{id2}" for id1, id2 in sorted_ids_wdss]

        # 4. Perform the cross-match (merge) using the 'pair_id'.
        df_matched = pd.merge(
            df_pairs[['wdss_id', 'pair_id']],
            df_el_badry[['pair_id', 'R_chance_align', 'binary_type']],
            on='pair_id',
            how='inner'
        )
        
        df_matched['in_el_badry_catalog'] = True
        log.info(f"Cross-match successful: Found {len(df_matched)} WDSS systems in the El-Badry catalog.")
        
        return df_matched[['wdss_id', 'R_chance_align', 'binary_type', 'in_el_badry_catalog']]
        
    except Exception as e:
        raise ElBadryCrossmatchError(f"Failed to execute El-Badry cross-match: {e}") from e