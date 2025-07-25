"""
Functions for cross-matching between astronomical catalogs.

This module handles the cross-matching between WDSS and El-Badry catalogs
to identify confirmed physical binary systems.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

from astrakairos.config import GAIA_ID_PATTERN
from astrakairos.exceptions import ElBadryCrossmatchError

log = logging.getLogger(__name__)


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
