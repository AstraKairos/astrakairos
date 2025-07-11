import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Literal
import os
from .source import DataSource

# Define Type Aliases for clarity and restrict options
# For WDS:
WdsDuplicateStrategy = Literal['first', 'last', 'most_observations']
# For ORB6:
Orb6DuplicateStrategy = Literal['first', 'last', 'highest_grade'] # 'highest_grade' implicitly means 'first' if sorted by grade

class LocalFileDataSource(DataSource):
    """
    Implements DataSource for local WDS and ORB6 catalog files.
    This class uses robust, fixed-width file parsing for accuracy and is designed
    for high-performance local analysis.
    """
    
    def __init__(self, 
                 wds_filepath: str, 
                 orb6_filepath: str,
                 wds_duplicate_strategy: WdsDuplicateStrategy = 'most_observations',
                 orb6_duplicate_strategy: Orb6DuplicateStrategy = 'first'):
        """
        Initializes the data source by loading WDS and ORB6 catalogs from local files.

        Args:
            wds_filepath: Path to the Washington Double Star (WDS) catalog file (wdsweb_summ.txt).
            orb6_filepath: Path to the Sixth Catalog of Orbits (ORB6) file (orb6.txt).
            wds_duplicate_strategy: Strategy to handle duplicate WDS IDs in the WDS catalog.
                                    Options: 'first', 'last', 'most_observations'.
                                    'most_observations' (default) keeps the entry with the highest 'obs' count.
            orb6_duplicate_strategy: Strategy to handle duplicate WDS IDs in the ORB6 catalog.
                                     Options: 'first', 'last', 'highest_grade'.
                                     'first' (default) keeps the first entry encountered (usually highest grade).
                                     'highest_grade' is conceptually equivalent to 'first' in standard ORB6 files.
        """
        if not os.path.exists(wds_filepath):
            raise FileNotFoundError(f"WDS file not found: {wds_filepath}")
        if not os.path.exists(orb6_filepath):
            raise FileNotFoundError(f"ORB6 file not found: {orb6_filepath}")
            
        # Store strategies
        self.wds_duplicate_strategy = wds_duplicate_strategy
        self.orb6_duplicate_strategy = orb6_duplicate_strategy

        print("Loading local catalogs...")
        self.wds_df = self._load_wds_catalog(wds_filepath)
        self.orb6_df = self._load_orb6_catalog(orb6_filepath)
        print("Local catalogs loaded and indexed successfully.")

    def _parse_ra_dec_from_str(self, ra_str: pd.Series, dec_str: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Parses RA/Dec strings (hms/dms format) into decimal degrees."""
        # RA: HHMMSS.s -> hours -> degrees
        ra_h = pd.to_numeric(ra_str.str[0:2], errors='coerce')
        ra_m = pd.to_numeric(ra_str.str[2:4], errors='coerce')
        ra_s = pd.to_numeric(ra_str.str[4:], errors='coerce')
        ra_deg = (ra_h + ra_m / 60 + ra_s / 3600) * 15

        # Dec: sDDMMSS.s -> degrees
        # Using vectorized operations for performance
        dec_sign = np.where(dec_str.str.startswith('-'), -1.0, 1.0)
        dec_d = pd.to_numeric(dec_str.str[1:3], errors='coerce')
        dec_m = pd.to_numeric(dec_str.str[3:5], errors='coerce')
        dec_s = pd.to_numeric(dec_str.str[5:], errors='coerce')
        dec_deg = dec_sign * (dec_d + dec_m / 60 + dec_s / 3600)
        
        return ra_deg, dec_deg

    def _load_wds_catalog(self, filepath: str) -> pd.DataFrame:
        """
        Loads and parses the WDS catalog using its official fixed-width format.
        Applies the configured duplicate resolution strategy.
        """
        colspecs = [
            (0, 10), (10, 17), (17, 22), (23, 27), (28, 32), (33, 37), (38, 43), 
            (44, 49), (50, 55), (56, 61), (62, 67), (68, 73), (74, 83), (84, 93), 
            (95, 102), (102, 110)
        ]
        names = [
            'wds_id', 'discoverer', 'components', 'date_first', 'date_last', 'obs',
            'pa_first', 'pa_last', 'sep_first', 'sep_last', 'mag_pri', 'mag_sec',
            'spectral_type', 'proper_motion', 'ra_hms_str', 'dec_dms_str'
        ]
        
        try:
            df = pd.read_fwf(filepath, colspecs=colspecs, names=names, dtype=str)
            df.dropna(subset=['wds_id'], inplace=True)
            
            # Normalize WDS ID to its 10-char canonical form
            df['wds_id'] = df['wds_id'].str.strip().str[:10]
            
            numeric_cols = ['date_first', 'date_last', 'obs', 'pa_first', 'pa_last', 
                            'sep_first', 'sep_last', 'mag_pri', 'mag_sec']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['ra_deg'], df['dec_deg'] = self._parse_ra_dec_from_str(df['ra_hms_str'], df['dec_dms_str'])
            
            # --- Apply duplicate resolution strategy for WDS ---
            if self.wds_duplicate_strategy == 'most_observations':
                # Sort by 'obs' in descending order to keep the entry with most observations
                df.sort_values('obs', ascending=False, inplace=True)
                df.drop_duplicates(subset=['wds_id'], keep='first', inplace=True)
            elif self.wds_duplicate_strategy == 'first':
                df.drop_duplicates(subset=['wds_id'], keep='first', inplace=True)
            elif self.wds_duplicate_strategy == 'last':
                df.drop_duplicates(subset=['wds_id'], keep='last', inplace=True)
            else:
                raise ValueError(f"Invalid WDS duplicate strategy: {self.wds_duplicate_strategy}")

            df.set_index('wds_id', inplace=True)
            print(f"Loaded and parsed {len(df)} unique entries from WDS catalog.")
            return df
        except Exception as e:
            raise IOError(f"Failed to parse WDS catalog from {filepath}: {e}")

    def _load_orb6_catalog(self, filepath: str) -> pd.DataFrame:
        """
        Loads and parses the ORB6 catalog using its official fixed-width format.
        Applies the configured duplicate resolution strategy.
        """
        colspecs = [
            (19, 29),   # WDS ID (ej. 00003-4417)
            (60, 69),   # Period (P)
            (76, 84),   # Semi-major axis (a)
            (89, 97),   # Eccentricity (e)
            (102, 109), # Inclination (i)
            (114, 122), # Long. of Asc. Node (Omega)
            (127, 135), # Arg. of Periastron (omega)
            (140, 151)  # Epoch of Periastron (T)
        ]
        names = ['wds_id', 'P', 'a', 'e', 'i', 'Omega', 'omega', 'T']
        
        try:
            df = pd.read_fwf(filepath, colspecs=colspecs, names=names, comment='R', dtype=str)
            df.dropna(subset=['wds_id'], inplace=True)

            # Normalize WDS ID to its 10-char canonical form
            df['wds_id'] = df['wds_id'].str.strip().str[:10]
            
            numeric_cols = ['P', 'a', 'e', 'i', 'Omega', 'omega', 'T']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # --- Apply duplicate resolution strategy for ORB6 ---
            if self.orb6_duplicate_strategy == 'highest_grade' or self.orb6_duplicate_strategy == 'first':
                # 'highest_grade' implicitly means 'first' as per ORB6 file convention
                df.drop_duplicates(subset=['wds_id'], keep='first', inplace=True)
            elif self.orb6_duplicate_strategy == 'last':
                df.drop_duplicates(subset=['wds_id'], keep='last', inplace=True)
            else:
                raise ValueError(f"Invalid ORB6 duplicate strategy: {self.orb6_duplicate_strategy}")
            
            df.set_index('wds_id', inplace=True)
            print(f"Loaded and parsed {len(df)} unique orbits from ORB6 catalog.")
            return df
        except Exception as e:
            raise IOError(f"Failed to parse ORB6 catalog from {filepath}: {e}")

    async def get_wds_data(self, wds_id: str) -> Dict[str, Any]:
        """Gets WDS data for a star using its normalized 10-character ID."""
        normalized_id = wds_id.strip()[:10]
        try:
            if normalized_id in self.wds_df.index:
                data_series = self.wds_df.loc[normalized_id]
                # Replace numpy NaNs with Python's None for cleaner downstream use
                return data_series.replace({np.nan: None}).to_dict()
        except KeyError:
            # This can happen if an ID exists but has issues; fail silently
            pass
        return {}
    
    async def get_orbital_elements(self, wds_id: str) -> Dict[str, Any]:
        """Gets ORB6 data for a star using its normalized 10-character ID."""
        normalized_id = wds_id.strip()[:10]
        try:
            if normalized_id in self.orb6_df.index:
                data_series = self.orb6_df.loc[normalized_id]
                return data_series.replace({np.nan: None}).to_dict()
        except KeyError:
            pass
        return {}

    # Mandatory method for DataSource interface
    async def validate_physicality(self, wds_data: Dict[str, Any]) -> str:
        """
        Local source cannot validate with Gaia. Returns 'Unknown'.
        The wds_data parameter is included to match the DataSource interface.
        """
        if not wds_data:
            return "Unknown"
        # In a future implementation, this could check a local copy of a Gaia-validated catalog
        return "Unknown"