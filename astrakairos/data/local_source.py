import logging
import os
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from astropy.table import Table

from .source import (DataSource, OrbitalElements, PhysicalityAssessment,
                     WdsSummary)

log = logging.getLogger(__name__)

# Type Aliases for configuration clarity
WdsDuplicateStrategy = Literal['first', 'last', 'most_observations']
Orb6DuplicateStrategy = Literal['first', 'last', 'highest_grade']


class LocalFileDataSource(DataSource):
    """
    Implements DataSource for local WDS, ORB6, and WDS Measurements catalog files.
    This class uses robust, fixed-width file parsing for accuracy and is designed
    for high-performance local analysis.
    """

    def __init__(self,
                 wds_filepath: str,
                 orb6_filepath: str,
                 wds_measures_filepath: Optional[str] = None,
                 wds_duplicate_strategy: WdsDuplicateStrategy = 'most_observations',
                 orb6_duplicate_strategy: Orb6DuplicateStrategy = 'first'):
        if not os.path.exists(wds_filepath):
            raise FileNotFoundError(f"WDS summary file not found: {wds_filepath}")
        if not os.path.exists(orb6_filepath):
            raise FileNotFoundError(f"ORB6 file not found: {orb6_filepath}")

        self.wds_duplicate_strategy = wds_duplicate_strategy
        self.orb6_duplicate_strategy = orb6_duplicate_strategy

        log.info("Loading local summary and orbit catalogs...")
        self.wds_df = self._load_wds_summary_catalog(wds_filepath)
        self.orb6_df = self._load_orb6_catalog(orb6_filepath)
        log.info("Summary and orbit catalogs loaded successfully.")

        self.wds_measures_df = None
        if wds_measures_filepath:
            if os.path.exists(wds_measures_filepath):
                log.info("Loading WDS Measurements Catalog (this may take time)...")
                self.wds_measures_df = self._load_wds_measurements_catalog(wds_measures_filepath)
            else:
                log.warning(f"WDS measurements file not found: {wds_measures_filepath}. "
                            "Full measurement analysis will be unavailable.")

    def _load_wds_summary_catalog(self, filepath: str) -> pd.DataFrame:
        """Loads and parses the WDS summary catalog (wdsweb_summ.txt)."""
        # Column specifications based on WDS BIBLE format
        colspecs = [
            (0, 10),    # wds_name
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
            'wds_name', 'discoverer', 'components', 'date_first', 'date_last', 'obs',
            'pa_first', 'pa_last', 'sep_first', 'sep_last', 'mag_pri', 'mag_sec',
            'spec_type', 'precise_coords_str'
        ]
        
        try:
            df = pd.read_fwf(filepath, colspecs=colspecs, names=names, dtype=str)
            df.dropna(subset=['wds_name'], inplace=True)
            df['wds_name'] = df['wds_name'].str.strip()

            numeric_cols = ['date_first', 'date_last', 'obs', 'pa_first', 'pa_last',
                            'sep_first', 'sep_last', 'mag_pri', 'mag_sec']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Parse precise coordinates in "HHMMSS.SS+DDMMSS.S" format
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

            # Handle duplicate entries
            if self.wds_duplicate_strategy == 'most_observations':
                df.sort_values('obs', ascending=False, inplace=True)
                df.drop_duplicates(subset=['wds_name'], keep='first', inplace=True)
            elif self.wds_duplicate_strategy == 'first':
                df.drop_duplicates(subset=['wds_name'], keep='first', inplace=True)
            elif self.wds_duplicate_strategy == 'last':
                df.drop_duplicates(subset=['wds_name'], keep='last', inplace=True)

            df.set_index('wds_name', inplace=True)
            return df
        except Exception as e:
            log.error(f"Failed to parse WDS summary catalog from {filepath}: {e}")
            raise IOError(f"Failed to parse WDS summary catalog from {filepath}")

    def _load_wds_measurements_catalog(self, filepath: str) -> pd.DataFrame:
        """Loads and parses the WDS Measurements Catalog."""
        # Column specifications based on WDSS format
        colspecs = [
            (0, 14),    # wds_name (WDSS identifier)
            (24, 34),   # epoch (Date of observation)
            (36, 43),   # theta (Position angle)
            (51, 52),   # rho_flag (Separation flag)
            (52, 61)    # rho (Separation)
        ]
        names = ['wds_name', 'epoch', 'theta', 'rho_flag', 'rho']
        
        try:
            df = pd.read_fwf(filepath, colspecs=colspecs, names=names, header=None, dtype=str)
            df['wds_name'] = df['wds_name'].str.strip()
            
            numeric_cols = ['epoch', 'theta', 'rho']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(subset=['wds_name', 'epoch', 'theta', 'rho'], inplace=True)

            # Handle unit conversion for rho if specified as milliarcseconds
            df.loc[df['rho_flag'] == 'm', 'rho'] /= 1000.0

            df.sort_values(by='wds_name', inplace=True)
            return df[['wds_name', 'epoch', 'theta', 'rho']]
        except Exception as e:
            log.error(f"Failed to parse WDS measurements catalog from {filepath}: {e}")
            raise IOError(f"Failed to parse WDS measurements catalog from {filepath}")

    def _load_orb6_catalog(self, filepath: str) -> pd.DataFrame:
        """Loads and parses the ORB6 catalog, handling unit conversions."""
        # Column specifications based on official ORB6 format
        # Note: Documentation uses 1-based indexing, Python uses 0-based
        colspecs = [
            (19, 29),   # wds_name (WDS designation) - T20,A10: columns 20-29
            (81, 93),   # P_str (Period with unit flag) - T82,F11.6,A1: columns 82-93  
            (105, 116), # a_str (Semi-major axis with unit flag) - T106,F9.5,A1: columns 106-116
            (125, 134), # i_str (Inclination) - T126,F8.4: columns 126-134
            (143, 154), # Omega_str (Node) - T144,F8.4,A1: columns 144-154
            (162, 177), # T_str (Time of periastron with unit flag) - T163,F12.6,A1: columns 163-177
            (187, 196), # e_str (Eccentricity) - T188,F8.6: columns 188-196
            (205, 215)  # omega_str (Longitude of periastron) - T206,F8.4,A1: columns 206-215
        ]
        names = ['wds_name', 'P_str', 'a_str', 'i_str', 'Omega_str', 
                'T_str', 'e_str', 'omega_str']
        
        try:
            df = pd.read_fwf(filepath, colspecs=colspecs, names=names, comment='R', dtype=str)
            df.dropna(subset=['wds_name'], inplace=True)
            df['wds_name'] = df['wds_name'].str.strip()

            # Parse values and unit flags, then convert to standard units
            df['P'] = pd.to_numeric(df['P_str'].str[:-1], errors='coerce')
            df.loc[df['P_str'].str.endswith('d', na=False), 'P'] /= 365.25 # days to years
            df.loc[df['P_str'].str.endswith('c', na=False), 'P'] *= 100.0   # centuries to years

            df['a'] = pd.to_numeric(df['a_str'].str[:-1], errors='coerce')
            df.loc[df['a_str'].str.endswith('m', na=False), 'a'] /= 1000.0 # mas to arcsec
            
            df['i'] = pd.to_numeric(df['i_str'], errors='coerce')
            df['Omega'] = pd.to_numeric(df['Omega_str'].str.rstrip('*q'), errors='coerce')
            df['T'] = pd.to_numeric(df['T_str'].str[:-1], errors='coerce') # Parse time of periastron
            df['e'] = pd.to_numeric(df['e_str'], errors='coerce')
            df['omega'] = pd.to_numeric(df['omega_str'].str.rstrip('q'), errors='coerce')

            # Handle duplicates
            if self.orb6_duplicate_strategy in ('highest_grade', 'first'):
                df.drop_duplicates(subset=['wds_name'], keep='first', inplace=True)
            elif self.orb6_duplicate_strategy == 'last':
                df.drop_duplicates(subset=['wds_name'], keep='last', inplace=True)

            df.set_index('wds_name', inplace=True)
            return df[['P', 'a', 'i', 'Omega', 'T', 'e', 'omega']]
        except Exception as e:
            log.error(f"Failed to parse ORB6 catalog from {filepath}: {e}")
            raise IOError(f"Failed to parse ORB6 catalog from {filepath}")

    # --- Implementation of DataSource Abstract Methods ---

    async def get_wds_summary(self, wds_id: str) -> Optional[WdsSummary]:
        normalized_id = wds_id.strip()
        try:
            if normalized_id in self.wds_df.index:
                row_series = self.wds_df.loc[normalized_id]
                summary_data = row_series.replace({np.nan: None}).to_dict()
                # Restore the wds_name field
                summary_data['wds_name'] = normalized_id
                # Filter to include only valid WdsSummary fields
                filtered_data = {k: v for k, v in summary_data.items() 
                            if k in WdsSummary.__annotations__ and v is not None and pd.notna(v)}
                return WdsSummary(**filtered_data) if filtered_data else None
        except (KeyError, TypeError):
            pass
        return None

    async def get_all_measurements(self, wds_id: str) -> Optional[Table]:
        if self.wds_measures_df is None:
            return None
            
        normalized_id = wds_id.strip()
        star_measures_df = self.wds_measures_df.loc[self.wds_measures_df['wds_name'] == normalized_id]
        
        return Table.from_pandas(star_measures_df) if not star_measures_df.empty else None

    async def get_orbital_elements(self, wds_id: str) -> Optional[OrbitalElements]:
        normalized_id = wds_id.strip()
        try:
            if normalized_id in self.orb6_df.index:
                row_series = self.orb6_df.loc[normalized_id]
                orbital_data = row_series.replace({np.nan: None}).to_dict()
                # Include the WDS identifier
                orbital_data['wds_name'] = normalized_id
                # Filter to include only valid OrbitalElements fields
                filtered_data = {k: v for k, v in orbital_data.items() 
                            if k in OrbitalElements.__annotations__ and v is not None and pd.notna(v)}
                return OrbitalElements(**filtered_data) if filtered_data else None
        except (KeyError, TypeError):
            pass
        return None

    async def validate_physicality(self, system_data: WdsSummary) -> Optional[PhysicalityAssessment]:
        if not system_data:
            return None
        return {'label': 'Unknown', 'p_value': None, 'test_used': 'Local Source (Not Performed)'}