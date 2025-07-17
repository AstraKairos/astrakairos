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
            df.dropna(subset=['wds_id'], inplace=True)
            df['wds_id'] = df['wds_id'].str.strip()

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
                df.drop_duplicates(subset=['wds_id'], keep='first', inplace=True)
            elif self.wds_duplicate_strategy == 'first':
                df.drop_duplicates(subset=['wds_id'], keep='first', inplace=True)
            elif self.wds_duplicate_strategy == 'last':
                df.drop_duplicates(subset=['wds_id'], keep='last', inplace=True)

            df.set_index('wds_id', inplace=True)
            return df
        except Exception as e:
            log.error(f"Failed to parse WDS summary catalog from {filepath}: {e}")
            raise IOError(f"Failed to parse WDS summary catalog from {filepath}")

    def _load_wds_measurements_catalog(self, filepath: str) -> pd.DataFrame:
        """Loads and parses the WDS Measurements Catalog."""
        # Column specifications based on WDSS format
        colspecs = [
            (0, 14),    # wds_id (WDSS identifier)
            (24, 34),   # epoch (Date of observation)
            (36, 43),   # theta (Position angle)
            (51, 52),   # rho_flag (Separation flag)
            (52, 61)    # rho (Separation)
        ]
        names = ['wds_id', 'epoch', 'theta', 'rho_flag', 'rho']
        
        try:
            df = pd.read_fwf(filepath, colspecs=colspecs, names=names, header=None, dtype=str)
            df['wds_id'] = df['wds_id'].str.strip()
            
            numeric_cols = ['epoch', 'theta', 'rho']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(subset=['wds_id', 'epoch', 'theta', 'rho'], inplace=True)

            # Handle unit conversion for rho if specified as milliarcseconds
            from ..config import MILLIARCSEC_PER_ARCSEC
            df.loc[df['rho_flag'] == 'm', 'rho'] /= MILLIARCSEC_PER_ARCSEC

            df.sort_values(by='wds_id', inplace=True)
            return df[['wds_id', 'epoch', 'theta', 'rho']]
        except Exception as e:
            log.error(f"Failed to parse WDS measurements catalog from {filepath}: {e}")
            raise IOError(f"Failed to parse WDS measurements catalog from {filepath}")

    def _load_orb6_catalog(self, filepath: str) -> pd.DataFrame:
        """Loads and parses the ORB6 catalog, handling unit conversions."""
        # Column specifications based on official ORB6 format
        # Note: Documentation uses 1-based indexing, Python uses 0-based
        colspecs = [
            (19, 29),   # wds_id (WDS designation) - T20,A10: columns 20-29
            (81, 93),   # P_str (Period with unit flag) - T82,F11.6,A1: columns 82-93  
            (105, 116), # a_str (Semi-major axis with unit flag) - T106,F9.5,A1: columns 106-116
            (125, 134), # i_str (Inclination) - T126,F8.4: columns 126-134
            (143, 154), # Omega_str (Node) - T144,F8.4,A1: columns 144-154
            (162, 177), # T_str (Time of periastron with unit flag) - T163,F12.6,A1: columns 163-177
            (187, 196), # e_str (Eccentricity) - T188,F8.6: columns 188-196
            (205, 215)  # omega_str (Longitude of periastron) - T206,F8.4,A1: columns 206-215
        ]
        names = ['wds_id', 'P_str', 'a_str', 'i_str', 'Omega_str', 
                'T_str', 'e_str', 'omega_str']
        
        try:
            df = pd.read_fwf(filepath, colspecs=colspecs, names=names, comment='R', dtype=str)
            df.dropna(subset=['wds_id'], inplace=True)
            df['wds_id'] = df['wds_id'].str.strip()

            # Import unit conversion constants
            from ..config import DAYS_PER_JULIAN_YEAR, CENTURIES_PER_YEAR, MILLIARCSEC_PER_ARCSEC

            # Parse values and unit flags, then convert to standard units
            df['P'] = pd.to_numeric(df['P_str'].str[:-1], errors='coerce')
            df.loc[df['P_str'].str.endswith('d', na=False), 'P'] /= DAYS_PER_JULIAN_YEAR
            df.loc[df['P_str'].str.endswith('c', na=False), 'P'] *= CENTURIES_PER_YEAR

            df['a'] = pd.to_numeric(df['a_str'].str[:-1], errors='coerce')
            df.loc[df['a_str'].str.endswith('m', na=False), 'a'] /= MILLIARCSEC_PER_ARCSEC
            
            df['i'] = pd.to_numeric(df['i_str'], errors='coerce')
            df['Omega'] = pd.to_numeric(df['Omega_str'].str.rstrip('*q'), errors='coerce')
            df['T'] = pd.to_numeric(df['T_str'].str[:-1], errors='coerce') # Parse time of periastron
            df['e'] = pd.to_numeric(df['e_str'], errors='coerce')
            df['omega'] = pd.to_numeric(df['omega_str'].str.rstrip('q'), errors='coerce')

            # Handle duplicates
            if self.orb6_duplicate_strategy in ('highest_grade', 'first'):
                df.drop_duplicates(subset=['wds_id'], keep='first', inplace=True)
            elif self.orb6_duplicate_strategy == 'last':
                df.drop_duplicates(subset=['wds_id'], keep='last', inplace=True)

            # Apply physical validation using config ranges
            self._apply_physical_validation(df)

            df.set_index('wds_id', inplace=True)
            return df[['P', 'a', 'i', 'Omega', 'T', 'e', 'omega']]
        except Exception as e:
            log.error(f"Failed to parse ORB6 catalog from {filepath}: {e}")
            raise IOError(f"Failed to parse ORB6 catalog from {filepath}")

    def _apply_physical_validation(self, df: pd.DataFrame) -> None:
        """
        Apply physical validation to orbital elements using efficient pandas operations.
        
        This method uses pandas boolean indexing for efficient validation of large catalogs,
        setting invalid values to NaN rather than dropping rows entirely.
        
        Args:
            df: DataFrame with orbital elements to validate
            
        Notes:
            - Uses centralized configuration for all validation ranges
            - Logs warnings for invalid values but preserves partial data
            - Much more efficient than row-by-row validation
        """
        from ..config import (
            MIN_PERIOD_YEARS, MAX_PERIOD_YEARS,
            MIN_SEMIMAJOR_AXIS_ARCSEC, MAX_SEMIMAJOR_AXIS_ARCSEC,
            MIN_ECCENTRICITY, MAX_ECCENTRICITY,
            MIN_INCLINATION_DEG, MAX_INCLINATION_DEG
        )
        
        # Period validation
        if 'P' in df.columns:
            invalid_P = (df['P'] < MIN_PERIOD_YEARS) | (df['P'] > MAX_PERIOD_YEARS)
            if invalid_P.any():
                n_invalid = invalid_P.sum()
                log.warning(f"Found {n_invalid} invalid periods outside [{MIN_PERIOD_YEARS}, {MAX_PERIOD_YEARS}] years")
                df.loc[invalid_P, 'P'] = np.nan
        
        # Semi-major axis validation
        if 'a' in df.columns:
            invalid_a = (df['a'] < MIN_SEMIMAJOR_AXIS_ARCSEC) | (df['a'] > MAX_SEMIMAJOR_AXIS_ARCSEC)
            if invalid_a.any():
                n_invalid = invalid_a.sum()
                log.warning(f"Found {n_invalid} invalid semi-major axes outside [{MIN_SEMIMAJOR_AXIS_ARCSEC}, {MAX_SEMIMAJOR_AXIS_ARCSEC}]\"")
                df.loc[invalid_a, 'a'] = np.nan
        
        # Eccentricity validation
        if 'e' in df.columns:
            invalid_e = (df['e'] < MIN_ECCENTRICITY) | (df['e'] > MAX_ECCENTRICITY)
            if invalid_e.any():
                n_invalid = invalid_e.sum()
                log.warning(f"Found {n_invalid} invalid eccentricities outside [{MIN_ECCENTRICITY}, {MAX_ECCENTRICITY}]")
                df.loc[invalid_e, 'e'] = np.nan
        
        # Inclination validation
        if 'i' in df.columns:
            invalid_i = (df['i'] < MIN_INCLINATION_DEG) | (df['i'] > MAX_INCLINATION_DEG)
            if invalid_i.any():
                n_invalid = invalid_i.sum()
                log.warning(f"Found {n_invalid} invalid inclinations outside [{MIN_INCLINATION_DEG}, {MAX_INCLINATION_DEG}]°")
                df.loc[invalid_i, 'i'] = np.nan
        
        # Angular elements validation (0-360°)
        for angle_col in ['Omega', 'omega']:
            if angle_col in df.columns:
                invalid_angle = (df[angle_col] < 0.0) | (df[angle_col] > 360.0)
                if invalid_angle.any():
                    n_invalid = invalid_angle.sum()
                    log.warning(f"Found {n_invalid} invalid {angle_col} values outside [0, 360]°")
                    df.loc[invalid_angle, angle_col] = np.nan

    # --- Implementation of DataSource Abstract Methods ---

    async def get_wds_summary(self, wds_id: str) -> Optional[WdsSummary]:
        normalized_id = wds_id.strip()
        try:
            if normalized_id in self.wds_df.index:
                row_series = self.wds_df.loc[normalized_id]
                summary_data = row_series.replace({np.nan: None}).to_dict()
                # Include the WDS identifier
                summary_data['wds_id'] = normalized_id
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
        star_measures_df = self.wds_measures_df.loc[self.wds_measures_df['wds_id'] == normalized_id]
        
        return Table.from_pandas(star_measures_df) if not star_measures_df.empty else None

    async def get_orbital_elements(self, wds_id: str) -> Optional[OrbitalElements]:
        normalized_id = wds_id.strip()
        try:
            if normalized_id in self.orb6_df.index:
                row_series = self.orb6_df.loc[normalized_id]
                orbital_data = row_series.replace({np.nan: None}).to_dict()
                # Include the WDS identifier
                orbital_data['wds_id'] = normalized_id
                # Filter to include only valid OrbitalElements fields
                filtered_data = {k: v for k, v in orbital_data.items() 
                            if k in OrbitalElements.__annotations__ and v is not None and pd.notna(v)}
                return OrbitalElements(**filtered_data) if filtered_data else None
        except (KeyError, TypeError):
            pass
        return None

    async def validate_physicality(self, system_data: WdsSummary) -> Optional[PhysicalityAssessment]:
        """
        Local source cannot perform physicality validation.
        Returns unknown status with proper PhysicalityAssessment structure.
        """
        if not system_data:
            return None
            
        from .source import PhysicalityLabel, ValidationMethod
        from datetime import datetime
        
        return {
            'label': PhysicalityLabel.UNKNOWN,
            'confidence': 0.0,
            'p_value': None,
            'method': ValidationMethod.INSUFFICIENT_DATA,
            'parallax_consistency': None,
            'proper_motion_consistency': None,
            'gaia_source_id_primary': None,
            'gaia_source_id_secondary': None,
            'validation_date': datetime.now().isoformat(),
            'search_radius_arcsec': 0.0,
            'significance_thresholds': {},
            'retry_attempts': 0
        }