"""
Orchestrator for the main conversion pipeline.

This module coordinates the entire conversion process from loading
files to creating the final SQLite database.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    # Try relative imports first (when used as a package)
    from . import parsers, database
    from .crossmatching import perform_el_badry_crossmatch
    from .summary import generate_summary_table
except ImportError:
    # Fall back to absolute imports (when run directly)
    import parsers
    import database
    from crossmatching import perform_el_badry_crossmatch
    from summary import generate_summary_table

from astrakairos.exceptions import ConversionProcessError

log = logging.getLogger(__name__)


class ConversionPipeline:
    """
    Pipeline principal para convertir catálogos astronómicos a SQLite.
    
    Esta clase maneja todo el flujo de trabajo de conversión:
    1. Carga y procesamiento de múltiples archivos WDSS
    2. Cross-matching opcional con catálogo El-Badry
    3. Procesamiento del catálogo ORB6
    4. Generación de tablas resumen
    5. Creación de base de datos SQLite optimizada
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the conversion pipeline.
        
        Args:
            config: Dictionary with configuration parameters:
                - wdss_files: List of WDSS catalog file paths
                - orb6_file: Path to ORB6 catalog file
                - el_badry_file: Optional path to El-Badry catalog
                - output_path: Path for output SQLite database
        """
        self.config = config
        self.combined_components = pd.DataFrame()
        self.combined_measurements = pd.DataFrame()
        self.combined_correspondence = pd.DataFrame()
        self.df_orb6 = pd.DataFrame()
        self.df_el_badry_matches: Optional[pd.DataFrame] = None

    def run(self) -> None:
        """
        Execute the complete conversion pipeline.
        
        Raises:
            ConversionProcessError: If any step of the pipeline fails
        """
        try:
            log.info("Iniciando el pipeline de conversión de catálogos.")
            
            # 1. Cargar y procesar datos WDSS
            self._load_wdss_data()
            
            # 2. Procesar catálogo ORB6
            self._load_orb6_data()
            
            # 3. Cross-matching opcional con El-Badry
            self._perform_el_badry_crossmatch()
            
            # 4. Generar tabla resumen
            df_summary = self._generate_summary_table()
            
            # 5. Crear base de datos SQLite
            self._create_database(df_summary)
            
            log.info("Pipeline de conversión finalizado con éxito.")
            
        except Exception as e:
            raise ConversionProcessError(f"Pipeline failed: {e}") from e

    def _load_wdss_data(self) -> None:
        """Load and combine multiple WDSS catalog files."""
        log.info(f"Parsing {len(self.config['wdss_files'])} WDSS catalog files...")
        all_components = []
        all_measurements = []
        all_correspondence = []
        
        for wdss_file in self.config['wdss_files']:
            log.info(f"Processing {wdss_file}...")
            df_components, df_measurements, df_correspondence = parsers.parse_wdss_master_catalog(wdss_file)
            all_components.append(df_components)
            all_measurements.append(df_measurements)
            all_correspondence.append(df_correspondence)
        
        # Combine all data
        self.combined_components = pd.concat(all_components, ignore_index=True) if all_components else pd.DataFrame()
        self.combined_measurements = pd.concat(all_measurements, ignore_index=True) if all_measurements else pd.DataFrame()
        self.combined_correspondence = pd.concat(all_correspondence, ignore_index=True) if all_correspondence else pd.DataFrame()
        
        # Remove duplicates across files
        if not self.combined_components.empty:
            self.combined_components = self.combined_components.drop_duplicates(subset=['wdss_id', 'component'], keep='first')
        if not self.combined_measurements.empty:
            self.combined_measurements = self.combined_measurements.drop_duplicates(subset=['wdss_id', 'pair', 'epoch'], keep='first')
        if not self.combined_correspondence.empty:
            self.combined_correspondence = self.combined_correspondence.drop_duplicates(subset=['wdss_id'], keep='first')
        
        log.info(f"Combined: {len(self.combined_components)} components, "
                f"{len(self.combined_measurements)} measurements, "
                f"{len(self.combined_correspondence)} correspondences")

    def _load_orb6_data(self) -> None:
        """Load and process ORB6 orbital elements catalog."""
        log.info("Parsing ORB6 catalog...")
        self.df_orb6 = parsers.parse_orb6_catalog(self.config['orb6_file'])

    def _perform_el_badry_crossmatch(self) -> None:
        """Perform cross-matching with El-Badry catalog if provided."""
        if self.config.get('el_badry_file'):
            log.info("Cross-matching with El-Badry catalog using improved pair-wise matching...")
            df_el_badry = parsers.parse_el_badry_catalog(self.config['el_badry_file'])
            self.df_el_badry_matches = perform_el_badry_crossmatch(self.combined_components, df_el_badry)
        else:
            log.info("No El-Badry catalog provided, skipping cross-match")
            self.df_el_badry_matches = None

    def _generate_summary_table(self) -> pd.DataFrame:
        """Generate the summary table from all processed data."""
        log.info("Generating summary table...")
        return generate_summary_table(
            self.combined_components, 
            self.combined_measurements, 
            self.combined_correspondence, 
            self.df_el_badry_matches
        )

    def _create_database(self, df_summary: pd.DataFrame) -> None:
        """Create the final SQLite database."""
        log.info("Creating SQLite database...")
        database.create_sqlite_database(
            df_summary, 
            self.df_orb6, 
            self.combined_measurements, 
            self.config['output_path']
        )

    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the processed data.
        
        Returns:
            Dictionary with counts of various data elements
        """
        return {
            'components': len(self.combined_components),
            'measurements': len(self.combined_measurements),
            'correspondences': len(self.combined_correspondence),
            'orbital_elements': len(self.df_orb6),
            'el_badry_matches': len(self.df_el_badry_matches) if self.df_el_badry_matches is not None else 0
        }
