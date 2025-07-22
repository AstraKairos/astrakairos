"""
Local data source implementation using SQLite database for catalog access.

This implementation provides access to WDS catalogs using a local SQLite database 
with proper indexing for efficient querying.
"""

import logging
import sqlite3
from typing import Optional, Dict, Any

from astropy.table import Table

from .source import (DataSource, OrbitalElements, PhysicalityAssessment,
                     WdsSummary, PhysicalityLabel, ValidationMethod)

log = logging.getLogger(__name__)


class LocalDataSource(DataSource):
    """
    Local catalog data source using SQLite database.
    
    This class provides access to local WDS catalogs using a SQLite database
    with proper indexing. Suitable for large-scale analysis of millions 
    of measurements.
    
    Note: Requires catalogs to be converted to SQLite format first using
    the convert_catalogs_to_sqlite.py script.
    """

    def __init__(self, database_path: str):
        """Initialize local data source with SQLite database.
        
        Args:
            database_path: Path to the SQLite database file containing
                          converted WDS catalogs.
                          
        Raises:
            FileNotFoundError: If the database file doesn't exist
            sqlite3.Error: If database structure is invalid
        """
        self.database_path = database_path
        self.conn = None
        
        try:
            # Open database in read-only mode for safety and performance
            self.conn = sqlite3.connect(f'file:{database_path}?mode=ro', uri=True)
            self.conn.row_factory = sqlite3.Row
            
            # Performance optimizations for read-only access
            self._configure_performance_settings()
            
            # Verify database structure
            self._verify_database_structure()
            
            log.info(f"Connected to local catalog database: {database_path}")
            
        except sqlite3.Error as e:
            log.error(f"Failed to connect to catalog database {database_path}: {e}")
            raise
        except FileNotFoundError as e:
            log.error(f"Catalog database not found: {database_path}")
            raise

    def _configure_performance_settings(self) -> None:
        """Configure SQLite for optimal read performance."""
        try:
            # Disable journal and synchronous writes (safe for read-only)
            self.conn.execute('PRAGMA journal_mode = OFF')
            self.conn.execute('PRAGMA synchronous = OFF')
            
            # Use memory for temporary operations
            self.conn.execute('PRAGMA temp_store = MEMORY')
            
            # Calculate optimal cache size (fallback to conservative 100MB if psutil unavailable)
            try:
                import psutil
                available_ram_kb = psutil.virtual_memory().available // 1024
                cache_size_kb = min(available_ram_kb // 10, 500 * 1024)  # 10% of RAM, max 500MB
            except ImportError:
                cache_size_kb = 100 * 1024  # Conservative 100MB fallback
                log.debug("psutil not available, using conservative cache size")
            
            self.conn.execute(f'PRAGMA cache_size = -{cache_size_kb}')
            
            # Enable memory-mapped I/O (can be very fast for large databases)
            mmap_size = min(1024 * 1024 * 1024, cache_size_kb * 1024 * 2)  # Up to 1GB or 2x cache
            self.conn.execute(f'PRAGMA mmap_size = {mmap_size}')
            
            # Optimize query planner
            self.conn.execute('PRAGMA optimize')
            
            log.debug(f"SQLite performance configured: cache={cache_size_kb}KB, mmap={mmap_size//1024//1024}MB")
            
        except Exception as e:
            log.warning(f"Could not apply all performance optimizations: {e}")
            # Continue anyway - these are optimizations, not requirements

    def _verify_database_structure(self) -> None:
        """Verify that database has required tables and structure."""
        cursor = self.conn.cursor()
        
        # Check for required tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        # Check for either WDSS format or traditional WDS format
        has_wdss = 'wdss_summary' in tables
        has_wds = 'wds_summary' in tables
        has_orbital = 'orbital_elements' in tables
        
        if not (has_wdss or has_wds):
            raise sqlite3.Error(f"Database missing summary table: need either 'wdss_summary' or 'wds_summary'")
        
        if not has_orbital:
            raise sqlite3.Error(f"Database missing required table: 'orbital_elements'")
        
        # Set the table name to use for queries
        self.summary_table = 'wdss_summary' if has_wdss else 'wds_summary'
        
        # Check if measurements table exists (optional)
        self.has_measurements = 'measurements' in tables
        
        log.info(f"Database verified. Using {self.summary_table}. Has measurements: {self.has_measurements}")

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            log.debug("Local database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.close()

    def __del__(self):
        """Ensure connection is closed on cleanup."""
        try:
            self.close()
        except Exception as e:
            log.warning(f"Error during local database cleanup: {e}")

    async def get_wds_summary(self, wds_id: str) -> Optional[WdsSummary]:
        """Get WDS summary data for a system.
        
        Args:
            wds_id: WDS identifier
            
        Returns:
            WdsSummary object or None if not found
        """
        normalized_id = wds_id.strip()
        
        try:
            cursor = self.conn.execute(
                f"""SELECT wds_id, discoverer_designation as discoverer, '' as components, 
                          date_first, date_last, n_obs as n_observations, 
                          pa_first, pa_last, sep_first, sep_last, 
                          vmag as mag_pri, kmag as mag_sec, spectral_type as spec_type, ra_deg, dec_deg,
                          wdss_id, discoverer_designation,
                          pa_first_error, pa_last_error,
                          sep_first_error, sep_last_error
                   FROM {self.summary_table} WHERE wds_id = ?""",
                (normalized_id,)
            )
            
            row = cursor.fetchone()
            if row:
                # Convert row to dict and filter None values for required fields
                # Keep None values for error fields as they indicate missing uncertainty data
                data = dict(row)
                
                # Filter out None values only for non-error fields
                filtered_data = {}
                for k, v in data.items():
                    if k.endswith('_error'):
                        # Keep error fields even if None - this indicates missing uncertainty
                        filtered_data[k] = v
                    elif v is not None:
                        # Filter None values for regular fields
                        filtered_data[k] = v
                
                # Validate against WdsSummary schema
                valid_fields = {k: v for k, v in filtered_data.items() 
                               if k in WdsSummary.__annotations__}
                
                if valid_fields:
                    return WdsSummary(**valid_fields)
            
            return None
            
        except sqlite3.Error as e:
            log.error(f"Database error retrieving WDS summary for {normalized_id}: {e}")
            return None
        except Exception as e:
            log.error(f"Error creating WdsSummary for {normalized_id}: {e}")
            return None

    async def get_all_measurements(self, wds_id: str) -> Optional[Table]:
        """Get all measurements for a system with complete uncertainty information.
        
        Args:
            wds_id: WDS identifier
            
        Returns:
            Astropy Table with measurements including error columns (when available) or None if not found
        """
        if not self.has_measurements:
            return None
            
        normalized_id = wds_id.strip()
        
        try:
            # First get the wdss_id for this wds_id
            wdss_cursor = self.conn.execute(
                f"SELECT wdss_id FROM {self.summary_table} WHERE wds_id = ?",
                (normalized_id,)
            )
            wdss_row = wdss_cursor.fetchone()
            
            if not wdss_row:
                return None
                
            wdss_id = wdss_row['wdss_id']
            
            # Check which columns exist in the measurements table to be backwards compatible
            schema_cursor = self.conn.execute("PRAGMA table_info(measurements)")
            available_columns = [row[1] for row in schema_cursor.fetchall()]
            
            # Build query based on available columns
            base_columns = ['wdss_id', 'epoch', 'theta', 'rho']
            optional_columns = ['theta_error', 'rho_error', 'error_source', 'technique', 'reference']
            
            # Only include columns that exist in the database
            query_columns = base_columns + [col for col in optional_columns if col in available_columns]
            column_list = ', '.join(query_columns)
            
            cursor = self.conn.execute(
                f"SELECT {column_list} FROM measurements WHERE wdss_id = ? ORDER BY epoch",
                (wdss_id,)
            )
            
            rows = cursor.fetchall()
            if rows:
                # Create astropy Table from the available data
                table_data = {}
                
                # Add normalized wds_id for consistency
                table_data['wds_id'] = [normalized_id] * len(rows)
                
                # Add all available columns
                for col in query_columns:
                    table_data[col] = [row[col] for row in rows]
                
                return Table(table_data)
            
            return None
            
        except sqlite3.Error as e:
            log.error(f"Database error retrieving measurements for {normalized_id}: {e}")
            return None

    async def get_orbital_elements(self, wds_id: str) -> Optional[OrbitalElements]:
        """Get orbital elements for a system.
        
        Args:
            wds_id: WDS identifier
            
        Returns:
            OrbitalElements object or None if not found
        """
        normalized_id = wds_id.strip()
        
        try:
            cursor = self.conn.execute(
                """SELECT wds_id, P, a, i, Omega, T, e, omega_arg as omega,
                          e_P, e_a, e_i, e_Omega, e_T, e_e, e_omega_arg,
                          grade
                   FROM orbital_elements WHERE wds_id = ?""",
                (normalized_id,)
            )
            
            row = cursor.fetchone()
            if row:
                # Convert sqlite3 row to dict manually with explicit column mapping
                columns = [
                    'wds_id', 'P', 'a', 'i', 'Omega', 'T', 'e', 'omega',
                    'e_P', 'e_a', 'e_i', 'e_Omega', 'e_T', 'e_e', 'e_omega_arg',
                    'grade'
                ]
                data = {col: row[i] for i, col in enumerate(columns)}
                
                # Filter out None values only for non-error fields
                filtered_data = {}
                for k, v in data.items():
                    if k.startswith('e_') and k != 'e':  # Error fields (but not eccentricity 'e')
                        # Keep error fields even if None - indicates missing uncertainty
                        filtered_data[k] = v
                    elif v is not None:
                        # Filter None values for regular fields
                        filtered_data[k] = v
                
                # Return filtered dictionary (TypedDict is just a dict at runtime)
                return filtered_data
            
            return None
            
        except sqlite3.Error as e:
            log.error(f"Database error retrieving orbital elements for {normalized_id}: {e}")
            return None
        except Exception as e:
            log.error(f"Error creating OrbitalElements for {normalized_id}: {e}")
            return None

    async def validate_physicality(self, system_data: WdsSummary) -> Optional[PhysicalityAssessment]:
        """
        Local source cannot perform physicality validation.
        Returns unknown status with proper PhysicalityAssessment structure.
        """
        if not system_data:
            return None
            
        from datetime import datetime
        
        return PhysicalityAssessment(
            label=PhysicalityLabel.UNKNOWN,
            confidence=0.0,
            p_value=None,
            method=ValidationMethod.INSUFFICIENT_DATA,
            parallax_consistency=None,
            proper_motion_consistency=None,
            gaia_source_id_primary=None,
            gaia_source_id_secondary=None,
            validation_date=datetime.now().isoformat(),
            search_radius_arcsec=0.0,
            significance_thresholds={},
            retry_attempts=0
        )

    def get_catalog_statistics(self) -> Optional[Dict[str, int]]:
        """Get statistics about the loaded catalogs.
        
        Returns:
            Dictionary with table sizes and other statistics
        """
        if not self.conn:
            log.error("Database connection is closed")
            return None
            
        try:
            stats = {}
            
            # Summary table count
            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {self.summary_table}")
            stats[f'{self.summary_table}_count'] = cursor.fetchone()[0]
            
            # Orbital elements count
            cursor = self.conn.execute("SELECT COUNT(*) FROM orbital_elements")
            stats['orbital_elements_count'] = cursor.fetchone()[0]
            
            # Measurements count (if available)
            if self.has_measurements:
                cursor = self.conn.execute("SELECT COUNT(*) FROM measurements")
                stats['measurements_count'] = cursor.fetchone()[0]
                
                cursor = self.conn.execute("SELECT COUNT(DISTINCT wdss_id) FROM measurements")
                stats['systems_with_measurements'] = cursor.fetchone()[0]
            
            return stats
            
        except sqlite3.Error as e:
            log.error(f"Error getting catalog statistics: {e}")
            return None

    def get_all_wds_ids(self, limit: Optional[int] = None, only_el_badry: bool = False) -> list[str]:
        """Get list of all WDS IDs in the database.
        
        Args:
            limit: Maximum number of IDs to return (None for all)
            only_el_badry: If True, only return systems in the El-Badry et al. (2021) catalog
            
        Returns:
            List of WDS identifiers
        """
        try:
            conditions = []
            if only_el_badry:
                conditions.append("in_el_badry_catalog = 1")  # SQLite uses 1 for True
            
            where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            query = f"SELECT wds_id FROM {self.summary_table}{where_clause} ORDER BY wds_id"
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor = self.conn.execute(query)
            result = [row[0] for row in cursor.fetchall()]
            
            if only_el_badry and result:
                log.info(f"Filtered to {len(result)} systems from El-Badry et al. (2021) catalog")
            
            return result
            
        except sqlite3.Error as e:
            if only_el_badry and "no such column: in_el_badry_catalog" in str(e):
                log.error("Database does not contain El-Badry catalog information.")
                log.error("Recreate the database with: python scripts/convert_catalogs_to_sqlite.py --el-badry-file <path_to_el_badry_catalog.fits>")
            else:
                log.error(f"Error getting WDS IDs: {e}")
            return []

    async def get_precomputed_physicality(self, wds_id: str) -> Optional[PhysicalityAssessment]:
        """
        Retrieves pre-computed physicality assessment from the local database
        based on El-Badry et al. (2021) cross-match data.
        
        This provides instant physicality validation for systems that are present
        in the gold-standard El-Badry catalog, using their R_chance_align values
        to determine physical vs. optical nature.
        
        Args:
            wds_id: WDS identifier to check
            
        Returns:
            PhysicalityAssessment if system is in El-Badry catalog, None otherwise
            
        References:
            El-Badry et al. (2021), MNRAS, 506, 2269-2295
        """
        try:
            # First, verify that the El-Badry columns exist to avoid failures
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({self.summary_table})")
            columns = {row[1] for row in cursor.fetchall()}
            
            if 'in_el_badry_catalog' not in columns:
                log.debug("Database does not contain El-Badry catalog information")
                return None  # The database was not enriched with El-Badry data

            # Query for El-Badry data
            cursor = self.conn.execute(
                f"""SELECT in_el_badry_catalog, R_chance_align, binary_type 
                   FROM {self.summary_table} WHERE wds_id = ?""",
                (wds_id,)
            )
            row = cursor.fetchone()

            if row and row['in_el_badry_catalog']:
                r_chance = row['R_chance_align']
                binary_type = row['binary_type']
                
                # Determine physicality label based on R_chance_align
                # R_chance_align: probability that alignment is by chance
                # Lower values = more likely to be physical
                label = PhysicalityLabel.AMBIGUOUS
                confidence = 0.5
                
                if r_chance is not None:
                    if r_chance < 0.1:
                        label = PhysicalityLabel.LIKELY_PHYSICAL
                        confidence = 1.0 - r_chance
                    elif r_chance > 0.9:
                        label = PhysicalityLabel.LIKELY_OPTICAL  
                        confidence = r_chance
                    else:
                        # Ambiguous range
                        confidence = 0.5
                else:
                    # If in catalog but no R_chance, consider likely physical
                    # (presence in El-Badry catalog implies high confidence)
                    label = PhysicalityLabel.LIKELY_PHYSICAL
                    confidence = 0.9

                # Import datetime here to avoid circular imports
                from datetime import datetime
                
                return PhysicalityAssessment(
                    label=label,
                    confidence=confidence,
                    p_value=(1.0 - r_chance) if r_chance is not None else 0.9,
                    method=ValidationMethod.STATISTICAL_ANALYSIS,  # Could create EL_BADRY_2021
                    parallax_consistency=None,  # Not provided by El-Badry catalog
                    proper_motion_consistency=None,  # Not provided by El-Badry catalog
                    gaia_source_id_primary=None,  # Available but not exposed in this interface
                    gaia_source_id_secondary=None,  # Available but not exposed in this interface
                    validation_date=datetime.now().isoformat(),
                    search_radius_arcsec=0.0,  # Not applicable for catalog match
                    significance_thresholds={
                        'r_chance_physical': 0.1,
                        'r_chance_optical': 0.9
                    },
                    retry_attempts=0,
                    notes=f"El-Badry catalog match. Binary type: {binary_type or 'unknown'}"
                )
                
            return None  # Not found in El-Badry catalog
            
        except Exception as e:
            log.error(f"Error retrieving precomputed physicality for {wds_id}: {e}")
            return None
