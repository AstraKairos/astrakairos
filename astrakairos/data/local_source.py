"""
Local data source implementation using SQLite database for catalog access.

This implementation provides access to WDS catalogs using a local SQLite database 
with proper indexing for efficient querying.
"""

import json
import logging
import sqlite3
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

from astropy.table import Table

from .source import (DataSource, OrbitalElements, PhysicalityAssessment,
                     WdsSummary, PhysicalityLabel, ValidationMethod,
                     WdsIdNotFoundError, MeasurementsUnavailableError,
                     OrbitalElementsUnavailableError, AstraKairosDataError)
from ..config import (
    DEFAULT_SQLITE_CACHE_SIZE_KB, MAX_SQLITE_CACHE_SIZE_KB, 
    SQLITE_MMAP_SIZE_GB, RAM_FRACTION_FOR_CACHE,
    WDSS_SUMMARY_TABLE, WDS_SUMMARY_TABLE, ORBITAL_ELEMENTS_TABLE,
    MEASUREMENTS_TABLE, REQUIRED_TABLES,
    EL_BADRY_PHYSICAL_THRESHOLD, EL_BADRY_OPTICAL_THRESHOLD,
    EL_BADRY_DEFAULT_CONFIDENCE
)

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
            AstraKairosDataError: If database structure is invalid
        """
        self.database_path = database_path
        self.conn = None
        self.summary_table = None
        self.has_measurements = False
        self.has_el_badry_data = False
        
        try:
            # Open database in read-only mode for safety and performance
            self.conn = sqlite3.connect(f'file:{database_path}?mode=ro', uri=True)
            self.conn.row_factory = sqlite3.Row
            
            # Performance optimizations for read-only access
            self._configure_performance_settings()
            
            # Verify database structure and detect capabilities
            self._verify_database_structure()
            
            log.info(f"Connected to local catalog database: {database_path}")
            
        except sqlite3.Error as e:
            log.error(f"Failed to connect to catalog database {database_path}: {e}")
            raise AstraKairosDataError(f"Database connection failed: {e}") from e
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
            
            # Calculate optimal cache size using configuration constants
            try:
                import psutil
                available_ram_kb = psutil.virtual_memory().available // 1024
                cache_size_kb = min(
                    int(available_ram_kb * RAM_FRACTION_FOR_CACHE), 
                    MAX_SQLITE_CACHE_SIZE_KB
                )
            except ImportError:
                cache_size_kb = DEFAULT_SQLITE_CACHE_SIZE_KB
                log.debug("psutil not available, using default cache size")
            
            self.conn.execute(f'PRAGMA cache_size = -{cache_size_kb}')
            
            # Enable memory-mapped I/O
            mmap_size = min(SQLITE_MMAP_SIZE_GB * 1024 * 1024 * 1024, cache_size_kb * 1024 * 2)
            self.conn.execute(f'PRAGMA mmap_size = {mmap_size}')
            
            # Optimize query planner
            self.conn.execute('PRAGMA optimize')
            
            log.debug(f"SQLite performance configured: cache={cache_size_kb}KB, mmap={mmap_size//1024//1024}MB")
            
        except Exception as e:
            log.warning(f"Could not apply all performance optimizations: {e}")

    def _verify_database_structure(self) -> None:
        """Verify database structure and detect available capabilities."""
        cursor = self.conn.cursor()
        
        # Check for required tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        # Check for either WDSS format or traditional WDS format
        has_wdss = WDSS_SUMMARY_TABLE in tables
        has_wds = WDS_SUMMARY_TABLE in tables
        has_orbital = ORBITAL_ELEMENTS_TABLE in tables
        
        if not (has_wdss or has_wds):
            raise AstraKairosDataError(
                f"Database missing summary table: need either '{WDSS_SUMMARY_TABLE}' or '{WDS_SUMMARY_TABLE}'"
            )
        
        if not has_orbital:
            raise AstraKairosDataError(f"Database missing required table: '{ORBITAL_ELEMENTS_TABLE}'")
        
        # Set the table name to use for queries
        self.summary_table = WDSS_SUMMARY_TABLE if has_wdss else WDS_SUMMARY_TABLE
        
        # Check if measurements table exists (optional)
        self.has_measurements = MEASUREMENTS_TABLE in tables
        
        # Check for El-Badry enrichment data
        if self.summary_table:
            cursor.execute(f"PRAGMA table_info({self.summary_table})")
            columns = {row[1] for row in cursor.fetchall()}
            self.has_el_badry_data = 'in_el_badry_catalog' in columns
            self.has_gaia_source_ids = 'gaia_source_ids' in columns
        
        log.info(f"Database verified. Using {self.summary_table}. "
                f"Has measurements: {self.has_measurements}. "
                f"Has El-Badry data: {self.has_el_badry_data}. "
                f"Has Gaia source IDs: {self.has_gaia_source_ids}")

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

    async def get_all_component_pairs(self, wds_id: str) -> List[WdsSummary]:
        """Get WDS summary data for ALL component pairs of a system.
        
        For multi-pair systems, this returns a separate WdsSummary for each 
        component pair (AB, AC, BD, CE, etc.) as independent systems.
        
        Args:
            wds_id: WDS identifier
            
        Returns:
            List of WdsSummary objects, one for each component pair
            
        Raises:
            WdsIdNotFoundError: If the WDS identifier is not found
            AstraKairosDataError: If database query fails
        """
        normalized_id = wds_id.strip()
        
        try:
            # Check available columns to tailor query to database schema
            cursor = self.conn.execute(f"PRAGMA table_info({self.summary_table})")
            columns = {row[1] for row in cursor.fetchall()}
            has_component_pair = 'component_pair' in columns

            component_pair_alias = (
                "component_pair as components" if has_component_pair else "'' as components"
            )
            component_pair_field = (
                'component_pair' if has_component_pair else 'NULL as component_pair'
            )
            system_pair_field = (
                'system_pair_id' if 'system_pair_id' in columns else 'NULL as system_pair_id'
            )
            primary_component_field = (
                'pair_primary_component' if 'pair_primary_component' in columns else 'NULL as pair_primary_component'
            )
            secondary_component_field = (
                'pair_secondary_component' if 'pair_secondary_component' in columns else 'NULL as pair_secondary_component'
            )
            gaia_primary_field = (
                'gaia_id_primary' if 'gaia_id_primary' in columns else 'NULL as gaia_id_primary'
            )
            gaia_secondary_field = (
                'gaia_id_secondary' if 'gaia_id_secondary' in columns else 'NULL as gaia_id_secondary'
            )

            # Determine which columns to select (gaia_source_ids may not exist)
            gaia_column = "gaia_source_ids" if self.has_gaia_source_ids else "NULL as gaia_source_ids"

            select_columns = [
                "wds_id",
                "discoverer_designation as discoverer",
                component_pair_alias,
                "date_first",
                "date_last",
                "n_obs as n_observations",
                "pa_first",
                "pa_last",
                "sep_first",
                "sep_last",
                "vmag as mag_pri",
                "kmag as mag_sec",
                "spectral_type as spec_type",
                "ra_deg",
                "dec_deg",
                "wdss_id",
                "discoverer_designation",
                system_pair_field,
                component_pair_field,
                primary_component_field,
                secondary_component_field,
                gaia_primary_field,
                gaia_secondary_field,
                "pa_first_error",
                "pa_last_error",
                "sep_first_error",
                "sep_last_error",
                gaia_column,
            ]

            select_clause = ", ".join(select_columns)
            order_clause = " ORDER BY component_pair" if has_component_pair else ""

            cursor = self.conn.execute(
                f"SELECT {select_clause} FROM {self.summary_table} WHERE wds_id = ?{order_clause}",
                (normalized_id,)
            )
            
            rows = cursor.fetchall()
            if not rows:
                raise WdsIdNotFoundError(f"WDS ID '{normalized_id}' not found in catalog")
            
            results = []
            for row in rows:
                # Convert row to dict and filter None values for required fields
                data = dict(row)
                
                # Keep gaia_source_ids from database if available
                # (Remove hardcoded None assignment to allow real Gaia IDs)
                
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
                
                results.append(WdsSummary(**valid_fields))
            
            return results
            
        except sqlite3.Error as e:
            log.error(f"Database error retrieving WDS summary for {normalized_id}: {e}")
            raise AstraKairosDataError(f"Database query failed: {e}") from e
        except TypeError as e:
            log.error(f"Error creating WdsSummary for {normalized_id}: {e}")
            raise AstraKairosDataError(f"Invalid data format: {e}") from e

    async def get_wds_summary(self, wds_id: str) -> WdsSummary:
        """Get WDS summary data for a system.
        
        Args:
            wds_id: WDS identifier
            
        Returns:
            WdsSummary object
            
        Raises:
            WdsIdNotFoundError: If the WDS identifier is not found
            AstraKairosDataError: If database query fails
        """
        normalized_id = wds_id.strip()
        
        try:
            cursor = self.conn.execute(f"PRAGMA table_info({self.summary_table})")
            columns = {row[1] for row in cursor.fetchall()}
            has_component_pair = 'component_pair' in columns

            component_pair_alias = (
                "component_pair as components" if has_component_pair else "'' as components"
            )
            component_pair_field = (
                'component_pair' if has_component_pair else 'NULL as component_pair'
            )
            system_pair_field = (
                'system_pair_id' if 'system_pair_id' in columns else 'NULL as system_pair_id'
            )
            primary_component_field = (
                'pair_primary_component' if 'pair_primary_component' in columns else 'NULL as pair_primary_component'
            )
            secondary_component_field = (
                'pair_secondary_component' if 'pair_secondary_component' in columns else 'NULL as pair_secondary_component'
            )
            gaia_primary_field = (
                'gaia_id_primary' if 'gaia_id_primary' in columns else 'NULL as gaia_id_primary'
            )
            gaia_secondary_field = (
                'gaia_id_secondary' if 'gaia_id_secondary' in columns else 'NULL as gaia_id_secondary'
            )

            gaia_column = "gaia_source_ids" if self.has_gaia_source_ids else "NULL as gaia_source_ids"

            select_columns = [
                "wds_id",
                "discoverer_designation as discoverer",
                component_pair_alias,
                "date_first",
                "date_last",
                "n_obs as n_observations",
                "pa_first",
                "pa_last",
                "sep_first",
                "sep_last",
                "vmag as mag_pri",
                "kmag as mag_sec",
                "spectral_type as spec_type",
                "ra_deg",
                "dec_deg",
                "wdss_id",
                "discoverer_designation",
                system_pair_field,
                component_pair_field,
                primary_component_field,
                secondary_component_field,
                gaia_primary_field,
                gaia_secondary_field,
                "pa_first_error",
                "pa_last_error",
                "sep_first_error",
                "sep_last_error",
                gaia_column,
            ]

            select_clause = ", ".join(select_columns)

            cursor = self.conn.execute(
                f"SELECT {select_clause} FROM {self.summary_table} WHERE wds_id = ?",
                (normalized_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                raise WdsIdNotFoundError(f"WDS ID '{normalized_id}' not found in catalog")
            
            # Convert row to dict and filter None values for required fields
            data = dict(row)
            
            # Keep gaia_source_ids from database if available
            # (Remove hardcoded None assignment to allow real Gaia IDs)
            
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
            
            return WdsSummary(**valid_fields)
            
        except sqlite3.Error as e:
            log.error(f"Database error retrieving WDS summary for {normalized_id}: {e}")
            raise AstraKairosDataError(f"Database query failed: {e}") from e
        except TypeError as e:
            log.error(f"Error creating WdsSummary for {normalized_id}: {e}")
            raise AstraKairosDataError(f"Invalid data format: {e}") from e

    async def get_all_measurements(self, wds_id: str) -> Table:
        """Get all measurements for a system with complete uncertainty information.
        
        Args:
            wds_id: WDS identifier
            
        Returns:
            Astropy Table with measurements including error columns
            
        Raises:
            WdsIdNotFoundError: If the WDS identifier is not found
            MeasurementsUnavailableError: If measurements table doesn't exist or system has no measurements
            AstraKairosDataError: If database query fails
        """
        if not self.has_measurements:
            raise MeasurementsUnavailableError("Database does not contain measurements table")
            
        normalized_id = wds_id.strip()
        
        try:
            # First get the wdss_id for this wds_id
            wdss_cursor = self.conn.execute(
                f"SELECT wdss_id FROM {self.summary_table} WHERE wds_id = ?",
                (normalized_id,)
            )
            wdss_row = wdss_cursor.fetchone()
            
            if not wdss_row:
                raise WdsIdNotFoundError(f"WDS ID '{normalized_id}' not found in catalog")
                
            wdss_id = wdss_row['wdss_id']
            
            schema_cursor = self.conn.execute(f"PRAGMA table_info({MEASUREMENTS_TABLE})")
            available_columns = [row[1] for row in schema_cursor.fetchall()]
            
            base_columns = ['wdss_id', 'epoch', 'theta', 'rho']
            optional_columns = ['theta_error', 'rho_error', 'error_source', 'technique', 'reference']
            
            # Only include columns that exist in the database
            query_columns = base_columns + [col for col in optional_columns if col in available_columns]
            column_list = ', '.join(query_columns)
            
            cursor = self.conn.execute(
                f"SELECT {column_list} FROM {MEASUREMENTS_TABLE} WHERE wdss_id = ? ORDER BY epoch",
                (wdss_id,)
            )
            
            rows = cursor.fetchall()
            if not rows:
                raise MeasurementsUnavailableError(f"No measurements found for WDS ID '{normalized_id}'")
            
            # Create astropy Table from the available data
            table_data = {'wds_id': [normalized_id] * len(rows)}
            
            # Add all available columns
            for col in query_columns:
                table_data[col] = [row[col] for row in rows]
            
            return Table(table_data)
            
        except sqlite3.Error as e:
            log.error(f"Database error retrieving measurements for {normalized_id}: {e}")
            raise AstraKairosDataError(f"Database query failed: {e}") from e

    async def get_orbital_elements(self, wds_id: str) -> OrbitalElements:
        """Get orbital elements for a system.
        
        Args:
            wds_id: WDS identifier
            
        Returns:
            OrbitalElements object
            
        Raises:
            WdsIdNotFoundError: If the WDS identifier is not found
            OrbitalElementsUnavailableError: If orbital elements are not available
            AstraKairosDataError: If database query fails
        """
        normalized_id = wds_id.strip()
        
        try:
            # For production databases, we can assume the schema. For tests, we need to check.
            # This is a compromise to maintain both performance and test compatibility.
            schema_cursor = self.conn.execute(f"PRAGMA table_info({ORBITAL_ELEMENTS_TABLE})")
            available_columns = [row[1] for row in schema_cursor.fetchall()]
            
            base_columns = ['wds_id', 'P', 'a', 'i', 'Omega', 'T', 'e', 'omega_arg', 'grade']
            error_columns = ['e_P', 'e_a', 'e_i', 'e_Omega', 'e_T', 'e_e', 'e_omega_arg'] 
            metadata_columns = ['reference', 'last_updated']
            
            # Only include columns that exist in the database
            query_columns = [col for col in base_columns if col in available_columns]
            query_columns += [col for col in error_columns if col in available_columns]
            query_columns += [col for col in metadata_columns if col in available_columns]
            
            # Handle omega_arg -> omega alias properly
            select_columns = []
            for col in query_columns:
                if col == 'omega_arg':
                    select_columns.append('omega_arg as omega')
                elif col == 'Omega':
                    # Use a different alias to avoid confusion with omega
                    select_columns.append('Omega as node_longitude')
                else:
                    select_columns.append(col)
            
            select_clause = ', '.join(select_columns)
            
            cursor = self.conn.execute(
                f"SELECT {select_clause} FROM {ORBITAL_ELEMENTS_TABLE} WHERE wds_id = ?",
                (normalized_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                raise OrbitalElementsUnavailableError(f"No orbital elements found for WDS ID '{normalized_id}'")
            
            # Convert sqlite3 row to dict
            data = dict(row)
            
            # Filter out None values only for non-error fields
            filtered_data = {}
            for k, v in data.items():
                # Skip omega_arg if we have omega alias
                if k == 'omega_arg' and 'omega' in data:
                    continue
                # Restore Omega from node_longitude alias
                elif k == 'node_longitude':
                    filtered_data['Omega'] = v
                elif k.startswith('e_') and k != 'e':  # Error fields (but not eccentricity 'e')
                    # Keep error fields even if None - indicates missing uncertainty
                    filtered_data[k] = v
                elif v is not None:
                    # Filter None values for regular fields
                    filtered_data[k] = v
            
            # Add required fields if missing
            if 'reference' not in filtered_data:
                filtered_data['reference'] = 'Unknown'
            if 'last_updated' not in filtered_data:
                filtered_data['last_updated'] = datetime.now().isoformat()
            
            return filtered_data
            
        except sqlite3.Error as e:
            log.error(f"Database error retrieving orbital elements for {normalized_id}: {e}")
            raise AstraKairosDataError(f"Database query failed: {e}") from e
        except TypeError as e:
            log.error(f"Error creating OrbitalElements for {normalized_id}: {e}")
            raise AstraKairosDataError(f"Invalid data format: {e}") from e

    async def validate_physicality(self, system_data: WdsSummary) -> PhysicalityAssessment:
        """
        Local source cannot perform physicality validation.
        Returns unknown status with proper PhysicalityAssessment structure.
        
        Args:
            system_data: System data (required but unused)
            
        Returns:
            PhysicalityAssessment with UNKNOWN label
        """
        return PhysicalityAssessment(
            label=PhysicalityLabel.UNKNOWN,
            confidence=0.0,
            p_value=None,
            method=None,
            parallax_consistency=None,
            proper_motion_consistency=None,
            gaia_source_id_primary=None,
            gaia_source_id_secondary=None,
            validation_date=datetime.now().isoformat(),
            search_radius_arcsec=0.0,
            significance_thresholds={},
            retry_attempts=0
        )

    def get_catalog_statistics(self) -> Dict[str, int]:
        """Get statistics about the loaded catalogs.
        
        Returns:
            Dictionary with table sizes and other statistics
            
        Raises:
            AstraKairosDataError: If database query fails
        """
        if not self.conn:
            raise AstraKairosDataError("Database connection is closed")
            
        try:
            stats = {}
            
            # Summary table count
            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {self.summary_table}")
            stats[f'{self.summary_table}_count'] = cursor.fetchone()[0]
            
            # Orbital elements count
            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {ORBITAL_ELEMENTS_TABLE}")
            stats['orbital_elements_count'] = cursor.fetchone()[0]
            
            # Measurements count (if available)
            if self.has_measurements:
                cursor = self.conn.execute(f"SELECT COUNT(*) FROM {MEASUREMENTS_TABLE}")
                stats['measurements_count'] = cursor.fetchone()[0]
                
                cursor = self.conn.execute(f"SELECT COUNT(DISTINCT wdss_id) FROM {MEASUREMENTS_TABLE}")
                stats['systems_with_measurements'] = cursor.fetchone()[0]
            
            return stats
            
        except sqlite3.Error as e:
            log.error(f"Error getting catalog statistics: {e}")
            raise AstraKairosDataError(f"Database query failed: {e}") from e

    def get_all_wds_ids(self, 
                       limit: Optional[int] = None, 
                       only_el_badry: bool = False,
                       ra_range: Optional[Tuple[float, float]] = None,
                       dec_range: Optional[Tuple[float, float]] = None) -> list[str]:
        """Get list of all WDS IDs in the database with optional spatial filtering.
        
        Args:
            limit: Maximum number of IDs to return (None for all)
            only_el_badry: If True, only return systems in the El-Badry et al. (2021) catalog
            ra_range: Optional RA range in hours (min, max). Handles wraparound cases.
            dec_range: Optional Dec range in degrees (min, max)
            
        Returns:
            List of WDS identifiers
            
        Raises:
            AstraKairosDataError: If database query fails or El-Badry data not available
        """
        try:
            conditions = []
            params = []
            
            # El-Badry catalog filtering
            if only_el_badry:
                if not self.has_el_badry_data:
                    raise AstraKairosDataError(
                        "Database does not contain El-Badry catalog information. "
                        "Recreate the database with: python scripts/convert_catalogs_to_sqlite.py "
                        "--el-badry-file <path_to_el_badry_catalog.fits>"
                    )
                conditions.append("in_el_badry_catalog = 1")
            
            # Spatial filtering: Declination (simpler case - no wraparound)
            if dec_range:
                min_dec, max_dec = dec_range
                conditions.append("dec_deg BETWEEN ? AND ?")
                params.extend([min_dec, max_dec])
                log.debug(f"Applied declination filter: [{min_dec:.2f}, {max_dec:.2f}] degrees")
            
            # Spatial filtering: Right Ascension (handles wraparound)
            if ra_range:
                min_ra_hours, max_ra_hours = ra_range
                # Convert hours to degrees for database query
                min_ra_deg = min_ra_hours * 15.0
                max_ra_deg = max_ra_hours * 15.0
                
                if min_ra_deg <= max_ra_deg:
                    # Normal case: range doesn't cross 0h (e.g., 6h to 18h)
                    conditions.append("ra_deg BETWEEN ? AND ?")
                    params.extend([min_ra_deg, max_ra_deg])
                    log.debug(f"Applied RA filter (normal): [{min_ra_deg:.1f}, {max_ra_deg:.1f}] degrees")
                else:
                    # Wraparound case: range crosses 0h/24h boundary (e.g., 22h to 2h)
                    # This translates to: (RA >= 22h OR RA <= 2h)
                    conditions.append("(ra_deg >= ? OR ra_deg <= ?)")
                    params.extend([min_ra_deg, max_ra_deg])
                    log.debug(f"Applied RA filter (wraparound): RA >= {min_ra_deg:.1f}° OR RA <= {max_ra_deg:.1f}°")

            where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            query = f"SELECT DISTINCT wds_id FROM {self.summary_table}{where_clause} ORDER BY wds_id"
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor = self.conn.execute(query, params)
            result = [row[0] for row in cursor.fetchall()]
            
            # Log filtering results for transparency
            if ra_range or dec_range:
                log.info(f"Spatial filtering returned {len(result)} systems")
            
            if only_el_badry and result:
                log.info(f"Filtered to {len(result)} systems from El-Badry et al. (2021) catalog")
            
            return result
            
        except sqlite3.Error as e:
            log.error(f"Error getting WDS IDs: {e}")
            raise AstraKairosDataError(f"Database query failed: {e}") from e

    async def get_precomputed_physicality(self, wds_id: str, wdss_id: Optional[str] = None) -> Optional[PhysicalityAssessment]:
        """
        Retrieves pre-computed physicality assessment from the local database
        based on El-Badry et al. (2021) cross-match data.
        
        This provides instant physicality validation for systems that are present
        in the gold-standard El-Badry catalog, using their R_chance_align values
        to determine physical vs. optical nature.
        
        Args:
            wds_id: WDS identifier to check
            wdss_id: Optional WDSS identifier (unique per component pair).
                     If provided, this ensures correct validation for multiple 
                     component systems (triples, quadruples) where wds_id alone
                     would return multiple rows.
            
        Returns:
            PhysicalityAssessment if system is in El-Badry catalog, None otherwise
            
        Raises:
            AstraKairosDataError: If database query fails
            
        References:
            El-Badry et al. (2021), MNRAS, 506, 2269-2295
        """
        if not self.has_el_badry_data:
            log.debug("Database does not contain El-Badry catalog information")
            return None

        try:
            # Use wdss_id if available (unique per component pair) to avoid ambiguity
            # in multiple component systems where wds_id returns multiple rows
            if wdss_id:
                cursor = self.conn.execute(
                    f"""SELECT in_el_badry_catalog, R_chance_align, binary_type 
                       FROM {self.summary_table} WHERE wdss_id = ?""",
                    (wdss_id,)
                )
            else:
                cursor = self.conn.execute(
                    f"""SELECT in_el_badry_catalog, R_chance_align, binary_type 
                       FROM {self.summary_table} WHERE wds_id = ?""",
                    (wds_id,)
                )
            row = cursor.fetchone()

            if row and row['in_el_badry_catalog']:
                r_chance = row['R_chance_align']
                binary_type = row['binary_type']
                
                if r_chance is not None:
                    # Clamp r_chance to [0, 1] to avoid invalid values
                    r_chance_clamped = max(0.0, min(1.0, r_chance))
                    
                    # Use conservative thresholds based on R_chance_align (chance alignment probability)
                    if r_chance_clamped < 0.01:  # < 1% chance alignment = Likely Physical (>99% confidence)
                        label = PhysicalityLabel.LIKELY_PHYSICAL
                        confidence = 1.0 - r_chance_clamped  # High confidence
                        p_value = confidence
                    elif r_chance_clamped > 0.99:  # > 99% chance alignment = Likely Optical
                        label = PhysicalityLabel.LIKELY_OPTICAL
                        confidence = r_chance_clamped  # High confidence it's optical
                        p_value = 1.0 - confidence  # Low p-value for physical binding
                    else:  # 0.01 <= r_chance <= 0.99 = Ambiguous
                        label = PhysicalityLabel.AMBIGUOUS
                        confidence = 0.5  # Neutral confidence
                        p_value = 0.5   # Neutral p-value
                else:
                    # If in catalog but no R_chance, use default moderate confidence
                    label = PhysicalityLabel.LIKELY_PHYSICAL
                    confidence = EL_BADRY_DEFAULT_CONFIDENCE
                    p_value = confidence
                
                return PhysicalityAssessment(
                    label=label,
                    confidence=confidence,
                    p_value=p_value,
                    method=ValidationMethod.STATISTICAL_ANALYSIS,
                    parallax_consistency=None,
                    proper_motion_consistency=None,
                    gaia_source_id_primary=None,
                    gaia_source_id_secondary=None,
                    validation_date=datetime.now().isoformat(),
                    search_radius_arcsec=0.0,
                    significance_thresholds={
                        'r_chance_physical': EL_BADRY_PHYSICAL_THRESHOLD,
                        'r_chance_optical': EL_BADRY_OPTICAL_THRESHOLD
                    },
                    retry_attempts=0,
                    notes=f"El-Badry catalog system (physical by definition). Binary type: {binary_type or 'unknown'}, R_chance_align: {r_chance}"
                )
                
            return None
            
        except sqlite3.Error as e:
            log.error(f"Error retrieving precomputed physicality for {wds_id}: {e}")
            raise AstraKairosDataError(f"Database query failed: {e}") from e
