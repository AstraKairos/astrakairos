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
            # Connect to SQLite database
            self.conn = sqlite3.connect(database_path)
            self.conn.row_factory = sqlite3.Row
            
            # Verify database structure
            self._verify_database_structure()
            
            log.info(f"Connected to local catalog database: {database_path}")
            
        except sqlite3.Error as e:
            log.error(f"Failed to connect to catalog database {database_path}: {e}")
            raise
        except FileNotFoundError as e:
            log.error(f"Catalog database not found: {database_path}")
            raise

    def _verify_database_structure(self) -> None:
        """Verify that database has required tables and structure."""
        cursor = self.conn.cursor()
        
        # Check for required tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        required_tables = {'wds_summary', 'orbital_elements'}
        missing_tables = required_tables - tables
        
        if missing_tables:
            raise sqlite3.Error(f"Database missing required tables: {missing_tables}")
        
        # Check if measurements table exists (optional)
        self.has_measurements = 'measurements' in tables
        
        log.info(f"Database verified. Has measurements: {self.has_measurements}")

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
                """SELECT wds_id, discoverer, components, date_first, date_last, 
                          obs as n_observations, pa_first, pa_last, sep_first, sep_last, 
                          mag_pri, mag_sec, spec_type, ra_deg, dec_deg 
                   FROM wds_summary WHERE wds_id = ?""",
                (normalized_id,)
            )
            
            row = cursor.fetchone()
            if row:
                # Convert row to dict and filter None values
                data = {k: v for k, v in dict(row).items() if v is not None}
                
                # Validate against WdsSummary schema
                valid_fields = {k: v for k, v in data.items() 
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
        """Get all measurements for a system.
        
        Args:
            wds_id: WDS identifier
            
        Returns:
            Astropy Table with measurements or None if not found
        """
        if not self.has_measurements:
            return None
            
        normalized_id = wds_id.strip()
        
        try:
            cursor = self.conn.execute(
                "SELECT wds_id, epoch, theta, rho FROM measurements WHERE wds_id = ? ORDER BY epoch",
                (normalized_id,)
            )
            
            rows = cursor.fetchall()
            if rows:
                # Convert to astropy table
                data = {
                    'wds_id': [row['wds_id'] for row in rows],
                    'epoch': [row['epoch'] for row in rows],
                    'theta': [row['theta'] for row in rows],
                    'rho': [row['rho'] for row in rows]
                }
                return Table(data)
            
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
                "SELECT wds_id, P, a, i, Omega, T, e, omega_arg as omega FROM orbital_elements WHERE wds_id = ?",
                (normalized_id,)
            )
            
            row = cursor.fetchone()
            if row:
                # Convert row to dict and filter None values
                data = {k: v for k, v in dict(row).items() if v is not None}
                
                # Validate against OrbitalElements schema
                valid_fields = {k: v for k, v in data.items() 
                               if k in OrbitalElements.__annotations__}
                
                if valid_fields:
                    return OrbitalElements(**valid_fields)
            
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
            
            # WDS summary count
            cursor = self.conn.execute("SELECT COUNT(*) FROM wds_summary")
            stats['wds_summary_count'] = cursor.fetchone()[0]
            
            # Orbital elements count
            cursor = self.conn.execute("SELECT COUNT(*) FROM orbital_elements")
            stats['orbital_elements_count'] = cursor.fetchone()[0]
            
            # Measurements count (if available)
            if self.has_measurements:
                cursor = self.conn.execute("SELECT COUNT(*) FROM measurements")
                stats['measurements_count'] = cursor.fetchone()[0]
                
                cursor = self.conn.execute("SELECT COUNT(DISTINCT wds_id) FROM measurements")
                stats['systems_with_measurements'] = cursor.fetchone()[0]
            
            return stats
            
        except sqlite3.Error as e:
            log.error(f"Error getting catalog statistics: {e}")
            return None

    def get_all_wds_ids(self, limit: Optional[int] = None) -> list[str]:
        """Get list of all WDS IDs in the database.
        
        Args:
            limit: Maximum number of IDs to return (None for all)
            
        Returns:
            List of WDS identifiers
        """
        try:
            query = "SELECT wds_id FROM wds_summary ORDER BY wds_id"
            if limit:
                query += f" LIMIT {limit}"
            
            cursor = self.conn.execute(query)
            return [row[0] for row in cursor.fetchall()]
            
        except sqlite3.Error as e:
            log.error(f"Error getting WDS IDs: {e}")
            return []
