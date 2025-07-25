"""
Functions to create and populate SQLite databases.

This module handles the creation of optimized SQLite databases
with appropriate indexes for efficient astronomical queries.
"""

import logging
import sqlite3
import pandas as pd
from typing import Optional

log = logging.getLogger(__name__)


def create_sqlite_database(df_wds: pd.DataFrame, df_orb6: pd.DataFrame, 
                          df_measurements: Optional[pd.DataFrame], 
                          output_path: str) -> None:
    """
    Create SQLite database with proper indexing.
    
    Args:
        df_wds: WDSS summary table
        df_orb6: ORB6 orbital elements table
        df_measurements: Individual measurements table (optional)
        output_path: Path for output SQLite database
    """
    log.info(f"Creating SQLite database: {output_path}")
    
    conn = sqlite3.connect(output_path)
    
    try:
        # Explicitly drop existing tables
        conn.execute('DROP TABLE IF EXISTS wdss_summary')
        conn.execute('DROP TABLE IF EXISTS orbital_elements')  
        conn.execute('DROP TABLE IF EXISTS measurements')
        log.info("Dropped existing tables")
        
        # Create WDSS summary table
        df_wds.to_sql('wdss_summary', conn, if_exists='replace', index=False)
        conn.execute('CREATE INDEX idx_wdss_summary_id ON wdss_summary(wds_id)')
        
        # Critical spatial index for coordinate queries (dec_deg first for better selectivity)
        conn.execute('CREATE INDEX idx_wdss_summary_coords ON wdss_summary(dec_deg, ra_deg)')
        
        # Additional performance indexes for common query patterns (only if columns exist)
        if 'date_last' in df_wds.columns:
            conn.execute('CREATE INDEX idx_wdss_summary_date_last ON wdss_summary(date_last)')
        if 'n_obs' in df_wds.columns:
            conn.execute('CREATE INDEX idx_wdss_summary_n_obs ON wdss_summary(n_obs)')
        
        log.info(f"Created wdss_summary table with {len(df_wds)} entries and performance indexes")
        
        # Debug ORB6 DataFrame
        log.info(f"ORB6 DataFrame columns: {df_orb6.columns.tolist()}")
        log.info(f"ORB6 DataFrame shape: {df_orb6.shape}")
        log.info(f"ORB6 duplicated columns: {df_orb6.columns.duplicated().any()}")
        if df_orb6.columns.duplicated().any():
            log.error(f"Duplicated column names found: {df_orb6.columns[df_orb6.columns.duplicated()]}")
        
        # Create ORB6 table
        df_orb6.to_sql('orbital_elements', conn, if_exists='replace', index=False)
        conn.execute('CREATE INDEX idx_orbital_elements_id ON orbital_elements(wds_id)')
        log.info(f"Created orbital_elements table with {len(df_orb6)} entries")
        
        # Create measurements table if available
        if df_measurements is not None:
            df_measurements.to_sql('measurements', conn, if_exists='replace', index=False)
            conn.execute('CREATE INDEX idx_measurements_wdss_id ON measurements(wdss_id)')
            conn.execute('CREATE INDEX idx_measurements_epoch ON measurements(epoch)')
            log.info(f"Created measurements table with {len(df_measurements)} entries")
        
        # Vacuum database for optimal performance
        conn.execute('VACUUM')
        
        log.info("SQLite database created successfully")
        
    finally:
        conn.close()
