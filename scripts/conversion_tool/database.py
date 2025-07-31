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
    
    # DEBUG: Check what columns we're actually saving
    log.info(f"WDSS summary table columns: {list(df_wds.columns)[:10]}... (showing first 10)")
    log.info(f"WDSS summary table shape: {df_wds.shape}")
    
    # CRITICAL CHECK: Verify multi-pair columns exist
    critical_cols = ['component_pair', 'system_pair_id']
    missing_cols = [col for col in critical_cols if col not in df_wds.columns]
    if missing_cols:
        log.error(f"CRITICAL: Missing multi-pair columns: {missing_cols}")
        log.error("The revolutionary multi-pair architecture was not preserved!")
    else:
        log.info("✅ Multi-pair columns found in summary table")
    
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
        
        # MULTI-PAIR ARCHITECTURE: Critical indexes for new columns
        if 'component_pair' in df_wds.columns:
            conn.execute('CREATE INDEX idx_wdss_summary_component_pair ON wdss_summary(component_pair)')
            log.info("✅ Created component_pair index")
        
        if 'system_pair_id' in df_wds.columns:
            conn.execute('CREATE INDEX idx_wdss_summary_system_pair_id ON wdss_summary(system_pair_id)')
            log.info("✅ Created system_pair_id index")
        
        # Combined index for efficient multi-pair queries
        if 'wds_id' in df_wds.columns and 'component_pair' in df_wds.columns:
            conn.execute('CREATE INDEX idx_wdss_summary_wds_pair ON wdss_summary(wds_id, component_pair)')
            log.info("✅ Created combined wds_id+component_pair index")
        
        log.info(f"Created wdss_summary table with {len(df_wds)} entries and performance indexes")
        
        # Debug ORB6 DataFrame
        log.info(f"ORB6 DataFrame columns: {df_orb6.columns.tolist()}")
        log.info(f"ORB6 DataFrame shape: {df_orb6.shape}")
        log.info(f"ORB6 duplicated columns: {df_orb6.columns.duplicated().any()}")
        if df_orb6.columns.duplicated().any():
            log.error(f"Duplicated column names found: {df_orb6.columns[df_orb6.columns.duplicated()]}")
        
        # Create ORB6 table
        df_orb6.to_sql('orbital_elements', conn, if_exists='replace', index=False)
        
        # Create index only if table has data and wds_id column exists
        if len(df_orb6) > 0 and 'wds_id' in df_orb6.columns:
            conn.execute('CREATE INDEX idx_orbital_elements_id ON orbital_elements(wds_id)')
            log.info(f"Created orbital_elements table with {len(df_orb6)} entries and index")
        else:
            log.info(f"Created orbital_elements table with {len(df_orb6)} entries (no index - empty or missing wds_id)")
        
        # Create measurements table if available
        if df_measurements is not None and len(df_measurements) > 0:
            df_measurements.to_sql('measurements', conn, if_exists='replace', index=False)
            
            # Check if required columns exist before creating indexes
            if 'wdss_id' in df_measurements.columns:
                conn.execute('CREATE INDEX idx_measurements_wdss_id ON measurements(wdss_id)')
                log.info(f"Created wdss_id index for measurements table")
            
            if 'epoch' in df_measurements.columns:
                conn.execute('CREATE INDEX idx_measurements_epoch ON measurements(epoch)')
                log.info(f"Created epoch index for measurements table")
            
            log.info(f"Created measurements table with {len(df_measurements)} entries")
        
        # Vacuum database for optimal performance
        conn.execute('VACUUM')
        
        log.info("SQLite database created successfully")
        
    finally:
        conn.close()
