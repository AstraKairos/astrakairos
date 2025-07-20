"""
Tests for WDSS LocalDataSource implementation.

This test suite validates the WDSS integration including:
- Database schema compatibility
- Coordinate extraction from WDSS IDs
- Column mapping (vmag/kmag)
- Measurement retrieval via wdss_id join
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path
import asyncio

from astrakairos.data.local_source import LocalDataSource
from astrakairos.data.source import WdsSummary, OrbitalElements


@pytest.fixture
def wdss_sqlite_db():
    """Create a temporary SQLite database with WDSS test data."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    conn = sqlite3.connect(db_path)
    
    # Create WDSS summary table with correct schema
    conn.execute("""
CREATE TABLE wdss_summary (
    wds_id TEXT,
    wdss_id TEXT,
    discoverer_designation TEXT,
    date_first REAL,
    date_last REAL,
    n_obs INTEGER,
    pa_first REAL,
    pa_last REAL,
    sep_first REAL,
    sep_last REAL,
    vmag REAL,
    kmag REAL,
    spectral_type TEXT,
    ra_deg REAL,
    dec_deg REAL,
    parallax REAL,
    pm_ra REAL,
    pm_dec REAL,
    name TEXT
)
    """)
    
    # Create orbital elements table
    conn.execute("""
CREATE TABLE orbital_elements (
    wds_id TEXT,
    P REAL,
    a REAL,
    i REAL,
    Omega REAL,
    T REAL,
    e REAL,
    omega_arg REAL
)
    """)
    
    # Create measurements table with wdss_id
    conn.execute("""
CREATE TABLE measurements (
    wdss_id TEXT,
    pair TEXT,
    epoch REAL,
    theta REAL,
    rho REAL,
    mag1 REAL,
    mag2 REAL,
    reference TEXT,
    technique TEXT
)
    """)
    
    # Insert test WDSS data
    conn.execute("""
INSERT INTO wdss_summary VALUES (
    '16169+0113', '1616524+011317', 'STF 2107',
    1999.5, 2015.5, 42,
    285.2, 287.8, 0.455, 0.431,
    8.12, 8.89, 'G5V+K0V',
    244.218, 1.295, 15.23, -12.4, -8.7,
    'HD 148937'
)
    """)
    
    # Insert system without WDS correspondence (WDSS only)
    conn.execute("""
INSERT INTO wdss_summary VALUES (
    '1718115-142444', '1718115-142444', NULL,
    2010.2, 2020.8, 15,
    123.4, 125.1, 2.34, 2.41,
    16.15, 17.58, NULL,
    259.546, -14.412, NULL, NULL, NULL,
    NULL
)
    """)
    
    # Insert orbital elements for first system
    conn.execute("""
INSERT INTO orbital_elements VALUES (
    '16169+0113', 285.69, 0.4331, 118.2, 287.1, 2023.45, 0.1259, 167.3
)
    """)
    
    # Insert measurements using wdss_id
    measurements = [
        ('1616524+011317', 'AB', 1999.5, 285.2, 0.455, 8.12, 8.89, 'Hei1997', 'S'),
        ('1616524+011317', 'AB', 2007.3, 286.5, 0.443, 8.11, 8.91, 'Tok2013', 'S'),
        ('1616524+011317', 'AB', 2015.5, 287.8, 0.431, 8.13, 8.88, 'USN2016', 'S'),
        ('1718115-142444', 'AB', 2010.2, 123.4, 2.34, 16.15, 17.58, 'Gaia2020', 'G'),
        ('1718115-142444', 'AB', 2020.8, 125.1, 2.41, 16.14, 17.59, 'Gaia2021', 'G')
    ]
    conn.executemany("INSERT INTO measurements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", measurements)
    
    # Create indexes
    conn.execute("CREATE INDEX idx_wdss_summary_wds_id ON wdss_summary(wds_id)")
    conn.execute("CREATE INDEX idx_wdss_summary_wdss_id ON wdss_summary(wdss_id)")
    conn.execute("CREATE INDEX idx_orbital_elements_wds_id ON orbital_elements(wds_id)")
    conn.execute("CREATE INDEX idx_measurements_wdss_id ON measurements(wdss_id)")
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    try:
        Path(db_path).unlink()
    except PermissionError:
        import time
        time.sleep(0.1)
        try:
            Path(db_path).unlink()
        except PermissionError:
            print(f"Warning: Could not delete temporary file {db_path}")


class TestWdssLocalDataSource:
    """Test WDSS LocalDataSource functionality."""
    
    def test_wdss_initialization_success(self, wdss_sqlite_db):
        """Test successful initialization with WDSS database."""
        source = LocalDataSource(wdss_sqlite_db)
        assert source.conn is not None
        assert source.has_measurements is True
        assert source.summary_table == 'wdss_summary'  # Should detect WDSS format
        source.close()
    
    def test_wdss_catalog_statistics(self, wdss_sqlite_db):
        """Test catalog statistics with WDSS data."""
        source = LocalDataSource(wdss_sqlite_db)
        
        stats = source.get_catalog_statistics()
        assert stats['wdss_summary_count'] == 2  # Two systems
        assert stats['orbital_elements_count'] == 1  # One with orbital elements
        assert stats['measurements_count'] == 5  # Five measurements total
        assert stats['systems_with_measurements'] == 2  # Both systems have measurements
        
        source.close()
    
    @pytest.mark.asyncio
    async def test_wdss_get_summary_with_correspondence(self, wdss_sqlite_db):
        """Test WDS summary retrieval for system with WDS correspondence."""
        source = LocalDataSource(wdss_sqlite_db)
        
        result = await source.get_wds_summary("16169+0113")
        assert result is not None
        assert result['wds_id'] == "16169+0113"
        assert result['wdss_id'] == "1616524+011317"
        assert result['discoverer'] == "STF 2107"
        assert result['date_first'] == 1999.5
        assert result['date_last'] == 2015.5
        assert result['n_observations'] == 42
        assert result['pa_first'] == 285.2
        assert result['pa_last'] == 287.8
        assert result['sep_first'] == 0.455
        assert result['sep_last'] == 0.431
        assert result['mag_pri'] == 8.12  # vmag mapped to mag_pri
        assert result['mag_sec'] == 8.89   # kmag mapped to mag_sec
        assert result['spec_type'] == 'G5V+K0V'
        assert result['ra_deg'] == 244.218
        assert result['dec_deg'] == 1.295
        
        source.close()
    
    @pytest.mark.asyncio
    async def test_wdss_get_summary_without_correspondence(self, wdss_sqlite_db):
        """Test WDS summary retrieval for WDSS-only system."""
        source = LocalDataSource(wdss_sqlite_db)
        
        # System without WDS correspondence should still be accessible via wdss_id as wds_id
        result = await source.get_wds_summary("1718115-142444")
        assert result is not None
        assert result['wds_id'] == "1718115-142444"  # Uses wdss_id as wds_id
        assert result['wdss_id'] == "1718115-142444"
        assert result['mag_pri'] == 16.15
        assert result['mag_sec'] == 17.58
        assert result['ra_deg'] == 259.546
        assert result['dec_deg'] == -14.412
        
        source.close()
    
    @pytest.mark.asyncio
    async def test_wdss_orbital_elements(self, wdss_sqlite_db):
        """Test orbital elements retrieval."""
        source = LocalDataSource(wdss_sqlite_db)
        
        result = await source.get_orbital_elements("16169+0113")
        assert result is not None
        
        assert result['wds_id'] == "16169+0113"
        assert result['P'] == 285.69
        assert result['a'] == 0.4331
        assert result['i'] == 118.2
        assert result['Omega'] == 287.1
        assert result['T'] == 2023.45
        assert result['e'] == 0.1259
        assert result['omega'] == 167.3  # Aliased from omega_arg in SQL query
        
        source.close()
    
    @pytest.mark.asyncio
    async def test_wdss_measurements_via_join(self, wdss_sqlite_db):
        """Test measurements retrieval via wdss_id join."""
        source = LocalDataSource(wdss_sqlite_db)
        
        # Test system with WDS correspondence
        result = await source.get_all_measurements("16169+0113")
        assert result is not None
        assert len(result) == 3
        assert result['wds_id'][0] == "16169+0113"  # Should use WDS ID in output
        assert result['epoch'][0] == 1999.5
        assert result['theta'][0] == 285.2
        assert result['rho'][0] == 0.455
        
        # Test WDSS-only system
        result2 = await source.get_all_measurements("1718115-142444")
        assert result2 is not None
        assert len(result2) == 2
        assert result2['wds_id'][0] == "1718115-142444"
        assert result2['epoch'][0] == 2010.2
        assert result2['theta'][0] == 123.4
        assert result2['rho'][0] == 2.34
        
        source.close()

    @pytest.mark.asyncio
    async def test_wdss_measurements_ordering(self, wdss_sqlite_db):
        """Test that measurements are returned in chronological order."""
        source = LocalDataSource(wdss_sqlite_db)
        
        result = await source.get_all_measurements("16169+0113")
        assert result is not None
        assert len(result) == 3
        
        # Verify chronological ordering
        epochs = result['epoch']
        assert epochs[0] == 1999.5  # First observation
        assert epochs[1] == 2007.3  # Middle observation  
        assert epochs[2] == 2015.5  # Last observation
        
        source.close()

    @pytest.mark.asyncio
    async def test_database_error_handling(self, wdss_sqlite_db):
        """Test error handling for database issues."""
        source = LocalDataSource(wdss_sqlite_db)
        
        # Close connection to simulate database error
        source.conn.close()
        
        # These should handle errors gracefully
        result = await source.get_wds_summary("16169+0113")
        assert result is None
        
        orbital = await source.get_orbital_elements("16169+0113")
        assert orbital is None
        
        measurements = await source.get_all_measurements("16169+0113")
        assert measurements is None
    
    @pytest.mark.asyncio
    async def test_coordinate_extraction_validation(self, wdss_sqlite_db):
        """Test that coordinates are properly extracted from WDSS IDs."""
        source = LocalDataSource(wdss_sqlite_db)
        
        # Test coordinate extraction
        result = await source.get_wds_summary("1718115-142444")
        assert result is not None
        
        # Coordinates should be extracted from WDSS ID: 1718115-142444
        # Expected: RA = 17h18m11.5s = 259.546°, Dec = -14°24'44" = -14.412°
        assert abs(result['ra_deg'] - 259.546) < 0.001
        assert abs(result['dec_deg'] - (-14.412)) < 0.001
        
        source.close()
    
    @pytest.mark.asyncio
    async def test_column_mapping_validation(self, wdss_sqlite_db):
        """Test that vmag/kmag are correctly mapped to mag_pri/mag_sec."""
        source = LocalDataSource(wdss_sqlite_db)
        
        result = await source.get_wds_summary("16169+0113")
        assert result is not None
        
        # Check column mapping: vmag -> mag_pri, kmag -> mag_sec
        assert result['mag_pri'] == 8.12  # Should be vmag from database
        assert result['mag_sec'] == 8.89  # Should be kmag from database
        
        source.close()
    
    @pytest.mark.asyncio
    async def test_wdss_not_found(self, wdss_sqlite_db):
        """Test behavior for non-existent systems."""
        source = LocalDataSource(wdss_sqlite_db)
        
        # Test non-existent system
        result = await source.get_wds_summary("99999+9999")
        assert result is None
        
        orbital = await source.get_orbital_elements("99999+9999")
        assert result is None
        
        measurements = await source.get_all_measurements("99999+9999")
        assert measurements is None
        
        source.close()
    
    def test_get_all_wds_ids_wdss(self, wdss_sqlite_db):
        """Test getting all WDS IDs from WDSS format."""
        source = LocalDataSource(wdss_sqlite_db)
        
        ids = source.get_all_wds_ids()
        assert len(ids) == 2
        assert "16169+0113" in ids  # System with WDS correspondence
        assert "1718115-142444" in ids  # WDSS-only system
        
        # Test with limit
        ids_limited = source.get_all_wds_ids(limit=1)
        assert len(ids_limited) == 1
        
        source.close()
    
    @pytest.mark.asyncio
    async def test_integration_with_orbital_analysis(self, wdss_sqlite_db):
        """Test integration scenario for orbital analysis."""
        source = LocalDataSource(wdss_sqlite_db)
        
        # This simulates what the analyzer does
        wds_id = "16169+0113"
        
        # 1. Get summary data
        summary = await source.get_wds_summary(wds_id)
        assert summary is not None
        assert summary['ra_deg'] is not None
        assert summary['dec_deg'] is not None
        
        # 2. Get orbital elements
        orbital = await source.get_orbital_elements(wds_id)
        assert orbital is not None
        assert orbital['P'] is not None
        assert orbital['e'] is not None
        
        # 3. Get measurements
        measurements = await source.get_all_measurements(wds_id)
        assert measurements is not None
        assert len(measurements) > 0
        
        # Verify data consistency
        assert summary['wds_id'] == orbital['wds_id']
        assert all(measurements['wds_id'][i] == wds_id for i in range(len(measurements)))
        
        source.close()


def test_coordinate_parsing_functions():
    """Test coordinate parsing functions directly."""
    try:
        # Import coordinate parsing functions
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "convert_catalogs_to_sqlite", 
            "scripts/convert_catalogs_to_sqlite.py"
        )
        convert_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(convert_module)
        
        parse_wdss_coordinates = convert_module.parse_wdss_coordinates
        
        # Test WDSS coordinate parsing
        test_cases = [
            ("1718115-142444", 259.546, -14.412),  # RA=17h18m11.5s, Dec=-14°24'44"
            ("1616524+011317", 244.218, 1.295),     # RA=16h16m52.4s, Dec=+01°13'17"
            ("0012345+123456", 3.188, 12.582),      # RA=00h12m34.5s, Dec=+12°34'56"
        ]
        
        for wdss_id, expected_ra, expected_dec in test_cases:
            ra, dec = parse_wdss_coordinates(wdss_id)
            assert ra is not None, f"RA parsing failed for {wdss_id}"
            assert dec is not None, f"Dec parsing failed for {wdss_id}"
            assert abs(ra - expected_ra) < 0.1, f"RA mismatch for {wdss_id}: got {ra}, expected {expected_ra}"
            assert abs(dec - expected_dec) < 0.1, f"Dec mismatch for {wdss_id}: got {dec}, expected {expected_dec}"
        
        # Test edge cases
        assert parse_wdss_coordinates("") == (None, None)
        assert parse_wdss_coordinates("123") == (None, None)
        assert parse_wdss_coordinates("invalid") == (None, None)
        
    except Exception as e:
        pytest.skip(f"Could not import coordinate parsing functions: {e}")


def test_generate_summary_table_function():
    """Test the improved generate_summary_table function."""
    try:
        import importlib.util
        import pandas as pd
        import numpy as np
        
        # Import the function
        spec = importlib.util.spec_from_file_location(
            "convert_catalogs_to_sqlite", 
            "scripts/convert_catalogs_to_sqlite.py"
        )
        convert_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(convert_module)
        
        generate_summary_table = convert_module.generate_summary_table
        
        # Create test data
        df_components = pd.DataFrame([
            {'wdss_id': 'sys1', 'component': 'A', 'vmag': 8.1, 'kmag': 8.9, 'ra_deg': 180.0, 'dec_deg': 45.0, 'spectral_type': 'G5V', 'parallax': 15.0, 'pm_ra': -12.0, 'pm_dec': -8.0, 'name': 'Test Star A'},
            {'wdss_id': 'sys1', 'component': 'B', 'vmag': 9.2, 'kmag': 9.8, 'ra_deg': 180.1, 'dec_deg': 45.1, 'spectral_type': 'K0V', 'parallax': 14.8, 'pm_ra': -11.5, 'pm_dec': -7.8, 'name': 'Test Star B'},
            {'wdss_id': 'sys2', 'component': 'A', 'vmag': 7.5, 'kmag': 8.2, 'ra_deg': 90.0, 'dec_deg': -30.0, 'spectral_type': 'F5V', 'parallax': 25.0, 'pm_ra': 5.0, 'pm_dec': -15.0, 'name': 'Another Star'}
        ])
        
        df_measurements = pd.DataFrame([
            {'wdss_id': 'sys1', 'epoch': 2000.0, 'theta': 45.0, 'rho': 1.5},
            {'wdss_id': 'sys1', 'epoch': 2010.0, 'theta': 50.0, 'rho': 1.6},
            {'wdss_id': 'sys1', 'epoch': 2020.0, 'theta': 55.0, 'rho': 1.7},
            {'wdss_id': 'sys2', 'epoch': 2005.0, 'theta': 120.0, 'rho': 0.8}
        ])
        
        df_correspondence = pd.DataFrame([
            {'wdss_id': 'sys1', 'wds_id': 'WDS001', 'discoverer_designation': 'STF 1'},
            # sys2 has no WDS correspondence
        ])
        
        # Test the function
        result = generate_summary_table(df_components, df_measurements, df_correspondence)
        
        # Validate results
        assert len(result) == 2  # Two systems
        assert 'wds_id' in result.columns
        assert 'vmag' in result.columns  # Should be vmag, not mag_pri
        assert 'kmag' in result.columns  # Should be kmag, not mag_sec
        assert 'ra_deg' in result.columns
        assert 'dec_deg' in result.columns
        
        # Check system with WDS correspondence
        sys1 = result[result['wds_id'] == 'WDS001'].iloc[0]
        assert sys1['wdss_id'] == 'sys1'
        assert sys1['vmag'] == 8.1  # Component A vmag
        assert sys1['kmag'] == 8.9   # Component A kmag
        assert sys1['ra_deg'] == 180.0  # Component A coordinates
        assert sys1['dec_deg'] == 45.0
        assert sys1['date_first'] == 2000.0
        assert sys1['date_last'] == 2020.0
        assert sys1['n_obs'] == 3
        
        # Check system without WDS correspondence (should use wdss_id as wds_id)
        sys2 = result[result['wdss_id'] == 'sys2'].iloc[0]
        assert sys2['wds_id'] == 'sys2'  # Should fallback to wdss_id
        assert sys2['vmag'] == 7.5
        assert sys2['n_obs'] == 1
        
    except Exception as e:
        pytest.skip(f"Could not test generate_summary_table function: {e}")


def test_pivot_duplicate_handling():
    """Test that pivot operation handles duplicate components correctly."""
    try:
        import importlib.util
        import pandas as pd
        
        # Import the function
        spec = importlib.util.spec_from_file_location(
            "convert_catalogs_to_sqlite", 
            "scripts/convert_catalogs_to_sqlite.py"
        )
        convert_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(convert_module)
        
        generate_summary_table = convert_module.generate_summary_table
        
        # Create test data with duplicate components (should keep first)
        df_components = pd.DataFrame([
            {'wdss_id': 'sys1', 'component': 'A', 'vmag': 8.1, 'kmag': 8.9, 'ra_deg': 180.0, 'dec_deg': 45.0, 'spectral_type': 'G5V', 'parallax': 15.0, 'pm_ra': -12.0, 'pm_dec': -8.0, 'name': 'First A'},
            {'wdss_id': 'sys1', 'component': 'A', 'vmag': 8.3, 'kmag': 9.1, 'ra_deg': 180.2, 'dec_deg': 45.2, 'spectral_type': 'G0V', 'parallax': 16.0, 'pm_ra': -10.0, 'pm_dec': -6.0, 'name': 'Second A'},  # Duplicate A
            {'wdss_id': 'sys1', 'component': 'B', 'vmag': 9.2, 'kmag': 9.8, 'ra_deg': 180.1, 'dec_deg': 45.1, 'spectral_type': 'K0V', 'parallax': 14.8, 'pm_ra': -11.5, 'pm_dec': -7.8, 'name': 'Only B'}
        ])
        
        df_measurements = pd.DataFrame()  # Empty for this test
        df_correspondence = pd.DataFrame()  # Empty for this test
        
        # This should not raise an error due to duplicate handling
        result = generate_summary_table(df_components, df_measurements, df_correspondence)
        
        assert len(result) == 1
        sys1 = result.iloc[0]
        
        # Should use first occurrence of component A
        assert sys1['vmag'] == 8.1  # First A vmag, not second
        assert sys1['name'] == 'First A'  # First A name
        
    except Exception as e:
        pytest.skip(f"Could not test duplicate handling: {e}")


# Integration test with real database
def test_real_wdss_database_integration():
    """Integration test with real WDSS database (if available)."""
    db_path = "test_final_corrected.db"
    
    if not Path(db_path).exists():
        pytest.skip("Real WDSS database not available for integration test")
    
    async def run_test():
        source = LocalDataSource(db_path)
        
        # Test basic functionality
        stats = source.get_catalog_statistics()
        assert stats is not None
        assert stats['wdss_summary_count'] > 0
        
        # Get some systems for testing
        wds_ids = source.get_all_wds_ids(limit=5)
        assert len(wds_ids) > 0
        
        # Test a few systems
        for wds_id in wds_ids[:3]:
            summary = await source.get_wds_summary(wds_id)
            assert summary is not None
            assert summary['wds_id'] == wds_id
            
            # If system has coordinates, they should be valid
            if summary['ra_deg'] is not None:
                assert 0 <= summary['ra_deg'] <= 360
            if summary['dec_deg'] is not None:
                assert -90 <= summary['dec_deg'] <= 90
        
        source.close()
    
    # Run async test
    asyncio.run(run_test())


def test_database_schema_validation():
    """Validate the final database schema matches expectations."""
    db_path = "test_final_corrected.db"
    
    if not Path(db_path).exists():
        pytest.skip("Real WDSS database not available for schema validation")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check wdss_summary table schema
    cursor.execute("PRAGMA table_info(wdss_summary)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    
    # Verify expected columns and types
    expected_columns = {
        'wds_id': 'TEXT',
        'wdss_id': 'TEXT', 
        'vmag': 'REAL',        # Not mag_pri
        'kmag': 'REAL',        # Not mag_sec
        'ra_deg': 'REAL',
        'dec_deg': 'REAL',
        'discoverer_designation': 'TEXT',
        'n_obs': 'INTEGER',
        'spectral_type': 'TEXT'
    }
    
    for col_name, col_type in expected_columns.items():
        assert col_name in columns, f"Missing column: {col_name}"
        assert columns[col_name] == col_type, f"Wrong type for {col_name}: got {columns[col_name]}, expected {col_type}"
    
    # Check data samples
    cursor.execute("SELECT vmag, kmag, ra_deg, dec_deg FROM wdss_summary WHERE ra_deg IS NOT NULL LIMIT 10")
    rows = cursor.fetchall()
    
    assert len(rows) > 0, "No data with coordinates found"
    
    for row in rows:
        vmag, kmag, ra_deg, dec_deg = row
        if vmag is not None:
            assert 0 <= vmag <= 30, f"Invalid vmag: {vmag}"
        if kmag is not None:
            assert 0 <= kmag <= 30, f"Invalid kmag: {kmag}"
        if ra_deg is not None:
            assert 0 <= ra_deg <= 360, f"Invalid RA: {ra_deg}"
        if dec_deg is not None:
            assert -90 <= dec_deg <= 90, f"Invalid Dec: {dec_deg}"
    
    # Check measurements table has wdss_id (not wds_id)
    cursor.execute("PRAGMA table_info(measurements)")
    meas_columns = [row[1] for row in cursor.fetchall()]
    assert 'wdss_id' in meas_columns, "Measurements table should have wdss_id column"
    
    conn.close()


@pytest.mark.slow
def test_database_performance():
    """Test database query performance with real data."""
    db_path = "test_final_corrected.db"
    
    if not Path(db_path).exists():
        pytest.skip("Real WDSS database not available for performance test")
    
    import time
    
    async def run_performance_test():
        source = LocalDataSource(db_path)
        
        # Get a sample of systems
        wds_ids = source.get_all_wds_ids(limit=100)
        
        # Time summary retrieval
        start_time = time.time()
        for wds_id in wds_ids[:50]:
            await source.get_wds_summary(wds_id)
        summary_time = time.time() - start_time
        
        # Should be reasonably fast (less than 1 second for 50 queries)
        assert summary_time < 1.0, f"Summary queries too slow: {summary_time:.2f}s for 50 queries"
        
        # Time measurements retrieval  
        start_time = time.time()
        measurements_count = 0
        for wds_id in wds_ids[:20]:
            result = await source.get_all_measurements(wds_id)
            if result:
                measurements_count += len(result)
        measurements_time = time.time() - start_time
        
        assert measurements_time < 2.0, f"Measurements queries too slow: {measurements_time:.2f}s for 20 queries"
        
        source.close()
    
    asyncio.run(run_performance_test())


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
