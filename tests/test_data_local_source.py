"""
Tests for LocalDataSource implementation.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path

from astrakairos.data.local_source import LocalDataSource
from astrakairos.data.source import (WdsSummary, OrbitalElements, 
                                     AstraKairosDataError, WdsIdNotFoundError,
                                     MeasurementsUnavailableError, OrbitalElementsUnavailableError)


@pytest.fixture
def sample_sqlite_db():
    """Create a temporary SQLite database with test data."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    conn = sqlite3.connect(db_path)
    
    # Create WDSS summary table (matching real schema with error columns)
    conn.execute("""
CREATE TABLE wdss_summary (
    wds_id TEXT PRIMARY KEY,
    discoverer_designation TEXT,
    date_first REAL,
    date_last REAL,
    n_obs INTEGER,
    pa_first REAL,
    pa_last REAL,
    sep_first REAL,
    sep_last REAL,
    pa_first_error REAL,
    pa_last_error REAL,
    sep_first_error REAL,
    sep_last_error REAL,
    vmag REAL,
    kmag REAL,
    spectral_type TEXT,
    ra_deg REAL,
    dec_deg REAL,
    wdss_id TEXT
)
    """)
    
    # Create orbital elements table
    conn.execute("""
CREATE TABLE orbital_elements (
    wds_id TEXT PRIMARY KEY,
    P REAL,
    e_P REAL,
    a REAL,
    e_a REAL,
    i REAL,
    e_i REAL,
    Omega REAL,
    e_Omega REAL,
    T REAL,
    e_T REAL,
    e REAL,
    e_e REAL,
    omega_arg REAL,
    e_omega_arg REAL,
    grade INTEGER
)
    """)
    
    # Create measurements table (using wdss_id to match local_source.py)
    conn.execute("""
CREATE TABLE measurements (
    wdss_id TEXT,
    epoch REAL,
    theta REAL,
    rho REAL
)
    """)
    
    # Insert test data (with error columns)
    conn.execute("""
INSERT INTO wdss_summary VALUES (
    '00001+0001', 'STF', 2000.0, 2020.0, 150,
    45.0, 50.0, 1.5, 1.6, 1.0, 1.0, 0.1, 0.1, 8.5, 9.2, 'G0V',
    0.25, 0.5, 'WDS00001'
)
    """)
    
    conn.execute("""
INSERT INTO orbital_elements VALUES (
    '00001+0001', 100.0, 5.0, 1.5, 0.1, 60.0, 2.0, 45.0, 3.0, 2010.0, 1.0, 0.3, 0.05, 90.0, 5.0, 1
)
    """)
    
    # Insert multiple measurements (using wdss_id)
    measurements = [
        ('WDS00001', 2000.0, 45.0, 1.5),
        ('WDS00001', 2010.0, 47.5, 1.55),
        ('WDS00001', 2020.0, 50.0, 1.6)
    ]
    conn.executemany("INSERT INTO measurements VALUES (?, ?, ?, ?)", measurements)
    
    # Create indexes
    conn.execute("CREATE INDEX idx_wdss_summary_id ON wdss_summary(wds_id)")
    conn.execute("CREATE INDEX idx_orbital_elements_id ON orbital_elements(wds_id)")
    conn.execute("CREATE INDEX idx_measurements_id ON measurements(wdss_id)")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup - use try/except for Windows file permission issues
    try:
        Path(db_path).unlink()
    except PermissionError:
        # On Windows, try again after a brief delay
        import time
        time.sleep(0.1)
        try:
            Path(db_path).unlink()
        except PermissionError:
            # If still fails, just log the issue
            print(f"Warning: Could not delete temporary file {db_path}")
class TestLocalDataSource:
    """Test LocalDataSource functionality."""
    
    def test_initialization_success(self, sample_sqlite_db):
        """Test successful initialization."""
        source = LocalDataSource(sample_sqlite_db)
        assert source.conn is not None
        assert source.has_measurements is True
        source.close()
    
    def test_initialization_file_not_found(self):
        """Test initialization with non-existent file."""
        with pytest.raises(AstraKairosDataError):
            LocalDataSource("nonexistent.db")
    
    def test_get_catalog_statistics(self, sample_sqlite_db):
        """Test catalog statistics retrieval."""
        source = LocalDataSource(sample_sqlite_db)
        
        stats = source.get_catalog_statistics()
        assert stats['wdss_summary_count'] == 1  # Use wdss_summary since that's the table name
        assert stats['orbital_elements_count'] == 1
        assert stats['measurements_count'] == 3
        assert stats['systems_with_measurements'] == 1
        
        source.close()
    
    @pytest.mark.asyncio
    async def test_get_wds_summary_success(self, sample_sqlite_db):
        """Test successful WDS summary retrieval."""
        source = LocalDataSource(sample_sqlite_db)
        
        result = await source.get_wds_summary("00001+0001")
        assert result is not None
        assert result['wds_id'] == "00001+0001"
        assert result['date_first'] == 2000.0
        assert result['date_last'] == 2020.0
        assert result['n_observations'] == 150
        assert result['pa_first'] == 45.0
        assert result['pa_last'] == 50.0
        assert result['sep_first'] == 1.5
        assert result['sep_last'] == 1.6
        assert result['mag_pri'] == 8.5
        assert result['mag_sec'] == 9.2
        assert result['ra_deg'] == 0.25
        assert result['dec_deg'] == 0.5
        
        source.close()
    
    @pytest.mark.asyncio
    async def test_get_wds_summary_not_found(self, sample_sqlite_db):
        """Test WDS summary retrieval for non-existent system."""
        source = LocalDataSource(sample_sqlite_db)
        
        with pytest.raises(WdsIdNotFoundError):
            await source.get_wds_summary("99999+9999")
        
        source.close()
    
    @pytest.mark.asyncio
    async def test_get_orbital_elements_success(self, sample_sqlite_db):
        """Test successful orbital elements retrieval."""
        source = LocalDataSource(sample_sqlite_db)
        
        result = await source.get_orbital_elements("00001+0001")
        assert result is not None
        assert result['wds_id'] == "00001+0001"
        assert result['P'] == 100.0
        assert result['a'] == 1.5
        assert result['e'] == 0.3
        
        source.close()
    
    @pytest.mark.asyncio
    async def test_get_orbital_elements_not_found(self, sample_sqlite_db):
        """Test orbital elements retrieval for non-existent system."""
        source = LocalDataSource(sample_sqlite_db)
        
        with pytest.raises(OrbitalElementsUnavailableError):
            await source.get_orbital_elements("99999+9999")
        
        source.close()
    
    @pytest.mark.asyncio
    async def test_get_all_measurements_success(self, sample_sqlite_db):
        """Test successful measurements retrieval."""
        source = LocalDataSource(sample_sqlite_db)
        
        result = await source.get_all_measurements("00001+0001")
        assert result is not None
        assert len(result) == 3
        assert result['epoch'][0] == 2000.0
        assert result['theta'][0] == 45.0
        assert result['rho'][0] == 1.5
        
        source.close()
    
    @pytest.mark.asyncio
    async def test_get_all_measurements_not_found(self, sample_sqlite_db):
        """Test measurements retrieval for non-existent system."""
        source = LocalDataSource(sample_sqlite_db)
        
        with pytest.raises(WdsIdNotFoundError):
            await source.get_all_measurements("99999+9999")
        
        source.close()
    
    @pytest.mark.asyncio
    async def test_validate_physicality_stub(self, sample_sqlite_db):
        """Test physicality validation returns unknown status."""
        source = LocalDataSource(sample_sqlite_db)
        
        wds_summary = await source.get_wds_summary("00001+0001")
        result = await source.validate_physicality(wds_summary)
        
        assert result is not None
        assert result['label'].value == 'Unknown'
        assert result['confidence'] == 0.0
        assert result['method'] is None
        
        source.close()
    
    def test_get_all_wds_ids(self, sample_sqlite_db):
        """Test getting all WDS IDs."""
        source = LocalDataSource(sample_sqlite_db)
        
        ids = source.get_all_wds_ids()
        assert len(ids) == 1
        assert ids[0] == "00001+0001"
        
        # Test with limit
        ids_limited = source.get_all_wds_ids(limit=1)
        assert len(ids_limited) == 1
        
        source.close()
    
    def test_close_and_cleanup(self, sample_sqlite_db):
        """Test proper resource cleanup."""
        source = LocalDataSource(sample_sqlite_db)
        
        # Should work before close
        stats = source.get_catalog_statistics()
        assert stats is not None
        
        # Close connection
        source.close()
        
        # Should raise exception after close
        with pytest.raises(AstraKairosDataError):
            source.get_catalog_statistics()
