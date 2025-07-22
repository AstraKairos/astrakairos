"""
Tests for LocalDataSource implementation - Fixed version.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path

from astrakairos.data.local_source import LocalDataSource
from astrakairos.data.source import WdsSummary, OrbitalElements


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
    
    # Cleanup
    try:
        Path(db_path).unlink()
    except FileNotFoundError:
        pass
    except PermissionError:
        print(f"Warning: Could not delete temporary file {db_path}")


class TestLocalDataSourceFixed:
    """Test cases for LocalDataSource."""
    
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
        # Check error columns
        assert result['e_P'] == 5.0
        assert result['e_a'] == 0.1
        assert result['e_e'] == 0.05
        
        source.close()
