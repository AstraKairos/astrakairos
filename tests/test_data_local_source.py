import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from astropy.table import Table

from astrakairos.data.local_source import LocalFileDataSource
from astrakairos.data.source import WdsSummary, OrbitalElements, PhysicalityAssessment

# Test data formatted according to official catalog specifications

# WDS Summary format based on wdsweb_format.txt
WDS_SUMM_CONTENT = """00002+0146WEI  AB      1879 2015   26  89  83   1.8   1.8 10.09 10.52 G0                                        000012.14+014617.2
00003-4417I  1477AB    1926 2022   33 261 183   0.5   0.2  6.80  7.56 G3IV                                      000019.10-441726.0
"""

# WDS Measurements format based on wdss_format.txt
WDS_MEASURES_CONTENT = """00002+0146              2010.123    85.500          1.75000  
00002+0146              2015.456    83.000          1.80000  
00003-4417              2020.500    180.100        m250.00000
"""

# ORB6 format based on orb6format.txt
ORB6_CONTENT = """                   00002+0146                                                    100.0y                  0.5a                 45.0              90.0               2000.0y                  0.5               30.0      
                   00003-4417                                                    36525.0d                250.0m               90.0              180.0              2010.0y                  0.1               60.0      
"""

@pytest.fixture
def mock_fs_and_source():
    """Mocks the filesystem and returns an instance of LocalFileDataSource."""
    mock_files = {
        "wds_summary.txt": WDS_SUMM_CONTENT,
        "wds_measures.txt": WDS_MEASURES_CONTENT,
        "orb6.txt": ORB6_CONTENT,
    }
    
    def mock_open_handler(filename, *args, **kwargs):
        for path, content in mock_files.items():
            if path in filename:
                return mock_open(read_data=content).return_value
        raise FileNotFoundError(f"File not found: {filename}")
    
    with patch("builtins.open", side_effect=mock_open_handler), \
         patch("os.path.exists", return_value=True):
        source = LocalFileDataSource(
            wds_filepath="wds_summary.txt",
            orb6_filepath="orb6.txt",
            wds_measures_filepath="wds_measures.txt"
        )
        yield source

@pytest.fixture
def mock_fs_and_source_no_measures():
    """Mocks the filesystem without measurements file."""
    mock_files = {
        "wds_summary.txt": WDS_SUMM_CONTENT,
        "orb6.txt": ORB6_CONTENT,
    }
    
    def mock_open_handler(filename, *args, **kwargs):
        for path, content in mock_files.items():
            if path in filename:
                return mock_open(read_data=content).return_value
        raise FileNotFoundError(f"File not found: {filename}")
    
    with patch("builtins.open", side_effect=mock_open_handler), \
         patch("os.path.exists", return_value=True):
        
        source = LocalFileDataSource(
            wds_filepath="wds_summary.txt",
            orb6_filepath="orb6.txt"
        )
        yield source

# Catalog Loading Tests

def test_init_success():
    """Test successful instantiation with valid files."""
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=WDS_SUMM_CONTENT)), \
         patch("astrakairos.data.local_source.LocalFileDataSource._load_wds_summary_catalog") as mock_wds, \
         patch("astrakairos.data.local_source.LocalFileDataSource._load_orb6_catalog") as mock_orb6, \
         patch("astrakairos.data.local_source.LocalFileDataSource._load_wds_measurements_catalog") as mock_measures:
        
        mock_wds.return_value = pd.DataFrame()
        mock_orb6.return_value = pd.DataFrame()
        mock_measures.return_value = pd.DataFrame()
        
        source = LocalFileDataSource(
            wds_filepath="wds_summary.txt",
            orb6_filepath="orb6.txt",
            wds_measures_filepath="wds_measures.txt"
        )
        
        assert source.wds_df is not None
        assert source.orb6_df is not None
        assert source.wds_measures_df is not None

def test_init_optional_measures():
    """Test instantiation without measurements file."""
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=WDS_SUMM_CONTENT)), \
         patch("astrakairos.data.local_source.LocalFileDataSource._load_wds_summary_catalog") as mock_wds, \
         patch("astrakairos.data.local_source.LocalFileDataSource._load_orb6_catalog") as mock_orb6:
        
        mock_wds.return_value = pd.DataFrame()
        mock_orb6.return_value = pd.DataFrame()
        
        source = LocalFileDataSource(
            wds_filepath="wds_summary.txt",
            orb6_filepath="orb6.txt"
        )
        
        assert source.wds_df is not None
        assert source.orb6_df is not None
        assert source.wds_measures_df is None

def test_init_file_not_found():
    """Test exception when required files don't exist."""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            LocalFileDataSource(
                wds_filepath="nonexistent.txt",
                orb6_filepath="orb6.txt"
            )

# WDS Summary Tests

@pytest.mark.asyncio
async def test_get_wds_summary_success(mock_fs_and_source):
    """Tests retrieving a valid WDS summary entry."""
    summary = await mock_fs_and_source.get_wds_summary("00002+0146")
    assert summary is not None
    # WdsSummary is a dict-like object, access using dictionary notation
    assert summary.get('date_last') == 2015
    assert summary.get('wds_name') == "00002+0146"


@pytest.mark.asyncio
async def test_get_wds_summary_coordinate_parsing(mock_fs_and_source):
    """Tests coordinate parsing from precise_coords_str."""
    summary = await mock_fs_and_source.get_wds_summary("00002+0146")
    assert summary is not None
    # Corrected expected values for "000012.14+014617.2"
    # RA: 00h 00m 12.14s = 12.14/3600 * 15 = 0.050583333 degrees
    # Dec: +01° 46' 17.2" = 1 + 46/60 + 17.2/3600 = 1.771444 degrees
    assert pytest.approx(summary.get('ra_deg'), abs=0.001) == 0.050583333
    assert pytest.approx(summary.get('dec_deg'), abs=0.001) == 1.771444

@pytest.mark.asyncio
async def test_get_wds_summary_not_found(mock_fs_and_source):
    """Tests retrieving a non-existent WDS summary entry."""
    summary = await mock_fs_and_source.get_wds_summary("99999+9999")
    assert summary is None

# Measurements Tests

@pytest.mark.asyncio
async def test_get_all_measurements_success(mock_fs_and_source):
    """Tests retrieving all measurements for a known star."""
    measurements = await mock_fs_and_source.get_all_measurements("00002+0146")
    assert isinstance(measurements, Table)
    assert len(measurements) == 2
    assert 'epoch' in measurements.colnames
    assert 'theta' in measurements.colnames
    assert 'rho' in measurements.colnames
    assert pytest.approx(measurements[0]['rho']) == 1.75

@pytest.mark.asyncio
async def test_get_all_measurements_unit_conversion(mock_fs_and_source):
    """Tests that milliarcsecond to arcsecond conversion for rho is correct."""
    measurements = await mock_fs_and_source.get_all_measurements("00003-4417")
    assert measurements is not None
    # File value is 250.0 with 'm' flag, should be converted to 0.250
    assert pytest.approx(measurements[0]['rho']) == 0.250

@pytest.mark.asyncio
async def test_get_all_measurements_not_found(mock_fs_and_source):
    """Tests retrieving measurements for a non-existent star."""
    measurements = await mock_fs_and_source.get_all_measurements("99999+9999")
    assert measurements is None

@pytest.mark.asyncio
async def test_get_all_measurements_no_catalog(mock_fs_and_source_no_measures):
    """Tests behavior when measurements catalog is not loaded."""
    measurements = await mock_fs_and_source_no_measures.get_all_measurements("00002+0146")
    assert measurements is None

# Orbital Elements Tests

@pytest.mark.asyncio
async def test_get_orbital_elements_success(mock_fs_and_source):
    """Tests retrieving orbital elements with standard units."""
    orbit = await mock_fs_and_source.get_orbital_elements("00002+0146")
    assert orbit is not None
    # Access orbital elements using dictionary notation
    assert pytest.approx(orbit.get('P')) == 100.0
    assert pytest.approx(orbit.get('a')) == 0.5

@pytest.mark.asyncio
async def test_get_orbital_elements_unit_conversion(mock_fs_and_source):
    """Tests unit conversion for orbital elements (days to years, mas to arcsec)."""
    orbit = await mock_fs_and_source.get_orbital_elements("00003-4417")
    assert orbit is not None
    # 36525.0d / 365.25 = 100.0 years, 250.0m / 1000 = 0.250 arcsec
    assert pytest.approx(orbit.get('P'), abs=0.1) == 100.0
    assert pytest.approx(orbit.get('a'), abs=0.01) == 0.250

@pytest.mark.asyncio
async def test_get_orbital_elements_not_found(mock_fs_and_source):
    """Tests retrieving orbital elements for a non-existent star."""
    orbit = await mock_fs_and_source.get_orbital_elements("99999+9999")
    assert orbit is None

# Physicality Validation Tests

@pytest.mark.asyncio
async def test_validate_physicality_stub(mock_fs_and_source):
    """Tests that validate_physicality returns expected stub response."""
    # Create a proper WdsSummary dict with required fields
    test_summary = {
        'wds_name': '00002+0146',
        'ra_deg': 0.0,
        'dec_deg': 0.0,
        'date_last': 2015
    }
    
    result = await mock_fs_and_source.validate_physicality(test_summary)
    assert result is not None
    assert result['label'] == 'Unknown'
    assert result['p_value'] is None
    assert 'Local Source' in result['test_used']

@pytest.mark.asyncio
async def test_validate_physicality_none_input(mock_fs_and_source):
    """Tests validate_physicality with None input."""
    result = await mock_fs_and_source.validate_physicality(None)
    assert result is None