import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from astropy.table import Table
from astroquery.exceptions import TimeoutError as AstroqueryTimeoutError

from astrakairos.data.online_source import OnlineDataSource, extract_float_value, extract_int_value, extract_string_value
from astrakairos.data.source import WdsSummary, OrbitalElements, PhysicalityAssessment, PhysicalityLabel, ValidationMethod


class TestOnlineDataSource:
    """Test suite for OnlineDataSource class."""

    def test_initialization_default(self):
        """Test OnlineDataSource initialization with default parameters."""
        source = OnlineDataSource()
        assert source.vizier_row_limit == 100  # DEFAULT_VIZIER_ROW_LIMIT
        assert source.vizier_timeout == 30    # DEFAULT_VIZIER_TIMEOUT

    def test_initialization_custom(self):
        """Test OnlineDataSource initialization with custom parameters."""
        source = OnlineDataSource(vizier_row_limit=100, vizier_timeout=120)
        assert source.vizier_row_limit == 100
        assert source.vizier_timeout == 120

    @pytest.mark.asyncio
    async def test_get_wds_summary_success(self):
        """Test successful WDS summary retrieval."""
        # Create mock data that resembles VizieR response
        mock_row = {
            'WDS': '00013+1944',
            'Name': 'STF 3',
            'RAJ2000': 0.875,
            'DEJ2000': 19.75,
            'Obs1': 1825.0,
            'Obs2': 2020.0,
            'Nobs': 150,
            'pa1': 120.5,
            'pa2': 121.0,
            'sep1': 22.1,
            'sep2': 22.0,
            'mag1': 6.5,
            'mag2': 7.2,
            'SpType': 'G0V'
        }
        
        mock_table = Mock()
        mock_table.__len__ = Mock(return_value=1)
        mock_table.__getitem__ = Mock(return_value=mock_row)
        
        mock_result = [mock_table]
        
        source = OnlineDataSource()
        
        with patch.object(source, '_retry_vizier_query', return_value=mock_result):
            result = await source.get_wds_summary("00013+1944")
            
            assert result is not None
            assert result['wds_id'] == "00013+1944"
            assert result['ra_deg'] == 0.875
            assert result['dec_deg'] == 19.75
            assert result['date_first'] == 1825.0
            assert result['date_last'] == 2020.0
            assert result['obs'] == 150
            assert result['pa_first'] == 120.5
            assert result['pa_last'] == 121.0
            assert result['sep_first'] == 22.1
            assert result['sep_last'] == 22.0
            assert result['mag_pri'] == 6.5
            assert result['mag_sec'] == 7.2
            assert result['spec_type'] == "G0V"

    @pytest.mark.asyncio
    async def test_get_wds_summary_not_found(self):
        """Test WDS summary when no data is found."""
        source = OnlineDataSource()
        
        with patch.object(source, '_retry_vizier_query', return_value=None):
            result = await source.get_wds_summary("99999+9999")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_wds_summary_empty_result(self):
        """Test WDS summary when VizieR returns empty result."""
        mock_table = Mock()
        mock_table.__len__ = Mock(return_value=0)
        mock_result = [mock_table]
        
        source = OnlineDataSource()
        
        with patch.object(source, '_retry_vizier_query', return_value=mock_result):
            result = await source.get_wds_summary("00013+1944")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_wds_summary_invalid_wds_id(self):
        """Test WDS summary with invalid WDS ID format."""
        mock_row = {'WDS': 'INVALID_FORMAT'}
        mock_table = Mock()
        mock_table.__len__ = Mock(return_value=1)
        mock_table.__getitem__ = Mock(return_value=mock_row)
        mock_result = [mock_table]
        
        source = OnlineDataSource()
        
        with patch.object(source, '_retry_vizier_query', return_value=mock_result):
            result = await source.get_wds_summary("00013+1944")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_wds_summary_data_conversion_error(self):
        """Test WDS summary when data conversion fails."""
        mock_row = {
            'WDS': '00013+1944',
            'RAJ2000': 'invalid_data'  # This will cause conversion error
        }
        mock_table = Mock()
        mock_table.__len__ = Mock(return_value=1)
        mock_table.__getitem__ = Mock(side_effect=Exception("Data conversion error"))
        mock_result = [mock_table]
        
        source = OnlineDataSource()
        
        with patch.object(source, '_retry_vizier_query', return_value=mock_result):
            result = await source.get_wds_summary("00013+1944")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_orbital_elements_success(self):
        """Test successful orbital elements retrieval."""
        mock_row = {
            'WDS': '00013+1944',
            'P': 50.0,
            'e_P': 1.5,
            'Axis': 1.2,
            'e_Axis': 0.05,
            'i': 65.0,
            'Node': 120.0,
            'T': 2020.5,
            'e': 0.15,
            'omega': 45.0
        }
        
        mock_table = Mock()
        mock_table.__len__ = Mock(return_value=1)
        mock_table.__getitem__ = Mock(return_value=mock_row)
        mock_result = [mock_table]
        
        source = OnlineDataSource()
        
        with patch.object(source, '_retry_vizier_query', return_value=mock_result):
            result = await source.get_orbital_elements("00013+1944")
            
            assert result is not None
            assert result['wds_id'] == "00013+1944"
            assert result['P'] == 50.0
            assert result['a'] == 1.2
            assert result['i'] == 65.0
            assert result['Omega'] == 120.0
            assert result['T'] == 2020.5
            assert result['e'] == 0.15
            assert result['omega'] == 45.0
            assert result['e_P'] == 1.5
            assert result['e_a'] == 0.05

    @pytest.mark.asyncio
    async def test_get_orbital_elements_not_found(self):
        """Test orbital elements when no data is found."""
        source = OnlineDataSource()
        
        with patch.object(source, '_retry_vizier_query', return_value=None):
            result = await source.get_orbital_elements("99999+9999")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_orbital_elements_invalid_period(self):
        """Test orbital elements with invalid period."""
        mock_row = {
            'WDS': '00013+1944',
            'P': np.ma.masked,  # Masked period
            'Axis': 1.2
        }
        
        mock_table = Mock()
        mock_table.__len__ = Mock(return_value=1)
        mock_table.__getitem__ = Mock(return_value=mock_row)
        mock_result = [mock_table]
        
        source = OnlineDataSource()
        
        with patch.object(source, '_retry_vizier_query', return_value=mock_result):
            result = await source.get_orbital_elements("00013+1944")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_orbital_elements_invalid_axis(self):
        """Test orbital elements with invalid semimajor axis."""
        mock_row = {
            'WDS': '00013+1944',
            'P': 50.0,
            'Axis': -1.0  # Invalid negative axis
        }
        
        mock_table = Mock()
        mock_table.__len__ = Mock(return_value=1)
        mock_table.__getitem__ = Mock(return_value=mock_row)
        mock_result = [mock_table]
        
        source = OnlineDataSource()
        
        with patch.object(source, '_retry_vizier_query', return_value=mock_result):
            result = await source.get_orbital_elements("00013+1944")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_all_measurements(self):
        """Test get_all_measurements always returns None."""
        source = OnlineDataSource()
        result = await source.get_all_measurements("00013+1944")
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_physicality(self):
        """Test physicality validation returns unknown assessment."""
        source = OnlineDataSource()
        mock_summary = Mock(spec=WdsSummary)
        
        result = await source.validate_physicality(mock_summary)
        
        assert result is not None
        assert result['label'] == PhysicalityLabel.UNKNOWN
        assert result['p_value'] is None
        assert result['method'] == ValidationMethod.INSUFFICIENT_DATA

    @pytest.mark.asyncio
    async def test_retry_vizier_query_success(self):
        """Test successful VizieR query on first attempt."""
        source = OnlineDataSource()
        mock_query_func = AsyncMock(return_value="success")
        
        with patch('astrakairos.data.online_source.asyncio.to_thread', return_value="success"):
            result = await source._retry_vizier_query(mock_query_func, arg1="test")
            assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_vizier_query_timeout_then_success(self):
        """Test VizieR query that times out then succeeds."""
        source = OnlineDataSource()
        mock_query_func = Mock()
        
        call_count = 0
        async def mock_to_thread(func, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise AstroqueryTimeoutError("Timeout")
            return "success"
        
        with patch('astrakairos.data.online_source.asyncio.to_thread', side_effect=mock_to_thread):
            with patch('astrakairos.data.online_source.asyncio.sleep', return_value=None):
                result = await source._retry_vizier_query(mock_query_func, arg1="test")
                assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_vizier_query_all_attempts_fail(self):
        """Test VizieR query that fails all retry attempts."""
        source = OnlineDataSource()
        mock_query_func = Mock()
        
        with patch('astrakairos.data.online_source.asyncio.to_thread', side_effect=Exception("Network error")):
            with patch('astrakairos.data.online_source.asyncio.sleep', return_value=None):
                result = await source._retry_vizier_query(mock_query_func, arg1="test")
                assert result is None


class TestHelperFunctions:
    """Test suite for helper functions."""

    def test_extract_float_value_valid(self):
        """Test extracting valid float value."""
        assert extract_float_value(3.14) == 3.14
        assert extract_float_value("2.718") == 2.718
        assert extract_float_value(42) == 42.0

    def test_extract_float_value_masked(self):
        """Test extracting masked value."""
        masked_value = np.ma.masked
        assert extract_float_value(masked_value) is None

    def test_extract_float_value_invalid(self):
        """Test extracting invalid float value."""
        assert extract_float_value("invalid") is None
        assert extract_float_value(None) is None
        assert extract_float_value([1, 2, 3]) is None

    def test_extract_int_value_valid(self):
        """Test extracting valid int value."""
        assert extract_int_value(42) == 42
        assert extract_int_value("123") == 123
        assert extract_int_value(3.14) == 3

    def test_extract_int_value_masked(self):
        """Test extracting masked int value."""
        masked_value = np.ma.masked
        assert extract_int_value(masked_value) is None

    def test_extract_int_value_invalid(self):
        """Test extracting invalid int value."""
        assert extract_int_value("invalid") is None
        assert extract_int_value(None) is None

    def test_extract_string_value_valid(self):
        """Test extracting valid string value."""
        assert extract_string_value("test") == "test"
        assert extract_string_value("  spaced  ") == "spaced"
        assert extract_string_value(123) == "123"

    def test_extract_string_value_masked(self):
        """Test extracting masked string value."""
        masked_value = np.ma.masked
        assert extract_string_value(masked_value) is None

    def test_extract_string_value_invalid(self):
        """Test extracting invalid string value."""
        # Most values can be converted to string, so we need extreme cases
        assert extract_string_value(None) == "None"  # None converts to "None"


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_wds_summary_with_masked_data(self):
        """Test WDS summary extraction with some masked data fields."""
        mock_row = {
            'WDS': '00013+1944',
            'Name': 'STF 3',
            'RAJ2000': 0.875,
            'DEJ2000': 19.75,
            'Obs1': np.ma.masked,  # Masked observation date
            'Obs2': 2020.0,
            'Nobs': 150,
            'pa1': np.ma.masked,   # Masked position angle
            'pa2': 121.0,
            'sep1': 22.1,
            'sep2': 22.0,
            'mag1': np.ma.masked,  # Masked magnitude
            'mag2': 7.2,
            'SpType': np.ma.masked  # Masked spectral type
        }
        
        mock_table = Mock()
        mock_table.__len__ = Mock(return_value=1)
        mock_table.__getitem__ = Mock(return_value=mock_row)
        mock_result = [mock_table]
        
        source = OnlineDataSource()
        
        with patch.object(source, '_retry_vizier_query', return_value=mock_result):
            result = await source.get_wds_summary("00013+1944")
            
            assert result is not None
            assert result['wds_id'] == "00013+1944"
            assert result['ra_deg'] == 0.875
            assert result['dec_deg'] == 19.75
            assert result['date_first'] is None  # Masked
            assert result['date_last'] == 2020.0
            assert result['obs'] == 150
            assert result['pa_first'] is None    # Masked
            assert result['pa_last'] == 121.0
            assert result['mag_pri'] is None     # Masked
            assert result['mag_sec'] == 7.2
            assert result['spec_type'] is None   # Masked

    @pytest.mark.asyncio
    async def test_orbital_elements_with_partial_data(self):
        """Test orbital elements extraction with some missing optional data."""
        mock_row = {
            'WDS': '00013+1944',
            'P': 50.0,
            'e_P': np.ma.masked,  # No uncertainty
            'Axis': 1.2,
            'e_Axis': 0.05,
            'i': np.ma.masked,    # No inclination
            'Node': 120.0,
            'T': 2020.5,
            'e': 0.15,
            'omega': np.ma.masked  # No omega
        }
        
        mock_table = Mock()
        mock_table.__len__ = Mock(return_value=1)
        mock_table.__getitem__ = Mock(return_value=mock_row)
        mock_result = [mock_table]
        
        source = OnlineDataSource()
        
        with patch.object(source, '_retry_vizier_query', return_value=mock_result):
            result = await source.get_orbital_elements("00013+1944")
            
            assert result is not None
            assert result['wds_id'] == "00013+1944"
            assert result['P'] == 50.0
            assert result['a'] == 1.2
            assert result['e_P'] is None      # Masked uncertainty
            assert result['e_a'] == 0.05
            assert result['i'] is None       # Masked inclination
            assert result['Omega'] == 120.0
            assert result['T'] == 2020.5
            assert result['e'] == 0.15
            assert result['omega'] is None    # Masked omega
