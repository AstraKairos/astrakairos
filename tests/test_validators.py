"""
Tests for the validators module.
"""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock
from datetime import datetime

from astrakairos.data.validators import HybridValidator
from astrakairos.data.source import (
    PhysicalityAssessment, PhysicalityLabel, ValidationMethod,
    InvalidInputError, CacheStatsError
)
from astrakairos.data.local_source import LocalDataSource
from astrakairos.data.gaia_source import GaiaValidator


@pytest.fixture
def mock_local_source():
    """Mock LocalDataSource for testing."""
    source = Mock(spec=LocalDataSource)
    source.summary_table = 'wdss_summary'
    source.conn = Mock()
    return source


@pytest.fixture
def mock_gaia_validator():
    """Mock GaiaValidator for testing."""
    return Mock(spec=GaiaValidator)


@pytest.fixture
def mock_physicality_assessment():
    """Mock PhysicalityAssessment for testing."""
    return {
        'label': PhysicalityLabel.LIKELY_PHYSICAL,
        'confidence': 0.95,
        'p_value': 0.001,
        'method': ValidationMethod.GAIA_3D_PARALLAX_PM,
        'parallax_consistency': 0.95,
        'proper_motion_consistency': 0.90,
        'gaia_source_id_primary': "123456789",
        'gaia_source_id_secondary': "987654321",
        'validation_date': datetime.now().isoformat(),
        'search_radius_arcsec': 5.0,
        'significance_thresholds': {'physical': 0.01, 'ambiguous': 0.001},
        'retry_attempts': 0
    }


@pytest.fixture
def sample_system_data():
    """Sample system data for testing."""
    return {
        'wds_id': '00001+0001',
        'ra_deg': 15.0,
        'dec_deg': 45.0,
        'mag_pri': 8.5,
        'mag_sec': 9.2
    }


class TestHybridValidator:
    """Test cases for HybridValidator class."""

    def test_init(self, mock_local_source, mock_gaia_validator):
        """Test HybridValidator initialization."""
        validator = HybridValidator(mock_local_source, mock_gaia_validator)
        
        assert validator.data_source == mock_local_source
        assert validator.online_validator == mock_gaia_validator

    def test_init_without_online_validator(self, mock_local_source):
        """Test HybridValidator initialization without online validator."""
        validator = HybridValidator(mock_local_source)
        
        assert validator.data_source == mock_local_source
        assert validator.online_validator is None

    @pytest.mark.asyncio
    async def test_validate_physicality_missing_wds_id(self, mock_local_source):
        """Test that missing wds_id raises InvalidInputError."""
        validator = HybridValidator(mock_local_source)
        
        with pytest.raises(InvalidInputError, match="Missing required field 'wds_id'"):
            await validator.validate_physicality({})

    @pytest.mark.asyncio
    async def test_validate_physicality_local_cache_hit(
        self, mock_local_source, mock_physicality_assessment, sample_system_data
    ):
        """Test validation when result is found in local cache."""
        mock_local_source.get_precomputed_physicality = AsyncMock(
            return_value=mock_physicality_assessment
        )
        
        validator = HybridValidator(mock_local_source)
        result = await validator.validate_physicality(sample_system_data)
        
        assert result == mock_physicality_assessment
        mock_local_source.get_precomputed_physicality.assert_called_once_with('00001+0001', None)

    @pytest.mark.asyncio
    async def test_validate_physicality_online_fallback(
        self, mock_local_source, mock_gaia_validator, mock_physicality_assessment, sample_system_data
    ):
        """Test validation fallback to online validator."""
        mock_local_source.get_precomputed_physicality = AsyncMock(return_value=None)
        mock_gaia_validator.validate_physicality = AsyncMock(
            return_value=mock_physicality_assessment
        )
        
        validator = HybridValidator(mock_local_source, mock_gaia_validator)
        result = await validator.validate_physicality(sample_system_data)
        
        assert result == mock_physicality_assessment
        mock_local_source.get_precomputed_physicality.assert_called_once_with('00001+0001', None)
        mock_gaia_validator.validate_physicality.assert_called_once_with(sample_system_data)

    @pytest.mark.asyncio
    async def test_validate_physicality_no_validator_insufficient_data(
        self, mock_local_source, sample_system_data
    ):
        """Test validation when no online validator is available."""
        mock_local_source.get_precomputed_physicality = AsyncMock(return_value=None)
        
        validator = HybridValidator(mock_local_source)
        result = await validator.validate_physicality(sample_system_data)
        
        assert result['label'] == PhysicalityLabel.INSUFFICIENT_DATA
        assert result['confidence'] == 0.0
        assert result['p_value'] is None
        assert result['method'] is None

    def test_get_cache_statistics_success(self, mock_local_source):
        """Test successful cache statistics retrieval."""
        mock_stats = {'wdss_summary_count': 1000}
        mock_local_source.get_catalog_statistics.return_value = mock_stats
        
        # Mock database cursor
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = [250]
        mock_local_source.conn.execute.return_value = mock_cursor
        
        validator = HybridValidator(mock_local_source)
        result = validator.get_cache_statistics()
        
        assert result['total_systems'] == 1000
        assert result['cached_systems'] == 250
        assert result['cache_coverage_percent'] == 25.0
        assert result['has_online_fallback'] is False
        assert 'El-Badry et al. (2021)' in result['cache_type']

    def test_get_cache_statistics_no_stats(self, mock_local_source):
        """Test cache statistics when no stats are available."""
        mock_local_source.get_catalog_statistics.return_value = None
        
        validator = HybridValidator(mock_local_source)
        
        with pytest.raises(CacheStatsError, match="Unable to retrieve catalog statistics"):
            validator.get_cache_statistics()

    def test_get_cache_statistics_database_error(self, mock_local_source):
        """Test cache statistics when database query fails."""
        mock_stats = {'wdss_summary_count': 1000}
        mock_local_source.get_catalog_statistics.return_value = mock_stats
        mock_local_source.conn.execute.side_effect = Exception("Database error")
        
        validator = HybridValidator(mock_local_source)
        result = validator.get_cache_statistics()
        
        assert result['total_systems'] == 1000
        assert result['cached_systems'] == -1  # VALIDATOR_CACHE_UNAVAILABLE_VALUE
        assert result['cache_coverage_percent'] == -1

    def test_get_cache_statistics_general_error(self, mock_local_source):
        """Test cache statistics when general error occurs."""
        mock_local_source.get_catalog_statistics.side_effect = Exception("General error")
        
        validator = HybridValidator(mock_local_source)
        
        with pytest.raises(CacheStatsError, match="Failed to retrieve cache statistics"):
            validator.get_cache_statistics()
