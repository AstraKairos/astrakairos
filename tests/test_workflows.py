# tests/test_workflows.py
"""
Test suite for analyzer workflows.

Tests the scientific analysis workflows for discovery, characterization, 
and orbital analysis. Focuses on the workflow logic rather than the
underlying physics calculations.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from astrakairos.analyzer.workflows import _perform_discovery_analysis
from astrakairos.data.source import WdsSummary


class TestDiscoveryWorkflow:
    """Test discovery analysis workflow."""
    
    @pytest.fixture
    def mock_data_source(self):
        """Mock data source for testing."""
        return Mock()
    
    @pytest.fixture
    def sample_multi_epoch_summary(self):
        """Sample WDS summary with multiple epochs."""
        return {
            'wds_id': '00001+0001',
            'date_first': 2000.0,
            'date_last': 2020.0,
            'pa_first': 45.0,
            'pa_last': 50.0,
            'sep_first': 1.0,
            'sep_last': 1.2
        }
    
    @pytest.fixture
    def sample_single_epoch_summary(self):
        """Sample WDS summary with single epoch only."""
        return {
            'wds_id': '00002+0002',
            'date_first': 2000.0,
            'pa_first': 45.0,
            'sep_first': 1.0
            # No date_last, pa_last, sep_last
        }
    
    @pytest.mark.asyncio
    async def test_discovery_multi_epoch_analysis(self, sample_multi_epoch_summary, mock_data_source):
        """Test discovery analysis with multi-epoch data."""
        with patch('astrakairos.analyzer.workflows.estimate_velocity_from_endpoints_mc') as mock_mc:
            mock_mc.return_value = {
                'v_total_estimate': 0.05,
                'pa_v_estimate': 45.0,
                'v_total_uncertainty': 0.01
            }
            
            result = await _perform_discovery_analysis(
                'test_id', sample_multi_epoch_summary, mock_data_source
            )
            
            assert result is not None
            assert 'v_total_estimate' in result
            assert 'v_total_uncertainty' in result
            mock_mc.assert_called_once_with(sample_multi_epoch_summary)
    
    @pytest.mark.asyncio
    async def test_discovery_single_epoch_analysis(self, sample_single_epoch_summary, mock_data_source):
        """Test discovery analysis with single-epoch data."""
        with patch('astrakairos.analyzer.workflows.estimate_velocity_from_endpoints') as mock_endpoints:
            mock_endpoints.return_value = {
                'vx_arcsec_per_year': None,
                'vy_arcsec_per_year': None,
                'v_total_estimate': None,
                'pa_v_estimate': None,
                'time_baseline_years': 0.0,
                'n_points_fit': 1,
                'method': 'single_epoch_position',
                'position_x_arcsec': 1.06,
                'position_y_arcsec': 1.06,
                'epoch_first': 2000.0,
                'pa_first_deg': 45.0,
                'sep_first_arcsec': 1.0
            }
            
            result = await _perform_discovery_analysis(
                'test_id', sample_single_epoch_summary, mock_data_source
            )
            
            assert result is not None
            assert result['v_total_estimate'] is None  # No velocity for single epoch
            assert result['method'] == 'single_epoch_position'
            assert result['analysis_type'] == 'single_epoch'
            assert result['n_points_fit'] == 1
            assert result['time_baseline_years'] == 0.0
            assert 'position_x_arcsec' in result
            assert 'position_y_arcsec' in result
            mock_endpoints.assert_called_once_with(sample_single_epoch_summary)
    
    @pytest.mark.asyncio
    async def test_discovery_single_epoch_config_disabled(self, sample_single_epoch_summary, mock_data_source):
        """Test discovery analysis when single-epoch systems are disabled."""
        with patch('astrakairos.analyzer.workflows.ALLOW_SINGLE_EPOCH_SYSTEMS', False):
            with pytest.raises(Exception) as exc_info:
                await _perform_discovery_analysis(
                    'test_id', sample_single_epoch_summary, mock_data_source
                )
            
            # Check the cause or the message in the chain
            error_message = str(exc_info.value)
            assert "Single-epoch system not allowed" in error_message or "Discovery analysis failed" in error_message
    
    @pytest.mark.asyncio
    async def test_discovery_missing_required_fields(self, mock_data_source):
        """Test discovery analysis with missing required fields."""
        incomplete_summary = {
            'wds_id': '00003+0003',
            'date_first': 2000.0,
            # Missing pa_first, sep_first
        }
        
        with pytest.raises(Exception) as exc_info:
            await _perform_discovery_analysis(
                'test_id', incomplete_summary, mock_data_source
            )
        
        # The exact error message might be wrapped, but analysis should fail
        error_message = str(exc_info.value)
        assert "Missing required fields" in error_message or "Discovery analysis failed" in error_message
    
    @pytest.mark.asyncio 
    async def test_discovery_single_epoch_calculation_failure(self, sample_single_epoch_summary, mock_data_source):
        """Test discovery analysis when single-epoch calculation fails."""
        with patch('astrakairos.analyzer.workflows.estimate_velocity_from_endpoints') as mock_endpoints:
            mock_endpoints.return_value = None  # Simulate calculation failure
            
            with pytest.raises(Exception) as exc_info:
                await _perform_discovery_analysis(
                    'test_id', sample_single_epoch_summary, mock_data_source
                )
            
            # Check that the analysis failed appropriately
            error_message = str(exc_info.value)
            assert "Failed to process single-epoch system" in error_message or "Discovery analysis failed" in error_message
