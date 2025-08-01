import pytest
import numpy as np
from unittest.mock import Mock, patch
from astropy.table import Table
from astrakairos.physics.dynamics import (
    calculate_observation_priority_index,
    calculate_robust_linear_fit,
    calculate_prediction_divergence,
    estimate_velocity_from_endpoints
)

class TestObservationPriorityIndex:
    """Test calculate_observation_priority_index function."""
    
    @pytest.fixture
    def sample_orbital_elements(self):
        return {
            'P': 50.0,
            'a': 1.0,
            'e': 0.3,
            'i': 60.0,
            'Omega': 120.0,
            'omega': 90.0,
            'T': 2000.0
        }
    
    @pytest.fixture
    def sample_wds_summary(self):
        return {
            'wds_name': '00000+0000',
            'date_last': 2020.0,
            'pa_last': 45.0,
            'sep_last': 1.5,
            'obs': 10
        }
    
    @patch('astrakairos.physics.dynamics.predict_position')
    def test_opi_calculation_success(self, mock_predict, sample_orbital_elements, sample_wds_summary):
        """Test successful OPI calculation."""
        # Mock the orbital prediction
        mock_predict.return_value = (47.0, 1.4)  # Slightly different from observed
        
        result = calculate_observation_priority_index(
            sample_orbital_elements,
            sample_wds_summary,
            2024.0
        )
        
        assert result is not None
        opi, deviation = result
        assert isinstance(opi, float)
        assert isinstance(deviation, float)
        assert opi >= 0
        assert deviation >= 0
        
        # Should have called predict_position with the last observation date
        mock_predict.assert_called_once_with(sample_orbital_elements, 2020.0)
    
    def test_opi_missing_data(self, sample_orbital_elements):
        """Test OPI calculation with missing data."""
        incomplete_summary = {
            'wds_name': '00000+0000',
            'date_last': 2020.0,
            'pa_last': 45.0,
            # Missing 'sep_last'
        }
        
        result = calculate_observation_priority_index(
            sample_orbital_elements,
            incomplete_summary,
            2024.0
        )
        
        assert result is None
    
    @patch('astrakairos.physics.dynamics.predict_position')
    def test_opi_prediction_failure(self, mock_predict, sample_orbital_elements, sample_wds_summary):
        """Test OPI calculation when orbital prediction fails."""
        mock_predict.return_value = None
        
        result = calculate_observation_priority_index(
            sample_orbital_elements,
            sample_wds_summary,
            2024.0
        )
        
        assert result is None
    
    @patch('astrakairos.physics.dynamics.predict_position')
    def test_opi_zero_time_baseline(self, mock_predict, sample_orbital_elements, sample_wds_summary):
        """Test OPI calculation with zero time baseline."""
        mock_predict.return_value = (47.0, 1.4)
        
        # Set current date equal to last observation date
        result = calculate_observation_priority_index(
            sample_orbital_elements,
            sample_wds_summary,
            2020.0
        )
        
        assert result is not None
        opi, deviation = result
        assert opi == np.inf or opi == 0.0  # Depends on deviation magnitude
    
    @patch('astrakairos.physics.dynamics.predict_position')
    def test_opi_calculation_values(self, mock_predict, sample_orbital_elements, sample_wds_summary):
        """Test OPI calculation with known values."""
        # Mock perfect prediction (no deviation)
        mock_predict.return_value = (45.0, 1.5)  # Exactly matches observed
        
        result = calculate_observation_priority_index(
            sample_orbital_elements,
            sample_wds_summary,
            2024.0
        )
        
        assert result is not None
        opi, deviation = result
        assert deviation < 0.001  # Should be very small
        assert opi < 0.001  # Should be very small

class TestRobustLinearFit:
    """Test calculate_robust_linear_fit function."""
    
    @pytest.fixture
    def sample_measurements_linear(self):
        """Create sample measurements with linear motion."""
        return Table({
            'epoch': [2000.0, 2005.0, 2010.0, 2015.0, 2020.0],
            'theta': [40.0, 41.0, 42.0, 43.0, 44.0],  # Linear increase
            'rho': [2.0, 1.9, 1.8, 1.7, 1.6]  # Linear decrease
        })
    
    @pytest.fixture
    def sample_measurements_curved(self):
        """Create sample measurements with curved motion."""
        epochs = np.array([2000.0, 2005.0, 2010.0, 2015.0, 2020.0])
        # Quadratic motion
        thetas = 40.0 + 0.5 * (epochs - 2000.0) + 0.01 * (epochs - 2000.0)**2
        rhos = 2.0 - 0.02 * (epochs - 2000.0) + 0.001 * (epochs - 2000.0)**2
        
        return Table({
            'epoch': epochs,
            'theta': thetas,
            'rho': rhos
        })
    
    @pytest.fixture
    def sample_measurements_with_outliers(self):
        """Create sample measurements with outliers."""
        return Table({
            'epoch': [2000.0, 2005.0, 2010.0, 2015.0, 2020.0],
            'theta': [40.0, 41.0, 50.0, 43.0, 44.0],  # Outlier at 2010
            'rho': [2.0, 1.9, 1.8, 1.7, 1.6]
        })
    
    def test_robust_fit_success(self, sample_measurements_linear):
        """Test successful robust linear fit."""
        result = calculate_robust_linear_fit(sample_measurements_linear)
        
        assert result is not None
        assert 'vx_arcsec_per_year' in result
        assert 'vy_arcsec_per_year' in result
        assert 'v_total_robust' in result
        assert 'pa_v_robust' in result
        assert 'rmse' in result
        assert 'n_points_fit' in result
        assert 'time_baseline_years' in result
        assert 'intercept_x' in result
        assert 'intercept_y' in result
        assert 'mean_epoch_fit' in result  # NEW: Verify centered regression support
        
        # Check that velocity components are reasonable
        assert isinstance(result['vx_arcsec_per_year'], float)
        assert isinstance(result['vy_arcsec_per_year'], float)
        assert result['v_total_robust'] >= 0
        assert 0 <= result['pa_v_robust'] < 360
        assert result['rmse'] >= 0
        assert result['n_points_fit'] == 5
        assert result['time_baseline_years'] == 20.0
    
    def test_robust_fit_insufficient_data(self):
        """Test robust fit with insufficient data."""
        insufficient_data = Table({
            'epoch': [2000.0, 2005.0],
            'theta': [40.0, 41.0],
            'rho': [2.0, 1.9]
        })
        
        result = calculate_robust_linear_fit(insufficient_data)
        assert result is None
    
    def test_robust_fit_curved_motion(self, sample_measurements_curved):
        """Test robust fit with curved motion (should have higher RMSE)."""
        result = calculate_robust_linear_fit(sample_measurements_curved)
        
        assert result is not None
        assert result['rmse'] > 0  # Should have some residual due to curvature
    
    def test_robust_fit_with_outliers(self, sample_measurements_with_outliers):
        """Test robust fit with outliers (should be resistant)."""
        result = calculate_robust_linear_fit(sample_measurements_with_outliers)
        
        assert result is not None
        # The robust fit should not be severely affected by the outlier
        assert result['rmse'] < 5.0  # Should be reasonable despite outlier
    
    def test_robust_fit_empty_table(self):
        """Test robust fit with empty table."""
        empty_table = Table({
            'epoch': [],
            'theta': [],
            'rho': []
        })
        
        result = calculate_robust_linear_fit(empty_table)
        assert result is None
    
    def test_robust_fit_single_point(self):
        """Test robust fit with single point."""
        single_point = Table({
            'epoch': [2000.0],
            'theta': [40.0],
            'rho': [2.0]
        })
        
        result = calculate_robust_linear_fit(single_point)
        assert result is None

class TestCurvatureIndex:
    """Test calculate_prediction_divergence function."""
    
    @pytest.fixture
    def sample_orbital_elements(self):
        return {
            'P': 50.0,
            'a': 1.0,
            'e': 0.3,
            'i': 60.0,
            'Omega': 120.0,
            'omega': 90.0,
            'T': 2000.0
        }
    
    @pytest.fixture
    def sample_measurements(self):
        return Table({
            'epoch': [2000.0, 2005.0, 2010.0, 2015.0, 2020.0],
            'theta': [40.0, 41.0, 42.0, 43.0, 44.0],
            'rho': [2.0, 1.9, 1.8, 1.7, 1.6]
        })
    
    @patch('astrakairos.physics.dynamics.predict_position')
    def test_prediction_divergence_success(self, mock_predict, 
                                   sample_orbital_elements, sample_measurements):
        """Test successful curvature index calculation."""
        # Create mock linear fit results
        linear_fit_results = {
            'vx_arcsec_per_year': 0.1,
            'vy_arcsec_per_year': -0.05,
            'rmse': 0.02,
            'intercept_x': 1.8,  # Required for correct curvature calculation
            'intercept_y': 0.2,   # Required for correct curvature calculation
            'mean_epoch_fit': 2010.0  # CRITICAL: Required for centered regression
        }
        
        # Mock the orbital prediction
        mock_predict.return_value = (45.0, 1.5)
        
        result = calculate_prediction_divergence(
            sample_orbital_elements,
            linear_fit_results,
            2024.0
        )
        
        assert result is not None
        assert isinstance(result, float)
        assert result >= 0
        
        mock_predict.assert_called_once_with(sample_orbital_elements, 2024.0)
    
    def test_prediction_divergence_no_orbital_elements(self, sample_measurements):
        """Test prediction divergence with no orbital elements."""
        linear_fit_results = {
            'vx_arcsec_per_year': 0.1,
            'vy_arcsec_per_year': -0.05,
            'intercept_x': 1.8,
            'intercept_y': 0.2,
            'mean_epoch_fit': 2010.0
        }
        
        result = calculate_prediction_divergence(
            None,
            linear_fit_results,
            2024.0
        )
        
        assert result is None
    
    def test_prediction_divergence_insufficient_measurements(self, sample_orbital_elements):
        """Test prediction divergence with insufficient linear fit results."""
        # Test with None linear fit results
        result = calculate_prediction_divergence(
            sample_orbital_elements,
            None,
            2024.0
        )
        
        assert result is None
    
    def test_prediction_divergence_robust_fit_failure(self, sample_orbital_elements, sample_measurements):
        """Test prediction divergence when no linear fit results provided."""
        # Test with empty linear fit results
        result = calculate_prediction_divergence(
            sample_orbital_elements,
            {},
            2024.0
        )
        
        assert result is None
    
    @patch('astrakairos.physics.dynamics.predict_position')
    def test_prediction_divergence_prediction_failure(self, mock_predict,
                                              sample_orbital_elements, sample_measurements):
        """Test prediction divergence when orbital prediction fails."""
        linear_fit_results = {
            'vx_arcsec_per_year': 0.1,
            'vy_arcsec_per_year': -0.05,
            'rmse': 0.02,
            'intercept_x': 1.8,
            'intercept_y': 0.2,
            'mean_epoch_fit': 2010.0
        }
        mock_predict.return_value = None
        
        result = calculate_prediction_divergence(
            sample_orbital_elements,
            linear_fit_results,
            2024.0
        )
        
        assert result is None

class TestMeanVelocityFromEndpoints:
    """Test estimate_velocity_from_endpoints function."""
    
    @pytest.fixture
    def sample_wds_summary(self):
        return {
            'wds_name': '00000+0000',
            'date_first': 2000.0,
            'date_last': 2020.0,
            'pa_first': 40.0,
            'pa_last': 44.0,
            'sep_first': 2.0,
            'sep_last': 1.6,
            'obs': 10
        }
    
    def test_endpoint_velocity_success(self, sample_wds_summary):
        """Test successful endpoint velocity calculation."""
        result = estimate_velocity_from_endpoints(sample_wds_summary)
        
        assert result is not None
        assert 'vx_arcsec_per_year' in result
        assert 'vy_arcsec_per_year' in result
        assert 'v_total_estimate' in result
        assert 'pa_v_estimate' in result
        assert 'time_baseline_years' in result
        assert 'n_points_fit' in result
        assert 'method' in result
        
        assert result['n_points_fit'] == 2
        assert result['method'] == 'two_point_estimate'
        assert result['time_baseline_years'] == 20.0
        assert result['v_total_estimate'] >= 0
        assert 0 <= result['pa_v_estimate'] < 360
    
    def test_endpoint_velocity_missing_data(self):
        """Test endpoint velocity with missing data."""
        incomplete_summary = {
            'wds_name': '00000+0000',
            'date_first': 2000.0,
            'date_last': 2020.0,
            'pa_first': 40.0,
            # Missing 'pa_last'
        }
        
        result = estimate_velocity_from_endpoints(incomplete_summary)
        assert result is None
    
    def test_endpoint_velocity_zero_time_baseline(self, sample_wds_summary):
        """Test endpoint velocity with zero time baseline."""
        sample_wds_summary['date_last'] = 2000.0  # Same as date_first
        
        result = estimate_velocity_from_endpoints(sample_wds_summary)
        assert result is None
    
    def test_endpoint_velocity_negative_time_baseline(self, sample_wds_summary):
        """Test endpoint velocity with negative time baseline."""
        sample_wds_summary['date_last'] = 1990.0  # Earlier than date_first
        
        result = estimate_velocity_from_endpoints(sample_wds_summary)
        assert result is None
    
    def test_endpoint_velocity_calculation_values(self, sample_wds_summary):
        """Test endpoint velocity with known values."""
        # Set up simple case: pure motion in theta
        sample_wds_summary.update({
            'pa_first': 0.0,
            'pa_last': 0.0,
            'sep_first': 1.0,
            'sep_last': 2.0  # Separation increases
        })
        
        result = estimate_velocity_from_endpoints(sample_wds_summary)
        
        assert result is not None
        # Should have pure radial velocity (in y-direction for PA=0)
        assert abs(result['vx_arcsec_per_year']) < 1e-10  # Should be ~0
    
    def test_endpoint_velocity_single_epoch_system(self):
        """Test endpoint velocity with single-epoch system (no date_last)."""
        single_epoch_summary = {
            'wds_name': '12345+6789',
            'date_first': 2000.0,
            'pa_first': 45.0,
            'sep_first': 1.5,
            # No date_last, pa_last, sep_last
        }
        
        result = estimate_velocity_from_endpoints(single_epoch_summary)
        
        assert result is not None, "Single-epoch systems should return basic position info"
        
        # Velocity fields should be None for single-epoch
        assert result['vx_arcsec_per_year'] is None
        assert result['vy_arcsec_per_year'] is None
        assert result['v_total_estimate'] is None
        assert result['pa_v_estimate'] is None
        
        # Basic fields should be present
        assert result['time_baseline_years'] == 0.0
        assert result['n_points_fit'] == 1
        assert result['method'] == 'single_epoch_position'
        
        # Position information should be available
        assert 'position_x_arcsec' in result
        assert 'position_y_arcsec' in result
        assert 'epoch_first' in result
        assert 'pa_first_deg' in result
        assert 'sep_first_arcsec' in result
        
        # Verify position calculation
        expected_x = 1.5 * np.sin(np.radians(45.0))
        expected_y = 1.5 * np.cos(np.radians(45.0))
        assert abs(result['position_x_arcsec'] - expected_x) < 1e-10
        assert abs(result['position_y_arcsec'] - expected_y) < 1e-10
