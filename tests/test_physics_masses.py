"""
Tests for astrakairos.physics.masses module.

Test cases for mass calculation functions including input validation,
Monte Carlo error propagation, and individual mass calculations.
"""

import numpy as np
import pytest
from dataclasses import replace

from astrakairos.physics.masses import (
    calculate_total_mass_kepler3,
    calculate_individual_masses,
    MassResult,
    _validate_mass_inputs,
    _calculate_mc_statistics
)
from astrakairos.exceptions import InvalidMassInputError, NumericalInstabilityError
from astrakairos.config import (
    MIN_PERIOD_YEARS, MAX_PERIOD_YEARS,
    MIN_SEMIMAJOR_AXIS_ARCSEC,
    MIN_PARALLAX_MAS, MAX_PARALLAX_MAS,
    MIN_MASS_RATIO, MAX_MASS_RATIO,
    DEFAULT_MC_SAMPLES
)


class TestMassValidation:
    """Test input validation for mass calculations."""
    
    def test_valid_inputs(self):
        """Test that valid inputs pass validation."""
        warnings = _validate_mass_inputs(
            period_years=50.0,
            semimajor_axis_arcsec=1.0,
            parallax_mas=10.0
        )
        assert isinstance(warnings, list)
    
    def test_invalid_period_too_small(self):
        """Test that too small period raises exception."""
        with pytest.raises(InvalidMassInputError, match="Period.*outside valid range"):
            _validate_mass_inputs(
                period_years=0.01,  # Below MIN_PERIOD_YEARS
                semimajor_axis_arcsec=1.0,
                parallax_mas=10.0
            )
    
    def test_invalid_period_too_large(self):
        """Test that too large period raises exception."""
        with pytest.raises(InvalidMassInputError, match="Period.*outside valid range"):
            _validate_mass_inputs(
                period_years=200000.0,  # Above MAX_PERIOD_YEARS
                semimajor_axis_arcsec=1.0,
                parallax_mas=10.0
            )
    
    def test_invalid_semimajor_axis(self):
        """Test that invalid semi-major axis raises exception."""
        with pytest.raises(InvalidMassInputError, match="Semi-major axis.*below minimum"):
            _validate_mass_inputs(
                period_years=50.0,
                semimajor_axis_arcsec=0.0001,  # Below minimum
                parallax_mas=10.0
            )
    
    def test_invalid_parallax_too_small(self):
        """Test that too small parallax raises exception."""
        with pytest.raises(InvalidMassInputError, match="Parallax.*outside valid range"):
            _validate_mass_inputs(
                period_years=50.0,
                semimajor_axis_arcsec=1.0,
                parallax_mas=0.01  # Below MIN_PARALLAX_MAS
            )
    
    def test_invalid_parallax_too_large(self):
        """Test that too large parallax raises exception."""
        with pytest.raises(InvalidMassInputError, match="Parallax.*outside valid range"):
            _validate_mass_inputs(
                period_years=50.0,
                semimajor_axis_arcsec=1.0,
                parallax_mas=2000.0  # Above MAX_PARALLAX_MAS
            )
    
    def test_negative_errors(self):
        """Test that negative errors raise exception."""
        with pytest.raises(InvalidMassInputError, match="Negative uncertainties are not allowed"):
            _validate_mass_inputs(
                period_years=50.0,
                semimajor_axis_arcsec=1.0,
                parallax_mas=10.0,
                period_error=-1.0
            )


class TestMCStatistics:
    """Test Monte Carlo statistics calculations."""
    
    def test_valid_samples(self):
        """Test statistics calculation with valid samples."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = _calculate_mc_statistics(samples)
        
        assert 'median' in stats
        assert 'uncertainty' in stats
        assert 'p_lower' in stats
        assert 'p_upper' in stats
        assert stats['median'] == 3.0
    
    def test_empty_samples(self):
        """Test that empty samples raise exception."""
        samples = np.array([])
        with pytest.raises(NumericalInstabilityError, match="No Monte Carlo samples provided"):
            _calculate_mc_statistics(samples)
    
    def test_all_invalid_samples(self):
        """Test that all invalid samples raise exception."""
        samples = np.array([np.nan, np.inf, -1.0, 0.0])
        with pytest.raises(NumericalInstabilityError, match="No valid samples"):
            _calculate_mc_statistics(samples)


class TestTotalMassCalculation:
    """Test total mass calculation with Kepler's Third Law."""
    
    def test_basic_calculation(self):
        """Test basic mass calculation without errors."""
        result = calculate_total_mass_kepler3(
            period_years=50.0,
            semimajor_axis_arcsec=1.0,
            parallax_mas=25.0  # 40 pc distance
        )
        
        assert isinstance(result, MassResult)
        assert result.total_mass_solar > 0
        assert result.total_mass_error == 0.0  # No input errors
        assert result.individual_masses_solar is None
        assert result.distance_used_pc == 40.0
        assert result.parallax_used_mas == 25.0
    
    def test_with_uncertainties(self):
        """Test mass calculation with input uncertainties."""
        result = calculate_total_mass_kepler3(
            period_years=50.0,
            semimajor_axis_arcsec=1.0,
            parallax_mas=25.0,
            period_error=1.0,
            semimajor_axis_error=0.1,
            parallax_error=1.0,
            mc_samples=100  # Small number for testing
        )
        
        assert result.total_mass_error > 0
        assert result.mc_samples == 100
    
    def test_invalid_inputs(self):
        """Test that invalid inputs raise exception."""
        with pytest.raises(InvalidMassInputError):
            calculate_total_mass_kepler3(
                period_years=0.01,  # Invalid
                semimajor_axis_arcsec=1.0,
                parallax_mas=25.0
            )


class TestIndividualMassCalculation:
    """Test individual mass calculation from total mass and mass ratio."""
    
    def test_pure_function_behavior(self):
        """Test that calculate_individual_masses is a pure function."""
        # Create initial mass result
        initial_result = calculate_total_mass_kepler3(
            period_years=50.0,
            semimajor_axis_arcsec=1.0,
            parallax_mas=25.0
        )
        
        # Calculate individual masses
        updated_result = calculate_individual_masses(
            initial_result,
            mass_ratio=0.5,
            mc_samples=100
        )
        
        # Original result should be unchanged
        assert initial_result.individual_masses_solar is None
        assert initial_result.mass_ratio is None
        
        # New result should have individual masses
        assert updated_result.individual_masses_solar is not None
        assert updated_result.mass_ratio == 0.5
        assert len(updated_result.individual_masses_solar) == 2
    
    def test_invalid_mass_ratio(self):
        """Test that invalid mass ratio raises exception."""
        result = calculate_total_mass_kepler3(
            period_years=50.0,
            semimajor_axis_arcsec=1.0,
            parallax_mas=25.0
        )
        
        with pytest.raises(InvalidMassInputError, match="Mass ratio.*outside valid range"):
            calculate_individual_masses(result, mass_ratio=2.0)  # > MAX_MASS_RATIO
    
    def test_mass_conservation(self):
        """Test that individual masses sum to total mass."""
        result = calculate_total_mass_kepler3(
            period_years=50.0,
            semimajor_axis_arcsec=1.0,
            parallax_mas=25.0
        )
        
        updated_result = calculate_individual_masses(
            result,
            mass_ratio=0.5
        )
        
        m1, m2 = updated_result.individual_masses_solar
        total_calculated = m1 + m2
        
        # Should be approximately equal (within numerical precision)
        assert abs(total_calculated - updated_result.total_mass_solar) < 1e-10


class TestIntegration:
    """Integration tests for the full mass calculation workflow."""
    
    def test_complete_workflow(self):
        """Test complete workflow from total mass to individual masses."""
        # Calculate total mass
        total_result = calculate_total_mass_kepler3(
            period_years=50.0,
            semimajor_axis_arcsec=1.0,
            parallax_mas=25.0,
            period_error=1.0,
            parallax_error=1.0,
            parallax_source='gaia_dr3',
            mc_samples=100
        )
        
        # Calculate individual masses
        final_result = calculate_individual_masses(
            total_result,
            mass_ratio=0.8,
            mass_ratio_error=0.1,
            mc_samples=100
        )
        
        # Verify structure
        assert final_result.total_mass_solar > 0
        assert final_result.total_mass_error > 0
        assert final_result.individual_masses_solar is not None
        assert final_result.individual_mass_errors is not None
        assert final_result.mass_ratio == 0.8
        assert final_result.mass_ratio_error == 0.1
        assert final_result.parallax_source == 'gaia_dr3'
        assert final_result.quality_score > 0
