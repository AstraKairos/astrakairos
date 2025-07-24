# tests/test_coordinate_parsing.py
"""
Tests for coordinate parsing utilities.
"""

import pytest
import logging
from unittest.mock import patch

from astrakairos.utils.coordinate_parsing import (
    parse_coordinate_range,
    validate_spatial_filter_combination
)
from astrakairos.exceptions import ConfigurationError


class TestParseCoordinateRange:
    """Test the parse_coordinate_range function."""
    
    def test_valid_ra_range(self):
        """Test valid RA range parsing."""
        result = parse_coordinate_range("18.5,20.5", "ra")
        assert result == (18.5, 20.5)
    
    def test_valid_dec_range(self):
        """Test valid Dec range parsing."""
        result = parse_coordinate_range("-30.0,15.5", "dec")
        assert result == (-30.0, 15.5)
    
    def test_ra_boundary_values(self):
        """Test RA boundary values."""
        # Test minimum and maximum RA values (0-24 hours)
        result = parse_coordinate_range("0.0,24.0", "ra")
        assert result == (0.0, 24.0)
        
        result = parse_coordinate_range("12.0,12.1", "ra")
        assert result == (12.0, 12.1)
    
    def test_dec_boundary_values(self):
        """Test Dec boundary values."""
        # Test minimum and maximum Dec values (-90 to +90 degrees)
        result = parse_coordinate_range("-90.0,90.0", "dec")
        assert result == (-90.0, 90.0)
        
        result = parse_coordinate_range("0.0,45.0", "dec")
        assert result == (0.0, 45.0)
    
    def test_ra_wraparound_case(self):
        """Test RA wraparound case (max > min)."""
        # This should be valid for RA to handle wraparound
        result = parse_coordinate_range("22.0,2.0", "ra")
        assert result == (22.0, 2.0)
    
    def test_whitespace_handling(self):
        """Test handling of whitespace in input."""
        result = parse_coordinate_range(" 18.5 , 20.5 ", "ra")
        assert result == (18.5, 20.5)
        
        result = parse_coordinate_range("  -30.0,  15.5  ", "dec")
        assert result == (-30.0, 15.5)
    
    def test_empty_string_error(self):
        """Test error on empty string."""
        with pytest.raises(ConfigurationError, match="Empty RA range provided"):
            parse_coordinate_range("", "ra")
        
        with pytest.raises(ConfigurationError, match="Empty DEC range provided"):
            parse_coordinate_range("   ", "dec")
    
    def test_invalid_format_no_comma(self):
        """Test error on missing comma."""
        with pytest.raises(ConfigurationError, match="Invalid RA range format"):
            parse_coordinate_range("18.5", "ra")
    
    def test_invalid_format_too_many_values(self):
        """Test error on too many values."""
        with pytest.raises(ConfigurationError, match="Expected exactly two values"):
            parse_coordinate_range("18.5,20.5,22.0", "ra")
    
    def test_invalid_numeric_values(self):
        """Test error on non-numeric values."""
        with pytest.raises(ConfigurationError, match="Invalid numeric values"):
            parse_coordinate_range("abc,20.5", "ra")
        
        with pytest.raises(ConfigurationError, match="Invalid numeric values"):
            parse_coordinate_range("18.5,def", "dec")
    
    def test_ra_out_of_bounds(self):
        """Test error on RA values out of bounds."""
        with pytest.raises(ConfigurationError, match="RA values must be between 0.0 and 24.0 hours"):
            parse_coordinate_range("-1.0,20.5", "ra")
        
        with pytest.raises(ConfigurationError, match="RA values must be between 0.0 and 24.0 hours"):
            parse_coordinate_range("18.5,25.0", "ra")
    
    def test_dec_out_of_bounds(self):
        """Test error on Dec values out of bounds."""
        with pytest.raises(ConfigurationError, match="Dec values must be between -90.0 and \\+90.0 degrees"):
            parse_coordinate_range("-95.0,15.5", "dec")
        
        with pytest.raises(ConfigurationError, match="Dec values must be between -90.0 and \\+90.0 degrees"):
            parse_coordinate_range("-30.0,95.0", "dec")
    
    def test_dec_min_greater_than_max(self):
        """Test error when Dec minimum > maximum."""
        with pytest.raises(ConfigurationError, match="Dec minimum \\(30.0\\) cannot be greater than maximum \\(15.0\\)"):
            parse_coordinate_range("30.0,15.0", "dec")
    
    def test_unknown_coordinate_type(self):
        """Test error on unknown coordinate type."""
        with pytest.raises(ConfigurationError, match="Unknown coordinate type: galactic"):
            parse_coordinate_range("18.5,20.5", "galactic")
    
    def test_case_insensitive_coordinate_type(self):
        """Test that coordinate type is case insensitive."""
        result1 = parse_coordinate_range("18.5,20.5", "RA")
        result2 = parse_coordinate_range("18.5,20.5", "ra")
        assert result1 == result2
        
        result3 = parse_coordinate_range("-30.0,15.5", "DEC")
        result4 = parse_coordinate_range("-30.0,15.5", "dec")
        assert result3 == result4


class TestValidateSpatialFilterCombination:
    """Test the validate_spatial_filter_combination function."""
    
    def test_no_filters(self):
        """Test with no filters applied."""
        # Should not raise any exceptions
        validate_spatial_filter_combination(None, None)
    
    def test_ra_filter_only(self, caplog):
        """Test with only RA filter."""
        with caplog.at_level(logging.INFO):
            validate_spatial_filter_combination((18.0, 20.0), None)
        
        assert "Spatial filter: RA=[18.00, 20.00]h (all declinations)" in caplog.text
    
    def test_dec_filter_only(self, caplog):
        """Test with only Dec filter."""
        with caplog.at_level(logging.INFO):
            validate_spatial_filter_combination(None, (-30.0, 15.0))
        
        assert "Spatial filter: Dec=[-30.00, 15.00]° (all right ascensions)" in caplog.text
    
    def test_both_filters(self, caplog):
        """Test with both RA and Dec filters."""
        with caplog.at_level(logging.INFO):
            validate_spatial_filter_combination((18.0, 20.0), (-30.0, 15.0))
        
        assert "Spatial filter: RA=[18.00, 20.00]h, Dec=[-30.00, 15.00]°" in caplog.text
    
    def test_narrow_ra_range_warning(self, caplog):
        """Test warning for very narrow RA range."""
        with caplog.at_level(logging.WARNING):
            validate_spatial_filter_combination((18.0, 18.05), (-30.0, 15.0))
        
        assert "Very narrow RA range detected" in caplog.text
        assert "This may result in very few or no systems" in caplog.text
    
    def test_narrow_dec_range_warning(self, caplog):
        """Test warning for very narrow Dec range."""
        with caplog.at_level(logging.WARNING):
            validate_spatial_filter_combination((18.0, 20.0), (0.0, 0.05))
        
        assert "Very narrow Dec range detected" in caplog.text
        assert "This may result in very few or no systems" in caplog.text
    
    def test_ra_wraparound_width_calculation(self, caplog):
        """Test RA width calculation for wraparound case."""
        # RA wraparound: 22.0 to 2.0 should be 4.0 hours wide
        with caplog.at_level(logging.INFO):
            validate_spatial_filter_combination((22.0, 2.0), (-30.0, 15.0))
        
        # Should not trigger narrow range warning since width is 4.0 hours
        assert "Very narrow RA range detected" not in caplog.text
        assert "Spatial filter: RA=[22.00, 2.00]h, Dec=[-30.00, 15.00]°" in caplog.text
    
    def test_ra_narrow_wraparound_warning(self, caplog):
        """Test warning for narrow RA wraparound case."""
        # RA wraparound: 23.96 to 0.03 should be 0.07 hours wide and trigger warning
        with caplog.at_level(logging.WARNING):
            validate_spatial_filter_combination((23.96, 0.03), (-30.0, 15.0))
        
        assert "Very narrow RA range detected: 0.070 hours" in caplog.text


class TestCoordinateParsingIntegration:
    """Integration tests combining multiple functions."""
    
    def test_parse_and_validate_workflow(self, caplog):
        """Test typical workflow of parsing then validating."""
        # Parse coordinates
        ra_range = parse_coordinate_range("18.0,20.0", "ra")
        dec_range = parse_coordinate_range("-30.0,15.0", "dec")
        
        # Validate combination
        with caplog.at_level(logging.INFO):
            validate_spatial_filter_combination(ra_range, dec_range)
        
        assert ra_range == (18.0, 20.0)
        assert dec_range == (-30.0, 15.0)
        assert "Spatial filter: RA=[18.00, 20.00]h, Dec=[-30.00, 15.00]°" in caplog.text
    
    def test_edge_case_coordinate_values(self):
        """Test edge cases with boundary coordinate values."""
        # Test exactly at boundaries
        ra_result = parse_coordinate_range("0.0,24.0", "ra")
        dec_result = parse_coordinate_range("-90.0,90.0", "dec")
        
        assert ra_result == (0.0, 24.0)
        assert dec_result == (-90.0, 90.0)
        
        # Should not raise exceptions
        validate_spatial_filter_combination(ra_result, dec_result)
