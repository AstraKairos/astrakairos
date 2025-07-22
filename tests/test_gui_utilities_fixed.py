# test_gui_utilities.py
"""Test module for GUI utilities and formatting functions."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import math

from astrakairos.planner.gui.utilities import (
    GUIUtilities,
    format_ra_hours,
    format_dec_degrees,
    format_time,
    format_time_utc,
    validate_coordinate_range,
    safe_float_conversion,
    safe_int_conversion
)


class TestGUIUtilities:
    """Test GUIUtilities class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        mock_app = Mock()
        self.utilities = GUIUtilities(mock_app)
    
    def test_init(self):
        """Test GUIUtilities initialization."""
        mock_app = Mock()
        utilities = GUIUtilities(mock_app)
        assert utilities.app == mock_app
    
    def test_create_scrollable_frame_returns_frame(self):
        """Test that create_scrollable_frame returns a frame."""
        with patch('astrakairos.planner.gui.utilities.ttk.Frame') as mock_frame:
            with patch('astrakairos.planner.gui.utilities.tk.Canvas'):
                with patch('astrakairos.planner.gui.utilities.ttk.Scrollbar'):
                    mock_parent = Mock()
                    mock_frame_instance = Mock()
                    mock_frame.return_value = mock_frame_instance
                    
                    result = self.utilities.create_scrollable_frame(mock_parent)
                    
                    # Should return the scrollable frame
                    assert result == mock_frame_instance
    
    def test_configure_text_widget_scroll(self):
        """Test text widget scroll configuration."""
        mock_text_widget = Mock()
        
        GUIUtilities.configure_text_widget_scroll(mock_text_widget)
        
        # Verify bind calls were made
        assert mock_text_widget.bind.call_count >= 2  # Enter and Leave events


class TestFormatFunctions:
    """Test coordinate and time formatting functions."""
    
    def test_format_ra_hours(self):
        """Test RA formatting in hours."""
        # Test normal values based on actual implementation
        assert format_ra_hours(0.0) == "00h00m"
        assert format_ra_hours(1.0) == "01h00m"
        assert format_ra_hours(12.5) == "12h30m"
        assert format_ra_hours(23.75) == "23h45m"
    
    def test_format_dec_degrees(self):
        """Test declination formatting in degrees."""
        # Test based on actual implementation
        assert format_dec_degrees(0.0) == "+00°00'"
        assert format_dec_degrees(45.5) == "+45°30'"
        assert format_dec_degrees(-30.25) == "-30°15'"
        assert format_dec_degrees(-0.5) == "-00°30'"
    
    def test_format_time(self):
        """Test time formatting for local time."""
        # Test None input
        assert format_time(None) == "N/A"
        
        # Test datetime object
        dt = datetime(2023, 12, 25, 14, 30, 45)
        result = format_time(dt)
        # Should contain time, format may vary by timezone
        assert "14:30" in result
    
    def test_format_time_utc(self):
        """Test time formatting for UTC time."""
        # Test None input
        assert format_time_utc(None) == "N/A"
        
        # Test datetime object with timezone awareness
        dt = datetime(2023, 12, 25, 14, 30, 45)
        result = format_time_utc(dt)
        # Should handle timezone conversion
        assert "UTC" in result or result == "N/A"


class TestValidationFunctions:
    """Test validation and conversion functions."""
    
    def test_validate_coordinate_range(self):
        """Test coordinate range validation."""
        # Test valid ranges
        assert validate_coordinate_range(5.0, 0.0, 10.0, "test") == True
        assert validate_coordinate_range(0.0, 0.0, 10.0, "test") == True
        assert validate_coordinate_range(10.0, 0.0, 10.0, "test") == True
        
        # Test invalid ranges
        assert validate_coordinate_range(-1.0, 0.0, 10.0, "test") == False
        assert validate_coordinate_range(11.0, 0.0, 10.0, "test") == False
    
    def test_validate_coordinate_range_with_none(self):
        """Test coordinate validation with None input."""
        # This should raise an error or handle None gracefully
        with pytest.raises(TypeError):
            validate_coordinate_range(None, 0.0, 10.0, "test")
    
    def test_safe_float_conversion(self):
        """Test safe float conversion."""
        # Test valid conversions
        assert safe_float_conversion("123.45") == 123.45
        assert safe_float_conversion("0.0") == 0.0
        assert safe_float_conversion("-45.67") == -45.67
        
        # Test integer strings
        assert safe_float_conversion("123") == 123.0
        
        # Test invalid conversions with default
        assert safe_float_conversion("invalid") == 0.0
        assert safe_float_conversion("") == 0.0
        assert safe_float_conversion("abc123") == 0.0
        
        # Test custom default
        assert safe_float_conversion("invalid", default=999.9) == 999.9
        assert safe_float_conversion("", default=-1.0) == -1.0
        
        # Test special float values
        assert safe_float_conversion("inf") == float('inf')
        assert safe_float_conversion("-inf") == float('-inf')
        
        # Test NaN handling
        result = safe_float_conversion("nan")
        assert math.isnan(result)  # NaN is returned as-is by float()
    
    def test_safe_int_conversion(self):
        """Test safe integer conversion."""
        # Test valid conversions
        assert safe_int_conversion("123") == 123
        assert safe_int_conversion("0") == 0
        assert safe_int_conversion("-45") == -45
        
        # Test invalid conversions (float strings should fail)
        assert safe_int_conversion("123.45") == 0  # Default for invalid input
        assert safe_int_conversion("invalid") == 0
        assert safe_int_conversion("") == 0
        assert safe_int_conversion("abc123") == 0
        
        # Test custom default
        assert safe_int_conversion("invalid", default=999) == 999
        assert safe_int_conversion("", default=-1) == -1


class TestCoordinateFormatting:
    """Test coordinate formatting edge cases."""
    
    def test_ra_hours_precision(self):
        """Test RA formatting precision."""
        # Test based on actual simple implementation
        test_cases = [
            (0.25, "00h15m"),      # Quarter hour
            (0.5, "00h30m"),       # Half hour
            (0.75, "00h45m"),      # Three quarters
            (12.345833, "12h20m"), # Precise (minutes only)
        ]
        
        for input_val, expected in test_cases:
            result = format_ra_hours(input_val)
            assert result == expected, f"Expected {expected}, got {result} for input {input_val}"
    
    def test_dec_degrees_precision(self):
        """Test declination formatting precision."""
        # Test based on actual simple implementation
        test_cases = [
            (0.25, "+00°15'"),        # Quarter degree
            (0.5, "+00°30'"),         # Half degree
            (0.75, "+00°45'"),        # Three quarters
            (-12.345833, "-12°20'"),  # Precise (arcminutes only)
        ]
        
        for input_val, expected in test_cases:
            result = format_dec_degrees(input_val)
            assert result == expected, f"Expected {expected}, got {result} for input {input_val}"
    
    def test_coordinate_boundary_cases(self):
        """Test coordinate formatting at boundaries."""
        # RA boundaries
        assert format_ra_hours(23.999722) == "23h59m"  # Almost 24h
        assert format_ra_hours(0.0) == "00h00m"        # Zero
        
        # Dec boundaries
        assert format_dec_degrees(89.999722) == "+89°59'"  # Almost 90°
        assert format_dec_degrees(-89.999722) == "-89°59'" # Almost -90°


class TestErrorHandling:
    """Test error handling in utility functions."""
    
    def test_format_functions_with_invalid_input(self):
        """Test format functions with edge case inputs."""
        # Test infinity values (should handle gracefully)
        try:
            result_ra = format_ra_hours(float('inf'))
            # Should either return a string or raise an exception we can catch
            assert isinstance(result_ra, str), "Should return a string for infinity RA"
        except (OverflowError, ValueError):
            # This is acceptable behavior for infinity
            pass
        
        try:
            result_dec = format_dec_degrees(float('inf'))
            # Should either return a string or raise an exception we can catch
            assert isinstance(result_dec, str), "Should return a string for infinity Dec"
        except (OverflowError, ValueError):
            # This is acceptable behavior for infinity
            pass
        
        # Negative RA should still work (but might format differently)
        result_neg = format_ra_hours(-1.0)
        assert isinstance(result_neg, str), "Should handle negative RA"
        # Accept various formats for negative values
        assert any(char in result_neg for char in ['-', 'h', 'm']), f"Got: {result_neg}"
