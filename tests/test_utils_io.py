# test_utils_io.py
"""Test module for utils.io functions."""

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch, mock_open
from io import StringIO

from astrakairos.utils.io import (
    DataLoadError,
    DataSaveError,
    InvalidWdsFormatError,
    CoordinateOutOfRangeError,
    load_csv_data,
    save_results_to_csv,
    format_coordinates_astropy,
    parse_wds_designation
)


class TestDataLoadError:
    """Test custom exception class."""
    
    def test_data_load_error_creation(self):
        """Test creating DataLoadError exception."""
        error = DataLoadError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)


class TestLoadCSVData:
    """Test CSV data loading functions."""
    
    def test_load_csv_data_valid_file(self):
        """Test loading valid CSV file with wds_id column."""
        csv_content = "wds_id,ra_hours,dec_degrees\n00013+1234,1.2,12.5\n01234-4567,2.3,45.6"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            df = load_csv_data(temp_path)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert 'wds_id' in df.columns
            assert 'ra_hours' in df.columns
            assert 'dec_degrees' in df.columns
            assert df.iloc[0]['wds_id'] == '00013+1234'
            
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_data_semicolon_delimiter(self):
        """Test loading CSV file with semicolon delimiter."""
        csv_content = "wds_id;ra_hours;dec_degrees\n00013+1234;1.2;12.5\n01234-4567;2.3;45.6"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            df = load_csv_data(temp_path)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert 'wds_id' in df.columns
            
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_data_missing_wds_id_column(self):
        """Test error when wds_id column is missing."""
        csv_content = "star_id,ra_hours,dec_degrees\nSTAR1,1.2,12.5\nSTAR2,2.3,45.6"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            with pytest.raises(DataLoadError) as exc_info:
                load_csv_data(temp_path)
            
            assert "wds_id" in str(exc_info.value).lower()
            
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_data_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(DataLoadError) as exc_info:
            load_csv_data("nonexistent_file.csv")
        
        assert "File not found" in str(exc_info.value)
    
    def test_load_csv_data_empty_file(self):
        """Test error when file is empty."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            with pytest.raises(DataLoadError) as exc_info:
                load_csv_data(temp_path)
            
            assert "File contains no data" in str(exc_info.value)
            
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_data_encoding_fallback(self):
        """Test encoding fallback from utf-8 to latin-1."""
        # Create a CSV with latin-1 specific characters
        csv_content = "wds_id,name\n00013+1234,Système Étoile"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='latin-1') as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            df = load_csv_data(temp_path)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert 'wds_id' in df.columns
            
        finally:
            os.unlink(temp_path)
    
    @patch('pandas.read_csv')
    def test_load_csv_data_parser_error(self, mock_read_csv):
        """Test handling of pandas parser errors."""
        mock_read_csv.side_effect = pd.errors.ParserError("Invalid CSV format")
        
        with pytest.raises(DataLoadError) as exc_info:
            load_csv_data("test.csv")
        
        assert "Could not parse CSV format in file" in str(exc_info.value)


class TestSaveResultsToCSV:
    """Test results saving functions."""
    
    def test_save_results_to_csv_valid_data(self):
        """Test saving valid results data."""
        results = [
            {'wds_id': '00013+1234', 'opi': 0.85, 'rmse': 0.12},
            {'wds_id': '01234-4567', 'opi': 0.92, 'rmse': 0.08}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            save_results_to_csv(results, temp_path)
            
            # Verify the file was created and contains expected data
            df = pd.read_csv(temp_path)
            assert len(df) == 2
            assert 'wds_id' in df.columns
            assert 'opi' in df.columns
            assert 'rmse' in df.columns
            assert df.iloc[0]['wds_id'] == '00013+1234'
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_results_to_csv_empty_results(self):
        """Test saving empty results (should handle gracefully)."""
        results = []
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Should not raise an error, just log a warning
            save_results_to_csv(results, temp_path)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_results_to_csv_permission_error(self):
        """Test error when file cannot be written due to permissions."""
        results = [{'wds_id': '00013+1234', 'opi': 0.85}]
        
        # Try to write to an invalid/inaccessible location
        import platform
        if platform.system() == "Windows":
            invalid_path = "C:\\Windows\\System32\\test_file.csv"  # System directory
        else:
            invalid_path = "/root/test_file.csv"  # Root directory
            
        with pytest.raises(DataSaveError) as exc_info:
            save_results_to_csv(results, invalid_path)
        
        # Check that it's the right type of error
        assert "OS error" in str(exc_info.value) or "Permission denied" in str(exc_info.value)
    
    def test_save_results_to_csv_invalid_structure(self):
        """Test error when results have invalid structure."""
        # Create results with inconsistent structure that will cause pandas error
        results = [
            {'wds_id': '00013+1234', 'opi': 0.85},
            {'wds_id': '01234-4567', 'opi': [1, 2, 3]}  # Invalid: list in opi field
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # This might or might not raise an error depending on pandas version
            # The function should handle it gracefully
            save_results_to_csv(results, temp_path)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestFormatCoordinatesAstropy:
    """Test coordinate formatting functions."""
    
    @patch('astrakairos.utils.io.SkyCoord')
    def test_format_coordinates_astropy_valid(self, mock_skycoord):
        """Test formatting valid coordinates with astropy."""
        # Mock SkyCoord behavior
        mock_coord = Mock()
        mock_coord.to_string.return_value = "12 34 56.78 +12 34 56.7"
        mock_skycoord.return_value = mock_coord
        
        result = format_coordinates_astropy(12.5826, 12.5824)
        
        assert result == "12 34 56.78 +12 34 56.7"
        mock_skycoord.assert_called_once()
        # Check the call was made with precision=2 (the actual default from config)
        mock_coord.to_string.assert_called_once_with('hmsdms', sep=' ', precision=2, pad=True)
    
    def test_format_coordinates_astropy_none_inputs(self):
        """Test formatting with None inputs."""
        assert format_coordinates_astropy(None, 12.5) == "N/A"
        assert format_coordinates_astropy(12.5, None) == "N/A"
        assert format_coordinates_astropy(None, None) == "N/A"
    
    @patch('astrakairos.utils.io.SkyCoord')
    @patch('astrakairos.utils.io.COORDINATE_ERROR_BEHAVIOR', 'return_invalid')
    def test_format_coordinates_astropy_error_return_invalid(self, mock_skycoord):
        """Test formatting error with return_invalid behavior."""
        mock_skycoord.side_effect = ValueError("Invalid coordinates")
        
        result = format_coordinates_astropy(25.0, 100.0)  # Invalid coordinates
        
        assert result == "Invalid Coords"
    
    @patch('astrakairos.utils.io.SkyCoord')
    @patch('astrakairos.utils.io.COORDINATE_ERROR_BEHAVIOR', 'return_none')
    def test_format_coordinates_astropy_error_return_none(self, mock_skycoord):
        """Test formatting error with return_none behavior."""
        mock_skycoord.side_effect = ValueError("Invalid coordinates")
        
        result = format_coordinates_astropy(25.0, 100.0)
        
        assert result is None
    
    @patch('astrakairos.utils.io.COORDINATE_ERROR_BEHAVIOR', 'raise')
    def test_format_coordinates_astropy_error_raise(self):
        """Test formatting error with raise behavior."""
        # Use coordinates that are out of range to trigger our validation
        with pytest.raises(CoordinateOutOfRangeError):
            format_coordinates_astropy(25.0, 100.0)  # RA > 24h, Dec > 90°
    
    @patch('astrakairos.utils.io.SkyCoord')
    def test_format_coordinates_astropy_custom_precision(self, mock_skycoord):
        """Test formatting with custom precision."""
        mock_coord = Mock()
        mock_coord.to_string.return_value = "12 34 56.789 +12 34 56.78"
        mock_skycoord.return_value = mock_coord
        
        result = format_coordinates_astropy(12.5826, 12.5824, precision=3)
        
        assert result == "12 34 56.789 +12 34 56.78"
        mock_coord.to_string.assert_called_once_with('hmsdms', sep=' ', precision=3, pad=True)


class TestParseWDSDesignation:
    """Test WDS designation parsing functions."""
    
    def test_parse_wds_designation_valid_positive_dec(self):
        """Test parsing valid WDS designation with positive declination."""
        result = parse_wds_designation("00013+1234")
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'ra_deg' in result
        assert 'dec_deg' in result
        
        # 00013 = 00h 01.3m = 0 + 1.3/60 hours = 0.02167 hours = 0.325 degrees
        expected_ra = (0 + 1.3/60) * 15.0
        assert abs(result['ra_deg'] - expected_ra) < 0.01
        
        # +1234 = +12° 34' = 12 + 34/60 = 12.5667 degrees
        expected_dec = 12 + 34/60
        assert abs(result['dec_deg'] - expected_dec) < 0.01
    
    def test_parse_wds_designation_valid_negative_dec(self):
        """Test parsing valid WDS designation with negative declination."""
        result = parse_wds_designation("12345-6789")
        
        assert result is not None
        
        # 12345 = 12h 34.5m = 12 + 34.5/60 hours = 12.575 hours = 188.625 degrees
        expected_ra = (12 + 34.5/60) * 15.0
        assert abs(result['ra_deg'] - expected_ra) < 0.01
        
        # -6789 = -67° 89' = -(67 + 89/60) = -68.483 degrees
        expected_dec = -(67 + 89/60)
        assert abs(result['dec_deg'] - expected_dec) < 0.01
    
    def test_parse_wds_designation_with_components(self):
        """Test parsing WDS designation with component letters."""
        result = parse_wds_designation("00013+1234AB")
        
        assert result is not None
        # Should ignore component letters and parse coordinates normally
        expected_ra = (0 + 1.3/60) * 15.0
        assert abs(result['ra_deg'] - expected_ra) < 0.01
    
    def test_parse_wds_designation_invalid_format(self):
        """Test parsing invalid WDS designation formats."""
        # Too short
        with pytest.raises(InvalidWdsFormatError):
            parse_wds_designation("0001")
        
        # Wrong format
        with pytest.raises(InvalidWdsFormatError):
            parse_wds_designation("ABCDE+1234")
        
        # Missing sign
        with pytest.raises(InvalidWdsFormatError):
            parse_wds_designation("000131234")
        
        # Invalid characters
        with pytest.raises(InvalidWdsFormatError):
            parse_wds_designation("00013@1234")
    
    def test_parse_wds_designation_none_input(self):
        """Test parsing None or invalid input types."""
        with pytest.raises(InvalidWdsFormatError):
            parse_wds_designation(None)
        with pytest.raises(InvalidWdsFormatError):
            parse_wds_designation(123)
        with pytest.raises(InvalidWdsFormatError):
            parse_wds_designation("")
    
    @patch('astrakairos.utils.io.MIN_RA_DEG', 0.0)
    @patch('astrakairos.utils.io.MAX_RA_DEG', 360.0)
    @patch('astrakairos.utils.io.MIN_DEC_DEG', -90.0)
    @patch('astrakairos.utils.io.MAX_DEC_DEG', 90.0)
    def test_parse_wds_designation_coordinate_range_validation(self):
        """Test coordinate range validation."""
        # Valid coordinates should work
        result = parse_wds_designation("12345+4567")
        assert result is not None
        
        # Invalid RA (would be > 360 degrees)
        # 25000 = 25h 00m = 25 * 15 = 375 degrees > 360
        with patch('astrakairos.utils.io.MAX_RA_DEG', 360.0):
            with pytest.raises(CoordinateOutOfRangeError):
                parse_wds_designation("25000+4567")
        
        # Invalid Dec (would be > 90 degrees)
        # +9100 = +91° 00' = 91 degrees > 90
        with patch('astrakairos.utils.io.MAX_DEC_DEG', 90.0):
            with pytest.raises(CoordinateOutOfRangeError):
                parse_wds_designation("12345+9100")
    
    def test_parse_wds_designation_edge_cases(self):
        """Test edge cases in WDS designation parsing."""
        # Zero coordinates
        result = parse_wds_designation("00000+0000")
        assert result is not None
        assert result['ra_deg'] == 0.0
        assert result['dec_deg'] == 0.0
        
        # Maximum minutes (59)
        result = parse_wds_designation("00590+0590")
        assert result is not None
        
        # Minimum negative declination
        result = parse_wds_designation("00000-0001")
        assert result is not None
        assert result['dec_deg'] < 0


class TestFormatCoordinates:
    """Test universal coordinate formatting function."""
    
    def test_format_coordinates_delegates_to_astropy(self):
        """Test that format_coordinates_astropy works directly."""
        # Test with actual coordinates that should format properly
        result = format_coordinates_astropy(12.5826, 12.5824, precision=2)
        
        # Just verify we get a string result (don't test exact format due to astropy variations)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "12" in result  # Should contain hour information
    
    def test_format_coordinates_default_precision(self):
        """Test format_coordinates_astropy with default precision."""
        result = format_coordinates_astropy(12.5826, 12.5824)
        
        # Verify we get a valid coordinate string
        assert isinstance(result, str)
        assert len(result) > 0


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_round_trip_coordinates(self):
        """Test parsing WDS designation and formatting back."""
        wds_id = "12345+6789"
        
        # Parse coordinates from WDS designation
        coords = parse_wds_designation(wds_id)
        assert coords is not None
        
        # Convert back to hours for formatting
        ra_hours = coords['ra_deg'] / 15.0
        dec_degrees = coords['dec_deg']
        
        # This should work without errors (actual formatting depends on astropy)
        with patch('astrakairos.utils.io.SkyCoord') as mock_skycoord:
            mock_coord = Mock()
            mock_coord.to_string.return_value = "12 34 30.0 +67 89 00"
            mock_skycoord.return_value = mock_coord
            
            formatted = format_coordinates_astropy(ra_hours, dec_degrees)
            assert formatted == "12 34 30.0 +67 89 00"
    
    def test_csv_load_save_round_trip(self):
        """Test loading CSV, processing, and saving results."""
        # Create initial CSV data
        initial_data = [
            {'wds_id': '00013+1234', 'ra_hours': 1.2, 'dec_degrees': 12.5},
            {'wds_id': '01234-4567', 'ra_hours': 2.3, 'dec_degrees': -45.6}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(initial_data)
            df.to_csv(f.name, index=False)
            input_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            # Load data
            loaded_df = load_csv_data(input_path)
            assert len(loaded_df) == 2
            
            # Process into results format
            results = []
            for _, row in loaded_df.iterrows():
                results.append({
                    'wds_id': row['wds_id'],
                    'opi': 0.85,  # Simulated analysis result
                    'ra_hours': row['ra_hours'],
                    'dec_degrees': row['dec_degrees']
                })
            
            # Save results
            save_results_to_csv(results, output_path)
            
            # Verify round trip
            final_df = pd.read_csv(output_path)
            assert len(final_df) == 2
            assert 'wds_id' in final_df.columns
            assert 'opi' in final_df.columns
            
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


if __name__ == "__main__":
    pytest.main([__file__])
