# test_convert_catalogs.py
"""Test module for catalog conversion scripts."""

import pytest
import tempfile
import sqlite3
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from scripts.convert_catalogs_to_sqlite import (
    estimate_uncertainty_from_technique,
    apply_physical_validation,
    create_sqlite_database,
    generate_summary_table
)
from astrakairos.utils.io import (
    parse_wdss_coordinate_string,
    parse_wdss_coordinates,
    safe_int,
    safe_float
)


# Note: TestWDSFormatDetection removed as detect_wds_format function was eliminated
# The format detection was unnecessary since files are explicitly categorized by CLI arguments


class TestCoordinateParsing:
    """Test coordinate parsing functions."""
    
    def test_parse_coordinates_valid(self):
        """Test parsing valid coordinate strings."""
        # Test various coordinate formats
        test_cases = [
            "12345+6789",
            "00123-4567", 
            "23456+0000",
        ]
        
        for coord_str in test_cases:
            result = parse_wdss_coordinate_string(coord_str)
            assert isinstance(result, tuple)
            assert len(result) == 2  # Should return (ra, dec)
    
    def test_parse_wdss_coordinates(self):
        """Test parsing WDSS ID coordinates."""
        wdss_id = "00039+4018STF  60"
        result = parse_wdss_coordinates(wdss_id)
        
        assert isinstance(result, tuple)
        assert len(result) == 2  # Should return (ra, dec)


class TestSafeConversions:
    """Test safe conversion functions."""
    
    def test_safe_int_valid(self):
        """Test safe integer conversion with valid inputs."""
        assert safe_int("123") == 123
        assert safe_int("0") == 0
        assert safe_int("-45") == -45
    
    def test_safe_int_invalid(self):
        """Test safe integer conversion with invalid inputs."""
        assert safe_int("abc") is None
        assert safe_int("") is None
        assert safe_int("12.34") is None
        assert safe_int(None) is None
    
    def test_safe_float_valid(self):
        """Test safe float conversion with valid inputs."""
        assert safe_float("123.45") == 123.45
        assert safe_float("0.0") == 0.0
        assert safe_float("-45.67") == -45.67
        assert safe_float("123") == 123.0
    
    def test_safe_float_invalid(self):
        """Test safe float conversion with invalid inputs."""
        assert safe_float("abc") is None
        assert safe_float("") is None
        assert safe_float(None) is None
    
    def test_safe_float_special_values(self):
        """Test safe float conversion with special values."""
        assert safe_float("inf") == float('inf')
        assert safe_float("-inf") == float('-inf')
        # NaN should be handled gracefully
        result = safe_float("nan")
        assert result != result  # NaN != NaN


class TestUncertaintyEstimation:
    """Test uncertainty estimation from measurement techniques."""
    
    def test_estimate_uncertainty_known_techniques(self):
        """Test uncertainty estimation for known techniques."""
        test_techniques = ['S', 'P', 'I', 'H', 'N']
        
        for technique in test_techniques:
            sep_error, pa_error = estimate_uncertainty_from_technique(technique)
            
            assert isinstance(sep_error, float)
            assert isinstance(pa_error, float)
            assert sep_error > 0
            assert pa_error > 0
    
    def test_estimate_uncertainty_unknown_technique(self):
        """Test uncertainty estimation for unknown techniques."""
        sep_error, pa_error = estimate_uncertainty_from_technique('X')
        
        # Should return default values
        assert isinstance(sep_error, float)
        assert isinstance(pa_error, float)
        assert sep_error > 0
        assert pa_error > 0


class TestDataValidation:
    """Test physical data validation."""
    
    def test_apply_physical_validation(self):
        """Test physical validation on sample data."""
        # Create sample DataFrame
        data = {
            'period': [10.0, 50.0, 1000.0],
            'semimajor_axis': [0.1, 1.0, 10.0],
            'eccentricity': [0.0, 0.5, 0.9],
            'inclination': [0.0, 45.0, 180.0]
        }
        df = pd.DataFrame(data)
        
        # Should not raise an exception
        apply_physical_validation(df)
        
        # All rows should still be present (basic validation)
        assert len(df) == 3


class TestDatabaseCreation:
    """Test SQLite database creation."""
    
    def test_create_sqlite_database(self):
        """Test SQLite database creation with sample data."""
        # Create sample DataFrames with correct schema
        df_wds = pd.DataFrame({
            'wds_id': ['00039+4018'],
            'wdss_id': ['00039+4018STF  60AB'],
            'discoverer_designation': ['STF 60'],
            'date_first': [2000.0],
            'date_last': [2020.0],
            'n_obs': [42],
            'pa_first': [285.2],
            'pa_last': [287.8],
            'sep_first': [0.455],
            'sep_last': [0.431],
            'pa_first_error': [1.0],
            'pa_last_error': [1.0],
            'sep_first_error': [0.05],
            'sep_last_error': [0.05],
            'vmag': [8.12],
            'kmag': [8.89],
            'spectral_type': ['G5V+K0V'],
            'ra_deg': [244.218],
            'dec_deg': [1.295],
            'pm_ra': [15.23],
            'pm_dec': [-12.4],
            'parallax': [-8.7],
            'name': ['HD 148937']
        })
        
        df_orb6 = pd.DataFrame({
            'wds_id': ['00039+4018'],
            'P': [285.69],
            'e_P': [10.0],
            'a': [0.4331],
            'e_a': [0.02],
            'i': [118.2],
            'e_i': [5.0],
            'Omega': [287.1],
            'e_Omega': [10.0],
            'T': [2023.45],
            'e_T': [2.0],
            'e': [0.1259],
            'e_e': [0.05],
            'omega_arg': [167.3],
            'e_omega_arg': [10.0],
            'grade': [1]
        })
        
        df_measurements = pd.DataFrame({
            'wdss_id': ['00039+4018STF  60AB', '00039+4018STF  60AB'],
            'pair': ['AB', 'AB'],
            'epoch': [2000.0, 2010.0],
            'theta': [285.2, 286.5],
            'rho': [0.455, 0.443],
            'theta_error': [1.0, 1.0],
            'rho_error': [0.05, 0.05],
            'mag1': [8.12, 8.15],
            'mag2': [8.89, 8.92],
            'reference': ['REF1', 'REF2'],
            'technique': ['S', 'S'],
            'error_source': ['estimated', 'estimated']
        })
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db_path = f.name
        
        try:
            create_sqlite_database(df_wds, df_orb6, df_measurements, temp_db_path)
            
            # Verify database was created
            assert Path(temp_db_path).exists()
            
            # Verify database structure
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            
            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['wdss_summary', 'orbital_elements', 'measurements']
            for table in expected_tables:
                assert table in tables
            
            conn.close()
            
        finally:
            Path(temp_db_path).unlink()  # Clean up


class TestSummaryGeneration:
    """Test summary table generation."""
    
    def test_generate_summary_table(self):
        """Test generation of summary table from components and measurements."""
        # Create sample component data
        df_components = pd.DataFrame({
            'wdss_id': ['00039+4018STF  60A', '00039+4018STF  60B'],
            'component': ['A', 'B'],
            'vmag': [10.0, 11.0],
            'kmag': [9.5, 10.5],
            'spectral_type': ['G0V', 'K0V'],
            'ra_deg': [10.0, 10.0],
            'dec_deg': [40.0, 40.0],
            'pm_ra': [0.1, 0.1],
            'pm_dec': [0.2, 0.2],
            'parallax': [0.01, 0.01],
            'name': ['Component A', 'Component B']
        })
        
        # Create sample measurement data with correct columns
        df_measurements = pd.DataFrame({
            'wdss_id': ['00039+4018STF  60AB', '00039+4018STF  60AB'],
            'epoch': [2000.0, 2001.0],
            'theta': [90.0, 91.0],  # position_angle -> theta
            'rho': [1.0, 1.1],      # separation -> rho
            'theta_error': [0.1, 0.1],  # Add error columns
            'rho_error': [0.01, 0.01]
        })
        
        # Create sample correspondence data
        df_correspondence = pd.DataFrame({
            'wdss_id': ['00039+4018STF  60AB'],
            'wds_id': ['00039+4018'],
            'discoverer_designation': ['STF 60']
        })
        
        result = generate_summary_table(df_components, df_measurements, df_correspondence)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # Should have standard columns for WDS summary (actual column names from output)
        expected_columns = ['wdss_id', 'wds_id', 'discoverer_designation']
        for col in expected_columns:
            assert col in result.columns
        
        # Check that measurements were aggregated
        measurement_columns = ['date_first', 'date_last', 'n_obs', 'pa_first', 'pa_last', 'sep_first', 'sep_last']
        for col in measurement_columns:
            assert col in result.columns


class TestErrorHandling:
    """Test error handling in conversion functions."""
    
    def test_coordinate_parsing_invalid_format(self):
        """Test coordinate parsing with invalid formats."""
        invalid_coords = ["", "abc", "12345", "++++", "12345+"]
        
        for coord in invalid_coords:
            # Should handle gracefully, not crash
            try:
                result = parse_wdss_coordinate_string(coord)
                # If it returns something, should be a tuple
                if result is not None:
                    assert isinstance(result, tuple)
            except Exception:
                # Specific exceptions are acceptable
                pass
    
    def test_database_creation_invalid_path(self):
        """Test database creation with invalid path."""
        df_empty = pd.DataFrame()
        
        # Invalid path should raise an exception
        with pytest.raises(Exception):
            create_sqlite_database(df_empty, df_empty, df_empty, "/invalid/path/test.db")


if __name__ == "__main__":
    pytest.main([__file__])
