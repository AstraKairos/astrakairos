"""
Test El-Badry catalog integration functionality.

This module tests the cross-matching functionality with the El-Badry et al. (2021)
high-confidence binary catalog and the CLI filtering capabilities.
"""

import pytest
import pandas as pd
import tempfile
import os
import sys
from unittest.mock import patch, Mock
from pathlib import Path

# Add scripts directory to path for conversion tool imports
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from conversion_tool.parsers import parse_el_badry_catalog
from conversion_tool.crossmatching import perform_el_badry_crossmatch


class TestElBadryCrossMatch:
    """Test cross-matching with El-Badry catalog."""

    def test_cross_match_success(self):
        """Test successful cross-match with El-Badry catalog."""
        # Mock WDSS components data with proper A/B pairs and valid Gaia IDs
        df_components = pd.DataFrame({
            'wdss_id': ['00001+1234', '00001+1234', '00002+5678', '00002+5678'],
            'component': ['A', 'B', 'A', 'B'],
            'name': ['Gaia DR3 1234567890123456789', 'Gaia DR3 9999999999999999999', 'Other Name', 'Gaia EDR3 9876543210987654321']
        })
        
        # Mock El-Badry catalog data with matching pairs
        mock_el_badry_data = pd.DataFrame({
            'source_id1': [1234567890123456789, 9876543210987654321],
            'source_id2': [9999999999999999999, 4444444444444444444],
            'R_chance_align': [0.1, 0.05],
            'binary_type': ['SB2', 'Visual']
        })
        
        with patch('astropy.table.Table.read') as mock_read:
            # Configure mock to return our test data
            mock_table = Mock()
            mock_el_badry_raw = pd.DataFrame({
                'source_id1': [1234567890123456789, 9876543210987654321],
                'source_id2': [9999999999999999999, 4444444444444444444],
                'R_chance_align': [0.1, 0.05],
                'binary_type': ['SB2', 'Visual']
            })
            mock_table.to_pandas.return_value = mock_el_badry_raw
            mock_read.return_value = mock_table
            
            # Run cross-match using the new two-step approach
            df_el_badry = parse_el_badry_catalog('fake_path.fits')
            result = perform_el_badry_crossmatch(df_components, df_el_badry)
            
            # Verify results - updated for refactored function
            assert not result.empty
            assert 'wdss_id' in result.columns
            assert 'in_el_badry_catalog' in result.columns
            assert 'R_chance_align' in result.columns
            assert 'binary_type' in result.columns
            assert len(result) == 1  # Only one system should match
            assert result.iloc[0]['wdss_id'] == '00001+1234'
            assert result.iloc[0]['in_el_badry_catalog'] == True
            assert result.iloc[0]['R_chance_align'] == 0.1
            assert result.iloc[0]['binary_type'] == 'SB2'

    def test_cross_match_no_matches(self):
        """Test cross-match when no systems match."""
        # Mock WDSS components with no Gaia IDs that match El-Badry
        df_components = pd.DataFrame({
            'wdss_id': ['00001+1234', '00002+5678'],
            'component': ['A', 'A'],
            'name': ['Other Name', 'Another Name']
        })
        
        # Mock El-Badry catalog data
        mock_el_badry_data = pd.DataFrame({
            'source_id1': [111111111, 222222222],
            'source_id2': [333333333, 444444444]
        })
        
        with patch('astropy.table.Table.read') as mock_read:
            mock_table = Mock()
            mock_table.to_pandas.return_value = mock_el_badry_data
            mock_read.return_value = mock_table
            
            df_el_badry = parse_el_badry_catalog('fake_path.fits')
            result = perform_el_badry_crossmatch(df_components, df_el_badry)
            
            # Should return empty DataFrame with correct columns
            assert result.empty
            assert 'wdss_id' in result.columns
            assert 'in_el_badry_catalog' in result.columns

    def test_cross_match_missing_astropy(self):
        """Test cross-match when astropy is not available."""
        df_components = pd.DataFrame({
            'wdss_id': ['00001+1234'],
            'component': ['A'],
            'name': ['Gaia DR3 123456789']
        })
        
        # Mock the import inside the function to raise ImportError
        import builtins
        original_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'astropy.table':
                raise ImportError("No module named 'astropy'")
            return original_import(name, *args, **kwargs)
        
        with patch.object(builtins, '__import__', side_effect=mock_import):
            # Should raise CatalogParsingError
            from astrakairos.exceptions import CatalogParsingError
            with pytest.raises(CatalogParsingError):
                parse_el_badry_catalog('fake_path.fits')

    def test_cross_match_file_error(self):
        """Test cross-match when El-Badry file cannot be read."""
        df_components = pd.DataFrame({
            'wdss_id': ['00001+1234'],
            'component': ['A'],
            'name': ['Gaia DR3 123456789']
        })
        
        with patch('astropy.table.Table.read') as mock_read:
            # Simulate file read error
            mock_read.side_effect = FileNotFoundError("File not found")
            
            # Should raise CatalogParsingError
            from astrakairos.exceptions import CatalogParsingError
            with pytest.raises(CatalogParsingError):
                parse_el_badry_catalog('nonexistent_path.fits')

    def test_gaia_id_parsing(self):
        """Test various Gaia ID parsing scenarios."""
        df_components = pd.DataFrame({
            'wdss_id': ['00001+1234', '00001+1234', '00002+5678', '00002+5678', '00003+9012', '00003+9012', '00004+3456', '00004+3456'],
            'component': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'name': [
                'Gaia DR3 1234567890123456789',      # Standard DR3 format - A component
                'Gaia DR3 5556667777777777777',      # Standard DR3 format - B component  
                'Gaia EDR3 9876543210987654321',     # EDR3 format - A component
                'Gaia EDR3 8889990000000000000',     # EDR3 format - B component
                'Gaia 1112223334444444444',          # Simple format - A component
                'Gaia 4445556666666666666',          # Simple format - B component
                'HD 12345',                # Non-Gaia name - A component
                'HD 67890'                 # Non-Gaia name - B component
            ]
        })
        
        # Mock El-Badry data with matching pairs
        mock_el_badry_data = pd.DataFrame({
            'source_id1': [1234567890123456789, 9876543210987654321, 1112223334444444444],
            'source_id2': [5556667777777777777, 8889990000000000000, 4445556666666666666],
            'R_chance_align': [0.1, 0.05, 0.2],
            'binary_type': ['SB2', 'Visual', 'Astrometric']
        })
        
        with patch('astropy.table.Table.read') as mock_read:
            mock_table = Mock()
            mock_table.to_pandas.return_value = mock_el_badry_data
            mock_read.return_value = mock_table
            
            df_el_badry = parse_el_badry_catalog('fake_path.fits')
            result = perform_el_badry_crossmatch(df_components, df_el_badry)
            
            # Should match the first 3 systems (all with proper Gaia A/B pairs)
            assert len(result) == 3
            expected_ids = {'00001+1234', '00002+5678', '00003+9012'}
            actual_ids = set(result['wdss_id'])
            assert actual_ids == expected_ids
            
            # Verify each match has the expected columns
            for _, row in result.iterrows():
                assert row['in_el_badry_catalog'] == True
                assert pd.notna(row['R_chance_align'])
                assert pd.notna(row['binary_type'])
