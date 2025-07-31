"""
Tests for astrakairos.data.source module.

Tests the base DataSource class and its abstract interface.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from abc import ABC
from typing import List, Dict, Any, Optional

from astrakairos.data.source import DataSource, WdsSummary


class TestDataSourceAbstractInterface:
    """Test the abstract interface of DataSource."""
    
    def test_data_source_is_abstract(self):
        """Test that DataSource cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataSource()
    
    def test_data_source_inheritance(self):
        """Test that DataSource properly inherits from ABC."""
        assert issubclass(DataSource, ABC)
        assert hasattr(DataSource, '__abstractmethods__')
    
    def test_abstract_methods_defined(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = DataSource.__abstractmethods__
        
        # Check that expected abstract methods are present
        expected_methods = {
            'get_all_measurements',
            'get_orbital_elements', 
            'get_wds_summary'
        }
        
        for method in expected_methods:
            assert method in abstract_methods, f"Method {method} should be abstract"


class ConcreteDataSource(DataSource):
    """Concrete implementation of DataSource for testing."""
    
    def __init__(self):
        self.measurements_data = []
        self.orbital_data = {}
        self.wds_summary_data = {}
    
    async def get_all_component_pairs(self, wds_id: str) -> List[WdsSummary]:
        """Mock implementation returning test component pairs."""
        # Return a single WdsSummary for simplicity
        if self.wds_summary_data:
            return [self.wds_summary_data]
        else:
            return []
    
    async def get_all_measurements(self, wds_id: str, **kwargs):
        """Mock implementation returning test measurements."""
        # Return astropy Table-like object
        from astropy.table import Table
        import numpy as np
        
        if not self.measurements_data:
            return None
            
        # Convert measurements data to Table
        table = Table()
        if self.measurements_data:
            epochs = [m.get('epoch', 0) for m in self.measurements_data]
            pas = [m.get('pa', 0) for m in self.measurements_data]
            seps = [m.get('sep', 0) for m in self.measurements_data]
            
            table['epoch'] = epochs
            table['theta'] = pas
            table['rho'] = seps
            
        return table
    
    async def get_orbital_elements(self, wds_id: str, **kwargs):
        """Mock implementation returning test orbital elements."""
        return self.orbital_data if self.orbital_data else None
    
    async def get_wds_summary(self, wds_id: str, **kwargs):
        """Mock implementation returning test WDS summary."""
        return self.wds_summary_data if self.wds_summary_data else None


class TestConcreteDataSource:
    """Test concrete implementation behavior."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.source = ConcreteDataSource()
    
    def test_concrete_instantiation(self):
        """Test that concrete implementation can be instantiated."""
        source = ConcreteDataSource()
        assert isinstance(source, DataSource)
        assert isinstance(source, ConcreteDataSource)
    
    @pytest.mark.asyncio
    async def test_get_all_measurements_interface(self):
        """Test get_all_measurements method interface."""
        # Test with empty data
        measurements = await self.source.get_all_measurements("12345+6789")
        assert measurements is None
        
        # Test with mock data
        test_measurements = [
            {'epoch': 2020.0, 'pa': 45.0, 'sep': 1.5},
            {'epoch': 2021.0, 'pa': 46.0, 'sep': 1.4}
        ]
        self.source.measurements_data = test_measurements
        
        measurements = await self.source.get_all_measurements("12345+6789")
        assert measurements is not None
        assert 'epoch' in measurements.colnames
        assert 'theta' in measurements.colnames
        assert 'rho' in measurements.colnames
        assert len(measurements) == 2
    
    @pytest.mark.asyncio
    async def test_get_orbital_elements_interface(self):
        """Test get_orbital_elements method interface."""
        # Test with empty data
        orbital = await self.source.get_orbital_elements("12345+6789")
        assert orbital is None
        
        # Test with mock data
        test_orbital = {
            'P': 100.0,
            'a': 1.5,
            'e': 0.3,
            'i': 45.0,
            'Omega': 120.0,
            'omega': 30.0,
            'T': 2020.0
        }
        self.source.orbital_data = test_orbital
        
        orbital = await self.source.get_orbital_elements("12345+6789")
        assert orbital == test_orbital
        assert isinstance(orbital, dict)
    
    @pytest.mark.asyncio
    async def test_get_wds_summary_interface(self):
        """Test get_wds_summary method interface."""
        # Test with empty data
        summary = await self.source.get_wds_summary("12345+6789")
        assert summary is None
        
        # Test with mock data
        test_summary = {
            'wds_id': '12345+6789',
            'ra': 123.45,
            'dec': 67.89,
            'mag_pri': 8.5,
            'mag_sec': 9.2
        }
        self.source.wds_summary_data = test_summary
        
        summary = await self.source.get_wds_summary("12345+6789")
        assert summary == test_summary
        assert isinstance(summary, dict)
    
    def test_method_signatures(self):
        """Test that all methods have correct signatures."""
        import inspect
        
        # Test get_all_measurements signature
        sig = inspect.signature(self.source.get_all_measurements)
        params = list(sig.parameters.keys())
        assert 'wds_id' in params
        assert 'kwargs' in params
        
        # Test get_orbital_elements signature
        sig = inspect.signature(self.source.get_orbital_elements)
        params = list(sig.parameters.keys())
        assert 'wds_id' in params
        assert 'kwargs' in params
        
        # Test get_wds_summary signature
        sig = inspect.signature(self.source.get_wds_summary)
        params = list(sig.parameters.keys())
        assert 'wds_id' in params
        assert 'kwargs' in params


class TestDataSourceErrorHandling:
    """Test error handling in DataSource implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.source = ConcreteDataSource()
    
    @pytest.mark.asyncio
    async def test_invalid_wds_id_handling(self):
        """Test handling of invalid WDS IDs."""
        # Test with None
        result = await self.source.get_all_measurements(None)
        assert result is None
        
        # Test with empty string
        result = await self.source.get_all_measurements("")
        assert result is None
        
        # Test with invalid format
        result = await self.source.get_all_measurements("invalid")
        assert result is None


class TestDataSourceIntegration:
    """Integration tests for DataSource with other components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.source = ConcreteDataSource()
    
    @pytest.mark.asyncio
    async def test_measurements_format_compatibility(self):
        """Test that measurements format is compatible with physics modules."""
        # Set up realistic measurement data
        measurements = [
            {
                'epoch': 2020.0,
                'pa': 45.0,
                'sep': 1.5
            },
            {
                'epoch': 2021.0,
                'pa': 46.0,
                'sep': 1.4
            }
        ]
        self.source.measurements_data = measurements
        
        result = await self.source.get_all_measurements("12345+6789")
        
        # Verify structure expected by physics modules
        assert 'epoch' in result.colnames
        assert 'theta' in result.colnames  # PA in source interface
        assert 'rho' in result.colnames    # Sep in source interface
        
        for i in range(len(result)):
            assert isinstance(result['epoch'][i], (int, float))
            assert isinstance(result['theta'][i], (int, float))
            assert isinstance(result['rho'][i], (int, float))
    
    @pytest.mark.asyncio
    async def test_orbital_elements_format_compatibility(self):
        """Test that orbital elements format is compatible with physics modules."""
        # Set up realistic orbital elements
        orbital = {
            'P': 100.0,      # Period in years
            'a': 1.5,        # Semi-major axis in arcsec
            'e': 0.3,        # Eccentricity
            'i': 45.0,       # Inclination in degrees
            'Omega': 120.0,  # Longitude of ascending node in degrees
            'omega': 30.0,   # Argument of periastron in degrees
            'T': 2020.0      # Time of periastron in years
        }
        self.source.orbital_data = orbital
        
        result = await self.source.get_orbital_elements("12345+6789")
        
        # Verify structure expected by physics modules
        required_keys = ['P', 'a', 'e', 'i', 'Omega', 'omega', 'T']
        for key in required_keys:
            assert key in result, f"Missing required orbital element: {key}"
            assert isinstance(result[key], (int, float)), f"Invalid type for {key}"


if __name__ == "__main__":
    pytest.main([__file__])
