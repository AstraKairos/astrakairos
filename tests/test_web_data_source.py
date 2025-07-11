# tests/test_data_web_source.py
import pytest
import numpy as np
import aiohttp
from unittest.mock import AsyncMock
import os
from astrakairos.data.web_source import StelleDoppieDataSource

# Ruta al archivo HTML de muestra
SAMPLE_ORB_HTML_PATH = "tests/sample_data/BU_773_AB.html"

# CORRECTED FIXTURE: It is just an async function, not a special pytest fixture.
async def get_web_source_instance():
    """
    Async helper that returns an instance of StelleDoppieDataSource with a mocked session.
    """
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    return StelleDoppieDataSource(mock_session)

def load_sample_html(file_path):
    """
    Helper to load HTML content from a file, using a robust encoding.
    """
    if not os.path.exists(file_path):
        pytest.skip(f"Sample HTML file not found: {file_path}")
    # Use 'latin-1' encoding which is robust against decoding errors for most web pages.
    with open(file_path, 'r', encoding='latin-1') as f:
        return f.read()

@pytest.mark.asyncio
async def test_extract_orbital_elements_success():
    """
    Tests if _extract_orbital_elements correctly parses orbital data from a sample HTML.
    """
    # CORRECTED: Await the async helper function to get the instance
    web_source = await get_web_source_instance()
    
    sample_html = load_sample_html(SAMPLE_ORB_HTML_PATH)
    
    orbital_elements = web_source._extract_orbital_elements(sample_html)

    assert orbital_elements is not None
    assert isinstance(orbital_elements, dict)
    
    # --- EXPECTED VALUES BASED ON THE IMAGE ---
    assert np.isclose(orbital_elements['P'], 26.28, atol=0.001) 
    assert np.isclose(orbital_elements['T'], 1989.4, atol=0.001) 
    assert np.isclose(orbital_elements['a'], 0.83, atol=0.001)
    assert np.isclose(orbital_elements['e'], 0.38, atol=0.001)
    assert np.isclose(orbital_elements['i'], 49.0, atol=0.001)
    assert np.isclose(orbital_elements['omega'], 96.0, atol=0.001)
    assert np.isclose(orbital_elements['Omega'], 290.0, atol=0.001)
    
    assert len(orbital_elements) == 7
    assert all(val is not None for val in orbital_elements.values())


@pytest.mark.asyncio
async def test_extract_orbital_elements_no_orbit():
    """
    Tests if _extract_orbital_elements returns an empty dict for HTML without orbital data.
    """
    # CORRECTED: Await the async helper function to get the instance
    web_source = await get_web_source_instance()

    sample_html_no_orbit = "<html><body><h1>Star XYZ</h1><p>No orbital elements found.</p></body></html>"
    
    orbital_elements = web_source._extract_orbital_elements(sample_html_no_orbit)
    
    assert orbital_elements is not None
    assert isinstance(orbital_elements, dict)
    assert all(val is None for val in orbital_elements.values())