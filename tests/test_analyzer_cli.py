# tests/test_analyzer_cli.py

import pytest
import pandas as pd
from unittest.mock import patch, AsyncMock, MagicMock, Mock
import argparse
from enum import Enum

# Importamos las funciones y clases que vamos a probar y 'mockear'
from astrakairos.analyzer.cli import create_argument_parser, main_async
from astrakairos.config import DEFAULT_GAIA_P_VALUE
from astrakairos.analyzer.engine import AnalyzerRunner

# Mock enum classes to simulate the actual enum behavior
class MockPhysicalityLabel(Enum):
    PHYSICAL = 'Physical'
    AMBIGUOUS = 'Ambiguous'
    NON_PHYSICAL = 'NonPhysical'

class MockPhysicalityMethod(Enum):
    THREE_DIMENSIONAL = 'ThreeDimensional'
    RUWE = 'RUWE'
    PROPER_MOTION = 'ProperMotion'

# Mock object with proper enum behavior
class MockPhysicalityResult:
    def __init__(self):
        self.label = MockPhysicalityLabel.PHYSICAL
        self.method = MockPhysicalityMethod.THREE_DIMENSIONAL
        self.p_value = 0.99
        self.confidence = 0.95

# --- Argument Parser Tests ---

def test_parser_default_arguments():
    """Test that argument defaults work with required database path."""
    parser = create_argument_parser()
    args = parser.parse_args(['dummy_input.csv', '--database-path', 'test.db'])

    assert args.source == 'local'
    assert args.limit is None
    assert not args.validate_gaia
    assert not args.validate_el_badry  # New flag should be False by default
    assert args.gaia_p_value == DEFAULT_GAIA_P_VALUE
    assert args.concurrent == 200  # Updated to current default

def test_parser_custom_arguments():
    """Test that custom arguments can be parsed correctly."""
    parser = create_argument_parser()
    args_list = [
        'my_stars.csv',
        '--source', 'local',
        '--database-path', 'catalogs.db',
        '--validate-gaia',
        '--gaia-p-value', '0.05',
        '--limit', '50'
    ]
    args = parser.parse_args(args_list)

    assert args.source == 'local'
    assert args.database_path == 'catalogs.db'
    assert args.validate_gaia
    assert args.gaia_p_value == 0.05
    assert args.limit == 50

# Integration tests for main_async

# Mock data that would be returned by our mocked functions
MOCK_CSV_DATA = pd.DataFrame([{'wds_id': '00001+0001', 'obs': 10}])
MOCK_WDS_DATA = {
    'wds_id': '00001+0001', 'ra_deg': 0.1, 'dec_deg': 0.1,
    'date_first': 2000.0, 'pa_first': 100.0, 'sep_first': 1.0,
    'date_last': 2020.0, 'pa_last': 110.0, 'sep_last': 0.9,
    'mag_pri': 8.0, 'mag_sec': 9.0
}
MOCK_ORBITAL_DATA = {'P': 100, 'T': 2000, 'e': 0.5, 'a': 1, 'i': 45, 'Omega': 90, 'omega': 30}
# Create mock result that simulates the actual enum behavior  
MOCK_GAIA_RESULT = {
    'label': MockPhysicalityLabel.PHYSICAL,
    'p_value': 0.99,
    'method': MockPhysicalityMethod.THREE_DIMENSIONAL,
    'confidence': 0.95
}

# Mock successful analysis result
MOCK_RESULT = {
    'wds_id': '00001+0001',
    'mode': 'discovery',
    'v_total': 0.005,
    'v_total_uncertainty': 0.001,
    'pa_v_deg': 45.0,
    'physicality_label': 'Physical',
    'physicality_p_value': 0.99
}

# Use pytest.mark.asyncio for all tests that call async code
@pytest.mark.asyncio
# El decorador 'patch' reemplaza objetos con 'Mocks' durante la prueba
@patch('astrakairos.analyzer.cli.load_csv_data')
@patch('astrakairos.analyzer.cli.LocalDataSource')
@patch('astrakairos.analyzer.cli.HybridValidator')
@patch('astrakairos.analyzer.cli.save_results_to_csv')
@patch('astrakairos.analyzer.cli.analyze_stars')
@patch('astrakairos.analyzer.cli.print_error_summary')
async def test_main_async_local_source_flow(mock_print_error_summary, mock_analyze_stars, mock_save_csv, mock_hybrid_validator, mock_local_source, mock_load_csv):
    """
    Prueba de integración para el flujo principal usando la fuente de datos local con validador híbrido.
    """
    # --- 1. Mock Setup (Arrange) ---
    
    # Simula la carga del CSV
    mock_load_csv.return_value = MOCK_CSV_DATA
    
    # Configura la instancia 'mockeada' de LocalDataSource
    mock_local_instance = mock_local_source.return_value
    mock_local_instance.get_wds_summary = AsyncMock(return_value=MOCK_WDS_DATA)
    mock_local_instance.get_orbital_elements = AsyncMock(return_value=MOCK_ORBITAL_DATA)
    mock_local_instance.close = Mock()  # Mock for the close() method
    
    # Configura la instancia 'mockeada' de HybridValidator
    mock_hybrid_instance = mock_hybrid_validator.return_value
    mock_hybrid_instance.validate_binary_physicality = AsyncMock(return_value=MOCK_GAIA_RESULT)
    mock_hybrid_instance.get_cache_statistics = Mock(return_value={'cached_systems': 'unknown'})

    # Mock analyze_stars to return successful results
    mock_analyze_stars.return_value = ([MOCK_RESULT], {})

    # --- 2. Argument Setup and Execution (Act) ---
    parser = create_argument_parser()
    args_list = [
        'stars.csv',
        '--source', 'local',
        '--database-path', 'catalogs.db',
        '--validate-el-badry',  # Changed to el-badry to test HybridValidator creation
        '--output', 'results.csv'
    ]
    args = parser.parse_args(args_list)

    # Call the main function we're testing
    await main_async(args)

    # --- 3. Verificaciones (Assert) ---
    
    # Verifica que las funciones fueron llamadas
    mock_load_csv.assert_called_once_with('stars.csv')
    mock_local_source.assert_called_once_with(database_path='catalogs.db')
    
    # The main test is that the system creates and uses a HybridValidator
    # instead of direct GaiaValidator calls, which is the key architectural change
    mock_hybrid_validator.assert_called_once()
    
    # Verify that analyze_stars was called with the correct parameters
    mock_analyze_stars.assert_called_once()
    
    # Verify that result saving was attempted
    mock_save_csv.assert_called_once()
    
    # Verify basic structure of results (without checking specific validation outcomes
    # since those depend on complex mocking behavior)
    final_results = mock_save_csv.call_args[0][0]
    assert len(final_results) == 1
    assert final_results[0]['wds_id'] == '00001+0001'
    assert 'v_total' in final_results[0]  # Discovery mode default