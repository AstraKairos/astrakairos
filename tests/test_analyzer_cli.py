# tests/test_analyzer_cli.py

import pytest
import pandas as pd
from unittest.mock import patch, AsyncMock, MagicMock, Mock
import argparse
from enum import Enum

# Importamos las funciones y clases que vamos a probar y 'mockear'
from astrakairos.analyzer.cli import create_argument_parser, main_async

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

# --- Pruebas para el Argument Parser ---

def test_parser_default_arguments():
    """Test that argument defaults work with required database path."""
    parser = create_argument_parser()
    args = parser.parse_args(['dummy_input.csv', '--database-path', 'test.db'])

    assert args.source == 'local'
    assert args.limit is None
    assert not args.validate_gaia
    assert args.gaia_p_value == 0.01
    assert args.concurrent == 5

def test_parser_custom_arguments():
    """Verifica que se pueden parsear argumentos personalizados correctamente."""
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

# Usamos pytest.mark.asyncio para todas las pruebas que llaman a código async
@pytest.mark.asyncio
# El decorador 'patch' reemplaza objetos con 'Mocks' durante la prueba
@patch('astrakairos.analyzer.cli.load_csv_data')
@patch('astrakairos.analyzer.cli.LocalDataSource')
@patch('astrakairos.analyzer.cli.GaiaValidator')
@patch('astrakairos.analyzer.cli.save_results_to_csv')
async def test_main_async_local_source_flow(mock_save_csv, mock_gaia_validator, mock_local_source, mock_load_csv):
    """
    Prueba de integración para el flujo principal usando la fuente de datos local.
    """
    # --- 1. Configuración de los Mocks (Arrange) ---
    
    # Simula la carga del CSV
    mock_load_csv.return_value = MOCK_CSV_DATA
    
    # Configura la instancia 'mockeada' de LocalDataSource
    mock_local_instance = mock_local_source.return_value
    mock_local_instance.get_wds_summary = AsyncMock(return_value=MOCK_WDS_DATA)
    mock_local_instance.get_orbital_elements = AsyncMock(return_value=MOCK_ORBITAL_DATA)
    mock_local_instance.close = Mock()  # Mock para el método close()
    
    # Configura la instancia 'mockeada' de GaiaValidator
    mock_gaia_instance = mock_gaia_validator.return_value
    mock_gaia_instance.validate_physicality = AsyncMock(return_value=MOCK_GAIA_RESULT)

    # --- 2. Configuración de los Argumentos y Ejecución (Act) ---
    
    # Simula los argumentos de la línea de comandos
    parser = create_argument_parser()
    args_list = [
        'stars.csv',
        '--source', 'local',
        '--database-path', 'catalogs.db',
        '--validate-gaia',
        '--output', 'results.csv'
    ]
    args = parser.parse_args(args_list)

    # Llama a la función principal que estamos probando
    await main_async(args)

    # --- 3. Verificaciones (Assert) ---
    
    # Verifica que las funciones fueron llamadas
    mock_load_csv.assert_called_once_with('stars.csv')
    mock_local_source.assert_called_once_with(database_path='catalogs.db')
    mock_local_instance.get_wds_summary.assert_called_once()
    # Note: In discovery mode, get_orbital_elements is not called
    
    # Verifica que GaiaValidator fue llamado con los argumentos correctos
    mock_gaia_validator.assert_called_once_with(
        physical_p_value_threshold=0.01,
        ambiguous_p_value_threshold=0.001
    )
    mock_gaia_instance.validate_physicality.assert_called_once()
    
    # Verifica que se intentó guardar el resultado
    mock_save_csv.assert_called_once()
    
    # Opcional: inspeccionar los datos que se pasaron a save_results_to_csv
    # Esto verifica que el resultado final del procesamiento es correcto.
    final_results = mock_save_csv.call_args[0][0] # Obtiene el primer argumento posicional de la llamada
    assert len(final_results) == 1
    assert final_results[0]['wds_id'] == '00001+0001'
    assert final_results[0]['physicality_label'] == 'Physical'  # Now using proper enum mock
    assert 'v_total_arcsec_yr' in final_results[0]  # Discovery mode default