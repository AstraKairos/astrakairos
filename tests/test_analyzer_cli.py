# tests/test_analyzer_cli.py

import pytest
import pandas as pd
from unittest.mock import patch, AsyncMock, MagicMock
import argparse

# Importamos las funciones y clases que vamos a probar y 'mockear'
from astrakairos.analyzer.cli import create_argument_parser, main_async

# --- Pruebas para el Argument Parser ---

def test_parser_default_arguments():
    """Verifica que los valores por defecto de los argumentos son los esperados."""
    parser = create_argument_parser()
    args = parser.parse_args(['dummy_input.csv']) # Se necesita un input file posicional

    assert args.source == 'web'
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
        '--wds-file', 'wds.txt',
        '--validate-gaia',
        '--gaia-p-value', '0.05',
        '--limit', '50',
        '--sort-by', 'opi'
    ]
    args = parser.parse_args(args_list)

    assert args.source == 'local'
    assert args.wds_file == 'wds.txt'
    assert args.validate_gaia
    assert args.gaia_p_value == 0.05
    assert args.limit == 50
    assert args.sort_by == 'opi'

# --- Pruebas de Integración para main_async ---

# Datos simulados que devolverían nuestras funciones 'mockeadas'
MOCK_CSV_DATA = pd.DataFrame([{'wds_name': '00001+0001', 'obs': 10}])
MOCK_WDS_DATA = {
    'wds_id': '00001+0001', 'ra_deg': 0.1, 'dec_deg': 0.1,
    'date_first': 2000.0, 'pa_first': 100.0, 'sep_first': 1.0,
    'date_last': 2020.0, 'pa_last': 110.0, 'sep_last': 0.9,
    'mag_pri': 8.0, 'mag_sec': 9.0
}
MOCK_ORBITAL_DATA = {'P': 100, 'T': 2000, 'e': 0.5, 'a': 1, 'i': 45, 'Omega': 90, 'omega': 30}
MOCK_GAIA_RESULT = {'label': 'Likely Physical', 'p_value': 0.99, 'test_used': '3D (plx+pm)'}

# Usamos pytest.mark.asyncio para todas las pruebas que llaman a código async
@pytest.mark.asyncio
# El decorador 'patch' reemplaza objetos con 'Mocks' durante la prueba
@patch('astrakairos.analyzer.cli.load_csv_data')
@patch('astrakairos.analyzer.cli.LocalFileDataSource')
@patch('astrakairos.analyzer.cli.GaiaValidator')
@patch('astrakairos.analyzer.cli.save_results_to_csv')
async def test_main_async_local_source_flow(mock_save_csv, mock_gaia_validator, mock_local_source, mock_load_csv):
    """
    Prueba de integración para el flujo principal usando la fuente de datos local.
    """
    # --- 1. Configuración de los Mocks (Arrange) ---
    
    # Simula la carga del CSV
    mock_load_csv.return_value = MOCK_CSV_DATA
    
    # Configura la instancia 'mockeada' de LocalFileDataSource
    mock_local_instance = mock_local_source.return_value
    mock_local_instance.get_wds_data = AsyncMock(return_value=MOCK_WDS_DATA)
    mock_local_instance.get_orbital_elements = AsyncMock(return_value=MOCK_ORBITAL_DATA)
    
    # Configura la instancia 'mockeada' de GaiaValidator
    mock_gaia_instance = mock_gaia_validator.return_value
    mock_gaia_instance.validate_physicality = AsyncMock(return_value=MOCK_GAIA_RESULT)

    # --- 2. Configuración de los Argumentos y Ejecución (Act) ---
    
    # Simula los argumentos de la línea de comandos
    parser = create_argument_parser()
    args_list = [
        'stars.csv',
        '--source', 'local',
        '--wds-file', 'wds.txt',
        '--orb6-file', 'orb6.txt',
        '--validate-gaia',
        '--output', 'results.csv'
    ]
    args = parser.parse_args(args_list)

    # Llama a la función principal que estamos probando
    await main_async(args)

    # --- 3. Verificaciones (Assert) ---
    
    # Verifica que las funciones fueron llamadas
    mock_load_csv.assert_called_once_with('stars.csv')
    mock_local_source.assert_called_once_with(wds_filepath='wds.txt', orb6_filepath='orb6.txt')
    mock_local_instance.get_wds_data.assert_called_once()
    mock_local_instance.get_orbital_elements.assert_called_once()
    
    # Verifica que GaiaValidator fue llamado con los argumentos correctos
    mock_gaia_validator.assert_called_once_with(p_value_threshold=0.01) # El valor por defecto
    mock_gaia_instance.validate_physicality.assert_called_once()
    
    # Verifica que se intentó guardar el resultado
    mock_save_csv.assert_called_once()
    
    # Opcional: inspeccionar los datos que se pasaron a save_results_to_csv
    # Esto verifica que el resultado final del procesamiento es correcto.
    final_results = mock_save_csv.call_args[0][0] # Obtiene el primer argumento posicional de la llamada
    assert len(final_results) == 1
    assert final_results[0]['wds_name'] == '00001+0001'
    assert final_results[0]['physicality_label'] == 'Likely Physical'
    assert 'opi_arcsec_yr' in final_results[0]