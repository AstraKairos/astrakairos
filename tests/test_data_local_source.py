# tests/test_data_local_source.py
import pytest
from astrakairos.data.local_source import LocalFileDataSource

# Define las rutas a los archivos de prueba para no repetirlas
WDS_SAMPLE_PATH = "tests/sample_data/wds_sample.txt"
ORB6_SAMPLE_PATH = "tests/sample_data/orb6_sample.txt"

# Usamos @pytest.fixture para crear un objeto 'source' reutilizable para todas las pruebas
@pytest.fixture
def local_source():
    """Crea una instancia de LocalFileDataSource para las pruebas."""
    # Asegúrate de que los archivos de muestra existen antes de correr
    return LocalFileDataSource(wds_filepath=WDS_SAMPLE_PATH, orb6_filepath=ORB6_SAMPLE_PATH)

@pytest.mark.asyncio
async def test_load_catalogs(local_source):
    """Verifica que los catálogos se cargan sin errores y no están vacíos."""
    assert local_source.wds_df is not None
    assert not local_source.wds_df.empty
    assert local_source.orb6_df is not None
    assert not local_source.orb6_df.empty

@pytest.mark.asyncio
async def test_get_wds_data_for_known_star(local_source):
    """Verifica que podemos obtener datos para una estrella que está en el archivo de muestra."""
    # Reemplaza 'ID_EN_TU_MUESTRA' con un WDS ID real de tu archivo wds_sample.txt
    known_id = "00000+7530A"
    data = await local_source.get_wds_data(known_id)
    assert data is not None
    assert isinstance(data, dict)
    assert 'date_last' in data  # Verifica que una columna clave exista

@pytest.mark.asyncio
async def test_get_orbital_elements_for_known_star(local_source):
    """Verifica que podemos obtener datos orbitales para una estrella de la muestra."""
    # Reemplaza 'ID_EN_TU_MUESTRA_ORB6' con un WDS ID real de tu archivo orb6_sample.txt
    known_id = "00000-1930"
    data = await local_source.get_orbital_elements(known_id)
    assert data is not None
    assert 'P' in data  # Verifica que una columna clave de órbita exista

@pytest.mark.asyncio
async def test_get_data_for_nonexistent_star(local_source):
    """Verifica que devuelve un diccionario vacío para una estrella que no existe."""
    data = await local_source.get_wds_data("99999-9999")
    assert data == {}