import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Importamos la clase que vamos a probar
from astrakairos.data.gaia_source import GaiaValidator

# --- Datos de Prueba Simulados (Mocks) ---

# Estrella 1: Datos completos y consistentes
STAR_1_FULL = {
    'source_id': 1, 'phot_g_mean_mag': 10.0,
    'parallax': 10.0, 'parallax_error': 0.1,
    'pmra': 50.0, 'pmra_error': 0.1,
    'pmdec': -20.0, 'pmdec_error': 0.1,
    'parallax_pmra_corr': 0.2, 'parallax_pmdec_corr': 0.1, 'pmra_pmdec_corr': -0.3
}

# Estrella 2: Físicamente asociada a Estrella 1 (datos muy similares)
STAR_2_PHYSICAL = {
    'source_id': 2, 'phot_g_mean_mag': 12.0,
    'parallax': 10.05, 'parallax_error': 0.1,
    'pmra': 50.05, 'pmra_error': 0.1,
    'pmdec': -20.05, 'pmdec_error': 0.1,
    'parallax_pmra_corr': 0.2, 'parallax_pmdec_corr': 0.1, 'pmra_pmdec_corr': -0.3
}

# Estrella 3: Óptica respecto a Estrella 1 (datos muy diferentes)
STAR_3_OPTICAL = {
    'source_id': 3, 'phot_g_mean_mag': 11.0,
    'parallax': 1.0, 'parallax_error': 0.1, # Paralaje 100x diferente
    'pmra': 5.0, 'pmra_error': 0.1,     # PM 10x diferente
    'pmdec': -2.0, 'pmdec_error': 0.1,
    'parallax_pmra_corr': 0.2, 'parallax_pmdec_corr': 0.1, 'pmra_pmdec_corr': -0.3
}

# Estrella 4: Datos incompletos (sin paralaje)
STAR_4_NO_PARALLAX = {
    'source_id': 4, 'phot_g_mean_mag': 13.0,
    'parallax': None, 'parallax_error': None, # Falta paralaje
    'pmra': 50.0, 'pmra_error': 0.1,
    'pmdec': -20.0, 'pmdec_error': 0.1,
    'parallax_pmra_corr': None, 'parallax_pmdec_corr': None, 'pmra_pmdec_corr': -0.3
}

# Estrella 5: Datos incompletos (sin movimiento propio)
STAR_5_NO_PM = {
    'source_id': 5, 'phot_g_mean_mag': 14.0,
    'parallax': 10.0, 'parallax_error': 0.1,
    'pmra': None, 'pmra_error': None, # Falta PM
    'pmdec': None, 'pmdec_error': None,
    'parallax_pmra_corr': None, 'parallax_pmdec_corr': None, 'pmra_pmdec_corr': None
}

# --- Fixture para el Validador ---

@pytest.fixture
def validator():
    """Crea una instancia de GaiaValidator con umbrales estándar para las pruebas."""
    return GaiaValidator(physical_p_value_threshold=0.05, ambiguous_p_value_threshold=0.001)

# --- Pruebas de la Lógica de Validación ---

@pytest.mark.asyncio
# Usamos 'patch' para reemplazar la función que hace la llamada a la red.
# 'autospec=True' asegura que el mock tenga la misma firma que la función real.
@patch('astrakairos.data.gaia_source.GaiaValidator._query_gaia_for_pair', autospec=True)
async def test_validation_physical_pair_3d(mock_query, validator):
    """Verifica que un par físico con datos 3D completos se clasifica correctamente."""
    # Configura el mock para que devuelva nuestras estrellas simuladas
    mock_query.return_value = [STAR_1_FULL, STAR_2_PHYSICAL]
    
    result = await validator.validate_physicality(
        primary_coords_deg=(0, 0), wds_magnitudes=(10.0, 12.0)
    )
    
    assert result['label'] == 'Likely Physical'
    assert result['test_used'] == '3D (plx+pm)'
    assert result['p_value'] > validator.physical_threshold

@pytest.mark.asyncio
@patch('astrakairos.data.gaia_source.GaiaValidator._query_gaia_for_pair', autospec=True)
async def test_validation_optical_pair_3d(mock_query, validator):
    """Verifica que un par óptico con datos 3D completos se clasifica correctamente."""
    mock_query.return_value = [STAR_1_FULL, STAR_3_OPTICAL]
    
    result = await validator.validate_physicality(
        primary_coords_deg=(0, 0), wds_magnitudes=(10.0, 11.0)
    )
    
    assert result['label'] == 'Likely Optical'
    assert result['test_used'] == '3D (plx+pm)'
    assert result['p_value'] < validator.ambiguous_threshold

@pytest.mark.asyncio
@patch('astrakairos.data.gaia_source.GaiaValidator._query_gaia_for_pair', autospec=True)
async def test_adaptive_logic_falls_back_to_2d(mock_query, validator):
    """Prueba que el validador usa la prueba 2D si los datos 3D no están completos."""
    # Estrella 1 tiene datos completos, Estrella 4 no tiene paralaje.
    # El test 3D debe fallar, y debe intentar el test 2D.
    # Las PM de STAR_4_NO_PARALLAX son idénticas a las de STAR_1_FULL, por lo que deberían ser físicas.
    mock_query.return_value = [STAR_1_FULL, STAR_4_NO_PARALLAX]
    
    result = await validator.validate_physicality(
        primary_coords_deg=(0, 0), wds_magnitudes=(10.0, 13.0)
    )
    
    assert result['label'] == 'Likely Physical'
    assert result['test_used'] == '2D (pm_only)'
    assert result['p_value'] > validator.physical_threshold

@pytest.mark.asyncio
@patch('astrakairos.data.gaia_source.GaiaValidator._query_gaia_for_pair', autospec=True)
async def test_adaptive_logic_falls_back_to_1d(mock_query, validator):
    """Prueba que el validador usa la prueba 1D si las pruebas 3D y 2D fallan."""
    # Estrella 1 tiene datos completos, Estrella 5 no tiene PM.
    # Los tests 3D y 2D deben fallar. Debe intentar el test 1D.
    # El paralaje de STAR_5_NO_PM es idéntico al de STAR_1_FULL.
    mock_query.return_value = [STAR_1_FULL, STAR_5_NO_PM]
    
    result = await validator.validate_physicality(
        primary_coords_deg=(0, 0), wds_magnitudes=(10.0, 14.0)
    )
    
    assert result['label'] == 'Likely Physical'
    assert result['test_used'] == '1D (plx_only)'
    assert result['p_value'] > validator.physical_threshold

@pytest.mark.asyncio
@patch('astrakairos.data.gaia_source.GaiaValidator._query_gaia_for_pair', autospec=True)
async def test_validation_unknown_if_no_valid_test(mock_query, validator):
    """Verifica que devuelve 'Unknown' si no se puede realizar ninguna prueba."""
    # Estrella 4 no tiene paralaje, Estrella 5 no tiene PM. No hay ningún test en común.
    mock_query.return_value = [STAR_4_NO_PARALLAX, STAR_5_NO_PM]
    
    result = await validator.validate_physicality(
        primary_coords_deg=(0, 0), wds_magnitudes=(13.0, 14.0)
    )
    
    assert result['label'] == 'Unknown'
    assert result['test_used'] == 'Incomplete astrometry'

@pytest.mark.asyncio
@patch('astrakairos.data.gaia_source.GaiaValidator._query_gaia_for_pair', autospec=True)
async def test_validation_unknown_if_not_enough_stars(mock_query, validator):
    """Verifica que devuelve 'Unknown' si la consulta a Gaia devuelve menos de 2 estrellas."""
    mock_query.return_value = [STAR_1_FULL] # Solo se encuentra una estrella
    
    result = await validator.validate_physicality(
        primary_coords_deg=(0, 0), wds_magnitudes=(10.0, 12.0)
    )
    
    assert result['label'] == 'Unknown'
    assert result['test_used'] == 'Not enough Gaia sources'

# --- Pruebas para la Lógica de Identificación de Componentes ---

@pytest.mark.parametrize("wds_mags, gaia_stars, expected_ids", [
    # Caso 1: Identificación clara por magnitud
    ( (10.0, 12.0), [STAR_2_PHYSICAL, STAR_1_FULL], (1, 2) ),
    # Caso 2: Secundaria es más brillante en Gaia (B < A)
    ( (12.0, 10.0), [STAR_2_PHYSICAL, STAR_1_FULL], (2, 1) ),
    # Caso 3: Sin magnitudes WDS, debe usar el orden de brillo de Gaia
    ( (None, None), [STAR_1_FULL, STAR_3_OPTICAL], (1, 3) ),
])
def test_identify_components_by_mag(validator, wds_mags, gaia_stars, expected_ids):
    """Prueba la lógica de identificación de componentes por magnitud."""
    primary, secondary = validator._identify_components_by_mag(gaia_stars, wds_mags)
    
    assert primary['source_id'] == expected_ids[0]
    assert secondary['source_id'] == expected_ids[1]

@pytest.mark.asyncio
@patch('astrakairos.data.gaia_source.GaiaValidator._query_gaia_for_pair', autospec=True)
async def test_validation_ambiguous_pair(mock_query, validator):
    """
    Verifies that a pair with moderately different data falls into the 'Ambiguous' category.
    """
    # Create a star with data that is more significantly different to ensure
    # the resulting p-value falls into the 'ambiguous' range.
    # A ~4-sigma deviation should be sufficient.
    star_ambiguous = {
        'source_id': 6, 'phot_g_mean_mag': 11.5,
        'parallax': 10.4, 'parallax_error': 0.1,  # 4-sigma difference in parallax
        'pmra': 50.4, 'pmra_error': 0.1,      # 4-sigma difference in pmra
        'pmdec': -20.0, 'pmdec_error': 0.1,     # No difference here
        'parallax_pmra_corr': 0.2, 'parallax_pmdec_corr': 0.1, 'pmra_pmdec_corr': -0.3
    }
    mock_query.return_value = [STAR_1_FULL, star_ambiguous]
    
    result = await validator.validate_physicality(
        primary_coords_deg=(0, 0), wds_magnitudes=(10.0, 11.5)
    )
    
    # Now, the p-value should be low enough to not be 'Likely Physical', but not
    # so low as to be 'Likely Optical'.
    assert result['label'] == 'Ambiguous'
    assert result['test_used'] == '3D (plx+pm)'
    # Check that the p-value is in the correct intermediate range
    assert validator.ambiguous_threshold < result['p_value'] <= validator.physical_threshold


def test_calculate_chi2_handles_singular_matrix(validator):
    """
    Tests that _calculate_chi2_... functions return None if the covariance matrix is singular.
    """
    # Create data that will result in a singular (non-invertible) matrix.
    # This happens if errors are zero.
    star_a = STAR_1_FULL.copy()
    star_b = STAR_2_PHYSICAL.copy()
    star_a['parallax_error'] = 0.0
    star_a['pmra_error'] = 0.0
    star_a['pmdec_error'] = 0.0
    star_b['parallax_error'] = 0.0
    star_b['pmra_error'] = 0.0
    star_b['pmdec_error'] = 0.0
    
    # The @patch is not needed here as we are calling a private synchronous method directly
    # to test its internal logic.
    result_3d = validator._calculate_chi2_3d(star_a, star_b)
    result_2d = validator._calculate_chi2_2d_pm(star_a, star_b)
    result_1d = validator._calculate_chi2_1d_plx(star_a, star_b)
    
    assert result_3d is None
    assert result_2d is None
    assert result_1d is None


def test_get_params_and_check_validity_handles_bad_data(validator):
    """
    Tests the _get_params_and_check_validity helper with various forms of bad data.
    """
    # Case 1: Key is missing entirely
    bad_star_1 = {'parallax': 10.0} # Missing 'parallax_error'
    assert not validator._get_params_and_check_validity(bad_star_1, ['parallax', 'parallax_error'])

    # Case 2: Value is None
    bad_star_2 = {'parallax': 10.0, 'parallax_error': None}
    assert not validator._get_params_and_check_validity(bad_star_2, ['parallax', 'parallax_error'])

    # Case 3: Value is a masked numpy value (as can happen with astroquery)
    bad_star_3 = {'parallax': 10.0, 'parallax_error': np.ma.masked}
    assert not validator._get_params_and_check_validity(bad_star_3, ['parallax', 'parallax_error'])

    # Case 4: Good data should pass
    good_star = {'parallax': 10.0, 'parallax_error': 0.1}
    assert validator._get_params_and_check_validity(good_star, ['parallax', 'parallax_error'])