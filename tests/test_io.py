import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open

# Importar las funciones del módulo a probar
from astrakairos.utils import io
# Importar dependencias necesarias para las pruebas
from astropy.coordinates import SkyCoord
import astropy.units as u

# --- Datos de prueba para simular archivos CSV ---
csv_content_comma = "wds_name,obs\n00001+0001,10\n00002+0002,5"
csv_content_semicolon = "wds_name;obs\n00003+0003;12\n00004+0004;8"

# --- Pruebas para load_csv_data ---

def test_load_csv_with_comma_delimiter():
    """Verifica que el archivo se carga correctamente con comas como delimitador."""
    with patch("builtins.open", mock_open(read_data=csv_content_comma)):
        df = io.load_csv_data("dummy_path.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'wds_name' in df.columns
        assert df['obs'].iloc[0] == 10

def test_load_csv_with_semicolon_delimiter():
    """Verifica que el fallback a punto y coma funciona."""
    with patch("builtins.open", mock_open(read_data=csv_content_semicolon)):
        df = io.load_csv_data("dummy_path.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df['wds_name'].iloc[0] == "00003+0003"
        assert df['obs'].iloc[1] == 8

def test_load_csv_raises_error_for_nonexistent_file():
    """
    Verifica que se lanza una excepción IOError si el archivo no se puede encontrar.
    La función interna captura FileNotFoundError y lanza un IOError más general.
    """
    with pytest.raises(IOError):
        io.load_csv_data("nonexistent_file.csv")


# --- Pruebas para save_results_to_csv ---

def test_save_results_to_csv():
    """Verifica que los datos se guardan correctamente en un CSV."""
    results_list = [
        {'wds_name': '00001+0001', 'opi': 0.5, 'v_total': 0.1},
        {'wds_name': '00002+0002', 'opi': 0.2, 'v_total': 0.3}
    ]
    with patch('pandas.DataFrame.to_csv') as mock_to_csv:
        io.save_results_to_csv(results_list, "output.csv")
        mock_to_csv.assert_called_once()
        assert mock_to_csv.call_args[0][0] == "output.csv"
        assert mock_to_csv.call_args[1]['index'] is False

def test_save_results_with_no_data():
    """Verifica que no se hace nada si la lista de resultados está vacía."""
    with patch('pandas.DataFrame.to_csv') as mock_to_csv:
        io.save_results_to_csv([], "output.csv")
        mock_to_csv.assert_not_called()


# --- Pruebas para format_coordinates_astropy ---

def test_format_coordinates_astropy_normal():
    """
    Prueba el formato para coordenadas normales.
    Genera la cadena esperada usando astropy para que la prueba sea robusta.
    """
    ra_hours = 12.53
    dec_degrees = 25.5
    # Generar la cadena de referencia con el mismo método y formato corregido
    expected_str = SkyCoord(ra=ra_hours*u.hourangle, dec=dec_degrees*u.deg).to_string(
        'hmsdms', sep=' ', precision=1, pad=True
    )
    assert io.format_coordinates_astropy(ra_hours, dec_degrees, precision=1) == expected_str

def test_format_coordinates_astropy_negative_dec():
    """Prueba el formato para declinaciones negativas."""
    ra_hours = 1.0
    dec_degrees = -5.75
    # Generar la cadena de referencia con el mismo método y formato corregido
    expected_str = SkyCoord(ra=ra_hours*u.hourangle, dec=dec_degrees*u.deg).to_string(
        'hmsdms', sep=' ', precision=1, pad=True
    )
    assert io.format_coordinates_astropy(ra_hours, dec_degrees, precision=1) == expected_str

def test_format_coordinates_astropy_with_none_input():
    """Prueba que devuelve 'N/A' si la entrada es None."""
    assert io.format_coordinates_astropy(None, 20.0) == "N/A"
    assert io.format_coordinates_astropy(10.0, None) == "N/A"


# --- Pruebas para parse_wds_designation ---

@pytest.mark.parametrize("wds_id, expected_ra_deg, expected_dec_deg", [
    ("00013+1234", pytest.approx(0.325), pytest.approx(12.5666, abs=1e-4)),
    ("15452-0812", pytest.approx(236.3), pytest.approx(-8.2, abs=1e-4)),
    ("23599+0000", pytest.approx(359.975), pytest.approx(0.0, abs=1e-4)),
])
def test_parse_wds_designation_valid(wds_id, expected_ra_deg, expected_dec_deg):
    """Prueba el parseo de designaciones WDS válidas usando parametrize."""
    result = io.parse_wds_designation(wds_id)
    assert result is not None
    assert result['ra_deg'] == expected_ra_deg
    assert result['dec_deg'] == expected_dec_deg

@pytest.mark.parametrize("invalid_wds_id", [
    "1234+5678",      # Formato incorrecto
    "J12345+6789",    # No es el formato numérico
    "12345+678",      # Longitud incorrecta
    None,             # Entrada nula
    12345,            # Tipo de dato incorrecto
    "ABCDE+FGHI"      # No son números
])
def test_parse_wds_designation_invalid(invalid_wds_id):
    """Prueba que la función devuelve None para entradas inválidas."""
    assert io.parse_wds_designation(invalid_wds_id) is None

# Test enhanced coordinate validation
def test_parse_wds_designation_coordinate_validation():
    """Test that parse_wds_designation validates astronomical coordinate ranges."""
    # Test RA out of range (> 360°)
    assert io.parse_wds_designation("25000+1234") is None  # RA = 375°
    
    # Test Dec out of range (> 90°)  
    assert io.parse_wds_designation("00000+9100") is None  # Dec = 91°
    
    # Test Dec out of range (< -90°)
    assert io.parse_wds_designation("00000-9100") is None  # Dec = -91°
    
    # Test valid coordinates at boundaries
    result = io.parse_wds_designation("00000+9000")  # Dec = 90° (valid)
    assert result is not None
    assert result['dec_deg'] == 90.0
    
    result = io.parse_wds_designation("23599+0000")  # RA = 359.98° (valid)
    assert result is not None
    assert pytest.approx(result['ra_deg'], abs=0.1) == 359.98

def test_format_coordinates_astropy_precision_config():
    """Test that format_coordinates_astropy uses configuration for default precision."""
    ra_hours = 1.0
    dec_degrees = 5.0
    
    # Test with default precision (should use config value)
    result_default = io.format_coordinates_astropy(ra_hours, dec_degrees)
    
    # Test with explicit precision
    result_explicit = io.format_coordinates_astropy(ra_hours, dec_degrees, precision=2)
    
    # Both should be valid coordinate strings in astropy format
    assert " " in result_default  # Should have spaces as separators
    assert "+" in result_default or "-" in result_default  # Should have sign for declination
    assert " " in result_explicit
    assert "+" in result_explicit or "-" in result_explicit
    assert result_explicit == io.format_coordinates_astropy(ra_hours, dec_degrees, precision=2)

# Test unified coordinate formatting functions
def test_format_coordinates_unified():
    """Test the unified coordinate formatting function."""
    ra_hours = 1.5
    dec_degrees = 45.5
    
    result = io.format_coordinates(ra_hours, dec_degrees)
    
    # Should be same as astropy function
    expected = io.format_coordinates_astropy(ra_hours, dec_degrees)
    assert result == expected

def test_format_ra_hours_unified():
    """Test unified RA formatting."""
    # Normal case
    assert io.format_ra_hours_unified(12.5) == "12h30m00.00s"
    
    # Boundary cases
    assert io.format_ra_hours_unified(0.0) == "00h00m00.00s"
    assert io.format_ra_hours_unified(23.999) == "23h59m56.40s"
    
    # Hour normalization (>24)
    assert io.format_ra_hours_unified(25.0) == "01h00m00.00s"
    
    # None handling
    assert io.format_ra_hours_unified(None) == "N/A"

def test_format_dec_degrees_unified():
    """Test unified Dec formatting."""
    # Positive case
    assert io.format_dec_degrees_unified(45.75) == "+45°45'00.00\""
    
    # Negative case  
    assert io.format_dec_degrees_unified(-30.25) == "-30°15'00.00\""
    
    # Zero case
    assert io.format_dec_degrees_unified(0.0) == "+00°00'00.00\""
    
    # None handling
    assert io.format_dec_degrees_unified(None) == "N/A"

def test_coordinate_error_behavior():
    """Test configurable error handling behavior."""
    # This test should pass regardless of current config
    result = io.format_coordinates_astropy("invalid", "invalid")
    
    # Should return something meaningful (not crash)
    assert result is not None
    assert isinstance(result, str)