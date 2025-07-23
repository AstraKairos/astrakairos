"""
Test for the improved main function architecture.
"""

import pytest
from unittest.mock import patch, MagicMock
from scripts.convert_catalogs_to_sqlite import main
from astrakairos.exceptions import ConversionProcessError


def test_main_desacoplada():
    """Test that main function raises ConversionProcessError instead of calling sys.exit."""
    # Mock the argparse to simulate missing required arguments
    with patch('scripts.convert_catalogs_to_sqlite.argparse.ArgumentParser') as mock_parser:
        mock_parser_instance = MagicMock()
        mock_parser.return_value = mock_parser_instance
        
        # Simulate an error condition (missing file)
        mock_args = MagicMock()
        mock_args.output = "/non/existent/path/test.db"
        mock_args.force = False
        mock_parser_instance.parse_args.return_value = mock_args
        
        # Mock Path.exists to return True (file exists but --force not used)
        with patch('scripts.convert_catalogs_to_sqlite.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            
            # Main should raise ConversionProcessError, not call sys.exit
            with pytest.raises(ConversionProcessError, match="already exists"):
                main()


if __name__ == "__main__":
    test_main_desacoplada()
    print("✅ Test de main desacoplada pasó correctamente!")
