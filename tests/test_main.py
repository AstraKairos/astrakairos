"""
Tests for main.py entry point.

Tests the command-line interface and argument parsing functionality.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock
from io import StringIO

# Import the main module
import main


class TestMain:
    """Test class for main.py entry point functionality."""
    
    def test_version_info_constants(self):
        """Test that version information constants are properly defined."""
        assert hasattr(main, '__version__')
        assert hasattr(main, '__author__')
        assert hasattr(main, '__institution__')
        assert main.__version__ == "1.0.0"
        assert main.__author__ == "Martín Rubina Scapini"
        assert "Universidad Técnica Federico Santa María" in main.__institution__
    
    @patch('astrakairos.planner.gui.main')
    def test_main_planner_tool(self, mock_planner_main):
        """Test that planner tool launches correctly."""
        # Mock sys.argv to simulate planner command
        test_args = ['main.py', 'planner']
        
        with patch.object(sys, 'argv', test_args):
            main.main()
        
        # Verify planner main was called
        mock_planner_main.assert_called_once()
    
    @patch('astrakairos.analyzer.cli.main')  
    def test_main_analyzer_tool(self, mock_analyzer_main):
        """Test that analyzer tool launches correctly with arguments."""
        # Mock sys.argv to simulate analyzer command with additional args
        test_args = ['main.py', 'analyzer', '--min-obs', '5']
        
        with patch.object(sys, 'argv', test_args):
            main.main()
        
        # Verify analyzer main was called with remaining args
        mock_analyzer_main.assert_called_once_with(['--min-obs', '5'])
    
    @patch('astrakairos.analyzer.cli.main')
    def test_main_analyzer_with_multiple_args(self, mock_analyzer_main):
        """Test analyzer with complex argument passing."""
        test_args = ['main.py', 'analyzer', 'orbital', '--min-obs', '5', '--sort-by', 'v_total']
        
        with patch.object(sys, 'argv', test_args):
            main.main()
        
        expected_args = ['orbital', '--min-obs', '5', '--sort-by', 'v_total']
        mock_analyzer_main.assert_called_once_with(expected_args)
    
    def test_main_invalid_tool(self):
        """Test that invalid tool selection raises SystemExit."""
        test_args = ['main.py', 'invalid_tool']
        
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit):
                main.main()
    
    @patch('sys.stderr', new_callable=StringIO)
    def test_main_import_error_handling(self, mock_stderr):
        """Test that ImportError is handled gracefully."""
        test_args = ['main.py', 'planner']
        
        with patch.object(sys, 'argv', test_args):
            with patch('astrakairos.planner.gui.main', side_effect=ImportError("Test import error")):
                with pytest.raises(SystemExit) as exc_info:
                    main.main()
                
                assert exc_info.value.code == 1
                stderr_output = mock_stderr.getvalue()
                assert "Failed to import required module" in stderr_output
                assert "Test import error" in stderr_output
    
    @patch('sys.stderr', new_callable=StringIO)
    def test_main_keyboard_interrupt_handling(self, mock_stderr):
        """Test that KeyboardInterrupt is handled gracefully."""
        test_args = ['main.py', 'analyzer']
        
        with patch.object(sys, 'argv', test_args):
            with patch('astrakairos.analyzer.cli.main', side_effect=KeyboardInterrupt()):
                with pytest.raises(SystemExit) as exc_info:
                    main.main()
                
                assert exc_info.value.code == 130  # Standard SIGINT exit code
                stderr_output = mock_stderr.getvalue()
                assert "Operation cancelled by user" in stderr_output
    
    @patch('sys.stderr', new_callable=StringIO)
    def test_main_unexpected_error_handling(self, mock_stderr):
        """Test that unexpected errors are handled gracefully."""
        test_args = ['main.py', 'planner']
        
        with patch.object(sys, 'argv', test_args):
            with patch('astrakairos.planner.gui.main', side_effect=RuntimeError("Unexpected test error")):
                with pytest.raises(SystemExit) as exc_info:
                    main.main()
                
                assert exc_info.value.code == 1
                stderr_output = mock_stderr.getvalue()
                assert "Unexpected error" in stderr_output
                assert "Unexpected test error" in stderr_output
    
    def test_main_version_argument(self):
        """Test --version argument displays version information."""
        test_args = ['main.py', '--version']
        
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main.main()
            
            # SystemExit with code 0 indicates successful version display
            assert exc_info.value.code == 0
    
    def test_main_help_argument(self):
        """Test --help argument displays help information."""
        test_args = ['main.py', '--help']
        
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main.main()
            
            # SystemExit with code 0 indicates successful help display
            assert exc_info.value.code == 0
    
    def test_main_no_arguments(self):
        """Test that no arguments triggers help/error."""
        test_args = ['main.py']
        
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit):
                main.main()


class TestMainIntegration:
    """Integration tests for main.py with actual module imports."""
    
    @patch('builtins.print')  # Suppress output during tests
    def test_main_can_import_modules(self, mock_print):
        """Test that required modules can be imported successfully."""
        # Test that we can import the required modules without errors
        try:
            import astrakairos.planner.gui
            import astrakairos.analyzer.cli
            import_success = True
        except ImportError:
            import_success = False
        
        assert import_success, "Required modules should be importable"
    
    def test_argument_parser_configuration(self):
        """Test that the argument parser is configured correctly."""
        import argparse
        
        # Create parser as in main.py
        parser = argparse.ArgumentParser(
            description=f'AstraKairos v{main.__version__} - Binary Star Research Assistant',
            epilog='Use "planner" for the GUI observation planner or "analyzer" for the CLI analyzer.',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        parser.add_argument('tool', 
                           choices=['planner', 'analyzer'],
                           help='Tool to run')
        parser.add_argument('--version', action='version', 
                           version=f'AstraKairos {main.__version__} by {main.__author__} ({main.__institution__})')
        
        # Test valid arguments
        args, remaining = parser.parse_known_args(['planner'])
        assert args.tool == 'planner'
        
        args, remaining = parser.parse_known_args(['analyzer', '--min-obs', '5'])
        assert args.tool == 'analyzer'
        assert remaining == ['--min-obs', '5']
        
        # Test invalid arguments
        with pytest.raises(SystemExit):
            parser.parse_args(['invalid'])


if __name__ == "__main__":
    pytest.main([__file__])
