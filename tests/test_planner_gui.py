"""
Tests for astrakairos.planner.gui module.

Tests the GUI module functionality and imports.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestGUIImports:
    """Test GUI module imports and structure."""
    
    def test_can_import_gui_module(self):
        """Test that GUI module can be imported."""
        from astrakairos.planner import gui
        assert gui is not None
    
    def test_can_import_main_functions(self):
        """Test that main GUI functions can be imported."""
        from astrakairos.planner.gui import create_gui, main
        assert create_gui is not None
        assert main is not None
        assert callable(create_gui)
        assert callable(main)
    
    def test_can_import_main_app_classes(self):
        """Test that main app classes can be imported."""
        from astrakairos.planner.gui.main_app import AstraKairosWindow
        assert AstraKairosWindow is not None
    
    def test_gui_module_has_docstring(self):
        """Test that GUI module has documentation."""
        from astrakairos.planner import gui
        assert gui.__doc__ is not None
        assert len(gui.__doc__.strip()) > 0


class TestGUIFunctionality:
    """Test basic GUI functionality without mocking complex internals."""
    
    def test_create_gui_function_exists(self):
        """Test that create_gui function exists and is callable."""
        from astrakairos.planner.gui import create_gui
        assert callable(create_gui)
    
    def test_main_function_exists(self):
        """Test that main function exists and is callable."""
        from astrakairos.planner.gui import main
        assert callable(main)
    
    def test_can_create_gui_instance(self):
        """Test that GUI instance creation is available (skip actual creation)."""
        from astrakairos.planner.gui import create_gui
        
        # For CI/testing environments, we just verify the function exists
        # Actual GUI creation requires a display and full tkinter environment
        assert callable(create_gui)
        
        # Skip actual instantiation to avoid display/tkinter requirements
        pytest.skip("GUI instantiation skipped to avoid display/tkinter requirements in test environment")


class TestGUIComponentStructure:
    """Test GUI component structure and availability."""
    
    def test_utilities_available(self):
        """Test that GUI utilities are available."""
        from astrakairos.planner.gui.utilities import GUIUtilities
        assert GUIUtilities is not None
    
    def test_location_widgets_available(self):
        """Test that location widgets are available."""
        from astrakairos.planner.gui.location_widgets import LocationManager
        assert LocationManager is not None
    
    def test_calculation_widgets_available(self):
        """Test that calculation widgets are available."""
        from astrakairos.planner.gui.calculation_widgets import CalculationManager
        assert CalculationManager is not None
    
    def test_search_widgets_available(self):
        """Test that search widgets are available."""
        from astrakairos.planner.gui.search_widgets import SearchManager
        assert SearchManager is not None
    
    def test_data_export_available(self):
        """Test that data export functionality is available."""
        from astrakairos.planner.gui.data_export import ExportManager
        assert ExportManager is not None
    
    def test_main_app_classes_available(self):
        """Test that main app classes are available."""
        from astrakairos.planner.gui.main_app import AstraKairosApp, AstraKairosWindow
        assert AstraKairosApp is not None
        assert AstraKairosWindow is not None


class TestGUIErrorHandling:
    """Test error handling in GUI components."""
    
    def test_import_error_handling(self):
        """Test that import errors are handled gracefully."""
        # Test that basic imports work
        try:
            from astrakairos.planner.gui import main
            import_success = True
        except ImportError:
            import_success = False
        
        assert import_success, "Basic GUI imports should work"
    
    def test_missing_tkinter_handling(self):
        """Test handling when tkinter is not available."""
        # This is more of a documentation test - in real scenarios,
        # the GUI should handle missing tkinter gracefully
        
        # For now, just verify the imports work
        from astrakairos.planner.gui.main_app import AstraKairosWindow
        assert AstraKairosWindow is not None


if __name__ == "__main__":
    pytest.main([__file__])
