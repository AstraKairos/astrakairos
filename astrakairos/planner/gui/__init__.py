# astrakairos/planner/gui/__init__.py

"""
GUI package for AstraKairos planner.
Contains modular GUI components for the binary star observation planner.
"""

from .utilities import GUIUtilities
from .location_widgets import LocationManager
from .calculation_widgets import CalculationManager
from .search_widgets import SearchManager
from .data_export import ExportManager
from .main_app import AstraKairosApp, AstraKairosWindow


# Main entry point function
def main():
    """Main entry point."""
    app = AstraKairosWindow()
    app.run()


# For direct import
def create_gui(root=None):
    """Create and return a GUI instance."""
    return AstraKairosWindow(root)


__all__ = [
    'GUIUtilities',
    'LocationManager', 
    'CalculationManager',
    'SearchManager',
    'ExportManager',
    'AstraKairosApp',
    'AstraKairosWindow',
    'main',
    'create_gui'
]
