# astrakairos/planner/gui.py

"""
Main GUI module for AstraKairos - now using modular architecture.

This module maintains backwards compatibility while using the new
modular GUI components.
"""

import tkinter as tk
from tkinter import ttk

# Import the new modular GUI components
from .gui.main_app import AstraKairosApp, AstraKairosWindow


class AstraKairosPlannerApp(AstraKairosWindow):
    """
    Backwards compatible wrapper for the original GUI class.
    
    This class maintains the same interface as the original monolithic GUI
    while using the new modular architecture internally.
    """
    
    def __init__(self, root=None):
        # Initialize the new modular app
        super().__init__(root)
        
        # Expose all original methods and properties for backwards compatibility
        self._setup_backwards_compatibility()
    
    def _setup_backwards_compatibility(self):
        """Setup backwards compatibility with the original interface."""
        # The new AstraKairosWindow already exposes app attributes
        # Additional compatibility mappings can be added here if needed
        pass


# For direct import compatibility
def create_gui(root=None):
    """Create and return a GUI instance."""
    return AstraKairosPlannerApp(root)


def main():
    """Main entry point - maintains original interface."""
    app = AstraKairosPlannerApp()  # Don't pass root, let it create its own
    app.run()


if __name__ == "__main__":
    main()
