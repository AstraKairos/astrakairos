# astrakairos/planner/gui.py

"""
Main GUI module for AstraKairos.

This module provides the main entry point for the GUI planner.
"""

# Import the modular GUI components
from .gui.main_app import AstraKairosWindow


def create_gui(root=None):
    """Create and return a GUI instance."""
    return AstraKairosWindow(root)


def main():
    """Main entry point for the GUI planner."""
    app = AstraKairosWindow()
    app.run()


if __name__ == "__main__":
    main()
