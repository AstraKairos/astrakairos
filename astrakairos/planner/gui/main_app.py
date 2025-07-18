# astrakairos/planner/gui/main_app.py

"""
Main application class that coordinates all GUI components.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import threading
import json

from .utilities import GUIUtilities
from .location_widgets import LocationManager
from .calculation_widgets import CalculationManager
from .search_widgets import SearchManager
from .data_export import ExportManager
from .. import calculations


class AstraKairosApp:
    """Main application class for AstraKairos GUI."""
    
    def __init__(self, root):
        self.root = root
        self._setup_window()
        self._initialize_managers()
        self._create_interface()
        self._load_initial_data()  # Load data after creating interface
    
    def _setup_window(self):
        """Setup the main window properties."""
        self.root.title("AstraKairos - Binary Star Observation Planner")
        self.root.geometry("900x700")  # Smaller initial size
        self.root.minsize(700, 500)    # Reasonable minimum size
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TLabel', foreground='#333333')
        style.configure('Title.TLabel', font=('Arial', 12, 'bold'), foreground='#2E5984')
        style.configure('TButton', padding=6)
        style.configure('TLabelFrame', foreground='#2E5984', font=('Arial', 9, 'bold'))
        
        # Configure grid for main window
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def _initialize_managers(self):
        """Initialize all manager classes."""
        self.utilities = GUIUtilities(self)
        self.location_manager = LocationManager(self)
        self.calculation_manager = CalculationManager(self)
        self.search_manager = SearchManager(self)
        self.export_manager = ExportManager(self)
    
    def _create_interface(self):
        """Create the main interface components."""
        # Create main container with scrolling
        self.main_frame = self.utilities.create_scrollable_frame(self.root)
        
        # Add padding to the main frame
        main_container = ttk.Frame(self.main_frame)
        main_container.grid(row=0, column=0, sticky="ew", padx=20, pady=10)  # Add padding
        main_container.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_container, 
            text="🔭 AstraKairos - Binary Star Observation Planner",
            style='Title.TLabel'
        )
        title_label.grid(row=0, column=0, pady=(0, 20), sticky="ew")
        
        # Create all sections
        self.location_manager.create_location_section(main_container)
        self.calculation_manager.create_calculation_section(main_container)
        self.search_manager.create_search_section(main_container)
        self.export_manager.create_export_section(main_container)
        self.export_manager.create_progress_section(main_container)
        
        # Configure main frame column weight
        self.main_frame.columnconfigure(0, weight=1)
    
    def _load_initial_data(self):
        """Load initial data and set default values."""
        # Load locations (this will also update the UI)
        self.location_manager.load_locations()
        
        # Set initial status
        self.status_var.set("Ready")
        self.results_var.set("No results")
        self.progress_var.set(0)
        
        # Auto-calculate if location is available
        if self.location_manager.selected_location:
            if hasattr(self, 'calculation_manager'):
                self.calculation_manager.auto_calculate()
    
    def get_current_location_data(self):
        """Get current location data for calculations."""
        return self.location_manager.get_current_location_data()
    
    def update_status(self, message: str, color: str = 'green'):
        """Update the status message."""
        self.status_var.set(message)
        self.status_label.configure(foreground=color)
        self.root.update_idletasks()
    
    def update_progress(self, value: float):
        """Update the progress bar."""
        self.progress_var.set(value)
        self.root.update_idletasks()
    
    def update_results_count(self, count: int):
        """Update the results counter."""
        if count == 0:
            self.results_var.set("No results")
        elif count == 1:
            self.results_var.set("1 result found")
        else:
            self.results_var.set(f"{count} results found")
    
    def show_error(self, title: str, message: str):
        """Show error message to user."""
        messagebox.showerror(title, message)
        self.update_status("Error", 'red')
    
    def show_info(self, title: str, message: str):
        """Show info message to user."""
        messagebox.showinfo(title, message)
    
    def show_warning(self, title: str, message: str):
        """Show warning message to user."""
        messagebox.showwarning(title, message)
    
    def run_async_task(self, task_func, *args, **kwargs):
        """Run a task asynchronously in a separate thread."""
        def wrapper():
            try:
                result = task_func(*args, **kwargs)
                # Schedule UI update in main thread
                self.root.after(0, lambda: self._handle_async_result(result))
            except Exception as e:
                # Schedule error handling in main thread
                self.root.after(0, lambda: self._handle_async_error(e))
        
        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()
    
    def _handle_async_result(self, result):
        """Handle successful async task result."""
        self.update_status("Task completed")
        self.update_progress(100)
    
    def _handle_async_error(self, error):
        """Handle async task error."""
        self.show_error("Task Error", f"An error occurred: {str(error)}")
        self.update_progress(0)


class AstraKairosWindow:
    """Main application window."""
    
    def __init__(self, root=None):
        if root is None:
            self.root = tk.Tk()
            self.owns_root = True
        else:
            self.root = root
            self.owns_root = False
        
        self.app = AstraKairosApp(self.root)
    
    def run(self):
        """Run the application main loop."""
        if self.owns_root:
            self.root.mainloop()
    
    def destroy(self):
        """Destroy the application window."""
        if self.owns_root:
            self.root.destroy()


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = AstraKairosApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
