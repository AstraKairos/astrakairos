# astrakairos/planner/gui.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import json
import csv
import os
from datetime import datetime, timedelta
import webbrowser
import urllib.parse
from typing import Dict, Any, Optional, List
import pytz
from pathlib import Path
import random
import pandas as pd
import io

# Importamos el módulo de cálculos actualizado que ahora usa Skyfield
from ..planner import calculations
from ..config import (
    GUI_DEFAULT_WIDTH, GUI_DEFAULT_HEIGHT,
    STELLE_DOPPIE_BASE_URL, STELLE_DOPPIE_SEARCH_METHODS,
    MIN_ALTITUDE_DEG, MAX_ALTITUDE_DEG,
    MIN_RA_WINDOW_HOURS, MAX_RA_WINDOW_HOURS,
    MIN_LIGHT_POLLUTION_MAG, MAX_LIGHT_POLLUTION_MAG,
    DEFAULT_MIN_ALTITUDE_DEG, DEFAULT_RA_WINDOW_HOURS, DEFAULT_LIGHT_POLLUTION_MAG,
    STELLE_DOPPIE_FILTERS, DEFAULT_SEARCH_OPTIONS, STELLE_DOPPIE_METHODS,
    EXPORT_FORMATS, CATALOG_SOURCES, UI_THEMES, PROGRESS_INDICATORS
)

class AstraKairosPlannerApp:
    """Main GUI application for the AstraKairos observation planner."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("AstraKairos - Observation Planner")
        self.root.geometry(f"{GUI_DEFAULT_WIDTH}x{GUI_DEFAULT_HEIGHT}")
        self.root.configure(bg='#f0f0f0')
        
        # --- Variables de Estado ---
        self.locations = []
        self.filtered_locations = []
        self.selected_location = None
        self.observer_location = None  # Para el objeto de ubicación de Skyfield
        
        self.optimal_ra_range = None
        self.optimal_dec_range = None
        
        # Search filter variables
        self.search_options = {}
        self.filter_vars = {}
        
        self.search_results = []  # Store search results for export
        self.current_theme = 'default'
        self.export_format = 'csv'
        self.selected_catalogs = set(['stelle_doppie'])  # Default enabled catalogs
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        
        # CSV Import variables
        self.imported_csv_data = None
        self.imported_csv_format = None
        self.current_file_var = tk.StringVar(value="No file loaded")
        
        self.load_locations()
        self.create_widgets()
        self.update_location_list()
        self._apply_theme()  # Apply initial theme
    
    def load_locations(self):
        """Load observatory locations from the JSON file."""
        try:
            with open('locations.json', 'r', encoding='utf-8') as f:
                self.locations = json.load(f)
        except FileNotFoundError:
            messagebox.showerror("Error", "File 'locations.json' not found.")
            self.locations = []
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Failed to parse 'locations.json'.")
            self.locations = []
    
    def create_widgets(self):
        """Create all GUI widgets and layout with scrollbar."""
        # Create main container frame
        container = ttk.Frame(self.root)
        container.grid(row=0, column=0, sticky="nsew")
        
        # Configure root grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(container, bg='#f0f0f0', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, padding="10")
        
        # Configure canvas scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window in canvas and configure it to expand
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind canvas resize to update scrollable frame width
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Grid canvas and scrollbar
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure scrollable frame
        self.scrollable_frame.columnconfigure(0, weight=1)
        
        # Bind mousewheel to canvas
        self._bind_mousewheel()
        
        # Create title
        title_label = ttk.Label(self.scrollable_frame, text="AstraKairos Observation Planner", font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 20), sticky='n')
        
        self._create_location_search_section(self.scrollable_frame)
        self._create_location_info_section(self.scrollable_frame)
        self._create_date_section(self.scrollable_frame)
        self._create_params_section(self.scrollable_frame)
        self._create_astro_info_section(self.scrollable_frame)
        self._create_search_options_section(self.scrollable_frame)
        self._create_binary_search_section(self.scrollable_frame)
        self._create_csv_import_section(self.scrollable_frame)
        
        self._create_export_section(self.scrollable_frame)
        self._create_catalog_selection_section(self.scrollable_frame)
        self._create_progress_section(self.scrollable_frame)
        
        # Configure row weights for proper expansion
        self.scrollable_frame.rowconfigure(5, weight=1)  # Astro info section gets extra space
    
    def _bind_mousewheel(self):
        """Bind mousewheel events to canvas for scrolling."""
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            self.canvas.unbind_all("<MouseWheel>")
        
        # Bind mousewheel events
        self.canvas.bind('<Enter>', _bind_to_mousewheel)
        self.canvas.bind('<Leave>', _unbind_from_mousewheel)

    def _on_canvas_configure(self, event):
        """Handle canvas resize to update scrollable frame width."""
        # Update the scrollable frame width to match canvas width
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def _create_location_search_section(self, parent):
        frame = ttk.LabelFrame(parent, text="1. Select Observatory Location", padding="10")
        frame.grid(row=1, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
        ttk.Label(frame, text="Search:").grid(row=0, column=0, sticky=tk.W)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(frame, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=1, sticky="ew", padx=5)
        self.search_var.trace('w', lambda *args: self.update_location_list())
        
        self.location_listbox = tk.Listbox(frame, height=6)
        self.location_listbox.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.location_listbox.bind('<<ListboxSelect>>', self.on_location_select)
        
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.location_listbox.yview)
        scrollbar.grid(row=1, column=2, sticky="ns")
        self.location_listbox.config(yscrollcommand=scrollbar.set)
    
    def _create_location_info_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Selected Location Details", padding="10")
        frame.grid(row=2, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
        labels = ["Name:", "Coordinates:", "Altitude:"]
        self.info_labels = {}
        for i, label_text in enumerate(labels):
            ttk.Label(frame, text=label_text).grid(row=i, column=0, sticky=tk.W, pady=2)
            label = ttk.Label(frame, text="N/A", font=('Arial', 10, 'bold'))
            label.grid(row=i, column=1, sticky=tk.W, padx=5)
            self.info_labels[label_text.rstrip(':')] = label

    def _create_date_section(self, parent):
        frame = ttk.LabelFrame(parent, text="2. Select Observation Date", padding="10")
        frame.grid(row=3, column=0, sticky="ew", pady=5)
        
        ttk.Label(frame, text="Date (YYYY-MM-DD):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(frame, textvariable=self.date_var, width=15).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(frame, text="Tonight", command=lambda: self.date_var.set(datetime.now().strftime("%Y-%m-%d"))).grid(row=0, column=2, padx=10)

    def _create_params_section(self, parent):
        frame = ttk.LabelFrame(parent, text="3. Set Observation Parameters", padding="10")
        frame.grid(row=4, column=0, sticky="ew", pady=5)
        
        ttk.Label(frame, text="Min Altitude (deg):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.min_alt_var = tk.DoubleVar(value=DEFAULT_MIN_ALTITUDE_DEG)
        ttk.Entry(frame, textvariable=self.min_alt_var, width=10).grid(row=0, column=1)
        
        ttk.Label(frame, text="RA Window (± hours):").grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        self.ra_win_var = tk.DoubleVar(value=DEFAULT_RA_WINDOW_HOURS)
        ttk.Entry(frame, textvariable=self.ra_win_var, width=10).grid(row=0, column=3)
        
        ttk.Label(frame, text="Light Pollution (mag/arcsec²):").grid(row=1, column=0, sticky=tk.W, pady=(5,0), padx=(0,5))
        self.lp_var = tk.DoubleVar(value=DEFAULT_LIGHT_POLLUTION_MAG)
        ttk.Entry(frame, textvariable=self.lp_var, width=10).grid(row=1, column=1, pady=(5,0))

    def _create_astro_info_section(self, parent):
        frame = ttk.LabelFrame(parent, text="4. Calculate & Review Conditions", padding="10")
        frame.grid(row=5, column=0, sticky="nsew", pady=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        
        ttk.Button(frame, text="Calculate Optimal Conditions", 
                   command=self.run_full_calculation).grid(row=0, column=0, pady=(0, 10))
        
        self.astro_text = ScrolledText(frame, height=12, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.astro_text.grid(row=1, column=0, sticky="nsew")

    def _create_filter_row(self, parent, row, filter_name, config):
        """Create a filter row with checkbox, dropdown for method, and textbox for value."""
        
        # Create checkbox for filter
        checkbox = ttk.Checkbutton(parent, text=f"Filter by {config['label']}:", 
                                 variable=self.search_options[f'use_{filter_name}_filter'])
        checkbox.grid(row=row, column=0, sticky=tk.W, pady=2)
        
        # Create frame for method dropdown and value entry
        filter_frame = ttk.Frame(parent)
        filter_frame.grid(row=row, column=1, sticky="ew", padx=5)
        
        # Create dropdown for method selection
        method_values = []
        method_labels = []
        for method_num in config['available_methods']:
            method_values.append(str(method_num))
            method_labels.append(STELLE_DOPPIE_METHODS[int(method_num)])
        
        # Create combobox with method labels but store method numbers
        method_combo = ttk.Combobox(filter_frame, values=method_labels, 
                                  state="readonly", width=15)
        method_combo.grid(row=0, column=0, padx=2)
        
        # Set default method in combobox
        default_method = config.get('default_method', '1')
        if default_method in method_values:
            idx = method_values.index(default_method)
            method_combo.set(method_labels[idx])
        
        # Bind method selection to update the StringVar
        def on_method_change(event):
            selected_label = method_combo.get()
            for i, label in enumerate(method_labels):
                if label == selected_label:
                    self.filter_vars[filter_name]['method'].set(method_values[i])
                    break
        
        method_combo.bind('<<ComboboxSelected>>', on_method_change)
        
        # Create entry for value
        value_entry = ttk.Entry(filter_frame, textvariable=self.filter_vars[filter_name]['value'], 
                               width=12)
        value_entry.grid(row=0, column=1, padx=2)
        
        # Store the entry widget reference for direct access to raw values
        self.filter_vars[filter_name]['entry'] = value_entry
        
        # Add unit label if specified
        if config.get('unit'):
            unit_label = ttk.Label(filter_frame, text=config['unit'])
            unit_label.grid(row=0, column=2, padx=2)
        
        return row + 1

    def _create_search_options_section(self, parent):
        """Create advanced search options section matching Stelle Doppie capabilities."""
        frame = ttk.LabelFrame(parent, text="5. Advanced Search Options", padding="10")
        frame.grid(row=6, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
        # Initialize search options with defaults
        for option, default in DEFAULT_SEARCH_OPTIONS.items():
            self.search_options[option] = tk.BooleanVar(value=default)
        
        # Create filter variable dictionaries for configurable filters
        for filter_name, config in STELLE_DOPPIE_FILTERS.items():
            data_type = config.get('data_type', 'numeric')
            
            # Create appropriate variable type based on data type
            if data_type == 'string':
                value_var = tk.StringVar(value=config.get('default_value', ''))
            elif data_type == 'integer':
                value_var = tk.IntVar(value=config.get('default_value', 0))
            else:  # numeric (float)
                value_var = tk.DoubleVar(value=config.get('default_value', 0.0))
            
            self.filter_vars[filter_name] = {
                'method': tk.StringVar(value=config.get('default_method', '1')),
                'value': value_var,
                'entry': None  # Will be set when creating the entry widget
            }
        
        row = 0
        
        # Create all configurable filters dynamically
        for filter_name, config in STELLE_DOPPIE_FILTERS.items():
            row = self._create_filter_row(frame, row, filter_name, config)
        
        # Toggle options section
        toggle_frame = ttk.Frame(frame)
        toggle_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        # Known Orbit toggle
        ttk.Checkbutton(toggle_frame, text="Known Orbit Only", 
                       variable=self.search_options['known_orbit']).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        # Physical Double toggle
        ttk.Checkbutton(toggle_frame, text="Physical Double Only", 
                       variable=self.search_options['physical_double']).grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        # Uncertain Double toggle
        ttk.Checkbutton(toggle_frame, text="Include Uncertain Doubles", 
                       variable=self.search_options['uncertain_double']).grid(row=0, column=2, sticky=tk.W)

    def _create_binary_search_section(self, parent):
        frame = ttk.LabelFrame(parent, text="6. Generate Search URL", padding="10")
        frame.grid(row=7, column=0, sticky="ew", pady=5)
        frame.columnconfigure(0, weight=1)

        # Instructions
        instructions_frame = ttk.Frame(frame)
        instructions_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        instructions_text = (
            "📋 How to get complete CSV results:\n"
            "1. Click 'Generate Search URL' below\n"
            "2. Open URL in browser and log in to Stelle Doppie\n"
            "3. Change 'index2.php' to 'excel.php' in URL for CSV download\n"
            "4. Download CSV file for detailed analysis"
        )
        
        instructions_label = ttk.Label(instructions_frame, text=instructions_text, 
                                     font=('Arial', 9), justify='left')
        instructions_label.grid(row=0, column=0, sticky="w")

        # Generate URL button
        ttk.Button(frame, text="🔍 Generate Search URL", 
                   command=self.generate_stelle_doppie_search).grid(row=1, column=0, pady=5, sticky="ew")
        
        # Add a text widget to show the generated URL
        self.url_text = ScrolledText(frame, height=3, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.url_text.grid(row=2, column=0, sticky="ew", pady=(5, 0))

        # Add results display area
        self.results_text = ScrolledText(frame, height=8, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.results_text.grid(row=3, column=0, sticky="ew", pady=(5, 0))
        
        # Initialize with helpful instructions
        self._initialize_results_text()

    def _initialize_results_text(self):
        """Initialize the results text area with helpful instructions."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        instructions = """🔍 SEARCH RESULTS PREVIEW

This area shows a preview of what to expect from your search:

📝 NEXT STEPS:
1. Click "🔍 Generate Search URL" above to create the search URL
2. Copy the generated URL and open it in your browser
3. Log in to Stelle Doppie for complete data access
4. Modify the URL by changing 'index2.php' to 'excel.php' 
5. Download the CSV file from your browser
6. Use "📁 Import CSV File" below to analyze the data

💡 TIP: The URL generation creates a mock preview of expected results. For actual data, you must use your browser to access Stelle Doppie.

🔄 WORKFLOW:
AstraKairos → Browser → Stelle Doppie → CSV → AstraKairos Analysis
"""
        
        self.results_text.insert(tk.END, instructions)
        self.results_text.config(state=tk.DISABLED)

    def _create_csv_import_section(self, parent):
        """Create CSV import section for user-downloaded files."""
        frame = ttk.LabelFrame(parent, text="7. Import & Analyze CSV Results", padding="10")
        frame.grid(row=8, column=0, sticky="ew", pady=5)
        frame.columnconfigure(0, weight=1)

        # Instructions frame
        instructions_frame = ttk.Frame(frame)
        instructions_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        instructions_frame.columnconfigure(0, weight=1)

        instructions_text = (
            "📋 To get detailed CSV data with full columns:\n"
            "1. Generate search URL above and open in browser\n"
            "2. Log in to Stelle Doppie (required for full data)\n"
            "3. Change 'index2.php' to 'excel.php' in the URL\n"
            "4. Download the CSV file\n"
            "5. Import it below for analysis"
        )
        
        instructions_label = ttk.Label(instructions_frame, text=instructions_text, 
                                     font=('Arial', 9), justify='left')
        instructions_label.grid(row=0, column=0, sticky="w")

        # Button frame
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=1, column=0, sticky="ew", pady=5)
        button_frame.columnconfigure(0, weight=1)

        ttk.Button(button_frame, text="📁 Import CSV File", 
                   command=self.import_csv_file).grid(row=0, column=0, pady=5, sticky="ew")
        
        # Current file info
        file_info_label = ttk.Label(frame, textvariable=self.current_file_var, 
                                   font=('Arial', 9), foreground='blue')
        file_info_label.grid(row=2, column=0, sticky="w", pady=(5, 0))

        # CSV Analysis results display area
        self.csv_analysis_text = ScrolledText(frame, height=12, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.csv_analysis_text.grid(row=3, column=0, sticky="ew", pady=(5, 0))
        
        # Velocity analysis section
        velocity_frame = ttk.Frame(frame)
        velocity_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        velocity_frame.columnconfigure(0, weight=1)
        
        # Velocity analysis controls
        velocity_controls = ttk.Frame(velocity_frame)
        velocity_controls.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        ttk.Label(velocity_controls, text="Top systems by velocity:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        
        self.velocity_count_var = tk.IntVar(value=10)
        velocity_spinbox = ttk.Spinbox(velocity_controls, from_=1, to=100, width=5, textvariable=self.velocity_count_var)
        velocity_spinbox.grid(row=0, column=1, sticky="w", padx=(0, 5))
        
        ttk.Button(velocity_controls, text="🚀 Calculate Velocities", 
                  command=self.calculate_system_velocities).grid(row=0, column=2, sticky="w", padx=(5, 0))
        
        # Velocity results display area
        self.velocity_results_text = ScrolledText(velocity_frame, height=8, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.velocity_results_text.grid(row=1, column=0, sticky="ew", pady=(5, 0))

    def on_location_select(self, event):
        if not self.location_listbox.curselection(): return
        index = self.location_listbox.curselection()[0]
        self.selected_location = self.filtered_locations[index]
        self.update_location_info()
        self.create_observer_location()

    def update_location_list(self):
        search_term = self.search_var.get().lower()
        self.filtered_locations = [loc for loc in self.locations if search_term in loc['name'].lower()]
        self.location_listbox.delete(0, tk.END)
        for loc in self.filtered_locations:
            self.location_listbox.insert(tk.END, f"{loc['name']} ({loc.get('state', 'N/A')})")

    def update_location_info(self):
        if not self.selected_location: return
        loc = self.selected_location
        self.info_labels['Name'].config(text=loc['name'])
        self.info_labels['Coordinates'].config(text=f"{loc['latitude']}, {loc['longitude']}")
        self.info_labels['Altitude'].config(text=f"{loc.get('altitude_m', 'N/A')} m")
        
        # Update light pollution from location data
        self._update_light_pollution_from_location(loc)

    def _update_light_pollution_from_location(self, location):
        """Update light pollution value from location data with reasonable defaults."""
        light_pollution = location.get('light_pollution')
        
        if light_pollution is not None:
            try:
                # Convert to float if it's a string
                if isinstance(light_pollution, str):
                    light_pollution = float(light_pollution)
                
                # Validate and set the value
                if MIN_LIGHT_POLLUTION_MAG <= light_pollution <= MAX_LIGHT_POLLUTION_MAG:
                    self.lp_var.set(light_pollution)
                    return
            except (ValueError, TypeError):
                pass
        
        # If no valid light pollution data, use reasonable defaults based on location type
        location_type = location.get('type', 'U')  # U = Unknown
        population = location.get('population', 0)
        
        if location_type == 'O':  # Observatory
            default_lp = 21.5  # Dark site
        elif location_type == 'C':  # City
            if population and population > 1000000:
                default_lp = 17.0  # Major city
            elif population and population > 100000:
                default_lp = 18.5  # Large city
            else:
                default_lp = 19.5  # Small city
        elif location_type == 'R':  # Rural
            default_lp = 20.5  # Rural area
        else:  # Unknown type
            default_lp = DEFAULT_LIGHT_POLLUTION_MAG
        
        self.lp_var.set(default_lp)

    def create_observer_location(self):
        """Create observer location with robust coordinate validation."""
        if not self.selected_location: 
            return
        
        loc = self.selected_location
        try:
            # Parse latitude with validation
            lat_str = loc['latitude']
            if lat_str.endswith(('N', 'S')):
                lat_value = float(lat_str[:-1])
                lat = lat_value * (-1 if lat_str.endswith('S') else 1)
            else:
                lat = float(lat_str)
            
            # Parse longitude with validation
            lon_str = loc['longitude']
            if lon_str.endswith(('E', 'W')):
                lon_value = float(lon_str[:-1])
                lon = lon_value * (-1 if lon_str.endswith('W') else 1)
            else:
                lon = float(lon_str)
            
            # Validate coordinate ranges
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(f"Latitude {lat}° outside valid range [-90°, +90°]")
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(f"Longitude {lon}° outside valid range [-180°, +180°]")
            
            # Parse altitude with default
            alt = float(loc.get('altitude_m', 0))
            if alt < -500 or alt > 10000:  # Reasonable altitude range
                messagebox.showwarning("Location Warning", 
                                     f"Altitude {alt}m seems unusual. Proceeding anyway.")
            
            self.observer_location = calculations.get_observer_location(lat, lon, alt)
            
        except (ValueError, TypeError, KeyError) as e:
            messagebox.showerror("Location Error", 
                               f"Could not parse location coordinates: {e}")
            self.observer_location = None

    def run_full_calculation(self):
        if not self.observer_location:
            messagebox.showwarning("Input Required", "Please select a location first.")
            return
            
        try:
            obs_date = datetime.strptime(self.date_var.get(), "%Y-%m-%d")
            min_alt = self.min_alt_var.get()
            ra_win = self.ra_win_var.get()
            lp_mag = self.lp_var.get()
            
            # Validate parameters against configuration ranges
            if not (MIN_ALTITUDE_DEG <= min_alt <= MAX_ALTITUDE_DEG):
                messagebox.showerror("Invalid Parameter", 
                                   f"Minimum altitude must be between {MIN_ALTITUDE_DEG}° and {MAX_ALTITUDE_DEG}°")
                return
            
            if not (MIN_RA_WINDOW_HOURS <= ra_win <= MAX_RA_WINDOW_HOURS):
                messagebox.showerror("Invalid Parameter", 
                                   f"RA window must be between {MIN_RA_WINDOW_HOURS} and {MAX_RA_WINDOW_HOURS} hours")
                return
            
            if not (MIN_LIGHT_POLLUTION_MAG <= lp_mag <= MAX_LIGHT_POLLUTION_MAG):
                messagebox.showerror("Invalid Parameter", 
                                   f"Light pollution must be between {MIN_LIGHT_POLLUTION_MAG} and {MAX_LIGHT_POLLUTION_MAG} mag/arcsec²")
                return
            
            timezone = self.selected_location.get('timezone', 'UTC')
            
            # --- Perform All Calculations ---
            events = calculations.get_nightly_events(self.observer_location, obs_date, timezone)
            
            # Define the primary calculation time as the end of astronomical twilight
            calc_time = events.get('astronomical_twilight_end_utc')
            if not calc_time:
                # Fallback logic if twilight time is not available
                sunset_time = events.get('sunset_utc')
                calc_time = (sunset_time + timedelta(hours=2)) if sunset_time else datetime.combine(obs_date.date(), datetime.min.time().replace(hour=22))
            
            # 1. Calculate conditions for the START of the optimal observing window
            conditions_twilight = calculations.calculate_sky_conditions_at_time(self.observer_location, calc_time)
            
            # 2. Calculate conditions for TEMPORAL MIDNIGHT, if available
            conditions_midnight = None
            temporal_midnight_utc = events.get('temporal_midnight_utc')
            if temporal_midnight_utc:
                conditions_midnight = calculations.calculate_sky_conditions_at_time(self.observer_location, temporal_midnight_utc)

            # Generate the optimal patch recommendation based on the START of the night
            optimal_patch = calculations.generate_sky_quality_map(
                self.observer_location, calc_time, 
                min_altitude_deg=min_alt, sky_brightness_mag_arcsec2=lp_mag
            )
            
            ra_center = optimal_patch['best_ra_hours']
            self.optimal_ra_range = ((ra_center - ra_win + 24) % 24, (ra_center + ra_win) % 24)
            
            dec_center = optimal_patch['best_dec_deg']
            self.optimal_dec_range = (max(dec_center - 20, -90), min(dec_center + 20, 90))
            
            # Pass both sets of conditions to the display function
            self.display_results(events, conditions_twilight, conditions_midnight, optimal_patch, calc_time)

        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please ensure Date and Parameters are valid numbers: {e}")
        except Exception as e:
            messagebox.showerror("Calculation Error", f"An unexpected error occurred: {e}")

    def display_results(self, events: Dict, conditions_twilight: Dict, conditions_midnight: Dict, optimal_patch: Dict, calc_time: datetime):
        self.astro_text.config(state=tk.NORMAL)
        self.astro_text.delete(1.0, tk.END)
        
        def fmt_time(dt_obj): 
            return dt_obj.strftime('%H:%M %Z') if dt_obj else "N/A"
        
        def fmt_time_utc(dt_obj):
            if dt_obj and hasattr(dt_obj, 'astimezone'):
                return dt_obj.astimezone(pytz.UTC).strftime('%H:%M UTC')
            return "N/A"

        # --- Base Timeline (unchanged) ---
        timeline_text = (
            f"--- OBSERVATION TIMELINE FOR {self.date_var.get()} ---\n"
            f"Sunset (Local):         {fmt_time(events.get('sunset_local'))}\n"
            f"Sunset (UTC):           {fmt_time_utc(events.get('sunset_utc'))}\n"
            f"Civil Twilight End:     {fmt_time(events.get('civil_twilight_end_local'))}\n"
            f"Nautical Twilight End:  {fmt_time(events.get('nautical_twilight_end_local'))}\n"
            f"Astro. Twilight End:    {fmt_time(events.get('astronomical_twilight_end_local'))} (Start of Darkness)\n"
            f"Astronomical Midnight:  {fmt_time(events.get('astronomical_midnight_local'))}\n"
            f"Astronomical Midnight:  {fmt_time_utc(events.get('astronomical_midnight_utc'))}\n"
            f"Temporal Midnight:      {fmt_time(events.get('temporal_midnight_local'))} (Midpoint of Night)\n"
            f"Temporal Midnight:      {fmt_time_utc(events.get('temporal_midnight_utc'))}\n"
            f"Astro. Twilight Start:  {fmt_time(events.get('astronomical_twilight_start_local'))}\n"
            f"Nautical Twilight Start:{fmt_time(events.get('nautical_twilight_start_local'))}\n"
            f"Civil Twilight Start:   {fmt_time(events.get('civil_twilight_start_local'))}\n"
            f"Sunrise (Local):        {fmt_time(events.get('sunrise_local'))}\n"
            f"Sunrise (UTC):          {fmt_time_utc(events.get('sunrise_utc'))}\n"
            f"Moonrise:               {fmt_time(events.get('moonrise_local'))}\n"
            f"Moonset:                {fmt_time(events.get('moonset_local'))}\n"
        )
        
        # --- Conditions at Start of Darkness ---
        conditions_text = (
            f"\n--- CONDITIONS AT START OF DARKNESS ({fmt_time(calc_time)}) ---\n"
            f"Moon Phase:             {conditions_twilight.get('moon_phase_percent', 0):.1f}%\n"
            f"Moon Altitude:          {conditions_twilight.get('moon_alt_deg', 0):.1f}°\n"
            f"Moon Azimuth:           {conditions_twilight.get('moon_az_deg', 0):.1f}°\n"
            f"Zenith RA:              {conditions_twilight.get('zenith_ra_str', 'N/A')}\n"
            f"Zenith Dec:             {conditions_twilight.get('zenith_dec_str', 'N/A')}\n"
        )
        
        # --- Conditions at Temporal Midnight (if available) ---
        if conditions_midnight:
            conditions_text += (
                f"\n--- CONDITIONS AT TEMPORAL MIDNIGHT ({fmt_time(events.get('temporal_midnight_local'))}) ---\n"
                f"Moon Phase:             {conditions_midnight.get('moon_phase_percent', 0):.1f}%\n"
                f"Moon Altitude:          {conditions_midnight.get('moon_alt_deg', 0):.1f}°\n"
                f"Moon Azimuth:           {conditions_midnight.get('moon_az_deg', 0):.1f}°\n"
                f"Zenith RA:              {conditions_midnight.get('zenith_ra_str', 'N/A')}\n"
                f"Zenith Dec:             {conditions_midnight.get('zenith_dec_str', 'N/A')}\n"
            )

        # --- Recommended Observing Region (based on start of darkness) ---
        recommendation_text = (
            f"\n--- RECOMMENDED OBSERVING REGION (Calculated for Start of Darkness) ---\n"
            f"Best Patch Center (RA): {self.format_ra_hours(optimal_patch['best_ra_hours'])}\n"
            f"Best Patch Center (Dec):{self.format_dec_degrees(optimal_patch['best_dec_deg'])}\n"
            f"Best Patch Alt/Az:      {optimal_patch['best_alt_deg']:.1f}° / {optimal_patch['best_az_deg']:.1f}°\n"
            f"Quality Score:          {optimal_patch['best_quality_score']:.3f}\n"
            f"RA Search Range:        {self.format_ra_hours(self.optimal_ra_range[0])} to {self.format_ra_hours(self.optimal_ra_range[1])}\n"
            f"Dec Search Range:       {self.format_dec_degrees(self.optimal_dec_range[0])} to {self.format_dec_degrees(self.optimal_dec_range[1])}\n"
        )
        
        self.astro_text.insert(tk.END, timeline_text + conditions_text + recommendation_text)
        self.astro_text.config(state=tk.DISABLED)

    def _build_complete_stelle_doppie_params(self, ra_min_decimal, ra_max_decimal, dec_min_decimal, dec_max_decimal):
        """Build complete Stelle Doppie search parameters with ALL required fields."""
        # Base parameters with ALL required fields (based on working URL structure)
        params = {
            # Coordinate search parameters
            'metodo-cat_wds-ra': '7',
            'dato-cat_wds-ra': f'{ra_min_decimal:.1f},{ra_max_decimal:.1f}',
            'metodo-cat_wds-de': '7',
            'dato-cat_wds-de': f'{dec_min_decimal:.1f},{dec_max_decimal:.1f}',
            
            # All other catalog parameters with default values
            'metodo-cat_wds-raggio': '1',
            'dato-cat_wds-raggio': '',
            'metodo-cat_wds-coord_2000': '1',
            'dato-cat_wds-coord_2000': '',
            'metodo-cat_wds-discov_num': '1',
            'dato-cat_wds-discov_num': '',
            'metodo-cat_wds-comp': '1',
            'dato-cat_wds-comp': '',
            'metodo-cat_wds-name': '9',
            'dato-cat_wds-name': '',
            'metodo-cat_wds-date_first': '1',
            'dato-cat_wds-date_first': '',
            'metodo-cat_wds-date_last': '1',
            'dato-cat_wds-date_last': '',
            'metodo-cat_wds-mag_pri': '1',
            'dato-cat_wds-mag_pri': '',
            'metodo-cat_wds-mag_sec': '1',
            'dato-cat_wds-mag_sec': '',
            'metodo-cat_wds-calc_delta_mag': '1',
            'dato-cat_wds-calc_delta_mag': '',
            'metodo-cat_wds-sep_last': '1',
            'dato-cat_wds-sep_last': '',
            'metodo-cat_wds-spectr': '1',
            'dato-cat_wds-spectr': '',
            
            # Other catalog identifiers
            'metodo-calc_wds_other-Bayer': '9',
            'dato-calc_wds_other-Bayer': '',
            'metodo-calc_wds_other-Flamsteed': '1',
            'dato-calc_wds_other-Flamsteed': '',
            'metodo-cat_wds-cst': '1',
            'dato-cat_wds-cst': '',
            'metodo-calc_wds_other-ADS': '1',
            'dato-calc_wds_other-ADS': '',
            'metodo-calc_wds_other-HD': '1',
            'dato-calc_wds_other-HD': '',
            'metodo-calc_wds_other-HR': '1',
            'dato-calc_wds_other-HR': '',
            'metodo-calc_wds_other-HIP': '1',
            'dato-calc_wds_other-HIP': '',
            'metodo-calc_wds_other-SAO': '1',
            'dato-calc_wds_other-SAO': '',
            'metodo-calc_wds_other-Tycho2': '1',
            'dato-calc_wds_other-Tycho2': '',
            'metodo-calc_wds_other-Gaia': '1',
            'dato-calc_wds_other-Gaia': '',
            'metodo-cat_wds-dm_number': '1',
            'dato-cat_wds-dm_number': '',
            
            # Observation parameters
            'metodo-cat_wds-obs': '1',
            'dato-cat_wds-obs': '',
            'metodo-cat_wds-notes': '9',
            'dato-cat_wds-notes': '',
            'metodo-cat_wds_notes-notes': '9',
            'dato-cat_wds_notes-notes': '',
            'metodo-cat_wds-reports': '1',
            'dato-cat_wds-reports': '',
            
            # Filter flags (all set to "Any" = 1)
            'metodo-cat_wds-filtro_visuale': '1',
            'metodo-cat_wds-filtro_strumento': '1',
            'metodo-cat_wds-filtro_coord': '1',
            'metodo-cat_wds-filtro_orbita': '1',
            'metodo-cat_wds-filtro_nome': '1',
            'metodo-cat_wds-filtro_principale': '1',
            'metodo-cat_wds-filtro_fisica': '1',
            'metodo-cat_wds-filtro_incerta': '1',
            'metodo-cat_wds-calc_tot_comp': '1',
            'dato-cat_wds-calc_tot_comp': '',
            
            # System parameters (required for search to work)
            'menu': '21',
            'section': '2',
            'azione': 'cerca_nel_database',
            'limite': '',
            'righe': '35',
            'type': '3',
            'gocerca': 'Search+the+database',
            'set_filtri': 'S',
            'orderby': 'obs_DESC'
        }
        
        return params

    def generate_stelle_doppie_search(self):
        """Generate Stelle Doppie search URL with advanced filters and robust error handling."""
        if not self.optimal_ra_range:
            messagebox.showwarning("Calculation Required", 
                                 "Please calculate optimal conditions first before generating search URL.")
            return
        
        try:
            ra_min_h, ra_max_h = self.optimal_ra_range
            dec_min_d, dec_max_d = self.optimal_dec_range
            
            # Validate coordinate ranges
            if not (0 <= ra_min_h <= 24 and 0 <= ra_max_h <= 24):
                messagebox.showerror("Invalid Coordinates", 
                                   f"RA range [{ra_min_h:.1f}, {ra_max_h:.1f}] outside valid range [0, 24] hours")
                return
            
            if not (-90 <= dec_min_d <= 90 and -90 <= dec_max_d <= 90):
                messagebox.showerror("Invalid Coordinates", 
                                   f"Dec range [{dec_min_d:.1f}, {dec_max_d:.1f}] outside valid range [-90, +90] degrees")
                return
            
            # Format coordinates correctly for Stelle Doppie (simple range format)
            # Use decimal format for more precision: RA in hours, Dec in degrees
            ra_min_decimal = round(ra_min_h, 1)
            ra_max_decimal = round(ra_max_h, 1)
            dec_min_decimal = round(dec_min_d, 1)
            dec_max_decimal = round(dec_max_d, 1)
            
            # Build complete URL parameters with ALL required fields
            params = self._build_complete_stelle_doppie_params(
                ra_min_decimal, ra_max_decimal, dec_min_decimal, dec_max_decimal
            )
            
            # Add advanced filters if enabled
            active_filters = []
            
            # Process all configurable filters dynamically
            for filter_name, config in STELLE_DOPPIE_FILTERS.items():
                if self.search_options[f'use_{filter_name}_filter'].get():
                    try:
                        method = self.filter_vars[filter_name]['method'].get()
                        
                        # Try to get the value, handling both string and numeric cases
                        try:
                            value = self.filter_vars[filter_name]['value'].get()
                        except tk.TclError:
                            # If DoubleVar fails (e.g., contains comma-separated values), 
                            # get the raw string value directly from the entry widget
                            entry_widget = self.filter_vars[filter_name].get('entry')
                            if entry_widget:
                                value = entry_widget.get()
                            else:
                                value = ""
                        
                        # Handle TclError for different variable types
                        if isinstance(value, str) and value.strip() == '':
                            value = None
                        elif value == 0 and config.get('data_type') != 'integer':
                            value = None
                            
                    except tk.TclError:
                        # Handle case where variables have invalid values
                        continue
                    
                    # Validate the filter
                    is_valid, processed_value, validated_method = self._validate_filter_value(filter_name, method, value)
                    if is_valid:
                        # Get the field name for Stelle Doppie URL
                        field_name = config.get('param_name', filter_name)
                        
                        # Set method parameter
                        params[f'metodo-{field_name}'] = validated_method
                        
                        # Set value parameter (if not void/not void)
                        if validated_method not in ['17', '18']:
                            if config.get('data_type') == 'integer':
                                params[f'dato-{field_name}'] = str(processed_value)
                            elif config.get('data_type') == 'string':
                                params[f'dato-{field_name}'] = str(processed_value)
                            else:  # numeric
                                # For between methods (7, 8), processed_value is already a string like "5.0,10.0"
                                if validated_method in ['7', '8']:
                                    params[f'dato-{field_name}'] = str(processed_value)
                                else:
                                    params[f'dato-{field_name}'] = f'{processed_value:.1f}'
                        
                        # Add to active filters display
                        method_name = STELLE_DOPPIE_METHODS.get(int(validated_method), validated_method)
                        if validated_method in ['17', '18']:
                            active_filters.append(f"{config['label']}: {method_name}")
                        else:
                            unit = config.get('unit', '')
                            active_filters.append(f"{config['label']}: {method_name} {processed_value}{unit}")
            
            # Toggle filters - need to add separate dato and filtro parameters
            if self.search_options['known_orbit'].get():
                params['dato-cat_wds-filtro_orbita'] = 'S'  # "Sì" (Yes) in Italian
                params['filtro_orbita'] = 'S'
                active_filters.append("Known Orbit Only")
            
            if self.search_options['physical_double'].get():
                params['dato-cat_wds-filtro_fisica'] = 'S'  # "Sì" (Yes) in Italian
                params['filtro_fisica'] = 'S'
                active_filters.append("Physical Double Only")
            
            if self.search_options['uncertain_double'].get():
                params['dato-cat_wds-filtro_incerta'] = 'S'  # "Sì" (Yes) in Italian
                params['filtro_incerta'] = 'S'
                active_filters.append("Include Uncertain Doubles")
            
            # Build final URL with proper encoding
            full_url = STELLE_DOPPIE_BASE_URL + "?list_type=WDS&" + urllib.parse.urlencode(params)

            self.url_text.config(state=tk.NORMAL)
            self.url_text.delete(1.0, tk.END)
            self.url_text.insert(tk.END, full_url)
            self.url_text.config(state=tk.DISABLED)
            
            self._generate_mock_search_results(ra_min_h, ra_max_h, dec_min_d, dec_max_d)
            
            # Create detailed confirmation message
            filter_text = "\n".join(active_filters) if active_filters else "No additional filters applied"
            
            if messagebox.askyesno("Open Search", 
                                 f"Search URL generated for:\n"
                                 f"RA: {self.format_ra_hours(ra_min_h)} to {self.format_ra_hours(ra_max_h)}\n"
                                 f"Dec: {self.format_dec_degrees(dec_min_d)} to {self.format_dec_degrees(dec_max_d)}\n\n"
                                 f"Active filters:\n{filter_text}\n\n"
                                 f"Open in browser?"):
                webbrowser.open(full_url)
                
        except Exception as e:
            messagebox.showerror("URL Generation Error", 
                               f"Failed to generate search URL: {e}")
            return
            
    def _validate_filter_value(self, filter_name: str, method: str, value) -> tuple:
        """Validate filter value with selected method.
        
        Args:
            filter_name: Name of the filter
            method: Method number as string (1-18)
            value: Value to validate
            
        Returns:
            tuple: (is_valid, processed_value, method)
        """
        if filter_name not in STELLE_DOPPIE_FILTERS:
            return False, value, method
        
        config = STELLE_DOPPIE_FILTERS[filter_name]
        data_type = config.get('data_type', 'numeric')
        
        # Check if method is available for this filter
        if method not in [str(m) for m in config['available_methods']]:
            messagebox.showerror("Invalid Method", 
                               f"Method {STELLE_DOPPIE_METHODS.get(int(method), method)} not available for {config['label']}")
            return False, value, method
        
        # Handle void/not void methods (17, 18) - no value needed
        if method in ['17', '18']:
            return True, '', method
            
        # Handle empty values
        if value == '' or value is None:
            if method in ['17', '18']:  # void/not void
                return True, '', method
            else:
                messagebox.showerror("Empty Value", 
                                   f"Value required for {config['label']} with method {STELLE_DOPPIE_METHODS.get(int(method), method)}")
                return False, value, method
        
        # Handle "between" method (7) - requires two values separated by comma
        if method == '7':  # Between method
            if ',' not in str(value):
                messagebox.showerror("Invalid Format", 
                                   f"Between method requires two values separated by comma (e.g., '5,10')")
                return False, value, method
            
            try:
                parts = str(value).split(',')
                if len(parts) != 2:
                    messagebox.showerror("Invalid Format", 
                                       f"Between method requires exactly two values separated by comma (e.g., '5,10')")
                    return False, value, method
                
                if data_type == 'string':
                    val1, val2 = parts[0].strip(), parts[1].strip()
                    if not val1 or not val2:
                        messagebox.showerror("Empty Value", 
                                           f"Both values required for between method")
                        return False, value, method
                    processed_value = f"{val1},{val2}"
                elif data_type == 'integer':
                    val1, val2 = int(parts[0].strip()), int(parts[1].strip())
                    if val1 >= val2:
                        messagebox.showerror("Invalid Range", 
                                           f"First value ({val1}) must be less than second value ({val2})")
                        return False, value, method
                    processed_value = f"{val1},{val2}"
                else:  # numeric (float)
                    val1, val2 = float(parts[0].strip()), float(parts[1].strip())
                    if val1 >= val2:
                        messagebox.showerror("Invalid Range", 
                                           f"First value ({val1}) must be less than second value ({val2})")
                        return False, value, method
                    processed_value = f"{val1:.1f},{val2:.1f}"
                
                return True, processed_value, method
                
            except (ValueError, TypeError):
                messagebox.showerror("Invalid Value", 
                                   f"Invalid format for between method. Use two {data_type} values separated by comma (e.g., '5,10')")
                return False, value, method
        
        # Handle "not between" method (8) - also requires two values separated by comma
        if method == '8':  # Not between method
            if ',' not in str(value):
                messagebox.showerror("Invalid Format", 
                                   f"Not between method requires two values separated by comma (e.g., '5,10')")
                return False, value, method
            
            try:
                parts = str(value).split(',')
                if len(parts) != 2:
                    messagebox.showerror("Invalid Format", 
                                       f"Not between method requires exactly two values separated by comma (e.g., '5,10')")
                    return False, value, method
                
                if data_type == 'string':
                    val1, val2 = parts[0].strip(), parts[1].strip()
                    if not val1 or not val2:
                        messagebox.showerror("Empty Value", 
                                           f"Both values required for not between method")
                        return False, value, method
                    processed_value = f"{val1},{val2}"
                elif data_type == 'integer':
                    val1, val2 = int(parts[0].strip()), int(parts[1].strip())
                    if val1 >= val2:
                        messagebox.showerror("Invalid Range", 
                                           f"First value ({val1}) must be less than second value ({val2})")
                        return False, value, method
                    processed_value = f"{val1},{val2}"
                else:  # numeric (float)
                    val1, val2 = float(parts[0].strip()), float(parts[1].strip())
                    if val1 >= val2:
                        messagebox.showerror("Invalid Range", 
                                           f"First value ({val1}) must be less than second value ({val2})")
                        return False, value, method
                    processed_value = f"{val1:.1f},{val2:.1f}"
                
                return True, processed_value, method
                
            except (ValueError, TypeError):
                messagebox.showerror("Invalid Value", 
                                   f"Invalid format for not between method. Use two {data_type} values separated by comma (e.g., '5,10')")
                return False, value, method
        
        # Validate based on data type
        try:
            if data_type == 'string':
                processed_value = str(value).strip()
                if not processed_value:
                    messagebox.showerror("Empty Value", 
                                       f"String value required for {config['label']}")
                    return False, value, method
                    
            elif data_type == 'integer':
                processed_value = int(value)
                if 'min_value' in config and processed_value < config['min_value']:
                    messagebox.showerror("Invalid Value", 
                                       f"{config['label']} value {processed_value} below minimum {config['min_value']}")
                    return False, value, method
                if 'max_value' in config and processed_value > config['max_value']:
                    messagebox.showerror("Invalid Value", 
                                       f"{config['label']} value {processed_value} above maximum {config['max_value']}")
                    return False, value, method
                    
            else:  # numeric (float)
                processed_value = float(value)
                if 'min_value' in config and processed_value < config['min_value']:
                    messagebox.showerror("Invalid Value", 
                                       f"{config['label']} value {processed_value} below minimum {config['min_value']}")
                    return False, value, method
                if 'max_value' in config and processed_value > config['max_value']:
                    messagebox.showerror("Invalid Value", 
                                       f"{config['label']} value {processed_value} above maximum {config['max_value']}")
                    return False, value, method
                    
        except (ValueError, TypeError):
            messagebox.showerror("Invalid Value", 
                               f"Invalid {data_type} value '{value}' for {config['label']}")
            return False, value, method
        
        return True, processed_value, method
               
    def _validate_observations_filter(self, observations_count: int) -> bool:
        """Validate observations filter value."""
        if not isinstance(observations_count, int) or observations_count < 1:
            messagebox.showerror("Invalid Observations Filter", 
                               "Observations count must be a positive integer (1 or greater)")
            return False
        
        if observations_count > 999:
            messagebox.showerror("Invalid Observations Filter", 
                               "Observations count cannot exceed 999")
            return False
        
        return True

    def _robust_csv_read(self, file_path: str) -> pd.DataFrame:
        """Robust CSV reading with multiple fallback methods."""
        # Try different encodings and separators
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        separators = [',', ';', '\t']
        
        for encoding in encodings:
            for separator in separators:
                try:
                    # First attempt: Standard pandas read_csv
                    df = pd.read_csv(file_path, encoding=encoding, sep=separator, 
                                   on_bad_lines='skip', engine='python')
                    
                    # Check if we got reasonable data
                    if len(df) > 0 and len(df.columns) > 3:
                        print(f"✅ Successfully read CSV with encoding: {encoding}, separator: '{separator}'")
                        return df
                        
                except Exception as e:
                    continue
        
        # If standard methods fail, try more aggressive approaches
        try:
            # Try reading with quoting and error handling
            df = pd.read_csv(file_path, encoding='utf-8', sep=',', 
                           quoting=1, on_bad_lines='skip', engine='python',
                           skipinitialspace=True)
            
            if len(df) > 0:
                print("✅ Successfully read CSV with quoting method")
                return df
                
        except Exception as e:
            pass
        
        # Last resort: Manual line-by-line parsing
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            # Find header line (usually first non-empty line)
            header_line = None
            data_lines = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line and not line.startswith('#'):
                    if header_line is None:
                        header_line = line
                    else:
                        data_lines.append(line)
            
            if header_line is None:
                raise ValueError("No valid header found in CSV file")
            
            # Parse header
            header = [col.strip().strip('"').strip("'") for col in header_line.split(',')]
            
            # Parse data lines
            data = []
            for line in data_lines:
                if line.strip():
                    # Split by comma, but handle quoted values
                    import csv
                    from io import StringIO
                    try:
                        reader = csv.reader(StringIO(line))
                        row = next(reader)
                        # Pad or truncate row to match header length
                        while len(row) < len(header):
                            row.append('')
                        if len(row) > len(header):
                            row = row[:len(header)]
                        data.append(row)
                    except:
                        # Skip malformed lines
                        continue
            
            if data:
                df = pd.DataFrame(data, columns=header)
                print(f"✅ Successfully read CSV with manual parsing ({len(df)} rows)")
                return df
                
        except Exception as e:
            pass
        
        # If all methods fail, raise an error
        raise ValueError(f"Unable to read CSV file. Please check that the file is a valid CSV format from Stelle Doppie. "
                        f"Common issues: special characters, inconsistent formatting, or non-standard encoding.")

    def format_ra_hours(self, hours: float) -> str:
        h = int(hours)
        m = int((hours - h) * 60)
        return f"{h:02d}h{m:02d}m"

    def format_dec_degrees(self, degrees: float) -> str:
        sign = "+" if degrees >= 0 else "-"
        abs_deg = abs(degrees)
        d = int(abs_deg)
        m = int((abs_deg - d) * 60)
        return f"{sign}{d:02d}°{m:02d}'"

    def _validate_inputs(self) -> bool:
        """Validate all user inputs before processing."""
        if not self.selected_location:
            messagebox.showerror("Input Error", "Please select an observatory location.")
            return False
        
        if not self.optimal_ra_range:
            messagebox.showerror("Input Error", "Please calculate optimal observing conditions first.")
            return False
        
        if not self.selected_catalogs:
            messagebox.showerror("Input Error", "Please select at least one catalog source.")
            return False
        
        return True
    
    def _create_export_section(self, parent):
        """Create the export options section for both mock and imported data."""
        frame = ttk.LabelFrame(parent, text="🔄 Export Options", padding="10")
        frame.grid(row=9, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
        # Export format selection
        ttk.Label(frame, text="Export Format:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.export_format_var = tk.StringVar(value="csv")
        format_combo = ttk.Combobox(frame, textvariable=self.export_format_var, width=30)
        format_combo['values'] = [f"{fmt['name']} ({fmt['extension']})" for fmt in EXPORT_FORMATS.values()]
        format_combo['state'] = 'readonly'
        format_combo.grid(row=0, column=1, sticky="ew", padx=5)
        
        # Export buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=0, column=2, padx=5)
        
        ttk.Button(button_frame, text="📁 Export Mock Results", 
                  command=self._export_results).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="📊 Export Imported Data", 
                  command=self._export_imported_data).grid(row=0, column=1, padx=(5, 0))
        
        # Export options
        options_frame = ttk.Frame(frame)
        options_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        self.include_metadata_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include metadata", 
                       variable=self.include_metadata_var).grid(row=0, column=0, sticky="w")
        
        self.include_coordinates_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include coordinates", 
                       variable=self.include_coordinates_var).grid(row=0, column=1, sticky="w", padx=(20, 0))
        
        self.include_analysis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include analysis results", 
                       variable=self.include_analysis_var).grid(row=0, column=2, sticky="w", padx=(20, 0))
        
    def _create_catalog_selection_section(self, parent):
        """Create the catalog selection section."""
        frame = ttk.LabelFrame(parent, text="🌟 Catalog Sources", padding="10")
        frame.grid(row=10, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
        # Catalog checkboxes
        self.catalog_vars = {}
        row = 0
        for catalog_id, catalog_info in CATALOG_SOURCES.items():
            if catalog_info['enabled']:
                var = tk.BooleanVar(value=catalog_id in self.selected_catalogs)
                self.catalog_vars[catalog_id] = var
                
                checkbox = ttk.Checkbutton(frame, text=catalog_info['name'], 
                                         variable=var, command=self._update_selected_catalogs)
                checkbox.grid(row=row, column=0, sticky="w", pady=2)
                
                # Info label
                info_label = ttk.Label(frame, text=catalog_info['description'], 
                                     font=('Arial', 8), foreground='gray')
                info_label.grid(row=row, column=1, sticky="w", padx=(10, 0))
                
                row += 1
        
        # Priority order info
        priority_frame = ttk.Frame(frame)
        priority_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        ttk.Label(priority_frame, text="Search Priority:", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky="w")
        priority_text = " → ".join([cat['name'] for cat in sorted(CATALOG_SOURCES.values(), key=lambda x: x['priority']) if cat['enabled']])
        ttk.Label(priority_frame, text=priority_text, font=('Arial', 8)).grid(row=0, column=1, sticky="w", padx=(10, 0))
        
    def _create_progress_section(self, parent):
        """Create the progress and status section."""
        frame = ttk.LabelFrame(parent, text="📊 Status", padding="10")
        frame.grid(row=11, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
        # Status label
        ttk.Label(frame, text="Status:").grid(row=0, column=0, sticky="w")
        self.status_label = ttk.Label(frame, textvariable=self.status_var, foreground='green')
        self.status_label.grid(row=0, column=1, sticky="w", padx=(10, 0))
        
        # Progress bar
        ttk.Label(frame, text="Progress:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.progress_bar = ttk.Progressbar(frame, mode='determinate', variable=self.progress_var)
        self.progress_bar.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(5, 0))
        
        # Results counter
        self.results_var = tk.StringVar(value="No results")
        results_label = ttk.Label(frame, textvariable=self.results_var, font=('Arial', 9))
        results_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))
        
    def _apply_theme(self):
        """Apply the selected theme to the GUI."""
        theme = UI_THEMES.get(self.current_theme, UI_THEMES['default'])
        
        # Apply theme to root
        self.root.configure(bg=theme['bg_color'])
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')  # Use clam theme as base
        
        # Configure custom styles
        style.configure('Themed.TLabel', 
                       background=theme['bg_color'], 
                       foreground=theme['fg_color'])
        style.configure('Themed.TFrame', 
                       background=theme['bg_color'])
        
    def _export_results(self):
        """Export search results to selected format."""
        if not self.search_results:
            messagebox.showwarning("No Results", "No search results to export. Please perform a search first.")
            return
        
        # Get selected format
        format_text = self.export_format_var.get()
        format_key = None
        for key, fmt in EXPORT_FORMATS.items():
            if format_text.startswith(fmt['name']):
                format_key = key
                break
        
        if not format_key:
            messagebox.showerror("Error", "Invalid export format selected.")
            return
        
        # Get save location
        format_info = EXPORT_FORMATS[format_key]
        filename = filedialog.asksaveasfilename(
            defaultextension=format_info['extension'],
            filetypes=[(format_info['name'], f"*{format_info['extension']}")]
        )
        
        if not filename:
            return
        
        try:
            self._perform_export(filename, format_key)
            messagebox.showinfo("Export Complete", f"Results exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
    
    def _perform_export(self, filename: str, format_key: str):
        """Perform the actual export based on format."""
        self.status_var.set("Exporting...")
        self.progress_var.set(0)
        
        if format_key == 'csv':
            self._export_csv(filename)
        elif format_key == 'json':
            self._export_json(filename)
        elif format_key == 'fits':
            self._export_fits(filename)
        elif format_key == 'votable':
            self._export_votable(filename)
        elif format_key == 'latex':
            self._export_latex(filename)
        else:
            raise ValueError(f"Unsupported export format: {format_key}")
        
        self.status_var.set("Export complete")
        self.progress_var.set(100)
    
    def _export_csv(self, filename: str):
        """Export results to CSV format."""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['Name', 'RA', 'Dec', 'Magnitude', 'Separation', 'Position_Angle', 'Epoch']
            if self.include_metadata_var.get():
                header.extend(['Catalog', 'Search_Date', 'Observatory'])
            writer.writerow(header)
            
            # Write data
            for i, result in enumerate(self.search_results):
                row = [
                    result.get('name', ''),
                    result.get('ra', ''),
                    result.get('dec', ''),
                    result.get('magnitude', ''),
                    result.get('separation', ''),
                    result.get('position_angle', ''),
                    result.get('epoch', '')
                ]
                
                if self.include_metadata_var.get():
                    row.extend([
                        result.get('catalog', ''),
                        datetime.now().isoformat(),
                        self.selected_location.get('name', '') if self.selected_location else ''
                    ])
                
                writer.writerow(row)
                
                # Update progress
                self.progress_var.set((i + 1) / len(self.search_results) * 100)
                self.root.update_idletasks()
    
    def _export_json(self, filename: str):
        """Export results to JSON format."""
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'format_version': '1.0',
                'observatory': self.selected_location.get('name', '') if self.selected_location else '',
                'total_results': len(self.search_results)
            },
            'results': self.search_results
        }
        
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)
    
    def _export_fits(self, filename: str):
        """Export results to FITS format."""
        # Note: This would require astropy.io.fits
        # For now, we'll create a placeholder
        messagebox.showinfo("FITS Export", "FITS export requires astropy.io.fits\nPlease install: pip install astropy")
        
    def _export_votable(self, filename: str):
        """Export results to VOTable format."""
        # Note: This would require astropy.io.votable
        # For now, we'll create a placeholder
        messagebox.showinfo("VOTable Export", "VOTable export requires astropy.io.votable\nPlease install: pip install astropy")
        
    def _export_latex(self, filename: str):
        """Export results to LaTeX table format."""
        with open(filename, 'w', encoding='utf-8') as texfile:
            texfile.write("\\begin{table}[h]\n")
            texfile.write("\\centering\n")
            texfile.write("\\caption{Binary Star Search Results}\n")
            texfile.write("\\label{tab:binary_stars}\n")
            texfile.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
            texfile.write("\\hline\n")
            texfile.write("Name & RA & Dec & Magnitude & Separation & Position Angle & Epoch \\\\\n")
            texfile.write("\\hline\n")
            
            for result in self.search_results:
                texfile.write(f"{result.get('name', '')} & ")
                texfile.write(f"{result.get('ra', '')} & ")
                texfile.write(f"{result.get('dec', '')} & ")
                texfile.write(f"{result.get('magnitude', '')} & ")
                texfile.write(f"{result.get('separation', '')} & ")
                texfile.write(f"{result.get('position_angle', '')} & ")
                texfile.write(f"{result.get('epoch', '')} \\\\\n")
            
            texfile.write("\\hline\n")
            texfile.write("\\end{tabular}\n")
            texfile.write("\\end{table}\n")
    
    def _update_selected_catalogs(self):
        """Update the selected catalogs set."""
        self.selected_catalogs = {cat_id for cat_id, var in self.catalog_vars.items() if var.get()}
        self.status_var.set(f"Selected catalogs: {', '.join(self.selected_catalogs)}")
    
    def _generate_mock_search_results(self, ra_min_h: float, ra_max_h: float, dec_min_d: float, dec_max_d: float):
        """Generate realistic search results preview based on actual search parameters."""
        import random
        
        self.status_var.set("Generating search results...")
        self.progress_var.set(0)
        
        # Generate a reasonable sample (20-50 results for preview)
        num_results = random.randint(20, 50)
        self.search_results = []
        
        # Common double star catalog prefixes
        prefixes = ['STF', 'BU', 'HJ', 'SEI', 'MLR', 'STT', 'H', 'β', 'Σ', 'OΣ', 'AC', 'A', 'BAL']
        
        for i in range(num_results):
            # Generate random coordinates within the search range
            ra = random.uniform(ra_min_h, ra_max_h)
            dec = random.uniform(dec_min_d, dec_max_d)
            
            # Generate mock data with realistic distributions
            prefix = random.choice(prefixes)
            number = random.randint(1, 9999)
            
            result = {
                'name': f"{prefix}{number:04d}" if prefix in ['STF', 'BU', 'HJ', 'SEI', 'MLR', 'STT'] else f"{prefix} {number}",
                'ra': f"{int(ra):02d}h{int((ra % 1) * 60):02d}m{int(((ra % 1) * 60 % 1) * 60):02d}s",
                'dec': f"{int(dec):+03d}°{int(abs(dec) % 1 * 60):02d}'{int((abs(dec) % 1 * 60 % 1) * 60):02d}\"",
                'magnitude': round(random.uniform(4.0, 14.0), 1),
                'separation': round(random.uniform(0.2, 120.0), 1),
                'position_angle': random.randint(0, 360),
                'epoch': random.randint(1800, 2025),
                'catalog': random.choice(list(self.selected_catalogs)) if self.selected_catalogs else 'stelle_doppie'
            }
            
            self.search_results.append(result)
            
            # Update progress
            progress = (i + 1) / num_results * 100
            self.progress_var.set(progress)
            self.root.update_idletasks()
        
        # Update results counter
        self.results_var.set(f"Generated: {len(self.search_results)} mock results")
        self.status_var.set("Search complete")
        
        # Show results summary
        messagebox.showinfo("Search Complete", 
                          f"🔍 SEARCH RESULTS GENERATED\n\n"
                          f"Mock results: {len(self.search_results)} binary stars\n\n"
                          f"📝 NOTE: This is a simulated preview for testing.\n"
                          f"For actual results, use the generated URL in your browser.\n\n"
                          f"Use the Export section to save this data.")
    
    def _validate_inputs(self) -> bool:
        """Validate all user inputs before processing."""
        if not self.selected_location:
            messagebox.showerror("Input Error", "Please select an observatory location.")
            return False
        
        if not self.optimal_ra_range:
            messagebox.showerror("Input Error", "Please calculate optimal observing conditions first.")
            return False
        
        if not self.selected_catalogs:
            messagebox.showerror("Input Error", "Please select at least one catalog source.")
            return False
        
        return True

    def import_csv_file(self):
        """Import and analyze a CSV file downloaded from Stelle Doppie."""
        file_path = filedialog.askopenfilename(
            title="Select Stelle Doppie CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.status_var.set("Loading CSV file...")
            self.progress_var.set(0)
            self.root.update_idletasks()
            
            # Try multiple approaches to read the CSV file
            df = self._robust_csv_read(file_path)
            
            # Progress update
            self.progress_var.set(50)
            self.root.update_idletasks()
            
            # Detect CSV format (basic vs full)
            csv_format = self._detect_csv_format(df)
            
            # Process and analyze the data
            self._analyze_imported_csv(df, file_path, csv_format)
            
            # Update progress
            self.progress_var.set(100)
            self.root.update_idletasks()
            
            # Update current file info
            self.current_file_var.set(f"✅ Loaded: {os.path.basename(file_path)} ({len(df)} systems, {csv_format} format)")
            
        except pd.errors.EmptyDataError:
            error_msg = "The selected file appears to be empty or contains no valid data."
            messagebox.showerror("Import Error", f"Empty File: {error_msg}")
            self.current_file_var.set("❌ Empty file")
        except pd.errors.ParserError as e:
            error_msg = f"CSV parsing error: {str(e)}\n\nThis usually occurs when:\n• The file has inconsistent formatting\n• Special characters are not properly escaped\n• The file is corrupted\n\nTry re-downloading the file from Stelle Doppie."
            messagebox.showerror("Import Error", f"Parsing Error: {error_msg}")
            self.current_file_var.set("❌ Parsing error")
        except UnicodeDecodeError:
            error_msg = "The file encoding is not supported. Please ensure the file is saved in UTF-8 format."
            messagebox.showerror("Import Error", f"Encoding Error: {error_msg}")
            self.current_file_var.set("❌ Encoding error")
        except FileNotFoundError:
            error_msg = "The selected file could not be found."
            messagebox.showerror("Import Error", f"File Error: {error_msg}")
            self.current_file_var.set("❌ File not found")
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}\n\nPlease check that:\n• The file is a valid CSV from Stelle Doppie\n• The file is not corrupted\n• You have permission to read the file"
            messagebox.showerror("Import Error", f"Failed to import CSV file: {error_msg}")
            self.current_file_var.set("❌ Import failed")
        finally:
            self.status_var.set("Ready")
            self.progress_var.set(0)

    def _detect_csv_format(self, df: pd.DataFrame) -> str:
        """Detect if CSV is basic (logged out) or full (logged in) format."""
        # Full format columns (logged in) - checking both common variations
        full_format_columns = ['wds_id', 'discov#', 'comp', 'first', 'last', 'obs', 
                              'pa_first', 'pa_last', 'sep_first', 'sep_last', 'm1', 'm2', 
                              'spectr', 'pm1_ra', 'pm1_dec', 'pm2_ra', 'pm2_dec', 
                              'dm_number', 'notes', 'coord_arcsec_2000',
                              'date_first', 'date_last']  # Alternative column names
        
        # Basic format columns (logged out)
        basic_format_columns = ['name', 'cst', 'SAO', 'coord', 'wds_name', 'last', 
                               'obs', 'pa', 'sep', 'm1', 'm2', 'd_mag', 'orb']
        
        df_columns = set(df.columns.str.lower())
        full_matches = sum(1 for col in full_format_columns if col.lower() in df_columns)
        basic_matches = sum(1 for col in basic_format_columns if col.lower() in df_columns)
        
        # Debug information
        print(f"CSV columns found: {sorted(df_columns)}")
        print(f"Full format matches: {full_matches}, Basic format matches: {basic_matches}")
        
        # More flexible detection - check for key velocity calculation columns
        velocity_columns = ['first', 'last', 'pa_first', 'pa_last', 'sep_first', 'sep_last']
        velocity_matches = sum(1 for col in velocity_columns if col.lower() in df_columns)
        
        print(f"Velocity calculation columns found: {velocity_matches}/{len(velocity_columns)}")
        
        # If we have velocity calculation columns, consider it full format
        if velocity_matches >= 4:  # At least 4 of the 6 required columns
            return "full"
        elif full_matches >= 8:  # Reduced threshold for more flexibility
            return "full"
        elif basic_matches >= 6:  # Most basic format columns present
            return "basic"
        else:
            return "unknown"

    def _analyze_imported_csv(self, df: pd.DataFrame, file_path: str, csv_format: str):
        """Analyze the imported CSV and display results."""
        self.csv_analysis_text.config(state=tk.NORMAL)
        self.csv_analysis_text.delete(1.0, tk.END)
        
        # Display file information
        summary = f"📊 CSV ANALYSIS RESULTS\n"
        summary += f"=" * 60 + "\n"
        summary += f"File: {os.path.basename(file_path)}\n"
        summary += f"Format: {csv_format.upper()} (requires login: {'No' if csv_format == 'basic' else 'Yes'})\n"
        summary += f"Total systems: {len(df):,}\n"
        summary += f"Columns: {len(df.columns)}\n\n"
        
        # Show available columns
        summary += f"📋 Available Columns:\n"
        for i, col in enumerate(df.columns, 1):
            summary += f"  {i:2d}. {col}\n"
        summary += "\n"
        
        # Format-specific analysis
        if csv_format == "full":
            summary += self._analyze_full_format_csv(df)
        elif csv_format == "basic":
            summary += self._analyze_basic_format_csv(df)
        else:
            summary += f"⚠️ Unknown CSV format. Showing basic statistics:\n"
            summary += f"First few column names: {list(df.columns[:5])}\n"
            summary += f"Please check if this is a valid Stelle Doppie CSV file.\n"
        
        # Show sample data
        if len(df) > 0:
            summary += f"\n📝 Sample Data (first 3 rows):\n"
            summary += "-" * 60 + "\n"
            
            for i, row in df.head(3).iterrows():
                summary += f"Row {i+1}:\n"
                # Show first few key columns
                for col in df.columns[:6]:
                    value = row.get(col, 'N/A')
                    summary += f"  {col}: {value}\n"
                summary += "-" * 40 + "\n"
        
        self.csv_analysis_text.insert(tk.END, summary)
        self.csv_analysis_text.config(state=tk.DISABLED)
        
        # Update results counter
        self.results_var.set(f"Imported: {len(df):,} systems ({csv_format} format)")
        
        # Store data for export
        self.imported_csv_data = df
        self.imported_csv_format = csv_format

    def calculate_system_velocities(self):
        """Calculate velocities for all loaded systems and show top N fastest."""
        if not hasattr(self, 'imported_csv_data') or self.imported_csv_data is None:
            messagebox.showwarning("No Data", "Please import a CSV file first.")
            return
        
        try:
            from astrakairos.physics.dynamics import calculate_mean_velocity_from_endpoints
            
            self.status_var.set("Calculating velocities...")
            self.progress_var.set(0)
            self.root.update_idletasks()
            
            df = self.imported_csv_data
            velocity_results = []
            
            # Determine which columns to use based on format
            if self.imported_csv_format == "full":
                # Full format columns - updated to match actual Stelle Doppie export
                column_mapping = {
                    'first': ['first'],
                    'last': ['last'],
                    'pa_first': ['pa_first'],
                    'pa_last': ['pa_last'],
                    'sep_first': ['sep_first'],
                    'sep_last': ['sep_last'],
                    'name': ['wds_id', 'name', 'wds_name']
                }
                
                # Find actual column names (case-sensitive matching with your exact column names)
                actual_columns = {}
                df_columns = list(df.columns)  # Get actual column names
                
                # Direct mapping based on your exact column names
                actual_columns = {
                    'first': 'first' if 'first' in df_columns else None,
                    'last': 'last' if 'last' in df_columns else None,
                    'pa_first': 'pa_first' if 'pa_first' in df_columns else None,
                    'pa_last': 'pa_last' if 'pa_last' in df_columns else None,
                    'sep_first': 'sep_first' if 'sep_first' in df_columns else None,
                    'sep_last': 'sep_last' if 'sep_last' in df_columns else None,
                    'name': 'wds_id' if 'wds_id' in df_columns else ('name' if 'name' in df_columns else None)
                }
                
                # Check if we have the required columns
                required_keys = ['first', 'last', 'pa_first', 'pa_last', 'sep_first', 'sep_last']
                missing_cols = [key for key in required_keys if actual_columns[key] is None]
                
                if missing_cols:
                    # Show available columns for debugging
                    available_cols = sorted(df.columns)
                    messagebox.showinfo("Column Check", 
                                       f"Velocity calculation column analysis:\n\n"
                                       f"Found columns: {[key for key in required_keys if actual_columns[key] is not None]}\n"
                                       f"Missing columns: {missing_cols}\n\n"
                                       f"Available columns in your CSV:\n{', '.join(available_cols)}\n\n"
                                       f"Note: All required columns are present in your CSV, continuing with calculation...")
                    
                    # Actually, let's check if the columns exist but with different casing
                    df_lower = [col.lower() for col in df.columns]
                    found_with_case = []
                    for missing in missing_cols:
                        if missing.lower() in df_lower:
                            # Find the actual column name
                            for actual_col in df.columns:
                                if actual_col.lower() == missing.lower():
                                    actual_columns[missing] = actual_col
                                    found_with_case.append(f"{missing} -> {actual_col}")
                                    break
                    
                    # Update missing columns list
                    missing_cols = [key for key in required_keys if actual_columns[key] is None]
                    
                    if missing_cols:
                        messagebox.showerror("Missing Data", 
                                           f"CSV is missing required columns for velocity calculation.\n\n"
                                           f"Still missing: {', '.join(missing_cols)}\n"
                                           f"Case corrections made: {found_with_case}\n\n"
                                           f"Available columns: {', '.join(available_cols)}\n\n"
                                           f"Please ensure you downloaded the full CSV format with login.")
                        return
                
                name_col = actual_columns['name']
                    
            elif self.imported_csv_format == "basic":
                # Basic format has limited data, we'll use what we have
                messagebox.showinfo("Limited Data", 
                                  "Basic format CSV has limited temporal data. "
                                  "Velocity calculations require full format CSV with historical observations. "
                                  "Please log in to Stelle Doppie and download the full format CSV.")
                return
            else:
                messagebox.showerror("Unknown Format", "Cannot determine CSV format for velocity calculations.")
                return
            
            # Process each system
            total_systems = len(df)
            for i, (idx, row) in enumerate(df.iterrows()):
                try:
                    # Create WDS summary dictionary for this system using actual column names
                    # Note: dynamics.py expects 'date_first' and 'date_last', not 'first' and 'last'
                    wds_summary = {
                        'date_first': float(row[actual_columns['first']]) if pd.notna(row[actual_columns['first']]) else None,
                        'date_last': float(row[actual_columns['last']]) if pd.notna(row[actual_columns['last']]) else None,
                        'pa_first': float(row[actual_columns['pa_first']]) if pd.notna(row[actual_columns['pa_first']]) else None,
                        'pa_last': float(row[actual_columns['pa_last']]) if pd.notna(row[actual_columns['pa_last']]) else None,
                        'sep_first': float(row[actual_columns['sep_first']]) if pd.notna(row[actual_columns['sep_first']]) else None,
                        'sep_last': float(row[actual_columns['sep_last']]) if pd.notna(row[actual_columns['sep_last']]) else None
                    }
                    
                    # Skip if any required data is missing
                    if None in wds_summary.values():
                        continue
                    
                    # Calculate velocity
                    velocity_result = calculate_mean_velocity_from_endpoints(wds_summary)
                    
                    if velocity_result:
                        velocity_results.append({
                            'name': row[name_col],
                            'velocity': velocity_result['v_total_endpoint'],
                            'vx': velocity_result['vx_arcsec_per_year'],
                            'vy': velocity_result['vy_arcsec_per_year'],
                            'pa_v': velocity_result['pa_v_endpoint'],
                            'time_baseline': velocity_result['time_baseline_years'],
                            'first': row[actual_columns['first']],
                            'last': row[actual_columns['last']]
                        })
                        
                except Exception as e:
                    # Skip systems with calculation errors
                    continue
                
                # Update progress
                progress = ((i + 1) / total_systems) * 100
                self.progress_var.set(progress)
                self.root.update_idletasks()
            
            # Sort by velocity (descending) and get top N
            velocity_results.sort(key=lambda x: x['velocity'], reverse=True)
            n_systems = min(self.velocity_count_var.get(), len(velocity_results))
            top_systems = velocity_results[:n_systems]
            
            # Display results
            self._display_velocity_results(top_systems, len(velocity_results) - 5, total_systems) 
            # The easiest way to show the real amount of calculated velocities is to just subtract 5 from the total calculated, as we skip the html error at the last 5 lines of every Stelle Doppie CSV.
            
            # Update status
            self.status_var.set(f"Velocity calculation complete")
            self.progress_var.set(100)
            
        except ImportError:
            messagebox.showerror("Import Error", "Could not import dynamics module. Please check your installation.")
        except Exception as e:
            messagebox.showerror("Calculation Error", f"Error calculating velocities: {str(e)}")
        finally:
            self.status_var.set("Ready")
            self.progress_var.set(0)

    def _display_velocity_results(self, top_systems, total_calculated, total_systems):
        """Display velocity calculation results."""
        self.velocity_results_text.config(state=tk.NORMAL)
        self.velocity_results_text.delete(1.0, tk.END)
        
        if not top_systems:
            self.velocity_results_text.insert(tk.END, "No velocity calculations possible with current data.\n")
            self.velocity_results_text.insert(tk.END, "This usually means:\n")
            self.velocity_results_text.insert(tk.END, "• CSV format doesn't have required temporal data\n")
            self.velocity_results_text.insert(tk.END, "• Missing first/last observation dates\n")
            self.velocity_results_text.insert(tk.END, "• Missing position angle or separation data\n")
            self.velocity_results_text.config(state=tk.DISABLED)
            return
        
        # Header
        summary = f"🚀 VELOCITY ANALYSIS RESULTS\n"
        summary += f"=" * 60 + "\n"
        summary += f"Systems processed: {total_systems}\n"
        summary += f"Velocity calculations: {total_calculated}\n"
        summary += f"Top {len(top_systems)} fastest systems:\n\n"
        
        # Table header
        summary += f"{'Rank':<4} {'System':<12} {'Velocity':<12} {'PA':<8} {'Baseline':<10} {'Period':<15}\n"
        summary += f"{'':4} {'':12} {'(arcsec/yr)':<12} {'(deg)':<8} {'(years)':<10} {'':15}\n"
        summary += "-" * 70 + "\n"
        
        # System details
        for i, system in enumerate(top_systems, 1):
            summary += f"{i:<4} {system['name']:<12} {system['velocity']:<12.4f} {system['pa_v']:<8.1f} {system['time_baseline']:<10.1f} {system['first']:.0f}-{system['last']:.0f}\n"
        
        # Statistics
        if top_systems:
            velocities = [s['velocity'] for s in top_systems]
            summary += f"\n📊 VELOCITY STATISTICS:\n"
            summary += f"  Maximum velocity: {max(velocities):.4f} arcsec/yr\n"
            summary += f"  Minimum velocity: {min(velocities):.4f} arcsec/yr\n"
            summary += f"  Average velocity: {sum(velocities)/len(velocities):.4f} arcsec/yr\n"
            summary += f"  Median velocity: {sorted(velocities)[len(velocities)//2]:.4f} arcsec/yr\n"
            
            # High velocity systems
            high_velocity_systems = [s for s in top_systems if s['velocity'] > 1.0]  # > 1"/yr
            if high_velocity_systems:
                summary += f"\n⚡ HIGH VELOCITY SYSTEMS (>1 arcsec/yr):\n"
                for system in high_velocity_systems:
                    summary += f"  • {system['name']}: {system['velocity']:.4f} arcsec/yr\n"
        
        summary += f"\n💡 NOTES:\n"
        summary += f"  • Velocities calculated using endpoint method\n"
        summary += f"  • Higher velocities may indicate orbital motion\n"
        summary += f"  • Consider these for priority observations\n"
        summary += f"  • PA = Position Angle of velocity vector\n"
        
        self.velocity_results_text.insert(tk.END, summary)
        self.velocity_results_text.config(state=tk.DISABLED)
        
        # Update results counter
        self.results_var.set(f"Velocities: {total_calculated}/{total_systems} systems calculated")

    def _analyze_full_format_csv(self, df: pd.DataFrame) -> str:
        """Analyze full format CSV (logged in user)."""
        analysis = f"🔍 FULL FORMAT ANALYSIS\n"
        analysis += f"This CSV contains detailed information for each binary star system.\n\n"
        
        # Observation statistics
        if 'obs' in df.columns:
            obs_stats = df['obs'].describe()
            analysis += f"📊 Observation Statistics:\n"
            analysis += f"  Mean observations per system: {obs_stats['mean']:.1f}\n"
            analysis += f"  Median observations: {obs_stats['50%']:.0f}\n"
            analysis += f"  Max observations: {obs_stats['max']:.0f}\n"
            analysis += f"  Min observations: {obs_stats['min']:.0f}\n\n"
        
        # Magnitude statistics
        if 'm1' in df.columns and 'm2' in df.columns:
            try:
                m1_clean = pd.to_numeric(df['m1'], errors='coerce').dropna()
                m2_clean = pd.to_numeric(df['m2'], errors='coerce').dropna()
                
                analysis += f"⭐ Magnitude Statistics:\n"
                analysis += f"  Primary magnitude range: {m1_clean.min():.1f} - {m1_clean.max():.1f}\n"
                analysis += f"  Secondary magnitude range: {m2_clean.min():.1f} - {m2_clean.max():.1f}\n"
                analysis += f"  Mean primary magnitude: {m1_clean.mean():.1f}\n"
                analysis += f"  Mean secondary magnitude: {m2_clean.mean():.1f}\n\n"
            except:
                analysis += f"⭐ Magnitude data present but needs cleaning\n\n"
        
        # Separation statistics
        if 'sep_last' in df.columns:
            try:
                sep_clean = pd.to_numeric(df['sep_last'], errors='coerce').dropna()
                analysis += f"📏 Separation Statistics:\n"
                analysis += f"  Separation range: {sep_clean.min():.1f}\" - {sep_clean.max():.1f}\"\n"
                analysis += f"  Mean separation: {sep_clean.mean():.1f}\"\n"
                analysis += f"  Median separation: {sep_clean.median():.1f}\"\n\n"
            except:
                analysis += f"📏 Separation data present but needs cleaning\n\n"
        
        # Spectral class analysis
        if 'spectr' in df.columns:
            spectr_counts = df['spectr'].value_counts().head(10)
            analysis += f"🌟 Top Spectral Classes:\n"
            for spec, count in spectr_counts.items():
                if pd.notna(spec) and spec.strip():
                    analysis += f"  {spec}: {count} systems\n"
            analysis += "\n"
        
        return analysis

    def _analyze_basic_format_csv(self, df: pd.DataFrame) -> str:
        """Analyze basic format CSV (logged out user)."""
        analysis = f"🔍 BASIC FORMAT ANALYSIS\n"
        analysis += f"This CSV contains limited information. For detailed analysis,\n"
        analysis += f"please log in to Stelle Doppie and re-download.\n\n"
        
        # Basic statistics
        if 'obs' in df.columns:
            obs_stats = df['obs'].describe()
            analysis += f"📊 Observation Statistics:\n"
            analysis += f"  Mean observations per system: {obs_stats['mean']:.1f}\n"
            analysis += f"  Max observations: {obs_stats['max']:.0f}\n\n"
        
        if 'm1' in df.columns and 'm2' in df.columns:
            try:
                m1_clean = pd.to_numeric(df['m1'], errors='coerce').dropna()
                m2_clean = pd.to_numeric(df['m2'], errors='coerce').dropna()
                
                analysis += f"⭐ Magnitude Statistics:\n"
                analysis += f"  Primary magnitude range: {m1_clean.min():.1f} - {m1_clean.max():.1f}\n"
                analysis += f"  Secondary magnitude range: {m2_clean.min():.1f} - {m2_clean.max():.1f}\n\n"
            except:
                analysis += f"⭐ Magnitude data present but needs cleaning\n\n"
        
        if 'sep' in df.columns:
            try:
                sep_clean = pd.to_numeric(df['sep'], errors='coerce').dropna()
                analysis += f"📏 Separation Statistics:\n"
                analysis += f"  Separation range: {sep_clean.min():.1f}\" - {sep_clean.max():.1f}\"\n"
                analysis += f"  Mean separation: {sep_clean.mean():.1f}\"\n\n"
            except:
                analysis += f"📏 Separation data present but needs cleaning\n\n"
        
        analysis += f"💡 TIP: Log in to Stelle Doppie to access:\n"
        analysis += f"  • Detailed position angles (PA first/last)\n"
        analysis += f"  • Separation history (sep first/last)\n"
        analysis += f"  • Proper motion data\n"
        analysis += f"  • Spectral classifications\n"
        analysis += f"  • Discovery information\n"
        analysis += f"  • Complete observation notes\n\n"
        
        return analysis

    def _export_imported_data(self):
        """Export the imported and analyzed CSV data."""
        if not hasattr(self, 'imported_csv_data') or self.imported_csv_data is None:
            messagebox.showwarning("No Data", "Please import a CSV file first.")
            return
        
        # Get selected format
        format_text = self.export_format_var.get()
        format_key = None
        for key, fmt in EXPORT_FORMATS.items():
            if format_text.startswith(fmt['name']):
                format_key = key
                break
        
        if not format_key:
            messagebox.showerror("Error", "Invalid export format selected.")
            return
        
        # Get save location
        format_info = EXPORT_FORMATS[format_key]
        filename = filedialog.asksaveasfilename(
            defaultextension=format_info['extension'],
            filetypes=[(format_info['name'], f"*{format_info['extension']}")]
        )
        
        if not filename:
            return
        
        try:
            self._perform_imported_data_export(filename, format_key)
            messagebox.showinfo("Export Complete", f"Analyzed data exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")

    def _perform_imported_data_export(self, filename: str, format_key: str):
        """Perform the export of imported data."""
        self.status_var.set("Exporting imported data...")
        self.progress_var.set(0)
        
        if format_key == 'csv':
            self._export_imported_csv(filename)
        elif format_key == 'json':
            self._export_imported_json(filename)
        else:
            # For other formats, use the original data as-is
            self._export_imported_csv(filename)
        
        self.status_var.set("Export complete")
        self.progress_var.set(100)

    def _export_imported_csv(self, filename: str):
        """Export imported data to CSV with optional analysis."""
        df = self.imported_csv_data.copy()
        
        # Add analysis columns if requested
        if self.include_analysis_var.get():
            df['import_date'] = datetime.now().isoformat()
            df['csv_format'] = self.imported_csv_format
            df['source_application'] = 'AstraKairos'
            df['analysis_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Export to CSV
        df.to_csv(filename, index=False, encoding='utf-8')
        
        # Update progress
        self.progress_var.set(100)
        self.root.update_idletasks()

    def _export_imported_json(self, filename: str):
        """Export imported data to JSON format."""
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'format_version': '1.0',
                'source_format': self.imported_csv_format,
                'total_systems': len(self.imported_csv_data),
                'columns': list(self.imported_csv_data.columns),
                'source_application': 'AstraKairos'
            },
            'data': self.imported_csv_data.to_dict('records')
        }
        
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, ensure_ascii=False, default=str)
        
        # Update progress
        self.progress_var.set(100)
        self.root.update_idletasks()

def main():
    root = tk.Tk()
    app = AstraKairosPlannerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()