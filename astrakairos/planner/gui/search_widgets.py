# astrakairos/planner/gui/search_widgets.py

"""
Search functionality for the GUI.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import webbrowser
import urllib.parse
import pandas as pd
import os

from ...config import (
    STELLE_DOPPIE_BASE_URL, STELLE_DOPPIE_FILTERS, DEFAULT_SEARCH_OPTIONS,
    STELLE_DOPPIE_METHODS, CATALOG_SOURCES
)


class SearchManager:
    """Manages search functionality for binary stars."""
    
    def __init__(self, parent_app):
        self.app = parent_app
        self.search_results = []
        self.imported_csv_data = None
        self.imported_csv_format = None
        
        # Initialize search options
        self.search_options = {}
        self.filter_vars = {}
        
        # Initialize search options with defaults
        for option, default in DEFAULT_SEARCH_OPTIONS.items():
            self.search_options[option] = tk.BooleanVar(value=default)
        
        # Create filter variable dictionaries for configurable filters
        for filter_name, config in STELLE_DOPPIE_FILTERS.items():
            self.filter_vars[filter_name] = {
                'method': tk.StringVar(value=config.get('default_method', '1')),
                'value': tk.StringVar(value=config.get('default_value', ''))
            }
    
    def create_search_section(self, parent):
        """Create the complete search options section."""
        frame = ttk.LabelFrame(parent, text="5. Search Options", padding="10")
        frame.grid(row=6, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
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

        # URL Generation Section
        self._create_url_generation_section(parent)
        
        # CSV Import Section
        self._create_csv_import_section(parent)
        
        return frame
    
    def _create_filter_row(self, parent, row, filter_name, config):
        """Create a filter row with checkbox, dropdown for method, and textbox for value."""
        
        # Create checkbox for filter
        use_filter_var = tk.BooleanVar(value=config.get('default_enabled', False))
        self.search_options[f'use_{filter_name}_filter'] = use_filter_var
        
        checkbox = ttk.Checkbutton(parent, text=f"Filter by {config['label']}:", 
                                 variable=use_filter_var)
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
            unit_label.grid(row=0, column=2, padx=2, sticky=tk.W)
        
        return row + 1

    def _create_url_generation_section(self, parent):
        """Create URL generation section."""
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
        self.app.url_text = ScrolledText(frame, height=3, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.app.url_text.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        
        # Configure mouse wheel scrolling
        from .utilities import GUIUtilities
        GUIUtilities.configure_text_widget_scroll(self.app.url_text)

        # Add results display area
        self.app.results_text = ScrolledText(frame, height=8, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.app.results_text.grid(row=3, column=0, sticky="ew", pady=(5, 0))
        
        # Configure mouse wheel scrolling
        GUIUtilities.configure_text_widget_scroll(self.app.results_text)
        
        # Initialize with helpful instructions
        self._initialize_results_text()

    def _initialize_results_text(self):
        """Initialize the results text area with helpful instructions."""
        self.app.results_text.config(state=tk.NORMAL)
        self.app.results_text.delete(1.0, tk.END)
        
        instructions = """🔍 STELLE DOPPIE SEARCH WORKFLOW

📝 STEPS TO GET DATA:
1. Click "🔍 Generate Search URL" above
2. Copy the generated URL and open it in your browser
3. Log in to Stelle Doppie (required for full data access)
4. Change 'index2.php' to 'excel.php' in the URL for CSV download
5. Download the CSV file from your browser
6. Use "📁 Import CSV File" below to analyze the data

� SIMPLE WORKFLOW:
AstraKairos → Browser → Stelle Doppie → CSV → Analysis

The URL will appear above after you click "Generate Search URL".
"""
        
        self.app.results_text.insert(tk.END, instructions)
        self.app.results_text.config(state=tk.DISABLED)

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
        self.app.current_file_var = tk.StringVar(value="No file loaded")
        file_info_label = ttk.Label(frame, textvariable=self.app.current_file_var, 
                                   font=('Arial', 9), foreground='blue')
        file_info_label.grid(row=2, column=0, sticky="w", pady=(5, 0))

        # CSV Analysis results display area
        self.app.csv_analysis_text = ScrolledText(frame, height=12, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.app.csv_analysis_text.grid(row=3, column=0, sticky="ew", pady=(5, 0))
        
        # Configure mouse wheel scrolling
        from .utilities import GUIUtilities
        GUIUtilities.configure_text_widget_scroll(self.app.csv_analysis_text)

        # Velocity analysis section
        velocity_frame = ttk.Frame(frame)
        velocity_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        velocity_frame.columnconfigure(0, weight=1)
        
        # Velocity analysis controls
        velocity_controls = ttk.Frame(velocity_frame)
        velocity_controls.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        ttk.Label(velocity_controls, text="Top systems by velocity:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        
        self.app.velocity_count_var = tk.IntVar(value=10)
        velocity_spinbox = ttk.Spinbox(velocity_controls, from_=1, to=100, width=5, textvariable=self.app.velocity_count_var)
        velocity_spinbox.grid(row=0, column=1, sticky="w", padx=(0, 5))
        
        ttk.Button(velocity_controls, text="🚀 Calculate Velocities", 
                  command=self.calculate_system_velocities).grid(row=0, column=2, sticky="w", padx=(5, 0))
        
        # Velocity results display area
        self.app.velocity_results_text = ScrolledText(velocity_frame, height=8, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.app.velocity_results_text.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        
        # Configure mouse wheel scrolling
        from .utilities import GUIUtilities
        GUIUtilities.configure_text_widget_scroll(self.app.velocity_results_text)

    def generate_stelle_doppie_search(self):
        """Generate Stelle Doppie search URL based on optimal conditions."""
        if not hasattr(self.app, 'calculation_manager') or not hasattr(self.app.calculation_manager, 'optimal_ra_range') or not self.app.calculation_manager.optimal_ra_range:
            messagebox.showwarning("No Conditions", 
                                 "Please calculate optimal conditions first.")
            return
        
        try:
            # Get RA/Dec ranges from calculations
            ra_min_h, ra_max_h = self.app.calculation_manager.optimal_ra_range
            dec_min_d, dec_max_d = self.app.calculation_manager.optimal_dec_range
            
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
            
            # Add filters if enabled
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

            # Display URL
            self.app.url_text.config(state=tk.NORMAL)
            self.app.url_text.delete(1.0, tk.END)
            self.app.url_text.insert(tk.END, full_url)
            self.app.url_text.config(state=tk.DISABLED)
            
            # Show simple confirmation in results area
            self.app.results_text.config(state=tk.NORMAL)
            self.app.results_text.delete(1.0, tk.END)
            
            confirmation = f"""✅ SEARCH URL GENERATED

🎯 Search Parameters:
RA: {self._format_ra_hours(ra_min_h)} to {self._format_ra_hours(ra_max_h)}
Dec: {self._format_dec_degrees(dec_min_d)} to {self._format_dec_degrees(dec_max_d)}

📋 Active Filters:
{chr(10).join(active_filters) if active_filters else "No additional filters applied"}

🔗 Next Steps:
1. Copy the URL above
2. Open it in your browser
3. Log in to Stelle Doppie
4. Change 'index2.php' to 'excel.php' for CSV download
5. Download CSV and import it below for analysis
"""
            
            self.app.results_text.insert(tk.END, confirmation)
            self.app.results_text.config(state=tk.DISABLED)
            
            # Create detailed confirmation message
            filter_text = "\n".join(active_filters) if active_filters else "No additional filters applied"
            
            if messagebox.askyesno("Open Search", 
                                 f"Search URL generated for:\n"
                                 f"RA: {self._format_ra_hours(ra_min_h)} to {self._format_ra_hours(ra_max_h)}\n"
                                 f"Dec: {self._format_dec_degrees(dec_min_d)} to {self._format_dec_degrees(dec_max_d)}\n\n"
                                 f"Active filters:\n{filter_text}\n\n"
                                 f"Open in browser?"):
                webbrowser.open(full_url)
            
        except Exception as e:
            messagebox.showerror("URL Generation Error", f"Failed to generate search URL: {str(e)}")

    def _format_ra_hours(self, hours: float) -> str:
        """Format RA hours to HH:MM format."""
        h = int(hours)
        m = int((hours - h) * 60)
        return f"{h:02d}h{m:02d}m"

    def _format_dec_degrees(self, degrees: float) -> str:
        """Format declination degrees to +/-DD:MM format."""
        sign = "+" if degrees >= 0 else "-"
        abs_deg = abs(degrees)
        d = int(abs_deg)
        m = int((abs_deg - d) * 60)
        return f"{sign}{d:02d}°{m:02d}'"

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
                        messagebox.showerror("Invalid Value", "Both values required for between method")
                        return False, value, method
                    processed_value = f"{val1},{val2}"
                elif data_type == 'integer':
                    val1, val2 = int(parts[0].strip()), int(parts[1].strip())
                    if val1 >= val2:
                        messagebox.showerror("Invalid Range", "First value must be less than second value")
                        return False, value, method
                    processed_value = f"{val1},{val2}"
                else:  # numeric (float)
                    val1, val2 = float(parts[0].strip()), float(parts[1].strip())
                    if val1 >= val2:
                        messagebox.showerror("Invalid Range", "First value must be less than second value")
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
                        messagebox.showerror("Invalid Value", "Both values required for not between method")
                        return False, value, method
                    processed_value = f"{val1},{val2}"
                elif data_type == 'integer':
                    val1, val2 = int(parts[0].strip()), int(parts[1].strip())
                    if val1 >= val2:
                        messagebox.showerror("Invalid Range", "First value must be less than second value")
                        return False, value, method
                    processed_value = f"{val1},{val2}"
                else:  # numeric (float)
                    val1, val2 = float(parts[0].strip()), float(parts[1].strip())
                    if val1 >= val2:
                        messagebox.showerror("Invalid Range", "First value must be less than second value")
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

    def _format_ra_hours(self, hours: float) -> str:
        """Format RA hours as HHhMMm."""
        h = int(hours)
        m = int((hours - h) * 60)
        return f"{h:02d}h{m:02d}m"

    def _format_dec_degrees(self, degrees: float) -> str:
        """Format Dec degrees as ±DDdMM'."""
        sign = "+" if degrees >= 0 else "-"
        abs_deg = abs(degrees)
        d = int(abs_deg)
        m = int((abs_deg - d) * 60)
        return f"{sign}{d:02d}°{m:02d}'"

    def import_csv_file(self):
        """Import and analyze a CSV file downloaded from Stelle Doppie."""
        file_path = filedialog.askopenfilename(
            title="Select Stelle Doppie CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Use robust CSV reading
            df = self._robust_csv_read(file_path)
            
            if df.empty:
                messagebox.showwarning("Empty File", "The selected CSV file is empty.")
                return
            
            self.imported_csv_data = df
            self.imported_csv_format = self._detect_csv_format(df)
            self.app.current_file_var.set(f"Loaded: {os.path.basename(file_path)} ({len(df)} rows)")
            
            # Detect format and analyze
            self._analyze_imported_csv(df, file_path, self.imported_csv_format)
            
        except pd.errors.EmptyDataError:
            messagebox.showerror("Import Error", "The selected file appears to be empty or corrupted.")
        except pd.errors.ParserError as e:
            messagebox.showerror("Import Error", f"Error parsing CSV file: {str(e)}\n\nTry saving the file with UTF-8 encoding.")
        except UnicodeDecodeError:
            messagebox.showerror("Import Error", "Unable to read file encoding. Please save the CSV file with UTF-8 encoding.")
        except FileNotFoundError:
            messagebox.showerror("Import Error", "The selected file could not be found.")
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import CSV file: {str(e)}")
        finally:
            self.app.update_status("Ready")

    def _analyze_imported_csv(self, df: pd.DataFrame, file_path: str):
        """Analyze the imported CSV data."""
        self.app.csv_analysis_text.config(state=tk.NORMAL)
        self.app.csv_analysis_text.delete(1.0, tk.END)
        
        # Basic analysis
        num_rows = len(df)
        num_cols = len(df.columns)
        
        analysis_text = f"📊 CSV ANALYSIS RESULTS\n"
        analysis_text += f"File: {os.path.basename(file_path)}\n"
        analysis_text += f"Rows: {num_rows}, Columns: {num_cols}\n\n"
        
        # Show column names
        analysis_text += "📋 AVAILABLE COLUMNS:\n"
        for i, col in enumerate(df.columns, 1):
            analysis_text += f"{i:2d}. {col}\n"
        
        # Show first few rows as sample
        analysis_text += f"\n📝 SAMPLE DATA (first 5 rows):\n"
        analysis_text += df.head().to_string(index=False)
        
        # Basic statistics if numeric columns exist
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            analysis_text += f"\n\n📈 NUMERIC STATISTICS:\n"
            analysis_text += df[numeric_cols].describe().to_string()
        
        self.app.csv_analysis_text.insert(tk.END, analysis_text)
        self.app.csv_analysis_text.config(state=tk.DISABLED)

    def _robust_csv_read(self, file_path: str) -> pd.DataFrame:
        """Robust CSV reading with multiple fallback methods."""
        import csv
        from io import StringIO
        
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

    def _detect_csv_format(self, df: pd.DataFrame) -> str:
        """Detect the format of the imported CSV file."""
        # Check column names to determine format
        columns = [col.lower() for col in df.columns]
        
        # Look for Stelle Doppie specific columns
        stelle_doppie_indicators = ['wds', 'discoverer', 'components', 'coord', 'magnitude', 'first', 'last', 'pa_first', 'sep_first']
        
        if any(indicator in ' '.join(columns) for indicator in stelle_doppie_indicators):
            return 'full'  # Changed to match original logic
        elif len(df.columns) >= 5:
            return 'basic'  # Changed to match original logic
        else:
            return 'basic'  # Changed to match original logic

    def _analyze_imported_csv(self, df: pd.DataFrame, file_path: str, csv_format: str):
        """Analyze the imported CSV data and display results."""
        self.app.csv_analysis_text.config(state=tk.NORMAL)
        self.app.csv_analysis_text.delete(1.0, tk.END)
        
        # Basic analysis
        num_rows = len(df)
        num_cols = len(df.columns)
        
        analysis_text = f"📊 CSV ANALYSIS RESULTS\n"
        analysis_text += f"File: {os.path.basename(file_path)}\n"
        analysis_text += f"Format: {csv_format}\n"
        analysis_text += f"Rows: {num_rows}, Columns: {num_cols}\n\n"
        
        # Show column names
        analysis_text += "📋 AVAILABLE COLUMNS:\n"
        for i, col in enumerate(df.columns, 1):
            analysis_text += f"{i:2d}. {col}\n"
        
        # Show first few rows as sample
        analysis_text += f"\n📝 SAMPLE DATA (first 5 rows):\n"
        try:
            sample_data = df.head().to_string(index=False, max_cols=10, max_colwidth=20)
            analysis_text += sample_data
        except Exception:
            analysis_text += "Error displaying sample data - file may contain special characters"
        
        # Basic statistics if numeric columns exist
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            analysis_text += f"\n\n📈 NUMERIC STATISTICS:\n"
            try:
                stats_text = df[numeric_cols].describe().to_string()
                analysis_text += stats_text
            except Exception:
                analysis_text += "Error calculating statistics"
        
        # Format-specific analysis
        if csv_format == 'full':
            analysis_text += self._analyze_full_format_csv(df)
        elif csv_format == 'basic':
            analysis_text += self._analyze_basic_format_csv(df)
        
        self.app.csv_analysis_text.insert(tk.END, analysis_text)
        self.app.csv_analysis_text.config(state=tk.DISABLED)

    def _analyze_full_format_csv(self, df: pd.DataFrame) -> str:
        """Analyze full format CSV from Stelle Doppie."""
        analysis = "\n\n🔍 STELLE DOPPIE ANALYSIS:\n"
        
        # Try to identify coordinate columns
        coord_cols = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['ra', 'dec', 'coord', 'position'])]
        
        if coord_cols:
            analysis += f"Coordinate columns: {', '.join(coord_cols)}\n"
        
        # Try to identify magnitude columns
        mag_cols = [col for col in df.columns if any(keyword in col.lower() 
                   for keyword in ['mag', 'magnitude', 'brightness'])]
        
        if mag_cols:
            analysis += f"Magnitude columns: {', '.join(mag_cols)}\n"
        
        # Count unique systems
        if len(df) > 0:
            analysis += f"Total systems: {len(df)}\n"
        
        return analysis

    def _analyze_basic_format_csv(self, df: pd.DataFrame) -> str:
        """Analyze basic format CSV."""
        analysis = "\n\n📝 BASIC FORMAT ANALYSIS:\n"
        analysis += f"This appears to be a generic multi-column CSV file.\n"
        analysis += f"You may need to manually identify which columns contain:\n"
        analysis += f"- Object names/designations\n"
        analysis += f"- Coordinates (RA/Dec)\n"
        analysis += f"- Magnitudes\n"
        analysis += f"- Separations and position angles\n"
        
        return analysis

    def calculate_system_velocities(self):
        """Calculate velocities for all loaded systems and show top N fastest."""
        if not hasattr(self, 'imported_csv_data') or self.imported_csv_data is None:
            messagebox.showwarning("No Data", "Please import a CSV file first.")
            return
        
        try:
            from ...physics.dynamics import estimate_velocity_from_endpoints
            
            self.app.update_status("Calculating velocities...")
            self.app.update_progress(0)
            
            df = self.imported_csv_data
            velocity_results = []
            
            # Determine which columns to use based on format
            csv_format = self.imported_csv_format or self._detect_csv_format(df)
            
            if csv_format == "full":
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
                    
            elif csv_format == "basic":
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
                    velocity_result = estimate_velocity_from_endpoints(wds_summary)
                    
                    if velocity_result:
                        velocity_results.append({
                            'name': row[name_col],
                            'velocity': velocity_result['v_total_estimate'],
                            'vx': velocity_result['vx_arcsec_per_year'],
                            'vy': velocity_result['vy_arcsec_per_year'],
                            'pa_v': velocity_result['pa_v_estimate'],
                            'time_baseline': velocity_result['time_baseline_years'],
                            'first': row[actual_columns['first']],
                            'last': row[actual_columns['last']]
                        })
                        
                except Exception as e:
                    # Skip systems with calculation errors
                    continue
                
                # Update progress
                progress = ((i + 1) / total_systems) * 100
                self.app.update_progress(progress)
            
            # Sort by velocity (descending) and get top N
            velocity_results.sort(key=lambda x: x['velocity'], reverse=True)
            n_systems = min(self.app.velocity_count_var.get(), len(velocity_results))
            top_systems = velocity_results[:n_systems]
            
            # Display results
            self._display_velocity_results(top_systems, len(velocity_results), total_systems)
            
            # Update status
            self.app.update_status(f"Velocity calculation complete")
            self.app.update_progress(100)
            
        except ImportError:
            messagebox.showerror("Import Error", "Could not import dynamics module. Please check your installation.")
        except Exception as e:
            messagebox.showerror("Calculation Error", f"Error calculating velocities: {str(e)}")
        finally:
            self.app.update_status("Ready")
            self.app.update_progress(0)

    def _display_velocity_results(self, top_systems, total_calculated, total_systems):
        """Display velocity calculation results."""
        self.app.velocity_results_text.config(state=tk.NORMAL)
        self.app.velocity_results_text.delete(1.0, tk.END)
        
        if not top_systems:
            self.app.velocity_results_text.insert(tk.END, "No velocity calculations possible with current data.\n")
            self.app.velocity_results_text.insert(tk.END, "This usually means:\n")
            self.app.velocity_results_text.insert(tk.END, "• CSV format doesn't have required temporal data\n")
            self.app.velocity_results_text.insert(tk.END, "• Missing first/last observation dates\n")
            self.app.velocity_results_text.insert(tk.END, "• Missing position angle or separation data\n")
            self.app.velocity_results_text.config(state=tk.DISABLED)
            return
        
        # Header
        summary = f"🚀 VELOCITY ANALYSIS RESULTS\n"
        summary += f"=" * 60 + "\n"
        summary += f"Systems processed: {total_systems}\n"
        summary += f"Velocity calculations: {total_calculated}\n"
        summary += f"Top {len(top_systems)} fastest systems:\n\n"
        
        # Table header
        summary += f"{'Rank':<4} {'System':<12} {'Velocity':<12} {'PA':<8} {'Baseline':<10} {'First':<8} {'Last':<8}\n"
        summary += f"{'':4} {'':12} {'(arcsec/yr)':<12} {'(deg)':<8} {'(years)':<10} {'':8} {'':8}\n"
        summary += "-" * 80 + "\n"

        # System details
        for i, system in enumerate(top_systems, 1):
            summary += f"{i:<4} {system['name']:<12} {system['velocity']:<12.4f} {system['pa_v']:<8.1f} {system['time_baseline']:<10.1f} {system['first']:<8} {system['last']:<8}\n"
        
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
        
        self.app.velocity_results_text.insert(tk.END, summary)
        self.app.velocity_results_text.config(state=tk.DISABLED)
        
        # Update results counter
        self.app.update_results_count(total_calculated)
