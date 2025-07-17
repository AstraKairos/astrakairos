# astrakairos/planner/gui.py

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import json
from datetime import datetime, timedelta
import webbrowser
from typing import Dict, Any
import pytz

# Importamos el módulo de cálculos actualizado que ahora usa Skyfield
from ..planner import calculations
from ..config import (
    GUI_DEFAULT_WIDTH, GUI_DEFAULT_HEIGHT,
    STELLE_DOPPIE_BASE_URL, STELLE_DOPPIE_SEARCH_METHODS,
    MIN_ALTITUDE_DEG, MAX_ALTITUDE_DEG,
    MIN_RA_WINDOW_HOURS, MAX_RA_WINDOW_HOURS,
    MIN_LIGHT_POLLUTION_MAG, MAX_LIGHT_POLLUTION_MAG,
    DEFAULT_MIN_ALTITUDE_DEG, DEFAULT_RA_WINDOW_HOURS, DEFAULT_LIGHT_POLLUTION_MAG,
    STELLE_DOPPIE_FILTERS, DEFAULT_SEARCH_OPTIONS
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
        
        self.load_locations()
        self.create_widgets()
        self.update_location_list()
    
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
        """Create all GUI widgets and layout."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        title_label = ttk.Label(main_frame, text="AstraKairos Observation Planner", font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 20), sticky='n')
        
        self._create_location_search_section(main_frame)
        self._create_location_info_section(main_frame)
        self._create_date_section(main_frame)
        self._create_params_section(main_frame)
        self._create_astro_info_section(main_frame)
        self._create_search_options_section(main_frame)
        self._create_binary_search_section(main_frame)
        
        main_frame.rowconfigure(6, weight=1)

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

    def _create_search_options_section(self, parent):
        """Create advanced search options section with configurable filters."""
        frame = ttk.LabelFrame(parent, text="5. Advanced Search Options", padding="10")
        frame.grid(row=6, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
        # Initialize search options with defaults
        for option, default in DEFAULT_SEARCH_OPTIONS.items():
            self.search_options[option] = tk.BooleanVar(value=default)
        
        # Create filter variable dictionaries
        for filter_name, config in STELLE_DOPPIE_FILTERS.items():
            self.filter_vars[filter_name] = {
                'min': tk.DoubleVar(value=config['default_min']),
                'max': tk.DoubleVar(value=config['default_max'])
            }
        
        row = 0
        
        # Magnitude filter
        mag_check = ttk.Checkbutton(frame, text="Filter by Magnitude:", 
                                   variable=self.search_options['use_magnitude_filter'])
        mag_check.grid(row=row, column=0, sticky=tk.W, pady=2)
        
        mag_frame = ttk.Frame(frame)
        mag_frame.grid(row=row, column=1, sticky="ew", padx=5)
        ttk.Label(mag_frame, text="Min:").grid(row=0, column=0, padx=2)
        ttk.Entry(mag_frame, textvariable=self.filter_vars['magnitude']['min'], width=8).grid(row=0, column=1, padx=2)
        ttk.Label(mag_frame, text="Max:").grid(row=0, column=2, padx=2)
        ttk.Entry(mag_frame, textvariable=self.filter_vars['magnitude']['max'], width=8).grid(row=0, column=3, padx=2)
        ttk.Label(mag_frame, text="mag").grid(row=0, column=4, padx=2)
        
        row += 1
        
        # Separation filter
        sep_check = ttk.Checkbutton(frame, text="Filter by Separation:", 
                                   variable=self.search_options['use_separation_filter'])
        sep_check.grid(row=row, column=0, sticky=tk.W, pady=2)
        
        sep_frame = ttk.Frame(frame)
        sep_frame.grid(row=row, column=1, sticky="ew", padx=5)
        ttk.Label(sep_frame, text="Min:").grid(row=0, column=0, padx=2)
        ttk.Entry(sep_frame, textvariable=self.filter_vars['separation']['min'], width=8).grid(row=0, column=1, padx=2)
        ttk.Label(sep_frame, text="Max:").grid(row=0, column=2, padx=2)
        ttk.Entry(sep_frame, textvariable=self.filter_vars['separation']['max'], width=8).grid(row=0, column=3, padx=2)
        ttk.Label(sep_frame, text="arcsec").grid(row=0, column=4, padx=2)
        
        row += 1
        
        # Position angle filter
        pa_check = ttk.Checkbutton(frame, text="Filter by Position Angle:", 
                                  variable=self.search_options['use_position_angle_filter'])
        pa_check.grid(row=row, column=0, sticky=tk.W, pady=2)
        
        pa_frame = ttk.Frame(frame)
        pa_frame.grid(row=row, column=1, sticky="ew", padx=5)
        ttk.Label(pa_frame, text="Min:").grid(row=0, column=0, padx=2)
        ttk.Entry(pa_frame, textvariable=self.filter_vars['position_angle']['min'], width=8).grid(row=0, column=1, padx=2)
        ttk.Label(pa_frame, text="Max:").grid(row=0, column=2, padx=2)
        ttk.Entry(pa_frame, textvariable=self.filter_vars['position_angle']['max'], width=8).grid(row=0, column=3, padx=2)
        ttk.Label(pa_frame, text="deg").grid(row=0, column=4, padx=2)
        
        row += 1
        
        # Epoch filter
        epoch_check = ttk.Checkbutton(frame, text="Filter by Epoch:", 
                                     variable=self.search_options['use_epoch_filter'])
        epoch_check.grid(row=row, column=0, sticky=tk.W, pady=2)
        
        epoch_frame = ttk.Frame(frame)
        epoch_frame.grid(row=row, column=1, sticky="ew", padx=5)
        ttk.Label(epoch_frame, text="Min:").grid(row=0, column=0, padx=2)
        ttk.Entry(epoch_frame, textvariable=self.filter_vars['epoch']['min'], width=8).grid(row=0, column=1, padx=2)
        ttk.Label(epoch_frame, text="Max:").grid(row=0, column=2, padx=2)
        ttk.Entry(epoch_frame, textvariable=self.filter_vars['epoch']['max'], width=8).grid(row=0, column=3, padx=2)
        ttk.Label(epoch_frame, text="year").grid(row=0, column=4, padx=2)

    def _create_binary_search_section(self, parent):
        frame = ttk.LabelFrame(parent, text="6. Generate Target Search", padding="10")
        frame.grid(row=7, column=0, sticky="ew", pady=5)
        frame.columnconfigure(0, weight=1)

        ttk.Button(frame, text="Generate Search on Stelle Doppie", 
                   command=self.generate_stelle_doppie_search).grid(row=0, column=0, pady=5)
        
        # Add a text widget to show the generated URL
        self.url_text = ScrolledText(frame, height=3, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.url_text.grid(row=1, column=0, sticky="ew", pady=(5, 0))

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
            calc_time = events.get('astronomical_twilight_end')
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
                min_altitude_deg=min_alt, light_pollution_mag=lp_mag
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
            f"Civil Twilight End:     {fmt_time(events.get('civil_twilight_end'))}\n"
            f"Nautical Twilight End:  {fmt_time(events.get('nautical_twilight_end'))}\n"
            f"Astro. Twilight End:    {fmt_time(events.get('astronomical_twilight_end'))} (Start of Darkness)\n"
            f"Astronomical Midnight:  {fmt_time(events.get('astronomical_midnight_local'))}\n"
            f"Astronomical Midnight:  {fmt_time_utc(events.get('astronomical_midnight_utc'))}\n"
            f"Temporal Midnight:      {fmt_time(events.get('temporal_midnight_local'))} (Midpoint of Night)\n"
            f"Temporal Midnight:      {fmt_time_utc(events.get('temporal_midnight_utc'))}\n"
            f"Astro. Twilight Start:  {fmt_time(events.get('astronomical_twilight_start'))}\n"
            f"Nautical Twilight Start:{fmt_time(events.get('nautical_twilight_start'))}\n"
            f"Civil Twilight Start:   {fmt_time(events.get('civil_twilight_start'))}\n"
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
            
            # Format coordinates correctly for Stelle Doppie URL encoding
            ra_min_str = f"{int(ra_min_h):02d}%2C{int((ra_min_h % 1) * 60):02d}"
            ra_max_str = f"{int(ra_max_h):02d}%2C{int((ra_max_h % 1) * 60):02d}"
            dec_min_str = f"{int(dec_min_d):+03d}%2C{int(abs(dec_min_d) % 1 * 60):02d}" if dec_min_d is not None else ""
            dec_max_str = f"{int(dec_max_d):+03d}%2C{int(abs(dec_max_d) % 1 * 60):02d}" if dec_max_d is not None else ""
            
            # Build base URL parameters
            params = {
                'menu': 21, 
                'azione': 'cerca_nel_database', 
                'gocerca': 'Search+the+database',
                'metodo-cat_wds-ra': STELLE_DOPPIE_SEARCH_METHODS['coordinate_range'], 
                'dato-cat_wds-ra': f'{ra_min_str}%2C{ra_max_str}',
                'metodo-cat_wds-de': STELLE_DOPPIE_SEARCH_METHODS['coordinate_range'], 
                'dato-cat_wds-de': f'{dec_min_str}%2C{dec_max_str}',
            }
            
            # Add advanced filters if enabled
            active_filters = []
            
            if self.search_options['use_magnitude_filter'].get():
                mag_min = self.filter_vars['magnitude']['min'].get()
                mag_max = self.filter_vars['magnitude']['max'].get()
                if self._validate_filter_range('magnitude', mag_min, mag_max):
                    params[f'metodo-{STELLE_DOPPIE_FILTERS["magnitude"]["param_name"]}'] = STELLE_DOPPIE_SEARCH_METHODS['magnitude_range']
                    params[f'dato-{STELLE_DOPPIE_FILTERS["magnitude"]["param_name"]}'] = f'{mag_min:.1f}%2C{mag_max:.1f}'
                    active_filters.append(f"Magnitude: {mag_min:.1f} - {mag_max:.1f} mag")
            
            if self.search_options['use_separation_filter'].get():
                sep_min = self.filter_vars['separation']['min'].get()
                sep_max = self.filter_vars['separation']['max'].get()
                if self._validate_filter_range('separation', sep_min, sep_max):
                    params[f'metodo-{STELLE_DOPPIE_FILTERS["separation"]["param_name"]}'] = STELLE_DOPPIE_SEARCH_METHODS['separation_range']
                    params[f'dato-{STELLE_DOPPIE_FILTERS["separation"]["param_name"]}'] = f'{sep_min:.1f}%2C{sep_max:.1f}'
                    active_filters.append(f"Separation: {sep_min:.1f} - {sep_max:.1f} arcsec")
            
            if self.search_options['use_position_angle_filter'].get():
                pa_min = self.filter_vars['position_angle']['min'].get()
                pa_max = self.filter_vars['position_angle']['max'].get()
                if self._validate_filter_range('position_angle', pa_min, pa_max):
                    params[f'metodo-{STELLE_DOPPIE_FILTERS["position_angle"]["param_name"]}'] = STELLE_DOPPIE_SEARCH_METHODS['coordinate_range']
                    params[f'dato-{STELLE_DOPPIE_FILTERS["position_angle"]["param_name"]}'] = f'{pa_min:.0f}%2C{pa_max:.0f}'
                    active_filters.append(f"Position Angle: {pa_min:.0f}° - {pa_max:.0f}°")
            
            if self.search_options['use_epoch_filter'].get():
                epoch_min = self.filter_vars['epoch']['min'].get()
                epoch_max = self.filter_vars['epoch']['max'].get()
                if self._validate_filter_range('epoch', epoch_min, epoch_max):
                    params[f'metodo-{STELLE_DOPPIE_FILTERS["epoch"]["param_name"]}'] = STELLE_DOPPIE_SEARCH_METHODS['coordinate_range']
                    params[f'dato-{STELLE_DOPPIE_FILTERS["epoch"]["param_name"]}'] = f'{epoch_min:.0f}%2C{epoch_max:.0f}'
                    active_filters.append(f"Epoch: {epoch_min:.0f} - {epoch_max:.0f}")
            
            # Build final URL
            full_url = STELLE_DOPPIE_BASE_URL + "?" + "&".join([f"{k}={v}" for k, v in params.items()])

            self.url_text.config(state=tk.NORMAL)
            self.url_text.delete(1.0, tk.END)
            self.url_text.insert(tk.END, full_url)
            self.url_text.config(state=tk.DISABLED)
            
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
            
    def _validate_filter_range(self, filter_name: str, min_val: float, max_val: float) -> bool:
        """Validate filter range values against configuration limits."""
        if filter_name not in STELLE_DOPPIE_FILTERS:
            return False
        
        config = STELLE_DOPPIE_FILTERS[filter_name]
        
        # Check if values are within valid range
        if not (config['min_value'] <= min_val <= config['max_value']):
            messagebox.showerror("Invalid Filter Range", 
                               f"{filter_name.title()} minimum value {min_val} outside valid range "
                               f"[{config['min_value']}, {config['max_value']}]")
            return False
        
        if not (config['min_value'] <= max_val <= config['max_value']):
            messagebox.showerror("Invalid Filter Range", 
                               f"{filter_name.title()} maximum value {max_val} outside valid range "
                               f"[{config['min_value']}, {config['max_value']}]")
            return False
        
        # Check if min <= max
        if min_val >= max_val:
            messagebox.showerror("Invalid Filter Range", 
                               f"{filter_name.title()} minimum value {min_val} must be less than maximum value {max_val}")
            return False
        
        return True
            
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

def main():
    root = tk.Tk()
    app = AstraKairosPlannerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()