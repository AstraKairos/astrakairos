import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import json
import datetime
from datetime import date
import ephem
import webbrowser
from typing import Optional

from ..planner import calculations
from ..utils import io

class AstraKairosPlannerApp:
    """Main GUI application for the AstroKairos observation planner."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("AstroKairos - Binary Star Observation Planner")
        self.root.geometry("800x750")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.locations = []
        self.filtered_locations = []
        self.selected_location = None
        self.observer = None
        self.optimal_ra_range = None
        self.optimal_dec_range = None
        self.moon_position = None
        
        # Load data
        self.load_locations()
        
        # Create interface
        self.create_widgets()
        
        # Update initial list
        self.update_location_list()
    
    def load_locations(self):
        """Load locations from JSON file."""
        try:
            with open('locations.json', 'r', encoding='utf-8') as f:
                self.locations = json.load(f)
            print(f"Loaded {len(self.locations)} locations")
        except FileNotFoundError:
            messagebox.showerror("Error", "File 'locations.json' not found")
            self.locations = []
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Error reading JSON file")
            self.locations = []
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="AstroKairos Observation Planner", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Location search section
        self._create_location_search_section(main_frame)
        
        # Selected location info section
        self._create_location_info_section(main_frame)
        
        # Date selection section
        self._create_date_section(main_frame)
        
        # Astronomical information section
        self._create_astro_info_section(main_frame)
        
        # Binary star search section
        self._create_binary_search_section(main_frame)
        
        # Configure row weights
        main_frame.rowconfigure(4, weight=1)  # Astro info expands
    
    def _create_location_search_section(self, parent):
        """Create location search widgets."""
        search_frame = ttk.LabelFrame(parent, text="Location Search", padding="10")
        search_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        search_frame.columnconfigure(1, weight=1)
        
        ttk.Label(search_frame, text="Search:").grid(row=0, column=0, sticky=tk.W)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=50)
        self.search_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        self.search_var.trace('w', self.on_search_change)
        
        # Location listbox
        self.location_listbox = tk.Listbox(search_frame, height=8, width=70)
        self.location_listbox.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        self.location_listbox.bind('<<ListboxSelect>>', self.on_location_select)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(search_frame, orient=tk.VERTICAL, command=self.location_listbox.yview)
        scrollbar.grid(row=1, column=2, sticky=(tk.N, tk.S))
        self.location_listbox.config(yscrollcommand=scrollbar.set)
    
    def _create_location_info_section(self, parent):
        """Create selected location info widgets."""
        info_frame = ttk.LabelFrame(parent, text="Selected Location", padding="10")
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        info_frame.columnconfigure(1, weight=1)
        
        labels = ["Name:", "Type:", "Coordinates:", "Altitude:"]
        self.info_labels = {}
        
        for i, label_text in enumerate(labels):
            ttk.Label(info_frame, text=label_text).grid(row=i, column=0, sticky=tk.W)
            label = ttk.Label(info_frame, text="", font=('Arial', 10, 'bold') if i == 0 else None)
            label.grid(row=i, column=1, sticky=tk.W, padx=(5, 0))
            self.info_labels[label_text.rstrip(':')] = label
    
    def _create_date_section(self, parent):
        """Create date selection widgets."""
        date_frame = ttk.LabelFrame(parent, text="Observation Date", padding="10")
        date_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(date_frame, text="Date:").grid(row=0, column=0, sticky=tk.W)
        self.date_var = tk.StringVar(value=date.today().strftime("%Y-%m-%d"))
        self.date_entry = ttk.Entry(date_frame, textvariable=self.date_var, width=15)
        self.date_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Button(date_frame, text="Tonight", command=self.set_tonight).grid(row=0, column=2, padx=(10, 0))
        ttk.Button(date_frame, text="Calculate Conditions", 
                  command=self.calculate_conditions).grid(row=0, column=3, padx=(10, 0))
    
    def _create_astro_info_section(self, parent):
        """Create astronomical information display."""
        astro_frame = ttk.LabelFrame(parent, text="Astronomical Information", padding="10")
        astro_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        astro_frame.columnconfigure(0, weight=1)
        astro_frame.rowconfigure(0, weight=1)
        
        self.astro_text = ScrolledText(astro_frame, height=12, width=70, wrap=tk.WORD)
        self.astro_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def _create_binary_search_section(self, parent):
        """Create binary star search widgets."""
        binary_frame = ttk.LabelFrame(parent, text="Binary Star Search", padding="10")
        binary_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        binary_frame.columnconfigure(1, weight=1)
        
        ttk.Button(binary_frame, text="Calculate Optimal Region", 
                  command=self.calculate_optimal_region).grid(row=0, column=0, sticky=tk.W)
        
        # Region info labels
        ttk.Label(binary_frame, text="Optimal RA Region:").grid(row=1, column=0, sticky=tk.W)
        self.ra_region_label = ttk.Label(binary_frame, text="", font=('Arial', 10, 'bold'))
        self.ra_region_label.grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(binary_frame, text="Optimal Dec Region:").grid(row=2, column=0, sticky=tk.W)
        self.dec_region_label = ttk.Label(binary_frame, text="", font=('Arial', 10, 'bold'))
        self.dec_region_label.grid(row=2, column=1, sticky=tk.W, padx=(5, 0))
        
        # Search generation
        ttk.Button(binary_frame, text="Generate Search on Stelle Doppie", 
                  command=self.generate_stelle_doppie_search).grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
        # URL display
        self.url_text = ScrolledText(binary_frame, height=3, width=70, wrap=tk.WORD)
        self.url_text.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def get_location_type_name(self, type_code):
        """Convert location type code to descriptive name."""
        type_map = {
            'C': 'Capital',
            'B': 'Capital',
            'R': 'Regional Capital',
            'O': 'Observatory',
            'H': 'Archaeoastronomical Site'
        }
        return type_map.get(type_code, 'Unknown')
    
    def on_search_change(self, *args):
        """Handle search text changes."""
        self.update_location_list()
    
    def update_location_list(self):
        """Update the location list based on search filter."""
        search_term = self.search_var.get().lower()
        
        # Filter locations
        self.filtered_locations = []
        for location in self.locations:
            if (search_term in location['name'].lower() or
                search_term in location.get('state', '').lower() or
                search_term in location.get('region', '').lower()):
                self.filtered_locations.append(location)
        
        # Update listbox
        self.location_listbox.delete(0, tk.END)
        for location in self.filtered_locations:
            display_text = f"{location['name']} ({self.get_location_type_name(location['type'])}) - {location.get('state', '')}"
            self.location_listbox.insert(tk.END, display_text)
    
    def on_location_select(self, event):
        """Handle location selection."""
        selection = self.location_listbox.curselection()
        if selection:
            index = selection[0]
            if index < len(self.filtered_locations):
                self.selected_location = self.filtered_locations[index]
                self.update_location_info()
    
    def update_location_info(self):
        """Update displayed location information."""
        if not self.selected_location:
            return
        
        loc = self.selected_location
        
        # Update labels
        self.info_labels['Name'].config(text=loc['name'])
        self.info_labels['Type'].config(text=self.get_location_type_name(loc['type']))
        self.info_labels['Coordinates'].config(text=f"{loc['latitude']}, {loc['longitude']}")
        self.info_labels['Altitude'].config(text=f"{loc['altitude_m']} m" if loc['altitude_m'] else "N/A")
        
        # Create observer
        self.create_observer()
    
    def create_observer(self):
        """Create PyEphem observer for selected location."""
        if not self.selected_location:
            return
        
        self.observer = ephem.Observer()
        
        # Parse coordinates
        lat_str = self.selected_location['latitude']
        lon_str = self.selected_location['longitude']
        
        # Parse latitude
        if lat_str.endswith('N'):
            lat_deg = float(lat_str[:-1])
        else:  # S
            lat_deg = -float(lat_str[:-1])
        
        # Parse longitude
        if lon_str.endswith('E'):
            lon_deg = float(lon_str[:-1])
        else:  # W
            lon_deg = -float(lon_str[:-1])
        
        self.observer.lat = str(lat_deg)
        self.observer.lon = str(lon_deg)
        
        # Altitude
        if self.selected_location['altitude_m']:
            self.observer.elevation = self.selected_location['altitude_m']
    
    def set_tonight(self):
        """Set date to today."""
        self.date_var.set(date.today().strftime("%Y-%m-%d"))
    
    def calculate_conditions(self):
        """Calculate and display astronomical conditions."""
        if not self.selected_location or not self.observer:
            messagebox.showwarning("Warning", "First select a location")
            return
        
        try:
            # Parse date
            obs_date = datetime.datetime.strptime(self.date_var.get(), "%Y-%m-%d")
            
            # Get timezone
            timezone = self.selected_location.get('timezone', 'UTC')
            
            # Calculate sun/moon info
            info = calculations.calculate_sun_moon_info(self.observer, obs_date, timezone)
            
            # Format and display
            self.display_astronomical_info(info, timezone)
            
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD")
    
    def display_astronomical_info(self, info, timezone):
        """Format and display astronomical information."""
        # Format times
        def format_time(dt):
            if dt is None:
                return "N/A"
            return dt.strftime('%H:%M:%S')
        
        # Determine moon phase description
        phase = info['moon_phase']
        if phase < 1:
            phase_desc = "New Moon"
        elif phase < 25:
            phase_desc = "Waxing Crescent"
        elif phase < 50:
            phase_desc = "First Quarter"
        elif phase < 75:
            phase_desc = "Waxing Gibbous"
        elif phase < 99:
            phase_desc = "Full Moon"
        else:
            phase_desc = "Waning Gibbous"
        
        # Build info text
        info_text = f"""ASTRONOMICAL CONDITIONS
Date: {self.date_var.get()}
Location: {self.selected_location['name']}
Timezone: {timezone}

SOLAR INFORMATION:
Sunset: {format_time(info['sunset'])}
Sunrise: {format_time(info['sunrise'])}
Midnight: {format_time(info['midnight'])}

LUNAR INFORMATION:
Moonrise: {format_time(info['moonrise'])}
Moonset: {format_time(info['moonset'])}
Moon Phase: {phase_desc} ({phase:.1f}%)
Position at midnight:
  - Altitude: {info['moon_alt']:.1f}°
  - Azimuth: {info['moon_az']:.1f}°
"""
        
        # Display
        self.astro_text.delete(1.0, tk.END)
        self.astro_text.insert(tk.END, info_text)
    
    def calculate_optimal_region(self):
        """Calculate optimal observation region."""
        if not self.selected_location or not self.observer:
            messagebox.showwarning("Warning", "First select a location and calculate conditions")
            return
        
        try:
            # Parse date
            obs_date = datetime.datetime.strptime(self.date_var.get(), "%Y-%m-%d")
            
            # Calculate optimal region
            region_info = calculations.calculate_optimal_region(self.observer, obs_date)
            
            # Store ranges
            self.optimal_ra_range = region_info['ra_range']
            self.optimal_dec_range = region_info['dec_range']
            
            # Format for display
            ra_min_str = self.format_ra_hours(self.optimal_ra_range[0])
            ra_max_str = self.format_ra_hours(self.optimal_ra_range[1])
            dec_min_str = self.format_dec_degrees(self.optimal_dec_range[0])
            dec_max_str = self.format_dec_degrees(self.optimal_dec_range[1])
            
            # Update labels
            self.ra_region_label.config(text=f"{ra_min_str} - {ra_max_str}")
            self.dec_region_label.config(text=f"{dec_min_str} - {dec_max_str}")
            
            # Show info
            strategy = "Opposite side of the moon" if region_info['moon_visible'] else "Zenithal region"
            messagebox.showinfo("Region Calculated", 
                              f"Optimal region calculated for {self.date_var.get()}\n"
                              f"Strategy: {strategy}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating optimal region: {str(e)}")
    
    def format_ra_hours(self, hours):
        """Format RA in hours:minutes."""
        if hours < 0:
            hours += 24
        if hours >= 24:
            hours -= 24
        
        h = int(hours)
        m = int((hours - h) * 60)
        return f"{h:02d}h{m:02d}m"
    
    def format_dec_degrees(self, degrees):
        """Format Dec in degrees:minutes."""
        sign = "+" if degrees >= 0 else "-"
        abs_deg = abs(degrees)
        d = int(abs_deg)
        m = int((abs_deg - d) * 60)
        return f"{sign}{d:02d}°{m:02d}'"
    
    def generate_stelle_doppie_search(self):
        """Generate Stelle Doppie search URL."""
        if not self.optimal_ra_range or not self.optimal_dec_range:
            messagebox.showwarning("Warning", "First calculate the optimal region")
            return
        
        try:
            # Convert ranges to format required by Stelle Doppie
            ra_min_hours, ra_max_hours = self.optimal_ra_range
            dec_min_deg, dec_max_deg = self.optimal_dec_range
            
            # Format for URL
            ra_min_formatted = f"{int(ra_min_hours):02d},{int((ra_min_hours - int(ra_min_hours)) * 60):02d}"
            ra_max_formatted = f"{int(ra_max_hours):02d},{int((ra_max_hours - int(ra_max_hours)) * 60):02d}"
            
            dec_min_sign = "" if dec_min_deg >= 0 else "-"
            dec_max_sign = "" if dec_max_deg >= 0 else "-"
            dec_min_formatted = f"{dec_min_sign}{int(abs(dec_min_deg)):02d},{int((abs(dec_min_deg) - int(abs(dec_min_deg))) * 60):02d}"
            dec_max_formatted = f"{dec_max_sign}{int(abs(dec_max_deg)):02d},{int((abs(dec_max_deg) - int(abs(dec_max_deg))) * 60):02d}"
            
            # Build URL
            base_url = "https://www.stelledoppie.it/index2.php"
            params = {
                'metodo-cat_wds-ra': '7',
                'dato-cat_wds-ra': f'{ra_min_formatted}%2C{ra_max_formatted}',
                'metodo-cat_wds-de': '7',
                'dato-cat_wds-de': f'{dec_min_formatted}%2C{dec_max_formatted}',
                'metodo-cat_wds-raggio': '6',
                'dato-cat_wds-raggio': '',
                'metodo-cat_wds-coord_2000': '1',
                'dato-cat_wds-coord_2000': '',
                'metodo-cat_wds-discov_num': '1',
                'dato-cat_wds-discov_num': '',
                'metodo-cat_wds-comp': '1',
                'dato-cat_wds-comp': '',
                'metodo-cat_wds-name': '9',
                'dato-cat_wds-name': '',
                'metodo-cat_wds-date_first': '6',
                'dato-cat_wds-date_first': '2000',
                'metodo-cat_wds-date_last': '6',
                'dato-cat_wds-date_last': '2020',
                'metodo-cat_wds-mag_pri': '7',
                'dato-cat_wds-mag_pri': '10%2C15',
                'metodo-cat_wds-mag_sec': '7',
                'dato-cat_wds-mag_sec': '10%2C15',
                'metodo-cat_wds-calc_delta_mag': '6',
                'dato-cat_wds-calc_delta_mag': '2',
                'metodo-cat_wds-sep_last': '4',
                'dato-cat_wds-sep_last': '10',
                'metodo-cat_wds-spectr': '11',
                'dato-cat_wds-spectr': '',
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
                'metodo-calc_wds_other-Gaia': '18',
                'dato-calc_wds_other-Gaia': '',
                'metodo-cat_wds-dm_number': '1',
                'dato-cat_wds-dm_number': '',
                'metodo-cat_wds-obs': '4',
                'dato-cat_wds-obs': '10',
                'metodo-cat_wds-notes': '9',
                'dato-cat_wds-notes': '',
                'metodo-cat_wds-notes-notes': '9',
                'dato-cat_wds-notes-notes': '',
                'metodo-cat_wds-reports': '3',
                'dato-cat_wds-reports': '',
                'metodo-cat_wds-filtro_visuale': '1',
                'metodo-cat_wds-filtro_strumento': '1',
                'metodo-cat_wds-filtro_coord': '1',
                'metodo-cat_wds-filtro_orbita': '1',
                'metodo-cat_wds-filtro_nome': '1',
                'metodo-cat_wds-filtro_principale': '1',
                'metodo-cat_wds-filtro_fisica': '1',
                'metodo-cat_wds-filtro_incerta': '1',
                'dato-cat_wds-filtro_incerta': 'S',
                'metodo-cat_wds-calc_tot_comp': '1',
                'dato-cat_wds-calc_tot_comp': '',
                'menu': '21',
                'section': '2',
                'azione': 'cerca_nel_database',
                'limite': '',
                'righe': '50',
                'orderby': '',
                'type': '3',
                'set_filtri': 'S',
                'gocerca': 'Search+the+database'
            }

            # Construct full URL
            url_parts = [f"{k}={v}" for k, v in params.items()]
            full_url = f"{base_url}?" + "&".join(url_parts)
            
            # Display URL
            self.url_text.delete(1.0, tk.END)
            self.url_text.insert(tk.END, full_url)
            
            # Option to open in browser
            result = messagebox.askyesno("Open Search", 
                                       "Do you want to open the search in your browser?")
            if result:
                webbrowser.open(full_url)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error generating search: {str(e)}")

def main():
    """Main entry point for the planner GUI."""
    root = tk.Tk()
    app = AstraKairosPlannerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()