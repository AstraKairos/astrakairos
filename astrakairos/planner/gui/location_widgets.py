# astrakairos/planner/gui/location_widgets.py

"""
Location and observatory selection widgets for the GUI.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from ...planner import calculations
from ...config import (
    MIN_LIGHT_POLLUTION_MAG, MAX_LIGHT_POLLUTION_MAG,
    DEFAULT_LIGHT_POLLUTION_MAG
)


class LocationManager:
    """Manages observatory location selection and validation."""
    
    def __init__(self, parent_app):
        self.app = parent_app
        self.locations: List[Dict[str, Any]] = []
        self.filtered_locations: List[Dict[str, Any]] = []
        self.selected_location: Optional[Dict[str, Any]] = None
        self.observer_location = None  # Skyfield topos object
        
    def load_locations(self):
        """Load observatory locations from JSON file."""
        try:
            locations_file = Path(__file__).parent.parent.parent.parent / "locations.json"
            if locations_file.exists():
                with open(locations_file, 'r', encoding='utf-8') as f:
                    self.locations = json.load(f)
                print(f"✅ Loaded {len(self.locations)} observatory locations")
                
                # Initialize filtered locations
                self.filtered_locations = self.locations.copy()
                
                # Update the UI if it exists
                if hasattr(self.app, 'location_listbox') and self.app.location_listbox:
                    self.update_location_list()
                    
            else:
                print(f"⚠️  Locations file not found: {locations_file}")
                self.locations = []
                self.filtered_locations = []
        except Exception as e:
            print(f"❌ Error loading locations: {e}")
            self.locations = []
            self.filtered_locations = []
    
    def create_location_section(self, parent):
        """Create the complete location section with search and info."""
        # Create search section
        search_section = self.create_location_search_section(parent)
        # Create info section  
        info_section = self.create_location_info_section(parent)
        return search_section, info_section
    
    def create_location_search_section(self, parent):
        """Create the location search and selection interface."""
        frame = ttk.LabelFrame(parent, text="🔍 Observatory Search", padding="10")
        frame.grid(row=0, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
        ttk.Label(frame, text="Search:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.app.search_var = tk.StringVar()
        self.app.search_var.trace('w', lambda *args: self.update_location_list())
        search_entry = ttk.Entry(frame, textvariable=self.app.search_var, width=30)
        search_entry.grid(row=0, column=1, sticky="ew", padx=5)
        
        # Location listbox with scrollbar
        list_frame = ttk.Frame(frame)
        list_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        list_frame.columnconfigure(0, weight=1)
        
        self.app.location_listbox = tk.Listbox(list_frame, height=6, exportselection=False)
        self.app.location_listbox.grid(row=0, column=0, sticky="ew")
        self.app.location_listbox.bind('<<ListboxSelect>>', self.on_location_select)
        
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.app.location_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.app.location_listbox.configure(yscrollcommand=scrollbar.set)
        
        return frame
    
    def create_location_info_section(self, parent):
        """Create the selected location information display."""
        frame = ttk.LabelFrame(parent, text="📍 Selected Observatory", padding="10")
        frame.grid(row=1, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
        # Info labels
        self.app.info_labels = {}
        info_items = ['Name', 'Coordinates', 'Altitude']
        
        for i, item in enumerate(info_items):
            ttk.Label(frame, text=f"{item}:").grid(row=i, column=0, sticky="w", padx=(0, 10))
            label = ttk.Label(frame, text="No location selected", foreground='black')
            label.grid(row=i, column=1, sticky="w")
            self.app.info_labels[item] = label
        
        return frame
    
    def on_location_select(self, event):
        """Handle location selection from listbox."""
        if not self.app.location_listbox.curselection(): 
            return
        index = self.app.location_listbox.curselection()[0]
        self.selected_location = self.filtered_locations[index]
        self.update_location_info()
        self.create_observer_location()

    def update_location_list(self):
        """Update the location listbox based on search term."""
        search_term = self.app.search_var.get().lower()
        self.filtered_locations = [loc for loc in self.locations if search_term in loc['name'].lower()]
        self.app.location_listbox.delete(0, tk.END)
        for loc in self.filtered_locations:
            self.app.location_listbox.insert(tk.END, f"{loc['name']} ({loc.get('state', 'N/A')})")

    def update_location_info(self):
        """Update the location information display."""
        if not self.selected_location: 
            return
        loc = self.selected_location
        self.app.info_labels['Name'].config(text=loc['name'], foreground='black')
        self.app.info_labels['Coordinates'].config(text=f"{loc['latitude']}, {loc['longitude']}", foreground='black')
        self.app.info_labels['Altitude'].config(text=f"{loc.get('altitude_m', 'N/A')} m", foreground='black')
        
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
                    self.app.lp_var.set(light_pollution)
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
        
        self.app.lp_var.set(default_lp)

    def _parse_coordinate(self, coord_str):
        """Parse coordinate string to float, handling various formats."""
        if isinstance(coord_str, (int, float)):
            return float(coord_str)
        
        if isinstance(coord_str, str):
            # Remove common suffixes and clean up
            coord_str = coord_str.strip().replace('°', '')
            
            # Handle N/S/E/W suffixes
            multiplier = 1
            if coord_str.upper().endswith(('S', 'W')):
                multiplier = -1
                coord_str = coord_str[:-1].strip()
            elif coord_str.upper().endswith(('N', 'E')):
                coord_str = coord_str[:-1].strip()
            
            # Try direct conversion
            try:
                return float(coord_str) * multiplier
            except ValueError:
                # If it fails, it might be in DMS format - just return 0 for now
                print(f"⚠️  Could not parse coordinate: {coord_str}")
                return 0.0
        
        return 0.0

    def create_observer_location(self):
        """Create observer location with robust coordinate validation."""
        if not self.selected_location: 
            return
        
        loc = self.selected_location
        try:
            latitude = self._parse_coordinate(loc['latitude'])
            longitude = self._parse_coordinate(loc['longitude'])
            altitude = float(loc.get('altitude_m', 0))
            
            self.observer_location = calculations.get_observer_location(
                latitude_deg=latitude,
                longitude_deg=longitude,
                altitude_m=altitude
            )
            print(f"✅ Observer location created for {loc['name']} ({latitude:.3f}, {longitude:.3f})")
        except (ValueError, TypeError, KeyError) as e:
            messagebox.showerror("Location Error", 
                               f"Invalid coordinates for {loc.get('name', 'selected location')}: {e}")
            self.observer_location = None
    
    def get_current_location_data(self):
        """Get current location data for calculations."""
        if not self.selected_location:
            return None
        
        loc = self.selected_location
        try:
            return {
                'name': loc['name'],
                'latitude': self._parse_coordinate(loc['latitude']),
                'longitude': self._parse_coordinate(loc['longitude']),
                'altitude_m': float(loc.get('altitude_m', 0)),
                'observer_location': self.observer_location
            }
        except Exception as e:
            print(f"⚠️  Error getting location data: {e}")
            return None
