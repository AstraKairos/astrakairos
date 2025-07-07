import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import json
import datetime
from datetime import date, timedelta
import pytz
import ephem
import requests
from typing import Dict, List, Optional
import os
import webbrowser

class ObservatorySelector:
    def __init__(self, root):
        self.root = root
        self.root.title("Observatory Selector - Binary Stars")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.locations = []
        self.filtered_locations = []
        self.selected_location = None
        self.observer = None
        self.optimal_ra_range = None
        self.optimal_dec_range = None
        self.moon_position = None
        

        # Cargar datos
        self.load_locations()
        
        # Crear interfaz
        self.create_widgets()
        
        # Actualizar lista inicial
        self.update_location_list()
    
    def load_locations(self):
        """Cargar ubicaciones desde el archivo JSON - Gracias Stellarium"""
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
        """Crear todos los widgets de la interfaz"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="Observatory Selector for Binary Stars", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Sección de búsqueda
        search_frame = ttk.LabelFrame(main_frame, text="Location Search", padding="10")
        search_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        search_frame.columnconfigure(1, weight=1)
        
        ttk.Label(search_frame, text="Search:").grid(row=0, column=0, sticky=tk.W)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=50)
        self.search_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        self.search_var.trace('w', self.on_search_change)
        
        # Lista de ubicaciones
        self.location_listbox = tk.Listbox(search_frame, height=8, width=70)
        self.location_listbox.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        self.location_listbox.bind('<<ListboxSelect>>', self.on_location_select)
        
        # Scrollbar para la lista
        scrollbar = ttk.Scrollbar(search_frame, orient=tk.VERTICAL, command=self.location_listbox.yview)
        scrollbar.grid(row=1, column=2, sticky=(tk.N, tk.S))
        self.location_listbox.config(yscrollcommand=scrollbar.set)
        
        # Información de ubicación seleccionada
        info_frame = ttk.LabelFrame(main_frame, text="Selected Location", padding="10")
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        info_frame.columnconfigure(1, weight=1)
        
        ttk.Label(info_frame, text="Name:").grid(row=0, column=0, sticky=tk.W)
        self.name_label = ttk.Label(info_frame, text="", font=('Arial', 10, 'bold'))
        self.name_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(info_frame, text="Type:").grid(row=1, column=0, sticky=tk.W)
        self.type_label = ttk.Label(info_frame, text="")
        self.type_label.grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(info_frame, text="Coordinates:").grid(row=2, column=0, sticky=tk.W)
        self.coords_label = ttk.Label(info_frame, text="")
        self.coords_label.grid(row=2, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(info_frame, text="Altitude:").grid(row=3, column=0, sticky=tk.W)
        self.altitude_label = ttk.Label(info_frame, text="")
        self.altitude_label.grid(row=3, column=1, sticky=tk.W, padx=(5, 0))
        
        # Sección de fecha
        date_frame = ttk.LabelFrame(main_frame, text="Observation Date", padding="10")
        date_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(date_frame, text="Date:").grid(row=0, column=0, sticky=tk.W)
        self.date_var = tk.StringVar(value=date.today().strftime("%Y-%m-%d"))
        self.date_entry = ttk.Entry(date_frame, textvariable=self.date_var, width=15)
        self.date_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        self.tonight_btn = ttk.Button(date_frame, text="Tonight", command=self.set_tonight)
        self.tonight_btn.grid(row=0, column=2, padx=(10, 0))
        
        self.calculate_btn = ttk.Button(date_frame, text="Calculate Conditions", 
                                      command=self.calculate_conditions)
        self.calculate_btn.grid(row=0, column=3, padx=(10, 0))
        
        # Información astronómica
        astro_frame = ttk.LabelFrame(main_frame, text="Astronomical Information", padding="10")
        astro_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        astro_frame.columnconfigure(0, weight=1)
        astro_frame.rowconfigure(0, weight=1)
        
        self.astro_text = ScrolledText(astro_frame, height=12, width=70, 
                                      wrap=tk.WORD, state=tk.DISABLED)
        self.astro_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar expansión
        main_frame.rowconfigure(4, weight=1)

        # Sección de búsqueda de estrellas binarias
        binary_frame = ttk.LabelFrame(main_frame, text="Binary Star Search", padding="10")
        binary_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        binary_frame.columnconfigure(1, weight=1)

        # Botón para calcular región óptima
        self.calculate_region_btn = ttk.Button(binary_frame, text="Calculate Optimal Region", 
                                            command=self.calculate_optimal_region)
        self.calculate_region_btn.grid(row=0, column=0, sticky=tk.W)

        # Información de la región calculada
        ttk.Label(binary_frame, text="Optimal RA Region:").grid(row=1, column=0, sticky=tk.W)
        self.ra_region_label = ttk.Label(binary_frame, text="", font=('Arial', 10, 'bold'))
        self.ra_region_label.grid(row=1, column=1, sticky=tk.W, padx=(5, 0))

        ttk.Label(binary_frame, text="Optimal Dec Region:").grid(row=2, column=0, sticky=tk.W)
        self.dec_region_label = ttk.Label(binary_frame, text="", font=('Arial', 10, 'bold'))
        self.dec_region_label.grid(row=2, column=1, sticky=tk.W, padx=(5, 0))

        # Botón para generar búsqueda
        self.generate_search_btn = ttk.Button(binary_frame, text="Generate Search on Stelle Doppie", 
                                            command=self.generate_stelle_doppie_search)
        self.generate_search_btn.grid(row=3, column=0, columnspan=2, pady=(10, 0))

        # URL generada
        self.url_text = ScrolledText(binary_frame, height=3, width=70, wrap=tk.WORD)
        self.url_text.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        # Cambiar la fila de expansión para incluir el nuevo frame
        main_frame.rowconfigure(5, weight=0)  # Frame binario no se expande
        main_frame.rowconfigure(4, weight=1)  # Frame astronómico sigue siendo expansible
        
    def get_location_type_name(self, type_code):
        """Convertir código de tipo a nombre descriptivo"""
        type_map = {
            'C': 'Capital',
            'B': 'Capital',
            'R': 'Regional Capital',
            'O': 'Observatory',
            'H': 'Archaeoastronomical Site'
        }
        return type_map.get(type_code, 'Unknown')
    
    def on_search_change(self, *args):
        """Actualizar lista cuando cambia la búsqueda"""
        self.update_location_list()
    
    def update_location_list(self):
        """Actualizar la lista de ubicaciones basada en la búsqueda"""
        search_term = self.search_var.get().lower()
        
        # Filtrar ubicaciones
        self.filtered_locations = []
        for location in self.locations:
            if (search_term in location['name'].lower() or
                search_term in location.get('state', '').lower() or
                search_term in location.get('region', '').lower()):
                self.filtered_locations.append(location)
        
        # Actualizar listbox
        self.location_listbox.delete(0, tk.END)
        for location in self.filtered_locations:
            display_text = f"{location['name']} ({self.get_location_type_name(location['type'])}) - {location.get('state', '')}"
            self.location_listbox.insert(tk.END, display_text)
    
    def on_location_select(self, event):
        """Manejar selección de ubicación"""
        selection = self.location_listbox.curselection()
        if selection:
            index = selection[0]
            if index < len(self.filtered_locations):
                self.selected_location = self.filtered_locations[index]
                self.update_location_info()
    
    def update_location_info(self):
        """Actualizar información de la ubicación seleccionada"""
        if not self.selected_location:
            return
        
        loc = self.selected_location
        
        # Actualizar labels
        self.name_label.config(text=loc['name'])
        self.type_label.config(text=self.get_location_type_name(loc['type']))
        
        # Formatear coordenadas
        coords_text = f"{loc['latitude']}, {loc['longitude']}"
        self.coords_label.config(text=coords_text)
        
        # Altitud
        altitude_text = f"{loc['altitude_m']} m" if loc['altitude_m'] else "N/A"
        self.altitude_label.config(text=altitude_text)
        
        # Crear observador de ephem
        self.create_observer()
    
    def create_observer(self):
        """Crear observador astronómico para la ubicación seleccionada"""
        if not self.selected_location:
            return
        
        self.observer = ephem.Observer()
        
        # Convertir coordenadas
        lat_str = self.selected_location['latitude']
        lon_str = self.selected_location['longitude']
        
        # Parsear latitud
        if lat_str.endswith('N'):
            lat_deg = float(lat_str[:-1])
        else:  # S
            lat_deg = -float(lat_str[:-1])
        
        # Parsear longitud
        if lon_str.endswith('E'):
            lon_deg = float(lon_str[:-1])
        else:  # W
            lon_deg = -float(lon_str[:-1])
        
        self.observer.lat = str(lat_deg)
        self.observer.lon = str(lon_deg)
        
        # Altitud
        if self.selected_location['altitude_m']:
            self.observer.elevation = self.selected_location['altitude_m']
    
    def set_tonight(self):
        """Establecer fecha a hoy"""
        self.date_var.set(date.today().strftime("%Y-%m-%d"))
    
    def calculate_conditions(self):
        """Calcular condiciones astronómicas"""
        if not self.selected_location or not self.observer:
            messagebox.showwarning("Warning", "First select a location")
            return
        
        try:
            # Parsear fecha
            obs_date = datetime.datetime.strptime(self.date_var.get(), "%Y-%m-%d")
            
            # Establecer fecha en el observador
            self.observer.date = obs_date
            
            # Calcular información
            self.calculate_sun_moon_info()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD")
    
    def calculate_sun_moon_info(self):
        """Calcular información del sol y la luna"""
        # Crear objetos celestes
        sun = ephem.Sun()
        moon = ephem.Moon()
        
        # Obtener huso horario local
        local_tz = pytz.timezone(self.selected_location.get('timezone', 'UTC'))
        
        # Información del sol (para el día seleccionado)
        try:
            # Calcular amanecer y atardecer para el día seleccionado
            obs_date = datetime.datetime.strptime(self.date_var.get(), "%Y-%m-%d")
            self.observer.date = obs_date
            
            sunrise = self.observer.next_rising(sun)
            sunset = self.observer.next_setting(sun)
            
            # Ajustar si es necesario
            if sunrise < sunset:
                sunset = self.observer.previous_setting(sun)
                
        except ephem.CircumpolarError:
            sunrise = "Circumpolar Sun"
            sunset = "Circumpolar Sun"
        
        # Calcular mitad de la noche (para el zenith)
        try:
            if isinstance(sunset, ephem.Date) and isinstance(sunrise, ephem.Date):
                # Convertir a datetime para cálculos
                sunset_dt = sunset.datetime()
                sunrise_dt = sunrise.datetime()
                
                # Si el amanecer es al día siguiente, ajustar
                if sunrise_dt < sunset_dt:
                    sunrise_dt += timedelta(days=1)
                
                # Calcular mitad de la noche
                midnight = sunset_dt + (sunrise_dt - sunset_dt) / 2
                midnight_ephem = ephem.Date(midnight)
                
                # Convertir a hora local
                sunset_local = pytz.utc.localize(sunset_dt).astimezone(local_tz)
                sunrise_local = pytz.utc.localize(sunrise_dt).astimezone(local_tz)
                midnight_local = pytz.utc.localize(midnight).astimezone(local_tz)
                
                # Establecer observer a mitad de la noche para zenith
                self.observer.date = midnight_ephem
                
                # Calcular zenith (tiempo sidéreo local)
                zenith_ra_rad = self.observer.sidereal_time()
                zenith_dec_rad = self.observer.lat
                
                # Convertir a horas y grados
                zenith_ra_hours = float(zenith_ra_rad) * 12.0 / 3.14159  # radianes a horas
                zenith_dec_degrees = float(zenith_dec_rad) * 180.0 / 3.14159  # radianes a grados
                
                # Formatear RA en horas:minutos:segundos
                ra_h = int(zenith_ra_hours)
                ra_m = int((zenith_ra_hours - ra_h) * 60)
                ra_s = ((zenith_ra_hours - ra_h) * 60 - ra_m) * 60
                ra_formatted = f"{ra_h:02d}h {ra_m:02d}m {ra_s:04.1f}s"
                
                # Formatear Dec en grados:minutos:segundos
                dec_sign = "+" if zenith_dec_degrees >= 0 else "-"
                dec_abs = abs(zenith_dec_degrees)
                dec_d = int(dec_abs)
                dec_m = int((dec_abs - dec_d) * 60)
                dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60
                dec_formatted = f"{dec_sign}{dec_d:02d}° {dec_m:02d}' {dec_s:04.1f}\""
                
                # Formatear horas locales y UTC
                sunset_str = f"{sunset_local.strftime('%H:%M:%S')} (UTC {sunset_dt.strftime('%H:%M:%S')})"
                sunrise_str = f"{sunrise_local.strftime('%H:%M:%S')} (UTC {sunrise_dt.strftime('%H:%M:%S')})"
                midnight_str = f"{midnight_local.strftime('%H:%M:%S')} (UTC {midnight.strftime('%H:%M:%S')})"
            else:
                ra_formatted = "N/A (Circumpolar sun)"
                dec_formatted = "N/A (Circumpolar sun)"
                sunset_str = "N/A"
                sunrise_str = "N/A"
                midnight_str = "N/A"
                
        except Exception as e:
            ra_formatted = "Calculation Error"
            dec_formatted = "Calculation Error"
            sunset_str = "Error"
            sunrise_str = "Error"
            midnight_str = "Error"
        
        # Información de la luna (durante toda la noche)
        try:
            # Resetear fecha al día seleccionado
            self.observer.date = datetime.datetime.strptime(self.date_var.get(), "%Y-%m-%d")
            
            moonrise = self.observer.next_rising(moon)
            moonset = self.observer.next_setting(moon)
            
            if moonrise < moonset:
                moonset = self.observer.previous_setting(moon)
            
            # Convertir a hora local
            moonrise_local = pytz.utc.localize(moonrise.datetime()).astimezone(local_tz)
            moonset_local = pytz.utc.localize(moonset.datetime()).astimezone(local_tz)
            moonrise_str = f"{moonrise_local.strftime('%H:%M:%S')} (UTC {moonrise.datetime().strftime('%H:%M:%S')})"
            moonset_str = f"{moonset_local.strftime('%H:%M:%S')} (UTC {moonset.datetime().strftime('%H:%M:%S')})"
                
        except ephem.CircumpolarError:
            moonrise_str = "Circumpolar Moon"
            moonset_str = "Circumpolar Moon"
        
        # Calcular luna a mitad de la noche
        if isinstance(midnight_ephem, ephem.Date):
            self.observer.date = midnight_ephem
            moon.compute(self.observer)
            
            # Fase lunar
            moon_phase = moon.phase
            
            # Determinar descripción de fase
            if moon_phase < 1:
                phase_desc = "New Moon"
            elif moon_phase < 25:
                phase_desc = "Waxing Crescent"
            elif moon_phase < 50:
                phase_desc = "First Quarter"
            elif moon_phase < 75:
                phase_desc = "Waxing Gibbous"
            elif moon_phase < 99:
                phase_desc = "Full Moon"
            else:
                phase_desc = "Waning Gibbous"
            
            # Posición lunar a mitad de la noche
            moon_alt_deg = float(moon.alt) * 180 / 3.14159
            moon_az_deg = float(moon.az) * 180 / 3.14159
            
            # Determinar visibilidad durante la noche
            # Simplificado: si la luna está sobre el horizonte a mitad de la noche, es visible
            if moon_alt_deg > 0:
                visibility_info = f"VISIBLE at midnight (altitude: {moon_alt_deg:.1f}°)"
                
                # Calcular período de visibilidad nocturna
                if isinstance(moonrise, ephem.Date) and isinstance(moonset, ephem.Date):
                    # Convertir tiempos a datetime locales para comparación
                    moonrise_dt_local = moonrise_local
                    moonset_dt_local = moonset_local
                    sunset_dt_local = sunset_local
                    sunrise_dt_local = sunrise_local
                    
                    # Ajustar si moonset es al día siguiente
                    if moonset_dt_local.time() < moonrise_dt_local.time():
                        moonset_dt_local = moonset_dt_local + timedelta(days=1)
                    
                    # Ajustar si sunrise es al día siguiente
                    if sunrise_dt_local.time() < sunset_dt_local.time():
                        sunrise_dt_local = sunrise_dt_local + timedelta(days=1)
                    
                    # Calcular ventana de visibilidad durante la noche
                    night_start = sunset_dt_local
                    night_end = sunrise_dt_local
                    
                    # Ventana de luna visible
                    moon_start = moonrise_dt_local
                    moon_end = moonset_dt_local
                    
                    # Intersección: cuándo la luna está visible Y es de noche
                    visible_start = max(night_start, moon_start)
                    visible_end = min(night_end, moon_end)
                    
                    if visible_start < visible_end:
                        duration_hours = (visible_end - visible_start).total_seconds() / 3600
                        visibility_times = f"Visible during the night from {visible_start.strftime('%H:%M')} to {visible_end.strftime('%H:%M')} ({duration_hours:.1f} hours)"
                    else:
                        visibility_times = "Not visible during full darkness hours"
                else:
                    visibility_times = "Rise/set information not available"
            else:
                visibility_info = f"NOT VISIBLE at midnight (altitude: {moon_alt_deg:.1f}°)"
                visibility_times = "The moon is below the horizon at midnight"
        else:
            moon_phase = 0
            phase_desc = "Calculation Error"
            moon_alt_deg = 0
            moon_az_deg = 0
            visibility_info = "Calculation Error"
            visibility_times = ""
        
        # Crear texto informativo
        timezone_name = self.selected_location.get('timezone', 'UTC')
        info_text = f"""ASTRONOMICAL CONDITIONS
Date: {self.date_var.get()}
Location: {self.selected_location['name']}
Timezone: {timezone_name}

SOLAR INFORMATION:
Sunset: {sunset_str}
Sunrise: {sunrise_str}
Midnight: {midnight_str}

ZENITH (Highest point in the sky at midnight):
Right Ascension: {ra_formatted}
Declination: {dec_formatted}

LUNAR INFORMATION:
Moonrise: {moonrise_str}
Moonset: {moonset_str}
Moon Phase: {phase_desc} ({moon_phase:.1f}%)
Position at midnight:
  - Altitude: {moon_alt_deg:.1f}°
  - Azimuth: {moon_az_deg:.1f}°

LUNAR VISIBILITY:
{visibility_info}
{visibility_times}
"""
        
        # Mostrar información (texto copiable)
        self.astro_text.config(state=tk.NORMAL)
        self.astro_text.delete(1.0, tk.END)
        self.astro_text.insert(tk.END, info_text)
        # NO deshabilitamos el texto para que sea copiable
        self.astro_text.config(state=tk.NORMAL)

    def calculate_optimal_region(self):
        """Calcular la región óptima para observación de estrellas binarias"""
        if not self.selected_location or not self.observer:
            messagebox.showwarning("Warning", "First select a location and calculate conditions")
            return
        
        try:
            # Obtener fecha y establecer mitad de la noche
            obs_date = datetime.datetime.strptime(self.date_var.get(), "%Y-%m-%d")
            
            # Calcular sunset y sunrise para obtener mitad de la noche
            self.observer.date = obs_date
            sun = ephem.Sun()
            
            try:
                sunset = self.observer.next_setting(sun)
                sunrise = self.observer.next_rising(sun)
                
                if sunrise < sunset:
                    sunset = self.observer.previous_setting(sun)
                
                # Calcular mitad de la noche
                midnight = sunset + (sunrise - sunset) / 2
                
            except ephem.CircumpolarError:
                # Si hay sol circumpolar, usar medianoche local
                midnight = ephem.Date(obs_date + timedelta(hours=12))
            
            # Establecer observador a mitad de la noche
            self.observer.date = midnight
            
            # Calcular posición de la luna
            moon = ephem.Moon()
            moon.compute(self.observer)
            
            # Guardar posición lunar
            self.moon_position = {
                'ra': float(moon.ra),  # en radianes
                'dec': float(moon.dec),  # en radianes
                'alt': float(moon.alt),  # en radianes
                'visible': float(moon.alt) > 0
            }
            
            # Calcular zenith (coordenadas del punto más alto)
            zenith_ra = self.observer.sidereal_time()  # en radianes
            zenith_dec = self.observer.lat  # en radianes
            
            # Región óptima: 40° bajo el zenith hasta el zenith
            # 40° = 40 * π/180 radianes ≈ 0.698 radianes
            min_altitude = 40 * 3.14159 / 180  # 40° en radianes
            
            # Calcular rango de declinación
            # Dec mínima = zenith_dec - 40°
            # Dec máxima = zenith_dec (o 90° si está más al norte)
            dec_min = zenith_dec - min_altitude
            dec_max = min(zenith_dec, 3.14159/2)  # máximo 90°
            
            # Calcular rango de RA considerando la posición de la luna
            if self.moon_position['visible']:
                # Luna visible: buscar en lado opuesto
                moon_ra = self.moon_position['ra']
                
                # Calcular RA opuesta (± 90° de la luna)
                ra_opposite_center = moon_ra + 3.14159  # 180° opuesto
                if ra_opposite_center > 2 * 3.14159:
                    ra_opposite_center -= 2 * 3.14159
                
                # Rango de ±3 horas (45°) alrededor del punto opuesto
                ra_range_radians = 3 * 3.14159 / 12  # 3 horas en radianes
                ra_min = ra_opposite_center - ra_range_radians
                ra_max = ra_opposite_center + ra_range_radians
                
                # Ajustar si cruza 0h/24h
                if ra_min < 0:
                    ra_min += 2 * 3.14159
                if ra_max > 2 * 3.14159:
                    ra_max -= 2 * 3.14159
                    
            else:
                # Luna no visible: usar rango alrededor del zenith
                ra_range_radians = 3 * 3.14159 / 12  # 3 horas en radianes
                ra_min = zenith_ra - ra_range_radians
                ra_max = zenith_ra + ra_range_radians
                
                # Ajustar si cruza 0h/24h
                if ra_min < 0:
                    ra_min += 2 * 3.14159
                if ra_max > 2 * 3.14159:
                    ra_max -= 2 * 3.14159
            
            # Convertir a formato de horas y grados
            ra_min_hours = ra_min * 12 / 3.14159
            ra_max_hours = ra_max * 12 / 3.14159
            dec_min_degrees = dec_min * 180 / 3.14159
            dec_max_degrees = dec_max * 180 / 3.14159
            
            # Guardar rangos
            self.optimal_ra_range = (ra_min_hours, ra_max_hours)
            self.optimal_dec_range = (dec_min_degrees, dec_max_degrees)
            
            # Formatear para mostrar
            ra_min_str = self.format_ra_hours(ra_min_hours)
            ra_max_str = self.format_ra_hours(ra_max_hours)
            dec_min_str = self.format_dec_degrees(dec_min_degrees)
            dec_max_str = self.format_dec_degrees(dec_max_degrees)
            
            # Actualizar labels
            self.ra_region_label.config(text=f"{ra_min_str} - {ra_max_str}")
            self.dec_region_label.config(text=f"{dec_min_str} - {dec_max_str}")
            
            # Mostrar información adicional
            moon_status = "visible" if self.moon_position['visible'] else "not visible"
            messagebox.showinfo("Region Calculated", 
                            f"Optimal region calculated for {self.date_var.get()}\n"
                            f"Moon: {moon_status}\n"
                            f"Strategy: {'Opposite side of the moon' if self.moon_position['visible'] else 'Zenithal region'}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating optimal region: {str(e)}")

    def format_ra_hours(self, hours):
        """Formatear RA en horas:minutos"""
        if hours < 0:
            hours += 24
        if hours >= 24:
            hours -= 24
        
        h = int(hours)
        m = int((hours - h) * 60)
        return f"{h:02d}h{m:02d}m"

    def format_dec_degrees(self, degrees):
        """Formatear Dec en grados:minutos"""
        sign = "+" if degrees >= 0 else "-"
        abs_deg = abs(degrees)
        d = int(abs_deg)
        m = int((abs_deg - d) * 60)
        return f"{sign}{d:02d}°{m:02d}'"

    def generate_stelle_doppie_search(self):
        """Generar URL de búsqueda para Stelle Doppie"""
        if not self.optimal_ra_range or not self.optimal_dec_range:
            messagebox.showwarning("Warning", "First calculate the optimal region")
            return
        
        try:
            # Convertir rangos a formato requerido por Stelle Doppie
            ra_min_hours, ra_max_hours = self.optimal_ra_range
            dec_min_deg, dec_max_deg = self.optimal_dec_range
            
            # Formatear RA para URL (formato HH,HH)
            ra_min_formatted = f"{int(ra_min_hours):02d},{int((ra_min_hours - int(ra_min_hours)) * 60):02d}"
            ra_max_formatted = f"{int(ra_max_hours):02d},{int((ra_max_hours - int(ra_max_hours)) * 60):02d}"
            
            # Formatear Dec para URL (formato DD,MM)
            dec_min_sign = "" if dec_min_deg >= 0 else "-"
            dec_max_sign = "" if dec_max_deg >= 0 else "-"
            dec_min_formatted = f"{dec_min_sign}{int(abs(dec_min_deg)):02d},{int((abs(dec_min_deg) - int(abs(dec_min_deg))) * 60):02d}"
            dec_max_formatted = f"{dec_max_sign}{int(abs(dec_max_deg)):02d},{int((abs(dec_max_deg) - int(abs(dec_max_deg))) * 60):02d}"
            
            # Construir URL con parámetros personalizados
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
            
            # Construir URL completa
            url_parts = [f"{k}={v}" for k, v in params.items()]
            full_url = f"{base_url}?" + "&".join(url_parts)
            
            # Mostrar URL en el texto
            self.url_text.delete(1.0, tk.END)
            self.url_text.insert(tk.END, full_url)
            
            # Opción para abrir en navegador
            import webbrowser
            result = messagebox.askyesno("Open search", 
                                    "Do you want to open the Stelle Doppie search in your browser?")
            if result:
                webbrowser.open(full_url)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error while generating search: {str(e)}")

def main():
    # Verificar dependencias
    try:
        import ephem
        import pytz
    except ImportError as e:
        print(f"Error: Missing dependencies: {e}")
        print("Install with: pip install pyephem pytz")
        return
    
    # Crear ventana principal
    root = tk.Tk()
    app = ObservatorySelector(root)
    root.mainloop()

if __name__ == "__main__":
    main()