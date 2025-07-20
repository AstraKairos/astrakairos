# astrakairos/planner/gui/calculation_widgets.py

"""
Calculation parameters and astronomical results widgets for the GUI.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
from typing import Dict

from ...planner import calculations
from ...config import (
    MIN_ALTITUDE_DEG, MAX_ALTITUDE_DEG,
    MIN_RA_WINDOW_HOURS, MAX_RA_WINDOW_HOURS,
    MIN_LIGHT_POLLUTION_MAG, MAX_LIGHT_POLLUTION_MAG,
    DEFAULT_MIN_ALTITUDE_DEG, DEFAULT_RA_WINDOW_HOURS, DEFAULT_LIGHT_POLLUTION_MAG
)
from .utilities import format_ra_hours, format_dec_degrees


class CalculationManager:
    """Manages astronomical calculation parameters and results."""
    
    def __init__(self, parent_app):
        self.app = parent_app
        self.optimal_ra_range = None
        self.optimal_dec_range = None
    
    def create_calculation_section(self, parent):
        """Create the complete calculation section with date, params, and astro info."""
        # Create all calculation-related sections
        date_section = self.create_date_section(parent)
        params_section = self.create_params_section(parent)
        astro_section = self.create_astro_info_section(parent)
        return date_section, params_section, astro_section
    
    def create_date_section(self, parent):
        """Create the observation date selection interface."""
        frame = ttk.LabelFrame(parent, text="2. Select Observation Date", padding="10")
        frame.grid(row=3, column=0, sticky="ew", pady=5)
        
        ttk.Label(frame, text="Date (YYYY-MM-DD):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.app.date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(frame, textvariable=self.app.date_var, width=15).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(frame, text="Tonight", 
                  command=lambda: self.app.date_var.set(datetime.now().strftime("%Y-%m-%d"))).grid(row=0, column=2, padx=10)
        
        return frame
    
    def create_params_section(self, parent):
        """Create the observation parameters interface."""
        frame = ttk.LabelFrame(parent, text="3. Set Observation Parameters", padding="10")
        frame.grid(row=4, column=0, sticky="ew", pady=5)
        
        ttk.Label(frame, text="Min Altitude (deg):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.app.min_alt_var = tk.DoubleVar(value=DEFAULT_MIN_ALTITUDE_DEG)
        ttk.Entry(frame, textvariable=self.app.min_alt_var, width=10).grid(row=0, column=1)
        
        ttk.Label(frame, text="RA Window (± hours):").grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        self.app.ra_win_var = tk.DoubleVar(value=DEFAULT_RA_WINDOW_HOURS)
        ttk.Entry(frame, textvariable=self.app.ra_win_var, width=10).grid(row=0, column=3)
        
        ttk.Label(frame, text="Light Pollution (mag/arcsec²):").grid(row=1, column=0, sticky=tk.W, pady=(5,0), padx=(0,5))
        self.app.lp_var = tk.DoubleVar(value=DEFAULT_LIGHT_POLLUTION_MAG)
        ttk.Entry(frame, textvariable=self.app.lp_var, width=10).grid(row=1, column=1, pady=(5,0))
        
        return frame
    
    def create_astro_info_section(self, parent):
        """Create the calculation results display interface."""
        frame = ttk.LabelFrame(parent, text="4. Calculate & Review Conditions", padding="10")
        frame.grid(row=5, column=0, sticky="nsew", pady=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        
        ttk.Button(frame, text="Calculate Optimal Conditions", 
                   command=self.run_full_calculation).grid(row=0, column=0, pady=(0, 10))
        
        self.app.astro_text = ScrolledText(frame, height=12, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.app.astro_text.grid(row=1, column=0, sticky="nsew")
        
        # Configure mouse wheel scrolling
        from .utilities import GUIUtilities
        GUIUtilities.configure_text_widget_scroll(self.app.astro_text)
        
        return frame
    
    def run_full_calculation(self):
        """Run the complete astronomical calculation."""
        location_manager = self.app.location_manager
        if not location_manager.observer_location:
            messagebox.showerror("Location Required", 
                               "Please select an observatory location first.")
            return
            
        try:
            # Parse date and validate parameters
            obs_date = datetime.strptime(self.app.date_var.get(), "%Y-%m-%d")
            
            # Get and validate numeric parameters
            try:
                min_alt = float(self.app.min_alt_var.get())
            except (ValueError, TypeError):
                messagebox.showerror("Invalid Input", "Minimum altitude must be a valid number")
                return
                
            try:
                ra_win = float(self.app.ra_win_var.get())
            except (ValueError, TypeError):
                messagebox.showerror("Invalid Input", "RA window must be a valid number")
                return
                
            try:
                light_pollution = float(self.app.lp_var.get())
            except (ValueError, TypeError):
                messagebox.showerror("Invalid Input", "Light pollution must be a valid number")
                return
            
            # Validate parameter ranges
            if not (MIN_ALTITUDE_DEG <= min_alt <= MAX_ALTITUDE_DEG):
                messagebox.showerror("Invalid Altitude", 
                                   f"Altitude must be between {MIN_ALTITUDE_DEG}° and {MAX_ALTITUDE_DEG}°")
                return
            
            if not (MIN_RA_WINDOW_HOURS <= ra_win <= MAX_RA_WINDOW_HOURS):
                messagebox.showerror("Invalid RA Window", 
                                   f"RA window must be between {MIN_RA_WINDOW_HOURS} and {MAX_RA_WINDOW_HOURS} hours")
                return
            
            if not (MIN_LIGHT_POLLUTION_MAG <= light_pollution <= MAX_LIGHT_POLLUTION_MAG):
                messagebox.showerror("Invalid Light Pollution", 
                                   f"Light pollution must be between {MIN_LIGHT_POLLUTION_MAG} and {MAX_LIGHT_POLLUTION_MAG} mag/arcsec²")
                return
            
            # Calculate nightly events
            timezone = 'UTC'  # Default to UTC, could be made configurable
            events = calculations.get_nightly_events(location_manager.observer_location, obs_date, timezone)
            
            # Calculate conditions at start of darkness (astronomical twilight end)
            calc_time = events.get('astronomical_twilight_end_utc') or obs_date.replace(hour=21)
            conditions_twilight = calculations.calculate_sky_conditions_at_time(
                location_manager.observer_location, calc_time)
            
            # Calculate conditions at temporal midnight if available
            conditions_midnight = None
            if events.get('temporal_midnight_utc'):
                conditions_midnight = calculations.calculate_sky_conditions_at_time(
                    location_manager.observer_location, events['temporal_midnight_utc'])
            
            # Generate optimal sky quality map
            optimal_patch = calculations.generate_sky_quality_map(
                location_manager.observer_location, calc_time,
                min_altitude_deg=min_alt,
                sky_brightness_mag_arcsec2=light_pollution,
                grid_resolution_arcmin=30  # 30 arcmin resolution for faster calculation
            )
            
            # Calculate optimal coordinate ranges for searches
            ra_center = optimal_patch['best_ra_hours']
            self.optimal_ra_range = ((ra_center - ra_win + 24) % 24, (ra_center + ra_win) % 24)
            
            dec_center = optimal_patch['best_dec_deg']
            self.optimal_dec_range = (max(dec_center - 20, -90), min(dec_center + 20, 90))
            
            # Display results
            self.display_results(events, conditions_twilight, conditions_midnight, optimal_patch, calc_time)

        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please ensure Date and Parameters are valid numbers: {e}")
        except Exception as e:
            messagebox.showerror("Calculation Error", f"An unexpected error occurred: {e}")

    def display_results(self, events: Dict, conditions_twilight: Dict, conditions_midnight: Dict, 
                       optimal_patch: Dict, calc_time: datetime):
        """Display the calculation results in the text widget."""
        self.app.astro_text.config(state=tk.NORMAL)
        self.app.astro_text.delete(1.0, tk.END)
        
        # Helper functions for formatting
        def fmt_time(dt_obj):
            return dt_obj.strftime("%H:%M UTC") if dt_obj else "N/A"
        
        def fmt_time_utc(dt_obj):
            return dt_obj.strftime("%H:%M UTC") if dt_obj else "N/A"
        
        def format_ra_hours(hours):
            h = int(hours)
            m = int((hours - h) * 60)
            return f"{h:02d}h{m:02d}m"
        
        def format_dec_degrees(degrees):
            d = int(degrees)
            m = int(abs(degrees - d) * 60)
            return f"{d:+03d}°{m:02d}'"

        # --- Base Timeline ---
        timeline_text = (
            f"--- OBSERVATION TIMELINE FOR {self.app.date_var.get()} ---\n"
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

        # --- Recommended Observing Region ---
        recommendation_text = (
            f"\n--- RECOMMENDED OBSERVING REGION (Calculated for Start of Darkness) ---\n"
            f"Best Patch Center (RA): {format_ra_hours(optimal_patch['best_ra_hours'])}\n"
            f"Best Patch Center (Dec):{format_dec_degrees(optimal_patch['best_dec_deg'])}\n"
            f"Best Patch Alt/Az:      {optimal_patch['best_alt_deg']:.1f}° / {optimal_patch['best_az_deg']:.1f}°\n"
            f"Quality Score:          {optimal_patch['best_quality_score']:.3f}\n"
            f"RA Search Range:        {format_ra_hours(self.optimal_ra_range[0])} to {format_ra_hours(self.optimal_ra_range[1])}\n"
            f"Dec Search Range:       {format_dec_degrees(self.optimal_dec_range[0])} to {format_dec_degrees(self.optimal_dec_range[1])}\n"
        )
        
        self.app.astro_text.insert(tk.END, timeline_text + conditions_text + recommendation_text)
        self.app.astro_text.config(state=tk.DISABLED)
