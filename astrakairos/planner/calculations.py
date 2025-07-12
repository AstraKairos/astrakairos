# astrakairos/planner/calculations.py

import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, Tuple

from skyfield.api import load, wgs84
from skyfield.almanac import find_discrete, risings_and_settings
from skyfield import almanac

# --- Global objects for Skyfield ---
# These are loaded once when the module is imported for efficiency.
# Skyfield will download the necessary ephemeris data on the first run.
ts = load.timescale()
eph = load('de421.bsp')
earth = eph['earth']
sun = eph['sun']
moon = eph['moon']


def get_observer_location(latitude_deg: float, longitude_deg: float, altitude_m: float):
    """
    Creates a Skyfield geographic location object (topos).

    Args:
        latitude_deg: Latitude in decimal degrees (+N, -S).
        longitude_deg: Longitude in decimal degrees (+E, -W).
        altitude_m: Altitude in meters.

    Returns:
        A Skyfield topos object representing the observer's location.
    """
    return wgs84.latlon(latitude_deg, longitude_deg, elevation_m=altitude_m)


def calculate_astronomical_midnight(observer_location, obs_date: datetime, timezone: str = 'UTC') -> Dict[str, Any]:
    """
    Calculates the astronomical midnight (when the Sun reaches its lowest point below the horizon).
    
    Args:
        observer_location: Skyfield topos object
        obs_date: Observation date
        timezone: Timezone string for local time conversion
        
    Returns:
        Dictionary with astronomical midnight times in UTC and local time
    """
    local_tz = pytz.timezone(timezone)
    
    # Create a 48-hour window to ensure we capture sunset and sunrise
    t0_utc = datetime(obs_date.year, obs_date.month, obs_date.day, 0, 0, 0)
    t2_utc = t0_utc + timedelta(days=2)
    
    # Find Sun's minimum altitude (astronomical midnight)
    observer = earth + observer_location
    
    # Create time range with higher resolution for accurate minimum finding
    times = []
    altitudes = []
    
    # Sample every 10 minutes over 48 hours
    current_time = t0_utc
    while current_time < t2_utc:
        utc_time = pytz.utc.localize(current_time)
        t = ts.from_datetime(utc_time)
        
        sun_apparent = observer.at(t).observe(sun).apparent()
        sun_alt, _, _ = sun_apparent.altaz()
        
        times.append(current_time)
        altitudes.append(sun_alt.degrees)
        
        current_time += timedelta(minutes=10)
    
    # Find the minimum altitude (astronomical midnight)
    min_idx = np.argmin(altitudes)
    astronomical_midnight_utc = times[min_idx]
    
    # Convert to local time
    astronomical_midnight_local = pytz.utc.localize(astronomical_midnight_utc).astimezone(local_tz)
    
    return {
        'astronomical_midnight_utc': astronomical_midnight_utc,
        'astronomical_midnight_local': astronomical_midnight_local,
        'sun_altitude_at_midnight': altitudes[min_idx]
    }


def get_nightly_events(observer_location, obs_date: datetime, timezone: str = 'UTC') -> Dict[str, Any]:
    """
    Calculates all relevant astronomical events for a given observation night.
    
    This robust version uses specific almanac functions for each event and correctly
    calculates temporal midnight (the midpoint between sunset and sunrise).
    """
    local_tz = pytz.timezone(timezone)
    
    # Define a 24-hour window around the observation night, from noon to noon.
    t0_utc = datetime(obs_date.year, obs_date.month, obs_date.day, 12, 0, 0)
    t1_utc = t0_utc + timedelta(days=1)
    t0 = ts.from_datetime(pytz.utc.localize(t0_utc))
    t1 = ts.from_datetime(pytz.utc.localize(t1_utc))

    # Helper to find the first 'set' (event code 0) and 'rise' (event code 1)
    def find_set_rise(times, events):
        set_time = None
        rise_time = None
        for t, e in zip(times, events):
            if e == 0 and set_time is None:
                set_time = t.utc_datetime()
            elif e == 1 and rise_time is None:
                rise_time = t.utc_datetime()
        return set_time, rise_time

    # --- Sun Events ---
    f_sun = almanac.risings_and_settings(eph, sun, observer_location, horizon_degrees=-0.833)
    sun_times, sun_events = find_discrete(t0, t1, f_sun)
    sunset_time, sunrise_time = find_set_rise(sun_times, sun_events)

    # --- Twilight Events ---
    f_civil = almanac.risings_and_settings(eph, sun, observer_location, horizon_degrees=-6.0)
    civil_times, civil_events = find_discrete(t0, t1, f_civil)
    civil_twilight_end, civil_twilight_start = find_set_rise(civil_times, civil_events)

    f_nautical = almanac.risings_and_settings(eph, sun, observer_location, horizon_degrees=-12.0)
    nautical_times, nautical_events = find_discrete(t0, t1, f_nautical)
    nautical_twilight_end, nautical_twilight_start = find_set_rise(nautical_times, nautical_events)
    
    f_astro = almanac.risings_and_settings(eph, sun, observer_location, horizon_degrees=-18.0)
    astro_times, astro_events = find_discrete(t0, t1, f_astro)
    astro_twilight_end, astro_twilight_start = find_set_rise(astro_times, astro_events)

    # --- Moon Events ---
    f_moon = risings_and_settings(eph, moon, observer_location)
    moon_times, moon_events = find_discrete(t0, t1, f_moon)
    moonset_time, moonrise_time = find_set_rise(moon_times, moon_events)

    results = {
        'sunset_utc': sunset_time,
        'civil_twilight_end': civil_twilight_end,
        'nautical_twilight_end': nautical_twilight_end,
        'astronomical_twilight_end': astro_twilight_end,
        'astronomical_twilight_start': astro_twilight_start,
        'nautical_twilight_start': nautical_twilight_start,
        'civil_twilight_start': civil_twilight_start,
        'sunrise_utc': sunrise_time,
        'moonrise_utc': moonrise_time,
        'moonset_utc': moonset_time,
    }

    # --- **FIX**: Calculate Temporal Midnight here ---
    temporal_midnight_utc = None
    if sunset_time and sunrise_time:
        # Handle the case where sunrise is the next calendar day
        if sunrise_time < sunset_time:
            sunrise_next_day = sunrise_time + timedelta(days=1)
            temporal_midnight_utc = sunset_time + (sunrise_next_day - sunset_time) / 2
        else:
            temporal_midnight_utc = sunset_time + (sunrise_time - sunset_time) / 2
    
    results['temporal_midnight_utc'] = temporal_midnight_utc
    # --- End of Fix ---

    # Calculate astronomical midnight and add it to the results
    midnight_data = calculate_astronomical_midnight(observer_location, obs_date, timezone)
    results.update(midnight_data)

    # Add local time versions for all found events
    for key, val in list(results.items()):
        if isinstance(val, datetime) and not key.endswith('_local') and key.endswith('_utc'):
            results[key.replace('_utc', '_local')] = val.astimezone(local_tz)
        elif not val and not key.endswith('_local') and key.endswith('_utc'):
            results[key.replace('_utc', '_local')] = None
            
    return results


def calculate_sky_conditions_at_time(observer_location, time_utc: datetime) -> Dict[str, Any]:
    """Calculates moon and zenith properties for a specific moment in time."""
    if not time_utc.tzinfo:
        time_utc = pytz.utc.localize(time_utc)
        
    t = ts.from_datetime(time_utc)
    observer = earth + observer_location
    
    # Moon properties
    moon_apparent = observer.at(t).observe(moon).apparent()
    moon_alt, moon_az, _ = moon_apparent.altaz()
    moon_ra, moon_dec, _ = moon_apparent.radec()
    
    # Calculate moon phase
    sun_apparent = observer.at(t).observe(sun).apparent()
    moon_phase_percent = almanac.fraction_illuminated(eph, 'moon', t) * 100.0

    # Zenith properties
    zenith_apparent = observer.at(t).from_altaz(alt_degrees=90, az_degrees=0)
    zenith_ra, zenith_dec, _ = zenith_apparent.radec()
    
    # Format RA and Dec strings manually
    def format_ra_hours(hours):
        h = int(hours)
        m = int((hours - h) * 60)
        s = int(((hours - h) * 60 - m) * 60)
        return f"{h:02d}h{m:02d}m{s:02d}s"
    
    def format_dec_degrees(degrees):
        sign = "+" if degrees >= 0 else "-"
        abs_deg = abs(degrees)
        d = int(abs_deg)
        m = int((abs_deg - d) * 60)
        s = int(((abs_deg - d) * 60 - m) * 60)
        return f"{sign}{d:02d}°{m:02d}'{s:02d}\""
    
    return {
        'moon_alt_deg': moon_alt.degrees,
        'moon_az_deg': moon_az.degrees,
        'moon_ra_hours': moon_ra.hours,
        'moon_dec_deg': moon_dec.degrees,
        'moon_phase_percent': moon_phase_percent,
        'zenith_ra_hours': zenith_ra.hours,
        'zenith_dec_deg': zenith_dec.degrees,
        'zenith_ra_str': format_ra_hours(zenith_ra.hours),
        'zenith_dec_str': format_dec_degrees(zenith_dec.degrees)
    }


def get_extinction_coefficient(band: str = 'V') -> float:
    """
    Returns a typical atmospheric extinction coefficient for a given band.
    Value in magnitudes per airmass.
    """
    extinction_coeffs = {'U': 0.6, 'B': 0.4, 'V': 0.2, 'R': 0.1, 'I': 0.08}
    return extinction_coeffs.get(band.upper(), 0.2)


def calculate_airmass(altitude_deg):
    """
    Calculates airmass using a robust formula. 
    Returns inf for altitudes <= 0.
    Works with both scalar values and numpy arrays.
    """
    # Handle both scalar and array inputs
    if np.isscalar(altitude_deg):
        if altitude_deg <= 0:
            return np.inf
        zenith_angle_rad = np.radians(90.0 - altitude_deg)
        return 1.0 / np.cos(zenith_angle_rad)
    else:
        # Array input
        zenith_angle_rad = np.radians(90.0 - altitude_deg)
        airmass = 1.0 / np.cos(zenith_angle_rad)
        return np.where(altitude_deg <= 0, np.inf, airmass)


def generate_sky_quality_map(observer_location, time_utc: datetime, 
                               min_altitude_deg: float = 30.0,
                               light_pollution_mag: float = 21.0,
                               grid_resolution_deg: int = 5) -> Dict[str, Any]:
    """
    Generates a full-sky quality map for a given moment in time.

    This advanced function models:
    - Airmass and atmospheric extinction.
    - Sky brightness from the Moon.
    - A base level of light pollution.
    The goal is to find the patch of sky with the highest observational quality.

    Args:
        observer_location: Skyfield topos object.
        time_utc: The specific moment to calculate the map for.
        min_altitude_deg: The minimum altitude to consider in the map.
        light_pollution_mag: The base sky brightness in mag/arcsec^2 (zenith, no moon).
                             (e.g., 22.0 for a pristine site, 18.0 for a city).
        grid_resolution_deg: The resolution of the sky grid in degrees.

    Returns:
        A dictionary containing the RA/Dec of the best patch and the full map data.
    """
    if not time_utc.tzinfo:
        time_utc = pytz.utc.localize(time_utc)
    
    t = ts.from_datetime(time_utc)
    observer = earth + observer_location
    conditions = calculate_sky_conditions_at_time(observer_location, time_utc)
    
    # 1. Create a grid of points in the sky (in Alt/Az coordinates)
    alt_range = np.arange(min_altitude_deg, 90, grid_resolution_deg)
    az_range = np.arange(0, 360, grid_resolution_deg)
    az_grid, alt_grid = np.meshgrid(az_range, alt_range)
    
    # 2. Model the total sky brightness at each grid point
    
    # Start with the base brightness from light pollution
    sky_brightness = np.full(alt_grid.shape, light_pollution_mag)
    
    # Add extinction effect: sky appears darker away from the zenith
    airmass = np.where(alt_grid <= 0, np.inf, 1.0 / np.cos(np.radians(90.0 - alt_grid)))
    extinction_v = get_extinction_coefficient('V')
    sky_brightness += extinction_v * (airmass - 1)
    
    # Add moon brightness contribution
    if conditions['moon_alt_deg'] > 0:
        moon_alt_rad = np.radians(conditions['moon_alt_deg'])
        moon_az_rad = np.radians(conditions['moon_az_deg'])
        alt_rad = np.radians(alt_grid)
        az_rad = np.radians(az_grid)
        
        # Spherical law of cosines for angular separation from moon
        cos_sep = np.sin(moon_alt_rad) * np.sin(alt_rad) + \
                  np.cos(moon_alt_rad) * np.cos(alt_rad) * np.cos(moon_az_rad - az_rad)
        cos_sep = np.clip(cos_sep, -1.0, 1.0)
        sep_from_moon_deg = np.degrees(np.arccos(cos_sep))
        
        # A simplified scattering model (e.g., from Krisciunas & Schaefer 1991)
        # This models how the moon's light scatters through the atmosphere.
        phase = conditions['moon_phase_percent']
        
        # Protect against numerical issues
        with np.errstate(divide='ignore', invalid='ignore'):
            moon_brightening = (
                10**(-0.4 * (3.84 + 0.026 * phase + 4e-9 * phase**4)) *
                (0.631 * (1.06 + np.cos(np.radians(sep_from_moon_deg))**2) + 10**(5.36 - sep_from_moon_deg/40.0))
            )
            moon_contribution_mag = -2.5 * np.log10(moon_brightening)
            
            # Handle potential infinities or NaNs
            moon_contribution_mag = np.where(np.isfinite(moon_contribution_mag), 
                                           moon_contribution_mag, 
                                           sky_brightness)
        
        # Combine fluxes (magnitudes are logarithmic, so we must convert to linear flux, add, and convert back)
        with np.errstate(divide='ignore', invalid='ignore'):
            sky_flux = 10**(-0.4 * sky_brightness) + 10**(-0.4 * moon_contribution_mag)
            combined_brightness = -2.5 * np.log10(sky_flux)
            
            # Use combined brightness where valid, otherwise use original sky brightness
            sky_brightness = np.where(np.isfinite(combined_brightness), 
                                    combined_brightness, 
                                    sky_brightness)
        
    # 3. Calculate a final "Observation Quality" score
    # We want a dark sky (high sky_brightness value) and low airmass (high altitude).
    # A good proxy for quality is proportional to (1 / sky_background_flux) / airmass.
    # Higher score is better.
    with np.errstate(divide='ignore', invalid='ignore'):
        quality_score = (10**(0.4 * sky_brightness)) / airmass
        
        # Set invalid values to 0 (they won't be selected as best)
        quality_score = np.where(np.isfinite(quality_score), quality_score, 0)
    
    # Find the grid point with the maximum quality score
    best_idx = np.unravel_index(np.argmax(quality_score), quality_score.shape)
    best_alt, best_az = alt_grid[best_idx], az_grid[best_idx]
    
    # Convert the best Alt/Az back to RA/Dec for astronomical targeting
    best_patch_apparent = observer.at(t).from_altaz(alt_degrees=float(best_alt), az_degrees=float(best_az))
    best_ra_obj, best_dec_obj, _ = best_patch_apparent.radec()
    
    return {
        'best_ra_hours': best_ra_obj.hours,
        'best_dec_deg': best_dec_obj.degrees,
        'best_alt_deg': float(best_alt),
        'best_az_deg': float(best_az),
        'best_quality_score': float(np.max(quality_score)),
        'sky_map_data': {
            'alt_grid': alt_grid,
            'az_grid': az_grid,
            'quality_map': quality_score,
            'brightness_map': sky_brightness
        }
    }