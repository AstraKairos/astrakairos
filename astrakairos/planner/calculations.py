import ephem
import datetime
from datetime import timedelta
import pytz
from typing import Dict, Tuple, Optional, Any
import numpy as np

def calculate_sun_moon_info(observer: ephem.Observer, obs_date: datetime.datetime, 
                          timezone: str = 'UTC') -> Dict[str, Any]:
    """
    Calculate sun, moon, and zenith information for the given observer and date.
    
    Args:
        observer: PyEphem observer object
        obs_date: Observation date
        timezone: Timezone string
        
    Returns:
        Dictionary with sun/moon rise/set times (local and UTC), moon phase, and zenith coordinates.
    """
    observer.date = obs_date
    sun = ephem.Sun()
    moon = ephem.Moon()
    local_tz = pytz.timezone(timezone)

    results = {
        'sunset_local': None, 'sunset_utc': None, 
        'sunrise_local': None, 'sunrise_utc': None, 
        'midnight_local': None, 'midnight_utc': None,
        'moonrise_local': None, 'moonrise_utc': None, 
        'moonset_local': None, 'moonset_utc': None, 
        'moon_phase': 0, 'moon_alt': 0, 'moon_az': 0,
        'zenith_ra_str': 'N/A', 'zenith_dec_str': 'N/A'
    }

    # --- Calculate sun times and midnight ---
    try:
        sunset_ephem = observer.next_setting(sun)
        sunrise_ephem = observer.next_rising(sun)
        
        # Adjust if sunrise is on the next day
        if sunrise_ephem < sunset_ephem:
            sunset_ephem = observer.previous_setting(sun)
            
        # Get UTC datetime objects from ephem.Date
        sunset_utc_dt = sunset_ephem.datetime()
        sunrise_utc_dt = sunrise_ephem.datetime()

        results['sunset_utc'] = sunset_utc_dt
        results['sunrise_utc'] = sunrise_utc_dt
        
        # Convert to local time
        results['sunset_local'] = pytz.utc.localize(sunset_utc_dt).astimezone(local_tz)
        results['sunrise_local'] = pytz.utc.localize(sunrise_utc_dt).astimezone(local_tz)
        
        # Calculate midnight (midpoint of the night)
        if sunrise_utc_dt < sunset_utc_dt: # If sunrise is effectively on the next calendar day
             sunrise_utc_dt_adjusted = sunrise_utc_dt + timedelta(days=1)
        else:
             sunrise_utc_dt_adjusted = sunrise_utc_dt

        midnight_utc_dt = sunset_utc_dt + (sunrise_utc_dt_adjusted - sunset_utc_dt) / 2
        results['midnight_utc'] = midnight_utc_dt
        results['midnight_local'] = pytz.utc.localize(midnight_utc_dt).astimezone(local_tz)
        
    except ephem.CircumpolarError:
        # If sun is circumpolar, approximate midnight as 12 hours after the start of the day
        midnight_utc_dt = datetime.datetime.combine(obs_date, datetime.time(12, 0)) # noon of obs_date
        results['midnight_utc'] = midnight_utc_dt
        results['midnight_local'] = pytz.utc.localize(midnight_utc_dt).astimezone(local_tz)
        # sunset/sunrise remain None
    
    # --- Calculate Moon Info ---
    # Reset observer date to the calculated midnight for moon position and phase
    observer.date = ephem.Date(results['midnight_utc'])
    moon.compute(observer)
    results['moon_phase'] = moon.phase
    results['moon_alt'] = np.degrees(float(moon.alt))
    results['moon_az'] = np.degrees(float(moon.az))

    # Reset observer date to start of observation day for moon rise/set calculations
    observer.date = obs_date
    try:
        moonrise_ephem = observer.next_rising(moon)
        moonset_ephem = observer.next_setting(moon)
        
        # Adjust if moonset is on the next day
        if moonrise_ephem < moonset_ephem:
            moonset_ephem = observer.previous_setting(moon)

        moonrise_utc_dt = moonrise_ephem.datetime()
        moonset_utc_dt = moonset_ephem.datetime()

        results['moonrise_utc'] = moonrise_utc_dt
        results['moonset_utc'] = moonset_utc_dt
        results['moonrise_local'] = pytz.utc.localize(moonrise_utc_dt).astimezone(local_tz)
        results['moonset_local'] = pytz.utc.localize(moonset_utc_dt).astimezone(local_tz)
        
    except ephem.CircumpolarError:
        # moonrise/moonset remain None
        pass
    
    return results

def calculate_optimal_region(observer: ephem.Observer, obs_date: datetime.datetime,
                           min_altitude: float = 40.0) -> Dict[str, Any]:
    """
    Calculates the optimal observation region based on moon position and zenith.
    It also calculates and returns the formatted coordinates of the zenith at midnight.

    Args:
        observer: A PyEphem observer object, configured with lat/lon.
        obs_date: The starting date of the observation night.
        min_altitude: The minimum altitude below the zenith for the region, in degrees.

    Returns:
        A dictionary containing:
        - 'ra_range': (min_hours, max_hours) for the optimal RA region.
        - 'dec_range': (min_deg, max_deg) for the optimal Dec region.
        - 'moon_visible': A boolean indicating if the moon is up at midnight.
        - 'strategy': A string ('opposite_moon' or 'zenith') describing the logic used.
        - 'zenith_ra_str': Formatted RA of the zenith.
        - 'zenith_dec_str': Formatted Dec of the zenith.
    """
    # 1. Calculate the time of astronomical midnight for the given night
    observer.date = obs_date
    sun = ephem.Sun()
    
    try:
        sunset = observer.next_setting(sun)
        sunrise = observer.next_rising(sun)
        if sunrise < sunset:
            sunset = observer.previous_setting(sun)
        
        # Midnight is the midpoint between sunset and the next sunrise
        midnight_utc = sunset.datetime() + (sunrise.datetime() - sunset.datetime()) / 2
        midnight_ephem = ephem.Date(midnight_utc)

    except ephem.CircumpolarError:
        # For polar regions, approximate midnight as 12 hours after the start of the day
        midnight_ephem = ephem.Date(obs_date + timedelta(hours=12))
    
    # 2. Set the observer to the calculated midnight time
    observer.date = midnight_ephem
    
    # 3. Determine if the moon is a factor at midnight
    moon = ephem.Moon()
    moon.compute(observer)
    moon_visible = float(moon.alt) > 0
    
    # 4. Calculate the Zenith coordinates at midnight
    # The Zenith's RA is the local sidereal time.
    # The Zenith's Dec is the observer's latitude.
    zenith_ra_rad = float(observer.sidereal_time())
    zenith_dec_rad = float(observer.lat)
    
    # 5. Calculate the optimal declination range
    # This is a band from the zenith down to a minimum altitude.
    min_altitude_rad = np.radians(min_altitude)
    dec_min_rad = zenith_dec_rad - (np.pi/2 - min_altitude_rad) # Corrected logic: zenith is 90 deg alt
    dec_max_rad = zenith_dec_rad
    
    # Ensure Dec range is valid
    dec_min_rad = max(dec_min_rad, -np.pi/2) # Clamp at -90 deg
    dec_max_rad = min(dec_max_rad, np.pi/2)  # Clamp at +90 deg

    # 6. Calculate the optimal Right Ascension range based on strategy
    strategy = 'zenith'
    if moon_visible:
        # Strategy 1: Moon is up. Observe on the opposite side of the sky.
        strategy = 'opposite_moon'
        moon_ra_rad = float(moon.ra)
        ra_center_rad = moon_ra_rad + np.pi  # 180 degrees opposite
    else:
        # Strategy 2: No moon. Observe around the zenith.
        ra_center_rad = zenith_ra_rad

    # Define a search window of +/- 3 hours (45 degrees) around the center RA
    ra_window_rad = np.radians(45.0) # 3 hours * 15 deg/hour
    ra_min_rad = ra_center_rad - ra_window_rad
    ra_max_rad = ra_center_rad + ra_window_rad
    
    # 7. Format all results for the return dictionary

    # Format Zenith RA
    ra_hours = np.degrees(zenith_ra_rad) / 15.0
    ra_h = int(ra_hours)
    ra_m = int((ra_hours - ra_h) * 60)
    ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60
    zenith_ra_str = f"{ra_h:02d}h {ra_m:02d}m {ra_s:04.1f}s"
    
    # Format Zenith Dec
    dec_degrees = np.degrees(zenith_dec_rad)
    dec_sign = "+" if dec_degrees >= 0 else "-"
    dec_abs = abs(dec_degrees)
    dec_d = int(dec_abs)
    dec_m = int((dec_abs - dec_d) * 60)
    dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60
    zenith_dec_str = f"{dec_sign}{dec_d:02d}° {dec_m:02d}' {dec_s:04.1f}\""

    # Convert RA/Dec ranges to hours and degrees for the return value
    ra_min_hours = np.degrees(ra_min_rad % (2 * np.pi)) / 15.0
    ra_max_hours = np.degrees(ra_max_rad % (2 * np.pi)) / 15.0
    dec_min_deg = np.degrees(dec_min_rad)
    dec_max_deg = np.degrees(dec_max_rad)

    # Handle RA wrap-around for display (e.g., from 23h to 2h)
    if ra_min_hours > ra_max_hours:
        # This indicates the range crosses the 0h line.
        # The logic downstream should handle this, but for now we return the values.
        pass

    return {
        'ra_range': (ra_min_hours, ra_max_hours),
        'dec_range': (dec_min_deg, dec_max_deg),
        'moon_visible': moon_visible,
        'strategy': strategy,
        'zenith_ra_str': zenith_ra_str,
        'zenith_dec_str': zenith_dec_str
    }

def get_twilight_times(observer: ephem.Observer, obs_date: datetime.datetime, 
                      timezone: str = 'UTC') -> Dict[str, Any]:
    """
    Calculate civil, nautical, and astronomical twilight times.
    
    Args:
        observer: PyEphem observer object
        obs_date: Observation date
        timezone: Timezone string
        
    Returns:
        Dictionary with twilight times
    """
    # Set observer date
    observer.date = obs_date
    sun = ephem.Sun()
    local_tz = pytz.timezone(timezone)
    
    # Store original horizon
    original_horizon = observer.horizon
    
    results = {}
    
    # Civil twilight (-6°)
    observer.horizon = '-6'
    try:
        civil_end = observer.next_setting(sun, use_center=True)
        civil_start = observer.next_rising(sun, use_center=True)
        results['civil_twilight_end'] = pytz.utc.localize(civil_end.datetime()).astimezone(local_tz)
        results['civil_twilight_start'] = pytz.utc.localize(civil_start.datetime()).astimezone(local_tz)
    except ephem.CircumpolarError:
        results['civil_twilight_end'] = None
        results['civil_twilight_start'] = None
    
    # Nautical twilight (-12°)
    observer.horizon = '-12'
    try:
        nautical_end = observer.next_setting(sun, use_center=True)
        nautical_start = observer.next_rising(sun, use_center=True)
        results['nautical_twilight_end'] = pytz.utc.localize(nautical_end.datetime()).astimezone(local_tz)
        results['nautical_twilight_start'] = pytz.utc.localize(nautical_start.datetime()).astimezone(local_tz)
    except ephem.CircumpolarError:
        results['nautical_twilight_end'] = None
        results['nautical_twilight_start'] = None
    
    # Astronomical twilight (-18°)
    observer.horizon = '-18'
    try:
        astro_end = observer.next_setting(sun, use_center=True)
        astro_start = observer.next_rising(sun, use_center=True)
        results['astronomical_twilight_end'] = pytz.utc.localize(astro_end.datetime()).astimezone(local_tz)
        results['astronomical_twilight_start'] = pytz.utc.localize(astro_start.datetime()).astimezone(local_tz)
    except ephem.CircumpolarError:
        results['astronomical_twilight_end'] = None
        results['astronomical_twilight_start'] = None
    
    # Restore original horizon
    observer.horizon = original_horizon
    
    return results

def calculate_airmass(altitude_deg: float) -> float:
    """
    Calculate airmass for a given altitude.
    
    Args:
        altitude_deg: Altitude in degrees above horizon
        
    Returns:
        Airmass value
    """
    if altitude_deg <= 0:
        return float('inf')
    
    # Pickering (2002) formula
    altitude_rad = altitude_deg * 3.14159 / 180
    h = altitude_deg  # height in degrees
    
    airmass = 1 / (np.sin(altitude_rad) + 0.50572 * (h + 6.07995) ** -1.6364)
    
    return airmass