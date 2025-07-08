import ephem
import datetime
from datetime import timedelta
import pytz
from typing import Dict, Tuple, Optional, Any
import numpy as np

def calculate_sun_moon_info(observer: ephem.Observer, obs_date: datetime.datetime, 
                          timezone: str = 'UTC') -> Dict[str, Any]:
    """
    Calculate sun and moon information for the given observer and date.
    
    Args:
        observer: PyEphem observer object
        obs_date: Observation date
        timezone: Timezone string
        
    Returns:
        Dictionary with sun/moon rise/set times and moon phase info
    """
    # Set observer date
    observer.date = obs_date
    
    # Create celestial objects
    sun = ephem.Sun()
    moon = ephem.Moon()
    
    # Get timezone
    local_tz = pytz.timezone(timezone)
    
    # Calculate sun times
    try:
        sunset = observer.next_setting(sun)
        sunrise = observer.next_rising(sun)
        
        # Adjust if needed
        if sunrise < sunset:
            sunset = observer.previous_setting(sun)
            
        # Convert to local time
        sunset_local = pytz.utc.localize(sunset.datetime()).astimezone(local_tz)
        sunrise_local = pytz.utc.localize(sunrise.datetime()).astimezone(local_tz)
        
        # Calculate midnight
        if sunrise_local.date() > sunset_local.date():
            midnight = sunset_local + (sunrise_local - sunset_local) / 2
        else:
            midnight = sunset_local + timedelta(hours=12)
            
    except ephem.CircumpolarError:
        sunset_local = None
        sunrise_local = None
        midnight = obs_date + timedelta(hours=12)
    
    # Calculate moon times
    try:
        moonrise = observer.next_rising(moon)
        moonset = observer.next_setting(moon)
        
        if moonrise < moonset:
            moonset = observer.previous_setting(moon)
            
        moonrise_local = pytz.utc.localize(moonrise.datetime()).astimezone(local_tz)
        moonset_local = pytz.utc.localize(moonset.datetime()).astimezone(local_tz)
        
    except ephem.CircumpolarError:
        moonrise_local = None
        moonset_local = None
    
    # Calculate moon phase at midnight
    if midnight:
        observer.date = midnight
        moon.compute(observer)
        moon_phase = moon.phase
        moon_alt = float(moon.alt) * 180 / 3.14159
        moon_az = float(moon.az) * 180 / 3.14159
    else:
        moon_phase = 0
        moon_alt = 0
        moon_az = 0
    
    return {
        'sunset': sunset_local,
        'sunrise': sunrise_local,
        'midnight': midnight,
        'moonrise': moonrise_local,
        'moonset': moonset_local,
        'moon_phase': moon_phase,
        'moon_alt': moon_alt,
        'moon_az': moon_az
    }

def calculate_optimal_region(observer: ephem.Observer, obs_date: datetime.datetime,
                           min_altitude: float = 40.0) -> Dict[str, Tuple[float, float]]:
    """
    Calculate optimal observation region based on moon position and zenith.
    
    Args:
        observer: PyEphem observer object
        obs_date: Observation date
        min_altitude: Minimum altitude in degrees (default: 40°)
        
    Returns:
        Dictionary with 'ra_range' and 'dec_range' as tuples of (min, max) values
    """
    # Calculate midnight
    observer.date = obs_date
    sun = ephem.Sun()
    
    try:
        sunset = observer.next_setting(sun)
        sunrise = observer.next_rising(sun)
        if sunrise < sunset:
            sunset = observer.previous_setting(sun)
        midnight = sunset + (sunrise - sunset) / 2
    except ephem.CircumpolarError:
        midnight = ephem.Date(obs_date + timedelta(hours=12))
    
    # Set observer to midnight
    observer.date = midnight
    
    # Calculate moon position
    moon = ephem.Moon()
    moon.compute(observer)
    moon_visible = float(moon.alt) > 0
    
    # Calculate zenith
    zenith_ra = float(observer.sidereal_time())  # in radians
    zenith_dec = float(observer.lat)  # in radians
    
    # Calculate declination range
    min_altitude_rad = min_altitude * 3.14159 / 180
    dec_min_rad = zenith_dec - min_altitude_rad
    dec_max_rad = min(zenith_dec, 3.14159/2)  # Max 90°
    
    # Calculate RA range
    if moon_visible:
        # Moon visible: observe opposite side
        moon_ra = float(moon.ra)
        ra_opposite = moon_ra + 3.14159  # 180° opposite
        if ra_opposite > 2 * 3.14159:
            ra_opposite -= 2 * 3.14159
        
        # ±3 hours around opposite point
        ra_range_rad = 3 * 3.14159 / 12
        ra_min_rad = ra_opposite - ra_range_rad
        ra_max_rad = ra_opposite + ra_range_rad
    else:
        # Moon not visible: use zenith region
        ra_range_rad = 3 * 3.14159 / 12
        ra_min_rad = zenith_ra - ra_range_rad
        ra_max_rad = zenith_ra + ra_range_rad
    
    # Wrap RA values to [0, 2π]
    if ra_min_rad < 0:
        ra_min_rad += 2 * 3.14159
    if ra_max_rad > 2 * 3.14159:
        ra_max_rad -= 2 * 3.14159
    
    # Convert to hours and degrees
    ra_min_hours = ra_min_rad * 12 / 3.14159
    ra_max_hours = ra_max_rad * 12 / 3.14159
    dec_min_deg = dec_min_rad * 180 / 3.14159
    dec_max_deg = dec_max_rad * 180 / 3.14159
    
    return {
        'ra_range': (ra_min_hours, ra_max_hours),
        'dec_range': (dec_min_deg, dec_max_deg),
        'moon_visible': moon_visible,
        'strategy': 'opposite_moon' if moon_visible else 'zenith'
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