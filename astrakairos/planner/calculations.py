"""
Astronomical calculations for observation planning.

This module provides functions for calculating astronomical events,
sky quality mapping, and other calculations needed for observation planning.
"""

import numpy as np
import logging
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any

from skyfield.api import load, wgs84
from skyfield.almanac import find_discrete, risings_and_settings
from skyfield import almanac

from ..config import (
    # Grid resolution parameters
    DEFAULT_GRID_RESOLUTION_ARCMIN,
    FINE_GRID_RESOLUTION_ARCMIN,
    COARSE_GRID_RESOLUTION_ARCMIN,
    # Extinction coefficients
    EXTINCTION_COEFFICIENT_U,
    EXTINCTION_COEFFICIENT_B,
    EXTINCTION_COEFFICIENT_V,
    EXTINCTION_COEFFICIENT_R,
    EXTINCTION_COEFFICIENT_I,
    DEFAULT_EXTINCTION_COEFFICIENT,
    # Sky brightness standards
    PRISTINE_SKY_BRIGHTNESS_V_MAG_ARCSEC2,
    EXCELLENT_SKY_BRIGHTNESS_V_MAG_ARCSEC2,
    GOOD_SKY_BRIGHTNESS_V_MAG_ARCSEC2,
    MODERATE_SKY_BRIGHTNESS_V_MAG_ARCSEC2,
    POOR_SKY_BRIGHTNESS_V_MAG_ARCSEC2,
    # Observational limits
    MIN_OBSERVABLE_ALTITUDE_DEG,
    OPTIMAL_MIN_ALTITUDE_DEG,
    # Airmass parameters
    MAX_AIRMASS_FOR_PHOTOMETRY,
    MAX_AIRMASS_FOR_SPECTROSCOPY,
    AIRMASS_WARNING_THRESHOLD,
    # Lunar model parameters
    LUNAR_K_EXTINCTION,
    LUNAR_C1_COEFFICIENT,
    LUNAR_C2_COEFFICIENT,
    LUNAR_C3_COEFFICIENT,
    LUNAR_C4_COEFFICIENT,
    LUNAR_C5_COEFFICIENT,
    LUNAR_C6_COEFFICIENT,
    LUNAR_C7_COEFFICIENT,
    # Time sampling configuration
    NIGHTLY_EVENTS_SAMPLING_MINUTES,
    ASTRONOMICAL_MIDNIGHT_SAMPLING_MINUTES,
    MAX_EVENT_SEARCH_WINDOW_HOURS,
    # Quality metrics
    MIN_QUALITY_SCORE_THRESHOLD,
    SKY_QUALITY_WEIGHT_BRIGHTNESS,
    SKY_QUALITY_WEIGHT_AIRMASS,
    # Validation ranges
    MIN_OBSERVATORY_LATITUDE_DEG,
    MAX_OBSERVATORY_LATITUDE_DEG,
    MIN_OBSERVATORY_LONGITUDE_DEG,
    MAX_OBSERVATORY_LONGITUDE_DEG,
    MIN_OBSERVATORY_ALTITUDE_M,
    MAX_OBSERVATORY_ALTITUDE_M
)

# Initialize Skyfield objects
ts = load.timescale()
eph = load('de421.bsp')  # Jet Propulsion Laboratory Development Ephemeris
earth = eph['earth']
sun = eph['sun']
moon = eph['moon']

# Configure logging
logger = logging.getLogger(__name__)

# Module Constants
# Use constant from config instead of hardcoding
EVENT_SEARCH_WINDOW_HOURS = MAX_EVENT_SEARCH_WINDOW_HOURS  # Hours to search for nightly events


def get_observer_location(latitude_deg: float, longitude_deg: float, altitude_m: float):
    """Creates a Skyfield geographic location object (topos) with validation.

    Args:
        latitude_deg: Latitude in decimal degrees (+N, -S).
        longitude_deg: Longitude in decimal degrees (+E, -W).
        altitude_m: Altitude in meters above sea level.

    Returns:
        A Skyfield topos object representing the observer's location.
        
    Raises:
        ValueError: If coordinates are outside valid ranges for Earth-based observatories.
    """
    # Validate observatory coordinates against defined constants
    if not (MIN_OBSERVATORY_LATITUDE_DEG <= latitude_deg <= MAX_OBSERVATORY_LATITUDE_DEG):
        raise ValueError(f"Latitude {latitude_deg}° outside valid range [{MIN_OBSERVATORY_LATITUDE_DEG}°, {MAX_OBSERVATORY_LATITUDE_DEG}°]")
    
    if not (MIN_OBSERVATORY_LONGITUDE_DEG <= longitude_deg <= MAX_OBSERVATORY_LONGITUDE_DEG):
        raise ValueError(f"Longitude {longitude_deg}° outside valid range [{MIN_OBSERVATORY_LONGITUDE_DEG}°, {MAX_OBSERVATORY_LONGITUDE_DEG}°]")
    
    if not (MIN_OBSERVATORY_ALTITUDE_M <= altitude_m <= MAX_OBSERVATORY_ALTITUDE_M):
        raise ValueError(f"Altitude {altitude_m}m outside valid range [{MIN_OBSERVATORY_ALTITUDE_M}m, {MAX_OBSERVATORY_ALTITUDE_M}m]")
    
    logger.debug(f"Creating observer location: lat={latitude_deg:.4f}°, lon={longitude_deg:.4f}°, alt={altitude_m}m")
    return wgs84.latlon(latitude_deg, longitude_deg, elevation_m=altitude_m)


def calculate_astronomical_midnight(observer_location, obs_date: datetime, timezone: str = 'UTC') -> Dict[str, Any]:
    """Calculates the astronomical midnight when the sun reaches its lowest altitude.
    
    Uses efficient grid search to find the sun's minimum altitude during the night,
    which corresponds to astronomical midnight.
    
    Args:
        observer_location: Skyfield topos object.
        obs_date: Observation date (datetime or date object).
        timezone: Timezone string for local time conversion.
        
    Returns:
        Dictionary with astronomical midnight times in UTC and local time.
    """
    
    # Handle both string and pytz timezone objects
    local_tz = pytz.timezone(timezone) if isinstance(timezone, str) else timezone
    
    # Ensure obs_date is a datetime object for consistent calculations
    obs_date_dt = datetime.combine(obs_date, datetime.min.time()) if not isinstance(obs_date, datetime) else obs_date
    
    # Create search window for the night (sunset to sunrise roughly)
    t_start = datetime(obs_date_dt.year, obs_date_dt.month, obs_date_dt.day, 15, 0, 0)  # 3 PM
    t_end = t_start + timedelta(hours=18)  # Next day 9 AM
    
    # Convert to Skyfield time objects
    t0 = ts.from_datetime(pytz.utc.localize(t_start))
    t1 = ts.from_datetime(pytz.utc.localize(t_end))
    
    # Setup observer
    observer = earth + observer_location
    
    # Use efficient grid search to find minimum sun altitude (astronomical midnight)
    # Sample every 30 seconds for high precision
    times_array = ts.linspace(t0, t1, 720)  # 30-second resolution over 18 hours
    sun_positions = observer.at(times_array).observe(sun).apparent()
    sun_altitudes = sun_positions.altaz()[0].degrees
    
    # Find the minimum altitude (astronomical midnight)
    min_idx = np.argmin(sun_altitudes)
    astronomical_midnight_ts = times_array[min_idx]
    sun_altitude_at_midnight = sun_altitudes[min_idx]
    
    astronomical_midnight_utc = astronomical_midnight_ts.utc_datetime()
    
    # Convert to local time - ensure UTC timezone is set
    if astronomical_midnight_utc.tzinfo is None:
        astronomical_midnight_utc_tz = pytz.utc.localize(astronomical_midnight_utc)
    else:
        astronomical_midnight_utc_tz = astronomical_midnight_utc
    astronomical_midnight_local = astronomical_midnight_utc_tz.astimezone(local_tz)
    
    return {
        'astronomical_midnight_utc': astronomical_midnight_utc,
        'astronomical_midnight_local': astronomical_midnight_local,
        'sun_altitude_at_midnight': float(sun_altitude_at_midnight)
    }


def get_nightly_events(observer_location, obs_date: datetime, timezone: str = 'UTC') -> Dict[str, Any]:
    """Calculates all relevant astronomical events for a given observation night.
    
    Args:
        observer_location: Skyfield topos object.
        obs_date: Observation date (datetime or date object).
        timezone: Timezone string for local time conversion.
        
    Returns:
        Dictionary with all astronomical events including twilight times in both UTC and local time.
    """
    # Handle timezone parameter
    local_tz = pytz.timezone(timezone) if isinstance(timezone, str) else timezone
    
    # Ensure obs_date is a datetime object for consistent calculations
    obs_date_dt = datetime.combine(obs_date, datetime.min.time()) if not isinstance(obs_date, datetime) else obs_date
    
    # Define a time window to capture all events for the night of obs_date
    # Start at 6am on the observation date and extend to 6pm the next day (36 hours total)
    t0_dt_naive = datetime(obs_date_dt.year, obs_date_dt.month, obs_date_dt.day, 6, 0, 0)
    t1_dt_naive = t0_dt_naive + timedelta(days=1, hours=12)
    
    # Make datetimes timezone-aware
    t0 = ts.from_datetime(t0_dt_naive.replace(tzinfo=pytz.utc))
    t1 = ts.from_datetime(t1_dt_naive.replace(tzinfo=pytz.utc))

    # Use the efficient `dark_twilight_day` function to get all sun-related events at once
    f_sun = almanac.dark_twilight_day(eph, observer_location)
    sun_times, sun_events = find_discrete(t0, t1, f_sun)
    
    # Event codes for dark_twilight_day:
    # 4: Day -> 3: Civil Twilight (Sunset)
    # 3: Civil -> 2: Nautical
    # 2: Nautical -> 1: Astronomical
    # 1: Astronomical -> 0: Dark
    # 0: Dark -> 1: Astronomical (Morning)
    # 1: Astro -> 2: Nautical
    # 2: Nautical -> 3: Civil
    # 3: Civil -> 4: Day (Sunrise)

    # 3. Parse the event transitions to find specific times
    results = {}
    
    # Initialize all keys to None
    evening_events = ['sunset_utc', 'civil_twilight_end_utc', 'nautical_twilight_end_utc', 'astronomical_twilight_end_utc']
    morning_events = ['astronomical_twilight_start_utc', 'nautical_twilight_start_utc', 'civil_twilight_start_utc', 'sunrise_utc']
    
    for key in evening_events + morning_events:
        results[key] = None

    # Define the transitions we're looking for
    transitions = {
        (4, 3): 'sunset_utc',
        (3, 2): 'civil_twilight_end_utc',
        (2, 1): 'nautical_twilight_end_utc',
        (1, 0): 'astronomical_twilight_end_utc',
        (0, 1): 'astronomical_twilight_start_utc',
        (1, 2): 'nautical_twilight_start_utc',
        (2, 3): 'civil_twilight_start_utc',
        (3, 4): 'sunrise_utc'
    }

    # Process events chronologically to find the correct evening and morning events
    evening_found = {key: False for key in evening_events}
    morning_found = {key: False for key in morning_events}
    
    # Convert obs_date to date object for comparison
    obs_date_only = obs_date_dt.date()
    
    for i in range(len(sun_events) - 1):
        transition = (sun_events[i], sun_events[i+1])
        if transition in transitions:
            key = transitions[transition]
            event_time = sun_times[i+1].utc_datetime()
            
            # For evening events, take the first occurrence after noon on obs_date
            if key in evening_events and not evening_found[key]:
                # Check if this event happens in the evening of obs_date
                event_date = event_time.date()
                if event_date >= obs_date_only:
                    results[key] = event_time
                    evening_found[key] = True
            
            # For morning events, take the first occurrence after midnight following obs_date
            elif key in morning_events and not morning_found[key]:
                # Check if this event happens in the morning after obs_date
                event_date = event_time.date()
                if event_date > obs_date_only:
                    results[key] = event_time
                    morning_found[key] = True

    # 4. Calculate Moon events separately
    f_moon = risings_and_settings(eph, moon, observer_location)
    moon_times, moon_events = find_discrete(t0, t1, f_moon)
    
    def find_first_moon_event(code):
        for t, e in zip(moon_times, moon_events):
            if e == code: return t.utc_datetime()
        return None

    results['moonrise_utc'] = find_first_moon_event(1) # Rise = 1
    results['moonset_utc'] = find_first_moon_event(0) # Set = 0

    # 5. Calculate Midnights
    if results.get('sunset_utc') and results.get('sunrise_utc'):
        results['temporal_midnight_utc'] = results['sunset_utc'] + (results['sunrise_utc'] - results['sunset_utc']) / 2
    else:
        results['temporal_midnight_utc'] = None
        
    results.update(calculate_astronomical_midnight(observer_location, obs_date_dt, timezone))

    # 6. Create local time versions for all UTC events found.
    for key, utc_val in list(results.items()):
        if key.endswith('_utc'):
            local_key = key.replace('_utc', '_local')
            if isinstance(utc_val, datetime):
                # Ensure UTC timezone is set before converting
                if utc_val.tzinfo is None:
                    utc_val = pytz.utc.localize(utc_val)
                results[local_key] = utc_val.astimezone(local_tz)
            else:
                results[local_key] = None
            
    return results


def calculate_sky_conditions_at_time(observer_location, time_utc: datetime) -> Dict[str, Any]:
    """Calculates moon and zenith properties for a specific moment in time.
    
    Args:
        observer_location: Skyfield topos object.
        time_utc: Time to calculate conditions for (datetime object).
        
    Returns:
        Dictionary with moon position, phase, and zenith coordinates.
    """
    # Ensure time has timezone
    time_utc = pytz.utc.localize(time_utc) if not time_utc.tzinfo else time_utc
        
    t = ts.from_datetime(time_utc)
    observer = earth + observer_location
    
    # Calculate all properties in a single observer.at() call to avoid redundant calculations
    observer_at_t = observer.at(t)
    
    # Moon properties
    moon_apparent = observer_at_t.observe(moon).apparent()
    moon_alt, moon_az, _ = moon_apparent.altaz()
    moon_ra, moon_dec, _ = moon_apparent.radec()
    
    # Calculate moon phase (requires sun position)
    moon_phase_percent = almanac.fraction_illuminated(eph, 'moon', t) * 100.0

    # Zenith properties
    zenith_apparent = observer_at_t.from_altaz(alt_degrees=90, az_degrees=0)
    zenith_ra, zenith_dec, _ = zenith_apparent.radec()
    
    # Format coordinates for API compatibility using existing astropy function
    # Create separate coordinate objects for individual formatting
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    zenith_coord = SkyCoord(ra=zenith_ra.hours * u.hourangle, dec=zenith_dec.degrees * u.deg, frame='icrs')
    zenith_ra_str = zenith_coord.ra.to_string(unit=u.hourangle, sep=' ', precision=2, pad=True)
    zenith_dec_str = zenith_coord.dec.to_string(sep=' ', precision=2, pad=True, alwayssign=True)
    
    # Return a complete set of sky conditions
    return {
        'moon_alt_deg': moon_alt.degrees,
        'moon_az_deg': moon_az.degrees,
        'moon_ra_hours': moon_ra.hours,
        'moon_dec_deg': moon_dec.degrees,
        'moon_phase_percent': moon_phase_percent,
        'zenith_ra_hours': zenith_ra.hours,
        'zenith_dec_deg': zenith_dec.degrees,
        'zenith_ra_str': zenith_ra_str,
        'zenith_dec_str': zenith_dec_str
    }


def get_extinction_coefficient(band: str = 'V') -> float:
    """Returns atmospheric extinction coefficient for a photometric band.
    
    Args:
        band: Photometric band designation (U, B, V, R, I).
        
    Returns:
        Extinction coefficient in magnitudes per airmass.
    """
    # Using consistent all-uppercase for dictionary lookup
    band_upper = band.upper()
    
    # Dictionary of standard extinction coefficients from config
    extinction_coefficients = {
        'U': EXTINCTION_COEFFICIENT_U,
        'B': EXTINCTION_COEFFICIENT_B, 
        'V': EXTINCTION_COEFFICIENT_V,
        'R': EXTINCTION_COEFFICIENT_R,
        'I': EXTINCTION_COEFFICIENT_I
    }
    
    # Get coefficient with default fallback
    coefficient = extinction_coefficients.get(band_upper, DEFAULT_EXTINCTION_COEFFICIENT)
    
    if band_upper not in extinction_coefficients:
        logger.warning(f"Unknown photometric band '{band}', using V-band default: {DEFAULT_EXTINCTION_COEFFICIENT}")
    
    return coefficient


def calculate_airmass(altitude_deg):
    """Calculates airmass using the plane-parallel approximation.
    
    Args:
        altitude_deg: Altitude angle in degrees (scalar or array).
        
    Returns:
        Airmass value(s). Returns np.inf for targets at or below the horizon.
    """
    # Efficient unified calculation for both scalar and array inputs
    with np.errstate(divide='ignore', invalid='ignore'):
        # Convert to zenith angle
        zenith_angle_rad = np.radians(90.0 - altitude_deg)
        
        # Calculate airmass, setting infinity for invalid values
        airmass = np.where(np.asarray(altitude_deg) > 0, 
                          1.0 / np.cos(zenith_angle_rad), 
                          np.inf)
        
        # Log warnings for high airmass values
        if np.isscalar(altitude_deg):
            if np.isfinite(airmass) and airmass > AIRMASS_WARNING_THRESHOLD:
                logger.warning(f"High airmass detected: {float(airmass):.2f} at altitude {float(altitude_deg):.1f}°")
        else:
            high_airmass_count = np.sum((airmass > AIRMASS_WARNING_THRESHOLD) & np.isfinite(airmass))
            if high_airmass_count > 0:
                logger.warning(f"{high_airmass_count} grid points have airmass > {AIRMASS_WARNING_THRESHOLD}")
    
        return airmass

def generate_sky_quality_map(observer_location, time_utc: datetime, 
                          min_altitude_deg: float = None,
                          sky_brightness_mag_arcsec2: float = None,
                          grid_resolution_arcmin: int = None) -> Dict[str, Any]:
    """Generates a sky quality map for observation planning.

    Args:
        observer_location: Skyfield topos object representing the observatory.
        time_utc: Specific observation time (datetime with timezone).
        min_altitude_deg: Minimum altitude for consideration.
        sky_brightness_mag_arcsec2: Base sky brightness in V mag/arcsec².
        grid_resolution_arcmin: Sky grid resolution in arcminutes.

    Returns:
        Dictionary containing optimal coordinates and complete sky quality data.
    """
    # Apply defaults from centralized configuration
    min_altitude_deg = OPTIMAL_MIN_ALTITUDE_DEG if min_altitude_deg is None else min_altitude_deg
    sky_brightness_mag_arcsec2 = EXCELLENT_SKY_BRIGHTNESS_V_MAG_ARCSEC2 if sky_brightness_mag_arcsec2 is None else sky_brightness_mag_arcsec2
    grid_resolution_arcmin = DEFAULT_GRID_RESOLUTION_ARCMIN if grid_resolution_arcmin is None else grid_resolution_arcmin
    
    # Convert resolution to degrees for calculations
    grid_resolution_deg = grid_resolution_arcmin / 60.0
    
    # Validate inputs
    if not (MIN_OBSERVABLE_ALTITUDE_DEG <= min_altitude_deg <= 90.0):
        raise ValueError(f"Minimum altitude {min_altitude_deg}° outside valid range [{MIN_OBSERVABLE_ALTITUDE_DEG}, 90.0]")
    
    if not (15.0 <= sky_brightness_mag_arcsec2 <= 23.0):
        raise ValueError(f"Sky brightness {sky_brightness_mag_arcsec2} mag/arcsec² outside realistic range [15.0, 23.0]")
    
    logger.debug(f"Generating sky quality map: alt_min={min_altitude_deg}°, "
                f"sky_brightness={sky_brightness_mag_arcsec2:.1f} mag/arcsec², "
                f"resolution={grid_resolution_arcmin}′")
    
    # Ensure time has timezone
    time_utc = pytz.utc.localize(time_utc) if not time_utc.tzinfo else time_utc
    
    t = ts.from_datetime(time_utc)
    observer = earth + observer_location
    conditions = calculate_sky_conditions_at_time(observer_location, time_utc)
    
    # Create optimized sky grid in Alt/Az coordinates
    alt_range = np.arange(min_altitude_deg, 90, grid_resolution_deg)
    az_range = np.arange(0, 360, grid_resolution_deg)
    az_grid, alt_grid = np.meshgrid(az_range, alt_range)
    
    # Model sky brightness with vectorized operations
    # Base sky brightness from natural + light pollution sources
    sky_brightness = np.full(alt_grid.shape, sky_brightness_mag_arcsec2)
    
    # Apply atmospheric extinction
    airmass = calculate_airmass(alt_grid)
    extinction_v = get_extinction_coefficient('V')
    sky_brightness += extinction_v * (airmass - 1)
    
    # Add lunar contamination using Krisciunas & Schaefer (1991) model
    if conditions['moon_alt_deg'] > 0:
        logger.debug(f"Applying lunar contamination model: moon at {conditions['moon_alt_deg']:.1f}° altitude")
        
        moon_alt_rad = np.radians(conditions['moon_alt_deg'])
        moon_az_rad = np.radians(conditions['moon_az_deg'])
        alt_rad = np.radians(alt_grid)
        az_rad = np.radians(az_grid)
        
        # Calculate angular separation from moon using spherical trigonometry
        cos_sep = (np.sin(moon_alt_rad) * np.sin(alt_rad) + 
                  np.cos(moon_alt_rad) * np.cos(alt_rad) * np.cos(moon_az_rad - az_rad))
        cos_sep = np.clip(cos_sep, -1.0, 1.0)
        sep_from_moon_deg = np.degrees(np.arccos(cos_sep))
        
        # Krisciunas & Schaefer (1991) lunar scattering model
        phase = conditions['moon_phase_percent']
        
        # Suppress numpy warnings for these specific calculations where out-of-bounds
        # values are expected (e.g., large angles) and handled by np.where.
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            # Phase-dependent lunar brightness term
            lunar_phase_term = (LUNAR_C1_COEFFICIENT + 
                              LUNAR_C2_COEFFICIENT * phase + 
                              LUNAR_C3_COEFFICIENT * phase**4)
            
            # Geometric scattering term
            geometric_term = (LUNAR_C4_COEFFICIENT * 
                            (LUNAR_C5_COEFFICIENT + np.cos(np.radians(sep_from_moon_deg))**2))
            
            # Distance-dependent scattering term  
            distance_term = 10**(LUNAR_C6_COEFFICIENT - sep_from_moon_deg / LUNAR_C7_COEFFICIENT)
            
            # Combined lunar brightness contribution
            moon_brightening = (10**(-0.4 * lunar_phase_term) * 
                              (geometric_term + distance_term))
            
            moon_contribution_mag = -2.5 * np.log10(moon_brightening)
            
            # Handle numerical issues by falling back to original sky brightness
            moon_contribution_mag = np.where(np.isfinite(moon_contribution_mag), 
                                           moon_contribution_mag, 
                                           sky_brightness)
        
        # Combine sky and lunar contributions using proper flux arithmetic
        with np.errstate(divide='ignore', invalid='ignore'):
            sky_flux = 10**(-0.4 * sky_brightness)
            lunar_flux = 10**(-0.4 * moon_contribution_mag)
            combined_flux = sky_flux + lunar_flux
            combined_brightness = -2.5 * np.log10(combined_flux)
            
            # Use combined brightness where valid, otherwise original sky brightness
            sky_brightness = np.where(np.isfinite(combined_brightness), 
                                    combined_brightness, 
                                    sky_brightness)
    
    # Calculate observational quality metric using proper physics
    with np.errstate(divide='ignore', invalid='ignore'):
        # Convert sky brightness to flux (linear scale)
        # Lower sky brightness magnitude = brighter sky = worse conditions
        # Higher sky brightness magnitude = darker sky = better conditions
        sky_flux = 10**(-0.4 * sky_brightness)
        
        # Invert flux so that darker skies (lower flux) give higher quality
        # Use the inverse so that lower flux (darker sky) = higher quality
        sky_quality = 1.0 / (sky_flux + 1e-10)  # Add small value to avoid division by zero
        
        # For airmass: lower values = better atmospheric transmission
        # Avoid division by infinity for below-horizon targets
        airmass_quality = np.where(np.isfinite(airmass), 1.0 / airmass, 0.0)
        
        # Normalize both qualities to [0,1] range for proper combination
        max_sky_quality = np.max(sky_quality[np.isfinite(sky_quality)])
        max_airmass_quality = np.max(airmass_quality[np.isfinite(airmass_quality)])
        
        normalized_sky_quality = sky_quality / max_sky_quality
        normalized_airmass_quality = airmass_quality / max_airmass_quality
        
        # Apply configured weights from config.py  
        weighted_sky_quality = normalized_sky_quality * SKY_QUALITY_WEIGHT_BRIGHTNESS
        weighted_airmass_quality = normalized_airmass_quality * SKY_QUALITY_WEIGHT_AIRMASS
        
        # Combine metrics using weighted sum
        quality_score = weighted_sky_quality + weighted_airmass_quality
        
        # Set invalid regions to zero (won't be selected as optimal)
        quality_score = np.where(np.isfinite(quality_score), quality_score, 0)
        
        # Apply minimum quality threshold
        quality_score = np.where(quality_score >= MIN_QUALITY_SCORE_THRESHOLD, 
                               quality_score, 0)
    
    # 5. Find optimal observing region
    best_idx = np.unravel_index(np.argmax(quality_score), quality_score.shape)
    best_alt, best_az = alt_grid[best_idx], az_grid[best_idx]
    best_quality = quality_score[best_idx]
    
    # Convert optimal Alt/Az to RA/Dec for telescope pointing
    best_patch_apparent = observer.at(t).from_altaz(alt_degrees=float(best_alt), 
                                                   az_degrees=float(best_az))
    best_ra_obj, best_dec_obj, _ = best_patch_apparent.radec()
    
    # Log summary
    logger.info(f"Optimal sky region: RA={best_ra_obj.hours:.2f}h, "
               f"Dec={best_dec_obj.degrees:.1f}°, Alt={best_alt:.1f}°, "
               f"Quality={best_quality:.3f}")
    
    return {
        'best_ra_hours': best_ra_obj.hours,
        'best_dec_deg': best_dec_obj.degrees,
        'best_alt_deg': float(best_alt),
        'best_az_deg': float(best_az),
        'best_quality_score': float(best_quality),
        'sky_map_data': {
            'alt_grid': alt_grid,
            'az_grid': az_grid,
            'quality_map': quality_score,
            'brightness_map': sky_brightness,
            'airmass_map': airmass
        },
        'model_parameters': {
            'min_altitude_deg': min_altitude_deg,
            'sky_brightness_mag_arcsec2': sky_brightness_mag_arcsec2,
            'grid_resolution_arcmin': grid_resolution_arcmin,
            'extinction_coefficient_v': extinction_v,
            'moon_included': conditions['moon_alt_deg'] > 0
        }
    }