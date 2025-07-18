import numpy as np
import logging
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, Tuple

from skyfield.api import load, wgs84
from skyfield.almanac import find_discrete, risings_and_settings
from skyfield import almanac

# Centralized configuration imports for scientific consistency
from ..config import (
    # Grid resolution parameters
    DEFAULT_GRID_RESOLUTION_ARCMIN,
    FINE_GRID_RESOLUTION_ARCMIN,
    COARSE_GRID_RESOLUTION_ARCMIN,
    # Extinction coefficients (La Silla Observatory standards)
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
    # Observational limits
    MIN_OBSERVABLE_ALTITUDE_DEG,
    OPTIMAL_MIN_ALTITUDE_DEG,
    ZENITH_AVOIDANCE_ZONE_DEG,
    # Airmass parameters
    MAX_AIRMASS_FOR_PHOTOMETRY,
    MAX_AIRMASS_FOR_SPECTROSCOPY,
    AIRMASS_WARNING_THRESHOLD,
    # Lunar model parameters (Krisciunas & Schaefer 1991)
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
    # Quality metrics
    SKY_QUALITY_WEIGHT_BRIGHTNESS,
    SKY_QUALITY_WEIGHT_AIRMASS,
    MIN_QUALITY_SCORE_THRESHOLD,
    # Validation ranges
    MIN_OBSERVATORY_LATITUDE_DEG,
    MAX_OBSERVATORY_LATITUDE_DEG,
    MIN_OBSERVATORY_LONGITUDE_DEG,
    MAX_OBSERVATORY_LONGITUDE_DEG,
    MIN_OBSERVATORY_ALTITUDE_M,
    MAX_OBSERVATORY_ALTITUDE_M
)

# --- Global objects for Skyfield ---
# These are loaded once when the module is imported for efficiency.
# Skyfield will download the necessary ephemeris data on the first run.
ts = load.timescale()
eph = load('de421.bsp')
earth = eph['earth']
sun = eph['sun']
moon = eph['moon']

# Configure scientific logging
logger = logging.getLogger(__name__)


def get_observer_location(latitude_deg: float, longitude_deg: float, altitude_m: float):
    """
    Creates a Skyfield geographic location object (topos) with validation.

    Args:
        latitude_deg: Latitude in decimal degrees (+N, -S).
        longitude_deg: Longitude in decimal degrees (+E, -W).
        altitude_m: Altitude in meters above sea level.

    Returns:
        A Skyfield topos object representing the observer's location.
        
    Raises:
        ValueError: If coordinates are outside valid ranges for Earth-based observatories.
        
    Notes:
        Validation ranges based on practical limits for ground-based astronomy.
        Coordinates must be within Earth's surface and reasonable altitude limits.
    """
    # Scientific validation of observatory coordinates
    if not (MIN_OBSERVATORY_LATITUDE_DEG <= latitude_deg <= MAX_OBSERVATORY_LATITUDE_DEG):
        raise ValueError(f"Latitude {latitude_deg}° outside valid range [{MIN_OBSERVATORY_LATITUDE_DEG}°, {MAX_OBSERVATORY_LATITUDE_DEG}°]")
    
    if not (MIN_OBSERVATORY_LONGITUDE_DEG <= longitude_deg <= MAX_OBSERVATORY_LONGITUDE_DEG):
        raise ValueError(f"Longitude {longitude_deg}° outside valid range [{MIN_OBSERVATORY_LONGITUDE_DEG}°, {MAX_OBSERVATORY_LONGITUDE_DEG}°]")
    
    if not (MIN_OBSERVATORY_ALTITUDE_M <= altitude_m <= MAX_OBSERVATORY_ALTITUDE_M):
        raise ValueError(f"Altitude {altitude_m}m outside valid range [{MIN_OBSERVATORY_ALTITUDE_M}m, {MAX_OBSERVATORY_ALTITUDE_M}m]")
    
    logger.debug(f"Creating observer location: lat={latitude_deg:.4f}°, lon={longitude_deg:.4f}°, alt={altitude_m}m")
    return wgs84.latlon(latitude_deg, longitude_deg, elevation_m=altitude_m)


def calculate_astronomical_midnight(observer_location, obs_date: datetime, timezone: str = 'UTC') -> Dict[str, Any]:
    """
    Calculates the astronomical midnight (when the Sun reaches its lowest point below the horizon).
    
    Args:
        observer_location: Skyfield topos object
        obs_date: Observation date (datetime object)
        timezone: Timezone string for local time conversion
        
    Returns:
        Dictionary with astronomical midnight times in UTC and local time
    """
    # Handle both string and pytz timezone objects
    if hasattr(timezone, 'zone'):
        # It's already a pytz timezone object
        local_tz = timezone
    else:
        # It's a string, convert to pytz timezone
        local_tz = pytz.timezone(timezone)
    
    # Ensure obs_date is a datetime object, not just a date
    if hasattr(obs_date, 'date'):
        # It's already a datetime
        obs_date_dt = obs_date
    else:
        # Convert date to datetime
        obs_date_dt = datetime.combine(obs_date, datetime.min.time())
    
    # Create a 48-hour window to ensure we capture sunset and sunrise
    t0_utc = datetime(obs_date_dt.year, obs_date_dt.month, obs_date_dt.day, 0, 0, 0)
    t2_utc = t0_utc + timedelta(days=2)
    
    # Find Sun's minimum altitude (astronomical midnight)
    observer = earth + observer_location
    
    # Create time range with higher resolution for accurate minimum finding
    times = []
    altitudes = []
    
    # Sample every N minutes over 48 hours using centralized configuration
    current_time = t0_utc
    while current_time < t2_utc:
        utc_time = pytz.utc.localize(current_time)
        t = ts.from_datetime(utc_time)
        
        sun_apparent = observer.at(t).observe(sun).apparent()
        sun_alt, _, _ = sun_apparent.altaz()
        
        times.append(current_time)
        altitudes.append(sun_alt.degrees)
        
        current_time += timedelta(minutes=ASTRONOMICAL_MIDNIGHT_SAMPLING_MINUTES)
    
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
    This version uses the robust `dark_twilight_day` almanac function,
    inspired by the previously working implementation.
    
    Args:
        observer_location: Skyfield topos object
        obs_date: Observation date (datetime object or date object)
        timezone: Timezone string for local time conversion
    """
    # Handle timezone parameter - it can be a string or a timezone object
    if isinstance(timezone, str):
        local_tz = pytz.timezone(timezone)
    else:
        local_tz = timezone
    
    # Ensure obs_date is a datetime object, not just a date
    if hasattr(obs_date, 'date'):
        # It's already a datetime
        obs_date_dt = obs_date
    else:
        # Convert date to datetime
        obs_date_dt = datetime.combine(obs_date, datetime.min.time())
    
    # 1. Define a wider time window to ensure we capture all events
    # Start from 6 AM on the observation date to 6 PM the next day
    t0_dt_naive = datetime(obs_date_dt.year, obs_date_dt.month, obs_date_dt.day, 6, 0, 0)
    t1_dt_naive = t0_dt_naive + timedelta(days=1, hours=12)
    
    # Use the modern, standard way to make datetimes timezone-aware
    t0 = ts.from_datetime(t0_dt_naive.replace(tzinfo=pytz.utc))
    t1 = ts.from_datetime(t1_dt_naive.replace(tzinfo=pytz.utc))

    # 2. Use the efficient `dark_twilight_day` function to get all sun-related events at once
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
    
    # Import unified coordinate formatting function
    from ..utils.io import format_coordinates_astropy
    
    return {
        'moon_alt_deg': moon_alt.degrees,
        'moon_az_deg': moon_az.degrees,
        'moon_ra_hours': moon_ra.hours,
        'moon_dec_deg': moon_dec.degrees,
        'moon_phase_percent': moon_phase_percent,
        'zenith_ra_hours': zenith_ra.hours,
        'zenith_dec_deg': zenith_dec.degrees,
        'zenith_ra_str': format_coordinates_astropy(zenith_ra, zenith_dec)[0],
        'zenith_dec_str': format_coordinates_astropy(zenith_ra, zenith_dec)[1]
    }


def get_extinction_coefficient(band: str = 'V') -> float:
    """
    Returns atmospheric extinction coefficient for specified photometric band.
    
    Based on measurements from La Silla Observatory (ESO) as reported in
    Burki et al. (1995), A&AS, 112, 383. Values represent typical extinction
    in magnitudes per airmass for a good astronomical site.
    
    Args:
        band: Photometric band designation (U, B, V, R, I)
        
    Returns:
        Extinction coefficient in magnitudes per airmass
        
    Notes:
        - Values are for typical atmospheric conditions at a good site
        - For site-specific work, local extinction measurements should be used
        - Default V-band value used for unknown/unsupported bands
        
    References:
        Burki et al. (1995), "The atmospheric extinction at the ESO La Silla 
        Observatory", A&AS, 112, 383
    """
    extinction_coefficients = {
        'U': EXTINCTION_COEFFICIENT_U,
        'B': EXTINCTION_COEFFICIENT_B, 
        'V': EXTINCTION_COEFFICIENT_V,
        'R': EXTINCTION_COEFFICIENT_R,
        'I': EXTINCTION_COEFFICIENT_I
    }
    
    coefficient = extinction_coefficients.get(band.upper(), DEFAULT_EXTINCTION_COEFFICIENT)
    
    if band.upper() not in extinction_coefficients:
        logger.warning(f"Unknown photometric band '{band}', using V-band default: {DEFAULT_EXTINCTION_COEFFICIENT}")
    
    return coefficient


def calculate_airmass(altitude_deg):
    """
    Calculates airmass using the plane-parallel approximation.
    
    Uses the simple sec(z) formula for altitudes above the horizon.
    Returns infinity for altitudes at or below 0 degrees, representing
    targets that are not observable.
    
    Args:
        altitude_deg: Altitude angle in degrees (scalar or array)
        
    Returns:
        Airmass value(s) - float for scalar input, ndarray for array input
        
    Notes:
        - Valid for altitudes > ~20° where plane-parallel approximation holds
        - For more accurate calculations at low altitudes, consider Pickering's formula
        - Returns np.inf for targets below horizon (altitude <= 0°)
        
    Scientific Context:
        Airmass quantifies atmospheric extinction path length. At zenith (90°),
        airmass = 1.0. At 60° altitude, airmass ≈ 2.0. Professional observations
        typically avoid airmass > 2.5 due to systematic errors.
    """
    # Handle both scalar and array inputs
    if np.isscalar(altitude_deg):
        if altitude_deg <= 0:
            return np.inf
        zenith_angle_rad = np.radians(90.0 - altitude_deg)
        airmass = 1.0 / np.cos(zenith_angle_rad)
        
        # Issue warnings for high airmass values
        if airmass > AIRMASS_WARNING_THRESHOLD:
            logger.warning(f"High airmass detected: {airmass:.2f} at altitude {altitude_deg:.1f}°")
        
        return airmass
    else:
        # Array input
        zenith_angle_rad = np.radians(90.0 - altitude_deg)
        airmass = 1.0 / np.cos(zenith_angle_rad)
        airmass = np.where(altitude_deg <= 0, np.inf, airmass)
        
        # Count high airmass values for logging
        high_airmass_count = np.sum((airmass > AIRMASS_WARNING_THRESHOLD) & np.isfinite(airmass))
        if high_airmass_count > 0:
            logger.warning(f"{high_airmass_count} grid points have airmass > {AIRMASS_WARNING_THRESHOLD}")
        
        return airmass

def generate_sky_quality_map(observer_location, time_utc: datetime, 
                               min_altitude_deg: float = None,
                               sky_brightness_mag_arcsec2: float = None,
                               grid_resolution_arcmin: int = None) -> Dict[str, Any]:
    """
    Generates a comprehensive sky quality map for observatory planning.

    Implements a scientifically rigorous model combining atmospheric extinction,
    lunar contamination (Krisciunas & Schaefer 1991), and observational quality
    metrics to identify optimal sky regions for astronomical observations.

    Args:
        observer_location: Skyfield topos object representing observatory
        time_utc: Specific observation time (datetime with timezone)
        min_altitude_deg: Minimum altitude for consideration (default: professional standard)
        sky_brightness_mag_arcsec2: Base sky brightness in V mag/arcsec² (default: excellent site)
        grid_resolution_arcmin: Sky grid resolution in arcminutes (default: professional standard)

    Returns:
        Dictionary containing optimal coordinates and complete sky quality data:
        - best_ra_hours: Right Ascension of optimal patch (hours)
        - best_dec_deg: Declination of optimal patch (degrees)  
        - best_alt_deg: Altitude of optimal patch (degrees)
        - best_az_deg: Azimuth of optimal patch (degrees)
        - best_quality_score: Maximum quality score achieved
        - sky_map_data: Complete grid data for visualization
        
    Scientific Method:
        1. Model atmospheric extinction using site-specific coefficients
        2. Calculate lunar sky brightness contribution via Krisciunas & Schaefer (1991)
        3. Combine extinction and scattering using proper flux arithmetic
        4. Weight brightness and airmass according to observational priorities
        5. Identify optimal region maximizing signal-to-noise ratio
        
    References:
        - Krisciunas, K. & Schaefer, B.E. (1991), PASP, 103, 1033
        - Garstang, R.H. (1989), PASP, 101, 306
        - Burki et al. (1995), A&AS, 112, 383
    """
    # Apply scientific defaults from centralized configuration
    if min_altitude_deg is None:
        min_altitude_deg = OPTIMAL_MIN_ALTITUDE_DEG
    if sky_brightness_mag_arcsec2 is None:
        sky_brightness_mag_arcsec2 = EXCELLENT_SKY_BRIGHTNESS_V_MAG_ARCSEC2
    if grid_resolution_arcmin is None:
        grid_resolution_arcmin = DEFAULT_GRID_RESOLUTION_ARCMIN
        
    # Convert resolution to degrees for calculations
    grid_resolution_deg = grid_resolution_arcmin / 60.0
    
    # Validate inputs
    if not (MIN_OBSERVABLE_ALTITUDE_DEG <= min_altitude_deg <= 90.0):
        raise ValueError(f"Minimum altitude {min_altitude_deg}° outside valid range")
    
    if not (15.0 <= sky_brightness_mag_arcsec2 <= 23.0):
        raise ValueError(f"Sky brightness {sky_brightness_mag_arcsec2} mag/arcsec² outside realistic range")
    
    logger.debug(f"Generating sky quality map: alt_min={min_altitude_deg}°, "
                f"sky_brightness={sky_brightness_mag_arcsec2:.1f} mag/arcsec², "
                f"resolution={grid_resolution_arcmin}′")
    
    if not time_utc.tzinfo:
        time_utc = pytz.utc.localize(time_utc)
    
    t = ts.from_datetime(time_utc)
    observer = earth + observer_location
    conditions = calculate_sky_conditions_at_time(observer_location, time_utc)
    
    # 1. Create sky grid in Alt/Az coordinates
    alt_range = np.arange(min_altitude_deg, 90, grid_resolution_deg)
    az_range = np.arange(0, 360, grid_resolution_deg)
    az_grid, alt_grid = np.meshgrid(az_range, alt_range)
    
    # 2. Model sky brightness at each grid point
    
    # Base sky brightness from natural + light pollution sources
    sky_brightness = np.full(alt_grid.shape, sky_brightness_mag_arcsec2)
    
    # Apply atmospheric extinction using scientifically validated coefficients
    airmass = calculate_airmass(alt_grid)
    extinction_v = get_extinction_coefficient('V')
    sky_brightness += extinction_v * (airmass - 1)
    
    # 3. Add lunar contamination using Krisciunas & Schaefer (1991) model
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
        
        # Protect against numerical overflow/underflow
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
    
    # 4. Calculate observational quality metric
    # Scientific weighting based on signal-to-noise considerations
    with np.errstate(divide='ignore', invalid='ignore'):
        # For sky brightness: lower values (brighter sky) = worse conditions
        # Invert sky brightness for quality calculation (darker = better)
        sky_quality = 1.0 / (10**(-0.4 * sky_brightness))  # Higher for darker skies
        
        # For airmass: lower values = better atmospheric transmission
        airmass_quality = 1.0 / airmass  # Higher for lower airmass
        
        # Balanced combination prioritizing both dark sky and low airmass
        # Use multiplicative combination for more balanced weighting
        quality_score = sky_quality * airmass_quality
        
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
    
    # Log scientific summary
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