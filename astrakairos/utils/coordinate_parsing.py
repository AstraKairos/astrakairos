# astrakairos/utils/coordinate_parsing.py
"""
Coordinate parsing utilities for command line arguments.

This module provides robust parsing and validation functions for command line
arguments that require special handling beyond argparse's built-in capabilities.
"""

import logging
from typing import Tuple, Optional
from ..exceptions import ConfigurationError
from ..config import MIN_RA_DEG, MAX_RA_DEG, MIN_DEC_DEG, MAX_DEC_DEG

log = logging.getLogger(__name__)


def parse_coordinate_range(range_str: str, coordinate_type: str) -> Tuple[float, float]:
    """
    Parse and validate a coordinate range string for astronomical coordinates.
    
    Args:
        range_str: Comma-separated range string (e.g., "18.5,20.5")
        coordinate_type: Type of coordinate for validation ('ra' or 'dec')
        
    Returns:
        Tuple of (min_value, max_value) as floats
        
    Raises:
        ConfigurationError: If format is invalid or values are out of bounds
        
    Examples:
        >>> parse_coordinate_range("18.5,20.5", "ra")
        (18.5, 20.5)
        >>> parse_coordinate_range("-30,15.5", "dec")
        (-30.0, 15.5)
    """
    if not range_str or not range_str.strip():
        raise ConfigurationError(f"Empty {coordinate_type.upper()} range provided")
    
    # Remove any whitespace and validate format
    clean_str = range_str.strip()
    if ',' not in clean_str:
        raise ConfigurationError(
            f"Invalid {coordinate_type.upper()} range format. "
            f"Expected 'min,max' (e.g., '18.5,20.5'), got '{range_str}'"
        )
    
    parts = clean_str.split(',')
    if len(parts) != 2:
        raise ConfigurationError(
            f"Invalid {coordinate_type.upper()} range format. "
            f"Expected exactly two values separated by comma, got {len(parts)} values"
        )
    
    try:
        min_val = float(parts[0].strip())
        max_val = float(parts[1].strip())
    except ValueError as e:
        raise ConfigurationError(
            f"Invalid numeric values in {coordinate_type.upper()} range '{range_str}'. "
            f"Both values must be valid numbers"
        ) from e
    
    # Validate coordinate bounds
    if coordinate_type.lower() == 'ra':
        # Right Ascension: use config constants (converted from degrees to hours)
        min_ra_hours = MIN_RA_DEG / 15.0  # Convert degrees to hours
        max_ra_hours = MAX_RA_DEG / 15.0  # Convert degrees to hours
        
        if not (min_ra_hours <= min_val <= max_ra_hours) or not (min_ra_hours <= max_val <= max_ra_hours):
            raise ConfigurationError(
                f"RA values must be between {min_ra_hours:.1f} and {max_ra_hours:.1f} hours. Got: {min_val}, {max_val}"
            )
        # Note: We don't require min_val < max_val for RA to handle wraparound cases
        
    elif coordinate_type.lower() == 'dec':
        # Declination: use config constants directly
        if not (MIN_DEC_DEG <= min_val <= MAX_DEC_DEG) or not (MIN_DEC_DEG <= max_val <= MAX_DEC_DEG):
            raise ConfigurationError(
                f"Dec values must be between {MIN_DEC_DEG:.1f} and +{MAX_DEC_DEG:.1f} degrees. Got: {min_val}, {max_val}"
            )
        if min_val > max_val:
            raise ConfigurationError(
                f"Dec minimum ({min_val}) cannot be greater than maximum ({max_val})"
            )
    else:
        raise ConfigurationError(f"Unknown coordinate type: {coordinate_type}")
    
    return min_val, max_val


def validate_spatial_filter_combination(ra_range: Optional[Tuple[float, float]], 
                                      dec_range: Optional[Tuple[float, float]]) -> None:
    """
    Validate that spatial filter combination makes astronomical sense.
    
    Args:
        ra_range: RA range tuple or None
        dec_range: Dec range tuple or None
        
    Raises:
        ConfigurationError: If the combination is invalid
    """
    if not ra_range and not dec_range:
        return  # No filtering is fine
    
    # Log the applied filters for transparency
    if ra_range and dec_range:
        ra_min, ra_max = ra_range
        dec_min, dec_max = dec_range
        
        # Check for extremely narrow ranges that might be user errors
        ra_width = ra_max - ra_min if ra_max >= ra_min else (24.0 - ra_min) + ra_max
        dec_width = dec_max - dec_min
        
        if ra_width < 0.1:  # Less than 6 arcminutes in RA
            log.warning(f"Very narrow RA range detected: {ra_width:.3f} hours. "
                       f"This may result in very few or no systems.")
        
        if dec_width < 0.1:  # Less than 6 arcminutes in Dec
            log.warning(f"Very narrow Dec range detected: {dec_width:.3f} degrees. "
                       f"This may result in very few or no systems.")
        
        log.info(f"Spatial filter: RA=[{ra_min:.2f}, {ra_max:.2f}]h, "
                f"Dec=[{dec_min:.2f}, {dec_max:.2f}]°")
        
    elif ra_range:
        ra_min, ra_max = ra_range
        log.info(f"Spatial filter: RA=[{ra_min:.2f}, {ra_max:.2f}]h (all declinations)")
        
    elif dec_range:
        dec_min, dec_max = dec_range
        log.info(f"Spatial filter: Dec=[{dec_min:.2f}, {dec_max:.2f}]° (all right ascensions)")
