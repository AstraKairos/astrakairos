"""
Configuration constants for AstraKairos.

This module centralizes all configuration parameters used throughout the application,
making them easily configurable and maintainable.
"""

# Physical Validation Ranges
MIN_PERIOD_YEARS = 0.1
MAX_PERIOD_YEARS = 100000.0
MIN_SEMIMAJOR_AXIS_ARCSEC = 0.001
MAX_SEMIMAJOR_AXIS_ARCSEC = 100.0
MIN_ECCENTRICITY = 0.0
MAX_ECCENTRICITY = 0.99
MIN_INCLINATION_DEG = 0.0
MAX_INCLINATION_DEG = 180.0


AMBIGUOUS_P_VALUE_RATIO = 10  # Ratio for calculating ambiguous threshold from primary threshold

# Orbital Quality Grades (ORB6)
MIN_ORBIT_GRADE = 1  # Best quality
MAX_ORBIT_GRADE = 5  # Lowest quality

# Coordinate Validation
MIN_RA_DEG = 0.0
MAX_RA_DEG = 360.0
MIN_DEC_DEG = -90.0
MAX_DEC_DEG = 90.0

# WDS Designation Parsing
MIN_WDS_ID_LENGTH = 10
WDS_COORDINATE_PATTERN = r'^[0-9]{5}[+-][0-9]{4}([A-Z]{1,3})?$'

# Logging Configuration
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Validation Logging Control
ENABLE_VALIDATION_WARNINGS = True  # Set to False to reduce spam
VALIDATION_WARNING_SAMPLE_RATE = 0.01  # Log only 1% of validation warnings
ALLOW_SINGLE_EPOCH_SYSTEMS = True  # Allow systems with date_first == date_last

# I/O Configuration
CSV_SNIFFER_SAMPLE_SIZE = 2048  # bytes
DEFAULT_COORDINATE_PRECISION = 2  # decimal places for coordinate display

# Error Handling Configuration
COORDINATE_ERROR_BEHAVIOR = "return_invalid"  # Options: "raise", "return_none", "return_invalid"

# Observational Technique Uncertainty Model
# Based on literature estimates for typical measurement uncertainties
# References:
# - Micrometry: Heintz (1978), van de Kamp (1969)
# - Speckle: Hartkopf & Mason (2004), Tokovinin (1997)
# - Adaptive Optics: ten Brummelaar et al. (2005), Roberts Jr. et al. (2011)
# - Gaia: Lindegren et al. (2021), Brown et al. (2018)
# - CCD/Digital: Mason et al. (2001), Hartkopf et al. (2012)
TECHNIQUE_ERROR_MODEL = {
    # Traditional visual techniques
    'M': {'pa_error': 1.0, 'rho_error': 0.3},     # Micrometry: 1°, 0.3"
    'P': {'pa_error': 1.0, 'rho_error': 0.3},     # Photographic: 1°, 0.3" 
    'V': {'pa_error': 1.0, 'rho_error': 0.3},     # Visual: 1°, 0.3"
    
    # Speckle interferometry
    'S': {'pa_error': 0.5, 'rho_error': 0.03},    # Speckle: 0.5°, 0.03"
    'SP': {'pa_error': 0.5, 'rho_error': 0.03},   # Speckle
    'SPK': {'pa_error': 0.5, 'rho_error': 0.03},  # Speckle
    
    # Adaptive optics
    'A': {'pa_error': 0.2, 'rho_error': 0.005},   # Adaptive Optics: 0.2°, 0.005"
    'AO': {'pa_error': 0.2, 'rho_error': 0.005},  # Adaptive Optics
    'LO': {'pa_error': 0.2, 'rho_error': 0.005},  # Lucky imaging
    
    # Space-based astrometry
    'HG': {'pa_error': 0.1, 'rho_error': 0.001},  # Hipparcos/Gaia: 0.1°, 1 mas
    'H': {'pa_error': 0.1, 'rho_error': 0.001},   # Hipparcos
    'G': {'pa_error': 0.1, 'rho_error': 0.001},   # Gaia
    
    # CCD/Electronic detectors  
    'C': {'pa_error': 0.3, 'rho_error': 0.02},    # CCD: 0.3°, 0.02"
    'CCD': {'pa_error': 0.3, 'rho_error': 0.02},  # CCD
    'D': {'pa_error': 0.3, 'rho_error': 0.02},    # Digital
    
    # Interferometry
    'I': {'pa_error': 0.1, 'rho_error': 0.001},   # Interferometry: 0.1°, 1 mas
    'INT': {'pa_error': 0.1, 'rho_error': 0.001}, # Interferometry
    
    # Default for unknown techniques (conservative estimate)
    'DEFAULT': {'pa_error': 2.0, 'rho_error': 0.5}  # 2°, 0.5"
}

# WDSS Measurement Column Specifications
# Positions based on format analysis of WDSS catalogs
WDSS_MEASUREMENT_COLSPECS = {
    'wdss_id': (0, 14),
    'pair': (16, 23), 
    'epoch': (24, 34),
    'theta': (36, 43),
    'theta_error': (44, 51),
    'rho': (52, 61),
    'rho_error': (62, 69),
    'mag1': (72, 78),
    'mag2': (86, 92),
    'reference': (119, 127),
    'technique': (128, 130)
}

# CLI Analysis Configuration
MIN_MEASUREMENTS_FOR_CHARACTERIZE = 5
DEFAULT_MIN_OBS = 2
DEFAULT_MIN_OBSERVATIONS = 2  # Alias for compatibility
DEFAULT_MAX_OBS = 10
DEFAULT_CONCURRENT_REQUESTS = 20
DEFAULT_GAIA_P_VALUE = 0.01
DEFAULT_GAIA_RADIUS_FACTOR = 1.2
DEFAULT_GAIA_MIN_RADIUS = 2.0
DEFAULT_GAIA_MAX_RADIUS = 15.0
DEFAULT_SORT_BY = 'v_total'
TOP_RESULTS_DISPLAY_COUNT = 10

# Analysis modes
ANALYSIS_MODES = ['orbital', 'characterize', 'discovery']

# GUI Configuration
GUI_DEFAULT_WIDTH = 800
GUI_DEFAULT_HEIGHT = 800
DEFAULT_MIN_ALTITUDE_DEG = 40.0
DEFAULT_RA_WINDOW_HOURS = 3.0
DEFAULT_LIGHT_POLLUTION_MAG = 21.0

# GUI Parameter Validation Ranges
MIN_ALTITUDE_DEG = 0.0
MAX_ALTITUDE_DEG = 90.0
MIN_RA_WINDOW_HOURS = 0.5
MAX_RA_WINDOW_HOURS = 12.0
MIN_LIGHT_POLLUTION_MAG = 15.0
MAX_LIGHT_POLLUTION_MAG = 25.0

# Stelle Doppie Search Configuration
STELLE_DOPPIE_BASE_URL = "https://www.stelledoppie.it/index2.php"
STELLE_DOPPIE_SEARCH_METHODS = {
    'coordinate_range': 7,
    'magnitude_range': 7,
    'separation_range': 7,
    'exact_match': 1,
    'contains': 2,
    'starts_with': 3,
    'ends_with': 4
}

# Search Parameters
# Stelle Doppie search method mapping
STELLE_DOPPIE_METHODS = {
    1: 'equal to',
    2: 'not equal to', 
    3: 'more than',
    4: 'more or equal to',
    5: 'less than',
    6: 'less or equal to',
    7: 'between',
    8: 'not between',
    9: 'contains',
    17: 'void',
    18: 'not void'
}

# Filter configurations with method options
STELLE_DOPPIE_FILTERS = {
    'first': {
        'min_value': 1800.0,
        'max_value': 2030.0,
        'default_value': 2000.0,
        'default_method': '4',  # More or equal to
        'param_name': 'cat_wds-date_first',
        'label': 'First Observation Date',
        'unit': 'year',
        'data_type': 'numeric',
        'available_methods': ['1', '2', '3', '4', '5', '6', '7', '8', '17', '18']
    },
    'last': {
        'min_value': 1800.0,
        'max_value': 2030.0,
        'default_value': 2020.0,
        'default_method': '5',  # Less than
        'param_name': 'cat_wds-date_last',
        'label': 'Last Observation Date',
        'unit': 'year',
        'data_type': 'numeric',
        'available_methods': ['1', '2', '3', '4', '5', '6', '7', '8', '17', '18']
    },
    'mag_pri': {
        'min_value': -2.0,
        'max_value': 20.0,
        'default_value': 10.0,
        'default_method': '6',  # Less or equal to
        'param_name': 'cat_wds-mag_pri',
        'label': 'Primary Magnitude',
        'unit': 'mag',
        'data_type': 'numeric',
        'available_methods': ['1', '2', '3', '4', '5', '6', '7', '8', '17', '18']
    },
    'mag_sec': {
        'min_value': -2.0,
        'max_value': 20.0,
        'default_value': 12.0,
        'default_method': '6',  # Less or equal to
        'param_name': 'cat_wds-mag_sec',
        'label': 'Secondary Magnitude',
        'unit': 'mag',
        'data_type': 'numeric',
        'available_methods': ['1', '2', '3', '4', '5', '6', '7', '8', '17', '18']
    },
    'delta_magnitude': {
        'min_value': 0.0,
        'max_value': 15.0,
        'default_value': 3.0,
        'default_method': '6',  # Less or equal to
        'param_name': 'cat_wds-calc_delta_mag',
        'label': 'Delta Magnitude',
        'unit': 'mag',
        'data_type': 'numeric',
        'available_methods': ['1', '2', '3', '4', '5', '6', '7', '8', '17', '18']
    },
    'separation': {
        'min_value': 0.1,
        'max_value': 300.0,
        'default_value': 30.0,
        'default_method': '6',  # Less or equal to
        'param_name': 'cat_wds-sep_last',
        'label': 'Separation',
        'unit': 'arcsec',
        'data_type': 'numeric',
        'available_methods': ['1', '2', '3', '4', '5', '6', '7', '8', '17', '18']
    },
    'spectral_class': {
        'default_value': 'G',
        'default_method': '9',  # Contains
        'param_name': 'cat_wds-spectr',
        'label': 'Spectral Class',
        'unit': '',
        'data_type': 'string',
        'available_methods': ['1', '2', '9', '17', '18']  # String-appropriate methods
    },
    'observations': {
        'min_value': 1,
        'max_value': 999,
        'default_value': 5,
        'default_method': '4',  # More or equal to
        'param_name': 'cat_wds-obs',
        'label': 'Observations Count',
        'unit': 'count',
        'data_type': 'integer',
        'available_methods': ['1', '2', '3', '4', '5', '6', '7', '8', '17', '18']
    }
}

# Default Search Options
DEFAULT_SEARCH_OPTIONS = {
    'use_first_filter': False,
    'use_last_filter': False,
    'use_mag_pri_filter': False,
    'use_mag_sec_filter': False,
    'use_delta_magnitude_filter': False,
    'use_separation_filter': True,
    'use_spectral_class_filter': False,
    'use_observations_filter': False,
    'known_orbit': False,
    'physical_double': False,
    'uncertain_double': False
}

# Export Format Configuration
EXPORT_FORMATS = {
    'csv': {
        'name': 'CSV (Comma Separated Values)',
        'extension': '.csv',
        'mime_type': 'text/csv',
        'description': 'Standard CSV format compatible with Excel and databases'
    },
    'json': {
        'name': 'JSON (JavaScript Object Notation)',
        'extension': '.json',
        'mime_type': 'application/json',
        'description': 'Machine-readable JSON format for APIs and web applications'
    },
    'fits': {
        'name': 'FITS (Flexible Image Transport System)',
        'extension': '.fits',
        'mime_type': 'application/fits',
        'description': 'Astronomical standard format for tables and images'
    },
    'votable': {
        'name': 'VOTable (Virtual Observatory Table)',
        'extension': '.xml',
        'mime_type': 'application/x-votable+xml',
        'description': 'Virtual Observatory standard for astronomical data exchange'
    },
    'latex': {
        'name': 'LaTeX Table',
        'extension': '.tex',
        'mime_type': 'application/x-latex',
        'description': 'LaTeX format for scientific publications'
    }
}

# Multi-Catalog Integration
CATALOG_SOURCES = {
    'stelle_doppie': {
        'name': 'Stelle Doppie',
        'description': 'Online grouping of binary star catalogs',
        'url': 'https://www.stelledoppie.it',
        'enabled': True,
        'priority': 1
    },
    'wds': {
        'name': 'Washington Double Star Catalog',
        'description': 'The main catalog of double and multiple stars',
        'url': 'https://www.usno.navy.mil/USNO/astrometry/optical-IR-prod/wds',
        'enabled': True,
        'priority': 2
    },
    'gaia': {
        'name': 'Gaia DR3',
        'description': 'ESA Gaia Data Release 3',
        'url': 'https://gea.esac.esa.int/archive/',
        'enabled': True,
        'priority': 3
    },
    'hipparcos': {
        'name': 'Hipparcos/Tycho',
        'description': 'Hipparcos and Tycho catalogs',
        'url': 'https://www.cosmos.esa.int/web/hipparcos',
        'enabled': False,
        'priority': 4
    }
}

# Analysis Mode Configuration
DEFAULT_ANALYSIS_MODE = 'discovery'
AVAILABLE_ANALYSIS_MODES = ['discovery', 'characterize', 'orbital']

# Default sort keys for each analysis mode
DEFAULT_SORT_KEYS = {
    'discovery': 'v_total_arcsec_yr',
    'characterize': 'rmse',
    'orbital': 'opi_arcsec_yr'
}

# === PLANNER Module Configuration - Observatory Planning & Sky Quality ===

# Sky Quality Map Configuration
DEFAULT_GRID_RESOLUTION_ARCMIN = 60  # 1° grid resolution for planning
FINE_GRID_RESOLUTION_ARCMIN = 15     # 0.25° for high-precision applications
COARSE_GRID_RESOLUTION_ARCMIN = 180  # 3° for quick surveys

# Atmospheric Extinction Coefficients (La Silla Observatory - ESO)
# Reference: Burki et al. (1995), A&AS, 112, 383
EXTINCTION_COEFFICIENT_U = 0.55  # magnitudes per airmass
EXTINCTION_COEFFICIENT_B = 0.35  # magnitudes per airmass  
EXTINCTION_COEFFICIENT_V = 0.25  # magnitudes per airmass
EXTINCTION_COEFFICIENT_R = 0.18  # magnitudes per airmass
EXTINCTION_COEFFICIENT_I = 0.12  # magnitudes per airmass
DEFAULT_EXTINCTION_COEFFICIENT = 0.25  # V-band default for unknown filters

# Sky Brightness Models (Dark Site Standards)
# Reference: Garstang (1989), PASP, 101, 306
PRISTINE_SKY_BRIGHTNESS_V_MAG_ARCSEC2 = 21.9  # Pristine dark site (V-band)
EXCELLENT_SKY_BRIGHTNESS_V_MAG_ARCSEC2 = 21.6  # Excellent observatory site
GOOD_SKY_BRIGHTNESS_V_MAG_ARCSEC2 = 21.0      # Good suburban site
MODERATE_SKY_BRIGHTNESS_V_MAG_ARCSEC2 = 19.5   # Moderate light pollution
POOR_SKY_BRIGHTNESS_V_MAG_ARCSEC2 = 17.0       # Urban environment

# Observational Limits (Astronomy Standards)
MIN_OBSERVABLE_ALTITUDE_DEG = 15.0    # Below this, atmospheric effects dominate
OPTIMAL_MIN_ALTITUDE_DEG = 30.0       # Standard minimum for quality observations
ZENITH_AVOIDANCE_ZONE_DEG = 5.0       # Degrees from zenith to avoid for tracking

# Airmass Calculation Parameters
MAX_AIRMASS_FOR_PHOTOMETRY = 3.0      # Beyond this, systematic errors increase
MAX_AIRMASS_FOR_SPECTROSCOPY = 2.0    # Stricter limit for spectroscopic work
AIRMASS_WARNING_THRESHOLD = 2.5       # Issue warnings above this threshold

# Lunar Contamination Model (Krisciunas & Schaefer 1991)
# Reference: PASP, 103, 1033-1039
LUNAR_K_EXTINCTION = 0.25             # Lunar V-band extinction coefficient
LUNAR_C1_COEFFICIENT = 3.84           # Moonlight scattering parameter 1
LUNAR_C2_COEFFICIENT = 0.026          # Phase-dependent scattering parameter
LUNAR_C3_COEFFICIENT = 4e-9           # Higher-order phase correction
LUNAR_C4_COEFFICIENT = 0.631          # Geometric scattering parameter
LUNAR_C5_COEFFICIENT = 1.06           # Angular scattering baseline
LUNAR_C6_COEFFICIENT = 5.36           # Distance-dependent scattering
LUNAR_C7_COEFFICIENT = 40.0           # Scattering distance scale (degrees)

# Time Sampling Configuration (Ephemeris Calculations)
NIGHTLY_EVENTS_SAMPLING_MINUTES = 10  # Time resolution for twilight calculations
ASTRONOMICAL_MIDNIGHT_SAMPLING_MINUTES = 5  # Higher resolution for midnight finding
MAX_EVENT_SEARCH_WINDOW_HOURS = 48   # Maximum window for event finding

# Quality Metric Configuration
SKY_QUALITY_WEIGHT_BRIGHTNESS = 0.6   # Relative importance of sky darkness
SKY_QUALITY_WEIGHT_AIRMASS = 0.4      # Relative importance of low airmass
MIN_QUALITY_SCORE_THRESHOLD = 0.1     # Minimum quality for recommendations

# Coordinate Precision for Observatory Planning
COORDINATE_PRECISION_DEGREES = 4      # Decimal places for RA/Dec (standard)
ALTITUDE_PRECISION_DEGREES = 2       # Decimal places for altitude display
AZIMUTH_PRECISION_DEGREES = 1        # Decimal places for azimuth display
TIME_PRECISION_MINUTES = 1           # Temporal precision for event predictions

# Validation Ranges for Observatory Parameters
MIN_OBSERVATORY_LATITUDE_DEG = -90.0  # South Pole
MAX_OBSERVATORY_LATITUDE_DEG = 90.0   # North Pole
MIN_OBSERVATORY_LONGITUDE_DEG = -180.0 # International Date Line
MAX_OBSERVATORY_LONGITUDE_DEG = 180.0  # International Date Line
MIN_OBSERVATORY_ALTITUDE_M = -500.0    # Below sea level observatories
MAX_OBSERVATORY_ALTITUDE_M = 6000.0    # Practical limit for ground-based astronomy

# Error Handling for Astronomical Calculations
SKYFIELD_EPHEMERIS_TOLERANCE = 1e-12  # Precision for ephemeris calculations
COORDINATE_CONVERSION_TOLERANCE = 1e-10 # Precision for coordinate transformations
TWILIGHT_CALCULATION_TOLERANCE_MINUTES = 0.5  # Acceptable error in twilight times

# === DATA SOURCES Configuration ===

# Available data sources for analysis
DATA_SOURCES = ['web', 'local', 'gaia']

# Gaia Archive Configuration
DEFAULT_GAIA_TABLE = 'gaiaedr3.gaia_source'  # Gaia EDR3 main table
DEFAULT_GAIA_SEARCH_RADIUS_ARCSEC = 10.0     # Search radius around target
DEFAULT_GAIA_MAG_LIMIT = 18.0                # Magnitude limit for Gaia queries
DEFAULT_GAIA_TIMEOUT_SECONDS = 30            # Timeout for Gaia queries
DEFAULT_GAIA_MAX_ROWS = 1000                 # Maximum rows to retrieve
DEFAULT_GAIA_ROW_LIMIT = 1000                # Alias for backward compatibility

# Gaia Physical Validation Thresholds
DEFAULT_PHYSICAL_P_VALUE_THRESHOLD = 0.05   # Threshold for physical companion classification
DEFAULT_AMBIGUOUS_P_VALUE_THRESHOLD = 0.001 # Threshold for ambiguous classification (must be < physical)
GAIA_QUERY_TIMEOUT_SECONDS = 30             # Timeout for individual Gaia queries
GAIA_MAX_RETRY_ATTEMPTS = 3                 # Maximum retry attempts for failed queries
GAIA_RETRY_DELAY_SECONDS = 2.0              # Delay between retry attempts

# Gaia Data Quality Thresholds
MIN_PARALLAX_SIGNIFICANCE = 3.0             # Minimum parallax/error ratio for reliability
MIN_PM_SIGNIFICANCE = 3.0                   # Minimum proper motion significance
GAIA_MAX_RUWE = 1.4                         # Maximum RUWE for good astrometric solution (Lindegren et al. 2018)
GAIA_DEFAULT_CORRELATION_MISSING = 0.0      # Default correlation coefficient when missing

# === PHYSICS Configuration - Kepler Solver ===

# Kepler's Equation Solver Parameters
DEFAULT_KEPLER_TOLERANCE = 1e-12             # Convergence tolerance for Kepler solver
DEFAULT_KEPLER_MAX_ITERATIONS = 50           # Maximum iterations for convergence
HIGH_ECCENTRICITY_THRESHOLD = 0.7           # Threshold for high-e initial guess
HIGH_E_COEFFICIENT = 1.2                    # Coefficient for high-e initial guess
DANGEROUS_ECCENTRICITY_WARNING = 0.95       # Issue warnings above this eccentricity

# Orbital Element Validation Ranges
MAX_ECCENTRICITY_STABLE = 0.99              # Maximum stable eccentricity

# Kepler Solver Quality Control
KEPLER_CONVERGENCE_WARNING_THRESHOLD = 95.0  # Percentage threshold for convergence warnings
KEPLER_LOGGING_PRECISION = 6                # Decimal places for logging

# Orbital Element Validation Ranges (Additional)
MIN_LONGITUDE_ASCENDING_NODE_DEG = 0.0      # Minimum Omega (degrees)
MAX_LONGITUDE_ASCENDING_NODE_DEG = 360.0    # Maximum Omega (degrees)
MIN_ARGUMENT_PERIASTRON_DEG = 0.0           # Minimum omega (degrees)
MAX_ARGUMENT_PERIASTRON_DEG = 360.0         # Maximum omega (degrees)
MIN_EPOCH_PERIASTRON_YEAR = 1900.0          # Minimum reasonable epoch
MAX_EPOCH_PERIASTRON_YEAR = 2100.0          # Maximum reasonable epoch

# === PHYSICS Configuration - Dynamics & OPI ===

# Observation Priority Index (OPI) Configuration
OPI_DEVIATION_THRESHOLD_ARCSEC = 0.1        # Threshold for zero-time-baseline OPI calculation
OPI_INFINITE_THRESHOLD = 1000.0             # Value representing infinite priority

# Robust Linear Fit Configuration
MIN_POINTS_FOR_ROBUST_FIT = 3               # Minimum measurements required for robust Theil-Sen analysis
ROBUST_REGRESSION_RANDOM_STATE = 42         # Random state for reproducible results

# Astrometric Motion Validation
MAX_ASTROMETRIC_VELOCITY_ARCSEC_PER_YEAR = 10.0  # Maximum realistic proper motion
MIN_TIME_BASELINE_YEARS = 1.0               # Minimum time baseline for meaningful analysis

# Curvature Analysis Configuration
MAX_CURVATURE_INDEX_ARCSEC = 50.0            # Maximum expected curvature deviation
MIN_PREDICTION_DATE_OFFSET_YEARS = -100.0  # Historical limit for reliable predictions
MAX_PREDICTION_DATE_OFFSET_YEARS = 100.0    # Future limit for reliable predictions

# Statistical Validation Thresholds
MAX_RMSE_FOR_LINEAR_FIT_ARCSEC = 1.0  # RMSE threshold for acceptable linear fits
MIN_RESIDUAL_SIGNIFICANCE = 0.001  # Minimum residual for statistical significance
MAX_EXTRAPOLATION_FACTOR = 2.0     # Maximum safe extrapolation beyond observation baseline

# === CLI Configuration - Display Parameters ===
# Terminal Display Configuration
CLI_RESULTS_SEPARATOR_WIDTH = 80  # Width for result display separators
CLI_HEADER_CHAR = "="  # Character for main headers
CLI_SUBHEADER_CHAR = "-"  # Character for sub-headers

# Formatting Precision
CLI_OPI_PRECISION = 3  # Decimal places for Observation Priority Index
CLI_RMSE_PRECISION = 4  # Decimal places for RMSE (arcsec)
CLI_VELOCITY_PRECISION = 3  # Decimal places for proper motion (arcsec/yr)
CLI_CURVATURE_PRECISION = 4  # Decimal places for curvature index
CLI_SEPARATION_PRECISION = 2  # Decimal places for angular separation (arcsec)

# Column Width Configuration
CLI_WDS_NAME_WIDTH = 12  # Fixed width for WDS designation display
CLI_METRIC_LABEL_WIDTH = 6  # Width for metric labels (OPI, RMSE, etc.)
CLI_NUMERIC_FIELD_WIDTH = 6  # Width for numeric value display
CLI_OBS_COUNT_WIDTH = 3  # Width for observation count display

# Result Display Formatting
CLI_VALUE_NOT_AVAILABLE = "N/A"  # Standard text for missing values
CLI_COLUMN_SEPARATOR = " | "  # Separator between table columns

# === CLI Validation Configuration ===
# Argument Validation Ranges - Bounds
MIN_OBSERVATION_COUNT = 1  # Minimum meaningful observation count
MAX_OBSERVATION_COUNT = 10000  # Upper bound for observation filtering
MIN_CONCURRENT_REQUESTS = 1  # Minimum for single-threaded operation
MAX_CONCURRENT_REQUESTS = 100  # Upper bound to prevent system overload
MIN_GAIA_P_VALUE = 1e-10  # Lower bound for statistical significance
MAX_GAIA_P_VALUE = 1.0  # Upper bound for probability values

# Data Processing Limits
MAX_INPUT_FILE_SIZE_MB = 100  # Maximum CSV file size for processing
MAX_STARS_PER_ANALYSIS = 10000  # Upper limit for single analysis run
DEFAULT_PROCESSING_TIMEOUT_SECONDS = 3600  # 1 hour timeout for large analyses

# Error Handling Configuration
CLI_MAX_RETRY_ATTEMPTS = 3  # Maximum retries for failed operations
CLI_RETRY_DELAY_SECONDS = 2.0  # Delay between retry attempts
CLI_ERROR_LOG_DETAIL_LEVEL = "DETAILED"  # Options: "BASIC", "DETAILED", "DEBUG"

# === Dynamics Module Configuration ===
# Observation Priority Index (OPI) Configuration
OPI_DEVIATION_THRESHOLD_ARCSEC = 0.1  # Threshold for zero-time-baseline OPI calculation
OPI_INFINITE_THRESHOLD = 1000.0  # Maximum reasonable OPI value before considering infinite

# Robust Fitting Configuration
MIN_POINTS_FOR_ROBUST_FIT = 3  # Minimum measurements required for robust Theil-Sen analysis
ROBUST_REGRESSION_RANDOM_STATE = 42  # Fixed seed for reproducible robust regression results

# Astrometric Motion Limits - Physical Constraints
MAX_ASTROMETRIC_VELOCITY_ARCSEC_PER_YEAR = 10.0  # Maximum reasonable astrometric velocity
MIN_TIME_BASELINE_YEARS = 1.0  # Minimum meaningful time baseline for motion analysis
MAX_CURVATURE_INDEX_ARCSEC = 50.0  # Maximum reasonable curvature index value

# Prediction and Extrapolation Safety Limits
MIN_PREDICTION_DATE_OFFSET_YEARS = -100.0  # Historical prediction limit (relative to fit epoch)
MAX_PREDICTION_DATE_OFFSET_YEARS = 100.0   # Future prediction limit (relative to fit epoch)
MAX_EXTRAPOLATION_FACTOR = 2.0  # Maximum safe extrapolation factor (prediction/baseline ratio)

# Fit Quality Thresholds
MAX_RMSE_FOR_LINEAR_FIT_ARCSEC = 1.0  # Maximum RMSE for acceptable linear fit quality
MIN_RESIDUAL_SIGNIFICANCE = 0.001  # Minimum significant residual value (arcsec)

# Validation Ranges for Input Data
MIN_EPOCH_YEAR = 1800.0  # Minimum reasonable historical epoch
MAX_EPOCH_YEAR = 2100.0  # Maximum reasonable future epoch
MIN_SEPARATION_ARCSEC = 0.001  # Minimum measurable angular separation
MAX_SEPARATION_ARCSEC = 100.0  # Maximum reasonable angular separation for binary stars

# Warning Thresholds for Validation
MAX_DEVIATION_WARNING_ARCSEC = 10.0  # Threshold for large positional deviation warning
MAX_OLD_OBSERVATION_WARNING_YEARS = 50.0  # Threshold for very old observation warning

# === Local Source Configuration ===
# Unit Conversion Constants for Catalog Parsing
DAYS_PER_JULIAN_YEAR = 365.25  # Standard Julian year for astronomical epoch conversions
CENTURIES_PER_YEAR = 100.0     # Century to year conversion factor
MILLIARCSEC_PER_ARCSEC = 1000.0  # Milliarcsecond to arcsecond conversion

# Validation Constants (CLI imports)
MIN_EPOCH_YEAR = 1800.0
MAX_EPOCH_YEAR = 2100.0
MIN_SEPARATION_ARCSEC = 0.1
MAX_SEPARATION_ARCSEC = 300.0
MIN_POSITION_ANGLE_DEG = 0.0
MAX_POSITION_ANGLE_DEG = 360.0
