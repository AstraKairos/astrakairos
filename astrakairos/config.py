"""
Configuration constants for AstraKairos.

This module centralizes all configuration parameters used throughout the application,
making them easily configurable and maintainable.
"""

# VizieR Service Configuration
DEFAULT_VIZIER_ROW_LIMIT = 100
DEFAULT_VIZIER_TIMEOUT = 30
DEFAULT_VIZIER_RETRY_ATTEMPTS = 3
DEFAULT_VIZIER_RETRY_DELAY = 1.0  # seconds

# Physical Validation Ranges
MIN_PERIOD_YEARS = 0.1
MAX_PERIOD_YEARS = 100000.0
MIN_SEMIMAJOR_AXIS_ARCSEC = 0.001
MAX_SEMIMAJOR_AXIS_ARCSEC = 100.0
MIN_ECCENTRICITY = 0.0
MAX_ECCENTRICITY = 0.99
MIN_INCLINATION_DEG = 0.0
MAX_INCLINATION_DEG = 180.0

# Coordinate Validation
MIN_RA_DEG = 0.0
MAX_RA_DEG = 360.0
MIN_DEC_DEG = -90.0
MAX_DEC_DEG = 90.0

# Logging Configuration
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# I/O Configuration
CSV_SNIFFER_SAMPLE_SIZE = 2048  # bytes
DEFAULT_COORDINATE_PRECISION = 2  # decimal places for coordinate display

# Error Handling Configuration
COORDINATE_ERROR_BEHAVIOR = "return_invalid"  # Options: "raise", "return_none", "return_invalid"

# CLI Analysis Configuration
MIN_MEASUREMENTS_FOR_CHARACTERIZE = 5
DEFAULT_MIN_OBS = 2
DEFAULT_MAX_OBS = 10
DEFAULT_CONCURRENT_REQUESTS = 5
DEFAULT_GAIA_P_VALUE = 0.01
TOP_RESULTS_DISPLAY_COUNT = 10

# Analysis modes
ANALYSIS_MODES = ['orbital', 'characterize', 'discovery']

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

# Sky Quality Map Configuration (Scientific Standards)
DEFAULT_GRID_RESOLUTION_ARCMIN = 60  # 1° grid resolution for professional planning
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

# Observational Limits (Professional Astronomy Standards)
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
COORDINATE_PRECISION_DEGREES = 4      # Decimal places for RA/Dec (professional standard)
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
KEPLER_LOGGING_PRECISION = 6                # Decimal places for scientific logging

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
MIN_POINTS_FOR_ROBUST_FIT = 5               # Minimum measurements required for robust Theil-Sen analysis
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

# Default sort keys for each analysis mode
DEFAULT_SORT_KEYS = {
    'orbital': 'opi',
    'characterize': 'rmse',  # Updated to match dynamics.py output
    'discovery': 'physicality_p_value'
}

# === CLI Configuration - Scientific Display Parameters ===
# Terminal Display Configuration
CLI_RESULTS_SEPARATOR_WIDTH = 80  # Width for result display separators
CLI_HEADER_CHAR = "="  # Character for main headers
CLI_SUBHEADER_CHAR = "-"  # Character for sub-headers

# Scientific Formatting Precision
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

# === CLI Scientific Validation Configuration ===
# Argument Validation Ranges - Scientific Bounds
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
MIN_POINTS_FOR_ROBUST_FIT = 5  # Minimum measurements required for robust Theil-Sen analysis
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

# Warning Thresholds for Scientific Validation
MAX_DEVIATION_WARNING_ARCSEC = 10.0  # Threshold for large positional deviation warning
MAX_OLD_OBSERVATION_WARNING_YEARS = 50.0  # Threshold for very old observation warning

# === Local Source Configuration ===
# Unit Conversion Constants for Catalog Parsing
DAYS_PER_JULIAN_YEAR = 365.25  # Standard Julian year for astronomical epoch conversions
CENTURIES_PER_YEAR = 100.0     # Century to year conversion factor
MILLIARCSEC_PER_ARCSEC = 1000.0  # Milliarcsecond to arcsecond conversion

# === Online Source Configuration ===
# VizieR Catalog Identifiers - Updated versions can be changed here
VIZIER_WDS_CATALOG = "B/wds/wds"
VIZIER_ORBITAL_CATALOG = "J/MNRAS/517/2925/tablea3"  # ORB6 catalog

# Backoff Strategy Configuration
VIZIER_BACKOFF_BASE = 2.0  # Exponential backoff base
VIZIER_BACKOFF_MAX_DELAY = 30.0  # Maximum delay in seconds

# Validation Ranges for External Data
MIN_MAGNITUDE = -5.0  # Minimum reasonable magnitude
MAX_MAGNITUDE = 25.0  # Maximum reasonable magnitude

# CLI Configuration
DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_HTTP_CONNECTIONS_PER_HOST = 5
DEFAULT_HTTP_TOTAL_CONNECTIONS = 50
DEFAULT_USER_AGENT = 'AstraKairos/1.0 (https://github.com/AstraKairos/astrakairos)'

# Analysis Configuration
DEFAULT_CONCURRENT_REQUESTS = 5
DEFAULT_MIN_OBSERVATIONS = 2
DEFAULT_SORT_BY = 'v_total'
DEFAULT_GAIA_P_VALUE = 0.01
DEFAULT_GAIA_RADIUS_FACTOR = 1.5
DEFAULT_GAIA_MIN_RADIUS = 2.0
DEFAULT_GAIA_MAX_RADIUS = 60.0

# Validation Constants for Scientific Rigor
MIN_EPOCH_YEAR = 1800.0
MAX_EPOCH_YEAR = 2100.0
MIN_SEPARATION_ARCSEC = 0.1
MAX_SEPARATION_ARCSEC = 300.0
MIN_POSITION_ANGLE_DEG = 0.0
MAX_POSITION_ANGLE_DEG = 360.0
