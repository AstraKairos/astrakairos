from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from astropy.table import Table
from enum import Enum

# --- Data Structures for Type Hinting ---
from typing import TypedDict

# Import configuration constants
from ..config import (
    DEFAULT_PHYSICAL_P_VALUE_THRESHOLD,
    DEFAULT_AMBIGUOUS_P_VALUE_THRESHOLD,
    GAIA_QUERY_TIMEOUT_SECONDS,
    MIN_PARALLAX_SIGNIFICANCE,
    MIN_PM_SIGNIFICANCE,
    DEFAULT_GAIA_SEARCH_RADIUS_ARCSEC,
    DEFAULT_GAIA_MAG_LIMIT,
    GAIA_MAX_RETRY_ATTEMPTS,
    GAIA_RETRY_DELAY_SECONDS
)

# --- Enums for categorical data ---
class PhysicalityLabel(Enum):
    """Standardized physicality assessment labels."""
    LIKELY_PHYSICAL = "Likely Physical"
    LIKELY_OPTICAL = "Likely Optical"
    AMBIGUOUS = "Ambiguous"
    UNKNOWN = "Unknown"
    INSUFFICIENT_DATA = "Insufficient Data"

class ValidationMethod(Enum):
    """Available validation methods for physicality assessment."""
    GAIA_3D_PARALLAX_PM = "Gaia 3D (plx+pm)"
    GAIA_PARALLAX_ONLY = "Gaia Parallax"
    PROPER_MOTION_ONLY = "Proper Motion"
    STATISTICAL_ANALYSIS = "Statistical Analysis"
    INSUFFICIENT_DATA = "Insufficient Data"

class WdsSummary(TypedDict, total=False):
    """Data structure for a single-line entry from the WDS summary catalog."""
    wds_id: str  # Changed from wds_name for consistency with astronomical standards
    ra_deg: float
    dec_deg: float
    date_first: float
    date_last: float
    n_observations: int  # Changed from obs for clarity
    pa_first: float      # Changed from int to float for precision
    pa_last: float       # Changed from int to float for precision
    sep_first: float
    sep_last: float
    mag_pri: float
    mag_sec: float

class OrbitalElements(TypedDict, total=False):
    """Data structure for complete Keplerian orbital elements with uncertainties."""
    wds_id: str  # Changed from wds_name for consistency
    
    # Orbital elements
    P: float      # Period (years)
    T: float      # Epoch of periastron (years)
    e: float      # Eccentricity
    a: float      # Semi-major axis (arcsec)
    i: float      # Inclination (degrees)
    Omega: float  # Longitude of ascending node (degrees)
    omega: float  # Argument of periastron (degrees)
    
    # Complete uncertainty set - all 7 elements for scientific completeness
    e_P: float    # Period uncertainty (years)
    e_T: float    # Epoch of periastron uncertainty (years)
    e_e: float    # Eccentricity uncertainty
    e_a: float    # Semi-major axis uncertainty (arcsec)
    e_i: float    # Inclination uncertainty (degrees)
    e_Omega: float # Longitude of ascending node uncertainty (degrees)
    e_omega: float # Argument of periastron uncertainty (degrees)
    
    # Scientific metadata
    reference: str  # Source reference/publication
    grade: int      # Orbit quality grade (1-5, where 1 is best)
    last_updated: str  # ISO format date string

class PhysicalityAssessment(TypedDict, total=False):
    """Data structure for physicality validation results."""
    label: PhysicalityLabel  # Enum instead of string for type safety
    confidence: float        # 0-1 confidence score
    p_value: float
    method: ValidationMethod # Enum instead of string for type safety
    
    # Detailed Gaia-specific metrics
    parallax_consistency: Optional[float]  # Parallax-based consistency metric
    proper_motion_consistency: Optional[float]  # Proper motion consistency metric
    gaia_source_id_primary: Optional[str]     # Gaia source ID for primary star
    gaia_source_id_secondary: Optional[str]   # Gaia source ID for secondary star
    
    # Analysis metadata for reproducibility
    validation_date: str  # ISO format timestamp
    search_radius_arcsec: float  # Search radius used
    significance_thresholds: Dict[str, float]  # Thresholds used in analysis
    retry_attempts: int  # Number of retry attempts made

# --- Abstract Base Class ---

class DataSource(ABC):
    """
    Abstract Base Class for all AstraKairos data sources.

    This interface defines a contract for fetching various types of data
    related to double star systems. Each method is designed to be explicit
    about the data it retrieves and integrates with the configuration system
    for consistent, configurable behavior.
    """

    @abstractmethod
    async def get_wds_summary(self, wds_id: str) -> Optional[WdsSummary]:
        """
        Obtains the summary data for a star from the main WDS catalog.
        This corresponds to a single-line entry with aggregated data.
        
        Args:
            wds_id: The WDS identifier for the star.
            
        Returns:
            A WdsSummary dictionary, or None if not found.
        """
        pass

    @abstractmethod
    async def get_all_measurements(self, wds_id: str) -> Optional[Table]:
        """
        Obtains all historical astrometric measurements for a star.
        This queries the comprehensive measurement catalog (e.g., WDSS I/313).

        Args:
            wds_id: The WDS identifier for the star.

        Returns:
            An astropy Table with all measurements containing at minimum:
            - 'epoch': Observation epoch (decimal years)
            - 'theta': Position angle (degrees)
            - 'rho': Separation (arcseconds)
            Returns None if not found or if data source doesn't support this operation.
        """
        pass

    @abstractmethod
    async def get_orbital_elements(self, wds_id: str) -> Optional[OrbitalElements]:
        """
        Obtains the 7 Keplerian orbital elements for a star.
        
        Args:
            wds_id: The WDS identifier for the star.
            
        Returns:
            An OrbitalElements dictionary, or None if not found.
        """
        pass

    @abstractmethod
    async def validate_physicality(
        self, 
        system_data: WdsSummary,
        p_value_threshold: float = DEFAULT_PHYSICAL_P_VALUE_THRESHOLD,
        ambiguous_threshold: float = DEFAULT_AMBIGUOUS_P_VALUE_THRESHOLD,
        search_radius_arcsec: float = DEFAULT_GAIA_SEARCH_RADIUS_ARCSEC,
        mag_limit: float = DEFAULT_GAIA_MAG_LIMIT,
        min_parallax_significance: float = MIN_PARALLAX_SIGNIFICANCE,
        min_pm_significance: float = MIN_PM_SIGNIFICANCE,
        timeout_seconds: float = GAIA_QUERY_TIMEOUT_SECONDS,
        max_retry_attempts: int = GAIA_MAX_RETRY_ATTEMPTS,
        retry_delay_seconds: float = GAIA_RETRY_DELAY_SECONDS
    ) -> Optional[PhysicalityAssessment]:
        """
        Validates if a system is likely physically bound using external data (e.g., Gaia).

        Uses configuration defaults but allows per-call customization for
        scientific analysis flexibility. This method requires summary data
        (coordinates, magnitudes) to perform its task.
        
        Args:
            system_data: A WdsSummary dictionary containing the star's coordinates and magnitudes
            p_value_threshold: Threshold for physical companion classification (default from config)
            ambiguous_threshold: Threshold for ambiguous classification (default from config)
            search_radius_arcsec: Search radius around target position (default from config)
            mag_limit: Magnitude limit for catalog queries (default from config)
            min_parallax_significance: Minimum parallax/error ratio (default from config)
            min_pm_significance: Minimum proper motion significance (default from config)
            timeout_seconds: Query timeout limit (default from config)
            max_retry_attempts: Maximum retry attempts for failed queries (default from config)
            retry_delay_seconds: Delay between retry attempts (default from config)
        
        Returns:
            A PhysicalityAssessment dictionary with validation results and metadata,
            or None if validation cannot be performed.
        """
        pass