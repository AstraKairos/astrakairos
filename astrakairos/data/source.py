from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypedDict, List
from astropy.table import Table
from enum import Enum


class AstraKairosDataError(Exception):
    """Base exception for AstraKairos data-related errors."""
    pass


class WdsIdNotFoundError(AstraKairosDataError):
    """Raised when a WDS identifier is not found in the data source."""
    pass


class OrbitalElementsUnavailableError(AstraKairosDataError):
    """Raised when orbital elements are not available for a given system."""
    pass


class MeasurementsUnavailableError(AstraKairosDataError):
    """Raised when measurements are not available for a given system."""
    pass


class PhysicalityValidationError(AstraKairosDataError):
    """Raised when physicality validation fails due to insufficient data or errors."""
    pass


class InvalidInputError(AstraKairosDataError):
    """Raised when input data is invalid or missing required fields."""
    pass


class CacheStatsError(AstraKairosDataError):
    """Raised when cache statistics cannot be retrieved."""
    pass


class AnalysisError(AstraKairosDataError):
    """Raised when analysis operations fail."""
    pass


class ValidationError(AstraKairosDataError):
    """Raised when data validation fails."""
    pass


class PhysicalityLabel(Enum):
    """Standardized physicality assessment labels."""
    LIKELY_PHYSICAL = "Likely Physical"
    LIKELY_OPTICAL = "Likely Optical"
    AMBIGUOUS = "Ambiguous"
    UNKNOWN = "Unknown"
    INSUFFICIENT_DATA = "Insufficient Data"


class ValidationMethod(Enum):
    """Available validation methods for physicality assessment.
    
    Note: This enum only contains actual validation methods.
    When validation cannot be performed due to insufficient data,
    the method field should be set to None rather than using an enum value.
    """
    GAIA_3D_PARALLAX_PM = "Gaia 3D (plx+pm)"
    GAIA_PARALLAX_ONLY = "Gaia Parallax"
    PROPER_MOTION_ONLY = "Proper Motion"
    STATISTICAL_ANALYSIS = "Statistical Analysis"

class WdsSummary(TypedDict, total=True):
    """Data structure for a single-line entry from the WDS summary catalog.
    
    Args:
        wds_id: Primary WDS identifier (required)
        discoverer: Discoverer designation (required)
        components: Component designations (e.g., "AB", "A,B,C") (required)
        ra_deg: Right ascension in degrees (required)
        dec_deg: Declination in degrees (required)
        date_first: First observation epoch (required)
        date_last: Last observation epoch (required)
        n_observations: Number of observations (required)
        pa_first: First position angle in degrees (required)
        pa_last: Last position angle in degrees (required)
        sep_first: First separation in arcseconds (required)
        sep_last: Last separation in arcseconds (required)
        mag_pri: Primary star magnitude (required)
        mag_sec: Secondary star magnitude (may be missing in some catalogs)
        spec_type: Spectral type (may be missing in some catalogs)
        pa_first_error: Uncertainty in first position angle (degrees), optional
        pa_last_error: Uncertainty in last position angle (degrees), optional
        sep_first_error: Uncertainty in first separation (arcsec), optional
        sep_last_error: Uncertainty in last separation (arcsec), optional
        wdss_id: Original WDSS ID for reference (WDSS only), optional
        discoverer_designation: Full discoverer designation (WDSS only), optional
    """
    # Essential fields - always required
    wds_id: str
    discoverer: str
    components: str
    ra_deg: float
    dec_deg: float
    date_first: float
    date_last: float
    n_observations: int
    pa_first: float
    pa_last: float
    sep_first: float
    sep_last: float
    mag_pri: float
    
    # Fields that may be missing in some catalogs
    mag_sec: Optional[float]
    spec_type: Optional[str]
    
    # Error fields - always optional
    pa_first_error: Optional[float]
    pa_last_error: Optional[float]
    sep_first_error: Optional[float]
    sep_last_error: Optional[float]
    
    # Catalog-specific fields - optional
    wdss_id: Optional[str]
    discoverer_designation: Optional[str]
    
    # Gaia source IDs for enhanced validation - scalable for multi-component systems
    # Note: While AstraKairos analyzes binary pairs, stellar systems can have many components (A-Z and beyond)
    # We store Gaia IDs as a flexible mapping to handle any component letter combination
    gaia_source_ids: Optional[Dict[str, str]]  # Component letter -> Gaia DR3 source_id mapping
    
    # Multi-pair architecture fields - optional  
    system_pair_id: Optional[str]

class OrbitalElements(TypedDict, total=True):
    """Data structure for complete Keplerian orbital elements with uncertainties.
    
    Args:
        wds_id: Primary WDS identifier (required)
        P: Period (years) (required)
        T: Epoch of periastron (years) (required)
        e: Eccentricity (required)
        a: Semi-major axis (arcsec) (required)
        i: Inclination (degrees) (required)
        Omega: Longitude of ascending node (degrees) (required)
        omega: Argument of periastron (degrees) (required)
        reference: Source reference/publication (required)
        grade: Orbit quality grade (1=best to 5=worst) (required)
        last_updated: ISO format date string (required)
        e_P: Period uncertainty (years), optional
        e_T: Epoch of periastron uncertainty (years), optional
        e_e: Eccentricity uncertainty (dimensionless), optional
        e_a: Semi-major axis uncertainty (arcsec), optional
        e_i: Inclination uncertainty (degrees), optional
        e_Omega: Longitude of ascending node uncertainty (degrees), optional
        e_omega_arg: Argument of periastron uncertainty (degrees), optional
    """
    # Essential orbital elements - always required
    wds_id: str
    P: float
    T: float
    e: float
    a: float
    i: float
    Omega: float
    omega: float
    
    # Essential metadata - always required
    reference: str
    grade: int
    last_updated: str
    
    # Complete uncertainty set - all optional as not all catalogs provide uncertainties
    e_P: Optional[float]
    e_T: Optional[float]
    e_e: Optional[float]
    e_a: Optional[float]
    e_i: Optional[float]
    e_Omega: Optional[float]
    e_omega_arg: Optional[float]

class BasePhysicalityAssessment(TypedDict, total=True):
    """Base data structure for physicality validation results.
    
    Args:
        label: Standardized assessment result (required)
        confidence: 0-1 confidence score (required)
        p_value: Statistical significance (required)
        method: Method used for validation, None if insufficient data (may be None)
        validation_date: ISO format timestamp (required)
        retry_attempts: Number of retry attempts made, optional
    """
    # Essential fields - always required
    label: PhysicalityLabel
    confidence: float
    p_value: float
    validation_date: str
    
    # Method may be None when label is INSUFFICIENT_DATA or UNKNOWN
    method: Optional[ValidationMethod]
    
    # Optional metadata
    retry_attempts: Optional[int]

class ThresholdsDict(TypedDict):
    """Type specification for significance thresholds in physicality assessment."""
    physical: float
    ambiguous: float

class PhysicalityAssessment(BasePhysicalityAssessment, total=False):
    """Extended physicality assessment with Gaia-specific metrics."""
    
    # Gaia-specific metrics
    parallax_consistency: Optional[float]
    proper_motion_consistency: Optional[float]
    gaia_source_id_primary: Optional[str]
    gaia_source_id_secondary: Optional[str]
    
    # Gaia-specific metadata
    search_radius_arcsec: float
    significance_thresholds: ThresholdsDict

# --- Abstract Base Class ---

class DataSource(ABC):
    """Abstract Base Class for all AstraKairos data sources.

    This interface defines a contract for fetching astronomical catalog data
    related to double star systems. Each method is designed to be explicit
    about the data it retrieves and integrates with the configuration system
    for consistent, configurable behavior.
    
    Physicality validation has been separated from data access.
    Use PhysicalityValidator classes for validation logic.
    """

    @abstractmethod
    async def get_all_component_pairs(self, wds_id: str) -> List[WdsSummary]:
        """Obtains summary data for ALL component pairs of a star system.
        
        For multi-pair systems, this returns a separate WdsSummary for each 
        component pair (AB, AC, BD, CE, etc.) as independent systems.
        For traditional databases, this returns a single-element list.
        
        Args:
            wds_id: The WDS identifier for the star.
            
        Returns:
            A list of WdsSummary objects, one for each component pair.
            
        Raises:
            WdsIdNotFoundError: If the WDS identifier is not found in the catalog.
        """
        pass

    @abstractmethod
    async def get_wds_summary(self, wds_id: str) -> WdsSummary:
        """Obtains the summary data for a star from the main WDS catalog.
        
        This corresponds to a single-line entry with aggregated data.
        
        Args:
            wds_id: The WDS identifier for the star.
            
        Returns:
            A WdsSummary dictionary.
            
        Raises:
            WdsIdNotFoundError: If the WDS identifier is not found in the catalog.
        """
        pass

    @abstractmethod
    async def get_all_measurements(self, wds_id: str) -> Table:
        """Obtains all historical astrometric measurements for a star.
        
        This queries the comprehensive measurement catalog (e.g., WDS B/wds/wds).

        Args:
            wds_id: The WDS identifier for the star.

        Returns:
            An astropy Table with all measurements containing at minimum:
            - 'epoch': Observation epoch (decimal years)
            - 'theta': Position angle (degrees)
            - 'rho': Separation (arcseconds)
            
        Raises:
            WdsIdNotFoundError: If the WDS identifier is not found.
            MeasurementsUnavailableError: If measurements exist but cannot be retrieved.
        """
        pass

    @abstractmethod
    async def get_orbital_elements(self, wds_id: str) -> OrbitalElements:
        """Obtains the 7 Keplerian orbital elements for a star.
        
        Args:
            wds_id: The WDS identifier for the star.
            
        Returns:
            An OrbitalElements dictionary.
            
        Raises:
            WdsIdNotFoundError: If the WDS identifier is not found.
            OrbitalElementsUnavailableError: If orbital elements are not available.
        """
        pass


class PhysicalityValidator(ABC):
    """Abstract base class for physicality validation methods.
    
    This interface separates validation logic from data access,
    allowing for different validation approaches (Gaia, spectroscopy, ML).
    """
    
    @abstractmethod
    async def validate_physicality(
        self,
        system_data: WdsSummary,
        **kwargs
    ) -> BasePhysicalityAssessment:
        """Validates if a binary system is likely physically bound.
        
        Args:
            system_data: Basic system information (coordinates, magnitudes)
            **kwargs: Method-specific parameters
            
        Returns:
            BasePhysicalityAssessment or subclass.
            
        Raises:
            PhysicalityValidationError: If validation fails due to insufficient data or errors.
        """
        pass