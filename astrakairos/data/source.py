from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from astropy.table import Table

# --- Data Structures for Type Hinting ---
# Using specific TypedDicts instead of Dict[str, Any] makes the code
# much clearer and allows static analysis tools to catch errors.

from typing import TypedDict

class WdsSummary(TypedDict, total=False):
    """Data structure for a single-line entry from the WDS summary catalog."""
    wds_id: str
    ra_deg: float
    dec_deg: float
    date_first: float
    date_last: float
    obs: int
    pa_first: int
    pa_last: int
    sep_first: float
    sep_last: float
    mag_pri: float
    mag_sec: float

class OrbitalElements(TypedDict, total=False):
    """Data structure for the 7 Keplerian orbital elements."""
    P: float
    T: float
    e: float
    a: float
    i: float
    Omega: float
    omega: float

class PhysicalityAssessment(TypedDict, total=False):
    """Data structure for the result of a physicality validation."""
    label: str  # e.g., 'Likely Physical', 'Likely Optical', 'Ambiguous', 'Unknown'
    p_value: float
    test_used: str # e.g., 'Gaia 3D (plx+pm)'

# --- Abstract Base Class ---

class DataSource(ABC):
    """
    Abstract Base Class for all AstraKairos data sources.

    This interface defines a contract for fetching various types of data
    related to double star systems. Each method is designed to be explicit
    about the data it retrieves.
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
            An astropy Table with all measurements, or None if not found.
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
    async def validate_physicality(self, system_data: WdsSummary) -> Optional[PhysicalityAssessment]:
        """
        Validates if a system is likely physically bound using external data (e.g., Gaia).

        This method requires summary data (coordinates, magnitudes) to perform its task.
        
        Args:
            system_data: A WdsSummary dictionary containing the star's coordinates and magnitudes.
        
        Returns:
            A PhysicalityAssessment dictionary with the validation result.
        """
        pass