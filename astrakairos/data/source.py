from abc import ABC, abstractmethod
from typing import Dict, Any

class DataSource(ABC):
    """Abstract base class for all AstraKairos data sources."""

    @abstractmethod
    async def get_wds_data(self, wds_id: str) -> Dict[str, Any]:
        """
        Obtains basic WDS data for a star (last measurement, etc.).
        
        Args:
            wds_id: The WDS identifier for the star
            
        Returns:
            Dictionary with WDS data including date_last, pa_last, sep_last, etc.
        """
        pass

    @abstractmethod
    async def get_orbital_elements(self, wds_id: str) -> Dict[str, Any]:
        """
        Obtains the 7 Keplerian orbital elements.
        
        Args:
            wds_id: The WDS identifier for the star
            
        Returns:
            Dictionary with orbital elements (P, T, e, a, i, Omega, omega)
        """
        pass

    @abstractmethod
    async def validate_physicality(self, wds_id: str) -> str:
        """
        Validates if a system is physical using external data (e.g., Gaia).
        
        Args:
            wds_id: The WDS identifier for the star
        
        Returns:
            A string: "Physical", "Optical", "Ambiguous", or "Unknown".
        """
        pass