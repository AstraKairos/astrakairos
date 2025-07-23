"""
Custom exceptions for AstraKairos.

This module defines domain-specific exceptions used throughout the application
to provide clear error context and enable precise error handling.
"""


class AstraKairosError(Exception):
    """Base exception for all AstraKairos-specific errors."""
    pass


class ConvergenceError(AstraKairosError):
    """Raised when numerical algorithms fail to converge."""
    pass


class InvalidOrbitalElementsError(AstraKairosError):
    """Raised when orbital elements are outside valid ranges."""
    pass


class NumericalInstabilityError(AstraKairosError):
    """Raised when numerical computations become unstable."""
    pass


class InvalidMassInputError(AstraKairosError):
    """Raised when mass calculation inputs are outside valid ranges."""
    pass


class InvalidDataFormatError(AstraKairosError):
    """Raised when data cannot be parsed due to format issues."""
    pass


class CatalogParsingError(AstraKairosError):
    """Raised when catalog parsing fails."""
    pass


class FileFormatError(AstraKairosError):
    """Raised when file format is invalid or unexpected."""
    pass


class DataValidationError(AstraKairosError):
    """Raised when data validation fails."""
    pass


class PhysicalityValidationError(AstraKairosError):
    """Raised when physicality validation cannot be completed."""
    pass


class ParallaxDataUnavailableError(AstraKairosError):
    """Raised when parallax data cannot be retrieved or is insufficient."""
    pass


class GaiaQueryError(AstraKairosError):
    """Raised when Gaia database queries fail."""
    pass


class InsufficientAstrometricDataError(AstraKairosError):
    """Raised when insufficient astrometric data is available for analysis."""
    pass


class ElBadryCrossmatchError(AstraKairosError):
    """Raised when El-Badry catalog cross-matching fails."""
    pass


class ConversionProcessError(AstraKairosError):
    """Raised when the catalog conversion process fails."""
    pass


__all__ = [
    'AstraKairosError',
    'ConvergenceError', 
    'InvalidOrbitalElementsError',
    'NumericalInstabilityError',
    'InvalidMassInputError',
    'InvalidDataFormatError',
    'CatalogParsingError',
    'FileFormatError',
    'DataValidationError',
    'ElBadryCrossmatchError',
    'ConversionProcessError'
]
