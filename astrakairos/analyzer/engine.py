# astrakairos/analyzer/engine.py
"""
Analysis Engine for Binary Star Processing.

This module contains the core analysis engine that orchestrates binary star
processing with proper dependency injection and error handling. It is completely
independent of CLI concerns and can be reused in other contexts.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import argparse

from ..data.source import (
    DataSource, WdsSummary, 
    WdsIdNotFoundError
)
from ..data.validators import HybridValidator
from ..exceptions import (
    DataSourceError, ConfigurationError, AnalysisError, 
    ValidationError, StarProcessingError, PhysicalityValidationError
)
from ..config import (
    CLI_RESULT_KEYS, CLI_STATS_KEYS, CLI_VALUE_NOT_AVAILABLE,
    MIN_EPOCH_YEAR, MAX_EPOCH_YEAR, MIN_SEPARATION_ARCSEC, MAX_SEPARATION_ARCSEC,
    MIN_POSITION_ANGLE_DEG, MAX_POSITION_ANGLE_DEG
)

log = logging.getLogger(__name__)


def _validate_wds_summary_for_analysis(wds_summary: WdsSummary) -> bool:
    """
    Validate WDS summary data for analysis.
    
    Performs validation of WDS summary data including:
    - Required field presence
    - Value range validation
    - Temporal consistency checks (relaxed for WDSS single-epoch systems)
    
    Args:
        wds_summary: WDS summary data to validate
        
    Returns:
        bool: True if data is valid for analysis, False otherwise
    """
    # Minimal required fields for any analysis
    essential_fields = ['wds_id', 'date_first']
    
    # Check essential fields exist and are not None
    for field in essential_fields:
        if field not in wds_summary or wds_summary[field] is None:
            log.warning(f"Missing essential field '{field}' in WDS summary")
            return False
    
    # Value range validation for existing fields
    if wds_summary.get('date_first') and not (MIN_EPOCH_YEAR <= wds_summary['date_first'] <= MAX_EPOCH_YEAR):
        log.warning(f"date_first {wds_summary['date_first']} out of valid range [{MIN_EPOCH_YEAR}, {MAX_EPOCH_YEAR}]")
        return False
    
    if wds_summary.get('date_last') and not (MIN_EPOCH_YEAR <= wds_summary['date_last'] <= MAX_EPOCH_YEAR):
        log.warning(f"date_last {wds_summary['date_last']} out of valid range [{MIN_EPOCH_YEAR}, {MAX_EPOCH_YEAR}]")
        return False
    
    # Temporal consistency check - only if both dates exist
    if wds_summary.get('date_first') and wds_summary.get('date_last'):
        if wds_summary['date_last'] < wds_summary['date_first']:
            log.warning(f"date_last {wds_summary['date_last']} < date_first {wds_summary['date_first']}")
            return False
    
    # Separation validation - only if fields exist
    if wds_summary.get('sep_first') and not (MIN_SEPARATION_ARCSEC <= wds_summary['sep_first'] <= MAX_SEPARATION_ARCSEC):
        log.warning(f"sep_first {wds_summary['sep_first']} out of valid range")
        return False
    if wds_summary.get('sep_last') and not (MIN_SEPARATION_ARCSEC <= wds_summary['sep_last'] <= MAX_SEPARATION_ARCSEC):
        log.warning(f"sep_last {wds_summary['sep_last']} out of valid range")
        return False
    
    # Position angle validation - only if fields exist (normalized to 0-360)
    if wds_summary.get('pa_first'):
        pa_first_norm = wds_summary['pa_first'] % 360
        if not (MIN_POSITION_ANGLE_DEG <= pa_first_norm <= MAX_POSITION_ANGLE_DEG):
            log.warning(f"pa_first {wds_summary['pa_first']} out of valid range after normalization")
            return False
    if wds_summary.get('pa_last'):
        pa_last_norm = wds_summary['pa_last'] % 360
        if not (MIN_POSITION_ANGLE_DEG <= pa_last_norm <= MAX_POSITION_ANGLE_DEG):
            log.warning(f"pa_last {wds_summary['pa_last']} out of valid range after normalization")
            return False
    
    return True


class AnalyzerRunner:
    """
    Main analyzer runner that handles dependency injection and configuration management.
    
    This class encapsulates all dependencies and provides a clean interface for running
    the analysis with proper error handling and resource management. It is completely
    independent of CLI concerns and can be reused in web APIs or other contexts.
    """
    
    def __init__(self, data_source: DataSource, gaia_validator: Optional[HybridValidator] = None):
        """
        Initialize the analyzer runner.
        
        Args:
            data_source: Data source for astronomical data
            gaia_validator: Optional Hybrid validator for physicality checks
        """
        self.data_source = data_source
        self.gaia_validator = gaia_validator
        self._stats = {
            CLI_STATS_KEYS['PROCESSED']: 0,
            CLI_STATS_KEYS['SUCCESSFUL']: 0,
            CLI_STATS_KEYS['FAILED']: 0,
            CLI_STATS_KEYS['VALIDATION_ERRORS']: 0,
            CLI_STATS_KEYS['ANALYSIS_ERRORS']: 0,
            CLI_STATS_KEYS['DATA_ERRORS']: 0
        }
    
    async def process_star(self, row: pd.Series, analysis_config: Dict[str, Any],
                          semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """
        Process a single star with comprehensive error handling and statistics tracking.
        
        This method contains the main analysis workflow for a single star,
        decomposed into logical steps for maintainability. It properly propagates
        exceptions for detailed error reporting.
        
        Args:
            row: DataFrame row containing star data with 'wds_id'
            analysis_config: Analysis configuration dictionary containing:
                - mode: Analysis mode ('discovery', 'characterize', 'orbital')
                - validate_gaia: Whether to perform Gaia validation
                - calculate_masses: Whether to calculate masses (orbital mode)
                - parallax_source: Parallax source preference
                - gaia_radius_factor: Factor for Gaia search radius
                - gaia_min_radius: Minimum Gaia search radius
                - gaia_max_radius: Maximum Gaia search radius
            semaphore: Concurrency control semaphore
            
        Returns:
            Analysis result dictionary
            
        Raises:
            ValidationError: When data validation fails
            AnalysisError: When analysis computation fails  
            DataSourceError: When data retrieval fails
            PhysicalityValidationError: When Gaia validation fails
        """
        async with semaphore:
            wds_id = row[CLI_RESULT_KEYS['WDS_ID']]
            self._stats[CLI_STATS_KEYS['PROCESSED']] += 1
            
            try:
                # Step 1: Retrieve and validate WDS data
                wds_summary = await self._get_and_validate_wds_data(wds_id)
                
                # Step 2: Log processing information
                n_obs = wds_summary.get(CLI_RESULT_KEYS['N_OBSERVATIONS'], CLI_VALUE_NOT_AVAILABLE)
                log.info(f"Processing {wds_id} in {analysis_config['mode']} mode (obs: {n_obs})")

                # Step 3: Initialize result with common fields
                result = self._initialize_result_dict(wds_id, wds_summary, analysis_config)

                # Step 4: Perform mode-specific analysis
                analysis_result = await self._run_analysis_mode(wds_id, wds_summary, analysis_config)
                if analysis_result is None:
                    raise AnalysisError(f"Mode-specific analysis failed for {wds_id}")
                result.update(analysis_result)

                # Step 5: Optional Gaia validation
                gaia_result = await self._perform_optional_gaia_validation(
                    wds_id, wds_summary, analysis_config
                )
                result.update(gaia_result)

                self._stats[CLI_STATS_KEYS['SUCCESSFUL']] += 1
                return result
                
            except (ValidationError, AnalysisError, DataSourceError, PhysicalityValidationError):
                # These are expected domain-specific exceptions that should be categorized
                self._stats[CLI_STATS_KEYS['FAILED']] += 1
                raise
            except Exception as e:
                # Unexpected errors get wrapped and re-raised for debugging
                self._stats[CLI_STATS_KEYS['FAILED']] += 1
                raise StarProcessingError(f"Critical error processing {wds_id}") from e
    
    async def _get_and_validate_wds_data(self, wds_id: str) -> Dict[str, Any]:
        """Retrieve and validate WDS data for a given star."""
        try:
            wds_summary = await self.data_source.get_wds_summary(wds_id)
            if not wds_summary:
                raise ValidationError(f"No WDS data found for {wds_id}")
            
            # Validate the WDS summary data
            if not _validate_wds_summary_for_analysis(wds_summary):
                raise ValidationError(f"WDS data validation failed for {wds_id}")
                
            return wds_summary
        except WdsIdNotFoundError as e:
            raise ValidationError(f"WDS ID not found: {wds_id}") from e
        except Exception as e:
            raise ValidationError(f"Failed to retrieve WDS data for {wds_id}") from e
    
    def _initialize_result_dict(self, wds_id: str, wds_summary: Dict[str, Any], 
                               analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the result dictionary with common fields."""
        from ..config import CLI_DEFAULT_PHYSICALITY_VALUES
        
        result = {
            CLI_RESULT_KEYS['WDS_ID']: wds_id,
            CLI_RESULT_KEYS['ANALYSIS_MODE']: analysis_config['mode'],
            CLI_RESULT_KEYS['N_OBSERVATIONS']: wds_summary.get('n_obs', CLI_VALUE_NOT_AVAILABLE),
            CLI_RESULT_KEYS['DATE_RANGE_YEARS']: wds_summary.get('date_range_years', CLI_VALUE_NOT_AVAILABLE)
        }
        
        # Initialize physicality fields with defaults
        result.update(CLI_DEFAULT_PHYSICALITY_VALUES)
        
        return result
    
    async def _perform_optional_gaia_validation(self, wds_id: str, wds_summary: Dict[str, Any], 
                                              analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform optional Gaia validation if enabled."""
        gaia_result = {}
        
        if analysis_config.get('validate_gaia', False) and self.gaia_validator:
            try:
                from .workflows import _perform_gaia_validation
                gaia_result = await _perform_gaia_validation(
                    wds_id, wds_summary, self.gaia_validator, analysis_config
                )
            except PhysicalityValidationError as e:
                log.warning(f"Gaia validation failed for {wds_id}: {e}")
                # Use default values for failed validation
                from ..config import CLI_DEFAULT_PHYSICALITY_VALUES
                gaia_result = CLI_DEFAULT_PHYSICALITY_VALUES.copy()
        
        return gaia_result
    
    async def _run_analysis_mode(self, wds_id: str, wds_summary: WdsSummary, 
                                analysis_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run the analysis workflow for the specified mode."""
        from .workflows import (
            _perform_discovery_analysis, 
            _perform_characterize_analysis, 
            _perform_orbital_analysis
        )
        
        if analysis_config['mode'] == 'discovery':
            return await _perform_discovery_analysis(wds_id, wds_summary, self.data_source)
        elif analysis_config['mode'] == 'characterize':
            return await _perform_characterize_analysis(wds_id, self.data_source)
        elif analysis_config['mode'] == 'orbital':
            return await _perform_orbital_analysis(
                wds_id, wds_summary, self.data_source,
                gaia_validator=self.gaia_validator,
                calculate_masses=analysis_config.get('calculate_masses', False),
                parallax_source=analysis_config.get('parallax_source', 'auto')
            )
        else:
            raise AnalysisError(f"Unknown analysis mode: {analysis_config['mode']}")
    
    def get_processing_statistics(self) -> Dict[str, int]:
        """Get current processing statistics."""
        return self._stats.copy()
    
    def log_processing_summary(self):
        """Log a summary of processing statistics."""
        stats = self.get_processing_statistics()
        log.info(f"Processing summary: {stats[CLI_STATS_KEYS['PROCESSED']]} processed, "
                f"{stats[CLI_STATS_KEYS['SUCCESSFUL']]} successful, "
                f"{stats[CLI_STATS_KEYS['FAILED']]} failed")


async def analyze_stars(runner: AnalyzerRunner,
                       df: pd.DataFrame,
                       analysis_config: Dict[str, Any],
                       max_concurrent: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Analyzes multiple stars concurrently using AnalyzerRunner.
    
    Args:
        runner: Configured AnalyzerRunner instance
        df: DataFrame with star data
        analysis_config: Analysis configuration dictionary
        max_concurrent: Maximum concurrent tasks
        
    Returns:
        Tuple of (successful_results, error_summary)
        where error_summary is a dict mapping error types to lists of affected star IDs
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []
    for _, row in df.iterrows():
        # Use AnalyzerRunner which properly propagates exceptions
        task = runner.process_star(row, analysis_config, semaphore)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Separate successful results from exceptions
    successful_results = []
    error_summary = {}
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Get the WDS ID for this row
            wds_id = df.iloc[i].get(CLI_RESULT_KEYS['WDS_ID'], f"Row_{i}")
            
            # Categorize the exception
            error_type = type(result).__name__
            if error_type not in error_summary:
                error_summary[error_type] = []
            error_summary[error_type].append(wds_id)
            
            # Log the specific error for debugging
            log.error(f"Failed to process {wds_id}: {error_type}: {result}")
            
        elif result is not None:
            successful_results.append(result)
    
    return successful_results, error_summary
