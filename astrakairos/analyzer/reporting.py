# astrakairos/analyzer/reporting.py
"""
Reporting and Display Functions for Binary Star Analysis Results.

This module handles the presentation layer for analysis results, including
console output formatting, error summaries, and result display. It is completely
independent of the analysis logic and can be easily modified for different
output formats.
"""

import logging
from typing import Dict, List, Any

from ..config import (
    CLI_TOP_RESULTS_DISPLAY_COUNT, CLI_DISPLAY_LINE_WIDTH,
    CLI_WDS_ID_COLUMN_WIDTH, CLI_METRIC_COLUMN_WIDTH
)

log = logging.getLogger(__name__)


def format_metric_with_uncertainty(result: dict, metric_key: str, uncertainty_key: str, quality_key: str) -> str:
    """
    Format a metric value with its uncertainty and quality score for CLI display.
    
    Args:
        result: Dictionary containing analysis results
        metric_key: Key for the main metric value
        uncertainty_key: Key for the uncertainty value
        quality_key: Key for the quality score
        
    Returns:
        Formatted string with value ± uncertainty (Q=quality)
    """
    value = result.get(metric_key)
    uncertainty = result.get(uncertainty_key)
    quality = result.get(quality_key)
    
    if value is None:
        return "N/A"
    
    value_str = f"{value:.4f}"
    
    if uncertainty is not None:
        uncertainty_str = f" ± {uncertainty:.4f}"
    else:
        uncertainty_str = ""
    
    if quality is not None and quality > 0:
        quality_str = f" (Q={quality:.2f})"
    else:
        quality_str = ""
        
    return f"{value_str}{uncertainty_str}{quality_str}"


def print_error_summary(total_stars: int, successful_count: int, error_summary: Dict[str, List[str]]) -> None:
    """
    Print a detailed summary of processing results and errors.
    
    Args:
        total_stars: Total number of stars processed
        successful_count: Number of successfully processed stars  
        error_summary: Dictionary mapping error types to affected star IDs
    """
    failed_count = sum(len(star_list) for star_list in error_summary.values())
    
    print(f"\nProcessed {total_stars} stars.")
    print(f"Success: {successful_count}")
    print(f"Failures: {failed_count}")
    
    if error_summary:
        print("\nError breakdown:")
        for error_type, star_ids in error_summary.items():
            print(f"  {error_type}: {len(star_ids)} stars")
            if len(star_ids) <= 5:
                print(f"    {', '.join(star_ids)}")
            else:
                print(f"    {', '.join(star_ids[:3])}, ... and {len(star_ids)-3} more")


def display_results_summary(results: List[Dict[str, Any]], mode: str, sort_key: str) -> None:
    """
    Display a formatted summary of analysis results in the console.
    
    Args:
        results: List of analysis result dictionaries
        mode: Analysis mode (discovery, characterize, orbital)
        sort_key: Key used for sorting results
    """
    if not results:
        print("No results to display.")
        return
    
    print("\n" + "=" * CLI_DISPLAY_LINE_WIDTH)
    print(f"TOP {CLI_TOP_RESULTS_DISPLAY_COUNT} ANALYSIS RESULTS - {mode.upper()} MODE (sorted by {sort_key})")
    print("=" * CLI_DISPLAY_LINE_WIDTH)
    
    for i, result in enumerate(results[:CLI_TOP_RESULTS_DISPLAY_COUNT], 1):
        # Mode-specific display format with uncertainty information
        if sort_key.endswith('_significance'):
            # Display significance value when sorting by significance
            significance_value = result.get(sort_key)
            if significance_value is not None:
                metric_str = f"Significance = {significance_value:.2f}"
            else:
                metric_str = f"Significance = N/A"
        elif mode == 'discovery':
            metric_value = format_metric_with_uncertainty(
                result, 'v_total_median', 'v_total_uncertainty', 'quality_score'
            )
            metric_str = f"V = {metric_value} arcsec/yr"
        elif mode == 'characterize':
            metric_value = format_metric_with_uncertainty(
                result, 'v_total_robust', 'v_total_uncertainty', 'bootstrap_success_rate'
            )
            metric_str = f"V = {metric_value} arcsec/yr"
        elif mode == 'orbital':
            metric_value = format_metric_with_uncertainty(
                result, 'opi_arcsec_yr', 'opi_uncertainty', 'uncertainty_quality'
            )
            metric_str = f"OPI = {metric_value}"
        else:
            metric_str = f"Value = {result.get(sort_key, 'N/A')}"
        
        # Format physicality information
        phys_str = (f"p_val: {result['physicality_p_value']}" 
                   if result.get('physicality_p_value') is not None 
                   else f"Gaia: {result['physicality_label']}")
        
        print(f"{i:2d}. {result['wds_id']:<{CLI_WDS_ID_COLUMN_WIDTH}} | {metric_str:<{CLI_METRIC_COLUMN_WIDTH}} | {phys_str}")
    
    print("-" * CLI_DISPLAY_LINE_WIDTH)


def display_processing_summary(successful_count: int, total_count: int, mode: str) -> None:
    """
    Display a summary of the processing results.
    
    Args:
        successful_count: Number of successfully processed stars
        total_count: Total number of stars attempted
        mode: Analysis mode that was used
    """
    print(f"\nProcessed {successful_count} of {total_count} stars successfully in {mode} mode.")


def log_cache_statistics(gaia_validator) -> None:
    """
    Log cache statistics for transparency in Gaia validation.
    
    Args:
        gaia_validator: The Gaia validator instance to query for statistics
    """
    try:
        cache_stats = gaia_validator.get_cache_statistics()
        if 'cached_systems' in cache_stats and cache_stats['cached_systems'] != 'unknown':
            log.info(f"Validation cache: {cache_stats['cached_systems']} systems pre-computed from El-Badry catalog "
                    f"({cache_stats.get('cache_coverage_percent', 0):.1f}% coverage)")
        else:
            log.info("No El-Badry cache available - will use online-only validation")
    except Exception as e:
        log.warning(f"Could not retrieve cache statistics: {e}")


def format_execution_time(start_time: float, end_time: float) -> str:
    """
    Format execution time for display.
    
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Formatted time string
    """
    execution_time = end_time - start_time
    return f"Total execution time: {execution_time:.2f} seconds"
