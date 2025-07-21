import argparse
import asyncio
import sys
import pandas as pd
import logging
import random
from typing import List, Dict, Any, Optional, Callable
import functools
from astropy.time import Time

from ..data.source import DataSource, WdsSummary, OrbitalElements, PhysicalityAssessment
from ..data.local_source import LocalDataSource
from ..data.gaia_source import GaiaValidator
from ..physics.dynamics import (
    estimate_velocity_from_endpoints, 
    estimate_velocity_from_endpoints_mc,
    calculate_observation_priority_index,
    calculate_observation_priority_index_mc,
    calculate_robust_linear_fit,
    calculate_robust_linear_fit_bootstrap,
    calculate_curvature_index
)
from ..utils.io import load_csv_data, save_results_to_csv
from ..config import (
    DEFAULT_CONCURRENT_REQUESTS, DEFAULT_MIN_OBSERVATIONS,
    DEFAULT_GAIA_P_VALUE, DEFAULT_GAIA_RADIUS_FACTOR,
    DEFAULT_GAIA_MIN_RADIUS, DEFAULT_GAIA_MAX_RADIUS,
    MIN_EPOCH_YEAR, MAX_EPOCH_YEAR, MIN_SEPARATION_ARCSEC, MAX_SEPARATION_ARCSEC,
    MIN_POSITION_ANGLE_DEG, MAX_POSITION_ANGLE_DEG,
    DEFAULT_ANALYSIS_MODE, AVAILABLE_ANALYSIS_MODES, DEFAULT_SORT_KEYS,
    AMBIGUOUS_P_VALUE_RATIO, ENABLE_VALIDATION_WARNINGS, 
    VALIDATION_WARNING_SAMPLE_RATE, ALLOW_SINGLE_EPOCH_SYSTEMS
)

# CLI-specific display constants
TOP_RESULTS_DISPLAY_COUNT = 10
DISPLAY_LINE_WIDTH = 90
WDS_ID_COLUMN_WIDTH = 18
METRIC_COLUMN_WIDTH = 35

log = logging.getLogger(__name__)

# Helper Functions
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

def _should_log_validation_warning() -> bool:
    """Determine if a validation warning should be logged based on sampling rate."""
    return ENABLE_VALIDATION_WARNINGS and random.random() < VALIDATION_WARNING_SAMPLE_RATE

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
            if _should_log_validation_warning():
                log.warning(f"Missing essential field: {field}")
            return False
    
    # Value range validation for existing fields
    if wds_summary.get('date_first') and not (MIN_EPOCH_YEAR <= wds_summary['date_first'] <= MAX_EPOCH_YEAR):
        if _should_log_validation_warning():
            log.warning(f"Invalid first epoch: {wds_summary['date_first']}")
        return False
    
    if wds_summary.get('date_last') and not (MIN_EPOCH_YEAR <= wds_summary['date_last'] <= MAX_EPOCH_YEAR):
        if _should_log_validation_warning():
            log.warning(f"Invalid last epoch: {wds_summary['date_last']}")
        return False
    
    # Temporal consistency check - only if both dates exist
    if wds_summary.get('date_first') and wds_summary.get('date_last'):
        if ALLOW_SINGLE_EPOCH_SYSTEMS:
            # For single epoch systems, allow date_first == date_last
            if wds_summary['date_first'] > wds_summary['date_last']:
                if _should_log_validation_warning():
                    log.warning(f"Invalid epoch sequence: {wds_summary['date_first']} > {wds_summary['date_last']}")
                return False
        else:
            # Original strict validation
            if wds_summary['date_first'] >= wds_summary['date_last']:
                if _should_log_validation_warning():
                    log.warning(f"Invalid epoch sequence: {wds_summary['date_first']} >= {wds_summary['date_last']}")
                return False
    
    # Separation validation - only if fields exist
    if wds_summary.get('sep_first') and not (MIN_SEPARATION_ARCSEC <= wds_summary['sep_first'] <= MAX_SEPARATION_ARCSEC):
        if _should_log_validation_warning():
            log.warning(f"Invalid first separation: {wds_summary['sep_first']}")
        return False
    if wds_summary.get('sep_last') and not (MIN_SEPARATION_ARCSEC <= wds_summary['sep_last'] <= MAX_SEPARATION_ARCSEC):
        if _should_log_validation_warning():
            log.warning(f"Invalid last separation: {wds_summary['sep_last']}")
        return False
    
    # Position angle validation - only if fields exist (normalized to 0-360)
    if wds_summary.get('pa_first'):
        pa_first = wds_summary['pa_first'] % 360
        if not (MIN_POSITION_ANGLE_DEG <= pa_first <= MAX_POSITION_ANGLE_DEG):
            if _should_log_validation_warning():
                log.warning(f"Invalid first position angle: {pa_first}")
            return False
    if wds_summary.get('pa_last'):
        pa_last = wds_summary['pa_last'] % 360
        if not (MIN_POSITION_ANGLE_DEG <= pa_last <= MAX_POSITION_ANGLE_DEG):
            if _should_log_validation_warning():
                log.warning(f"Invalid last position angle: {pa_last}")
            return False
    
    return True

def _get_current_decimal_year() -> float:
    """
    Get current time as decimal year using astropy.time.Time.
    
    Returns:
        float: Current decimal year
    """
    return Time.now().decimalyear

def _calculate_search_radius(wds_summary: WdsSummary, cli_args: argparse.Namespace) -> float:
    """
    Calculate Gaia search radius based on system separation.
    
    Calculates the appropriate search radius for Gaia catalog queries
    based on the last separation measurement and CLI configuration parameters.
    
    Args:
        wds_summary: WDS summary data
        cli_args: CLI arguments with radius configuration
        
    Returns:
        float: Search radius in arcseconds
    """
    if wds_summary.get('sep_last') is not None:
        radius = wds_summary['sep_last'] * cli_args.gaia_radius_factor
        radius = max(radius, cli_args.gaia_min_radius)
        radius = min(radius, cli_args.gaia_max_radius)
        return radius
    
    return DEFAULT_GAIA_MIN_RADIUS

# Analysis Functions

async def _perform_discovery_analysis(wds_id: str, wds_summary: WdsSummary) -> Optional[Dict[str, Any]]:
    """
    Perform discovery mode analysis for basic motion estimation with uncertainty propagation.
    
    Args:
        wds_id: WDS identifier
        wds_summary: WDS summary data
        
    Returns:
        Dict containing discovery analysis results with uncertainties or None if failed
    """
    log.debug(f"Running discovery analysis for {wds_id}")
    
    try:
        # Try Monte Carlo analysis first (if error data available)
        velocity_result = estimate_velocity_from_endpoints_mc(wds_summary)
        
        if velocity_result is None:
            log.debug(f"Monte Carlo analysis failed for {wds_id}, falling back to point estimate")
            # Fallback to point estimate
            velocity_result = estimate_velocity_from_endpoints(wds_summary)
            if velocity_result is None:
                log.error(f"Could not calculate velocity for {wds_id}")
                return None
        
        # Handle both Monte Carlo and point estimate results
        if 'v_total_median' in velocity_result:
            # Monte Carlo result
            result = {
                'v_total_arcsec_yr': velocity_result['v_total_median'],
                'v_total_uncertainty': velocity_result.get('v_total_uncertainty'),
                'pa_v_deg': velocity_result['pa_v_median'],
                'pa_v_uncertainty': velocity_result.get('pa_v_uncertainty'),
                'uncertainty_quality': velocity_result.get('quality_score', 0.0),
                'uncertainty_source': velocity_result.get('uncertainty_source', 'none'),
                'analysis_method': velocity_result.get('method', 'two_point_mc')
            }
        else:
            # Point estimate result
            result = {
                'v_total_arcsec_yr': velocity_result['v_total_estimate'],
                'v_total_uncertainty': None,
                'pa_v_deg': velocity_result['pa_v_estimate'],
                'pa_v_uncertainty': None,
                'uncertainty_quality': 0.0,
                'uncertainty_source': 'none',
                'analysis_method': velocity_result.get('method', 'two_point_estimate')
            }
        
        log.debug(f"Discovery analysis complete: v_total = {result['v_total_arcsec_yr']:.6f} ± {result['v_total_uncertainty'] or 0:.6f} arcsec/year")
        return result
        
    except Exception as e:
        log.error(f"Discovery analysis failed for {wds_id}: {e}")
        return None


async def _perform_characterize_analysis(wds_id: str, data_source: DataSource) -> Optional[Dict[str, Any]]:
    """
    Perform characterize mode analysis with robust fitting and bootstrap uncertainties.
    
    Args:
        wds_id: WDS identifier
        data_source: Data source for measurements
        
    Returns:
        Dict containing characterization results with uncertainties or None if failed
    """
    log.debug(f"Running characterization analysis for {wds_id}")
    
    try:
        # Get all measurements for robust fitting
        all_measurements = await data_source.get_all_measurements(wds_id)
        if not all_measurements or len(all_measurements) < 3:
            log.warning(f"Insufficient measurements for characterization of {wds_id}")
            return None
        
        # Try bootstrap analysis first
        robust_fit = calculate_robust_linear_fit_bootstrap(all_measurements)
        
        if not robust_fit:
            log.error(f"Robust fitting failed for {wds_id}")
            return None
        
        # Handle both bootstrap and standard results
        result = {
            'rmse': robust_fit['rmse'],
            'v_total_robust': robust_fit['v_total_robust'],
            'v_total_uncertainty': robust_fit.get('v_total_uncertainty'),
            'pa_v_robust': robust_fit['pa_v_robust'],
            'pa_v_uncertainty': robust_fit.get('pa_v_uncertainty'),
            'n_measurements': len(all_measurements),
            'time_baseline_years': robust_fit.get('time_baseline_years'),
            'uncertainty_method': robust_fit.get('uncertainty_method', 'none'),
            'bootstrap_success_rate': robust_fit.get('bootstrap_success_rate', 0.0),
            'analysis_method': 'robust_linear_fit'
        }
        
        log.debug(f"Characterization complete: v_total = {result['v_total_robust']:.4f} ± {result['v_total_uncertainty'] or 0:.4f} arcsec/yr")
        return result
        
    except Exception as e:
        log.error(f"Characterization analysis failed for {wds_id}: {e}")
        return None


async def _perform_orbital_analysis(wds_id: str, wds_summary: WdsSummary, data_source: DataSource) -> Optional[Dict[str, Any]]:
    """
    Perform orbital mode analysis with OPI calculation.
    
    Args:
        wds_id: WDS identifier
        wds_summary: WDS summary data
        data_source: Data source for orbital elements
        
    Returns:
        Dict containing orbital analysis results or None if failed
    """
    log.debug(f"Running orbital analysis for {wds_id}")
    
    try:
        # Get orbital elements (required for this mode)
        orbital_elements = await data_source.get_orbital_elements(wds_id)
        if not orbital_elements:
            if _should_log_validation_warning():
                log.warning(f"No orbital elements found for {wds_id}")
            return None
        
        # Calculate OPI with Monte Carlo uncertainty propagation
        current_year = _get_current_decimal_year()
        
        # Try Monte Carlo OPI calculation first
        opi_result = calculate_observation_priority_index_mc(
            orbital_elements, wds_summary, current_year
        )
        
        if opi_result is None:
            log.debug(f"Monte Carlo OPI failed for {wds_id}, falling back to point estimate")
            # Fallback to point estimate
            opi_point = calculate_observation_priority_index(
                orbital_elements, wds_summary, current_year
            )
            if not opi_point:
                log.error(f"OPI calculation failed for {wds_id}")
                return None
            
            opi, deviation = opi_point
            opi_result = {
                'opi_median': opi,
                'opi_uncertainty': None,
                'deviation_median': deviation,
                'deviation_uncertainty': None,
                'quality_score': 0.0,
                'uncertainty_source': 'none',
                'method': 'opi_point'
            }
        
        # Calculate curvature index if measurements available
        curvature_index = None
        try:
            all_measurements = await data_source.get_all_measurements(wds_id)
            if all_measurements and len(all_measurements) >= 3:
                log.debug(f"Calculating curvature for {wds_id} with {len(all_measurements)} measurements")

                linear_fit_results = calculate_robust_linear_fit_bootstrap(all_measurements)
                if linear_fit_results:
                    log.debug(f"Linear fit successful for {wds_id}, calculating curvature index")
                    curvature_index = calculate_curvature_index(
                        orbital_elements, 
                        linear_fit_results, 
                        current_year
                    )
                    if curvature_index is not None:
                        log.debug(f"Curvature index calculated for {wds_id}: {curvature_index:.4f}\"")
                    else:
                        log.warning(f"Curvature index calculation failed for {wds_id}")
                else:
                    log.warning(f"Linear fit failed for {wds_id} with {len(all_measurements)} measurements")
            else:
                log.debug(f"Insufficient measurements for curvature calculation: {wds_id} has {len(all_measurements) if all_measurements else 0} measurements")
        except Exception as e:
            log.warning(f"Curvature calculation failed for {wds_id}: {e}")
        
        result = {
            'opi_arcsec_yr': opi_result['opi_median'],
            'opi_uncertainty': opi_result.get('opi_uncertainty'),
            'deviation_arcsec': opi_result['deviation_median'],
            'deviation_uncertainty': opi_result.get('deviation_uncertainty'),
            'curvature_index': curvature_index,
            'orbital_period': orbital_elements.get('P'),
            'eccentricity': orbital_elements.get('e'),
            'semi_major_axis': orbital_elements.get('a'),
            'uncertainty_quality': opi_result.get('quality_score', 0.0),
            'uncertainty_source': opi_result.get('uncertainty_source', 'none'),
            'analysis_method': opi_result.get('method', 'opi_mc')
        }
        
        log.debug(f"Orbital analysis complete: OPI = {result['opi_arcsec_yr']:.4f} ± {result['opi_uncertainty'] or 0:.4f}")
        return result
    except Exception as e:
        log.error(f"Orbital analysis failed for {wds_id}: {e}")
        return None


async def _perform_gaia_validation(wds_id: str, wds_summary: WdsSummary, 
                                   gaia_validator: GaiaValidator, cli_args: argparse.Namespace) -> Dict[str, Any]:
    """
    Perform Gaia validation for physicality assessment.
    
    Args:
        wds_id: WDS identifier
        wds_summary: WDS summary data
        gaia_validator: Gaia validator instance
        cli_args: CLI arguments for search radius calculation
        
    Returns:
        Dict containing Gaia validation results
    """
    log.debug(f"Running Gaia validation for {wds_id}")
    
    try:
        search_radius = _calculate_search_radius(wds_summary, cli_args)
        
        physicality_assessment = await gaia_validator.validate_physicality(
            wds_summary, search_radius_arcsec=search_radius
        )
        
        if physicality_assessment:
            result = {
                'physicality_p_value': physicality_assessment.get('p_value'),
                'physicality_label': physicality_assessment['label'].value,
                'physicality_method': physicality_assessment['method'].value,
                'physicality_confidence': physicality_assessment.get('confidence')
            }
            log.debug(f"Gaia validation complete: p_value={physicality_assessment.get('p_value')}")
            return result
        else:
            return {
                'physicality_p_value': None,
                'physicality_label': 'Failed',
                'physicality_method': None,
                'physicality_confidence': None
            }
    except Exception as e:
        log.error(f"Gaia validation failed for {wds_id}: {e}")
        return {
            'physicality_p_value': None,
            'physicality_label': 'Error',
            'physicality_method': None,
            'physicality_confidence': None
        }


async def process_star(row: pd.Series,
                       data_source: DataSource,
                       gaia_validator: Optional[GaiaValidator],
                       cli_args: argparse.Namespace,
                       semaphore: asyncio.Semaphore) -> Optional[Dict[str, Any]]:
    """
    Process a single star with mode-specific analysis.
    
    Performs analysis based on the selected mode:
    - discovery: Basic motion analysis for new discoveries
    - characterize: Robust fitting for detailed characterization
    - orbital: OPI calculation for orbital systems
    """
    async with semaphore:
        wds_id = row['wds_id']
        
        try:
            # 1. Always get basic WDS summary data
            wds_summary = await data_source.get_wds_summary(wds_id)
            if not wds_summary:
                log.warning(f"No WDS data found for {wds_id}")
                return None

            # Log with correct observation count
            n_obs = wds_summary.get('n_observations', 'N/A')
            log.info(f"Processing {wds_id} in {cli_args.mode} mode (obs: {n_obs})")

            # Validate data
            if not _validate_wds_summary_for_analysis(wds_summary):
                if _should_log_validation_warning():
                    log.warning(f"Invalid WDS data for {wds_id}")
                return None

            # Initialize result with common fields
            result = {
                'wds_id': wds_id,
                'mode': cli_args.mode,
                'obs_wds': wds_summary.get('n_observations'),
                'date_last': wds_summary.get('date_last', wds_summary.get('date_first')),
            }

            # 2. Mode-specific analysis using specialized functions
            if cli_args.mode == 'discovery':
                analysis_result = await _perform_discovery_analysis(wds_id, wds_summary)
            elif cli_args.mode == 'characterize':
                analysis_result = await _perform_characterize_analysis(wds_id, data_source)
            elif cli_args.mode == 'orbital':
                analysis_result = await _perform_orbital_analysis(wds_id, wds_summary, data_source)
            else:
                log.error(f"Unknown analysis mode: {cli_args.mode}")
                return None
            
            if analysis_result is None:
                return None
                
            result.update(analysis_result)

            # 3. Optional Gaia validation (available for all modes)
            if cli_args.validate_gaia and wds_summary.get('ra_deg') is not None:
                gaia_result = await _perform_gaia_validation(wds_id, wds_summary, gaia_validator, cli_args)
                result.update(gaia_result)
            else:
                result.update({
                    'physicality_p_value': None,
                    'physicality_label': 'Not checked',
                    'physicality_method': None,
                    'physicality_confidence': None
                })

            return result

        except Exception as e:
            log.error(f"Unexpected error processing {wds_id}: {e}")
            return None

async def analyze_stars(df: pd.DataFrame,
                       process_func: Callable,
                       max_concurrent: int = 5) -> List[Dict[str, Any]]:
    """
    Analyzes multiple stars concurrently using a pre-configured processing function.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []
    for _, row in df.iterrows():
        # Call the pre-configured function, passing only the arguments that vary per star
        task = process_func(row=row, semaphore=semaphore)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
    return valid_results

def create_argument_parser():
    """Create command line argument parser with proper defaults from config."""
    parser = argparse.ArgumentParser(
        description='AstraKairos Binary Star Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s stars.csv --database-path catalogs.db --limit 10
  %(prog)s --all --database-path catalogs.db --output results.csv --validate-gaia
        """
    )
    
    # Positional arguments
    parser.add_argument('input_file', nargs='?', default=None,
                    help='CSV file with wds_id column. Omit if using --all.')

    parser.add_argument('--all', action='store_true',
                    help='Analyze all systems in the local database.')
    
    # Data source options - Only local source supported
    parser.add_argument('--source', 
                       choices=['local'], 
                       default='local',
                       help='Data source to use (only local supported)')
    
    parser.add_argument('--database-path',
                       required=True,
                       help='Path to local SQLite catalog database')
    
    # Processing options
    parser.add_argument('--mode', '-m',
                       choices=AVAILABLE_ANALYSIS_MODES,
                       default=DEFAULT_ANALYSIS_MODE,
                       help=f'Analysis mode: discovery (motion analysis), characterize (robust fitting), orbital (OPI calculation) (default: {DEFAULT_ANALYSIS_MODE})')
    
    parser.add_argument('--limit', '-n',
                       type=int,
                       help='Limit number of stars to process')
    
    parser.add_argument('--min-obs',
                       type=int,
                       default=DEFAULT_MIN_OBSERVATIONS,
                       help=f'Minimum number of observations (default: {DEFAULT_MIN_OBSERVATIONS})')
    
    # Output options
    parser.add_argument('--output', '-o',
                       help='Output CSV file for results')
    
    parser.add_argument('--sort-by',
                       help='Sort results by this field (default: mode-specific)')
    
    parser.add_argument('--concurrent',
                       type=int,
                       default=DEFAULT_CONCURRENT_REQUESTS,
                       help=f'Maximum concurrent requests (default: {DEFAULT_CONCURRENT_REQUESTS})')
                       
    gaia_group = parser.add_argument_group('Gaia Validation Options')
    
    gaia_group.add_argument('--validate-gaia',
                       action='store_true',
                       help='Validate physicality using Gaia data (requires network).')
    
    gaia_group.add_argument('--gaia-p-value',
                       type=float,
                       default=DEFAULT_GAIA_P_VALUE,
                       help=f'P-value threshold for physicality test (default: {DEFAULT_GAIA_P_VALUE})')
    
    gaia_group.add_argument('--gaia-radius-factor',
                       type=float,
                       default=DEFAULT_GAIA_RADIUS_FACTOR,
                       help=f'Factor to multiply separation for search radius (default: {DEFAULT_GAIA_RADIUS_FACTOR})')
    
    gaia_group.add_argument('--gaia-min-radius',
                       type=float,
                       default=DEFAULT_GAIA_MIN_RADIUS,
                       help=f'Minimum search radius in arcseconds (default: {DEFAULT_GAIA_MIN_RADIUS})')

    gaia_group.add_argument('--gaia-max-radius',
                       type=float,
                       default=DEFAULT_GAIA_MAX_RADIUS,
                       help=f'Maximum search radius in arcseconds (default: {DEFAULT_GAIA_MAX_RADIUS})')

    return parser

async def main_async(args: argparse.Namespace):
    """
    Main asynchronous function to orchestrate the star analysis process.
    
    This function handles:
    1. Loading and filtering the input data.
    2. Setting up the local data source.
    3. Initializing the GaiaValidator with command-line configurations.
    4. Creating a pre-configured processing function for cleaner execution.
    5. Running the analysis concurrently.
    6. Sorting and presenting the final results.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Set up the data source (only local source supported)
    data_source: DataSource
    if not args.database_path:
        log.error("--database-path is required for local source.")
        log.error("Run: python scripts/convert_catalogs_to_sqlite.py to create the database first.")
        sys.exit(1)
    data_source = LocalDataSource(database_path=args.database_path)

    # 2. Determine target list (from file or --all flag)
    df_targets = None
    if args.all:
        if args.input_file:
            log.warning("Ignoring input_file because --all flag was specified.")
        
        log.info("Fetching all WDS IDs from the local database...")

        all_ids = data_source.get_all_wds_ids()
        if not all_ids:
            log.error("Could not retrieve any WDS IDs from the database.")
            sys.exit(1)
        df_targets = pd.DataFrame(all_ids, columns=['wds_id'])
        log.info(f"Found {len(df_targets)} total systems to analyze.")

    elif args.input_file:
        log.info(f"Loading data from: {args.input_file}")
        try:
            df_targets = load_csv_data(args.input_file)
        except Exception as e:
            log.error(f"Could not load or parse the input file '{args.input_file}': {e}")
            sys.exit(1)
    else:
        log.error("You must specify either an input_file or the --all flag.")
        sys.exit(1)

    # Verify required wds_id column exists
    if 'wds_id' not in df_targets.columns:
        log.error("Input data must contain a 'wds_id' column.")
        sys.exit(1)
        
    # Filter by minimum number of observations if the column exists
    if 'obs' in df_targets.columns and args.min_obs > 0:
        df_filtered = df_targets[df_targets['obs'] >= args.min_obs].copy()
        log.info(f"Filtered to {len(df_filtered)} stars with >= {args.min_obs} observations.")
    else:
        df_filtered = df_targets.copy()
    
    # Apply limit to the number of stars to process if specified
    if args.limit:
        df_filtered = df_filtered.head(args.limit)
        log.info(f"Limiting analysis to a maximum of {len(df_filtered)} stars.")
    
    if df_filtered.empty:
        log.warning("No stars to process after filtering. Exiting.")
        return

    # 3. Initialize the Gaia validator if requested
    gaia_validator = None
    if args.validate_gaia:
        log.info(f"Gaia validation enabled with p-value threshold: {args.gaia_p_value}")
        gaia_validator = GaiaValidator(
            physical_p_value_threshold=args.gaia_p_value,
            ambiguous_p_value_threshold=args.gaia_p_value / AMBIGUOUS_P_VALUE_RATIO
        )

    # 4. Create a pre-configured processing function using functools.partial
    configured_process_star = functools.partial(
        process_star,
        data_source=data_source,
        gaia_validator=gaia_validator,
        cli_args=args
    )
    
    try:
        # 5. Run the concurrent analysis
        log.info(f"Processing {len(df_filtered)} stars with up to {args.concurrent} concurrent tasks...")
        results = await analyze_stars(df_filtered, configured_process_star, args.concurrent)
        
        # 6. Sort and present the results
        if results:
            # Use mode-specific sorting
            sort_key = args.sort_by if args.sort_by else DEFAULT_SORT_KEYS[args.mode]
            log.info(f"Sorting results by: {sort_key}")
            
            results_sorted = sorted(results, 
                                  key=lambda x: x.get(sort_key, -1) or -1, 
                                  reverse=True)
            
            # Display summary in the console
            print("\n" + "=" * DISPLAY_LINE_WIDTH)
            print(f"TOP {TOP_RESULTS_DISPLAY_COUNT} ANALYSIS RESULTS - {args.mode.upper()} MODE (sorted by {sort_key})")
            print("=" * DISPLAY_LINE_WIDTH)
            
            for i, result in enumerate(results_sorted[:TOP_RESULTS_DISPLAY_COUNT], 1):
                # Mode-specific display format with uncertainty information
                if args.mode == 'discovery':
                    metric_value = format_metric_with_uncertainty(
                        result, 'v_total_arcsec_yr', 'v_total_uncertainty', 'uncertainty_quality'
                    )
                    metric_str = f"V = {metric_value} arcsec/yr"
                elif args.mode == 'characterize':
                    metric_value = format_metric_with_uncertainty(
                        result, 'v_total_robust', 'v_total_uncertainty', 'bootstrap_success_rate'
                    )
                    metric_str = f"V = {metric_value} arcsec/yr"
                elif args.mode == 'orbital':
                    metric_value = format_metric_with_uncertainty(
                        result, 'opi_arcsec_yr', 'opi_uncertainty', 'uncertainty_quality'
                    )
                    metric_str = f"OPI = {metric_value}"
                else:
                    metric_str = f"Value = {result.get(sort_key, 'N/A')}"
                
                phys_str = f"p_val: {result['physicality_p_value']}" if result.get('physicality_p_value') is not None else f"Gaia: {result['physicality_label']}"
                print(f"{i:2d}. {result['wds_id']:<{WDS_ID_COLUMN_WIDTH}} | {metric_str:<{METRIC_COLUMN_WIDTH}} | {phys_str}")
            
            print("-" * DISPLAY_LINE_WIDTH)
            print(f"\nProcessed {len(results)} of {len(df_filtered)} stars successfully in {args.mode} mode.")
            
            # Save full results to a CSV file if requested
            if args.output:
                save_results_to_csv(results_sorted, args.output)
                log.info(f"Results saved to {args.output}")
        else:
            log.warning("No stars were successfully processed.")
            
    finally:
        # 7. Clean up resources
        if args.source == 'local' and 'data_source' in locals() and hasattr(data_source, 'close'):
            try:
                data_source.close()
                log.debug("Local database connection closed.")
            except Exception as e:
                log.warning(f"Error closing local database connection: {e}")

def main(args_list: Optional[List[str]] = None):
    """Main entry point for the analyzer CLI.
    
    Args:
        args_list: Optional list of command line arguments. 
                  If None, will parse from sys.argv
    """
    parser = create_argument_parser()
    args = parser.parse_args(args_list)
    
    # Run async main
    import time
    start_time = time.time()
    
    asyncio.run(main_async(args))
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()