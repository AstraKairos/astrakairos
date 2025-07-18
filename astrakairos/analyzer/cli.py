import argparse
import asyncio
import sys
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Callable
import functools
from astropy.time import Time

from ..data.source import DataSource, WdsSummary, OrbitalElements, PhysicalityAssessment
from ..data.local_source import LocalDataSource
from ..data.online_source import OnlineDataSource
from ..data.gaia_source import GaiaValidator
from ..physics.dynamics import estimate_velocity_from_endpoints, calculate_observation_priority_index
from ..physics.kepler import predict_position
from ..utils.io import load_csv_data, save_results_to_csv
from ..config import (
    DEFAULT_CONCURRENT_REQUESTS, DEFAULT_MIN_OBSERVATIONS,
    DEFAULT_SORT_BY, DEFAULT_GAIA_P_VALUE, DEFAULT_GAIA_RADIUS_FACTOR,
    DEFAULT_GAIA_MIN_RADIUS, DEFAULT_GAIA_MAX_RADIUS,
    MIN_EPOCH_YEAR, MAX_EPOCH_YEAR, MIN_SEPARATION_ARCSEC, MAX_SEPARATION_ARCSEC,
    MIN_POSITION_ANGLE_DEG, MAX_POSITION_ANGLE_DEG,
    DEFAULT_ANALYSIS_MODE, AVAILABLE_ANALYSIS_MODES, DEFAULT_SORT_KEYS
)

log = logging.getLogger(__name__)

# Scientific Helper Functions
def _validate_wds_summary_for_analysis(wds_summary: WdsSummary) -> bool:
    """
    Validate WDS summary data for scientific analysis.
    
    Performs comprehensive validation of WDS summary data including:
    - Required field presence
    - Scientific value range validation
    - Temporal consistency checks
    
    Args:
        wds_summary: WDS summary data to validate
        
    Returns:
        bool: True if data is valid for analysis, False otherwise
    """
    required_fields = ['date_first', 'date_last', 'pa_first', 'pa_last', 'sep_first', 'sep_last']
    
    # Check required fields exist and are not None
    for field in required_fields:
        if field not in wds_summary or wds_summary[field] is None:
            log.warning(f"Missing required field: {field}")
            return False
    
    # Scientific validation
    if not (MIN_EPOCH_YEAR <= wds_summary['date_first'] <= MAX_EPOCH_YEAR):
        log.warning(f"Invalid first epoch: {wds_summary['date_first']}")
        return False
    if not (MIN_EPOCH_YEAR <= wds_summary['date_last'] <= MAX_EPOCH_YEAR):
        log.warning(f"Invalid last epoch: {wds_summary['date_last']}")
        return False
    if wds_summary['date_first'] >= wds_summary['date_last']:
        log.warning(f"Invalid epoch sequence: {wds_summary['date_first']} >= {wds_summary['date_last']}")
        return False
    
    # Separation validation
    if not (MIN_SEPARATION_ARCSEC <= wds_summary['sep_first'] <= MAX_SEPARATION_ARCSEC):
        log.warning(f"Invalid first separation: {wds_summary['sep_first']}")
        return False
    if not (MIN_SEPARATION_ARCSEC <= wds_summary['sep_last'] <= MAX_SEPARATION_ARCSEC):
        log.warning(f"Invalid last separation: {wds_summary['sep_last']}")
        return False
    
    # Position angle validation (normalized to 0-360)
    pa_first = wds_summary['pa_first'] % 360
    pa_last = wds_summary['pa_last'] % 360
    if not (MIN_POSITION_ANGLE_DEG <= pa_first <= MAX_POSITION_ANGLE_DEG):
        log.warning(f"Invalid first position angle: {pa_first}")
        return False
    if not (MIN_POSITION_ANGLE_DEG <= pa_last <= MAX_POSITION_ANGLE_DEG):
        log.warning(f"Invalid last position angle: {pa_last}")
        return False
    
    return True

def _get_current_decimal_year() -> float:
    """
    Get current time as decimal year using astronomical standard.
    
    Uses astropy.time.Time for maximum precision, which is the gold standard
    in astronomy for epoch calculations.
    
    Returns:
        float: Current decimal year
    """
    return Time.now().decimalyear

def _calculate_search_radius(wds_summary: WdsSummary, cli_args: argparse.Namespace) -> float:
    """
    Calculate Gaia search radius using scientific constraints.
    
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

def _build_analysis_result(wds_id: str, wds_summary: WdsSummary, 
                          orbital_elements: Optional[OrbitalElements],
                          v_total: float, pa_v: float, 
                          opi: Optional[float], deviation: Optional[float],
                          physicality_assessment: Optional[PhysicalityAssessment]) -> Dict[str, Any]:
    """
    Build structured analysis result with proper type handling.
    
    Args:
        wds_id: WDS identifier
        wds_summary: WDS summary data
        orbital_elements: Orbital elements if available
        v_total: Total velocity
        pa_v: Position angle of velocity
        opi: Observation priority index
        deviation: Orbital deviation
        physicality_assessment: Physicality assessment result
        
    Returns:
        Dict[str, Any]: Structured analysis result
    """
    result = {
        'wds_id': wds_id,
        'obs_wds': wds_summary.get('obs'),
        'date_last': wds_summary['date_last'],
        'v_total_arcsec_yr': v_total,
        'pa_v_deg': pa_v,
        'opi_arcsec_yr': opi,
        'deviation_arcsec': deviation,
    }
    
    # Add physicality assessment with proper enum handling
    if physicality_assessment:
        result.update({
            'physicality_label': physicality_assessment['label'].value,
            'physicality_p_value': physicality_assessment.get('p_value'),
            'physicality_method': physicality_assessment['method'].value,
            'physicality_confidence': physicality_assessment.get('confidence')
        })
    else:
        result.update({
            'physicality_label': 'Not checked',
            'physicality_p_value': None,
            'physicality_method': None,
            'physicality_confidence': None
        })
    
    # Add orbital elements
    if orbital_elements:
        result.update({f'orb_{k}': v for k, v in orbital_elements.items()})
    
    return result

# En astrakairos/analyzer/cli.py

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
        log.info(f"Processing {wds_id} in {cli_args.mode} mode (obs: {row.get('obs', 'N/A')})")

        try:
            # 1. Always get basic WDS summary data
            wds_summary = await data_source.get_wds_summary(wds_id)
            if not wds_summary:
                log.warning(f"No WDS data found for {wds_id}")
                return None

            # FIXED: Validate data scientifically
            if not _validate_wds_summary_for_analysis(wds_summary):
                log.warning(f"Invalid WDS data for {wds_id}")
                return None

            # Initialize result with common fields
            result = {
                'wds_id': wds_id,
                'mode': cli_args.mode,
                'obs_wds': wds_summary.get('obs'),
                'date_last': wds_summary['date_last'],
            }

            # 2. Mode-specific analysis
            if cli_args.mode == 'discovery':
                # Discovery mode: Basic motion analysis
                log.debug(f"Running discovery analysis for {wds_id}")
                
                try:
                    velocity_result = estimate_velocity_from_endpoints(wds_summary)
                    if velocity_result is None:
                        log.error(f"Could not calculate velocity for {wds_id}")
                        return None
                    
                    v_total = velocity_result['v_total_estimate']
                    pa_v = velocity_result['pa_v_estimate']
                    
                    result.update({
                        'v_total_arcsec_yr': v_total,
                        'pa_v_deg': pa_v,
                    })
                    log.debug(f"Discovery analysis complete: v_total = {v_total:.6f} arcsec/year")
                except Exception as e:
                    log.error(f"Discovery analysis failed for {wds_id}: {e}")
                    return None

            elif cli_args.mode == 'characterize':
                # Characterize mode: Robust fitting analysis
                log.debug(f"Running characterization analysis for {wds_id}")
                
                try:
                    # Get all measurements for robust fitting
                    all_measurements = await data_source.get_all_measurements(wds_id)
                    if not all_measurements or len(all_measurements) < 3:
                        log.warning(f"Insufficient measurements for characterization of {wds_id}")
                        return None
                    
                    # Import robust fitting function
                    from ..physics.dynamics import calculate_robust_linear_fit
                    
                    robust_fit = calculate_robust_linear_fit(all_measurements)
                    if not robust_fit:
                        log.error(f"Robust fitting failed for {wds_id}")
                        return None
                    
                    result.update({
                        'rmse': robust_fit['rmse'],
                        'v_total_robust': robust_fit['v_total'],
                        'pa_v_robust': robust_fit['pa_v'],
                        'n_measurements': len(all_measurements),
                        'fit_quality': robust_fit.get('quality', 'unknown')
                    })
                    log.debug(f"Characterization complete: RMSE = {robust_fit['rmse']:.4f}")
                except Exception as e:
                    log.error(f"Characterization analysis failed for {wds_id}: {e}")
                    return None

            elif cli_args.mode == 'orbital':
                # Orbital mode: OPI calculation
                log.debug(f"Running orbital analysis for {wds_id}")
                
                try:
                    # Get orbital elements (required for this mode)
                    orbital_elements = await data_source.get_orbital_elements(wds_id)
                    if not orbital_elements:
                        log.warning(f"No orbital elements found for {wds_id}")
                        return None
                    
                    # Calculate OPI
                    current_year = _get_current_decimal_year()
                    opi_result = calculate_observation_priority_index(
                        orbital_elements, wds_summary, current_year
                    )
                    
                    if not opi_result:
                        log.error(f"OPI calculation failed for {wds_id}")
                        return None
                    
                    opi, deviation = opi_result
                    
                    # Calculate curvature index if measurements available
                    curvature_index = None
                    try:
                        all_measurements = await data_source.get_all_measurements(wds_id)
                        if all_measurements and len(all_measurements) >= 3:
                            from ..physics.dynamics import calculate_curvature_index
                            curvature_index = calculate_curvature_index(all_measurements, orbital_elements)
                    except Exception as e:
                        log.debug(f"Curvature calculation failed for {wds_id}: {e}")
                    
                    result.update({
                        'opi_arcsec_yr': opi,
                        'deviation_arcsec': deviation,
                        'curvature_index': curvature_index,
                        'orbital_period': orbital_elements.get('P'),
                        'eccentricity': orbital_elements.get('e'),
                        'semi_major_axis': orbital_elements.get('a')
                    })
                    log.debug(f"Orbital analysis complete: OPI = {opi:.4f}")
                except Exception as e:
                    log.error(f"Orbital analysis failed for {wds_id}: {e}")
                    return None

            # 3. Optional Gaia validation (available for all modes)
            if cli_args.validate_gaia and wds_summary.get('ra_deg') is not None:
                log.debug(f"Running Gaia validation for {wds_id}")
                
                try:
                    search_radius = _calculate_search_radius(wds_summary, cli_args)
                    
                    physicality_assessment = await gaia_validator.validate_physicality(
                        wds_summary, search_radius_arcsec=search_radius
                    )
                    
                    if physicality_assessment:
                        result.update({
                            'physicality_label': physicality_assessment['label'].value,
                            'physicality_p_value': physicality_assessment.get('p_value'),
                            'physicality_method': physicality_assessment['method'].value,
                            'physicality_confidence': physicality_assessment.get('confidence')
                        })
                        log.debug(f"Gaia validation complete: {physicality_assessment['label']}")
                    else:
                        result.update({
                            'physicality_label': 'Failed',
                            'physicality_p_value': None,
                            'physicality_method': None,
                            'physicality_confidence': None
                        })
                except Exception as e:
                    log.error(f"Gaia validation failed for {wds_id}: {e}")
                    result.update({
                        'physicality_label': 'Error',
                        'physicality_p_value': None,
                        'physicality_method': None,
                        'physicality_confidence': None
                    })
            else:
                result.update({
                    'physicality_label': 'Not checked',
                    'physicality_p_value': None,
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
  %(prog)s stars.csv --source web --limit 10
  %(prog)s stars.csv --source local --database-path catalogs.db
  %(prog)s stars.csv --source web --output results.csv --validate-gaia
        """
    )
    
    # Positional arguments
    parser.add_argument('input_file', 
                       help='CSV file containing star list with wds_id column')
    
    # Data source options
    parser.add_argument('--source', 
                       choices=['web', 'local'], 
                       default='web',
                       help='Data source to use (default: web)')
    
    parser.add_argument('--database-path',
                       help='Path to local SQLite catalog database (required for local source)')
    
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
    2. Setting up the appropriate data source (local or web).
    3. Initializing the GaiaValidator with command-line configurations.
    4. Creating a pre-configured processing function for cleaner execution.
    5. Running the analysis concurrently.
    6. Sorting and presenting the final results.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # 1. Load and filter input data from CSV
    log.info(f"Loading data from: {args.input_file}")
    try:
        df = load_csv_data(args.input_file)
    except Exception as e:
        log.error(f"Could not load or parse the input file '{args.input_file}': {e}")
        sys.exit(1)
    
    # FIXED: Check for wds_id column
    if 'wds_id' not in df.columns:
        log.error("Input file must contain a 'wds_id' column")
        sys.exit(1)
        
    # Filter by minimum number of observations if the column exists
    if 'obs' in df.columns and args.min_obs > 0:
        df_filtered = df[df['obs'] >= args.min_obs].copy()
        log.info(f"Filtered to {len(df_filtered)} stars with >= {args.min_obs} observations")
    else:
        df_filtered = df.copy()
    
    # Apply limit to the number of stars to process if specified
    if args.limit:
        df_filtered = df_filtered.head(args.limit)
        log.info(f"Limited to processing a maximum of {len(df_filtered)} stars")
    
    if df_filtered.empty:
        log.warning("No stars to process after filtering. Exiting.")
        return

    # 2. Set up the data source
    data_source: DataSource
    session = None
    if args.source == 'local':
        if not args.database_path:
            log.error("--database-path is required for local source")
            log.error("Run: python scripts/convert_catalogs_to_sqlite.py to create the database first")
            sys.exit(1)
            
        data_source = LocalDataSource(database_path=args.database_path)
    else:  # 'web' source
        data_source = OnlineDataSource()
    
    # 3. Initialize the Gaia validator if requested
    gaia_validator = None
    if args.validate_gaia:
        log.info(f"Gaia validation enabled with p-value threshold: {args.gaia_p_value}")
        # FIXED: Use correct GaiaValidator constructor
        gaia_validator = GaiaValidator(
            physical_p_value_threshold=args.gaia_p_value,
            ambiguous_p_value_threshold=args.gaia_p_value / 10  # Standard ratio
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
        log.info(f"Processing {len(df_filtered)} stars with up to {args.concurrent} concurrent tasks")
        results = await analyze_stars(df_filtered, configured_process_star, args.concurrent)
        
        # 6. Sort and present the results
        if results:
            # Use mode-specific sorting
            sort_key = args.sort_by if args.sort_by else DEFAULT_SORT_KEYS[args.mode]
            log.info(f"Sorting results by: {sort_key}")
            
            # Handle different sorting requirements
            if sort_key == 'opi_arcsec_yr':
                # For OPI, None values should go to the end
                results_sorted = sorted(results, key=lambda x: x.get(sort_key, -1) or -1, reverse=True)
            else:
                # For other numeric fields, use standard sorting
                results_sorted = sorted(results, 
                                      key=lambda x: x.get(sort_key, 0) if x.get(sort_key) is not None else 0, 
                                      reverse=True)
            
            # Display summary in the console
            print("\n" + "="*80)
            print(f"TOP 10 ANALYSIS RESULTS - {args.mode.upper()} MODE (sorted by {sort_key})")
            print("="*80)
            
            for i, result in enumerate(results_sorted[:10], 1):
                # Mode-specific display format
                if args.mode == 'discovery':
                    metric_str = f"Velocity = {result.get('v_total_arcsec_yr', 0):.6f} arcsec/yr"
                elif args.mode == 'characterize':
                    metric_str = f"RMSE = {result.get('rmse', 0):.4f}"
                elif args.mode == 'orbital':
                    metric_str = f"OPI = {result.get('opi_arcsec_yr', 0):.4f}"
                else:
                    metric_str = f"Value = {result.get(sort_key, 'N/A')}"
                
                phys_str = f"Gaia: {result['physicality_label']}" if result.get('physicality_label') != 'Not checked' else ''
                print(f"{i:2d}. {result['wds_id']:<18} | {metric_str:<25} | {phys_str}")
            
            print("-" * 80)
            print(f"\nProcessed {len(results)} of {len(df_filtered)} stars successfully in {args.mode} mode.")
            
            # Save full results to a CSV file if requested
            if args.output:
                save_results_to_csv(results_sorted, args.output)
                log.info(f"Results saved to {args.output}")
        else:
            log.warning("No stars were successfully processed")
            
    finally:
        # 7. Clean up resources
        pass  # No cleanup needed for current implementation

def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV data with proper error handling and validation.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    try:
        df = pd.read_csv(filepath)
        log.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

def save_results_to_csv(results: List[Dict[str, Any]], output_path: str):
    """
    Save analysis results to CSV file with proper formatting.
    
    Args:
        results: List of analysis results
        output_path: Path to output CSV file
    """
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        log.info(f"Saved {len(results)} results to {output_path}")
    except Exception as e:
        log.error(f"Failed to save results to {output_path}: {e}")
        raise

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