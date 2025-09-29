import argparse
import asyncio
import sys
import pandas as pd
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

from ..data.source import DataSource
from ..data.local_source import LocalDataSource
from ..data.gaia_source import GaiaValidator
from ..data.validators import HybridValidator
from ..exceptions import ConfigurationError
from ..config import (
    CLI_RESULT_KEYS, DEFAULT_CONCURRENT_REQUESTS, DEFAULT_MIN_OBS,
    DEFAULT_GAIA_P_VALUE,
    DEFAULT_ANALYSIS_MODE, AVAILABLE_ANALYSIS_MODES, DEFAULT_SORT_KEYS,
    AMBIGUOUS_P_VALUE_RATIO, DEFAULT_MC_SAMPLES,
    CLI_TOP_RESULTS_DISPLAY_COUNT, CLI_DISPLAY_LINE_WIDTH,
    MIN_VELOCITY_UNCERTAINTY_ARCSEC_YR, MIN_OPI_UNCERTAINTY, MAX_REASONABLE_SIGNIFICANCE,
    CLI_WDS_ID_COLUMN_WIDTH, CLI_METRIC_COLUMN_WIDTH,
    CLI_VALID_SORT_KEYS
)
from ..utils.io import load_csv_data, save_results_to_csv
from .engine import AnalyzerRunner, analyze_stars
from .reporting import format_metric_with_uncertainty, print_error_summary

log = logging.getLogger(__name__)


def _create_analysis_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Create analysis configuration dictionary from CLI arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Analysis configuration dictionary for AnalyzerRunner
    """
    return {
        'mode': args.mode,
        'validate_gaia': args.validate_gaia,
        'validate_el_badry': args.validate_el_badry,
        'calculate_masses': getattr(args, 'calculate_masses', False),
        'parallax_source': getattr(args, 'parallax_source', 'auto'),
        'gaia_p_value': args.gaia_p_value,
    }


def _create_dependencies(args: argparse.Namespace) -> Tuple[DataSource, Optional[HybridValidator]]:
    """
    Factory function to create and configure data source and validator dependencies.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple of (data_source, gaia_validator)
        
    Raises:
        ConfigurationError: When required configuration is missing or invalid
    """
    # 1. Set up the data source (only local source supported)
    if not args.database_path:
        raise ConfigurationError("--database-path is required for local source. "
                               "Run: python scripts/convert_catalogs_to_sqlite.py to create the database first.")
    
    data_source = LocalDataSource(database_path=args.database_path)
    
    # 2. Initialize the validator based on validation flags
    gaia_validator = None
    
    if args.validate_gaia and args.validate_el_badry:
        # Hybrid mode: El-Badry cache + Gaia fallback
        log.info("Hybrid validation enabled: El-Badry cache + Gaia fallback (requires network for uncached systems).")
        
        # Create the online validator as a component with Expert Hierarchical Validator
        from ..data.gaia_source import GaiaValidator
        online_validator = GaiaValidator(
            physical_p_value_threshold=args.gaia_p_value,
            ambiguous_p_value_threshold=args.gaia_p_value / AMBIGUOUS_P_VALUE_RATIO
        )
        
        # The main validator is the hybrid one
        gaia_validator = HybridValidator(data_source, online_validator)
        
        # Log cache statistics for transparency
        try:
            cache_stats = gaia_validator.get_cache_statistics()
            if 'cached_systems' in cache_stats and cache_stats['cached_systems'] != 'unknown':
                log.info(f"Validation cache: {cache_stats['cached_systems']} systems pre-computed from El-Badry catalog "
                        f"({cache_stats.get('cache_coverage_percent', 0):.1f}% coverage)")
                log.info("Cached systems will use El-Badry data, uncached systems will query Gaia with Expert Hierarchical Validator")
            else:
                log.info("No El-Badry cache available - will use Gaia-only validation with Expert Hierarchical Validator")
        except Exception as e:
            log.warning(f"Could not retrieve cache statistics: {e}")
    
    elif args.validate_gaia:
        # Gaia-only mode: Direct online queries without cache
        log.info("Gaia-only validation enabled (direct online queries, no cache).")
        
        # Create only the online validator with Expert Hierarchical Validator
        from ..data.gaia_source import GaiaValidator
        gaia_validator = GaiaValidator(
            physical_p_value_threshold=args.gaia_p_value,
            ambiguous_p_value_threshold=args.gaia_p_value / AMBIGUOUS_P_VALUE_RATIO
        )
        
        log.info("All systems will be validated using direct Gaia queries with Expert Hierarchical Validator")
    
    elif args.validate_el_badry:
        # El-Badry-only mode: Local cache only
        log.info("El-Badry-only validation enabled (local cache only, no network required).")
        
        # Create hybrid validator without online fallback (cache-only mode)
        gaia_validator = HybridValidator(data_source, online_validator=None)
        
        # Log cache statistics for transparency
        try:
            cache_stats = gaia_validator.get_cache_statistics()
            if 'cached_systems' in cache_stats and cache_stats['cached_systems'] != 'unknown':
                log.info(f"Validation cache: {cache_stats['cached_systems']} systems pre-computed from El-Badry catalog "
                        f"({cache_stats.get('cache_coverage_percent', 0):.1f}% coverage)")
                log.info("Systems not in El-Badry cache will be marked as 'Insufficient Data'")
            else:
                log.warning("No El-Badry cache available - all systems will be marked as 'Insufficient Data'")
        except Exception as e:
            log.warning(f"Could not retrieve cache statistics: {e}")
    
    # If neither flag is specified, gaia_validator remains None (no validation)
    
    return data_source, gaia_validator


def create_argument_parser():
    """Create command line argument parser with proper defaults from config."""
    parser = argparse.ArgumentParser(
        description='AstraKairos Binary Star Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s stars.csv --database-path catalogs.db --limit 10
  %(prog)s --all --database-path catalogs.db --output results.csv --validate-gaia
  %(prog)s --all --database-path catalogs.db --validate-el-badry --limit 1000
  %(prog)s --all --database-path catalogs.db --validate-gaia --validate-el-badry --output hybrid.csv
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
                       default=DEFAULT_MIN_OBS,
                       help=f'Minimum number of observations (default: {DEFAULT_MIN_OBS})')
    
    # Output options
    parser.add_argument('--output', '-o',
                       help='Output CSV file for results')
    
    parser.add_argument('--sort-by',
                       help='Sort results by this field. Options: discovery: v_total, v_total_significance, rmse; characterize: rmse, v_total_significance; orbital: opi_arcsec_yr, opi_significance, prediction_divergence_arcsec (default: mode-specific)')
    
    parser.add_argument('--concurrent',
                       type=int,
                       default=DEFAULT_CONCURRENT_REQUESTS,
                       help=f'Maximum concurrent requests (default: {DEFAULT_CONCURRENT_REQUESTS})')
                       
    gaia_group = parser.add_argument_group('Gaia Validation Options')
    
    gaia_group.add_argument('--validate-gaia',
                       action='store_true',
                       help='Validate physicality using direct online Gaia queries (requires network).')
    
    gaia_group.add_argument('--validate-el-badry',
                       action='store_true',
                       help='Validate physicality using local El-Badry catalog cache (no network required).')
    
    gaia_group.add_argument('--gaia-p-value',
                       type=float,
                       default=DEFAULT_GAIA_P_VALUE,
                       help=f'P-value threshold for physicality test (default: {DEFAULT_GAIA_P_VALUE})')

    # Mass calculation options
    mass_group = parser.add_argument_group('Mass Calculation Options')
    mass_group.add_argument(
        '--calculate-masses', 
        action='store_true',
        help='Calculate system masses using Kepler\'s Third Law (requires orbital elements and parallax)'
    )
    mass_group.add_argument(
        '--parallax-source',
        choices=['auto', 'gaia', 'none'],
        default='auto',
        help='Source for parallax data: auto (best available), gaia (Gaia only), none (skip mass calculation) (default: auto)'
    )
    mass_group.add_argument(
        '--mass-mc-samples',
        type=int,
        default=DEFAULT_MC_SAMPLES,
        help=f'Monte Carlo samples for mass uncertainty (default: {DEFAULT_MC_SAMPLES})'
    )

    # El-Badry catalog filtering
    parser.add_argument('--only-el-badry', 
                       action='store_true',
                       help='Only analyze systems confirmed to be in the El-Badry et al. (2021) high-confidence catalog.')

    # Spatial filtering options
    spatial_group = parser.add_argument_group('Spatial Filtering Options')
    spatial_group.add_argument(
        '--ra-range',
        type=str,
        help='Filter by Right Ascension range in hours (format: "min,max", e.g., "18.5,20.5"). '
             'Supports wraparound (e.g., "22.0,2.0" for 22h to 2h). No spaces in the string.'
    )
    spatial_group.add_argument(
        '--dec-range', 
        type=str,
        help='Filter by Declination range in degrees (format: "min,max", e.g., "-30.0,15.5"). '
             'Values must be between -90.0 and +90.0 degrees. No spaces in the string.'
    )
    
    # Debug options
    parser.add_argument('--debug', 
                       action='store_true',
                       help='Enable debug mode (allows generation of mock Gaia IDs for testing).')

    return parser

async def main_async(args: argparse.Namespace):
    """
    Main asynchronous function to orchestrate the star analysis process.
    
    This function handles:
    1. Loading and filtering the input data.
    2. Setting up dependencies (data source and validators).
    3. Creating the analysis runner.
    4. Running the analysis concurrently.
    5. Sorting and presenting the final results.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Validate that at least one validation method is specified if any validation is requested
    if not args.validate_gaia and not args.validate_el_badry:
        # No validation specified - this is valid, will skip physicality validation
        pass
    
    # 1. Create dependencies using factory function
    data_source, gaia_validator = _create_dependencies(args)

    # 2. Determine target list (from file or --all flag)
    df_targets = None
    if args.all:
        if args.input_file:
            log.warning("Ignoring input_file because --all flag was specified.")
        
        # Parse and validate spatial filtering arguments
        ra_range = None
        dec_range = None
        
        if args.ra_range:
            from ..utils.coordinate_parsing import parse_coordinate_range
            ra_range = parse_coordinate_range(args.ra_range, 'ra')
            
        if args.dec_range:
            from ..utils.coordinate_parsing import parse_coordinate_range
            dec_range = parse_coordinate_range(args.dec_range, 'dec')
        
        # Log applied filters for transparency
        if ra_range or dec_range:
            if ra_range and dec_range:
                ra_min, ra_max = ra_range
                dec_min, dec_max = dec_range
                log.info(f"Spatial filter: RA=[{ra_min:.2f}, {ra_max:.2f}]h, "
                        f"Dec=[{dec_min:.2f}, {dec_max:.2f}]°")
            elif ra_range:
                ra_min, ra_max = ra_range
                log.info(f"Spatial filter: RA=[{ra_min:.2f}, {ra_max:.2f}]h (all declinations)")
            elif dec_range:
                dec_min, dec_max = dec_range
                log.info(f"Spatial filter: Dec=[{dec_min:.2f}, {dec_max:.2f}]° (all right ascensions)")
        
        log.info("Fetching all WDS IDs from the local database...")
        
        all_ids = data_source.get_all_wds_ids(
            only_el_badry=args.only_el_badry,
            ra_range=ra_range,
            dec_range=dec_range
        )
        if not all_ids:
            if args.only_el_badry:
                log.error("No systems found in the El-Badry et al. (2021) catalog. Check that the database was created with --el-badry-file option.")
            else:
                log.error("Could not retrieve any WDS IDs from the database.")
            sys.exit(1)
        df_targets = pd.DataFrame(all_ids, columns=['wds_id'])
        
        if args.only_el_badry:
            log.info(f"Found {len(df_targets)} systems in El-Badry et al. (2021) high-confidence catalog to analyze.")
        else:
            log.info(f"Found {len(df_targets)} total systems to analyze.")

    elif args.input_file:
        # Validate that spatial filtering is not used with input files
        if args.ra_range or args.dec_range:
            log.error("Spatial filtering (--ra-range, --dec-range) can only be used with --all flag.")
            log.error("To filter stars from a file, pre-filter the CSV file or use --all with spatial filters.")
            sys.exit(1)
        
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

    # 3. Create AnalyzerRunner with configured dependencies
    runner = AnalyzerRunner(data_source, gaia_validator)
    
    # 4. Create analysis configuration from CLI arguments
    analysis_config = _create_analysis_config(args)
    
    try:
        # 5. Run the concurrent analysis with detailed error reporting
        log.info(f"Processing {len(df_filtered)} stars with up to {args.concurrent} concurrent tasks...")
        results, error_summary = await analyze_stars(runner, df_filtered, analysis_config, args.concurrent)
        
        # Print detailed error summary
        print_error_summary(len(df_filtered), len(results), error_summary)
        
        # Calculate statistical significance for results with uncertainties
        if results:
            log.info("Calculating significance for results with uncertainties...")
            for result in results:
                # Calculate velocity significance for discovery and characterize modes
                v_total_unc = result.get('v_total_uncertainty')
                if v_total_unc and v_total_unc > 0:
                    # Apply minimum uncertainty threshold to prevent extreme significance values
                    effective_uncertainty = max(v_total_unc, MIN_VELOCITY_UNCERTAINTY_ARCSEC_YR)
                    significance = abs(result.get('v_total_median', 0)) / effective_uncertainty
                    # Cap significance at reasonable maximum
                    result['v_total_significance'] = min(significance, MAX_REASONABLE_SIGNIFICANCE)

                # Calculate OPI significance for orbital mode
                opi_unc = result.get('opi_uncertainty')
                if opi_unc and opi_unc > 0:
                    # Apply minimum uncertainty threshold
                    effective_opi_uncertainty = max(opi_unc, MIN_OPI_UNCERTAINTY)
                    opi_significance = abs(result.get('opi_arcsec_yr', 0)) / effective_opi_uncertainty
                    # Cap OPI significance at reasonable maximum
                    result['opi_significance'] = min(opi_significance, MAX_REASONABLE_SIGNIFICANCE)
        
        # 6. Sort and present the results
        if results:
            # Use mode-specific sorting with strict validation
            sort_key = args.sort_by if args.sort_by else DEFAULT_SORT_KEYS[args.mode]
            
            # Validate sort key for current mode - exit on error for robustness
            if sort_key not in CLI_VALID_SORT_KEYS[args.mode]:
                log.error(f"Sort key '{sort_key}' is not valid for {args.mode} mode.")
                log.error(f"Valid options for {args.mode} mode: {', '.join(CLI_VALID_SORT_KEYS[args.mode])}")
                log.error("Please use a valid sort key or omit --sort-by to use the default.")
                sys.exit(1)
            
            log.info(f"Sorting results by: {sort_key}")
            
            # Improved sorting logic to handle None values safely
            results_sorted = sorted(
                results, 
                key=lambda x: x.get(sort_key) if x.get(sort_key) is not None else -1, 
                reverse=True
            )
            
            # Display summary in the console
            print("\n" + "=" * CLI_DISPLAY_LINE_WIDTH)
            print(f"TOP {CLI_TOP_RESULTS_DISPLAY_COUNT} ANALYSIS RESULTS - {args.mode.upper()} MODE (sorted by {sort_key})")
            print("=" * CLI_DISPLAY_LINE_WIDTH)
            
            for i, result in enumerate(results_sorted[:CLI_TOP_RESULTS_DISPLAY_COUNT], 1):
                # Mode-specific display format with uncertainty information
                if sort_key.endswith('_significance'):
                    # Display significance value when sorting by significance
                    significance_value = result.get(sort_key, 'N/A')
                    if significance_value != 'N/A':
                        metric_str = f"Significance = {significance_value:.2f}σ"
                    else:
                        metric_str = f"Significance = N/A"
                elif args.mode == 'discovery':
                    metric_value = format_metric_with_uncertainty(
                        result, 'v_total_median', 'v_total_uncertainty', 'quality_score'
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
                print(f"{i:2d}. {result['wds_id']:<{CLI_WDS_ID_COLUMN_WIDTH}} | {metric_str:<{CLI_METRIC_COLUMN_WIDTH}} | {phys_str}")
            
            print("-" * CLI_DISPLAY_LINE_WIDTH)
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