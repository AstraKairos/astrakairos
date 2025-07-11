import argparse
import asyncio
import aiohttp
import sys
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
import functools

from ..data.source import DataSource
from ..data.local_source import LocalFileDataSource
from ..data.web_source import StelleDoppieDataSource
from ..data.gaia_source import GaiaValidator
from ..physics.dynamics import calculate_velocity_vector, calculate_observation_priority_index
from ..physics.kepler import predict_position
from ..utils.io import load_csv_data, save_results_to_csv
from datetime import datetime

# En astrakairos/analyzer/cli.py

# ... (otros imports) ...
from datetime import datetime
from ..data.source import DataSource
from ..data.gaia_source import GaiaValidator
from ..physics.dynamics import calculate_velocity_vector, calculate_observation_priority_index

async def process_star(row: pd.Series,
                       data_source: DataSource,
                       gaia_validator: Optional[GaiaValidator],
                       cli_args: argparse.Namespace, # Accepts the full CLI args
                       semaphore: asyncio.Semaphore) -> Optional[Dict[str, Any]]:
    """
    Process a single star: fetch its data, calculate dynamics, and validate its physicality.
    """
    async with semaphore:
        wds_name = row['wds_name']
        print(f"\n--- Processing {wds_name} (obs: {row.get('obs', 'N/A')}) ---")

        try:
            # 1. Get observational data
            wds_data = await data_source.get_wds_data(wds_name)
            if not wds_data or not all(key in wds_data and wds_data[key] is not None for key in
                                      ['date_first', 'pa_first', 'sep_first', 'date_last', 'pa_last', 'sep_last']):
                print(f"✗ Incomplete WDS data for {wds_name}.")
                return None

            # 2. Calculate apparent motion dynamics
            v_total, pa_v = calculate_velocity_vector(wds_data)
            print(f"✓ Apparent motion calculated: v_total = {v_total:.6f} arcsec/year")

            # 3. Calculate OPI if orbital data is available
            opi, deviation = None, None
            orbital_elements = await data_source.get_orbital_elements(wds_name)
            if orbital_elements and all(key in orbital_elements and orbital_elements[key] is not None for key in ['P', 'T', 'e', 'a', 'i', 'Omega', 'omega']):
                print("  -> Orbital elements found. Calculating OPI...")
                current_year = datetime.utcnow().year + datetime.utcnow().timetuple().tm_yday / 365.25
                opi_result = calculate_observation_priority_index(orbital_elements, wds_data, current_year)
                if opi_result:
                    opi, deviation = opi_result
                    print(f"✓ OPI calculated: {opi:.4f} (deviation: {deviation:.3f}\")")

            # 4. Validate physicality using Gaia
            physicality_result = {'label': 'Not checked', 'p_value': None, 'test_used': None}
            if gaia_validator and 'ra_deg' in wds_data and wds_data['ra_deg'] is not None:
                print(f"  -> Validating physicality with Gaia for {wds_name}...")
                
                search_radius = None
                if wds_data.get('sep_last') is not None:
                    # Use CLI args to calculate search radius
                    search_radius = wds_data['sep_last'] * cli_args.gaia_radius_factor
                    search_radius = max(search_radius, cli_args.gaia_min_radius)
                    search_radius = min(search_radius, cli_args.gaia_max_radius)

                wds_mags = (wds_data.get('mag_pri'), wds_data.get('mag_sec'))
                
                physicality_result = await gaia_validator.validate_physicality(
                    primary_coords_deg=(wds_data['ra_deg'], wds_data['dec_deg']),
                    wds_magnitudes=wds_mags,
                    search_radius_arcsec=search_radius
                )
                print(f"✓ Physicality assessment: {physicality_result['label']} (p-value: {physicality_result.get('p_value')}, test: {physicality_result.get('test_used')})")

            # 5. Build final results dictionary
            result = {
                'wds_name': wds_name,
                'obs_wds': wds_data.get('obs', None),
                'date_last': wds_data['date_last'],
                'v_total_arcsec_yr': v_total,
                'pa_v_deg': pa_v,
                'opi_arcsec_yr': opi,
                'deviation_arcsec': deviation,
                'physicality_label': physicality_result['label'],
                'physicality_p_value': physicality_result['p_value'],
                'physicality_test_used': physicality_result['test_used']
            }
            if orbital_elements:
                result.update({f'orb_{k}': v for k, v in orbital_elements.items()})
            return result

        except Exception as e:
            print(f"✗ An unexpected error occurred while processing {wds_name}: {e}")
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
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='AstraKairos Binary Star Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s stars.csv --source web --limit 10
  %(prog)s stars.csv --source local --wds-file wds.txt --orb6-file orb6.txt
  %(prog)s stars.csv --source web --output results.csv --validate-gaia
        """
    )
    
    # Positional arguments
    parser.add_argument('input_file', 
                       help='CSV file containing star list')
    
    # Data source options
    parser.add_argument('--source', 
                       choices=['web', 'local'], 
                       default='web',
                       help='Data source to use (default: web)')
    
    parser.add_argument('--wds-file',
                       help='Path to local WDS catalog file (required for local source)')
    
    parser.add_argument('--orb6-file',
                       help='Path to local ORB6 catalog file (required for local source)')
    
    # Processing options
    parser.add_argument('--limit', '-n',
                       type=int,
                       help='Limit number of stars to process')
    
    parser.add_argument('--min-obs',
                       type=int,
                       default=2,
                       help='Minimum number of observations (default: 2)')
    
    # Output options
    parser.add_argument('--output', '-o',
                       help='Output CSV file for results')
    
    parser.add_argument('--sort-by',
                       choices=['v_total', 'obs', 'date_last', 'opi'],
                       default='v_total',
                       help='Sort results by this field (default: v_total)')
    
    parser.add_argument('--concurrent',
                       type=int,
                       default=5,
                       help='Maximum concurrent requests (default: 5)')
                       
    gaia_group = parser.add_argument_group('Gaia Validation Options')
    
    gaia_group.add_argument('--validate-gaia',
                       action='store_true',
                       help='Validate physicality using Gaia data (requires network).')
    
    gaia_group.add_argument('--gaia-p-value',
                       type=float,
                       default=0.01,
                       help='P-value threshold for the chi-squared test to reject the "physical pair" hypothesis (default: 0.01).')
    
    gaia_group.add_argument('--gaia-radius-factor',
                       type=float,
                       default=1.5,
                       help='Factor to multiply by the last separation to get the Gaia search radius (default: 1.5).')
    
    gaia_group.add_argument('--gaia-min-radius',
                       type=float,
                       default=2.0,
                       help='Minimum search radius in arcseconds for Gaia query (default: 2.0).')

    gaia_group.add_argument('--gaia-max-radius',
                       type=float,
                       default=60.0,
                       help='Maximum search radius in arcseconds for Gaia query (default: 60.0).')

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
    # 1. Load and filter input data from CSV
    print(f"Loading data from: {args.input_file}")
    try:
        df = load_csv_data(args.input_file)
    except Exception as e:
        print(f"Error: Could not load or parse the input file '{args.input_file}'.")
        print(f"Details: {e}")
        sys.exit(1)
        
    # Filter by minimum number of observations if the column exists
    if 'obs' in df.columns and args.min_obs > 0:
        df_filtered = df[df['obs'] >= args.min_obs].copy()
        print(f"Filtered to {len(df_filtered)} stars with >= {args.min_obs} observations.")
    else:
        df_filtered = df.copy()
    
    # Apply limit to the number of stars to process if specified
    if args.limit:
        df_filtered = df_filtered.head(args.limit)
        print(f"Limited to processing a maximum of {len(df_filtered)} stars.")
    
    if df_filtered.empty:
        print("No stars to process after filtering. Exiting.")
        return

    # 2. Set up the data source
    data_source: DataSource
    session = None
    if args.source == 'local':
        if not args.wds_file or not args.orb6_file:
            print("Error: --wds-file and --orb6-file are required for local source.")
            sys.exit(1)
        data_source = LocalFileDataSource(wds_filepath=args.wds_file, orb6_filepath=args.orb6_file)
    else:  # 'web' source
        # Configure the HTTP session for web scraping
        headers = {'User-Agent': 'AstraKairos/1.0 (https://github.com/AstraKairos/astrakairos)'}
        timeout = aiohttp.ClientTimeout(total=30) # Total timeout for the entire request
        connector = aiohttp.TCPConnector(limit_per_host=5, limit=50) # Concurrency limits
        session = aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers)
        data_source = StelleDoppieDataSource(session)
    
    # 3. Initialize the Gaia validator if requested
    gaia_validator = None
    if args.validate_gaia:
        # Configure the validator with parameters from the command line
        print(f"Gaia validation enabled with p-value threshold: {args.gaia_p_value}")
        gaia_validator = GaiaValidator(p_value_threshold=args.gaia_p_value)

    # 4. Create a pre-configured processing function using functools.partial
    configured_process_star = functools.partial(
        process_star,
        data_source=data_source,
        gaia_validator=gaia_validator,
        cli_args=args
    )
    
    try:
        # 5. Run the concurrent analysis
        print(f"\nProcessing {len(df_filtered)} stars with up to {args.concurrent} concurrent tasks...")
        results = await analyze_stars(df_filtered, configured_process_star, args.concurrent)
        
        # 6. Sort and present the results
        if results:
            # Sorting key requires careful handling of None values for OPI
            sort_key = args.sort_by if args.sort_by != 'opi' else 'opi_arcsec_yr'
            # For sorting, treat None as a very small number so it goes to the end in reverse sort
            results_sorted = sorted(results, key=lambda x: x.get(sort_key, -1) or -1, reverse=True)
            
            # Display summary in the console
            print("\n" + "="*80)
            print(f"TOP 10 ANALYSIS RESULTS (sorted by {args.sort_by})")
            print("="*80)
            
            for i, result in enumerate(results_sorted[:10], 1):
                opi_str = f"OPI = {result['opi_arcsec_yr']:.4f}" if result.get('opi_arcsec_yr') is not None else "OPI = N/A"
                phys_str = f"Gaia: {result['physicality_label']}" if result.get('physicality_label') != 'Not checked' else ''
                print(f"{i:2d}. {result['wds_name']:<18} | {opi_str:<18} | {phys_str}")
            
            print("-" * 80)
            print(f"\nProcessed {len(results)} of {len(df_filtered)} stars successfully.")
            
            # Save full results to a CSV file if requested
            if args.output:
                save_results_to_csv(results_sorted, args.output)
        else:
            print("\nNo stars were successfully processed.")
            
    finally:
        # 7. Clean up resources
        if session:
            await session.close()

def main():
    """Main entry point for the analyzer CLI."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Run async main
    import time
    start_time = time.time()
    
    asyncio.run(main_async(args))
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()