#!/usr/bin/env python3
"""
Convert WDS text catalogs to SQLite for efficient querying.

This script is the entry point for the autonomous conversion tool.
It coordinates the complete conversion pipeline from text catalogs 
to optimized SQLite database.
"""

import argparse
import logging
import sys
from pathlib import Path

from conversion_tool.pipeline import ConversionPipeline
from astrakairos.exceptions import ConversionProcessError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def main():
    """
    Main entry point for converting WDSS catalogs to SQLite database.
    
    This function coordinates the complete conversion pipeline using
    the modular conversion tool components.
    """
    parser = argparse.ArgumentParser(description='Convert WDSS catalogs to SQLite')
    parser.add_argument('--wdss-files', required=True, nargs='+', 
                       help='Paths to WDSS catalog files (e.g., wdss1.txt wdss2.txt wdss3.txt wdss4.txt)')
    parser.add_argument('--orb6', required=True, help='Path to ORB6 catalog')
    parser.add_argument('--output', required=True, help='Output SQLite database path')
    parser.add_argument('--force', action='store_true', help='Overwrite existing database')
    parser.add_argument('--el-badry-file', 
                       help='Path to El-Badry et al. (2021) binary catalog FITS file for cross-matching')
    
    args = parser.parse_args()
    
    # Check if output exists
    if Path(args.output).exists() and not args.force:
        raise ConversionProcessError(f"Output file {args.output} already exists. Use --force to overwrite.")
    
    # Prepare configuration for the pipeline
    config = {
        'wdss_files': args.wdss_files,
        'orb6_file': args.orb6,
        'el_badry_file': args.el_badry_file,
        'output_path': args.output
    }

    try:
        # Create and run the conversion pipeline
        pipeline = ConversionPipeline(config)
        pipeline.run()
        
        # Display final statistics
        stats = pipeline.get_statistics()
        log.info("Conversion completed successfully!")
        log.info(f"Final statistics: {stats}")
        
    except ConversionProcessError as e:
        log.error(f"Conversion failed: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Unexpected error during conversion: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
