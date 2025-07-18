#!/usr/bin/env python
"""
AstraKairos - Binary Star Research Assistant

This is the main entry point for the AstraKairos software suite.
It provides access to both the observation planner GUI and the analyzer CLI.

Version: 1.0.0
Author: Martín Rubina Scapini
Institution: Universidad Técnica Federico Santa María, Departamento de Física
"""

import sys
import argparse

# Version information for scientific reproducibility
__version__ = "1.0.0"
__author__ = "Martín Rubina Scapini"
__institution__ = "Universidad Técnica Federico Santa María, Departamento de Física"

def main():
    """Main entry point for AstraKairos."""
    parser = argparse.ArgumentParser(
        description=f'AstraKairos v{__version__} - Binary Star Research Assistant',
        epilog='Use "planner" for the GUI observation planner or "analyzer" for the CLI analyzer.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('tool', 
                       choices=['planner', 'analyzer'],
                       help='Tool to run')
    parser.add_argument('--version', action='version', 
                       version=f'AstraKairos {__version__} by {__author__} ({__institution__})')
    
    args, remaining_args = parser.parse_known_args()
    
    try:
        if args.tool == 'planner':
            # Run the planner GUI
            from astrakairos.planner.gui import main as planner_main
            planner_main()
        elif args.tool == 'analyzer':
            # Run the analyzer CLI with proper argument handling
            from astrakairos.analyzer.cli import main as analyzer_main
            # Pass arguments properly without modifying sys.argv
            analyzer_main(remaining_args)
    except ImportError as e:
        print(f"ERROR: Failed to import required module: {e}", file=sys.stderr)
        print("Ensure all dependencies are installed: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()