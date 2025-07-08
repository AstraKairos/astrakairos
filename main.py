#!/usr/bin/env python
"""
AstraKairos - Binary Star Research Assistant

This is the main entry point for the AstraKairos software suite.
It provides access to both the observation planner GUI and the analyzer CLI.
"""

import sys
import argparse

def main():
    """Main entry point for AstraKairos."""
    parser = argparse.ArgumentParser(
        description='AstraKairos - Binary Star Research Assistant',
        epilog='Use "planner" for the GUI observation planner or "analyzer" for the CLI analyzer.'
    )
    
    parser.add_argument('tool', 
                       choices=['planner', 'analyzer'],
                       help='Tool to run')
    
    args, remaining_args = parser.parse_known_args()
    
    if args.tool == 'planner':
        # Run the planner GUI
        from astrakairos.planner.gui import main as planner_main
        planner_main()
    elif args.tool == 'analyzer':
        # Run the analyzer CLI with remaining arguments
        sys.argv = ['analyzer'] + remaining_args
        from astrakairos.analyzer.cli import main as analyzer_main
        analyzer_main()

if __name__ == "__main__":
    main()