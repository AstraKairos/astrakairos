# AstraKairos 🌠

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/AstraKairos/astrakairos/actions/workflows/ci.yml/badge.svg)](https://github.com/AstraKairos/astrakairos/actions/workflows/ci.yml)
<!-- [![PyPI version](https://badge.fury.io/py/astrakairos.svg)](https://badge.fury.io/py/astrakairos) -->

**AstraKairos is an integrated framework to find the *kairos*—the opportune moment—for astronomical discovery. v1.0 focuses on Binary Stars projects by presenting several QOL-improving as well as analysis tools. This open-source Python suite bridges the gap between massive astronomical catalogs (Such as the WDS, WDSS, and the ORB6) and practical night-to-night observation.**

AstraKairos brings researchers and amateur astronomers modern tools to identify, prioritize, and analyze targets in the Binary Stars realm.

## Key Features (v1.0)

### 🔭 **Observation Planner (GUI)**
A desktop application that optimizes astronomical observations by analyzing:
- **Location-aware calculations:** Over 3,000+ observatory locations worldwide  
- **Twilight calculations:** Civil, nautical, and astronomical twilight times with timezone support
- **Sky quality mapping:** Atmospheric extinction and lunar contamination models
- **Target search generation:** Filtering with 18 search methods based on [Stelle Doppie](https://www.stelledoppie.it/)'s searching tool
- **CSV import & analysis:** Parsing of downloaded Stelle Doppie's data with format auto-detection
- **Multi-platform compatibility:** Compatible with Windows, macOS, and Linux distributions

### 🚀 **Velocity Analysis Engine** 
- **Binary velocity calculation:** Calculates total observed angular velocity for imported results from Stelle Doppie
- **Endpoint velocity calculations:** Uses first/last observation epochs for motion analysis
- **Velocity statistics:** Statistics and high-priority target identification
- **Column mapping:** Handles both basic and full CSV formats from Stelle Doppie

### 🌙 **Astronomical Calculations**
- **Precise ephemeris:** Powered by Skyfield with JPL DE421 ephemeris data
- **Sky brightness modeling:** Krisciunas & Schaefer (1991) lunar scattering implementation
- **Atmospheric extinction:** Site-specific extinction coefficients for photometry
- **Time-domain optimization:** Finds optimal observing windows considering environmental factors

### 📊 **Data-Driven Target Prioritization (CLI)**
A command-line tool that analyzes star catalogs to find high-priority targets:
- **Apparent Motion Vectors:** Identifies high-velocity systems with significant relative motion
- **Observation Priority Index (OPI):** Metric quantifying orbital deviation urgency
- **Physicality Validation:** Uses Gaia DR3 data to distinguish physical binaries from optical alignments

### 🔧 **Data Handling**
- **Multiple export formats:** CSV, JSON, LaTeX tables with metadata
- **File processing:** Handles various encodings and CSV formats with error handling
- **Modular data sources:** Local catalogs (WDSS, ORB6) or real-time web scraping
- **Orbital Prediction Engine:** Implements Kepler's equation solver to predict future positions of stars in known orbits
- **Modular Data Sources:** Can operate using local, offline catalogs (for performance and reproducibility) or by scraping up-to-date web sources

## Installation

AstraKairos requires Python 3.8+ and is designed for easy installation. A virtual environment is recommended.

```bash
# Create and activate a virtual environment
python -m venv venv
# On Windows:   .\venv\Scripts\activate  
# On macOS/Linux: source venv/bin/activate

# Install from the repository
pip install git+https://github.com/AstraKairos/astrakairos.git

# Or install in development mode
git clone https://github.com/AstraKairos/astrakairos.git
cd astrakairos
pip install -e .
```

**Dependencies automatically installed:**
- `skyfield` - High-precision astronomical calculations
- `pandas` - Data analysis and CSV processing
- `astropy` - Astronomical data structures and utilities
- `astroquery` - Access to astronomical databases  
- `pytz` - Timezone calculations
- `numpy`, `scipy` - Numerical computations

**First-time setup:** On first run, Skyfield will automatically download the JPL DE421 ephemeris file (~17MB) for precise planetary positions.

## Usage

AstraKairos provides two main entry points: a graphical planner and a command-line analyzer.

### 1. Observation Planner (GUI)

Launch the observation planning interface:

```bash
python main.py planner
```

The GUI provides:
- **Observatory selection** from 3,000+ worldwide locations
- **Optimal sky region calculation** using atmospheric models  
- **Stelle Doppie search generation** with 18 filtering methods
- **CSV data import & analysis** with automatic format detection
- **Velocity analysis** for identifying high-motion binary systems
- **Export capabilities** in multiple formats

### 2. Data Analyzer (CLI)

The CLI analyzer processes binary star catalogs to find high-priority observation targets. It requires a **SQLite database** created from WDSS/ORB6 catalogs using the conversion script.

#### **Prerequisites**

First, create the SQLite database from your catalog files:
```bash
# Convert WDSS and ORB6 catalogs to SQLite database
python scripts/convert_catalogs_to_sqlite.py --wdss-file wdss_master.txt --orb6-file orb6_catalog.txt --output catalogs.db
```

You only need to do this once. The required catalog files are:
- **WDSS Master Catalog** (`wdss_master.txt`) - Binary star measurements
- **ORB6 Catalog** (`orb6_catalog.txt`) - Orbital elements for known binaries

#### **Two Analysis Modes**

**Option A: Analyze ALL systems in database (recommended for discovery)**
```bash
# Discover high-motion systems from entire database
python -m astrakairos.analyzer.cli --all --database-path catalogs.db --limit 100

# Find orbital priority targets with Gaia validation
python -m astrakairos.analyzer.cli --all --database-path catalogs.db --mode orbital --validate-gaia --limit 50 --output high_priority.csv

# Characterize motion of all systems with robust fitting
python -m astrakairos.analyzer.cli --all --database-path catalogs.db --mode characterize --limit 200
```

**Option B: Analyze specific targets from a CSV file**
```bash
# Create a targets file with specific WDS IDs
echo "wds_id" > my_targets.csv
echo "00003+1154" >> my_targets.csv
echo "00005+1259" >> my_targets.csv
echo "00006+2012" >> my_targets.csv

# Analyze only these specific systems
python -m astrakairos.analyzer.cli my_targets.csv --database-path catalogs.db --mode orbital --validate-gaia
```

#### **Analysis Modes**
- **`discovery`** (default): Fast motion analysis using endpoint velocities - ideal for finding high-velocity systems
- **`characterize`**: Robust linear fitting with Theil-Sen regression - provides detailed motion characterization  
- **`orbital`**: Observation Priority Index (OPI) calculation - ranks systems by orbital deviation urgency

#### **Key Options**
- `--all`: Analyze all systems in database (alternative to providing CSV file)
- `--database-path`: **Required** - Path to SQLite database created by conversion script
- `--validate-gaia`: Enable Gaia DR3 physicality validation (requires internet connection)
- `--mode`: Analysis type (discovery/characterize/orbital)
- `--limit`: Maximum number of systems to process (recommended for large databases)
- `--output`: Save results to CSV file
- `--sort-by`: Custom sorting field (e.g., `v_total_arcsec_yr`, `opi_arcsec_yr`)

#### **Example Output**
```
TOP 10 ANALYSIS RESULTS - ORBITAL MODE (sorted by opi_arcsec_yr)
================================================================================
 1. 07142+2357           | OPI = 12.4567               | Gaia: Likely Physical
 2. 15234+4021           | OPI = 8.9123                | Gaia: Likely Physical
 3. 23456+1234           | OPI = 6.7890                | Gaia: Ambiguous
```

#### **Complete Documentation**
```bash
python -m astrakairos.analyzer.cli --help
```

#### **Setup: Generating the Required Database**

Before using the CLI, you need to create the SQLite database from the catalog files:

```bash
# Convert WDSS and ORB6 catalogs to SQLite format
python scripts/convert_catalogs_to_sqlite.py

# This creates: results/wdss3-data.db (used by the CLI)
```

**Required catalog files:**
- `data_catalogs/wdss*.txt` (WDSS catalog files)
- `data_catalogs/orb6orbits.txt` (ORB6 orbital elements)

These are included in the repository and only need to be converted once.

#### **Input File Requirements**

**Required files:**
- **SQLite database** (`catalogs.db`) - Created once from WDSS/ORB6 catalogs using the conversion script

**Optional files:**
- **Targets CSV** (`my_targets.csv`) - Used only when analyzing specific systems instead of the entire database

**Targets CSV format (when using Option B):**
```csv
wds_id
00003+1154
00005+1259
00006+2012
20126+4003
```

The CSV must contain a `wds_id` column with valid WDS designations. You can create this file:
- Manually (as shown above)
- By exporting from the GUI planner
- By filtering results from previous analyses
- From any astronomical database that provides WDS identifiers

## Project Roadmap

The development of AstraKairos is phased to deliver a valuable tool at each stage.

### v1.0: The Binary Star Research Suite
*Binary star validation, analysis, and observation planning tool.*

**Core Architecture & Data Layer**
- [x] **Data Source Abstraction:** Modular `DataSource` interface implemented
- [x] **Local Source:** Parsers for offline **WDSS** and **ORB6** catalogs  
- [x] **Catalog Hub:** Cross-matching and **SIMBAD** name resolution

**Engine**
- [x] **Kepler's Equation Solver:** High-precision numerical solver with hybrid initial guess strategy
- [x] **Orbital Prediction:** Precise ephemerides (PA/Sep) from orbital elements
- [x] **Dynamics Analysis:** Apparent motion vectors (`v_total`, `PA_v`) with endpoint method
- [x] **Observation Priority Index (OPI):** Complete implementation for ranking orbital deviations
- [x] **Physicality Validation:** Fully integrated **Gaia DR3** parallax and proper motion validation
- [x] **Mass Calculation:** Dynamic mass calculation using Kepler's Third Law

**User Interfaces & Workflow**
- [x] **Planner (GUI):** Complete interface with location selection, twilight calculations, and CSV analysis
- [x] **Analyzer (CLI):** Full batch processing with report generation and export
- [x] **Features:** Velocity analysis, high-motion detection, and multi-format export

**Project Quality**
- [x] Test coverage with `pytest` (>80% of functions tested)
- [x] Error handling and file processing
- [x] Documentation with proper citations and methodology  
- [ ] Configure Continuous Integration (CI) workflow with GitHub Actions
- [ ] Complete API reference documentation with Sphinx

---

### v2.0 & Beyond: The Time-Domain Astrophysics Framework 🔮
*Expand AstraKairos into a general-purpose platform for time-domain astrophysics.*

**Variable Stars Module:**
- [ ] **AAVSO VSX Integration:** Automatic period prediction and minima/maxima calculations
- [ ] **Comparison Star Finder:** Automated selection of photometric comparison stars  
- [ ] **Time-Series Analysis:** Lomb-Scargle periodograms and phase-folding tools
- [ ] **Light Curve Modeling:** Template fitting for eclipsing binaries and pulsating variables

**Analysis & Visualization:**
- [ ] **MCMC Orbit Fitting:** Bayesian orbit determination with uncertainty quantification
- [ ] **Universal Solver:** Support for parabolic/hyperbolic orbits (comets, interstellar objects)
- [ ] **Statistical Framework:** Complete error propagation and uncertainty analysis
- [ ] **Interactive Visualization:** 3D orbit displays and sky chart overlays with real-time updates

**Minor Planet Module:** 
- [ ] **Asteroid ephemeris:** Integration with MPC database for accurate position predictions
- [ ] **Orbit determination:** Least-squares and differential correction algorithms
- [ ] **Discovery tools:** Automated moving object detection in image sequences

## Contributing

Contributions are welcome! Whether you're reporting a bug, proposing a new feature, or submitting code, your input is valuable. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgements & Transparency Notice

This project builds upon the foundational work of the astronomical community and the availability of open data.

**Special recognition to:**
- **U.S. Naval Observatory (USNO)** for curation of the **Washington Double Star (WDS) Catalog**, the **Washington Double Star Supplemental (WDSS) Catalog**, and the **Sixth Catalog of Orbits of Visual Binary Stars (ORB6)**
- **Stelle Doppie** project for providing accessible, centralized astronomical data resources
- **ESA Gaia Mission** and the **Gaia Data Processing and Analysis Consortium (DPAC)** for astrometric data
- **Stellarium project** for the observatory location database used in `/locations.json`

**Software foundation:**
This project leverages the Python ecosystem, particularly **NumPy**, **SciPy**, **Pandas**, **Astropy**, **Skyfield**, and **AstroQuery**.

Additionally, this project was developed with the assistance of AI-powered tools such as OpenAI's ChatGPT. These tools were used for tasks including generating boilerplate code, debugging algorithms, and writing/translating documentation. All AI-assisted output was carefully reviewed, tested, and adapted by the human author to ensure its viability and correctness for real-world research environments.

**Methodology:**
All astronomical calculations implement peer-reviewed algorithms with proper citations. Ephemeris calculations use JPL DE421, twilight calculations follow standard astronomical definitions, and sky brightness modeling implements the Krisciunas & Schaefer (1991) lunar scattering model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.