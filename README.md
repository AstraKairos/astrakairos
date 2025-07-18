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
- **Modular data sources:** Local catalogs (WDS, ORB6) or real-time web scraping
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

Process star catalogs to find high-priority observation targets:

```bash
python main.py analyzer targets.csv --source web --mode discovery --limit 10 --sort-by velocity
```

For analysis with local catalogs:
```bash
python main.py analyzer targets.csv --source local --wds-file wds.txt --orb6-file orb6.txt --mode orbital --validate-gaia
```

**Analysis modes:**
- `discovery`: Motion analysis and velocity calculations
- `characterize`: Orbital fitting and parameter estimation  
- `orbital`: OPI calculation for observation priority ranking

For complete CLI documentation:
```bash
python main.py analyzer --help
```

## Project Roadmap

The development of AstraKairos is phased to deliver a valuable tool at each stage.

### v1.0: The Binary Star Research Suite
*Production-ready tool for binary star validation, analysis, and observation planning.*

**Core Architecture & Data Layer**
- [x] **Data Source Abstraction:** Modular `DataSource` interface implemented
- [x] **Local Source:** Parsers for offline **WDS**, **WDSS**, and **ORB6** catalogs  
- [x] **Web Source:** VizieR integration
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