# AstraKairos 🌠

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/AstraKairos/astrakairos/actions/workflows/ci.yml/badge.svg)](https://github.com/AstraKairos/astrakairos/actions/workflows/ci.yml)
<!-- [![PyPI version](https://badge.fury.io/py/astrakairos.svg)](https://badge.fury.io/py/astrakairos) -->

<<<<<<< HEAD
**AstraKairos is an open-source Python framework for binary star research. It bridges the gap between large-scale astronomical catalogs and practical, targeted observation by providing tools for the validation, characterization, and prioritization of binary systems.**

AstraKairos equips both professional researchers and amateur astronomers with a modern toolset to answer two fundamental questions: "Is this star system a true physical binary?" and "Which systems are the most scientifically valuable to observe next?".
=======
**NOTE**: *AstraKairos is currently under development for version 1.0. If you want to track the current progress of the project, please check out the [development](https://github.com/AstraKairos/astrakairos/tree/development) branch.*

**AstraKairos is an integrated framework to find the *kairos*—the opportune moment—for astronomical discovery. v1.0 focuses on Binary Stars projects by presenting several QOL-improving as well as analysis tools. This open-source Python suite bridges the gap between massive astronomical catalogs (Such as the WDS, WDSS, and the ORB6) and practical night-to-night observation.**
>>>>>>> c5c948ce674403bc5193310f6e8a797c03aa5e59

AstraKairos brings researchers and amateur astronomers modern tools to identify, prioritize, and analyze targets in the Binary Stars realm.

<<<<<<< HEAD
AstraKairos is composed of two primary components: a **Catalog Analyzer (CLI)** for large-scale research and an **Observation Planner (GUI)** for preparing targeted observing campaigns.

### 🔬 **Catalog Analyzer (CLI)**
A powerful command-line tool for the batch analysis of binary star catalogs.

-   **Unified Data Pipeline:** Includes scripts to parse legacy text-based catalogs (WDS, WDSS, ORB6) into a single, query-optimized SQLite database.
-   **Machine Learning Physicality Classifier:**
    -   Utilizes a pre-trained **Random Forest** classifier to assign a **physicality probability (`P_phys`)** to each pair, distinguishing genuine physical binaries from optical chance alignments.
    -   The model is trained on a high-confidence ground-truth sample that fuses data from El-Badry et al. (2021) and the Gaia DR3 Non-Single Star (NSS) catalog.
-   **Dynamic Prioritization Metrics:**
    -   **Observation Priority Index (OPI):** Quantifies the urgency of re-observing systems with known orbits (from the ORB6 catalog) by measuring the deviation between the published orbit and the latest measurements.
    -   **Curvature Index:** Identifies systems *without* known orbits that exhibit significant non-linear motion, flagging them as high-priority candidates for a first orbit determination.
    -   **Robust Motion Characterization:** Employs Theil-Sen regression for detailed motion analysis, providing outlier-resistant velocity and acceleration estimates.
-   **Orbital Prediction Engine:** Implements a high-precision Kepler's Equation solver to compute ephemerides (future positions) from known orbital elements.

### 🔭 **Observation Planner (GUI)**
A desktop application designed for amateur and professional astronomers to plan a night of observation.

-   **Precise Astronomical Calculations:**
    -   High-precision ephemerides for solar system objects (powered by Skyfield).
    -   Calculation of twilight times (civil, nautical, astronomical) and Moon position for any location and date.
-   **Observation Optimization:**
    -   **Sky Quality Maps:** Generates visualizations of the observable sky, modeling atmospheric extinction and lunar light pollution to find the best region to observe.
    -   **Observatory Database:** Includes coordinates for over 3,000 observatories worldwide.
-   **Integration with External Catalogs:**
    -   Advanced search interface for the [Stelle Doppie](https://www.stelledoppie.it/) catalog, enabling the construction of custom target lists.
    -   Import and analysis of downloaded data in CSV format.

## Installation

AstraKairos requires Python 3.9+ and is designed for easy installation. A virtual environment is highly recommended.
=======
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
>>>>>>> c5c948ce674403bc5193310f6e8a797c03aa5e59

```bash
# Create and activate a virtual environment
python -m venv venv
# On Windows:   .\venv\Scripts\activate  
# On macOS/Linux: source venv/bin/activate

<<<<<<< HEAD
# Install the latest version directly from GitHub
=======
# Install from the repository
>>>>>>> c5c948ce674403bc5193310f6e8a797c03aa5e59
pip install git+https://github.com/AstraKairos/astrakairos.git

# Or install in development mode
git clone https://github.com/AstraKairos/astrakairos.git
cd astrakairos
pip install -e .
```

<<<<<<< HEAD
**Key dependencies:** `astropy`, `astroquery`, `pandas`, `scikit-learn`, `skyfield`, `numpy`, `scipy`.
=======
**Dependencies automatically installed:**
- `skyfield` - High-precision astronomical calculations
- `pandas` - Data analysis and CSV processing
- `astropy` - Astronomical data structures and utilities
- `astroquery` - Access to astronomical databases  
- `pytz` - Timezone calculations
- `numpy`, `scipy` - Numerical computations

**First-time setup:** On first run, Skyfield will automatically download the JPL DE421 ephemeris file (~17MB) for precise planetary positions.
>>>>>>> c5c948ce674403bc5193310f6e8a797c03aa5e59

## Usage

### 1. Data Preparation (CLI Analyzer Only)

<<<<<<< HEAD
Before using the analyzer, you must build the local database from the raw catalog files. This is a one-time step.

```bash
# Download the required catalog files (WDSS, ORB6, etc.)
# ...

# Run the conversion script
python scripts/convert_catalogs_to_sqlite.py \
  --wdss-files /path/to/wdss*.txt \
  --orb6 /path/to/orb6.txt \
  --output catalogs.db \
  --el-badry-file /path/to/el-badry.fits
```
**Optional but highly recommended:** For maximum performance and offline analysis, you can pre-fetch all required Gaia data.
```bash
python scripts/prefetch_gaia_data.py catalogs.db
```
=======
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
>>>>>>> c5c948ce674403bc5193310f6e8a797c03aa5e59

### 2. Catalog Analyzer (CLI)

<<<<<<< HEAD
The `analyzer` is the primary tool for large-scale research.

```bash
# Basic syntax
python -m astrakairos.analyzer.cli <input_file.csv_or_--all> --database-path catalogs.db [options]

# Example 1: Generate the physicality catalog for the entire WDSS
python -m astrakairos.analyzer.cli --all --database-path catalogs.db \
  --validate-gaia --output wdss_physicality_catalog.csv

# Example 2: Find the top 50 highest-priority orbital systems from ORB6
python -m astrakairos.analyzer.cli ORB6_systems.csv --database-path catalogs.db \
  --mode orbital --sort-by opi_arcsec_yr --limit 50 \
  --output orbital_priorities.csv

# Example 3: Search the WDS for new orbit candidates
python -m astrakairos.analyzer.cli WDS_systems.csv --database-path catalogs.db \
  --mode characterize --sort-by curvature_index --min-observations 5 \
  --output new_orbit_candidates.csv
```
For a complete guide to all arguments and options, run:
```bash
python -m astrakairos.analyzer.cli --help
```

### 3. Observation Planner (GUI)

Launch the graphical interface to plan an observing session:

```bash
python main.py planner
=======
Process star catalogs to find high-priority observation targets using local databases:

```bash
# Basic analysis with discovery mode (motion analysis)
python -m astrakairos.analyzer.cli targets.csv --database-path catalogs.db --limit 10

# Orbital analysis with Gaia validation
python -m astrakairos.analyzer.cli targets.csv --database-path catalogs.db --mode orbital --validate-gaia --output results.csv

# Analyze all systems in database with characterization mode
python -m astrakairos.analyzer.cli --all --database-path catalogs.db --mode characterize --limit 100

# Discovery mode with custom sorting
python -m astrakairos.analyzer.cli targets.csv --database-path catalogs.db --mode discovery --sort-by v_total_arcsec_yr
```

**Analysis modes:**
- `discovery`: Motion analysis and velocity calculations (default)
- `characterize`: Robust linear fitting with Theil-Sen regression  
- `orbital`: Observation Priority Index (OPI) calculation for ranking

**Key options:**
- `--database-path`: Required path to local SQLite catalog database 
- `--validate-gaia`: Enable Gaia DR3 physicality validation (requires network)
- `--mode`: Analysis type (discovery/characterize/orbital)
- `--limit`: Maximum number of systems to process
- `--output`: Output CSV file for results

For complete documentation:
```bash
python -m astrakairos.analyzer.cli --help
>>>>>>> c5c948ce674403bc5193310f6e8a797c03aa5e59
```
<!-- Or, if an entry point is defined in setup.py: `astrakairos-planner` -->

## Project Roadmap

<<<<<<< HEAD
AstraKairos is an actively developed project with a long-term vision.

### **Current Version (v1.0 - The Binary Star Suite)**
-   [x] **Core:** Modular software architecture and data pipeline.
-   [x] **Classifier:** Random Forest model for `P_phys` implemented and validated.
-   [x] **Metrics:** `OPI`, `Curvature Index`, and robust motion characterization implemented.
-   [x] **GUI:** Functional observation planner.
-   [x] **CLI:** Functional batch analysis engine.
-   [x] **Testing:** Unit test coverage >80%.
-   [ ] **Documentation:** Complete API reference and user tutorials.

### **Future (v2.0 - The Time-Domain Astrophysics Framework)**
The vision is to expand AstraKairos into a general-purpose platform for time-domain astrophysics.
-   [ ] **Orbit Fitting:** Integration of orbit-fitting tools (e.g., MCMC) to compute new orbital solutions.
-   [ ] **Variable Stars:** A module for light curve analysis (periodograms, template fitting) and prioritization of photometric observations.
-   [ ] **Minor Bodies:** Tools for ephemeris prediction and orbit determination of asteroids and comets.
=======
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
- [ ] **Mass Calculation:** Dynamic mass calculation using Kepler's Third Law

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
>>>>>>> c5c948ce674403bc5193310f6e8a797c03aa5e59

## Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on bug reports, feature requests, and code submissions.

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
