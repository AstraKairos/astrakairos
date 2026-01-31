# AstraKairos 🌠

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/AstraKairos/astrakairos/actions/workflows/ci.yml/badge.svg)](https://github.com/AstraKairos/astrakairos/actions/workflows/ci.yml)
<!-- [![PyPI version](https://badge.fury.io/py/astrakairos.svg)](https://badge.fury.io/py/astrakairos) -->

**AstraKairos is an open-source Python framework for binary star research. It bridges the gap between large-scale astronomical catalogs and practical, targeted observation by providing tools for the validation, characterization, and prioritization of binary systems.**

<!-- Include PDF image: -->
AstraKairos was awarded with the First Place for Astronomy Posters at VI Congreso de Estudiantes de Ciencias Físicas y Astronómicas (CECFA). ![Link to the poster](AstraKairos_Poster_CECFA.pdf).

AstraKairos brings researchers and amateur astronomers modern tools to identify, prioritize, and analyze targets in the Binary Stars realm with a focus on data purification.

AstraKairos is composed of two primary components: a **Catalog Analyzer (CLI)** for large-scale research and an **Observation Planner (GUI)** for preparing targeted observing campaigns.

### 🔬 **Catalog Analyzer (CLI)**
A powerful command-line tool for the batch analysis of binary star catalogs.

-   **Unified Data Pipeline:** Includes scripts to parse legacy text-based catalogs (WDS, WDSS, ORB6) into a single, query-optimized SQLite database.
-   **Machine Learning Physicality Classifier:**
    -   Utilizes a pre-trained **LightGBM** Regression model to predict the a **chance alignment** probability to each pair, helping distinguish genuine physical binaries from optical chance alignments.
    -   The model is trained on a sample data from El-Badry et al. (2021)'s "A million binaries from Gaia eDR3" catalog.
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

```bash
# Create and activate a virtual environment
python -m venv venv
# On Windows:   .\venv\Scripts\activate  
# On macOS/Linux: source venv/bin/activate

# Install the latest version directly from GitHub
pip install git+https://github.com/AstraKairos/astrakairos.git

# Or install in development mode
git clone https://github.com/AstraKairos/astrakairos.git
cd astrakairos
pip install -e .
```

**Key dependencies:** `astropy`, `astroquery`, `pandas`, `scikit-learn`, `skyfield`, `numpy`, `scipy`.

## Usage

### 1. Data Preparation (CLI Analyzer Only)

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

### 2. Catalog Analyzer (CLI)

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
```
<!-- Or, if an entry point is defined in setup.py: `astrakairos-planner` -->

## Project Roadmap

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
