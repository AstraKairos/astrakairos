# AstraKairos 🌠

<!-- [![PyPI version](https://badge.fury.io/py/astrakairos.svg)](https://badge.fury.io/py/astrakairos) --> <!-- Placeholder: Activate once published to PyPI -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/AstraKairos/astrakairos/actions/workflows/ci.yml/badge.svg)](https://github.com/AstraKairos/astrakairos/actions/workflows/ci.yml)

 **AstraKairos is an integrated scientific framework to find the *kairos*—the opportune moment—for astronomical discovery. From validating binary star orbits to planning variable star photometry.**

This open-source Python suite bridges the gap between massive astronomical catalogs (like Gaia, WDS, and ORB6) and practical night-to-night observation, enabling researchers and amateur astronomers to find and prioritize the most scientifically valuable targets.

## Key Features

- **Intelligent Observation Planner (GUI):** A user-friendly desktop application that recommends the optimal sky region for observation by analyzing local conditions, moon position, and zenith.
- **Physicality Validation Engine:** Uses high-precision data from the Gaia satellite to definitively confirm whether a star pair is a true physical binary or just an optical illusion.
- **Orbital Prediction & Deviation Analysis:** Implements a robust Kepler's equation solver to predict the positions of stars in known orbits and identifies systems whose published orbits are outdated or incorrect.
- **Data-Driven Target Prioritization:** Analyzes catalogs to identify scientifically interesting targets, such as high-velocity systems or neglected historical binaries.
- **Modular Data Sources:** Can operate using local, offline catalogs (for performance and reproducibility) or by scraping up-to-date web sources.

## Installation

AstraKairos is designed to be easily installed via pip.

```bash
# It is recommended to use a virtual environment
python -m venv venv
# On Windows: .\venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

pip install astrakairos
```

For the latest development version:

```bash
pip install git+https://github.com/AstraKairos/astrakairos.git
```

## Quickstart

AstraKairos provides two main entry points: a graphical planner and a command-line analyzer.

### 1. Launch the Observation Planner (GUI)

After installation, simply run the following command in your terminal:

```bash
astrakairos-plan
```

### 2. Run the Data Analyzer (CLI)

The analyzer can process a CSV list of stars. For example, to analyze the top 10 stars from `targets.csv` using the web data source:

```bash
astrakairos-analyze targets.csv --source web --limit 10
```

For more details, run `astrakairos-analyze --help`.

## Project Roadmap (v1.0 - Binary Stars Module)

This roadmap outlines the core features for the initial release.

- [x] **Phase 0: Project Scaffolding & Fusion**
  - [x] Create unified `astrakairos` repository and organization.
  - [x] Merge `Candidate-Searcher` and `StelleDoppie-Filtering` histories.
  - [x] Establish "professional" project structure (`pyproject.toml`, `LICENSE`, etc.).
- [ ] **Phase 1: Data Abstraction Layer**
  - [ ] Define abstract `DataSource` interface.
  - [ ] Implement `LocalFileDataSource` with robust WDS and ORB6 parsers.
  - [ ] Implement `StelleDoppieDataSource` by refactoring web scraping logic.
  - [ ] Implement `GaiaValidator` using `astroquery` for physicality checks.
- [x] **Phase 2: Scientific Core & Interfaces**
  - [x] Implement robust Kepler's equation solver in `physics.kepler`.
  - [x] Implement velocity vector calculations in `physics.dynamics`.
  - [ ] Refactor the Analyzer (CLI) to use the new data and physics layers.
  - [ ] Refactor the Planner (GUI) to be more modular and connect to the analysis pipeline.
- [ ] **Phase 3: Finalization & Documentation**
  - [ ] Achieve >80% test coverage with `pytest`.
  - [ ] Configure and pass a Continuous Integration (CI) workflow with GitHub Actions.
  - [ ] Write comprehensive user tutorials and API reference documentation with Sphinx.
  - [ ] Prepare and submit manuscript to JOSS (Journal of Open Source Software).

## 🚀 Feature Roadmap

This roadmap outlines the planned features for the AstroKairos suite. The development is phased to deliver a robust and scientifically valuable tool at each stage.

### v1.0: The Binary Star Research Suite

*The primary goal of v1.0 is to be the world's most comprehensive open-source tool for the validation, planning, analysis, and visualization of visual double stars.*

**Core Architecture & Data Layer**
- [ ] **Data Source Abstraction:**
    - [ ] Define abstract `DataSource` interface for modular data access.
    - [ ] Implement `LocalFileDataSource` with robust, fixed-width parsers for local **WDS** and **ORB6** catalogs.
    - [ ] Implement `StelleDoppieDataSource` as a web-scraping alternative for up-to-date data.
- [ ] **Centralized Catalog Hub:**
    - [ ] Implement a `CatalogHub` to query and cross-match data from WDS, ORB6, and external services.
    - [ ] Integrate **SIMBAD** name resolution (via `astroquery`) to find objects by common names (e.g., "Sirius").

**Scientific Engine (`physics` module)**
- [x] **Kepler's Equation Solver:** Implement a high-precision, robust numerical solver by using an advanced initial guess combined with a fast Newton-Raphson refiner.
- [ ] **Orbital Prediction Engine:**
    - [ ] Predict precise ephemerides (PA/Sep) for any date from known orbital elements.
    - [ ] Calculate all orbital anomalies (Mean, Eccentric, True) for analysis.
- [ ] **Physicality & Dynamics Analysis:**
    - [ ] Implement a **Gaia Validator** to confirm or refute physical companionship using parallax and proper motion data.
    - [ ] Implement a **Velocity Vector Analyzer** to calculate apparent motion (`v_total`) and its direction (`PA_v`) from historical data.
    - [ ] Implement a **Dynamic Mass Calculator** to derive total system mass using Kepler's Third Law and Gaia parallaxes.

**User Interfaces & Workflow**
- [ ] **Graphical Planner (GUI - `astrakairos-plan`):**
    - [x] Location Management: Search and select observatories from a built-in JSON database.
    - [x] **Local Conditions Engine:**
        - [x] Calculate precise times for sunset, sunrise, and all three twilights (Civil, Nautical, Astronomical).
        - [x] Compute Moon phase, rise/set times, and position/visibility throughout the night.
    - [x] **"Kairos" Recommender:**
        - [x] Implement the "Optimal Sky Region" calculation (anti-moon/zenith strategy).
    - [ ] **Integrated Workflow:**
        - [ ] Add a pipeline to generate and display a list of high-priority targets directly in the GUI based on the optimal region.
        - [ ] Generate pre-filled search URLs for external databases like Stelle Doppie.
- [ ] **Command-Line Analyzer (CLI - `astrakairos-analyze`):**
    - [x] Process star lists in batch from CSV files.
    - [x] Implement robust command-line argument parsing (`argparse`).
    - [ ] Allow user selection of data source (`local` vs. `web`).
    - [ ] Generate comprehensive analysis reports for each target, including physicality, orbital deviation, mass, etc.
    - [ ] Export detailed, structured results to a new CSV file for further analysis.

**Essential Utilities**
- [ ] **Visualization Module:**
    - [ ] Implement `plot_orbit_and_residuals`, a publication-quality function to plot orbits over observational data and display (O-C) residuals.
- [ ] **I/O & Formatting:**
    - [ ] Implement robust CSV loading and saving functions.
    - [ ] Create utility functions for formatting coordinates and other astronomical data.

**Project Quality & Publication**
- [ ] Achieve high test coverage (>80%) with `pytest` for all core modules.
- [ ] Implement and pass a Continuous Integration (CI) workflow with GitHub Actions.
- [ ] Write comprehensive user tutorials and a full API reference with Sphinx.
- [ ] **Submit v1.0 manuscript to JOSS (Journal of Open Source Software).**

---

### v2.0 & Beyond: The Time-Domain Astrophysics Framework

*The vision for future versions is to expand AstroKairos into a general-purpose platform for time-domain astrophysics, enabling new avenues of research.*

**New Core Modules**
- [ ] **Variable Stars Module:**
    - [ ] Integration with the AAVSO VSX catalog.
    - [ ] Ephemeris engine to predict minima/maxima of periodic variables.
    - [ ] Automated "Comparison Star Finder" tool.
    - [ ] Core time-series analysis tools (e.g., Lomb-Scargle Periodogram).
    - [ ] Light curve plotting and phase-folding visualization functions.

**Advanced Analysis Capabilities (for future research papers)**
- [ ] **Orbit Fitting Engine:** Implement a preliminary orbit calculator (e.g., using least-squares or MCMC) to derive new orbits from historical and new data.
- [ ] **Galactic Kinematics Module:** Add tools to calculate the 3D space velocity (U,V,W) of systems, enabling studies of stellar populations.
- [ ] **Advanced Visualization Suite:** Interactive plots, sky-chart overlays, and 3D orbit visualizations.
- [ ] **Dedicated "Science Pipelines"** for specific research questions, such as:
    - [ ] A pipeline to search for stellar remnants by analyzing astrometric accelerations.
    - [ ] A pipeline to perform statistical analysis on the binary-metallicity-planet connection.
- [ ] **Asteroid/Comet Module:** Implement tools for orbit determination and ephemeris prediction.
- [ ] **Kepler's equation solver with universal variables:** Implement universal variables for a more general Kepler's equation solver (specifically, more robust, with objects like asteroids in mind).

## Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to report issues, propose features, and submit pull requests.

## Acknowledgements & Transparency Notice

This project would not be possible without the foundational work of the astronomical community and the availability of open data.

Special thanks to the creators and maintainers of **Stelle Doppie** for providing a wonderfully accessible, centralized resource for astronomical data. We also extend our gratitude to the **U.S. Naval Observatory (USNO)** for their tireless curation of the **Washington Double Star (WDS) Catalog** and the **Sixth Catalog of Orbits of Visual Binary Stars (ORB6)**, which form the backbone of this work. This research has made use of the **Gaia** data from the European Space Agency (ESA) mission, processed by the Gaia Data Processing and Analysis Consortium (DPAC).

The locations found at /locations.json are derived from **Stellarium**'s location database. Thanks to them too!

This project also stands on the shoulders of giants in the open-source scientific Python ecosystem, including **NumPy**, **Pandas**, **Astropy**, and **PyEphem**.

Additionally, this project was developed with the assistance of AI-powered tools such as OpenAI's **ChatGPT**. These tools were used for tasks including generating boilerplate code, debugging algorithms, and writing/translating documentation. All AI-assisted output was carefully reviewed, tested, and adapted by the human author to ensure its viability and correctness for real-world research environments.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
