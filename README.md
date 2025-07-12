# AstraKairos 🌠

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/AstraKairos/astrakairos/actions/workflows/ci.yml/badge.svg)](https://github.com/AstraKairos/astrakairos/actions/workflows/ci.yml)
<!-- [![PyPI version](https://badge.fury.io/py/astrakairos.svg)](https://badge.fury.io/py/astrakairos) -->

**AstraKairos is an integrated scientific framework to find the *kairos*—the opportune moment—for astronomical discovery. From validating binary star orbits to planning variable star photometry, this open-source Python suite bridges the gap between massive astronomical catalogs and practical night-to-night observation.**

AstraKairos empowers researchers and amateur astronomers to identify, prioritize, and analyze the most scientifically valuable targets in the night sky.

## Key Features

- **Intelligent Observation Planner (GUI):** A user-friendly desktop application that recommends the optimal sky region for observation by analyzing your location, date, moon position, and other local conditions.
- **Data-Driven Target Prioritization:** A powerful command-line tool that analyzes star catalogs to find high-priority targets. It calculates:
    - **Apparent Motion Vectors:** Identifies high-velocity systems or pairs with significant relative motion.
    - **Observation Priority Index (OPI):** A novel metric that quantifies how much a star's observed position deviates from its published orbit, highlighting systems that urgently need new measurements.
- **Physicality Validation Engine:** Uses high-precision data from the Gaia satellite to help determine if a star pair is a true physical binary or a chance optical alignment.
- **Orbital Prediction Engine:** Implements a robust Kepler's equation solver to predict the future positions of stars in known orbits.
- **Modular Data Sources:** Can operate using local, offline catalogs (for performance and reproducibility) or by scraping up-to-date web sources.

## Installation

AstraKairos is designed for easy installation via pip. A virtual environment is highly recommended.

```bash
# Create and activate a virtual environment
python -m venv venv
# On Windows:   .\venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Install from PyPI (once published)
# pip install astrakairos

# Install the latest development version directly from GitHub
pip install git+https://github.com/AstraKairos/astrakairos.git
```

## Usage

AstraKairos provides two main entry points: a graphical planner and a command-line analyzer.

### 1. Observation Planner (GUI)

Launch the user-friendly observation planner from your terminal:

```bash
astrakairos-plan
```
This will open a desktop application where you can select your observatory, a date, and get recommendations for where to point your telescope.

### 2. Data Analyzer (CLI)

Process a list of stars from a CSV file to find high-priority targets. For example, to analyze `targets.csv`, sort by the highest OPI, and limit to the top 10 results:

```bash
astrakairos-analyze targets.csv --source local --wds-file wds.txt --orb6-file orb6.txt --sort-by opi --limit 10
```

For a full list of options and commands, use the help flag:
```bash
astrakairos-analyze --help
```

## Project Roadmap

The development of AstraKairos is phased to deliver a robust and scientifically valuable tool at each stage.

### v1.0: The Binary Star Research Suite
*The goal of v1.0 is to be a world-class open-source tool for the validation, analysis, and observation planning of visual double stars.*

**Core Architecture & Data Layer (`[ ]` In Progress)**
- [x] Establish professional project structure with `pyproject.toml`.
- [ ] **Data Source Abstraction:** Define a modular `DataSource` interface.
- [ ] **Local Source:** Implement robust, fixed-width parsers for offline **WDS** and **ORB6** catalogs.
- [ ] **Web Source:** Implement a web-scraping source for up-to-date data.
- [ ] **Catalog Hub:** Implement cross-matching and **SIMBAD** name resolution.

**Scientific Engine (`[x]` Core Complete)**
- [x] **Kepler's Equation Solver:** Implement a high-precision, robust numerical solver with a hybrid initial guess strategy.
- [x] **Orbital Prediction:** Predict precise ephemerides (PA/Sep) from orbital elements.
- [x] **Dynamics Analysis:** Calculate apparent motion vectors (`v_total`, `PA_v`).
- [x] **Observation Priority Index (OPI):** Implement the OPI to rank targets based on orbital deviation.
- [ ] **Physicality Validation:** Fully integrate **Gaia** parallax and proper motion checks.
- [ ] **Mass Calculation:** Implement dynamic mass calculation using Kepler's Third Law.

**User Interfaces & Workflow (`[ ]` In Progress)**
- [x] **Planner (GUI):** Core functionality for location, date, and local conditions (sun/moon) is implemented.
- [ ] Refine Planner GUI to display generated target lists directly.
- [x] **Analyzer (CLI):** Core functionality for batch processing from CSV is implemented.
- [ ] Refine Analyzer CLI with comprehensive report generation and CSV export.

**Project Quality & Publication (`[ ]` To Do)**
- [ ] Achieve >80% test coverage with `pytest`.
- [ ] Configure and pass a Continuous Integration (CI) workflow with GitHub Actions.
- [ ] Write comprehensive user tutorials and API reference documentation with Sphinx.
- [ ] **Submit v1.0 manuscript to a suitable journal (e.g., JOSS).**

---

### v2.0 & Beyond: The Time-Domain Astrophysics Framework
*Expand AstraKairos into a general-purpose platform for time-domain astrophysics.*

- [ ] **Variable Stars Module:**
    - [ ] Integrate with the AAVSO VSX catalog and predict minima/maxima.
    - [ ] Implement a "Comparison Star Finder" tool.
    - [ ] Add time-series analysis tools (e.g., Lomb-Scargle Periodogram).
- [ ] **Advanced Analysis & Visualization:**
    - [ ] Implement an **Orbit Fitting Engine** (e.g., using MCMC) to derive new orbits.
    - [ ] Implement a **Universal Variable Solver** for parabolic/hyperbolic orbits (comets, interstellar objects).
    - [ ] Add a **Statistical Analysis Module** with error propagation for all calculations.
    - [ ] Create an advanced visualization suite with interactive plots and sky-chart overlays.
- [ ] **Asteroid/Comet Module:** Add tools for orbit determination and ephemeris prediction.

## Contributing

Contributions are welcome! Whether you're reporting a bug, proposing a new feature, or submitting code, your input is valuable. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgements & Transparency Notice

This project would not be possible without the foundational work of the astronomical community and the availability of open data.

Special thanks to the creators and maintainers of **Stelle Doppie** for providing a wonderfully accessible, centralized resource for astronomical data. We also extend our gratitude to the **U.S. Naval Observatory (USNO)** for their tireless curation of the **Washington Double Star (WDS) Catalog** and the **Sixth Catalog of Orbits of Visual Binary Stars (ORB6)**, which form the backbone of this work. This research has made use of the **Gaia** data from the European Space Agency (ESA) mission, processed by the Gaia Data Processing and Analysis Consortium (DPAC).

The locations found at /locations.json are derived from **Stellarium**'s location database. Thanks to them too!

This project also stands on the shoulders of giants in the open-source scientific Python ecosystem, including **NumPy**, **SciPy**, **Pandas**, **Astropy**, and **Skyfield**.

Additionally, this project was developed with the assistance of AI-powered tools such as OpenAI's **ChatGPT**. These tools were used for tasks including generating boilerplate code, debugging algorithms, and writing/translating documentation. All AI-assisted output was carefully reviewed, tested, and adapted by the human author to ensure its viability and correctness for real-world research environments.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.