# AstroKairos 🌠

<!-- [![PyPI version](https://badge.fury.io/py/astrakairos.svg)](https://badge.fury.io/py/astrakairos) --> <!-- Placeholder: Activate once published to PyPI -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/AstraKairos/astrakairos/actions/workflows/ci.yml/badge.svg)](https://github.com/AstraKairos/astrakairos/actions/workflows/ci.yml)

 **AstroKairos is an integrated scientific framework to find the *kairos*—the opportune moment—for astronomical discovery. From validating binary star orbits to planning variable star photometry.**

This open-source Python suite bridges the gap between massive astronomical catalogs (like Gaia, WDS, and ORB6) and practical night-to-night observation, enabling researchers and amateur astronomers to find and prioritize the most scientifically valuable targets.

## Key Features

- **Intelligent Observation Planner (GUI):** A user-friendly desktop application that recommends the optimal sky region for observation by analyzing local conditions, moon position, and zenith.
- **Physicality Validation Engine:** Uses high-precision data from the Gaia satellite to definitively confirm whether a star pair is a true physical binary or just an optical illusion.
- **Orbital Prediction & Deviation Analysis:** Implements a robust Kepler's equation solver to predict the positions of stars in known orbits and identifies systems whose published orbits are outdated or incorrect.
- **Data-Driven Target Prioritization:** Analyzes catalogs to identify scientifically interesting targets, such as high-velocity systems or neglected historical binaries.
- **Modular Data Sources:** Can operate using local, offline catalogs (for performance and reproducibility) or by scraping up-to-date web sources.

## Installation

AstroKairos is designed to be easily installed via pip.

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

AstroKairos provides two main entry points: a graphical planner and a command-line analyzer.

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
  - [x] Establish professional project structure (`pyproject.toml`, `LICENSE`, etc.).
- [ ] **Phase 1: Data Abstraction Layer**
  - [ ] Define abstract `DataSource` interface.
  - [ ] Implement `LocalFileDataSource` with robust WDS and ORB6 parsers.
  - [ ] Implement `StelleDoppieDataSource` by refactoring web scraping logic.
  - [ ] Implement `GaiaValidator` using `astroquery` for physicality checks.
- [ ] **Phase 2: Scientific Core & Interfaces**
  - [ ] Implement robust Kepler's equation solver in `physics.kepler`.
  - [ ] Implement velocity vector calculations in `physics.dynamics`.
  - [ ] Refactor the Analyzer (CLI) to use the new data and physics layers.
  - [ ] Refactor the Planner (GUI) to be more modular and connect to the analysis pipeline.
- [ ] **Phase 3: Finalization & Documentation**
  - [ ] Achieve >80% test coverage with `pytest`.
  - [ ] Configure and pass a Continuous Integration (CI) workflow with GitHub Actions.
  - [ ] Write comprehensive user tutorials and API reference documentation with Sphinx.
  - [ ] Prepare and submit manuscript to JOSS (Journal of Open Source Software).

### Future Modules (v2.0 and beyond)

- [ ] **Variable Stars Module:** Add tools for photometric analysis, including period finding, light curve plotting, and comparison star selection.
- [ ] **Asteroid/Comet Module:** Implement tools for orbit determination and ephemeris prediction for minor planets.

## Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to report issues, propose features, and submit pull requests.

## Acknowledgements & Transparency Notice

This project would not be possible without the foundational work of the astronomical community and the availability of open data.

Special thanks to the creators and maintainers of **Stelle Doppie** for providing a wonderfully accessible, centralized resource for astronomical data. We also extend our gratitude to the **U.S. Naval Observatory (USNO)** for their tireless curation of the **Washington Double Star (WDS) Catalog** and the **Sixth Catalog of Orbits of Visual Binary Stars (ORB6)**, which form the backbone of this work. This research has made use of the **Gaia** data from the European Space Agency (ESA) mission, processed by the Gaia Data Processing and Analysis Consortium (DPAC).

This project also stands on the shoulders of giants in the open-source scientific Python ecosystem, including **NumPy**, **Pandas**, **Astropy**, and **PyEphem**.

Additionally, this project was developed with the assistance of AI-powered tools such as OpenAI's **ChatGPT**. These tools were used for tasks including generating boilerplate code, debugging algorithms, and writing/translating documentation. All AI-assisted output was carefully reviewed, tested, and adapted by the human author to ensure its viability and correctness for real-world research environments.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
