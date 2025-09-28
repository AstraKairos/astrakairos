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

The CLI analyzer processes binary star catalogs to find high-priority observation targets. It supports comprehensive analysis of orbital systems with multiple modes and filtering options.

#### **Prerequisites**

First, create the SQLite database from your catalog files:
```bash
# Convert WDSS and ORB6 catalogs to SQLite database
python scripts/convert_catalogs_to_sqlite.py --wdss-files wdss1.txt wdss2.txt wdss3.txt wdss4.txt --orb6 orb6orbits.txt --output catalogs.db
```

You only need to do this once. The required catalog files are:
- **WDSS Catalogs** (`wdss1.txt`, `wdss2.txt`, `wdss3.txt`, `wdss4.txt`) - Binary star measurements
- **ORB6 Catalog** (`orb6orbits.txt`) - Orbital elements for known binaries

#### **Basic Usage Syntax**

```bash
# Analyze specific targets from CSV file
python -m astrakairos.analyzer.cli <input_file.csv> [options]

# Analyze entire database
python -m astrakairos.analyzer.cli --all [options]
```

#### **Analysis Modes**

**Discovery Mode (`discovery`)** - Default mode for finding fast-moving binary systems:
- **Purpose**: Quickly identify binary star systems with significant apparent motion
- **Method**: Uses endpoint velocity calculation (compares first and last observations)
- **Best for**: Initial surveys, finding high-velocity systems, quick screening of large catalogs
- **Output**: Angular velocities in RA/Dec, total velocity, curvature index for motion assessment
- **Performance**: Fastest mode, suitable for processing thousands of systems

**Characterization Mode (`characterize`)** - Detailed motion analysis with robust statistical methods:
- **Purpose**: Precisely characterize the motion patterns of binary systems over time
- **Method**: Applies Theil-Sen regression (robust against outliers) to fit linear motion models
- **Best for**: Detailed studies of known systems, understanding measurement quality, research-grade analysis
- **Output**: Velocity vectors with uncertainties, fit quality metrics (RMSE), statistical confidence intervals
- **Performance**: Moderate speed, requires multiple observations for meaningful results

**Orbital Mode (`orbital`)** - Priority ranking based on orbital predictions:
- **Purpose**: Identify systems where current observations deviate most from predicted orbital motion
- **Method**: Calculates Observation Priority Index (OPI) by comparing observed positions to orbital predictions
- **Best for**: Planning follow-up observations, finding systems with orbital changes, validating orbital elements
- **Output**: OPI values (higher = more urgent), predicted vs observed separations, orbital uncertainty estimates
- **Performance**: Slowest mode due to orbital calculations, but provides most scientifically valuable prioritization

#### **Complete Command Reference**

**Required Arguments:**
- `input_file` - CSV file containing a `wds_id` column with WDS catalog designations (format: HHMMM±DDMM)
- `--database-path` - Path to the SQLite database created by the catalog conversion script

**Core Analysis Options:**
```bash
--mode {discovery,characterize,orbital}    # Selects analysis algorithm (default: discovery)
--all                                      # Process entire database instead of specific targets
--min-observations N                       # Filter systems with fewer than N observations (default: 2)
--max-observations N                       # Use only the N most recent observations (default: 10)
--concurrent N                    # Number of parallel Gaia queries (default: 20)
```

**Detailed Option Explanations:**

**`--mode` - Analysis Algorithm Selection:**
- `discovery`: Fastest screening method using endpoint velocities
- `characterize`: Statistical motion modeling with uncertainty quantification  
- `orbital`: Orbital prediction comparison for observation prioritization

**`--all` - Database Scope:**
- When specified: Processes all systems in the SQLite database
- When omitted: Processes only systems listed in the input CSV file
- **Performance note**: `--all` can process thousands of systems; use `--limit` to control output size

**`--min-observations` - Data Quality Filter:**
- Excludes systems with insufficient observational data
- Higher values (5-10) improve statistical reliability but reduce sample size
- Lower values (2-3) include more systems but may have unreliable motion estimates

**`--max-observations` - Temporal Focus:**
- Uses only the N most recent observations for analysis
- Helps focus on current orbital motion (older observations may show different behavior)
- Useful for systems with decades of observations where recent motion is most relevant

**`--concurrent` - Performance Tuning:**
- Controls parallel processing for Gaia validation queries
- Higher values: Faster processing but more network load
- Lower values: Slower processing but more stable for unreliable internet connections
- **Recommended**: 10-20 for home internet, 50+ for institutional connections

**Gaia DR3 Validation Options:**
```bash
--gaia-validation                          # Enable physicality assessment using Gaia astrometry
--gaia-p-value FLOAT                      # Statistical threshold for physical association (default: 0.01)
--gaia-radius-factor FLOAT                # Search radius multiplier based on separation (default: 1.2)
--gaia-min-radius FLOAT                   # Minimum search radius in arcseconds (default: 2.0)
--gaia-max-radius FLOAT                   # Maximum search radius in arcseconds (default: 15.0)
```

**Gaia Validation Details:**

**`--gaia-validation` - Physical vs Optical Binary Assessment:**
- Queries Gaia DR3 for precise astrometry of both components
- Compares parallaxes and proper motions to determine if stars are physically associated
- Results: "Likely Physical", "Likely Optical", "Ambiguous", or "Not Available"
- **Note**: Requires internet connection; adds ~1-2 seconds per system

**`--gaia-p-value` - Statistical Confidence Threshold:**
- Lower values (0.001): More conservative, fewer false positives
- Higher values (0.05): More liberal, includes marginal cases
- Default (0.01): Balanced approach following astronomical conventions

**`--gaia-radius-factor` - Adaptive Search Strategy:**
- Multiplies the observed binary separation to set Gaia search radius
- Accounts for orbital motion and measurement uncertainties
- Values > 1.5: May include unrelated field stars
- Values < 1.0: May miss components with significant proper motion

**`--gaia-min-radius` and `--gaia-max-radius` - Search Boundaries:**
- Prevents extremely small searches (miss due to astrometric errors) 
- Prevents extremely large searches (include unrelated sources)
- Automatically scaled based on binary separation and uncertainty

**Output and Display Options:**
```bash
--output FILE                             # Save results to CSV file with full analysis data
--sort-by FIELD                          # Order results by specific metric (mode-dependent defaults)
--limit N                                # Maximum number of results to display/save
```

**Output Control Details:**

**`--output` - Data Export:**
- Saves complete analysis results in CSV format with all calculated metrics
- Includes metadata: analysis mode, timestamp, parameters used
- Format compatible with spreadsheet software and further analysis
- **Tip**: Use descriptive filenames like `high_priority_orbital_2024.csv`

**`--sort-by` - Result Ordering:**
- **Discovery mode options**: `v_total` (total observed angular velocity), `curvature_index` (motion complexity)
- **Characterize mode options**: `rmse` (fit quality), `fit_quality` (statistical confidence)
- **Orbital mode options**: `opi_arcsec_yr` (observation priority), `prediction_uncertainty_arcsec`
- **All modes**: `wds_id` (alphabetical), `physicality_p_value` (Gaia confidence)

**`--limit` - Result Management:**
- Controls both display output and saved file size
- **Recommended**: 50-100 for detailed review, 500+ for comprehensive surveys
- **Performance**: Larger limits don't significantly slow analysis but may overwhelm display

#### **Practical Examples**

**Discovery Mode - Find High-Motion Systems:**
```bash
# Basic discovery analysis - Find fastest-moving binaries in entire database
python -m astrakairos.analyzer.cli --all --database-path catalogs.db --mode discovery --limit 100
# → Identifies top 100 systems with highest apparent motion using endpoint velocity method
# → Useful for: Initial target screening, finding systems requiring urgent observation

# Discovery with Gaia validation and custom output
python -m astrakairos.analyzer.cli --all --database-path catalogs.db --mode discovery \
    --gaia-validation --limit 50 --output high_motion_systems.csv
# → Adds physical vs optical binary classification using Gaia DR3 astrometry
# → Saves results to CSV file for further analysis or observation planning
# → Best for: Research-quality target lists with validated physical associations

# Analyze specific targets for motion
python -m astrakairos.analyzer.cli my_targets.csv --database-path catalogs.db --mode discovery
# → Processes only systems listed in my_targets.csv (must contain wds_id column)
# → Useful for: Follow-up analysis of previously identified interesting systems
```

**Characterization Mode - Detailed Motion Analysis:**
```bash
# Characterize motion patterns with robust fitting
python -m astrakairos.analyzer.cli --all --database-path catalogs.db --mode characterize \
    --min-observations 5 --limit 200 --output motion_characterization.csv
# → Uses Theil-Sen regression to model linear motion with outlier resistance
# → Requires minimum 5 observations for statistical reliability
# → Provides: velocity vectors, uncertainties, fit quality metrics (RMSE)
# → Best for: Understanding measurement precision, detecting non-linear motion

# High-precision characterization with more data points
python -m astrakairos.analyzer.cli selected_targets.csv --database-path catalogs.db \
    --mode characterize --max-observations 20 --sort-by rmse
# → Uses up to 20 most recent observations for detailed temporal analysis
# → Sorts by RMSE (root mean square error) - lower values indicate better linear fits
# → Useful for: Validating orbital motion assumptions, quality assessment
```

**Orbital Mode - Priority Target Identification:**
```bash
# Find highest priority orbital targets
python -m astrakairos.analyzer.cli --all --database-path catalogs.db --mode orbital \
    --limit 50 --output priority_targets.csv
# → Calculates Observation Priority Index (OPI) for all systems with known orbits
# → Higher OPI = larger deviation from predicted orbital motion = more urgent observation
# → Best for: Scheduling follow-up observations, validating orbital elements

# Orbital analysis with Gaia validation and custom parameters
python -m astrakairos.analyzer.cli orbital_candidates.csv --database-path catalogs.db \
    --mode orbital --gaia-validation --gaia-p-value 0.05 --sort-by opi_arcsec_yr
# → Combines orbital priority with physical validation
# → Uses more liberal p-value (0.05) to include marginal physical associations
# → Sorts by OPI value (arcsec/year) for immediate priority assessment

# Conservative orbital analysis (higher observation threshold)
python -m astrakairos.analyzer.cli --all --database-path catalogs.db --mode orbital \
    --min-observations 8 --gaia-validation --limit 25
# → Requires minimum 8 observations for robust orbital comparison
# → Returns only top 25 highest-priority systems with validated physical associations
# → Ideal for: High-confidence target lists for limited observing time
```

**Advanced Usage:**
```bash
# High-throughput analysis with custom concurrency
python -m astrakairos.analyzer.cli --all --database-path catalogs.db --mode discovery \
    --concurrent 50 --limit 1000 --output massive_survey.csv
# → Processes large database with increased parallel Gaia queries
# → Suitable for: Institutional networks, comprehensive sky surveys
# → WARNING: High concurrent requests may overwhelm network or trigger rate limits

# Precision analysis with restrictive Gaia validation
python -m astrakairos.analyzer.cli priority_list.csv --database-path catalogs.db \
    --mode orbital --gaia-validation --gaia-p-value 0.001 --gaia-min-radius 5.0
# → Uses very conservative p-value for highest confidence physical associations
# → Increases minimum search radius for systems with large uncertainties
# → Best for: Publication-quality research requiring highest statistical rigor

# Complete characterization pipeline
python -m astrakairos.analyzer.cli --all --database-path catalogs.db --mode characterize \
    --min-observations 10 --max-observations 25 --gaia-validation --output complete_analysis.csv
# → Comprehensive analysis with strict observation requirements
# → Uses extensive observational baseline (up to 25 observations)
# → Includes both motion characterization and physical validation
# → Suitable for: Comprehensive binary star studies, orbital determination projects
```

**Workflow Examples:**
```bash
# Step 1: Initial discovery survey
python -m astrakairos.analyzer.cli --all --database-path catalogs.db --mode discovery \
    --limit 500 --output initial_survey.csv

# Step 2: Characterize most interesting systems (filter initial_survey.csv to interesting_systems.csv)
python -m astrakairos.analyzer.cli interesting_systems.csv --database-path catalogs.db \
    --mode characterize --gaia-validation --output detailed_analysis.csv

# Step 3: Identify orbital priorities (filter for systems with orbits)
python -m astrakairos.analyzer.cli orbit_systems.csv --database-path catalogs.db \
    --mode orbital --gaia-validation --limit 25 --output observing_priorities.csv
```

#### **Input File Format**

When analyzing specific targets (not using `--all`), provide a CSV file with:
```csv
wds_id
00003+1154
00005+1259
00006+2012
20126+4003
23456+1234
```

The CSV must contain a `wds_id` column with valid WDS designations. You can create this file:
- Manually (as shown above)
- By exporting from the GUI planner
- By filtering results from previous analyses
- From any astronomical database that provides WDS identifiers

#### **Example Output**

```
TOP 10 ANALYSIS RESULTS - ORBITAL MODE (sorted by opi_arcsec_yr)
================================================================================
 1. 07142+2357           | OPI = 12.456 ± 1.234       | Gaia: Likely Physical
 2. 15234+4021           | OPI = 8.912 ± 0.789        | Gaia: Likely Physical  
 3. 23456+1234           | OPI = 6.789 ± 0.456        | Gaia: Ambiguous
 4. 01234+5678           | OPI = 5.234 ± 0.321        | Gaia: Not Available
 5. 18765+4321           | OPI = 4.567 ± 0.234        | Gaia: Likely Physical
```

#### **Sort Field Options**

**Understanding Sort Fields by Analysis Mode:**

**Discovery Mode Sort Options:**
- **`v_total`** (default): Total apparent velocity magnitude combining RA and Dec motion
  - Higher values = faster apparent motion = potentially more interesting systems
  - Useful for identifying systems requiring urgent observation due to rapid changes
- **`v_ra_arcsec_yr`**: Velocity component in Right Ascension direction
  - Shows east-west motion component, important for orbital orientation analysis
- **`v_dec_arcsec_yr`**: Velocity component in Declination direction  
  - Shows north-south motion component, complements RA velocity for full motion vector
- **`curvature_index`**: Measure of non-linear motion detected in observations
  - Higher values suggest orbital motion rather than linear proper motion
  - Useful for distinguishing true binary orbital motion from background star drift

**Characterization Mode Sort Options:**
- **`rmse`** (default): Root Mean Square Error of the linear motion fit
  - Lower values = better linear fit = more reliable velocity measurements
  - Higher values may indicate orbital motion, measurement errors, or stellar variability
- **`fit_quality`**: Statistical confidence metric for the motion model
  - Combines multiple fit statistics into single quality indicator
  - Higher values indicate more reliable characterization results
- **`ra_velocity_arcsec_per_year`**: Precisely measured RA velocity with uncertainties
  - From robust Theil-Sen regression, resistant to outlier observations
- **`dec_velocity_arcsec_per_year`**: Precisely measured Dec velocity with uncertainties
  - Complements RA velocity for complete motion characterization

**Orbital Mode Sort Options:**
- **`opi_arcsec_yr`** (default): Observation Priority Index in arcseconds per year
  - Higher values = larger deviation from predicted orbital motion = higher observation priority
  - Directly indicates which systems most urgently need follow-up observations
- **`predicted_sep_arcsec`**: Current predicted separation from orbital elements
  - Useful for planning observations (smaller separations need higher resolution)
- **`prediction_uncertainty_arcsec`**: Uncertainty in orbital position prediction
  - Higher values indicate less reliable orbital elements needing confirmation
- **`orbital_period_years`**: Period of the binary system orbit
  - Shorter periods may show more rapid changes requiring frequent monitoring

**Universal Sort Options (All Modes):**
- **`wds_id`**: Washington Double Star catalog designation (alphabetical sorting)
  - Useful for systematic processing or cross-referencing with other catalogs
- **`physicality_p_value`**: Gaia-based statistical confidence of physical association
  - Lower values = higher confidence that components are physically bound
  - Only available when `--gaia-validation` is enabled
- **`gaia_separation_arcsec`**: Current separation measured by Gaia DR3
  - Independent verification of binary separation from high-precision astrometry
- **`observation_count`**: Number of historical observations available
  - More observations generally provide more reliable analysis results

**Sorting Strategy Recommendations:**

**For Target Discovery:**
- Sort by `v_total` descending to find fastest-moving systems
- Use `curvature_index` descending to find systems showing orbital motion signatures

**For Observation Planning:**
- Sort by `opi_arcsec_yr` descending to prioritize most urgent follow-ups  
- Use `predicted_sep_arcsec` ascending to group systems by required telescope resolution

**For Data Quality Assessment:**
- Sort by `rmse` ascending to find most reliable motion measurements
- Use `observation_count` descending to prioritize well-observed systems

**For Research Publications:**
- Sort by `physicality_p_value` ascending to focus on confirmed physical binaries
- Combine with appropriate mode-specific metrics for scientific relevance

#### **Performance Tips**

**Database and System Optimization:**
- **Use `--limit` for large databases** to avoid overwhelming output and improve processing speed
  - Start with small limits (50-100) for initial exploration
  - Increase gradually based on system performance and analysis needs
- **Optimize `--concurrent` based on your system:**
  - **Home internet**: 10-20 concurrent requests (stable but not overwhelming)
  - **Institutional networks**: 30-50 concurrent requests (faster processing)
  - **High-performance systems**: 50+ concurrent requests (maximum throughput)
  - **Unstable connections**: 5-10 concurrent requests (prioritize reliability)

**Data Quality and Filtering:**
- **Use `--min-observations` strategically:**
  - **2-3 observations**: Maximum catalog coverage but lower reliability
  - **5-8 observations**: Balanced approach for most analyses  
  - **10+ observations**: High-quality results for publication work
- **Apply `--max-observations` for temporal focus:**
  - **5-10 observations**: Recent motion analysis (last decade)
  - **15-20 observations**: Extended baseline for long-period systems
  - **25+ observations**: Complete observational history analysis

**Network and Performance Considerations:**
- **Enable `--gaia-validation` only when needed:**
  - Adds 1-3 seconds per system due to external database queries
  - Essential for research work but optional for preliminary surveys
  - Ensure stable internet connection before processing large batches
- **Batch processing strategies:**
  - Process in chunks: Use `--limit` with multiple runs for very large databases
  - Save intermediate results: Use `--output` to preserve progress
  - Monitor system resources: Large analyses may consume significant RAM

**Mode-Specific Performance:**
- **Discovery mode**: Fastest processing, suitable for databases with 10,000+ systems
- **Characterization mode**: Moderate speed, best for 1,000-5,000 systems per run
- **Orbital mode**: Slowest due to complex calculations, optimal for 100-1,000 systems

**Troubleshooting Common Issues:**
- **Memory usage**: Large databases may require 2-8 GB RAM; close other applications if needed
- **Network timeouts**: Reduce `--concurrent` if Gaia queries fail frequently
- **Disk space**: Output CSV files can be large; ensure adequate storage for results
- **Processing time**: Orbital mode with Gaia validation may take 30+ minutes for large datasets

**Optimization Workflow:**
```bash
# Step 1: Quick survey to estimate processing time
python -m astrakairos.analyzer.cli --all --database-path catalogs.db --mode discovery --limit 10

# Step 2: Scale up based on performance
python -m astrakairos.analyzer.cli --all --database-path catalogs.db --mode discovery --limit 1000

# Step 3: Add validation for final results  
python -m astrakairos.analyzer.cli selected_targets.csv --database-path catalogs.db \
    --mode orbital --gaia-validation --concurrent 20
```

#### **Help and Documentation**

```bash
# Display complete help
python -m astrakairos.analyzer.cli --help

# Check version
python -m astrakairos.analyzer.cli --version
```

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

### Future Development: The Time-Domain Astrophysics Framework 🔮
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