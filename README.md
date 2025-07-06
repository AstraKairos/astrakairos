# Double Star Celestial Motion Analysis

This project consists of a script designed to analyze the total angular motion (`v_total`) (aka "total observed angular velocity") of double star systems. The script automates the process of:
1.  Loading a list of stars from a CSV file **exported directly from Stelle Doppie**.
2.  Performing massive and concurrent web scraping on the [Stelle Doppie](https://www.stelledoppie.it/) database.
3.  Extracting key astrometric data from the first and last recorded observation for each star.
4.  Calculating the total angular motion based on an astrometric formula.
5.  Presenting a ranking of the stars with the highest angular motion.

The overall goal is to slightly simplify the process of selecting observation candidates.

The use of `asyncio` and `aiohttp` allows processing a large number of stars in a fraction of the time a synchronous script would take.

## Main Features

-   **Concurrent Processing:** Uses `asyncio` and `aiohttp` to perform multiple web requests in parallel, significantly optimizing execution time.
-   **Direct Integration:** Works directly with data exported from the "Export to Excel" function on the `stelledoppie.it` website.
-   **Robust Web Scraping:** Implements scraping logic that handles different types of server responses, including redirects and results pages.
-   **v_total Calculation:** Applies a formula to calculate the total angular velocity from the Position Angle (PA) and Separation (Sep).
-   **Results Ranking:** Sorts the processed stars from highest to lowest `v_total` and presents a clear summary.

## Calculation Methodology

The total angular velocity (`v_total`), expressed in arcseconds per year, is calculated to determine the relative displacement of the components of a double system over time. This is very useful for systems with slow movement, whose orbits are very difficult to compute. Now, for a brief explanation of this formula:

The polar coordinates for each observation are (ρ, θ), where:
-   **ρ (rho)** is the angular separation between the stars (Sep).
-   **θ (theta)** is the position angle (PA).

These coordinates are converted to a Cartesian system (x, y) to calculate the Euclidean distance between the first and last observation:

-   `x = ρ * sin(θ)`
-   `y = ρ * cos(θ)`

The formula for the total angular velocity is:

**v_total = √( (x₂ - x₁)² + (y₂ - y₁)² ) / (t₂ - t₁)**

Where:
-   `(x₁, y₁)` are the Cartesian coordinates at time `t₁` (first observation).
-   `(x₂, y₂)` are the Cartesian coordinates at time `t₂` (last observation).

This methodology is based on the work presented in the following paper:

> Rubina & Hilburn (2025). *Astrometric Observations Suggest that WDS 09483-5904 is not Physically Associated*. [Link to Paper on ResearchGate](https://www.researchgate.net/profile/Martin-Rubina/publication/391600528_Astrometric_Observations_Suggest_that_WDS_09483-5904_is_not_Physically_Associated/links/681e623fbd3f1930dd6f5669/Astrometric-Observations-Suggest-that-WDS-09483-5904-is-not-Physically-Associated.pdf)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TehMartinXz/stelle-doppie-filtering.git
    cd stelle-doppie-filtering
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    Create a `requirements.txt` file with the following content:
    ```
    pandas
    aiohttp
    beautifulsoup4
    ```
    And then install it with pip:
    ```bash
    pip install -r requirements.txt
    ```

    or more directly,
    ```bash
    pip install pandas aiohttp beautifulsoup4
    ```

## Usage

The script is designed to work with the data file obtained directly from the Stelle Doppie website.

**Step 1: Export Data from Stelle Doppie**

1.  Go to the [Stelle Doppie](https://www.stelledoppie.it/index2.php?section=2) website.
2.  Perform your desired search or apply filters.
3.  In the "ACTIONS" menu on the left, click **"Export to Excel"**. This will download the .csv file required for analysis by this script.

**Step 2: Run the Script**

Use the downloaded file as input for the script. Run it from the command line, providing the path to the file and the number of stars you want to process (the ones with the most observations will be selected).

**Syntax:**
```bash
python main.py <exported_csv_file> <number_of_stars>
```

**Example:**
```bash
python main.py "Stelle Doppie.csv" 50
```
This command will process the 50 stars with the most observations from the `Stelle Doppie.csv` file you downloaded.

## Acknowledgements & Transparency Notice

Special thanks to the creators and maintainers of [Stelle Doppie](https://www.stelledoppie.it/) for providing a centralized place to retrieve data for this project.

Additionally, this project was developed with the assistance of AI-powered tools such as OpenAI's ChatGPT. These tools were used for generating boilerplate code, debugging, and writing/translating documentation. All AI-assisted output was carefully reviewed, tested, and adapted to be viable for real-world research environments.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.