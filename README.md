# AstraKairos

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
<!--[![Tests](https://github.com/AstraKairos/astrakairos/actions/workflows/ci.yml/badge.svg)](https://github.com/AstraKairos/astrakairos/actions/workflows/ci.yml) -->
<!-- [![PyPI version](https://badge.fury.io/py/astrakairos.svg)](https://badge.fury.io/py/astrakairos) -->

**AstraKairos is an open-source software tool that aims to allow further Binary Star research based on modern peer-reviewed methodologies**

NOTE: AstraKairos was awarded First Place for Astronomy Posters at VI Congreso de Estudiantes de Ciencias Físicas y Astronómicas (CECFA). ![Link to the poster](AstraKairos_Poster_CECFA.pdf).

AstraKairos is still under development for Version 1.0! Some features are still under testing and/or construction. If you are interested in our work, please email us at mrubina@usm.cl

### Main features (v1.0)
- **WDS-Gaia DR3 Crossmatch catalog:** The most complete component-based crossmatch between the Washington Double Star Catalog and Gaia DR3 (status: unpublished; if you'd like to try our crossmatch for your own research, please email us).
- **Chance alignment probability using Machine Learning:** A LightGBM regression model that predicts the R_(chance align) value from El-Badry et al. 2021 for any given system that passes their minimum physicality criteria.
- **The Orbital Deviation Index (ODI):** A metric that quantifies the deviation of a given astrometric measurement (rho, theta, epoch) from the predicted position at the same epoch based on the known orbital parameters.

<!--### Secondary features
- **Astrometrical orbital fitting:** Based on `orbitize!`.
- ** -->

## Detailed information
A more detailed README including an Installation guide and an in-depth review of all features/methods will be published alongside the official documentation in preparation for JOSS publication.


## Acknowledgements & Transparency Notice

This project builds upon the foundational work of the astronomical community and the availability of open data.

**Special recognition to:**
- **U.S. Naval Observatory (USNO)** for maintaining the **Washington Double Star (WDS) Catalog**, the **Washington Double Star Supplemental (WDSS) Catalog**, and the **Sixth Catalog of Orbits of Visual Binary Stars (ORB6)**
- **ESA Gaia Mission** and the **Gaia Data Processing and Analysis Consortium (DPAC)** for publishing the different Gaia Data Releases used in this project.
<!-- - **Stellarium project** for the observatory location database used in `/locations.json` -->

<!--**Software foundation:**
This project leverages the Python ecosystem, particularly **NumPy**, **SciPy**, **Pandas**, **Astropy**, **Skyfield**, and **AstroQuery**. -->

Additionally, this project was developed with the assistance of AI-powered tools such as Google DeepMind's Gemini 3 Pro. These tools were exclusively used for writing the base code of some of AstraKairos' features. All AI output was deeply steered and reviewed by the authors of AstraKairos. AstraKairos' methodology is a result of several months of human work reviewing studies related to the field of Stellar Evolution and Binary Stars.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
