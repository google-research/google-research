<!-- disableFinding(WHITESPACE_LINES) -->

# JAX-RTM: JAX-Differentiable Radiative Transfer Model

JAX-RTM is a high-performance, fully JAX-differentiable Radiative Transfer
Model (RTM) solver designed for satellite simulation and meteorological data
assimilation. This repository implements a calibrated physical solver optimized
for cloud and ice microphysics simulation.

By leveraging JAX, this simulator is fully differentiable, allowing
TPU/GPU hardware acceleration as well as gradient-based optimization of
cloud properties (like ice water content, effective radius, and
spatial alignment) directly against satellite observations.

---

## Installation

### 1. Clone the Repository

Because `google-research` is a large monorepo containing many projects, we
highly recommend using Git's **sparse-checkout** to clone only the `jax_rtm`
directory, saving gigabytes of download:

```bash
# Clone the repository metadata without downloading any files
git clone --filter=blob:none --no-checkout https://github.com/google-research/google-research.git
cd google-research

# Configure Git to only checkout the jax_rtm subdirectory
git sparse-checkout set jax_rtm

# Checkout the files
git checkout master

# Navigate to the package directory
cd jax_rtm
```

### 2. Set Up Virtual Environment & Dependencies

We highly recommend using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 3. Download the Ice Properties Database (Required)

The simulator requires the pre-compiled **Ping Yang Ice Crystal
Single-Scattering Database** (155 MB) to run. Download it automatically
using the provided script:

```bash
python3 download_data.py
```

*Note: The raw database consists of 27 GB of ASCII tables. We host a
pre-compiled, compressed NumPy grid (`ping_yang_multi_habit.npz`) for
convenient distribution.*

---

## Quick Start: Run the Example

Once installed, you can run a sample simulation over a downsampled ERA5
weather grid:

```bash
python3 examples/example_simulation.py
```

This script will initialize the JIT-compiled simulator, run a forward pass,
and save an **Ash RGB** composite image to
`examples/simulated_ash_rgb_85x85.png`.

---

## Running Tests

To run the unit test suite:

```bash
# Using the standard runner
./run.sh

# Or directly via pytest
pytest
```

*The test suite includes 5 unit tests verifying physical solver correctness,
radiative energy conservation, and Kirchhoff's law.*

---

## Citing JAX-RTM

If you use this code or its outputs in your research, please cite the JAX-RTM
software release:

- **DOI**: [`https://doi.org/10.5281/zenodo.21228209`]


## Citations & Licensing Details

### Ping Yang Ice Database

The ice crystal single-scattering properties database was developed by Ping
Yang's group and is hosted on Zenodo.

- **Source**: [Zenodo Record 5348402](https://zenodo.org/records/5348402)
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode))
- **Citation**:
  > Yang, P., Bi, L., Baum, B. A., Liou, K. N., Kattawar, G. W., Mishchenko,
  > M. I., & Cole, B. (2013). Spectrally Consistent Scattering, Absorption,
  > and Polarization Properties of Atmospheric Ice Crystals at Wavelengths
  > from 0.2 to 100 μm. *Journal of the Atmospheric Sciences*, 70(1), 330-347.
  > [DOI Link](https://doi.org/10.1175/JAS-D-12-039.1)

### ERA5 Reanalysis & ARCO-ERA5

The meteorological profiles used in our examples and unit tests are derived
from the **ERA5 Reanalysis** dataset, accessed via the Google Cloud
**Analysis-Ready Cloud-Optimized (ARCO) ERA5** public dataset.

- **Source**: [ARCO-ERA5 GCS Bucket](https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5) & [Google Cloud Marketplace](https://console.cloud.google.com/marketplace/product/bigquery-public-data/arco-era5)
- **License**: Copernicus License Agreement ([ECMWF Terms of Use](https://cds.climate.copernicus.eu/disclaimer-privacy)). Free, worldwide, non-exclusive redistribution and adaptation are permitted for any purpose, including commercial.
- **Attribution Notice**:
  > *Contains modified Copernicus Climate Change Service information [2026].
  > Neither the European Commission nor ECMWF is responsible for any use
  > that may be made of the Copernicus information or data it contains.*
- **Citation**:
  > Carver, R. W., & Merose, A. (2023). ARCO-ERA5: An Analysis-Ready
  > Cloud-Optimized Reanalysis Dataset. *22nd Conference on Artificial
  > Intelligence for Environmental Science, American Meteorological Society*,
  > 4A.1. [AMS Presentation](https://ams.confex.com/ams/103ANNUAL/meetingapp.cgi/Paper/415842)

### Calibrated Model Parameters (Candidate 992)

The optimal physical tuning parameters contained in `data/params_992.json`
(such as ice/water extinction scaling, cloud overlap parameters, and
convective thresholds) were developed and optimized using the AI-assisted
empirical software engineering system detailed in:

- **Citation**:
  > Aygün, E., Belyaeva, A., Comanici, G., Coram, M., Cui, H., Garrison, J.,
  > ... & Brenner, M. P. (2026). An AI system to help scientists write
  > expert-level empirical software. *Nature*, s41586-026-10658-6.
  > [DOI Link](https://doi.org/10.1038/s41586-026-10658-6)

### Ice & Snow Cloud Microphysics (Adapted)

The ice-phase microphysical and radiative calculations in `microphysics.py`
incorporate two key physical adaptations optimized by the ERA search for
Candidate 992:

*   **Wyser Ice Particle Sizing (Adapted)**: The ice crystal effective radius
    ($r_{eff}$) sizing model is based on the Wyser empirical formulation,
    calibrated for geostationary thermal infrared channels using parameter
    values (`wyser_offset = -1.78`, `wyser_slope = 1.15e-3`) discovered
    during the ERA search.
    *   **Original Citation**:
        > Wyser, K. (1998). The effective radius in ice clouds. *Journal of
        > Climate*, 11(8), 1793-1802.
        > [DOI Link](https://doi.org/10.1175/1520-0442(1998)011%3C1793:TERIIC%3E2.0.CO;2)
*   **Ice-Snow Radiative Coupling (ECMWF IFS Pattern)**: To compute total
    radiatively active ice optical properties, the model combines cloud ice
    water content (`ciwc`) and cloud snow water content (`cswc`) as
    radiatively active frozen hydrometeors, following the radiative coupling
    pattern of the ECMWF Integrated Forecasting System (IFS). The relative
    radiative weights/multipliers (`ciwc_multiplier = 5.65`,
    `cswc_multiplier = 3.75`) were optimized by the ERA search for Candidate
    992.
    *   **Reference**:
        > ECMWF (2015). *IFS Documentation CY41R2, Part IV: Physical
        > Processes, Chapter 2: Radiation*. European Centre for Medium-Range
        > Weather Forecasts. [DOI Link](https://doi.org/10.21957/tr5rv27xu)

### Kokhanovsky Benchmarks

Validation benchmarks for the Adding-Doubling solver are transcribed from:

- **Citation**:
  > Kokhanovsky, A. A., Budak, V. P., Cornet, C., Duan, M., Emde, C., Katsev,
  > I. L., ... & Zege, E. P. (2010). Benchmark results in vector radiative
  > transfer. *Journal of Quantitative Spectroscopy and Radiative Transfer*,
  > 111(12-13), 1931-1946.
  > [DOI Link](https://doi.org/10.1016/j.jqsrt.2010.03.005)

### Cox-Munk Ocean Emissivity

The view-angle and wind-speed dependent rough ocean emissivity model in
`microphysics.py` is based on the classic Cox-Munk wave slope distribution.

- **Citation**:
  > Cox, C., & Munk, W. (1954). Measurement of the Roughness of the Sea
  > Surface from Photographs of the Sun's Glitter. *Journal of the Optical
  > Society of America*, 44(11), 838-850.
  > [DOI Link](https://doi.org/10.1364/JOSA.44.000838)

### Liquid Cloud Microphysics

The parameterization for liquid water cloud droplet effective radius in
`microphysics.py` is based on the formulation by Martin et al. (1994), which
relates effective radius to liquid water content and droplet number
concentration over land and ocean.

- **Citation**:
  > Martin, G. M., Johnson, D. W., & Spice, A. (1994). The measurement and
  > parameterization of effective radius of droplets in warm stratocumulus
  > clouds. *Journal of the Atmospheric Sciences*, 51(13), 1823-1842.
  > [DOI Link](https://doi.org/10.1175/1520-0469(1994)051%3C1823:TMAPOE%3E2.0.CO;2)

### Delta-M Truncation

To handle highly asymmetric phase functions in the Adding-Doubling solver,
the model utilizes the Delta-M truncation method in `adding_doubling.py` to
scale optical depth, single-scattering albedo, and Legendre expansion
coefficients.

- **Citation**:
  > Wiscombe, W. J. (1977). The Delta–M Method: Rapid Yet Accurate Radiative
  > Flux Calculations for Strongly Asymmetric Phase Functions. *Journal of
  > the Atmospheric Sciences*, 34(9), 1408-1422.
  > [DOI Link](https://doi.org/10.1175/1520-0469(1977)034%3C1408:TDMRYA%3E2.0.CO;2)

### Ash RGB Visualization

The Ash RGB composition in `camera.py` and examples follows the standard
EUMETSAT multi-spectral composite recipe designed for detecting volcanic ash
and sulfur dioxide.

- **Reference**:
  > Kerkmann, J., Lutz, H. J., & Koenig, M. (2003). MSG RGB Recipes.
  > *EUMETSAT*, 1-13.

---

## Disclaimer

This is not an officially supported Google product.
