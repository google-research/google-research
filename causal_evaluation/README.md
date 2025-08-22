# Code to reproduce experiments in "Understanding challenges to the interpretation of disaggregated evaluations of algorithmic fairness"

This repository contains code to reproduce the experiments presented in Pfohl et
al., "Understanding challenges to the interpretation of disaggregated
evaluations of algorithmic fairness" (2025).

To run the code, begin by creating a python virtual environment and installing
the package with its dependencies. We assume that all commands are run from the
base-level directory containing this README file.

```
virtualenv causal_evaluation_env
source causal_evaluation_env/bin/activate
pip install .
```

## Experiments
The steps to reproduce the experiments are as follows:

1.  Run `colab/simulation_fit_models.ipynb` using Jupyter or Colab. This
    notebook generates the simulated data, fits models to the data, and writes
    the predictions to files. The directory where outputs are written may be
    modified. Similarly, the size of the datasets simulated datasets generated
    may also be changed. By default, outputs are written to a directory
    `data/simulation`.

2.  Call `sh experiments/simulation/run_evaluate.sh` to generate metrics and
    bootstrap confidence intervals. The number of bootstrap iterations may be
    modified. If the data directory or the size of the simulated datasets were
    changed in step 1, it is necessary to change the environment variables
    defined at the top of this script to match.

3.  Run `colab/simulation_selection.ipynb` using Jupyter or Colab. This
    self-contained notebook generates the simulated data for the data generating
    processes with selection, generates calibration curves, and saves the
    results to files.

4.  Run `colab/acs_fit_models.ipynb` using Jupyter or Colab. This processes the
    data from the American Community Survey (ACS) Public Use Microdata Sample
    (PUMS), fits models to the data, and writes the predictions to files. As
    before, paths may be modified.

5.  Call `sh experiments/acs_pums/run_evaluate.sh` to generate metrics and
    bootstrap confidence intervals for the ACS PUMS data, building on the task
    definitions provided by the folktables [1] library. The number of bootstrap
    iterations may be modified. If the data directory or the size of the
    simulated datasets were changed in step 4, it is necessary to change the
    environment variables defined at the top of this script to match.

6.  Run `colab/results.ipynb` to visualize the results and write the results to
    files. This covers both the simulation study and the ACS PUMS experiments.

## References
1. Ding, Frances, et al. "Retiring adult: New datasets for fair machine learning." Advances in neural information processing systems 34 (2021): 6478-6490.