# JRL

The JRL repository is a research codebase for offline RL research, implemented
in Jax with the use of the Acme RL library. This repository contains the
reference implementation for the following works:

* "Why so pessimistic? Estimating uncertainties for offline RL through ensembles, and why their independence matters."

Note: For using this codebase we recommend creating a conda virtual environment
as follows: `conda env create -f requirements.yml`

## Repository Structure:

### Agents
Please refer to `jrl/agents/README.md` for instructions on how to train various
agents, as well as how to add your own agents.

### Datasets
Please refer to `jrl/data/README.md` for instructions on how to add new datasets.

### Environments
Please refer to `jrl/envs/README.md` for instructions on how to add new
environments.
