### Latent shift adaptation

Ibrahim Alabdulmohsin, Nicole Chiou, Alexander D’Amour, Arthur Gretton, Sanmi Koyejo, Matt J. Kusner, Stephen R. Pfohl, Olawale Salaudeen, Jessica Schrouff, and Katherine Tsai. "Adapting to Latent Subgroup Shifts via Concepts and Proxies." In International Conference on Artificial Intelligence and Statistics, pp. 9637-9661. PMLR, 2023. Available at https://proceedings.mlr.press/v206/alabdulmohsin23a.

The provided code replicates the synthetic data experiments described in the paper.

The library may be installed as a python package with all its dependencies by executing `pip install .` from the top-level directory.

To replicate the synthetic data experiment, first generate synthetic data and write to a file by executing `colab/synthetic_data_to_file.ipynb`.

Then, execute the following in any order:

  * Evaluate the discrete eigendecomposition approach to reproduce Table 1: `colab/discrete_eigen.ipynb`
  * Train and evaluate models to replicate Table 2: `colab/lsa_synthetic_10seeds.ipynb`
  * Sweep over the strength of the distribution shift and create figure for the WAE method: `colab/synthetic_sweep_experiment.ipynb`
  * Sweep over the strength of the distribution shift and create figure for the continuous spectral method: `colab/continuous_spectrail_method.ipynb`