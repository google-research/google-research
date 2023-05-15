### Latent shift adaptation

Code to accompany "Adapting to Latent Subgroup Shifts via Concepts and Proxies" by Ibrahim Alabdulmohsin, Nicole Chiou, Alexander D'Amour, Arthur Gretton, Sanmi Koyejo, Matt J. Kusner, Stephen R. Pfohl, Olawale Salaudeen, Jessica Schrouff, Katherine Tsai. Available at https://arxiv.org/abs/2212.11254. 

The provided code replicates a portion of the synthetic data experiment described in the paper. Code for continuous spectral method described in the paper is forthcoming. 

The library may be installed as a python package with all its dependencies by executing `pip install .` from the top-level directory.

To replicate the synthetic data experiment, first generate synthetic data and write to a file by executing `colab/synthetic_data_to_file.ipynb`.

Then, execute the following in any order:

  * Evaluate the discrete eigendecomposition approach to reproduce Table 1: `colab/discrete_eigen.ipynb`
  * Train and evaluate models to replicate Table 2: `colab/lsa_synthetic_10seeds.ipynb`
  * Sweep over the strength of the distribution shift and create figure: `colab/synthetic_sweep_experiment.ipynb`