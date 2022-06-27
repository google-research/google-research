# Evolving Symbolic Density Functionals

This directory contains demonstration of GAS22 functional from paper
"Evolving Symbolic Density Functionals" (https://arxiv.org/abs/2203.02540)

The paper uses the MGCDB84 dataset, which can be downloaded from
https://doi.org/10.1080/00268976.2017.1333644. One can perform SCF calculations
on the dataset using helper functions in syfes/scf/scf.py and parse the results
using the Dataset class in the syfes/dataset/dataset.py. The parsed results can
be dumped to files using the Dataset.save method.

In run.sh, a simple example is provided to perform evolutionary search of the
B97 exchange functional. To run the example, one needs to set the
config.dataset.dataset_directory in run.sh to a directory that contains the SCF
results of MGCDB84.
