# Differentially private posets

This directory contains source code to run experiments in the paper
[Privately Counting Partially Ordered Data](https://arxiv.org/abs/2410.06881).

`random_poset.py` contains code for the random poset experiments, and
`nhis_poset.py` contains code for the NHIS experiments. The relevant
`compute_expected_norm_comparison*` functions in each file run the experiments.
`run.sh` provides an example of installing requirements and running one binary.