# Generative Trees: Adversarial and Copycat

This directory contains the companion code for the ICML'22 paper "Generative
Trees: Adversarial and Copycat", by Richard Nock and Mathieu Guillame-Bert.

## Citation (BibTex):

```
@inproceedings{ngbGT,
    title={Generative Trees: Adversarial and Copycat},
    author={R. Nock and M. Guillame-Bert},
    booktitle={39$^{~th}$ International Conference on Machine Learning},
    year={2022}
}
```

## Basic usage example

Simply run:

```shell
run_example.sh
```

## Instructions

This code has two key parts: training generative models using the copycat
approach (class Wrapper) and using a pretrained model to just generate examples
or density plots from a pretrained model (class Generate)

Compile with Java and:

*   run 'java Wrapper --help' for help on the options available to train a
    generative tree from data;
*   run 'java Generate --help' for help on the options available to just
    generate data from a pretained model;
*   run script script-missing-data-imputation.sh for the script we used for
    missing data imputation (automates the process, can be edited easily).
