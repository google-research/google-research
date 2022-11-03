# Generative Trees: Adversarial and Copycat

This directory contains the companion code for the ICML'22 paper
[Generative Trees: Adversarial and Copycat](https://proceedings.mlr.press/v162/nock22a.html),
by Richard Nock and Mathieu Guillame-Bert.

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

In a shell, run:

```shell
git clone https://github.com/google-research/google-research.git
cd google-research/generative_trees/
run_example.sh
```

At the end of the execution, you will see a list of generated sampled for the
Iris dataset:

```
Display some of the generated samples
Sepal.Length,Sepal.Width,Petal.Length,Petal.Width,class
5.117246154727025,3.294665099621395,1.5415873061790373,0.34693251377205403,setosa
4.938340282187983,3.306772168630169,2.1019090151019775,0.4123936617890174,setosa
5.577975609495907,3.453786064420899,3.671345561310016,0.7218885473617979,versicolor
6.146461600520874,3.7586348414987745,5.222165962947139,2.3442290234292913,versicolor
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
