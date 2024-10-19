# Generative Forests

This directory contains the companion code for the paper
[Generative Forests](https://arxiv.org/abs/2308.03648),
by Richard Nock and Mathieu Guillame-Bert.

## Basic usage example

In a shell, run:

```shell
git clone https://github.com/google-research/google-research.git
cd google-research/generative_forests/src

# Run compile twice
./compile_all.sh
./compile_all.sh

# Print summary of the command line and explanations of parameters
java Wrapper --help

# A script run on the 5 folds (Cf paper) of the UCI winered dataset, provided
# -- parameters self explained (MAKE SURE TO EDIT THE FILE with appropriate
# paths) ==> this generates and stores datasets that can be compared with the
# corresponding "_test.csv" datasets in the 5 folders
./icml24_script-generate.sh
```
