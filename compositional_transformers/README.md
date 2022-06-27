# Making Transformers Solve Compositional Tasks

This repository contains source code for the paper "Making Transformers Solve
Compositional Tasks", ACL 2022.

Arxiv pre-print: https://arxiv.org/abs/2108.04378

## Requirements

The codebase assumes it is going to run in Google Colaboratory (or Jupyter
notebook with Tensorflow installed), and that a GPU is available in the runtime
for use by Tensorflow.

## Datasets

The Addition, AdditionNegatives, Reversing, Duplication, Cartesian and
Intersection datasets are synthetic, and are generated automatically on the fly.
The SCAN, PCFG, COGS and CFQ dataset variants are external and can be
downloaded from:

-   SCAN: (https://github.com/brendenlake/SCAN)
-   PCFG: (https://github.com/i-machine-think/am-i-compositional)
-   COGS: (https://github.com/najoungkim/COGS)
-   CFQ: (https://github.com/google-research/google-research/tree/master/cfq)

Once downloaded, please modify the appropriate folders in the colab notebooks,
so the code can find the dataset files.

## Running seq2seq experiments

Evaluate all the cells in the "seq2seq.ipynb" notebook, except for the last two.
The last two cells contain the code to actually run the experiments. The second
to last cell allows running individual experiments, and the last cell would run
all the experiments in the paper one after another. This might take several days
to complete, but datasets/models can be added/removed from the experiments
modifying the "datasets" and "models" arrays, and the number of repetitions of
each experiment can be controlled with the "num_repetitions" variable.

## COGS sequence tagging experiments

The "cogs_seq_tagging_data_gen.ipynb" notebook can be used to generate the
sequence tagging version of the COGS dataset, given the original dataset. After
that, the "cogs_tagging.ipynb" is the equivalent version of the "seq2seq.iypnb"
notebook, but for sequence tagging models.
