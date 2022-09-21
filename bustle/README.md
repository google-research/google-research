# BUSTLE: Bottom-Up Program Synthesis Through Learning-Guided Exploration

This repository contains the source code associated with the paper published at
ICLR'21 ([OpenReview](https://openreview.net/forum?id=yHeg4PbFHh)):

> Augustus Odena, Kensen Shi, David Bieber, Rishabh Singh, Charles Sutton,
> Hanjun Dai. **BUSTLE: Bottom-Up Program Synthesis Through Learning-Guided
> Exploration.** International Conference on Learning Representations (ICLR),
> 2021.

In this research project, we use a learned model to guide a bottom-up program
synthesis search to efficiently synthesize spreadsheet programs.

To cite this work, you can use the following BibTeX entry:

```
@inproceedings{odena2021bustle,
    title={{BUSTLE}: Bottom-Up Program Synthesis Through Learning-Guided Exploration},
    author={Augustus Odena and Kensen Shi and David Bieber and Rishabh Singh and Charles Sutton and Hanjun Dai},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=yHeg4PbFHh}
}
```

## Setup

Most of this project is written in Java. Use the `compile.sh` script to download
necessary `.jar` files and compile the code. The `clean.sh` script will remove
the downloaded `.jar` files and delete compiled `.class` files that were
produced by `compile.sh`.

Use the `download_sygus_benchmarks.sh` script to download SyGuS benchmark tasks.

We use Python for model training. If you want to train new models, you'll need
Python and TensorFlow.

## Running the Synthesizer

The trained model is provided in the `models` directory. Use the
`run_synthesis.sh` script to run synthesis using the BUSTLE model. This will
output results as JSON files in a new `results` directory.

## Training a New Model

First, generate synthetic training data with the script `generate_data.sh`,
which will produce a JSON file in a new `training_data` directory. Then, run the
`train.sh` script, which will produce model files in the `models` directory.

## Running Tests

Run tests with `run_tests.sh`.

## Disclaimer

This is not an official Google product.
