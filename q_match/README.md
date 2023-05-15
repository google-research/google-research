# Q-Match: Self-Supervised Learning

This repository contains code for our Q-Match [paper](https://arxiv.org/abs/2302.05444).  Our code runs our algorithm, Q-Match, along with other popular methods such as VICReg, SimCLR, VIME, etc.

**Note**: Our code can be used to benchmark future self-supervised learning methods and datasets.

## Experiments

To launch the few-shot learning experiments (on 1% datasets) for an algorithm--for example, Q-Match--run

`sh scripts/one_percent/q_match.sh`

This will run experiments using the tuned hyperparameters for both the linear classification and the fine tuning classification tasks on all four datasets.

## Installation

The code was tested using Python 3.10.9 and the library versions listed in `requirements.txt`.  Use `virtualenv` to create a new environment, and install the python packages with

```bash
pip install -r requirements.txt
```

## Datasets

The Higgs and MNIST datasets are available from TFDS, but the CoverType and Adult datasets can be downloaded from UCI.

* [CoverType](https://archive.ics.uci.edu/ml/datasets/covertype)

* [Adult](https://archive.ics.uci.edu/ml/datasets/Adult)

 Use our `create_adult_datasets.py` and `create_covertype_datasets.py` scripts to create the splits for these datasets.


## How to Cite
If you use or extend this work, please cite the [paper]((https://arxiv.org/abs/2302.05444)) where it was introduced.

 ```bibtex
 @misc{https://doi.org/10.48550/arxiv.2302.05444,
  doi = {10.48550/ARXIV.2302.05444},

  url = {https://arxiv.org/abs/2302.05444},

  author = {Mulc, Thomas and Dwibedi, Debidatta},

  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},

  title = {Q-Match: Self-supervised Learning by Matching Distributions Induced by a Queue},

  publisher = {arXiv},

  year = {2023},

  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Disclaimer
This is not an officially supported Google product.
