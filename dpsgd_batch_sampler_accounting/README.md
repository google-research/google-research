# Privacy Accounting for DP-SGD with Batch Samplers

This directory contains code for obtaining bounds on privacy accounting of
Differentially Private Stochastic Gradient Descent (DP-SGD) for various
batch samplers, as developed in the following papers:

> **Title:** How Private are DP-SGD implementations?\
> **Authors:** Lynn Chua, Badih Ghazi, Pritish Kamath, Ravi Kumar,
               Pasin Manurangsi, Amer Sinha, Chiyuan Zhang\
> **Link:** https://arxiv.org/abs/2403.17673 (Appeared at ICML 2024)

> **Title:** Scalable DP-SGD: Shuffling vs. Poisson Subsampling\
> **Authors:** Lynn Chua, Badih Ghazi, Pritish Kamath, Ravi Kumar,
               Pasin Manurangsi, Amer Sinha, Chiyuan Zhang\
> **Link:** https://arxiv.org/abs/2411.04205 (Appeared at NeurIPS 2024)

> **Title:** Balls-and-Bins Sampling for DP-SGD\
> **Authors:** Lynn Chua, Badih Ghazi, Charlie Harrison, Ethan Leeman,
               Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Amer Sinha,
               Chiyuan Zhang\
> **Link:** https://arxiv.org/abs/2412.16802 (To appear at AISTATS 2025)

Note: This is not an officially supported Google product.

## Requirements
The codebase depends on common Python packages of
[numpy](https://pypi.org/project/numpy/),
[scipy](https://pypi.org/project/scipy/) and
[pandas](https://pypi.org/project/pandas/) (for pretty printing of outputs in
the demonstration example). In addition, it depends on the Google
[dp_accounting](https://pypi.org/project/dp-accounting/) package.

You can install the dependencies using:

```
pip install -r requirements.txt
```

## Getting Started

The `example.py` file provides a simple example for using all the different
accounting methods in this codebase. To run it, simply run:

```
python3 -m dpsgd_batch_sampler_accounting.example
```

The example does not cover all the functionality, and we recommend looking at
the source code corresponding to the particular classes created in `example.py`
to understand the other methods supported by the classes.

## Citations

```
@inproceedings{chua24howprivate,
  author    = {Lynn Chua and Badih Ghazi and Pritish Kamath and Ravi Kumar and
               Pasin Manurangsi and Amer Sinha and Chiyuan Zhang},
  title     = {How Private are {DP-SGD} Implementations?},
  booktitle = {International Conference on Machine Learning, {ICML}},
  publisher = {OpenReview.net},
  year      = {2024},
  url       = {https://openreview.net/forum?id=xWI0MKwJSS},
}

@inproceedings{chua24scalable,
  author    = {Lynn Chua and Badih Ghazi and Pritish Kamath and Ravi Kumar and
               Pasin Manurangsi and Amer Sinha and Chiyuan Zhang},
  title     = {Scalable DP-SGD: Shuffling vs. Poisson Subsampling},
  booktitle = {Advances in Neural Information Processing Systems {NeurIPS}},
  year      = {2023},
  url       = {https://www.arxiv.org/abs/2411.04205},
}

@misc{chua24ballsandbins,
  author        = {Lynn Chua and Badih Ghazi and Charlie Harrison and
                   Ethan Leeman and Pritish Kamath and Ravi Kumar and
                   Pasin Manurangsi and Amer Sinha and Chiyuan Zhang},
  title         = {Balls-and-Bins Sampling for DP-SGD},
  year          = {2024},
  eprint        = {2412.16802},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2412.16802},
}
```