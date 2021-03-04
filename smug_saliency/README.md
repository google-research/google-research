# SMUG
Paper: [Scaling Symbolic Methods using Gradients for Neural Model Explanation](https://arxiv.org/abs/2006.16322)

By [Subham S. Sahoo](https://research.google/people/SubhamSekharSahoo/), [Subhashini Venugopalan](https://vsubhashini.github.io), [Li Li](https://research.google/people/LiLi/), [Rishabh Singh](https://research.google/people/RishabhSingh/), [Patrick Riley](https://research.google/people/PatrickRiley/)

# Introduction

This repository contains the implementation of the SMUG saliency method as proposed in the paper [Scaling Symbolic Methods using Gradients for Neural Model Explanation](https://arxiv.org/abs/2006.16322), **ICLR 2021**. SMUG combines gradient-based methods with symbolic techniques to generate explanations for a Neural Network's predictions.

# Dependencies
* python>=3.5
* numpy>=1.19.4
* saliency>=0.0.5
* scikit-image>=0.18.1
* scikit-learn>=0.24.1
* scipy>=1.6.0
* tensorflow>=2.4.1
* tensorflow-datasets>=4.2.0
* tensorflow-estimator>=2.4.0
* z3-solver>=4.8.10.0

# Usage
1. Download the repository and name it `smug_saliency`.
2. In the parent directory, which contains the `smug_saliency` package, run the following commands to ensure that all the tests pass -
  * `chmod+x smug_saliency/run.sh`
  * `./smug_saliency/run.sh`
3. Refer to the `saliency.ipynb` file to check the usage.
