# M-Layer

This directory contains code accompanying the paper

[Thomas Fischbacher, Iulia M. Comsa, Krzysztof Potempa, Moritz Firsching,
Luca Versari, Jyrki Alakuijala "Intelligent Matrix Exponentiation", arxiv:2008.03936.](https://arxiv.org/abs/2008.03936)


## Files

  * `m_layer.py`

     The python module containing M-layer as a Keras layer.

  * `M_layer_Experiments.ipynb`

     A colab notebook showing how to use M-layer in some applications.

  * `M_layer_Robustness.ipynb`

     A colab notebook exploring robustness of the M-layer as discussed in the paper.

  * `cifar10_model.npy`

     A trained model for CIFAR10, as used in `M_layer_Robustness.ipynb`.


## How to use

Run the [M-layer experiments colab](https://colab.research.google.com/github/google-research/google-research/blob/master/m_layer/M_Layer_Experiments.ipynb)
or the [M-layer robustness colab](https://colab.research.google.com/github/google-research/google-research/blob/master/m_layer/M_Layer_Robustness.ipynb) directly!

Cloning this part of the google-research github repository is done by installing subversion and running:

```shell
svn export https://github.com/google-research/google-research/trunk/m_layer
```

You can also clone the entire google-research github (without its history) by installing git and running:

```shell
git clone git@github.com:google-research/google-
research.git --depth=1
```

Once you have a copy of `m_layer`, you should be able to import the Keras layer `Mlayer` in your project:

```python3
from m_layer import MLayer
```

The dataset for the periodicity experiment is obtained from [datamarket.com](https://datamarket.com/data/set/2324/daily-minimum-temperatures-in-melbourne-australia-1981-1990#!ds=2324&display=line), as cited by [kaggle.com](https://www.kaggle.com/paulbrabban/daily-minimum-temperatures-in-melbourne).
It was released under a default open license, as recorded by [archive.org](https://web.archive.org/web/20160402201520/https://datamarket.com/data/license/0/default-open-license.html).
