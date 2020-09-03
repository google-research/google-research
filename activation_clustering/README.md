# Activation Clustering

## Overview

This is an implementation of the Activation Clustering model used in [Explaining Deep Neural Networks using Unsupervised Clustering](https://arxiv.org/abs/2007.07477).

## Instructions

1. Clone the repository:

```
git clone https://github.com/google-research/google-research.git --depth=1
```

1. Build the docker image (includes downloading example model and notebooks):

```
cd google-research/activation_clustering && docker build -t activation_clustering .
```

1. Run the docker image as follows, which starts a jupyter server in the docker container and allows access from the host machine:

```
docker run -it -p 8888:8888 activation_clustering
```

1. In a web browser, navigate to the address shown by the docker run command above, it should look something like `http://127.0.0.1:8888/?token=1234abcd...`.

1. Follow the example notebooks:

* For training an activation clustering model from a baseline model: [train](examples/cifar10/train.ipynb)

* For use cases of activation clustering model: [similar](examples/cifar10/similar_images_concepts.ipynb)
