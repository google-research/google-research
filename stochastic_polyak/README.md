# Experiments on Stochastic Polyak Step-sizes

This project implements the stochastic Polyak step-size and its variants
as described in the paper

Loizou, Nicolas, et al. "Stochastic polyak step-size for sgd: An adaptive
learning rate for fast convergence." International Conference on Artificial
Intelligence and Statistics. PMLR, 2021. https://arxiv.org/pdf/2002.10542.pdf

Run the experiments with:

$ python -m stochastic_polyak.train_MNIST

to run the experiment on the MNIST dataset. You can replace MNIST with
CIFAR10 to run the experiment with this last dataset.


You can then monitor the experiment by launching tensorboard on that directory:

$ learning/brain/tensorboard/tensorboard.sh --logdir $WORKDIR
