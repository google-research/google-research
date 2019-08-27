# Model-Agnostic Meta-Learning

This repo contains code accompaning the paper, 	[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (Finn et al., ICML 2017)](https://arxiv.org/abs/1703.03400). It includes code for running the few-shot supervised learning domain experiments, including sinusoid regression, Omniglot classification, and MiniImagenet classification.

For the experiments in the RL domain, see [this codebase](https://github.com/cbfinn/maml_rl).

### Dependencies
This code requires the following:
* python 2.\* or python 3.\*
* TensorFlow v1.0+

### Data
For the Omniglot and MiniImagenet data, see the usage instructions in `data/omniglot_resized/resize_images.py` and `data/miniImagenet/proc_images.py` respectively.

### Usage
To run the code, see the usage instructions at the top of `main.py`.

### Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/cbfinn/maml/issues).
