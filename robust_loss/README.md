# A General and Adaptive Robust Loss Function

This directory contains reference code for the paper
[A General and Adaptive Robust Loss Function](https://arxiv.org/abs/1701.03077),
Jonathan T. Barron CVPR, 2019

The code is implemented in Tensorflow and the required packages are listed in
`requirements.txt`.

If you'd like this loss, include `general.py` or `adaptive.py` and call the loss
function. `general.py` implements the "general" form of the loss, which assumes
you are prepared to set and tune hyperparameters yourself, and `adaptive.py`
implements the "adaptive" form of the loss, which tries to adapt the
hyperparameters automatically and also includes support for imposing losses in
different image representations. The probability distribution underneath the
adaptive loss is implemented in `distribution.py`. Demo code for training
different variants of a VAE on Celeb-A as was done in the paper is in `vae.py`.
