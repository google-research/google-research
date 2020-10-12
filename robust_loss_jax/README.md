# A General and Adaptive Robust Loss Function

This directory contains JAX reference code for the paper
[A General and Adaptive Robust Loss Function](https://arxiv.org/abs/1701.03077),
Jonathan T. Barron CVPR, 2019

To use this code, include `general.py` or `distribution.py`.
`general.py` implements the "general" form of the loss, which assumes
you are prepared to set and tune hyperparameters yourself, and `distribution.py`
implements the probability distribution whose negative log-likelihood
corresponds to a shifted "general" loss. This negative log-likelihood used with
free parameters for `alpha` and/or `scale` corresponds to the "adaptive" loss
used in the paper (this code release does not provide a wrapper for this
adaptive loss).

This code repository is shared with all of Google Research, so it's not very
useful for reporting or tracking bugs. If you have any issues using this code,
please do not open an issue, and instead just email jonbarron@gmail.com.

If you use this code, please cite it:
```
@article{BarronCVPR2019,
  Author = {Jonathan T. Barron},
  Title = {A General and Adaptive Robust Loss Function},
  Journal = {CVPR},
  Year = {2019}
}
```
