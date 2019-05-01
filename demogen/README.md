# DEep MOdel GENeralization dataset (DEMOGEN)

This codebase contains code necessary for using the generalization dataset used
in "Predicting the Generalization Gap in Deep Networks with Margin
Distributions" (ICLR 2019) https://arxiv.org/abs/1810.00113

Link to dataset will be in the paper once available. A typical use case can be
found shown in `example.py`. Run this by `python -m demogen.example`.

Examples of computing the margin and total variation on the dataset can be found
in the docstring of `margin_utils.py` and `total_variation_util.py`.

Data (15.57GB) for this code based can be downloaded at:

https://storage.googleapis.com/margin_dist_public_files/demogen_models.tar.gz
