# Spectral method in continuous setting
This directory implements the method of spectral identification in the continuous setting. The conditional density estimator and its variants are from Sugiyama et al (2010). The full estimation procedure is in the directory `./colab`.


## Demo

Run the `continuous_spectral_method.ipynb` notebook in `./colab` folder. \

## Library functions

The least-squares conditional density estimators are implemented in `ls_conditional_de.py` and `multi_ls_conditional_de.py`. The prefix `multi_` stands for multivariate setting.

The least-squares density estimator of the joint distribution and marginal 
distribution are implemented in `ls_de.py` (`multi_ls_de.py`) and `ls_marginal_de.py`
(`multi_ls_marginal_de.py`), respectively.

### Least-squares conditional density estimator
  - Item Function class `CDEBase`: Estimate the conditional density of p(y|x), where both y and x is univariate. We use the object `LSEigenBase` in the `cosde` package to construct the basis function. 
  - Item Function class `MultiCDEBase`: Estimate the conditional density of p(y|x), where y is univariate and x is multivariate. 
