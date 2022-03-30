# Images as Covariates in Estimation of Treatment Effects Analysis (ICETEA)

In this project, we investigated the use of ML models for estimating treatment
effects. Benchmarking causal inference methods is challenging, and after our
initial analysis, we identified a need for a new benchmark with a reasonable
causal story and a clear need for ML models. Thus, we proposed a new
semi-synthetic benchmark based on images and performed experiments using retinal
fundus images. Recent evidence suggests retinal fundus images are
correlated with general vascular conditions, which affects several medical
conditions. Therefore, it is reasonable to consider include retinal
images to make a confounder adjustment to study causal effects.

## Summary

This repo contains:

-   A `README.md` that contains the current status, list of existing files, and
    example on how to run the code.
-   file/directory structure:

    -   `train.py` : Code for training/evaluating the model.
    -   `utils.py' : Code for data simulation, and to run the experiments.
    -   `estimators.py': Implementation of the estimators.
    -   `config.py`: Organize the parameters.
    -   `beam_utils.py': Functions to run beam pipeline.
    -   `ukb.py`: Functions to pre process the UK Biobank Retinal Image data.
    -   `ukb_utils.py`: Auxiliary methods of the ukb.py

-   Datasets:

    -   IHDP: We adopt the same datasets from CEVAE's paper.
        [Source url](https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP/csv)
    -   ACIC: Download datasets
        [link](https://sites.google.com/corp/view/acic2019datachallenge/home)


Note: The ukb.py has dependencies on ukb_model_utils.py. This file is available
at
[Google-Health/genomics-research](https://github.com/Google-Health/genomics-research/tree/main/ml-based-vcdr)
github repo.

Known caveats:

-   When using the AIPW, the logistic regression can sometimes (~10% of runs)
    cause issues if estimates the probabilities as 1 or 0, resulting in very
    poor treatment effect estimates. In this case, the best current solution is
    to run that specific repetition again. A different random initialization
    usually sorts it out.
-   When working with observed clinical features as the outcome, you might need
    to update the range from the tau simulation in simulating_y_from_clinical()
    at ukb.py.

Important Args:

-   `--setting`:
    -   quick: b=1, run all methods, two small samples and two small covariates;
    -   samples: explore several sample sizes with fix covariates;
    -   covariates: explore several covariates sizes with fixed sample size;
    -   synthetic: semi-synthetic datasets, sample size and covariates are
        fixed;
    -   ukb: uk biobank images dataset.
-   `--overwrite`:
    -   If True, the files on path_output will be overwritten.

This is not an officially supported Google product.