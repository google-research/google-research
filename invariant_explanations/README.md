## Project description

The primary purpose of this repository is to train a series of meta-models that
can predict the predictions of underlying base-models trained on different
hparam settings. Essentially, the aim of the meta-model is to emulate the
post-training predictions of an entire class of base-models, without needing to
fully train the base-models. Therefore, the covariates used in the training of
the meta-model are a combination of either X, H, i.e., sample images & hparams,
or X, W_@_epoch, i.e., samples and weights of the base-model @ epoch < 86. The
final epoch in this CNN zoo is set to 86. The targets of the meta-model are
always set to be the predictions of the meta model at epoch 86.

Besides training the meta-model on different covariate combinations, we also
iterate over different splits of instances in the train/test sets. The total
number of instances in the meta-model training set is the product of the number
of base models and the number of samples (images) per base model. From this
product, a train_fraction fraction of them are chosen to comprise the train set
and the remainder are used for evaluation.

config.py keeps track of all setups on which the meta-model is trained. After
the training of each setup, the train and test accuracy are saved to file to be
processed and displayed later in aggregate.

The codebase is accompanied with sample weights and metrics data extracted from
the predict_dnn_accuracy repository (link below), and can run out of the box.
To run on the actual weights/metrics files, download the data form the link
below and set the `RUN_ON_TEST_DATA` variable to False.

https://github.com/google-research/google-research/tree/master/dnn_predict_accuracy


## Execution

Single command:
`./run.sh`

Alternative, run the following once:

```console
python3 -m venv _venv
source _venv/bin/activate
pip install -r requirements.txt
mkdir -p _experiments
```

Check that the test data and code are correct by running (all test should pass):
```console
python utils_test.py
```

Then, before each execution, run:

```console
source _venv/bin/activate
python -m main
```

Finally, view the output under the corresponding (lasest timestamped) folder
under `_experiments`.


## To generate data and figures, call main.py as below, 3 times:
```console
#!/bin/sh

for DATASET in 'cifar10' 'svhn_cropped' 'mnist' 'fashion_mnist'
do
  for MEDIATION_TYPE in 'mediated' 'unmediated'
  do
    python main.py \
      --dataset=$DATASET \
      --mediation_type=$MEDIATION_TYPE
  done
done
```
1st, call with process_and_resave_cnn_zoo_data(). Then copy the files
from the experimental folder into the corresponding dir under MERGED_DATA_PATH.

2nd, call with measure_prediction_explanation_variance(). Then copy the files
from the experimental folder into the corresponding dir under MERGED_ITES_PATH.

If steps 1 and 2 are run separately, add the following argument on the 2nd call:
`--run_on_precomputed_gcp_data=True`

3rd, call with plot_paper_figures().


## TODO(amirhkarimi):
[ ] add details of how to merge results downloaded from GCP.
[ ] complete below
[ ] s/col_type/hparam_type
[ ] `s/_EXPLANATION_TYPE/_EXPLAINER`
[ ] check error bars for corr plots (@ 10, 30, 100 samples per model)