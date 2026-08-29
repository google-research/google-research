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

Then, for each execution, run:

```console
source _venv/bin/activate
python -m main
```

Finally, view the output under the corresponding (lasest timestamped) folder
under `_experiments`.


TODO(amirhkarimi): add details of how to merge results downloaded from GCP.
TODO(amirhkarimi): complete below
TODO(amirhkarimi): s/col_type/hparam_type
```
s/_EXPLANATION_TYPE/_EXPLAINER
```

For Fig 3:
```console
python main.py --dataset=cifar10 --min_base_model_accuracy=0 --num_samples_per_base_model=100 --num_samples_to_plot_te_for=10 --run_on_precomputed_gcp_data=True
```
