# Model training

Each of the 3 different architectures evaluated in the manuscript are included
within this directory:
* RNN: rnn.py
* CNN: cnn.py
* LR: lr.py

And `train.py` corresponds to the model training harness used in all cases.

Models were implemented using the TensorFlow 1.3 API and use the tf.Estimator
framework of APIs specifically. Each model is implemented as a TensorFlow
Estimator `model_fn` and consumes tf.Datasets for training or inference.

The batch size for a single step was 25 examples in every case during training.


## Training model ensembles

Note that invocations of train.py were used to train a single model replica. For
an ensemble as used within the manuscript, N of these model replicas are trained
separately with distinct randomized weight initializations.


## Input (sequence) encoding

The fixed-length encoding was used for both the CNN and LR models while the
variable length encoding was used for the RNN.

Encoding of sequences to model input features occurs in `util.dataset_utils`:
see `encode_varlen` and `encode_fixedlen`. Individual residues were encoded as
one hot vectors as implemented within `util.residue_encoding`.


### Fixed-length sequence encoding used

The fixed encoding length used provides 1 slot for each wildtype position, plus
a prefix position; all of these positions can also have a single insertion,
making the max number of residues representable: `58 = 2 * (28 + 1)`, given the
28 WT AAV2 residue positions: `DEEEIRTTNPVATEQYGSVSTNLQRGNR`

## Model regularization

Models were regularized via early stopping using the implementation provided
within `train_utils.EarlyStopper`. Model training progression was monitored
using a hold-out validation set of sequences that was the same for every
architecture. Early stopping halted training after the model's precision on the
validation set did not increase for 10 evaluation periods. An evaluation periods
occurs every 500 steps in our setup.
