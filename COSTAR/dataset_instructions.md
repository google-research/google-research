Instructions to Add Customized Datasets
==========================================

## Overview

The training script of each method loads data through a realization of the
dataset collection class inherited from one of the `SyntheticDatasetCollection`
or `RealDatasetCollection` in `src/data/dataset_collection.py` based on whether
the dataset to add is synthetic (users have access to the ground truth of
counterfactual treatment outcomes in test time) or real world (users can only
evaluate with factual treatment outcomes in test time).

The two classes mentioned above also lists the necessary preprocessing steps
to take for each method. Users only need to implement the data loading/
generation and the substeps of preprocessing for their own dataset. Next we
will go over the necessary members and functions required in implementation.

## General steps to build a synthetic/real-world dataset

To construct a dataset, the users need to implement the data
generation process first and identify which features are used as covariates,
treatments, and outcomes. Users can follow examples of the Tumor growth
dataset (`src/data/cancer_sim/dataset.py`) or the MIMIC-III Synthetic dataset
(`src/data/mimic_iii/semi_synthetic_dataset.py`).
Real-world dataset example is the M5 dataset (`src/data/m5/real_datasets.py`).

### Implement the dataset collection class

The first step is to implement the dataset collection class of the customized
dataset by inheriting the `SyntheticDatasetCollection` class
(for real-world datasets inherit the `RealDatasetCollection` class).
The dataset
collection class needs to construct the following member variables in
initialization:

- `train_f`: the training subset with factual treatment outcomes only;
- `val_f`: the validation subset with factual treatment outcomes only;
- `test_cf_one_step` (only for synthetic datasets): the test subset with 1-step counterfactual treatment
outcomes only;
- `test_cf_treatment_seq` (only for synthetic datasets): the test subset with multi-step counterfactual
treatment outcomes only;
- `test_f` (only for real-world datasets): the test subset with factual treatment outcomes;
- `seed`: random seed used in dataset construction;
- `projection_horizon`: range of tau-step-ahead prediction (tau =
projection_horizon + 1);
- `autoregressive`: include the outcome in the previous step in covariates;
- `has_vitals`: set to True if the dataset has features other than previous
outcomes in covariates;
- `train_scaling_params`: the parameters required for normalizing inputs,
ususally calculated from the training subset;
- `max_seq_length`: the maximum length of sequences in constructed datasets.

Users also need to specify the list of necessary arguments of the initialization
function in the configuration file. Ususally it can be placed under
`config/dataset`.

### Implement the dataset class

`train_f`, `val_f`, `test_cf_one_step` and `test_cf_treatment_seq` should
inherit the `torch.utils.data.Dataset` class. Users can reuse the implementation
of commonly required `__getitem__` and `__len__` methods in examples. In addition to that, the users also need to
implement the following methods of the dataset class:

`get_scaling_params(self)`: calculate the parameters used for normalizing inputs
. Ususaly it contains the mean and variance of each input features.

`process_data(self, scaling_params)`: convert the generated/loaded raw data into
the formatted data dictionary, including all necesasry preprocessing and
normalization of features. Denote the generated/loaded data has a sample size
$N$ and a unified sequence length $T+1$ (after padding if necessary), the raw
data should align the observed covariates at time $t$ with the treatment and its
corresponding outcome right after time $t$.

The method should reformat the raw data and return a dictionary with at least
the following entries:

- `prev_treatments`: treatment variables sliced between time $[0, T - 1]$.
Shape: $[N, T, \text{treatment\_feature\_num}]$.
- `current_treatments`: treatment variables between time $[1, T]$.
Shape: $[N, T, \text{treatment\_feature\_num}]$.
- `prev_outputs`: output variables sliced between time $[0, T - 1]$.
Shape: $[N, T, \text{output\_feature\_num}]$.
- `outputs`: output variables between time $[1, T]$.
Shape: $[N, T, \text{output\_feature\_num}]$.
- `vitals`: covariate variables excluing previous outcomes between time
$[0, T - 1]$. Shape: $[N, T, \text{vital\_feature\_num}]$.
- `static_features`: features describing subject properties that do not
vary with time. Shape: $[N, \text{static\_feature\_num}]$.
- `sequence_lengths`: the actual lengths of each sequence before padding.
Shape: $[N]$.
- `active_entries`: binary masks of valid data, 1 for valid timesteps and 0 for
padded steps. Shape: $[N, T, 1]$

### Method-specific pre-processing steps

#### MSM/RMSN/CRN

`process_sequential(self, encoder_r, projection_horizon)`:
Process the data from `process_data` for multi-step-ahead prediction by
exploding the dataset to a larger one with rolling origin and add the
embeddings from the encoder. For the $i$-th sequence from `processed_data`,
we slice a subsequence of length `projection_horizon` starting from each
timestep between 1 and `sequence_lengths[i] - projection_horizon - 1` for each
of the members `prev_treatments`, `current_treatments`, `prev_outputs`,
`outputs`, `active_entries`. Notice that we do not include `vitals` since future
covariates are not accessible in multi-step-ahead predictions. During the
explosion process, we re-calculate the `sequence_lengths` of each exploded
subsequence and copy the `static_features` of the original whole sequence.

In addtion to re-organizing the processed data, we also add some extra entries:

- `init_state`: the embeddings of history observed until the beginning time of
each subsequence given by encoder.
- `original_index`: the index of the whole sequence that each subsequence
is exploded from.
- `active_encoder_r`: similar to `active_entries`, but indicating which
history steps are used to embed the `init_state`.
- `unscaled_outputs`: exploded `outputs` after de-normalization.

After the explosion, each sequence expand from shape $[T, \text{feature\_num}]$
to $[M, T, \text{feature\_num}]$. We concatenate the exploded data of all
sequences and reuse the key name to construct entries in the returned dictionary
.

`process_sequential_test(self, projection_horizon, encoder_r=None)`:
Similar to `process_sequential` but only applies to the `test_cf_one_step` and
`test_cf_treatment_seq`
subset. The main difference is that the unrolling has been done in the
generation of counterfactual mult-step treatment outcome data. Therefore, we
only need to slice the last `projection_horizon` steps of each sequence and skip
the unrolling step.

`process_autoregressive_test(encoder_r, encoder_outputs, projection_horizon)`:
Only applies to the `test_cf_treatment_seq` subset. It is similar to
`process_sequential_test` but additionaly set `prev_outputs` as the placeholder
of the prediciton result from last step to achieve autoregressive prediction.

`explode_trajectories(self, projection_horizon)`:
It works in a similar way to `process_sequential`, but it rolls the end instead of the origin within the time range $t\geq\text{projection\_horizon}$.

#### CT

`process_sequential_test(self, projection_horizon, encoder_r=None)`:
Same as the `process_sequential_test` function above.

`process_sequential_multi(self, projection_horizon)`:
Process the data from `process_data` for multi-step-ahead prediction. Here we
only need to use a key named `future_past_split` - an integer for each
sequence to specify the last timestep
index of the history part thanks to the design of CausalTransformer.

#### COST

In our COST implementation we integrate all the manipulations of data
re-organization after the `process_data` step in our training and evaluation
codes directly, thus we do not need COST-specific preprocessing steps here.

