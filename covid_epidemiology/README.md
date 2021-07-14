# Covid Epidemiology

This repository corresponds to the
paper [A prospective evaluation of AI-augmented epidemiology to forecast COVID-19 in the USA and Japan](https://www.researchsquare.com/article/rs-312419/v1)
.

### Contacts

For question, please reach out to soarik@google.com or joelshor@google.com.

# Overview

* The tf_seir.py file contains the main functions to define, train and evaluate
  the proposed model, which integrates learnable encoders into compartmental
  (SEIR-extended) models. We have separate models for US country, state and
  country level models, as well as Japan prefecture-level model. Each of these
  have specific functions and config files, with names
  'generic_seir_XXX_constructor.py' and 'generic_seir_specs_XXX.py'. The
  encoder-related functions are in the 'encoders' directory.

* To see how evaluation metrics were calculation, please see [evaluation_metrics.ipynb](https://github.com/google-research/google-research/blob/master/covid_epidemiology/evaluation_metrics.ipynb).

# Compartmental Modeling

The compartmental model's configuration, rates, and dynamics are defined via the
`ModelDefinition`. A new custom model definition should inherit from
`models.definitions.base_model_definition.BaseModelDefinition` and extra helper
functions for common tasks like pre-processing features can be used by
inheriting from the
`models.definitions.base_model_definition.BaseCovidModelDefinition` instead.
Further code pertaining to the compartmental dynamics can also be shared amoung
multiple models by inheriting from the classes defined in
`models.definitions.compartmental_model_definitions`.

## Model Definition Components

There are four general (but interconnected) areas of the model definition which
are:

1. Selecting or creating the model's features.
2. Defining the model's rates.
3. Constructing the model's states and dynamics.
4. Pairing model outputs with ground truth data.

While methods described in the sections below represent most of the main
differences in the model there are additional helper methods which may need
custom implementations as well and are documented more fully in the code.

### 1. Selecting or creating the model's features

All of the data used to create the model is stored in BigQuery data tables and
the model definition is used to define which of these features are used as
static or time-varying covariates, or the ground truth values for each of the
compartments. To use features available in the BigQuery data tables the feature
alias (name used to identify the feature in the code) and the feature_name (used
to identify the feature in the BigQuery table) should be returned
by `get_static_features`
and `get_ts_features` for static and time series data respectively. The
dictionaries returned by these functions should include all data needed from the
BigQuery tables including the ground truth data and covariates.

Prior to being used by the model the BigQuery data is converted into the format
used for the model using the `extract_all_features` method which allows for
engineered features to be created for the model. This may be done either by
modifying the input static and time series DataFrames directly or by
constructing one feature from another (many models do this in the
`transform_{static|ts}_features` method). If the features are added to the input
DataFrames directly they must also be returned by the corresponding
`get_{static|ts}_features` method.

The identification the ground truth data based on the incoming features is
performed in the `initialize_ground_truth_timeseries` method. During this
identification the method should also format the ground truth data into
TensorFlow tensors so that it can be easily used for teacher forcing and
computing the model's loss. The last function of this ground truth
identification process is to determine when the infection should be considered
active in each location.

### 2. Defining the model's rates

`FeatureSpec`s should be created for each of the features that will be used as a
covariate in the model using the feature alias for the name. Which features are
used by each encoder are defined using `EncderSpec`s. The `ModelSpec` is used to
define which encoders are used and what the model's hyperparameters are.

There should be an encoder for each of the model's rates (variables). A list of
the model's rates must be defined in the `_ENCODER_RATE_LIST` which must match
the names of the encoders listed in the model's encoder_specs and there must
also be a hyperparameter for each of the rates with the name `{rate_name}_init`
to initialize the rate for the model.

To ensure that the model's rates are realistic, prior knowledge should be used
to limit them between their expected bounds. This is done in
the `bound_variables`
method which maps the rates output by the encoders into the model's operating
space using functions that constrain the output (e.g. sigmoid functions).

### 3. Constructing the model's states and dynamics

The model's current state is contained in a single tensor and the ordering of
that tensor is defined in the `initialize_seir_state` method. The changes in the
model's state at each timestep is calculated using the `seir_dynamics` method of
the model constructor (
i.e. `state[t + 1] = state[t] + seir_dynamics(state[t], variables[t])`).

### 4. Pairing model outputs with ground truth data

The pairing of the ground truth data with the model's predicted states is
important for both the model training and the evaluation. During training the
`sync_values` method is used to implement teacher forcing where the model's
predictions are combined with the ground truth values at each timestep using a
weighted average to help the model's training and avoid error accumulation. The
model's state estimates must also be paired with the ground truth data during
model training in the model constructor method `compute_losses`, which
calculates the model's loss function for the training and validation data sets.
The method `generate_compartment_predictions` pairs states with their
corresponding original ground truth values so that accuracy and error metrics
can be calculated.
