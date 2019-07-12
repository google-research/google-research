# Code to explain increases in risk predictions

This directory contains code to train a model that generates a sequence of risk predictions and explain increases.
The explanations are expressed as weights over the inputs.

Example usage:

```python
from explaining_risk_increase import input_fn
from explaining_risk_increase import observation_sequence_model as osm

train_steps = 20000
eval_steps = 100

# tf.SequenceExample in TFRecord format. See test_data for examples and
# https://github.com/google/fhir on how to generate those.
train_data = ...
eval_data = ...
train_input_fn = input_fn.get_input_fn(
    tf.estimator.ModeKeys.TRAIN,
        train_data,
        'label.in_hospital_death',
        sequence_features=[
            'Observation.code', 'Observation.value.quantity.value',
            'Observation.value.quantity.unit',
            'Observation.code.harmonized:valueset-observation-name'
        ],
        dense_sequence_feature='Observation.value.quantity.value',
        required_sequence_feature='Observation.code.harmonized:valueset-'
        'observation-name',
        batch_size=64,
        shuffle=True)
eval_input_fn = input_fn.get_input_fn(
    tf.estimator.ModeKeys.EVAL,
        eval_data,
        'label.in_hospital_death',
        sequence_features=[
            'Observation.code', 'Observation.value.quantity.value',
            'Observation.value.quantity.unit',
            'Observation.code.harmonized:valueset-observation-name'
        ],
        dense_sequence_feature='Observation.value.quantity.value',
        required_sequence_feature='Observation.code.harmonized:valueset-'
        'observation-name',
        batch_size=128,
        shuffle=False)
model = osm.ObservationSequenceModel()
hparams = model.create_model_hparams()

estimator = tf.estimator.Estimator(
    model_fn=model.create_model_fn(hparams))

experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=train_steps,
        eval_steps=eval_steps,
        continuous_eval_throttle_secs=10)

experiment.train_and_evaluate()
```
