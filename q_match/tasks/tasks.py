# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All the tasks for the test dataset.
"""

import functools

from absl import logging
from flax.core import freeze
import jax
import numpy as np
from sklearn.linear_model import LogisticRegression


from q_match.configs.enums import EvalTask


def eval_test_dataset(eval_model, finetune_params, finetune_state,
                      linear_params, linear_state, test_ds,
                      linear_over_features=False, eval_task=EvalTask.BOTH):
  """Evalulate tasks for the test dataset.

  Currently, just return the test accuracy.

  Args:
    eval_model: FLAX model
    finetune_params: model params after finetune training
    finetune_state: model state after finetune training (for batch norm)
    linear_params: model params after linear head training
    linear_state: model state after linear head training (for batch norm)
    test_ds: the test dataset
    linear_over_features: Whether to use the features instead of the pretext
      encoding for the linear head.
    eval_task: The evaluation tasks to run.

  Returns:
    Results accuracy of the test dataset.
  """

  @functools.partial(jax.jit, static_argnums=(4, 5,))
  def compute_num_correct(params, state, features, targets, linear_head=False,
                          linear_over_features=False):
    variables = freeze({'params': params, **state})
    outputs = eval_model.apply(variables, features,
                               linear_over_features=linear_over_features)

    if linear_head:
      preds = outputs['main']['linear_head']['logits']
    else:
      preds = outputs['main']['finetune_head']['logits']

    return jax.numpy.sum(jax.numpy.argmax(preds, axis=-1) == targets)

  def eval_test_ds_accuracy(params, state, test_ds, linear_head=False,
                            linear_over_features=False):
    correct = 0.
    num_seen = 0.
    for example in test_ds:
      features = jax.numpy.array(example['features'])
      targets = jax.numpy.array(example['target'])

      current_correct = float(
          compute_num_correct(
              params,
              state,
              features,
              targets,
              linear_head=linear_head,
              linear_over_features=linear_over_features))
      current_size = float(targets.shape[0])

      num_seen += current_size
      correct += current_correct
    return correct/num_seen

  if eval_task == EvalTask.BOTH:
    return dict(
        linear_accuracy=eval_test_ds_accuracy(
            linear_params, linear_state, test_ds, linear_head=True,
            linear_over_features=linear_over_features),
        finetune_accuracy=eval_test_ds_accuracy(
            finetune_params, finetune_state, test_ds, linear_head=False))
  elif eval_task == EvalTask.LC:
    return dict(
        linear_accuracy=eval_test_ds_accuracy(
            linear_params, linear_state, test_ds, linear_head=True,
            linear_over_features=linear_over_features)
        )
  else:
    return dict(
        finetune_accuracy=eval_test_ds_accuracy(
            finetune_params, finetune_state, test_ds, linear_head=False
        )
    )


def pinv_eval(eval_model, pretext_params, pretext_state, train_ds, test_ds):
  """Uses psuedo-inverse to get the LC results."""
  variables = freeze({'params': pretext_params, **pretext_state})

  # Train dataset
  features = []
  pretext_features = []
  targets = []
  for z in train_ds.as_numpy_iterator():
    features.append(z['features'])
    targets.append(z['target'])
    outputs = eval_model.apply(variables, z['features'])
    pretext_features.append(outputs['pretext']['encoded'])
  features = np.vstack(features)
  targets = np.concatenate(targets)
  pretext_features = np.vstack(pretext_features)

  lr = LogisticRegression(penalty='l2', fit_intercept=False,
                          solver='lbfgs',
                          multi_class='multinomial',
                          max_iter=100,
                          class_weight='balanced')
  lr.fit(features, targets)

  lr_pretext = LogisticRegression(
      penalty='l2',
      fit_intercept=False,
      solver='lbfgs',
      multi_class='multinomial',
      max_iter=100,
      class_weight='balanced')
  lr_pretext.fit(pretext_features, targets)

  # Test dataset
  features = []
  targets = []
  pretext_features = []
  for z in test_ds.as_numpy_iterator():
    features.append(z['features'])
    targets.append(z['target'])
    outputs = eval_model.apply(variables, z['features'])
    pretext_features.append(outputs['pretext']['encoded'])
  features = np.vstack(features)
  targets = np.concatenate(targets)
  pretext_features = np.vstack(pretext_features)

  results = dict(
      linear_accuracy_pretext=lr_pretext.score(pretext_features, targets),
      linear_accuracy_features=lr.score(features, targets))
  logging.info(results)
  return results
