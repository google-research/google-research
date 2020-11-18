# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Model training utilities."""


class EarlyStopper(object):
  """Early stopping logic for model training.

  This class provides a stateful predicate function that implements early
  stopping logic to be used in conjunction with
  tf.Experiment.continuous_train_and_eval.
  """

  def __init__(
      self,
      num_evals_to_wait,
      metric_key='precision',
      epsilon=1e-3):

    self._num_evals_to_wait = num_evals_to_wait
    self._evals = []
    self._epsilon = epsilon
    self._metric_key = metric_key
    self._num_evals_since_best = 0
    self._best_so_far = None

  def early_stop_predicate_fn(self, tf_eval_results):
    """Checks the evaluation results and decides if training should continue.

    Args:
      tf_eval_results: (dict or None) A dict containing items defined by a
        model's tf.estimator.EstimatorSpec.eval_metric_ops configuration; the
        metric key selected for early stopping monitoring must be provided by
        the model's evaluation results as defined by the corresponding
        tf.estimator.EstimatorSpec. Note the first call to the predicate from
        the tf.Experiment framework will pass None instead of a populated eval
        results dict (by contract).
    Returns:
      A boolean to indicate if training should continue.
    """
    if tf_eval_results is None:
      return True

    curr = tf_eval_results[self._metric_key]
    self._evals.append(curr)

    if (self._best_so_far is None
        or ((self._best_so_far + self._epsilon) < curr)):
      self._best_so_far = curr
      self._num_evals_since_best = 0
    else:
      self._num_evals_since_best += 1
      if self._num_evals_since_best >= self._num_evals_to_wait:
        return False  # i.e., stop training

    return True  # i.e., keep going
