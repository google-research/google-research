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

"""Tests for membership_inference module."""
from typing import Any, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import random
import jaxopt
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn import model_selection

from learn_to_forget import membership_inference


N_SAMPLES = 500
N_FEATURES = 500
N_CLASSES = 2
N_OBLITERATE = 5


class SimpleMLP(nn.Module):
  # members set by the class parent's __init__ method
  features: Sequence[int]
  dtype: Any

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      x = nn.Dense(feat, name=f'layers_{i}', param_dtype=self.dtype)(x)
      if i != len(self.features) - 1:
        x = nn.swish(x)
    return x


class MembershipInferenceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    data = datasets.make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_FEATURES,
        n_redundant=0,
        n_repeated=0,
        n_classes=N_CLASSES,
        flip_y=0,
        random_state=0)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(*data)
    self.data_train = (x_train, y_train)
    self.data_test = (x_test, y_test)

  def test_evaluate_attack_model(self):
    data_x, data_y = self.data_train
    cv = model_selection.StratifiedShuffleSplit(test_size=0.5)
    idx_train, _ = next(cv.split(data_x, data_y))
    clf = linear_model.LogisticRegression(
        C=1e3, max_iter=2000).fit(data_x[idx_train], data_y[idx_train])
    # check that the classifier has a high train set accuracy
    self.assertGreater(clf.score(data_x[idx_train], data_y[idx_train]), 0.99)

    members = np.zeros_like(data_y)
    members[idx_train] = 1

    # the logistic loss on the single data iterms, minus regularization
    loss_data = clf.predict_log_proba(data_x)[:, data_y]
    # it can be - inf, so we clip the data
    loss_data = np.clip(loss_data, -100, 100)
    cv_scores = membership_inference.evaluate_attack_model(loss_data, members)
    self.assertGreater(np.mean(cv_scores), 0.7)

  # @parameterized.parameters({'use_data_keep', True}, {'use_data_keep', False})
  @parameterized.named_parameters(('with_data_keep', True),
                                  ('without_data_keep', False))
  def test_second_order_defense(self, use_data_keep):
    # create a simple network in FLAX
    model = SimpleMLP(
        features=[N_FEATURES, N_CLASSES], dtype=jnp.float32)

    def sample_loss(params, data):
      """Sample-wise logistic loss, returns an array of size y.size."""
      x, y = data
      logits = model.apply(params, x)
      return jax.vmap(jaxopt.loss.multiclass_logistic_loss)(y, logits)

    # train model on self.data
    key = random.PRNGKey(0)
    dummy_input = random.uniform(key, (N_FEATURES,))
    # instantiate a linear predictive model
    params_init = model.init(key, dummy_input)
    solver = jaxopt.LBFGS(lambda *args: jnp.mean(sample_loss(*args)))
    params, _ = solver.run(params_init, self.data_train)

    # use loss on the test set as loss_target
    loss_target = jnp.mean(sample_loss(params, self.data_test))
    data_obliterate = (self.data_train[0][:N_OBLITERATE],
                       self.data_train[1][:N_OBLITERATE])
    if use_data_keep:
      data_keep = (self.data_train[0][N_OBLITERATE:],
                   self.data_train[1][N_OBLITERATE:])
      obliterated_params = membership_inference.second_order_defense(
          sample_loss, params, data_obliterate, loss_target, data_keep)
    else:
      obliterated_params = membership_inference.second_order_defense(
          sample_loss, params, data_obliterate, loss_target)

    # check that the loss on the obliterated data is smaller than loss_target
    # on the original parameters but higher than it on the obliterated ones.
    self.assertLess(
        jnp.mean(sample_loss(params, data_obliterate)), 0.9 * loss_target)
    self.assertGreater(
        jnp.mean(sample_loss(obliterated_params, data_obliterate)),
        0.9 * loss_target)

    if use_data_keep:
      # check also that the attack doesn't increase much the loss on data_keep
      self.assertLess(jnp.mean(sample_loss(params, data_keep)), loss_target)
      self.assertLess(
          jnp.mean(sample_loss(obliterated_params, data_keep)), loss_target)


if __name__ == '__main__':
  absltest.main()
