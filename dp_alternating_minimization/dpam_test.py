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

"""Tests for DPAM."""

import copy

from absl.testing import parameterized
import dp_accounting

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras_sparse

from dp_alternating_minimization import dpam


def _generate_dataset():
  num_items = 50
  num_users = 500
  def make_data(sparsity):
    user_sizes = np.maximum(np.random.randint(
        0, int(np.ceil(num_items*sparsity*2)), size=num_users), 1)
    uid = np.concatenate(
        [[i]*size for i, size in enumerate(user_sizes)]).astype(np.int32)
    sid = np.concatenate(
        [np.random.choice(num_items, size=size, replace=False)
         for size in user_sizes])
    rating = np.ones_like(uid, dtype=np.float32)
    return pd.DataFrame({"uid": uid, "sid": sid, "rating": rating})
  train_df = make_data(sparsity=0.3)
  validation_df = make_data(sparsity=0.1)
  test_df = make_data(sparsity=0.1)
  dataset = dpam.Dataset(
      num_users, num_items, metadata_df=None,
      train=dpam.SparseMatrix(train_df),
      validation=dpam.SparseMatrix(validation_df, train_df),
      test=dpam.SparseMatrix(test_df, train_df))
  return dataset


class _Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, reg):
    super().__init__()
    self.embeddings = tf.keras.layers.Embedding(
        vocab_size, embedding_dim,
        embeddings_initializer=tf.keras.initializers.random_normal(stddev=0.01),
        activity_regularizer=None)
    self.reg = reg

  def call(self, features):
    return self.embeddings(features["sid"])

  def loss_vector(self, predictions, labels, weights, features):
    """Returns a loss vector."""
    col_emb = self.trainable_variables[0]
    col_norms = tf.reduce_sum(tf.square(col_emb), axis=1)/2
    loss_vector = weights*(
        tf.pow(labels - predictions, 2)/2
        + self.reg * tf.gather(col_norms, features["sid"])
        )
    return loss_vector


class DPAMTest(tf.test.TestCase, parameterized.TestCase):

  def _assert_metrics(self, model):
    """Runs training and check the keys in metrics."""
    # Only designed to check that training runs. Does not check correctness of
    # the solution.
    metrics = model.metrics
    expected_keys = [
        "valid_RMSE", "test_RMSE", "valid_RMSE_user", "test_RMSE_user",
        "valid_R@20", "test_R@20",
    ]
    self.assertContainsSubset(expected_keys, metrics.keys())

  # Non-Private training =======================================================
  def test_non_private_training_wals(self):
    ds = _generate_dataset()
    model = dpam.DPALSModel(
        ds, embedding_dim=8, unobserved_weight=0.1, init_stddev=0.1,
        regularization_weight=10, recall_positions=[20])
    model.train(num_steps=3, compute_metrics=True)
    self._assert_metrics(model)

  def test_non_private_training_wals_with_frequency_reg(self):
    """Tests frequency-based regularization."""
    ds = _generate_dataset()
    model = dpam.DPALSModel(
        ds, embedding_dim=8, unobserved_weight=0, init_stddev=0.1,
        regularization_weight=10, row_reg_exponent=1, col_reg_exponent=1,
        recall_positions=[20])
    model.train(num_steps=3, compute_metrics=True)
    self._assert_metrics(model)

  def test_non_private_training_am(self):
    """Tests Alternating Minimization (AM) with the keras SGD optimizer."""
    ds = _generate_dataset()
    model = dpam.AMModel(
        ds, embedding_dim=8, unobserved_weight=0, init_stddev=0.1,
        regularization_weight=10, recall_positions=[20])
    model.compile_item_tower(
        item_encoder=_Encoder(ds.num_items, 8, reg=10),
        item_batch_size=100,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01))
    model.train(num_steps=3, inner_steps=2, compute_metrics=True)
    self._assert_metrics(model)

  # Private training ===========================================================
  @parameterized.parameters(
      ("uniform",),
      ("tail",),
      ("adaptive_weights"),
  )
  def test_private_training_dpals(self, method):
    """Tests private training."""
    ds = _generate_dataset()
    weight_exponent = None
    if method == "adaptive_weights":
      weight_exponent = -0.5
    sanitizer = dpam.Sanitizer(
        budget=100, method=method, item_frac=0.9,
        center=True, count_stddev=100, count_sample=100,
        weight_exponent=weight_exponent)
    optimizer = dpam.DPALSOptimizer(budget=100, max_norm=1)
    accountant = dpam.DPALSAccountant(sanitizer, optimizer, steps=3)
    accountant.set_sigmas(target_epsilon=10, target_delta=1e-3)
    model = dpam.DPALSModel(
        ds, embedding_dim=8, unobserved_weight=0, init_stddev=0.1,
        regularization_weight=10, recall_positions=[20],
        sanitizer=sanitizer, optimizer=optimizer)
    model.train(num_steps=3, compute_metrics=True)
    self._assert_metrics(model)

  @parameterized.parameters(
      (1.0, 1.0),
      (1.0, 0.5),
      (0.1, 0.2))
  def test_sensitivity_dpals(self, budget, max_norm):
    """Verifies the L2 sensitivity of the LHS and RHS of the normal equation."""
    ds = _generate_dataset()
    dim = 8
    uid = 10
    # Remove one user
    def remove_user_data(ds):
      ds = copy.deepcopy(ds)
      def remove_fn(df):
        return df[df["uid"] != uid]
      ds._update_all(remove_fn)
      return ds
    ds_1 = copy.deepcopy(ds)
    ds_2 = remove_user_data(ds)
    # list of items rated by this user.
    train_data = ds_1.train.data
    user_items = set(train_data[train_data["uid"] == uid]["sid"].values)
    init_row_emb = np.ones([ds.num_users, dim])
    def normalize(xs):
      return max_norm*xs/np.linalg.norm(xs, axis=1, keepdims=True)
    def get_model_stats(ds):
      # Use a weight exponent of 0, so small changes in counts due to removing
      # one user do not affect the other items' LHS/RHS.
      sanitizer = dpam.Sanitizer(
          budget=budget, method="adaptive_weights", item_frac=1.0,
          center=False, count_stddev=0, count_sample=0, weight_exponent=0.0)
      optimizer = dpam.DPALSOptimizer(budget=budget, max_norm=max_norm)
      model = dpam.DPALSModel(
          ds, binarize=True, embedding_dim=dim, unobserved_weight=0,
          init_stddev=0.1, recall_positions=[20], optimizer=optimizer,
          sanitizer=sanitizer)
      model.row_embeddings.assign(normalize(init_row_emb))
      # Compute the sufficient statistics
      lhss = []
      rhss = []
      for batch in model._col_data:
        lhs, rhs = dpam._get_lhs_rhs(batch, model.row_embeddings)
        lhss.append(lhs)
        rhss.append(rhs)
      lhs_norms = tf.linalg.norm(tf.concat(lhss, axis=0), axis=[1, 2])
      rhs_norms = tf.linalg.norm(tf.concat(rhss, axis=0), axis=[1, 2])
      return lhs_norms, rhs_norms
    m1_lhs_norm, m1_rhs_norm = get_model_stats(ds_1)
    m2_lhs_norm, m2_rhs_norm = get_model_stats(ds_2)
    lhs_diff = m1_lhs_norm - m2_lhs_norm
    rhs_diff = m1_rhs_norm - m2_rhs_norm
    # Check that items this user didn't contribute to are equal.
    item_mask = np.array([i in user_items for i in range(ds.num_items)])
    self.assertAllLess(np.abs(lhs_diff[~item_mask]), 1e-4)
    self.assertAllLess(np.abs(rhs_diff[~item_mask]), 1e-4)
    # Check the total budget (note that budgets are normalized by the max norm)
    lhs_sum = sum(lhs_diff*lhs_diff)/max_norm**4
    rhs_sum = sum(rhs_diff*rhs_diff)/max_norm**2
    self.assertAllClose(lhs_sum, budget, rtol=1e-4)
    self.assertAllClose(rhs_sum, budget, rtol=1e-4)

  def test_private_training_dpam(self):
    """Tests Alternating Minimization (AM) with the keras DPSGD optimizer."""
    ds = _generate_dataset()
    steps = 2
    inner_steps = 2
    num_examples_per_user = 2
    num_users_per_batch = 10
    sanitizer = dpam.Sanitizer(
        budget=num_examples_per_user, method="tail", center=True,
        item_frac=1.0, count_stddev=20, count_sample=num_examples_per_user)
    accountant = dpam.DPAMAccountant(
        sanitizer,
        steps=steps*inner_steps,
        noise_multiplier=0,
        num_users_per_batch=num_users_per_batch,
        gradient_accumulation_steps=1,
        num_users=ds.num_users)
    accountant.set_noise_multiplier(
        target_epsilon=10, target_delta=1/ds.num_users)
    optimizer = dp_optimizer_keras_sparse.DPSparseKerasSGDOptimizer(
        l2_norm_clip=0.1,
        noise_multiplier=accountant.noise_multiplier,
        num_microbatches=num_users_per_batch,
        gradient_accumulation_steps=1,
        learning_rate=1.0)
    model = dpam.AMModel(
        ds, embedding_dim=8, unobserved_weight=0, init_stddev=0.1,
        regularization_weight=10, recall_positions=[20], sanitizer=sanitizer)
    model.compile_item_tower(
        item_encoder=_Encoder(ds.num_items, 8, reg=10),
        num_users_per_batch=num_users_per_batch,
        num_examples_per_user=num_examples_per_user,
        optimizer=optimizer,
    )
    model.train(num_steps=steps, inner_steps=inner_steps, compute_metrics=True)
    self._assert_metrics(model)


class DPAMUtilitiesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (0.1, 0.0),
      (0.1, -0.5),
      (0.1, -1.0),
      (10, 0.0),
      (10, -0.5),
      (10, -1.0))
  def test_adaptive_weights(self, budget, exponent):
    # 5 users and 3 items.
    col_data = tf.SparseTensor(
        indices=np.array([
            [0, 1], [0, 2], [0, 4],
            [1, 0],
            [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
        ]),
        values=tf.ones([9]), dense_shape=[3, 5])
    counts = [3, 1, 5]
    ws = dpam._compute_adaptive_weights(col_data, counts, exponent, budget)
    sp_weights = col_data.with_values(ws)
    dense_ws = tf.sparse.to_dense(sp_weights, default_value=np.nan)
    # Verify that for each user, the squared weights sum to the budget
    self.assertAllClose(
        np.nansum(dense_ws*dense_ws, axis=0), budget*np.ones(5),
        rtol=1e-5)

  def test_project_psd(self):
    q1 = [[-0.65542968, -0.7552562],
          [-0.7552562, 0.65542968]]
    q2 = [[-0.12622619, 0.99200149],
          [0.99200149, 0.12622619]]
    def basis_change(a, q):
      # returns (q*a*q.T)
      return np.dot(np.dot(q, a), np.transpose(q))
    # Matrices and their expected projections.
    x0 = [[2.0, 1.0],
          [-1.0, -1.0]]
    x0_proj_expected = [[2.0, 0.0],
                        [0.0, 0.0]]
    x1 = np.array([
        basis_change([[-1.0, 0.0], [0.0, 1.0]], q1),
        basis_change([[3.0, 10.0], [-10.0, -1.0]], q2),
    ])
    x1_proj_expected = np.array([
        basis_change([[0.0, 0.0], [0.0, 1.0]], q1),
        basis_change([[3.0, 0.0], [0.0, 0.0]], q2),
    ])

    x0_proj = dpam._project_psd(x0, rank=2)
    self.assertAllClose(x0_proj, x0_proj_expected, atol=1e-4)
    x1_proj = dpam._project_psd(x1, rank=3)
    self.assertAllClose(x1_proj, x1_proj_expected, atol=1e-4)


class PrivacyAccountingTest(parameterized.TestCase):

  @parameterized.parameters(
      (True,),
      (False,),
  )
  def test_sanitizer(self, center):
    orders = [0.001 * i for i in range(1000, 100000)]
    delta = 1e-6
    noise = 4.530877117
    count_sample = 1
    if center:
      sensitivity = np.sqrt(2 + 2 * count_sample)
    else:
      # We don't compute the noisy mean.
      sensitivity = np.sqrt(2 * count_sample)
    # Reverse compute count_stddev and count_sample to fit the above noise.
    # noise = count_stddev / sensitivity
    count_stddev = noise * sensitivity
    sanitizer = dpam.Sanitizer(center=center,
                               budget=count_sample,
                               item_frac=0.5,
                               count_sample=count_sample,
                               count_stddev=count_stddev)
    accountant = dp_accounting.rdp.RdpAccountant(orders)
    accountant.compose(sanitizer.get_dp_event_noisy_count())
    eps = accountant.get_epsilon(delta)
    self.assertAlmostEqual(eps, 1)

  @parameterized.parameters(
      # epsilon should be infinity when either budget or count_sample is not set
      (0, 123),
      (123, 0),
      (0, 0),
  )
  def test_sanitizer_nonprivate(self, budget, count_sample):
    orders = [0.001 * i for i in range(1000, 100000)]
    sanitizer = dpam.Sanitizer(budget=budget,
                               count_sample=count_sample,
                               count_stddev=1.0)
    accountant = dp_accounting.rdp.RdpAccountant(orders)
    accountant.compose(sanitizer.get_dp_event_noisy_count())
    eps = accountant.get_epsilon(1e-5)
    self.assertAlmostEqual(eps, np.inf)

  @parameterized.parameters(
      (1, True),
      (2, True),
      (10, True),
      (10, False)
  )
  def test_dpam_only_full_batch(self, steps, account_for_count):
    # We want to test the privacy of the training part only, without the noisy
    # count. This should happen when 1). count_stddev is set of 0 (while both
    # budget and count_sample are non-zero), and 2). account_for_count is set
    # to false when calling compute_epsilon.
    count_stddev = 0.0 if account_for_count else 1.0
    orders = [0.001 * i for i in range(1000, 100000)]
    delta = 1e-6
    noise = 4.530877117
    sanitizer = dpam.Sanitizer(
        budget=123,
        count_sample=123,
        count_stddev=count_stddev)
    accountant = dpam.DPAMAccountant(
        sanitizer,
        num_users_per_batch=123,
        num_users=123,
        noise_multiplier=noise * np.sqrt(steps),
        gradient_accumulation_steps=1,
        steps=steps)
    eps = accountant.compute_epsilon(delta, account_for_count, orders=orders)
    self.assertAlmostEqual(eps, 1)

  # Test case copied from
  # tensorflow_privacy/privacy/analysis/compute_dp_sgd_privacy_test.py.
  @parameterized.parameters(
      (60000, 150, 1, 1.3, 15, 1e-5, 0.7242234026109595),
      (100000, 100, 1, 1.0, 30, 1e-7, 1.4154988495444845),
      (100000000, 1024, 1, 0.1, 10, 1e-7, 5907982.31138195),
      (100000000, 512, 2, 0.1, 10, 1e-7, 5907982.31138195),
  )
  def test_dpam_only_minibatch(
      self, num_examples, batch_size, acc_steps, noise_multiplier, epochs,
      delta, expected_eps):
    # Orders copied from
    # tensorflow_privacy/privacy/analysis/compute_dp_sgd_privacy_lib.py.
    orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
              list(range(5, 64)) + [128, 256, 512])
    sanitizer = dpam.Sanitizer(
        budget=123,
        count_sample=123,
        count_stddev=1.0)
    accountant = dpam.DPAMAccountant(
        sanitizer,
        num_users_per_batch=batch_size,
        num_users=num_examples,
        noise_multiplier=noise_multiplier,
        gradient_accumulation_steps=acc_steps,
        steps=int(np.ceil(epochs * num_examples / (batch_size*acc_steps))))
    eps = accountant.compute_epsilon(delta, account_for_count=False,
                                     orders=orders)
    self.assertAlmostEqual(eps, expected_eps)

  def test_dpam_only_no_op(self):
    sanitizer = dpam.Sanitizer(
        budget=123,
        count_sample=123,
        count_stddev=1.0)
    accountant = dpam.DPAMAccountant(
        sanitizer,
        num_users_per_batch=123,
        num_users=123,
        noise_multiplier=1.0,
        gradient_accumulation_steps=1,
        steps=0)  # set to 0 such that there is no privacy loss from training.
    eps = accountant.compute_epsilon(1e-5, account_for_count=False)
    self.assertAlmostEqual(eps, 0)

  def test_dpals_set_sigmas(self):
    epsilon = 1
    delta = 1e-5
    sanitizer = dpam.Sanitizer(budget=100, count_sample=10, count_stddev=100.0)
    optimizer = dpam.DPALSOptimizer(budget=100)
    accountant = dpam.DPALSAccountant(sanitizer, optimizer, steps=5)
    accountant.set_sigmas(target_delta=delta, target_epsilon=epsilon)
    self.assertAlmostEqual(accountant.compute_epsilon(delta), epsilon, places=4)

  def test_dpam_set_noise_multiplier(self):
    epsilon = 1
    delta = 1e-5
    sanitizer = dpam.Sanitizer(
        budget=100,
        count_sample=10,
        count_stddev=100.0)
    accountant = dpam.DPAMAccountant(
        sanitizer,
        num_users_per_batch=10,
        noise_multiplier=1.0,
        gradient_accumulation_steps=1,
        steps=5,
        num_users=100)
    accountant.set_noise_multiplier(target_delta=delta, target_epsilon=epsilon)
    self.assertAlmostEqual(accountant.compute_epsilon(delta), epsilon, places=4)


if __name__ == "__main__":
  tf.test.main()
