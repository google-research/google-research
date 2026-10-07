# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Unit tests for data_cleansing.py (Section 7.2 Data Cleansing evaluation)."""

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from reversible_data_attribution import base_model
from reversible_data_attribution import data_cleansing
from reversible_data_attribution import utils


class DummyNet(nn.Module):
  """Simple binary classification model for testing."""

  def __init__(self, input_dim = 4):
    super().__init__()
    self.fc = nn.Linear(input_dim, 2)

  def forward(self, x):
    return self.fc(x)


class DataCleansingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    np.random.seed(0)

    self.input_dim = 4
    self.num_train = 20
    self.num_val = 10
    self.num_test = 10

    # Generate synthetic training, validation, and test data
    x_tr = torch.randn(self.num_train, self.input_dim)
    y_tr = torch.randint(0, 2, (self.num_train,))
    self.train_dataset = torch.utils.data.TensorDataset(x_tr, y_tr)
    self.train_loader = torch.utils.data.DataLoader(
        self.train_dataset, batch_size=5, shuffle=False
    )

    x_v = torch.randn(self.num_val, self.input_dim)
    y_v = torch.randint(0, 2, (self.num_val,))
    self.val_dataset = torch.utils.data.TensorDataset(x_v, y_v)
    self.val_loader = torch.utils.data.DataLoader(
        self.val_dataset, batch_size=5, shuffle=False
    )

    x_t = torch.randn(self.num_test, self.input_dim)
    y_t = torch.randint(0, 2, (self.num_test,))
    self.test_dataset = torch.utils.data.TensorDataset(x_t, y_t)
    self.test_loader = torch.utils.data.DataLoader(
        self.test_dataset, batch_size=5, shuffle=False
    )

    self.model_fn = lambda: DummyNet(self.input_dim)

    # Train dummy checkpoints with Adam optimizer to record valid momentum and variance
    self.checkpoints = []
    self.momentum_lst = []
    self.variance_lst = []
    m = self.model_fn()
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    criterion = nn.functional.cross_entropy

    for batch in self.train_loader:
      self.checkpoints.append(copy_model(m))
      m_t = [
          opt.state[p]['exp_avg'].clone()
          if p in opt.state and 'exp_avg' in opt.state[p]
          else torch.zeros_like(p)
          for p in m.parameters()
      ]
      v_t = [
          opt.state[p]['exp_avg_sq'].clone()
          if p in opt.state and 'exp_avg_sq' in opt.state[p]
          else torch.zeros_like(p)
          for p in m.parameters()
      ]
      self.momentum_lst.append(m_t)
      self.variance_lst.append(v_t)

      out = m(batch[0])
      loss = criterion(out, batch[1])
      opt.zero_grad()
      loss.backward()
      opt.step()

    self.checkpoints.append(copy_model(m))
    m_t = [
        opt.state[p]['exp_avg'].clone()
        if p in opt.state and 'exp_avg' in opt.state[p]
        else torch.zeros_like(p)
        for p in m.parameters()
    ]
    v_t = [
        opt.state[p]['exp_avg_sq'].clone()
        if p in opt.state and 'exp_avg_sq' in opt.state[p]
        else torch.zeros_like(p)
        for p in m.parameters()
    ]
    self.momentum_lst.append(m_t)
    self.variance_lst.append(v_t)

    self.base_model_obj = base_model.AttributionBaseModel(
        model=self.model_fn(),
        loss_fn=nn.functional.cross_entropy,
        lr=0.01,
        seed=42,
    )
    self.base_model_obj.train_model(
        self.train_loader, num_epochs=1, checkpoint_freq=1
    )
    self.base_model_artifacts = {
        'train_loader': self.train_loader,
        'checkpoints': self.checkpoints,
        'momentum_lst': self.momentum_lst,
        'variance_lst': self.variance_lst,
        'loss_fn': nn.functional.cross_entropy,
        'hyperparams': {
            'lr': 0.01,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'eps': 1e-8,
            'seed': 42,
        },
    }

  def test_compute_validation_gradient(self):
    m = self.checkpoints[-1]
    u = data_cleansing.compute_validation_gradient(
        m, nn.functional.cross_entropy, self.val_loader
    )
    self.assertLen(u, len(list(m.parameters())))
    for tensor in u:
      self.assertFalse(torch.isnan(tensor).any())

  def test_compute_sgd_influence(self):
    # Full checkpoints -> uses recursive_update
    sgd_all = data_cleansing.compute_scores_for_method(
        method='sgd_all',
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
    )
    self.assertEqual(sgd_all.shape, (self.num_train,))
    self.assertFalse(np.isnan(sgd_all).any())

    # Partial/single checkpoint -> uses forward_update
    partial_artifacts = dict(self.base_model_artifacts)
    partial_artifacts['checkpoints'] = [self.checkpoints[0]]
    sgd_partial = data_cleansing.compute_scores_for_method(
        method='sgd_all',
        base_model_obj=partial_artifacts,
        val_loader=self.val_loader,
    )
    self.assertEqual(sgd_partial.shape, (self.num_train,))
    self.assertFalse(np.isnan(sgd_partial).any())

    # Candidate indices with partial checkpoint
    sgd_partial_cand = data_cleansing.compute_scores_for_method(
        method='sgd_all',
        base_model_obj=partial_artifacts,
        val_loader=self.val_loader,
        candidate_indices=[0, 1],
    )
    self.assertIsInstance(sgd_partial_cand, dict)
    self.assertLen(sgd_partial_cand, 2)

  def test_compute_adam_influence_recursive(self):
    adam_infl = data_cleansing.compute_scores_for_method(
        method='adam_recursive',
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
    )
    self.assertEqual(adam_infl.shape, (self.num_train,))
    self.assertFalse(np.isnan(adam_infl).any())

  def test_compute_adam_influence_recursive_nodv(self):
    adam_infl = data_cleansing.compute_scores_for_method(
        method='adam_recursive_nodv',
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
    )
    self.assertEqual(adam_infl.shape, (self.num_train,))
    self.assertFalse(np.isnan(adam_infl).any())

  def test_get_adam_method_name(self):
    self.assertEqual(data_cleansing.get_adam_method_name(None), 'adam_exact')
    self.assertEqual(data_cleansing.get_adam_method_name(0.0), 'adam_exact')
    self.assertEqual(
        data_cleansing.get_adam_method_name(0.1), 'adam_masked_10%'
    )
    self.assertEqual(
        data_cleansing.get_adam_method_name(0.05), 'adam_masked_5%'
    )
    self.assertEqual(
        data_cleansing.get_adam_method_name(0.01), 'adam_masked_1%'
    )
    self.assertEqual(
        data_cleansing.get_adam_method_name(0.015), 'adam_masked_1.5%'
    )
    self.assertEqual(
        data_cleansing.get_adam_method_name(0.001), 'adam_masked_1e-3'
    )
    self.assertEqual(
        data_cleansing.get_adam_method_name(0.0005), 'adam_masked_5e-4'
    )
    self.assertEqual(
        data_cleansing.get_adam_method_name(1e-4), 'adam_masked_1e-4'
    )

  def test_compute_adam_influence_exact(self):
    adam_infl = data_cleansing.compute_scores_for_method(
        method='adam_exact',
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
    )
    self.assertEqual(adam_infl.shape, (self.num_train,))
    self.assertFalse(np.isnan(adam_infl).any())

  def test_compute_adam_influence_forward(self):
    adam_infl = data_cleansing.compute_scores_for_method(
        method='adam_forward_nodv',
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
    )
    self.assertEqual(adam_infl.shape, (self.num_train,))
    self.assertFalse(np.isnan(adam_infl).any())

  def test_compute_adam_influence_forward_masked(self):
    mask = utils.generate_random_mask(self.checkpoints[-1], prob=0.5, seed=42)
    adam_infl = data_cleansing.compute_scores_for_method(
        method='adam_masked_50%',
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
        random_mask=mask,
    )
    self.assertEqual(adam_infl.shape, (self.num_train,))
    self.assertFalse(np.isnan(adam_infl).any())

  def test_compute_tracin_sgd_influence(self):
    tracin_sgd_infl = data_cleansing.compute_tracin_sgd_influence(
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
    )
    self.assertEqual(tracin_sgd_infl.shape, (self.num_train,))
    self.assertFalse(np.isnan(tracin_sgd_infl).any())

  def test_compute_tracin_adam_influence(self):
    tracin_adam_infl = data_cleansing.compute_tracin_adam_influence(
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
    )
    self.assertEqual(tracin_adam_infl.shape, (self.num_train,))
    self.assertFalse(np.isnan(tracin_adam_infl).any())

  def test_compute_scores_for_methods_dict_artifacts(self):
    ckpt_dict = {0: self.checkpoints[0], -1: self.checkpoints[-1]}
    m_dict = {0: self.momentum_lst[0], -1: self.momentum_lst[-1]}
    v_dict = {0: self.variance_lst[0], -1: self.variance_lst[-1]}
    num_steps = len(self.checkpoints) - 1

    artifacts = {
        'train_loader': self.train_loader,
        'checkpoints': ckpt_dict,
        'momentum_lst': m_dict,
        'variance_lst': v_dict,
        'loss_fn': nn.functional.cross_entropy,
        'hyperparams': {'lr': 0.01},
    }

    scores = data_cleansing.compute_scores_for_methods(
        methods=['adam_recursive_nodv', 'adam_exact', 'tracin_adam', 'sgd_all'],
        base_model_obj=artifacts,
        val_loader=self.val_loader,
        num_steps=num_steps,
    )
    for m in ['adam_recursive_nodv', 'adam_exact', 'tracin_adam', 'sgd_all']:
      self.assertIn(m, scores)
      self.assertEqual(scores[m].shape, (self.num_train,))
      self.assertFalse(np.isnan(scores[m]).any())

  def test_retrain_model_randomized_reproducibility(self):
    model1 = data_cleansing.retrain_model_with_indices_removed_randomized(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        indices_to_remove=[1, 3],
        num_epochs=2,
        batch_size=5,
        lr=0.05,
        seed=42,
    )
    model2 = data_cleansing.retrain_model_with_indices_removed_randomized(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        indices_to_remove=[1, 3],
        num_epochs=2,
        batch_size=5,
        lr=0.05,
        seed=42,
    )
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
      self.assertTrue(torch.allclose(p1, p2))

  def test_retrain_model_randomized_different_seeds(self):
    model1 = data_cleansing.retrain_model_with_indices_removed_randomized(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        indices_to_remove=[1, 3],
        num_epochs=2,
        batch_size=5,
        lr=0.05,
        seed=42,
    )
    model2 = data_cleansing.retrain_model_with_indices_removed_randomized(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        indices_to_remove=[1, 3],
        num_epochs=2,
        batch_size=5,
        lr=0.05,
        seed=123,
    )
    params_differ = False
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
      if not torch.allclose(p1, p2):
        params_differ = True
        break
    self.assertTrue(params_differ)

  def test_retrain_model_empty_indices_matches_pipeline_training(self):
    # Verify that retraining with indices_to_remove=[] produces weights identical
    # to AttributionBaseModel.train_model.
    seed = 42
    torch.manual_seed(seed)
    initial_model = self.model_fn()

    base_obj = base_model.AttributionBaseModel(
        model=initial_model,
        loss_fn=nn.functional.cross_entropy,
        lr=0.01,
        beta_1=0.9,
        beta_2=0.999,
        eps=1e-8,
        seed=seed,
    )
    final_base_model = base_obj.train_model(self.train_loader, num_epochs=2)

    retrained_model = data_cleansing.retrain_model_with_indices_removed(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        indices_to_remove=[],
        num_epochs=2,
        batch_size=5,
        lr=0.01,
        optimizer_type='adam',
        beta_1=0.9,
        beta_2=0.999,
        eps=1e-8,
        criterion=nn.functional.cross_entropy,
        seed=seed,
        device='cpu',
    )

    for p_base, p_retrain in zip(
        final_base_model.parameters(), retrained_model.parameters()
    ):
      self.assertTrue(torch.allclose(p_base, p_retrain, atol=1e-6))

  def test_retrain_and_evaluate_randomized(self):
    metrics = data_cleansing.retrain_and_evaluate_randomized(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        indices_to_remove=[0, 2],
        num_epochs=1,
        batch_size=5,
        lr=0.05,
        seed=42,
    )
    self.assertIn('train', metrics)
    self.assertIn('val', metrics)
    self.assertIn('test', metrics)
    self.assertIsInstance(metrics['train'][0], float)
    self.assertIsInstance(metrics['train'][1], float)

  def test_evaluate_cleansing_for_indices(self):
    res = data_cleansing.evaluate_cleansing_for_indices(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        indices_to_remove=[0, 2],
        num_epochs=1,
        batch_size=5,
        lr=0.05,
    )
    self.assertIn('baseline_val', res)
    self.assertIn('cleansed_val', res)
    self.assertIn('loss_change_val', res)

  def test_evaluate_data_cleansing(self):
    results = data_cleansing.evaluate_data_cleansing(
        model_fn=self.model_fn,
        checkpoints=self.checkpoints,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        corrupted_indices=[1, 3],
        momentum_lst=self.momentum_lst,
        variance_lst=self.variance_lst,
        criterion=nn.functional.cross_entropy,
        lr=0.05,
        num_epochs=1,
        batch_size=5,
        k_list=[1, 3],
        methods=[
            'sgd_all',
            'adam',
            'tracin_adam',
            'tracin_sgd',
            'random',
            'ae',
        ],
    )
    self.assertIn('baseline', results)
    self.assertIn(('remove_corrupted', 2), results)
    self.assertIn(('sgd_all', 1), results)
    self.assertIn(('adam', 1), results)
    self.assertIn(('tracin_adam', 1), results)
    self.assertIn(('tracin_sgd', 1), results)
    self.assertIn(('random', 1), results)

  def test_evaluate_corruption_and_cleansing(self):
    corrupted_indices = [1, 5, 9]
    res = data_cleansing.evaluate_corruption_and_cleansing(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        checkpoints=self.checkpoints,
        corrupted_indices=corrupted_indices,
        momentum_lst=self.momentum_lst,
        variance_lst=self.variance_lst,
        optimizer_type='adam',
        influence_method='adam_recursive',
        num_epochs=1,
        batch_size=5,
        lr=0.05,
        beta_1=0.9,
        beta_2=0.999,
    )
    self.assertIn('baseline', res)
    self.assertIn('cleansed', res)
    self.assertIn('delta', res)
    self.assertIn('oracle', res)
    self.assertIn('oracle_delta', res)
    self.assertIn('selected_indices', res)
    self.assertIn('estimated_loss_changes', res)
    self.assertIn('detection_metrics', res)
    self.assertIn('precision', res['detection_metrics'])
    self.assertIn('recall', res['detection_metrics'])
    self.assertIn('f1_score', res['detection_metrics'])

  def test_evaluate_corruption_and_cleansing_adam_exact(self):
    corrupted_indices = [1, 5, 9]
    res = data_cleansing.evaluate_corruption_and_cleansing(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        checkpoints=self.checkpoints,
        corrupted_indices=corrupted_indices,
        momentum_lst=self.momentum_lst,
        variance_lst=self.variance_lst,
        optimizer_type='adam',
        influence_method='adam_exact',
        num_epochs=1,
        batch_size=5,
        lr=0.05,
        beta_1=0.9,
        beta_2=0.999,
    )
    self.assertIn('baseline', res)
    self.assertIn('cleansed', res)
    self.assertIn('delta', res)
    self.assertIn('oracle', res)
    self.assertIn('oracle_delta', res)

  def test_evaluate_corruption_and_cleansing_adam_masked(self):
    corrupted_indices = [1, 5, 9]
    mask = utils.generate_random_mask(self.checkpoints[-1], prob=0.1, seed=42)
    res = data_cleansing.evaluate_corruption_and_cleansing(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        checkpoints=self.checkpoints,
        corrupted_indices=corrupted_indices,
        momentum_lst=self.momentum_lst,
        variance_lst=self.variance_lst,
        optimizer_type='adam',
        influence_method='adam_masked_10%',
        num_epochs=1,
        batch_size=5,
        lr=0.05,
        beta_1=0.9,
        beta_2=0.999,
        random_mask=mask,
    )
    self.assertIn('baseline', res)
    self.assertIn('cleansed', res)
    self.assertIn('delta', res)
    self.assertIn('oracle', res)
    self.assertIn('oracle_delta', res)

  def test_evaluate_corruption_and_cleansing_tracin_adam(self):
    corrupted_indices = [1, 5, 9]
    res = data_cleansing.evaluate_corruption_and_cleansing(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        checkpoints=self.checkpoints,
        corrupted_indices=corrupted_indices,
        momentum_lst=self.momentum_lst,
        variance_lst=self.variance_lst,
        optimizer_type='adam',
        influence_method='tracin_adam',
        num_epochs=1,
        batch_size=5,
        lr=0.05,
        beta_1=0.9,
        beta_2=0.999,
    )
    self.assertIn('baseline', res)
    self.assertIn('cleansed', res)
    self.assertIn('delta', res)
    self.assertIn('oracle', res)
    self.assertIn('oracle_delta', res)

  def test_evaluate_corruption_and_cleansing_tracin_sgd(self):
    corrupted_indices = [1, 5, 9]
    res = data_cleansing.evaluate_corruption_and_cleansing(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        checkpoints=self.checkpoints,
        corrupted_indices=corrupted_indices,
        optimizer_type='sgd',
        influence_method='tracin_sgd',
        num_epochs=1,
        batch_size=5,
        lr=0.05,
    )
    self.assertIn('baseline', res)
    self.assertIn('cleansed', res)
    self.assertIn('delta', res)
    self.assertIn('oracle', res)
    self.assertIn('oracle_delta', res)

  def test_evaluate_ground_truth_counterfactuals_and_cleansing(self):
    res = data_cleansing.evaluate_ground_truth_counterfactuals_and_cleansing(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        checkpoints=self.checkpoints,
        corrupted_indices=[0, 2],
        momentum_lst=self.momentum_lst,
        variance_lst=self.variance_lst,
        optimizer_type='adam',
        methods=['adam_recursive_nodv', 'sgd_all'],
        num_epochs=1,
        batch_size=5,
        lr=0.05,
        k_list=[1, 3],
        output_dir=self.create_tempdir().full_path,
    )
    self.assertIn('baseline', res)
    self.assertIn('actual_cf_losses', res)
    self.assertIn('oracle_corrupted', res)
    self.assertIn('summary_rows', res)
    self.assertIn('topk_actual_results', res)
    self.assertEqual(len(res['actual_cf_losses']['val_losses']), self.num_train)
    self.assertIn(('actual_cf', 1), res['topk_actual_results'])
    self.assertIn(('remove_corrupted', 2), res['topk_actual_results'])
    summary_methods = [r['method'] for r in res['summary_rows']]
    self.assertIn('remove_corrupted', summary_methods)
    self.assertIn('actual_cf', summary_methods)

  def test_evaluate_data_cleansing_device_offloading(self):
    cpu_checkpoints = [copy_model(m).cpu() for m in self.checkpoints]
    cpu_momentum = [[t.cpu() for t in step] for step in self.momentum_lst]
    cpu_variance = [[t.cpu() for t in step] for step in self.variance_lst]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    res = data_cleansing.evaluate_data_cleansing(
        model_fn=self.model_fn,
        checkpoints=cpu_checkpoints,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        momentum_lst=cpu_momentum,
        variance_lst=cpu_variance,
        criterion=nn.functional.cross_entropy,
        lr=0.01,
        methods=['adam_recursive_nodv', 'sgd_all', 'icml'],
        k_list=[1, 2],
        device=dev,
    )
    self.assertIn(('adam_recursive_nodv', 1), res)
    self.assertIn(('sgd_all', 1), res)
    self.assertIn(('icml', 1), res)

  def test_score_ranking_order_by_method_type(self):
    # Verify that counterfactual/influence scores rank ascending (most negative first)
    # while outlier scores (ae/iso) rank descending (highest anomaly first).
    scores = np.array([-5.0, 10.0, -1.0, 3.0], dtype=np.float32)
    # Influence methods: ascending -> indices [0, 2, 3, 1] (values -5, -1, 3, 10)
    infl_ranked = np.argsort(scores)
    np.testing.assert_array_equal(infl_ranked, [0, 2, 3, 1])

    # Outlier methods: descending -> indices [1, 3, 2, 0] (values 10, 3, -1, -5)
    outlier_ranked = np.argsort(scores)[::-1]
    np.testing.assert_array_equal(outlier_ranked, [1, 3, 2, 0])

  def test_compute_partitioned_corrupted_metrics(self):
    scores = np.array([-2.0, 1.0, 3.0, -5.0], dtype=np.float32)
    corrupted_indices = [0, 3]  # indices 0 (-2.0) and 3 (-5.0) are corrupted
    res = data_cleansing.compute_partitioned_corrupted_metrics(
        scores, corrupted_indices, is_outlier_score=False
    )
    self.assertIn('mean_score_clean', res)
    self.assertIn('mean_score_corrupted', res)
    self.assertIn('roc_auc', res)
    self.assertAlmostEqual(res['mean_score_clean'], 2.0)
    self.assertAlmostEqual(res['mean_score_corrupted'], -3.5)
    self.assertEqual(res['roc_auc'], 1.0)

  def test_evaluate_corruption_and_cleansing_with_retrain_seeds(self):
    corrupted_indices = [1, 5]
    res = data_cleansing.evaluate_corruption_and_cleansing(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        checkpoints=self.checkpoints,
        corrupted_indices=corrupted_indices,
        momentum_lst=self.momentum_lst,
        variance_lst=self.variance_lst,
        optimizer_type='adam',
        influence_method='adam_recursive',
        num_epochs=1,
        batch_size=5,
        lr=0.05,
        retrain_seeds=[42, 123],
    )
    self.assertIn('baseline', res)
    self.assertIn('cleansed', res)
    self.assertIn('delta', res)
    self.assertIn('oracle', res)
    self.assertIn('randomized_cleansed', res)
    self.assertIn(42, res['randomized_cleansed'])
    self.assertIn(123, res['randomized_cleansed'])
    self.assertIn('randomized_delta', res)
    self.assertIn(42, res['randomized_delta'])
    self.assertIn(123, res['randomized_delta'])
    self.assertIn('randomized_oracle', res)
    self.assertIn(42, res['randomized_oracle'])
    self.assertIn(123, res['randomized_oracle'])

  def test_evaluate_data_cleansing_with_retrain_seeds(self):
    results = data_cleansing.evaluate_data_cleansing(
        model_fn=self.model_fn,
        checkpoints=self.checkpoints,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        corrupted_indices=[1, 3],
        momentum_lst=self.momentum_lst,
        variance_lst=self.variance_lst,
        criterion=nn.functional.cross_entropy,
        lr=0.05,
        num_epochs=1,
        batch_size=5,
        k_list=[1],
        methods=['adam'],
        retrain_seeds=[42, 123],
    )
    self.assertIn('baseline', results)
    self.assertIn(('remove_corrupted', 2), results)
    self.assertIn(('remove_corrupted_seed_42', 2), results)
    self.assertIn(('remove_corrupted_seed_123', 2), results)
    self.assertIn(('adam', 1), results)
    self.assertIn(('adam_seed_42', 1), results)
    self.assertIn(('adam_seed_123', 1), results)

  def test_evaluate_data_cleansing_unified_details(self):
    details = data_cleansing.evaluate_data_cleansing(
        model_fn=self.model_fn,
        checkpoints=self.checkpoints,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        corrupted_indices=[1, 3],
        momentum_lst=self.momentum_lst,
        variance_lst=self.variance_lst,
        criterion=nn.functional.cross_entropy,
        lr=0.05,
        num_epochs=1,
        batch_size=5,
        k_list=[1],
        methods=['adam_exact'],
        retrain_seeds=[42],
        eval_auto=True,
        eval_oracle=True,
        return_details=True,
    )
    self.assertIn('results', details)
    self.assertIn('baseline', details)
    self.assertIn('auto_cleansing', details)
    self.assertIn('detection_metrics', details)
    self.assertIn('oracle', details)
    self.assertIn('oracle_delta', details)
    self.assertIn(('adam_exact', 1), details['results'])
    self.assertIn(
        ('adam_exact', 2), details['results']
    )  # Exact number of corrupted samples
    self.assertIn(('adam_exact', 'auto'), details['results'])
    self.assertIn(('adam_exact_seed_42', 'auto'), details['results'])
    self.assertIn('adam_exact', details['auto_cleansing'])
    auto_res = details['auto_cleansing']['adam_exact']
    self.assertIn('baseline', auto_res)
    self.assertIn('cleansed', auto_res)
    self.assertIn('delta', auto_res)
    self.assertIn('detection_metrics', auto_res)
    self.assertIn('precision', auto_res['detection_metrics'])

  def test_evaluate_data_cleansing_detection_metrics_without_auto_retraining(
      self,
  ):
    details = data_cleansing.evaluate_data_cleansing(
        model_fn=self.model_fn,
        checkpoints=self.checkpoints,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        corrupted_indices=[1, 3],
        momentum_lst=self.momentum_lst,
        variance_lst=self.variance_lst,
        criterion=nn.functional.cross_entropy,
        lr=0.05,
        num_epochs=1,
        batch_size=5,
        k_list=[1],
        methods=['adam_exact'],
        eval_auto=False,
        eval_oracle=False,
        return_details=True,
    )
    self.assertIn('detection_metrics', details)
    self.assertIn('adam_exact', details['detection_metrics'])
    det = details['detection_metrics']['adam_exact']
    self.assertIn('precision', det)
    self.assertIn('recall', det)
    self.assertIn('f1_score', det)
    self.assertIn('roc_auc', det)

  def test_evaluate_ground_truth_counterfactuals_with_retrain_seeds(self):
    res = data_cleansing.evaluate_ground_truth_counterfactuals_and_cleansing(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        checkpoints=self.checkpoints,
        corrupted_indices=[0, 2],
        momentum_lst=self.momentum_lst,
        variance_lst=self.variance_lst,
        optimizer_type='adam',
        methods=['adam_recursive_nodv'],
        num_epochs=1,
        batch_size=5,
        lr=0.05,
        k_list=[1],
        retrain_seeds=[42],
        output_dir=self.create_tempdir().full_path,
    )
    self.assertIn(('actual_cf', 1), res['topk_actual_results'])
    self.assertIn(('actual_cf_seed_42', 1), res['topk_actual_results'])
    self.assertIn(('remove_corrupted_seed_42', 2), res['topk_actual_results'])

  def test_evaluate_with_precomputed_scores(self):
    scores_dict = data_cleansing.compute_scores_for_methods(
        methods=['adam_recursive_nodv', 'sgd_all'],
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
    )
    # 1. evaluate_data_cleansing with precomputed scores_dict
    res_dc = data_cleansing.evaluate_data_cleansing(
        model_fn=self.model_fn,
        checkpoints=self.checkpoints,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        scores_dict=scores_dict,
        methods=['adam_recursive_nodv', 'sgd_all'],
        num_epochs=1,
        batch_size=5,
        k_list=[1],
    )
    self.assertIn(('adam_recursive_nodv', 1), res_dc)
    self.assertIn(('sgd_all', 1), res_dc)

    # 2. evaluate_corruption_and_cleansing with precomputed scores
    res_cc = data_cleansing.evaluate_corruption_and_cleansing(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        checkpoints=self.checkpoints,
        scores=scores_dict['adam_recursive_nodv'],
        influence_method='adam_recursive_nodv',
        corrupted_indices=[1, 3],
        num_epochs=1,
        batch_size=5,
    )
    self.assertIn('baseline', res_cc)
    self.assertIn('cleansed', res_cc)

    # 3. evaluate_ground_truth_counterfactuals_and_cleansing with precomputed scores_dict
    res_gt = data_cleansing.evaluate_ground_truth_counterfactuals_and_cleansing(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        checkpoints=self.checkpoints,
        scores_dict=scores_dict,
        corrupted_indices=[0, 2],
        methods=['adam_recursive_nodv'],
        num_epochs=1,
        batch_size=5,
        lr=0.05,
        k_list=[1],
    )
    self.assertIn('actual_cf_losses', res_gt)
    self.assertIn(('actual_cf', 1), res_gt['topk_actual_results'])

  def test_forward_update_parallel_and_candidate_indices(self):
    indices = [0, 2, 4]
    seq_scores = data_cleansing.compute_scores_for_method(
        method='adam_exact',
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
        candidate_indices=indices,
        max_workers=1,
    )
    par_scores = data_cleansing.compute_scores_for_method(
        method='adam_exact',
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
        candidate_indices=indices,
        max_workers=2,
    )
    self.assertIsInstance(seq_scores, dict)
    self.assertIsInstance(par_scores, dict)
    self.assertEqual(sorted(list(par_scores.keys())), indices)
    for idx in indices:
      self.assertAlmostEqual(seq_scores[idx], par_scores[idx], places=5)
    self.assertNotIn(1, par_scores)

  def test_compute_scores_for_methods_with_candidate_indices(self):
    candidate_indices = [2, 5, 8]
    methods = [
        'adam_recursive',
        'adam_recursive_nodv',
        'tracin_adam',
        'tracin_sgd',
        'sgd_all',
    ]
    scores = data_cleansing.compute_scores_for_methods(
        methods=methods,
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
        candidate_indices=candidate_indices,
    )
    self.assertLen(scores, 5)
    for m in methods:
      self.assertIn(m, scores)
      self.assertIsInstance(scores[m], dict)
      self.assertEqual(sorted(list(scores[m].keys())), candidate_indices)
      for idx in candidate_indices:
        self.assertFalse(np.isnan(scores[m][idx]))

  def test_save_and_load_sharded_npz_with_methods(self):
    methods = [
        'adam_recursive',
        'adam_recursive_nodv',
        'tracin_adam',
        'tracin_sgd',
        'sgd_all',
    ]
    cand_indices = [10, 11, 12, 13, 14]
    shard_scores = {
        m: {idx: float(idx * 0.1) for idx in cand_indices} for m in methods
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
      npz_path = os.path.join(tmp_dir, 'scores_shard_0000.npz')
      data_cleansing.save_scores(shard_scores, npz_path)

      # 1. Direct np.load check
      npz_raw = np.load(npz_path)
      for m in methods:
        self.assertIn(m, npz_raw.files)
        self.assertIn(f'{m}__indices', npz_raw.files)
        self.assertIn(f'{m}__values', npz_raw.files)
        self.assertEqual(len(npz_raw[m]), len(cand_indices))
        self.assertEqual(len(npz_raw[f'{m}__indices']), len(cand_indices))
        self.assertEqual(len(npz_raw[f'{m}__values']), len(cand_indices))

      # 2. load_scores check
      loaded = data_cleansing.load_scores(npz_path)
      for m in methods:
        self.assertIn(m, loaded)
        self.assertIsInstance(loaded[m], dict)
        self.assertEqual(sorted(list(loaded[m].keys())), cand_indices)
        for idx in cand_indices:
          self.assertAlmostEqual(loaded[m][idx], idx * 0.1, places=5)

  def test_save_load_merge_sharded_scores(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      shard0_file = os.path.join(tmp_dir, 'scores_shard_0000.npz')
      shard1_file = os.path.join(tmp_dir, 'scores_shard_0001.npz')

      shard0_scores = {
          'adam_exact': {0: -0.5, 1: 0.2},
          'actual_cf': {0: -0.45, 1: 0.18},
      }
      shard1_scores = {
          'adam_exact': {2: -0.1, 3: 0.8},
          'actual_cf': {2: -0.05, 3: 0.75},
      }

      data_cleansing.save_scores(shard0_scores, shard0_file)
      data_cleansing.save_scores(shard1_scores, shard1_file)

      # Test load_scores
      loaded0 = data_cleansing.load_scores(shard0_file)
      self.assertIn('adam_exact', loaded0)
      self.assertAlmostEqual(loaded0['adam_exact'][0], -0.5, places=5)

      # Test merge_sharded_scores
      merged = data_cleansing.merge_sharded_scores(tmp_dir, num_train=4)
      self.assertIn('adam_exact', merged)
      self.assertIn('actual_cf', merged)
      self.assertLen(merged['adam_exact'], 4)
      self.assertAlmostEqual(merged['adam_exact'][0], -0.5, places=5)
      self.assertAlmostEqual(merged['adam_exact'][1], 0.2, places=5)
      self.assertAlmostEqual(merged['adam_exact'][2], -0.1, places=5)
      self.assertAlmostEqual(merged['adam_exact'][3], 0.8, places=5)

      # Test merge_sharded_scores with shards/ subfolder
      shards_sub_dir = os.path.join(tmp_dir, 'shards')
      os.makedirs(shards_sub_dir, exist_ok=True)
      shard_sub_file = os.path.join(shards_sub_dir, 'scores_shard_0002.npz')
      data_cleansing.save_scores({'adam_exact': {0: 1.5}}, shard_sub_file)

      # Also place metadata JSON files in tmp_dir to ensure they are ignored
      with open(os.path.join(tmp_dir, 'config.json'), 'w') as f:
        f.write('{"learning_rate": 0.001}')
      with open(os.path.join(tmp_dir, 'corrupted_indices.json'), 'w') as f:
        f.write('[0, 1]')

      merged_parent = data_cleansing.merge_sharded_scores(tmp_dir, num_train=4)
      self.assertAlmostEqual(merged_parent['adam_exact'][0], 1.5, places=5)
      self.assertNotIn('learning_rate', merged_parent)
      self.assertNotIn('config', merged_parent)

  def test_compute_leave_one_out_counterfactuals_parallel(self):
    indices = [0, 1]
    res = data_cleansing.compute_leave_one_out_counterfactuals(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        sample_indices=indices,
        num_epochs=1,
        batch_size=5,
        lr=0.05,
        max_workers=2,
    )
    self.assertIn('val_loss_changes', res)
    self.assertIn(0, res['val_loss_changes'])
    self.assertIn(1, res['val_loss_changes'])

  def test_evaluate_ground_truth_counterfactuals_sharded_and_parallel(self):
    res = data_cleansing.evaluate_ground_truth_counterfactuals_and_cleansing(
        model_fn=self.model_fn,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        checkpoints=self.checkpoints,
        sample_indices=[0, 1],
        corrupted_indices=[0],
        methods=['adam_recursive_nodv'],
        num_epochs=1,
        batch_size=5,
        lr=0.05,
        k_list=[1],
        max_workers=2,
    )
    self.assertIn('actual_cf_losses', res)
    self.assertIn(('actual_cf', 1), res['topk_actual_results'])

  def test_compute_scores_use_reversible_transform(self):
    scores_memoized = data_cleansing.compute_scores_for_methods(
        methods=['adam_recursive', 'adam_recursive_nodv', 'adam_exact'],
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
        use_reversible_transform=False,
    )
    self.assertLen(scores_memoized['adam_recursive'], self.num_train)
    self.assertLen(scores_memoized['adam_recursive_nodv'], self.num_train)
    self.assertLen(scores_memoized['adam_exact'], self.num_train)
    self.assertFalse(np.isnan(scores_memoized['adam_recursive']).any())
    self.assertFalse(np.isnan(scores_memoized['adam_recursive_nodv']).any())

    scores_reversible = data_cleansing.compute_scores_for_methods(
        methods=['adam_recursive', 'adam_recursive_nodv', 'adam_exact'],
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
        use_reversible_transform=True,
    )
    self.assertLen(scores_reversible['adam_recursive'], self.num_train)
    self.assertLen(scores_reversible['adam_recursive_nodv'], self.num_train)
    self.assertLen(scores_reversible['adam_exact'], self.num_train)

  def test_compute_scores_start_step(self):
    num_batches = len(utils.prepare_batches(self.train_loader))
    # Test adam_recursive with start_step = num_batches
    adam_infl_start0 = data_cleansing.compute_scores_for_method(
        method='adam_recursive',
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
        start_step=0,
    )
    self.assertEqual(adam_infl_start0.shape, (self.num_train,))
    self.assertFalse(np.isnan(adam_infl_start0).any())

    adam_infl_start_batch = data_cleansing.compute_scores_for_method(
        method='adam_recursive',
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
        start_step=num_batches,
    )
    self.assertEqual(adam_infl_start_batch.shape, (self.num_train,))
    self.assertFalse(np.isnan(adam_infl_start_batch).any())

    # Test adam_recursive_nodv with start_step = num_batches
    adam_nodv_start_batch = data_cleansing.compute_scores_for_method(
        method='adam_recursive_nodv',
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
        start_step=num_batches,
    )
    self.assertEqual(adam_nodv_start_batch.shape, (self.num_train,))
    self.assertFalse(np.isnan(adam_nodv_start_batch).any())

    # Test sgd_all with start_step = num_batches
    sgd_start_batch = data_cleansing.compute_scores_for_method(
        method='sgd_all',
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
        start_step=num_batches,
    )
    self.assertEqual(sgd_start_batch.shape, (self.num_train,))
    self.assertFalse(np.isnan(sgd_start_batch).any())

  def test_compute_scores_for_methods_start_step(self):
    num_batches = len(utils.prepare_batches(self.train_loader))
    scores = data_cleansing.compute_scores_for_methods(
        methods=['adam_recursive', 'adam_recursive_nodv', 'sgd_all'],
        base_model_obj=self.base_model_artifacts,
        val_loader=self.val_loader,
        start_step=num_batches,
    )
    self.assertLen(scores['adam_recursive'], self.num_train)
    self.assertLen(scores['adam_recursive_nodv'], self.num_train)
    self.assertLen(scores['sgd_all'], self.num_train)
    self.assertFalse(np.isnan(scores['adam_recursive']).any())
    self.assertFalse(np.isnan(scores['adam_recursive_nodv']).any())
    self.assertFalse(np.isnan(scores['sgd_all']).any())

  def test_evaluate_data_cleansing_ignore_first(self):
    results = data_cleansing.evaluate_data_cleansing(
        model_fn=self.model_fn,
        checkpoints=self.checkpoints,
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        momentum_lst=self.momentum_lst,
        variance_lst=self.variance_lst,
        k_list=[1, 2],
        methods=['adam_recursive_nodv'],
        eval_auto=False,
        eval_oracle=False,
        ignore_first=True,
    )
    self.assertIn(('adam_recursive_nodv', 1), results)
    self.assertIn(('adam_recursive_nodv', 2), results)
    self.assertIn('baseline', results)

  def test_compute_scores_with_attribution_base_model(self):
    base = base_model.AttributionBaseModel(
        model=self.model_fn(),
        loss_fn=nn.functional.cross_entropy,
        lr=0.01,
        seed=0,
    )
    base.train_model(
        train_loader=self.train_loader,
        num_epochs=2,
        checkpoint_freq=1,
        device='cpu',
    )
    scores = data_cleansing.compute_scores_for_method(
        method='adam_recursive_nodv',
        val_loader=self.val_loader,
        base_model_obj=base,
        device='cpu',
    )
    self.assertIsInstance(scores, np.ndarray)
    self.assertEqual(len(scores), self.num_train)
    self.assertFalse(np.isnan(scores).any())

  def test_compute_scores_reversible_with_two_passes(self):
    base = base_model.AttributionBaseModel(
        model=self.model_fn(),
        loss_fn=nn.functional.cross_entropy,
        lr=0.01,
        seed=0,
    )
    base.train_model(
        train_loader=self.train_loader,
        num_epochs=2,
        checkpoint_freq=None,
        save_all_ckpts=False,
        device='cpu',
    )
    (
        quantized_model,
        quantized_checkpoints,
        momentum_buffer,
        variance_buffer,
    ) = base.retrain_with_quantize(
        train_loader=self.train_loader,
        num_epochs=2,
        checkpoint_freq=None,
        quantization_scale=1000000,
        device='cpu',
    )
    scores = data_cleansing.compute_scores_for_methods(
        methods=['adam_recursive', 'adam_recursive_nodv', 'sgd_all', 'random'],
        base_model_obj=base,
        val_loader=self.val_loader,
        device='cpu',
        use_reversible_transform=True,
        quantized_checkpoints=quantized_checkpoints,
        momentum_buffer=momentum_buffer,
        variance_buffer=variance_buffer,
        quantization_scale=1000000,
    )
    self.assertIn('adam_recursive', scores)
    self.assertIn('adam_recursive_nodv', scores)
    self.assertIn('sgd_all', scores)
    self.assertIn('random', scores)
    self.assertLen(scores['adam_recursive'], self.num_train)
    self.assertLen(scores['adam_recursive_nodv'], self.num_train)
    self.assertLen(scores['sgd_all'], self.num_train)
    self.assertFalse(np.isnan(scores['adam_recursive']).any())
    self.assertFalse(np.isnan(scores['adam_recursive_nodv']).any())
    self.assertFalse(np.isnan(scores['sgd_all']).any())

  def test_evaluate_data_cleansing_reversible_with_two_passes(self):
    base = base_model.AttributionBaseModel(
        model=self.model_fn(),
        loss_fn=nn.functional.cross_entropy,
        lr=0.01,
        seed=0,
    )
    base.train_model(
        train_loader=self.train_loader,
        num_epochs=2,
        checkpoint_freq=None,
        save_all_ckpts=False,
        device='cpu',
    )
    (
        quantized_model,
        quantized_checkpoints,
        momentum_buffer,
        variance_buffer,
    ) = base.retrain_with_quantize(
        train_loader=self.train_loader,
        num_epochs=2,
        checkpoint_freq=None,
        quantization_scale=1000000,
        device='cpu',
    )
    results = data_cleansing.evaluate_data_cleansing(
        model_fn=self.model_fn,
        checkpoints=base.get_checkpoint_models(),
        train_dataset=self.train_dataset,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        test_loader=self.test_loader,
        momentum_lst=base.get_momentum_dict(),
        variance_lst=base.get_variance_dict(),
        k_list=[1, 2],
        methods=['adam_recursive_nodv', 'random'],
        lr=0.01,
        eval_auto=False,
        eval_oracle=False,
        use_reversible_transform=True,
        quantized_checkpoints=quantized_checkpoints,
        momentum_buffer=momentum_buffer,
        variance_buffer=variance_buffer,
        quantization_scale=1000000,
    )
    self.assertIn(('adam_recursive_nodv', 1), results)
    self.assertIn(('adam_recursive_nodv', 2), results)
    self.assertIn(('random', 1), results)
    self.assertIn('baseline', results)


def copy_model(model):
  copied = DummyNet(4)
  copied.load_state_dict(model.state_dict())
  return copied


if __name__ == '__main__':
  absltest.main()
