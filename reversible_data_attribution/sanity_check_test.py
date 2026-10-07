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

r"""Unit tests for sanity_check.py."""

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
from torch import nn
from torch.utils import data as torch_data

from reversible_data_attribution import sanity_check


class SanityCheckTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    np.random.seed(42)

  def test_resolve_device(self):
    dev_cpu = sanity_check.resolve_device('cpu')
    self.assertEqual(dev_cpu.type, 'cpu')
    dev_auto = sanity_check.resolve_device('auto')
    self.assertIn(dev_auto.type, ('cpu', 'cuda'))

  def test_load_sanity_dataset_synthetic(self):
    train_loader, val_loader, test_loader, input_dim, num_classes, corrupted = (
        sanity_check.load_sanity_dataset(
            dataset_name='synthetic',
            num_train=60,
            num_val=20,
            num_test=20,
            batch_size=15,
            noise_type='none',
            noise_rate=0.0,
            seed=42,
        )
    )
    self.assertEqual(input_dim, 64)
    self.assertEqual(num_classes, 10)
    self.assertEmpty(corrupted)
    self.assertLen(train_loader.dataset, 60)
    self.assertLen(val_loader.dataset, 20)
    self.assertLen(test_loader.dataset, 20)

    # Check batch shapes
    x_b, y_b = next(iter(train_loader))
    self.assertEqual(x_b.shape, torch.Size([15, 64]))
    self.assertEqual(y_b.shape, torch.Size([15]))

  def test_load_sanity_dataset_with_label_flip(self):
    train_loader, _, test_loader, _, _, corrupted = (
        sanity_check.load_sanity_dataset(
            dataset_name='synthetic',
            num_train=100,
            num_val=20,
            num_test=20,
            batch_size=20,
            noise_type='label_flip',
            noise_rate=0.2,
            seed=42,
        )
    )
    self.assertLen(corrupted, 20)
    self.assertLen(train_loader.dataset, 100)
    self.assertLen(test_loader.dataset, 20)

  def test_load_sanity_dataset_with_gaussian_noise(self):
    train_loader, _, test_loader, _, _, corrupted = (
        sanity_check.load_sanity_dataset(
            dataset_name='synthetic',
            num_train=100,
            num_val=20,
            num_test=20,
            batch_size=20,
            noise_type='gaussian',
            noise_rate=0.15,
            gaussian_std=0.2,
            seed=42,
        )
    )
    self.assertLen(corrupted, 15)
    self.assertLen(train_loader.dataset, 100)
    self.assertLen(test_loader.dataset, 20)

  def test_evaluate_model(self):
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )
    x = torch.randn(30, 10)
    y = torch.randint(0, 2, (30,))
    ds = torch_data.TensorDataset(x, y)
    loader = torch_data.DataLoader(ds, batch_size=10, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    loss, acc = sanity_check.evaluate_model(
        model, loader, criterion, torch.device('cpu')
    )
    self.assertIsInstance(loss, float)
    self.assertIsInstance(acc, float)
    self.assertGreater(loss, 0.0)
    self.assertBetween(acc, 0.0, 100.0)

  def test_evaluate_clean_subset_accuracy(self):
    x = torch.randn(20, 5)
    y = torch.zeros(20, dtype=torch.long)
    ds = torch_data.TensorDataset(x, y)
    corrupted_indices = np.array([0, 1, 2, 3, 4])
    model = nn.Linear(5, 2)
    acc = sanity_check.evaluate_clean_subset_accuracy(
        model, ds, corrupted_indices, torch.device('cpu')
    )
    self.assertBetween(acc, 0.0, 100.0)

  def test_check_convergence_criteria_target_loss(self):
    converged, reason = sanity_check.check_convergence_criteria(
        epoch=5,
        loss_history=[1.0, 0.8, 0.5, 0.2, 0.04],
        train_loss=0.04,
        train_acc=98.0,
        val_loss=0.05,
        val_acc=97.0,
        target_loss=0.05,
    )
    self.assertTrue(converged)
    self.assertIn('Target train loss reached', reason)

  def test_check_convergence_criteria_target_acc(self):
    converged, reason = sanity_check.check_convergence_criteria(
        epoch=5,
        loss_history=[1.0, 0.8, 0.5, 0.2, 0.1],
        train_loss=0.1,
        train_acc=96.5,
        val_loss=0.15,
        val_acc=95.0,
        target_acc=95.0,
    )
    self.assertTrue(converged)
    self.assertIn('Target train accuracy reached', reason)

  def test_check_convergence_criteria_plateau(self):
    converged, reason = sanity_check.check_convergence_criteria(
        epoch=6,
        loss_history=[1.0, 0.5, 0.2001, 0.20008, 0.20005, 0.20002],
        train_loss=0.20002,
        train_acc=85.0,
        val_loss=0.22,
        val_acc=84.0,
        loss_tolerance=1e-3,
        patience=3,
    )
    self.assertTrue(converged)
    self.assertIn('Loss plateau detected', reason)

  @parameterized.parameters(
      ('dnn',),
      ('small_dnn',),
      ('linear',),
  )
  def test_run_convergence_sanity_check_architectures(self, model_type):
    results = sanity_check.run_convergence_sanity_check(
        model_type=model_type,
        dataset_name='synthetic',
        num_train=40,
        num_val=10,
        num_test=10,
        num_epochs=3,
        batch_size=10,
        learning_rate=0.01,
        noise_type='label_flip',
        noise_rate=0.1,
        device='cpu',
    )
    self.assertIn('convergence_summary', results)
    self.assertIn('history', results)
    self.assertIn('config', results)
    self.assertIn('model_summary', results)
    self.assertLen(results['history'], 3)
    self.assertEqual(results['config']['model_type'], model_type)
    self.assertGreater(results['model_summary']['trainable_parameters'], 0)

  def test_run_convergence_sanity_check_image_models_mnist(self):
    # Tests CNN and ViT with MNIST npz dataset
    with tempfile.TemporaryDirectory() as tmpdir:
      npz_path = os.path.join(tmpdir, 'mnist_dummy.npz')
      x_tr_dummy = np.random.randint(0, 256, (50, 28, 28), dtype=np.uint8)
      y_tr_dummy = np.random.randint(0, 10, (50,), dtype=np.int64)
      x_te_dummy = np.random.randint(0, 256, (20, 28, 28), dtype=np.uint8)
      y_te_dummy = np.random.randint(0, 10, (20,), dtype=np.int64)
      np.savez_compressed(
          npz_path,
          x_train=x_tr_dummy,
          y_train=y_tr_dummy,
          x_test=x_te_dummy,
          y_test=y_te_dummy,
      )
      for arch in ('cnn', 'vit'):
        results = sanity_check.run_convergence_sanity_check(
            model_type=arch,
            dataset_name='mnist',
            data_path=npz_path,
            num_train=30,
            num_val=10,
            num_test=10,
            num_epochs=2,
            batch_size=10,
            learning_rate=0.001,
            noise_type='gaussian',
            noise_rate=0.1,
            device='cpu',
        )
        self.assertLen(results['history'], 2)
        self.assertEqual(results['model_summary']['input_dim'], 784)
        self.assertGreater(results['model_summary']['trainable_parameters'], 0)

  def test_run_convergence_sanity_check_early_stopping(self):
    results = sanity_check.run_convergence_sanity_check(
        model_type='linear',
        dataset_name='synthetic',
        num_train=40,
        num_val=10,
        num_test=10,
        num_epochs=20,
        batch_size=10,
        learning_rate=0.01,
        target_acc=0.0,  # Will trigger immediately on epoch 1
        early_stopping=True,
        device='cpu',
    )
    self.assertTrue(results['convergence_summary']['converged'])
    self.assertEqual(results['convergence_summary']['total_epochs_trained'], 1)

  def test_save_sanity_check_results(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      sanity_check.run_convergence_sanity_check(
          model_type='dnn',
          dataset_name='synthetic',
          num_train=30,
          num_val=10,
          num_test=10,
          num_epochs=2,
          batch_size=10,
          learning_rate=0.01,
          output_dir=tmpdir,
          save_csv=True,
          device='cpu',
      )
      history_csv = os.path.join(tmpdir, 'convergence_history.csv')
      summary_csv = os.path.join(tmpdir, 'convergence_summary.csv')
      summary_json = os.path.join(tmpdir, 'sanity_check_summary.json')

      self.assertTrue(os.path.exists(history_csv))
      self.assertTrue(os.path.exists(summary_csv))
      self.assertTrue(os.path.exists(summary_json))

      with open(history_csv, 'r') as f:
        content = f.read()
        self.assertIn('train_loss', content)
        self.assertIn('test_acc', content)


if __name__ == '__main__':
  absltest.main()
