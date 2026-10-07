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

"""Unit tests for online_pruning.py."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torch.utils import data as torch_data

from reversible_data_attribution import base_model
from reversible_data_attribution import online_pruning


class SimpleClassifier(nn.Module):

  def __init__(self, input_dim = 4, output_dim = 2):
    super().__init__()
    self.fc = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    return self.fc(x)


class IndexedDataset(torch_data.Dataset):

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx], idx


class OnlinePruningTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    self.x_train = torch.randn(30, 4)
    self.y_train = torch.randint(0, 2, (30,))
    # Corrupt 5 labels
    self.corrupted_indices = [2, 5, 8, 12, 17]
    for idx in self.corrupted_indices:
      self.y_train[idx] = 1 - self.y_train[idx]

    self.train_dataset = IndexedDataset(self.x_train, self.y_train)
    self.train_loader = torch_data.DataLoader(
        self.train_dataset, batch_size=5, shuffle=False
    )

    self.x_val = torch.randn(10, 4)
    self.y_val = torch.randint(0, 2, (10,))
    self.val_dataset = IndexedDataset(self.x_val, self.y_val)
    self.val_loader = torch_data.DataLoader(
        self.val_dataset, batch_size=5, shuffle=False
    )

    self.model = SimpleClassifier(input_dim=4, output_dim=2)
    self.criterion = nn.CrossEntropyLoss()

  def test_compute_validation_gradient(self):
    flat_grad = online_pruning.compute_validation_gradient(
        self.model,
        criterion=self.criterion,
        val_loader=self.val_loader,
        device='cpu',
        flatten=True,
    )
    num_params = sum(p.numel() for p in self.model.parameters())
    self.assertEqual(flat_grad.shape, (num_params,))
    self.assertFalse(torch.isnan(flat_grad).any())

  @parameterized.parameters(
      'counterfactual', 'counterfactual_backward', 'tracin', 'loss', 'random'
  )
  def test_compute_windowed_scores(self, score_type):
    base = base_model.AttributionBaseModel(
        model=self.model, loss_fn=self.criterion, lr=0.01, seed=42
    )
    batches = list(self.train_loader)
    _, steps, win_batches = base.train_model_continue(
        batches=batches[:2],
        save_all_ckpts=True,
        checkpoint_freq=1,
        device='cpu',
    )
    self.assertLen(steps, 2)
    self.assertLen(win_batches, 2)

    scores = online_pruning.compute_windowed_scores(
        base_model_obj=base,
        window_steps=steps,
        window_batches=win_batches,
        val_loader=self.val_loader,
        score_type=score_type,
        device='cpu',
    )
    # Samples in first 2 batches (batch_size=5 -> 10 samples)
    self.assertLen(scores, 10)
    for sample_id, score in scores.items():
      self.assertIsInstance(sample_id, int)
      self.assertIsInstance(score, float)
      self.assertFalse(torch.isnan(torch.tensor(score)))

  def test_run_online_pruning_curriculum(self):
    config = online_pruning.OnlinePruningConfig(
        warmup_epochs=1,
        pruning_epochs=2,
        post_pruning_epochs=1,
        prune_budget=0.20,
        window_size=2,
        score_type='counterfactual',
        device='cpu',
    )
    result = online_pruning.run_online_pruning(
        model=self.model,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        config=config,
        loss_fn=self.criterion,
        lr=0.01,
        corrupted_indices=self.corrupted_indices,
        seed=42,
    )
    # Total samples = 30, budget = 20% -> 6 samples pruned
    self.assertIsInstance(result.online_model, nn.Module)
    self.assertIsInstance(result.retrained_model, nn.Module)
    self.assertNotEmpty(result.pruned_indices)
    self.assertLessEqual(len(result.pruned_indices), 6)

    # Check history entries
    self.assertLen(result.history, 4)  # 1 warmup + 2 pruning + 1 post-pruning
    self.assertEqual(result.history[0]['phase'], 'A_warmup')
    self.assertEqual(result.history[1]['phase'], 'B_pruning')
    self.assertEqual(result.history[2]['phase'], 'B_pruning')
    self.assertEqual(result.history[3]['phase'], 'C_post_pruning')

    # Metrics
    self.assertIn('online_val_loss', result.metrics)
    self.assertIn('online_val_acc', result.metrics)
    self.assertIn('retrain_val_loss', result.metrics)
    self.assertIn('retrain_val_acc', result.metrics)
    self.assertIn('data_precision', result.metrics)
    self.assertIn('data_recall', result.metrics)
    self.assertIn('data_f1', result.metrics)


if __name__ == '__main__':
  absltest.main()
