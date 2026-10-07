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

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torch.utils import data as torch_data

from reversible_data_attribution import base_model


class SimpleTestNet(nn.Module):

  def __init__(self, input_dim = 4, output_dim = 2):
    super().__init__()
    self.fc = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    return self.fc(x)


class AttributionBaseModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    self.x_train = torch.randn(20, 4)
    self.y_train = torch.randint(0, 2, (20,))
    self.dataset = torch_data.TensorDataset(self.x_train, self.y_train)
    self.data_loader = torch_data.DataLoader(
        self.dataset, batch_size=5, shuffle=False
    )
    self.model = SimpleTestNet(input_dim=4, output_dim=2)
    self.criterion = nn.CrossEntropyLoss()

  def test_initialization_and_getters(self):
    base = base_model.AttributionBaseModel(
        model=self.model,
        loss_fn=self.criterion,
        lr=0.005,
        beta_1=0.85,
        beta_2=0.995,
        eps=1e-7,
        seed=123,
    )
    self.assertEqual(base.lr(), 0.005)
    self.assertEqual(base.learning_rate(), 0.005)
    self.assertEqual(base.beta_1(), 0.85)
    self.assertEqual(base.beta_2(), 0.995)
    self.assertEqual(base.eps(), 1e-7)
    self.assertEqual(base.epsilon(), 1e-7)
    self.assertEqual(base.seed(), 123)
    self.assertFalse(base.is_trained())
    self.assertIsNotNone(base.optimizer())
    self.assertIsNotNone(base.model())
    self.assertIsNotNone(base.initial_model())

  def test_alias(self):
    self.assertIs(base_model.BaseModel, base_model.AttributionBaseModel)

  def test_train_model_and_checkpoints(self):
    base = base_model.AttributionBaseModel(
        model=self.model,
        loss_fn=self.criterion,
        lr=0.01,
        seed=42,
    )
    # 20 samples, batch_size=5 -> 4 batches per epoch. 2 epochs -> 8 steps.
    trained_model = base.train_model(
        train_loader=self.data_loader,
        num_epochs=2,
        checkpoint_freq=2,
        device='cpu',
    )
    self.assertTrue(base.is_trained())
    self.assertEqual(base.num_epochs(), 2)
    self.assertEqual(base.total_steps(), 8)
    self.assertIs(trained_model, base.model())

    ckpts = base.checkpoints()
    # Should contain step 0, step 2, step 4, step 6, step 8, and step -1
    self.assertIn(0, ckpts)
    self.assertIn(2, ckpts)
    self.assertIn(4, ckpts)
    self.assertIn(6, ckpts)
    self.assertIn(8, ckpts)
    self.assertIn(-1, ckpts)

    # Step 0 momentum and variance should be zeros
    for m in ckpts[0].momentum:
      self.assertTrue(torch.all(m == 0))
    for v in ckpts[0].variance:
      self.assertTrue(torch.all(v == 0))

    # Helper getters
    models_dict = base.get_checkpoint_models()
    momentum_dict = base.get_momentum_dict()
    variance_dict = base.get_variance_dict()
    gradients_dict = base.get_gradients_dict()

    self.assertEqual(len(models_dict), len(ckpts))
    self.assertEqual(len(momentum_dict), len(ckpts))
    self.assertEqual(len(variance_dict), len(ckpts))
    self.assertIn(8, gradients_dict)

  def test_cannot_train_more_than_once(self):
    base = base_model.AttributionBaseModel(
        model=self.model,
        loss_fn=self.criterion,
    )
    base.train_model(self.data_loader, num_epochs=1)
    with self.assertRaises(RuntimeError):
      base.train_model(self.data_loader, num_epochs=1)

  def test_retrain_remove_indices_single(self):
    base = base_model.AttributionBaseModel(
        model=self.model,
        loss_fn=self.criterion,
        lr=0.01,
        seed=42,
    )
    base.train_model(self.data_loader, num_epochs=2)

    # Retrain with index 0 removed
    cf_model = base.retrain_remove_indices(indices_to_remove=0)
    self.assertIsInstance(cf_model, nn.Module)

    # Verify parameters differ from full model
    base_params = [p.clone() for p in base.model().parameters()]
    cf_params = [p.clone() for p in cf_model.parameters()]
    diffs = [
        torch.norm(p1 - p2).item() for p1, p2 in zip(base_params, cf_params)
    ]
    self.assertGreater(sum(diffs), 0.0)

  def test_retrain_remove_indices_multiple(self):
    base = base_model.AttributionBaseModel(
        model=self.model,
        loss_fn=self.criterion,
        lr=0.01,
        seed=42,
    )
    base.train_model(self.data_loader, num_epochs=2)

    cf_model = base.retrain_remove_indices(indices_to_remove=[0, 1, 2])
    self.assertIsInstance(cf_model, nn.Module)

  def test_retrain_remove_indices_dict_dataset(self):
    dict_batches = [
        {
            'x': self.x_train[0:5],
            'y': self.y_train[0:5],
            'index': torch.tensor([10, 11, 12, 13, 14]),
        },
        {
            'x': self.x_train[5:10],
            'y': self.y_train[5:10],
            'index': torch.tensor([15, 16, 17, 18, 19]),
        },
    ]
    base = base_model.AttributionBaseModel(
        model=self.model,
        loss_fn=self.criterion,
        lr=0.01,
        seed=42,
    )
    base.train_model(dict_batches, num_epochs=1)
    cf_model = base.retrain_remove_indices(indices_to_remove=10)
    self.assertIsInstance(cf_model, nn.Module)

  def test_train_model_default_saves_first_and_last_only(self):
    base = base_model.AttributionBaseModel(
        model=self.model,
        loss_fn=self.criterion,
        lr=0.01,
        seed=42,
    )
    # 20 samples, batch_size=5 -> 4 batches per epoch. 2 epochs -> 8 steps.
    base.train_model(
        train_loader=self.data_loader,
        num_epochs=2,
        checkpoint_freq=None,
        save_all_ckpts=False,
    )
    ckpts = base.checkpoints()
    self.assertEqual(sorted(ckpts.keys()), [-1, 0, 8])
    self.assertIs(ckpts[-1], ckpts[8])

  def test_retrain_with_quantize(self):
    base = base_model.AttributionBaseModel(
        model=self.model,
        loss_fn=self.criterion,
        lr=0.01,
        seed=42,
    )
    base.train_model(
        train_loader=self.data_loader,
        num_epochs=2,
        checkpoint_freq=None,
        save_all_ckpts=False,
    )
    (
        quantized_model,
        quantized_checkpoints,
        momentum_buffer,
        variance_buffer,
    ) = base.retrain_with_quantize(
        train_loader=self.data_loader,
        num_epochs=2,
        checkpoint_freq=None,
        quantization_scale=1000000,
    )
    self.assertIsInstance(quantized_model, nn.Module)
    self.assertIn(0, quantized_checkpoints)
    self.assertIn(-1, quantized_checkpoints)
    self.assertIn(8, quantized_checkpoints)
    self.assertEqual(sorted(quantized_checkpoints.keys()), [-1, 0, 8])
    self.assertIsNotNone(momentum_buffer)
    self.assertIsNotNone(variance_buffer)

  def test_retrain_with_quantize_with_checkpoint_freq(self):
    base = base_model.AttributionBaseModel(
        model=self.model,
        loss_fn=self.criterion,
        lr=0.01,
        seed=42,
    )
    (
        quantized_model,
        quantized_checkpoints,
        momentum_buffer,
        variance_buffer,
    ) = base.retrain_with_quantize(
        train_loader=self.data_loader,
        num_epochs=2,
        checkpoint_freq=2,
        quantization_scale=1000000,
    )
    self.assertIsInstance(quantized_model, nn.Module)
    self.assertIn(0, quantized_checkpoints)
    self.assertIn(2, quantized_checkpoints)
    self.assertIn(4, quantized_checkpoints)
    self.assertIn(6, quantized_checkpoints)
    self.assertIn(8, quantized_checkpoints)
    self.assertIn(-1, quantized_checkpoints)
    self.assertIsNotNone(momentum_buffer)
    self.assertIsNotNone(variance_buffer)

  def test_retrain_with_quantize_save_all_ckpts(self):
    base = base_model.AttributionBaseModel(
        model=self.model,
        loss_fn=self.criterion,
        lr=0.01,
        seed=42,
    )
    (
        quantized_model,
        quantized_checkpoints,
        momentum_buffer,
        variance_buffer,
    ) = base.retrain_with_quantize(
        train_loader=self.data_loader,
        num_epochs=2,
        save_all_ckpts=True,
        quantization_scale=1000000,
    )
    self.assertIsInstance(quantized_model, nn.Module)
    self.assertEqual(
        sorted(quantized_checkpoints.keys()), [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    )
    self.assertIsNotNone(momentum_buffer)
    self.assertIsNotNone(variance_buffer)


if __name__ == '__main__':
  absltest.main()
