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

import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torch.utils import data as torch_data

from reversible_data_attribution import base_model
from reversible_data_attribution import infl_sgd


class SimpleModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(4, 2)

  def forward(self, x):
    return self.linear(x)


class InflSgdTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    self.x_train = torch.randn(20, 4)
    self.y_train = torch.randint(0, 2, (20,))
    self.dataset = torch_data.TensorDataset(self.x_train, self.y_train)
    self.data_loader = torch_data.DataLoader(
        self.dataset, batch_size=5, shuffle=False
    )

    self.model = SimpleModel()
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(
        self.model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8
    )

    self.checkpoints = [copy.deepcopy(self.model)]

    for batch_x, batch_y in self.data_loader:
      self.optimizer.zero_grad()
      out = self.model(batch_x)
      loss = self.criterion(out, batch_y)
      loss.backward()
      self.optimizer.step()
      self.checkpoints.append(copy.deepcopy(self.model))

    self.base_model_artifacts = {
        'train_loader': self.data_loader,
        'checkpoints': self.checkpoints,
        'loss_fn': self.criterion,
        'hyperparams': {'lr': 0.01, 'beta_1': 0.9, 'beta_2': 0.999, 'eps': 1e-8},
    }

  def test_forward_update_with_dataloader(self):
    theta_diff = infl_sgd.forward_update(
        base_model=self.base_model_artifacts,
        ind=3,
        num_steps=4,
    )
    self.assertLen(theta_diff, len(list(self.model.parameters())))

  def test_recursive_update_with_dataloader(self):
    num_params = sum(p.numel() for p in self.model.parameters())
    u = torch.randn(num_params)
    theta_diff_dict = infl_sgd.recursive_update(
        base_model=self.base_model_artifacts,
        inds=3,
        query_vec=u,
        num_steps=4,
    )
    self.assertIn(3, theta_diff_dict)
    self.assertEqual(
        theta_diff_dict[3].shape, torch.Size([])
    )  # scalar dot product

  def test_forward_and_recursive_equivalence_no_query_vec(self):
    # Verify that forward_update and recursive_update return the same
    # theta_diff when query_vec is None.
    fwd_theta_diff = infl_sgd.forward_update(
        base_model=self.base_model_artifacts,
        ind=3,
        num_steps=4,
    )
    fwd_theta_diff_flat = torch.cat([p.flatten() for p in fwd_theta_diff])

    rec_theta_diff_dict = infl_sgd.recursive_update(
        base_model=self.base_model_artifacts,
        inds=3,
        query_vec=None,
        num_steps=4,
    )
    rec_theta_diff = rec_theta_diff_dict[3]

    torch.testing.assert_close(
        fwd_theta_diff_flat, rec_theta_diff, rtol=1e-4, atol=1e-4
    )

  def test_forward_and_recursive_equivalence_with_query_vec(self):
    # Verify equivalence with query_vec: dot(v, u) == w
    x_val = torch.randn(1, 4)
    y_val = torch.randint(0, 2, (1,))
    model = self.checkpoints[-1]
    loss_val = self.criterion(model(x_val), y_val)
    grads_val = torch.autograd.grad(loss_val, model.parameters())
    u = torch.cat([g.flatten() for g in grads_val])

    fwd_theta_diff = infl_sgd.forward_update(
        base_model=self.base_model_artifacts,
        ind=3,
        num_steps=4,
    )
    fwd_theta_diff_flat = torch.cat([p.flatten() for p in fwd_theta_diff])
    expected_dot = torch.dot(fwd_theta_diff_flat, u)

    rec_dot_dict = infl_sgd.recursive_update(
        base_model=self.base_model_artifacts,
        inds=3,
        query_vec=u,
        num_steps=4,
    )
    rec_dot = rec_dot_dict[3]

    torch.testing.assert_close(expected_dot, rec_dot, rtol=1e-4, atol=1e-4)

  def test_recursive_update_with_precomputed_gradients(self):
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    checkpoints = [model]
    gradients_lst = []

    for batch_x, batch_y in self.data_loader:
      out = model(batch_x)
      loss = self.criterion(out, batch_y)
      grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
      gradients_lst.append(grads)

      for p, g in zip(model.parameters(), grads):
        p.grad = g.detach().clone()
      optimizer.step()
      optimizer.zero_grad()
      checkpoints.append(model)

    num_params = sum(p.numel() for p in model.parameters())
    u = torch.randn(num_params)

    custom_artifacts = {
        'train_loader': self.data_loader,
        'checkpoints': checkpoints,
        'loss_fn': self.criterion,
        'hyperparams': {'lr': 0.01},
    }

    theta_diff_precomputed_dict = infl_sgd.recursive_update(
        base_model=custom_artifacts,
        inds=3,
        query_vec=u,
        num_steps=4,
        gradients_lst=gradients_lst,
    )

    theta_diff_recomputed_dict = infl_sgd.recursive_update(
        base_model=custom_artifacts,
        inds=3,
        query_vec=u,
        num_steps=4,
        gradients_lst=None,
    )

    torch.testing.assert_close(
        theta_diff_precomputed_dict[3],
        theta_diff_recomputed_dict[3],
        rtol=1e-4,
        atol=1e-4,
    )

  def test_recursive_update_list_ind(self):
    indices = [0, 1, 3]
    num_params = sum(p.numel() for p in self.model.parameters())
    u = torch.randn(num_params)
    res_dict = infl_sgd.recursive_update(
        base_model=self.base_model_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=2,
    )
    self.assertIsInstance(res_dict, dict)

    for idx in indices:
      single_res_dict = infl_sgd.recursive_update(
          base_model=self.base_model_artifacts,
          inds=idx,
          query_vec=u,
          num_steps=2,
      )
      torch.testing.assert_close(res_dict[idx], single_res_dict[idx])

  def test_tracin_with_query_vec(self):
    indices = [0, 1, 3]
    num_params = sum(p.numel() for p in self.model.parameters())
    u = torch.randn(num_params)
    res_dict = infl_sgd.tracin(
        base_model=self.base_model_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=4,
    )
    self.assertIsInstance(res_dict, dict)
    for idx in indices:
      self.assertEqual(res_dict[idx].shape, torch.Size([]))

  def test_tracin_requires_query_vec(self):
    with self.assertRaises(ValueError):
      infl_sgd.tracin(
          base_model=self.base_model_artifacts,
          inds=[0, 1],
          query_vec=None,
          num_steps=4,
      )

  def test_tracin_single_step_equivalence_with_recursive(self):
    # On single step (num_steps=1), Hessian propagation is not triggered,
    # so TracIn matches recursive_update.
    indices = [0, 1, 3]
    num_params = sum(p.numel() for p in self.model.parameters())
    u = torch.randn(num_params)
    tracin_res = infl_sgd.tracin(
        base_model=self.base_model_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=1,
    )
    rec_res = infl_sgd.recursive_update(
        base_model=self.base_model_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=1,
    )
    for idx in indices:
      torch.testing.assert_close(
          tracin_res[idx], rec_res[idx], rtol=1e-4, atol=1e-4
      )

  def test_tracin_forward_rolling_equivalence(self):
    indices = [0, 1, 3]
    num_params = sum(p.numel() for p in self.model.parameters())
    u = torch.randn(num_params)
    tracin_full = infl_sgd.tracin(
        base_model=self.base_model_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=3,
    )
    sparse_artifacts = {
        'train_loader': self.data_loader,
        'checkpoints': {0: self.checkpoints[0]},
        'loss_fn': self.criterion,
        'lr': 0.01,
    }
    tracin_sparse = infl_sgd.tracin(
        base_model=sparse_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=3,
    )
    for idx in indices:
      torch.testing.assert_close(
          tracin_sparse[idx], tracin_full[idx], rtol=1e-4, atol=1e-4
      )

  def test_tracin_periodic_and_first_last_checkpointing(self):
    indices = [0, 6, 12, 18]
    num_params = sum(p.numel() for p in self.model.parameters())
    u = torch.randn(num_params)
    tracin_full = infl_sgd.tracin(
        base_model=self.base_model_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=4,
    )

    # Periodic checkpointing: step 0 and step 2
    periodic_artifacts = {
        'train_loader': self.data_loader,
        'checkpoints': {0: self.checkpoints[0], 2: self.checkpoints[2]},
        'loss_fn': self.criterion,
        'lr': 0.01,
    }
    tracin_periodic = infl_sgd.tracin(
        base_model=periodic_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=4,
    )
    for idx in indices:
      torch.testing.assert_close(
          tracin_periodic[idx], tracin_full[idx], rtol=1e-4, atol=1e-4
      )

    # Extreme case: only first and last step
    first_last_artifacts = {
        'train_loader': self.data_loader,
        'checkpoints': {0: self.checkpoints[0], 4: self.checkpoints[4], -1: self.checkpoints[4]},
        'loss_fn': self.criterion,
        'lr': 0.01,
    }
    tracin_fl = infl_sgd.tracin(
        base_model=first_last_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=4,
    )
    for idx in indices:
      torch.testing.assert_close(
          tracin_fl[idx], tracin_full[idx], rtol=1e-4, atol=1e-4
      )

  def test_forward_update_batched_multi_index(self):
    indices = [0, 1, 3]
    m = len(indices)
    num_params = sum(p.numel() for p in self.model.parameters())

    batched_theta = infl_sgd.forward_update(
        base_model=self.base_model_artifacts,
        ind=indices,
        num_steps=4,
    )
    self.assertEqual(batched_theta.shape, (m, num_params))

    for k, idx in enumerate(indices):
      single_theta = infl_sgd.forward_update(
          base_model=self.base_model_artifacts,
          ind=idx,
          num_steps=4,
      )
      flat_single_theta = torch.cat([p.flatten() for p in single_theta])
      torch.testing.assert_close(
          batched_theta[k], flat_single_theta, rtol=1e-4, atol=1e-4
      )

  def test_forward_update_chunking(self):
    indices = [0, 1, 2, 3]

    expected_theta = infl_sgd.forward_update(
        base_model=self.base_model_artifacts,
        ind=indices,
        num_steps=4,
    )

    # Force chunk_size=1 so each sample is processed in its own chunk
    with mock.patch.object(infl_sgd.utils, 'get_dynamic_chunk_size', return_value=1):
      chunked_theta_1 = infl_sgd.forward_update(
          base_model=self.base_model_artifacts,
          ind=indices,
          num_steps=4,
      )
    torch.testing.assert_close(
        chunked_theta_1, expected_theta, rtol=1e-4, atol=1e-4
    )

    # Force chunk_size=2
    with mock.patch.object(infl_sgd.utils, 'get_dynamic_chunk_size', return_value=2):
      chunked_theta_2 = infl_sgd.forward_update(
          base_model=self.base_model_artifacts,
          ind=indices,
          num_steps=4,
      )
    torch.testing.assert_close(
        chunked_theta_2, expected_theta, rtol=1e-4, atol=1e-4
    )

  def test_attribution_base_model_integration(self):
    bm = base_model.AttributionBaseModel(
        model=SimpleModel(),
        loss_fn=nn.CrossEntropyLoss(),
        lr=0.01,
        seed=42,
    )
    bm.train_model(self.data_loader, num_epochs=1, checkpoint_freq=1)

    theta_diff = infl_sgd.forward_update(
        base_model=bm,
        ind=1,
        num_steps=2,
    )
    self.assertEqual(len(theta_diff), len(list(self.model.parameters())))

    query_vec = torch.randn(sum(p.numel() for p in self.model.parameters()))
    rec_res = infl_sgd.recursive_update(
        base_model=bm,
        inds=[0, 1],
        query_vec=query_vec,
        num_steps=2,
    )
    self.assertIn(0, rec_res)
    self.assertIn(1, rec_res)

    tracin_res = infl_sgd.tracin(
        base_model=bm,
        inds=[0, 1],
        query_vec=query_vec,
        num_steps=2,
    )
    self.assertIn(0, tracin_res)
    self.assertIn(1, tracin_res)


  def test_tracin_attribution_base_model_checkpointing_equivalence(self):
    """Tests that TracIn-SGD produces identical scores on AttributionBaseModel (trained with Adam) under full vs periodic vs first-and-last checkpointing."""
    indices = [0, 6, 12, 18]
    num_params = sum(p.numel() for p in self.model.parameters())
    u = torch.randn(num_params)

    # 1. Full checkpointing
    bm_full = base_model.AttributionBaseModel(
        model=copy.deepcopy(self.model),
        loss_fn=nn.CrossEntropyLoss(),
        lr=0.01,
        seed=42,
    )
    bm_full.train_model(self.data_loader, num_epochs=2, save_all_ckpts=True)

    # 2. Periodic checkpointing
    bm_periodic = base_model.AttributionBaseModel(
        model=copy.deepcopy(self.model),
        loss_fn=nn.CrossEntropyLoss(),
        lr=0.01,
        seed=42,
    )
    bm_periodic.train_model(self.data_loader, num_epochs=2, checkpoint_freq=2)

    # 3. Only first and last step
    bm_fl = base_model.AttributionBaseModel(
        model=copy.deepcopy(self.model),
        loss_fn=nn.CrossEntropyLoss(),
        lr=0.01,
        seed=42,
    )
    bm_fl.train_model(self.data_loader, num_epochs=2, checkpoint_freq=None)

    tracin_sgd_full = infl_sgd.tracin(base_model=bm_full, inds=indices, query_vec=u)
    tracin_sgd_periodic = infl_sgd.tracin(base_model=bm_periodic, inds=indices, query_vec=u)
    tracin_sgd_fl = infl_sgd.tracin(base_model=bm_fl, inds=indices, query_vec=u)

    for idx in indices:
      torch.testing.assert_close(
          tracin_sgd_periodic[idx], tracin_sgd_full[idx], rtol=1e-4, atol=1e-4
      )
      torch.testing.assert_close(
          tracin_sgd_fl[idx], tracin_sgd_full[idx], rtol=1e-4, atol=1e-4
      )


if __name__ == '__main__':
  absltest.main()
