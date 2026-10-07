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

"""Unit tests for infl_adam.py."""

import copy
import gc
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torch.utils import data as torch_data

from reversible_data_attribution import base_model
from reversible_data_attribution import infl_adam
from reversible_data_attribution import utils


class SimpleModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(4, 2)

  def forward(self, x):
    return self.linear(x)


class InflAdamTest(parameterized.TestCase):

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

    # Perform a few dummy training steps to collect checkpoints,
    # momentum, variance, and gradients
    self.checkpoints = []
    cur_m = [torch.zeros_like(p) for p in self.model.parameters()]
    cur_v = [torch.zeros_like(p) for p in self.model.parameters()]
    self.momentum_lst = [cur_m]
    self.variance_lst = [cur_v]
    self.gradients_lst = []

    for batch_x, batch_y in self.data_loader:
      model_t = copy.deepcopy(self.model)
      out_t = model_t(batch_x)
      loss_t = self.criterion(out_t, batch_y)
      grads_t = torch.autograd.grad(
          loss_t, model_t.parameters(), create_graph=True
      )
      self.gradients_lst.append(grads_t)
      self.checkpoints.append(model_t)

      self.optimizer.zero_grad()
      out = self.model(batch_x)
      loss = self.criterion(out, batch_y)
      loss.backward()
      self.optimizer.step()

      cur_m = [
          self.optimizer.state[p]['exp_avg'].clone()
          for p in self.model.parameters()
      ]
      cur_v = [
          self.optimizer.state[p]['exp_avg_sq'].clone()
          for p in self.model.parameters()
      ]

      self.momentum_lst.append(cur_m)
      self.variance_lst.append(cur_v)

    self.checkpoints.append(copy.deepcopy(self.model))
    self.base_model_artifacts = {
        'train_loader': self.data_loader,
        'checkpoints': self.checkpoints,
        'momentum_lst': self.momentum_lst,
        'variance_lst': self.variance_lst,
        'loss_fn': self.criterion,
        'hyperparams': {
            'lr': 0.01,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'eps': 1e-8,
        },
    }

  def test_forward_update_with_dataloader(self):
    theta_diff, _, _ = infl_adam.forward_update(
        base_model=self.base_model_artifacts,
        ind=3,
        num_steps=4,
        remove_dv=True,
    )
    self.assertEqual(len(theta_diff), len(list(self.model.parameters())))

  # --- 2x2 Matrix Tests: (no dv, dv) x (no query vec, query vec) ---

  def test_recursive_nodv_no_query_vec(self):
    # 1. No DV, No Query Vector
    indices = [0, 1, 3]
    rec_dict = infl_adam.recursive_update_nodv(
        base_model=self.base_model_artifacts,
        inds=indices,
        query_vec=None,
        num_steps=4,
        gradients_lst=self.gradients_lst,
    )

    for idx in indices:
      fwd_theta_diff, _, _ = infl_adam.forward_update(
          base_model=self.base_model_artifacts,
          ind=idx,
          num_steps=4,
          remove_dv=True,
      )
      fwd_theta_diff_flat = torch.cat([p.flatten() for p in fwd_theta_diff])
      torch.testing.assert_close(
          rec_dict[idx], fwd_theta_diff_flat, rtol=1e-4, atol=1e-4
      )

  def test_recursive_nodv_with_query_vec(self):
    # 2. No DV, With Query Vector
    indices = [0, 1, 3]
    x_val = torch.randn(1, 4)
    y_val = torch.randint(0, 2, (1,))
    model = self.checkpoints[-1]
    loss_val = self.criterion(model(x_val), y_val)
    grads_val = torch.autograd.grad(loss_val, model.parameters())
    u = torch.cat([g.flatten() for g in grads_val])

    rec_dict = infl_adam.recursive_update_nodv(
        base_model=self.base_model_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=4,
        gradients_lst=self.gradients_lst,
    )

    for idx in indices:
      fwd_theta_diff, _, _ = infl_adam.forward_update(
          base_model=self.base_model_artifacts,
          ind=idx,
          num_steps=4,
          remove_dv=True,
      )
      fwd_theta_diff_flat = torch.cat([p.flatten() for p in fwd_theta_diff])
      expected_dot = torch.dot(fwd_theta_diff_flat, u)
      torch.testing.assert_close(
          rec_dict[idx], expected_dot, rtol=1e-4, atol=1e-4
      )

  # @absltest.skip('Recursive update with DV is work in progress')
  def test_recursive_dv_no_query_vec(self):
    # 3. With DV, No Query Vector
    indices = [0, 1, 3]
    rec_dict = infl_adam.recursive_update(
        base_model=self.base_model_artifacts,
        inds=indices,
        query_vec=None,
        num_steps=4,
        gradients_lst=self.gradients_lst,
    )

    for idx in indices:
      fwd_theta_diff, _, _ = infl_adam.forward_update(
          base_model=self.base_model_artifacts,
          ind=idx,
          num_steps=4,
          remove_dv=False,
      )
      fwd_theta_diff_flat = torch.cat([p.flatten() for p in fwd_theta_diff])
      torch.testing.assert_close(
          rec_dict[idx], fwd_theta_diff_flat, rtol=1e-3, atol=1e-3
      )

  # @absltest.skip('Recursive update with DV is work in progress')
  def test_recursive_dv_with_query_vec(self):
    # 4. With DV, With Query Vector
    indices = [0, 1, 3]
    x_val = torch.randn(1, 4)
    y_val = torch.randint(0, 2, (1,))
    model = self.checkpoints[-1]
    loss_val = self.criterion(model(x_val), y_val)
    grads_val = torch.autograd.grad(loss_val, model.parameters())
    u = torch.cat([g.flatten() for g in grads_val])

    rec_dict = infl_adam.recursive_update(
        base_model=self.base_model_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=4,
        gradients_lst=self.gradients_lst,
    )

    for idx in indices:
      fwd_theta_diff, _, _ = infl_adam.forward_update(
          base_model=self.base_model_artifacts,
          ind=idx,
          num_steps=4,
          remove_dv=False,
      )
      fwd_theta_diff_flat = torch.cat([p.flatten() for p in fwd_theta_diff])
      expected_dot = torch.dot(fwd_theta_diff_flat, u)
      torch.testing.assert_close(
          rec_dict[idx], expected_dot, rtol=1e-3, atol=1e-3
      )

  def test_recursive_dv_w_agreement(self):
    indices = [0, 1, 3]
    x_val = torch.randn(1, 4)
    y_val = torch.randint(0, 2, (1,))
    model = self.checkpoints[-1]
    loss_val = self.criterion(model(x_val), y_val)
    grads_val = torch.autograd.grad(loss_val, model.parameters())
    u = torch.cat([g.flatten() for g in grads_val])
    rec_dict_query = infl_adam.recursive_update(
        base_model=self.base_model_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=4,
        gradients_lst=self.gradients_lst,
    )
    rec_dict_noquery = infl_adam.recursive_update(
        base_model=self.base_model_artifacts,
        inds=indices,
        query_vec=None,
        num_steps=4,
        gradients_lst=self.gradients_lst,
    )
    for ind in indices:
      torch.testing.assert_close(
          rec_dict_query[ind],
          torch.dot(u, rec_dict_noquery[ind]),
          rtol=1e-3,
          atol=1e-3,
      )

  @parameterized.parameters(
      ('exact', 'first_order'),
      ('exact', 'second_order'),
      ('exact', 'exact'),
  )
  def test_forward_update_options(self, dv_option, theta_option):
    theta_diff, _, _ = infl_adam.forward_update(
        base_model=self.base_model_artifacts,
        ind=3,
        num_steps=4,
        remove_dv=False,
        dv_option=dv_option,
        theta_option=theta_option,
    )
    self.assertEqual(len(theta_diff), len(list(self.model.parameters())))
    for p in theta_diff:
      self.assertFalse(torch.isnan(p).any())

  def test_forward_update_0th_timestamp_only(self):
    full_res = infl_adam.forward_update(
        base_model=self.base_model_artifacts,
        ind=3,
        num_steps=4,
        remove_dv=True,
    )
    single_artifacts = dict(self.base_model_artifacts)
    single_artifacts['checkpoints'] = [self.checkpoints[0]]
    single_artifacts['momentum_lst'] = [self.momentum_lst[0]]
    single_artifacts['variance_lst'] = [self.variance_lst[0]]
    single_res = infl_adam.forward_update(
        base_model=single_artifacts,
        ind=3,
        num_steps=4,
        remove_dv=True,
    )
    dict_artifacts = dict(self.base_model_artifacts)
    dict_artifacts['checkpoints'] = {0: self.checkpoints[0]}
    dict_artifacts['momentum_lst'] = {0: self.momentum_lst[0]}
    dict_artifacts['variance_lst'] = {0: self.variance_lst[0]}
    dict_res = infl_adam.forward_update(
        base_model=dict_artifacts,
        ind=3,
        num_steps=4,
        remove_dv=True,
    )
    for p_full, p_single in zip(full_res[0], single_res[0]):
      torch.testing.assert_close(p_full, p_single, rtol=1e-4, atol=1e-4)
    for p_full, p_dict in zip(full_res[0], dict_res[0]):
      torch.testing.assert_close(p_full, p_dict, rtol=1e-4, atol=1e-4)

  def test_recursive_update_nodv_sth_timestamp_only(self):
    num_steps = 4
    full_artifacts = dict(self.base_model_artifacts)
    full_artifacts['checkpoints'] = self.checkpoints[: num_steps + 1]
    full_artifacts['momentum_lst'] = self.momentum_lst[: num_steps + 1]
    full_artifacts['variance_lst'] = self.variance_lst[: num_steps + 1]
    full_res = infl_adam.recursive_update_nodv(
        base_model=full_artifacts,
        inds=[0, 1, 3],
        num_steps=num_steps,
    )
    single_artifacts = dict(self.base_model_artifacts)
    single_artifacts['checkpoints'] = [self.checkpoints[num_steps]]
    single_artifacts['momentum_lst'] = [self.momentum_lst[num_steps]]
    single_artifacts['variance_lst'] = [self.variance_lst[num_steps]]
    single_res = infl_adam.recursive_update_nodv(
        base_model=single_artifacts,
        inds=[0, 1, 3],
        num_steps=num_steps,
    )
    dict_artifacts = dict(self.base_model_artifacts)
    dict_artifacts['checkpoints'] = {-1: self.checkpoints[num_steps]}
    dict_artifacts['momentum_lst'] = {-1: self.momentum_lst[num_steps]}
    dict_artifacts['variance_lst'] = {-1: self.variance_lst[num_steps]}
    dict_res = infl_adam.recursive_update_nodv(
        base_model=dict_artifacts,
        inds=[0, 1, 3],
        num_steps=num_steps,
    )
    for idx in [0, 1, 3]:
      torch.testing.assert_close(
          full_res[idx], single_res[idx], rtol=1e-4, atol=1e-4
      )
      torch.testing.assert_close(
          full_res[idx], dict_res[idx], rtol=1e-4, atol=1e-4
      )

  def test_recursive_update_nodv_with_partial_checkpoints(self):
    num_steps = 4
    full_artifacts = dict(self.base_model_artifacts)
    full_artifacts['checkpoints'] = self.checkpoints[: num_steps + 1]
    full_artifacts['momentum_lst'] = self.momentum_lst[: num_steps + 1]
    full_artifacts['variance_lst'] = self.variance_lst[: num_steps + 1]
    full_res = infl_adam.recursive_update_nodv(
        base_model=full_artifacts,
        inds=[0, 1, 3],
        num_steps=num_steps,
    )

    partial_artifacts = dict(self.base_model_artifacts)
    # Provide metadata at steps 0, 2, 4 (missing steps 1 and 3)
    partial_artifacts['checkpoints'] = {
        0: self.checkpoints[0],
        2: self.checkpoints[2],
        4: self.checkpoints[4],
    }
    partial_artifacts['momentum_lst'] = {
        0: self.momentum_lst[0],
        2: self.momentum_lst[2],
        4: self.momentum_lst[4],
    }
    partial_artifacts['variance_lst'] = {
        0: self.variance_lst[0],
        2: self.variance_lst[2],
        4: self.variance_lst[4],
    }
    partial_res = infl_adam.recursive_update_nodv(
        base_model=partial_artifacts,
        inds=[0, 1, 3],
        num_steps=num_steps,
    )
    for idx in [0, 1, 3]:
      torch.testing.assert_close(
          full_res[idx], partial_res[idx], rtol=1e-4, atol=1e-4
      )

  def test_recursive_update_sth_timestamp_only(self):
    num_steps = 4
    full_artifacts = dict(self.base_model_artifacts)
    full_artifacts['checkpoints'] = self.checkpoints[: num_steps + 1]
    full_artifacts['momentum_lst'] = self.momentum_lst[: num_steps + 1]
    full_artifacts['variance_lst'] = self.variance_lst[: num_steps + 1]
    full_res = infl_adam.recursive_update(
        base_model=full_artifacts,
        inds=[0, 1, 3],
        num_steps=num_steps,
    )
    single_artifacts = dict(self.base_model_artifacts)
    single_artifacts['checkpoints'] = [self.checkpoints[num_steps]]
    single_artifacts['momentum_lst'] = [self.momentum_lst[num_steps]]
    single_artifacts['variance_lst'] = [self.variance_lst[num_steps]]
    single_res = infl_adam.recursive_update(
        base_model=single_artifacts,
        inds=[0, 1, 3],
        num_steps=num_steps,
    )
    dict_artifacts = dict(self.base_model_artifacts)
    dict_artifacts['checkpoints'] = {-1: self.checkpoints[num_steps]}
    dict_artifacts['momentum_lst'] = {-1: self.momentum_lst[num_steps]}
    dict_artifacts['variance_lst'] = {-1: self.variance_lst[num_steps]}
    dict_res = infl_adam.recursive_update(
        base_model=dict_artifacts,
        inds=[0, 1, 3],
        num_steps=num_steps,
    )
    for idx in [0, 1, 3]:
      torch.testing.assert_close(
          full_res[idx], single_res[idx], rtol=1e-4, atol=1e-4
      )
      torch.testing.assert_close(
          full_res[idx], dict_res[idx], rtol=1e-4, atol=1e-4
      )

  def test_recursive_update_with_partial_checkpoints(self):
    num_steps = 4
    full_artifacts = dict(self.base_model_artifacts)
    full_artifacts['checkpoints'] = self.checkpoints[: num_steps + 1]
    full_artifacts['momentum_lst'] = self.momentum_lst[: num_steps + 1]
    full_artifacts['variance_lst'] = self.variance_lst[: num_steps + 1]
    full_res = infl_adam.recursive_update(
        base_model=full_artifacts,
        inds=[0, 1, 3],
        num_steps=num_steps,
    )

    partial_artifacts = dict(self.base_model_artifacts)
    # Provide metadata at steps 0, 2, 4 (missing steps 1 and 3)
    partial_artifacts['checkpoints'] = {
        0: self.checkpoints[0],
        2: self.checkpoints[2],
        4: self.checkpoints[4],
    }
    partial_artifacts['momentum_lst'] = {
        0: self.momentum_lst[0],
        2: self.momentum_lst[2],
        4: self.momentum_lst[4],
    }
    partial_artifacts['variance_lst'] = {
        0: self.variance_lst[0],
        2: self.variance_lst[2],
        4: self.variance_lst[4],
    }
    partial_res = infl_adam.recursive_update(
        base_model=partial_artifacts,
        inds=[0, 1, 3],
        num_steps=num_steps,
    )
    for idx in [0, 1, 3]:
      torch.testing.assert_close(
          full_res[idx], partial_res[idx], rtol=1e-4, atol=1e-4
      )

  def test_compute_metadata_adam_forward_and_backward(self):
    batches = utils.prepare_batches(self.data_loader)
    checkpoints = self.checkpoints
    momentum_lst = self.momentum_lst
    variance_lst = self.variance_lst

    curr_model = copy.deepcopy(checkpoints[0])
    curr_m = [m.clone() for m in momentum_lst[0]]
    curr_v = [v.clone() for v in variance_lst[0]]

    for step_idx, (x_b, y_b, _) in enumerate(batches, start=1):
      theta_next, m_next, v_next = utils.compute_metadata_adam_forward(
          x_batch=x_b,
          y_batch=y_b,
          model=curr_model,
          loss_fn=self.criterion,
          momentum=curr_m,
          variance=curr_v,
          lr=0.01,
          beta_1=0.9,
          beta_2=0.999,
          eps=1e-8,
          step=step_idx,
      )
      for p_calc, p_saved in zip(
          theta_next, checkpoints[step_idx].parameters()
      ):
        torch.testing.assert_close(p_calc, p_saved, rtol=1e-4, atol=1e-4)
      for m_calc, m_saved in zip(m_next, momentum_lst[step_idx]):
        torch.testing.assert_close(m_calc, m_saved, rtol=1e-4, atol=1e-4)
      for v_calc, v_saved in zip(v_next, variance_lst[step_idx]):
        torch.testing.assert_close(v_calc, v_saved, rtol=1e-4, atol=1e-4)

      curr_model = copy.deepcopy(curr_model)
      utils.update_model_parameters(curr_model, theta_next)
      curr_m = m_next
      curr_v = v_next

    total_steps = len(batches)
    curr_model = copy.deepcopy(checkpoints[total_steps])
    curr_m = [m.clone() for m in momentum_lst[total_steps]]
    curr_v = [v.clone() for v in variance_lst[total_steps]]

    for t in range(total_steps - 1, -1, -1):
      step_idx = t + 1
      x_b, y_b, _ = batches[t]
      model_prev, m_prev, v_prev = utils.compute_metadata_adam_backward(
          x_batch=x_b,
          y_batch=y_b,
          model=curr_model,
          loss_fn=self.criterion,
          momentum=curr_m,
          variance=curr_v,
          lr=0.01,
          beta_1=0.9,
          beta_2=0.999,
          eps=1e-8,
          step=step_idx,
      )
      for p_calc, p_saved in zip(
          model_prev.parameters(), checkpoints[t].parameters()
      ):
        torch.testing.assert_close(p_calc, p_saved, rtol=1e-4, atol=1e-4)
      for m_calc, m_saved in zip(m_prev, momentum_lst[t]):
        torch.testing.assert_close(m_calc, m_saved, rtol=1e-4, atol=1e-4)
      for v_calc, v_saved in zip(v_prev, variance_lst[t]):
        torch.testing.assert_close(v_calc, v_saved, rtol=1e-4, atol=1e-4)

      curr_model = model_prev
      curr_m = m_prev
      curr_v = v_prev

  def test_forward_update_no_memory_accumulation(self):
    """Tests that memory and tensor count do not accumulate with increased step count in forward_update."""
    single_artifacts = dict(self.base_model_artifacts)
    single_artifacts['checkpoints'] = [self.checkpoints[0]]
    single_artifacts['momentum_lst'] = [self.momentum_lst[0]]
    single_artifacts['variance_lst'] = [self.variance_lst[0]]

    # 1. Warmup run (num_steps=2)
    _ = infl_adam.forward_update(
        base_model=single_artifacts,
        ind=3,
        num_steps=2,
        remove_dv=True,
    )
    gc.collect()

    # 2. Medium run (num_steps=20)
    _ = infl_adam.forward_update(
        base_model=single_artifacts,
        ind=3,
        num_steps=20,
        remove_dv=True,
    )
    gc.collect()
    tensors_medium = [
        obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)
    ]

    # 3. Long run (num_steps=100)
    theta_diff, momentum_diff, variance_diff = infl_adam.forward_update(
        base_model=single_artifacts,
        ind=3,
        num_steps=100,
        remove_dv=True,
    )
    gc.collect()
    tensors_long = [
        obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)
    ]

    # The 6 returned tensors stored in local variables account for 6 new tensors.
    # Comparing (len(tensors_long) - len(tensors_medium)) minus the 6 returned tensors:
    returned_tensors_count = (
        len(theta_diff) + len(momentum_diff) + len(variance_diff)
    )
    loop_leak_count = (
        len(tensors_long) - len(tensors_medium) - returned_tensors_count
    )

    self.assertLessEqual(
        loop_leak_count,
        0,
        'Memory accumulation detected in loop! Tensor count grew by'
        f' {loop_leak_count} across 80 additional loop iterations.',
    )

    # 3. Assert computational graphs are detached (requires_grad is False, grad_fn is None)
    for tensor_list in [theta_diff, momentum_diff, variance_diff]:
      for t in tensor_list:
        self.assertFalse(t.requires_grad)
        self.assertIsNone(t.grad_fn)

  def test_tracin_with_query_vec(self):
    indices = [0, 1, 3]
    num_params = sum(p.numel() for p in self.model.parameters())
    u = torch.randn(num_params)
    res_dict = infl_adam.tracin(
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
      infl_adam.tracin(
          base_model=self.base_model_artifacts,
          inds=[0, 1],
          query_vec=None,
          num_steps=4,
      )

  def test_tracin_cosine_similarity(self):
    indices = [0, 1, 3]
    num_params = sum(p.numel() for p in self.model.parameters())
    u = torch.randn(num_params)
    tracin_res = infl_adam.tracin(
        base_model=self.base_model_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=2,
    )
    for idx in indices:
      self.assertTrue(-2.0 <= tracin_res[idx].item() <= 2.0)

  def test_tracin_forward_rolling_equivalence(self):
    indices = [0, 1, 3]
    num_params = sum(p.numel() for p in self.model.parameters())
    u = torch.randn(num_params)
    tracin_full = infl_adam.tracin(
        base_model=self.base_model_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=3,
    )
    sparse_artifacts = {
        'train_loader': self.data_loader,
        'checkpoints': {0: self.checkpoints[0]},
        'loss_fn': self.criterion,
        'hyperparams': self.base_model_artifacts['hyperparams'],
    }
    tracin_sparse = infl_adam.tracin(
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
    indices = [0, 1, 3]
    num_params = sum(p.numel() for p in self.model.parameters())
    u = torch.randn(num_params)
    tracin_full = infl_adam.tracin(
        base_model=self.base_model_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=4,
    )

    # Periodic checkpointing: step 0 and step 2
    periodic_artifacts = {
        'train_loader': self.data_loader,
        'checkpoints': {0: self.checkpoints[0], 2: self.checkpoints[2]},
        'momentum_lst': {0: self.momentum_lst[0], 2: self.momentum_lst[2]},
        'variance_lst': {0: self.variance_lst[0], 2: self.variance_lst[2]},
        'loss_fn': self.criterion,
        'hyperparams': self.base_model_artifacts['hyperparams'],
    }
    tracin_periodic = infl_adam.tracin(
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
        'momentum_lst': {0: self.momentum_lst[0], 4: self.momentum_lst[4], -1: self.momentum_lst[4]},
        'variance_lst': {0: self.variance_lst[0], 4: self.variance_lst[4], -1: self.variance_lst[4]},
        'loss_fn': self.criterion,
        'hyperparams': self.base_model_artifacts['hyperparams'],
    }
    tracin_fl = infl_adam.tracin(
        base_model=first_last_artifacts,
        inds=indices,
        query_vec=u,
        num_steps=4,
    )
    for idx in indices:
      torch.testing.assert_close(
          tracin_fl[idx], tracin_full[idx], rtol=1e-4, atol=1e-4
      )

  def test_tracin_attribution_base_model_checkpointing_equivalence(self):
    """Tests that TracIn-Adam produces identical scores with AttributionBaseModel under full vs periodic vs first-and-last checkpointing."""
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

    tracin_adam_full = infl_adam.tracin(base_model=bm_full, inds=indices, query_vec=u)
    tracin_adam_periodic = infl_adam.tracin(base_model=bm_periodic, inds=indices, query_vec=u)
    tracin_adam_fl = infl_adam.tracin(base_model=bm_fl, inds=indices, query_vec=u)

    for idx in indices:
      torch.testing.assert_close(
          tracin_adam_periodic[idx], tracin_adam_full[idx], rtol=1e-4, atol=1e-4
      )
      torch.testing.assert_close(
          tracin_adam_fl[idx], tracin_adam_full[idx], rtol=1e-4, atol=1e-4
      )

  def test_recursive_update_no_memory_accumulation(self):
    """Tests that memory and tensor count do not accumulate with increased step count in recursive_update."""
    num_params = sum(p.numel() for p in self.model.parameters())
    u = torch.randn(num_params)
    indices = [0, 1, 3]

    artifacts = dict(self.base_model_artifacts)
    artifacts['checkpoints'] = [self.checkpoints[-1]] * 101
    artifacts['momentum_lst'] = [self.momentum_lst[-1]] * 101
    artifacts['variance_lst'] = [self.variance_lst[-1]] * 101

    # 1. Warmup run (num_steps=2)
    _ = infl_adam.recursive_update(
        base_model=artifacts,
        inds=indices,
        query_vec=u,
        num_steps=2,
    )
    gc.collect()

    # 2. Medium run (num_steps=20)
    _ = infl_adam.recursive_update(
        base_model=artifacts,
        inds=indices,
        query_vec=u,
        num_steps=20,
    )
    gc.collect()
    tensors_medium = [
        obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)
    ]

    # 3. Long run (num_steps=100)
    res_dict = infl_adam.recursive_update(
        base_model=artifacts,
        inds=indices,
        query_vec=u,
        num_steps=100,
    )
    gc.collect()
    tensors_long = [
        obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)
    ]

    returned_tensors_count = len(res_dict)
    loop_leak_count = (
        len(tensors_long) - len(tensors_medium) - returned_tensors_count
    )

    self.assertLessEqual(
        loop_leak_count,
        0,
        'Memory accumulation detected in recursive_update loop! Tensor count'
        f' grew by {loop_leak_count} across 80 additional loop iterations.',
    )

  def test_forward_update_random_mask_all_true(self):
    """When random mask is all True, output matches unmasked forward_update."""
    mask_all_true = [
        torch.ones_like(p, dtype=torch.bool) for p in self.model.parameters()
    ]
    unmasked_res, _, _ = infl_adam.forward_update(
        base_model=self.base_model_artifacts,
        ind=3,
        num_steps=4,
        remove_dv=False,
        dv_option='exact',
        theta_option='exact',
    )
    masked_res, _, _ = infl_adam.forward_update(
        base_model=self.base_model_artifacts,
        ind=3,
        num_steps=4,
        remove_dv=False,
        dv_option='exact',
        theta_option='exact',
        random_mask=mask_all_true,
    )
    unmasked_flat = torch.cat([p.flatten() for p in unmasked_res])
    masked_flat = torch.cat([p.flatten() for p in masked_res])
    torch.testing.assert_close(masked_flat, unmasked_flat, rtol=1e-4, atol=1e-4)

  @parameterized.parameters(
      ('first_order', 'first_order'),
      ('exact', 'first_order'),
      ('exact', 'second_order'),
      ('exact', 'exact'),
  )
  def test_forward_update_random_mask_bernoulli(self, dv_option, theta_option):
    """Forward update runs smoothly with Bernoulli random mask and outputs valid masked tensors."""
    mask = utils.generate_random_mask(self.model, prob=0.5, seed=42)
    theta_diff, momentum_diff, variance_diff = infl_adam.forward_update(
        base_model=self.base_model_artifacts,
        ind=2,
        num_steps=4,
        remove_dv=False,
        dv_option=dv_option,
        theta_option=theta_option,
        random_mask=mask,
    )
    self.assertEqual(len(theta_diff), len(mask))
    for t_diff, m_diff, v_diff, m in zip(
        theta_diff, momentum_diff, variance_diff, mask
    ):
      expected_len = int(m.sum().item())
      self.assertEqual(t_diff.numel(), expected_len)
      self.assertEqual(m_diff.numel(), expected_len)
      self.assertEqual(v_diff.numel(), expected_len)
      self.assertFalse(torch.isnan(t_diff).any())
      self.assertFalse(t_diff.requires_grad)

  def test_get_grad_diff_random_mask_pearlmutter_vs_explicit(self):
    """get_grad_diff gives matching results for Pearlmutter vs explicit Hessian with random mask."""
    batches = utils.prepare_batches(self.data_loader)
    x_b, y_b, ind_b = utils.get_batch_and_ind(batches, 0, 1)
    mask = utils.generate_random_mask(self.model, prob=0.6, seed=123)
    theta_diff = [torch.randn(int(m.sum().item())) for m in mask]

    grads_p, diff_p = utils.get_grad_diff(
        x_b,
        y_b,
        self.model,
        self.criterion,
        theta_diff,
        ind_b,
        use_pearlmutter=True,
        random_mask=mask,
    )
    grads_e, diff_e = utils.get_grad_diff(
        x_b,
        y_b,
        self.model,
        self.criterion,
        theta_diff,
        ind_b,
        use_pearlmutter=False,
        random_mask=mask,
    )

    for gp, ge in zip(grads_p, grads_e):
      torch.testing.assert_close(gp, ge, rtol=1e-4, atol=1e-4)
    for dp, de in zip(diff_p, diff_e):
      torch.testing.assert_close(dp, de, rtol=1e-4, atol=1e-4)

  def test_forward_update_random_mask_on_the_fly_metadata(self):
    """Forward update with random mask works with on-the-fly metadata."""
    mask = utils.generate_random_mask(self.model, prob=0.5, seed=42)
    single_artifacts = dict(self.base_model_artifacts)
    single_artifacts['checkpoints'] = [self.checkpoints[0]]
    single_artifacts['momentum_lst'] = [self.momentum_lst[0]]
    single_artifacts['variance_lst'] = [self.variance_lst[0]]
    theta_diff, _, _ = infl_adam.forward_update(
        base_model=single_artifacts,
        ind=1,
        num_steps=4,
        remove_dv=False,
        dv_option='exact',
        theta_option='exact',
        random_mask=mask,
    )
    for t_diff, m in zip(theta_diff, mask):
      self.assertEqual(t_diff.numel(), int(m.sum().item()))
      self.assertFalse(torch.isnan(t_diff).any())

  def test_forward_update_batched_multi_index(self):
    """Batched forward_update matches individual single-index forward_update calls."""
    indices = [0, 1, 3]
    m = len(indices)
    num_params = sum(p.numel() for p in self.model.parameters())

    # 1. Test first-order with remove_dv=True
    batched_theta, batched_m, batched_v = infl_adam.forward_update(
        base_model=self.base_model_artifacts,
        ind=indices,
        num_steps=4,
        remove_dv=True,
    )
    self.assertEqual(batched_theta.shape, (m, num_params))
    self.assertEqual(batched_m.shape, (m, num_params))
    self.assertEqual(batched_v.shape, (m, num_params))

    for k, idx in enumerate(indices):
      single_theta, single_m, single_v = infl_adam.forward_update(
          base_model=self.base_model_artifacts,
          ind=idx,
          num_steps=4,
          remove_dv=True,
      )
      flat_single_theta = torch.cat([p.flatten() for p in single_theta])
      flat_single_m = torch.cat([p.flatten() for p in single_m])
      flat_single_v = torch.cat([p.flatten() for p in single_v])
      torch.testing.assert_close(
          batched_theta[k], flat_single_theta, rtol=1e-4, atol=1e-4
      )
      torch.testing.assert_close(
          batched_m[k], flat_single_m, rtol=1e-4, atol=1e-4
      )
      torch.testing.assert_close(
          batched_v[k], flat_single_v, rtol=1e-4, atol=1e-4
      )

    # 2. Test exact options with remove_dv=False and random mask
    mask = utils.generate_random_mask(self.model, prob=0.6, seed=123)
    mask_flat = torch.cat([r.flatten() for r in mask])
    p_dim = int(mask_flat.sum().item())

    single_artifacts = dict(self.base_model_artifacts)
    single_artifacts['checkpoints'] = [self.checkpoints[0]]
    single_artifacts['momentum_lst'] = [self.momentum_lst[0]]
    single_artifacts['variance_lst'] = [self.variance_lst[0]]

    batched_m_theta, _, _ = infl_adam.forward_update(
        base_model=single_artifacts,
        ind=indices,
        num_steps=4,
        remove_dv=False,
        dv_option='exact',
        theta_option='exact',
        random_mask=mask,
    )
    self.assertEqual(batched_m_theta.shape, (m, p_dim))

    for k, idx in enumerate(indices):
      single_m_theta, _, _ = infl_adam.forward_update(
          base_model=single_artifacts,
          ind=idx,
          num_steps=4,
          remove_dv=False,
          dv_option='exact',
          theta_option='exact',
          random_mask=mask,
      )
      flat_single_m_theta = torch.cat([p.flatten() for p in single_m_theta])
      torch.testing.assert_close(
          batched_m_theta[k], flat_single_m_theta, rtol=1e-4, atol=1e-4
      )

  def test_forward_update_chunking(self):
    indices = [0, 1, 2, 3]

    expected_theta, expected_m, expected_v = infl_adam.forward_update(
        base_model=self.base_model_artifacts,
        ind=indices,
        num_steps=4,
    )

    # Force chunk_size=1 so each sample is processed in its own chunk
    with mock.patch.object(
        infl_adam.utils, 'get_dynamic_chunk_size', return_value=1
    ):
      chunked_theta_1, chunked_m_1, chunked_v_1 = infl_adam.forward_update(
          base_model=self.base_model_artifacts,
          ind=indices,
          num_steps=4,
      )
    torch.testing.assert_close(
        chunked_theta_1, expected_theta, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(chunked_m_1, expected_m, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(chunked_v_1, expected_v, rtol=1e-4, atol=1e-4)

    # Force chunk_size=2
    with mock.patch.object(
        infl_adam.utils, 'get_dynamic_chunk_size', return_value=2
    ):
      chunked_theta_2, chunked_m_2, chunked_v_2 = infl_adam.forward_update(
          base_model=self.base_model_artifacts,
          ind=indices,
          num_steps=4,
      )
    torch.testing.assert_close(
        chunked_theta_2, expected_theta, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(chunked_m_2, expected_m, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(chunked_v_2, expected_v, rtol=1e-4, atol=1e-4)

  def test_attribution_base_model_integration(self):
    bm = base_model.AttributionBaseModel(
        model=SimpleModel(),
        loss_fn=nn.CrossEntropyLoss(),
        lr=0.01,
        beta_1=0.9,
        beta_2=0.999,
        eps=1e-8,
        seed=42,
    )
    bm.train_model(self.data_loader, num_epochs=1, checkpoint_freq=1)

    theta_diff, _, _ = infl_adam.forward_update(
        base_model=bm,
        ind=1,
        num_steps=2,
    )
    self.assertEqual(len(theta_diff), len(list(self.model.parameters())))

    query_vec = torch.randn(sum(p.numel() for p in self.model.parameters()))
    rec_res = infl_adam.recursive_update(
        base_model=bm,
        inds=[0, 1],
        query_vec=query_vec,
        num_steps=2,
    )
    self.assertIn(0, rec_res)
    self.assertIn(1, rec_res)

    rec_nodv_res = infl_adam.recursive_update_nodv(
        base_model=bm,
        inds=[0, 1],
        query_vec=query_vec,
        num_steps=2,
    )
    self.assertIn(0, rec_nodv_res)
    self.assertIn(1, rec_nodv_res)

    tracin_res = infl_adam.tracin(
        base_model=bm,
        inds=[0, 1],
        query_vec=query_vec,
        num_steps=2,
    )
    self.assertIn(0, tracin_res)
    self.assertIn(1, tracin_res)


if __name__ == '__main__':
  absltest.main()
