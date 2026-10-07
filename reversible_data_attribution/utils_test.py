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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
from torch import nn
from torch.utils import data as torch_data

from reversible_data_attribution import models
from reversible_data_attribution import utils


class SimpleModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(4, 2)

  def forward(self, x):
    return self.linear(x)


class UtilsTest(parameterized.TestCase):

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

  def test_prepare_batches_tuple(self):
    batches = utils.prepare_batches(self.data_loader)
    self.assertEqual(len(batches), 4)
    self.assertEqual(batches[0][0].shape, (5, 4))
    self.assertEqual(batches[0][1].shape, (5,))
    self.assertIsNone(batches[0][2])

  def test_prepare_batches_dict(self):
    dict_batches = [
        {
            'image': self.x_train[:5],
            'label': self.y_train[:5],
            'index': torch.arange(0, 5),
        }
    ]
    batches = utils.prepare_batches(dict_batches)
    self.assertEqual(len(batches), 1)
    self.assertTrue(torch.equal(batches[0][2], torch.arange(0, 5)))

  def test_get_batch_and_ind_with_indices(self):
    batches = utils.prepare_batches([
        {
            'image': self.x_train[:5],
            'label': self.y_train[:5],
            'index': torch.tensor([10, 11, 12, 13, 14]),
        }
    ])
    x_batch, y_batch, ind_batch = utils.get_batch_and_ind(batches, 0, ind=12)
    self.assertEqual(ind_batch, 2)

    _, _, ind_batch_missing = utils.get_batch_and_ind(batches, 0, ind=99)
    self.assertIsNone(ind_batch_missing)

  def test_get_batch_and_ind_without_indices(self):
    batches = utils.prepare_batches(self.data_loader)
    # batch 1 covers global indices 5..9
    _, _, ind_batch = utils.get_batch_and_ind(batches, 1, ind=7)
    self.assertEqual(ind_batch, 2)

    _, _, ind_batch_out = utils.get_batch_and_ind(batches, 1, ind=2)
    self.assertIsNone(ind_batch_out)

  def test_compute_gradient(self):
    x_batch = self.x_train[:5]
    y_batch = self.y_train[:5]
    grads = utils.compute_gradient(
        x_batch, y_batch, self.model, self.criterion
    )
    self.assertEqual(len(grads), len(list(self.model.parameters())))

    # Test with ind_remove
    grads_remove = utils.compute_gradient(
        x_batch, y_batch, self.model, self.criterion, ind_remove=2
    )
    self.assertEqual(len(grads_remove), len(list(self.model.parameters())))

  def test_hessian_and_hessian_prod(self):
    x_batch = self.x_train[:5]
    y_batch = self.y_train[:5]
    grads = utils.compute_gradient(
        x_batch, y_batch, self.model, self.criterion
    )
    hessian = utils.compute_hessian_from_gradient(self.model, grads)

    num_params = sum(p.numel() for p in self.model.parameters())
    self.assertEqual(hessian.shape, (num_params, num_params))

    vec = torch.randn(num_params)
    h_prod = utils.compute_hessian_prod_from_gradient(self.model, grads, vec)
    expected = hessian @ vec

    torch.testing.assert_close(h_prod, expected, rtol=1e-4, atol=1e-4)

  def test_get_grad_diff(self):
    x_batch = self.x_train[:5]
    y_batch = self.y_train[:5]
    theta_diff = [torch.zeros_like(p) for p in self.model.parameters()]
    grads_list, diff_grad = utils.get_grad_diff(
        x_batch, y_batch, self.model, self.criterion, theta_diff, ind_remove=1
    )
    self.assertLen(grads_list, len(list(self.model.parameters())))
    self.assertLen(diff_grad, len(list(self.model.parameters())))

  def test_get_grad_diff_batched(self):
    x_batch = self.x_train[:5]
    y_batch = self.y_train[:5]
    num_params = sum(p.numel() for p in self.model.parameters())
    inds = [0, 1, 3, None]
    m = len(inds)

    theta_diff_mat = torch.randn(m, num_params)

    # Test Pearlmutter mode
    grads_batched, diff_batched = utils.get_grad_diff(
        x_batch,
        y_batch,
        self.model,
        self.criterion,
        theta_diff_mat,
        ind_remove=inds,
        use_pearlmutter=True,
    )
    self.assertEqual(grads_batched.shape, (m, num_params))
    self.assertEqual(diff_batched.shape, (m, num_params))

    # Compare with individual single-index calls
    for k, idx in enumerate(inds):
      single_theta_diff = theta_diff_mat[k]
      grads_k, diff_k = utils.get_grad_diff(
          x_batch,
          y_batch,
          self.model,
          self.criterion,
          single_theta_diff,
          ind_remove=idx,
          use_pearlmutter=True,
      )
      torch.testing.assert_close(
          grads_batched[k], grads_k.squeeze(0), rtol=1e-5, atol=1e-5
      )
      torch.testing.assert_close(
          diff_batched[k], diff_k.squeeze(0), rtol=1e-5, atol=1e-5
      )

    # Test with random mask
    mask = utils.generate_random_mask(self.model, prob=0.6, seed=42)
    mask_flat = torch.cat([r.flatten() for r in mask])
    p_dim = int(mask_flat.sum().item())
    theta_diff_masked = torch.randn(m, p_dim)

    grads_m_batched, diff_m_batched = utils.get_grad_diff(
        x_batch,
        y_batch,
        self.model,
        self.criterion,
        theta_diff_masked,
        ind_remove=inds,
        use_pearlmutter=True,
        random_mask=mask,
    )
    self.assertEqual(grads_m_batched.shape, (m, p_dim))
    self.assertEqual(diff_m_batched.shape, (m, p_dim))

    for k, idx in enumerate(inds):
      single_theta_diff = theta_diff_masked[k]
      grads_k, diff_k = utils.get_grad_diff(
          x_batch,
          y_batch,
          self.model,
          self.criterion,
          single_theta_diff,
          ind_remove=idx,
          use_pearlmutter=True,
          random_mask=mask,
      )
      torch.testing.assert_close(
          grads_m_batched[k], grads_k.squeeze(0), rtol=1e-5, atol=1e-5
      )
      torch.testing.assert_close(
          diff_m_batched[k], diff_k.squeeze(0), rtol=1e-5, atol=1e-5
      )


  def test_compute_hessian_prod_nan_imputation(self):
    class SqrtPowerModel(nn.Module):

      def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([0.0]))

      def forward(self, x):
        return self.param**1.5 * x

    model = SqrtPowerModel()
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    y = torch.tensor([0])
    grads = utils.compute_gradient(x, y, model, self.criterion)
    vec = torch.tensor([1.0])

    h_prod_default = utils.compute_hessian_prod_from_gradient(
        model, grads, vec
    )
    self.assertEqual(h_prod_default.item(), 0.0)

    h_prod_custom = utils.compute_hessian_prod_from_gradient(
        model, grads, vec, nan_impute_value=-999.0
    )
    self.assertEqual(h_prod_custom.item(), -999.0)

  def test_precomputed_gradients_autograd(self):
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = self.x_train[:5]
    y = self.y_train[:5]

    out = model(x)
    loss = criterion(out, y)
    saved_grads = torch.autograd.grad(
        loss, model.parameters(), create_graph=True
    )
    leaf_params = list(model.parameters())

    for p, g in zip(model.parameters(), saved_grads):
      p.grad = g.detach().clone()

    optimizer.step()
    model.zero_grad()

    vec = torch.randn(sum(p.numel() for p in leaf_params))
    flat_grads = torch.cat([g.flatten() for g in saved_grads])
    prod = torch.dot(flat_grads, vec)

    h_prod = torch.autograd.grad(prod, leaf_params, retain_graph=True)
    self.assertLen(h_prod, len(leaf_params))

  def test_pearlmutter_stored_grads(self):
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = self.x_train[:5]
    y = self.y_train[:5]

    out = model(x)
    loss = criterion(out, y)
    saved_grads = torch.autograd.grad(
        loss, model.parameters(), create_graph=True
    )
    leaf_params = list(model.parameters())

    for p, g in zip(model.parameters(), saved_grads):
      p.grad = g.detach().clone()
    optimizer.step()
    optimizer.zero_grad()

    v = torch.randn(sum(p.numel() for p in leaf_params))
    flat_grads = torch.cat([g.flatten() for g in saved_grads])
    prod = torch.dot(flat_grads, v)

    # 1. Differentiating non-detached saved_grads (Pearlmutter's H v trick) succeeds!
    hv_product = torch.autograd.grad(prod, leaf_params, retain_graph=True)
    self.assertLen(hv_product, len(leaf_params))

    # 2. Differentiating detached saved_grads fails as graph was cut
    detached_grads = tuple(g.detach() for g in saved_grads)
    flat_detached = torch.cat([g.flatten() for g in detached_grads])
    prod_detached = torch.dot(flat_detached, v)

    with self.assertRaises(RuntimeError):
      torch.autograd.grad(prod_detached, leaf_params)

  def test_autograd_grad_optimizer_step_equivalence(self):
    # Model 1 trained with loss.backward()
    torch.manual_seed(123)
    model1 = SimpleModel()
    opt1 = torch.optim.Adam(model1.parameters(), lr=0.01)

    # Model 2 trained with autograd.grad() + p.grad assignment
    torch.manual_seed(123)
    model2 = SimpleModel()
    opt2 = torch.optim.Adam(model2.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    for _ in range(5):
      x = torch.randn(5, 4)
      y = torch.randint(0, 2, (5,))

      # Step model 1 (backward)
      opt1.zero_grad()
      out1 = model1(x)
      loss1 = criterion(out1, y)
      loss1.backward()
      opt1.step()

      # Step model 2 (autograd.grad + p.grad assignment)
      opt2.zero_grad()
      out2 = model2(x)
      loss2 = criterion(out2, y)
      grads2 = torch.autograd.grad(
          loss2, model2.parameters(), create_graph=True
      )
      for p, g in zip(model2.parameters(), grads2):
        p.grad = g.detach().clone()
      opt2.step()

    # Compare model parameters after training
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
      torch.testing.assert_close(p1, p2, rtol=0.0, atol=0.0)

    # Compare Adam optimizer state (exp_avg, exp_avg_sq)
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
      torch.testing.assert_close(
          opt1.state[p1]['exp_avg'],
          opt2.state[p2]['exp_avg'],
          rtol=0.0,
          atol=0.0,
      )
      torch.testing.assert_close(
          opt1.state[p1]['exp_avg_sq'],
          opt2.state[p2]['exp_avg_sq'],
          rtol=0.0,
          atol=0.0,
      )

  def test_add_label_noise(self):
    import numpy as np
    labels_np = np.random.randint(0, 10, size=(100,), dtype=np.int64)
    corrupted_np, indices_np = utils.add_label_noise(
        labels_np, noise_rate=0.2, num_classes=10, seed=42
    )
    self.assertEqual(len(indices_np), 20)
    self.assertEqual(len(np.unique(indices_np)), 20)

    # Verify modified labels differ from original labels
    for idx in indices_np:
      self.assertNotEqual(corrupted_np[idx], labels_np[idx])

    # Verify unmodified labels remain identical
    clean_indices = np.setdiff1d(np.arange(100), indices_np)
    np.testing.assert_array_equal(corrupted_np[clean_indices], labels_np[clean_indices])

    # Test with PyTorch Tensor
    labels_tensor = torch.from_numpy(labels_np)
    corrupted_tensor, indices_tensor = utils.add_label_noise(
        labels_tensor, noise_rate=0.2, num_classes=10, seed=42
    )
    self.assertIsInstance(corrupted_tensor, torch.Tensor)
    np.testing.assert_array_equal(corrupted_tensor.numpy(), corrupted_np)
    np.testing.assert_array_equal(indices_tensor, indices_np)

  def test_add_gaussian_noise(self):
    import numpy as np
    x_np = np.zeros((100, 10), dtype=np.float32)
    corrupted_np, indices_np = utils.add_gaussian_noise(
        x_np, noise_rate=0.3, std=0.5, clip_min=0.0, clip_max=1.0, seed=42
    )
    self.assertEqual(len(indices_np), 30)

    # Verify uncorrupted samples are all zeros
    clean_indices = np.setdiff1d(np.arange(100), indices_np)
    np.testing.assert_array_equal(corrupted_np[clean_indices], 0.0)

    # Verify corrupted samples have noise and respect bounds [0.0, 1.0]
    for idx in indices_np:
      self.assertTrue(np.any(corrupted_np[idx] != 0.0))
      self.assertTrue(np.all(corrupted_np[idx] >= 0.0))
      self.assertTrue(np.all(corrupted_np[idx] <= 1.0))

    # Test PyTorch Tensor
    x_tensor = torch.zeros((100, 10), dtype=torch.float32)
    corrupted_tensor, indices_tensor = utils.add_gaussian_noise(
        x_tensor, noise_rate=0.3, std=0.5, clip_min=0.0, clip_max=1.0, seed=42
    )
    self.assertIsInstance(corrupted_tensor, torch.Tensor)
    np.testing.assert_array_equal(corrupted_tensor.numpy(), corrupted_np)

  def test_apply_pass_filter(self):
    x_np = np.random.default_rng(0).uniform(0.0, 1.0, size=(50, 100)).astype(np.float32)

    # Test zero noise rate
    x_clean, inds = utils.apply_pass_filter(x_np, noise_rate=0.0)
    self.assertEqual(len(inds), 0)
    np.testing.assert_array_equal(x_clean, x_np)

    # Test lowpass filter with numpy array
    x_low, inds_low = utils.apply_pass_filter(
        x_np,
        noise_rate=0.2,
        cutoff_freq=0.5,
        filter_order=2,
        sampling_rate=60.0,
        clip_min=0.0,
        clip_max=1.0,
        seed=42,
        btype='lowpass',
    )
    self.assertEqual(len(inds_low), 10)
    self.assertTrue(np.all(x_low >= 0.0))
    self.assertTrue(np.all(x_low <= 1.0))
    # Filtered samples should differ from original
    for idx in inds_low:
      self.assertFalse(np.array_equal(x_low[idx], x_np[idx]))
    # Unfiltered samples should remain identical
    clean_inds = np.setdiff1d(np.arange(50), inds_low)
    np.testing.assert_array_equal(x_low[clean_inds], x_np[clean_inds])

    # Test highpass filter with PyTorch Tensor
    x_tensor = torch.from_numpy(x_np)
    x_high, inds_high = utils.apply_pass_filter(
        x_tensor,
        noise_rate=0.2,
        cutoff_freq=0.5,
        filter_order=2,
        sampling_rate=60.0,
        clip_min=0.0,
        clip_max=1.0,
        seed=42,
        btype='highpass',
    )
    self.assertIsInstance(x_high, torch.Tensor)
    self.assertEqual(len(inds_high), 10)
    self.assertTrue(torch.all(x_high >= 0.0))
    self.assertTrue(torch.all(x_high <= 1.0))

  def test_apply_dataset_noise(self):
    x_train = np.random.default_rng(0).uniform(0.0, 1.0, size=(50, 100)).astype(np.float32)
    y_train = np.zeros((50,), dtype=np.int64)

    # Test 'none'
    x_c, y_c, inds = utils.apply_dataset_noise(x_train, y_train, noise_type='none')
    self.assertEqual(len(inds), 0)

    # Test 'both'
    x_c, y_c, inds = utils.apply_dataset_noise(
        x_train, y_train, noise_type='both', noise_rate=0.2, num_classes=10, gaussian_std=0.1, seed=42
    )
    self.assertGreater(len(inds), 0)
    self.assertFalse(np.array_equal(x_c, x_train))
    self.assertTrue(np.any(y_c != 0))

    # Test 'lowpass'
    x_lp, y_lp, inds_lp = utils.apply_dataset_noise(
        x_train, y_train, noise_type='lowpass', noise_rate=0.2, cutoff_freq=0.5, sampling_rate=60.0, seed=42
    )
    self.assertEqual(len(inds_lp), 10)
    self.assertFalse(np.array_equal(x_lp, x_train))
    # Labels should NOT be modified for filter noise
    np.testing.assert_array_equal(y_lp, y_train)

    # Test 'highpass' / 'high_pass' alias
    x_hp, y_hp, inds_hp = utils.apply_dataset_noise(
        x_train, y_train, noise_type='high_pass', noise_rate=0.2, cutoff_freq=0.5, sampling_rate=60.0, seed=42
    )
    self.assertEqual(len(inds_hp), 10)
    self.assertFalse(np.array_equal(x_hp, x_train))
    np.testing.assert_array_equal(y_hp, y_train)

  def test_garbage_collect(self):
    utils.garbage_collect()
    utils.log_cuda_memory('test_log')

  def test_cnn_and_vit_models(self):
    # Test CNN with 1D input (MNIST shape: 784)
    cnn_mnist = models.get_model('cnn', input_dim=784, output_dim=10)
    x_1d = torch.randn(4, 784)
    out_cnn = cnn_mnist(x_1d)
    self.assertEqual(out_cnn.shape, (4, 10))

    # Test CNN with 1D input (CIFAR shape: 3072)
    cnn_cifar = models.get_model('cnn', input_dim=3072, output_dim=10)
    x_cifar = torch.randn(4, 3072)
    out_cnn_cifar = cnn_cifar(x_cifar)
    self.assertEqual(out_cnn_cifar.shape, (4, 10))

    # Test VisionTransformer with 1D input (MNIST shape: 784)
    vit_mnist = models.get_model('vit', input_dim=784, output_dim=10)
    out_vit = vit_mnist(x_1d)
    self.assertEqual(out_vit.shape, (4, 10))

  def test_vit_gradient_and_hessian(self):
    vit = models.get_model('vit', input_dim=784, output_dim=10)
    x = torch.randn(4, 784)
    y = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss()

    # Test gradient with create_graph=True (scoped SDPA MATH backend)
    grads = utils.compute_gradient(x, y, vit, loss_fn, create_graph=True)
    self.assertEqual(len(grads), len(list(vit.parameters())))

    # Test HVP via get_grad_diff (Pearlmutter's trick)
    theta_diff = [torch.randn_like(p) for p in vit.parameters()]
    grads_list, diff_grad = utils.get_grad_diff(
        x, y, vit, loss_fn, theta_diff, ind_remove=0, use_pearlmutter=True
    )
    self.assertEqual(len(grads_list), len(list(vit.parameters())))
    self.assertEqual(len(diff_grad), len(list(vit.parameters())))

    # Test explicit Hessian on a small ViT
    small_vit = models.VisionTransformer(
        input_dim=64, patch_size=4, embed_dim=16, depth=1, num_heads=2, output_dim=2
    )
    x_small = torch.randn(2, 64)
    y_small = torch.tensor([0, 1], dtype=torch.long)
    grads_small = utils.compute_gradient(
        x_small, y_small, small_vit, loss_fn, create_graph=True
    )
    hessian = utils.compute_hessian_from_gradient(small_vit, grads_small)
    num_params = sum(p.numel() for p in small_vit.parameters())
    self.assertEqual(hessian.shape, (num_params, num_params))

  def test_resnet_models(self):
    # Test ResNet with 1D input (MNIST shape: 784)
    resnet_mnist = models.get_model('resnet', input_dim=784, output_dim=10)
    x_1d = torch.randn(4, 784)
    out_resnet_mnist = resnet_mnist(x_1d)
    self.assertEqual(out_resnet_mnist.shape, (4, 10))

    # Test ResNet18 alias with 1D input (CIFAR shape: 3072)
    resnet_cifar = models.get_model('resnet18', input_dim=3072, output_dim=10)
    x_cifar = torch.randn(4, 3072)
    out_resnet_cifar = resnet_cifar(x_cifar)
    self.assertEqual(out_resnet_cifar.shape, (4, 10))

    # Test ResNet with 4D input
    x_4d = torch.randn(4, 3, 32, 32)
    out_4d = resnet_cifar(x_4d)
    self.assertEqual(out_4d.shape, (4, 10))

  def test_cifar_hwc_flatten_equivalence(self):
    # Create random HWC images (4, 32, 32, 3)
    x_hwc = torch.randn(4, 32, 32, 3)
    # Flattened from HWC as in load_cifar_data
    x_flat = x_hwc.reshape(4, -1)
    # Ground truth CHW tensor (4, 3, 32, 32)
    x_chw = x_hwc.permute(0, 3, 1, 2).contiguous()

    for model_name in ['cnn', 'vit', 'resnet18']:
      model = models.get_model(model_name, input_dim=3072, output_dim=10)
      model.eval()
      with torch.no_grad():
        out_flat = model(x_flat)
        out_chw = model(x_chw)
        out_hwc = model(x_hwc)
        torch.testing.assert_close(out_flat, out_chw)
        torch.testing.assert_close(out_hwc, out_chw)

  def test_resnet_gradient_and_hessian(self):
    resnet = models.get_model('resnet18', input_dim=784, output_dim=10)
    resnet.eval()
    x = torch.randn(4, 784)
    y = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss()

    grads = utils.compute_gradient(x, y, resnet, loss_fn, create_graph=True)
    self.assertEqual(len(grads), len(list(resnet.parameters())))

    theta_diff = [torch.randn_like(p) for p in resnet.parameters()]
    grads_list, diff_grad = utils.get_grad_diff(
        x, y, resnet, loss_fn, theta_diff, ind_remove=0, use_pearlmutter=True
    )
    self.assertEqual(len(grads_list), len(list(resnet.parameters())))
    self.assertEqual(len(diff_grad), len(list(resnet.parameters())))


if __name__ == '__main__':
  absltest.main()
