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

"""Tests for GPUReversibleTransform."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from reversible_data_attribution import reversible


class GPUReversibleTransformTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def test_10000_steps_reversible_decay_gamma_0_9(self):
    """Test 10,000 forward and backward decay steps with gamma=0.9."""
    dim = 100
    num_steps = 10000
    gamma = 0.9
    checkpoint_interval = 500

    transform = reversible.GPUReversibleTransform(
        num_params=dim,
        gamma=gamma,
        max_steps=num_steps,
        device=self.device,
        mode="floor",
    )

    # 1. Generate N=10000 int64 vectors of length 100 simulating typical
    # integer-quantized momentum/gradients (e.g. 0 to 1e7)
    v_vectors = [
        torch.randint(
            low=0,
            high=10_000_000,
            size=(dim,),
            dtype=torch.int64,
            device=self.device,
        )
        for _ in range(num_steps)
    ]

    # Caches for periodic state checking before forward decay
    cached_active_buffers = {}
    cached_limb_indices = {}
    cached_tapes = {}

    # 2. Forward decay v[0], ..., v[N-1]
    w_vectors = []
    for m in range(num_steps):
      if m % checkpoint_interval == 0 or m == 0:
        cached_active_buffers[m] = transform.active_buffer.clone()
        cached_limb_indices[m] = transform.limb_idx
        cached_tapes[m] = transform.tape.clone()

      w_m = transform.forward_decay(v_vectors[m].clone())
      w_vectors.append(w_m)

    # Verify that the forward pass actually utilized multi-limb tape
    # (For 10,000 steps with gamma=0.9, limb_idx will advance beyond 0)
    self.assertGreater(
        transform.limb_idx,
        0,
        "Limb tape should have spilled to multiple limbs over 10,000 steps.",
    )

    # 3. Backward decay w[N-1], ..., w[0] in reverse order
    v_reconstructed = []
    for m in reversed(range(num_steps)):
      v_recon_m = transform.backward_decay(w_vectors[m].clone())
      v_reconstructed.append(v_recon_m)

      # Check reconstructed vector matches original v[m] exactly
      torch.testing.assert_close(
          v_recon_m,
          v_vectors[m],
          msg=f"Reconstructed vector mismatch at step {m}",
      )

      # Periodically check that buffer state after backward decay of w[m]
      # matches the cached buffer state before forward decay of v[m]
      if m % checkpoint_interval == 0 or m == 0:
        torch.testing.assert_close(
            transform.active_buffer,
            cached_active_buffers[m],
            msg=f"Active buffer mismatch at step {m}",
        )
        self.assertEqual(
            transform.limb_idx,
            cached_limb_indices[m],
            msg=f"Limb index mismatch at step {m}",
        )
        torch.testing.assert_close(
            transform.tape,
            cached_tapes[m],
            msg=f"Tape contents mismatch at step {m}",
        )

    # 4. Final checks after full backward unrolling
    # Check that final active buffer is all zeros
    self.assertTrue(
        (transform.active_buffer == 0).all().item(),
        "Final active buffer must be all zeros, got:"
        f" {transform.active_buffer}",
    )
    # Check that limb_idx is back to 0
    self.assertEqual(
        transform.limb_idx,
        0,
        f"Final limb_idx must be 0, got: {transform.limb_idx}",
    )

  def test_10000_steps_reversible_decay_gamma_0_999(self):
    """Test 10,000 forward and backward decay steps with gamma=0.999."""
    dim = 100
    num_steps = 10000
    gamma = 0.999

    transform = reversible.GPUReversibleTransform(
        num_params=dim,
        gamma=gamma,
        max_steps=num_steps,
        device=self.device,
        mode="floor",
    )

    v_vectors = [
        torch.randint(
            low=0,
            high=10_000_000,
            size=(dim,),
            dtype=torch.int64,
            device=self.device,
        )
        for _ in range(num_steps)
    ]

    w_vectors = [transform.forward_decay(v.clone()) for v in v_vectors]

    for m in reversed(range(num_steps)):
      v_recon = transform.backward_decay(w_vectors[m].clone())
      torch.testing.assert_close(
          v_recon,
          v_vectors[m],
          msg=f"Reconstructed variance vector mismatch at step {m}",
      )

    self.assertTrue((transform.active_buffer == 0).all().item())
    self.assertEqual(transform.limb_idx, 0)

  def test_10000_steps_reversible_decay_gamma_0_9_ceil_mode(self):
    """Test 10,000 forward and backward decay steps with gamma=0.9 in ceil mode."""
    dim = 100
    num_steps = 10000
    gamma = 0.9
    checkpoint_interval = 500

    transform = reversible.GPUReversibleTransform(
        num_params=dim,
        gamma=gamma,
        max_steps=num_steps,
        device=self.device,
        mode="ceil",
    )

    # 1. Generate N=10000 int64 vectors of length 100 simulating typical
    # integer-quantized momentum/gradients (e.g. 0 to 1e7)
    v_vectors = [
        torch.randint(
            low=0,
            high=10_000_000,
            size=(dim,),
            dtype=torch.int64,
            device=self.device,
        )
        for _ in range(num_steps)
    ]

    # Caches for periodic state checking before forward decay
    cached_active_buffers = {}
    cached_limb_indices = {}
    cached_tapes = {}

    # 2. Forward decay v[0], ..., v[N-1]
    w_vectors = []
    for m in range(num_steps):
      if m % checkpoint_interval == 0 or m == 0:
        cached_active_buffers[m] = transform.active_buffer.clone()
        cached_limb_indices[m] = transform.limb_idx
        cached_tapes[m] = transform.tape.clone()

      w_m = transform.forward_decay(v_vectors[m].clone())
      w_vectors.append(w_m)

    # Verify that the forward pass actually utilized multi-limb tape
    self.assertGreater(
        transform.limb_idx,
        0,
        "Limb tape should have spilled to multiple limbs over 10,000 steps.",
    )

    # 3. Backward decay w[N-1], ..., w[0] in reverse order
    v_reconstructed = []
    for m in reversed(range(num_steps)):
      v_recon_m = transform.backward_decay(w_vectors[m].clone())
      v_reconstructed.append(v_recon_m)

      # Check reconstructed vector matches original v[m] exactly
      torch.testing.assert_close(
          v_recon_m,
          v_vectors[m],
          msg=f"Reconstructed vector mismatch at step {m}",
      )

      # Periodically check that buffer state after backward decay of w[m]
      # matches the cached buffer state before forward decay of v[m]
      if m % checkpoint_interval == 0 or m == 0:
        torch.testing.assert_close(
            transform.active_buffer,
            cached_active_buffers[m],
            msg=f"Active buffer mismatch at step {m}",
        )
        self.assertEqual(
            transform.limb_idx,
            cached_limb_indices[m],
            msg=f"Limb index mismatch at step {m}",
        )
        torch.testing.assert_close(
            transform.tape,
            cached_tapes[m],
            msg=f"Tape contents mismatch at step {m}",
        )

    # 4. Final checks after full backward unrolling
    self.assertTrue(
        (transform.active_buffer == 0).all().item(),
        "Final active buffer must be all zeros, got:"
        f" {transform.active_buffer}",
    )
    self.assertEqual(
        transform.limb_idx,
        0,
        f"Final limb_idx must be 0, got: {transform.limb_idx}",
    )

  def test_10000_steps_reversible_decay_gamma_0_999_ceil_mode(self):
    """Test 10,000 forward and backward decay steps with gamma=0.999 in ceil mode."""
    dim = 100
    num_steps = 10000
    gamma = 0.999

    transform = reversible.GPUReversibleTransform(
        num_params=dim,
        gamma=gamma,
        max_steps=num_steps,
        device=self.device,
        mode="ceil",
    )

    v_vectors = [
        torch.randint(
            low=0,
            high=10_000_000,
            size=(dim,),
            dtype=torch.int64,
            device=self.device,
        )
        for _ in range(num_steps)
    ]

    w_vectors = [transform.forward_decay(v.clone()) for v in v_vectors]

    for m in reversed(range(num_steps)):
      v_recon = transform.backward_decay(w_vectors[m].clone())
      torch.testing.assert_close(
          v_recon,
          v_vectors[m],
          msg=f"Reconstructed variance vector mismatch at step {m}",
      )

    self.assertTrue((transform.active_buffer == 0).all().item())
    self.assertEqual(transform.limb_idx, 0)

  @parameterized.parameters(
      (0.5, 500, "floor"),
      (0.5, 500, "ceil"),
      (0.8, 1000, "floor"),
      (0.8, 1000, "ceil"),
      (0.9, 2000, "floor"),
      (0.9, 2000, "ceil"),
      (0.95, 2000, "floor"),
      (0.95, 2000, "ceil"),
      (0.99, 2000, "floor"),
      (0.99, 2000, "ceil"),
  )
  def test_different_gamma_values(self, gamma, num_steps, mode):
    """Test reversibility across various gamma values and modes."""
    dim = 50
    transform = reversible.GPUReversibleTransform(
        num_params=dim,
        gamma=gamma,
        max_steps=num_steps,
        device=self.device,
        mode=mode,
    )

    v_vectors = [
        torch.randint(
            low=0,
            high=5_000_000,
            size=(dim,),
            dtype=torch.int64,
            device=self.device,
        )
        for _ in range(num_steps)
    ]

    w_vectors = [transform.forward_decay(v.clone()) for v in v_vectors]

    for m in reversed(range(num_steps)):
      v_recon = transform.backward_decay(w_vectors[m].clone())
      torch.testing.assert_close(v_recon, v_vectors[m])

    self.assertTrue((transform.active_buffer == 0).all().item())
    self.assertEqual(transform.limb_idx, 0)

  @parameterized.parameters(
      (0.5, 500),
      (0.8, 1000),
      (0.9, 2000),
      (0.95, 2000),
      (0.99, 2000),
      (0.999, 2000),
  )
  def test_forward_decay_bounds_floor_mode(self, gamma, num_steps):
    """Test that w = forward_decay(v) satisfies floor(v/d)*n <= w < (floor(v/d)+1)*n."""
    dim = 100
    transform = reversible.GPUReversibleTransform(
        num_params=dim,
        gamma=gamma,
        max_steps=num_steps,
        device=self.device,
        mode="floor",
    )
    d = transform.den
    n = transform.num

    for _ in range(num_steps):
      v = torch.randint(
          low=0,
          high=10_000_000,
          size=(dim,),
          dtype=torch.int64,
          device=self.device,
      )
      w = transform.forward_decay(v)
      floor_div = torch.div(v, d, rounding_mode="floor")
      lower_bound = floor_div * n
      upper_bound = (floor_div + 1) * n

      self.assertTrue(
          (w >= lower_bound).all().item(),
          f"Floor mode lower bound violated: w >= floor(v/d)*n failed for gamma={gamma}",
      )
      self.assertTrue(
          (w < upper_bound).all().item(),
          f"Floor mode upper bound violated: w < (floor(v/d)+1)*n failed for gamma={gamma}",
      )

  @parameterized.parameters(
      (0.5, 500),
      (0.8, 1000),
      (0.9, 2000),
      (0.95, 2000),
      (0.99, 2000),
      (0.999, 2000),
  )
  def test_forward_decay_bounds_ceil_mode(self, gamma, num_steps):
    """Test that w = forward_decay(v) satisfies (ceil(v/d)-1)*n < w <= ceil(v/d)*n."""
    dim = 100
    transform = reversible.GPUReversibleTransform(
        num_params=dim,
        gamma=gamma,
        max_steps=num_steps,
        device=self.device,
        mode="ceil",
    )
    d = transform.den
    n = transform.num

    for _ in range(num_steps):
      v = torch.randint(
          low=0,
          high=10_000_000,
          size=(dim,),
          dtype=torch.int64,
          device=self.device,
      )
      w = transform.forward_decay(v)
      # Compute ceil(v / d) using exact integer division: -((-v) // d)
      ceil_div = -torch.div(-v, d, rounding_mode="floor")
      lower_bound = (ceil_div - 1) * n
      upper_bound = ceil_div * n

      self.assertTrue(
          (w > lower_bound).all().item(),
          f"Ceil mode lower bound violated: (ceil(v/d)-1)*n < w failed for gamma={gamma}",
      )
      self.assertTrue(
          (w <= upper_bound).all().item(),
          f"Ceil mode upper bound violated: w <= ceil(v/d)*n failed for gamma={gamma}",
      )

  def test_reset(self):
    """Test reset restores the transform to its initial clean state."""
    dim = 10
    for mode in ["floor", "ceil"]:
      transform = reversible.GPUReversibleTransform(
          num_params=dim,
          gamma=0.9,
          max_steps=1000,
          device=self.device,
          mode=mode,
      )
      for _ in range(500):
        v = torch.randint(
            0, 100000, (dim,), dtype=torch.int64, device=self.device
        )
        transform.forward_decay(v)

      transform.reset()
      self.assertTrue((transform.active_buffer == 0).all().item())
      self.assertTrue((transform.tape == 0).all().item())
      self.assertEqual(transform.limb_idx, 0)


if __name__ == "__main__":
  absltest.main()
