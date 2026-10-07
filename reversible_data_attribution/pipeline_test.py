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

"""Unit tests for pipeline.py dataset loading and I/O utilities."""

import os
import tempfile

from absl.testing import absltest
import numpy as np

from reversible_data_attribution import pipeline


class PipelineTest(absltest.TestCase):

  def test_load_mnist_data(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      npz_path = os.path.join(tmpdir, 'mnist_dummy.npz')
      x_tr_dummy = np.random.randint(0, 256, (20, 28, 28), dtype=np.uint8)
      y_tr_dummy = np.random.randint(0, 10, (20,), dtype=np.int64)
      x_te_dummy = np.random.randint(0, 256, (10, 28, 28), dtype=np.uint8)
      y_te_dummy = np.random.randint(0, 10, (10,), dtype=np.int64)
      np.savez_compressed(
          npz_path,
          x_train=x_tr_dummy,
          y_train=y_tr_dummy,
          x_test=x_te_dummy,
          y_test=y_te_dummy,
      )

      x_tr, y_tr, x_te, y_te = pipeline.load_mnist_data(
          npz_path, num_train=10, num_test=5
      )
      self.assertEqual(x_tr.shape, (10, 784))
      self.assertEqual(y_tr.shape, (10,))
      self.assertEqual(x_te.shape, (5, 784))
      self.assertEqual(y_te.shape, (5,))

  def test_load_cifar_data(self):
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
      npz_path = os.path.join(tmpdir, 'cifar_dummy.npz')
      x_tr_dummy = np.random.randint(0, 256, (20, 32, 32, 3), dtype=np.uint8)
      y_tr_dummy = np.random.randint(0, 10, (20,), dtype=np.int64)
      x_te_dummy = np.random.randint(0, 256, (10, 32, 32, 3), dtype=np.uint8)
      y_te_dummy = np.random.randint(0, 10, (10,), dtype=np.int64)
      np.savez_compressed(
          npz_path,
          x_train=x_tr_dummy,
          y_train=y_tr_dummy,
          x_test=x_te_dummy,
          y_test=y_te_dummy,
      )

      x_tr, y_tr, x_te, y_te = pipeline.load_cifar_data(
          npz_path, num_train=10, num_test=5, flatten=True
      )
      self.assertEqual(x_tr.shape, (10, 3072))
      self.assertEqual(y_tr.shape, (10,))
      self.assertEqual(x_te.shape, (5, 3072))
      self.assertEqual(y_te.shape, (5,))

  def test_load_mnist_corrupted_data(self):
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
      npz_path = os.path.join(tmpdir, 'mnist_corrupted_dummy.npz')
      x_tr_dummy = np.random.randint(0, 256, (20, 28, 28), dtype=np.uint8)
      y_tr_dummy = np.random.randint(0, 10, (20,), dtype=np.int64)
      x_te_dummy = np.random.randint(0, 256, (10, 28, 28), dtype=np.uint8)
      y_te_dummy = np.random.randint(0, 10, (10,), dtype=np.int64)
      np.savez_compressed(
          npz_path,
          x_train=x_tr_dummy,
          y_train=y_tr_dummy,
          x_test=x_te_dummy,
          y_test=y_te_dummy,
      )

      x_tr, y_tr, x_te, y_te = pipeline.load_mnist_corrupted_data(
          npz_path, num_train=10, num_test=5
      )
      self.assertEqual(x_tr.shape, (10, 784))
      self.assertEqual(y_tr.shape, (10,))
      self.assertEqual(x_te.shape, (5, 784))
      self.assertEqual(y_te.shape, (5,))

  def test_load_mnist_data_with_noise(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      npz_path = os.path.join(tmpdir, 'mnist_dummy.npz')
      x_tr_dummy = np.random.randint(0, 256, (60, 28, 28), dtype=np.uint8)
      y_tr_dummy = np.random.randint(0, 10, (60,), dtype=np.int64)
      x_te_dummy = np.random.randint(0, 256, (20, 28, 28), dtype=np.uint8)
      y_te_dummy = np.random.randint(0, 10, (20,), dtype=np.int64)
      np.savez_compressed(
          npz_path,
          x_train=x_tr_dummy,
          y_train=y_tr_dummy,
          x_test=x_te_dummy,
          y_test=y_te_dummy,
      )

      x_clean_tr, y_clean_tr, x_clean_te, y_clean_te = pipeline.load_mnist_data(
          npz_path, num_train=50, num_test=10
      )
      x_noisy_tr, y_noisy_tr, x_noisy_te, y_noisy_te = pipeline.load_mnist_data(
          npz_path,
          num_train=50,
          num_test=10,
          noise_type='both',
          noise_rate=0.2,
          gaussian_std=0.2,
          seed=42,
      )

      # Test split MUST remain completely identical (unaffected by noise)
      np.testing.assert_array_equal(x_clean_te, x_noisy_te)
      np.testing.assert_array_equal(y_clean_te, y_noisy_te)

      # Train split MUST be modified on a subset
      self.assertFalse(np.array_equal(x_clean_tr, x_noisy_tr))
      self.assertFalse(np.array_equal(y_clean_tr, y_noisy_tr))

      # Test lowpass filter noise
      x_lp_tr, y_lp_tr, x_lp_te, y_lp_te = pipeline.load_mnist_data(
          npz_path,
          num_train=50,
          num_test=10,
          noise_type='lowpass',
          noise_rate=0.2,
          cutoff_freq=0.5,
          sampling_rate=60.0,
          seed=42,
      )
      np.testing.assert_array_equal(x_clean_te, x_lp_te)
      np.testing.assert_array_equal(y_clean_tr, y_lp_tr)
      self.assertFalse(np.array_equal(x_clean_tr, x_lp_tr))

      # Test highpass filter noise
      x_hp_tr, y_hp_tr, x_hp_te, y_hp_te = pipeline.load_mnist_data(
          npz_path,
          num_train=50,
          num_test=10,
          noise_type='highpass',
          noise_rate=0.2,
          cutoff_freq=0.5,
          sampling_rate=60.0,
          seed=42,
      )
      np.testing.assert_array_equal(x_clean_te, x_hp_te)
      np.testing.assert_array_equal(y_clean_tr, y_hp_tr)
      self.assertFalse(np.array_equal(x_clean_tr, x_hp_tr))

  def test_load_cifar_data_with_noise(self):
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
      npz_path = os.path.join(tmpdir, 'cifar_dummy.npz')
      x_tr_dummy = np.random.randint(0, 256, (50, 32, 32, 3), dtype=np.uint8)
      y_tr_dummy = np.random.randint(0, 10, (50,), dtype=np.int64)
      x_te_dummy = np.random.randint(0, 256, (10, 32, 32, 3), dtype=np.uint8)
      y_te_dummy = np.random.randint(0, 10, (10,), dtype=np.int64)
      np.savez_compressed(
          npz_path,
          x_train=x_tr_dummy,
          y_train=y_tr_dummy,
          x_test=x_te_dummy,
          y_test=y_te_dummy,
      )

      x_clean_tr, y_clean_tr, x_clean_te, y_clean_te = pipeline.load_cifar_data(
          npz_path, num_train=50, num_test=10, flatten=True
      )
      x_noisy_tr, y_noisy_tr, x_noisy_te, y_noisy_te = pipeline.load_cifar_data(
          npz_path,
          num_train=50,
          num_test=10,
          flatten=True,
          noise_type='both',
          noise_rate=0.2,
          gaussian_std=0.2,
          seed=42,
      )

      # Test split MUST remain completely clean/identical
      np.testing.assert_array_equal(x_clean_te, x_noisy_te)
      np.testing.assert_array_equal(y_clean_te, y_noisy_te)

      # Train split MUST be modified on a subset
      self.assertFalse(np.array_equal(x_clean_tr, x_noisy_tr))
      self.assertFalse(np.array_equal(y_clean_tr, y_noisy_tr))

  def test_load_fashion_mnist_data(self):
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
      npz_path = os.path.join(tmpdir, 'fashion_mnist_dummy.npz')
      x_tr_dummy = np.random.randint(0, 256, (20, 28, 28), dtype=np.uint8)
      y_tr_dummy = np.random.randint(0, 10, (20,), dtype=np.int64)
      x_te_dummy = np.random.randint(0, 256, (10, 28, 28), dtype=np.uint8)
      y_te_dummy = np.random.randint(0, 10, (10,), dtype=np.int64)
      np.savez_compressed(
          npz_path,
          x_train=x_tr_dummy,
          y_train=y_tr_dummy,
          x_test=x_te_dummy,
          y_test=y_te_dummy,
      )

      x_tr, y_tr, x_te, y_te = pipeline.load_fashion_mnist_data(
          npz_path, num_train=10, num_test=5
      )
      self.assertEqual(x_tr.shape, (10, 784))
      self.assertEqual(y_tr.shape, (10,))
      self.assertEqual(x_te.shape, (5, 784))
      self.assertEqual(y_te.shape, (5,))

  def test_load_imdb_data(self):
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
      npz_path = os.path.join(tmpdir, 'imdb_dummy.npz')
      x_tr_dummy = np.array(
          [[1, 14, 22, 50], [1, 3, 4, 100], [1, 2, 5]], dtype=object
      )
      y_tr_dummy = np.array([1, 0, 1], dtype=np.int64)
      x_te_dummy = np.array([[1, 4, 8], [1, 99]], dtype=object)
      y_te_dummy = np.array([0, 1], dtype=np.int64)
      np.savez_compressed(
          npz_path,
          x_train=x_tr_dummy,
          y_train=y_tr_dummy,
          x_test=x_te_dummy,
          y_test=y_te_dummy,
      )

      x_tr, y_tr, x_te, y_te = pipeline.load_imdb_data(
          npz_path, num_train=3, num_test=2, num_words=100
      )
      self.assertEqual(x_tr.shape, (3, 100))
      self.assertEqual(y_tr.shape, (3,))
      self.assertEqual(x_te.shape, (2, 100))
      self.assertEqual(y_te.shape, (2,))
      # Verify multi-hot encoding
      self.assertEqual(x_tr[0, 14], 1.0)
      self.assertEqual(x_tr[0, 22], 1.0)
      self.assertEqual(x_tr[0, 15], 0.0)


if __name__ == '__main__':
  absltest.main()
