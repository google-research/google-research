# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Benchmarks for symmetric_matrices.

The custom benchmarks are used to show runtime for different chosen block sizes.
"""
import time

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
import jax.numpy as jnp
import pandas as pd




from scalable_shampoo.optax.symmetric_matrices import symmetric_matrices


def make_device_mat(shape, dtype, seed=0):
  """Returns a matrix on device."""
  key = jax.random.PRNGKey(seed)
  mat = jax.random.uniform(key=key, shape=shape, dtype=dtype)
  return jax.device_put(mat)


def time_func(func, kwargs, iters):
  """Returns the average function runtime."""
  start = time.monotonic()
  for _ in range(iters):
    _ = func(**kwargs)  # output is unused
  end = time.monotonic()
  return (end - start) / iters


def optimal_savings_square(mat_size, block_size):
  """optimal savings based on the block size of a square matrix."""
  if mat_size % block_size != 0:
    raise ValueError('block size must evenly divide mat_size.')
  num_blocks = mat_size // block_size
  return num_blocks * (num_blocks + 1) * block_size**2 / (2 * mat_size**2)


def time_setup_and_run(
    func,
    kwargs,
    iters=100,
):
  """Returns a pandas series with compile and average iteration time.

  To get compile run

  Args:
    func: The function to run.
    kwargs: The arguments to pass to func, as a dict.
    iters: The number of iterations to use.
  """
  # Run the function once to get the compile time.
  compile_time = time_func(func, kwargs, iters=1)

  # Run the function several times, since we want a good estimate here.
  average_iter_time = time_func(func, kwargs, iters=iters)

  return pd.Series(
      data=[compile_time - average_iter_time, average_iter_time],
      index=['compile_time', 'average_iter_time'],
      name='time (s)')


def run_benchmarks_sliced_transposed_product(matmul_func,
                                             mat_size=4096,
                                             batch_size=16,
                                             min_block_size=128,
                                             precision=lax.Precision.DEFAULT,
                                             dtype=jnp.float32,
                                             seed=0):
  """Returns a Dataframe with benchmarks for sliced_transposed_product.

  The sliced_transposed_product is applied to compute G * G^T for a matrix G of
  size mat_size. Varied block sizes are used, from min_block_size up to mat_size
  in increments of powers of 2. The last row corresponds to the full matrix
  multiplication.

  The returned dataframe will have index 'block_size' corresponding to the
  benchmark being run. The other columns are:

    - compile_time: The time to compile (based on a single compilation). We
      subtract the average_iter_time since the compilation step also runs a
      single iteration.
    - average_iter_time: The averaged time over 100 iterations.
    - optimal_savings: The ideal proportion of compute based on the block size,
      as compared to the full matrix multiplication (last row).
    - realized_runtime_savings: The realized proportion of time to run each row
      compared to the last row (full matrix multiplication). It is lower-bounded
      by the optimal_savings.

  Args:
    matmul_func: Which function to use for the matrix multiplication G * G^T.
      This function is expected to have inputs mat, block_size, and precision.
    mat_size: The size of the matrix to use.
    batch_size: Batch size to use. The batching will be done via vmap.
    min_block_size: The smallest block size to use. The various block sizes will
      multiply this by powers of 2 until the matrix size is reached.
    precision: The JAX precision to use.
    dtype: The dtype to use for the input matrix.
    seed: Seed to use for randomly generating the input matrix.
  """
  if mat_size % min_block_size != 0:
    raise ValueError(f'min_block_size={min_block_size} does not divide '
                     f'mat_size={mat_size}.')
  block_sizes = [
      min_block_size * 2**i
      for i in range(1 + int(jnp.around(jnp.log2(mat_size // min_block_size))))
  ]
  if block_sizes[-1] != mat_size:
    raise ValueError(f'Largest computed block size={block_sizes[-1]} did not '
                     f'match mat_size={mat_size}.')

  print(f'Collecting data for mat_size={mat_size}.')
  results = {}

  def make_matmul_func(block_size, precision):
    """Make the matmul function on batches."""
    def _matmul_func(mat):
      return matmul_func(mat=mat, block_size=block_size, precision=precision)
    return jax.vmap(_matmul_func)

  for block_size in block_sizes:
    vmap_matmul = make_matmul_func(block_size, precision)
    mat = make_device_mat(
        shape=(batch_size, mat_size, mat_size), dtype=dtype, seed=seed)
    results[block_size] = time_setup_and_run(
        func=vmap_matmul,
        kwargs={'mat': mat},
        iters=100,
    )
  print('Done collecting data.')

  df = pd.DataFrame(results).T
  df['optimal_savings'] = [
      optimal_savings_square(mat_size, block_size) for block_size in block_sizes
  ]
  df['realized_runtime_savings'] = df['average_iter_time'] / df[
      'average_iter_time'][mat_size]
  df.index.name = 'block size'
  return df


def _print_df(df):
  with pd.option_context(
      'display.max_rows', None,
      'display.max_columns', None,
      'display.expand_frame_repr', False):
    print(df)


class SymmetricMatrixBenchmarkTest(parameterized.TestCase):
  """Run benchmarks for symmetric matrix operations."""

  @parameterized.named_parameters([
      ('transposed_product', symmetric_matrices.sliced_transposed_product),
      ('transposed_product_concat',
       symmetric_matrices.sliced_transposed_product_concat)
  ])
  def test_run_benchmarks_sliced_transposed_product(self, matmul_func):
    """Run the benchmarks for symmetric matrix multiplication."""
    print(f'devices: {jax.devices()}')
    print(f'Running benchmarks in test {self.id()}:')
    xprof = xprof_session.XprofSession()
    xprof.start_session(enable_python_tracer=True)
    df = run_benchmarks_sliced_transposed_product(matmul_func=matmul_func)
    _print_df(df)
    xprof_url = xprof.end_session_and_get_url(tag='')
    print(f'xprof URL: {xprof_url}')

if __name__ == '__main__':
  absltest.main()
