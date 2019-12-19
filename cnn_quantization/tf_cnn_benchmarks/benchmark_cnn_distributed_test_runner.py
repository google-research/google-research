# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Used to run benchmark_cnn for distributed tests.

In distributed tests, we spawn processes to run tf_cnn_benchmark tasks. We could
directly spawn tf_cnn_benchmark processes, but we want some added functionality,
such as being able to inject custom images during training. So instead, this
file is spawned as a Python process, which supports the added functionality.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags as absl_flags
import numpy as np
import tensorflow.compat.v1 as tf
from cnn_quantization.tf_cnn_benchmarks import benchmark_cnn
from cnn_quantization.tf_cnn_benchmarks import flags
from cnn_quantization.tf_cnn_benchmarks import preprocessing
from cnn_quantization.tf_cnn_benchmarks import test_util


absl_flags.DEFINE_string('fake_input', 'none',
                         """What fake input to inject into benchmark_cnn. This
                            is ignored if --model=test_model.
                            Options are:
                            none: Do not inject any fake input.
                            zeros_and_ones: Half the images will be all 0s with
                            a label of 0. Half the images will be all 1s with a
                            label of 1.""")

flags.define_flags()
FLAGS = flags.FLAGS


def get_test_image_preprocessor(batch_size, params):
  """Returns the preprocessing.TestImagePreprocessor that should be injected.

  Returns None if no preprocessor should be injected.

  Args:
    batch_size: The batch size across all GPUs.
    params: BenchmarkCNN's parameters.
  Returns:
    Returns the preprocessing.TestImagePreprocessor that should be injected.
  Raises:
    ValueError: Flag --fake_input is an invalid value.
  """
  if FLAGS.fake_input == 'none':
    return None
  elif FLAGS.fake_input == 'zeros_and_ones':
    half_batch_size = batch_size // 2
    images = np.zeros((batch_size, 227, 227, 3), dtype=np.float32)
    images[half_batch_size:, :, :, :] = 1
    labels = np.array([0] * half_batch_size + [1] * half_batch_size,
                      dtype=np.int32)
    preprocessor = preprocessing.TestImagePreprocessor(
        batch_size, [227, 227, 3], params.num_gpus,
        benchmark_cnn.get_data_type(params))
    preprocessor.set_fake_data(images, labels)
    preprocessor.expected_subset = 'validation' if params.eval else 'train'
    return preprocessor
  else:
    raise ValueError('Invalid --fake_input: %s' % FLAGS.fake_input)


def run_with_real_model(params):
  """Runs tf_cnn_benchmarks with a real model."""
  bench = benchmark_cnn.BenchmarkCNN(params)
  bench.print_info()
  preprocessor = get_test_image_preprocessor(bench.batch_size, params)
  if preprocessor is not None:
    # The test image preprocessor requires queue runners. Since this file is
    # used for testing, it is OK to access protected members.
    # pylint: disable=protected-access
    bench.dataset._queue_runner_required = True
    # pylint: enable=protected-access
    bench.input_preprocessor = preprocessor
  bench.run()


def run_with_test_model(params):
  """Runs tf_cnn_benchmarks with a test model."""
  model = test_util.TestCNNModel()
  inputs = test_util.get_fake_var_update_inputs()
  with test_util.monkey_patch(benchmark_cnn,
                              LOSS_AND_ACCURACY_DIGITS_TO_SHOW=15):
    bench = benchmark_cnn.BenchmarkCNN(params, dataset=test_util.TestDataSet(),
                                       model=model)
    # The test model does not use labels when computing loss, so the label
    # values do not matter as long as it's the right shape.
    labels = np.array([1] * inputs.shape[0])
    bench.input_preprocessor.set_fake_data(inputs, labels)
    bench.run()


def main(_):
  params = benchmark_cnn.make_params_from_flags()
  params = benchmark_cnn.setup(params)
  if params.model == 'test_model':
    run_with_test_model(params)
  else:
    run_with_real_model(params)


if __name__ == '__main__':
  tf.app.run()
