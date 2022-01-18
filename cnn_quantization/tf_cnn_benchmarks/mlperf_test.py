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

"""Contains tests related to MLPerf.

Note this test only passes if the MLPerf compliance library is installed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import logging
import re

import six
import tensorflow.compat.v1 as tf
from cnn_quantization.tf_cnn_benchmarks import benchmark_cnn
from cnn_quantization.tf_cnn_benchmarks import datasets
from cnn_quantization.tf_cnn_benchmarks import mlperf
from cnn_quantization.tf_cnn_benchmarks import test_util
from cnn_quantization.tf_cnn_benchmarks.models import model
from tensorflow_models.mlperf.models.rough.mlperf_compliance import mlperf_log


class _MlPerfTestModel(model.CNNModel):
  """A model to test the MLPerf compliance logging on."""

  def __init__(self):
    super(_MlPerfTestModel, self).__init__(
        'mlperf_test_model', image_size=224, batch_size=2, learning_rate=1)

  def add_inference(self, cnn):
    assert cnn.top_layer.shape[1:] == (3, 224, 224)
    cnn.conv(1, 1, 1, 1, 1, use_batch_norm=True)
    cnn.mpool(1, 1, 1, 1, num_channels_in=1)
    cnn.reshape([-1, 224 * 224])
    cnn.affine(1, activation=None)

    # Assert that the batch norm variables are filtered out for L2 loss.
    variables = tf.global_variables() + tf.local_variables()
    assert len(variables) > len(self.filter_l2_loss_vars(variables))


class MlPerfComplianceTest(tf.test.TestCase):
  """Tests the MLPerf compliance logs.

  This serves as a quick check that we probably didn't break the compliance
  logging. It is not mean to be as comprehensive as the official MLPerf
  compliance checker will be.
  """

  def setUp(self):
    super(MlPerfComplianceTest, self).setUp()
    benchmark_cnn.setup(benchmark_cnn.make_params())

  # Map between regex and the number of times we expect to see that regex in the
  # logs. Entry commented out with the comment FIXME indicate that
  # tf_cnn_benchmarks currently fails compliance in that regard, and needs to be
  # fixed to be MLPerf compliant.
  EXPECTED_LOG_REGEXES = {
      # Preprocessing tags
      mlperf.tags.INPUT_ORDER: 2,  # 1 for training, 1 for eval
      # We pass --tf_random_seed=9876 in the test.
      r'%s: 9876' % mlperf.tags.RUN_SET_RANDOM_SEED: 2,
      # The Numpy random seed is hardcoded to 4321.
      r'%s: 4321' % mlperf.tags.RUN_SET_RANDOM_SEED: 2,
      r'%s: %d' % (mlperf.tags.PREPROC_NUM_TRAIN_EXAMPLES,
                   datasets.IMAGENET_NUM_TRAIN_IMAGES): 1,
      r'%s: %d' % (mlperf.tags.PREPROC_NUM_EVAL_EXAMPLES,
                   datasets.IMAGENET_NUM_VAL_IMAGES): 1,
      mlperf.tags.PREPROC_NUM_EVAL_EXAMPLES + '.*': 1,
      mlperf.tags.INPUT_DISTORTED_CROP_MIN_OBJ_COV + '.*': 1,
      mlperf.tags.INPUT_DISTORTED_CROP_RATIO_RANGE + '.*': 1,
      mlperf.tags.INPUT_DISTORTED_CROP_AREA_RANGE + '.*': 1,
      mlperf.tags.INPUT_DISTORTED_CROP_MAX_ATTEMPTS + '.*': 1,
      mlperf.tags.INPUT_RANDOM_FLIP + '.*': 1,
      r'%s: \[224, 224\].*' % mlperf.tags.INPUT_CENTRAL_CROP: 1,

      r'%s: \[123.68, 116.78, 103.94\].*' % mlperf.tags.INPUT_MEAN_SUBTRACTION:
          2,

      r'%s: {"min": 256}.*' % mlperf.tags.INPUT_RESIZE_ASPECT_PRESERVING: 1,

      # 1 for training, 1 for eval
      r'%s: \[224, 224\].*' % mlperf.tags.INPUT_RESIZE: 2,

      # Resnet model tags
      mlperf.tags.MODEL_HP_BATCH_NORM + '.*': 2,
      # 2 for training, 2 for eval. Although there's only 1 conv2d, each conv2d
      # produces 2 logs.
      mlperf.tags.MODEL_HP_CONV2D_FIXED_PADDING + '.*': 4,
      mlperf.tags.MODEL_HP_RELU + '.*': 2,
      mlperf.tags.MODEL_HP_INITIAL_MAX_POOL + '.*': 2,
      mlperf.tags.MODEL_HP_DENSE + '.*': 4,
      mlperf.tags.MODEL_HP_DENSE + '.*': 4,

      # Note that tags our test model does not emit, like MODEL_HP_SHORTCUT_ADD,
      # are omitted here.

      r'%s: "categorical_cross_entropy".*' % mlperf.tags.MODEL_HP_LOSS_FN: 1,

      # 1 for training, 2 because the _MlPerfTestModel calls this when building
      # the model for both training and eval
      r'%s: true' % mlperf.tags.MODEL_EXCLUDE_BN_FROM_L2: 3,

      r'%s: 0.5.*' % mlperf.tags.MODEL_L2_REGULARIZATION: 1,

      # Note we do not handle OPT_LR, since that is printed to stderr using
      # tf.Print, which we cannot easily intercept.

      # Other tags
      '%s: "%s"' % (mlperf.tags.OPT_NAME, mlperf.tags.SGD_WITH_MOMENTUM): 1,
      '%s: 0.5' % mlperf.tags.OPT_MOMENTUM: 1,
      mlperf.tags.RUN_START: 1,
      '%s: 2' % mlperf.tags.INPUT_BATCH_SIZE: 1,
      mlperf.tags.TRAIN_LOOP: 1,
      mlperf.tags.TRAIN_EPOCH + '.*': 1,
      '%s: 2' % mlperf.tags.INPUT_SIZE: 2,
      mlperf.tags.EVAL_START: 2,
      mlperf.tags.EVAL_STOP: 2,
      '%s: 6' % mlperf.tags.EVAL_SIZE: 2,
      mlperf.tags.EVAL_ACCURACY + '.*': 2,
      '%s: 2.0' % mlperf.tags.EVAL_TARGET: 2,
      mlperf.tags.RUN_STOP + '.*': 1,
      mlperf.tags.RUN_FINAL: 1
  }
  EXPECTED_LOG_REGEXES = Counter({re.compile(k): v for
                                  k, v in EXPECTED_LOG_REGEXES.items()})

  def testMlPerfCompliance(self):
    string_io = six.StringIO()
    handler = logging.StreamHandler(string_io)
    data_dir = test_util.create_black_and_white_images()
    try:
      mlperf_log.LOGGER.addHandler(handler)
      params = benchmark_cnn.make_params(data_dir=data_dir,
                                         data_name='imagenet',
                                         batch_size=2,
                                         num_warmup_batches=0,
                                         num_batches=2,
                                         num_eval_batches=3,
                                         eval_during_training_every_n_steps=1,
                                         distortions=False,
                                         weight_decay=0.5,
                                         optimizer='momentum',
                                         momentum=0.5,
                                         stop_at_top_1_accuracy=2.0,
                                         tf_random_seed=9876,
                                         ml_perf=True)
      with mlperf.mlperf_logger(use_mlperf_logger=True, model='resnet50_v1.5'):
        bench_cnn = benchmark_cnn.BenchmarkCNN(params, model=_MlPerfTestModel())
        bench_cnn.run()
      logs = string_io.getvalue().splitlines()
      log_regexes = Counter()
      for log in logs:
        for regex in self.EXPECTED_LOG_REGEXES:
          if regex.search(log):
            log_regexes[regex] += 1
      if log_regexes != self.EXPECTED_LOG_REGEXES:
        diff_counter = Counter(log_regexes)
        diff_counter.subtract(self.EXPECTED_LOG_REGEXES)
        differences = []
        for regex in (k for k in diff_counter.keys() if diff_counter[k]):
          found_count = log_regexes[regex]
          expected_count = self.EXPECTED_LOG_REGEXES[regex]
          differences.append('  For regex %s: Found %d lines matching but '
                             'expected to find %d' %
                             (regex.pattern, found_count, expected_count))
        raise AssertionError('Logs did not match expected logs. Differences:\n'
                             '%s' % '\n'.join(differences))
    finally:
      mlperf_log.LOGGER.removeHandler(handler)

if __name__ == '__main__':
  tf.test.main()
