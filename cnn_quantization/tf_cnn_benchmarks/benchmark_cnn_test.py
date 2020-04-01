# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Tests for benchmark_cnn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import os
import re

import mock
import numpy as np
import tensorflow.compat.v1 as tf
from google.protobuf import text_format
from cnn_quantization.tf_cnn_benchmarks import benchmark_cnn
from cnn_quantization.tf_cnn_benchmarks import datasets
from cnn_quantization.tf_cnn_benchmarks import flags
from cnn_quantization.tf_cnn_benchmarks import preprocessing
from cnn_quantization.tf_cnn_benchmarks import test_util
from cnn_quantization.tf_cnn_benchmarks import variable_mgr_util
from cnn_quantization.tf_cnn_benchmarks.platforms import util as platforms_util
from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.profiler import tfprof_log_pb2
from tensorflow.python.platform import test


def _check_has_gpu():
  if not test.is_gpu_available(cuda_only=True):
    raise ValueError(
        """You have asked to run part or all of this on GPU, but it appears
        that no GPU is available. If your machine has GPUs it is possible you
        do not have a version of TensorFlow with GPU support. To build with GPU
        support, add --config=cuda to the build flags.\n """)


class TfCnnBenchmarksModelTest(tf.test.TestCase):
  """Tests which are run with multiple models."""

  def setUp(self):
    super(TfCnnBenchmarksModelTest, self).setUp()
    benchmark_cnn.setup(benchmark_cnn.make_params())

  def get_model_name(self):
    return None

  # Return true to run tests that don't need to be run on every model.
  # This should be done for one or two cheap models.
  def extended_tests(self):
    return False

  # Return false to suppress actually running the model; this is useful
  # for tests that are large.
  def model_execution_test(self):
    return False

  # Return false to suppress actually saving and loading the model.
  def model_save_load_test(self):
    return False

  def testSaveLoadModel(self):
    _check_has_gpu()
    if not self.get_model_name() or not self.model_save_load_test():
      return

    params = benchmark_cnn.make_params(
        model=self.get_model_name(),
        num_batches=1,
        num_intra_threads=0,
        num_inter_threads=0,
        distortions=False,
        batch_size=2,
        variable_update='replicated',
        num_warmup_batches=0,
        num_gpus=2,
        train_dir=test_util.get_temp_dir('testSaveLoadModel_' +
                                         self.get_model_name()))

    # Run one batch and save the model.
    # Note that this uses a non-test session.
    bench = benchmark_cnn.BenchmarkCNN(params)
    bench.run()
    self.assertEquals(bench.init_global_step, 0)
    # Clear the default graph.
    tf.reset_default_graph()
    # Test if checkpoint had been saved.
    ckpt = tf.train.get_checkpoint_state(params.train_dir)
    match = re.match(os.path.join(params.train_dir, r'model.ckpt-(\d+).index'),
                     ckpt.model_checkpoint_path + '.index')
    self.assertTrue(match)
    self.assertGreaterEqual(int(match.group(1)), params.num_batches)
    params = params._replace(num_batches=2)
    # Reload the model
    bench = benchmark_cnn.BenchmarkCNN(params)
    bench.run()
    # Check if global step has been restored.
    self.assertNotEquals(bench.init_global_step, 0)
    ckpt = tf.train.get_checkpoint_state(params.train_dir)
    match = re.match(os.path.join(params.train_dir, r'model.ckpt-(\d+).index'),
                     ckpt.model_checkpoint_path + '.index')
    self.assertTrue(match)
    self.assertGreaterEqual(int(match.group(1)), params.num_batches)
    # Check that the batch norm moving averages are restored from checkpoints
    with tf.Graph().as_default():
      bench = benchmark_cnn.BenchmarkCNN(params)
      bench._build_model()
      saver = tf.train.Saver(bench.variable_mgr.savable_variables())
      with tf.Session(config=benchmark_cnn.create_config_proto(params)) as sess:
        benchmark_cnn.load_checkpoint(saver, sess, params.train_dir)
        sess.run(bench.variable_mgr.get_post_init_ops())
        bn_moving_vars = [
            v for v in tf.global_variables()
            if '/batchnorm' in v.name and '/moving' in v.name
        ]
        self.assertGreater(len(bn_moving_vars), 0)
        for moving_var in bn_moving_vars:
          moving_var_value = sess.run(moving_var)
          # Check that the moving means and moving variances have been restored
          # by asserting they are not their default values of 0 and 1,
          # respectively
          if '/moving_mean' in moving_var.name:
            self.assertFalse(np.array_equal(moving_var_value,
                                            np.zeros(moving_var_value.shape,
                                                     moving_var_value.dtype)))
          else:
            self.assertIn('/moving_variance', moving_var.name)
            self.assertFalse(np.array_equal(moving_var_value,
                                            np.ones(moving_var_value.shape,
                                                    moving_var_value.dtype)))

  def testModel(self):
    _check_has_gpu()
    if not self.get_model_name() or not self.model_execution_test():
      return

    params = benchmark_cnn.make_params(
        model=self.get_model_name(),
        num_batches=1,
        num_intra_threads=1,
        num_inter_threads=12,
        batch_size=2,
        distortions=False)

    # Run this one; note that this uses a non-test session.
    bench = benchmark_cnn.BenchmarkCNN(params)
    bench.run()

  def testSendRecvVariables(self):
    self._testVariables('parameter_server')
    if self.extended_tests():
      self._testVariables('parameter_server', local_parameter_device='CPU')
      self._testVariables('parameter_server', optimizer='sgd')

  def testReplicatedVariables(self):
    self._testVariables('replicated')
    if self.extended_tests():
      self._testVariables('replicated', all_reduce_spec=None)
      self._testVariables('replicated', use_fp16=True, fp16_vars=False)
      self._testVariables(
          'replicated',
          all_reduce_spec=None,
          use_fp16=True,
          fp16_vars=False,
          fp16_enable_auto_loss_scale=True,
          fp16_inc_loss_scale_every_n=4)

  def testIndependentVariables(self):
    self._testVariables('independent')
    self._testVariables(
        'independent',
        all_reduce_spec=None,
        use_fp16=True,
        fp16_vars=False,
        fp16_enable_auto_loss_scale=True,
        fp16_inc_loss_scale_every_n=4)

  def testSummaryVerbosity(self):
    self._testVariables('parameter_server', summary_verbosity=1)
    if self.extended_tests():
      self._testVariables('parameter_server', summary_verbosity=2)
      self._testVariables('parameter_server', summary_verbosity=3)

  def testStagedVariables(self):
    self._testVariables('parameter_server', staged_vars=True)
    if self.extended_tests():
      self._testVariables('parameter_server', staged_vars=True,
                          local_parameter_device='CPU')
      self._testVariables('parameter_server', staged_vars=True, use_fp16=True,
                          fp16_vars=True)

  def _assert_correct_var_type(self, var, params):
    if 'gpu_cached_inputs' not in var.name:
      if params.use_fp16 and params.fp16_vars and 'batchnorm' not in var.name:
        expected_type = tf.float16
      else:
        expected_type = tf.float32
      self.assertEqual(var.dtype.base_dtype, expected_type)

  def _testVariables(self,
                     variable_update,
                     summary_verbosity=0,
                     local_parameter_device='GPU',
                     staged_vars=False,
                     optimizer='momentum',
                     # TODO(b/80125832): Enable nccl in tests
                     # all_reduce_spec='nccl',
                     all_reduce_spec='',
                     use_fp16=False,
                     fp16_vars=False,
                     fp16_enable_auto_loss_scale=False,
                     fp16_inc_loss_scale_every_n=10):
    if not self.get_model_name():
      return
    _check_has_gpu()

    params = benchmark_cnn.make_params(
        model=self.get_model_name(),
        num_batches=1,
        num_intra_threads=1,
        num_inter_threads=12,
        distortions=False,
        variable_update=variable_update,
        local_parameter_device=local_parameter_device,
        num_gpus=2,
        summary_verbosity=summary_verbosity,
        staged_vars=staged_vars,
        optimizer=optimizer,
        all_reduce_spec=all_reduce_spec,
        compact_gradient_transfer=False if all_reduce_spec == 'nccl' else True,
        use_fp16=use_fp16,
        fp16_loss_scale=2.,
        fp16_vars=fp16_vars,
        fp16_enable_auto_loss_scale=fp16_enable_auto_loss_scale,
        fp16_inc_loss_scale_every_n=fp16_inc_loss_scale_every_n,
    )

    # Test building models using multiple GPUs, but don't
    # run them.
    with self.test_session(graph=tf.Graph()):
      bench = benchmark_cnn.BenchmarkCNN(params)
      bench._build_model()

      # Rough validation of variable type and placement, depending on mode.
      all_vars = tf.global_variables() + tf.local_variables()
      if params.variable_update == 'parameter_server':
        for v in all_vars:
          tf.logging.debug('var: %s' % v.name)
          match = re.match(r'tower_(\d+)/v/gpu_cached_inputs:0', v.name)
          if match:
            self.assertEquals(v.device, '/device:GPU:%s' % match.group(1))
          elif v.name.startswith('v/'):
            self.assertEquals(v.device,
                              '/device:%s:0' % local_parameter_device)
            self._assert_correct_var_type(v, params)
          elif v.name in ('input_processing/images:0',
                          'input_processing/labels:0', 'init_learning_rate:0',
                          'global_step:0', 'loss_scale:0',
                          'loss_scale_normal_steps:0'):
            self.assertEquals(v.device, '/device:CPU:0')
          else:
            raise ValueError('Unexpected variable %s' % v.name)
      else:
        v0_count = 0
        v1_count = 0
        for v in all_vars:
          if v.name.startswith('tower_0/v0/'):
            self.assertEquals(v.name, 'tower_0/v0/gpu_cached_inputs:0')
            self.assertEquals(v.device, '/device:GPU:0')
          elif v.name.startswith('tower_1/v1/'):
            self.assertEquals(v.name, 'tower_1/v1/gpu_cached_inputs:0')
            self.assertEquals(v.device, '/device:GPU:1')
          elif v.name.startswith('v0/'):
            v0_count += 1
            self.assertEquals(v.device, '/device:GPU:0')
            self._assert_correct_var_type(v, params)
          elif v.name.startswith('v1/'):
            v1_count += 1
            self.assertEquals(v.device, '/device:GPU:1')
            self._assert_correct_var_type(v, params)
          elif v.name in ('input_processing/images:0',
                          'input_processing/labels:0', 'init_learning_rate:0',
                          'global_step:0', 'loss_scale:0',
                          'loss_scale_normal_steps:0'):
            self.assertEquals(v.device, '/device:CPU:0')
          else:
            raise ValueError('Unexpected variable %s' % v.name)
        self.assertEquals(v0_count, v1_count)

      # Validate summary ops in the model depending on verbosity level
      summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
      num_summary_ops = len(summary_ops)
      self.assertEquals(num_summary_ops > 0, summary_verbosity > 0)
      if summary_verbosity > 0:
        has_affine_histogram = False
        has_gradient_histogram = False
        has_log_gradients_histogram = False
        for op in summary_ops:
          if '/gradients' in op.name:
            has_gradient_histogram = True
          elif '/affine' in op.name:
            has_affine_histogram = True
          elif 'log_gradients' in op.name:
            has_log_gradients_histogram = True
        self.assertEqual(summary_verbosity >= 3, has_affine_histogram)
        self.assertEqual(summary_verbosity >= 3, has_gradient_histogram)
        self.assertEqual(summary_verbosity >= 2, has_log_gradients_histogram)
        if summary_verbosity == 1:
          self.assertLess(num_summary_ops, 10)


class TrivialModelTest(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'trivial'


class TestVgg1Model(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'vgg11'


class TestVgg19Model(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'vgg19'


class TestLenet5Model(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'lenet'


class TestGooglenetModel(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'googlenet'


class TestOverfeatModel(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'overfeat'


class TestAlexnetModel(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'alexnet'

  def extended_tests(self):
    return True


class TestTrivialModel(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'trivial'


class TestInceptionv3Model(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'inception3'

  def extended_tests(self):
    return True


class TestInceptionv4Model(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'inception4'


class TestResnet50Model(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'resnet50'

  def model_save_load_test(self):
    return True


class TestResnet101Model(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'resnet101'


class TestResnet152Model(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'resnet152'


class TestResnet50V2Model(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'resnet50_v2'


class TestResnet101V2Model(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'resnet101_v2'


class TestResnet152V2Model(TfCnnBenchmarksModelTest):

  def get_model_name(self):
    return 'resnet152_v2'


class TfCnnBenchmarksTest(tf.test.TestCase):
  """Tests that benchmark_cnn runs correctly."""

  def setUp(self):
    super(TfCnnBenchmarksTest, self).setUp()
    _check_has_gpu()
    benchmark_cnn.setup(benchmark_cnn.make_params())

  def _run_benchmark_cnn(self, params):
    logs = []
    benchmark_cnn.log_fn = test_util.print_and_add_to_list(logs)
    benchmark_cnn.BenchmarkCNN(params).run()
    return logs

  def _run_benchmark_cnn_with_fake_images(self, params, images, labels):
    logs = []
    benchmark_cnn.log_fn = test_util.print_and_add_to_list(logs)
    bench = benchmark_cnn.BenchmarkCNN(params)
    bench.input_preprocessor = preprocessing.TestImagePreprocessor(
        params.batch_size * params.num_gpus,
        [[params.batch_size, 227, 227, 3], [params.batch_size]],
        params.num_gpus,
        bench.model.data_type)
    bench.dataset._queue_runner_required = True
    bench.input_preprocessor.set_fake_data(images, labels)
    bench.input_preprocessor.expected_subset = (
        'validation' if params.eval else 'train')
    bench.run()
    return logs

  def _run_benchmark_cnn_with_black_and_white_images(self, params):
    """Runs BenchmarkCNN with black and white images.

    A BenchmarkCNN is created and run with black and white images as input. Half
    the images are black (i.e., filled with 0s) and half are white (i.e., filled
    with 255s).

    Args:
      params: Params for BenchmarkCNN.

    Returns:
      A list of lines from the output of BenchmarkCNN.
    """
    # TODO(reedwm): Instead of generating images here, use black and white
    # tfrecords by calling test_util.create_black_and_white_images().
    effective_batch_size = params.batch_size * params.num_gpus
    half_batch_size = effective_batch_size // 2
    images = np.zeros((effective_batch_size, 227, 227, 3), dtype=np.float32)
    images[half_batch_size:, :, :, :] = 255
    labels = np.array([0] * half_batch_size + [1] * half_batch_size,
                      dtype=np.int32)
    return self._run_benchmark_cnn_with_fake_images(params, images, labels)

  def _train_and_eval_local(self,
                            params,
                            check_output_values=False,
                            max_final_loss=10.,
                            skip=None,
                            use_test_preprocessor=True):
    # TODO(reedwm): check_output_values should default to True and be enabled
    # on every test. Currently, if check_output_values=True and the calls to
    # tf.set_random_seed(...) and np.seed(...) are passed certain seed values in
    # benchmark_cnn.py, then most tests will fail. This indicates the tests
    # are brittle and could fail with small changes when
    # check_output_values=True, so check_output_values defaults to False for
    # now.

    def run_fn(run_type, inner_params):
      del run_type
      if use_test_preprocessor:
        return [
            self._run_benchmark_cnn_with_black_and_white_images(inner_params)
        ]
      else:
        return [self._run_benchmark_cnn(inner_params)]

    return test_util.train_and_eval(self, run_fn, params,
                                    check_output_values=check_output_values,
                                    max_final_loss=max_final_loss,
                                    skip=skip)

  def testAlexnet(self):
    params = test_util.get_params('testAlexnet')._replace(
        num_batches=30, init_learning_rate=0.01, model='alexnet')
    self._train_and_eval_local(params)

  def testNoPrintAccuracy(self):
    params = test_util.get_params('testNoPrintAccuracy')._replace(
        print_training_accuracy=False)
    self._train_and_eval_local(params)

  def testLowAccuracy(self):
    params = test_util.get_params('testLowAccuracy')._replace(
        print_training_accuracy=True, batch_size=5, num_batches=10)
    # We force low accuracy by having each batch containing 10 identical images,
    # each with a different label. This guarantees a top-1 accuracy of exactly
    # 0.1 and a top-5 accuracy of exactly 0.5.
    images = np.zeros((10, 227, 227, 3), dtype=np.float32)
    labels = np.arange(10, dtype=np.int32)
    logs = self._run_benchmark_cnn_with_fake_images(params, images, labels)
    training_outputs = test_util.get_training_outputs_from_logs(
        logs, params.print_training_accuracy)
    last_output = training_outputs[-1]
    # TODO(reedwm): These should be assertEqual but for some reason,
    # occasionally the accuracies are lower (Running this test 500 times, these
    # asserts failed twice). Investigate this problem.
    self.assertLessEqual(last_output.top_1_accuracy, 0.1)
    self.assertLessEqual(last_output.top_5_accuracy, 0.5)

  def testParameterServer(self):
    params = test_util.get_params('testParameterServer')
    self._train_and_eval_local(params)

  def testParameterServerStaged(self):
    params = test_util.get_params('testParameterServerStaged')._replace(
        staged_vars=True)
    self._train_and_eval_local(params)

  def testReplicated(self):
    params = test_util.get_params('testReplicated')._replace(
        variable_update='replicated')
    self._train_and_eval_local(params)

  def testIndependent(self):
    params = test_util.get_params('testIndependent')._replace(
        variable_update='independent')
    self._train_and_eval_local(params)

  def testForwardOnly(self):
    params = test_util.get_params('testForwardOnly')._replace(forward_only=True)
    # Evaluation is not supported with --forward_only, so we set skip='eval'.
    self._train_and_eval_local(params, skip='eval')

  def testForwardOnlyAndFreeze(self):
    params = test_util.get_params('testForwardOnlyAndFreeze')._replace(
        forward_only=True, freeze_when_forward_only=True, train_dir=None)
    # Training is not supported with --freeze_when_forward_only.
    self._train_and_eval_local(params, skip='eval_and_train_from_checkpoint')

  def testNoDistortions(self):
    params = test_util.get_params('testNoDistortions')._replace(
        distortions=False)
    self._train_and_eval_local(params)

  def testCpuAsLocalParamDevice(self):
    params = test_util.get_params('testCpuAsLocalParamDevice')._replace(
        local_parameter_device='cpu')
    self._train_and_eval_local(params)

  def testNHWC(self):
    params = test_util.get_params('testNHWC')._replace(data_format='NHWC')
    self._train_and_eval_local(params)

  def testCpuAsDevice(self):
    params = test_util.get_params('testCpuAsDevice')._replace(
        device='cpu', data_format='NHWC')  # NHWC required when --device=cpu
    self._train_and_eval_local(params)

  def testMomentumParameterServer(self):
    params = test_util.get_params('testMomentumParameterServer')._replace(
        optimizer='momentum', momentum=0.8)
    self._train_and_eval_local(params)

  def testRmspropReplicated(self):
    params = test_util.get_params('testRmspropReplicated')._replace(
        variable_update='replicated',
        optimizer='rmsprop',
        rmsprop_decay=0.8,
        rmsprop_momentum=0.6,
        rmsprop_epsilon=0.7,
        init_learning_rate=0.01)
    self._train_and_eval_local(params)

  def testBatchGroupSize(self):
    params = test_util.get_params('testBatchGroupSize')._replace(
        batch_group_size=4, num_batches=100, num_warmup_batches=5)
    self._train_and_eval_local(params)

  def testGradientClip(self):
    params = test_util.get_params('testGradientClip')._replace(
        gradient_clip=100.0)
    self._train_and_eval_local(params)

  def testWeightDecay(self):
    params = test_util.get_params('testWeightDecay')._replace(
        weight_decay=0.0001)
    self._train_and_eval_local(params)

  def testNoLayers(self):
    params = test_util.get_params('testNoLayers')._replace(use_tf_layers=False)
    self._train_and_eval_local(params)

  def testSaveModelSteps(self):
    params = test_util.get_params('testSaveModelSteps')._replace(
        save_model_steps=2, num_warmup_batches=0, num_batches=10,
        max_ckpts_to_keep=3)
    self._train_and_eval_local(params)
    for i in range(1, 20 + 1):
      # We train for 20 steps, since self._train_and_eval_local() does two
      # training runs of 10 steps each. We save a checkpoint every 2 steps and
      # keep the last 3 checkpoints, so at the end, we should have checkpoints
      # for steps 16, 18, and 20.
      matches = glob.glob(os.path.join(params.train_dir,
                                       'model.ckpt-{}.*'.format(i)))
      if i in (16, 18, 20):
        self.assertTrue(matches)
      else:
        self.assertFalse(matches)

  def testFp16WithFp32Vars(self):
    params = test_util.get_params('testFp16WithFp32Vars')._replace(
        use_fp16=True, fp16_vars=False, fp16_loss_scale=1.)
    self._train_and_eval_local(params)

  def testFp16WithFp16Vars(self):
    params = test_util.get_params('testFp16WithFp16Vars')._replace(
        use_fp16=True, fp16_vars=True)
    self._train_and_eval_local(params)

  def testXlaCompile(self):
    params = test_util.get_params('testXlaCompile')._replace(xla_compile=True)
    self._train_and_eval_local(params)

  def testXlaCompileWithFp16(self):
    params = test_util.get_params('testXlaCompileWithFp16')._replace(
        use_fp16=True, xla_compile=True)
    self._train_and_eval_local(params)

  def testGradientRepacking(self):
    params = test_util.get_params('testGradientRepacking1')._replace(
        gradient_repacking=2)
    self._train_and_eval_local(params, skip='eval_and_train_from_checkpoint')
    params = test_util.get_params('testGradientRepacking2')._replace(
        gradient_repacking=2, use_fp16=True)
    self._train_and_eval_local(params, skip='eval_and_train_from_checkpoint')

  def testTraceFileChromeTraceFormat(self):
    trace_file = os.path.join(self.get_temp_dir(),
                              'testTraceFileChromeTraceFormat_tracefile')
    params = test_util.get_params('testTraceFileChromeTraceFormat')._replace(
        trace_file=trace_file, use_chrome_trace_format=True)
    self._train_and_eval_local(params)
    self.assertGreater(os.stat(trace_file).st_size, 0)

  def testTraceFileStepStatsProto(self):
    trace_file = os.path.join(self.get_temp_dir(),
                              'testTraceFileStepStatsProto_tracefile')
    params = test_util.get_params('testTraceFileStepStatsProto')._replace(
        trace_file=trace_file, use_chrome_trace_format=False)
    self._train_and_eval_local(params)
    self.assertGreater(os.stat(trace_file).st_size, 0)
    with open(trace_file) as f:
      step_stats = step_stats_pb2.StepStats()
      # The following statement should not raise an exception.
      contents = f.read()
      text_format.Merge(contents, step_stats)

  def testTfprofFile(self):
    tfprof_file = os.path.join(self.get_temp_dir(), 'testTfprofFile_tfproffile')
    params = test_util.get_params('testTfprofFile')._replace(
        tfprof_file=tfprof_file)
    self._train_and_eval_local(params, skip='eval_and_train_from_checkpoint')
    self.assertGreater(os.stat(tfprof_file).st_size, 0)
    with open(tfprof_file, 'rb') as f:
      profile_proto = tfprof_log_pb2.ProfileProto()
      # The following statement should not raise an exception.
      profile_proto.ParseFromString(f.read())

  def testMoveTrainDir(self):
    params = test_util.get_params('testMoveTrainDir')
    self._train_and_eval_local(params)
    new_train_dir = params.train_dir + '_moved'
    os.rename(params.train_dir, new_train_dir)
    params = params._replace(train_dir=new_train_dir, eval=True)
    self._run_benchmark_cnn_with_black_and_white_images(params)

  @mock.patch.object(tf.train, 'Saver')
  @mock.patch('benchmark_cnn._get_checkpoint_to_load')  # pylint: disable=line-too-long
  def testLoadCheckpoint(self, mock_checkpoint_to_load, mock_saver):
    """Tests load checkpoint with full path to checkpoint."""
    expected_checkpoint = '/path/to/checkpoints/model.ckpt-1243'
    mock_checkpoint_to_load.return_value = expected_checkpoint

    global_batch = benchmark_cnn.load_checkpoint(mock_saver,
                                                 None,
                                                 expected_checkpoint)
    self.assertEqual(global_batch, 1243)

  def testGetCheckpointToLoadFullPath(self):
    """Tests passing full path."""
    ckpt_path = '/foo/bar/model.ckpt-189'
    full_path = benchmark_cnn._get_checkpoint_to_load(ckpt_path)
    self.assertEqual(full_path, ckpt_path)

  def testGetCheckpointToLoadException(self):
    """Tests exception for directory without a checkpoint."""
    ckpt_path = '/foo/bar/checkpoints'
    self.assertRaises(benchmark_cnn.CheckpointNotFoundException,
                      benchmark_cnn._get_checkpoint_to_load, ckpt_path)

  @mock.patch.object(tf.train, 'get_checkpoint_state')
  def testGetCheckpointToLoad(self, mock_checkpoint_state):
    """Tests passing path to checkpoint folder."""
    expected_checkpoint = '/path/to/checkpoints/model.ckpt-1243'
    mock_checkpoint_state.return_value = mock.Mock(
        model_checkpoint_path=expected_checkpoint)
    ckpt_path = '/path/to/checkpoints/'
    full_path = benchmark_cnn._get_checkpoint_to_load(ckpt_path)
    self.assertEqual(full_path, expected_checkpoint)

  def testImagenetPreprocessor(self):
    imagenet_dir = os.path.join(platforms_util.get_test_data_dir(),
                                'fake_tf_record_data')
    params = test_util.get_params('testImagenetPreprocessor')._replace(
        data_dir=imagenet_dir, data_name='imagenet')
    self._train_and_eval_local(params, use_test_preprocessor=False)

  def testImagenetPreprocessorNoDistortions(self):
    imagenet_dir = os.path.join(platforms_util.get_test_data_dir(),
                                'fake_tf_record_data')
    params = test_util.get_params(
        'testImagenetPreprocessorNoDistortions')._replace(
            data_dir=imagenet_dir, data_name='imagenet', distortions=False)
    self._train_and_eval_local(params, use_test_preprocessor=False)

  def testImagenetPreprocessorVerboseSummary(self):
    imagenet_dir = os.path.join(platforms_util.get_test_data_dir(),
                                'fake_tf_record_data')
    params = test_util.get_params(
        'testImagenetPreprocessorVerboseSummary')._replace(
            data_dir=imagenet_dir, data_name='imagenet', distortions=False,
            summary_verbosity=2)
    self._train_and_eval_local(params, use_test_preprocessor=False)

  def testCifar10SyntheticData(self):
    params = test_util.get_params('testCifar10SyntheticData')._replace(
        data_name='cifar10')
    self._train_and_eval_local(params)

  def testShiftRatio(self):
    test_util.monkey_patch_base_cluster_manager()
    params = benchmark_cnn.make_params(
        data_name='imagenet',
        data_dir=os.path.join(platforms_util.get_test_data_dir(),
                              'fake_tf_record_data'),
        job_name='worker',
        worker_hosts='w1,w2,w3,w4',
        ps_hosts='p1',
        task_index=0)
    self.assertEqual(
        benchmark_cnn.BenchmarkCNN(params).input_preprocessor.shift_ratio, 0.0)
    params = params._replace(task_index=3)
    self.assertEqual(
        benchmark_cnn.BenchmarkCNN(params).input_preprocessor.shift_ratio, 0.75)

  def testDistributedReplicatedSavableVars(self):
    test_util.monkey_patch_base_cluster_manager()
    params = benchmark_cnn.make_params(
        variable_update='distributed_replicated',
        model='inception4',
        data_name='imagenet',
        data_dir=os.path.join(platforms_util.get_test_data_dir(),
                              'fake_tf_record_data'),
        job_name='worker',
        worker_hosts='w1,w2,w3,w4',
        ps_hosts='p1',
        datasets_use_prefetch=False)

    bench = benchmark_cnn.BenchmarkCNN(params)
    with tf.Graph().as_default():
      bench._build_model()
      savable_vars = bench.variable_mgr.savable_variables()
      # Assert all global variables are in savable_vars
      for v in tf.global_variables():
        if not v.name.startswith(
            variable_mgr_util.PS_SHADOW_VAR_PREFIX + '/v0'):
          self.assertEqual(v.name, 'global_step:0')
        name = bench.variable_mgr._strip_port(v.name)
        if name.startswith(variable_mgr_util.PS_SHADOW_VAR_PREFIX):
          name = name[len(variable_mgr_util.PS_SHADOW_VAR_PREFIX + '/'):]
        self.assertIn(name, savable_vars)
        self.assertIn(savable_vars[name], tf.global_variables())
      # Assert all local variables on the first tower are in savable_vars
      for v in tf.local_variables():
        if v.name.startswith('v0/'):
          name = bench.variable_mgr._strip_port(v.name)
          self.assertIn(name, savable_vars)

  def _test_preprocessing_eval(self, image_height, image_width, output_height,
                               output_width):
    image = tf.fill((image_height, image_width, 3),
                    tf.constant(128, dtype=tf.uint8))
    params = benchmark_cnn.make_params()
    new_image = preprocessing.eval_image(image, output_height, output_width, 0,
                                         'bilinear', params.summary_verbosity)
    with self.test_session() as sess:
      new_image_value = sess.run(new_image)
    self.assertAllEqual(new_image_value,
                        np.full((output_height, output_width, 3), 128,
                                dtype=np.uint8))

  def testPreprocessingEval(self):
    self._test_preprocessing_eval(10, 10, 4, 4)
    self._test_preprocessing_eval(4, 4, 10, 10)
    self._test_preprocessing_eval(1, 100, 100, 1)
    self._test_preprocessing_eval(100, 1, 1, 100)
    self._test_preprocessing_eval(1, 100, 1, 100)

  def _test_preprocessing_traing(self, image_buf, image_color,
                                 output_height, output_width, bbox,
                                 batch_position, resize_method, distortions,
                                 summary_verbosity, fuse_decode_and_crop):
    new_image = preprocessing.train_image(
        image_buf,
        output_height,
        output_width,
        bbox,
        batch_position,
        resize_method,
        distortions,
        summary_verbosity=summary_verbosity,
        fuse_decode_and_crop=fuse_decode_and_crop)
    self.assertEqual(new_image.shape, [output_height, output_width, 3])
    with self.test_session(use_gpu=True) as sess:
      new_image_value = sess.run(new_image)
    self.assertAllClose(
        new_image_value,
        np.full(
            [output_height, output_width, 3],
            image_color,
            dtype=np.float32),
        atol=50.,
        rtol=0.)

  def testPreprocessingTrain(self):
    test_data_dir = os.path.join(platforms_util.get_test_data_dir(), 'images')
    black_file = os.path.join(test_data_dir, 'black_image.jpg')
    with open(black_file, 'rb') as f:
      black_jpg_buffer = f.read()
    white_file = os.path.join(test_data_dir, 'white_image.jpg')
    with open(white_file, 'rb') as f:
      white_jpg_buffer = f.read()
    bbox = tf.zeros((1, 0, 4), dtype=tf.float32)
    batch_position = 0
    # Each size config is (output_height, output_width, resize_method)
    size_configs = [(100, 100, 'round_robin'), (150, 10, 'bilinear'),
                    (10, 150, 'nearest')]
    # Each image config is (image_buf, image_color)
    image_configs = [(white_jpg_buffer, 255), (black_jpg_buffer, 0)]
    for (image_buf, image_color) in image_configs:
      for output_height, output_width, resize_method in size_configs:
        for distortions in [True, False]:
          for summary_verbosity in [0, 2]:
            for fuse_decode_and_crop in [True, False]:
              self._test_preprocessing_traing(
                  image_buf, image_color, output_height, output_width, bbox,
                  batch_position, resize_method, distortions, summary_verbosity,
                  fuse_decode_and_crop)

  def _test_learning_rate(self, params, global_step_to_expected_learning_rate):
    self.longMessage = True  # pylint: disable=invalid-name
    bench = benchmark_cnn.BenchmarkCNN(params)
    with tf.Graph().as_default() as graph:
      bench._build_model()
      global_step = graph.get_tensor_by_name('global_step:0')
      learning_rate = graph.get_tensor_by_name('learning_rate_tensor:0')
      with self.test_session(graph=graph, use_gpu=True) as sess:
        items = global_step_to_expected_learning_rate.items()
        for global_step_val, expected_learning_rate in items:
          self.assertAlmostEqual(sess.run(learning_rate,
                                          {global_step: global_step_val}),
                                 expected_learning_rate,
                                 msg='at global_step:{}'.
                                 format(global_step_val))

  def testLearningRateModelSpecificResNet(self):
    params = benchmark_cnn.make_params(model='resnet50',
                                       batch_size=256,
                                       variable_update='parameter_server',
                                       num_gpus=1)
    self._test_learning_rate(params, {
        0: 0,
        150136: 0.128,
        150137: 0.0128,
        300273: 0.0128,
        300274: 0.00128,
        10000000: 0.0000128
    })

  def testLearningRateUserProvidedInitLr(self):
    params = benchmark_cnn.make_params(model='resnet50',
                                       batch_size=256,
                                       variable_update='replicated',
                                       init_learning_rate=1.)
    self._test_learning_rate(params, {
        0: 1.,
        10000000: 1.
    })

  def testLearningRateUserProvidedInitLrAndWarmup(self):
    params = benchmark_cnn.make_params(model='resnet50',
                                       batch_size=256,
                                       variable_update='replicated',
                                       init_learning_rate=1.,
                                       num_learning_rate_warmup_epochs=5)
    self._test_learning_rate(params, {
        0: 0.,
        12511: 0.5,
        25022: 1.,
        10000000: 1.
    })

  def testLearningRateUserProvidedDecayInfo(self):
    params = benchmark_cnn.make_params(model='resnet50',
                                       init_learning_rate=1.,
                                       learning_rate_decay_factor=0.5,
                                       num_epochs_per_decay=2,
                                       minimum_learning_rate=0.3750,
                                       batch_size=32)
    self._test_learning_rate(params, {
        0: 1.,
        80071: 1.,
        80072: 0.5,
        160143: 0.5,
        160144: 0.375,
        10000000: 0.375
    })

  def testLearningRateUserProvidedZeroDecay(self):
    params = benchmark_cnn.make_params(model='resnet50',
                                       num_learning_rate_warmup_epochs=0,
                                       learning_rate_decay_factor=0.5,
                                       num_epochs_per_decay=0,
                                       minimum_learning_rate=0.3750,
                                       batch_size=32)
    with self.assertRaises(ValueError):
      with tf.Graph().as_default():
        # This will fail because params.learning_rate_decay_factor cannot be
        # nonzero if params.num_epochs_per_decay is zero.
        benchmark_cnn.BenchmarkCNN(params)._build_model()

  def testLearningRateUserProvidedSchedule(self):
    params = benchmark_cnn.make_params(
        model='trivial',
        batch_size=32,
        piecewise_learning_rate_schedule='1;3;.1;5;.01')
    self._test_learning_rate(params, {
        0: 1.,
        120108: 1.,
        120109: 0.1,
        200181: 0.1,
        200182: 0.01,
        100000000: 0.01
    })

  def testNumBatchesAndEpochs(self):
    params = benchmark_cnn.make_params()
    batches, epochs = benchmark_cnn.get_num_batches_and_epochs(params, 10, 100)
    self.assertEqual(batches, benchmark_cnn._DEFAULT_NUM_BATCHES)
    self.assertAlmostEqual(epochs,
                           float(benchmark_cnn._DEFAULT_NUM_BATCHES) / 10)

    params = benchmark_cnn.make_params(num_batches=21)
    batches, epochs = benchmark_cnn.get_num_batches_and_epochs(params, 25, 50)
    self.assertEqual(batches, 21)
    self.assertAlmostEqual(epochs, 10.5)

    params = benchmark_cnn.make_params(num_epochs=3)
    batches, epochs = benchmark_cnn.get_num_batches_and_epochs(params, 2, 3)
    self.assertEqual(batches, 5)
    self.assertAlmostEqual(epochs, 10./3.)

    params = benchmark_cnn.make_params(num_epochs=4)
    batches, epochs = benchmark_cnn.get_num_batches_and_epochs(params, 2, 3)
    self.assertEqual(batches, 6)
    self.assertAlmostEqual(epochs, 4)

    with self.assertRaises(ValueError):
      params = benchmark_cnn.make_params(num_batches=100, num_epochs=100)
      benchmark_cnn.get_num_batches_and_epochs(params, 1, 1)

  def _testEvalDuringTraining(self, params, expected_num_eval_batches_found):
    # The idea of this test is that all train images are black and all eval
    # images are white. We pass the images through the TestModel, and ensure
    # the outputs are as expected.

    batch_size = params.batch_size
    eval_batch_size = params.eval_batch_size or params.batch_size

    class TestModel(test_util.TestCNNModel):

      def __init__(self):
        super(TestModel, self).__init__()
        self.depth = 3

      def add_inference(self, cnn):
        if cnn.phase_train:
          # This will allow us to test that 100 is only added during training
          # and not during eval.
          cnn.top_layer += 100
          assert cnn.top_layer.shape[0] == batch_size
        else:
          assert cnn.top_layer.shape[0] == eval_batch_size

        # Reduce the image to a single number. The number should be (-1 + 100)
        # during training and 1 during testing.
        cnn.top_layer = tf.reshape(cnn.top_layer, (cnn.top_layer.shape[0], -1))
        cnn.top_layer = tf.reduce_mean(cnn.top_layer, axis=1)
        cnn.top_layer = tf.reshape(cnn.top_layer,
                                   (cnn.top_layer.shape[0], 1, 1, 1))
        cnn.top_size = 1
        trainable_vars = tf.trainable_variables()

        # The super method will compute image*A*B, where A=1 and B=2.
        super(TestModel, self).add_inference(cnn)

        if not cnn.phase_train:
          # Assert no new variables were added, since they should be reused from
          # training.
          assert len(trainable_vars) == len(tf.trainable_variables())

    model = TestModel()
    dataset = datasets.ImagenetDataset(params.data_dir)
    logs = []
    bench_cnn = benchmark_cnn.BenchmarkCNN(params, model=model, dataset=dataset)
    with test_util.monkey_patch(benchmark_cnn,
                                log_fn=test_util.print_and_add_to_list(logs)):
      bench_cnn.run()
    training_outputs = test_util.get_training_outputs_from_logs(
        logs, print_training_accuracy=False)
    self.assertEqual(len(training_outputs), params.num_batches)
    expected_training_output = (-1 + 100) * 1 * 2
    for training_output in training_outputs:
      self.assertEqual(training_output.loss, expected_training_output)
    eval_outputs = test_util.get_evaluation_outputs_from_logs(logs)
    self.assertTrue(eval_outputs)
    expected_eval_output = 1 * 1 * 2
    for eval_output in eval_outputs:
      self.assertEqual(eval_output.top_1_accuracy, expected_eval_output)
      self.assertEqual(eval_output.top_5_accuracy, expected_eval_output)

    num_eval_batches_found = 0
    eval_batch_regex = re.compile(r'^\d+\t[0-9.]+ examples/sec$')
    for log in logs:
      if eval_batch_regex.match(log):
        num_eval_batches_found += 1
    self.assertEqual(num_eval_batches_found, expected_num_eval_batches_found)

  def testEvalDuringTraining(self):
    data_dir = test_util.create_black_and_white_images()
    base_params = test_util.get_params('testEvalDuringTraining')
    train_dir = base_params.train_dir
    base_params = base_params._replace(
        train_dir=None, print_training_accuracy=False, num_warmup_batches=0,
        num_batches=7, num_eval_batches=2, display_every=1,
        init_learning_rate=0, weight_decay=0,
        distortions=False, data_dir=data_dir)
    expected_num_eval_batches_found = (
        base_params.num_eval_batches * (base_params.num_batches // 2 + 1))

    # Test --eval_during_training_every_n_steps
    self._testEvalDuringTraining(
        base_params._replace(eval_during_training_every_n_steps=2,
                             variable_update='parameter_server'),
        expected_num_eval_batches_found)
    self._testEvalDuringTraining(
        base_params._replace(eval_during_training_every_n_steps=2,
                             variable_update='replicated'),
        expected_num_eval_batches_found)
    self._testEvalDuringTraining(
        base_params._replace(eval_during_training_every_n_steps=2,
                             variable_update='replicated',
                             summary_verbosity=2,
                             save_summaries_steps=2,
                             datasets_use_prefetch=False),
        expected_num_eval_batches_found)
    self._testEvalDuringTraining(
        base_params._replace(eval_during_training_every_n_steps=2,
                             variable_update='replicated',
                             use_fp16=True, train_dir=train_dir,
                             eval_batch_size=base_params.batch_size + 2),
        expected_num_eval_batches_found)

    # Test --eval_during_training_every_n_epochs
    every_n_epochs = (2 * base_params.batch_size * base_params.num_gpus /
                      datasets.IMAGENET_NUM_TRAIN_IMAGES)
    self._testEvalDuringTraining(
        base_params._replace(eval_during_training_every_n_epochs=every_n_epochs,
                             variable_update='replicated'),
        expected_num_eval_batches_found)

    # Test --eval_during_training_at_specified_steps
    list_steps = [2, 3, 5, 7, 1000]
    num_eval_steps = 1 + sum(1 for step in list_steps
                             if step < base_params.num_batches)
    expected_num_eval_batches_found = (
        base_params.num_eval_batches * num_eval_steps)

    self._testEvalDuringTraining(
        base_params._replace(eval_during_training_at_specified_steps=list_steps,
                             variable_update='replicated'),
        expected_num_eval_batches_found)

    # Test --eval_during_training_at_specified_epochs
    list_epochs = [(step * base_params.batch_size * base_params.num_gpus /
                    datasets.IMAGENET_NUM_TRAIN_IMAGES)
                   for step in list_steps]
    self._testEvalDuringTraining(
        base_params._replace(
            eval_during_training_at_specified_epochs=list_epochs,
            variable_update='replicated'),
        expected_num_eval_batches_found)

    # Test --eval_during_training_every_n_steps runs with synthetic data.
    params = base_params._replace(
        variable_update='replicated', data_dir=None,
        eval_during_training_every_n_steps=2, num_batches=2)
    benchmark_cnn.BenchmarkCNN(params).run()

  def testEvalDuringTrainingNumEpochs(self):
    params = benchmark_cnn.make_params(
        batch_size=1, eval_batch_size=2, eval_during_training_every_n_steps=1,
        num_batches=30, num_eval_epochs=100 / datasets.IMAGENET_NUM_VAL_IMAGES)
    bench_cnn = benchmark_cnn.BenchmarkCNN(params)
    self.assertEqual(bench_cnn.num_batches, 30)
    self.assertAlmostEqual(bench_cnn.num_epochs,
                           30 / datasets.IMAGENET_NUM_TRAIN_IMAGES)
    self.assertAlmostEqual(bench_cnn.num_eval_batches, 50)
    self.assertAlmostEqual(bench_cnn.num_eval_epochs,
                           100 / datasets.IMAGENET_NUM_VAL_IMAGES)

  def testEarlyStopping(self):
    params = benchmark_cnn.make_params(
        batch_size=2,
        display_every=1,
        num_batches=100,
        eval_during_training_every_n_steps=2,
        stop_at_top_1_accuracy=0.4,
    )
    with mock.patch.object(benchmark_cnn.BenchmarkCNN, '_eval_once',
                           side_effect=[(0.1, 0.1), (0.5, 0.5), (0.2, 0.2)]
                          ) as mock_eval_once:
      logs = []
      bench_cnn = benchmark_cnn.BenchmarkCNN(params)
      with test_util.monkey_patch(benchmark_cnn,
                                  log_fn=test_util.print_and_add_to_list(logs)):
        bench_cnn.run()
      training_outputs = test_util.get_training_outputs_from_logs(
          logs, print_training_accuracy=False)
      # We should stop after the second evaluation, and we evaluate every 2
      # steps. So there should be 2 * 2 = 4 training outputs.
      self.assertEqual(len(training_outputs), 4)
      self.assertEqual(mock_eval_once.call_count, 2)

  def testOutOfRangeErrorsAreNotIgnored(self):
    error_msg = 'Fake OutOfRangeError error message'
    with mock.patch.object(benchmark_cnn.BenchmarkCNN, 'benchmark_with_session',
                           side_effect=tf.errors.OutOfRangeError(None, None,
                                                                 error_msg)):
      with self.assertRaisesRegexp(RuntimeError, error_msg):
        benchmark_cnn.BenchmarkCNN(benchmark_cnn.make_params()).run()

  def testInvalidFlags(self):
    params = benchmark_cnn.make_params(device='cpu', data_format='NCHW')
    with self.assertRaises(ValueError):
      benchmark_cnn.BenchmarkCNN(params)

    params = benchmark_cnn.make_params(use_fp16=True, fp16_vars=True,
                                       variable_update='replicated',
                                       all_reduce_spec='nccl')
    with self.assertRaises(ValueError):
      benchmark_cnn.BenchmarkCNN(params)

    # Automatic loss scaling is only supported for 'replicated', 'ps',
    # and 'independent' variable_updates.
    invalid_variable_updates = [
        'distributed_replicated', 'distributed_all_reduce'
    ]
    for variable_update in invalid_variable_updates:
      params = benchmark_cnn.make_params(
          use_fp16=True,
          fp16_vars=True,
          fp16_enable_auto_loss_scale=True,
          variable_update=variable_update)
      with self.assertRaises(ValueError):
        benchmark_cnn.BenchmarkCNN(params)

    # Automatic loss scaling is not supported for 'nccl'.
    params = benchmark_cnn.make_params(
        use_fp16=True,
        fp16_vars=True,
        fp16_enable_auto_loss_scale=True,
        all_reduce_spec='nccl')
    with self.assertRaises(ValueError):
      benchmark_cnn.BenchmarkCNN(params)

    # Automatic loss scaling is not supported for 'staged_vars'.
    params = benchmark_cnn.make_params(
        use_fp16=True,
        fp16_vars=True,
        fp16_enable_auto_loss_scale=True,
        staged_vars=True)
    with self.assertRaises(ValueError):
      benchmark_cnn.BenchmarkCNN(params)

  def testMakeParams(self):
    default_params = benchmark_cnn.make_params()
    self.assertEqual(default_params.model,
                     flags.param_specs['model'].default_value)
    params = benchmark_cnn.make_params(model='foo')
    self.assertEqual(params.model, 'foo')
    with self.assertRaises(ValueError):
      benchmark_cnn.make_params(job_name='foo')
    with self.assertRaises(ValueError):
      benchmark_cnn.make_params(gpu_memory_frac_for_testing=-1.)


class VariableUpdateTest(tf.test.TestCase):
  """Tests that variables are updated correctly.

  These tests use a very simple deterministic model. For example, some tests use
  the model

    loss = image * A * B

  where image is a 1x1 images (with a single scalar value), and A and B are
  scalar variables. Tests will run tf_cnn_benchmarks with such a model, on a
  sequence of scalar images, and assert that the losses are the correct value.
  Since the losses depend on the variables, this indirectly tests variables are
  updated correctly.
  """

  def setUp(self):
    super(VariableUpdateTest, self).setUp()
    _check_has_gpu()
    benchmark_cnn.setup(benchmark_cnn.make_params())

  def _get_benchmark_cnn_losses(self, inputs, params):
    """Returns the losses of BenchmarkCNN on the given inputs and params."""
    logs = []
    model = test_util.TestCNNModel()
    with test_util.monkey_patch(benchmark_cnn,
                                log_fn=test_util.print_and_add_to_list(logs),
                                LOSS_AND_ACCURACY_DIGITS_TO_SHOW=15):
      bench = benchmark_cnn.BenchmarkCNN(
          params, dataset=test_util.TestDataSet(), model=model)
      # The test model does not use labels when computing loss, so the label
      # values do not matter as long as it's the right shape.
      labels = np.array([1] * inputs.shape[0])
      bench.input_preprocessor.set_fake_data(inputs, labels)
      if bench.eval_input_preprocessor:
        bench.eval_input_preprocessor.set_fake_data(inputs, labels)
      bench.run()

    outputs = test_util.get_training_outputs_from_logs(
        logs, params.print_training_accuracy)
    return [x.loss for x in outputs]

  def _test_variable_update(self, params):
    """Tests variables are updated correctly when the given params are used.

    A BenchmarkCNN is created with a TestCNNModel, and is run with some scalar
    images. The losses are then compared with the losses obtained with
    TestCNNModel().manually_compute_losses()

    Args:
      params: a Params tuple used to create BenchmarkCNN.
    """
    inputs = test_util.get_fake_var_update_inputs()
    actual_losses = self._get_benchmark_cnn_losses(inputs, params)
    expected_losses, = test_util.TestCNNModel().manually_compute_losses(
        inputs, 1, params)
    rtol = 3e-2 if params.use_fp16 else 1e-5
    self.assertAllClose(actual_losses[:len(expected_losses)], expected_losses,
                        rtol=rtol, atol=0.)

  def _test_variable_updates(self, params,
                             var_updates=('parameter_server', 'replicated')):
    for var_update in var_updates:
      self._test_variable_update(params._replace(variable_update=var_update))

  def testDefault(self):
    params = test_util.get_var_update_params()
    self._test_variable_updates(params)

  # For some reason, this test doesn't always pass

  # def testCpuAsDevice(self):
  #   params = test_util.get_var_update_params()._replace(
  #       device='cpu',
  #       data_format='NHWC')  # NHWC required when --device=cpu
  #   self._test_variable_updates(params)

  def testCpuAsLocalParamDevice(self):
    params = test_util.get_var_update_params()._replace(
        local_parameter_device='cpu')
    self._test_variable_updates(params)

  def testFp16(self):
    params = test_util.get_var_update_params()._replace(use_fp16=True)
    self._test_variable_updates(params)

  def testMomentum(self):
    params = test_util.get_var_update_params()._replace(optimizer='momentum')
    self._test_variable_updates(params)

  def testRmsprop(self):
    params = test_util.get_var_update_params()._replace(optimizer='rmsprop')
    self._test_variable_updates(params)

  def testNoLayers(self):
    params = test_util.get_var_update_params()._replace(use_tf_layers=False)
    self._test_variable_updates(params)

  def testVariousAllReduceSpecs(self):
    # We do not test xring, because it requires all Variables to have at least
    # two elements.
    params = test_util.get_var_update_params()._replace(all_reduce_spec='pscpu')
    self._test_variable_updates(params, var_updates=('replicated',))
    params = params._replace(all_reduce_spec='psgpu')
    self._test_variable_updates(params, var_updates=('replicated',))
    # TODO(b/80125832): Enable nccl in tests
    # params = params._replace(all_reduce_spec='nccl',
    #                          compact_gradient_transfer=False)
    # self._test_variable_updates(params, var_updates=('replicated',))

  def testPrintBaseLoss(self):
    params = test_util.get_var_update_params()._replace(
        loss_type_to_report='base_loss')
    self._test_variable_updates(params)

  def testSingleL2LossOp(self):
    params = test_util.get_var_update_params()._replace(
        single_l2_loss_op=True)
    self._test_variable_updates(params)

  def testResourceVars(self):
    params = test_util.get_var_update_params()._replace(
        use_resource_vars=True)
    self._test_variable_updates(params)

  def testEvalDuringTrainingEveryNSteps(self):
    # TODO(reedwm): Test that the eval results are correct. This only tests that
    # training results are correct.
    params = test_util.get_var_update_params()._replace(
        eval_during_training_every_n_steps=1)
    self._test_variable_updates(params, var_updates=('replicated',))


class VariableMgrLocalReplicatedTest(tf.test.TestCase):

  def _test_grad_aggregation_with_var_mgr(self, variable_mgr, num_towers,
                                          num_vars, deferred_grads):
    tower_devices = ['/gpu:%d' % i for i in range(num_towers)]
    tower_grads = []
    expected_sums = [0.] * num_vars
    for i, tower_device in enumerate(tower_devices):
      with tf.device(tower_device):
        grad_vars = []
        for j in range(num_vars):
          n = num_towers * i + j
          grad_vars.append((tf.constant(n, dtype=tf.float32),
                            tf.Variable(n, dtype=tf.float32)))
          expected_sums[j] += n
      tower_grads.append(grad_vars)

    _, agg_device_grads = variable_mgr.preprocess_device_grads(
        tower_grads)
    expected_device_grads = []
    for i in range(num_towers):
      expected_grad_vars = []
      for j in range(num_vars):
        expected_grad_and_var = [expected_sums[j], num_towers * i + j]
        if isinstance(agg_device_grads[i][j], tuple):
          # agg_device_grads[i][j] can be a list or tuple.
          expected_grad_and_var = tuple(expected_grad_and_var)
        expected_grad_vars.append(expected_grad_and_var)
      if isinstance(agg_device_grads[i], tuple):
        # agg_device_grads[i] can be a list or tuple.
        expected_grad_vars = tuple(expected_grad_vars)
      expected_device_grads.append(expected_grad_vars)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
      sess.run(tf.initialize_all_variables())
      sess.run(variable_mgr._warmup_ops)
      if deferred_grads:
        # With deferred grads, the result of a session run is always the summed
        # gradients from the previous session run.
        sess.run(agg_device_grads)
        feed_dict = {g: 0 for grad_vars in tower_grads for g, _ in grad_vars}
        agg_device_grads_ = sess.run(agg_device_grads, feed_dict)
      else:
        agg_device_grads_ = sess.run(agg_device_grads)
    self.assertEqual(agg_device_grads_, expected_device_grads)

  def _test_grad_aggregation(self, params, num_vars):
    bench = benchmark_cnn.BenchmarkCNN(params)
    deferred_grads = (params.variable_consistency == 'relaxed')
    self._test_grad_aggregation_with_var_mgr(bench.variable_mgr, bench.num_gpus,
                                             num_vars, deferred_grads)

  def test_grad_aggregation(self):
    base_params = benchmark_cnn.make_params(num_gpus=10,
                                            variable_update='replicated',
                                            use_fp16=True)
    params = base_params
    self._test_grad_aggregation(params, 10)
    params = base_params._replace(gradient_repacking=3)
    self._test_grad_aggregation(params, 10)
    params = base_params._replace(variable_consistency='relaxed')
    self._test_grad_aggregation(params, 10)
    params = base_params._replace(compact_gradient_transfer=False)
    self._test_grad_aggregation(params, 10)
    params = base_params._replace(gradient_repacking=3,
                                  variable_consistency='relaxed')
    self._test_grad_aggregation(params, 10)
    params = base_params._replace(gradient_repacking=3,
                                  compact_gradient_transfer=False)
    self._test_grad_aggregation(params, 10)
    params = base_params._replace(variable_consistency='relaxed',
                                  compact_gradient_transfer=False)
    self._test_grad_aggregation(params, 10)
    params = base_params._replace(gradient_repacking=3,
                                  variable_consistency='relaxed',
                                  compact_gradient_transfer=False)
    self._test_grad_aggregation(params, 10)
    params = base_params._replace(num_gpus=8, hierarchical_copy=True)
    self._test_grad_aggregation(params, 10)
    # TODO(b/80125832): Enable nccl in tests
    # params = base_params._replace(all_reduce_spec='nccl',
    #                               compact_gradient_transfer=False,
    #                               # For some reason, this test freezes when
    #                               # num_gpus=10
    #                               num_gpus=8)
    # self._test_grad_aggregation(params, 10)
    params = base_params._replace(all_reduce_spec='pscpu')
    self._test_grad_aggregation(params, 10)

    params = base_params._replace(num_gpus=8,
                                  gradient_repacking=3,
                                  variable_consistency='relaxed',
                                  hierarchical_copy=True)
    self._test_grad_aggregation(params, 10)
    # TODO(b/80125832): Enable nccl in tests
    # params = base_params._replace(num_gpus=8,
    #                               gradient_repacking=3,
    #                               variable_consistency='relaxed',
    #                               all_reduce_spec='nccl',
    #                               compact_gradient_transfer=False)
    # self._test_grad_aggregation(params, 10)
    params = base_params._replace(gradient_repacking=3,
                                  variable_consistency='relaxed',
                                  all_reduce_spec='pscpu')
    self._test_grad_aggregation(params, 10)
    params = base_params._replace(gradient_repacking=3,
                                  variable_consistency='relaxed',
                                  all_reduce_spec='xring')
    self._test_grad_aggregation(params, 10)


if __name__ == '__main__':
  tf.test.main()
