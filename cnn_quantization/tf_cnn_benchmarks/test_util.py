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

"""Shared functionality across multiple test files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
from contextlib import contextmanager
import os

import numpy as np
import tensorflow.compat.v1 as tf
from cnn_quantization.tf_cnn_benchmarks import benchmark_cnn
from cnn_quantization.tf_cnn_benchmarks import cnn_util
from cnn_quantization.tf_cnn_benchmarks import datasets
from cnn_quantization.tf_cnn_benchmarks import preprocessing
from cnn_quantization.tf_cnn_benchmarks.models import model
from cnn_quantization.tf_cnn_benchmarks.platforms import util as platforms_util
from cnn_quantization.tf_cnn_benchmarks.test_data import tfrecord_image_generator
from tensorflow.python.platform import test


@contextmanager
def monkey_patch(obj, **kwargs):
  """Context mgr to monkey patch attributes on an object (such as a module).

  The attributes are patched back to their original value when the context
  manager exits.

  For example, to replace benchmark_cnn.get_data_type with an identity function,
  do:

  ```
  with monkey_patch(benchmark_cnn, get_data_type=lambda x: x)
    loss1 = benchmark_cnn.loss_function(1)  # loss1 will be 1
  loss2 = benchmark_cnn.loss_function(params)  # Call the original function
  ```

  Args:
    obj: The object (which can be a module) to monkey patch attributes on.
    **kwargs: Dictionary mapping from attribute name to value that the attribute
      will be patched with.
  Yields:
    Nothing.
  """
  old_values = {key: getattr(obj, key) for key in kwargs}
  try:
    for key, value in kwargs.items():
      setattr(obj, key, value)
    yield
  finally:
    for key, value in old_values.items():
      setattr(obj, key, value)


def monkey_patch_base_cluster_manager():
  """Monkey patches get_cluster_manager to return a BaseClusterManager.

  This function replaces platforms_util.get_cluster_manager with a function that
  always return a BaseClusterManager.

  This is useful for testing creating a graph in distributed mode, with only a
  single process. GrpcClusterManager's constructor blocks until a cluster is set
  up, which requires multiple processes to be created.
  """
  def get_test_cluster_manager(params, config_proto):
    del config_proto
    return cnn_util.BaseClusterManager(params)
  platforms_util.get_cluster_manager = get_test_cluster_manager


def print_and_add_to_list(print_list):
  """Returns a function which prints the input, then adds it to print_list."""
  def f(string):
    print(string)
    print_list.append(string)
  return f


TrainingOutput = namedtuple('TrainingOutput',
                            ['loss', 'top_1_accuracy', 'top_5_accuracy'])


EvalOutput = namedtuple('EvalOutput', ['top_1_accuracy', 'top_5_accuracy'])


def get_training_outputs_from_logs(logs, print_training_accuracy):
  """Returns a list of TrainingOutputs by parsing the logs of a training run.

  Args:
    logs: A list of strings, each which is a line from the standard output of
      tf_cnn_benchmarks from training. Only lines in the form:
        10 images/sec: 14.2 +/- 0.0 (jitter = 0.0) 7.020
      are parsed (the line may also contain the training accuracies).
    print_training_accuracy: The value of the param print_training_accuracy.
  Returns:
    A list of TrainingOutputs. The list has one element per element of logs
    that is in the format above. top_1_accuracy and top_5_accuracy are set to -1
    if the line does not contain accuracies.
  """
  outputs = []
  for log in logs:
    if 'images/sec' in log and '+/-' in log:
      parts = log.split()
      if print_training_accuracy:
        # Example log with training accuracy:
        #   10 images/sec: 0.2 +/- 0.0 (jitter = 0.0) 6.908 0.500 1.000
        assert len(parts) == 11
        top_1_acc = float(parts[9])
        top_5_acc = float(parts[10])
      else:
        # Example log without training accuracy:
        #   10 images/sec: 0.2 +/- 0.0 (jitter = 0.0) 6.908
        assert len(parts) == 9
        top_1_acc = -1
        top_5_acc = -1
      loss = float(parts[8])
      outputs.append(TrainingOutput(loss=loss, top_1_accuracy=top_1_acc,
                                    top_5_accuracy=top_5_acc))
  assert len(outputs) >= 1
  return outputs


def get_evaluation_outputs_from_logs(logs):
  """Returns the top 1 and 5 accuracies by parsing the logs of an eval run.

  Args:
    logs: A list of strings, each which is a line from the standard output of
      tf_cnn_benchmarks from evaluation. Only lines in the form:
        Accuracy @ 1 = 0.5000 Accuracy @ 5 = 1.0000 [80 examples]
      is parsed.
  Returns:
    A list of EvalOutputs. Normally this list only has one EvalOutput, but can
    contain multiple if training is done and
    --eval_during_training_every_n_steps is specified.
  """
  eval_outputs = []
  for log in logs:
    if 'Accuracy @ ' in log:
      # Example log:
      #   Accuracy @ 1 = 0.5000 Accuracy @ 5 = 1.0000 [80 examples]
      parts = log.split()
      assert len(parts) == 12
      top_1_accuracy = float(parts[4])
      top_5_accuracy = float(parts[9])
      eval_outputs.append(EvalOutput(top_1_accuracy, top_5_accuracy))
  assert eval_outputs
  return eval_outputs


def check_training_outputs_are_reasonable(testcase, training_outputs,
                                          print_training_accuracy,
                                          max_final_loss=10.,
                                          previous_final_loss=None):
  """Checks the outputs from training a model are reasonable.

  An assert is failed if the outputs are not reasonable. The final top-1 and
  top-5 accuracies are asserted to be 1, and so the dataset used to train should
  be trivial to learn. For example, the dataset could consist of a black image
  with label 0 and a white image with label 1.

  Args:
    testcase: A tf.test.TestCase used for assertions.
    training_outputs: A list of TrainingOutputs, as returned from
      get_training_outputs_from_logs().
    print_training_accuracy: Whether training accuracies were printed and stored
      in training_outputs.
    max_final_loss: The loss of the final training output is asserted to be at
      most this value.
    previous_final_loss: If training was resumed from a checkpoint, the loss of
      the final step from the previous training run that saved the checkpoint.
  """
  if previous_final_loss is not None:
    # Ensure the loss hasn't raised significantly from the final loss of the
    # previous training run.
    testcase.assertLessEqual(training_outputs[0].loss,
                             previous_final_loss * 1.01)
  for output in training_outputs:
    testcase.assertLessEqual(output.loss, 100.)
  last_output = training_outputs[-1]
  if print_training_accuracy:
    testcase.assertEqual(last_output.top_1_accuracy, 1.0)
    testcase.assertEqual(last_output.top_5_accuracy, 1.0)
  if max_final_loss is not None:
    testcase.assertLessEqual(last_output.loss, max_final_loss)


def train_and_eval(testcase,
                   run_fn,
                   params,
                   check_output_values,
                   max_final_loss=10.,
                   skip=None):
  """Trains a model then evaluates it.

  This function should be used to verify training and evaluating
  BenchmarkCNN works without crashing and that it outputs reasonable
  values. BenchmarkCNN will be run three times. First, it will train a
  model from scratch, saving a checkpoint. Second, it will load the checkpoint
  to continue training. Finally, it evaluates based on the loaded checkpoint.

  Args:
    testcase: A tf.test.TestCase used for assertions.
    run_fn: Must run `BenchmarkCNN` exactly once. BenchmarkCNN is
      never used directly, but instead is only run through `run_fn`. `run_fn`
      has the signature (run_type, inner_params) -> output_list, where:
        * run_type is a string indicating how BenchmarkCNN will be run.
          Either 'InitialTraining', 'TrainingFromCheckpoint' or 'Evaluation'.
        * inner_params is the params BenchmarkCNN should be run with.
        * output_list[i] is a list of lines from the ith worker's stdout.
    params: The params BenchmarkCNN will be run with.
      Will be passed to `run_fn` slightly modified in order to run with both
      training and evaluation.
    check_output_values: Whether the outputs of the workers, such as training
      accuracy, should be checked to make sure their values are reasonable.
      Fails an assert on `testcase` if a check fails.
    max_final_loss: The loss of the final training output is asserted to be at
      most this value for both training runs.
    skip: If 'eval', evaluation is not done. if
      'eval_and_train_from_checkpoint', evaluation and training from a
      checkpoint are both not done.
  """

  assert not skip or skip in {'eval', 'eval_and_train_from_checkpoint'}

  # Part 1: Train from scratch.
  tf.logging.info('Training model from scratch')
  print_training_accuracy = (params.print_training_accuracy or
                             params.forward_only)
  initial_train_logs = run_fn('InitialTraining', params)
  testcase.assertGreaterEqual(len(initial_train_logs), 1)
  for lines in initial_train_logs:
    initial_train_outputs = get_training_outputs_from_logs(
        lines, print_training_accuracy)
    if params.cross_replica_sync and params.batch_group_size == 1:
      testcase.assertEqual(len(initial_train_outputs), params.num_batches)
    if check_output_values:
      check_training_outputs_are_reasonable(testcase, initial_train_outputs,
                                            print_training_accuracy,
                                            max_final_loss=max_final_loss)
  if params.train_dir is not None:
    train_dir_entries = set(os.listdir(params.train_dir))
    testcase.assertGreater(len(train_dir_entries), 0)
  else:
    train_dir_entries = None

  if skip == 'eval_and_train_from_checkpoint':
    return

  # Part 2: Train from the loaded checkpoint.
  testcase.assertIsNotNone(train_dir_entries)
  tf.logging.info('Training model from loaded checkpoint')
  # Run for same number of batches as before.
  params = params._replace(num_batches=params.num_batches * 2)
  train_logs_from_ckpt = run_fn('TrainingFromCheckpoint', params)
  testcase.assertGreaterEqual(len(train_logs_from_ckpt), 1)
  for lines in train_logs_from_ckpt:
    train_outputs_from_ckpt = get_training_outputs_from_logs(
        lines, print_training_accuracy)
    if params.cross_replica_sync and params.batch_group_size == 1:
      testcase.assertEqual(len(train_outputs_from_ckpt),
                           params.num_batches // 2 - params.num_warmup_batches)
    if check_output_values:
      check_training_outputs_are_reasonable(
          testcase, train_outputs_from_ckpt, print_training_accuracy,
          max_final_loss=max_final_loss,
          previous_final_loss=initial_train_outputs[-1].loss)
  # Ensure a new checkpoint was written out.
  testcase.assertNotEqual(train_dir_entries, set(os.listdir(params.train_dir)))

  if skip == 'eval':
    return

  # Part 3: Evaluate from the loaded checkpoint.
  tf.logging.info('Evaluating model from checkpoint')
  params = params._replace(num_batches=params.num_batches // 2, eval=True)
  eval_logs = run_fn('Evaluation', params)
  testcase.assertGreaterEqual(len(eval_logs), 1)
  for lines in eval_logs:
    eval_outputs = get_evaluation_outputs_from_logs(lines)
    assert len(eval_outputs) == 1
    top_1_accuracy, top_5_accuracy = eval_outputs[0]
    if check_output_values:
      testcase.assertEqual(top_1_accuracy, 1.0)
      testcase.assertEqual(top_5_accuracy, 1.0)


def get_temp_dir(dir_name):
  dir_path = os.path.join(test.get_temp_dir(), dir_name)
  os.mkdir(dir_path)
  return dir_path


def create_black_and_white_images():
  dir_path = get_temp_dir('black_and_white_images')
  tfrecord_image_generator.write_black_and_white_tfrecord_data(dir_path,
                                                               num_classes=1)
  return dir_path


def get_params(train_dir_name):
  """Returns params that can be used to train."""
  return benchmark_cnn.make_params(
      batch_size=2,
      display_every=1,
      init_learning_rate=0.005,
      model='trivial',
      num_batches=20,
      num_gpus=2,
      num_warmup_batches=5,
      optimizer='sgd',
      print_training_accuracy=True,
      train_dir=get_temp_dir(train_dir_name),
      variable_update='parameter_server',
      weight_decay=0)


def get_var_update_params():
  """Returns params that are used when testing variable updates."""
  return benchmark_cnn.make_params(
      batch_size=2,
      model='test_model',
      num_gpus=2,
      display_every=1,
      num_warmup_batches=0,
      num_batches=4,
      weight_decay=2 ** -4,
      init_learning_rate=2 ** -4,
      optimizer='sgd')


def get_fake_var_update_inputs():
  """Returns fake input 1x1 images to use in variable update tests."""
  # BenchmarkCNN divides by 127.5 then subtracts 1.0 from the images, so after
  # that, the images will be -1., 0., 1., ..., 14.
  return np.resize(127.5 * np.array(range(16)), (16, 1, 1, 1))


def _worker_batches_in_numpy_array(numpy_inputs, batch_size, shift_ratio):
  """Yields batches from a numpy array, for a single worker."""
  numpy_inputs = cnn_util.roll_numpy_batches(numpy_inputs, batch_size,
                                             shift_ratio)
  i = 0
  total_batches = numpy_inputs.shape[0]
  assert total_batches % batch_size == 0
  while True:
    yield numpy_inputs[i:i + batch_size, Ellipsis]
    i = (i + batch_size) % total_batches


def manually_compute_losses(numpy_inputs, inputs_placeholder, loss, num_workers,
                            params):
  """Manually compute the losses each worker should report in tf_cnn_benchmarks.

  This function essentially simulates tf_cnn_benchmarks, computing what the loss
  of each worker should be. The caller should create a model, that takes in
  images from `inputs_placeholder`, a tf.placeholder, and computes `loss`.

  This function, and all ops passed to this function, must be run under a
  tf.device('cpu:0') context manager.

  Non-SGD optimizers are not supported with multiple workers.

  Args:
    numpy_inputs: A Numpy array to use as the input images.
    inputs_placeholder: A tf.placeholder tensor, where input images can be fed
      into.
    loss: A scalar tensor representing the loss of the model, which is obtained
      from the input images in inputs_placeholder.
    num_workers: How many workers should be simulated.
    params: Params tuple. This doesn't have to have information about the
      distributed cluster, such as --num_workers, as num_workers is passed in
      separately.

  Returns:
    A list of list of losses. return_value[i][j] is the loss of the ith worker
    after the jth step.
  """
  batch_size = params.batch_size * params.num_gpus
  assert numpy_inputs.shape[0] % (num_workers * batch_size) == 0
  l2_loss = tf.add_n([tf.nn.l2_loss(x) for x in tf.trainable_variables()])
  total_loss = loss + params.weight_decay * l2_loss
  reported_loss = (loss if params.loss_type_to_report == 'base_loss'
                   else total_loss)
  gradient_multiplier = 1
  if params.variable_update in ('replicated', 'distributed_all_reduce'):
    # In certain variable updates, tf_cnn_benchmarks add the gradients of the
    # GPUs instead of taking their mean, making the gradients effectively
    # params.num_gpu times higher.
    # TODO(b/62722498): Make all variable updates consistent.
    gradient_multiplier = params.num_gpus

  opt = benchmark_cnn.get_optimizer(params, params.init_learning_rate)
  grad_vars = opt.compute_gradients(
      total_loss, grad_loss=tf.constant(gradient_multiplier, dtype=tf.float32))
  grads = [g for g, _ in grad_vars]
  # We apply gradients from a placeholder. That way, we can first compute the
  # gradients from each worker, then afterwards apply them one by one by feeding
  # them into the placeholder.
  placeholder_grad_vars = [(tf.placeholder(g.dtype, g.shape), v)
                           for g, v in grad_vars]
  placeholder_grads = [g for g, _ in placeholder_grad_vars]
  apply_grads_op = opt.apply_gradients(placeholder_grad_vars)

  batch_iterators = [_worker_batches_in_numpy_array(numpy_inputs, batch_size,
                                                    shift_ratio=i / num_workers)
                     for i in range(num_workers)]
  # Set the GPU count to 0, to avoid taking all the GPU memory. Unfortunately,
  # doing so still takes up about ~1GB for some reason.
  with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
    sess.run(tf.global_variables_initializer())
    losses = [[] for _ in range(num_workers)]
    for i in range(params.num_batches):
      computed_grads = []
      for j in range(num_workers):
        batch_feed = next(batch_iterators[j])
        batch_feed = batch_feed / 127.5 - 1
        worker_loss, worker_grads = sess.run((reported_loss, grads),
                                             {inputs_placeholder: batch_feed})
        losses[j].append(worker_loss)
        computed_grads.append(worker_grads)
      for worker_grads in computed_grads:
        # TODO(reedwm): With multiple workers, applying the gradients
        # sequentially per worker is not equivalent to what tf_cnn_benchmarks
        # does when the optmizer is not SGD. Therefore, this currently does not
        # work currently when num_workers > 1 and params.optimizer != 'sgd'.
        feed_dict = dict(zip(placeholder_grads, worker_grads))
        sess.run(apply_grads_op, feed_dict)
  return losses


class TestCNNModel(model.CNNModel):
  """A simple model used for testing.

  The input is a 1-channel 1x1 image, consisting of a single number. The model
  has two scalar variables: A and B, initialized to 1 and 2 respectively. Given
  an image x, the loss is defined as:

      loss = x * A * B
  """

  def __init__(self):
    super(TestCNNModel, self).__init__(
        'test_cnn_model', image_size=1, batch_size=1, learning_rate=1)
    self.depth = 1

  VAR_A_INITIAL_VALUE = 1.
  VAR_B_INITIAL_VALUE = 2.

  def add_inference(self, cnn):
    # This model only supports 1x1 images with 1 channel
    assert cnn.top_layer.shape[1:] == (1, 1, 1)
    # Multiply by variable A.
    with tf.name_scope('mult_by_var_A'):
      cnn.conv(1, 1, 1, 1, 1, use_batch_norm=None, activation=None, bias=None,
               kernel_initializer=tf.constant_initializer(
                   self.VAR_A_INITIAL_VALUE))
    # Multiply by variable B.
    with tf.name_scope('mult_by_var_B'):
      cnn.conv(1, 1, 1, 1, 1, use_batch_norm=None, activation=None, bias=None,
               kernel_initializer=tf.constant_initializer(
                   self.VAR_B_INITIAL_VALUE))
    with tf.name_scope('reshape_to_scalar'):
      cnn.reshape([-1, 1])

  def skip_final_affine_layer(self):
    return True

  def loss_function(self, inputs, build_network_result):
    del inputs
    return tf.reduce_mean(build_network_result.logits)

  def manually_compute_losses(self, inputs, num_workers, params):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
      a = tf.Variable(self.VAR_A_INITIAL_VALUE, name='A')
      b = tf.Variable(self.VAR_B_INITIAL_VALUE, name='B')
      inputs_placeholder = tf.placeholder(tf.float32,
                                          (None, 1, 1, 1),
                                          name='inputs_placeholder')
      inputs_reshaped = tf.reshape(inputs_placeholder, (-1, 1))
      loss = self.loss_function(
          None,
          model.BuildNetworkResult(logits=inputs_reshaped * a * b,
                                   extra_info=None))
      return manually_compute_losses(inputs, inputs_placeholder, loss,
                                     num_workers, params)

  def accuracy_function(self, inputs, logits):
    del inputs
    # Let the accuracy be the same as the loss function.
    return {'top_1_accuracy': logits, 'top_5_accuracy': logits}


class TestDataSet(datasets.ImageDataset):
  """A Dataset consisting of 1x1 images with a depth of 1."""

  def __init__(self, height=1, width=1, depth=1):
    super(TestDataSet, self).__init__('test_dataset', height=height,
                                      width=width, depth=depth, data_dir=None,
                                      queue_runner_required=True, num_classes=1)

  def num_examples_per_epoch(self, subset='train'):
    del subset
    return 1

  def get_input_preprocessor(self, input_preprocessor='default'):
    return preprocessing.TestImagePreprocessor

  def use_synthetic_gpu_inputs(self):
    return False
