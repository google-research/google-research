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

"""Benchmarks the all-reduce algorithms of tf_cnn_benchmarks.

tf_cnn_benchmarks uses all-reduce to aggregate gradients. This benchmark is
useful for benchmarking the performance of just this gradient aggregation,
instead of the entire model. All the flags that tf_cnn_benchmarks accepts are
also accepted by this script, although many are silently ignored.

The number and shapes of the tensors all-reduced are those of the variables of
the model specified by the --model flag.
TODO(reedwm): Allow custom sizes to be specified.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import time

from absl import flags as absl_flags
import tensorflow as tf
from cnn_quantization.tf_cnn_benchmarks import benchmark_cnn
from cnn_quantization.tf_cnn_benchmarks import cnn_util
from cnn_quantization.tf_cnn_benchmarks import flags
from cnn_quantization.tf_cnn_benchmarks.cnn_util import log_fn
from tensorflow.python.ops import control_flow_ops


absl_flags.DEFINE_integer('iters_per_step', 5,
                          'Number of iterations to run all-reduce for, per '
                          'step. Every step, a session will be run on a Graph '
                          'that contains this many copies of the all-reduce. '
                          'The copies are run sequentially. Setting this above '
                          '1 is useful to lower the overhead of starting the '
                          'session run, running the VariableV2 ops at the '
                          'start of the step, etc.')


flags.define_flags()
for name in flags.param_specs.keys():
  absl_flags.declare_key_flag(name)


def get_var_shapes(model):
  """Returns the list of variable shapes for a tf_cnn_benchmarks Model."""
  with tf.Graph().as_default():
    # The variable shapes do not depend on the batch size.
    images = tf.placeholder(tf.float32, model.get_input_shapes('train')[0])
    model.build_network([images])
    return [[int(d) for d in v.shape.dims] for v in tf.trainable_variables()]


def all_reduce(all_device_tensors, variable_mgr):
  """Performs a single batch all-reduce.

  Args:
    all_device_tensors: List of lists of tensors. all_device_tensors[t][i] is
      a tensor, where t is the tower the tensor is on and i is the index of
      the tensor.
    variable_mgr: The VariableMgr to perform the all-reduce.
  Returns:
    List of list of tensors in the same form as `all_device_tensors`, except the
    tensors are aggregated across towers.
  """
  tower_grads = [[(g, None) for g in device_tensors] for
                 device_tensors in all_device_tensors]
  _, aggregated_tower_grads = variable_mgr.preprocess_device_grads(tower_grads)
  return [
      [g for g, _ in agg_device_tensors]
      for agg_device_tensors in aggregated_tower_grads]


def build_all_reduce_iterations(all_device_tensors, tower_devices, variable_mgr,
                                num_iters):
  """Builds the all-reduce ops for multiple iterations to aggregate tensors.

  The tensors in `all_device_tensors` are aggregated `num_iters` times. Each
  iteration aggregates the results from the previous iteration. The iterations
  are run sequentially, so the aggregations for an iteration do not start
  running until the previous iteration has completed. Each iteration after the
  first is aggregating already-aggregated values, but it does not matter because
  we are only aggregating for benchmarking purposes.

  Args:
    all_device_tensors: List of lists of tensors. all_device_tensors[t][i] is
      a tensor, where t is the tower the tensor is on and i is the index of
      the tensor.
    tower_devices: A list of device strings. tower_devices[t] is the device
      of the tensors in all_device_tensors[t].
    variable_mgr: The VariableMgr to perform the all-reduce.
    num_iters: Number of iterations to aggregate tensors for.
  Returns:
    An op that when run, causes the all-reduce ops to run.
  """
  for i in range(num_iters):
    with tf.name_scope('iteration_%d' % i):
      # Step 1: Do the aggregation.
      with tf.name_scope('tensor_aggregation'):
        all_device_tensors = all_reduce(all_device_tensors, variable_mgr)

      # Step 2. Create identity ops, to bring the aggregated results back to
      # each device.
      new_all_device_tensors = []
      for device, device_tensors in zip(tower_devices, all_device_tensors):
        with tf.device(device):
          new_all_device_tensors.append([
              tf.identity(t, name='identity_after_allreduce')
              for t in device_tensors
          ])
      all_device_tensors = new_all_device_tensors

      # Step 3. Add control dependencies to delay the next iteration until this
      # iteration is complete. To avoid extra overhead, we do not have any
      # cross-device control dependencies, which means it's possible for two
      # iterations to slightly overlap.
      new_all_device_tensors = []
      for device_tensors in all_device_tensors:
        new_all_device_tensors.append([
            control_flow_ops.with_dependencies(
                device_tensors, t, name='identity_after_dependencies')
            for t in device_tensors
        ])
      all_device_tensors = new_all_device_tensors

  # To prevent the dependency optimizer from removing every op we created,
  # we store the results in variables.
  ops_to_run = []
  for device, device_tensors in zip(tower_devices, all_device_tensors):
    with tf.device(device):
      for t in device_tensors:
        # The placeholder initial value is never run.
        var = tf.Variable(tf.placeholder(tf.float32, t.shape), collections=[])
        ops_to_run.append(var.assign(t))
  return tf.group(*ops_to_run)


def build_graph(tower_devices, tensor_shapes, variable_mgr, num_iters):
  """Builds the graph for the benchmark.

  Args:
    tower_devices: A list of device strings of the devices to run the all-reduce
      benchmark on.
    tensor_shapes: A list of shapes of the tensors that will be aggregated for
      the all-reduce.
    variable_mgr: The VariableMgr to perform the all-reduce.
    num_iters: Number of iterations to aggregate tensors for.
  Returns:
    An op that runs the benchmark.
  """
  all_device_tensors = []
  for i, tower_device in enumerate(tower_devices):
    with tf.device(tower_device):
      device_tensors = []
      for j, shape in enumerate(tensor_shapes):
        tensor = tf.Variable(tf.random_normal(shape, dtype=tf.float32),
                             name='tensor_%d_on_device_%d' % (j, i))
        device_tensors.append(tensor)
    all_device_tensors.append(device_tensors)

  log_fn('Building all-reduce ops')
  benchmark_op = build_all_reduce_iterations(all_device_tensors, tower_devices,
                                             variable_mgr, num_iters)
  log_fn('Done building all-reduce ops')
  return benchmark_op


def run_graph(benchmark_op, bench_cnn, init_ops, dummy_loss_op):
  """Runs the graph for the benchmark.

  Args:
    benchmark_op: An op that runs the benchmark.
    bench_cnn: The BenchmarkCNN where params and other attributes are obtained.
    init_ops: A list of ops that are run before `benchmark_op` for
      initialization.
    dummy_loss_op: Any op. We must pass a loss op to
      `benchmark_cnn.benchmark_one_step`, but the result of the op is never
      actually used.
  """
  config = benchmark_cnn.create_config_proto(bench_cnn.params)
  with tf.Session(config=config) as sess:
    for op in init_ops:
      sess.run(op)
    step_train_times = []
    fetches = {'average_loss': dummy_loss_op, 'benchmark_op': benchmark_op}
    log_fn('Running warmup')
    for i in range(-bench_cnn.num_warmup_batches, bench_cnn.num_batches):
      if i == 0:
        log_fn('Running all-reduce ops')
        start = time.time()
      if i > 0 and i % bench_cnn.params.display_every == 0:
        log_fn('Iteration: %d. Average time per step so far: %s' %
               (i, (time.time() - start) / i))
      # Call benchmark_one_step instead of directly calling sess.run(...), to
      # potentially get a trace file, partitioned graphs, etc.
      benchmark_cnn.benchmark_one_step(
          sess=sess,
          fetches=fetches,
          step=i,
          # The batch size is only used for the images/sec calculation, which is
          # not actually calculated because we pass show_images_per_sec=False.
          batch_size=None,
          step_train_times=step_train_times,
          trace_filename=bench_cnn.trace_filename,
          partitioned_graph_file_prefix=(
              bench_cnn.params.partitioned_graph_file_prefix),
          profiler=None,
          image_producer=None,
          params=bench_cnn.params,
          show_images_per_sec=False)
    log_fn('Average time per step: %s' %
           ((time.time() - start) / bench_cnn.num_batches))


def run_benchmark(bench_cnn, num_iters):
  """Runs the all-reduce benchmark.

  Args:
    bench_cnn: The BenchmarkCNN where params, the variable manager, and other
      attributes are obtained.
    num_iters: Number of iterations to do all-reduce for for.

  Raises:
    ValueError: Invalid params of bench_cnn.
  """
  if bench_cnn.params.variable_update != 'replicated':
    raise ValueError('--variable_update=replicated must be specified to use'
                     'the all-reduce benchmark')
  if bench_cnn.params.variable_consistency == 'relaxed':
    raise ValueError('--variable_consistency=relaxed is not supported')

  benchmark_op = build_graph(bench_cnn.raw_devices,
                             get_var_shapes(bench_cnn.model),
                             bench_cnn.variable_mgr, num_iters)
  init_ops = [
      tf.global_variables_initializer(),
      bench_cnn.variable_mgr.get_post_init_ops()
  ]
  loss_op = tf.no_op()

  if bench_cnn.graph_file:
    path, filename = os.path.split(bench_cnn.graph_file)
    as_text = filename.endswith('txt')
    log_fn('Writing GraphDef as %s to %s' % (
        'text' if as_text else 'binary', bench_cnn.graph_file))
    tf.train.write_graph(tf.get_default_graph().as_graph_def(add_shapes=True),
                         path, filename, as_text)

  run_graph(benchmark_op, bench_cnn, init_ops, loss_op)


# TODO(reedwm): Reduce redundancy with tf_cnn_benchmarks
def main(positional_arguments):
  # Command-line arguments like '--distortions False' are equivalent to
  # '--distortions=True False', where False is a positional argument. To prevent
  # this from silently running with distortions, we do not allow positional
  # arguments.
  assert len(positional_arguments) >= 1
  if len(positional_arguments) > 1:
    raise ValueError('Received unknown positional arguments: %s'
                     % positional_arguments[1:])

  params = benchmark_cnn.make_params_from_flags()
  params = benchmark_cnn.setup(params)
  bench = benchmark_cnn.BenchmarkCNN(params)

  tfversion = cnn_util.tensorflow_version_tuple()
  log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

  run_benchmark(bench, absl_flags.FLAGS.iters_per_step)

if __name__ == '__main__':
  tf.app.run()
