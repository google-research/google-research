# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests running benchmark_cnn in distributed mode.

This is done by spawning one process per task. Each process runs
benchmark_cnn_distributed_test_runner.py.

The output for each process is written to disk and can be viewed to debug tests.
See get_test_output_dir() in platforms/default/util.py for more info.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
import os
import subprocess
import time

from absl import flags as absl_flags
import portpicker
import six
import tensorflow.compat.v1 as tf
from cnn_quantization.tf_cnn_benchmarks import flags
from cnn_quantization.tf_cnn_benchmarks import test_util
from cnn_quantization.tf_cnn_benchmarks.platforms import util as platforms_util

FLAGS = absl_flags.FLAGS


def _convert_params_to_flags_list(params):
  """Converts Params to a list of flags. Skips default-valued parameters.

  E.g., converts
    benchmark_cnn.make_params(batch_size=32, model='resnet50')
  to
    ['--batch_size=32', '--model=resnet50']

  Args:
    params: Params for BenchmarkCNN.
  Returns:
    A list of flags.
  """
  return [
      '--%s=%s' % (k, str(v)) for k, v in six.iteritems(params._asdict())
      if v != flags.param_specs[k].default_value
  ]


# When outputting a process's output in the log, maximum number of characters
# to output. The log system does not allow us to output more than this in a
# single log message, but this limit is also useful to avoid the logs from
# becoming too large (the full process output is written to disk).
MAX_OUTPUT_CHARS = 15000


# A process. name is a string identifying the process in logs. stdout and
# stderr are file objects of the process's stdout and stderr, respectively.
_ProcessInfo = namedtuple('_ProcessInfo', ['name', 'popen', 'stdout', 'stderr'])


def _create_task_process(job_name, task_index, args, env, output_dir):
  """Creates a process for a single task for benchmark_cnn.

  Args:
    job_name: 'worker' or 'ps' or ''. Empty string used for non-distributed
      mode.
    task_index: The index of the task within the cluster.
    args: A list of arguments to pass to the task. This function additionally
      sets --task_index and --job_name
    env: The environment to use for the task.
    output_dir: Where to place the output files, storing the task's stdout and
      stderr.
  Returns:
    A _ProcessInfo namedtuple of the running process. The stdout and stderr
    fields of this tuple must be closed by the caller once the process ends.
  """
  args = args[:]
  args += ['--task_index=%s' % task_index, '--job_name=%s' % job_name]
  name_prefix = job_name or 'local'
  process_name = '%s_%s' % (name_prefix, task_index)
  tf.logging.info('Spawning %s process: %s' % (process_name, ' '.join(args)))
  stdout_filename = os.path.join(output_dir, '%s_stdout.txt' % process_name)
  stderr_filename = os.path.join(output_dir, '%s_stderr.txt' % process_name)
  stdout_file = open(stdout_filename, 'w+')
  stderr_file = open(stderr_filename, 'w+')
  popen = subprocess.Popen(
      args, stdout=stdout_file, stderr=stderr_file, env=env)
  return _ProcessInfo(process_name, popen, stdout_file, stderr_file)


def _wait_for_processes(wait_processes, kill_processes):
  """Waits until all `wait_processes` finish, then kills `kill_processes`.

  Fails an assert if a process in `wait_processes` finishes unsuccessfully.
  The processes in `kill_processes` are assumed to never finish so they are
  killed.

  Args:
    wait_processes: A list of _ProcessInfo tuples. This function will wait
      for each to finish.
    kill_processes: A list of _ProcessInfo tuples. Each will be killed once
      every process in `wait_processes` is finished.
  Returns:
    A list of strings, each which is a string of the stdout of a wait process.
  """
  wait_process_stdouts = [None] * len(wait_processes)
  finished_wait_processes = set()
  while len(finished_wait_processes) < len(wait_processes):
    for i, wait_process in enumerate(wait_processes):
      if i in finished_wait_processes:
        continue
      ret_code = wait_process.popen.poll()
      if ret_code is None:
        continue
      tf.logging.info('{} finished'.format(wait_process.name))
      wait_process.stdout.seek(0)
      wait_process_stdouts[i] = wait_process.stdout.read()
      tf.logging.info('stdout for {} (last {} chars): {}\n'.format(
          wait_process.name, MAX_OUTPUT_CHARS,
          wait_process_stdouts[i][-MAX_OUTPUT_CHARS:]))
      wait_process.stderr.seek(0)
      tf.logging.info('stderr for {} (last {} chars): {}\n'.format(
          wait_process.name, MAX_OUTPUT_CHARS,
          wait_process.stderr.read()[-MAX_OUTPUT_CHARS:]))
      assert ret_code == 0, 'Process failed with return code %d' % ret_code
      finished_wait_processes.add(i)
    for kill_process in kill_processes:
      ret_code = kill_process.popen.poll()
      # kill processes should not end until we kill them.
      assert ret_code is None, 'Process returned early with code %d' % ret_code
    time.sleep(0.25)
  tf.logging.info('All wait processes finished')
  for i, kill_process in enumerate(kill_processes):
    # Kill each kill process.
    kill_process.popen.kill()
    kill_process.popen.wait()
    kill_process.stdout.seek(0)
    tf.logging.info('stdout for {} (last {} chars): {}\n'.format(
        kill_process.name, MAX_OUTPUT_CHARS,
        kill_process.stdout.read()[-MAX_OUTPUT_CHARS:]))
    kill_process.stderr.seek(0)
    tf.logging.info('stderr for {} (last {} chars): {}\n'.format(
        kill_process.name, MAX_OUTPUT_CHARS,
        kill_process.stderr.read()[-MAX_OUTPUT_CHARS:]))
  return wait_process_stdouts


def _spawn_benchmark_processes(output_dir_path, num_workers, num_ps,
                               num_controllers, params):
  """Run training or evaluation in spawned processes.

  Runs locally if num_workers == 1, num_ps == 0, and num_controllers == 0,
  otherwise runs in distributed mode. In either case, one process is spawned
  per worker and ps. Waits for training/evaluation to finish before returning.

  Args:
    output_dir_path: Relative path where stdout and stderr files will be
      placed.
    num_workers: Number of workers to spawn.
    num_ps: Number of ps processes to spawn.
    num_controllers: Number of controller processes to spawn (must be 0 or 1).
    params: Params for BenchmarkCNN in each subprocess.
  Returns:
    A list output_list of outputs from all processes that output the
    images/sec and accuracy. This process is the controller host in
    distributed_all_reduce, and the workers otherwise. output_list[i] is a
    list of lines from the ith worker's stdout.
  """
  run_distributed = num_workers != 1 or num_ps != 0 or num_controllers != 0
  if params.variable_update == 'distributed_all_reduce':
    assert num_controllers == 1 or not run_distributed
    assert num_ps == 0
  else:
    assert num_controllers == 0
  output_base_dir = platforms_util.get_test_output_dir()
  output_dir = os.path.join(output_base_dir, output_dir_path)
  os.makedirs(output_dir)
  tf.logging.info('Outputs of processes will be outputted to: %s' % output_dir)

  args = platforms_util.get_command_to_run_python_module(
      'benchmark_cnn_distributed_test_runner')
  args += _convert_params_to_flags_list(params)
  if run_distributed:
    worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
    ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]
    controller_ports = [portpicker.pick_unused_port()
                        for _ in range(num_controllers)]
    # The numerator is 0.7 instead of 1 to leave some memory for the Cuda
    # runtime, etc.
    gpu_memory_frac = 0.7 / num_workers
    args += [
        '--gpu_memory_frac_for_testing=%f' % gpu_memory_frac,
        '--worker_hosts=' + ','.join('localhost:%d' % p for p in worker_ports)
    ]
    if num_ps > 0:
      ps_hosts_str = ','.join('localhost:%d' % p for p in ps_ports)
      args.append('--ps_hosts=' + ps_hosts_str)
    else:
      controller_host_str = ','.join('localhost:%d' % p
                                     for p in controller_ports)
      args.append('--controller_host=' + controller_host_str)
  env = os.environ.copy()
  # Allow stdout to be viewed before the process ends.
  env['PYTHONUNBUFFERED'] = '1'

  worker_processes = []
  ps_processes = []
  controller_processes = []
  try:
    for i in range(num_workers):
      job_name = 'worker' if run_distributed else ''
      process = _create_task_process(job_name, i, args, env, output_dir)
      worker_processes.append(process)
    # Don't let ps or controller processes use the gpu.
    env['CUDA_VISIBLE_DEVICES'] = ''

    for i in range(num_ps):
      process = _create_task_process('ps', i, args, env, output_dir)
      ps_processes.append(process)
    for i in range(num_controllers):
      process = _create_task_process('controller', i, args, env, output_dir)
      controller_processes.append(process)
    # If all distributed all reduce mode is being used, the controller process
    # finishes and the worker processes block forever. Otherwise, the worker
    # processes finish and the ps processes block forever. We set
    # wait_processes and kill_processes accordingly.
    if controller_processes:
      wait_processes = controller_processes
      kill_processes = worker_processes
    else:
      wait_processes = worker_processes
      kill_processes = ps_processes
    outputs = _wait_for_processes(wait_processes, kill_processes)
  finally:
    for process in worker_processes + ps_processes + controller_processes:
      try:
        process.popen.kill()
      except OSError:
        pass  # It's OK (and expected) if the process already exited.
      process.stdout.close()
      process.stderr.close()
  return [output.splitlines() for output in outputs]


# When this test class is run, a method will fail about 0.3% of the time with a
# gRPC error. It is not clear why this occurs.
# TODO(reedwm): Fix this test class.
class TfCnnBenchmarksDistributedTest(tf.test.TestCase):
  """Tests running benchmark_cnn in distributed mode."""

  # We cannot check for a GPU via tf.test.is_gpu_available() before the tests in
  # this class because it allocates all the GPU memory which would cause the
  # spawned processes to run out of GPU memory.

  def _test_distributed(self,
                        test_name,
                        num_workers,
                        num_ps,
                        params,
                        num_controllers=0,
                        check_output_values=False,
                        skip=None):
    # TODO(reedwm): check_output_values should default to True and be enabled
    # on every test. See the TODO in benchmark_cnn_test.py.
    def run_fn(run_type, inner_params):
      output_dir_path = os.path.join(test_name, run_type)
      if run_type == 'Evaluation':
        # Distributed evaluation is not supported, so we use a single process.
        # We still must spawn another process, because if we evaluate in the
        # current process, it would allocate the GPU memory causing future test
        # methods to fail.
        if inner_params.variable_update == 'distributed_replicated':
          inner_params = inner_params._replace(variable_update='replicated')
        return _spawn_benchmark_processes(
            output_dir_path, num_workers=1, num_ps=0, num_controllers=0,
            params=inner_params)
      else:
        return _spawn_benchmark_processes(output_dir_path, num_workers, num_ps,
                                          num_controllers, inner_params)

    return test_util.train_and_eval(self, run_fn, params,
                                    check_output_values=check_output_values,
                                    skip=skip)

  def testParameterServer(self):
    test_name = 'testParameterServer'
    params = test_util.get_params(test_name)
    self._test_distributed(test_name, 2, 2, params)

  def testParameterServerStaged(self):
    test_name = 'testParameterServerStaged'
    params = test_util.get_params(test_name)._replace(staged_vars=True)
    self._test_distributed(test_name, 2, 2, params)

  def testReplicated(self):
    test_name = 'testReplicated'
    params = test_util.get_params(test_name)._replace(
        variable_update='distributed_replicated')
    self._test_distributed(test_name, 2, 2, params)

  def testAllReducePsgpu(self):
    test_name = 'testAllReducePsgpu'
    flags_dict = test_util.get_params(test_name)._replace(
        variable_update='distributed_all_reduce',
        all_reduce_spec='psgpu#4')
    self._test_distributed(test_name, 2, 0, flags_dict, num_controllers=1)

  def testAllReducePscpuXring(self):
    test_name = 'testAllReducePscpuXring'
    flags_dict = test_util.get_params(test_name)._replace(
        variable_update='distributed_all_reduce',
        all_reduce_spec='pscpu:2k:xring')
    self._test_distributed(test_name, 2, 0, flags_dict, num_controllers=1)

  def testForwardOnly(self):
    test_name = 'testForwardOnly'
    params = test_util.get_params(test_name)._replace(forward_only=True)
    # Evaluation is not supported with --forward_only, so we set skip='eval'.
    self._test_distributed(test_name, 2, 2, params, skip='eval')

  def testSingleWorkerAndPs(self):
    test_name = 'testSingleWorkerAndPs'
    params = test_util.get_params(test_name)
    self._test_distributed(test_name, 1, 1, params)

  def testThreeWorkersAndPses(self):
    test_name = 'testThreeWorkersAndPses'
    params = test_util.get_params(test_name)
    self._test_distributed(test_name, 3, 3, params)

  def testOneWorkerThreePses(self):
    test_name = 'testOneWorkerThreePses'
    params = test_util.get_params(test_name)
    self._test_distributed(test_name, 1, 3, params)

  def testThreeWorkersOnePs(self):
    test_name = 'testThreeWorkersOnePs'
    params = test_util.get_params(test_name)
    self._test_distributed(test_name, 3, 1, params)

  def testNoPrintTrainingAccuracy(self):
    test_name = 'testNoPrintTrainingAccuracy'
    params = test_util.get_params(test_name)._replace(
        print_training_accuracy=False)
    self._test_distributed(test_name, 2, 2, params)

  def testRmspropParameterServer(self):
    test_name = 'testRmspropParameterServer'
    params = test_util.get_params(test_name)._replace(optimizer='rmsprop')
    self._test_distributed(test_name, 2, 2, params)

  def testMomentumReplicated(self):
    test_name = 'testMomentumReplicated'
    params = test_util.get_params(test_name)._replace(
        optimizer='momentum', variable_update='distributed_replicated')
    self._test_distributed(test_name, 2, 2, params)

  def testNoCrossReplicaSyncParameterServerStaged(self):
    test_name = 'testNoCrossReplicaSyncParameterServerStaged'
    params = test_util.get_params(test_name)._replace(
        staged_vars=True, cross_replica_sync=False)
    self._test_distributed(test_name, 2, 2, params)

  def testSingleGpu(self):
    test_name = 'testSingleGpu'
    params = test_util.get_params(test_name)._replace(num_gpus=1)
    self._test_distributed(test_name, 2, 2, params)

  def testBatchGroupSize(self):
    test_name = 'testBatchGroupSize'
    params = test_util.get_params(test_name)._replace(
        batch_group_size=4, num_batches=100, num_warmup_batches=5)
    self._test_distributed(test_name, 2, 2, params)

  def testFp16WithFp32Vars(self):
    test_name = 'testFp16WithFp32Vars'
    params = test_util.get_params(test_name)._replace(
        use_fp16=True, fp16_vars=False)
    self._test_distributed(test_name, 2, 2, params)

  def testFp16WithFp16Vars(self):
    test_name = 'testFp16WithFp16Vars'
    params = test_util.get_params(test_name)._replace(
        use_fp16=True, fp16_vars=True, fp16_loss_scale=1.)
    self._test_distributed(test_name, 2, 2, params)

  def testFp16Replicated(self):
    test_name = 'testFp16Replicated'
    params = test_util.get_params(test_name)._replace(
        use_fp16=True, variable_update='distributed_replicated')
    self._test_distributed(test_name, 2, 2, params)

  def testReplicatedRealData(self):
    test_name = 'testReplicatedRealData'
    imagenet_dir = os.path.join(platforms_util.get_test_data_dir(),
                                'fake_tf_record_data')
    params = test_util.get_params(test_name)._replace(
        variable_update='distributed_replicated',
        data_dir=imagenet_dir,
        data_name='imagenet')
    self._test_distributed(test_name, 2, 2, params)


class DistributedVariableUpdateTest(tf.test.TestCase):
  """Tests that variables are updated correctly in distributed mode."""

  def _test_variable_update(self,
                            test_name,
                            num_workers,
                            num_ps,
                            params,
                            num_controllers=0):
    """Tests variables are updated correctly when the given params are used."""
    output_dir_path = os.path.join(test_name, 'variable_update')
    logs = _spawn_benchmark_processes(output_dir_path, num_workers, num_ps,
                                      num_controllers, params)
    actual_losses = []
    for worker_logs in logs:
      outputs = test_util.get_training_outputs_from_logs(
          worker_logs, params.print_training_accuracy)
      actual_losses.append([x.loss for x in outputs])

    inputs = test_util.get_fake_var_update_inputs()
    expected_losses = test_util.TestCNNModel().manually_compute_losses(
        inputs, num_workers, params)
    if params.variable_update == 'distributed_all_reduce':
      # In distributed all reduce, each step, the controller outputs the average
      # of the loss from each worker. So we modify expected losses accordingly.
      # E.g, we change [[1, 2], [4, 5]] to [[2.5, 3.5]]
      expected_losses = [[sum(losses) / num_workers
                          for losses in zip(*expected_losses)]]
    rtol = 3e-2 if params.use_fp16 else 1e-5
    for worker_actual_losses, worker_expected_losses in zip(actual_losses,
                                                            expected_losses):
      self.assertAllClose(worker_actual_losses[:len(worker_expected_losses)],
                          worker_expected_losses, rtol=rtol, atol=0.)

  def _test_variable_updates(self, test_name, params):
    """Tests variables are updated correctly with various variable updates."""

    # Unfortunately, distributed parameter server is non-deterministic with
    # multiple workers, because one worker may write to a variable before
    # another worker reads it. This probably does not harm training, but it
    # does mean we cannot easily test that case. So, we use one worker.
    self._test_variable_update(
        test_name + '_ps', num_workers=1, num_ps=2, num_controllers=0,
        params=params._replace(variable_update='parameter_server'))

    self._test_variable_update(
        test_name + '_rep', num_workers=2, num_ps=1, num_controllers=0,
        params=params._replace(variable_update='distributed_replicated'))

    self._test_variable_update(
        test_name + '_allreduce', num_workers=2, num_ps=0, num_controllers=1,
        params=params._replace(variable_update='distributed_all_reduce',
                               all_reduce_spec='psgpu#%d' % params.num_gpus))

  def testVarUpdateDefault(self):
    params = test_util.get_var_update_params()
    self._test_variable_updates('testVarUpdateDefault', params)

  def testVarUpdateCpuAsLocalParamDevice(self):
    params = test_util.get_var_update_params()._replace(
        local_parameter_device='cpu')
    self._test_variable_updates('testVarUpdateCpuAsLocalParamDevice', params)

  def testVarUpdateFp16(self):
    params = test_util.get_var_update_params()._replace(use_fp16=True)
    self._test_variable_updates('testVarUpdateFp16', params)

  def testVarUpdateResourceVars(self):
    params = test_util.get_var_update_params()._replace(use_resource_vars=True)
    self._test_variable_updates('testVarUpdateResourceVars', params)


if __name__ == '__main__':
  tf.test.main()
