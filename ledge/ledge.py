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

"""This is the code associated with our paper "Interpretable Actions:
Controlling Experts with Understandable Commands" by Shumeet Baluja,
David Marwood, and Michele Covell.  The paper is published in the
thirty-fifth Conference on Artificial Intelligence (AAAI-21).

This code implements the Function Approximator example from the paper,
approximately reconstructing a 1-D function. Following the paper, the
results of step 1 and step 2 are provided in the ckpts directory. This
code performs:

  Step 3: Correct the residual error.
  Step 4: Extract the interpretable commands. Commands are printed
    to the console.
  Step 5: Fine-tune the extracted commands.

As a simple demo:
```
python -m ledge
eog /tmp/refit_fig0.png
```

For a list of flags and options:
```
python -m ledge --help
```

Some terminology used in the code:
  - function: A periodic 1-D function, one of relu, pieceLinear,
    triangle, square, parabola, or sinusoid. Figure 3 f1-f5.
  - generator: An implementation of a function as a DNN that outputs
    samples from that function. G in the paper.
  - controller: A DNN that outputs the commands for the generators
    (Phi and lambda) and the parameters (A and k) for amplitude scale
    and offset.  C in the paper.
  - the net: The combination of the controller and generators in
    Figure 4. The output of the net is after the sum of generators, V
    in the paper, a sampled approximation of the target function. "The
    net" doesn't include residual iterations.
  - basis: An external basis function. B in the paper.
  - samples: A sequence of sampled values from a basis or generator.
  - target function: The sampled input to the controller. Figure 3 top.
"""

import math
import numpy as np
import random

from absl import flags
import tensorflow.compat.v1 as tf

import networks
import utils
import functions

tf.compat.v1.disable_eager_execution()


# Number of samples output from each generator, or n from the paper.
num_samples = 500

# A "pass" is one execution of the net. num_passes is the number of
# passes, including residuals, t in the paper.
num_passes = 4


flags.DEFINE_string (
    'checkpoints_dir', 'ckpts',
    'Path to generator and controller checkpoints.')
flags.DEFINE_string (
    'out_dir', '/tmp',
    'Output graphs are placed here.')
flags.DEFINE_string (
    'target_function_file', 'targets/approximate_20',
    'Target function. File with two columns x & y. x is 0..num_samples in'
    ' order and y is [-1,1]')
FLAGS = flags.FLAGS


def getTargetFunction():
  """Reads the target  function from --target_function_file.

  Returns: List of float, len num_samples.
  """
  f = tf.io.gfile.GFile(FLAGS.target_function_file, 'r')
  func = []
  for v in range(num_samples):
    l = f.readline()
    l = l.split()
    assert int(l[0]) == v
    func.append(float(l[1]))

  return func


def generatorVariablesToRestore(extraName):
  """Returns generator variables to restore from the checkpoint.

  Each generator checkpoint contains pre-computed weights and biases
  for one basis function.

  It additionally contains variables for the controller and the other
  generators because these get saved during training. These are
  ignored. We return only the weights and biases for the extraName
  generator.

  Returns: A list of tf.Variables.
  """
  restoreThese = []
  for x in tf.global_variables():
   if 'sqn' in x.name:        # Include only generator vars.
     if extraName in x.name:  # Only vars for generator extraName.
       if 'weights' in x.name or 'biases' in x.name:  # Only weights and biases.
         restoreThese.append(x)

  return restoreThese


def controllerVariablesToRestore():
  """Returns controller variables to restore from the checkpoint.
  """
  restoreThese = []
  for x in tf.global_variables():
    if not 'sqn' in x.name:          # Exclude generator vars.
      assert not 'global' in x.name  # No global_step to restore.
      restoreThese.append(x)
  return restoreThese


def createCheckpointRestorers(generator_specs):
  """Each returned list entry is a (directory_name, tf.train.Saver).
  The Saver contains the variables to restore. The directory_name is
  the checkpoint to restore from.

  Contains both the controller and the generator checkpoints. This
  allows deferring restores until after initialization.
  """
  checkpoints_to_restore = []
  for spec in generator_specs:
    variablesInNet = generatorVariablesToRestore(spec.function_name)
    assert variablesInNet
    saver = tf.train.Saver(variablesInNet)

    directory = '%s/%sFunction_weights/' % (
        FLAGS.checkpoints_dir, spec.function_name)
    checkpoints_to_restore.append((directory, saver))

  saver = tf.train.Saver(controllerVariablesToRestore())
  directory = '%s/controller_weights' % FLAGS.checkpoints_dir
  checkpoints_to_restore.append((directory, saver))
  return checkpoints_to_restore


def restoreCheckpoints(checkpoints_to_restore, sess):
  for directory, saver in checkpoints_to_restore:
    latest_ckp = tf.train.latest_checkpoint(directory)
    if not latest_ckp:
      print('No checkpoints exists in directory', directory)
      assert False
    print('Loading checkpoint from', latest_ckp)
    saver.restore(sess, latest_ckp)
  print('Done loading checkpoints.')


def runNetPasses(sess, target_placeholder, target_func, net_output,
                 cnp, residual):
  """Runs the net for num_passes.

  Step 3 in the paper.

  Args:
    target_placeholder: (tf.placeholder shape [1, num_samples]) For feeding
      the target function to the net.
    target_func: (np.array of float shape [1, num_samples]) The initial
      target function.
    net_output: (tf.Tensor of float shape [1, num_samples]) The net output, V.
    cnp: The output commands and parameters from each generator, see below.
    residual: target_func - net_output.

  cnp stands for "commands and parameters". cnp[i] is a list
  [commands, multiplier, offset] for generator i.
  commands is a Tensor shape [1, 2] for Phi_i and lambda_i.
  multiplier and offset are each a Tensor shape [1, 1] for A_i and k_i.

  Returns: A list of the computed cnps for each pass.
  """
  predictions = []  # Sum of the net_output from the first i+1 passes.

  # List of cnps for each pass.
  pass_cnps = []

  pass_target = target_func
  for pass_num in range(0, num_passes):
    with sess.as_default():
      net_output_np, cnp_np, residual_np  = sess.run(
          [net_output, cnp, residual],
          feed_dict={target_placeholder: pass_target})
    pass_target = residual_np
    if not predictions:
      predictions.append(net_output_np)
    else:
      predictions.append(net_output_np + predictions[-1])

    pass_cnps.append(cnp_np)

    error = math.sqrt(np.sum(np.square(target_func - predictions[-1])))
    print('Pass %d: error sqrt (sum(error*error)): %f' %
          (pass_num, error))

  utils.graphBatch([target_func[0]],
                   FLAGS.out_dir + '/target.png',
                   title = 'Target',
                   ylim=[-1.5, 1.5])

  utils.graphBatch(
      [target_func[0]] + [p[0] for p in predictions],
      FLAGS.out_dir + '/networkPredictions.png',
      title='Target + Network Predictions with Residuals',
      labels=['Target', 'Network Prediction'] + [
          '+Residual Correction %d' % i for i in range(1, len(predictions))],
      ylim=[-1.5, 1.5])

  return pass_cnps


def reconstructUsingBases(generator_specs, pass_cnps, target_func):
  """Apply the commands to the basis (rather than generators).

  Step 4 in the paper.

  Writes graphs to --out_dir/extractedCommands*.png comparing the
  target_func to the basis after applying the multiplier and offset.

  pass_cnps contains the commands and parameters for each pass and
  for each generator.

  Returns: (list of np.array shape [1, num_samples]) Each entry is a
    sampled basis for each generator before the parameters are applied.
  """
  # The sampled basis for each function before applying the multiplier
  # and offset. There are num_passes * len(generator_specs) bases in
  # this list.
  bases_before_params = []

  # The sampled function after each pass, including multiplier and
  # offset. Written to the graphs.
  command_fits = []

  for pass_num in range(num_passes):
    current = np.zeros([num_samples])
    for gen_num in range(len(generator_specs)):
      start = pass_cnps[pass_num][gen_num][0][0][0]       # Phi
      period = pass_cnps[pass_num][gen_num][0][0][1]      # lambda
      multiplier = pass_cnps[pass_num][gen_num][1][0][0]  # A
      offset = pass_cnps[pass_num][gen_num][2][0][0]      # k

      print('Pass %d function %d: command (unscaled) (%f, %f),'
            ' multiplier %f, offset %f' % (
            pass_num, gen_num, start, period, multiplier, offset))

      sampled_basis = functions.applyFunc(
          start, period, num_samples, generator_specs[gen_num].function_name)
      bases_before_params.append(sampled_basis.copy())
      sampled_basis *= multiplier
      sampled_basis += offset

      current += sampled_basis

    if pass_num == 0:
      command_fits.append(current)
    else:
      command_fits.append(current + command_fits[-1])

    err = target_func[0] - command_fits[-1]
    print('recon based on commands iteration: %d : %f' % (
        pass_num, math.sqrt(np.sum(err*err))))

  utils.graphBatch(
      [target_func[0]] + command_fits,
      FLAGS.out_dir + '/extractedCommands.png',
      title='Target + Extracted Commands',
      labels=['Target'] + [
          'Extracted Commands (pass %d)' % i for i in range(len(command_fits))],
      ylim=[-1.5, 1.5])

  return bases_before_params


def reweightSampledBases(bases_before_params, target_func):
  """Compute optimal parameters to fit the bases to the target_func.

  Step 5 in the paper.

  Writes graphs comparing the optimized parameters to the target_func
  into --out_dir/refit*.png.

  Args:
    bases_before_params: The sampled bases produced by the commands.
      List of length len(generator_specs) * num_passes since every
      generator in every pass produces one command. Each entry
      is an np.array shape [1, num_samples].
  """
  # Av is shape [num_samples, len(bases_before_params) + 1].
  # As described in Step 5, we compute just one Delta parameter that
  # is the sum of k_i.  The added row of Av computes Delta.
  Av = np.transpose(bases_before_params)
  Av = np.concatenate([Av, np.ones([num_samples, 1])], axis=1)

  # bv is shape [num_samples, 1]
  bv = np.transpose(target_func)

  A = tf.placeholder(tf.float32, Av.shape, name='A')
  b = tf.placeholder(tf.float32, bv.shape, name='b')
  linsol = tf.linalg.lstsq(A, b)

  sess = tf.Session()
  done = False
  while not done:
    try:
      with sess.as_default():
        x = sess.run(linsol, feed_dict={A:Av, b:bv})
      done = True
    except tf.errors.InvalidArgumentError as e:
      print('Cholesky likely failed. Adding miniscule noise.')
      for c in range(Av.shape[1] - 1):
        Av[random.randint(0, Av.shape[0] - 1)][c] *= random.uniform (
            0.999, 1.001)

  ans = np.matmul(Av, x)
  err = ans - bv
  print('linSolve final error:', math.sqrt(np.sum(err*err)))

  utils.graphBatch([target_func[0], ans],
                   FLAGS.out_dir + '/refit.png',
                   title='After Reweighting',
                   labels=['Target', 'After Reweighting'],
                   ylim=[-1.5, 1.5])


def main(_):
  # Construct these 5 generators and load them from the checkpoints.
  generator_specs = [
      networks.GeneratorSpec(
          function_name='relu', network_arch='arch1'),
      networks.GeneratorSpec('pieceLinear', 'arch2'),
      networks.GeneratorSpec('triangle', 'arch2'),
      networks.GeneratorSpec('square', 'arch1'),
      networks.GeneratorSpec('parabola', 'arch2')]

  # Creates (but doesn't run) the controller and the
  # generators. target_placeholder is the controller input, an
  # arbitrary sampled target function. The rest are outputs.
  target_placeholder, net_output, cnp, residual = (
      networks.createNet(generator_specs, num_samples))

  # Contains the generator and controller checkpoints.
  checkpoints_to_restore = createCheckpointRestorers(generator_specs)

  sess = tf.Session ()
  with sess.as_default ():
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
  restoreCheckpoints(checkpoints_to_restore, sess)

  # Step 3: Run the net once for each pass to compute the commands and
  # parameters pass_cnps.
  target_func = np.array([getTargetFunction()])
  pass_cnps = runNetPasses(
      sess, target_placeholder, target_func, net_output, cnp, residual)

  # Step 4: Apply the computed commands to the bases.
  bases_before_params = reconstructUsingBases(
      generator_specs, pass_cnps, target_func)

  # Step 5: Optimize the parameters to fit the bases to the target_func.
  reweightSampledBases(bases_before_params, target_func)


if __name__ == '__main__':
  tf.app.run()
