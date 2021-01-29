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

import collections

import tensorflow.compat.v1 as tf
import tf_slim as slim

import functions


def createGenerator(inputs, out_size, layers, extra_name=None):
  assert extra_name
  name = 'sqn_' + extra_name
  with tf.variable_scope(name):
    ins = slim.layers.fully_connected(
        inputs, 1000, scope = 'fully_connected1',
        reuse=tf.AUTO_REUSE, activation_fn=tf.nn.relu, trainable=False)
    ins = tf.concat([ins, inputs], 1)

    for i in range(layers):
      ins = slim.layers.fully_connected(
          ins, 1000, scope = 'fully_connected%d' % (i + 2),
          reuse=tf.AUTO_REUSE, activation_fn=tf.nn.relu, trainable=False)
      ins = tf.concat([ins, inputs], 1)

    ins = slim.layers.fully_connected(
        ins, out_size, scope = 'fully_connected_final',
        reuse=tf.AUTO_REUSE, activation_fn=None, trainable=False)
  return ins


generator_nets = {
    'arch1': lambda *args, **kwargs: createGenerator(
        *args, layers=4, **kwargs),
    'arch2': lambda *args, **kwargs: createGenerator(
        *args, layers=6, **kwargs),
}


def createController(inputs, out_size, act=tf.nn.relu):
  ins = slim.layers.fully_connected(
      inputs, 1000, scope='controller_fully_connected_1',
      reuse=tf.AUTO_REUSE, activation_fn=act, trainable=False)
  for l in range(9):
    ins = slim.layers.fully_connected(
        ins, 1000, scope='controller_fully_connected_%d' % (l + 2),
        reuse=tf.AUTO_REUSE, activation_fn=act, trainable=False)

  ins = slim.layers.fully_connected(
      ins, out_size, scope = 'controller_fully_connected_final',
      reuse=tf.AUTO_REUSE, activation_fn=tf.nn.tanh, trainable=False)
  ins = tf.clip_by_value(ins, -0.99999, 0.99999)
  return ins


# Description of a generator functions. These specs tell us which net
# architecture to build and which weights to load from which
# checkpoint filename.
GeneratorSpec = collections.namedtuple('GeneratorSpec', [
    # relu, pieceLinear, triangle, square, or parabola.
    'function_name',
    # Key into generator_nets.
    'network_arch'])


def createNet(generator_specs, num_samples):
  """Constructs the net, including the controller and generators.

  Returns:
    target_placeholder: tf.placeholder shape [1, num_samples]. The target
      function input to the net.
    net_output: (tf.Tensor of float shape [1, num_samples]) The net output, V.
    cnp: The output commands and parameters from each generator.
    residual: target_placeholder - net_output.

  cnp stands for "commands and parameters". cnp[i] is a list
  [command, multiplier, offset] for generator i.
  command is a Tensor shape [1, 2] for Phi_i and lambda_i.
  multiplier and offset are each a Tensor shape [1, 1] for A_i and k_i.
  """
  # Target function. Changes at each residual iteration.
  target_placeholder =  tf.placeholder(
      tf.float32, [1, num_samples], name = "target_placeholder")

  # Create controller net.
  num_generators = len(generator_specs)
  insize = 4  # 2 for command (start, period) plus multiplier and offset

  # At the end, generator_cnps is shape [1, num_generators,
  # insize]. It stores the commands and parameters output by the
  # controller for each generator.
  generator_cnp = createController(target_placeholder, num_generators * insize)
  generator_cnp = tf.split(generator_cnp, num_generators, axis=1)
  generator_cnp = tf.stack(generator_cnp, axis=1)

  # commands, multipliers, and offset are the unpacked insize
  # dimension for all the generators.  They are shape [1,
  # num_generators, 2], [1, num_generators, 1], and [1,
  # num_generators, 1], just like the cnp return values except with
  # the added num_generators dimension.
  commands    = tf.slice(generator_cnp, [0, 0, 0], [-1, -1, 2])
  multipliers = tf.slice(generator_cnp, [0, 0, 2], [-1, -1, 1])
  offsets     = tf.slice(generator_cnp, [0, 0, 3], [-1, -1, 1])

  # Run each generator using its generator_cnp values.
  #
  # net_output is the sum of the generator outputs, ie.  the initial
  # approximation of the target function plus the approximation of all
  # the residuals.
  #
  # cnp is a re-shaping of generator_cnp to be a list of commands and
  # parameters for each generator.
  net_output = tf.zeros_like(target_placeholder)
  cnp = []
  for spec, command, multiplier, offset in zip(
      generator_specs,
      tf.unstack(commands, axis=1),
      tf.unstack(multipliers, axis=1),
      tf.unstack(offsets, axis=1)):
    cnp.append([command, multiplier, offset])
    net_fn = generator_nets[spec.network_arch]
    net = net_fn(command, num_samples, extra_name=spec.function_name)
    net = net * multiplier + offset
    net_output += net

  residual = target_placeholder - net_output
  return target_placeholder, net_output, cnp, residual
