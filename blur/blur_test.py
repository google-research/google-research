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

"""Unit tests for BLUR."""
import numpy as np
import tensorflow.compat.v1 as tf

from blur import blur
from blur import blur_env
from blur import blur_meta
from blur import genome_util
from blur import synapse_util


def d_sigmoid(x):
  s = tf.math.sigmoid(x)
  return s * (1 - s)


def sigmoid_with_grad(x):
  return tf.stack([tf.math.sigmoid(x[Ellipsis, 0]), d_sigmoid(x[Ellipsis, 1])], axis=-1)


def random_dataset():
  n = 1000
  ds = tf.data.Dataset.from_tensor_slices({
      'support': (
          np.random.normal(0, 255,
                           size=(0, 1, 1, n,
                                 784)).astype(blur_env.NP_FLOATING_TYPE),
          np.random.randint(0, 10, size=(0, 1, 1, n,
                                         2)).astype(blur_env.NP_FLOATING_TYPE),
      )})
  return ds


def random_dense(num_in=10, num_out=5):
  inp = np.random.random((1, 1, 1, num_in, 1))
  out = np.random.random((1, 1, 1, num_out, 1))

  # Neuron: input + [1] + output.
  num_in_bias = num_in + 1
  ow = np.random.random((1, 1, num_in_bias, num_out))

  return inp, out, ow


def get_blur_state(env, inp, out, ow):
  pre = np.concatenate([inp, np.zeros_like(inp)],
                       axis=-1).astype(blur_env.NP_FLOATING_TYPE)
  post = np.concatenate([np.zeros_like(out), out],
                        axis=-1).astype(blur_env.NP_FLOATING_TYPE)
  ww = ow.astype(blur_env.NP_FLOATING_TYPE)
  ww = ww[Ellipsis, None]
  synapse = synapse_util.combine_in_out_synapses(
      ww, synapse_util.transpose_synapse(ww, env), env=env)
  synapse = synapse_util.sync_states_synapse(synapse, env, num_states=2)

  genome = genome_util.convert_genome_to_tf_variables(
      genome_util.create_backprop_genome(num_species=1))

  network_spec = blur.NetworkSpec()
  network_spec.symmetric_synapses = True
  network_spec.batch_average = False
  network_spec.backward_update = 'multiplicative_second_state'

  return pre, post, synapse, genome, network_spec


def tf_gradients(inp, out, ow):
  tfinp = tf.constant(inp[Ellipsis, 0])

  # Append '1' to the end of the input
  bias = tf.constant([[[[1]]]], tfinp.dtype)
  inp_with_bias = tf.concat([tfinp, bias], axis=-1)
  tfw = tf.constant(ow)
  y = inp_with_bias @ tfw
  _, grad_weights, grad_image = tf.gradients(y, [bias, tfw, tfinp], out[Ellipsis, 0])
  return grad_weights, grad_image, y


def verify_equal(update1, update2, hebbian_update, grad_weights, grad_image, y,
                 num_in, num_out):
  # Hebbian update is [#in + #out, #in + #out, 2] matrix
  # and it should look like this.
  # Z   U
  # Z   Z
  #
  # Z   Z
  # U^T Z
  # Where Z is zeros
  np.testing.assert_allclose(
      hebbian_update[Ellipsis, 0],
      np.swapaxes(hebbian_update[Ellipsis, 1], -1, -2),
      rtol=1e-5)

  np.testing.assert_allclose(
      hebbian_update[Ellipsis, :num_in + 1, num_in + 1:, 0], grad_weights, rtol=1e-5)

  np.testing.assert_allclose(hebbian_update[Ellipsis, :num_in + 1, :num_in, 0],
                             np.zeros((1, 1, num_in + 1, 10)))
  np.testing.assert_allclose(hebbian_update[Ellipsis, num_in + 1:, num_in + 1:, 0],
                             np.zeros((1, 1, num_out, num_out)))

  np.testing.assert_allclose(update1[Ellipsis, 1], grad_image, rtol=1e-5)
  np.testing.assert_allclose(update1[Ellipsis, 0], np.zeros_like(update1[Ellipsis, 0]))

  np.testing.assert_allclose(update2[Ellipsis, 0], y, rtol=1e-5)


class BlurTest(tf.test.TestCase):

  def test_sync_in_out_synapses(self):
    num_in = 3
    num_out = 2
    num_states = 2
    env = blur_env.tf_env
    in_out_synapse = tf.random.normal(shape=(num_in + 1, num_out, num_states))
    out_in_synapse = tf.random.normal(shape=(num_out, num_in + 1, num_states))

    synapse = synapse_util.combine_in_out_synapses(in_out_synapse,
                                                   out_in_synapse, env)
    synapse_synced = synapse_util.sync_in_and_out_synapse(synapse, num_in, env)
    fwd_sync_submatrix = synapse_util.synapse_submatrix(
        synapse_synced,
        num_in,
        synapse_util.UpdateType.FORWARD,
        include_bias=True)

    bkw_sync_submatrix = synapse_util.synapse_submatrix(
        synapse_synced,
        num_in,
        synapse_util.UpdateType.BACKWARD,
        include_bias=True)

    with tf.Session() as s:
      bwd, fwd, inp = s.run([
          synapse_util.transpose_synapse(bkw_sync_submatrix, env),
          fwd_sync_submatrix, in_out_synapse
      ])
    self.assertAllEqual(fwd, inp)
    self.assertAllEqual(bwd, inp)

  def test_verify_gradient_match_tf(self):
    num_in = 10
    num_out = 15
    tf.reset_default_graph()
    tf.disable_v2_behavior()
    inp, out, ow = random_dense(num_in, num_out)
    env = blur_env.tf_env
    pre, post, synapse, _, network_spec = get_blur_state(env, inp, out, ow)

    genome = genome_util.create_backprop_genome(num_species=1)

    update1, update2 = blur.get_synaptic_update(
        pre,
        post,
        synapse=synapse,
        input_transform_gn=genome.neuron.transform,
        update_type=synapse_util.UpdateType.BOTH,
        env=env)

    hebbian_update = blur.get_hebbian_update(
        pre,
        post,
        genome.synapse.transform,
        global_spec=network_spec,
        env=env)

    grad_weights, grad_image, y = tf_gradients(inp, out, ow)

    np.set_printoptions(precision=4, linewidth=200)

    with tf.Session():
      verify_equal(update1.eval(), update2.eval(), hebbian_update.eval(),
                   grad_weights.eval(), grad_image.eval(), y.eval(), num_in,
                   num_out)

  def test_verify_gradient_match_jp(self):
    tf.reset_default_graph()
    tf.disable_v2_behavior()
    num_in = 10
    num_out = 15
    inp, out, ow = random_dense(num_in, num_out)
    env = blur_env.jp_env
    pre, post, synapse, _, network_spec = get_blur_state(env, inp, out, ow)

    genome = genome_util.create_backprop_genome(num_species=1)

    update1, update2 = blur.get_synaptic_update(
        pre,
        post,
        synapse=synapse,
        input_transform_gn=genome.neuron.transform,
        update_type=synapse_util.UpdateType.BOTH,
        env=env)

    hebbian_update = blur.get_hebbian_update(
        pre,
        post,
        genome.synapse.transform,
        global_spec=network_spec,
        env=env)
    grad_weights, grad_image, y = tf_gradients(inp, out, ow)

    np.set_printoptions(precision=4, linewidth=200)

    with tf.Session():
      verify_equal(update1, update2, hebbian_update, grad_weights.eval(),
                   grad_image.eval(), y.eval(), num_in, num_out)

  def test_get_synaptic_update_forward(self):
    tf.reset_default_graph()
    inp, out, w = random_dense()
    env = blur_env.tf_env
    pre, post, synapse, genome, _, = get_blur_state(env, inp, out, w)

    _, update_fwd = blur.get_synaptic_update(
        pre,
        post,
        synapse,
        input_transform_gn=genome.neuron.transform,
        update_type=synapse_util.UpdateType.FORWARD,
        env=env)

    inp = tf.constant(inp.astype(blur_env.NP_FLOATING_TYPE))
    ww = w.astype(blur_env.NP_FLOATING_TYPE)
    inp_with_bias = tf.concat([inp[Ellipsis, 0], [[[[1]]]]], axis=-1)
    exp_results = inp_with_bias @ ww

    with tf.Session() as s:
      s.run(tf.initialize_all_variables())
      self.assertAllClose(update_fwd[Ellipsis, 0], exp_results)
      self.assertAllClose(update_fwd[Ellipsis, 1], exp_results)

  def test_network_step_mix_forward(self):
    spec = blur.NetworkSpec(use_forward_activations_for_synapse_update=True)
    genome = genome_util.convert_genome_to_tf_variables(
        genome_util.create_backprop_genome(num_species=1))
    initializer = lambda params: 2 * tf.ones(params.shape, dtype=tf.float32)
    data = random_dataset()
    state = blur_meta.init_first_state(
        genome,
        synapse_initializer=initializer,
        data=data,
        hidden_layers=[256, 128])
    input_fn = data.make_one_shot_iterator().get_next
    data_support_fn, _ = blur_meta.episode_data_fn_split(input_fn)
    blur.network_step(
        state, genome, data_support_fn, network_spec=spec, env=blur_env.tf_env)
    g = tf.get_default_graph()

    synapse_pre = g.get_operation_by_name(
        'step/backward/synapse_update/hebbian_pre').inputs[0]
    synapse_post = g.get_operation_by_name(
        'step/backward/synapse_update/hebbian_post').inputs[0]
    self.assertIn('forward', synapse_pre.name)
    self.assertIn('backward', synapse_post.name)

    self.assertNotIn('backward', synapse_pre.name)
    self.assertNotIn('forward', synapse_post.name)

  def test_network_step_no_mix_forward(self):
    spec = blur.NetworkSpec(use_forward_activations_for_synapse_update=False)
    genome = genome_util.convert_genome_to_tf_variables(
        genome_util.create_backprop_genome(num_species=1))
    initializer = lambda params: 2 * tf.ones(params.shape, dtype=tf.float32)
    data = random_dataset()
    state = blur_meta.init_first_state(
        genome,
        synapse_initializer=initializer,
        data=data,
        hidden_layers=[256, 128])
    input_fn = data.make_one_shot_iterator().get_next
    data_support_fn, _ = blur_meta.episode_data_fn_split(input_fn)
    blur.network_step(
        state,
        genome,
        data_support_fn,
        data.make_one_shot_iterator().get_next,
        network_spec=spec,
        env=blur_env.tf_env)
    g = tf.get_default_graph()

    synapse_pre = g.get_operation_by_name(
        'step/backward/synapse_update/hebbian_pre').inputs[0]
    synapse_post = g.get_operation_by_name(
        'step/backward/synapse_update/hebbian_post').inputs[0]
    self.assertIn('backward', synapse_pre.name)
    self.assertIn('backward', synapse_post.name)
    self.assertNotIn('forward', synapse_pre.name)
    self.assertNotIn('forward', synapse_post.name)

  def test_get_synaptic_update_backward(self):
    tf.reset_default_graph()
    tf.disable_v2_behavior()
    n_in, n_out = 10, 5
    inp, out, w = random_dense(n_in, n_out)
    env = blur_env.tf_env
    pre, post, synapse, genome, _ = get_blur_state(env, inp, out, w)

    update_bwd, _ = blur.get_synaptic_update(
        pre,
        post,
        synapse,
        input_transform_gn=genome.neuron.transform,
        update_type=synapse_util.UpdateType.BACKWARD,
        env=env)

    out = tf.constant(out.astype(blur_env.NP_FLOATING_TYPE))
    ww = w.astype(blur_env.NP_FLOATING_TYPE)
    exp_results = out[Ellipsis, 0] @ tf.transpose(ww, (0, 1, 3, 2))

    with tf.Session() as s:
      s.run(tf.initialize_all_variables())
      self.assertAllClose(update_bwd[Ellipsis, 0], tf.zeros((1, 1, 1, n_in)))
      self.assertAllClose(update_bwd[Ellipsis, 1], exp_results[Ellipsis, :-1])

  def test_neuron_update_fwd(self):
    tf.reset_default_graph()
    tf.disable_v2_behavior()
    n_in, n_out = 10, 5
    inp, out, w = random_dense(n_in, n_out)
    env = blur_env.tf_env
    pre, post, synapse, genome, network_spec = get_blur_state(env, inp, out, w)
    pre_fwd, post_fwd = blur.dense_neuron_update(
        pre,
        post,
        synapse,
        inp_act=None,
        out_act=sigmoid_with_grad,
        neuron_genome=genome.neuron,
        update_type=synapse_util.UpdateType.FORWARD,
        global_spec=network_spec,
        env=env)

    inp = tf.constant(inp.astype(blur_env.NP_FLOATING_TYPE))
    ww = w.astype(blur_env.NP_FLOATING_TYPE)
    inp_with_bias = tf.concat([inp[Ellipsis, 0], [[[[1]]]]], axis=-1)
    exp_results = inp_with_bias @ ww

    with tf.Session() as s:
      s.run(tf.initialize_all_variables())
      self.assertAllClose(pre_fwd, pre)
      self.assertAllClose(post_fwd[Ellipsis, 0], tf.math.sigmoid(exp_results))
      self.assertAllClose(post_fwd[Ellipsis, 1],
                          d_sigmoid(out[Ellipsis, 0] + exp_results))

  def test_neuron_update_bwd(self):
    tf.reset_default_graph()
    tf.disable_v2_behavior()
    n_in, n_out = 10, 5
    inp, out, w = random_dense(n_in, n_out)
    env = blur_env.tf_env
    pre, post, synapse, genome, network_spec = get_blur_state(env, inp, out, w)

    inp_act_fn = lambda x: x
    out_act_fn = sigmoid_with_grad

    pre_fwd, post_fwd = blur.dense_neuron_update(
        pre,
        post,
        synapse,
        inp_act=None,
        out_act=out_act_fn,
        neuron_genome=genome.neuron,
        update_type=synapse_util.UpdateType.FORWARD,
        global_spec=network_spec,
        env=env)

    pre_bkw, _ = blur.dense_neuron_update(
        pre_fwd,
        post_fwd,
        synapse=synapse,
        inp_act=inp_act_fn,
        out_act=out_act_fn,
        neuron_genome=genome.neuron,
        update_type=synapse_util.UpdateType.BACKWARD,
        global_spec=network_spec,
        env=env)

    inp = tf.constant(inp.astype(blur_env.NP_FLOATING_TYPE))
    ww = w.astype(blur_env.NP_FLOATING_TYPE)

    exp_result = post_fwd[Ellipsis, 1] @ tf.transpose(ww, (0, 1, 3, 2))

    with tf.Session() as s:
      s.run(tf.initialize_all_variables())
      self.assertAllClose(pre_bkw[Ellipsis, 0], pre_fwd[Ellipsis, 0])
      self.assertAllClose(pre_bkw[Ellipsis, 1],
                          exp_result[Ellipsis, :-1] * pre_fwd[Ellipsis, 1])

  def test_get_hebbian_update(self):
    tf.reset_default_graph()
    tf.disable_v2_behavior()
    n_in, n_out = 10, 5
    inp, out, w = random_dense(n_in, n_out)
    env = blur_env.tf_env
    pre, post, _, genome, network_spec = get_blur_state(env, inp, out, w)

    hebbian_update = blur.get_hebbian_update(pre, post,
                                             genome.synapse.transform,
                                             network_spec, env)
    hebbian_update_submatrix = synapse_util.synapse_submatrix(
        hebbian_update, n_in, synapse_util.UpdateType.FORWARD)

    inp = tf.constant(inp.astype(blur_env.NP_FLOATING_TYPE))
    out = tf.constant(out.astype(blur_env.NP_FLOATING_TYPE))
    inp_transpose = tf.transpose(inp[Ellipsis, 0], (0, 1, 3, 2))
    exp_result = env.concat_row(inp_transpose) @ out[Ellipsis, 0]

    with tf.Session() as s:
      s.run(tf.initialize_all_variables())
      self.assertAllClose(hebbian_update_submatrix[Ellipsis, 0], exp_result)

  def test_synapse_derivative(self):
    tf.reset_default_graph()
    tf.disable_v2_behavior()
    n_in, n_out = 10, 5
    inp, out, w = random_dense(n_in, n_out)

    env = blur_env.tf_env
    pre, post, synapse, genome, network_spec = get_blur_state(env, inp, out, w)
    post = np.concatenate(
        2 * [np.zeros_like(out)], axis=-1).astype(blur_env.NP_FLOATING_TYPE)

    pre_fwd, post_fwd = blur.dense_neuron_update(
        pre,
        post,
        synapse,
        inp_act=None,
        out_act=sigmoid_with_grad,
        neuron_genome=genome.neuron,
        update_type=synapse_util.UpdateType.FORWARD,
        global_spec=network_spec,
        env=env)

    hebbian_update = blur.get_hebbian_update(pre_fwd, post_fwd,
                                             genome.synapse.transform,
                                             network_spec, env)

    hebbian_update_submatrix = synapse_util.synapse_submatrix(
        hebbian_update, n_in, synapse_util.UpdateType.FORWARD)

    inp = tf.constant(inp.astype(blur_env.NP_FLOATING_TYPE))
    inp_with_bias = tf.concat([inp[Ellipsis, 0], [[[[1]]]]], axis=-1)
    ww = tf.constant(w.astype(blur_env.NP_FLOATING_TYPE))
    out = tf.nn.sigmoid(inp_with_bias @ ww)
    grad_w = tf.gradients(out, ww)

    with tf.Session() as s:
      s.run(tf.initialize_all_variables())
      hebb_update, grad_w_val = s.run(
          [hebbian_update_submatrix[Ellipsis, 0], grad_w[0]])
    self.assertAllClose(hebb_update, grad_w_val)

