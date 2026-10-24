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

"""Tests for the train state used by SNERG.

These cover the optimizer plumbing that `train.py`, `eval.py`, `bake.py` and
`snerg/eval_and_refine.py` all rely on: building the state, driving the learning
rate per step, taking a gradient step, and checkpointing. They are small enough
to run on CPU.
"""

import tempfile

from absl.testing import absltest
import flax
import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
import optax

from snerg.nerf import utils


class _TinyModel(nn.Module):
  """A stand-in for `NerfModel` with the same init/apply shape."""

  @nn.compact
  def __call__(self, x, scale=1.0):
    x = nn.Dense(8)(x)
    x = nn.relu(x)
    return nn.Dense(1)(x) * scale


def _make_state(learning_rate=0.1):
  model = _TinyModel()
  variables = model.init(jax.random.PRNGKey(0), jnp.ones((4, 3)))
  return model, variables, utils.create_train_state(
      model, variables, learning_rate)


def _loss(params, model, x, target):
  return jnp.mean((model.apply(params, x) - target)**2)


class CreateTrainStateTest(absltest.TestCase):

  def test_state_holds_the_whole_variables_dict(self):
    """`model.apply(state.params, ...)` must work, as the render path relies on it."""
    model, variables, state = _make_state()
    self.assertEqual(variables.keys(), {'params'})
    self.assertIsInstance(state.params, dict)
    self.assertIn('params', state.params)
    # Rendering code passes `state.params` straight to `model.apply`.
    out = model.apply(state.params, jnp.ones((2, 3)))
    self.assertEqual(out.shape, (2, 1))

  def test_starts_at_step_zero(self):
    """`train.py` computes `init_step = state.step + 1` to resume training."""
    _, _, state = _make_state()
    self.assertEqual(int(state.step), 0)

  def test_uses_adam(self):
    _, _, state = _make_state()
    self.assertIsInstance(state.tx, optax.GradientTransformationExtraArgs)


class ApplyGradientsTest(absltest.TestCase):

  def test_step_updates_params_and_counter(self):
    model, _, state = _make_state()
    x = jnp.ones((4, 3))
    target = jnp.zeros((4, 1))
    before = jax.tree_util.tree_leaves(state.params)

    grad = jax.grad(_loss)(state.params, model, x, target)
    state = state.apply_gradients(grads=grad)

    after = jax.tree_util.tree_leaves(state.params)
    self.assertEqual(int(state.step), 1)
    for new, old in zip(after, before):
      self.assertFalse(np.allclose(new, old), 'a parameter did not move')

  def test_learning_rate_is_used_per_step(self):
    """`update_learning_rate` must change the step size the optimizer takes.

    `train.py` recomputes the rate every step from `learning_rate_decay` and
    passes it into `train_step`, so a state whose rate was never updated - or
    an update that the optimizer ignored - would train at the wrong rate for
    the whole run.
    """
    x = jnp.ones((4, 3))
    target = jnp.zeros((4, 1))
    steps = []
    for learning_rate in (1.0, 0.001):
      model, variables, state = _make_state(learning_rate)
      grad = jax.grad(_loss)(state.params, model, x, target)
      state = utils.update_learning_rate(state, learning_rate)
      state = state.apply_gradients(grads=grad)
      delta = jax.tree_util.tree_map(lambda a, b: a - b, state.params, variables)
      steps.append(float(sum(float(jnp.abs(d).sum())
                             for d in jax.tree_util.tree_leaves(delta))))
    # Adam's first step is ~learning_rate per element, so a 1000x larger rate
    # must give a much larger update.
    self.assertGreater(steps[0], 50 * steps[1])

  def test_learning_rate_can_be_changed_after_creation(self):
    """The rate is injected, so replacing it must take effect immediately."""
    model, _, state = _make_state(1.0)
    x = jnp.ones((4, 3))
    target = jnp.zeros((4, 1))
    grad = jax.grad(_loss)(state.params, model, x, target)

    small = utils.update_learning_rate(state, 1e-4).apply_gradients(grads=grad)
    large = utils.update_learning_rate(state, 1.0).apply_gradients(grads=grad)
    self.assertFalse(np.allclose(small.params['params']['Dense_0']['kernel'],
                                 large.params['params']['Dense_0']['kernel']))
    self.assertAlmostEqual(
        float(state.opt_state.hyperparams['learning_rate']), 1.0, places=6)
    self.assertAlmostEqual(
        float(utils.update_learning_rate(state,
                                         0.5).opt_state.hyperparams[
                                             'learning_rate']), 0.5, places=6)

  def test_adam_moments_accumulate(self):
    """A second step reuses the first step's moments, so mu is not reset."""
    model, _, state = _make_state()
    x = jnp.ones((4, 3))
    target = jnp.zeros((4, 1))
    grad = jax.grad(_loss)(state.params, model, x, target)
    first = state.apply_gradients(grads=grad)
    second = first.apply_gradients(grads=grad)
    mu_first = first.opt_state.inner_state[0].mu['params']['Dense_0']['kernel']
    mu_second = second.opt_state.inner_state[0].mu['params']['Dense_0'][
        'kernel']
    self.assertGreater(float(jnp.abs(mu_first).sum()), 0.0)
    # Same gradient twice: mu must keep growing towards it, not restart.
    self.assertGreater(float(jnp.abs(mu_second).sum()),
                       float(jnp.abs(mu_first).sum()))
    self.assertEqual(int(second.step), 2)


class CheckpointTest(absltest.TestCase):
  """`eval.py` and `bake.py` block on `checkpoints.restore_checkpoint`."""

  def test_roundtrip_restores_params_and_step(self):
    model, _, state = _make_state()
    x = jnp.ones((4, 3))
    target = jnp.zeros((4, 1))
    grad = jax.grad(_loss)(state.params, model, x, target)
    trained = state.apply_gradients(grads=grad)

    with tempfile.TemporaryDirectory() as tmpdir:
      saved = jax.device_get(trained)
      checkpoints.save_checkpoint(tmpdir, saved, int(trained.step), keep=1)
      restored = checkpoints.restore_checkpoint(tmpdir, state)

    self.assertEqual(int(restored.step), 1)
    for a, b in zip(jax.tree_util.tree_leaves(restored.params),
                    jax.tree_util.tree_leaves(trained.params)):
      np.testing.assert_allclose(a, b)

  def test_pmap_replicate_roundtrip(self):
    """`train.py` replicates the state across local devices."""
    _, _, state = _make_state()
    replicated = flax.jax_utils.replicate(state)
    self.assertEqual(replicated.step.shape, (jax.local_device_count(),))
    unreplicated = flax.jax_utils.unreplicate(replicated)
    self.assertEqual(int(unreplicated.step), int(state.step))


if __name__ == '__main__':
  absltest.main()
