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

import itertools
from functools import partial
import os
from flax.training import checkpoints
from flax import optim
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import jax.random
from jax import vmap
from jax import custom_vjp

from tensorflow.io import gfile


def create_learning_rate_scheduler(
    factors="constant * linear_warmup * rsqrt_decay * rsqrt_hidden_size",
    base_learning_rate=0.5,
    warmup_steps=1000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000,
    hidden_size=1024):
  """creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: a string with factors separated by '*' that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: how many steps to warm up for in the warmup schedule.
    decay_factor: The amount to decay the learning rate by.
    steps_per_decay: How often to decay the learning rate.
    steps_per_cycle: Steps per cycle when using cosine decay.
    hidden_size: size of feature dimension in attention layers.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split("*")]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == "constant":
        ret *= base_learning_rate
      elif name == "linear_warmup":
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == "rsqrt_decay":
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "rsqrt_normalized_decay":
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "decay_every":
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == "cosine_decay":
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      elif name == "rsqrt_hidden_size":
        ret /= jnp.sqrt(1.0 * hidden_size)
      else:
        raise ValueError("Unknown factor %s." % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


@partial(jnp.vectorize, excluded=(2,), signature="(m),(m)->()")
def permutation_invariant_accuracy(predictions, labels, k):
  permutations = jnp.array(list(itertools.permutations(range(k))))
  permuted_labels = jax.lax.map(lambda p: p[labels], permutations)
  acc = jnp.max(
      jax.lax.map(lambda ls: jnp.mean(ls == predictions), permuted_labels))
  return acc


def has_checkpoint(logdir):
  return (gfile.isdir(logdir) and
          gfile.glob(os.path.join(logdir, "checkpoint_*")))


def load_parameters(logdir, init_params):
  if has_checkpoint(logdir):
    print("Loading checkpoint from %s" % logdir)
    optimizer_def = optim.Adam()
    optimizer = optimizer_def.create(init_params)
    optimizer = checkpoints.restore_checkpoint(logdir, optimizer)
    print("Checkpoint loaded from step %d" % optimizer.state.step)
    return optimizer.target
  else:
    print("No checkpoint found in %s" % logdir)
    return None


def maybe_load_checkpoint(logdir, optimizer, clobber_checkpoint=False):
  if not clobber_checkpoint:
    if has_checkpoint(logdir):
      print("Loading checkpoint from %s" % logdir)
      optimizer = checkpoints.restore_checkpoint(logdir, optimizer)
      print("Checkpoint loaded from step %d" % optimizer.state.step)
  else:
    if gfile.isdir(logdir):
      gfile.rmtree(logdir)
  return optimizer


def bernoulli_logpmf(logits, labels):
  """Bernoulli log pmf of data x given logits."""
  return -jnp.sum(
      jnp.logaddexp(0.,
                    jnp.where(labels, -1., 1.) * logits), axis=-1)


@partial(jax.numpy.vectorize, signature="(n),()->()")
def categorical_logpmf(logits, label):
  return jax.nn.log_softmax(logits)[label]


def categorical_kl(p_probs, q_log_probs):
  p_log_probs = jnp.log(p_probs)
  outs = jnp.where(p_probs > 0, p_probs*(p_log_probs - q_log_probs), p_probs)
  return jnp.sum(outs)


@partial(jax.numpy.vectorize, signature="(n,k),(n)->(n)")
def permutation_invariant_categorical_logpmf(logits, labels):
  k = logits.shape[-1]
  permutations = jnp.array(list(itertools.permutations(range(k))))
  permuted_labels = permutations[labels].T
  # [k!, num_data_points]
  all_lls = jax.vmap(categorical_logpmf, in_axes=(None, 0))(
      logits, permuted_labels)
  max_ll_ind = jax.lax.stop_gradient(jnp.argmax(jnp.sum(all_lls, axis=1)))
  return all_lls[max_ll_ind]


@jax.custom_vjp
def l2_dist(x, y):
  """A custom l2 dist with 0 gradient when x=y."""
  return jnp.linalg.norm(x-y)


def fwd_l2_dist(x, y):
  dist = l2_dist(x, y)
  return dist, (dist, x-y)


def bwd_l2_dist(res, g):
  dist, diff = res
  grd = jnp.where(dist <= 1e-8, jnp.zeros_like(diff), diff / dist)
  return (grd*g, -grd*g)


l2_dist.defvjp(fwd_l2_dist, bwd_l2_dist)


@partial(jax.numpy.vectorize, signature="(n,d),(m,d)->(n,m)")
def pair_dists(X, Y):
  return jax.vmap(jax.vmap(l2_dist, in_axes=(None, 0)), in_axes=(0, None))(X, Y)


def pair_vectors(vs):
  num_vs, _ = vs.shape
  expanded_v = jnp.tile(vs[:, jnp.newaxis, :], [1, num_vs, 1])
  matrix = jnp.concatenate(
      [expanded_v, jnp.transpose(expanded_v, axes=[1, 0, 2])], axis=2)
  return matrix[jnp.tril_indices(num_vs, k=-1)]


def subsample_pairs(key, vs, num_subsampled_pairs):
  paired_vs = pair_vectors(vs)
  num_pairs = paired_vs.shape[0]
  inds = jax.random.choice(
      key, num_pairs, (num_subsampled_pairs,), replace=False)
  return paired_vs[inds], inds


def to_pairwise_preds(preds):
  paired_preds = pair_vectors(preds[Ellipsis, jnp.newaxis])
  paired_preds = 1 * (paired_preds[Ellipsis, 0] == paired_preds[Ellipsis, 1])
  return paired_preds


@partial(jax.numpy.vectorize, signature="(n),(n)->()")
def binary_f1(y_pred, y_true):
  true_pos = jnp.sum((y_pred == y_true)*(y_true == 1))
  false_pos = jnp.sum((y_pred != y_true)*(y_true == 0))
  false_neg = jnp.sum((y_pred != y_true)*(y_true == 1))
  return true_pos / (true_pos + 0.5*(false_pos + false_neg))


@partial(jnp.vectorize, signature="(m),(m)->()")
def permutation_invariant_binary_f1(predictions, labels):
  f1_pos = binary_f1(predictions, labels)
  permuted_predictions = jnp.array([1, 0])[predictions]
  f1_neg = binary_f1(permuted_predictions, labels)
  return jnp.maximum(jnp.mean(f1_pos), jnp.mean(f1_neg))


@partial(custom_vjp, nondiff_argnums=(0,))
def fixed_point(f, a, x_init):
  """Computes the fixed point of f.

  Given a fixed point equation x* = f(a,x*), this function computes x*.

  Args:
    f: The function to compute a fixed point of.
    a: The 'auxiliary' information to feed f.
    x_init: The initial x to use when computing the fixed point.
  Returns:
    x_*: the fixed point of f.
  """
  def cond_fun(carry):
    x_prev, x = carry
    return jnp.logical_not(jnp.allclose(x_prev, x, rtol=1e-4, atol=1e-4))

  def body_fun(carry):
    _, x = carry
    return x, f(a, x)

  _, x_star = jax.lax.while_loop(cond_fun, body_fun, (x_init, f(a, x_init)))
  return x_star


def fixed_point_fwd(f, a, x_init):
  x_star = fixed_point(f, a, x_init)
  return x_star, (a, x_star)


def fixed_point_rev(f, res, x_star_bar):
  a, x_star = res
  _, vjp_a = jax.vjp(lambda a: f(a, x_star), a)
  a_bar, = vjp_a(fixed_point(partial(rev_iter, f),
                             (a, x_star, x_star_bar),
                             x_star_bar))
  return a_bar, jnp.zeros_like(x_star)


def rev_iter(f, packed, u):
  a, x_star, x_star_bar = packed
  _, vjp_x = jax.vjp(lambda x: f(a, x), x_star)
  return x_star_bar + vjp_x(u)[0]


fixed_point.defvjp(fixed_point_fwd, fixed_point_rev)


@partial(custom_vjp, nondiff_argnums=(0,))
def fixed_point2(f, a, x_init):
  """Computes the fixed point of f.

  Given a fixed point equation x* = f(a,x*), this function computes x*.

  Args:
    f: The function to compute a fixed point of.
    a: The 'auxiliary' information to feed f.
    x_init: The initial x to use when computing the fixed point.
  Returns:
    x_*: the fixed point of f.
  """
  def cond_fun(carry):
    x_prev, x = carry
    return jnp.logical_not(jnp.allclose(x_prev, x, rtol=1e-5, atol=1e-8))

  def body_fun(carry):
    _, x = carry
    return x, f(a, x)

  _, x_star = jax.lax.while_loop(cond_fun, body_fun, (x_init, f(a, x_init)))
  return x_star


def fixed_point2_fwd(f, a, x_init):
  """Custom vector-jacobian product forward function.

  Runs fixed_point forward and saves values useful for computing the
  vector-jacobian product in the backward pass.
  """
  x_star = fixed_point2(f, a, x_init)
  return x_star, (a, x_star)


def fixed_point2_rev(f, res, x_star_bar):
  """Custom VJP backward function.
  """
  a, x_star = res
  d_xstar = jax.jacrev(f, argnums=1)(a, x_star)
  d_a = jax.jacrev(f, argnums=0)(a, x_star)
  f_output_dim = x_star.shape[0]
  da_shapes = jax.tree_util.tree_map(lambda x: x.shape, d_a)
  reshaped_da = jax.tree_util.tree_map(
      lambda x: jnp.reshape(x, [f_output_dim, -1]), d_a)
  pre_inv = jnp.eye(x_star.shape[0]) - d_xstar
  jacs = jax.tree_util.tree_map(lambda x: jnp.linalg.solve(pre_inv, x),
                                reshaped_da)
  vjps = jax.tree_util.tree_map(lambda j: jnp.matmul(x_star_bar, j), jacs)
  reshaped_vjps = jax.tree_util.tree_multimap(
      lambda x, s: jnp.reshape(x, s[1:]), vjps, da_shapes)
  return reshaped_vjps, jnp.zeros_like(x_star)


fixed_point2.defvjp(fixed_point2_fwd, fixed_point2_rev)


def sinkhorn(C, log_w_p, log_w_q, key, alpha=0.01):
  """Uses sinkhorn iterations to solve an optimal transport problem.

  Computes the optimal cost and transport plan for moving mass from an
  atomic measure p to another atomic measure q.

  Args:
    C: The cost matrix, C_ij contains the cost of moving mass from the ith
      atom of p to the jth atom of q.
    log_w_p: The log weights of the atoms of p.
    log_w_q: The log weights of the atoms of q.
    key: JAX PRNGKey for intializing the dual variables.
    alpha: The stepsize of the Sinkhorn iteration.
  Returns:
    cost: The optimal cost
    log_pi: The log of the transport plan.
  """
  def sinkhorn_step(a, nu):
    log_w_p, log_w_q, C = a
    pre_lambda = jscipy.special.logsumexp(
        (-C - nu[Ellipsis, jnp.newaxis]) / alpha, axis=0)
    new_lambda = alpha * (pre_lambda - log_w_q - 1.)
    pre_nu = jscipy.special.logsumexp(
        (-C - new_lambda[jnp.newaxis, Ellipsis]) / alpha, axis=1)
    new_nu = alpha * (pre_nu - log_w_p - 1.)
    return new_nu  # - jnp.amax(new_nu)

  nu_0 = jax.random.normal(key, [log_w_q.shape[0]])

  nu_star = fixed_point(sinkhorn_step, (log_w_p, log_w_q, C), nu_0)

  pre_lambda = jscipy.special.logsumexp(
      (-C - nu_star[Ellipsis, jnp.newaxis]) / alpha, axis=0)
  lambda_star = alpha * (pre_lambda - log_w_q - 1.)

  log_pi = (-lambda_star[jnp.newaxis, Ellipsis] - nu_star[Ellipsis, jnp.newaxis] -
            C) / alpha - 1
  cost = jnp.sum(C * jnp.exp(log_pi))
  return cost, log_pi


def atomic_sinkhorn(p_locs, log_w_p, q_locs, log_w_q, key, alpha=0.01):
  """Solves an optimal transport problem between two atomic measures.

  p and q are assumed to be weighted atomic measures. The weights must
  sum to one.

  Args:
    p_locs: The locations of the atoms of p.
    log_w_p: The log weights of the atoms of p.
    q_locs: The locations of the atoms of q.
    log_w_q: The log weights of the atoms of q.
    key: A JAX PRNGKey for intializing the dual variables in the Sinkhorn
      iterations.
    alpha: The stepsize for the Sinkhorn iterations.
  Returns:
    cost: The optimal cost.
    log_pi: The log of the transport plan.
  """
  C = vmap(vmap(l2_dist, in_axes=(0, None)), in_axes=(None, 0))(q_locs, p_locs)
  return sinkhorn(C, log_w_p, log_w_q, key, alpha=alpha)


def shift_right(x):
  """Shift the input to the right by padding on axis -2."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[-2] = (1, 0)  # Padding on axis=-2
  padded = jnp.pad(
      x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
  return padded[Ellipsis, :-1, :]


def make_mask(lengths, max_length):
  batch_size = lengths.shape[0]
  mask = jnp.arange(max_length)
  mask = jnp.tile(mask[jnp.newaxis, :], [batch_size, 1])
  mask = mask < lengths[Ellipsis, jnp.newaxis]
  return mask
