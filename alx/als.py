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

"""Common JAX functions for ALS implementations."""

import dataclasses
import functools
from typing import Optional

import flax
import jax
import jax.numpy as jnp
import jax.profiler
import jax.scipy as jsp

from alx import topk as topk_utils

NINF = -1e19


# TODO(harshm): split into multiple configs.
@dataclasses.dataclass
class ALSConfig:
  """Top level config params for ALS behavior."""
  num_cols: int
  num_rows: int
  tie_rows_and_cols: bool
  embedding_dim: int
  reg: float
  unobserved_weight: float
  stddev: float
  is_bfloat16: bool
  eval_topk: int

  # Training config.
  seq_len: int
  batch_size: int
  transpose_batch_size: int
  eval_batch_size: int
  num_rows_per_batch: int
  transpose_num_rows_per_batch: int
  eval_num_rows_per_batch: int
  ground_truth_batch_size: int
  num_epochs: int
  train_files: str
  train_transpose_files: str
  test_files: str
  is_pre_batched: bool

  # Number of eval steps we run. Use the full eval set if set to -1.
  #
  # We typically don't evaluate on the full set since its slow for large models
  # bottlenecked by top_k op. Unfortunately, Top_k is implemented as a full sort
  # currently on TPUs. We may be able to optimize this further in the future.
  num_eval_iterations: int

  # If true, we use approximate (lossy but fast) topk for inference at eval
  # time.
  #
  # Exact TopK on TPUs can be very slow, we recommend using approximate version
  # for large vocabs. Since apporoximate TopK is lossy, the eval metrics are
  # a lower bound on the actual performance with high probability.
  approx_topk_for_eval: bool

  # Can take any value between [lu, cholesky, qr, cg], If None, we use 'cg'
  # since its fastest on TPU in our experience.
  linear_solver: Optional[str] = None

  # Controls how many TPU cores to use. Can take values between [0, 8].
  # Does not work in multi-process setups.
  local_device_count: Optional[int] = None

  # If True, each device gathers embeddings in a loop. Since number of
  # embeddings to be gathered scales linearly with number of TPUs, at some
  # point it becomes to big to fit in memory. Loop gather allows us to gather
  # embeddings without OOM at that stage.
  loop_gather: bool = True


def score(item_embedding_table, user_history, item_gramian, users_from_batch,
          id_list, device_item_table_size, batch_size, num_devices, cfg):
  """Returns topk highest scores for each user.

  Args:
    item_embedding_table: Sharded, stores embedding for every item.
    user_history: Sharded, batch of user history with static shape of
      [batch_size, seq_len]
    item_gramian: Sharded, gramian used for projecting.
    users_from_batch: Sharded, [num_users] shaped tensor, each row containing a
      user id. Padded with -1 in case user_history tensor fills up before this
      does. See batching_utils more details.
    id_list: Sharded, [batch_size] shaped tensor, contains accounting
      information about each row in user_history. Each row contains a number
      between [0, num_users). This is used to segment_sum LHS and RHS for solve.
    device_item_table_size: size of this device's item_embedding_table.
    batch_size: Number of rows in user_history. We pass batch_size explicitely
      to avoid inferring from tensors.
    num_devices: Number of devices this function is pmapped with.
    cfg: Instance of ALSConfig.

  Returns:
    Topk scores, topk ids. Both shaped [topk, num_users, num_devices]
  """
  axis_index = jax.lax.axis_index('i')
  print(f'user_history: {user_history.shape}')

  user_history = jnp.reshape(user_history, [-1])
  user_embeddings = solve(item_embedding_table, user_history, item_gramian,
                          users_from_batch, id_list, device_item_table_size,
                          cfg.reg, batch_size, num_devices, cfg)

  # This device computed the user embeddings for this shard, we concat user
  # embeddings computed by all the devices so that we can compute the scores.
  #
  # [num_users, embedding_dim] -> [num_devices, num_users, embedding_dim]
  user_embeddings = jax.lax.all_gather(user_embeddings, 'i')

  print('will exclude')
  # Exclude user history from scores.

  # [batch_size * seq_len] -> [num_devices, batch_size * seq_len]
  user_history = jax.lax.all_gather(user_history, 'i')

  # First compute the mask then shift the ids.
  user_history_mask = (user_history >= device_item_table_size * axis_index) & (
      user_history < device_item_table_size * (axis_index + 1))
  user_history -= device_item_table_size * axis_index
  ninf_scores = jnp.ones_like(user_history) * NINF * user_history_mask

  # [batch_size] -> [num_devices, batch_size]
  x_indices = jax.lax.all_gather(id_list, 'i')

  # [num_devices, batch_size] -> [num_devices, batch_size, seq_len]
  x_indices = jnp.tile(x_indices[:, :, jnp.newaxis], (1, 1, cfg.seq_len))
  x_indices = jnp.reshape(x_indices, [num_devices, batch_size * cfg.seq_len])
  y_indices = user_history

  def compute_topk(args):
    user_embeddings, x_indices, y_indices, ninf_scores = args

    # [num_users, device_item_table_size]
    scores = (user_embeddings @ item_embedding_table.T).astype(jnp.bfloat16)
    scores = scores.at[(x_indices, y_indices)].add(ninf_scores)

    print(f'scores: {scores.shape}')
    if cfg.approx_topk_for_eval:
      top_scores, top_ids = topk_utils.top_k_approx(scores, cfg.eval_topk)
    else:
      top_scores, top_ids = jax.lax.top_k(scores, cfg.eval_topk)
    return top_scores, top_ids

  # We intentionally use jax.lax.map instead of vmap here since the scores
  # tensor can get too big to fit in memory if we materialize for all devices
  # together. Using map instead of vmap allows us to do this computation in a
  # serial fashion.
  #
  # In practive, we haven't seen much difference in performance when we use vmap
  # instead of map, for cases where we are able to materialize the full scores
  # tensor.
  top_scores, top_ids = jax.lax.map(
      compute_topk, [user_embeddings, x_indices, y_indices, ninf_scores])

  # [eval_topk, num_users, num_devices]
  return top_scores.T, (top_ids.T + axis_index * device_item_table_size)


def eval_step(item_embedding_table, user_history, item_gramian,
              users_from_batch, id_list, batched_ground_truths,
              ground_truth_batch_ids, device_item_table_size, batch_size,
              num_devices, cfg):
  """Returns topk highest scores for each user.

  Args:
    item_embedding_table: Sharded, stores embedding for every item.
    user_history: Sharded, batch of user history with static shape of
      [batch_size, seq_len]
    item_gramian: Sharded, gramian used for projecting.
    users_from_batch: Sharded, [num_users] shaped tensor, each row containing a
      user id. Padded with -1 in case user_history tensor fills up before this
      does. See batching_utils more details.
    id_list: Sharded, [batch_size] shaped tensor, contains accounting
      information about each row in user_history. Each row contains a number
      between [0, num_users). This is used to segment_sum LHS and RHS for solve.
    batched_ground_truths: Sharded, [num_users, max_ground_truth] shaped tensor.
      We compare the final topk ids with this set in order to calculate recall
      metrics.
    ground_truth_batch_ids: Sharded, mapping between dense and sparse ground
      truth.
    device_item_table_size: size of this device's item_embedding_table.
    batch_size: Number of rows in user_history. We pass batch_size explicitely
      to avoid inferring from tensors.
    num_devices: Number of devices this function is pmapped with.
    cfg: Instance of ALSConfig.

  Returns:
    R@20 sum, R@50 sum, num_valid_users. All scalars. We return raw sums so
    that host can aggregate over all the full eval set.
  """
  num_users = users_from_batch.shape[0]
  axis_index = jax.lax.axis_index('i')
  # [topk, num_users, num_devices]
  topk_scores, topk_ids = score(item_embedding_table, user_history,
                                item_gramian, users_from_batch, id_list,
                                device_item_table_size, batch_size, num_devices,
                                cfg)

  # [eval_topk, num_users, num_devices] ->
  # [num_devices, eval_topk, num_users, num_devices]
  all_device_topk_scores = jax.lax.all_gather(topk_scores, 'i')
  all_device_topk_ids = jax.lax.all_gather(topk_ids, 'i')

  # [num_devices, eval_topk, num_users, num_devices] ->
  # [num_devices, num_users, eval_topk, num_devices]
  all_device_topk_scores = all_device_topk_scores.T
  all_device_topk_ids = all_device_topk_ids.T

  # [num_devices, num_users, eval_topk, num_devices] ->
  # [num_users, eval_topk, num_devices]
  all_device_topk_scores = jnp.squeeze(
      jax.lax.dynamic_slice_in_dim(
          all_device_topk_scores, start_index=axis_index, slice_size=1, axis=0),
      axis=0)
  all_device_topk_ids = jnp.squeeze(
      jax.lax.dynamic_slice_in_dim(
          all_device_topk_ids, start_index=axis_index, slice_size=1, axis=0),
      axis=0)

  # [num_users, eval_topk, num_devices] -> [num_users, eval_topk * num_devices]
  all_device_topk_scores = jnp.reshape(all_device_topk_scores,
                                       [num_users, cfg.eval_topk * num_devices])
  all_device_topk_ids = jnp.reshape(all_device_topk_ids,
                                    [num_users, cfg.eval_topk * num_devices])

  # [num_users, eval_topk * num_devices] -> [num_users, eval_topk]
  topk_ids_and_scores_fn = jax.vmap(
      functools.partial(topk_ids_and_scores, topk=cfg.eval_topk))
  _, top_ids = topk_ids_and_scores_fn(all_device_topk_ids,
                                      all_device_topk_scores)

  recall_at20_fn = jax.vmap(
      functools.partial(
          batched_recall,
          batched_ground_truths=batched_ground_truths,
          ground_truth_batch_ids=ground_truth_batch_ids,
          r_at=20),
      axis_name='j')
  recall_at50_fn = jax.vmap(
      functools.partial(
          batched_recall,
          batched_ground_truths=batched_ground_truths,
          ground_truth_batch_ids=ground_truth_batch_ids,
          r_at=50),
      axis_name='j')

  # []
  recall_at20_sum = jnp.sum(recall_at20_fn(top_ids))
  recall_at50_sum = jnp.sum(recall_at50_fn(top_ids))
  num_valid_examples = jnp.sum(users_from_batch > -1)

  # []
  all_device_recall_at20_sum = jax.lax.psum(recall_at20_sum, 'i')
  all_device_recall_at50_sum = jax.lax.psum(recall_at50_sum, 'i')
  all_device_num_valid_examples = jax.lax.psum(num_valid_examples, 'i')

  # []
  return (all_device_recall_at20_sum, all_device_recall_at50_sum,
          all_device_num_valid_examples)


def compute_gramain(embedding_table):
  item_gramian = jnp.matmul(embedding_table.T, embedding_table)
  return jax.lax.psum(item_gramian, 'i')


def loop_gather(item_embedding_table, user_history, user_history_mask,
                batch_size, seq_len, embedding_dim, num_devices):
  """Gather embeddings in a loop and return only for the current device."""
  axis_index = jax.lax.axis_index('i')

  def device_gather(carry, single_device_input):
    """Gathers embeddings for every device, unrolled for num_devices."""
    device_user_history, device_user_history_mask = single_device_input
    i, this_device_item_embed = carry

    device_item_emb = item_embedding_table[device_user_history, :]

    # Zero out embeddings.
    device_item_emb *= device_user_history_mask
    # Distributed sum.
    device_item_emb = jax.lax.psum(device_item_emb, 'i')

    # If the unroll index i is same as axis_index, which means we are gathering
    # this device's embeddings, we assign to carry so that scan can return it
    # at the end.
    this_device_item_embed = jax.lax.cond(
        i == axis_index,
        lambda _: device_item_emb,
        lambda _: this_device_item_embed,
        operand=None)

    return (i+1, this_device_item_embed), None

  # Initialize this device's item_emb with zeros, we will fill it out as part
  # of scan carry.
  item_emb = jnp.zeros(
      shape=(batch_size * seq_len, embedding_dim),
      dtype=item_embedding_table.dtype)

  # Since we all_gathered user_history, the size of that tensor increases with
  # number of devices we use. For very large topologies this becomes big enough
  # to go out of memory.
  #
  # Since we only want embedddings for this device's user_history eventually
  # we can use scan and carry that in the result instead of materializing the
  # embeddings for all the devices at once. This also means that we may incur
  # a performance penalty since the computation/data-communication is now
  # serial. In practice, we have not seen much regression from this though.
  (_, item_emb), _ = jax.lax.scan(
      f=device_gather,
      init=(0, item_emb),
      xs=(user_history, user_history_mask),
      length=num_devices)

  item_emb = jnp.reshape(item_emb, [batch_size, seq_len, embedding_dim])
  return item_emb


def direct_gather(item_embedding_table, user_history, user_history_mask,
                  batch_size, seq_len, embedding_dim, num_devices):
  """Gather embeddings for all devices but return only for current device."""
  axis_index = jax.lax.axis_index('i')

  item_emb = item_embedding_table[user_history, :]

  # Zero out embeddings.
  item_emb *= user_history_mask
  # Distributed sum.
  item_emb = jax.lax.psum(item_emb, 'i')
  print(f'item_emb pre-reshape: {item_emb.shape}')
  item_emb = jnp.reshape(item_emb,
                         [num_devices, batch_size, seq_len, embedding_dim])

  # Now that all the devices have embeddings for their shard of user histories,
  # we slice into the embeddings tensor to obtain the shard that this device
  # was supposed to solve for.
  #
  # [num_devices, batch_size, seq_len, embedding_dim] ->
  # [batch_size, seq_len, embedding_dim]
  item_emb = jnp.squeeze(
      jax.lax.dynamic_slice_in_dim(
          item_emb, start_index=axis_index, slice_size=1, axis=0),
      axis=0)
  return item_emb


def gather_embeddings(item_embedding_table, user_history,
                      device_item_table_size, batch_size, seq_len,
                      embedding_dim, num_devices, cfg):
  """Given embedding table and user history, peform sharded gather."""
  axis_index = jax.lax.axis_index('i')

  # We want to gather items only from the local device shard of
  # item_embedding_table. But the gather operation doesn't complain if we go
  # out of bound, so we gather all the embeddings in user history even though we
  # know that it wouldn't find some of the entries on this device's shard. We
  # will zero out the invalid embeddings later.

  # User history is sharded across devices, so we first concatenate user history
  # from all devices in order to obtain the embeddings.
  #
  # [batch_size * seq_len] -> [num_devices, batch_size * seq_len]
  user_history = jax.lax.all_gather(user_history, 'i')

  # First compute the mask then shift the ids.
  user_history_mask = (user_history >= device_item_table_size * axis_index) & (
      user_history < device_item_table_size * (axis_index + 1))
  user_history_mask = jnp.expand_dims(user_history_mask, -1)

  # Offset
  user_history = user_history - (device_item_table_size * axis_index)

  if cfg.loop_gather:
    item_emb = loop_gather(item_embedding_table, user_history,
                           user_history_mask, batch_size, seq_len,
                           embedding_dim, num_devices)
  else:
    item_emb = direct_gather(item_embedding_table, user_history,
                             user_history_mask, batch_size, seq_len,
                             embedding_dim, num_devices)
  return item_emb


def solve(item_embedding_table, user_history, item_gramian, users_from_batch,
          id_list, device_item_table_size, reg, batch_size, num_devices,
          cfg):
  """Gather item embeddings and solver for a batch of users."""
  embedding_dim = item_embedding_table.shape[1]
  num_users = users_from_batch.shape[0]

  item_emb = gather_embeddings(item_embedding_table, user_history,
                               device_item_table_size, batch_size, cfg.seq_len,
                               embedding_dim, num_devices, cfg)

  # We use lax.convert_element_type directly intentionally instead of
  # tensor.astype, since XLA sometimes fuses it in a worse way such that >50%
  # of TPU time goes in convert_element_type ops.
  item_emb = jax.lax.convert_element_type(item_emb, jnp.float32)

  # Local compute.
  lambda_batch = jnp.einsum('bij,bik->bjk', item_emb, item_emb)
  mu_batch = jnp.einsum('bij->bj', item_emb)

  # Local segment sum over sharded batch.
  lambda_batch_summed = jax.ops.segment_sum(
      lambda_batch, jnp.asarray(id_list), num_segments=num_users)
  mu_batch_summed = jax.ops.segment_sum(
      mu_batch, jnp.asarray(id_list), num_segments=num_users)

  assert lambda_batch_summed.shape == (num_users, embedding_dim, embedding_dim)
  reg = jnp.broadcast_to(reg, [num_users])
  reg = reg[Ellipsis, jnp.newaxis, jnp.newaxis]
  print(f'reg: {reg.shape}')

  lambda_batch_summed += cfg.unobserved_weight * item_gramian  # G_i += G
  print(f'lambda_batch_summed: {lambda_batch_summed.shape}')
  process_reg = jnp.expand_dims(jnp.identity(embedding_dim), 0) * reg
  print(f'process_reg: {process_reg.shape}')
  post_lambda = lambda_batch_summed + jnp.expand_dims(
      jnp.identity(embedding_dim), 0) * reg  # G_i += reg * I

  @jax.vmap
  def cg_solve(a, b):
    x, _ = jax.scipy.sparse.linalg.cg(a, b)
    return x

  @jax.vmap
  def cholesky_solve(a, b):
    factors = jsp.linalg.cho_factor(a, overwrite_a=True)
    return jsp.linalg.cho_solve(factors, b, overwrite_b=True)

  @jax.vmap
  def qr_solve(a, b):
    q, r = jax.lax.linalg.qr(a)
    return jax.lax.linalg.triangular_solve(
        r, q.T @ b, left_side=True, lower=False)

  solve_fn = None
  if cfg.linear_solver == 'lu':
    solve_fn = jnp.linalg.solve
  elif cfg.linear_solver == 'qr':
    solve_fn = qr_solve
  elif cfg.linear_solver == 'cholesky':
    solve_fn = cholesky_solve
  else:
    solve_fn = cg_solve

  solve_fn = jax.named_call(solve_fn, name='linear_solver')

  user_embeddings = solve_fn(post_lambda, mu_batch_summed)  # U_i = G_i^{-1} b_i
  return user_embeddings


# TODO(harshm): rename this to solve_and_update.
def project(item_embedding_table, user_embedding_table, user_history,
            item_gramian, item_lengths_from_batch, users_from_batch, id_list,
            device_item_table_size, num_items, device_user_table_size,
            batch_size, num_devices, cfg):
  """Gather item embeddings, solver for batch of users and update the table."""
  axis_index = jax.lax.axis_index('i')
  num_users = users_from_batch.shape[0]
  embedding_dim = item_embedding_table.shape[1]

  reg = cfg.reg * (item_lengths_from_batch + cfg.unobserved_weight * num_items)
  user_embeddings = solve(item_embedding_table, user_history, item_gramian,
                          users_from_batch, id_list, device_item_table_size,
                          reg, batch_size, num_devices, cfg)

  # This device computed the user embeddings for this shard, we concat user
  # embeddings computed by all the devices so that we can update this device's
  # user_embedding_table.
  #
  # [batch_size, embedding_dim] -> [num_devices, batch_size, embedding_dim]

  # We use lax.convert_element_type directly intentionally instead of
  # tensor.astype, since XLA sometimes fuses it in a worse way such that >50%
  # of TPU time goes in convert_element_type ops.
  user_embeddings = jax.lax.convert_element_type(user_embeddings, jnp.bfloat16)
  user_embeddings = jax.lax.all_gather(user_embeddings, 'i')

  users_from_batch = jax.lax.all_gather(users_from_batch, 'i')
  user_embeddings = jnp.reshape(user_embeddings,
                                [num_users * num_devices, embedding_dim])
  users_from_batch = jnp.reshape(users_from_batch, [num_users * num_devices])
  print(f'user_embeddings: {user_embeddings.shape}')
  print(f'users_from_batch: {users_from_batch.shape}')

  # All the devices will have access to embeddings for all the users in the
  # batch. But, we will select the users for this shard for final update.
  #
  # First compute the mask then shift the ids.
  users_mask = (users_from_batch >= device_user_table_size * axis_index) & (
      users_from_batch < device_user_table_size * (axis_index + 1))
  users_mask = jnp.expand_dims(users_mask, -1)
  users_from_batch -= device_user_table_size * axis_index
  print(f'users_mask: {users_mask.shape}')

  # Compute the residual and index add.
  original_user_embedding = user_embedding_table[users_from_batch]
  original_user_embedding *= users_mask
  user_embeddings *= users_mask
  user_embeddings_residual = user_embeddings - original_user_embedding
  print(f'user_embedding_table: {user_embedding_table.shape}')
  print(f'user_embeddings: {user_embeddings.shape}')
  return user_embedding_table.at[users_from_batch, :].add(
      user_embeddings_residual)


# XLA requires 2-3x of memory while creating the embedding table. For large
# embedding table sizes, this can go out of memory. Thus, we intentionally
# create multiple smaller embedding tables and then concatenate all of them.
#
# On a single core of TPU v3, naively creating the embedding table only scales
# to ~10M embeddings of size 128. This strategy enables creation of embedding
# tables of size 60M.
def create_embedding_table(rng, num_x, cfg):
  """Creates sharded embedding table."""
  batch_size = 1000000
  table_list = []
  for start in range(0, num_x, batch_size):
    end = min(start + batch_size, num_x)
    num = end - start
    table = cfg.stddev * jax.random.normal(
        rng, [num, cfg.embedding_dim], dtype=jnp.float32)

    # We do the conversion as a seperate step due to a bug in XLA.
    if cfg.is_bfloat16:
      table = table.astype(jnp.bfloat16)
    table_list.append(table)
  return jnp.concatenate(table_list)


def topk_ids_and_scores(ids, scores, topk=100):
  permutation = jnp.argsort(scores)[::-1]
  sorted_ids = ids[permutation]
  sorted_scores = scores[permutation]
  return sorted_scores[:topk], sorted_ids[:topk]


def batched_recall(top_ids,
                   batched_ground_truths,
                   ground_truth_batch_ids,
                   r_at=20):
  """Calculates recall for a batch of predictions."""
  batch_id = jax.lax.axis_index('j')

  # Invalidate the ground truth entries for other examples in the batch.
  batched_ground_truths = jnp.where(
      (ground_truth_batch_ids == batch_id)[:, jnp.newaxis],
      batched_ground_truths, -1)
  return recall(top_ids, batched_ground_truths, r_at)


def recall(top_ids, ground_truth, r_at=20):
  num_valid_ground_truth = jnp.sum(ground_truth > -1)
  divide_by = jnp.minimum(r_at, num_valid_ground_truth)
  sum_recall = jnp.sum(jnp.isin(top_ids[:r_at], ground_truth))
  return jnp.where(divide_by == 0, 0.0, sum_recall / divide_by)


@flax.struct.dataclass
class ALSState:
  step: int
  col_embedding: jax.pxla.ShardedDeviceArray
  row_embedding: jax.pxla.ShardedDeviceArray


def get_local_devices(cfg):
  """Returns a list of devices to be used by ALX.

  In most cases we use all 8 available devices connected to a host but one can
  use less by setting ALSConfig.local_device_count parameter.

  Args:
    cfg: plumbed ALSConfig for local_device_count param.
  """
  local_devices = jax.local_devices(jax.process_index())
  if cfg.local_device_count:
    local_devices = local_devices[:cfg.local_device_count]
  return local_devices


def get_devices(cfg):
  devices = jax.devices()
  if cfg.local_device_count:
    devices = devices[:cfg.local_device_count]
  return devices


def device_embedding_table_size(num_items, device_count):
  # Its possible that num_items is not divisible by device_count, so we over
  # allocate embeddings in each device by 1.
  return (num_items // device_count) + 1


class ALS():
  """Top level ALS class."""

  def __init__(self, cfg, als_state = None):
    self.cfg = cfg

    if cfg.local_device_count:
      if jax.process_count() > 1:
        raise ValueError(
            'local_device_count is not available for multi-process setups.')

    self.local_devices = get_local_devices(cfg)
    self.devices = get_devices(cfg)
    self.local_device_count = len(self.local_devices)
    self.device_count = len(self.devices)

    # Create pmapped functions.
    self.pmapped_create_embedding_table_fn = jax.pmap(
        functools.partial(create_embedding_table, cfg=cfg),
        static_broadcasted_argnums=(1),
        devices=self.devices)
    self.pmapped_user_project_fn = jax.pmap(
        functools.partial(
            project,
            batch_size=cfg.batch_size,
            num_devices=self.device_count,
            cfg=cfg),
        axis_name='i',
        devices=self.devices)
    self.pmapped_item_project_fn = jax.pmap(
        functools.partial(
            project,
            batch_size=cfg.transpose_batch_size,
            num_devices=self.device_count,
            cfg=cfg),
        axis_name='i',
        devices=self.devices)
    self.pmapped_compute_gramian_fn = jax.pmap(
        compute_gramain, axis_name='i', devices=self.devices)

    # Prepare for embedding table creation.
    self.rng = jax.random.PRNGKey(jax.process_index())
    self.device_user_table_size = device_embedding_table_size(
        cfg.num_rows, self.device_count)
    self.device_item_table_size = device_embedding_table_size(
        cfg.num_cols, self.device_count)

    if not als_state:
      self.user_rng, self.item_rng = jax.random.split(self.rng, num=2)
      self.user_rng_array = jax.random.split(
          self.user_rng, num=self.local_device_count)
      self.item_rng_array = jax.random.split(
          self.item_rng, num=self.local_device_count)
      # Create embedding tables.
      user_embedding = self.pmapped_create_embedding_table_fn(
          self.user_rng_array, self.device_user_table_size)
      print(f'user_embedding.shape: {user_embedding.shape}')
      item_embedding = None
      if cfg.tie_rows_and_cols:
        item_embedding = user_embedding
        if cfg.num_rows != cfg.num_cols:
          raise ValueError(
              'Num rows should be equal to num cols for weight tieing.')
      else:
        item_embedding = self.pmapped_create_embedding_table_fn(
            self.item_rng_array, self.device_item_table_size)
      print(f'item_embedding.shape: {item_embedding.shape}')
      als_state = ALSState(
          step=0, col_embedding=item_embedding, row_embedding=user_embedding)

    self.als_state = als_state
    self.pmapped_eval_step_fn = jax.pmap(
        functools.partial(
            eval_step,
            device_item_table_size=self.device_item_table_size,
            num_devices=self.device_count,
            batch_size=cfg.eval_batch_size,
            cfg=cfg),
        axis_name='i',
        devices=self.devices)

    # Update gramians.
    self.update_user_gramian()
    self.update_item_gramian()

    # Cached embeddings for eval.
    self.materialized_item_embeddding = None
    self.materialized_item_gramian = None

  def update_user_gramian(self):
    self.user_gramian = self.pmapped_compute_gramian_fn(
        self.als_state.row_embedding)

  def update_item_gramian(self):
    self.item_gramian = self.pmapped_compute_gramian_fn(
        self.als_state.col_embedding)

  def eval(self, test_batches):
    """Given test set batches, generates predictions and calculates metrics."""
    recall_at20_sum = 0
    recall_at50_sum = 0
    num_valid_examples = 0

    for i, batch in enumerate(test_batches):
      # Break once we reach max eval iterations.
      if (self.cfg.num_eval_iterations > -1 and
          i >= self.cfg.num_eval_iterations):
        break

      item_list_2d_array = batch['batched_history']
      id_list = batch['batch_ids']
      users_from_batch = batch['row_ids']
      batched_ground_truths = batch['batched_ground_truths']
      ground_truth_batch_ids = batch['ground_truth_batch_ids']
      (all_device_recall_at20, all_device_recall_at50,
       all_device_num_valid_examples) = self.pmapped_eval_step_fn(
           self.als_state.col_embedding, item_list_2d_array, self.item_gramian,
           users_from_batch, id_list, batched_ground_truths,
           ground_truth_batch_ids)
      recall_at20_sum += all_device_recall_at20[0]
      recall_at50_sum += all_device_recall_at50[0]
      num_valid_examples += all_device_num_valid_examples[0]

      recall_at_20 = recall_at20_sum / num_valid_examples
      recall_at_50 = recall_at50_sum / num_valid_examples

    return recall_at_20, recall_at_50, num_valid_examples

  def train(self, user_batches, item_batches):
    """Trains for a single epoch and updates gramians."""
    current_step = self.als_state.step
    # Optimize the user embeddings
    for batch in user_batches:
      self.solve(batch, is_user=True)
    self.update_user_gramian()
    # Optimize the item embeddings
    for batch in item_batches:
      self.solve(batch, is_user=False)
    self.update_item_gramian()
    self.als_state = self.als_state.replace(step=current_step + 1)

  def solve(self, batch, is_user):
    """Given an item or a user batch, trains one of the embedding tables."""
    if is_user:
      user_embedding = self.als_state.row_embedding
      item_embedding = self.als_state.col_embedding
      device_user_table_size = self.device_user_table_size
      device_item_table_size = self.device_item_table_size
      gramian = self.item_gramian
      num_items = self.cfg.num_cols
      pmapped_project_fn = self.pmapped_user_project_fn
    else:
      user_embedding = self.als_state.col_embedding
      item_embedding = self.als_state.row_embedding
      device_user_table_size = self.device_item_table_size
      device_item_table_size = self.device_user_table_size
      gramian = self.user_gramian
      num_items = self.cfg.num_rows
      pmapped_project_fn = self.pmapped_item_project_fn

    item_list_2d_array = batch['batched_history']
    id_list = batch['batch_ids']
    users_from_batch = batch['row_ids']
    item_lengths_from_batch = batch['item_lengths']

    device_item_table_size = jnp.tile(device_item_table_size,
                                      [self.local_device_count])
    device_user_table_size = jnp.tile(device_user_table_size,
                                      [self.local_device_count])
    num_items = jnp.tile(num_items, [self.local_device_count])

    user_table = pmapped_project_fn(item_embedding, user_embedding,
                                    item_list_2d_array, gramian,
                                    item_lengths_from_batch, users_from_batch,
                                    id_list, device_item_table_size, num_items,
                                    device_user_table_size)
    if is_user:
      assert self.als_state.row_embedding.shape == user_table.shape
      self.als_state = self.als_state.replace(row_embedding=user_table)
    else:
      assert self.als_state.col_embedding.shape == user_table.shape
      self.als_state = self.als_state.replace(col_embedding=user_table)
