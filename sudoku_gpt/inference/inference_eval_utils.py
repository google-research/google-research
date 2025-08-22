# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""This file contains functions for evaluation of the trained model."""

import io

from flax.training import common_utils
import jax
from jax import numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import tensorflow as tf

from sudoku_gpt import model
from sudoku_gpt import othello
from sudoku_gpt import sudoku


def valid_solution(output_seq):
  """Checks if the output sequence is valid."""
  ## returns 1 if correct solution, otherwise returns 0
  rows = np.zeros((9, 9))
  cols = np.zeros((9, 9))
  boxes = np.zeros((9, 9))

  for j in range(81):
    if int(output_seq[3 * j + 2] - 1) > 8:
      return False
    if int(output_seq[3 * j] - 1) > 8:
      return False
    if int(output_seq[3 * j + 1] - 1) > 8:
      return False
    row_num = int(output_seq[3 * j] - 1)
    col_num = int(output_seq[3 * j + 1] - 1)
    rows[row_num, int(output_seq[3 * j + 2] - 1)] += 1
    cols[col_num, int(output_seq[3 * j + 2] - 1)] += 1
    boxes[
        int(3 * (row_num // 3) + (col_num // 3)), int(output_seq[3 * j + 2] - 1)
    ] += 1

  if np.all(rows) and np.all(cols) and np.all(boxes):
    return True
  else:
    return False


def eval_step(state, batch, config):
  pred_logits = model.TransformerLMHeadModel(config).apply(
      {"params": state.params}, batch
  )
  return pred_logits


def get_othello_eval_metrics(
    state, eval_data_iter, p_eval_step, config
    ):
  """Get eval metrics for Othello dataset."""
  eval_metrics = {"acc": []}
  for eval_epoch in range(config.eval_epochs):
    with jax.profiler.StepTraceAnnotation("eval", step_num=eval_epoch):

      batch = np.array(next(eval_data_iter))
      total_pred, sucess_pred = 0, 0

      for i in range(config.seq_len):
        padding = np.zeros(
            (batch.shape[0], config.seq_len - (i + 1)), dtype=np.int32
        )
        concat_batch = np.hstack((batch[:, : (i + 1)], padding))
        concat_batch = common_utils.shard(
            jax.tree_util.tree_map(np.asarray, concat_batch)
        )
        pred_logits = p_eval_step(state, concat_batch)

        max_action = pred_logits[:, :, i, :].argmax(axis=-1)
        pred_seq = np.hstack(
            (batch[:, : (i + 1)], jnp.reshape(max_action, shape=(-1, 1)))
        )

        for j in range(pred_seq.shape[0]):
          ## When length of the game is small, then the model can simply keep
          ## predicting the next token which will increase the accuracy
          total_pred += 1
          try:
            othello.OthelloBoardState().update(pred_seq[j], prt=False)
          except AssertionError:
            ### Wrong prediction
            pass
          else:
            sucess_pred += 1

      eval_metrics["acc"].append(sucess_pred * 1.0 / total_pred)

  return eval_metrics


def get_edit_distance(config, generated_input_seq, original_input_seq):
  """Get the edit distance between model's output and solver's output."""
  total_distance = 0

  for i in range(config.start_index, config.block_size):
    # Iterate through model's output
    flg = False
    for j in range(config.start_index, config.block_size):

      # Iterate through solver's output to find the location of model's output
      same_row = generated_input_seq[3 * i] == original_input_seq[3 * j]
      same_col = (
          generated_input_seq[3 * i + 1] == original_input_seq[3 * j + 1]
      )
      if same_row and same_col:

        # When model's output cell location and solver's output cell location
        # matches, then calculate edit distance.
        total_distance += abs(j - i)
        flg = True
        break

    if not flg:
      total_distance += abs(config.block_size - i)

  return total_distance


def get_set_accuracy_for_pairs(
    pairs,
    state,
    p_eval_step,
    input_seq,
    possible_vals,
    given_vals,
    config,
    ):
  """Get the accuracy of the set of possible values for different cell positions."""
  correct_cnt = np.zeros(9)
  total_cnt = np.zeros(9)

  for i in range(len(pairs)):
    cur_input_seq = np.hstack(
        (input_seq[:, : (config.start_index * 3)], pairs[i] + 1)
    )

    padding = np.zeros(
        (input_seq.shape[0], config.seq_len - len(cur_input_seq[0])),
        dtype=np.int32,
    )
    concat_batch = np.hstack((cur_input_seq, padding))

    concat_batch = common_utils.shard(
        jax.tree_util.tree_map(np.asarray, concat_batch)
    )

    pred_logits = p_eval_step(state, concat_batch)

    cur_pred_logits = pred_logits[:, :, 3 * config.start_index + 1, :].reshape(
        (-1, pred_logits.shape[-1])
    )

    for k in range(input_seq.shape[0]):
      if given_vals[k, pairs[i, k, 0], pairs[i, k, 1]] == 1:
        continue
      total_possible_vals = np.int32(
          np.sum(possible_vals[k, pairs[i, k, 0], pairs[i, k, 1], :])
      )
      ordering_ind = np.argsort(cur_pred_logits[np.int32(k), :])[::-1][
          :total_possible_vals
      ]

      assert len(ordering_ind) <= 9

      for t, ind in enumerate(ordering_ind):
        if ind <= 9 and ind >= 1:
          correct_cnt[t] += (
              possible_vals[k, pairs[i, k, 0], pairs[i, k, 1], ind - 1] == 1
          )

        total_cnt[t] += 1

  accuracy = np.ones(9)
  for i in range(9):
    if total_cnt[i] > 0:
      accuracy[i] = correct_cnt[i] / total_cnt[i]

  return accuracy, correct_cnt, total_cnt


def get_sampled_pairs(input_seq, pred_logits, state, p_eval_step, config, key):
  """Get sampled pairs in a sequence."""
  pairs_set = []
  for _ in range(input_seq.shape[0]):
    pairs_set.append(set())

  pairs = np.zeros(
      (config.set_accuracy_top_k, input_seq.shape[0], 2), dtype=np.int32
  )
  flag = True  # Denotes if we want to sample next time or not.

  while flag:
    pred_logits_row = pred_logits[:, :, 3 * config.start_index - 1, :].reshape(
        (-1, pred_logits.shape[-1])
    )
    rkey, key = jax.random.split(key, 2)

    pair_row = jax.random.categorical(rkey, pred_logits_row)

    assert len(pair_row) == input_seq.shape[0] and pair_row.ndim == 1

    cur_input_seq = np.hstack(
        (input_seq[:, : (config.start_index * 3)], pair_row.reshape(-1, 1))
    )

    padding = np.zeros(
        (input_seq.shape[0], config.seq_len - len(cur_input_seq[0])),
        dtype=np.int32,
    )

    concat_batch = np.hstack((cur_input_seq, padding))

    concat_batch = common_utils.shard(
        jax.tree_util.tree_map(np.asarray, concat_batch)
    )

    pred_logits = p_eval_step(state, concat_batch)
    pred_logits_col = pred_logits[:, :, 3 * config.start_index, :].reshape(
        (-1, pred_logits.shape[-1])
    )

    rkey, key = jax.random.split(key, 2)
    pair_col = jax.random.categorical(rkey, pred_logits_col)

    assert len(pair_col) == input_seq.shape[0] and pair_col.ndim == 1

    flag = False
    for i in range(input_seq.shape[0]):
      if pair_row[i] < 1 or pair_row[i] > 9:
        continue

      if pair_col[i] < 1 or pair_col[i] > 9:
        continue
      pairs_set[i].add(tuple((int(pair_row[i]), int(pair_col[i]))))
      if len(pairs_set[i]) < config.set_accuracy_top_k:
        flag = True

  for i in range(input_seq.shape[0]):
    j = 0
    for a_pair in pairs_set[i]:
      pairs[j, i, 0] = int(a_pair[0] - 1)
      pairs[j, i, 1] = int(a_pair[1] - 1)
      j += 1

      if j == config.set_accuracy_top_k:
        break

  return pairs


def get_topk_probability_pairs(
    input_seq, pred_logits, state, p_eval_step, config
    ):
  """Get topk probability pairs in a sequence."""
  pairs = np.zeros(
      (config.set_accuracy_top_k, input_seq.shape[0], 2), dtype=np.int32
  )
  pred_logits_row = pred_logits[:, :, 3 * config.start_index - 1, :].reshape(
      (-1, pred_logits.shape[-1])
  )
  row_log_prob = jax.nn.log_softmax(pred_logits_row[:, 1:10])

  pairs_log_prob = np.zeros((input_seq.shape[0], 81))

  for i in range(9):
    row_num = np.ones((input_seq.shape[0], 1), dtype=np.int32) * (i + 1)
    cur_input_seq = np.hstack(
        (input_seq[:, : (config.start_index * 3)], row_num)
    )

    padding = np.zeros(
        (input_seq.shape[0], config.seq_len - len(cur_input_seq[0])),
        dtype=np.int32,
    )

    concat_batch = np.hstack((cur_input_seq, padding))

    concat_batch = common_utils.shard(
        jax.tree_util.tree_map(np.asarray, concat_batch)
    )

    pred_logits_col = p_eval_step(state, concat_batch)
    pred_logits_col = pred_logits_col[:, :, 3 * config.start_index, :].reshape(
        (-1, pred_logits.shape[-1])
    )

    col_log_prob = jax.nn.log_softmax(pred_logits_col[:, 1:10])

    for j in range(input_seq.shape[0]):
      for k in range(9):
        pairs_log_prob[j, i * 9 + k] = col_log_prob[j, k] + row_log_prob[j, i]

  for i in range(input_seq.shape[0]):
    topk_indices = np.argsort(pairs_log_prob[i, :])[::-1][
        : config.set_accuracy_top_k
    ]
    for j, ind in enumerate(topk_indices):
      pairs[j, i, 0] = ind // 9
      pairs[j, i, 1] = ind % 9

  return pairs


def get_set_accuracies(state, p_eval_step, input_seq, config):
  """Get set accuracies in a sequence."""
  possible_vals = np.ones((input_seq.shape[0], 9, 9, 9))
  given_vals = np.zeros((input_seq.shape[0], 9, 9))

  for i in range(input_seq.shape[0]):
    for j in range(config.start_index):
      row_num = input_seq[i, 3 * j] - 1
      col_num = input_seq[i, 3 * j + 1] - 1
      val = input_seq[i, 3 * j + 2] - 1

      possible_vals[i, row_num, :, val] = 0
      possible_vals[i, :, col_num, val] = 0

      given_vals[i, row_num, col_num] = 1

  if config.set_accuracy == "top-k":
    cur_input_seq = input_seq[:, : (config.start_index * 3)]
    padding = np.zeros(
        (input_seq.shape[0], config.seq_len - len(cur_input_seq[0])),
        dtype=np.int32,
    )

    concat_batch = np.hstack((cur_input_seq, padding))

    concat_batch = common_utils.shard(
        jax.tree_util.tree_map(np.asarray, concat_batch)
    )

    _ = jax.random.PRNGKey(98)
    pred_logits = p_eval_step(state, concat_batch)

    print("get_topk_probability_pairs", flush=True)
    pairs = get_topk_probability_pairs(
        input_seq, pred_logits, state, p_eval_step, config
    )
    print("got_topk_probability_pairs", flush=True)
    return get_set_accuracy_for_pairs(
        pairs,
        state,
        p_eval_step,
        input_seq,
        possible_vals,
        given_vals,
        config,
    )

  elif config.set_accuracy == "all":
    pairs = np.zeros((81, input_seq.shape[0], 2), dtype=np.int32)
    for i in range(81):
      pairs[i, :, 0] = np.ones(input_seq.shape[0], dtype=np.int32) * (i // 9)
      pairs[i, :, 1] = np.ones(input_seq.shape[0], dtype=np.int32) * (i % 9)

    return get_set_accuracy_for_pairs(
        pairs,
        state,
        p_eval_step,
        input_seq,
        possible_vals,
        given_vals,
        config,
    )


def get_pred_logits(cur_input_seq, input_seq, state, p_eval_step, config):
  padding = np.zeros(
      (input_seq.shape[0], config.seq_len - len(cur_input_seq[0])),
      dtype=np.int32,
  )
  concat_batch = np.hstack((cur_input_seq, padding))
  concat_batch = common_utils.shard(
      jax.tree_util.tree_map(np.asarray, concat_batch)
  )

  pred_logits = p_eval_step(state, concat_batch)
  return pred_logits


def get_beam_search_candidates(
    input_seq, beam_search_candidates, state, p_eval_step, pos, config
    ):
  """Get beam search candidates for decoding."""
  new_beam_candidate_list = []
  new_beam_candidate_likelihood_list = []
  for i in range(len(beam_search_candidates)):
    ### Iterate through all the beam search candidates

    # predict the logits for row/column/value
    pred_logits = get_pred_logits(
        beam_search_candidates[i][0], input_seq, state, p_eval_step, config
    )

    # Choose top beam_search_n most probable predictions
    max_pos = (
        pred_logits[:, :, pos, :]
        .argpartition(-config.beam_search_n, axis=-1)[
            :, :, -config.beam_search_n :
        ]
        .reshape((-1, config.beam_search_n))
    )
    log_likelihood = jax.nn.log_softmax(pred_logits[:, :, pos, :]).reshape(
        (-1, pred_logits.shape[-1])
    )
    log_likelihood = np.take_along_axis(log_likelihood, max_pos, 1)

    # Append all of the candidates in new_beam_candidate_list
    for j in range(config.beam_search_n):
      cur_candidate = beam_search_candidates[i]
      new_beam_candidate = np.hstack(
          (cur_candidate[0], jnp.reshape(max_pos[:, j], shape=(-1, 1)))
      )
      new_beam_candidate_likelihood = cur_candidate[1] + log_likelihood[:, j]
      new_beam_candidate_likelihood_list.append(new_beam_candidate_likelihood)
      new_beam_candidate_list.append(
          (new_beam_candidate, new_beam_candidate_likelihood, cur_candidate[2])
      )

  # Likelihood list for new candidates
  new_beam_candidate_likelihood_list = np.stack(
      new_beam_candidate_likelihood_list, axis=0
  )
  assert new_beam_candidate_likelihood_list.shape == (
      len(beam_search_candidates) * config.beam_search_n,
      config.minibatch_size,
  ), new_beam_candidate_likelihood_list.shape

  # Find index of top beam_search_n in new candidates
  new_beam_candidate_ind = new_beam_candidate_likelihood_list.argpartition(
      -config.beam_search_n, axis=0
  )[-config.beam_search_n :, :]
  assert new_beam_candidate_ind.shape == (
      config.beam_search_n,
      config.minibatch_size,
  ), new_beam_candidate_ind.shape

  # Create the new list by truncating to top beam_search_n candidate
  truncated_candidate_list = []
  for i in range(config.beam_search_n):
    new_candidate = np.zeros_like(new_beam_candidate_list[0][0])
    new_candidate_likelihood = np.zeros_like(new_beam_candidate_list[0][1])
    new_candidate_success_pred = np.zeros_like(new_beam_candidate_list[0][2])

    for j in range(config.minibatch_size):
      index = new_beam_candidate_ind[i, j]

      new_candidate[j] = new_beam_candidate_list[index][0][j]
      new_candidate_likelihood[j] = new_beam_candidate_list[index][1][j]
      new_candidate_success_pred[j] = new_beam_candidate_list[index][2][j]

    truncated_candidate_list.append(
        (new_candidate, new_candidate_likelihood, new_candidate_success_pred)
    )

  return truncated_candidate_list


def get_greedy_row_col(
    beam_search_candidates, pos, input_seq, state, p_eval_step, config
    ):
  """Perform greedy row and column decoding using beam search candidates."""

  ### Get beam search candidates for row
  beam_search_candidates = get_beam_search_candidates(
      input_seq, beam_search_candidates, state, p_eval_step, pos - 3, config
  )

  ### Get beam search candidates for column
  beam_search_candidates = get_beam_search_candidates(
      input_seq, beam_search_candidates, state, p_eval_step, pos - 2, config
  )
  ### Predict most confident column according to row
  # pred_logits = get_pred_logits(cur_input_seq, input_seq,
  #                                 state, p_eval_step, config)

  # max_pos = pred_logits[:, :, pos-2, :].argmax(axis=-1).flatten()
  # cur_input_seq = np.hstack((
      # cur_input_seq, jnp.reshape(max_pos, shape=(-1, 1))))
  return beam_search_candidates


def get_greedy_pair(cur_input_seq, pos, input_seq, state, p_eval_step, config):
  """Get greedy pair decoding."""
  pred_logits = get_pred_logits(
      cur_input_seq, input_seq, state, p_eval_step, config
  )

  row_pred_logits = pred_logits[:, :, pos - 3, :].reshape(
      (-1, pred_logits.shape[-1])
  )
  row_log_prob = jax.nn.log_softmax(row_pred_logits[:, 1:10])

  pairs_log_prob = np.zeros((input_seq.shape[0], 81))

  for i in range(9):
    row_num = np.ones((input_seq.shape[0], 1), dtype=np.int32) * (i + 1)
    cur_input_seq = np.hstack((cur_input_seq, row_num))

    pred_logits_col = get_pred_logits(
        cur_input_seq, input_seq, state, p_eval_step, config
    )
    pred_logits_col = pred_logits_col[:, :, pos - 2, :].reshape(
        (-1, pred_logits.shape[-1])
    )

    col_log_prob = jax.nn.log_softmax(pred_logits_col[:, 1:10])

    for j in range(input_seq.shape[0]):
      for k in range(9):
        pairs_log_prob[j, i * 9 + k] = col_log_prob[j, k] + row_log_prob[j, i]

  pair = np.hstack((
      pairs_log_prob.argmax(axis=-1, keepdims=True) // 9,
      pairs_log_prob.argmax(axis=-1, keepdims=True) % 9,
  ))
  return np.hstack((cur_input_seq, pair))


def get_accuracy(
    cur_input_seq,
    state,
    p_eval_step,
    input_seq,
    puzzle_sol,
    config,
    eval_metrics,
    mistakes_metrics,
    ):
  """Get accuracy of a decoding sequence."""
  total_pred, _ = 0, 0

  ### Keeps tuple of best n sequences, log probability and correct pred for it
  beam_search_candidates = [(
      cur_input_seq,
      np.zeros(len(cur_input_seq)),
      np.zeros(len(cur_input_seq)),
  )]

  for i in range(config.start_index * 3 + 2, config.seq_len, 3):
    if config.sampling_method == "greedy-row-col":
      # greedy-row-col: selects first max probability row and
      #                 then max probability column.
      beam_search_candidates = get_greedy_row_col(
          beam_search_candidates, i, input_seq, state, p_eval_step, config
      )

    elif config.sampling_method == "greedy-pair":
      # greedy-pair: selects max probability (row, column) pair
      cur_input_seq = get_greedy_pair(
          cur_input_seq, i, input_seq, state, p_eval_step, config
      )

    beam_search_candidates = get_beam_search_candidates(
        input_seq, beam_search_candidates, state, p_eval_step, i - 1, config
    )

    total_pred += len(beam_search_candidates[0][0])
    for candidate in beam_search_candidates:
      for j in range(
          len(candidate[0])
      ):  ## Iterate through all examples in batch
        try:
          sudoku.SudokuBoardStateUpdate(
              puzzle_sol[j],
              candidate[0][j][i - 2],
              candidate[0][j][i - 1],
              candidate[0][j][i],
          )

          # row_num = cur_input_seq[j, i-2] - 1
          # col_num = cur_input_seq[j, i-1] - 1

          # strategy_id = input_seq_strategies[ j, row_num * 9 + col_num ]
          # mistakes_metrics['total_strategies'][ strategy_id ] += 1

        except AssertionError:
          # mistakes_metrics['mistakes'].append((
              # concat_batch[j], puzzle_sol[j]))
          # # if i < 81:
          # mistakes_metrics['mistake_pos'][i // 3] += 1
          # if first_mistake_ind[j] == 0:
          #   mistakes_metrics['first_mistake_pos'][i // 3] += 1

          #   row_num = cur_input_seq[j, i-2] - 1
          #   col_num = cur_input_seq[j, i-1] - 1
          #   strategy_id = input_seq_strategies[j, row_num * 9 + col_num ]
          #   mistakes_metrics['first_mistake_strategies'][ strategy_id ] += 1

          # g3pdb.set_trace()
          # if strategy_id == 0:
          #   g3pdb.set_trace()

          # first_mistake_ind[j] = 1
          pass
        else:
          candidate[2][j] += 1

    # cur_input_seq = input_seq[:, :(i+1)]

  max_prob_seq = np.zeros_like(beam_search_candidates[0][0])
  max_prob = np.zeros(
      (len(beam_search_candidates), beam_search_candidates[0][1].shape[0])
  )

  for j, candidate in enumerate(beam_search_candidates):
    max_prob[j, :] = candidate[1]

  max_prob_seq_ind = max_prob.argmax(axis=0)
  sucess_pred = np.zeros(len(max_prob_seq))

  for i in range(len(max_prob_seq)):
    max_prob_seq[i] = beam_search_candidates[max_prob_seq_ind[i]][0][i]
    sucess_pred[i] = beam_search_candidates[max_prob_seq_ind[i]][2][i]

  eval_metrics["acc"].append(sucess_pred.sum() * 1.0 / total_pred)
  return eval_metrics, mistakes_metrics, max_prob_seq


def set_set_accuracies(eval_metrics, set_acc, correct_cnt):
  eval_metrics["set_acc1"].append(set_acc[0])
  eval_metrics["set_acc2"].append(set_acc[1])
  eval_metrics["set_acc3"].append(set_acc[2])

  eval_metrics["correct_cnt1"].append(correct_cnt[0])
  eval_metrics["correct_cnt2"].append(correct_cnt[1])
  eval_metrics["correct_cnt3"].append(correct_cnt[2])

  return eval_metrics


def get_position_hinted_eval_acc(
    input_seq, puzzle_sol, state, p_eval_step, eval_metrics, config
    ):
  """This function computes the accuracy of the position hinted decoding model."""

  total_pred, sucess_pred = 0, 0

  cur_input_seq = input_seq[:, : (config.start_index * 3)]
  for i in range(config.start_index, config.block_size):
    ### i^th cell in sequence will predict

    # Append the row number from the ground truth sequence
    cur_input_seq = np.hstack(
        (cur_input_seq, jnp.reshape(input_seq[:, 3 * i], shape=(-1, 1)))
    )

    # Append the column number from the ground truth sequence
    cur_input_seq = np.hstack(
        (cur_input_seq, jnp.reshape(input_seq[:, 3 * i + 1], shape=(-1, 1)))
    )

    padding = np.zeros(
        (input_seq.shape[0], config.seq_len - len(cur_input_seq[0])),
        dtype=np.int32,
    )
    concat_batch = np.hstack((cur_input_seq, padding))
    concat_batch = common_utils.shard(
        jax.tree_util.tree_map(np.asarray, concat_batch)
    )

    # Predict and append value at the pos chosen by the ground truth sequence
    pred_logits = p_eval_step(state, concat_batch)
    max_number = pred_logits[:, :, (3 * i + 1), :].argmax(axis=-1).flatten()
    cur_input_seq = np.hstack(
        (cur_input_seq, jnp.reshape(max_number, shape=(-1, 1)))
    )
    for j in range(
        len(cur_input_seq)
    ):  ## Iterate through all examples in batch
      total_pred += 1
      try:
        sudoku.SudokuBoardStateUpdate(
            puzzle_sol[j],
            cur_input_seq[j, -3],
            cur_input_seq[j, -2],
            cur_input_seq[j, -1],
        )
      except AssertionError:
        pass
      else:
        sucess_pred += 1

  eval_metrics["hinted_acc"].append(sucess_pred * 1.0 / total_pred)
  return eval_metrics


def get_internal_model_stats(
    cur_input_seq,
    state,
    p_eval_step,
    input_seq,
    candidate_list,
    config,
    eval_metrics,
    ):
  """This function computes the internal model stats."""

  for i in range(10):  ### Checks internal model stats at [35, 40, 45,..., 80]
    ## Find already filled cell upto 35th position
    filled_cells = np.zeros((len(cur_input_seq), 81), dtype=np.int8)

    for i1 in range(len(cur_input_seq)):
      for j1 in range(5 * i + 35):
        cell_pos = int(
            (cur_input_seq[i1, 3 * j1] - 1) * 9
            + (cur_input_seq[i1, 3 * j1 + 1] - 1)
        )
        filled_cells[i1, cell_pos] = 1

    cur_board_state = cur_input_seq[:, : (3 * (5 * i + 35))]
    correct_pred = 0
    total_pred = 0

    for j in range(81):
      row = (j // 9) + 1
      col = (j % 9) + 1
      test_board_state = np.hstack((
          cur_board_state,
          np.ones((len(cur_input_seq), 1), dtype=np.int8) * row,
      ))
      test_board_state = np.hstack((
          test_board_state,
          np.ones((len(cur_input_seq), 1), dtype=np.int8) * col,
      ))

      pred_logits = get_pred_logits(
          test_board_state, input_seq, state, p_eval_step, config
      )

      pos = 3 * (5 * i + 35) + 1
      pred_logits = pred_logits[:, :, pos, :].reshape(
          (len(cur_input_seq), pred_logits.shape[-1])
      )

      for k in range(len(cur_input_seq)):

        num_candidates = np.sum(candidate_list[k, i, j])
        if filled_cells[k, j] == 1 or num_candidates == 0:
          continue

        model_candidates = pred_logits[k].argpartition(
            -num_candidates, axis=-1
        )[-num_candidates:]
        correct_pred += np.sum(candidate_list[k, i, j][model_candidates - 1])
        total_pred += num_candidates

    eval_metrics["intermediate_calc_acc" + str(5 * i + 35)].append(
        correct_pred * 1.0 / total_pred
    )
  return eval_metrics


def get_sudoku_eval_metrics(
    state, eval_data_iter, p_eval_step, config
    ):
  """This function computes given evaluation metrics (e.g, accuracy).

  Args:
    state: contains model parameters, optimizer, etc.
    eval_data_iter: data iterator for evaluation dataset
    p_eval_step: pmap function for forward pass of model for evaluation
    config: general config file

  Returns:
    eval_metrics: contains list of evaluation metrics for each batch
  """

  eval_metrics = {
      "acc": [],
      "hinted_acc": [],
      "acc_complete_puzzle": [],
      "edit_distance": [],
      "set_acc1": [],
      "set_acc2": [],
      "set_acc3": [],
      "correct_cnt1": [],
      "correct_cnt2": [],
      "correct_cnt3": [],
  }

  eval_metrics.update(
      {"intermediate_calc_acc" + str(5 * i + 35): [] for i in range(10)}
  )

  mistakes = []
  mistake_pos = np.zeros(81, dtype=np.int32)
  first_mistake_pos = np.zeros(81, dtype=np.int32)
  first_mistake_strategies = np.zeros(8, dtype=np.int32)
  total_strategies = np.zeros(8, dtype=np.int32)
  mistakes_metrics = {
      "mistakes": mistakes,
      "mistake_pos": mistake_pos,
      "first_mistake_pos": first_mistake_pos,
      "first_mistake_strategies": first_mistake_strategies,
      "total_strategies": total_strategies,
  }

  for eval_epoch in range(config.eval_epochs):
    with jax.profiler.StepTraceAnnotation("eval", step_num=eval_epoch):

      batch_tuple = next(eval_data_iter)

      # Input seq is of the shape (batchsize, 3*81) and 3*81 because row, column
      # and value for each cell. Row, column and value all are in {1, ..., 9}
      input_seq = np.array(batch_tuple[0])

      # Puzzle solution is of the shape (batchsize, 81). Each pos in {0,.., 80}
      # for each puzzle contains value at cell (pos//9+1, pos%9 + 1)
      puzzle_sol = np.array(batch_tuple[1])

      cur_input_seq = input_seq[:, : (config.start_index * 3)]

      set_acc, correct_cnt, _ = get_set_accuracies(
          state, p_eval_step, input_seq, config
      )

      eval_metrics = set_set_accuracies(eval_metrics, set_acc, correct_cnt)

      eval_metrics, mistakes_metrics, cur_input_seq = get_accuracy(
          cur_input_seq,
          state,
          p_eval_step,
          input_seq,
          puzzle_sol,
          config,
          eval_metrics,
          mistakes_metrics,
      )

      eval_metrics = get_position_hinted_eval_acc(
          input_seq, puzzle_sol, state, p_eval_step, eval_metrics, config
      )

      correct_eval_sudoku_puzzle = 0
      solution_edit_distance = 0.0

      for i, _ in enumerate(cur_input_seq):
        correct_eval_sudoku_puzzle += valid_solution(cur_input_seq[i])
        solution_edit_distance += get_edit_distance(
            config, cur_input_seq[i], input_seq[i]
        )

      eval_metrics["acc_complete_puzzle"].append(
          correct_eval_sudoku_puzzle * 1.0 / len(cur_input_seq)
      )

      eval_metrics["edit_distance"].append(
          solution_edit_distance * 1.0 / len(cur_input_seq)
      )
  return eval_metrics, mistakes_metrics


def get_eval_metrics(
    step, state, eval_data_iter, p_eval_step, config
):
  if config.dataset == "othello":
    return get_othello_eval_metrics(
        state, eval_data_iter, p_eval_step, config
    )
  elif "sudoku" in config.dataset:
    return get_sudoku_eval_metrics(
        step, state, eval_data_iter, p_eval_step, config
    )


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and returns it."""
  # The supplied figure is closed and inaccessible after this call.
  buf = io.BytesIO()
  plt.savefig(buf, format="png")
  plt.close(figure)
  buf.seek(0)

  image = tf.image.decode_png(buf.getvalue(), channels=4)
  image = tf.expand_dims(image, 0)
  return image


def plot_ax(ax, num, wr, wc):
  """Plots the given axis with the given number of values."""
  for i in range(9):
    for j in range(9):
      if num[i, j] == 0:
        continue
      ax.text(
          i + 0.5, (8 - j) + 0.5, str(int(num[i, j])), ha="center", va="center"
      )

  ax.axis([0, 9, 0, 9])

  rect = matplotlib.patches.Rectangle((wr, 8 - wc), 1, 1, color="red")
  ax.add_patch(rect)

  for axis in [ax.xaxis, ax.yaxis]:
    axis.set_minor_locator(mticker.MultipleLocator(1))
    axis.set_major_locator(mticker.MultipleLocator(3))
  #     axis.set_ticks(np.arange(maxnum) + 0.5)
  #     axis.set_ticklabels(range(maxnum))

  ax.grid(which="minor")
  # ax.axis('off')
  ax.xaxis.set_ticks_position("top")

  ax.hlines(y=3, xmin=0, xmax=10, color="0")
  ax.hlines(y=6, xmin=0, xmax=10, color="0")
  ax.vlines(x=6, ymin=0, ymax=10, color="0")
  ax.vlines(x=3, ymin=0, ymax=10, color="0")
