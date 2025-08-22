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

"""Evaluation related functions."""

from flax.training import common_utils
import jax
from jax import numpy as jnp
import numpy as np

from sudoku_gpt import model
from sudoku_gpt import othello
from sudoku_gpt import sudoku


def valid_solution(output_seq):
  """Checks if the output sequence is a valid solution for the sudoku puzzle."""
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
      {"params": state.params}, batch)
  return pred_logits


def get_othello_eval_metrics(state, eval_data_iter, p_eval_step, config):
  """Get evaluation metrics for Othello game.

  Args:
    state:
    eval_data_iter: Iterator for evaluation dataset.
    p_eval_step: Function to compute forward pass on a single evaluation batch.
    config: The config for the experiment.

  Returns:

  """
  eval_metrics = {"acc": []}
  for eval_epoch in range(config.eval_epochs):
    with jax.profiler.StepTraceAnnotation("eval", step_num=eval_epoch):

      batch = np.array(next(eval_data_iter))
      total_pred, sucess_pred = 0, 0

      for i in range(config.seq_len):
        padding = np.zeros((batch.shape[0],
                            config.seq_len - (i+1)), dtype=np.int32)
        concat_batch = np.hstack((batch[:, :(i + 1)], padding))
        concat_batch = common_utils.shard(
            jax.tree_util.tree_map(np.asarray, concat_batch)
        )
        pred_logits = p_eval_step(state, concat_batch)

        max_action = pred_logits[:, :, i, :].argmax(axis=-1)
        pred_seq = np.hstack((batch[:, :(i + 1)],
                              jnp.reshape(max_action, shape=(-1, 1))))

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

      eval_metrics["acc"].append(sucess_pred * 1.0/ total_pred)

  return eval_metrics


def get_edit_distance(config, generated_input_seq, original_input_seq):
  """Get edit distance between generated input and original input."""
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
  """Computes accuracy of set of possible values."""
  correct_cnt = np.zeros(9)
  total_cnt = np.zeros(9)

  min_start_index = 31
  for i in range(len(pairs)):
    cur_input_seq = np.hstack(
        (input_seq[:, : (min_start_index * 3)], pairs[i] + 1)
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

    cur_pred_logits = pred_logits[:, :, 3 * min_start_index + 1, :].reshape(
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
  """Computes the sampled pairs at config.start_index + 1 location and return them as pairs."""
  pairs_set = []
  for _ in range(input_seq.shape[0]):
    pairs_set.append(set())

  pairs = np.zeros(
      (config.set_accuracy_top_k, input_seq.shape[0], 2), dtype=np.int32
  )
  flag = True  ## Denotes if we want to sample next time or not.

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
  """This function computes the top k most probable pairs at config.start_index + 1 location and return them as pairs."""

  min_start_index = 31
  pairs = np.zeros(
      (config.set_accuracy_top_k, input_seq.shape[0], 2), dtype=np.int32
  )
  pred_logits_row = pred_logits[:, :, 3 * min_start_index - 1, :].reshape(
      (-1, pred_logits.shape[-1])
  )

  # Row log probability
  row_log_prob = jax.nn.log_softmax(pred_logits_row[:, 1:10])

  pairs_log_prob = np.zeros((input_seq.shape[0], 81))

  for i in range(9):
    row_num = np.ones((input_seq.shape[0], 1), dtype=np.int32) * (i + 1)
    cur_input_seq = np.hstack((input_seq[:, : (min_start_index * 3)], row_num))

    padding = np.zeros(
        (input_seq.shape[0], config.seq_len - len(cur_input_seq[0])),
        dtype=np.int32,
    )

    concat_batch = np.hstack((cur_input_seq, padding))

    concat_batch = common_utils.shard(
        jax.tree_util.tree_map(np.asarray, concat_batch)
    )

    pred_logits_col = p_eval_step(state, concat_batch)
    pred_logits_col = pred_logits_col[:, :, 3 * min_start_index, :].reshape(
        (-1, pred_logits.shape[-1])
    )

    # Column log probability
    col_log_prob = jax.nn.log_softmax(pred_logits_col[:, 1:10])

    # Calculates log probability for each cell by combining log probability for
    # each row and each column
    for j in range(input_seq.shape[0]):
      for k in range(9):
        pairs_log_prob[j, i * 9 + k] = col_log_prob[j, k] + row_log_prob[j, i]

  for i in range(input_seq.shape[0]):
    # Selects top k most probable cells
    topk_indices = np.argsort(pairs_log_prob[i, :])[::-1][
        : config.set_accuracy_top_k
    ]
    for j, ind in enumerate(topk_indices):
      pairs[j, i, 0] = ind // 9
      pairs[j, i, 1] = ind % 9

  return pairs


def get_set_accuracies(state, p_eval_step, input_seq, config):
  """This function computes set accuracies for empty cells in the puzzle at config.start_index + 1 location."""

  possible_vals = np.ones((input_seq.shape[0], 9, 9, 9))
  given_vals = np.zeros((input_seq.shape[0], 9, 9))

  min_start_index = 31
  for i in range(input_seq.shape[0]):
    for j in range(min_start_index):
      row_num = input_seq[i, 3 * j] - 1
      col_num = input_seq[i, 3 * j + 1] - 1
      val = input_seq[i, 3 * j + 2] - 1

      possible_vals[i, row_num, :, val] = 0
      possible_vals[i, :, col_num, val] = 0

      given_vals[i, row_num, col_num] = 1

  if config.set_accuracy == "top-k":
    # Computes the set accuracy for top k most probable positions
    # at config.start_index + 1 location
    cur_input_seq = input_seq[:, : (min_start_index * 3)]
    padding = np.zeros(
        (input_seq.shape[0], config.seq_len - len(cur_input_seq[0])),
        dtype=np.int32,
    )

    concat_batch = np.hstack((cur_input_seq, padding))

    concat_batch = common_utils.shard(
        jax.tree_util.tree_map(np.asarray, concat_batch)
    )

    key = jax.random.PRNGKey(98)
    pred_logits = p_eval_step(state, concat_batch)

    # pairs = get_sampled_pairs(input_seq, pred_logits, state,
    #                           p_eval_step, config, key)

    print("get_topk_probability_pairs", flush=True)
    pairs = get_topk_probability_pairs(
        input_seq, pred_logits, state, p_eval_step, config, key
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
    # Computes the set accuracy for all the pairs at config.start_index + 1
    # location

    pairs = np.zeros((81, input_seq.shape[0], 2), dtype=np.int32)
    for i in range(81):
      pairs[i, :, 0] = np.ones(input_seq.shape[0], dtype=np.int32) * (i // 9)
      pairs[i, :, 1] = np.ones(input_seq.shape[0], dtype=np.int32) * (i % 9)

    # After computing pairs for which we want set accuracy
    # (config.set_accuracy == "all" => pairs contain all position)
    # (config.set_accuracy == "top-k" => pairs containing top-k most probable)
    return get_set_accuracy_for_pairs(
        pairs,
        state,
        p_eval_step,
        input_seq,
        possible_vals,
        given_vals,
        config,
    )


def get_sudoku_eval_metrics(state, eval_data_iter, p_eval_step, config):
  """This function computes given evaluation metrics (e.g, accuracy) in eval metrics for each batch and appends the metric in the list of eval_metrics.

  Args:
    state: contains model parameters, optimizer, etc.
    eval_data_iter: data iterator for evaluation dataset
    p_eval_step: pmap function for forward pass of model for evaluation
    config: general experiment config file

  Returns:
    eval_metrics: contains list of evaluation metrics for each batch
  """

  eval_metrics = {
      "acc": [],
      "acc_complete_puzzle": [],
      "edit_distance": [],
      "set_acc1": [],
      "set_acc2": [],
      "set_acc3": [],
      "correct_cnt1": [],
      "correct_cnt2": [],
      "correct_cnt3": [],
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
      starting_index = np.array(batch_tuple[2])
      total_pred, sucess_pred = 0, 0

      min_start_index = 31
      starting_index = (
          np.ones_like(starting_index, dtype=np.int32) * min_start_index
      )
      cur_input_seq = input_seq[:, :(config.start_index*3)]

      # Computes set accuracy for empty cells in the puzzle
      set_acc, correct_cnt, _ = get_set_accuracies(
          state, p_eval_step, input_seq, config
      )

      eval_metrics["set_acc1"].append(set_acc[0])
      eval_metrics["set_acc2"].append(set_acc[1])
      eval_metrics["set_acc3"].append(set_acc[2])

      eval_metrics["correct_cnt1"].append(correct_cnt[0])
      eval_metrics["correct_cnt2"].append(correct_cnt[1])
      eval_metrics["correct_cnt3"].append(correct_cnt[2])

      for i in range(min_start_index * 3, config.seq_len):
        ### In i^th iteration, i^th number in sequence will predict
        padding = np.zeros((input_seq.shape[0],
                            config.seq_len - len(cur_input_seq[0])),
                           dtype=np.int32)
        concat_batch = np.hstack((cur_input_seq, padding))
        concat_batch = common_utils.shard(
            jax.tree_util.tree_map(np.asarray, concat_batch)
        )

        pred_logits = p_eval_step(state, concat_batch)

        if i%3 == 2:
          # Model predicts the value at the cell (cur_input_seq[j][i-2],
          # cur_input_seq[j][i-1])
          max_number = pred_logits[:, :, i-1, :].argmax(axis=-1).flatten()
          mask_arr = np.array(i >= (3 * starting_index))

          next_number = max_number * mask_arr + (1 - mask_arr) * input_seq[:, i]

          cur_input_seq = np.hstack(
              (cur_input_seq, jnp.reshape(next_number, shape=(-1, 1)))
          )

          # Iterate through all examples in batch and calculate successful
          # predictions of numbers
          for j in range(len(cur_input_seq)):
            if not mask_arr[j]:
              continue

            total_pred += 1
            try:
              sudoku.SudokuBoardStateUpdate(puzzle_sol[j],
                                            cur_input_seq[j][i-2],
                                            cur_input_seq[j][i-1],
                                            cur_input_seq[j][i])
            except AssertionError:
              ### Wrong update
              # if cur_input_seq[j, i-2] * 9 + cur_input_seq[j, i-1] <= 80:
              #   print(puzzle_sol[j][cur_input_seq[j, i-2] * 9 +
              #                       cur_input_seq[j,i-1]],
              #                       cur_input_seq[j][i])
              pass
            else:
              sucess_pred += 1
        else:
          # Model predicts either a row number or column number
          max_pos = pred_logits[:, :, i-1, :].argmax(axis=-1).flatten()
          mask = i >= (3 * starting_index)
          next_pos = max_pos * mask + (1 - mask) * input_seq[:, i]
          cur_input_seq = np.hstack(
              (cur_input_seq, jnp.reshape(next_pos, shape=(-1, 1)))
          )

      eval_metrics["acc"].append(sucess_pred * 1.0/ total_pred)

      correct_eval_sudoku_puzzle = 0
      solution_edit_distance = 0.0

      for i in range(len(cur_input_seq)):

        # increase correct_eval_sudoku_puzzle when the model output solution
        # for a given puzzle is correct
        correct_eval_sudoku_puzzle += valid_solution(cur_input_seq[i])

        # edit distance = distance between model's output order and solver's
        #                 output order
        solution_edit_distance += get_edit_distance(
            config, cur_input_seq[i], input_seq[i]
        )

      eval_metrics["acc_complete_puzzle"].append(
          correct_eval_sudoku_puzzle * 1.0 / len(cur_input_seq)
      )

      eval_metrics["edit_distance"].append(
          solution_edit_distance * 1.0 / len(cur_input_seq)
      )

  return eval_metrics


def get_eval_metrics(state, eval_data_iter,
                     p_eval_step, config):
  if config.dataset == "othello":
    return get_othello_eval_metrics(state, eval_data_iter, p_eval_step,
                                    config)
  elif "sudoku" in config.dataset:
    return get_sudoku_eval_metrics(state, eval_data_iter, p_eval_step,
                                   config)
