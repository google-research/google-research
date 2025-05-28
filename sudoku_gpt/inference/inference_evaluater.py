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

"""This file coordinates with all other files to get the evaluation metrics."""

import functools

from absl import logging
from clu import metric_writers
from flax import jax_utils
from flax import linen as nn
from flax.training import checkpoints
import jax
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sudoku_gpt import model
from sudoku_gpt import trainer
from sudoku_gpt.inference import data
from sudoku_gpt.inference import inference_eval_utils


def log_hyperparams_tb(
    config, model_config, initial_variables, tf_summary_writer
    ):
  """Log hyperparameters to TensorBoard."""
  config.num_model_parameters = sum(
      x.size for x in jax.tree_util.tree_leaves(initial_variables)
  )

  config_hyperparameters = [
      tf.convert_to_tensor([k, str(v)]) for k, v in config.items()
  ]
  model_config_hyperparameters = [
      tf.convert_to_tensor([k, str(v)])
      for k, v in model_config.__dict__.items()
  ]

  with tf_summary_writer.as_default():
    tf.summary.text(
        'Model hyperparameters', tf.stack(model_config_hyperparameters), step=0
    )
    tf.summary.text(
        'Config hyperparameters', tf.stack(config_hyperparameters), step=0
    )

  return tf_summary_writer, config


def inference_evaluate(config, workdir, ckpt_loc):
  """Perform inference evaluation on a checkpoint."""
  logging.info('Creating training dataset iterator')
  eval_data_iter = data.create_iter(
      config.dataset_path, config, config.minibatch_size, eval=True
  )

  logging.info('Finished creating training dataset iterator')

  model_config = model.TransformerConfig(
      dataset_fn=config.dataset,
      dtype=config.dtype,
      vocab_size=config.vocab_size,
      seq_len=config.seq_len,
      num_heads=config.num_heads,
      num_layers=config.num_layers,
      emb_dim=config.emb_dim,
      qkv_dim=config.qkv_dim,
      mlp_dim=config.mlp_dim,
      dropout_rate=config.dropout_rate,
      attention_dropout_rate=config.attention_dropout_rate,
      deterministic=False,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6),
  )

  logging.info('Starting print training config')
  logging.info('train_config: %s', str(model_config.__dict__))
  print(str(model_config.__dict__), flush=True)

  rng = jax.random.PRNGKey(config.seed)
  rng, init_rng, inference_rng = random.split(rng, num=3)

  rng, dropout_rng = jax.random.split(rng)
  input_shape = (config.minibatch_size, config.seq_len)
  net = model.TransformerLMHeadModel(model_config)
  rng_keys = {'params': init_rng, 'dropout': dropout_rng}
  _, initial_variables = jax.jit(net.init_with_output)(
      rng_keys, jnp.ones(input_shape, jnp.int32)
  )

  ### Defines optimizer and learning rate function
  state, _ = trainer.get_state(config, net, initial_variables)
  state = checkpoints.restore_checkpoint(ckpt_loc, state)

  writer = metric_writers.create_default_writer(
      workdir, asynchronous=False, just_logging=(jax.process_index() > 0)
  )
  tf_summary_writer = tf.summary.create_file_writer(workdir)

  logging.info('config: %s', str(config.__dict__))
  state = jax_utils.replicate(state)

  _ = jax.random.split(rng, jax.local_device_count())

  p_eval_step = jax.pmap(
      functools.partial(
          inference_eval_utils.eval_step,
          config=model_config.replace(deterministic=True),  # pylint: disable=attribute-error
          rng=inference_rng,
      ),
      axis_name='batch',
      donate_argnums=(0,),
  )

  _ = trainer.get_metrics_report_progress(
      config, workdir, writer)

  tf_summary_writer, config = log_hyperparams_tb(
      config, model_config, initial_variables, tf_summary_writer
  )

  step = 0
  with metric_writers.ensure_flushes(writer):
    eval_metrics, mistakes_metrics = inference_eval_utils.get_eval_metrics(
        step,
        state,
        eval_data_iter,
        p_eval_step,
        config,
    )

    with tf_summary_writer.as_default():
      for key in eval_metrics.keys():
        tf.summary.scalar(
            'eval_' + key, np.array(eval_metrics[key]).mean(), step=step
        )

      fig_mistake_pos = plt.figure()
      plt.plot(np.arange(81), mistakes_metrics['mistake_pos'])
      tf.summary.image(
          'mistakes position',
          inference_eval_utils.plot_to_image(fig_mistake_pos),
          step=step,
      )

      fig_first_mistake_pos = plt.figure()
      plt.plot(np.arange(81), mistakes_metrics['first_mistake_pos'])
      tf.summary.image(
          'first mistakes position',
          inference_eval_utils.plot_to_image(fig_first_mistake_pos),
          step=step,
      )

      fig_first_mistake_strategies = plt.figure()
      fig_label = [
          str(element)
          for element in mistakes_metrics['first_mistake_strategies']
      ]
      fig_color = [
          'red',
          'tan',
          'lime',
          'lightblue',
          'blue',
          'purple',
          'darkred',
          'orange',
      ]

      plt.bar(
          np.arange(8),
          mistakes_metrics['first_mistake_strategies'],
          label=fig_label,
          color=fig_color,
      )

      plt.legend()
      tf.summary.image(
          'first mistakes strategies',
          inference_eval_utils.plot_to_image(fig_first_mistake_strategies),
          step=step,
      )

      fig_strategies_list = plt.figure()
      fig_label = [
          str(element) for element in mistakes_metrics['total_strategies']
      ]

      plt.bar(
          np.arange(8),
          mistakes_metrics['total_strategies'],
          label=fig_label,
          color=fig_color,
      )

      plt.legend()
      tf.summary.image(
          'Total strategies',
          inference_eval_utils.plot_to_image(fig_strategies_list),
          step=step,
      )

      for eid in range(config.num_examples):
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        cur_board = np.zeros((9, 9))
        puzzle_sol_board = np.zeros((9, 9))

        for j in range(0, 3 * 81, 3):
          row_num = mistakes_metrics['mistakes'][eid][0][j] - 1
          col_num = mistakes_metrics['mistakes'][eid][0][j + 1] - 1
          val = mistakes_metrics['mistakes'][eid][0][j + 2]
          cur_board[row_num, col_num] = val

        for j in range(9):
          for k in range(9):
            puzzle_sol_board[k, j] = mistakes_metrics['mistakes'][eid][1][
                9 * j + k
            ]

        wr, wc = 0, 0
        for j in range(9):
          for k in range(9):
            if cur_board[j, k] == 0:
              continue
            if cur_board[j, k] != puzzle_sol_board[k, j]:
              wr = j
              wc = k

        inference_eval_utils.plot_ax(axs[0], cur_board, wr, wc)
        inference_eval_utils.plot_ax(axs[1], puzzle_sol_board.T, wr, wc)

        tf.summary.image(
            'Mistakes ' + str(eid),
            inference_eval_utils.plot_to_image(fig),
            step=step,
        )
