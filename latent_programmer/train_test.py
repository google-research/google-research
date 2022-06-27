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

"""Test seq-to-seq training."""

# pytype: disable=wrong-arg-count
# pytype: disable=attribute-error

import functools
import os
import random
import sys

from absl.testing import absltest

from flax import jax_utils
from flax import optim
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf

from latent_programmer import models
from latent_programmer import train_lib
from latent_programmer.tasks.robust_fill import dsl
from latent_programmer.tasks.robust_fill import tokens as dsl_tokens
from latent_programmer.tasks.robust_fill.dataset import input_pipeline

gfile = tf.io.gfile
sys.path.append('..')


class TrainTest(absltest.TestCase):

  def test_train(self):
    tf.enable_v2_behavior()

    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    dataset_filepattern = os.path.join(
        os.path.dirname(__file__),
        'tasks/robust_fill/dataset/test_dataset/program_tasks.tf_records-*')

    print('dataset_filepattern = {}'.format(dataset_filepattern))

    batch_size = 4
    num_strings_per_task = 4
    max_characters = 10
    max_program_length = 15

    # Build token tables.
    id_char_table = {i+1: char for (i, char) in enumerate(dsl.CHARACTER)}
    char_id_table = {char: id for id, char in id_char_table.items()}
    _, token_id_table = dsl_tokens.build_token_tables()
    io_vocab_size = len(char_id_table) + 1  # For padding.
    program_vocab_size = len(token_id_table) + 1

    bos_token = token_id_table[dsl.BOS]

    # Load dataset.
    dataset = input_pipeline.create_dataset_from_tf_record(
        dataset_filepattern, token_id_table, char_id_table)
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=((num_strings_per_task, max_characters),
                       (num_strings_per_task, max_characters),
                       (max_program_length,)),
        drop_remainder=True)
    dataset_iter = dataset.repeat().as_numpy_iterator()

    train_config = models.TransformerConfig(
        vocab_size=io_vocab_size,
        output_vocab_size=program_vocab_size,
        shift=True,
        emb_dim=32,
        num_heads=4,
        num_layers=2,
        qkv_dim=32,
        mlp_dim=32,
        max_len=max(max_characters, max_program_length),
        deterministic=False,
        decode=False,
        bos_token=bos_token)
    eval_config = train_config.replace(deterministic=True)

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    m = models.ProgramTransformer(eval_config)
    initial_variables = jax.jit(m.init)(
        init_rng,
        jnp.ones((batch_size, num_strings_per_task, max_characters),
                 jnp.float32),
        jnp.ones((batch_size, num_strings_per_task, max_characters),
                 jnp.float32),
        jnp.ones((batch_size, max_program_length), jnp.float32))

    optimizer_def = optim.Adam(
        1e-2,
        beta1=0.9,
        beta2=0.98,
        eps=1e-9,
        weight_decay=0.1)
    optimizer = optimizer_def.create(initial_variables['params'])

    del initial_variables  # Don't keep a copy of the initial model.

    optimizer = jax_utils.replicate(optimizer)

    learning_rate_fn = train_lib.create_learning_rate_scheduler(
        base_learning_rate=1e-2)
    p_train_step = jax.pmap(
        functools.partial(
            train_lib.train_step,
            learning_rate_fn=learning_rate_fn,
            config=train_config),
        axis_name='batch')
    p_eval_step = jax.pmap(
        functools.partial(train_lib.eval_step, config=eval_config),
        axis_name='batch')

    # Training loop.
    start_step = 0
    rngs = jax.random.split(rng, jax.local_device_count())
    del rng

    for _ in range(start_step, 1000):
      inputs, outputs, programs = common_utils.shard(next(dataset_iter))
      optimizer, _, rngs = p_train_step(
          optimizer, inputs, outputs, programs, train_rng=rngs)

    # Evaluation.
    eval_metrics = []
    for batches in dataset.as_numpy_iterator():
      inputs, outputs, programs = common_utils.shard(batches)

      metrics = p_eval_step(optimizer.target, inputs, outputs, programs)
      eval_metrics.append(metrics)

    eval_metrics = common_utils.get_metrics(eval_metrics)
    eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
    eval_denominator = eval_metrics_sums.pop('denominator')
    eval_summary = jax.tree_map(
        lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
        eval_metrics_sums)

    if jax.host_id() == 0:
      self.assertGreater(eval_summary['accuracy'], 0.1)


if __name__ == '__main__':
  absltest.main()
