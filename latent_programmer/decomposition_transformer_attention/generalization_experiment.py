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

"""Runs trained models on datasets for decomposition-based generalization."""

import collections
import functools
import itertools
import json
import os

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import optim
from flax.training import checkpoints
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from latent_programmer.decomposition_transformer_attention import decomposition_models as models
from latent_programmer.decomposition_transformer_attention import input_pipeline
from latent_programmer.decomposition_transformer_attention import train as train_lib
from latent_programmer.models import base_models
from latent_programmer.tasks.robust_fill import dsl
from latent_programmer.tasks.robust_fill import tokens as dsl_tokens
from latent_programmer.tasks.robust_fill.dataset import experiment as exp_module

FLAGS = flags.FLAGS

flags.DEFINE_string('datasets_directory', None,
                    'Directory that contains different experiment datasets.')
flags.DEFINE_string('train_directory', None,
                    'Directory containing training runs.')

flags.DEFINE_integer('beam_size', 10, 'Beam size.')


# Constants that never change during this evaluation run, as global constants
# for convenience.

slow_decode = True
use_dropout = False

per_device_batch_size = 16
n_devices = 8
batch_size = per_device_batch_size * n_devices
num_strings_per_task = 4
max_expressions = 10
max_program_length = 100
max_characters = 200

embedding_dim = 256
hidden_dim = 512
num_heads = 4
num_layers = 3

io_shape = (per_device_batch_size,
            num_strings_per_task,
            max_characters)
program_shape = (per_device_batch_size, max_program_length)

id_char_table = {i+1: char for (i, char) in enumerate(dsl.CHARACTER)}
char_id_table = {char: id for id, char in id_char_table.items()}
id_token_table, token_id_table = dsl_tokens.build_token_tables()
io_vocab_size = len(char_id_table) + 1  # For padding.
program_vocab_size = len(token_id_table) + 1

bos_token = token_id_table[dsl.BOS]
eos_token = token_id_table[dsl.EOS]


DATASETS = [e.name for e in exp_module.Experiment
            if e != exp_module.Experiment.NONE]
# DATASETS = [DATASETS[0]]  # TODO(kshi): temporary

# Model-specific hyperparameters.
# (attention_mask_type, use_relative_attention, bos_special_attention)
MODELS = []
attention_mask_types = [
    'baseline',
    'bos_full_attention',
    'bos_to_last',
    'bos_to_bos_and_last',
]
for amt, ura, bsa in itertools.product(attention_mask_types,
                                       [False, True], [False, True]):
  if bsa and (not ura or amt == 'baseline'):
    continue
  MODELS.append((amt, ura, bsa))


def load_dataset(dataset_name, split):
  """Loads the dataset for a particular experiment and split."""
  assert split in ['valid', 'test']
  # Create dataset.
  dataset = input_pipeline.create_dataset_from_tf_record(
      os.path.join(
          FLAGS.datasets_directory,
          '{}_data/program_tasks_{}.tf_records*'.format(dataset_name, split)),
      token_id_table,
      char_id_table)
  dataset = dataset.padded_batch(
      batch_size,
      padded_shapes=(io_shape[1:], io_shape[1:], program_shape[1:]),
      drop_remainder=False)

  dataset = dataset.take(50)  # TODO(kshi): Temporarily run quicker
  return dataset


def get_predict_config(attention_mask_type, use_relative_attention,
                       bos_special_attention):
  """Constructs a config for prediction."""
  base_config = base_models.TransformerConfig(
      vocab_size=io_vocab_size,
      output_vocab_size=program_vocab_size,
      shift=True,
      emb_dim=embedding_dim,
      num_heads=num_heads,
      num_layers=num_layers,
      qkv_dim=embedding_dim,
      mlp_dim=hidden_dim,
      max_len=max(max_characters, max_program_length),
      use_relative_attention=use_relative_attention,
      deterministic=not use_dropout,
      decode=False,
      bos_token=bos_token)
  predict_config = models.DecomposeAttentionTransformerConfig(
      base_config=base_config.replace(
          shift=False,
          deterministic=not use_dropout,
          decode=not slow_decode),
      attention_mask_type=attention_mask_type,
      bos_special_attention=bos_special_attention)
  return predict_config


def load_model(dataset_name, attention_mask_type, use_relative_attention,
               bos_special_attention, predict_config):
  """Loads a checkpoint."""
  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)

  m = models.DecomposeAttentionTransformer(predict_config)
  initial_variables = jax.jit(m.init)(
      {'params': init_rng, 'dropout': init_rng},
      jnp.ones(io_shape, jnp.float32),
      jnp.ones(io_shape, jnp.float32),
      jnp.ones(program_shape, jnp.float32))

  optimizer_def = optim.Adam(
      1e-3,
      beta1=0.9,
      beta2=0.98,
      eps=1e-9,
      weight_decay=1e-1)
  optimizer = optimizer_def.create(initial_variables['params'])

  checkpoint_fname = os.path.join(
      FLAGS.train_directory,
      'train-{}/checkpoints/'
      'amt={},bsa={},ed=256,hd=512,l=0.001,nh=4,nl=3,s=0,ura={}/'.format(
          dataset_name, attention_mask_type, bos_special_attention,
          use_relative_attention))
  logging.info('Loading checkpoint: %s', checkpoint_fname)

  optimizer = checkpoints.restore_checkpoint(checkpoint_fname, optimizer)
  checkpoint_num_trained_steps = int(optimizer.state.step)
  logging.info('Found model checkpointed at step %s.',
               checkpoint_num_trained_steps)
  optimizer = jax_utils.replicate(optimizer)

  return optimizer


def decode_io(inputs, outputs):
  """Decode io examples tokens."""
  def decode_str(s):
    """Decode string tokens."""
    return ''.join([id_char_table[c_id] for c_id in s if c_id > 0])

  inps, outs = [], []
  for inp, out in zip(inputs, outputs):
    inps.append(decode_str(inp))
    outs.append(decode_str(out))
  return inps, outs


def decode_program(program):
  """Decode program tokens."""
  program = program[:np.argmax(program == eos_token) + 1].astype(np.int32)
  program = program[program != bos_token]

  try:
    return dsl.decode_program(program.tolist(), id_token_table)
  except:  # pylint: disable=bare-except
    return None  # Program does not compile.


def do_prediction(p_init_cache, p_pred_step, dataset, optimizer, beam_size,
                  verbose=False):
  """Runs the model on a dataset."""
  total_acc = 0
  total_denominator = 0
  pred_accs = collections.defaultdict(int)
  pred_denominators = collections.defaultdict(int)
  program_lengths = collections.defaultdict(list)
  ios, targets, predictions = [], [], []

  for batches in dataset.as_numpy_iterator():
    inputs, outputs, programs = common_utils.shard(batches)
    cache = (p_init_cache(inputs, outputs, programs)
             if not slow_decode else None)
    predicted = p_pred_step(optimizer.target, inputs, outputs, cache, beam_size)
    predicted = train_lib.tohost(predicted)
    inputs, outputs, programs = map(train_lib.tohost,
                                    (inputs, outputs, programs))

    for i, beams in enumerate(predicted):
      inps, outs = decode_io(inputs[i], outputs[i])
      p, p_score = train_lib.eval_predicted(
          beams, inps, outs, parse_beam_fn=decode_program)

      # Split by length of program.
      num_expressions = len(decode_program(programs[i]).expressions)
      program = programs[i]
      program_length = len(program[:np.argmax(program == eos_token)])
      program_lengths[num_expressions].append(program_length)
      pred_denominators[num_expressions] += 1
      total_denominator += 1
      if p_score >= len(inps):
        pred_accs[num_expressions] += 1
        total_acc += 1

      ios.append(' ; '.join(map(str, zip(inps, outs))))
      targets.append(decode_program(programs[i]).to_string())
      try:
        predictions.append(p.to_string())
      except:  # pylint: disable=bare-except
        predictions.append('')

      if verbose:
        logging.info('IOs: %s', ios[-1])
        logging.info('Target: %s', targets[-1])
        logging.info('Top of beam:')
        for index, beam in enumerate(beams[:-5:-1]):
          try:
            decoded_program = decode_program(beam).to_string()
          except:  # pylint: disable=bare-except
            decoded_program = 'Did not compile'
          logging.info('index: %s\n  decoded: %s\n  tokens: %s',
                       index, decoded_program, beam)

  if verbose:
    logging.info('Total: %s/%s = %s%%', total_acc, total_denominator,
                 100 * total_acc / total_denominator)
  return total_acc, total_denominator, pred_accs, pred_denominators


def run_generalization_experiment():
  """Runs all models on all datasets."""
  results = {}

  for dataset_name in DATASETS:
    logging.info('Experiment: %s', dataset_name)
    results[dataset_name] = {}

    valid_dataset = load_dataset(dataset_name, split='valid')
    test_dataset = load_dataset(dataset_name, split='test')

    for model in MODELS:
      logging.info('  Model: %s', model)
      attention_mask_type, use_relative_attention, bos_special_attention = model
      model_name = '{}-{}-{}'.format(
          attention_mask_type, use_relative_attention, bos_special_attention)
      results[dataset_name][model_name] = {}

      predict_config = get_predict_config(
          attention_mask_type, use_relative_attention, bos_special_attention)

      p_init_cache = jax.pmap(
          functools.partial(
              train_lib.initialize_cache,
              max_decode_len=max_program_length,
              config=predict_config),
          axis_name='batch')

      p_predict_step = jax.pmap(
          functools.partial(
              train_lib.predict_step,
              eos_token=eos_token,
              max_decode_len=max_program_length,
              config=predict_config,
              slow_decode=slow_decode),
          axis_name='batch',
          static_broadcasted_argnums=(4,))

      optimizer = load_model(dataset_name, attention_mask_type,
                             use_relative_attention, bos_special_attention,
                             predict_config)

      for split, dataset in [
          ('valid', valid_dataset),
          ('test', test_dataset),
      ]:
        correct, denominator, pred_accs, pred_denominators = do_prediction(
            p_init_cache, p_predict_step, dataset, optimizer,
            beam_size=FLAGS.beam_size)
        frac = 100 * correct / denominator
        logging.info('      %s split: %s/%s = %s%%',
                     split, correct, denominator, frac)
        for length in sorted(pred_accs.keys()):
          logging.info('        length %s: %s/%s = %s%%',
                       length, pred_accs[length], pred_denominators[length],
                       100 * pred_accs[length] / pred_denominators[length])

        results[dataset_name][model_name][split] = (
            correct, denominator, frac, pred_accs, pred_denominators)

  return results


def main(_):
  assert n_devices == jax.local_device_count()

  results = run_generalization_experiment()
  filename = 'results_beam_{}.json'.format(FLAGS.beam_size)
  full_path = os.path.join(FLAGS.train_directory, filename)
  logging.info('Saving results to %s', full_path)
  with tf.io.gfile.GFile(full_path, 'w') as f:
    json.dump(results, f, sort_keys=True, indent=4)
  logging.info('Done!')

if __name__ == '__main__':
  app.run(main)
