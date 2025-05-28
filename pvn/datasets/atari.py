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

"""Dataset."""
import enum
import functools
from typing import Any, Dict, TypedDict, Union

from absl import logging
from dopamine.discrete_domains import atari_lib
import gym
import jax
import jax.numpy as jnp
import rlds
import tensorflow as tf
import tensorflow_datasets as tfds
from pvn.utils.typing_utils import AtariGames


class ElementSpec(TypedDict):
  observation: jax.ShapeDtypeStruct
  action: jax.ShapeDtypeStruct
  next_observation: jax.ShapeDtypeStruct
  terminal: jax.ShapeDtypeStruct


def _episode_to_transitions(
    episode, num_actions, stack_size = 4
):
  """Convert episodes to transitions.

  Args:
    episode: Dict[str, Any], episode to map over.
    num_actions: Number of actions in this game.
    stack_size: # observations to stack.

  Returns:
    tf.data.Dataset: dataset of transitions.
  """

  def _steps_to_transition(batch):
    # Stack batched observations
    states = tf.squeeze(batch[rlds.OBSERVATION], axis=-1)
    states = tf.transpose(states, perm=[1, 2, 0])
    states = tf.cast(states, tf.uint8)
    # We must assign a shape to states or else TF gets confused
    # and we won't be able to derive the shape.
    width, height, _ = states.get_shape().as_list()
    states.set_shape([width, height, stack_size + 1])

    actions = tf.cast(batch[rlds.ACTION], tf.int32)
    terminals = tf.cast(batch[rlds.IS_TERMINAL], tf.bool)

    return {
        'observation': states[:, :, :stack_size],
        'action': tf.one_hot(
            actions[stack_size - 1], depth=num_actions, dtype=tf.uint8
        ),
        'next_observation': states[:, :, 1:],
        'terminal': tf.reduce_any(
            terminals[stack_size : stack_size + 1], keepdims=True
        ),
    }

  batched_steps = rlds.transformations.batch(
      episode[rlds.STEPS], size=stack_size + 1, shift=1, drop_remainder=True
  )
  return batched_steps.map(_steps_to_transition)


def _transpose_observations(batch):
  batch['observation'] = tf.transpose(batch['observation'], perm=[1, 2, 3, 0])
  batch['next_observation'] = tf.transpose(
      batch['next_observation'], perm=[1, 2, 3, 0]
  )

  return batch


@enum.unique
class DatasetSplit(enum.Enum):
  """Different dataset splits.

  These come from the SGI paper: https://arxiv.org/pdf/2106.04799.pdf.

  FULL: full dataset of 50M transitions.
  WEAK: dataset of first 1M transitions from all 5 replay buffer checkpoints
      (5M total transitions).
  MIXED_SMALL: 1M transitions from the beginning, middle, and end of training.
  MIXED: 6M transitions, made of 1M transition buffers equally spread out
      throughout training.
  """

  FULL: str = 'full'
  WEAK: str = 'weak'
  MIXED_SMALL: str = 'mixed_small'
  MIXED: str = 'mixed'


def create_dataset(
    *,
    game,
    run,
    batch_size,
    split = DatasetSplit.FULL,
    version = ':1.0.0',
    data_proportion_per_checkpoint = 1.0,
    episode_shuffle_buffer_size = 1_000,
    transition_shuffle_buffer_size = 200_000,
    cycle_length = 100,
    cache = False,
):
  """Create Atari dataset.

  Reference: https://github.com/google-research/rlds

  Args:
    game: AtariGame, game to create split from.
    run: int, specific run to create dataset from.
    batch_size: int, Batch size.
    split: The dataset split to load.
    version: str, dataset version.
    data_proportion_per_checkpoint: float, percentage of data to sample from
      each checkpoint.
    episode_shuffle_buffer_size: int, buffer size when shuffling episodes.
    transition_shuffle_buffer_size: int, buffer size when shuffling transitions.
    cycle_length: int, cycle length when interleaving checkpoints.
    cache: Cache dataset after interleave.

  Returns:
    Iterable[Transitions]: Iterator over Transitions dataclass.
  """
  path = f'rlu_atari_checkpoints_ordered/{game}_run_{run}{version}'
  builder = tfds.builder(path)

  env = create_environment(game)
  num_actions = env.action_space.n

  splits = []

  split = DatasetSplit(split)
  if split == DatasetSplit.FULL:
    for split_name, info in builder.info.splits.items():
      # Convert `data_percent` to number of episodes to allow
      # for fractional percentages.
      num_episodes = int(data_proportion_per_checkpoint * info.num_examples)
      if num_episodes == 0:
        raise ValueError(
            f'{data_proportion_per_checkpoint*100.0}% leads to 0 '
            f'episodes in {split_name}!'
        )
      # Sample first `data_percent` episodes from each of the data split
      splits.append(f'{split_name}[:{num_episodes}]')
  elif split == DatasetSplit.MIXED_SMALL:
    splits = ['checkpoint_00', 'checkpoint_24', 'checkpoint_49']
  elif split == DatasetSplit.MIXED:
    splits = [
        'checkpoint_00',
        'checkpoint_09',
        'checkpoint_19',
        'checkpoint_29',
        'checkpoint_39',
        'checkpoint_49',
    ]
  elif split == DatasetSplit.WEAK:
    splits = ['checkpoint_00']

  logging.info('Loading dataset with split %s.', split.value)

  tfds_split = tfds.split_for_jax_process('+'.join(splits))

  options = tf.data.Options()
  options.experimental_optimization.map_parallelization = True
  options.experimental_threading.private_threadpool_size = 48
  options.experimental_threading.max_intra_op_parallelism = 1

  # Interleave episodes across different splits/checkpoints
  # Set `shuffle_files=True` to shuffle episodes across files within splits
  read_config = tfds.ReadConfig(
      options=options,
      shuffle_reshuffle_each_iteration=True,
      enable_ordering_guard=False,
  )

  if split == DatasetSplit.WEAK:
    logging.warning('Using the weak dataset and ignoring the run parameter.')
    path_template = f'rlu_atari_checkpoints_ordered/{game}_run_{{}}{version}'
    datasets = []
    for run in range(1, 6):
      ds = tfds.load(
          path_template.format(run), split=tfds_split, read_config=read_config
      )
      datasets.append(ds)

    # This interleaves samples from the various datasets. However,
    # as different datasets reach the end, it will start skipping them,
    # meaning the distribution of checkpoints will change towards the
    # end of this combined dataset. For our use case, this probably
    # isn't too much of a problem.
    dataset = tf.data.Dataset.sample_from_datasets(datasets)
  else:
    dataset = tfds.load(
        path, split=tfds_split, read_config=read_config, shuffle_files=True
    )

  # Configure dataset
  dataset = dataset.shuffle(episode_shuffle_buffer_size)
  dataset = dataset.interleave(
      functools.partial(_episode_to_transitions, num_actions=num_actions),
      cycle_length=cycle_length,
      block_length=1,
      deterministic=False,
      num_parallel_calls=tf.data.AUTOTUNE,
  )
  if cache:
    dataset = dataset.cache()

  # Shuffle + take
  dataset = dataset.shuffle(transition_shuffle_buffer_size)
  dataset = dataset.repeat()
  # Batch + Prefetch
  dataset = dataset.batch(
      batch_size // jax.process_count(),
      drop_remainder=True,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False,
  )
  dataset = dataset.map(
      _transpose_observations, num_parallel_calls=tf.data.AUTOTUNE
  )
  dataset = dataset.prefetch(tf.data.AUTOTUNE)

  return dataset


def element_spec(game, batch_size):
  num_actions = get_num_actions(game)  # pytype: disable=wrong-arg-types
  obs_spec = jax.ShapeDtypeStruct(
      shape=(84, 84, 4, batch_size), dtype=jnp.uint8
  )
  return {
      'observation': obs_spec,
      'next_observation': obs_spec,
      'action': jax.ShapeDtypeStruct(
          shape=(
              batch_size,
              num_actions,
          ),
          dtype=jnp.int32,
      ),
      'terminal': jax.ShapeDtypeStruct(shape=(batch_size, 1), dtype=jnp.int32),
  }


def create_environment(game):
  return atari_lib.create_atari_environment(game_name=game)


def get_num_actions(game):
  return create_environment(game).action_space.n
