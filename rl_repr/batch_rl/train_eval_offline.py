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

"""Run training loop for batch rl."""
import os

import gym
import numpy as np

from absl import app
from absl import flags
import tensorflow as tf
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import tqdm
import time

from rl_repr.batch_rl import behavioral_cloning
from rl_repr.batch_rl import latent_behavioral_cloning
from rl_repr.batch_rl import brac
from rl_repr.batch_rl import d4rl_utils
from rl_repr.batch_rl import evaluation
from rl_repr.batch_rl import sac
from rl_repr.batch_rl import embed
from rl_repr.batch_rl import action_embed


FLAGS = flags.FLAGS

flags.DEFINE_string('task_name', 'halfcheetah-expert-v0', 'Env name.')
flags.DEFINE_string('downstream_task_name', None,
                    'Specify if you want downstream task to be different '
                    'from representation learning task.')
flags.DEFINE_string('downstream_data_name', None,
                    'Specify if you want downstream data to be different '
                    'from representation learning task.')
flags.DEFINE_enum('downstream_mode', 'offline', ['offline', 'online'],
                  'Mode of training for downstream task.')
flags.DEFINE_float('proportion_downstream_data', 0.0,
                   'Proportion of downstream data to include in dataset '
                   'used for representation learning.')
flags.DEFINE_integer('downstream_data_size', None,
                     'Specify if you want downstream offline dataset to be smaller.')
flags.DEFINE_enum(
    'downstream_input_mode', 'embed', [
        'embed', 'ctx', 'state-embed', 'state-ctx', 'embed-ctx',
        'state-embed-ctx'
    ],
    'Input form for training downstream task. Only used when learn_ctx is true')
flags.DEFINE_enum('algo_name', 'brac', ['bc', 'sac', 'brac', 'latent_bc'],
                  'Algorithm.')
flags.DEFINE_boolean('learn_ctx', False, 'Whether to learn context embeddings.')
flags.DEFINE_enum('network', 'default', [
    'default', 'small', 'none'
], 'Whether to use small actor/critic net or no net (linear) for RL agent.')
flags.DEFINE_boolean('finetune', False,
                     'Whether to finetune pretrained embeddings.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('num_updates', 1_000_000, 'Num updates.')
flags.DEFINE_integer('num_eval_episodes', 10, 'Num eval episodes.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10_000, 'Evaluation interval.')
flags.DEFINE_string('save_dir', '/tmp/save/', 'Saving directory.')
flags.DEFINE_boolean('eager', False, 'Execute functions eagerly.')
flags.DEFINE_string('embed_learner', None, 'Algorithm to use for learning state embeddings.')
flags.DEFINE_integer('state_embed_dim', None, 'Optional state embedding.')
flags.DEFINE_integer('state_embed_dists', None,
                     'Optional number of state distributions. If specified, '
                     'the state_embed_dim is split into state_embed_dists '
                     'equal sized components, each of which is a one-hot '
                     'vector. The combination of these one-hot vectors '
                     'composes the full state representation.')
flags.DEFINE_float('state_embed_lr', None,
                   'Learning rate for state representation learning.')
flags.DEFINE_integer('embed_training_window', 2,
                     'Determines N for N-step window used to train state embeddings.')
flags.DEFINE_integer('embed_pretraining_steps', -1,
                     'Number of training steps to train embedding prior to offline RL. '
                     'Set at -1 for training in tandem with offline RL.')
flags.DEFINE_integer('num_random_actions', 10_000,
                     'Fill replay buffer with N random actions when doing online training.')
flags.DEFINE_integer(
    'state_mask_dims', 0, 'Number of state dimensions to mask out to'
    'imitate partially observed envs. -1 is equivalent to all dims.')
flags.DEFINE_enum('state_mask_index', 'fixed', ['fixed', 'random'],
                  'How to mask out state observations.')
flags.DEFINE_string(
    'state_mask_value', 'zero', 'How to mask out state observations.'
    'One of [zero, gaussian, quantize#]')
flags.DEFINE_boolean('state_mask_eval', False,
                     'Whether to apply state mask during eval.')

# ACL configs
flags.DEFINE_boolean('input_actions', True, 'Predict action.')
flags.DEFINE_boolean('input_rewards', True, 'Predict action.')
flags.DEFINE_boolean(
    'predict_actions', False, 'Predict action from embedding.'
    '(policy_decoder_on_embeddings = True) or from transformer'
    'output (policy_decoder_on_embeddings = False)')
flags.DEFINE_boolean('policy_decoder_on_embeddings', False,
                     'Whether to use policy decoder on transformer output.')
flags.DEFINE_boolean(
    'predict_rewards', False, 'Predict reward from embedding.'
    '(reward_decoder_on_embeddings = True) or from transformer'
    'output (reward_decoder_on_embeddings = False)')
flags.DEFINE_boolean('reward_decoder_on_embeddings', False,
                     'Whether to use reward decoder on transformer output.')
flags.DEFINE_boolean('embed_on_input', False,
                     'Whether to pass embedding or raw state to transformer.')
flags.DEFINE_boolean(
    'extra_embedder', False, 'Whether to use an extra embedder on input states.'
    'Set to false in ACL ablations.')
flags.DEFINE_string(
    'positional_encoding_type', 'identity', 'Positional encoding.'
    'One of [None, "identity", "sinusoid"]')
flags.DEFINE_enum('direction', 'backward',
                  ['forward', 'backward', 'bidirectional'],
                  'Direction of prediction in pretraining.')

# TRAIL configs
flags.DEFINE_boolean('finetune_primitive', True,
                     'Whether to finetune primitive policy.')
flags.DEFINE_integer('state_action_embed_dim', None,
                     'Optional state-action embedding.')
flags.DEFINE_integer('state_action_fourier_dim', None,
                     'Optional state-action embedding.')
flags.DEFINE_float('latent_bc_lr', 1e-4, 'Learning rate for latent bc.')
flags.DEFINE_float('latent_bc_lr_decay', None,
                   'Decay learning rate for latent bc.')
flags.DEFINE_string('kl_regularizer', 'uniform',
                    'KL regularization for downstream learning')


def get_ctx_length():
  if not FLAGS.state_embed_dim or not FLAGS.learn_ctx:
    return None

  ctx_length = None
  if FLAGS.embed_learner == 'cpc':
    ctx_length = int(FLAGS.embed_training_window * 0.5)
  elif FLAGS.embed_learner == 'mom_cpc':
    ctx_length = int(FLAGS.embed_training_window * 0.5)
  elif FLAGS.embed_learner == 'bert':
    ctx_length = FLAGS.embed_training_window - 1
  elif FLAGS.embed_learner == 'bert2':
    ctx_length = FLAGS.embed_training_window - 1
  elif FLAGS.embed_learner == 'bert3':
    ctx_length = FLAGS.embed_training_window - 1
  elif FLAGS.embed_learner == 'acl':
    ctx_length = FLAGS.embed_training_window
  elif FLAGS.embed_learner == 'mom_acl':
    ctx_length = FLAGS.embed_training_window
  return ctx_length


def get_embed_model(env):
  if FLAGS.embed_learner == 'action_fourier':
    embed_model = action_embed.ActionFourierLearner(
        env.observation_spec().shape[0],
        env.action_spec(),
        embedding_dim=FLAGS.state_action_embed_dim,
        fourier_dim=FLAGS.state_action_fourier_dim,
        sequence_length=FLAGS.embed_training_window,
        learning_rate=FLAGS.state_embed_lr,
        kl_regularizer=FLAGS.kl_regularizer)
  elif FLAGS.embed_learner in ['action_opal', 'action_skild']:
    embed_model = action_embed.ActionOpalLearner(
        env.observation_spec().shape[0],
        env.action_spec(),
        latent_dim=FLAGS.state_action_embed_dim,
        sequence_length=FLAGS.embed_training_window,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'action_spirl':
    embed_model = action_embed.ActionOpalLearner(
        env.observation_spec().shape[0],
        env.action_spec(),
        latent_dim=FLAGS.state_action_embed_dim,
        sequence_length=FLAGS.embed_training_window,
        action_only=True,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'cpc':
    embed_model = embed.CpcLearner(
        env.observation_spec().shape[0],
        env.action_spec().shape[0],
        embedding_dim=FLAGS.state_embed_dim,
        num_distributions=FLAGS.state_embed_dists,
        sequence_length=FLAGS.embed_training_window,
        ctx_length=get_ctx_length(),
        downstream_input_mode=FLAGS.downstream_input_mode,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'mom_cpc':
    embed_model = embed.MomentumCpcLearner(
        env.observation_spec().shape[0],
        env.action_spec().shape[0],
        embedding_dim=FLAGS.state_embed_dim,
        sequence_length=FLAGS.embed_training_window,
        ctx_length=get_ctx_length(),
        downstream_input_mode=FLAGS.downstream_input_mode,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'hiro':
    embed_model = embed.HiroLearner(
        env.observation_spec().shape[0],
        env.action_spec().shape[0],
        embedding_dim=FLAGS.state_embed_dim,
        sequence_length=FLAGS.embed_training_window,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'opal':
    embed_model = embed.OpalLearner(
        env.observation_spec().shape[0],
        env.action_spec(),
        embedding_dim=FLAGS.state_embed_dim,
        sequence_length=FLAGS.embed_training_window,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'avae':
    embed_model = embed.ActionVaeLearner(
        env.observation_spec().shape[0],
        env.action_spec(),
        embedding_dim=FLAGS.state_embed_dim,
        num_distributions=FLAGS.state_embed_dists,
        sequence_length=FLAGS.embed_training_window,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'bert':
    embed_model = embed.BertLearner(
        env.observation_spec().shape[0],
        env.action_spec().shape[0],
        embedding_dim=FLAGS.state_embed_dim,
        num_distributions=FLAGS.state_embed_dists,
        sequence_length=FLAGS.embed_training_window,
        ctx_length=get_ctx_length(),
        downstream_input_mode=FLAGS.downstream_input_mode,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'bert2':
    embed_model = embed.Bert2Learner(
        env.observation_spec().shape[0],
        env.action_spec().shape[0],
        embedding_dim=FLAGS.state_embed_dim,
        num_distributions=FLAGS.state_embed_dists,
        sequence_length=FLAGS.embed_training_window,
        ctx_length=get_ctx_length(),
        downstream_input_mode=FLAGS.downstream_input_mode,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'bert3':
    embed_model = embed.Bert3Learner(
        env.observation_spec().shape[0],
        env.action_spec(),
        embedding_dim=FLAGS.state_embed_dim,
        num_distributions=FLAGS.state_embed_dists,
        sequence_length=FLAGS.embed_training_window,
        ctx_length=get_ctx_length(),
        downstream_input_mode=FLAGS.downstream_input_mode,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'mom_acl':
    embed_model = embed.MomentumACLLearner(
        env.observation_spec().shape[0],
        env.action_spec(),
        embedding_dim=FLAGS.state_embed_dim,
        num_distributions=FLAGS.state_embed_dists,
        sequence_length=FLAGS.embed_training_window,
        ctx_length=get_ctx_length(),
        downstream_input_mode=FLAGS.downstream_input_mode,
        learning_rate=FLAGS.state_embed_lr,
        input_actions=FLAGS.input_actions,
        predict_actions=FLAGS.predict_actions,
        policy_decoder_on_embeddings=FLAGS.policy_decoder_on_embeddings,
        input_rewards=FLAGS.input_rewards,
        predict_rewards=FLAGS.predict_rewards,
        reward_decoder_on_embeddings=FLAGS.reward_decoder_on_embeddings,
        embed_on_input=FLAGS.embed_on_input,
        extra_embedder=FLAGS.extra_embedder,
        positional_encoding_type=FLAGS.positional_encoding_type,
        direction=FLAGS.direction)
  elif FLAGS.embed_learner == 'acl':
    embed_model = embed.ACLLearner(
        env.observation_spec().shape[0],
        env.action_spec(),
        embedding_dim=FLAGS.state_embed_dim,
        num_distributions=FLAGS.state_embed_dists,
        sequence_length=FLAGS.embed_training_window,
        ctx_length=get_ctx_length(),
        downstream_input_mode=FLAGS.downstream_input_mode,
        learning_rate=FLAGS.state_embed_lr,
        input_actions=FLAGS.input_actions,
        predict_actions=FLAGS.predict_actions,
        policy_decoder_on_embeddings=FLAGS.policy_decoder_on_embeddings,
        input_rewards=FLAGS.input_rewards,
        predict_rewards=FLAGS.predict_rewards,
        reward_decoder_on_embeddings=FLAGS.reward_decoder_on_embeddings,
        embed_on_input=FLAGS.embed_on_input,
        extra_embedder=FLAGS.extra_embedder,
        positional_encoding_type=FLAGS.positional_encoding_type,
        direction=FLAGS.direction)
  elif FLAGS.embed_learner == 'vpn':
    embed_model = embed.VpnLearner(
        env.observation_spec().shape[0],
        env.action_spec().shape[0],
        embedding_dim=FLAGS.state_embed_dim,
        sequence_length=FLAGS.embed_training_window,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'dreamer':
    embed_model = embed.DreamerV2Learner(
        env.observation_spec().shape[0],
        env.action_spec().shape[0],
        embedding_dim=FLAGS.state_embed_dim,
        num_distributions=FLAGS.state_embed_dists,
        sequence_length=FLAGS.embed_training_window,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'diverse':
    embed_model = embed.DiversePolicyLearner(
        env.observation_spec().shape[0],
        env.action_spec(),
        embedding_dim=FLAGS.state_embed_dim,
        num_distributions=FLAGS.state_embed_dists,
        sequence_length=FLAGS.embed_training_window,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'super':
    embed_model = embed.SuperModelLearner(
        env.observation_spec().shape[0],
        env.action_spec(),
        embedding_dim=FLAGS.state_embed_dim,
        num_distributions=FLAGS.state_embed_dists,
        sequence_length=FLAGS.embed_training_window,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'deepmdp':
    embed_model = embed.DeepMdpLearner(
        env.observation_spec().shape[0],
        env.action_spec(),
        embedding_dim=FLAGS.state_embed_dim,
        num_distributions=FLAGS.state_embed_dists,
        sequence_length=FLAGS.embed_training_window,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'forward':
    embed_model = embed.ForwardModelLearner(
        env.observation_spec().shape[0],
        env.action_spec(),
        embedding_dim=FLAGS.state_embed_dim,
        num_distributions=FLAGS.state_embed_dists,
        sequence_length=FLAGS.embed_training_window,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'inverse':
    embed_model = embed.InverseModelLearner(
        env.observation_spec().shape[0],
        env.action_spec(),
        embedding_dim=FLAGS.state_embed_dim,
        num_distributions=FLAGS.state_embed_dists,
        sequence_length=FLAGS.embed_training_window,
        learning_rate=FLAGS.state_embed_lr)
  elif FLAGS.embed_learner == 'bisim':
    embed_model = embed.BisimulationLearner(
        env.observation_spec().shape[0],
        env.action_spec(),
        embedding_dim=FLAGS.state_embed_dim,
        num_distributions=FLAGS.state_embed_dists,
        sequence_length=FLAGS.embed_training_window,
        learning_rate=FLAGS.state_embed_lr)
  else:
    raise ValueError('Unknown embed learner %s.' % FLAGS.embed_learner)

  return embed_model


def main(_):
  tf.config.experimental_run_functions_eagerly(FLAGS.eager)

  def preprocess_fn(dataset):
    return dataset.cache().shuffle(1_000_000, reshuffle_each_iteration=True)

  def state_mask_fn(states):
    if FLAGS.state_mask_dims == 0:
      return states
    assert FLAGS.state_mask_dims <= states.shape[1]
    state_mask_dims = (
        states.shape[1]
        if FLAGS.state_mask_dims == -1 else FLAGS.state_mask_dims)
    if FLAGS.state_mask_index == 'fixed':
      mask_indices = range(states.shape[1] - state_mask_dims, states.shape[1])
    else:
      mask_indices = np.random.permutation(np.arange(
          states.shape[1]))[:state_mask_dims]
    if FLAGS.state_mask_value == 'gaussian':
      mask_values = states[:, mask_indices]
      mask_values = (
          mask_values + np.std(mask_values, axis=0) *
          np.random.normal(size=mask_values.shape))
    elif 'quantize' in FLAGS.state_mask_value:
      mask_values = states[:, mask_indices]
      mask_values = np.around(
          mask_values, decimals=int(FLAGS.state_mask_value[-1]))
    else:
      mask_values = 0
    states[:, mask_indices] = mask_values
    return states

  gym_env, dataset, embed_dataset = d4rl_utils.create_d4rl_env_and_dataset(
      task_name=FLAGS.task_name,
      batch_size=FLAGS.batch_size,
      sliding_window=FLAGS.embed_training_window,
      state_mask_fn=state_mask_fn)

  downstream_embed_dataset = None
  if (FLAGS.downstream_task_name is not None or
      FLAGS.downstream_data_name is not None or
      FLAGS.downstream_data_size is not None):
    downstream_data_name = FLAGS.downstream_data_name
    assert downstream_data_name is None
    gym_env, dataset, downstream_embed_dataset = d4rl_utils.create_d4rl_env_and_dataset(
        task_name=FLAGS.downstream_task_name,
        batch_size=FLAGS.batch_size,
        sliding_window=FLAGS.embed_training_window,
        data_size=FLAGS.downstream_data_size,
        state_mask_fn=state_mask_fn)

    if FLAGS.proportion_downstream_data:
      zipped_dataset = tf.data.Dataset.zip((embed_dataset, downstream_embed_dataset))

      def combine(*elems1_and_2):
        batch_size = tf.shape(elems1_and_2[0][0])[0]
        which = tf.random.uniform([batch_size]) >= FLAGS.proportion_downstream_data
        from1 = tf.where(which)
        from2 = tf.where(tf.logical_not(which))
        new_elems = map(
            lambda x: tf.concat([tf.gather_nd(x[0], from1), tf.gather_nd(x[1], from2)], 0),
            zip(*elems1_and_2))
        return tuple(new_elems)

      embed_dataset = zipped_dataset.map(combine)

  if FLAGS.embed_learner and 'action' in FLAGS.embed_learner:
    assert FLAGS.embed_training_window >= 2
    dataset = downstream_embed_dataset or embed_dataset

  if FLAGS.downstream_mode == 'online':

    downstream_task = FLAGS.downstream_task_name or FLAGS.task_name
    try:
      train_gym_env = gym.make(downstream_task)
    except:
      train_gym_env = gym.make('DM-' + downstream_task)
    train_env = gym_wrapper.GymWrapper(train_gym_env)

    train_env = tf_py_environment.TFPyEnvironment(train_env)

    replay_spec = (
        train_env.observation_spec(),
        train_env.action_spec(),
        train_env.reward_spec(),
        train_env.reward_spec(),  # discount spec
        train_env.observation_spec(),  # next observation spec
    )
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        replay_spec,
        batch_size=1,
        max_length=FLAGS.num_updates,
        dataset_window_shift=1 if get_ctx_length() else None)

    @tf.function
    def add_to_replay(state, action, reward, discount, next_states,
                      replay_buffer=replay_buffer):
      replay_buffer.add_batch((state, action, reward, discount, next_states))

    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        train_env.time_step_spec(), train_env.action_spec())

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=FLAGS.batch_size,
        num_steps=FLAGS.embed_training_window
        if get_ctx_length() else None).prefetch(3)
    dataset = dataset.map(lambda *data: data[0])
  else:
    train_env = None
    replay_buffer = None
    add_to_replay = None
    initial_collect_policy = None

  env = gym_wrapper.GymWrapper(gym_env)
  env = tf_py_environment.TFPyEnvironment(env)

  dataset_iter = iter(dataset)
  embed_dataset_iter = iter(embed_dataset) if embed_dataset else None

  tf.random.set_seed(FLAGS.seed)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, 'tb'))
  result_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, 'results'))

  if (FLAGS.state_embed_dim or FLAGS.state_action_embed_dim
     ) and FLAGS.embed_learner and FLAGS.embed_pretraining_steps != 0:
    embed_model = get_embed_model(env)
    if FLAGS.finetune:
      other_embed_model = get_embed_model(env)
      other_embed_model2 = get_embed_model(env)
    else:
      other_embed_model = None
      other_embed_model2 = None
  else:
    embed_model = None
    other_embed_model = None
    other_embed_model2 = None

  config_str = f'{FLAGS.task_name}_{FLAGS.embed_learner}_{FLAGS.state_embed_dim}_{FLAGS.state_embed_dists}_{FLAGS.embed_training_window}_{FLAGS.downstream_input_mode}_{FLAGS.finetune}_{FLAGS.network}_{FLAGS.seed}'
  if FLAGS.embed_learner == 'acl':
    config_str += f'_{FLAGS.predict_actions}_{FLAGS.policy_decoder_on_embeddings}_{FLAGS.reward_decoder_on_embeddings}_{FLAGS.predict_rewards}_{FLAGS.embed_on_input}_{FLAGS.extra_embedder}_{FLAGS.positional_encoding_type}_{FLAGS.direction}'
  elif FLAGS.embed_learner and 'action' in FLAGS.embed_learner:
    config_str += f'_{FLAGS.state_action_embed_dim}_{FLAGS.state_action_fourier_dim}'
  save_dir = os.path.join(FLAGS.save_dir, config_str)

  # Embed pretraining
  if FLAGS.embed_pretraining_steps > 0 and embed_model is not None:
    model_folder = os.path.join(
        save_dir, 'embed_models%d' % FLAGS.embed_pretraining_steps,
        config_str)
    if not tf.io.gfile.isdir(model_folder):
      embed_pretraining_steps = FLAGS.embed_pretraining_steps
      for i in tqdm.tqdm(range(embed_pretraining_steps)):
        embed_dict = embed_model.update_step(embed_dataset_iter)
        if i % FLAGS.log_interval == 0:
          with summary_writer.as_default():
            for k, v in embed_dict.items():
              tf.summary.scalar(f'embed/{k}', v, step=i-embed_pretraining_steps)
              print(k, v)
            print('embed pretraining')
      embed_model.save_weights(os.path.join(model_folder, 'embed'))
    else:
      time.sleep(np.random.randint(5, 20))  # Try to suppress checksum errors.
      embed_model.load_weights(os.path.join(model_folder, 'embed'))

    if other_embed_model and other_embed_model2:
      try:  # Try to suppress checksum errors.
        other_embed_model.load_weights(os.path.join(model_folder, 'embed'))
        other_embed_model2.load_weights(os.path.join(model_folder, 'embed'))
      except:
        embed_model.save_weights(os.path.join(model_folder, 'embed'))
        other_embed_model.load_weights(os.path.join(model_folder, 'embed'))
        other_embed_model2.load_weights(os.path.join(model_folder, 'embed'))

  if FLAGS.algo_name == 'bc':
    hidden_dims = ([] if FLAGS.network == 'none' else
                   (256,) if FLAGS.network == 'small' else
                   (256, 256))
    model = behavioral_cloning.BehavioralCloning(
        env.observation_spec().shape[0],
        env.action_spec(),
        hidden_dims=hidden_dims,
        embed_model=embed_model,
        finetune=FLAGS.finetune)
  elif FLAGS.algo_name == 'latent_bc':
    hidden_dims = ([] if FLAGS.network == 'none' else
                   (256,) if FLAGS.network == 'small' else (256, 256))
    model = latent_behavioral_cloning.LatentBehavioralCloning(
        env.observation_spec().shape[0],
        env.action_spec(),
        hidden_dims=hidden_dims,
        embed_model=embed_model,
        finetune=FLAGS.finetune,
        finetune_primitive=FLAGS.finetune_primitive,
        learning_rate=FLAGS.latent_bc_lr,
        latent_bc_lr_decay=FLAGS.latent_bc_lr_decay,
        kl_regularizer=FLAGS.kl_regularizer)
  elif 'sac' in FLAGS.algo_name:
    model = sac.SAC(
        env.observation_spec().shape[0],
        env.action_spec(),
        target_entropy=-env.action_spec().shape[0],
        embed_model=embed_model,
        other_embed_model=other_embed_model,
        network=FLAGS.network,
        finetune=FLAGS.finetune)
  elif 'brac' in FLAGS.algo_name:
    model = brac.BRAC(
        env.observation_spec().shape[0],
        env.action_spec(),
        target_entropy=-env.action_spec().shape[0],
        embed_model=embed_model,
        other_embed_model=other_embed_model,
        bc_embed_model=other_embed_model2,
        network=FLAGS.network,
        finetune=FLAGS.finetune)

    # Agent pretraining.
    if not tf.io.gfile.isdir(os.path.join(save_dir, 'model')):
      bc_pretraining_steps = 200_000
      for i in tqdm.tqdm(range(bc_pretraining_steps)):
        if get_ctx_length():
          info_dict = model.bc.update_step(embed_dataset_iter)
        else:
          info_dict = model.bc.update_step(dataset_iter)

        if i % FLAGS.log_interval == 0:
          with summary_writer.as_default():
            for k, v in info_dict.items():
              tf.summary.scalar(
                  f'training/{k}', v, step=i - bc_pretraining_steps)
            print('bc pretraining')
      model.bc.policy.save_weights(os.path.join(save_dir, 'model'))
    else:
      model.bc.policy.load_weights(os.path.join(save_dir, 'model'))

  if train_env:
    timestep = train_env.reset()
  else:
    timestep = None

  actor = None
  if hasattr(model, 'actor'):
    actor = model.actor
  elif hasattr(model, 'policy'):
    actor = model.policy

  ctx_states = []
  ctx_actions = []
  ctx_rewards = []
  for i in tqdm.tqdm(range(FLAGS.num_updates)):
    if (train_env and timestep and
        replay_buffer and initial_collect_policy and
        add_to_replay and actor):
      if timestep.is_last():
        timestep = train_env.reset()
      if replay_buffer.num_frames() < FLAGS.num_random_actions:
        policy_step = initial_collect_policy.action(timestep)
        action = policy_step.action
        ctx_states.append(state_mask_fn(timestep.observation.numpy()))
        ctx_actions.append(action)
        ctx_rewards.append(timestep.reward)
      else:
        states = state_mask_fn(timestep.observation.numpy())
        actions = None
        rewards = None
        if get_ctx_length():
          ctx_states.append(states)
          states = tf.stack(ctx_states[-get_ctx_length():], axis=1)
          actions = tf.stack(ctx_actions[-get_ctx_length() + 1:], axis=1)
          rewards = tf.stack(ctx_rewards[-get_ctx_length() + 1:], axis=1)
        if hasattr(model, 'embed_model') and model.embed_model:
          states = model.embed_model(states, actions, rewards)
        action = actor(states, sample=True)
        ctx_actions.append(action)
      next_timestep = train_env.step(action)
      ctx_rewards.append(next_timestep.reward)
      add_to_replay(
          state_mask_fn(timestep.observation.numpy()), action,
          next_timestep.reward, next_timestep.discount,
          state_mask_fn(next_timestep.observation.numpy()))
      timestep = next_timestep

    with summary_writer.as_default():
      if embed_model and FLAGS.embed_pretraining_steps == -1:
        embed_dict = embed_model.update_step(embed_dataset_iter)
        if other_embed_model:
          other_embed_dict = other_embed_model.update_step(embed_dataset_iter)
          embed_dict.update(dict(('other_%s' % k, v) for k, v in other_embed_dict.items()))
      else:
        embed_dict = {}

      if FLAGS.downstream_mode == 'offline':
        if get_ctx_length():
          info_dict = model.update_step(embed_dataset_iter)
        else:
          info_dict = model.update_step(dataset_iter)
      elif i + 1 >= FLAGS.num_random_actions:
        info_dict = model.update_step(dataset_iter)
      else:
        info_dict = {}

    if i % FLAGS.log_interval == 0:
      with summary_writer.as_default():
        for k, v in info_dict.items():
          tf.summary.scalar(f'training/{k}', v, step=i)
        for k, v in embed_dict.items():
          tf.summary.scalar(f'embed/{k}', v, step=i)
          print(k, v)

    if (i + 1) % FLAGS.eval_interval == 0:
      average_returns, average_length = evaluation.evaluate(
          env,
          model,
          ctx_length=get_ctx_length(),
          embed_training_window=(FLAGS.embed_training_window
                                 if FLAGS.embed_learner and
                                 'action' in FLAGS.embed_learner else None),
          state_mask_fn=state_mask_fn if FLAGS.state_mask_eval else None)

      average_returns = gym_env.get_normalized_score(average_returns) * 100.0

      with result_writer.as_default():
        tf.summary.scalar('evaluation/returns', average_returns, step=i+1)
        tf.summary.scalar('evaluation/length', average_length, step=i+1)
        print('evaluation/returns', average_returns)
        print('evaluation/length', average_length)

if __name__ == '__main__':
  app.run(main)
