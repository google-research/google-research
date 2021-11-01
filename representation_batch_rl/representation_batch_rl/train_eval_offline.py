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

r"""Run training loop for batch rl.

"""
import os
import re

from absl import app
from absl import flags
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_eager_policy
import tqdm

from representation_batch_rl.batch_rl import bcq as bcq_state
from representation_batch_rl.batch_rl import cql as cql_state
from representation_batch_rl.batch_rl import evaluation
from representation_batch_rl.data.dataset import load_tfrecord_dataset_sequence
from representation_batch_rl.gym.wrappers import procgen_wrappers
from representation_batch_rl.representation_batch_rl import bcq_pixels as bcq_pixel
from representation_batch_rl.representation_batch_rl import behavioral_cloning_pixels as bc_pixel
from representation_batch_rl.representation_batch_rl import cql_pixels as cql_pixel
from representation_batch_rl.representation_batch_rl import cssc_pixels as cssc_pixel
from representation_batch_rl.representation_batch_rl import deepmdp_pixels as deepmdp_pixel
from representation_batch_rl.representation_batch_rl import fisher_brac_pixels as fisher_brac_pixel
from representation_batch_rl.representation_batch_rl import ours
from representation_batch_rl.representation_batch_rl import pse_pixels as pse_pixel
from representation_batch_rl.representation_batch_rl import tf_utils
from representation_batch_rl.representation_batch_rl import vpn_pixels as vpn_pixel
from representation_batch_rl.twin_sac import utils


PROCGEN_ENVS = [
    'bigfish', 'bossfight', 'caveflyer', 'chaser', 'climber', 'coinrun',
    'dodgeball', 'fruitbot', 'heist', 'jumper', 'leaper', 'maze', 'miner',
    'ninja', 'plunder', 'starpilot'
]

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'pixels-dm-walker-walk', 'Env name.')
flags.DEFINE_string('task_name', 'halfcheetah-expert-v0', 'Env name.')
flags.DEFINE_enum('algo_name', 'bcq', [
    'bcq', 'cql', 'fbrac', 'ours', 'bc', 'deepmdp', 'vpn', 'cssc', 'pse'
], 'Algorithm.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('action_repeat', 8,
                     '(optional) action repeat used when instantiating env.')
flags.DEFINE_integer('frame_stack', 3,
                     '(optional) frame stack used when instantiating env.')
flags.DEFINE_integer('max_timesteps', 100_000,
                     'Size of dataset to load (typically 100k).')
flags.DEFINE_integer('ckpt_timesteps', 100_000,
                     'Checkpoint timesteps to load dataset from.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('num_updates', 100_000, 'Num updates.')
flags.DEFINE_integer('num_eval_episodes', 10, 'Num eval episodes.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('save_interval', 10_000, 'Model save interval.')
flags.DEFINE_integer('eval_interval', 1000, 'Evaluation interval.')
flags.DEFINE_string('save_dir', '/tmp/save/', 'Saving directory.')
flags.DEFINE_boolean('eager', False, 'Execute functions eagerly.')
flags.DEFINE_float('f_reg', 0.1, 'BR regularization.')
flags.DEFINE_float('reward_bonus', 5.0, 'CQL style reward bonus.')
flags.DEFINE_boolean(
    'numpy_dataset', False,
    'If true, saves and loads the data into/from NumPy arrays with shards')
flags.DEFINE_integer('num_data_augs', 0,
                     'Number of DrQ-style data augs in pixel space.')
flags.DEFINE_enum(
    'obs_type', 'pixels', ['pixels', 'state'],
    'Type of observations to write in the dataset (`state` or `pixels`)')
flags.DEFINE_boolean('load_bc', False,
                     ('Whether to pre-load the online policy'
                      ' corresponding to the dataset, or train from scratch.'))
flags.DEFINE_integer('dataset_size', 100_000, 'Num transitions to train on.')
flags.DEFINE_string('rep_learn_keywords', 'CQL',
                    'Representation learning method list')
flags.DEFINE_integer('n_step_returns', 1, 'N-step returns.')
flags.DEFINE_integer('pretrain', 200_000,
                     ('Pretrain our algorithm using contrastive loss.'
                      ' If `>0`, splits pre-training and FQI. '
                      'If `0`, optimize encoder and FQI losses jointly'))
flags.DEFINE_integer('n_quantiles', 5, 'Number of GVF quantiles.')
flags.DEFINE_float('temp', 0.1, 'NCE softmax temperature.')


def main(_):
  tf.config.experimental_run_functions_eagerly(FLAGS.eager)

  print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
  if FLAGS.env_name.startswith('procgen'):
    print('Test env: %s' % FLAGS.env_name)
    _, env_name, train_levels, _ = FLAGS.env_name.split('-')
    print('Train env: %s' % FLAGS.env_name)
    env = tf_py_environment.TFPyEnvironment(
        procgen_wrappers.TFAgentsParallelProcGenEnv(
            1,
            normalize_rewards=False,  # no normalization for evaluation
            env_name=env_name,
            num_levels=int(train_levels),
            start_level=0))
    env_all = tf_py_environment.TFPyEnvironment(
        procgen_wrappers.TFAgentsParallelProcGenEnv(
            1,
            normalize_rewards=False,  # no normalization for evaluation
            env_name=env_name,
            num_levels=0,
            start_level=0))

    if int(train_levels) == 0:
      train_levels = '200'

  elif FLAGS.env_name.startswith('pixels-dm'):
    if 'distractor' in FLAGS.env_name:
      _, _, domain_name, _, _ = FLAGS.env_name.split('-')
    else:
      _, _, domain_name, _ = FLAGS.env_name.split('-')

    if domain_name in ['cartpole']:
      FLAGS.set_default('action_repeat', 8)
    elif domain_name in ['reacher', 'cheetah', 'ball_in_cup', 'hopper']:
      FLAGS.set_default('action_repeat', 4)
    elif domain_name in ['finger', 'walker']:
      FLAGS.set_default('action_repeat', 2)

    env, _ = utils.load_env(FLAGS.env_name, FLAGS.seed, FLAGS.action_repeat,
                            FLAGS.frame_stack, FLAGS.obs_type)

    if FLAGS.obs_type == 'pixels':
      env, _ = utils.load_env(FLAGS.env_name, FLAGS.seed, FLAGS.action_repeat,
                              FLAGS.frame_stack, FLAGS.obs_type)
    else:
      _, env = utils.load_env(FLAGS.env_name, FLAGS.seed, FLAGS.action_repeat,
                              FLAGS.frame_stack, FLAGS.obs_type)

  if FLAGS.obs_type != 'state':
    if FLAGS.env_name.startswith('procgen'):
      bcq = bcq_pixel
      cql = cql_pixel
      fisher_brac = fisher_brac_pixel
      deepmdp = deepmdp_pixel
      vpn = vpn_pixel
      cssc = cssc_pixel
      pse = pse_pixel
  else:
    bcq = bcq_state
    cql = cql_state
  print('Loading dataset')

  # Use load_tfrecord_dataset_sequence to load transitions of size k>=2.
  if FLAGS.numpy_dataset:
    n_shards = 10
    def shard_fn(shard):
      return (
          'experiments/'
          '20210617_0105.dataset_dmc_50k,100k,'
          '200k_SAC_pixel_numpy/datasets/'
          '%s__%d__%d__%d.npy' %
          (FLAGS.env_name, FLAGS.ckpt_timesteps, FLAGS.max_timesteps, shard))

    np_observer = tf_utils.NumpyObserver(shard_fn, env)
    dataset = np_observer.load(n_shards)
  else:
    if FLAGS.env_name.startswith('procgen'):
      if FLAGS.n_step_returns > 0:
        if FLAGS.max_timesteps == 100_000:
          dataset_path = ('experiments/'
                          '20210624_2033.dataset_procgen__ppo_pixel/'
                          'datasets/%s__%d__%d.tfrecord'%(FLAGS.env_name,
                                                          FLAGS.ckpt_timesteps,
                                                          FLAGS.max_timesteps))
        elif FLAGS.max_timesteps == 3_000_000:
          if int(train_levels) == 1:
            print('Using dataset with 1 level')
            dataset_path = (
                'experiments/'
                '20210713_1557.dataset_procgen__ppo_pixel_1_level/'
                'datasets/%s__%d__%d.tfrecord' %
                (FLAGS.env_name, FLAGS.ckpt_timesteps, FLAGS.max_timesteps))
          elif int(train_levels) == 200:
            print('Using dataset with 200 levels')
            # Mixture dataset between 10M,15M,20M and 25M in equal amounts
            # dataset_path = 'experiments/
            # 20210718_1522.dataset_procgen__ppo_pixel_mixture10,15,20,25M/
            # datasets/%s__%d__%d.tfrecord'%(FLAGS.env_name,
            # FLAGS.ckpt_timesteps,FLAGS.max_timesteps)
            # PPO after 25M steps
            dataset_path = (
                'experiments/'
                '20210702_2234.dataset_procgen__ppo_pixel/'
                'datasets/%s__%d__%d.tfrecord' %
                (FLAGS.env_name, FLAGS.ckpt_timesteps, FLAGS.max_timesteps))
        elif FLAGS.max_timesteps == 5_000_000:
          # epsilon-greedy, eps: 0.1->0.001
          dataset_path = ('experiments/'
                          '20210805_1958.dataset_procgen__ppo_pixel_'
                          'egreedy_levelIDs/datasets/'
                          '%s__%d__%d.tfrecord*' %
                          (FLAGS.env_name, FLAGS.ckpt_timesteps, 100000))
          # Pure greedy (epsilon=0)
          # dataset_path = ('experiments/'
          #                 '20210820_1348.dataset_procgen__ppo_pixel_'
          #                 'egreedy_levelIDs/datasets/'
          #                 '%s__%d__%d.tfrecord*' %
          #                 (FLAGS.env_name, FLAGS.ckpt_timesteps, 100000))

    elif FLAGS.env_name.startswith('pixels-dm'):
      if 'distractor' in FLAGS.env_name:
        dataset_path = ('experiments/'
                        '20210623_1749.dataset_dmc__sac_pixel/datasets/'
                        '%s__%d__%d.tfrecord'%(FLAGS.env_name,
                                               FLAGS.ckpt_timesteps,
                                               FLAGS.max_timesteps))
      else:
        if FLAGS.obs_type == 'pixels':
          dataset_path = ('experiments/'
                          '20210612_1644.dataset_dmc_50k,100k,200k_SAC_pixel/'
                          'datasets/%s__%d__%d.tfrecord'%(FLAGS.env_name,
                                                          FLAGS.ckpt_timesteps,
                                                          FLAGS.max_timesteps))
        else:
          dataset_path = ('experiments/'
                          '20210621_1436.dataset_dmc__SAC_pixel/datasets/'
                          '%s__%d__%d.tfrecord'%(FLAGS.env_name,
                                                 FLAGS.ckpt_timesteps,
                                                 FLAGS.max_timesteps))
    shards = tf.io.gfile.glob(dataset_path)
    shards = [s for s in shards if not s.endswith('.spec')]
    print('Found %d shards under path %s' % (len(shards), dataset_path))
    if FLAGS.n_step_returns > 1:
      # Load sequences of length N
      dataset = load_tfrecord_dataset_sequence(
          shards,
          buffer_size_per_shard=FLAGS.dataset_size // len(shards),
          deterministic=False,
          compress_image=True,
          seq_len=FLAGS.n_step_returns)  # spec=data_spec,
      dataset = dataset.take(FLAGS.dataset_size).shuffle(
          buffer_size=FLAGS.batch_size, reshuffle_each_iteration=False).batch(
              FLAGS.batch_size, drop_remainder=True).prefetch(1).repeat()

      dataset_iter = iter(dataset)
    else:
      dataset_iter = tf_utils.create_data_iterator(
          ('experiments/20210805'
           '_1958.dataset_procgen__ppo_pixel_egreedy_'
           'levelIDs/datasets/%s__%d__%d.tfrecord.shard-*-of-*' %
           (FLAGS.env_name, FLAGS.ckpt_timesteps, 100000)),
          FLAGS.batch_size,
          shuffle_buffer_size=FLAGS.batch_size,
          obs_to_float=False)

  tf.random.set_seed(FLAGS.seed)

  hparam_str = utils.make_hparam_string(
      FLAGS.xm_parameters,
      algo_name=FLAGS.algo_name,
      seed=FLAGS.seed,
      task_name=FLAGS.env_name,
      ckpt_timesteps=FLAGS.ckpt_timesteps,
      rep_learn_keywords=FLAGS.rep_learn_keywords)
  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, 'tb', hparam_str))
  result_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, 'results', hparam_str))

  pretrain = (FLAGS.pretrain > 0)

  if FLAGS.env_name.startswith('procgen'):
    # disable entropy reg for discrete spaces
    action_dim = env.action_spec().maximum.item()+1
  else:
    action_dim = env.action_spec().shape[0]
  if 'cql' in FLAGS.algo_name:
    model = cql.CQL(
        env.observation_spec(),
        env.action_spec(),
        reg=FLAGS.f_reg,
        target_entropy=-action_dim,
        num_augmentations=FLAGS.num_data_augs,
        rep_learn_keywords=FLAGS.rep_learn_keywords,
        batch_size=FLAGS.batch_size)
  elif 'bcq' in FLAGS.algo_name:
    model = bcq.BCQ(
        env.observation_spec(),
        env.action_spec(),
        num_augmentations=FLAGS.num_data_augs)
  elif 'fbrac' in FLAGS.algo_name:
    model = fisher_brac.FBRAC(
        env.observation_spec(),
        env.action_spec(),
        target_entropy=-action_dim,
        f_reg=FLAGS.f_reg,
        reward_bonus=FLAGS.reward_bonus,
        num_augmentations=FLAGS.num_data_augs,
        env_name=FLAGS.env_name,
        batch_size=FLAGS.batch_size)
  elif 'ours' in FLAGS.algo_name:
    model = ours.OURS(
        env.observation_spec(),
        env.action_spec(),
        target_entropy=-action_dim,
        f_reg=FLAGS.f_reg,
        reward_bonus=FLAGS.reward_bonus,
        num_augmentations=FLAGS.num_data_augs,
        env_name=FLAGS.env_name,
        rep_learn_keywords=FLAGS.rep_learn_keywords,
        batch_size=FLAGS.batch_size,
        n_quantiles=FLAGS.n_quantiles,
        temp=FLAGS.temp,
        num_training_levels=train_levels)
    bc_pretraining_steps = FLAGS.pretrain
    if pretrain:
      model_save_path = os.path.join(FLAGS.save_dir, 'weights', hparam_str)
      checkpoint = tf.train.Checkpoint(**model.model_dict)
      tf_step_counter = tf.Variable(0, dtype=tf.int32)
      manager = tf.train.CheckpointManager(
          checkpoint,
          directory=model_save_path,
          max_to_keep=1,
          checkpoint_interval=FLAGS.save_interval,
          step_counter=tf_step_counter)

      # Load the checkpoint in case it exists
      state = manager.restore_or_initialize()
      if state is not None:
        # loaded variables from checkpoint folder
        timesteps_already_done = int(re.findall(
            'ckpt-([0-9]*)', state)[0])  #* FLAGS.save_interval
        print('Loaded model from timestep %d' % timesteps_already_done)
      else:
        print('Training from scratch')
        timesteps_already_done = 0

      tf_step_counter.assign(timesteps_already_done)

      print('Pretraining')
      for i in tqdm.tqdm(range(bc_pretraining_steps)):
        info_dict = model.update_step(
            dataset_iter, train_target='encoder')
        # (quantile_states, quantile_bins)
        if i % FLAGS.log_interval == 0:
          with summary_writer.as_default():
            for k, v in info_dict.items():
              v = tf.reduce_mean(v)
              tf.summary.scalar(f'pretrain/{k}', v, step=i)

        tf_step_counter.assign(i)
        manager.save(checkpoint_number=i)
  elif 'bc' in FLAGS.algo_name:
    model = bc_pixel.BehavioralCloning(
        env.observation_spec(),
        env.action_spec(),
        mixture=False,
        encoder=None,
        num_augmentations=FLAGS.num_data_augs,
        rep_learn_keywords=FLAGS.rep_learn_keywords,
        env_name=FLAGS.env_name,
        batch_size=FLAGS.batch_size)
  elif 'deepmdp' in FLAGS.algo_name:
    model = deepmdp.DeepMdpLearner(
        env.observation_spec(),
        env.action_spec(),
        embedding_dim=512,
        num_distributions=1,
        sequence_length=2,
        learning_rate=3e-4,
        num_augmentations=FLAGS.num_data_augs,
        rep_learn_keywords=FLAGS.rep_learn_keywords,
        batch_size=FLAGS.batch_size)
  elif 'vpn' in FLAGS.algo_name:
    model = vpn.ValuePredictionNetworkLearner(
        env.observation_spec(),
        env.action_spec(),
        embedding_dim=512,
        learning_rate=3e-4,
        num_augmentations=FLAGS.num_data_augs,
        rep_learn_keywords=FLAGS.rep_learn_keywords,
        batch_size=FLAGS.batch_size)
  elif 'cssc' in FLAGS.algo_name:
    model = cssc.CSSC(
        env.observation_spec(),
        env.action_spec(),
        embedding_dim=512,
        actor_lr=3e-4,
        critic_lr=3e-4,
        num_augmentations=FLAGS.num_data_augs,
        rep_learn_keywords=FLAGS.rep_learn_keywords,
        batch_size=FLAGS.batch_size)
  elif 'pse' in FLAGS.algo_name:
    model = pse.PSE(
        env.observation_spec(),
        env.action_spec(),
        embedding_dim=512,
        actor_lr=3e-4,
        critic_lr=3e-4,
        num_augmentations=FLAGS.num_data_augs,
        rep_learn_keywords=FLAGS.rep_learn_keywords,
        batch_size=FLAGS.batch_size,
        temperature=FLAGS.temp)
    bc_pretraining_steps = FLAGS.pretrain
    if pretrain:
      print('Pretraining')
      for i in tqdm.tqdm(range(bc_pretraining_steps)):
        info_dict = model.update_step(
            dataset_iter, train_target='encoder')
        if i % FLAGS.log_interval == 0:
          with summary_writer.as_default():
            for k, v in info_dict.items():
              v = tf.reduce_mean(v)
              tf.summary.scalar(f'pretrain/{k}', v, step=i)

  if 'fbrac' in FLAGS.algo_name or FLAGS.algo_name == 'bc':
    # Either load the online policy:
    if FLAGS.load_bc and FLAGS.env_name.startswith('procgen'):
      env_id = [i for i, name in enumerate(PROCGEN_ENVS) if name == env_name
               ][0] + 1  # map env string to digit [1,16]
      if FLAGS.ckpt_timesteps == 10_000_000:
        ckpt_iter = '0000020480'
      elif FLAGS.ckpt_timesteps == 25_000_000:
        ckpt_iter = '0000051200'
      policy_weights_dir = ('ppo_darts/'
                            '2021-06-22-16-36-54/%d/policies/checkpoints/'
                            'policy_checkpoint_%s/' % (env_id, ckpt_iter))
      policy_def_dir = ('ppo_darts/'
                        '2021-06-22-16-36-54/%d/policies/policy/' % (env_id))
      bc = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
          policy_def_dir,
          time_step_spec=env._time_step_spec,  # pylint: disable=protected-access
          action_spec=env._action_spec,  # pylint: disable=protected-access
          policy_state_spec=env._observation_spec,  # pylint: disable=protected-access
          info_spec=tf.TensorSpec(shape=(None,)),
          load_specs_from_pbtxt=False)
      bc.update_from_checkpoint(policy_weights_dir)
      model.bc.policy = tf_utils.TfAgentsPolicy(bc)
    else:
      if FLAGS.algo_name == 'fbrac':
        bc_pretraining_steps = 100_000
      elif FLAGS.algo_name == 'bc':
        bc_pretraining_steps = 1_000_000

      if 'fbrac' in FLAGS.algo_name:
        bc = model.bc
      else:
        bc = model
      for i in tqdm.tqdm(range(bc_pretraining_steps)):

        info_dict = bc.update_step(dataset_iter)
        if i % FLAGS.log_interval == 0:
          with summary_writer.as_default():
            for k, v in info_dict.items():
              v = tf.reduce_mean(v)
              tf.summary.scalar(f'bc/{k}', v, step=i)

        if FLAGS.algo_name == 'bc':
          if (i + 1) % FLAGS.eval_interval == 0:
            average_returns, average_length = evaluation.evaluate(
                env,
                bc)  # (FLAGS.env_name.startswith('procgen'))
            average_returns_all, average_length_all = evaluation.evaluate(
                env_all,
                bc)

            with result_writer.as_default():
              tf.summary.scalar(
                  'evaluation/returns', average_returns, step=i + 1)
              tf.summary.scalar('evaluation/length', average_length, step=i + 1)
              tf.summary.scalar(
                  'evaluation/returns-all', average_returns_all, step=i + 1)
              tf.summary.scalar(
                  'evaluation/length-all', average_length_all, step=i + 1)

  if FLAGS.algo_name == 'bc':
    exit()

  if not (FLAGS.algo_name == 'ours' and pretrain):
    model_save_path = os.path.join(FLAGS.save_dir, 'weights', hparam_str)
    checkpoint = tf.train.Checkpoint(**model.model_dict)
    tf_step_counter = tf.Variable(0, dtype=tf.int32)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=model_save_path,
        max_to_keep=1,
        checkpoint_interval=FLAGS.save_interval,
        step_counter=tf_step_counter)

    # Load the checkpoint in case it exists
    weights_path = tf.io.gfile.glob(model_save_path + '/ckpt-*.index')
    key_fn = lambda x: int(re.findall(r'(\d+)', x)[-1])
    weights_path.sort(key=key_fn)
    if weights_path:
      weights_path = weights_path[-1]  # take most recent
    state = manager.restore_or_initialize()  # restore(weights_path)
    if state is not None:
      # loaded variables from checkpoint folder
      timesteps_already_done = int(re.findall('ckpt-([0-9]*)',
                                              state)[0])  #* FLAGS.save_interval
      print('Loaded model from timestep %d' % timesteps_already_done)
    else:
      print('Training from scratch')
      timesteps_already_done = 0

  tf_step_counter.assign(timesteps_already_done)

  for i in tqdm.tqdm(range(timesteps_already_done, FLAGS.num_updates)):
    with summary_writer.as_default():
      info_dict = model.update_step(
          dataset_iter,
          train_target='rl' if pretrain else 'both')
    if i % FLAGS.log_interval == 0:
      with summary_writer.as_default():
        for k, v in info_dict.items():
          v = tf.reduce_mean(v)
          tf.summary.scalar(f'training/{k}', v, step=i)

    if (i + 1) % FLAGS.eval_interval == 0:
      average_returns, average_length = evaluation.evaluate(
          env,
          model)
      average_returns_all, average_length_all = evaluation.evaluate(
          env_all,
          model)

      with result_writer.as_default():
        tf.summary.scalar('evaluation/returns-200', average_returns, step=i + 1)
        tf.summary.scalar('evaluation/length-200', average_length, step=i + 1)
        tf.summary.scalar(
            'evaluation/returns-all', average_returns_all, step=i + 1)
        tf.summary.scalar(
            'evaluation/length-all', average_length_all, step=i + 1)

    tf_step_counter.assign(i)
    manager.save(checkpoint_number=i)

if __name__ == '__main__':
  app.run(main)
