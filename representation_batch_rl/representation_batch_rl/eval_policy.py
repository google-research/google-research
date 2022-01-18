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

# python3
# pylint: disable=line-too-long
r"""Run training loop.

"""
# pylint: enable=line-too-long

import os

from absl import app
from absl import flags
import tensorflow as tf
from tf_agents import trajectories
from tf_agents.policies import py_tf_eager_policy

from representation_batch_rl.batch_rl import asac
from representation_batch_rl.batch_rl import awr
from representation_batch_rl.batch_rl import ddpg
from representation_batch_rl.batch_rl import evaluation
from representation_batch_rl.batch_rl import pcl
from representation_batch_rl.batch_rl import sac
from representation_batch_rl.batch_rl import sac_v1
from representation_batch_rl.gym.wrappers import procgen_wrappers
from representation_batch_rl.twin_sac import utils

PROCGEN_ENVS = [
    'bigfish', 'bossfight', 'caveflyer', 'chaser', 'climber', 'coinrun',
    'dodgeball', 'fruitbot', 'heist', 'jumper', 'leaper', 'maze', 'miner',
    'ninja', 'plunder', 'starpilot'
]

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'pixels-dm-cartpole-swingup',
                    'Environment for training/evaluation.')
flags.DEFINE_integer('seed', 42, 'Fixed random seed for training.')
flags.DEFINE_float('actor_lr', 3e-4, 'Actor learning rate.')
flags.DEFINE_float('alpha_lr', 3e-4, 'Temperature learning rate.')
flags.DEFINE_float('critic_lr', 3e-4, 'Critic learning rate.')
flags.DEFINE_integer('deployment_batch_size', 1, 'Batch size.')
flags.DEFINE_integer('sample_batch_size', 256, 'Batch size.')
flags.DEFINE_float('discount', 0.99, 'Discount used for returns.')
flags.DEFINE_float('tau', 0.005,
                   'Soft update coefficient for the target network.')
flags.DEFINE_integer('max_timesteps', 100_000, 'Max timesteps to train.')
flags.DEFINE_integer('ckpt_timesteps', 25_000_000,
                     'Checkpoint timesteps to load policy from.')
flags.DEFINE_string('save_dir', '/tmp/save/', 'Directory to save results to.')
flags.DEFINE_integer('log_interval', 1_000, 'Log every N timesteps.')
flags.DEFINE_integer('eval_interval', 10_000, 'Evaluate every N timesteps.')
flags.DEFINE_integer('action_repeat', 8,
                     '(optional) action repeat used when instantiating env.')
flags.DEFINE_integer('frame_stack', 3,
                     '(optional) frame stack used when instantiating env.')
flags.DEFINE_boolean(
    'numpy_dataset', False,
    'If true, saves and loads the data into/from NumPy arrays with shards')
flags.DEFINE_enum('algo_name', 'sac', [
    'ddpg',
    'crossnorm_ddpg',
    'sac',
    'pc_sac',
    'pcl',
    'crossnorm_sac',
    'crr',
    'awr',
    'sac_v1',
    'asac',
], 'Algorithm.')
flags.DEFINE_boolean('eager', False, 'Execute functions eagerly.')
flags.DEFINE_enum(
    'obs_type', 'pixels', ['pixels', 'state'],
    'Type of observations to write in the dataset (`state` or `pixels`)')


class TfAgentsPolicy():
  """Wrapper to  allow PPO policy rollouts in current codebase format.
  """

  def __init__(self, policy):
    self.policy = policy

  def act(self, states):
    """Act from states.

    Args:
      states: batch of states

    Returns:
      actions
    """
    ts = trajectories.TimeStep(trajectories.StepType.MID,
                               tf.constant(1, dtype=tf.float32),
                               tf.constant(1, dtype=tf.float32),
                               tf.cast(states[0], tf.float32))
    ts2 = trajectories.TimeStep(
        tf.expand_dims(trajectories.StepType.MID, 0),
        tf.constant([1], dtype=tf.float32), tf.constant([1], dtype=tf.float32),
        tf.cast(states, tf.float32))

    act_d = self.policy.action(ts)
    action = tf.constant(act_d.action.item())
    log_prob = self.policy._policy.distribution(ts2).action.log_prob(  # pylint: disable=protected-access
        act_d.action)
    return action, log_prob


def main(_):
  if FLAGS.eager:
    tf.config.experimental_run_functions_eagerly(FLAGS.eager)

  tf.random.set_seed(FLAGS.seed)
  # np.random.seed(FLAGS.seed)
  # random.seed(FLAGS.seed)

  if 'procgen' in FLAGS.env_name:
    _, env_name, train_levels, _ = FLAGS.env_name.split('-')
    env = procgen_wrappers.TFAgentsParallelProcGenEnv(
        1,
        normalize_rewards=False,
        env_name=env_name,
        num_levels=int(train_levels),
        start_level=0)

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

  hparam_str = utils.make_hparam_string(
      FLAGS.xm_parameters,
      algo_name=FLAGS.algo_name,
      seed=FLAGS.seed,
      task_name=FLAGS.env_name,
      ckpt_timesteps=FLAGS.ckpt_timesteps)
  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, 'tb', hparam_str))

  if FLAGS.env_name.startswith('procgen'):
    # map env string to digit [1,16]
    env_id = [i for i, name in enumerate(PROCGEN_ENVS) if name == env_name
             ][0] + 1
    if FLAGS.ckpt_timesteps == 10_000_000:
      ckpt_iter = '0000020480'
    elif FLAGS.ckpt_timesteps == 25_000_000:
      ckpt_iter = '0000051200'
    policy_weights_dir = ('ppo_darts/'
                          '2021-06-22-16-36-54/%d/policies/checkpoints/'
                          'policy_checkpoint_%s/' % (env_id, ckpt_iter))
    policy_def_dir = ('ppo_darts/'
                      '2021-06-22-16-36-54/%d/policies/policy/' % (env_id))
    model = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        policy_def_dir,
        time_step_spec=env._time_step_spec,  # pylint: disable=protected-access
        action_spec=env._action_spec,  # pylint: disable=protected-access
        policy_state_spec=env._observation_spec,  # pylint: disable=protected-access
        info_spec=tf.TensorSpec(shape=(None,)),
        load_specs_from_pbtxt=False)
    model.update_from_checkpoint(policy_weights_dir)
    model = TfAgentsPolicy(model)
  else:
    if 'ddpg' in FLAGS.algo_name:
      model = ddpg.DDPG(
          env.observation_spec(),
          env.action_spec(),
          cross_norm='crossnorm' in FLAGS.algo_name)
    elif 'crr' in FLAGS.algo_name:
      model = awr.AWR(env.observation_spec(), env.action_spec(), f='bin_max')
    elif 'awr' in FLAGS.algo_name:
      model = awr.AWR(env.observation_spec(), env.action_spec(), f='exp_mean')
    elif 'sac_v1' in FLAGS.algo_name:
      model = sac_v1.SAC(
          env.observation_spec(),
          env.action_spec(),
          target_entropy=-env.action_spec().shape[0])
    elif 'asac' in FLAGS.algo_name:
      model = asac.ASAC(
          env.observation_spec(),
          env.action_spec(),
          target_entropy=-env.action_spec().shape[0])
    elif 'sac' in FLAGS.algo_name:
      model = sac.SAC(
          env.observation_spec(),
          env.action_spec(),
          target_entropy=-env.action_spec().shape[0],
          cross_norm='crossnorm' in FLAGS.algo_name,
          pcl_actor_update='pc' in FLAGS.algo_name)
    elif 'pcl' in FLAGS.algo_name:
      model = pcl.PCL(
          env.observation_spec(),
          env.action_spec(),
          target_entropy=-env.action_spec().shape[0])
    if 'distractor' in FLAGS.env_name:
      ckpt_path = os.path.join(
          ('experiments/20210622_2023.policy_weights_sac'
           '_1M_dmc_distractor_hard_pixel/'), 'results',
          FLAGS.env_name + '__' + str(FLAGS.ckpt_timesteps))
    else:
      ckpt_path = os.path.join(
          ('experiments/20210607_2023.'
           'policy_weights_dmc_1M_SAC_pixel'), 'results',
          FLAGS.env_name + '__' + str(FLAGS.ckpt_timesteps))

    model.load_weights(ckpt_path)
  print('Loaded model weights')

  with summary_writer.as_default():
    env = procgen_wrappers.TFAgentsParallelProcGenEnv(
        1,
        normalize_rewards=False,
        env_name=env_name,
        num_levels=0,
        start_level=0)
    (avg_returns, avg_len) = evaluation.evaluate(
        env,
        model,
        num_episodes=100,
        return_distributions=False)
    tf.summary.scalar('evaluation/returns-all', avg_returns, step=0)
    tf.summary.scalar('evaluation/length-all', avg_len, step=0)
    # tf.summary.histogram(
    #     'evaluation/level_%d_reward' % level, reward_acc, step=0)
    # tf.summary.histogram(
    #     'evaluation/level_%d_return' % level, return_acc, step=0)


if __name__ == '__main__':
  app.run(main)
