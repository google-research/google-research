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

r"""Script to create dataset for offline RL.

Includes ProcGen, DMC suite (pixel and state, as well as DMC with distractor
features).

"""

import os

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from tf_agents import specs
from tf_agents import trajectories
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import example_encoding_dataset
import tqdm

from representation_batch_rl.batch_rl import asac
from representation_batch_rl.batch_rl import awr
from representation_batch_rl.batch_rl import ddpg
from representation_batch_rl.batch_rl import pcl
from representation_batch_rl.batch_rl import sac
from representation_batch_rl.batch_rl import sac_v1
from representation_batch_rl.gym.wrappers import procgen_wrappers
from representation_batch_rl.representation_batch_rl import tf_utils
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
flags.DEFINE_list(
    'ckpt_timesteps',
    default=None,
    help='Checkpoint timesteps to load policy from.')
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
flags.DEFINE_integer('n_step_returns', 1, 'N-step returns.')
flags.DEFINE_integer('worker_id', 0, 'Worker id for TFRecord shard naming')
flags.DEFINE_integer('total_workers', 128, 'Total number of workers')


class FlatTimestep(trajectories.TimeStep):

  def __init__(self, step_type, reward, discount, observation):
    self.observation = observation[0]
    self.discount = discount[0]
    self.reward = reward[0]
    self.step_type = step_type[0]


class Traj():

  def __init__(self, traj, next_obs=None):
    if next_obs is not None:
      self.observation = np.stack([traj.observation, next_obs])
    else:
      self.observation = traj.observation
    self.action = traj.action
    self.reward = traj.reward
    self.discount = traj.discount


class DummyObserver(example_encoding_dataset.TFRecordObserver):

  def save(self, n_shards):
    pass


def main(_):
  if FLAGS.eager:
    tf.config.experimental_run_functions_eagerly(FLAGS.eager)

  tf.random.set_seed(FLAGS.seed)
  # np.random.seed(FLAGS.seed)
  # random.seed(FLAGS.seed)

  print('Env name: %s'%FLAGS.env_name)

  if 'procgen' in FLAGS.env_name:
    _, env_name, train_levels, _ = FLAGS.env_name.split('-')
    env = procgen_wrappers.TFAgentsParallelProcGenEnv(
        1,
        normalize_rewards=True,
        env_name=env_name,
        num_levels=int(train_levels),
        start_level=0)
    state_env = None

    timestep_spec = trajectories.time_step_spec(
        observation_spec=specs.ArraySpec(env._observation_spec.shape, np.uint8),  # pylint: disable=protected-access
        reward_spec=specs.ArraySpec(shape=(), dtype=np.float32))

    data_spec = trajectory.from_transition(
        timestep_spec,
        policy_step.PolicyStep(
            action=env._action_spec,  # pylint: disable=protected-access
            info=specs.ArraySpec(shape=(), dtype=np.int32)), timestep_spec)

    n_state = None
    # ckpt_steps = [10_000_000,15_000_000,20_000_000,25_000_000]
    ckpt_steps = [25_000_000]
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

    env, state_env = utils.load_env(FLAGS.env_name, FLAGS.seed,
                                    FLAGS.action_repeat, FLAGS.frame_stack,
                                    FLAGS.obs_type)

    if FLAGS.obs_type == 'pixels':
      data_spec = trajectory.from_transition(
          env.time_step_spec(), policy_step.PolicyStep(env.action_spec()),
          env.time_step_spec())
      ckpt_steps = FLAGS.ckpt_timesteps[0]
    else:
      data_spec = trajectory.from_transition(
          state_env.time_step_spec(),
          policy_step.PolicyStep(state_env.action_spec()),
          state_env.time_step_spec())
      n_state = state_env.observation_spec().shape[0]
      ckpt_steps = FLAGS.ckpt_timesteps[0]

  if FLAGS.numpy_dataset:
    tf.io.gfile.makedirs(os.path.join(FLAGS.save_dir, 'datasets'))
    def shard_fn(shard):
      return os.path.join(
          FLAGS.save_dir, 'datasets', FLAGS.env_name + '__%d__%d__%d.npy' %
          (int(ckpt_steps[-1]), FLAGS.max_timesteps, shard))

    observer = tf_utils.NumpyObserver(shard_fn, env)
    observer.allocate_arrays(FLAGS.max_timesteps)
  else:
    shard_fn = os.path.join(
        FLAGS.save_dir, 'datasets',
        FLAGS.env_name + '__%d__%d.tfrecord.shard-%d-of-%d' %
        (int(ckpt_steps[-1]), FLAGS.max_timesteps, FLAGS.worker_id,
         FLAGS.total_workers))
    observer = DummyObserver(
        shard_fn, data_spec, py_mode=True, compress_image=True)

  def load_model(checkpoint):
    checkpoint = int(checkpoint)
    print(checkpoint)
    if FLAGS.env_name.startswith('procgen'):
      env_id = [i for i, name in enumerate(
          PROCGEN_ENVS) if name == env_name][0]+1
      if checkpoint == 10_000_000:
        ckpt_iter = '0000020480'
      elif checkpoint == 15_000_000:
        ckpt_iter = '0000030720'
      elif checkpoint == 20_000_000:
        ckpt_iter = '0000040960'
      elif checkpoint == 25_000_000:
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
      model.actor = model.action
    else:
      if 'ddpg' in FLAGS.algo_name:
        model = ddpg.DDPG(
            env.observation_spec(),
            env.action_spec(),
            cross_norm='crossnorm' in FLAGS.algo_name)
      elif 'crr' in FLAGS.algo_name:
        model = awr.AWR(
            env.observation_spec(),
            env.action_spec(), f='bin_max')
      elif 'awr' in FLAGS.algo_name:
        model = awr.AWR(
            env.observation_spec(),
            env.action_spec(), f='exp_mean')
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
            ('experiments/'
             '20210622_2023.policy_weights_sac_1M_dmc_distractor_hard_pixel/'),
            'results', FLAGS.env_name+'__'+str(checkpoint))
      else:
        ckpt_path = os.path.join(
            ('experiments/'
             '20210607_2023.policy_weights_dmc_1M_SAC_pixel'), 'results',
            FLAGS.env_name + '__' + str(checkpoint))

      model.load_weights(ckpt_path)
    print('Loaded model weights')
    return model

  # previous_time = time.time()
  timestep = env.reset()
  episode_return = 0
  episode_timesteps = 0

  actions = []
  time_steps = []

  def get_state_or_pixels(obs, obs_type):
    # obs of shape 1 x 84 x 84 x (n_state*frame_stack + 3*frame_stack)
    if len(obs.shape) == 4:
      obs = obs[0]
    if obs_type == 'state':
      obs = obs[0, 0, :n_state]
    else:
      obs_tmp = []
      for i in range(FLAGS.frame_stack):
        obs_tmp.append(obs[:, :, (i + 1) * (n_state) +
                           i * 3:((i + 1) * (n_state) + (i + 1) * 3)])
      obs = np.concatenate(obs_tmp, axis=-1)
    return obs

  k_model = 0
  model = load_model(ckpt_steps[k_model])
  reload_model = False

  def linear_scheduling(t):  # pylint: disable=unused-variable
    return 0.1 - 3.96e-9* t

  mixture_freq = FLAGS.max_timesteps // len(ckpt_steps)
  for i in tqdm.tqdm(range(FLAGS.max_timesteps)):
    if (i % mixture_freq) == 0 and i > 0:
      reload_model = True
    if np.all(timestep.is_last()):
      if FLAGS.env_name.startswith('procgen'):
        timestep = trajectories.TimeStep(
            timestep.step_type[0], timestep.reward[0], timestep.discount[0],
            (timestep.observation[0] * 255).astype(np.uint8))

      time_steps.append(
          ts.termination(
              get_state_or_pixels(timestep.observation()[0], 'state')
              if FLAGS.obs_type == 'state' else timestep.observation,
              timestep.reward if timestep.reward is not None else 1.0))
      # Write the episode into the TF Record

      for l in range(len(time_steps) - 1):
        t_ = min(l + FLAGS.n_step_returns, len(time_steps) - 1)
        n_step_return = 0.
        for j in range(l, t_):
          if len(time_steps[j].reward.shape) == 1:
            r_t = time_steps[j].reward[0]
          else:
            r_t = time_steps[j].reward

          n_step_return += FLAGS.discount**j * r_t

        t_ = min(l + 1 + FLAGS.n_step_returns, len(time_steps) - 1)
        n_step_return_tp1 = 0.
        for j in range(l + 1, t_):
          if len(time_steps[j].reward.shape) == 1:
            r_t = time_steps[j].reward[0]
          else:
            r_t = time_steps[j].reward

          n_step_return_tp1 += FLAGS.discount**j * r_t

        if len(time_steps[l].observation.shape) == 4:
          if len(time_steps[l].reward.shape) == 1:
            time_steps[l] = trajectories.TimeStep(time_steps[l].step_type[0],
                                                  n_step_return,
                                                  time_steps[l].discount[0],
                                                  time_steps[l].observation[0])
          else:
            time_steps[l] = trajectories.TimeStep(time_steps[l].step_type,
                                                  n_step_return,
                                                  time_steps[l].discount,
                                                  time_steps[l].observation[0])
        if len(time_steps[l + 1].observation.shape) == 4:
          if len(time_steps[l + 1].reward.shape) == 1:
            time_steps[l + 1] = trajectories.TimeStep(
                time_steps[l + 1].step_type[0], n_step_return_tp1,
                time_steps[l + 1].discount[0], time_steps[l + 1].observation[0])
          else:
            time_steps[l + 1] = trajectories.TimeStep(
                time_steps[l + 1].step_type, n_step_return_tp1,
                time_steps[l + 1].discount, time_steps[l + 1].observation[0])
        traj = trajectory.from_transition(time_steps[l], actions[l],
                                          time_steps[l + 1])
        if FLAGS.numpy_dataset:
          traj = Traj(traj, next_obs=time_steps[l+1].observation)
          observer(traj)
        else:
          observer(traj)

      timestep = env.reset()
      print(episode_return)
      episode_return = 0
      episode_timesteps = 0
      # previous_time = time.time()

      actions = []
      time_steps = []

      if reload_model:
        k_model += 1
        model = load_model(ckpt_steps[k_model])
        reload_model = False
    if FLAGS.env_name.startswith('procgen'):
      timestep = trajectories.TimeStep(
          timestep.step_type[0], timestep.reward[0], timestep.discount[0],
          (timestep.observation[0] * 255).astype(np.uint8))

    if episode_timesteps == 0:
      time_steps.append(
          ts.restart(
              get_state_or_pixels(timestep.observation, 'state') if FLAGS
              .obs_type == 'state' else (timestep.observation)))
    elif not timestep.is_last():
      time_steps.append(
          ts.transition(
              get_state_or_pixels(timestep.observation[0], 'state')
              if FLAGS.obs_type == 'state' else (timestep.observation),
              timestep.reward if timestep.reward is not None else 0.0,
              timestep.discount))

    if FLAGS.env_name.startswith('procgen'):
      # eps_t = linear_scheduling(i)
      eps_t = 0
      u = np.random.uniform(0, 1, size=1)
      if u > eps_t:
        timestep_act = trajectories.TimeStep(
            timestep.step_type, timestep.reward, timestep.discount,
            timestep.observation.astype(np.float32) / 255.)
        action = model.actor(timestep_act)
        action = action.action
      else:
        action = np.random.choice(
            env.action_spec().maximum.item() + 1, size=1)[0]
      next_timestep = env.step(action)
      info_arr = np.array(env._infos[0]['level_seed'], dtype=np.int32)  # pylint: disable=protected-access
      actions.append(policy_step.PolicyStep(action=action, state=(),
                                            info=info_arr))
    else:
      action = model.actor(
          tf.expand_dims(
              get_state_or_pixels(timestep.observation[0], 'pixel')
              if FLAGS.obs_type == 'state' else (timestep.observation[0]), 0),
          sample=True)
      next_timestep = env.step(action)
      actions.append(
          policy_step.PolicyStep(action=action.numpy()[0], state=(), info=()))

    episode_return += next_timestep.reward[0]
    episode_timesteps += 1

    timestep = next_timestep

  if FLAGS.numpy_dataset:
    observer.save(n_shards=10)
if __name__ == '__main__':
  app.run(main)
