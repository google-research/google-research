# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

r"""Train and Eval DQN on Atari environments.

Training and evaluation proceeds alternately in iterations, where each
iteration consists of a 1M frame training phase followed by a 500K frame
evaluation phase. In the literature, some papers report averages of the train
phases, while others report averages of the eval phases.

This example is configured to use dopamine.atari.preprocessing, which, among
other things, repeats every action it receives for 4 frames, and then returns
the max-pool over the last 2 frames in the group. In this example, when we
refer to "ALE frames" we refer to the frames before the max-pooling step (i.e.
the raw data available for processing). Because of this, many of the
configuration parameters (like initial_collect_steps) are divided by 4 in the
body of the trainer (e.g. if you want to evaluate with 400 frames in the
initial collection, you actually only need to .step the environment 100 times).

For a good survey of training on Atari, see Machado, et al. 2017:
https://arxiv.org/pdf/1709.06009.pdf.

To run:

```bash
tf_agents/agents/dqn/examples/v1/train_eval_atari \
  --root_dir=$HOME/atari/pong \
  --atari_roms_path=/tmp
  --alsologtostderr
```

Additional flags are available such as `--replay_buffer_capacity` and
`--n_step_update`.

"""

import copy
import gc
import json
import os
import time

from absl import flags
from absl import logging

import gin
import numpy as np
import tensorflow.compat.v1 as tf
# from tensorflow.contrib.training.python.training import evaluation

from tf_agents.environments import batched_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_atari
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.networks import q_network
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_py_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.utils import timer

from abps import hparam as hparam_lib
from abps import new_pymetrics
from abps import py_hashed_replay_buffer
from abps.agents.dqn import dqn_agent

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('game_name', 'Pong', 'Name of Atari game to run.')
flags.DEFINE_string(
    'eval_agents', 'worker_0',
    'names of the agents the evaluator will evalutate, multiple agent names should be separated by comma'
)
flags.DEFINE_string(
    'train_agents', None,
    'names of the agents the trainer will train, multiple agent names should be separated by comma'
)
flags.DEFINE_string(
    'architect_prob', None,
    'probability over each type of architecture being selected as initial architecture'
)
flags.DEFINE_string('select_policy_way', 'random',
                    'Way to select behavior policy')
flags.DEFINE_string(
    'hparam_path', None,
    'JSON file that contains hyperparameters and name for each agent')
flags.DEFINE_string('dqn_type', 'dqn', 'type of dqn agent')

flags.DEFINE_integer('num_iterations', None,
                     'Number of train/eval iterations to run.')
flags.DEFINE_integer(
    'initial_collect_steps', None,
    'Number of frames to ALE frames to process before '
    'beginning to train. Since this is in ALE frames, there '
    'will be initial_collect_steps/4 items in the replay '
    'buffer when training starts.')
flags.DEFINE_integer('replay_buffer_capacity', None,
                     'Maximum number of items to store in the replay buffer.')
flags.DEFINE_integer(
    'train_steps_per_iteration', None,
    'Number of ALE frames to run through for each iteration '
    'of training.')
flags.DEFINE_integer(
    'n_step_update', None, 'The number of steps to consider '
    'when computing TD error and TD loss.')
flags.DEFINE_integer('bandit_buffer_size', None,
                     'size of the buffer window size')
flags.DEFINE_integer(
    'eval_episode_per_iteration', None,
    'Number of ALE frames to run through for each iteration '
    'of evaluation.')
flags.DEFINE_integer(
    'eval_interval_secs', None,
    'interval of waiting time for evaluator to detect new ckpt')
flags.DEFINE_integer('epsilon_decay_selection', 200,
                     'Period over which to decay epsilon, for Bandit')
flags.DEFINE_integer('update_policy_iteration', 10,
                     'number of train episode between change policy')
flags.DEFINE_integer('eval_parallel_size', None,
                     'number of process used for parallelization')
flags.DEFINE_integer('num_worker', None, 'number of workers')
flags.DEFINE_integer('pbt_period', 10, 'number of abps runs between pbt')

flags.DEFINE_float('eval_epsilon_greedy', 0.0,
                   'epsilon for the policy when doing evaluation')
flags.DEFINE_float('learning_rate', None, 'Learning rate')
flags.DEFINE_float('ucb_coeff', 2.0, 'coefficient for UCB in best online')
flags.DEFINE_float('bandit_ucb_coeff', 2.0, 'coefficient for UCB in bandit')
flags.DEFINE_float('pbt_percent_low', 0.4, 'percent of agents to be replaced')
flags.DEFINE_float('pbt_percent_top', 0.6, 'percent of agents as good')

flags.DEFINE_boolean('enable_functions', False, '')
flags.DEFINE_boolean('adjust_metric', False, '')
flags.DEFINE_boolean('is_eval', False, 'is this run a evaluator')
flags.DEFINE_boolean('pbt', True, 'if or not using pbt')
flags.DEFINE_boolean(
    'online_eval_use_train', True,
    'when doing online eval for policy selection whether or not to use epsilon greedy'
)
flags.DEFINE_boolean(
    'create_hparam', False,
    'whether or not create hparam when no hparam file is found')

FLAGS = flags.FLAGS

# AtariPreprocessing runs 4 frames at a time, max-pooling over the last 2
# frames. We need to account for this when computing things like update
# intervals.
ATARI_FRAME_SKIP = 4


def softmax(q_table):
  return np.exp(q_table) / sum(np.exp(q_table))


def sigmoid(x, coeff=1, truncate=1):
  prob = 1.0 / (1.0 + np.exp(-coeff * x))
  prob[prob > truncate] = truncate
  return prob


def write_policy_step(old_steps, new_steps, step_index):
  old_steps.step_type[step_index] = new_steps.step_type
  old_steps.reward[step_index] = new_steps.reward
  old_steps.discount[step_index] = new_steps.discount
  old_steps.observation[step_index] = new_steps.observation
  return old_steps


def perturb(num, low=0.95, high=1.05):
  return int(num * np.random.uniform(low, high))


def unstack_time_steps(stack_timesteps):
  st_component = [item for item in stack_timesteps]
  t_components = zip(*st_component)
  return [ts.TimeStep(*t_component) for t_component in t_components]


# Fix the placeholder.
def get_available_gpus():
  return []


def change_from_last_to_mid(time_step):
  return ts.transition(time_step.observation, time_step.reward)


def add_summary(file_writer, tag, value, step):
  summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
  file_writer.add_summary(summary, step)


def write_csv(outdir, tag, value, step, iteration):
  with tf.gfile.GFile(os.path.join(outdir, tag + '%r=3.2:sl=8M'),
                      'a+') as writer:
    if isinstance(value, str):
      writer.write('%d\t%d\t%s\n' % (iteration, step, value))
    else:
      writer.write('%d\t%d\t%f\n' % (iteration, step, value))


class AtariQNetwork(q_network.QNetwork):
  """QNetwork subclass that divides observations by 255."""

  def call(self, observation, step_type=None, network_state=None):
    state = tf.cast(observation, tf.float32)
    # We divide the grayscale pixel values by 255 here rather than storing
    # normalized values beause uint8s are 4x cheaper to store than float32s.
    state = state / 255
    return super(AtariQNetwork, self).call(
        state, step_type=step_type, network_state=network_state)


def convert_list_to_tuple(orig_list):
  if isinstance(orig_list, list):
    return tuple(convert_list_to_tuple(x) for x in orig_list)
  else:
    return orig_list


def log_metric(metric, prefix):
  tag = common.join_scope(prefix, metric.name)
  logging.info('%s', '{0} = {1}'.format(tag, metric.result()))


def game_over(env):
  if env._num_envs == 1:  # pylint: disable=protected-access
    return env.envs[0].game_over
  else:
    return [e.game_over for e in env._envs]  # pylint: disable=protected-access


@gin.configurable
class Runner(object):
  """Train and evaluate DQN on Atari."""

  def __init__(  # pylint: disable=dangerous-default-value
      self,
      root_dir,
      env_name,
      max_episode_frames=108000,  # ALE frames
      terminal_on_life_loss=False,
      conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)],
      fc_layer_params=(512,),
      # Params for collect
      epsilon_greedy=0.01,
      epsilon_decay_period=1000000,  # ALE frames
      # Params for train
      update_period=16,  # ALE frames
      target_update_tau=1.0,
      target_update_period=32000,  # ALE frames
      learning_rate=2.5e-4,
      n_step_update=1,
      gamma=0.99,
      reward_scale_factor=1.0,
      gradient_clipping=None,
      # Params for checkpoints, summaries, and logging
      log_interval=1000,
      summary_interval=1000,
      debug_summaries=False,
      summarize_grads_and_vars=False,
      eval_metrics_callback=None,
      max_ckpt=200,
      hparam_path=None,
      dqn_type='dqn',
      use_gpu=True,
      freeze_before_select=False,
      num_worker=16,
      architect_prob=[0.2, 0.2, 0.2, 0.4],
      is_eval=False,
      create_hparam=False,
  ):
    """A Base Runner class for multi-agent ABPS training.

    Args:
      root_dir: Directory to write log files to.
      env_name: Fully-qualified name of the Atari environment (i.e. Pong-v0).
      max_episode_frames: Maximum length of a single episode, in ALE frames.
      terminal_on_life_loss: Whether to simulate an episode termination when a
        life is lost.
      conv_layer_params: Params for convolutional layers of QNetwork.
      fc_layer_params: Params for fully connected layers of QNetwork.
      epsilon_greedy: Final epsilon value to decay to for training.
      epsilon_decay_period: Period over which to decay epsilon, from 1.0 to
        epsilon_greedy (defined above).
      update_period: Run a train operation every update_period ALE frames.
      target_update_tau: Coeffecient for soft target network updates (1.0 ==
        hard updates).
      target_update_period: Period, in ALE frames, to copy the live network to
        the target network.
      learning_rate: RMS optimizer learning rate.
      n_step_update: The number of steps to consider when computing TD error and
        TD loss. Applies standard single-step updates when set to 1.
      gamma: Discount for future rewards.
      reward_scale_factor: Scaling factor for rewards.
      gradient_clipping: Norm length to clip gradients.
      log_interval: Log stats to the terminal every log_interval training steps.
      summary_interval: Write TF summaries every summary_interval training
        steps.
      debug_summaries: If True, write additional summaries for debugging (see
        dqn_agent for which summaries are written).
      summarize_grads_and_vars: Include gradients in summaries.
      eval_metrics_callback: A callback function that takes (metric_dict,
        global_step) as parameters. Called after every eval with the results of
        the evaluation.
      max_ckpt: Max ckpt.
      hparam_path: Path to the JSON file that contains hyperparameters for each
        individual agent. Tunable hyperparams including:
          epsilon_greedy,epsilon_decay_period,target_update_tau,target_update_period.
          If not speicified in the JSON the agent will use arguments passed in
          _init() as default values for each hparam. learning_rate=2.5e-4,
      dqn_type: A string specifying if dqn or double dqn is used
      use_gpu: whether or not to use GPU
      freeze_before_select: whether to freeze the model parameters while collect
        data.
      num_worker: Number of workers.
      architect_prob: Architecture probabilities.
      is_eval: Is evalutation.
      create_hparam: Create hparam.

    """
    update_period = update_period / ATARI_FRAME_SKIP
    self._max_ckpt = max_ckpt
    self._update_period = update_period
    self._summary_interval = summary_interval
    self._log_interval = log_interval
    self._eval_metrics_callback = eval_metrics_callback
    self._dqn_type = dqn_type
    self._env_name = env_name
    self._max_episode_frames = max_episode_frames
    self._freeze_before_select = freeze_before_select
    self._architect_prob = architect_prob
    self._do_eval = is_eval
    self._reward_scale_factor = reward_scale_factor
    self._gradient_clipping = gradient_clipping
    self._debug_summaries = debug_summaries
    self._summarize_grads_and_vars = summarize_grads_and_vars
    self._create_hparam = create_hparam

    if use_gpu:
      self._devices = get_available_gpus()
      if not self._devices:
        logging.info('gpu devices not available, switching to cpu mode.')
        use_gpu = False
      else:
        logging.info('find gpu devices: %s', self._devices)
    self._use_gpu = use_gpu
    self._hyper_range = {
        'edecay': [250000, 4000000],
        'lr': [0.00001, 0.005],
        'architect':
            np.array(
                ['agent_deep', 'agent_normal', 'agent_wide', 'agent_small'])
    }

    self._default_hparams = {
        'conv': conv_layer_params,
        'fc': fc_layer_params,
        'epsilon_greedy': epsilon_greedy,
        'edecay': epsilon_decay_period,
        'update_period': update_period,
        'target_update_tau': target_update_tau,
        'target_update_period': target_update_period,
        'lr': learning_rate,
        'n_step_update': n_step_update,
        'gamma': gamma,
        'architect': 'agent_deep',
        'name': 'placeholder',
    }

    with gin.unlock_config():
      gin.bind_parameter('AtariPreprocessing.terminal_on_life_loss',
                         terminal_on_life_loss)

    root_dir = os.path.expanduser(root_dir)
    logging.info('root dir:%s', root_dir)
    self._home_dir = os.path.dirname(root_dir)
    logging.info('home dir:%s', self._home_dir)
    self._train_dir = os.path.join(root_dir, 'train')
    self._csv_dir = os.path.join(root_dir, 'eval')
    self._hparam_dir = os.path.join(root_dir, 'hparam')
    if not tf.gfile.Exists(self._csv_dir):
      tf.gfile.MakeDirs(self._csv_dir)
    if not tf.gfile.Exists(self._hparam_dir):
      tf.gfile.MakeDirs(self._hparam_dir)
    self._train_file_writer = tf.summary.FileWriter(self._train_dir)
    self._worker_names = None
    self._env = suite_atari.load(
        self._env_name,
        max_episode_steps=self._max_episode_frames / ATARI_FRAME_SKIP,
        gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
    self._env = batched_py_environment.BatchedPyEnvironment([self._env])

    self._observation_spec = tensor_spec.from_spec(self._env.observation_spec())
    self._time_step_spec = ts.time_step_spec(self._observation_spec)
    self._action_spec = tensor_spec.from_spec(self._env.action_spec())

    self._py_observation_spec = self._env.observation_spec()
    self._py_time_step_spec = ts.time_step_spec(self._py_observation_spec)
    self._py_action_spec = policy_step.PolicyStep(self._env.action_spec())
    self._data_spec = trajectory.from_transition(self._py_time_step_spec,
                                                 self._py_action_spec,
                                                 self._py_time_step_spec)
    self._hparams = {}
    hparam_path = None
    if tf.gfile.Glob(os.path.join(self._hparam_dir, 'hparam-*.json')):
      logging.info('found existing hyper parameters')
      most_recent_hparam = max([
          int(x.split('.json')[0].split('-')[-1]) for x in tf.gfile.Glob(
              os.path.join(self._hparam_dir, 'hparam-*.json'))
      ])
      hparam_path = os.path.join(self._hparam_dir,
                                 'hparam-{}.json'.format(most_recent_hparam))
      self._load_hparam(hparam_path)
    elif self._create_hparam:
      logging.info('creating hyper parameters')
      self._worker_names = ['worker_' + str(x) for x in range(num_worker)]
      self._num_agents = num_worker
      for i, worker_name in enumerate(self._worker_names):
        hparam = hparam_lib.HParams(**self._default_hparams)
        architect = np.random.choice(
            self._hyper_range['architect'], p=self._architect_prob)
        hparam = self._gen_hparam(hparam, architect=architect)
        hparam.name = '_'.join([
            architect, 'conv{:d}_fc{:d}-{:d}-0-lr{:.6f}-edecay{:d}'.format(
                hparam.conv[0][0], hparam.fc[0], i + 1, hparam.lr,
                hparam.edecay)
        ])
        hparam.architect = architect
        logging.info('hparam=%s', hparam.values())
        self._hparams[worker_name] = hparam
      if not self._do_eval:
        self.save_hparam(ckpt_step=0)
        self.save_hparam(ckpt_step=0, snapshot=True)
      self._agent_names = {
          worker: self._hparams[worker].name for worker in self._worker_names
      }
      tf.gfile.Copy(
          os.path.join(self._hparam_dir, 'hparam-0.json'),
          os.path.join(self._home_dir, 'hparam-0.json'),
          overwrite=True)
      hparam_path = os.path.join(self._home_dir, 'hparam-0.json')
    else:
      hparam_path = os.path.join(self._home_dir, 'hparam-0.json')
      self._load_hparam(hparam_path)
      if not self._do_eval:
        tf.gfile.Copy(
            hparam_path,
            os.path.join(self._hparam_dir, 'hparam-0.json'),
            overwrite=True)
    self._hparam_step = hparam_path.split('.json')[0].split('-')[-1]
    logging.info('agent_name:%s', self._agent_names)
    self._iteration_metric = py_metrics.CounterMetric(name='Iteration')

  def build_graph_and_assign_device(self):
    """Build graph and assign device."""
    self._agents = {}
    self._init_agent_ops = []
    self._device_name = {}
    for i, worker in enumerate(self._worker_names):
      hparam = self._hparams[worker]
      if self._devices:
        device = '/gpu:' + str(i % len(self._devices))
      else:
        device = '/cpu:0'

      with tf.device(device):
        logging.info('%s (%s) is assigned to machine %s', worker, hparam.name,
                     device)
        agent = self.create_or_copy_agent(
            hparam, qtype=self._dqn_type, device=device)
        self._agents[worker] = agent
        self._device_name[worker] = device
        self._init_agent_ops.append(agent.initialize())
    self._init_agent_op = tf.group(self._init_agent_ops)
    with (tf.device('/gpu:' +
                    str((len(self._worker_names) + 1) % len(self._devices)))
          if self._devices else tf.device('/cpu:0')):
      self._behavior_index = tf.Variable(0, dtype=tf.int32, trainable=False)

  def save_hparam(self, ckpt_step, snapshot=False):
    """Save hparam."""
    hparam_dict = {
        worker: self._hparams[worker].values() for worker in self._worker_names
    }
    if snapshot:
      filename = os.path.join(self._hparam_dir,
                              'stathparam-' + str(ckpt_step) + '.json')
    else:
      filename = os.path.join(self._hparam_dir,
                              'hparam-' + str(ckpt_step) + '.json')
    with tf.gfile.GFile(filename, 'w') as json_file:
      json.dump(hparam_dict, json_file)

  def _load_hparam(self, hparam_path):
    """Load hparam."""
    logging.info('waiting for hparam path:%s', hparam_path)
    read = False
    while not read:
      # try:
      with tf.gfile.Open(hparam_path) as json_file:
        self._hparam_dict = json.load(json_file)
      read = True

    self._num_agents = len(self._hparam_dict)
    if not self._worker_names:
      self._worker_names = np.asarray(sorted(list(self._hparam_dict.keys())))
    logging.info('found Hparam file at: %s, loading', hparam_path)
    # set unspecified hparams to default values
    for worker_name, d in self._hparam_dict.items():
      hparam = hparam_lib.HParams(**self._default_hparams)
      for k in d:
        if k in ['conv', 'fc']:
          d[k] = list(convert_list_to_tuple(d[k]))
      hparam.override_from_dict(d)
      # hparam.edecay = int(hparam.edecay /
      #                     ATARI_FRAME_SKIP / hparam.update_period)
      # hparam.target_update_period = int(hparam.target_update_period
      #                              / ATARI_FRAME_SKIP / hparam.update_period)
      self._hparams[worker_name] = hparam
    self._agent_names = {
        worker: self._hparams[worker].name for worker in self._worker_names
    }

  def _gen_hparam(self, hparam, architect='agent_small'):
    """Gen hparam."""
    tuning_hparams = {
        'edecay': np.log([250000, 4000000]),
        'lr': np.log([0.00001, 0.005])
    }
    for k in tuning_hparams:
      if k == 'edecay':
        hparam.set_hparam(k, int(np.exp(np.random.uniform(*tuning_hparams[k]))))
      else:
        hparam.set_hparam(k, np.exp(np.random.uniform(*tuning_hparams[k])))
    if architect == 'agent_deep':
      hparam.set_hparam('conv', [(perturb(32), (8, 8), 4),
                                 (perturb(64), (4, 4), 2),
                                 (perturb(64), (3, 3), 1)])
      hparam.set_hparam('fc', [
          perturb(512),
      ])
    if architect == 'agent_normal':
      hparam.set_hparam('conv', [(perturb(16), (8, 8), 4),
                                 (perturb(32), (4, 4), 2)])
      hparam.set_hparam('fc', [
          perturb(256),
      ])
    if architect == 'agent_wide':
      hparam.set_hparam('conv', [(perturb(32), (5, 5), 4),
                                 (perturb(64), (3, 3), 1)])
      hparam.set_hparam('fc', [
          perturb(1024),
      ])
    if architect == 'agent_small':
      hparam.set_hparam('conv', [(perturb(8), (8, 8), 4),
                                 (perturb(8), (4, 4), 2)])
      hparam.set_hparam('fc', [
          perturb(32),
      ])
    return hparam

  def _init_graph(self, sess):
    self._train_checkpointer.initialize_or_restore(sess)
    common.initialize_uninitialized_variables(sess)
    sess.run(self._init_agent_op)
    self._collect_timer = timer.Timer()
    self._train_timer = timer.Timer()
    self._select_timer = timer.Timer()

  def _store_to_rb(self, traj, worker_name=None):
    # Clip the reward to (-1, 1) to normalize rewards in training.
    traj = traj._replace(reward=np.asarray(np.clip(traj.reward, -1, 1)))
    if worker_name:
      self._replay_buffer[worker_name].add_batch(traj)
    else:
      self._replay_buffer.add_batch(traj)

  def create_or_copy_agent(self,
                           hparam,
                           qtype='dqn',
                           device=None,
                           sess=None,
                           parent_agent=None,
                           current_worker=None,
                           do_copy=False):
    """Create or copy agent."""
    hname = hparam.name.replace('-', '_').replace('.', '_')
    if parent_agent:
      step_int = sess.run(parent_agent.train_step_counter)
    with tf.device(device):
      if do_copy:
        train_step = tf.Variable(
            step_int,
            dtype=tf.int64,
            trainable=False,
            name='train_step_' + hname)
      else:
        train_step = tf.Variable(
            0, dtype=tf.int64, trainable=False, name='train_step_' + hname)
      epsilon = tf.train.polynomial_decay(
          1.0,
          train_step,
          int(hparam.edecay / ATARI_FRAME_SKIP /
              hparam.update_period),  # hparam.edecay,
          end_learning_rate=hparam.epsilon_greedy)
      optimizer = tf.train.RMSPropOptimizer(
          learning_rate=hparam.lr,
          decay=0.95,
          momentum=0.0,
          epsilon=0.00001,
          centered=True)
      q_net = AtariQNetwork(
          self._observation_spec,
          self._action_spec,
          conv_layer_params=hparam.conv,
          fc_layer_params=hparam.fc)
      if qtype == 'dqn':
        agent = dqn_agent.DqnAgent(
            self._time_step_spec,
            self._action_spec,
            q_network=q_net,
            optimizer=optimizer,
            epsilon_greedy=epsilon,
            n_step_update=hparam.n_step_update,
            update_period=hparam.update_period,
            target_update_tau=hparam.target_update_tau,
            target_update_period=int(
                hparam.target_update_period / ATARI_FRAME_SKIP /
                hparam.update_period),  # hparam.target_update_period,
            td_errors_loss_fn=dqn_agent.element_wise_huber_loss,
            gamma=hparam.gamma,
            reward_scale_factor=self._reward_scale_factor,
            gradient_clipping=self._gradient_clipping,
            debug_summaries=self._debug_summaries,
            enable_functions=FLAGS.enable_functions,
            summarize_grads_and_vars=self._summarize_grads_and_vars,
            train_step_counter=train_step,
            name=hname)
      elif qtype == 'ddqn':
        agent = dqn_agent.DdqnAgent(
            self._time_step_spec,
            self._action_spec,
            q_network=q_net,
            optimizer=optimizer,
            epsilon_greedy=epsilon,
            n_step_update=hparam.n_step_update,
            update_period=hparam.update_period,
            target_update_tau=hparam.target_update_tau,
            target_update_period=int(
                hparam.target_update_period / ATARI_FRAME_SKIP /
                hparam.update_period),  # hparam.target_update_period,
            td_errors_loss_fn=dqn_agent.element_wise_huber_loss,
            gamma=hparam.gamma,
            reward_scale_factor=self._reward_scale_factor,
            gradient_clipping=self._gradient_clipping,
            debug_summaries=self._debug_summaries,
            enable_functions=FLAGS.enable_functions,
            summarize_grads_and_vars=self._summarize_grads_and_vars,
            train_step_counter=train_step,
            name=hname)
    if do_copy:
      agent.train(self._rb_iterator[current_worker])
      common.initialize_uninitialized_variables(sess)
      sess.run(agent.initialize())
      common.soft_variables_update(
          agent._q_network.variables,  # pylint: disable=protected-access
          parent_agent._q_network.variables,  # pylint: disable=protected-access
          tau=1.0)
      common.soft_variables_update(
          agent._target_q_network.variables,  # pylint: disable=protected-access
          parent_agent._target_q_network.variables,  # pylint: disable=protected-access
          tau=1.0)
    return agent

  def _online_eval_parallel(self,
                            sess,
                            batched_policy,
                            batched_env,
                            episode_limit,
                            metrics=None):
    """Online eval parallel."""
    del sess
    with self._eval_timer:
      num_env = batched_env.batch_size
      time_steps = batched_env.reset()
      for metric in metrics:
        metric.reset()
      count_started_episode = len(np.where(time_steps.is_first())[0])
      done_create_process = False
      count_masked = 0
      while True:
        action_steps = batched_policy.action(time_steps)
        next_time_steps = batched_env.step(action_steps.action)
        is_last = np.where(next_time_steps.is_last())[0]
        if is_last:
          is_over = is_last[np.asarray(
              [batched_env._envs[i].game_over for i in is_last])]  # pylint: disable=protected-access
          yet_alive = np.setdiff1d(is_last, is_over)
          logging.info('Find %d cases that need in place restart',
                       len(yet_alive))
        else:
          is_over = []
        batched_traj = trajectory.from_transition(time_steps, action_steps,
                                                  next_time_steps)
        self._update_metrics(metrics, batched_traj)
        time_steps = next_time_steps
        if is_over:
          if not done_create_process:
            unpacked_time_steps = unstack_time_steps(time_steps)
            for process_id in is_over:
              unpacked_time_steps[process_id] = batched_env._envs[  # pylint: disable=protected-access
                  process_id].reset(False)()
            time_steps = batched_env._stack_time_steps(unpacked_time_steps)  # pylint: disable=protected-access
            count_started_episode += len(np.where(time_steps.is_first())[0])
            logging.info('started episode:%d', count_started_episode)
            if count_started_episode >= episode_limit:
              done_create_process = True
          else:
            for metric in metrics:
              count_masked = metric.set_mask(is_over)
              logging.info('finished_process:%d', count_masked)
        if count_masked == num_env:
          break
    logging.info('%s', 'online eval time = {}'.format(self._eval_timer.value()))
    self._eval_timer.reset()

  def _run_step(self, sess, env, time_step, policy):
    """Run step."""
    del sess
    action_step = policy.action(time_step)
    # 1. Previous comment.
    # When AtariPreprocessing.terminal_on_life_loss is True, we receive LAST
    # time_steps when lives are lost but the game is not over.In this mode, the
    # replay buffer and agent's policy must see the life loss as a LAST step
    # and the subsequent step as a FIRST step. However, we do not want to
    # actually terminate the episode and metrics should be computed as if all
    # steps were MID steps, since life loss is not actually a terminal event
    # (it is mostly a trick to make it easier to propagate rewards backwards by
    # shortening episode durations from the agent's perspective).
    #
    # 2. Here is the implementation.
    # (1) Store the terminal step to the replay buffer if necessary.
    # (2) Update the metrics as if it is not a terminal step.
    # (3) Restart from the terminal step.
    next_time_step = env.step(action_step.action)
    train_traj = trajectory.from_transition(time_step, action_step,
                                            next_time_step)
    metric_traj = train_traj

    if next_time_step.is_last() and not game_over(env):
      # Update metrics as if this is a mid-episode step.
      next_time_step_mid = change_from_last_to_mid(next_time_step)
      metric_traj = trajectory.from_transition(time_step, action_step,
                                               next_time_step_mid)

      # Resume the game in place.
      next_time_step = ts.restart(next_time_step.observation)

    return next_time_step, metric_traj, metric_traj

  def _update_metrics(self, metrics, traj):
    for metric in metrics:
      metric(traj)

  def _online_eval(self, sess, policy, run_steps=1000, metrics=None):
    env_steps = 0
    for metric in metrics:
      metric.reset()
    while env_steps < run_steps:
      env_steps += self._run_episode(
          sess, env=self._env, policy=policy, metrics=metrics, train=False)

  def _maybe_log_train(self, train_step, total_loss, agent_name):
    """Log some stats if train_step is a multiple of log_interval."""
    if train_step % self._log_interval == 0:
      logging.info('agent %s', agent_name)
      logging.info('step = %d, loss = %f', train_step, total_loss.loss)

  def create_pypolicy_and_train_op(self):
    """Create pypolicy and train op."""
    self._collect_py_policies = {}
    self._select_py_policies = {}
    self._rb_iterator = {}
    for i in range(len(self._worker_names)):
      worker = self._worker_names[i]
      agent = self._agents[worker]
      device = self._device_name[worker]
      with tf.device('/cpu:0'):
        ds = self._replay_buffer[worker].as_dataset(
            sample_batch_size=self._batch_size,
            num_steps=self._hparams[worker].n_step_update + 1)
        ds = ds.prefetch(4)
        ds = ds.apply(tf.data.experimental.prefetch_to_device(device))
      with tf.device(device):
        self._collect_py_policies[worker] = py_tf_policy.PyTFPolicy(
            agent.collect_policy)
        self._select_py_policies[worker] = py_tf_policy.PyTFPolicy(
            agent.collect_policy)
        self._rb_iterator[worker] = tf.data.make_one_shot_iterator(
            ds).get_next()
        agent.train(self._rb_iterator[worker])

  def create_metrics_checkpointer(self):
    for worker in self._worker_names:
      self._metric_checkpointer[worker] = common.Checkpointer(
          ckpt_dir=os.path.join(self._train_dir, worker, 'behaviormetric'),
          max_to_keep=self._max_ckpt,
          **{
              ('behavior_metric_' + worker):
                  metric_utils.MetricsGroup(self._behavior_metrics[worker],
                                            worker + '_metric')
          })

  def record_log_metric(self):
    """Record log metric."""
    for worker in self._worker_names:
      env_step = int(self._env_steps_metric[worker].result())
      if self._pbt or self._use_bandit:
        add_summary(self._train_file_writer,
                    'QMetrics/' + self._bandit_arm_q[worker].name,
                    self._bandit_arm_q[worker].result('most_recent'), env_step)
        write_csv(self._csv_dir, self._bandit_arm_q[worker].name,
                  self._bandit_arm_q[worker].result(), env_step,
                  self._episode_metric[worker].result())
      add_summary(self._train_file_writer, 'WorkerID/' + 'pbt_id_' + worker,
                  int(self._agent_names[worker].split('-')[1]), env_step)
      add_summary(self._train_file_writer,
                  'WorkerID/' + 'pbt_parent_id_' + worker,
                  int(self._agent_names[worker].split('-')[2]), env_step)
      add_summary(
          self._train_file_writer, 'WorkerHparam/' + 'architect_' + worker,
          np.where(self._hyper_range['architect'] ==
                   self._hparams[worker].architect)[0][0], env_step)
      add_summary(self._train_file_writer, 'WorkerHparam/' + 'lr_' + worker,
                  self._hparams[worker].lr, env_step)
      add_summary(self._train_file_writer, 'WorkerHparam/' + 'edecay_' + worker,
                  self._hparams[worker].edecay, env_step)
      write_csv(self._csv_dir, 'worker_status_' + worker,
                self._agent_names[worker], env_step,
                self._episode_metric[worker].result())

  def update_rb_metric_checkpointer(
      self, use_common=False):  # iteration_metrics=self._iteration_metric,
    """Update rb metric checkpointer."""
    logging.info('updating rb and metric checkpointer with common=%s',
                 use_common)
    self._metric_checkpointer = {}
    self._rb_checkpointer = {}
    # all_iterable={('behavior_metrics_'+worker) :
    #   metric_utils.MetricsGroup(self._behavior_metrics[worker],
    #   worker+'_metric') for worker in self._worker_names}
    if use_common:
      for worker in self._worker_names:
        # all = {
        #     ('behavior_metric_' + worker):
        #         metric_utils.MetricsGroup(self._behavior_metrics[worker],
        #                                   worker + '_metric')
        # }
        self._rb_checkpointer[worker] = common.Checkpointer(
            ckpt_dir=os.path.join(self._train_dir, worker, 'replay_buffer'),
            max_to_keep=10,
            replay_buffer=self._replay_buffer[worker])
        self._metric_checkpointer[worker] = common.Checkpointer(
            ckpt_dir=os.path.join(self._train_dir, worker, 'behaviormetric'),
            max_to_keep=self._max_ckpt,
            behavior_metric=metric_utils.MetricsGroup(
                self._behavior_metrics[worker], worker + '_metric'))
      self._basic_checkpointer = common.Checkpointer(
          ckpt_dir=os.path.join(self._train_dir, 'basic'),
          max_to_keep=self._max_ckpt,
          metrics=metric_utils.MetricsGroup(
              [self._iteration_metric, self._pbt_id], 'basice_metric'))
    else:
      for worker in self._worker_names:
        # all = {
        #     ('behavior_metric_' + worker):
        #         metric_utils.MetricsGroup(self._behavior_metrics[worker],
        #                                   worker + '_metric')
        # }
        self._rb_checkpointer[worker] = tf.train.CheckpointManager(
            tf.train.Checkpoint(replay_buffer=self._replay_buffer[worker]),
            directory=os.path.join(self._train_dir, worker, 'replay_buffer'),
            max_to_keep=10)
        self._metric_checkpointer[worker] = tf.train.CheckpointManager(
            tf.train.Checkpoint(
                behavior_metric=metric_utils.MetricsGroup(
                    self._behavior_metrics[worker], worker + '_metric')),
            directory=os.path.join(self._train_dir, worker, 'behaviormetric'),
            max_to_keep=self._max_ckpt)
      self._basic_checkpointer = tf.train.CheckpointManager(
          tf.train.Checkpoint(
              metrics=metric_utils.MetricsGroup(
                  [self._iteration_metric, self._pbt_id], 'basice_metric')),
          directory=os.path.join(self._train_dir, 'basic'),
          max_to_keep=self._max_ckpt)

  def update_train_bandit_checkpointer(self,
                                       update_bandit=True,
                                       use_common=False):
    """Update train bandit checkpointer."""
    self._train_checkpointer = {}
    self._bandit_checkpointer = {}
    logging.info('updating train and bandit checkpointer with common=%s',
                 use_common)
    if use_common:
      for worker in self._worker_names:
        self._train_checkpointer[worker] = common.Checkpointer(
            ckpt_dir=os.path.join(self._train_dir, worker),
            max_to_keep=self._max_ckpt,
            **{worker: self._agents[worker]})
        if update_bandit:
          self._bandit_checkpointer[worker] = common.Checkpointer(
              ckpt_dir=os.path.join(self._train_dir, worker, 'bandit'),
              max_to_keep=self._max_ckpt,
              **{
                  ('pbt_metrics_' + worker):
                      metric_utils.MetricsGroup([self._bandit_arm_q[worker]],
                                                'metric_q_' + worker)
              })
    else:
      for worker in self._worker_names:
        self._train_checkpointer[worker] = tf.train.CheckpointManager(
            tf.train.Checkpoint(**{worker: self._agents[worker]}),
            directory=os.path.join(self._train_dir, worker),
            max_to_keep=self._max_ckpt)
        if update_bandit:
          self._bandit_checkpointer[worker] = tf.train.CheckpointManager(
              tf.train.Checkpoint(
                  **{
                      ('pbt_metrics_' + worker):
                          metric_utils.MetricsGroup(
                              [self._bandit_arm_q[worker]], 'metric_q_' +
                              worker)
                  }),
              directory=os.path.join(self._train_dir, worker, 'bandit'),
              max_to_keep=self._max_ckpt)

  def update_train_checkpointer_large(self, use_common=False):
    all_iterable = {worker: self._agents[worker] for worker in self._agents}
    if use_common:
      self._train_checkpointer_large = common.Checkpointer(
          ckpt_dir=self._train_dir, max_to_keep=self._max_ckpt, **all_iterable)
    else:
      self._train_checkpointer_large = tf.train.CheckpointManager(
          tf.train.Checkpoint(**all_iterable),
          directory=self._train_dir,
          max_to_keep=self._max_ckpt)

  def save_checkpoints(self, ep_step_int, use_common=False, save_basic=False):
    """Save checkpoints."""
    if use_common:
      for worker in self._worker_names:
        self._train_checkpointer[worker].save(global_step=ep_step_int)
        if self._use_bandit or self._pbt:
          self._bandit_checkpointer[worker].save(global_step=ep_step_int)
        self._rb_checkpointer[worker].save(global_step=ep_step_int)
        self._metric_checkpointer[worker].save(global_step=ep_step_int)
      if save_basic:
        self._basic_checkpointer.save(global_step=ep_step_int)
    else:
      for worker in self._worker_names:
        self._train_checkpointer[worker].save(checkpoint_number=ep_step_int)
        if self._use_bandit or self._pbt:
          self._bandit_checkpointer[worker].save(checkpoint_number=ep_step_int)
        self._rb_checkpointer[worker].save(checkpoint_number=ep_step_int)
        self._metric_checkpointer[worker].save(checkpoint_number=ep_step_int)
      if save_basic:
        self._basic_checkpointer.save(checkpoint_number=ep_step_int)


@gin.configurable
class PBTRunner(Runner):
  """Train and evaluate DQN on Atari."""

  def __init__(  # pylint: disable=dangerous-default-value
      self,
      num_iterations=200,
      initial_collect_steps=80000,  # ALE frames
      replay_buffer_capacity=400000,
      update_policy_iteration=10,  # train episode between switching policy
      ucb_coeff=1.96,
      train_steps_per_iteration=1000000,  # ALE frames
      batch_size=32,
      online_eval_use_train=True,
      epsilon_decay_selection=200,
      epsilon_selection=0.01,
      bandit_discount=1.0,
      bandit_ucb_coeff=2.0,
      bandit_buffer_size=80,
      eval_episode_select=50,  # ALE frames
      eval_parallel_size=25,
      pbt=False,
      pbt_period=100,
      pbt_exploit_way='uniform',
      pbt_low='ucb',
      pbt_high='lcb',
      pbt_update_requirement=10,
      push_when_eval=False,
      pbt_percent_low=0.2,
      pbt_percent_top=0.2,
      pbt_drop_prob=0.8,
      pbt_mutation_rate=0.8,
      pbt_mutate_list=['lr', 'edecay'],
      pbt_perturb_factors=[0.8, 1.2],
      train_agents=None,
      # select_policy_way='independent',
      **kwargs):
    super(PBTRunner, self).__init__(**kwargs)
    self._num_iterations = num_iterations
    self._update_policy_period = update_policy_iteration
    self._ucb_coeff = ucb_coeff
    self._initial_collect_steps = initial_collect_steps / ATARI_FRAME_SKIP
    self._online_eval_use_train = online_eval_use_train
    self._use_bandit = pbt
    self._bandit_discount = bandit_discount
    self._bandit_ucb_coeff = bandit_ucb_coeff
    self._bandit_buffer_size = bandit_buffer_size
    self._pbt = pbt
    self._pbt_period = pbt_period
    self._pbt_exploit_way = pbt_exploit_way
    self._pbt_low = pbt_low
    self._pbt_high = pbt_high
    self._pbt_update_requirement = pbt_update_requirement
    self._push_when_eval = push_when_eval
    self._pbt_percent_low = pbt_percent_low
    self._pbt_percent_top = pbt_percent_top
    self._pbt_drop_prob = pbt_drop_prob
    self._pbt_mutation_rate = pbt_mutation_rate
    self._pbt_mutate_list = pbt_mutate_list
    self._pbt_perturb_factors = pbt_perturb_factors
    self._pbt_id = py_metrics.CounterMetric(name='pbt_id')
    if self._pbt:
      self._train_episode_per_iteration = pbt_period
    else:
      self._train_episode_per_iteration = int(train_steps_per_iteration /
                                              (ATARI_FRAME_SKIP * 1200))
    self._eval_episode_select = eval_episode_select  # / ATARI_FRAME_SKIP
    self._eval_parallel_size = eval_parallel_size
    self._epsilon_decay_selection = epsilon_decay_selection
    self._epsilon_selection = epsilon_selection
    self._batch_size = batch_size
    self._replay_buffer_capacity = replay_buffer_capacity

    if train_agents:
      self._worker_names = train_agents

    assert not FLAGS.enable_functions
    for _ in range(self._num_agents):
      self._pbt_id()
    self._env_steps_metric = {}
    self._episode_metric = {}
    self._step_metrics = {}
    self._behavior_metrics = {}
    self._replay_buffer = {}
    self._rb_checkpointer = {}
    self._train_file_writers = {}

    self._env_select = parallel_py_environment.ParallelPyEnvironment([
        lambda: suite_atari.load(  # pylint: disable=g-long-lambda
            self._env_name,
            max_episode_steps=self._max_episode_frames / ATARI_FRAME_SKIP,
            gym_env_wrappers=suite_atari.
            DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
    ] * self._eval_parallel_size)

    self._selection_metrics = [
        new_pymetrics.DistributionReturnMetric(
            name='SelectAverageReturn',
            buffer_size=np.inf,
            batch_size=self._eval_parallel_size),
        new_pymetrics.DistributionEpisodeLengthMetric(
            name='SelectAverageEpisodeLength',
            buffer_size=np.inf,
            batch_size=self._eval_parallel_size),
    ]

    for worker in self._worker_names:
      if not tf.gfile.Exists(os.path.join(self._train_dir, worker)):
        tf.gfile.MakeDirs(os.path.join(self._train_dir, worker))
      self._train_file_writers[worker] = tf.summary.FileWriter(
          os.path.join(self._train_dir, worker))
      self._env_steps_metric[worker] = py_metrics.EnvironmentSteps(
      )  # name='EnvironmentSteps_'+worker
      self._episode_metric[worker] = py_metrics.NumberOfEpisodes(
      )  # name='NumberOfEpisodes_'+worker
      self._step_metrics[worker] = [
          self._episode_metric[worker], self._env_steps_metric[worker]
      ]
      # The rolling average of the last 10 episodes.
      self._behavior_metrics[worker] = self._step_metrics[worker] + [
          py_metrics.AverageReturnMetric(
              buffer_size=10),  # name='AverageReturn_'+worker,
          py_metrics.AverageEpisodeLengthMetric(
              buffer_size=10),  # name='AverageEpisode_'+worker,
      ]
      self._replay_buffer[worker] = (
          py_hashed_replay_buffer.PyHashedReplayBuffer(
              data_spec=self._data_spec, capacity=replay_buffer_capacity))
      logging.info('replay_buffer_capacity: %s', replay_buffer_capacity)

    self.build_graph_and_assign_device()
    self.create_pypolicy_and_train_op()

    if self._use_bandit or self._pbt:
      self._bandit_arm_q = {}
      for worker in self._worker_names:
        self._bandit_arm_q[worker] = new_pymetrics.QMetric(
            name='QMetric_' + worker, buffer_size=self._bandit_buffer_size)

    self.update_train_bandit_checkpointer(
        update_bandit=(self._use_bandit or self._pbt), use_common=True)
    self.update_rb_metric_checkpointer(use_common=True)
    logging.info('finished init')

  def pbtdone(self):
    return tf.gfile.Exists(os.path.join(self._train_dir, 'PBTDone'))

  def iterdone(self):
    done = True
    for worker in self._worker_names:
      done = done & tf.gfile.Exists(
          os.path.join(self._train_dir, worker, 'IterDone'))
    return done

  def run(self):
    """Execute the train loop."""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self._count_reevaluate = 0
    num_iter = 0
    while num_iter < self._num_iterations:
      while self.iterdone() and not self.pbtdone():
        logging.info('waiting for controller to complete')
        time.sleep(10)
      self.update_train_bandit_checkpointer(
          update_bandit=(self._use_bandit or self._pbt), use_common=True)
      self.update_rb_metric_checkpointer(use_common=True)
      logging.info('starting trainer')
      with tf.Session(config=config) as sess:
        if num_iter == 0:
          self._initialize_graph(sess)
        else:
          self._initialize_graph(sess, is_first=False)
        if self._iteration_metric.result() == 0:
          self._initial_collect()
          self._initial_eval(sess)
        if self._iteration_metric.result() > self._num_iterations:
          break
        # Train phase
        num_episode = 0
        while num_episode < self._train_episode_per_iteration:
          for worker in self._worker_names:
            self._run_episode(
                sess,
                env=self._env,
                policy=self._collect_py_policies[worker],
                worker_name=worker,
                metrics=self._behavior_metrics[worker],
                train=True,
                collect=True)
            logging.info('finished episode %d for agent %s',
                         self._episode_metric[worker].result(), worker)
          num_episode += 1
          # Fix worker
          worker = self._worker_names[0]
          if self._episode_metric[worker].result(
          ) % self._update_policy_period == 0:
            self.record_log_metric()
        # self._iteration_metric()
        ep_step = self._iteration_metric.result()
        ep_step_int = int(ep_step)
        num_iter = ep_step_int + 1
        self._update_eval(sess)
        with self._checkpoint_timer:
          logging.info('saving checkpoints')
          self.save_checkpoints(num_iter, use_common=True)
      self._maybe_log_and_reset_timer()
      if self._pbt:
        tf.reset_default_graph()
        if self.pbtdone():
          tf.gfile.Remove(os.path.join(self._train_dir, 'PBTDone'))
        for worker in self._worker_names:
          with tf.gfile.GFile(
              os.path.join(self._train_dir, worker, 'IterDone'), 'w') as writer:
            writer.write('IterDone')
        while not self.pbtdone():
          logging.info('waiting for controller to complete')
          time.sleep(10)
        most_recent_hparam = num_iter
        hparam_path = os.path.join(self._hparam_dir,
                                   'hparam-{}.json'.format(most_recent_hparam))
        self._load_hparam(hparam_path)
        self.build_graph_and_assign_device()
        self.create_pypolicy_and_train_op()
      self.update_train_bandit_checkpointer(update_bandit=True, use_common=True)
      self.update_rb_metric_checkpointer(use_common=True)
    with tf.gfile.GFile(os.path.join(self._train_dir, 'TrainDone'),
                        'w') as writer:
      writer.write('TrainDone')

  def _initialize_graph(self, sess, is_first=True):
    """Initialize the graph for sess."""
    self._basic_checkpointer.initialize_or_restore(sess)
    for worker in self._worker_names:
      self._rb_checkpointer[worker].initialize_or_restore(sess)
      self._train_checkpointer[worker].initialize_or_restore(sess)
      if self._use_bandit or self._pbt:
        self._bandit_checkpointer[worker].initialize_or_restore(sess)
      self._metric_checkpointer[worker].initialize_or_restore(sess)

    common.initialize_uninitialized_variables(sess)
    sess.run(self._init_agent_op)

    env_step = self._iteration_metric.result()
    env_step_int = int(env_step)

    if is_first:
      self._timed_at_step = {}
      self._collect_timer = {}
      self._eval_timer = timer.Timer()
      self._train_timer = {}
      self._pbt_timer = {}
      self._checkpoint_timer = {}
      for worker in self._worker_names:
        self._collect_timer[worker] = timer.Timer()
        self._train_timer[worker] = timer.Timer()
      self._pbt_timer = timer.Timer()
      self._checkpoint_timer = timer.Timer()

    for worker in self._worker_names:
      self._timed_at_step[worker] = sess.run(
          self._agents[worker].train_step_counter)

    # Call save to initialize the save_counter (need to do this before
    # finalizing the graph).
    if is_first:
      self.save_checkpoints(env_step_int, use_common=True, save_basic=True)
    else:
      self.save_checkpoints(env_step_int, use_common=True)

  def _initial_collect(self):
    """Collect initial experience before training begins."""
    logging.info('Collecting initial experience...')
    time_step_spec = ts.time_step_spec(self._env.observation_spec())
    random_policy = random_py_policy.RandomPyPolicy(time_step_spec,
                                                    self._env.action_spec())
    time_step = self._env.reset()
    for worker in self._worker_names:
      logging.info('collecting experience for %s', worker)
      while self._replay_buffer[worker].size < self._initial_collect_steps:
        if game_over(self._env):
          time_step = self._env.reset()
        action_step = random_policy.action(time_step)
        next_time_step = self._env.step(action_step.action)
        self._store_to_rb(
            trajectory.from_transition(time_step, action_step, next_time_step),
            worker)
        time_step = next_time_step
    logging.info('Done.')

  def _initial_eval(self, sess):
    """Initial eval."""
    for worker in self._agents:
      self._online_eval_parallel(sess, self._select_py_policies[worker],
                                 self._env_select, self._update_policy_period,
                                 self._selection_metrics)
      if self._use_bandit or self._pbt:
        reward = self._selection_metrics[0].result()
        self._bandit_arm_q[worker].add_to_buffer(reward, update_time=False)
        logging.info('initial q=%s', [
            self._bandit_arm_q[worker].result('most_recent')
            for worker in self._agents
        ])
      self._behavior_metrics[worker][2].add_to_buffer(
          [self._selection_metrics[0].result()])
      self._behavior_metrics[worker][3].add_to_buffer(
          [self._selection_metrics[1].result()])

  def _run_episode(self,
                   sess,
                   env,
                   policy,
                   worker_name,
                   metrics=None,
                   train=False,
                   collect=False):
    """Run a single episode."""
    logging.info('running episode %d for agent %s',
                 self._episode_metric[worker_name].result() + 1, worker_name)
    num_steps = 0
    time_step = env.reset()
    while True:
      with self._collect_timer[worker_name]:
        time_step, train_traj, metric_traj = self._run_step(
            sess, env, time_step, policy)
        num_steps += 1

        if collect:
          self._store_to_rb(train_traj, worker_name)
        self._update_metrics(metrics, metric_traj)

      if train and not self._freeze_before_select:
        with self._train_timer[worker_name]:
          train_step, _ = self._train_one_step(sess, worker_name)
      if train_step:
        self._maybe_time_train(train_step, worker_name)
      if game_over(env):
        break
    return num_steps

  def _train_one_step(self, sess, worker_name):
    """Train one step."""
    env_step = self._env_steps_metric[worker_name].result()
    # if self._use_bandit or self._pbt:
    #   self._update_metrics([self._bandit_reward_metric], metric_traj)
    train_step = None
    agent = self._agents[worker_name]
    if env_step % agent.update_period == 0:
      train_step, loss = agent.train_one_step(
          sess, self._train_file_writers[worker_name])
      self._maybe_log_train(train_step, loss, worker_name)
      add_summary(self._train_file_writers[worker_name],
                  'TrainStep/' + worker_name, train_step, env_step)
      self._maybe_record_behavior_summaries(env_step, worker_name)
    return train_step, env_step

  def _update_eval(self, sess):
    del sess
    for worker in self._worker_names:
      reward = self._behavior_metrics[worker][2].result()
      self._bandit_arm_q[worker](reward)
    logging.info('bandit updated:%s', [
        self._bandit_arm_q[worker].result('most_recent')
        for worker in self._worker_names
    ])

  def _maybe_record_behavior_summaries(self, env_step, worker_name):
    """Record summaries if env_step is a multiple of summary_interval."""
    if env_step % self._summary_interval == 0:
      for metric in self._behavior_metrics[worker_name]:
        add_summary(self._train_file_writers[worker_name],
                    'Metrics/' + metric.name, metric.result(), env_step)

  def _maybe_time_train(self, train_step, worker_name):
    """Maybe time train."""
    if train_step % self._log_interval == 0:
      steps_per_sec = ((train_step - self._timed_at_step[worker_name]) /
                       (self._collect_timer[worker_name].value() +
                        self._train_timer[worker_name].value()))
      add_summary(self._train_file_writers[worker_name], 'train_steps_per_sec',
                  steps_per_sec, train_step)
      logging.info('%.3f steps/sec', steps_per_sec)
      logging.info(
          '%s', 'collect_time = {}, train_time = {}'.format(
              self._collect_timer[worker_name].value(),
              self._train_timer[worker_name].value()))
    self._timed_at_step[worker_name] = train_step
    self._collect_timer[worker_name].reset()
    self._train_timer[worker_name].reset()

  def _maybe_log_and_reset_timer(self):
    logging.info(
        'iteration time: %s', 'pbt_time = {}, checkpoint_time = {}'.format(
            self._pbt_timer.value(), self._checkpoint_timer.value()))
    self._pbt_timer.reset()
    self._checkpoint_timer.reset()


@gin.configurable
class PBTController(Runner):
  """Train and evaluate DQN on Atari."""

  def __init__(  # pylint: disable=dangerous-default-value
      self,
      num_iterations=200,
      initial_collect_steps=80000,  # ALE frames
      replay_buffer_capacity=400000,
      update_policy_iteration=10,  # train episode between switching policy
      ucb_coeff=1.96,
      train_steps_per_iteration=1000000,  # ALE frames
      batch_size=32,
      online_eval_use_train=True,
      epsilon_decay_selection=200,
      epsilon_selection=0.01,
      bandit_discount=1.0,
      bandit_ucb_coeff=2.0,
      bandit_buffer_size=80,
      eval_episode_select=50,  # ALE frames
      eval_parallel_size=25,
      # select_policy_way='controller',
      pbt=False,
      pbt_period=100,
      pbt_exploit_way='uniform',
      pbt_low='ucb',
      pbt_high='lcb',
      pbt_update_requirement=10,
      push_when_eval=False,
      pbt_percent_low=0.2,
      pbt_percent_top=0.2,
      pbt_drop_prob=0.8,
      pbt_mutation_rate=0.8,
      pbt_mutate_list=['lr', 'edecay'],
      pbt_perturb_factors=[0.8, 1.2],
      **kwargs):
    super(PBTController, self).__init__(**kwargs)
    self._num_iterations = num_iterations
    self._update_policy_period = update_policy_iteration
    self._ucb_coeff = ucb_coeff
    self._initial_collect_steps = initial_collect_steps / ATARI_FRAME_SKIP
    self._online_eval_use_train = online_eval_use_train
    self._use_bandit = pbt
    self._bandit_discount = bandit_discount
    self._bandit_ucb_coeff = bandit_ucb_coeff
    self._bandit_buffer_size = bandit_buffer_size
    self._pbt = pbt
    self._pbt_period = pbt_period
    self._pbt_exploit_way = pbt_exploit_way
    self._pbt_low = pbt_low
    self._pbt_high = pbt_high
    self._pbt_update_requirement = pbt_update_requirement
    self._push_when_eval = push_when_eval
    self._pbt_percent_low = pbt_percent_low
    self._pbt_percent_top = pbt_percent_top
    self._pbt_drop_prob = pbt_drop_prob
    self._pbt_mutation_rate = pbt_mutation_rate
    self._pbt_mutate_list = pbt_mutate_list
    self._pbt_perturb_factors = pbt_perturb_factors
    self._pbt_id = py_metrics.CounterMetric(name='pbt_id')
    if self._pbt:
      self._train_episode_per_iteration = pbt_period
    else:
      self._train_episode_per_iteration = int(train_steps_per_iteration /
                                              (ATARI_FRAME_SKIP * 1200))
    self._eval_episode_select = eval_episode_select  # / ATARI_FRAME_SKIP
    self._eval_parallel_size = eval_parallel_size
    self._epsilon_decay_selection = epsilon_decay_selection
    self._epsilon_selection = epsilon_selection
    self._batch_size = batch_size
    self._replay_buffer_capacity = replay_buffer_capacity

    assert not FLAGS.enable_functions
    for _ in range(self._num_agents):
      self._pbt_id()
    self._env_steps_metric = {}
    self._episode_metric = {}
    self._step_metrics = {}
    self._behavior_metrics = {}
    self._replay_buffer = {}
    self._rb_checkpointer = {}
    self._train_file_writers = {}

    self._env_select = parallel_py_environment.ParallelPyEnvironment([
        lambda: suite_atari.load(  # pylint: disable=g-long-lambda
            self._env_name,
            max_episode_steps=self._max_episode_frames / ATARI_FRAME_SKIP,
            gym_env_wrappers=suite_atari.
            DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)  # pylint: disable=g-long-lambda
    ] * self._eval_parallel_size)  # pylint: disable=g-long-lambda

    self._selection_metrics = [
        new_pymetrics.DistributionReturnMetric(
            name='SelectAverageReturn',
            buffer_size=np.inf,
            batch_size=self._eval_parallel_size),
        new_pymetrics.DistributionEpisodeLengthMetric(
            name='SelectAverageEpisodeLength',
            buffer_size=np.inf,
            batch_size=self._eval_parallel_size),
    ]

    for worker in self._worker_names:
      if not tf.gfile.Exists(os.path.join(self._train_dir, worker)):
        tf.gfile.MakeDirs(os.path.join(self._train_dir, worker))
      self._train_file_writers[worker] = tf.summary.FileWriter(
          os.path.join(self._train_dir, worker))
      self._env_steps_metric[worker] = py_metrics.EnvironmentSteps(
      )  # name='EnvironmentSteps_'+worker
      self._episode_metric[worker] = py_metrics.NumberOfEpisodes(
      )  # name='NumberOfEpisodes_'+worker
      self._step_metrics[worker] = [
          self._episode_metric[worker], self._env_steps_metric[worker]
      ]
      # The rolling average of the last 10 episodes.
      self._behavior_metrics[worker] = self._step_metrics[worker] + [
          py_metrics.AverageReturnMetric(
              buffer_size=10),  # name='AverageReturn_'+worker,
          py_metrics.AverageEpisodeLengthMetric(
              buffer_size=10),  # name='AverageEpisode_'+worker,
      ]
      self._replay_buffer[worker] = (
          py_hashed_replay_buffer.PyHashedReplayBuffer(
              data_spec=self._data_spec, capacity=replay_buffer_capacity))
      logging.info('replay_buffer_capacity: %s', replay_buffer_capacity)

    self.build_graph_and_assign_device()
    self.create_pypolicy_and_train_op()

    if self._use_bandit or self._pbt:
      self._bandit_arm_q = {}
      for worker in self._worker_names:
        self._bandit_arm_q[worker] = new_pymetrics.QMetric(
            name='QMetric_' + worker, buffer_size=self._bandit_buffer_size)

    self.update_train_bandit_checkpointer(
        update_bandit=(self._use_bandit or self._pbt), use_common=True)
    self.update_rb_metric_checkpointer(use_common=True)
    self.update_train_checkpointer_large(use_common=True)

  def iterdone(self):
    done = True
    for worker in self._worker_names:
      done = done & tf.gfile.Exists(
          os.path.join(self._train_dir, worker, 'IterDone'))
    return done

  def run(self):
    """Run."""
    if self._pbt:
      config = tf.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      self._count_reevaluate = 0
      num_iter = 0
      while num_iter < self._num_iterations:
        while not self.iterdone():
          logging.info('waiting for trainers to complete')
          time.sleep(10)
        logging.info('starting controller')
        self.update_train_bandit_checkpointer(
            update_bandit=(self._use_bandit or self._pbt), use_common=True)
        self.update_rb_metric_checkpointer(use_common=True)
        with tf.Session(config=config) as sess:
          self._initialize_graph(sess)
          if self._iteration_metric.result() > self._num_iterations:
            break
          with self._pbt_timer:
            logging.info('pbt exploitation start',)
            self.exploit(sess, way=self._pbt_exploit_way)
            self._iteration_metric()
            ep_step = self._iteration_metric.result()
            ep_step_int = int(ep_step)
            self.save_hparam(ckpt_step=ep_step_int)
          num_iter = self._iteration_metric.result()
          with self._checkpoint_timer:
            self.update_train_bandit_checkpointer(
                update_bandit=True, use_common=False)
            self.update_rb_metric_checkpointer(use_common=False)
            self.update_train_checkpointer_large(use_common=False)
            logging.info('saving checkpoints')
            self.save_checkpoints(
                ep_step_int, use_common=False, save_basic=True)
            self._train_checkpointer_large.save(checkpoint_number=ep_step_int)
            # self.record_log_metric()
        for worker in self._worker_names:
          tf.gfile.Remove(os.path.join(self._train_dir, worker, 'IterDone'))
        with tf.gfile.GFile(os.path.join(self._train_dir, 'PBTDone'),
                            'w') as writer:
          writer.write('PBTDone')
    else:
      return

  def _initialize_graph(self, sess, is_first=True):
    """Initialize the graph for sess."""
    self._basic_checkpointer.initialize_or_restore(sess)
    for worker in self._worker_names:
      self._rb_checkpointer[worker].initialize_or_restore(sess)
      self._train_checkpointer[worker].initialize_or_restore(sess)
      if self._use_bandit or self._pbt:
        self._bandit_checkpointer[worker].initialize_or_restore(sess)
      self._metric_checkpointer[worker].initialize_or_restore(sess)

    common.initialize_uninitialized_variables(sess)
    sess.run(self._init_agent_op)
    logging.info('bandit:%s', [
        self._bandit_arm_q[worker].result('most_recent')
        for worker in self._worker_names
    ])
    for worker in self._worker_names:
      logging.info('bandit buffer for %s:%s', worker,
                   self._bandit_arm_q[worker]._buffer._buffer)  # pylint: disable=protected-access

    # env_step = self._episode_metric[self._worker_names[0]].result()
    # env_step_int = int(env_step)

    if is_first:
      self._timed_at_step = {}
      self._collect_timer = {}
      self._eval_timer = timer.Timer()
      self._train_timer = {}
      self._pbt_timer = {}
      self._checkpoint_timer = {}
      for worker in self._worker_names:
        self._collect_timer[worker] = timer.Timer()
        self._train_timer[worker] = timer.Timer()
      self._pbt_timer = timer.Timer()
      self._checkpoint_timer = timer.Timer()

    for worker in self._worker_names:
      self._timed_at_step[worker] = sess.run(
          self._agents[worker].train_step_counter)

  def _update_eval(self, sess):
    del sess
    for worker in self._worker_names:
      reward = self._behavior_metrics[worker][2].result()
      self._bandit_arm_q[worker](reward)
    logging.info('bandit updated:%s', [
        self._bandit_arm_q[worker].result('most_recent')
        for worker in self._worker_names
    ])

  def exploit(self, sess, way='uniform'):
    """Exploit."""
    bottom_table = np.asarray([
        self._bandit_arm_q[worker].result('most_recent')
        for worker in self._worker_names
    ])
    top_table = np.asarray([
        self._bandit_arm_q[worker].result('most_recent')
        for worker in self._worker_names
    ])
    logging.info('bottom table = %s, top table=%s', bottom_table, top_table)
    top_list = np.argsort(
        top_table)[-int(len(bottom_table) * self._pbt_percent_top):]
    bottom_list = np.argsort(
        bottom_table)[:int(len(bottom_table) * self._pbt_percent_low)]
    if way == 'weighted':
      top_prob = softmax(top_table[top_list] - np.median(top_table))
      bottom_norm = (np.median(bottom_table) - bottom_table[bottom_list]) / (
          np.quantile(bottom_table, 0.75) - np.quantile(bottom_table, 0.25))
      bottom_prob = sigmoid(
          bottom_norm - 1.0, coeff=0.3, truncate=self._pbt_drop_prob)
    elif way == 'uniform':
      top_prob = np.ones_like(top_list, dtype=np.float32) / len(top_table)
      bottom_prob = np.ones_like(
          bottom_list, dtype=np.float32) * self._pbt_drop_prob
    for i, target_id in enumerate(bottom_list):
      if np.random.binomial(1, bottom_prob[i]):
        target_agent = self._worker_names[target_id]
        source_id = top_list[np.where(
            np.random.multinomial(1, top_prob) == 1)[0][0]]
        source_agent = self._worker_names[source_id]
        logging.info('exploit agent %s(%s) to worker %s(%s), exploration start',
                     source_agent, self._agent_names[source_agent],
                     target_agent, self._agent_names[target_agent])
        new_hparam = self._mutate(self._hparams[source_agent],
                                  self._pbt_mutation_rate,
                                  self._pbt_mutate_list)
        self._hparams[target_agent] = new_hparam
        self._agent_names[target_agent] = new_hparam.name
        self.update_rb(target_agent, source_agent)
        new_agent = self.create_or_copy_agent(
            new_hparam,
            qtype=self._dqn_type,
            device=self._device_name[target_agent],
            sess=sess,
            parent_agent=self._agents[source_agent],
            current_worker=target_agent,
            do_copy=True)
        self._collect_py_policies[target_agent] = py_tf_policy.PyTFPolicy(
            new_agent.collect_policy)
        self._select_py_policies[target_agent] = py_tf_policy.PyTFPolicy(
            new_agent.collect_policy)
        # new_agent.train(self._rb_iterator[target_agent])
        self._agents[target_agent] = new_agent
        self._bandit_arm_q[target_agent] = copy.deepcopy(
            self._bandit_arm_q[source_agent])
        self._env_steps_metric[target_agent] = copy.deepcopy(
            self._env_steps_metric[source_agent])
        self._episode_metric[target_agent] = copy.deepcopy(
            self._episode_metric[source_agent])
        self._step_metrics[target_agent] = [
            self._episode_metric[target_agent],
            self._env_steps_metric[target_agent]
        ]
        self._behavior_metrics[
            target_agent] = self._step_metrics[target_agent] + [
                copy.deepcopy(self._behavior_metrics[source_agent]
                              [2]),  # name='AverageReturn_'+worker,
                copy.deepcopy(self._behavior_metrics[source_agent]
                              [3])  # name='AverageEpisode_'+worker,
            ]
        self._bandit_arm_q[target_agent].rename('QMetric_' + target_agent)
        logging.info('created new worker agent %s with hyperparam %s',
                     target_agent, self._agent_names[target_agent])

  def update_rb(self, target_worker, source_worker):
    """Update rb."""
    logging.info('copying rb buffer')
    del self._replay_buffer[target_worker]
    gc.collect()
    self._replay_buffer[target_worker] = (
        py_hashed_replay_buffer.PyHashedReplayBuffer(
            data_spec=self._data_spec, capacity=self._replay_buffer_capacity))
    items = self._replay_buffer[source_worker].gather_all()
    for item in items:
      self._replay_buffer[target_worker].add_batch(
          nest_utils.batch_nested_array(item))
    device = self._device_name[target_worker]
    with tf.device('/cpu:0'):
      ds = self._replay_buffer[target_worker].as_dataset(
          sample_batch_size=self._batch_size,
          num_steps=self._hparams[target_worker].n_step_update + 1)
      ds = ds.prefetch(4)
      ds = ds.apply(tf.data.experimental.prefetch_to_device(device))
    with tf.device(device):
      self._rb_iterator[target_worker] = tf.data.make_one_shot_iterator(
          ds).get_next()

  def _mutate(self, parent_hparam, mutation_rate, mutate_list):
    """Mutate."""
    new_hparam = copy.deepcopy(parent_hparam)
    suffix = []
    # try:
    # parent_id = parent_hparam.name.split('-')[1]
    parent_id = '0'
    for hp in mutate_list:
      if np.random.binomial(1, mutation_rate):
        coeff = np.random.choice(self._pbt_perturb_factors)
        new_val = parent_hparam.get(hp) * coeff
        if hp == 'edecay':
          new_hparam.set_hparam(hp, int(new_val))
          suffix.append(hp + str(new_val))
        else:
          new_hparam.set_hparam(hp, new_val)
          suffix.append(hp + '{:.6f}'.format(new_val))
    self._pbt_id()
    new_hparam.name = '-'.join([
        parent_hparam.name.split('-')[0],
        str(self._pbt_id.result()), parent_id
    ] + suffix)
    return new_hparam


@gin.configurable
class EvalRunner(Runner):
  """evaluate DQN on Atari."""

  def __init__(
      self,
      ucb_coeff=1.96,
      num_iterations=200,
      eval_episode_per_iteration=100,  # ALE frames
      eval_parallel_size=20,
      eval_epsilon_greedy=0.0,
      eval_interval_secs=60,
      eval_agents=None,
      **kwargs):
    super(EvalRunner, self).__init__(**kwargs)
    self._num_iterations = num_iterations
    self._eval_interval_secs = eval_interval_secs
    self._eval_parallel_size = eval_parallel_size
    self._eval_episode_per_iteration = eval_episode_per_iteration
    self._eval_epsilon_greedy = eval_epsilon_greedy
    self._eval_agents = eval_agents
    self._ucb_coeff = ucb_coeff
    self._env_eval = parallel_py_environment.ParallelPyEnvironment([
        lambda: suite_atari.load(  # pylint: disable=g-long-lambda
            self._env_name,
            max_episode_steps=self._max_episode_frames / ATARI_FRAME_SKIP,
            gym_env_wrappers=suite_atari.
            DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
    ] * self._eval_parallel_size)
    if eval_agents:
      self._worker_names = eval_agents

    self._eval_metrics = {}
    self._eval_py_policies = {}

    for worker in self._worker_names:
      # Use the policy directly for eval.
      self._eval_metrics[worker] = [
          new_pymetrics.DistributionReturnMetric(
              name='EvalAverageReturn_' + worker, buffer_size=np.inf),
          new_pymetrics.DistributionEpisodeLengthMetric(
              name='EvalAverageEpisodeLength_' + worker, buffer_size=np.inf)
      ]
    # self.build_graph_and_assign_device()
    # self.update_train_bandit_checkpointer(update_bandit=False,use_common=True)

  def _initialize_graph(self, sess, checkpoint_path):
    """Initialize the graph for sess."""
    self._train_checkpointer_large._checkpoint.restore(  # pylint: disable=protected-access
        checkpoint_path).initialize_or_restore(sess)
    common.initialize_uninitialized_variables(sess)

    sess.run(self._init_agent_op)
    self._eval_timer = timer.Timer()

  def traindone(self):
    return tf.gfile.Exists(os.path.join(self._train_dir, 'TrainDone'))

  def run(self):
    """Execute the eval loop."""
    self._eval_file_writer = {}
    for worker_name in self._eval_agents:
      if not tf.gfile.Exists(os.path.join(self._train_dir, worker_name)):
        tf.gfile.MakeDirs(os.path.join(self._train_dir, worker_name))
      self._eval_file_writer[worker_name] = tf.summary.FileWriter(
          os.path.join(self._train_dir, worker_name))
    for checkpoint_path in tf.train.checkpoints_iterator(
        self._train_dir,
        min_interval_secs=self._eval_interval_secs,
        timeout_fn=self.traindone):
      logging.info('find new checkpoint %s', checkpoint_path)
      step = checkpoint_path.split('ckpt-')[1]
      hparam_file = os.path.join(self._hparam_dir,
                                 'hparam-{}.json'.format(step))
      if tf.gfile.Exists(hparam_file):
        logging.info('hparam file %s found, loading', hparam_file)
        self._load_hparam(hparam_file)
        logging.info('agent_names:%s', self._agent_names)
        assert not FLAGS.enable_functions
        tf.reset_default_graph()
        gc.collect()
        self.build_graph_and_assign_device()
        fail = True
        while fail:
          # try:
          self.update_train_checkpointer_large(use_common=True)
          fail = False
          # except:
          #   logging.info('restoring ckpt failed,  try again')
          #   time.sleep(10)
        logging.info(
            'new checkpoint manager at ckpt:%s, hparam at ckpt:%s',
            self._train_checkpointer_large._manager.latest_checkpoint,  # pylint: disable=protected-access
            step)
      else:
        logging.info('hparam file not found for %s, try next ckpt', hparam_file)
        continue

      with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Initialize the graph.
        self._initialize_graph(sess, checkpoint_path)
        logging.info('Starting evaluation')
        for worker_name in self._eval_agents:
          logging.info('evaluating agent %s', worker_name)
          if worker_name in self._worker_names:
            agent = self._agents[worker_name]
            with tf.device(self._device_name[worker_name]):
              self._eval_py_policies[worker_name] = py_tf_policy.PyTFPolicy(
                  agent.policy)
            self._online_eval_parallel(sess,
                                       self._eval_py_policies[worker_name],
                                       self._env_eval,
                                       self._eval_episode_per_iteration,
                                       self._eval_metrics[worker_name])
            train_step = sess.run(agent.train_step_counter)
            for metric in self._eval_metrics[worker_name]:
              add_summary(self._eval_file_writer[worker_name],
                          metric.name.split('_')[0] + '(mean)/' + metric.name,
                          metric.result(), train_step)
              write_csv(self._csv_dir, metric.name + '_mean', metric.result(),
                        train_step, self._iteration_metric.result())
              write_csv(self._csv_dir, metric.name + '_std',
                        metric.result('std'), train_step,
                        self._iteration_metric.result())
              log_metric(metric, prefix='Eval/Metrics')
          else:
            logging.info('agent %s does not exist!', worker_name)
        self._iteration_metric()
        logging.info('Finished evaluation')
