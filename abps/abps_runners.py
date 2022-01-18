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

import abc
import copy
import gc
import json
import os

from absl import flags
from absl import logging

import gin
import numpy as np
import tensorflow.compat.v1 as tf

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
flags.DEFINE_string('pbt_low', None, 'how to choose low value agents')
flags.DEFINE_string('pbt_high', None, 'how to choose high value agents')

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
flags.DEFINE_integer(
    'eval_episode_per_iteration', None,
    'Number of ALE frames to run through for each iteration '
    'of evaluation.')
flags.DEFINE_integer(
    'eval_interval_secs', None,
    'interval of waiting time for evaluator to detect new ckpt')
flags.DEFINE_integer('epsilon_decay_selection', 400,
                     'Period over which to decay epsilon, for Bandit')
flags.DEFINE_integer('update_policy_iteration', 10,
                     'number of train episode between change policy')
flags.DEFINE_integer('eval_parallel_size', None,
                     'number of process used for parallelization')
flags.DEFINE_integer('num_worker', None, 'number of workers')
flags.DEFINE_integer('pbt_period', 10, 'number of abps runs between pbt')
flags.DEFINE_integer('bandit_buffer_size', None,
                     'size of the buffer window size')

flags.DEFINE_float('eval_epsilon_greedy', 0.0,
                   'epsilon for the policy when doing evaluation')
flags.DEFINE_float('learning_rate', None, 'Learning rate')
flags.DEFINE_float('ucb_coeff', 5.0, 'coefficient for UCB in best online')
flags.DEFINE_float('bandit_ucb_coeff', 5.0, 'coefficient for UCB in bandit')
flags.DEFINE_float('pbt_percent_low', 0.2, 'percent of agents to be replaced')
flags.DEFINE_float('pbt_percent_top', 0.4, 'percent of agents as good')

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


# Fix the place holder.
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
      enable_functions=False,
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
      num_worker=2,
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
      enable_functions: Enable functions.
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
        data
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
    self._enable_functions = enable_functions
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
    elif tf.gfile.Exists(os.path.join(self._home_dir, 'hparam-0.json')):
      hparam_path = os.path.join(self._home_dir, 'hparam-0.json')
      self._load_hparam(hparam_path)
      if not self._do_eval:
        tf.gfile.Copy(
            hparam_path,
            os.path.join(self._hparam_dir, 'hparam-0.json'),
            overwrite=True)
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
        # hparam.edecay = int(hparam.edecay
        # / ATARI_FRAME_SKIP / hparam.update_period)
        # hparam.target_update_period = int(hparam.target_update_period
        # / ATARI_FRAME_SKIP / hparam.update_period)
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

  def create_pypolicy_and_train_op(self):
    """Create pypolicy and train op."""
    self._collect_py_policies = {}
    self._select_py_policies = {}
    self._rb_iterator = {}
    self._select_step = tf.Variable(
        0, dtype=tf.int64, trainable=False, name='bandit_step')
    self._select_epsilon = tf.train.polynomial_decay(
        1.0,
        self._select_step,
        self._epsilon_decay_selection,
        end_learning_rate=self._epsilon_selection)
    for i in range(len(self._worker_names)):
      worker = self._worker_names[i]
      agent = self._agents[worker]
      device = self._device_name[worker]
      with tf.device('/cpu:0'):
        ds = self._replay_buffer.as_dataset(
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
      # try
      with tf.gfile.Open(hparam_path) as json_file:
        self._hparam_dict = json.load(json_file)
      read = True

    self._num_agents = len(self._hparam_dict)
    self._worker_names = np.asarray(sorted(list(self._hparam_dict.keys())))
    logging.info('found Hparam file at: %s, loading', hparam_path)
    # set unspecified hparams to default values
    for worker_name, d in self._hparam_dict.items():
      hparam = hparam_lib.HParams(**self._default_hparams)
      for k in d:
        if k in ['conv', 'fc']:
          d[k] = list(convert_list_to_tuple(d[k]))
      hparam.override_from_dict(d)
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

  def update_train_bandit_checkpointer(self,
                                       update_bandit=True,
                                       use_common=False):
    """Update train bandit checkpointer."""
    all_iterable = {worker: self._agents[worker] for worker in self._agents}
    if use_common:
      self._train_checkpointer = common.Checkpointer(
          ckpt_dir=self._train_dir,
          max_to_keep=self._max_ckpt,
          behavior_index=self._behavior_index,
          selection_step=self._select_step,
          **all_iterable)
      if update_bandit:
        self._bandit_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(self._train_dir, 'bandit'),
            max_to_keep=self._max_ckpt,
            pbt_metrics=metric_utils.MetricsGroup(
                [self._bandit_arm_q[worker] for worker in self._worker_names] +
                [self._bandit_reward_metric, self._pbt_id], 'bandit_metrics'))
    else:
      self._train_checkpointer = tf.train.CheckpointManager(
          tf.train.Checkpoint(
              behavior_index=self._behavior_index,
              selection_step=self._select_step,
              **all_iterable),
          directory=self._train_dir,
          max_to_keep=self._max_ckpt)
      if update_bandit:
        self._bandit_checkpointer = tf.train.CheckpointManager(
            tf.train.Checkpoint(
                pbt_metrics=metric_utils.MetricsGroup([
                    self._bandit_arm_q[worker] for worker in self._worker_names
                ] + [self._bandit_reward_metric, self._pbt_id],
                                                      'bandit_metrics')),
            directory=os.path.join(self._train_dir, 'bandit'),
            max_to_keep=self._max_ckpt)

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
        # deal with the case where restart in place is needed
        # ...next_time_step_mid = change_from_last_to_mid(next_time_step)
        # ...next_time_step = ts.restart(next_time_step.observation)
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

  def _run_episode(self,
                   sess,
                   env,
                   policy,
                   metrics=None,
                   train=False,
                   collect=False):
    """Run a single episode."""
    num_steps = 0
    time_step = env.reset()
    while True:
      with self._collect_timer:
        time_step, train_traj, metric_traj = self._run_step(
            sess, env, time_step, policy)
        num_steps += 1

        if collect:
          self._store_to_rb(train_traj)
        self._update_metrics(metrics, metric_traj)

      if train and not self._freeze_before_select:
        with self._train_timer:
          train_step, _ = self._train_one_step(sess, metric_traj)
      if train_step:
        self._maybe_time_train(train_step)
      if game_over(env):
        break
    return num_steps

  @abc.abstractmethod
  def _train_one_step(self, sess, metric_traj):
    """Abstract method for train step."""
    pass


@gin.configurable
class TrainRunner(Runner):
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
      select_policy_way=None,
      epsilon_decay_selection=400,
      epsilon_selection=0.01,
      bandit_discount=1.0,
      bandit_ucb_coeff=5.0,
      bandit_buffer_size=80,
      eval_episode_select=50,  # ALE frames
      eval_parallel_size=25,
      pbt=False,
      pbt_period=10,
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
    super(TrainRunner, self).__init__(**kwargs)
    self._num_iterations = num_iterations
    self._update_policy_period = update_policy_iteration
    self._train_steps_per_iteration = (
        train_steps_per_iteration / ATARI_FRAME_SKIP)
    self._ucb_coeff = ucb_coeff
    self._initial_collect_steps = initial_collect_steps / ATARI_FRAME_SKIP
    self._select_policy_way = select_policy_way
    self._online_eval_use_train = online_eval_use_train
    self._use_bandit = 'bandit' in self._select_policy_way
    self._bandit_discount = bandit_discount
    self._bandit_ucb_coeff = bandit_ucb_coeff
    self._bandit_buffer_size = bandit_buffer_size
    self._pbt = pbt
    self._pbt_period = pbt_period
    self._pbt_exploit_way = pbt_exploit_way
    self._pbt_low = pbt_low
    self._pbt_high = pbt_high
    self._pbt_update_requirement = pbt_update_requirement
    # self._pbt_percent = pbt_percent
    self._push_when_eval = push_when_eval
    self._pbt_percent_low = pbt_percent_low
    self._pbt_percent_top = pbt_percent_top
    self._pbt_drop_prob = pbt_drop_prob
    self._pbt_mutation_rate = pbt_mutation_rate
    self._pbt_mutate_list = pbt_mutate_list
    self._pbt_perturb_factors = pbt_perturb_factors
    self._pbt_id = py_metrics.CounterMetric(name='pbt_id')
    self._eval_episode_select = eval_episode_select  # / ATARI_FRAME_SKIP
    self._eval_parallel_size = eval_parallel_size
    self._epsilon_decay_selection = epsilon_decay_selection
    self._epsilon_selection = epsilon_selection
    self._batch_size = batch_size

    assert not FLAGS.enable_functions
    for _ in range(self._num_agents):
      self._pbt_id()

    with (tf.device('/gpu:' +
                    str((len(self._worker_names) + 1) % len(self._devices)))
          if self._devices else tf.device('/cpu:0')):
      self._env_steps_metric = py_metrics.EnvironmentSteps()
      self._episode_metric = py_metrics.NumberOfEpisodes()
      self._step_metrics = [self._episode_metric, self._env_steps_metric]
      # The rolling average of the last 10 episodes.
      self._behavior_metrics = self._step_metrics + [
          py_metrics.AverageReturnMetric(buffer_size=10),
          py_metrics.AverageEpisodeLengthMetric(buffer_size=10),
      ]
      self._env_select = parallel_py_environment.ParallelPyEnvironment([
          lambda: suite_atari.load(  # pylint: disable=g-long-lambda
              self._env_name,
              max_episode_steps=self._max_episode_frames / ATARI_FRAME_SKIP,
              gym_env_wrappers=suite_atari.
              DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
      ] * self._eval_parallel_size)
      self._replay_buffer = (
          py_hashed_replay_buffer.PyHashedReplayBuffer(
              data_spec=self._data_spec, capacity=replay_buffer_capacity))
      logging.info('replay_buffer_capacity: %s', replay_buffer_capacity)

    self._rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(self._train_dir, 'replay_buffer'),
        max_to_keep=10,
        replay_buffer=self._replay_buffer)

    self._selection_metrics = {}
    for worker in self._worker_names:
      self._selection_metrics[worker] = [
          new_pymetrics.DistributionReturnMetric(
              name='SelectAverageReturn_' + worker,
              buffer_size=np.inf,
              batch_size=self._eval_parallel_size),
          new_pymetrics.DistributionEpisodeLengthMetric(
              name='SelectAverageEpisodeLength_' + worker,
              buffer_size=np.inf,
              batch_size=self._eval_parallel_size),
      ]

    self.build_graph_and_assign_device()
    self.create_pypolicy_and_train_op()
    logging.info('method %s, use bandit=%s, use pbt = %s',
                 self._select_policy_way, self._use_bandit, self._pbt)

    if self._use_bandit or self._pbt:
      self._bandit_arm_q = {}
      for worker in self._worker_names:
        self._bandit_arm_q[worker] = new_pymetrics.QMetric(
            name='QMetric_' + worker, buffer_size=self._bandit_buffer_size)
      self._bandit_reward_metric = py_metrics.AverageReturnMetric(
          buffer_size=np.inf)
      self.update_train_bandit_checkpointer(update_bandit=True, use_common=True)
    else:
      self.update_train_bandit_checkpointer(
          update_bandit=False, use_common=True)

    self.create_policy_metrics_and_checkpointer()

  def run(self):
    """Execute the train loop."""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self._count_reevaluate = 0
    num_iter = 0
    while num_iter < self._num_iterations:
      with tf.Session(config=config) as sess:
        # Initial evaluation for choosing first agent
        if num_iter == 0:
          # Initialize the graph.
          self._initialize_graph(sess)
        else:
          self._initialize_graph(sess, is_first=False)
        if self._iteration_metric.result() == 0:
          # Initial collect
          self._initial_collect()
          # Initialize policy selection
          env_step = int(self._env_steps_metric.result())
          self._initial_eval(sess, self._select_policy_way, env_step)
        if self._iteration_metric.result() > self._num_iterations:
          break
        # Train phase
        num_steps = 0
        while num_steps < self._train_steps_per_iteration:
          num_steps += self._run_episode(
              sess,
              env=self._env,
              policy=self._behavior_policy,
              metrics=self._behavior_metrics,
              train=True,
              collect=True)
          env_step = self._env_steps_metric.result()
          env_step_int = int(env_step)

          if self._episode_metric.result() % self._update_policy_period == 0:
            sess.run(self._select_step.assign(sess.run(self._select_step) + 1))
            if self._use_bandit or self._pbt:
              current_agent = self._worker_names[sess.run(self._behavior_index)]
              self._update_bandit(sess, self._select_policy_way, current_agent)
            if self._pbt and (sess.run(self._select_step) %
                              self._pbt_period == 0):
              with self._pbt_timer:
                logging.info(
                    'pbt exploitation start at select_step %d, env_step %d',
                    sess.run(self._select_step),
                    self._env_steps_metric.result())
                self.exploit(sess, way=self._pbt_exploit_way)
                self.save_hparam(ckpt_step=env_step_int, snapshot=True)
            # select next behavior policy
            with self._select_timer:
              worker_index = self._select_policy(sess, self._select_policy_way,
                                                 env_step)
              sess.run(self._behavior_index.assign(worker_index))
              logging.info('change policy to %s(%s) at env step=%d',
                           self._worker_names[worker_index],
                           self._agent_names[self._worker_names[worker_index]],
                           env_step)
              self._behavior_policy = self._collect_py_policies[
                  self._worker_names[worker_index]]
            with self._policy_timer:
              self.record_log_policy_metric(worker_index, env_step)
        self._iteration_metric()
        num_iter = self._iteration_metric.result()
        with self._checkpoint_timer:
          self.save_hparam(ckpt_step=env_step_int)
          logging.info('saving checkpoints')
          if self._pbt:
            self.update_train_bandit_checkpointer(
                update_bandit=(self._use_bandit or self._pbt))
            self.update_rb_policy_checkpointer()
            self._train_checkpointer.save(checkpoint_number=env_step_int)
            if self._use_bandit or self._pbt:
              self._bandit_checkpointer.save(checkpoint_number=env_step_int)
            self._rb_checkpointer.save(checkpoint_number=env_step_int)
            self._policy_checkpointer.save(checkpoint_number=env_step_int)
          else:
            self._train_checkpointer.save(global_step=env_step_int)
            if self._use_bandit or self._pbt:
              self._bandit_checkpointer.save(global_step=env_step_int)
            self._rb_checkpointer.save(global_step=env_step_int)
            self._policy_checkpointer.save(global_step=env_step_int)
          add_summary(self._train_file_writer,
                      'Metrics/' + self._iteration_metric.name,
                      self._iteration_metric.result(), env_step)
      self._maybe_log_and_reset_timer()
      if self._pbt:
        tf.reset_default_graph()
        logging.info('agent_list in current iteration:%s', self._agent_names)
        logging.info('hparam_list in current iteration:%s', self._hparams)
        self.build_graph_and_assign_device()
        self.create_pypolicy_and_train_op()
      self.update_train_bandit_checkpointer(
          update_bandit=(self._use_bandit or self._pbt), use_common=True)
      # self.update_rb_policy_checkpointer(use_common=True)

    with tf.gfile.GFile(os.path.join(self._train_dir, 'TrainDone'),
                        'w') as writer:
      writer.write('TrainDone')

  def _initialize_graph(self, sess, is_first=True):
    """Initialize the graph for sess."""
    if is_first:
      self._train_checkpointer.initialize_or_restore(sess)
      if self._use_bandit or self._pbt:
        self._bandit_checkpointer.initialize_or_restore(sess)
      self._rb_checkpointer.initialize_or_restore(sess)
      self._policy_checkpointer.initialize_or_restore(sess)
    else:
      # only reload train checkpointer if the job is not just restarted
      self._train_checkpointer.initialize_or_restore(sess)

    common.initialize_uninitialized_variables(sess)
    sess.run(self._init_agent_op)

    behavior_index = sess.run(self._behavior_index)
    self._behavior_policy = self._collect_py_policies[
        self._worker_names[behavior_index]]

    env_step = self._env_steps_metric.result()
    env_step_int = int(env_step)

    self._collect_timer = timer.Timer()
    self._eval_timer = timer.Timer()
    self._train_timer = timer.Timer()
    self._select_timer = timer.Timer()
    self._pbt_timer = timer.Timer()
    self._policy_timer = timer.Timer()
    self._checkpoint_timer = timer.Timer()

    self._timed_at_step = sess.run(
        self._agents[self._worker_names[0]].train_step_counter)

    # Call save to initialize the save_counter (need to do this before
    # finalizing the graph).
    if is_first:
      self._train_checkpointer.save(global_step=env_step_int)
      self._policy_checkpointer.save(global_step=env_step_int)
      self._rb_checkpointer.save(global_step=env_step_int)
      if self._use_bandit or self._pbt:
        self._bandit_checkpointer.save(global_step=env_step_int)

  def _initial_collect(self):
    """Collect initial experience before training begins."""
    logging.info('Collecting initial experience...')
    time_step_spec = ts.time_step_spec(self._env.observation_spec())
    random_policy = random_py_policy.RandomPyPolicy(time_step_spec,
                                                    self._env.action_spec())
    time_step = self._env.reset()
    while self._replay_buffer.size < self._initial_collect_steps:
      if game_over(self._env):
        time_step = self._env.reset()
      action_step = random_policy.action(time_step)
      next_time_step = self._env.step(action_step.action)
      self._store_to_rb(
          trajectory.from_transition(time_step, action_step, next_time_step))
      time_step = next_time_step
    logging.info('Done.')

  def _initial_eval(self, sess, method, env_step=None, push_value=False):
    """Initial eval."""
    del env_step
    for worker in self._agents:
      self._online_eval_parallel(sess, self._select_py_policies[worker],
                                 self._env_select, self._update_policy_period,
                                 self._selection_metrics[worker])
      if self._use_bandit or self._pbt:
        reward = self._selection_metrics[worker][0].result()
        if push_value:
          if 'discount' in method:
            self._bandit_arm_q[worker].add_to_buffer(
                reward, discount=self._bandit_discount)
            self._add_zero_to_bandit(
                worker, discount=self._bandit_discount, update_time=False)
          else:
            self._bandit_arm_q[worker](reward)
            self._add_zero_to_bandit(worker, update_time=False)
          sess.run(self._select_step.assign(sess.run(self._select_step) + 1))
        else:
          self._bandit_arm_q[worker].add_to_buffer(reward, update_time=False)
        logging.info(
            'initial q=%s',
            [self._bandit_arm_q[worker].result() for worker in self._agents])
    behavior_index = self._which_policy(sess, method)
    sess.run(self._behavior_index.assign(behavior_index))
    self._behavior_policy = self._collect_py_policies[
        self._worker_names[behavior_index]]
    self._behavior_metrics[2].add_to_buffer([
        self._selection_metrics[self._worker_names[behavior_index]][0].result()
    ])
    self._behavior_metrics[3].add_to_buffer([
        self._selection_metrics[self._worker_names[behavior_index]][1].result()
    ])

  def create_policy_metric_group(self, name):
    return [
        new_pymetrics.PolicyUsageFrequency(
            name='PolicyUsageFrequencyAccumulated_' + name, buffer_size=np.inf),
        new_pymetrics.PolicyUsageFrequency(
            name='PolicyUsageFrequencyRolling_' + name, buffer_size=10),
    ]

  def create_policy_metrics_and_checkpointer(self):
    """Create policy metrics and checkpointer."""
    self._policy_metrics = {}
    for i, worker in enumerate(self._worker_names):
      with tf.device(self._device_name[worker]):
        self._policy_metrics[worker] = self.create_policy_metric_group(worker)
    # Should use i + 1
    with (tf.device('/gpu:' + str((0 + 1) % len(self._devices)))
          if self._devices else tf.device('/cpu:0')):
      self._policy_metrics_hyper = {}
      for hp in ['lr', 'edecay', 'architect']:
        self._policy_metrics_hyper[hp] = {}
        if hp in ['lr', 'edecay']:
          low = np.log(self._hyper_range[hp][0] * 0.001)
          high = np.log(self._hyper_range[hp][1] * 100)
          histo = np.concatenate(
              ([-np.inf], np.arange(low, high, (high - low) / 10.0), [np.inf]))
          for i in range(len(histo) - 1):
            name = (hp + '_{:.5f}'.format(histo[i]) + '_' +
                    '{:.5f}'.format(histo[i + 1])).replace('-', '_').replace(
                        '.', '_')
            self._policy_metrics_hyper[hp][
                histo[i]] = self.create_policy_metric_group(name)
        else:
          for x in self._hyper_range[hp]:
            self._policy_metrics_hyper[hp][x] = self.create_policy_metric_group(
                hp + '_' + x)
    all_iterable = {
        (worker + '_policy'):  # pylint: disable=g-complex-comprehension
        metric_utils.MetricsGroup(self._policy_metrics[worker],
                                  worker + '_policy') for worker in self._agents
    }
    for hp in self._policy_metrics_hyper:
      for param in self._policy_metrics_hyper[hp].keys():
        all_iterable.update({
            (hp + '_' + '{:.5}'.format(param) + '_policy'):
                metric_utils.MetricsGroup(self._policy_metrics_hyper[hp][param],
                                          (hp + '_' + '{:.5}'.format(param) +
                                           '_policy').replace('.', '_').replace(
                                               '-', '_'))
        })
    self._policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(self._train_dir, 'policy'),
        max_to_keep=self._max_ckpt,
        behave_metrics=metric_utils.MetricsGroup(
            self._behavior_metrics + [self._iteration_metric],
            'behavior_metrics'),
        **all_iterable)

  def update_rb_policy_checkpointer(self, use_common=False):
    """Update rb policy checkpointer."""
    all_iterable = {
        (worker + '_policy'):  # pylint: disable=g-complex-comprehension
        metric_utils.MetricsGroup(self._policy_metrics[worker],
                                  worker + '_policy') for worker in self._agents
    }
    for hp in self._policy_metrics_hyper:
      for param in self._policy_metrics_hyper[hp].keys():
        all_iterable.update({
            (hp + '_' + '{:.5}'.format(param) + '_policy'):
                metric_utils.MetricsGroup(self._policy_metrics_hyper[hp][param],
                                          (hp + '_' + '{:.5}'.format(param) +
                                           '_policy').replace('.', '_').replace(
                                               '-', '_'))
        })
    if use_common:
      self._rb_checkpointer = common.Checkpointer(
          ckpt_dir=os.path.join(self._train_dir, 'replay_buffer'),
          max_to_keep=10,
          replay_buffer=self._replay_buffer)
      self._policy_checkpointer = common.Checkpointer(
          ckpt_dir=os.path.join(self._train_dir, 'policy'),
          max_to_keep=self._max_ckpt,
          behave_metrics=metric_utils.MetricsGroup(
              self._behavior_metrics + [self._iteration_metric],
              'behavior_metrics'),
          **all_iterable)
    else:
      self._rb_checkpointer = tf.train.CheckpointManager(
          tf.train.Checkpoint(replay_buffer=self._replay_buffer),
          directory=os.path.join(self._train_dir, 'replay_buffer'),
          max_to_keep=10)
      self._policy_checkpointer = tf.train.CheckpointManager(
          tf.train.Checkpoint(
              behave_metrics=metric_utils.MetricsGroup(
                  self._behavior_metrics + [self._iteration_metric],
                  'behavior_metrics'),
              **all_iterable),
          directory=os.path.join(self._train_dir, 'policy'),
          max_to_keep=self._max_ckpt)

  def record_log_policy_metric(self, worker_index, env_step):
    """Record log policy metric."""
    for i, worker in enumerate(self._worker_names):
      self._update_metrics(self._policy_metrics[worker],
                           [float(i == worker_index)])
      for metric in self._policy_metrics[worker]:
        add_summary(self._train_file_writer, 'PolicyMetrics/' + metric.name,
                    metric.result(), env_step)
        write_csv(self._csv_dir, metric.name, metric.result(), env_step,
                  self._iteration_metric.result())
      if self._pbt or self._use_bandit:
        add_summary(self._train_file_writer,
                    'QMetrics/' + self._bandit_arm_q[worker].name,
                    self._bandit_arm_q[worker].result(), env_step)
        write_csv(self._csv_dir, self._bandit_arm_q[worker].name,
                  self._bandit_arm_q[worker].result(), env_step,
                  self._iteration_metric.result())
      logging.info('worker id:%s', self._agent_names[worker])
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
                self._iteration_metric.result())
    write_csv(self._csv_dir, 'behavior_agent_name',
              self._agent_names[self._worker_names[worker_index]], env_step,
              self._iteration_metric.result())
    for hp in self._policy_metrics_hyper:
      value = self._hparams[self._worker_names[worker_index]].get(hp)
      if hp in ['lr', 'edecay']:
        histo = sorted(self._policy_metrics_hyper[hp].keys())
        histo.append(np.inf)
        histo = np.asarray(histo)
        bins = histo[np.argmax(histo >= np.log(value)) - 1]
      else:
        bins = value
      for param in self._policy_metrics_hyper[hp]:
        self._update_metrics(self._policy_metrics_hyper[hp][param],
                             [float(bins == param)])
        for metric in self._policy_metrics_hyper[hp][param]:
          add_summary(self._train_file_writer,
                      'PolicyMetrics(hparams)/' + metric.name, metric.result(),
                      env_step)
          write_csv(self._csv_dir, metric.name, metric.result(), env_step,
                    self._iteration_metric.result())

  def exploit(self, sess, way='uniform'):
    """Exploit."""
    bottom_table = []
    top_table = []
    for worker in self._worker_names:
      if not self._bandit_arm_q[worker].is_recent(
          update_time=self._pbt_update_requirement):
        self._count_reevaluate += 1
        logging.info('running immediate eval for agent %s, total eval:%d',
                     worker, self._count_reevaluate)
        self._update_bandit(
            sess,
            self._select_policy_way,
            worker,
            immediate_eval=True,
            push_value=self._push_when_eval)
      bottom_table.append(self._bandit_arm_q[worker].result(way=self._pbt_low))
      top_table.append(self._bandit_arm_q[worker].result(way=self._pbt_high))
    logging.info('worker name: %s', self._agent_names)
    logging.info('bottom table = %s, top table=%s', bottom_table, top_table)
    top_list = np.argsort(
        top_table)[-int(len(bottom_table) * self._pbt_percent_top):]
    bottom_list = np.argsort(
        bottom_table)[:int(len(bottom_table) * self._pbt_percent_low)]
    top_table = np.asarray(top_table)
    bottom_table = np.asarray(bottom_table)
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
        self._hparams[target_agent] = new_hparam
        self._agent_names[target_agent] = new_hparam.name
        self._agents[target_agent] = new_agent
        self._bandit_arm_q[target_agent] = copy.deepcopy(
            self._bandit_arm_q[source_agent])
        self._bandit_arm_q[target_agent].rename('QMetric_' + target_agent)
        logging.info('created new worker agent %s with hyperparam %s',
                     target_agent, self._agent_names[target_agent])

  def _mutate(self, parent_hparam, mutation_rate, mutate_list):
    """Mutate."""
    new_hparam = copy.deepcopy(parent_hparam)
    suffix = []
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

  def _train_one_step(self, sess, metric_traj):
    env_step = self._env_steps_metric.result()
    if self._use_bandit or self._pbt:
      self._update_metrics([self._bandit_reward_metric], metric_traj)
    train_step = None
    for worker_name in self._agents:
      agent = self._agents[worker_name]
      if env_step % agent.update_period == 0:
        train_step, loss = agent.train_one_step(sess, self._train_file_writer)
        self._maybe_log_train(train_step, loss, worker_name)
        add_summary(self._train_file_writer, 'TrainStep/' + worker_name,
                    train_step, env_step)
      self._maybe_record_behavior_summaries(env_step)
    return train_step, env_step

  def _update_bandit(self,
                     sess,
                     method,
                     current_agent,
                     immediate_eval=False,
                     push_value=True):
    """Update bandit."""
    if immediate_eval:
      # run immediate online eval for the designated agent, but don't increase
      # the discount on other agents nor increase their most_recent_time
      self._online_eval_parallel(sess, self._select_py_policies[current_agent],
                                 self._env_select, self._update_policy_period,
                                 self._selection_metrics[current_agent])
      reward = self._selection_metrics[current_agent][0].result()
    else:
      reward = self._bandit_reward_metric.result()
      self._bandit_reward_metric.reset()
    if push_value:
      if 'discount' in method:
        self._bandit_arm_q[current_agent].add_to_buffer(
            reward, discount=self._bandit_discount)
        self._add_zero_to_bandit(current_agent, discount=self._bandit_discount)
      else:
        self._bandit_arm_q[current_agent](reward)
        self._add_zero_to_bandit(current_agent)
    else:
      self._bandit_arm_q[current_agent].modify_last_buffer(reward)
    logging.info(
        'bandit updated with immediate eval=%s,push_value=%s,method=%s, newest Q=%s',
        immediate_eval, push_value, method,
        [self._bandit_arm_q[worker].result() for worker in self._worker_names])

  def _add_zero_to_bandit(self, current_agent, discount=1.0, update_time=True):
    for worker_name in self._agents:
      if worker_name != current_agent:
        self._bandit_arm_q[worker_name].add_to_buffer(
            0.0, discount=discount, update_time=update_time)

  def _which_policy(self, sess, method):
    """Which policy."""
    if method.startswith('default_'):
      logging.info('agent_name:%s', self._worker_names)
      try:
        worker_index = np.where(
            np.array(self._worker_names) == method[8:])[0][0]
      except ValueError:
        logging.info('Invalid agent name: %s', method[8:])
    elif method == 'random':
      worker_index = np.random.choice(len(self._agents))
    elif 'bandit' in method:
      _, current_epsilon = sess.run([self._select_step, self._select_epsilon])
      if 'ucb' in method:
        q_table = [
            self._bandit_arm_q[worker_name].result(
                way='ucb', coeff=self._bandit_ucb_coeff)
            for worker_name in self._worker_names
        ]
      else:
        q_table = [
            self._bandit_arm_q[worker_name].result(way='q')
            for worker_name in self._worker_names
        ]
      # select arm
      q_table = np.asarray(q_table)
      if 'epsilon' in method:
        if np.random.rand() < current_epsilon:
          worker_index = np.random.choice(len(self._agents))
        else:
          worker_index = np.argmax(q_table)
      elif 'softmax' in method:
        maxval = np.max(q_table[~np.isinf(q_table)])
        q_table[np.isinf(q_table)] = maxval * 2
        soft_q = np.exp(q_table) / sum(np.exp(q_table))
        logging.info('before soft q table = %s, sum_q = %f', soft_q,
                     sum(soft_q))
        soft_q /= (soft_q.sum() + 1e-7)
        logging.info('q table = %s', q_table)
        logging.info('soft q table = %s, sum_q = %f', soft_q, sum(soft_q))
        worker_index = np.where(np.random.multinomial(1, soft_q) == 1)[0][0]
      else:  # if 'greedy' in method:
        worker_index = np.argmax(q_table)
        logging.info('q table = %s', q_table)
      # else:
      #   raise ValueError('Invalid method: %s' % method)
    elif 'online' in method:
      logging.info('selection_metrics:%s',
                   self._selection_metrics[self._worker_names[0]])
      _, current_epsilon = sess.run([self._select_step, self._select_epsilon])
      if method == 'best_online':
        online_return = [
            self._selection_metrics[worker][0].result('mean')
            for worker in self._worker_names
        ]
        worker_index = np.argmax(online_return)
      elif method == 'best_online_ucb':
        online_ucb = [
            self._selection_metrics[worker][0].result('ucb', self._ucb_coeff)
            for worker in self._worker_names
        ]
        worker_index = np.argmax(online_ucb)
      elif method == 'best_online_variance':
        online_std = [
            self._selection_metrics[worker][0].result('std')
            for worker in self._worker_names
        ]
        worker_index = np.argmax(online_std)
      elif method == 'best_online_epsilon':
        if np.random.rand() < current_epsilon:
          worker_index = np.random.choice(len(self._agents))
        else:
          online_return = [
              self._selection_metrics[worker][0].result('mean')
              for worker in self._worker_names
          ]
          worker_index = np.argmax(online_return)
      else:
        raise ValueError('Invalid method: %s' % method)
    else:
      raise ValueError('Invalid method: %s' % method)
    return worker_index

  def _select_policy(self, sess, method, env_step=None):
    """Select policy."""
    if 'online' in method:
      for worker in self._worker_names:
        if self._online_eval_use_train:
          self._online_eval_parallel(sess, self._select_py_policies[worker],
                                     self._env_select,
                                     self._eval_episode_select,
                                     self._selection_metrics[worker])
        else:
          self._online_eval_parallel(sess, self._eval_py_policies[worker],
                                     self._env_select,
                                     self._eval_episode_select,
                                     self._selection_metrics[worker])
        for metric in self._selection_metrics[worker]:
          add_summary(self._train_file_writer,
                      metric.name.split('_')[0] + '(mean)/' + metric.name,
                      metric.result('mean'), env_step)
          # add_summary(self._train_file_writer,
          #             'SelectionEvalMetrics(95ucb)/' + metric.name,
          #             metric.result('95ucb'), env_step)
    worker_index = self._which_policy(sess, method)
    return worker_index

  def _maybe_record_behavior_summaries(self, env_step):
    """Record summaries if env_step is a multiple of summary_interval."""
    if env_step % self._summary_interval == 0:
      for metric in self._behavior_metrics:
        if metric.result() != 0:
          add_summary(self._train_file_writer, 'Metrics/' + metric.name,
                      metric.result(), env_step)

  def _maybe_time_train(self, train_step):
    """Maybe time train."""
    if train_step % self._log_interval == 0:
      steps_per_sec = (
          (train_step - self._timed_at_step) /
          (self._collect_timer.value() + self._train_timer.value()))
      add_summary(self._train_file_writer, 'train_steps_per_sec', steps_per_sec,
                  train_step)
      logging.info('%.3f steps/sec', steps_per_sec)
      logging.info(
          '%s', 'collect_time = {}, train_time = {}'.format(
              self._collect_timer.value(), self._train_timer.value()))
    self._timed_at_step = train_step
    self._collect_timer.reset()
    self._train_timer.reset()

  def _maybe_log_and_reset_timer(self):
    logging.info(
        'iteration time: %s',
        'pbt_time = {}, select_time = {}, policy_time = {}, checkpoint_time = {}'
        .format(self._pbt_timer.value(), self._select_timer.value(),
                self._policy_timer.value(), self._checkpoint_timer.value()))
    self._pbt_timer.reset()
    self._select_timer.reset()
    self._policy_timer.reset()
    self._checkpoint_timer.reset()


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
    self._train_checkpointer._checkpoint.restore(  # pylint: disable=protected-access
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
        self._select_step = tf.Variable(
            0, dtype=tf.int64, trainable=False, name='bandit_step')
        # self.create_pypolicy_and_train_op()
        fail = True
        while fail:
          # try
          self.update_train_bandit_checkpointer(
              update_bandit=False, use_common=True)
          fail = False
        logging.info('new checkpoint manager at ckpt:%s, hparam at ckpt:%s',
                     self._train_checkpointer._manager.latest_checkpoint, step)  # pylint: disable=protected-access
      else:
        logging.info('hparam file not found for %s, try next ckpt', hparam_file)
        continue

      with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Initialize the graph.
        self._initialize_graph(sess, checkpoint_path)
        logging.info('Starting evaluation')
        logging.info('choosing from: %s', self._worker_names)
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
