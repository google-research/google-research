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

"""Implementation of RL agents."""

from __future__ import division
from __future__ import print_function

import collections
import heapq
import pprint
import random
import numpy as np
import tensorflow.compat.v1 as tf
from meta_reward_learning.semantic_parsing.nsm import data_utils
from meta_reward_learning.semantic_parsing.nsm import env_factory
from meta_reward_learning.semantic_parsing.nsm import graph_factory

# To suppress division by zero in np.log, because the
# behaviour np.log(0.0) = -inf is correct in this context.
np.warnings.filterwarnings('ignore')


def samples_with_features(batch_envs, samples, max_n_exp):
  """Add sim_features to samples."""
  new_samples = []
  batch_env_dict = {env.name: env for env in batch_envs}
  for sample in samples:
    traj = sample.traj
    if sum(traj.rewards):
      # Computing the features for program with zero rewards is wasteful
      env = batch_env_dict[traj.env_name]
      program = traj_to_program(traj, env.de_vocab)
      sim_features = env_factory.create_features(env, program, max_n_exp)
      new_traj = traj._replace(sim_features=sim_features)
      sample = sample._replace(traj=new_traj)
    new_samples.append(sample)
  return new_samples


def get_data_for_train(samples):
  trajs = [s.traj for s in samples]
  obs = [t.obs for t in trajs]
  actions = [t.actions for t in trajs]
  context = [t.context for t in trajs]
  returns = [sum(t.rewards) for t in trajs]
  weights = [s.prob * r for s, r in zip(samples, returns)]
  return obs, actions, context, weights


def scale_probs(samples, scale):
  """Weight each samples with the weight. Reflected on probs."""
  scaled_probs = [scale * s.prob for s in samples]
  new_samples = []
  for s, p in zip(samples, scaled_probs):
    new_samples.append(Sample(traj=s.traj, prob=p))
  return new_samples


def normalize_probs(samples, smoothing=1e-8, beta=1):
  """Normalize the probability of the samples (in each env) to sum to 1.0."""
  sum_prob_dict = {}
  probs = [np.power(s.prob, beta) + smoothing for s in samples]
  for s, prob in zip(samples, probs):
    name = s.traj.env_name
    if name in sum_prob_dict:
      sum_prob_dict[name] += prob
    else:
      sum_prob_dict[name] = prob
  new_samples = []
  for s, prob in zip(samples, probs):
    new_prob = prob / sum_prob_dict[s.traj.env_name]
    new_samples.append(Sample(traj=s.traj, prob=new_prob))
  return new_samples


def compute_baselines(returns, probs, env_names):
  """Compute baseline for samples."""
  baseline_dict = {}
  for ret, p, name in zip(returns, probs, env_names):
    if name not in baseline_dict:
      baseline_dict[name] = ret * p
    else:
      baseline_dict[name] += ret * p
  return baseline_dict


class PGAgent(object):
  """Agent trained by policy gradient."""

  def __init__(self,
               model,
               train_writer=None,
               discount_factor=1.0,
               extra_monitors=False,
               score_model='tabular',
               score_norm_fn='sigmoid',
               use_baseline=False):

    # Used for weights experiment
    self.env_to_index = {}
    # Set this parameter when doing meta learning
    self.meta_learn = model.meta_learn
    self.model = model
    self.discount_factor = discount_factor
    self._extra_monitors = extra_monitors
    self.train_writer = train_writer
    self.score_model = score_model
    self.score_norm_fn = score_norm_fn
    self.use_baseline = use_baseline
    if self.use_baseline:
      self.baseline_dict = dict()

    self.monitor_graph = graph_factory.MonitorGraph()
    for name in [
        'avg_return', 'avg_len', 'avg_prob'
        # 'val_loss', min_return', 'max_return', 'std_return',
        # 'min_len', 'max_len', 'std_len', 'clip_frac'
    ]:
      self.monitor_graph.add_scalar_monitor(name, dtype=tf.float32)
    if extra_monitors:
      self.extra_monitor_list = []
      # 'buffer_prob', 'reweighted_buffer_prob', 'replay_buffer_size',
      # 'mean_buffer_size', 'num_contexts']
      for name in self.extra_monitor_list:
        self.monitor_graph.add_scalar_monitor(name, dtype=tf.float32)

    self.monitor_graph.launch()

  def generate_samples(self,
                       envs,
                       n_samples=1,
                       greedy=False,
                       use_cache=False,
                       filter_error=True):
    """Returns Actions, rewards, obs, other info."""
    samples = sampling(
        self.model,
        envs,
        n_samples=n_samples,
        greedy=greedy,
        use_cache=use_cache,
        filter_error=filter_error)
    return samples

  def beam_search(self,
                  envs=None,
                  beam_size=1,
                  use_cache=False,
                  greedy=False,
                  renorm=True):
    """Returns Actions, rewards, obs and probs."""
    samples = beam_search(
        self.model,
        envs,
        beam_size=beam_size,
        use_cache=use_cache,
        greedy=greedy,
        renorm=renorm)
    return samples

  def update_replay_prob(self, samples, min_replay_weight=0.0):
    """Update the probability of the replay samples and recompute the weights (prob)."""
    prob_sum_dict = {}
    trajs_to_update = [sample.traj for sample in samples if sample.prob is None]
    new_probs = self.compute_probs(trajs_to_update)
    for traj, prob in zip(trajs_to_update, new_probs):
      name = traj.env_name
      if name in prob_sum_dict:
        prob_sum_dict[name] += prob
      else:
        prob_sum_dict[name] = prob

    i = 0
    new_samples = []
    for sample in samples:
      name = sample.traj.env_name
      if name in prob_sum_dict:
        w = max(prob_sum_dict[name], min_replay_weight)
      else:
        w = 0.0
      if sample.prob is None:
        new_samples.append(
            sample._replace(prob=new_probs[i] * w / prob_sum_dict[name]))
        i += 1
      else:
        prob = sample.prob * (1.0 - w)
        new_samples.append(sample._replace(prob=prob))
    assert i == len(trajs_to_update)

    return new_samples

  # pylint: disable=missing-docstring
  def train(self,
            samples,
            debug=False,
            parameters=None,
            min_prob=0.0,
            scale=1.0,
            de_vocab=None,
            val_samples=None):
    trajs = [s.traj for s in samples]
    probs = [s.prob for s in samples]
    env_names = [t.env_name for t in trajs]
    rewards = [sum(t.rewards) for t in trajs]

    obs = [t.obs for t in trajs]
    actions = [t.actions for t in trajs]
    context = [t.context for t in trajs]
    sim_features, env_indices = self.create_score_features(trajs)

    # Compute the product of pi(a) * R(a) where R(a) is the binary reward
    # given by the executor
    weights = [p * r * scale for p, r in zip(probs, rewards)]
    if self.use_baseline:
      baselines = self.get_baselines(samples)
    else:
      baselines = None

    if debug:
      print('+' * 50)
      model_probs = self.compute_probs(trajs)
      print('scale: {}, min_prob: {}'.format(scale, min_prob))
      for i, (name, r, p, mp, w, traj) in enumerate(
          zip(env_names, rewards, probs, model_probs, weights, trajs)):
        print(('sample {}, name: {}, return: {}, '
               'prob: {}, model prob: {}, weight: {}').format(
                   i, name, r, p, mp, w))
        if de_vocab is not None:
          print(' '.join(traj_to_program(traj, de_vocab)))
      print('+' * 50)

    if self.meta_learn:
      val_obs, val_actions, val_context, val_weights = get_data_for_train(
          val_samples)
      val_feed_dict = dict(
          val_inputs=val_obs,
          val_targets=val_actions,
          val_context=val_context,
          val_weights=val_weights)
    else:
      val_feed_dict = None
    self.model.train(
        obs,
        actions,
        weights=weights,
        context=context,
        parameters=parameters,
        val_feed_dict=val_feed_dict,
        writer=self.train_writer,
        baselines=baselines,
        sim_features=sim_features,
        env_indices=env_indices)
  # pylint: enable=missing-docstring

  def compute_probs(self, trajs):
    obs = [s.obs for s in trajs]
    actions = [s.actions for s in trajs]
    context = [s.context for s in trajs]
    probs = self.model.compute_probs(obs, actions, context=context)
    return probs

  def create_score_features(self, trajs):
    env_indices, sim_features = None, None
    if self.score_model == 'tabular' or self.score_norm_fn == 'softmax':
      sim_features = [[self.env_to_index[t.env_name], t.idx] for t in trajs]
    elif self.score_model in ['linear', 'local_linear', 'local_attn']:
      sim_features = [t.sim_features for t in trajs]
      if self.score_model == 'local_linear':
        env_indices = [self.env_to_index[t.env_name] for t in trajs]
    return sim_features, env_indices

  def compute_scores(self, trajs):
    """Compute the scores assigned to the trajs."""
    if self.score_model not in ['attn', 'complex']:
      sim_features, env_indices = self.create_score_features(trajs)
      scores = self.model.compute_simple_scores(
          sim_features=sim_features, env_indices=env_indices)
    else:
      obs = [s.obs for s in trajs]
      actions = [s.actions for s in trajs]
      context = [s.context for s in trajs]
      scores = self.model.compute_scores(obs, actions, context=context)
    return scores

  def get_baselines(self, samples):
    return [self.baseline_dict[s.traj.env_name] for s in samples]

  def compute_step_logprobs(self, trajs):
    obs = [s.obs for s in trajs]
    actions = [s.actions for s in trajs]
    context = [s.context for s in trajs]
    logprobs = self.model.compute_step_logprobs(obs, actions, context=context)
    return logprobs

  def evaluate(
      self,
      samples,
      writer=None,
      true_n=None,
      # clip_frac=0.0,
      extra_monitors=None):
    """Evaluate the agent on the envs."""

    trajs = [s.traj for s in samples]
    actions = [t.actions for t in trajs]
    probs = [s.prob for s in samples]

    returns = [
        compute_returns(t.rewards, self.discount_factor)[0] for t in trajs
    ]

    # pylint: disable=unused-variable
    avg_return, _, max_return, min_return, n_w = compute_weighted_stats(
        returns, probs)
    # pylint: enable=unused-variable

    if true_n is not None:
      # Account for the fact that some environment doesn't
      # generate any valid samples, but we still need to
      # consider them when computing returns.
      new_avg_return = avg_return * n_w / true_n
      tf.logging.info(
          'avg return adjusted from {} to {} based on true n'.format(
              avg_return, new_avg_return))
      avg_return = new_avg_return

    lens = [len(acs) for acs in actions]
    # pylint: disable=unused-variable
    avg_len, std_len, max_len, min_len, _ = compute_weighted_stats(lens, probs)
    # pylint: enable=unused-variable

    if writer is not None:
      if extra_monitors is None:
        if self._extra_monitors:
          extra_monitors = {monitor: 0.0 for monitor in self.extra_monitor_list}
        else:
          extra_monitors = {}
      feed_dict = dict(
          avg_return=avg_return,
          avg_len=avg_len,
          **extra_monitors)
      self.write_to_monitor(feed_dict, writer)
    return avg_return, avg_len

  def write_to_monitor(self, feed_dict, writer):
    summary = self.monitor_graph.generate_summary(feed_dict)
    writer.add_summary(summary, self.model.get_global_step())
    writer.flush()


def compute_weighted_stats(array, weight):
  """Compute the stats (e.g. mean, std) of array weighted by `weight`."""
  n = len(array)
  if n < 1:
    return (0.0, 0.0, 0.0, 0.0, 0.0)
  sum_ = 0.0
  min_ = array[0]
  max_ = array[0]
  n_w = sum(weight)
  for a, w in zip(array, weight):
    min_ = min([min_, a])
    max_ = max([max_, a])
    sum_ += a * w
  mean = sum_ / n_w
  sum_square_std = 0.0
  for a, w in zip(array, weight):
    sum_square_std += (a - mean)**2 * w
  std = np.sqrt(sum_square_std / n_w)
  return (mean, std, max_, min_, n_w)


def compute_returns(rewards, discount_factor=1.0):
  """Compute returns of a trace (sum of discounted rewards).

  Args:
    rewards: list of float rewards discount_factor
    discount_factor: Discount factor to be used for return calculation.

  Returns:
    list[float32]: A list of discounted returns of same size as rewards.
  """
  returns = []
  t = len(rewards)
  returns = [0.0] * t
  sum_return_so_far = 0.0
  for i in xrange(t):
    returns[-i - 1] = sum_return_so_far * discount_factor + rewards[-i - 1]
    sum_return_so_far = returns[-1 - i]
  return returns


def compute_td_errors(values, rewards, discount_factor=1.0, td_n=0):
  """Compute TD errors."""
  td_errors = []
  td_n += 1
  backup_values = compute_backup_values(values, rewards, discount_factor, td_n)
  for vs, bvs in zip(values, backup_values):
    td_errors.append((np.array(bvs) - np.array(vs)).tolist())
  return td_errors


def compute_backup_values(values, rewards, discount_factor=1.0, n_steps=1):
  """Compute backup value."""
  backup_values = []
  for vs, rs in zip(values, rewards):
    bvs = []
    # pylint: disable=invalid-name
    T = len(vs)
    # pylint: enable=invalid-name
    for i in xrange(T):
      end = min(i + n_steps, T)
      if end == T:
        bv = 0.0
      else:
        bv = vs[end] * (discount_factor**(end - i))
      for t in xrange(i, end):
        bv += rs[t] * (discount_factor**(t - i))
      bvs.append(bv)
    backup_values.append(bvs)
  return backup_values


class ReplayBuffer(object):

  def save(self, samples):
    pass

  def replay(self, envs):
    pass


class AllGoodReplayBuffer(ReplayBuffer):
  """Class for the replay buffer containing successful programs."""

  def __init__(self, agent=None, de_vocab=None, discount_factor=1.0):

    # Mapping env names to good trajectories in that env.
    self._buffer = dict()
    self.discount_factor = discount_factor
    self.agent = agent
    self.de_vocab = de_vocab
    # Persistent mapping from env names to good program strs
    self.program_prob_dict = dict()
    self.prob_sum_dict = dict()

  def has_found_solution(self, env_name):
    return env_name in self._buffer and self._buffer[env_name]

  @property
  def traj_buffer(self):
    return self._buffer

  def contain(self, traj):
    """Checks whether a given action sequence is present in the buffer or not."""
    name = traj.env_name
    if name not in self.program_prob_dict:
      return False
    program = traj_to_program(traj, self.de_vocab)
    program_str = u' '.join(program)
    if program_str in self.program_prob_dict[name]:
      return True
    else:
      return False

  def get_all_progs(self):
    return {k: v.keys() for k, v in self.program_prob_dict.iteritems()}

  @property
  def size(self):
    n = 0
    for v in self._buffer.values():
      n += len(v)
    return n

  def save(self, samples):
    trajs = [s.traj for s in samples]
    self.save_trajs(trajs)

  def check_not_in_buffer(self, sample):
    traj = sample.traj
    name = traj.env_name
    program = traj_to_program(traj, self.de_vocab)
    program_str = ' '.join(program)
    return  program and (program[-1] == self.de_vocab.end_tk) and \
            (not (name in self.program_prob_dict and
                  (program_str in self.program_prob_dict[name])))

  def save_trajs(self, trajs):
    # pylint: disable=g-doc-args
    """Saves only good trajectories not currently present in the buffer.

    A good trajectory has length > 0, achieves a return greater than 0.5,
    and contains the end token as it's last item.
    """
    # pylint: enable=g-doc-args
    total_returns = [
        compute_returns(t.rewards, self.discount_factor)[0] for t in trajs
    ]
    for t, return_ in zip(trajs, total_returns):
      name = t.env_name
      program = traj_to_program(t, self.de_vocab)
      program_str = ' '.join(program)
      # pylint: disable=g-explicit-length-test
      if (return_ > 0.5 and len(program) > 0 and
          (program[-1] == self.de_vocab.end_tk) and
          (not (name in self.program_prob_dict and
                (program_str in self.program_prob_dict[name])))):
        if name in self.program_prob_dict:
          self.program_prob_dict[name][program_str] = True
        else:
          self.program_prob_dict[name] = {program_str: True}
        if name in self._buffer:
          self._buffer[name].append(t)
        else:
          self._buffer[name] = [t]
      # pylint: enable=g-explicit-length-test

  def all_samples(self, envs, agent=None, sampling_strategy=None, is_val=False):
    """All samples with correct probability values for sampling strategy."""
    select_env_names = set([e.name for e in envs])
    trajs = []
    # Collect all the trajs for the selected envs.
    for name in select_env_names:
      if name in self._buffer:
        trajs += self._buffer[name]
    if agent is None or sampling_strategy is None:
      # All traj has the same probability, since it will be
      # normalized later, we just assign them all as 1.0.
      probs = [1.0] * len(trajs)
    else:
      if (not is_val) and sampling_strategy != 'probs':
        probs = agent.compute_scores(trajs)
        if agent.score_norm_fn == 'identity' and agent.score_model == 'linear':
          probs = np.exp(probs)
        if sampling_strategy == 'probs_and_reward':
          probs *= agent.compute_probs(trajs)
      else:
        probs = agent.compute_probs(trajs)
        if (not is_val) and agent.use_baseline:
          rewards = agent.compute_scores(trajs)
          env_names = [traj.env_name for traj in trajs]
          agent.baseline_dict.update(
              compute_baselines(rewards, probs, env_names))
    samples = [Sample(traj=t, prob=p) for t, p in zip(trajs, probs)]
    return samples

  def replay(self,
             envs,
             n_samples=1,
             use_top_k=False,
             agent=None,
             truncate_at_n=0,
             is_val=False,
             sampling_strategy=None,
             beta=1):
    select_env_names = set([e.name for e in envs])
    samples = self.all_samples(
        envs, agent=agent, sampling_strategy=sampling_strategy, is_val=is_val)
    # Put the samples into an dictionary keyed by env names.
    env_sample_dict = dict(
        [(name, []) for name in select_env_names if name in self._buffer])
    for s in samples:
      name = s.traj.env_name
      env_sample_dict[name].append(s)

    replay_samples = []
    for name, samples in env_sample_dict.iteritems():
      n = len(samples)
      # Truncated the number of samples in the selected
      # samples and in the buffer.
      if truncate_at_n > 0 and n > truncate_at_n:
        # Randomize the samples before truncation in case
        # when no prob information is provided and the trajs
        # need to be truncated randomly.
        random.shuffle(samples)
        samples = heapq.nlargest(truncate_at_n, samples, key=lambda s: s.prob)
        self._buffer[name] = [sample.traj for sample in samples]

      # Compute the sum of prob of replays in the buffer.
      self.prob_sum_dict[name] = sum([sample.prob for sample in samples])

      # Used for hard EM
      if use_top_k:
        # Select the top k samples weighted by their probs.
        selected_samples = heapq.nlargest(
            n_samples, samples, key=lambda s: s.prob)
        replay_samples += normalize_probs(selected_samples)
      else:
        # Randomly samples according to their probs.
        samples = normalize_probs(samples, beta=beta)
        idxs = np.random.choice(
            len(samples), n_samples, p=[sample.prob for sample in samples])
        selected_samples = [samples[i] for i in idxs]
        replay_samples += [
            Sample(traj=s.traj, prob=1.0 / n_samples) for s in selected_samples
        ]

    return replay_samples


def traj_to_program(traj, de_vocab):
  program = []
  for a, ob in zip(traj.actions, traj.obs):
    ob = ob[0]
    token = de_vocab.lookup(ob.valid_indices[a], reverse=True)
    program.append(token)
  return program

# pylint: disable=invalid-name
Traj = data_utils.namedtuple_with_defaults(
    'Traj', 'obs actions rewards context env_name answer idx sim_features')

Sample = collections.namedtuple('Sample', 'traj prob')
# pylint: enable=invalid-name


def sampling(model,
             envs,
             temperature=1.0,
             use_encode=True,
             greedy=False,
             n_samples=1,
             debug=False,
             use_cache=False,
             filter_error=True):
  """Code for sampling programs using the model."""

  if not envs:
    raise ValueError('No environment provided!')

  if use_cache:
    # if already explored everything, then don't explore this environment
    # anymore.
    envs = [env for env in envs if not env.cache.is_full()]

  duplicated_envs = []
  for env in envs:
    for i in range(n_samples):
      duplicated_envs.append(env.clone())

  envs = duplicated_envs

  for env in envs:
    env.use_cache = use_cache

  if use_encode:
    env_context = [env.get_context() for env in envs]
    encoded_context, initial_state = model.encode(env_context)
  else:
    # env_context = [None for env in envs]
    encoded_context, initial_state = None, None

  obs = [[env.start_ob] for env in envs]
  state = initial_state

  while True:
    outputs, state = model.step(obs, state, context=encoded_context)

    if greedy:
      actions = model.predict(cell_outputs=outputs)
    else:
      actions = model.sampling(cell_outputs=outputs, temperature=temperature)

    if debug:
      print('*' * 50)
      print('actions: ')
      pprint.pprint(actions)
      print('action_prob: ')
      action_prob = model.predict_prob(cell_outputs=outputs)
      pprint.pprint(action_prob)
      print('*' * 50)

    # Get rid of the time dimension so that actions is just one list.
    actions = [a[0] for a in actions]
    action_probs = model.predict_prob(cell_outputs=outputs)
    action_probs = [ap[0] for ap in action_probs]

    obs = []
    for env, action, p in zip(envs, actions, action_probs):
      try:
        # pylint: disable=unused-variable
        ob, _, _, info = env.step(action)
        # pylint: enable=unused-variable
        obs.append([ob])
      except IndexError:
        print(p)
        raise IndexError
    step_pairs = [
        x for x in zip(obs, state, encoded_context, envs) if not x[-1].done
    ]
    if step_pairs:
      obs, state, encoded_context, envs = zip(*step_pairs)
      obs = list(obs)
      state = list(state)
      envs = list(envs)
      encoded_context = list(encoded_context)
      assert len(obs) == len(state)
      assert len(obs) == len(encoded_context)
      assert len(obs) == len(envs)
    else:
      break

  # pylint: disable=unused-variable
  obs, actions, rewards = zip(*[(env.obs, env.actions, env.rewards)
                                for env in duplicated_envs])
  # pylint: enable=unused-variable
  env_names = [env.name for env in duplicated_envs]
  answers = [env.interpreter.result for env in duplicated_envs]

  samples = []
  for i, env in enumerate(duplicated_envs):
    if not (filter_error and env.error):
      samples.append(
          Sample(
              traj=Traj(
                  obs=env.obs,
                  actions=env.actions,
                  rewards=env.rewards,
                  context=env_context[i],
                  env_name=env_names[i],
                  answer=answers[i]),
              prob=1.0 / n_samples))
  return samples


Hyph = collections.namedtuple('Hyph', ['state', 'env', 'score'])
Candidate = collections.namedtuple('Candidate',
                                   ['state', 'env', 'score', 'action'])


def beam_search(model,
                envs,
                use_encode=True,
                beam_size=1,
                debug=False,
                renorm=True,
                use_cache=False,
                filter_error=True,
                greedy=False):
  """Beam search using the model."""
  if use_cache:
    # if already explored everything, then don't explore this environment
    # anymore.
    envs = [env for env in envs if not env.cache.is_full()]

  if use_encode:
    env_context = [env.get_context() for env in envs]
    encoded_context, initial_state = model.encode(env_context)
    env_context_dict = dict([(env.name, env.get_context()) for env in envs])
    context_dict = dict(
        [(env.name, c) for env, c in zip(envs, encoded_context)])
    beam = [Hyph(s, env.clone(), 0.0) for env, s in zip(envs, initial_state)]
    state = initial_state
    context = encoded_context
  else:
    beam = [Hyph(None, env.clone(), 0.0) for env in envs]
    state = None
    context = None
    env_context_dict = dict([(env.name, None) for env in envs])

  for hyp in beam:
    hyp.env.use_cache = use_cache

  finished_dict = dict([(env.name, []) for env in envs])
  obs = [[h.env.start_ob] for h in beam]

  while beam:
    if debug:
      print('@' * 50)
      print('beam is')
      for h in beam:
        print('env {}'.format(h.env.name))
        print(h.env.show())
        print(h.score)
        print()

    # Run the model for one step to get probabilities for new actions.
    outputs, state = model.step(obs, state, context=context)

    probs = model.predict_prob(outputs)
    scores = (np.log(np.array(probs)) + np.array([[[h.score]] for h in beam]))

    # Collect candidates.
    candidate_dict = {}
    for hyph, st, score in zip(beam, state, scores):
      env_name = hyph.env.name
      if env_name not in candidate_dict:
        candidate_dict[env_name] = []
      for action, s in enumerate(score[0]):
        # pylint: disable=g-comparison-negation
        if not s == -np.inf:
          candidate_dict[env_name].append(Candidate(st, hyph.env, s, action))
        # pylint: enable=g-comparison-negation

    if debug:
      print('*' * 20)
      print('candidates are')
      for k, v in candidate_dict.iteritems():
        print('env {}'.format(k))
        for x in v:
          print(x.env.show())
          print(x.action)
          print(x.score)
          print(type(x))
          print(isinstance(x, Candidate))
          print()

    # Collect the new beam.
    new_beam = []
    obs = []
    for env_name, candidates in candidate_dict.iteritems():
      # Find the top k from the union of candidates and
      # finished hypotheses.
      all_candidates = candidates + finished_dict[env_name]
      topk = heapq.nlargest(beam_size, all_candidates, key=lambda x: x.score)

      # Step the environment and collect the hypotheses into
      # new beam (unfinished hypotheses) or finished_dict
      finished_dict[env_name] = []
      for c in topk:
        if isinstance(c, Hyph):
          finished_dict[env_name].append(c)
        else:
          env = c.env.clone()
          # pylint: disable=unused-variable
          ob, _, done, info = env.step(c.action)
          # pylint: enable=unused-variable
          new_hyph = Hyph(c.state, env, c.score)
          if not done:
            obs.append([ob])
            new_beam.append(new_hyph)
          else:
            if not (filter_error and new_hyph.env.error):
              finished_dict[env_name].append(new_hyph)

    if debug:
      print('#' * 20)
      print('finished programs are')
      for k, v in finished_dict.iteritems():
        print('env {}'.format(k))
        for x in v:
          print(x.env.show())
          print(x.score)
          print(type(x))
          print(isinstance(x, Hyph))
          print()

    beam = new_beam

    if use_encode:
      state = [h.state for h in beam]
      context = [context_dict[h.env.name] for h in beam]
    else:
      state = None
      context = None

  final = []
  env_names = [e.name for e in envs]
  for name in env_names:
    sorted_final = sorted(
        finished_dict[name], key=lambda h: h.score, reverse=True)
    if greedy:
      # Consider the time when sorted_final is empty (didn't
      # find any programs without error).
      if sorted_final:
        final += [sorted_final[0]]
    else:
      final += sorted_final

  if not final:
    return []
  # Collect the training examples.
  obs, actions, rewards, env_names, scores = zip(*[(h.env.obs, h.env.actions,
                                                    h.env.rewards, h.env.name,
                                                    h.score) for h in final])
  answers = [h.env.interpreter.result for h in final]

  samples = []
  for i, name in enumerate(env_names):
    samples.append(
        Sample(
            traj=Traj(
                obs=obs[i],
                actions=actions[i],
                rewards=rewards[i],
                context=env_context_dict[name],
                env_name=name,
                answer=answers[i]),
            prob=np.exp(scores[i])))

  if renorm:
    samples = normalize_probs(samples)
  return samples


class RandomAgent(object):
  """A random agent."""

  def __init__(self, discount_factor=1.0):
    self.discount_factor = discount_factor

  # pylint: disable=missing-docstring
  def generate_samples(self, envs, n_samples=1, use_cache=False):
    if use_cache:
      # if already explored everything, then don't explore this environment
      # anymore.
      envs = [env for env in envs if not env.cache.is_full()]

    for env in envs:
      env.use_cache = use_cache

    duplicated_envs = []
    for env in envs:
      for i in range(n_samples):
        duplicated_envs.append(env.clone())

    envs = duplicated_envs

    for env in envs:
      ob = env.start_ob
      while not env.done:
        valid_actions = ob[0].valid_indices
        action = np.random.randint(0, len(valid_actions))
        ob, _, _, _ = env.step(action)

    env_context = [env.get_context() for env in envs]
    env_names = [env.name for env in envs]
    samples = []
    for i, env in enumerate(envs):
      samples.append(
          Sample(
              traj=Traj(
                  obs=env.obs,
                  actions=env.actions,
                  rewards=env.rewards,
                  context=env_context[i],
                  env_name=env_names[i]),
              prob=1.0 / n_samples))
    return samples

  def evaluate(self, samples):
    trajs = [s.traj for s in samples]
    actions = [t.actions for t in trajs]

    probs = [s.prob for s in samples]

    returns = [
        compute_returns(t.rewards, self.discount_factor)[0] for t in trajs
    ]

    # pylint: disable=unused-variable
    avg_return, std_return, max_return, min_return = compute_weighted_stats(
        returns, probs)

    lens = [len(acs) for acs in actions]
    avg_len, std_len, max_len, min_len = compute_weighted_stats(lens, probs)
    # pylint: enable=unused-variable
    return avg_return, avg_len
