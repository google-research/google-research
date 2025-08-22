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

"""Web Environment."""

import json

from absl import logging
from CoDE import utils
from CoDE import vocabulary_node
from CoDE import vocabulary_utils
from CoDE import web_observation_wrappers
import CoDE.gym_spaces as custom_gym_spaces
import gin
import gym
import gym.spaces
from miniwob.action import MiniWoBTerminate
from miniwob.fields import get_field_extractor
from miniwob.screenshot import get_screenshot
import numpy as np
from selenium.common.exceptions import JavascriptException


class BaseJavascriptError(JavascriptException):
  """Base javascript error for gminiwob code execution."""


class PotentialComputationError(BaseJavascriptError):
  """Raised when the potential function could not be computed in the website."""


class EnvironmentTerminateError(Exception):
  """Raised when the step is called while environment is done."""


@gin.configurable('WebEnvironment')
class WebEnvironment(gym.Env):
  """Wrapper for WoB environments."""

  def __init__(
      self,
      base_url,
      subdomain,
      # Profile params.
      profile_length,
      number_of_fields,
      use_only_profile_key,
      # Dom params.
      number_of_dom_elements,
      dom_attribute_sequence_length,
      use_dom_profile_intersection=False,
      # Action params.
      number_of_action_types=2,
      keyboard_action_size=5,
      # Env params.
      step_limit=6,
      cyclic_action_penalty=0.0,
      timestep_penalty=0.0,
      exact_wob_reward=True,
      step_limit_reward=0.0,
      final_reward_bias=0.0,
      # General params.
      mode='train',
      verbose=False,
      verbose_prob=0.1,
      # Gminiwob parameters.
      gminiwob_required_complexity='original',
      gminiwob_unrequired_complexity='original',
      subtasks=None,
      randomized_within_groups=None,
      randomized_across_groups=None,
      randomized_env_components=None,
      use_potential_based_reward=False,
      generate_screenshots=False,
      global_vocabulary=None,
      # Kwargs for WoB Env.
      kwargs_dict=None):
    """Initialize a web environment.

    Implement:
        step
        reset
        close
        seed

    functions.
    Set the following attributes for gym interface:

        action_space: The Discrete space object corresponding to valid actions
        observation_space: The Dict space object corresponding to valid
          observations

    Args:
      base_url: Base url or file path for miniwob html files.
      subdomain: WoB task name.
      profile_length: Max length of an profile field sequence.
      number_of_fields: Number of fields in user profile.
      use_only_profile_key: If true, use only key of the user profile for
        computing dom profile intersection, not the value. Actions, point to
        values by pointing to the corresponding keys.
      number_of_dom_elements: Number of dom elements. If greater than number of
        elements in the page, they are padded and masked.
      dom_attribute_sequence_length: Max dom sequence length.
      use_dom_profile_intersection: If true, use dom profile word overlap as
        input.
      number_of_action_types: Number of action types (click, type, etc.). We
        assume that 'click' action always exists.
      keyboard_action_size: Number of fields that can be typed. If using
        structured profiles, this should be equal to number_of_fields.
        Otherwise, this should be number of tokens in the profile.
      step_limit: Max number of allowed steps.
      cyclic_action_penalty: Penalty for cyclic actions.
      timestep_penalty: Penalty for each timestep.
      exact_wob_reward: If true, use axact wob reward. Otherwise, use
        timestep_penaly and cyclic_action_penalty in addition to wob reward.
      step_limit_reward: Reward when reaching step limit.
      final_reward_bias: Add to the final reward.
      mode: [train, test[ Mode of the environment.
      verbose: If true, verbose episodes.
      verbose_prob: Verbose probability.
      gminiwob_required_complexity: [easiest, easy, medium, hard, hardest,
        original] Complexity of required elements in the page for gminiwob
        environments. This includes number of elements as well as their
        complexity such as a select element requires text matching while a
        textbox doesn't. By default ("original"), full page is used.
      gminiwob_unrequired_complexity: [easiest, easy, medium, hard, hardest,
        original] Complexity of additional elements in the page that are not
        required for the task. By default ("original"), full page is used.
      subtasks: ["fillUserInfo", "fillAddress", "fillPayment"] List of subtasks
        to use to set up the environment.
      randomized_within_groups: If given, these are used to randomize within
        grouped elements such as by changing orders of elements within groups.
      randomized_across_groups: If given, these are used to randomize across
        grouped elements such as by changing the order of groups.
      randomized_env_components: If given, these environment components will be
        randomized such as by random number of items in a cart.
      use_potential_based_reward: If true, use potentials from the website to
        estimate dense rewards.
      generate_screenshots: If true, screenshots for each state will be
        generated. These can be written into a file afterwards.
      global_vocabulary: Global vocabulary node for distributed env.
      kwargs_dict: Key-value args to be passed to the WoB environment
        constructor. Also useful to specify arguments to chromedriver such as
        'headless'.

    Raises:
      ValueError: if typing from profile and number of fields is not given or
        temporal discount is outside of [0, 1].
    """
    logging.info('kwargs passes to wob environment : %s', str(kwargs_dict))
    # create wob environment
    self.subdomain = subdomain
    self._wob_env = utils.create_environment(subdomain, base_url, kwargs_dict)
    self._step_limit = step_limit
    self._mode = mode
    self._verbose = verbose
    self.seed()
    # 5 main attributes: tag, value, text, placeholder, class, [name]
    self.number_of_dom_attributes = 5
    # The global vocabulary is wrapped with a client.Client class. Creating the
    # local vocab Vocabulary object, allows us to use the property decorator
    # members, that are not exposed from the Client obj.
    self.local_vocab = vocabulary_node.Vocabulary(
        global_vocabulary,
        max_vocabulary_size=global_vocabulary.max_vocabulary_size)
    self.local_vocab.add_to_vocabulary(vocabulary_utils.VOCAB)
    self.vocab_size = self.local_vocab.max_vocabulary_size

    self.profile_length = profile_length
    self.dom_attribute_sequence_length = dom_attribute_sequence_length
    self.number_of_fields = number_of_fields
    self.structured_field_extractor = get_field_extractor(subdomain)
    self.number_of_dom_elements = number_of_dom_elements
    self.number_of_action_types = number_of_action_types
    self.keyboard_action_size = keyboard_action_size
    self.exact_wob_reward = exact_wob_reward
    self.number_of_dom_features = 8
    self.step_limit_reward = step_limit_reward
    self.verbose = True  # log dialogues
    self.verbose_prob = verbose_prob
    self.use_dom_profile_intersection = use_dom_profile_intersection
    self.current_actions = None  # only used within step
    self.episode_number = -1
    self.use_only_profile_key = use_only_profile_key
    self.final_reward_bias = final_reward_bias
    self.gminiwob_required_complexity = gminiwob_required_complexity
    self.gminiwob_unrequired_complexity = gminiwob_unrequired_complexity
    self.use_potential_based_reward = use_potential_based_reward
    self.generate_screenshots = generate_screenshots
    if kwargs_dict['threading'] and (
        gminiwob_required_complexity != 'original' or
        gminiwob_unrequired_complexity != 'original'):
      raise ValueError(
          'Setting environment difficult levels ({}, {}) via javascript requires no threading.'
          .format(gminiwob_required_complexity, gminiwob_unrequired_complexity))
    self.subtasks = subtasks
    self.randomized_within_groups = randomized_within_groups
    self.randomized_across_groups = randomized_across_groups
    self.randomized_env_components = randomized_env_components
    if self.number_of_fields != self.keyboard_action_size:
      raise ValueError(
          'Number of fields for structured profile '
          'should be equal to type action size but got {} != {}.'.format(
              self.number_of_fields, self.keyboard_action_size))

    if timestep_penalty < 0.0 or timestep_penalty > 1.0:
      raise ValueError(
          'Timestep penalty should be between 0.0 and 1.0 but got {}'.format(
              timestep_penalty))
    self.timestep_penalty = timestep_penalty
    if cyclic_action_penalty < 0.0 or cyclic_action_penalty > 1.0:
      raise ValueError(
          'Cyclic action discount should be between 0.0 and 1.0 but got {}'
          .format(cyclic_action_penalty))
    self.cyclic_action_penalty = cyclic_action_penalty

    # this assumes that the observations are already converted to numpy

    profile_space = gym.spaces.Box(
        low=0.0,
        high=float(self.vocab_size),
        shape=(
            number_of_fields,
            profile_length,
        ),
        dtype=np.int32)
    profile_mask_space = gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(
            number_of_fields,
            profile_length,
        ),
        dtype=np.float32)
    self.space_dict = {
        'profile_key': profile_space,
        'profile_value': profile_space,
        'profile_key_mask': profile_mask_space,
        'profile_value_mask': profile_mask_space
    }
    self.space_dict.update({
        'dom_elements':
            gym.spaces.Box(
                low=0.0,
                high=float(self.vocab_size),
                shape=(
                    number_of_dom_elements,
                    self.number_of_dom_attributes,
                    self.dom_attribute_sequence_length,
                ),
                dtype=np.int32),
        'dom_attribute_mask':  # this is to mask word embeddings
            gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(
                    number_of_dom_elements,
                    self.number_of_dom_attributes,
                    self.dom_attribute_sequence_length,
                ),
                dtype=np.float32),
        # mask dom elements and profile fields in action distribution
        'dom_profile_joint_mask':
            gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(number_of_fields, number_of_dom_elements),
                dtype=np.float32),
        # mask dom elements
        'dom_elements_mask':
            gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(number_of_dom_elements,),
                dtype=np.float32),
        'time_step':  # current time step
            gym.spaces.Box(
                low=0.0, high=self._step_limit, shape=(1,), dtype=np.float32),
    })
    if self.use_dom_profile_intersection:
      self.space_dict['dom_profile_intersection'] = gym.spaces.Box(
          low=0.0,
          high=float(self.vocab_size),
          shape=(
              number_of_dom_elements,
              self.number_of_dom_attributes,
              number_of_fields,
              2,  # key and value
              self.dom_attribute_sequence_length,
          ),
          dtype=np.int32)
      self.space_dict['dom_profile_intersection_mask'] = gym.spaces.Box(
          low=0.0,
          high=1.0,
          shape=(
              number_of_dom_elements,
              self.number_of_dom_attributes,
              number_of_fields,
              2,  # key and value
              self.dom_attribute_sequence_length,
          ),
          dtype=np.float32)
      self.space_dict['dom_profile_jaccard_sim'] = gym.spaces.Box(
          low=0.0,
          high=1.0,
          shape=(
              number_of_dom_elements,
              self.number_of_dom_attributes,
              number_of_fields,
              2,  # key and value
              3,  # jaccard, overlap/profile, overlap/dom
          ),
          dtype=np.float32)
    if self.number_of_dom_features > 0:
      self.space_dict['dom_features'] = gym.spaces.Box(
          low=0.0,
          high=1.0,
          shape=(
              number_of_dom_elements,
              self.number_of_dom_features,
          ),
          dtype=np.float32)  # focused, tampered, is_new, is_div
    self.observation_space = gym.spaces.Dict(self.space_dict)
    # We need a custom discrete space to configure dtype for acme framework
    self.action_space = custom_gym_spaces.Discrete(
        number_of_dom_elements * number_of_action_types * keyboard_action_size,
        np.int32)

    self._obs = None
    self._num_steps = 0
    self.current_reward = 0.0
    self.current_info = None
    self.done = False

  @property
  def mode(self):
    """Current mode (train / test)."""
    return self._mode

  def seed(self, seed=None):
    """Seeds internal random generator."""
    if seed is None:
      seed = np.random.randint(1000)
    self._random = np.random.RandomState(seed)
    return seed

  def set_environment_options(self):
    options = {}
    if self.gminiwob_required_complexity != 'original' or self.gminiwob_unrequired_complexity != 'original':
      options = {
          'requiredComplexity': self.gminiwob_required_complexity,
          'unrequiredComplexity': self.gminiwob_unrequired_complexity
      }
    if self.subtasks:
      options['subTasks'] = self.subtasks
    if self.randomized_within_groups:
      options['randomizedWithinGroups'] = self.randomized_within_groups
    if self.randomized_across_groups:
      options['randomizedAcrossGroups'] = self.randomized_across_groups
    if self.randomized_env_components:
      options['randomizedEnvComponents'] = self.randomized_env_components
    if self.verbose:
      logging.info('Options for environment randomization: %s', str(options))
    if options:
      for instance in self._wob_env.instances:
        try:
          instance.driver.execute_script('createEnvironment({});'.format(
              json.dumps(options)))
        except JavascriptException as e:
          logging.info(
              'Error in running createEnvironment(...) function in the environment: %s',
              str(e))

  def reset(self, raw_state=False):
    """Reset the wob environment and other related fields in this class.

    The main purpose of the 'reset' function is to initialize all episode level
    fields and call miniwob.reset() function to reset the low level
    environment state. Depending on the 'raw_state', it sets whether the raw
    observations or numpy-wrapped observations should be returned at each
    step. This allows wrapping the 'WebEnvironment' and adding custom code.

    Args:
      raw_state: If true, return unwrapped raw state.

    Returns:
      If not raw_state, returns wrapped representation of initial observation;
      otherwise returns initial observation.
    """
    # Number of steps in the episode to check if limit is reached.
    self._num_steps = 0
    # Current reward. This will be updated by custom reward shaping.
    self.current_reward = 0.0
    # Info from MiniWoB.
    self.current_info = {}
    # If the episode is terminated.
    self.done = False
    # Keeps the number of episodes that this environment instance had.
    self.episode_number += 1
    # Current action that is performed.
    self.current_actions = None
    # If true, verbose some information at each step.
    self.verbose = False
    # Decide to verbose or not.
    if np.random.random() < self.verbose_prob and self._verbose:
      self.verbose = True
    # A list of tuples that keep which dom element is used with which profile.
    self.dom_profile_acted_list = []
    # List of screenshots, mainly for generating episode gifs.
    self.screenshots = []
    # Potential of the previous state for reward shaping.
    self.prev_potential = 0.0
    # Keep track of which elements are already typed a text so that we can put
    # a space when we type another text to the element.
    self.typed_refs = set()
    # gMiniWoB allows tuning environment difficulty from outside if it is
    # implemented in the website.
    if self.subdomain.startswith('gminiwob'):
      self.set_environment_options()
    # Reset the low level observation from MiniWoB.
    self._obs = self._wob_env.reset([self.seed()])[0]
    # Generate the initial screenshot and save.
    if self.generate_screenshots:
      self.generate_screenshot_from_driver()
    # Extract user profile from the environment and parse.
    raw_profile = self.structured_field_extractor(self._obs.utterance.strip())
    self.tokenized_profile = [raw_profile[key] for key in raw_profile.keys]
    self.raw_profile = raw_profile

    if self.verbose:
      logging.info('New Episode @ %d', self.episode_number)
      logging.info('Profile: %s --[Parsed]--> %s', self._obs.utterance.strip(),
                   str(raw_profile))
    # Update the 'ref's of elements in the current observation.
    self.prev_refs = [
        dom_elem.ref for dom_elem in utils.get_dom_elements(self._obs)
    ]
    if not raw_state:  # wrap into a numpy array
      return web_observation_wrappers.wrap_observation(
          self._obs, self.structured_field_extractor, self._num_steps,
          self._step_limit, self.use_dom_profile_intersection,
          self.number_of_dom_features, self.local_vocab, self.profile_length,
          self.number_of_fields, self.dom_attribute_sequence_length,
          self.number_of_dom_attributes, self.prev_refs,
          self.number_of_dom_elements, self.use_only_profile_key,
          self.dom_profile_acted_list)
    else:
      return self._obs

  def step(self, action, raw_state=False):
    """Run the action in the WoB environment.

    The main purpose of the 'step' function is to convert the input action into
    a MiniWoB level action, execute the action on the environment, generate a
    new state, execute reward shaping, and wrap observation into numpy arrays.

    Action is a scalar value to be decomposed into an action type
    (type or click), type sequence action (a field from the instruction),
    and dom action (a dom element).

    Args:
      action: If it is a tuple, it is of the form (action_type, profile_index,
        dom_element_index). If it is scalar, it will be converted into this
        form.
      raw_state: If true, return unwrapped raw state.

    Returns:
      (new_state, reward, done, info)
      If not raw_state, returns wrapped representation of initial observation;
      otherwise returns initial observation.
    Raises:
      EnvironmentTerminateError: Raised when step is called after environment
      is done.
    """
    if self.done:
      raise EnvironmentTerminateError(
          'Step is called while environment is done.')

    # Convert input action to a web action tuple.
    action_type, profile_index, dom_element_index = self._convert_to_action_tuple(
        action)

    # Create miniwob level action.
    miniwob_action = self._create_miniwob_action(action_type, profile_index,
                                                 dom_element_index)

    # Execute the action.
    states, _, _, infos = self._execute_miniwob_action(miniwob_action)

    # Keep a list of user profile and elements that are already used together.
    self.dom_profile_acted_list.append((profile_index, dom_element_index))

    # Estimate final reward and difference between current and previous states.
    self._estimate_reward_and_diff(infos, states)

    # Increment step number.
    self._num_steps += 1

    # If raw state is needed, return the observation without wrapping.
    if raw_state:
      return self._obs, np.array(self.current_reward,
                                 np.float32), self.done, self.current_info

    # Log current step.
    if self.verbose:
      logging.info('Timestep@%d', self._num_steps)
      logging.info('Mouse : %s, Type : %s, DOM : %s',
                   str(self.current_actions[0]), str(self.current_actions[1]),
                   str(self.current_actions[2]))
      logging.info('System Action: %s', str(miniwob_action))
      logging.info('Reward : %f', self.current_reward)
      if self.use_potential_based_reward:
        logging.info('Potential : %f', self.prev_potential)

    # Return observation in numpy arrays.
    return self.wrap_observation(), np.array(
        self.current_reward, np.float32), self.done, self.current_info

  @property
  def utterance(self):
    """Returns utterance of observation if exists. Otherwise, None."""
    return self._obs.utterance if self._obs else None

  def wrap_observation(self, obs=None):
    """Wrap 'obs' if provided. Otherwise, return a pre-existing observation."""
    if not obs:
      obs = self._obs
    return web_observation_wrappers.wrap_observation(
        obs=obs,
        structured_field_extractor=self.structured_field_extractor,
        num_steps=self._num_steps,
        step_limit=self._step_limit,
        use_dom_profile_intersection=self.use_dom_profile_intersection,
        number_of_dom_features=self.number_of_dom_features,
        local_vocabulary=self.local_vocab,
        profile_length=self.profile_length,
        number_of_fields=self.number_of_fields,
        dom_attribute_sequence_length=self.dom_attribute_sequence_length,
        number_of_dom_attributes=self.number_of_dom_attributes,
        prev_refs=self.prev_refs,
        number_of_dom_elements=self.number_of_dom_elements,
        use_only_profile_key=self.use_only_profile_key,
        dom_profile_acted_list=self.dom_profile_acted_list)

  def _convert_to_action_tuple(self, action):
    """Convert a given action to an action tuple.

    Action is either a tuple or a scalar. It is usually a scalar as the RL
    frameworks require a flattened action space. What we do here is that
    take a flattened value of an action tuple and convert it back to its
    corresponding action tuple. Consider the following example where a policy
    network converts an input observation to an output 3D tensor of shape
    [2, number_of_profile_fields, number_of_dom_elements]. A sample from this
    output would correspond to an action tuple as above. However, to be more
    compatible with the RL libraries, we flatten this output tensor into a 1D
    array and sample from the array instead as the action.

    Args:
      action: If it is a tuple, it is of the form (action_type, profile_index,
        dom_element_index). If it is scalar, it will be converted into this
        form.

    Returns:
      A tuple of action type (click or keyboard), index of the field from the
      profile, and index of the dom element.
    """
    if isinstance(action, tuple):
      (action_type, profile_index, dom_element_index) = action
    else:
      action_type = int(
          action / (self.number_of_dom_elements * self.keyboard_action_size))
      action = action - action_type * (
          self.number_of_dom_elements * self.keyboard_action_size)
      profile_index = int(action / self.number_of_dom_elements)
      dom_element_index = action - profile_index * self.number_of_dom_elements
    self.current_actions = (action_type, profile_index, dom_element_index)
    return action_type, profile_index, dom_element_index

  def _create_miniwob_action(self, action_type, profile_index,
                             dom_element_index):
    """Create a miniwob level action that is executable.

    Args:
      action_type: Type of the action (click or keyboard for miniwob).
      profile_index: Index of the field from the profile.
      dom_element_index: Index of the dom element.

    Returns:
      Low level miniwob action that is executable on a miniwob environment.
    """
    # Environment step limit is reached, terminate the episode.
    if self._num_steps == self._step_limit:
      miniwob_action = MiniWoBTerminate()  # Doesn't update any state, done=True
    else:
      if profile_index < len(self.tokenized_profile):
        type_seq = self.tokenized_profile[profile_index]
      else:
        type_seq = ''
      # Generate the miniwob action. This will be directly run on the low level
      # environment.
      miniwob_action = utils.generate_web_action(
          utils.get_dom_elements(self._obs),
          action_type,
          dom_element_index,
          type_seq=type_seq,
          typed_refs=self.typed_refs)
    return miniwob_action

  def _execute_miniwob_action(self, miniwob_action):
    """Execute miniwob level action on the miniwob environment.

    Args:
      miniwob_action: A Low level miniwob action that is executable.

    Returns:
      Returns new state and reward, if the episode is done, and additional
      information from the miniwob environment.

    Raises:
      PotentialComputationError:
    """
    # Websites are sometimes unreliable. Instead of terminating the episode in
    # case the miniwob action is not successfully created, this returns the
    # previous state allowing the agent to take the same action again. The
    # reward is the same as the cyclic action penalty as semantically this is a
    # cyclic action.
    if not miniwob_action:
      return [self._obs], [np.float32(self.cyclic_action_penalty)
                          ], [self.done], {
                              'n': [self.current_info]
                          }
    try:
      # Run the miniwob action on the environment.
      states, rewards, dones, infos = self._wob_env.step([miniwob_action])
    except ValueError as e:
      logging.info(
          'Got a value error while getting utterance form the website: %s. Terminating episiode.',
          str(e))
      dones = [True]
      rewards = [0.0]

    self.done = dones[0]
    self.current_reward = rewards[0]
    return states, rewards, dones, infos

  def _compute_potential(self):
    """Compute potential based reward.

    Returns:
      A state dependent potential that monotonically indicates success.

    Raises:
      PotentialComputationError: Raised when potential computation is failed.
    """
    potential = 0.0
    for instance in self._wob_env.instances:
      try:
        potential += instance.driver.execute_script('return potential();')
      except JavascriptException as e:
        raise PotentialComputationError(
            f'Can not compute potential: {e}.') from e
    return potential

  def _estimate_reward_and_diff(self, infos, states):
    """Do reward shaping and also estimate difference between states.

    Args:
      infos: Additional information from the miniwob environment.
      states: States from miniwob environment instances. Currently, there is
        only one instance, hence only the 0th instance is used.
    """
    potential = None
    if self.use_potential_based_reward:
      potential = self._compute_potential()

    # Potential based reward shaping.
    if (not self.exact_wob_reward and self.use_potential_based_reward and
        potential is not None):  # Estimate potential based reward.
      self.current_reward += potential - self.prev_potential

    # Penalize if the step limit is reached but episode is not terminated.
    if self._num_steps >= self._step_limit:
      self.done = True
      self.current_reward += self.step_limit_reward

    if not self.done:
      # Keep screenshots.
      if self.generate_screenshots:
        self.generate_screenshot_from_driver()

      # Miniwob info for the 0th instance.
      self.current_info = infos.get('n', [{}])[0]

      # Difference between current and previous state.
      diff = states[0].dom.diff(self._obs.dom, interactive=False)

      # Keep a list of refs to elements. From this, a feature that corresponds
      # to if an element is new or it has been encountered before is computed.
      self.prev_refs = [
          dom_elem.ref for dom_elem in utils.get_dom_elements(self._obs)
      ]
      if self.verbose:
        logging.info('Diff: %s', str(diff))
      self._obs = states[0]

      # Estimate the final reward. If exact reward is needed, it will be the raw
      # sparse environment reward with the potential.
      if not self.exact_wob_reward:
        if not diff:
          # Diff is empty, meaning that the action didn't cause any change,
          # penalize it.
          self.current_reward -= self.cyclic_action_penalty
        # A time step penalty to encourage shortest solutions.
        self.current_reward -= self.timestep_penalty

    elif self._num_steps < self._step_limit:
      # Final state is reached, add a reward for reaching.
      self.current_reward += self.final_reward_bias

    if self.use_potential_based_reward:
      self.prev_potential = potential

  def close(self):
    """Close WoB Environment instances."""
    self._wob_env.close()

  def write_screenshots(self, screenshot_save_dir):
    """Write screenshots into a file.

    Args:
      screenshot_save_dir: Directory to save the screenshots.
    """
    for i, screenshot in enumerate(self.screenshots):
      screenshot.save(
          f'{screenshot_save_dir}/episode_{str(self.episode_number)}_{str(i)}.png'
      )

  def generate_screenshot_from_driver(self):
    """Generate a screenshot from current page."""
    self.screenshots.append(self.render())

  def render(self):
    """Render current observation."""
    return get_screenshot(self._wob_env.instances[0].driver,
                          self._wob_env.instances[0].task_width,
                          self._wob_env.instances[0].task_height)
