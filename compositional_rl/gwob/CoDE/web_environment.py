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

"""GMiniWoB Web Environment."""

import json
import time

from absl import logging
from CoDE import base_web_environment
from CoDE import web_primitives
from CoDE.utils import get_dom_elements
import gin
from miniwob.action import MiniWoBTerminate
import numpy as np
from selenium.common.exceptions import JavascriptException


class AddPrimitiveError(base_web_environment.BaseJavascriptError):
  """Raised when a primitive can not be added to the website."""


class AddTransitionError(base_web_environment.BaseJavascriptError):
  """Raised when a transition can not be added to the website."""


class UseAbstractNavigationError(base_web_environment.BaseJavascriptError):
  """Raised when abstract navigation can not be set in the website."""


class FetchActionablePrimitivesError(base_web_environment.BaseJavascriptError):
  """Raised when all primitives can not be fetched from the website."""


class VisitPrimitivesError(base_web_environment.BaseJavascriptError):
  """Raised when a primitive can not be visited in the website."""


@gin.configurable('GMiniWoBWebEnvironment')
class GMiniWoBWebEnvironment(base_web_environment.WebEnvironment):
  """GMiniWoB environment wrapper."""

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
      use_potential_based_reward=True,
      generate_screenshots=False,
      # Distribution parameters.
      global_vocabulary=None,
      # Kwargs for WoB Env.
      kwargs_dict=None,
      # GMiniWoB related parameters
      use_conceptual=False,
      transitions_given=False,
      # Primitive params.
      primitive_names=None,
      primitive_to_design_dict=None,
      primitive_transitions_to_design_dict=None):
    super().__init__(
        base_url=base_url,
        subdomain=subdomain,
        profile_length=profile_length,
        number_of_fields=number_of_fields,
        use_only_profile_key=use_only_profile_key,
        number_of_dom_elements=number_of_dom_elements,
        dom_attribute_sequence_length=dom_attribute_sequence_length,
        use_dom_profile_intersection=use_dom_profile_intersection,
        number_of_action_types=number_of_action_types,
        keyboard_action_size=keyboard_action_size,
        step_limit=step_limit,
        cyclic_action_penalty=cyclic_action_penalty,
        timestep_penalty=timestep_penalty,
        exact_wob_reward=exact_wob_reward,
        step_limit_reward=step_limit_reward,
        final_reward_bias=final_reward_bias,
        mode=mode,
        verbose=verbose,
        verbose_prob=verbose_prob,
        gminiwob_required_complexity=gminiwob_required_complexity,
        gminiwob_unrequired_complexity=gminiwob_unrequired_complexity,
        subtasks=subtasks,
        randomized_within_groups=randomized_within_groups,
        randomized_across_groups=randomized_across_groups,
        randomized_env_components=randomized_env_components,
        use_potential_based_reward=use_potential_based_reward,
        generate_screenshots=generate_screenshots,
        global_vocabulary=global_vocabulary,
        kwargs_dict=kwargs_dict)
    self._use_conceptual = use_conceptual
    # Set to true to use predefined transitions.
    self._transitions_given = transitions_given

    # Init bookkeeping to None, so it is clear that it is neither True or False.
    self._is_task_completed = None

    self._primitive_names = web_primitives.CONCEPTS
    if primitive_names:
      self._primitive_names = primitive_names
    self._primitive_to_design_dict = web_primitives.CONCEPTS2DESIGN
    if primitive_to_design_dict:
      self._primitive_to_design_dict = primitive_to_design_dict
    self._primitive_transitions_to_design_dict = web_primitives.TRANSITIONS2DESIGN
    if primitive_transitions_to_design_dict:
      self._primitive_transitions_to_design_dict = primitive_transitions_to_design_dict

  def design_environment(self,
                         env_design,
                         auto_num_pages=True,
                         additional_transitions=None):
    """Design an environment from primitives and transitions.

    This function calls multiple javascript functions to update the HTML of the
    website and wrap around miniwob for RL interaction.

    Primitives should be unique per page, ex. a navigation bar or first name
    field can only appear once in a page.
    An example env_design object is {"number_of_nodes": 5,
      "node_assignments": [0,1], "primitive_assignments": [5,12]}. where
      primitive_assignments denote primitives
    and node_assignments denote their corresponding pages. Transitions in this
    example is defined by predefined events associated to certain elements.
    See web_primitives.CONCEPTS and web_primitives.TRANSITIONS
    for a sample of primitives and transitions.

    Args:
      env_design: A dictionary of environment designs. Each field is a tuple of
        (primitive, page index) or (transition, page index) pairs in addition to
        number of pages.
      auto_num_pages: If true, number of pages will be deduced from
        node_assignments field rather than number_of_nodes. This ensures there
        are no empty dangling pages.
      additional_transitions: List of additional transitions to add to the
        website.
    """
    if self._use_conceptual:  # set conceptual in website
      self._set_conceptual_in_website()
    logging.info('Environment design object: %s', str(env_design))
    num_pages = env_design.get('number_of_nodes',
                               env_design.get('number_of_pages'))
    if auto_num_pages:
      num_pages = 0
    actions = env_design.get('primitive_assignments', env_design.get('action'))
    pages = env_design.get('node_assignments', env_design.get('action_page'))
    env_design_node = {'actions': []}
    env_design_edge = {'actions': []}
    selected_designs = []
    for action, page in zip(actions, pages):
      concept = self._primitive_names[action]
      if not self._primitive_to_design_dict[concept]:
        continue
      primitive_design = self._primitive_to_design_dict[concept]
      if not isinstance(primitive_design, list):
        primitive_design = [primitive_design]
      primitive_designs = []
      for des in primitive_design:
        if web_primitives.PAGE_PH in des:
          primitive_designs.append(
              des.replace(web_primitives.PAGE_PH, str(page)))
        else:
          primitive_designs.append(des)
      if (concept,
          page) not in selected_designs:  # Check if design is duplicate.
        for design in primitive_designs:
          if '"is_transition": true' in design:  # this is a transition
            env_design_edge['actions'].append(json.loads(design))
          else:  # This is a primitive.
            env_design_node['actions'].append(json.loads(design))
            if auto_num_pages:  # Deduce number of pages.
              num_pages = max(
                  num_pages,
                  env_design_node['actions'][-1].get('source_page', -1))
        selected_designs.append((concept, page))
      else:  # There is a duplicate design, skipping.
        logging.info('%s is duplicate.', str(primitive_design))
    # Number of pages is 0 indexed.
    env_design_node['num_pages'] = num_pages + 1
    env_design_node = str(env_design_node).replace('True', 'true').replace(
        'False', 'false')
    self._design_environment_add_nodes(
        env_design_node)  # Add nodes or primitives to pages.

    if additional_transitions:
      env_design_edge['actions'].extend(additional_transitions)
    env_design_edge = str(env_design_edge).replace('True', 'true').replace(
        'False', 'false')
    if self._transitions_given:  # Transitions are predefined.
      design = self._primitive_transitions_to_design_dict.copy()
      design = {
          'actions': [
              json.loads(value)
              for value in list(design.values())
              if '"is_transition": true' in value
          ]
      }
      self._design_environment_add_edges(
          json.dumps(design).replace('True', 'true').replace('False', 'false'))
    else:  # These are generated outside.
      print(env_design_edge)
      self._design_environment_add_edges(env_design_edge)

  def _set_conceptual_in_website(self):
    """Set USE_CONCEPTUAL to true in the website html.

    Raises:
      UseAbstractNavigationError: Raised when the abstract navigation can't
        be set in the website.
    """
    for instance in self._wob_env.instances:
      try:
        instance.driver.execute_script('USE_CONCEPTUAL=true;')
        print('Set conceptual in the website')
      except JavascriptException as e:
        raise UseAbstractNavigationError(
            f'Can not add primitives to the website: {e}') from e

  def _design_environment_add_nodes(self, env_design):
    """Add primitives to the website.

    Args:
      env_design: A set of primitives with their corresponding page indices.

    Raises:
      AddPrimitiveError: Raised when adding primitives to the website fails.
    """
    logging.info('Add primitives: %s', str(env_design))
    for instance in self._wob_env.instances:
      try:
        instance.driver.execute_script(
            'return addNodes({}, addDummyField=true);'.format(env_design))
      except JavascriptException as e:
        raise AddPrimitiveError('Can not add primitives to the website: '
                                f'{e}.') from e

  def _design_environment_add_edges(self, env_design):
    """'Add transitions to the website.

    Args:
      env_design: A set of transitions with their corresponding elements.

    Raises:
      AddTransitionEror: Raised when adding transitions to the website fails.
    """
    logging.info('Add edges: %s', str(env_design))
    for instance in self._wob_env.instances:
      try:
        instance.driver.execute_script(
            'return addEdges({});'.format(env_design))
      except JavascriptException as e:
        raise AddTransitionError('Can not add transitions to the website: '
                                 f'{e}.') from e

  def get_all_actionable_primitives(self):
    """Returns a list of all actionable primitives from the website.

    Raises:
      FetchActionablePrimitivesError: Raised when fetching all actionable
        element ids fails.
    """
    all_primitives = []
    for instance in self._wob_env.instances:
      try:
        primitives = instance.driver.execute_script(
            'return $("[id^=actionable_]").map(function(){return $(this).attr("id");}).get();'
        )
        all_primitives.extend(primitives)
      except JavascriptException as e:
        raise FetchActionablePrimitivesError(
            'Can not fetch all actionable primitives: '
            f'{e}.') from e
    return sorted(list(set(all_primitives)))

  def get_all_actionable_primitives_with_classnames(self):
    """Returns a list of all actionable primitives from the website.

    This extends the get_all_actionable_primitives by returning also element
    classes.

    Raises:
      FetchActionablePrimitivesError: Raised when fetching all actionable
        element ids and class names fails.
    """
    all_visited = []
    for instance in self._wob_env.instances:
      try:
        visited = instance.driver.execute_script(
            'return $("[id^=actionable_]").map(function(){return {"id": $(this).attr("id"), "class": $(this).attr("class")};}).get();'
        )
        all_visited.extend(visited)
      except JavascriptException as e:
        raise FetchActionablePrimitivesError(
            'Can not fetch all actionable primitives '
            f'with their class names: {e}.') from e
    return all_visited

  def visit_primitive(self, concept, value=None):
    """Visit a primitive and execute an action using value as input.

    Primitives are identified by unique concepts for each page.

    Args:
      concept: A concept that uniquely identifies a primitive at each page.
      value: A string value that will be used to execute an action on the
        primitive.

    Raises:
      VisitPrimitivesError: Raised when visiting a primitive with a given value
        fails.
    """
    for instance in self._wob_env.instances:
      try:
        instance.driver.execute_script('return visitGroup("{}", "{}")'.format(
            concept,
            value).replace('"None"',
                           'null').replace('"True"',
                                           'true').replace('"False"', 'false'))
      except JavascriptException as e:
        raise VisitPrimitivesError('Can not do visit based navigation: '
                                   f'{e}.') from e

  def _act_on_element_with_given_value(self, element, value=None):
    """Visit an element with a given value.

      Element HTML ids uniquely define a concept for each page. An example would
      be "navbar_1_3" which corresponds to the first navbar item at page 3. Page
      indices are optional and if they are not given, concept uniquely
      identifies
      the primitive across all website, ex. "first_name" where this is a core
      primitive and its value will be used in the reward estimation.

      This function also wraps miniwob step function where we don't use selenium
      to execute actions on the website but rather use javascript. This can
      easily be replaced by moving javascript logic to python.

    Args:
      element: An element from a DOM tree.
      value: A string value from user profile.

    Returns:
      A tuple of state, reward, done, and additional info.

    Raises:
      RuntimeError: Raised when miniwob.step fails.
    """
    # Visit the element and take an action using the given value.
    self.visit_primitive(element.id, value)

    # Initialize.
    states = [None] * len(self._wob_env.instances)
    rewards = [-1.] * len(self._wob_env.instances)
    dones = [True] * len(self._wob_env.instances)
    info = {'n': [{} for _ in self._wob_env.instances]}

    # Loop over instances but currently there is only one instance.
    for instance in self._wob_env.instances:
      try:
        metadata = instance.get_metadata()
        i = instance.index
        # Fetch reward.
        rewards[i] = instance.reward_processor(metadata)
        # Fetch done.
        dones[i] = metadata['done']
        if not metadata['done']:
          # Fetch state.
          if not instance.cache_state:
            states[i] = instance.get_state()
          else:
            states[i] = instance.initial_state
        metadata['elapsed'] = max(0., time.time() - instance.start_time)
        info['n'][i] = metadata
      except RuntimeError as e:
        raise RuntimeError(f'Can not do miniwob.step: {e}') from e
    return states, rewards, dones, info

  @property
  def is_task_completed(self):
    """Returns a boolean telling whether the task is completed in the current step."""
    return self._is_task_completed

  def _compute_potential(self):
    """Compute reward potential from the environment.

    Returns:
      A state dependent potential that monotonically indicates success.
    Raises:
      PotentialComputationError: Raised when potential computation is failed.
    """
    potential = 0.0
    for instance in self._wob_env.instances:
      try:
        if self._use_conceptual:
          # In abstract navigation, we only care about visiting an element and
          # profile is irrelevant. This requires a different potential
          # estimation.
          potential += instance.driver.execute_script(
              'return conceptualPotential();')
        else:  # Potential for the real web navigation problem.
          potential += instance.driver.execute_script('return potential(true);')
      except JavascriptException as e:
        raise base_web_environment.PotentialComputationError(
            'Can not compute potential:'
            f'{e}.') from e
    return potential

  def _convert_to_action_tuple(self, action):
    """Convert a given action to an action tuple.

    Action is either a tuple or a scalar. It is usually a scalar as the RL
    frameworks require a flattened action space. What we do here is that
    take a flattened value of an action tuple and convert it back to its
    corresponding action tuple. Consider the following example where a policy
    network converts an input observation to an output 2D tensor of shape
    [number_of_profile_fields, number_of_dom_elements]. A sample from this
    output would correspond to an action tuple as above. However, to be more
    compatible with the RL libraries, we flatten this output tensor into a 1D
    array and sample from the array instead as the action.

    Args:
      action: If it is a tuple, it is of the form (value of profile, dom element
        index). If it is scalar, it will be converted into this form.

    Returns:
      A tuple of action type (click or keyboard), index of the field from the
      profile, and index of the dom element.
    """
    if self._use_conceptual:  # Abstract navigation, action is an element index.
      profile_value = None
      dom_element_index = action
    else:  # Action is a scalar corresponding to a tuple of (element, profile).
      key = int(action / self.number_of_dom_elements)
      dom_element_index = action - key * self.number_of_dom_elements
      profile_value = self.tokenized_profile[key] if key >= 0 and key < len(
          self.tokenized_profile) else None  # Points to the value of a field.
    return dom_element_index, profile_value

  def _get_focus_element(self, dom_element_index):
    # List of all elements.
    elements_in_dom = get_dom_elements(self._obs)
    element = None
    try:  # Get the corresponding element.
      element = elements_in_dom[dom_element_index]
    except IndexError as e:
      # Element index is invalid, terminating the episode.
      logging.info(
          'Element is not found in the page. The index of the element was %s'
          ' and the number of elements in the DOM was %s. Make sure the DOM is'
          ' not empty and the index is within the bounds of the DOM. Original'
          ' error was: %s', dom_element_index, len(elements_in_dom), str(e))
    return element

  def _execute_miniwob_action(self, element, profile_value):
    """Execute the action on the miniwob environment.

    This function doesn't directly create a miniwob action but uses a
    gminiwob specific javascript function to execute the action. The javascript
    function also selects the action type, providing an abstraction.

    Args:
      element: A DOM element instance.
      profile_value: A string value from the user profile.

    Returns:
      Returns new state and reward, if the episode is done, and additional
      information from the miniwob environment.
    """
    if not element:
      return [self._obs], [np.array(self.cyclic_action_penalty,
                                    np.float32)], [self.done], {
                                        'n': [self.current_info]
                                    }
    if self._num_steps == self._step_limit:
      # Reached the limit of the episode, terminate environment.
      click = MiniWoBTerminate()  # Doesn't update any state, done=True.
      states, rewards, dones, infos = self._wob_env.step([click])
    else:  # Run the action on element with value and return state and reward.
      states, rewards, dones, infos = self._act_on_element_with_given_value(
          element, profile_value)

    self.done = dones[0]
    self._is_task_completed = bool(rewards[0] == 1)  # a boolean
    self.current_reward = rewards[0]
    return states, rewards, dones, infos

  def step(self, action, raw_state=False):
    """Run the action in the WoB environment and get observation and reward.

      There are two types of actions: (i) (Abstract Navigation) Action directly
      refers to an element and profile is irrelevant, (ii) Action refers to a
      pair of elements and profile fields. In both cases, action is a scalar,
      as available frameworks only support scalar action spaces, and is
      converted into a tuple in the abstract navigation case. To use
      abstract navigation, pass 'use_conceptual=True' on object instantiation.

      For example, with abstract navigation, 'action=5' refers to the 5-th
      element in the DOM tree where the tree is linearized using the
      'get_dom_elements' function. Without abstract navigation, '5' refers to
      both profile and element indices, i.e., (element_index, profile_index)
      where 'action=profile_index*number_of_dom_elements+element_index'.


    Args:
      action: A scalar composite action of element and profile tuples or only
        elements.
      raw_state: If true, return unwrapped raw state instead of GYM compatible
        states.

    Returns:
      (state, reward, done, info)
      If not raw_state, returns numpy wrapped DOM and profile observations;
      otherwise returns raw observations from environment.

    Raises:
      EnvironmentTerminateError: Raised when step is called after environment
      is done.
    """
    if self.done:
      raise base_web_environment.EnvironmentTerminateError(
          'Step is called while environment is done.')

    # Convert input action to a web action tuple.
    dom_element_index, profile_value = self._convert_to_action_tuple(action)

    # Fetch the focus element from the DOM.
    element = self._get_focus_element(dom_element_index)

    # Execute the action.
    states, _, _, infos = self._execute_miniwob_action(element, profile_value)

    # Estimate final reward and difference between current and previous states.
    self._estimate_reward_and_diff(infos, states)

    # Increment step number.
    self._num_steps += 1

    if raw_state:  # If raw state, return state without wrapping in numpy.
      return self._obs, np.array(self.current_reward,
                                 np.float32), self.done, self.current_info

    # Log current step.
    if self.verbose:
      logging.info('Timestep@%d', self._num_steps)
      logging.info('Action : %s, %s', str(element), profile_value)
      logging.info('Reward : %f', self.current_reward)
      if self.use_potential_based_reward:
        logging.info('Potential : %f', self.prev_potential)

    # Return observation in numpy arrays.
    return self.wrap_observation(), np.array(
        self.current_reward, np.float32), self.done, self.current_info

  def close(self):
    """Close WoB Environment instances."""
    self._wob_env.close()

  @property
  def state(self):
    """Environment state to save."""
    return {'word_to_id': self.local_vocab.local_vocab}

  def load(self, state):
    """Load environment parameters that are updated during training.

    Args:
      state: State to load from.
    """
    self.local_vocab._local_vocab = state['word_to_id']  # pylint: disable=protected-access
    self.id_to_word = dict([
        (self.local_vocab[w], w) for w in self.local_vocab.local_vocab
    ])
