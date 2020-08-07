# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Find expression by Monte Carlo Tree Search guided by neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from neural_guided_symbolic_regression.mcts import policies
from neural_guided_symbolic_regression.mcts import rewards
from neural_guided_symbolic_regression.mcts import states
from neural_guided_symbolic_regression.models import metrics
from neural_guided_symbolic_regression.models import partial_sequence_model_generator


class NeuralProductionRuleAppendPolicy(policies.PolicyBase):
  """Appends a valid production rule on existing list of production rules.

  The probabilities of the actions will be determined by the partial sequence
  model.
  """

  def __init__(self, sess, grammar, max_length, symbolic_properties_dict):
    """Initializer.

    Args:
      sess: tf.Session, the session contains the trained model to predict next
          production rule from input partial sequence. If None, each step will
          be selected randomly.
      grammar: arithmetic_grammar.Grammar object.
      max_length: Integer, the max length of production rule sequence.
      symbolic_properties_dict: Dict, the keys are the symbolic properties used
          as conditions. Values are the corresponding desired values of the
          symbolic properties.
    """
    self._sess = sess
    self._grammar = grammar
    self._max_length = max_length

    conditions = {}
    if symbolic_properties_dict is not None:
      conditions.update({
          key: np.array([value], dtype=np.float32)
          for key, value in symbolic_properties_dict.iteritems()
      })
    self._conditions = conditions

  def get_new_states_probs(self, state):
    """Gets new state from current state by appending a valid production rule.

    Args:
      state: A mcts.states.ProductionRulesState object. Contains a list of
          nltk.grammar.Production objects in attribute
          production_rules_sequence.

    Returns:
      new_states: A list of next states. Each state is a result from apply an
          action in the instance attribute actions to the input state.
      action_probs: A float numpy array with shape [num_actions,]. The
          probability of each action in the class attribute actions.

    Raises:
      TypeError: If input state is not states.ProductionRulesState object.
    """
    if not isinstance(state, states.ProductionRulesState):
      raise TypeError('Input state shoud be an instance of '
                      'states.ProductionRulesState but got %s' % type(state))

    production_rules_sequence = state.production_rules_sequence
    if len(production_rules_sequence) > self._max_length:
      # Do not allow the length of production rules sequence exceed _max_length.
      # All nan probabilities will stop the rollout in MCTS.
      masked_probabilities = [np.nan] * self._grammar.num_production_rules
    else:
      masked_probabilities = (
          partial_sequence_model_generator.get_masked_probabilities_from_model(
              sess=self._sess,
              max_length=self._max_length,
              partial_sequence=[self._grammar.prod_rule_to_index[str(prod_rule)]
                                for prod_rule in production_rules_sequence],
              next_production_rule_mask=self._grammar.masks[
                  self._grammar.lhs_to_index[state.stack_peek()]],
              conditions=self._conditions))

    new_states = []
    action_probs = []
    for probability, production_rule in zip(
        masked_probabilities, self._grammar.prod_rules):
      if state.is_valid_to_append(production_rule):
        new_state = state.copy()
        new_state.append_production_rule(production_rule)
        new_states.append(new_state)
        action_probs.append(probability)
      else:
        new_states.append(None)
        action_probs.append(np.nan)
    action_probs = np.asarray(action_probs)
    action_probs /= np.nansum(action_probs)
    return new_states, action_probs


class LeadingPowers(rewards.RewardBase):
  """Computes reward for univariate expression only on leading powers.

  This reward measures a univariate expression by whether this expression
  satisfies the desired leading powers at 0 and infinity.

  reward = -abs(leading power difference at 0)
      - abs(leading power difference at infinity))
  """

  def __init__(
      self,
      leading_at_0,
      leading_at_inf,
      variable_symbol='x',
      post_transformer=None,
      allow_nonterminal=False,
      default_value=None):
    """Initializer.

    Args:
      leading_at_0: Float, desired leading power at 0.
      leading_at_inf: Float, desired leading power at inf.
      variable_symbol: String, the symbol of variable in function expression.
      post_transformer: Callable. This function takes one float number and
          output a float number as the transformed value of input. It is used
          to post-transformation the reward evaluated on a state. Default None,
          no post-transformation will be applied.
      allow_nonterminal: Boolean, if False, ValueError will be raised when
          list of symbols to evaluate contains non-terminal symbol and
          default_value is None. Default False.
      default_value: Float, if allow_nonterminal is False and non-terminal
          symbol exists, instead of raising a ValueError, return default_value
          as the reward value.
    """
    super(LeadingPowers, self).__init__(
        post_transformer=post_transformer,
        allow_nonterminal=allow_nonterminal,
        default_value=default_value)
    self._leading_at_0 = leading_at_0
    self._leading_at_inf = leading_at_inf
    self._variable_symbol = variable_symbol

  def get_leading_power_error(self, state):
    """Gets the leading power error.

    The leading power error is defined as
    abs(leading power difference at 0) + abs(leading power difference at inf).

    Args:
      state: mcts.states.StateBase object. Records all the information of
          expression.

    Returns:
      Float.
    """
    true_leading_at_0, true_leading_at_inf = (
        metrics.evaluate_leading_powers_at_0_inf(
            expression_string=state.get_expression(),
            symbol=self._variable_symbol))

    return (abs(true_leading_at_0 - self._leading_at_0)
            + abs(true_leading_at_inf - self._leading_at_inf))

  def _evaluate(self, state):
    """Evaluates the reward from input state.

    Args:
      state: mcts.states.StateBase object. Records all the information of
          expression.

    Returns:
      Float, the reward of the current state.
    """
    leading_power_error = self.get_leading_power_error(state)
    if np.isfinite(leading_power_error):
      return -float(leading_power_error)
    else:
      return self._default_value


class NumericalPointsAndLeadingPowers(LeadingPowers):
  """Computes reward for univariate expression with leading powers and values.

  This reward measures an univariate expression in two aspects:
  1. The mean square error of numerical values defined by input_values and
     output_values.
  2. Whether this expression satisfies the desired leading powers at 0 and
     infinity.

  hard_penalty_default_value decides whether to use soft or hard penalty when
  the expression does not match the desired leading powers.

  Soft penalty
    reward = (
        -(root mean square error)
        - abs(leading power difference at 0)
        - abs(leading power difference at infinity))

  Hard penalty
    If leading power at 0 and infinity are both correct
      reward = -(root mean square error)
    Otherwise reward = hard_penalty_default_value

  If include_leading_powers is False, the reward is just
  -(root mean square error).
  """

  def __init__(
      self,
      input_values,
      output_values,
      leading_at_0,
      leading_at_inf,
      hard_penalty_default_value=None,
      variable_symbol='x',
      include_leading_powers=True,
      post_transformer=None,
      allow_nonterminal=False,
      default_value=None):
    """Initializer.

    Args:
      input_values: Numpy array with shape [num_input_values]. List of input
          values to univariate function.
      output_values: Numpy array with shape [num_output_values]. List of output
          values from the univariate function.
      leading_at_0: Float, desired leading power at 0.
      leading_at_inf: Float, desired leading power at inf.
      hard_penalty_default_value: Float, the default value for hard penalty.
          Default None, the reward will be computed by soft penalty instead of
          hard penalty.
      variable_symbol: String, the symbol of variable in function expression.
      include_leading_powers: Boolean, whether to include leading powers in
          reward.
      post_transformer: Callable. This function takes one float number and
          output a float number as the transformed value of input. It is used
          to post-transformation the reward evaluated on a state. Default None,
          no post-transformation will be applied.
      allow_nonterminal: Boolean, if False, ValueError will be raised when
          list of symbols to evaluate contains non-terminal symbol and
          default_value is None. Default False.
      default_value: Float, if allow_nonterminal is False and non-terminal
          symbol exists, instead of raising a ValueError, return default_value
          as the reward value.
    """
    super(NumericalPointsAndLeadingPowers, self).__init__(
        leading_at_0=leading_at_0,
        leading_at_inf=leading_at_inf,
        variable_symbol=variable_symbol,
        post_transformer=post_transformer,
        allow_nonterminal=allow_nonterminal,
        default_value=default_value)
    self._input_values = input_values
    self._output_values = output_values
    self._include_leading_powers = include_leading_powers
    self._hard_penalty_default_value = hard_penalty_default_value

  def get_input_values_rmse(self, state):
    """Evaluates root mean square error on input_values.

    Args:
      state: mcts.states.StateBase object. Records all the information of
          expression.

    Returns:
      Float.
    """
    expression_output_values = metrics.evaluate_expression(
        expression_string=state.get_expression(),
        grids=self._input_values,
        symbol=self._variable_symbol)

    return np.sqrt(
        np.mean((expression_output_values - self._output_values) ** 2))

  def _evaluate(self, state):
    """Evaluates the reward from input state.

    Args:
      state: mcts.states.StateBase object. Records all the information of
          expression.

    Returns:
      Float, the reward of the current state.
    """
    input_values_rmse = self.get_input_values_rmse(state)
    if not self._include_leading_powers:
      if np.isfinite(input_values_rmse):
        return -input_values_rmse
      else:
        return self._default_value
    # NOTE(leeley): If computing the leading power fails
    # (timeout or sympy ValueError) or functions in symbolic_properties return
    # nan (for example, 1 / (x - x)).
    leading_power_error = self.get_leading_power_error(state)

    if self._hard_penalty_default_value is None:
      # Soft penalty.
      if np.isfinite(leading_power_error):
        return -input_values_rmse - leading_power_error
      else:
        return self._default_value
    else:
      # Hard penalty.
      if (np.isfinite(leading_power_error)
          and np.isclose(leading_power_error, 0)):
        return -input_values_rmse
      else:
        return self._hard_penalty_default_value
