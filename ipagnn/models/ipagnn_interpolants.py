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

# Lint as: python3
"""Learned Interpreters IPA-GNN models with ablations / interpolants."""

from absl import logging  # pylint: disable=unused-import
from flax import nn
import jax
from jax import lax
import jax.numpy as jnp

from ipagnn.modules import common_modules

Embed = common_modules.Embed
StackedRNNCell = common_modules.StackedRNNCell
Tag = common_modules.Tag


def create_lstm_cells(n):
  """Creates a list of n LSTM cells."""
  cells = []
  for i in range(n):
    cell = nn.LSTMCell.partial(
        gate_fn=nn.sigmoid,
        activation_fn=nn.tanh,
        kernel_init=nn.initializers.xavier_uniform(),
        recurrent_kernel_init=nn.initializers.orthogonal(),
        bias_init=nn.initializers.zeros,
        name=f'lstm_{i}',
    )
    cells.append(cell)
  return cells


class IPAGNNInterpolant(nn.Module):
  """IPAGNN interpolant model with batch dimension."""

  def apply(self, inputs, info, config, train=False, cache=None):
    # Inputs
    output_token_vocabulary_size = info.output_vocab_size
    true_indexes = inputs['true_branch_nodes']
    false_indexes = inputs['false_branch_nodes']
    start_indexes = inputs['start_index']  # pylint: disable=unused-variable
    exit_indexes = inputs['exit_index']
    steps_all = inputs['steps']
    vocab_size = info.features[info._builder.key('statements')].vocab_size  # pylint: disable=protected-access
    hidden_size = config.model.hidden_size
    data = inputs['data'].astype('int32')
    batch_size, num_nodes, unused_statement_length = data.shape

    # An upper bound on the number of steps to take.
    max_steps = int(1.5 * info.max_diameter)

    # Init parameters
    def emb_init(key, shape, dtype=jnp.float32):
      return jax.random.uniform(
          key, shape, dtype,
          -config.initialization.maxval,
          config.initialization.maxval)

    embed = Embed.shared(num_embeddings=vocab_size,
                         features=hidden_size,
                         emb_init=emb_init,
                         name='embed')
    branch_decide_dense = nn.Dense.shared(
        name='branch_decide_dense',
        features=2,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    cells = create_lstm_cells(config.model.rnn_cell.layers)
    lstm = StackedRNNCell.shared(cells=cells)
    if config.model.interpolant.apply_dense:
      dense_parent_to_true_child = nn.Dense.shared(
          name='dense_parent_to_true_child',
          features=hidden_size,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6))
      dense_parent_to_false_child = nn.Dense.shared(
          name='dense_parent_to_false_child',
          features=hidden_size,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6))
      dense_true_child_to_parent = nn.Dense.shared(
          name='dense_true_child_to_parent',
          features=hidden_size,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6))
      dense_false_child_to_parent = nn.Dense.shared(
          name='dense_false_child_to_parent',
          features=hidden_size,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6))
    if config.model.interpolant.apply_gru:
      gru_cell = nn.recurrent.GRUCell.shared(name='gru_cell')
    output_dense = nn.Dense.shared(
        name='output_dense',
        features=output_token_vocabulary_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))

    # Apply
    def execute_single_node(hidden_state, node_embedding):
      carry, _ = lax.scan(lstm, hidden_state, node_embedding)
      return carry
    execute = jax.vmap(execute_single_node)  # Single example.

    def branch_decide_single_node(hidden_state):
      # leaves(hidden_state).shape: hidden_size
      hidden_state_concat = jnp.concatenate(
          jax.tree_leaves(hidden_state), axis=0)
      return branch_decide_dense(hidden_state_concat)
    branch_decide = jax.vmap(branch_decide_single_node)

    def update_instruction_pointer(
        instruction_pointer, branch_decisions, true_indexes, false_indexes):
      # instruction_pointer.shape: num_nodes,
      # branch_decisions: num_nodes, 2,
      # true_indexes: num_nodes,
      # false_indexes: num_nodes
      p_true = branch_decisions[:, 0]
      p_false = branch_decisions[:, 1]
      if not config.model.interpolant.use_b:
        p_true = jnp.ones_like(p_true)
        p_false = jnp.ones_like(p_false)
      if not config.model.interpolant.use_p:
        instruction_pointer = jnp.ones_like(instruction_pointer)
      true_contributions = jax.ops.segment_sum(
          p_true * instruction_pointer, true_indexes,
          num_segments=num_nodes)
      false_contributions = jax.ops.segment_sum(
          p_false * instruction_pointer, false_indexes,
          num_segments=num_nodes)
      return true_contributions + false_contributions

    def aggregate(
        hidden_states, instruction_pointer, branch_decisions,
        true_indexes, false_indexes):
      # leaves(hidden_states).shape: num_nodes, hidden_size
      # instruction_pointer.shape: num_nodes,
      # branch_decisions: num_nodes, 2,
      # true_indexes: num_nodes,
      # false_indexes: num_nodes
      p_true = branch_decisions[:, 0]
      p_false = branch_decisions[:, 1]
      if not config.model.interpolant.use_b:
        p_true = jnp.ones_like(p_true)
        p_false = jnp.ones_like(p_false)
      if not config.model.interpolant.use_p:
        instruction_pointer = jnp.ones_like(instruction_pointer)
      denominators = update_instruction_pointer(
          instruction_pointer, branch_decisions, true_indexes, false_indexes)
      denominators += 1e-7
      # denominator.shape: num_nodes,
      if not config.model.interpolant.normalize:
        denominators = jnp.ones_like(denominators)

      def aggregate_component(h):
        # h.shape: num_nodes
        # p_true.shape: num_nodes
        # instruction_pointer.shape: num_nodes
        true_contributions = jax.ops.segment_sum(
            h * p_true * instruction_pointer, true_indexes,
            num_segments=num_nodes)
        false_contributions = jax.ops.segment_sum(
            h * p_false * instruction_pointer, false_indexes,
            num_segments=num_nodes)
        # *_contributions.shape: num_nodes, hidden_size
        return (true_contributions + false_contributions) / denominators
      aggregate_component = jax.vmap(aggregate_component, in_axes=1, out_axes=1)

      return jax.tree_map(aggregate_component, hidden_states)

    def step_single_example(hidden_states, instruction_pointer,
                            node_embeddings, true_indexes, false_indexes,
                            exit_index):
      # Execution (e.g. apply RNN)
      # leaves(hidden_states).shape: num_nodes, hidden_size
      # instruction_pointer.shape: num_nodes,
      # node_embeddings.shape: num_nodes, statement_length, hidden_size
      if config.model.interpolant.apply_code_rnn:
        hidden_state_contributions = execute(hidden_states, node_embeddings)
        # leaves(hidden_state_contributions).shape: num_nodes, hidden_size
      else:
        hidden_state_contributions = hidden_states

      if config.model.interpolant.apply_dense:
        parent_to_true_child = jax.tree_map(
            dense_parent_to_true_child, hidden_state_contributions)
        parent_to_false_child = jax.tree_map(
            dense_parent_to_false_child, hidden_state_contributions)
        true_child_to_parent = jax.tree_map(
            dense_true_child_to_parent, hidden_state_contributions)
        false_child_to_parent = jax.tree_map(
            dense_false_child_to_parent, hidden_state_contributions)
      else:
        parent_to_true_child = hidden_state_contributions
        parent_to_false_child = hidden_state_contributions
        true_child_to_parent = hidden_state_contributions
        false_child_to_parent = hidden_state_contributions

      # Use the exit node's hidden state as it's hidden state contribution
      # to avoid "executing" the exit node.
      def mask_h(h_contribution, h):
        return h_contribution.at[exit_index, :].set(h[exit_index, :])
      hidden_state_contributions = jax.tree_multimap(
          mask_h, hidden_state_contributions, hidden_states)

      # Branch decisions (e.g. Dense layer)
      branch_decision_logits = branch_decide(hidden_state_contributions)
      branch_decisions = nn.softmax(branch_decision_logits, axis=-1)

      # Update state
      if config.model.interpolant.use_ipa:
        instruction_pointer_new = update_instruction_pointer(
            instruction_pointer, branch_decisions, true_indexes, false_indexes)
        hidden_states_new = aggregate(
            hidden_state_contributions, instruction_pointer, branch_decisions,
            true_indexes, false_indexes)
      else:
        assert config.model.interpolant.use_parent_embeddings
        assert config.model.interpolant.use_child_embeddings
        instruction_pointer_new = instruction_pointer
        normalization = jnp.sqrt(
            2 + (  # Each node has a true and false child.
                # jnp.bincount(true_indexes, minlength=num_nodes)
                jax.ops.segment_sum(
                    jnp.ones_like(true_indexes), true_indexes,
                    num_segments=num_nodes)
                # + jnp.bincount(false_indexes, minlength=num_nodes)
                + jax.ops.segment_sum(
                    jnp.ones_like(false_indexes), false_indexes,
                    num_segments=num_nodes)
            )
        )
        # normalization.shape: num_nodes,
        def aggregate_parent_and_child_contributions(p1, p2, c3, c4):
          return (
              jax.ops.segment_sum(
                  p1, true_indexes, num_segments=num_nodes)
              + jax.ops.segment_sum(
                  p2, false_indexes, num_segments=num_nodes)
              + c3[true_indexes]
              + c4[false_indexes]
          ) / normalization[:, None]
        hidden_states_new = jax.tree_multimap(
            aggregate_parent_and_child_contributions,
            parent_to_true_child,
            parent_to_false_child,
            true_child_to_parent,  # true_child_to_parent[child] -> parent
            false_child_to_parent)
      if config.model.interpolant.apply_gru:
        def apply_gru(h2, h1):
          output, _ = gru_cell(h2, h1)
          return output
        hidden_states_new = (
            jax.tree_multimap(apply_gru, hidden_states_new, hidden_states))

      to_tag = {
          'branch_decisions': branch_decisions,
          'hidden_state_contributions': hidden_state_contributions,
          'hidden_states_before': hidden_states,
          'hidden_states': hidden_states_new,
          'instruction_pointer_before': instruction_pointer,
          'instruction_pointer': instruction_pointer_new,
          'true_indexes': true_indexes,
          'false_indexes': false_indexes,
      }
      return hidden_states_new, instruction_pointer_new, to_tag

    def compute_logits_single_example(
        hidden_states, instruction_pointer, exit_index, steps,
        node_embeddings, true_indexes, false_indexes):
      """single_example refers to selecting a single exit node hidden state."""
      # leaves(hidden_states).shape: num_nodes, hidden_size

      def step_(carry, _):
        hidden_states, instruction_pointer, index = carry
        hidden_states_new, instruction_pointer_new, to_tag = (
            step_single_example(
                hidden_states, instruction_pointer,
                node_embeddings, true_indexes, false_indexes,
                exit_index)
        )
        carry = jax.tree_multimap(
            lambda new, old, index=index: jnp.where(index < steps, new, old),
            (hidden_states_new, instruction_pointer_new, index + 1),
            (hidden_states, instruction_pointer, index + 1),
        )
        return carry, to_tag
      if config.model.ipagnn.checkpoint and not self.is_initializing():
        step_ = jax.checkpoint(step_)

      carry = (hidden_states, instruction_pointer, jnp.array([0]))
      (hidden_states, instruction_pointer, _), to_tag = lax.scan(
          step_, carry, None, length=max_steps)

      final_state = jax.tree_map(lambda hs: hs[exit_index], hidden_states)
      # leaves(final_state).shape: hidden_size
      final_state_concat = jnp.concatenate(jax.tree_leaves(final_state), axis=0)
      logits = output_dense(final_state_concat)
      to_tag.update({
          'instruction_pointer_final': instruction_pointer,
          'hidden_states_final': hidden_states,
      })
      return logits, to_tag
    compute_logits = jax.vmap(compute_logits_single_example,
                              in_axes=(0, 0, 0, 0, 0, 0, 0))

    # Init state
    node_embeddings = embed(data)
    # node_embeddings.shape:
    #     batch_size, num_nodes, statement_length, hidden_size
    hidden_states = StackedRNNCell.initialize_carry(
        jax.random.PRNGKey(0), cells, (batch_size, num_nodes,), hidden_size)
    if config.model.interpolant.init_with_code_embeddings:
      hidden_states = jax.vmap(execute)(hidden_states, node_embeddings)
    # leaves(hidden_states).shape: batch_size, num_nodes, hidden_size
    instruction_pointer = jax.ops.index_add(
        jnp.zeros((batch_size, num_nodes,)),
        jax.ops.index[:, 0],  # TODO(dbieber): Use "start_index" instead of 0.
        1
    )
    # instruction_pointer.shape: batch_size, num_nodes,

    logits, to_tag = compute_logits(
        hidden_states, instruction_pointer, exit_indexes, steps_all,
        node_embeddings, true_indexes, false_indexes)
    for key, value in to_tag.items():
      value = Tag(value, name=key)
    logits = jnp.expand_dims(logits, axis=1)
    return logits
