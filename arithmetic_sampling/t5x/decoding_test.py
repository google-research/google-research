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

"""Tests for decoding module."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import io_callback
import jax.numpy as jnp
import numpy as np
from t5x import decoding

from arithmetic_sampling.t5x import decoding as sampling

EOS_ID = 1
NEG_INF = decoding.NEG_INF


class DecodingTest(parameterized.TestCase):

  def test_arithmetic_sample_uneven_prefix(self):
    def token_to_logits(decoding_state):
      del decoding_state
      # Always sample id 2 for batch element 0 and id 3 for element 1.
      logits = np.array(
          [[-1e7, -1e7, 0, -1e7], [-1e7, -1e7, -1e7, 0]], dtype=np.float32
      )
      return logits, {}

    inputs = np.array([[0, 5, 7, 1, 0, 0], [0, 6, 1, 0, 0, 0]])
    rng = jax.random.PRNGKey(0)
    codes = decoding.flatten_beam_dim(sampling._make_default_codes(2, 1, rng))
    sampled_sequences, _ = sampling._arithmetic_sample_single_trial(
        inputs,
        codes,
        {},
        token_to_logits,
        EOS_ID,
        rng,
        topk=0,
        initial_index=np.array([3, 2]),
    )
    expected = np.array([[5, 7, 1, 2, 2, 2], [6, 1, 3, 3, 3, 3]])
    np.testing.assert_array_equal(expected, sampled_sequences)

  def test_arithmetic_sample_no_prefix(self):
    batch, max_decode_len = 2, 3

    def token_to_logits(decoding_state):
      del decoding_state
      # Always sample id 2 for batch element 0 and id 3 for element 1.
      logits = np.array(
          [[-1e7, -1e7, 0, -1e7], [-1e7, -1e7, -1e7, 0]], dtype=np.float32
      )
      return logits, {}

    inputs = np.zeros((batch, max_decode_len), dtype=np.int32)
    rng = jax.random.PRNGKey(0)
    codes = decoding.flatten_beam_dim(sampling._make_default_codes(2, 1, rng))
    sampled_sequences, _ = sampling._arithmetic_sample_single_trial(
        inputs, codes, {}, token_to_logits, EOS_ID, rng, topk=0
    )

    expected = [[2, 2, 2], [3, 3, 3]]
    np.testing.assert_array_equal(expected, sampled_sequences)

  def test_arithmetic_sample_prefix(self):
    def token_to_logits(decoding_state):
      del decoding_state
      # Always sample id 2 for batch element 0 and id 3 for element 1.
      logits = np.array(
          [[-1e7, -1e7, 0, -1e7], [-1e7, -1e7, -1e7, 0]], dtype=np.float32
      )
      return logits, {}

    # Batch element 0 has length 3 prefix and element 1 has length 2.
    inputs = np.array([[0, 5, 6, 7, 0], [0, 8, 9, 0, 0]], dtype=np.int32)
    rng = jax.random.PRNGKey(0)
    codes = decoding.flatten_beam_dim(sampling._make_default_codes(2, 1, rng))
    sampled_sequences, _ = sampling._arithmetic_sample_single_trial(
        inputs, codes, {}, token_to_logits, EOS_ID, rng, topk=0
    )

    expected = [[5, 6, 7, 2, 2], [8, 9, 3, 3, 3]]
    np.testing.assert_array_equal(expected, sampled_sequences)

  def test_arithmetic_sample_with_zero_temperature(self):
    batch, max_decode_len = 2, 3

    def token_to_logits(decoding_state):
      del decoding_state
      # Use very large logits that are close to one another.
      logits = np.array(
          [[1700.47, 1700.48, 1700.51, 1700.45], [3.2, 4.8, -5.3, 5.6]],
          dtype=np.float32,
      )
      return logits, {}

    inputs = np.zeros((batch, max_decode_len), dtype=np.int32)
    rng = jax.random.PRNGKey(0)
    codes = decoding.flatten_beam_dim(sampling._make_default_codes(2, 1, rng))
    sampled_sequences, _ = sampling._arithmetic_sample_single_trial(
        inputs, codes, {}, token_to_logits, EOS_ID, rng, topk=4, temperature=0.0
    )

    expected = [[2, 2, 2], [3, 3, 3]]
    np.testing.assert_array_equal(expected, sampled_sequences)

  def test_arithmetic_sample_prefix_ending_with_eos(self):
    def token_to_logits(decoding_state):
      del decoding_state
      # Always sample id 2 for batch element 0 and id 3 for element 1.
      logits = np.array(
          [[-1e7, -1e7, 0, -1e7], [-1e7, -1e7, -1e7, 0]], dtype=np.float32
      )
      return logits, {}

    # Batch element 0 has length 4 prefix (including the initial dummy token and
    # the last eos) and element 1 has length 3.
    inputs = np.array([[0, 5, 6, 1, 0], [0, 8, 1, 0, 0]], dtype=np.int32)
    rng = jax.random.PRNGKey(0)
    codes = decoding.flatten_beam_dim(sampling._make_default_codes(2, 1, rng))
    sampled_sequences, _ = sampling._arithmetic_sample_single_trial(
        inputs, codes, {}, token_to_logits, EOS_ID, rng, topk=1
    )

    expected = [[5, 6, 1, 2, 2], [8, 1, 3, 3, 3]]
    np.testing.assert_array_equal(expected, sampled_sequences)

  def test_arithmetic_sample_with_state_callback(self):
    def token_to_logits(decoding_state):
      del decoding_state
      # A distribution with roughly all probability mass in sample id 3.
      logits = np.array(
          [[-1e7, -1e7, -1e7, 0], [-1e7, -1e7, -1e7, 0]], dtype=np.float32
      )
      return logits, {}

    def state_callback_fn(state):
      def callback_fn(current_index_and_sequences):
        """Add EOS token after first time token id 3 has been sampled."""
        current_index, sequences = current_index_and_sequences
        sequences = np.array(sequences)
        for i in range(len(current_index)):
          if sequences[i, current_index[i]] == 3:
            sequences[i, current_index[i] + 1] = EOS_ID
        return sequences

      sequences = io_callback(
          callback_fn,
          jax.ShapeDtypeStruct(state.sequences.shape, state.sequences.dtype),
          (state.cur_index, state.sequences),
      )
      return state.replace(sequences=sequences)

    inputs = np.array([[0, 5, 6, 7, 0], [0, 8, 9, 0, 0]], dtype=np.int32)
    rng = jax.random.PRNGKey(0)
    codes = decoding.flatten_beam_dim(sampling._make_default_codes(2, 1, rng))
    sampled_sequences, _ = sampling._arithmetic_sample_single_trial(
        inputs,
        codes,
        {},
        token_to_logits,
        EOS_ID,
        rng,
        topk=0,
        temperature=0.0,
        state_callback_fn=state_callback_fn,
    )

    expected = [[5, 6, 7, 3, EOS_ID], [8, 9, 3, EOS_ID, 0]]
    np.testing.assert_array_equal(expected, sampled_sequences)

  def test_arithmetic_sample_with_logit_callback(self):
    def token_to_logits(decoding_state):
      del decoding_state
      # Uniform distribution over targets from model.
      logits = np.array(
          [[-1e7, -1e7, -1e7, -1e7], [-1e7, -1e7, -1e7, -1e7]], dtype=np.float32
      )
      return logits, {}

    def logit_callback_fn(logits, state):
      del state  # Unused.
      # Rewrite logits to always sample id 2 for batch element 0 and
      # id 3 for element 1.
      logits[0, 2] = 0
      logits[1, 3] = 0
      return logits

    # Batch element 0 has length 3 prefix and element 1 has length 2.
    inputs = np.array([[0, 5, 6, 7, 0], [0, 8, 9, 0, 0]], dtype=np.int32)
    rng = jax.random.PRNGKey(0)
    codes = decoding.flatten_beam_dim(sampling._make_default_codes(2, 1, rng))
    sampled_sequences, _ = sampling._arithmetic_sample_single_trial(
        inputs,
        codes,
        {},
        token_to_logits,
        EOS_ID,
        rng,
        topk=0,
        temperature=0.0,
        logit_callback_fn=logit_callback_fn,
    )

    expected = [[5, 6, 7, 2, 2], [8, 9, 3, 3, 3]]
    np.testing.assert_array_equal(expected, sampled_sequences)

  def test_arithmetic_sample_prefix_ending_with_eos_early_stop(self):
    batch, max_decode_len = 2, 7
    rng0 = jax.random.PRNGKey(0)

    ret = [np.array([2, 3]) for _ in range(max_decode_len)]
    # Sequence 1 outputs EOS=1 when i = 3 where `i` is the while loop counter of
    # `decoding._temperature_sample_single_trial`.
    ret[3] = np.array([2, 1])
    # Sequence 0 outputs EOS=1 when i = 4.
    ret[4] = np.array([1, 3])
    ret = jax.numpy.array(ret)

    def mocked_categorical(rng_input, logits, codes):  # pylint: disable=unused-argument
      """Ignores logit and codes and returns only based on the rng_input."""
      rng = rng0
      k = 0
      # Mimic the rng split done in `decoding.sample_loop_body_fn`.
      for j in range(max_decode_len):
        rng1, rng = jax.random.split(rng)
        # We want to sift out `j` for which rng1 == rng_input
        # rngs are a pair of ints. So sum the bool and divide by 2.
        k += j * (rng1 == rng_input).sum() // 2
      # `k` at this point is equal to the while loop variable `i` of the caller.
      return ret[k], codes

    def token_to_logits(decoding_state):
      del decoding_state
      # These values are not used in this test because random.categorical is
      # directly mocked.
      dummy_logits = np.zeros((batch, 4), dtype=np.float32)
      return dummy_logits, {}

    inputs = np.array(
        [[0, 5, 1, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0]], dtype=np.int32
    )
    codes = decoding.flatten_beam_dim(sampling._make_default_codes(2, 1, rng0))
    with mock.patch.object(
        sampling, '_arithmetic_categorical', new=mocked_categorical
    ):
      sampled_sequences, _ = sampling._arithmetic_sample_single_trial(
          inputs, codes, {}, token_to_logits, EOS_ID, rng0, topk=0
      )

    expected = [[5, 1, 2, 2, 1, 0, 0], [8, 3, 3, 1, 0, 0, 0]]
    np.testing.assert_array_equal(expected, sampled_sequences)

  def test_greedy_decoding_topk_sample_log_probs(self):
    def token_to_logits(decoding_state):
      del decoding_state
      # Sample [2, 3] with probability [0.6, 0.4].
      logits = np.array(
          [[-1e7, -1e7, -0.510825624, -0.916290732]], dtype=np.float32
      )
      return logits, {}

    inputs = np.array([[0, 2, 2, 2, 0]], dtype=np.int32)
    rng = jax.random.PRNGKey(0)
    input_codes = decoding.flatten_beam_dim(
        sampling._make_default_codes(1, 1, rng)
    )
    sampled_sequences, sampled_log_probs = (
        sampling._arithmetic_sample_single_trial(
            inputs,
            input_codes,
            {},
            token_to_logits,
            EOS_ID,
            rng,
            topk=1,
            rescale_log_probs=True,
        )
    )

    expected_sequence = [[2, 2, 2, 2, 2]]
    expected_log_probs = [0.0]
    np.testing.assert_array_equal(expected_sequence, sampled_sequences)
    np.testing.assert_array_almost_equal(expected_log_probs, sampled_log_probs)

    inputs = np.array([[0, 2, 2, 3, 0]], dtype=np.int32)
    rng = jax.random.PRNGKey(0)
    input_codes = decoding.flatten_beam_dim(
        sampling._make_default_codes(1, 1, rng)
    )
    sampled_sequences, sampled_log_probs = (
        sampling._arithmetic_sample_single_trial(
            inputs,
            input_codes,
            {},
            token_to_logits,
            EOS_ID,
            rng,
            topk=1,
            rescale_log_probs=False,
        )
    )

    expected_sequence = [[2, 2, 3, 2, 2]]
    expected_log_probs = [-1.02165125]
    np.testing.assert_array_equal(expected_sequence, sampled_sequences)
    np.testing.assert_array_almost_equal(expected_log_probs, sampled_log_probs)

  def test_arithmetic_sample_log_prob(self):
    batch, max_decode_len = 2, 7
    rng0 = jax.random.PRNGKey(0)

    ret = [np.array([2, 3]) for _ in range(max_decode_len)]
    # Sequence 1 outputs EOS=1 when i = 3 where `i` is the while loop counter of
    # `decoding._temperature_sample_single_trial`.
    ret[3] = np.array([2, 1])
    # Sequence 0 outputs EOS=1 when i = 4.
    ret[4] = np.array([1, 3])
    ret = jax.numpy.array(ret)

    def mocked_categorical(rng_input, logits, codes):  # pylint: disable=unused-argument
      """Ignores logit and codes and returns only based on the rng_input."""
      rng = rng0
      k = 0
      # Mimic the rng split done in `decoding.sample_loop_body_fn`.
      for j in range(max_decode_len):
        rng1, rng = jax.random.split(rng)
        # We want to sift out `j` for which rng1 == rng_input
        # rngs are a pair of ints. So sum the bool and divide by 2.
        k += j * (rng1 == rng_input).sum() // 2
      # `k` at this point is equal to the while loop variable `i` of the caller.
      return ret[k], codes

    logits = np.random.randn(batch, 4)
    token_to_logits = lambda decoding_state: (logits, {})
    inputs = np.array(
        [[0, 5, 1, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0]], dtype=np.int32
    )
    codes = decoding.flatten_beam_dim(sampling._make_default_codes(2, 1, rng0))
    with mock.patch.object(
        sampling, '_arithmetic_categorical', new=mocked_categorical
    ):
      sampled_sequences, log_prob = sampling._arithmetic_sample_single_trial(
          inputs, codes, {}, token_to_logits, EOS_ID, rng0, topk=0
      )

    log_probs = jax.nn.log_softmax(logits)
    expected = [[5, 1, 2, 2, 1, 0, 0], [8, 3, 3, 1, 0, 0, 0]]
    expected_log_prob = [
        log_probs[0, 2] + log_probs[0, 2] + log_probs[0, 1],
        log_probs[1, 3] + log_probs[1, 3] + log_probs[1, 1],
    ]
    expected_log_prob = np.array(expected_log_prob)
    np.testing.assert_array_equal(expected, sampled_sequences)
    np.testing.assert_allclose(expected_log_prob, log_prob, atol=1e-5)

  def test_arithmetic_sample_num_decodes(self):
    num_decodes = 3
    rng0 = jax.random.PRNGKey(0)
    inputs = np.array([[0, 5, 1, 0], [0, 8, 7, 0]], dtype=np.int32)

    with mock.patch.object(
        sampling, '_arithmetic_sample_single_trial'
    ) as mocked:
      # Expanded_decodes: [batch * num_decodes, max_decode_len]
      expanded_decodes = np.array([
          [5, 1, 4, 4],
          [5, 1, 5, 5],
          [5, 1, 3, 3],
          [8, 7, 5, 5],
          [8, 7, 3, 3],
          [8, 7, 4, 4],
      ])
      # Expanded_log_prob: [batch * num_decodes]
      expanded_log_prob = np.array([-2.3, -1.3, -3.6, -0.5, -2.5, -1.9])
      mocked.return_value = expanded_decodes, expanded_log_prob

      decodes, scores = sampling.arithmetic_sample(
          inputs, {}, mock.Mock(), EOS_ID, rng0, num_decodes=num_decodes
      )

      expanded_inputs = jnp.array([
          [0, 5, 1, 0],
          [0, 5, 1, 0],
          [0, 5, 1, 0],
          [0, 8, 7, 0],
          [0, 8, 7, 0],
          [0, 8, 7, 0],
      ])
      # Test that the actual decode function is called with the expanded values.
      np.testing.assert_array_equal(mocked.call_args[0][0], expanded_inputs)

    np.testing.assert_array_equal(
        decodes,
        [
            [[5, 1, 3, 3], [5, 1, 4, 4], [5, 1, 5, 5]],
            [[8, 7, 3, 3], [8, 7, 4, 4], [8, 7, 5, 5]],
        ],
    )
    np.testing.assert_allclose(scores, [[-3.6, -2.3, -1.3], [-2.5, -1.9, -0.5]])

  def test_arithmetic_sample_num_decodes_with_initial_index(self):
    num_decodes = 3
    rng0 = jax.random.PRNGKey(0)
    inputs = np.array([[0, 5, 1, 0], [0, 8, 7, 0]], dtype=np.int32)
    initial_index = np.array([1, 2], dtype=np.int32)

    with mock.patch.object(
        sampling, '_arithmetic_sample_single_trial'
    ) as mocked:
      with mock.patch.object(decoding, 'cache_map') as mocked_cache_map:
        # Expanded_decodes: [batch * num_decodes, max_decode_len]
        expanded_decodes = np.array([
            [5, 1, 4, 4],
            [5, 1, 5, 5],
            [5, 1, 3, 3],
            [8, 7, 5, 5],
            [8, 7, 3, 3],
            [8, 7, 4, 4],
        ])
        # Expanded_log_prob: [batch * num_decodes]
        expanded_log_prob = np.array([-2.3, -1.3, -3.6, -0.5, -2.5, -1.9])
        mocked.return_value = expanded_decodes, expanded_log_prob

        decodes, scores = sampling.arithmetic_sample(
            inputs,
            {},
            mock.Mock(),
            EOS_ID,
            rng0,
            num_decodes=num_decodes,
            initial_index=initial_index,
        )

        expanded_inputs = jnp.array([
            [0, 5, 1, 0],
            [0, 5, 1, 0],
            [0, 5, 1, 0],
            [0, 8, 7, 0],
            [0, 8, 7, 0],
            [0, 8, 7, 0],
        ])
        expanded_initial_index = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
        # Test that the actual decode function is called with the expanded
        # values.
        np.testing.assert_array_equal(mocked.call_args[0][0], expanded_inputs)
        np.testing.assert_array_equal(
            mocked.call_args[1]['initial_index'], expanded_initial_index
        )
        # Test that the function was applied to the index in the cache map.
        self.assertTrue(mocked_cache_map.call_args[1]['apply_to_index'])

    np.testing.assert_array_equal(
        decodes,
        [
            [[5, 1, 3, 3], [5, 1, 4, 4], [5, 1, 5, 5]],
            [[8, 7, 3, 3], [8, 7, 4, 4], [8, 7, 5, 5]],
        ],
    )
    np.testing.assert_allclose(scores, [[-3.6, -2.3, -1.3], [-2.5, -1.9, -0.5]])

  @parameterized.named_parameters(
      dict(
          testcase_name='no_initial_index',
          initial_index=None,
          expected_calls=6,
      ),
      dict(
          testcase_name='initial_index',
          initial_index=np.array([1, 2], dtype=np.int32),
          expected_calls=4,
      ),
      dict(
          testcase_name='lower_initial_index',
          initial_index=np.array([1, 1], dtype=np.int32),
          expected_calls=5,  # We decode 4 tokens out of the prompt.
      ),
  )
  def test_arithmetic_sample_max_decode_steps_with_initial_index(
      self, initial_index, expected_calls
  ):
    max_decode_steps = 4
    rng0 = jax.random.PRNGKey(0)
    inputs = np.array(
        [[0, 2, 0, 0, 0, 0, 0, 0], [0, 2, 2, 0, 0, 0, 0, 0]], dtype=np.int32
    )

    token_to_logits = mock.Mock()
    token_to_logits.return_value = (
        np.array(
            [[-1e7, -1e7, -1e7, 0], [-1e7, -1e7, -1e7, 0]], dtype=np.float32
        ),
        {},
    )

    # Unroll while loop.
    with jax.disable_jit():
      decodes, scores = sampling.arithmetic_sample(
          inputs,
          {},
          token_to_logits,
          EOS_ID,
          rng0,
          initial_index=initial_index,
          topk=4,
          max_decode_steps=max_decode_steps,
      )

    self.assertLen(token_to_logits.call_args_list, expected_calls)

    expected_output = np.array(
        [[2, 3, 3, 3, 3, 0, 0, 0], [2, 2, 3, 3, 3, 3, 0, 0]]
    )
    expected_output = jnp.expand_dims(expected_output, 1)

    np.testing.assert_array_equal(decodes, expected_output)
    np.testing.assert_array_equal(scores, [[0.0], [0.0]])

  def test_arithmetic_sample_max_decode_steps_endpad(self):
    max_decode_steps = 4
    rng0 = jax.random.PRNGKey(0)
    inputs = np.array(
        [
            [0, 2, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 2, 2, 0],
            [0, 2, 2, 2, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    initial_index = np.array([1, 6, 0])

    token_to_logits = mock.Mock()
    token_to_logits.return_value = (
        np.array(
            [
                [-1e7, -1e7, -1e7, 0],
                [-1e7, -1e7, -1e7, 0],
                [-1e7, -1e7, -1e7, 0],
            ],
            dtype=np.float32,
        ),
        {},
    )

    # Unroll while loop.
    with jax.disable_jit():
      decodes, scores = sampling.arithmetic_sample(
          inputs,
          {},
          token_to_logits,
          EOS_ID,
          rng0,
          initial_index=initial_index,
          topk=4,
          max_decode_steps=max_decode_steps,
      )

    # `inputs[2]` starts from index 0. So it requires 3 calls to
    # `token_to_logits` to exit the prompt (these generated tokens are
    # overridden) and 4 more calls to fill the rest. `inputs[0]` only need 4
    # calls. In the last 3 calls, it generates but MUST NOT populate the
    # sequences because it is already ended.
    self.assertLen(token_to_logits.call_args_list, 7)
    expected_output = np.array(
        [
            [2, 3, 3, 3, 3, 0, 0, 0],
            [2, 2, 2, 2, 2, 2, 3, 3],
            [2, 2, 2, 3, 3, 3, 3, 0],
        ],
        dtype=np.int32,
    )
    expected_output = jnp.expand_dims(expected_output, 1)

    np.testing.assert_array_equal(decodes, expected_output)
    np.testing.assert_allclose(scores, [[0.0], [0.0], [0.0]])

  def test_arithmetic_sample_max_decode_steps_docstring_ex4(self):
    max_decode_steps = 2
    rng0 = jax.random.PRNGKey(0)
    inputs = np.array(
        [[0, 2, 0, 0, 0, 0, 0, 0], [0, 3, 4, 0, 0, 0, 0, 0]], dtype=np.int32
    )
    initial_index = np.array([1, 2])

    token_to_logits = mock.Mock()
    token_to_logits.return_value = (
        np.array(
            [[-1e7, -1e7, 0, -1e7], [-1e7, -1e7, -1e7, 0]], dtype=np.float32
        ),
        {},
    )

    # Unroll while loop.
    with jax.disable_jit():
      decodes, _ = sampling.arithmetic_sample(
          inputs,
          {},
          token_to_logits,
          EOS_ID,
          rng0,
          initial_index=initial_index,
          topk=4,
          max_decode_steps=max_decode_steps,
      )
    self.assertLen(token_to_logits.call_args_list, 2)
    expected_output = np.array(
        [[2, 2, 2, 0, 0, 0, 0, 0], [3, 4, 3, 3, 0, 0, 0, 0]], dtype=np.int32
    )
    expected_output = jnp.expand_dims(expected_output, 1)

    np.testing.assert_array_equal(decodes, expected_output)

  def test_arithmetic_sample_max_decode_steps_hard_limit(self):
    max_decode_steps = 10
    max_decode_steps_hard_limit = 4
    rng0 = jax.random.PRNGKey(0)
    inputs = np.array(
        [[0, 2, 0, 0, 0, 0, 0, 0], [0, 2, 2, 0, 0, 0, 0, 0]], dtype=np.int32
    )

    token_to_logits = mock.Mock()
    token_to_logits.return_value = (
        np.array(
            [[-1e7, -1e7, -1e7, 0], [-1e7, -1e7, -1e7, 0]], dtype=np.float32
        ),
        {},
    )

    # Unroll while loop.
    with jax.disable_jit():
      decodes, scores = sampling.arithmetic_sample(
          inputs,
          {},
          token_to_logits,
          EOS_ID,
          rng0,
          topk=4,
          max_decode_steps=max_decode_steps,
          max_decode_steps_hard_limit=max_decode_steps_hard_limit,
      )

    expected_output = np.array(
        [[2, 3, 3, 3, 3, 0, 0, 0], [2, 2, 3, 3, 3, 3, 0, 0]]
    )
    expected_output = jnp.expand_dims(expected_output, 1)

    np.testing.assert_array_equal(decodes, expected_output)
    np.testing.assert_array_equal(scores, [[0.0], [0.0]])

  def test_arithmetic_sample_topp(self):
    with jax.disable_jit():
      rng0 = jax.random.PRNGKey(0)
      inputs = np.zeros((1, 20), dtype=np.int32)

      token_to_logits = mock.Mock()

      # Logits correspond to (0.3, 0, 0.1, 0.6).
      token_to_logits.return_value = (
          np.array([[-1.2, -1e7, -2.3, -0.51]], dtype=np.float32),
          {},
      )

      decodes, scores = sampling.arithmetic_sample(
          inputs, {}, token_to_logits, EOS_ID, rng0, topp=0.55, topk=0
      )  # Anything under 0.6 will trigger deterministic decoding.

      expected_output = np.array([[3] * 20])
      expected_output = jnp.expand_dims(expected_output, 1)

      np.testing.assert_array_equal(decodes, expected_output)
      np.testing.assert_array_equal(scores, [[0.0]])

      # Temperature is applied first, so the distribution becomes
      # (0.27, 0, 0.069, 0.65), so if topp is 0.63, it should become greedy.
      decodes, scores = sampling.arithmetic_sample(
          inputs,
          {},
          token_to_logits,
          EOS_ID,
          rng0,
          temperature=0.8,
          topp=0.63,
          topk=0,
      )

      expected_output = np.array([[3] * 20])
      expected_output = jnp.expand_dims(expected_output, 1)

      np.testing.assert_array_equal(decodes, expected_output)
      np.testing.assert_array_equal(scores, [[0.0]])

  def test_dynamic_topp_max_decode_steps(self):
    rng0 = jax.random.PRNGKey(0)
    inputs = np.zeros((1, 20), dtype=np.int32)

    token_to_logits = mock.Mock()

    # Logits correspond to (0.3, 0, 0.1, 0.6).
    token_to_logits.return_value = (
        np.array([[-1.2, -1e7, -2.3, -0.51]], dtype=np.float32),
        {},
    )

    def dynamic_decode_fn(inputs, temperature, topp, max_decode_steps):
      return sampling.arithmetic_sample(
          inputs,
          {},
          token_to_logits,
          EOS_ID,
          rng0,
          temperature=temperature,
          topp=topp,
          topk=0,
          max_decode_steps=max_decode_steps,
      )

    dynamic_decode_fn_jit = jax.jit(dynamic_decode_fn)

    decodes, scores = dynamic_decode_fn_jit(inputs, 0.8, 0.63, 10)

    expected_output = np.array([[3] * 10 + [0] * 10])
    expected_output = jnp.expand_dims(expected_output, 1)

    np.testing.assert_array_equal(decodes, expected_output)
    np.testing.assert_array_equal(scores, [[0.0]])

  def test_topp_log_probs(self):
    rng0 = jax.random.PRNGKey(0)
    inputs = np.zeros((1, 1), dtype=np.int32)

    token_to_logits = mock.Mock()

    # Logits correspond to (0.3, 0, 0.1, 0.6).
    token_to_logits.return_value = (
        np.array([[-1.2, NEG_INF, -2.3, -0.51]], dtype=np.float32),
        {},
    )

    with jax.disable_jit():
      # This lets us see logits after topp and topk are applied.

      with mock.patch.object(sampling, '_arithmetic_categorical') as mocked:
        mocked.return_value = jnp.array([0], dtype=jnp.int32), jnp.array(
            [0], dtype=jnp.int32
        )
        decodes, _ = sampling.arithmetic_sample(
            inputs,
            {},
            token_to_logits,
            EOS_ID,
            rng0,
            temperature=1.4,
            topp=0.7,
            topk=0,
        )

    self.assertLen(token_to_logits.call_args_list, 1)
    np.testing.assert_array_equal(decodes, jnp.asarray([[[0]]]))

    np.testing.assert_array_almost_equal(
        mocked.call_args_list[0][0][1],
        jnp.asarray([[-0.85714293, NEG_INF, NEG_INF, -0.36428571]]),
    )

  def test_sequential_cumsum(self):
    test_arr = jnp.arange(0, 1000)
    test_cumsum = sampling._sequential_cumsum(test_arr, 0)
    target_cumsum = np.cumsum(jnp.asarray(test_arr))
    np.testing.assert_array_equal(jnp.asarray(test_cumsum), target_cumsum)


if __name__ == '__main__':
  absltest.main()
