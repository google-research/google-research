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

import unittest

from flax import linen as nn
import jax
import jax.numpy as jnp
from study_recommend import models
from study_recommend import types

INPUT_FIELDS = types.ModelInputFields
VOCAB_SIZE = 100
ATOL = 1e-4
SEP_TOKEN = 99
TEST_MODEL_CONFIG = models.TransformerConfig(
    output_vocab_size=VOCAB_SIZE,
    dtype=jnp.bfloat16,
    emb_dim=32,
    num_heads=2,
    num_layers=2,
    qkv_dim=32,
    mlp_dim=32,
    max_len=15,
    dropout_rate=0.1,
    deterministic=True,
    decode=False,
    kernel_init=nn.initializers.xavier_uniform(),
    bias_init=nn.initializers.normal(stddev=1e-6),
    separator_token_value=SEP_TOKEN,
)

SEQ_LEN = 10


def generate_individual_test_data():
  titles = jnp.arange(1, 11)[None, :]  # starting from 1 because 0 is padding
  input_positions = jnp.arange(SEQ_LEN)[None, :]
  student_ids = jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
  grade_lvl = jnp.array([1], dtype=jnp.int32).reshape(1)
  return {
      INPUT_FIELDS.TITLES: titles,
      INPUT_FIELDS.INPUT_POSITIONS: input_positions,
      INPUT_FIELDS.STUDENT_IDS: student_ids,
      INPUT_FIELDS.GRADE_LEVELS: grade_lvl,
  }


def generate_individual_test_data_batch():
  sample = generate_individual_test_data()
  sample = {
      key: jnp.concatenate([value, value]) for key, value in sample.items()
  }
  return sample


class TestIndividualModel(unittest.TestCase):

  def test_correct_shape_return(self):
    """Assert the model returns a Jax array of the correct shape."""
    data = generate_individual_test_data()
    model = models.IndividualRecommender(TEST_MODEL_CONFIG)

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, data)
    results = model.apply(params, data)

    self.assertEqual(results.shape, (1, SEQ_LEN, VOCAB_SIZE))

  def test_correct_shape_returned_batch(self):
    """Assert the model passed a batch returns an array of the correct shape."""
    data = generate_individual_test_data_batch()
    model = models.IndividualRecommender(TEST_MODEL_CONFIG)

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, data)
    results = model.apply(params, data)

    self.assertEqual(results.shape, (2, SEQ_LEN, VOCAB_SIZE))

  def test_correct_shape__grade_lvl_model_returned_batch(self):
    """Assert grade level model passed a batch returns an array of the correct shape."""
    data = generate_individual_test_data_batch()
    config = TEST_MODEL_CONFIG.replace(num_grade_levels=5)
    model = models.IndividualRecommender(config)

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, data)
    results = model.apply(params, data)

    self.assertEqual(results.shape, (2, SEQ_LEN, VOCAB_SIZE))

  def test_correct_shape_returned_with_grade_lvl(self):
    """Check model with grade level biases returns an array with correct shape."""

    data = generate_individual_test_data()
    config = TEST_MODEL_CONFIG.replace(num_grade_levels=5)
    model = models.IndividualRecommender(config)

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, data)
    results = model.apply(params, data)

    self.assertEqual(results.shape, (1, SEQ_LEN, VOCAB_SIZE))

  def test_causality(self):
    """Assert changing an input only changes outputs later in the sequence."""
    model = models.IndividualRecommender(TEST_MODEL_CONFIG)
    data = generate_individual_test_data()
    length = data[INPUT_FIELDS.TITLES].shape[1]
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, data)

    reference_results = model.apply(params, data)

    for i in range(1, length):
      # generate new data with the (i-1)th item changed.
      modified_data = generate_individual_test_data()
      modified_data[INPUT_FIELDS.TITLES] = (
          modified_data[INPUT_FIELDS.TITLES].at[:, i - 1].add(1)
      )
      results = model.apply(params, modified_data)

      # Assert outputs up to (i-1)th are unchanged. This is necessary if
      # causality is expected
      prefix_reference = reference_results[:, :i]
      prefix_new_results = results[:, :i]
      self.assertTrue(
          jnp.allclose(prefix_reference, prefix_new_results, atol=ATOL)
      )

      # Assert the output sequence from ith onwards has changed. This is not
      # stricly necessary for causality but if this tensor with dims
      # (1, (10 - i), 100) has not changed, that is suggestive of vanishing
      # gradients / zero gradients and should not occur in a freshly initialized
      # model.
      suffix_reference = reference_results[:, i:]
      suffix_new_results = results[:, i:]
      self.assertFalse(
          jnp.allclose(suffix_reference, suffix_new_results, atol=ATOL)
      )

  def test_input_segmentation(self):
    """Assert model deals with segment packed inputs correctly.

    Segment packed inputs are multiple separated data points concatenated to
    for a longer sequence. This should produce identical results to processing
    each data point individually.
    """
    second_student = 5
    # create data and modify it to reflect the packing of two students
    data = generate_individual_test_data()
    data.pop(INPUT_FIELDS.GRADE_LEVELS)
    data[INPUT_FIELDS.STUDENT_IDS] = (
        data[INPUT_FIELDS.STUDENT_IDS].at[0, second_student:].add(1)
    )
    data[INPUT_FIELDS.INPUT_POSITIONS] = (
        data[INPUT_FIELDS.INPUT_POSITIONS].at[0, second_student:].add(-5)
    )

    # compute the results of the input with two samples packed
    model = models.IndividualRecommender(TEST_MODEL_CONFIG)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, data)
    segment_packed_results = model.apply(params, data)

    # reshape the data so each sample is a separate row in the batch
    data = {key: value.reshape(2, 5) for key, value in data.items()}
    batched_results = model.apply(params, data)
    batched_results = batched_results.reshape(*segment_packed_results.shape)

    self.assertTrue(
        jnp.allclose(batched_results, segment_packed_results, atol=ATOL)
    )


def generate_study_test_data():
  titles = jnp.array([[1, 2, 3, 4, 5, SEP_TOKEN, 7, 8, 9, 10]], dtype=jnp.int32)

  timestamps = jnp.array([[0, 1, 2, 3, 4, 2, 3, 4, 5, 6]], dtype=jnp.int32)

  student_ids = jnp.array([[1, 1, 1, 1, 1, 99, 2, 2, 2, 2]], dtype=jnp.int32)

  return {
      INPUT_FIELDS.TITLES: titles,
      INPUT_FIELDS.STUDENT_IDS: student_ids,
      INPUT_FIELDS.TIMESTAMPS: timestamps,
  }


class TestStudyModel(unittest.TestCase):

  def test_correct_shape_returned(self):
    """Assert the model returns a Jax array of the correct shape."""
    data = generate_study_test_data()
    model = models.StudyRecommender(TEST_MODEL_CONFIG)

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, data)
    results = model.apply(params, data)

    self.assertEqual(results.shape, (1, SEQ_LEN, VOCAB_SIZE))

  def test_causality(self):
    """Assert changing an input only changes outputs which is is allowed to.

    Changing an entries within titles should only affect outputs for
    predictions with a later timestamp or predictions with the same timestamp
    that occur later within the same student. This unit verifies this by
    modifying entries in input and comparing the new output to the original
    reference output.
    """
    # compute reference results on the original data.
    data = generate_study_test_data()
    model = models.StudyRecommender(TEST_MODEL_CONFIG)

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, data)

    reference_results = model.apply(params, data)
    reference_titles = data[INPUT_FIELDS.TITLES]

    # The i'th column in the j'th row is zero if the j'th output is expected to
    # change with the i'th input. -1 denotes either the i'th or j'th input is
    # a separator token.
    causality_map = jnp.array(
        [
            # pylint: disable-next=bad-continuation
            # [0, 1, 2, 3, 4, *,  3, 4, 5, 6]
            [0, 0, 0, 0, 0, -1, 0, 0, 0, 0],  # 0
            [1, 0, 0, 0, 0, -1, 0, 0, 0, 0],  # 1
            [1, 1, 0, 0, 0, -1, 0, 0, 0, 0],  # 2
            [1, 1, 1, 0, 0, -1, 0, 0, 0, 0],  # 3
            [1, 1, 1, 1, 0, -1, 1, 0, 0, 0],  # 4
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # *
            [1, 1, 1, 0, 0, -1, 0, 0, 0, 0],  # 3
            [1, 1, 1, 1, 0, -1, 1, 0, 0, 0],  # 4
            [1, 1, 1, 1, 1, -1, 1, 1, 0, 0],  # 5
            [1, 1, 1, 1, 1, -1, 1, 1, 1, 0],  # 6
        ],
        dtype=jnp.int32,
    )

    length = reference_titles.shape[1]
    for i in range(length):
      # compute results on output with the i'th entry changed
      data[INPUT_FIELDS.TITLES] = reference_titles.at[0, i].add(1)
      results = model.apply(params, data)
      # changing last output changes the (i+1)th output, hence i < length - 1
      if i < length - 1:
        # Assert changing the input changes the results. This is not necessarily
        # implied by the causal structure, but if it does not happen it is
        # suggestive of vanishing-gradients or other problems.
        self.assertFalse(jnp.allclose(results, reference_results, atol=ATOL))

      for j in range(length):
        causality_key = causality_map[j, i]
        # i or j is separator token, skip check.
        if causality_key == -1:
          continue

        if causality_key == 0:
          self.assertTrue(
              jnp.allclose(reference_results[0, j], results[0, j], atol=ATOL),
              f'Changing {i}th input should not affect {j}th output',
          )


if __name__ == '__main__':
  unittest.main()
