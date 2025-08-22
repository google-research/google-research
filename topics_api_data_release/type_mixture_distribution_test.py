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

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp

from topics_api_data_release import type_mixture_distribution


class TypeMixtureDistributionTest(absltest.TestCase):

  def test_randomly_initialize_has_right_shape(self):
    key = jax.random.PRNGKey(0)
    dist = type_mixture_distribution.TypeMixtureTopicDistribution.initialize_randomly(
        key, 10, 9, 5, 4
    )
    self.assertEqual(dist.theta.shape, (10, 9, 5, 4))

  def test_get_slot_prob_array_correct(self):
    theta = jnp.array([
        [  # Type 0:
            [  # Week 0:
                [0.0, 0.0, 0.0],  # Slot 0:
                [1.9459102, 0.6931472, 0.0],  # Slot 1:
            ],
            [  # Week 1:
                [100.0, 0.0, 0.0],  # Slot 0:
                [0.0, 100.0, 0.0],  # Slot 1:
            ],
        ],
        [  # Type 1:
            [  # Week 0:
                [0.0, 0.0, 100.0],  # Slot 0:
                [100.0, 0.0, 0.0],  # Slot 1:
            ],
            [  # Week 1:
                [100.0, 0.0, 0.0],  # Slot 0:
                [100.0, 0.0, 0.0],  # Slot 1:
            ],
        ],
    ])  # fmt: skip
    dist = type_mixture_distribution.TypeMixtureTopicDistribution(theta=theta)
    expected = jnp.array([
        [  # Type 0:
            [  # Week 0:
                [1/3, 1/3, 1/3],  # Slot 0:
                [0.7, 0.2, 0.1],  # Slot 1:
            ],
            [  # Week 1:
                [1.0, 0.0, 0.0],  # Slot 0:
                [0.0, 1.0, 0.0],  # Slot 1:
            ],
        ],
        [  # Type 1:
            [  # Week 0:
                [0.0, 0.0, 1.0],  # Slot 0:
                [1.0, 0.0, 0.0],  # Slot 1:
            ],
            [  # Week 1:
                [1.0, 0.0, 0.0],  # Slot 0:
                [1.0, 0.0, 0.0],  # Slot 1:
            ],
        ],
    ])  # fmt: skip
    chex.assert_trees_all_close(dist.get_slot_prob_array(), expected)

  def test_sample_topic_indices_correct_shape(self):
    theta = jnp.array([
        [  # Type 0:
            [  # Week 0:
                [100.0, 0.0, 0.0],  # Slot 0
                [0.0, 100.0, 0.0],  # Slot 1
                [0.0, 0.0, 100.0],  # Slot 2
            ],
            [  # Week 1:
                [100.0, 0.0, 0.0],  # Slot 0
                [0.0, 100.0, 0.0],  # Slot 1
                [0.0, 0.0, 100.0],  # Slot 2
            ],
        ],
        [  # Type 1:
            [  # Week 0:
                [100.0, 0.0, 0.0],  # Slot 0
                [0.0, 100.0, 0.0],  # Slot 1
                [0.0, 0.0, 100.0],  # Slot 2
            ],
            [  # Week 1:
                [100.0, 0.0, 0.0],  # Slot 0
                [0.0, 100.0, 0.0],  # Slot 1
                [0.0, 0.0, 100.0],  # Slot 2
            ],
        ],
    ])  # fmt: skip
    dist = type_mixture_distribution.TypeMixtureTopicDistribution(theta=theta)

    samples_100 = dist.sample_topic_indices(jax.random.PRNGKey(0), 100)
    self.assertEqual(samples_100.shape, (100, 2, 3))

    samples_25 = dist.sample_topic_indices(jax.random.PRNGKey(0), 25)
    self.assertEqual(samples_25.shape, (25, 2, 3))

  def test_sample_topic_indices_uses_uniform_types(self):
    """Tests that sample_topic_indices picks types uniformly at random.

    This test creates a topics distribution with one week and one slot so that
    type i deterministically samples topic i. Then we sample 10000 users and
    check that the fraction of each topic (equal to the fraction of each type)
    is close to uniform.
    """
    num_samples = 10000

    theta = jnp.array([
        [  # Type 0:
            [  # Week 0:
                [1000, 0.0, 0.0],
            ],
        ],
        [  # Type 1:
            [  # Week 0:
                [0.0, 1000, 0.0],
            ]
        ],
        [  # Type 2:
            [  # Week 0:
                [0.0, 0.0, 1000],
            ]
        ],
    ])  # fmt: skip
    dist = type_mixture_distribution.TypeMixtureTopicDistribution(theta=theta)
    samples = dist.sample_topic_indices(jax.random.PRNGKey(0), num_samples)

    chex.assert_trees_all_close(
        jnp.mean(jax.nn.one_hot(jnp.squeeze(samples), num_classes=3), axis=0),
        jnp.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        atol=0.05,
    )

  def test_sample_topic_indices_correct_for_single_type(self):
    """Tests that sample_topic_indices samples from a single type correctly.

    This test creates a topics distribution
    """
    num_samples = 10000
    theta = jnp.array([
        [  # Type 0:
            [  # Week 0:
                [0.0, 0.0, 0.0],  # Slot 0:
                [1.9459102, 0.6931472, 0.0],  # Slot 1:
            ],
            [  # Week 1:
                [100.0, 0.0, 0.0],  # Slot 0:
                [0.0, 100.0, 0.0],  # Slot 1:
            ],
        ],
    ])  # fmt: skip
    dist = type_mixture_distribution.TypeMixtureTopicDistribution(theta=theta)

    samples = dist.sample_topic_indices(jax.random.PRNGKey(0), num_samples)
    empirical_probs = jnp.mean(
        jax.nn.one_hot(jnp.squeeze(samples), num_classes=3), axis=0
    )

    chex.assert_trees_all_close(
        empirical_probs, dist.get_slot_prob_array()[0, Ellipsis], atol=0.05
    )

if __name__ == "__main__":
  absltest.main()
