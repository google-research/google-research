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

# Lint as: python3
"""Tests for snlds.model_cavi_snlds."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from snlds import model_cavi_snlds
from snlds import utils
from snlds.examples.config_utils import ConfigDict


class ModelCaviSnldsTest(tf.test.TestCase):

  def setUp(self):
    super(ModelCaviSnldsTest, self).setUp()
    self.build_model_components()

  def build_model_components(self):
    self.batch_size = 11
    self.seq_len = 17
    self.obs_dim = 3
    self.hidden_dim = 8
    self.num_categ = 5
    self.config_emission = self.get_default_distribution_config()
    self.config_inference = self.get_default_distribution_config()
    self.config_z_initial = self.get_default_distribution_config()
    self.config_z_transition = self.get_default_distribution_config()

    self.network_z_transition = [
        utils.build_dense_network(
            [3*self.hidden_dim, self.hidden_dim], ["relu", None])
        for _ in range(self.num_categ)]
    self.z_trans_dist = model_cavi_snlds.ContinuousStateTransition(
        self.network_z_transition,
        distribution_dim=self.hidden_dim,
        num_categories=self.num_categ,
        **self.config_z_transition)

    num_categ_squared = self.num_categ * self.num_categ
    self.network_s_transition = utils.build_dense_network(
        [4 * num_categ_squared, num_categ_squared],
        ["relu", None])
    self.s_trans = model_cavi_snlds.DiscreteStateTransition(
        transition_network=self.network_s_transition,
        num_categories=self.num_categ)

    self.network_emission = utils.build_dense_network(
        [4 * self.obs_dim, self.obs_dim],
        ["relu", None])
    self.x_dist = model_cavi_snlds.GaussianDistributionFromMean(
        emission_mean_network=self.network_emission,
        observation_dim=self.obs_dim,
        name="GaussianDistributionFromMean",
        **self.config_emission)

    self.posterior_rnn = utils.build_rnn_cell(
        rnn_type="lstm", rnn_hidden_dim=32)
    self.network_posterior_mlp = utils.build_dense_network(
        [self.hidden_dim], [None])
    self.posterior_distribution = model_cavi_snlds.GaussianDistributionFromMean(
        emission_mean_network=self.network_posterior_mlp,
        observation_dim=self.hidden_dim,
        name="PosteriorDistribution",
        **self.config_inference)
    self.network_input_embedding = lambda x: x
    self.inference_network = model_cavi_snlds.RnnInferenceNetwork(
        posterior_rnn=self.posterior_rnn,
        posterior_dist=self.posterior_distribution,
        latent_dim=self.hidden_dim,
        embedding_network=self.network_input_embedding)

    self.init_z0_distribution = (
        model_cavi_snlds.construct_initial_state_distribution(
            self.hidden_dim,
            self.num_categ,
            use_triangular_cov=True))

  def get_default_distribution_config(self):
    config = ConfigDict()
    config.cov_mat = None
    config.use_triangular_cov = True
    config.use_trainable_cov = False
    config.raw_sigma_bias = 0.0
    config.sigma_min = 1e-5
    config.sigma_scale = 0.05
    return config

  def test_continuous_state_transition(self):
    test_inputs = tf.ones(
        [self.batch_size, self.seq_len, self.hidden_dim])
    z_trans_dist = self.z_trans_dist(test_inputs)

    self.assertEqual(z_trans_dist.batch_shape.as_list(),
                     [self.batch_size, self.seq_len, self.num_categ])
    self.assertEqual(z_trans_dist.event_shape.as_list(),
                     [self.hidden_dim])
    self.assertEqual(self.z_trans_dist.output_event_dims,
                     self.hidden_dim)

  def test_discrete_state_transition(self):
    test_inputs = tf.ones(
        [self.batch_size, self.seq_len, self.obs_dim])
    s_trans = self.s_trans(test_inputs)
    self.assertAllEqual(
        self.evaluate(tf.shape(s_trans)),
        [self.batch_size, self.seq_len, self.num_categ, self.num_categ])
    self.assertEqual(self.s_trans.output_event_dims,
                     self.num_categ)

  def test_emission_distribution(self):
    test_inputs = tf.ones(
        [self.batch_size, self.seq_len, self.hidden_dim])
    x_emit = self.x_dist(test_inputs)
    self.assertAllEqual(
        x_emit.batch_shape.as_list(),
        [self.batch_size, self.seq_len])
    self.assertAllEqual(
        x_emit.event_shape.as_list(),
        [self.obs_dim])
    self.assertEqual(
        self.x_dist.output_event_dims,
        self.obs_dim)

  def test_rnn_inference_network(self):
    test_inputs = tf.ones(
        [self.batch_size, self.seq_len, self.obs_dim])
    sampled_z, entropies, log_prob_q = self.inference_network(test_inputs)
    self.assertAllEqual(
        self.evaluate(tf.shape(sampled_z)),
        [1, self.batch_size, self.seq_len, self.hidden_dim])
    self.assertAllEqual(
        self.evaluate(tf.shape(entropies)),
        [1, self.batch_size, self.seq_len])
    self.assertAllEqual(
        self.evaluate(tf.shape(log_prob_q)),
        [1, self.batch_size, self.seq_len])

  def test_rnn_inference_network_multisamples(self):
    test_inputs = tf.ones([self.batch_size, self.seq_len, self.obs_dim])
    sampled_z, entropies, log_prob_q = self.inference_network(
        test_inputs, num_samples=10)
    self.assertAllEqual(
        self.evaluate(tf.shape(sampled_z)),
        [10, self.batch_size, self.seq_len, self.hidden_dim])
    self.assertAllEqual(
        self.evaluate(tf.shape(entropies)),
        [10, self.batch_size, self.seq_len])
    self.assertAllEqual(
        self.evaluate(tf.shape(log_prob_q)),
        [10, self.batch_size, self.seq_len])

  def test_create_initial_state_distribution(self):
    self.assertEqual(self.init_z0_distribution.batch_shape.as_list(),
                     [self.num_categ])
    self.assertEqual(self.init_z0_distribution.event_shape.as_list(),
                     [self.hidden_dim])

  def test_create_model(self):
    snlds_model = model_cavi_snlds.create_model(
        num_categ=self.num_categ,
        hidden_dim=self.hidden_dim,
        observation_dim=self.obs_dim,
        config_emission=self.config_emission,
        config_inference=self.config_inference,
        config_z_initial=self.config_z_initial,
        config_z_transition=self.config_z_transition,
        network_emission=self.network_emission,
        network_input_embedding=self.network_input_embedding,
        network_posterior_mlp=self.network_posterior_mlp,
        network_posterior_rnn=self.posterior_rnn,
        network_s_transition=self.network_s_transition,
        networks_z_transition=self.network_z_transition)

    input_tensor = tf.random.normal(
        shape=[self.batch_size, self.seq_len, self.obs_dim])

    output = snlds_model(input_tensor)
    output["elbo"].numpy()  # actually test the code runs.

  def test_create_model_multisample(self):
    snlds_model = model_cavi_snlds.create_model(
        num_categ=self.num_categ,
        hidden_dim=self.hidden_dim,
        observation_dim=self.obs_dim,
        config_emission=self.config_emission,
        config_inference=self.config_inference,
        config_z_initial=self.config_z_initial,
        config_z_transition=self.config_z_transition,
        network_emission=self.network_emission,
        network_input_embedding=self.network_input_embedding,
        network_posterior_mlp=self.network_posterior_mlp,
        network_posterior_rnn=self.posterior_rnn,
        network_s_transition=self.network_s_transition,
        networks_z_transition=self.network_z_transition)

    input_tensor = tf.random.normal(
        shape=[self.batch_size, self.seq_len, self.obs_dim])
    snlds_model(input_tensor, num_samples=10)

  def test_iwae_is_elbo_when_num_sample_is_one(self):
    tf.random.set_seed(12345)
    snlds_model = model_cavi_snlds.create_model(
        num_categ=self.num_categ,
        hidden_dim=self.hidden_dim,
        observation_dim=self.obs_dim,
        config_emission=self.config_emission,
        config_inference=self.config_inference,
        config_z_initial=self.config_z_initial,
        config_z_transition=self.config_z_transition,
        network_emission=self.network_emission,
        network_input_embedding=self.network_input_embedding,
        network_posterior_mlp=self.network_posterior_mlp,
        network_posterior_rnn=self.posterior_rnn,
        network_s_transition=self.network_s_transition,
        networks_z_transition=self.network_z_transition)

    input_tensor = tf.random.normal(
        shape=[self.batch_size, self.seq_len, self.obs_dim])
    snlds_model.build(input_shape=(self.batch_size, self.seq_len, self.obs_dim))
    result_dict = snlds_model(input_tensor, num_samples=1)
    self.assertAllClose(
        self.evaluate(result_dict["iwae"]), self.evaluate(result_dict["elbo"]))

  def test_iwae_bound_is_tighter(self):
    tf.random.set_seed(12345)
    snlds_model = model_cavi_snlds.create_model(
        num_categ=self.num_categ,
        hidden_dim=self.hidden_dim,
        observation_dim=self.obs_dim,
        config_emission=self.config_emission,
        config_inference=self.config_inference,
        config_z_initial=self.config_z_initial,
        config_z_transition=self.config_z_transition,
        network_emission=self.network_emission,
        network_input_embedding=self.network_input_embedding,
        network_posterior_mlp=self.network_posterior_mlp,
        network_posterior_rnn=self.posterior_rnn,
        network_s_transition=self.network_s_transition,
        networks_z_transition=self.network_z_transition)

    input_tensor = tf.random.normal(
        shape=[self.batch_size, self.seq_len, self.obs_dim])
    snlds_model.build(input_shape=(self.batch_size, self.seq_len, self.obs_dim))
    result_dict = snlds_model(input_tensor, num_samples=10)
    self.assertGreater(
        self.evaluate(result_dict["iwae"]), self.evaluate(result_dict["elbo"]))


if __name__ == "__main__":
  tf.test.main()
