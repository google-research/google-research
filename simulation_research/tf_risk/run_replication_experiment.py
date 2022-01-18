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

"""Run an experiment of the synthesis of a basket call with mc methods.

Consider we are selling an European basket call option which in general cannot
be priced or hedged analytically and cannot be replicated statically even
under a simple Black-Scholes model for the dynamics of the underlyings.

For simplicity, bid-ask spread and fees are not taken into account.

We simulate correlated stock price variations under the historical probability
and hedge at each time period with a delta computed by a Monte Carlo method
(path-wise differentiation) under the risk neutral probability.
Tensorflow's automated differentiation capabilities make such a Monte Carlo
simulation simple to differentiate to extract the delta and easy to accelerate
on GPU/TPU.

See http://www.cmap.polytechnique.fr/~touzi/Poly-MAP552.pdf p99 7.4.2.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import app
from absl import flags
from absl import logging

import numpy as np
import scipy.linalg
import tensorflow.compat.v1 as tf

from simulation_research.tf_risk import controllers
from simulation_research.tf_risk import monte_carlo_manager
from simulation_research.tf_risk.dynamics import gbm_log_euler_step_nd
from simulation_research.tf_risk.payoffs import basket_call_payoff


flags.DEFINE_integer("num_dims", 100,
                     "Number of assets in basket option.")
flags.DEFINE_integer("num_batch_samples", 1000,
                     "Number of Monte Carlo per batch.")
flags.DEFINE_integer("num_batches", 10,
                     "Number of Monte Carlo batches.")

flags.DEFINE_float("initial_price", 100.0, "Initial price.")
flags.DEFINE_float("drift", 0.0, "Stock drift.")
flags.DEFINE_float("volatility", 0.2,
                   "Volatilty of each stock (assumed equal across stock).")
flags.DEFINE_float("correlation", 0.5,
                   "Pairwise correlation between stocks "
                   "(assumed equal across pairs).")

flags.DEFINE_float("strike", 100.0, "Basket strike.")
flags.DEFINE_float("maturity", 1.0, "European option maturity (in years).")

flags.DEFINE_float("delta_t_historical", 0.1,
                   "Time step of historical stock price simulation.")
flags.DEFINE_float("delta_t_monte_carlo", 0.1,
                   "Time step of monte carlo risk neutral price simulation.")

FLAGS = flags.FLAGS


def hist_log_euler_step(hist_state, hist_drift, hist_vol_matrix, dt):
  """Simulation of price movements under the historical probability."""
  num_dims = hist_state.shape[0]
  hist_dw_t = np.random.normal(size=[num_dims]) * np.sqrt(dt)
  return np.maximum(hist_state * (
      1.0 + hist_drift * dt + np.matmul(hist_dw_t, hist_vol_matrix)), 1e-7)


def main(_):
  num_dims = FLAGS.num_dims
  num_batches = FLAGS.num_batches
  hist_drift = FLAGS.drift
  hist_vol = FLAGS.volatility
  hist_cor = FLAGS.correlation
  hist_cor_matrix = (hist_cor * np.ones((num_dims, num_dims))
                     + (1.0 - hist_cor) * np.eye(num_dims))

  hist_price = FLAGS.initial_price * np.ones(num_dims)
  hist_vol_matrix = hist_vol * np.real(scipy.linalg.sqrtm(hist_cor_matrix))

  hist_dt = FLAGS.delta_t_historical
  sim_dt = FLAGS.delta_t_monte_carlo

  strike = FLAGS.strike
  maturity = FLAGS.maturity

  # Placeholders for tensorflow-based simulator's arguments.
  sim_price = tf.placeholder(shape=[num_dims], dtype=tf.float32)
  sim_drift = tf.placeholder(shape=(), dtype=tf.float32)
  sim_vol_matrix = tf.constant(hist_vol_matrix, dtype=tf.float32)
  sim_maturity = tf.placeholder(shape=(), dtype=tf.float32)

  # Transition operation between t and t + dt with price in log scale.
  def _dynamics_op(log_s, t, dt):
    return gbm_log_euler_step_nd(
        log_s, sim_drift, sim_vol_matrix, t, dt)

  # Terminal payoff function (with price in log scale).
  def _payoff_fn(log_s):
    return basket_call_payoff(tf.exp(log_s), strike)

  # Call's price and delta estimates (sensitivity to current underlying price).
  # Monte Carlo estimation under the risk neutral probability is used.
  # The reason why we employ the risk neutral probability is that the position
  # is hedged each day depending on the value of the underlying.
  # See http://www.cmap.polytechnique.fr/~touzi/Poly-MAP552.pdf for a complete
  # explanation.
  price_estimate, _, _ = monte_carlo_manager.non_callable_price_mc(
      initial_state=tf.log(sim_price),
      dynamics_op=_dynamics_op,
      payoff_fn=_payoff_fn,
      maturity=sim_maturity,
      num_samples=FLAGS.num_batch_samples,
      dt=sim_dt)
  delta_estimate = monte_carlo_manager.sensitivity_autodiff(
      price_estimate, sim_price)

  # Start the hedging experiment.
  session = tf.Session()

  hist_price_profile = []
  cash_profile = []
  underlying_profile = []
  wall_times = []

  t = 0
  cash_owned = 0.0
  underlying_owned = np.zeros(num_dims)
  while t <= maturity:
    # Each day, a new stock price is observed.

    cash_eval = 0.0
    delta_eval = 0.0
    for _ in range(num_batches):
      if t == 0.0:
        # The first day a derivative price is computed to decide how mush cash
        # is initially needed to replicate the derivative's payoff at maturity.
        cash_eval_batch = controllers.price_derivative(
            price_estimate,
            session,
            params={
                sim_drift: 0.0,
                sim_price: hist_price,
                sim_maturity: maturity - t
            })

      # Each day the delta of the derivative is computed to decide how many
      # shares of the underlying should be owned to replicate the derivative's
      # payoff at maturity.
      start_time = time.time()
      delta_eval_batch = controllers.hedge_derivative(
          delta_estimate,
          session,
          params={
              sim_drift: 0.0,
              sim_price: hist_price,
              sim_maturity: maturity - t
          })
      wall_times.append(time.time() - start_time)
      delta_eval += delta_eval_batch / num_batches
      cash_eval += cash_eval_batch / num_batches

    if t == 0.0:
      logging.info("Initial price estimate: %.2f", cash_eval)

    # Self-financing portfolio dynamics, held cash is used to buy the underlying
    # or increases when the underlying is sold.
    if t == 0.0:
      cash_owned = cash_eval - np.sum(delta_eval * hist_price)
      underlying_owned = delta_eval
    else:
      cash_owned -= np.sum((delta_eval - underlying_owned) * hist_price)
      underlying_owned = delta_eval
    logging.info("Cash at t=%.2f: %.2f", t, cash_owned)
    logging.info("Mean delta at t=%.2f: %.4f", t, np.mean(delta_eval))
    logging.info("Mean underlying at t=%.2f: %.2f ", t, np.mean(hist_price))

    hist_price_profile.append(hist_price)
    cash_profile.append(cash_owned)
    underlying_profile.append(underlying_owned)

    # Simulation of price movements under the historical probability (i.e. what
    # is actually happening in the stock market).
    hist_price = hist_log_euler_step(
        hist_price, hist_drift, hist_vol_matrix, hist_dt)

    t += hist_dt

  session.close()

  # At maturity, the value of the replicating portfolio should be exactly
  # the opposite of the payoff of the option being sold.
  # The reason why the match is not exact here is two-fold: we only hedge once
  # a day and we use noisy Monte Carlo estimates to do so.
  underlying_owned_value = np.sum(underlying_owned * hist_price)
  profit = np.sum(underlying_owned * hist_price) + cash_owned
  loss = (np.mean(hist_price) - strike) * (np.mean(hist_price) > strike)

  logging.info("Cash owned at maturity %.3f.", cash_owned)
  logging.info("Value of underlying owned at maturity %.3f.",
               underlying_owned_value)
  logging.info("Profit (value held) = %.3f.", profit)
  logging.info("Loss (payoff sold, 0 if price is below strike) = %.3f.", loss)
  logging.info("PnL (should be close to 0) = %.3f.", profit - loss)


if __name__ == "__main__":
  app.run(main)
