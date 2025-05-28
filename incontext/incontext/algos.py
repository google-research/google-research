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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provides algorithms for linear regression problems."""
import abc
import functools
from typing import Any, Callable, List, Mapping, Optional, Tuple
import warnings

import flax.linen as nn
from flax.training import common_utils
from incontext import utils
import jax
import numpy as np
from sklearn import exceptions as sklearn_exceptions
from sklearn import linear_model
from sklearn import metrics
from sklearn import neighbors
from tensorflow.io import gfile

warnings.filterwarnings(
    "ignore", category=sklearn_exceptions.UndefinedMetricWarning)


class RegressionAlgorithm(metaclass=abc.ABCMeta):
  """A regression learning algorithm that learns predictor for the given regression dataset."""

  @abc.abstractmethod
  def fit(self, x, y):
    """Fits regression to a data given inputs x and outputs y.

    Args:
      x (utils.Array): Inputs w. shape (n_samples, input_features).
      y (utils.Array): Outputs w. shape (n_samples, output_features).
    """

  @abc.abstractmethod
  def predict(self, x):
    """Produce predictions for new inputs .

    Args:
      x (utils.Array): Inputs w. shape (n_samples, input_features).

    Returns:
      utils.Array: Predictions w. shape (n_samples, output_features).
    """

  def get_parameters(self):
    """Returns the parameters of the algorithm.

    Returns:
      utils.Array: model parameters.
    """
    return None

  def iterate(self, x, y):
    """Iterate the algorithm over the data given inputs x and outputs y."""
    raise NotImplementedError("This algorithm does not support iterate (yet)")

  def is_iterative(self):
    """Returns whether the algorithm is iterative.

    Returns:
      bool: True if the algorithm is iterative.
    """
    return False

  @abc.abstractmethod
  def reset(self):
    """Resets the algorithm."""

  def scores(self,
             y,
             y_hat = None):
    """Gets fit statistics such as R2 and MSE.

    Args:
      y (utils.Array): gold outputs.
      y_hat (utils.Array): predictions.

    Returns:
      Mapping[str, float]: fit statistics.
    """
    if y.ndim == 3:
      r2_scores = [
          metrics.r2_score(y[:, i, :], y_hat[:, i, :])
          for i in range(y.shape[1])
      ]
      r2_value = np.array(r2_scores)
      mse_value = ((y_hat - y)**2).mean(axis=(0, -1))
    else:
      r2_value = metrics.r2_score(y, y_hat)
      mse_value = ((y_hat - y)**2).mean(axis=-1)

    return {"R2": r2_value, "MSE": mse_value}


class LeastSquareAlgorithm(RegressionAlgorithm):
  """Least square regression algorithm."""

  def __init__(self, seed = 0, fit_intercept = False):
    self.seed = seed
    self.regressor = None
    self.fit_intercept = fit_intercept
    self.is_fitted = False

  def reset(self):
    self.regressor = None
    self.is_fitted = False

  def fit(self, x, y):
    assert x.ndim == 2 and y.ndim == 2
    self.regressor = linear_model.LinearRegression(
        fit_intercept=self.fit_intercept)
    self.regressor.fit(x, y)
    self.is_fitted = True

  def predict(self, x):
    assert x.ndim == 2
    return self.regressor.predict(x)

  def get_parameters(self):
    return {"W": self.regressor.coef_, "b": self.regressor.intercept_}


class FakeLeastSquareAlgorithm(RegressionAlgorithm):
  """Fake least square regression algorithm."""

  def __init__(self,
               precision,
               seed = 0,
               fit_intercept = False):
    self.seed = seed
    self.regressor = None
    assert not fit_intercept
    self.fit_intercept = fit_intercept
    self.is_fitted = False
    self.precision = precision

  def reset(self):
    self.regressor = None
    self.is_fitted = False

  def fit(self, x, y):
    assert x.ndim == 2 and y.ndim == 2
    weight = (self.precision @ x.T @ y) / x.shape[0]
    self.regressor = weight
    self.is_fitted = True

  def predict(self, x):
    assert x.ndim == 2
    return x @ self.regressor

  def get_parameters(self):
    return {"W": self.regressor}


class RidgeRegressionAlgorithm(RegressionAlgorithm):
  """Ridge regression algorithm with regularized least square."""

  def __init__(self,
               alpha = 0.01,
               fit_intercept = False,
               seed = 0):
    self.alpha = alpha
    self.seed = seed
    self.regressor = None
    self.fit_intercept = fit_intercept
    self.is_fitted = False

  def reset(self):
    self.regressor = None
    self.is_fitted = False

  def fit(self, x, y):
    assert x.ndim == 2 and y.ndim == 2
    self.regressor = linear_model.Ridge(
        alpha=self.alpha,
        fit_intercept=self.fit_intercept,
        random_state=self.seed)
    self.regressor.fit(x, y)
    self.is_fitted = True

  def predict(self, x):
    assert x.ndim == 2
    return self.regressor.predict(x)

  def get_parameters(self):
    return {"W": self.regressor.coef_, "b": self.regressor.intercept_}


class KNNAlgorithm(RegressionAlgorithm):
  """KNN regression algorithm."""

  def __init__(self,
               k = 5,
               weighting = "uniform",
               seed = 0):
    self.regressor = None
    self.k = k
    self.seed = seed
    self.weighting = weighting
    self.is_fitted = False

  def reset(self):
    self.regressor = None
    self.is_fitted = False

  def fit(self, x, y):
    assert x.ndim == 2 and y.ndim == 2
    k = min(self.k, x.shape[0])
    self.regressor = neighbors.KNeighborsRegressor(k, weights=self.weighting)
    self.regressor.fit(x, y)
    self.is_fitted = True

  def predict(self, x):
    assert x.ndim == 2
    return self.regressor.predict(x)


class SGD(RegressionAlgorithm):
  """Stochastic gradient descent variants."""

  def __init__(
      self,
      dim,
      init_fn,
      learning_rate_fn,
      weight_decay = 0.0,
      window = 1,
      seed = 0,
  ):
    self.learning_rate_fn = learning_rate_fn
    self.window = window
    self.init_fn = init_fn
    self.x_dim = dim
    self.seed = seed
    self.key = jax.random.PRNGKey(self.seed)
    self.weight = None
    self.is_fitted = False
    self.weight_decay = weight_decay

  def init_weight(self):
    self.weight = self.init_fn(self.key, (self.x_dim, 1))

  def reset(self):
    self.init_weight()
    self.is_fitted = False

  def iterate(self, x, y):
    assert x.ndim == 2 and y.ndim == 2

    if self.weight is None:
      self.init_weight()

    for i in range(x.shape[0]):
      if self.window == -1:
        start = 0
      else:
        start = min(0, i - self.window)

      x_batch = x[start:i + 1]
      grad = -2 * x_batch.T @ (y[start:i + 1] - self.predict(x_batch))
      if self.weight_decay > 0:
        grad += 2 * self.weight_decay * self.weight

      grad = jax.lax.clamp(-20.0, grad, 20.0)
      self.weight -= self.learning_rate_fn(i) * grad  # / x_batch.shape[0]

  def fit(self, x, y):
    assert x.ndim == 2 and y.ndim == 2
    self.init_weight()
    self.iterate(x, y)
    self.is_fitted = True

  def predict(self, x):
    assert x.ndim == 2
    return x @ self.weight

  def is_iterative(self):
    return self.window == 1


def online_regression(
    algo_fn,
    x,
    y,
):
  """Runs online regression for linear algorithms."""

  assert x.ndim == 2 and y.ndim == 2
  predictions, parameters = [], []
  algo = algo_fn()
  for i in range(1, x.shape[0]):
    if algo.is_iterative():
      algo.iterate(x[i - 1:i, :], y[i - 1:i, :])
    else:
      algo = algo_fn()
      algo.fit(x[:i, :], y[:i, :])
    x_i = x[i:i + 1, :]
    y_hat = algo.predict(x_i).squeeze(axis=0)
    predictions.append(y_hat)
    parameters.append(algo.get_parameters())
  return tuple(map(common_utils.stack_forest, (predictions, parameters)))


def online_regression_with_batch(
    algo_fn,
    xs,
    ys,
):
  """Runs online regression for linear algorithms for a batch."""
  batched_predictions, batched_parameters = [], []
  for i in range(xs.shape[0]):
    x_batch = xs[i, Ellipsis]
    y_batch = ys[i, Ellipsis]
    predictions, parameters = online_regression(algo_fn, x_batch, y_batch)
    batched_predictions.append(predictions)
    batched_parameters.append(parameters)

  batched_predictions, batched_parameters = tuple(
      map(common_utils.stack_forest, (batched_predictions, batched_parameters)))
  batched_errors = algo_fn().scores(ys[:, 1:, :], batched_predictions)
  return batched_predictions, batched_parameters, batched_errors


if __name__ == "__main__":
  # pylint: disable=g-import-not-at-top
  import matplotlib.pyplot as plt
  from incontext import sampler_lib
  # pylint: enable=g-import-not-at-top

  plt.style.use(".mplstyle")

  x_dim = 10
  hidden_size = 128
  num_exemplars = 64
  batch_size = 10

  def plot_errors(
      *,
      axis,
      algo_fn,
      x_distribution_fn,
      w_distribution_fn,
      label="algorithm",
      # path="algos_normal.jpeg",
  ):
    """Plot algorithm errors."""

    sampler = sampler_lib.Sampler(
        num_exemplars,
        x_dim,
        hidden_size,
        x_distribution_fn=x_distribution_fn,
        w_distribution_fn=w_distribution_fn,
    )

    data = sampler.sample(n=batch_size)
    _, _, xs, ys = data  # [np.repeat(d, batch_size, axis=0) for d in data]
    _, _, errors = online_regression_with_batch(algo_fn, xs, ys)
    mse_values = errors["MSE"]
    axis.plot(np.arange(1, len(mse_values) + 1), mse_values, label=label)

  algo_fns = {
      "Lstsq":
          LeastSquareAlgorithm,
      "Lstsq-Constant-Sigma":
          functools.partial(FakeLeastSquareAlgorithm, precision=np.eye(x_dim)),
      "Ridge(0.1)":
          functools.partial(RidgeRegressionAlgorithm, alpha=0.1),
      "Ridge(0.5)":
          functools.partial(RidgeRegressionAlgorithm, alpha=0.5),
      "Ridge(1.0)":
          functools.partial(RidgeRegressionAlgorithm, alpha=1.0),
      "KNN(5, distance)":
          functools.partial(KNNAlgorithm, k=5, weighting="distance"),
      "KNN(5, uniform)":
          functools.partial(KNNAlgorithm, k=5, weighting="uniform"),
      "SGD(0.01, w=1)":
          functools.partial(
              SGD,
              x_dim,
              nn.initializers.zeros,
              learning_rate_fn=lambda i: 0.01,
              window=1),
      "SGD(0.02, w=1)":
          functools.partial(
              SGD,
              x_dim,
              nn.initializers.zeros,
              learning_rate_fn=lambda i: 0.02,
              window=1),
      "SGD(0.01, w=full)":
          functools.partial(
              SGD,
              x_dim,
              nn.initializers.zeros,
              learning_rate_fn=lambda i: 0.01,
              window=-1),
      "SGD(0.02, w=full)":
          functools.partial(
              SGD,
              x_dim,
              nn.initializers.zeros,
              learning_rate_fn=lambda i: 0.02,
              window=-1),
      "SGD(0.02, w=full, lambda=0.001)":
          functools.partial(
              SGD,
              x_dim,
              nn.initializers.zeros,
              learning_rate_fn=lambda i: 0.02,
              window=-1,
              weight_decay=0.001,
          ),
      "SGD(0.02, w=full, lambda=0.0001)":
          functools.partial(
              SGD,
              x_dim,
              nn.initializers.zeros,
              learning_rate_fn=lambda i: 0.02,
              window=-1,
              weight_decay=0.0001,
          ),
      "SGD(0.02, w=full, lambda=0.01)":
          functools.partial(
              SGD,
              x_dim,
              nn.initializers.zeros,
              learning_rate_fn=lambda i: 0.02,
              window=-1,
              weight_decay=0.01,
          ),
      "SGD(0.001, w=full)":
          functools.partial(
              SGD,
              x_dim,
              nn.initializers.zeros,
              learning_rate_fn=lambda i: 0.001,
              window=-1),
  }

  distributions = ("normal*1+0", "normal*2.5+0", "normal*1+2.5")
  for distribution_str in distributions:
    ax = plt.axes()
    for name, algo_f in algo_fns.items():
      plot_errors(
          axis=ax,
          algo_fn=algo_f,
          x_distribution_fn=sampler_lib.str_to_distribution_fn("normal*1+0"),
          w_distribution_fn=sampler_lib.str_to_distribution_fn(
              distribution_str),
          label=name,
      )
    ax.legend()
    with gfile.GFile(f"algos_w_{distribution_str}.jpeg", "wb") as f:
      plt.savefig(f, dpi=300)
    plt.close()

  for distribution_str in distributions:
    ax = plt.axes()
    for name, algo_f in algo_fns.items():
      plot_errors(
          axis=ax,
          algo_fn=algo_f,
          x_distribution_fn=sampler_lib.str_to_distribution_fn(
              distribution_str),
          w_distribution_fn=sampler_lib.str_to_distribution_fn("normal*1+0"),
          label=name,
          # path="algos_normal.jpeg",
      )
    ax.legend()
    with gfile.GFile(f"algos_x_{distribution_str}.jpeg", "wb") as f:
      plt.savefig(f, dpi=300)
    plt.close()
