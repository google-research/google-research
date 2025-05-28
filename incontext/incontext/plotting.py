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
"""Plotting functions for the transformer model."""
import functools
import os
import pickle
from typing import Callable, Optional

import flax.linen as nn
from incontext import algos
from incontext import predictor_flax
from incontext import sampler_lib
from incontext import utils
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.io import gfile

Array = utils.Array


def plot_3d_line(ax, w, color="gray"):
  interval = np.arange(-3, 3, 0.01)
  x1, x2 = np.meshgrid(interval, interval)
  z = x1 * w[0] + x2 * w[1]
  ax.plot_surface(x1, x2, z, color=color, alpha=0.4)


def get_fix_mins_maxs(mins, maxs):
  deltas = (maxs - mins) / 12.0
  mins = mins + deltas / 4.0
  maxs = maxs - deltas / 4.0
  return [mins, maxs]


def plot_planes_3d(ax, w_empirical, w_gold):
  """Plot planes of two differen ws."""
  ax.set_xlim(get_fix_mins_maxs(-3.0, 3.0))
  ax.set_ylim(get_fix_mins_maxs(-3.0, 3.0))
  ax.set_zlim(get_fix_mins_maxs(-3.0, 3.0))
  ax.view_init(20, -120)
  ax.set_xlabel("x[0]")
  ax.set_ylabel("x[1]")
  ax.set_zlabel("y")
  plot_3d_line(ax, w_empirical.flatten(), color="red")
  plot_3d_line(ax, w_gold.flatten(), color="blue")


def plot_empirical_distribution(
    model,
    params,
    seq,
    sampler,
    coefficients = None,
    xs = None,
    ys = None,
    path = "test.jpeg",
    algo_fn = algos.LeastSquareAlgorithm,
    plot = True,
    plot_planes = True,
    plot_dots = True,
    plot_errors = True,
    plot_least_square_errors = True,
    plot_fake_least_square_errors = True,
    plot_knn_errors = False,
    return_w = False,
    save_predictions = False,
):
  """Calculates the empirical weights for the model's predictive function.

  Args:
    model (predictor_flax.CausalLM): causal transformer trained for icl.
    params (Any): model params.
    seq (torch.Tensor): input tensor with shape (batch_size, seq_length,
      hidden_size)
    sampler (sampler_lib.Sampler): sampler object to sample x vectors from
    coefficients (Optional[torch.Tensor], optional): Gold weights. Defaults to
      None.
    xs (Optional[Array]): x sequence.
    ys (Optional[Array]): y sequence.
    path (str): save prefix.
    algo_fn (Callable): main algorithm to compare the transformer.
    plot (bool): enable plotting
    plot_planes (bool): For 2D problem enable empirical plane plots
    plot_dots (bool):
    plot_errors (bool): Plot loss curve.
    plot_least_square_errors (bool):
    plot_fake_least_square_errors (bool):
    plot_knn_errors (bool):
    return_w (bool): whether to return w instead of plotting
    save_predictions (bool): whether to save the predictions

  Returns:
    torch.Tensor: empirical weights with shape (x_dim).
  """
  x, x_vec = sampler.sample_x(seq.shape[0])
  y, y_vec = sampler.calculate_y(x, coefficients)
  x, x_vec = jnp.array(x), jnp.array(x_vec)
  y_vec = jnp.array(y_vec)

  seq = jnp.concatenate([seq, x_vec[:, None, :], y_vec[:, None, :]], axis=1)
  errors, (y_errors, y_pred, seq_pred, _) = model.apply({"params": params},
                                                        inputs=seq,
                                                        train=False)
  y_pred = seq_pred[:, -1, :1]
  algo = algo_fn(fit_intercept=True)
  algo.fit(x, y_pred)
  w_empirical = algo.get_parameters()["W"]
  scores = algo.scores(y, y_pred)
  r2_empirical = float(scores["R2"])

  if return_w:
    return w_empirical, r2_empirical, x, y

  predictions_dict = {}
  y_pred_all = predictor_flax.extract_y(seq_pred)
  y_gold_all = predictor_flax.extract_y(seq, offset=1)
  predictions_dict["Transformer"] = np.array(y_pred_all)
  predictions_dict["Gold"] = np.array(y_gold_all)

  if plot:
    if plot_planes:
      ax = plt.axes(projection="3d")
      plot_planes_3d(ax, w_empirical, coefficients[0])
      ax.set_title("Empirical plane (red), gold plane (blue), R2={:.2f}".format(
          r2_empirical))

    if plot_dots:
      ax.scatter3D(
          np.array(x[:, 0]),
          np.array(x[:, 1]),
          np.array(y_pred),
          color="red",
      )
      if xs is not None and ys is not None:
        ax.scatter3D(
            xs[0, :, 0],
            xs[0, :, 1],
            ys[0, :, 0],
            color="green",
        )

    if plot_planes or plot_dots:
      with gfile.GFile(path, "wb") as handle:
        plt.savefig(handle, dpi=300)
      plt.close()

    if plot_errors:
      ax = plt.axes()
      errors = y_errors.mean(axis=0)
      ax.plot(np.arange(len(errors)), errors, color="red", label="Transformer")
      plt.xticks(np.arange(len(errors)))

    algo_fns = {
        "Lstsq":
            algos.LeastSquareAlgorithm,
        "Lstsq-Constant-Sigma":
            functools.partial(
                algos.FakeLeastSquareAlgorithm,
                precision=sampler.get_precision()),
        "Ridge(0.1)":
            functools.partial(algos.RidgeRegressionAlgorithm, alpha=0.1),
        "Ridge(0.5)":
            functools.partial(algos.RidgeRegressionAlgorithm, alpha=0.5),
        "Ridge(1.0)":
            functools.partial(algos.RidgeRegressionAlgorithm, alpha=1.0),
        "KNN(5, distance)":
            functools.partial(algos.KNNAlgorithm, k=5, weighting="distance"),
        "SGD(0.02, w=1)":
            functools.partial(
                algos.SGD,
                sampler.dim,
                nn.initializers.zeros,
                learning_rate_fn=lambda i: 0.02,
                window=1,
            ),
        "SGD(0.02, w=full)":
            functools.partial(
                algos.SGD,
                sampler.dim,
                nn.initializers.zeros,
                learning_rate_fn=lambda i: 0.02,
                window=-1,
            ),
        "SGD(0.03, w=full)":
            functools.partial(
                algos.SGD,
                sampler.dim,
                nn.initializers.zeros,
                learning_rate_fn=lambda i: 0.03,
                window=-1,
            ),
        "SGD(0.02, w=full, lambda=0.0001)":
            functools.partial(
                algos.SGD,
                sampler.dim,
                nn.initializers.zeros,
                learning_rate_fn=lambda i: 0.02,
                weight_decay=0.0001,
                window=-1,
            ),
        "SGD(0.03, w=full, lambda=0.0001)":
            functools.partial(
                algos.SGD,
                sampler.dim,
                nn.initializers.zeros,
                learning_rate_fn=lambda i: 0.03,
                weight_decay=0.0001,
                window=-1,
            ),
    }

    def plot_an_algorithm(algo_fn, label):
      predictions, ws, errors = algos.online_regression_with_batch(
          algo_fn=algo_fn, xs=xs, ys=ys)
      mse_values = errors["MSE"]
      ax.plot(np.arange(1, len(mse_values) + 1), mse_values, label=label)
      return predictions, ws, errors

    if plot_least_square_errors:
      predictions_dict["Lstsq"], _, _ = plot_an_algorithm(
          algo_fns["Lstsq"], "Least2")

    if plot_fake_least_square_errors:
      predictions_dict["Lstsq-Constant-Sigma"], _, _ = plot_an_algorithm(
          algo_fns["Lstsq-Constant-Sigma"], "Least2-Constant-Sigma")

    if plot_knn_errors:
      # plot_an_algorithm(algo_fns["KNN(5, distance)"], "KNN(5, distance)")
      predictions_dict["Ridge(0.1)"], _, _ = plot_an_algorithm(
          algo_fns["Ridge(0.1)"], "Ridge(0.1)")
      predictions_dict["Ridge(0.5)"], _, _ = plot_an_algorithm(
          algo_fns["Ridge(0.5)"], "Ridge(0.5)")
      predictions_dict["SGD(0.02, w=1)"], _, _ = plot_an_algorithm(
          algo_fns["SGD(0.02, w=1)"], "SGD(0.02, w=1)")
      predictions_dict["SGD(0.02, w=full)"], _, _ = plot_an_algorithm(
          algo_fns["SGD(0.02, w=full)"], "SGD(0.02, w=full)")
      predictions_dict["SGD(0.03, w=full)"], _, _ = plot_an_algorithm(
          algo_fns["SGD(0.03, w=full)"], "SGD(0.03, w=full)")
      predictions_dict[
          "SGD(0.02, w=full, lambda=0.0001)"], _, _ = plot_an_algorithm(
              algo_fns["SGD(0.02, w=full, lambda=0.0001)"],
              "SGD(0.02, w=full, lambda=0.0001)")
      predictions_dict[
          "SGD(0.03, w=full, lambda=0.0001)"], _, _ = plot_an_algorithm(
              algo_fns["SGD(0.03, w=full, lambda=0.0001)"],
              "SGD(0.03, w=full, lambda=0.0001)")

    if plot_errors or plot_least_square_errors or plot_fake_least_square_errors:
      ax.legend()
      prefix = path.replace(".jpeg", "")
      fpath = prefix + "_errors.jpeg"
      plt.title("Mean Prediction Loss after #Exemplars")
      ax.set_xlabel("#Exemplars")
      ax.set_ylabel("MSE per unit")
      with gfile.GFile(fpath, "wb") as handle:
        plt.savefig(handle, dpi=300)
      plt.close()

      # prediction divergence plots:
      ax = plt.axes()
      plt.xticks(np.arange(len(errors)))
      fpath = prefix + "_divergence.jpeg"
      for algo_name, pred_algo in predictions_dict.items():
        if algo_name != "Transformer" and algo_name != "Gold":
          algo_divergence = ((pred_algo -
                              y_pred_all[:, 1:-1, :1])**2).mean(axis=(0, 2))
          ax.plot(
              np.arange(1,
                        len(algo_divergence) + 1),
              algo_divergence,
              label=algo_name + "-Transformer",
          )
        if algo_name == "Lstsq":
          algo_divergence = ((pred_algo -
                              y_gold_all[:, 1:-1, :1])**2).mean(axis=(0, 2))
          ax.plot(
              np.arange(1,
                        len(algo_divergence) + 1),
              algo_divergence,
              label="Lstsq-Gold")

      plt.title("Prediction Divergence after #Exemplars")
      ax.set_xlabel("#Exemplars")
      ax.set_ylabel("MSE per unit")
      ax.legend()
      with gfile.GFile(fpath, "wb") as handle:
        plt.savefig(handle, dpi=300)
      plt.close()

  if save_predictions:
    fpath = path.replace(".jpeg", "_predictions.pkl")
    with gfile.GFile(fpath, "wb") as handle:
      pickle.dump(predictions_dict, handle)

  return x, y, scores, predictions_dict


def plot_implicit_w(
    model,
    params,
    seqs,
    xs,
    ys,
    sampler,
    coefficients,
    path,
):
  """Plot path of behavirol w."""
  implicit_ws = []
  implicit_r2s = []
  for i in range(seqs.shape[1] // 2 - 1):
    w_i, r2_i, _, _ = plot_empirical_distribution(
        model,
        params,
        seqs[:, :2 * i, :],
        sampler,
        coefficients,
        plot=False,
        return_w=True)
    implicit_ws.append(w_i)
    implicit_r2s.append(r2_i)
  implicit_ws = np.array(implicit_ws).squeeze(axis=1)
  implicit_r2s = np.array(implicit_r2s)

  ax = plt.axes()
  ax.scatter(implicit_ws[:, 0], implicit_ws[:, 1])
  for i, step in enumerate(range(implicit_ws.shape[0])):
    ax.annotate(step, (implicit_ws[i, 0], implicit_ws[i, 1]))

  ax.scatter([coefficients[0, 0]], [coefficients[0, 1]], color="red")
  plt.title("Path of implicit weights")
  ax.set_xlabel("W[0]")
  ax.set_ylabel("W[1]")
  with gfile.GFile(path.replace("name", "w"), "wb") as handle:
    plt.savefig(handle, dpi=300)
  plt.close()

  ax = plt.axes()
  ax.plot(np.arange(len(implicit_r2s)), implicit_r2s, label="Transformer R2s")
  with gfile.GFile(path.replace("name", "r2"), "wb") as handle:
    plt.savefig(handle, dpi=300)
  plt.close()

  _, parameters, _ = algos.online_regression_with_batch(
      algos.LeastSquareAlgorithm, xs, ys)
  ws = parameters["W"][0, :, 0, :]
  w_diff_loss = ((ws - implicit_ws)**2).mean(axis=-1)
  ax = plt.axes()
  ax.plot(np.arange(1, len(w_diff_loss) + 1), w_diff_loss)
  plt.title("||Wlsq2-Wimp||^2")
  ax.set_xlabel("#Exemplars")
  ax.set_ylabel("||Wlsq2-Wimp||^2")
  with gfile.GFile(path.replace("name", "wdiff"), "wb") as handle:
    plt.savefig(handle, dpi=300)
  plt.close()


def plot_basis_image(
    model,
    params,
    seqs,
    sampler,
    coefficients,
    path,
):
  """Plot path of behavirol w."""
  for t in range(sampler.dim):
    path_t = os.path.join(path, f"{t}/")
    gfile.makedirs(path_t)
    for i in range(sampler.dim):
      x, x_vec = sampler.sample_x(seqs.shape[0])
      mask = np.zeros((1, sampler.dim))
      mask[:, i] = 1.0
      x = x * mask
      x_vec[:, 1:] = x_vec[:, 1:] * mask
      y, y_vec = sampler.calculate_y(x, coefficients)
      x, x_vec = jnp.array(x), jnp.array(x_vec)
      y_vec = jnp.array(y_vec)
      seq_input = jnp.concatenate(
          [seqs[:, :2 * t, :], x_vec[:, None, :], y_vec[:, None, :]], axis=1)
      _, (_, y_pred, seq_pred, _) = model.apply({"params": params},
                                                inputs=seq_input,
                                                train=False)
      y_pred = np.array(seq_pred[:, -1, :1])
      ax = plt.axes()
      ax.scatter(
          np.array(x[:, i]),
          np.array(y[:, 0]),
          color="blue",
      )

      ax.scatter(
          np.array(x[:, i]),
          np.array(y_pred[:, 0]),
          color="red",
      )

      with gfile.GFile(os.path.join(path_t, f"dim_{i}.jpeg"), "wb") as handle:
        plt.savefig(handle, dpi=300)

      plt.close()
