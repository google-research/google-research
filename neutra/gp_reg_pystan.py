# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import csv

import numpy as np
import pystan
import simplejson
import pickle
import hashlib

FLAGS = flags.FLAGS

flags.DEFINE_string("cloud_path", "/tmp/dataset_2196_cloud.csv", "")


class NumpyEncoder(simplejson.JSONEncoder):

  def default(self, obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
      return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    return simplejson.JSONEncoder.default(self, obj)


def LoadCloud(path):
  with open(path) as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]
    cols = list(zip(*rows))[1:]

    numeric_cols = []
    for col in cols:
      try:
        x = np.zeros([len(col), 1])
        for i, v in enumerate(col):
          x[i, 0] = float(v)

        x_min = np.min(x, 0, keepdims=True)
        x_max = np.max(x, 0, keepdims=True)

        x /= (x_max - x_min)
        x = 2.0 * x - 1.0

      except ValueError:
        keys = list(sorted(set(col)))
        vocab = {k: v for v, k in enumerate(keys)}
        x = np.zeros([len(col), len(keys)])
        for i, v in enumerate(col):
          one_hot = np.zeros(len(keys))
          one_hot[vocab[v]] = 1.
          x[i] = one_hot
      numeric_cols.append(x)

    data = np.concatenate(numeric_cols, -1)
    return data[:, :-1], data[:, -1]


def SaveJSON(obj, path):
  with open(path, "w") as f:
    simplejson.dump(obj, f, cls=NumpyEncoder)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  x, y = LoadCloud(FLAGS.cloud_path)

  code = """
  functions {
  matrix cov_exp_quad_ARD(vector[] x,
                          real alpha,
                          vector rho,
                          real delta) {
      int N = size(x);
      matrix[N, N] K;
      real neg_half = -0.5;
      real sq_alpha = square(alpha);
      for (i in 1:(N-1)) {
        K[i, i] = sq_alpha + delta;
        for (j in (i + 1):N) {
          real v = sq_alpha * exp(neg_half *
                                  dot_self((x[i] - x[j]) ./ rho));
          K[i, j] = v;
          K[j, i] = v;
        }
      }
      K[N, N] = sq_alpha + delta;
      return K;
    }
  }
  data {
    int<lower=1> N;
    int<lower=1> D;
    vector[D] x[N];
    vector[N] y;
  }
  transformed data {
    real delta = 1e-9;
  }
  parameters {
    vector<lower=0>[D] rho;
    real<lower=0> alpha;
    real<lower=0> sigma;
  }
  model {
    matrix[N, N] cov = cov_exp_quad_ARD(x, alpha, rho, delta)
      + diag_matrix(rep_vector(square(sigma), N));
    matrix[N, N] L_cov = cholesky_decompose(cov);

    rho ~ inv_gamma(5, 5);
    alpha ~ normal(0, 1);
    sigma ~ normal(0, 1);

    y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
  }
  """

  data = {
      "N": x.shape[0],
      "D": x.shape[1],
      "x": x,
      "y": y,
  }

  filename = "/tmp/stan_model_%s" % hashlib.md5(
      code.encode("ascii")).hexdigest()
  print(filename)
  try:
    sm = pickle.load(open(filename, "rb"))
  except FileNotFoundError:
    sm = pystan.StanModel(model_code=code)
    with open(filename, "wb") as f:
      pickle.dump(sm, f)
  fit = sm.sampling(data=data, iter=100000, chains=12)

  print(fit)

  params = fit.extract(["rho", "alpha", "sigma"])

  params = np.concatenate([
      params["rho"],
      params["alpha"][Ellipsis, np.newaxis],
      params["sigma"][Ellipsis, np.newaxis],
  ], -1)

  mean = params.mean(0)
  square = (params**2.).mean(0)

  print(params.shape)
  print(mean.shape)

  SaveJSON({"mean": mean, "square": square}, "/tmp/gp_reg_0")


if __name__ == "__main__":
  app.run(main)
