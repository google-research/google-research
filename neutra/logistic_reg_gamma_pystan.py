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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import pystan
import simplejson
import pickle
import hashlib

FLAGS = flags.FLAGS

flags.DEFINE_string("german_path", "/tmp/german.data-numeric", "")


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


def SaveJSON(obj, path):
  with open(path, "w") as f:
    simplejson.dump(obj, f, cls=NumpyEncoder)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  data = np.genfromtxt(FLAGS.german_path)
  x = data[:, :-1]
  y = (data[:, -1] - 1).astype(np.int32)

  x_min = np.min(x, 0, keepdims=True)
  x_max = np.max(x, 0, keepdims=True)

  x /= (x_max - x_min)
  x = 2.0 * x - 1.0

  x = np.concatenate([x, np.ones([x.shape[0], 1])], -1)

  code = """
  data {
    int <lower=0> n; // number of observations
    int <lower=0> d; // number of predictors
    int <lower=0,upper=1> y[n]; // outputs
    matrix[n,d] x; // inputs
  }
  parameters {
    // auxiliary variables that define the global and local parameters
    vector[d] z;
    vector <lower=0>[d] local_scale;
    real <lower=0> global_scale;
  }
  transformed parameters {
    vector[d] beta; // regression coefficients
    vector[n] f; // latent values

    beta = z .* local_scale * global_scale;
    f = x * beta;
  }
  model {
    z ~ normal(0, 1);
    local_scale ~ gamma(0.5, 0.5);
    global_scale ~ gamma(0.5, 0.5);
    y ~ bernoulli_logit(f);
  }
  """

  data = {
      "n": x.shape[0],
      "d": x.shape[1],
      "x": x,
      "y": y,
  }

  filename = '/tmp/stan_model_%s' % hashlib.md5(code.encode('ascii')).hexdigest()
  print(filename)
  try:
    sm = pickle.load(open(filename, "rb"))
  except FileNotFoundError:
    sm = pystan.StanModel(model_code=code)
    with open(filename, 'wb') as f:
      pickle.dump(sm, f)
  fit = sm.sampling(data=data, iter=100000, chains=12)

  print(fit)

  params = fit.extract(["z", "local_scale", "global_scale"])

  params = np.concatenate([
      params["z"], params["local_scale"], params["global_scale"][Ellipsis, np.newaxis],
  ], -1)

  mean = params.mean(0)
  square = (params**2.).mean(0)

  print(params.shape)
  print(mean.shape)

  SaveJSON({"mean": mean, "square": square}, "/tmp/logistic_gamma_0")


if __name__ == "__main__":
  app.run(main)
