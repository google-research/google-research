# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Model definitions."""
# pylint: disable=missing-docstring,g-doc-args,g-doc-return-or-yield
# pylint: disable=g-short-docstring-punctuation,g-no-space-after-docstring-summary
# pylint: disable=invalid-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import os
import numpy as np
import pandas as pd
from six.moves import urllib

import tensorflow.compat.v1 as tf

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from tensorflow_probability import edward2 as ed
from edward2_autoreparam.tfp import program_transformations as ed_transforms

data_dir = '/tmp/datasets'

ModelConfig = collections.namedtuple(
    'ModelConfig',
    ('model', 'model_args', 'observed_data',
     'to_centered', 'to_noncentered', 'make_to_centered'))


def build_make_to_centered(model, model_args):
  """Make a fn to convert a model's state to centered parameterisation."""

  def make_to_centered(**centering_kwargs):
    (_,
     parametrisation,
     _) = ed_transforms.make_learnable_parametrisation(
         learnable_parameters=centering_kwargs)

    def to_centered(uncentered_state):
      set_values = ed_transforms.make_value_setter(*uncentered_state)
      with ed.interception(set_values):
        with ed.interception(parametrisation):
          with ed.tape() as centered_tape:
            model(*model_args)
      return [tf.identity(v) for v in list(centered_tape.values())[:-1]]

    return to_centered
  return make_to_centered


def make_to_noncentered(model, model_args):
  """Make a fn to convert a model's state to noncentered parameterisation."""
  def to_noncentered(centered_state):
    set_values = ed_transforms.make_value_setter(*centered_state)
    with ed.tape() as noncentered_tape:
      with ed.interception(ed_transforms.ncp):
        with ed.interception(set_values):
          model(*model_args)
    return [tf.identity(v) for v in list(noncentered_tape.values())[:-1]]
  return to_noncentered


def get_eight_schools():
  """Eight schools model."""
  num_schools = 8
  treatment_effects = np.array([28, 8, -3, 7, -1, 1, 18, 12],
                               dtype=np.float32)
  treatment_stddevs = np.array([15, 10, 16, 11, 9, 11, 10, 18],
                               dtype=np.float32)

  def schools_model(num_schools, stddevs):
    mu = ed.Normal(loc=0., scale=5., name='mu')
    log_tau = ed.Normal(loc=0., scale=5., name='log_tau')
    theta = ed.Normal(
        loc=mu * tf.ones(num_schools),
        scale=tf.exp(log_tau) * tf.ones(num_schools), name='theta')
    y = ed.Normal(loc=theta, scale=stddevs, name='y')
    return y

  model_args = [num_schools, treatment_stddevs]
  observed = {'y': treatment_effects}

  varnames = ['mu', 'log_tau', 'theta']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(schools_model,
                                            model_args=model_args)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(schools_model, model_args=model_args)

  return ModelConfig(
      schools_model, model_args, observed,
      to_centered, to_noncentered, make_to_centered)


@contextlib.contextmanager
def open_from_url(url):
  filename = os.path.basename(url)
  path = os.path.expanduser(data_dir)
  filepath = os.path.join(path, filename)
  if not os.path.exists(filepath):
    if not tf.gfile.Exists(path):
      tf.gfile.MakeDirs(path)
    print('Downloading %s to %s' % (url, filepath))
    urllib.request.urlretrieve(url, filepath)
  with open(filepath, 'r') as f:
    yield f


def load_radon_data(state_code):
  """Load the radon dataset.

  Code from http://mc-stan.org/users/documentation/case-studies/radon.html.
  (Apache2 licensed)
  """

  # Non-NaN possibilities: MN, IN, MO, ND, PA

  with open_from_url(
      'http://www.stat.columbia.edu/~gelman/arm/examples/radon/srrs2.dat') as f:
    srrs2 = pd.read_csv(f)
  srrs2.columns = srrs2.columns.map(str.strip)
  srrs_mn = srrs2.assign(
      fips=srrs2.stfips * 1000 + srrs2.cntyfips)[srrs2.state == state_code]

  with open_from_url(
      'http://www.stat.columbia.edu/~gelman/arm/examples/radon/cty.dat') as f:
    cty = pd.read_csv(f)
  cty_mn = cty[cty.st == state_code].copy()
  cty_mn['fips'] = 1000 * cty_mn.stfips + cty_mn.ctfips

  srrs_mn.county = srrs_mn.county.str.strip()

  counties = srrs_mn[['county', 'fips']].drop_duplicates()
  county_map_uranium = {
      a: b for a, b in zip(counties['county'], range(len(counties['county'])))
  }
  uranium_levels = cty_mn.merge(counties, on='fips')['Uppm']

  srrs_mn_new = srrs_mn.merge(cty_mn[['fips', 'Uppm']], on='fips')
  srrs_mn_new = srrs_mn_new.drop_duplicates(subset='idnum')

  srrs_mn_new.county = srrs_mn_new.county.str.strip()
  mn_counties = srrs_mn_new.county.unique()
  county_lookup = dict(zip(mn_counties, range(len(mn_counties))))

  county = srrs_mn_new['county_code'] = srrs_mn_new.county.replace(
      county_lookup).values
  radon = srrs_mn_new.activity
  srrs_mn_new['log_radon'] = log_radon = np.log(radon + 0.1).values
  floor_measure = srrs_mn_new.floor.values

  n_county = srrs_mn_new.groupby('county')['idnum'].count()

  uranium = np.zeros(len(n_county), dtype=np.float32)
  for k, _ in county_lookup.items():
    uranium[county_lookup[k]] = uranium_levels[county_map_uranium[k]]

  uranium = [(np.log(ur) if ur > 0. else 0.) for ur in uranium]

  c = county  # length N (with J unique values)
  u = np.float32(uranium)  # length J
  x = np.float32(floor_measure)  # length N
  data = np.float32(log_radon).reshape(-1, 1)  # length N

  return c, u, x, data


def get_radon_model_stddvs(state_code='MN'):
  """Radon model with inferred scale parameters."""

  c, u, x, data = load_radon_data(state_code=state_code)
  N = data.shape[0]
  J = len(u)

  print('Radon N={}, J={}'.format(N, J))

  def radon(J, county, u, x):
    mua = ed.Normal(loc=0., scale=1., name='mua')  # scalar
    b1 = ed.Normal(loc=0., scale=1., name='b1')  # scalar
    b2 = ed.Normal(loc=0., scale=1., name='b2')  # scalar

    m_mu = mua + u * b1

    m = ed.Normal(loc=m_mu, scale=tf.ones(J), name='m')  # J

    C = tf.one_hot(county, J)  # shape (N, J)

    log_m_stddv = ed.Normal(
        loc=tf.zeros(J), scale=tf.ones(J), name='log_m_stddv')

    y_mu = tf.matmul(C, tf.expand_dims(m, 1)) + tf.expand_dims(x, 1) * b2
    y_stddv = tf.matmul(C, tf.expand_dims(tf.exp(log_m_stddv), 1))
    return ed.Normal(loc=y_mu, scale=y_stddv, name='y')

  model_args = [J, c, u, x]
  observed = {'y': data}

  varnames = ['mua', 'b1', 'b2', 'm', 'log_m_stddv']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(radon,
                                            model_args=model_args)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(radon, model_args=model_args)

  return ModelConfig(radon, model_args, observed,
                     to_centered, to_noncentered, make_to_centered)


def get_radon(state_code='MN'):
  """Build a model for the radon dataset.

  'N': len(log_radon),
  'J': len(n_county),
  'county': county+1, # Stan counts starting at 1
  'u': u,
  'x': floor_measure,
  'y': log_radon}
  """

  c, u, x, data = load_radon_data(state_code=state_code)
  N = data.shape[0]
  J = len(u)

  print('Radon N={}, J={}'.format(N, J))

  def radon(J, county, u, x, sigma_y):
    mua = ed.Normal(loc=0., scale=1., name='mua')  # scalar
    b1 = ed.Normal(loc=0., scale=1., name='b1')  # scalar
    b2 = ed.Normal(loc=0., scale=1., name='b2')  # scalar

    m_mu = mua + u * b1
    m = ed.Normal(loc=m_mu, scale=tf.ones(J), name='m')

    C = tf.one_hot(county, J)  # shape (N, J)

    y_mu = tf.matmul(C, tf.expand_dims(m, 1)) + tf.expand_dims(x, 1) * b2
    return ed.Normal(loc=y_mu, scale=sigma_y, name='y')

  sigma_y = 1.

  model_args = [J, c, u, x, sigma_y]
  observed = {'y': data}

  varnames = ['mua', 'b1', 'b2', 'm']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(radon,
                                            model_args=model_args)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(radon, model_args=model_args)

  return ModelConfig(radon, model_args, observed,
                     to_centered, to_noncentered, make_to_centered)


def load_german_credit_data():
  """Load the German credit dataset."""
  with open_from_url(
      'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'  # pylint: disable=line-too-long
  ) as f:
    data = pd.read_csv(f, delim_whitespace=True, header=None)

  def categorical_to_int(x):
    d = {u: i for i, u in enumerate(np.unique(x))}
    return np.array([d[i] for i in x])

  categoricals = []
  numericals = []
  numericals.append(np.ones([len(data)]))
  for column in data.columns[:-1]:
    column = data[column]
    if column.dtype == 'O':
      categoricals.append(categorical_to_int(column))
    else:
      numericals.append((column - column.mean()) / column.std())
  numericals = np.array(numericals).T
  status = np.array(data[20] == 1, dtype=np.int32)

  return numericals, categoricals, status


def get_german_credit_lognormalcentered():
  """German credit model with LogNormal priors on the coefficient scales."""
  numericals, categoricals, status = load_german_credit_data()

  def german_credit_model():
    x_numeric = tf.constant(numericals.astype(np.float32))
    x_categorical = [tf.one_hot(c, c.max() + 1) for c in categoricals]
    all_x = tf.concat([x_numeric] + x_categorical, 1)
    num_features = int(all_x.shape[1])

    overall_log_scale = ed.Normal(loc=0., scale=10., name='overall_log_scale')
    beta_log_scales = ed.Normal(loc=overall_log_scale,
                                scale=tf.ones([num_features]),
                                name='beta_log_scales')
    beta = ed.Normal(loc=tf.zeros([num_features]),
                     scale=tf.exp(beta_log_scales),
                     name='beta')
    logits = tf.einsum('nd,md->mn', all_x, beta[tf.newaxis, :])
    return ed.Bernoulli(logits=logits, name='y')

  observed = {'y': status[np.newaxis, Ellipsis]}
  model_args = []

  varnames = ['overall_log_scale', 'beta_log_scales', 'beta']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(german_credit_model,
                                            model_args=model_args)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(german_credit_model,
                                       model_args=model_args)

  return ModelConfig(german_credit_model, model_args, observed,
                     to_centered, to_noncentered, make_to_centered)


def get_german_credit_gammascale():
  """German credit model with Gamma priors on the coefficient scales."""
  numericals, categoricals, status = load_german_credit_data()

  def german_credit_model():
    x_numeric = tf.constant(numericals.astype(np.float32))
    x_categorical = [tf.one_hot(c, c.max() + 1) for c in categoricals]
    all_x = tf.concat([x_numeric] + x_categorical, 1)
    num_features = int(all_x.shape[1])

    overall_log_scale = ed.Normal(loc=0., scale=10., name='overall_log_scale')
    beta_log_scales = ed.TransformedDistribution(
        tfd.Gamma(0.5 * tf.ones([num_features]), 0.5),
        bijector=tfb.Invert(tfb.Exp()), name='beta_log_scales')
    beta = ed.Normal(loc=tf.zeros([num_features]),
                     scale=tf.exp(overall_log_scale + beta_log_scales),
                     name='beta')
    logits = tf.einsum('nd,md->mn', all_x, beta[tf.newaxis, :])
    return ed.Bernoulli(logits=logits, name='y')

  observed = {'y': status[np.newaxis, Ellipsis]}
  model_args = []

  varnames = ['overall_log_scale', 'beta_log_scales', 'beta']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(german_credit_model,
                                            model_args=model_args)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(german_credit_model,
                                       model_args=model_args)

  return (german_credit_model, model_args, observed,
          to_centered, to_noncentered, make_to_centered)
