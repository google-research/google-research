# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# pylint: skip-file
from absl import logging
from jax import random

from jax.experimental import stax
from jax.experimental.stax import Dense, MaxPool, Relu, Flatten, LogSoftmax
from jax.experimental import optimizers
from flax import optim as flax_optim
from . import equations
from . import meshes
from . import gmres
from . import flax_cnn
import os
import functools
import jax
import flax
from flax import nn
from jax import lax
import jax.numpy as np
import numpy as onp
import jax.ops
from jax.tree_util import Partial

randn = stax.randn
glorot = stax.glorot

# Loss functions


@functools.partial(jax.vmap, in_axes=(None, None, None, None, 0, 0))
def losses_gmres_inf(preconditioner, params, n, new_matvec, x0, b):
  A = Partial(new_matvec)
  M = Partial(preconditioner, params)
  x_opt = gmres.gmres(A, b, x0, n=n, M=M)

  return np.linalg.norm(A(x_opt) - b, np.inf) * 1000 * x_opt.shape[0]


@functools.partial(jax.vmap, in_axes=(None, None, None, None, 0, 0))
def losses_gmres(preconditioner, params, n, new_matvec, x0, b):
  A = Partial(new_matvec)
  M = Partial(preconditioner, params)
  loss = gmres.gmres_training(A, b, x0, n=n, M=M)
  return loss * 10000000

@functools.partial(jax.vmap, in_axes=(None, None, None, None, 0, 0))
def losses_gmres_flax(preconditioner, model, n, new_matvec, x0, b):
  A = Partial(new_matvec)
  M = Partial(preconditioner, model)
  #loss = gmres.gmres_training(A, b, x0, n=n, M=M)
  x = gmres.gmres(A, b, x0, n=n, M=M)

  return np.linalg.norm(A(x) - b) * 10000000


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def loss_gmres(preconditioner,
               n,
               shapeL,
               mesh,
               params,
               inputs,
               bs,
               x=0,
               k=0,
               aspect_ratio=1.0,
               **kwargs):
  if shapeL == 'R':
    new_matvec = lambda x: mesh.matvec_helmholtz(
        k, aspect_ratio, equations.make_mask, equations.make_mask_dual, x)
  elif shapeL == 'L':
    new_matvec = lambda x: mesh.matvec_helmholtz(
        k, aspect_ratio, equations.make_mask_L, equations.make_mask_L_dual, x)
  return np.mean(
      losses_gmres(preconditioner, params, n, new_matvec, inputs, bs))


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def loss_gmresR(preconditioner,
                n,
                mesh,
                params,
                inputs,
                bs,
                x=0,
                k=0,
                aspect_ratio=1.0,
                **kwargs):
  new_matvec = lambda y: mesh.matvec_helmholtz(
      k, aspect_ratio, equations.make_mask, equations.make_mask_dual, y)
  return np.mean(
      losses_gmres(preconditioner, params, n, new_matvec, inputs, bs))

@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def loss_gmresR_flax(preconditioner,
                n,
                mesh,
                model,
                inputs,
                bs,
                x=0,
                k=0,
                aspect_ratio=1.0,
                **kwargs):
  new_matvec = lambda y: mesh.matvec_helmholtz(
      k, aspect_ratio, equations.make_mask, equations.make_mask_dual, y)
  return np.mean(
      losses_gmres_flax(preconditioner, model, n, new_matvec, inputs, bs))

# CNN definition
# Like the convolutions from stax, but without bias.

def GeneralUnbiasedConv(dimension_numbers,
                        out_chan,
                        filter_shape,
                        strides=None,
                        padding='SAME',
                        W_init=None,
                        b_init=randn(1e-6)):
  """Layer construction function for a general convolution layer."""
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  one = (1,) * len(filter_shape)
  strides = strides or one
  W_init = W_init or glorot(rhs_spec.index('O'), rhs_spec.index('I'))

  def init_fun(rng, input_shape):
    filter_shape_iter = iter(filter_shape)
    kernel_shape = [
        out_chan if c == 'O' else input_shape[lhs_spec.index('C')]
        if c == 'I' else next(filter_shape_iter) for c in rhs_spec
    ]
    output_shape = lax.conv_general_shape_tuple(input_shape, kernel_shape,
                                                strides, padding,
                                                dimension_numbers)
    W = W_init(rng, kernel_shape)
    return output_shape, W

  def apply_fun(params, inputs, **kwargs):
    W = params
    return lax.conv_general_dilated(inputs, W, strides, padding, one, one,
                                    dimension_numbers)

  return init_fun, apply_fun


UnbiasedConv = functools.partial(GeneralUnbiasedConv, ('NHWC', 'HWIO', 'NHWC'))


def GeneralUnbiasedConvTranspose(dimension_numbers,
                                 out_chan,
                                 filter_shape,
                                 strides=None,
                                 padding='SAME',
                                 W_init=None,
                                 b_init=randn(1e-6)):
  """Layer construction function for a general transposed-convolution layer."""
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  one = (1,) * len(filter_shape)
  strides = strides or one
  W_init = W_init or glorot(rhs_spec.index('O'), rhs_spec.index('I'))

  def init_fun(rng, input_shape):
    filter_shape_iter = iter(filter_shape)
    kernel_shape = [
        out_chan if c == 'O' else input_shape[lhs_spec.index('C')]
        if c == 'I' else next(filter_shape_iter) for c in rhs_spec
    ]
    output_shape = lax.conv_transpose_shape_tuple(input_shape, kernel_shape,
                                                  strides, padding,
                                                  dimension_numbers)
    W = W_init(rng, kernel_shape)
    return output_shape, W

  def apply_fun(params, inputs, **kwargs):
    W = params
    return lax.conv_transpose(
        inputs, W, strides, padding, dimension_numbers=dimension_numbers)

  return init_fun, apply_fun


UnbiasedConvTranspose = functools.partial(GeneralUnbiasedConvTranspose,
                                          ('NHWC', 'HWIO', 'NHWC'))


def UNetBlock(filters, kernel_size, inner_block, **kwargs):

  def make_main(input_shape):
    return stax.serial(
        UnbiasedConv(filters, kernel_size, **kwargs),
        inner_block,
        UnbiasedConvTranspose(input_shape[3], kernel_size, **kwargs),
    )

  Main = stax.shape_dependent(make_main)
  return stax.serial(
      stax.FanOut(2), stax.parallel(Main, stax.Identity), stax.FanInSum)


class UNet:
  """UNet that mimics 3-cycle V_cycle structure."""

  def __init__(self,
               n=2**7 - 1,
               rng=None,
               channels=8,
               loss=loss_gmres,
               iter_gmres=lambda i: 10,
               training_iter=500,
               name='net',
               model_dir=None,
               lr=3e-4,
               k=0.0,
               n_test=10,
               beta1=0.9,
               beta2=0.999,
               lr_og = 3e-3,
               flaxd = False):
    self.n = n
    self.n_test = n_test
    self.mesh = meshes.Mesh(n)
    self.in_shape = (-1, n, n, 1)
    self.inner_channels = channels
    def itera(i):
      return onp.random.choice([5, 10, 10, 10, 10, 15, 15, 15, 20, 25])
    self.iter_gmres = itera
    self.training_iter = training_iter
    self.name = name
    self.k = k
    self.model_dir = model_dir
    if flaxd:
      self.test_loss = loss_gmresR_flax
    else:
      self.test_loss = loss_gmresR
    self.beta1 = beta1
    self.beta2 = beta2
    if rng is None:
      rng = random.PRNGKey(1)
    if not flaxd:
      self.net_init, self.net_apply = stax.serial(
          UNetBlock(
              1, (3, 3),
              stax.serial(
                  UnbiasedConv(self.inner_channels, (3, 3), padding='SAME'),
                  UnbiasedConv(self.inner_channels, (3, 3), padding='SAME'),
                  UNetBlock(
                      self.inner_channels, (3, 3),
                      stax.serial(
                          UnbiasedConv(
                              self.inner_channels, (3, 3), padding='SAME'),
                          UnbiasedConv(
                              self.inner_channels, (3, 3), padding='SAME'),
                          UnbiasedConv(
                              self.inner_channels, (3, 3), padding='SAME'),
                      ),
                      strides=(2, 2),
                      padding='VALID'),
                  UnbiasedConv(self.inner_channels, (3, 3), padding='SAME'),
                  UnbiasedConv(self.inner_channels, (3, 3), padding='SAME'),
              ),
              strides=(2, 2),
              padding='VALID'),)
      out_shape, net_params = self.net_init(rng, self.in_shape)
    else:
      #import pdb;pdb.set_trace()
      model_def = flax_cnn.new_CNN.partial(
          inner_channels=self.inner_channels)
      out_shape, net_params = model_def.init_by_shape(
          rng,[(self.in_shape, np.float32)])
      self.model_def = model_def
      self.model = nn.Model(model_def, net_params)
      self.net_apply = lambda param, x: nn.Model(model_def,
                                                 param)(x) #.reshape(self.in_shape))
    self.out_shape = out_shape
    self.net_params = net_params
    self.loss = loss
    self.lr_og = lr_og
    self.lr = lr
    if not flaxd:
      self.opt_init, self.opt_update, self.get_params = optimizers.adam(
          step_size=lambda i: np.where(i < 100, lr_og, lr), b1=beta1, b2=beta2)
      self.opt_state = self.opt_init(self.net_params)
      self.step = self.step_notflax

    if flaxd:
      self.step = self.step_flax
      self.optimizer = flax.optim.Adam(
          learning_rate= lr, beta1=beta1,
          beta2=beta2).create(self.model)
      #self.optimizer = flax.optim.Momentum(
      #    learning_rate= lr, beta=beta1,
      #    weight_decay=0, nesterov=False).create(self.model)
    self.alpha = lambda i: 0.0
    self.flaxd = flaxd
    if flaxd:
      self.preconditioner = self.preconditioner_flaxed
    else:
      self.preconditioner = self.preconditioner_unflaxed

  def preconditioner_unflaxed(self, params, x):
    return self.net_apply(params, x.reshape(1, self.n, self.n, 1)).ravel()

  def preconditioner_flaxed(self, model, x):
    return model(x.reshape(1, self.n, self.n, 1)).ravel()

  @functools.partial(jax.jit, static_argnums=(0,))
  def step_notflax(self, i, opt_state, batch, bs, solutions=None):
    params = self.get_params(opt_state)
    curr_loss, g = jax.value_and_grad(
        self.loss, argnums=3)(
            self.preconditioner,
            self.iter_gmres(i),
            self.mesh,
            params,
            batch,
            bs,
            self.alpha(i),
            self.k,
            solutions=solutions)
    return curr_loss, g, self.opt_update(i, g, opt_state)

  @functools.partial(jax.jit, static_argnums=(0,))
  def step_flax(self, i, optimizer, batch, bs, solutions=None):
    curr_loss, grad = jax.value_and_grad(
        self.loss, argnums=3)(
            self.preconditioner,
            self.iter_gmres(i),
            self.mesh,
            optimizer.target,
            batch,
            bs,
            self.alpha(i),
            self.k,
            solutions=solutions)
    optimizer = optimizer.apply_gradient(grad)
    return curr_loss, grad, optimizer

  def save(self, i=''):
    if self.model_dir is None:
      serialization.save_params(self.name + 'params' + i, self.opt_params)
    else:
      serialization.save_params(self.model_dir + '/' + self.name + 'params' + i,
                                self.opt_params)


  def load(self, i=''):
    directory = os.path.join(self.model_dir, self.name + 'params' + i)
    if not Exists(directory):
      logging.info('still training')
      return 1
    self.opt_params = serialization.load_params(directory)
    if self.flaxd:
      self.model = nn.Model(self.model_def, self.opt_params)
    return 0


  def train(self,
            bs,
            solutions=[None],
            retrain=False,
            tensorboard_writer=None,
            work_unit=None):

    if not retrain and not self.flaxd:
      opt_state = self.opt_init(self.net_params)
    if retrain:
      opt_state = self.opt_init(self.opt_params)
    loss = onp.zeros(self.training_iter // 10 + 1)
    gradients = onp.zeros(self.training_iter // 10 + 1)
    if not self.flaxd:
      param = self.get_params(opt_state)
    else:
      param = self.optimizer.target
      opt_state = self.optimizer
    og_loss = self.test_loss(
      self.preconditioner, self.n_test, self.mesh,
      param, np.zeros(
          (bs.shape[1], self.n * self.n)),
        bs[0].reshape(bs.shape[1],
                   self.n * self.n), 0, self.k) / 10000000
    print(og_loss)
    if work_unit is not None:
      work_unit.get_measurement_series(
          label='train/loss').create_measurement(
              objective_value=og_loss, step=0)
    for i in range(self.training_iter):
      m = bs.shape[0]
      order = random.shuffle(random.PRNGKey(i), np.arange(m))
      for _ in range(50):
        for b in bs[order]:
          current_loss, grad, opt_state = self.step(i,
                                                    opt_state,
                                                    np.zeros((b.shape[0],
                                                              self.n * self.n)),
                                                    b,
                                                    solutions[min(m,
                                                                  len(solutions)
                                                                  - 1)])

      if i % 10 == 0:
        if not self.flaxd:
          param = self.get_params(opt_state)
        else:
          param = opt_state.target
        current_loss_test = self.test_loss(
            self.preconditioner, self.n_test, self.mesh, param, np.zeros(
                (b.shape[0], self.n * self.n)), b, 0, self.k) / 10000000
        current_loss = current_loss / 10000000
        avg_grad = onp.mean(onp.abs(onp_utils.flatten(grad)[-1]))
        print(f'step{i: 5d}: loss { current_loss :1.5f} : avg_gradient \
              { avg_grad :1.5f} : current_loss_test { current_loss_test :1.5f}')
        logging.info(f'step{i: 5d}: loss { current_loss :1.5f} : avg_gradient \
              { avg_grad :1.5f} : current_loss_test { current_loss_test :1.5f}')
        loss[i // 10] = current_loss
        gradients[i // 10] = avg_grad
        if work_unit is not None:
          work_unit.get_measurement_series(
              label='train/loss').create_measurement(
                  objective_value=current_loss_test, step=i)
          tensorboard_writer.scalar('train/loss', current_loss_test, step=i+1)
          work_unit.get_measurement_series(
              label='train/loss ' + str(self.iter_gmres(i))).create_measurement(
                  objective_value=current_loss, step=i+1)
          tensorboard_writer.scalar(
              'train/loss ' + str(self.iter_gmres(i)), current_loss, step=i+1)
      if i % 50 == 0:
        if self.flaxd:
          self.opt_params = opt_state.target.params
        else:
          self.opt_params = self.get_params(opt_state)
        self.save(str(i))
    if self.flaxd:
      self.optimizer = opt_state
    else:
      self.opt_params = self.get_params(opt_state)
      self.opt_state = opt_state
    if self.model_dir is None:
      self.model_dir = ''

    with open(os.path.join(self.model_dir, 'train_loss.np'), 'wb') as f:
      onp.save(f, loss)
    with open(os.path.join(self.model_dir, 'train_gradients.np'),
                    'wb') as f:
      onp.save(f, gradients)
    self.save()
    if work_unit is not None:
      tensorboard_writer.close()

  @functools.partial(jax.jit, static_argnums=(0,))
  def approximate_inverse(self, inputs):
    return self.net_apply(self.opt_params, inputs.reshape(1, self.n, self.n,
                                                          1)).reshape(-1)

  def print_layer_shape(self):
    print([array.shape for array in jax.tree_flatten(self.net_params)[0]])
