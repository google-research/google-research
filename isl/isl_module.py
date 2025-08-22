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

"""The file contains definition source classes and functions for ISL module."""
import math

import metrics
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import torch
from torch import nn

# use GPU if it has
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# pylint: disable=invalid-name
# pylint: disable=f-string-without-interpolation
# pylint: disable=unused-variable
# pylint: disable=redefined-outer-name
# pylint: disable=redefined-builtin
class traceexpm(torch.autograd.Function):
  """Adapted from: https://github.com/xunzheng/notears/tree/master/notears."""

  @staticmethod
  def forward(ctx, input_tensor):
    edge_tensor = slin.expm(input_tensor.detach().numpy())
    f = np.trace(edge_tensor)
    edge_tensor = torch.from_numpy(edge_tensor)
    ctx.save_for_backward(edge_tensor)
    return torch.as_tensor(f, dtype=input_tensor.dtype)

  @staticmethod
  def backward(ctx, grad_output):
    edge_tensor, = ctx.saved_tensors
    grad_input = grad_output * edge_tensor.t()
    return grad_input


trace_expm = traceexpm.apply


class lbfgsbscipy(torch.optim.Optimizer):
  """Wraps L-BFGS-B algorithm, using scipy routines.

  Adapted from:: https://github.com/xunzheng/notears/tree/master/notears
  and
  https://gist.github.com/arthurmensch/c55ac413868550f89225a0b9212aa4cd
  """

  def __init__(self, params):
    defaults = dict()
    super(lbfgsbscipy, self).__init__(params, defaults)

    if len(self.param_groups) != 1:
      raise ValueError("LBFGSBScipy doesn't support per-parameter options"
                       ' (parameter groups)')

    self._params = self.param_groups[0]['params']
    self._numel = sum([p.numel() for p in self._params])

  def _gather_flat_grad(self):
    views = []
    for p in self._params:
      if p.grad is None:
        view = p.data.new(p.data.numel()).zero_()
      elif p.grad.data.is_sparse:
        view = p.grad.data.to_dense().view(-1)
      else:
        view = p.grad.data.view(-1)
      views.append(view)
    return torch.cat(views, 0)

  def _gather_flat_bounds(self):
    bounds = []
    for p in self._params:
      if hasattr(p, 'bounds'):
        b = p.bounds
      else:
        b = [(None, None)] * p.numel()
      bounds += b
    return bounds

  def _gather_flat_params(self):
    views = []
    for p in self._params:
      if p.data.is_sparse:
        view = p.data.to_dense().view(-1)
      else:
        view = p.data.view(-1)
      views.append(view)
    return torch.cat(views, 0)

  def _distribute_flat_params(self, params):
    offset = 0
    for p in self._params:
      numel = p.numel()
      # view as to avoid deprecated pointwise semantics
      p.data = params[offset:offset + numel].view_as(p.data)
      offset += numel
    assert offset == self._numel

  def step(self, closure):
    """Performs a single optimization step.

    Args:
      closure (callable): A closure that reevaluates the model and returns loss.
    """
    assert len(self.param_groups) == 1

    def wrapped_closure(flat_params):

      flat_params = torch.from_numpy(flat_params)
      flat_params = flat_params.to(torch.get_default_dtype())
      self._distribute_flat_params(flat_params)
      loss = closure()
      loss = loss.item()
      flat_grad = self._gather_flat_grad().cpu().detach().numpy()
      return loss, flat_grad.astype('float64')

    initial_params = self._gather_flat_params()
    initial_params = initial_params.cpu().detach().numpy()

    bounds = self._gather_flat_bounds()

    sol = sopt.minimize(
        wrapped_closure,
        initial_params,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds)

    final_params = torch.from_numpy(sol.x)
    final_params = final_params.to(torch.get_default_dtype())
    self._distribute_flat_params(final_params)


class locally_connected(nn.Module):
  """Local linear layer.

  Adapted from: https://github.com/xunzheng/notears/tree/master/notears
  """

  def __init__(self, num_linear, input_features, output_features, bias=True):
    super(locally_connected, self).__init__()

    self.num_linear = num_linear
    self.input_features = input_features
    self.output_features = output_features
    self.weight = nn.Parameter(
        torch.Tensor(num_linear, input_features, output_features))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
    else:

      self.register_parameter('bias', None)

    self.reset_parameters()

  @torch.no_grad()
  def reset_parameters(self):
    k = 1.0 / self.input_features
    bound = math.sqrt(k)
    nn.init.uniform_(self.weight, -bound, bound)
    if self.bias is not None:
      nn.init.uniform_(self.bias, -bound, bound)

  def forward(self, input_tensor):
    out = torch.matmul(
        input_tensor.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
    out = out.squeeze(dim=2)
    if self.bias is not None:
      out += self.bias
    return out

  def extra_repr(self):

    return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
        self.num_linear, self.in_features, self.out_features, self.bias
        is not None)


class notearsmlp_module(nn.Module):
  """Notears-MLP class."""

  def __init__(self, dims, bias=True):
    super(notearsmlp_module, self).__init__()
    assert len(dims) >= 2
    assert dims[-1] == 1
    d = dims[0]
    self.dims = dims
    # fc1: variable splitting for l1
    self.fc1_pos = nn.Linear(
        d, (d - 1) * dims[1], bias=bias)  # d-1 means variables except Y
    self.fc1_neg = nn.Linear(d, (d - 1) * dims[1], bias=bias)
    self.fc1_pos.weight.bounds = self._bounds()[d *
                                                dims[1]:]  # skip Y related dims
    self.fc1_neg.weight.bounds = self._bounds()[d *
                                                dims[1]:]  # skip Y related dims
    # fc2: local linear layers
    layers = []
    for l in range(len(dims) - 2):
      layers.append(locally_connected(d, dims[l + 1], dims[l + 2], bias=bias))
    self.fc2 = nn.ModuleList(layers)

  def _bounds(self):
    d = self.dims[0]
    bounds = []
    for j in range(d):
      for m in range(self.dims[1]):
        for i in range(d):
          if i == j:
            bound = (0, 0)
          else:
            bound = (0, None)
          bounds.append(bound)
    return bounds

  def forward(self, x, y):  # [n, d] -> [n, d]
    x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, (d-1) * m1]  contain no y
    x = torch.cat((y, x), 1)  # put y into x, # [n, d * m1]
    x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
    for fc in self.fc2:
      # x = torch.sigmoid(x)  # [n, d, m1]
      x = torch.nn.functional.relu(x)  # [n, d, m1]
      x = fc(x)  # [n, d, m2]
    x = x.squeeze(dim=2)  # [n, d]
    return x

  def h_func(self, fc1_Y_weight):
    """Constrains 2-norm-squared of fc1 weights along m1 dim to be a DAG."""
    d = self.dims[0]
    fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [(j-1) * m1, i]
    fc1_weight = torch.cat((torch.tensor(fc1_Y_weight), fc1_weight),
                           0)  # [j * m1, i]
    fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
    A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
    h = trace_expm(A) - d  # (Zheng et al. 2018)
    return h

  def l2_reg(self):
    """Takes 2-norm-squared of all parameters."""
    reg = 0.
    fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
    reg += torch.sum(fc1_weight**2)
    for fc in self.fc2:
      reg += torch.sum(fc.weight**2)
    return reg

  def fc1_l1_reg(self):
    """Takes l1 norm of fc1 weight."""
    reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
    return reg

  @torch.no_grad()
  def fc1_to_adj(self, fc1_Y_weight):  # [j * m1, i] -> [i, j]
    """Gets W from fc1 weights, take 2-norm over m1 dim."""
    d = self.dims[0]
    fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [(j-1) * m1, i]
    fc1_weight = torch.cat((torch.tensor(fc1_Y_weight), fc1_weight),
                           0)  # [j * m1, i]
    fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
    A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
    W = torch.sqrt(A)  # [i, j]
    W = W.cpu().detach().numpy()  # [i, j]
    return W


class notearsmlp_self_supervised(nn.Module):
  """Notears-MLP class for self-supervised learning."""

  def __init__(self, dims, bias=True, init_dag=None):
    super(notearsmlp_self_supervised, self).__init__()
    assert len(dims) >= 2
    assert dims[-1] == 1
    d = dims[0]
    self.dims = dims

    self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
    self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)

    self.fc1_pos.weight.bounds = self._bounds_Init(init_dag)
    self.fc1_neg.weight.bounds = self._bounds_Init(init_dag)

    layers = []
    for l in range(len(dims) - 2):
      layers.append(locally_connected(d, dims[l + 1], dims[l + 2], bias=bias))
    self.fc2 = nn.ModuleList(layers)

  def _bounds_Init(self, init_dag):
    d = self.dims[0]
    bounds = []
    for j in range(d):  # column in DAG
      for m in range(self.dims[1]):
        for i in range(d):  # row in DAG
          if i == j:
            bound = (0, 0)  # no cheat
          elif init_dag[i, j] == 0:
            bound = (0, 0)  # i  f i is not potential parent of j, then bound 0
          else:
            bound = (0, None)
          bounds.append(bound)
    return bounds

  def _bounds(self):
    d = self.dims[0]
    bounds = []
    for j in range(d):
      for m in range(self.dims[1]):
        for i in range(d):
          if i == j:
            bound = (0, 0)
          else:
            bound = (0, None)
          bounds.append(bound)
    return bounds

  def forward(self, x):  # [n, d] -> [n, d]
    x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
    x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
    for fc in self.fc2:
      # x = torch.sigmoid(x)  # [n, d, m1]
      x = torch.nn.functional.relu(x)  # [n, d, m1]
      x = fc(x)  # [n, d, m2]
    x = x.squeeze(dim=2)  # [n, d]
    return x

  def h_func(self):
    """Constrains 2-norm-squared of fc1 weights along m1 dim to be a DAG."""

    d = self.dims[0]
    fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
    fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
    A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
    h = trace_expm(A) - d  # (Zheng et al. 2018)
    return h

  def l2_reg(self):
    """Takes 2-norm-squared of all parameters."""

    reg = 0.
    fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
    reg += torch.sum(fc1_weight**2)
    for fc in self.fc2:
      reg += torch.sum(fc.weight**2)
    return reg

  def fc1_l1_reg(self):
    """Takes l1 norm of fc1 weight."""

    reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
    return reg

  @torch.no_grad()
  def fc1_to_adj(self):  # [j * m1, i] -> [i, j]
    """Gets W from fc1 weights, take 2-norm over m1 dim."""

    d = self.dims[0]
    fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
    fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
    A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
    W = torch.sqrt(A)  # [i, j]
    W = W.cpu().detach().numpy()  # [i, j]
    return W

  @torch.no_grad()
  def test(self, x):
    """Tests function."""

    x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
    x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
    for fc in self.fc2:
      # x = torch.sigmoid(x)  # [n, d, m1]
      x = torch.nn.functional.relu(x)  # [n, d, m1]
      x = fc(x)  # [n, d, m2]
    x = x.squeeze(dim=2)  # [n, d]
    return x


class isl_module(nn.Module):
  """ISL module class."""

  def __init__(self, n_envs, Y_dims, dims, bias=True):
    super(isl_module, self).__init__()
    assert len(dims) >= 2
    assert dims[-1] == 1
    d = dims[0]
    self.dims = dims
    # fc1_Y shared among all envirobments
    self.fc1_Y_pos = nn.Linear(d, dims[1], bias=bias)
    self.fc1_Y_neg = nn.Linear(d, dims[1], bias=bias)
    YX_bounds = self._bounds()
    self.fc1_Y_pos.weight.bounds = YX_bounds[:d * dims[
        1]]  # only select Y related region
    self.fc1_Y_neg.weight.bounds = YX_bounds[:d * dims[
        1]]  # only select Y related region
    Y_layers = []
    for l in range(len(Y_dims) - 2):  # normal MLP
      Y_layers.append(nn.Linear(Y_dims[l + 1], Y_dims[l + 2], bias=bias))
    self.fc2_Y = nn.ModuleList(Y_layers)
    # create model for each environment
    self.IRM = nn.ModuleList()  # for each environment
    for env_index in range(n_envs):  # for each environment
      self.IRM.append(notearsmlp_module(dims, bias=True))

  def _bounds(self):
    d = self.dims[0]
    bounds = []
    for j in range(d):
      for m in range(self.dims[1]):
        for i in range(d):
          if i == j:
            bound = (0, 0)
          else:
            bound = (0, None)
          bounds.append(bound)
    return bounds

  def forward(self, x):  # [n, d] -> [n, d]
    x_all_env = []
    y_all_env = []
    for env_index in range(x.shape[0]):  # for each environment
      x_env = x[env_index]
      y_env = self.fc1_Y_pos(x_env) - self.fc1_Y_neg(x_env)  # [n, 1 * m1]
      x_env = self.IRM[env_index](x_env,
                                  y_env)  # input [n, (d-1) * m1]  output [n, d]
      for i in range(len(self.fc2_Y)):
        if i == len(self.fc2_Y) - 1:  # last layer (output layer)
          y_env = self.fc2_Y[i](y_env)
        else:  # hidden layer,
          y_env = torch.nn.functional.relu(self.fc2_Y[i](y_env))
      x_all_env.append(x_env)
      y_all_env.append(y_env)
    return x_all_env, y_all_env

  def test(self, x):
    x = torch.from_numpy(x)
    y = self.fc1_Y_pos(x) - self.fc1_Y_neg(x)  # [n, 1 * m1]
    for i in range(len(self.fc2_Y)):
      if i == len(self.fc2_Y) - 1:  # last layer (output layer)
        y = self.fc2_Y[i](y)

      else:  # hidden layer,
        y = torch.nn.functional.relu(self.fc2_Y[i](y))

    return y.detach().numpy()

  def h_func(self):
    """Constrains 2-norm-squared of fc1 weights along m1 dim to be a DAG."""
    d = self.dims[0]
    fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
    fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
    A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
    h = trace_expm(A) - d  # (Zheng et al. 2018)
    return h

  def l2_reg(self):
    """Takes 2-norm-squared of all parameters."""
    reg = 0.
    fc1_weight = self.fc1_Y_pos.weight - self.fc1_Y_neg.weight  # [j * m1, i]
    reg += torch.sum(fc1_weight**2)
    for fc in self.fc2_Y:
      reg += torch.sum(fc.weight**2)
    return reg

  def fc1_l2_reg(self):
    """Takes 2-norm-squared of all parameters."""

    reg = 0.
    fc1_weight = self.fc1_Y_pos.weight - self.fc1_Y_neg.weight  # [j * m1, i]
    reg += torch.sum(fc1_weight**2)
    return reg

  def fc2_l2_reg(self):
    """Takes 2-norm-squared of all parameters."""

    reg = 0.
    for fc in self.fc2_Y:
      reg += torch.sum(fc.weight**2)
    return reg

  def fc1_l1_reg(self):
    """Takes l1 norm of fc1 weight."""

    reg = torch.sum(self.fc1_Y_pos.weight + self.fc1_Y_neg.weight)
    return reg

  @torch.no_grad()
  def fc1_to_adj(self):  # [j * m1, i] -> [i, j]
    """Gets W from fc1 weights, take 2-norm over m1 dim."""

    d = self.dims[0]
    fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
    fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
    A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
    W = torch.sqrt(A)  # [i, j]
    W = W.cpu().detach().numpy()  # [i, j]
    return W


def dual_ascent_step(model, X, y_index, lambda1, lambda2, lambda1_Y,
                     lambda2_Y_fc1, lambda2_Y_fc2, param, rho_max):
  """Performs one step of dual ascent in augmented Lagrangian."""
  optimizer = lbfgsbscipy(model.parameters())
  X_torch = torch.from_numpy(X)
  X_torch.to(device)
  h_new = []
  for env_index in range(
      X.shape[0]):  # for each env, have different rho, alpha, beta, h
    h_new.append(None)

  while (min([env_param['rho'] for env_param in param]) <
         rho_max):  # all rho > rho_max can jump out

    def closure():
      optimizer.zero_grad()
      X_hat, Y_hat = model(X_torch)
      primal_obj = 0
      for env_index in range(X.shape[0]):  # for each environment
        loss = metrics.mean_squared_loss(X_hat[env_index], X_torch[env_index])
        y_pred_loss = metrics.mean_squared_loss(Y_hat[env_index].squeeze(1),
                                                X_torch[env_index][:, y_index])
        h_val = model.IRM[env_index].h_func(model.fc1_Y_pos.weight -
                                            model.fc1_Y_neg.weight)
        penalty = (0.5 * param[env_index]['rho'] * h_val * h_val +
                   param[env_index]['alpha'] * h_val)
        l2_reg = (0.5 * lambda2 * (model.IRM[env_index].l2_reg()) +
                  0.5 * lambda2_Y_fc1 * model.fc1_l2_reg() +
                  0.5 * lambda2_Y_fc2 * model.fc2_l2_reg())
        l1_reg = (
            lambda1 * (model.IRM[env_index].fc1_l1_reg()) +
            lambda1_Y * model.fc1_l1_reg())
        primal_obj += (param[env_index]['beta'] * y_pred_loss + loss + penalty +
                       l2_reg + l1_reg)  # accumulate loss of all envs

      primal_obj.backward()
      return primal_obj

    break_flag = True
    optimizer.step(closure)  # NOTE: updates model in-place
    with torch.no_grad():
      h_new_envs = []
      for env_index in range(X.shape[0]):  # for each environment
        h_new = model.IRM[env_index].h_func(model.fc1_Y_pos.weight -
                                            model.fc1_Y_neg.weight).item()
        if h_new > 0.25 * param[env_index][
            'h']:  # any env do not satisfied break requirement of h
          param[env_index]['rho'] *= 10  # update rho in each step
          break_flag = False
        h_new_envs.append(h_new)
    if break_flag:
      break
  # update params
  for env_index in range(X.shape[0]):  # for each environment
    param[env_index]['alpha'] += param[env_index]['rho'] * h_new_envs[env_index]
    param[env_index]['h'] = h_new_envs[env_index]

  return param


def notears_nonlinear(model,
                      X,
                      y_index = 0,
                      lambda1 = 0.,
                      lambda2 = 0.,
                      lambda1_Y = 0.,
                      lambda2_Y_fc1 = 0.,
                      lambda2_Y_fc2 = 0.,
                      max_iter = 100,
                      h_tol = 1e-8,
                      rho_max = 1e+16,
                      w_threshold = 0.3,
                      beta = 1):
  """Nonliner notears function."""
  n_envs = X.shape[0]
  param = []
  for env_index in range(n_envs):  # each environment
    param_e = {}
    param_e['rho'], param_e['alpha'], param_e['beta'], param_e[
        'h'] = 1.0, 0.0, beta, np.inf
    param.append(param_e)

  # rho, alpha, beta, h = 1.0, 0.0, 1.0, np.inf
  for iter in range(max_iter):
    print('staring iter {}'.format(iter))
    param = dual_ascent_step(model, X, y_index, lambda1, lambda2, lambda1_Y,
                             lambda2_Y_fc1, lambda2_Y_fc2, param, rho_max)
    for env_index in range(n_envs):
      print('Finish Step {}, env-{}, rho: {}, alpha: {}, h: {}'.format(
          env_index, iter, param[env_index]['rho'], param[env_index]['alpha'],
          param[env_index]['h']))
    h_max = max([env_param['h'] for env_param in param])
    rho_min = min([env_param['rho'] for env_param in param])
    if h_max <= h_tol or rho_min >= rho_max:  # all environment are dag now
      break

  W_est_envs = []
  y_pred_loss = 0
  for env_index in range(n_envs):
    W_est = model.IRM[env_index].fc1_to_adj(model.fc1_Y_pos.weight -
                                            model.fc1_Y_neg.weight)
    W_est[np.abs(W_est) < w_threshold] = 0
    W_est_envs.append(W_est)

    X_torch = torch.from_numpy(X)
    X_hat, Y_hat = model(X_torch)
    y_pred_loss += metrics.mean_squared_loss(Y_hat[env_index].squeeze(1),
                                             X_torch[env_index][:, y_index])

  return y_pred_loss / n_envs, W_est_envs
