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

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import torch
from torch import nn
from torch.utils.cpp_extension import load

stacked_fc = load(name='stacked_fc', sources=['ops/stacked_fc_cuda.cu'])


class ShiftedSoftplus(nn.Module):

  def __init__(self, shift=-5., beta=1., threshold=20.):
    super().__init__()
    self.shift = shift
    self.softplus = nn.Softplus(beta=beta, threshold=threshold)

  def forward(self, x):
    return self.softplus(x + self.shift)


class StackedFcFastFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, input_feat, input_idx, weights, bias):
    input_feat = input_feat.contiguous()
    out = stacked_fc.forward(input_feat, input_idx, weights, bias)
    ctx.save_for_backward(out, input_feat, input_idx, weights, bias)
    return out

  @staticmethod
  def backward(ctx, grad_out):
    outputs = stacked_fc.backward(*ctx.saved_tensors, grad_out.contiguous())
    d_feat, d_idx, d_weight, d_bias = outputs
    return d_feat, d_idx, d_weight, d_bias

  @staticmethod
  def jvp(ctx, *grad_inputs):
    pass


class StackedFcDense(torch.nn.Module):
  """Inputs: x (B, IN_C)

    Outputs: (B, N, OUT_C)
  """

  def __init__(self, n, k, in_c, out_c, act_func='relu'):
    super().__init__()

    assert act_func in ['relu', 'sigmoid', 'ssoftplus', 'none']
    if act_func == 'relu':
      gain = math.sqrt(2)
      self.act_func = torch.nn.ReLU(inplace=True)
    elif act_func == 'sigmoid':
      gain = 1.0
      self.act_func = torch.nn.Sigmoid()
    elif act_func == 'ssoftplus':
      gain = math.sqrt(2)
      self.act_func = ShiftedSoftplus()
    else:
      gain = 1.0
      self.act_func = torch.nn.Identity()

    self.n = n
    self.in_c = in_c
    self.out_c = out_c

    # (N, IN_C, OUT_C)
    self.w = torch.nn.Parameter(
        torch.nn.init.xavier_uniform_(torch.empty(n, in_c, out_c), gain))
    # (N, 1, OUT_C)
    self.b = torch.nn.Parameter(torch.zeros(n, 1, out_c))

  def forward(self, x, *_, **kwargs):
    """Inputs: x (B, IN_C)

        Outputs: (N, B, OUT_C)
    """
    return self.act_func(torch.matmul(x, self.w) + self.b)


class StackedFcSlow(torch.nn.Module):
  """Evaluate a batch of samples (x) on k FC layers (out of a total of n layers)

    according to the input index.

    Inputs: x (B, IN_C)  idx (B, K)
    Outputs: (B, K, OUT_C)
  """

  def __init__(self, n, k, in_c, out_c, act_func='relu'):
    super().__init__()

    assert act_func in ['relu', 'sigmoid', 'ssoftplus', 'none']
    if act_func == 'relu':
      gain = math.sqrt(2)
      self.act_func = torch.nn.ReLU(inplace=False)
    elif act_func == 'sigmoid':
      gain = 1.0
      self.act_func = torch.nn.Sigmoid()
    elif act_func == 'ssoftplus':
      gain = math.sqrt(2)
      self.act_func = ShiftedSoftplus()
    else:
      gain = 1.0
      self.act_func = torch.nn.Identity()

    self.n = n
    self.k = k
    self.in_c = in_c
    self.out_c = out_c

    # (N, IN_C, OUT_C)
    self.w = torch.nn.Parameter(
        torch.nn.init.xavier_uniform_(torch.empty(n, in_c, out_c), gain))
    # (N, 1, OUT_C)
    self.b = torch.nn.Parameter(torch.zeros(n, 1, out_c))

  def forward(self, x, idx):
    """Inputs:

        x (B, IN_C) or (B, K, IN_C)
        idx (B, K)

        Outputs:
        (B, K, OUT_C)
    """
    weight = self.w[idx].view(-1, self.in_c, self.out_c)  # (B*K, IN_C, OUT_C)
    bias = self.b[idx].view(-1, 1, self.out_c)  # (B*K, 1, OUT_C)
    if x.ndim == 2:
      x = x.repeat(1, self.k).view(-1, 1, x.size(-1))  # (B*K, 1, IN_C)
    else:
      x = x.view(-1, 1, x.size(-1))
    outputs = torch.bmm(x, weight) + bias  # (B*K, 1, OUT_C)
    return self.act_func(outputs).view(-1, self.k, self.out_c)


class StackedFcFast(torch.nn.Module):
  """Evaluate a batch of samples (x) on k FC layers (out of a total of n layers)

    according to the input index.

    Inputs: x (B, IN_C)  idx (B, K)
    Outputs: (B, K, OUT_C)
  """

  def __init__(self, n, k, in_c, out_c, act_func='relu'):
    super().__init__()

    # assert in_c in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # assert out_c in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # assert k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    assert act_func in ['relu', 'sigmoid', 'ssoftplus', 'none']
    if act_func == 'relu':
      gain = math.sqrt(2)
      self.act_func = torch.nn.ReLU(inplace=False)
    elif act_func == 'sigmoid':
      gain = 1.0
      self.act_func = torch.nn.Sigmoid()
    elif act_func == 'ssoftplus':
      gain = math.sqrt(2)
      self.act_func = ShiftedSoftplus()
    else:
      gain = 1.0
      self.act_func = torch.nn.Identity()

    self.n = n
    self.k = k
    self.in_c = in_c
    self.out_c = out_c

    # (N, IN_C, OUT_C)
    self.w = torch.nn.Parameter(
        torch.nn.init.xavier_uniform_(torch.empty(n, in_c, out_c), gain))
    # (N, 1, OUT_C)
    self.b = torch.nn.Parameter(torch.zeros(n, 1, out_c))

  def forward(self, x, idx):
    # TODO: size checks
    return self.act_func(StackedFcFastFunction.apply(x, idx, self.w, self.b))


if __name__ == '__main__':
  B = 256
  N = 2048
  K = 16
  IN_C = 32
  OUT_C = 32

  net = StackedFcFast(n=N, k=K, in_c=IN_C, out_c=OUT_C, act_func='relu').cuda()
  net2 = StackedFcSlow(n=N, k=K, in_c=IN_C, out_c=OUT_C, act_func='relu').cuda()
  net2.load_state_dict(net.state_dict())
  net3 = StackedFcDense(n=N, in_c=IN_C, out_c=OUT_C, act_func='relu').cuda()
  net3.load_state_dict(net.state_dict())

  for i in range(3):
    print(f'Test {i}:')
    x = torch.randn(B, IN_C).cuda()
    x.requires_grad = True
    idx = torch.randint(N, size=[B, K]).cuda()
    x2 = x.detach().clone()
    x2.requires_grad = True

    # y correct?
    y = net(x, idx)
    y2 = net2(x2, idx)
    print('y correct:', torch.allclose(y, y2, atol=1e-5))
    # print('max diff:', (y - y2).abs().max().item())

    # d correct?
    y.sum().backward()
    y2.sum().backward()
    print('dx correct:', torch.allclose(x.grad, x2.grad, rtol=1e-3, atol=1e-5))
    # print('max diff:', (x.grad - x2.grad).abs().max().item())
    print('dw correct:',
          torch.allclose(net.w.grad, net2.w.grad, rtol=1e-3, atol=1e-5))
    # print('max diff:', (net.w.grad - net2.w.grad).abs().max().item())
    print('db correct:',
          torch.allclose(net.b.grad, net2.b.grad, rtol=1e-3, atol=1e-5))
    # print('max diff:', (net.b.grad - net2.b.grad).abs().max().item())
    print()

  # overall speed
  import time

  x = torch.randn(1000, B, IN_C).cuda()
  x.requires_grad = True
  idx = torch.randint(N, size=[1000, B, K]).cuda()
  x = [x[i].clone().detach() for i in range(x.size(0))]

  b = 0
  a = time.time()
  for i in range(1000):
    xi = x[i]
    xi.requires_grad = True
    y = net3(xi)
    b = y.sum()
    y.sum().backward()
    _ = xi.grad[0]
  print('dense speed (ms):', time.time() - a)

  a = time.time()
  for i in range(1000):
    xi = x[i]
    xi.requires_grad = True
    y = net2(xi, idx[i])
    b = y.sum()
    y.sum().backward()
    _ = xi.grad[0]
  print('baseline speed (ms):', time.time() - a)

  a = time.time()
  for i in range(1000):
    xi = x[i]
    xi.requires_grad = True
    y = net(xi, idx[i])
    b = y.sum()
    y.sum().backward()
    _ = xi.grad[0]
  print('new speed (ms):', time.time() - a)

  import IPython
  IPython.embed()
