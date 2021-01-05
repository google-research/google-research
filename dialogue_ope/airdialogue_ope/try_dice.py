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

import numpy as np
import random
import torch
"""time:   1    2    3    4    5    Reward state:  0--a-1--b-2-c--3--d-4    1

         |-e-5--f-6              0
        7--g-8--h-9              1
"""

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Parameters
tgt_prob = 0.0
n = 100  # samples
gamma = 1
alphaR = 0
alphaQ = 0
alphaC = 1
learning_rate = 1e-2
normalize_state = False
use_lamb = True
squareC = True

print('Target Policy estimated performance: ', 0.5 + 0.5 * tgt_prob)
print('Target Policy estimated performance (discount): ',
      0.5 * gamma + 0.5 * tgt_prob * gamma**3)


def makedata(n, tgt_prob):
  data = []
  # sample data from perfect policy
  for i in range(n):
    if random.random() < 0.5:
      # start from state 0
      tgt_act1 = 'a' if random.random() < tgt_prob else 'e'
      data.append({
          'states': ['<eod>', 0, 1, 2, 3],
          'acts': ['<eod>', 'a', 'b', 'c', 'd'],
          'tgt_acts': ['<eod>', tgt_act1, 'b', 'c', 'd'],
          'reward': 1,
      })
    else:
      # start from state 7
      data.append({
          'states': ['<eod>', 7, 8, '<eod2>', '<eod1>'],
          'acts': ['<eod>', 'g', 'h', '<eod2>', '<eod1>'],
          'tgt_acts': ['<eod>', 'g', 'h', '<eod2>', '<eod1>'],
          'reward': 1,
      })
  # sample data from worst policy
  for i in range(n):
    if random.random() < 0.5:
      # start from state 0
      tgt_act1 = 'a' if random.random() < tgt_prob else 'e'
      data.append({
          'states': ['<eod>', 0, 5, '<eod2>', '<eod1>'],
          'acts': ['<eod>', 'e', 'f', '<eod2>', '<eod1>'],
          'tgt_acts': ['<eod>', tgt_act1, 'f', '<eod2>', '<eod1>'],
          'reward': 0,
      })
    else:
      # start from state 7
      data.append({
          'states': ['<eod>', 7, 8, '<eod2>', '<eod1>'],
          'acts': ['<eod>', 'g', 'h', '<eod2>', '<eod1>'],
          'tgt_acts': ['<eod>', 'g', 'h', '<eod2>', '<eod1>'],
          'reward': 1,
      })
  return data


data = makedata(n=n, tgt_prob=tgt_prob)
Q_val = {}
C_val = {}
for s in list(range(10)) + ['<eod>', '<eod1>', '<eod2>']:
  if s in [4, 6, 9]:
    continue
  Q_val[s] = {}
  C_val[s] = {}
for s, a in [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (7, 'g'), (8, 'h'),
             (0, 'e'), (5, 'f'), (7, 'g'), (8, 'h'), ('<eod>', '<eod>'),
             ('<eod1>', '<eod1>'), ('<eod2>', '<eod2>')]:
  Q_val[s][a] = torch.randn(1, requires_grad=True)
  C_val[s][a] = torch.randn(1, requires_grad=True)

if use_lamb:
  lamb = torch.randn(1, requires_grad=True)
else:
  lamb = torch.zeros(1, requires_grad=False)

optimizer_Q = torch.optim.SGD(
    [Q_val[k][k2] for k in Q_val for k2 in Q_val[k]] +
    ([lamb] if use_lamb else []),
    lr=learning_rate,
    momentum=0.5)
optimizer_C = torch.optim.SGD([C_val[k][k2] for k in C_val for k2 in C_val[k]],
                              lr=learning_rate,
                              momentum=0.9)

optimizer_Q.zero_grad()
optimizer_C.zero_grad()


def Q_fun(s, a):
  return Q_val[s][a]


class GradientScale(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x, scale):
    ctx.scale = scale
    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    scale = ctx.scale
    return scale * grad_output, None


def grad_reverse(x, scale=1.0):
  return GradientScale.apply(x, -scale)


def C_fun(s, a):
  c = grad_reverse(C_val[s][a])
  if squareC:
    c = c * c
  return c


import tqdm

for i in tqdm.tqdm(range(20000)):
  loss = 0
  reward = 0
  reward_q = 0
  norm_reg = 0
  for d in data:
    _loss = Q_fun(d['states'][0], d['tgt_acts'][0]) * (1 - gamma)
    reward_q += Q_fun(d['states'][0], d['tgt_acts'][0]).item() / len(data)
    num_states = len(d['states']) if normalize_state else 1
    c_reward = None
    for si in range(len(d['states']) - 1):
      s = d['states'][si]
      a = d['acts'][si]
      s_ = d['states'][si + 1]
      a_ = d['tgt_acts'][si + 1]
      if '<eod' in str(s_) and '<eod' not in str(s):
        r = d['reward']
        c_reward = C_fun(s, a)
      else:
        r = 0
      _loss += C_fun(s,
                     a) * (r + gamma * Q_fun(s_, a_) - Q_fun(s, a)) / num_states
      _loss += alphaQ * (Q_fun(s, a)**2) / num_states
      _loss -= alphaC * (C_fun(s, a)**2) / num_states
    s = d['states'][-1]
    a = d['acts'][-1]
    if '<eod' in str(s):
      r = 0
    else:
      r = d['reward']
      c_reward = C_fun(s, a)
    _loss += C_fun(
        s, a) * (r + gamma * Q_fun('<eod>', '<eod>') - Q_fun(s, a)) / num_states
    _loss += alphaQ * (Q_fun(s, a)**2) / num_states
    _loss -= alphaC * (C_fun(s, a)**2) / num_states

    _norm_reg = 0
    for si in range(len(d['states'])):
      s = d['states'][si]
      a = d['acts'][si]
      _norm_reg += 1 - C_fun(s, a)
    _norm_reg = _norm_reg / num_states

    loss += _loss / len(data) + _norm_reg * lamb / len(data)
    reward += c_reward.item() * d['reward'] / len(data)
    norm_reg += _norm_reg.item() / len(data)
  if i % 2 == 0:
    optimizer_Q.zero_grad()
    loss.backward()
    optimizer_Q.step()
  else:
    optimizer_C.zero_grad()
    loss.backward()
    optimizer_C.step()
  if i % 100 == 0:
    print(i, '-step loss: ', loss.item(), ' reward: ', reward,
          ' reward_q: ', (1 - gamma) * reward_q + lamb.item(), ' lamb: ',
          lamb.item(), ' ({})'.format(lamb.grad.item()), ' norm_reg: ',
          norm_reg)
    print('Q values:')
    for k in Q_val:
      print(k, end=' state:  ')

      def g(v):
        grad = Q_val[k][v].grad
        grad = grad.item() if grad is not None else 0
        return grad

      print([(v, Q_val[k][v].item(), g(v)) for v in Q_val[k]])
    print(C_fun(8, 'h').item() - C_fun(7, 'g').item())
    print('C values:')
    for k in C_val:
      print(k, end=' state:  ')

      def g(v):
        grad = C_val[k][v].grad
        grad = grad.item() if grad is not None else 0
        return grad

      print([(v, C_fun(k, v).item(), g(v)) for v in C_val[k]])
