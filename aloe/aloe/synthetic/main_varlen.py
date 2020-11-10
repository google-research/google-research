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

# pylint: skip-file
import numpy as np
import os
import sys
import random
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import chain

from aloe.common.configs import cmd_args, set_device
from aloe.common.plot_2d import plot_samples
from aloe.common.f_family import MLPScore
from aloe.common.sampler import GibbsSampler, MLPVarLenSampler, GeoUniBinarySampler
from tqdm import tqdm
from aloe.synthetic.utils import learn_score, get_binmap, float2bin, bin2float, plot_heat, setup_data
from aloe.synthetic.reverse_train import main_loop, get_base_sampler_and_f
from torch_scatter import scatter_add


def calc_log_prob(init_prob, traj_prob, rep_init, rep_rows, final_samples):
    if rep_rows.shape[0]:
        zeros = torch.zeros(rep_init.shape[0], 1).to(init_prob.device)
        nonstop = sampler.pred_stop(rep_init, zeros)[0]
        nonstop = scatter_add(nonstop, rep_rows, dim=0, dim_size=final_samples.shape[0])
    else:
        nonstop = 0
    ones = torch.ones(final_samples.shape[0], 1).to(init_prob.device)
    last_stop = sampler.pred_stop(final_samples, ones)[0]

    return init_prob + traj_prob + nonstop + last_stop

def fn_sample_init(ns, num):
    samples, nsteps, _, x0 = sampler(ns, num_samples=num)
    return samples, nsteps, x0

def proposal_optimizer(optimizer, sampler, x0, n_steps):
    optimizer.zero_grad()
    logits = sampler.t_pred(x0.float())
    loss = F.cross_entropy(logits, n_steps.view(-1))
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    db, bm, inv_bm = setup_data(cmd_args)

    base_sampler, score_func = get_base_sampler_and_f(cmd_args)
    sampler = MLPVarLenSampler(base_sampler, cmd_args.discrete_dim,
                               embed_dim=cmd_args.embed_dim,
                               learn_stop=cmd_args.learn_stop,
                               mu0=cmd_args.mu0,
                               device=cmd_args.device).to(cmd_args.device)

    uni_sampler = GeoUniBinarySampler(cmd_args.discrete_dim, cmd_args.mu0, cmd_args.device)
    proposal_dist = lambda n, s: uni_sampler(n, init_samples=s)
    proposal_opt = None

    main_loop(db, bm, inv_bm, score_func, sampler, proposal_dist, fn_sample_init, calc_log_prob, proposal_opt)
