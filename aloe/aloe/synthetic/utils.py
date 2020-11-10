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
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from sympy.combinatorics.graycode import GrayCode
import matplotlib.pyplot as plt
from aloe.common.configs import cmd_args
from aloe.common.synthetic.dataset import OnlineToyDataset
from aloe.common.pytorch_util import logsumexp, hamming_mmd, MMD


def compress(x):
    bx = np.binary_repr(int(abs(x)), width=cmd_args.discrete_dim // 2 - 1)
    bx = '0' + bx if x >= 0 else '1' + bx
    return bx


def recover(bx):
    x = int(bx[1:], 2)
    return x if bx[0] == '0' else -x


def float2bin(samples, bm):
    bin_list = []
    for i in range(samples.shape[0]):
        x, y = samples[i] * cmd_args.int_scale
        bx, by = compress(x), compress(y)
        bx, by = bm[bx], bm[by]
        bin_list.append(np.array(list(bx + by), dtype=int))
    return np.array(bin_list)


def bin2float(samples, inv_bm):
    floats = []
    for i in range(samples.shape[0]):
        s = ''
        for j in range(samples.shape[1]):
            s += str(samples[i, j])
        x, y = s[:cmd_args.discrete_dim//2], s[cmd_args.discrete_dim//2:]
        x, y = inv_bm[x], inv_bm[y]
        x, y = recover(x), recover(y)
        x /= cmd_args.int_scale
        y /= cmd_args.int_scale
        floats.append((x, y))
    return np.array(floats)


def setup_data(args):
    bm, inv_bm = get_binmap(args.discrete_dim, args.binmode)
    db = OnlineToyDataset(args.data)
    if args.int_scale is None:
        args.int_scale = db.int_scale
    else:
        db.int_scale = args.int_scale
    if args.plot_size is None:
        args.plot_size = db.f_scale
    else:
        db.f_scale = args.plot_size
    return db, bm, inv_bm


def plot_heat(score_func, bm, out_file=None):
    w = 100
    size = cmd_args.plot_size
    x = np.linspace(-size, size, w)
    y = np.linspace(-size, size, w)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, [-1, 1])
    yy = np.reshape(yy, [-1, 1])
    heat_samples = float2bin(np.concatenate((xx, yy), axis=-1), bm)
    heat_samples = torch.from_numpy(heat_samples).to(cmd_args.device)
    heat_score = F.softmax(score_func(heat_samples).view(1, -1), dim=-1)
    a = heat_score.view(w, w).data.cpu().numpy()
    a = np.flip(a, axis=0)
    plt.imshow(a)
    plt.axis('equal')
    plt.axis('off')
    if out_file is None:
        out_file = os.path.join(cmd_args.save_dir, 'heat.pdf')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


def get_binmap(discrete_dim, binmode):
    b = discrete_dim // 2 - 1
    all_bins = []
    for i in range(1 << b):
        bx = np.binary_repr(i, width=discrete_dim // 2 - 1)
        all_bins.append('0' + bx)
        all_bins.append('1' + bx)
    vals = all_bins[:]
    if binmode == 'rand':
        print('remapping binary repr with random permute')
        random.shuffle(vals)
    elif binmode == 'gray':
        print('remapping binary repr with gray code')
        a = GrayCode(b)
        vals = []
        for x in a.generate_gray():
            vals.append('0' + x)
            vals.append('1' + x)
    else:
        assert binmode == 'normal'
    bm = {}
    inv_bm = {}
    for i, key in enumerate(all_bins):
        bm[key] = vals[i]
        inv_bm[vals[i]] = key
    return bm, inv_bm


def learn_score(samples, score_fn, opt_score, sampler=None, neg_samples=None):
    opt_score.zero_grad()
    if sampler is not None:
        neg_samples = sampler(samples.shape[0])
    else:
        assert neg_samples is not None
    f_loss = -torch.mean(score_fn(samples)) + torch.mean(score_fn(neg_samples))
    f_loss.backward()
    if cmd_args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(score_fn.parameters(), max_norm=cmd_args.grad_clip)
    opt_score.step()
    if cmd_args.weight_clip > 0:
        for p in score_fn.parameters():
            p.data.clamp_(-cmd_args.weight_clip, cmd_args.weight_clip)
    return f_loss.item()


def estimate_ll(score_func, samples, n_partition=None, rand_samples=None):
    with torch.no_grad():
        if rand_samples is None:
            rand_samples = torch.randint(2, (n_partition, samples.shape[1])).to(samples.device)
        n_partition = rand_samples.shape[0]
        f_z_list = []
        for i in range(0, n_partition, samples.shape[0]):
            f_z = score_func(rand_samples[i:i+samples.shape[0]]).view(-1, 1)
            f_z_list.append(f_z)
        f_z = torch.cat(f_z_list, dim=0)
        f_z = f_z - samples.shape[1] * np.log(0.5) - np.log(n_partition)

        log_part = logsumexp(f_z)
        f_sample = score_func(samples)
        ll = f_sample - log_part

    return torch.mean(ll).item()


def estimate_hamming(score_func, true_samples, rand_samples, gibbs_sampler):
    with torch.no_grad():
        gibbs_samples = gibbs_sampler(score_func, 20, init_samples=rand_samples)
        return hamming_mmd(true_samples, gibbs_samples)
