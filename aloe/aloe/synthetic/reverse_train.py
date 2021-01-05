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
from aloe.common.sampler import GibbsSampler, RnnSampler, MLPSampler, IidSampler, MLPVarLenSampler, GeoUniBinarySampler
from tqdm import tqdm
from aloe.synthetic.utils import learn_score, get_binmap, float2bin, bin2float, plot_heat, setup_data, estimate_ll, estimate_hamming
from torch_scatter import scatter_add


def get_base_sampler_and_f(args):
    score_func = MLPScore(args.discrete_dim, [args.embed_dim] * 3 + [1]).to(args.device)

    if args.base_type == 'mlp':
        print('using mlp autoreg sampler')
        base_sampler = MLPSampler(2, args.discrete_dim, args.embed_dim)
    elif args.base_type == 'rnn':
        print('using rnn autoreg sampler')
        base_sampler = RnnSampler(2, args.discrete_dim, args.embed_dim)
    elif args.base_type == 'iid':
        print('using iid sampler')
        base_sampler = IidSampler(2, args.discrete_dim)
    else:
        raise NotImplementedError
    base_sampler = base_sampler.to(args.device)
    return base_sampler, score_func


def get_samples(fn_sample_init, gibbs_sampler, score_func, proposal_opt):
    with torch.no_grad():
        init_samples, cur_steps, x0 = fn_sample_init(cmd_args.num_q_steps, cmd_args.batch_size)
        neg_samples = gibbs_sampler(score_func, cmd_args.gibbs_rounds, init_samples=init_samples)
    if proposal_opt is not None:
        proposal_opt(init_samples, cur_steps)
    avg_steps = torch.mean(cur_steps.float()).item()
    return neg_samples, avg_steps


def plot_sampler(n_step, fn_sample_init, inv_bm, fname):
    samples = []
    avg_steps = 0
    for _ in tqdm(range(100)):
        with torch.no_grad():
            init_samples, cur_steps, _  = fn_sample_init(n_step, cmd_args.batch_size)
            avg_steps += torch.mean(cur_steps.float()).item()
        samples.append(bin2float(init_samples.data.cpu().numpy(), inv_bm))
    samples = np.concatenate(samples, axis=0)
    plot_samples(samples, fname, lim=4.1)
    print('avg edits: %.2f' % (avg_steps / 100))


def prepare_diff_pos(diff_pos, need_target_vals=False):
    diff_pos = diff_pos.data.cpu().numpy()
    list_pos = []
    prev_row = -1
    row_ids = []
    col_ids = []
    col_targets = []
    traj_lens = []
    row_index = 0
    target_val_rows = []
    target_val_cols = []
    for i in range(diff_pos.shape[0]):
        if diff_pos[i, 0] != prev_row or i + 1 == diff_pos.shape[0]:
            if i + 1 == diff_pos.shape[0]:
                list_pos.append(diff_pos[i, 1])
            if len(list_pos):
                if cmd_args.shuffle_edit:
                    random.shuffle(list_pos)
                traj_lens.append(len(list_pos))
                for j in range(len(list_pos)):
                    if j:
                        row_ids += [row_index] * j
                        col_ids += list_pos[:j]
                    if need_target_vals:
                        target_val_rows.append(prev_row)
                        target_val_cols.append(list_pos[j])
                    row_index += 1
                col_targets += list_pos
            list_pos = []
        prev_row = diff_pos[i, 0]
        list_pos.append(diff_pos[i, 1])
    if need_target_vals:
        return row_ids, col_ids, col_targets, traj_lens, target_val_rows, target_val_cols
    else:
        return row_ids, col_ids, col_targets, traj_lens


def main_loop(db, bm, inv_bm, score_func, sampler, proposal_dist, fn_sample_init, fn_log_prob, proposal_opt=None):
    gibbs_sampler = GibbsSampler(2, cmd_args.discrete_dim, cmd_args.device)

    rand_samples = torch.randint(2, (1000, cmd_args.discrete_dim)).to(cmd_args.device)

    if cmd_args.energy_model_dump is not None:
        print('loading score_func from', cmd_args.energy_model_dump)
        score_func.load_state_dict(torch.load(cmd_args.energy_model_dump, map_location=cmd_args.device))
    if cmd_args.sampler_model_dump is not None:
        print('loading sampler from', cmd_args.sampler_model_dump)
        sampler.load_state_dict(torch.load(cmd_args.sampler_model_dump, map_location=cmd_args.device))

    samples = float2bin(db.gen_batch(1000), bm)
    samples = torch.from_numpy(samples).to(cmd_args.device)
    print('true score: %.4f' % torch.mean(score_func(samples)).item())

    opt_score = optim.Adam(score_func.parameters(), lr=cmd_args.learning_rate * cmd_args.f_lr_scale)
    opt_sampler = optim.Adam(sampler.parameters(), lr=cmd_args.learning_rate)

    if cmd_args.phase == 'plot':
        cmd_args.plot_size = 4.1
        plot_heat(score_func, bm, out_file=os.path.join(cmd_args.save_dir, '%s-heat.pdf' % cmd_args.data))
        plot_sampler(0, fn_sample_init, inv_bm, os.path.join(cmd_args.save_dir, '%s-init.pdf' % cmd_args.data))
        plot_sampler(cmd_args.num_q_steps, fn_sample_init, inv_bm, os.path.join(cmd_args.save_dir, '%s-edit.pdf' % cmd_args.data))
        sys.exit()

    for epoch in range(cmd_args.num_epochs):
        pbar = tqdm(range(cmd_args.iter_per_epoch))
        for it in pbar:
            samples = float2bin(db.gen_batch(cmd_args.batch_size), bm)
            samples = torch.from_numpy(samples).to(cmd_args.device)

            if cmd_args.learn_mode == 'ebm':
                neg_samples, avg_steps = get_samples(fn_sample_init, gibbs_sampler, score_func, proposal_opt)
                f_loss = learn_score(samples, score_func, opt_score, neg_samples=neg_samples)
            else:
                neg_samples = samples
                f_loss = 0.0

            for q_it in range(cmd_args.q_iter):
                opt_sampler.zero_grad()
                with torch.no_grad():
                    neg_samples = neg_samples.repeat(cmd_args.num_importance_samples, 1)
                    cur_samples, proposal_logprob = proposal_dist(cmd_args.num_q_steps, neg_samples)
                    diff_pos = (cur_samples - neg_samples).nonzero()
                    rep_rows = diff_pos[:, 0].view(-1)
                    rep_init = torch.index_select(cur_samples, 0, rep_rows)
                    row_ids, col_ids, col_target, traj_lens = prepare_diff_pos(diff_pos)
                    rep_init[row_ids, col_ids] = 1 - rep_init[row_ids, col_ids]
                    rep_target = torch.LongTensor(col_target).to(cmd_args.device).view(-1, 1)

                init_prob = sampler.base_logprob(cur_samples)
                if rep_rows.shape[0]:
                    traj_prob, _ = sampler.forward_onestep(init_samples=rep_init, target_pos=rep_target)
                    traj_prob = scatter_add(traj_prob, rep_rows, dim=0, dim_size=cur_samples.shape[0])
                else:
                    traj_prob = 0
                log_prob = fn_log_prob(init_prob, traj_prob, rep_init, rep_rows, neg_samples)  #TODO: shouldn't be neg_samples?

                if cmd_args.lb_type == 'is':  # calc weights using self-normalization
                    with torch.no_grad():
                        log_ratio = (log_prob - proposal_logprob).view(cmd_args.num_importance_samples, -1)
                        weight = F.softmax(log_ratio, dim=0).view(log_prob.shape)
                else:
                    weight = 1.0 / cmd_args.num_importance_samples
                log_prob = log_prob * weight

                loss = -torch.mean(log_prob) * cmd_args.num_importance_samples
                loss.backward()
                if cmd_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(sampler.parameters(), max_norm=cmd_args.grad_clip)
                opt_sampler.step()

                if q_it + 1 < cmd_args.q_iter:
                    neg_samples, _ = get_samples(fn_sample_init, gibbs_sampler, score_func, proposal_opt=None)

            g_loss = loss.item()
            true_score = torch.mean(score_func(samples)).item()
            fake_score = torch.mean(score_func(neg_samples)).item()
            rand_score = torch.mean(score_func(rand_samples)).item()
            pbar.set_description('epoch: %d, f: %.2f, g: %.2f, n: %.2f, true: %.2f, fake: %.2f, rand: %.2f' % (epoch, f_loss, g_loss, avg_steps, true_score, fake_score, rand_score))

        if epoch and epoch % cmd_args.epoch_save == 0:
            torch.save(score_func.state_dict(), os.path.join(cmd_args.save_dir, 'score_func-%d.ckpt' % epoch))
            torch.save(sampler.state_dict(), os.path.join(cmd_args.save_dir, 'sampler-%d.ckpt' % epoch))

            plot_heat(score_func, bm, out_file=os.path.join(cmd_args.save_dir, 'heat-%d.pdf' % epoch))
            for n_step in [0, cmd_args.num_q_steps]:
                plot_sampler(n_step, fn_sample_init, inv_bm, os.path.join(cmd_args.save_dir, 'rand-samples-%d-%d.pdf' % (epoch, n_step)))

