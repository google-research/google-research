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

# pylint: skip-file
import numpy as np
import os
import sys
import random
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm


from aloe.common.configs import cmd_args, set_device
from aloe.common.f_family import MLPScore
from aloe.common.sampler import GibbsSampler
from aloe.synthetic.utils import learn_score
from aloe.synthetic.reverse_train import prepare_diff_pos
from aloe.fuzz.utils import CondRnnSampler, MLPVarLenMultinomialSampler, HexStreamDataset, CondRnnScore, CondMLPScore, GeoMultinomialSampler
from torch_scatter import scatter_add


def learn_score(pos_list, true_samples, score_fn, opt_score, neg_samples):
    opt_score.zero_grad()
    true_scores = score_fn(pos_list, true_samples)
    f_loss = -torch.mean(true_scores) + torch.mean(score_fn(pos_list, neg_samples))
    f_loss.backward()
    if cmd_args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(score_fn.parameters(), max_norm=cmd_args.grad_clip)
    opt_score.step()
    if cmd_args.weight_clip > 0:
        for p in score_fn.parameters():
            p.data.clamp_(-cmd_args.weight_clip, cmd_args.weight_clip)
    return f_loss.item(), torch.mean(true_scores).item()


def get_score_func(args):
    if args.score_func == 'rnn':
        score_func = CondRnnScore(args, n_choices=256, act_last=args.f_out, f_scale=args.f_scale)
    elif args.score_func == 'mlp':
        score_func = CondMLPScore(n_choices=256, discrete_dim=args.window_size, embed_dim=args.embed_dim, act_last=args.f_out, f_scale=args.f_scale)
    else:
        raise NotImplementedError
    score_func = score_func.to(args.device)
    return score_func


def get_editor(args):
    # build base sampler
    if args.base_type == 'mlp':
        print('using mlp autoreg base sampler')
    elif args.base_type == 'rnn':
        print('using rnn autoreg base sampler')
        base_sampler = CondRnnSampler(n_choices=256, discrete_dim=args.window_size, embed_dim=args.embed_dim)
    else:
        raise NotImplementedError
    # build editor
    if args.io_enc == 'rnn':
        pass
    elif args.io_enc == 'mlp':
        sampler = MLPVarLenMultinomialSampler(base_sampler, args.window_size,
                                    n_choices=256,
                                    embed_dim=args.embed_dim).to(args.device)
    else:
        raise NotImplementedError
    return sampler


if __name__ == '__main__':
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    set_device(cmd_args.gpu)

    # build score func
    score_func = get_score_func(cmd_args)
    sampler = get_editor(cmd_args)

    db = HexStreamDataset(cmd_args)
    train_load = DataLoader(db, batch_size=cmd_args.batch_size, shuffle=True,
                            collate_fn=db.collate_fn, num_workers=cmd_args.num_proc, drop_last=True)
    train_gen = iter(train_load)
    gibbs_sampler = GibbsSampler(n_choices=256, discrete_dim=cmd_args.window_size, device=cmd_args.device)
    proposal_dist = GeoMultinomialSampler(n_choices=256, discrete_dim=cmd_args.window_size, stop_prob=cmd_args.mu0, device=cmd_args.device)
    opt_score = optim.Adam(score_func.parameters(), lr=cmd_args.learning_rate * cmd_args.f_lr_scale)
    opt_sampler = optim.Adam(sampler.parameters(), lr=cmd_args.learning_rate)

    rand_samples = torch.randint(256, (cmd_args.batch_size, cmd_args.window_size)).to(cmd_args.device)

    for epoch in range(cmd_args.num_epochs):
        pbar = tqdm(range(cmd_args.iter_per_epoch))
        for it in pbar:
            try:
                samples = next(train_gen)
            except StopIteration:
                train_gen = iter(train_load)
                samples = next(train_gen)
            pos_list, hex_stream = samples
            pos_list = pos_list.to(cmd_args.device)
            hex_stream = hex_stream.to(cmd_args.device)
            # get samples
            with torch.no_grad():
                init_samples, n_steps, _, _  = sampler(cmd_args.num_q_steps, pos_list)
                cur_score_fn = lambda samples: score_func(pos_list, samples)
                neg_samples = gibbs_sampler(cur_score_fn, cmd_args.gibbs_rounds, init_samples=init_samples)

            f_loss, true_scores = learn_score(pos_list, hex_stream, score_func, opt_score, neg_samples)

            with torch.no_grad():
                rand_scores = torch.mean(score_func(pos_list, rand_samples)).item()
                neg_scores = torch.mean(score_func(pos_list, neg_samples)).item()

            neg_samples = neg_samples.repeat(cmd_args.num_importance_samples, 1)
            pos_list = pos_list.repeat(cmd_args.num_importance_samples, 1)
            for q_it in range(cmd_args.q_iter):
                opt_sampler.zero_grad()
                if cmd_args.num_q_steps:
                    with torch.no_grad():
                        cur_samples, proposal_logprob = proposal_dist(cmd_args.num_q_steps, init_samples=neg_samples)
                        diff_pos = (cur_samples - neg_samples).nonzero()
                        rep_rows = diff_pos[:, 0].view(-1)
                        rep_init = torch.index_select(cur_samples, 0, rep_rows)
                        rep_final = torch.index_select(neg_samples, 0, rep_rows)
                        row_ids, col_ids, col_target, traj_lens, tval_rows, tval_cols = prepare_diff_pos(diff_pos, need_target_vals=True)
                        rep_init[row_ids, col_ids] = rep_final[row_ids, col_ids]
                        rep_val = neg_samples[tval_rows, tval_cols]
                        rep_target = torch.LongTensor(col_target).to(cmd_args.device).view(-1, 1)
                else:
                    cur_samples = neg_samples
                    proposal_logprob = 0
                init_prob = sampler.base_logprob(pos_list, cur_samples)
                if cmd_args.num_q_steps and rep_rows.shape[0]:
                    ctx_pos_list = torch.index_select(pos_list, 0, rep_rows)
                    context = sampler.get_context_from_raw(ctx_pos_list, rep_init)
                    traj_prob, _, _ = sampler.forward_onestep(context, target_pos=rep_target, target_bits=rep_val)
                    traj_prob = scatter_add(traj_prob, rep_rows, dim=0, dim_size=cur_samples.shape[0])

                    zeros = torch.zeros(rep_init.shape[0], 1).to(cmd_args.device)
                    nonstop = sampler.pred_stop(context, zeros)[0]
                    nonstop = scatter_add(nonstop, rep_rows, dim=0, dim_size=neg_samples.shape[0])
                else:
                    traj_prob = nonstop = 0

                ones = torch.ones(neg_samples.shape[0], 1).to(cmd_args.device)
                context = sampler.get_context_from_raw(pos_list, neg_samples)
                last_stop = sampler.pred_stop(context, ones)[0]
                log_prob = init_prob + traj_prob + nonstop + last_stop
                with torch.no_grad():
                    log_ratio = (log_prob - proposal_logprob).view(cmd_args.num_importance_samples, -1)
                    weight = F.softmax(log_ratio, dim=0).view(log_prob.shape)

                log_prob = log_prob * weight
                loss = -torch.mean(log_prob) * cmd_args.num_importance_samples
                loss.backward()
                if cmd_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(sampler.parameters(), max_norm=cmd_args.grad_clip)
                opt_sampler.step()
            g_loss = loss.item()

            avg_steps = torch.mean(n_steps.float()).item()
            pbar.set_description('epoch: %d, f: %.2f, g: %.2f, n: %.2f, true: %.2f, neg: %.2f, rand: %.2f' % (epoch, f_loss, g_loss, avg_steps, true_scores, neg_scores, rand_scores))

        if epoch % cmd_args.epoch_save == 0:
            save_dict = {
                'epoch': epoch,
                'score_func': score_func.state_dict(),
                'sampler': sampler.state_dict(),
                'opt_score': opt_score.state_dict(),
                'opt_sampler': opt_sampler.state_dict(),
            }

            torch.save(save_dict, os.path.join(cmd_args.save_dir, 'model-%d.ckpt' % epoch))
