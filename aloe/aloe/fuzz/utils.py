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
import os
import pickle as cp
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset
from aloe.common.pytorch_util import PosEncoding, MLP, glorot_uniform
from aloe.common.sampler import UniMultinomialSampler


def to_hex(np_arr, length=None):
    if length is None:
        length = len(np_arr)
    hex_list = []
    for j in range(length):
        x = hex(np_arr[j])[2:]
        if len(x) < 2:
            x = '0' + x
        hex_list.append(x)
    return ''.join(hex_list)


class HexStreamDataset(Dataset):
    def __init__(self, args):
        print('loading chunked hex stream from', args.data_dir)
        self.raw_hex_stream = np.load(os.path.join(args.data_dir, 'hex_stream.npy'))
        self.common_prefix = np.load(os.path.join(args.data_dir, 'prefix.npy'))
        self.common_suffix = np.load(os.path.join(args.data_dir, 'suffix.npy'))

        self.dim = self.raw_hex_stream.shape[1] - 1
        assert self.dim == args.window_size
        print('# samples', len(self))
        with open(os.path.join(args.data_dir, 'raw.pkl'), 'rb') as f:
            self.list_raw = cp.load(f)
        print('# raw png files', len(self.list_raw))

    def rand_len(self):
        return np.random.choice(self.list_raw).shape[0]

    def __len__(self):
        return self.raw_hex_stream.shape[0]

    def __getitem__(self, index):
        row = self.raw_hex_stream[index]
        pos = row[0]
        hex_data = torch.LongTensor(row[1:])
        pos_list = list(range(pos, pos + self.dim))
        pos_list = torch.LongTensor(pos_list)
        return pos_list.view(1, -1), hex_data.view(1, -1)

    def collate_fn(self, list_samples):
        list_pos, list_hex = zip(*list_samples)
        pos = torch.cat(list_pos, dim=0)
        hex_data = torch.cat(list_hex, dim=0)
        return pos, hex_data


class CondMLPScore(nn.Module):
    def __init__(self, n_choices, discrete_dim, embed_dim, act_last, f_scale):
        super(CondMLPScore, self).__init__()
        self.discrete_dim = discrete_dim
        tok_dim = 8
        self.input_tok_embed = nn.Embedding(n_choices, tok_dim)
        self.pos_encode = PosEncoding(tok_dim)
        self.f_scale = f_scale
        self.mlp = MLP(self.discrete_dim * tok_dim, [embed_dim * 2] * 3 + [1], act_last=act_last)

    def forward(self, pos_list, samples):
        bsize = pos_list.shape[0]
        if samples.shape[0] != bsize:
            assert samples.shape[0] % bsize == 0
            pos_list = pos_list.repeat(samples.shape[0] // bsize, 1)
            bsize = samples.shape[0]
        pos_embed = self.pos_encode(pos_list).view(bsize, -1)
        tok_embed = self.input_tok_embed(samples).view(bsize, -1)

        input_embed = pos_embed + tok_embed
        return self.mlp(input_embed) * self.f_scale


class CondRnnScore(nn.Module):
    def __init__(self, args, n_choices, act_last, f_scale):
        super(CondRnnScore, self).__init__()
        self.pos_encode = PosEncoding(args.embed_dim)
        self.input_tok_embed = nn.Embedding(n_choices, args.embed_dim)
        self.lstm = nn.LSTM(args.embed_dim, args.embed_dim, args.rnn_layers, bidirectional=True, batch_first=True)
        self.f_scale = f_scale
        self.mlp = MLP(2 * args.embed_dim, [args.embed_dim * 2] * 2 + [1], act_last=act_last)

    def forward(self, pos_list, samples):
        bsize = pos_list.shape[0]
        if samples.shape[0] != bsize:
            assert samples.shape[0] % bsize == 0
            pos_list = pos_list.repeat(samples.shape[0] // bsize, 1)
            bsize = samples.shape[0]
        pos_embed = self.pos_encode(pos_list)
        tok_embed = self.input_tok_embed(samples)
        input_embed = pos_embed + tok_embed

        embed_out, state = self.lstm(input_embed)
        embed, _ = torch.max(embed_out, dim=1)
        score = self.mlp(embed) * self.f_scale
        return score


class CondAutoregSampler(nn.Module):
    def __init__(self, n_choices, discrete_dim, embed_dim):
        super(CondAutoregSampler, self).__init__()
        self.discrete_dim = discrete_dim
        self.embed_dim = embed_dim
        self.out_pred = MLP(embed_dim, [embed_dim * 2, n_choices])
        self.pos_encode = PosEncoding(embed_dim)

    def one_step(self, state, true_samples=None):
        cur_log_prob = F.log_softmax(self.out_pred(state), dim=-1)
        if true_samples is None:
            cur_prob = torch.exp(cur_log_prob)
            sampled = torch.multinomial(cur_prob, 1)
        else:
            sampled = true_samples
        cur_log_prob = cur_log_prob.gather(1, sampled)
        return sampled, cur_log_prob


class CondRnnSampler(CondAutoregSampler):
    def __init__(self, n_choices, discrete_dim, embed_dim):
        super(CondRnnSampler, self).__init__(n_choices, discrete_dim, embed_dim)

        self.token_embed = nn.Parameter(torch.Tensor(n_choices, embed_dim))
        glorot_uniform(self)
        self.lstm = nn.LSTMCell(embed_dim, embed_dim)

    def forward(self, pos_list, input_samples=None):
        assert pos_list.shape[1] == self.discrete_dim
        num_samples = pos_list.shape[0]

        pos_embedding = self.pos_encode(pos_list).split(1, dim=1)
        cur_state = self.lstm(pos_embedding[0].squeeze(1))
        if input_samples is not None:
            samples = input_samples.split(1, dim=1)
        else:
            samples = []
        log_probs = []
        for i in range(self.discrete_dim):
            h, c = cur_state
            if input_samples is None:
                sampled, cur_log_prob = self.one_step(h)
                samples.append(sampled)
            else:
                sampled = samples[i]
                _, cur_log_prob = self.one_step(h, sampled)
            log_probs.append(cur_log_prob)
            embed_update = self.token_embed[sampled.view(-1)] + pos_embedding[i].squeeze(1)
            cur_state = self.lstm(embed_update, cur_state)
        log_probs = torch.cat(log_probs, dim=-1)
        log_probs = torch.sum(log_probs, dim=1, keepdim=True)
        if input_samples is None:
            samples = torch.cat(samples, dim=-1)
            return samples, log_probs
        else:
            return log_probs


class VarlenMultinomialSampler(nn.Module):
    def __init__(self, base_sampler, discrete_dim, n_choices, embed_dim):
        super(VarlenMultinomialSampler, self).__init__()
        self.discrete_dim = discrete_dim
        self.n_choices = n_choices
        self.base_sampler = base_sampler
        ctx_dim = self.get_context_dim()

        self.pos_pred = MLP(ctx_dim, [embed_dim * 2, embed_dim * 2, discrete_dim])
        self.val_pred = MLP(ctx_dim + embed_dim, [embed_dim * 2] * 2 + [n_choices])
        self.stop_pred = MLP(ctx_dim, [embed_dim * 2, embed_dim * 2, 1])

        self.mod_pos_embed = nn.Embedding(discrete_dim, embed_dim)

    def base_logprob(self, pos_list, init_samples):
        log_prob = self.base_sampler(pos_list, init_samples)
        return log_prob

    def get_context_dim(self):
        raise NotImplementedError

    def get_all_pos_encode(self, pos_list):
        raise NotImplementedError

    def get_context(self, all_poses, cur_samples):
        raise NotImplementedError

    def get_context_from_raw(self, pos_list, cur_samples):
        return self.get_context(self.get_all_pos_encode(pos_list), cur_samples)

    def _get_multinomial(self, logits, target=None, cur_val=None):
        if cur_val is not None and target is None:
            logits[range(logits.shape[0]), cur_val.view(-1)] = -1000000
        log_prob = F.log_softmax(logits, dim=-1)
        if target is None:
            prob = torch.exp(log_prob)
            target_i = torch.multinomial(prob, 1)
        else:
            target_i = target.view(-1, 1)

        log_target = log_prob.gather(1, target_i)
        return log_target, target_i

    def forward_onestep(self, cur_context, cur_val=None, target_pos=None, target_bits=None):
        pos_logit = self.pos_pred(cur_context)
        log_target_pos, target_pos_i = self._get_multinomial(pos_logit, target_pos)

        mod_pos = self.mod_pos_embed(target_pos_i.view(-1))
        merged_ctx = torch.cat((cur_context, mod_pos), dim=-1)
        bit_logit = self.val_pred(merged_ctx)

        if cur_val is not None:
            cur_val = cur_val.gather(1, target_pos_i)
        log_target_bits, target_bits_i = self._get_multinomial(bit_logit, target_bits, cur_val=cur_val)
        log_prob = log_target_pos + log_target_bits
        return log_prob, target_pos_i, target_bits_i

    def pred_stop(self, x, stopped=None):
        stop_prob = torch.sigmoid(self.stop_pred(x))
        if stopped is None:
            stopped = torch.rand(stop_prob.shape).to(x.device) < stop_prob
        f_stopped = stopped.float()
        log_prob = f_stopped * torch.log(stop_prob + 1e-18) + (1 - f_stopped) * torch.log(1 - stop_prob + 1e-18)
        return log_prob, stopped

    def forward(self, max_steps, pos_list, init_samples=None):
        assert not torch.is_grad_enabled()  # only do inference, so that we can do inplace operation
        assert pos_list.shape[1] == self.discrete_dim
        num_samples = pos_list.shape[0]
        device = pos_list.device
        if init_samples is None:
            init_samples, total_log = self.base_sampler(pos_list)
        else:
            total_log = torch.zeros(init_samples.shape[0], 1).to(device)

        very_init_samples = init_samples.clone()
        ids = torch.LongTensor(list(range(init_samples.shape[0]))).to(device)
        n_steps = torch.zeros(init_samples.shape[0], 1, dtype=torch.long).to(device)
        all_poses = self.get_all_pos_encode(pos_list)

        cur_samples = init_samples
        cur_all_poses = all_poses
        for i in range(max_steps):
            cur_context = self.get_context(cur_all_poses, cur_samples)
            log_stop_prob, stopped = self.pred_stop(cur_context)
            stopped = stopped.view(-1)
            total_log[ids] += log_stop_prob
            if torch.all(stopped).item():
                break
            ids = ids[~stopped]
            cur_context = cur_context[~stopped]
            n_steps[ids] += 1
            cur_samples = init_samples[ids]
            cur_all_poses = all_poses[ids]

            log_step_prob, target_pos_i, target_bit_i = self.forward_onestep(cur_context, cur_val=cur_samples)
            total_log[ids] += log_step_prob

            cur_samples[range(cur_samples.shape[0]), target_pos_i.view(-1)] = target_bit_i.view(-1)
            init_samples[ids] = cur_samples
        return init_samples, n_steps, total_log, very_init_samples


class MLPVarLenMultinomialSampler(VarlenMultinomialSampler):
    def __init__(self, base_sampler, discrete_dim, n_choices, embed_dim):
        super(MLPVarLenMultinomialSampler, self).__init__(base_sampler, discrete_dim, n_choices, embed_dim)
        self.input_tok_embed = nn.Embedding(n_choices, 4)
        self.pos_encode = PosEncoding(4)
        self.input_encode = MLP(self.discrete_dim * 4, [embed_dim * 2] + [embed_dim])

    def get_context_dim(self):
        return self.discrete_dim * 4

    def get_all_pos_encode(self, pos_list):
        return self.pos_encode(pos_list)

    def get_context(self, all_poses, cur_samples):
        sample_embed = self.input_tok_embed(cur_samples)
        ctx = all_poses + sample_embed
        ctx = ctx.view(all_poses.shape[0], -1)
        return ctx


class GeoMultinomialSampler(UniMultinomialSampler):
    def __init__(self, n_choices, discrete_dim, stop_prob, device):
        super(GeoMultinomialSampler, self).__init__(n_choices, discrete_dim, device)
        self.stop_prob = stop_prob

    def logprob_fn_step(self, total_log, ids, stopped):
        total_log[ids[stopped]] += np.log(self.stop_prob)
        total_log[ids[~stopped]] += np.log(1 - self.stop_prob)
        return total_log

    def forward(self, max_steps, num_samples=None, init_samples=None):
        stopped_fn = lambda i, ids, y: torch.rand(ids.shape[0]) < self.stop_prob
        logprob_fn_init = lambda t: (0, None)

        return self.get_samples(max_steps, stopped_fn, logprob_fn_init, self.logprob_fn_step, num_samples, init_samples)


