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
import scipy
import scipy.stats
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

from aloe.common.pytorch_util import glorot_uniform, MLP


class AutoregSampler(nn.Module):
    def __init__(self, n_choices, discrete_dim, embed_dim):
        super(AutoregSampler, self).__init__()
        self.discrete_dim = discrete_dim
        self.embed_dim = embed_dim
        self.out_pred = MLP(embed_dim, [embed_dim * 2, n_choices])
        self.baseline_pred = MLP(embed_dim, [embed_dim * 2, 1])

    def policy(self, state, true_samples=None):
        cur_log_prob = F.log_softmax(self.out_pred(state), dim=-1)

        if true_samples is None:
            cur_baseline = self.baseline_pred(state)
            cur_prob = torch.exp(cur_log_prob)
            sampled = torch.multinomial(cur_prob, 1)
        else:
            sampled = true_samples
            cur_baseline = None
        cur_log_prob = cur_log_prob.gather(1, sampled)
        return sampled, cur_log_prob, cur_baseline


class RnnSampler(AutoregSampler):
    def __init__(self, n_choices, discrete_dim, embed_dim):
        super(RnnSampler, self).__init__(n_choices, discrete_dim, embed_dim)

        self.init_h = nn.Parameter(torch.Tensor(1, embed_dim))
        self.init_c = nn.Parameter(torch.Tensor(1, embed_dim))
        self.token_embed = nn.Parameter(torch.Tensor(n_choices, embed_dim))
        glorot_uniform(self)
        self.lstm = nn.LSTMCell(embed_dim, embed_dim)

    def forward(self, num_samples=None, input_samples=None):
        assert num_samples is not None or input_samples is not None
        if num_samples is None:
            num_samples = input_samples.shape[0]
        cur_state = (self.init_h.repeat(num_samples, 1), self.init_h.repeat(num_samples, 1))

        if input_samples is not None:
            samples = input_samples.split(1, dim=1)
        else:
            samples = []
            baselines = []
        log_probs = []
        for i in range(self.discrete_dim):
            h, c = cur_state
            if input_samples is None:
                sampled, cur_log_prob, cur_base = self.policy(h)
                samples.append(sampled)
                baselines.append(cur_base)
            else:
                sampled = samples[i]
                _, cur_log_prob, _ = self.policy(h, sampled)
            log_probs.append(cur_log_prob)
            embed_update = self.token_embed[sampled.view(-1)]
            cur_state = self.lstm(embed_update, cur_state)

        log_probs = torch.cat(log_probs, dim=-1)
        if input_samples is None:
            samples = torch.cat(samples, dim=-1)
            baselines = torch.cat(baselines, dim=-1)
            return samples, log_probs, baselines
        else:
            return log_probs


class IidSampler(nn.Module):
    def __init__(self, n_choices, discrete_dim):
        super(IidSampler, self).__init__()
        self.logits = nn.Parameter(torch.Tensor(discrete_dim, n_choices))
        self.baselines = nn.Parameter(torch.Tensor(1, discrete_dim))
        glorot_uniform(self)

    def forward(self, num_samples=None, input_samples=None):
        assert num_samples is not None or input_samples is not None
        if num_samples is None:
            num_samples = input_samples.shape[0]

        log_prob = F.log_softmax(self.logits, dim=-1)
        if input_samples is None:
            out_samples = torch.multinomial(torch.exp(log_prob), num_samples, replacement=True)
        else:
            out_samples = input_samples.t()

        sample_log_prob = log_prob.gather(1, out_samples).t().contiguous()
        out_samples = out_samples.t().contiguous()
        if input_samples is None:
            return out_samples, sample_log_prob, self.baselines.repeat(num_samples, 1)
        else:
            return sample_log_prob


class MLPSampler(AutoregSampler):
    def __init__(self, n_choices, discrete_dim, embed_dim):
        super(MLPSampler, self).__init__(n_choices, discrete_dim, embed_dim)
        self.init_h = nn.Parameter(torch.Tensor(1, embed_dim))
        glorot_uniform(self)

        list_mods = []
        for i in range(1, self.discrete_dim):
            mlp = MLP(i, [embed_dim * 2, embed_dim * 2, embed_dim])
            list_mods.append(mlp)
        self.list_mods = nn.ModuleList(list_mods)

    def forward(self, num_samples=None, input_samples=None):
        assert num_samples is not None or input_samples is not None
        if num_samples is None:
            num_samples = input_samples.shape[0]

        if input_samples is not None:
            samples = input_samples.split(1, dim=1)
        else:
            samples = []
            baselines = []
        log_probs = []
        cur_state = self.init_h.repeat(num_samples, 1)
        for i in range(self.discrete_dim):
            if input_samples is None:
                sampled, cur_log_prob, cur_base = self.policy(cur_state)
                samples.append(sampled)
                baselines.append(cur_base)
            else:
                sampled = samples[i]
                _, cur_log_prob, _ = self.policy(cur_state, sampled)
            log_probs.append(cur_log_prob)
            cur_hist = torch.cat(samples[:i+1], dim=-1)
            if i + 1 < self.discrete_dim:
                cur_state = self.list_mods[i](cur_hist.float())
        log_probs = torch.cat(log_probs, dim=-1)
        if input_samples is None:
            baselines = torch.cat(baselines, dim=-1)
            return cur_hist, log_probs, baselines
        else:
            return log_probs


class BinaryGibbsSampler(nn.Module):
    def __init__(self, discrete_dim, device):
        super(BinaryGibbsSampler, self).__init__()
        self.discrete_dim = discrete_dim
        self.device = device

    def forward(self, score_func, num_rounds, num_samples=None, init_samples=None):
        assert num_samples is not None or init_samples is not None
        if init_samples is None:
            init_samples = torch.randint(2, (num_samples, self.discrete_dim)).to(self.device)

        cur_samples = init_samples.clone()

        for r in range(num_rounds):
            for i in range(self.discrete_dim):
                sample_flip = cur_samples.clone()
                sample_flip[:, i] = 1 - sample_flip[:, i]
                cur_score = score_func(cur_samples)
                new_score = score_func(sample_flip)
                prob_flip = 1.0 / (1.0 + torch.exp(cur_score - new_score))
                xi = torch.rand(prob_flip.shape).to(prob_flip.device)
                flip = (prob_flip > xi).long()
                cur_samples = sample_flip * flip + cur_samples * (1 - flip)

        return cur_samples


def gibbs_step(orig_samples, axis, n_choices, score_func):
    orig_samples = orig_samples.clone()
    with torch.no_grad():
        cur_samples = orig_samples.clone().repeat(n_choices, 1)
        b = torch.LongTensor(list(range(n_choices))).to(cur_samples.device).view(-1, 1)
        b = b.repeat(1, orig_samples.shape[0]).view(-1)
        cur_samples[:, axis] = b
        score = score_func(cur_samples).view(n_choices, -1).transpose(0, 1)

        prob = F.softmax(score, dim=-1)
        samples = torch.multinomial(prob, 1)
        orig_samples[:, axis] = samples.view(-1)
    return orig_samples


class GibbsSampler(nn.Module):
    def __init__(self, n_choices, discrete_dim, device):
        super(GibbsSampler, self).__init__()
        self.n_choices = n_choices
        self.discrete_dim = discrete_dim
        self.device = device

    def forward(self, score_func, num_rounds, num_samples=None, init_samples=None):
        assert num_samples is not None or init_samples is not None
        if init_samples is None:
            init_samples = torch.randint(self.n_choices, (num_samples, self.discrete_dim)).to(self.device)

        with torch.no_grad():
            cur_samples = init_samples.clone()
            for r in range(num_rounds):
                for i in range(self.discrete_dim):
                    cur_samples = gibbs_step(cur_samples, i, self.n_choices, score_func)

        return cur_samples


class UniformBinarySampler(nn.Module):
    def __init__(self, discrete_dim, with_replacement, device):
        super(UniformBinarySampler, self).__init__()
        self.discrete_dim = discrete_dim
        self.device = device
        self.with_replacement = with_replacement

    def forward(self, num_steps, num_samples=None, init_samples=None, target_pos=None):
        assert num_samples is not None or init_samples is not None
        if num_samples is None:
            num_samples = init_samples.shape[0]
        total_log = torch.zeros(num_samples, 1).to(self.device)

        with torch.no_grad():
            if init_samples is None:
                init_samples = torch.randint(2, (num_samples, self.discrete_dim)).to(self.device)
                total_log = total_log - np.log(2) * self.discrete_dim
            else:
                init_samples = init_samples.clone()

            delta = []
            if not self.with_replacement:
                target_pos_list = torch.stack([torch.randperm(self.discrete_dim) for _ in range(num_samples)]).to(self.device)
            else:
                target_pos_list = torch.randint(self.discrete_dim, (num_samples, num_steps)).to(self.device)
            for i in range(num_steps):
                target_pos_i = target_pos_list[:, [i]].to(self.device)
                step_prob = -np.log(self.discrete_dim) if self.with_replacement else -np.log(self.discrete_dim - i)
                total_log = total_log + step_prob
                orig_bit = init_samples.gather(1, target_pos_i).view(-1)
                init_samples[range(num_samples), target_pos_i.view(-1)] = 1 - orig_bit
                delta.append(target_pos_i)
        return init_samples, delta, total_log


class UniMultinomialSampler(nn.Module):
    def __init__(self, n_choices, discrete_dim, device):
        super(UniMultinomialSampler, self).__init__()
        self.discrete_dim = discrete_dim
        self.n_choices = n_choices
        self.device = device

    def get_samples(self, max_steps, stopped_fn, logprob_fn_init, logprob_fn_step, num_samples=None, init_samples=None):
        assert num_samples is not None or init_samples is not None
        if num_samples is None:
            num_samples = init_samples.shape[0]
        total_log = torch.zeros(num_samples, 1).to(self.device)

        with torch.no_grad():
            if init_samples is None:
                init_samples = torch.randint(self.n_choices, (num_samples, self.discrete_dim)).to(self.device)
                total_log = total_log - np.log(self.n_choices) * self.discrete_dim
            else:
                init_samples = init_samples.clone()

            ids = torch.LongTensor(list(range(init_samples.shape[0]))).to(self.device)
            target_pos_list = torch.stack([torch.randperm(self.discrete_dim) for _ in range(ids.shape[0])]).to(self.device)
            init_logprob, init_struct = logprob_fn_init(init_samples)
            total_log = total_log + init_logprob
            for i in range(max_steps):
                stopped = stopped_fn(i, ids, init_struct)
                total_log = logprob_fn_step(total_log, ids, stopped)

                if torch.all(stopped).item():  #
                    break
                ids = ids[~stopped]
                cur_samples = init_samples[ids]
                num_samples = cur_samples.shape[0]
                target_pos_i = target_pos_list[ids][:, [i]].to(self.device)
                total_log[ids] = total_log[ids] - np.log(self.discrete_dim - i)
                orig_bit = cur_samples.gather(1, target_pos_i).view(-1)
                if self.n_choices == 2:
                    cur_samples[range(num_samples), target_pos_i.view(-1)] = 1 - orig_bit
                else:
                    new_opts = torch.randint(self.n_choices - 1, (num_samples,)).to(self.device)
                    cur_samples[range(num_samples), target_pos_i.view(-1)] = (orig_bit + new_opts + 1) % self.n_choices
                    total_log[ids] = total_log[ids] - np.log(self.n_choices - 1)
                init_samples[ids] = cur_samples
        return init_samples, total_log


class GeoUniBinarySampler(UniMultinomialSampler):
    def __init__(self, discrete_dim, stop_prob, device):
        super(GeoUniBinarySampler, self).__init__(2, discrete_dim, device)
        self.stop_prob = stop_prob

    def logprob_fn_step(self, total_log, ids, stopped):
        total_log[ids[stopped]] += np.log(self.stop_prob)
        total_log[ids[~stopped]] += np.log(1 - self.stop_prob)
        return total_log

    def forward(self, max_steps, num_samples=None, init_samples=None):
        stopped_fn = lambda i, ids, y: torch.rand(ids.shape[0]) < self.stop_prob
        logprob_fn_init = lambda t: (0, None)

        return self.get_samples(max_steps, stopped_fn, logprob_fn_init, self.logprob_fn_step, num_samples, init_samples)


class MLPGibbsSampler(nn.Module):
    def __init__(self, base_sampler, discrete_dim, embed_dim):
        super(MLPGibbsSampler, self).__init__()
        self.discrete_dim = discrete_dim
        self.base_sampler = base_sampler
        self.pos_pred = MLP(discrete_dim, [embed_dim * 2, embed_dim * 2, discrete_dim])

    def base_logprob(self, init_samples):
        log_prob = self.base_sampler(input_samples=init_samples)
        return torch.sum(log_prob, dim=-1, keepdim=True)

    def forward_onestep(self, init_samples=None, target_pos=None):
        pos_logit = self.pos_pred(init_samples.float())
        log_pos_prob = F.log_softmax(pos_logit, dim=-1)
        if target_pos is None:
            pos_prob = torch.exp(log_pos_prob)
            target_pos_i = torch.multinomial(pos_prob, 1)
        else:
            target_pos_i = target_pos
        log_target_pos = log_pos_prob.gather(1, target_pos_i)
        return log_target_pos, target_pos_i

    def forward(self, num_steps, mode, num_samples=None, init_samples=None, target_pos=None):
        assert num_samples is not None or init_samples is not None
        total_log = None
        if init_samples is None:
            init_samples, total_log, _ = self.base_sampler(num_samples=num_samples)
        else:
            init_samples = init_samples.clone()
            if mode == 'forward':
                total_log = self.base_sampler(input_samples=init_samples)
        very_init_samples = init_samples.clone()

        total_log = torch.sum(total_log, dim=-1).view(-1, 1) if total_log is not None else 0.0

        delta = []
        for i in range(num_steps):
            if torch.is_grad_enabled():
                init_samples = init_samples.clone()
            cur_target = None if target_pos is None else target_pos[i]
            log_target_pos, target_pos_i = self.forward_onestep(init_samples, cur_target)

            total_log = total_log + log_target_pos

            orig_bit = init_samples.gather(1, target_pos_i).view(-1)
            init_samples[range(init_samples.shape[0]), target_pos_i.view(-1)] = 1 - orig_bit
            delta.append(target_pos_i)
        return init_samples, delta, total_log, very_init_samples


class MLPVarLenSampler(MLPGibbsSampler):
    def __init__(self, base_sampler, discrete_dim, embed_dim, learn_stop, mu0, device):
        super(MLPVarLenSampler, self).__init__(base_sampler, discrete_dim, embed_dim)
        self.device = device
        self.learn_stop = learn_stop
        self.mu0 = mu0
        if self.learn_stop:
            self.stop_pred = MLP(discrete_dim, [embed_dim * 2, embed_dim * 2, 1])

    def pred_stop(self, x, stopped=None):
        if self.learn_stop:
            stop_prob = torch.sigmoid(self.stop_pred(x.float()))
        else:
            stop_prob = torch.zeros(x.shape[0], 1).to(self.device) + self.mu0
        if stopped is None:
            stopped = torch.rand(stop_prob.shape).to(x.device) < stop_prob
        f_stopped = stopped.float()
        log_prob = f_stopped * torch.log(stop_prob + 1e-18) + (1 - f_stopped) * torch.log(1 - stop_prob + 1e-18)
        return log_prob, stopped

    def pred_stop_prob(self, x):
        stop_prob = torch.sigmoid(self.stop_pred(x.float()))
        stopped = torch.rand(stop_prob.shape).to(x.device) < stop_prob
        f_stopped = stopped.float()
        return stop_prob, stopped

    def pos_prob(self, init_samples):
        pos_logit = self.pos_pred(init_samples.float())
        log_pos_prob = F.log_softmax(pos_logit, dim=-1)
        pos_prob = torch.exp(log_pos_prob)
        target_pos_i = torch.multinomial(pos_prob, 1)
        return pos_prob, target_pos_i

    def forward(self, max_steps, num_samples=None, init_samples=None):
        assert not torch.is_grad_enabled()  # only do inference
        assert num_samples is not None or init_samples is not None
        if init_samples is None:
            init_samples, total_log, _ = self.base_sampler(num_samples=num_samples)
        else:
            num_samples = init_samples.shape[0]
            total_log = torch.zeros(init_samples.shape[0], 1).to(self.device)
        very_init_samples = init_samples.clone()
        ids = torch.LongTensor(list(range(init_samples.shape[0]))).to(self.device)
        cur_samples = init_samples
        n_steps = torch.zeros(init_samples.shape[0], 1, dtype=torch.long).to(self.device)
        for i in range(max_steps):
            log_stop_prob, stopped = self.pred_stop(cur_samples)
            stopped = stopped.view(-1)
            total_log[ids] += log_stop_prob
            if torch.all(stopped).item():  #
                break
            ids = ids[~stopped]
            n_steps[ids] += 1
            cur_samples = init_samples[ids]

            log_target_pos, target_pos_i = self.forward_onestep(cur_samples)
            total_log[ids] += log_target_pos

            orig_bit = cur_samples.gather(1, target_pos_i).view(-1)
            cur_samples[range(cur_samples.shape[0]), target_pos_i.view(-1)] = 1 - orig_bit
            init_samples[ids] = cur_samples
        return init_samples, n_steps, total_log, very_init_samples
