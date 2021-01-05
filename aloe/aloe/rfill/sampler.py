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
import sys
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from functools import partial
from aloe.rfill.utils.rfill_grammar import STATE_TRANS, DECISION_MASK, IDX2STATE, STATE_MAP, RFILL_VOCAB, RFILL_INV_VOCAB
from aloe.rfill.utils.rfill_parser import RFillNode
from aloe.common.pytorch_util import glorot_uniform, MLP
from aloe.common.consts import N_INF, EPS
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence, pad_sequence
from copy import deepcopy
from torch_scatter import scatter_add


class UniformSubseqSampler(nn.Module):
    def __init__(self):
        super(UniformSubseqSampler, self).__init__()
        self.DECISION_MASK = torch.tensor(DECISION_MASK)
        self.STATE_TRANS = torch.LongTensor(STATE_TRANS)

    def forward(self, num, allow_empty=False):
        start_state = STATE_MAP['start']
        expr_state = STATE_MAP['expr'] if allow_empty else STATE_MAP['non_empty_expr']
        halt_state = STATE_MAP['halt']

        cur_states = torch.LongTensor([expr_state] * num)
        final_states = torch.LongTensor([expr_state] * num)
        ids = torch.LongTensor(list(range(num)))
        tokens = [['|'] for _ in range(num)]
        while True:
            cur_mask = self.DECISION_MASK[cur_states]
            cur_prob = cur_mask / (torch.sum(cur_mask, dim=1, keepdim=True) + 1e-18)
            cur_pred = torch.multinomial(cur_prob, 1)

            for i in range(ids.shape[0]):
                cur_tok = RFILL_INV_VOCAB[cur_pred[i].item()]
                if cur_tok != 'eos':
                    tokens[ids[i].item()].append(cur_tok)

            cur_states = self.STATE_TRANS[cur_states.view(-1), cur_pred.view(-1)]

            finished = (cur_states == start_state) | (cur_states == halt_state)
            final_states[ids[finished]] = cur_states[finished]
            ids = ids[~finished]
            cur_states = cur_states[~finished]
            if torch.all(finished).item():
                break
        for i in range(len(tokens)):
            tokens[i].append('|')
        return tokens, final_states


def beam_step(raw_scores, cur_sizes, beam_size):
    pad_size = max(cur_sizes)
    batch_size = len(cur_sizes)
    n_choices = raw_scores.shape[1]
    if pad_size != min(cur_sizes):
        raw_scores = raw_scores.split(cur_sizes, dim=0)
        padded_scores = pad_sequence(raw_scores, batch_first=True, padding_value=N_INF)
    else:
        padded_scores = raw_scores

    padded_scores = padded_scores.view(batch_size, -1)
    topk_scores, candidates = padded_scores.topk(min(beam_size, padded_scores.shape[1]), dim=-1, sorted=True)
    pred_opts = candidates % n_choices
    pos_index = []
    gap = 0
    for i, s in enumerate(cur_sizes):
        pos_index.append(i * pad_size - gap)
        gap += pad_size - s
    pos_index = torch.LongTensor(pos_index).to(padded_scores.device).view(-1, 1)
    predecessors = candidates / n_choices + pos_index.expand_as(candidates)
    valid = topk_scores > EPS + N_INF
    n_valid = valid.sum(dim=-1)
    cur_sizes = n_valid.data.cpu().numpy().tolist()
    predecessors = predecessors[valid]
    pred_opts = pred_opts[valid]
    scores = topk_scores[valid].view(-1, 1)
    return predecessors, pred_opts, scores, cur_sizes


class RfillAutoreg(nn.Module):
    def __init__(self, args):
        super(RfillAutoreg, self).__init__()
        glorot_uniform(self)
        self.DECISION_MASK = torch.tensor(DECISION_MASK).to(args.device)
        self.STATE_TRANS = torch.LongTensor(STATE_TRANS).to(args.device)
        self.cell_type = args.cell_type
        self.vocab = deepcopy(RFILL_VOCAB)
        self.tok_start = self.vocab['|']
        self.tok_stop = self.vocab['eos']
        self.tok_pad = self.vocab['pad']
        assert self.tok_pad == 0
        self.inv_map = {}
        for key in self.vocab:
            self.inv_map[self.vocab[key]] = key
        self.rnn_state_proj = args.rnn_state_proj
        self.rnn_layers = args.rnn_layers
        if self.rnn_state_proj:
            self.ctx2h = MLP(args.embed_dim, [args.embed_dim * self.rnn_layers], nonlinearity=args.act_func, act_last=args.act_func)
            if self.cell_type == 'lstm':
                self.ctx2c = MLP(args.embed_dim, [args.embed_dim * self.rnn_layers], nonlinearity=args.act_func, act_last=args.act_func)
        if args.tok_type == 'embed':
            self.tok_embed = nn.Embedding(len(self.vocab), args.embed_dim)
            input_size = args.embed_dim
        elif args.tok_type == 'onehot':
            input_size = len(self.vocab)
            self.tok_embed = partial(self._get_onehot, vsize=input_size)
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(input_size, args.embed_dim, self.rnn_layers, bidirectional=False)
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(input_size, args.embed_dim, self.rnn_layers, bidirectional=False)
        else:
            raise NotImplementedError
        self.out_pred = nn.Linear(args.embed_dim, len(self.vocab))

    def _get_onehot(self, idx, vsize):
        out_shape = idx.shape + (vsize,)
        idx = idx.view(-1, 1)
        out = torch.zeros(idx.shape[0], vsize).to(idx.device)
        out.scatter_(1, idx, 1)
        return out.view(out_shape)

    def get_init_state(self, state):
        if self.cell_type == 'lstm':
            if not isinstance(state, tuple):
                num_samples = state.shape[0]
                if self.rnn_state_proj:
                    h = self.ctx2h(state).view(num_samples, self.rnn_layers, -1).transpose(0, 1).contiguous()
                    c = self.ctx2c(state).view(num_samples, self.rnn_layers, -1).transpose(0, 1).contiguous()
                    state = (h, c)
                else:
                    _, state = self.rnn(state.unsqueeze(0))
        elif self.cell_type == 'gru':
            if len(state.shape) != 3:
                num_samples = state.shape[0]
                if self.rnn_layers == 1:
                    state = state.unsqueeze(0)
                else:
                    if self.rnn_state_proj:
                        state = self.ctx2h(state).view(num_samples, self.rnn_layers, -1).transpose(0, 1).contiguous()
                    else:
                        _, state = self.rnn(state.unsqueeze(0))
        return state

    def get_likelihood(self, state, expr_list, enforce_sorted, cooked_data=None):
        int_seqs = [self.prog2idx(x).to(state[0].device) for x in expr_list]
        packed_seq = pack_sequence(int_seqs, enforce_sorted=enforce_sorted)
        tok_embed = self.tok_embed(packed_seq.data)
        packed_input = PackedSequence(data=tok_embed, batch_sizes=packed_seq.batch_sizes,
                        sorted_indices=packed_seq.sorted_indices, unsorted_indices=packed_seq.unsorted_indices)

        packed_out, state = self.rnn(packed_input, state)
        unpacked_out, _ = pad_packed_sequence(packed_out)
        out_logits = self.out_pred(unpacked_out)[:-1, :, :].view(-1, len(self.vocab))
        target_seq = pad_packed_sequence(packed_seq)[0][1:, :].view(-1)

        loss = F.cross_entropy(out_logits, target_seq, ignore_index=self.tok_pad, reduction='none').view(-1, len(expr_list))
        ll = -torch.sum(loss, 0).view(-1, 1)
        return ll

    def setup_init_tokens(self, init_states, num_samples, device):
        fsm_state = [STATE_MAP['non_empty_expr']] * num_samples if init_states is None else init_states
        cur_tok = []
        for c in fsm_state:
            cur_tok.append(self.tok_start if c == STATE_MAP['non_empty_expr'] else self.tok_stop)
        fsm_state = torch.LongTensor(fsm_state).to(device)
        tokens = [[self.inv_map[t]] for t in cur_tok]
        cur_tok = torch.LongTensor(cur_tok).to(device).view(-1, 1)
        return fsm_state, cur_tok, tokens

    def beam_search(self, state, beam_size, max_len, cur_sizes=None, init_states=None, init_ll=None):
        state = self.get_init_state(state)
        num_samples = state[0].shape[1] if self.cell_type == 'lstm' else state.shape[1]
        device = state[0].device if self.cell_type == 'lstm' else state.device
        if cur_sizes is None:
            cur_sizes = [1] * num_samples
            batch_size = num_samples
        else:
            batch_size = len(cur_sizes)
        ll = torch.zeros(num_samples, 1).to(device) if init_ll is None else init_ll
        num_cat = torch.LongTensor([0] * num_samples).to(device)

        fsm_state, cur_tok, _ = self.setup_init_tokens(init_states, num_samples, device)
        ones = torch.LongTensor([1] * beam_size * batch_size).to(device)
        all_toks = cur_tok
        ancestors = torch.LongTensor(list(range(num_samples))).to(device)
        while True:
            cur_mask = self.DECISION_MASK[fsm_state]
            cur_embed = self.tok_embed(cur_tok.view(1, -1))
            out, state = self.rnn(cur_embed, state)
            out_logits = self.out_pred(out.squeeze(0))
            out_logits = out_logits * cur_mask + (1 - cur_mask) * N_INF
            out_logprob = F.log_softmax(out_logits, dim=-1)

            # do one step (topk)
            raw_scores = out_logprob + ll
            predecessors, pred_toks, ll, cur_sizes = beam_step(raw_scores, cur_sizes, beam_size)
            ancestors = ancestors[predecessors]
            fsm_state = self.STATE_TRANS[fsm_state[predecessors].view(-1), pred_toks.view(-1)]
            cur_tok = pred_toks
            all_toks = torch.cat([all_toks[predecessors], pred_toks.view(-1, 1)], dim=-1)
            state = (state[0][:, predecessors, :], state[1][:, predecessors, :]) if self.cell_type == 'lstm' else state[:, predecessors, :]
            num_cat = num_cat[predecessors] + (pred_toks == self.tok_start)
            fsm_state[num_cat >= max_len] = STATE_MAP['halt']
            if torch.all(fsm_state == STATE_MAP['halt']).item():
                break
        tokens = []
        all_toks = all_toks.data.cpu().numpy()
        for i in range(all_toks.shape[0]):
            cur_prog = [self.inv_map[j] for j in all_toks[i] if j != self.tok_stop]
            tokens.append(cur_prog)
        return ll, tokens, cur_sizes, ancestors

    def gen_exprs(self, state, gen_method, max_len, cur_sizes, init_states, init_ll):
        if gen_method.startswith('beam'):
            return self.beam_search(state, int(gen_method.split('-')[-1]), max_len, cur_sizes, init_states, init_ll)
        num_samples = state[0].shape[1] if self.cell_type == 'lstm' else state.shape[1]
        device = state[0].device if self.cell_type == 'lstm' else state.device
        fsm_state, cur_tok, tokens = self.setup_init_tokens(init_states, num_samples, device)
        ids = list(range(num_samples))
        num_cat = [0] * num_samples
        ll = init_ll
        while True:
            cur_mask = self.DECISION_MASK[fsm_state]
            cur_embed = self.tok_embed(cur_tok.view(1, -1))
            out, state = self.rnn(cur_embed, state)
            out_logits = self.out_pred(out.squeeze(0))
            out_logits = out_logits * cur_mask + (1 - cur_mask) * N_INF
            out_prob = F.softmax(out_logits, dim=-1)

            if gen_method == 'sample':
                cur_tok = torch.multinomial(out_prob, 1)
            else:
                cur_tok = out_logits.argmax(dim=-1, keepdim=True)
            np_tok = cur_tok.view(-1).data.cpu().numpy()
            has_more = []
            new_ids = []
            for i in range(len(ids)):
                tt = RFILL_INV_VOCAB[np_tok[i]]
                cur_finish = False
                if np_tok[i] != self.tok_stop:
                    tokens[ids[i]].append(tt)
                else:
                    cur_finish = True
                if np_tok[i] == self.tok_start:
                    num_cat[ids[i]] += 1
                    cur_finish |= num_cat[ids[i]] >= max_len
                has_more.append(not cur_finish)
                if not cur_finish:
                    new_ids.append(ids[i])
            step_prob = torch.log(out_prob.gather(1, cur_tok))
            if ll is None:
                ll = step_prob
            else:
                ll[ids] += step_prob
            if not any(has_more):
                break
            fsm_state = self.STATE_TRANS[fsm_state.view(-1), cur_tok.view(-1)]
            cur_tok = cur_tok[has_more]
            fsm_state = fsm_state[has_more]
            ids = new_ids
            state = (state[0][:, has_more, :], state[1][:, has_more, :]) if self.cell_type == 'lstm' else state[:, has_more, :]
        return ll, tokens, [1] * len(tokens), torch.LongTensor(list(range(len(tokens)))).to(device)


class RfillSubexprRnnSampler(RfillAutoreg):
    def __init__(self, args):
        super(RfillSubexprRnnSampler, self).__init__(args)

    def prog2idx(self, expr):
        return torch.LongTensor([self.vocab[c] for c in expr])

    def forward(self, state, expr_list=None, cooked_data=None, gen_method='sample', cur_sizes=None, init_states=None, init_ll=None):
        state = self.get_init_state(state)

        if expr_list is None:
            return self.gen_exprs(state, gen_method, 1, cur_sizes, init_states=init_states, init_ll=init_ll)
        else:
            ll = self.get_likelihood(state, expr_list, enforce_sorted=cooked_data is not None, cooked_data=cooked_data)
            return ll, expr_list


class RfillRnnSampler(RfillAutoreg):
    def __init__(self, args):
        super(RfillRnnSampler, self).__init__(args)
        self.max_len = args.maxNumConcats

    def prog2idx(self, expr):
        return torch.LongTensor([self.vocab[c] for c in expr] + [self.tok_stop])

    def forward(self, state, expr_list=None, cooked_data=None, gen_method='sample', cur_sizes=None, init_states=None, init_ll=None):
        state = self.get_init_state(state)

        if expr_list is None:
            return self.gen_exprs(state, gen_method, self.max_len, cur_sizes, init_states=init_states, init_ll=init_ll)
        else:
            ll = self.get_likelihood(state, expr_list, enforce_sorted=cooked_data is not None, cooked_data=cooked_data)
            return ll, expr_list
