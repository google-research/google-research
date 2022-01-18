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
import random
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence, pack_padded_sequence
from torch_scatter import scatter_add
from copy import deepcopy
from aloe.common.pytorch_util import glorot_uniform, MLP, pad_sequence
from aloe.rfill.utils.rfill_grammar import STATE_TRANS, DECISION_MASK, IDX2STATE, STATE_MAP, RFILL_VOCAB, RFILL_INV_VOCAB
from aloe.rfill.sampler import RfillSubexprRnnSampler, RfillRnnSampler, beam_step
from aloe.rfill.seq_embed import BidirIOEmbed, MLPIOEmbed
from aloe.common.consts import N_INF, EPS


def expr_cmp(e1, s1, t1, e2, s2, t2):
    if t1 - s1 != t2 - s2:
        return 1
    for i in range(t1 - s1):
        if e1[i + s1] != e2[i + s2]:
            return 1
    return 0


def build_sol(i, j, src_expr, src_bars, tgt_expr, tgt_bars, dp, cost, sols):
    if i == 0:
        for k in range(j):
            sols.append((tgt_bars[k], tgt_expr[tgt_bars[k] : tgt_bars[k + 1] + 1]))
        return 0, tgt_bars[j]
    elif j == 0:
        for k in range(i):
            sols.append((2, None))
        return 0, -src_bars[i]
    else:
        c1 = dp[i - 1, j] + 1 # delete current one
        p1 = src_bars[i - 1] + 2
        d1 = (1, 0)
        c2 = dp[i - 1, j - 1] + cost[i, j] # modify current one
        p2 = src_bars[i - 1] + 1
        d2 = (1, 1)
        c3 = dp[i, j - 1] + 1 # insert a new one
        p3 = src_bars[i]
        d3 = (0, 1)
        opts = [t for t in [(c1, p1, d1), (c2, p2, d2), (c3, p3, d3)] if t[0] == dp[i, j]]
        opt = random.choice(opts)
        _, p, d = opt
        log_prob, delta = build_sol(i - d[0], j - d[1], src_expr, src_bars, tgt_expr, tgt_bars, dp, cost, sols)
        log_prob += np.log(1.0 / len(opts))

        if d[0] + d[1] < 2 or cost[i, j]:
            new_sub = tgt_expr[tgt_bars[j - 1] : tgt_bars[j] + 1] if d[1] else None
            sols.append((p + delta, new_sub))
            new_len = 1 if new_sub is None else len(new_sub)
            orig_len = 1 if d[0] == 0 else (src_bars[i] - src_bars[i - 1] + 1)
            delta += new_len - orig_len
        return log_prob, delta


def shortest_rand_edit(src_expr, tgt_expr):
    src_bars = [i for i in range(len(src_expr)) if src_expr[i] == '|']
    tgt_bars = [i for i in range(len(tgt_expr)) if tgt_expr[i] == '|']
    n = len(src_bars)
    m = len(tgt_bars)
    dp = np.zeros((n, m), dtype=np.int32)
    cost = np.zeros((n, m), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            if i == 0:
                dp[i, j] = j  # insert that many subexprs
            elif j == 0:
                dp[i, j] = i  # delete that many subexprs
            else:
                c1 = dp[i - 1, j] + 1 # delete current one
                cost[i, j] = expr_cmp(src_expr, src_bars[i - 1], src_bars[i],
                                      tgt_expr, tgt_bars[j - 1], tgt_bars[j])
                c2 = dp[i - 1, j - 1] + cost[i, j] # modify current one
                c3 = dp[i, j - 1] + 1# insert a new one
                dp[i, j] = min(c1, c2, c3)

    sols = []
    log_prob, _ = build_sol(n - 1, m - 1, src_expr, src_bars, tgt_expr, tgt_bars, dp, cost, sols)
    assert len(sols) == dp[n - 1, m - 1]
    return dp[n - 1, m - 1], log_prob, sols


def perform_edit(orig_expr, edit_pos, edit_patt):
    if orig_expr[edit_pos] == '|':  # insert a new subexpr
        new_expr = orig_expr[:edit_pos] + edit_patt + orig_expr[edit_pos+1:]
    else:
        j = edit_pos
        while orig_expr[j] != '|':
            j += 1
        if orig_expr[edit_pos] == 'SubStr' or orig_expr[edit_pos] == 'ConstStr':  # replace
            new_expr = orig_expr[:edit_pos - 1] + edit_patt + orig_expr[j+1:]
        else:  # delete current one
            new_expr = orig_expr[:edit_pos - 2] + orig_expr[j:]
    return new_expr


class LocScoreFunc(Function):
    @staticmethod
    def forward(ctx, mod_scores, del_scores, insert_scores, stop_scores, expr_poses, ins_poses, stop_poses, has_stopped):
        out_score = insert_scores.new(ins_poses.shape).zero_() + N_INF

        if mod_scores is not None and mod_scores.shape[0]:
            ex, ey = expr_poses.nonzero().split(1, dim=-1)
            out_score[ex, ey] = mod_scores
            #TODO: fix this magic; currently we assume each expr takes at least 2 positions,
            # so we use position of (ConstStr, SubStr) for the score of mod, and the one after that for del
            out_score[ex + 1, ey] = del_scores
        else:
            ex = ey = None
        ix, iy = ins_poses.nonzero().split(1, dim=-1)
        out_score[ix, iy] = insert_scores
        if has_stopped is not None:
            out_score[:, has_stopped] = N_INF
        sx, sy = stop_poses.nonzero().split(1, dim=-1)
        out_score[sx, sy] = stop_scores
        ctx.ix = ix
        ctx.iy = iy
        ctx.ex = ex
        ctx.ey = ey
        ctx.sx = sx
        ctx.sy = sy
        ctx.has_stopped = has_stopped
        return out_score

    @staticmethod
    def backward(ctx, grad_output):
        ex, ey, ix, iy, sx, sy, has_stopped = ctx.ex, ctx.ey, ctx.ix, ctx.iy, ctx.sx, ctx.sy, ctx.has_stopped
        assert has_stopped is None
        if ex is not None:
            g_mod = grad_output[ex, ey]
            g_del = grad_output[ex + 1, ey]
        else:
            g_mod = g_del = None
        g_ins = grad_output[ix, iy]
        g_stop = grad_output[sx, sy]
        return g_mod, g_del, g_ins, g_stop, None, None, None, None


class LocScore(nn.Module):
    def forward(self, mod_scores, del_scores, insert_scores, stop_scores, expr_poses, ins_poses, stop_poses, has_stopped):
        return LocScoreFunc.apply(mod_scores, del_scores, insert_scores, stop_scores, expr_poses, ins_poses, stop_poses, has_stopped)

loc_score = LocScore()

class EditLocationPredictor(nn.Module):
    def __init__(self, args):
        super(EditLocationPredictor, self).__init__()
        self.vocab = deepcopy(RFILL_VOCAB)
        self.tok_start = self.vocab['|']
        self.tok_constexpr = self.vocab['ConstStr']
        self.tok_subexpr = self.vocab['SubStr']
        self.tok_stop = self.vocab['eos']
        self.tok_pad = self.vocab['pad']

        self.tok_embed = nn.Embedding(len(self.vocab), args.embed_dim)
        self.rnn_layers = args.rnn_layers
        self.lstm = nn.LSTM(args.embed_dim, args.embed_dim, num_layers=self.rnn_layers, batch_first=False, bidirectional=True)

        self.ctx2h = MLP(args.embed_dim, [args.embed_dim * 2 * self.rnn_layers], act_last='tanh')
        self.ctx2c = MLP(args.embed_dim, [args.embed_dim * 2 * self.rnn_layers], act_last='tanh')

        self.del_score = MLP(args.embed_dim * 2, [args.embed_dim * 2, 1])
        self.modify_score = MLP(args.embed_dim * 2, [args.embed_dim * 2, 1])
        self.insert_score = MLP(args.embed_dim * 2, [args.embed_dim * 2, 1])
        self.stop_score = MLP(args.embed_dim * 2, [args.embed_dim * 2, 1])

    def forward(self, list_progs, context_embeds, ll=None, target_list=None, gen_method='sample', sizes=None, has_stopped=None):
        n_prog = len(list_progs)
        prog_int_seqs = [torch.LongTensor([self.vocab[c] for c in expr] + [self.tok_stop]).to(context_embeds.device) for expr in list_progs]
        lengths = [v.size(0) for v in prog_int_seqs]
        padded_int_seqs = pad_sequence(prog_int_seqs, batch_first=False, padding_value=self.tok_pad)

        packed_seq = pack_padded_sequence(padded_int_seqs, lengths=lengths, batch_first=False, enforce_sorted=False)
        tok_embed = self.tok_embed(packed_seq.data)
        packed_input = PackedSequence(data=tok_embed, batch_sizes=packed_seq.batch_sizes,
                        sorted_indices=packed_seq.sorted_indices, unsorted_indices=packed_seq.unsorted_indices)

        h = self.ctx2h(context_embeds).view(n_prog, 2 * self.rnn_layers, -1).transpose(0, 1)
        c = self.ctx2c(context_embeds).view(n_prog, 2 * self.rnn_layers, -1).transpose(0, 1)
        packed_out, _ = self.lstm(packed_input, (h, c))
        unpacked_out, _ = pad_packed_sequence(packed_out)

        # positions to mod/del
        expr_poses = (padded_int_seqs == self.tok_constexpr) | (padded_int_seqs == self.tok_subexpr)
        embed_expr = unpacked_out[expr_poses]
        if embed_expr.shape[0]:
            mod_scores = self.modify_score(embed_expr)
            del_scores = self.del_score(embed_expr)
        else:
            mod_scores = del_scores = None
        # positions to insert
        ins_poses = padded_int_seqs == self.tok_start
        insert_scores = self.insert_score(unpacked_out[ins_poses])

        # positions to stop
        stop_poses = padded_int_seqs == self.tok_stop
        stop_scores = self.stop_score(unpacked_out[stop_poses])
        logits = loc_score(mod_scores, del_scores, insert_scores, stop_scores, expr_poses, ins_poses, stop_poses, has_stopped)
        log_prob = F.log_softmax(logits, dim=0).t().contiguous()
        ll_target = None
        predecessors = None
        if target_list is None:
            if gen_method == 'sample':
                target = torch.multinomial(torch.exp(log_prob), 1)
            elif gen_method == 'argmax':
                target = torch.argmax(log_prob, dim=1)
            elif gen_method.startswith('beam'):
                beam_size = int(gen_method.split('-')[-1])
                raw_scores = log_prob + ll if ll is not None else log_prob
                predecessors, target, ll_target, sizes = beam_step(raw_scores, sizes, beam_size)
                update_embed = unpacked_out[target, predecessors]
            else:
                raise NotImplementedError
        else:
            target = torch.LongTensor(target_list).to(log_prob.device)
        target = target.view(-1)
        if predecessors is None:
            ll_step = log_prob[range(n_prog), target]
            ll_target = ll_step.view(ll.shape) + ll if ll is not None else ll_step
            update_embed = unpacked_out[target, range(n_prog)]
        return ll_target.view(-1, 1), target, update_embed, predecessors, sizes


class RFillOneStepEditor(nn.Module):
    def __init__(self, args):
        super(RFillOneStepEditor, self).__init__()
        self.rnn_layers = args.rnn_layers
        self.rnn_state_proj = args.rnn_state_proj

        if self.rnn_state_proj:
            self.ctx2h = MLP(args.embed_dim, [args.embed_dim * self.rnn_layers], act_last='tanh')
            self.ctx2c = MLP(args.embed_dim, [args.embed_dim * self.rnn_layers], act_last='tanh')

        self.editor_loc = EditLocationPredictor(args)
        self.subexpr_sampler = RfillSubexprRnnSampler(args)
        self.update_cell = nn.LSTM(args.embed_dim * 2, args.embed_dim, self.rnn_layers)

    def forward(self, io_embed, expr_list, ll=None, pos_list=None, target_subexpr=None, gen_method='sample', sizes=None, has_stopped=None):
        # get lstm state using io_embed
        if self.rnn_state_proj:
            h = self.ctx2h(io_embed).view(io_embed.shape[0], self.rnn_layers, -1).transpose(0, 1).contiguous()
            c = self.ctx2c(io_embed).view(io_embed.shape[0], self.rnn_layers, -1).transpose(0, 1).contiguous()
            state = (h, c)
        else:
            _, state = self.update_cell(io_embed.repeat(1, 2).unsqueeze(0))

        # predict edit location
        ll_pos, pos_list, update_embed, predecessors, sizes = self.editor_loc(expr_list, io_embed,
                                    ll=ll, target_list=pos_list, gen_method=gen_method, sizes=sizes, has_stopped=has_stopped)
        pos_list = pos_list.data.cpu().numpy()
        if predecessors is not None:
            state = (state[0][:, predecessors, :], state[1][:, predecessors, :])
            np_predecessors = predecessors.data.cpu().numpy()
        else:
            np_predecessors = range(len(pos_list))
        _, state = self.update_cell(update_embed.unsqueeze(0), state)

        offset = 0
        fsm_states = []
        ne_ids = []
        ne_exprs = []
        for i in range(len(pos_list)):
            prev_expr = expr_list[np_predecessors[i]]
            if pos_list[i] >= len(prev_expr):
                fsm_states.append(STATE_MAP['halt'])
            elif any([prev_expr[pos_list[i]] == c for c in ['|', 'SubStr', 'ConstStr']]):
                ne_ids.append(i)
                ne_exprs.append(target_subexpr[i] if target_subexpr is not None else None)
                fsm_states.append(STATE_MAP['non_empty_expr'])
            else:
                fsm_states.append(STATE_MAP['halt'])
        if target_subexpr is not None:  # training only
            assert ll is None
            ll_subexpr, _ = self.subexpr_sampler((state[0][:, ne_ids, :], state[1][:, ne_ids, :]), ne_exprs)
            ne_ids = torch.LongTensor(ne_ids).to(ll_subexpr.device).view(-1, 1)
            ll = ll_pos + scatter_add(ll_subexpr, ne_ids, 0, dim_size=ll_pos.shape[0])
            return ll

        ll, pred_subexpr, sizes, ancestors = self.subexpr_sampler(state, gen_method=gen_method, cur_sizes=sizes, init_states=fsm_states, init_ll=ll_pos)
        np_ancestors = ancestors.data.cpu().numpy()
        pos_list = [pos_list[i] for i in np_ancestors]
        if predecessors is not None:
            ancestors = predecessors[ancestors]
            np_ancestors = ancestors.data.cpu().numpy()
        new_exprs = [None] * len(pos_list)
        has_stopped = []

        for i in range(len(pos_list)):
            prev_expr = expr_list[np_ancestors[i]]
            if pos_list[i] >= len(prev_expr): # stop
                new_exprs[i] = prev_expr
                has_stopped.append(True)
            else:
                has_stopped.append(False)
                if any([prev_expr[pos_list[i]] == c for c in ['|', 'SubStr', 'ConstStr']]):
                    new_exprs[i] = perform_edit(prev_expr, pos_list[i], pred_subexpr[i])
                else:
                    new_exprs[i] = perform_edit(prev_expr, pos_list[i], None)
        return ll, new_exprs, sizes, ancestors, has_stopped


class RfillSampler(nn.Module):
    def __init__(self, args):
        super(RfillSampler, self).__init__()

        if args.io_enc == 'rnn':
            self.encoder = BidirIOEmbed(args)
        elif args.io_enc == 'mlp':
            self.encoder = MLPIOEmbed(args)
        else:
            raise NotImplementedError
        self.q0 = RfillRnnSampler(args)
        self.onestep = RFillOneStepEditor(args)

    def forward_io(self, samples):
        list_inputs, list_outputs, _, cooked_data = samples
        _, _, io_embed = self.encoder(list_inputs, list_outputs, cooked_data=cooked_data)
        return io_embed

    def forward_q0(self, io_embed, expr_list=None, cooked_data=None, gen_method='sample', sizes=None):
        return self.q0(io_embed, expr_list=expr_list, cooked_data=cooked_data, gen_method=gen_method, cur_sizes=sizes)

    def forward_onestep(self, io_embed, expr_list, ll=None, pos_list=None, subexpr_list=None, gen_method='sample', sizes=None, has_stopped=None):
        return self.onestep(io_embed, expr_list, ll=ll, pos_list=pos_list, target_subexpr=subexpr_list, gen_method=gen_method, sizes=sizes, has_stopped=has_stopped)

    def forward(self, samples, max_steps, sizes=None, gen_method='sample', phase=None):
        io_embed = self.forward_io(samples)
        ll, init_prog_list, sizes, ancestors = self.q0(io_embed, expr_list=None, cooked_data=samples[-1], gen_method=gen_method, cur_sizes=sizes)
        cur_exprs = init_prog_list
        has_stopped = None
        expr_traj = [cur_exprs]
        for i in range(max_steps):
            cur_io_embed = io_embed[ancestors]
            ll, new_exprs, sizes, sub_ancestors, has_stopped = self.forward_onestep(cur_io_embed,
                                                                            cur_exprs,
                                                                            ll=ll,
                                                                            gen_method=gen_method,
                                                                            sizes=sizes,
                                                                            has_stopped=has_stopped)
            ancestors = ancestors[sub_ancestors]
            cur_exprs = new_exprs
            if all(has_stopped):
                break
            expr_traj.append(new_exprs)
        if phase == 'plot':
            return expr_traj
        if phase == 'sampling':
            return io_embed, init_prog_list, cur_exprs
        return ll, cur_exprs, sizes, ancestors


def test_edit_dist():
    e_list = []
    for i in tqdm(range(10000)):
        e1 = sample_prog(cmd_args)[-1]
        e2 = sample_prog(cmd_args)[-1]

        e1 = e1.to_tokens()
        e2 = e2.to_tokens()
        e_list.append((e1, e2))
    for e1, e2 in tqdm(e_list):
        dist, prob, diff_list = shortest_rand_edit(e1, e2)
        for diff in diff_list:
            e1 = perform_edit(e1, diff[0], diff[1])
        assert ''.join(e1) == ''.join(e2)
    sys.exit()

def test_onestep():
    list_progs = []
    for _ in range(4):
        i, o, _, expr_root = sample_prog(cmd_args)
        prog_tokens = expr_root.to_tokens()
        list_progs.append(prog_tokens)
    editor = RFillOneStepEditor(cmd_args)

    exprs = deepcopy(list_progs)
    for _ in range(4):
        _, new_exprs, _, _, _ = editor(torch.randn(len(exprs), cmd_args.embed_dim), exprs)
        for i in range(len(new_exprs)):
            if new_exprs[i] is not None:
                exprs[i] = new_exprs[i]

    for expr, new_expr in zip(list_progs, exprs):
        print(expr)
        print(new_expr)
        dist, prob, diff_list = shortest_rand_edit(expr, new_expr)
        print(dist, prob, diff_list)
        for diff in diff_list:
            expr = perform_edit(expr, diff[0], diff[1])
        print(expr)
        print('\n')

        assert ''.join(expr) == ''.join(new_expr)
    sys.exit()
