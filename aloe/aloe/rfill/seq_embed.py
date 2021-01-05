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
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence, pack_padded_sequence

from aloe.common.pytorch_util import glorot_uniform, MLP, pad_sequence, MaskedEmbedding
from aloe.rfill.utils.rfill_consts import STR_VOCAB
from torch_scatter import scatter_mean, scatter_max


class AbstractIOEmbed(nn.Module):
    def __init__(self, args):
        super(AbstractIOEmbed, self).__init__()
        self.numPublicIO = args.numPublicIO
        if args.io_agg_type == 'max':
            self.agg_func = lambda x, idx: scatter_max(x, idx, dim=0)[0]
        elif args.io_agg_type == 'mean':
            self.agg_func = lambda x, idx: scatter_mean(x, idx, dim=0)
        else:
            raise NotImplementedError

    def io2idx_func(self, x):
        raise NotImplementedError

    def padded_inputs(self, int_seqs):
        raise NotImplementedError

    def padded_outputs(self, int_seqs):
        raise NotImplementedError

    def cook_raw_io(self, inputs_list, outputs_list):
        seq_input_ints = []
        seq_outputs_ints = []
        scatter_idx = []
        for i, (inputs, outputs) in enumerate(zip(inputs_list, outputs_list)):
            for x, y in zip(inputs[:self.numPublicIO], outputs[:self.numPublicIO]):
                seq_input_ints.append(self.io2idx_func(x))
                seq_outputs_ints.append(self.io2idx_func(y))
            scatter_idx += [i] * len(inputs)
        padded_i = self.padded_inputs(seq_input_ints)
        padded_o = self.padded_outputs(seq_outputs_ints)
        scatter_idx = torch.LongTensor(scatter_idx)
        return padded_i, padded_o, scatter_idx


class MLPIOEmbed(AbstractIOEmbed):
    def __init__(self, args, n_hidden=2):
        super(MLPIOEmbed, self).__init__(args)
        self.max_input_len = args.maxInputLength
        self.max_output_len = args.maxOutputLength
        self.vocab = {'unk': 0}
        for i, c in enumerate(STR_VOCAB):
            self.vocab[c] = i + 1
        if args.io_embed_type == 'normal':
            self.input_tok_embed = nn.Embedding(len(self.vocab), 4)
            self.output_tok_embed = nn.Embedding(len(self.vocab), 4)
        else:
            self.input_tok_embed = MaskedEmbedding(len(self.vocab), 4, masked_token=self.vocab['unk'])
            self.output_tok_embed = MaskedEmbedding(len(self.vocab), 4, masked_token=self.vocab['unk'])
        self.embed_merge = MLP(4 * (self.max_input_len + self.max_output_len), [args.embed_dim] * n_hidden,
                               nonlinearity=args.act_func,
                               act_last=args.act_func)
        self.device = args.device

    def io2idx_func(self, x):
        return [self.vocab[c] for c in x] if len(x) else [self.vocab['unk']]

    def _padded_io(self, int_seqs, max_len):
        int_seqs = [torch.LongTensor(x) for x in int_seqs]
        padded = pad_sequence(int_seqs, max_len=max_len, batch_first=True)
        return padded

    def padded_inputs(self, int_seqs):
        return self._padded_io(int_seqs, max_len=self.max_input_len)

    def padded_outputs(self, int_seqs):
        return self._padded_io(int_seqs, max_len=self.max_output_len)

    def embed_seq(self, padded, tok_embed):
        tok_embed = tok_embed(padded.data)
        seq_embed = tok_embed.view(padded.shape[0], -1)
        return seq_embed

    def forward(self, inputs_list, outputs_list, cooked_data=None):
        if cooked_data is None:
            padded_i, padded_o, scatter_idx = self.cook_raw_io(inputs_list, outputs_list)
        else:
            padded_i, padded_o, scatter_idx = cooked_data.get_ios()
        padded_i = padded_i.to(self.device)
        padded_o = padded_o.to(self.device)
        scatter_idx = scatter_idx.to(self.device)

        input_embed = self.embed_seq(padded_i, self.input_tok_embed)
        output_embed = self.embed_seq(padded_o, self.output_tok_embed)

        single_io_input = torch.cat((input_embed, output_embed), dim=-1)
        single_io_embed = self.embed_merge(single_io_input)
        ctx_embed = self.agg_func(single_io_embed, scatter_idx)
        return None, None, ctx_embed


class TripletIOEmbed(MLPIOEmbed):
    def __init__(self, args):
        super(MLPIOEmbed, self).__init__(args)
        self.max_output_len = args.maxOutputLength
        self.vocab = {'unk': 0}
        for i, c in enumerate(STR_VOCAB):
            self.vocab[c] = i + 1
        if args.io_embed_type == 'normal':
            self.tok_embed = nn.Embedding(len(self.vocab), 4)
        else:
            self.tok_embed = MaskedEmbedding(len(self.vocab), 4, masked_token=self.vocab['unk'])

        self.embed_merge = MLP(4 * 3 * self.max_output_len, [args.embed_dim] * 5,
                               nonlinearity=args.act_func,
                               act_last=args.act_func)
        self.device = args.device

    def padded_inputs(self, int_seqs):
        return self._padded_io(int_seqs, max_len=self.max_output_len)

    def forward(self, inputs_list, outputs_list, cooked_data=None):
        if cooked_data is None:
            padded_i, padded_o, scatter_idx = self.cook_raw_io(inputs_list, outputs_list)
        else:
            padded_i, padded_o, scatter_idx = cooked_data.get_ios()
        padded_i = padded_i.to(self.device)
        padded_o = padded_o.to(self.device)
        scatter_idx = scatter_idx.to(self.device)
        input_embed = self.embed_seq(padded_i, self.tok_embed)
        output_embed = self.embed_seq(padded_o, self.tok_embed)
        diff_embed = output_embed - input_embed

        single_io_input = torch.cat((input_embed, output_embed, diff_embed), dim=-1)
        single_io_embed = self.embed_merge(single_io_input)
        ctx_embed = self.agg_func(single_io_embed, scatter_idx)
        return None, None, ctx_embed


class BidirIOEmbed(AbstractIOEmbed):
    def __init__(self, args):
        super(BidirIOEmbed, self).__init__(args)
        self.vocab = {'unk': 0, 'eos': 1}
        for i, c in enumerate(STR_VOCAB):
            self.vocab[c] = i + 2
        self.tok_embed = nn.Embedding(len(self.vocab), args.embed_dim)
        self.lstm = nn.LSTM(args.embed_dim, args.embed_dim, 3, bidirectional=False)
        self.embed_merge = MLP(args.embed_dim * 2, [args.embed_dim], nonlinearity=args.act_func)
        self.device = args.device

    def io2idx_func(self, x):
        return [self.vocab[c] for c in x] + [self.vocab['eos']]

    def _pad_io(self, int_seqs):
        int_seqs = [torch.LongTensor(x) for x in int_seqs]
        lengths = [v.size(0) for v in int_seqs]
        return pad_sequence(int_seqs), lengths

    def padded_inputs(self, int_seqs):
        return self._pad_io(int_seqs)

    def padded_outputs(self, int_seqs):
        return self._pad_io(int_seqs)

    def embed_seq(self, packed_seq, scatter_idx):
        tok_embed = self.tok_embed(packed_seq.data)
        packed_input = PackedSequence(data=tok_embed, batch_sizes=packed_seq.batch_sizes,
                        sorted_indices=packed_seq.sorted_indices, unsorted_indices=packed_seq.unsorted_indices)

        _, (h, c) = self.lstm(packed_input)
        return self.agg_func(h[-1], scatter_idx)

    def forward(self, inputs_list, outputs_list, cooked_data=None):
        if cooked_data is None:
            padded_i, padded_o, scatter_idx = self.cook_raw_io(inputs_list, outputs_list)
        else:
            padded_i, padded_o, scatter_idx = cooked_data.get_ios()
        scatter_idx = scatter_idx.to(self.device)
        packed_i = pack_padded_sequence(padded_i[0].to(self.device), padded_i[1], enforce_sorted=False)
        packed_o = pack_padded_sequence(padded_o[0].to(self.device), padded_o[1], enforce_sorted=False)

        input_embed = self.embed_seq(packed_i, scatter_idx)
        output_embed = self.embed_seq(packed_o, scatter_idx)
        merged = self.embed_merge(torch.cat((input_embed, output_embed), dim=-1))

        return input_embed, output_embed, merged
