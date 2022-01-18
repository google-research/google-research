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
import os
import gzip
import json
import itertools
from .rfill_parser import RobustFillParser
import random
import torch
from torch.utils.data import Dataset, IterableDataset
from aloe.common.pytorch_util import pad_sequence
from aloe.rfill.utils.rfill_grammar import STATE_TRANS, trans_map, STATE_MAP, RFILL_VOCAB


class RFillSample(object):
    def __init__(self, prog_idx, tree_root, ios, db):
        self.prog_idx = prog_idx
        self.db = db
        self.tree_root = tree_root
        self.raw_ios = ios


class CookedData(object):
    def __init__(self, padded_i, padded_o, scatter_idx, padded_states):
        self.padded_i = padded_i
        self.padded_o = padded_o
        self.scatter_idx = scatter_idx
        self.padded_states = padded_states

    def get_ios(self):
        return self.padded_i, self.padded_o, self.scatter_idx


def cooked_collade_fn(cls, list_samples):
    list_i = []
    list_o = []
    list_p = []
    seq_input_ints = []
    seq_outputs_ints = []
    scatter_idx = []
    list_samples.sort(key=lambda x: -len(x[2]))
    list_states = []
    for idx, (i, o, p, int_i, int_o, states) in enumerate(list_samples):
        list_i.append(i)
        list_o.append(o)
        list_p.append(p)
        list_states.append(states)
        scatter_idx += [idx] * len(int_i)
        seq_input_ints += int_i
        seq_outputs_ints += int_o
    if cls.need_mask:
        list_states = [torch.LongTensor(x) for x in list_states]
        padded_states = pad_sequence(list_states, batch_first=True, padding_value=STATE_MAP['halt'])
    else:
        padded_states = None
    padded_i = cls.fn_pad_in(seq_input_ints)
    padded_o = cls.fn_pad_out(seq_outputs_ints)
    scatter_idx = torch.LongTensor(scatter_idx)

    return list_i, list_o, list_p, CookedData(padded_i, padded_o, scatter_idx, padded_states)


def raw_txt2int(cls, inputs, outputs, prog_tokens):
    seq_input_ints = []
    seq_outputs_ints = []
    for x, y in zip(inputs[:cls.numPublicIO], outputs[:cls.numPublicIO]):
        seq_input_ints.append(cls.io2idx_func(x))
        seq_outputs_ints.append(cls.io2idx_func(y))
    if cls.need_mask:
        cur_state = 'start'
        states = []
        for t in prog_tokens:
            cur_state = trans_map[(cur_state, t)]
            states.append(STATE_MAP[cur_state])
    else:
        states = None
    return inputs, outputs, prog_tokens, seq_input_ints, seq_outputs_ints, states


class CookedInfRfill(IterableDataset):
    def __init__(self, args, io2idx_func, fn_pad_in, fn_pad_out, need_mask=False):
        self.args = args
        self.iter_per_epoch = args.iter_per_epoch

        self.data_dir = args.data_dir
        self.numPublicIO = args.numPublicIO
        self.io2idx_func = io2idx_func
        self.fn_pad_in = fn_pad_in
        self.fn_pad_out = fn_pad_out
        self.need_mask = need_mask

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1
        cur_id = 0
        if worker_info is not None:
            num_workers = worker_info.num_workers
            cur_id = worker_info.id

        list_files = []
        for fname in os.listdir(self.data_dir):
            if not fname.startswith('train'):
                continue
            fidx = int(fname.split('.')[0].split('-')[1])
            if fidx % num_workers == cur_id:
                list_files.append(os.path.join(self.data_dir, fname))

        while True:
            random.shuffle(list_files)
            for fname in list_files:
                with gzip.open(fname, 'rb') as f:
                    raw_json_strings = f.readlines()
                random.shuffle(raw_json_strings)
                for row in raw_json_strings:
                    data_dict = json.loads(row.strip())
                    inputs, outputs, prog_tokens = data_dict['inputs'], data_dict['outputs'], data_dict['exprs']
                    yield raw_txt2int(self, inputs, outputs, prog_tokens)

    def collate_fn(self, list_samples):
        return cooked_collade_fn(self, list_samples)


class RawStaticRfill(Dataset):
    def __init__(self, args, json_path):
        print('loading data %s' % json_path)
        with open(json_path, 'r') as f:
            self.raw_json_strings = f.readlines()
        print('loaded')
        self.num_programs = len(self.raw_json_strings)

        self.samples = [None] * self.num_programs
        self.iter_per_epoch = args.iter_per_epoch

    def __len__(self):
        return self.num_programs

    def __getitem__(self, index):
        if self.samples[index] is None:
            row = self.raw_json_strings[index].strip()
            data_dict = json.loads(row.strip())
            self.samples[index] = (data_dict['inputs'], data_dict['outputs'], data_dict['exprs'])
        return self.samples[index]

    def collate_fn(self, list_samples):
        list_i = []
        list_o = []
        list_p = []
        list_samples.sort(key=lambda x: -len(x[2]))
        for i, o, p in list_samples:
            list_i.append(i)
            list_o.append(o)
            list_p.append(p)
        return list_i, list_o, list_p, None
