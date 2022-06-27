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
import sys
import ctypes
import numpy as np
import torch
from lego.cpp_sampler import libsampler
from lego.cpp_sampler import sampler_clib


from collections import defaultdict
from tqdm import tqdm


def is_all_relation(query_structure):
    for ele in query_structure[-1]:
        if ele not in ['r', 'n']:
            return False
    return True


def has_negation(query_structure):
    for ele in query_structure[-1]:
        if ele == 'n':
            return True
    return False


def build_query_tree(query_structure, fn_qt_create):
    if is_all_relation(query_structure):
        assert len(query_structure) == 2
        if query_structure[0] == 'e':
            prev_node = fn_qt_create(libsampler.entity)
        else:
            prev_node = build_query_tree(query_structure[0], fn_qt_create)
        for i, c in enumerate(query_structure[-1]):
            if c == 'r':
                cur_op = libsampler.relation
            else:
                assert c == 'n'
                cur_op = libsampler.negation
            cur_root = fn_qt_create(libsampler.entity_set)
            cur_root.add_child(cur_op, prev_node)
            prev_node = cur_root
        return cur_root
    else:
        last_qt = query_structure[-1]
        node_type = libsampler.intersect
        if len(last_qt) == 1 and last_qt[0] == 'u':
            node_type = libsampler.union
            query_structure = query_structure[:-1]
        sub_root = fn_qt_create(node_type)
        for c in query_structure:
            ch_node = build_query_tree(c, fn_qt_create)
            sub_root.add_child(libsampler.no_op, ch_node)
        return sub_root


class OnlineSampler(object):
    def __init__(self, kg, query_structures, negative_sample_size,
                 sample_mode, normalized_structure_prob, sampler_type='naive',
                 share_negative=False, same_in_batch=False,
                 weighted_answer_sampling=False, weighted_negative_sampling=False,
                 nprefetch=10, num_threads=8):
        self.kg = kg
        kg_dtype = kg.dtype
        fn_qt_create = libsampler.create_qt32 if kg_dtype == 'uint32' else libsampler.create_qt64
        self.query_structures = query_structures
        self.normalized_structure_prob = normalized_structure_prob
        assert len(normalized_structure_prob) == len(query_structures)
        self.negative_sample_size = negative_sample_size
        self.share_negative = share_negative
        self.same_in_batch = same_in_batch
        self.nprefetch = nprefetch
        if len(sample_mode) == 5:
            self.rel_bandwidth, self.max_to_keep, self.weighted_style, self.structure_weighted_style, self.max_n_partial_answers = sample_mode
            self.weighted_ans_sample = False
            self.weighted_neg_sample = False
        else:
            self.rel_bandwidth, self.max_to_keep, self.weighted_style, self.structure_weighted_style, self.max_n_partial_answers, self.weighted_ans_sample, self.weighted_neg_sample = sample_mode
        if self.rel_bandwidth <= 0:
            self.rel_bandwidth = kg.num_ent
        if self.max_to_keep <= 0:
            self.max_to_keep = kg.num_ent
        if self.max_n_partial_answers <= 0:
            self.max_n_partial_answers = kg.num_ent
        if self.structure_weighted_style == 'wstruct':
            assert self.normalized_structure_prob is not None

        list_qt = []
        for qs in query_structures:
            if qs[0] == '<':  # inverse query
                assert is_all_relation(qs[1]) and not has_negation(qs[1])
                qt = build_query_tree(qs[1], fn_qt_create)
                qt.is_inverse = True
            else:
                qt = build_query_tree(qs, fn_qt_create)
            list_qt.append(qt)
        self.list_qt = list_qt
        no_search_list = []

        if sampler_type == 'naive':
            sampler_cls = sampler_clib.naive_sampler(kg_dtype)
        elif sampler_type.startswith('sqrt'):
            sampler_cls = sampler_clib.rejection_sampler(kg_dtype)
            if '-' in sampler_type:
                no_search_list = [int(x) for x in sampler_type.split('-')[1].split('.')]
        elif sampler_type == 'nosearch':
            sampler_cls = sampler_clib.no_search_sampler(kg_dtype)
        else:
            raise ValueError("Unknown sampler %s" % sampler_type)
        self.sampler = sampler_cls(kg, list_qt, normalized_structure_prob, self.share_negative, self.same_in_batch,
                                    self.weighted_ans_sample, self.weighted_neg_sample,
                                    negative_sample_size, self.rel_bandwidth, self.max_to_keep, self.max_n_partial_answers, num_threads, no_search_list)

    def print_queries(self):
        self.sampler.print_queries()

    def set_seed(self, seed):
        self.sampler.set_seed(seed)

    def batch_generator(self, batch_size):
        self.sampler.prefetch(batch_size, self.nprefetch)
        uniform_weigths = torch.ones(batch_size)
        list_buffer = []
        for i in range(2):
            t_pos_ans = torch.LongTensor(batch_size)
            if self.share_negative:
                t_neg_ans = torch.LongTensor(1, self.negative_sample_size)
                t_is_neg_mat = torch.FloatTensor(batch_size, self.negative_sample_size)
            else:
                t_neg_ans = torch.LongTensor(batch_size, self.negative_sample_size)
                t_is_neg_mat = torch.FloatTensor(1, 2)
            t_weights = torch.FloatTensor(batch_size)
            list_buffer.append((t_pos_ans, t_neg_ans, t_is_neg_mat, t_weights))

        buf_idx = 0
        while True:
            pos_ans, neg_ans, is_neg_mat, weights = list_buffer[buf_idx]
            q_args = []
            q_structs = []
            self.sampler.next_batch(pos_ans.numpy(), neg_ans.numpy(), weights.numpy(), is_neg_mat.numpy(),
                                    q_args, q_structs)
            if self.weighted_style == 'u':
                weights = uniform_weigths
            yield pos_ans, neg_ans, is_neg_mat if self.share_negative else None, weights, q_args, [self.query_structures[x] for x in q_structs]
            buf_idx = 1 - buf_idx

