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
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict
from copy import deepcopy
from lego.cpp_sampler.online_sampler import OnlineSampler
from smore.evaluation.dataloader import TestDataset
from lego.common.utils import query_name_dict, flatten, get_edge_index, get_powerset, list2tuple, recursive_main, flatten, flatten_list


class RelationSampler(OnlineSampler):
    def batch_generator(self, batch_size):
        super_batch_gen = super(RelationSampler, self).batch_generator(batch_size)
        while True:
            pos_ans, _, _, _, q_args, q_structs = next(super_batch_gen)
            single_ents = np.random.randint(low=0, high=self.kg.num_ent, size=(batch_size))
            pos_ans = torch.cat([pos_ans, torch.LongTensor(single_ents)], dim=0)
            q_args = q_args + np.expand_dims(single_ents, axis=1).tolist()
            q_structs = q_structs + [('e',())]*batch_size
            yield pos_ans, q_args, q_structs


class BranchSampler(OnlineSampler):
    def batch_generator(self, batch_size):
        super_batch_gen = super(BranchSampler, self).batch_generator(batch_size)
        while True:
            pos_ans, _, _, _, q_args, q_structs = next(super_batch_gen)
            labels = [1]*len(q_structs)
            q_args_flatten = []
            fake_q_args = []
            fake_q_structs = []
            for idx, q_struct in enumerate(q_structs):
                if query_name_dict[q_struct] == '2i':
                    q_args_flatten.append(q_args[idx][:2])
                    q_args_flatten.append(q_args[idx][2:4])
                elif query_name_dict[q_struct] == '3i':
                    q_args_flatten.append(q_args[idx][:2])
                    q_args_flatten.append(q_args[idx][2:4])
                    q_args_flatten.append(q_args[idx][4:6])
                elif query_name_dict[q_struct] == 'pi':
                    q_args_flatten.append(q_args[idx][:3])
                    q_args_flatten.append(q_args[idx][3:5])

                    fake_q = q_args[idx]
                    fake_q_args.append(fake_q[:2])
                    fake_q_args.append(fake_q[3:5])
                    fake_q_structs.append((('e', ('r',)), ('e', ('r',))))
                    labels.append(0)
                elif query_name_dict[q_struct] == '2pi':
                    q_args_flatten.append(q_args[idx][:3])
                    q_args_flatten.append(q_args[idx][3:6])

                    fake_q = q_args[idx]
                    fake_q_args.append(fake_q[:2])
                    fake_q_args.append(fake_q[3:6])
                    fake_q_structs.append((('e', ('r',)), ('e', ('r', 'r'))))
                    labels.append(0)

                    fake_q = q_args[idx]
                    fake_q_args.append(fake_q[:3])
                    fake_q_args.append(fake_q[3:5])
                    fake_q_structs.append((('e', ('r', 'r')), ('e', ('r',))))
                    labels.append(0)
                elif query_name_dict[q_struct] == 'p3i':
                    q_args_flatten.append(q_args[idx][:3])
                    q_args_flatten.append(q_args[idx][3:5])
                    q_args_flatten.append(q_args[idx][5:7])

                    fake_q = q_args[idx]
                    fake_q_args.append(fake_q[:2])
                    fake_q_args.append(fake_q[3:5])
                    fake_q_args.append(fake_q[5:7])
                    fake_q_structs.append((('e', ('r',)), ('e', ('r',)), ('e', ('r',))))
                    labels.append(0)
            q_args = q_args_flatten + fake_q_args
            q_structs += fake_q_structs
            q_structs_flatten = [x for y in q_structs for x in y]
            assert len(q_args) == len(q_structs_flatten)
            assert len(q_structs) == len(labels)
            labels = torch.FloatTensor(labels).unsqueeze(1)
            yield q_args, q_structs, q_structs_flatten, labels


class EvalRelationDataset(Dataset):
    def __init__(self, data, nentity, nrelation):
        self.queries = data['query']
        self.structures = data['structure']
        self.rels = data['rel']
        self.len = len(self.queries)
        self.nentity = nentity
        self.nrelation = nrelation

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return flatten(self.queries[idx]), self.queries[idx], self.structures[idx], self.rels[idx]

    @staticmethod
    def collate_fn(data):
        query = [_[0] for _ in data]
        query_unflatten = [_[1] for _ in data]
        query_structure = [_[2] for _ in data]
        rel = torch.LongTensor([_[3] for _ in data]).unsqueeze(1)
        return query, query_unflatten, query_structure, rel


class EvalBranchDataset(Dataset):
    def __init__(self, data, nentity, nrelation):
        self.queries = data['query']
        self.structures = data['structure']
        self.labels = data['label']
        self.len = len(self.queries)
        self.nentity = nentity
        self.nrelation = nrelation

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.queries[idx], self.structures[idx], self.labels[idx]

    @staticmethod
    def collate_fn(data):
        query_unflatten = [_[0] for _ in data]
        query = [flatten(x) for y in query_unflatten for x in y]
        query_structure_unflatten = [_[1] for _ in data]
        query_structure = [x for y in query_structure_unflatten for x in y]
        label = torch.LongTensor([_[2] for _ in data]).unsqueeze(1)
        return query, query_unflatten, query_structure, query_structure_unflatten, label


class EvalQuestionDataset(TestDataset):
    def __init__(self, data, answers, nentity, nrelation):
        super(EvalQuestionDataset, self).__init__(nentity, nrelation)
        self.data = data
        self.answers = answers
        self.test_all = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][0]
        query = self.data[idx][1]
        query_structure = self.data[idx][2]
        easy_answers = set()
        hard_answers = self.answers[question]
        if self.test_all:
            neg_samples = None
        else:
            neg_samples = torch.LongTensor(list(hard_answers) + list(self.data[idx][4]))
        return neg_samples, flatten(query), query, query_structure, easy_answers, hard_answers

    def subset(self, pos, num):
        data = self.data[pos : pos + num]
        return EvalQuestionDataset(data, self.answers, self.nentity, self.nrelation)


class UnfoldedProgramDataloader(Dataset):
    def __init__(self, data, nentity, nrelation, batch_size, query2box,
                 supervised_batch_size=0, supervised_minimum_reward=1.,
                 supervised_update_strictly_better=False,
                 max_nentity=4, shuffle=True, eval=False,
                 reduce='sum', weighted_sample=False, temperature=1.,
                 skip_indices=[], supervised_candidate_batch_size=1,):
        self.len = len(data)
        self.data = data
        self.nentity = nentity
        self.nrelation = nrelation
        self.batch_size = batch_size
        self.query2box = query2box
        assert self.batch_size == 1, "batching not supported"
        self.max_nentity = max_nentity
        assert max_nentity > 1
        self.max_y_score_len = int(pow(2, max_nentity)) - 1
        self.i = 0
        self.idxs = list(range(self.len))
        self.shuffle = shuffle
        self.eval = eval
        self.all_edge_indices = []
        self.all_powersets = []
        self.reduce = reduce
        self.supervised_batch_size = supervised_batch_size
        self.supervised_minimum_reward = supervised_minimum_reward
        self.supervised_update_strictly_better = supervised_update_strictly_better
        self.supervised_candidate_batch_size = supervised_candidate_batch_size
        self.skip_indices = skip_indices
        for i in range(max_nentity + 1):
            if i == 0:
                self.all_edge_indices.append([])
                self.all_powersets.append([])
            else:
                if i == 1:
                    self.all_edge_indices.append(np.array(get_edge_index(i)).T)
                    self.all_powersets.append(get_powerset(i))
                else:
                    edge_index = np.array(get_edge_index(i)).T
                    edge_index[1] -= 1
                    self.all_edge_indices.append(edge_index)
                    self.all_powersets.append(get_powerset(i)[1:])
        self.max_rewards = np.zeros((self.len))
        self.avg_rewards = 0.01*np.ones((self.len))
        self.n_sampled = np.ones((self.len))
        self.weighted_sample = weighted_sample
        self.temperature = temperature
        self.best_solutions = [[] for _ in range(self.len)]
        if shuffle:
            np.random.shuffle(self.idxs)

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(data):

        x_scores = np.concatenate([_[0] for _ in data], 0)
        x_relations = np.concatenate([_[1] for _ in data], 0)
        y_scores = np.concatenate([_[2] for _ in data], 0)
        y_relations = np.concatenate([_[3] for _ in data], 0)
        mask_relations = np.concatenate([_[4] for _ in data], 0)
        w_scores = np.concatenate([_[5] for _ in data], 0)
        w_relations = np.concatenate([_[6] for _ in data], 0)
        berts = np.concatenate([_[7] for _ in data], 0)
        max_y_score_len = data[0][11]
        mask_relations_class = np.concatenate([_[12] for _ in data], 0)
        noisy_mask_relations = np.concatenate([_[15] for _ in data], 0)
        nrelation = data[0][22]
        idx_list = [_[23] for _ in data]

        edge_indices, additional_edge_indices, question_indices, step_indices, softmax_edge_indices = [], [], [], [], []
        n_program, n_data, n_powerset, n_question, n_candidate = 0, 0, 0, 0, 0
        for i in range(len(data)):
            edge_index = data[i][8]
            edge_index[0] += n_data
            edge_index[1] += n_powerset
            edge_indices.append(edge_index)

            additional_edge_index = data[i][16]
            if len(additional_edge_index) > 0:
                additional_edge_index += n_powerset
                additional_edge_indices.append(additional_edge_index)

            question_index = data[i][13]
            question_index[0] += n_candidate
            question_index[1] += n_question
            question_indices.append(question_index)

            step_index = data[i][14]
            step_index[0] += n_program
            step_index[1] += n_candidate
            step_indices.append(step_index)

            softmax_edge_index = data[i][9]
            softmax_edge_index[0] += n_powerset
            softmax_edge_index[1] += n_program * max_y_score_len
            softmax_edge_indices.append(softmax_edge_index)

            n_program += data[i][17]
            n_data += data[i][18]
            n_powerset += data[i][19]
            n_question += data[i][20]
            n_candidate += data[i][21]

        if len(additional_edge_indices) > 0:
            additional_edge_indices = np.concatenate(additional_edge_indices, 0)
            additional_edge_indices = np.stack([[n_data]*len(additional_edge_indices), additional_edge_indices])
            edge_indices.append(additional_edge_indices)


        edge_indices = np.concatenate(edge_indices, axis=1)
        softmax_edge_indices = np.concatenate(softmax_edge_indices, axis=1)
        question_indices = np.concatenate(question_indices, axis=1)
        step_indices = np.concatenate(step_indices, axis=1)

        return x_scores, x_relations, y_scores, y_relations, mask_relations, w_scores, w_relations, berts, edge_indices, softmax_edge_indices, n_program, max_y_score_len, mask_relations_class, question_indices, step_indices, noisy_mask_relations, nrelation, idx_list

    def __getitem__(self, idx):
        '''
        x_scores: (n_data, center dim + offset dim)
        x_relations: (n_data, center dim + offset dim + bert dim)
        y_scores: (n_program, max_y_score_len)
        y_relations: (n_data, nrelation)
        mask_relations: (n_data)
        w_scores: (n_program)
        w_relations: (n_data)
        berts: (n_powerset, bert dim)
        edge_indices: (2, n_message_passing)
        softmax_edge_indices: (2, n_powerset)
        mask_relations_class: (n_data, nrelation)
        note that n_powerset != n_data * max_y_score_len.
            n_powerset = \sum_i 2^n_i (e.g. 2+8+4+16), n_data * max_y_score_len = 4*16,
            n_message_passing = \sum n_i * 2^(n_i - 1),
            n_program = args.batch_size
        '''
        x_scores, x_relations, y_scores, y_relations = [], [], [], []
        mask_relations, w_scores, w_relations, berts = [], [], [], []
        noisy_mask_relations = []
        edge_indices, softmax_edge_indices = [], []
        mask_relations_class = []
        if self.reduce == 'max':
            additional_edge_indices = []
        else:
            additional_edge_indices = []
        n_data = 0
        n_powerset = 0
        n_program = 0
        n_question = 0
        n_candidate = 0
        n_all, n_negative_one, n_str = 0, 0, 0
        question_indices, step_indices = [], []

        i = 0
        num_cand = len(self.data[idx])
        if num_cand < self.supervised_candidate_batch_size:
            cand_indices = np.arange(num_cand)
        else:
            cand_indices = np.random.choice(num_cand, size=self.supervised_candidate_batch_size, replace=False)
        for cand_idx in cand_indices:
            for step_idx in range(len(self.data[idx][cand_idx])):
                bert, center, offset, action, w, query, query_structure, question, mask_relation_class = self.data[idx][cand_idx][step_idx]
                bert = np.array(bert) # (1, 768)
                if len(center) == 1:
                    cur_powerset_len = 2
                else:
                    cur_powerset_len = int(pow(2, len(center))-1)
                berts.append(np.tile(bert, (cur_powerset_len, 1)))
                powerset = self.all_powersets[len(center)]
                assert len(powerset) == cur_powerset_len
                target = tuple(np.arange(len(action[0]))[np.array(action[0]) == 1]) #(0,1) or (1) or (0,1,2)
                y_scores.append(powerset.index(target))
                edge_index = deepcopy(self.all_edge_indices[len(center)])
                edge_index[0] += n_data
                edge_index[1] += n_powerset
                edge_indices.append(edge_index)
                if self.reduce == 'max' and len(center) == 1:
                    additional_edge_indices.append(n_powerset)
                softmax_edge_index = np.stack([n_powerset + np.arange(cur_powerset_len), n_program * self.max_y_score_len + np.arange(cur_powerset_len)])
                softmax_edge_indices.append(softmax_edge_index)
                w_scores.append(1/w)
                step_indices.append(np.array([[n_program], [n_candidate]]))
                n_program += 1
                n_data += len(center)
                n_powerset += cur_powerset_len

                no_relation = type(action[-1]) == str
                for j in range(len(center)):
                    x_scores.append(np.concatenate([center[j], offset[j]]))
                    x_relations.append(np.concatenate([center[j], offset[j], bert[0]]))
                    y_relations.append(action[-1] if not no_relation else 0)
                    mask_relations_class.append(mask_relation_class)
                    if no_relation:
                        mask_relations.append(0.)
                        if j == 0:
                            noisy_mask_relations.append(1.)
                        else:
                            noisy_mask_relations.append(0.)
                    else:
                        if self.eval: # a notable change here, it means that when evaluation, we should do all the necessary prediction
                            mask_relations.append(1. if action[0][j] == 1 else 0.)
                            noisy_mask_relations.append(1. if action[0][j] == 1 else 0.)
                        else: # but when it is training, we need to filter those datapoints when mask_relation_class does not even have the ground truth task
                            mask_relations.append(1. if action[0][j] == 1 and mask_relation_class[action[-1]] else 0.)
                            noisy_mask_relations.append(1. if action[0][j] == 1 and mask_relation_class[action[-1]] else 0.)
                    w_relations.append(1/w)

            question_indices.append(np.array([[n_candidate], [n_question]]))
            n_candidate += 1
        n_question += 1

        berts = np.concatenate(berts, axis=0)
        edge_indices = np.concatenate(edge_indices, axis=1)
        softmax_edge_indices = np.concatenate(softmax_edge_indices, axis=1)
        question_indices = np.concatenate(question_indices, axis=1)
        step_indices = np.concatenate(step_indices, axis=1)
        if len(additional_edge_indices) > 0:
            additional_edge_indices = np.array(additional_edge_indices)

        return x_scores, x_relations, y_scores, y_relations, mask_relations, w_scores, w_relations, berts, edge_indices, softmax_edge_indices, n_program, self.max_y_score_len, mask_relations_class, question_indices, step_indices, noisy_mask_relations, additional_edge_indices, n_program, n_data, n_powerset, n_question, n_candidate, self.nrelation, idx


class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0
        self.max_y_score_len = dataloader.dataset.max_y_score_len
        self.eval = dataloader.dataset.eval
        self.len = dataloader.dataset.len

    def __len__(self):
        return self.len

    def __next__(self):
        self.step += 1
        data = next(self.iterator)

        x_scores, x_relations, y_scores, y_relations, mask_relations, w_scores, w_relations, berts, edge_indices, softmax_edge_indices, n_program, max_y_score_len, mask_relations_class, question_indices, step_indices, noisy_mask_relations, nrelation, idx_list = data
        x_scores = torch.Tensor(x_scores)
        x_relations = torch.Tensor(x_relations)
        y_scores = torch.LongTensor(y_scores)
        y_relations = torch.LongTensor(y_relations)
        # y_relations = F.one_hot(torch.LongTensor(y_relations), nrelation)
        mask_relations = torch.Tensor(mask_relations)
        w_scores = torch.Tensor(w_scores)
        w_relations = torch.Tensor(w_relations)
        berts = torch.Tensor(berts)
        mask_relations_class = torch.Tensor(mask_relations_class).bool()
        noisy_mask_relations = torch.Tensor(noisy_mask_relations).bool()

        edge_indices = torch.LongTensor(edge_indices)
        softmax_edge_indices = torch.LongTensor(softmax_edge_indices)
        question_indices = torch.LongTensor(question_indices)
        step_indices = torch.LongTensor(step_indices)

        return x_scores, x_relations, y_scores, y_relations, mask_relations, w_scores, w_relations, berts, edge_indices, softmax_edge_indices, n_program, max_y_score_len, mask_relations_class, question_indices, step_indices, noisy_mask_relations

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data

    def train_supervised(self, maxlen, ground_truth=False):
        return True

    def next_supervised(self):
        return self.__next__()


class ProgramDataloader(object):
    def __init__(self, data, nentity, nrelation, batch_size, query2box,
                 supervised_batch_size=0, supervised_minimum_reward=1.,
                 supervised_update_strictly_better=False,
                 max_nentity=4, shuffle=True, eval=False,
                 reduce='sum', weighted_sample=False, temperature=1.,
                 skip_indices=[]):
        # print (data, batch_size)
        self.len = len(data)
        self.data = data
        self.nentity = nentity
        self.nrelation = nrelation
        self.batch_size = batch_size
        self.query2box = query2box
        assert self.batch_size == 1, "batching not supported"
        self.max_nentity = max_nentity
        assert max_nentity > 1
        self.max_y_score_len = int(pow(2, max_nentity)) - 1
        self.i = 0
        self.idxs = list(range(self.len))
        self.shuffle = shuffle
        self.eval = eval
        self.all_edge_indices = []
        self.all_powersets = []
        self.reduce = reduce
        self.supervised_batch_size = supervised_batch_size
        self.supervised_minimum_reward = supervised_minimum_reward
        self.supervised_update_strictly_better = supervised_update_strictly_better
        self.skip_indices = skip_indices
        for i in range(max_nentity + 1):
            if i == 0:
                self.all_edge_indices.append([])
                self.all_powersets.append([])
            else:
                if i == 1:
                    self.all_edge_indices.append(np.array(get_edge_index(i)).T)
                    self.all_powersets.append(get_powerset(i))
                else:
                    edge_index = np.array(get_edge_index(i)).T
                    edge_index[1] -= 1
                    self.all_edge_indices.append(edge_index)
                    self.all_powersets.append(get_powerset(i)[1:])

        self.max_rewards = np.zeros((self.len))
        self.avg_rewards = 0.01*np.ones((self.len))
        self.n_sampled = np.ones((self.len))
        self.weighted_sample = weighted_sample
        self.temperature = temperature
        self.best_solutions = [[] for _ in range(self.len)]
        if shuffle:
            np.random.shuffle(self.idxs)

    def train_supervised(self, maxlen, ground_truth=False):
        if ground_truth:
            try:
                tmp = len(self.best_solutions_over_bar)
            except:
                self.best_solutions_over_bar = [item for i in range(self.len) for item in self.data[i]]
        else:
            self.best_solutions_over_bar = [item for i in range(self.len) if self.max_rewards[i] > self.supervised_minimum_reward and len(self.best_solutions[i]) <= maxlen+1 for item in self.best_solutions[i]]
        return len(self.best_solutions_over_bar) > 10

    def next_supervised(self):
        cur_idxs = np.random.choice(len(self.best_solutions_over_bar), size=self.supervised_batch_size, replace=self.supervised_batch_size>len(self.best_solutions_over_bar))
        return self.prepare_supervised_batch(cur_idxs)

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        if self.weighted_sample:
            cur_idxs = np.random.choice(self.len, size=self.batch_size, p=softmax(-self.temperature * self.avg_rewards))
            return self.prepare_batch(cur_idxs)
        else:
            if self.i == self.len and self.eval:
                return [None]*21
            if self.i + self.batch_size > self.len:
                cur_idxs = self.idxs[self.i:]
                if self.eval:
                    self.i = self.len
                else:
                    if self.shuffle:
                        np.random.shuffle(self.idxs)
                    cur_idxs += self.idxs[:self.i + self.batch_size - self.len]
                    if self.i + self.batch_size - self.len >= self.len:
                        self.i = 0
                    else:
                        self.i = self.i + self.batch_size - self.len
            else:
                cur_idxs = self.idxs[self.i:self.i + self.batch_size]
                self.i += self.batch_size
            cur_data = self.prepare_batch(cur_idxs)
            return cur_data

    def update_weight_and_solutions(self, cur_idxs, rewards, tmp_solutions, update_solution):
        assert len(cur_idxs) == len(rewards)
        for i, (idx, reward) in enumerate(zip(cur_idxs, rewards)):
            self.avg_rewards[idx] = (self.avg_rewards[idx] * self.n_sampled[idx] + reward) / (self.n_sampled[idx] + 1)
            self.n_sampled[idx] += 1
            update_flag = (self.supervised_update_strictly_better and reward > self.max_rewards[idx]) \
                            or (not self.supervised_update_strictly_better and reward >= self.max_rewards[idx])
            self.max_rewards[idx] = max(self.max_rewards[idx], reward)
            if update_flag and update_solution:
                for ii in range(len(tmp_solutions[i])):
                    tmp_solutions[i][ii][0] = self.data[idx][0][0] # bert
                    tmp_solutions[i][ii][4] = len(tmp_solutions[i]) # w
                    tmp_solutions[i][ii][5] = self.data[idx][0][5] # query
                    tmp_solutions[i][ii][6] = self.data[idx][0][6] # query structure
                    tmp_solutions[i][ii][7] = self.data[idx][0][7] # question
                self.best_solutions[idx] = tmp_solutions[i]

    def prepare_batch(self, cur_idxs):
        '''
        x_scores: (n_data, center dim + offset dim)
        x_relations: (n_data, center dim + offset dim + bert dim)
        y_scores: (n_program, max_y_score_len)
        y_relations: (n_data, nrelation)
        mask_relations: (n_data)
        w_scores: (n_program)
        w_relations: (n_data)
        berts: (n_powerset, bert dim)
        edge_indices: (2, n_message_passing)
        softmax_edge_indices: (2, n_powerset)
        mask_relations_class: (n_data, nrelation)
        note that n_powerset != n_data * max_y_score_len.
            n_powerset = \sum_i 2^n_i (e.g. 2+8+4+16), n_data * max_y_score_len = 4*16,
            n_message_passing = \sum n_i * 2^(n_i - 1),
            n_program = args.batch_size
        '''
        x_scores, x_relations, y_scores, y_relations = [], [], [], []
        mask_relations, w_scores, w_relations, berts = [], [], [], []
        edge_indices, softmax_edge_indices = [], []
        mask_relations_class = []
        queries = []
        questions = []
        powersets = []
        value_edge_indices = []
        sampled_score, sampled_relation, branches_picked = [], [], []
        if self.reduce == 'max':
            additional_edge_indices = []
        n_data = 0
        n_powerset = 0
        n_program = 0
        n_all, n_negative_one, n_str = 0, 0, 0
        for i, idx in enumerate(cur_idxs):
            bert, _, _, action, w, query, query_structure, question, mask_relation_class = self.data[idx][0]
            # bert, center, offset, action, w, query, query_structure, question, mask_relation_class = self.data[idx][0]
            if type(query) == list:
                query = list2tuple(query)
            tmp_structure, _, _, _ = recursive_main(flatten(query), query_structure, 0, [], 0, 0)
            if self.query2box.geo == 'box':
                center = self.query2box.entity_embedding(flatten_list(tmp_structure))
                offset = torch.zeros_like(center)
            elif self.query2box.geo == 'rotate':
                embedding = self.query2box.entity_embedding(flatten_list(tmp_structure))
                center, offset = torch.chunk(embedding, 2, dim=1)
            assert len(center) == len(action[0])
            '''
            key difference here is that we further have query structure
            '''
            queries.append(query)
            questions.append(question)
            bert = np.array(bert) # (1, 768)
            if len(center) == 1:
                cur_powerset_len = 2
            else:
                cur_powerset_len = int(pow(2, len(center))-1)
            berts.append(np.tile(bert, (cur_powerset_len, 1)))
            powerset = self.all_powersets[len(center)]
            powersets.append(powerset)
            assert len(powerset) == cur_powerset_len
            target = tuple(np.arange(len(action[0]))[np.array(action[0]) == 1]) #(0,1) or (1) or (0,1,2)
            y = np.zeros(self.max_y_score_len)
            y[powerset.index(target)] = 1
            y_scores.append(y)

            for j in self.data[idx]:
                tmp_powerset = self.all_powersets[len(j[3][0])]
                tmp_action = j[3]
                tmp_target = tuple(np.arange(len(tmp_action[0]))[np.array(tmp_action[0]) == 1]) #(0,1) or (1) or (0,1,2)
                sampled_score.append(torch.LongTensor([tmp_powerset.index(tmp_target)]))
                branches_picked.append(tmp_target)
                if len(tmp_target) != 1:
                    assert type(tmp_action[-1]) == str, tmp_action[-1]
                    sampled_relation.append(None)
                else:
                    assert type(tmp_action[-1]) == int or type(tmp_action[-1]) == np.int32, tmp_action[-1]
                    sampled_relation.append([tmp_action[-1]])
            edge_index = deepcopy(self.all_edge_indices[len(center)])
            edge_index[0] += n_data
            edge_index[1] += n_powerset
            edge_indices.append(edge_index)
            if self.reduce == 'max' and len(center) == 1:
                additional_edge_indices.append(n_powerset)
            softmax_edge_index = np.stack([n_powerset + np.arange(cur_powerset_len), i * self.max_y_score_len + np.arange(cur_powerset_len)])
            softmax_edge_indices.append(softmax_edge_index)
            w_scores.append(1/w)
            value_edge_indices.append(n_powerset + cur_powerset_len - 1)

            n_program += 1
            n_data += len(center)
            n_powerset += cur_powerset_len

            if type(action[-1]) == str:
                no_relation = True
            else:
                no_relation = False
            for j in range(len(center)):
                if type(center) == np.ndarray:
                    x_scores.append(np.concatenate([center[j], offset[j]]))
                    x_relations.append(np.concatenate([center[j], offset[j], bert[0]]))
                elif type(center) == torch.Tensor:
                    x_scores.append(torch.cat([center[j], offset[j]], dim=-1))
                    x_relations.append(torch.cat([center[j], offset[j], torch.Tensor(bert[0]).to(center[j].device)], dim=-1))
                else:
                    assert False
                y_relations.append(action[-1] if type(action[-1]) == int else 0)
                mask_relations_class.append(mask_relation_class)
                if no_relation:
                    mask_relations.append(0.)
                else:
                    if self.eval: # a notable change here, it means that when evaluation, we should do all the necessary prediction
                        mask_relations.append(1. if action[0][j] == 1 else 0.)
                    else: # but when it is training, we need to filter those datapoints when mask_relation_class does not even have the ground truth task
                        mask_relations.append(1. if action[0][j] == 1 and mask_relation_class[action[-1]] else 0.)
                w_relations.append(1/w)

        if self.reduce == 'max':
            additional_edge_indices = np.stack([[n_data]*len(additional_edge_indices), additional_edge_indices])
            edge_indices.append(additional_edge_indices)
        x_scores = torch.stack(x_scores)
        x_relations = torch.stack(x_relations).to(x_scores.device)
        y_scores = torch.LongTensor(y_scores).to(x_scores.device)
        y_relations = F.one_hot(torch.LongTensor(y_relations), self.nrelation).to(x_scores.device)
        mask_relations = torch.Tensor(mask_relations).to(x_scores.device)
        w_scores = torch.Tensor(w_scores).to(x_scores.device)
        w_relations = torch.Tensor(w_relations).to(x_scores.device)
        berts = torch.Tensor(np.concatenate(berts, axis=0)).to(x_scores.device)
        edge_indices = torch.LongTensor(np.concatenate(edge_indices, axis=1)).to(x_scores.device)
        softmax_edge_indices = torch.LongTensor(np.concatenate(softmax_edge_indices, axis=1)).to(x_scores.device)
        mask_relations_class = torch.Tensor(mask_relations_class).bool().to(x_scores.device)

        return x_scores, x_relations, y_scores, y_relations, mask_relations, \
                w_scores, w_relations, berts, edge_indices, softmax_edge_indices, \
                n_program, self.max_y_score_len, mask_relations_class, queries, powersets, \
                value_edge_indices, sampled_score, sampled_relation, branches_picked, questions, \
                cur_idxs
