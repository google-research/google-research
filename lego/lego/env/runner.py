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
import gym
from collections import defaultdict
from lego.common.utils import is_arithmetic_seq, list2tuple, tuple2list, query2structure, recursive_main, flatten
from copy import deepcopy
from math import ceil
from tqdm import tqdm
import torch
import torch.nn.functional as F

class ProgramEnv(gym.Env):
    def __init__(self, dataloader, latent_space_executor, answer, ent_out, force, dataset_name, reward_metrics='h3', traversed_answer=None, min_eps_len=-1, filter_3p=False, answer_type='query', keep_original_intersection_order=False, intersection_selector=None, intersection_threshold=-1, max_eps_len_list=None, delayed_evaluation=False, min_eps_len_list=None):
        super(ProgramEnv, self).__init__()
        self.dataloader = dataloader
        self.eval = dataloader.eval
        assert len(max_eps_len_list) == 4, "max number of topic entities are 4"
        self.max_eps_len_list = max_eps_len_list
        self.latent_space_executor = latent_space_executor
        self.answer = answer
        self.reward_metrics = reward_metrics
        self.ent_out = ent_out
        self.force = force
        self.n_query = len(dataloader)
        self.traversed_answer = traversed_answer
        self.dataset_name = dataset_name
        self.min_eps_len = min_eps_len
        if min_eps_len_list is not None:
            assert len(min_eps_len_list) == 4
            self.min_eps_len_list = min_eps_len_list
        else:
            self.min_eps_len_list = [self.min_eps_len]*4
        self.filter_3p = filter_3p
        self.answer_type = answer_type
        self.keep_original_intersection_order = keep_original_intersection_order
        self.intersection_selector = intersection_selector
        self.intersection_threshold = intersection_threshold
        if self.intersection_selector is not None:
            assert self.intersection_threshold > 0
        self.delayed_evaluation = delayed_evaluation

    def reset(self):
        self.x_scores, self.x_relations, y_scores, y_relations, \
            mask_relations, w_scores, w_relations, \
            self.berts, self.edge_indices, self.softmax_edge_indices, \
            self.n_program, self.max_y_score_len, mask_relations_class, \
            self.queries, self.powersets, self.value_edge_indices, \
            self.true_sampled_score, self.true_sampled_relation, \
            self.true_branches_picked, self.questions, self.cur_idxs = next(self.dataloader)
        self.n_branches = len(self.x_scores)
        if self.force:
            self.mask_scores_class = np.zeros([1, self.max_y_score_len])
            if self.n_branches == 1:
                self.mask_scores_class[:, 1] = 1
            else:
                self.mask_scores_class[:, 0] = 1
            self.mask_scores_class = torch.Tensor(self.mask_scores_class).bool().to(self.x_scores.device)
        else:
            self.mask_scores_class = torch.ones([1, self.max_y_score_len]).bool().to(self.x_scores.device)
        self.steps_so_far = 0
        if self.min_eps_len != -1:
            self.steps_minimum = self.min_eps_len
        else:
            if len(self.x_scores) == 1:
                self.steps_minimum = 2
            else:
                self.steps_minimum = len(self.x_scores) + 2
        self.max_eps_len = self.max_eps_len_list[len(self.x_scores)-1]
        self.steps_minimum = self.min_eps_len_list[len(self.x_scores)-1]
        assert self.steps_minimum < self.max_eps_len
        assert len(self.cur_idxs) == 1, "currently only 1 is supported because self.tmp_solutions only look at 1 query"
        self.tmp_solutions = [[] for _ in range(len(self.cur_idxs))]
        query_structure = tuple2list(deepcopy(self.queries[0]))
        query2structure(query_structure)
        self.query_structure = list2tuple(query_structure)
        tmp_structure = []
        query = flatten(self.queries[0])
        recursive_main(query, query_structure, 0, tmp_structure, 0, 0)
        self.tmp_structure = tmp_structure
        if self.delayed_evaluation:
            self.batch_query_embeddings, self.batch_test_queries, self.batch_test_questions, self.batch_tmp_structures = [], [], [], []

        return [self.x_scores,
                self.x_relations,
                self.berts,
                self.edge_indices,
                self.softmax_edge_indices,
                self.n_program,
                self.max_y_score_len,
                self.powersets,
                self.value_edge_indices,
                self.mask_scores_class,
                self.tmp_structure,
                self.steps_so_far]

    def update_weight_and_solutions(self, rewards, update_solution):
        self.dataloader.update_weight_and_solutions(self.cur_idxs, rewards, self.tmp_solutions, update_solution)

    def set_state(self, x_scores,
                x_relations,
                berts,
                edge_indices,
                softmax_edge_indices,
                n_program,
                max_y_score_len,
                powersets,
                value_edge_indices,
                mask_scores_class,
                tmp_structure,
                steps_so_far):
        self.x_scores = x_scores
        self.x_relations = x_relations
        self.berts = berts
        self.edge_indices = edge_indices
        self.softmax_edge_indices = softmax_edge_indices
        self.n_program = n_program
        self.max_y_score_len = max_y_score_len
        self.powersets = powersets
        self.value_edge_indices = value_edge_indices
        self.mask_scores_class = mask_scores_class
        self.tmp_structure = deepcopy(tmp_structure)
        self.steps_so_far = steps_so_far

    def push_to_current_solution_buffer(self, sampled_score, sampled_relation, branches_picked, mask_relations_class):
        centers, offsets = torch.chunk(self.x_scores, 2, axis=-1)
        action = [[0 for _ in range(len(centers))]]
        for branch in branches_picked:
            action[0][branch] = 1
        if len(branches_picked) == 1:
            assert len(sampled_relation) == 1
            action.append(sampled_relation[0])
        elif len(branches_picked) == 0:
            action.append('Stop')
        elif len(branches_picked) > 1:
            action.append('Intersection')
        self.tmp_solutions[0].append([None, centers.numpy(), offsets.numpy(), action, None, None, None, None, mask_relations_class[0].numpy().astype(np.float32)])

    def batch_evaluation(self):
        all_rewards, all_logs = [], []
        test_batch_size = 64
        for i in tqdm(range(ceil(len(self.batch_query_embeddings)/test_batch_size))):
            rewards, logs = self.latent_space_executor.quick_evaluate_query_batch(self.batch_query_embeddings[i*test_batch_size:(i+1)*test_batch_size], self.batch_test_queries[i*test_batch_size:(i+1)*test_batch_size], self.batch_test_questions[i*test_batch_size:(i+1)*test_batch_size], self.answer, answer_type=self.answer_type)
            all_rewards.extend(rewards)
            all_logs.extend(logs)
        return all_rewards, all_logs, self.batch_tmp_structures

    def step(self, sampled_score, sampled_relation, branches_picked, mask_relations_class=None):
        self.steps_so_far += 1

        if (self.steps_so_far == self.max_eps_len) or (len(branches_picked) == 0 and self.steps_so_far < self.steps_minimum):
            assert not self.force
            return [self.x_scores,
                        self.x_relations,
                        self.berts,
                        self.edge_indices,
                        self.softmax_edge_indices,
                        self.n_program,
                        self.max_y_score_len,
                        self.powersets,
                        self.value_edge_indices,
                        self.mask_scores_class,
                        self.tmp_structure,
                        self.steps_so_far], \
                    -0.1, \
                    True, \
                    defaultdict(float)

        if len(branches_picked) == 0:
            assert len(self.x_scores) == len(self.queries)
            assert len(self.tmp_structure) == 1
            if self.delayed_evaluation:
                self.batch_test_questions.extend(self.questions)
                self.batch_test_queries.extend(self.queries)
                self.batch_query_embeddings.append(self.x_scores)
                self.batch_tmp_structures.append(deepcopy(self.tmp_structure))
                return None, None, True, defaultdict(float)

            _, all_rewards = self.latent_space_executor.quick_evaluate_query_batch([self.x_scores], self.queries, self.questions, self.answer, answer_type=self.answer_type)
            if self.dataset_name == 'metaqa':
                all_traversed_rewards = self.latent_space_executor.evaluate_query(self.latent_space_executor, self.dataset_name, self.x_scores, self.queries, self.questions, self.reward_metrics, self.traversed_answer, self.traversed_answer, self.ent_out, mode="traverse-", filter_3p=self.filter_3p, answer_type=self.answer_type)
                all_rewards.update(all_traversed_rewards)
            all_rewards = all_rewards[0]
            return [self.x_scores,
                        self.x_relations,
                        self.berts,
                        self.edge_indices,
                        self.softmax_edge_indices,
                        self.n_program,
                        self.max_y_score_len,
                        self.powersets,
                        self.value_edge_indices,
                        self.mask_scores_class,
                        self.tmp_structure,
                        self.steps_so_far], \
                    all_rewards[self.reward_metrics], \
                    True, \
                    all_rewards

        if len(branches_picked) == 1:
            idx = branches_picked[0]
            rel_idx = sampled_relation[0]
            if len(self.tmp_structure[idx]) == 1:
                self.tmp_structure[idx].append([rel_idx])
            else:
                all_rel_flag = True
                for rel in self.tmp_structure[idx][-1]:
                    if type(rel) == list:
                        all_rel_flag = False
                        break
                if all_rel_flag:
                    self.tmp_structure[idx][-1].append(rel_idx)
                else:
                    self.tmp_structure[idx] = [self.tmp_structure[idx], [rel_idx]]
            if self.latent_space_executor.geo == 'box':
                center, offset = torch.chunk(self.x_scores[idx], 2)
                center = torch.clone(center)
                offset = torch.clone(offset)

                r_center = self.latent_space_executor.relation_embedding(rel_idx)
                r_offset = self.latent_space_executor.offset_embedding(rel_idx)
                center += r_center
                offset += r_offset
                self.x_scores = torch.cat([self.x_scores[:idx], torch.unsqueeze(torch.cat([center, offset], 0), 0), self.x_scores[idx+1:]], 0)
                self.x_relations = torch.cat([self.x_relations[:idx], torch.unsqueeze(torch.cat([center, offset, self.berts[0]], 0), 0), self.x_relations[idx+1:]], 0)
            elif self.latent_space_executor.geo == 'rotate':
                re_embedding, im_embedding = torch.chunk(self.x_scores[idx], 2)
                r_embedding = self.latent_space_executor.relation_embedding(rel_idx)
                phase_relation = r_embedding/(self.latent_space_executor.embedding_range/3.1415926)
                re_relation = torch.cos(phase_relation)
                im_relation = torch.sin(phase_relation)
                new_re_embedding = re_embedding * re_relation - im_embedding * im_relation
                new_im_embedding = re_embedding * im_relation + im_embedding * re_relation
                re_embedding = new_re_embedding
                im_embedding = new_im_embedding
                self.x_scores = torch.cat([self.x_scores[:idx], torch.unsqueeze(torch.cat([re_embedding, im_embedding], 0), 0), self.x_scores[idx+1:]], 0)
                self.x_relations = torch.cat([self.x_relations[:idx], torch.unsqueeze(torch.cat([re_embedding, im_embedding, self.berts[0]], 0), 0), self.x_relations[idx+1:]], 0)

        if len(branches_picked) > 1:
            # assert self.latent_space_executor.geo in ['box']
            if self.latent_space_executor.geo == 'box':
                center, offset = torch.chunk(self.x_scores[list(branches_picked)], 2, -1)
                center = torch.unsqueeze(center, 1)
                offset = torch.unsqueeze(offset, 1)
                center = self.latent_space_executor.center_net(center)
                offset = self.latent_space_executor.offset_net(offset)
            elif self.latent_space_executor.geo == 'rotate':
                embedding = self.x_scores[list(branches_picked)]
                embedding = torch.unsqueeze(embedding, 1)
                embedding = self.latent_space_executor.center_net(embedding)
                center, offset = torch.chunk(embedding, 2, -1)

            if len(branches_picked) == len(self.x_scores): # meaning that this step takes intersection of all branches and will only result in one branch left
                self.x_scores = torch.cat([center, offset], -1)
                self.powersets = [self.dataloader.all_powersets[len(self.x_scores)]]
                self.value_edge_indices = [len(self.powersets[0]) - 1]
                self.x_relations = torch.cat([center, offset, torch.unsqueeze(self.berts[0], 0)], -1)
                self.berts = (torch.unsqueeze(self.berts[0], 0)).repeat([2, 1]) # 2 means that since we only have one branch now, then the size of the powerset will be 2.
                if self.dataloader.reduce == 'max':
                    self.edge_indices = torch.LongTensor([[0, 1], [1, 0]]).to(self.x_scores.device)
                else:
                    self.edge_indices = torch.LongTensor([[0], [1]]).to(self.x_scores.device)
                self.softmax_edge_indices = torch.LongTensor([[0, 1], [0, 1]]).to(self.x_scores.device)
                self.tmp_structure = [self.tmp_structure]
            else:
                if self.keep_original_intersection_order and is_arithmetic_seq(branches_picked, 1):
                    branches_left = [i for i in range(len(self.x_scores)) if i < list(branches_picked)[0]]
                    branches_right = [i for i in range(len(self.x_scores)) if i > list(branches_picked)[-1]]
                    if len(branches_left) > 0 and len(branches_right) > 0:
                        self.x_scores = torch.cat([self.x_scores[branches_left], torch.cat([center, offset], -1), self.x_scores[branches_right]], 0)
                        self.tmp_structure = [self.tmp_structure[i] for i in branches_left] + [[self.tmp_structure[i] for i in branches_picked]] + [self.tmp_structure[i] for i in branches_right]
                    elif len(branches_left) > 0:
                        self.x_scores = torch.cat([self.x_scores[branches_left], torch.cat([center, offset], -1)], 0)
                        self.tmp_structure = [self.tmp_structure[i] for i in branches_left] + [[self.tmp_structure[i] for i in branches_picked]]
                    elif len(branches_right) > 0:
                        self.x_scores = torch.cat([torch.cat([center, offset], -1), self.x_scores[branches_right]], 0)
                        self.tmp_structure = [[self.tmp_structure[i] for i in branches_picked]] + [self.tmp_structure[i] for i in branches_right]
                    else:
                        assert False
                else:
                    branches_left = [i for i in range(len(self.x_scores)) if i not in list(branches_picked)]
                    self.x_scores = torch.cat([self.x_scores[branches_left], torch.cat([center, offset], -1)], 0)
                    self.tmp_structure = [self.tmp_structure[i] for i in branches_left] + [[self.tmp_structure[i] for i in branches_picked]]
                self.powersets = [self.dataloader.all_powersets[len(self.x_scores)]]
                self.value_edge_indices = [len(self.powersets[0]) - 1]
                self.x_relations = torch.cat([self.x_scores, torch.unsqueeze(self.berts[0], 0).repeat([len(self.x_scores), 1])], -1)
                self.berts = torch.unsqueeze(self.berts[0], 0).repeat([len(self.powersets[0]), 1])
                self.edge_indices = torch.LongTensor(self.dataloader.all_edge_indices[len(self.x_scores)]).to(self.x_scores.device)
                self.softmax_edge_indices = torch.LongTensor(np.stack([np.arange(len(self.powersets[0])), np.arange(len(self.powersets[0]))])).to(self.x_scores.device)

            self.n_program = 1 # always 1

        new_obs = [self.x_scores,
                    self.x_relations,
                    self.berts,
                    self.edge_indices,
                    self.softmax_edge_indices,
                    self.n_program,
                    self.max_y_score_len,
                    self.powersets,
                    self.value_edge_indices]

        if self.force:
            if self.n_branches > self.steps_so_far:
                self.mask_scores_class = np.zeros([1, self.max_y_score_len])
                self.mask_scores_class[:, self.steps_so_far] = 1
            elif self.steps_so_far == self.max_eps_len - 3 and len(self.x_scores) > 1:
                self.mask_scores_class = np.zeros([1, self.max_y_score_len])
                if self.intersection_selector is not None:
                    logits = F.sigmoid(self.intersection_selector(torch.unsqueeze(self.x_scores, 1)))
                    if logits < self.intersection_threshold:
                        return [self.x_scores,
                                    self.x_relations,
                                    self.berts,
                                    self.edge_indices,
                                    self.softmax_edge_indices,
                                    self.n_program,
                                    self.max_y_score_len,
                                    self.powersets,
                                    self.value_edge_indices,
                                    self.mask_scores_class,
                                    self.tmp_structure,
                                    self.steps_so_far], \
                                -0.1, \
                                True, \
                                defaultdict(float)
                    else:
                        self.mask_scores_class[:, len(self.powersets[0]) - 1] = 1
                else:
                    self.mask_scores_class[:, len(self.powersets[0]) - 1] = 1
            elif self.steps_so_far < self.steps_minimum - 1 and len(self.x_scores) == 1: # only take effect for metaqa
                self.mask_scores_class = np.zeros([1, self.max_y_score_len])
                self.mask_scores_class[:, 1] = 1 # pick the first branch and extend one relation
            elif self.steps_so_far == self.max_eps_len - 2:
                assert len(self.x_scores) == 1, "structure: {}, max_eps_len: {}, min_eps_len: {}".format(self.tmp_structure, self.max_eps_len, self.steps_minimum)
                self.mask_scores_class = np.zeros([1, self.max_y_score_len])
                self.mask_scores_class[:, 0] = 1
            else:
                self.mask_scores_class = np.zeros([1, self.max_y_score_len])
                self.mask_scores_class[:, :len(self.powersets[0])] = 1
                if len(self.x_scores) > 1 and self.intersection_selector is not None:
                    for idx, branches in enumerate(self.powersets[0]):
                        if len(branches) == 1:
                            continue
                        logits = F.sigmoid(self.intersection_selector(torch.unsqueeze(self.x_scores[list(branches)], 1)))
                        if logits < self.intersection_threshold:
                            self.mask_scores_class[:, idx] = 0
            self.mask_scores_class = torch.LongTensor(self.mask_scores_class).bool().to(self.x_scores.device)
        else:
            self.mask_scores_class = torch.ones([1, self.max_y_score_len]).bool().to(self.x_scores.device)
        new_obs.extend([self.mask_scores_class, self.tmp_structure, self.steps_so_far])
        return new_obs, 0., False, {}
