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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from lego.common.utils import list2tuple, tuple2list, eval_tuple
import numpy as np
import time


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.layer1_ln = nn.LayerNorm(self.hidden_dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, xs):
        l1 = F.relu(self.layer1_ln(self.layer1(xs)))
        l2 = self.layer2(l1)
        return l2

class BatchPowersetParser(nn.Module):

    def __init__(self, center_dim, offset_dim, bert_dim, hidden_dim, nrelation, reduce='max', deep_arch='identity', max_y_score_len=15, requires_vf=False):
        super(BatchPowersetParser, self).__init__()
        self.hidden_dim = hidden_dim
        self.nrelation = nrelation
        self.getScore = MLP(hidden_dim+bert_dim, hidden_dim, 1) # hidden_dim + bert_dim = 1568
        self.getRelation = MLP(center_dim+offset_dim+bert_dim, hidden_dim, nrelation) # center_dim + offset_dim + bert_dim = 2368
        self.requires_vf = requires_vf
        if requires_vf:
            self.getValue = MLP(hidden_dim+bert_dim, hidden_dim, 1) # hidden_dim + bert_dim = 1568
        self.reduce = reduce
        self.deep_arch = deep_arch

        if deep_arch == 'identity':
            self.getFeature = nn.Identity()
            self.processFeature = nn.Identity()
        elif deep_arch == 'deepsets':
            self.getFeature = MLP(center_dim+offset_dim, hidden_dim, hidden_dim) # center_dim + offset_dim = 1600
            self.processFeature = MLP(hidden_dim, hidden_dim, hidden_dim) # hidden_dim = 800

        if reduce == 'max':
            self.gatherFeature = torch_scatter.scatter_max
        else:
            assert False

        self.tfneginf = nn.Parameter(torch.Tensor(- np.ones((1, nrelation)) * float('inf')), requires_grad=False)

        self.t_data = 0
        self.t_fwd = 0
        self.t_loss = 0
        self.t_opt = 0

    '''
    x_scores: (n_data, center dim + offset dim)
    x_relations: (n_data, center dim + offset dim + bert dim)
    y_scores: (n_program, max_y_score_len)
    y_relations: (n_data, nrelation)
    mask_relations: (n_data)
    w_scores: (n_program)
    w_relations: (n_data)
    berts: (n_program, bert dim)
    edge_indices: (2, n_message_passing)
    softmax_edge_indices: (2, n_powerset)

    note that n_powerset != n_data * max_y_score_len.
        n_powerset = \sum_i 2^n_i (e.g. 2+8+4+16), n_data * max_y_score_len = 4*16,
        n_message_passing = \sum n_i * 2^(n_i - 1),
        n_program = args.batch_size
    '''
    def action_value(self, x_scores, x_relations, berts, edge_indices, softmax_edge_indices, value_edge_indices, n_program, max_y_score_len):
        feature = self.getFeature(x_scores)
        if self.reduce == 'max':
            feature = torch.cat([feature, torch.zeros(size=[1, feature.shape[1]]).to(feature.device)], dim=0)
        expanded_feature = feature[edge_indices[0]]
        gathered_feature, _ = self.gatherFeature(expanded_feature, edge_indices[1], dim_size=torch.max(edge_indices[1]).long()+1, dim=0)
        processed_feature = self.processFeature(gathered_feature)
        processed_feature = torch.cat([processed_feature, berts], dim=-1)
        full_set_processed_feature = processed_feature[value_edge_indices]
        value = self.getValue(full_set_processed_feature)
        score = self.getScore(processed_feature)
        assert not (score == 0).any()
        score, _ = torch_scatter.scatter_max(score, softmax_edge_indices[1], dim_size=n_program * max_y_score_len, dim=0) #! to update here
        score = torch.where(score != 0, score, torch.tensor(-float('inf')).to(score.device))
        score = torch.reshape(score, [n_program, max_y_score_len])

        relation = self.getRelation(x_relations)
        return score, relation, value

    def sample_action(self, score, relation, powersets, train_with_masking, query2box, x_scores, threshold, ent_out, mask_scores_class, transform, relation_selector, old=False, stochastic=True, epsilon=0., beam_size=1, base_ll=0., mask_mode='none'):
        transform_type, degree = eval_tuple(transform)
        if beam_size > 1:
            assert not stochastic
            assert mask_scores_class.shape[0] == 1
        assert not stochastic
        if stochastic:
            if np.random.rand() < epsilon:
                random_score = torch.cat([torch.ones([1, len(powersets[0])]).to(score.device) / len(powersets[0]), score[:, len(powersets[0]):]], dim=1) #! update for batch
                random_score = torch.where(mask_scores_class, random_score, torch.tensor(-float('inf')).to(score.device))
                if old:
                    score = torch.where(mask_scores_class, score, torch.tensor(-float('inf')).to(score.device))
                    sampled_score = torch.distributions.categorical.Categorical(logits=score).sample()
                else:
                    sampled_score = torch.distributions.categorical.Categorical(logits=score).sample()
            else:
                score = torch.where(mask_scores_class, score, torch.tensor(-float('inf')).to(score.device))
                sampled_score = torch.distributions.categorical.Categorical(logits=score).sample()
            sampled_scores_1 = [sampled_score]
            updated_lls_1 = [0]
        else:
            score = torch.where(mask_scores_class, score, torch.tensor(-float('inf')).to(score.device))

            assert score.shape[0] == 1
            n_legal_choices = torch.sum((score>-1000.).int()).item()
            sampled_scores_1 = torch.topk(score, min(n_legal_choices, beam_size))[1]
            ll = torch.log(F.softmax(score, dim=-1) + 1e-10)
            updated_lls_1 = [base_ll + ll[0][sampled_score].item() for sampled_score in sampled_scores_1[0]]

        sampled_scores, sampled_relations, branches_picked_list, mask_relations_class_list, updated_lls = [], [], [], [], []
        for i, (sampled_score, updated_ll) in enumerate(zip(sampled_scores_1, updated_lls_1)):
            branches_picked = powersets[0][sampled_score[0].item()] #! update for batch; (0,1) or (0) or ()
            if len(branches_picked) == 1:
                if train_with_masking:
                    if mask_mode == 'rs':
                        rel_logits = relation_selector(x_scores[branches_picked[0]].unsqueeze(0))
                        _, indices = torch.topk(rel_logits, threshold)
                        mask_relations_class = torch.zeros([len(x_scores), self.nrelation]).to(relation.device)
                        mask_relations_class[:, indices[0]] = 1
                        mask_relations_class = mask_relations_class.bool()
                        relation = torch.where(mask_relations_class, relation, torch.tensor(-float('inf')).to(relation.device))
                    else:
                        assert False, "%s not supported as mask_mode" % mask_mode
                else:
                    mask_relations_class = torch.ones([len(x_scores), self.nrelation]).bool().to(relation.device)
                if stochastic:
                    if np.random.rand() < epsilon:
                        random_relation = mask_relations_class.float()
                        sampled_relation = torch.distributions.categorical.Categorical(logits=random_relation[branches_picked[0]].unsqueeze(0)).sample().cpu().numpy()
                    else:
                        sampled_relation = torch.distributions.categorical.Categorical(logits=relation[branches_picked[0]].unsqueeze(0)).sample().cpu().numpy()
                    updated_ll = 0
                    sampled_scores.append(sampled_score)
                    sampled_relations.append(sampled_relation)
                    branches_picked_list.append(branches_picked)
                    mask_relations_class_list.append(mask_relations_class)
                    updated_lls.append(updated_ll)
                else:
                    n_legal_choices = torch.sum(mask_relations_class[branches_picked[0]].int()).item()
                    tmp_sampled_relations = torch.topk(relation[branches_picked[0]], min(n_legal_choices, beam_size))[1].unsqueeze(1).cpu().numpy().tolist()
                    ll = torch.log(F.softmax(relation[branches_picked[0]], dim=-1)+1e-10)
                    tmp_updated_lls = [updated_ll + ll[sampled_relation[0]].item() for sampled_relation in tmp_sampled_relations]

                    sampled_scores.extend([sampled_score]*len(tmp_sampled_relations))
                    sampled_relations.extend(tmp_sampled_relations)
                    branches_picked_list.extend([branches_picked]*len(tmp_sampled_relations))
                    mask_relations_class_list.extend([mask_relations_class]*len(tmp_sampled_relations))
                    updated_lls.extend(tmp_updated_lls)
            else:
                mask_relations_class = torch.ones_like(relation).bool()
                sampled_relation = None

                sampled_scores.append(sampled_score)
                sampled_relations.append(sampled_relation)
                branches_picked_list.append(branches_picked)
                mask_relations_class_list.append(mask_relations_class)
                updated_lls.append(updated_ll)

        return sampled_scores, sampled_relations, branches_picked_list, mask_relations_class_list, updated_lls

    def forward(self, x_scores, x_relations, berts, edge_indices, softmax_edge_indices, n_program, max_y_score_len):
        feature = self.getFeature(x_scores)
        if self.reduce == 'max':
            feature = torch.cat([feature, torch.zeros(size=[1, feature.shape[1]]).to(feature.device)], dim=0)
        expanded_feature = feature[edge_indices[0]]
        gathered_feature, _ = self.gatherFeature(expanded_feature, edge_indices[1], dim_size=torch.max(edge_indices[1]).long()+1, dim=0)
        processed_feature = self.processFeature(gathered_feature)
        processed_feature = torch.cat([processed_feature, berts], dim=-1)
        score = self.getScore(processed_feature)
        assert not (score == 0).any()
        score, _ = torch_scatter.scatter_max(score, softmax_edge_indices[1], dim_size=n_program * max_y_score_len, dim=0) #! to update here
        score = torch.where(score != 0, score, torch.tensor(-float('inf')).to(score.device))
        score = torch.reshape(score, [n_program, max_y_score_len])
        relation = self.getRelation(x_relations)

        return score, relation

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step, writer):
        optimizer.zero_grad()
        x_scores, x_relations, y_scores, y_relations, mask_relations, w_scores, w_relations, berts, edge_indices, softmax_edge_indices, n_program, max_y_score_len, mask_relations_class, question_indices, step_indices, noisy_mask_relations = train_iterator.next_supervised()

        if args.cuda:
            x_scores = x_scores.cuda()
            x_relations = x_relations.cuda()
            y_scores = y_scores.cuda()
            y_relations = y_relations.cuda()
            mask_relations = mask_relations.cuda()
            w_scores = w_scores.cuda()
            w_relations = w_relations.cuda()
            berts = berts.cuda()
            edge_indices = edge_indices.cuda()
            softmax_edge_indices = softmax_edge_indices.cuda()
            mask_relations_class = mask_relations_class.cuda()
            question_indices = question_indices.cuda()
            step_indices = step_indices.cuda()
            noisy_mask_relations = noisy_mask_relations.cuda()

        scores, relations = model(x_scores, x_relations, berts, edge_indices, softmax_edge_indices, n_program, max_y_score_len)

        score_loss = torch.nn.CrossEntropyLoss(reduction='none')(scores, y_scores)

        if args.train_with_masking:
            relations = torch.where(mask_relations_class, relations, torch.tensor(-float('inf')).to(relations.device))
            relation_loss = torch.nn.CrossEntropyLoss(reduction='none')(relations, y_relations) * mask_relations
        else:
            relation_loss = torch.nn.CrossEntropyLoss(reduction='none')(relations, y_relations) * mask_relations

        relation_loss = relation_loss[noisy_mask_relations]
        all_loss = score_loss + args.relation_coeff * relation_loss
        all_loss = torch_scatter.scatter_add(all_loss, step_indices[1], dim=0, dim_size=torch.max(step_indices[1])+1)
        loss, _ = torch_scatter.scatter_min(all_loss, question_indices[1], dim=0, dim_size=torch.max(question_indices[1])+1)

        loss = torch.mean(loss)
        score_loss = torch.mean(torch_scatter.scatter_min(torch_scatter.scatter_add(score_loss, step_indices[1], dim=0, dim_size=torch.max(step_indices[1])+1), question_indices[1], dim=0, dim_size=torch.max(question_indices[1])+1)[0])
        relation_loss = torch.mean(torch_scatter.scatter_min(torch_scatter.scatter_add(relation_loss, step_indices[1], dim=0, dim_size=torch.max(step_indices[1])+1), question_indices[1], dim=0, dim_size=torch.max(question_indices[1])+1)[0])

        loss.backward()

        optimizer.step()
        log = {
            'supervised_loss': loss.item(),
            'supervised_score_loss': score_loss.item(),
            'supervised_relation_loss': relation_loss.item(),
        }

        for metric in log:
            writer.add_scalar(metric, log[metric], step)

        return log
