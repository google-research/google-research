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

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import defaultdict

from smore.common.torchext.ext_ops import box_dist_in, box_dist_out, rotate_dist
from smore.models.kg_reasoning import KGReasoning
from smore.models.box import BoxReasoning
from smore.models.rotate import RotateReasoning


class BaseExecutor(KGReasoning):
    def emb_forward(self, batch_queries_dict, batch_idxs_dict, device="cuda:0"):
        all_center_embeddings = []
        all_offset_embeddings = []
        all_idxs = []
        for query_structure in batch_queries_dict:
            query_embedding, _ = self.embed_query(batch_queries_dict[query_structure],
                                                                            query_structure,
                                                                            0, device)
            center_embedding, offset_embedding = query_embedding
            assert center_embedding.shape[1] == 1 and offset_embedding.shape[1] == 1
            all_center_embeddings.append(center_embedding.squeeze(1))
            all_offset_embeddings.append(offset_embedding.squeeze(1))
            all_idxs.extend(batch_idxs_dict[query_structure])

        all_center_embeddings = torch.cat(all_center_embeddings, dim=0)
        all_offset_embeddings = torch.cat(all_offset_embeddings, dim=0)

        return [all_center_embeddings, all_offset_embeddings], all_idxs

    def intersection_forward(self, batch_queries_dict, batch_idxs_dict, query_structures_unflatten, intersection_selector):
        with torch.no_grad():
            all_embeddings, all_idxs = self.emb_forward(batch_queries_dict, batch_idxs_dict)
        inverse_idxs = np.arange(len(all_idxs))
        inverse_idxs[all_idxs] = np.arange(len(all_idxs))
        all_embeddings = torch.cat(all_embeddings, -1)
        all_embeddings = all_embeddings[inverse_idxs]

        cnt = 0
        all_idxs, all_logits = [], []
        batch_queries_dict = defaultdict(list)
        batch_idxs_dict = defaultdict(list)
        for i in range(len(query_structures_unflatten)):
            batch_queries_dict[len(query_structures_unflatten[i])].append(all_embeddings[cnt:cnt+len(query_structures_unflatten[i])])
            batch_idxs_dict[len(query_structures_unflatten[i])].append(i)
            cnt += len(query_structures_unflatten[i])
        assert cnt == len(all_embeddings)

        for leng in batch_queries_dict:
            batch_queries_dict[leng] = torch.stack(batch_queries_dict[leng], 0).transpose(0, 1)
            all_idxs.extend(batch_idxs_dict[leng])
            cur_logits = F.sigmoid(intersection_selector(batch_queries_dict[leng]))
            all_logits.append(cur_logits)
        return all_logits, all_idxs

    def quick_evaluate_query_batch(self, query_embeddings, test_queries, test_questions, fn_answers, mode='', answer_type='query'):
        fn_logs = []
        fn_rewards = []

        query_embeddings = torch.cat(query_embeddings, 0).unsqueeze(1)
        entity_embeddings = self.entity_embedding(None).unsqueeze(0)
        distance = self.cal_distance(entity_embeddings, query_embeddings)
        negative_logits = self.gamma - distance

        for idx, (negative_logit, query, question) in enumerate(zip(negative_logits, test_queries, test_questions)):
            negative_logit = negative_logit.unsqueeze(0)
            if answer_type == 'query':
                fn_answer = fn_answers[query]
            elif answer_type == 'question':
                fn_answer = fn_answers[question]
            num_fn = len(fn_answer)

            logits_topk, indices_topk = torch.topk(negative_logit, k=100)
            hits1max = float(indices_topk[0][0].item() in fn_answer)

            all_rank = []
            for ans in fn_answer:
                pos = torch.where(indices_topk[0] == ans)[0]
                all_rank.append(pos)
            all_rank = torch.cat(all_rank, dim=0).cpu().numpy()

            if len(all_rank) == 0:
                hits10 = 0
            else:
                all_rank.sort()
                all_rank = all_rank - np.arange(len(all_rank)) + 1
                hits10 = np.sum(all_rank <= 10)/len(fn_answer)

            fn_logs.append({
                mode+'mrr': 0,
                mode+'h1m': hits1max,
                mode+'h10': hits10,
                mode+'num_answer': num_fn,
                mode+'logits_topk': logits_topk.detach().cpu().numpy(),
                mode+'indices_topk': indices_topk.cpu().numpy(),
                mode+'all_ans': fn_answer
            })
            fn_rewards.append(hits1max)

        return fn_rewards, fn_logs


class BoxExecutor(BoxReasoning, BaseExecutor):
    def cal_distance(self, entity_embeddings, query_embeddings):
        query_center_embeddings, query_offset_embeddings = torch.chunk(query_embeddings, 2, dim=-1)
        query_center_embeddings = query_center_embeddings.contiguous()
        query_offset_embeddings = query_offset_embeddings.contiguous()
        d1 = box_dist_out(entity_embeddings, query_center_embeddings, query_offset_embeddings)
        d2 = box_dist_in(entity_embeddings, query_center_embeddings, query_offset_embeddings)
        distance = d1 + self.cen * d2
        return distance

class RotateExecutor(RotateReasoning, BaseExecutor):
    def cal_distance(self, entity_embeddings, query_embeddings):
        re_q, im_q = torch.chunk(query_embeddings, 2, dim=-1)
        re_q = re_q.contiguous()
        im_q = im_q.contiguous()
        distance = rotate_dist(entity_embeddings, re_q, im_q)
        return distance
