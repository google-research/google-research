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

import pickle
import os
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import argparse
import json
import numpy as np
import torch

from lego.common.utils import set_logger, parse_time, set_global_seed, query_name_dict, name_query_dict, list2tuple, tuple2list, construct_graph, eval_tuple, get_optimizer, save_model, log_and_write_metrics, group_queries, load_lse_checkpoint, set_train_mode, flatten
from lego.model import get_lse_model
from smore.common.config import parse_args


def achieve_answer(query, ent_in, ent_out):
    assert type(query[-1]) == list
    all_relation_flag = True
    for ele in query[-1]:
        if (type(ele) != int) or (ele == -1):
            all_relation_flag = False
            break
    if all_relation_flag:
        if type(query[0]) == int:
            ent_set = set([query[0]])
        else:
            ent_set = achieve_answer(query[0], ent_in, ent_out)
        for i in range(len(query[-1])):
            if query[-1][i] == -2:
                ent_set = set(range(len(ent_in))) - ent_set
            else:
                ent_set_traverse = set()
                for ent in ent_set:
                    if query[-1][i] in ent_out[ent]:
                        ent_set_traverse = ent_set_traverse.union(ent_out[ent][query[-1][i]])
                ent_set = ent_set_traverse
    else:
        ent_set = achieve_answer(query[0], ent_in, ent_out)
        union_flag = False
        if len(query[-1]) == 1 and query[-1][0] == -1:
            union_flag = True
        for i in range(1, len(query)):
            if not union_flag:
                ent_set = ent_set.intersection(achieve_answer(query[i], ent_in, ent_out))
            else:
                if i == len(query) - 1:
                    continue
                ent_set = ent_set.union(achieve_answer(query[i], ent_in, ent_out))
    return ent_set

def recursive_structure(queries):
    all_relation_flag = True
    for ele in queries[-1]:
        if type(ele) == list:
            all_relation_flag = False
            break
    if all_relation_flag:
        if type(queries[0]) == int:
            queries[0] = 'e'
        else:
            recursive_structure(queries[0])
        for i in range(len(queries[-1])):
            queries[-1][i] = 'r'
    else:
        for i in range(len(queries)):
            recursive_structure(queries[i])

def recursive_main(queries, query_structure, idx, tmp_structure, ent_id, num_ent):
    # return all the leaf nodes in a big query structure
    all_relation_flag = True
    for ele in query_structure[-1]:
        if ele not in ['r', 'n']:
            all_relation_flag = False
            break
    if all_relation_flag:
        if query_structure[0] == 'e':
            tmp_structure.append([queries[idx]])
            idx += 1
        else:
            tmp_structure, idx, ent_id, num_ent = recursive_main(queries, query_structure[0], idx, tmp_structure, ent_id, num_ent)
        for i in range(len(query_structure[-1])):
            idx += 1
    else:
        for i in range(len(query_structure)):
            tmp_structure, idx, _, num_ent = recursive_main(queries, query_structure[i], idx, tmp_structure, ent_id + i, num_ent)
    return tmp_structure, idx, ent_id, num_ent

def recursive_embed_first_step(queries, query_structure, idx, tmp_structure, ent_id, num_ent, tmp_center, tmp_offset, question, query_unflattened):
    # used to find the step by step embedding representation
    # assert type(query[-1]) == list
    global num_inter
    global num_path_embedding, num_path_embedding_with_relation, num_path_embedding_without_relation
    global num_path_traversal, num_path_traversal_with_relation, num_path_traversal_without_relation
    global num_path, num_path_with_relation, num_path_without_relation
    global num_relations, num_relations_embedding, num_relations_traversal
    all_relation_flag = True
    for ele in query_structure[-1]:
        if ele not in ['r', 'n']:
            all_relation_flag = False
            break
    if all_relation_flag:
        if query_structure[0] == 'e':
            ent_set = set([queries[idx]])
            idx += 1
            for i in range(len(query_structure[-1])):
                if i == 0:
                    if latent_space_executor.geo == 'rotate':
                        r_embedding = latent_space_executor.relation_embedding(queries[idx])
                        phase_relation = r_embedding/(latent_space_executor.embedding_range/3.1415926)
                        re_relation = torch.cos(phase_relation)
                        im_relation = torch.sin(phase_relation)
                        new_center = tmp_center[ent_id] * re_relation - tmp_offset[ent_id] * im_relation
                        new_offset = tmp_center[ent_id] * im_relation + tmp_offset[ent_id] * re_relation
                        tmp_center = torch.cat([tmp_center[:ent_id], new_center.unsqueeze(0), tmp_center[ent_id+1:]], 0)
                        tmp_offset = torch.cat([tmp_offset[:ent_id], new_offset.unsqueeze(0), tmp_offset[ent_id+1:]], 0)
                        all_centers[question][query_unflattened].append(tmp_center)
                        all_offsets[question][query_unflattened].append(tmp_offset)
                    elif latent_space_executor.geo == 'box':
                        r_embedding = latent_space_executor.relation_embedding(queries[idx])
                        r_offset_embedding = latent_space_executor.offset_embedding(queries[idx])
                        # print(r_embedding.shape, r_offset_embedding.shape, tmp_center.shape, ent_id, tmp_offset.shape)
                        # pdb.set_trace()
                        tmp_center = torch.cat([tmp_center[:ent_id], (tmp_center[ent_id] + r_embedding).unsqueeze(0), tmp_center[ent_id+1:]], 0)
                        tmp_offset = torch.cat([tmp_offset[:ent_id], (tmp_offset[ent_id] + r_offset_embedding).unsqueeze(0), tmp_offset[ent_id+1:]], 0)
                        all_centers[question][query_unflattened].append(tmp_center)
                        all_offsets[question][query_unflattened].append(tmp_offset)

                    if len(tmp_structure[ent_id]) == 1:
                        tmp_structure[ent_id].append([queries[idx]])
                    else:
                        assert False
                        tmp_structure[ent_id][1].append(queries[idx])
                    all_structures[question][query_unflattened].append(deepcopy(tmp_structure))
                    tmp_order = [[0 for _ in range(len(tmp_structure))], queries[idx]]
                    tmp_order[0][ent_id] = 1
                    all_orders[question][query_unflattened].append(tmp_order)
                idx += 1
            ent_id += 1
        else:
            tmp_structure, idx, ent_id, num_ent, tmp_center, tmp_offset, ent_set = recursive_embed_first_step(queries, query_structure[0], idx, tmp_structure, ent_id, num_ent, tmp_center, tmp_offset, question, query_unflattened)
            # tmp_structure[ent_id] = [tmp_structure[ent_id]]
            for i in range(len(query_structure[-1])):
                idx += 1
    else:
        for i in range(len(query_structure)):
            tmp_structure, idx, ent_id, num_ent, tmp_center, tmp_offset, ent_set_inter = recursive_embed_first_step(queries, query_structure[i], idx, tmp_structure, ent_id, num_ent, tmp_center, tmp_offset, question, query_unflattened)
            if i == 0:
                ent_set = ent_set_inter
            else:
                ent_set = ent_set.intersection(ent_set_inter)

    return tmp_structure, idx, ent_id, num_ent, tmp_center, tmp_offset, ent_set

def recursive_embed_second_step(queries, query_structure, idx, tmp_structure, ent_id, num_ent, tmp_center, tmp_offset, question, query_unflattened):
    # used to find the step by step embedding representation
    # assert type(query[-1]) == list
    global num_inter
    global num_path_embedding, num_path_embedding_with_relation, num_path_embedding_without_relation
    global num_path_traversal, num_path_traversal_with_relation, num_path_traversal_without_relation
    global num_path, num_path_with_relation, num_path_without_relation
    global num_relations, num_relations_embedding, num_relations_traversal
    all_relation_flag = True
    for ele in query_structure[-1]:
        if ele not in ['r', 'n']:
            all_relation_flag = False
            break
    if all_relation_flag:
        if query_structure[0] == 'e':
            ent_set = set([queries[idx]])
            idx += 1
        else:
            tmp_structure, idx, ent_id, num_ent, tmp_center, tmp_offset, ent_set = recursive_embed_second_step(queries, query_structure[0], idx, tmp_structure, ent_id, num_ent, tmp_center, tmp_offset, question, query_unflattened)
            tmp_structure[ent_id] = [tmp_structure[ent_id]]
        for i in range(len(query_structure[-1])):
            if i != 0 or query_structure[0] != 'e':
                if latent_space_executor.geo == 'rotate':
                    r_embedding = latent_space_executor.relation_embedding(queries[idx])
                    phase_relation = r_embedding/(latent_space_executor.embedding_range/3.1415926)
                    re_relation = torch.cos(phase_relation)
                    im_relation = torch.sin(phase_relation)
                    new_center = tmp_center[ent_id] * re_relation - tmp_offset[ent_id] * im_relation
                    new_offset = tmp_center[ent_id] * im_relation + tmp_offset[ent_id] * re_relation
                    tmp_center = torch.cat([tmp_center[:ent_id], new_center.unsqueeze(0), tmp_center[ent_id+1:]], 0)
                    tmp_offset = torch.cat([tmp_offset[:ent_id], new_offset.unsqueeze(0), tmp_offset[ent_id+1:]], 0)
                    all_centers[question][query_unflattened].append(tmp_center)
                    all_offsets[question][query_unflattened].append(tmp_offset)
                elif latent_space_executor.geo == 'box':
                    r_embedding = latent_space_executor.relation_embedding(queries[idx])
                    r_offset_embedding = latent_space_executor.offset_embedding(queries[idx])
                    tmp_center = torch.cat([tmp_center[:ent_id], (tmp_center[ent_id] + r_embedding).unsqueeze(0), tmp_center[ent_id+1:]], 0)
                    tmp_offset = torch.cat([tmp_offset[:ent_id], (tmp_offset[ent_id] + r_offset_embedding).unsqueeze(0), tmp_offset[ent_id+1:]], 0)
                    all_centers[question][query_unflattened].append(tmp_center)
                    all_offsets[question][query_unflattened].append(tmp_offset)

                if len(tmp_structure[ent_id]) == 1:
                    assert query_structure[0] != 'e'
                    tmp_structure[ent_id].append([queries[idx]])
                else:
                    tmp_structure[ent_id][1].append(queries[idx])
                all_structures[question][query_unflattened].append(deepcopy(tmp_structure))
                tmp_order = [[0 for _ in range(len(tmp_structure))], queries[idx]]
                tmp_order[0][ent_id] = 1
                all_orders[question][query_unflattened].append(tmp_order)
            idx += 1

    else:
        for i in range(len(query_structure)):
            tmp_structure, idx, _, num_ent, tmp_center, tmp_offset, ent_set_inter = recursive_embed_second_step(queries, query_structure[i], idx, tmp_structure, ent_id + i, num_ent, tmp_center, tmp_offset, question, query_unflattened)
            if i == 0:
                ent_set = ent_set_inter
            else:
                ent_set = ent_set.intersection(ent_set_inter)
        if latent_space_executor.geo == 'box':
            new_center = latent_space_executor.center_net(tmp_center[ent_id:ent_id+len(query_structure)])
            new_offset = latent_space_executor.offset_net(tmp_offset[ent_id:ent_id+len(query_structure)])
        elif latent_space_executor.geo == 'rotate':
            embedding_list = []
            for i in range(ent_id, ent_id+len(query_structure)):
                embedding_list.append(torch.cat([tmp_center[i], tmp_offset[i]], dim=-1))
            embedding = latent_space_executor.center_net(torch.stack(embedding_list))
            new_center, new_offset = torch.chunk(embedding, 2, dim=0)

        tmp_center = torch.cat([tmp_center[:ent_id], new_center.unsqueeze(0), tmp_center[ent_id+len(query_structure):]], 0)
        tmp_offset = torch.cat([tmp_offset[:ent_id], new_offset.unsqueeze(0), tmp_offset[ent_id+len(query_structure):]], 0)
        all_centers[question][query_unflattened].append(tmp_center)
        all_offsets[question][query_unflattened].append(tmp_offset)

        tmp_structure[ent_id] = [tmp_structure[ent_id+i] for i in range(len(query_structure))]
        tmp_order = [[0 for _ in range(len(tmp_structure))], 'Intersection']
        for i in range(len(query_structure)):
            tmp_order[0][ent_id + i] = 1
        all_orders[question][query_unflattened].append(tmp_order)
        del tmp_structure[ent_id+1:ent_id+len(query_structure)]
        num_ent -= len(query_structure) - 1
        all_structures[question][query_unflattened].append(deepcopy(tmp_structure))

    return tmp_structure, idx, ent_id, num_ent, tmp_center, tmp_offset, ent_set

def dl():
    return defaultdict(list)


parser = parse_args()

parser.add_argument('--split', type=str, default=None)
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=0)
parser.add_argument('--save_start_end_idx', action='store_true', default=False)
parser.add_argument('--target_k', type=int, default=-1)
parser.add_argument('--target_rank', type=int, default=-1)
parser.add_argument('--target_mrr', type=float, default=-1)
parser.add_argument('--second_round_k', type=int, default=-1)
parser.add_argument('--question_path', type=str, default=None)
parser.add_argument('--candidate_path', type=str, default=None)

args = parser.parse_args()
set_train_mode(args)
assert args.target_rank != -1 and args.target_mrr != -1

query_name_dict = {('e',('r',)): '1p',
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up'
                    }

all_structures = defaultdict(dl)
all_orders = defaultdict(dl)
all_centers = defaultdict(dl)
all_offsets = defaultdict(dl)


with open('%s/stats.txt'%args.data_path) as f:
    entrel = f.readlines()
    nentity = int(entrel[0].split(' ')[-1])
    nrelation = int(entrel[1].split(' ')[-1])

print ("Loading model from %s" % args.checkpoint_path)
latent_space_executor = get_lse_model(args)
load_lse_checkpoint(args, latent_space_executor)

test_queries = defaultdict(set)
test_tp_answers = defaultdict(set)
test_fp_answers = defaultdict(set)
test_fn_answers = defaultdict(set)

ent_in, ent_out, _ = construct_graph(args.data_path, ['train_indexified.txt'])
ent2id = pickle.load(open(os.path.join(args.data_path, "ent2id.pkl"), 'rb'))
id2ent = pickle.load(open(os.path.join(args.data_path, "id2ent.pkl"), 'rb'))
rel2id = pickle.load(open(os.path.join(args.data_path, "rel2id.pkl"), 'rb'))
id2rel = pickle.load(open(os.path.join(args.data_path, "id2rel.pkl"), 'rb'))

num_diff, num_all = 0, 0
if args.split == 'train':
    all_vanilla_datasets_with_query_structure = pickle.load(open(os.path.join(args.question_path, 'all_train_vanilla_datasets_with_query_structure.pkl'), 'rb'))
elif args.split == 'valid':
    all_vanilla_datasets_with_query_structure = pickle.load(open(os.path.join(args.question_path, 'all_valid_vanilla_datasets_with_query_structure.pkl'), 'rb'))
elif args.split == 'test':
    all_vanilla_datasets_with_query_structure = pickle.load(open(os.path.join(args.question_path, 'all_test_vanilla_datasets_with_query_structure.pkl'), 'rb'))
print (len(all_vanilla_datasets_with_query_structure))
new_ntm_datasets_with_query_structure, new_vanilla_datasets_with_query_structure = [], []

question_idx = dict()
if 'FB150k-30' in args.data_path:
    kb_perc = 30
elif 'FB150k-50' in args.data_path:
    kb_perc = 50
else:
    assert False

extracted_query_and_candidate = pickle.load(open(os.path.join(args.candidate_path,
            "question_and_query_candidate_rank_{}_mrr_{}.pkl".format(args.target_rank, args.target_mrr)), 'rb'))

for q_idx, vanilla_data in tqdm(enumerate(all_vanilla_datasets_with_query_structure), total=len(all_vanilla_datasets_with_query_structure)):
    vanilla_question = vanilla_data[0][-1]
    if q_idx < args.start_idx:
        continue
    if q_idx == args.end_idx:
        break
    query = vanilla_data[0][-3]
    query_structure = vanilla_data[0][-2]
    assert type(query) == list and type(query_structure) == list
    assert vanilla_question not in all_structures

    if vanilla_question not in extracted_query_and_candidate:
        continue

    many_solutions = []
    for candidate_query in extracted_query_and_candidate[vanilla_question]:
        candidate_query_unflattened = candidate_query
        candidate_query_structure = tuple2list(candidate_query)
        recursive_structure(candidate_query_structure)
        assert type(candidate_query) == tuple
        candidate_query = flatten(candidate_query)
        tf_candidate_query = torch.LongTensor([candidate_query]).cuda()
        tmp_structure = []
        recursive_main(candidate_query, list2tuple(candidate_query_structure), 0, tmp_structure, 0, 0)

        tf_main_entity = torch.LongTensor(tmp_structure).cuda()
        all_structures[vanilla_question][candidate_query_unflattened].append(deepcopy(tmp_structure))
        tf_main_embedding = latent_space_executor.entity_embedding(tf_main_entity)
        assert tf_main_embedding.shape[1] == 1
        if latent_space_executor.geo == 'rotate':
            tmp_center, tmp_offset = torch.chunk(tf_main_embedding.squeeze(dim=1), 2, dim=1)
        elif latent_space_executor.geo == 'box':
            tmp_center = tf_main_embedding[:,0,:]
            tmp_offset = torch.zeros_like(tmp_center).to(tmp_center)
        all_centers[vanilla_question][candidate_query_unflattened].append(tmp_center)
        all_offsets[vanilla_question][candidate_query_unflattened].append(tmp_offset)

        _, _, _, _, tmp_center, tmp_offset, _ = recursive_embed_first_step(candidate_query, list2tuple(candidate_query_structure), 0, tmp_structure, 0, 0, tmp_center, tmp_offset, vanilla_question, candidate_query_unflattened)
        recursive_embed_second_step(candidate_query, list2tuple(candidate_query_structure), 0, tmp_structure, 0, 0, tmp_center, tmp_offset, vanilla_question, candidate_query_unflattened)

        all_orders[vanilla_question][candidate_query_unflattened].append([[0], 'Stop'])

        tmp_vanilla_datasets_with_query_structure = []
        for idx, i in enumerate(all_structures[vanilla_question][candidate_query_unflattened]):
            tmp_order = deepcopy(all_orders[vanilla_question][candidate_query_unflattened][idx])
            tmp_vanilla_datasets_with_query_structure.append([None, # bert
                                all_centers[vanilla_question][candidate_query_unflattened][idx].detach().cpu().numpy(), # center
                                all_offsets[vanilla_question][candidate_query_unflattened][idx].detach().cpu().numpy(), # offset
                                tmp_order,
                                len(all_structures[vanilla_question][candidate_query_unflattened]),
                                query,
                                candidate_query_structure,
                                vanilla_question,
                                ])
        many_solutions.append(tmp_vanilla_datasets_with_query_structure)
        if len(all_centers) > 5000:
            all_centers = defaultdict(dl)
            all_offsets = defaultdict(dl)
    new_vanilla_datasets_with_query_structure.append(many_solutions)
    question_idx[vanilla_question] = num_all
    num_all += 1
    assert len(new_vanilla_datasets_with_query_structure) == num_all

pickle.dump(new_vanilla_datasets_with_query_structure, open(os.path.join(args.candidate_path, '%s_datasets_with_query_structure_rank_%s_mrr_%s_%d_%s.pkl'%(args.split, args.target_rank, args.target_mrr, kb_perc, latent_space_executor.geo)), 'wb'))
