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
import os

import numpy as np
import torch
import torch.nn as nn

import pickle
from collections import defaultdict
from tqdm import tqdm

from lego.model.pruner import BranchPruner, RelationPruner
from lego.data_process.dataloader import ProgramDataloader
from lego.common.utils import set_logger, parse_time, set_global_seed, construct_graph, eval_tuple, load_lse_checkpoint, load_pruner_checkpoint, set_train_mode

from smore.common.config import parse_args
from smore.cpp_sampler import sampler_clib
from lego.model import get_lse_model
from lego.env.runner import ProgramEnv

def parse_additional_args(parser):
    parser.add_argument('--do_search', action='store_true', default=False)
    parser.add_argument('--pruner_save_path', default="", type=str)
    parser.add_argument('--question_path', type=str, default=None)

    parser.add_argument('--search_time_stamp', default=None, type=str)
    parser.add_argument('--max_eps_len_list', default=None, type=str, help="")
    parser.add_argument('--start_idx', default=0, type=int, help="random seed")
    parser.add_argument('--end_idx', default=-1, type=int, help="random seed")

    parser.add_argument('--dataset_name', default=None, type=str)
    parser.add_argument('--branch_threshold', default=0.1, type=float, help="margin in the loss")
    parser.add_argument('--relation_threshold', default=50, type=int, help="margin in the loss")
    parser.add_argument('--beam_size', default=1, type=int, help="margin in the loss")
    parser.add_argument('--mask_mode', default="rs", type=str)
    return parser

def set_save_path(args):
    cur_time = parse_time()
    if args.search_time_stamp is None:
        args.search_time_stamp = cur_time
    args.save_path = os.path.join(args.pruner_save_path, "candidates/%s" % args.search_time_stamp)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with open(os.path.join(args.save_path, "config.txt"), 'a+') as f:
        f.write("----------------------------------------------\n")
        f.write("%s\n"%cur_time)
        f.write("max_eps_len_list\n")
        f.write(str(args.max_eps_len_list)+"\n")
        f.write("----------------------------------------------\n")

def all_available_action(powersets, nrelation, x_scores, relation_threshold, mask_scores_class, relation_pruner, geo, mask_mode='none'):
    sampled_scores, sampled_relations, branches_picked_list = [], [], []
    for i in range(len(mask_scores_class[0])):
        if not mask_scores_class[0][i]:
            continue
        sampled_score = torch.LongTensor(np.array([i]))
        branches_picked = powersets[0][sampled_score[0].numpy()]
        if len(branches_picked) == 1:
            if mask_mode == 'rs':
                if geo == 'rotate':
                    rel_logits = relation_pruner(x_scores[branches_picked[0]].unsqueeze(0))
                elif geo == 'box':
                    center, offset = torch.chunk(x_scores[branches_picked[0]], 2, -1)
                    rel_logits = relation_pruner(center.unsqueeze(0))
                _, indices = torch.topk(rel_logits, relation_threshold)
                indices = indices.detach().cpu().numpy().T
            else:
                assert mask_mode == 'none'
                indices = np.random.choice(nrelation, relation_threshold, replace=False).reshape([relation_threshold, 1])

            sampled_scores.extend([sampled_score]*len(indices))
            sampled_relations.extend(indices)
            branches_picked_list.extend([branches_picked]*len(indices))
        else:
            sampled_relation = None

            sampled_scores.append(sampled_score)
            sampled_relations.append(sampled_relation)
            branches_picked_list.append(branches_picked)

    return sampled_scores, sampled_relations, branches_picked_list


def search_beam(args, relation_pruner, test_env, min_idx, max_idx, save_path='./'):
    all_logs = []

    print("number of queries", test_env.n_query)
    if max_idx == -1:
        max_idx = test_env.n_query
    for i in range(test_env.n_query):
        obs, done, ep_reward = test_env.reset(), False, 0
        if i < min_idx:
            continue
        if i >= max_idx:
            break
        if len(obs[0]) == 4:
            continue
        print ('-'*50, '(%d)'%i, test_env.queries[0], '-'*50)
        print ('-'*50, '(%d)'%i, test_env.questions[0], '-'*50)

        if args.dataset_name == 'webqsp':
            if len(obs[0]) == 3:
                continue
            threshold = args.relation_threshold
        elif args.dataset_name == 'cwq':
            if len(obs[0]) == 3 and args.mask_mode == 'none':
                threshold = 20
            else:
                threshold = args.relation_threshold

        assert obs[0] != None
        steps = 0
        same = True
        obs_list = [obs]
        # base_lls = [0]
        final_ll, final_reward, final_logs, final_structures = [], [], [], []
        while True:
            prev_obs_list = []
            all_sampled_scores, all_sampled_relations, all_branches_picked = [], [], []
            new_obs_list = []
            for obs in obs_list:
                x_scores, x_relations, berts, edge_indices, softmax_edge_indices, \
                    n_program, max_y_score_len, powersets, value_edge_indices, mask_scores_class, tmp_structure, steps_so_far = obs

                sampled_scores, sampled_relations, branches_picked_list = all_available_action(powersets, test_env.latent_space_executor.nrelation,
                                                        x_scores, threshold,
                                                        mask_scores_class, relation_pruner, args.geo, mask_mode=args.mask_mode)

                prev_obs_list.extend([obs]*len(sampled_scores))
                all_sampled_scores.extend(sampled_scores)
                all_sampled_relations.extend(sampled_relations)
                all_branches_picked.extend(branches_picked_list)

            for idx in tqdm(range(len(prev_obs_list))):
                x_scores, x_relations, berts, edge_indices, softmax_edge_indices, \
                    n_program, max_y_score_len, powersets, value_edge_indices, mask_scores_class, tmp_structure, steps_so_far = prev_obs_list[idx]
                test_env.set_state(*prev_obs_list[idx])

                new_obs, reward, done, info = test_env.step(all_sampled_scores[idx], all_sampled_relations[idx], all_branches_picked[idx])
                if done:
                    if test_env.delayed_evaluation:
                        continue
                    final_reward.append(reward)
                    final_logs.append(info)
                    final_structures.append(test_env.tmp_structure)
                else:
                    new_obs_list.append(new_obs)

            obs_list = new_obs_list
            steps += 1
            if len(obs_list) == 0:
                break
        if test_env.delayed_evaluation:
            final_reward, final_logs, final_structures = test_env.batch_evaluation()
        pickle.dump({'reward': final_reward, 'logs': final_logs, 'structures': final_structures}, open(os.path.join(save_path, "%d.pkl"%i), 'wb'))


def main(parser):
    parser = parse_additional_args(parser)
    args = parser.parse_args(None)
    set_global_seed(args.seed)
    assert args.do_train or args.do_valid or args.do_test or args.do_search
    set_train_mode(args)
    set_save_path(args)
    print ("Logging to", args.save_path)
    writer = set_logger(args)
    del writer

    latent_space_executor = get_lse_model(args)
    logging.info("Latent space executor created")
    load_lse_checkpoint(args, latent_space_executor)
    logging.info("Latent space executor loaded")

    relation_pruner = RelationPruner(latent_space_executor.entity_dim, args.nrelation)
    if args.geo == 'box':
        branch_pruner = BranchPruner(latent_space_executor.entity_dim * 2)
    elif args.geo == 'rotate':
        branch_pruner = BranchPruner(latent_space_executor.entity_dim)
    logging.info("Pruner created")
    load_pruner_checkpoint(args, relation_pruner, branch_pruner)
    logging.info("Pruner loaded")
    if args.cuda:
        latent_space_executor = latent_space_executor.cuda()
        relation_pruner = relation_pruner.cuda()
        branch_pruner = branch_pruner.cuda()

    kg = sampler_clib.create_kg(args.nentity, args.nrelation, args.kg_dtype)
    kg.load_triplets(os.path.join(args.data_path, "train_indexified.txt"), True)
    ent_in, ent_out, _ = construct_graph(args.data_path, ["train_indexified.txt"])
    logging.info("KG constructed")

    if args.dataset_name == 'webqsp':
        all_berts = pickle.load(open(os.path.join(args.question_path, "q_berts.pkl"), 'rb'))
        questions_to_search = pickle.load(open(os.path.join(args.question_path, "all_train_vanilla_datasets_with_query_structure.pkl"), 'rb'))
        with open(os.path.join(args.question_path, 'all_test_%s_fn_answers.pkl'%args.dataset_name), 'rb') as f:
            question_answers = pickle.load(f)
        for i in tqdm(range(len(questions_to_search)), disable=not args.print_on_screen):
            question = questions_to_search[i][0][-1]
            for j in range(len(questions_to_search[i])):
                assert (question in question_answers)
                questions_to_search[i][j][0] = all_berts[question]
                questions_to_search[i][j].append(np.ones(args.nrelation)) #! place holder
    else:
        raise NotImplemented

    test_train_iterator = ProgramDataloader(questions_to_search, args.nentity, args.nrelation, 1, latent_space_executor, shuffle=False, reduce='max')

    args.force = True
    args.reward_metrics = 'mrr'
    test_traversed_fn_answers = None
    args.min_eps_len = 2
    args.filter_3p = False
    args.keep_original_intersection_order = True
    args.delayed_evaluation = True
    args.min_eps_len_list = None

    env = ProgramEnv(test_train_iterator, latent_space_executor, question_answers, ent_out, args.force, args.dataset_name, reward_metrics=args.reward_metrics, traversed_answer=test_traversed_fn_answers, min_eps_len=args.min_eps_len, filter_3p=args.filter_3p, answer_type='query' if args.dataset_name=='metaqa' else 'question', keep_original_intersection_order=args.keep_original_intersection_order, intersection_selector=branch_pruner, intersection_threshold=args.branch_threshold, max_eps_len_list=None if args.max_eps_len_list is None else eval_tuple(args.max_eps_len_list), delayed_evaluation=args.delayed_evaluation, min_eps_len_list=None if args.min_eps_len_list is None else eval_tuple(args.min_eps_len_list))

    search_beam(args, relation_pruner, env, min_idx=args.start_idx, max_idx=args.end_idx, save_path=args.save_path)

if __name__ == '__main__':
    main(parse_args())
