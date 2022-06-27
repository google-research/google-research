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

import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import pickle
from collections import defaultdict
from tqdm import tqdm

from lego.model.synthesizer import BatchPowersetParser
from lego.data_process.dataloader import SingledirectionalOneShotIterator, UnfoldedProgramDataloader, ProgramDataloader
from lego.common.utils import set_logger, parse_time, set_global_seed, construct_graph, eval_tuple, get_optimizer, save_model, load_lse_checkpoint, set_train_mode

from smore.cpp_sampler import sampler_clib
from lego.model import get_lse_model
from lego.env.runner import ProgramEnv

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")
    parser.add_argument('--do_search', action='store_true', help="do search")

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--eval_path', type=str, default=None, help="KG eval data path")
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=500, type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=12.0, type=float, help="margin in the loss")
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int, help="batch size of queries")
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=6, type=int, help="used to speed up torch.dataloader")
    parser.add_argument('-save', '--save_path', default=None, type=str, help="no need to set manually, will configure automatically")
    parser.add_argument('--max_steps', default=100000, type=int, help="maximum iterations to train")
    parser.add_argument('--warm_up_steps', default=None, type=int, help="no need to set manually, will configure automatically")

    parser.add_argument('--save_checkpoint_steps', default=50000, type=int, help="save checkpoints every xx steps")
    parser.add_argument('--valid_steps', default=10000, type=int, help="evaluate validation queries every xx steps")
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--geo', default='vec', type=str, choices=['vec', 'box', 'beta', 'rotate', 'distmult', 'complex'], help='the reasoning model, vec for GQE, box for Query2box, beta for BetaE')
    parser.add_argument('--lr_schedule', default='none', type=str, choices=['none', 'step'], help='learning rate scheduler')

    parser.add_argument('--print_on_screen', action='store_true')

    parser.add_argument('--tasks', default='1p', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--training_tasks', default=None, type=str, help="training tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('-betam', '--beta_mode', default="(1600,2)", type=str, help='(hidden_dim,num_layer) for BetaE relational projection')
    parser.add_argument('-boxm', '--box_mode', default="(none,0.02)", type=str, help='(offset activation,center_reg) for Query2box, center_reg balances the in_box dist and out_box dist')
    parser.add_argument('-rotatem', '--rotate_mode', default="(Mean,True)", type=str, help='(intersection aggr,nonlinearity) for Rotate')
    parser.add_argument('-complexm', '--complex_mode', default="(Mean,True)", type=str, help='(intersection aggr,nonlinearity) for Rotate')
    parser.add_argument('-distmultm', '--distmult_mode', default="(Mean,True)", type=str, help='(intersection aggr,nonlinearity) for Rotate')
    parser.add_argument('--prefix', default=None, type=str, help='prefix of the log path')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path for loading the checkpoints')
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'], help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')

    parser.add_argument('--share_optim_stats', action='store_true', default=False)
    parser.add_argument('--filter_test', action='store_true', default=False)
    parser.add_argument('--online_sample', action='store_true', default=False)
    parser.add_argument('--sampler_type', type=str, default='naive', help="type of sampler, choose from [naive, sqrt, nosearch, mix_0.x]")

    parser.add_argument('--share_negative', action='store_true', default=False)
    parser.add_argument('--online_sample_mode', default="(5000,-1,w,u,80)", type=str,
                help='(0,0,w/u,wstruct/u,0) or (relation_bandwidth,max_num_of_intermediate,w/u,wstruct/u,max_num_of_partial_answer), weighted or uniform')
    parser.add_argument('--online_weighted_structure_prob', default="(70331,141131,438875)", type=str,
                help='(same,0,w/u,wstruct/u)')

    parser.add_argument('--gpus', default='-1', type=str, help="gpus")
    parser.add_argument('--port', default='29500', type=str, help="dist port")
    parser.add_argument('--train_online_mode', default="(single,0,n,False,before)", type=str,
                help='(mix/single,sync_steps,er/e/r/n trained on cpu,async flag, before/after)')
    parser.add_argument('--optim_mode', default="(fast,adagrad,cpu,True,5)", type=str,
                help='(fast/aggr,adagrad/rmsprop,cpu/gpu,True/False,queue_size)')

    parser.add_argument('--reg_coeff', default=0., type=float, help="margin in the loss")

    parser.add_argument('--lse_model', default='box', type=str)
    parser.add_argument('--lse_save_path', default="", type=str)
    parser.add_argument('--question_path', type=str, default=None)

    parser.add_argument('--max_eps_len_list', default=None, type=str, help="")
    parser.add_argument('--dataset_name', default=None, type=str)
    parser.add_argument('--relation_threshold', default=50, type=int, help="margin in the loss")

    parser.add_argument('--train_with_ground_truth', action='store_true', default=False)
    parser.add_argument('--train_with_candidates', action='store_true', default=False)
    parser.add_argument('--candidate_config', default="(5,mrr,0,0)", type=str, help="(50,mrr,0/50,5), target_rank, metric, max_rank, r4e")
    parser.add_argument('--optim', default="adam", type=str, help="rmsprop or adam")
    parser.add_argument('--load_mode', default="aligned", type=str, help="")
    parser.add_argument('--supervised_candidate_batch_size', default=0, type=int, help='train supervised every xx steps')
    parser.add_argument('--relation_coeff', default=0.1, type=float, help='train supervised every xx steps')
    parser.add_argument('--reduce', default="max", type=str, help="ori or aligned or comb")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--train_with_masking', action='store_true', default=False)
    parser.add_argument('--masking', action='store_true', default=False)
    parser.add_argument('--candidate_path', type=str, default=None)
    parser.add_argument('--kg_dtype', type=str, default="uint32")
    return parser.parse_args(args)

def set_save_path(args):
    cur_time = parse_time()
    args.save_path = os.path.join(args.candidate_path, "synthesizer/%s-%s-%s/%s"%(args.candidate_config,
                                                                           args.supervised_candidate_batch_size,
                                                                           args.relation_coeff,
                                                                           cur_time))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

def all_available_action(powersets, nrelation, x_scores, relation_threshold, mask_scores_class, relation_pruner, mask_mode='none'):
    sampled_scores, sampled_relations, branches_picked_list = [], [], []
    for i in range(len(mask_scores_class[0])):
        if not mask_scores_class[0][i]:
            continue
        sampled_score = torch.LongTensor(np.array([i]))
        branches_picked = powersets[0][sampled_score[0].numpy()] #! update for batch; (0,1) or (0) or ()
        if len(branches_picked) == 1:
            if mask_mode == 'rs':
                rel_logits = relation_pruner(x_scores[branches_picked[0]].unsqueeze(0))
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


def evaluate(parser, args, writer, test_env, test_name, step):
    all_logs = []

    for i in tqdm(range(test_env.n_query), disable=(not args.debug)):
        obs, done, ep_reward = test_env.reset(), False, 0
        assert obs[0] != None
        steps = 0
        same = True
        while True:
            x_scores, x_relations, berts, edge_indices, softmax_edge_indices, \
                n_program, max_y_score_len, powersets, value_edge_indices, mask_scores_class, _, _ = obs

            score, relation, value = parser.action_value(x_scores,
                                                            x_relations,
                                                            berts,
                                                            edge_indices,
                                                            softmax_edge_indices,
                                                            value_edge_indices,
                                                            n_program,
                                                            max_y_score_len)

            sampled_scores, sampled_relations, branches_picked_list, mask_relations_class_list, _ \
                        = parser.sample_action(score, relation, powersets,
                                                    args.masking, test_env.latent_space_executor,
                                                    x_scores, args.relation_threshold, test_env.ent_out,
                                                    mask_scores_class, "(softmax,0)",
                                                    None, stochastic=False,
                                                    old=False, epsilon=0., mask_mode="none")

            assert len(sampled_scores) == 1 and len(sampled_relations) == 1 and len(branches_picked_list) == 1 and len(mask_relations_class_list) == 1
            sampled_score = sampled_scores[0]
            sampled_relation = sampled_relations[0]
            branches_picked = branches_picked_list[0]
            mask_relations_class = mask_relations_class_list[0]
            obs, reward, done, info = test_env.step(sampled_score, sampled_relation, branches_picked)

            ep_reward += reward #! this is not used since current reward only appears at last step
            steps += 1
            if done:
                info['nf'] = (reward < 0)
                all_logs.append(info)
                break

    to_return = None
    for metric in all_logs[0].keys():
        if 'num_answer' in metric or 'logits_topk' in metric or 'indices_topk' in metric or 'all_ans' in metric:
            continue
        avg = np.mean([log[metric] for log in all_logs])
        writer.add_scalar("%s-%s"%(test_name, metric), avg, step)
        logging.info("Step: {}, {}-{}: {}".format(step, test_name, metric, avg))
        print("Step: {}, {}-{}: {}".format(step, test_name, metric, avg))
        if metric == 'h1m':
            to_return = avg
    return to_return


def train(parser, args, writer, env, test_envs, test_names, batch_sz=64, updates=250):

    current_learning_rate = args.learning_rate
    optimizer = get_optimizer(parser, current_learning_rate)

    ep_rewards = [0.0]
    best_so_far = 0

    for update in tqdm(range(updates)):
        parser.train_step(parser, optimizer, env.dataloader, args, update, writer)

        if update % args.valid_steps == 0 and update > 0:
            for test_env, test_name in zip(test_envs, test_names):
                cur_result = evaluate(parser, args, writer, test_env, test_name, update)
            if cur_result > best_so_far:
                save_variable_list = {
                    'step': update,
                }
                save_model(parser, optimizer, save_variable_list, args, os.path.join(args.save_path, "synthesizer_checkpoint"))
                best_so_far = cur_result
    try:
        update
    except:
        update = 0
    for test_env, test_name in zip(test_envs, test_names):
        cur_result = evaluate(parser, args, writer, test_env, test_name, update)
    if cur_result > best_so_far and update != 0:
        save_variable_list = {
            'step': update,
        }
        save_model(parser, optimizer, save_variable_list, args, os.path.join(args.save_path, "synthesizer_checkpoint"))
    return ep_rewards


def main(args):
    set_global_seed(args.seed)
    assert args.do_train or args.do_valid or args.do_test or args.do_search
    set_train_mode(args)
    set_save_path(args)
    print ("Logging to", args.save_path)
    writer = set_logger(args)

    latent_space_executor = get_lse_model(args)
    logging.info("Latent space executor created")
    load_lse_checkpoint(args, latent_space_executor)
    logging.info("Latent space executor loaded")

    kg = sampler_clib.create_kg(args.nentity, args.nrelation, args.kg_dtype)
    kg.load_triplets(os.path.join(args.data_path, "train_indexified.txt"), True)
    ent_in, ent_out, _ = construct_graph(args.data_path, ["train_indexified.txt"])
    logging.info("KG constructed")
    args.load_mode = 'aligned'

    if args.dataset_name in ['webqsp', 'cwq']:
        assert args.load_mode in ['aligned', 'ori']
        target_k = eval_tuple(args.candidate_config)
        if 'FB150k-30' in args.data_path:
            kg_perc = 30
        elif 'FB150k-50' in args.data_path:
            kg_perc = 50
        target_rank, target_mrr = target_k

        print ("kg information check passed!")

        train_datasets = pickle.load(open(os.path.join(args.candidate_path, 'train_datasets_with_query_structure_rank_%s_mrr_%s_%d_%s.pkl'%(target_rank, target_mrr, kg_perc, latent_space_executor.geo)), 'rb'))
        args.train_with_ground_truth = True
        args.additional_search = True

        valid_datasets = pickle.load(open(os.path.join(args.question_path, "all_valid_vanilla_datasets_with_query_structure.pkl"), 'rb'))
        test_datasets = pickle.load(open(os.path.join(args.question_path, "all_test_vanilla_datasets_with_query_structure.pkl"), 'rb'))
        all_berts = pickle.load(open(os.path.join(args.question_path, "q_berts.pkl"), 'rb'))

        with open(os.path.join(args.question_path, 'all_test_%s_fn_answers.pkl'%args.dataset_name), 'rb') as f:
            question_answers = pickle.load(f)

        for i in tqdm(range(len(train_datasets)), disable=not args.print_on_screen):
            for k in range(len(train_datasets[i])):
                question = train_datasets[i][k][0][-1]
                for j in range(len(train_datasets[i][k])):
                    assert (question in question_answers)
                    train_datasets[i][k][j][0] = all_berts[question]
                    train_datasets[i][k][j].append(np.ones(args.nrelation)) #! place holder

        for datasets in [valid_datasets, test_datasets]:
            for i in tqdm(range(len(datasets)), disable=not args.print_on_screen):
                question = datasets[i][0][-1]
                for j in range(len(datasets[i])):
                    assert (question in question_answers)
                    datasets[i][j][0] = all_berts[question]
                    datasets[i][j].append(np.ones(args.nrelation)) #! place holder
        bert_dim = len(valid_datasets[0][0][0][0])
    else:
        raise NotImplemented

    train_iterator = SingledirectionalOneShotIterator(DataLoader(
                        UnfoldedProgramDataloader(train_datasets, args.nentity, args.nrelation, 1, latent_space_executor,
                                reduce=args.reduce,
                                supervised_candidate_batch_size=args.supervised_candidate_batch_size,
                                shuffle=True if not args.debug else False),
                        batch_size=args.batch_size,
                        shuffle=True if not args.debug else False,
                        num_workers=args.cpu_num,
                        collate_fn=UnfoldedProgramDataloader.collate_fn
                    ))

    valid_iterator = ProgramDataloader(valid_datasets, args.nentity, args.nrelation, 1, latent_space_executor, shuffle=False, reduce='max')
    test_iterator = ProgramDataloader(test_datasets, args.nentity, args.nrelation, 1, latent_space_executor, shuffle=False, reduce='max')

    synthesizer = BatchPowersetParser(args.hidden_dim, args.hidden_dim, bert_dim, args.hidden_dim, args.nrelation, args.reduce, 'deepsets', train_iterator.max_y_score_len, requires_vf=True)

    if args.cuda:
        latent_space_executor = latent_space_executor.cuda()
        synthesizer = synthesizer.cuda()

    args.force = True
    args.reward_metrics = 'mrr'
    test_traversed_fn_answers = None
    args.min_eps_len = 2
    args.filter_3p = False
    args.keep_original_intersection_order = True
    args.delayed_evaluation = False
    args.min_eps_len_list = None

    train_env = ProgramEnv(train_iterator, latent_space_executor, question_answers, ent_out, args.force, args.dataset_name, reward_metrics=args.reward_metrics, traversed_answer=test_traversed_fn_answers, min_eps_len=args.min_eps_len, filter_3p=args.filter_3p, answer_type='query' if args.dataset_name=='metaqa' else 'question', keep_original_intersection_order=args.keep_original_intersection_order, max_eps_len_list=None if args.max_eps_len_list is None else eval_tuple(args.max_eps_len_list), delayed_evaluation=False, min_eps_len_list=None if args.min_eps_len_list is None else eval_tuple(args.min_eps_len_list))
    valid_env = ProgramEnv(valid_iterator, latent_space_executor, question_answers, ent_out, args.force, args.dataset_name, reward_metrics=args.reward_metrics, traversed_answer=test_traversed_fn_answers, min_eps_len=args.min_eps_len, filter_3p=args.filter_3p, answer_type='query' if args.dataset_name=='metaqa' else 'question', keep_original_intersection_order=args.keep_original_intersection_order, max_eps_len_list=None if args.max_eps_len_list is None else eval_tuple(args.max_eps_len_list), delayed_evaluation=False, min_eps_len_list=None if args.min_eps_len_list is None else eval_tuple(args.min_eps_len_list))
    test_env = ProgramEnv(test_iterator, latent_space_executor, question_answers, ent_out, args.force, args.dataset_name, reward_metrics=args.reward_metrics, traversed_answer=test_traversed_fn_answers, min_eps_len=args.min_eps_len, filter_3p=args.filter_3p, answer_type='query' if args.dataset_name=='metaqa' else 'question', keep_original_intersection_order=args.keep_original_intersection_order, max_eps_len_list=None if args.max_eps_len_list is None else eval_tuple(args.max_eps_len_list), delayed_evaluation=False, min_eps_len_list=None if args.min_eps_len_list is None else eval_tuple(args.min_eps_len_list))

    train(synthesizer, args, writer, train_env, [test_env], ['test'], args.batch_size, args.max_steps)

if __name__ == '__main__':
    main(parse_args())
