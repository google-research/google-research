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
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from torch.utils.data import DataLoader
import time
import pickle
from collections import defaultdict
from tqdm import tqdm

from lego.model.pruner import BranchPruner, RelationPruner
from lego.data_process.dataloader import RelationSampler, BranchSampler, EvalRelationDataset, EvalBranchDataset
from lego.common.utils import set_logger, parse_time, set_global_seed, query_name_dict, name_query_dict, list2tuple, tuple2list, construct_graph, eval_tuple, get_optimizer, save_model, log_and_write_metrics, group_queries, load_lse_checkpoint, set_train_mode
from torch_sparse import SparseTensor

from smore.cpp_sampler import sampler_clib
from smore.common.config import parse_args
from lego.model import get_lse_model

def parse_additional_args(parser):
    parser.add_argument('--do_search', action='store_true', default=False)
    parser.add_argument('--question_path', type=str, default=None)
    parser.add_argument('--relation_tasks', default='1p.2p.3p', type=str)
    parser.add_argument('--branch_tasks', default='2i.3i.pi', type=str)

    return parser

def override_config(args): #! may update here
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

def parse_structures(tasks):
    task_list = tasks.split('.')
    query_structures = []
    for task in task_list:
        query_structures.append(name_query_dict[task])
    return query_structures

def check_valid_branch_structures(query_structures):
    for query_structure in query_structures:
       assert query_name_dict[query_structure] in ['2i', '3i', '4i', 'pi', '2pi', 'p3i']

def set_save_path(args):
    cur_time = parse_time()
    args.save_path = os.path.join(args.checkpoint_path, "pruners/r-%s-b-%s/%s"%(args.relation_tasks, args.branch_tasks,
                                                                             cur_time))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

def get_eval_data(args):
    train_query__structure__rel_pretraining = pickle.load(open(os.path.join(args.question_path, 'train_query__structure__rel_pretraining.pkl'), 'rb'))
    if 'CWQ' in args.question_path:
        valid_query__structure__rel_pretraining = pickle.load(open(os.path.join(args.question_path, 'valid_query__structure__rel_pretraining.pkl'), 'rb'))
    elif 'webqsp' in args.question_path:
        valid_query__structure__rel_pretraining = pickle.load(open(os.path.join(args.question_path, 'test_query__structure__rel_pretraining.pkl'), 'rb'))

    for key in train_query__structure__rel_pretraining:
        valid_query__structure__rel_pretraining[key] = valid_query__structure__rel_pretraining[key]+train_query__structure__rel_pretraining[key]
        print(key, len(valid_query__structure__rel_pretraining[key]))


    train_query__structure__inter_pretraining = pickle.load(open(os.path.join(args.question_path, 'train_query__structure__inter_pretraining.pkl'), 'rb'))
    if "CWQ" in args.question_path:
        valid_query__structure__inter_pretraining = pickle.load(open(os.path.join(args.question_path, 'valid_query__structure__inter_pretraining.pkl'), 'rb'))
    elif "webqsp" in args.question_path:
        valid_query__structure__inter_pretraining = pickle.load(open(os.path.join(args.question_path, 'test_query__structure__inter_pretraining.pkl'), 'rb'))
    for key in train_query__structure__inter_pretraining:
        valid_query__structure__inter_pretraining[key] = valid_query__structure__inter_pretraining[key]+train_query__structure__inter_pretraining[key]
        print(key, len(valid_query__structure__inter_pretraining[key]))

    return valid_query__structure__rel_pretraining, valid_query__structure__inter_pretraining

def pretrain_relation_step(latent_space_executor, relation_pruner, optimizer, train_iterator, args, step, ent_rel_mat):
    relation_pruner.train()
    optimizer.zero_grad()
    positive_sample, batch_queries, query_structures = next(train_iterator)

    batch_queries_dict, batch_idxs_dict = group_queries(batch_queries, query_structures, args.cuda)

    with torch.no_grad():
        all_embeddings, all_idxs = latent_space_executor.emb_forward(batch_queries_dict, batch_idxs_dict)

    if latent_space_executor.geo == 'rotate':
        rel_logits = relation_pruner(torch.cat(all_embeddings, dim=1))
    elif latent_space_executor.geo == 'box':
        rel_logits = relation_pruner(all_embeddings[0])
    positive_sample = positive_sample[all_idxs]
    row, col, _ = ent_rel_mat[positive_sample].coo()
    rel_logits = rel_logits[row]

    loss = nn.CrossEntropyLoss()(rel_logits, col)
    loss.backward()
    optimizer.step()

    return loss.item()

def pretrain_branch_step(latent_space_executor, branch_pruner, optimizer, train_iterator, args, step):
    branch_pruner.train()
    optimizer.zero_grad()
    batch_queries, query_structures_unflatten, query_structures, labels = next(train_iterator)
    batch_queries_dict, batch_idxs_dict = group_queries(batch_queries, query_structures, args.cuda)
    if args.cuda:
        labels = labels.cuda()

    all_logits, all_idxs = latent_space_executor.intersection_forward(batch_queries_dict,
                                                                      batch_idxs_dict,
                                                                      query_structures_unflatten,
                                                                      branch_pruner)
    all_logits = torch.cat(all_logits, 0)
    labels = labels[all_idxs]

    loss = nn.BCELoss()(all_logits, labels)
    loss.backward()
    optimizer.step()

    return loss.item()

def eval_branch_pruner(latent_space_executor, branch_pruner, args, test_dataloader, query_name_dict):
    latent_space_executor.eval()
    branch_pruner.eval()

    step = 0
    total_steps = len(test_dataloader)
    logs = defaultdict(float)

    num_correct = 0
    num_cases = 0
    all_logits, all_labels, all_recall = [], [], []

    with torch.no_grad():
        for queries, queries_unflatten, query_structures, query_structures_unflatten, labels in tqdm(test_dataloader, disable=not args.print_on_screen):
            batch_queries_dict, batch_idxs_dict = group_queries(queries, query_structures, args.cuda)
            if args.cuda:
                labels = labels.cuda()

            logits, idxs = latent_space_executor.intersection_forward(batch_queries_dict,
                                                                batch_idxs_dict,
                                                                query_structures_unflatten,
                                                                branch_pruner)

            logits = torch.cat(logits, 0)
            all_logits.extend(logits.squeeze(1).cpu().numpy())
            logits = (logits > 0.5).float()
            labels = labels[idxs]
            all_labels.extend(labels.squeeze(1).cpu().numpy())
            num_correct += (labels == logits).float().sum().item()
            num_cases += len(query_structures_unflatten)

            if step % args.test_log_steps == 0:
                logging.info('Evaluating the branch pruner ... (%d/%d)' % (step, total_steps))

            step += 1
    roc_auc = roc_auc_score(all_labels, all_logits)
    acc = num_correct / num_cases
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    positives = all_logits[all_labels==1]
    for t in [0.1*i for i in range(1, 10)]:
        all_recall.append(np.sum(positives > t) / len(positives))

    logs['roc_auc'] = roc_auc
    logs['acc'] = acc
    logs['recall'] = all_recall

    return logs

def eval_relation_pruner(latent_space_executor, relation_pruner, args, test_dataloader, query_name_dict):
    latent_space_executor.eval()
    relation_pruner.eval()

    step = 0
    total_steps = len(test_dataloader)
    logs = defaultdict(float)

    all_rankings = []
    with torch.no_grad():
        for queries, queries_unflatten, query_structures, rels in tqdm(test_dataloader, disable=not args.print_on_screen):
            batch_queries_dict, batch_idxs_dict = group_queries(queries, query_structures, args.cuda)
            if args.cuda:
                rels = rels.cuda()
            all_embeddings, all_idxs = latent_space_executor.emb_forward(batch_queries_dict, batch_idxs_dict)
            rels = rels[all_idxs]

            if latent_space_executor.geo == 'rotate':
                rel_logits = relation_pruner(torch.cat(all_embeddings, dim=1))
            elif latent_space_executor.geo == 'box':
                rel_logits = relation_pruner(all_embeddings[0])
            argsort = torch.argsort(rel_logits, dim=-1, descending=True)
            ranking = argsort.clone().to(torch.float)
            if args.cuda:
                range_vec = torch.arange(args.nrelation).to(torch.float).repeat(argsort.shape[0], 1).cuda()
            else:
                range_vec = torch.arange(args.nrelation).to(torch.float).repeat(argsort.shape[0], 1)
            ranking = ranking.scatter_(1, argsort, range_vec)
            target_ranking = torch.gather(ranking, dim=1, index=rels)
            all_rankings.append(target_ranking)

            if step % args.test_log_steps == 0:
                logging.info('Evaluating the latent_space_executor... (%d/%d)' % (step, total_steps))

            step += 1

    all_rankings = torch.cat(all_rankings, dim=0)
    all_rankings += 1
    logs['mean_rank'] = torch.mean(all_rankings).item()
    logs['max_rank'] = torch.max(all_rankings).item()
    logs['mrr'] = torch.mean(1./all_rankings).item()

    return logs

def pretrain_pruners(args, writer, ent_rel_mat, latent_space_executor,
                     relation_pruner, branch_pruner,
                     relation_iterator, branch_iterator,
                     eval_relation_dataloader, eval_branch_dataloader):
    current_learning_rate = args.learning_rate
    relation_optimizer = get_optimizer(relation_pruner, current_learning_rate)
    branch_optimizer = get_optimizer(branch_pruner, current_learning_rate)
    warm_up_steps = args.max_steps // 2

    best_relation_mrr, best_branch_auc = 0, 0
    training_logs = []
    for step in tqdm(range(args.max_steps), disable=not args.print_on_screen):
        log = {}
        log['relation_loss'] = pretrain_relation_step(latent_space_executor, relation_pruner, relation_optimizer,
                                     relation_iterator, args, step, ent_rel_mat)
        log['branch_loss'] = pretrain_branch_step(latent_space_executor, branch_pruner, branch_optimizer,
                                     branch_iterator, args, step)
        if step % args.log_steps == 0:
            for metric in log:
                writer.add_scalar(metric, log[metric], step)

        if step >= warm_up_steps:
            current_learning_rate = current_learning_rate / 5
            logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
            relation_optimizer = get_optimizer(relation_pruner, current_learning_rate)
            branch_optimizer = get_optimizer(branch_pruner, current_learning_rate)
            warm_up_steps = warm_up_steps * 1.5

        if step % args.valid_steps == 0 and args.do_test:
            logging.info('Evaluating on Test Dataset...')
            eval_branch_metrics = eval_branch_pruner(latent_space_executor, branch_pruner, args, eval_branch_dataloader, query_name_dict)
            log_and_write_metrics('Test average', step, eval_branch_metrics, writer)
            eval_relation_metrics = eval_relation_pruner(latent_space_executor, relation_pruner, args, eval_relation_dataloader, query_name_dict)
            log_and_write_metrics('Test average', step, eval_relation_metrics, writer)

            if eval_branch_metrics['roc_auc'] > best_branch_auc:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps,
                    'checkpoint_path': os.path.join(args.checkpoint_path, "checkpoint")
                }
                save_model(branch_pruner, branch_optimizer, save_variable_list, args,
                           os.path.join(args.save_path, "branch_checkpoint"))
                pickle.dump(eval_branch_metrics['recall'], open(os.path.join(args.save_path, "best_recall.pkl"), "wb"))
                best_branch_auc = eval_branch_metrics['roc_auc']
                logging.info("Branch pruner saved")

            if eval_relation_metrics['mrr'] > best_relation_mrr:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps,
                    'checkpoint_path': os.path.join(args.checkpoint_path, "checkpoint")
                }
                save_model(relation_pruner, relation_optimizer, save_variable_list, args,
                           os.path.join(args.save_path, "relation_checkpoint"))
                best_relation_mrr = eval_relation_metrics['mrr']
                logging.info("Relation pruner saved")
    return step

def main(parser):
    parser = parse_additional_args(parser)
    args = parser.parse_args(None)
    set_global_seed(args.seed)
    set_train_mode(args)
    set_save_path(args)
    gpus = [int(i) for i in args.gpus.split(".")]
    assert len(gpus) == 1, "pruner pretraining only supports single GPU"
    print ("Logging to", args.save_path)
    writer = set_logger(args)

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

    kg = sampler_clib.create_kg(args.nentity, args.nrelation, args.kg_dtype)
    kg.load_triplets(os.path.join(args.data_path, "train_indexified.txt"), True)
    logging.info("KG constructed")

    ent_in, ent_out, adj = construct_graph(args.data_path, ['train_indexified.txt'], True)
    adj = torch.LongTensor(adj).transpose(0,1)
    ent_rel_mat = SparseTensor(row=adj[0], col=adj[1])
    logging.info("Adj matrix constructed")

    relation_query_structures = parse_structures(args.relation_tasks)
    branch_query_structures = parse_structures(args.branch_tasks)
    check_valid_branch_structures(branch_query_structures)
    branch_sampler = BranchSampler(kg, branch_query_structures,
                                   1, # placeholder does not matter
                                   eval_tuple(args.online_sample_mode),
                                   [1./len(branch_query_structures) for _ in range(len(branch_query_structures))],
                                   sampler_type='naive', same_in_batch=False, share_negative=True, num_threads=args.cpu_num)
    relation_sampler = RelationSampler(kg, relation_query_structures,
                                       1, # placeholder does not matter
                                       eval_tuple(args.online_sample_mode),
                                       [1./len(relation_query_structures) for _ in range(len(relation_query_structures))],
                                       sampler_type='naive', same_in_batch=False, share_negative=True, num_threads=args.cpu_num)
    branch_iterator = branch_sampler.batch_generator(args.batch_size)
    relation_iterator = relation_sampler.batch_generator(args.batch_size)
    logging.info("Train samplers constructed")

    if args.cuda:
        ent_rel_mat = ent_rel_mat.cuda()
        latent_space_executor = latent_space_executor.cuda()
        relation_pruner = relation_pruner.cuda()
        branch_pruner = branch_pruner.cuda()

    eval_relation_data, eval_branch_data = get_eval_data(args)
    eval_relation_dataloader = DataLoader(
        EvalRelationDataset(
            eval_relation_data,
            args.nentity,
            args.nrelation,
        ),
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num,
        collate_fn=EvalRelationDataset.collate_fn
    )
    eval_branch_dataloader = DataLoader(
        EvalBranchDataset(
            eval_branch_data,
            args.nentity,
            args.nrelation,
        ),
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num,
        collate_fn=EvalBranchDataset.collate_fn
    )
    logging.info("Eval dataloader constructed")

    step = 0
    if args.do_train:
        step = pretrain_pruners(args, writer, ent_rel_mat, latent_space_executor,
                            relation_pruner, branch_pruner,
                            relation_iterator, branch_iterator,
                            eval_relation_dataloader, eval_branch_dataloader)

    logging.info('Evaluating on Test Dataset...')
    test_all_metrics = eval_branch_pruner(latent_space_executor, branch_pruner, args, eval_branch_dataloader, query_name_dict)
    log_and_write_metrics('Test average', step, test_all_metrics, writer)
    test_all_metrics = eval_relation_pruner(latent_space_executor, relation_pruner, args, eval_relation_dataloader, query_name_dict)
    log_and_write_metrics('Test average', step, test_all_metrics, writer)

    print ('Training finished!!')
    logging.info("Training finished!!")


if __name__ == '__main__':
    main(parse_args())
