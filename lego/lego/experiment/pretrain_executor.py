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
import time
import pickle5 as pickle
from collections import defaultdict
from tqdm import tqdm
from tensorboardX import SummaryWriter, GlobalSummaryWriter
import torch.multiprocessing as mp
import math

from smore.models import build_model
from smore.common.config import parse_args, all_tasks, query_name_dict, name_query_dict
from smore.common.embedding.embed_optimizer import get_optim_class
from smore.cpp_sampler.sampler_clib import KGMem
from smore.training.train_process import async_aggr, train_mp

from lego.common.utils import parse_time, set_global_seed, eval_tuple, load_lse_checkpoint, set_logger
from lego.data_process.dataloader import EvalQuestionDataset
from collections import namedtuple

QueryData = namedtuple('QueryData', ['data', 'buffer', 'writer_buffer'])


def setup_train_mode(args):
    tasks = args.tasks.split('.')
    if args.training_tasks is None:
        args.training_tasks = args.tasks
    training_tasks = args.training_tasks.split('.')
    for task in training_tasks:
        assert task in tasks, "training tasks should be a subset of full tasks"

    if args.online_sample:
        if eval_tuple(args.online_sample_mode)[3] == 'wstruct':
            normalized_structure_prob = np.array(eval_tuple(args.online_weighted_structure_prob)).astype(np.float32)
            normalized_structure_prob /= np.sum(normalized_structure_prob)
            normalized_structure_prob = normalized_structure_prob.tolist()
            assert len(normalized_structure_prob) == len(training_tasks)
        else:
            normalized_structure_prob = [1/len(training_tasks)] * len(training_tasks)
        args.normalized_structure_prob = normalized_structure_prob
        train_dataset_mode, sync_steps, sparse_embeddings, async_optim, merge_mode = eval_tuple(args.train_online_mode)
        update_mode, optimizer_name, optimizer_device, squeeze_flag, queue_size = eval_tuple(args.optim_mode)
        assert train_dataset_mode in ['single'], "mix has been deprecated"
        assert update_mode in ['aggr'], "fast has been deprecated"
        assert optimizer_name in ['adagrad', 'rmsprop', 'adam']
        args.sync_steps = sync_steps
        args.async_optim = async_optim
        args.merge_mode = merge_mode
        args.sparse_embeddings = sparse_embeddings
        args.sparse_device = optimizer_device
        args.train_dataset_mode = train_dataset_mode


def setup_save_path(args):
    cur_time = parse_time()
    if args.prefix is None:
        prefix = 'logs'
    else:
        prefix = args.prefix
    print("overwritting args.save_path")
    args.save_path = os.path.join(prefix, args.data_path.split('/')[-1], "{}-{}".format(args.training_tasks, args.tasks), args.geo)
    if args.geo in ['box']:
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.box_mode)
    elif args.geo in ['vec']:
        tmp_str = "g-{}".format(args.gamma)
    elif args.geo == 'beta':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.beta_mode)
    elif args.geo == 'rotate':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.rotate_mode)
    elif args.geo == 'distmult':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.distmult_mode)
    elif args.geo == 'complex':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.complex_mode)
    if args.negative_adversarial_sampling:
        tmp_str += '-adv-{}'.format(args.adversarial_temperature)
    if args.reg_coeff != 0:
        tmp_str += '-reg-{}'.format(args.reg_coeff)
    tmp_str += '-ngpu-{}'.format(args.gpus)

    if args.online_sample:
        tmp_str += '-os-{}'.format(args.online_sample_mode)
        if eval_tuple(args.online_sample_mode)[3] == 'wstruct':
            tmp_str += '-({})'.format(",".join(["%.2f"%i for i in args.normalized_structure_prob]))

        tmp_str += '-dataset-{}'.format(args.train_online_mode)
        tmp_str += '-opt-{}'.format(args.optim_mode)
        if args.share_negative:
            tmp_str += '-sharen'
        tmp_str += '-%s' % args.sampler_type
    tmp_str += '-lr_%s' % args.lr_schedule
    if args.checkpoint_path is not None:
        args.save_path = args.checkpoint_path
    else:
        args.save_path = os.path.join(args.save_path, tmp_str, cur_time)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print ("logging to", args.save_path)


def get_model(args):
    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    args.nentity = nentity
    args.nrelation = nrelation
    model = build_model(args, nentity, nrelation, query_name_dict)
    EmbedOpt = get_optim_class(args)
    EmbedOpt.prepare_optimizers(args, [x[1] for x in model.named_sparse_embeddings()])
    gpus = [int(i) for i in args.gpus.split(".")]

    if len(gpus) > 1:
        assert not args.cuda
        model.share_memory()

    logging.info('-------------------------------'*3)
    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)
    if args.geo == 'box':
        logging.info('box mode = %s' % args.box_mode)
    elif args.geo == 'beta':
        logging.info('beta mode = %s' % args.beta_mode)
    return model


def load_question_eval_data(args, phase):
    tasks = args.tasks.split('.')
    logging.info("loading %s data" % phase)
    all_data = pickle.load(open(os.path.join(args.eval_path, "%s_%s_question_query_answers.pkl" % (phase, args.dataset_name)), 'rb'))
    answers = pickle.load(open(os.path.join(args.eval_path, "all_test_%s_fn_answers.pkl" % args.dataset_name), 'rb'))

    test_dataset = EvalQuestionDataset(all_data, answers, args.nentity, args.nrelation)
    logging.info("%s info:" % phase)
    logging.info("num queries: %s" % len(test_dataset))
    buf = mp.Queue()
    writer_buffer = mp.Queue()
    return QueryData(test_dataset, buf, writer_buffer)


def write_to_writer(eval_dict, writer):

    def collect_and_write(writer_buffer, mode):
        metrics, average_metrics, step = writer_buffer.get()
        if step == -1:
            return False
        for query_structure in metrics:
            for metric in metrics[query_structure]:
                qname = query_name_dict[query_structure] if query_structure in query_name_dict else str(query_structure)
                writer.add_scalar("_".join([mode, qname, metric]), metrics[query_structure][metric], step)
        for metric in average_metrics:
            writer.add_scalar("_".join([mode, 'average', metric]), average_metrics[metric], step)
        return True

    writer_flag = True
    while writer_flag:
        writer_flag = False
        for key in eval_dict:
            if collect_and_write(eval_dict[key].writer_buffer, key):
                writer_flag = True


def parse_additional_args(parser):
    parser.add_argument('--do_search', action='store_true', default=False)
    parser.add_argument('--dataset_name', default=None, type=str)
    return parser

def main(parser):
    parser = parse_additional_args(parser)
    args = parser.parse_args(None)
    set_global_seed(args.seed)
    gpus = [int(i) for i in args.gpus.split(".")]
    assert args.gpus == '.'.join([str(i) for i in range(len(gpus))]), 'only support continuous gpu ids starting from 0, please set CUDA_VISIBLE_DEVICES instead'

    setup_train_mode(args)

    setup_save_path(args)

    writer = set_logger(args)

    model = get_model(args)

    logging.info('-------------------------------'*3)
    logging.info('Geo: %s' % args.geo)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % args.nentity)
    logging.info('#relation: %d' % args.nrelation)
    logging.info('#max steps: %d' % args.max_steps)
    logging.info('Evaluate unions using: %s' % args.evaluate_union)

    kg_mem = KGMem(dtype=args.kg_dtype)
    kg_mem.load(os.path.join(args.data_path, 'train_bidir.bin'))
    kg_mem.share_memory()

    opt_stats = load_lse_checkpoint(args, model)
    logging.info('tasks = %s' % args.tasks)
    logging.info('init_step = %d' % opt_stats['init_step'])
    if args.do_train:
        logging.info("Training info:")
        logging.info("{}: infinite".format(args.training_tasks))
        logging.info('Start Training...')
        logging.info('learning_rate = %d' % opt_stats['current_learning_rate'])
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)

    eval_dict = {}
    aggr_procs = []
    args.gpus = gpus
    for phase in ['valid', 'test']:
        if getattr(args, 'do_%s' % phase, False):
            d = load_question_eval_data(args, phase)
            result_aggregator = mp.Process(target=async_aggr, args=(args, d.buffer, d.writer_buffer, 'phase'))
            result_aggregator.start()
            aggr_procs.append(result_aggregator)
            eval_dict[phase] = d

    if args.feature_folder is not None:
        logging.info('loading static entity+relation features from %s' % args.feature_folder)
        ro_feat = torch.load(os.path.join(args.feature_folder, 'feat.pt'))
    else:
        ro_feat = None
    procs = []
    training_tasks = args.training_tasks.split('.')
    for rank, gpu_id in enumerate(gpus):
        logging.info("[GPU {}] tasks: {}".format(gpu_id, args.training_tasks))
        local_eval_dict = {}
        for phase in eval_dict:
            q_data = eval_dict[phase]
            nq_per_proc = math.ceil(len(q_data.data) / len(gpus))
            local_eval_dict[phase] = QueryData(q_data.data.subset(rank * nq_per_proc, nq_per_proc), q_data.buffer, q_data.writer_buffer)
        proc = mp.Process(target=train_mp, args=(args, kg_mem, opt_stats, model, local_eval_dict, training_tasks, ro_feat, gpu_id))
        procs.append(proc)
        proc.start()

    write_to_writer(eval_dict, writer)
    for proc in procs + aggr_procs:
        proc.join()

    logging.info("Training finished!!")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main(parse_args())
