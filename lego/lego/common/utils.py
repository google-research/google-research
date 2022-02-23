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
import torch
from tensorboardX import SummaryWriter
import logging
import os
import time
import numpy as np
from collections import defaultdict, OrderedDict
from itertools import chain, combinations
import json

query_name_dict = {('e',('r',)): '1p',
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    ('e', ('r', 'r', 'r', 'r')): '4p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '4i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    ((('e', ('r',)), ('e', ('r',)), ('e', ('r',))), ('r',)): '3ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM',
                    ((('e', ('r', 'r')), ('e', ('r',))), ('r',)): 'pip',
                    (('e', ('r', 'r')), ('e', ('r',)), ('e', ('r',))): 'p3i',
                    (('e', ('r', 'r')), ('e', ('r', 'r'))): '2pi',
                }
name_query_dict = {value: key for key, value in query_name_dict.items()}


def load_lse_checkpoint(args, latent_space_executor):
    init_step = 0
    current_learning_rate = args.learning_rate
    warm_up_steps = args.max_steps // 2
    optimizer_stats = None
    if args.checkpoint_path is not None:
        logging.info('Loading checkpoint %s...' % args.checkpoint_path)
        checkpoint = torch.load(os.path.join(args.checkpoint_path, 'checkpoint'), map_location='cpu')
        try:
            init_step = checkpoint['step']
        except:
            logging.info("step not in checkpoint, init with zero")
            init_step = 0
        all_keys = set(checkpoint['model_state_dict'].keys())
        for key in all_keys:
            if key not in latent_space_executor.state_dict():
                logging.info("{} does not exist in the state dict of model, remove it from the checkpoint".format(key))
                del checkpoint['model_state_dict'][key]
        latent_space_executor.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            try:
                current_learning_rate = checkpoint['current_learning_rate']
                warm_up_steps = checkpoint['warm_up_steps']
                optimizer_stats = checkpoint['optimizer_state_dict']
            except:
                pass
    else:
        logging.info('Randomly Initializing %s Model...' % args.geo)
    opt_stats = {
        'init_step': init_step,
        'warm_up_steps': warm_up_steps,
        'current_learning_rate': current_learning_rate,
        'optimizer_stats': optimizer_stats
    }
    return opt_stats


def load_pruner_checkpoint(args, relation_pruner, branch_pruner):
    assert args.pruner_save_path is not None
    logging.info('Loading checkpoint %s...' % args.pruner_save_path)
    relation_checkpoint = torch.load(os.path.join(args.pruner_save_path, 'relation_checkpoint'), map_location='cpu')
    branch_checkpoint = torch.load(os.path.join(args.pruner_save_path, 'branch_checkpoint'), map_location='cpu')
    relation_pruner.load_state_dict(relation_checkpoint['model_state_dict'])
    branch_pruner.load_state_dict(branch_checkpoint['model_state_dict'])


def group_queries(queries, query_structures, use_cuda):
    batch_queries_dict = defaultdict(list)
    batch_idxs_dict = defaultdict(list)
    for i, query in enumerate(queries):
        batch_queries_dict[query_structures[i]].append(query)
        batch_idxs_dict[query_structures[i]].append(i)
    for query_structure in batch_queries_dict:
        if use_cuda:
            batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
        else:
            batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])

    return batch_queries_dict, batch_idxs_dict

def save_model(model, optimizer, save_variable_list, args, save_path_and_name):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        save_path_and_name
    )

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        try:
            logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        except:
            pass

def log_and_write_metrics(mode, step, metrics, writer):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        try:
            logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
            writer.add_scalar(metric, metrics[metric], step)
        except:
            pass

def get_optimizer(model, lr):
    return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )

def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return

def get_edge_index(s):
    s = list(range(s))
    powerset = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
    edge_index = []
    for idx, p in enumerate(powerset):
        for j in p:
            edge_index.append([j, idx])
    return edge_index

def get_powerset(s):
    s = list(range(s))
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def construct_graph(base_path, indexified_files, return_ent_rel_mat=False):
    adj = []
    ent_in, ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    for indexified_p in indexified_files:
        with open(os.path.join(base_path, indexified_p)) as f:
            for i, line in enumerate(f):
                if len(line) == 0:
                    continue
                e1, rel, e2 = line.split('\t')
                e1 = int(e1.strip())
                e2 = int(e2.strip())
                rel = int(rel.strip())
                ent_out[e1][rel].add(e2)
                ent_in[e2][rel].add(e1)
                if return_ent_rel_mat:
                    adj.append([e1, rel])

    return ent_in, ent_out, adj

def list2tuple(l):
    return tuple(list2tuple(x) if type(x)==list else x for x in l)

def tuple2list(t):
    return list(tuple2list(x) if type(x)==tuple else x for x in t)

def tuple2filterlist(t):
    return list(tuple2filterlist(x) if type(x)==tuple else -1 if x == 'u' else -2 if x == 'n' else x for x in t)

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
            pass
    return tmp_structure, idx, ent_id, num_ent

def query2structure(query):
    assert type(query[-1]) == list
    all_relation_flag = True
    for ele in query[-1]:
        if (type(ele) != int) or (ele == -1):
            all_relation_flag = False
            break
    if all_relation_flag:
        if type(query[0]) == int:
            query[0] = 'e'
        else:
            query2structure(query[0])
        for i in range(len(query[-1])):
            query[-1][i] = 'r'
    else:
        for i in range(len(query)):
            query2structure(query[i])

flatten=lambda l: sum(map(flatten, l),[]) if isinstance(l,tuple) else [l]
flatten_list=lambda l: sum(map(flatten_list, l),[]) if isinstance(l,list) else [l]

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

def set_train_mode(args):
    train_dataset_mode, sync_steps, sparse_embeddings, async_optim, merge_mode = eval_tuple(args.train_online_mode)
    update_mode, optimizer_name, optimizer_device, squeeze_flag, queue_size = eval_tuple(args.optim_mode)
    assert train_dataset_mode in ['single'], "mix has been deprecated"
    args.sparse_embeddings = sparse_embeddings
    args.sparse_device = optimizer_device

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    elif args.do_search:
        log_file = os.path.join(args.save_path, 'search.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    if args.do_train:
        writer = SummaryWriter(args.save_path)
    elif args.do_search:
        writer = None
    else:  # if not training, then create tensorboard files in some tmp location
        test_name = args.eval_path.split('/')[-1]
        writer = SummaryWriter(os.path.join(args.save_path, test_name))

    return writer

def is_arithmetic_seq(seq, k):
    assert len(list(seq)) >= 2
    for i in range(1, len(list(seq))):
        if seq[i] - seq[i-1] != k:
            return False
    return True
