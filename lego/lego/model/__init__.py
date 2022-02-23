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
from .executor import BoxExecutor, RotateExecutor
import logging
from lego.common.utils import eval_tuple, query_name_dict
from smore.common.embedding.embed_optimizer import get_optim_class
import numpy as np


def build_lse_model(args, nentity, nrelation, query_name_dict):
    if args.geo == 'box':
        model = BoxExecutor(nentity=nentity,
                             nrelation=nrelation,
                             hidden_dim=args.hidden_dim,
                             gamma=args.gamma,
                             use_cuda = args.cuda,
                             box_mode=eval_tuple(args.box_mode),
                             batch_size = args.batch_size,
                             test_batch_size=args.test_batch_size,
                             sparse_embeddings=args.sparse_embeddings,
                             sparse_device=args.sparse_device,
                             query_name_dict = query_name_dict,
                             optim_mode = args.optim_mode)
    elif args.geo == 'rotate':
        model = RotateExecutor(nentity=nentity,
                             nrelation=nrelation,
                             hidden_dim=args.hidden_dim,
                             gamma=args.gamma,
                             use_cuda = args.cuda,
                             rotate_mode=eval_tuple(args.rotate_mode),
                             batch_size = args.batch_size,
                             test_batch_size=args.test_batch_size,
                             sparse_embeddings=args.sparse_embeddings,
                             sparse_device=args.sparse_device,
                             query_name_dict = query_name_dict,
                             optim_mode = args.optim_mode)
    else:
        raise ValueError('unknown geo %s' % args.geo)
    return model

def get_lse_model(args):
    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    args.nentity = nentity
    args.nrelation = nrelation
    model = build_lse_model(args, nentity, nrelation, query_name_dict)
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
    elif args.geo == 'rotate':
        logging.info('rotate mode = %s' % args.rotate_mode)
    return model
